# -*- coding: utf-8 -*-
"""板块内身位特征（custom_sector_rank 家族，4个因子）。

核心逻辑：A股短线做板块轮动时，"买在板块前排领涨龙头"与"买在同板块后排
跟风股"次日溢价与抗跌性天差地别；全市场横截面排名看不出个股在所属申万
二级行业内的地位。全部严格只用 ≤ T 日数据（shift/rolling 全为回溯窗），
无前视。

四个因子：
  ① `is_ret3_rank`      板块内3日涨幅身位排名 [0,1]（1.0=板块绝对领涨龙头）
  ② `is_leader_gap5`    距板块龙头5日涨幅差距（行业std归一化，3日均；
                        越接近0越是前排真龙，负值越大越掉队）
  ③ `is_amt_surge_rank` 板块内放量倍数排名（3日均）[0,1]（资金抢筹核心度）
  ④ `is_turn_rank`      板块内换手率人气排名（5日均）[0,1]（活跃度地位）

依赖：frame 已按 (stock_code, date) 排序（custom_registry 约定；rolling 按
行序回溯，乱序输入会得到错误窗口）。行业 <5 只成分股的板块输出 NaN。
实现说明：同一 frame 对象上一次性向量化计算并缓存（weakref 校验对象身份，
防 id 复用读到旧值）；rolling 用 groupby.rolling 向量化（非 per-group lambda）。
"""
from __future__ import annotations

import os
import sys
import weakref

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.dirname(_HERE)
if _MERGE not in sys.path:
    sys.path.insert(0, _MERGE)

from custom_registry import register_py_factor  # noqa: E402
from field_catalog import sw2_mapping           # noqa: E402

_CACHE_REF: weakref.ref | None = None
_CACHE_COLS: dict[str, np.ndarray] = {}


def _ensure_computed(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """在同一 frame 对象上一次性向量化算出 4 个板块身位特征并缓存。
    weakref 校验身份：旧 frame 被 GC 后引用失效即重算，防 id 复用串值。"""
    global _CACHE_REF, _CACHE_COLS
    if _CACHE_REF is not None:
        try:
            if _CACHE_REF() is frame and _CACHE_COLS:
                return _CACHE_COLS
        except Exception:
            pass

    codes = frame['stock_code'].astype(str)
    dates = pd.to_datetime(frame['date'])
    sec = codes.map(sw2_mapping()).fillna('__NA__')

    close = pd.to_numeric(frame['CLOSE'], errors='coerce')
    amt = pd.to_numeric(frame['AMOUNT'], errors='coerce')
    turn = pd.to_numeric(frame['TURNOVER'], errors='coerce')

    # 个股3日/5日累计收益、成交额相对20日均量放大倍数（全部回溯窗）
    g_code = close.groupby(codes, sort=False)
    ret3 = close / g_code.shift(3) - 1.0
    ret5 = close / g_code.shift(5) - 1.0
    amt_ma20 = (amt.groupby(codes, sort=False)
                .rolling(20, min_periods=10).mean()
                .reset_index(level=0, drop=True))
    amt_surge = amt / amt_ma20.where(amt_ma20 > 0)

    tmp = pd.DataFrame({
        'date': dates.to_numpy(),
        'sec': sec.to_numpy(),
        'ret3': ret3.to_numpy(dtype=float),
        'ret5': ret5.to_numpy(dtype=float),
        'amt_surge': amt_surge.to_numpy(dtype=float),
        'turn': turn.to_numpy(dtype=float),
    }, index=frame.index)

    g_sec = tmp.groupby(['date', 'sec'], sort=False)
    sec_size = g_sec['ret3'].transform('count')
    valid_sec = ((tmp['sec'] != '__NA__') & (sec_size >= 5)).to_numpy()

    # ① 3日涨幅行业内百分位排名 [0,1]
    r3_rank = g_sec['ret3'].rank(pct=True)

    # ② 距板块龙头差距：行业内 max 归一（防单票异常用 clip 界 [-5,1]），3日均
    sec_top = g_sec['ret5'].transform('max')
    sec_std = g_sec['ret5'].transform('std').clip(lower=0.01)
    leader_gap = ((tmp['ret5'] - sec_top) / sec_std).clip(-5.0, 1.0)
    # 注意：groupby.rolling 返回组序（股票主序）行序，必须 reindex 回原始行序，
    # 否则 to_numpy 错位（生产帧 stock-major 恰好掩盖此问题，date-major 帧会串值）。
    leader_gap_3d = (leader_gap.groupby(codes, sort=False)
                     .rolling(3, min_periods=2).mean()
                     .reset_index(level=0, drop=True)
                     .reindex(frame.index))

    # ③ 放量倍数行业内排名 [0,1]，3日均
    amt_rk = g_sec['amt_surge'].rank(pct=True)
    amt_rk_3d = (amt_rk.groupby(codes, sort=False)
                 .rolling(3, min_periods=2).mean()
                 .reset_index(level=0, drop=True)
                 .reindex(frame.index))

    # ④ 换手率行业内排名 [0,1]，5日均
    turn_rk = g_sec['turn'].rank(pct=True)
    turn_rk_5d = (turn_rk.groupby(codes, sort=False)
                  .rolling(5, min_periods=3).mean()
                  .reset_index(level=0, drop=True)
                  .reindex(frame.index))

    _CACHE_COLS = {
        'is_ret3_rank': np.where(valid_sec, r3_rank.to_numpy(float), np.nan),
        'is_leader_gap5': np.where(valid_sec, leader_gap_3d.to_numpy(float), np.nan),
        'is_amt_surge_rank': np.where(valid_sec, amt_rk_3d.to_numpy(float), np.nan),
        'is_turn_rank': np.where(valid_sec, turn_rk_5d.to_numpy(float), np.nan),
    }
    try:
        _CACHE_REF = weakref.ref(frame)
    except TypeError:
        _CACHE_REF = None
    return _CACHE_COLS


@register_py_factor(
    factor_id='is_ret3_rank', axis='trend_position',
    description='板块内3日涨幅身位排名[0,1]：区分板块前排领涨龙头与后排跟风股')
def is_ret3_rank(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['is_ret3_rank']


@register_py_factor(
    factor_id='is_leader_gap5', axis='trend_position',
    description='距板块龙头5日涨幅差距(行业std归一化, 3日均)：越接近0越是前排真龙')
def is_leader_gap5(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['is_leader_gap5']


@register_py_factor(
    factor_id='is_amt_surge_rank', axis='liquidity_size',
    description='板块内放量倍数排名(3日均)[0,1]：捕捉同板块内资金最集中抢筹的前排核心')
def is_amt_surge_rank(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['is_amt_surge_rank']


@register_py_factor(
    factor_id='is_turn_rank', axis='volatility',
    description='板块内换手率人气排名(5日均)[0,1]：刻画个股在板块内的活跃度地位')
def is_turn_rank(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['is_turn_rank']
