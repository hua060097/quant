# -*- coding: utf-8 -*-
"""WQ101 / GTJA191 精选微观力学与双正交残差因子（custom_wq_resid 家族，10个因子）。

设计背景
--------
当前 84 列主特征矩阵中，Class 1 主力因子高度集中于 TURNOVER/AMOUNT 的水平与波动，
缺乏：
  1. K线日内实体力学与多日线性衰减加权（WQ101 #54/#101/#12/#55、GTJA191 #19/#47）；
  2. 剥离市场与申万二级行业共同波动后的“纯个股特质残差 Alpha 与顺逆风捕获率”。

核心纪律
--------
- 100% 无量纲（消除高价股/低价股绝对价格量纲污染，且对前复权因子跨期常数完全免疫）；
- 100% 因果回溯（shift/rolling 均仅使用 <= T 日数据，零前视泄漏）；
- 同一 frame 对象经 weakref 缓存一次性向量化计算，10 个因子共享中间矩阵，耗时 ~2 秒。

因子清单
--------
【A组：WQ101 / GTJA191 精选量价力学与线性衰减（6个）】
  ① `wq054_body_top_decay3`   WQ101#54/#101 改良：K线实体控盘度 (C-O)/(H-L)*(C-L)/(H-L) 的 3日线性衰减加权
  ② `wq012_vol_ret_asym5`     WQ101#12 改良：5日量价涨跌非对称配合度（放量上涨与缩量回调共振强度）
  ③ `wq055_stoch_turn_sync10` WQ101#55 改良：10日通道位置 (C-L10)/(H10-L10) 与换手率分位的去中心化协同度
  ④ `gtja019_trend_eff5`      GTJA191#19/#53 改良：5日路径趋势效率 ret5 / sum(|ret1|, 5) × 10日阳线胜率
  ⑤ `gtja047_high_clv_anchor` GTJA191#47 改良：距20日高点回撤锚定 × 3日日内收盘重心 (CLV) 强弱
  ⑥ `wq083_coil_compress`     WQ101#83 改良：突破前波动与换手双压缩比（3日振幅/20日振幅 × 3日换手/20日换手，反向）

【B组：市场 + 申万二级行业双正交残差 Alpha（4个）】
  ⑦ `res_sec_alpha5_ir`       行业残差5日信息比：(个股日收益 - 行业中位收益) 的 5日均值 / 10日残差波动率
  ⑧ `res_sec_up_down_cap10`   行业顺逆风非对称捕获10日：逆风抗跌1.5倍加权的板块内超额稳定性
  ⑨ `res_sec_clv_lead5`       行业内日内收盘重心领先度5日：个股 CLV 减去同日同行业中位 CLV 的 5日均值
  ⑩ `res_sec_upvol_ratio10`   行业残差10日上下行半方差比：残差正收益平方和 / 残差总平方和（右尾爆发偏度）
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


def _roll_mean(s: pd.Series, codes: pd.Series, w: int, mp: int, idx) -> pd.Series:
    return (s.groupby(codes, sort=False)
            .rolling(w, min_periods=mp).mean()
            .reset_index(level=0, drop=True)
            .reindex(idx))


def _roll_sum(s: pd.Series, codes: pd.Series, w: int, mp: int, idx) -> pd.Series:
    return (s.groupby(codes, sort=False)
            .rolling(w, min_periods=mp).sum()
            .reset_index(level=0, drop=True)
            .reindex(idx))


def _roll_std(s: pd.Series, codes: pd.Series, w: int, mp: int, idx) -> pd.Series:
    return (s.groupby(codes, sort=False)
            .rolling(w, min_periods=mp).std()
            .reset_index(level=0, drop=True)
            .reindex(idx))


def _decay_linear3(s: pd.Series, codes: pd.Series) -> pd.Series:
    """3日线性衰减加权 (3*x_t + 2*x_{t-1} + 1*x_{t-2}) / 6.0。"""
    g = s.groupby(codes, sort=False)
    s1 = g.shift(1)
    s2 = g.shift(2)
    return (3.0 * s + 2.0 * s1 + 1.0 * s2) / 6.0


def _ensure_computed(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    global _CACHE_REF, _CACHE_COLS
    if _CACHE_REF is not None:
        try:
            if _CACHE_REF() is frame and _CACHE_COLS:
                return _CACHE_COLS
        except Exception:
            pass

    idx = frame.index
    codes = frame['stock_code'].astype(str)
    dates = pd.to_datetime(frame['date'])
    sec = codes.map(sw2_mapping()).fillna('__NA__')

    c = pd.to_numeric(frame['CLOSE'], errors='coerce')
    o = pd.to_numeric(frame['OPEN'], errors='coerce')
    h = pd.to_numeric(frame['HIGH'], errors='coerce')
    l = pd.to_numeric(frame['LOW'], errors='coerce')
    v = pd.to_numeric(frame['VOLUME'], errors='coerce')
    t = pd.to_numeric(frame['TURNOVER'], errors='coerce')

    g_code = c.groupby(codes, sort=False)
    c_prev1 = g_code.shift(1)
    c_prev5 = g_code.shift(5)
    ret1 = (c / c_prev1.where(c_prev1 > 0) - 1.0).clip(-0.35, 0.35)
    ret5 = (c / c_prev5.where(c_prev5 > 0) - 1.0).clip(-0.80, 1.50)

    # 日内价格区间与 K 线重心 (CLV ∈ [-1, 1])
    rng = (h - l)
    # 2026-10-21 修复：一字板/无振幅日（H==L）rng=0，旧版 rng_hl=NaN 会让
    # clv/pos_in_bar/body_ratio 连续 3 日 NaN（_decay_linear3 传播）。
    # 按 ret1 方向回填并保持无量纲：涨→clv=+1/pos=1，跌→-1/0，平→0/0.5；
    # body_ratio 无实体=0。amp 用原始 rng（0 振幅合法，不再置 NaN）。
    one_price = ~(rng > 1e-6)
    _up = (ret1 > 0).to_numpy()
    _dn = (ret1 < 0).to_numpy()
    _rng = rng.where(~one_price)
    _clv_raw = ((2.0 * c - h - l) / _rng).clip(-1.0, 1.0)
    _pos_raw = ((c - l) / _rng).clip(0.0, 1.0)
    _body_raw = ((c - o) / _rng).clip(-1.0, 1.0)
    clv = pd.Series(np.where(one_price, np.select([_up, _dn], [1.0, -1.0], 0.0),
                             _clv_raw), index=idx)
    pos_in_bar = pd.Series(np.where(one_price,
                                    np.select([_up, _dn], [1.0, 0.0], 0.5),
                                    _pos_raw), index=idx)
    body_ratio = pd.Series(np.where(one_price,
                                   np.select([_up, _dn], [1.0, -1.0], 0.0),
                                   _body_raw), index=idx)

    # ① wq054_body_top_decay3: 实体阳线力度 × 收盘高位数 的 3日线性衰减加权
    raw_054 = body_ratio * pos_in_bar
    f_wq054 = _decay_linear3(raw_054, codes)

    # ② wq012_vol_ret_asym5: 5日量价非对称配合度（放量涨 vs 缩量跌）
    v_ma5 = _roll_mean(v, codes, 5, 3, idx)
    rel_vol = np.log((v / v_ma5.where(v_ma5 > 0)).clip(0.1, 10.0))
    # 当 ret1 > 0 且放量(rel_vol > 0) 为正；当 ret1 < 0 且缩量(rel_vol < 0) 乘积亦为正！
    asym_daily = ret1 * rel_vol
    ret_std10 = _roll_std(ret1, codes, 10, 5, idx).clip(lower=0.005)
    f_wq012 = _roll_mean(asym_daily, codes, 5, 3, idx) / ret_std10

    # ③ wq055_stoch_turn_sync10: 10日通道位置与换手率相对强度的协同度
    h10 = (h.groupby(codes, sort=False).rolling(10, min_periods=5).max()
           .reset_index(level=0, drop=True).reindex(idx))
    l10 = (l.groupby(codes, sort=False).rolling(10, min_periods=5).min()
           .reset_index(level=0, drop=True).reindex(idx))
    # 2026-10-21：避免"10日低位+缩量"(-0.5×-0.5=+0.25) 与
    # "10日高位+放量"(+0.5×+0.5=+0.25) 同号混淆。改为：
    # 通道位置（居中 [-0.5,0.5]）× 换手强度（严格正 [0.2,5]），
    # 方向由通道位置决定、强度由换手放大；低分位×任何量能均为负。
    stoch10 = ((c - l10) / (h10 - l10).where(lambda x: x > 1e-6)).clip(0.0, 1.0) - 0.5
    t_ma20 = _roll_mean(t, codes, 20, 10, idx)
    turn_int = (t / t_ma20.where(t_ma20 > 0)).clip(0.2, 5.0)
    f_wq055 = _roll_mean(stoch10 * turn_int, codes, 5, 3, idx)

    # ④ gtja019_trend_eff5: 5日趋势路径效率 × 10日阳线占比
    abs_ret_sum5 = _roll_sum(ret1.abs(), codes, 5, 3, idx).clip(lower=0.005)
    up_rate10 = _roll_mean((ret1 > 0).astype(float), codes, 10, 5, idx)
    f_gtja019 = (ret5 / abs_ret_sum5).clip(-1.0, 1.0) * up_rate10

    # ⑤ gtja047_high_clv_anchor: 距20日最高价逼近度 × 3日CLV均值
    h20 = (h.groupby(codes, sort=False).rolling(20, min_periods=10).max()
           .reset_index(level=0, drop=True).reindex(idx))
    prox20 = (c / h20.where(h20 > 0) - 1.0).clip(-0.50, 0.0)
    clv_3d = _decay_linear3(clv, codes)
    f_gtja047 = (1.0 + prox20) * clv_3d

    # ⑥ wq083_coil_compress: 3日振幅与换手率相对20日均值的双压缩度（负号使蓄势紧凑为正）
    amp = (rng / c.where(c > 0)).clip(0.0, 0.25)
    amp_ratio = _roll_mean(amp, codes, 3, 2, idx) / _roll_mean(amp, codes, 20, 10, idx).where(lambda x: x > 1e-4)
    turn_ratio = _roll_mean(t, codes, 3, 2, idx) / t_ma20.where(t_ma20 > 0)
    f_wq083 = -np.log((amp_ratio.clip(0.2, 5.0) * turn_ratio.clip(0.2, 5.0)))

    # ==================== B组：市场 + 申万二级行业双正交残差因子 ====================
    tmp = pd.DataFrame({
        'date': dates.to_numpy(),
        'sec': sec.to_numpy(),
        'ret1': ret1.to_numpy(dtype=float),
        'clv': clv.to_numpy(dtype=float),
    }, index=idx)
    g_sec = tmp.groupby(['date', 'sec'], sort=False)
    sec_cnt = g_sec['ret1'].transform('count')
    valid_sec = ((tmp['sec'] != '__NA__') & (sec_cnt >= 5)).to_numpy()

    sec_ret_med = g_sec['ret1'].transform('median')
    sec_clv_med = g_sec['clv'].transform('median')

    # 个股日收益剥离申万二级行业中位数（同时消除了当日市场与行业共同Beta）
    resid_ret = pd.Series(np.where(valid_sec, tmp['ret1'] - sec_ret_med, np.nan), index=idx)
    resid_clv = pd.Series(np.where(valid_sec, tmp['clv'] - sec_clv_med, np.nan), index=idx)

    # ⑦ res_sec_alpha5_ir: 5日行业残差均值 / 10日残差波动率
    res_mean5 = _roll_mean(resid_ret, codes, 5, 3, idx)
    res_std10 = _roll_std(resid_ret, codes, 10, 5, idx).clip(lower=0.004)
    f_res_ir5 = (res_mean5 / res_std10).clip(-5.0, 5.0)

    # ⑧ res_sec_up_down_cap10: 行业上涨日超额捕获 + 行业下跌日抗跌超额（10日滚动）
    sec_down_weight = np.where(sec_ret_med < 0, 1.5, 1.0)
    weighted_excess = resid_ret * sec_down_weight
    f_res_cap10 = (_roll_mean(weighted_excess, codes, 10, 5, idx) / res_std10).clip(-5.0, 5.0)

    # ⑨ res_sec_clv_lead5: 个股日内收盘重心相对于同行业中位数的 5日领先度
    f_res_clv5 = _roll_mean(resid_clv, codes, 5, 3, idx)

    # ⑩ res_sec_upvol_ratio10: 10日行业残差正收益半方差占总残差方差比例 [0, 1]
    res_pos_sq = (resid_ret.clip(lower=0.0)) ** 2
    res_tot_sq = (resid_ret ** 2)
    pos_var10 = _roll_sum(res_pos_sq, codes, 10, 5, idx)
    tot_var10 = _roll_sum(res_tot_sq, codes, 10, 5, idx)
    f_res_upvol10 = (pos_var10 / tot_var10.where(tot_var10 > 1e-8)).clip(0.0, 1.0)

    _CACHE_COLS = {
        'wq054_body_top_decay3': f_wq054.to_numpy(dtype=float),
        'wq012_vol_ret_asym5': f_wq012.to_numpy(dtype=float),
        'wq055_stoch_turn_sync10': f_wq055.to_numpy(dtype=float),
        'gtja019_trend_eff5': f_gtja019.to_numpy(dtype=float),
        'gtja047_high_clv_anchor': f_gtja047.to_numpy(dtype=float),
        'wq083_coil_compress': f_wq083.to_numpy(dtype=float),
        'res_sec_alpha5_ir': f_res_ir5.to_numpy(dtype=float),
        'res_sec_up_down_cap10': f_res_cap10.to_numpy(dtype=float),
        'res_sec_clv_lead5': f_res_clv5.to_numpy(dtype=float),
        'res_sec_upvol_ratio10': f_res_upvol10.to_numpy(dtype=float),
    }
    try:
        _CACHE_REF = weakref.ref(frame)
    except TypeError:
        _CACHE_REF = None
    return _CACHE_COLS


@register_py_factor(
    factor_id='wq054_body_top_decay3', axis='trend_position',
    description='WQ101#54/#101改良：K线实体控盘度×收盘高位数 3日线性衰减')
def wq054_body_top_decay3(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['wq054_body_top_decay3']


@register_py_factor(
    factor_id='wq012_vol_ret_asym5', axis='liquidity_size',
    description='WQ101#12改良：5日放量上涨与缩量洗盘非对称配合度')
def wq012_vol_ret_asym5(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['wq012_vol_ret_asym5']


@register_py_factor(
    factor_id='wq055_stoch_turn_sync10', axis='volatility',
    description='WQ101#55改良：10日通道位置×换手强度（方向由位置决定，避免双负得正）')
def wq055_stoch_turn_sync10(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['wq055_stoch_turn_sync10']


@register_py_factor(
    factor_id='gtja019_trend_eff5', axis='trend_position',
    description='GTJA191#19/#53改良：5日路径趋势效率×10日阳线胜率')
def gtja019_trend_eff5(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['gtja019_trend_eff5']


@register_py_factor(
    factor_id='gtja047_high_clv_anchor', axis='trend_position',
    description='GTJA191#47改良：20日高点逼近度×3日CLV日内收盘重心')
def gtja047_high_clv_anchor(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['gtja047_high_clv_anchor']


@register_py_factor(
    factor_id='wq083_coil_compress', axis='volatility',
    description='WQ101#83改良：突破前3日vs20日振幅与换手双压缩度')
def wq083_coil_compress(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['wq083_coil_compress']


@register_py_factor(
    factor_id='res_sec_alpha5_ir', axis='trend_position',
    description='申万二级行业残差5日信息比：剥离大盘与板块Beta后的纯特质动量IR')
def res_sec_alpha5_ir(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['res_sec_alpha5_ir']


@register_py_factor(
    factor_id='res_sec_up_down_cap10', axis='trend_position',
    description='申万二级行业顺逆风捕获10日：逆风抗跌1.5倍加权的板块内超额稳定性')
def res_sec_up_down_cap10(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['res_sec_up_down_cap10']


@register_py_factor(
    factor_id='res_sec_clv_lead5', axis='distribution',
    description='申万二级行业内CLV收盘重心5日领先度：识别板块分歧日日内承接最强的真龙')
def res_sec_clv_lead5(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['res_sec_clv_lead5']


@register_py_factor(
    factor_id='res_sec_upvol_ratio10', axis='distribution',
    description='申万二级行业残差10日上行半方差占比[0,1]：剥离板块后的右尾爆发偏度')
def res_sec_upvol_ratio10(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ensure_computed(frame)['res_sec_upvol_ratio10']