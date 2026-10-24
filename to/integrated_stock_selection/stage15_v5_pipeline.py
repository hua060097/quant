#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage15 V5.1 终极独立训练、CV选参与 2026 OOS 分钟级回测一体管线
===================================================================
本次升级重点：
1. 标签口径（2026-10-24 扩展，`--label_mode {classic,v51,excess}`）：
   - `classic`（默认，原经典 multihorizon_35）：等权 TP=(O+H+L+C)/4，
     core = r3>=7% & (r5>=r3-3% 或 NaN) & dd3>=-5%；无 T+2 硬底线、
     无缓冲正样本（正类率~2%，匹配 scale_pos_weight 30~38）。
   - `v51`（或旧 `--label_v51`）：TP_real 收盘加权 + T+2 硬底线 +
     r5 绝对涨幅豁免，仅用于与上一轮结果对拍。
   - `excess`（2026-10-24）：classic 量尺 + 截面超额双锚点
     `r3_excess = r3 - 当日可交易有效样本 r3 中位数`，
     core = (r3_excess>=6%) & (r3>=3.5%) & r5_ok & (dd3>=-5%)；
     弱市靠 3.5% 绝对底线防死水股，牛市随中位数自动抬高门槛过滤跟风股。
     训练标签/权重用 r3_excess，但 `y_raw` 始终是绝对 r3，测试集同尺评估。
2. 预测与选参（2026-10-19 定版）：
   - 主评估 = 纯 LightGBM（多 seed 在 logit 空间平均）；CatBoost/XGBoost 仅作
     aux/blend 平行对照表输出（cv_summary_<scheme>.csv），不再混入主预测；
   - CHECKPOINTS 加密至 12 档（150~800），提升平台平滑分辨率；HP_GRID 增补 H23（23叶）。
3. 正样本权重口径（2026-10-24，`--weight_mode` / `--spw_mode`）：
   - `tail`（默认）：核心正样本按绝对超额涨幅给 1.0~1.5x，历史结果逐值复现；
   - `snr`：核心正样本按波动率调整爆发信噪比
     `snr = r3 / (max(vol20, 1.5%) * sqrt(3))` 给 0.7~1.5x
     （0.8σ→0.7x，2.5σ→1.5x），其中 `vol20` 为 T 日盘后可知的过去 20 个
     交易日日收益标准差（PIT）。负样本恒 1.0x，绝不加权；缓冲正样本仍 0.5x。
   - `--spw_mode fixed`（默认）沿用 HP_GRID 的 scale_pos_weight；
     `match_tail` 按当折正样本总质量把 spw 匹配到 tail；
     `adaptive` 按当折 N+/N- 与正样本均权相对基准动态校准（clip [5,50]）。
4. 完整支持 3 种模式：
   - `--mode cv`       : 2021-06 ~ 2025-12 全量 11 个季度折滚动 CV 选参（双核 + 3 Seeds + 三邻域平滑）；
   - `--mode oos_2026` : 用 <=2025-12 数据训练 8-Seed 终极模型，在 2026 年纯 OOS 上跑日频 & 5分钟级状态机回测！

用法：
  # 1. 跑 11 折滚动 CV 选参 (对比 84列 vs 91列)
  python stage15_v5_pipeline.py --mode cv --features_parquet results/merge_v2/lgb_ready/lgb_features.parquet

  # 2. 跑 2026 年纯样本外 OOS + 5分钟级状态机真实回测
  python stage15_v5_pipeline.py --mode oos_2026 --features_parquet results/merge_v2/lgb_ready/lgb_features.parquet --pid H15 --cp 350 --with_minute_backtest

  # 3. 波动率调整正样本权重 A/B（务必与 tail 用不同 --out_dir）
  #    match_tail：把 snr 的当折正样本总质量匹配到 tail，隔离正样本内部重分配
  python stage15_v5_pipeline.py --mode cv --features_parquet results/merge_v2/lgb_ready/lgb_features.parquet --weight_mode snr --spw_mode match_tail --out_dir results/stage15_v5_cv_snr

  # 4. 截面超额标签 A/B（训练标签用 r3_excess，评估仍用绝对 r3）
  python stage15_v5_pipeline.py --mode cv --features_parquet results/merge_v2/lgb_ready/lgb_features.parquet --label_mode excess --out_dir results/stage15_v5_cv_excess
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
RESULTS_ROOT = _HERE / "results"

# ==================== 核心时间与策略常量 ====================
DATA_START = "2021-06-01"          # 完整启用 2021-06 起的全部预热后因子数据
CV_END = "2025-12-31"              # CV 严格锁死在 merge 的 data_cut=2025-12-31 内
TRAIN_MONTHS = 21                  # 默认 21 个月滚动训练窗 (~440 交易日)
TEST_MONTHS = 3                    # 默认 3 个月季度测试折 (~63 交易日)
EMBARGO_DAYS = 5                   # 5 个交易日物理隔离带，防止 T+1..T+5 标签跨折泄漏
NEW_STOCK_MIN_DAYS = 120           # 新股上市未满 120 个交易日过滤
MIN_DAILY_AMOUNT = 10_000_000.0    # T 日最低成交额 1000 万元

# 标签参数（V5.1 升级版）
R3_CORE_MIN = 0.07                 # 核心正样本 3日加权涨幅 >= 7.0%
T2_FLOOR_MIN = 0.01                # 可卖首日(T+2)硬底线：TP_real(T+2) 涨幅 >= +1.0% 且 Close(T+2) > Buy
DD3_CORE_MIN = -0.05               # 核心正样本 T+2/T+3 最大回撤 > -5.0%
R5_TOLERANCE = 0.03                # R5 相比 R3 允许正常休整回吐 3.0% (或 R5 绝对涨幅 >= 10%)
R3_BUF_MIN = 0.045                 # 缓冲稳健正样本 4.5% <= R3 < 7.0%（托底 Rank 2~3 单调性）
DD3_BUF_MIN = -0.035               # 缓冲正样本最大回撤 > -3.5%
BUF_POS_WEIGHT = 0.50              # 缓冲正样本赋予 0.5 倍半权
TAIL_WEIGHT_ALPHA = 0.50           # 核心正样本尾部加权系数 (1.0 ~ 1.5x，从 R0=7% 起算)

# 波动率调整正样本权重（2026-10-24，--weight_mode snr 时启用）：
# 目标 = 低波收敛后突然高信噪比爆发（真突破）重奖，高波常态噪音碰巧过 7% 降权。
# snr = r3 / (max(vol20, VOL20_FLOOR) * sqrt(3))，vol20 为 T 日盘后可知的
# 过去 20 个交易日日收益标准差（PIT，不含未来信息）。
VOL20_FLOOR = 0.015                # 日波动率下限 1.5%（防低波票 SNR 爆炸）
VOL20_RET_CLIP = 0.30              # 估计 vol20 时单日收益截断 ±30%（防复牌/除权异常值）
SNR_ANCHOR_LO = 0.80               # snr<=0.8σ → SNR_W_LO（高波噪音伪正样本）
SNR_ANCHOR_HI = 2.50               # snr>=2.5σ → SNR_W_HI（低波真突破；避免1.5σ顶格拥挤）
SNR_W_LO = 0.70                    # 核心正样本最低权重
SNR_W_HI = 1.50                    # 核心正样本最高权重

# 有效 scale_pos_weight 解析（2026-10-24，--spw_mode）：
# fixed=原样使用 HP_GRID；match_tail=把当前权重的当折正样本总质量匹配到 tail；
# adaptive=按当折 N+/N- 与正样本平均权重相对基准动态校准（clip 到 [5,50]）。
SPW_ADAPT_R0 = 0.032               # 基准正负样本比 N+/N-（经典标签 ~3.1% 正类率）
SPW_ADAPT_W0 = 1.15                # 基准正样本平均权重（tail 口径实测 ~1.14）
SPW_ADAPT_MIN = 5.0                # 自适应 spw 下限
SPW_ADAPT_MAX = 50.0               # 自适应 spw 上限

# 截面超额标签（2026-10-24，--label_mode excess）：
# r3_excess = r3 - 当日可交易有效样本 r3 中位数；
# core = (r3_excess >= 6.0%) & (r3 >= 3.5%) & r5_ok & (dd3 >= -5%)。
# 3.5% 绝对底线防止弱市把下跌/死水股标成正样本；牛市随中位数自动抬高
# 绝对门槛（M_t+6%），过滤靠 Beta 混过 7% 的跟风股。
EXCESS_DELTA_MIN = 0.060           # 跑赢当日市场中位数至少 +6.0%
EXCESS_R3_FLOOR = 0.035            # 绝对涨幅底线 +3.5%

DRIFT_NORM_COLS = ("LOG_MV", "AMIHUD20")


def resolve_drift_cols(spec: str | None = None) -> tuple[str, ...]:
    raw = str(spec).strip() if spec is not None else ""
    if not raw:
        return DRIFT_NORM_COLS
    cols = tuple(dict.fromkeys(
        c.strip() for c in raw.split(",") if c.strip()))
    if not cols:
        raise SystemExit("[特征] --drift_cols 不能为空")
    return cols
# 槽位净值模拟参数（2026-10-20 P0-2 定版）：3 槽轮动、每日投入 1/SLOT 资金、
# 双边成本 0.16%；mdd/净值一律按此口径，禁止对 3 日持有收益做 100% 满仓 cumprod。
COST_PER_TRADE = 0.0016
SLOT_CAPITAL = 3.0
# 每超参容量专属加密 checkpoints（2026-10-19 定版，Owner 批准）：
# lightgbm 各 cp predict 为纯内存操作，加密近零成本；浅树低轮区加密、
# 深树延伸至 1000 轮防截断。
CHECKPOINTS_SHALLOW = [80, 110, 140, 170, 200, 230, 260, 290, 320, 360, 400, 450, 500, 550]
CHECKPOINTS_MEDIUM = [120, 160, 200, 240, 280, 320, 360, 400, 450, 500, 550, 600, 650, 700]
CHECKPOINTS_DEEP = [200, 280, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]
CHECKPOINTS = sorted(set(CHECKPOINTS_SHALLOW + CHECKPOINTS_MEDIUM + CHECKPOINTS_DEEP))
# CatBoost 独立对称树网格（2026-10-19 定版）：深度 4/5/6 + 强正则深度6；
# Bernoulli 子采样 0.80；单次训至 1000 轮，RawFormulaVal + ntree_end 切片全套 cp。
CB_CHECKPOINTS = [150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
CB_HP_GRID = [
    {"pid": "CB_D4",   "depth": 4, "l2_leaf_reg": 10.0},
    {"pid": "CB_D5",   "depth": 5, "l2_leaf_reg": 15.0},
    {"pid": "CB_D6",   "depth": 6, "l2_leaf_reg": 25.0},
    {"pid": "CB_D6_R", "depth": 6, "l2_leaf_reg": 40.0},
]
_CB_SCALE_POS_WEIGHT = 20.0

HP_GRID = [
    # 2026-10-19 diag_labels 定版：L0 经典标签下 spw=20 → 有效正类占比 41.5%
    #（目标区间 35%~42%）；spw=25/30/38 分别为 47.0%/51.6%/57.4%（过高）。
    {"pid": "H15S", "num_leaves": 15,  "max_depth": 5, "min_data_in_leaf": 1000, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 10.0, "reg_lambda": 15.0, "scale_pos_weight": 20.0, "checkpoints": CHECKPOINTS_SHALLOW},
    {"pid": "H23",  "num_leaves": 23,  "max_depth": 5, "min_data_in_leaf": 1000, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 10.0, "reg_lambda": 12.0, "scale_pos_weight": 20.0, "checkpoints": CHECKPOINTS_SHALLOW},
    {"pid": "H12",  "num_leaves": 31,  "max_depth": 6, "min_data_in_leaf": 1000, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 10.0, "reg_lambda": 10.0, "scale_pos_weight": 20.0, "checkpoints": CHECKPOINTS_MEDIUM},
    {"pid": "H15",  "num_leaves": 31,  "max_depth": 6, "min_data_in_leaf": 1000, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 10.0, "reg_lambda": 15.0, "scale_pos_weight": 20.0, "checkpoints": CHECKPOINTS_MEDIUM},
    {"pid": "H63",  "num_leaves": 63,  "max_depth": 7, "min_data_in_leaf": 1200, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 15.0, "reg_lambda": 25.0, "scale_pos_weight": 20.0, "checkpoints": CHECKPOINTS_DEEP},
    {"pid": "H14",  "num_leaves": 127, "max_depth": 7, "min_data_in_leaf": 1800, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 20.0, "reg_lambda": 35.0, "scale_pos_weight": 20.0, "checkpoints": CHECKPOINTS_DEEP},
]

# 2026-10-20 P0：OOS 分钟回测必须与旧主脚本生产配置同构。
# 旧主脚本（main_alpha_binary_v17_1...）通过 minute_strategy_overrides 显式开启
# Top1 状态机与 T+2 10:00 软止损；stage15 早前漏传 → 引擎默认
# enable_top1_state_machine_risk=False / enable_t2_1000_soft_stop=False。
# 下列取值逐项抄自旧主脚本常量（v17_1: 277-306/493/509 行），勿擅改。
OOS_MINUTE_OVERRIDES = {
    "entry_risk_down_threshold_non_top1": -0.03,
    "entry_risk_down_threshold": -0.06,
    "entry_risk_up_threshold": 0.05,
    "entry_risk_control_strategy": "gap_based",
    "entry_risk_ma5_down_threshold": 0.00,
    "entry_risk_ma5_up_threshold": 0.05,
    "delayed_entry_mode": "disabled",
    "stop_loss_threshold": -0.05,
    "profit_bearish_required_count": 3,
    "enable_t2_0935_hard_stop": True,
    "t2_0935_hard_stop_threshold": -0.04,
    "enable_gate_hit_strict_mode": False,
    "gate_hit_entry_gap_up_threshold": 0.02,
    "enable_t2_1000_soft_stop": True,
    "t2_1000_soft_stop_threshold": -0.02,
    "enable_second_layer_time_stop": True,
    "t2_close_breakeven_stop_threshold": 0.00,
    "t3_0935_breakeven_stop_threshold": 0.00,
    "enable_small_profit_failure_exit": True,
    "small_profit_activation_return": 0.04,
    "small_profit_retrace_return": -0.005,
    "small_profit_trailing_drawdown": -0.04,
    "small_profit_bearish_required_count": 2,
    "small_profit_watch_window_bars": 3,
    "enable_profit_threshold_for_bearish_exit": True,
    "profit_bearish_activation_return": 0.05,
    "enable_top1_state_machine_risk": True,
    "enable_dynamic_rank_switch": False,
    "enable_exhaustion_filter": False,
}


# ==================== 基础工具函数 ====================
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("stage15_v5")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


def mad_zscore_by_day(values: np.ndarray, day_codes: np.ndarray, clip_cap: float = 5.0) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    s = pd.Series(v)
    g = s.groupby(day_codes, sort=False)
    med = g.transform("median").to_numpy()
    mad = (s - med).abs().groupby(day_codes, sort=False).transform("median").to_numpy()
    std = g.transform("std").to_numpy()
    scale = np.where(mad > 1e-12, 1.4826 * mad, np.where(std > 1e-12, std, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (v - med) / scale
        az = np.abs(z)
        z_soft = np.sign(z) * (clip_cap + np.log1p(np.maximum(az - clip_cap, 0.0)))
    out = np.where(az <= clip_cap, z, z_soft)
    out[~np.isfinite(v)] = np.nan
    return out.astype(np.float32)


def compute_vol20(pdf: pd.DataFrame) -> np.ndarray:
    """PIT 日波动率：过去 20 个交易日日收益（pct_chg）的标准差。

    2026-10-24（--weight_mode snr）：T 日盘后选股、T+1 开盘买入，故含 T 日
    收盘收益合法可得。pct_chg 为百分数，先 /100 转小数；单日收益截断
    ±VOL20_RET_CLIP，防止复牌/除权等单点异常值污染 20 日波动率估计；
    停牌日（volume/amount<=0）按缺失处理而不是 0 收益（0 会系统性低估波动）。
    每只股票至少 10 个有效收益才输出，否则 NaN。

    返回与 pdf 行对齐的 float64 数组。
    """
    ret = pd.to_numeric(pdf["pct_chg"], errors="coerce").to_numpy(dtype=np.float64) / 100.0
    ret = np.clip(ret, -VOL20_RET_CLIP, VOL20_RET_CLIP)
    if "volume" in pdf.columns and "amount" in pdf.columns:
        halted = ((pdf["volume"].fillna(0) <= 0)
                  | (pdf["amount"].fillna(0) <= 0)).to_numpy(dtype=bool)
        ret[halted] = np.nan
    s = pd.Series(ret)
    vol = s.groupby(pdf["stock_code"].to_numpy(), sort=False).transform(
        lambda x: x.rolling(20, min_periods=10).std(ddof=1))
    return vol.to_numpy(dtype=np.float64)


def find_best_price_cache(start_date: str) -> Path:
    cands = sorted(glob.glob(str(_HERE / "cache" / "stock_features_*.parquet")))
    best_path, best_end, best_span = None, "", -1
    for c in cands:
        m = re.search(r"_(\d{8})_(\d{8})\.parquet$", Path(c).name)
        if m and m.group(1) != m.group(2):
            span = int(m.group(2)) - int(m.group(1))
            if m.group(1) <= start_date.replace("-", "") and (m.group(2) > best_end or span > best_span):
                best_path, best_end, best_span = Path(c), m.group(2), span
    if best_path is None and cands:
        best_path = Path(cands[-1])
    if best_path is None:
        raise FileNotFoundError("cache/ 下未找到 stock_features_*.parquet 行情缓存文件")
    return best_path


def load_sw2_map() -> dict[str, str]:
    pkl_path = _HERE / "config" / "sw2_industry_encoding.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        mp = obj.get("mappings", {}).get("stock_to_industry", {})
        if mp:
            return {str(k): str(v) for k, v in mp.items()}
    json_path = _HERE.parent / "local_data" / "sw_sector2" / "sw_sector.json"
    if json_path.exists():
        return {str(k): str(v) for k, v in json.loads(json_path.read_text(encoding="utf-8")).items()}
    return {}


def forward_window_ok(dates, codes, horizon: int = 5) -> np.ndarray:
    """t+1..t+horizon 是否为连续市场交易日（该股无停牌/缺行）→ True。

    2026-10-20 Owner 决策：停牌日不在面板中，价格 shift(-k) 会跨过停牌日改用
    复牌价，形成"买在/卖在复牌日"的不可执行标签。对停牌日 s 之前的样本
    t∈[s-horizon, s-1]（前向标签窗含 s），一律剔除出训练与评估。
    """
    d = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(dates)).to_numpy(),
        "code": pd.Series(codes).astype(str).to_numpy(),
    })
    mkt = np.sort(pd.unique(d["date"]))
    ii = np.searchsorted(mkt, d["date"].to_numpy()).astype(np.int64)
    d["i"] = ii
    g = d.groupby("code", sort=False)["i"]
    ok = np.ones(len(d), dtype=bool)
    for k in range(1, horizon + 1):
        nxt = g.shift(-k).to_numpy(dtype=float)
        ok &= np.isfinite(nxt) & (nxt == (ii + k))
    return ok


def build_forward_folds(unique_days, train_months=TRAIN_MONTHS, test_months=TEST_MONTHS,
                        embargo_days=EMBARGO_DAYS, cv_end=CV_END, logger=None):
    uniq = np.asarray(unique_days)
    n = len(uniq)
    cv_end_ts = pd.Timestamp(cv_end)
    train_len = int(round(train_months * 21))
    test_len = int(round(test_months * 21))
    folds, start_idx, fid = [], 0, 1
    while True:
        train_end_idx = start_idx + train_len
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = test_start_idx + test_len
        if test_start_idx >= n:
            break
        actual_end_idx = min(test_end_idx, n)
        test_end_ts = pd.Timestamp(uniq[actual_end_idx - 1])
        if pd.Timestamp(uniq[test_start_idx]) > cv_end_ts:
            break
        if test_end_ts > cv_end_ts:
            # 截断到 cv_end_ts 以内
            valid_te = [i for i in range(test_start_idx, actual_end_idx) if pd.Timestamp(uniq[i]) <= cv_end_ts]
            if len(valid_te) < 20:
                break
            actual_end_idx = valid_te[-1] + 1
        folds.append(dict(
            fold_id=fid,
            train_start=str(pd.Timestamp(uniq[start_idx]).date()),
            train_end=str(pd.Timestamp(uniq[train_end_idx - 1]).date()),
            test_start=str(pd.Timestamp(uniq[test_start_idx]).date()),
            test_end=str(pd.Timestamp(uniq[actual_end_idx - 1]).date()),
        ))
        fid += 1
        start_idx += test_len
    if logger:
        logger.info(f"[CV] 构建 {len(folds)} 个滚动折 (train={train_months}M, test={test_months}M, embargo={embargo_days}d):")
        for f_ in folds:
            logger.info(f"  Fold{f_['fold_id']:02d}: Train {f_['train_start']}~{f_['train_end']} | Test {f_['test_start']}~{f_['test_end']}")
    return folds


def build_oos_rolling_folds(unique_days, test_start, test_end, train_months=TRAIN_MONTHS,
                            embargo_days=EMBARGO_DAYS, logger=None):
    """OOS 滚动重训折：按自然季度切测试窗，训练窗 = 前 train_months 个月，5 日隔离。

    2026-10-20 P0：与 CV 完全同构（静态单模型硬扛 9 个月会老化衰减）。
    测试窗 [test_start, test_end] 内按月历季度切分（末段不足一季度按实际截止日）。
    """
    uniq = np.asarray(unique_days)
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    bounds, cur = [], pd.Timestamp(year=ts.year, month=((ts.month - 1) // 3) * 3 + 1,
                                   day=1)
    while cur <= te:
        nxt = cur + pd.DateOffset(months=3)
        s_ = max(cur, ts)
        e_ = min(nxt - pd.Timedelta(days=1), te)
        if s_ <= e_:
            bounds.append((s_, e_))
        cur = nxt
    folds = []
    for k, (s_, e_) in enumerate(bounds):
        idx_s = int(np.searchsorted(uniq, np.datetime64(s_)))
        tr_end_i = idx_s - 1 - int(embargo_days)
        if tr_end_i < 6 * 21:
            continue
        tr_start_i = max(0, tr_end_i - int(round(train_months * 21)) + 1)
        folds.append(dict(
            fold_id=100 + k,
            train_start=str(pd.Timestamp(uniq[tr_start_i]).date()),
            train_end=str(pd.Timestamp(uniq[tr_end_i]).date()),
            test_start=str(s_.date()), test_end=str(e_.date()),
        ))
    if logger:
        logger.info(f"[OOS-ROLL] 构建 {len(folds)} 个季度滚动折 "
                    f"(train={train_months}M, embargo={embargo_days}d):")
        for f_ in folds:
            logger.info(f"  Fold{f_['fold_id']}: Train {f_['train_start']}~{f_['train_end']} "
                        f"| Test {f_['test_start']}~{f_['test_end']}")
    return folds


# ==================== 数据、V5.1标签与权重构建 ====================
def load_dataset_and_labels(args, logger: logging.Logger) -> dict:
    feat_path = Path(args.features_parquet)
    if not feat_path.is_absolute():
        # 相对路径按脚本所在目录解析（防 CWD 不同导致 FileNotFoundError）
        feat_path = _HERE / feat_path
    # 2026-10-21：91 列 with_h3 变体在 84 列 A/B 中全面更差，已废弃；
    # 禁止再用它训练/评估，防止误用历史产物。
    if "with_h3" in feat_path.name:
        raise SystemExit(
            "[DEPRECATED] 91 列 with_h3 变体已废弃（84 列 A/B 全面更差）；"
            "请使用 lgb_features.parquet（84 列）")
    if not feat_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {feat_path}")

    t0 = time.time()
    _explicit_label_mode = getattr(args, "label_mode", None)
    if (_explicit_label_mode not in (None, "", "v51")
            and getattr(args, "label_v51", False)):
        raise SystemExit(
            "--label_v51 与 label_mode 冲突：--label_v51 等价于 --label_mode v51")
    label_mode = str(_explicit_label_mode or
                     ("v51" if getattr(args, "label_v51", False) else "classic"))
    if label_mode not in ("classic", "v51", "excess"):
        raise SystemExit(
            f"[标签] 未知 label_mode={label_mode!r}（可选 classic/v51/excess）")
    logger.info(f"[标签] label_mode={label_mode}")
    logger.info(f"[1/4] 读取特征矩阵: {feat_path}")
    fdf = pd.read_parquet(feat_path)
    fdf["date"] = pd.to_datetime(fdf["date"]).dt.normalize()
    fdf["stock_code"] = fdf["stock_code"].astype(str)
    fdf = fdf[fdf["date"] >= pd.Timestamp(args.data_start)].copy()

    sw_map = load_sw2_map()
    if not sw_map:
        raise SystemExit(
            "sw2 行业映射为空（config/sw2_industry_encoding.pkl 与 "
            "local_data/sw_sector2/sw_sector.json 均不可用），"
            "行业内去重与评估将退化，终止。")
    fdf = fdf.loc[~fdf["stock_code"].str.endswith((".BJ", ".bj"), na=False)].copy()
    fdf = fdf[fdf["stock_code"].isin(set(sw_map.keys()))].copy()

    feature_cols = [c for c in fdf.columns if c not in ("stock_code", "date")]
    logger.info(f"      特征数={len(feature_cols)} 行数={len(fdf):,} 日期={fdf['date'].min().date()}~{fdf['date'].max().date()}")

    price_path = find_best_price_cache(args.data_start)
    logger.info(f"[2/4] 读取行情缓存构造 V5.1 标签与过滤掩码: {price_path.name}")
    want_cols = ["stock_code", "date", "open", "high", "low", "close", "volume", "amount",
                 "pct_chg", "is_st", "is_limit_up", "is_limit_down", "is_zhaban", "is_can_buy", "circ_mv"]
    import pyarrow.parquet as pq
    avail_pcols = [c for c in want_cols if c in set(pq.read_schema(price_path).names)]
    pdf = pd.read_parquet(price_path, columns=avail_pcols)
    pdf["date"] = pd.to_datetime(pdf["date"]).dt.normalize()
    pdf["stock_code"] = pdf["stock_code"].astype(str)
    pdf = pdf.sort_values(["stock_code", "date"]).drop_duplicates(["stock_code", "date"], keep="last").reset_index(drop=True)

    # PIT 波动率（--weight_mode snr 用）：过去 20 个交易日日收益标准差，含 T 日
    # 收盘（T 日盘后可知）；行情缓存起始日早于 data_start，data_start 附近样本
    # 仍有完整 20 日预热。默认 tail 模式下仅随 bundle 透传，不参与权重。
    if "pct_chg" in pdf.columns:
        pdf["vol20"] = compute_vol20(pdf)
    else:
        if str(getattr(args, "weight_mode", "tail")) in (
                "snr", "tail_snr_max", "snr_inv", "highvol_up"):
            raise SystemExit(
                f"[标签] --weight_mode {args.weight_mode} 需要行情缓存含 pct_chg 列，"
                f"当前缓存缺列: {price_path.name}")
        logger.warning("[标签] 行情缓存缺 pct_chg，vol20 全置 NaN（tail 模式不受影响）")
        pdf["vol20"] = np.nan

    g_code = pdf.groupby("stock_code", sort=False)
    pdf["listed_days"] = g_code.cumcount() + 1
    is_susp = (pdf["volume"].fillna(0) <= 0) | (pdf["amount"].fillna(0) <= 0)
    recent_susp = is_susp.astype(float).groupby(pdf["stock_code"], sort=False).rolling(10, min_periods=1).max().reset_index(level=0, drop=True) > 0
    # 2026-10-20 Owner 决策：剔除"未来 5 个交易日内含停牌"的样本（标签前向窗
    # t+1..t+5 必须连续，否则 shift(-k) 会跨停牌改用复牌价，标签不可执行）。
    _fw_ok = forward_window_ok(pdf["date"].to_numpy(), pdf["stock_code"].to_numpy(),
                               horizon=5)
    pdf["forward_ok"] = _fw_ok
    logger.info(f"[标签] 前向停牌窗剔除 {int((~_fw_ok).sum()):,} 行"
                f"（占 {float((~_fw_ok).mean()):.2%}）")

    # 标签口径（2026-10-24 扩展）：classic=经典 multihorizon_35（等权 TP）；
    # v51=旧 --label_v51（收盘加权 TP_real，压低最高价权重改变量尺）；
    # excess=classic 等权 TP + 截面超额门槛（不改变 TP 量尺，r1_mean 可与
    # classic 同尺比较，但训练标签的正类构成不同）。
    if label_mode == "v51":
        tp_real = 0.15 * pdf["open"] + 0.15 * pdf["high"] + 0.20 * pdf["low"] + 0.50 * pdf["close"]
    else:
        tp_real = (pdf["open"] + pdf["high"] + pdf["low"] + pdf["close"]) / 4.0
    buy_p = g_code["open"].shift(-1)
    can_buy_t1 = g_code["is_can_buy"].shift(-1).fillna(1) if "is_can_buy" in pdf.columns else 1

    tp_t2 = tp_real.groupby(pdf["stock_code"], sort=False).shift(-2)
    tp_t3 = tp_real.groupby(pdf["stock_code"], sort=False).shift(-3)
    tp_t4 = tp_real.groupby(pdf["stock_code"], sort=False).shift(-4)
    tp_t5 = tp_real.groupby(pdf["stock_code"], sort=False).shift(-5)
    close_t2 = g_code["close"].shift(-2)

    sell_r3 = 0.70 * tp_t2 + 0.30 * tp_t3
    sell_r5 = 0.70 * tp_t4 + 0.30 * tp_t5

    low_t2 = g_code["low"].shift(-2)
    low_t3 = g_code["low"].shift(-3)
    min_low_23 = np.minimum(low_t2, low_t3)

    valid_buy = (buy_p > 1e-6) & (can_buy_t1 == 1)
    pdf["ret_t2"] = np.where(valid_buy, (tp_t2 - buy_p) / buy_p, np.nan)
    pdf["close_t2_ret"] = np.where(valid_buy, (close_t2 - buy_p) / buy_p, np.nan)
    pdf["r3"] = np.where(valid_buy, (sell_r3 - buy_p) / buy_p, np.nan)
    pdf["r5"] = np.where(valid_buy, (sell_r5 - buy_p) / buy_p, np.nan)
    pdf["dd3"] = np.where(valid_buy, (min_low_23 - buy_p) / buy_p, np.nan)

    tradable = (
        (pdf.get("is_st", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf["listed_days"] >= NEW_STOCK_MIN_DAYS)
        & (~recent_susp)
        & (pdf.get("is_limit_up", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf.get("is_limit_down", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf.get("is_zhaban", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf["amount"].fillna(0) >= MIN_DAILY_AMOUNT)
    )
    pdf["tradable_mask"] = tradable

    merged = fdf.merge(
        pdf[["stock_code", "date", "ret_t2", "close_t2_ret", "r3", "r5", "dd3",
             "vol20", "tradable_mask", "forward_ok"]],
        on=["stock_code", "date"], how="inner"
    )
    merged = merged.sort_values(["date", "stock_code"]).reset_index(drop=True)
    del fdf, pdf

    day_codes, _ = pd.factorize(merged["date"].to_numpy(), sort=True)
    drift_cols = resolve_drift_cols(getattr(args, "drift_cols", None))
    _missing_drift = [c for c in drift_cols if c not in merged.columns]
    if _missing_drift:
        raise SystemExit(
            f"[特征] --drift_cols 含缺失列: {_missing_drift}")
    for col in drift_cols:
        merged[col] = mad_zscore_by_day(
            merged[col].to_numpy(dtype=float), day_codes)
    logger.info(f"[特征] 日内漂移标准化: {','.join(drift_cols)}")

    ret_t2 = merged["ret_t2"].to_numpy(dtype=np.float64)
    close_t2_ret = merged["close_t2_ret"].to_numpy(dtype=np.float64)
    r3 = merged["r3"].to_numpy(dtype=np.float64)
    r5 = merged["r5"].to_numpy(dtype=np.float64)
    dd3 = merged["dd3"].to_numpy(dtype=np.float64)
    vol20 = merged["vol20"].to_numpy(dtype=np.float64)

    # 正样本口径：
    # classic（默认）：core = r3>=7% & (r5>=r3-3% 或 NaN) & dd3>=-5%，
    #   无 T+2 硬底线、无缓冲正样本；
    # v51：附加 r5>=+10% 豁免、T+2 硬底线、可选缓冲正样本；
    # excess（2026-10-24）：classic 量尺 + 截面超额双锚点
    #   r3_excess = r3 - 当日可交易有效样本 r3 中位数；
    #   core = (r3_excess>=6%) & (r3>=3.5%) & r5_ok & (dd3>=-5%)，
    #   无 T+2 硬底线、无缓冲正样本。r3_excess 只用于训练标签与权重，
    #   bundle["y_raw"] 仍保持绝对 r3，供测试集同尺评估。
    y_weight_raw = r3
    r3_excess = None
    r3_med_day = None
    if label_mode == "v51":
        r5_ok = np.isnan(r5) | (r5 >= (r3 - R5_TOLERANCE)) | (r5 >= 0.10)
        t2_ok = (ret_t2 >= T2_FLOOR_MIN) & (close_t2_ret > 0.0)
        core_pos = (r3 >= R3_CORE_MIN) & t2_ok & r5_ok & (dd3 >= DD3_CORE_MIN)
        if args.enable_buffer_pos:
            buf_pos = (~core_pos) & (r3 >= R3_BUF_MIN) & (r3 < R3_CORE_MIN) & t2_ok & r5_ok & (dd3 >= DD3_BUF_MIN)
        else:
            buf_pos = np.zeros_like(core_pos, dtype=bool)
    elif label_mode == "excess":
        _med_mask = (merged["tradable_mask"].to_numpy(dtype=bool)
                     & np.isfinite(r3))
        _r3_for_med = np.where(_med_mask, r3, np.nan)
        r3_med_day = pd.Series(_r3_for_med).groupby(
            day_codes, sort=False).transform("median").to_numpy(dtype=np.float64)
        r3_excess = r3 - r3_med_day
        r5_ok = np.isnan(r5) | (r5 >= (r3 - R5_TOLERANCE))
        core_pos = ((r3_excess >= EXCESS_DELTA_MIN)
                    & (r3 >= EXCESS_R3_FLOOR)
                    & r5_ok & (dd3 >= DD3_CORE_MIN))
        buf_pos = np.zeros_like(core_pos, dtype=bool)
        y_weight_raw = r3_excess
    else:
        r5_ok = np.isnan(r5) | (r5 >= (r3 - R5_TOLERANCE))
        core_pos = (r3 >= R3_CORE_MIN) & r5_ok & (dd3 >= DD3_CORE_MIN)
        buf_pos = np.zeros_like(core_pos, dtype=bool)

    y_bin = np.where(core_pos | buf_pos, 1.0, 0.0).astype(np.float32)
    # 2026-10-21 修复（Gemini/Owner）：把"评估/训练"与"交易"掩码拆开。
    # - trade_mask：仅 T 日盘后可知条件（ST/新股/近停牌/涨跌停/炸板/成交额），
    #   不含 valid_buy（T+1 才可知）、不含 r3/dd3（未来标签）、不含 forward_ok。
    # -> 用于 OOS 预测与分钟回测，恢复末尾无标签交易日。
    # - valid_mask：日频评估需要 r3/dd3 有限（标签可算），
    #   否则该日无法计入 Rank1 收益统计。
    # - train_mask：在 valid_mask 基础上再剔"未来 5 日含停牌"的不可执行标签。
    trade_valid = merged["tradable_mask"].to_numpy(dtype=bool)
    y_valid = np.isfinite(r3) & np.isfinite(dd3) & trade_valid
    train_valid = y_valid & merged["forward_ok"].to_numpy(dtype=bool)
    # 2026-10-21 A/B：old 复现旧版 valid_mask（含 forward_ok，训练/测试同口径）；
    # new 训练用 train_mask、测试用 trade_mask（恢复 2026 末段分钟回测日）。
    _mode = str(getattr(args, "mask_mode", "new"))
    _old_mask = _mode == "old"
    # old: y_valid 训练/测试（复现 A2）；new: train_mask 训练 + trade 测试；
    # fwd: train_mask(剔 forward_ok) 训练 + y_valid 评估（阶段四重搜超参）。
    test_valid = trade_valid if _mode == "new" else y_valid
    train_out = y_valid if _old_mask else train_valid
    # 2026-10-21 修复：掩码拆分时误删了 X/dates/codes/sectors 的构建，补回。
    X = np.ascontiguousarray(merged[feature_cols].to_numpy(dtype=np.float32))
    dates = merged["date"].to_numpy()
    codes = merged["stock_code"].to_numpy()
    sectors = np.array([sw_map.get(c, "__NA__") for c in codes], dtype=object)
    if label_mode == "excess":
        _thr_day = np.maximum(EXCESS_R3_FLOOR, r3_med_day + EXCESS_DELTA_MIN)
        _mf = r3_med_day[np.isfinite(r3_med_day)]
        _tf = _thr_day[np.isfinite(_thr_day)]
        if len(_mf) and len(_tf):
            _mq = np.percentile(_mf, [10, 50, 90])
            _tq = np.percentile(_tf, [10, 50, 90])
            logger.info(
                f"[标签] excess 日度中位数 P10/P50/P90="
                f"{_mq[0]:+.2%}/{_mq[1]:+.2%}/{_mq[2]:+.2%} | "
                f"绝对门槛 P10/P50/P90="
                f"{_tq[0]:.2%}/{_tq[1]:.2%}/{_tq[2]:.2%}")
        _ex = r3_excess[np.isfinite(r3_excess) & y_valid]
        if len(_ex):
            _eq = np.percentile(_ex, [10, 50, 90])
            logger.info(
                f"[标签] r3_excess(有效样本) P10/P50/P90="
                f"{_eq[0]:+.2%}/{_eq[1]:+.2%}/{_eq[2]:+.2%}")
    _core_vol_na = float(np.mean(~np.isfinite(vol20[core_pos]))) if core_pos.any() else 0.0
    logger.info(
        f"[3/4] 数据准备就绪 ({time.time()-t0:.0f}s): label_mode={label_mode} "
        f"交易有效行={int(trade_valid.sum()):,}/{len(merged):,} "
        f"| 评估有效行(标签可算)={int(y_valid.sum()):,} "
        f"| 训练有效行(前向停牌窗剔除)={int(train_valid.sum()):,} "
        f"| 核心正类率={core_pos[y_valid].mean():.2%} 缓冲正类率={buf_pos[y_valid].mean():.2%} "
        f"| vol20 缺失率={float(np.mean(~np.isfinite(vol20))):.2%}"
        f"(核心正样本内 {_core_vol_na:.2%})"
    )
    return {
        "X": X, "y": y_bin, "y_raw": r3, "y_weight_raw": y_weight_raw,
        "label_mode": label_mode,
        "r3_excess": r3_excess, "r3_med_day": r3_med_day,
        "core_pos": core_pos, "buf_pos": buf_pos,
        "vol20": vol20,
        "valid_mask": y_valid, "train_mask": train_out, "trade_mask": trade_valid,
        "test_mask": test_valid,
        "dates": dates, "codes": codes, "sectors": sectors,
        "feature_cols": feature_cols, "price_path": price_path,
        "drift_cols": list(drift_cols),
    }


def compute_sample_weights(core_pos: np.ndarray, buf_pos: np.ndarray,
                           y_raw: np.ndarray,
                           dates: np.ndarray | None = None,
                           weight_mode: str = "tail",
                           vol20: np.ndarray | None = None,
                           weight_raw: np.ndarray | None = None,
                           weight_r0: float | None = None) -> np.ndarray:
    """构造训练样本权重（负样本恒 1.0，绝不加权）。

    weight_mode:
      - "tail"（默认，历史定版）：核心正样本按 weight_raw 超出 weight_r0
        的幅度 g=clip((weight_raw-r0)/(q95-r0),0,1) 给 1.0~1.5x；缓冲 0.5x。
      - "snr"（2026-10-24）：核心正样本按波动率调整爆发信噪比
        snr=weight_raw/(max(vol20,1.5%)*sqrt(3)) 线性给 0.7~1.5x
        （0.8σ→0.7x，2.5σ→1.5x），压低"高波常态噪音碰巧过阈"的伪正
        样本、抬高"低波收敛后高信噪比爆发"的真突破；缓冲正样本仍 0.5x，
        负样本仍 1.0x。vol20 缺失的正样本取中性 1.0x（不奖不罚）。
      - "tail_snr_max"（2026-10-24）：核心正样本逐样本取 max(tail, snr)，
        保留高波赢家的 tail 权重，同时额外抬高低波突破权重。
      - "snr_inv"（2026-10-24）：snr 的反向权重，低信噪比（高波）给 1.5x，
        高信噪比（低波）给 0.7x，用于检验波动方向反转。
      - "highvol_up"（2026-10-24）：只抬高波正样本，低波保持 1.0x 不降权。
      五种模式最终都叠加海啸日正样本总量线性封顶（不改 y、不碰负样本）。

    weight_raw / weight_r0:
      - 默认 weight_raw=y_raw、weight_r0=R3_CORE_MIN(7%)，即历史绝对涨幅口径；
      - label_mode=excess 时由调用方传 weight_raw=r3_excess、
        weight_r0=EXCESS_DELTA_MIN(6%)，使权重也按截面超额幅度计算。
    """
    if weight_mode not in (
            "tail", "snr", "tail_snr_max", "snr_inv", "highvol_up"):
        raise ValueError(
            f"未知 weight_mode={weight_mode!r}"
            "（可选 tail/snr/tail_snr_max/snr_inv/highvol_up）")
    _wr = (np.asarray(y_raw, dtype=np.float64) if weight_raw is None
           else np.asarray(weight_raw, dtype=np.float64))
    if _wr.shape[0] != len(y_raw):
        raise ValueError(
            f"weight_raw 长度 {_wr.shape[0]} 与 y_raw 长度 {len(y_raw)} 不一致")
    _r0 = R3_CORE_MIN if weight_r0 is None else float(weight_r0)
    w = np.ones(len(y_raw), dtype=np.float32)
    if buf_pos.any():
        w[buf_pos] = BUF_POS_WEIGHT
    if core_pos.any():
        if weight_mode == "snr":
            if vol20 is None:
                raise ValueError("weight_mode='snr' 需要提供 vol20 数组")
            _v_all = np.asarray(vol20, dtype=np.float64)
            if _v_all.shape[0] != len(y_raw):
                raise ValueError(
                    f"vol20 长度 {_v_all.shape[0]} 与 y_raw 长度 {len(y_raw)} 不一致")
            _v = _v_all[core_pos]
            if not np.isfinite(_v).any():
                raise ValueError("weight_mode='snr' 但核心正样本 vol20 全缺失")
            _snr = _wr[core_pos] / (np.maximum(_v, VOL20_FLOOR) * np.sqrt(3.0))
            _w = (SNR_W_LO + (SNR_W_HI - SNR_W_LO)
                  * (_snr - SNR_ANCHOR_LO) / (SNR_ANCHOR_HI - SNR_ANCHOR_LO))
            _w = np.clip(_w, SNR_W_LO, SNR_W_HI)
            _w[~np.isfinite(_snr)] = 1.0
            w[core_pos] = _w.astype(np.float32)
        elif weight_mode == "snr_inv":
            if vol20 is None:
                raise ValueError("weight_mode='snr_inv' 需要提供 vol20 数组")
            _v_all = np.asarray(vol20, dtype=np.float64)
            if _v_all.shape[0] != len(y_raw):
                raise ValueError(
                    f"vol20 长度 {_v_all.shape[0]} 与 y_raw 长度 {len(y_raw)} 不一致")
            _v = _v_all[core_pos]
            if not np.isfinite(_v).any():
                raise ValueError("weight_mode='snr_inv' 但核心正样本 vol20 全缺失")
            _snr = _wr[core_pos] / (np.maximum(_v, VOL20_FLOOR) * np.sqrt(3.0))
            _w = (SNR_W_HI - (SNR_W_HI - SNR_W_LO)
                  * (_snr - SNR_ANCHOR_LO) / (SNR_ANCHOR_HI - SNR_ANCHOR_LO))
            _w = np.clip(_w, SNR_W_LO, SNR_W_HI)
            _w[~np.isfinite(_snr)] = 1.0
            w[core_pos] = _w.astype(np.float32)
        elif weight_mode == "highvol_up":
            if vol20 is None:
                raise ValueError("weight_mode='highvol_up' 需要提供 vol20 数组")
            _v_all = np.asarray(vol20, dtype=np.float64)
            if _v_all.shape[0] != len(y_raw):
                raise ValueError(
                    f"vol20 长度 {_v_all.shape[0]} 与 y_raw 长度 {len(y_raw)} 不一致")
            _v = _v_all[core_pos]
            if not np.isfinite(_v).any():
                raise ValueError("weight_mode='highvol_up' 但核心正样本 vol20 全缺失")
            _snr = _wr[core_pos] / (np.maximum(_v, VOL20_FLOOR) * np.sqrt(3.0))
            _w = (SNR_W_HI - (SNR_W_HI - SNR_W_LO)
                  * (_snr - SNR_ANCHOR_LO) / (SNR_ANCHOR_HI - SNR_ANCHOR_LO))
            _w = np.clip(_w, SNR_W_LO, SNR_W_HI)
            _w = np.maximum(_w, 1.0)
            _w[~np.isfinite(_snr)] = 1.0
            w[core_pos] = _w.astype(np.float32)
        elif weight_mode == "tail_snr_max":
            if vol20 is None:
                raise ValueError(
                    "weight_mode='tail_snr_max' 需要提供 vol20 数组")
            _v_all = np.asarray(vol20, dtype=np.float64)
            if _v_all.shape[0] != len(y_raw):
                raise ValueError(
                    f"vol20 长度 {_v_all.shape[0]} 与 y_raw 长度 {len(y_raw)} 不一致")
            _v = _v_all[core_pos]
            if not np.isfinite(_v).any():
                raise ValueError(
                    "weight_mode='tail_snr_max' 但核心正样本 vol20 全缺失")
            _snr = _wr[core_pos] / (np.maximum(_v, VOL20_FLOOR) * np.sqrt(3.0))
            _w_snr = (SNR_W_LO + (SNR_W_HI - SNR_W_LO)
                      * (_snr - SNR_ANCHOR_LO) / (SNR_ANCHOR_HI - SNR_ANCHOR_LO))
            _w_snr = np.clip(_w_snr, SNR_W_LO, SNR_W_HI)
            _w_snr[~np.isfinite(_snr)] = 1.0
            r_pos = _wr[core_pos]
            q95 = float(np.nanpercentile(r_pos, 95)) if len(r_pos) > 10 else 0.20
            q95 = max(q95, _r0 + 0.02)
            g = np.clip((_wr - _r0) / (q95 - _r0), 0.0, 1.0)
            _w_tail = 1.0 + TAIL_WEIGHT_ALPHA * g[core_pos]
            w[core_pos] = np.maximum(_w_snr, _w_tail).astype(np.float32)
        else:
            r_pos = _wr[core_pos]
            q95 = float(np.nanpercentile(r_pos, 95)) if len(r_pos) > 10 else 0.20
            q95 = max(q95, _r0 + 0.02)
            g = np.clip((_wr - _r0) / (q95 - _r0), 0.0, 1.0)
            w[core_pos] = (1.0 + TAIL_WEIGHT_ALPHA * g[core_pos]).astype(np.float32)
    # 极端普涨海啸日软封顶（V5.1）：2024-09-24~30 等单日数千正样本的极值日会在
    # 损失函数里绑架整折正类梯度。纪律：绝不改 y=1 标签（改标签=特征认知错乱），
    # 只对"单日正样本数 > max(300, P95)"的极少数海啸日，把当日**正样本**权重乘
    # sqrt(N_cap/N_pos)；其余 95% 的正常交易日乘数=1.0 零改动。
    if dates is not None:
        _pos = core_pos | buf_pos
        if _pos.any():
            day_codes, _ = pd.factorize(dates, sort=True)
            nd = int(day_codes.max()) + 1 if len(day_codes) else 0
            if nd > 20:
                npos = np.bincount(day_codes[_pos], minlength=nd).astype(float)
                _v = npos[npos > 0]
                if len(_v) > 20:
                    cap = max(300.0, float(np.percentile(_v, 95)))
                    over = npos > cap
                    if over.any():
                        scale = np.ones(nd, dtype=np.float32)
                        # 2026-10-19 定版（Owner 批准）：线性硬封顶——海啸日全市场
                        # 正样本总质量硬封顶于 cap=max(300,P95)；sqrt 软封顶后海啸日
                        # 仍占 19.1% 正类权（2024-09-25 缩放后仍≈1106 个正样本），
                        # 线性封顶将其压至 ~11%。
                        scale[over] = (cap / npos[over]).astype(np.float32)
                        _sc = scale[day_codes]
                        w[_pos] = (w[_pos] * _sc[_pos]).astype(np.float32)
    return w


def resolve_effective_spw(y_train: np.ndarray, w_train: np.ndarray,
                          base_spw: float,
                          tail_w_train: np.ndarray | None = None,
                          spw_mode: str = "fixed") -> float:
    """解析当折训练集实际使用的 scale_pos_weight。

    - fixed（默认）：原样返回 base_spw，历史行为逐值不变。
    - match_tail：`spw_eff = base_spw * sum(tail_w[pos]) / sum(w[pos])`。
      使当前权重口径与 tail 口径在该折上的**正样本总梯度质量严格相等**，
      从而把 tail→snr 的对比隔离为"正样本内部重分配"，不混杂正类总推力
      变化。海啸日封顶对两种口径按同一 cap/npos 缩放，故封顶前后比值一致；
      但 tail_w_train 必须与 w_train 在同一训练子集上、用同一 dates 构造。
    - adaptive：`spw_eff = base_spw * (r0/(N+/N-)) * (w0/mean_w_pos)`，
      其中 r0=SPW_ADAPT_R0、w0=SPW_ADAPT_W0，结果 clip 到
      [SPW_ADAPT_MIN, SPW_ADAPT_MAX]。用于抑制跨折正样本率/平均权重波动
      带来的有效正类梯度占比漂移。

    负样本权重不参与任何缩放；本函数只决定 LightGBM/CatBoost 的正类乘数。
    """
    if spw_mode not in ("fixed", "match_tail", "adaptive"):
        raise ValueError(
            f"未知 spw_mode={spw_mode!r}（可选 fixed/match_tail/adaptive）")
    base = float(base_spw)
    if not np.isfinite(base) or base <= 0:
        raise ValueError(f"base_spw 必须为正有限数，得到 {base_spw!r}")
    if spw_mode == "fixed":
        return base

    y = np.asarray(y_train)
    w = np.asarray(w_train, dtype=np.float64)
    if y.shape[0] != w.shape[0]:
        raise ValueError(
            f"y_train 长度 {y.shape[0]} 与 w_train 长度 {w.shape[0]} 不一致")
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"spw_mode={spw_mode} 需要训练集同时含正负样本"
            f"（n_pos={n_pos}, n_neg={n_neg}）")
    pos_mass = float(w[pos].sum())
    if not np.isfinite(pos_mass) or pos_mass <= 0:
        raise ValueError(f"正样本权重和非法: {pos_mass!r}")

    if spw_mode == "match_tail":
        if tail_w_train is None:
            raise ValueError("spw_mode='match_tail' 需要提供 tail_w_train")
        tw = np.asarray(tail_w_train, dtype=np.float64)
        if tw.shape[0] != w.shape[0]:
            raise ValueError(
                f"tail_w_train 长度 {tw.shape[0]} 与 w_train 长度 {w.shape[0]} 不一致")
        tail_pos_mass = float(tw[pos].sum())
        if not np.isfinite(tail_pos_mass) or tail_pos_mass <= 0:
            raise ValueError(f"tail 正样本权重和非法: {tail_pos_mass!r}")
        return base * (tail_pos_mass / pos_mass)

    # adaptive
    ratio = n_pos / n_neg
    mean_w_pos = pos_mass / n_pos
    raw = base * (SPW_ADAPT_R0 / ratio) * (SPW_ADAPT_W0 / mean_w_pos)
    return float(np.clip(raw, SPW_ADAPT_MIN, SPW_ADAPT_MAX))


def build_train_weights(core_pos: np.ndarray, buf_pos: np.ndarray,
                        y_raw: np.ndarray, dates: np.ndarray,
                        y_train: np.ndarray, base_spw: float,
                        weight_mode: str = "tail",
                        spw_mode: str = "fixed",
                        vol20: np.ndarray | None = None,
                        weight_raw: np.ndarray | None = None,
                        weight_r0: float | None = None) -> tuple[np.ndarray, float]:
    """构造单个训练子集的样本权重与有效 spw（CV/OOS 共用，避免口径漂移）。

    match_tail 时在同一子集上额外重算 tail 权重，保证两种口径的正样本
    总梯度质量严格匹配；fixed/adaptive 不额外重算。
    weight_raw/weight_r0 透传给 compute_sample_weights（excess 模式传
    r3_excess / 6%，classic/v51 保持默认绝对 r3 / 7%）。
    """
    w = compute_sample_weights(core_pos, buf_pos, y_raw, dates,
                               weight_mode=weight_mode, vol20=vol20,
                               weight_raw=weight_raw, weight_r0=weight_r0)
    tail_w = None
    if spw_mode == "match_tail":
        tail_w = compute_sample_weights(core_pos, buf_pos, y_raw, dates,
                                        weight_mode="tail", vol20=None,
                                        weight_raw=weight_raw,
                                        weight_r0=weight_r0)
    spw = resolve_effective_spw(y_train, w, base_spw,
                                tail_w_train=tail_w, spw_mode=spw_mode)
    return w, spw


# ==================== 双核异构多 Seed 训练器 ====================
def _logit(p: np.ndarray) -> np.ndarray:
    """概率→对数几率（clip 防爆）。多 seed 在 logit 空间平均，避免
    `0.5*rank_pct+0.5*p` 量纲污染 score_z_in_day（rank_pct 有界均匀分布会
    锁死 z 分母，使 z>1.5 门控只剩原始概率畸高的高危断板股）。"""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


# ==================== Leaf-Orthogonal Pick2（2026-10-21） ====================
LEAF_PICK2_CPS = (170, 260)
LEAF_PICK2_TOPN = 15
LEAF_PICK2_OVERLAP_MAX = 0.25
LEAF_PICK2_THRESHOLDS = (0.25, 0.30, 0.35, 0.40)
LEAF_PICK2_MIN_RANK_END = 8  # MinOverlap-Pick2 只在 Dedup Rank 2..8 中选


def _leaf_pick2_stats(X_te, dates_te, codes_te, sectors_te, y_raw_te,
                      preds, bst_list, cp) -> dict:
    """行业去重 Top15 内，Rank1 之后按叶子重合率选 Pick2（多阈值 + MinOverlap）。

    叶子向量 = 3 seeds × cp 棵树的叶子编号拼接；重合率 = 同叶子树数占比。
    除旧版 0.25 聚合外，额外保存逐日记录 daily，供离线统计：
      - Rank2~10 与 Rank1 的叶子重合率分位数；
      - 0.25/0.30/0.35/0.40 阈值在同一天的 Rank1/Rank2/Pick2 配对收益；
      - MinOverlap-Pick2：Dedup Rank2..8 中与 Rank1 重合率最低者（每日满勤）。
    """
    n = len(preds)
    if n == 0 or not bst_list:
        return {}
    t = pd.DataFrame({
        "i": np.arange(n, dtype=np.int64),
        "date": pd.to_datetime(dates_te),
        "sector": np.asarray(sectors_te, dtype=object),
        "pred": np.asarray(preds, dtype=float),
        "ret": np.asarray(y_raw_te, dtype=float),
    })
    t = t[np.isfinite(t["pred"].to_numpy(dtype=float))]
    if len(t) == 0:
        return {}
    day_n = t.groupby("date", sort=False)["pred"].transform("size")
    t = t[day_n.to_numpy() >= 30]
    if len(t) == 0:
        return {}
    t = t.sort_values(["date", "pred"], ascending=[True, False], kind="stable")
    t = t.drop_duplicates(subset=["date", "sector"], keep="first")
    t = t.groupby("date", sort=True, group_keys=False).head(LEAF_PICK2_TOPN)
    if len(t) == 0:
        return {}
    t = t.sort_values(["date", "pred"], ascending=[True, False],
                      kind="stable").reset_index(drop=True)
    idx = t["i"].to_numpy(dtype=np.int64)
    leaves = [np.asarray(bst.predict(X_te[idx], num_iteration=int(cp),
                                     pred_leaf=True), dtype=np.int32)
              for bst in bst_list]
    L = np.concatenate(leaves, axis=1)
    rets_all = t["ret"].to_numpy(dtype=float)

    def _f(x):
        x = float(x)
        return x if np.isfinite(x) else None

    daily = []
    pick2_sum = 0.0
    pick2_n = 0
    pick2_win = 0
    top2_sum = 0.0
    top2_n = 0
    for _d, pos in t.groupby("date", sort=True).indices.items():
        if len(pos) == 0:
            continue
        rets = rets_all[pos]
        lv = L[pos]
        r1 = lv[0]
        ov = np.mean(lv == r1, axis=1)
        row = {"date": str(pd.Timestamp(_d).date()),
               "r1": _f(rets[0]),
               "r2": _f(rets[1]) if len(pos) > 1 else None,
               "ovs": [float(x) for x in ov[1:min(10, len(pos))]],
               "thr": {}, "min": None}
        if len(pos) >= 2:
            for _thr in LEAF_PICK2_THRESHOLDS:
                _key = f"{_thr:.2f}"
                _pick = None
                for _r in range(1, len(pos)):
                    if ov[_r] <= _thr:
                        _pick = _r
                        break
                row["thr"][_key] = ({"ret": _f(rets[_pick]),
                                     "rank": int(_pick + 1)}
                                    if _pick is not None else None)
            _kmax = min(len(pos), LEAF_PICK2_MIN_RANK_END)
            if _kmax >= 2:
                _k = min(range(1, _kmax), key=lambda r: (float(ov[r]), r))
                row["min"] = {"ret": _f(rets[_k]), "rank": int(_k + 1),
                              "ov": float(ov[_k])}
        daily.append(row)
        _p = row["thr"].get("0.25")
        if _p is not None and _p.get("ret") is not None:
            _r2 = _p["ret"]
            pick2_sum += _r2
            pick2_n += 1
            pick2_win += int(_r2 > 0)
            if row["r1"] is not None:
                top2_sum += 0.5 * (row["r1"] + _r2)
                top2_n += 1
    return {"cp": int(cp), "pick2_sum": float(pick2_sum), "pick2_n": int(pick2_n),
            "pick2_win": int(pick2_win), "top2_sum": float(top2_sum),
            "top2_n": int(top2_n), "overlap_max": float(LEAF_PICK2_OVERLAP_MAX),
            "daily": daily}


def _leaf_daily_metrics(daily: list[dict]) -> dict:
    """从逐日 leaf 记录计算：重合率分位数、各阈值/MinOverlap 配对均值。"""
    def _mean(v):
        return float(np.mean(v)) if v else float("nan")

    def _q(v, p):
        return float(np.percentile(v, p)) if v else float("nan")

    all_ov = []
    by_rank = {r: [] for r in range(2, 11)}
    for d in daily or []:
        for k, v in enumerate(d.get("ovs") or []):
            if v is None:
                continue
            x = float(v)
            if not np.isfinite(x):
                continue
            all_ov.append(x)
            r = k + 2
            if r in by_rank:
                by_rank[r].append(x)
    out = {"overlap_all": {"n": len(all_ov), "mean": _mean(all_ov),
                           "p10": _q(all_ov, 10), "p25": _q(all_ov, 25),
                           "p50": _q(all_ov, 50), "p75": _q(all_ov, 75)},
           "overlap_by_rank": {}}
    for r, vals in by_rank.items():
        if vals:
            out["overlap_by_rank"][str(r)] = {
                "n": len(vals), "mean": _mean(vals), "p10": _q(vals, 10),
                "p25": _q(vals, 25), "p50": _q(vals, 50), "p75": _q(vals, 75)}
    for _thr in LEAF_PICK2_THRESHOLDS:
        key = f"{_thr:.2f}"
        picks = []
        for d in daily or []:
            p = (d.get("thr") or {}).get(key)
            if p and p.get("ret") is not None and np.isfinite(p["ret"]):
                picks.append((d, p))
        r1s = [d["r1"] for d, _ in picks if d.get("r1") is not None]
        r2s = [d["r2"] for d, _ in picks if d.get("r2") is not None]
        tops = [0.5 * (d["r1"] + p["ret"]) for d, p in picks
                if d.get("r1") is not None]
        out[f"thr_{key}"] = {
            "n": len(picks), "mean": _mean([p["ret"] for _, p in picks]),
            "win": _mean([1.0 if p["ret"] > 0 else 0.0 for _, p in picks]),
            "rank1_mean": _mean(r1s), "rank2_mean": _mean(r2s),
            "top2_mean": _mean(tops), "top2_n": len(tops)}
    mins = []
    for d in daily or []:
        p = d.get("min")
        if p and p.get("ret") is not None and np.isfinite(p["ret"]):
            mins.append((d, p))
    r1s = [d["r1"] for d, _ in mins if d.get("r1") is not None]
    r2s = [d["r2"] for d, _ in mins if d.get("r2") is not None]
    tops = [0.5 * (d["r1"] + p["ret"]) for d, p in mins if d.get("r1") is not None]
    out["min"] = {
        "n": len(mins), "mean": _mean([p["ret"] for _, p in mins]),
        "win": _mean([1.0 if p["ret"] > 0 else 0.0 for _, p in mins]),
        "rank1_mean": _mean(r1s), "rank2_mean": _mean(r2s),
        "top2_mean": _mean(tops), "top2_n": len(tops),
        "mean_overlap": _mean([p.get("ov") for _, p in mins if p.get("ov") is not None])}
    return out


def _write_leaf_reports(out_dir: Path, tasks_dir: Path) -> None:
    """汇总 CV 任务 leaf JSON 的 daily，输出多阈值/MinOverlap 报告。"""
    daily_by_cp: dict[int, list[dict]] = {}
    for fp in sorted(Path(tasks_dir).glob("*_leaf.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("side") != "lgb":
            continue
        for cp_s, st in (obj.get("leaf") or {}).items():
            try:
                cp = int(cp_s)
            except Exception:
                continue
            daily_by_cp.setdefault(cp, []).extend(st.get("daily") or [])
    if not daily_by_cp:
        return
    rows, ovrows = [], []
    for cp, daily in sorted(daily_by_cp.items()):
        m = _leaf_daily_metrics(daily)
        for _thr in LEAF_PICK2_THRESHOLDS:
            key = f"{_thr:.2f}"
            rows.append({"cp": cp, "method": f"thr_{key}", **m[f"thr_{key}"]})
        rows.append({"cp": cp, "method": "min_overlap", **m["min"]})
        for scope in ["all"] + [f"rank{r}" for r in range(2, 11)]:
            src = (m["overlap_all"] if scope == "all"
                   else m["overlap_by_rank"].get(scope[4:]))
            if src:
                ovrows.append({"cp": cp, "scope": scope, **src})
    pd.DataFrame(rows).to_csv(out_dir / "leaf_orth_report.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(ovrows).to_csv(out_dir / "leaf_overlap_quantiles.csv",
                                index=False, encoding="utf-8-sig")
    print(f"[LEAF] report -> {out_dir/'leaf_orth_report.csv'} "
          f"/ {out_dir/'leaf_overlap_quantiles.csv'}", flush=True)


def train_predict_fold(
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
    X_te: np.ndarray, dates_te: np.ndarray,
    hp: dict, checkpoints: list[int], seeds: list[int],
    engines: list[str], num_threads: int, leaf_ctx: dict | None = None,
    want_train: bool = False, save_model_dir: str | None = None,
    save_name: str = '', save_cp: int | None = None
) -> dict:
    """训练并返回三套独立预测（2026-10-19 定版，Owner 批准）：

    {"lgb": {cp: logit 分数}, "aux": {cp: logit 分数} | None,
     "aux_name": "catboost"/"xgboost"/None, "blend": {cp: 0.65*lgb+0.35*aux} | None}

    主评估 = 纯 LightGBM（多 seed 在 logit 空间平均）；CatBoost/XGBoost 仅作
    平行对照（单 seed，logit；旧版固定 450 轮 aux 以 0.35 权重混入所有 cp，
    稀释轮数差异并污染 z 门控——已剔除出主预测）。
    """
    import lightgbm as lgb
    lgb_by_cp = {cp: np.zeros(len(X_te), dtype=np.float64) for cp in checkpoints}
    train_by_cp = ({cp: np.zeros(len(X_tr), dtype=np.float64) for cp in checkpoints}
                   if want_train else None)
    bst_keep = []
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, params={"feature_pre_filter": False})
    max_cp = max(checkpoints)

    for sd in seeds:
        params = {
            "objective": "binary", "metric": "None", "boosting_type": "gbdt",
            "learning_rate": 0.02, "num_leaves": hp["num_leaves"],
            "max_depth": hp.get("max_depth", -1), "min_data_in_leaf": hp["min_data_in_leaf"],
            "feature_fraction": hp["feature_fraction"], "bagging_fraction": hp["bagging_fraction"],
            "bagging_freq": 1, "reg_alpha": hp["reg_alpha"], "reg_lambda": hp["reg_lambda"],
            "scale_pos_weight": hp["scale_pos_weight"],
            "random_state": sd, "bagging_seed": sd + 100, "feature_fraction_seed": sd + 200,
            "deterministic": True, "force_col_wise": True, "num_threads": num_threads, "verbosity": -1,
        }
        bst = lgb.train(params, dtr, num_boost_round=max_cp, callbacks=[lgb.log_evaluation(period=0)])
        if save_model_dir and save_cp is not None and save_cp in checkpoints:
            try:
                _md = Path(save_model_dir)
                _md.mkdir(parents=True, exist_ok=True)
                _mp = _md / f"{save_name}_seed{sd}_cp{int(save_cp)}.txt"
                bst.save_model(str(_mp), num_iteration=int(save_cp))
            except Exception as _e:
                print(f'warning: save model failed {str(_e)[:120]}', flush=True)
        for cp in checkpoints:
            p = bst.predict(X_te, num_iteration=cp)
            lgb_by_cp[cp] += _logit(p) / len(seeds)
            if want_train:
                pt = bst.predict(X_tr, num_iteration=cp)
                train_by_cp[cp] += _logit(pt) / len(seeds)
        if leaf_ctx is not None and any(cp in checkpoints for cp in leaf_ctx.get("cps", ())):
            bst_keep.append(bst)
        else:
            del bst
    leaf_stats = {}
    if leaf_ctx is not None and bst_keep:
        for cp in leaf_ctx.get("cps", ()):
            if cp not in checkpoints:
                continue
            try:
                _st = _leaf_pick2_stats(
                    X_te, dates_te, leaf_ctx.get("codes"), leaf_ctx.get("sectors"),
                    leaf_ctx.get("y_raw"), lgb_by_cp[cp], bst_keep, cp)
                if _st:
                    leaf_stats[int(cp)] = _st
            except Exception:
                pass
        del bst_keep
    try:
        dtr._handle = None  # 尽早释放数据集句柄（lightgbm 版本差异时静默跳过）
    except Exception:
        pass

    aux_by_cp = None
    aux_name = None
    if "catboost" in engines:
        try:
            from catboost import CatBoostClassifier
            cb_w = w_tr * np.where(y_tr > 0.5, float(hp["scale_pos_weight"]), 1.0)
            cb = CatBoostClassifier(
                iterations=min(450, max_cp), learning_rate=0.03,
                depth=min(6, max(4, hp.get("max_depth", 6))),
                l2_leaf_reg=hp["reg_lambda"], random_seed=seeds[0],
                thread_count=num_threads, verbose=0
            )
            cb.fit(X_tr, y_tr, sample_weight=cb_w)
            p_cb = cb.predict_proba(X_te)[:, 1]
            aux_by_cp = {cp: _logit(p_cb) for cp in checkpoints}
            aux_name = "catboost"
        except ImportError:
            pass
    elif "xgboost" in engines:
        try:
            import xgboost as xgb
            dtr_x = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
            dte_x = xgb.DMatrix(X_te)
            x_params = {
                "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                "max_depth": min(6, max(4, hp.get("max_depth", 6))), "learning_rate": 0.02,
                "subsample": hp["bagging_fraction"], "colsample_bytree": hp["feature_fraction"],
                "min_child_weight": max(50, hp["min_data_in_leaf"] // 10),
                "reg_alpha": hp["reg_alpha"], "reg_lambda": hp["reg_lambda"],
                "scale_pos_weight": hp["scale_pos_weight"], "seed": seeds[0], "nthread": num_threads,
            }
            xbst = xgb.train(x_params, dtr_x, num_boost_round=350, verbose_eval=False)
            p_xb = xbst.predict(dte_x)
            aux_by_cp = {cp: _logit(p_xb) for cp in checkpoints}
            aux_name = "xgboost"
        except ImportError:
            pass

    blend_by_cp = None
    if aux_by_cp is not None:
        blend_by_cp = {cp: 0.65 * lgb_by_cp[cp] + 0.35 * aux_by_cp[cp]
                       for cp in checkpoints}

    return {"lgb": lgb_by_cp, "aux": aux_by_cp, "aux_name": aux_name,
            "blend": blend_by_cp, "leaf": leaf_stats, "train": train_by_cp}


def train_predict_cb(
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
    X_te: np.ndarray, hp: dict, seeds: list[int], num_threads: int,
    scale_pos_weight: float = _CB_SCALE_POS_WEIGHT
) -> dict:
    """CatBoost 独立对称树训练（2026-10-20 P2 定版，与 LGB 完全对称）：

    - 多 Seed 遍历，RawFormulaVal 逐 seed 求平均（与 LGB 的 logit 均值同口径）；
    - rsm=0.75 列采样（对齐 LGB feature_fraction=0.75）；
    - Bernoulli 子采样 0.80；单次训至 CB 最大轮，ntree_end 切片全套 cp。
    - scale_pos_weight 由 resolve_effective_spw 按折解析（默认 _CB_SCALE_POS_WEIGHT）。
    """
    from catboost import CatBoostClassifier
    cb_w = w_tr * np.where(y_tr > 0.5, float(scale_pos_weight), 1.0)
    out = {int(cp): np.zeros(len(X_te), dtype=np.float64) for cp in CB_CHECKPOINTS}
    for _sd in seeds:
        cb = CatBoostClassifier(
            iterations=max(CB_CHECKPOINTS), learning_rate=0.03,
            depth=int(hp["depth"]), l2_leaf_reg=float(hp["l2_leaf_reg"]),
            bootstrap_type="Bernoulli", subsample=0.80, rsm=0.75,
            loss_function="Logloss", random_seed=int(_sd),
            thread_count=int(num_threads), verbose=0)
        cb.fit(X_tr, y_tr, sample_weight=cb_w)
        for cp in CB_CHECKPOINTS:
            out[int(cp)] += np.asarray(
                cb.predict(X_te, prediction_type="RawFormulaVal",
                           ntree_end=int(cp)), dtype=np.float64) / len(seeds)
    return out


# ==================== 评估、同行业去重与分钟级回测对接 ====================
def evaluate_predictions(
    preds_by_cp: dict[int, np.ndarray], dates_te: np.ndarray, codes_te: np.ndarray,
    sectors_te: np.ndarray, y_raw_te: np.ndarray, fold_id: int, pid: str,
    topk: int = 30, universes: tuple = ("raw", "dedup")
) -> list[dict]:
    """逐 cp 评估：向量化横截面 z + 每日 topk，**同时输出 raw 与 dedup 两套**。

    2026-10-20（Owner 批准，P0-1）：旧版只留"同行业去重"一套，无法回答
    "不去重买 Raw Top1~2"是否更好。现在对同一天的候选分别输出：
      - universe="raw"  ：纯按 pred 降序的 topk（允许同行业多席）；
      - universe="dedup"：同申万二级行业仅留第 1 名后的 topk。
    两套共享同一 score_z_in_day（当日全量行统计），供离线槽位/去重对照。
    topk 默认 30：供 fuse_champions 融合与槽位研究（top5 不够）。
    """
    _base = pd.DataFrame({
        "date": pd.to_datetime(dates_te),
        "code": codes_te, "sector": sectors_te,
        "ret": np.asarray(y_raw_te, dtype=float),
    })
    rows = []
    for cp, pred in preds_by_cp.items():
        df = _base.copy()
        df["pred"] = np.asarray(pred, dtype=float)
        g = df.groupby("date", sort=False)["pred"]
        day_n = g.transform("size")
        med = g.transform("median")
        dev = (df["pred"] - med).abs()
        mad = dev.groupby(df["date"], sort=False).transform("median")
        std = g.transform("std")
        scale = np.where(mad.to_numpy(dtype=float) > 1e-12,
                         1.4826 * mad.to_numpy(dtype=float),
                         std.to_numpy(dtype=float) + 1e-9)
        df["score_z_in_day"] = (df["pred"].to_numpy(dtype=float)
                                - med.to_numpy(dtype=float)) / scale
        df = df[day_n.to_numpy(dtype=float) >= 30]
        if not len(df):
            continue
        df = df.sort_values(["date", "pred"], ascending=[True, False],
                            kind="stable")
        for _uni in universes:
            if _uni == "dedup":
                sub = df.drop_duplicates(subset=["date", "sector"], keep="first")
            else:
                sub = df
            sub = sub.groupby("date", sort=True, group_keys=False).head(topk).copy()
            if not len(sub):
                continue
            sub["rank"] = sub.groupby("date", sort=False).cumcount() + 1
            sub["param"] = str(pid)
            sub["fold_id"] = int(fold_id)
            sub["cp"] = int(cp)
            sub["universe"] = str(_uni)
            sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
            sub = sub.rename(columns={"code": "stock_code", "ret": "return"})
            cols = ["param", "fold_id", "cp", "date", "rank", "stock_code",
                    "sector", "pred", "score_z_in_day", "return", "universe"]
            rows.extend(sub[cols].to_dict("records"))
    return rows


def summarize_and_pick_champion(all_rows: list[dict], z_gate: float = 1.50) -> pd.DataFrame:
    """跨折平台平滑排行（仅 dedup 口径；mdd 修正如 2026-10-20 Owner 批准）。

    mdd_min 修正：旧版对 rank1 的 3 日持有收益按每日 100% 满仓 cumprod（300%
    重叠杠杆，单折微亏 -0.49% 会算出 -74% 假回撤）。现改为 **3 槽轮动**：
    每日投入 1/3 资金、双边成本 0.16%，eq = cumprod(1 + (ret-0.0016)/3)。
    """
    df = pd.DataFrame(all_rows)
    df_raw = df[df["universe"] == "raw"].copy() if "universe" in df.columns \
        else pd.DataFrame()
    if "universe" in df.columns:
        df = df[df["universe"] == "dedup"]
    # P0-1（2026-10-20）：raw 口径同指标（不去重买 Raw Top1~2 的对照）
    _raw_stats: dict = {}
    if len(df_raw):
        rr1 = df_raw[df_raw["rank"] == 1]
        rr12 = (df_raw[df_raw["rank"].isin([1, 2])]
                .groupby(["param", "fold_id", "cp", "date"])["return"].mean()
                .reset_index())
        for (pid, cp), sub in rr1.groupby(["param", "cp"], sort=False):
            ret = sub["return"].dropna().to_numpy(dtype=float)
            if len(ret) == 0:
                continue
            mdds = []
            for _, fg in sub.groupby("fold_id"):
                _r = fg.sort_values("date")["return"].to_numpy(dtype=float)
                eq = np.cumprod(1.0 + (_r - COST_PER_TRADE) / SLOT_CAPITAL)
                mdds.append(float((eq / np.maximum.accumulate(eq) - 1.0).min()))
            s12 = rr12[(rr12["param"] == pid) & (rr12["cp"] == cp)]["return"].dropna()
            _raw_stats[(pid, int(cp))] = {
                "r1_raw_mean": float(ret.mean()),
                "r1_raw_win": float((ret > 0).mean()),
                "r1_raw_q10": float(np.quantile(ret, 0.10)),
                "mdd_raw_min": float(min(mdds)),
                "top2_raw_mean": float(s12.mean()) if len(s12) else np.nan,
            }
    r1 = df[df["rank"] == 1].copy()
    r12 = df[df["rank"].isin([1, 2])].groupby(["param", "fold_id", "cp", "date"])["return"].mean().reset_index()

    records = []
    for (pid, cp), sub in r1.groupby(["param", "cp"], sort=False):
        ret = sub["return"].dropna().to_numpy(dtype=float)
        if len(ret) == 0:
            continue
        tail_pen_ret = float(np.mean(ret - 1.5 * np.maximum(0.0, -ret - 0.05)))
        fold_means = sub.groupby("fold_id")["return"].mean()
        min_fold_mean = float(fold_means.min())
        pos_folds = int((fold_means > 0).sum())

        mdds = []
        for _, fg in sub.groupby("fold_id"):
            _r = fg.sort_values("date")["return"].to_numpy(dtype=float)
            eq = np.cumprod(1.0 + (_r - COST_PER_TRADE) / SLOT_CAPITAL)
            mdds.append(float((eq / np.maximum.accumulate(eq) - 1.0).min()))

        active_sub = sub[sub["score_z_in_day"] > z_gate]["return"]
        sub12 = r12[(r12["param"] == pid) & (r12["cp"] == cp)]["return"].dropna()

        raw_score = 0.50 * tail_pen_ret + 0.30 * min_fold_mean + 0.20 * float(sub12.mean())
        rec = {
            "param": pid, "cp": int(cp), "n_days": len(ret), "n_folds": len(fold_means),
            "r1_mean": float(ret.mean()), "r1_median": float(np.median(ret)),
            "r1_win": float((ret > 0).mean()), "r1_q10": float(np.quantile(ret, 0.10)),
            "mdd_min": float(min(mdds)), "pos_folds": pos_folds, "min_fold_mean": min_fold_mean,
            "top2_dedup_mean": float(sub12.mean()),
            "z_gated_mean": float(active_sub.mean()) if len(active_sub) else 0.0,
            "z_gated_active_rate": float((sub["score_z_in_day"] > z_gate).mean()),
            "raw_score": raw_score,
        }
        rec.update(_raw_stats.get((pid, int(cp)), {}))
        records.append(rec)

    sum_df = pd.DataFrame(records).sort_values(["param", "cp"]).reset_index(drop=True)
    smooth_scores = []
    for pid, grp in sum_df.groupby("param", sort=False):
        sc = grp["raw_score"].to_numpy(dtype=float)
        n = len(sc)
        sm = np.copy(sc)
        for i in range(n):
            left = sc[i - 1] if i > 0 else sc[i]
            right = sc[i + 1] if i < n - 1 else sc[i]
            sm[i] = 0.25 * left + 0.50 * sc[i] + 0.25 * right
        smooth_scores.extend(sm.tolist())
    sum_df["plateau_score"] = smooth_scores
    return sum_df.sort_values("plateau_score", ascending=False).reset_index(drop=True)


def pick_backtest_min_dir(test_start: str, test_end: str,
                          logger: logging.Logger) -> Path:
    """回测行情目录：**前复权（minute_adj）优先**——不复权价格在除权除息日存在
    假跳空，会污染买入价/收益/涨跌停判断；因子缓存用不复权无碍（日内比值对复权
    不变，见 intraday_minute 模块），但回测价格必须前复权。按测试窗口月份覆盖
    选择候选，无前复权覆盖时退回不复权并显式告警。"""
    try:
        months_need = set(pd.period_range(test_start, test_end,
                                          freq="M").strftime("%Y%m"))
    except Exception:
        months_need = set()
    cands = [_HERE.parent / "local_data" / "minute_adj",
             _HERE / "local_data" / "minute_adj",
             _HERE.parent / "local_data" / "minute",
             _HERE / "local_data" / "minute"]
    fallback = None
    for cand in cands:
        if not cand.exists():
            continue
        if fallback is None:
            fallback = cand
        try:
            have = {d.name for d in cand.iterdir()
                    if d.is_dir() and (d / "minute_data.parquet").exists()}
        except Exception:
            continue
        if months_need and not (have & months_need):
            continue
        if "minute_adj" in cand.name:
            logger.info(f"[MINUTE-BT] 回测行情: {cand}（前复权，覆盖 "
                        f"{len(have & months_need)}/{len(months_need)} 月）")
        else:
            logger.warning(f"[MINUTE-BT] 回测行情: {cand}（**不复权**！无前复权数据"
                           f"覆盖测试窗口，除权跳空风险）")
        return cand
    logger.warning("[MINUTE-BT] 未找到覆盖测试窗口的分钟行情，使用兜底目录（可能无数据）")
    return fallback or (_HERE.parent / "local_data" / "minute")


def dedup_sector_preds(preds: np.ndarray, codes_te: np.ndarray,
                       dates_te: np.ndarray, logger: logging.Logger,
                       tag: str = "DEDUP") -> np.ndarray:
    """同申万二级行业仅保留 pred 最高者，其余置 -inf（CV/分钟回测/风控同口径）。

    2026-10-20 抽公共函数：OOS 无否决分钟回测也必须先去重（旧版只在 veto
    函数内部去重，导致无否决回测原样喂原始分数，口径不一致）。"""
    out = np.asarray(preds, dtype=float).copy()
    try:
        _swm = load_sw2_map()
        if _swm:
            _sec = np.array([_swm.get(str(c), "__NA__") for c in codes_te],
                            dtype=object)
            _t = pd.DataFrame({"i": np.arange(len(out)),
                               "date": pd.to_datetime(dates_te),
                               "pred": out, "sec": _sec})
            _t = _t.sort_values(["date", "pred"], ascending=[True, False],
                                kind="stable")
            _dup = _t.duplicated(subset=["date", "sec"], keep="first").to_numpy()
            _dup_idx = _t["i"].to_numpy()[_dup]
            out[_dup_idx] = -np.inf
            logger.info(f"[{tag}] 行业去重：{len(_dup_idx)} 个非头名候选置 -inf")
    except Exception as _e:
        logger.warning(f"[{tag}] 行业去重失败（{str(_e)[:100]}），按原顺序处理")
    return out


def apply_risk_veto_to_preds(preds: np.ndarray, X_te: np.ndarray,
                             feature_cols: list[str], codes_te: np.ndarray,
                             dates_te: np.ndarray, y_raw_te: np.ndarray,
                             price_path: Path, rules_path: str,
                             logger: logging.Logger) -> np.ndarray:
    """OOS 分钟回测前应用风控级联否决：被否候选 predicted_return 置 -inf，
    回测引擎按 pred 降序取第一个未被否者（Rank1 被否→顺延→全否空仓）。

    顺序铁律（2026-10-19 修复）：**先同申万二级行业去重（非头名置 -inf），
    后应用风控否决**——与 CV 选参口径一致；否则被否行业真龙头会让同行业
    第 2 名浮上来被买（跟风股顶替龙头）。
    特征组装与 risk_veto_train 训练期完全一致（矩阵 + 行情派生 + 日内降频缓存
    + score_z_in_day），规则引用特征缺失时严格报错（防静默失效）。"""
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8")).get("rules", [])
    if not rules:
        logger.warning(f"[VETO] {rules_path} 中无规则，跳过否决")
        return preds
    preds = np.asarray(preds, dtype=float).copy()
    # 行业去重先行：每行业仅头名参与否决与顺延（与 CV/分钟回测口径一致）
    preds = dedup_sector_preds(preds, codes_te, dates_te, logger, tag="VETO")
    from risk_veto_train import apply_veto, derive_price_features
    fdf = pd.DataFrame(np.asarray(X_te), columns=feature_cols)
    fdf["stock_code"] = codes_te
    fdf["date"] = pd.to_datetime(dates_te)
    # score_z_in_day：与 evaluate_predictions 同日同公式（med / 1.4826*MAD）。
    # 2026-10-20 修复：行业去重后非头名是 -inf，若混入截面统计 med=-inf →
    # (-inf)-(-inf)=NaN（RuntimeWarning）；统计只在**有限 pred** 上计算，
    # 非有限行 score_z 置 NaN（风控规则不消费这些行）。
    _t = pd.DataFrame({"date": fdf["date"], "pred": preds})
    _fin = _t[np.isfinite(_t["pred"].to_numpy(dtype=float))].copy()
    _z = np.full(len(_t), np.nan, dtype=float)
    if len(_fin):
        _g = _fin.groupby("date")["pred"]
        med = _g.transform("median").to_numpy(dtype=float)
        mad = (_fin["pred"].to_numpy(dtype=float) - med)
        mad = pd.Series(np.abs(mad)).groupby(_fin["date"], sort=False) \
            .transform("median").to_numpy(dtype=float)
        std = _g.transform("std").to_numpy(dtype=float)
        scale = np.where(mad > 1e-12, 1.4826 * mad, std + 1e-9)
        _z[_fin.index.to_numpy()] = (_fin["pred"].to_numpy(dtype=float) - med) / scale
    fdf["score_z_in_day"] = _z
    need = {f for r in rules for f in r.get("features", []) if f}
    need |= {r.get("fa") for r in rules if r.get("fa")}
    need |= {r.get("fb") for r in rules if r.get("fb")}
    if need - set(fdf.columns):
        try:
            pder = derive_price_features(Path(price_path), fdf["date"].min(),
                                         fdf["date"].max())
            fdf = fdf.merge(pder, on=["stock_code", "date"], how="left")
        except Exception as e:
            logger.warning(f"[VETO] 行情派生特征失败: {str(e)[:120]}")
    if need - set(fdf.columns):
        try:
            from risk_veto_train import find_intraday_cache, load_intraday_cache
            ip = find_intraday_cache()
            if ip is not None and Path(ip).exists():
                idf = load_intraday_cache(ip)  # 裸列名→id_* 注册名（2026-10-19 修复）
                if idf is not None:
                    icols = [c for c in idf.columns if c in need]
                    if icols:
                        fdf = fdf.merge(idf[["stock_code", "date"] + icols],
                                        on=["stock_code", "date"], how="left")
        except Exception as e:
            logger.warning(f"[VETO] 日内降频缓存 join 失败: {str(e)[:120]}")

    # ===== 2026-10-20 P1→P0-3 衔接：同步组装新风控特征（与 risk_veto_train 同口径）
    # ② 板块门控 GS_sec_*（gate_search_sector.parquet 长表→按(date,sector)宽表）
    _need_sec = [c for c in need if str(c).startswith("GS_sec_")
                 and c not in fdf.columns]
    if _need_sec:
        try:
            from risk_veto_train import find_sector_gates
            _gp = find_sector_gates()
            if _gp is not None:
                _g = pd.read_parquet(_gp)
                _g = _g[_g["gate_id"].isin(_need_sec)]
                if len(_g):
                    _g["date"] = pd.to_datetime(_g["date"])
                    _piv = _g.pivot_table(index=["date", "sector"],
                                          columns="gate_id", values="p",
                                          aggfunc="mean").reset_index()
                    _swm2 = load_sw2_map() or {}
                    fdf["sector"] = fdf["stock_code"].map(_swm2).fillna("__NA__")
                    fdf = fdf.merge(_piv, on=["date", "sector"], how="left")
                    fdf = fdf.drop(columns=["sector"])
        except Exception as e:
            logger.warning(f"[VETO] 板块门控 join 失败: {str(e)[:120]}")
    # ③ 因子库列（flow_* / 因子库任意列，schema 过滤）
    if need - set(fdf.columns):
        try:
            from risk_veto_train import load_factor_lab_columns
            _f_need = sorted(
                c for c in (need - {"gap12", "z_in_top5", "score_z_in_day",
                                    "ret3", "close_pos"})
                if not str(c).startswith(("GM_", "id_")))
            if _f_need:
                fdf, _got = load_factor_lab_columns(fdf, _f_need, logger)
                if _got:
                    logger.info(f"[VETO] 因子库补齐 {_got}")
        except Exception as e:
            logger.warning(f"[VETO] 因子库 join 失败: {str(e)[:120]}")
    # ④ 内生置信度 gap12 / z_in_top5（当日去重头名 Top5 预测分派生，按日广播）
    if {"gap12", "z_in_top5"} & need:
        try:
            _t3 = pd.DataFrame({"date": fdf["date"],
                                "pred": np.asarray(preds, dtype=float)})
            _t3 = _t3[np.isfinite(_t3["pred"])].sort_values(
                ["date", "pred"], ascending=[True, False], kind="stable")
            _top5 = _t3.groupby("date", sort=False, group_keys=False).head(5)
            _rows = []
            for _d, _s in _top5.groupby("date", sort=True)["pred"]:
                _a = _s.to_numpy(dtype=float)
                _p1 = float(_a[0])
                _p2 = float(_a[1]) if len(_a) > 1 else float("nan")
                _rest = _a[1:]
                if len(_rest):
                    _med = float(np.median(_rest))
                    _mad = float(np.median(np.abs(_rest - _med)))
                    _z = (_p1 - _med) / (1.4826 * _mad + 1e-9)
                else:
                    _z = float("nan")
                _rows.append({"date": _d, "gap12": _p1 - _p2, "z_in_top5": _z})
            fdf = fdf.merge(pd.DataFrame(_rows), on="date", how="left")
            logger.info("[VETO] 内生置信度补齐 gap12/z_in_top5")
        except Exception as e:
            logger.warning(f"[VETO] 内生置信度派生失败: {str(e)[:120]}")
    # ⑤ 市场门控 GM_*（日级广播；v2 规则可能引用）
    _gm_need = [c for c in need
                if str(c).startswith("GM_") and c not in fdf.columns]
    if _gm_need:
        try:
            _gp = (_HERE / "results" / "merge_v2" / "gates_search"
                   / "market_gates_search.parquet")
            if _gp.exists():
                import pyarrow.parquet as _pqq
                _have = set(_pqq.read_schema(_gp).names)
                _use = [c for c in _gm_need if c in _have]
                if _use:
                    _gm = pd.read_parquet(_gp, columns=["date"] + _use)
                    _gm["date"] = pd.to_datetime(_gm["date"])
                    fdf = fdf.merge(_gm, on="date", how="left")
                    logger.info(f"[VETO] 市场门控补齐 {_use}")
        except Exception as e:
            logger.warning(f"[VETO] 市场门控 join 失败: {str(e)[:120]}")
    missing = sorted(need - set(fdf.columns))
    if missing:
        raise SystemExit(f"[VETO] 规则引用特征缺失: {missing}（先并入对应数据源再回测）")
    # 2026-10-20 Owner 决策：**只对当日 Rank1 应用否决**；Rank1 被否 → 当日空仓，
    # 不顺延 Rank2（11 折 CV 只有 Rank1 值得买，顺延等于买跟风股）。
    out = preds.copy()
    _idx_fin = np.where(np.isfinite(preds))[0]
    _veto_days = 0
    _r1v_ret = np.array([], dtype=float)
    if len(_idx_fin):
        _t2 = pd.DataFrame({"i": _idx_fin,
                            "date": fdf["date"].to_numpy()[_idx_fin],
                            "pred": preds[_idx_fin]})
        _t2 = _t2.sort_values(["date", "pred"], ascending=[True, False],
                              kind="stable")
        _r1 = _t2.groupby("date", sort=True).head(1)
        _r1_idx = _r1["i"].to_numpy()
        _m = apply_veto(rules, fdf.iloc[_r1_idx])
        # 逐规则 OOS 诊断（触发数/均值/大亏/大涨），便于判断哪条在起作用
        try:
            from risk_veto_train import apply_veto_mask as _avm
            _r1_df = fdf.iloc[_r1_idx]
            _r1_ret = np.asarray(y_raw_te, dtype=float)[_r1_idx]
            for _r in rules:
                _mr = _avm(_r1_df, _r)
                if _mr.any():
                    _rr = _r1_ret[_mr]
                    logger.info(
                        f"[VETO] 规则{_r.get('id')} | 触发 {int(_mr.sum())} | "
                        f"均值 {float(_rr.mean()):+.4f} | "
                        f"大亏 {int((_rr <= -0.05).sum())} | "
                        f"大涨 {int((_rr >= 0.07).sum())}")
        except Exception as _e:
            logger.warning(f"[VETO] 逐规则诊断失败: {str(_e)[:100]}")
        _vdate = pd.to_datetime(fdf["date"].to_numpy()[_r1_idx][_m])
        if len(_vdate):
            _veto_days = int(len(np.unique(_vdate)))
            _r1v_ret = np.asarray(y_raw_te, dtype=float)[_r1_idx][_m]
            out[np.isin(fdf["date"].to_numpy(), _vdate.to_numpy())] = -np.inf
    _n_days = int(pd.to_datetime(fdf["date"]).nunique())
    logger.info(
        f"[VETO] 规则 {len(rules)} 条（Rank1-only，不顺延）| 否决 Rank1 "
        f"{_veto_days}/{_n_days} 天 → 当日空仓"
        f"（被否 Rank1 实际收益均值 "
        f"{(_r1v_ret.mean() if len(_r1v_ret) else 0.0):.4f}；"
        f"空仓相对基准日收益差均值 "
        f"{(-_r1v_ret.mean() if len(_r1v_ret) else 0.0):.4f}）")
    return out


def veto_daily_cascade(preds: np.ndarray, dates_te: np.ndarray,
                       codes_te: np.ndarray, sectors_te: np.ndarray,
                       y_raw_te: np.ndarray, pid: str, cp: int,
                       topk: int = 5) -> list[dict]:
    """[VETO-OOS] 日频终验明细（2026-10-20 Owner 决策：Rank1-only）。

    preds 已含行业去重（非头名 -inf）与风控否决（被否日全 -inf）。
    **只买当日 Rank1；Rank1 被否 → 当日空仓（CASH，return=0），不顺延 Rank2。**
    返回与 evaluate_predictions 同构的 rows（供 summarize_and_pick_champion
    出表；空仓日以 stock_code="CASH" 明确标记，避免与"无样本"混淆）。"""
    _t = pd.DataFrame({"date": pd.to_datetime(dates_te), "code": codes_te,
                       "sector": sectors_te,
                       "pred": np.asarray(preds, dtype=float),
                       "ret": np.asarray(y_raw_te, dtype=float)})
    rows = []
    n_cash = 0
    for d, g in _t.groupby("date", sort=True):
        if len(g) < 30:
            continue
        _alive = g[g["pred"] > -np.inf].sort_values(
            "pred", ascending=False, kind="stable").head(topk)
        _pf = g["pred"].to_numpy(dtype=float)
        _pf = _pf[np.isfinite(_pf)]
        if len(_pf):
            med = float(np.median(_pf))
            mad = float(np.median(np.abs(_pf - med)))
            scale = 1.4826 * mad if mad > 1e-12 else float(np.std(_pf) + 1e-9)
        else:
            med, scale = 0.0, 1.0
        if not len(_alive):
            n_cash += 1
            rows.append({"param": pid, "fold_id": 99, "cp": cp,
                         "date": str(pd.Timestamp(d).date()), "rank": 1,
                         "stock_code": "CASH", "sector": "__CASH__",
                         "pred": 0.0, "score_z_in_day": 0.0, "return": 0.0})
            continue
        r = _alive.iloc[0]
        rows.append({"param": pid, "fold_id": 99, "cp": cp,
                     "date": str(pd.Timestamp(d).date()), "rank": 1,
                     "stock_code": r["code"], "sector": r["sector"],
                     "pred": float(r["pred"]),
                     "score_z_in_day": (float(r["pred"]) - med) / scale,
                     "return": float(r["ret"])})
    if n_cash:
        print(f"[VETO] Rank1 被否空仓 {n_cash} 天（按 0 收益计入日频表）",
              flush=True)
    return rows


def run_minute_backtest_oos(pred_df: pd.DataFrame, price_path: Path, test_start: str,
                            test_end: str, out_dir: Path, logger: logging.Logger,
                            strategy_overrides: dict | None = None,
                            out_name: str = "minute_backtest_oos.pkl",
                            position_schedule: list | None = None,
                            max_concurrent_positions: int = 2):
    """直接调用 strategy_backtest_top1_gate_ma5_binary 跑 5 分钟级状态机真实回测（**仅 Rank1**）。

    2026-10-20 Owner 决策：
    - 只跑 Top-1 策略（当日 Rank1 单只）；Top-2=Rank2 单只的旧对照已删除。
    - 仓位表改为旧产线 [0.8, 0.2]（两槽：首仓 80%、次仓 20%，持有 3 日）。
    - 分钟回测必须显式传 strategy_config_overrides（与旧主脚本生产配置同构，
      见 OOS_MINUTE_OVERRIDES）——否则引擎默认
      enable_top1_state_machine_risk=False / enable_t2_1000_soft_stop=False，
      Top1 状态机静默失效。"""
    try:
        from rqalpha_minute_datasource.strategy_backtest_top1_gate_ma5_binary import run_optimized_backtest
    except ImportError:
        # 2026-10-20：服务器无 rqalpha（包级导入会因引擎内部 `import rqalpha`
        # 失败），回退到把引擎目录加入 sys.path 的顶层导入（引擎同时支持两种）。
        import sys as _sys
        _md = str(_HERE / "rqalpha_minute_datasource")
        if _md not in _sys.path:
            _sys.path.insert(0, _md)
        try:
            from strategy_backtest_top1_gate_ma5_binary import run_optimized_backtest
        except ImportError as e:
            logger.warning(f"[MINUTE-BT] 未找到分钟回测引擎，跳过: {e}")
            return

    logger.info(f"[MINUTE-BT] 组装行情骨架并启动 5 分钟级状态机回测 ({test_start} ~ {test_end}) ...")
    # 与 CV 选参同口径：同 (date, 申万二级行业) 仅保留 pred 最高者，其余置 -inf
    # （CV 评估用 drop_duplicates(keep=first) 的行业去重；回测若无此步则两口径不一致）。
    try:
        _swm = load_sw2_map()
        if _swm and len(pred_df):
            pred_df = pred_df.copy()
            pred_df["_sec"] = pred_df["stock_code"].map(_swm).fillna("__NA__")
            pred_df = pred_df.sort_values(["date", "predicted_return"],
                                          ascending=[True, False], kind="stable")
            _dup = pred_df.duplicated(subset=["date", "_sec"], keep="first")
            pred_df.loc[_dup, "predicted_return"] = -np.inf
            pred_df = pred_df.drop(columns=["_sec"])
            logger.info(f"[MINUTE-BT] 行业去重：{int(_dup.sum())} 个行业非头名候选置 -inf")
    except Exception as _e:
        logger.warning(f"[MINUTE-BT] 行业去重失败（{str(_e)[:100]}），保持原排序")
    sf = pd.read_parquet(price_path)
    sf["date"] = pd.to_datetime(sf["date"]).dt.normalize()
    sf["stock_code"] = sf["stock_code"].astype(str)
    sf = sf[(sf["date"] >= pd.Timestamp(test_start)) & (sf["date"] <= pd.Timestamp(test_end))].copy()

    _pred_cols = ["date", "stock_code", "predicted_return"]
    _pred_cols += [c for c in ("regime_skip_buy", "regime_position_scale")
                   if c in pred_df.columns]
    bt_df = sf.merge(pred_df[_pred_cols], on=["date", "stock_code"], how="left")
    bt_df["predicted_return"] = bt_df["predicted_return"].fillna(-np.inf)

    # 自动定位 5 分钟数据目录（前复权优先，见 pick_backtest_min_dir）
    min_dir = pick_backtest_min_dir(test_start, test_end, logger)

    res = run_optimized_backtest(
        prediction_data=bt_df,
        stock_counts=[1],            # 2026-10-20 Owner：只买当日 Rank1（Top-2 已删除）
        minute_data_path=str(min_dir),
        holding_days=3,
        enable_dynamic_rank_switch=False,
        max_concurrent_positions=int(max_concurrent_positions),
        fixed_position_pct=0.5,
        position_schedule=([0.8, 0.2] if position_schedule is None
                           else position_schedule),
        strategy_config_overrides=(strategy_overrides
                                   if strategy_overrides is not None
                                   else OOS_MINUTE_OVERRIDES),
    )
    if res:
        pd.to_pickle(res, out_dir / out_name)
        for k, v in res.items():
            if isinstance(v, dict):
                logger.info(
                    f"[MINUTE-BT] {k}: 总收益={v.get('total_return', 0):.2f}% | "
                    f"年化={v.get('annual_return', 0)*100:.2f}% | 最大回撤={v.get('max_drawdown', 0):.2f}% | "
                    f"夏普={v.get('sharpe_ratio', 0):.2f} | 胜率={v.get('win_rate', 0):.1f}% | "
                    f"交易笔数={v.get('total_trades', 0)}"
                )


    return res


# ==================== CV 折级并行（fork 继承大数组，串行/并行同一 worker） ====================
_CV_CTX: dict | None = None


def _cv_worker(task) -> tuple:
    """CV 单 (side, 超参, 折) 任务：训练+评估（串行/并行共用，零漂移）。

    side ∈ {"lgb", "catboost"}；返回 (任务序号, side, pid, fold_id, 评估行数, rows)。
    2026-10-19 定版：两侧彻底解耦——LGB 用每超参专属 checkpoints（3 seeds logit
    均值）；CatBoost 独立对称树网格（RawFormulaVal + ntree_end 全套切片）。
    """
    t_i, side, hp, fold = task
    c = _CV_CTX
    dates, valid_mask = c["dates"], c["valid_mask"]
    train_mask = c.get("train_mask", valid_mask)
    trade_mask = c.get("trade_mask", valid_mask)
    test_mask = c.get("test_mask", trade_mask)
    tr_m = ((dates >= np.datetime64(fold["train_start"]))
            & (dates <= np.datetime64(fold["train_end"])) & train_mask)
    # 测试集只用 T 日盘后可知条件（trade_mask），不剔除无标签/未来停牌日
    te_m = ((dates >= np.datetime64(fold["test_start"]))
            & (dates <= np.datetime64(fold["test_end"])) & test_mask)
    if tr_m.sum() < c["min_train"] or te_m.sum() < c["min_test"]:
        return t_i, side, hp["pid"], fold["fold_id"], 0, [], {}
    _vol_tr = c["vol20"][tr_m] if c.get("vol20") is not None else None
    _label_mode = str(c.get("label_mode", "classic"))
    _weight_all = c.get("y_weight_raw", c["y_raw"])
    _weight_tr = _weight_all[tr_m]
    _weight_r0 = EXCESS_DELTA_MIN if _label_mode == "excess" else R3_CORE_MIN
    _spw_override = c.get("base_spw")
    _base_spw = (float(_spw_override) if side == "lgb" and _spw_override is not None
                 else float(hp["scale_pos_weight"]) if side == "lgb"
                 else float(_CB_SCALE_POS_WEIGHT))
    w_tr, _spw_eff = build_train_weights(
        c["core_pos"][tr_m], c["buf_pos"][tr_m], c["y_raw"][tr_m], dates[tr_m],
        c["y"][tr_m], _base_spw,
        weight_mode=c.get("weight_mode", "tail"),
        spw_mode=c.get("spw_mode", "fixed"),
        vol20=_vol_tr, weight_raw=_weight_tr, weight_r0=_weight_r0)
    # 浅拷贝避免把当折有效 spw 写回共享 HP_GRID（串行路径多折复用同一 dict）
    hp_eff = dict(hp)
    hp_eff["scale_pos_weight"] = _spw_eff
    _leaf = {}
    if side == "lgb":
        _cps = hp_eff.get("checkpoints", CHECKPOINTS)
        _leaf_ctx = None
        if c.get("collect_leaf"):
            _leaf_ctx = {"codes": c["codes"][te_m], "sectors": c["sectors"][te_m],
                         "y_raw": c["y_raw"][te_m], "cps": LEAF_PICK2_CPS}
        _out = train_predict_fold(
            c["X"][tr_m], c["y"][tr_m], w_tr, c["X"][te_m], dates[te_m],
            hp_eff, _cps, c["seeds"], ["lgb"], c["num_threads"], leaf_ctx=_leaf_ctx,
            save_model_dir=c.get("save_model_dir"),
            save_name=f"{side}_{hp['pid']}_Fold{int(fold['fold_id']):02d}",
            save_cp=c.get("save_cp"))
        preds = _out["lgb"]
        _leaf = _out.get("leaf", {})
    else:
        preds = train_predict_cb(
            c["X"][tr_m], c["y"][tr_m], w_tr, c["X"][te_m],
            hp_eff, c["seeds"], c["num_threads"],
            scale_pos_weight=_spw_eff)
    rows = evaluate_predictions(
        preds, dates[te_m], c["codes"][te_m], c["sectors"][te_m],
        c["y_raw"][te_m], fold["fold_id"], hp_eff["pid"])
    for r in rows:
        r["scheme"] = side
    if rows:
        # 仅挂在首行供父进程写任务标记/日志；_persist_task 会按 _ROWS_COLS
        # reindex 自动丢弃，不改变 parquet 行 schema。
        rows[0]["spw_eff"] = float(_spw_eff)
    return t_i, side, hp["pid"], fold["fold_id"], len(rows), rows, _leaf


def auto_cv_jobs(X: np.ndarray, engines: list[str], num_threads: int,
                 n_tasks: int, logger: logging.Logger) -> int:
    """按服务器 CPU/内存自动定折级并行数（--jobs auto）。
    规则（防 OOM 的保守估算，Linux fork 共享只读 X，主份额=折切片+引擎副本）：
      cpu_jobs = floor(核数 / num_threads)   ← 满载（专用服务器，去 0.9 折扣）
      单worker ≈ 1.0GB + 1.2×X.nbytes + (1.5GB if catboost else 0.3GB)
      mem_jobs = floor(0.7×可用内存 / 单worker)
      jobs = clamp(min(cpu_jobs, mem_jobs), 1, 任务数)"""
    cores = os.cpu_count() or 4
    cpu_jobs = max(1, cores // max(1, num_threads))
    mem_gb = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_gb = int(line.split()[1]) / 1048576.0
                    break
    except Exception:
        pass
    if mem_gb is None:
        try:
            import psutil
            mem_gb = psutil.virtual_memory().available / 1073741824.0
        except Exception:
            mem_gb = 8.0
            logger.warning("[AUTO] 无法读取可用内存（无 /proc/meminfo 且无 psutil），按保守 8GB 估算")
    x_gb = float(X.nbytes) / 1073741824.0
    per_worker = 1.0 + 1.2 * x_gb + (1.5 if "catboost" in engines else 0.3)
    mem_jobs = max(1, int(mem_gb * 0.7 / per_worker))
    jobs = max(1, min(cpu_jobs, mem_jobs, max(1, n_tasks)))
    logger.info(f"[AUTO] 核数={cores} 可用内存={mem_gb:.1f}GB X={x_gb:.2f}GB "
                f"引擎={engines} 单worker估≈{per_worker:.1f}GB -> jobs={jobs} "
                f"(cpu限={cpu_jobs} 内存限={mem_jobs} 任务={n_tasks})")
    return jobs


# ==================== 标签与权重全景诊断（--mode diag_labels） ====================
def run_label_diagnostics(args, out_dir: Path, logger: logging.Logger) -> None:
    """标签与权重全景诊断（2026-10-19 定版，无模型训练）。

    在全量可交易样本（data_start ~ cv_end，特征∩行情同域）上输出：
      A. L0~L5 六种标签组合漏斗：正样本数/率、真实等权 r3/r5/dd3 均值与中位数；
         L5=截面超额 (r3_excess>=6% & r3>=3.5%)，收益统计仍用绝对 r3/r5/dd3；
      B. 各标签每日正样本数分布、Top10 海啸日、P95 软封顶前后海啸日权重占比；
      C. spw ∈ {15,20,25,30,38} 有效正类梯度质量占比（目标区间 35%~42%）。
    """
    t0 = time.time()
    # ---- [1] 特征键（与训练样本严格同域：特征∩行情、剔除 BJ、限 sw2 映射）----
    feat_path = Path(args.features_parquet)
    if not feat_path.is_absolute():
        feat_path = _HERE / feat_path
    if "with_h3" in feat_path.name:
        raise SystemExit(
            "[DEPRECATED] 91 列 with_h3 变体已废弃（84 列 A/B 全面更差）；"
            "请使用 lgb_features.parquet（84 列）")
    logger.info(f"[diag] 特征矩阵键: {feat_path}")
    fkeys = pd.read_parquet(feat_path, columns=["stock_code", "date"])
    fkeys["date"] = pd.to_datetime(fkeys["date"]).dt.normalize()
    fkeys["stock_code"] = fkeys["stock_code"].astype(str)
    fkeys = fkeys[fkeys["date"] >= pd.Timestamp(args.data_start)]
    sw_map = load_sw2_map()
    if not sw_map:
        raise SystemExit("sw2 行业映射为空，诊断与训练口径不一致，终止")
    fkeys = fkeys.loc[~fkeys["stock_code"].str.endswith((".BJ", ".bj"), na=False)]
    fkeys = fkeys[fkeys["stock_code"].isin(set(sw_map.keys()))]
    fkeys = fkeys.drop_duplicates(["stock_code", "date"])

    # ---- [2] 行情 + 标签原语（与 load_dataset_and_labels 逐行同口径）----
    price_path = find_best_price_cache(args.data_start)
    logger.info(f"[diag] 行情缓存: {price_path.name}")
    want_cols = ["stock_code", "date", "open", "high", "low", "close", "volume",
                 "amount", "pct_chg", "is_st", "is_limit_up", "is_limit_down",
                 "is_zhaban", "is_can_buy"]
    import pyarrow.parquet as pq
    avail_pcols = [c for c in want_cols if c in set(pq.read_schema(price_path).names)]
    pdf = pd.read_parquet(price_path, columns=avail_pcols)
    pdf["date"] = pd.to_datetime(pdf["date"]).dt.normalize()
    pdf["stock_code"] = pdf["stock_code"].astype(str)
    pdf = pdf.sort_values(["stock_code", "date"]).drop_duplicates(
        ["stock_code", "date"], keep="last").reset_index(drop=True)
    if "pct_chg" not in pdf.columns:
        raise SystemExit(
            f"[diag] 行情缓存缺 pct_chg，无法输出 D 节 vol20/SNR 诊断: {price_path.name}")
    pdf["vol20"] = compute_vol20(pdf)
    g_code = pdf.groupby("stock_code", sort=False)
    pdf["listed_days"] = g_code.cumcount() + 1
    is_susp = (pdf["volume"].fillna(0) <= 0) | (pdf["amount"].fillna(0) <= 0)
    recent_susp = is_susp.astype(float).groupby(
        pdf["stock_code"], sort=False).rolling(10, min_periods=1).max() \
        .reset_index(level=0, drop=True) > 0
    # 与 load_dataset_and_labels 同口径：未来 5 个交易日含停牌的样本剔除
    pdf["forward_ok"] = forward_window_ok(
        pdf["date"].to_numpy(), pdf["stock_code"].to_numpy(), horizon=5)

    tp_eq = (pdf["open"] + pdf["high"] + pdf["low"] + pdf["close"]) / 4.0
    tp_real = 0.15 * pdf["open"] + 0.15 * pdf["high"] + 0.20 * pdf["low"] + 0.50 * pdf["close"]
    buy_p = g_code["open"].shift(-1)
    can_buy_t1 = g_code["is_can_buy"].shift(-1).fillna(1) if "is_can_buy" in pdf.columns else 1

    def _sell(tp):
        _g = tp.groupby(pdf["stock_code"], sort=False)
        _s3 = 0.70 * _g.shift(-2) + 0.30 * _g.shift(-3)
        _s5 = 0.70 * _g.shift(-4) + 0.30 * _g.shift(-5)
        return _s3, _s5, _g.shift(-2)

    sell3_eq, sell5_eq, _ = _sell(tp_eq)
    sell3_r, sell5_r, tp2_r = _sell(tp_real)
    close_t2 = g_code["close"].shift(-2)
    min_low_23 = np.minimum(g_code["low"].shift(-2), g_code["low"].shift(-3))
    valid_buy = (buy_p > 1e-6) & (can_buy_t1 == 1)

    def _ret(sell):
        return np.where(valid_buy, (sell - buy_p) / buy_p, np.nan)

    r3_eq, r5_eq = _ret(sell3_eq), _ret(sell5_eq)
    r3_r, r5_r = _ret(sell3_r), _ret(sell5_r)
    dd3 = np.where(valid_buy, (min_low_23 - buy_p) / buy_p, np.nan)
    ret_t2_r = np.where(valid_buy, (tp2_r - buy_p) / buy_p, np.nan)
    close_t2_ret = np.where(valid_buy, (close_t2 - buy_p) / buy_p, np.nan)

    tradable = (
        (pdf.get("is_st", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf["listed_days"] >= NEW_STOCK_MIN_DAYS)
        & (~recent_susp)
        & pdf["forward_ok"]
        & (pdf.get("is_limit_up", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf.get("is_limit_down", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf.get("is_zhaban", pd.Series(0, index=pdf.index)).fillna(0) == 0)
        & (pdf["amount"].fillna(0) >= MIN_DAILY_AMOUNT)
        & valid_buy
    )
    pdf_sub = pd.DataFrame({
        "stock_code": pdf["stock_code"].to_numpy(), "date": pdf["date"].to_numpy(),
        "tradable": tradable.to_numpy(), "r3_eq": r3_eq, "r5_eq": r5_eq,
        "r3_r": r3_r, "r5_r": r5_r, "dd3": dd3,
        "ret_t2_r": ret_t2_r, "close_t2_ret": close_t2_ret,
        "vol20": pdf["vol20"].to_numpy()})
    merged = fkeys.merge(pdf_sub, on=["stock_code", "date"], how="inner")
    merged = merged[merged["date"] <= pd.Timestamp(args.cv_end)]

    _r3 = merged["r3_eq"].to_numpy(dtype=float)
    _r5 = merged["r5_eq"].to_numpy(dtype=float)
    _r3r = merged["r3_r"].to_numpy(dtype=float)
    _r5r = merged["r5_r"].to_numpy(dtype=float)
    _dd = merged["dd3"].to_numpy(dtype=float)
    _t2 = merged["ret_t2_r"].to_numpy(dtype=float)
    _c2 = merged["close_t2_ret"].to_numpy(dtype=float)
    _vol = merged["vol20"].to_numpy(dtype=float)
    _dates = merged["date"].to_numpy()
    base = (merged["tradable"].to_numpy(dtype=bool)
            & np.isfinite(_r3) & np.isfinite(_dd))
    n_valid = int(base.sum())
    day_codes, uniq_days = pd.factorize(_dates[base], sort=True)
    nd = len(uniq_days)
    logger.info(f"[diag] 样本: 可交易有效行={n_valid:,} | 交易日={nd} | "
                f"{pd.Timestamp(uniq_days[0]).date()}~{pd.Timestamp(uniq_days[-1]).date()}")
    if not n_valid:
        raise SystemExit("[diag] 无有效样本，检查 --data_start/--cv_end/缓存覆盖")

    r5_ok = np.isnan(_r5) | (_r5 >= (_r3 - R5_TOLERANCE))
    r5_strict = np.isnan(_r5) | (_r5 >= _r3)
    r5_ok_r = np.isnan(_r5r) | (_r5r >= (_r3r - R5_TOLERANCE))
    t2_ok = (_t2 >= T2_FLOOR_MIN) & (_c2 > 0.0)
    dd_ok = _dd >= DD3_CORE_MIN
    core0 = base & (_r3 >= R3_CORE_MIN) & r5_ok & dd_ok
    core1 = base & (_r3 >= R3_CORE_MIN) & r5_strict & dd_ok
    core2 = base & (_r3r >= R3_CORE_MIN) & r5_ok_r & dd_ok
    core3 = core0 & t2_ok
    buf4 = base & (~core0) & (_r3 >= R3_BUF_MIN) & (_r3 < R3_CORE_MIN) \
        & t2_ok & r5_ok & (_dd >= DD3_BUF_MIN)

    # ---- 正样本权重倍数（与 compute_sample_weights 同口径，供 C/D 节对照）----
    _q95_core = (float(np.percentile(_r3[core0], 95)) if int(core0.sum()) > 10
                 else 0.20)
    _q95_core = max(_q95_core, R3_CORE_MIN + 0.02)
    _tail_mult = np.ones(len(core0), dtype=float)
    _g_core = np.clip((_r3[core0] - R3_CORE_MIN) / (_q95_core - R3_CORE_MIN),
                      0.0, 1.0)
    _tail_mult[core0] = 1.0 + TAIL_WEIGHT_ALPHA * _g_core
    _snr_core = _r3[core0] / (np.maximum(_vol[core0], VOL20_FLOOR) * np.sqrt(3.0))
    _snr_mult = np.ones(len(core0), dtype=float)
    _w_core = (SNR_W_LO + (SNR_W_HI - SNR_W_LO)
               * (_snr_core - SNR_ANCHOR_LO) / (SNR_ANCHOR_HI - SNR_ANCHOR_LO))
    _w_core = np.clip(_w_core, SNR_W_LO, SNR_W_HI)
    _w_core[~np.isfinite(_snr_core)] = 1.0
    _snr_mult[core0] = _w_core

    # L5 截面超额标签（2026-10-24）：按日可交易有效样本的 r3 中位数；
    # 只改训练标签，A 节收益统计仍用绝对 r3/r5/dd3，与 L0 同尺可比。
    _r3_med_base = pd.Series(_r3[base]).groupby(
        day_codes, sort=False).transform("median").to_numpy(dtype=float)
    _r3_med = np.full(len(_r3), np.nan, dtype=float)
    _r3_med[base] = _r3_med_base
    _r3_excess = _r3 - _r3_med
    core5 = (base & (_r3_excess >= EXCESS_DELTA_MIN)
             & (_r3 >= EXCESS_R3_FLOOR) & r5_ok & dd_ok)
    _q95_core5 = (float(np.percentile(_r3_excess[core5], 95)) if int(core5.sum()) > 10
                  else 0.15)
    _q95_core5 = max(_q95_core5, EXCESS_DELTA_MIN + 0.02)
    _tail_mult5 = np.ones(len(core5), dtype=float)
    _g_core5 = np.clip((_r3_excess[core5] - EXCESS_DELTA_MIN) / (_q95_core5 - EXCESS_DELTA_MIN),
                       0.0, 1.0)
    _tail_mult5[core5] = 1.0 + TAIL_WEIGHT_ALPHA * _g_core5
    _snr_core5 = _r3_excess[core5] / (np.maximum(_vol[core5], VOL20_FLOOR) * np.sqrt(3.0))
    _snr_mult5 = np.ones(len(core5), dtype=float)
    _w_core5 = (SNR_W_LO + (SNR_W_HI - SNR_W_LO)
                * (_snr_core5 - SNR_ANCHOR_LO) / (SNR_ANCHOR_HI - SNR_ANCHOR_LO))
    _w_core5 = np.clip(_w_core5, SNR_W_LO, SNR_W_HI)
    _w_core5[~np.isfinite(_snr_core5)] = 1.0
    _snr_mult5[core5] = _w_core5

    _md: list[str] = []

    def _emit(line: str = ""):
        logger.info(line)
        _md.append(line)

    def _stat(x, pos):
        v = x[pos]
        v = v[np.isfinite(v)]
        if not len(v):
            return "n/a"
        return f"{np.mean(v):+.4f} / {np.median(v):+.4f}"

    # ---- A. 标签变体漏斗对照 ----
    _emit("\n## A. 标签变体漏斗对照（L0~L5；收益统计=等权 TP 真实前向绝对收益）")
    _emit("| 变体 | 正样本数 | 正类率 | r3_eq 均值/中位 | r5_eq 均值/中位 | dd3 均值/中位 |")
    _emit("|---|---|---|---|---|---|")
    _rows = [
        ("L0 经典基准", core0),
        ("L1 严格 r5>=r3", core1),
        ("L2 TP_real", core2),
        ("L3 L0+t2_ok", core3),
        ("L4 L0+buf4.5~7%", core0 | buf4),
        ("L5 截面超额(r3_ex>=6% & r3>=3.5%)", core5),
    ]
    for _name, _pos in _rows:
        _np_ = int(_pos.sum())
        _emit(f"| {_name} | {_np_:,} | {_np_ / n_valid:.2%} | {_stat(_r3, _pos)} | "
              f"{_stat(_r5, _pos)} | {_stat(_dd, _pos)} |")
    _n0 = int(core0.sum())
    _n5 = int(core5.sum())
    _thr_day = np.maximum(EXCESS_R3_FLOOR, _r3_med + EXCESS_DELTA_MIN)
    # 门槛按"日"取分位（每日常数），避免大截面日重复计数
    _base_pos = np.flatnonzero(base)
    _, _first_day = np.unique(_dates[_base_pos], return_index=True)
    _thr_days = _thr_day[_base_pos[_first_day]]
    _thr_days = _thr_days[np.isfinite(_thr_days)]
    if len(_thr_days):
        _tq = np.percentile(_thr_days, [10, 50, 90])
        _emit(f"- L5 相对 L0：正样本 {_n5:,} vs {_n0:,}（{_n5 / max(_n0, 1):.2f}×），"
              f"正类率 {_n5 / n_valid:.2%}；每日绝对门槛 P10/P50/P90 = "
              f"{_tq[0]:.2%}/{_tq[1]:.2%}/{_tq[2]:.2%}")
    _emit(f"\n- L1 相对 L0 减少 {_n0 - int(core1.sum()):,} 行（r5 容差 -3% 的贡献）")
    _emit(f"- L2 相对 L0 减少 {_n0 - int(core2.sum()):,} 行；L2 自身 r3_real 均值/中位 = "
          f"{_stat(_r3r, core2)}（量尺变化）")
    _emit(f"- L3 相对 L0 被杀 {_n0 - int(core3.sum()):,} 行"
          f"（{( _n0 - int(core3.sum())) / max(_n0, 1):.1%} of L0）")
    _killed = core0 & (~t2_ok)
    if _killed.any():
        _kr3 = _r3[_killed]
        _kr5 = _r5[_killed]
        _kr3f = _kr3[np.isfinite(_kr3)]
        _kr5f = _kr5[np.isfinite(_kr5)]
        _emit(f"  - 被杀样本 r3_eq: 均值 {_kr3f.mean():+.4f} / 中位 {np.median(_kr3f):+.4f} / "
              f"P25 {np.percentile(_kr3f, 25):+.4f} / P75 {np.percentile(_kr3f, 75):+.4f}")
        _emit(f"  - 被杀样本 r5_eq: 均值 {_kr5f.mean():+.4f} / 中位 {np.median(_kr5f):+.4f}；"
              f"其中 r5>=r3（T+3 反包力度）占比 "
              f"{float(np.mean(_kr5 >= _kr3)):.1%}，r5>=r3+5% 占比 "
              f"{float(np.mean(_kr5 >= _kr3 + 0.05)):.1%}")
    _n_buf = int(buf4.sum())
    _wt = int(core0.sum()) + BUF_POS_WEIGHT * _n_buf
    _emit(f"- L4 缓冲膨胀：核心 {_n0:,} + 缓冲 {_n_buf:,} = {_n0 + _n_buf:,} 行"
          f"（计数 ×{(_n0 + _n_buf) / max(_n0, 1):.2f}；半权质量 ×{_wt / max(_n0, 1):.2f}）")

    # ---- B. 每日正样本分布 / Top10 海啸日 / P95 封顶前后权重占比 ----
    _emit("\n## B. 每日正样本数分布与海啸日（P95 软封顶 = max(300, P95)）")
    _emit("| 变体 | min | P10 | P25 | P50 | P75 | P90 | P95 | P99 | max | 零正样本日占比 |")
    _emit("|---|---|---|---|---|---|---|---|---|---|---|")
    _specs = []
    for _name, _pos, _w in (
            ("L0", core0, np.ones(len(core0), dtype=float)),
            ("L1", core1, np.ones(len(core1), dtype=float)),
            ("L2", core2, np.ones(len(core2), dtype=float)),
            ("L3", core3, np.ones(len(core3), dtype=float)),
            ("L4", core0 | buf4,
             np.where(core0, 1.0, np.where(buf4, BUF_POS_WEIGHT, 0.0))),
            ("L5", core5, np.ones(len(core5), dtype=float))):
        _pos_b = _pos[base]  # _pos 为全行掩码；day_codes 是 valid 行域 → 取子集
        _cnt = np.bincount(day_codes[_pos_b], minlength=nd).astype(float)
        _q = np.percentile(_cnt, [10, 25, 50, 75, 90, 95, 99])
        _emit(f"| {_name} | {_cnt.min():.0f} | " + " | ".join(f"{v:.1f}" for v in _q)
              + f" | {_cnt.max():.0f} | {float((_cnt == 0).mean()):.1%} |")
        _specs.append((_name, _pos, _w, _cnt))
    _emit("\n### Top10 海啸日（以 L0 计数排序；缩放系数 = sqrt(cap/n)）")
    _emit("| 日期 | 当日正样本数 | cap | 缩放系数 |")
    _emit("|---|---|---|---|")
    _c0 = _specs[0][3]
    _pos_cnt = _c0[_c0 > 0]
    _cap0 = max(300.0, float(np.percentile(_pos_cnt, 95))) if len(_pos_cnt) else 300.0
    for _i in np.argsort(-_c0)[:10]:
        _n = _c0[_i]
        _sc = (float(np.sqrt(_cap0 / _n)) if _n > _cap0 else 1.0)
        _emit(f"| {pd.Timestamp(uniq_days[_i]).date()} | {_n:.0f} | {_cap0:.0f} | {_sc:.4f} |")
    _emit("\n### 海啸日（n>cap）占全库正样本权重比例：封顶前 → 封顶后")
    for _name, _pos, _w, _cnt in _specs:
        _pos_b = _pos[base]
        _pv = _cnt[_cnt > 0]
        _cap = max(300.0, float(np.percentile(_pv, 95))) if len(_pv) else 300.0
        _over = _cnt > _cap
        _scale = np.ones(nd, dtype=float)
        _scale[_over] = np.sqrt(_cap / _cnt[_over])
        _wd = np.bincount(day_codes[_pos_b], weights=_w[_pos], minlength=nd)
        _tot = float(_wd.sum())
        _before = float(_wd[_over].sum() / _tot) if _tot > 0 else 0.0
        _after_num = float((_wd[_over] * _scale[_over]).sum())
        _after_den = _after_num + float(_wd[~_over].sum())
        _after = _after_num / _after_den if _after_den > 0 else 0.0
        _emit(f"- {_name}: 封顶前 {_before:.2%} → 封顶后 {_after:.2%}"
              f"（超限日 {int(_over.sum())} 天）")

    # ---- C. 有效正类梯度质量占比（spw 选择；tail / snr 两口径并列） ----
    _spw_mode = str(getattr(args, "spw_mode", "fixed"))
    _base_spw = float(HP_GRID[0].get("scale_pos_weight", 20.0)) if HP_GRID else 20.0
    _emit("\n## C. 有效正类梯度质量占比（L0 / L4 / L5 标签，目标区间 35%~42%）")
    _emit(f"（q95_L0={_q95_core:.4f}，q95_L5={_q95_core5:.4f}，base_spw={_base_spw:.2f}，spw_mode={_spw_mode}；"
          f"正类权重口径 tail=绝对/超额涨幅 1.0~1.5x；snr=波动率调整 0.7~1.5x）")
    for _name, _pos, _w, _tmult, _smult in (
            ("L0", core0, np.ones(len(core0), dtype=float), _tail_mult, _snr_mult),
            ("L4", core0 | buf4,
             np.where(core0, 1.0, np.where(buf4, BUF_POS_WEIGHT, 0.0)), _tail_mult, _snr_mult),
            ("L5", core5, np.ones(len(core5), dtype=float), _tail_mult5, _snr_mult5)):
        _n_neg = n_valid - int(_pos.sum())
        _n_pos = int(_pos.sum())
        for _mode, _mult in (("tail", _tmult), ("snr", _smult)):
            _wpos = _w[_pos] * _mult[_pos]
            _pos_mass = float(_wpos.sum())
            _lines = []
            for _spw in (15, 20, 25, 30, 38):
                _pm = _pos_mass * _spw
                _share = _pm / (_pm + _n_neg) if (_pm + _n_neg) > 0 else 0.0
                _mark = " [OK]" if 0.35 <= _share <= 0.42 else ""
                _lines.append(f"spw={_spw}: {_share:.1%}{_mark}")
            _emit(f"- {_name}/{_mode}（n_pos={_n_pos:,}, n_neg={_n_neg:,}）："
                  + " | ".join(_lines))
            # 当折 spw_mode 的实际解析值（诊断用全样本参考；CV 中逐折重算）
            _eff_line = None
            if _spw_mode == "match_tail" and _mode != "tail":
                _tail_mass = float((_w[_pos] * _tail_mult[_pos]).sum())
                if _pos_mass > 0:
                    _spw_eff = _base_spw * (_tail_mass / _pos_mass)
                    _pm = _pos_mass * _spw_eff
                    _share = _pm / (_pm + _n_neg) if (_pm + _n_neg) > 0 else 0.0
                    _eff_line = (f"match_tail spw_eff={_spw_eff:.2f} → "
                                 f"{_share:.1%}")
            elif _spw_mode == "adaptive" and _n_pos > 0 and _n_neg > 0:
                _ratio = _n_pos / _n_neg
                _mean_w = _pos_mass / _n_pos
                _spw_eff = float(np.clip(
                    _base_spw * (SPW_ADAPT_R0 / _ratio)
                    * (SPW_ADAPT_W0 / _mean_w),
                    SPW_ADAPT_MIN, SPW_ADAPT_MAX))
                _pm = _pos_mass * _spw_eff
                _share = _pm / (_pm + _n_neg) if (_pm + _n_neg) > 0 else 0.0
                _eff_line = (f"adaptive spw_eff={_spw_eff:.2f} → "
                             f"{_share:.1%}（全样本参考）")
            if _eff_line:
                _emit(f"  - {_name}/{_mode}: {_eff_line}")

    # ---- D. 波动率调整权重诊断（验证"高波噪音 vs 低波真突破"假设） ----
    _emit("\n## D. 波动率调整正样本权重（snr = r3/(max(vol20,1.5%)*sqrt3)）")
    _emit(f"- vol20 缺失率：可交易有效样本 {float(np.mean(~np.isfinite(_vol[base]))):.2%}；"
          f"核心正样本内 {float(np.mean(~np.isfinite(_vol[core0]))):.2%}"
          "（缺失样本按中性 1.0x）")
    _csnr = _snr_core[np.isfinite(_snr_core)]
    if len(_csnr):
        _q = np.percentile(_csnr, [10, 25, 50, 75, 90])
        _emit(f"- 核心正样本 SNR 分位：P10 {_q[0]:.2f} | P25 {_q[1]:.2f} | "
              f"P50 {_q[2]:.2f} | P75 {_q[3]:.2f} | P90 {_q[4]:.2f}")
        _edges = [(-np.inf, SNR_ANCHOR_LO), (SNR_ANCHOR_LO, 1.0),
                  (1.0, 1.5), (1.5, 2.0), (2.0, SNR_ANCHOR_HI),
                  (SNR_ANCHOR_HI, np.inf)]
        _parts = []
        for _lo, _hi in _edges:
            _m = (_csnr >= _lo) if np.isinf(_hi) else ((_csnr > _lo) & (_csnr <= _hi))
            _lab = (f"<{SNR_ANCHOR_LO:.1f}σ" if _hi == SNR_ANCHOR_LO else
                    f"{_lo:.1f}~{_hi:.1f}σ" if np.isfinite(_hi) else
                    f">={SNR_ANCHOR_HI:.1f}σ")
            _parts.append(f"{_lab}: {float(_m.mean()):.1%}")
        _emit("- SNR 分布：" + " | ".join(_parts))
        _emit(f"- 核心正样本权重：tail 均值 {float(_tail_mult[core0].mean()):.3f}"
              f" / 中位 {float(np.median(_tail_mult[core0])):.3f}；"
              f"snr 均值 {float(_snr_mult[core0].mean()):.3f}"
              f" / 中位 {float(np.median(_snr_mult[core0])):.3f}"
              f" / P10 {float(np.percentile(_snr_mult[core0], 10)):.3f}"
              f" / P90 {float(np.percentile(_snr_mult[core0], 90)):.3f}")
    else:
        _emit("- 核心正样本 SNR 全缺失，无法统计")
    _emit("\n### vol20 五分位 × 核心正样本质量（高波票是否更差）")
    _emit("| vol20 五分位 | 正样本数 | 占正样本 | r3_eq 均值/中位 | r5_eq 均值/中位 | "
          "dd3 均值/中位 | SNR 中位 | snr 权重均值 |")
    _emit("|---|---|---|---|---|---|---|---|")
    _vc = _vol[core0]
    _ok_v = np.isfinite(_vc)
    if int(_ok_v.sum()) > 100:
        _qs = np.percentile(_vc[_ok_v], [20, 40, 60, 80])
        _qedges = [-np.inf, *_qs.tolist(), np.inf]
        _qidx = np.digitize(_vc, _qs, right=True)
        _core_idx = np.flatnonzero(core0)
        _snr_all_pos = np.full(len(_vc), np.nan)
        _snr_all_pos[_ok_v] = _snr_core[_ok_v]
        _w_all_pos = np.full(len(_vc), np.nan)
        _w_all_pos[_ok_v] = _w_core[_ok_v]

        def _mstat(x, m):
            v = x[m]
            v = v[np.isfinite(v)]
            if not len(v):
                return "n/a"
            return f"{np.mean(v):+.4f} / {np.median(v):+.4f}"

        for _qi in range(5):
            _m = (_qidx == _qi) & _ok_v
            _n = int(_m.sum())
            if not _n:
                continue
            _rows_full = _core_idx[_m]
            _snr_med = (float(np.median(_snr_all_pos[_m]))
                        if np.isfinite(_snr_all_pos[_m]).any() else float("nan"))
            _w_mean = (float(np.mean(_w_all_pos[_m]))
                       if np.isfinite(_w_all_pos[_m]).any() else float("nan"))
            _lo = _qedges[_qi]
            _hi = _qedges[_qi + 1]
            _lab = (f"<{_hi:.2%}" if not np.isfinite(_lo) else
                    f">={_lo:.2%}" if not np.isfinite(_hi) else
                    f"{_lo:.2%}~{_hi:.2%}")
            _emit(f"| {_lab} | {_n:,} | {_n / max(int(core0.sum()), 1):.1%} | "
                  f"{_mstat(_r3, _rows_full)} | {_mstat(_r5, _rows_full)} | "
                  f"{_mstat(_dd, _rows_full)} | {_snr_med:.2f} | {_w_mean:.3f} |")
    else:
        _emit("| n/a | 有效 vol20 核心正样本不足 | | | | | | |")
    _emit(f"\n[diag] 完成 {time.time() - t0:.0f}s")

    # 落盘 markdown
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _md_path = out_dir / "label_diagnostics.md"
        _md_path.write_text("\n".join(_md), encoding="utf-8")
        logger.info(f"[diag] 报告已写入 {_md_path}")
    except Exception as e:
        logger.warning(f"[diag] markdown 落盘失败（忽略）: {str(e)[:100]}")


# ==================== 离线融合寻优（--mode fuse_champions） ====================
def run_fuse_champions(args, out_dir: Path, logger: logging.Logger) -> None:
    """离线融合寻优（2026-10-19 定版，Owner 批准）：读取 CV 任务明细 parquet，
    无需重训（秒级），在同一 11 折 OOS 候选上对比 5 类策略：

      ① 纯 LGB 冠军（rank1 口径）；
      ② 纯 CB 冠军；
      ③ logit z-score 加权 w∈{0.1..0.9}（两侧当日横截面 z 线性加权）；
      ④ 秩平均（两侧当日名次均值，越小越好）；
      ⑤ 双核共振过滤（按 LGB 排序，要求候选进入 CB 当日头名 top-K，否则顺延；
         全部不满足则当天空仓）。
    """
    t0 = time.time()
    cv_dir = Path(args.cv_dir) if str(args.cv_dir).strip() else out_dir
    _lgb_p = cv_dir / "champion.json"
    _cb_p = cv_dir / "champion_cb.json"
    if not _lgb_p.exists() or not _cb_p.exists():
        raise SystemExit(f"[FUSE] 缺少 {_lgb_p} 或 {_cb_p}（需先跑含 lgb+catboost 的 CV）")
    lgb_ch = json.loads(_lgb_p.read_text(encoding="utf-8"))
    cb_ch = json.loads(_cb_p.read_text(encoding="utf-8"))
    _parts = [pd.read_parquet(p) for p in
              sorted((cv_dir / "cv_tasks").glob("*_rows.parquet"))]
    if not _parts:
        raise SystemExit(f"[FUSE] {cv_dir}/cv_tasks 下无任务明细")
    all_df = pd.concat(_parts, ignore_index=True)
    # 2026-10-20 P0-1 修复：任务明细同时含 raw+dedup 两套，融合必须只用 dedup
    # （旧版未过滤 → 同一 (fold,date,stock) 两行、merge 后行膨胀，融合成绩失真）。
    if "universe" in all_df.columns:
        all_df = all_df[all_df["universe"] == "dedup"].copy()
    _lgb = all_df[(all_df["scheme"] == "lgb")
                  & (all_df["param"] == lgb_ch["param"])
                  & (all_df["cp"] == int(lgb_ch["cp"]))].copy()
    _cb = all_df[(all_df["scheme"] == "catboost")
                 & (all_df["param"] == cb_ch["param"])
                 & (all_df["cp"] == int(cb_ch["cp"]))].copy()
    if not len(_lgb) or not len(_cb):
        raise SystemExit("[FUSE] 冠军明细为空，检查 champion 文件与任务明细")
    logger.info(f"[FUSE] LGB 冠军 {lgb_ch['param']}@{int(lgb_ch['cp'])} "
                f"({len(_lgb):,} 行) | CB 冠军 {cb_ch['param']}@{int(cb_ch['cp'])} "
                f"({len(_cb):,} 行)")

    _l = _lgb[["fold_id", "date", "stock_code", "sector", "rank",
               "pred", "score_z_in_day", "return"]].rename(columns={
        "rank": "rank_lgb", "pred": "pred_lgb",
        "score_z_in_day": "z_lgb", "return": "ret"})
    _c = _cb[["fold_id", "date", "stock_code", "sector", "rank",
              "pred", "score_z_in_day", "return"]].rename(columns={
        "rank": "rank_cb", "pred": "pred_cb", "score_z_in_day": "z_cb",
        "return": "ret_cb", "sector": "sector_cb"})
    m = _l.merge(_c, on=["fold_id", "date", "stock_code"], how="outer")
    # 2026-10-20 P2-2 修复：CB 独有候选的 sector 先取 CB 的（旧版漏取导致全 __NA__）
    m["sector"] = m["sector"].fillna(m["sector_cb"]).fillna("__NA__")
    m["ret"] = m["ret"].fillna(m["ret_cb"])
    for _cc in ("rank_lgb", "rank_cb"):
        m[_cc] = m[_cc].fillna(9999.0)
    # 2026-10-20 P2-2 修复：缺失一侧的 z 用中性 0.0（旧版 -99.0 会把 LGB 榜首
    # 好票在 w=0.9 加权下腰斩淘汰——单侧缺失应视为"无意见"，而非极端反对）
    for _cc in ("z_lgb", "z_cb"):
        m[_cc] = m[_cc].fillna(0.0)
    for _cc in ("pred_lgb", "pred_cb"):
        m[_cc] = m[_cc].fillna(-1e18)

    def _eval(strategy: str, score, keep_mask=None) -> dict:
        d = pd.DataFrame({
            "date": m["date"].to_numpy(), "fold_id": m["fold_id"].to_numpy(),
            "sector": m["sector"].to_numpy(), "code": m["stock_code"].to_numpy(),
            "ret": m["ret"].to_numpy(), "score": np.asarray(score, dtype=float)})
        if keep_mask is not None:
            d = d[np.asarray(keep_mask, dtype=bool)]
        if not len(d):
            return {"strategy": strategy, "n_days": 0, "r1_mean": float("nan")}
        d = d.sort_values(["date", "score"], ascending=[True, False],
                          kind="stable")
        d = d.drop_duplicates(subset=["date", "sector"], keep="first")
        d["rk"] = d.groupby("date", sort=False).cumcount() + 1
        r1 = d[d["rk"] == 1]
        r12 = d[d["rk"] <= 2].groupby("date", sort=False)["ret"].mean()
        ret = r1["ret"].to_numpy(dtype=float)
        _fi = r1.groupby("fold_id")["ret"].mean()
        return {"strategy": strategy, "n_days": int(len(r1)),
                "r1_mean": float(np.mean(ret)),
                "r1_median": float(np.median(ret)),
                "r1_win": float((ret > 0).mean()),
                "r1_q10": float(np.quantile(ret, 0.10)),
                "pos_folds": int((_fi > 0).sum()), "n_folds": int(len(_fi)),
                "min_fold_mean": float(_fi.min()),
                "top2_mean": float(r12.mean())}

    _rows = [_eval("lgb_only", m["pred_lgb"].to_numpy()),
             _eval("cb_only", m["pred_cb"].to_numpy())]
    for w in np.round(np.arange(0.1, 1.0, 0.1), 2):
        _rows.append(_eval(
            f"z_blend_w{w:.1f}",
            w * m["z_lgb"].to_numpy() + (1.0 - w) * m["z_cb"].to_numpy()))
    _rows.append(_eval("rank_avg",
                       -(m["rank_lgb"].to_numpy() + m["rank_cb"].to_numpy())))
    for K in (2, 3, 5, 10):
        _rows.append(_eval(f"resonance_K{K}", m["pred_lgb"].to_numpy(),
                           keep_mask=(m["rank_cb"].to_numpy() <= K)))
    _res = pd.DataFrame(_rows)
    _res.to_csv(out_dir / "fuse_comparison.csv", index=False,
                encoding="utf-8-sig")
    logger.info("\n===== 融合寻优对比（同一 11 折 OOS 候选，rank1 口径） =====")
    logger.info("\n" + _res.to_string(index=False,
                float_format=lambda x: f"{x:.4f}"))
    _ok = _res.dropna(subset=["r1_mean"])
    if len(_ok):
        _best = _ok.sort_values("r1_mean", ascending=False).iloc[0]
        logger.info(f"[FUSE] 最优策略: {_best['strategy']} | "
                    f"r1_mean={_best['r1_mean']:.4f} | "
                    f"pos_folds={int(_best['pos_folds'])}/{int(_best['n_folds'])}")
    logger.info(f"[FUSE] 完成 {time.time() - t0:.1f}s")


# ==================== 主流程入口 ====================
def main():
    p = argparse.ArgumentParser(description="Stage15 V5.1 独立训练、CV选参与 2026 OOS 回测管线")
    p.add_argument("--mode", choices=["cv", "oos_2026", "diag_labels", "fuse_champions"], default="cv")
    p.add_argument("--features_parquet", default="results/merge_v2/lgb_ready/lgb_features.parquet")
    p.add_argument("--data_start", default=DATA_START)
    p.add_argument("--cv_end", default=CV_END)
    p.add_argument("--train_months", type=int, default=TRAIN_MONTHS)
    p.add_argument("--test_months", type=int, default=TEST_MONTHS)
    p.add_argument("--seeds", default="42,2024,777", help="CV默认3种子；oos_2026建议传8种子")
    p.add_argument("--engines", default="lgb", help="模型核: lgb (默认，严格对齐A2) 或 lgb,catboost 或 lgb,xgboost")
    p.add_argument("--enable_buffer_pos", action="store_true", default=False,
                   help="缓冲稳健正样本 4.5%%<=R3<7%%（仅 --label-v51 生效；"
                        "2026-10-19 定版默认关闭：正类率~2%%与 scale_pos_weight 30~38 匹配）")
    p.add_argument("--label_v51", action="store_true", default=False,
                   help="[兼容别名] 等价于 --label_mode v51；两者冲突时终止。"
                        "V5.1 口径=TP_real 收盘加权 + T+2 硬底线 + r5 绝对涨幅豁免")
    p.add_argument("--label_mode", choices=["classic", "v51", "excess"],
                   default=None,
                   help="标签口径：classic=经典 r3>=7%%（默认）；v51=旧 --label_v51；"
                        "excess=截面超额 (r3-r3_med_day>=6%%) 且绝对 r3>=3.5%%；"
                        "不传时按 --label_v51 兼容旧行为。评估 y_raw 始终为绝对 r3")
    p.add_argument("--weight_mode",
                   choices=["tail", "snr", "tail_snr_max", "snr_inv",
                            "highvol_up"],
                   default="tail",
                   help="核心正样本权重口径：tail=绝对涨幅尾部加权 1.0~1.5x"
                        "（默认，逐值复现历史结果）；snr=波动率调整信噪比加权 "
                        "0.7~1.5x（低波真突破重奖、高波噪音降权；负样本恒 1.0x）；"
                        "tail_snr_max=逐样本取 max(tail,snr)，不降高波赢家；"
                        "snr_inv=snr 反向，低波降权、高波加权；"
                        "highvol_up=只抬高波，低波保持 1.0x 不降权")
    p.add_argument("--spw_mode", choices=["fixed", "match_tail", "adaptive"],
                   default="fixed",
                   help="有效 scale_pos_weight 口径：fixed=沿用 HP_GRID（默认）；"
                        "match_tail=按当折正样本总质量匹配到 tail（隔离正样本内部"
                        "重分配效果）；adaptive=按当折 N+/N- 与正样本均权相对基准"
                        "动态校准并 clip 到 [5,50]")
    p.add_argument("--base_spw", type=float, default=None,
                   help="实验用：覆盖 LGB HP_GRID 的 base scale_pos_weight；"
                        "不传则使用各 pid 原值；仅影响 LGB，不改变默认配置")
    p.add_argument("--drift_cols", default=",".join(DRIFT_NORM_COLS),
                   help="按日做 MAD Z-score 的漂移列，逗号分隔；"
                        "默认 LOG_MV,AMIHUD20；实验可加 IVOL20,TURNOVER")
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--jobs", default="1",
                   help="CV 折级并行进程数：整数或 auto（按服务器核数+可用内存自动定，"
                        "防 OOM 保守估算；每 worker 内部再用 --num_threads 线程，"
                        "jobs×num_threads≤核数；结果与串行逐值一致，非 posix 平台自动退化串行）")
    p.add_argument("--test_start", default="2026-01-05", help="oos_2026 模式的测试起点")
    p.add_argument("--test_end", default="2026-10-18", help="oos_2026 模式的测试终点")
    p.add_argument("--pid", default="H15", help="oos_2026 模式使用的超参 ID")
    p.add_argument("--cp", type=int, default=350, help="oos_2026 模式使用的 Checkpoint")
    p.add_argument("--with_minute_backtest", action="store_true", help="开启 5 分钟级状态机真实回测")
    p.add_argument("--oos_rolling", action="store_true", default=False,
                   help="OOS 改为与 CV 同构的季度滚动重训（3 个月一折，21 个月训练+5 日隔离）；"
                        "默认静态单模型（旧行为）")
    p.add_argument("--veto_minute_bt", action="store_true", default=False,
                   help="分钟回测基准默认用无否决去重 preds（防 -inf 污染）；"
                        "加此开关额外跑一份风控否决版（minute_backtest_oos_veto.pkl）")
    p.add_argument("--veto_rules", default="",
                   help="risk_veto_rules.json 路径；提供后对 oos_2026 候选应用"
                        "行业去重先行+风控级联否决（Rank1 被否→顺延），输出 "
                        "[VETO-OOS] 日频级联对照表；分钟回测同步使用否决后候选")
    p.add_argument("--pids", default="",
                   help="仅跑指定超参（逗号分隔，LGB/CB 网格通用；空=全部）；"
                        "用于开机后对 H23@170/260 做 raw vs dedup 定向对拍")
    p.add_argument("--cps", default="",
                   help="仅评估指定 checkpoint（逗号分隔，与每超参专属网格取交集；"
                        "空=全部）。定向对拍示例：--pids H23 --cps 170,260")
    p.add_argument("--save_model_dir", default="",
                   help="CV 模式保存 LightGBM booster 的目录（供 SHAP 归因）")
    p.add_argument("--save_model_cp", type=int, default=0,
                   help="保存哪个 checkpoint 的 booster；0=不保存")
    p.add_argument("--mask_mode", choices=["new", "old", "fwd"], default="old",
                   help="掩码口径：new=训练 train_mask、测试 trade_mask（OOS 末段）；"
                        "old=旧 valid_mask(y_valid) 训练/测试（复现 A2）；"
                        "fwd=训练 train_mask(剔 forward_ok)、评估/测试 y_valid（阶段四新超参用）")
    p.add_argument("--min-train-rows", type=int, default=50000,
                   help="单折训练样本下限（默认50000；小样本调试可调低）")
    p.add_argument("--min-test-rows", type=int, default=5000,
                   help="单折测试样本下限（默认5000；小样本调试可调低）")
    p.add_argument("--out_dir", default="",
                   help="输出目录；断点续跑：用同一 out_dir 重跑自动跳过已完成任务"
                        "（任务级结果存 out_dir/cv_tasks/，配置指纹不符则拒绝续跑）")
    p.add_argument("--cv_dir", default="",
                   help="fuse_champions 模式读取的 CV 输出目录（默认同 --out_dir）")
    args = p.parse_args()

    # label_mode 解析（必须在 diag_labels 分支之前，供标签诊断/训练共用）：
    # 未显式传 --label_mode 时，按旧 --label_v51 兼容；显式传 v51 则自动置位
    # args.label_v51，保持下游 run_meta/日志字段一致。
    if getattr(args, "label_mode", None) is None:
        args.label_mode = "v51" if args.label_v51 else "classic"
    elif args.label_v51 and args.label_mode != "v51":
        raise SystemExit(
            "--label_v51 与 --label_mode 冲突：--label_v51 等价于 --label_mode v51")
    args.label_v51 = (args.label_mode == "v51")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (RESULTS_ROOT / f"stage15_v5_{args.mode}_{ts}")
    logger = setup_logger(out_dir / "run.log")

    if args.mode == "diag_labels":
        run_label_diagnostics(args, out_dir, logger)
        return 0
    if args.mode == "fuse_champions":
        run_fuse_champions(args, out_dir, logger)
        return 0

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    engines = [x.strip().lower() for x in args.engines.split(",") if x.strip()]
    # 2026-10-20: --cps 定向 checkpoint 过滤（raw-vs-dedup 对拍只重训 170/260 等）
    _cps = {int(x) for x in str(getattr(args, "cps", "") or "").split(",") if x.strip()}
    if _cps:
        for _hp in HP_GRID:
            _hp["checkpoints"] = [c for c in _hp.get("checkpoints", CHECKPOINTS)
                                  if c in _cps]
        HP_GRID[:] = [h for h in HP_GRID if h.get("checkpoints")]
        if "catboost" in engines:
            CB_CHECKPOINTS[:] = [c for c in CB_CHECKPOINTS if c in _cps]
            if not CB_CHECKPOINTS:
                engines = [e for e in engines if e != "catboost"]
    bundle = load_dataset_and_labels(args, logger)
    logger.info(
        f"[权重] label_mode={bundle.get('label_mode', 'classic')} "
        f"weight_mode={args.weight_mode} spw_mode={args.spw_mode} "
        + ("核心正样本 snr=weight_raw/(max(vol20,1.5%)*sqrt3) → 0.7~1.5x，负样本 1.0x"
           if args.weight_mode == "snr" else
           "核心正样本 max(tail,snr)，tail 1.0~1.5x 且不低于 tail，负样本 1.0x"
           if args.weight_mode == "tail_snr_max" else
           "核心正样本 snr_inv 反向权重，低信噪比 1.5x、高信噪比 0.7x，负样本 1.0x"
           if args.weight_mode == "snr_inv" else
           "核心正样本 highvol_up，只抬高波，低波保持 1.0x，负样本 1.0x"
           if args.weight_mode == "highvol_up" else
           "核心正样本 weight_raw 尾部 1.0~1.5x，负样本 1.0x"))

    X, y, y_raw = bundle["X"], bundle["y"], bundle["y_raw"]
    y_weight_raw = bundle.get("y_weight_raw")
    label_mode = str(bundle.get("label_mode", "classic"))
    core_pos, buf_pos = bundle["core_pos"], bundle["buf_pos"]
    vol20 = bundle.get("vol20")
    valid_mask, dates, codes, sectors = bundle["valid_mask"], bundle["dates"], bundle["codes"], bundle["sectors"]
    train_mask = bundle["train_mask"]
    trade_mask = bundle.get("trade_mask", valid_mask)
    test_mask = bundle.get("test_mask", trade_mask)
    uniq_days = np.unique(dates[trade_mask])

    if args.mode == "cv":
        folds = build_forward_folds(uniq_days, train_months=args.train_months, test_months=args.test_months,
                                    embargo_days=EMBARGO_DAYS, cv_end=args.cv_end, logger=logger)
        # 折级并行（fork 继承 X 等大数组；串行路径直调同一 worker，逐值一致）
        global _CV_CTX
        _CV_CTX = {"X": X, "y": y, "y_raw": y_raw,
                   "y_weight_raw": y_weight_raw, "label_mode": label_mode,
                   "core_pos": core_pos,
                   "buf_pos": buf_pos, "vol20": vol20,
                   "weight_mode": str(args.weight_mode),
                   "spw_mode": str(args.spw_mode),
                   "base_spw": getattr(args, "base_spw", None),
                   "valid_mask": valid_mask,
                   "train_mask": bundle["train_mask"],
                   "trade_mask": bundle["trade_mask"],
                   "test_mask": bundle["test_mask"], "dates": dates,
                   "codes": codes, "sectors": sectors, "seeds": seeds,
                   "engines": engines, "num_threads": args.num_threads,
                   "min_train": args.min_train_rows, "min_test": args.min_test_rows,
                   "collect_leaf": True,
                   "save_model_dir": getattr(args, "save_model_dir", ""),
                   "save_cp": getattr(args, "save_model_cp", None)}
        # 任务：LGB 6 组 × 11 折 +（可选）CB 4 组 × 11 折（2026-10-19 双引擎解耦）
        _only = {x.strip() for x in str(args.pids).split(",") if x.strip()}
        tasks = []
        for hp in HP_GRID:
            if _only and hp["pid"] not in _only:
                continue
            for fold in folds:
                tasks.append((len(tasks), "lgb", hp, fold))
        if "catboost" in engines:
            for hp in CB_HP_GRID:
                if _only and hp["pid"] not in _only:
                    continue
                for fold in folds:
                    tasks.append((len(tasks), "catboost", hp, fold))
        if not tasks:
            raise SystemExit(f"[CV] --pids={args.pids} 未匹配任何超参")
        # 断点续跑（2026-10-19）：任务级明细落盘 out_dir/cv_tasks/{side}_{pid}_Fold{id}
        # （_rows.parquet 全量 top30 + .json 标记），进程被杀/实例被抢占后同 --out_dir
        # 重跑自动跳过已完成任务；run_meta 配置指纹不一致则拒绝续跑。
        _tasks_dir = out_dir / "cv_tasks"
        _tasks_dir.mkdir(parents=True, exist_ok=True)
        _meta = {"features_parquet": str(args.features_parquet),
                 "n_features": len(bundle["feature_cols"]),
                 "drift_cols": list(bundle.get("drift_cols", DRIFT_NORM_COLS)),
                 "data_start": args.data_start, "cv_end": args.cv_end,
                 "train_months": args.train_months, "test_months": args.test_months,
                 "embargo_days": EMBARGO_DAYS, "seeds": seeds, "engines": engines,
                 "min_train_rows": args.min_train_rows,
                 "min_test_rows": args.min_test_rows,
                 "mask_mode": str(getattr(args, "mask_mode", "new")),
                 "label_mode": str(args.label_mode),
                 "label_v51": bool(args.label_v51),
                 "enable_buffer_pos": bool(args.enable_buffer_pos),
                 "excess_label_cfg": {"delta_min": EXCESS_DELTA_MIN,
                                      "r3_floor": EXCESS_R3_FLOOR},
                 "weight_mode": str(args.weight_mode),
                 "snr_weight_cfg": {"vol20_floor": VOL20_FLOOR,
                                    "vol20_ret_clip": VOL20_RET_CLIP,
                                    "snr_anchor": [SNR_ANCHOR_LO, SNR_ANCHOR_HI],
                                    "w_range": [SNR_W_LO, SNR_W_HI]},
                 "spw_mode": str(args.spw_mode),
                 "base_spw": getattr(args, "base_spw", None),
                 "spw_adapt_cfg": {"r0": SPW_ADAPT_R0, "w0": SPW_ADAPT_W0,
                                   "clip": [SPW_ADAPT_MIN, SPW_ADAPT_MAX]},
                 "lgb_checkpoints": {hp["pid"]: list(hp.get("checkpoints", CHECKPOINTS))
                                     for hp in HP_GRID},
                 "cb_checkpoints": list(CB_CHECKPOINTS),
                 "cb_grid": [hp["pid"] for hp in CB_HP_GRID],
                 "topk": 30, "grid_rev": "2026-10-20",
                 "universes": ["raw", "dedup"], "task_cols_rev": 2,
                 "pids": sorted(_only)}
        _meta_path = _tasks_dir / "run_meta.json"
        if _meta_path.exists():
            try:
                _old_meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            except Exception:
                _old_meta = None
            if isinstance(_old_meta, dict):
                # 兼容 2026-10-24 之前的断点目录：旧 meta 无权重/标签模式字段，
                # 等价于 tail/classic（或旧 label_v51）；注入默认值后，历史
                # 口径续跑仍与逐值一致。
                _old_meta.setdefault(
                    "label_mode",
                    "v51" if _old_meta.get("label_v51") else "classic")
                _old_meta.setdefault("label_v51",
                                     _old_meta.get("label_mode") == "v51")
                _old_meta.setdefault("excess_label_cfg", {
                    "delta_min": EXCESS_DELTA_MIN,
                    "r3_floor": EXCESS_R3_FLOOR,
                })
                _old_meta.setdefault("drift_cols", list(DRIFT_NORM_COLS))
                _old_meta.setdefault("weight_mode", "tail")
                _old_meta.setdefault("snr_weight_cfg", {
                    "vol20_floor": VOL20_FLOOR,
                    "vol20_ret_clip": VOL20_RET_CLIP,
                    "snr_anchor": [SNR_ANCHOR_LO, SNR_ANCHOR_HI],
                    "w_range": [SNR_W_LO, SNR_W_HI],
                })
                _old_meta.setdefault("spw_mode", "fixed")
                _old_meta.setdefault("base_spw", None)
                _old_meta.setdefault("spw_adapt_cfg", {
                    "r0": SPW_ADAPT_R0, "w0": SPW_ADAPT_W0,
                    "clip": [SPW_ADAPT_MIN, SPW_ADAPT_MAX],
                })
            if _old_meta != _meta:
                logger.error(f"[CV] 断点目录 {_tasks_dir} 的 run_meta 与本次配置不一致，"
                             "拒绝续跑（换 --out_dir 或删除该目录以全新开跑）")
                return 1
        else:
            _meta_path.write_text(json.dumps(_meta, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

        def _task_key(side, hp, fold):
            return f"{side}_{hp['pid']}_Fold{int(fold['fold_id']):02d}"

        def _task_path(side, hp, fold):
            return _tasks_dir / (_task_key(side, hp, fold) + ".json")

        def _rows_path(side, hp, fold):
            return _tasks_dir / (_task_key(side, hp, fold) + "_rows.parquet")

        _ROWS_COLS = ["param", "fold_id", "cp", "date", "rank", "stock_code",
                      "sector", "pred", "score_z_in_day", "return", "scheme",
                      "universe"]

        def _persist_task(side, pid, fid, rows, leaf=None):
            _df = pd.DataFrame(rows)
            if not len(_df):
                _df = pd.DataFrame({
                    "param": pd.Series(dtype="object"),
                    "fold_id": pd.Series(dtype="int64"),
                    "cp": pd.Series(dtype="int64"),
                    "date": pd.Series(dtype="object"),
                    "rank": pd.Series(dtype="int64"),
                    "stock_code": pd.Series(dtype="object"),
                    "sector": pd.Series(dtype="object"),
                    "pred": pd.Series(dtype="float64"),
                    "score_z_in_day": pd.Series(dtype="float64"),
                    "return": pd.Series(dtype="float64"),
                    "scheme": pd.Series(dtype="object"),
                    "universe": pd.Series(dtype="object")})
            _spw_eff = None
            if "spw_eff" in _df.columns and len(_df):
                _v = _df["spw_eff"].iloc[0]
                if pd.notna(_v):
                    _spw_eff = float(_v)
            _df = _df.reindex(columns=_ROWS_COLS)
            _f = _tasks_dir / f"{side}_{pid}_Fold{int(fid):02d}_rows.parquet"
            _df.to_parquet(_f, index=False)
            _mark = _tasks_dir / f"{side}_{pid}_Fold{int(fid):02d}.json"
            _tmp = _mark.with_name(_mark.name + ".tmp")
            _tmp.write_text(json.dumps({"side": side, "pid": pid,
                                        "fold_id": int(fid),
                                        "nrows": int(len(rows)),
                                        "spw_eff": _spw_eff},
                                       ensure_ascii=False), encoding="utf-8")
            _tmp.replace(_mark)
            if leaf:
                _lf = _tasks_dir / f"{side}_{pid}_Fold{int(fid):02d}_leaf.json"
                _lt = _lf.with_name(_lf.name + ".tmp")
                _lt.write_text(json.dumps(
                    {"side": side, "pid": pid, "fold_id": int(fid),
                     "leaf": {str(k): v for k, v in leaf.items()}},
                    ensure_ascii=False), encoding="utf-8")
                _lt.replace(_lf)

        _done_keys = {_task_key(t[1], t[2], t[3]) for t in tasks
                      if _task_path(t[1], t[2], t[3]).exists()
                      and _rows_path(t[1], t[2], t[3]).exists()}
        todo = [t for t in tasks
                if _task_key(t[1], t[2], t[3]) not in _done_keys]
        logger.info(f"[CV] 断点续跑：已完成 {len(_done_keys)}/{len(tasks)} 任务，"
                    f"本次待跑 {len(todo)}")
        _nj_raw = str(args.jobs).strip().lower()
        if _nj_raw in ("auto", "0", "-1"):
            _nj = auto_cv_jobs(X, engines, args.num_threads, max(1, len(todo)), logger)
        else:
            _nj = max(1, int(args.jobs or 1))
        if _nj > 1 and os.name != "posix":
            logger.warning("[CV] 非 posix 平台，--jobs 退化为串行")
            _nj = 1
        _nj = max(1, min(_nj, max(1, len(todo))))
        _t0 = time.time()
        _done = 0
        if todo:
            if _nj <= 1:
                for task in todo:
                    _ti, _side, _pid, _fid, _nrows, _rows, _leaf = _cv_worker(task)
                    _persist_task(_side, _pid, _fid, _rows, _leaf)
                    _done += 1
                    _spw_log = (f" spw_eff={float(_rows[0]['spw_eff']):.3f}"
                                if _rows and "spw_eff" in _rows[0] else "")
                    logger.info(f"[CV] {_side} {_pid} Fold{_fid:02d} 完成 rows={_nrows}"
                                f"{_spw_log} {time.time()-_t0:.0f}s")
            else:
                import concurrent.futures as _cf
                logger.info(f"[CV] 并行：{len(todo)} 个(side×超参×折)任务 × {_nj} 进程 "
                            f"(每进程 {args.num_threads} 线程) …")
                with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
                    for _ti, _side, _pid, _fid, _nrows, _rows, _leaf in _ex.map(
                            _cv_worker, todo):
                        _persist_task(_side, _pid, _fid, _rows, _leaf)
                        _done += 1
                        _spw_log = (f" spw_eff={float(_rows[0]['spw_eff']):.3f}"
                                    if _rows and "spw_eff" in _rows[0] else "")
                        logger.info(f"[CV] {_side} {_pid} Fold{_fid:02d} 完成 "
                                    f"rows={_nrows}{_spw_log} [{_done}/{len(todo)}] "
                                    f"{time.time()-_t0:.0f}s")

        # 汇总以任务文件为准（含续跑前完成的），保证任意中断后重跑结果一致
        _parts = []
        _miss_files = []
        for t in tasks:
            _p = _rows_path(t[1], t[2], t[3])
            if not _p.exists():
                _miss_files.append(_task_key(t[1], t[2], t[3]))
                continue
            try:
                _parts.append(pd.read_parquet(_p))
            except Exception:
                _miss_files.append(_task_key(t[1], t[2], t[3]))
        if _miss_files:
            logger.error(f"[CV] 汇总缺失/损坏任务文件 {len(_miss_files)} 个"
                         f"（{_miss_files[:5]}…），重跑同一命令补齐后再汇总")
            return 1
        if not _parts:
            logger.error("[CV] 无任何有效折样本产出（检查 --data_start/--cv_end/"
                         "--train_months/--min-train-rows/--min-test-rows），终止。")
            return
        all_df = pd.concat(_parts, ignore_index=True)
        _top5 = all_df[all_df["rank"] <= 5]
        if "universe" in _top5.columns:  # 旧消费方（risk_veto_train）只吃去重口径
            _top5 = _top5[_top5["universe"] == "dedup"]
        _top5.to_csv(  # 兼容旧消费方（risk_veto_train 等）
            out_dir / "cv_top5_details.csv", index=False, encoding="utf-8-sig")
        # 主榜 = 纯 LightGBM（多 seed logit 均值）；CatBoost 独立榜（2026-10-19 定版）
        main = all_df[all_df["scheme"] == "lgb"]
        if len(main):
            summary = summarize_and_pick_champion(main.to_dict("records"))
            _leaf_acc = {}
            for _lfp in sorted(_tasks_dir.glob("*_leaf.json")):
                try:
                    _lfd = json.loads(_lfp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if _lfd.get("side") != "lgb":
                    continue
                for _cps_s, _st in (_lfd.get("leaf") or {}).items():
                    _k = (_lfd.get("pid"), int(_cps_s))
                    _a = _leaf_acc.setdefault(
                        _k, {"sum": 0.0, "n": 0, "win": 0, "sum2": 0.0, "n2": 0})
                    _a["sum"] += float(_st.get("pick2_sum", 0.0))
                    _a["n"] += int(_st.get("pick2_n", 0))
                    _a["win"] += int(_st.get("pick2_win", 0))
                    _a["sum2"] += float(_st.get("top2_sum", 0.0))
                    _a["n2"] += int(_st.get("top2_n", 0))
            if _leaf_acc:
                _ldf = pd.DataFrame([{
                    "param": _k[0], "cp": int(_k[1]),
                    "pick2_leaf_orth_mean": (_a["sum"] / _a["n"]) if _a["n"] else np.nan,
                    "top2_leaf_orth_mean": (_a["sum2"] / _a["n2"]) if _a["n2"] else np.nan,
                    "pick2_leaf_win": (_a["win"] / _a["n"]) if _a["n"] else np.nan,
                    "pick2_leaf_days": int(_a["n"]),
                } for _k, _a in _leaf_acc.items()])
                summary = summary.merge(_ldf, on=["param", "cp"], how="left")
            summary.to_csv(out_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")
            logger.info("\n===== Stage15 主榜：纯 LGB（logit 均值）× 专属加密 checkpoint =====")
            logger.info("\n" + summary.head(25).to_string(index=False,
                        float_format=lambda x: f"{x:.4f}"))
            champ = summary.iloc[0].to_dict()
            (out_dir / "champion.json").write_text(
                json.dumps(champ, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[CHAMPION] 胜出配置: {champ['param']} @ cp={int(champ['cp'])} "
                        f"| plateau_score={champ['plateau_score']:.4f}")
            try:
                _write_leaf_reports(out_dir, _tasks_dir)
            except Exception as _le:
                logger.warning(f"[LEAF] 多阈值报告生成失败: {_le}")
        else:
            logger.info("[CV] 本次无 LGB 任务（纯 CatBoost 运行），跳过 LGB 主榜")
        _cb = all_df[all_df["scheme"] == "catboost"]
        if len(_cb):
            _sm = summarize_and_pick_champion(_cb.to_dict("records"))
            _sm.to_csv(out_dir / "cv_summary_catboost.csv", index=False,
                       encoding="utf-8-sig")
            logger.info("\n===== 对照榜：CatBoost 独立对称树网格 =====")
            logger.info("\n" + _sm.head(15).to_string(
                index=False, float_format=lambda x: f"{x:.4f}"))
            _champ_cb = _sm.iloc[0].to_dict()
            (out_dir / "champion_cb.json").write_text(
                json.dumps(_champ_cb, ensure_ascii=False, indent=2),
                encoding="utf-8")
            logger.info(f"[CHAMPION-CB] 胜出配置: {_champ_cb['param']} @ "
                        f"cp={int(_champ_cb['cp'])} | "
                        f"plateau_score={_champ_cb['plateau_score']:.4f}")

    else:
        # oos_2026 模式：静态单模型 或 --oos_rolling 季度滚动重训（与 CV 同构）
        hp = next(h for h in HP_GRID if h["pid"] == args.pid)
        if len(seeds) != 3:
            logger.warning(
                f"[OOS] seeds={len(seeds)} 与 CV 标定口径(3 seeds)不一致：多 seed "
                "平均会压缩预测方差，gap12/z_in_top5 等风控阈值尺度会漂移")
        _leaf_folds = []
        if args.oos_rolling:
            folds_oos = build_oos_rolling_folds(
                uniq_days, args.test_start, args.test_end, args.train_months,
                EMBARGO_DAYS, logger)
            if not folds_oos:
                raise SystemExit("[OOS-ROLL] 无有效滚动折（检查 --test_start/--data_start）")
            _ra, _pl, _dl, _cl, _yl, _xl, _sl = [], [], [], [], [], [], []
            for fo in folds_oos:
                tr_m = ((dates >= np.datetime64(fo["train_start"]))
                        & (dates <= np.datetime64(fo["train_end"])) & train_mask)
                te_m = ((dates >= np.datetime64(fo["test_start"]))
                        & (dates <= np.datetime64(fo["test_end"])) & test_mask)
                logger.info(
                    f"[OOS-ROLL] Fold{fo['fold_id']}: Train {fo['train_start']}~"
                    f"{fo['train_end']} (样本={int(tr_m.sum()):,}) | Test "
                    f"{fo['test_start']}~{fo['test_end']} (样本={int(te_m.sum()):,})")
                if (tr_m.sum() < args.min_train_rows
                        or te_m.sum() < args.min_test_rows):
                    logger.warning(f"[OOS-ROLL] Fold{fo['fold_id']} 样本不足，跳过")
                    continue
                _weight_tr = (y_weight_raw[tr_m] if y_weight_raw is not None
                              else y_raw[tr_m])
                w_tr, _spw_eff = build_train_weights(
                    core_pos[tr_m], buf_pos[tr_m], y_raw[tr_m], dates[tr_m],
                    y[tr_m], float(hp["scale_pos_weight"]),
                    weight_mode=args.weight_mode,
                    spw_mode=args.spw_mode,
                    vol20=(vol20[tr_m] if vol20 is not None else None),
                    weight_raw=_weight_tr,
                    weight_r0=(EXCESS_DELTA_MIN if label_mode == "excess"
                               else R3_CORE_MIN))
                _hp = dict(hp)
                _hp["scale_pos_weight"] = _spw_eff
                _r = train_predict_fold(
                    X[tr_m], y[tr_m], w_tr, X[te_m], dates[te_m], _hp,
                    [args.cp], seeds, ["lgb"], args.num_threads,
                    leaf_ctx={"codes": codes[te_m], "sectors": sectors[te_m],
                              "y_raw": y_raw[te_m], "cps": (args.cp,)})
                _leaf_folds.append(_r.get("leaf", {}).get(int(args.cp), {}))
                _p = _r["lgb"][args.cp]
                _ra.extend(evaluate_predictions(
                    {args.cp: _p}, dates[te_m], codes[te_m], sectors[te_m],
                    y_raw[te_m], fold_id=fo["fold_id"], pid=hp["pid"]))
                _pl.append(_p)
                _dl.append(dates[te_m])
                _cl.append(codes[te_m])
                _yl.append(y_raw[te_m])
                _xl.append(X[te_m])
                _sl.append(sectors[te_m])
            if not _ra:
                logger.error("[OOS-ROLL] 无有效样本产出，终止。")
                return
            d_rows = _ra
            _preds_oos = np.concatenate(_pl)
            _dates_oos = np.concatenate(_dl)
            _codes_oos = np.concatenate(_cl)
            _y_oos = np.concatenate(_yl)
            _X_oos = np.concatenate(_xl)
            _sectors_oos = np.concatenate(_sl)
        else:
            idx_s = int(np.searchsorted(uniq_days, np.datetime64(pd.Timestamp(args.test_start))))
            tr_end_idx = idx_s - 1 - EMBARGO_DAYS
            if tr_end_idx < 6 * 21:
                raise SystemExit(
                    f"[OOS] 训练窗不足：test_start={args.test_start} 之前仅 {max(tr_end_idx, 0)} "
                    f"个交易日（<126），检查 --data_start/--test_start")
            tr_start_idx = max(0, tr_end_idx - args.train_months * 21 + 1)
            tr_start, tr_end = uniq_days[tr_start_idx], uniq_days[tr_end_idx]

            tr_m = (dates >= tr_start) & (dates <= tr_end) & train_mask
            te_m = (dates >= np.datetime64(pd.Timestamp(args.test_start))) & (dates <= np.datetime64(pd.Timestamp(args.test_end))) & test_mask
            logger.info(f"[OOS-2026] 训练窗: {pd.Timestamp(tr_start).date()} ~ {pd.Timestamp(tr_end).date()} (样本={int(tr_m.sum()):,})")
            logger.info(f"[OOS-2026] 测试窗: {args.test_start} ~ {args.test_end} (样本={int(te_m.sum()):,})")

            _weight_tr = (y_weight_raw[tr_m] if y_weight_raw is not None
                          else y_raw[tr_m])
            w_tr, _spw_eff = build_train_weights(
                core_pos[tr_m], buf_pos[tr_m], y_raw[tr_m], dates[tr_m],
                y[tr_m], float(hp["scale_pos_weight"]),
                weight_mode=args.weight_mode,
                spw_mode=args.spw_mode,
                vol20=(vol20[tr_m] if vol20 is not None else None),
                weight_raw=_weight_tr,
                weight_r0=(EXCESS_DELTA_MIN if label_mode == "excess"
                           else R3_CORE_MIN))
            _hp = dict(hp)
            _hp["scale_pos_weight"] = _spw_eff
            _res = train_predict_fold(
                X[tr_m], y[tr_m], w_tr, X[te_m], dates[te_m],
                _hp, [args.cp], seeds, ["lgb"], args.num_threads,
                leaf_ctx={"codes": codes[te_m], "sectors": sectors[te_m],
                          "y_raw": y_raw[te_m], "cps": (args.cp,)})
            _leaf_folds.append(_res.get("leaf", {}).get(int(args.cp), {}))
            preds_by_cp = _res["lgb"]  # 主评估 = 纯 LGB（多 seed logit 均值）
            d_rows = evaluate_predictions(
                preds_by_cp, dates[te_m], codes[te_m], sectors[te_m],
                y_raw[te_m], fold_id=99, pid=hp["pid"]
            )
            _preds_oos = preds_by_cp[args.cp]
            _dates_oos = dates[te_m]
            _codes_oos = codes[te_m]
            _y_oos = y_raw[te_m]
            _X_oos = X[te_m]
            _sectors_oos = sectors[te_m]
        pd.DataFrame(d_rows).to_csv(out_dir / "oos2026_top5_details.csv", index=False, encoding="utf-8-sig")
        if not d_rows:
            logger.error("[OOS] 测试窗无有效样本产出，终止。")
            return
        summary = summarize_and_pick_champion(d_rows)
        logger.info("\n===== 2026 OOS 日频 Rank1 终验结果（Top2 列仅作诊断，不交易）=====")
        logger.info("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        _leaf_daily = []
        for _st in _leaf_folds:
            if _st:
                _leaf_daily.extend(_st.get("daily") or [])
        _lm = _leaf_daily_metrics(_leaf_daily)
        _m25 = _lm.get("thr_0.25", {})
        _mmin = _lm.get("min", {})
        if _m25.get("n", 0) or _mmin.get("n", 0):
            logger.info(
                f"[LEAF-PICK2] cp={args.cp} 0.25阈值日均={_m25.get('mean', float('nan')):.4f} "
                f"胜率={_m25.get('win', float('nan')):.1%} 天数={int(_m25.get('n', 0))} | "
                f"MinOverlap日均={_mmin.get('mean', float('nan')):.4f} "
                f"胜率={_mmin.get('win', float('nan')):.1%} 天数={int(_mmin.get('n', 0))} | "
                f"同日Rank1={_mmin.get('rank1_mean', float('nan')):.4f} "
                f"Rank2={_mmin.get('rank2_mean', float('nan')):.4f}")
            _payload = {
                "pid": hp["pid"], "cp": int(args.cp),
                "pick2_leaf_orth_mean": _m25.get("mean", float("nan")),
                "top2_leaf_orth_mean": _m25.get("top2_mean", float("nan")),
                "pick2_leaf_win": _m25.get("win", float("nan")),
                "pick2_leaf_days": int(_m25.get("n", 0)),
                "overlap_max": LEAF_PICK2_OVERLAP_MAX,
                "pick2_030_mean": _lm.get("thr_0.30", {}).get("mean", float("nan")),
                "pick2_035_mean": _lm.get("thr_0.35", {}).get("mean", float("nan")),
                "pick2_040_mean": _lm.get("thr_0.40", {}).get("mean", float("nan")),
                "pick2_min_mean": _mmin.get("mean", float("nan")),
                "pick2_min_days": int(_mmin.get("n", 0)),
                "pick2_min_win": _mmin.get("win", float("nan")),
                "top2_min_mean": _mmin.get("top2_mean", float("nan")),
                "rank1_same_day_mean": _mmin.get("rank1_mean", float("nan")),
                "rank2_same_day_mean": _mmin.get("rank2_mean", float("nan")),
                "min_mean_overlap": _mmin.get("mean_overlap", float("nan")),
                "overlap_p10": _lm["overlap_all"]["p10"],
                "overlap_p25": _lm["overlap_all"]["p25"],
                "overlap_p50": _lm["overlap_all"]["p50"],
                "overlap_p75": _lm["overlap_all"]["p75"],
            }
            (out_dir / "oos2026_leaf_pick2.json").write_text(
                json.dumps(_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if _leaf_daily:
                _rows = []
                for _d in _leaf_daily:
                    _row = {"cp": int(args.cp), "date": _d.get("date"),
                            "r1": _d.get("r1"), "r2": _d.get("r2")}
                    _mn = _d.get("min") or {}
                    _row.update({"min_rank": _mn.get("rank"), "min_ov": _mn.get("ov"),
                                 "min_ret": _mn.get("ret")})
                    for _thr in LEAF_PICK2_THRESHOLDS:
                        _key = f"{_thr:.2f}"
                        _pp = (_d.get("thr") or {}).get(_key) or {}
                        _row[f"thr_{_key}_rank"] = _pp.get("rank")
                        _row[f"thr_{_key}_ret"] = _pp.get("ret")
                    _rows.append(_row)
                pd.DataFrame(_rows).to_csv(out_dir / "oos2026_leaf_daily.csv",
                                           index=False, encoding="utf-8-sig")

        # 分钟回测基准 = 纯模型（行业去重，**无风控否决**）；否决仅用于日频对照，
        # 需要时用 --veto_minute_bt 另跑一份（2026-10-20 P0：防止 -inf 污染）。
        _preds_bt = dedup_sector_preds(_preds_oos, _codes_oos, _dates_oos,
                                       logger, tag="MINUTE-BT")
        _preds_veto = None
        if args.veto_rules:
            _preds_veto = apply_risk_veto_to_preds(
                _preds_oos, _X_oos, bundle["feature_cols"], _codes_oos,
                _dates_oos, _y_oos, bundle["price_path"],
                args.veto_rules, logger)
            # [VETO-OOS] 风控排雷后日频级联终验（独立于分钟回测；与 CV 选参
            # 同口径：行业头名顺延，被否头名剔除后取存活者 rank1..5）
            _rows_v = veto_daily_cascade(_preds_veto, _dates_oos, _codes_oos,
                                         _sectors_oos, _y_oos,
                                         hp["pid"], args.cp)
            if _rows_v:
                _sum_v = summarize_and_pick_champion(_rows_v)
                logger.info("\n===== [VETO-OOS] 风控排雷后日频级联终验"
                            "（基准=上方无否决表；rank 为存活行业头名顺延） =====")
                logger.info("\n" + _sum_v.to_string(
                    index=False, float_format=lambda x: f"{x:.4f}"))
                pd.DataFrame(_rows_v).to_csv(
                    out_dir / "oos2026_veto_cascade_details.csv",
                    index=False, encoding="utf-8-sig")
            else:
                logger.warning("[VETO] 级联终验无有效样本产出")
        if args.with_minute_backtest:
            pred_df = pd.DataFrame({
                "date": pd.to_datetime(_dates_oos),
                "stock_code": _codes_oos,
                "predicted_return": _preds_bt,
            })
            run_minute_backtest_oos(pred_df, bundle["price_path"],
                                    args.test_start, args.test_end, out_dir,
                                    logger,
                                    strategy_overrides=OOS_MINUTE_OVERRIDES)
            if _preds_veto is not None and args.veto_minute_bt:
                pred_df_v = pd.DataFrame({
                    "date": pd.to_datetime(_dates_oos),
                    "stock_code": _codes_oos,
                    "predicted_return": _preds_veto,
                })
                run_minute_backtest_oos(pred_df_v, bundle["price_path"],
                                        args.test_start, args.test_end, out_dir,
                                        logger,
                                        strategy_overrides=OOS_MINUTE_OVERRIDES,
                                        out_name="minute_backtest_oos_veto.pkl")


if __name__ == "__main__":
    main()