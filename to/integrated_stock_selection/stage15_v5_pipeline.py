#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage15 V5.1 终极独立训练、CV选参与 2026 OOS 分钟级回测一体管线
===================================================================
本次升级重点：
1. 标签升级：
   - 采用收盘加权真实均价 TP_real = 0.15*O + 0.15*H + 0.20*L + 0.50*C（防早盘瞬间冲高回落骗线）；
   - 新增 T+2（可卖首日）盈利硬底线：要求 T+2 当日收益 >= +1.0% 且收盘水上（对齐 T+2 收盘保本止损）；
   - 升级 R5 持续性条件：R5 >= R3 - 3.0% 或 R5 >= +10.0%（防误杀 3 日暴涨 +18% 后正常休整的真龙）。
2. 完整支持 3 种模式：
   - `--mode cv`       : 2021-06 ~ 2025-12 全量 11 个季度折滚动 CV 选参（双核 + 3 Seeds + 三邻域平滑）；
   - `--mode oos_2026` : 用 <=2025-12 数据训练 8-Seed 终极模型，在 2026 年纯 OOS 上跑日频 & 5分钟级状态机回测！

用法：
  # 1. 跑 11 折滚动 CV 选参 (对比 84列 vs 91列)
  python stage15_v5_pipeline.py --mode cv --features_parquet results/merge_v2/lgb_ready/lgb_features.parquet

  # 2. 跑 2026 年纯样本外 OOS + 5分钟级状态机真实回测
  python stage15_v5_pipeline.py --mode oos_2026 --features_parquet results/merge_v2/lgb_ready/lgb_features.parquet --pid H15 --cp 350 --with_minute_backtest
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

DRIFT_NORM_COLS = ("LOG_MV", "AMIHUD20")
CHECKPOINTS = [200, 300, 350, 400, 450, 500, 600]

HP_GRID = [
    {"pid": "H15S", "num_leaves": 15,  "max_depth": 5, "min_data_in_leaf": 1000, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 10.0, "reg_lambda": 15.0, "scale_pos_weight": 35.0},
    {"pid": "H12",  "num_leaves": 31,  "max_depth": 6, "min_data_in_leaf": 1000, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 10.0, "reg_lambda": 10.0, "scale_pos_weight": 25.0},
    {"pid": "H15",  "num_leaves": 31,  "max_depth": 6, "min_data_in_leaf": 1000, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 10.0, "reg_lambda": 15.0, "scale_pos_weight": 38.0},
    {"pid": "H63",  "num_leaves": 63,  "max_depth": 7, "min_data_in_leaf": 1200, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 15.0, "reg_lambda": 25.0, "scale_pos_weight": 38.0},
    {"pid": "H14",  "num_leaves": 127, "max_depth": 7, "min_data_in_leaf": 1800, "feature_fraction": 0.75, "bagging_fraction": 0.80, "reg_alpha": 20.0, "reg_lambda": 35.0, "scale_pos_weight": 38.0},
]


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


# ==================== 数据、V5.1标签与权重构建 ====================
def load_dataset_and_labels(args, logger: logging.Logger) -> dict:
    feat_path = Path(args.features_parquet)
    if not feat_path.is_absolute():
        # 相对路径按脚本所在目录解析（防 CWD 不同导致 FileNotFoundError）
        feat_path = _HERE / feat_path
    if not feat_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {feat_path}")

    t0 = time.time()
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

    g_code = pdf.groupby("stock_code", sort=False)
    pdf["listed_days"] = g_code.cumcount() + 1
    is_susp = (pdf["volume"].fillna(0) <= 0) | (pdf["amount"].fillna(0) <= 0)
    recent_susp = is_susp.astype(float).groupby(pdf["stock_code"], sort=False).rolling(10, min_periods=1).max().reset_index(level=0, drop=True) > 0

    # V5.1 升级：收盘加权真实可实现均价 TP_real = 0.15*O + 0.15*H + 0.20*L + 0.50*C
    tp_real = 0.15 * pdf["open"] + 0.15 * pdf["high"] + 0.20 * pdf["low"] + 0.50 * pdf["close"]
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
        & valid_buy
    )
    pdf["tradable_mask"] = tradable

    merged = fdf.merge(
        pdf[["stock_code", "date", "ret_t2", "close_t2_ret", "r3", "r5", "dd3", "tradable_mask"]],
        on=["stock_code", "date"], how="inner"
    )
    merged = merged.sort_values(["date", "stock_code"]).reset_index(drop=True)
    del fdf, pdf

    day_codes, _ = pd.factorize(merged["date"].to_numpy(), sort=True)
    for col in DRIFT_NORM_COLS:
        if col in merged.columns:
            merged[col] = mad_zscore_by_day(merged[col].to_numpy(dtype=float), day_codes)

    ret_t2 = merged["ret_t2"].to_numpy(dtype=np.float64)
    close_t2_ret = merged["close_t2_ret"].to_numpy(dtype=np.float64)
    r3 = merged["r3"].to_numpy(dtype=np.float64)
    r5 = merged["r5"].to_numpy(dtype=np.float64)
    dd3 = merged["dd3"].to_numpy(dtype=np.float64)

    # V5.1 持续性与 T+2 可卖首日硬底线：
    # 1. r5_ok: 第4~5天回吐不超过 3%，或第5天仍保有 >=10% 收益；
    # 2. t2_ok: 可卖首日(T+2) TP_real 收益 >= +1.0% 且收盘在水上 (>0)！
    r5_ok = np.isnan(r5) | (r5 >= (r3 - R5_TOLERANCE)) | (r5 >= 0.10)
    t2_ok = (ret_t2 >= T2_FLOOR_MIN) & (close_t2_ret > 0.0)

    core_pos = (r3 >= R3_CORE_MIN) & t2_ok & r5_ok & (dd3 >= DD3_CORE_MIN)
    if args.enable_buffer_pos:
        buf_pos = (~core_pos) & (r3 >= R3_BUF_MIN) & (r3 < R3_CORE_MIN) & t2_ok & r5_ok & (dd3 >= DD3_BUF_MIN)
    else:
        buf_pos = np.zeros_like(core_pos, dtype=bool)

    y_bin = np.where(core_pos | buf_pos, 1.0, 0.0).astype(np.float32)
    y_valid = np.isfinite(r3) & np.isfinite(dd3) & merged["tradable_mask"].to_numpy(dtype=bool)

    X = np.ascontiguousarray(merged[feature_cols].to_numpy(dtype=np.float32))
    dates = merged["date"].to_numpy()
    codes = merged["stock_code"].to_numpy()
    sectors = np.array([sw_map.get(c, "__NA__") for c in codes], dtype=object)

    logger.info(
        f"[3/4] 数据准备就绪 ({time.time()-t0:.0f}s): 可交易有效行={int(y_valid.sum()):,}/{len(merged):,} "
        f"| 核心正类率={core_pos[y_valid].mean():.2%} 缓冲正类率={buf_pos[y_valid].mean():.2%}"
    )
    return {
        "X": X, "y": y_bin, "y_raw": r3, "core_pos": core_pos, "buf_pos": buf_pos,
        "valid_mask": y_valid, "dates": dates, "codes": codes, "sectors": sectors,
        "feature_cols": feature_cols, "price_path": price_path,
    }


def compute_sample_weights(core_pos: np.ndarray, buf_pos: np.ndarray,
                           y_raw: np.ndarray,
                           dates: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(y_raw), dtype=np.float32)
    if buf_pos.any():
        w[buf_pos] = BUF_POS_WEIGHT
    if core_pos.any():
        r_pos = y_raw[core_pos]
        q95 = float(np.nanpercentile(r_pos, 95)) if len(r_pos) > 10 else 0.20
        q95 = max(q95, R3_CORE_MIN + 0.02)
        g = np.clip((y_raw - R3_CORE_MIN) / (q95 - R3_CORE_MIN), 0.0, 1.0)
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
                        scale[over] = np.sqrt(cap / npos[over]).astype(np.float32)
                        _sc = scale[day_codes]
                        w[_pos] = (w[_pos] * _sc[_pos]).astype(np.float32)
    return w


# ==================== 双核异构多 Seed 训练器 ====================
def train_predict_fold(
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
    X_te: np.ndarray, dates_te: np.ndarray,
    hp: dict, checkpoints: list[int], seeds: list[int],
    engines: list[str], num_threads: int
) -> dict[int, np.ndarray]:
    import lightgbm as lgb
    day_codes_te, _ = pd.factorize(dates_te, sort=True)

    def _daily_rank_pct(arr: np.ndarray) -> np.ndarray:
        return pd.Series(arr).groupby(day_codes_te, sort=False).rank(pct=True).to_numpy(dtype=np.float64)

    lgb_preds_by_cp = {cp: np.zeros(len(X_te), dtype=np.float64) for cp in checkpoints}
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
        for cp in checkpoints:
            p = bst.predict(X_te, num_iteration=cp)
            lgb_preds_by_cp[cp] += (0.5 * _daily_rank_pct(p) + 0.5 * p) / len(seeds)
        del bst
    try:
        dtr._handle = None  # 尽早释放数据集句柄（lightgbm 版本差异时静默跳过）
    except Exception:
        pass

    aux_pred_rank = None
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
            aux_pred_rank = 0.5 * _daily_rank_pct(p_cb) + 0.5 * p_cb
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
            aux_pred_rank = 0.5 * _daily_rank_pct(p_xb) + 0.5 * p_xb
        except ImportError:
            pass

    if aux_pred_rank is not None:
        for cp in checkpoints:
            lgb_preds_by_cp[cp] = 0.65 * lgb_preds_by_cp[cp] + 0.35 * aux_pred_rank

    return lgb_preds_by_cp


# ==================== 评估、同行业去重与分钟级回测对接 ====================
def evaluate_predictions(
    preds_by_cp: dict[int, np.ndarray], dates_te: np.ndarray, codes_te: np.ndarray,
    sectors_te: np.ndarray, y_raw_te: np.ndarray, fold_id: int, pid: str
) -> list[dict]:
    rows = []
    for cp, pred in preds_by_cp.items():
        df = pd.DataFrame({
            "date": dates_te, "code": codes_te, "sector": sectors_te,
            "pred": pred, "ret": y_raw_te
        })
        for d, g in df.groupby("date", sort=True):
            if len(g) < 30:
                continue
            p_arr = g["pred"].to_numpy(dtype=float)
            med = float(np.median(p_arr))
            mad = float(np.median(np.abs(p_arr - med)))
            scale = 1.4826 * mad if mad > 1e-12 else float(np.std(p_arr) + 1e-9)

            g_sorted = g.sort_values("pred", ascending=False)
            # 同申万二级行业去重：每个行业当天最多只取排名第 1 的龙头股
            g_dedup = g_sorted.drop_duplicates(subset=["sector"], keep="first").head(5)
            for rk_idx, r in enumerate(g_dedup.itertuples(index=False), 1):
                z_in_day = (float(r.pred) - med) / scale
                rows.append({
                    "param": pid, "fold_id": fold_id, "cp": cp,
                    "date": str(pd.Timestamp(d).date()), "rank": rk_idx,
                    "stock_code": r.code, "sector": r.sector,
                    "pred": float(r.pred), "score_z_in_day": z_in_day,
                    "return": float(r.ret),
                })
    return rows


def summarize_and_pick_champion(all_rows: list[dict], z_gate: float = 1.50) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
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
            eq = np.cumprod(1.0 + fg.sort_values("date")["return"].to_numpy(dtype=float))
            mdds.append(float((eq / np.maximum.accumulate(eq) - 1.0).min()))

        active_sub = sub[sub["score_z_in_day"] > z_gate]["return"]
        sub12 = r12[(r12["param"] == pid) & (r12["cp"] == cp)]["return"].dropna()

        raw_score = 0.50 * tail_pen_ret + 0.30 * min_fold_mean + 0.20 * float(sub12.mean())
        records.append({
            "param": pid, "cp": int(cp), "n_days": len(ret), "n_folds": len(fold_means),
            "r1_mean": float(ret.mean()), "r1_median": float(np.median(ret)),
            "r1_win": float((ret > 0).mean()), "r1_q10": float(np.quantile(ret, 0.10)),
            "mdd_min": float(min(mdds)), "pos_folds": pos_folds, "min_fold_mean": min_fold_mean,
            "top2_dedup_mean": float(sub12.mean()),
            "z_gated_mean": float(active_sub.mean()) if len(active_sub) else 0.0,
            "z_gated_active_rate": float((sub["score_z_in_day"] > z_gate).mean()),
            "raw_score": raw_score,
        })

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


def apply_risk_veto_to_preds(preds: np.ndarray, X_te: np.ndarray,
                             feature_cols: list[str], codes_te: np.ndarray,
                             dates_te: np.ndarray, y_raw_te: np.ndarray,
                             price_path: Path, rules_path: str,
                             logger: logging.Logger) -> np.ndarray:
    """OOS 分钟回测前应用风控级联否决：被否候选 predicted_return 置 -inf，
    回测引擎按 pred 降序取第一个未被否者（Rank1 被否→顺延→全否空仓）。
    特征组装与 risk_veto_train 训练期完全一致（矩阵 + 行情派生 + 日内降频缓存
    + score_z_in_day），规则引用特征缺失时严格报错（防静默失效）。"""
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8")).get("rules", [])
    if not rules:
        logger.warning(f"[VETO] {rules_path} 中无规则，跳过否决")
        return preds
    from risk_veto_train import apply_veto, derive_price_features
    fdf = pd.DataFrame(np.asarray(X_te), columns=feature_cols)
    fdf["stock_code"] = codes_te
    fdf["date"] = pd.to_datetime(dates_te)
    # score_z_in_day：与 evaluate_predictions 同日同公式（med / 1.4826*MAD）
    _t = pd.DataFrame({"date": fdf["date"], "pred": preds})
    _g = _t.groupby("date")["pred"]
    med = _g.transform("median").to_numpy(dtype=float)
    mad = _g.transform(lambda s: np.median(np.abs(s - np.median(s)))).to_numpy(dtype=float)
    std = _g.transform("std").to_numpy(dtype=float)
    scale = np.where(mad > 1e-12, 1.4826 * mad, std + 1e-9)
    fdf["score_z_in_day"] = (np.asarray(preds, dtype=float) - med) / scale
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
            from risk_veto_train import find_intraday_cache
            ip = find_intraday_cache()
            if ip is not None and Path(ip).exists():
                idf = pd.read_parquet(ip)
                idf["date"] = pd.to_datetime(idf["date"])
                idf["stock_code"] = idf["stock_code"].astype(str)
                icols = [c for c in idf.columns if c in need]
                if icols:
                    fdf = fdf.merge(idf[["stock_code", "date"] + icols],
                                    on=["stock_code", "date"], how="left")
        except Exception as e:
            logger.warning(f"[VETO] 日内降频缓存 join 失败: {str(e)[:120]}")
    missing = sorted(need - set(fdf.columns))
    if missing:
        raise SystemExit(f"[VETO] 规则引用特征缺失: {missing}（先并入对应数据源再回测）")
    veto = apply_veto(rules, fdf)
    # 排雷统计（级联口径与回测一致：每日按 pred 降序取第一个未被否者）
    _t2 = pd.DataFrame({"date": fdf["date"], "pred": np.asarray(preds, dtype=float),
                        "ret": np.asarray(y_raw_te, dtype=float), "veto": veto})
    _t2 = _t2.sort_values(["date", "pred"], ascending=[True, False], kind="stable")
    first = _t2.groupby("date").first()
    kept = _t2[~_t2["veto"]].groupby("date").first()
    delta = (kept["ret"] - first["ret"]).dropna()
    _r1v = first[first["veto"]]
    logger.info(
        f"[VETO] 规则 {len(rules)} 条 | 否决候选 {int(veto.sum())}/{len(preds)} | "
        f"Rank1 被否 {len(_r1v)}/{len(first)} 天"
        f"（被否 Rank1 实际收益均值 {(_r1v['ret'].mean() if len(_r1v) else 0.0):.4f}）"
        f" | 级联净增 Δ均值 {delta.mean():.4f}（{len(delta)} 天）")
    out = np.asarray(preds, dtype=float).copy()
    out[veto] = -np.inf
    return out


def run_minute_backtest_oos(pred_df: pd.DataFrame, price_path: Path, test_start: str,
                            test_end: str, out_dir: Path, logger: logging.Logger):
    """直接调用 strategy_backtest_top1_gate_ma5_binary 跑 5 分钟级状态机真实回测（Rank 1 & Rank 2）。"""
    try:
        from rqalpha_minute_datasource.strategy_backtest_top1_gate_ma5_binary import run_optimized_backtest
    except ImportError:
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

    bt_df = sf.merge(pred_df[["date", "stock_code", "predicted_return"]], on=["date", "stock_code"], how="left")
    bt_df["predicted_return"] = bt_df["predicted_return"].fillna(-np.inf)

    # 自动定位 5 分钟数据目录（前复权优先，见 pick_backtest_min_dir）
    min_dir = pick_backtest_min_dir(test_start, test_end, logger)

    res = run_optimized_backtest(
        prediction_data=bt_df,
        stock_counts=[1, 2],
        minute_data_path=str(min_dir),
        holding_days=3,
        enable_dynamic_rank_switch=False,
        max_concurrent_positions=2,
        fixed_position_pct=0.5,
        position_schedule=[0.5, 0.5],
    )
    if res:
        pd.to_pickle(res, out_dir / "minute_backtest_oos.pkl")
        for k, v in res.items():
            if isinstance(v, dict):
                logger.info(
                    f"[MINUTE-BT] {k}: 总收益={v.get('total_return', 0):.2f}% | "
                    f"年化={v.get('annual_return', 0)*100:.2f}% | 最大回撤={v.get('max_drawdown', 0):.2f}% | "
                    f"夏普={v.get('sharpe_ratio', 0):.2f} | 胜率={v.get('win_rate', 0):.1f}% | "
                    f"交易笔数={v.get('total_trades', 0)}"
                )


# ==================== CV 折级并行（fork 继承大数组，串行/并行同一 worker） ====================
_CV_CTX: dict | None = None


def _cv_worker(task) -> tuple:
    """CV 单 (超参, 折) 任务：训练+评估（串行/并行共用，零漂移）。
    返回 (任务序号, pid, fold_id, 评估行数, rows)。样本不足返回空 rows。"""
    t_i, hp, fold = task
    c = _CV_CTX
    dates, valid_mask = c["dates"], c["valid_mask"]
    tr_m = ((dates >= np.datetime64(fold["train_start"]))
            & (dates <= np.datetime64(fold["train_end"])) & valid_mask)
    te_m = ((dates >= np.datetime64(fold["test_start"]))
            & (dates <= np.datetime64(fold["test_end"])) & valid_mask)
    if tr_m.sum() < c["min_train"] or te_m.sum() < c["min_test"]:
        return t_i, hp["pid"], fold["fold_id"], 0, []
    w_tr = compute_sample_weights(c["core_pos"][tr_m], c["buf_pos"][tr_m],
                                  c["y_raw"][tr_m], dates[tr_m])
    preds_by_cp = train_predict_fold(
        c["X"][tr_m], c["y"][tr_m], w_tr, c["X"][te_m], dates[te_m],
        hp, CHECKPOINTS, c["seeds"], c["engines"], c["num_threads"])
    rows = evaluate_predictions(
        preds_by_cp, dates[te_m], c["codes"][te_m], c["sectors"][te_m],
        c["y_raw"][te_m], fold["fold_id"], hp["pid"])
    return t_i, hp["pid"], fold["fold_id"], len(rows), rows


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


# ==================== 主流程入口 ====================
def main():
    p = argparse.ArgumentParser(description="Stage15 V5.1 独立训练、CV选参与 2026 OOS 回测管线")
    p.add_argument("--mode", choices=["cv", "oos_2026"], default="cv")
    p.add_argument("--features_parquet", default="results/merge_v2/lgb_ready/lgb_features.parquet")
    p.add_argument("--data_start", default=DATA_START)
    p.add_argument("--cv_end", default=CV_END)
    p.add_argument("--train_months", type=int, default=TRAIN_MONTHS)
    p.add_argument("--test_months", type=int, default=TEST_MONTHS)
    p.add_argument("--seeds", default="42,2024,777", help="CV默认3种子；oos_2026建议传8种子")
    p.add_argument("--engines", default="lgb,catboost", help="模型核: lgb,catboost 或 lgb,xgboost 或 lgb")
    p.add_argument("--enable_buffer_pos", action="store_true", default=True)
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
    p.add_argument("--veto_rules", default="",
                   help="risk_veto_rules.json 路径；提供后对 oos_2026 分钟回测候选"
                        "应用风控级联否决（Rank1 被否→顺延；需配合 --with_minute_backtest）")
    p.add_argument("--min-train-rows", type=int, default=50000,
                   help="单折训练样本下限（默认50000；小样本调试可调低）")
    p.add_argument("--min-test-rows", type=int, default=5000,
                   help="单折测试样本下限（默认5000；小样本调试可调低）")
    p.add_argument("--out_dir", default="",
                   help="输出目录；断点续跑：用同一 out_dir 重跑自动跳过已完成任务"
                        "（任务级结果存 out_dir/cv_tasks/，配置指纹不符则拒绝续跑）")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (RESULTS_ROOT / f"stage15_v5_{args.mode}_{ts}")
    logger = setup_logger(out_dir / "run.log")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    engines = [x.strip().lower() for x in args.engines.split(",") if x.strip()]
    bundle = load_dataset_and_labels(args, logger)

    X, y, y_raw = bundle["X"], bundle["y"], bundle["y_raw"]
    core_pos, buf_pos = bundle["core_pos"], bundle["buf_pos"]
    valid_mask, dates, codes, sectors = bundle["valid_mask"], bundle["dates"], bundle["codes"], bundle["sectors"]
    uniq_days = np.unique(dates[valid_mask])

    if args.mode == "cv":
        folds = build_forward_folds(uniq_days, train_months=args.train_months, test_months=args.test_months,
                                    embargo_days=EMBARGO_DAYS, cv_end=args.cv_end, logger=logger)
        # 折级并行（fork 继承 X 等大数组；串行路径直调同一 worker，逐值一致）
        global _CV_CTX
        _CV_CTX = {"X": X, "y": y, "y_raw": y_raw, "core_pos": core_pos,
                   "buf_pos": buf_pos, "valid_mask": valid_mask, "dates": dates,
                   "codes": codes, "sectors": sectors, "seeds": seeds,
                   "engines": engines, "num_threads": args.num_threads,
                   "min_train": args.min_train_rows, "min_test": args.min_test_rows}
        tasks = []
        for hp in HP_GRID:
            for fold in folds:
                tasks.append((len(tasks), hp, fold))
        # 断点续跑（2026-10-19）：任务级结果落盘 out_dir/cv_tasks/{pid}_Fold{id}.json，
        # 进程被杀/实例被抢占后用同一 --out_dir 重跑即自动跳过已完成任务；
        # run_meta 配置指纹不一致则拒绝续跑（防串配置污染）。
        _tasks_dir = out_dir / "cv_tasks"
        _tasks_dir.mkdir(parents=True, exist_ok=True)
        _meta = {"features_parquet": str(args.features_parquet),
                 "n_features": len(bundle["feature_cols"]),
                 "data_start": args.data_start, "cv_end": args.cv_end,
                 "train_months": args.train_months, "test_months": args.test_months,
                 "embargo_days": EMBARGO_DAYS, "seeds": seeds, "engines": engines,
                 "min_train_rows": args.min_train_rows,
                 "min_test_rows": args.min_test_rows}
        _meta_path = _tasks_dir / "run_meta.json"
        if _meta_path.exists():
            try:
                _old_meta = json.loads(_meta_path.read_text(encoding="utf-8"))
            except Exception:
                _old_meta = None
            if _old_meta != _meta:
                logger.error(f"[CV] 断点目录 {_tasks_dir} 的 run_meta 与本次配置不一致，"
                             "拒绝续跑（换 --out_dir 或删除该目录以全新开跑）")
                return 1
        else:
            _meta_path.write_text(json.dumps(_meta, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

        def _task_key(hp, fold):
            return f"{hp['pid']}_Fold{int(fold['fold_id']):02d}"

        def _task_path(hp, fold):
            return _tasks_dir / (_task_key(hp, fold) + ".json")

        def _persist_task(pid, fid, rows):
            _f = _tasks_dir / f"{pid}_Fold{int(fid):02d}.json"
            _tmp = _f.with_name(_f.name + ".tmp")
            _tmp.write_text(json.dumps({"pid": pid, "fold_id": int(fid),
                                        "nrows": len(rows), "rows": rows},
                                       ensure_ascii=False, default=float),
                            encoding="utf-8")
            _tmp.replace(_f)

        _done_keys = {_task_key(t[1], t[2]) for t in tasks
                      if _task_path(t[1], t[2]).exists()}
        todo = [t for t in tasks if _task_key(t[1], t[2]) not in _done_keys]
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
                    _ti, _pid, _fid, _nrows, _rows = _cv_worker(task)
                    _persist_task(_pid, _fid, _rows)
                    _done += 1
                    logger.info(f"[CV] {_pid} Fold{_fid:02d} 完成 rows={_nrows} "
                                f"{time.time()-_t0:.0f}s")
            else:
                import concurrent.futures as _cf
                logger.info(f"[CV] 并行：{len(todo)} 个(超参×折)任务 × {_nj} 进程 "
                            f"(每进程 {args.num_threads} 线程) …")
                with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
                    for _ti, _pid, _fid, _nrows, _rows in _ex.map(_cv_worker, todo):
                        _persist_task(_pid, _fid, _rows)
                        _done += 1
                        logger.info(f"[CV] {_pid} Fold{_fid:02d} 完成 rows={_nrows} "
                                    f"[{_done}/{len(todo)}] {time.time()-_t0:.0f}s")

        # 汇总以任务文件为准（含续跑前完成的），保证任意中断后重跑结果一致
        all_rows = []
        _miss_files = []
        for t in tasks:
            _p = _task_path(t[1], t[2])
            if not _p.exists():
                _miss_files.append(_task_key(t[1], t[2]))
                continue
            try:
                all_rows.extend(json.loads(_p.read_text(encoding="utf-8")).get("rows", []))
            except Exception:
                _miss_files.append(_task_key(t[1], t[2]))
        if _miss_files:
            logger.error(f"[CV] 汇总缺失/损坏任务文件 {len(_miss_files)} 个"
                         f"（{_miss_files[:5]}…），重跑同一命令补齐后再汇总")
            return 1
        pd.DataFrame(all_rows).to_csv(out_dir / "cv_top5_details.csv", index=False, encoding="utf-8-sig")
        if not all_rows:
            logger.error("[CV] 无任何有效折样本产出（检查 --data_start/--cv_end/--train_months/"
                         "--min-train-rows/--min-test-rows），终止。")
            return
        summary = summarize_and_pick_champion(all_rows)
        summary.to_csv(out_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")
        logger.info("\n===== Stage15 V5.1 跨折平台平滑排行榜 (11折) =====")
        logger.info("\n" + summary.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        champ = summary.iloc[0].to_dict()
        (out_dir / "champion.json").write_text(json.dumps(champ, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[CHAMPION] 胜出配置: {champ['param']} @ cp={int(champ['cp'])} | plateau_score={champ['plateau_score']:.4f}")

    else:
        # oos_2026 模式：在 test_start 前倒推训练窗（严格 <= 2025-12），评估 2026 OOS + 分钟回测
        hp = next(h for h in HP_GRID if h["pid"] == args.pid)
        idx_s = int(np.searchsorted(uniq_days, np.datetime64(pd.Timestamp(args.test_start))))
        tr_end_idx = idx_s - 1 - EMBARGO_DAYS
        if tr_end_idx < 6 * 21:
            raise SystemExit(
                f"[OOS] 训练窗不足：test_start={args.test_start} 之前仅 {max(tr_end_idx, 0)} "
                f"个交易日（<126），检查 --data_start/--test_start")
        tr_start_idx = max(0, tr_end_idx - args.train_months * 21 + 1)
        tr_start, tr_end = uniq_days[tr_start_idx], uniq_days[tr_end_idx]

        tr_m = (dates >= tr_start) & (dates <= tr_end) & valid_mask
        te_m = (dates >= np.datetime64(pd.Timestamp(args.test_start))) & (dates <= np.datetime64(pd.Timestamp(args.test_end))) & valid_mask
        logger.info(f"[OOS-2026] 训练窗: {pd.Timestamp(tr_start).date()} ~ {pd.Timestamp(tr_end).date()} (样本={int(tr_m.sum()):,})")
        logger.info(f"[OOS-2026] 测试窗: {args.test_start} ~ {args.test_end} (样本={int(te_m.sum()):,})")

        w_tr = compute_sample_weights(core_pos[tr_m], buf_pos[tr_m], y_raw[tr_m],
                                      dates[tr_m])
        preds_by_cp = train_predict_fold(
            X[tr_m], y[tr_m], w_tr, X[te_m], dates[te_m],
            hp, [args.cp], seeds, engines, args.num_threads
        )
        d_rows = evaluate_predictions(
            preds_by_cp, dates[te_m], codes[te_m], sectors[te_m],
            y_raw[te_m], fold_id=99, pid=hp["pid"]
        )
        pd.DataFrame(d_rows).to_csv(out_dir / "oos2026_top5_details.csv", index=False, encoding="utf-8-sig")
        if not d_rows:
            logger.error("[OOS] 测试窗无有效样本产出，终止。")
            return
        summary = summarize_and_pick_champion(d_rows)
        logger.info("\n===== 2026 OOS 日频 Rank1 & 同行业去重 Top2 终验结果 =====")
        logger.info("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if args.veto_rules and not args.with_minute_backtest:
            logger.warning("[VETO] --veto_rules 已提供但未开 --with_minute_backtest，否决未生效")
        if args.with_minute_backtest:
            _preds_oos = preds_by_cp[args.cp]
            if args.veto_rules:
                _preds_oos = apply_risk_veto_to_preds(
                    _preds_oos, X[te_m], bundle["feature_cols"], codes[te_m],
                    dates[te_m], y_raw[te_m], bundle["price_path"],
                    args.veto_rules, logger)
            pred_df = pd.DataFrame({
                "date": pd.to_datetime(dates[te_m]),
                "stock_code": codes[te_m],
                "predicted_return": _preds_oos,
            })
            run_minute_backtest_oos(pred_df, bundle["price_path"], args.test_start, args.test_end, out_dir, logger)


if __name__ == "__main__":
    main()