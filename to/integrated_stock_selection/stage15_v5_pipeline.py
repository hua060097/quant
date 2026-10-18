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


def compute_sample_weights(core_pos: np.ndarray, buf_pos: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    w = np.ones(len(y_raw), dtype=np.float32)
    if buf_pos.any():
        w[buf_pos] = BUF_POS_WEIGHT
    if core_pos.any():
        r_pos = y_raw[core_pos]
        q95 = float(np.nanpercentile(r_pos, 95)) if len(r_pos) > 10 else 0.20
        q95 = max(q95, R3_CORE_MIN + 0.02)
        g = np.clip((y_raw - R3_CORE_MIN) / (q95 - R3_CORE_MIN), 0.0, 1.0)
        w[core_pos] = (1.0 + TAIL_WEIGHT_ALPHA * g[core_pos]).astype(np.float32)
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
    sf = pd.read_parquet(price_path)
    sf["date"] = pd.to_datetime(sf["date"]).dt.normalize()
    sf["stock_code"] = sf["stock_code"].astype(str)
    sf = sf[(sf["date"] >= pd.Timestamp(test_start)) & (sf["date"] <= pd.Timestamp(test_end))].copy()

    bt_df = sf.merge(pred_df[["date", "stock_code", "predicted_return"]], on=["date", "stock_code"], how="left")
    bt_df["predicted_return"] = bt_df["predicted_return"].fillna(-np.inf)

    # 自动定位 5 分钟数据目录
    min_dir = _HERE.parent / "local_data" / "minute"
    if not min_dir.exists():
        min_dir = _HERE / "local_data" / "minute"

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
    p.add_argument("--test_start", default="2026-01-05", help="oos_2026 模式的测试起点")
    p.add_argument("--test_end", default="2026-10-18", help="oos_2026 模式的测试终点")
    p.add_argument("--pid", default="H15", help="oos_2026 模式使用的超参 ID")
    p.add_argument("--cp", type=int, default=350, help="oos_2026 模式使用的 Checkpoint")
    p.add_argument("--with_minute_backtest", action="store_true", help="开启 5 分钟级状态机真实回测")
    p.add_argument("--min-train-rows", type=int, default=50000,
                   help="单折训练样本下限（默认50000；小样本调试可调低）")
    p.add_argument("--min-test-rows", type=int, default=5000,
                   help="单折测试样本下限（默认5000；小样本调试可调低）")
    p.add_argument("--out_dir", default="")
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
        all_rows = []
        for hp in HP_GRID:
            for fold in folds:
                tr_m = (dates >= np.datetime64(fold["train_start"])) & (dates <= np.datetime64(fold["train_end"])) & valid_mask
                te_m = (dates >= np.datetime64(fold["test_start"])) & (dates <= np.datetime64(fold["test_end"])) & valid_mask
                if tr_m.sum() < args.min_train_rows or te_m.sum() < args.min_test_rows:
                    continue
                w_tr = compute_sample_weights(core_pos[tr_m], buf_pos[tr_m], y_raw[tr_m])
                t_f = time.time()
                preds_by_cp = train_predict_fold(
                    X[tr_m], y[tr_m], w_tr, X[te_m], dates[te_m],
                    hp, CHECKPOINTS, seeds, engines, args.num_threads
                )
                f_rows = evaluate_predictions(
                    preds_by_cp, dates[te_m], codes[te_m], sectors[te_m],
                    y_raw[te_m], fold["fold_id"], hp["pid"]
                )
                all_rows.extend(f_rows)
                logger.info(f"[CV] {hp['pid']} Fold{fold['fold_id']:02d} ({fold['test_start']}~{fold['test_end']}) 完成 {time.time()-t_f:.0f}s")

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

        w_tr = compute_sample_weights(core_pos[tr_m], buf_pos[tr_m], y_raw[tr_m])
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

        if args.with_minute_backtest:
            pred_df = pd.DataFrame({
                "date": pd.to_datetime(dates[te_m]),
                "stock_code": codes[te_m],
                "predicted_return": preds_by_cp[args.cp],
            })
            run_minute_backtest_oos(pred_df, bundle["price_path"], args.test_start, args.test_end, out_dir, logger)


if __name__ == "__main__":
    main()