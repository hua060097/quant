"""Audit candidate three_rule_v3 additions against the 185-day walkforward sample.

Candidates:
1. regime_aware_reversal_alpha high AND shadow_dominance_pct high
2. runup_ma5_exhaustion high AND volume_price_exhaustion high
3. low score_gap_1_2 only, then intraday_clv high OR sector_broken_resilience_alpha high OR momentum_peak_decay_3d high

The script replays the current three_rule_v3, checks actual trades and shadow opportunities
separately and combined, and scans distribution-based thresholds.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"

METRICS_FILE = RESULTS / "walkforward_metrics_20260601_024558.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260601_024558.json"
MODEL_FILE = RESULTS / "lightgbm_model_window3_20260427_231053.pkl"


CURRENT_RULE_COLS = [
    "runup_ma5_exhaustion",
    "cpv_factor",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
    "sector_leadership_persistence_v3",
    "return_20d_intra_z",
    "market_liquidity_change",
    "ulcer_index",
    "momentum_peak_decay_3d",
    "intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2",
    "market_advance_decline_ratio",
    "limit_down_3d",
    "top10_10d_strength_level_v3",
    "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d",
    "kdj_j_9",
    "score_z_in_day",
    "reg_severe_up_10d_ratio_pct",
    "price_ma5_ratio_raw",
    "ma_convergence_score",
    "ma5_pullback_depth_10d",
    "ret_3d_vs_10d_accel",
]

EXTRA_RULE_COLS = [
    "regime_aware_reversal_alpha",
    "shadow_dominance_pct",
    "volume_price_exhaustion",
    "intraday_clv",
    "sector_broken_resilience_alpha",
]

MODEL_FEATURE_EXTRAS = ["date", "stock_code"]

RULE_NAMES = [f"rule{i}" for i in range(1, 13)]


def _load_metrics_records(path: Path, key: str) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "all_results" in obj[0]:
            records = obj[0]["all_results"]["1"][key]
        else:
            records = obj
    elif isinstance(obj, dict):
        records = obj.get(key, obj)
    else:
        records = []
    return pd.DataFrame(records)


def load_actual_trades() -> pd.DataFrame:
    trades = _load_metrics_records(METRICS_FILE, "trades").copy()
    trades["source"] = "actual"
    trades["buy_date"] = pd.to_datetime(trades["buy_datetime"].astype(str).str[:10])
    for col in ["actual_return", "profit_loss", "profit_loss_pct", "predicted_return"]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")
    return trades


def load_shadow_trades() -> pd.DataFrame:
    shadow = pd.DataFrame(json.loads(SHADOW_FILE.read_text(encoding="utf-8"))).copy()
    shadow["source"] = "shadow"
    buy_date_col = shadow.get("shadow_attempt_buy_date", shadow.get("buy_datetime"))
    shadow["buy_date"] = pd.to_datetime(buy_date_col.astype(str).str[:10])
    for col in ["actual_return", "profit_loss", "profit_loss_pct", "predicted_return"]:
        if col in shadow.columns:
            shadow[col] = pd.to_numeric(shadow[col], errors="coerce")
    return shadow


def load_model() -> tuple[object, list[str]]:
    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_names = list(bundle.get("feature_names") or [])
    if not feature_names:
        raise ValueError(f"模型文件缺少 feature_names: {MODEL_FILE}")
    return model, feature_names


def load_factor_frame(columns: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    cols = ["date", "stock_code"] + [c for c in columns if c not in {"date", "stock_code"}]
    cols = list(dict.fromkeys(cols))
    available = set(pq.ParquetFile(FACTORS).schema.names)
    cols = [c for c in cols if c in available]
    data = pd.read_parquet(FACTORS, columns=cols)
    data["date"] = pd.to_datetime(data["date"])
    data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))].copy()
    data = data.sort_values(["date", "stock_code"]).reset_index(drop=True)
    return data


def _previous_trade_date_series(date_series: pd.Series, available_dates: np.ndarray) -> pd.Series:
    prev_dates = []
    for d in pd.to_datetime(date_series):
        pos = np.searchsorted(available_dates, np.datetime64(pd.Timestamp(d).normalize()), side="left") - 1
        prev_dates.append(pd.Timestamp(available_dates[pos]) if pos >= 0 else pd.NaT)
    return pd.Series(prev_dates, index=date_series.index)


def attach_prev_snapshot(events: pd.DataFrame, factors: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    events = events.copy()
    trading_days = np.array(sorted(pd.to_datetime(factors["date"].dropna().unique())))
    events["factor_date"] = _previous_trade_date_series(events["buy_date"], trading_days)

    snap_cols = [c for c in cols if c in factors.columns]
    snap = factors[["date", "stock_code"] + snap_cols].copy()
    snap = snap.rename(columns={"date": "factor_date"})
    snap = snap.rename(columns={c: f"{c}__snap" for c in snap_cols})
    out = events.merge(snap, on=["factor_date", "stock_code"], how="left")
    for c in snap_cols:
        snap_c = f"{c}__snap"
        if c in out.columns:
            out[c] = pd.to_numeric(out[snap_c], errors="coerce").combine_first(pd.to_numeric(out[c], errors="coerce"))
        else:
            out[c] = pd.to_numeric(out[snap_c], errors="coerce")
        if snap_c in out.columns:
            out = out.drop(columns=[snap_c])
    return out


def z(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


def current_three_rule_v3(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rule1"] = (z(df, "runup_ma5_exhaustion") > 0.8) & (z(df, "cpv_factor") <= -0.4)
    out["rule2"] = (
        ((z(df, "prev_day_blowoff_risk") >= 2.0) & (z(df, "sector_momentum_accel_rank") <= 0.30))
        | ((z(df, "prev_day_blowoff_risk") > 5.0) & (z(df, "sector_momentum_accel_rank") <= 0.50))
        | ((z(df, "prev_day_blowoff_risk") > 1.0) & (z(df, "sector_momentum_accel_rank") <= 0.15))
    )
    out["rule3"] = (z(df, "sector_leadership_persistence_v3") <= -0.18) & (z(df, "return_20d_intra_z") <= 0.40)
    out["rule4"] = (z(df, "market_liquidity_change") < 0.94) & (
        (z(df, "ulcer_index") < -0.80)
        | (z(df, "momentum_peak_decay_3d") > 1.0)
        | (z(df, "intra_sector_momentum_rank") < 0.67)
    )
    out["rule5"] = (z(df, "mkt_dispersion_shock_v2") < 0.98) & (z(df, "market_advance_decline_ratio") < 0.38)
    out["rule6"] = z(df, "limit_down_3d") == 1
    out["rule7"] = (z(df, "top10_10d_strength_level_v3") >= 0.70) & (z(df, "buy_sm_amount_rate") <= -1.25)
    out["rule8"] = (z(df, "prev_non_first_limit_up_performance_percentile_20d") >= 0.80) & (z(df, "kdj_j_9") <= 0.75)
    out["rule9"] = z(df, "score_z_in_day") <= 1.50
    out["rule10"] = (z(df, "reg_severe_up_10d_ratio_pct") >= 0.80) & (z(df, "market_advance_decline_ratio") <= 0.45)
    out["rule11"] = (z(df, "price_ma5_ratio_raw") >= 9.0) & (z(df, "ma_convergence_score") >= 0.20)
    out["rule12"] = (z(df, "ma5_pullback_depth_10d") <= -0.05) & (z(df, "ret_3d_vs_10d_accel") >= 6.0)
    out["current_v3"] = out.any(axis=1)
    return out.fillna(False)


def bucket_stats(df: pd.DataFrame) -> dict:
    ret = pd.to_numeric(df.get("actual_return"), errors="coerce")
    return {
        "n": int(len(df)),
        "big_loss_le_-4pct": int((ret <= -0.04).sum()),
        "small_loss": int(((ret > -0.04) & (ret < 0)).sum()),
        "small_gain_3_8pct": int(((ret >= 0.03) & (ret < 0.08)).sum()),
        "big_gain_ge_8pct": int((ret >= 0.08).sum()),
        "big_gain_ge_10pct": int((ret >= 0.10).sum()),
        "mean_return": float(ret.mean()) if len(df) else np.nan,
    }


def overlap_matrix(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    names = list(masks.keys())
    rows = []
    for a in names:
        for b in names:
            inter = (masks[a] & masks[b]).sum()
            denom = masks[a].sum()
            rows.append({
                "rule_a": a,
                "rule_b": b,
                "intersection": int(inter),
                "rule_a_hits": int(masks[a].sum()),
                "overlap_rate_a": float(inter / denom) if denom else np.nan,
            })
    return pd.DataFrame(rows)


def quantile_thresholds(series: pd.Series, qs: list[float]) -> list[tuple[float, float]]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return []
    return [(q, float(s.quantile(q))) for q in qs]


def safe_auc_high_value(values: pd.Series, labels: pd.Series) -> float:
    try:
        x = pd.to_numeric(values, errors="coerce")
        y = pd.to_numeric(labels, errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask].astype(float)
        y = y[mask].astype(int)
        if len(x) == 0 or len(np.unique(y)) < 2:
            return np.nan
        pos = y.to_numpy() == 1
        n_pos = int(pos.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return np.nan
        ranks = pd.Series(x.to_numpy()).rank(method="average").to_numpy()
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    except Exception:
        return np.nan


def safe_ks(a: pd.Series, b: pd.Series) -> float:
    try:
        aa = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
        bb = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
        if len(aa) == 0 or len(bb) == 0:
            return np.nan
        from scipy.stats import ks_2samp

        return float(ks_2samp(aa, bb, alternative="two-sided", mode="auto").statistic)
    except Exception:
        return np.nan


def summarize_mask(df: pd.DataFrame, mask: pd.Series, *, big_loss=-0.04, big_gain=0.08) -> dict:
    mask = pd.Series(mask, index=df.index).fillna(False)
    sel = df[mask].copy()
    ret = pd.to_numeric(sel.get("actual_return"), errors="coerce")
    stats = bucket_stats(sel)
    out = {
        **stats,
        "big_loss_rate": float((ret <= big_loss).mean()) if len(sel) else np.nan,
        "big_gain_rate": float((ret >= big_gain).mean()) if len(sel) else np.nan,
    }
    return out


def score_gap_panel(model: object, factors: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=factors.index)
    for col in feature_names:
        if col in factors.columns:
            frame[col] = pd.to_numeric(factors[col], errors="coerce")
        else:
            frame[col] = 0.0
    X = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    best_iter = getattr(model, "best_iteration", None)
    pred = model.predict(X, num_iteration=best_iter) if best_iter else model.predict(X)
    df = factors.copy()
    df["pred_score"] = np.asarray(pred, dtype=float)
    top_rows = []
    for d, group in df.groupby("date", sort=True):
        sg = group.sort_values(["pred_score", "stock_code"], ascending=[False, True]).reset_index(drop=True)
        row = sg.iloc[0].copy()
        top2 = float(sg.iloc[1]["pred_score"]) if len(sg) > 1 else np.nan
        top3 = float(sg.iloc[2]["pred_score"]) if len(sg) > 2 else np.nan
        top5 = float(sg.iloc[4]["pred_score"]) if len(sg) > 4 else np.nan
        row["top2_score"] = top2
        row["top3_score"] = top3
        row["top5_score"] = top5
        row["score_gap_1_2"] = float(row["pred_score"] - top2) if pd.notna(top2) else np.nan
        row["score_gap_1_3"] = float(row["pred_score"] - top3) if pd.notna(top3) else np.nan
        row["score_gap_1_5"] = float(row["pred_score"] - top5) if pd.notna(top5) else np.nan
        top_rows.append(row)
    return pd.DataFrame(top_rows)


def scan_two_factor_rule(df: pd.DataFrame, f1: str, f2: str, qlist: list[float], direction: str, rule_name: str) -> pd.DataFrame:
    rows = []
    qs1 = quantile_thresholds(df[f1], qlist)
    qs2 = quantile_thresholds(df[f2], qlist)
    for q1, t1 in qs1:
        for q2, t2 in qs2:
            if direction == "high_high":
                mask = (df[f1] >= t1) & (df[f2] >= t2)
            else:
                raise ValueError(direction)
            rows.append({
                "rule_name": rule_name,
                "factor_1": f1,
                "factor_2": f2,
                "q1": q1,
                "q2": q2,
                "t1": t1,
                "t2": t2,
                "mask_n": int(mask.sum()),
                "mask_big_loss": int(((mask) & (df["actual_return"] <= -0.04)).sum()),
                "mask_big_gain": int(((mask) & (df["actual_return"] >= 0.08)).sum()),
                "mask_mean_return": float(pd.to_numeric(df.loc[mask, "actual_return"], errors="coerce").mean()) if mask.any() else np.nan,
                "incremental_n": int((mask & ~df["current_v3"]).sum()),
                "incremental_big_loss": int((((mask & ~df["current_v3"]) & (df["actual_return"] <= -0.04))).sum()),
                "incremental_big_gain": int((((mask & ~df["current_v3"]) & (df["actual_return"] >= 0.08))).sum()),
                "current_overlap": int((mask & df["current_v3"]).sum()),
                "actual_n": int(((mask) & (df["source"] == "actual")).sum()),
                "shadow_n": int(((mask) & (df["source"] == "shadow")).sum()),
                "actual_big_loss": int((((mask) & (df["source"] == "actual") & (df["actual_return"] <= -0.04))).sum()),
                "actual_big_gain": int((((mask) & (df["source"] == "actual") & (df["actual_return"] >= 0.08))).sum()),
                "shadow_big_loss": int((((mask) & (df["source"] == "shadow") & (df["actual_return"] <= -0.04))).sum()),
                "shadow_big_gain": int((((mask) & (df["source"] == "shadow") & (df["actual_return"] >= 0.08))).sum()),
            })
    return pd.DataFrame(rows)


def scan_or_rule_with_gap(df: pd.DataFrame, low_gap_q: float, factor_list: list[str], qlist: list[float]) -> pd.DataFrame:
    gap = pd.to_numeric(df["score_gap_1_2"], errors="coerce")
    low_gap_th = float(gap.dropna().quantile(low_gap_q))
    base = gap <= low_gap_th
    if "score_gap_1_2" not in df.columns:
        raise ValueError("missing score_gap_1_2")
    rows = []
    qs_map = {f: quantile_thresholds(pd.to_numeric(df.loc[base, f], errors="coerce"), qlist) for f in factor_list}
    for q1, t1 in qs_map[factor_list[0]]:
        for q2, t2 in qs_map[factor_list[1]]:
            for q3, t3 in qs_map[factor_list[2]]:
                mask = base & ((df[factor_list[0]] >= t1) | (df[factor_list[1]] >= t2) | (df[factor_list[2]] >= t3))
                rows.append({
                    "rule_name": "rule3",
                    "low_gap_q": low_gap_q,
                    "low_gap_threshold": low_gap_th,
                    "factor_1": factor_list[0],
                    "factor_2": factor_list[1],
                    "factor_3": factor_list[2],
                    "q1": q1,
                    "q2": q2,
                    "q3": q3,
                    "t1": t1,
                    "t2": t2,
                    "t3": t3,
                    "mask_n": int(mask.sum()),
                    "mask_big_loss": int(((mask) & (df["actual_return"] <= -0.04)).sum()),
                    "mask_big_gain": int(((mask) & (df["actual_return"] >= 0.08)).sum()),
                    "mask_mean_return": float(pd.to_numeric(df.loc[mask, "actual_return"], errors="coerce").mean()) if mask.any() else np.nan,
                    "incremental_n": int((mask & ~df["current_v3"]).sum()),
                    "incremental_big_loss": int((((mask & ~df["current_v3"]) & (df["actual_return"] <= -0.04))).sum()),
                    "incremental_big_gain": int((((mask & ~df["current_v3"]) & (df["actual_return"] >= 0.08))).sum()),
                    "current_overlap": int((mask & df["current_v3"]).sum()),
                    "actual_n": int(((mask) & (df["source"] == "actual")).sum()),
                    "shadow_n": int(((mask) & (df["source"] == "shadow")).sum()),
                    "actual_big_loss": int((((mask) & (df["source"] == "actual") & (df["actual_return"] <= -0.04))).sum()),
                    "actual_big_gain": int((((mask) & (df["source"] == "actual") & (df["actual_return"] >= 0.08))).sum()),
                    "shadow_big_loss": int((((mask) & (df["source"] == "shadow") & (df["actual_return"] <= -0.04))).sum()),
                    "shadow_big_gain": int((((mask) & (df["source"] == "shadow") & (df["actual_return"] >= 0.08))).sum()),
                })
    return pd.DataFrame(rows)


def add_candidate_masks(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    out["candidate_rule1"] = (out["regime_aware_reversal_alpha"] >= thresholds["rule1"]["regime_aware_reversal_alpha"]) & (
        out["shadow_dominance_pct"] >= thresholds["rule1"]["shadow_dominance_pct"]
    )
    out["candidate_rule2"] = (out["runup_ma5_exhaustion"] >= thresholds["rule2"]["runup_ma5_exhaustion"]) & (
        out["volume_price_exhaustion"] >= thresholds["rule2"]["volume_price_exhaustion"]
    )
    low_gap_th = thresholds["rule3"]["low_gap_threshold"]
    low_gap_mask = out["score_gap_1_2"] <= low_gap_th
    out["candidate_rule3"] = low_gap_mask & (
        (out["intraday_clv"] >= thresholds["rule3"]["intraday_clv"])
        | (out["sector_broken_resilience_alpha"] >= thresholds["rule3"]["sector_broken_resilience_alpha"])
        | (out["momentum_peak_decay_3d"] >= thresholds["rule3"]["momentum_peak_decay_3d"])
    )
    out["candidate_union"] = out[["candidate_rule1", "candidate_rule2", "candidate_rule3"]].any(axis=1)
    return out


def top_hits(df: pd.DataFrame, mask_col: str, n: int = 12) -> list[dict]:
    cols = [c for c in [
        "source", "buy_date", "stock_code", "actual_return", "profit_loss_pct", "predicted_return",
        "current_v3", "candidate_rule1", "candidate_rule2", "candidate_rule3",
        "regime_aware_reversal_alpha", "shadow_dominance_pct", "runup_ma5_exhaustion", "volume_price_exhaustion",
        "score_gap_1_2", "intraday_clv", "sector_broken_resilience_alpha", "momentum_peak_decay_3d",
    ] if c in df.columns]
    return (
        df.loc[df[mask_col], cols]
        .sort_values(["actual_return", "score_gap_1_2"], ascending=[True, True], na_position="last")
        .head(n)
        .to_dict("records")
    )


def main() -> None:
    actual = load_actual_trades()
    shadow = load_shadow_trades()
    events = pd.concat([actual, shadow], ignore_index=True, sort=False)
    events["buy_date"] = pd.to_datetime(events["buy_date"])
    events["stock_code"] = events["stock_code"].astype(str)
    events["actual_return"] = pd.to_numeric(events["actual_return"], errors="coerce")
    events["profit_loss_pct"] = pd.to_numeric(events.get("profit_loss_pct"), errors="coerce")

    model, feature_names = load_model()
    start_date = str(pd.to_datetime(events["buy_date"]).min().date())
    end_date = str(pd.to_datetime(events["buy_date"]).max().date())
    load_start = (pd.to_datetime(start_date) - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    load_end = end_date

    factor_cols = list(dict.fromkeys(
        MODEL_FEATURE_EXTRAS
        + feature_names
        + CURRENT_RULE_COLS
        + EXTRA_RULE_COLS
        + ["runup_ma5_exhaustion", "volume_price_exhaustion", "intraday_clv", "sector_broken_resilience_alpha", "momentum_peak_decay_3d"]
    ))
    factors = load_factor_frame(factor_cols, load_start, load_end)

    # Build top1 panel and attach score-gap to the selected event universe.
    top1_panel = score_gap_panel(model, factors, feature_names)
    top1_panel["date"] = pd.to_datetime(top1_panel["date"])
    top1_panel = top1_panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

    # Previous-day factor snapshot for event-level analysis.
    event_cols = list(dict.fromkeys(CURRENT_RULE_COLS + EXTRA_RULE_COLS))
    events = attach_prev_snapshot(events, factors, event_cols)
    events = events.merge(
        top1_panel[["date", "stock_code", "score_gap_1_2", "score_gap_1_3", "score_gap_1_5", "pred_score"]].rename(
            columns={"stock_code": "top1_stock_code", "pred_score": "top1_pred_score"}
        ),
        left_on=["buy_date"],
        right_on=["date"],
        how="left",
        suffixes=("", "_top1"),
    )
    events = events.drop(columns=[c for c in ["date_top1"] if c in events.columns])
    events["score_gap_1_2"] = pd.to_numeric(events["score_gap_1_2"], errors="coerce")

    # Bring in rule factors.
    for col in event_cols:
        if col not in events.columns:
            events[col] = np.nan

    rule_masks = current_three_rule_v3(events)
    events = pd.concat([events, rule_masks], axis=1)
    events["current_v3"] = rule_masks["current_v3"].fillna(False)
    events["source"] = events["source"].fillna("actual")

    # Rules are scanned on the event sample itself.
    qlist = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    rule1_scan = scan_two_factor_rule(events, "regime_aware_reversal_alpha", "shadow_dominance_pct", qlist, "high_high", "rule1")
    rule2_scan = scan_two_factor_rule(events, "runup_ma5_exhaustion", "volume_price_exhaustion", qlist, "high_high", "rule2")
    rule3_scan = scan_or_rule_with_gap(events, 0.30, ["intraday_clv", "sector_broken_resilience_alpha", "momentum_peak_decay_3d"], [0.60, 0.70, 0.80, 0.85, 0.90, 0.95])

    scan_all = pd.concat([rule1_scan, rule2_scan, rule3_scan], ignore_index=True)
    if scan_all.empty:
        raise ValueError("threshold scan produced no rows")

    # Heuristic ranking: favor incremental big loss capture and penalize gain kills / overlap.
    for col in ["incremental_big_loss", "incremental_big_gain", "current_overlap", "mask_big_loss", "mask_big_gain", "actual_n", "shadow_n"]:
        scan_all[col] = pd.to_numeric(scan_all[col], errors="coerce").fillna(0).astype(int)
    scan_all["loss_gain_score"] = (
        scan_all["incremental_big_loss"]
        - 2.0 * scan_all["incremental_big_gain"]
        - 0.15 * scan_all["current_overlap"]
        + 0.10 * scan_all["mask_big_loss"]
        - 0.05 * scan_all["mask_big_gain"]
    )
    scan_all["loss_to_gain_ratio"] = scan_all["incremental_big_loss"] / (scan_all["incremental_big_gain"] + 1.0)

    thresholds = {
        "rule1": scan_all[scan_all["rule_name"] == "rule1"].sort_values(["loss_gain_score", "incremental_big_loss", "current_overlap"], ascending=[False, False, True]).iloc[0].to_dict(),
        "rule2": scan_all[scan_all["rule_name"] == "rule2"].sort_values(["loss_gain_score", "incremental_big_loss", "current_overlap"], ascending=[False, False, True]).iloc[0].to_dict(),
        "rule3": scan_all[scan_all["rule_name"] == "rule3"].sort_values(["loss_gain_score", "incremental_big_loss", "current_overlap"], ascending=[False, False, True]).iloc[0].to_dict(),
    }

    # Rebuild masks with chosen thresholds and attach to event frame.
    candidate_events = add_candidate_masks(events, {
        "rule1": {
            "regime_aware_reversal_alpha": float(thresholds["rule1"]["t1"]),
            "shadow_dominance_pct": float(thresholds["rule1"]["t2"]),
        },
        "rule2": {
            "runup_ma5_exhaustion": float(thresholds["rule2"]["t1"]),
            "volume_price_exhaustion": float(thresholds["rule2"]["t2"]),
        },
        "rule3": {
            "low_gap_threshold": float(events["score_gap_1_2"].dropna().quantile(0.30)),
            "intraday_clv": float(thresholds["rule3"]["t1"]),
            "sector_broken_resilience_alpha": float(thresholds["rule3"]["t2"]),
            "momentum_peak_decay_3d": float(thresholds["rule3"]["t3"]),
        },
    })

    candidate_events["candidate_union"] = candidate_events[["candidate_rule1", "candidate_rule2", "candidate_rule3"]].any(axis=1)
    candidate_events["current_v3"] = events["current_v3"].values

    # Summaries
    subset_specs = {
        "combined": candidate_events.index == candidate_events.index,
        "actual": candidate_events["source"] == "actual",
        "shadow": candidate_events["source"] == "shadow",
    }
    summary_rows = []
    for cand_col in ["candidate_rule1", "candidate_rule2", "candidate_rule3", "candidate_union"]:
        for subset_name, subset_mask in subset_specs.items():
            df = candidate_events[subset_mask].copy()
            cand_mask = df[cand_col].fillna(False)
            current_mask = df["current_v3"].fillna(False)
            inc_mask = cand_mask & ~current_mask
            row = {
                "candidate": cand_col,
                "subset": subset_name,
                "n": int(len(df)),
                "hit": int(cand_mask.sum()),
                "hit_big_loss": int(((cand_mask) & (df["actual_return"] <= -0.04)).sum()),
                "hit_small_loss": int(((cand_mask) & (df["actual_return"] > -0.04) & (df["actual_return"] < 0)).sum()),
                "hit_small_gain_3_8": int(((cand_mask) & (df["actual_return"] >= 0.03) & (df["actual_return"] < 0.08)).sum()),
                "hit_big_gain_8": int(((cand_mask) & (df["actual_return"] >= 0.08)).sum()),
                "hit_big_gain_10": int(((cand_mask) & (df["actual_return"] >= 0.10)).sum()),
                "incremental_hit": int(inc_mask.sum()),
                "incremental_big_loss": int(((inc_mask) & (df["actual_return"] <= -0.04)).sum()),
                "incremental_big_gain_8": int(((inc_mask) & (df["actual_return"] >= 0.08)).sum()),
                "overlap_current_v3": int((cand_mask & current_mask).sum()),
                "candidate_overlap_rate": float((cand_mask & current_mask).mean()) if len(df) else np.nan,
                "candidate_only_vs_rule": int((cand_mask & ~current_mask).sum()),
                "rule_only_vs_candidate": int((current_mask & ~cand_mask).sum()),
                "mean_return": float(pd.to_numeric(df.loc[cand_mask, "actual_return"], errors="coerce").mean()) if cand_mask.any() else np.nan,
                "median_return": float(pd.to_numeric(df.loc[cand_mask, "actual_return"], errors="coerce").median()) if cand_mask.any() else np.nan,
                "win_rate": float((pd.to_numeric(df.loc[cand_mask, "actual_return"], errors="coerce") > 0).mean()) if cand_mask.any() else np.nan,
                "big_loss_rate": float((pd.to_numeric(df.loc[cand_mask, "actual_return"], errors="coerce") <= -0.04).mean()) if cand_mask.any() else np.nan,
                "big_gain_rate": float((pd.to_numeric(df.loc[cand_mask, "actual_return"], errors="coerce") >= 0.08).mean()) if cand_mask.any() else np.nan,
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Existing rule overlap matrix on the combined sample.
    existing_masks = {name: candidate_events[name].fillna(False) for name in RULE_NAMES if name in candidate_events.columns}
    existing_masks["current_v3"] = candidate_events["current_v3"].fillna(False)
    candidate_masks = {k: candidate_events[k].fillna(False) for k in ["candidate_rule1", "candidate_rule2", "candidate_rule3", "candidate_union"]}
    overlap_df = overlap_matrix(candidate_events, {**existing_masks, **candidate_masks})

    # Event-level details for review.
    detail_cols = [
        "source", "buy_date", "stock_code", "actual_return", "profit_loss_pct",
        "current_v3", "candidate_rule1", "candidate_rule2", "candidate_rule3", "candidate_union",
        "score_gap_1_2", "score_gap_1_3", "score_gap_1_5", "top1_pred_score",
        "regime_aware_reversal_alpha", "shadow_dominance_pct",
        "runup_ma5_exhaustion", "volume_price_exhaustion",
        "intraday_clv", "sector_broken_resilience_alpha", "momentum_peak_decay_3d",
        "rule1", "rule2", "rule3", "rule4", "rule5", "rule6", "rule7", "rule8", "rule9", "rule10", "rule11", "rule12",
    ]
    detail_cols = [c for c in detail_cols if c in candidate_events.columns]
    detail_df = candidate_events[detail_cols].copy()

    # Simple candidate-rule comparison table.
    rule_compare_rows = []
    for cand in ["candidate_rule1", "candidate_rule2", "candidate_rule3"]:
        for existing in RULE_NAMES + ["current_v3"]:
            if existing not in candidate_events.columns:
                continue
            inter = int((candidate_events[cand].fillna(False) & candidate_events[existing].fillna(False)).sum())
            cand_hits = int(candidate_events[cand].fillna(False).sum())
            existing_hits = int(candidate_events[existing].fillna(False).sum())
            rule_compare_rows.append({
                "candidate": cand,
                "existing_rule": existing,
                "intersection": inter,
                "candidate_hits": cand_hits,
                "existing_hits": existing_hits,
                "candidate_overlap_rate": float(inter / cand_hits) if cand_hits else np.nan,
                "existing_overlap_rate": float(inter / existing_hits) if existing_hits else np.nan,
            })
    compare_df = pd.DataFrame(rule_compare_rows)

    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_all.to_csv(out_dir / "three_rule_v3_candidate_threshold_scan.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "three_rule_v3_candidate_summary.csv", index=False, encoding="utf-8-sig")
    overlap_df.to_csv(out_dir / "three_rule_v3_candidate_overlap_matrix.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(out_dir / "three_rule_v3_candidate_rule_overlap_compare.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(out_dir / "three_rule_v3_candidate_event_detail.csv", index=False, encoding="utf-8-sig")

    print("\n===== 候选规则阈值选择 =====")
    print(pd.DataFrame(thresholds).T[["rule_name", "factor_1", "factor_2", "factor_3", "q1", "q2", "q3", "t1", "t2", "t3", "loss_gain_score", "incremental_big_loss", "incremental_big_gain", "current_overlap"]].to_string())

    print("\n===== 候选规则汇总（actual / shadow / combined） =====")
    show_cols = ["candidate", "subset", "n", "hit", "hit_big_loss", "hit_big_gain_8", "incremental_hit", "incremental_big_loss", "incremental_big_gain_8", "overlap_current_v3", "candidate_overlap_rate", "mean_return", "median_return", "win_rate", "big_loss_rate", "big_gain_rate"]
    print(summary_df[show_cols].to_string(index=False))

    print("\n===== 候选 vs existing current_v3 / old rules 重叠率 Top30 =====")
    print(compare_df.sort_values(["candidate_overlap_rate", "existing_overlap_rate"], ascending=[False, False]).head(30).to_string(index=False))

    print("\n===== 4段/组合阈值扫描 Top30 =====")
    scan_show = [
        "rule_name", "factor_1", "factor_2", "factor_3", "q1", "q2", "q3", "t1", "t2", "t3",
        "mask_n", "mask_big_loss", "mask_big_gain", "incremental_n", "incremental_big_loss", "incremental_big_gain",
        "current_overlap", "actual_n", "shadow_n", "loss_gain_score", "loss_to_gain_ratio",
    ]
    scan_show = [c for c in scan_show if c in scan_all.columns]
    print(scan_all.sort_values(["loss_gain_score", "incremental_big_loss", "current_overlap"], ascending=[False, False, True]).head(30)[scan_show].to_string(index=False))

    # Candidate specific examples
    for cand in ["candidate_rule1", "candidate_rule2", "candidate_rule3"]:
        print(f"\n===== {cand} 命中样本（按大亏优先） =====")
        hit_examples = candidate_events[candidate_events[cand].fillna(False)].sort_values(["actual_return", "score_gap_1_2"], ascending=[True, True], na_position="last")
        cols = [c for c in [
            "source", "buy_date", "stock_code", "actual_return", "profit_loss_pct",
            "score_gap_1_2", "regime_aware_reversal_alpha", "shadow_dominance_pct",
            "runup_ma5_exhaustion", "volume_price_exhaustion", "intraday_clv",
            "sector_broken_resilience_alpha", "momentum_peak_decay_3d",
            "current_v3",
        ] if c in hit_examples.columns]
        print(hit_examples[cols].head(12).to_string(index=False))

    print("\n输出文件:")
    print(f"  阈值扫描: {out_dir / 'three_rule_v3_candidate_threshold_scan.csv'}")
    print(f"  汇总: {out_dir / 'three_rule_v3_candidate_summary.csv'}")
    print(f"  重叠矩阵: {out_dir / 'three_rule_v3_candidate_overlap_matrix.csv'}")
    print(f"  对比表: {out_dir / 'three_rule_v3_candidate_rule_overlap_compare.csv'}")
    print(f"  事件明细: {out_dir / 'three_rule_v3_candidate_event_detail.csv'}")


if __name__ == "__main__":
    main()
