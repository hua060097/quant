"""Audit daily-x-percentile pair rules for three_rule_v3 candidates.

Candidate rules:
1. close_below_ma5_after_runup >= daily x% percentile
   AND regime_aware_reversal_alpha >= daily x% percentile
2. log_price_slope_20d >= daily x% percentile
   AND runup_ma5_exhaustion >= daily x% percentile

This script replays actual trades and shadow opportunities from the 185-day
walkforward sample, attaches previous-day factor snapshots, transforms the
candidate factors into same-day cross-sectional percentiles, scans x thresholds,
and reports overlap against the current three_rule_v3.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
METRICS_FILE = RESULTS / "walkforward_metrics_20260601_024558.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260601_024558.json"


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

PAIR_RULE_COLS = [
    "close_below_ma5_after_runup",
    "regime_aware_reversal_alpha",
    "log_price_slope_20d",
]


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
        (z(df, "ulcer_index") < -2.25)
        | (z(df, "momentum_peak_decay_3d") > 3.20)
        | (z(df, "intra_sector_momentum_rank") < 0.67)
    )
    out["rule5"] = (z(df, "mkt_dispersion_shock_v2") < 0.98) & (z(df, "market_advance_decline_ratio") < 0.38)
    out["rule6"] = z(df, "limit_down_3d") == 1
    out["rule7"] = (z(df, "top10_10d_strength_level_v3") >= 0.70) & (z(df, "buy_sm_amount_rate") <= -1.25)
    out["rule8"] = (z(df, "prev_non_first_limit_up_performance_percentile_20d") >= 0.80) & (z(df, "kdj_j_9") <= 0.75)
    out["rule9"] = z(df, "score_z_in_day") <= 1.50
    out["rule11"] = (z(df, "price_ma5_ratio_raw") >= 9.0) & (z(df, "ma_convergence_score") >= 0.20)
    out["rule12"] = (z(df, "ma5_pullback_depth_10d") <= -0.05) & (z(df, "ret_3d_vs_10d_accel") >= 6.0)
    out["current_v3"] = out.any(axis=1)
    return out.fillna(False)


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


def attach_prev_snapshot(events: pd.DataFrame, factors: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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


def daily_percentile_transform(factors: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = factors[["date", "stock_code"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in cols:
        if col not in factors.columns:
            continue
        series = pd.to_numeric(factors[col], errors="coerce")
        out[col] = series.groupby(factors["date"], sort=False).rank(pct=True, method="average")
    return out


def summarize_mask(df: pd.DataFrame, mask: pd.Series) -> dict:
    mask = pd.Series(mask, index=df.index).fillna(False)
    sel = df[mask].copy()
    ret = pd.to_numeric(sel.get("actual_return"), errors="coerce")
    all_ret = pd.to_numeric(df.get("actual_return"), errors="coerce")
    big_loss = ret <= -0.04
    big_gain = ret >= 0.08
    small_loss = (ret > -0.04) & (ret < 0)
    small_gain = (ret >= 0.03) & (ret < 0.08)
    return {
        "mask_n": int(mask.sum()),
        "hit_big_loss": int(big_loss.sum()),
        "hit_small_loss": int(small_loss.sum()),
        "hit_small_gain_3_8": int(small_gain.sum()),
        "hit_big_gain_8": int(big_gain.sum()),
        "hit_big_gain_10": int((ret >= 0.10).sum()),
        "mean_return": float(ret.mean()) if len(sel) else np.nan,
        "median_return": float(ret.median()) if len(sel) else np.nan,
        "win_rate": float((ret > 0).mean()) if len(sel) else np.nan,
        "big_loss_rate": float(big_loss.mean()) if len(sel) else np.nan,
        "big_gain_rate": float(big_gain.mean()) if len(sel) else np.nan,
        "big_loss_capture_rate": float(big_loss.sum() / max(int((all_ret <= -0.04).sum()), 1)),
        "big_gain_kill_rate": float(big_gain.sum() / max(int((all_ret >= 0.08).sum()), 1)),
        "small_gain_kill_rate": float(small_gain.sum() / max(int(((all_ret >= 0.03) & (all_ret < 0.08)).sum()), 1)),
    }


def scan_single_threshold_pair(df: pd.DataFrame, f1: str, f2: str, xs: list[float], rule_name: str) -> pd.DataFrame:
    rows = []
    ret = pd.to_numeric(df["actual_return"], errors="coerce")
    cur = df["current_v3"].fillna(False)
    for x in xs:
        m = (pd.to_numeric(df[f1], errors="coerce") >= x) & (pd.to_numeric(df[f2], errors="coerce") >= x)
        row = {
            "rule": rule_name,
            "x": float(x),
            "factor_1": f1,
            "factor_2": f2,
            "current_overlap": int((m & cur).sum()),
            "incremental_n": int((m & ~cur).sum()),
            "incremental_big_loss": int(((m & ~cur) & (ret <= -0.04)).sum()),
            "incremental_big_gain_8": int(((m & ~cur) & (ret >= 0.08)).sum()),
            "actual_hits": int((m & (df["source"] == "actual")).sum()),
            "shadow_hits": int((m & (df["source"] == "shadow")).sum()),
            "actual_big_loss": int((m & (df["source"] == "actual") & (ret <= -0.04)).sum()),
            "actual_big_gain_8": int((m & (df["source"] == "actual") & (ret >= 0.08)).sum()),
            "shadow_big_loss": int((m & (df["source"] == "shadow") & (ret <= -0.04)).sum()),
            "shadow_big_gain_8": int((m & (df["source"] == "shadow") & (ret >= 0.08)).sum()),
        }
        row.update(summarize_mask(df, m))
        row["score"] = float(row["big_loss_capture_rate"] - 2.0 * row["big_gain_kill_rate"] - 0.2 * row["current_overlap"] / max(row["mask_n"], 1))
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_compare(df: pd.DataFrame, cand_mask: pd.Series, cand_name: str) -> pd.DataFrame:
    rows = []
    for rule in [f"rule{i}" for i in range(1, 13)] + ["current_v3"]:
        if rule not in df.columns:
            continue
        m = df[rule].fillna(False)
        inter = int((cand_mask & m).sum())
        rows.append({
            "candidate": cand_name,
            "existing_rule": rule,
            "candidate_hits": int(cand_mask.sum()),
            "existing_hits": int(m.sum()),
            "intersection": inter,
            "candidate_overlap_rate": float(inter / max(int(cand_mask.sum()), 1)),
            "existing_overlap_rate": float(inter / max(int(m.sum()), 1)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    actual = load_actual_trades()
    shadow = load_shadow_trades()
    events = pd.concat([actual, shadow], ignore_index=True, sort=False)
    events["buy_date"] = pd.to_datetime(events["buy_date"])
    events["stock_code"] = events["stock_code"].astype(str)
    events["actual_return"] = pd.to_numeric(events["actual_return"], errors="coerce")
    events["source"] = events["source"].fillna("actual")

    factor_cols = list(dict.fromkeys(
        CURRENT_RULE_COLS
        + PAIR_RULE_COLS
    ))

    # load enough history for previous-day snapshots
    load_start = (pd.to_datetime(events["buy_date"].min()) - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    load_end = pd.to_datetime(events["buy_date"].max()).strftime("%Y-%m-%d")
    factors = load_factor_frame(factor_cols, load_start, load_end)
    raw_events = attach_prev_snapshot(events, factors, factor_cols)

    # convert candidate columns to same-day cross-sectional percentile ranks
    pct_source_cols = ["close_below_ma5_after_runup", "regime_aware_reversal_alpha", "log_price_slope_20d", "runup_ma5_exhaustion"]
    pct_frame = daily_percentile_transform(factors, pct_source_cols)
    pct_events = attach_prev_snapshot(events, pct_frame, pct_source_cols)

    # merge raw and percentile-transformed data
    analysis = raw_events.copy()
    for c in pct_source_cols:
        if c in pct_events.columns:
            analysis[f"{c}__pct"] = pd.to_numeric(pct_events[c], errors="coerce")

    # current rules
    cur_masks = current_three_rule_v3(analysis)
    analysis = pd.concat([analysis, cur_masks], axis=1)
    analysis["current_v3"] = cur_masks["current_v3"].fillna(False)

    xs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    rule_a_scan = scan_single_threshold_pair(
        analysis,
        "close_below_ma5_after_runup__pct",
        "regime_aware_reversal_alpha__pct",
        xs,
        "rule_a",
    )
    rule_b_scan = scan_single_threshold_pair(
        analysis,
        "log_price_slope_20d__pct",
        "runup_ma5_exhaustion__pct",
        xs,
        "rule_b",
    )

    def choose_best(scan: pd.DataFrame) -> dict:
        if scan.empty:
            raise ValueError("empty scan")
        zero_gain = scan[scan["hit_big_gain_8"] == 0].copy()
        if not zero_gain.empty:
            best = zero_gain.sort_values(
                ["big_loss_capture_rate", "hit_big_loss", "mask_n", "current_overlap"],
                ascending=[False, False, True, True],
                na_position="last",
            ).iloc[0]
        else:
            best = scan.sort_values(
                ["big_loss_capture_rate", "big_gain_kill_rate", "hit_big_loss", "current_overlap"],
                ascending=[False, True, False, True],
                na_position="last",
            ).iloc[0]
        return best.to_dict()

    best_a = choose_best(rule_a_scan)
    best_b = choose_best(rule_b_scan)

    mask_a = (analysis["close_below_ma5_after_runup__pct"] >= float(best_a["x"])) & (analysis["regime_aware_reversal_alpha__pct"] >= float(best_a["x"]))
    mask_b = (analysis["log_price_slope_20d__pct"] >= float(best_b["x"])) & (analysis["runup_ma5_exhaustion__pct"] >= float(best_b["x"]))

    summary_rows = []
    for name, mask in [("rule_a", mask_a), ("rule_b", mask_b), ("union", mask_a | mask_b)]:
        for subset in ["combined", "actual", "shadow"]:
            if subset == "combined":
                df = analysis
                submask = mask
            else:
                df = analysis[analysis["source"] == subset].copy()
                submask = mask.loc[df.index]
            row = {"rule": name, "subset": subset}
            row.update(summarize_mask(df, submask))
            row["current_overlap"] = int((submask & df["current_v3"].fillna(False)).sum())
            row["candidate_only_vs_current_v3"] = int((submask & ~df["current_v3"].fillna(False)).sum())
            row["current_only_vs_candidate"] = int((~submask & df["current_v3"].fillna(False)).sum())
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    compare_df = pd.concat([
        overlap_compare(analysis, mask_a.fillna(False), "rule_a_best"),
        overlap_compare(analysis, mask_b.fillna(False), "rule_b_best"),
    ], ignore_index=True)

    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    rule_a_scan.to_csv(out_dir / "three_rule_v3_daily_pct_rule_a_scan.csv", index=False, encoding="utf-8-sig")
    rule_b_scan.to_csv(out_dir / "three_rule_v3_daily_pct_rule_b_scan.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "three_rule_v3_daily_pct_summary.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(out_dir / "three_rule_v3_daily_pct_overlap_compare.csv", index=False, encoding="utf-8-sig")
    analysis_cols = [
        "source", "buy_date", "stock_code", "actual_return",
        "current_v3", "close_below_ma5_after_runup__pct", "regime_aware_reversal_alpha__pct",
        "log_price_slope_20d__pct", "runup_ma5_exhaustion__pct",
        "close_below_ma5_after_runup", "regime_aware_reversal_alpha", "log_price_slope_20d", "runup_ma5_exhaustion",
    ]
    analysis_cols = [c for c in analysis_cols if c in analysis.columns]
    analysis[analysis_cols].to_csv(out_dir / "three_rule_v3_daily_pct_event_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best_a]).to_csv(out_dir / "three_rule_v3_daily_pct_rule_a_best.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best_b]).to_csv(out_dir / "three_rule_v3_daily_pct_rule_b_best.csv", index=False, encoding="utf-8-sig")

    print("\n===== Rule A best =====")
    print(pd.DataFrame([best_a])[["x", "mask_n", "hit_big_loss", "hit_big_gain_8", "big_loss_capture_rate", "big_gain_kill_rate", "current_overlap", "score"]].to_string(index=False))
    print("\n===== Rule B best =====")
    print(pd.DataFrame([best_b])[["x", "mask_n", "hit_big_loss", "hit_big_gain_8", "big_loss_capture_rate", "big_gain_kill_rate", "current_overlap", "score"]].to_string(index=False))
    print("\n===== Summary =====")
    print(summary_df.to_string(index=False))
    print("\n===== Overlap compare Top20 =====")
    print(compare_df.sort_values(["candidate_overlap_rate", "existing_overlap_rate"], ascending=[False, False]).head(20).to_string(index=False))
    print("\nOutputs written to results/")
    return


if __name__ == "__main__":
    main()
