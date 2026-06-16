"""Replay current three_rule_v3 against actual trades and shadow opportunities.

Also checks whether the market position-control candidate
`ma5_strength_breadth_v3_hist_pct >= 0.80 and market_vols_adaptive_iv_hist_pct <= 0.70`
would already be filtered by the current three_rule_v3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"

METRICS_FILE = RESULTS / "walkforward_metrics_20260520_210454.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.json"
SCORE_ACTUAL_FILE = RESULTS / "score_z_in_day_rule_audit_actual_trades_attached.csv"
SCORE_SHADOW_FILE = RESULTS / "score_z_in_day_rule_audit_shadow_attached.csv"

STOCK_RULE_COLS = [
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
    "reg_severe_up_10d_ratio_pct",
    "price_ma5_ratio_raw",
    "ma_convergence_score",
    "ma5_pullback_depth_10d",
    "ret_3d_vs_10d_accel",
]

MARKET_RULE_COLS = [
    "market_trend_adaptive_beta",
    "market_60d_new_high_ratio",
    "ma5_strength_breadth_v3",
    "market_vols_adaptive_iv",
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
    df = pd.DataFrame(records)
    return df


def load_actual_trades() -> pd.DataFrame:
    trades = _load_metrics_records(METRICS_FILE, "trades")
    trades = trades.copy()
    trades["sample_type"] = "actual"
    trades["buy_date"] = pd.to_datetime(trades["buy_datetime"].astype(str).str[:10])
    trades["actual_return"] = pd.to_numeric(trades["actual_return"], errors="coerce")
    trades["profit_loss"] = pd.to_numeric(trades["profit_loss"], errors="coerce")
    return trades


def load_shadow_trades() -> pd.DataFrame:
    shadow = pd.DataFrame(json.loads(SHADOW_FILE.read_text(encoding="utf-8")))
    shadow = shadow.copy()
    shadow["sample_type"] = "shadow"
    shadow["buy_date"] = pd.to_datetime(
        shadow.get("shadow_attempt_buy_date", shadow.get("buy_datetime")).astype(str)
    )
    shadow["buy_date"] = pd.to_datetime(shadow["buy_date"].astype(str).str[:10])
    shadow["actual_return"] = pd.to_numeric(shadow["actual_return"], errors="coerce")
    shadow["profit_loss"] = pd.to_numeric(shadow["profit_loss"], errors="coerce")
    return shadow


def load_score_z_lookup() -> pd.DataFrame:
    actual = pd.read_csv(SCORE_ACTUAL_FILE, usecols=["stock_code", "buy_datetime", "score_z_in_day"])
    shadow = pd.read_csv(SCORE_SHADOW_FILE, usecols=["stock_code", "buy_datetime", "score_z_in_day"])
    score = pd.concat([actual, shadow], ignore_index=True)
    score["stock_code"] = score["stock_code"].astype(str)
    score["buy_datetime"] = pd.to_datetime(score["buy_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return score.drop_duplicates(["stock_code", "buy_datetime"], keep="last")


def load_stock_factor_data() -> pd.DataFrame:
    cols = ["date", "stock_code"] + STOCK_RULE_COLS
    data = pd.read_parquet(FACTORS, columns=cols)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values(["stock_code", "date"]).reset_index(drop=True)


def load_market_factor_data() -> pd.DataFrame:
    cols = ["date"] + MARKET_RULE_COLS
    data = pd.read_parquet(FACTORS, columns=cols)
    data["date"] = pd.to_datetime(data["date"])
    daily = data.groupby("date", as_index=False)[MARKET_RULE_COLS].median(numeric_only=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    for col in MARKET_RULE_COLS:
        values = pd.to_numeric(daily[col], errors="coerce").to_numpy(dtype=float)
        hist_pct = np.full(len(values), np.nan)
        for idx, value in enumerate(values):
            history = values[:idx]
            history = history[np.isfinite(history)]
            if np.isfinite(value) and len(history) >= 60:
                hist_pct[idx] = float(np.mean(history <= value))
        daily[f"{col}_hist_pct"] = hist_pct
    return daily


def attach_previous_stock_snapshot(events: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        code: (group["date"].to_numpy(dtype="datetime64[ns]"), group.reset_index(drop=True))
        for code, group in factors.groupby("stock_code", sort=False)
    }
    rows = []
    factor_cols = [col for col in factors.columns if col not in {"date", "stock_code"}]
    for _, event in events.iterrows():
        row = event.to_dict()
        item = grouped.get(str(event["stock_code"]))
        if item:
            dates, group = item
            pos = np.searchsorted(dates, np.datetime64(pd.Timestamp(event["buy_date"])), side="left") - 1
            if pos >= 0:
                snapshot = group.iloc[int(pos)]
                row["factor_date"] = snapshot["date"]
                for col in factor_cols:
                    row[col] = snapshot[col]
        rows.append(row)
    return pd.DataFrame(rows)


def attach_previous_market_snapshot(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    dates = daily["date"].to_numpy(dtype="datetime64[ns]")
    rows = []
    value_cols = [col for col in daily.columns if col != "date"]
    for _, event in events.iterrows():
        row = event.to_dict()
        pos = np.searchsorted(dates, np.datetime64(pd.Timestamp(event["buy_date"])), side="left") - 1
        if pos >= 0:
            snapshot = daily.iloc[int(pos)]
            for col in value_cols:
                row[col] = snapshot[col]
        rows.append(row)
    return pd.DataFrame(rows)


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
    out["rule8"] = (
        (z(df, "prev_non_first_limit_up_performance_percentile_20d") >= 0.80)
        & (z(df, "kdj_j_9") <= 0.75)
    )
    out["rule9"] = z(df, "score_z_in_day") <= 1.50
    out["rule10"] = (z(df, "reg_severe_up_10d_ratio_pct") >= 0.80) & (z(df, "market_advance_decline_ratio") <= 0.45)
    out["rule11"] = (z(df, "price_ma5_ratio_raw") >= 9.0) & (z(df, "ma_convergence_score") >= 0.20)
    out["rule12"] = (z(df, "ma5_pullback_depth_10d") <= -0.05) & (z(df, "ret_3d_vs_10d_accel") >= 6.0)
    out["current_v3"] = out.any(axis=1)
    return out.fillna(False)


def bucket_stats(df: pd.DataFrame) -> dict:
    ret = pd.to_numeric(df["actual_return"], errors="coerce")
    pnl = pd.to_numeric(df["profit_loss"], errors="coerce")
    return {
        "n": int(len(df)),
        "big_loss_le_-4pct": int((ret <= -0.04).sum()),
        "small_loss": int(((ret > -0.04) & (ret < 0)).sum()),
        "small_gain_3_8pct": int(((ret >= 0.03) & (ret < 0.08)).sum()),
        "big_gain_ge_8pct": int((ret >= 0.08).sum()),
        "big_gain_ge_10pct": int((ret >= 0.10).sum()),
        "mean_return": float(ret.mean()) if len(df) else np.nan,
        "pnl": float(pnl.sum()) if len(df) else 0.0,
    }


def analyze_subset(df: pd.DataFrame, mask: pd.Series) -> dict:
    mask = pd.Series(mask, index=df.index).fillna(False)
    subset = df[mask].copy()
    rule_hits = {col: int(subset[col].sum()) for col in [f"rule{i}" for i in range(1, 13)] if col in subset}
    return {
        "stats": bucket_stats(subset),
        "current_v3_hits": int(subset["current_v3"].sum()),
        "current_v3_overlap_rate": float(subset["current_v3"].mean()) if len(subset) else np.nan,
        "rule_hits": rule_hits,
        "examples": subset.loc[
            :,
            [
                "sample_type",
                "buy_date",
                "stock_code",
                "actual_return",
                "profit_loss",
                "current_v3",
                "rule1",
                "rule2",
                "rule3",
                "rule4",
                "rule5",
                "rule6",
                "rule7",
                "rule8",
                "rule9",
                "rule10",
                "rule11",
                "rule12",
                "ma5_strength_breadth_v3",
                "market_vols_adaptive_iv",
                "ma5_strength_breadth_v3_hist_pct",
                "market_vols_adaptive_iv_hist_pct",
            ],
        ]
        .sort_values(["current_v3", "actual_return"], ascending=[False, True])
        .head(50)
        .to_dict("records"),
    }


def main() -> None:
    actual = load_actual_trades()
    shadow = load_shadow_trades()
    events = pd.concat([actual, shadow], ignore_index=True, sort=False)
    score_lookup = load_score_z_lookup()
    events["buy_datetime"] = pd.to_datetime(events["buy_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    events = events.merge(score_lookup, on=["stock_code", "buy_datetime"], how="left")

    stock_factors = load_stock_factor_data()
    market_daily = load_market_factor_data()

    events = attach_previous_stock_snapshot(events, stock_factors)
    events = attach_previous_market_snapshot(events, market_daily)

    current_mask = current_three_rule_v3(events)
    events = pd.concat([events, current_mask], axis=1)

    # Market position-control candidate used in the previous audit.
    candidate_mask = (
        (pd.to_numeric(events["ma5_strength_breadth_v3_hist_pct"], errors="coerce") >= 0.80)
        & (pd.to_numeric(events["market_vols_adaptive_iv_hist_pct"], errors="coerce") <= 0.70)
    )
    events["position_control_candidate"] = candidate_mask

    actual_df = events[events["sample_type"].eq("actual")].copy()
    shadow_df = events[events["sample_type"].eq("shadow")].copy()

    actual_candidate = actual_df[actual_df["position_control_candidate"]].copy()
    shadow_candidate = shadow_df[shadow_df["position_control_candidate"]].copy()

    summary = []
    for name, df in [("actual_all", actual_df), ("shadow_all", shadow_df), ("actual_candidate12", actual_candidate), ("shadow_candidate", shadow_candidate)]:
        row = {"scope": name}
        row.update(bucket_stats(df))
        row["current_v3_hits"] = int(df["current_v3"].sum())
        row["current_v3_overlap_rate"] = float(df["current_v3"].mean()) if len(df) else np.nan
        row["candidate_hits"] = int(df["position_control_candidate"].sum())
        row["candidate_overlap_with_current_v3"] = int((df["position_control_candidate"] & df["current_v3"]).sum())
        row["candidate_overlap_rate"] = (
            float((df["position_control_candidate"] & df["current_v3"]).sum() / max(int(df["position_control_candidate"].sum()), 1))
            if len(df)
            else np.nan
        )
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS / "three_rule_v3_position_control_overlap_summary.csv", index=False, encoding="utf-8-sig")
    events.to_csv(RESULTS / "three_rule_v3_position_control_overlap_events.csv", index=False, encoding="utf-8-sig")

    actual_detail = actual_candidate.loc[
        :,
        [
            "buy_date",
            "stock_code",
            "actual_return",
            "profit_loss",
            "current_v3",
            "rule1",
            "rule2",
            "rule3",
            "rule4",
            "rule5",
            "rule6",
            "rule7",
            "rule8",
            "rule9",
            "rule10",
            "rule11",
            "rule12",
            "ma5_strength_breadth_v3",
            "market_vols_adaptive_iv",
            "ma5_strength_breadth_v3_hist_pct",
            "market_vols_adaptive_iv_hist_pct",
        ],
    ].sort_values(["current_v3", "actual_return"], ascending=[False, True])
    actual_detail.to_csv(RESULTS / "three_rule_v3_position_control_candidate12_actual.csv", index=False, encoding="utf-8-sig")

    shadow_detail = shadow_candidate.loc[
        :,
        [
            "buy_date",
            "stock_code",
            "actual_return",
            "profit_loss",
            "shadow_skip_reason",
            "exhaustion_blocked",
            "exhaustion_rule1_hit",
            "exhaustion_rule2_hit",
            "exhaustion_rule3_hit",
            "exhaustion_rule4_hit",
            "exhaustion_rule5_hit",
            "exhaustion_rule6_hit",
            "current_v3",
            "rule1",
            "rule2",
            "rule3",
            "rule4",
            "rule5",
            "rule6",
            "rule7",
            "rule8",
            "rule9",
            "rule10",
            "rule11",
            "rule12",
            "ma5_strength_breadth_v3",
            "market_vols_adaptive_iv",
            "ma5_strength_breadth_v3_hist_pct",
            "market_vols_adaptive_iv_hist_pct",
        ],
    ].sort_values(["current_v3", "actual_return"], ascending=[False, True])
    shadow_detail.to_csv(RESULTS / "three_rule_v3_position_control_candidate12_shadow.csv", index=False, encoding="utf-8-sig")

    print("SUMMARY")
    print(summary_df.to_string(index=False))
    print("\nACTUAL candidate12 overlap with current_v3:", int((actual_candidate["current_v3"]).sum()), "/", len(actual_candidate))
    print("SHADOW candidate overlap with current_v3:", int((shadow_candidate["current_v3"]).sum()), "/", len(shadow_candidate))
    print("\nACTUAL candidate12 detail")
    print(actual_detail.to_string(index=False))
    print("\nSHADOW candidate detail")
    print(shadow_detail.to_string(index=False))


if __name__ == "__main__":
    main()
