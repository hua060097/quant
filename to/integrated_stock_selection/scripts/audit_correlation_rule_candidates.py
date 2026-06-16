"""Audit correlation-discovered candidate rules against 185 actual + shadow events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
METRICS = RESULTS / "walkforward_metrics_20260520_210454.json"
SHADOW = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.csv"
FACTORS = BASE / "factor_lab_data" / "factors.parquet"

EXISTING_COLS = [
    "runup_ma5_exhaustion", "cpv_factor", "prev_day_blowoff_risk",
    "sector_momentum_accel_rank", "sector_leadership_persistence_v3",
    "return_20d_intra_z", "market_liquidity_change", "ulcer_index",
    "momentum_peak_decay_3d", "intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2", "market_advance_decline_ratio", "limit_down_3d",
    "top10_10d_strength_level_v3", "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d", "kdj_j_9",
    "reg_severe_up_10d_ratio_pct", "price_ma5_ratio_raw", "ma_convergence_score",
    "ma5_pullback_depth_10d", "ret_3d_vs_10d_accel",
]

CANDIDATE_COLS = [
    "market_60d_new_high_ratio", "ma5_strength_breadth_v3", "market_trend_adaptive_beta",
    "close_position_60d", "upper_shadow_pressure_5d", "close_to_high_3d_mean",
    "abnormal_turnover", "true_turnover_10d", "max_rally_20d",
    "return_5d_intra_z", "ma5_slope",
]


def load_events() -> pd.DataFrame:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    trades = metrics[0]["all_results"]["1"]["trades"]
    actual = pd.DataFrame([{
        "event_type": "actual",
        "stock_code": row["stock_code"],
        "buy_date": pd.Timestamp(row["buy_datetime"][:10]),
        "actual_return": float(row["actual_return"]),
        "profit_loss": float(row["profit_loss"]),
    } for row in trades])
    shadow_raw = pd.read_csv(SHADOW)
    shadow = pd.DataFrame({
        "event_type": "shadow",
        "stock_code": shadow_raw["stock_code"].astype(str),
        "buy_date": pd.to_datetime(shadow_raw["buy_datetime"].astype(str).str[:10]),
        "actual_return": pd.to_numeric(shadow_raw["actual_return"], errors="coerce"),
        "profit_loss": pd.to_numeric(shadow_raw["profit_loss"], errors="coerce"),
    })
    return pd.concat([actual, shadow], ignore_index=True).dropna(subset=["actual_return"])


def load_factor_data() -> pd.DataFrame:
    names = set(pq.ParquetFile(FACTORS).schema_arrow.names)
    columns = ["date", "stock_code"] + [
        col for col in dict.fromkeys(EXISTING_COLS + CANDIDATE_COLS) if col in names
    ]
    factors = pd.read_parquet(FACTORS, columns=columns)
    factors["date"] = pd.to_datetime(factors["date"])
    return factors.sort_values(["stock_code", "date"]).reset_index(drop=True)


def attach_previous_snapshot(events: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        code: (group["date"].to_numpy(dtype="datetime64[ns]"), group.reset_index(drop=True))
        for code, group in factors.groupby("stock_code", sort=False)
    }
    rows = []
    factor_cols = [col for col in factors.columns if col not in {"date", "stock_code"}]
    for _, event in events.iterrows():
        row = event.to_dict()
        item = grouped.get(event["stock_code"])
        if item:
            dates, group = item
            pos = np.searchsorted(dates, np.datetime64(event["buy_date"]), side="left") - 1
            if pos >= 0:
                snapshot = group.iloc[int(pos)]
                row["factor_date"] = snapshot["date"]
                for col in factor_cols:
                    row[col] = snapshot[col]
        rows.append(row)
    return pd.DataFrame(rows)


def z(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


def existing_union(df: pd.DataFrame) -> pd.Series:
    rules = [
        (z(df, "runup_ma5_exhaustion") > 0.8) & (z(df, "cpv_factor") <= -0.4),
        ((z(df, "prev_day_blowoff_risk") >= 2) & (z(df, "sector_momentum_accel_rank") <= .30))
        | ((z(df, "prev_day_blowoff_risk") > 5) & (z(df, "sector_momentum_accel_rank") <= .50))
        | ((z(df, "prev_day_blowoff_risk") > 1) & (z(df, "sector_momentum_accel_rank") <= .15)),
        (z(df, "sector_leadership_persistence_v3") <= -.18) & (z(df, "return_20d_intra_z") <= .40),
        (z(df, "market_liquidity_change") < .94) & (
            (z(df, "ulcer_index") < -.8)
            | (z(df, "momentum_peak_decay_3d") > 1)
            | (z(df, "intra_sector_momentum_rank") < .67)
        ),
        (z(df, "mkt_dispersion_shock_v2") < .98) & (z(df, "market_advance_decline_ratio") < .38),
        z(df, "limit_down_3d") == 1,
        (z(df, "top10_10d_strength_level_v3") >= .70) & (z(df, "buy_sm_amount_rate") <= -1.25),
        (z(df, "prev_non_first_limit_up_performance_percentile_20d") >= .80) & (z(df, "kdj_j_9") <= .75),
        (z(df, "reg_severe_up_10d_ratio_pct") >= .80) & (z(df, "market_advance_decline_ratio") <= .45),
        (z(df, "price_ma5_ratio_raw") >= 9) & (z(df, "ma_convergence_score") >= .20),
        (z(df, "ma5_pullback_depth_10d") <= -.05) & (z(df, "ret_3d_vs_10d_accel") >= 6),
    ]
    union = pd.Series(False, index=df.index)
    for rule in rules:
        union |= rule.fillna(False)
    return union


def stats(df: pd.DataFrame, mask: pd.Series) -> dict:
    mask = pd.Series(mask, index=df.index).fillna(False)
    ret = z(df, "actual_return")
    return {
        "hits": int(mask.sum()),
        "big_loss": int((mask & (ret <= -.04)).sum()),
        "small_loss": int((mask & (ret > -.04) & (ret < 0)).sum()),
        "small_gain_3_8": int((mask & (ret >= .03) & (ret < .08)).sum()),
        "big_gain_8": int((mask & (ret >= .08)).sum()),
        "big_gain_10": int((mask & (ret >= .10)).sum()),
        "mean_return": float(ret[mask].mean()) if mask.any() else np.nan,
        "pnl": float(z(df, "profit_loss")[mask].sum()) if mask.any() else 0.0,
    }


def threshold_map(factors: pd.DataFrame) -> dict:
    result = {}
    for col in CANDIDATE_COLS:
        if col not in factors:
            continue
        values = pd.to_numeric(factors[col], errors="coerce").dropna()
        result[col] = {q: float(values.quantile(q)) for q in (.10, .20, .30, .70, .80, .90)}
    return result


def candidate_rules(df: pd.DataFrame, q: dict) -> dict[str, pd.Series]:
    hi = lambda c, p: z(df, c) >= q[c][p]
    lo = lambda c, p: z(df, c) <= q[c][p]
    rules = {}
    for p in (.70, .80, .90):
        rules[f"A_newhigh_hi{p:.1f}"] = hi("market_60d_new_high_ratio", p)
        rules[f"A_breadth_hi{p:.1f}"] = hi("ma5_strength_breadth_v3", p)
        rules[f"A_beta_hi{p:.1f}"] = hi("market_trend_adaptive_beta", p)
    for p in (.70, .80, .90):
        for low_p in (.10, .20, .30):
            rules[f"B_pos{p:.1f}_shadow{p:.1f}_cth{low_p:.1f}"] = (
                hi("close_position_60d", p) & hi("upper_shadow_pressure_5d", p)
                & lo("close_to_high_3d_mean", low_p)
            )
            rules[f"B2_pos{p:.1f}_cth{low_p:.1f}"] = hi("close_position_60d", p) & lo("close_to_high_3d_mean", low_p)
            rules[f"B2_shadow{p:.1f}_cth{low_p:.1f}"] = hi("upper_shadow_pressure_5d", p) & lo("close_to_high_3d_mean", low_p)
        rules[f"B2_pos{p:.1f}_shadow{p:.1f}"] = hi("close_position_60d", p) & hi("upper_shadow_pressure_5d", p)
        rules[f"C_pos{p:.1f}_abn{p:.1f}_rally{p:.1f}"] = (
            hi("close_position_60d", p) & hi("abnormal_turnover", p) & hi("max_rally_20d", p)
        )
        rules[f"C2_pos{p:.1f}_rally{p:.1f}"] = hi("close_position_60d", p) & hi("max_rally_20d", p)
        rules[f"C2_abn{p:.1f}_rally{p:.1f}"] = hi("abnormal_turnover", p) & hi("max_rally_20d", p)
        rules[f"C2_true{p:.1f}_rally{p:.1f}"] = hi("true_turnover_10d", p) & hi("max_rally_20d", p)
        rules[f"D_ret5{p:.1f}_slope{p:.1f}_shadow{p:.1f}"] = (
            hi("return_5d_intra_z", p) & hi("ma5_slope", p) & hi("upper_shadow_pressure_5d", p)
        )
        rules[f"D2_ret5{p:.1f}_shadow{p:.1f}"] = hi("return_5d_intra_z", p) & hi("upper_shadow_pressure_5d", p)
        rules[f"D2_slope{p:.1f}_shadow{p:.1f}"] = hi("ma5_slope", p) & hi("upper_shadow_pressure_5d", p)
    return rules


def main() -> None:
    factors = load_factor_data()
    events = attach_previous_snapshot(load_events(), factors)
    union = existing_union(events)
    q = threshold_map(factors)
    rows = []
    for name, mask in candidate_rules(events, q).items():
        for event_type in ("actual", "shadow", "combined"):
            subset = pd.Series(True, index=events.index) if event_type == "combined" else events["event_type"].eq(event_type)
            for scope, scope_mask in (
                ("standalone", mask & subset),
                ("incremental", mask & ~union & subset),
                ("overlap", mask & union & subset),
            ):
                row = {"rule": name, "event_type": event_type, "scope": scope}
                row.update(stats(events, scope_mask))
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "correlation_candidate_rule_audit.csv", index=False, encoding="utf-8-sig")
    events.to_csv(RESULTS / "correlation_candidate_events_enriched.csv", index=False, encoding="utf-8-sig")

    combined_inc = result[(result["event_type"] == "combined") & (result["scope"] == "incremental")].copy()
    combined_inc["quality"] = (
        combined_inc["big_loss"] * 5 + combined_inc["small_loss"]
        - combined_inc["big_gain_8"] * 6 - combined_inc["small_gain_3_8"] * 2
    )
    combined_inc = combined_inc.sort_values(["quality", "big_loss", "hits"], ascending=[False, False, True])
    print("CURRENT UNION (without R9 score_z_in_day)", stats(events, union))
    print("\nTOP INCREMENTAL CANDIDATES")
    print(combined_inc.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
