"""Audit regulatory and run-up-turbulence risks against 185-log actual/shadow events."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
METRICS = RESULTS / "walkforward_metrics_20260520_210454.json"
SELECTIONS = RESULTS / "walkforward_selections_20260520_210454.json"
SHADOW = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.json"
SOURCE_LOG = ROOT / "stock_selection_test_mode_walkforward_分类模型_原因子185_optuna.log"
OUT_SCAN = RESULTS / "reg_severe_up_current_v3_185_scan.csv"
OUT_DETAIL = RESULTS / "reg_severe_up_current_v3_185_event_detail.csv"
OUT_CURRENT_RULE_AUDIT = RESULTS / "current_v3_rule_audit_185.csv"
OUT_AUDIT_METADATA = RESULTS / "current_v3_rule_audit_185_metadata.json"
OUT_TURBULENCE_SCAN = RESULTS / "runup_turbulence_current_v3_185_scan.csv"
OUT_TURBULENCE_DETAIL = RESULTS / "runup_turbulence_current_v3_185_event_detail.csv"
OUT_FOCUSED_SCAN = RESULTS / "runup_turbulence_focused_pairs_current_v3_185_scan.csv"
OUT_FOCUSED_NEIGHBORHOOD = RESULTS / "runup_turbulence_focused_pairs_current_v3_185_neighborhood.csv"
SCORE_ACTUAL = RESULTS / "score_z_in_day_rule_audit_actual_trades_attached.csv"
SCORE_SHADOW = RESULTS / "score_z_in_day_rule_audit_shadow_attached.csv"

TURBULENCE_FACTORS = [
    "runup_range_expansion_10d",
    "runup_path_turbulence_10d",
    "runup_signed_path_turbulence_10d",
    "runup_volume_friction_10d",
    "runup_upper_shadow_turbulence_10d",
    "high_level_churn_risk_10d",
]

FOCUSED_TURBULENCE_FACTORS = [
    "high_level_churn_risk_10d",
    "runup_path_turbulence_10d",
]

# Direction follows the 20260611_182824 stock-level daily Spearman RankIC.
# "high" means high values are the risk side; "low" means low values remove a
# positively associated quality confirmation. Gap quality is scanned both ways
# because its weak IC and non-monotonic buckets do not justify one direction.
FOCUSED_PAIR_DIRECTIONS = {
    "gap_intraday_quality_score_5d": ("high", "low"),
    "buy_sm_amount_rate": ("high",),
    "upper_shadow_pressure_5d": ("high",),
    "sector_relative_volatility": ("high",),
    "beta_20d": ("high",),
    "downside_beta": ("high",),
    "upper_shadow_ratio_3d_mean": ("high",),
    "main_outflow_retail_takeover": ("high",),
    "abnormal_turnover": ("high",),
    "high_pos_upper_shadow": ("high",),
    "high_pos_reversal_volume": ("high",),
    "shadow_reversal_volume_pressure": ("high",),
    "reg_severe_up_10d_ratio_pct": ("high",),
    "sector_dispersion_leader_alpha": ("high",),
    "alpha_001": ("high",),
    "relative_strength_20d": ("high",),
    "price_ma60_ratio": ("high",),
    "return_20d_intra_z": ("high",),
    "return_10d_intra_z": ("high",),
    "max_rally_20d": ("high",),
    "max_rally_10d": ("high",),
    "ppo_strength": ("high",),
    # Its IC is negative but bucket monotonicity is positive, so do not assume
    # which tail is risky before testing the event panel.
    "lower_shadow_absorption_5d": ("high", "low"),
    "sector_leader_capture_rank": ("high",),
    "sector_correlation_20d": ("low",),
    "close_to_high_3d_mean": ("low",),
    "trend_acceleration": ("low",),
    "pre_second_breakout_shapelet_corr_20d": ("low",),
    "pre_breakout_local_knn_quality_20d_v2": ("low",),
    "turnover_stability": ("low",),
    "support_distance": ("low",),
    "alpha_002": ("low",),
}

PAIR_DIRECTIONS = {
    "runup_ma5_exhaustion": "high",
    "volume_price_exhaustion": "high",
    "high_volume_no_gain_3d": "high",
    "shadow_reversal_volume_pressure": "high",
    "upper_shadow_pressure_5d": "high",
    "close_to_high_3d_mean": "low",
    "close_position_60d": "high",
    "abnormal_turnover": "high",
    "true_turnover_10d": "high",
    "reg_severe_up_10d_ratio_pct": "high",
    "price_ma5_ratio_raw": "high",
    "momentum_peak_decay_3d": "high",
    "sector_momentum_accel_rank": "low",
    "intra_sector_momentum_rank": "low",
    "market_liquidity_change": "low",
    "market_advance_decline_ratio": "low",
    "market_60d_new_high_ratio": "high",
}

FACTOR_COLS = [
    "runup_ma5_exhaustion", "cpv_factor", "prev_day_blowoff_risk",
    "sector_momentum_accel_rank", "sector_leadership_persistence_v3",
    "return_20d_intra_z", "market_liquidity_change", "ulcer_index",
    "momentum_peak_decay_3d", "intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2", "market_advance_decline_ratio", "limit_down_3d",
    "top10_10d_strength_level_v3", "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d", "kdj_j_9",
    "price_ma5_ratio_raw", "ma_convergence_score", "ma5_pullback_depth_10d",
    "ret_3d_vs_10d_accel", "reg_severe_up_10d_ratio",
    "reg_severe_up_10d_ratio_pct", "market_60d_new_high_ratio",
    "ma5_strength_breadth_v3", "market_trend_adaptive_beta",
    "close_position_60d", "upper_shadow_pressure_5d", "close_to_high_3d_mean",
    "abnormal_turnover", "true_turnover_10d", "max_rally_20d",
    "return_5d_intra_z", "ma5_slope",
    "volume_price_exhaustion", "high_volume_no_gain_3d",
    "shadow_reversal_volume_pressure",
    "gap_intraday_quality_score_5d", "buy_sm_amount_rate",
    "sector_relative_volatility", "main_outflow_retail_takeover",
    "high_pos_upper_shadow", "high_pos_reversal_volume", "alpha_001",
    "relative_strength_20d", "sector_dispersion_leader_alpha",
    "sector_correlation_20d", "trend_acceleration",
    "pre_second_breakout_shapelet_corr_20d", "turnover_stability", "alpha_002",
    *TURBULENCE_FACTORS,
]


def z(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")


def current_v3(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["r1"] = (z(df, "runup_ma5_exhaustion") > 0.8) & (z(df, "cpv_factor") <= -0.4)
    out["r2"] = (
        ((z(df, "prev_day_blowoff_risk") >= 2.0) & (z(df, "sector_momentum_accel_rank") <= 0.36))
        | ((z(df, "prev_day_blowoff_risk") > 5.0) & (z(df, "sector_momentum_accel_rank") <= 0.50))
    )
    out["r3"] = (
        (z(df, "sector_leadership_persistence_v3") <= -0.18)
        & (z(df, "return_20d_intra_z") <= 0.40)
    )
    out["r4"] = (z(df, "market_liquidity_change") < 0.94) & (
        (z(df, "ulcer_index") < -2.25)
        | (z(df, "momentum_peak_decay_3d") > 3.20)
        | (z(df, "intra_sector_momentum_rank") < 0.67)
    )
    out["r5"] = (
        (z(df, "mkt_dispersion_shock_v2") < 0.98)
        & (z(df, "market_advance_decline_ratio") < 0.38)
    )
    out["r6"] = z(df, "limit_down_3d") == 1
    out["r7"] = (
        (z(df, "top10_10d_strength_level_v3") >= 0.80)
        & (z(df, "buy_sm_amount_rate") <= -0.20)
    )
    out["r8"] = (
        (z(df, "prev_non_first_limit_up_performance_percentile_20d") >= 0.80)
        & (z(df, "kdj_j_9") <= 0.75)
    )
    out["r9"] = z(df, "score_z_in_day") <= 1.50
    out["r10"] = (
        (z(df, "reg_severe_up_10d_ratio_pct") >= 0.98)
        & (z(df, "market_advance_decline_ratio") >= 4.50)
    )
    out["r11"] = (
        (z(df, "price_ma5_ratio_raw") >= 9.0)
        & (z(df, "ma_convergence_score") >= 0.20)
    )
    out["r12"] = (
        (z(df, "ma5_pullback_depth_10d") <= -0.05)
        & (z(df, "ret_3d_vs_10d_accel") >= 6.0)
    )
    out["current_union"] = out.any(axis=1)
    return out.fillna(False)


def audit_metadata(available_columns: set[str]) -> dict:
    stat = FACTORS.stat()
    parquet = pq.ParquetFile(FACTORS)
    source_log_text = SOURCE_LOG.read_text(encoding="utf-8", errors="ignore") if SOURCE_LOG.exists() else ""
    missing_turbulence = [c for c in TURBULENCE_FACTORS if c not in available_columns]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "factors_path": str(FACTORS),
        "factors_modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "factors_size_bytes": stat.st_size,
        "factors_rows": parquet.metadata.num_rows,
        "factors_columns": parquet.metadata.num_columns,
        "source_log": str(SOURCE_LOG),
        "source_log_filter_enabled": "exhaustion_filter=True" in source_log_text,
        "source_log_filter_disabled": "exhaustion_filter=False" in source_log_text,
        "missing_turbulence_factors": missing_turbulence,
        "notes": [
            "current_union is a retrospective union of matched rules, not a big-loss count.",
            "R10 is the optimized reg_severe_up_10d_ratio_pct>=0.98 and market_advance_decline_ratio>=4.50 rule.",
            "R11 must be re-audited whenever ma_convergence_score or factors.parquet changes.",
            "Candidate scans are not current when their required factors are absent from factors.parquet.",
        ],
    }


def build_current_rule_audit(events: pd.DataFrame, rule_masks: pd.DataFrame) -> pd.DataFrame:
    returns = z(events, "actual_return")
    rows: list[dict] = []
    rule_columns = [c for c in rule_masks.columns if c != "current_union"]
    for rule in rule_columns:
        standalone = rule_masks[rule]
        other_rules = [c for c in rule_columns if c != rule]
        exclusive = standalone & ~rule_masks[other_rules].any(axis=1)
        row = {"rule": rule}
        for prefix, mask in (("standalone", standalone), ("exclusive", exclusive)):
            row.update({
                f"{prefix}_hits": int(mask.sum()),
                f"{prefix}_actual": int((mask & events["event_type"].eq("actual")).sum()),
                f"{prefix}_shadow": int((mask & events["event_type"].eq("shadow")).sum()),
                f"{prefix}_big_loss": int((mask & (returns <= -0.04)).sum()),
                f"{prefix}_small_loss": int((mask & (returns > -0.04) & (returns < 0)).sum()),
                f"{prefix}_gain_0_8": int((mask & (returns >= 0) & (returns < 0.08)).sum()),
                f"{prefix}_big_gain_8": int((mask & (returns >= 0.08)).sum()),
                f"{prefix}_big_gain_10": int((mask & (returns >= 0.10)).sum()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def union_outcome_stats(events: pd.DataFrame, union: pd.Series) -> dict:
    returns = z(events, "actual_return")
    return {
        "events": len(events),
        "actual_events": int(events["event_type"].eq("actual").sum()),
        "shadow_events": int(events["event_type"].eq("shadow").sum()),
        "union_hits": int(union.sum()),
        "union_actual": int((union & events["event_type"].eq("actual")).sum()),
        "union_shadow": int((union & events["event_type"].eq("shadow")).sum()),
        "big_loss_total": int((returns <= -0.04).sum()),
        "big_loss_captured": int((union & (returns <= -0.04)).sum()),
        "small_loss_captured": int((union & (returns > -0.04) & (returns < 0)).sum()),
        "gain_0_8_killed": int((union & (returns >= 0) & (returns < 0.08)).sum()),
        "big_gain_8_total": int((returns >= 0.08).sum()),
        "big_gain_8_killed": int((union & (returns >= 0.08)).sum()),
        "big_gain_10_killed": int((union & (returns >= 0.10)).sum()),
    }


def _records_from_json(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []

    def walk(value) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if "stock_code" in value and "actual_return" in value:
                records.append(value)
            else:
                for item in value.values():
                    walk(item)

    walk(obj)
    return records


def _normalize_records(records: list[dict], event_type: str) -> pd.DataFrame:
    df = pd.DataFrame(records).copy()
    df["event_type"] = event_type
    date_col = df.get("buy_datetime", df.get("shadow_attempt_buy_date"))
    df["buy_date"] = pd.to_datetime(date_col.astype(str).str[:10])
    df["stock_code"] = df["stock_code"].astype(str)
    df["actual_return"] = pd.to_numeric(df["actual_return"], errors="coerce")
    df["profit_loss"] = pd.to_numeric(df.get("profit_loss"), errors="coerce")
    return df[["event_type", "stock_code", "buy_date", "actual_return", "profit_loss"]]


def load_events() -> pd.DataFrame:
    metrics_obj = json.loads(METRICS.read_text(encoding="utf-8"))
    metrics_records = metrics_obj[0]["all_results"]["1"]["trades"]
    selection_records = _records_from_json(SELECTIONS)
    shadow_records = _records_from_json(SHADOW)

    actual = _normalize_records(metrics_records, "actual")
    selections = _normalize_records(selection_records, "actual")
    shadow = _normalize_records(shadow_records, "shadow")
    keys = ["stock_code", "buy_date", "actual_return"]
    metrics_keys = set(map(tuple, actual[keys].itertuples(index=False, name=None)))
    selection_keys = set(map(tuple, selections[keys].itertuples(index=False, name=None)))
    if metrics_keys != selection_keys:
        raise ValueError(
            f"metrics 与 selections 真实交易不一致: metrics={len(metrics_keys)}, "
            f"selections={len(selection_keys)}, diff={len(metrics_keys ^ selection_keys)}"
        )
    return pd.concat([actual, shadow], ignore_index=True).drop_duplicates(
        ["event_type", "stock_code", "buy_date", "actual_return"]
    )


def attach_runtime_scores(events: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for path, event_type in ((SCORE_ACTUAL, "actual"), (SCORE_SHADOW, "shadow")):
        if not path.exists():
            continue
        score = pd.read_csv(path, usecols=["stock_code", "buy_datetime", "score_z_in_day"])
        score["event_type"] = event_type
        score["buy_date"] = pd.to_datetime(score["buy_datetime"].astype(str).str[:10])
        score["stock_code"] = score["stock_code"].astype(str)
        parts.append(score[["event_type", "stock_code", "buy_date", "score_z_in_day"]])
    if not parts:
        events = events.copy()
        events["score_z_in_day"] = np.nan
        return events
    scores = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["event_type", "stock_code", "buy_date"], keep="last"
    )
    return events.merge(scores, on=["event_type", "stock_code", "buy_date"], how="left")


def attach_factor_snapshots(events: pd.DataFrame) -> pd.DataFrame:
    available = set(pq.ParquetFile(FACTORS).schema_arrow.names)
    factor_columns = list(dict.fromkeys(c for c in FACTOR_COLS if c in available))
    columns = ["date", "stock_code"] + factor_columns
    factors = pd.read_parquet(FACTORS, columns=columns)
    factors["date"] = pd.to_datetime(factors["date"])
    factors["stock_code"] = factors["stock_code"].astype(str)
    trading_days = np.array(
        sorted(factors["date"].dropna().unique()),
        dtype="datetime64[ns]",
    )
    positions = np.searchsorted(
        trading_days,
        events["buy_date"].to_numpy(dtype="datetime64[ns]"),
        side="left",
    ) - 1
    events = events.copy()
    events["factor_date"] = [
        pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT for pos in positions
    ]
    factors.rename(columns={"date": "factor_date"}, inplace=True)
    return events.merge(factors, on=["factor_date", "stock_code"], how="left")


def attach_daily_percentile_snapshots(events: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Attach previous-day same-day cross-sectional percentile ranks."""
    available = set(pq.ParquetFile(FACTORS).schema_arrow.names)
    columns = [c for c in columns if c in available]
    start = events["factor_date"].min()
    end = events["factor_date"].max()
    factors = pd.read_parquet(
        FACTORS,
        columns=["date", "stock_code"] + columns,
        filters=[("date", ">=", start), ("date", "<=", end)],
    )
    factors["date"] = pd.to_datetime(factors["date"])
    factors["stock_code"] = factors["stock_code"].astype(str)
    for col in columns:
        factors[f"{col}__daily_pct"] = (
            pd.to_numeric(factors[col], errors="coerce")
            .groupby(factors["date"], sort=False)
            .rank(pct=True, method="average")
        )
    pct_cols = [f"{c}__daily_pct" for c in columns]
    snapshot = factors[["date", "stock_code"] + pct_cols].rename(columns={"date": "factor_date"})
    return events.merge(snapshot, on=["factor_date", "stock_code"], how="left")


def stats(df: pd.DataFrame, mask: pd.Series) -> dict:
    mask = mask.fillna(False)
    ret = z(df, "actual_return")
    return {
        "hits": int(mask.sum()),
        "big_loss": int((mask & (ret <= -0.04)).sum()),
        "small_loss": int((mask & (ret > -0.04) & (ret < 0)).sum()),
        "small_gain_3_8": int((mask & (ret >= 0.03) & (ret < 0.08)).sum()),
        "big_gain_8": int((mask & (ret >= 0.08)).sum()),
        "big_gain_10": int((mask & (ret >= 0.10)).sum()),
        "mean_return": float(ret[mask].mean()) if mask.any() else np.nan,
    }


def evaluate(df: pd.DataFrame, union: pd.Series, name: str, mask: pd.Series) -> list[dict]:
    rows = []
    incremental = mask.fillna(False) & ~union
    for subset in ("combined", "actual", "shadow"):
        subset_mask = (
            pd.Series(True, index=df.index)
            if subset == "combined"
            else df["event_type"].eq(subset)
        )
        row = {"rule": name, "subset": subset}
        row.update(stats(df, incremental & subset_mask))
        row["standalone_hits"] = int((mask.fillna(False) & subset_mask).sum())
        row["overlap_current"] = int((mask.fillna(False) & union & subset_mask).sum())
        row["incremental_rate"] = float(
            incremental[subset_mask].sum() / max(int((mask.fillna(False) & subset_mask).sum()), 1)
        )
        rows.append(row)
    return rows


def segment_stats(df: pd.DataFrame, mask: pd.Series, union: pd.Series) -> dict:
    unique_dates = np.array(sorted(df["buy_date"].dropna().unique()))
    chunks = np.array_split(unique_dates, 4)
    ret = z(df, "actual_return")
    out = {}
    for index, dates in enumerate(chunks, start=1):
        segment = df["buy_date"].isin(dates)
        incremental = mask.fillna(False) & ~union & segment
        out[f"s{index}_inc_hits"] = int(incremental.sum())
        out[f"s{index}_inc_big_loss"] = int((incremental & (ret <= -0.04)).sum())
        out[f"s{index}_inc_big_gain_8"] = int((incremental & (ret >= 0.08)).sum())
    out["inc_big_loss_segment_count"] = sum(out[f"s{i}_inc_big_loss"] > 0 for i in range(1, 5))
    return out


def evaluate_turbulence_rule(
    df: pd.DataFrame,
    rule_masks: pd.DataFrame,
    name: str,
    mask: pd.Series,
    threshold_type: str,
) -> dict:
    union = rule_masks["current_union"]
    ret = z(df, "actual_return")
    mask = mask.fillna(False)
    incremental = mask & ~union
    row = {
        "rule": name,
        "threshold_type": threshold_type,
        **stats(df, mask),
        **{f"inc_{k}": v for k, v in stats(df, incremental).items()},
        "overlap_current": int((mask & union).sum()),
        "overlap_current_rate": float((mask & union).sum() / max(int(mask.sum()), 1)),
    }
    for source in ("actual", "shadow"):
        source_mask = df["event_type"].eq(source)
        source_stats = stats(df, mask & source_mask)
        row.update({f"{source}_{k}": v for k, v in source_stats.items()})
        inc_source_stats = stats(df, incremental & source_mask)
        row.update({f"inc_{source}_{k}": v for k, v in inc_source_stats.items()})
    for rule in [c for c in rule_masks.columns if c.startswith("r")]:
        row[f"overlap_{rule}"] = int((mask & rule_masks[rule]).sum())
    row.update(segment_stats(df, mask, union))
    row["quality"] = (
        6 * row["inc_big_loss"]
        + 2 * row["inc_small_loss"]
        - 15 * row["inc_big_gain_8"]
        - 4 * row["inc_small_gain_3_8"]
        + row["inc_big_loss_segment_count"]
    )
    return row


def scan_turbulence_rules(events: pd.DataFrame, rule_masks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    raw_levels = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    pct_levels = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]

    for factor in TURBULENCE_FACTORS:
        values = z(events, factor)
        pct_values = z(events, f"{factor}__daily_pct")
        for threshold in raw_levels:
            mask = values >= threshold
            rows.append(evaluate_turbulence_rule(
                events, rule_masks, f"{factor}>={threshold:.2f}", mask, "absolute_z",
            ))
        for threshold in pct_levels:
            mask = pct_values >= threshold
            rows.append(evaluate_turbulence_rule(
                events, rule_masks, f"{factor} daily_pct>={threshold:.2f}", mask, "daily_pct",
            ))

        for other, direction in PAIR_DIRECTIONS.items():
            other_pct = z(events, f"{other}__daily_pct")
            for threshold1 in pct_levels:
                first = pct_values >= threshold1
                for threshold2 in pct_levels:
                    second = (
                        other_pct >= threshold2
                        if direction == "high"
                        else other_pct <= 1.0 - threshold2
                    )
                    rows.append(evaluate_turbulence_rule(
                        events,
                        rule_masks,
                        (
                            f"{factor} daily_pct>={threshold1:.2f}"
                            f"&{other} daily_pct {'>=' if direction == 'high' else '<='}"
                            f"{threshold2 if direction == 'high' else 1.0 - threshold2:.2f}"
                        ),
                        first & second,
                        "daily_pct_pair",
                    ))
    return pd.DataFrame(rows)


def scan_focused_turbulence_pairs(events: pd.DataFrame, rule_masks: pd.DataFrame) -> pd.DataFrame:
    """Scan only report-supported two-factor combinations."""
    rows: list[dict] = []
    pct_levels = [0.80, 0.85, 0.90, 0.95, 0.98]
    primary_abs_levels = [0.50, 1.00, 1.50, 2.00]
    high_abs_levels = [0.50, 1.00, 1.50, 2.00]
    low_abs_levels = [-0.50, -1.00, -1.50, -2.00]

    for primary in FOCUSED_TURBULENCE_FACTORS:
        primary_raw = z(events, primary)
        primary_pct = z(events, f"{primary}__daily_pct")
        for other, directions in FOCUSED_PAIR_DIRECTIONS.items():
            other_raw = z(events, other)
            other_pct = z(events, f"{other}__daily_pct")
            for direction in directions:
                for primary_threshold in pct_levels:
                    first = primary_pct >= primary_threshold
                    for other_threshold in pct_levels:
                        second = (
                            other_pct >= other_threshold
                            if direction == "high"
                            else other_pct <= 1.0 - other_threshold
                        )
                        rows.append(evaluate_turbulence_rule(
                            events,
                            rule_masks,
                            (
                                f"{primary} daily_pct>={primary_threshold:.2f}"
                                f"&{other} daily_pct {'>=' if direction == 'high' else '<='}"
                                f"{other_threshold if direction == 'high' else 1.0 - other_threshold:.2f}"
                            ),
                            first & second,
                            "focused_daily_pct_pair",
                        ) | {
                            "primary": primary,
                            "other": other,
                            "direction": direction,
                            "primary_threshold": primary_threshold,
                            "other_threshold": other_threshold,
                        })

                second_levels = high_abs_levels if direction == "high" else low_abs_levels
                for primary_threshold in primary_abs_levels:
                    first = primary_raw >= primary_threshold
                    for other_threshold in second_levels:
                        second = (
                            other_raw >= other_threshold
                            if direction == "high"
                            else other_raw <= other_threshold
                        )
                        rows.append(evaluate_turbulence_rule(
                            events,
                            rule_masks,
                            (
                                f"{primary}>={primary_threshold:.2f}"
                                f"&{other}{'>=' if direction == 'high' else '<='}{other_threshold:.2f}"
                            ),
                            first & second,
                            "focused_absolute_pair",
                        ) | {
                            "primary": primary,
                            "other": other,
                            "direction": direction,
                            "primary_threshold": primary_threshold,
                            "other_threshold": other_threshold,
                        })
    return pd.DataFrame(rows)


def focused_neighborhood_stability(scan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["threshold_type", "primary", "other", "direction"]
    for group_keys, group in scan.groupby(keys, sort=False):
        primary_steps = sorted(group["primary_threshold"].dropna().unique())
        other_steps = sorted(group["other_threshold"].dropna().unique())
        for _, center in group.iterrows():
            p_index = primary_steps.index(center["primary_threshold"])
            o_index = other_steps.index(center["other_threshold"])
            p_near = set(primary_steps[max(0, p_index - 1):p_index + 2])
            o_near = set(other_steps[max(0, o_index - 1):o_index + 2])
            near = group[
                group["primary_threshold"].isin(p_near)
                & group["other_threshold"].isin(o_near)
            ]
            rows.append({
                **dict(zip(keys, group_keys)),
                "primary_threshold": center["primary_threshold"],
                "other_threshold": center["other_threshold"],
                "neighbor_cells": int(len(near)),
                "neighbor_zero_inc_big_gain_ratio": float((near["inc_big_gain_8"] == 0).mean()),
                "neighbor_positive_inc_big_loss_ratio": float((near["inc_big_loss"] > 0).mean()),
                "neighbor_inc_big_loss_mean": float(near["inc_big_loss"].mean()),
                "neighbor_inc_big_gain_mean": float(near["inc_big_gain_8"].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    available_columns = set(pq.ParquetFile(FACTORS).schema_arrow.names)
    metadata = audit_metadata(available_columns)
    OUT_AUDIT_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    events = attach_factor_snapshots(attach_runtime_scores(load_events()))
    pct_columns = list(dict.fromkeys(
        TURBULENCE_FACTORS + list(PAIR_DIRECTIONS) + list(FOCUSED_PAIR_DIRECTIONS)
    ))
    events = attach_daily_percentile_snapshots(events, pct_columns)
    rule_masks = current_v3(events)
    union = rule_masks["current_union"]
    current_rule_audit = build_current_rule_audit(events, rule_masks)
    current_rule_audit.to_csv(OUT_CURRENT_RULE_AUDIT, index=False, encoding="utf-8-sig")
    union_stats = union_outcome_stats(events, union)

    rows: list[dict] = []
    reg_raw = z(events, "reg_severe_up_10d_ratio")
    reg_pct = z(events, "reg_severe_up_10d_ratio_pct")

    for threshold in [0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        rows.extend(evaluate(events, union, f"reg_raw>={threshold:g}", reg_raw >= threshold))
    for threshold in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
        rows.extend(evaluate(events, union, f"reg_pct>={threshold:g}", reg_pct >= threshold))

    adr = z(events, "market_advance_decline_ratio")
    for reg_threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for adr_threshold in [2.0, 2.5, 3.0, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5]:
            rows.extend(evaluate(
                events, union,
                f"reg_raw>={reg_threshold:g}&market_advance_decline_ratio>={adr_threshold:g}",
                (reg_raw >= reg_threshold) & (adr >= adr_threshold),
            ))

    explicit_pct_pairs = {
        "market_advance_decline_ratio": [
            ("<=", value) for value in [0.30, 0.35, 0.38, 0.40, 0.45, 0.50]
        ] + [
            (">=", value) for value in [3.0, 4.0, 4.2, 4.5, 5.0, 5.5]
        ],
        "buy_sm_amount_rate": [
            ("<=", value) for value in [-2.0, -1.5, -1.25, -1.0]
        ],
        "kdj_j_9": [
            ("<=", value) for value in [0.50, 0.75, 1.0]
        ],
        "sector_leadership_persistence_v3": [
            ("<=", value) for value in [-0.30, -0.18, 0.0]
        ],
    }

    def compare(values: pd.Series, op: str, threshold: float) -> pd.Series:
        if op == "<=":
            return values <= threshold
        if op == ">=":
            return values >= threshold
        raise ValueError(op)

    for pct_threshold in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
        for factor, tests in explicit_pct_pairs.items():
            values = z(events, factor)
            for op, threshold in tests:
                rows.extend(evaluate(
                    events, union,
                    f"reg_pct>={pct_threshold:g}&{factor}{op}{threshold:g}",
                    (reg_pct >= pct_threshold) & compare(values, op, threshold),
                ))

    excluded = {
        "actual_return", "profit_loss", "factor_date", "buy_date", "stock_code",
        "event_type", "reg_severe_up_10d_ratio", "reg_severe_up_10d_ratio_pct",
    }
    numeric_candidates = [
        col for col in events.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(events[col])
    ]
    for reg_name, reg_values, reg_thresholds in [
        ("reg_raw", reg_raw, [0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]),
        ("reg_pct", reg_pct, [0.75, 0.80, 0.85, 0.90, 0.95, 0.98]),
    ]:
        for factor in numeric_candidates:
            values = z(events, factor)
            valid = values.dropna()
            if valid.nunique() < 4:
                continue
            thresholds = sorted(set(float(valid.quantile(q)) for q in (0.10, 0.20, 0.30, 0.70, 0.80, 0.90)))
            for reg_threshold in reg_thresholds:
                reg_mask = reg_values >= reg_threshold
                for threshold in thresholds:
                    rows.extend(evaluate(
                        events, union,
                        f"{reg_name}>={reg_threshold:g}&{factor}<={threshold:.6g}",
                        reg_mask & (values <= threshold),
                    ))
                    rows.extend(evaluate(
                        events, union,
                        f"{reg_name}>={reg_threshold:g}&{factor}>={threshold:.6g}",
                        reg_mask & (values >= threshold),
                    ))

    result = pd.DataFrame(rows)
    result["quality"] = (
        result["big_loss"] * 5
        + result["small_loss"]
        - result["big_gain_8"] * 10
        - result["small_gain_3_8"] * 2
    )
    result.to_csv(OUT_SCAN, index=False, encoding="utf-8-sig")

    detail = events[[
        "event_type", "stock_code", "buy_date", "factor_date", "actual_return",
        "reg_severe_up_10d_ratio", "reg_severe_up_10d_ratio_pct",
        "market_advance_decline_ratio", "buy_sm_amount_rate", "kdj_j_9",
        "sector_leadership_persistence_v3",
    ]].copy()
    detail["current_union"] = union
    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    turbulence_scan = scan_turbulence_rules(events, rule_masks)
    turbulence_scan.to_csv(OUT_TURBULENCE_SCAN, index=False, encoding="utf-8-sig")
    focused_scan = scan_focused_turbulence_pairs(events, rule_masks)
    focused_neighborhood = focused_neighborhood_stability(focused_scan)
    focused_scan = focused_scan.merge(
        focused_neighborhood,
        on=[
            "threshold_type", "primary", "other", "direction",
            "primary_threshold", "other_threshold",
        ],
        how="left",
    )
    focused_scan.to_csv(OUT_FOCUSED_SCAN, index=False, encoding="utf-8-sig")
    focused_neighborhood.to_csv(OUT_FOCUSED_NEIGHBORHOOD, index=False, encoding="utf-8-sig")
    turbulence_detail_cols = [
        "event_type", "stock_code", "buy_date", "factor_date", "actual_return",
        *TURBULENCE_FACTORS,
        *[f"{c}__daily_pct" for c in pct_columns],
    ]
    turbulence_detail = events[
        [c for c in turbulence_detail_cols if c in events.columns]
    ].copy()
    for col in rule_masks.columns:
        turbulence_detail[col] = rule_masks[col].to_numpy()
    turbulence_detail.to_csv(OUT_TURBULENCE_DETAIL, index=False, encoding="utf-8-sig")

    combined = result[result["subset"] == "combined"]
    viable = combined[(combined["big_loss"] > 0) & (combined["big_gain_8"] == 0)].sort_values(
        ["quality", "big_loss", "small_gain_3_8", "hits"],
        ascending=[False, False, True, True],
    )
    print("\nAUDIT CONTEXT")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if metadata["source_log_filter_disabled"]:
        print("WARNING: source log has exhaustion_filter=False; all rule hits below are retrospective simulations.")
    if metadata["missing_turbulence_factors"]:
        print(
            "WARNING: turbulence candidate scans are incomplete because factors.parquet is missing: "
            + ", ".join(metadata["missing_turbulence_factors"])
        )
    print("\nCURRENT V3 UNION OUTCOMES")
    print(json.dumps(union_stats, ensure_ascii=False, indent=2))
    print("\nCURRENT V3 RULE STANDALONE / EXCLUSIVE")
    print(current_rule_audit.to_string(index=False))
    print(f"current_union={int(union.sum())}, unfiltered={int((~union).sum())}")
    print("unfiltered buckets:", stats(events, ~union))
    print("\nTOP ZERO-BIG-GAIN CANDIDATES")
    print(viable.head(40).to_string(index=False))
    turbulence_viable = turbulence_scan[
        (turbulence_scan["inc_big_loss"] > 0)
        & (turbulence_scan["inc_big_gain_8"] == 0)
    ].sort_values(
        ["quality", "inc_big_loss", "inc_big_loss_segment_count", "inc_hits"],
        ascending=[False, False, False, True],
    )
    turbulence_cols = [
        "rule", "threshold_type", "hits", "big_loss", "small_loss",
        "small_gain_3_8", "big_gain_8", "inc_hits", "inc_big_loss",
        "inc_small_loss", "inc_small_gain_3_8", "inc_big_gain_8",
        "overlap_current", "inc_big_loss_segment_count", "quality",
    ]
    print("\nTOP TURBULENCE ZERO-INCREMENTAL-BIG-GAIN CANDIDATES")
    print(turbulence_viable[turbulence_cols].head(60).to_string(index=False))
    focused_viable = focused_scan[
        (focused_scan["inc_big_loss"] > 0)
        & (focused_scan["inc_big_gain_8"] == 0)
    ].sort_values(
        [
            "inc_big_loss", "inc_big_loss_segment_count",
            "neighbor_zero_inc_big_gain_ratio",
            "neighbor_positive_inc_big_loss_ratio", "inc_hits",
        ],
        ascending=[False, False, False, False, True],
    )
    focused_cols = [
        "rule", "threshold_type", "hits", "big_loss", "small_loss",
        "small_gain_3_8", "big_gain_8", "inc_hits", "inc_big_loss",
        "inc_small_loss", "inc_small_gain_3_8", "inc_big_gain_8",
        "overlap_current", "inc_big_loss_segment_count",
        "neighbor_zero_inc_big_gain_ratio", "neighbor_positive_inc_big_loss_ratio",
    ]
    print("\nTOP FOCUSED ZERO-INCREMENTAL-BIG-GAIN CANDIDATES")
    print(focused_viable[focused_cols].head(60).to_string(index=False))
    print(f"\nscan={OUT_SCAN}")
    print(f"detail={OUT_DETAIL}")
    print(f"turbulence_scan={OUT_TURBULENCE_SCAN}")
    print(f"turbulence_detail={OUT_TURBULENCE_DETAIL}")
    print(f"focused_scan={OUT_FOCUSED_SCAN}")
    print(f"focused_neighborhood={OUT_FOCUSED_NEIGHBORHOOD}")
    print(f"current_rule_audit={OUT_CURRENT_RULE_AUDIT}")
    print(f"audit_metadata={OUT_AUDIT_METADATA}")


if __name__ == "__main__":
    main()
