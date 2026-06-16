"""Audit new factors and their pair rules against current three_rule_v3.

All thresholds use the previous trading day's same-day cross-sectional
percentile ranks. This makes the rules executable without test-window
distribution leakage.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from audit_three_rule_v3_daily_pct_pairs import (
    CURRENT_RULE_COLS,
    FACTORS,
    RESULTS,
    attach_prev_snapshot,
    current_three_rule_v3,
)


METRICS_FILE = RESULTS / "walkforward_metrics_20260520_210454.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.json"
SCORE_ACTUAL_FILE = RESULTS / "score_z_in_day_rule_audit_actual_trades_attached.csv"
SCORE_SHADOW_FILE = RESULTS / "score_z_in_day_rule_audit_shadow_attached.csv"

NEW_FACTORS = [
    "gap_intraday_quality_score_5d",
    "pre_second_breakout_shapelet_corr_20d",
]

PCT_LEVELS = [0.70, 0.80, 0.85, 0.90, 0.95]
BATCH_SIZE = 20
MIN_HITS = 3


def numeric_factor_columns() -> list[str]:
    schema = pq.ParquetFile(FACTORS).schema_arrow
    out = []
    for field in schema:
        if field.name in {"date", "stock_code"}:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            out.append(field.name)
    return out


def load_actual_trades() -> pd.DataFrame:
    obj = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    records = obj[0]["all_results"]["1"]["trades"]
    trades = pd.DataFrame(records).copy()
    trades["source"] = "actual"
    trades["buy_date"] = pd.to_datetime(trades["buy_datetime"].astype(str).str[:10])
    return trades


def load_shadow_trades() -> pd.DataFrame:
    shadow = pd.DataFrame(json.loads(SHADOW_FILE.read_text(encoding="utf-8"))).copy()
    shadow["source"] = "shadow"
    shadow["buy_date"] = pd.to_datetime(
        shadow.get("shadow_attempt_buy_date", shadow.get("buy_datetime")).astype(str).str[:10]
    )
    return shadow


def load_events() -> pd.DataFrame:
    events = pd.concat([load_actual_trades(), load_shadow_trades()], ignore_index=True, sort=False)
    events["buy_date"] = pd.to_datetime(events["buy_date"])
    events["stock_code"] = events["stock_code"].astype(str)
    events["actual_return"] = pd.to_numeric(events["actual_return"], errors="coerce")
    events = events[events["actual_return"].notna()].copy()
    scores = pd.concat(
        [
            pd.read_csv(SCORE_ACTUAL_FILE, usecols=["stock_code", "buy_datetime", "score_z_in_day"]),
            pd.read_csv(SCORE_SHADOW_FILE, usecols=["stock_code", "buy_datetime", "score_z_in_day"]),
        ],
        ignore_index=True,
    )
    scores["stock_code"] = scores["stock_code"].astype(str)
    scores["buy_datetime"] = pd.to_datetime(scores["buy_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    scores = scores.drop_duplicates(["stock_code", "buy_datetime"], keep="last")
    events["buy_datetime"] = pd.to_datetime(events["buy_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    events = events.drop(columns=["score_z_in_day"], errors="ignore").merge(
        scores,
        on=["stock_code", "buy_datetime"],
        how="left",
    )
    return events.sort_values(["buy_date", "stock_code", "source"]).reset_index(drop=True)


def load_raw_current_rule_snapshots(events: pd.DataFrame) -> pd.DataFrame:
    start = (events["buy_date"].min() - pd.Timedelta(days=35)).normalize()
    end = events["buy_date"].max().normalize()
    available = set(pq.ParquetFile(FACTORS).schema.names)
    snapshot_cols = [c for c in CURRENT_RULE_COLS if c != "score_z_in_day"]
    cols = ["date", "stock_code"] + [c for c in snapshot_cols if c in available]
    factors = pd.read_parquet(
        FACTORS,
        columns=cols,
        filters=[("date", ">=", start), ("date", "<=", end)],
    )
    factors["date"] = pd.to_datetime(factors["date"])
    out = attach_prev_snapshot(events, factors, snapshot_cols)
    masks = current_three_rule_v3(out)
    return pd.concat([out, masks], axis=1)


def build_event_percentile_panel(events: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Attach previous-day daily cross-sectional percentile ranks in batches."""
    base = events[["buy_date", "stock_code"]].copy()
    parts = [base]
    start = (events["buy_date"].min() - pd.Timedelta(days=35)).normalize()
    end = events["buy_date"].max().normalize()

    for offset in range(0, len(factor_cols), BATCH_SIZE):
        batch = factor_cols[offset:offset + BATCH_SIZE]
        frame = pd.read_parquet(
            FACTORS,
            columns=["date", "stock_code"] + batch,
            filters=[("date", ">=", start), ("date", "<=", end)],
        )
        frame["date"] = pd.to_datetime(frame["date"])
        for col in batch:
            frame[col] = frame.groupby("date", sort=False)[col].rank(pct=True, method="average")
        snapped = attach_prev_snapshot(events[["buy_date", "stock_code"]], frame, batch)
        part = pd.DataFrame(
            {
                col: pd.to_numeric(snapped[col], errors="coerce").to_numpy()
                for col in batch
            },
            index=base.index,
        )
        parts.append(part)
        print(f"percentile snapshots: {min(offset + len(batch), len(factor_cols))}/{len(factor_cols)}")
    return pd.concat(parts, axis=1)


def make_condition(values: pd.Series, direction: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if direction == "high":
        return values >= threshold
    return values <= 1.0 - threshold


def outcome_counts(df: pd.DataFrame, mask: pd.Series, prefix: str = "") -> dict:
    ret = pd.to_numeric(df["actual_return"], errors="coerce")
    mask = pd.Series(mask, index=df.index).fillna(False)
    return {
        f"{prefix}hits": int(mask.sum()),
        f"{prefix}big_loss": int((mask & (ret <= -0.04)).sum()),
        f"{prefix}small_loss": int((mask & (ret > -0.04) & (ret < 0)).sum()),
        f"{prefix}small_gain_3_8": int((mask & (ret >= 0.03) & (ret < 0.08)).sum()),
        f"{prefix}big_gain_8": int((mask & (ret >= 0.08)).sum()),
        f"{prefix}big_gain_10": int((mask & (ret >= 0.10)).sum()),
    }


def segment_stats(events: pd.DataFrame, mask: pd.Series) -> dict:
    dates = pd.Series(sorted(events["buy_date"].dropna().unique()))
    chunks = np.array_split(dates.to_numpy(), 4)
    ret = pd.to_numeric(events["actual_return"], errors="coerce")
    out = {}
    loss_segments = 0
    gain_segments = 0
    active_segments = 0
    for i, chunk in enumerate(chunks, start=1):
        segment_mask = events["buy_date"].isin(chunk)
        hit = mask & segment_mask
        hits = int(hit.sum())
        losses = int((hit & (ret <= -0.04)).sum())
        gains = int((hit & (ret >= 0.08)).sum())
        out[f"s{i}_hits"] = hits
        out[f"s{i}_big_loss"] = losses
        out[f"s{i}_big_gain_8"] = gains
        active_segments += int(hits > 0)
        loss_segments += int(losses > 0)
        gain_segments += int(gains > 0)
    out["active_segment_count"] = active_segments
    out["big_loss_segment_count"] = loss_segments
    out["big_gain_segment_count"] = gain_segments
    return out


def evaluate_rule(
    events: pd.DataFrame,
    factor1: str,
    direction1: str,
    threshold1: float,
    factor2: str | None = None,
    direction2: str | None = None,
    threshold2: float | None = None,
) -> dict:
    mask = make_condition(events[factor1], direction1, threshold1)
    if factor2 is not None and direction2 is not None and threshold2 is not None:
        mask &= make_condition(events[factor2], direction2, threshold2)

    current = events["current_v3"].fillna(False).astype(bool)
    incremental = mask & ~current
    ret = pd.to_numeric(events["actual_return"], errors="coerce")
    row = {
        "factor1": factor1,
        "direction1": direction1,
        "threshold1": threshold1,
        "factor2": factor2 or "",
        "direction2": direction2 or "",
        "threshold2": threshold2 if threshold2 is not None else np.nan,
        "current_overlap": int((mask & current).sum()),
        "current_overlap_rate": float((mask & current).sum() / max(int(mask.sum()), 1)),
        "incremental_hits": int(incremental.sum()),
        "incremental_big_loss": int((incremental & (ret <= -0.04)).sum()),
        "incremental_big_gain_8": int((incremental & (ret >= 0.08)).sum()),
    }
    row.update(outcome_counts(events, mask))
    for source in ["actual", "shadow"]:
        source_mask = events["source"].eq(source)
        row.update(outcome_counts(events, mask & source_mask, prefix=f"{source}_"))
    row.update(segment_stats(events, mask))

    all_big_losses = max(int((ret <= -0.04).sum()), 1)
    all_big_gains = max(int((ret >= 0.08).sum()), 1)
    row["big_loss_capture_rate"] = row["big_loss"] / all_big_losses
    row["big_gain_kill_rate"] = row["big_gain_8"] / all_big_gains
    row["incremental_purity"] = row["incremental_big_loss"] / max(row["incremental_hits"], 1)
    row["diagnostic_score"] = (
        3.0 * row["incremental_big_loss"]
        + 1.0 * row["big_loss"]
        - 5.0 * row["incremental_big_gain_8"]
        - 3.0 * row["big_gain_8"]
        + 0.5 * row["big_loss_segment_count"]
        - 0.5 * row["current_overlap"]
    )
    return row


def scan_rules(events: pd.DataFrame, all_factors: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    directions = ["high", "low"]

    for new_factor in NEW_FACTORS:
        for direction in directions:
            for threshold in PCT_LEVELS:
                rows.append(evaluate_rule(events, new_factor, direction, threshold))

        for other in all_factors:
            if other == new_factor:
                continue
            for direction1 in directions:
                for direction2 in directions:
                    for threshold1 in PCT_LEVELS:
                        for threshold2 in PCT_LEVELS:
                            rows.append(
                                evaluate_rule(
                                    events,
                                    new_factor,
                                    direction1,
                                    threshold1,
                                    other,
                                    direction2,
                                    threshold2,
                                )
                            )
    return pd.DataFrame(rows)


def add_neighborhood_stability(scan: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether nearby threshold cells keep the same risk direction."""
    pairs = scan[scan["factor2"].ne("")].copy()
    rows = []
    group_cols = ["factor1", "direction1", "factor2", "direction2"]
    for keys, group in pairs.groupby(group_cols, sort=False):
        for _, center in group.iterrows():
            near = group[
                (group["threshold1"].sub(center["threshold1"]).abs() <= 0.05 + 1e-12)
                & (group["threshold2"].sub(center["threshold2"]).abs() <= 0.05 + 1e-12)
            ]
            rows.append({
                **dict(zip(group_cols, keys)),
                "threshold1": center["threshold1"],
                "threshold2": center["threshold2"],
                "neighbor_cells": int(len(near)),
                "neighbor_zero_big_gain_ratio": float((near["big_gain_8"] == 0).mean()),
                "neighbor_incremental_loss_positive_ratio": float((near["incremental_big_loss"] > 0).mean()),
                "neighbor_big_loss_mean": float(near["big_loss"].mean()),
                "neighbor_incremental_big_loss_mean": float(near["incremental_big_loss"].mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    events = load_events()
    current_events = load_raw_current_rule_snapshots(events)

    factors = numeric_factor_columns()
    pct_panel = build_event_percentile_panel(events, factors)
    analysis = pd.concat(
        [
            current_events.drop(columns=factors, errors="ignore").reset_index(drop=True),
            pct_panel[factors].reset_index(drop=True),
        ],
        axis=1,
    )

    scan = scan_rules(analysis, factors)
    neighborhood = add_neighborhood_stability(scan)
    scan = scan.merge(
        neighborhood,
        on=["factor1", "direction1", "factor2", "direction2", "threshold1", "threshold2"],
        how="left",
    )

    viable = scan[
        (scan["factor2"].ne(""))
        & (scan["hits"] >= MIN_HITS)
        & (scan["big_gain_8"] == 0)
        & (scan["incremental_big_gain_8"] == 0)
        & (scan["incremental_big_loss"] > 0)
    ].copy()
    viable = viable.sort_values(
        [
            "incremental_big_loss",
            "big_loss_segment_count",
            "neighbor_incremental_loss_positive_ratio",
            "current_overlap_rate",
            "hits",
        ],
        ascending=[False, False, False, True, True],
    )

    mutual = scan[
        scan["factor2"].isin(NEW_FACTORS)
        & scan["factor1"].isin(NEW_FACTORS)
        & scan["factor1"].ne(scan["factor2"])
    ].copy()

    scan.to_csv(RESULTS / "new_factors_three_rule_v3_all_scan.csv", index=False, encoding="utf-8-sig")
    viable.to_csv(RESULTS / "new_factors_three_rule_v3_viable.csv", index=False, encoding="utf-8-sig")
    mutual.to_csv(RESULTS / "new_factors_three_rule_v3_mutual.csv", index=False, encoding="utf-8-sig")
    analysis[
        ["source", "buy_date", "stock_code", "actual_return", "current_v3"] + NEW_FACTORS
    ].to_csv(RESULTS / "new_factors_three_rule_v3_event_panel.csv", index=False, encoding="utf-8-sig")

    summary = {
        "metrics_file": str(METRICS_FILE),
        "shadow_file": str(SHADOW_FILE),
        "events": int(len(analysis)),
        "actual_events": int(analysis["source"].eq("actual").sum()),
        "shadow_events": int(analysis["source"].eq("shadow").sum()),
        "all_big_losses": int((analysis["actual_return"] <= -0.04).sum()),
        "all_big_gains": int((analysis["actual_return"] >= 0.08).sum()),
        "scanned_factor_count": len(factors),
        "scan_rows": int(len(scan)),
        "viable_rows": int(len(viable)),
    }
    (RESULTS / "new_factors_three_rule_v3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop viable rules:")
    cols = [
        "factor1", "direction1", "threshold1", "factor2", "direction2", "threshold2",
        "hits", "big_loss", "big_gain_8", "incremental_hits", "incremental_big_loss",
        "current_overlap", "current_overlap_rate", "big_loss_segment_count",
        "neighbor_zero_big_gain_ratio", "neighbor_incremental_loss_positive_ratio",
    ]
    print(viable[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
