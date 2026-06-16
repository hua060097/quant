"""Dedicated three_rule_v3 audit for shadow/close-position factors."""

from __future__ import annotations

import itertools

import pandas as pd
import pyarrow.parquet as pq

from audit_correlation_rule_candidates import (
    FACTORS,
    RESULTS,
    EXISTING_COLS,
    attach_previous_snapshot,
    existing_union,
    load_events,
    stats,
    z,
)


ANCHORS_HIGH = ["lower_shadow_absorption_5d", "upper_shadow_pressure_5d"]
ANCHORS_LOW = ["close_to_high_3d_mean"]
OTHER_HIGH = [
    "close_position_60d", "max_rally_20d", "abnormal_turnover", "true_turnover_10d",
    "return_5d_intra_z", "ma5_slope", "return_20d_sector_mean",
    "market_trend_adaptive_beta", "sector_relative_volatility", "downside_beta",
    "severe_abnormal_pressure_10d", "consecutive_up_days_5d", "range_compression",
]
OTHER_LOW = [
    "size_rotation_strength", "strong_stock_volume_confirmation_v3",
    "trend_pullback_composite_alpha", "turnover_stability",
    "sector_correlation_20d", "ret_3d_vs_10d_accel",
]
HIGH_PS = [.60, .70, .80, .90, .95]
LOW_PS = [.40, .30, .20, .10, .05]


def main() -> None:
    names = set(pq.ParquetFile(FACTORS).schema_arrow.names)
    columns = [
        col for col in dict.fromkeys(EXISTING_COLS + ANCHORS_HIGH + ANCHORS_LOW + OTHER_HIGH + OTHER_LOW)
        if col in names
    ]
    factors = pd.read_parquet(FACTORS, columns=["date", "stock_code"] + columns)
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.sort_values(["stock_code", "date"])
    events = attach_previous_snapshot(load_events(), factors)
    union = existing_union(events)

    quantiles = {}
    for col in ANCHORS_HIGH + ANCHORS_LOW + OTHER_HIGH + OTHER_LOW:
        if col not in factors:
            continue
        values = pd.to_numeric(factors[col], errors="coerce").dropna()
        quantiles[col] = {p: float(values.quantile(p)) for p in set(HIGH_PS + LOW_PS)}

    hi = lambda col, p: z(events, col) >= quantiles[col][p]
    lo = lambda col, p: z(events, col) <= quantiles[col][p]
    candidates = []

    # All pair/triple combinations among the three requested factors.
    for hp1 in HIGH_PS:
        for hp2 in HIGH_PS:
            candidates.append((
                "anchor_pair_lower_upper", hp1, hp2,
                hi("lower_shadow_absorption_5d", hp1) & hi("upper_shadow_pressure_5d", hp2),
            ))
    for high_factor in ANCHORS_HIGH:
        for hp in HIGH_PS:
            for lp in LOW_PS:
                candidates.append((
                    f"anchor_pair_{high_factor}_closeweak", hp, lp,
                    hi(high_factor, hp) & lo("close_to_high_3d_mean", lp),
                ))
    for hp1 in HIGH_PS:
        for hp2 in HIGH_PS:
            for lp in LOW_PS:
                candidates.append((
                    "anchor_triple", min(hp1, hp2), lp,
                    hi("lower_shadow_absorption_5d", hp1)
                    & hi("upper_shadow_pressure_5d", hp2)
                    & lo("close_to_high_3d_mean", lp),
                ))

    # Requested factors combined with other stable negative/positive-confirmation factors.
    for anchor in ANCHORS_HIGH:
        for other in OTHER_HIGH:
            if anchor not in quantiles or other not in quantiles:
                continue
            for hp in [.80, .90, .95]:
                candidates.append((
                    f"other_HH_{anchor}_{other}", hp, hp, hi(anchor, hp) & hi(other, hp)
                ))
        for support in OTHER_LOW:
            if anchor not in quantiles or support not in quantiles:
                continue
            for hp, lp in [(.80, .20), (.90, .10), (.95, .05)]:
                candidates.append((
                    f"other_HL_{anchor}_{support}", hp, lp, hi(anchor, hp) & lo(support, lp)
                ))
    for risk in OTHER_HIGH:
        if risk not in quantiles:
            continue
        for hp, lp in [(.80, .20), (.90, .10), (.95, .05)]:
            candidates.append((
                f"other_HL_{risk}_closeweak", hp, lp,
                hi(risk, hp) & lo("close_to_high_3d_mean", lp),
            ))

    rows = []
    for family, high_p, low_or_high_p, mask in candidates:
        for event_type in ("actual", "shadow", "combined"):
            subset = pd.Series(True, index=events.index) if event_type == "combined" else events["event_type"].eq(event_type)
            row = {
                "family": family, "high_p": high_p, "second_p": low_or_high_p,
                "event_type": event_type,
            }
            row.update(stats(events, mask & ~union & subset))
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "shadow_absorption_candidate_audit.csv", index=False, encoding="utf-8-sig")
    combined = result[result.event_type.eq("combined")].copy()
    combined["quality"] = (
        combined.big_loss * 5 + combined.small_loss
        - combined.big_gain_8 * 8 - combined.small_gain_3_8 * 2
    )
    good = combined[(combined.hits >= 2) & (combined.big_loss >= 1)].sort_values(
        ["quality", "big_loss", "big_gain_8", "hits"], ascending=[False, False, True, True]
    )
    print("QUANTILES")
    for col in ANCHORS_HIGH + ANCHORS_LOW:
        print(col, quantiles[col])
    print("\nTOP")
    print(good.head(60).to_string(index=False))
    print("\nZERO BIG GAIN")
    print(good[good.big_gain_8.eq(0)].head(60).to_string(index=False))


if __name__ == "__main__":
    main()
