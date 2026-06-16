"""Discover constrained two-factor rules from stable factor-label correlations."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_correlation_rule_candidates import (
    BASE,
    FACTORS,
    RESULTS,
    EXISTING_COLS,
    attach_previous_snapshot,
    existing_union,
    load_events,
    stats,
    z,
)


def select_factors() -> tuple[list[str], list[str]]:
    stability = pd.read_csv(RESULTS / "factor_label_correlation_segment_stability.csv")
    negative = stability[
        (stability["negative_segment_count"] == 4)
        & (stability["same_sign_segment_ratio"] == 1)
        & (stability["overall_spearman_corr"] < 0)
    ].copy()
    positive = stability[
        (stability["positive_segment_count"] == 4)
        & (stability["same_sign_segment_ratio"] == 1)
        & (stability["overall_spearman_corr"] > 0)
    ].copy()
    negative["strength"] = negative["segment_spearman_abs_mean"]
    positive["strength"] = positive["segment_spearman_abs_mean"]
    return (
        negative.sort_values("strength", ascending=False)["factor"].head(25).tolist(),
        positive.sort_values("strength", ascending=False)["factor"].head(15).tolist(),
    )


def main() -> None:
    negative, positive = select_factors()
    available = set(pq.ParquetFile(FACTORS).schema_arrow.names)
    use_cols = [col for col in dict.fromkeys(EXISTING_COLS + negative + positive) if col in available]
    factors = pd.read_parquet(FACTORS, columns=["date", "stock_code"] + use_cols)
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.sort_values(["stock_code", "date"]).reset_index(drop=True)
    events = attach_previous_snapshot(load_events(), factors)
    union = existing_union(events)

    quantiles = {}
    for col in negative + positive:
        if col not in factors:
            continue
        values = pd.to_numeric(factors[col], errors="coerce").dropna()
        quantiles[col] = {p: float(values.quantile(p)) for p in (.05, .10, .20, .80, .90, .95)}

    rows = []
    candidates = []
    for left, right in itertools.combinations([c for c in negative if c in quantiles], 2):
        for p in (.80, .90, .95):
            candidates.append((
                f"HH:{left}>={p:.2f}&{right}>={p:.2f}",
                (z(events, left) >= quantiles[left][p]) & (z(events, right) >= quantiles[right][p]),
            ))
    for risk in [c for c in negative if c in quantiles]:
        for support in [c for c in positive if c in quantiles]:
            for high_p, low_p in ((.80, .20), (.90, .10), (.95, .05)):
                candidates.append((
                    f"HL:{risk}>={high_p:.2f}&{support}<={low_p:.2f}",
                    (z(events, risk) >= quantiles[risk][high_p]) & (z(events, support) <= quantiles[support][low_p]),
                ))

    for name, mask in candidates:
        for event_type in ("actual", "shadow", "combined"):
            subset = pd.Series(True, index=events.index) if event_type == "combined" else events["event_type"].eq(event_type)
            row = {"rule": name, "event_type": event_type}
            row.update(stats(events, mask & ~union & subset))
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "stable_correlation_two_factor_scan.csv", index=False, encoding="utf-8-sig")
    combined = result[result["event_type"] == "combined"].copy()
    combined["quality"] = (
        combined["big_loss"] * 5 + combined["small_loss"]
        - combined["big_gain_8"] * 8 - combined["small_gain_3_8"] * 2
    )
    good = combined[(combined["hits"] >= 2) & (combined["big_loss"] >= 1)].sort_values(
        ["quality", "big_loss", "big_gain_8", "hits"],
        ascending=[False, False, True, True],
    )
    print("NEGATIVE FACTORS", negative)
    print("POSITIVE SUPPORT FACTORS", positive)
    print("\nTOP")
    print(good.head(50).to_string(index=False))
    print("\nZERO BIG GAIN")
    print(good[good["big_gain_8"] == 0].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
