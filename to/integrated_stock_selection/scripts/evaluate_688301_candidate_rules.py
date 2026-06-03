from __future__ import annotations

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from audit_reg_severe_up_10d_ratio import (
    FACTORS,
    SELECTIONS_FILE,
    SHADOW_FILE,
    RULE_COLS,
    add_prev_factor_snapshot,
    bucket_counts,
    current_v3_masks,
    load_json_records,
    normalize_events,
)


EXTRA_COLS = [
    "market_60d_new_high_ratio",
    "upper_shadow_pressure_5d",
    "high_volume_no_gain_3d",
    "shadow_reversal_volume_pressure",
    "max_rally_10d",
    "max_rally_20d",
    "price_ma5_ratio",
    "distance_from_52w_high",
    "close_position_20d",
    "close_position_60d",
]


def eval_rule(df: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    mask = mask.fillna(False)
    inc = mask & ~df["current_v3"].fillna(False)
    rows = df[inc]
    return {
        "rule": name,
        **bucket_counts(rows),
        "actual_big_loss": int(((rows["source"] == "actual") & (rows["actual_return"] <= -0.04)).sum()),
        "actual_big_gain": int(((rows["source"] == "actual") & (rows["actual_return"] >= 0.08)).sum()),
        "shadow_big_loss": int(((rows["source"] == "shadow") & (rows["actual_return"] <= -0.04)).sum()),
        "shadow_big_gain": int(((rows["source"] == "shadow") & (rows["actual_return"] >= 0.08)).sum()),
        "examples": rows[
            [
                "source", "date", "stock_code", "actual_return",
                "reg_severe_up_10d_ratio", "market_60d_new_high_ratio",
                "upper_shadow_pressure_5d", "high_volume_no_gain_3d",
                "shadow_reversal_volume_pressure", "max_rally_10d",
                "price_ma5_ratio", "prev_day_blowoff_risk", "sector_momentum_accel_rank",
            ]
        ].sort_values("actual_return").head(12).to_dict("records"),
        "big_gain_examples": rows[rows["actual_return"] >= 0.08][
            [
                "source", "date", "stock_code", "actual_return",
                "reg_severe_up_10d_ratio", "market_60d_new_high_ratio",
                "upper_shadow_pressure_5d", "high_volume_no_gain_3d",
                "shadow_reversal_volume_pressure", "max_rally_10d",
                "price_ma5_ratio", "prev_day_blowoff_risk", "sector_momentum_accel_rank",
            ]
        ].sort_values("actual_return", ascending=False).head(12).to_dict("records"),
    }


def main() -> None:
    selections = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(
        ["source", "date", "stock_code", "actual_return"]
    )
    schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
    factor_cols = sorted(set(["date", "stock_code", *[c for c in [*RULE_COLS, *EXTRA_COLS] if c in schema_cols]]))
    factors = pd.read_parquet(FACTORS, columns=factor_cols)
    for col in [*RULE_COLS, *EXTRA_COLS]:
        if col not in factors.columns:
            factors[col] = np.nan
    factors["date"] = pd.to_datetime(factors["date"])
    events = add_prev_factor_snapshot(events, factors)
    events = pd.concat([events, current_v3_masks(events)], axis=1)

    rules = {
        "reg>=0.35 & market_60d_new_high_ratio<=0.03": (
            (events["reg_severe_up_10d_ratio"] >= 0.35)
            & (events["market_60d_new_high_ratio"] <= 0.03)
        ),
        "reg>=0.35 & upper_shadow_pressure_5d>=5": (
            (events["reg_severe_up_10d_ratio"] >= 0.35)
            & (events["upper_shadow_pressure_5d"] >= 5)
        ),
        "reg>=0.35 & high_volume_no_gain_3d>=5": (
            (events["reg_severe_up_10d_ratio"] >= 0.35)
            & (events["high_volume_no_gain_3d"] >= 5)
        ),
        "reg>=0.35 & shadow_reversal_volume_pressure>=1.5": (
            (events["reg_severe_up_10d_ratio"] >= 0.35)
            & (events["shadow_reversal_volume_pressure"] >= 1.5)
        ),
        "reg>=0.35 & max_rally_10d>=5": (
            (events["reg_severe_up_10d_ratio"] >= 0.35)
            & (events["max_rally_10d"] >= 5)
        ),
        "reg>=0.35 & price_ma5_ratio>=2": (
            (events["reg_severe_up_10d_ratio"] >= 0.35)
            & (events["price_ma5_ratio"] >= 2)
        ),
        "r2_relaxed: blowoff>1 & sector_accel<=0.36": (
            (events["prev_day_blowoff_risk"] > 1)
            & (events["sector_momentum_accel_rank"] <= 0.36)
        ),
        "r2_relaxed_plus_reg: reg>=0.35 & blowoff>1 & sector_accel<=0.36": (
            (events["reg_severe_up_10d_ratio"] >= 0.35)
            & (events["prev_day_blowoff_risk"] > 1)
            & (events["sector_momentum_accel_rank"] <= 0.36)
        ),
        "r2_relaxed_plus_shadow: upper_shadow>=5 & blowoff>1 & sector_accel<=0.36": (
            (events["upper_shadow_pressure_5d"] >= 5)
            & (events["prev_day_blowoff_risk"] > 1)
            & (events["sector_momentum_accel_rank"] <= 0.36)
        ),
    }
    rows = [eval_rule(events, name, mask) for name, mask in rules.items()]
    out = pd.DataFrame(rows)
    show_cols = [
        "rule", "n", "mean", "big_loss<=-4%", "small_loss", "small_gain3-8%",
        "big_gain>=8%", "big_gain>=10%", "actual_big_loss", "actual_big_gain",
        "shadow_big_loss", "shadow_big_gain",
    ]
    print(out[show_cols].to_string(index=False))
    for row in rows:
        print("\nDETAIL", row["rule"])
        print("examples", row["examples"])
        print("big_gain_examples", row["big_gain_examples"])


if __name__ == "__main__":
    main()
