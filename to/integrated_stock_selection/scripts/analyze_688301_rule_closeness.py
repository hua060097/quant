from __future__ import annotations

import pandas as pd


cols = [
    "date", "stock_code",
    "reg_severe_up_10d_ratio",
    "prev_non_first_limit_up_performance_percentile_20d",
    "prev_first_limit_up_performance_percentile_20d",
    "market_60d_new_high_ratio",
    "market_advance_decline_ratio",
    "top10_10d_strength_level_v3",
    "market_liquidity_change",
    "mkt_dispersion_shock_v2",
    "runup_ma5_exhaustion",
    "cpv_factor",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
    "sector_leadership_persistence_v3",
    "return_20d_intra_z",
    "ulcer_index",
    "momentum_peak_decay_3d",
    "intra_sector_momentum_rank",
    "limit_down_3d",
    "buy_sm_amount_rate",
    "kdj_j_9",
    "turnover_z_20",
    "upper_shadow_pressure_5d",
    "high_position_volume_surge",
    "high_volume_no_gain_3d",
    "shadow_reversal_volume_pressure",
    "close_below_ma5_after_runup",
    "low_break_prev_low_after_runup",
    "max_rally_10d",
    "max_rally_20d",
    "price_ma5_ratio",
    "distance_from_52w_high",
    "close_position_20d",
    "close_position_60d",
]


def main() -> None:
    df = pd.read_parquet("factor_lab_data/factors.parquet", columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    row = df[(df.stock_code == "688301.SH") & (df.date == pd.Timestamp("2026-05-29"))].iloc[0]

    rules = {
        "r1": row.runup_ma5_exhaustion > 0.8 and row.cpv_factor <= -0.4,
        "r2": (
            (row.prev_day_blowoff_risk >= 2 and row.sector_momentum_accel_rank <= 0.30)
            or (row.prev_day_blowoff_risk > 5 and row.sector_momentum_accel_rank <= 0.50)
            or (row.prev_day_blowoff_risk > 1 and row.sector_momentum_accel_rank <= 0.15)
        ),
        "r3": row.sector_leadership_persistence_v3 <= -0.18 and row.return_20d_intra_z <= 0.40,
        "r4": row.market_liquidity_change < 0.94 and (
            row.ulcer_index < -0.80
            or row.momentum_peak_decay_3d > 1.0
            or row.intra_sector_momentum_rank < 0.67
        ),
        "r5": row.mkt_dispersion_shock_v2 < 0.98 and row.market_advance_decline_ratio < 0.38,
        "r6": row.limit_down_3d == 1,
        "r7": row.top10_10d_strength_level_v3 >= 0.70 and row.buy_sm_amount_rate <= -1.25,
        "r8": row.prev_non_first_limit_up_performance_percentile_20d >= 0.80 and row.kdj_j_9 <= 0.75,
    }
    print("RULES")
    print(rules)
    print("\nKEY_VALUES")
    keys = [c for c in cols if c not in {"date", "stock_code"}]
    for key in keys:
        print(f"{key}: {float(row[key]):.6f}")

    print("\nCROSS_SECTION_RANKS_20260529")
    same = df[df.date == pd.Timestamp("2026-05-29")]
    for c in [
        "reg_severe_up_10d_ratio",
        "prev_non_first_limit_up_performance_percentile_20d",
        "market_60d_new_high_ratio",
        "runup_ma5_exhaustion",
        "kdj_j_9",
        "turnover_z_20",
        "upper_shadow_pressure_5d",
        "high_position_volume_surge",
        "high_volume_no_gain_3d",
        "shadow_reversal_volume_pressure",
        "max_rally_10d",
        "max_rally_20d",
        "price_ma5_ratio",
        "distance_from_52w_high",
        "close_position_20d",
    ]:
        s = same[c].dropna()
        rank = (s <= row[c]).mean()
        print(
            f"{c}: value={float(row[c]):.6f}, rank={rank:.4f}, "
            f"q95={float(s.quantile(.95)):.6f}, q99={float(s.quantile(.99)):.6f}"
        )


if __name__ == "__main__":
    main()
