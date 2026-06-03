import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
METRICS_162 = ROOT / "results" / "walkforward_metrics_20260512_194610.json"

SL_THRESHOLD = -0.18
RET20_ANCHOR = 0.40


def _load_162_trades() -> pd.DataFrame:
    metrics = json.loads(METRICS_162.read_text(encoding="utf-8"))[0]["all_results"]["1"]
    trades = pd.DataFrame(metrics["trades"])
    trades["buy_date"] = pd.to_datetime(trades["buy_datetime"]).dt.normalize()
    trades["actual_return_pct"] = pd.to_numeric(trades["actual_return"], errors="coerce") * 100.0
    return trades


def _load_factors() -> pd.DataFrame:
    columns = [
        "date",
        "stock_code",
        "runup_ma5_exhaustion",
        "cpv_factor",
        "prev_day_blowoff_risk",
        "sector_momentum_accel_rank",
        "sector_leadership_persistence_v3",
        "return_20d_intra_z",
        "return_10d_intra_z",
        "return_5d_intra_z",
        "sector_strong_leader_alpha",
        "sector_leader_capture_rank",
        "intra_sector_momentum_rank",
        "ma5_strength_breadth_v3",
        "strong_stock_volume_confirmation_v3",
        "sector_relative_strength_vs_market",
        "intra_sector_breadth",
    ]
    table = pq.read_table(
        str(FACTOR_PATH),
        columns=columns,
        filters=[
            ("date", ">=", pd.Timestamp("2025-08-01")),
            ("date", "<=", pd.Timestamp("2026-04-30")),
        ],
    )
    factors = table.to_pandas()
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    return factors


def _attach_prediction_date(trades: pd.DataFrame, factor_dates: np.ndarray) -> pd.DataFrame:
    def previous_factor_date(date_value):
        idx = np.searchsorted(
            factor_dates,
            np.datetime64(pd.Timestamp(date_value).normalize()),
            side="left",
        ) - 1
        return pd.Timestamp(factor_dates[idx]) if idx >= 0 else pd.NaT

    trades = trades.copy()
    trades["prediction_date"] = trades["buy_date"].map(previous_factor_date)
    return trades


def _summary(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    hit = df[mask.fillna(False)]
    return {
        "name": name,
        "n": len(hit),
        "avg": hit["actual_return_pct"].mean(),
        "median": hit["actual_return_pct"].median(),
        "big_win_ge5": int((hit["actual_return_pct"] >= 5).sum()),
        "big_loss_le5": int((hit["actual_return_pct"] <= -5).sum()),
        "min": hit["actual_return_pct"].min() if len(hit) else np.nan,
        "max": hit["actual_return_pct"].max() if len(hit) else np.nan,
    }


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    trades = _load_162_trades()
    factors = _load_factors()
    factor_dates = np.array(sorted(factors["date"].unique()))
    trades = _attach_prediction_date(trades, factor_dates)
    df = trades.merge(
        factors,
        left_on=["prediction_date", "stock_code"],
        right_on=["date", "stock_code"],
        how="left",
    )

    df["sector_leadership_break_no_alpha"] = (
        (-df["sector_leadership_persistence_v3"]).clip(lower=0.0)
        * (RET20_ANCHOR - df["return_20d_intra_z"]).clip(lower=0.0)
    )
    df["sector_leadership_break_no_alpha_032"] = (
        (-df["sector_leadership_persistence_v3"]).clip(lower=0.0)
        * (0.32 - df["return_20d_intra_z"]).clip(lower=0.0)
    )

    r1 = (df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)
    r2 = (df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30)
    old_r3 = df["sector_leadership_persistence_v3"] <= SL_THRESHOLD
    guarded_r3 = old_r3 & (df["return_20d_intra_z"] <= RET20_ANCHOR)

    _print_table(
        "Current Rule Comparison On 162 True Minute Returns",
        [
            _summary(df, r1, "R1"),
            _summary(df, r2, "R2"),
            _summary(df, old_r3, "old_R3_sl_only"),
            _summary(df, guarded_r3, "guarded_R3_sl_and_ret20"),
            _summary(df, r1 | r2 | old_r3, "old_any"),
            _summary(df, r1 | r2 | guarded_r3, "guarded_any"),
        ],
    )

    print("\n=== Guarded R3 Samples ===")
    print(
        df[guarded_r3.fillna(False)][
            [
                "stock_code",
                "prediction_date",
                "actual_return_pct",
                "sector_leadership_persistence_v3",
                "return_20d_intra_z",
                "sector_leadership_break_no_alpha",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n=== Composite Factor Quantiles In 162 Top1 Trades ===")
    for col in ["sector_leadership_break_no_alpha", "sector_leadership_break_no_alpha_032"]:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        nonzero = values[values > 0]
        print(f"\n{col}: nonzero={len(nonzero)}/{len(values)}")
        if len(nonzero):
            print(
                nonzero.quantile([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]).to_string(
                    float_format=lambda x: f"{x:.6f}"
                )
            )

    rows = []
    composite = df["sector_leadership_break_no_alpha"]
    positive_thresholds = [
        0.000001,
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.006,
        0.008,
        0.010,
        0.012,
        0.015,
        0.020,
    ]
    for threshold in positive_thresholds:
        comp_r3 = composite >= threshold
        any_mask = r1 | r2 | comp_r3
        hit = df[comp_r3.fillna(False)]
        rows.append(
            {
                "threshold": threshold,
                "r3_n": len(hit),
                "r3_avg": hit["actual_return_pct"].mean(),
                "r3_big_win": int((hit["actual_return_pct"] >= 5).sum()),
                "r3_big_loss": int((hit["actual_return_pct"] <= -5).sum()),
                "any_n": int(any_mask.fillna(False).sum()),
                "any_avg": df[any_mask.fillna(False)]["actual_return_pct"].mean(),
                "any_big_win": int((df[any_mask.fillna(False)]["actual_return_pct"] >= 5).sum()),
                "any_big_loss": int((df[any_mask.fillna(False)]["actual_return_pct"] <= -5).sum()),
            }
        )
    _print_table("Composite Threshold Sweep", rows)

    rows = []
    for threshold in [0.20, 0.25, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        ret_guard = old_r3 & (df["return_20d_intra_z"] <= threshold)
        any_mask = r1 | r2 | ret_guard
        hit = df[ret_guard.fillna(False)]
        rows.append(
            {
                "ret20_threshold": threshold,
                "r3_n": len(hit),
                "r3_avg": hit["actual_return_pct"].mean(),
                "r3_big_win": int((hit["actual_return_pct"] >= 5).sum()),
                "r3_big_loss": int((hit["actual_return_pct"] <= -5).sum()),
                "any_n": int(any_mask.fillna(False).sum()),
                "any_avg": df[any_mask.fillna(False)]["actual_return_pct"].mean(),
                "any_big_win": int((df[any_mask.fillna(False)]["actual_return_pct"] >= 5).sum()),
                "any_big_loss": int((df[any_mask.fillna(False)]["actual_return_pct"] <= -5).sum()),
            }
        )
    _print_table("Simple Return-Guard Threshold Sweep", rows)


if __name__ == "__main__":
    main()
