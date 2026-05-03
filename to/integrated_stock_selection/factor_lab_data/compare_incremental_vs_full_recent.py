"""Compare full-build vs incremental-build factor files on a recent date window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FULL_FILE = BASE_DIR / "factors_0422.parquet"
INCREMENTAL_FILE = BASE_DIR / "factors.parquet"
START_DATE = pd.Timestamp("2026-04-17")
END_DATE = pd.Timestamp("2026-04-21")
INDEX_COLS = ["stock_code", "date"]
ABS_DIFF_THRESHOLD = 1e-6
SIGNIFICANT_DIFF_THRESHOLD = 1e-4
SIGNIFICANT_ROW_RATIO = 0.01


def _load_filtered(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
    return df.loc[mask].copy()


def main() -> None:
    print("=" * 100)
    print("Compare full vs incremental factor files")
    print("=" * 100)
    print(f"Full file       : {FULL_FILE}")
    print(f"Incremental file: {INCREMENTAL_FILE}")
    print(f"Window          : {START_DATE.date()} ~ {END_DATE.date()}")

    full_df = _load_filtered(FULL_FILE)
    inc_df = _load_filtered(INCREMENTAL_FILE)

    print("\n[Filtered rows]")
    print(f"  Full       : {len(full_df):,}")
    print(f"  Incremental: {len(inc_df):,}")

    full_cols = set(full_df.columns) - set(INDEX_COLS)
    inc_cols = set(inc_df.columns) - set(INDEX_COLS)
    common_cols = sorted(full_cols & inc_cols)
    only_full = sorted(full_cols - inc_cols)
    only_inc = sorted(inc_cols - full_cols)

    print("\n[Columns]")
    print(f"  Common factor columns: {len(common_cols)}")
    if only_full:
        print(f"  Only in full       : {only_full}")
    if only_inc:
        print(f"  Only in incremental: {only_inc}")

    merged = full_df[INDEX_COLS + common_cols].merge(
        inc_df[INDEX_COLS + common_cols],
        on=INDEX_COLS,
        how="inner",
        suffixes=("_full", "_inc"),
    )

    print("\n[Aligned rows]")
    print(f"  Inner-joined rows: {len(merged):,}")

    factor_rows = []
    date_diff_accumulator: dict[pd.Timestamp, list[float]] = {}

    for factor in common_cols:
        full_col = pd.to_numeric(merged[f"{factor}_full"], errors="coerce")
        inc_col = pd.to_numeric(merged[f"{factor}_inc"], errors="coerce")
        valid_mask = ~(full_col.isna() | inc_col.isna())

        if valid_mask.sum() == 0:
            factor_rows.append(
                {
                    "factor": factor,
                    "valid_rows": 0,
                    "max_abs_diff": np.nan,
                    "mean_abs_diff": np.nan,
                    "median_abs_diff": np.nan,
                    "rows_diff_gt_1e6": 0,
                    "rows_diff_gt_1e4": 0,
                    "ratio_diff_gt_1e6": np.nan,
                    "ratio_diff_gt_1e4": np.nan,
                }
            )
            continue

        abs_diff = (inc_col[valid_mask] - full_col[valid_mask]).abs()
        date_series = merged.loc[valid_mask, "date"]

        for dt, diff_val in zip(date_series, abs_diff):
            date_diff_accumulator.setdefault(pd.Timestamp(dt), []).append(float(diff_val))

        factor_rows.append(
            {
                "factor": factor,
                "valid_rows": int(valid_mask.sum()),
                "max_abs_diff": float(abs_diff.max()),
                "mean_abs_diff": float(abs_diff.mean()),
                "median_abs_diff": float(abs_diff.median()),
                "rows_diff_gt_1e6": int((abs_diff > ABS_DIFF_THRESHOLD).sum()),
                "rows_diff_gt_1e4": int((abs_diff > SIGNIFICANT_DIFF_THRESHOLD).sum()),
                "ratio_diff_gt_1e6": float((abs_diff > ABS_DIFF_THRESHOLD).mean()),
                "ratio_diff_gt_1e4": float((abs_diff > SIGNIFICANT_DIFF_THRESHOLD).mean()),
            }
        )

    factor_diff_df = pd.DataFrame(factor_rows).sort_values(
        ["ratio_diff_gt_1e4", "max_abs_diff", "mean_abs_diff"], ascending=[False, False, False]
    )

    significant_factors = factor_diff_df[
        (factor_diff_df["ratio_diff_gt_1e4"] >= SIGNIFICANT_ROW_RATIO)
        | (factor_diff_df["max_abs_diff"] >= SIGNIFICANT_DIFF_THRESHOLD)
    ].copy()

    date_rows = []
    for dt, values in sorted(date_diff_accumulator.items()):
        arr = np.array(values, dtype=float)
        date_rows.append(
            {
                "date": dt,
                "mean_abs_diff": float(arr.mean()),
                "median_abs_diff": float(np.median(arr)),
                "max_abs_diff": float(arr.max()),
                "rows_diff_gt_1e6": int((arr > ABS_DIFF_THRESHOLD).sum()),
                "rows_diff_gt_1e4": int((arr > SIGNIFICANT_DIFF_THRESHOLD).sum()),
                "ratio_diff_gt_1e6": float((arr > ABS_DIFF_THRESHOLD).mean()),
                "ratio_diff_gt_1e4": float((arr > SIGNIFICANT_DIFF_THRESHOLD).mean()),
                "compared_values": int(arr.size),
            }
        )
    date_diff_df = pd.DataFrame(date_rows).sort_values(
        ["ratio_diff_gt_1e4", "max_abs_diff", "mean_abs_diff"], ascending=[False, False, False]
    )

    factor_out = BASE_DIR / "compare_incremental_vs_full_recent_factor_diff_20260420.csv"
    date_out = BASE_DIR / "compare_incremental_vs_full_recent_date_diff_20260420.csv"
    factor_diff_df.to_csv(factor_out, index=False, encoding="utf-8-sig")
    date_diff_df.to_csv(date_out, index=False, encoding="utf-8-sig")

    print("\n[Summary]")
    print(f"  Factors with any diff > 1e-6 : {(factor_diff_df['rows_diff_gt_1e6'] > 0).sum()}")
    print(f"  Factors with notable diff     : {len(significant_factors)}")

    if significant_factors.empty:
        print("  Conclusion: no significant difference found in the target window.")
    else:
        print("  Conclusion: significant differences still exist in the target window.")
        print("\n  Top factor differences:")
        print(
            significant_factors[
                ["factor", "max_abs_diff", "mean_abs_diff", "ratio_diff_gt_1e4", "rows_diff_gt_1e4"]
            ]
            .head(15)
            .to_string(index=False)
        )

        print("\n  Top date differences:")
        print(
            date_diff_df[
                ["date", "max_abs_diff", "mean_abs_diff", "ratio_diff_gt_1e4", "rows_diff_gt_1e4"]
            ]
            .head(15)
            .to_string(index=False)
        )

    print(f"\n[Saved]")
    print(f"  Factor diff: {factor_out}")
    print(f"  Date diff  : {date_out}")


if __name__ == "__main__":
    main()
