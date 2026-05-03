"""Analyze root cause of incremental vs full differences for 52w factors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FULL_FILE = BASE_DIR / "factors_原始.parquet"
INC_FILE = BASE_DIR / "factors.parquet"

TARGET_START = pd.Timestamp("2026-01-01")
TARGET_END = pd.Timestamp("2026-04-17")
LOOKBACK_TRADING_DAYS = 250
SIGNIFICANT_DIFF = 1e-4

FACTORS = ["distance_from_52w_high", "distance_from_52w_low"]


def _load_for_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["stock_code", "date"] + FACTORS
    full = pd.read_parquet(FULL_FILE, columns=cols)
    inc = pd.read_parquet(INC_FILE, columns=cols)
    for df in (full, inc):
        df["date"] = pd.to_datetime(df["date"])
    return full, inc


def _build_aligned_target(full: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    full_t = full[(full["date"] >= TARGET_START) & (full["date"] <= TARGET_END)].copy()
    inc_t = inc[(inc["date"] >= TARGET_START) & (inc["date"] <= TARGET_END)].copy()
    merged = full_t.merge(inc_t, on=["stock_code", "date"], suffixes=("_full", "_inc"), how="inner")
    for factor in FACTORS:
        merged[f"{factor}_abs_diff"] = (
            pd.to_numeric(merged[f"{factor}_inc"], errors="coerce")
            - pd.to_numeric(merged[f"{factor}_full"], errors="coerce")
        ).abs()
    return merged


def _trading_calendar(full: pd.DataFrame) -> list[pd.Timestamp]:
    cal = sorted(full.loc[full["date"] < TARGET_START, "date"].drop_duplicates().tolist())
    return cal


def _calc_window_start(cal: list[pd.Timestamp], target_start: pd.Timestamp, lookback_days: int) -> pd.Timestamp:
    idx = max(i for i, dt in enumerate(cal) if dt < target_start)
    start_idx = max(0, idx - lookback_days + 1)
    return cal[start_idx]


def _history_stats(full: pd.DataFrame, window_start: pd.Timestamp) -> pd.DataFrame:
    history = full[(full["date"] >= window_start) & (full["date"] < TARGET_START)][["stock_code", "date"]].copy()
    if history.empty:
        return pd.DataFrame(columns=["stock_code", "history_rows", "first_hist_date", "last_hist_date", "missing_from_250", "max_gap_days", "median_gap_days"])

    history = history.sort_values(["stock_code", "date"])
    rows = []
    for stock_code, g in history.groupby("stock_code", sort=False):
        dates = g["date"].tolist()
        gaps = np.diff(pd.Series(dates).view("int64")) / (24 * 3600 * 1e9)
        rows.append(
            {
                "stock_code": stock_code,
                "history_rows": len(dates),
                "first_hist_date": dates[0],
                "last_hist_date": dates[-1],
                "missing_from_250": max(0, LOOKBACK_TRADING_DAYS - len(dates)),
                "max_gap_days": float(np.max(gaps)) if len(gaps) else 0.0,
                "median_gap_days": float(np.median(gaps)) if len(gaps) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    full, inc = _load_for_analysis()
    aligned = _build_aligned_target(full, inc)

    cal = _trading_calendar(full)
    window_start = _calc_window_start(cal, TARGET_START, LOOKBACK_TRADING_DAYS)
    hist_stats = _history_stats(full, window_start)

    stock_rows = []
    for factor in FACTORS:
        diff_col = f"{factor}_abs_diff"
        sig = aligned[aligned[diff_col] > SIGNIFICANT_DIFF].copy()

        stock_summary = (
            sig.groupby("stock_code")
            .agg(
                diff_rows=("stock_code", "size"),
                max_abs_diff=(diff_col, "max"),
                mean_abs_diff=(diff_col, "mean"),
                first_diff_date=("date", "min"),
                last_diff_date=("date", "max"),
            )
            .reset_index()
        )
        stock_summary["factor"] = factor
        stock_summary = stock_summary.merge(hist_stats, on="stock_code", how="left")
        stock_summary["history_rows"] = stock_summary["history_rows"].fillna(0).astype(int)
        stock_summary["missing_from_250"] = stock_summary["missing_from_250"].fillna(LOOKBACK_TRADING_DAYS).astype(int)
        stock_summary["insufficient_history"] = stock_summary["history_rows"] < LOOKBACK_TRADING_DAYS
        stock_rows.append(stock_summary)

    stock_diff_df = pd.concat(stock_rows, ignore_index=True).sort_values(
        ["factor", "diff_rows", "max_abs_diff"], ascending=[True, False, False]
    )

    factor_summary_rows = []
    for factor in FACTORS:
        tmp = stock_diff_df[stock_diff_df["factor"] == factor]
        factor_summary_rows.append(
            {
                "factor": factor,
                "diff_stock_count": tmp["stock_code"].nunique(),
                "diff_row_count": int(tmp["diff_rows"].sum()),
                "stocks_with_insufficient_history": int(tmp.loc[tmp["insufficient_history"], "stock_code"].nunique()),
                "ratio_insufficient_history_stocks": float(tmp.loc[tmp["insufficient_history"], "stock_code"].nunique() / max(1, tmp["stock_code"].nunique())),
                "median_history_rows": float(tmp["history_rows"].median()) if not tmp.empty else np.nan,
                "min_history_rows": int(tmp["history_rows"].min()) if not tmp.empty else np.nan,
                "max_missing_from_250": int(tmp["missing_from_250"].max()) if not tmp.empty else np.nan,
            }
        )
    factor_summary_df = pd.DataFrame(factor_summary_rows)

    date_rows = []
    for factor in FACTORS:
        diff_col = f"{factor}_abs_diff"
        tmp = aligned[aligned[diff_col] > SIGNIFICANT_DIFF]
        by_date = (
            tmp.groupby("date")
            .agg(
                diff_rows=("stock_code", "size"),
                stock_count=("stock_code", "nunique"),
                max_abs_diff=(diff_col, "max"),
                mean_abs_diff=(diff_col, "mean"),
            )
            .reset_index()
        )
        by_date["factor"] = factor
        date_rows.append(by_date)
    date_df = pd.concat(date_rows, ignore_index=True).sort_values(
        ["factor", "diff_rows", "max_abs_diff"], ascending=[True, False, False]
    )

    stock_path = BASE_DIR / "analyze_52w_diff_root_cause_stock_20260420.csv"
    factor_path = BASE_DIR / "analyze_52w_diff_root_cause_summary_20260420.csv"
    date_path = BASE_DIR / "analyze_52w_diff_root_cause_date_20260420.csv"
    stock_diff_df.to_csv(stock_path, index=False, encoding="utf-8-sig")
    factor_summary_df.to_csv(factor_path, index=False, encoding="utf-8-sig")
    date_df.to_csv(date_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("52w factor diff root cause analysis")
    print("=" * 100)
    print(f"History window start used for diagnosis: {window_start.date()} -> {TARGET_START.date()} (exclusive)")
    print("\n[Factor summary]")
    print(factor_summary_df.to_string(index=False))
    print("\n[Top affected stocks]")
    print(
        stock_diff_df[
            [
                "factor",
                "stock_code",
                "diff_rows",
                "max_abs_diff",
                "history_rows",
                "missing_from_250",
                "insufficient_history",
                "first_diff_date",
                "last_diff_date",
                "max_gap_days",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
    print(f"\nSaved:\n  {factor_path}\n  {stock_path}\n  {date_path}")


if __name__ == "__main__":
    main()
