from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")
MKT = "mkt_ma5_followthrough_v3"
GATE = "max_rally_20d"


def summarize(s: pd.Series) -> dict[str, float]:
    s = pd.Series(s).dropna().astype(float)
    if s.empty:
        return {"days": 0, "mean": float("nan"), "cum": float("nan"), "mdd": float("nan"), "win": float("nan")}
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "days": int(len(s)),
        "mean": float(s.mean()),
        "cum": float(eq.iloc[-1] - 1.0),
        "mdd": float(dd.min()),
        "win": float((s > 0).mean()),
    }


def load_top_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in sorted(raw.items()):
        rows.append(
            {
                "date": pd.to_datetime(ds),
                "top1_ret": float(daily.get("top1", {}).get("real_return", np.nan)),
                "top2_ret": float(daily.get("top2", {}).get("real_return", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factor_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", MKT, GATE])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[[MKT, GATE]].first().sort_values("date").reset_index(drop=True)


def build_expanding_threshold(series: pd.Series, q: float, fallback: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def build_rolling_threshold(series: pd.Series, q: float, window: int, fallback: float) -> pd.Series:
    vals = []
    min_obs = min(window, 20)
    for i in range(len(series)):
        hist = series.iloc[max(0, i - window):i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= min_obs else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def eval_partitions(df: pd.DataFrame, scheme_cols: list[str]) -> pd.DataFrame:
    parts = {
        "full": df,
        "first_half": df.iloc[: len(df) // 2].copy(),
        "second_half": df.iloc[len(df) // 2 :].copy(),
        "added_10d": df[(df["date"] >= ADDED_START) & (df["date"] <= ADDED_END)].copy(),
    }
    rows = []
    for part_name, part in parts.items():
        for scheme in scheme_cols:
            rows.append({"partition": part_name, "scheme": scheme, **summarize(part[scheme])})
    return pd.DataFrame(rows)


def main() -> None:
    top_df = load_top_json(JSON_PATH)
    factor_df = load_factor_daily(FACTOR_PATH)
    df = top_df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

    train = factor_df[(factor_df["date"] >= TRAIN_START) & (factor_df["date"] <= TRAIN_END)]
    mkt_q30 = float(train[MKT].dropna().quantile(0.3))
    maxr_q50 = float(train[GATE].dropna().quantile(0.5))
    maxr_q60 = float(train[GATE].dropna().quantile(0.6))

    mkt_exp_q30 = build_expanding_threshold(df[MKT], 0.3, mkt_q30)
    maxr_exp_q50 = build_expanding_threshold(df[GATE], 0.5, maxr_q50)
    maxr_roll20_q60 = build_rolling_threshold(df[GATE], 0.6, 20, maxr_q60)

    mkt_choose_top1 = df[MKT] >= mkt_exp_q30
    maxr_low_exp_q50 = df[GATE] < maxr_exp_q50
    maxr_low_roll20_q60 = df[GATE] < maxr_roll20_q60

    df["scheme_top1"] = df["top1_ret"]
    df["scheme_mkt_exp_q30"] = np.where(mkt_choose_top1, df["top1_ret"], df["top2_ret"])
    df["scheme_maxr_low_top1_cash_exp_q50"] = np.where(maxr_low_exp_q50, df["top1_ret"], 0.0)
    df["scheme_maxr_low_top1_cash_roll20_q60"] = np.where(maxr_low_roll20_q60, df["top1_ret"], 0.0)
    df["scheme_mkt_then_maxr_exp_q50"] = np.where(
        mkt_choose_top1,
        np.where(maxr_low_exp_q50, df["top1_ret"], 0.0),
        df["top2_ret"],
    )
    df["scheme_mkt_then_maxr_roll20_q60"] = np.where(
        mkt_choose_top1,
        np.where(maxr_low_roll20_q60, df["top1_ret"], 0.0),
        df["top2_ret"],
    )

    schemes = [
        "scheme_top1",
        "scheme_mkt_exp_q30",
        "scheme_maxr_low_top1_cash_exp_q50",
        "scheme_maxr_low_top1_cash_roll20_q60",
        "scheme_mkt_then_maxr_exp_q50",
        "scheme_mkt_then_maxr_roll20_q60",
    ]

    part_df = eval_partitions(df, schemes)

    month_rows = []
    work = df.copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    for month, grp in work.groupby("month", sort=True):
        for scheme in schemes:
            month_rows.append({"month": month, "scheme": scheme, **summarize(grp[scheme])})
    month_df = pd.DataFrame(month_rows)

    pass_stats = []
    for name, mask in [
        ("scheme_maxr_low_top1_cash_exp_q50", maxr_low_exp_q50),
        ("scheme_maxr_low_top1_cash_roll20_q60", maxr_low_roll20_q60),
        ("scheme_mkt_then_maxr_exp_q50", mkt_choose_top1 & maxr_low_exp_q50),
        ("scheme_mkt_then_maxr_roll20_q60", mkt_choose_top1 & maxr_low_roll20_q60),
    ]:
        pass_stats.append(
            {
                "scheme": name,
                "pass_days": int(mask.sum()),
                "filter_days": int((~mask).sum()),
                "pass_rate": float(mask.mean()),
                "filter_rate": float((~mask).mean()),
            }
        )
    pass_df = pd.DataFrame(pass_stats)

    out_dir = ROOT / "results"
    part_path = out_dir / "mkt_with_maxrally_partition_20260413.csv"
    month_path = out_dir / "mkt_with_maxrally_monthly_20260413.csv"
    pass_path = out_dir / "mkt_with_maxrally_passrate_20260413.csv"
    part_df.to_csv(part_path, index=False, encoding="utf-8-sig")
    month_df.to_csv(month_path, index=False, encoding="utf-8-sig")
    pass_df.to_csv(pass_path, index=False, encoding="utf-8-sig")

    print("Partition comparison:")
    print(part_df.to_string(index=False))
    print()
    print("Pass/filter stats:")
    print(pass_df.to_string(index=False))
    print()
    print(f"Saved: {part_path}")
    print(f"Saved: {month_path}")
    print(f"Saved: {pass_path}")


if __name__ == "__main__":
    main()
