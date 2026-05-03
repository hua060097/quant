from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_010230.json"

TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")

MKT = "mkt_ma5_followthrough_v3"
NH = "market_60d_new_high_ratio"


def max_drawdown(returns: pd.Series) -> float:
    s = pd.Series(returns).fillna(0.0).astype(float)
    if s.empty:
        return float("nan")
    equity = (1.0 + s).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def summarize(returns: pd.Series) -> dict[str, float]:
    s = pd.Series(returns).dropna().astype(float)
    if s.empty:
        return {"days": 0, "mean": float("nan"), "cum": float("nan"), "mdd": float("nan"), "win": float("nan")}
    return {
        "days": int(len(s)),
        "mean": float(s.mean()),
        "cum": float((1.0 + s).prod() - 1.0),
        "mdd": max_drawdown(s),
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


def load_factors(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", MKT, NH])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[[MKT, NH]].first().sort_values("date").reset_index(drop=True)


def train_quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q))


def build_expanding_threshold(series: pd.Series, q: float, fallback: float) -> pd.Series:
    out = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        out.append(float(hist.quantile(q)) if len(hist) >= 20 else fallback)
    return pd.Series(out, index=series.index, dtype="float64")


def choose_single(df: pd.DataFrame, factor: str, threshold: pd.Series | float) -> pd.Series:
    if not isinstance(threshold, pd.Series):
        threshold = pd.Series(threshold, index=df.index, dtype="float64")
    high = df[factor] >= threshold
    return pd.Series(np.where(high, df["top1_ret"], df["top2_ret"]), index=df.index, dtype="float64")


def choose_dual(df: pd.DataFrame, mkt_threshold: float, nh_threshold: float) -> pd.Series:
    mkt_high = df[MKT] >= mkt_threshold
    nh_high = df[NH] >= nh_threshold
    choose_top1 = mkt_high & nh_high
    return pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index, dtype="float64")


def add_scheme_returns(df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(factors, on="date", how="inner").sort_values("date").reset_index(drop=True)
    train = factors[(factors["date"] >= TRAIN_START) & (factors["date"] <= TRAIN_END)]

    mkt_q40 = train_quantile(train[MKT].dropna(), 0.4)
    nh_q70 = train_quantile(train[NH].dropna(), 0.7)
    mkt_exp_q30 = build_expanding_threshold(merged[MKT], q=0.3, fallback=train_quantile(train[MKT].dropna(), 0.3))

    merged["scheme_top1"] = merged["top1_ret"]
    merged["scheme_mkt_q40"] = choose_single(merged, MKT, mkt_q40)
    merged["scheme_mkt_exp_q30"] = choose_single(merged, MKT, mkt_exp_q30)
    merged["scheme_dual"] = choose_dual(merged, mkt_q40, nh_q70)
    return merged


def by_half(df: pd.DataFrame, scheme_cols: list[str]) -> pd.DataFrame:
    mid = len(df) // 2
    pieces = {
        "full": df,
        "first_half": df.iloc[:mid].copy(),
        "second_half": df.iloc[mid:].copy(),
    }
    rows = []
    for period, part in pieces.items():
        for scheme in scheme_cols:
            rows.append({"period": period, "scheme": scheme, **summarize(part[scheme])})
    return pd.DataFrame(rows)


def by_month(df: pd.DataFrame, scheme_cols: list[str]) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    for month, grp in work.groupby("month", sort=True):
        for scheme in scheme_cols:
            rows.append({"month": month, "scheme": scheme, **summarize(grp[scheme])})
    return pd.DataFrame(rows)


def rolling_window_stats(df: pd.DataFrame, scheme_cols: list[str], window: int = 20) -> pd.DataFrame:
    rows = []
    for start in range(0, len(df) - window + 1):
        part = df.iloc[start:start + window]
        label = f"{part['date'].iloc[0].date()}~{part['date'].iloc[-1].date()}"
        for scheme in scheme_cols:
            rows.append({"window": label, "scheme": scheme, **summarize(part[scheme])})
    return pd.DataFrame(rows)


def stability_summary(month_df: pd.DataFrame, roll_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, grp in month_df.groupby("scheme"):
        roll = roll_df[roll_df["scheme"] == scheme]
        rows.append(
            {
                "scheme": scheme,
                "monthly_positive_month_ratio": float((grp["cum"] > 0).mean()),
                "monthly_mean_of_mean": float(grp["mean"].mean()),
                "monthly_std_of_mean": float(grp["mean"].std()),
                "monthly_worst_cum": float(grp["cum"].min()),
                "rolling20_positive_ratio": float((roll["cum"] > 0).mean()),
                "rolling20_mean_cum": float(roll["cum"].mean()),
                "rolling20_std_cum": float(roll["cum"].std()),
                "rolling20_worst_cum": float(roll["cum"].min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    top = load_top_json(JSON_PATH)
    factors = load_factors(FACTOR_PATH)
    df = add_scheme_returns(top, factors)

    schemes = ["scheme_top1", "scheme_mkt_q40", "scheme_mkt_exp_q30", "scheme_dual"]
    half_df = by_half(df, schemes)
    month_df = by_month(df, schemes)
    roll_df = rolling_window_stats(df, schemes, window=20)
    stab_df = stability_summary(month_df, roll_df)

    out_dir = ROOT / "results"
    half_csv = out_dir / "gate_scheme_half_compare_20260413.csv"
    month_csv = out_dir / "gate_scheme_monthly_compare_20260413.csv"
    roll_csv = out_dir / "gate_scheme_rolling20_compare_20260413.csv"
    stab_csv = out_dir / "gate_scheme_stability_summary_20260413.csv"
    half_df.to_csv(half_csv, index=False, encoding="utf-8-sig")
    month_df.to_csv(month_csv, index=False, encoding="utf-8-sig")
    roll_df.to_csv(roll_csv, index=False, encoding="utf-8-sig")
    stab_df.to_csv(stab_csv, index=False, encoding="utf-8-sig")

    print("Half-period comparison:")
    print(half_df.to_string(index=False))
    print()
    print("Stability summary:")
    print(stab_df.sort_values("rolling20_positive_ratio", ascending=False).to_string(index=False))
    print()
    print(f"Saved: {half_csv}")
    print(f"Saved: {month_csv}")
    print(f"Saved: {roll_csv}")
    print(f"Saved: {stab_csv}")


if __name__ == "__main__":
    main()
