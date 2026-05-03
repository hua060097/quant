from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
JSON_3D = BASE_DIR / "results" / "daily_top5_details_Window_3_20260320_012750.json"
JSON_2D = BASE_DIR / "results" / "daily_top5_details_Window_3_20260320_022143.json"

TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")

STATE_FACTORS = [
    "mkt_ma5_followthrough_v3",
    "ma5_strength_breadth_v3",
    "strong_stock_volume_confirmation_v3",
    "emotion_quality_balance_v3",
    "top10_10d_recent3_strength_v3",
    "top10_5d_recent3_strength_v3",
    "top10_3d_recent3_strength_v3",
    "sector_hotspot_concentration_v3",
]


def load_top_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        row = {"date": pd.to_datetime(ds)}
        for i in range(1, 6):
            item = daily.get(f"top{i}", {})
            row[f"top{i}"] = float(item.get("real_return")) if item.get("real_return") is not None else np.nan
        row["top12_avg"] = np.nanmean([row["top1"], row["top2"]])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_daily_factors() -> pd.DataFrame:
    df = pd.read_parquet(FACTOR_PATH, columns=["date"] + STATE_FACTORS)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False).first().sort_values("date").reset_index(drop=True)


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def tail_stats(x: pd.Series) -> dict[str, float]:
    x = pd.Series(x).dropna().astype(float)
    neg = x[x < 0]
    return {
        "days": int(len(x)),
        "mean": float(x.mean()) if len(x) else np.nan,
        "cum": float((1.0 + x).prod() - 1.0) if len(x) else np.nan,
        "mdd": max_drawdown(x),
        "neg_rate": float((x < 0).mean()) if len(x) else np.nan,
        "big5_rate": float((x <= -0.05).mean()) if len(x) else np.nan,
        "big8_rate": float((x <= -0.08).mean()) if len(x) else np.nan,
        "q10": float(x.quantile(0.1)) if len(x) else np.nan,
        "neg_mean": float(neg.mean()) if len(neg) else np.nan,
        "worst": float(x.min()) if len(x) else np.nan,
    }


def format_stats(prefix: str, stats: dict[str, float]) -> str:
    return (
        f"{prefix} mean={stats['mean']:.3%} cum={stats['cum']:.4f} mdd={stats['mdd']:.3%} "
        f"neg={stats['neg_rate']:.1%} <=-5={stats['big5_rate']:.1%} <=-8={stats['big8_rate']:.1%} "
        f"q10={stats['q10']:.3%} neg_mean={stats['neg_mean']:.3%} worst={stats['worst']:.3%}"
    )


def analyze_switch(df: pd.DataFrame, factor: str, train_daily: pd.DataFrame) -> None:
    thr = float(train_daily[factor].quantile(0.4))
    high = df[factor] >= thr
    low = ~high

    high_top1 = tail_stats(df.loc[high, "top1"])
    high_top2 = tail_stats(df.loc[high, "top2"])
    low_top1 = tail_stats(df.loc[low, "top1"])
    low_top2 = tail_stats(df.loc[low, "top2"])

    # Determine direction under the fixed q40 threshold
    high_choose_top1 = high_top1["mean"] >= high_top2["mean"]
    low_choose_top1 = low_top1["mean"] >= low_top2["mean"]

    if high_choose_top1:
        high_ret = df.loc[high, "top1"]
        high_label = "high->top1"
    else:
        high_ret = df.loc[high, "top2"]
        high_label = "high->top2"

    if low_choose_top1:
        low_ret = df.loc[low, "top1"]
        low_label = "low->top1"
    else:
        low_ret = df.loc[low, "top2"]
        low_label = "low->top2"

    combined = pd.concat([high_ret, low_ret]).sort_index()
    combined_stats = tail_stats(combined)

    print(f"[SWITCH] {factor} | q40 threshold={thr:.6f} | {high_label}, {low_label}")
    print("  " + format_stats("high_top1", high_top1))
    print("  " + format_stats("high_top2", high_top2))
    print("  " + format_stats("low_top1", low_top1))
    print("  " + format_stats("low_top2", low_top2))
    print("  " + format_stats("switch", combined_stats))


def analyze_position_control(df: pd.DataFrame, factor: str) -> None:
    q20 = df[factor].quantile(0.2)
    q80 = df[factor].quantile(0.8)
    low = df[df[factor] <= q20]
    high = df[df[factor] >= q80]
    full = tail_stats(df["top12_avg"])
    low_avg = tail_stats(low["top12_avg"])
    high_avg = tail_stats(high["top12_avg"])

    print(f"[POSITION] {factor} | q20={q20:.6f} q80={q80:.6f}")
    print("  " + format_stats("full_avg12", full))
    print("  " + format_stats("low20_avg12", low_avg))
    print("  " + format_stats("high20_avg12", high_avg))


def main() -> None:
    factors = load_daily_factors()
    train_daily = factors[(factors["date"] >= TRAIN_START) & (factors["date"] <= TRAIN_END)].copy()

    for label, json_path in [("3D_WEIGHTED", JSON_3D), ("2D_WEIGHTED", JSON_2D)]:
        top_df = load_top_json(json_path)
        df = top_df.merge(factors, on="date", how="inner").sort_values("date").reset_index(drop=True)
        print(f"=== {label} ===")
        print(f"days={len(df)} range={df['date'].min().date()} ~ {df['date'].max().date()}")
        print("BASE " + format_stats("top1", tail_stats(df["top1"])))
        print("BASE " + format_stats("top2", tail_stats(df["top2"])))
        print("BASE " + format_stats("avg12", tail_stats(df["top12_avg"])))
        print()

        for factor in STATE_FACTORS:
            analyze_switch(df, factor, train_daily)
            analyze_position_control(df, factor)
            print()


if __name__ == "__main__":
    main()
