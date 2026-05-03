from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
JSON_PATH = BASE_DIR / "results" / "daily_top5_details_Window_3_20260325_230521.json"

TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
MIN_EXPANDING_HISTORY = 20
CALIBRATION_RATIO = 0.6
SIGNAL = "mkt_ma5_followthrough_v3"
Q = 0.3


def max_drawdown(returns: pd.Series) -> float:
    x = pd.Series(returns).dropna().astype(float)
    if x.empty:
        return float("nan")
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def summarize(returns: pd.Series) -> dict[str, float]:
    x = pd.Series(returns).dropna().astype(float)
    if x.empty:
        return {
            "days": 0,
            "mean": np.nan,
            "cum": np.nan,
            "win": np.nan,
            "mdd": np.nan,
        }
    return {
        "days": int(len(x)),
        "mean": float(x.mean()),
        "cum": float((1.0 + x).prod() - 1.0),
        "win": float((x > 0).mean()),
        "mdd": max_drawdown(x),
    }


def load_top5_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        row = {"date": pd.to_datetime(ds)}
        for k in range(1, 6):
            info = daily[f"top{k}"]
            row[f"ret{k}"] = float(info["real_return"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_daily_factors() -> pd.DataFrame:
    df = pd.read_parquet(FACTOR_PATH, columns=["date", SIGNAL])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False).first().sort_values("date").reset_index(drop=True)


def threshold_series_expanding(df: pd.DataFrame, signal: str, q: float) -> pd.Series:
    s = df[signal].astype(float)
    return s.expanding(min_periods=MIN_EXPANDING_HISTORY).quantile(q).shift(1)


def print_stats(name: str, stats: dict[str, float]) -> None:
    print(
        f"{name}: days={stats['days']} mean={stats['mean']:.4%} "
        f"cum={stats['cum']:.2%} win={stats['win']:.2%} mdd={stats['mdd']:.2%}"
    )


def analyze_base(df: pd.DataFrame, base_name: str, base_returns: pd.Series, favorable: pd.Series) -> None:
    split_idx = int(len(df) * CALIBRATION_RATIO)
    gate_returns = base_returns.where(favorable, 0.0)

    print(f"\n=== {base_name} ===")
    print_stats("baseline_full", summarize(base_returns))
    print_stats("gate_full", summarize(gate_returns))
    print_stats("baseline_cal", summarize(base_returns.iloc[:split_idx]))
    print_stats("gate_cal", summarize(gate_returns.iloc[:split_idx]))
    print_stats("baseline_val", summarize(base_returns.iloc[split_idx:]))
    print_stats("gate_val", summarize(gate_returns.iloc[split_idx:]))

    active = df.loc[favorable, "date"]
    print(
        f"gate_active_days={int(favorable.sum())}/{len(favorable)} "
        f"({float(favorable.mean()):.2%}) first={active.min().date() if not active.empty else 'NA'} "
        f"last={active.max().date() if not active.empty else 'NA'}"
    )


def main() -> None:
    top = load_top5_json(JSON_PATH)
    factors = load_daily_factors()
    df = top.merge(factors, on="date", how="left").sort_values("date").reset_index(drop=True)

    thresholds = threshold_series_expanding(df, SIGNAL, Q)
    favorable = df[SIGNAL].astype(float) >= thresholds

    print(f"SIGNAL={SIGNAL} rule=expanding q{int(Q * 100)} high")
    print(f"days={len(df)} range={df['date'].min().date()}~{df['date'].max().date()}")

    analyze_base(df, "top1", df["ret1"], favorable)
    analyze_base(df, "top2", df["ret2"], favorable)

    # Also show the conditional quality of allowed vs skipped samples.
    for base_name, col in [("top1", "ret1"), ("top2", "ret2")]:
        allowed = summarize(df.loc[favorable, col])
        skipped = summarize(df.loc[~favorable, col])
        print(f"\n[{base_name}] conditional")
        print_stats("allowed_only", allowed)
        print_stats("skipped_only", skipped)


if __name__ == "__main__":
    main()
