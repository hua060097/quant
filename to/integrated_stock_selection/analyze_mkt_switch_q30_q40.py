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
SIGNAL = "mkt_ma5_followthrough_v3"
MIN_EXPANDING_HISTORY = 20
CALIBRATION_RATIO = 0.6


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
        return {"days": 0, "mean": np.nan, "cum": np.nan, "win": np.nan, "mdd": np.nan}
    return {
        "days": int(len(x)),
        "mean": float(x.mean()),
        "cum": float((1.0 + x).prod() - 1.0),
        "win": float((x > 0).mean()),
        "mdd": max_drawdown(x),
    }


def print_stats(name: str, stats: dict[str, float]) -> None:
    print(
        f"{name}: days={stats['days']} mean={stats['mean']:.4%} "
        f"cum={stats['cum']:.2%} win={stats['win']:.2%} mdd={stats['mdd']:.2%}"
    )


def load_top5_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        row = {"date": pd.to_datetime(ds)}
        for k in range(1, 6):
            info = daily[f"top{k}"]
            row[f"ret{k}"] = float(info["real_return"])
            row[f"pred{k}"] = float(info["pred"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_daily_signal() -> pd.DataFrame:
    df = pd.read_parquet(FACTOR_PATH, columns=["date", SIGNAL])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False).first().sort_values("date").reset_index(drop=True)


def threshold_series_expanding(df: pd.DataFrame, q: float) -> pd.Series:
    s = df[SIGNAL].astype(float)
    return s.expanding(min_periods=MIN_EXPANDING_HISTORY).quantile(q).shift(1)


def threshold_series_fixed_train(df: pd.DataFrame, train_daily: pd.DataFrame, q: float) -> pd.Series:
    thr = float(train_daily[SIGNAL].quantile(q))
    return pd.Series(thr, index=df.index, dtype=float)


def eval_series(name: str, returns: pd.Series, split_idx: int) -> None:
    print(f"\n=== {name} ===")
    print_stats("full", summarize(returns))
    print_stats("cal", summarize(returns.iloc[:split_idx]))
    print_stats("val", summarize(returns.iloc[split_idx:]))


def main() -> None:
    top = load_top5_json(JSON_PATH)
    signal_df = load_daily_signal()
    df = top.merge(signal_df, on="date", how="left").sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * CALIBRATION_RATIO)

    train_daily = signal_df[(signal_df["date"] >= TRAIN_START) & (signal_df["date"] <= TRAIN_END)].copy()
    q40_fixed = threshold_series_fixed_train(df, train_daily, 0.4)
    q30_exp = threshold_series_expanding(df, 0.3)

    top1 = df["ret1"]
    top2 = df["ret2"]

    # Reverse gate check for Top2: low state allowed
    top2_gate_reverse = top2.where(df[SIGNAL] < q30_exp, 0.0)
    top2_gate_forward = top2.where(df[SIGNAL] >= q30_exp, 0.0)

    print(f"days={len(df)} range={df['date'].min().date()}~{df['date'].max().date()}")
    print(f"signal={SIGNAL}")

    eval_series("top2_baseline", top2, split_idx)
    eval_series("top2_gate_reverse_exp_q30_low", top2_gate_reverse, split_idx)
    eval_series("top2_gate_forward_exp_q30_high", top2_gate_forward, split_idx)

    # Switch candidates
    switch_q40 = pd.Series(np.where(df[SIGNAL] >= q40_fixed, df["ret1"], df["ret2"]), index=df.index)
    switch_q30 = pd.Series(np.where(df[SIGNAL] >= q30_exp, df["ret1"], df["ret2"]), index=df.index)

    # For reference, three-way interpretation:
    # high >= q40_fixed -> top1, low < q30_exp -> top2, middle -> no trade
    three_way = pd.Series(0.0, index=df.index, dtype=float)
    high_mask = df[SIGNAL] >= q40_fixed
    low_mask = df[SIGNAL] < q30_exp
    three_way.loc[high_mask] = df.loc[high_mask, "ret1"]
    three_way.loc[low_mask] = df.loc[low_mask, "ret2"]

    eval_series("switch_fixed_q40", switch_q40, split_idx)
    eval_series("switch_expanding_q30", switch_q30, split_idx)
    eval_series("three_way_q30_gate_q40_switch", three_way, split_idx)

    # Conditional bucket summary for q30 split.
    high = df[SIGNAL] >= q30_exp
    low = df[SIGNAL] < q30_exp
    print("\n=== conditional by expanding q30 split ===")
    print_stats("high_top1", summarize(df.loc[high, "ret1"]))
    print_stats("high_top2", summarize(df.loc[high, "ret2"]))
    print_stats("low_top1", summarize(df.loc[low, "ret1"]))
    print_stats("low_top2", summarize(df.loc[low, "ret2"]))


if __name__ == "__main__":
    main()
