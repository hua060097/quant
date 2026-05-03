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

FACTORS = [
    "mkt_ma5_followthrough_v3",
    "market_prev_60d_new_high_avg_return",
    "market_60d_new_high_ratio",
    "limit_up_count_percentile_20d",
    "zhaban_count_percentile_20d",
    "limit_down_count_percentile_20d",
]

Q_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ROLLING_WINDOWS = [20, 60, 120]


def max_drawdown(returns: pd.Series) -> float:
    s = pd.Series(returns).fillna(0.0).astype(float)
    if s.empty:
        return float("nan")
    equity = (1.0 + s).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def summarize_returns(returns: pd.Series) -> dict[str, float]:
    s = pd.Series(returns).dropna().astype(float)
    if s.empty:
        return {
            "days": 0,
            "mean": float("nan"),
            "cum": float("nan"),
            "win_rate": float("nan"),
            "mdd": float("nan"),
        }
    return {
        "days": int(len(s)),
        "mean": float(s.mean()),
        "cum": float((1.0 + s).prod() - 1.0),
        "win_rate": float((s > 0).mean()),
        "mdd": max_drawdown(s),
    }


def load_top_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in sorted(raw.items()):
        row = {"date": pd.to_datetime(ds)}
        for i in range(1, 6):
            item = daily.get(f"top{i}", {})
            row[f"top{i}_ret"] = float(item.get("real_return", np.nan))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["top1_positive"] = (df["top1_ret"] > 0).astype(int)
    return df


def load_daily_factors(path: Path, factor_names: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", *factor_names])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[factor_names].first().sort_values("date").reset_index(drop=True)


def build_expanding_threshold(series: pd.Series, q: float, fallback: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        if len(hist) < 20:
            vals.append(fallback)
        else:
            vals.append(float(hist.quantile(q)))
    return pd.Series(vals, index=series.index, dtype="float64")


def build_rolling_threshold(series: pd.Series, q: float, window: int, fallback: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[max(0, i - window):i].dropna()
        if len(hist) < min(window, 20):
            vals.append(fallback)
        else:
            vals.append(float(hist.quantile(q)))
    return pd.Series(vals, index=series.index, dtype="float64")


def choose_switch_returns(work: pd.DataFrame, factor: str, threshold: pd.Series | float, direction: str) -> pd.Series:
    if not isinstance(threshold, pd.Series):
        threshold = pd.Series(threshold, index=work.index, dtype="float64")
    high = work[factor] >= threshold
    if direction == "high_top1_low_top2":
        return pd.Series(np.where(high, work["top1_ret"], work["top2_ret"]), index=work.index, dtype="float64")
    if direction == "high_top2_low_top1":
        return pd.Series(np.where(high, work["top2_ret"], work["top1_ret"]), index=work.index, dtype="float64")
    raise ValueError(direction)


def evaluate_top1_sign_group(work: pd.DataFrame, factor: str, threshold: pd.Series | float, mode: str) -> dict[str, float]:
    if not isinstance(threshold, pd.Series):
        threshold = pd.Series(threshold, index=work.index, dtype="float64")
    high = work[factor] >= threshold
    high_group = work.loc[high]
    low_group = work.loc[~high]
    return {
        "factor": factor,
        "threshold_mode": mode,
        "high_days": int(len(high_group)),
        "low_days": int(len(low_group)),
        "high_top1_mean": float(high_group["top1_ret"].mean()) if len(high_group) else float("nan"),
        "low_top1_mean": float(low_group["top1_ret"].mean()) if len(low_group) else float("nan"),
        "high_top1_positive_rate": float(high_group["top1_positive"].mean()) if len(high_group) else float("nan"),
        "low_top1_positive_rate": float(low_group["top1_positive"].mean()) if len(low_group) else float("nan"),
    }


def main() -> None:
    top_df = load_top_json(JSON_PATH)
    factor_df = load_daily_factors(FACTOR_PATH, FACTORS)
    merged = top_df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

    baseline_top1 = summarize_returns(merged["top1_ret"])
    baseline_top2 = summarize_returns(merged["top2_ret"])

    switch_rows = []
    sign_rows = []
    best_rows = []

    for factor in FACTORS:
        work = merged[["date", factor, "top1_ret", "top2_ret", "top1_positive"]].dropna().copy()
        if len(work) < 30:
            continue

        train_series = factor_df.loc[
            (factor_df["date"] >= TRAIN_START) & (factor_df["date"] <= TRAIN_END),
            factor,
        ].dropna()
        if train_series.empty:
            continue

        for q in Q_GRID:
            train_threshold = float(train_series.quantile(q))
            mode = f"train_q{int(q * 100)}"

            for direction in ("high_top1_low_top2", "high_top2_low_top1"):
                ret = choose_switch_returns(work, factor, train_threshold, direction)
                switch_rows.append({
                    "factor": factor,
                    "threshold_mode": mode,
                    "direction": direction,
                    "threshold_value": train_threshold,
                    **summarize_returns(ret),
                })
            sign_rows.append(evaluate_top1_sign_group(work, factor, train_threshold, mode))

            expanding_threshold = build_expanding_threshold(work[factor], q=q, fallback=train_threshold)
            mode_exp = f"expanding_q{int(q * 100)}_shift1"
            for direction in ("high_top1_low_top2", "high_top2_low_top1"):
                ret = choose_switch_returns(work, factor, expanding_threshold, direction)
                switch_rows.append({
                    "factor": factor,
                    "threshold_mode": mode_exp,
                    "direction": direction,
                    "threshold_value": float("nan"),
                    **summarize_returns(ret),
                })
            sign_rows.append(evaluate_top1_sign_group(work, factor, expanding_threshold, mode_exp))

            for window in ROLLING_WINDOWS:
                rolling_threshold = build_rolling_threshold(work[factor], q=q, window=window, fallback=train_threshold)
                mode_roll = f"rolling{window}_q{int(q * 100)}_shift1"
                for direction in ("high_top1_low_top2", "high_top2_low_top1"):
                    ret = choose_switch_returns(work, factor, rolling_threshold, direction)
                    switch_rows.append({
                        "factor": factor,
                        "threshold_mode": mode_roll,
                        "direction": direction,
                        "threshold_value": float("nan"),
                        **summarize_returns(ret),
                    })
                sign_rows.append(evaluate_top1_sign_group(work, factor, rolling_threshold, mode_roll))

        if work[factor].dropna().min() < 0 < work[factor].dropna().max():
            zero_mode = "zero_sign"
            zero_threshold = 0.0
            for direction in ("high_top1_low_top2", "high_top2_low_top1"):
                ret = choose_switch_returns(work, factor, zero_threshold, direction)
                switch_rows.append({
                    "factor": factor,
                    "threshold_mode": zero_mode,
                    "direction": direction,
                    "threshold_value": zero_threshold,
                    **summarize_returns(ret),
                })
            sign_rows.append(evaluate_top1_sign_group(work, factor, zero_threshold, zero_mode))

    switch_df = pd.DataFrame(switch_rows)
    sign_df = pd.DataFrame(sign_rows)

    for factor, grp in switch_df.groupby("factor"):
        best = grp.sort_values(["cum", "mean"], ascending=False).iloc[0].to_dict()
        best["baseline_top1_cum"] = baseline_top1["cum"]
        best["baseline_top2_cum"] = baseline_top2["cum"]
        best["vs_top1_cum_delta"] = float(best["cum"] - baseline_top1["cum"])
        best["vs_top2_cum_delta"] = float(best["cum"] - baseline_top2["cum"])
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)

    out_dir = ROOT / "results"
    switch_csv = out_dir / "market_factor_dynamic_switch_scan_20260413.csv"
    best_csv = out_dir / "market_factor_dynamic_switch_best_20260413.csv"
    sign_csv = out_dir / "market_factor_dynamic_top1_sign_20260413.csv"
    switch_df.to_csv(switch_csv, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")
    sign_df.to_csv(sign_csv, index=False, encoding="utf-8-sig")

    print(
        f"Baseline top1: mean={baseline_top1['mean']:.4%} cum={baseline_top1['cum']:.4f} "
        f"mdd={baseline_top1['mdd']:.4%}"
    )
    print(
        f"Baseline top2: mean={baseline_top2['mean']:.4%} cum={baseline_top2['cum']:.4f} "
        f"mdd={baseline_top2['mdd']:.4%}"
    )
    print()
    print("Best switch rule per factor:")
    print(
        best_df[
            ["factor", "threshold_mode", "direction", "mean", "cum", "mdd", "vs_top1_cum_delta"]
        ].to_string(index=False)
    )
    print()
    print("Best top1 sign split per factor (by high_top1_positive_rate - low_top1_positive_rate):")
    sign_best = sign_df.assign(
        positive_rate_delta=sign_df["high_top1_positive_rate"] - sign_df["low_top1_positive_rate"],
        mean_delta=sign_df["high_top1_mean"] - sign_df["low_top1_mean"],
    )
    sign_best = (
        sign_best.sort_values(["factor", "positive_rate_delta", "mean_delta"], ascending=[True, False, False])
        .groupby("factor", as_index=False)
        .first()
        .sort_values("positive_rate_delta", ascending=False)
    )
    print(
        sign_best[
            [
                "factor",
                "threshold_mode",
                "high_top1_positive_rate",
                "low_top1_positive_rate",
                "positive_rate_delta",
                "high_top1_mean",
                "low_top1_mean",
                "mean_delta",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved: {switch_csv}")
    print(f"Saved: {best_csv}")
    print(f"Saved: {sign_csv}")


if __name__ == "__main__":
    main()
