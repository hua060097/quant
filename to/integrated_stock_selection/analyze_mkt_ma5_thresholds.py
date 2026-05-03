from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
JSON_3D = BASE_DIR / "results" / "daily_top5_details_Window_3_20260320_012750.json"
JSON_2D = BASE_DIR / "results" / "daily_top5_details_Window_3_20260320_022143.json"

FACTOR_NAME = "mkt_ma5_followthrough_v3"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
ROLLING_WINDOWS = [20, 60, 120]


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def summarize(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {"mean": float("nan"), "cum": float("nan"), "mdd": float("nan"), "days": 0}
    return {
        "mean": float(returns.mean()),
        "cum": float((1.0 + returns).prod() - 1.0),
        "mdd": max_drawdown(returns),
        "days": int(len(returns)),
    }


def load_top_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        row = {"date": pd.to_datetime(ds)}
        for i in range(1, 6):
            item = daily.get(f"top{i}", {})
            row[f"top{i}"] = float(item.get("real_return")) if item.get("real_return") is not None else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factor_daily() -> pd.DataFrame:
    df = pd.read_parquet(FACTOR_PATH, columns=["date", FACTOR_NAME])
    df["date"] = pd.to_datetime(df["date"])
    # State factors are broadcast daily; keep one row per date.
    return df.groupby("date", as_index=False).first().sort_values("date").reset_index(drop=True)


def choose_by_threshold(df: pd.DataFrame, threshold: pd.Series | float) -> pd.Series:
    if not isinstance(threshold, pd.Series):
        threshold = pd.Series(threshold, index=df.index)
    choose_top1 = df[FACTOR_NAME] >= threshold
    out = pd.Series(index=df.index, dtype=float)
    out[choose_top1] = df.loc[choose_top1, "top1"]
    out[~choose_top1] = df.loc[~choose_top1, "top2"]
    return out


def evaluate_rule(label: str, returns: pd.Series, dates: pd.Series) -> list[tuple[str, dict[str, float]]]:
    mid = len(returns) // 2
    pieces = [
        ("full", returns),
        ("first_half", returns.iloc[:mid]),
        ("second_half", returns.iloc[mid:]),
    ]
    results = []
    for part, part_returns in pieces:
        results.append((part, summarize(part_returns)))
    return results


def print_rule_results(name: str, results: list[tuple[str, dict[str, float]]]) -> None:
    print(f"-- {name} --")
    for part, stats in results:
        print(
            f"{part}: mean={stats['mean']:.4%} cum={stats['cum']:.4f} "
            f"mdd={stats['mdd']:.4%} days={stats['days']}"
        )
    print()


def main() -> None:
    factor_df = load_factor_daily()
    train_mask = (factor_df["date"] >= TRAIN_START) & (factor_df["date"] <= TRAIN_END)
    train_median = float(factor_df.loc[train_mask, FACTOR_NAME].median())

    print(f"Factor: {FACTOR_NAME}")
    print(f"Train-period median ({TRAIN_START.date()} ~ {TRAIN_END.date()}): {train_median:.6f}")
    print()

    for label, json_path in [("3D_WEIGHTED", JSON_3D), ("2D_WEIGHTED", JSON_2D)]:
        top_df = load_top_json(json_path)
        df = top_df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

        print(f"=== {label} ===")
        print(f"days={len(df)} range={df['date'].min().date()} ~ {df['date'].max().date()}")
        print(
            f"BASE top1 mean={df['top1'].mean():.4%} cum={(1.0 + df['top1']).prod() - 1.0:.4f} "
            f"mdd={max_drawdown(df['top1']):.4%} | "
            f"top2 mean={df['top2'].mean():.4%} cum={(1.0 + df['top2']).prod() - 1.0:.4f} "
            f"mdd={max_drawdown(df['top2']):.4%}"
        )
        print()

        all_sample_threshold = float(df[FACTOR_NAME].median())
        all_sample_returns = choose_by_threshold(df, all_sample_threshold)
        print_rule_results(
            f"all_sample_median={all_sample_threshold:.6f}",
            evaluate_rule("all_sample_median", all_sample_returns, df["date"]),
        )

        train_fixed_returns = choose_by_threshold(df, train_median)
        print_rule_results(
            f"train_fixed_median={train_median:.6f}",
            evaluate_rule("train_fixed_median", train_fixed_returns, df["date"]),
        )

        expanding_threshold = df[FACTOR_NAME].shift(1).expanding(min_periods=20).median().fillna(train_median)
        expanding_returns = choose_by_threshold(df, expanding_threshold)
        print_rule_results(
            "expanding_median_shift1",
            evaluate_rule("expanding_median_shift1", expanding_returns, df["date"]),
        )

        for window in ROLLING_WINDOWS:
            threshold = (
                df[FACTOR_NAME].shift(1).rolling(window, min_periods=window).median().fillna(train_median)
            )
            returns = choose_by_threshold(df, threshold)
            print_rule_results(
                f"rolling_{window}d_median_shift1",
                evaluate_rule(f"rolling_{window}d_median_shift1", returns, df["date"]),
            )


if __name__ == "__main__":
    main()
