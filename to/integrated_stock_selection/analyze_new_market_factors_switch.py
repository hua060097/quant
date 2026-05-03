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

BASE_GATE_FACTOR = "mkt_ma5_followthrough_v3"
NEW_FACTORS = [
    "market_prev_60d_new_high_avg_return",
    "market_60d_new_high_ratio",
    "limit_up_count_percentile_20d",
    "zhaban_count_percentile_20d",
    "limit_down_count_percentile_20d",
]

THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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
            row[f"top{i}_pred"] = float(item.get("pred", np.nan))
            row[f"top{i}_ret"] = float(item.get("real_return", np.nan))
            row[f"top{i}_stock"] = item.get("stock_code")
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["spread_12"] = df["top1_ret"] - df["top2_ret"]
    df["winner_is_top1"] = (df["top1_ret"] >= df["top2_ret"]).astype(int)
    df["top12_pred_gap"] = df["top1_pred"] - df["top2_pred"]
    return df


def load_daily_factors(path: Path, factor_names: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", *factor_names])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[factor_names].first().sort_values("date").reset_index(drop=True)


def choose_returns(work: pd.DataFrame, factor: str, threshold: float, direction: str) -> pd.Series:
    high = work[factor] >= threshold
    if direction == "high_top1_low_top2":
        return pd.Series(np.where(high, work["top1_ret"], work["top2_ret"]), index=work.index, dtype="float64")
    if direction == "high_top2_low_top1":
        return pd.Series(np.where(high, work["top2_ret"], work["top1_ret"]), index=work.index, dtype="float64")
    raise ValueError(f"unknown direction: {direction}")


def evaluate_threshold_switch(
    merged: pd.DataFrame,
    factor_daily: pd.DataFrame,
    factor: str,
    threshold_quantiles: list[float],
) -> tuple[pd.DataFrame, pd.Series]:
    train_series = factor_daily.loc[
        (factor_daily["date"] >= TRAIN_START) & (factor_daily["date"] <= TRAIN_END),
        factor,
    ].dropna()
    work = merged[["date", factor, "top1_ret", "top2_ret"]].dropna().copy()
    rows = []
    best_returns = pd.Series(index=work.index, dtype="float64")
    best_key = None
    best_cum = -np.inf

    for q in threshold_quantiles:
        if train_series.empty:
            continue
        threshold = float(train_series.quantile(q))
        for direction in ("high_top1_low_top2", "high_top2_low_top1"):
            switch_returns = choose_returns(work, factor, threshold, direction)
            stats = summarize_returns(switch_returns)
            row = {
                "factor": factor,
                "train_quantile": q,
                "threshold": threshold,
                "direction": direction,
                **stats,
            }
            rows.append(row)
            if np.isfinite(stats["cum"]) and stats["cum"] > best_cum:
                best_cum = stats["cum"]
                best_returns = switch_returns
                best_key = row

    result = pd.DataFrame(rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    if best_key is None:
        return result, pd.Series(dtype="float64")
    best_returns.name = f"{best_key['factor']}@q{int(best_key['train_quantile'] * 100)}"
    return result, best_returns


def quantile_bucket_stats(merged: pd.DataFrame, factor: str) -> dict[str, float]:
    work = merged[["date", factor, "top1_ret", "top2_ret", "spread_12", "winner_is_top1"]].dropna().copy()
    if len(work) < 30:
        return {}
    work["bucket"] = pd.qcut(work[factor], q=5, labels=False, duplicates="drop")
    low = work[work["bucket"] == work["bucket"].min()]
    high = work[work["bucket"] == work["bucket"].max()]
    return {
        "q1_top1_mean": float(low["top1_ret"].mean()),
        "q5_top1_mean": float(high["top1_ret"].mean()),
        "q1_spread_mean": float(low["spread_12"].mean()),
        "q5_spread_mean": float(high["spread_12"].mean()),
        "q1_top1_win": float(low["winner_is_top1"].mean()),
        "q5_top1_win": float(high["winner_is_top1"].mean()),
    }


def correlation_stats(merged: pd.DataFrame, factor: str) -> dict[str, float]:
    work = merged[[factor, "top1_ret", "top2_ret", "spread_12", "winner_is_top1"]].dropna().copy()
    if len(work) < 20:
        return {}
    return {
        "pearson_top1": float(work[factor].corr(work["top1_ret"], method="pearson")),
        "spearman_top1": float(work[factor].corr(work["top1_ret"], method="spearman")),
        "pearson_spread12": float(work[factor].corr(work["spread_12"], method="pearson")),
        "spearman_spread12": float(work[factor].corr(work["spread_12"], method="spearman")),
        "pearson_top1_win": float(work[factor].corr(work["winner_is_top1"], method="pearson")),
        "spearman_top1_win": float(work[factor].corr(work["winner_is_top1"], method="spearman")),
    }


def regime_slice_stats(merged: pd.DataFrame, factor: str, base_gate_threshold: float) -> list[dict[str, float]]:
    rows = []
    for regime_name, mask in (
        ("mkt_q40_high", merged[BASE_GATE_FACTOR] >= base_gate_threshold),
        ("mkt_q40_low", merged[BASE_GATE_FACTOR] < base_gate_threshold),
    ):
        work = merged.loc[mask, [factor, "top1_ret", "spread_12", "winner_is_top1"]].dropna().copy()
        if len(work) < 15:
            continue
        rows.append(
            {
                "factor": factor,
                "regime": regime_name,
                "days": int(len(work)),
                "pearson_top1": float(work[factor].corr(work["top1_ret"], method="pearson")),
                "spearman_top1": float(work[factor].corr(work["top1_ret"], method="spearman")),
                "pearson_spread12": float(work[factor].corr(work["spread_12"], method="pearson")),
                "spearman_spread12": float(work[factor].corr(work["spread_12"], method="spearman")),
                "top1_mean": float(work["top1_ret"].mean()),
                "top1_win": float(work["winner_is_top1"].mean()),
            }
        )
    return rows


def main() -> None:
    factor_names = [BASE_GATE_FACTOR, *NEW_FACTORS]
    factor_daily = load_daily_factors(FACTOR_PATH, factor_names)
    top_df = load_top_json(JSON_PATH)
    merged = top_df.merge(factor_daily, on="date", how="inner").sort_values("date").reset_index(drop=True)

    if merged.empty:
        raise RuntimeError("No overlapping dates between top json and factor daily data.")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_train = factor_daily.loc[
        (factor_daily["date"] >= TRAIN_START) & (factor_daily["date"] <= TRAIN_END),
        BASE_GATE_FACTOR,
    ].dropna()
    base_q40 = float(base_train.quantile(0.4))

    base_switch = choose_returns(merged.dropna(subset=[BASE_GATE_FACTOR]), BASE_GATE_FACTOR, base_q40, "high_top1_low_top2")
    base_stats = summarize_returns(base_switch)
    top1_stats = summarize_returns(merged["top1_ret"])
    top2_stats = summarize_returns(merged["top2_ret"])

    summary_rows = []
    switch_rows = []
    regime_rows = []

    for factor in NEW_FACTORS:
        summary = {"factor": factor}
        summary.update(correlation_stats(merged, factor))
        summary.update(quantile_bucket_stats(merged, factor))

        switch_df, _ = evaluate_threshold_switch(merged, factor_daily, factor, THRESHOLD_GRID)
        if not switch_df.empty:
            best_row = switch_df.iloc[0].to_dict()
            summary["best_switch_direction"] = best_row["direction"]
            summary["best_switch_train_quantile"] = float(best_row["train_quantile"])
            summary["best_switch_threshold"] = float(best_row["threshold"])
            summary["best_switch_mean"] = float(best_row["mean"])
            summary["best_switch_cum"] = float(best_row["cum"])
            summary["best_switch_win_rate"] = float(best_row["win_rate"])
            summary["best_switch_mdd"] = float(best_row["mdd"])
            summary["vs_base_q40_cum_delta"] = float(best_row["cum"] - base_stats["cum"])
            summary["vs_top1_cum_delta"] = float(best_row["cum"] - top1_stats["cum"])
            switch_rows.extend(switch_df.to_dict("records"))

        regime_rows.extend(regime_slice_stats(merged, factor, base_q40))
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    switch_df = pd.DataFrame(switch_rows)
    regime_df = pd.DataFrame(regime_rows)

    summary_csv = out_dir / "new_market_factor_switch_summary_20260413.csv"
    switch_csv = out_dir / "new_market_factor_switch_grid_20260413.csv"
    regime_csv = out_dir / "new_market_factor_regime_slice_20260413.csv"
    report_json = out_dir / "new_market_factor_switch_report_20260413.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    switch_df.to_csv(switch_csv, index=False, encoding="utf-8-sig")
    regime_df.to_csv(regime_csv, index=False, encoding="utf-8-sig")

    report = {
        "json_path": str(JSON_PATH),
        "factor_path": str(FACTOR_PATH),
        "test_range": {
            "start": merged["date"].min().strftime("%Y-%m-%d"),
            "end": merged["date"].max().strftime("%Y-%m-%d"),
            "days": int(len(merged)),
        },
        "train_window": {
            "start": TRAIN_START.strftime("%Y-%m-%d"),
            "end": TRAIN_END.strftime("%Y-%m-%d"),
        },
        "base_gate": {
            "factor": BASE_GATE_FACTOR,
            "train_q40_threshold": base_q40,
            "rule": "high_top1_low_top2",
            "stats": base_stats,
        },
        "baseline_top1": top1_stats,
        "baseline_top2": top2_stats,
        "summary_csv": str(summary_csv),
        "switch_csv": str(switch_csv),
        "regime_csv": str(regime_csv),
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Test days: {len(merged)} | range: {merged['date'].min().date()} ~ {merged['date'].max().date()}")
    print(
        f"Baseline top1: mean={top1_stats['mean']:.4%} cum={top1_stats['cum']:.4f} "
        f"mdd={top1_stats['mdd']:.4%} win={top1_stats['win_rate']:.4%}"
    )
    print(
        f"Baseline top2: mean={top2_stats['mean']:.4%} cum={top2_stats['cum']:.4f} "
        f"mdd={top2_stats['mdd']:.4%} win={top2_stats['win_rate']:.4%}"
    )
    print(
        f"Base gate {BASE_GATE_FACTOR} q40={base_q40:.6f}: mean={base_stats['mean']:.4%} "
        f"cum={base_stats['cum']:.4f} mdd={base_stats['mdd']:.4%} win={base_stats['win_rate']:.4%}"
    )
    print()

    if not summary_df.empty:
        cols = [
            "factor",
            "pearson_top1",
            "spearman_top1",
            "pearson_spread12",
            "spearman_spread12",
            "pearson_top1_win",
            "best_switch_direction",
            "best_switch_train_quantile",
            "best_switch_cum",
            "vs_base_q40_cum_delta",
            "vs_top1_cum_delta",
        ]
        print("Top factors by |spread corr|:")
        print(
            summary_df.reindex(summary_df["spearman_spread12"].abs().sort_values(ascending=False).index)[cols]
            .to_string(index=False)
        )
        print()
        print(f"Saved: {summary_csv}")
        print(f"Saved: {switch_csv}")
        print(f"Saved: {regime_csv}")
        print(f"Saved: {report_json}")


if __name__ == "__main__":
    main()
