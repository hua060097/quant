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
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


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


def build_expanding_threshold(series: pd.Series, q: float, fallback: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


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


def assign_blocks_by_size(df: pd.DataFrame, block_size: int) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    work["block_id"] = np.arange(len(work)) // block_size + 1
    return work


def add_schemes(df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    train = factor_df[(factor_df["date"] >= TRAIN_START) & (factor_df["date"] <= TRAIN_END)]

    mkt_q40 = float(train[MKT].dropna().quantile(0.4))
    mkt_q30 = float(train[MKT].dropna().quantile(0.3))
    nh_q70 = float(train[NH].dropna().quantile(0.7))
    exp_q30 = build_expanding_threshold(merged[MKT], 0.3, mkt_q30)

    merged["top1"] = merged["top1_ret"]
    merged["mkt_q40"] = choose_single(merged, MKT, mkt_q40)
    merged["mkt_exp_q30"] = choose_single(merged, MKT, exp_q30)
    merged["dual_gate"] = choose_dual(merged, mkt_q40, nh_q70)
    return merged


def evaluate_blocks(df: pd.DataFrame, scheme_cols: list[str], block_size: int, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = assign_blocks_by_size(df, block_size)
    rows = []
    for block_id, grp in work.groupby("block_id", sort=True):
        block_label = f"{grp['date'].iloc[0].date()}~{grp['date'].iloc[-1].date()}"
        block_stats = {}
        for scheme in scheme_cols:
            stats = summarize(grp[scheme])
            row = {
                "partition": label,
                "block_id": int(block_id),
                "block_label": block_label,
                "scheme": scheme,
                **stats,
            }
            rows.append(row)
            block_stats[scheme] = stats["cum"]

    block_df = pd.DataFrame(rows)
    summary_rows = []
    for scheme, grp in block_df.groupby("scheme"):
        summary_rows.append(
            {
                "partition": label,
                "scheme": scheme,
                "blocks": int(grp["block_id"].nunique()),
                "positive_block_ratio": float((grp["cum"] > 0).mean()),
                "best_block_cum": float(grp["cum"].max()),
                "worst_block_cum": float(grp["cum"].min()),
                "mean_block_cum": float(grp["cum"].mean()),
                "median_block_cum": float(grp["cum"].median()),
                "std_block_cum": float(grp["cum"].std()),
                "mean_block_mean": float(grp["mean"].mean()),
            }
        )

    # 对每个块做相对胜出统计
    for summary in summary_rows:
        scheme = summary["scheme"]
        wins = 0
        ties = 0
        total = 0
        for block_id, grp in block_df.groupby("block_id"):
            total += 1
            best_cum = grp["cum"].max()
            cur = grp.loc[grp["scheme"] == scheme, "cum"].iloc[0]
            if np.isclose(cur, best_cum):
                if (grp["cum"] == best_cum).sum() > 1:
                    ties += 1
                else:
                    wins += 1
        summary["best_block_win_ratio"] = wins / total if total else float("nan")
        summary["best_block_tie_ratio"] = ties / total if total else float("nan")

    return block_df, pd.DataFrame(summary_rows)


def evaluate_month_blocks(df: pd.DataFrame, scheme_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    rows = []
    for block_id, (month, grp) in enumerate(work.groupby("month", sort=True), start=1):
        block_label = f"{grp['date'].iloc[0].date()}~{grp['date'].iloc[-1].date()}"
        for scheme in scheme_cols:
            rows.append(
                {
                    "partition": "month",
                    "block_id": block_id,
                    "block_label": block_label,
                    "month": month,
                    "scheme": scheme,
                    **summarize(grp[scheme]),
                }
            )
    block_df = pd.DataFrame(rows)
    summary_rows = []
    for scheme, grp in block_df.groupby("scheme"):
        summary_rows.append(
            {
                "partition": "month",
                "scheme": scheme,
                "blocks": int(grp["block_id"].nunique()),
                "positive_block_ratio": float((grp["cum"] > 0).mean()),
                "best_block_cum": float(grp["cum"].max()),
                "worst_block_cum": float(grp["cum"].min()),
                "mean_block_cum": float(grp["cum"].mean()),
                "median_block_cum": float(grp["cum"].median()),
                "std_block_cum": float(grp["cum"].std()),
                "mean_block_mean": float(grp["mean"].mean()),
            }
        )
    for summary in summary_rows:
        scheme = summary["scheme"]
        wins = 0
        ties = 0
        total = 0
        for block_id, grp in block_df.groupby("block_id"):
            total += 1
            best_cum = grp["cum"].max()
            cur = grp.loc[grp["scheme"] == scheme, "cum"].iloc[0]
            if np.isclose(cur, best_cum):
                if (grp["cum"] == best_cum).sum() > 1:
                    ties += 1
                else:
                    wins += 1
        summary["best_block_win_ratio"] = wins / total if total else float("nan")
        summary["best_block_tie_ratio"] = ties / total if total else float("nan")
    return block_df, pd.DataFrame(summary_rows)


def main() -> None:
    top_df = load_top_json(JSON_PATH)
    factor_df = load_factors(FACTOR_PATH)
    df = add_schemes(top_df, factor_df)
    schemes = ["top1", "mkt_q40", "mkt_exp_q30", "dual_gate"]

    month_block_df, month_summary_df = evaluate_month_blocks(df, schemes)
    block20_df, block20_summary_df = evaluate_blocks(df, schemes, block_size=20, label="20d")
    block30_df, block30_summary_df = evaluate_blocks(df, schemes, block_size=30, label="30d")

    all_summary_df = pd.concat([month_summary_df, block20_summary_df, block30_summary_df], ignore_index=True)

    out_dir = ROOT / "results"
    month_block_csv = out_dir / "gate_scheme_month_blocks_20260413.csv"
    block20_csv = out_dir / "gate_scheme_20d_blocks_20260413.csv"
    block30_csv = out_dir / "gate_scheme_30d_blocks_20260413.csv"
    summary_csv = out_dir / "gate_scheme_timeblock_summary_20260413.csv"

    month_block_df.to_csv(month_block_csv, index=False, encoding="utf-8-sig")
    block20_df.to_csv(block20_csv, index=False, encoding="utf-8-sig")
    block30_df.to_csv(block30_csv, index=False, encoding="utf-8-sig")
    all_summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("Time-block summary:")
    print(all_summary_df.sort_values(["partition", "best_block_win_ratio", "positive_block_ratio"], ascending=[True, False, False]).to_string(index=False))
    print()
    print(f"Saved: {month_block_csv}")
    print(f"Saved: {block20_csv}")
    print(f"Saved: {block30_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
