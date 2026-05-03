from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_010230.json"

TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")

MKT_FACTOR = "mkt_ma5_followthrough_v3"
NH_FACTOR = "market_60d_new_high_ratio"
NEW_FACTORS = [
    "market_prev_60d_new_high_avg_return",
    "market_60d_new_high_ratio",
    "limit_up_count_percentile_20d",
    "zhaban_count_percentile_20d",
    "limit_down_count_percentile_20d",
]

MKT_Q = 0.4
NH_Q_CANDIDATES = [0.5, 0.6, 0.7, 0.8]


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
            row[f"top{i}_pred"] = float(item.get("pred", np.nan))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["top1_positive"] = (df["top1_ret"] > 0).astype(int)
    df["top12_spread"] = df["top1_ret"] - df["top2_ret"]
    return df


def load_daily_factors(path: Path, factor_names: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", *factor_names])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[factor_names].first().sort_values("date").reset_index(drop=True)


def train_quantile_threshold(factor_daily: pd.DataFrame, factor: str, q: float) -> float:
    train = factor_daily.loc[
        (factor_daily["date"] >= TRAIN_START) & (factor_daily["date"] <= TRAIN_END),
        factor,
    ].dropna()
    return float(train.quantile(q))


def single_gate_returns(df: pd.DataFrame, factor: str, threshold: float, high_pick: str, low_pick: str) -> pd.Series:
    high = df[factor] >= threshold
    out = pd.Series(index=df.index, dtype="float64")
    out.loc[high] = df.loc[high, f"{high_pick}_ret"]
    out.loc[~high] = df.loc[~high, f"{low_pick}_ret"]
    return out


def dual_gate_returns(
    df: pd.DataFrame,
    mkt_threshold: float,
    nh_threshold: float,
    mapping: dict[tuple[int, int], str],
) -> pd.Series:
    mkt_high = (df[MKT_FACTOR] >= mkt_threshold).astype(int)
    nh_high = (df[NH_FACTOR] >= nh_threshold).astype(int)
    chosen = []
    for mh, nh in zip(mkt_high, nh_high):
        chosen.append(mapping[(int(mh), int(nh))])
    chosen = pd.Series(chosen, index=df.index)
    out = pd.Series(index=df.index, dtype="float64")
    for rank in ("top1", "top2"):
        mask = chosen == rank
        out.loc[mask] = df.loc[mask, f"{rank}_ret"]
    return out


def positive_relation_stats(df: pd.DataFrame, factor: str) -> dict[str, float]:
    work = df[[factor, "top1_ret", "top1_positive"]].dropna().copy()
    if len(work) < 20:
        return {}
    work["bucket"] = pd.qcut(work[factor], q=5, labels=False, duplicates="drop")
    low = work[work["bucket"] == work["bucket"].min()]
    high = work[work["bucket"] == work["bucket"].max()]
    return {
        "factor": factor,
        "pearson_top1_ret": float(work[factor].corr(work["top1_ret"], method="pearson")),
        "spearman_top1_ret": float(work[factor].corr(work["top1_ret"], method="spearman")),
        "pearson_top1_positive": float(work[factor].corr(work["top1_positive"], method="pearson")),
        "spearman_top1_positive": float(work[factor].corr(work["top1_positive"], method="spearman")),
        "q1_top1_positive_rate": float(low["top1_positive"].mean()),
        "q5_top1_positive_rate": float(high["top1_positive"].mean()),
        "q1_top1_mean": float(low["top1_ret"].mean()),
        "q5_top1_mean": float(high["top1_ret"].mean()),
    }


def main() -> None:
    factor_names = [MKT_FACTOR, NH_FACTOR, *[f for f in NEW_FACTORS if f != NH_FACTOR]]
    top_df = load_top_json(JSON_PATH)
    factor_daily = load_daily_factors(FACTOR_PATH, factor_names)
    df = top_df.merge(factor_daily, on="date", how="inner").sort_values("date").reset_index(drop=True)

    mkt_threshold = train_quantile_threshold(factor_daily, MKT_FACTOR, MKT_Q)
    nh_threshold_best_single = train_quantile_threshold(factor_daily, NH_FACTOR, 0.7)

    top1_stats = summarize_returns(df["top1_ret"])
    top2_stats = summarize_returns(df["top2_ret"])
    mkt_single = single_gate_returns(df, MKT_FACTOR, mkt_threshold, "top1", "top2")
    nh_single = single_gate_returns(df, NH_FACTOR, nh_threshold_best_single, "top1", "top2")
    mkt_single_stats = summarize_returns(mkt_single)
    nh_single_stats = summarize_returns(nh_single)

    dual_rows = []
    best_dual_stats = None
    best_dual_returns = None
    best_dual_meta = None

    for nh_q in NH_Q_CANDIDATES:
        nh_threshold = train_quantile_threshold(factor_daily, NH_FACTOR, nh_q)
        for choices in product(("top1", "top2"), repeat=4):
            mapping = {
                (0, 0): choices[0],
                (0, 1): choices[1],
                (1, 0): choices[2],
                (1, 1): choices[3],
            }
            if len(set(mapping.values())) == 1:
                continue
            rets = dual_gate_returns(df, mkt_threshold, nh_threshold, mapping)
            stats = summarize_returns(rets)
            row = {
                "mkt_q": MKT_Q,
                "mkt_threshold": mkt_threshold,
                "nh_q": nh_q,
                "nh_threshold": nh_threshold,
                "state_00_mktLow_nhLow": mapping[(0, 0)],
                "state_01_mktLow_nhHigh": mapping[(0, 1)],
                "state_10_mktHigh_nhLow": mapping[(1, 0)],
                "state_11_mktHigh_nhHigh": mapping[(1, 1)],
                **stats,
            }
            dual_rows.append(row)
            if best_dual_stats is None or stats["cum"] > best_dual_stats["cum"]:
                best_dual_stats = stats
                best_dual_returns = rets
                best_dual_meta = row

    dual_df = pd.DataFrame(dual_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)

    positive_rows = [positive_relation_stats(df, factor) for factor in [MKT_FACTOR, *NEW_FACTORS]]
    positive_df = pd.DataFrame([row for row in positive_rows if row])

    out_dir = ROOT / "results"
    dual_csv = out_dir / "dual_gate_mkt_q40_plus_newhigh_20260413.csv"
    positive_csv = out_dir / "market_factor_top1_positive_relation_20260413.csv"
    dual_df.to_csv(dual_csv, index=False, encoding="utf-8-sig")
    positive_df.to_csv(positive_csv, index=False, encoding="utf-8-sig")

    print("Single-gate comparison")
    print(
        f"top1 baseline: mean={top1_stats['mean']:.4%} cum={top1_stats['cum']:.4f} "
        f"mdd={top1_stats['mdd']:.4%}"
    )
    print(
        f"{MKT_FACTOR} q40 ({mkt_threshold:.6f}): mean={mkt_single_stats['mean']:.4%} "
        f"cum={mkt_single_stats['cum']:.4f} mdd={mkt_single_stats['mdd']:.4%}"
    )
    print(
        f"{NH_FACTOR} q70 ({nh_threshold_best_single:.6f}): mean={nh_single_stats['mean']:.4%} "
        f"cum={nh_single_stats['cum']:.4f} mdd={nh_single_stats['mdd']:.4%}"
    )
    print()

    if best_dual_meta is not None and best_dual_stats is not None:
        print("Best dual-gate")
        print(
            f"mkt q40 ({best_dual_meta['mkt_threshold']:.6f}) + nh q{int(best_dual_meta['nh_q'] * 100)} "
            f"({best_dual_meta['nh_threshold']:.6f})"
        )
        print(
            "states: "
            f"(mktLow,nhLow)->{best_dual_meta['state_00_mktLow_nhLow']} "
            f"(mktLow,nhHigh)->{best_dual_meta['state_01_mktLow_nhHigh']} "
            f"(mktHigh,nhLow)->{best_dual_meta['state_10_mktHigh_nhLow']} "
            f"(mktHigh,nhHigh)->{best_dual_meta['state_11_mktHigh_nhHigh']}"
        )
        print(
            f"mean={best_dual_stats['mean']:.4%} cum={best_dual_stats['cum']:.4f} "
            f"mdd={best_dual_stats['mdd']:.4%}"
        )
        print(
            f"vs {MKT_FACTOR} q40: cum_delta={best_dual_stats['cum'] - mkt_single_stats['cum']:.4f} "
            f"mdd_delta={best_dual_stats['mdd'] - mkt_single_stats['mdd']:.4%}"
        )
        print(
            f"vs {NH_FACTOR} q70: cum_delta={best_dual_stats['cum'] - nh_single_stats['cum']:.4f} "
            f"mdd_delta={best_dual_stats['mdd'] - nh_single_stats['mdd']:.4%}"
        )
        if best_dual_returns is not None:
            _ = best_dual_returns
    print()

    print("Top1 sign relation")
    cols = [
        "factor",
        "pearson_top1_ret",
        "spearman_top1_ret",
        "pearson_top1_positive",
        "spearman_top1_positive",
        "q1_top1_positive_rate",
        "q5_top1_positive_rate",
        "q1_top1_mean",
        "q5_top1_mean",
    ]
    print(
        positive_df.sort_values("spearman_top1_positive", ascending=False)[cols].to_string(index=False)
    )
    print()
    print(f"Saved: {dual_csv}")
    print(f"Saved: {positive_csv}")


if __name__ == "__main__":
    main()
