import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.array(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak == 0, 1.0, peak) - 1.0
    return float(np.min(dd))


def perf_stats(returns):
    arr = np.array(list(returns), dtype=float)
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": None,
            "win_rate": None,
            "cum_return": None,
            "max_drawdown": None,
        }
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "cum_return": float(np.prod(1.0 + arr) - 1.0),
        "max_drawdown": float(max_drawdown(arr)),
    }


def build_dataframe(json_path: Path) -> pd.DataFrame:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for d in sorted(obj):
        day = obj[d]
        filtered_rank = (day.get("filtered_rank") or {})
        topn = filtered_rank.get("rank_topn", [])
        if len(topn) < 2:
            continue
        t1, t2 = topn[0], topn[1]
        raw_top1 = ((day.get("raw_rank") or {}).get("rank_top1") or {})
        rows.append(
            {
                "date": pd.to_datetime(d),
                "t1_ret": float(t1["real_return"]),
                "t2_ret": float(t2["real_return"]),
                "switch_alpha": float(t2["real_return"]) - float(t1["real_return"]),
                "margin": float((t1.get("rank_score") or 0.0) - (t2.get("rank_score") or 0.0)),
                "t1_pcls": float(t1["p_cls"]) if t1.get("p_cls") is not None else np.nan,
                "t2_pcls": float(t2["p_cls"]) if t2.get("p_cls") is not None else np.nan,
                "limit_up": int((day.get("event_counts") or {}).get("limit_up") or 0),
                "limit_down": int((day.get("event_counts") or {}).get("limit_down") or 0),
                "zhaban": int((day.get("event_counts") or {}).get("zhaban") or 0),
                "raw_changed": int(raw_top1.get("code") != t1.get("code")) if raw_top1.get("code") is not None else 0,
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("JSON 中没有足够的 filtered_rank Top2 数据")
    return df


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["regime20"] = out["switch_alpha"].rolling(20).mean().shift(1)
    out["regime10"] = out["switch_alpha"].rolling(10).mean().shift(1)
    return out


def summarize_gap_buckets(df: pd.DataFrame):
    q1, q2 = df["margin"].quantile([1 / 3, 2 / 3]).tolist()

    def bucket(v):
        if v <= q1:
            return "small"
        if v <= q2:
            return "mid"
        return "large"

    tmp = df.copy()
    tmp["gap_bucket"] = tmp["margin"].apply(bucket)
    rows = []
    for bucket_name, g in tmp.groupby("gap_bucket"):
        rows.append(
            {
                "bucket": bucket_name,
                "n": len(g),
                "p_top2_gt_top1": float((g["switch_alpha"] > 0).mean()),
                "mean_alpha_top2_minus_top1": float(g["switch_alpha"].mean()),
                "mean_margin": float(g["margin"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_margin")


def compare_strategies(df: pd.DataFrame):
    strategies = {
        "top1": df["t1_ret"],
        "top2": df["t2_ret"],
        "margin_le_0.005": np.where(df["margin"] <= 0.005, df["t2_ret"], df["t1_ret"]),
        "margin_le_0.003": np.where(df["margin"] <= 0.003, df["t2_ret"], df["t1_ret"]),
        "regime20_gt_0": np.where(df["regime20"] > 0, df["t2_ret"], df["t1_ret"]),
        "regime10_gt_0": np.where(df["regime10"] > 0, df["t2_ret"], df["t1_ret"]),
        "margin0.005_and_regime20": np.where((df["margin"] <= 0.005) & (df["regime20"] > 0), df["t2_ret"], df["t1_ret"]),
        "margin0.003_and_regime20": np.where((df["margin"] <= 0.003) & (df["regime20"] > 0), df["t2_ret"], df["t1_ret"]),
        "margin0.005_and_pcls0.7": np.where((df["margin"] <= 0.005) & (df["t1_pcls"] >= 0.7), df["t2_ret"], df["t1_ret"]),
    }

    rows = []
    for name, values in strategies.items():
        values = pd.Series(values).dropna()
        stats = perf_stats(values)
        switch_count = None
        if name not in {"top1", "top2"}:
            switch_count = int(np.sum(np.array(values) == np.array(df.loc[values.index, "t2_ret"])))
        rows.append(
            {
                "strategy": name,
                **stats,
                "switch_count": switch_count,
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="对 daily_rank_stats JSON 做 Top1/Top2 切换策略分析")
    parser.add_argument("json_path")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    df = add_regimes(build_dataframe(json_path))

    print(f"json={json_path.name}")
    print(f"rows={len(df)}  date_range={df['date'].min().date()}~{df['date'].max().date()}")
    print()

    print("[Global]")
    print(f"p(top2 > top1) = {(df['switch_alpha'] > 0).mean():.4f}")
    print(f"mean(top2 - top1) = {df['switch_alpha'].mean():.4%}")
    print()

    print("[Gap Buckets]")
    print(summarize_gap_buckets(df).to_string(index=False))
    print()

    print("[Strategy Compare]")
    print(compare_strategies(df).to_string(index=False))


if __name__ == "__main__":
    main()
