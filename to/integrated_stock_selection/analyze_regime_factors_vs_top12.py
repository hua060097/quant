from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_STORE = ROOT / "factor_lab_data" / "factors.parquet"
JSON_3D = ROOT / "results" / "daily_top5_details_Window_3_20260320_012750.json"
JSON_2D = ROOT / "results" / "daily_top5_details_Window_3_20260320_022143.json"

STATE_FACTORS = [
    "mkt_ma5_followthrough_v3",
    "sector_leadership_persistence_v3",
    "sector_hotspot_concentration_v3",
    "ma5_strength_breadth_v3",
    "strong_stock_volume_confirmation_v3",
    "emotion_quality_balance_v3",
    "top10_10d_recent3_strength_v3",
    "top10_5d_recent3_strength_v3",
]


def load_daily_top(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for d, rec in sorted(data.items()):
        row = {"date": pd.to_datetime(d)}
        for i in range(1, 6):
            item = rec.get(f"top{i}", {})
            row[f"top{i}_pred"] = float(item.get("pred", 0.0) or 0.0)
            row[f"top{i}_ret"] = float(item.get("real_return", 0.0) or 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["spread_12"] = df["top1_ret"] - df["top2_ret"]
    df["winner_is_top1"] = (df["top1_ret"] >= df["top2_ret"]).astype(int)
    return df


def load_daily_state_factors(path: Path, factor_names: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", *factor_names])
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date", as_index=False)[factor_names].first().sort_values("date").reset_index(drop=True)
    return daily


def max_drawdown(rets: pd.Series) -> float:
    s = pd.Series(rets).fillna(0.0)
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def summarize(rets: pd.Series) -> dict[str, float]:
    s = pd.Series(rets).fillna(0.0)
    return {
        "mean": float(s.mean()),
        "cum": float((1.0 + s).prod() - 1.0),
        "mdd": max_drawdown(s),
        "win": float((s > 0).mean()),
    }


def bucket_stats(df: pd.DataFrame, factor: str, q: int = 5) -> dict[str, float]:
    work = df[["date", factor, "top1_ret", "top2_ret", "spread_12"]].dropna()
    if len(work) < 20:
        return {}
    work = work.copy()
    work["bucket"] = pd.qcut(work[factor], q=q, labels=False, duplicates="drop")
    top_bucket = int(work["bucket"].max())
    bot_bucket = int(work["bucket"].min())
    high = work[work["bucket"] == top_bucket]
    low = work[work["bucket"] == bot_bucket]
    return {
        "high_top1": float(high["top1_ret"].mean()),
        "high_top2": float(high["top2_ret"].mean()),
        "high_spread": float(high["spread_12"].mean()),
        "low_top1": float(low["top1_ret"].mean()),
        "low_top2": float(low["top2_ret"].mean()),
        "low_spread": float(low["spread_12"].mean()),
        "spread_delta": float(high["spread_12"].mean() - low["spread_12"].mean()),
    }


def best_binary_switch(df: pd.DataFrame, factor: str) -> dict[str, float]:
    work = df[["date", factor, "top1_ret", "top2_ret"]].dropna()
    if len(work) < 30:
        return {}
    med = float(work[factor].median())
    high_choose_top1 = pd.Series(np.where(work[factor] >= med, work["top1_ret"], work["top2_ret"]))
    high_choose_top2 = pd.Series(np.where(work[factor] >= med, work["top2_ret"], work["top1_ret"]))
    s1 = summarize(high_choose_top1)
    s2 = summarize(high_choose_top2)
    if s1["cum"] >= s2["cum"]:
        return {
            "threshold": med,
            "rule": "high->top1, low->top2",
            **s1,
        }
    return {
        "threshold": med,
        "rule": "high->top2, low->top1",
        **s2,
    }


def analyze_one(label: str, top_df: pd.DataFrame, factor_df: pd.DataFrame) -> None:
    merged = top_df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    print(f"\n=== {label} ===")
    print(f"days={len(merged)} range={merged['date'].min().date()} ~ {merged['date'].max().date()}")
    base_top1 = summarize(merged["top1_ret"])
    base_top2 = summarize(merged["top2_ret"])
    print(
        f"BASE top1 mean={base_top1['mean']:.4%} cum={base_top1['cum']:.4f} mdd={base_top1['mdd']:.4%} | "
        f"top2 mean={base_top2['mean']:.4%} cum={base_top2['cum']:.4f} mdd={base_top2['mdd']:.4%}"
    )

    rows = []
    for factor in STATE_FACTORS:
        work = merged[[factor, "top1_ret", "top2_ret", "spread_12", "winner_is_top1"]].dropna()
        if len(work) < 30:
            continue
        bucket = bucket_stats(merged, factor)
        switch = best_binary_switch(merged, factor)
        rows.append({
            "factor": factor,
            "corr_top1": float(work[factor].corr(work["top1_ret"])),
            "corr_top2": float(work[factor].corr(work["top2_ret"])),
            "corr_spread": float(work[factor].corr(work["spread_12"])),
            "top1_win_rate_high_factor": float(work.loc[work[factor] >= work[factor].median(), "winner_is_top1"].mean()),
            **bucket,
            "switch_rule": switch.get("rule"),
            "switch_mean": switch.get("mean"),
            "switch_cum": switch.get("cum"),
            "switch_mdd": switch.get("mdd"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("no analyzable factor rows")
        return

    print("\n-- strongest by |corr_spread| --")
    cols = ["factor", "corr_top1", "corr_top2", "corr_spread", "high_spread", "low_spread", "spread_delta"]
    print(out.reindex(out["corr_spread"].abs().sort_values(ascending=False).index)[cols].head(8).to_string(index=False))

    print("\n-- best median-threshold top1/top2 switch factors (by cum) --")
    cols2 = ["factor", "switch_rule", "switch_mean", "switch_cum", "switch_mdd", "corr_spread", "spread_delta"]
    tmp = out.dropna(subset=["switch_cum"]).sort_values("switch_cum", ascending=False)
    print(tmp[cols2].head(8).to_string(index=False))


def main() -> None:
    factor_df = load_daily_state_factors(FACTOR_STORE, STATE_FACTORS)
    top3 = load_daily_top(JSON_3D)
    top2 = load_daily_top(JSON_2D)
    analyze_one("3D_WEIGHTED", top3, factor_df)
    analyze_one("2D_WEIGHTED", top2, factor_df)


if __name__ == "__main__":
    main()
