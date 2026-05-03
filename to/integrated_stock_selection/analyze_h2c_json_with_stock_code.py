from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "results" / "daily_top5_details_Window_3_20260326_020635.json"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"

TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
RAW_FACTOR = "high_to_close_drawdown"
PCT_FACTOR = "high_to_close_drawdown_pct"
QUANTILES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def max_drawdown(returns: pd.Series) -> float:
    x = pd.Series(returns).dropna().astype(float)
    if x.empty:
        return float("nan")
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


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


def load_json_picks() -> pd.DataFrame:
    raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        date = pd.to_datetime(ds)
        for k in range(1, 6):
            info = daily[f"top{k}"]
            rows.append(
                {
                    "date": date,
                    "rank": k,
                    "stock_code": info.get("stock_code"),
                    "pred": float(info["pred"]) if info.get("pred") is not None else np.nan,
                    "real_return": float(info["real_return"]) if info.get("real_return") is not None else np.nan,
                }
            )
    return pd.DataFrame(rows).dropna(subset=["stock_code"]).sort_values(["date", "rank"]).reset_index(drop=True)


def load_factor_panel() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_parquet(FACTOR_PATH, columns=["date", "stock_code", RAW_FACTOR])
    df["date"] = pd.to_datetime(df["date"])
    df[RAW_FACTOR] = pd.to_numeric(df[RAW_FACTOR], errors="coerce")
    df = df.dropna(subset=[RAW_FACTOR]).copy()
    df[PCT_FACTOR] = df.groupby("date")[RAW_FACTOR].rank(method="average", pct=True)
    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    return df, train[RAW_FACTOR], train[PCT_FACTOR]


def apply_gate(df: pd.DataFrame, factor_col: str, threshold: float, orientation: str) -> pd.Series:
    if orientation == "low":
        keep = df[factor_col] <= threshold
    else:
        keep = df[factor_col] >= threshold
    return df["real_return"].where(keep, 0.0)


def evaluate(df: pd.DataFrame, train_raw: pd.Series, train_pct: pd.Series, rank_name: str) -> pd.DataFrame:
    rows = []
    for feature, train_dist in [(RAW_FACTOR, train_raw), (PCT_FACTOR, train_pct)]:
        for q in QUANTILES:
            threshold = float(train_dist.quantile(q))
            for orientation in ["low", "high"]:
                gated = apply_gate(df, feature, threshold, orientation)
                s = summarize(gated)
                rows.append(
                    {
                        "rank_group": rank_name,
                        "feature": feature,
                        "q": q,
                        "orientation": orientation,
                        "threshold": threshold,
                        "days": s["days"],
                        "mean": s["mean"],
                        "cum": s["cum"],
                        "win": s["win"],
                        "mdd": s["mdd"],
                    }
                )
    return pd.DataFrame(rows).sort_values(["cum", "mean"], ascending=False)


def main() -> None:
    picks = load_json_picks()
    panel, train_raw, train_pct = load_factor_panel()
    merged = picks.merge(panel[["date", "stock_code", RAW_FACTOR, PCT_FACTOR]], on=["date", "stock_code"], how="left")
    merged = merged.dropna(subset=[RAW_FACTOR, PCT_FACTOR, "real_return"]).copy()

    print(
        f"sample_days={merged['date'].nunique()} picks={len(merged)} "
        f"range={merged['date'].min().date()} ~ {merged['date'].max().date()}"
    )

    groups = {
        "top1": merged[merged["rank"] == 1].copy(),
        "q40_like_top12": merged[merged["rank"].isin([1, 2])].copy(),
    }

    # Approximate q40 choice using train fixed q40 from mkt_ma5 factor on current factor file.
    mkt = pd.read_parquet(FACTOR_PATH, columns=["date", "mkt_ma5_followthrough_v3"])
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.groupby("date", as_index=False).first()
    train_mkt = mkt[(mkt["date"] >= TRAIN_START) & (mkt["date"] <= TRAIN_END)]
    q40 = float(train_mkt["mkt_ma5_followthrough_v3"].quantile(0.4))
    mkt["q40_rank"] = np.where(mkt["mkt_ma5_followthrough_v3"] >= q40, 1, 2)
    merged = merged.merge(mkt[["date", "q40_rank"]], on="date", how="left")
    groups["q40"] = merged[merged["rank"] == merged["q40_rank"]].copy()

    for name, grp in groups.items():
        if grp.empty:
            continue
        base = summarize(grp["real_return"])
        print(
            f"\n=== {name.upper()} BASE ===\n"
            f"days={base['days']} mean={base['mean']:.3%} cum={base['cum']:.4f} "
            f"win={base['win']:.1%} mdd={base['mdd']:.3%}"
        )
        out = evaluate(grp, train_raw, train_pct, name).head(14)
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
