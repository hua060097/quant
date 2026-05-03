from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
JSON_PATH = RESULTS_DIR / "daily_top5_details_Window_3_20260325_230521.json"
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


def load_json_returns() -> pd.DataFrame:
    raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        row = {"date": pd.to_datetime(ds)}
        for k in range(1, 6):
            row[f"ret{k}"] = float(daily[f"top{k}"]["real_return"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_pred_stock_files() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("pred_stocks_*.csv")):
        if "final_pick" in path.name:
            continue
        try:
            date_str = path.stem.split("_")[-1]
            pred_date = pd.to_datetime(date_str)
        except Exception:
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "stock_code" not in df.columns or RAW_FACTOR not in df.columns:
            continue

        df["prediction"] = pd.to_numeric(df.get("prediction"), errors="coerce")
        df[RAW_FACTOR] = pd.to_numeric(df[RAW_FACTOR], errors="coerce")
        df = df.dropna(subset=["prediction", RAW_FACTOR]).copy()
        if df.empty:
            continue

        df = df.sort_values("prediction", ascending=False).reset_index(drop=True)
        df[PCT_FACTOR] = df[RAW_FACTOR].rank(method="average", pct=True)

        top1 = df.iloc[0]
        rows.append(
            {
                "date": pred_date,
                "base": "top1",
                "rank": 1,
                "stock_code": top1["stock_code"],
                RAW_FACTOR: float(top1[RAW_FACTOR]),
                PCT_FACTOR: float(top1[PCT_FACTOR]),
            }
        )

        if "q40_selected" in df.columns:
            chosen = df[df["q40_selected"].astype(bool)]
            if not chosen.empty:
                q = chosen.iloc[0]
                rows.append(
                    {
                        "date": pred_date,
                        "base": "q40",
                        "rank": int(df.index[df["stock_code"] == q["stock_code"]][0]) + 1,
                        "stock_code": q["stock_code"],
                        RAW_FACTOR: float(q[RAW_FACTOR]),
                        PCT_FACTOR: float(q[PCT_FACTOR]),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["date", "base"]).reset_index(drop=True)


def load_train_distributions() -> tuple[pd.Series, pd.Series]:
    cols = ["date", RAW_FACTOR]
    df = pd.read_parquet(FACTOR_PATH, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df[RAW_FACTOR] = pd.to_numeric(df[RAW_FACTOR], errors="coerce")
    df = df.dropna(subset=[RAW_FACTOR])
    df[PCT_FACTOR] = df.groupby("date")[RAW_FACTOR].rank(method="average", pct=True)
    return df[RAW_FACTOR], df[PCT_FACTOR]


def apply_gate(df: pd.DataFrame, factor_col: str, threshold: float, orientation: str) -> pd.Series:
    if orientation == "low":
        keep = df[factor_col] <= threshold
    else:
        keep = df[factor_col] >= threshold
    return df["ret"].where(keep, 0.0)


def evaluate_base(base_df: pd.DataFrame, train_raw: pd.Series, train_pct: pd.Series) -> pd.DataFrame:
    rows = []
    for factor_col, train_dist in [(RAW_FACTOR, train_raw), (PCT_FACTOR, train_pct)]:
        for q in QUANTILES:
            threshold = float(train_dist.quantile(q))
            for orientation in ["low", "high"]:
                gated = apply_gate(base_df, factor_col, threshold, orientation)
                s = summarize(gated)
                rows.append(
                    {
                        "feature": factor_col,
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
    ret_df = load_json_returns()
    picks = load_pred_stock_files()
    if picks.empty:
        print("No overlapping pred_stocks files found.")
        return

    train_raw, train_pct = load_train_distributions()
    merged = picks.merge(ret_df, on="date", how="inner")
    merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce").astype("Int64")
    merged["ret"] = np.select(
        [
            merged["rank"] == 1,
            merged["rank"] == 2,
            merged["rank"] == 3,
            merged["rank"] == 4,
            merged["rank"] == 5,
        ],
        [
            merged["ret1"],
            merged["ret2"],
            merged["ret3"],
            merged["ret4"],
            merged["ret5"],
        ],
        default=np.nan,
    )

    print(
        f"overlap_days_top1={merged[merged['base']=='top1']['date'].nunique()} "
        f"overlap_days_q40={merged[merged['base']=='q40']['date'].nunique()}"
    )

    for base in ["top1", "q40"]:
        base_df = merged[merged["base"] == base].copy().sort_values("date").reset_index(drop=True)
        if base_df.empty:
            continue
        base_stats = summarize(base_df["ret"])
        print(
            f"\n=== {base.upper()} OVERLAP BASE ===\n"
            f"days={base_stats['days']} mean={base_stats['mean']:.3%} cum={base_stats['cum']:.4f} "
            f"win={base_stats['win']:.1%} mdd={base_stats['mdd']:.3%}"
        )
        out = evaluate_base(base_df, train_raw, train_pct).head(16)
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
