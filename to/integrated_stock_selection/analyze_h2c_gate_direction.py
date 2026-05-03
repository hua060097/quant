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
FACTOR = "high_to_close_drawdown"
QUANTILES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_EXPANDING_HISTORY = 20
CAL_RATIO = 0.6


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


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        row = {"date": pd.to_datetime(ds)}
        for k in range(1, 6):
            info = daily[f"top{k}"]
            row[f"ret{k}"] = float(info["real_return"])
        rows.append(row)
    top = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    factors = pd.read_parquet(FACTOR_PATH, columns=["date", "mkt_ma5_followthrough_v3", FACTOR])
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.groupby("date", as_index=False).first().sort_values("date").reset_index(drop=True)

    train = factors[(factors["date"] >= TRAIN_START) & (factors["date"] <= TRAIN_END)]
    q40 = float(train["mkt_ma5_followthrough_v3"].quantile(0.4))

    df = top.merge(factors, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["ret_q40"] = np.where(df["mkt_ma5_followthrough_v3"] >= q40, df["ret1"], df["ret2"])
    return df, train[FACTOR].dropna()


def print_bucket_stats(df: pd.DataFrame, base_col: str) -> None:
    print(f"\n=== {base_col.upper()} 分桶单调性 ===")
    bucket = pd.qcut(df[FACTOR], 5, labels=False, duplicates="drop")
    work = df.assign(bucket=bucket).dropna(subset=["bucket"])
    stats = []
    for b, grp in work.groupby("bucket"):
        s = summarize(grp[base_col])
        stats.append(
            {
                "bucket": int(b),
                "factor_min": float(grp[FACTOR].min()),
                "factor_max": float(grp[FACTOR].max()),
                "days": s["days"],
                "mean": s["mean"],
                "cum": s["cum"],
                "win": s["win"],
                "mdd": s["mdd"],
            }
        )
    out = pd.DataFrame(stats).sort_values("bucket")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def threshold_series_fixed(df: pd.DataFrame, train_factor: pd.Series, q: float) -> pd.Series:
    thr = float(train_factor.quantile(q))
    return pd.Series(thr, index=df.index, dtype=float)


def threshold_series_expanding(df: pd.DataFrame, q: float) -> pd.Series:
    return df[FACTOR].astype(float).expanding(min_periods=MIN_EXPANDING_HISTORY).quantile(q).shift(1)


def apply_gate(base_returns: pd.Series, signal: pd.Series, thresholds: pd.Series, orientation: str) -> pd.Series:
    valid = signal.notna() & thresholds.notna()
    keep = pd.Series(True, index=base_returns.index)
    if orientation == "high":
        keep.loc[valid] = signal.loc[valid] >= thresholds.loc[valid]
    else:
        keep.loc[valid] = signal.loc[valid] <= thresholds.loc[valid]
    return base_returns.where(keep, 0.0)


def evaluate(df: pd.DataFrame, train_factor: pd.Series, base_col: str) -> pd.DataFrame:
    cut = int(len(df) * CAL_RATIO)
    rows = []
    for method in ["fixed_train", "expanding"]:
        for q in QUANTILES:
            thresholds = threshold_series_fixed(df, train_factor, q) if method == "fixed_train" else threshold_series_expanding(df, q)
            for orientation in ["high", "low"]:
                gated = apply_gate(df[base_col], df[FACTOR], thresholds, orientation)
                cal = summarize(gated.iloc[:cut])
                val = summarize(gated.iloc[cut:])
                full = summarize(gated)
                rows.append(
                    {
                        "base": base_col,
                        "method": method,
                        "q": q,
                        "orientation": orientation,
                        "cal_cum": cal["cum"],
                        "val_cum": val["cum"],
                        "val_mean": val["mean"],
                        "val_mdd": val["mdd"],
                        "full_cum": full["cum"],
                        "full_mean": full["mean"],
                        "full_mdd": full["mdd"],
                    }
                )
    return pd.DataFrame(rows).sort_values(["val_cum", "full_cum"], ascending=False)


def main() -> None:
    df, train_factor = load_dataset()
    print(f"sample_days={len(df)} range={df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"factor_train_q20={train_factor.quantile(0.2):.6f} q50={train_factor.quantile(0.5):.6f} q80={train_factor.quantile(0.8):.6f}")

    for base_col in ["ret1", "ret_q40"]:
        base = summarize(df[base_col])
        print(
            f"\nBASE {base_col}: mean={base['mean']:.3%} cum={base['cum']:.4f} "
            f"win={base['win']:.1%} mdd={base['mdd']:.3%}"
        )
        print_bucket_stats(df, base_col)
        out = evaluate(df, train_factor, base_col).head(12)
        print(f"\n=== {base_col.upper()} 方向扫描 TOP 12 ===")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
