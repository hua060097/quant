from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
MKT = "mkt_ma5_followthrough_v3"
NH = "market_60d_new_high_ratio"


def summarize(returns: pd.Series) -> dict[str, float]:
    s = pd.Series(returns).dropna().astype(float)
    if s.empty:
        return {"days": 0, "mean": float("nan"), "cum": float("nan"), "mdd": float("nan"), "win": float("nan")}
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "days": int(len(s)),
        "mean": float(s.mean()),
        "cum": float(eq.iloc[-1] - 1.0),
        "mdd": float(dd.min()),
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


def load_factor_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", MKT, NH])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[[MKT, NH]].first().sort_values("date").reset_index(drop=True)


def build_expanding_threshold(series: pd.Series, q: float, fallback: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def apply_schemes(top_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    merged = top_df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    train = factor_df[(factor_df["date"] >= TRAIN_START) & (factor_df["date"] <= TRAIN_END)]
    mkt_q40 = float(train[MKT].dropna().quantile(0.4))
    mkt_q30 = float(train[MKT].dropna().quantile(0.3))
    nh_q70 = float(train[NH].dropna().quantile(0.7))
    exp_thr = build_expanding_threshold(merged[MKT], 0.3, mkt_q30)

    merged["scheme_top1"] = merged["top1_ret"]
    merged["scheme_mkt_q40"] = np.where(merged[MKT] >= mkt_q40, merged["top1_ret"], merged["top2_ret"])
    merged["scheme_mkt_exp_q30"] = np.where(merged[MKT] >= exp_thr, merged["top1_ret"], merged["top2_ret"])
    merged["scheme_dual"] = np.where(
        (merged[MKT] >= mkt_q40) & (merged[NH] >= nh_q70),
        merged["top1_ret"],
        merged["top2_ret"],
    )
    return merged


def print_summary(label: str, df: pd.DataFrame) -> None:
    print(f"=== {label} ===")
    if df.empty:
        print("no overlapping dates")
        print()
        return
    print(f"range {df['date'].min().date()} ~ {df['date'].max().date()} days={len(df)}")
    for col in ["scheme_top1", "scheme_mkt_q40", "scheme_mkt_exp_q30", "scheme_dual"]:
        st = summarize(df[col])
        print(f"{col}: mean={st['mean']:.4%} cum={st['cum']:.6f} mdd={st['mdd']:.4%} win={st['win']:.4%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gate schemes across two daily_top5_details json files.")
    parser.add_argument("--old", required=True, help="旧 json 文件路径")
    parser.add_argument("--new", required=True, help="新 json 文件路径")
    parser.add_argument("--factor-path", default=str(DEFAULT_FACTOR_PATH), help="factors.parquet 路径")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    factor_path = Path(args.factor_path)

    old_top = load_top_json(old_path)
    new_top = load_top_json(new_path)
    factor_df = load_factor_daily(factor_path)

    old_dates = set(old_top["date"])
    new_dates = set(new_top["date"])
    added_dates = sorted(new_dates - old_dates)

    merged = apply_schemes(new_top, factor_df)
    added_df = merged[merged["date"].isin(added_dates)].copy()

    print(f"old_days={len(old_top)} new_days={len(new_top)} added_days={len(added_dates)}")
    if added_dates:
        print("added_dates:", ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in added_dates))
    print()

    print_summary("added_only", added_df)
    print_summary("full_new_json", merged)

    if not added_df.empty:
        detail_cols = ["date", "top1_ret", "top2_ret", MKT, NH, "scheme_mkt_q40", "scheme_mkt_exp_q30", "scheme_dual"]
        print("=== added_details ===")
        print(added_df[detail_cols].to_string(index=False))


if __name__ == "__main__":
    main()
# daily_top5_details_Window_3_20260413_043438.json