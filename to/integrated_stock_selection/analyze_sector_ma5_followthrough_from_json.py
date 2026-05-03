from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
DEFAULT_JSON = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
FACTOR = "sector_ma5_followthrough_v3"
Q_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def summarize(returns: pd.Series) -> dict[str, float]:
    s = pd.Series(returns).dropna().astype(float)
    if s.empty:
        return {
            "days": 0,
            "mean": float("nan"),
            "cum": float("nan"),
            "mdd": float("nan"),
            "win": float("nan"),
        }
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
                "top1_code": daily.get("top1", {}).get("stock_code"),
                "top2_code": daily.get("top2", {}).get("stock_code"),
                "top1_ret": float(daily.get("top1", {}).get("real_return", np.nan)),
                "top2_ret": float(daily.get("top2", {}).get("real_return", np.nan)),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["top1_pos"] = (df["top1_ret"] > 0).astype(int)
    df["spread12"] = df["top1_ret"] - df["top2_ret"]
    return df


def load_factor_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "stock_code", FACTOR])
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_top_factor(df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    top1_fac = factor_df.rename(columns={"stock_code": "top1_code", FACTOR: "top1_sector_ma5_followthrough_v3"})
    top2_fac = factor_df.rename(columns={"stock_code": "top2_code", FACTOR: "top2_sector_ma5_followthrough_v3"})
    merged = df.merge(top1_fac, on=["date", "top1_code"], how="left")
    merged = merged.merge(top2_fac, on=["date", "top2_code"], how="left")
    merged["sector_ma5_followthrough_diff_12"] = (
        merged["top1_sector_ma5_followthrough_v3"] - merged["top2_sector_ma5_followthrough_v3"]
    )
    return merged


def build_expanding_threshold(series: pd.Series, q: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else np.nan)
    return pd.Series(vals, index=series.index, dtype="float64")


def build_rolling_threshold(series: pd.Series, q: float, window: int) -> pd.Series:
    vals = []
    min_obs = min(window, 20)
    for i in range(len(series)):
        hist = series.iloc[max(0, i - window):i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= min_obs else np.nan)
    return pd.Series(vals, index=series.index, dtype="float64")


def scan_factor(df: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    switch_rows = []
    gate_rows = []
    series = df[factor].astype("float64")
    threshold_defs = [("zero", pd.Series(0.0, index=df.index, dtype="float64"))]
    for q in Q_GRID:
        threshold_defs.extend(
            [
                (f"expanding_q{int(q * 100)}_shift1", build_expanding_threshold(series, q)),
                (f"rolling10_q{int(q * 100)}_shift1", build_rolling_threshold(series, q, 10)),
                (f"rolling20_q{int(q * 100)}_shift1", build_rolling_threshold(series, q, 20)),
            ]
        )

    for mode, thr in threshold_defs:
        valid = series.notna() & thr.notna() & df["top1_ret"].notna() & df["top2_ret"].notna()
        if not valid.any():
            continue
        high = series >= thr
        ret_high_top1 = np.where(high, df["top1_ret"], df["top2_ret"])
        ret_high_top2 = np.where(high, df["top2_ret"], df["top1_ret"])
        top1_cash_high = np.where(high, df["top1_ret"], 0.0)
        high_df = df.loc[valid & high]
        low_df = df.loc[valid & ~high]

        switch_rows.append(
            {"factor": factor, "mode": mode, "direction": "high_top1_low_top2", **summarize(pd.Series(ret_high_top1).where(valid))}
        )
        switch_rows.append(
            {"factor": factor, "mode": mode, "direction": "high_top2_low_top1", **summarize(pd.Series(ret_high_top2).where(valid))}
        )
        gate_rows.append(
            {
                "factor": factor,
                "mode": mode,
                "direction": "high_top1_else_cash",
                **summarize(pd.Series(top1_cash_high).where(valid)),
                "high_days": int(len(high_df)),
                "low_days": int(len(low_df)),
                "pos_delta": (
                    float(high_df["top1_pos"].mean() - low_df["top1_pos"].mean())
                    if len(high_df) and len(low_df)
                    else float("nan")
                ),
                "high_top1_mean": float(high_df["top1_ret"].mean()) if len(high_df) else float("nan"),
                "low_top1_mean": float(low_df["top1_ret"].mean()) if len(low_df) else float("nan"),
            }
        )
    return pd.DataFrame(switch_rows), pd.DataFrame(gate_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sector_ma5_followthrough_v3 against top1/top2 json.")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="daily_top5_details json 路径")
    parser.add_argument("--factor-path", default=str(FACTOR_PATH), help="factors.parquet 路径")
    args = parser.parse_args()

    top_df = load_top_json(Path(args.json))
    factor_df = load_factor_panel(Path(args.factor_path))
    merged = attach_top_factor(top_df, factor_df)

    switch_parts = []
    gate_parts = []
    for factor in [
        "top1_sector_ma5_followthrough_v3",
        "top2_sector_ma5_followthrough_v3",
        "sector_ma5_followthrough_diff_12",
    ]:
        switch_df, gate_df = scan_factor(merged, factor)
        switch_parts.append(switch_df)
        gate_parts.append(gate_df)

    switch_df = pd.concat(switch_parts, ignore_index=True)
    gate_df = pd.concat(gate_parts, ignore_index=True)

    out_dir = ROOT / "results"
    stem = Path(args.json).stem
    switch_path = out_dir / f"{stem}_sector_ma5_followthrough_switch_scan.csv"
    gate_path = out_dir / f"{stem}_sector_ma5_followthrough_top1_gate_scan.csv"
    detail_path = out_dir / f"{stem}_sector_ma5_followthrough_detail.csv"
    switch_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    merged.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Baseline")
    for col in ["top1_ret", "top2_ret"]:
        st = summarize(merged[col])
        print(f"{col}: mean={st['mean']:.4%} cum={st['cum']:.6f} mdd={st['mdd']:.4%} win={st['win']:.4%}")
    print()
    print("Best switch rules")
    print(
        switch_df.sort_values(["cum", "mean"], ascending=False)
        .head(20)[["factor", "mode", "direction", "mean", "cum", "mdd", "win"]]
        .to_string(index=False)
    )
    print()
    print("Best top1 gate rules")
    print(
        gate_df.sort_values(["cum", "mean"], ascending=False)
        .head(20)[["factor", "mode", "direction", "mean", "cum", "mdd", "win", "pos_delta"]]
        .to_string(index=False)
    )
    print()
    print("Correlation check")
    for factor in [
        "top1_sector_ma5_followthrough_v3",
        "top2_sector_ma5_followthrough_v3",
        "sector_ma5_followthrough_diff_12",
    ]:
        work = merged[[factor, "top1_ret", "spread12", "top1_pos"]].dropna()
        if len(work) < 20:
            continue
        print(
            f"{factor}: "
            f"spearman_top1={work[factor].corr(work['top1_ret'], method='spearman'):.4f} "
            f"spearman_spread12={work[factor].corr(work['spread12'], method='spearman'):.4f} "
            f"spearman_top1_pos={work[factor].corr(work['top1_pos'], method='spearman'):.4f}"
        )
    print()
    print(f"Saved: {switch_path}")
    print(f"Saved: {gate_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
