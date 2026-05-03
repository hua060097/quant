from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
DEFAULT_JSON = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
FACTORS = [
    "abnormal_deviation_3d_ratio",
    "abnormal_event_count_10d",
    "severe_abnormal_pressure_10d",
    "severe_abnormal_pressure_30d",
]
Q_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")


def summarize(series: pd.Series) -> dict[str, float]:
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return {"days": 0, "mean": np.nan, "cum": np.nan, "mdd": np.nan, "win": np.nan}
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "days": int(len(s)),
        "mean": float(s.mean()),
        "cum": float(eq.iloc[-1] - 1.0),
        "mdd": float(dd.min()),
        "win": float((s > 0).mean()),
    }


def build_expanding_threshold(series: pd.Series, q: float) -> pd.Series:
    vals = []
    fallback = float(series.dropna().quantile(q)) if series.notna().any() else np.nan
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def build_rolling_threshold(series: pd.Series, q: float, window: int) -> pd.Series:
    vals = []
    fallback = float(series.dropna().quantile(q)) if series.notna().any() else np.nan
    min_obs = min(window, 20)
    for i in range(len(series)):
        hist = series.iloc[max(0, i - window):i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= min_obs else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def natural_thresholds(factor: str) -> list[tuple[str, float]]:
    if factor == "abnormal_deviation_3d_ratio":
        return [("nat_0p8", 0.8), ("nat_1p0", 1.0), ("nat_1p2", 1.2)]
    if factor == "abnormal_event_count_10d":
        return [("nat_1", 1.0), ("nat_2", 2.0), ("nat_3", 3.0), ("nat_4", 4.0)]
    if factor == "severe_abnormal_pressure_10d":
        return [("nat_0p5", 0.5), ("nat_0p8", 0.8), ("nat_1p0", 1.0)]
    if factor == "severe_abnormal_pressure_30d":
        return [("nat_0p5", 0.5), ("nat_0p8", 0.8), ("nat_1p0", 1.0)]
    return []


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
    df["spread12"] = df["top1_ret"] - df["top2_ret"]
    df["top1_pos"] = (df["top1_ret"] > 0).astype(int)
    return df


def load_factor_panel(path: Path, factors: list[str]) -> pd.DataFrame:
    cols = ["date", "stock_code"] + factors
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_factors(top_df: pd.DataFrame, factor_df: pd.DataFrame, factor: str) -> pd.DataFrame:
    top1 = factor_df[["date", "stock_code", factor]].rename(
        columns={"stock_code": "top1_code", factor: "top1_factor"}
    )
    top2 = factor_df[["date", "stock_code", factor]].rename(
        columns={"stock_code": "top2_code", factor: "top2_factor"}
    )
    merged = top_df.merge(top1, on=["date", "top1_code"], how="left")
    merged = merged.merge(top2, on=["date", "top2_code"], how="left")
    merged["factor_diff_12"] = merged["top1_factor"] - merged["top2_factor"]
    return merged


def partition_stats(df: pd.DataFrame, ret_col: str) -> list[dict]:
    parts = {
        "full": df,
        "first_half": df.iloc[: len(df) // 2],
        "second_half": df.iloc[len(df) // 2 :],
        "added_10d": df[(df["date"] >= ADDED_START) & (df["date"] <= ADDED_END)],
    }
    rows = []
    for name, part in parts.items():
        rows.append({"partition": name, **summarize(part[ret_col])})
    return rows


def scan_top1_series(df: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series = df["top1_factor"].astype("float64")
    switch_rows = []
    gate_rows = []
    partition_rows = []
    thr_defs: list[tuple[str, pd.Series]] = [("zero", pd.Series(0.0, index=df.index, dtype="float64"))]
    for name, value in natural_thresholds(factor):
        thr_defs.append((name, pd.Series(value, index=df.index, dtype="float64")))
    for q in Q_GRID:
        thr_defs.extend(
            [
                (f"expanding_q{int(q*100)}_shift1", build_expanding_threshold(series, q)),
                (f"rolling10_q{int(q*100)}_shift1", build_rolling_threshold(series, q, 10)),
                (f"rolling20_q{int(q*100)}_shift1", build_rolling_threshold(series, q, 20)),
            ]
        )

    for mode, thr in thr_defs:
        valid = series.notna() & thr.notna() & df["top1_ret"].notna() & df["top2_ret"].notna()
        if not valid.any():
            continue
        high = series >= thr

        ret_high_top2 = np.where(high, df["top2_ret"], df["top1_ret"])
        ret_low_top2 = np.where(high, df["top1_ret"], df["top2_ret"])
        ret_low_top1_cash = np.where(high, 0.0, df["top1_ret"])
        ret_high_top1_cash = np.where(high, df["top1_ret"], 0.0)

        switch_rows.append(
            {
                "factor": factor,
                "driver": "top1_factor",
                "mode": mode,
                "direction": "high_top2_low_top1",
                **summarize(pd.Series(ret_high_top2).where(valid)),
            }
        )
        switch_rows.append(
            {
                "factor": factor,
                "driver": "top1_factor",
                "mode": mode,
                "direction": "high_top1_low_top2",
                **summarize(pd.Series(ret_low_top2).where(valid)),
            }
        )
        gate_rows.append(
            {
                "factor": factor,
                "driver": "top1_factor",
                "mode": mode,
                "direction": "low_top1_else_cash",
                **summarize(pd.Series(ret_low_top1_cash).where(valid)),
                "pass_days": int((valid & ~high).sum()),
                "filter_days": int((valid & high).sum()),
                "pass_rate": float((valid & ~high).sum() / valid.sum()),
            }
        )
        gate_rows.append(
            {
                "factor": factor,
                "driver": "top1_factor",
                "mode": mode,
                "direction": "high_top1_else_cash",
                **summarize(pd.Series(ret_high_top1_cash).where(valid)),
                "pass_days": int((valid & high).sum()),
                "filter_days": int((valid & ~high).sum()),
                "pass_rate": float((valid & high).sum() / valid.sum()),
            }
        )

    switch_df = pd.DataFrame(switch_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    gate_df = pd.DataFrame(gate_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    if not switch_df.empty:
        best_switch = switch_df.iloc[0]
        df["best_switch_ret"] = materialize_top1_rule(df, factor, best_switch["mode"], best_switch["direction"])
        for row in partition_stats(df, "best_switch_ret"):
            partition_rows.append({"factor": factor, "rule_type": "switch", "scheme": f"{best_switch['mode']}|{best_switch['direction']}", **row})
    if not gate_df.empty:
        best_gate = gate_df.iloc[0]
        df["best_gate_ret"] = materialize_top1_rule(df, factor, best_gate["mode"], best_gate["direction"])
        for row in partition_stats(df, "best_gate_ret"):
            partition_rows.append({"factor": factor, "rule_type": "gate", "scheme": f"{best_gate['mode']}|{best_gate['direction']}", **row})
    return switch_df, gate_df, pd.DataFrame(partition_rows)


def materialize_top1_rule(df: pd.DataFrame, factor: str, mode: str, direction: str) -> pd.Series:
    series = df["top1_factor"].astype("float64")
    if mode == "zero":
        thr = pd.Series(0.0, index=df.index, dtype="float64")
    elif mode.startswith("nat_"):
        mapping = dict(natural_thresholds(factor))
        thr = pd.Series(mapping[mode], index=df.index, dtype="float64")
    elif mode.startswith("expanding_"):
        q = int(mode.split("q")[1].split("_")[0]) / 100.0
        thr = build_expanding_threshold(series, q)
    elif mode.startswith("rolling10_"):
        q = int(mode.split("q")[1].split("_")[0]) / 100.0
        thr = build_rolling_threshold(series, q, 10)
    else:
        q = int(mode.split("q")[1].split("_")[0]) / 100.0
        thr = build_rolling_threshold(series, q, 20)
    high = series >= thr
    if direction == "high_top2_low_top1":
        return pd.Series(np.where(high, df["top2_ret"], df["top1_ret"]), index=df.index, dtype="float64")
    if direction == "high_top1_low_top2":
        return pd.Series(np.where(high, df["top1_ret"], df["top2_ret"]), index=df.index, dtype="float64")
    if direction == "low_top1_else_cash":
        return pd.Series(np.where(high, 0.0, df["top1_ret"]), index=df.index, dtype="float64")
    return pd.Series(np.where(high, df["top1_ret"], 0.0), index=df.index, dtype="float64")


def scan_diff_series(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    series = df["factor_diff_12"].astype("float64")
    rows = []
    thr_defs = [("zero", pd.Series(0.0, index=df.index, dtype="float64"))]
    for q in Q_GRID:
        thr_defs.extend(
            [
                (f"expanding_q{int(q*100)}_shift1", build_expanding_threshold(series.abs(), q)),
                (f"rolling10_q{int(q*100)}_shift1", build_rolling_threshold(series.abs(), q, 10)),
                (f"rolling20_q{int(q*100)}_shift1", build_rolling_threshold(series.abs(), q, 20)),
            ]
        )
    for mode, thr in thr_defs:
        valid = series.notna() & thr.notna() & df["top1_ret"].notna() & df["top2_ret"].notna()
        if not valid.any():
            continue
        trigger = series.abs() >= thr
        choose_top1 = trigger & (series <= 0)  # 风险更低的一侧
        choose_top2 = trigger & (series > 0)
        ret = np.where(choose_top1, df["top1_ret"], np.where(choose_top2, df["top2_ret"], df["top1_ret"]))
        rows.append(
            {
                "factor": factor,
                "driver": "top1_top2_diff",
                "mode": mode,
                "direction": "trigger_choose_lower_risk_else_top1",
                **summarize(pd.Series(ret).where(valid)),
                "trigger_days": int((valid & trigger).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--factor-path", default=str(FACTOR_PATH))
    args = parser.parse_args()

    top_df = load_top_json(Path(args.json))
    factor_df = load_factor_panel(Path(args.factor_path), FACTORS)

    switch_parts = []
    gate_parts = []
    diff_parts = []
    partition_parts = []
    summary_rows = []

    for factor in FACTORS:
        merged = attach_factors(top_df, factor_df[["date", "stock_code", factor]], factor)
        switch_df, gate_df, part_df = scan_top1_series(merged.copy(), factor)
        diff_df = scan_diff_series(merged.copy(), factor)
        switch_parts.append(switch_df)
        gate_parts.append(gate_df)
        diff_parts.append(diff_df)
        partition_parts.append(part_df)

        top1_work = merged[["top1_factor", "top1_ret", "spread12", "top1_pos"]].dropna()
        summary_rows.append(
            {
                "factor": factor,
                "spearman_top1": float(top1_work["top1_factor"].corr(top1_work["top1_ret"], method="spearman")),
                "spearman_spread12": float(top1_work["top1_factor"].corr(top1_work["spread12"], method="spearman")),
                "spearman_top1_pos": float(top1_work["top1_factor"].corr(top1_work["top1_pos"], method="spearman")),
            }
        )

    switch_df = pd.concat(switch_parts, ignore_index=True)
    gate_df = pd.concat(gate_parts, ignore_index=True)
    diff_df = pd.concat(diff_parts, ignore_index=True)
    partition_df = pd.concat(partition_parts, ignore_index=True) if partition_parts else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    out_dir = ROOT / "results"
    stem = Path(args.json).stem
    switch_path = out_dir / f"{stem}_regulatory_risk_switch_scan.csv"
    gate_path = out_dir / f"{stem}_regulatory_risk_top1_gate_scan.csv"
    diff_path = out_dir / f"{stem}_regulatory_risk_diff_switch_scan.csv"
    part_path = out_dir / f"{stem}_regulatory_risk_partition.csv"
    summary_path = out_dir / f"{stem}_regulatory_risk_summary.csv"
    switch_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    diff_df.to_csv(diff_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(part_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Best switch rules")
    print(switch_df.head(20)[["factor", "driver", "mode", "direction", "mean", "cum", "mdd", "win"]].to_string(index=False))
    print()
    print("Best top1 gate rules")
    print(gate_df.head(20)[["factor", "driver", "mode", "direction", "mean", "cum", "mdd", "win", "pass_rate"]].to_string(index=False))
    print()
    print("Best diff-driven rules")
    print(diff_df.head(20)[["factor", "driver", "mode", "direction", "mean", "cum", "mdd", "win", "trigger_days"]].to_string(index=False))
    print()
    print("Correlation summary")
    print(summary_df.to_string(index=False))
    print()
    print(f"Saved: {switch_path}")
    print(f"Saved: {gate_path}")
    print(f"Saved: {diff_path}")
    print(f"Saved: {part_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
