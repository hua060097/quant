from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = ROOT / "results" / "daily_top5_details_Window_3_20260413_154759.json"
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
        preds = [daily.get(f"top{i}", {}).get("pred", np.nan) for i in range(1, 6)]
        top1_ret = float(daily.get("top1", {}).get("real_return", np.nan))
        top2_ret = float(daily.get("top2", {}).get("real_return", np.nan))
        rows.append(
            {
                "date": pd.to_datetime(ds),
                "top1_ret": top1_ret,
                "top2_ret": top2_ret,
                "top1_pos": int(top1_ret > 0) if pd.notna(top1_ret) else np.nan,
                "pred_gap_12": float(preds[0] - preds[1]) if pd.notna(preds[0]) and pd.notna(preds[1]) else np.nan,
                "pred_gap_1_mean25": (
                    float(preds[0] - np.mean(preds[1:5]))
                    if pd.notna(preds[0]) and np.isfinite(preds[1:5]).all()
                    else np.nan
                ),
                "pred_gap_1_mean25_rel": (
                    float((preds[0] - np.mean(preds[1:5])) / max(abs(float(preds[0])), 1e-9))
                    if pd.notna(preds[0]) and np.isfinite(preds[1:5]).all()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


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
    switch_rows: list[dict[str, float | str | int]] = []
    gate_rows: list[dict[str, float | str | int]] = []
    series = df[factor].astype("float64")

    threshold_defs: list[tuple[str, pd.Series]] = []
    if factor.endswith("_rel"):
        threshold_defs.append(("zero", pd.Series(0.0, index=df.index, dtype="float64")))
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
        top1_cash_low = np.where(high, 0.0, df["top1_ret"])

        switch_rows.append(
            {
                "factor": factor,
                "mode": mode,
                "direction": "high_top1_low_top2",
                **summarize(pd.Series(ret_high_top1).where(valid)),
            }
        )
        switch_rows.append(
            {
                "factor": factor,
                "mode": mode,
                "direction": "high_top2_low_top1",
                **summarize(pd.Series(ret_high_top2).where(valid)),
            }
        )

        high_df = df.loc[valid & high].copy()
        low_df = df.loc[valid & ~high].copy()
        gate_rows.append(
            {
                "factor": factor,
                "mode": mode,
                "direction": "high_top1_else_cash",
                **summarize(pd.Series(top1_cash_high).where(valid)),
                "high_days": int(len(high_df)),
                "low_days": int(len(low_df)),
                "high_top1_pos": float(high_df["top1_pos"].mean()) if len(high_df) else float("nan"),
                "low_top1_pos": float(low_df["top1_pos"].mean()) if len(low_df) else float("nan"),
                "pos_delta": (
                    float(high_df["top1_pos"].mean() - low_df["top1_pos"].mean())
                    if len(high_df) and len(low_df)
                    else float("nan")
                ),
            }
        )
        gate_rows.append(
            {
                "factor": factor,
                "mode": mode,
                "direction": "low_top1_else_cash",
                **summarize(pd.Series(top1_cash_low).where(valid)),
                "high_days": int(len(high_df)),
                "low_days": int(len(low_df)),
                "high_top1_pos": float(high_df["top1_pos"].mean()) if len(high_df) else float("nan"),
                "low_top1_pos": float(low_df["top1_pos"].mean()) if len(low_df) else float("nan"),
                "pos_delta": (
                    float(low_df["top1_pos"].mean() - high_df["top1_pos"].mean())
                    if len(high_df) and len(low_df)
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(switch_rows), pd.DataFrame(gate_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pred gap factors from daily_top5_details json.")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="daily_top5_details json 路径")
    args = parser.parse_args()

    json_path = Path(args.json)
    df = load_top_json(json_path)

    all_switch = []
    all_gate = []
    for factor in ["pred_gap_12", "pred_gap_1_mean25", "pred_gap_1_mean25_rel"]:
        switch_df, gate_df = scan_factor(df, factor)
        all_switch.append(switch_df)
        all_gate.append(gate_df)

    switch_df = pd.concat(all_switch, ignore_index=True)
    gate_df = pd.concat(all_gate, ignore_index=True)

    out_dir = ROOT / "results"
    stem = json_path.stem
    switch_path = out_dir / f"{stem}_pred_gap_switch_scan.csv"
    gate_path = out_dir / f"{stem}_pred_gap_top1_gate_scan.csv"
    switch_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")

    print("Best top1/top2 switch by factor:")
    print(
        switch_df.sort_values(["cum", "mean"], ascending=False)
        .groupby("factor", as_index=False)
        .head(10)[["factor", "mode", "direction", "mean", "cum", "mdd", "win"]]
        .to_string(index=False)
    )
    print()
    print("Best top1 gate by factor:")
    print(
        gate_df.sort_values(["cum", "mean"], ascending=False)
        .groupby("factor", as_index=False)
        .head(10)[["factor", "mode", "direction", "mean", "cum", "mdd", "win", "pos_delta"]]
        .to_string(index=False)
    )
    print()
    print(f"Saved: {switch_path}")
    print(f"Saved: {gate_path}")


if __name__ == "__main__":
    main()
