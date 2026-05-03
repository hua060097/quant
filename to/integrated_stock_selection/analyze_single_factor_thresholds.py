from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
DEFAULT_JSON = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
Q_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def summarize(s: pd.Series) -> dict[str, float]:
    s = pd.Series(s).dropna().astype(float)
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
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["top1_pos"] = (df["top1_ret"] > 0).astype(int)
    return df


def load_factor_daily(path: Path, factor: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", factor])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[[factor]].first().sort_values("date").reset_index(drop=True)


def build_expanding_threshold(series: pd.Series, q: float, fallback: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def build_rolling_threshold(series: pd.Series, q: float, window: int, fallback: float) -> pd.Series:
    vals = []
    min_obs = min(window, 20)
    for i in range(len(series)):
        hist = series.iloc[max(0, i - window):i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= min_obs else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan threshold modes for one factor against top1/top2 json.")
    parser.add_argument("--factor", required=True, help="因子名")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="daily_top5_details json 路径")
    parser.add_argument("--factor-path", default=str(FACTOR_PATH), help="factors.parquet 路径")
    args = parser.parse_args()

    factor = args.factor
    top_df = load_top_json(Path(args.json))
    factor_df = load_factor_daily(Path(args.factor_path), factor)
    merged = top_df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

    train = factor_df[(factor_df["date"] >= TRAIN_START) & (factor_df["date"] <= TRAIN_END)][factor].dropna()
    switch_rows = []
    gate_rows = []

    for q in Q_GRID:
        train_thr = float(train.quantile(q))
        threshold_defs: list[tuple[str, pd.Series | float]] = [
            (f"train_q{int(q * 100)}", train_thr),
            (f"expanding_q{int(q * 100)}_shift1", build_expanding_threshold(merged[factor], q, train_thr)),
            (f"rolling10_q{int(q * 100)}_shift1", build_rolling_threshold(merged[factor], q, 10, train_thr)),
            (f"rolling20_q{int(q * 100)}_shift1", build_rolling_threshold(merged[factor], q, 20, train_thr)),
        ]

        for mode, threshold in threshold_defs:
            threshold_series = threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=merged.index, dtype="float64")
            high = merged[factor] >= threshold_series

            ret_high_top1 = np.where(high, merged["top1_ret"], merged["top2_ret"])
            ret_high_top2 = np.where(high, merged["top2_ret"], merged["top1_ret"])
            ret_high_top1_cash = np.where(high, merged["top1_ret"], 0.0)
            ret_low_top1_cash = np.where(high, 0.0, merged["top1_ret"])

            switch_rows.append({"mode": mode, "direction": "high_top1_low_top2", **summarize(ret_high_top1)})
            switch_rows.append({"mode": mode, "direction": "high_top2_low_top1", **summarize(ret_high_top2)})

            high_df = merged.loc[high]
            low_df = merged.loc[~high]
            gate_rows.append(
                {
                    "mode": mode,
                    "direction": "high_top1_else_cash",
                    **summarize(ret_high_top1_cash),
                    "high_days": int(len(high_df)),
                    "low_days": int(len(low_df)),
                    "high_top1_pos": float(high_df["top1_pos"].mean()) if len(high_df) else float("nan"),
                    "low_top1_pos": float(low_df["top1_pos"].mean()) if len(low_df) else float("nan"),
                    "pos_delta": float(high_df["top1_pos"].mean() - low_df["top1_pos"].mean()) if len(high_df) and len(low_df) else float("nan"),
                }
            )
            gate_rows.append(
                {
                    "mode": mode,
                    "direction": "low_top1_else_cash",
                    **summarize(ret_low_top1_cash),
                    "high_days": int(len(high_df)),
                    "low_days": int(len(low_df)),
                    "high_top1_pos": float(high_df["top1_pos"].mean()) if len(high_df) else float("nan"),
                    "low_top1_pos": float(low_df["top1_pos"].mean()) if len(low_df) else float("nan"),
                    "pos_delta": float(low_df["top1_pos"].mean() - high_df["top1_pos"].mean()) if len(high_df) and len(low_df) else float("nan"),
                }
            )

    switch_df = pd.DataFrame(switch_rows)
    gate_df = pd.DataFrame(gate_rows)

    out_dir = ROOT / "results"
    stem = Path(args.json).stem
    safe_factor = factor.replace("/", "_")
    switch_csv = out_dir / f"{stem}_{safe_factor}_switch_scan.csv"
    gate_csv = out_dir / f"{stem}_{safe_factor}_top1_gate_scan.csv"
    switch_df.to_csv(switch_csv, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_csv, index=False, encoding="utf-8-sig")

    print("Best top1/top2 switch rules by cum:")
    print(
        switch_df.sort_values(["cum", "mean"], ascending=False)
        .head(20)[["mode", "direction", "mean", "cum", "mdd", "win"]]
        .to_string(index=False)
    )
    print()
    print("Best top1 gate rules by cum:")
    print(
        gate_df.sort_values(["cum", "mean"], ascending=False)
        .head(20)[["mode", "direction", "mean", "cum", "mdd", "win", "pos_delta"]]
        .to_string(index=False)
    )
    print()
    print(f"Saved: {switch_csv}")
    print(f"Saved: {gate_csv}")


if __name__ == "__main__":
    main()
