from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_103953.json"

TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
FACTOR = "market_60d_new_high_ratio"
Q_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def summarize(s: pd.Series) -> dict[str, float]:
    s = pd.Series(s).dropna().astype(float)
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
                "top1_ret": float(daily.get("top1", {}).get("real_return", np.nan)),
                "top2_ret": float(daily.get("top2", {}).get("real_return", np.nan)),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["top1_pos"] = (df["top1_ret"] > 0).astype(int)
    return df


def load_factor_daily(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", FACTOR])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False)[[FACTOR]].first().sort_values("date").reset_index(drop=True)


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
    top_df = load_top_json(JSON_PATH)
    factor_df = load_factor_daily(FACTOR_PATH)
    merged = top_df.merge(factor_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

    train = factor_df[(factor_df["date"] >= TRAIN_START) & (factor_df["date"] <= TRAIN_END)][FACTOR].dropna()
    switch_rows = []
    sign_rows = []

    for q in Q_GRID:
        train_thr = float(train.quantile(q))
        threshold_defs: list[tuple[str, pd.Series | float]] = [
            (f"train_q{int(q * 100)}", train_thr),
            (f"expanding_q{int(q * 100)}_shift1", build_expanding_threshold(merged[FACTOR], q, train_thr)),
            (f"rolling10_q{int(q * 100)}_shift1", build_rolling_threshold(merged[FACTOR], q, 10, train_thr)),
            (f"rolling20_q{int(q * 100)}_shift1", build_rolling_threshold(merged[FACTOR], q, 20, train_thr)),
        ]

        for mode, threshold in threshold_defs:
            threshold_series = threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=merged.index, dtype="float64")
            high = merged[FACTOR] >= threshold_series

            ret_high_top1 = np.where(high, merged["top1_ret"], merged["top2_ret"])
            ret_high_top2 = np.where(high, merged["top2_ret"], merged["top1_ret"])

            switch_rows.append(
                {
                    "mode": mode,
                    "direction": "high_top1_low_top2",
                    **summarize(ret_high_top1),
                }
            )
            switch_rows.append(
                {
                    "mode": mode,
                    "direction": "high_top2_low_top1",
                    **summarize(ret_high_top2),
                }
            )

            high_df = merged.loc[high]
            low_df = merged.loc[~high]
            sign_rows.append(
                {
                    "mode": mode,
                    "high_days": int(len(high_df)),
                    "low_days": int(len(low_df)),
                    "high_top1_pos": float(high_df["top1_pos"].mean()) if len(high_df) else float("nan"),
                    "low_top1_pos": float(low_df["top1_pos"].mean()) if len(low_df) else float("nan"),
                    "pos_delta": float(high_df["top1_pos"].mean() - low_df["top1_pos"].mean()) if len(high_df) and len(low_df) else float("nan"),
                    "high_top1_mean": float(high_df["top1_ret"].mean()) if len(high_df) else float("nan"),
                    "low_top1_mean": float(low_df["top1_ret"].mean()) if len(low_df) else float("nan"),
                    "mean_delta": float(high_df["top1_ret"].mean() - low_df["top1_ret"].mean()) if len(high_df) and len(low_df) else float("nan"),
                }
            )

    switch_df = pd.DataFrame(switch_rows)
    sign_df = pd.DataFrame(sign_rows)

    out_dir = ROOT / "results"
    switch_csv = out_dir / "market_new_high_ratio_switch_scan_20260413.csv"
    sign_csv = out_dir / "market_new_high_ratio_top1_sign_scan_20260413.csv"
    switch_df.to_csv(switch_csv, index=False, encoding="utf-8-sig")
    sign_df.to_csv(sign_csv, index=False, encoding="utf-8-sig")

    print("Best top1/top2 switch rules by cum:")
    print(
        switch_df.sort_values(["cum", "mean"], ascending=False)
        .head(20)[["mode", "direction", "mean", "cum", "mdd", "win"]]
        .to_string(index=False)
    )
    print()
    print("Best top1 sign split by pos_delta:")
    print(
        sign_df.sort_values(["pos_delta", "mean_delta"], ascending=False)
        .head(20)[["mode", "high_top1_pos", "low_top1_pos", "pos_delta", "high_top1_mean", "low_top1_mean", "mean_delta"]]
        .to_string(index=False)
    )
    print()
    print(f"Saved: {switch_csv}")
    print(f"Saved: {sign_csv}")


if __name__ == "__main__":
    main()
