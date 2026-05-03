from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")
MKT = "mkt_ma5_followthrough_v3"
FACTOR = "max_rally_20d"
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
                "top1_code": daily.get("top1", {}).get("stock_code"),
                "top1_ret": float(daily.get("top1", {}).get("real_return", np.nan)),
                "top2_ret": float(daily.get("top2", {}).get("real_return", np.nan)),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["top1_pos"] = (df["top1_ret"] > 0).astype(int)
    return df


def load_factors(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "stock_code", MKT, FACTOR])
    df["date"] = pd.to_datetime(df["date"])
    return df


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


def attach_top1_daily_factor(top_df: pd.DataFrame, factor_panel: pd.DataFrame) -> pd.DataFrame:
    top1_factor = factor_panel.rename(columns={"stock_code": "top1_code", FACTOR: "top1_daily_max_rally_20d"})
    mkt_daily = factor_panel.groupby("date", as_index=False)[[MKT]].first()
    merged = top_df.merge(mkt_daily, on="date", how="left")
    merged = merged.merge(top1_factor[["date", "top1_code", "top1_daily_max_rally_20d"]], on=["date", "top1_code"], how="left")
    return merged.sort_values("date").reset_index(drop=True)


def add_partitions(rows: list[dict], df: pd.DataFrame, scheme_col: str, scheme_name: str) -> None:
    parts = {
        "full": df,
        "first_half": df.iloc[: len(df) // 2].copy(),
        "second_half": df.iloc[len(df) // 2 :].copy(),
        "added_10d": df[(df["date"] >= ADDED_START) & (df["date"] <= ADDED_END)].copy(),
    }
    for partition, part in parts.items():
        rows.append({"scheme": scheme_name, "partition": partition, **summarize(part[scheme_col])})


def main() -> None:
    top_df = load_top_json(JSON_PATH)
    factor_panel = load_factors(FACTOR_PATH)
    df = attach_top1_daily_factor(top_df, factor_panel)

    train_mask = (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)
    train_factor = df.loc[train_mask, "top1_daily_max_rally_20d"].dropna()
    train_mkt = (
        factor_panel.groupby("date", as_index=False)[[MKT]]
        .first()
        .loc[lambda x: (x["date"] >= TRAIN_START) & (x["date"] <= TRAIN_END), MKT]
        .dropna()
    )
    mkt_q30 = float(train_mkt.quantile(0.3))
    df["mkt_exp_q30_thr"] = build_expanding_threshold(df[MKT], 0.3, mkt_q30)
    df["mkt_choose_top1"] = df[MKT] >= df["mkt_exp_q30_thr"]

    switch_rows = []
    gate_rows = []
    combo_rows = []
    partition_rows = []

    for q in Q_GRID:
        train_thr = float(train_factor.quantile(q))
        threshold_defs: list[tuple[str, pd.Series | float]] = [
            (f"train_q{int(q * 100)}", train_thr),
            (f"expanding_q{int(q * 100)}_shift1", build_expanding_threshold(df["top1_daily_max_rally_20d"], q, train_thr)),
            (f"rolling10_q{int(q * 100)}_shift1", build_rolling_threshold(df["top1_daily_max_rally_20d"], q, 10, train_thr)),
            (f"rolling20_q{int(q * 100)}_shift1", build_rolling_threshold(df["top1_daily_max_rally_20d"], q, 20, train_thr)),
        ]

        for mode, threshold in threshold_defs:
            threshold_series = threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=df.index, dtype="float64")
            high = df["top1_daily_max_rally_20d"] >= threshold_series

            # standalone switch
            ret_high_top1 = np.where(high, df["top1_ret"], df["top2_ret"])
            ret_high_top2 = np.where(high, df["top2_ret"], df["top1_ret"])
            switch_rows.append({"mode": mode, "direction": "high_top1_low_top2", **summarize(ret_high_top1)})
            switch_rows.append({"mode": mode, "direction": "high_top2_low_top1", **summarize(ret_high_top2)})

            # standalone top1 gate
            ret_high_top1_cash = np.where(high, df["top1_ret"], 0.0)
            ret_low_top1_cash = np.where(high, 0.0, df["top1_ret"])
            high_df = df.loc[high]
            low_df = df.loc[~high]
            gate_rows.append(
                {
                    "mode": mode,
                    "direction": "high_top1_else_cash",
                    **summarize(ret_high_top1_cash),
                    "pass_days": int(high.sum()),
                    "pass_rate": float(high.mean()),
                    "filter_days": int((~high).sum()),
                    "filter_rate": float((~high).mean()),
                    "pos_delta": float(high_df["top1_pos"].mean() - low_df["top1_pos"].mean()) if len(high_df) and len(low_df) else float("nan"),
                }
            )
            gate_rows.append(
                {
                    "mode": mode,
                    "direction": "low_top1_else_cash",
                    **summarize(ret_low_top1_cash),
                    "pass_days": int((~high).sum()),
                    "pass_rate": float((~high).mean()),
                    "filter_days": int(high.sum()),
                    "filter_rate": float(high.mean()),
                    "pos_delta": float(low_df["top1_pos"].mean() - high_df["top1_pos"].mean()) if len(high_df) and len(low_df) else float("nan"),
                }
            )

            # mkt combo: if mkt says top1 then apply gate, else keep top2
            combo_high_allows = np.where(df["mkt_choose_top1"], np.where(high, df["top1_ret"], 0.0), df["top2_ret"])
            combo_low_allows = np.where(df["mkt_choose_top1"], np.where(~high, df["top1_ret"], 0.0), df["top2_ret"])
            combo_rows.append({"mode": mode, "direction": "mkt_then_high_top1_else_cash", **summarize(combo_high_allows)})
            combo_rows.append({"mode": mode, "direction": "mkt_then_low_top1_else_cash", **summarize(combo_low_allows)})

    switch_df = pd.DataFrame(switch_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    gate_df = pd.DataFrame(gate_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    combo_df = pd.DataFrame(combo_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)

    # partition compare for best standalone gate / best combo / mkt baseline
    best_gate = gate_df.iloc[0]
    best_combo = combo_df.iloc[0]

    best_gate_thr = None
    best_combo_thr = None
    # reconstruct chosen masks
    def threshold_from_mode(mode: str) -> pd.Series:
        if mode.startswith("train_q"):
            q = int(mode.split("train_q")[1]) / 100.0
            thr = float(train_factor.quantile(q))
            return pd.Series(thr, index=df.index, dtype="float64")
        if mode.startswith("expanding_q"):
            q = int(mode.split("expanding_q")[1].split("_")[0]) / 100.0
            thr = float(train_factor.quantile(q))
            return build_expanding_threshold(df["top1_daily_max_rally_20d"], q, thr)
        if mode.startswith("rolling10_q"):
            q = int(mode.split("rolling10_q")[1].split("_")[0]) / 100.0
            thr = float(train_factor.quantile(q))
            return build_rolling_threshold(df["top1_daily_max_rally_20d"], q, 10, thr)
        if mode.startswith("rolling20_q"):
            q = int(mode.split("rolling20_q")[1].split("_")[0]) / 100.0
            thr = float(train_factor.quantile(q))
            return build_rolling_threshold(df["top1_daily_max_rally_20d"], q, 20, thr)
        raise ValueError(mode)

    gate_thr = threshold_from_mode(str(best_gate["mode"]))
    gate_high = df["top1_daily_max_rally_20d"] >= gate_thr
    gate_col = "best_top1_gate"
    if best_gate["direction"] == "high_top1_else_cash":
        df[gate_col] = np.where(gate_high, df["top1_ret"], 0.0)
    else:
        df[gate_col] = np.where(gate_high, 0.0, df["top1_ret"])

    combo_thr = threshold_from_mode(str(best_combo["mode"]))
    combo_high = df["top1_daily_max_rally_20d"] >= combo_thr
    combo_col = "best_mkt_combo"
    if best_combo["direction"] == "mkt_then_high_top1_else_cash":
        df[combo_col] = np.where(df["mkt_choose_top1"], np.where(combo_high, df["top1_ret"], 0.0), df["top2_ret"])
    else:
        df[combo_col] = np.where(df["mkt_choose_top1"], np.where(combo_high, 0.0, df["top1_ret"]), df["top2_ret"])

    df["mkt_exp_q30"] = np.where(df["mkt_choose_top1"], df["top1_ret"], df["top2_ret"])
    add_partitions(partition_rows, df, "mkt_exp_q30", "mkt_exp_q30_baseline")
    add_partitions(partition_rows, df, gate_col, f"{best_gate['mode']}|{best_gate['direction']}")
    add_partitions(partition_rows, df, combo_col, f"{best_combo['mode']}|{best_combo['direction']}")
    partition_df = pd.DataFrame(partition_rows)

    out_dir = ROOT / "results"
    stem = Path(JSON_PATH).stem
    switch_path = out_dir / f"{stem}_top1_daily_maxrally20d_switch_scan.csv"
    gate_path = out_dir / f"{stem}_top1_daily_maxrally20d_top1_gate_scan.csv"
    combo_path = out_dir / f"{stem}_top1_daily_maxrally20d_mkt_combo_scan.csv"
    part_path = out_dir / f"{stem}_top1_daily_maxrally20d_partition_compare.csv"
    detail_path = out_dir / f"{stem}_top1_daily_maxrally20d_detail.csv"

    switch_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    combo_df.to_csv(combo_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(part_path, index=False, encoding="utf-8-sig")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Best standalone switch:")
    print(switch_df.head(15)[["mode", "direction", "mean", "cum", "mdd", "win"]].to_string(index=False))
    print()
    print("Best standalone top1 gate:")
    print(gate_df.head(15)[["mode", "direction", "mean", "cum", "mdd", "win", "pass_days", "pass_rate", "filter_days", "filter_rate"]].to_string(index=False))
    print()
    print("Best mkt+gate combo:")
    print(combo_df.head(15)[["mode", "direction", "mean", "cum", "mdd", "win"]].to_string(index=False))
    print()
    print("Partition compare:")
    print(partition_df.to_string(index=False))
    print()
    print(f"Saved: {switch_path}")
    print(f"Saved: {gate_path}")
    print(f"Saved: {combo_path}")
    print(f"Saved: {part_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
