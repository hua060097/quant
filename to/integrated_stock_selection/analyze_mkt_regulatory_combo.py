from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
MKT = "mkt_ma5_followthrough_v3"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")

FACTOR_CONFIG = [
    {
        "factor": "abnormal_deviation_3d_ratio",
        "modes": [("nat_0p8", 0.8), ("nat_1p0", 1.0), ("expanding_q60_shift1", 0.6), ("rolling20_q50_shift1", 0.5)],
        "prefer_high_risk": True,
    },
    {
        "factor": "severe_abnormal_pressure_10d",
        "modes": [("nat_0p8", 0.8), ("nat_1p0", 1.0), ("expanding_q60_shift1", 0.6), ("expanding_q70_shift1", 0.7)],
        "prefer_high_risk": True,
    },
]


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
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factor_panel(path: Path, extra_factors: list[str]) -> pd.DataFrame:
    cols = ["date", "stock_code", MKT] + extra_factors
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_factors(top_df: pd.DataFrame, factor_df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    daily_mkt = factor_df.groupby("date", as_index=False)[[MKT]].first()
    merged = top_df.merge(daily_mkt, on="date", how="left")
    for factor in factors:
        top1_factor = factor_df[["date", "stock_code", factor]].rename(
            columns={"stock_code": "top1_code", factor: f"top1_{factor}"}
        )
        merged = merged.merge(top1_factor, on=["date", "top1_code"], how="left")
    return merged.sort_values("date").reset_index(drop=True)


def add_partition_rows(rows: list[dict], df: pd.DataFrame, ret_col: str, scheme: str) -> None:
    parts = {
        "full": df,
        "first_half": df.iloc[: len(df) // 2].copy(),
        "second_half": df.iloc[len(df) // 2 :].copy(),
        "added_10d": df[(df["date"] >= ADDED_START) & (df["date"] <= ADDED_END)].copy(),
    }
    for name, part in parts.items():
        rows.append({"scheme": scheme, "partition": name, **summarize(part[ret_col])})


def main() -> None:
    top_df = load_top_json(JSON_PATH)
    factor_names = [cfg["factor"] for cfg in FACTOR_CONFIG]
    factor_panel = load_factor_panel(FACTOR_PATH, factor_names)
    df = attach_factors(top_df, factor_panel, factor_names)

    train_daily_mkt = (
        factor_panel.groupby("date", as_index=False)[[MKT]]
        .first()
        .loc[lambda x: (x["date"] >= TRAIN_START) & (x["date"] <= TRAIN_END), MKT]
        .dropna()
    )
    mkt_q30 = float(train_daily_mkt.quantile(0.3))
    df["mkt_thr"] = build_expanding_threshold(df[MKT], 0.3, mkt_q30)
    df["mkt_choose_top1"] = df[MKT] >= df["mkt_thr"]
    df["baseline_ret"] = np.where(df["mkt_choose_top1"], df["top1_ret"], df["top2_ret"])

    combo_rows = []
    partition_rows = []

    for cfg in FACTOR_CONFIG:
        factor = cfg["factor"]
        series = df[f"top1_{factor}"].astype("float64")
        for mode_name, param in cfg["modes"]:
            if mode_name.startswith("nat_"):
                thr = pd.Series(float(param), index=df.index, dtype="float64")
            elif mode_name.startswith("expanding_"):
                fallback = float(series.dropna().quantile(float(param))) if series.notna().any() else np.nan
                thr = build_expanding_threshold(series, float(param), fallback)
            else:
                fallback = float(series.dropna().quantile(float(param))) if series.notna().any() else np.nan
                thr = build_rolling_threshold(series, float(param), 20, fallback)

            high_risk = series >= thr

            top1_cash = np.where(
                df["mkt_choose_top1"],
                np.where(high_risk, 0.0, df["top1_ret"]),
                df["top2_ret"],
            )
            top1_top2 = np.where(
                df["mkt_choose_top1"],
                np.where(high_risk, df["top2_ret"], df["top1_ret"]),
                df["top2_ret"],
            )

            combo_rows.append(
                {
                    "factor": factor,
                    "mode": mode_name,
                    "direction": "mkt_top1_then_lowrisk_top1_else_cash",
                    **summarize(pd.Series(top1_cash)),
                    "top1_pass_days": int((df["mkt_choose_top1"] & ~high_risk).sum()),
                    "top1_cash_days": int((df["mkt_choose_top1"] & high_risk).sum()),
                    "top2_days": int((~df["mkt_choose_top1"]).sum()),
                }
            )
            combo_rows.append(
                {
                    "factor": factor,
                    "mode": mode_name,
                    "direction": "mkt_top1_then_highrisk_top2_else_top1",
                    **summarize(pd.Series(top1_top2)),
                    "top1_days": int((df["mkt_choose_top1"] & ~high_risk).sum()),
                    "top2_from_mkt_days": int((~df["mkt_choose_top1"]).sum()),
                    "top2_from_risk_days": int((df["mkt_choose_top1"] & high_risk).sum()),
                }
            )

    combo_df = pd.DataFrame(combo_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)

    best_cash = combo_df[combo_df["direction"] == "mkt_top1_then_lowrisk_top1_else_cash"].iloc[0]
    best_switch = combo_df[combo_df["direction"] == "mkt_top1_then_highrisk_top2_else_top1"].iloc[0]

    def materialize(best_row: pd.Series) -> pd.Series:
        factor = best_row["factor"]
        mode_name = best_row["mode"]
        series = df[f"top1_{factor}"].astype("float64")
        if mode_name.startswith("nat_"):
            thr = pd.Series(float(mode_name.split("nat_")[1].replace("p", ".")), index=df.index, dtype="float64")
        elif mode_name.startswith("expanding_"):
            q = float(best_row["mode"].split("q")[1].split("_")[0]) / 100.0
            fallback = float(series.dropna().quantile(q)) if series.notna().any() else np.nan
            thr = build_expanding_threshold(series, q, fallback)
        else:
            q = float(best_row["mode"].split("q")[1].split("_")[0]) / 100.0
            fallback = float(series.dropna().quantile(q)) if series.notna().any() else np.nan
            thr = build_rolling_threshold(series, q, 20, fallback)
        high_risk = series >= thr
        if best_row["direction"] == "mkt_top1_then_lowrisk_top1_else_cash":
            return pd.Series(
                np.where(df["mkt_choose_top1"], np.where(high_risk, 0.0, df["top1_ret"]), df["top2_ret"]),
                index=df.index,
                dtype="float64",
            )
        return pd.Series(
            np.where(df["mkt_choose_top1"], np.where(high_risk, df["top2_ret"], df["top1_ret"]), df["top2_ret"]),
            index=df.index,
            dtype="float64",
        )

    df["best_cash_ret"] = materialize(best_cash)
    df["best_switch_ret"] = materialize(best_switch)

    add_partition_rows(partition_rows, df, "baseline_ret", "mkt_exp_q30_baseline")
    add_partition_rows(partition_rows, df, "best_cash_ret", f"{best_cash['factor']}|{best_cash['mode']}|{best_cash['direction']}")
    add_partition_rows(partition_rows, df, "best_switch_ret", f"{best_switch['factor']}|{best_switch['mode']}|{best_switch['direction']}")
    partition_df = pd.DataFrame(partition_rows)

    out_dir = ROOT / "results"
    stem = JSON_PATH.stem
    combo_path = out_dir / f"{stem}_mkt_regulatory_combo_scan.csv"
    partition_path = out_dir / f"{stem}_mkt_regulatory_combo_partition.csv"
    detail_path = out_dir / f"{stem}_mkt_regulatory_combo_detail.csv"
    combo_df.to_csv(combo_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(partition_path, index=False, encoding="utf-8-sig")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Best combos")
    print(combo_df.head(20).to_string(index=False))
    print()
    print("Partition compare")
    print(partition_df.to_string(index=False))
    print()
    print(f"Saved: {combo_path}")
    print(f"Saved: {partition_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
