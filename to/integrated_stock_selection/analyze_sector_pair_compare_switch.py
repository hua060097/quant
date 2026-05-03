from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
MKT = "mkt_ma5_followthrough_v3"
SECTOR = "sector_ma5_followthrough_v3"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")
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
        top1 = daily.get("top1", {})
        top2 = daily.get("top2", {})
        rows.append(
            {
                "date": pd.to_datetime(ds),
                "top1_code": top1.get("stock_code"),
                "top2_code": top2.get("stock_code"),
                "top1_sector": top1.get("sw_sector"),
                "top2_sector": top2.get("sw_sector"),
                "top1_ret": float(top1.get("real_return", np.nan)),
                "top2_ret": float(top2.get("real_return", np.nan)),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["same_sector"] = df["top1_sector"].fillna("") == df["top2_sector"].fillna("")
    return df


def load_factor_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "stock_code", MKT, SECTOR])
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_factors(top_df: pd.DataFrame, factor_panel: pd.DataFrame) -> pd.DataFrame:
    daily_mkt = factor_panel.groupby("date", as_index=False)[[MKT]].first()
    top1_sector = factor_panel.rename(columns={"stock_code": "top1_code", SECTOR: "top1_sector_ma5"})
    top2_sector = factor_panel.rename(columns={"stock_code": "top2_code", SECTOR: "top2_sector_ma5"})
    merged = top_df.merge(daily_mkt, on="date", how="left")
    merged = merged.merge(top1_sector[["date", "top1_code", "top1_sector_ma5"]], on=["date", "top1_code"], how="left")
    merged = merged.merge(top2_sector[["date", "top2_code", "top2_sector_ma5"]], on=["date", "top2_code"], how="left")
    merged["sector_diff_12"] = merged["top1_sector_ma5"] - merged["top2_sector_ma5"]
    merged["sector_abs_diff_12"] = merged["sector_diff_12"].abs()
    return merged.sort_values("date").reset_index(drop=True)


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


def add_partitions(rows: list[dict], df: pd.DataFrame, ret_col: str, scheme: str) -> None:
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
    factor_panel = load_factor_panel(FACTOR_PATH)
    df = attach_factors(top_df, factor_panel)

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

    cross = df.loc[~df["same_sector"], "sector_abs_diff_12"].dropna()
    if cross.empty:
        raise RuntimeError("跨行业样本为空，无法分析")

    rows = []
    partition_rows = []

    direct_ret = np.where(df["same_sector"], df["baseline_ret"], np.where(df["sector_diff_12"] >= 0.0, df["top1_ret"], df["top2_ret"]))
    df["direct_compare_ret"] = direct_ret
    rows.append(
        {
            "mode": "direct_sign",
            "threshold_type": "none",
            "direction": "cross_compare_sign_else_mkt",
            **summarize(df["direct_compare_ret"]),
            "same_sector_days": int(df["same_sector"].sum()),
            "cross_sector_days": int((~df["same_sector"]).sum()),
            "top1_days": int(((~df["same_sector"]) & (df["sector_diff_12"] >= 0.0)).sum() + (df["same_sector"] & df["mkt_choose_top1"]).sum()),
            "top2_days": int(((~df["same_sector"]) & (df["sector_diff_12"] < 0.0)).sum() + (df["same_sector"] & ~df["mkt_choose_top1"]).sum()),
            "fallback_days": int(df["same_sector"].sum()),
        }
    )

    for q in Q_GRID:
        fallback = float(cross.quantile(q))
        threshold_defs = [
            (f"expanding_q{int(q * 100)}_shift1", build_expanding_threshold(df["sector_abs_diff_12"], q, fallback)),
            (f"rolling10_q{int(q * 100)}_shift1", build_rolling_threshold(df["sector_abs_diff_12"], q, 10, fallback)),
            (f"rolling20_q{int(q * 100)}_shift1", build_rolling_threshold(df["sector_abs_diff_12"], q, 20, fallback)),
        ]

        for mode, thr in threshold_defs:
            choose_sector = (~df["same_sector"]) & df["sector_abs_diff_12"].notna() & thr.notna() & (df["sector_abs_diff_12"] >= thr)
            choose_top1 = choose_sector & (df["sector_diff_12"] >= 0.0)
            choose_top2 = choose_sector & (df["sector_diff_12"] < 0.0)
            ret = np.where(
                choose_top1,
                df["top1_ret"],
                np.where(choose_top2, df["top2_ret"], df["baseline_ret"]),
            )
            rows.append(
                {
                    "mode": mode,
                    "threshold_type": "abs_diff",
                    "direction": "cross_absdiff_trigger_else_mkt",
                    **summarize(pd.Series(ret)),
                    "same_sector_days": int(df["same_sector"].sum()),
                    "cross_sector_days": int((~df["same_sector"]).sum()),
                    "top1_days": int(choose_top1.sum() + ((~choose_sector) & df["mkt_choose_top1"]).sum()),
                    "top2_days": int(choose_top2.sum() + ((~choose_sector) & ~df["mkt_choose_top1"]).sum()),
                    "fallback_days": int((~choose_sector).sum()),
                    "trigger_days": int(choose_sector.sum()),
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    best = result_df.iloc[0]

    if best["mode"] == "direct_sign":
        df["best_ret"] = df["direct_compare_ret"]
    else:
        best_q = int(str(best["mode"]).split("q")[1].split("_")[0]) / 100.0
        best_fallback = float(cross.quantile(best_q))
        if str(best["mode"]).startswith("expanding_"):
            best_thr = build_expanding_threshold(df["sector_abs_diff_12"], best_q, best_fallback)
        elif str(best["mode"]).startswith("rolling10_"):
            best_thr = build_rolling_threshold(df["sector_abs_diff_12"], best_q, 10, best_fallback)
        else:
            best_thr = build_rolling_threshold(df["sector_abs_diff_12"], best_q, 20, best_fallback)
        choose_sector = (~df["same_sector"]) & df["sector_abs_diff_12"].notna() & best_thr.notna() & (df["sector_abs_diff_12"] >= best_thr)
        choose_top1 = choose_sector & (df["sector_diff_12"] >= 0.0)
        choose_top2 = choose_sector & (df["sector_diff_12"] < 0.0)
        df["best_ret"] = np.where(
            choose_top1,
            df["top1_ret"],
            np.where(choose_top2, df["top2_ret"], df["baseline_ret"]),
        )
        df["best_trigger"] = choose_sector.astype(int)

    add_partitions(partition_rows, df, "baseline_ret", "mkt_exp_q30_baseline")
    add_partitions(partition_rows, df, "direct_compare_ret", "direct_sign_cross_else_mkt")
    add_partitions(partition_rows, df, "best_ret", f"{best['mode']}|{best['direction']}")
    partition_df = pd.DataFrame(partition_rows)

    out_dir = ROOT / "results"
    stem = JSON_PATH.stem
    result_path = out_dir / f"{stem}_sector_pair_compare_switch_scan.csv"
    partition_path = out_dir / f"{stem}_sector_pair_compare_partition.csv"
    detail_path = out_dir / f"{stem}_sector_pair_compare_detail.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(partition_path, index=False, encoding="utf-8-sig")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Best sector-pair compare rules:")
    print(
        result_df.head(20)[
            ["mode", "threshold_type", "direction", "mean", "cum", "mdd", "win", "trigger_days", "fallback_days"]
        ].to_string(index=False)
    )
    print()
    print("Partition compare:")
    print(partition_df.to_string(index=False))
    print()
    print(f"Saved: {result_path}")
    print(f"Saved: {partition_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
