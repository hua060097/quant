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
        rows.append(
            {
                "date": pd.to_datetime(ds),
                "top1_code": daily.get("top1", {}).get("stock_code"),
                "top1_ret": float(daily.get("top1", {}).get("real_return", np.nan)),
                "top2_ret": float(daily.get("top2", {}).get("real_return", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factor_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "stock_code", MKT, SECTOR])
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


def attach_factors(top_df: pd.DataFrame, factor_panel: pd.DataFrame) -> pd.DataFrame:
    daily_mkt = factor_panel.groupby("date", as_index=False)[[MKT]].first()
    top1_sector = factor_panel.rename(columns={"stock_code": "top1_code", SECTOR: "top1_sector_ma5"})
    merged = top_df.merge(daily_mkt, on="date", how="left")
    merged = merged.merge(top1_sector[["date", "top1_code", "top1_sector_ma5"]], on=["date", "top1_code"], how="left")
    return merged.sort_values("date").reset_index(drop=True)


def add_partition_rows(rows: list[dict], df: pd.DataFrame, col: str, scheme: str) -> None:
    parts = {
        "full": df,
        "first_half": df.iloc[: len(df) // 2].copy(),
        "second_half": df.iloc[len(df) // 2 :].copy(),
        "added_10d": df[(df["date"] >= ADDED_START) & (df["date"] <= ADDED_END)].copy(),
    }
    for name, part in parts.items():
        rows.append({"scheme": scheme, "partition": name, **summarize(part[col])})


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
    df["mkt_exp_q30_thr"] = build_expanding_threshold(df[MKT], 0.3, mkt_q30)
    df["mkt_choose_top1"] = df[MKT] >= df["mkt_exp_q30_thr"]
    df["scheme_mkt_exp_q30"] = np.where(df["mkt_choose_top1"], df["top1_ret"], df["top2_ret"])

    sector_hist = df["top1_sector_ma5"].dropna()
    if sector_hist.empty:
        raise RuntimeError("top1_sector_ma5 全为空，无法分析")

    combo_rows = []
    partition_rows = []

    for q in Q_GRID:
        fallback = float(sector_hist.quantile(q))
        threshold_defs = [
            (f"expanding_q{int(q * 100)}_shift1", build_expanding_threshold(df["top1_sector_ma5"], q, fallback)),
            (f"rolling10_q{int(q * 100)}_shift1", build_rolling_threshold(df["top1_sector_ma5"], q, 10, fallback)),
            (f"rolling20_q{int(q * 100)}_shift1", build_rolling_threshold(df["top1_sector_ma5"], q, 20, fallback)),
        ]

        for mode, thr in threshold_defs:
            high = df["top1_sector_ma5"] >= thr

            combo_high = np.where(
                df["mkt_choose_top1"],
                np.where(high, df["top1_ret"], 0.0),
                df["top2_ret"],
            )
            combo_low = np.where(
                df["mkt_choose_top1"],
                np.where(high, 0.0, df["top1_ret"]),
                df["top2_ret"],
            )

            combo_rows.append(
                {
                    "mode": mode,
                    "direction": "mkt_top1_then_high_top1_else_cash",
                    **summarize(combo_high),
                    "top1_pass_days": int((df["mkt_choose_top1"] & high).sum()),
                    "top1_cash_days": int((df["mkt_choose_top1"] & ~high).sum()),
                    "top2_days": int((~df["mkt_choose_top1"]).sum()),
                }
            )
            combo_rows.append(
                {
                    "mode": mode,
                    "direction": "mkt_top1_then_low_top1_else_cash",
                    **summarize(combo_low),
                    "top1_pass_days": int((df["mkt_choose_top1"] & ~high).sum()),
                    "top1_cash_days": int((df["mkt_choose_top1"] & high).sum()),
                    "top2_days": int((~df["mkt_choose_top1"]).sum()),
                }
            )

    combo_df = pd.DataFrame(combo_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    best = combo_df.iloc[0]

    best_q = int(str(best["mode"]).split("q")[1].split("_")[0]) / 100.0
    best_fallback = float(sector_hist.quantile(best_q))
    if str(best["mode"]).startswith("expanding_"):
        best_thr = build_expanding_threshold(df["top1_sector_ma5"], best_q, best_fallback)
    elif str(best["mode"]).startswith("rolling10_"):
        best_thr = build_rolling_threshold(df["top1_sector_ma5"], best_q, 10, best_fallback)
    else:
        best_thr = build_rolling_threshold(df["top1_sector_ma5"], best_q, 20, best_fallback)

    best_high = df["top1_sector_ma5"] >= best_thr
    if best["direction"] == "mkt_top1_then_high_top1_else_cash":
        df["best_combo"] = np.where(df["mkt_choose_top1"], np.where(best_high, df["top1_ret"], 0.0), df["top2_ret"])
    else:
        df["best_combo"] = np.where(df["mkt_choose_top1"], np.where(best_high, 0.0, df["top1_ret"]), df["top2_ret"])

    add_partition_rows(partition_rows, df, "scheme_mkt_exp_q30", "mkt_exp_q30_baseline")
    add_partition_rows(partition_rows, df, "best_combo", f"{best['mode']}|{best['direction']}")
    partition_df = pd.DataFrame(partition_rows)

    out_dir = ROOT / "results"
    stem = Path(JSON_PATH).stem
    combo_path = out_dir / f"{stem}_sector_low_freedom_combo_scan.csv"
    part_path = out_dir / f"{stem}_sector_low_freedom_combo_partition.csv"
    detail_path = out_dir / f"{stem}_sector_low_freedom_combo_detail.csv"
    combo_df.to_csv(combo_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(part_path, index=False, encoding="utf-8-sig")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Best standalone sector switch/gate candidates still reference:")
    print("switch: top1_sector_ma5_followthrough_v3 rolling10_q50_shift1 high->top1 low->top2")
    print("gate:   top1_sector_ma5_followthrough_v3 expanding_q40_shift1 high->top1 else cash")
    print()
    print("Best low-freedom mkt+sector combo:")
    print(combo_df.head(20)[["mode", "direction", "mean", "cum", "mdd", "win", "top1_pass_days", "top1_cash_days", "top2_days"]].to_string(index=False))
    print()
    print("Partition compare:")
    print(partition_df.to_string(index=False))
    print()
    print(f"Saved: {combo_path}")
    print(f"Saved: {part_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
