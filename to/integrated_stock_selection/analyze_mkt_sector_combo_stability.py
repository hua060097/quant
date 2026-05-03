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
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")


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


def evaluate_partitions(df: pd.DataFrame, scheme_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    full = {"partition": "full", "data": df}
    first = {"partition": "first_half", "data": df.iloc[: len(df) // 2].copy()}
    second = {"partition": "second_half", "data": df.iloc[len(df) // 2 :].copy()}
    added = {"partition": "added_10d", "data": df[(df["date"] >= ADDED_START) & (df["date"] <= ADDED_END)].copy()}
    for block in [full, first, second, added]:
        for scheme in scheme_cols:
            rows.append({"partition": block["partition"], "scheme": scheme, **summarize(block["data"][scheme])})
    partition_df = pd.DataFrame(rows)

    month_rows = []
    work = df.copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    for month, grp in work.groupby("month", sort=True):
        for scheme in scheme_cols:
            month_rows.append({"month": month, "scheme": scheme, **summarize(grp[scheme])})
    month_df = pd.DataFrame(month_rows)

    def rolling_blocks(window: int) -> pd.DataFrame:
        out = []
        for start in range(0, len(df) - window + 1):
            grp = df.iloc[start:start + window]
            label = f"{grp['date'].iloc[0].date()}~{grp['date'].iloc[-1].date()}"
            for scheme in scheme_cols:
                out.append({"window": label, "window_size": window, "scheme": scheme, **summarize(grp[scheme])})
        return pd.DataFrame(out)

    roll20_df = rolling_blocks(20)

    stab_rows = []
    for scheme in scheme_cols:
        month_grp = month_df[month_df["scheme"] == scheme]
        roll_grp = roll20_df[roll20_df["scheme"] == scheme]
        stab_rows.append(
            {
                "scheme": scheme,
                "monthly_positive_ratio": float((month_grp["cum"] > 0).mean()),
                "monthly_mean_cum": float(month_grp["cum"].mean()),
                "monthly_worst_cum": float(month_grp["cum"].min()),
                "rolling20_positive_ratio": float((roll_grp["cum"] > 0).mean()),
                "rolling20_mean_cum": float(roll_grp["cum"].mean()),
                "rolling20_worst_cum": float(roll_grp["cum"].min()),
            }
        )
    return partition_df, month_df, roll20_df, pd.DataFrame(stab_rows)


def main() -> None:
    top_df = load_top_json(JSON_PATH)
    factor_panel = load_factor_panel(FACTOR_PATH)
    df = attach_factors(top_df, factor_panel)

    mkt_train_q30 = float(factor_panel[MKT].dropna().quantile(0.3))
    sector_hist = df["top1_sector_ma5"].dropna()
    sector_q50_fallback = float(sector_hist.quantile(0.5))
    sector_q40_fallback = float(sector_hist.quantile(0.4))

    mkt_exp_q30 = build_expanding_threshold(df[MKT], 0.3, mkt_train_q30)
    sector_roll10_q50 = build_rolling_threshold(df["top1_sector_ma5"], 0.5, 10, sector_q50_fallback)
    sector_exp_q40 = build_expanding_threshold(df["top1_sector_ma5"], 0.4, sector_q40_fallback)

    mkt_high = df[MKT] >= mkt_exp_q30
    sector_high_switch = df["top1_sector_ma5"] >= sector_roll10_q50
    sector_high_gate = df["top1_sector_ma5"] >= sector_exp_q40

    df["scheme_top1"] = df["top1_ret"]
    df["scheme_mkt_exp_q30"] = np.where(mkt_high, df["top1_ret"], df["top2_ret"])
    df["scheme_sector_roll10_q50"] = np.where(sector_high_switch, df["top1_ret"], df["top2_ret"])
    df["scheme_combo_and_switch"] = np.where(mkt_high & sector_high_switch, df["top1_ret"], df["top2_ret"])
    df["scheme_combo_or_switch"] = np.where(mkt_high | sector_high_switch, df["top1_ret"], df["top2_ret"])
    df["scheme_sector_exp_q40_gate"] = np.where(sector_high_gate, df["top1_ret"], 0.0)
    df["scheme_combo_and_gate"] = np.where(mkt_high & sector_high_gate, df["top1_ret"], 0.0)

    schemes = [
        "scheme_top1",
        "scheme_mkt_exp_q30",
        "scheme_sector_roll10_q50",
        "scheme_combo_and_switch",
        "scheme_combo_or_switch",
        "scheme_sector_exp_q40_gate",
        "scheme_combo_and_gate",
    ]

    partition_df, month_df, roll20_df, stab_df = evaluate_partitions(df, schemes)

    out_dir = ROOT / "results"
    part_path = out_dir / "mkt_sector_combo_partition_20260413.csv"
    month_path = out_dir / "mkt_sector_combo_monthly_20260413.csv"
    roll_path = out_dir / "mkt_sector_combo_rolling20_20260413.csv"
    stab_path = out_dir / "mkt_sector_combo_stability_20260413.csv"
    partition_df.to_csv(part_path, index=False, encoding="utf-8-sig")
    month_df.to_csv(month_path, index=False, encoding="utf-8-sig")
    roll20_df.to_csv(roll_path, index=False, encoding="utf-8-sig")
    stab_df.to_csv(stab_path, index=False, encoding="utf-8-sig")

    print("Partition comparison:")
    print(partition_df.to_string(index=False))
    print()
    print("Stability summary:")
    print(stab_df.sort_values(["rolling20_positive_ratio", "monthly_positive_ratio", "monthly_mean_cum"], ascending=False).to_string(index=False))
    print()
    print(f"Saved: {part_path}")
    print(f"Saved: {month_path}")
    print(f"Saved: {roll_path}")
    print(f"Saved: {stab_path}")


if __name__ == "__main__":
    main()
