from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DAILY_DIR = BASE_DIR.parent / "local_data" / "features" / "daily"
SW2_PATH = BASE_DIR / "config" / "sw2_industry_encoding.json"


def load_sw2_map() -> dict[str, str]:
    raw = json.loads(SW2_PATH.read_text(encoding="utf-8"))
    return raw["mappings"]["stock_to_industry"]


def next_trade_date_map(start: str, end: str) -> dict[str, str]:
    days = sorted([p.stem for p in DAILY_DIR.glob("*.parquet") if p.stem.isdigit() and start <= p.stem <= end])
    return {days[i]: days[i + 1] for i in range(len(days) - 1)}


def build_raw_scheme_samples(top_n: int = 20) -> pd.DataFrame:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    rows: list[pd.DataFrame] = []
    for json_path in am.JSON_PATHS:
        label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"
        df = am.build_base_df(json_path, factor_df)
        gap_map = am.build_buy_gap_map(df)

        top1_raw = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
        trend = pd.to_numeric(df["market_trend_strength"], errors="coerce")
        disp = pd.to_numeric(df["mkt_sector_dispersion"], errors="coerce")
        trend_q50 = am.expanding_threshold(trend, 0.50)
        disp_q60 = am.expanding_threshold(disp, 0.60)
        weak = (trend <= trend_q50) & (disp >= disp_q60) & trend_q50.notna() & disp_q60.notna()
        raw_thr = pd.Series(np.where(weak, 0.25, 0.40), index=df.index)
        choose_top1 = top1_raw <= raw_thr
        selected_rank = pd.Series(np.where(choose_top1, 1, 2), index=df.index)
        selected_code = pd.Series(np.where(choose_top1, df["top1_code"], df["top2_code"]), index=df.index)
        selected_ret = pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index)
        filtered_ret, buy_skip = am.apply_buy_filter(df, selected_rank, selected_ret, gap_map)

        out = df[["date", "top1_code", "top1_ret", "top2_code", "top2_ret"]].copy()
        out["json_name"] = json_path.name
        out["label_version"] = label_version
        out["selected_rank"] = selected_rank
        out["selected_code"] = selected_code
        out["selected_ret"] = selected_ret
        out["filtered_ret"] = filtered_ret
        out["buy_skip"] = buy_skip
        out["top1_raw"] = top1_raw
        out["raw_threshold"] = raw_thr
        out["weak_env"] = weak
        rows.append(out)
    all_samples = pd.concat(rows, ignore_index=True)
    all_samples = all_samples[(all_samples["filtered_ret"].notna()) & (~all_samples["buy_skip"])].copy()
    return all_samples.sort_values("filtered_ret").head(top_n).reset_index(drop=True)


def load_day_with_open_gap(day: str, next_day: str, sw2_map: dict[str, str]) -> pd.DataFrame:
    d0 = pd.read_parquet(DAILY_DIR / f"{day}.parquet", columns=["stock_code", "close", "amount", "circ_mv"])
    d1 = pd.read_parquet(
        DAILY_DIR / f"{next_day}.parquet",
        columns=["stock_code", "open", "close", "pct_chg", "is_limit_down", "is_limit_up"],
    )
    df = d0.merge(d1, on="stock_code", how="inner", suffixes=("_t", "_t1"))
    df["open_gap"] = df["open"] / df["close_t"] - 1.0
    df["sw2"] = df["stock_code"].map(sw2_map)
    return df


def calc_market_day_stats(df: pd.DataFrame) -> dict[str, float]:
    return {
        "market_mean_open_gap": float(df["open_gap"].mean()),
        "market_median_open_gap": float(df["open_gap"].median()),
        "market_pct_open_le_m3": float((df["open_gap"] <= -0.03).mean()),
        "market_pct_open_le_m5": float((df["open_gap"] <= -0.05).mean()),
        "market_limit_down_count": int(df["is_limit_down"].fillna(0).sum()),
        "market_limit_down_rate": float(df["is_limit_down"].fillna(0).mean()),
        "market_day_drop7_count": int((pd.to_numeric(df["pct_chg"], errors="coerce") <= -7).sum()),
        "market_day_drop9_count": int((pd.to_numeric(df["pct_chg"], errors="coerce") <= -9.5).sum()),
    }


def calc_sector_stats(df: pd.DataFrame, sector: str) -> dict[str, float]:
    sub = df[df["sw2"] == sector].copy()
    if sub.empty:
        return {}
    return {
        "sector_name": sector,
        "sector_size": int(len(sub)),
        "sector_mean_open_gap": float(sub["open_gap"].mean()),
        "sector_median_open_gap": float(sub["open_gap"].median()),
        "sector_pct_open_le_m3": float((sub["open_gap"] <= -0.03).mean()),
        "sector_pct_open_le_m5": float((sub["open_gap"] <= -0.05).mean()),
        "sector_limit_down_count": int(sub["is_limit_down"].fillna(0).sum()),
        "sector_limit_down_rate": float(sub["is_limit_down"].fillna(0).mean()),
        "sector_day_drop7_count": int((pd.to_numeric(sub["pct_chg"], errors="coerce") <= -7).sum()),
        "sector_day_drop9_count": int((pd.to_numeric(sub["pct_chg"], errors="coerce") <= -9.5).sum()),
    }


def calc_sector_leaders(df: pd.DataFrame, sector: str, exclude_code: str) -> str:
    sub = df[(df["sw2"] == sector) & (df["stock_code"] != exclude_code)].copy()
    if sub.empty:
        return ""
    leaders = sub.sort_values(["amount", "circ_mv"], ascending=[False, False]).head(5)
    parts = []
    for _, r in leaders.iterrows():
        parts.append(
            f"{r['stock_code']}:gap={r['open_gap']:.2%},day={float(r['pct_chg']):.2f}%,LD={int(r['is_limit_down'])}"
        )
    return " | ".join(parts)


def build_market_baseline(start: str, end: str, sw2_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    nmap = next_trade_date_map(start, end)
    market_rows = []
    sector_rows = []
    for day, next_day in nmap.items():
        df = load_day_with_open_gap(day, next_day, sw2_map)
        m = calc_market_day_stats(df)
        m["date"] = pd.to_datetime(next_day)
        market_rows.append(m)
        for sector, sub in df.groupby("sw2"):
            if pd.isna(sector) or len(sub) < 5:
                continue
            sector_rows.append({
                "date": pd.to_datetime(next_day),
                "sector_name": sector,
                "sector_mean_open_gap": float(sub["open_gap"].mean()),
                "sector_pct_open_le_m3": float((sub["open_gap"] <= -0.03).mean()),
                "sector_pct_open_le_m5": float((sub["open_gap"] <= -0.05).mean()),
                "sector_limit_down_rate": float(sub["is_limit_down"].fillna(0).mean()),
                "sector_day_drop9_rate": float((pd.to_numeric(sub["pct_chg"], errors="coerce") <= -9.5).mean()),
            })
    return pd.DataFrame(market_rows), pd.DataFrame(sector_rows)


def percentile_of_value(hist: pd.Series, value: float, lower_is_worse: bool = True) -> float:
    h = hist.dropna()
    if h.empty or pd.isna(value):
        return np.nan
    if lower_is_worse:
        return float((h <= value).mean())
    return float((h >= value).mean())


def run() -> None:
    sw2_map = load_sw2_map()
    samples = build_raw_scheme_samples(top_n=20)
    start = samples["date"].min().strftime("%Y%m%d")
    end = "20260416"
    next_map = next_trade_date_map(start, end)
    market_base, sector_base = build_market_baseline(start, end, sw2_map)

    rows: list[dict] = []
    for _, r in samples.iterrows():
        day = r["date"].strftime("%Y%m%d")
        if day not in next_map:
            continue
        next_day = next_map[day]
        next_ts = pd.to_datetime(next_day)
        dd = load_day_with_open_gap(day, next_day, sw2_map)
        selected_sector = sw2_map.get(r["selected_code"], None)
        market_stats = calc_market_day_stats(dd)
        sector_stats = calc_sector_stats(dd, selected_sector) if selected_sector else {}

        out = {
            "json_name": r["json_name"],
            "label_version": r["label_version"],
            "signal_date": r["date"].strftime("%Y-%m-%d"),
            "next_trade_date": next_ts.strftime("%Y-%m-%d"),
            "selected_rank": int(r["selected_rank"]),
            "selected_code": r["selected_code"],
            "filtered_ret": float(r["filtered_ret"]),
            "top1_code": r["top1_code"],
            "top1_ret": float(r["top1_ret"]),
            "top2_code": r["top2_code"],
            "top2_ret": float(r["top2_ret"]),
            "top1_raw": float(r["top1_raw"]),
            "raw_threshold": float(r["raw_threshold"]),
            "weak_env": bool(r["weak_env"]),
            "selected_sw2": selected_sector,
        }
        out.update(market_stats)
        out.update(sector_stats)

        # market percentile in full window
        mb = market_base
        out["market_pct_open_le_m3_pctile"] = percentile_of_value(mb["market_pct_open_le_m3"], out["market_pct_open_le_m3"], lower_is_worse=False)
        out["market_pct_open_le_m5_pctile"] = percentile_of_value(mb["market_pct_open_le_m5"], out["market_pct_open_le_m5"], lower_is_worse=False)
        out["market_limit_down_rate_pctile"] = percentile_of_value(mb["market_limit_down_rate"], out["market_limit_down_rate"], lower_is_worse=False)
        out["market_mean_open_gap_pctile_worse"] = percentile_of_value(mb["market_mean_open_gap"], out["market_mean_open_gap"], lower_is_worse=True)

        if selected_sector:
            sb = sector_base[sector_base["sector_name"] == selected_sector]
            out["sector_mean_open_gap_pctile_worse"] = percentile_of_value(sb["sector_mean_open_gap"], out.get("sector_mean_open_gap"), lower_is_worse=True)
            out["sector_pct_open_le_m3_pctile"] = percentile_of_value(sb["sector_pct_open_le_m3"], out.get("sector_pct_open_le_m3"), lower_is_worse=False)
            out["sector_pct_open_le_m5_pctile"] = percentile_of_value(sb["sector_pct_open_le_m5"], out.get("sector_pct_open_le_m5"), lower_is_worse=False)
            out["sector_limit_down_rate_pctile"] = percentile_of_value(sb["sector_limit_down_rate"], out.get("sector_limit_down_rate"), lower_is_worse=False)
            out["sector_day_drop9_rate_pctile"] = percentile_of_value(sb["sector_day_drop9_rate"], out.get("sector_day_drop9_rate"), lower_is_worse=False)
            out["sector_top_other_liquid_names"] = calc_sector_leaders(dd, selected_sector, r["selected_code"])
        rows.append(out)

    out = pd.DataFrame(rows).sort_values("filtered_ret")
    out_path = RESULTS_DIR / "raw_scheme_open_shock_analysis_20260418.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(out[[
        "signal_date",
        "next_trade_date",
        "selected_code",
        "selected_sw2",
        "filtered_ret",
        "sector_mean_open_gap",
        "sector_pct_open_le_m3",
        "sector_limit_down_rate",
        "market_mean_open_gap",
        "market_pct_open_le_m3",
        "market_limit_down_count",
        "market_limit_down_rate",
        "sector_mean_open_gap_pctile_worse",
        "sector_pct_open_le_m3_pctile",
        "market_pct_open_le_m3_pctile",
        "market_limit_down_rate_pctile",
        "sector_top_other_liquid_names",
    ]].to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
