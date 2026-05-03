from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"

EXTRA_MARKET_COLS = [
    "market_trend_strength",
    "mkt_sector_dispersion",
    "mkt_regime_trend_pressure_v2",
    "ma5_strength_breadth_v3",
    "market_volatility_context",
]


def expanding_pct(series: pd.Series, min_hist: int = 20) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = np.full(len(vals), np.nan)
    hist: list[float] = []
    for i, v in enumerate(vals):
        if len(hist) >= min_hist and pd.notna(v):
            s = pd.Series(hist)
            out[i] = float((s <= float(v)).mean())
        if pd.notna(v):
            hist.append(float(v))
    return pd.Series(out, index=series.index)


def calc_stats(series: pd.Series) -> tuple[float, float, float, float]:
    traded = series.dropna()
    if traded.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(traded.mean()),
        float((1 + traded).prod() - 1),
        am.calc_mdd(traded),
        float((traded > 0).mean()),
    )


def build_thresholds(df: pd.DataFrame) -> dict[str, pd.Series]:
    def s(name: str) -> pd.Series:
        for cand in [name, f"{name}_x", f"{name}_y"]:
            if cand in df.columns:
                return pd.to_numeric(df[cand], errors="coerce")
        raise KeyError(name)

    trend = s("market_trend_strength")
    disp = s("mkt_sector_dispersion")
    pressure = s("mkt_regime_trend_pressure_v2")
    breadth = s("ma5_strength_breadth_v3")
    vol = s("market_volatility_context")

    trend_pct = expanding_pct(trend)
    disp_pct = expanding_pct(disp)
    pressure_pct = expanding_pct(pressure)
    breadth_pct = expanding_pct(breadth)
    vol_pct = expanding_pct(vol)

    weak_td = ((1 - trend_pct) + disp_pct) / 2.0
    weak_tp = ((1 - trend_pct) + (1 - pressure_pct)) / 2.0
    weak_bv = ((1 - breadth_pct) + vol_pct) / 2.0
    weak_combo = ((1 - trend_pct) + disp_pct + vol_pct) / 3.0

    def linear(score: pd.Series, lo: float = 0.25, hi: float = 0.40) -> pd.Series:
        s = score.clip(0.0, 1.0)
        return hi - (hi - lo) * s

    def gated_linear(score: pd.Series, lo: float = 0.25, hi: float = 0.40, start: float = 0.50) -> pd.Series:
        s = ((score - start) / max(1e-9, (1.0 - start))).clip(0.0, 1.0)
        return hi - (hi - lo) * s

    def convex(score: pd.Series, lo: float = 0.25, hi: float = 0.40, power: float = 1.8) -> pd.Series:
        s = score.clip(0.0, 1.0) ** power
        return hi - (hi - lo) * s

    return {
        "step_hi0p40_lo0p25_td": pd.Series(np.where(weak_td >= 0.50, 0.25, 0.40), index=df.index),
        "linear_td_0p25_0p40": linear(weak_td, 0.25, 0.40),
        "gated_td_0p25_0p40_s50": gated_linear(weak_td, 0.25, 0.40, 0.50),
        "convex_td_0p25_0p40_p18": convex(weak_td, 0.25, 0.40, 1.8),
        "linear_tp_0p25_0p40": linear(weak_tp, 0.25, 0.40),
        "linear_bv_0p25_0p40": linear(weak_bv, 0.25, 0.40),
        "linear_combo_0p25_0p40": linear(weak_combo, 0.25, 0.40),
        "gated_combo_0p25_0p40_s55": gated_linear(weak_combo, 0.25, 0.40, 0.55),
    }


def run() -> None:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    extra_df = pd.read_parquet(FACTOR_PATH, columns=["date", "stock_code"] + EXTRA_MARKET_COLS).copy()
    extra_df["date"] = pd.to_datetime(extra_df["date"])
    extra_map = extra_df.groupby("date")[EXTRA_MARKET_COLS].first().reset_index()

    rows: list[dict] = []
    stability_rows: list[dict] = []

    for json_path in am.JSON_PATHS:
        label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"
        df = am.build_base_df(json_path, factor_df).merge(extra_map, on="date", how="left")
        gap_map = am.build_buy_gap_map(df)
        top1_raw = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
        top1_ret = pd.to_numeric(df["top1_ret"], errors="coerce")
        top2_ret = pd.to_numeric(df["top2_ret"], errors="coerce")

        threshold_map = build_thresholds(df)
        # baselines
        threshold_map["baseline_fixed_0p40"] = pd.Series(0.40, index=df.index)
        threshold_map["baseline_step_hi0p40_lo0p25_td"] = threshold_map["step_hi0p40_lo0p25_td"]

        for name, thr in threshold_map.items():
            rank = pd.Series(np.where(top1_raw <= thr, 1, 2), index=df.index)
            ret = pd.Series(np.where(top1_raw <= thr, top1_ret, top2_ret), index=df.index)
            filt_ret, buy_skip = am.apply_buy_filter(df, rank, ret, gap_map)

            for mode, series in [("no_buy_filter", ret), ("buy_filter_open_gap", filt_ret)]:
                mean_ret, cum_ret, mdd, win_rate = calc_stats(series)
                rows.append({
                    "json_name": json_path.name,
                    "label_version": label_version,
                    "scheme": name,
                    "filter_mode": mode,
                    "active_days": int(series.notna().sum()),
                    "pass_rate": float(series.notna().mean()),
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                    "win_rate": win_rate,
                })

            # stability only on buy-filter version
            work = pd.DataFrame({"date": df["date"], "ret": filt_ret})
            n = len(work)
            half = n // 2
            recent_n = min(13, n)
            parts = {
                "full": work,
                "first_half": work.iloc[:half],
                "second_half": work.iloc[half:],
                f"recent_{recent_n}d": work.iloc[-recent_n:],
            }
            for seg, part in parts.items():
                mean_ret, cum_ret, mdd, win_rate = calc_stats(part["ret"])
                stability_rows.append({
                    "json_name": json_path.name,
                    "label_version": label_version,
                    "scheme": name,
                    "segment": seg,
                    "active_days": int(part["ret"].notna().sum()),
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                    "win_rate": win_rate,
                })

    out = pd.DataFrame(rows)
    agg = (
        out.groupby(["scheme", "filter_mode"], as_index=False)
        .agg(
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_mdd=("max_drawdown", "mean"),
            worst_mdd=("max_drawdown", "min"),
            avg_win_rate=("win_rate", "mean"),
            avg_pass_rate=("pass_rate", "mean"),
        )
        .sort_values(["filter_mode", "avg_cum_ret"], ascending=[True, False])
        .reset_index(drop=True)
    )
    stability = pd.DataFrame(stability_rows)

    out_path = RESULTS_DIR / "smooth_reg_threshold_20260418.csv"
    agg_path = RESULTS_DIR / "smooth_reg_threshold_aggregate_20260418.csv"
    stability_path = RESULTS_DIR / "smooth_reg_threshold_stability_20260418.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
    stability.to_csv(stability_path, index=False, encoding="utf-8-sig")

    for mode in ["buy_filter_open_gap", "no_buy_filter"]:
        print(f"\n=== {mode} top 15 ===")
        print(
            agg[agg["filter_mode"] == mode]
            .head(15)[["scheme", "avg_cum_ret", "worst_cum_ret", "avg_mean_ret", "avg_mdd", "worst_mdd", "avg_pass_rate"]]
            .to_string(index=False)
        )
    print(f"\nSaved: {out_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {stability_path}")


if __name__ == "__main__":
    run()
