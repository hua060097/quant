from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
EXTRA_MARKET_COLS = [
    "market_trend_adaptive_beta",
    "ma5_strength_breadth_v3",
    "market_volatility_context",
    "strong_stock_volume_confirmation_v3",
    "sector_hotspot_concentration_v3",
]


def summarize(series: pd.Series) -> tuple[float, float, float]:
    traded = series.dropna()
    if traded.empty:
        return np.nan, np.nan, np.nan
    mean_ret = float(traded.mean())
    cum_ret = float((1 + traded).prod() - 1)
    mdd = am.calc_mdd(traded)
    return mean_ret, cum_ret, mdd


def build_market_state_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}

    trend_beta = pd.to_numeric(df["market_trend_adaptive_beta"], errors="coerce")
    trend_strength = pd.to_numeric(df["market_trend_strength"], errors="coerce")
    trend_pressure = pd.to_numeric(df["mkt_regime_trend_pressure_v2"], errors="coerce")
    breadth = pd.to_numeric(df["ma5_strength_breadth_v3"], errors="coerce")
    volatility = pd.to_numeric(df["market_volatility_context"], errors="coerce")
    strong_vol = pd.to_numeric(df["strong_stock_volume_confirmation_v3"], errors="coerce")
    dispersion = pd.to_numeric(df["mkt_sector_dispersion"], errors="coerce")
    hotspot = pd.to_numeric(df["sector_hotspot_concentration_v3"], errors="coerce")

    beta_q50 = am.expanding_threshold(trend_beta, 0.50)
    strength_q50 = am.expanding_threshold(trend_strength, 0.50)
    pressure_q50 = am.expanding_threshold(trend_pressure, 0.50)
    breadth_q40 = am.expanding_threshold(breadth, 0.40)
    vol_q60 = am.expanding_threshold(volatility, 0.60)
    strong_vol_q40 = am.expanding_threshold(strong_vol, 0.40)
    disp_q60 = am.expanding_threshold(dispersion, 0.60)
    hotspot_q70 = am.expanding_threshold(hotspot, 0.70)

    out["weak_trend_core"] = (
        (trend_beta <= beta_q50) &
        (trend_strength <= strength_q50) &
        beta_q50.notna() & strength_q50.notna()
    )
    out["weak_trend_pressure"] = (
        (trend_beta <= beta_q50) &
        (trend_pressure <= pressure_q50) &
        beta_q50.notna() & pressure_q50.notna()
    )
    out["weak_trend_dispersion"] = (
        (trend_strength <= strength_q50) &
        (dispersion >= disp_q60) &
        strength_q50.notna() & disp_q60.notna()
    )
    out["weak_breadth_vol"] = (
        (breadth <= breadth_q40) &
        (volatility >= vol_q60) &
        breadth_q40.notna() & vol_q60.notna()
    )
    out["weak_quality_hot"] = (
        (strong_vol <= strong_vol_q40) &
        (hotspot >= hotspot_q70) &
        strong_vol_q40.notna() & hotspot_q70.notna()
    )
    out["weak_trend_quality3"] = (
        (trend_pressure <= pressure_q50) &
        (trend_strength <= strength_q50) &
        (strong_vol <= strong_vol_q40) &
        pressure_q50.notna() & strength_q50.notna() & strong_vol_q40.notna()
    )
    return out


def run() -> None:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    rows: list[dict] = []

    for json_path in am.JSON_PATHS:
        df = am.build_base_df(json_path, factor_df)
        missing_cols = [c for c in EXTRA_MARKET_COLS if c not in df.columns]
        if missing_cols:
            extra_df = pd.read_parquet(FACTOR_PATH, columns=["date", "stock_code"] + missing_cols).copy()
            extra_df["date"] = pd.to_datetime(extra_df["date"])
            extra_map = extra_df.groupby("date")[missing_cols].first().reset_index()
            df = df.merge(extra_map, on="date", how="left")
        gap_map = am.build_buy_gap_map(df)
        label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"

        top1_pct = pd.to_numeric(df["top1__severe_abnormal_pressure_10d__pct"], errors="coerce")
        top1_raw = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
        top1_ret = pd.to_numeric(df["top1_ret"], errors="coerce")
        top2_ret = pd.to_numeric(df["top2_ret"], errors="coerce")

        # Baselines
        baseline_defs: list[tuple[str, pd.Series, pd.Series, str]] = []
        state_thr = pd.Series(np.where(
            (pd.to_numeric(df["mkt_ma5_followthrough_v3"], errors="coerce") >= am.expanding_threshold(df["mkt_ma5_followthrough_v3"], 0.30)) &
            am.expanding_threshold(df["mkt_ma5_followthrough_v3"], 0.30).notna(),
            0.95,
            0.85,
        ), index=df.index)
        baseline_defs.append((
            "switch_reg_pct_mktstate_hi95_lo85",
            pd.Series(np.where(top1_pct >= state_thr, 2, 1), index=df.index),
            pd.Series(np.where(top1_pct >= state_thr, top2_ret, top1_ret), index=df.index),
            "baseline_pct_state",
        ))
        baseline_defs.append((
            "switch_top1_reg_pct95_to_top2",
            pd.Series(np.where(top1_pct >= 0.95, 2, 1), index=df.index),
            pd.Series(np.where(top1_pct >= 0.95, top2_ret, top1_ret), index=df.index),
            "baseline_pct_fixed",
        ))
        baseline_defs.append((
            "switch_top1_raw_le_0p30_else_top2",
            pd.Series(np.where(top1_raw <= 0.30, 1, 2), index=df.index),
            pd.Series(np.where(top1_raw <= 0.30, top1_ret, top2_ret), index=df.index),
            "baseline_raw_fixed",
        ))

        market_masks = build_market_state_masks(df)

        candidates: list[tuple[str, pd.Series, pd.Series, str]] = []
        for weak_name, weak_mask in market_masks.items():
            for weak_thr in [0.80, 0.85]:
                strong_thr = 0.95
                pct_thr = pd.Series(np.where(weak_mask, weak_thr, strong_thr), index=df.index)
                rank = pd.Series(np.where(top1_pct >= pct_thr, 2, 1), index=df.index)
                ret = pd.Series(np.where(top1_pct >= pct_thr, top2_ret, top1_ret), index=df.index)
                candidates.append((
                    f"switch_reg_pct_{weak_name}_hi95_lo{int(weak_thr*100)}",
                    rank,
                    ret,
                    "pct_state_variant",
                ))

            for weak_raw in [0.25, 0.30, 0.35]:
                strong_raw = 0.40
                raw_thr = pd.Series(np.where(weak_mask, weak_raw, strong_raw), index=df.index)
                rank = pd.Series(np.where(top1_raw <= raw_thr, 1, 2), index=df.index)
                ret = pd.Series(np.where(top1_raw <= raw_thr, top1_ret, top2_ret), index=df.index)
                candidates.append((
                    f"switch_reg_raw_{weak_name}_lo{str(weak_raw).replace('.', 'p')}_hi0p40",
                    rank,
                    ret,
                    "raw_state_variant",
                ))

        for scheme_name, rank, ret, family in baseline_defs + candidates:
            for mode in ["no_buy_filter", "buy_filter_open_gap"]:
                if mode == "buy_filter_open_gap":
                    series, buy_skip = am.apply_buy_filter(df, rank, ret, gap_map)
                else:
                    series = ret
                    buy_skip = pd.Series(False, index=df.index)
                mean_ret, cum_ret, mdd = summarize(series)
                rows.append({
                    "json_name": json_path.name,
                    "label_version": label_version,
                    "scheme": scheme_name,
                    "family": family,
                    "filter_mode": mode,
                    "active_days": int(series.notna().sum()),
                    "pass_rate": float(series.notna().mean()),
                    "buy_skip_rate": float(buy_skip.mean()),
                    "win_rate": float((series.dropna() > 0).mean()) if series.notna().any() else np.nan,
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                })

    out = pd.DataFrame(rows)
    agg = (
        out.groupby(["scheme", "family", "filter_mode"], as_index=False)
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

    out_path = RESULTS_DIR / "reg_pct_weak_env_variants_20260417.csv"
    agg_path = RESULTS_DIR / "reg_pct_weak_env_variants_aggregate_20260417.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    for mode in ["buy_filter_open_gap", "no_buy_filter"]:
        print(f"\n=== {mode} top 20 ===")
        print(
            agg[agg["filter_mode"] == mode]
            .head(20)[["scheme", "family", "avg_cum_ret", "worst_cum_ret", "avg_mean_ret", "avg_mdd", "worst_mdd", "avg_pass_rate"]]
            .to_string(index=False)
        )
    print(f"\nSaved: {out_path}")
    print(f"Saved: {agg_path}")


if __name__ == "__main__":
    run()
