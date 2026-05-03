#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am
import analyze_raw_pct_hybrid_and_failures as rawmod


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"

FEATURE_COLS = [
    "cpv_factor",
    "ppo_strength",
    "amihud_illiquidity",
    "volume_structure_alpha",
    "return_10d_sector_mean",
    "shadow_dominance",
]


def summarize(series: pd.Series) -> tuple[float, float, float, float]:
    traded = pd.to_numeric(series, errors="coerce").dropna()
    if traded.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(traded.mean()),
        float((1 + traded).prod() - 1),
        am.calc_mdd(traded),
        float((traded > 0).mean()),
    )


def build_raw_scheme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    raw_thr = rawmod.build_raw_state_threshold(df)
    top1_raw = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
    choose_top1 = top1_raw <= raw_thr
    rank = pd.Series(np.where(choose_top1, 1, 2), index=df.index)
    ret = pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index)
    return rank, pd.to_numeric(ret, errors="coerce")


def build_pct_scheme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pct_thr = rawmod.build_pct_state_threshold(df, 0.95, 0.85)
    top1_pct = pd.to_numeric(df["top1__severe_abnormal_pressure_10d__pct"], errors="coerce")
    choose_top1 = top1_pct < pct_thr
    rank = pd.Series(np.where(choose_top1, 1, 2), index=df.index)
    ret = pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index)
    return rank, pd.to_numeric(ret, errors="coerce")


def attach_selected_features(df: pd.DataFrame, rank: pd.Series) -> pd.DataFrame:
    selected_code = np.where(pd.to_numeric(rank, errors="coerce") == 1, df["top1_code"], df["top2_code"])
    out = df.copy()
    out["selected_code"] = selected_code
    factor_df = pd.read_parquet(FACTOR_PATH, columns=["date", "stock_code"] + FEATURE_COLS).rename(columns={"stock_code": "selected_code"})
    out = out.merge(factor_df, on=["date", "selected_code"], how="left")
    return out


def expanding_condition(series: pd.Series, side: str, q: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    thr = am.expanding_threshold(s, q)
    if side == "high":
        return (s >= thr) & thr.notna()
    return (s <= thr) & thr.notna()


def apply_veto(df: pd.DataFrame, base_ret: pd.Series, veto_name: str) -> pd.Series:
    if veto_name == "none":
        return base_ret
    if veto_name == "cpv_ppo":
        cond = expanding_condition(df["cpv_factor"], "high", 0.70) & expanding_condition(df["ppo_strength"], "high", 0.75)
    elif veto_name == "return10_shadow":
        cond = expanding_condition(df["return_10d_sector_mean"], "low", 0.40) & expanding_condition(df["shadow_dominance"], "high", 0.85)
    else:
        raise ValueError(veto_name)
    return pd.to_numeric(base_ret, errors="coerce").mask(cond)


def dd_breach_count(series: pd.Series, threshold: float = -0.15) -> int:
    traded = pd.to_numeric(series, errors="coerce").dropna()
    if traded.empty:
        return 0
    eq = (1 + traded).cumprod()
    dd = eq / eq.cummax() - 1.0
    flag = dd <= threshold
    return int((flag & (~flag.shift(1, fill_value=False))).sum())


def split_segments(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.sort_values("date").reset_index(drop=True)
    n = len(work)
    half = n // 2
    fifth = n // 5 if n >= 5 else n
    recent_n = min(13, n)
    return {
        "full": work,
        "first_half": work.iloc[:half],
        "second_half": work.iloc[half:],
        "seg5_last": work.iloc[-fifth:] if fifth > 0 else work.iloc[0:0],
        f"recent_{recent_n}d": work.iloc[-recent_n:],
    }


def main() -> None:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    rows: list[dict] = []

    for json_path in am.JSON_PATHS:
        df = am.build_base_df(json_path, factor_df)
        df = rawmod.attach_extra(df, factor_df)
        gap_map = am.build_buy_gap_map(df)
        label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"

        scheme_defs = {
            "current_switch_mkt_exp_q30_reg10_raw_0p40": am.current_switch(df),
            "switch_reg_pct_mktstate_hi95_lo85": build_pct_scheme(df),
            "switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40": build_raw_scheme(df),
        }

        combo_defs = [
            ("current_switch_mkt_exp_q30_reg10_raw_0p40", "none"),
            ("current_switch_mkt_exp_q30_reg10_raw_0p40", "cpv_ppo"),
            ("current_switch_mkt_exp_q30_reg10_raw_0p40", "return10_shadow"),
            ("switch_reg_pct_mktstate_hi95_lo85", "none"),
            ("switch_reg_pct_mktstate_hi95_lo85", "cpv_ppo"),
            ("switch_reg_pct_mktstate_hi95_lo85", "return10_shadow"),
            ("switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40", "none"),
            ("switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40", "cpv_ppo"),
            ("switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40", "return10_shadow"),
        ]

        prepared: dict[tuple[str, str], pd.DataFrame] = {}
        filtered_map: dict[tuple[str, str], pd.Series] = {}

        for scheme_name, veto_name in combo_defs:
            rank, ret = scheme_defs[scheme_name]
            feat_df = attach_selected_features(df, rank)
            veto_ret = apply_veto(feat_df, ret, veto_name)
            filt_ret, buy_skip = am.apply_buy_filter(df, rank, veto_ret, gap_map)
            feat_df["filtered_ret"] = filt_ret
            feat_df["buy_skip"] = buy_skip
            prepared[(scheme_name, veto_name)] = feat_df
            filtered_map[(scheme_name, veto_name)] = filt_ret

        for (scheme_name, veto_name), feat_df in prepared.items():
            segs = split_segments(feat_df)
            full_series = filtered_map[(scheme_name, veto_name)]
            full_dd_count = dd_breach_count(full_series)
            for seg_name, seg_df in segs.items():
                series = seg_df["filtered_ret"]
                mean_ret, cum_ret, mdd, win_rate = summarize(series)
                rows.append({
                    "json_name": json_path.name,
                    "label_version": label_version,
                    "scheme": scheme_name,
                    "veto": veto_name,
                    "combo_name": f"{scheme_name}__{veto_name}",
                    "segment": seg_name,
                    "active_days": int(series.notna().sum()),
                    "trade_rate": float(series.notna().mean()),
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                    "win_rate": win_rate,
                    "full_dd_le_15_count": full_dd_count,
                })

    detailed = pd.DataFrame(rows)
    agg = (
        detailed.groupby(["combo_name", "scheme", "veto", "segment"], as_index=False)
        .agg(
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_mdd=("max_drawdown", "mean"),
            worst_mdd=("max_drawdown", "min"),
            avg_win_rate=("win_rate", "mean"),
            avg_trade_rate=("trade_rate", "mean"),
            avg_dd_le_15_count=("full_dd_le_15_count", "mean"),
        )
        .sort_values(["segment", "avg_cum_ret"], ascending=[True, False])
        .reset_index(drop=True)
    )

    detailed.to_csv(RESULTS_DIR / "scheme_veto_cross_compare_20260418.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(RESULTS_DIR / "scheme_veto_cross_compare_aggregate_20260418.csv", index=False, encoding="utf-8-sig")

    print("=== Aggregate Cross Compare ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
