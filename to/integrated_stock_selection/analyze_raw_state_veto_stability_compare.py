#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am
import analyze_raw_pct_hybrid_and_failures as rawmod


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"

HIGH_BAD_FEATURES = [
    "volume_structure_alpha",
    "money_flow_pressure",
    "ppo_strength",
    "abnormal_turnover",
    "cpv_factor",
    "shadow_dominance",
]

LOW_BAD_FEATURES = [
    "open_to_high_ratio",
    "amihud_illiquidity",
    "return_10d_sector_mean",
]


TARGET_RULES = {
    "base_raw_state": None,
    "veto_cpv_factor_ge_q70__and__ppo_strength_ge_q75": (
        ("cpv_factor", "high", 0.70),
        ("ppo_strength", "high", 0.75),
    ),
    "veto_amihud_illiquidity_le_q40__and__volume_structure_alpha_ge_q80": (
        ("amihud_illiquidity", "low", 0.40),
        ("volume_structure_alpha", "high", 0.80),
    ),
    "veto_return_10d_sector_mean_le_q40__and__shadow_dominance_ge_q85": (
        ("return_10d_sector_mean", "low", 0.40),
        ("shadow_dominance", "high", 0.85),
    ),
}


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


def attach_selected_features(loss_df: pd.DataFrame) -> pd.DataFrame:
    cols = list({c for specs in TARGET_RULES.values() if specs for (c, _, _) in specs})
    factor_df = pd.read_parquet(
        FACTOR_PATH,
        columns=["date", "stock_code"] + cols,
    ).rename(columns={"stock_code": "selected_code"})
    return loss_df.merge(factor_df, on=["date", "selected_code"], how="left")


def build_base_df() -> pd.DataFrame:
    _, _, loss_df, _ = rawmod.compare_hybrids()
    loss_df = loss_df.copy()
    loss_df["selected_code"] = np.where(
        pd.to_numeric(loss_df["selected_rank"], errors="coerce") == 1,
        loss_df["top1_code"],
        loss_df["top2_code"],
    )
    loss_df = attach_selected_features(loss_df)
    return loss_df.copy()


def expanding_condition(series: pd.Series, side: str, q: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    thr = am.expanding_threshold(s, q)
    if side == "high":
        return (s >= thr) & thr.notna()
    return (s <= thr) & thr.notna()


def apply_rule(df: pd.DataFrame, rule_name: str) -> pd.Series:
    if rule_name == "base_raw_state":
        return pd.to_numeric(df["filtered_ret"], errors="coerce")
    specs = TARGET_RULES[rule_name]
    cond = pd.Series(True, index=df.index)
    for col, side, q in specs:
        cond &= expanding_condition(df[col], side, q)
    return pd.to_numeric(df["filtered_ret"], errors="coerce").mask(cond)


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
    base = build_base_df()
    rows: list[dict] = []

    for (json_name, label_version), sub in base.groupby(["json_name", "label_version"], sort=True):
        segments = split_segments(sub)
        for rule_name in TARGET_RULES:
            series_all = apply_rule(sub, rule_name)
            overall_mean, overall_cum, overall_mdd, overall_win = summarize(series_all)
            dd_flag = (1 + series_all.dropna()).cumprod()
            if dd_flag.empty:
                dd_breach_count = 0
            else:
                drawdown = dd_flag / dd_flag.cummax() - 1
                in_breach = drawdown <= -0.15
                dd_breach_count = int(((in_breach) & (~in_breach.shift(1, fill_value=False))).sum())

            for seg_name, seg_df in segments.items():
                seg_series = apply_rule(seg_df, rule_name)
                mean_ret, cum_ret, mdd, win_rate = summarize(seg_series)
                rows.append({
                    "json_name": json_name,
                    "label_version": label_version,
                    "rule": rule_name,
                    "segment": seg_name,
                    "active_days": int(seg_series.notna().sum()),
                    "trade_rate": float(seg_series.notna().mean()),
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                    "win_rate": win_rate,
                    "full_dd_le_15_count": dd_breach_count,
                    "full_cum_ret": overall_cum,
                    "full_max_drawdown": overall_mdd,
                })

    detailed = pd.DataFrame(rows)
    agg = (
        detailed.groupby(["rule", "segment"], as_index=False)
        .agg(
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_mdd=("max_drawdown", "mean"),
            worst_mdd=("max_drawdown", "min"),
            avg_win_rate=("win_rate", "mean"),
            avg_trade_rate=("trade_rate", "mean"),
            avg_dd_le_15_count=("full_dd_le_15_count", "mean"),
            avg_full_cum_ret=("full_cum_ret", "mean"),
            avg_full_mdd=("full_max_drawdown", "mean"),
        )
        .sort_values(["segment", "avg_cum_ret"], ascending=[True, False])
        .reset_index(drop=True)
    )

    detailed.to_csv(RESULTS_DIR / "raw_state_veto_stability_compare_20260418.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(RESULTS_DIR / "raw_state_veto_stability_compare_aggregate_20260418.csv", index=False, encoding="utf-8-sig")

    print("=== Aggregate Stability Compare ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
