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

HIGH_QS = [0.70, 0.75, 0.80, 0.85, 0.90]
LOW_QS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


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
    feature_cols = HIGH_BAD_FEATURES + LOW_BAD_FEATURES
    factor_df = pd.read_parquet(
        FACTOR_PATH,
        columns=["date", "stock_code"] + feature_cols,
    ).rename(columns={"stock_code": "selected_code"})
    return loss_df.merge(factor_df, on=["date", "selected_code"], how="left")


def build_base_loss_df() -> pd.DataFrame:
    _, _, loss_df, _ = rawmod.compare_hybrids()
    loss_df = loss_df.copy()
    loss_df["selected_code"] = np.where(
        pd.to_numeric(loss_df["selected_rank"], errors="coerce") == 1,
        loss_df["top1_code"],
        loss_df["top2_code"],
    )
    loss_df = attach_selected_features(loss_df)
    return loss_df


def threshold_series(series: pd.Series, q: float) -> pd.Series:
    return am.expanding_threshold(pd.to_numeric(series, errors="coerce"), q)


def build_conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    conds: dict[str, pd.Series] = {}
    for col in HIGH_BAD_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        for q in HIGH_QS:
            thr = threshold_series(s, q)
            conds[f"{col}_ge_q{int(q*100)}"] = (s >= thr) & thr.notna()
    for col in LOW_BAD_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        for q in LOW_QS:
            thr = threshold_series(s, q)
            conds[f"{col}_le_q{int(q*100)}"] = (s <= thr) & thr.notna()
    return conds


def evaluate_rules(loss_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    agg_rows: list[dict] = []

    for (json_name, label_version), sub in loss_df.groupby(["json_name", "label_version"], sort=True):
        base = sub[~sub["buy_skip"]].copy().sort_values("date").reset_index(drop=True)
        conds = build_conditions(base)

        base_series = base["filtered_ret"]
        mean_ret, cum_ret, mdd, win_rate = summarize(base_series)
        rows.append({
            "json_name": json_name,
            "label_version": label_version,
            "rule": "base_raw_state",
            "type": "baseline",
            "active_days": int(base_series.notna().sum()),
            "trade_rate": float(base_series.notna().mean()),
            "mean_ret": mean_ret,
            "cum_ret": cum_ret,
            "max_drawdown": mdd,
            "win_rate": win_rate,
        })

        # single veto
        for name, cond in conds.items():
            series = base_series.mask(cond)
            mean_ret, cum_ret, mdd, win_rate = summarize(series)
            rows.append({
                "json_name": json_name,
                "label_version": label_version,
                "rule": f"veto_{name}",
                "type": "single",
                "active_days": int(series.notna().sum()),
                "trade_rate": float(series.notna().mean()),
                "veto_rate": float(cond.mean()),
                "mean_ret": mean_ret,
                "cum_ret": cum_ret,
                "max_drawdown": mdd,
                "win_rate": win_rate,
            })

        # pair veto with AND only to keep it practical
        keys = sorted(conds.keys())
        for a, b in combinations(keys, 2):
            cond = conds[a] & conds[b]
            if cond.mean() < 0.02 or cond.mean() > 0.35:
                continue
            series = base_series.mask(cond)
            mean_ret, cum_ret, mdd, win_rate = summarize(series)
            rows.append({
                "json_name": json_name,
                "label_version": label_version,
                "rule": f"veto_{a}__and__{b}",
                "type": "pair",
                "active_days": int(series.notna().sum()),
                "trade_rate": float(series.notna().mean()),
                "veto_rate": float(cond.mean()),
                "mean_ret": mean_ret,
                "cum_ret": cum_ret,
                "max_drawdown": mdd,
                "win_rate": win_rate,
            })

    detailed = pd.DataFrame(rows)
    agg = (
        detailed[detailed["rule"] != "base_raw_state"]
        .groupby(["rule", "type"], as_index=False)
        .agg(
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_mdd=("max_drawdown", "mean"),
            worst_mdd=("max_drawdown", "min"),
            avg_win_rate=("win_rate", "mean"),
            avg_trade_rate=("trade_rate", "mean"),
            avg_veto_rate=("veto_rate", "mean"),
        )
    )
    # Improvement score: preserve return while rewarding mdd improvement.
    base_agg = detailed[detailed["rule"] == "base_raw_state"].groupby(lambda _: 0).agg(
        base_avg_cum=("cum_ret", "mean"),
        base_avg_mdd=("max_drawdown", "mean"),
    ).iloc[0]
    agg["cum_ret_drop"] = base_agg["base_avg_cum"] - agg["avg_cum_ret"]
    agg["mdd_improve"] = agg["avg_mdd"] - base_agg["base_avg_mdd"]
    agg["score"] = agg["mdd_improve"] * 2.0 - agg["cum_ret_drop"] * 0.3
    agg = agg.sort_values(["score", "avg_cum_ret"], ascending=[False, False]).reset_index(drop=True)
    return detailed, agg


def main() -> None:
    loss_df = build_base_loss_df()
    detailed, agg = evaluate_rules(loss_df)

    detailed.to_csv(RESULTS_DIR / "raw_state_selected_feature_veto_search_20260418.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(RESULTS_DIR / "raw_state_selected_feature_veto_search_aggregate_20260418.csv", index=False, encoding="utf-8-sig")

    print("=== Top 20 Rules ===")
    print(agg.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
