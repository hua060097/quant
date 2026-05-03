#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am
import analyze_raw_pct_hybrid_and_failures as rawmod


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"

FEATURE_SPECS = {
    "cpv_factor": {"bad_side": "high", "raw_qs": [0.70, 0.75, 0.80, 0.85], "pct_vals": [0.80, 0.85, 0.90, 0.95]},
    "ppo_strength": {"bad_side": "high", "raw_qs": [0.70, 0.75, 0.80, 0.85], "pct_vals": [0.80, 0.85, 0.90, 0.95]},
    "return_10d_sector_mean": {"bad_side": "low", "raw_qs": [0.20, 0.25, 0.30, 0.40], "pct_vals": [0.20, 0.15, 0.10, 0.05]},
    "shadow_dominance": {"bad_side": "high", "raw_qs": [0.70, 0.75, 0.80, 0.85], "pct_vals": [0.80, 0.85, 0.90, 0.95]},
    "high_to_close_drawdown": {"bad_side": "low", "raw_qs": [0.15, 0.20, 0.25, 0.30], "pct_vals": [0.20, 0.15, 0.10, 0.05]},
}

BASE_SCHEMES = [
    "current_switch_mkt_exp_q30_reg10_raw_0p40",
    "switch_reg_pct_mktstate_hi95_lo85",
    "switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40",
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


def dd_breach_count(series: pd.Series, threshold: float = -0.15) -> int:
    traded = pd.to_numeric(series, errors="coerce").dropna()
    if traded.empty:
        return 0
    eq = (1 + traded).cumprod()
    dd = eq / eq.cummax() - 1.0
    flag = dd <= threshold
    return int((flag & (~flag.shift(1, fill_value=False))).sum())


def build_pct_scheme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pct_thr = rawmod.build_pct_state_threshold(df, 0.95, 0.85)
    top1_pct = pd.to_numeric(df["top1__severe_abnormal_pressure_10d__pct"], errors="coerce")
    choose_top1 = top1_pct < pct_thr
    rank = pd.Series(np.where(choose_top1, 1, 2), index=df.index)
    ret = pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index)
    return rank, pd.to_numeric(ret, errors="coerce")


def build_raw_scheme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    raw_thr = rawmod.build_raw_state_threshold(df)
    top1_raw = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
    choose_top1 = top1_raw <= raw_thr
    rank = pd.Series(np.where(choose_top1, 1, 2), index=df.index)
    ret = pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index)
    return rank, pd.to_numeric(ret, errors="coerce")


def build_base_scheme_map(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    return {
        "current_switch_mkt_exp_q30_reg10_raw_0p40": am.current_switch(df),
        "switch_reg_pct_mktstate_hi95_lo85": build_pct_scheme(df),
        "switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40": build_raw_scheme(df),
    }


def attach_selected_features(df: pd.DataFrame, rank: pd.Series) -> pd.DataFrame:
    selected_code = np.where(pd.to_numeric(rank, errors="coerce") == 1, df["top1_code"], df["top2_code"])
    out = df.copy()
    out["selected_code"] = selected_code

    factor_cols = list(FEATURE_SPECS.keys())
    raw_df = pd.read_parquet(FACTOR_PATH, columns=["date", "stock_code"] + factor_cols).rename(columns={"stock_code": "selected_code"})
    out = out.merge(raw_df, on=["date", "selected_code"], how="left")

    pct_df = pd.read_parquet(FACTOR_PATH, columns=["date", "stock_code"] + factor_cols).copy()
    for col in factor_cols:
        pct_df[f"{col}__pct"] = pct_df.groupby("date")[col].rank(pct=True, method="average")
    pct_df = pct_df[["date", "stock_code"] + [f"{c}__pct" for c in factor_cols]].rename(columns={"stock_code": "selected_code"})
    out = out.merge(pct_df, on=["date", "selected_code"], how="left")
    return out


def cond_raw(series: pd.Series, side: str, q: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    thr = am.expanding_threshold(s, q)
    if side == "high":
        return (s >= thr) & thr.notna()
    return (s <= thr) & thr.notna()


def cond_pct(series: pd.Series, side: str, value: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if side == "high":
        return s >= value
    return s <= value


def build_veto_conditions(df: pd.DataFrame) -> dict[str, pd.Series]:
    conds: dict[str, pd.Series] = {}
    for feature, spec in FEATURE_SPECS.items():
        side = spec["bad_side"]
        for q in spec["raw_qs"]:
            conds[f"raw__{feature}__{'ge' if side=='high' else 'le'}__q{int(q*100)}"] = cond_raw(df[feature], side, q)
        pct_col = f"{feature}__pct"
        for v in spec["pct_vals"]:
            val_tag = int(v * 100)
            conds[f"pct__{feature}__{'ge' if side=='high' else 'le'}__{val_tag}"] = cond_pct(df[pct_col], side, v)
    return conds


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

        scheme_map = build_base_scheme_map(df)
        for scheme_name in BASE_SCHEMES:
            rank, ret = scheme_map[scheme_name]
            feat_df = attach_selected_features(df, rank)
            conds = build_veto_conditions(feat_df)

            # baseline
            filt_ret, buy_skip = am.apply_buy_filter(df, rank, ret, gap_map)
            feat_df["filtered_ret"] = filt_ret
            feat_df["buy_skip"] = buy_skip
            segs = split_segments(feat_df)
            full_dd_count = dd_breach_count(filt_ret)
            for seg_name, seg_df in segs.items():
                mean_ret, cum_ret, mdd, win_rate = summarize(seg_df["filtered_ret"])
                rows.append({
                    "json_name": json_path.name,
                    "label_version": label_version,
                    "scheme": scheme_name,
                    "veto": "none",
                    "veto_type": "baseline",
                    "segment": seg_name,
                    "active_days": int(seg_df["filtered_ret"].notna().sum()),
                    "trade_rate": float(seg_df["filtered_ret"].notna().mean()),
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                    "win_rate": win_rate,
                    "full_dd_le_15_count": full_dd_count,
                })

            # singles
            for veto_name, cond in conds.items():
                veto_ret = pd.to_numeric(ret, errors="coerce").mask(cond)
                filt_ret, buy_skip = am.apply_buy_filter(df, rank, veto_ret, gap_map)
                feat_df["filtered_ret"] = filt_ret
                feat_df["buy_skip"] = buy_skip
                segs = split_segments(feat_df)
                full_dd_count = dd_breach_count(filt_ret)
                for seg_name, seg_df in segs.items():
                    mean_ret, cum_ret, mdd, win_rate = summarize(seg_df["filtered_ret"])
                    rows.append({
                        "json_name": json_path.name,
                        "label_version": label_version,
                        "scheme": scheme_name,
                        "veto": veto_name,
                        "veto_type": "single",
                        "segment": seg_name,
                        "active_days": int(seg_df["filtered_ret"].notna().sum()),
                        "trade_rate": float(seg_df["filtered_ret"].notna().mean()),
                        "mean_ret": mean_ret,
                        "cum_ret": cum_ret,
                        "max_drawdown": mdd,
                        "win_rate": win_rate,
                        "full_dd_le_15_count": full_dd_count,
                    })

            # pairs within same representation only
            keys = sorted(conds.keys())
            for a, b in combinations(keys, 2):
                if a.split("__", 1)[0] != b.split("__", 1)[0]:
                    continue
                cond = conds[a] & conds[b]
                veto_rate = float(cond.mean())
                if veto_rate < 0.02 or veto_rate > 0.35:
                    continue
                veto_ret = pd.to_numeric(ret, errors="coerce").mask(cond)
                filt_ret, buy_skip = am.apply_buy_filter(df, rank, veto_ret, gap_map)
                feat_df["filtered_ret"] = filt_ret
                feat_df["buy_skip"] = buy_skip
                segs = split_segments(feat_df)
                full_dd_count = dd_breach_count(filt_ret)
                pair_name = f"{a}__and__{b}"
                for seg_name, seg_df in segs.items():
                    mean_ret, cum_ret, mdd, win_rate = summarize(seg_df["filtered_ret"])
                    rows.append({
                        "json_name": json_path.name,
                        "label_version": label_version,
                        "scheme": scheme_name,
                        "veto": pair_name,
                        "veto_type": "pair",
                        "segment": seg_name,
                        "active_days": int(seg_df["filtered_ret"].notna().sum()),
                        "trade_rate": float(seg_df["filtered_ret"].notna().mean()),
                        "mean_ret": mean_ret,
                        "cum_ret": cum_ret,
                        "max_drawdown": mdd,
                        "win_rate": win_rate,
                        "full_dd_le_15_count": full_dd_count,
                    })

    detailed = pd.DataFrame(rows)
    agg = (
        detailed.groupby(["scheme", "veto", "veto_type", "segment"], as_index=False)
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
    )

    # score per scheme to rank vetoes: prefer seg5/recent/second_half stability over full-sample greed
    score_parts = []
    for scheme, sub in agg.groupby("scheme"):
        pivot = sub.pivot_table(index=["veto", "veto_type"], columns="segment", values=["avg_cum_ret", "avg_mdd"])
        pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
        pivot = pivot.reset_index()
        for col in [
            "avg_cum_ret__full", "avg_cum_ret__second_half", "avg_cum_ret__seg5_last", "avg_cum_ret__recent_13d",
            "avg_mdd__full", "avg_mdd__second_half", "avg_mdd__seg5_last", "avg_mdd__recent_13d",
        ]:
            if col not in pivot.columns:
                pivot[col] = np.nan
        pivot["score"] = (
            pivot["avg_cum_ret__second_half"].fillna(0) * 3.0 +
            pivot["avg_cum_ret__seg5_last"].fillna(0) * 4.0 +
            pivot["avg_cum_ret__recent_13d"].fillna(0) * 3.0 +
            pivot["avg_cum_ret__full"].fillna(0) * 0.5 +
            (pivot["avg_mdd__full"].fillna(0) - 0) * 0.5
        )
        pivot["scheme"] = scheme
        score_parts.append(pivot)
    score_df = pd.concat(score_parts, ignore_index=True)
    score_df = score_df.sort_values(["scheme", "score"], ascending=[True, False]).reset_index(drop=True)

    detailed.to_csv(RESULTS_DIR / "selected_factor_raw_pct_veto_compare_20260419.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(RESULTS_DIR / "selected_factor_raw_pct_veto_compare_aggregate_20260419.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(RESULTS_DIR / "selected_factor_raw_pct_veto_compare_ranked_20260419.csv", index=False, encoding="utf-8-sig")

    for scheme in BASE_SCHEMES:
        print(f"\n=== Top 15 for {scheme} ===")
        print(score_df[score_df["scheme"] == scheme].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
