from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am
import analyze_reg_pct_weak_env_variants as arv


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

EXTRA_MARKET_COLS = [
    "market_trend_adaptive_beta",
    "ma5_strength_breadth_v3",
    "market_volatility_context",
    "strong_stock_volume_confirmation_v3",
    "sector_hotspot_concentration_v3",
    "mkt_breadth_thrust_v2",
    "mkt_liquidity_impulse_v2",
    "market_60d_new_high_ratio",
    "top10_10d_strength_level_v3",
    "top10_10d_recent3_strength_v3",
    "limit_up_sw2_concentration",
]

TOP_FACTOR_COLS = [
    "severe_abnormal_pressure_10d",
    "abnormal_turnover",
    "ppo_strength",
    "volume_structure_alpha",
    "sector_turnover_change_5d_20d",
    "return_10d_sector_mean",
    "sector_leader_capture_rank",
]


def summarize(series: pd.Series) -> tuple[float, float, float, float]:
    traded = series.dropna()
    if traded.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(traded.mean()),
        float((1 + traded).prod() - 1),
        am.calc_mdd(traded),
        float((traded > 0).mean()),
    )


def attach_extra(df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in EXTRA_MARKET_COLS if c not in df.columns]
    if missing_cols:
        extra = pd.read_parquet(
            BASE_DIR / "factor_lab_data" / "factors.parquet",
            columns=["date", "stock_code"] + missing_cols,
        ).copy()
        extra["date"] = pd.to_datetime(extra["date"])
        extra_map = extra.groupby("date")[missing_cols].first().reset_index()
        df = df.merge(extra_map, on="date", how="left")
    return df


def build_raw_state_threshold(df: pd.DataFrame) -> pd.Series:
    masks = arv.build_market_state_masks(df)
    weak = masks["weak_trend_dispersion"]
    return pd.Series(np.where(weak, 0.25, 0.40), index=df.index)


def build_pct_state_threshold(df: pd.DataFrame, hi: float, lo: float) -> pd.Series:
    mkt = pd.to_numeric(df["mkt_ma5_followthrough_v3"], errors="coerce")
    thr = am.expanding_threshold(mkt, 0.30)
    strong = (mkt >= thr) & thr.notna()
    return pd.Series(np.where(strong, hi, lo), index=df.index)


def build_hybrid_schemes(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    top1_raw = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
    top1_pct = pd.to_numeric(df["top1__severe_abnormal_pressure_10d__pct"], errors="coerce")
    top1_ret = pd.to_numeric(df["top1_ret"], errors="coerce")
    top2_ret = pd.to_numeric(df["top2_ret"], errors="coerce")

    raw_thr = build_raw_state_threshold(df)
    pct_state_95_85 = build_pct_state_threshold(df, 0.95, 0.85)
    pct_state_95_90 = build_pct_state_threshold(df, 0.95, 0.90)
    pct_state_97_90 = build_pct_state_threshold(df, 0.97, 0.90)

    schemes: dict[str, tuple[pd.Series, pd.Series]] = {}

    # Baselines
    choose_top1 = top1_raw <= raw_thr
    schemes["switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40"] = (
        pd.Series(np.where(choose_top1, 1, 2), index=df.index),
        pd.Series(np.where(choose_top1, top1_ret, top2_ret), index=df.index),
    )

    pct_choose = top1_pct < pct_state_95_85
    schemes["switch_reg_pct_mktstate_hi95_lo85"] = (
        pd.Series(np.where(pct_choose, 1, 2), index=df.index),
        pd.Series(np.where(pct_choose, top1_ret, top2_ret), index=df.index),
    )

    # Hybrid veto versions: raw is primary, pct vetoes overheated top1.
    hybrid_defs = {
        "hybrid_rawstate_pct95_veto": pd.Series(0.95, index=df.index),
        "hybrid_rawstate_pct90_veto": pd.Series(0.90, index=df.index),
        "hybrid_rawstate_pctstate_hi95_lo85_veto": pct_state_95_85,
        "hybrid_rawstate_pctstate_hi95_lo90_veto": pct_state_95_90,
        "hybrid_rawstate_pctstate_hi97_lo90_veto": pct_state_97_90,
    }

    for name, pct_thr in hybrid_defs.items():
        choose = (top1_raw <= raw_thr) & (top1_pct < pct_thr)
        schemes[name] = (
            pd.Series(np.where(choose, 1, 2), index=df.index),
            pd.Series(np.where(choose, top1_ret, top2_ret), index=df.index),
        )

    return schemes


def compare_hybrids() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    rows: list[dict] = []
    stability_rows: list[dict] = []
    loss_rows: list[pd.DataFrame] = []

    for json_path in am.JSON_PATHS:
        df = am.build_base_df(json_path, factor_df)
        df = attach_extra(df, factor_df)
        gap_map = am.build_buy_gap_map(df)
        label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"

        schemes = build_hybrid_schemes(df)

        for scheme_name, (rank, ret) in schemes.items():
            filt_ret, buy_skip = am.apply_buy_filter(df, rank, ret, gap_map)
            for mode, series in [("no_buy_filter", ret), ("buy_filter_open_gap", filt_ret)]:
                mean_ret, cum_ret, mdd, win_rate = summarize(series)
                rows.append({
                    "json_name": json_path.name,
                    "label_version": label_version,
                    "scheme": scheme_name,
                    "filter_mode": mode,
                    "active_days": int(series.notna().sum()),
                    "pass_rate": float(series.notna().mean()),
                    "buy_skip_rate": float(buy_skip.mean()) if mode == "buy_filter_open_gap" else 0.0,
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                    "win_rate": win_rate,
                })

            if scheme_name == "switch_reg_raw_weak_trend_dispersion_lo0p25_hi0p40":
                work = df.copy()
                work["selected_rank"] = rank
                work["selected_ret"] = ret
                work["filtered_ret"] = filt_ret
                work["buy_skip"] = buy_skip
                work["label_version"] = label_version
                work["json_name"] = json_path.name
                loss_rows.append(work)

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
                    mean_ret, cum_ret, mdd, win_rate = summarize(part["filtered_ret"])
                    stability_rows.append({
                        "json_name": json_path.name,
                        "label_version": label_version,
                        "scheme": scheme_name,
                        "segment": seg,
                        "active_days": int(part["filtered_ret"].notna().sum()),
                        "mean_ret": mean_ret,
                        "cum_ret": cum_ret,
                        "max_drawdown": mdd,
                        "win_rate": win_rate,
                    })

    detailed = pd.DataFrame(rows)
    agg = (
        detailed.groupby(["scheme", "filter_mode"], as_index=False)
        .agg(
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_mdd=("max_drawdown", "mean"),
            worst_mdd=("max_drawdown", "min"),
            avg_win_rate=("win_rate", "mean"),
            avg_pass_rate=("pass_rate", "mean"),
            avg_buy_skip_rate=("buy_skip_rate", "mean"),
        )
        .sort_values(["filter_mode", "avg_cum_ret"], ascending=[True, False])
        .reset_index(drop=True)
    )
    loss_df = pd.concat(loss_rows, ignore_index=True)
    stability = pd.DataFrame(stability_rows)
    return detailed, agg, loss_df, stability


def analyze_raw_failures(loss_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = loss_df.copy()
    work = work[~work["buy_skip"]].copy()

    # Large losses after buy-filter.
    work["bad3"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.03
    work["bad5"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.05

    feature_cols = [
        "top1__severe_abnormal_pressure_10d",
        "top1__severe_abnormal_pressure_10d__pct",
        "top1__abnormal_turnover",
        "top1__abnormal_turnover__pct",
        "top1__ppo_strength",
        "top1__ppo_strength__pct",
        "top1__volume_structure_alpha",
        "top1__volume_structure_alpha__pct",
        "top1__sector_turnover_change_5d_20d",
        "top1__sector_turnover_change_5d_20d__pct",
        "top1__return_10d_sector_mean",
        "top1__return_10d_sector_mean__pct",
        "mkt_ma5_followthrough_v3",
        "market_trend_strength",
        "mkt_regime_trend_pressure_v2",
        "mkt_sector_dispersion",
        "mkt_dispersion_shock_v2",
        "market_trend_adaptive_beta",
        "ma5_strength_breadth_v3",
        "market_volatility_context",
        "strong_stock_volume_confirmation_v3",
        "sector_hotspot_concentration_v3",
        "mkt_breadth_thrust_v2",
        "mkt_liquidity_impulse_v2",
        "market_60d_new_high_ratio",
        "top10_10d_strength_level_v3",
        "top10_10d_recent3_strength_v3",
        "limit_up_sw2_concentration",
    ]

    rows: list[dict] = []
    for bad_col in ["bad3", "bad5"]:
        bad_mask = work[bad_col].fillna(False)
        good_mask = (~bad_mask) & work["filtered_ret"].notna()
        if bad_mask.sum() < 5 or good_mask.sum() < 5:
            continue
        for col in feature_cols:
            s = pd.to_numeric(work[col], errors="coerce")
            bad = s[bad_mask].dropna()
            good = s[good_mask].dropna()
            if len(bad) < 5 or len(good) < 5:
                continue
            rows.append({
                "bad_def": bad_col,
                "factor": col,
                "bad_mean": float(bad.mean()),
                "good_mean": float(good.mean()),
                "mean_diff": float(bad.mean() - good.mean()),
                "effect_z": float((bad.mean() - good.mean()) / (s.dropna().std(ddof=0) + 1e-9)),
                "spearman_bad": float(pd.Series(s).corr(work[bad_col].astype(float), method="spearman")),
            })

    rel = pd.DataFrame(rows)
    rel["abs_effect"] = rel["effect_z"].abs()
    rel = rel.sort_values(["bad_def", "abs_effect"], ascending=[True, False]).reset_index(drop=True)

    # Search simple avoid rules on top factors only.
    simple_rules: list[dict] = []
    top_candidates = [
        ("top1__severe_abnormal_pressure_10d__pct", ">=", [0.90, 0.95]),
        ("top1__severe_abnormal_pressure_10d", ">=", [0.30, 0.35, 0.40]),
        ("top1__abnormal_turnover__pct", ">=", [0.80, 0.85, 0.90]),
        ("top1__ppo_strength__pct", ">=", [0.80, 0.85, 0.90]),
        ("market_trend_strength", "<=exp", [0.40, 0.50, 0.60]),
        ("mkt_regime_trend_pressure_v2", "<=exp", [0.40, 0.50, 0.60]),
        ("mkt_sector_dispersion", ">=exp", [0.60, 0.70]),
        ("strong_stock_volume_confirmation_v3", "<=exp", [0.40, 0.50]),
        ("sector_hotspot_concentration_v3", ">=exp", [0.70, 0.80]),
    ]

    for bad_col in ["bad3", "bad5"]:
        bad_mask = work[bad_col].fillna(False)
        base_ret = pd.to_numeric(work["filtered_ret"], errors="coerce")
        for factor, op, thrs in top_candidates:
            s = pd.to_numeric(work[factor], errors="coerce")
            for thr in thrs:
                if op == ">=":
                    mask = s >= float(thr)
                elif op == "<=exp":
                    q = am.expanding_threshold(s, float(thr))
                    mask = (s <= q) & q.notna()
                elif op == ">=exp":
                    q = am.expanding_threshold(s, float(thr))
                    mask = (s >= q) & q.notna()
                else:
                    continue

                filtered = base_ret.mask(mask, np.nan)
                mean_ret, cum_ret, mdd, win_rate = summarize(filtered)
                simple_rules.append({
                    "bad_def": bad_col,
                    "factor": factor,
                    "rule": f"{factor}{op}{thr}",
                    "skip_rate": float(mask.mean()),
                    "capture_rate": float(mask[bad_mask].mean()) if bad_mask.sum() else np.nan,
                    "mean_ret": mean_ret,
                    "cum_ret": cum_ret,
                    "max_drawdown": mdd,
                    "win_rate": win_rate,
                })

    avoid = pd.DataFrame(simple_rules).sort_values(
        ["bad_def", "capture_rate", "cum_ret"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return rel, avoid


def run() -> None:
    detailed, agg, loss_df, stability = compare_hybrids()
    rel, avoid = analyze_raw_failures(loss_df)

    detailed_path = RESULTS_DIR / "raw_pct_hybrid_compare_20260418.csv"
    agg_path = RESULTS_DIR / "raw_pct_hybrid_compare_aggregate_20260418.csv"
    stability_path = RESULTS_DIR / "raw_scheme_failure_stability_20260418.csv"
    rel_path = RESULTS_DIR / "raw_scheme_bad_sample_factor_relation_20260418.csv"
    avoid_path = RESULTS_DIR / "raw_scheme_bad_sample_avoid_rule_scan_20260418.csv"

    detailed.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")
    stability.to_csv(stability_path, index=False, encoding="utf-8-sig")
    rel.to_csv(rel_path, index=False, encoding="utf-8-sig")
    avoid.to_csv(avoid_path, index=False, encoding="utf-8-sig")

    for mode in ["buy_filter_open_gap", "no_buy_filter"]:
        print(f"\n=== Hybrid compare: {mode} ===")
        print(
            agg[agg["filter_mode"] == mode][
                ["scheme", "avg_cum_ret", "worst_cum_ret", "avg_mean_ret", "avg_mdd", "avg_pass_rate", "avg_buy_skip_rate"]
            ].head(12).to_string(index=False)
        )

    for bad_def in ["bad3", "bad5"]:
        print(f"\n=== Bad sample relation: {bad_def} ===")
        print(rel[rel["bad_def"] == bad_def].head(12)[["factor", "mean_diff", "effect_z", "spearman_bad"]].to_string(index=False))
        print(f"\n=== Avoid rules: {bad_def} ===")
        print(avoid[avoid["bad_def"] == bad_def].head(12)[["rule", "skip_rate", "capture_rate", "cum_ret", "max_drawdown"]].to_string(index=False))

    print(f"\nSaved: {detailed_path}")
    print(f"Saved: {agg_path}")
    print(f"Saved: {stability_path}")
    print(f"Saved: {rel_path}")
    print(f"Saved: {avoid_path}")


if __name__ == "__main__":
    run()
