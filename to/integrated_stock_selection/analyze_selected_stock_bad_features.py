from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Focus on stock-level factors with nontrivial importance from the user's model log.
SELECTED_STOCK_FACTORS = [
    "intra_sector_momentum_rank",
    "price_ma5_ratio",
    "abnormal_turnover",
    "max_rally_20d",
    "coskewness",
    "value_growth_rotation",
    "distance_from_52w_low",
    "sector_limit_overflow_alpha",
    "ma_convergence_score",
    "sector_momentum_interaction",
    "sector_momentum_consistency",
    "return_5d_sector_mean",
    "sector_relative_volatility",
    "ppo_strength",
    "kurtosis_20d",
    "sw_circ_mv_rank_intra_sector",
    "range_compression",
    "return_3d_sector_mean",
    "ma60_slope",
    "sector_leader_capture_rank",
    "relative_strength_20d",
    "volume_structure_alpha",
    "sector_relative_strength_vs_market",
    "downside_beta",
    "sector_correlation_20d",
    "sector_momentum_accel_rank",
    "max_drawdown_20d",
    "return_10d_sector_mean",
    "amihud_illiquidity",
    "sector_liquidity_gated_turnover_z",
    "shadow_dominance",
    "sector_volatility_5d",
    "open_to_high_ratio",
    "sector_rank_change_5d",
    "resistance_distance",
    "pct_chg_sector_mean",
    "return_20d_sector_mean",
    "sector_turnover_change_5d_20d",
    "sector_momentum_dispersion_rank",
    "intraday_reversal",
    "sector_vols_divergence_alpha",
    "sector_strong_leader_alpha",
    "money_flow_pressure",
    "size_rotation_strength",
    "stock_sector_pos_align",
    "beta_60d",
    "pct_chg_2d",
    "intra_sector_breadth",
]


def build_selected_samples() -> pd.DataFrame:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    need_cols = ["date", "stock_code"] + [c for c in SELECTED_STOCK_FACTORS if c in factor_df.columns]
    factor_slim = factor_df[need_cols].copy()

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

        selected_rank = pd.Series(np.where(top1_raw <= raw_thr, 1, 2), index=df.index)
        selected_code = pd.Series(np.where(top1_raw <= raw_thr, df["top1_code"], df["top2_code"]), index=df.index)
        selected_ret = pd.Series(np.where(top1_raw <= raw_thr, df["top1_ret"], df["top2_ret"]), index=df.index)
        filtered_ret, buy_skip = am.apply_buy_filter(df, selected_rank, selected_ret, gap_map)

        out = pd.DataFrame({
            "date": df["date"],
            "json_name": json_path.name,
            "label_version": label_version,
            "selected_rank": selected_rank,
            "selected_code": selected_code,
            "selected_ret": selected_ret,
            "filtered_ret": filtered_ret,
            "buy_skip": buy_skip,
        })
        out = out[(out["filtered_ret"].notna()) & (~out["buy_skip"])].copy()
        out = out.merge(
            factor_slim,
            left_on=["date", "selected_code"],
            right_on=["date", "stock_code"],
            how="left",
        )
        out = out.drop(columns=["stock_code"])
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarize_effects(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["bad3"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.03
    work["bad5"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.05
    work["good"] = pd.to_numeric(work["filtered_ret"], errors="coerce") > -0.03

    rows: list[dict] = []
    numeric_cols = [c for c in SELECTED_STOCK_FACTORS if c in work.columns]
    for bad_col in ["bad3", "bad5"]:
        bad_mask = work[bad_col].fillna(False)
        good_mask = (~bad_mask) & work["filtered_ret"].notna()
        for col in numeric_cols:
            s = pd.to_numeric(work[col], errors="coerce")
            bad = s[bad_mask].dropna()
            good = s[good_mask].dropna()
            if len(bad) < 8 or len(good) < 20:
                continue
            rows.append({
                "bad_def": bad_col,
                "factor": col,
                "bad_mean": float(bad.mean()),
                "good_mean": float(good.mean()),
                "mean_diff": float(bad.mean() - good.mean()),
                "effect_z": float((bad.mean() - good.mean()) / (s.dropna().std(ddof=0) + 1e-9)),
                "spearman_bad": float(s.corr(work[bad_col].astype(float), method="spearman")),
            })
    out = pd.DataFrame(rows)
    out["abs_effect"] = out["effect_z"].abs()
    return out.sort_values(["bad_def", "abs_effect"], ascending=[True, False]).reset_index(drop=True)


def scan_simple_rules(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["bad3"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.03
    work["bad5"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.05

    # Only simple, interpretable stock-level vetoes.
    candidates = [
        ("volume_structure_alpha", ">=", "exp", [0.70, 0.80, 0.85]),
        ("abnormal_turnover", ">=", "exp", [0.70, 0.80, 0.85]),
        ("sector_turnover_change_5d_20d", ">=", "exp", [0.70, 0.80, 0.85]),
        ("return_10d_sector_mean", "<=", "exp", [0.20, 0.30, 0.40]),
        ("price_ma5_ratio", ">=", "exp", [0.70, 0.80, 0.85]),
        ("max_rally_20d", ">=", "exp", [0.70, 0.80, 0.85]),
        ("sector_limit_overflow_alpha", ">=", "exp", [0.70, 0.80, 0.85]),
        ("sector_relative_volatility", ">=", "exp", [0.70, 0.80, 0.85]),
        ("sector_volatility_5d", ">=", "exp", [0.70, 0.80, 0.85]),
        ("open_to_high_ratio", "<=", "exp", [0.20, 0.30, 0.40]),
    ]

    rows: list[dict] = []
    base = pd.to_numeric(work["filtered_ret"], errors="coerce")
    for bad_col in ["bad3", "bad5"]:
        bad_mask = work[bad_col].fillna(False)
        for factor, side, mode, q_list in candidates:
            if factor not in work.columns:
                continue
            s = pd.to_numeric(work[factor], errors="coerce")
            for q in q_list:
                thr = am.expanding_threshold(s, q)
                if side == ">=":
                    veto = (s >= thr) & thr.notna()
                else:
                    veto = (s <= thr) & thr.notna()
                filtered = base.mask(veto, np.nan)
                traded = filtered.dropna()
                rows.append({
                    "bad_def": bad_col,
                    "rule": f"{factor}{side}exp_q{int(q*100)}",
                    "skip_rate": float(veto.mean()),
                    "capture_rate": float(veto[bad_mask].mean()) if bad_mask.sum() else np.nan,
                    "mean_ret": float(traded.mean()) if len(traded) else np.nan,
                    "cum_ret": float((1 + traded).prod() - 1) if len(traded) else np.nan,
                    "max_drawdown": am.calc_mdd(traded),
                    "win_rate": float((traded > 0).mean()) if len(traded) else np.nan,
                })
    return pd.DataFrame(rows).sort_values(
        ["bad_def", "capture_rate", "cum_ret"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def run() -> None:
    samples = build_selected_samples()
    effects = summarize_effects(samples)
    rules = scan_simple_rules(samples)

    sample_path = RESULTS_DIR / "selected_stock_bad_feature_samples_20260418.csv"
    effect_path = RESULTS_DIR / "selected_stock_bad_feature_relation_20260418.csv"
    rule_path = RESULTS_DIR / "selected_stock_bad_feature_rule_scan_20260418.csv"

    samples.to_csv(sample_path, index=False, encoding="utf-8-sig")
    effects.to_csv(effect_path, index=False, encoding="utf-8-sig")
    rules.to_csv(rule_path, index=False, encoding="utf-8-sig")

    for bad_def in ["bad3", "bad5"]:
        print(f"\n=== {bad_def} top effects ===")
        print(effects[effects["bad_def"] == bad_def].head(15)[["factor", "mean_diff", "effect_z", "spearman_bad"]].to_string(index=False))
        print(f"\n=== {bad_def} top simple rules ===")
        print(rules[rules["bad_def"] == bad_def].head(15)[["rule", "skip_rate", "capture_rate", "cum_ret", "max_drawdown"]].to_string(index=False))

    print(f"\nSaved: {sample_path}")
    print(f"Saved: {effect_path}")
    print(f"Saved: {rule_path}")


if __name__ == "__main__":
    run()
