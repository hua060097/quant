import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "results" / "daily_top5_details_Window_3_20260413_103953.json"
FACTORS_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
SCRIPT_PATH = BASE_DIR / "main_alpha原始_overlay.py"
OUT_DIR = BASE_DIR / "results"


PRIORITY_FACTORS = [
    "intra_sector_momentum_rank",
    "price_ma5_ratio",
    "abnormal_turnover",
    "max_rally_20d",
    "coskewness",
    "value_growth_rotation",
    "market_vols_adaptive_iv",
    "distance_from_52w_low",
    "sector_limit_overflow_alpha",
    "ma_convergence_score",
    "sector_momentum_interaction",
    "sector_momentum_consistency",
    "return_5d_sector_mean",
    "market_sentiment_ad",
    "sector_relative_volatility",
    "sector_mkt_corr_20d",
    "market_trend_strength",
    "trend_acceleration",
    "kurtosis_20d",
    "ppo_strength",
    "sw_circ_mv_rank_intra_sector",
    "range_compression",
    "return_3d_sector_mean",
    "ma60_slope",
    "mkt_sector_dispersion",
    "sector_leader_capture_rank",
    "relative_strength_20d",
    "market_liquidity_change",
    "volume_confirmed_pullback_alpha",
    "sector_relative_strength_vs_market",
    "volume_structure_alpha",
    "kdj_golden_cross_strength",
    "downside_beta",
    "sector_correlation_20d",
    "sector_momentum_accel_rank",
    "volatility_structure_alpha",
    "max_drawdown_20d",
    "return_10d_sector_mean",
    "amihud_illiquidity",
    "sector_liquidity_gated_turnover_z",
    "shadow_dominance",
    "sector_volatility_5d",
    "market_volatility_context",
    "sector_hhi_change",
    "sector_concentration",
    "cpv_factor",
    "open_to_high_ratio",
    "sector_rank_change_5d",
    "resistance_distance",
    "trend_pullback_composite_alpha",
    "pct_chg_sector_mean",
    "return_20d_sector_mean",
    "sector_turnover_change_5d_20d",
    "market_trend_adaptive_beta",
    "sector_momentum_dispersion_rank",
    "intraday_reversal",
    "turnover_stability",
    "sector_vols_divergence_alpha",
    "sector_strong_leader_alpha",
    "ulcer_index",
    "momentum_decay",
    "sector_dispersion_leader_alpha",
    "ma_alignment_score",
    "money_flow_pressure",
    "size_rotation_strength",
    "stock_sector_pos_align",
    "beta_60d",
    "pct_chg_2d",
    "intra_sector_breadth",
    "sector_broken_resilience_alpha",
    "sector_dispersion",
    "return_10d_intra_z",
    "limit_sustain_alpha",
    "skewness_20d",
    "return_20d_intra_z",
    "kdj_k_9",
    "market_emotion_resonance_alpha",
    "support_distance",
    "trend_consistency",
    "severe_abnormal_pressure_10d",
    "mkt_regime_trend_pressure_v2",
    "mkt_dispersion_shock_v2",
    "mkt_ma5_followthrough_v3",
    "mkt_emotion_stability_v2",
    "emotion_quality_balance_v3",
    "sector_hotspot_concentration_v3",
    "sector_leadership_persistence_v3",
    "strong_stock_volume_confirmation_v3",
]


def _parse_stock_factors() -> list[str]:
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    m = re.search(
        r"def _get_stock_factors\(\):\s*\n\s*all_factors = \[(.*?)\n\s*\]\n\n\s*return all_factors",
        text,
        re.S,
    )
    if not m:
        raise RuntimeError("无法解析 _get_stock_factors")
    parsed = re.findall(r"'([^']+)'", m.group(1))
    keep = [f for f in PRIORITY_FACTORS if f in parsed]
    return keep


def _load_top5_json() -> pd.DataFrame:
    raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for date_str, rec in raw.items():
        row = {"date": pd.to_datetime(date_str)}
        for topk in ["top1", "top2", "top3", "top4", "top5"]:
            row[f"{topk}_code"] = rec[topk]["stock_code"]
            row[f"{topk}_ret"] = rec[topk]["real_return"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["spread12"] = df["top1_ret"] - df["top2_ret"]
    df["avg12"] = (df["top1_ret"] + df["top2_ret"]) / 2.0
    df["both_big_loss"] = (df["top1_ret"] <= -0.03) & (df["top2_ret"] <= -0.03)
    df["t1_crash_t2_boom"] = (df["top1_ret"] <= -0.03) & (df["top2_ret"] >= 0.03)
    df["t1_boom_t2_crash"] = (df["top1_ret"] >= 0.03) & (df["top2_ret"] <= -0.03)
    return df


def _load_factor_panel(columns: list[str]) -> pd.DataFrame:
    schema = set(pq.ParquetFile(FACTORS_PATH).schema.names)
    use = [c for c in columns if c in schema]
    fac = pd.read_parquet(FACTORS_PATH, columns=["date", "stock_code"] + use)
    fac["date"] = pd.to_datetime(fac["date"])
    return fac


def _detect_market_factors(fac: pd.DataFrame, factors: list[str]) -> tuple[list[str], list[str]]:
    market = []
    stock = []
    for col in factors:
        if col not in fac.columns:
            continue
        nunq = fac.groupby("date")[col].nunique(dropna=True)
        share_one = float((nunq <= 1).mean()) if len(nunq) else 0.0
        med = float(nunq.median()) if len(nunq) else np.nan
        if med <= 1 and share_one >= 0.95:
            market.append(col)
        else:
            stock.append(col)
    return market, stock


def _attach_factor_views(top_df: pd.DataFrame, fac: pd.DataFrame, market_factors: list[str], stock_factors: list[str]) -> pd.DataFrame:
    out = top_df.copy()
    if market_factors:
        daily = fac.groupby("date", as_index=False)[market_factors].first()
        out = out.merge(daily, on="date", how="left")
    for side in ["top1", "top2"]:
        if stock_factors:
            tmp = fac[["date", "stock_code"] + stock_factors].rename(
                columns={
                    "stock_code": f"{side}_code",
                    **{c: f"{side}__{c}" for c in stock_factors},
                }
            )
            out = out.merge(tmp, on=["date", f"{side}_code"], how="left")
    return out


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    z = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(z) < 30:
        return np.nan
    if z.iloc[:, 0].nunique() <= 1 or z.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(z.iloc[:, 0].corr(z.iloc[:, 1], method="spearman"))


def _event_shift_rows(work: pd.DataFrame, market_factors: list[str], stock_factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_defs = {
        "both_big_loss": "both_big_loss",
        "t1_crash_t2_boom": "t1_crash_t2_boom",
        "t1_boom_t2_crash": "t1_boom_t2_crash",
    }

    spread_rows = []
    event_rows = []

    for factor in market_factors:
        s = work[factor]
        spread_rows.append(
            {
                "factor": factor,
                "kind": "market_raw",
                "spearman_spread12": _safe_spearman(s, work["spread12"]),
                "spearman_avg12": _safe_spearman(s, work["avg12"]),
            }
        )
        for event_name, event_col in event_defs.items():
            mask = work[event_col]
            mu_event = float(pd.to_numeric(s[mask], errors="coerce").mean()) if int(mask.sum()) else np.nan
            mu_other = float(pd.to_numeric(s[~mask], errors="coerce").mean()) if int((~mask).sum()) else np.nan
            event_rows.append(
                {
                    "factor": factor,
                    "kind": "market_raw",
                    "event": event_name,
                    "event_mean": mu_event,
                    "other_mean": mu_other,
                    "delta": mu_event - mu_other if pd.notna(mu_event) and pd.notna(mu_other) else np.nan,
                    "spearman_event": _safe_spearman(s, work[event_col].astype(int)),
                }
            )

    for factor in stock_factors:
        top1 = work[f"top1__{factor}"]
        top2 = work[f"top2__{factor}"]
        diff = top1 - top2
        for kind, series in [
            ("top1_raw", top1),
            ("top2_raw", top2),
            ("diff_12", diff),
        ]:
            spread_rows.append(
                {
                    "factor": factor,
                    "kind": kind,
                    "spearman_spread12": _safe_spearman(series, work["spread12"]),
                    "spearman_avg12": _safe_spearman(series, work["avg12"]),
                }
            )
            for event_name, event_col in event_defs.items():
                mask = work[event_col]
                mu_event = float(pd.to_numeric(series[mask], errors="coerce").mean()) if int(mask.sum()) else np.nan
                mu_other = float(pd.to_numeric(series[~mask], errors="coerce").mean()) if int((~mask).sum()) else np.nan
                event_rows.append(
                    {
                        "factor": factor,
                        "kind": kind,
                        "event": event_name,
                        "event_mean": mu_event,
                        "other_mean": mu_other,
                        "delta": mu_event - mu_other if pd.notna(mu_event) and pd.notna(mu_other) else np.nan,
                        "spearman_event": _safe_spearman(series, work[event_col].astype(int)),
                    }
                )

    return pd.DataFrame(spread_rows), pd.DataFrame(event_rows)


def _expanding_threshold(series: pd.Series, q: float, min_hist: int = 20) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = np.full(len(vals), np.nan)
    hist: list[float] = []
    for i, v in enumerate(vals):
        if len(hist) >= min_hist:
            out[i] = float(pd.Series(hist).quantile(q))
        if pd.notna(v):
            hist.append(float(v))
    return pd.Series(out, index=series.index)


def _calc_mdd(ret: pd.Series) -> float:
    eq = (1 + pd.Series(ret).fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def _summary(name: str, ret: pd.Series, active: pd.Series | None = None) -> dict:
    ret = pd.Series(ret).fillna(0.0)
    act = pd.Series(True, index=ret.index) if active is None else pd.Series(active, index=ret.index).fillna(False)
    return {
        "scheme": name,
        "days": int(len(ret)),
        "active_days": int(act.sum()),
        "pass_rate": float(act.mean()),
        "win_rate": float((ret > 0).mean()),
        "mean_ret": float(ret.mean()),
        "cum_ret": float((1 + ret).prod() - 1),
        "max_drawdown": _calc_mdd(ret),
    }


def _design_rules(work: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # 当前默认切换
    mkt_q30 = _expanding_threshold(work["mkt_ma5_followthrough_v3"], 0.3)
    choose_top1 = (
        (pd.to_numeric(work["mkt_ma5_followthrough_v3"], errors="coerce") >= mkt_q30)
        & (pd.to_numeric(work["top1__severe_abnormal_pressure_10d"], errors="coerce") < 0.40)
        & mkt_q30.notna()
    )
    current_switch = pd.Series(np.where(choose_top1, work["top1_ret"], work["top2_ret"]), index=work.index)
    rows.append(_summary("current_switch_mkt_exp_q30_reg10_raw_0p40", current_switch))

    # 市场 skip 候选
    trend_q50 = _expanding_threshold(work["market_trend_strength"], 0.5)
    pressure_q50 = _expanding_threshold(work["mkt_regime_trend_pressure_v2"], 0.5)
    pressure_q40 = _expanding_threshold(work["mkt_regime_trend_pressure_v2"], 0.4)
    ma5_q60 = _expanding_threshold(work["mkt_ma5_followthrough_v3"], 0.6)
    disp_q50 = _expanding_threshold(work["mkt_sector_dispersion"], 0.5)
    shock_q70 = _expanding_threshold(work["mkt_dispersion_shock_v2"], 0.7)

    skip_defs = {
        "skip_trend_low_and_pressure_low": (work["market_trend_strength"] <= trend_q50) & (work["mkt_regime_trend_pressure_v2"] <= pressure_q50),
        "skip_ma5_low_and_shock_low": (work["mkt_ma5_followthrough_v3"] <= ma5_q60) & (work["mkt_dispersion_shock_v2"] <= shock_q70),
        "skip_trend_low_and_dispersion_high": (work["market_trend_strength"] <= trend_q50) & (work["mkt_sector_dispersion"] >= disp_q50),
    }
    for name, skip in skip_defs.items():
        rows.append(_summary(f"current_switch_plus_{name}", np.where(skip, 0.0, current_switch), ~skip))

    # 个股差值切换增强
    abn_diff = pd.to_numeric(work["top1__abnormal_turnover"], errors="coerce") - pd.to_numeric(work["top2__abnormal_turnover"], errors="coerce")
    ppo_diff = pd.to_numeric(work["top1__ppo_strength"], errors="coerce") - pd.to_numeric(work["top2__ppo_strength"], errors="coerce")
    turn_diff = pd.to_numeric(work["top1__sector_turnover_change_5d_20d"], errors="coerce") - pd.to_numeric(work["top2__sector_turnover_change_5d_20d"], errors="coerce")

    abn_q30 = _expanding_threshold(abn_diff, 0.3)
    ppo_q40 = _expanding_threshold(ppo_diff, 0.4)
    turn_q60 = _expanding_threshold(turn_diff, 0.6)

    flip_to_top2_1 = (abn_diff >= abn_q30) & (ppo_diff >= ppo_q40) & abn_q30.notna() & ppo_q40.notna()
    rows.append(_summary("current_switch_plus_stockdiff_abn_ppo_to_top2", np.where(flip_to_top2_1, work["top2_ret"], current_switch)))

    flip_to_top2_2 = (abn_diff >= abn_q30) & (turn_diff >= turn_q60) & abn_q30.notna() & turn_q60.notna()
    rows.append(_summary("current_switch_plus_stockdiff_abn_turn_to_top2", np.where(flip_to_top2_2, work["top2_ret"], current_switch)))

    # 市场 skip + 个股差值切换
    combined_skip = skip_defs["skip_trend_low_and_pressure_low"]
    combined_switch = np.where(flip_to_top2_1, work["top2_ret"], current_switch)
    rows.append(_summary("current_switch_plus_skip_trend_pressure_plus_abn_ppo", np.where(combined_skip, 0.0, combined_switch), ~combined_skip))

    combined_skip2 = skip_defs["skip_ma5_low_and_shock_low"]
    combined_switch2 = np.where(flip_to_top2_2, work["top2_ret"], current_switch)
    rows.append(_summary("current_switch_plus_skip_ma5_shock_plus_abn_turn", np.where(combined_skip2, 0.0, combined_switch2), ~combined_skip2))

    return pd.DataFrame(rows).sort_values("cum_ret", ascending=False)


def main():
    factors = _parse_stock_factors()
    fac = _load_factor_panel(factors)
    market_factors, stock_factors = _detect_market_factors(fac, factors)
    top_df = _load_top5_json()
    work = _attach_factor_views(top_df, fac, market_factors, stock_factors)

    spread_df, event_df = _event_shift_rows(work, market_factors, stock_factors)
    spread_df.to_csv(OUT_DIR / "model_decomposition_spread_factor_relation_20260417.csv", index=False, encoding="utf-8-sig")
    event_df.to_csv(OUT_DIR / "model_decomposition_event_factor_relation_20260417.csv", index=False, encoding="utf-8-sig")

    rule_df = _design_rules(work)
    rule_df.to_csv(OUT_DIR / "model_decomposition_rule_compare_20260417.csv", index=False, encoding="utf-8-sig")

    print("Top spread relations:")
    print(spread_df.reindex(spread_df["spearman_spread12"].abs().sort_values(ascending=False).index).head(30).to_string(index=False))
    print("\nTop event relations:")
    print(event_df.reindex(event_df["spearman_event"].abs().sort_values(ascending=False).index).head(40).to_string(index=False))
    print("\nRule compare:")
    print(rule_df.to_string(index=False))


if __name__ == "__main__":
    main()
