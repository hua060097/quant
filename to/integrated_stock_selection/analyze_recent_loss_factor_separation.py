import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"

RUN_FILES = {
    "old162_off": RESULTS / "walkforward_metrics_20260512_194610.json",
    "latest184_v3": RESULTS / "walkforward_metrics_20260520_190404.json",
    "latest185_off": RESULTS / "walkforward_metrics_20260520_210454.json",
}

FACTOR_COLS = [
    "market_60d_new_high_ratio",
    "market_advance_decline_ratio",
    "market_sentiment_ad",
    "market_volatility_context",
    "mkt_dispersion_shock_v2",
    "market_liquidity_change",
    "market_trend_strength",
    "mkt_regime_trend_pressure_v2",
    "market_advance_decline_ratio_percentile_20d",
    "mkt_emotion_stability_v2",
    "mkt_breadth_thrust_v2",
    "mkt_ma5_followthrough_v3",
    "mkt_ma5_followthrough_v3_exp",
    "mkt_liquidity_impulse_v2",
    "mkt_sector_dispersion",
    "sector_leadership_persistence_v3",
    "sector_hotspot_concentration_v3",
    "pct_chg_sector_mean",
    "return_3d_sector_mean",
    "return_10d_sector_mean",
    "return_20d_sector_mean",
    "sector_momentum_interaction",
    "sector_dispersion",
    "sector_momentum_dispersion_rank",
    "sector_momentum_consistency",
    "sector_volatility_5d",
    "return_5d_sector_mean",
    "sw_circ_mv_rank_intra_sector",
    "sector_relative_volatility",
    "prev_day_blowoff_risk",
    "ma5_ma10_flattening_after_runup",
    "volume_price_exhaustion",
    "sector_strong_leader_alpha",
    "sector_leader_capture_rank",
    "sector_rotation_breakout_intra_z",
    "sector_trend_gated_momentum_intra_z",
    "sector_momentum_accel_rank",
    "sector_liquidity_gated_turnover_z",
    "sector_weak_reversal_alpha",
    "intra_sector_momentum_rank",
    "sector_relative_strength_vs_market",
    "market_trend_adaptive_beta",
    "sector_turnover_change_5d_20d",
    "mkt_sector_dispersion",
    "market_vols_adaptive_iv",
    "sector_vols_divergence_alpha",
    "sector_hhi_change",
    "intra_sector_breadth",
    "sector_excess_intra_rank",
    "sector_mkt_corr_20d",
    "sector_turnover_surge_pv_alpha",
    "sector_broken_resilience_alpha",
    "market_emotion_resonance_alpha",
    "sector_dispersion_leader_alpha",
    "sector_limit_overflow_alpha",
    "prior_high_volume_contest_20d",
    "sector_correlation_20d",
    "stock_sector_pos_align",
    "gap_power",
    "sector_concentration",
    "sector_breadth",
    "sector_rank_change_5d",
    "abnormal_turnover",
    "turnover_stability",
    "momentum_decay",
    "stable_momentum",
    "ulcer_index",
    "money_flow_pressure",
    "money_flow_pressure_pct",
    "max_rally_20d",
    "max_rally_10d",
    "max_drawdown_20d",
    "volume_structure_alpha",
    "cpv_factor",
    "close_position_60d",
    "volume_confirmed_pullback_alpha",
    "close_position_20d",
    "high_position_volume_surge",
    "stock_volume_ratio_20d",
    "high_pos_reversal_volume",
    "high_volume_no_gain_3d",
    "shadow_reversal_volume_pressure",
    "high_pos_volume_no_gain",
    "turnover_z_20",
    "close_below_ma5_after_runup",
    "true_turnover_10d",
    "low_break_prev_low_after_runup",
    "runup_low_break",
    "runup_ma5_exhaustion",
    "momentum_peak_decay_3d",
    "return_20d_intra_z",
    "limit_down_3d",
]


def load_events():
    rows = []
    for ds, path in RUN_FILES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for t in data[0]["all_results"]["1"]["trades"]:
            rows.append(
                {
                    "dataset": ds,
                    "stock_code": t["stock_code"],
                    "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                    "sell_date": pd.Timestamp(t["sell_datetime"][:10]),
                    "actual_return": float(t["actual_return"]),
                    "profit_loss": float(t["profit_loss"]),
                }
            )
    events = pd.DataFrame(rows)
    events["key"] = events["stock_code"] + "|" + events["buy_date"].dt.strftime("%Y-%m-%d")
    return events


def enrich(events):
    df_all = pd.read_parquet(FACTOR_FILE)
    factor_cols = list(dict.fromkeys(c for c in FACTOR_COLS if c in df_all.columns))
    cols = ["date", "stock_code"] + factor_cols
    fac = df_all[cols].copy()
    del df_all
    fac["date"] = pd.to_datetime(fac["date"])
    fac = fac.sort_values(["stock_code", "date"])
    grouped = {}
    for stock, g in fac.groupby("stock_code", sort=False):
        grouped[stock] = (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))

    rows = []
    for _, e in events.iterrows():
        rec = e.to_dict()
        item = grouped.get(e["stock_code"])
        if item is None:
            rows.append(rec)
            continue
        dates, g = item
        pos = np.searchsorted(dates, np.datetime64(e["buy_date"]), side="left") - 1
        if pos >= 0:
            r = g.iloc[int(pos)]
            rec["factor_date"] = r["date"]
            for c in FACTOR_COLS:
                if c in r.index:
                    rec[c] = r[c]
        rows.append(rec)
    return pd.DataFrame(rows)


def current_v3_mask(df):
    return (
        ((df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4))
        | (
            ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
            | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
            | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15))
        )
        | ((df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40))
        | (
            (df["market_liquidity_change"] < 0.94)
            & (
                (df["ulcer_index"] < -0.8)
                | (df["momentum_peak_decay_3d"] > 1.0)
                | (df["intra_sector_momentum_rank"] < 0.67)
            )
        )
        | ((df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38))
        | (df.get("limit_down_3d", pd.Series(0, index=df.index)).fillna(0).astype(float) == 1)
    ).fillna(False)


def dedup_reference(df):
    # Prefer the latest no-filter run for recent full sample, then old162, then v3 run.
    priority = {"latest185_off": 0, "old162_off": 1, "latest184_v3": 2}
    x = df.copy()
    x["priority"] = x["dataset"].map(priority)
    x = x.sort_values(["key", "priority"])
    return x.drop_duplicates("key", keep="first").drop(columns=["priority"])


def zdist(a, b):
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    av, bv = av[np.isfinite(av)], bv[np.isfinite(bv)]
    if len(av) < 2 or len(bv) < 3:
        return np.nan
    pooled = np.nanstd(np.r_[av, bv])
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return (np.nanmedian(av) - np.nanmedian(bv)) / pooled


def percentile_position(values, target_median):
    v = pd.Series(values).dropna()
    if len(v) == 0 or pd.isna(target_median):
        return np.nan
    return float((v <= target_median).mean())


def side_stats(series, recent, wins, losses):
    rm = recent[series].median()
    wm = wins[series].median()
    lm = losses[series].median()
    wp = percentile_position(wins[series], rm)
    lp = percentile_position(losses[series], rm)
    all_ref = pd.concat([wins[series], losses[series]], axis=0).dropna()
    ap = percentile_position(all_ref, rm)
    zw = zdist(recent[series], wins[series])
    zl = zdist(recent[series], losses[series])
    return rm, wm, lm, wp, lp, ap, zw, zl


def main():
    events = load_events()
    df = enrich(events)
    df["current_v3"] = current_v3_mask(df)
    ref = dedup_reference(df)

    recent = df[
        df["dataset"].eq("latest184_v3")
        & (df["buy_date"] >= pd.Timestamp("2026-04-01"))
        & (df["actual_return"] <= -0.05)
        & ~df["current_v3"]
    ].copy()
    hist = ref[~ref["key"].isin(recent["key"])].copy()
    wins = hist[hist["actual_return"] >= 0.10].copy()
    losses = hist[hist["actual_return"] <= -0.05].copy()

    print("== group sizes ==")
    print(f"recent_v3_leak_losses={len(recent)}")
    print(f"historical_big_wins={len(wins)}")
    print(f"historical_big_losses={len(losses)}")

    print("\n== recent leak losses factor snapshot ==")
    show_cols = [
        "stock_code",
        "buy_date",
        "actual_return",
        "profit_loss",
        "market_advance_decline_ratio",
        "mkt_dispersion_shock_v2",
        "market_liquidity_change",
        "market_60d_new_high_ratio",
        "mkt_liquidity_impulse_v2",
        "ulcer_index",
        "prev_day_blowoff_risk",
        "sector_momentum_accel_rank",
        "return_20d_intra_z",
        "runup_ma5_exhaustion",
        "cpv_factor",
    ]
    print(recent[show_cols].sort_values("profit_loss").to_string(index=False))

    rows = []
    for c in FACTOR_COLS:
        if c not in recent.columns:
            continue
        if recent[c].notna().sum() < max(2, len(recent) // 2):
            continue
        rm, wm, lm, wp, lp, ap, zw, zl = side_stats(c, recent, wins, losses)
        if not np.isfinite(zw):
            continue
        # Separation favors factors where recent losses are far from wins and not merely identical to old losses.
        rows.append(
            {
                "factor": c,
                "recent_med": rm,
                "win_med": wm,
                "loss_med": lm,
                "recent_pct_in_wins": wp,
                "recent_pct_in_losses": lp,
                "recent_pct_in_ref": ap,
                "z_vs_win": zw,
                "z_vs_loss": zl,
                "abs_z_vs_win": abs(zw),
                "direction": "high" if zw > 0 else "low",
                "score": abs(zw) + 0.25 * abs(zl),
            }
        )
    sep = pd.DataFrame(rows).sort_values(["score", "abs_z_vs_win"], ascending=False)
    out_dir = RESULTS
    recent.to_csv(out_dir / "recent_v3_leak_losses_factor_snapshot.csv", index=False, encoding="utf-8-sig")
    sep.to_csv(out_dir / "recent_loss_factor_separation.csv", index=False, encoding="utf-8-sig")
    print("\n== top factors separating recent losses from historical big wins ==")
    cols = [
        "factor",
        "direction",
        "recent_med",
        "win_med",
        "loss_med",
        "recent_pct_in_wins",
        "recent_pct_in_losses",
        "z_vs_win",
        "z_vs_loss",
        "score",
    ]
    print(sep[cols].head(40).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n== candidate thresholds from top factors ==")
    for _, r in sep.head(25).iterrows():
        c = r["factor"]
        vals = hist[c].dropna()
        if vals.empty:
            continue
        if r["direction"] == "low":
            ths = [recent[c].quantile(q) for q in [0.50, 0.75, 1.00]]
            for th in sorted(set(float(t) for t in ths if pd.notna(t))):
                mask_recent = recent[c] <= th
                mask_wins = wins[c] <= th
                mask_losses = losses[c] <= th
                print(
                    f"{c} <= {th:.4g}: recent {mask_recent.sum()}/{len(recent)}, "
                    f"wins {mask_wins.sum()}/{len(wins)}, losses {mask_losses.sum()}/{len(losses)}"
                )
        else:
            ths = [recent[c].quantile(q) for q in [0.00, 0.25, 0.50]]
            for th in sorted(set(float(t) for t in ths if pd.notna(t)), reverse=True):
                mask_recent = recent[c] >= th
                mask_wins = wins[c] >= th
                mask_losses = losses[c] >= th
                print(
                    f"{c} >= {th:.4g}: recent {mask_recent.sum()}/{len(recent)}, "
                    f"wins {mask_wins.sum()}/{len(wins)}, losses {mask_losses.sum()}/{len(losses)}"
                )

    print("\n== top 2-factor combinations among separating factors ==")
    top_factors = sep.head(24).to_dict("records")
    combos = []
    for i, a in enumerate(top_factors):
        for b in top_factors[i + 1 :]:
            fa, fb = a["factor"], b["factor"]
            if fa not in recent.columns or fb not in recent.columns:
                continue
            qa_list = [0.00, 0.25, 0.50] if a["direction"] == "high" else [0.50, 0.75, 1.00]
            qb_list = [0.00, 0.25, 0.50] if b["direction"] == "high" else [0.50, 0.75, 1.00]
            for qa in qa_list:
                tha = recent[fa].quantile(qa)
                if pd.isna(tha):
                    continue
                ma_r = recent[fa] >= tha if a["direction"] == "high" else recent[fa] <= tha
                ma_w = wins[fa] >= tha if a["direction"] == "high" else wins[fa] <= tha
                ma_l = losses[fa] >= tha if a["direction"] == "high" else losses[fa] <= tha
                for qb in qb_list:
                    thb = recent[fb].quantile(qb)
                    if pd.isna(thb):
                        continue
                    mb_r = recent[fb] >= thb if b["direction"] == "high" else recent[fb] <= thb
                    mb_w = wins[fb] >= thb if b["direction"] == "high" else wins[fb] <= thb
                    mb_l = losses[fb] >= thb if b["direction"] == "high" else losses[fb] <= thb
                    rh = int((ma_r & mb_r).sum())
                    wh = int((ma_w & mb_w).sum())
                    lh = int((ma_l & mb_l).sum())
                    if rh < 2:
                        continue
                    score = rh * 10 + lh * 2 - wh * 6
                    op_a = ">=" if a["direction"] == "high" else "<="
                    op_b = ">=" if b["direction"] == "high" else "<="
                    combos.append(
                        {
                            "score": score,
                            "recent_hits": rh,
                            "win_hits": wh,
                            "hist_loss_hits": lh,
                            "rule": f"{fa} {op_a} {tha:.4g} & {fb} {op_b} {thb:.4g}",
                            "fa": fa,
                            "fb": fb,
                            "tha": tha,
                            "thb": thb,
                        }
                    )
    combo_df = pd.DataFrame(combos).sort_values(
        ["score", "recent_hits", "win_hits", "hist_loss_hits"],
        ascending=[False, False, True, False],
    )
    combo_df.to_csv(out_dir / "recent_loss_top_factor_combos.csv", index=False, encoding="utf-8-sig")
    print(combo_df.head(40).to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
