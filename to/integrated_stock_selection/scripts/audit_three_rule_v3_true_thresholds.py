"""True-threshold audit for candidate three_rule_v3 additions.

Focus:
1. regime_aware_reversal_alpha high AND shadow_dominance_pct high
2. score_gap_1_2 low gate, then intraday_clv high OR sector_broken_resilience_alpha high OR momentum_peak_decay_3d high

This script uses actual threshold values rounded to 2 decimals for the scanned
factor thresholds. The low-gap gate for rule3 stays as the fixed 30% score-gap
gate from the original idea.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_three_rule_v3_candidate_rules import (
    CURRENT_RULE_COLS,
    EXTRA_RULE_COLS,
    FACTORS,
    MODEL_FILE,
    METRICS_FILE,
    RULE_NAMES,
    RESULTS,
    SHADOW_FILE,
    attach_prev_snapshot,
    current_three_rule_v3,
    load_actual_trades,
    load_factor_frame,
    load_model,
    load_shadow_trades,
    score_gap_panel,
)


OUT_DIR = RESULTS
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bucket_counts(ret: pd.Series) -> dict:
    ret = pd.to_numeric(ret, errors="coerce")
    return {
        "n": int(ret.notna().sum()),
        "big_loss_n": int((ret <= -0.04).sum()),
        "small_loss_n": int(((ret > -0.04) & (ret < 0)).sum()),
        "small_gain_n": int(((ret >= 0.03) & (ret < 0.08)).sum()),
        "big_gain_n": int((ret >= 0.08).sum()),
        "strict_big_gain_n": int((ret >= 0.10).sum()),
    }


def threshold_candidates(series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.array([], dtype=float)
    return np.array(sorted({round(float(v), 2) for v in s.to_numpy()}), dtype=float)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    try:
        xx = pd.to_numeric(x, errors="coerce")
        yy = pd.to_numeric(y, errors="coerce")
        mask = xx.notna() & yy.notna()
        if mask.sum() < 3:
            return np.nan
        return float(xx[mask].rank().corr(yy[mask].rank(), method="pearson"))
    except Exception:
        return np.nan


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    try:
        xx = pd.to_numeric(x, errors="coerce")
        yy = pd.to_numeric(y, errors="coerce")
        mask = xx.notna() & yy.notna()
        if mask.sum() < 3:
            return np.nan
        return float(xx[mask].corr(yy[mask]))
    except Exception:
        return np.nan


def summarize_mask(df: pd.DataFrame, mask: pd.Series) -> dict:
    mask = pd.Series(mask, index=df.index).fillna(False)
    sel = df[mask].copy()
    ret = pd.to_numeric(sel.get("actual_return"), errors="coerce")
    total = pd.to_numeric(df.get("actual_return"), errors="coerce")
    buckets = bucket_counts(ret)
    total_buckets = bucket_counts(total)
    return {
        "mask_n": int(mask.sum()),
        "actual_n": int((df["source"] == "actual").sum()),
        "shadow_n": int((df["source"] == "shadow").sum()),
        "mask_mean_return": float(ret.mean()) if len(sel) else np.nan,
        "mask_median_return": float(ret.median()) if len(sel) else np.nan,
        "mask_win_rate": float((ret > 0).mean()) if len(sel) else np.nan,
        "mask_big_loss_rate": float((ret <= -0.04).mean()) if len(sel) else np.nan,
        "mask_small_loss_rate": float(((ret > -0.04) & (ret < 0)).mean()) if len(sel) else np.nan,
        "mask_small_gain_rate": float(((ret >= 0.03) & (ret < 0.08)).mean()) if len(sel) else np.nan,
        "mask_big_gain_rate": float((ret >= 0.08).mean()) if len(sel) else np.nan,
        "mask_strict_big_gain_rate": float((ret >= 0.10).mean()) if len(sel) else np.nan,
        "big_loss_capture_rate": float(buckets["big_loss_n"] / max(total_buckets["big_loss_n"], 1)),
        "small_loss_capture_rate": float(buckets["small_loss_n"] / max(total_buckets["small_loss_n"], 1)),
        "small_gain_kill_rate": float(buckets["small_gain_n"] / max(total_buckets["small_gain_n"], 1)),
        "big_gain_kill_rate": float(buckets["big_gain_n"] / max(total_buckets["big_gain_n"], 1)),
        "strict_big_gain_kill_rate": float(buckets["strict_big_gain_n"] / max(total_buckets["strict_big_gain_n"], 1)),
    }


def scan_rule1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = pd.to_numeric(df["regime_aware_reversal_alpha"], errors="coerce")
    y = pd.to_numeric(df["shadow_dominance_pct"], errors="coerce")
    tx = threshold_candidates(x)
    ty = threshold_candidates(y)
    rows = []
    ret = pd.to_numeric(df["actual_return"], errors="coerce")
    cur = df["current_v3"].fillna(False)
    for t1 in tx:
        mx = x >= t1
        for t2 in ty:
            mask = mx & (y >= t2)
            sel_ret = ret[mask]
            row = {
                "rule": "rule1",
                "threshold_1": round(float(t1), 2),
                "threshold_2": round(float(t2), 2),
                "mask_n": int(mask.sum()),
                "current_overlap": int((mask & cur).sum()),
                "incremental_n": int((mask & ~cur).sum()),
                "incremental_big_loss": int(((mask & ~cur) & (ret <= -0.04)).sum()),
                "incremental_small_loss": int(((mask & ~cur) & (ret > -0.04) & (ret < 0)).sum()),
                "incremental_small_gain": int(((mask & ~cur) & (ret >= 0.03) & (ret < 0.08)).sum()),
                "incremental_big_gain": int(((mask & ~cur) & (ret >= 0.08)).sum()),
                "incremental_strict_big_gain": int(((mask & ~cur) & (ret >= 0.10)).sum()),
                "mean_return": float(sel_ret.mean()) if mask.any() else np.nan,
                "median_return": float(sel_ret.median()) if mask.any() else np.nan,
                "pearson_with_return": safe_pearson(pd.Series(np.where(mask, x, np.nan)), ret),
                "spearman_with_return": safe_spearman(pd.Series(np.where(mask, x, np.nan)), ret),
            }
            row.update(summarize_mask(df, mask))
            row["score"] = (
                2.0 * row["big_loss_capture_rate"]
                - 1.5 * row["big_gain_kill_rate"]
                - 0.8 * row["small_gain_kill_rate"]
                - 0.3 * row["current_overlap"] / max(row["mask_n"], 1)
                + 0.2 * row["mask_big_loss_rate"]
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("rule1 scan produced no rows")
    best = out.sort_values(
        ["score", "big_loss_capture_rate", "incremental_big_loss", "big_gain_kill_rate", "current_overlap"],
        ascending=[False, False, False, True, True],
    ).iloc[0].to_dict()
    return out, best


def scan_rule3(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, float]:
    gap = pd.to_numeric(df["score_gap_1_2"], errors="coerce")
    low_gap_th = float(gap.dropna().quantile(0.30))
    low_gap = gap <= low_gap_th
    sub = df[low_gap].copy()
    if sub.empty:
        raise ValueError("rule3 low-gap subset is empty")
    a = pd.to_numeric(df["intraday_clv"], errors="coerce")
    b = pd.to_numeric(df["sector_broken_resilience_alpha"], errors="coerce")
    c = pd.to_numeric(df["momentum_peak_decay_3d"], errors="coerce")
    ta = threshold_candidates(sub["intraday_clv"])
    tb = threshold_candidates(sub["sector_broken_resilience_alpha"])
    tc = threshold_candidates(sub["momentum_peak_decay_3d"])
    ret = pd.to_numeric(df["actual_return"], errors="coerce")
    cur = df["current_v3"].fillna(False)
    rows = []
    for t1 in ta:
        ma = a >= t1
        for t2 in tb:
            mb = b >= t2
            for t3 in tc:
                mask = low_gap & (ma | mb | (c >= t3))
                sel_ret = ret[mask]
                row = {
                    "rule": "rule3",
                    "low_gap_threshold": round(float(low_gap_th), 6),
                    "threshold_1": round(float(t1), 2),
                    "threshold_2": round(float(t2), 2),
                    "threshold_3": round(float(t3), 2),
                    "mask_n": int(mask.sum()),
                    "current_overlap": int((mask & cur).sum()),
                    "incremental_n": int((mask & ~cur).sum()),
                    "incremental_big_loss": int(((mask & ~cur) & (ret <= -0.04)).sum()),
                    "incremental_small_loss": int(((mask & ~cur) & (ret > -0.04) & (ret < 0)).sum()),
                    "incremental_small_gain": int(((mask & ~cur) & (ret >= 0.03) & (ret < 0.08)).sum()),
                    "incremental_big_gain": int(((mask & ~cur) & (ret >= 0.08)).sum()),
                    "incremental_strict_big_gain": int(((mask & ~cur) & (ret >= 0.10)).sum()),
                    "mean_return": float(sel_ret.mean()) if mask.any() else np.nan,
                    "median_return": float(sel_ret.median()) if mask.any() else np.nan,
                    "pearson_with_return": safe_pearson(pd.Series(np.where(mask, a, np.nan)), ret),
                    "spearman_with_return": safe_spearman(pd.Series(np.where(mask, a, np.nan)), ret),
                }
                row.update(summarize_mask(df, mask))
                row["score"] = (
                    2.0 * row["big_loss_capture_rate"]
                    - 1.5 * row["big_gain_kill_rate"]
                    - 0.8 * row["small_gain_kill_rate"]
                    - 0.3 * row["current_overlap"] / max(row["mask_n"], 1)
                    + 0.2 * row["mask_big_loss_rate"]
                )
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("rule3 scan produced no rows")
    best = out.sort_values(
        ["score", "big_loss_capture_rate", "incremental_big_loss", "big_gain_kill_rate", "current_overlap"],
        ascending=[False, False, False, True, True],
    ).iloc[0].to_dict()
    return out, best, low_gap_th


def neighborhood_scan_rule1(df: pd.DataFrame, best: dict) -> pd.DataFrame:
    x = pd.to_numeric(df["regime_aware_reversal_alpha"], errors="coerce")
    y = pd.to_numeric(df["shadow_dominance_pct"], errors="coerce")
    ret = pd.to_numeric(df["actual_return"], errors="coerce")
    cur = df["current_v3"].fillna(False)
    vals1 = np.round(np.array([best["threshold_1"] + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02]]), 2)
    vals2 = np.round(np.array([best["threshold_2"] + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02]]), 2)
    rows = []
    for t1 in vals1:
        for t2 in vals2:
            mask = (x >= t1) & (y >= t2)
            rows.append({
                "rule": "rule1",
                "threshold_1": float(t1),
                "threshold_2": float(t2),
                **summarize_mask(df, mask),
                "incremental_big_loss": int(((mask & ~cur) & (ret <= -0.04)).sum()),
                "incremental_big_gain": int(((mask & ~cur) & (ret >= 0.08)).sum()),
                "incremental_small_loss": int(((mask & ~cur) & (ret > -0.04) & (ret < 0)).sum()),
                "incremental_small_gain": int(((mask & ~cur) & (ret >= 0.03) & (ret < 0.08)).sum()),
                "mean_return": float(pd.to_numeric(df.loc[mask, "actual_return"], errors="coerce").mean()) if mask.any() else np.nan,
                "median_return": float(pd.to_numeric(df.loc[mask, "actual_return"], errors="coerce").median()) if mask.any() else np.nan,
            })
    return pd.DataFrame(rows)


def neighborhood_scan_rule3(df: pd.DataFrame, best: dict, low_gap_th: float) -> pd.DataFrame:
    gap = pd.to_numeric(df["score_gap_1_2"], errors="coerce")
    low_gap = gap <= low_gap_th
    a = pd.to_numeric(df["intraday_clv"], errors="coerce")
    b = pd.to_numeric(df["sector_broken_resilience_alpha"], errors="coerce")
    c = pd.to_numeric(df["momentum_peak_decay_3d"], errors="coerce")
    ret = pd.to_numeric(df["actual_return"], errors="coerce")
    cur = df["current_v3"].fillna(False)
    vals1 = np.round(np.array([best["threshold_1"] + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02]]), 2)
    vals2 = np.round(np.array([best["threshold_2"] + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02]]), 2)
    vals3 = np.round(np.array([best["threshold_3"] + d for d in [-0.02, -0.01, 0.0, 0.01, 0.02]]), 2)
    rows = []
    for t1 in vals1:
        ma = a >= t1
        for t2 in vals2:
            mb = b >= t2
            for t3 in vals3:
                mask = low_gap & (ma | mb | (c >= t3))
                rows.append({
                    "rule": "rule3",
                    "low_gap_threshold": round(float(low_gap_th), 6),
                    "threshold_1": float(t1),
                    "threshold_2": float(t2),
                    "threshold_3": float(t3),
                    **summarize_mask(df, mask),
                    "incremental_big_loss": int(((mask & ~cur) & (ret <= -0.04)).sum()),
                    "incremental_big_gain": int(((mask & ~cur) & (ret >= 0.08)).sum()),
                    "incremental_small_loss": int(((mask & ~cur) & (ret > -0.04) & (ret < 0)).sum()),
                    "incremental_small_gain": int(((mask & ~cur) & (ret >= 0.03) & (ret < 0.08)).sum()),
                    "mean_return": float(pd.to_numeric(df.loc[mask, "actual_return"], errors="coerce").mean()) if mask.any() else np.nan,
                    "median_return": float(pd.to_numeric(df.loc[mask, "actual_return"], errors="coerce").median()) if mask.any() else np.nan,
                })
    return pd.DataFrame(rows)


def monotonicity_table(scan_df: pd.DataFrame, rule: str, varying_col: str) -> pd.DataFrame:
    out = scan_df[[varying_col, "mask_n", "big_loss_capture_rate", "big_gain_kill_rate", "small_gain_kill_rate", "mean_return"]].copy()
    out = out.sort_values(varying_col).reset_index(drop=True)
    out["delta_mask_n"] = out["mask_n"].diff()
    out["delta_big_loss_capture_rate"] = out["big_loss_capture_rate"].diff()
    out["delta_big_gain_kill_rate"] = out["big_gain_kill_rate"].diff()
    out["delta_small_gain_kill_rate"] = out["small_gain_kill_rate"].diff()
    out["delta_mean_return"] = out["mean_return"].diff()
    out["spearman_thresh_vs_mask_n"] = safe_spearman(out[varying_col], out["mask_n"])
    out["spearman_thresh_vs_big_loss_capture"] = safe_spearman(out[varying_col], out["big_loss_capture_rate"])
    out["spearman_thresh_vs_big_gain_kill"] = safe_spearman(out[varying_col], out["big_gain_kill_rate"])
    out["spearman_thresh_vs_mean_return"] = safe_spearman(out[varying_col], out["mean_return"])
    out["rule"] = rule
    return out


def rule_overlap_compare(df: pd.DataFrame, cand_mask: pd.Series, cand_name: str) -> pd.DataFrame:
    rows = []
    for rule in RULE_NAMES + ["current_v3"]:
        if rule not in df.columns:
            continue
        m = df[rule].fillna(False)
        inter = int((cand_mask & m).sum())
        rows.append({
            "candidate": cand_name,
            "existing_rule": rule,
            "candidate_hits": int(cand_mask.sum()),
            "existing_hits": int(m.sum()),
            "intersection": inter,
            "candidate_overlap_rate": float(inter / max(int(cand_mask.sum()), 1)),
            "existing_overlap_rate": float(inter / max(int(m.sum()), 1)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    actual = load_actual_trades()
    shadow = load_shadow_trades()
    events = pd.concat([actual, shadow], ignore_index=True, sort=False)
    events["buy_date"] = pd.to_datetime(events["buy_date"])
    events["stock_code"] = events["stock_code"].astype(str)
    events["actual_return"] = pd.to_numeric(events["actual_return"], errors="coerce")

    model, feature_names = load_model()
    load_start = (pd.to_datetime(events["buy_date"].min()) - pd.Timedelta(days=35)).strftime("%Y-%m-%d")
    load_end = pd.to_datetime(events["buy_date"].max()).strftime("%Y-%m-%d")

    factor_cols = list(dict.fromkeys(
        ["date", "stock_code"]
        + feature_names
        + CURRENT_RULE_COLS
        + EXTRA_RULE_COLS
        + ["runup_ma5_exhaustion", "volume_price_exhaustion", "intraday_clv", "sector_broken_resilience_alpha", "momentum_peak_decay_3d"]
    ))
    factors = load_factor_frame(factor_cols, load_start, load_end)

    top1_panel = score_gap_panel(model, factors, feature_names)
    top1_panel["date"] = pd.to_datetime(top1_panel["date"])

    event_cols = list(dict.fromkeys(CURRENT_RULE_COLS + EXTRA_RULE_COLS))
    events = attach_prev_snapshot(events, factors, event_cols)
    events = events.merge(
        top1_panel[["date", "score_gap_1_2", "score_gap_1_3", "score_gap_1_5", "pred_score"]].rename(
            columns={"pred_score": "top1_pred_score"}
        ),
        left_on="buy_date",
        right_on="date",
        how="left",
    )
    events = events.drop(columns=[c for c in ["date_y"] if c in events.columns]).rename(columns={"date_x": "date"})
    events["score_gap_1_2"] = pd.to_numeric(events["score_gap_1_2"], errors="coerce")

    # Current three_rule_v3 masks.
    cur_masks = current_three_rule_v3(events)
    events = pd.concat([events, cur_masks], axis=1)
    events["current_v3"] = cur_masks["current_v3"].fillna(False)
    events["source"] = events["source"].fillna("actual")

    # Rule1 true thresholds.
    rule1_scan, rule1_best = scan_rule1(events)
    rule1_nb = neighborhood_scan_rule1(events, rule1_best)

    # Rule3 true thresholds.
    rule3_scan, rule3_best, low_gap_th = scan_rule3(events)
    rule3_nb = neighborhood_scan_rule3(events, rule3_best, low_gap_th)

    # Monotonicity sweeps around the chosen best point.
    rule1_mono_t1 = monotonicity_table(
        rule1_scan[rule1_scan["threshold_2"] == round(float(rule1_best["threshold_2"]), 2)].copy(),
        "rule1",
        "threshold_1",
    )
    rule1_mono_t2 = monotonicity_table(
        rule1_scan[rule1_scan["threshold_1"] == round(float(rule1_best["threshold_1"]), 2)].copy(),
        "rule1",
        "threshold_2",
    )
    rule3_mono_t1 = monotonicity_table(
        rule3_scan[(rule3_scan["threshold_2"] == round(float(rule3_best["threshold_2"]), 2)) & (rule3_scan["threshold_3"] == round(float(rule3_best["threshold_3"]), 2))].copy(),
        "rule3",
        "threshold_1",
    )
    rule3_mono_t2 = monotonicity_table(
        rule3_scan[(rule3_scan["threshold_1"] == round(float(rule3_best["threshold_1"]), 2)) & (rule3_scan["threshold_3"] == round(float(rule3_best["threshold_3"]), 2))].copy(),
        "rule3",
        "threshold_2",
    )
    rule3_mono_t3 = monotonicity_table(
        rule3_scan[(rule3_scan["threshold_1"] == round(float(rule3_best["threshold_1"]), 2)) & (rule3_scan["threshold_2"] == round(float(rule3_best["threshold_2"]), 2))].copy(),
        "rule3",
        "threshold_3",
    )

    # Summaries for actual/shadow/combined.
    summaries = []
    best_masks = {
        "rule1": (events["regime_aware_reversal_alpha"] >= float(rule1_best["threshold_1"])) & (events["shadow_dominance_pct"] >= float(rule1_best["threshold_2"])),
        "rule3": (events["score_gap_1_2"] <= float(low_gap_th))
        & (
            (events["intraday_clv"] >= float(rule3_best["threshold_1"]))
            | (events["sector_broken_resilience_alpha"] >= float(rule3_best["threshold_2"]))
            | (events["momentum_peak_decay_3d"] >= float(rule3_best["threshold_3"]))
        ),
    }
    for name, mask in best_masks.items():
        for subset in ["combined", "actual", "shadow"]:
            subset_mask = np.ones(len(events), dtype=bool)
            if subset != "combined":
                subset_mask = events["source"].eq(subset).to_numpy()
            df = events[subset_mask].copy()
            m = mask[subset_mask]
            row = {"rule": name, "subset": subset, **summarize_mask(df, m)}
            row["current_overlap"] = int((m & df["current_v3"].fillna(False)).sum())
            row["candidate_only_vs_current_v3"] = int((m & ~df["current_v3"].fillna(False)).sum())
            row["current_only_vs_candidate"] = int((~m & df["current_v3"].fillna(False)).sum())
            row["big_loss_false_kill"] = int(((m) & (pd.to_numeric(df["actual_return"], errors="coerce") > -0.04) & (pd.to_numeric(df["actual_return"], errors="coerce") < 0)).sum())
            row["big_gain_false_kill"] = int(((m) & (pd.to_numeric(df["actual_return"], errors="coerce") >= 0.08)).sum())
            row["small_gain_false_kill"] = int(((m) & (pd.to_numeric(df["actual_return"], errors="coerce") >= 0.03) & (pd.to_numeric(df["actual_return"], errors="coerce") < 0.08)).sum())
            summaries.append(row)
    summary_df = pd.DataFrame(summaries)

    rule1_compare = rule_overlap_compare(events, best_masks["rule1"].fillna(False), "rule1_best")
    rule3_compare = rule_overlap_compare(events, best_masks["rule3"].fillna(False), "rule3_best")
    compare_df = pd.concat([rule1_compare, rule3_compare], ignore_index=True)

    # Event detail for review.
    detail_cols = [
        "source", "buy_date", "stock_code", "actual_return", "profit_loss_pct", "predicted_return",
        "current_v3", "score_gap_1_2", "score_gap_1_3", "score_gap_1_5", "top1_pred_score",
        "regime_aware_reversal_alpha", "shadow_dominance_pct",
        "intraday_clv", "sector_broken_resilience_alpha", "momentum_peak_decay_3d",
    ]
    detail_cols = [c for c in detail_cols if c in events.columns]
    detail_df = events[detail_cols].copy()
    for name, mask in best_masks.items():
        detail_df[f"{name}_hit"] = mask.to_numpy()

    # Persist.
    rule1_scan.to_csv(OUT_DIR / "three_rule_v3_rule1_true_threshold_scan.csv", index=False, encoding="utf-8-sig")
    rule3_scan.to_csv(OUT_DIR / "three_rule_v3_rule3_true_threshold_scan.csv", index=False, encoding="utf-8-sig")
    rule1_nb.to_csv(OUT_DIR / "three_rule_v3_rule1_neighborhood.csv", index=False, encoding="utf-8-sig")
    rule3_nb.to_csv(OUT_DIR / "three_rule_v3_rule3_neighborhood.csv", index=False, encoding="utf-8-sig")
    pd.concat([rule1_mono_t1, rule1_mono_t2, rule3_mono_t1, rule3_mono_t2, rule3_mono_t3], ignore_index=True).to_csv(
        OUT_DIR / "three_rule_v3_true_threshold_monotonicity.csv", index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(OUT_DIR / "three_rule_v3_true_threshold_summary.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUT_DIR / "three_rule_v3_true_threshold_overlap_compare.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(OUT_DIR / "three_rule_v3_true_threshold_event_detail.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([rule1_best]).to_csv(OUT_DIR / "three_rule_v3_rule1_best.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([rule3_best]).to_csv(OUT_DIR / "three_rule_v3_rule3_best.csv", index=False, encoding="utf-8-sig")

    print("\n===== Rule1 best =====")
    print(pd.DataFrame([rule1_best])[["threshold_1", "threshold_2", "mask_n", "incremental_n", "incremental_big_loss", "incremental_big_gain", "big_loss_capture_rate", "big_gain_kill_rate", "small_gain_kill_rate", "score"]].to_string(index=False))
    print("\n===== Rule3 best =====")
    print(pd.DataFrame([rule3_best])[["low_gap_threshold", "threshold_1", "threshold_2", "threshold_3", "mask_n", "incremental_n", "incremental_big_loss", "incremental_big_gain", "big_loss_capture_rate", "big_gain_kill_rate", "small_gain_kill_rate", "score"]].to_string(index=False))
    print("\n===== Summary =====")
    print(summary_df.to_string(index=False))
    print("\n===== Overlap compare Top20 =====")
    print(compare_df.sort_values(["candidate_overlap_rate", "existing_overlap_rate"], ascending=[False, False]).head(20).to_string(index=False))
    print("\nOutputs written to:")
    print(f"  {OUT_DIR / 'three_rule_v3_rule1_true_threshold_scan.csv'}")
    print(f"  {OUT_DIR / 'three_rule_v3_rule3_true_threshold_scan.csv'}")
    print(f"  {OUT_DIR / 'three_rule_v3_rule1_neighborhood.csv'}")
    print(f"  {OUT_DIR / 'three_rule_v3_rule3_neighborhood.csv'}")
    print(f"  {OUT_DIR / 'three_rule_v3_true_threshold_monotonicity.csv'}")
    print(f"  {OUT_DIR / 'three_rule_v3_true_threshold_summary.csv'}")
    print(f"  {OUT_DIR / 'three_rule_v3_true_threshold_overlap_compare.csv'}")
    print(f"  {OUT_DIR / 'three_rule_v3_true_threshold_event_detail.csv'}")


if __name__ == "__main__":
    main()
