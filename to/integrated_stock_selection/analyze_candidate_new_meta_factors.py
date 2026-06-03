import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import main_alpha_二分类_原因子_with_test_metrics_妖股_copy as alpha


DEFAULT_MODEL = "results/lightgbm_model_window3_20260427_231053.pkl"
DEFAULT_START = "2025-08-01"
DEFAULT_END = "2026-05-22"
DEFAULT_ANALYSIS_END = "2026-05-18"


EXTRA_FACTOR_COLS = [
    "kdj_j_9",
    "kdj_k_9",
    "kdj_diff_9",
    "high_to_close_drawdown",
    "high_to_close_drawdown_pct",
    "distance_from_52w_high",
    "distance_from_52w_low",
    "max_rally_10d",
    "max_rally_20d",
    "return_20d_intra_z",
    "momentum_peak_decay_3d",
    "money_flow_pressure",
    "money_flow_pressure_pct",
    "top10_10d_strength_level_v3",
    "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d",
    "prev_first_limit_up_performance_percentile_20d",
    "limit_sustain_alpha",
    "ulcer_index",
    "market_liquidity_change",
    "market_advance_decline_ratio",
    "mkt_dispersion_shock_v2",
]


def _rolling_percentile(s: pd.Series, window: int = 20) -> pd.Series:
    out = []
    vals = []
    for x in s:
        hist = [v for v in vals[-window:] if pd.notna(v)]
        if len(hist) == 0 or pd.isna(x):
            out.append(np.nan)
        else:
            out.append(float((np.asarray(hist) <= x).mean()))
        vals.append(x)
    return pd.Series(out, index=s.index)


def _safe_auc(y: pd.Series, x: pd.Series) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        mask = y.notna() & x.notna()
        if mask.sum() < 10 or y[mask].nunique() < 2:
            return np.nan
        return float(roc_auc_score(y[mask].astype(int), x[mask].astype(float)))
    except Exception:
        return np.nan


def _one_side_scan(df: pd.DataFrame, factor: str, side: str, q: float) -> dict:
    x = df[factor]
    if x.notna().sum() < 20 or x.nunique(dropna=True) < 2:
        return {}
    thr = x.quantile(q if side == "low" else 1 - q)
    hit = x <= thr if side == "low" else x >= thr
    hit = hit & x.notna()
    sub = df[hit]
    keep = df[~hit]
    return {
        "factor": factor,
        "side": side,
        "q": q,
        "threshold": float(thr),
        "hit_n": int(hit.sum()),
        "hit_big_loss": int((sub["top1_return"] <= -0.04).sum()),
        "hit_big_gain": int((sub["top1_return"] >= 0.08).sum()),
        "hit_small_gain_3_8": int(((sub["top1_return"] >= 0.03) & (sub["top1_return"] < 0.08)).sum()),
        "hit_mean_return": float(sub["top1_return"].mean()) if len(sub) else np.nan,
        "keep_mean_return": float(keep["top1_return"].mean()) if len(keep) else np.nan,
        "keep_big_loss": int((keep["top1_return"] <= -0.04).sum()),
        "keep_big_gain": int((keep["top1_return"] >= 0.08).sum()),
    }


def _prepare_scored_data(args) -> tuple[pd.DataFrame, list[str]]:
    model_obj = joblib.load(args.model)
    model = model_obj["model"]
    feature_names = list(model_obj["feature_names"])
    required_factors = list(
        dict.fromkeys(feature_names + alpha._get_exhaustion_filter_extra_factors() + EXTRA_FACTOR_COLS)
    )

    data = alpha._load_factor_cache_data(
        required_factors=required_factors,
        include_label_generation_features=True,
        start_date=args.start,
        end_date=args.end,
    )
    if data is None or data.empty:
        raise RuntimeError("failed to load factor data")

    valid_cs_mask, st_keep, new_keep = alpha._build_valid_cs_mask(data, logging.getLogger(__name__))
    data["valid_cs_mask"] = valid_cs_mask.astype(bool)
    del st_keep, new_keep

    labeled = alpha._generate_training_labels_for_all_data(data)
    labeled = labeled[labeled["date"] <= pd.to_datetime(args.analysis_end)].copy()

    missing = [c for c in feature_names if c not in labeled.columns]
    if missing:
        raise RuntimeError(f"missing model features: {missing[:20]}")

    X, _ = alpha._prepare_feature_data(labeled, feature_names, data_type="新增候选因子分析集")
    if X is None:
        raise RuntimeError("failed to prepare features")

    scored = labeled.copy()
    scored["predicted_return"] = model.predict(X, num_iteration=model.best_iteration)
    return scored, feature_names


def _daily_top1_candidates(scored: pd.DataFrame, apply_rule_filter: bool) -> pd.DataFrame:
    sw_mapping = alpha.load_sw_sector_mapping() or {}
    rows = []
    for date, day in scored.groupby("date", sort=True):
        day = day.loc[day["atr_return_label"].notna()].copy()
        day = day[
            (day.get("is_limit_up", 0) != 1)
            & (day.get("is_limit_down", 0) != 1)
            & (day.get("is_zhaban", 0) != 1)
        ].copy()

        if apply_rule_filter:
            keep = []
            for _, row in day.iterrows():
                hit, _ = alpha._is_exhaustion_filtered_row(row)
                keep.append(not hit)
            day = day.loc[keep].copy()

        if len(day) < 20:
            continue

        day = day.sort_values("predicted_return", ascending=False).reset_index(drop=True)
        top = day.head(20).copy()
        top5 = top.head(5)
        top1 = top.iloc[0]

        pred = top["predicted_return"].astype(float).to_numpy()
        pred5 = pred[:5]
        pred20 = pred[:20]
        p1 = float(pred5[0])
        p2 = float(pred5[1])
        p5 = float(pred5[4])
        mean25 = float(np.mean(pred5[1:5]))
        std25 = float(np.std(pred5[1:5], ddof=0))
        day_pred = day["predicted_return"].astype(float)

        sectors = [sw_mapping.get(code, "UNKNOWN") for code in top5["stock_code"]]
        top5_sector_share = max(pd.Series(sectors).value_counts(normalize=True).max(), 0)
        sectors10 = [sw_mapping.get(code, "UNKNOWN") for code in top.head(10)["stock_code"]]
        sector10_share = pd.Series(sectors10).value_counts(normalize=True)
        top10_sector_hhi = float((sector10_share ** 2).sum())

        row = {
            "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
            "top1_code": top1["stock_code"],
            "top1_return": float(top1["atr_return_label"]),
            "top1_pred": p1,
            "score_z_in_day": float((p1 - day_pred.mean()) / (day_pred.std(ddof=0) + 1e-12)),
            "score_gap_12": p1 - p2,
            "score_gap_1_vs_25": p1 - mean25,
            "score_gap_1_vs_25_z": (p1 - mean25) / (std25 + 1e-12),
            "score_top5_range": p1 - p5,
            "score_top5_std": float(np.std(pred5, ddof=0)),
            "score_top20_std": float(np.std(pred20, ddof=0)),
            "score_near_top1_count_005": int((day_pred >= p1 - 0.005).sum()),
            "score_high_breadth_045": float((day_pred >= 0.45).mean()),
            "top5_same_sector_share": float(top5_sector_share),
            "top10_sector_hhi": top10_sector_hhi,
        }

        for c in EXTRA_FACTOR_COLS:
            if c in top1.index:
                row[c] = top1.get(c, np.nan)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if df.empty:
        return df

    df["score_pct20"] = _rolling_percentile(df["top1_pred"], 20)
    df["score_gap_12_pct20"] = _rolling_percentile(df["score_gap_12"], 20)
    df["score_gap_1_vs_25_pct20"] = _rolling_percentile(df["score_gap_1_vs_25"], 20)
    df["score_crowded_pct20"] = _rolling_percentile(df["score_near_top1_count_005"], 20)

    kdj = df.get("kdj_j_9", pd.Series(np.nan, index=df.index))
    h2c = df.get("high_to_close_drawdown_pct", df.get("high_to_close_drawdown", pd.Series(np.nan, index=df.index)))
    non_first = df.get("prev_non_first_limit_up_performance_percentile_20d", pd.Series(np.nan, index=df.index))
    mom_peak = df.get("momentum_peak_decay_3d", pd.Series(np.nan, index=df.index))
    mfp = df.get("money_flow_pressure_pct", df.get("money_flow_pressure", pd.Series(np.nan, index=df.index)))
    top10 = df.get("top10_10d_strength_level_v3", pd.Series(np.nan, index=df.index))
    buy_sm = df.get("buy_sm_amount_rate", pd.Series(np.nan, index=df.index))
    ulcer = df.get("ulcer_index", pd.Series(np.nan, index=df.index))

    # 候选新增“元因子”：只用现有日线、资金流、板块和涨停环境，不使用竞价。
    df["score_overheat_weak_kdj"] = df["score_pct20"] * np.maximum(0.0, 0.75 - kdj)
    df["score_overheat_high_drawdown"] = df["score_pct20"] * h2c
    df["score_crowding_weak_gap"] = df["score_crowded_pct20"] * (1.0 - df["score_gap_1_vs_25_pct20"])
    df["sector_crowding_score"] = df["top5_same_sector_share"] * df["top10_sector_hhi"]
    df["hot_env_weak_kdj"] = non_first * np.maximum(0.0, 0.75 - kdj)
    df["hot_env_mom_decay"] = non_first * mom_peak
    df["hot_env_money_pressure"] = non_first * mfp
    df["strong_stock_retail_outflow"] = top10 * np.maximum(0.0, -buy_sm)
    df["overheat_ulcer_combo"] = df["score_pct20"] * ulcer

    return df


def _evaluate(df: pd.DataFrame, name: str, out_prefix: str) -> dict:
    candidate_cols = [
        c
        for c in df.columns
        if c not in {"date", "top1_code", "top1_return"}
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() >= 40
        and df[c].nunique(dropna=True) >= 5
    ]
    rows = []
    scans = []
    y_loss = (df["top1_return"] <= -0.04).astype(int)
    y_gain = (df["top1_return"] >= 0.08).astype(int)

    for c in candidate_cols:
        x = df[c]
        rows.append(
            {
                "factor": c,
                "n": int(x.notna().sum()),
                "spearman_ret": float(x.corr(df["top1_return"], method="spearman")),
                "pearson_ret": float(x.corr(df["top1_return"], method="pearson")),
                "auc_big_loss_high_is_risk": _safe_auc(y_loss, x),
                "auc_big_gain_high_is_good": _safe_auc(y_gain, x),
                "q10": float(x.quantile(0.10)),
                "q50": float(x.quantile(0.50)),
                "q90": float(x.quantile(0.90)),
            }
        )
        for side in ["low", "high"]:
            for q in [0.10, 0.20, 0.30]:
                rec = _one_side_scan(df, c, side, q)
                if rec:
                    scans.append(rec)

    score = pd.DataFrame(rows)
    scan_df = pd.DataFrame(scans)
    if not scan_df.empty:
        scan_df["loss_gain_ratio"] = scan_df["hit_big_loss"] / (scan_df["hit_big_gain"] + 1)
        scan_df["net_loss_minus_gain"] = scan_df["hit_big_loss"] - scan_df["hit_big_gain"]
        scan_df["mean_lift"] = scan_df["keep_mean_return"] - df["top1_return"].mean()
        scan_df = scan_df.sort_values(
            ["net_loss_minus_gain", "loss_gain_ratio", "mean_lift"],
            ascending=[False, False, False],
        )

    out_base = Path(f"{out_prefix}_{name}")
    df.to_csv(f"{out_base}_daily.csv", index=False, encoding="utf-8-sig")
    score.sort_values("auc_big_loss_high_is_risk", ascending=False).to_csv(
        f"{out_base}_factor_scores.csv", index=False, encoding="utf-8-sig"
    )
    scan_df.to_csv(f"{out_base}_threshold_scans.csv", index=False, encoding="utf-8-sig")

    return {
        "name": name,
        "n": int(len(df)),
        "mean_return": float(df["top1_return"].mean()),
        "big_loss_<=-4": int((df["top1_return"] <= -0.04).sum()),
        "big_gain_>=8": int((df["top1_return"] >= 0.08).sum()),
        "top_loss_auc": score.sort_values("auc_big_loss_high_is_risk", ascending=False)
        .head(12)
        .to_dict("records"),
        "top_threshold_scans": scan_df.head(20).to_dict("records") if not scan_df.empty else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--analysis-end", default=DEFAULT_ANALYSIS_END)
    parser.add_argument("--out-prefix", default="results/candidate_new_meta_factors")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
    scored, _ = _prepare_scored_data(args)

    outputs = {}
    for name, apply_rule in [("base_tradable_no_rule", False), ("after_three_rule_v3", True)]:
        daily = _daily_top1_candidates(scored, apply_rule_filter=apply_rule)
        outputs[name] = _evaluate(daily, name, args.out_prefix)

    out_path = Path(f"{args.out_prefix}_summary.json")
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
