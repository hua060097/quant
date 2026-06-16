"""Diagnose ma_convergence_score stability and model threshold sensitivity.

Outputs:
1. Daily cross-sectional dispersion stats.
2. Split-threshold proximity stats for the LightGBM model.
3. 20-bucket label-return monotonicity by daily cross-sectional rank buckets.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main_alpha_二分类_原因子_with_test_metrics_妖股_copy as alpha


DEFAULT_MODEL = Path("results/lightgbm_model_window3_20260427_231053.pkl")
DEFAULT_START = "2025-08-01"
DEFAULT_END = "2026-05-18"


def load_model(model_path: Path):
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)
    model = model_obj["model"]
    feature_names = list(model_obj["feature_names"])
    return model, feature_names, model_obj


def prepare_scored_data(
    model_path: Path,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[str]]:
    model, feature_names, _ = load_model(model_path)
    required_factors = list(dict.fromkeys(feature_names + alpha._get_exhaustion_filter_extra_factors()))
    data = alpha._load_factor_cache_data(
        required_factors=required_factors,
        include_label_generation_features=True,
        start_date=start_date,
        end_date=end_date,
    )
    if data is None or data.empty:
        raise RuntimeError("failed to load factor data")

    valid_cs_mask, st_keep, new_keep = alpha._build_valid_cs_mask(data, alpha.logger)
    data = data.copy()
    data["valid_cs_mask"] = valid_cs_mask.astype(bool)
    del st_keep, new_keep

    labeled = alpha._generate_training_labels_for_all_data(data)
    labeled = labeled[labeled["date"] <= pd.to_datetime(end_date)].copy()

    missing = [c for c in feature_names if c not in labeled.columns]
    if missing:
        raise RuntimeError(f"missing model features: {missing[:20]}")

    X, available_features = alpha._prepare_feature_data(labeled, feature_names, data_type="ma_convergence_score诊断集")
    if X is None:
        raise RuntimeError("failed to prepare features")

    pred = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    scored = labeled.copy()
    scored["predicted_return"] = pred
    scored = scored[scored["atr_return_label"].notna()].copy()
    scored["date"] = pd.to_datetime(scored["date"])
    return scored, available_features


def daily_dispersion_stats(scored: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for date, day in scored.groupby(scored["date"].dt.normalize(), sort=True):
        s = pd.to_numeric(day[factor], errors="coerce").dropna()
        if s.empty:
            continue
        row = {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "q01": float(s.quantile(0.01)),
            "q05": float(s.quantile(0.05)),
            "q25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q75": float(s.quantile(0.75)),
            "q95": float(s.quantile(0.95)),
            "q99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "nunique": int(s.nunique()),
        }
        row["iqr"] = row["q75"] - row["q25"]
        row["range_5_95"] = row["q95"] - row["q05"]
        row["unique_ratio"] = row["nunique"] / max(row["count"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def extract_model_thresholds(model, feature_names: list[str], factor: str) -> list[float]:
    booster = model.booster_ if hasattr(model, "booster_") else model
    dump = booster.dump_model()
    thresholds = []

    def walk(node):
        if not isinstance(node, dict):
            return
        split_feature = node.get("split_feature")
        threshold = node.get("threshold")
        if split_feature is not None and threshold is not None:
            if isinstance(split_feature, int):
                feat_name = feature_names[split_feature] if split_feature < len(feature_names) else None
            else:
                feat_name = str(split_feature)
            if feat_name == factor:
                try:
                    thresholds.append(float(threshold))
                except Exception:
                    pass
        if "left_child" in node:
            walk(node["left_child"])
        if "right_child" in node:
            walk(node["right_child"])

    for tree in dump.get("tree_info", []):
        walk(tree.get("tree_structure", {}))
    thresholds = sorted(set(thresholds))
    return thresholds


def proximity_stats(values: pd.Series, thresholds: Iterable[float], eps_list=(0.001, 0.002, 0.005)) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"count": 0, "threshold_count": 0}
    thr = np.asarray(list(thresholds), dtype=float)
    if len(thr) == 0:
        return {"count": int(len(values)), "threshold_count": 0}

    dist = np.min(np.abs(values[:, None] - thr[None, :]), axis=1)
    out = {
        "count": int(len(values)),
        "threshold_count": int(len(thr)),
        "mean_min_distance": float(np.mean(dist)),
        "median_min_distance": float(np.median(dist)),
        "p10_min_distance": float(np.quantile(dist, 0.10)),
        "p25_min_distance": float(np.quantile(dist, 0.25)),
        "p50_min_distance": float(np.quantile(dist, 0.50)),
        "p75_min_distance": float(np.quantile(dist, 0.75)),
        "p90_min_distance": float(np.quantile(dist, 0.90)),
    }
    for eps in eps_list:
        out[f"within_{eps:.3f}"] = int((dist <= eps).sum())
        out[f"within_{eps:.3f}_ratio"] = float((dist <= eps).mean())
    return out


def qcut_buckets_by_day(scored: pd.DataFrame, factor: str, label_col: str = "atr_return_label") -> pd.DataFrame:
    rows = []
    for date, day in scored.groupby(scored["date"].dt.normalize(), sort=True):
        sub = day[[factor, label_col]].copy()
        sub[factor] = pd.to_numeric(sub[factor], errors="coerce")
        sub[label_col] = pd.to_numeric(sub[label_col], errors="coerce")
        sub = sub.dropna(subset=[factor, label_col]).copy()
        if len(sub) < 20:
            continue

        try:
            ranks = sub[factor].rank(method="first")
            sub["bucket"] = pd.qcut(ranks, 20, labels=False, duplicates="drop")
        except Exception:
            rank_pct = sub[factor].rank(method="first", pct=True)
            sub["bucket"] = np.minimum((rank_pct * 20).astype(int), 19)

        for bucket, g in sub.groupby("bucket", sort=True):
            y = g[label_col]
            rows.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "bucket": int(bucket),
                    "count": int(len(g)),
                    "mean_return": float(y.mean()),
                    "win_rate": float((y > 0).mean()),
                    "big_loss_rate": float((y <= -0.04).mean()),
                    "big_gain_rate": float((y >= 0.08).mean()),
                    "median_return": float(y.median()),
                }
            )

    return pd.DataFrame(rows)


def selection_threshold_proximity(
    scored: pd.DataFrame,
    model,
    feature_names: list[str],
    factor: str,
    topn: int = 5,
) -> pd.DataFrame:
    thresholds = extract_model_thresholds(model, feature_names, factor)
    rows = []
    for date, day in scored.groupby(scored["date"].dt.normalize(), sort=True):
        day = day.loc[day["atr_return_label"].notna()].copy()
        day = day[(day.get("is_limit_up", 0) != 1) & (day.get("is_limit_down", 0) != 1) & (day.get("is_zhaban", 0) != 1)].copy()
        if len(day) < topn:
            continue
        day = day.sort_values("predicted_return", ascending=False).reset_index(drop=True)
        top = day.head(topn).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        top[f"{factor}_min_distance"] = top[factor].apply(
            lambda x: float(np.min(np.abs(np.asarray(thresholds, dtype=float) - float(x)))) if len(thresholds) else np.nan
        )
        top["date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
        rows.append(top[["date", "rank", "stock_code", factor, "predicted_return", "atr_return_label", f"{factor}_min_distance"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_daily_dispersion(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {"days": 0}
    return {
        "days": int(len(daily)),
        "count_mean": float(daily["count"].mean()),
        "iqr_mean": float(daily["iqr"].mean()),
        "iqr_median": float(daily["iqr"].median()),
        "iqr_p10": float(daily["iqr"].quantile(0.10)),
        "iqr_p90": float(daily["iqr"].quantile(0.90)),
        "range_5_95_mean": float(daily["range_5_95"].mean()),
        "unique_ratio_mean": float(daily["unique_ratio"].mean()),
        "unique_ratio_median": float(daily["unique_ratio"].median()),
        "days_iqr_lt_0.05": int((daily["iqr"] < 0.05).sum()),
        "days_iqr_lt_0.10": int((daily["iqr"] < 0.10).sum()),
        "days_range_5_95_lt_0.20": int((daily["range_5_95"] < 0.20).sum()),
        "days_unique_ratio_gt_0.9": int((daily["unique_ratio"] > 0.9).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--start_date", default=DEFAULT_START)
    parser.add_argument("--end_date", default=DEFAULT_END)
    parser.add_argument("--factor", default="ma_convergence_score")
    args = parser.parse_args()

    model_path = Path(args.model)
    scored, feature_names = prepare_scored_data(model_path, args.start_date, args.end_date)

    factor = args.factor
    if factor not in scored.columns:
        raise RuntimeError(f"factor not found: {factor}")

    model, _, _ = load_model(model_path)
    thresholds = extract_model_thresholds(model, feature_names, factor)

    daily = daily_dispersion_stats(scored, factor)
    daily_summary = summarize_daily_dispersion(daily)

    prox_all = proximity_stats(scored[factor], thresholds)
    top1_top5 = selection_threshold_proximity(scored, model, feature_names, factor, topn=5)
    if not top1_top5.empty:
        prox_top1 = proximity_stats(top1_top5.loc[top1_top5["rank"] == 1, factor], thresholds)
        prox_top5 = proximity_stats(top1_top5.loc[top1_top5["rank"] <= 5, factor], thresholds)
    else:
        prox_top1 = {"count": 0, "threshold_count": len(thresholds)}
        prox_top5 = {"count": 0, "threshold_count": len(thresholds)}

    buckets = qcut_buckets_by_day(scored, factor)
    bucket_summary = (
        buckets.groupby("bucket", as_index=False)
        .agg(
            count=("count", "sum"),
            mean_return=("mean_return", "mean"),
            win_rate=("win_rate", "mean"),
            big_loss_rate=("big_loss_rate", "mean"),
            big_gain_rate=("big_gain_rate", "mean"),
            median_return=("median_return", "mean"),
        )
        .sort_values("bucket")
        .reset_index(drop=True)
    )

    out_prefix = Path("results") / f"ma_convergence_score_diagnosis_{model_path.stem}"
    daily.to_csv(f"{out_prefix}_daily_dispersion.csv", index=False, encoding="utf-8-sig")
    buckets.to_csv(f"{out_prefix}_bucket_daily.csv", index=False, encoding="utf-8-sig")
    bucket_summary.to_csv(f"{out_prefix}_bucket_summary.csv", index=False, encoding="utf-8-sig")
    top1_top5.to_csv(f"{out_prefix}_top1_top5_threshold_distance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "factor": factor,
        "model": str(model_path),
        "date_range": [args.start_date, args.end_date],
        "threshold_count": len(thresholds),
        "daily_summary": daily_summary,
        "thresholds_sample": thresholds[:50],
        "proximity_all": prox_all,
        "proximity_top1": prox_top1,
        "proximity_top5": prox_top5,
        "bucket_summary": bucket_summary.to_dict("records"),
    }
    summary_path = f"{out_prefix}_summary.json"
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DAILY SUMMARY")
    print(json.dumps(daily_summary, ensure_ascii=False, indent=2))
    print("\nTHRESHOLD COUNT:", len(thresholds))
    print("PROXIMITY ALL")
    print(json.dumps(prox_all, ensure_ascii=False, indent=2))
    print("\nPROXIMITY TOP1")
    print(json.dumps(prox_top1, ensure_ascii=False, indent=2))
    print("\nPROXIMITY TOP5")
    print(json.dumps(prox_top5, ensure_ascii=False, indent=2))
    print("\nBUCKET SUMMARY")
    print(bucket_summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
