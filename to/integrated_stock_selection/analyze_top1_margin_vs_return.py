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


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(x[mask].corr(y[mask], method=method))


def _bucket_summary(df: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    work = df[[col, "top1_return"]].dropna().copy()
    if work.empty or work[col].nunique() < 2:
        return pd.DataFrame()

    labels = [f"Q{i}" for i in range(1, q + 1)]
    work["bucket"] = pd.qcut(work[col], q=min(q, work[col].nunique()), labels=labels[: min(q, work[col].nunique())], duplicates="drop")

    def agg(g: pd.DataFrame) -> pd.Series:
        r = g["top1_return"]
        return pd.Series(
            {
                "n": len(g),
                f"{col}_min": g[col].min(),
                f"{col}_max": g[col].max(),
                "mean_ret": r.mean(),
                "median_ret": r.median(),
                "big_loss_<=-4": int((r <= -0.04).sum()),
                "mid_loss_-4~-1": int(((r > -0.04) & (r < -0.01)).sum()),
                "flat_-1~3": int(((r >= -0.01) & (r < 0.03)).sum()),
                "small_gain_3~8": int(((r >= 0.03) & (r < 0.08)).sum()),
                "big_gain_>=8": int((r >= 0.08).sum()),
            }
        )

    return work.groupby("bucket", observed=True).apply(agg).reset_index()


def _make_daily_margin_rows(scored: pd.DataFrame, apply_rule_filter: bool) -> pd.DataFrame:
    rows = []
    for date, day in scored.groupby("date", sort=True):
        day = day.loc[day["atr_return_label"].notna()].copy()
        day = day[
            (day.get("is_limit_up", 0) != 1)
            & (day.get("is_limit_down", 0) != 1)
            & (day.get("is_zhaban", 0) != 1)
        ].copy()
        if apply_rule_filter:
            keep_mask = []
            for _, row in day.iterrows():
                hit, _ = alpha._is_exhaustion_filtered_row(row)
                keep_mask.append(not hit)
            day = day.loc[keep_mask].copy()

        if len(day) < 5:
            continue

        top = day.sort_values("predicted_return", ascending=False).head(5).reset_index(drop=True)
        pred = top["predicted_return"].astype(float).to_numpy()
        ret = top["atr_return_label"].astype(float).to_numpy()

        top2_5_pred = pred[1:5]
        top2_5_std = float(np.std(top2_5_pred, ddof=0))
        rows.append(
            {
                "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "top1_code": top.loc[0, "stock_code"],
                "top1_pred": pred[0],
                "top2_pred": pred[1],
                "top5_pred": pred[4],
                "margin_12": pred[0] - pred[1],
                "margin_1_vs_25": pred[0] - float(np.mean(top2_5_pred)),
                "margin_12_rel_top1": (pred[0] - pred[1]) / (abs(pred[0]) + 1e-12),
                "margin_1_vs_25_z": (pred[0] - float(np.mean(top2_5_pred))) / (top2_5_std + 1e-12),
                "top1_return": ret[0],
                "top2_return": ret[1],
                "top2_5_mean_return": float(np.mean(ret[1:5])),
                "top1_best_among_top5": bool(ret[0] >= np.max(ret[1:5])),
            }
        )
    return pd.DataFrame(rows)


def _summarize(name: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"name": name, "n": 0}

    summary = {
        "name": name,
        "n": int(len(df)),
        "date_start": df["date"].min(),
        "date_end": df["date"].max(),
        "top1_return_mean": float(df["top1_return"].mean()),
        "top1_return_median": float(df["top1_return"].median()),
        "top1_win_rate": float((df["top1_return"] > 0).mean()),
        "big_loss_<=-4": int((df["top1_return"] <= -0.04).sum()),
        "small_or_mid_loss_-4~-1": int(((df["top1_return"] > -0.04) & (df["top1_return"] < -0.01)).sum()),
        "flat_-1~3": int(((df["top1_return"] >= -0.01) & (df["top1_return"] < 0.03)).sum()),
        "small_gain_3~8": int(((df["top1_return"] >= 0.03) & (df["top1_return"] < 0.08)).sum()),
        "big_gain_>=8": int((df["top1_return"] >= 0.08).sum()),
        "top1_best_among_top5_rate": float(df["top1_best_among_top5"].mean()),
    }

    for col in ["top1_pred", "margin_12", "margin_1_vs_25", "margin_12_rel_top1", "margin_1_vs_25_z"]:
        summary[f"{col}_pearson"] = _safe_corr(df[col], df["top1_return"], "pearson")
        summary[f"{col}_spearman"] = _safe_corr(df[col], df["top1_return"], "spearman")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--analysis-end", default=DEFAULT_ANALYSIS_END)
    parser.add_argument("--out-prefix", default="results/top1_margin_vs_return")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")

    model_obj = joblib.load(args.model)
    model = model_obj["model"]
    feature_names = list(model_obj["feature_names"])
    required_factors = list(dict.fromkeys(feature_names + alpha._get_exhaustion_filter_extra_factors()))

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
    analysis_end = pd.to_datetime(args.analysis_end)
    labeled = labeled[labeled["date"] <= analysis_end].copy()

    missing = [c for c in feature_names if c not in labeled.columns]
    if missing:
        raise RuntimeError(f"missing model features: {missing[:20]}")

    X, _ = alpha._prepare_feature_data(labeled, feature_names, data_type="margin分析集")
    if X is None:
        raise RuntimeError("failed to prepare features")

    scored = labeled.copy()
    scored["predicted_return"] = model.predict(X, num_iteration=model.best_iteration)

    outputs = {}
    for name, apply_rule in [("base_tradable_no_rule", False), ("after_three_rule_v3", True)]:
        daily = _make_daily_margin_rows(scored, apply_rule_filter=apply_rule)
        summary = _summarize(name, daily)
        outputs[name] = summary

        daily_path = Path(f"{args.out_prefix}_{name}_daily.csv")
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

        for col in ["margin_12", "margin_1_vs_25", "margin_12_rel_top1", "margin_1_vs_25_z"]:
            bucket = _bucket_summary(daily, col)
            bucket_path = Path(f"{args.out_prefix}_{name}_{col}_buckets.csv")
            bucket.to_csv(bucket_path, index=False, encoding="utf-8-sig")

    summary_path = Path(f"{args.out_prefix}_summary.json")
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
