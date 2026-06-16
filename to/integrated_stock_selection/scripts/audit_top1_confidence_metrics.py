"""Audit causal Top1 confidence metrics on the basic tradeable stock pool."""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
MAIN_PATH = ROOT / "main_alpha_二分类_原因子_with_test_metrics_妖股_copy.py"
MODEL_PATH = RESULTS / "lightgbm_model_window3_20260427_231053.pkl"
EVENT_PATH = RESULTS / "three_rule_v3_185_218_round_event_audit.csv"
OUT_DAILY = RESULTS / "top1_confidence_basic_tradeable_daily.csv"
OUT_THRESHOLDS = RESULTS / "top1_confidence_basic_tradeable_thresholds.csv"
OUT_SEGMENTS = RESULTS / "top1_confidence_basic_tradeable_segments.csv"
OUT_EVENTS = RESULTS / "top1_confidence_basic_tradeable_events.csv"

START = "20250801"
END = "20260611"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def causal_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def last_rank(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return np.nan
        return float((values <= values[-1]).mean())

    return series.rolling(window, min_periods=min_periods).apply(last_rank, raw=True)


def outcome_stats(frame: pd.DataFrame, mask: pd.Series) -> dict:
    mask = mask.fillna(False)
    ret = numeric(frame, "top1_return")
    return {
        "hits": int(mask.sum()),
        "big_loss": int((mask & ret.le(-0.04)).sum()),
        "small_loss": int((mask & ret.gt(-0.04) & ret.lt(0)).sum()),
        "gain_0_3": int((mask & ret.ge(0) & ret.lt(0.03)).sum()),
        "gain_3_8": int((mask & ret.ge(0.03) & ret.lt(0.08)).sum()),
        "big_gain_8": int((mask & ret.ge(0.08)).sum()),
        "big_gain_10": int((mask & ret.ge(0.10)).sum()),
        "mean_return": float(ret[mask].mean()) if mask.any() else np.nan,
        "median_return": float(ret[mask].median()) if mask.any() else np.nan,
    }


def build_daily_panel(main_module) -> pd.DataFrame:
    with MODEL_PATH.open("rb") as file:
        model_data = pickle.load(file)
    model = model_data["model"]
    features = list(model_data["feature_names"])

    data = main_module._load_factor_cache_data(
        required_factors=features,
        include_label_generation_features=True,
        start_date=START,
        end_date=(pd.to_datetime(END) + pd.Timedelta(days=14)).strftime("%Y%m%d"),
    ).copy()
    data["date"] = pd.to_datetime(data["date"])
    data["stock_code"] = data["stock_code"].astype(str)
    labeled = main_module._generate_training_labels_for_all_data(data.copy())
    if isinstance(labeled, tuple):
        labeled = labeled[0]
    labeled["date"] = pd.to_datetime(labeled["date"])
    labeled = labeled[
        labeled["date"].between(pd.to_datetime(START), pd.to_datetime(END))
        & labeled["atr_return_label"].notna()
    ].copy()

    tradeable = pd.Series(True, index=labeled.index)
    for column in ("is_limit_up", "is_limit_down", "is_zhaban"):
        if column in labeled.columns:
            tradeable &= numeric(labeled, column).fillna(0).ne(1)
    if "amount" in labeled.columns:
        tradeable &= numeric(labeled, "amount").ge(10_000_000)
    labeled = labeled.loc[tradeable].copy()

    x_model = pd.DataFrame(index=labeled.index)
    for feature in features:
        x_model[feature] = numeric(labeled, feature) if feature in labeled.columns else 0.0
    x_model = x_model.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    labeled["pred"] = np.asarray(main_module._predict_lightgbm_model(model, x_model), dtype=float).reshape(-1)

    rows = []
    for date, day in labeled.groupby("date", sort=True):
        day = day.sort_values("pred", ascending=False).reset_index(drop=True)
        if len(day) < 20:
            continue
        pred = day["pred"].to_numpy(dtype=float)
        p1, p2, p3, p5 = pred[0], pred[1], pred[2], pred[4]
        p25 = pred[1:5]
        p210 = pred[1:10]
        p20 = pred[:20]
        pool_mean = float(np.nanmean(pred))
        pool_std = float(np.nanstd(pred, ddof=0))
        pool_median = float(np.nanmedian(pred))
        mad = float(np.nanmedian(np.abs(pred - pool_median))) * 1.4826
        p25_std = float(np.nanstd(p25, ddof=0))
        p210_std = float(np.nanstd(p210, ddof=0))
        p20_std = float(np.nanstd(p20, ddof=0))
        adjacent_gaps_2_20 = pred[1:19] - pred[2:20]
        adjacent_gap_median = float(np.nanmedian(adjacent_gaps_2_20))
        adjacent_gap_mean = float(np.nanmean(adjacent_gaps_2_20))
        top20_range = float(p1 - pred[19])
        standardized_top20 = (p20 - float(np.mean(p20))) / max(p20_std, 1e-12)
        softmax_top20 = np.exp(standardized_top20 - np.max(standardized_top20))
        softmax_top20 /= softmax_top20.sum()
        top20_entropy = float(-(softmax_top20 * np.log(softmax_top20 + 1e-12)).sum())
        top1 = day.iloc[0]
        rows.append({
            "date": date,
            "top1_code": top1["stock_code"],
            "top1_return": float(top1["atr_return_label"]),
            "pool_size": len(day),
            "top1_pred": p1,
            "score_z_pool": (p1 - pool_mean) / pool_std if pool_std > 0 else np.nan,
            "score_robust_z_pool": (p1 - pool_median) / mad if mad > 0 else np.nan,
            "gap_1_2": p1 - p2,
            "gap_1_3": p1 - p3,
            "gap_1_5": p1 - p5,
            "gap_1_vs_2_5_mean": p1 - float(np.mean(p25)),
            "z_1_vs_2_5": (p1 - float(np.mean(p25))) / p25_std if p25_std > 0 else np.nan,
            "z_1_vs_2_10": (p1 - float(np.mean(p210))) / p210_std if p210_std > 0 else np.nan,
            "gap_1_2_rel": (p1 - p2) / max(abs(p1), 1e-12),
            "gap_1_vs_2_5_rel": (p1 - float(np.mean(p25))) / max(abs(p1), 1e-12),
            "gap_1_vs_2_5_over_top20_std": (p1 - float(np.mean(p25))) / p20_std if p20_std > 0 else np.nan,
            "gap_1_2_over_top20_std": (p1 - p2) / p20_std if p20_std > 0 else np.nan,
            "gap_1_2_over_adjacent_median": (
                (p1 - p2) / adjacent_gap_median if adjacent_gap_median > 0 else np.nan
            ),
            "gap_1_2_over_adjacent_mean": (
                (p1 - p2) / adjacent_gap_mean if adjacent_gap_mean > 0 else np.nan
            ),
            "top1_share_top20_range": (p1 - p2) / top20_range if top20_range > 0 else np.nan,
            "top1_softmax_share_top20": float(softmax_top20[0]),
            "top20_entropy_neg": -top20_entropy,
            "near_top1_count_001_neg": -int((pred >= p1 - 0.001).sum()),
            "near_top1_count_002_neg": -int((pred >= p1 - 0.002).sum()),
            "near_top1_count_005_neg": -int((pred >= p1 - 0.005).sum()),
            "near_top1_count_010_neg": -int((pred >= p1 - 0.010).sum()),
        })
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    base_metrics = [
        "top1_pred", "score_z_pool", "score_robust_z_pool", "gap_1_2", "gap_1_3",
        "gap_1_5", "gap_1_vs_2_5_mean", "z_1_vs_2_5", "z_1_vs_2_10",
        "gap_1_2_rel", "gap_1_vs_2_5_rel", "gap_1_vs_2_5_over_top20_std",
        "gap_1_2_over_top20_std", "gap_1_2_over_adjacent_median",
        "gap_1_2_over_adjacent_mean", "top1_share_top20_range",
        "top1_softmax_share_top20", "top20_entropy_neg",
        "near_top1_count_001_neg", "near_top1_count_002_neg",
        "near_top1_count_005_neg", "near_top1_count_010_neg",
    ]
    for metric in base_metrics:
        daily[f"{metric}_pct20"] = causal_percentile(daily[metric], 20, 10)
        daily[f"{metric}_pct60"] = causal_percentile(daily[metric], 60, 20)

    # Low values consistently mean weak Top1 confidence.
    daily["head_confidence_pct20"] = daily[
        ["score_z_pool_pct20", "gap_1_2_pct20", "gap_1_vs_2_5_mean_pct20", "z_1_vs_2_5_pct20"]
    ].mean(axis=1)
    daily["head_confidence_pct60"] = daily[
        ["score_z_pool_pct60", "gap_1_2_pct60", "gap_1_vs_2_5_mean_pct60", "z_1_vs_2_5_pct60"]
    ].mean(axis=1)
    daily["head_confidence_min_pct60"] = daily[
        ["score_z_pool_pct60", "gap_1_2_pct60", "gap_1_vs_2_5_mean_pct60", "z_1_vs_2_5_pct60"]
    ].min(axis=1)
    return daily


def analyze(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = {"date", "top1_code", "top1_return", "pool_size"}
    metrics = [column for column in daily.columns if column not in excluded]
    unique_dates = sorted(daily["date"].unique())
    halves = np.array_split(unique_dates, 2)
    calibration = daily[daily["date"].isin(halves[0])]
    validation = daily[daily["date"].isin(halves[1])]
    segments = np.array_split(unique_dates, 4)
    rows = []
    segment_rows = []

    for metric in metrics:
        cal = numeric(calibration, metric).dropna()
        if len(cal) < 20 or cal.nunique() < 3:
            continue
        for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            threshold = float(cal.quantile(q))
            all_hit = numeric(daily, metric).le(threshold)
            val_hit = numeric(validation, metric).le(threshold)
            record = {
                "metric": metric,
                "calibration_quantile": q,
                "threshold": threshold,
                **{f"all_{key}": value for key, value in outcome_stats(daily, all_hit).items()},
                **{f"validation_{key}": value for key, value in outcome_stats(validation, val_hit).items()},
            }
            rows.append(record)
            for segment_no, dates in enumerate(segments, 1):
                part = daily[daily["date"].isin(dates)]
                hit = numeric(part, metric).le(threshold)
                segment_rows.append({
                    "metric": metric,
                    "calibration_quantile": q,
                    "threshold": threshold,
                    "segment": segment_no,
                    "segment_start": part["date"].min(),
                    "segment_end": part["date"].max(),
                    **outcome_stats(part, hit),
                })
    return pd.DataFrame(rows), pd.DataFrame(segment_rows)


def main() -> None:
    main_module = load_module(MAIN_PATH, "top1_confidence_main")
    daily = build_daily_panel(main_module)
    thresholds, segments = analyze(daily)

    if EVENT_PATH.exists():
        events = pd.read_csv(EVENT_PATH)
        events["factor_date"] = pd.to_datetime(events["factor_date"])
        event_panel = events.merge(
            daily.rename(columns={"date": "factor_date", "top1_code": "stock_code"}),
            on=["factor_date", "stock_code"],
            how="inner",
            suffixes=("", "_daily"),
        )
        event_panel.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")

    daily.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    thresholds.to_csv(OUT_THRESHOLDS, index=False, encoding="utf-8-sig")
    segments.to_csv(OUT_SEGMENTS, index=False, encoding="utf-8-sig")

    candidates = thresholds[
        (thresholds["validation_hits"] >= 3)
        & (thresholds["validation_big_gain_8"] == 0)
    ].copy()
    candidates["validation_loss_capture"] = (
        candidates["validation_big_loss"] + candidates["validation_small_loss"]
    )
    candidates = candidates.sort_values(
        ["validation_big_loss", "validation_loss_capture", "validation_mean_return"],
        ascending=[False, False, True],
    )
    print(f"daily rows: {len(daily)}, dates: {daily['date'].nunique()}")
    print("\nTop no-big-gain validation candidates:")
    print(candidates.head(40)[[
        "metric", "calibration_quantile", "threshold",
        "all_hits", "all_big_loss", "all_small_loss", "all_big_gain_8", "all_mean_return",
        "validation_hits", "validation_big_loss", "validation_small_loss",
        "validation_big_gain_8", "validation_mean_return",
    ]].to_string(index=False))
    print(f"\nSaved: {OUT_DAILY}")
    print(f"Saved: {OUT_THRESHOLDS}")
    print(f"Saved: {OUT_SEGMENTS}")


if __name__ == "__main__":
    main()
