"""Audit Rule9 under every score_z_in_day pool definition used in the repo."""

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
SYSTEMATIC_PATH = ROOT / "scripts" / "audit_three_rule_v3_systematic.py"
MODEL_PATH = RESULTS / "lightgbm_model_window3_20260427_231053.pkl"
EVENT_PATH = RESULTS / "three_rule_v3_185_218_round_event_audit.csv"
OUT_EVENTS = RESULTS / "score_z_r9_multi_pool_events.csv"
OUT_THRESHOLDS = RESULTS / "score_z_r9_multi_pool_thresholds.csv"
OUT_BINS = RESULTS / "score_z_r9_multi_pool_bins.csv"

RULE_FACTORS = [
    "runup_ma5_exhaustion",
    "cpv_factor",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
    "sector_leadership_persistence_v3",
    "return_20d_intra_z",
    "market_liquidity_change",
    "ulcer_index",
    "momentum_peak_decay_3d",
    "intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2",
    "market_advance_decline_ratio",
    "limit_down_3d",
    "top10_10d_strength_level_v3",
    "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d",
    "kdj_j_9",
    "reg_severe_up_10d_ratio_pct",
    "price_ma5_ratio_raw",
    "ma_convergence_score",
    "ma5_pullback_depth_10d",
    "ret_3d_vs_10d_accel",
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def attach_z(pool: pd.DataFrame, mask: pd.Series, name: str) -> pd.DataFrame:
    subset = pool.loc[mask.fillna(False)].copy()
    mean = subset.groupby("date")["predicted_return_rebuilt"].transform("mean")
    std = subset.groupby("date")["predicted_return_rebuilt"].transform(
        lambda values: float(np.nanstd(pd.to_numeric(values, errors="coerce"), ddof=0))
    )
    subset[name] = np.where(
        std > 0,
        (subset["predicted_return_rebuilt"] - mean) / std,
        np.nan,
    )
    return subset[["date", "stock_code", name]]


def stats(frame: pd.DataFrame, mask: pd.Series) -> dict:
    mask = mask.fillna(False)
    ret = numeric(frame, "actual_return")
    return {
        "hits": int(mask.sum()),
        "actual_hits": int((mask & frame["event_type"].eq("actual")).sum()),
        "shadow_hits": int((mask & frame["event_type"].eq("shadow")).sum()),
        "big_loss": int((mask & (ret <= -0.04)).sum()),
        "small_loss": int((mask & (ret > -0.04) & (ret < 0)).sum()),
        "gain_0_3": int((mask & (ret >= 0) & (ret < 0.03)).sum()),
        "gain_3_8": int((mask & (ret >= 0.03) & (ret < 0.08)).sum()),
        "big_gain_8": int((mask & (ret >= 0.08)).sum()),
        "big_gain_10": int((mask & (ret >= 0.10)).sum()),
        "mean_return": float(ret[mask].mean()) if mask.any() else np.nan,
    }


def main() -> None:
    main_module = load_module(MAIN_PATH, "score_z_r9_main")
    systematic = load_module(SYSTEMATIC_PATH, "score_z_r9_systematic")
    events = pd.read_csv(EVENT_PATH)
    events["factor_date"] = pd.to_datetime(events["factor_date"])
    events["stock_code"] = events["stock_code"].astype(str)

    with MODEL_PATH.open("rb") as file:
        model_data = pickle.load(file)
    model = model_data["model"]
    features = list(model_data["feature_names"])

    factor_dates = events["factor_date"].dropna()
    required_factors = list(dict.fromkeys(features + RULE_FACTORS))
    pool = main_module._load_factor_cache_data(
        required_factors=required_factors,
        include_label_generation_features=True,
        start_date=factor_dates.min().strftime("%Y%m%d"),
        end_date=(factor_dates.max() + pd.Timedelta(days=14)).strftime("%Y%m%d"),
    ).copy()
    pool["date"] = pd.to_datetime(pool["date"])
    pool["stock_code"] = pool["stock_code"].astype(str)
    pool = pool[pool["date"].dt.normalize().isin(set(factor_dates.dt.normalize()))].copy()

    x_model = pd.DataFrame(index=pool.index)
    for feature in features:
        x_model[feature] = numeric(pool, feature) if feature in pool.columns else 0.0
    x_model = x_model.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pool["predicted_return_rebuilt"] = np.asarray(
        main_module._predict_lightgbm_model(model, x_model), dtype=float
    ).reshape(-1)

    all_mask = pd.Series(True, index=pool.index)
    limit_mask = pd.Series(True, index=pool.index)
    for column in ("is_limit_up", "is_limit_down", "is_zhaban"):
        if column in pool.columns:
            limit_mask &= numeric(pool, column).fillna(0).ne(1)

    live_mask = limit_mask.copy()
    if "amount" in pool.columns:
        live_mask &= numeric(pool, "amount").ge(10_000_000)
    if "list_date" in pool.columns:
        list_date = pd.to_datetime(pool["list_date"].astype(str), format="%Y%m%d", errors="coerce")
        live_mask &= (pool["date"] - list_date).dt.days.ge(30)
    live_mask &= ~pool["stock_code"].str.endswith(".BJ", na=False)

    pool_rule_masks = systematic.build_current_masks(pool.assign(score_z_in_day=np.nan))
    pool_other_rule_hit = pool_rule_masks.drop(columns="r9").any(axis=1)
    backtest_after_rules_mask = limit_mask & ~pool_other_rule_hit
    live_after_rules_mask = live_mask & ~pool_other_rule_hit

    score_frames = [
        attach_z(pool, all_mask, "score_z_all"),
        attach_z(pool, limit_mask, "score_z_backtest_tradeable"),
        attach_z(pool, live_mask, "score_z_live_tradeable"),
        attach_z(pool, backtest_after_rules_mask, "score_z_backtest_after_other_rules"),
        attach_z(pool, live_after_rules_mask, "score_z_live_after_other_rules"),
    ]
    score_snapshot = score_frames[0]
    for frame in score_frames[1:]:
        score_snapshot = score_snapshot.merge(frame, on=["date", "stock_code"], how="outer")

    events = events.merge(
        score_snapshot.rename(columns={"date": "factor_date"}),
        on=["factor_date", "stock_code"],
        how="left",
    )

    # Historical post-rule score is retained only to demonstrate its circular pool definition.
    old_parts = []
    for kind, filename in (
        ("actual", "score_z_in_day_rule_audit_actual_trades_attached.csv"),
        ("shadow", "score_z_in_day_rule_audit_shadow_attached.csv"),
    ):
        path = RESULTS / filename
        if not path.exists():
            continue
        old = pd.read_csv(path)
        old["event_type"] = kind
        old["buy_date"] = pd.to_datetime(old["buy_datetime"]).dt.strftime("%Y-%m-%d")
        old_parts.append(
            old[["event_type", "stock_code", "buy_date", "score_z_after", "score_z_base"]]
        )
    if old_parts:
        old_scores = pd.concat(old_parts, ignore_index=True).drop_duplicates(
            ["event_type", "stock_code", "buy_date"], keep="last"
        )
        events = events.merge(old_scores, on=["event_type", "stock_code", "buy_date"], how="left")

    base_masks = systematic.build_current_masks(events.assign(score_z_in_day=np.nan))
    other_union = base_masks.drop(columns="r9").any(axis=1)
    thresholds = np.round(np.arange(1.30, 2.31, 0.05), 2)
    score_columns = [
        "score_z_all",
        "score_z_backtest_tradeable",
        "score_z_live_tradeable",
        "score_z_backtest_after_other_rules",
        "score_z_live_after_other_rules",
        "score_z_base",
        "score_z_after",
    ]

    rows = []
    for score_column in score_columns:
        if score_column not in events.columns:
            continue
        score = numeric(events, score_column)
        for threshold in thresholds:
            standalone = score.le(threshold)
            incremental = standalone & ~other_union
            overlap = standalone & other_union
            rows.append({
                "score_definition": score_column,
                "threshold": threshold,
                "coverage": int(score.notna().sum()),
                **{f"standalone_{key}": value for key, value in stats(events, standalone).items()},
                **{f"incremental_{key}": value for key, value in stats(events, incremental).items()},
                **{f"overlap_{key}": value for key, value in stats(events, overlap).items()},
            })
    threshold_result = pd.DataFrame(rows)

    bin_rows = []
    for score_column in score_columns[:5]:
        valid = events.loc[numeric(events, score_column).notna()].copy()
        if valid.empty:
            continue
        valid["score_bin"] = pd.cut(
            numeric(valid, score_column),
            bins=[-np.inf, 1.5, 1.75, 2.0, 2.25, 2.5, np.inf],
            right=True,
        )
        for score_bin, group in valid.groupby("score_bin", observed=True):
            record = stats(group, pd.Series(True, index=group.index))
            record.update({
                "score_definition": score_column,
                "score_bin": str(score_bin),
                "score_min": float(numeric(group, score_column).min()),
                "score_max": float(numeric(group, score_column).max()),
            })
            bin_rows.append(record)

    events.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")
    threshold_result.to_csv(OUT_THRESHOLDS, index=False, encoding="utf-8-sig")
    pd.DataFrame(bin_rows).to_csv(OUT_BINS, index=False, encoding="utf-8-sig")

    interesting = threshold_result[
        threshold_result["threshold"].isin([1.50, 1.60, 1.70, 1.75, 1.80, 1.90, 2.00])
    ]
    print(interesting[
        [
            "score_definition", "threshold", "coverage",
            "standalone_hits", "standalone_big_loss", "standalone_small_loss",
            "standalone_big_gain_8", "standalone_mean_return",
            "incremental_hits", "incremental_big_loss", "incremental_big_gain_8",
        ]
    ].to_string(index=False))
    print(f"\nSaved: {OUT_EVENTS}")
    print(f"Saved: {OUT_THRESHOLDS}")
    print(f"Saved: {OUT_BINS}")


if __name__ == "__main__":
    main()
