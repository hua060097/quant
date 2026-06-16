"""Systematic audit of current three_rule_v3 rules and internal OR paths."""

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
EVENT_AUDIT = RESULTS / "three_rule_v3_185_218_round_event_audit.csv"
MODEL_PATH = RESULTS / "lightgbm_model_window3_20260427_231053.pkl"
MAIN_SCRIPT = ROOT / "main_alpha_二分类_原因子_with_test_metrics_妖股_copy.py"

OUT_EVENTS = RESULTS / "three_rule_v3_systematic_events.csv"
OUT_RULES = RESULTS / "three_rule_v3_systematic_rule_quality.csv"
OUT_PAIRS = RESULTS / "three_rule_v3_systematic_pair_overlap.csv"
OUT_PATHS = RESULTS / "three_rule_v3_systematic_path_quality.csv"
OUT_NEIGHBORHOOD = RESULTS / "three_rule_v3_systematic_threshold_neighborhood.csv"

RULE_COLUMNS = [f"r{i}" for i in range(1, 13)]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def outcome_stats(frame: pd.DataFrame, mask: pd.Series) -> dict:
    mask = mask.fillna(False).astype(bool)
    ret = numeric(frame, "actual_return")
    hits = int(mask.sum())
    return {
        "hits": hits,
        "actual_hits": int((mask & frame["event_type"].eq("actual")).sum()),
        "shadow_hits": int((mask & frame["event_type"].eq("shadow")).sum()),
        "big_loss": int((mask & (ret <= -0.04)).sum()),
        "small_loss": int((mask & (ret > -0.04) & (ret < 0)).sum()),
        "gain_0_3": int((mask & (ret >= 0) & (ret < 0.03)).sum()),
        "gain_3_8": int((mask & (ret >= 0.03) & (ret < 0.08)).sum()),
        "big_gain_8": int((mask & (ret >= 0.08)).sum()),
        "big_gain_10": int((mask & (ret >= 0.10)).sum()),
        "mean_return": float(ret[mask].mean()) if hits else np.nan,
        "median_return": float(ret[mask].median()) if hits else np.nan,
        "big_loss_rate": float((ret[mask] <= -0.04).mean()) if hits else np.nan,
        "downside_rate": float((ret[mask] < 0).mean()) if hits else np.nan,
    }


def attach_rebuilt_score_z(events: pd.DataFrame) -> pd.DataFrame:
    """Rebuild score_z_in_day from the exact fixed model and full daily candidate pool."""
    main = load_module(MAIN_SCRIPT, "three_rule_main_for_score_z")
    with MODEL_PATH.open("rb") as file:
        model_data = pickle.load(file)
    model = model_data["model"]
    features = list(model_data["feature_names"])

    factor_dates = pd.to_datetime(events["factor_date"]).dropna()
    start = factor_dates.min().strftime("%Y%m%d")
    end = (factor_dates.max() + pd.Timedelta(days=14)).strftime("%Y%m%d")
    pool = main._load_factor_cache_data(
        required_factors=features,
        include_label_generation_features=True,
        start_date=start,
        end_date=end,
    )
    pool = pool.copy()
    pool["date"] = pd.to_datetime(pool["date"])
    labeled = main._generate_training_labels_for_all_data(pool.copy())
    labeled_pool = labeled[0] if isinstance(labeled, tuple) else labeled
    labeled_pool["date"] = pd.to_datetime(labeled_pool["date"])
    valid_mask = labeled_pool["atr_return_label"].notna()
    if "relevance" in labeled_pool.columns:
        valid_mask &= labeled_pool["relevance"].notna()
    valid_keys = labeled_pool.loc[valid_mask, ["date", "stock_code"]].drop_duplicates()
    required_dates = set(factor_dates.dt.normalize())
    pool = pool[pool["date"].dt.normalize().isin(required_dates)].copy()

    x_model = pd.DataFrame(index=pool.index)
    for feature in features:
        x_model[feature] = numeric(pool, feature) if feature in pool.columns else 0.0
    x_model = x_model.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pool["predicted_return_rebuilt"] = np.asarray(
        main._predict_lightgbm_model(model, x_model), dtype=float
    ).reshape(-1)
    daily_mean = pool.groupby("date")["predicted_return_rebuilt"].transform("mean")
    daily_std = pool.groupby("date")["predicted_return_rebuilt"].transform(
        lambda values: float(np.nanstd(pd.to_numeric(values, errors="coerce"), ddof=0))
    )
    pool["score_z_in_day_full_pool"] = np.where(
        daily_std > 0,
        (pool["predicted_return_rebuilt"] - daily_mean) / daily_std,
        np.nan,
    )

    valid_pool = pool.merge(valid_keys, on=["date", "stock_code"], how="inner")
    valid_daily_mean = valid_pool.groupby("date")["predicted_return_rebuilt"].transform("mean")
    valid_daily_std = valid_pool.groupby("date")["predicted_return_rebuilt"].transform(
        lambda values: float(np.nanstd(pd.to_numeric(values, errors="coerce"), ddof=0))
    )
    valid_pool["score_z_in_day_valid_pool"] = np.where(
        valid_daily_std > 0,
        (valid_pool["predicted_return_rebuilt"] - valid_daily_mean) / valid_daily_std,
        np.nan,
    )
    pool = pool.merge(
        valid_pool[["date", "stock_code", "score_z_in_day_valid_pool"]],
        on=["date", "stock_code"],
        how="left",
    )
    score_snapshot = pool[
        [
            "date",
            "stock_code",
            "predicted_return_rebuilt",
            "score_z_in_day_full_pool",
            "score_z_in_day_valid_pool",
        ]
    ].rename(columns={"date": "factor_date"})
    score_snapshot["stock_code"] = score_snapshot["stock_code"].astype(str)

    result = events.drop(
        columns=[
            "score_z_in_day",
            "predicted_return_rebuilt",
            "score_z_in_day_full_pool",
            "score_z_in_day_valid_pool",
        ],
        errors="ignore",
    ).merge(score_snapshot, on=["factor_date", "stock_code"], how="left")
    result["score_z_in_day"] = result["score_z_in_day_full_pool"]
    return result


def build_current_masks(frame: pd.DataFrame) -> pd.DataFrame:
    z = lambda col: numeric(frame, col)
    masks = pd.DataFrame(index=frame.index)
    masks["r1"] = (z("runup_ma5_exhaustion") > 0.8) & (z("cpv_factor") <= -0.4)
    masks["r2"] = (
        ((z("prev_day_blowoff_risk") >= 2.0) & (z("sector_momentum_accel_rank") <= 0.36))
        | ((z("prev_day_blowoff_risk") > 5.0) & (z("sector_momentum_accel_rank") <= 0.50))
    )
    masks["r3"] = (
        (z("sector_leadership_persistence_v3") <= -0.18)
        & (z("return_20d_intra_z") <= 0.40)
    )
    masks["r4"] = (z("market_liquidity_change") < 0.94) & (
        (z("ulcer_index") < -2.25)
        | (z("momentum_peak_decay_3d") > 3.20)
        | (z("intra_sector_momentum_rank") < 0.67)
    )
    masks["r5"] = (z("mkt_dispersion_shock_v2") < 0.98) & (
        z("market_advance_decline_ratio") < 0.38
    )
    masks["r6"] = z("limit_down_3d") == 1
    masks["r7"] = (z("top10_10d_strength_level_v3") >= 0.80) & (
        z("buy_sm_amount_rate") <= -0.20
    )
    masks["r8"] = (z("prev_non_first_limit_up_performance_percentile_20d") >= 0.80) & (
        z("kdj_j_9") <= 0.75
    )
    masks["r9"] = z("score_z_in_day") <= 1.50
    masks["r10"] = (z("reg_severe_up_10d_ratio_pct") >= 0.98) & (
        z("market_advance_decline_ratio") >= 4.50
    )
    masks["r11"] = (z("price_ma5_ratio_raw") >= 9.0) & (z("ma_convergence_score") >= 0.20)
    masks["r12"] = (z("ma5_pullback_depth_10d") <= -0.05) & (
        z("ret_3d_vs_10d_accel") >= 6.0
    )
    return masks.fillna(False)


def rule_quality(frame: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    ret = numeric(frame, "actual_return")
    base_big_loss_rate = float((ret <= -0.04).mean())
    hit_count = masks.sum(axis=1)
    union = masks.any(axis=1)
    union_stats = outcome_stats(frame, union)
    rows = []

    for rule in RULE_COLUMNS:
        standalone = masks[rule]
        other = masks.drop(columns=rule).any(axis=1)
        exclusive = standalone & ~other
        corroborated = standalone & other
        union_without = other
        stand = outcome_stats(frame, standalone)
        excl = outcome_stats(frame, exclusive)
        corr = outcome_stats(frame, corroborated)
        without = outcome_stats(frame, union_without)
        rows.append({
            "rule": rule,
            **{f"standalone_{key}": value for key, value in stand.items()},
            **{f"exclusive_{key}": value for key, value in excl.items()},
            **{f"corroborated_{key}": value for key, value in corr.items()},
            "big_loss_lift_vs_base": (
                stand["big_loss_rate"] / base_big_loss_rate if base_big_loss_rate > 0 else np.nan
            ),
            "big_loss_overlap_rate": (
                corr["big_loss"] / stand["big_loss"] if stand["big_loss"] else np.nan
            ),
            "positive_overlap_rate": (
                (corr["gain_0_3"] + corr["gain_3_8"] + corr["big_gain_8"])
                / (stand["gain_0_3"] + stand["gain_3_8"] + stand["big_gain_8"])
                if (stand["gain_0_3"] + stand["gain_3_8"] + stand["big_gain_8"]) else np.nan
            ),
            "union_big_loss_lost_if_removed": union_stats["big_loss"] - without["big_loss"],
            "union_small_loss_lost_if_removed": union_stats["small_loss"] - without["small_loss"],
            "union_positive_released_if_removed": (
                union_stats["gain_0_3"] + union_stats["gain_3_8"] + union_stats["big_gain_8"]
                - without["gain_0_3"] - without["gain_3_8"] - without["big_gain_8"]
            ),
            "union_big_gain_released_if_removed": union_stats["big_gain_8"] - without["big_gain_8"],
            "multi_rule_hits": int((standalone & (hit_count >= 2)).sum()),
        })
    return pd.DataFrame(rows)


def pair_overlap(frame: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left_idx, left in enumerate(RULE_COLUMNS):
        for right in RULE_COLUMNS[left_idx + 1:]:
            overlap = masks[left] & masks[right]
            rows.append({"rule_left": left, "rule_right": right, **outcome_stats(frame, overlap)})
    return pd.DataFrame(rows).sort_values(["big_loss", "hits"], ascending=False)


def path_quality(frame: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    z = lambda col: numeric(frame, col)
    paths = {
        "r2_main": (z("prev_day_blowoff_risk") >= 2.0) & (z("sector_momentum_accel_rank") <= 0.36),
        "r2_xr2a": (z("prev_day_blowoff_risk") > 5.0) & (z("sector_momentum_accel_rank") <= 0.50),
        "r4_a_ulcer": (z("market_liquidity_change") < 0.94) & (z("ulcer_index") < -2.25),
        "r4_b_mom_peak": (z("market_liquidity_change") < 0.94) & (z("momentum_peak_decay_3d") > 3.20),
        "r4_c_sector_rank": (z("market_liquidity_change") < 0.94) & (z("intra_sector_momentum_rank") < 0.67),
    }
    siblings = {
        "r2_main": paths["r2_xr2a"],
        "r2_xr2a": paths["r2_main"],
        "r4_a_ulcer": paths["r4_b_mom_peak"] | paths["r4_c_sector_rank"],
        "r4_b_mom_peak": paths["r4_a_ulcer"] | paths["r4_c_sector_rank"],
        "r4_c_sector_rank": paths["r4_a_ulcer"] | paths["r4_b_mom_peak"],
    }
    parent = {name: name.split("_")[0] for name in paths}
    rows = []
    for name, standalone in paths.items():
        other_rules = masks.drop(columns=parent[name]).any(axis=1)
        sibling_overlap = standalone & siblings[name]
        other_rule_overlap = standalone & other_rules
        unique_within_parent = standalone & ~siblings[name]
        lost_from_union = standalone & ~siblings[name] & ~other_rules
        rows.append({
            "path": name,
            **{f"standalone_{key}": value for key, value in outcome_stats(frame, standalone).items()},
            **{f"sibling_overlap_{key}": value for key, value in outcome_stats(frame, sibling_overlap).items()},
            **{f"other_rule_overlap_{key}": value for key, value in outcome_stats(frame, other_rule_overlap).items()},
            **{f"unique_within_parent_{key}": value for key, value in outcome_stats(frame, unique_within_parent).items()},
            **{f"lost_if_removed_{key}": value for key, value in outcome_stats(frame, lost_from_union).items()},
        })
    return pd.DataFrame(rows)


def threshold_candidates(frame: pd.DataFrame) -> list[tuple[str, str, float, float, pd.Series]]:
    z = lambda col: numeric(frame, col)
    candidates: list[tuple[str, str, float, float, pd.Series]] = []

    def add_grid(rule: str, path: str, levels_a, levels_b, build):
        for a in levels_a:
            for b in levels_b:
                candidates.append((rule, path, float(a), float(b), build(a, b).fillna(False)))

    add_grid("r1", "main", [0.6, 0.7, 0.8, 0.9, 1.0], [-0.6, -0.5, -0.4, -0.3, -0.2],
             lambda a, b: (z("runup_ma5_exhaustion") > a) & (z("cpv_factor") <= b))
    add_grid("r2", "main_only", [1.5, 2.0, 2.5, 3.0], [0.25, 0.30, 0.36, 0.42, 0.50],
             lambda a, b: (z("prev_day_blowoff_risk") >= a) & (z("sector_momentum_accel_rank") <= b))
    add_grid("r2", "xr2a_only", [4.0, 5.0, 6.0], [0.40, 0.50, 0.60],
             lambda a, b: (z("prev_day_blowoff_risk") > a) & (z("sector_momentum_accel_rank") <= b))
    add_grid("r3", "main", [-0.30, -0.24, -0.18, -0.12, -0.06], [0.20, 0.30, 0.40, 0.50, 0.60],
             lambda a, b: (z("sector_leadership_persistence_v3") <= a) & (z("return_20d_intra_z") <= b))
    add_grid("r4", "a_ulcer_only", [0.90, 0.94, 0.98, 1.02], [-3.0, -2.5, -2.25, -2.0, -1.5],
             lambda a, b: (z("market_liquidity_change") < a) & (z("ulcer_index") < b))
    add_grid("r4", "b_mom_peak_only", [0.90, 0.94, 0.98, 1.02], [2.0, 2.5, 3.0, 3.2, 3.5, 4.0],
             lambda a, b: (z("market_liquidity_change") < a) & (z("momentum_peak_decay_3d") > b))
    add_grid("r4", "c_sector_rank_only", [0.90, 0.94, 0.98, 1.02], [0.50, 0.60, 0.67, 0.75, 0.85],
             lambda a, b: (z("market_liquidity_change") < a) & (z("intra_sector_momentum_rank") < b))
    add_grid("r5", "main", [0.90, 0.94, 0.98, 1.02], [0.30, 0.34, 0.38, 0.42, 0.46],
             lambda a, b: (z("mkt_dispersion_shock_v2") < a) & (z("market_advance_decline_ratio") < b))
    add_grid("r7", "main", [0.70, 0.75, 0.80, 0.85, 0.90], [-0.40, -0.30, -0.20, -0.10, 0.0],
             lambda a, b: (z("top10_10d_strength_level_v3") >= a) & (z("buy_sm_amount_rate") <= b))
    add_grid("r8", "main", [0.70, 0.75, 0.80, 0.85, 0.90], [0.50, 0.65, 0.75, 0.85, 1.0],
             lambda a, b: (z("prev_non_first_limit_up_performance_percentile_20d") >= a) & (z("kdj_j_9") <= b))
    for threshold in [1.0, 1.25, 1.50, 1.75, 2.0]:
        candidates.append(("r9", "main", float(threshold), np.nan, (z("score_z_in_day") <= threshold).fillna(False)))
    add_grid("r10", "main", [0.95, 0.97, 0.98, 0.99], [3.5, 4.0, 4.5, 5.0, 5.5],
             lambda a, b: (z("reg_severe_up_10d_ratio_pct") >= a) & (z("market_advance_decline_ratio") >= b))
    add_grid("r11", "main", [7.0, 8.0, 9.0, 10.0, 11.0], [0.10, 0.15, 0.20, 0.25, 0.30],
             lambda a, b: (z("price_ma5_ratio_raw") >= a) & (z("ma_convergence_score") >= b))
    add_grid("r12", "main", [-0.04, -0.05, -0.06, -0.07, -0.08], [5.0, 5.5, 6.0, 6.5],
             lambda a, b: (z("ma5_pullback_depth_10d") <= a) & (z("ret_3d_vs_10d_accel") >= b))
    return candidates


def threshold_neighborhood(frame: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    z = lambda col: numeric(frame, col)
    fixed_siblings = {
        ("r2", "main_only"): (z("prev_day_blowoff_risk") > 5.0) & (z("sector_momentum_accel_rank") <= 0.50),
        ("r2", "xr2a_only"): (z("prev_day_blowoff_risk") >= 2.0) & (z("sector_momentum_accel_rank") <= 0.36),
        ("r4", "a_ulcer_only"): (
            (z("market_liquidity_change") < 0.94)
            & ((z("momentum_peak_decay_3d") > 3.20) | (z("intra_sector_momentum_rank") < 0.67))
        ),
        ("r4", "b_mom_peak_only"): (
            (z("market_liquidity_change") < 0.94)
            & ((z("ulcer_index") < -2.25) | (z("intra_sector_momentum_rank") < 0.67))
        ),
        ("r4", "c_sector_rank_only"): (
            (z("market_liquidity_change") < 0.94)
            & ((z("ulcer_index") < -2.25) | (z("momentum_peak_decay_3d") > 3.20))
        ),
    }
    rows = []
    for rule, path, threshold_a, threshold_b, candidate in threshold_candidates(frame):
        other_union = current.drop(columns=rule).any(axis=1)
        replacement_rule = candidate | fixed_siblings.get((rule, path), False)
        replacement_union = other_union | replacement_rule
        candidate_path = outcome_stats(frame, candidate)
        standalone = outcome_stats(frame, replacement_rule)
        incremental = outcome_stats(frame, replacement_rule & ~other_union)
        union = outcome_stats(frame, replacement_union)
        row = {
            "rule": rule,
            "path": path,
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            **{f"candidate_path_{key}": value for key, value in candidate_path.items()},
            **{f"standalone_{key}": value for key, value in standalone.items()},
            **{f"incremental_{key}": value for key, value in incremental.items()},
            **{f"replacement_union_{key}": value for key, value in union.items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    events = pd.read_csv(EVENT_AUDIT)
    events["buy_date"] = pd.to_datetime(events["buy_date"])
    events["factor_date"] = pd.to_datetime(events["factor_date"])
    events = attach_rebuilt_score_z(events)
    masks = build_current_masks(events)
    events["rule_hit_count"] = masks.sum(axis=1)
    for column in RULE_COLUMNS:
        events[column] = masks[column].to_numpy()
    events["current_union"] = masks.any(axis=1).to_numpy()

    rules = rule_quality(events, masks)
    pairs = pair_overlap(events, masks)
    paths = path_quality(events, masks)
    neighborhood = threshold_neighborhood(events, masks)

    events.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")
    rules.to_csv(OUT_RULES, index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig")
    paths.to_csv(OUT_PATHS, index=False, encoding="utf-8-sig")
    neighborhood.to_csv(OUT_NEIGHBORHOOD, index=False, encoding="utf-8-sig")

    print("RULE QUALITY")
    print(rules.to_string(index=False))
    print("\nPATH QUALITY")
    print(paths.to_string(index=False))
    print("\nTOP BIG-LOSS OVERLAPS")
    print(pairs.head(20).to_string(index=False))
    print(f"\nevents={OUT_EVENTS}")
    print(f"rules={OUT_RULES}")
    print(f"pairs={OUT_PAIRS}")
    print(f"paths={OUT_PATHS}")
    print(f"neighborhood={OUT_NEIGHBORHOOD}")


if __name__ == "__main__":
    main()
