"""reg_severe_up_10d_ratio 作为 three_rule_v3 候选规则r10的全面审计。

对照: three-rule-v3-risk-rule skill 标准
- big loss: actual_return <= -0.04
- small loss: -0.04 < actual_return < 0
- small gain: 0.03 <= actual_return < 0.08
- big gain: actual_return >= 0.08
- big gain strict: actual_return >= 0.10

使用最新匹配文件: 20260601_024558
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"

METRICS_FILE = RESULTS / "walkforward_metrics_20260601_024558.json"
SELECTIONS_FILE = RESULTS / "walkforward_selections_20260601_024558.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260601_024558.json"

RULE_COLS = [
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
    "score_z_in_day",
    "reg_severe_up_10d_ratio",
    # Additional factors that may help combo
    "sector_momentum_rank",
    "volume_ratio_v3",
    "intra_day_amplitude_20d",
    "turnover_rate_z_20d",
    "hs300_relative_strength_20d",
]


def load_json_records(path: Path) -> pd.DataFrame:
    obj = json.load(open(path, encoding="utf-8"))
    records = []

    def walk(x):
        if isinstance(x, list):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            keys = set(x)
            if "stock_code" in keys and (
                "actual_return" in keys
                or "profit_loss_pct" in keys
                or "buy_datetime" in keys
                or "date" in keys
                or "trade_date" in keys
            ):
                records.append(x)
            else:
                for v in x.values():
                    walk(v)

    walk(obj)
    return pd.DataFrame(records)


def normalize_events(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        for c in ["trade_date", "buy_date", "selection_date", "buy_datetime", "shadow_attempt_buy_date", "shadow_prediction_date"]:
            if c in df.columns:
                df["date"] = df[c]
                break
    if "stock_code" not in df.columns:
        for c in ["code", "ts_code", "selected_stock"]:
            if c in df.columns:
                df["stock_code"] = df[c]
                break
    ret_candidates = [
        "actual_return",
        "return",
        "trade_return",
        "future_return",
        "future_return_5d",
        "final_return",
        "realized_return",
    ]
    if "actual_return" not in df.columns:
        for c in ret_candidates:
            if c in df.columns:
                df["actual_return"] = df[c]
                break
    keep = [c for c in df.columns if c in {"date", "stock_code", "actual_return", "rank", "predicted_return", "score", "filter_reason"}]
    df = df[keep].copy()
    df["source"] = source
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str)
    df["actual_return"] = pd.to_numeric(df.get("actual_return"), errors="coerce")
    return df.dropna(subset=["date", "stock_code", "actual_return"])


def add_prev_factor_snapshot(events: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    trading_days = np.array(sorted(pd.to_datetime(factors["date"].dropna().unique())))
    events = events.copy()
    prev_dates = []
    for d in events["date"]:
        trade_date = pd.Timestamp(d).normalize()
        pos = np.searchsorted(trading_days, np.datetime64(trade_date), side="left") - 1
        prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)
    events["factor_date"] = prev_dates
    snap = factors.rename(columns={"date": "factor_date"})
    return events.merge(snap, on=["factor_date", "stock_code"], how="left")


def current_v3_masks(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["r1"] = (df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)
    out["r2"] = (
        ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
        | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
        | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15))
    )
    out["r3"] = (df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40)
    out["r4"] = (df["market_liquidity_change"] < 0.94) & (
        (df["ulcer_index"] < -0.80)
        | (df["momentum_peak_decay_3d"] > 1.0)
        | (df["intra_sector_momentum_rank"] < 0.67)
    )
    out["r5"] = (df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38)
    out["r6"] = df["limit_down_3d"] == 1
    out["r7"] = (df["top10_10d_strength_level_v3"] >= 0.70) & (df["buy_sm_amount_rate"] <= -1.25)
    out["r8"] = (df["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80) & (df["kdj_j_9"] <= 0.75)
    out["r9"] = df["score_z_in_day"] <= 1.50
    out["current_v3"] = out.any(axis=1)
    return out.fillna(False)


def bucket_counts(df: pd.DataFrame) -> dict:
    r = df["actual_return"]
    return {
        "n": int(len(df)),
        "mean": float(r.mean()) if len(df) else np.nan,
        "big_loss<=-4%": int((r <= -0.04).sum()),
        "small_loss": int(((r > -0.04) & (r < 0)).sum()),
        "small_gain3-8%": int(((r >= 0.03) & (r < 0.08)).sum()),
        "big_gain>=8%": int((r >= 0.08).sum()),
        "big_gain>=10%": int((r >= 0.10).sum()),
    }


def add_bucket(df: pd.DataFrame) -> pd.Series:
    r = df["actual_return"]
    return np.select(
        [
            r <= -0.04,
            (r > -0.04) & (r < 0),
            (r >= 0) & (r < 0.03),
            (r >= 0.03) & (r < 0.08),
            r >= 0.08,
        ],
        ["big_loss", "small_loss", "flat_small_gain", "small_gain", "big_gain"],
        default="unknown",
    )


def summarize_threshold(df: pd.DataFrame, threshold: float) -> dict:
    cand = df["reg_severe_up_10d_ratio"] >= threshold
    cur = df["current_v3"]
    inc = cand & ~cur
    hit = df[cand]
    inc_df = df[inc]
    return {
        "threshold": threshold,
        "standalone": bucket_counts(hit),
        "incremental_buckets": bucket_counts(inc_df),
        "overlap_current": int((cand & cur).sum()),
        "candidate_hit_top12": hit.loc[:, ["source", "date", "stock_code", "actual_return", "reg_severe_up_10d_ratio"]]
        .sort_values("reg_severe_up_10d_ratio", ascending=False)
        .head(12)
        .to_dict("records"),
    }


def eval_mask(df: pd.DataFrame, mask: pd.Series, label: str = "") -> dict:
    mask = mask.fillna(False)
    cur = df["current_v3"].fillna(False)
    inc = mask & ~cur
    rows = df[mask]
    inc_rows = df[inc]
    bl_mask = inc_rows["actual_return"] <= -0.04
    bg_mask = inc_rows["actual_return"] >= 0.08
    return {
        "label": label,
        "standalone": bucket_counts(rows),
        "incremental": bucket_counts(inc_rows),
        "overlap_current": int((mask & cur).sum()),
        "inc_actual_big_loss": int(((inc_rows["source"] == "actual") & bl_mask).sum()),
        "inc_actual_big_gain": int(((inc_rows["source"] == "actual") & bg_mask).sum()),
        "inc_shadow_big_loss": int(((inc_rows["source"] == "shadow") & bl_mask).sum()),
        "inc_shadow_big_gain": int(((inc_rows["source"] == "shadow") & bg_mask).sum()),
        "inc_examples": inc_rows.loc[
            :,
            ["source", "date", "stock_code", "actual_return", "reg_severe_up_10d_ratio"],
        ]
        .sort_values("actual_return")
        .head(10)
        .to_dict("records"),
        "inc_big_gain_examples": inc_rows.loc[bg_mask, [
            "source", "date", "stock_code", "actual_return", "reg_severe_up_10d_ratio",
            "market_advance_decline_ratio", "top10_10d_strength_level_v3", "market_liquidity_change",
        ]]
        .sort_values("actual_return", ascending=False)
        .head(10)
        .to_dict("records"),
    }


def scan_reg_thresholds(events: pd.DataFrame) -> list:
    """扫描 reg_severe_up_10d_ratio 单独阈值"""
    results = []
    reg = events["reg_severe_up_10d_ratio"]
    percentiles = reg.dropna().quantile([.75, .80, .85, .90, .95, .98, .99])
    thresholds = sorted(set([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30] +
                            [round(float(q), 6) for q in percentiles]))
    for t in thresholds:
        info = summarize_threshold(events, t)
        inc = info["incremental_buckets"]
        results.append({
            "threshold": t,
            "pct_rank": float((reg.dropna() <= t).mean()) if not reg.dropna().empty else np.nan,
            "standalone_n": info["standalone"]["n"],
            "standalone_big_loss": info["standalone"]["big_loss<=-4%"],
            "standalone_big_gain8": info["standalone"]["big_gain>=8%"],
            "inc_n": inc["n"],
            "inc_mean": inc["mean"],
            "inc_big_loss": inc["big_loss<=-4%"],
            "inc_small_loss": inc["small_loss"],
            "inc_small_gain": inc["small_gain3-8%"],
            "inc_big_gain8": inc["big_gain>=8%"],
            "inc_big_gain10": inc["big_gain>=10%"],
            "overlap_current": info["overlap_current"],
        })
    return sorted(results, key=lambda x: (x["inc_big_loss"] - 2.5 * x["inc_big_gain8"]), reverse=True)


def scan_two_factor(events: pd.DataFrame) -> pd.DataFrame:
    """扫描 reg + 另一个因子的组合规则"""
    df = events  # Keep alias for readability
    reg = df["reg_severe_up_10d_ratio"]
    candidates = {
        "market_advance_decline_ratio": [
            ("<=", 0.30), ("<=", 0.35), ("<=", 0.38), ("<=", 0.45), ("<=", 0.50),
            (">=", 1.5), (">=", 2.0), (">=", 3.0),
        ],
        "top10_10d_strength_level_v3": [
            (">=", 0.55), (">=", 0.60), (">=", 0.65), (">=", 0.70), (">=", 0.75), (">=", 0.80),
        ],
        "market_liquidity_change": [
            ("<", 0.90), ("<", 0.94), ("<", 0.98), ("<", 1.00), (">=", 1.05), (">=", 1.10),
        ],
        "mkt_dispersion_shock_v2": [
            ("<", 0.90), ("<", 0.98), ("<", 1.05), (">=", 1.20), (">=", 1.50),
        ],
        "runup_ma5_exhaustion": [
            (">", 0.6), (">", 0.7), (">", 0.8), (">", 0.9),
        ],
        "cpv_factor": [
            ("<=", -0.8), ("<=", -0.6), ("<=", -0.4), (">=", 0.4), (">=", 0.6),
        ],
        "prev_day_blowoff_risk": [
            (">=", 1.0), (">=", 2.0), (">=", 3.0), (">=", 5.0),
        ],
        "sector_momentum_accel_rank": [
            ("<=", 0.15), ("<=", 0.30), ("<=", 0.50), (">=", 0.70),
        ],
        "sector_leadership_persistence_v3": [
            ("<=", -0.30), ("<=", -0.18), ("<=", 0.0),
        ],
        "return_20d_intra_z": [
            ("<=", 0.0), ("<=", 0.40), ("<=", 0.60), (">=", 1.0),
        ],
        "ulcer_index": [
            ("<", -1.2), ("<", -0.8), ("<", -0.4),
        ],
        "momentum_peak_decay_3d": [
            (">", 0.5), (">", 1.0), (">", 2.0),
        ],
        "intra_sector_momentum_rank": [
            ("<", 0.30), ("<", 0.50), ("<", 0.67),
        ],
        "limit_down_3d": [
            ("==", 1.0),
        ],
        "buy_sm_amount_rate": [
            ("<=", -1.5), ("<=", -1.25), ("<=", -1.0), (">=", 1.0),
        ],
        "prev_non_first_limit_up_performance_percentile_20d": [
            (">=", 0.70), (">=", 0.80), (">=", 0.90),
        ],
        "kdj_j_9": [
            ("<=", 0.5), ("<=", 0.75), ("<=", 1.0), (">=", 1.5),
        ],
    }

    def compare(series: pd.Series, op: str, value: float) -> pd.Series:
        if op == "<=":
            return series.astype(float) <= value
        if op == "<":
            return series.astype(float) < value
        if op == ">=":
            return series.astype(float) >= value
        if op == ">":
            return series.astype(float) > value
        if op == "==":
            return series.astype(float) == value
        raise ValueError(op)

    rows = []
    reg_thresholds = [0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.621359, 0.65, 0.70, 0.75, 0.80]
    for r in reg_thresholds:
        reg_mask = reg >= r
        for factor, tests in candidates.items():
            if factor not in df.columns:
                continue
            for op, value in tests:
                mask = reg_mask & compare(df[factor], op, value)
                stats = eval_mask(df, mask, f"reg>={r:.3f} & {factor}{op}{value:g}")
                inc = stats["incremental"]
                if inc["n"] == 0:
                    continue
                rows.append({
                    "rule": stats["label"],
                    "reg_threshold": r,
                    "factor": factor,
                    "op": op,
                    "factor_threshold": value,
                    "standalone_n": stats["standalone"]["n"],
                    "standalone_big_loss": stats["standalone"]["big_loss<=-4%"],
                    "standalone_big_gain8": stats["standalone"]["big_gain>=8%"],
                    "inc_n": inc["n"],
                    "inc_mean": inc["mean"],
                    "inc_big_loss": inc["big_loss<=-4%"],
                    "inc_small_loss": inc["small_loss"],
                    "inc_small_gain": inc["small_gain3-8%"],
                    "inc_big_gain8": inc["big_gain>=8%"],
                    "inc_big_gain10": inc["big_gain>=10%"],
                    "inc_actual_big_loss": stats["inc_actual_big_loss"],
                    "inc_actual_big_gain": stats["inc_actual_big_gain"],
                    "inc_shadow_big_loss": stats["inc_shadow_big_loss"],
                    "inc_shadow_big_gain": stats["inc_shadow_big_gain"],
                    "overlap_current": stats["overlap_current"],
                    "score": inc["big_loss<=-4%"] - 3.0 * inc["big_gain>=8%"] - 0.5 * inc["small_gain3-8%"],
                    "examples": stats["inc_examples"],
                    "big_gain_examples": stats["inc_big_gain_examples"],
                })
    return pd.DataFrame(rows).sort_values(["score", "inc_big_loss", "inc_big_gain8", "inc_n"],
                                            ascending=[False, False, True, True])


def scan_three_factor(events: pd.DataFrame) -> pd.DataFrame:
    """扫描 reg + 两个其他因子的三因子组合"""
    df = events
    reg = df["reg_severe_up_10d_ratio"]
    factors_cfg = {
        "market_advance_decline_ratio": [("<=", 0.35), ("<=", 0.38), ("<=", 0.45), (">=", 2.0), (">=", 3.0)],
        "top10_10d_strength_level_v3": [(">=", 0.60), (">=", 0.65), (">=", 0.70), (">=", 0.75)],
        "market_liquidity_change": [("<", 0.94), ("<", 0.98), ("<", 1.00)],
        "mkt_dispersion_shock_v2": [("<", 0.90), ("<", 0.98)],
        "runup_ma5_exhaustion": [(">", 0.7), (">", 0.8)],
        "buy_sm_amount_rate": [("<=", -1.0), ("<=", -1.25)],
    }

    def compare(series: pd.Series, op: str, value: float) -> pd.Series:
        if op == "<=":
            return series.astype(float) <= value
        if op == "<":
            return series.astype(float) < value
        if op == ">=":
            return series.astype(float) >= value
        if op == ">":
            return series.astype(float) > value
        raise ValueError(op)

    rows = []
    reg_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]

    factor_names = list(factors_cfg.keys())
    for r in reg_thresholds:
        reg_mask = reg >= r
        for i, f1_name in enumerate(factor_names):
            if f1_name not in df.columns:
                continue
            for f2_name in factor_names[i + 1:]:
                if f2_name not in df.columns:
                    continue
                for op1, v1 in factors_cfg[f1_name]:
                    for op2, v2 in factors_cfg[f2_name]:
                        mask = reg_mask & compare(df[f1_name], op1, v1) & compare(df[f2_name], op2, v2)
                        stats = eval_mask(df, mask, f"reg>={r:.2f}&{f1_name}{op1}{v1:g}&{f2_name}{op2}{v2:g}")
                        inc = stats["incremental"]
                        if inc["n"] == 0:
                            continue
                        rows.append({
                            "rule": stats["label"],
                            "inc_n": inc["n"],
                            "inc_mean": inc["mean"],
                            "inc_big_loss": inc["big_loss<=-4%"],
                            "inc_small_loss": inc["small_loss"],
                            "inc_small_gain": inc["small_gain3-8%"],
                            "inc_big_gain8": inc["big_gain>=8%"],
                            "inc_big_gain10": inc["big_gain>=10%"],
                            "inc_actual_big_loss": stats["inc_actual_big_loss"],
                            "inc_actual_big_gain": stats["inc_actual_big_gain"],
                            "inc_shadow_big_loss": stats["inc_shadow_big_loss"],
                            "inc_shadow_big_gain": stats["inc_shadow_big_gain"],
                            "overlap_current": stats["overlap_current"],
                            "score": inc["big_loss<=-4%"] - 3.0 * inc["big_gain>=8%"] - 0.5 * inc["small_gain3-8%"],
                            "examples": stats["inc_examples"],
                            "big_gain_examples": stats["inc_big_gain_examples"],
                        })
    return pd.DataFrame(rows).sort_values(["score", "inc_big_loss", "inc_big_gain8", "inc_n"],
                                           ascending=[False, False, True, True])


def analyze_rule_replacement(events: pd.DataFrame, candidate_rules: list[dict]) -> None:
    """分析候选规则替换现有规则的效果"""
    masks = current_v3_masks(events)
    cur = masks["current_v3"]

    for cr in candidate_rules:
        label = cr["label"]
        cand_mask = cr["mask"].fillna(False)
        inc = cand_mask & ~cur
        inc_rows = events[inc]
        print(f"\n{'='*80}")
        print(f"CANDIDATE: {label}")
        print(f"  standalone: {bucket_counts(events[cand_mask])}")
        print(f"  incremental (over current v3): {bucket_counts(inc_rows)}")
        print(f"  overlap with current v3: {int((cand_mask & cur).sum())}")
        print(f"  inc actual big_loss: {int(((inc_rows['source']=='actual')&(inc_rows['actual_return']<=-0.04)).sum())}")
        print(f"  inc actual big_gain: {int(((inc_rows['source']=='actual')&(inc_rows['actual_return']>=0.08)).sum())}")
        print(f"  inc shadow big_loss: {int(((inc_rows['source']=='shadow')&(inc_rows['actual_return']<=-0.04)).sum())}")
        print(f"  inc shadow big_gain: {int(((inc_rows['source']=='shadow')&(inc_rows['actual_return']>=0.08)).sum())}")

        # Replacement analysis: drop each existing rule, add candidate
        print(f"\n  Replacement analysis (drop_one + candidate):")
        for drop_name in sorted(masks.keys()):
            replacement = pd.Series(False, index=events.index)
            for name, m in masks.items():
                if name != drop_name and name != "current_v3":
                    replacement |= m
            replacement_union = replacement | cand_mask
            rep_inc = replacement_union & ~cur
            rep_inc_rows = events[rep_inc]
            print(f"    drop_{drop_name}: orig_hit={int(replacement.sum())}, "
                  f"+cand_hit={int(replacement_union.sum())}, "
                  f"new_inc_big_loss={int((rep_inc_rows['actual_return']<=-0.04).sum())}, "
                  f"new_inc_big_gain8={int((rep_inc_rows['actual_return']>=0.08).sum())}, "
                  f"new_inc_small_gain={int(((rep_inc_rows['actual_return']>=0.03)&(rep_inc_rows['actual_return']<0.08)).sum())}")


def main() -> None:
    print("Loading data...")
    selections = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(
        ["source", "date", "stock_code", "actual_return"]
    )

    schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
    factor_cols = ["date", "stock_code"] + [c for c in RULE_COLS if c in schema_cols]
    factors = pd.read_parquet(FACTORS, columns=factor_cols)
    for col in RULE_COLS:
        if col not in factors.columns:
            factors[col] = np.nan
    factors["date"] = pd.to_datetime(factors["date"])
    events = add_prev_factor_snapshot(events, factors)
    masks = current_v3_masks(events)
    events = pd.concat([events, masks], axis=1)
    events["bucket"] = add_bucket(events)

    print("=" * 80)
    print("FILES USED")
    print(f"  Metrics: {METRICS_FILE}")
    print(f"  Selections: {SELECTIONS_FILE}")
    print(f"  Shadow: {SHADOW_FILE}")
    print(f"  Factors: {FACTORS}")

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print(events.groupby("source").size().to_string())
    print(f"missing reg_severe_up_10d_ratio: {int(events['reg_severe_up_10d_ratio'].isna().sum())}")

    print("\n" + "=" * 80)
    print("BASE BUCKETS (by source)")
    for source, group in events.groupby("source"):
        print(f"  {source}: {bucket_counts(group)}")
    print(f"  combined: {bucket_counts(events)}")

    print("\n" + "=" * 80)
    print("CURRENT V3 RULE HITS")
    for source, group in events.groupby("source"):
        cur_hits = group[group["current_v3"]]
        print(f"  {source}: hit={int(group['current_v3'].sum())}, {bucket_counts(cur_hits)}")
    all_cur = events[events["current_v3"]]
    print(f"  combined: hit={int(events['current_v3'].sum())}, {bucket_counts(all_cur)}")

    print("\n" + "=" * 80)
    print("CURRENT V3 UNFILTERED (not hit by any rule)")
    unfiltered = events[~events["current_v3"]]
    print(f"  combined unfiltered: {bucket_counts(unfiltered)}")

    print("\n" + "=" * 80)
    print("REG_SEVERE_UP_10D_RATIO DISTRIBUTION")
    reg_col = events["reg_severe_up_10d_ratio"]
    print(f"  overall: {reg_col.describe(percentiles=[.5, .75, .8, .85, .9, .95, .98, .99]).to_string()}")
    for source, group in events.groupby("source"):
        print(f"\n  {source}:")
        print(f"  {group['reg_severe_up_10d_ratio'].describe(percentiles=[.5, .75, .8, .85, .9, .95, .98, .99]).to_string()}")

    print("\n" + "=" * 80)
    print("REG FACTOR BY BUCKET (combined)")
    print(events.groupby(["source", "bucket"])["reg_severe_up_10d_ratio"]
          .describe(percentiles=[.5, .75, .9]).to_string())

    # === SINGLE THRESHOLD SCAN ===
    print("\n" + "=" * 80)
    print("SINGLE THRESHOLD SCAN (reg_severe_up_10d_ratio >= T)")
    print("Score = inc_big_loss - 2.5 * inc_big_gain8")
    single_results = scan_reg_thresholds(events)
    for s in single_results:
        score = s["inc_big_loss"] - 2.5 * s["inc_big_gain8"]
        print(f"  T>={s['threshold']:.4f} (pct={s['pct_rank']:.3f}) "
              f"standalone_n={s['standalone_n']:4d} inc_n={s['inc_n']:3d} "
              f"inc_big_loss={s['inc_big_loss']} inc_big_gain8={s['inc_big_gain8']} "
              f"inc_big_gain10={s['inc_big_gain10']} inc_small_gain={s['inc_small_gain']} "
              f"inc_small_loss={s['inc_small_loss']} inc_mean={s['inc_mean']:.4f} "
              f"overlap={s['overlap_current']} score={score:.1f}")

    # Best single threshold detail
    best_single = single_results[0]
    t = best_single["threshold"]
    cand_mask = reg_col >= t
    detail = eval_mask(events, cand_mask, f"reg>={t:.4f} (single best)")
    print(f"\nBest single threshold detail: reg >= {t:.4f}")
    inc_rows = events[cand_mask & ~events["current_v3"]]
    print(f"  incremental examples (worst returns):")
    for ex in detail["inc_examples"]:
        print(f"    {ex['source']:8s} {ex['date']} {ex['stock_code']:12s} ret={ex['actual_return']:.4f} reg={ex['reg_severe_up_10d_ratio']:.4f}")
    print(f"  incremental big_gain examples:")
    for ex in detail["inc_big_gain_examples"]:
        print(f"    {ex['source']:8s} {ex['date']} {ex['stock_code']:12s} ret={ex['actual_return']:.4f} reg={ex['reg_severe_up_10d_ratio']:.4f}")

    # === TWO-FACTOR SCAN ===
    print("\n" + "=" * 80)
    print("TWO-FACTOR COMBINATION SCAN (reg + one other factor)")
    print("Score = inc_big_loss - 3.0 * inc_big_gain8 - 0.5 * inc_small_gain")
    twof_df = scan_two_factor(events)
    show_cols = ["rule", "inc_n", "inc_mean", "inc_big_loss", "inc_small_loss", "inc_small_gain",
                 "inc_big_gain8", "inc_big_gain10", "inc_actual_big_loss", "inc_actual_big_gain",
                 "inc_shadow_big_loss", "inc_shadow_big_gain", "overlap_current", "score"]

    print("\n--- TOP 40 by score ---")
    print(twof_df.head(40)[show_cols].to_string(index=False))

    print("\n--- ZERO big_gain WITH big_loss > 0 ---")
    zbg = twof_df[(twof_df["inc_big_gain8"] == 0) & (twof_df["inc_big_loss"] > 0)]
    if not zbg.empty:
        print(zbg.head(30)[show_cols].to_string(index=False))
    else:
        print("NONE FOUND")

    print("\n--- LOW FALSE-KILL (big_loss >= 1 AND big_gain8 <= 1) ---")
    low_fk = twof_df[(twof_df["inc_big_loss"] >= 1) & (twof_df["inc_big_gain8"] <= 1)]
    if not low_fk.empty:
        print(low_fk.head(30)[show_cols].to_string(index=False))
    else:
        print("NONE FOUND")

    # Show details for top 5 two-factor combos
    print("\n--- TOP 5 TWO-FACTOR COMBOS DETAIL ---")
    for _, row in twof_df.head(5).iterrows():
        print(f"\n  Rule: {row['rule']}")
        print(f"  inc_n={row['inc_n']}, inc_big_loss={row['inc_big_loss']}, inc_big_gain8={row['inc_big_gain8']}, score={row['score']:.1f}")
        if row["inc_big_loss"] > 0:
            print(f"  big_loss examples: {json.dumps(row['examples'][:5], ensure_ascii=False, default=str)}")
        if row["inc_big_gain8"] > 0:
            print(f"  big_gain examples: {json.dumps(row['big_gain_examples'][:5], ensure_ascii=False, default=str)}")

    # Show ALL zero-big-gain rules' detail
    if not zbg.empty:
        print("\n--- ZERO BIG-GAIN RULES DETAIL ---")
        for _, row in zbg.head(10).iterrows():
            print(f"\n  Rule: {row['rule']} (inc_n={row['inc_n']}, inc_big_loss={row['inc_big_loss']}, score={row['score']:.1f})")
            print(f"  examples: {json.dumps(row['examples'][:8], ensure_ascii=False, default=str)}")

    # === THREE-FACTOR SCAN ===
    print("\n" + "=" * 80)
    print("THREE-FACTOR COMBINATION SCAN (reg + two other factors)")
    threef_df = scan_three_factor(events)
    if not threef_df.empty:
        print("\n--- TOP 30 by score ---")
        show_cols3 = ["rule", "inc_n", "inc_mean", "inc_big_loss", "inc_big_gain8", "inc_big_gain10",
                      "inc_actual_big_loss", "inc_actual_big_gain", "inc_shadow_big_loss",
                      "inc_shadow_big_gain", "overlap_current", "score"]
        print(threef_df.head(30)[show_cols3].to_string(index=False))

        print("\n--- ZERO big_gain WITH big_loss > 0 ---")
        zbg3 = threef_df[(threef_df["inc_big_gain8"] == 0) & (threef_df["inc_big_loss"] > 0)]
        if not zbg3.empty:
            print(zbg3.head(20)[show_cols3].to_string(index=False))
            print("\n--- ZERO BIG-GAIN THREE-FACTOR DETAIL ---")
            for _, row in zbg3.head(5).iterrows():
                print(f"\n  Rule: {row['rule']} (inc_n={row['inc_n']}, inc_big_loss={row['inc_big_loss']}, score={row['score']:.1f})")
                print(f"  examples: {json.dumps(row['examples'][:8], ensure_ascii=False, default=str)}")
        else:
            print("NONE FOUND")
    else:
        print("No valid three-factor combinations found.")

    # === RULE REPLACEMENT ANALYSIS ===
    print("\n" + "=" * 80)
    print("RULE REPLACEMENT ANALYSIS")

    # Build candidate rules
    candidates = []
    # Best single threshold
    best_t = single_results[0]["threshold"]
    candidates.append({
        "label": f"reg_severe_up_10d_ratio >= {best_t:.4f}",
        "mask": reg_col >= best_t,
    })
    # Best two-factor from zero-big-gain list
    if not zbg.empty:
        best_2f = zbg.iloc[0]
        reg_t = best_2f["reg_threshold"]
        f_name = best_2f["factor"]
        f_op = best_2f["op"]
        f_val = best_2f["factor_threshold"]
        if f_op == "<=":
            f_mask = events[f_name].astype(float) <= f_val
        elif f_op == "<":
            f_mask = events[f_name].astype(float) < f_val
        elif f_op == ">=":
            f_mask = events[f_name].astype(float) >= f_val
        elif f_op == ">":
            f_mask = events[f_name].astype(float) > f_val
        elif f_op == "==":
            f_mask = events[f_name].astype(float) == f_val
        else:
            f_mask = pd.Series(False, index=events.index)
        candidates.append({
            "label": f"reg>={reg_t:.3f} & {f_name}{f_op}{f_val:g}",
            "mask": (reg_col >= reg_t) & f_mask,
        })
    analyze_rule_replacement(events, candidates)

    # === SUMMARY: BIG LOSS & BIG GAIN TOP LISTS ===
    print("\n" + "=" * 80)
    print("TOP BIG LOSSES (actual_return <= -4%)")
    cols = ["source", "date", "stock_code", "actual_return", "current_v3", "reg_severe_up_10d_ratio"]
    big_losses = events[events["actual_return"] <= -0.04][cols].sort_values("actual_return")
    print(big_losses.to_string(index=False))

    print("\n" + "=" * 80)
    print("TOP BIG GAINS (actual_return >= 8%)")
    big_gains = events[events["actual_return"] >= 0.08][cols].sort_values("actual_return", ascending=False)
    print(big_gains.to_string(index=False))

    # === UNFILTERED BIG LOSS ANALYSIS ===
    print("\n" + "=" * 80)
    print("UNFILTERED BIG LOSSES (not hit by any v3 rule, return <= -4%)")
    uf_bl = events[(~events["current_v3"]) & (events["actual_return"] <= -0.04)]
    uf_cols = ["source", "date", "stock_code", "actual_return", "reg_severe_up_10d_ratio"]
    print(uf_bl[uf_cols].sort_values("actual_return").to_string(index=False))
    print(f"\nreg_severe_up_10d_ratio distribution among unfiltered big losses:")
    if len(uf_bl) > 0:
        print(uf_bl["reg_severe_up_10d_ratio"].describe(percentiles=[.5, .75, .9, .95]).to_string())
        # Show what threshold would catch them
        for t in [0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
            caught = int((uf_bl["reg_severe_up_10d_ratio"] >= t).sum())
            if caught > 0:
                print(f"  T>={t:.2f}: catches {caught}/{len(uf_bl)}")

    # === UNFILTERED BIG GAIN ANALYSIS ===
    print("\n" + "=" * 80)
    print("UNFILTERED BIG GAINS (not hit by any v3 rule, return >= 8%)")
    uf_bg = events[(~events["current_v3"]) & (events["actual_return"] >= 0.08)]
    print(uf_bg[uf_cols].sort_values("actual_return", ascending=False).to_string(index=False))
    print(f"\nreg_severe_up_10d_ratio distribution among unfiltered big gains:")
    if len(uf_bg) > 0:
        print(uf_bg["reg_severe_up_10d_ratio"].describe(percentiles=[.5, .75, .9, .95]).to_string())
        for t in [0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
            killed = int((uf_bg["reg_severe_up_10d_ratio"] >= t).sum())
            if killed > 0:
                print(f"  T>={t:.2f}: kills {killed}/{len(uf_bg)}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
