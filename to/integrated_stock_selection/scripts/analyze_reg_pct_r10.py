"""reg_severe_up_10d_ratio 横截面分位数作为 three_rule_v3 候选规则。

计算每日横截面百分位排名，分析是否能作为独立规则或组合规则。
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
    "runup_ma5_exhaustion", "cpv_factor", "prev_day_blowoff_risk",
    "sector_momentum_accel_rank", "sector_leadership_persistence_v3",
    "return_20d_intra_z", "market_liquidity_change", "ulcer_index",
    "momentum_peak_decay_3d", "intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2", "market_advance_decline_ratio",
    "limit_down_3d", "top10_10d_strength_level_v3", "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d", "kdj_j_9",
    "score_z_in_day", "reg_severe_up_10d_ratio",
]


def load_json_records(path: Path) -> pd.DataFrame:
    obj = json.load(open(path, encoding="utf-8"))
    records = []
    def walk(x):
        if isinstance(x, list):
            for i in x: walk(i)
        elif isinstance(x, dict):
            keys = set(x)
            if "stock_code" in keys and (
                "actual_return" in keys or "profit_loss_pct" in keys
                or "buy_datetime" in keys or "date" in keys or "trade_date" in keys
            ):
                records.append(x)
            else:
                for v in x.values(): walk(v)
    walk(obj)
    return pd.DataFrame(records)


def normalize_events(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        for c in ["trade_date", "buy_date", "selection_date", "buy_datetime",
                   "shadow_attempt_buy_date", "shadow_prediction_date"]:
            if c in df.columns:
                df["date"] = df[c]; break
    if "stock_code" not in df.columns:
        for c in ["code", "ts_code", "selected_stock"]:
            if c in df.columns:
                df["stock_code"] = df[c]; break
    ret_candidates = ["actual_return", "return", "trade_return", "future_return",
                      "future_return_5d", "final_return", "realized_return"]
    if "actual_return" not in df.columns:
        for c in ret_candidates:
            if c in df.columns:
                df["actual_return"] = df[c]; break
    keep = [c for c in df.columns if c in {"date", "stock_code", "actual_return"}]
    df = df[keep].copy()
    df["source"] = source
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str)
    df["actual_return"] = pd.to_numeric(df.get("actual_return"), errors="coerce")
    return df.dropna(subset=["date", "stock_code", "actual_return"])


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
        (df["ulcer_index"] < -0.80) | (df["momentum_peak_decay_3d"] > 1.0)
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
        [r <= -0.04, (r > -0.04) & (r < 0), (r >= 0) & (r < 0.03),
         (r >= 0.03) & (r < 0.08), r >= 0.08],
        ["big_loss", "small_loss", "flat_small_gain", "small_gain", "big_gain"],
        default="unknown",
    )


def eval_mask(df: pd.DataFrame, mask: pd.Series, label: str = "") -> dict:
    mask = mask.fillna(False)
    cur = df["current_v3"].fillna(False)
    inc = mask & ~cur
    rows = df[mask]
    inc_rows = df[inc]
    bl = inc_rows["actual_return"] <= -0.04
    bg = inc_rows["actual_return"] >= 0.08
    return {
        "label": label,
        "standalone": bucket_counts(rows),
        "incremental": bucket_counts(inc_rows),
        "overlap_current": int((mask & cur).sum()),
        "inc_actual_big_loss": int(((inc_rows["source"] == "actual") & bl).sum()),
        "inc_actual_big_gain": int(((inc_rows["source"] == "actual") & bg).sum()),
        "inc_shadow_big_loss": int(((inc_rows["source"] == "shadow") & bl).sum()),
        "inc_shadow_big_gain": int(((inc_rows["source"] == "shadow") & bg).sum()),
    }


def main() -> None:
    print("Loading data...")
    selections = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(
        ["source", "date", "stock_code", "actual_return"]
    )

    # Load factors and compute cross-sectional percentiles
    schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
    factor_cols = ["date", "stock_code"] + [c for c in RULE_COLS if c in schema_cols]
    factors = pd.read_parquet(FACTORS, columns=factor_cols)
    for col in RULE_COLS:
        if col not in factors.columns:
            factors[col] = np.nan
    factors["date"] = pd.to_datetime(factors["date"])

    # Compute daily cross-sectional percentile rank for reg_severe_up_10d_ratio
    print("Computing daily cross-sectional percentiles for reg_severe_up_10d_ratio...")
    reg_col = "reg_severe_up_10d_ratio"
    pct_col = "reg_severe_up_10d_ratio_pct"
    factors[pct_col] = factors.groupby("date")[reg_col].transform(
        lambda x: x.rank(pct=True)
    )
    print(f"  Percentile stats: {factors[pct_col].describe().to_dict()}")

    # Attach factors to events (t-1 trading day)
    trading_days = np.array(sorted(factors["date"].dropna().unique()))
    prev_dates = []
    for d in events["date"]:
        trade_date = pd.Timestamp(d).normalize()
        pos = np.searchsorted(trading_days, np.datetime64(trade_date), side="left") - 1
        prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)
    events["factor_date"] = prev_dates

    snap = factors.rename(columns={"date": "factor_date"})
    events = events.merge(snap, on=["factor_date", "stock_code"], how="left")
    masks = current_v3_masks(events)
    events = pd.concat([events, masks], axis=1)
    events["bucket"] = add_bucket(events)

    # Also compute reg raw value percentile for big_loss vs big_gain comparison
    reg_raw = events["reg_severe_up_10d_ratio"]
    reg_pct = events["reg_severe_up_10d_ratio_pct"]

    print(f"\n{'='*80}")
    print(f"DATA: {len(events)} events, missing reg_raw={int(reg_raw.isna().sum())}, missing reg_pct={int(reg_pct.isna().sum())}")

    print(f"\n{'='*80}")
    print("REG_SEVERE_UP_10D_RATIO 横截面百分位 (reg_pct) 分布")
    print(reg_pct.describe(percentiles=[.5, .6, .7, .75, .8, .85, .9, .95, .98, .99]).to_string())

    print(f"\n{'='*80}")
    print("REG_PCT 按分桶 (combined)")
    print(events.groupby(["source", "bucket"])[pct_col].describe(
        percentiles=[.5, .75, .9]).to_string())

    # Unfiltered analysis
    uf = events[~events["current_v3"]]
    uf_bl = uf[uf["actual_return"] <= -0.04]
    uf_bg = uf[uf["actual_return"] >= 0.08]

    print(f"\n{'='*80}")
    print("UNFILTERED BIG LOSSES reg_raw & reg_pct")
    uf_bl_cols = ["source", "date", "stock_code", "actual_return", reg_col, pct_col]
    print(uf_bl[uf_bl_cols].sort_values("actual_return").to_string(index=False))

    print(f"\nUNFILTERED BIG GAINS reg_raw & reg_pct")
    print(uf_bg[uf_bl_cols].sort_values("actual_return", ascending=False).to_string(index=False))

    # =========================================================
    # SINGLE THRESHOLD SCAN using cross-sectional percentile
    # =========================================================
    print(f"\n{'='*80}")
    print("SINGLE THRESHOLD SCAN (reg_pct >= T) - 横截面百分位阈值")
    print("Score = inc_big_loss - 2.5 * inc_big_gain8")

    pct_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
    for t in pct_thresholds:
        mask = reg_pct >= t
        stats = eval_mask(events, mask, f"reg_pct>={t:.2f}")
        inc = stats["incremental"]
        score = inc["big_loss<=-4%"] - 2.5 * inc["big_gain>=8%"]
        print(f"  T>={t:.2f} standalone_n={stats['standalone']['n']:3d} inc_n={inc['n']:3d} "
              f"inc_big_loss={inc['big_loss<=-4%']} inc_big_gain8={inc['big_gain>=8%']} "
              f"inc_big_gain10={inc['big_gain>=10%']} inc_small_gain={inc['small_gain3-8%']} "
              f"inc_mean={inc['mean']:.4f} overlap={stats['overlap_current']} score={score:.1f}")

    # Best single percentile detail
    best_t = 0.75
    print(f"\nBest single pct threshold detail: reg_pct >= {best_t:.2f}")
    cand = reg_pct >= best_t
    inc_rows = events[cand & ~events["current_v3"]]
    detail_cols = ["source", "date", "stock_code", "actual_return", reg_col, pct_col]
    print("  Incremental worst returns:")
    for _, r in inc_rows.sort_values("actual_return").head(12).iterrows():
        print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:.4f} reg_raw={r[reg_col]:.4f} reg_pct={r[pct_col]:.4f}")
    print("  Incremental big_gains:")
    bg_inc = inc_rows[inc_rows["actual_return"] >= 0.08]
    for _, r in bg_inc.sort_values("actual_return", ascending=False).iterrows():
        print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:.4f} reg_raw={r[reg_col]:.4f} reg_pct={r[pct_col]:.4f}")

    # =========================================================
    # TWO-FACTOR SCAN using reg_pct + other factors
    # =========================================================
    print(f"\n{'='*80}")
    print("TWO-FACTOR SCAN (reg_pct + one other factor)")
    print("Score = inc_big_loss - 3.0 * inc_big_gain8 - 0.5 * inc_small_gain")

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
        "runup_ma5_exhaustion": [(">", 0.6), (">", 0.7), (">", 0.8), (">", 0.9)],
        "cpv_factor": [("<=", -0.8), ("<=", -0.6), ("<=", -0.4), (">=", 0.4), (">=", 0.6)],
        "prev_day_blowoff_risk": [(">=", 1.0), (">=", 2.0), (">=", 3.0), (">=", 5.0)],
        "sector_momentum_accel_rank": [("<=", 0.15), ("<=", 0.30), ("<=", 0.50), (">=", 0.70)],
        "sector_leadership_persistence_v3": [("<=", -0.30), ("<=", -0.18), ("<=", 0.0)],
        "return_20d_intra_z": [("<=", 0.0), ("<=", 0.40), ("<=", 0.60), (">=", 1.0)],
        "ulcer_index": [("<", -1.2), ("<", -0.8), ("<", -0.4)],
        "momentum_peak_decay_3d": [(">", 0.5), (">", 1.0), (">", 2.0)],
        "intra_sector_momentum_rank": [("<", 0.30), ("<", 0.50), ("<", 0.67)],
        "limit_down_3d": [("==", 1.0)],
        "buy_sm_amount_rate": [("<=", -1.5), ("<=", -1.25), ("<=", -1.0), (">=", 1.0)],
        "prev_non_first_limit_up_performance_percentile_20d": [(">=", 0.70), (">=", 0.80), (">=", 0.90)],
        "kdj_j_9": [("<=", 0.5), ("<=", 0.75), ("<=", 1.0), (">=", 1.5)],
    }

    def compare(series, op, value):
        s = series.astype(float)
        if op == "<=": return s <= value
        if op == "<": return s < value
        if op == ">=": return s >= value
        if op == ">": return s > value
        if op == "==": return s == value
        raise ValueError(op)

    twof_rows = []
    pct_thresholds_2f = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for pt in pct_thresholds_2f:
        reg_mask = reg_pct >= pt
        for factor, tests in candidates.items():
            if factor not in events.columns: continue
            for op, value in tests:
                mask = reg_mask & compare(events[factor], op, value)
                stats = eval_mask(events, mask, f"reg_pct>={pt:.2f}&{factor}{op}{value:g}")
                inc = stats["incremental"]
                if inc["n"] == 0: continue
                twof_rows.append({
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
                })
    twof = pd.DataFrame(twof_rows).sort_values(
        ["score", "inc_big_loss", "inc_big_gain8", "inc_n"],
        ascending=[False, False, True, True])

    show_cols = ["rule", "inc_n", "inc_mean", "inc_big_loss", "inc_small_loss",
                 "inc_small_gain", "inc_big_gain8", "inc_big_gain10",
                 "inc_actual_big_loss", "inc_actual_big_gain",
                 "inc_shadow_big_loss", "inc_shadow_big_gain",
                 "overlap_current", "score"]

    print(f"\n--- TWO-FACTOR: TOP 30 by score ---")
    print(twof.head(30)[show_cols].to_string(index=False))

    print(f"\n--- TWO-FACTOR: ZERO big_gain WITH big_loss > 0 ---")
    zbg = twof[(twof["inc_big_gain8"] == 0) & (twof["inc_big_loss"] > 0)]
    if not zbg.empty:
        print(zbg.head(30)[show_cols].to_string(index=False))
    else:
        print("NONE FOUND")

    # Show best zero-big-gain combo detail
    if not zbg.empty:
        print(f"\n--- BEST ZERO-BIG-GAIN TWO-FACTOR DETAIL ---")
        for _, row in zbg.head(5).iterrows():
            print(f"\n  Rule: {row['rule']} (inc_n={row['inc_n']}, inc_big_loss={row['inc_big_loss']}, score={row['score']:.1f})")
            # Reconstruct mask to get examples
            parts = row["rule"].split("&")
            mask = pd.Series(True, index=events.index)
            for part in parts:
                part = part.strip()
                if part.startswith("reg_pct>="):
                    val = float(part.replace("reg_pct>=", ""))
                    mask &= reg_pct >= val
                else:
                    for factor, tests in candidates.items():
                        for op, value in tests:
                            tag = f"{factor}{op}{value:g}"
                            if tag == part:
                                mask &= compare(events[factor], op, value)
                                break
            inc_rows = events[mask & ~events["current_v3"]]
            dc = ["source", "date", "stock_code", "actual_return", reg_col, pct_col]
            print(f"  big_losses:")
            for _, r in inc_rows[inc_rows["actual_return"] <= -0.04].sort_values("actual_return").iterrows():
                print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:.4f} reg_raw={r[reg_col]:.4f} reg_pct={r[pct_col]:.4f}")
            bg = inc_rows[inc_rows["actual_return"] >= 0.08]
            if len(bg) > 0:
                print(f"  big_gains (FALSE KILLS):")
                for _, r in bg.sort_values("actual_return", ascending=False).iterrows():
                    print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:.4f} reg_raw={r[reg_col]:.4f} reg_pct={r[pct_col]:.4f}")
            else:
                print(f"  big_gains: NONE (zero false kills)")

    # =========================================================
    # COMPARE raw vs pct: which gives better separation?
    # =========================================================
    print(f"\n{'='*80}")
    print("RAW vs PCT COMPARISON: 未过滤大亏和大赚的区分能力")

    for label, uf_sub, col_name in [
        ("unfiltered big_losses", uf_bl, reg_col),
        ("unfiltered big_gains", uf_bg, reg_col),
    ]:
        print(f"\n  {label} ({col_name}):")
        s = uf_sub[col_name].dropna()
        print(f"    mean={s.mean():.4f} median={s.median():.4f} "
              f"p75={s.quantile(0.75):.4f} p90={s.quantile(0.90):.4f}")

    for label, uf_sub, col_name in [
        ("unfiltered big_losses", uf_bl, pct_col),
        ("unfiltered big_gains", uf_bg, pct_col),
    ]:
        print(f"\n  {label} ({col_name}):")
        s = uf_sub[col_name].dropna()
        print(f"    mean={s.mean():.4f} median={s.median():.4f} "
              f"p75={s.quantile(0.75):.4f} p90={s.quantile(0.90):.4f}")

    # Which threshold catches most big_loss while minimizing big_gain kills?
    print(f"\n  PCT threshold sweep (unfiltered only):")
    for t in pct_thresholds:
        bl_caught = int((uf_bl[pct_col] >= t).sum())
        bg_killed = int((uf_bg[pct_col] >= t).sum())
        if bl_caught > 0:
            print(f"    reg_pct>={t:.2f}: catches {bl_caught}/{len(uf_bl)} big_loss, kills {bg_killed}/{len(uf_bg)} big_gain")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
