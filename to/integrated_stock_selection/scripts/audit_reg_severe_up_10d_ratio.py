from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"

METRICS_FILE = RESULTS / "walkforward_metrics_20260520_210454.json"
SELECTIONS_FILE = RESULTS / "walkforward_selections_20260520_210454.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.json"

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
        "incremental": bucket_counts(inc_df),
        "overlap_current": int((cand & cur).sum()),
        "candidate_hit_codes": hit.loc[:, ["source", "date", "stock_code", "actual_return", "reg_severe_up_10d_ratio"]]
        .sort_values("reg_severe_up_10d_ratio", ascending=False)
        .head(12)
        .to_dict("records"),
    }


def eval_mask(df: pd.DataFrame, mask: pd.Series) -> dict:
    mask = mask.fillna(False)
    cur = df["current_v3"].fillna(False)
    inc = mask & ~cur
    rows = df[mask]
    inc_rows = df[inc]
    return {
        "standalone": bucket_counts(rows),
        "incremental": bucket_counts(inc_rows),
        "overlap_current": int((mask & cur).sum()),
        "inc_actual_big_loss": int(((inc_rows["source"] == "actual") & (inc_rows["actual_return"] <= -0.04)).sum()),
        "inc_actual_big_gain": int(((inc_rows["source"] == "actual") & (inc_rows["actual_return"] >= 0.08)).sum()),
        "inc_shadow_big_loss": int(((inc_rows["source"] == "shadow") & (inc_rows["actual_return"] <= -0.04)).sum()),
        "inc_shadow_big_gain": int(((inc_rows["source"] == "shadow") & (inc_rows["actual_return"] >= 0.08)).sum()),
        "inc_examples": inc_rows.loc[
            :,
            ["source", "date", "stock_code", "actual_return", "reg_severe_up_10d_ratio",
             "market_advance_decline_ratio", "top10_10d_strength_level_v3", "market_liquidity_change"],
        ]
        .sort_values("actual_return")
        .head(8)
        .to_dict("records"),
        "inc_big_gain_examples": inc_rows.loc[inc_rows["actual_return"] >= 0.08, [
            "source", "date", "stock_code", "actual_return", "reg_severe_up_10d_ratio",
            "market_advance_decline_ratio", "top10_10d_strength_level_v3", "market_liquidity_change",
        ]]
        .sort_values("actual_return", ascending=False)
        .head(8)
        .to_dict("records"),
    }


def scan_regime_combos(df: pd.DataFrame) -> pd.DataFrame:
    reg = df["reg_severe_up_10d_ratio"]
    adr = df["market_advance_decline_ratio"]
    top10 = df["top10_10d_strength_level_v3"]
    liq = df["market_liquidity_change"]
    disp = df["mkt_dispersion_shock_v2"]

    specs = []
    reg_thresholds = [0.50, 0.55, 0.60, 0.621359, 0.65, 0.70, 0.75, 0.80]
    adr_low = [0.35, 0.38, 0.45, 0.50, 0.60]
    adr_high = [1.5, 2.0, 3.0, 4.0]
    top_high = [0.55, 0.60, 0.65, 0.70, 0.75]
    liq_low = [0.90, 0.94, 0.98, 1.00]
    disp_low = [0.90, 0.98, 1.05]

    for r in reg_thresholds:
        base = reg >= r
        for a in adr_low:
            specs.append((f"reg>={r:.3f} & adr<={a:.2f}", base & (adr <= a)))
        for a in adr_high:
            specs.append((f"reg>={r:.3f} & adr>={a:.2f}", base & (adr >= a)))
        for t in top_high:
            specs.append((f"reg>={r:.3f} & top10>={t:.2f}", base & (top10 >= t)))
        for l in liq_low:
            specs.append((f"reg>={r:.3f} & liq<{l:.2f}", base & (liq < l)))
        for d in disp_low:
            specs.append((f"reg>={r:.3f} & disp<{d:.2f}", base & (disp < d)))
        for a in adr_low:
            for t in top_high:
                specs.append((f"reg>={r:.3f} & adr<={a:.2f} & top10>={t:.2f}", base & (adr <= a) & (top10 >= t)))
        for l in liq_low:
            for t in top_high:
                specs.append((f"reg>={r:.3f} & liq<{l:.2f} & top10>={t:.2f}", base & (liq < l) & (top10 >= t)))
        for a in adr_low:
            for l in liq_low:
                specs.append((f"reg>={r:.3f} & adr<={a:.2f} & liq<{l:.2f}", base & (adr <= a) & (liq < l)))
        for a in adr_low:
            for d in disp_low:
                specs.append((f"reg>={r:.3f} & adr<={a:.2f} & disp<{d:.2f}", base & (adr <= a) & (disp < d)))

    rows = []
    for name, mask in specs:
        stats = eval_mask(df, mask)
        inc = stats["incremental"]
        if inc["n"] == 0:
            continue
        row = {
            "rule": name,
            "standalone_n": stats["standalone"]["n"],
            "standalone_big_loss": stats["standalone"]["big_loss<=-4%"],
            "standalone_big_gain": stats["standalone"]["big_gain>=8%"],
            "inc_n": inc["n"],
            "inc_mean": inc["mean"],
            "inc_big_loss": inc["big_loss<=-4%"],
            "inc_small_loss": inc["small_loss"],
            "inc_small_gain": inc["small_gain3-8%"],
            "inc_big_gain": inc["big_gain>=8%"],
            "inc_big_gain10": inc["big_gain>=10%"],
            "inc_actual_big_loss": stats["inc_actual_big_loss"],
            "inc_actual_big_gain": stats["inc_actual_big_gain"],
            "inc_shadow_big_loss": stats["inc_shadow_big_loss"],
            "inc_shadow_big_gain": stats["inc_shadow_big_gain"],
            "overlap_current": stats["overlap_current"],
            "score": inc["big_loss<=-4%"] - 2.5 * inc["big_gain>=8%"] - 0.5 * inc["small_gain3-8%"],
            "examples": stats["inc_examples"],
            "big_gain_examples": stats["inc_big_gain_examples"],
        }
        rows.append(row)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values(["score", "inc_big_loss", "inc_big_gain", "inc_n"], ascending=[False, False, True, True])


def scan_two_factor_reg_rules(df: pd.DataFrame) -> pd.DataFrame:
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
            return series <= value
        if op == "<":
            return series < value
        if op == ">=":
            return series >= value
        if op == ">":
            return series > value
        if op == "==":
            return series == value
        raise ValueError(op)

    rows = []
    reg_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for r in reg_thresholds:
        reg_mask = reg >= r
        for factor, tests in candidates.items():
            if factor not in df.columns:
                continue
            for op, value in tests:
                mask = reg_mask & compare(df[factor], op, value)
                stats = eval_mask(df, mask)
                inc = stats["incremental"]
                if inc["n"] == 0:
                    continue
                rows.append({
                    "rule": f"reg>={r:.2f} & {factor}{op}{value:g}",
                    "reg_threshold": r,
                    "factor": factor,
                    "op": op,
                    "factor_threshold": value,
                    "standalone_n": stats["standalone"]["n"],
                    "standalone_big_loss": stats["standalone"]["big_loss<=-4%"],
                    "standalone_big_gain": stats["standalone"]["big_gain>=8%"],
                    "inc_n": inc["n"],
                    "inc_mean": inc["mean"],
                    "inc_big_loss": inc["big_loss<=-4%"],
                    "inc_small_loss": inc["small_loss"],
                    "inc_small_gain": inc["small_gain3-8%"],
                    "inc_big_gain": inc["big_gain>=8%"],
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
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res.sort_values(["score", "inc_big_loss", "inc_big_gain", "inc_n"], ascending=[False, False, True, True])


def main() -> None:
    selections = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(["source", "date", "stock_code", "actual_return"])

    schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
    factor_cols = ["date", "stock_code", *[c for c in RULE_COLS if c in schema_cols]]
    factors = pd.read_parquet(FACTORS, columns=factor_cols)
    for col in RULE_COLS:
        if col not in factors.columns:
            factors[col] = np.nan
    factors["date"] = pd.to_datetime(factors["date"])
    events = add_prev_factor_snapshot(events, factors)
    masks = current_v3_masks(events)
    events = pd.concat([events, masks], axis=1)
    events["bucket"] = add_bucket(events)

    print("FILES")
    print(METRICS_FILE)
    print(SELECTIONS_FILE)
    print(SHADOW_FILE)
    print("\nDATASET")
    print(events.groupby("source").size().to_string())
    print("missing reg factor:", int(events["reg_severe_up_10d_ratio"].isna().sum()))
    print("\nBASE BUCKETS")
    for source, group in events.groupby("source"):
        print(source, bucket_counts(group))
    print("combined", bucket_counts(events))
    print("\nCURRENT V3")
    for source, group in events.groupby("source"):
        print(source, "current hits", int(group["current_v3"].sum()), bucket_counts(group[group["current_v3"]]))
    print("combined current hits", int(events["current_v3"].sum()), bucket_counts(events[events["current_v3"]]))

    print("\nREG FACTOR DISTRIBUTION ON EVENTS")
    print(events["reg_severe_up_10d_ratio"].describe(percentiles=[.5, .75, .8, .85, .9, .95, .98, .99]).to_string())
    for source, group in events.groupby("source"):
        print("\n", source)
        print(group["reg_severe_up_10d_ratio"].describe(percentiles=[.5, .75, .8, .85, .9, .95, .98, .99]).to_string())
    print("\nREG FACTOR BY BUCKET")
    print(
        events.groupby(["source", "bucket"])["reg_severe_up_10d_ratio"]
        .describe(percentiles=[.5, .75, .9])
        .to_string()
    )

    thresholds = sorted(set([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30] + [
        round(float(q), 6)
        for q in events["reg_severe_up_10d_ratio"].dropna().quantile([.75, .8, .85, .9, .95, .98, .99]).tolist()
    ]))
    print("\nTHRESHOLDS")
    for t in thresholds:
        print(json.dumps(summarize_threshold(events, t), ensure_ascii=False, default=str))

    combo_scan = scan_regime_combos(events)
    print("\nCOMBO SCAN TOP BY SCORE")
    if not combo_scan.empty:
        show_cols = [
            "rule", "inc_n", "inc_mean", "inc_big_loss", "inc_small_loss", "inc_small_gain",
            "inc_big_gain", "inc_big_gain10", "inc_actual_big_loss", "inc_actual_big_gain",
            "inc_shadow_big_loss", "inc_shadow_big_gain", "overlap_current", "score",
        ]
        print(combo_scan.head(40).loc[:, show_cols].to_string(index=False))
        print("\nCOMBO SCAN ZERO BIG-GAIN WITH BIG-LOSS")
        zbg = combo_scan[(combo_scan["inc_big_gain"] == 0) & (combo_scan["inc_big_loss"] > 0)]
        print(zbg.head(30).loc[:, show_cols].to_string(index=False) if not zbg.empty else "NONE")
        print("\nCOMBO SCAN LOW FALSE-KILL CANDIDATES")
        low_fk = combo_scan[(combo_scan["inc_big_loss"] >= 1) & (combo_scan["inc_big_gain"] <= 1)]
        print(low_fk.head(30).loc[:, show_cols].to_string(index=False) if not low_fk.empty else "NONE")
        for _, row in combo_scan.head(8).iterrows():
            print("\nCOMBO_DETAIL", row["rule"])
            print("examples", json.dumps(row["examples"], ensure_ascii=False, default=str))
            print("big_gain_examples", json.dumps(row["big_gain_examples"], ensure_ascii=False, default=str))
    else:
        print("NONE")

    two_factor_scan = scan_two_factor_reg_rules(events)
    print("\nTWO FACTOR REG RULES")
    if not two_factor_scan.empty:
        show_cols = [
            "rule", "inc_n", "inc_mean", "inc_big_loss", "inc_small_loss", "inc_small_gain",
            "inc_big_gain", "inc_big_gain10", "inc_actual_big_loss", "inc_actual_big_gain",
            "inc_shadow_big_loss", "inc_shadow_big_gain", "overlap_current", "score",
        ]
        print(two_factor_scan.head(50).loc[:, show_cols].to_string(index=False))
        print("\nTWO FACTOR ZERO BIG-GAIN WITH BIG-LOSS")
        zbg = two_factor_scan[(two_factor_scan["inc_big_gain"] == 0) & (two_factor_scan["inc_big_loss"] > 0)]
        print(zbg.head(50).loc[:, show_cols].to_string(index=False) if not zbg.empty else "NONE")
        for _, row in two_factor_scan.head(10).iterrows():
            print("\nTWO_FACTOR_DETAIL", row["rule"])
            print("examples", json.dumps(row["examples"], ensure_ascii=False, default=str))
            print("big_gain_examples", json.dumps(row["big_gain_examples"], ensure_ascii=False, default=str))
    else:
        print("NONE")

    print("\nTOP BIG LOSSES")
    cols = ["source", "date", "stock_code", "actual_return", "current_v3", "reg_severe_up_10d_ratio", "factor_date"]
    print(events.loc[events["actual_return"] <= -0.04, cols].sort_values("actual_return").to_string(index=False))
    print("\nTOP BIG GAINS")
    print(events.loc[events["actual_return"] >= 0.08, cols].sort_values("actual_return", ascending=False).head(80).to_string(index=False))


if __name__ == "__main__":
    main()
