from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


METRICS = Path("results/walkforward_metrics_20260524_131700.json")
SHADOW = Path("results/walkforward_shadow_skipped_opportunities_20260524_131700.csv")
FACTORS = Path("factor_lab_data/factors.parquet")


BASE_COLS = [
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
]

CAND_COLS = [
    "prev_non_first_limit_up_performance",
    "prev_first_limit_up_performance",
    "prev_first_limit_up_performance_percentile_20d",
    "market_60d_new_high_ratio",
    "mkt_ma5_followthrough_v3",
    "market_trend_strength",
    "market_sentiment_ad",
    "market_volatility_context",
    "market_liquidity_change",
    "mkt_dispersion_shock_v2",
    "market_advance_decline_ratio",
    "runup_ma5_exhaustion",
    "cpv_factor",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
    "sector_leadership_persistence_v3",
    "return_20d_intra_z",
    "ulcer_index",
    "momentum_peak_decay_3d",
    "intra_sector_momentum_rank",
    "top10_10d_strength_level_v3",
    "buy_sm_amount_rate",
    "kdj_j_9",
    "money_flow_pressure",
    "high_to_close_drawdown",
    "high_to_close_drawdown_pct",
    "price_ma5_ratio_raw",
    "price_ma5_ratio_pct",
    "close_position_60d",
    "stock_volume_ratio_20d",
    "prior_high_volume_contest_20d",
    "open_to_high_ratio",
    "open_to_high_ratio_pct",
    "abnormal_turnover",
    "abnormal_turnover_pct",
    "true_turnover_10d",
    "tightness_3d",
    "post_accumulation_supply_tightness",
    "ret_3d_vs_10d_accel",
    "consecutive_up_days_5d",
    "runup_low_break",
    "low_break_prev_low_after_runup",
    "lg_pos_sm_neg_10d_days",
]


def load_actual() -> pd.DataFrame:
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = []
    for win in data:
        result = win["all_results"]["1"]
        for t in result["trades"]:
            rows.append({
                "event_type": "actual",
                "stock_code": t["stock_code"],
                "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                "buy_datetime": pd.Timestamp(t["buy_datetime"]),
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
            })
    return pd.DataFrame(rows)


def load_shadow() -> pd.DataFrame:
    if not SHADOW.exists():
        return pd.DataFrame()
    df = pd.read_csv(SHADOW)
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "event_type": "shadow",
        "stock_code": df["stock_code"].astype(str),
        "buy_date": pd.to_datetime(df["buy_datetime"].astype(str).str[:10]),
        "buy_datetime": pd.to_datetime(df["buy_datetime"]),
        "actual_return": pd.to_numeric(df["actual_return"], errors="coerce"),
        "profit_loss": pd.to_numeric(df["profit_loss"], errors="coerce"),
    })


def attach_factors(events: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(events["buy_date"]).dt.normalize().unique())
    codes = sorted(events["stock_code"].astype(str).unique())
    dataset = ds.dataset(str(FACTORS), format="parquet")
    table = dataset.to_table(
        columns=["date", "stock_code", *factor_cols],
        filter=ds.field("stock_code").isin(codes),
    )
    fac = table.to_pandas()
    fac["date"] = pd.to_datetime(fac["date"]).dt.normalize()
    fac = fac[fac["date"] <= max(dates)].sort_values(["stock_code", "date"])
    out = []
    for code, grp in events.sort_values(["stock_code", "buy_date"]).groupby("stock_code", sort=False):
        fgrp = fac[fac["stock_code"] == code].sort_values("date")
        if fgrp.empty:
            merged = grp.copy()
            for col in factor_cols:
                merged[col] = np.nan
            merged["factor_date"] = pd.NaT
        else:
            left = grp.sort_values("buy_date").copy()
            left["lookup_date"] = pd.to_datetime(left["buy_date"]).dt.normalize() - pd.Timedelta(days=1)
            merged = pd.merge_asof(
                left,
                fgrp,
                left_on="lookup_date",
                right_on="date",
                by="stock_code",
                direction="backward",
                allow_exact_matches=True,
            )
            merged = merged.rename(columns={"date": "factor_date"})
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    r = {}
    r["r1"] = (df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)
    blow = df["prev_day_blowoff_risk"]
    sec = df["sector_momentum_accel_rank"]
    r["r2"] = ((blow >= 2.0) & (sec <= 0.30)) | ((blow > 5.0) & (sec <= 0.50)) | ((blow > 1.0) & (sec <= 0.15))
    r["r3"] = (df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40)
    r["r4"] = (df["market_liquidity_change"] < 0.94) & (
        (df["ulcer_index"] < -0.80)
        | (df["momentum_peak_decay_3d"] > 1.0)
        | (df["intra_sector_momentum_rank"] < 0.67)
    )
    r["r5"] = (df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38)
    r["r6"] = df["limit_down_3d"].fillna(0).eq(1)
    r["r7"] = (df["top10_10d_strength_level_v3"] >= 0.70) & (df["buy_sm_amount_rate"] <= -1.25)
    r["r8"] = (df["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80) & (df["kdj_j_9"] <= 0.75)
    return r


def stats(df: pd.DataFrame, mask: pd.Series) -> dict:
    sub = df[mask.fillna(False)]
    return {
        "n": int(len(sub)),
        "big_loss8": int((sub["actual_return"] <= -0.08).sum()),
        "big_loss4": int((sub["actual_return"] <= -0.04).sum()),
        "small_loss": int(((sub["actual_return"] > -0.04) & (sub["actual_return"] < 0)).sum()),
        "small_gain": int(((sub["actual_return"] >= 0.03) & (sub["actual_return"] < 0.08)).sum()),
        "big_gain8": int((sub["actual_return"] >= 0.08).sum()),
        "big_gain4": int((sub["actual_return"] >= 0.04).sum()),
        "pnl": float(sub["profit_loss"].sum()) if "profit_loss" in sub else np.nan,
        "codes": ",".join((sub["stock_code"] + "@" + sub["buy_date"].dt.strftime("%Y%m%d")).head(20).tolist()),
    }


def scan_single_thresholds(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    base = df[df["event_type"].isin(["actual", "shadow"])].copy()
    base = base[~base["current_rule_hit"]].copy()
    if scope == "actual":
        base = base[base["event_type"] == "actual"]
    elif scope == "shadow":
        base = base[base["event_type"] == "shadow"]
    for col in CAND_COLS:
        if col not in base.columns:
            continue
        s = pd.to_numeric(base[col], errors="coerce")
        if s.notna().sum() < 10:
            continue
        vals = s.dropna()
        for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            th = float(vals.quantile(q))
            for op in ["<=", ">="]:
                mask = s <= th if op == "<=" else s >= th
                st = stats(base, mask)
                if st["n"] == 0:
                    continue
                rows.append({"scope": scope, "factor": col, "op": op, "q": q, "th": th, **st})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = 6 * out["big_loss8"] + 3 * out["big_loss4"] + out["small_loss"] - 3 * out["small_gain"] - 8 * out["big_gain8"] - 4 * out["big_gain4"] - 0.05 * out["n"]
    return out.sort_values(["score", "big_loss8", "big_loss4", "big_gain8", "small_gain"], ascending=[False, False, False, True, True])


def scan_pair_rules(df: pd.DataFrame) -> pd.DataFrame:
    base = df[~df["current_rule_hit"]].copy()
    thresholds = []
    for col in CAND_COLS:
        if col not in base.columns:
            continue
        s = pd.to_numeric(base[col], errors="coerce")
        if s.notna().sum() < 10:
            continue
        vals = s.dropna()
        for q in [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]:
            thresholds.append((col, "<=", float(vals.quantile(q))))
            thresholds.append((col, ">=", float(vals.quantile(q))))

    rows = []
    for i, a in enumerate(thresholds):
        col1, op1, th1 = a
        s1 = pd.to_numeric(base[col1], errors="coerce")
        m1 = s1 <= th1 if op1 == "<=" else s1 >= th1
        if m1.sum() < 2:
            continue
        for col2, op2, th2 in thresholds[i + 1:]:
            if col1 == col2:
                continue
            s2 = pd.to_numeric(base[col2], errors="coerce")
            m2 = s2 <= th2 if op2 == "<=" else s2 >= th2
            mask = m1 & m2
            st_all = stats(base, mask)
            if st_all["n"] < 2 or st_all["big_loss4"] == 0:
                continue
            st_actual = stats(base[base["event_type"] == "actual"], mask[base["event_type"] == "actual"])
            st_shadow = stats(base[base["event_type"] == "shadow"], mask[base["event_type"] == "shadow"])
            rows.append({
                "rule": f"{col1}{op1}{th1:.4g} & {col2}{op2}{th2:.4g}",
                **{f"all_{k}": v for k, v in st_all.items()},
                **{f"actual_{k}": v for k, v in st_actual.items()},
                **{f"shadow_{k}": v for k, v in st_shadow.items()},
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = (
        8 * out["all_big_loss8"]
        + 4 * out["all_big_loss4"]
        + out["all_small_loss"]
        - 5 * out["all_small_gain"]
        - 10 * out["all_big_gain8"]
        - 4 * out["all_big_gain4"]
        - 0.1 * out["all_n"]
    )
    return out.sort_values(["score", "all_big_loss8", "all_big_loss4", "all_big_gain8", "all_small_gain"], ascending=[False, False, False, True, True])


def main() -> None:
    actual = load_actual()
    shadow = load_shadow()
    events = pd.concat([actual, shadow], ignore_index=True)
    factor_cols = sorted(set(BASE_COLS + CAND_COLS))
    df = attach_factors(events, factor_cols)

    masks = rule_masks(df)
    current = pd.Series(False, index=df.index)
    for m in masks.values():
        current = current | m.fillna(False)
    df["current_rule_hit"] = current

    print("events", len(df), "actual", (df.event_type == "actual").sum(), "shadow", (df.event_type == "shadow").sum())
    print("current filtered", int(current.sum()))
    for scope in ["actual", "shadow", "all"]:
        sub = df if scope == "all" else df[df.event_type == scope]
        remain = sub[~sub.current_rule_hit]
        print("\nSCOPE", scope)
        print("remaining", stats(remain, pd.Series(True, index=remain.index)))
        print("remaining big losses:")
        cols = ["event_type", "stock_code", "buy_date", "actual_return", "profit_loss", *BASE_COLS]
        print(remain[remain.actual_return <= -0.04][cols].sort_values("actual_return").to_string(index=False))

    extra = pd.DataFrame([{
        "event_type": "live_check",
        "stock_code": "300687.SZ",
        "buy_date": pd.Timestamp("2026-05-21"),
        "buy_datetime": pd.Timestamp("2026-05-21 09:30:00"),
        "actual_return": -0.0982,
        "profit_loss": np.nan,
    }])
    live = attach_factors(extra, factor_cols)
    print("\nLIVE 300687 snapshot:")
    print(live[["stock_code", "buy_date", "factor_date", *factor_cols]].to_string(index=False))

    for scope in ["actual", "shadow", "all"]:
        out = scan_single_thresholds(df, scope)
        print("\nTOP SINGLE", scope)
        print(out.head(20).to_string(index=False))

    pairs = scan_pair_rules(df)
    print("\nTOP PAIRS")
    print(pairs.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
