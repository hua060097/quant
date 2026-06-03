"""用 factors.parquet 中实际存储的 price_ma5_ratio_raw_pct 重新验证 r11 阈值。"""
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

FACTOR = "price_ma5_ratio_raw_pct"

def load_json_records(path):
    obj = json.load(open(path, encoding="utf-8"))
    records = []
    def walk(x):
        if isinstance(x, list):
            for i in x: walk(i)
        elif isinstance(x, dict):
            if "stock_code" in x and ("actual_return" in x or "profit_loss_pct" in x
                    or "buy_datetime" in x or "date" in x or "trade_date" in x):
                records.append(x)
            else:
                for v in x.values(): walk(v)
    walk(obj)
    return pd.DataFrame(records)

def normalize_events(df, source):
    df = df.copy()
    if "date" not in df.columns:
        for c in ["trade_date","buy_date","selection_date","buy_datetime",
                   "shadow_attempt_buy_date","shadow_prediction_date"]:
            if c in df.columns: df["date"] = df[c]; break
    if "stock_code" not in df.columns:
        for c in ["code","ts_code","selected_stock"]:
            if c in df.columns: df["stock_code"] = df[c]; break
    for c in ["actual_return","return","trade_return","future_return",
              "future_return_5d","final_return","realized_return"]:
        if c in df.columns: df["actual_return"] = df[c]; break
    df = df[["date","stock_code","actual_return"]].copy()
    df["source"] = source
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str)
    df["actual_return"] = pd.to_numeric(df.get("actual_return"), errors="coerce")
    return df.dropna(subset=["date","stock_code","actual_return"])

def current_v3_masks(df):
    out = pd.DataFrame(index=df.index)
    out["r1"] = (df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)
    out["r2"] = (
        ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
        | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
        | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15)))
    out["r3"] = (df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40)
    out["r4"] = (df["market_liquidity_change"] < 0.94) & (
        (df["ulcer_index"] < -0.80) | (df["momentum_peak_decay_3d"] > 1.0)
        | (df["intra_sector_momentum_rank"] < 0.67))
    out["r5"] = (df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38)
    out["r6"] = df["limit_down_3d"] == 1
    out["r7"] = (df["top10_10d_strength_level_v3"] >= 0.70) & (df["buy_sm_amount_rate"] <= -1.25)
    out["r8"] = (df["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80) & (df["kdj_j_9"] <= 0.75)
    out["r9"] = df["score_z_in_day"] <= 1.50
    out["r10"] = (df["reg_severe_up_10d_ratio"] >= 0.18) & (df["market_advance_decline_ratio"] <= 0.45)
    out["current_v3"] = out[["r1","r2","r3","r4","r5","r6","r7","r8","r9","r10"]].any(axis=1)
    return out.fillna(False)

def bucket_counts(df):
    r = df["actual_return"]
    return {"n":int(len(df)), "mean":float(r.mean()) if len(df) else np.nan,
            "big_loss":int((r<=-0.04).sum()), "small_loss":int(((r>-0.04)&(r<0)).sum()),
            "small_gain":int(((r>=0.03)&(r<0.08)).sum()), "big_gain8":int((r>=0.08).sum()),
            "big_gain10":int((r>=0.10).sum())}

def main():
    print("Loading...")
    selections = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(
        ["source","date","stock_code","actual_return"])

    schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
    ALL_NEEDED = [
        "runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
        "sector_momentum_accel_rank","sector_leadership_persistence_v3",
        "return_20d_intra_z","market_liquidity_change","ulcer_index",
        "momentum_peak_decay_3d","intra_sector_momentum_rank",
        "mkt_dispersion_shock_v2","market_advance_decline_ratio",
        "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
        "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9",
        "score_z_in_day","reg_severe_up_10d_ratio",FACTOR]
    use = ["date","stock_code"] + [c for c in ALL_NEEDED if c in schema_cols]
    factors = pd.read_parquet(FACTORS, columns=use)
    factors["date"] = pd.to_datetime(factors["date"])
    # Fill missing columns with NaN
    for c in ALL_NEEDED:
        if c not in factors.columns:
            factors[c] = np.nan

    # attach t-1 snapshot
    trading_days = np.array(sorted(factors["date"].dropna().unique()))
    prev_dates = []
    for d in events["date"]:
        td = pd.Timestamp(d).normalize()
        pos = np.searchsorted(trading_days, np.datetime64(td), side="left") - 1
        prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)
    events["factor_date"] = prev_dates
    snap = factors.rename(columns={"date":"factor_date"})
    events = events.merge(snap, on=["factor_date","stock_code"], how="left")
    masks = current_v3_masks(events)
    events = pd.concat([events, masks], axis=1)

    val = events[FACTOR]
    cur = events["current_v3"]
    print(f"Events: {len(events)}, missing {FACTOR}: {int(val.isna().sum())}")
    print(f"\n{FACTOR} distribution: {val.describe(percentiles=[.5,.75,.9,.95,.98,.99]).to_string()}")

    # Scan thresholds around 0.99
    print(f"\n{'='*70}")
    print("THRESHOLD SCAN (using stored factor)")
    print(f"{'T':>6s} {'standalone':>10s} {'inc_n':>5s} {'bl':>3s} {'bg8':>3s} {'sg':>3s} {'sl':>3s} {'mean':>8s} {'overlap':>7s} {'score':>5s}")
    for t in [0.95, 0.97, 0.98, 0.985, 0.988, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]:
        mask = val >= t
        inc = mask & ~cur
        inc_rows = events[inc]
        bl = int((inc_rows["actual_return"] <= -0.04).sum())
        bg = int((inc_rows["actual_return"] >= 0.08).sum())
        sg = int(((inc_rows["actual_return"] >= 0.03) & (inc_rows["actual_return"] < 0.08)).sum())
        sl = int(((inc_rows["actual_return"] > -0.04) & (inc_rows["actual_return"] < 0)).sum())
        score = bl - 3.0*bg - 0.5*sg
        print(f"{t:6.3f} {int(mask.sum()):>10d} {int(inc.sum()):>5d} {bl:>3d} {bg:>3d} {sg:>3d} {sl:>3d} "
              f"{inc_rows['actual_return'].mean() if len(inc_rows)>0 else 0:>+8.4f} "
              f"{int((mask&cur).sum()):>7d} {score:>5.1f}")

    # Detail for key thresholds
    for t in [0.988, 0.99, 0.991, 0.992]:
        mask = val >= t
        inc = mask & ~cur
        inc_rows = events[inc]
        bl_rows = inc_rows[inc_rows["actual_return"] <= -0.04]
        bg_rows = inc_rows[inc_rows["actual_return"] >= 0.08]
        print(f"\n--- Detail: {FACTOR}>={t} ---")
        print(f"  inc_n={len(inc_rows)} big_loss={len(bl_rows)} big_gain={len(bg_rows)}")
        if len(bl_rows):
            print("  Big losses caught:")
            for _, r in bl_rows.sort_values("actual_return").iterrows():
                print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f} pct={r[FACTOR]:.6f}")
        if len(bg_rows):
            print("  Big gains KILLED:")
            for _, r in bg_rows.sort_values("actual_return", ascending=False).iterrows():
                print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f} pct={r[FACTOR]:.6f}")
        else:
            print("  Big gains: NONE killed")

    # Show all unfiltered events with their pct values (sorted)
    uf = events[~cur]
    print(f"\n{'='*70}")
    print("ALL UNFILTERED EVENTS sorted by pct (top 30)")
    cols = ["source","date","stock_code","actual_return",FACTOR,
            "market_advance_decline_ratio"]
    top = uf.sort_values(FACTOR, ascending=False).head(30)
    for _, r in top.iterrows():
        tag = ""
        if r["actual_return"] <= -0.04: tag = " *** BIG LOSS"
        elif r["actual_return"] >= 0.08: tag = " *** BIG GAIN"
        print(f"  {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f} pct={r[FACTOR]:.6f}{tag}")

    print(f"\n{'='*70}")
    print("CUMULATIVE EFFECT")
    uf_bl = uf[uf["actual_return"]<=-0.04]
    uf_bg = uf[uf["actual_return"]>=0.08]
    for t in [0.95,0.97,0.98,0.985,0.988,0.99,0.991,0.992]:
        caught = int((uf_bl[FACTOR]>=t).sum())
        killed = int((uf_bg[FACTOR]>=t).sum())
        print(f"  T>={t:.3f}: catches {caught}/{len(uf_bl)} big_loss, kills {killed}/{len(uf_bg)} big_gain (among unfiltered)")
    print(f"  Total unfiltered: {len(uf_bl)} big_loss, {len(uf_bg)} big_gain")

if __name__ == "__main__":
    main()
