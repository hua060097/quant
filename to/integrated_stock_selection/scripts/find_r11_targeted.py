"""targeted scan: price_ma5_ratio_raw + user-specified factors.

User-specified factors to try with price_ma5_ratio_raw:
  reg_severe_up_10d_ratio_pct, sector_turnover_change_5d_20d,
  market_volatility_context, market_60d_new_high_ratio

Thresholds derived from actual event distribution percentiles.
"""
import json, pandas as pd, numpy as np, pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
TIMESTAMP = "20260520_210454"

# ===== Load events =====
def load_all(fname, label):
    obj = json.load(open(str(RESULTS / fname), encoding="utf-8"))
    records = []
    def walk(x):
        if isinstance(x, list):
            for i in x: walk(i)
        elif isinstance(x, dict):
            if "stock_code" in x and "actual_return" in x:
                records.append(x)
            else:
                for v in x.values(): walk(v)
    walk(obj)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df.get("buy_datetime", df.get("date", None)))
    df["source"] = label
    return df[["date","stock_code","actual_return","source"]]

actual = load_all(f"walkforward_selections_{TIMESTAMP}.json", "actual")
shadow = load_all(f"walkforward_shadow_skipped_opportunities_{TIMESTAMP}.json", "shadow")
events = pd.concat([actual, shadow], ignore_index=True)
events.drop_duplicates(["source","date","stock_code","actual_return"], inplace=True)

# ===== Load factors =====
schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
V3 = ["runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","market_advance_decline_ratio",
    "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9"]

# Primary candidate + user-specified combo factors
CANDIDATE = "price_ma5_ratio_raw"
COMBO_FACTORS = [
    "reg_severe_up_10d_ratio_pct",
    "sector_turnover_change_5d_20d",
    "market_volatility_context",
    "market_60d_new_high_ratio",
    "market_advance_decline_ratio",  # keep as reference
    "buy_sm_amount_rate",           # keep as reference
]

ALL_NEEDED = V3 + [CANDIDATE] + COMBO_FACTORS
use = ["date","stock_code"] + [c for c in ALL_NEEDED if c in schema_cols]
factors = pd.read_parquet(FACTORS, columns=use)
factors["date"] = pd.to_datetime(factors["date"])

trading_days = np.array(sorted(factors["date"].dropna().unique()))
prev_dates = []
for d in events["date"]:
    td = pd.Timestamp(d).normalize()
    pos = np.searchsorted(trading_days, np.datetime64(td), side="left") - 1
    prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)
events["factor_date"] = prev_dates
snap = factors.rename(columns={"date":"factor_date"})
# Ensure no duplicate keys in snap (one row per factor_date+stock_code)
snap = snap.drop_duplicates(["factor_date","stock_code"], keep="last")
events = events.merge(snap, on=["factor_date","stock_code"], how="left")
# Fill missing columns AFTER merge
for c in ALL_NEEDED:
    if c not in events.columns:
        events[c] = np.nan

r = events

# Debug: check for duplicate columns
from collections import Counter
dupes = [c for c, n in Counter(r.columns).items() if n > 1]
if dupes:
    print(f"WARNING: duplicate columns: {dupes}")
    # Drop duplicates by taking first occurrence
    r = r.loc[:, ~r.columns.duplicated()]

# v3 baseline r1-r8 - use .values to avoid index alignment issues
def col(name):
    s = r[name].values if name in r.columns else np.full(len(r), np.nan)
    return pd.to_numeric(pd.Series(s), errors="coerce").values

r1 = (col("runup_ma5_exhaustion") > 0.8) & (col("cpv_factor") <= -0.4)
r2 = (((col("prev_day_blowoff_risk") >= 2.0) & (col("sector_momentum_accel_rank") <= 0.30))
    | ((col("prev_day_blowoff_risk") > 5.0) & (col("sector_momentum_accel_rank") <= 0.50))
    | ((col("prev_day_blowoff_risk") > 1.0) & (col("sector_momentum_accel_rank") <= 0.15)))
r3 = (col("sector_leadership_persistence_v3") <= -0.18) & (col("return_20d_intra_z") <= 0.40)
r4 = (col("market_liquidity_change") < 0.94) & (
    (col("ulcer_index") < -0.80) | (col("momentum_peak_decay_3d") > 1.0)
    | (col("intra_sector_momentum_rank") < 0.67))
r5 = (col("mkt_dispersion_shock_v2") < 0.98) & (col("market_advance_decline_ratio") < 0.38)
r6 = np.nan_to_num(col("limit_down_3d"), nan=0) == 1
r7 = (col("top10_10d_strength_level_v3") >= 0.70) & (col("buy_sm_amount_rate") <= -1.25)
r8 = (col("prev_non_first_limit_up_performance_percentile_20d") >= 0.80) & (col("kdj_j_9") <= 0.75)
baseline = r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8

# ===== Show factor distributions on events =====
print("=" * 70)
print("FACTOR DISTRIBUTIONS ON EVENTS (by return bucket)")
print("=" * 70)
buckets = pd.cut(pd.Series(col("actual_return")),
    bins=[-np.inf, -0.04, 0, 0.03, 0.08, np.inf],
    labels=["big_loss","small_loss","flat","small_gain","big_gain"])

all_factors = [CANDIDATE] + COMBO_FACTORS
for fac in all_factors:
    fvals = col(fac)
    valid = fvals[~np.isnan(fvals)]
    if len(valid) == 0: continue
    print(f"\n--- {fac} ---")
    print(f"  overall: mean={valid.mean():.4f} median={np.median(valid):.4f} "
          f"std={valid.std():.4f} min={valid.min():.4f} max={valid.max():.4f}")
    for bkt in ["big_loss","big_gain"]:
        sub = fvals[buckets.values == bkt]
        sub = sub[~np.isnan(sub)]
        if len(sub) > 0:
            print(f"  {bkt:12s}: mean={sub.mean():.4f} median={np.median(sub):.4f} "
                  f"p25={np.percentile(sub,25):.4f} p75={np.percentile(sub,75):.4f}")
    ps = np.percentile(valid, [10,25,50,75,85,90,95,98,99])
    print(f"  event percentiles: p10={ps[0]:.4f} p25={ps[1]:.4f} p50={ps[2]:.4f} "
          f"p75={ps[3]:.4f} p90={ps[5]:.4f} p95={ps[6]:.4f} p99={ps[8]:.4f}")

# ===== Scan: price_ma5_ratio_raw + each combo factor =====
print("\n" + "=" * 70)
print("TWO-FACTOR SCAN: price_ma5_ratio_raw + each combo factor")
print("Thresholds from distribution percentiles")
print("=" * 70)

# Thresholds for price_ma5_ratio_raw based on event distribution
# mean=2.48, median=2.07, p75=5.7, p90=8.8, p95=11.1 (from earlier analysis)
MA5_THRESHOLDS = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# For each combo factor, generate thresholds based on its distribution
def gen_thresholds(series, name, n=8):
    """Generate meaningful thresholds from distribution."""
    vals = series.dropna()
    ps = vals.quantile(np.linspace(0.1, 0.95, n))
    ts = sorted(set([round(float(p), 4) for p in ps]))
    # Also add some key values
    ts.append(round(float(vals.median()), 4))
    return sorted(set(ts))

def compare(values, op, v):
    s = np.array(values, dtype=float)
    nan_mask = np.isnan(s)
    if op in (">=", ">"): s[nan_mask] = -999
    else: s[nan_mask] = 999
    if op == "<=": return s <= v
    if op == "<": return s < v
    if op == ">=": return s >= v
    if op == ">": return s > v
    raise ValueError(op)

def eval_rule(mask, label):
    inc_mask = mask & ~baseline
    rets = col("actual_return")
    inc_rets = rets[inc_mask]
    inc_n = inc_mask.sum()
    bl = int((inc_rets <= -0.04).sum())
    bg = int((inc_rets >= 0.08).sum())
    sg = int(((inc_rets >= 0.03) & (inc_rets < 0.08)).sum())
    return {"rule":label, "standalone_n":int(mask.sum()), "inc_n":inc_n,
            "inc_mean":float(inc_rets.mean()) if inc_n>0 else 0,
            "inc_bl":bl, "inc_bg":bg, "inc_sg":sg,
            "inc_sl":int(((inc_rets>-0.04)&(inc_rets<0)).sum()),
            "overlap":int((mask & baseline).sum()),
            "score":bl - 5.0*bg - 2.0*sg}

for combo_fac in COMBO_FACTORS:
    cf_vals_arr = col(combo_fac)
    cf_valid = cf_vals_arr[~np.isnan(cf_vals_arr)]
    if len(cf_valid) == 0: continue
    ts = gen_thresholds(pd.Series(cf_valid), combo_fac)

    print(f"\n--- {combo_fac} (event range: {cf_valid.min():.4f} ~ {cf_valid.max():.4f}) ---")

    results = []
    cand_arr = col(CANDIDATE)
    for mt in MA5_THRESHOLDS:
        base = cand_arr >= mt
        if base.sum() == 0: continue
        for ct in ts:
            for op in [">=", "<="]:
                mask = base & compare(col(combo_fac), op, ct)
                res = eval_rule(mask, f"{CANDIDATE}>={mt} & {combo_fac}{op}{ct:.4f}")
                if res["inc_n"] == 0: continue
                results.append(res)

    # Sort: zero-BG first, then by score
    results.sort(key=lambda x: (x["inc_bg"]==0 and x["inc_bl"]>0, x["score"]), reverse=True)

    # Show zero-BG rules
    zbg = [x for x in results if x["inc_bg"] == 0 and x["inc_bl"] > 0]
    if zbg:
        print(f"  ZERO BIG-GAIN ({len(zbg)} found):")
        for res in zbg[:10]:
            print(f"    {res['rule']:65s} n={res['inc_n']:2d} bl={res['inc_bl']} bg=0 sg={res['inc_sg']} sl={res['inc_sl']} mean={res['inc_mean']:+.4f} ov={res['overlap']:3d}")
    else:
        # Show best overall
        print(f"  NO ZERO-BG rules. Best overall:")
        for res in results[:5]:
            print(f"    {res['rule']:65s} n={res['inc_n']:2d} bl={res['inc_bl']} bg={res['inc_bg']} sg={res['inc_sg']} mean={res['inc_mean']:+.4f} ov={res['overlap']:3d}")

    # Detail for best zero-BG
    if zbg:
        best = zbg[0]
        print(f"\n  BEST: {best['rule']}")
        # Parse rule to rebuild mask
        parts = best["rule"].split(" & ")
        mask = np.ones(len(r), dtype=bool)
        for part in parts:
            part = part.strip()
            if part.startswith(CANDIDATE):
                rest = part[len(CANDIDATE):]
                for op in [">=","<="]:
                    if rest.startswith(op):
                        mask &= compare(col(CANDIDATE), op, float(rest[len(op):]))
                        break
            elif part.startswith(combo_fac):
                rest = part[len(combo_fac):]
                for op in [">=","<="]:
                    if rest.startswith(op):
                        mask &= compare(col(combo_fac), op, float(rest[len(op):]))
                        break
        inc_mask = mask & ~baseline
        inc_idx = np.where(inc_mask)[0]
        rets = col("actual_return")
        print(f"    Big losses caught:")
        for idx in inc_idx[np.argsort(rets[inc_idx])]:
            if rets[idx] <= -0.04:
                src = r.iloc[idx]["source"]
                dt = r.iloc[idx]["date"]
                code = r.iloc[idx]["stock_code"]
                print(f"      {src:6s} {pd.Timestamp(dt).date()} {code:12s} ret={rets[idx]:+.4f}  {CANDIDATE}={col(CANDIDATE)[idx]:.2f} {combo_fac}={col(combo_fac)[idx]:.4f}")
        print(f"    Big gains (should be 0):")
        for idx in inc_idx[np.argsort(-rets[inc_idx])]:
            if rets[idx] >= 0.08:
                src = r.iloc[idx]["source"]
                dt = r.iloc[idx]["date"]
                code = r.iloc[idx]["stock_code"]
                print(f"      {src:6s} {pd.Timestamp(dt).date()} {code:12s} ret={rets[idx]:+.4f}")

print(f"\n{'='*70}")
print(f"SUMMARY: baseline r1-r8 hits {int(baseline.sum())}/{len(events)}")
uf_rets = rets[~baseline]
print(f"Unfiltered: {int((~baseline).sum())} ({int((uf_rets<=-0.04).sum())} big_loss, {int((uf_rets>=0.08).sum())} big_gain)")
