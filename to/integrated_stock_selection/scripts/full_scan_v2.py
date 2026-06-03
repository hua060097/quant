"""
Full scan v2: price_ma5_ratio_raw x ALL factors in factors_非zscore.parquet
Stricter criteria: bl>=2, bg=0, sg=0 (zero small gain kills too)
Also scan with reg_severe_up_10d_ratio as alternative primary factor.
"""
import json, pandas as pd, numpy as np, pyarrow.parquet as pq
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors_非zscore.parquet"
TIMESTAMP = "20260520_210454"

def load_all(fname, label):
    obj = json.load(open(str(RESULTS / fname), encoding="utf-8"))
    records = []
    def walk(x):
        if isinstance(x, list):
            for i in x: walk(i)
        elif isinstance(x, dict):
            if "stock_code" in x and "actual_return" in x: records.append(x)
            else:
                for v in x.values(): walk(v)
    walk(obj)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df.get("buy_datetime", df.get("date", None)))
    df["source"] = label
    return df[["date","stock_code","actual_return","source"]]

print("Loading events...")
actual = load_all(f"walkforward_selections_{TIMESTAMP}.json", "actual")
shadow = load_all(f"walkforward_shadow_skipped_opportunities_{TIMESTAMP}.json", "shadow")
events = pd.concat([actual, shadow], ignore_index=True)
events.drop_duplicates(["source","date","stock_code","actual_return"], inplace=True)

# All factor columns
schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
ALL_FACTORS = sorted([c for c in schema_cols if c not in {"date","stock_code"}])

V3 = ["runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","market_advance_decline_ratio",
    "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9"]

PRIMARY = ["price_ma5_ratio_raw", "reg_severe_up_10d_ratio"]

# All other factors
OTHERS = [c for c in ALL_FACTORS if c not in PRIMARY and c not in V3]
print(f"Non-zscore parquet: {len(ALL_FACTORS)} factors, scanning {len(PRIMARY)} primary x {len(OTHERS)} others")

use = ["date","stock_code"] + V3 + PRIMARY + OTHERS
factors = pd.read_parquet(FACTORS, columns=[c for c in use if c in schema_cols])
factors["date"] = pd.to_datetime(factors["date"])

tds = np.array(sorted(factors["date"].dropna().unique()))
prev = []
for d in events["date"]:
    pos = np.searchsorted(tds, np.datetime64(pd.Timestamp(d).normalize()), side="left") - 1
    prev.append(pd.Timestamp(tds[pos]) if pos >= 0 else pd.NaT)
events["factor_date"] = prev
snap = factors.rename(columns={"date":"factor_date"})
snap = snap.drop_duplicates(["factor_date","stock_code"], keep="last")
events = events.merge(snap, on=["factor_date","stock_code"], how="left")
events = events.loc[:, ~events.columns.duplicated()]

for c in V3 + PRIMARY + OTHERS:
    if c not in events.columns: events[c] = np.nan

r = events; N = len(r)
def col(name):
    if name not in r.columns: return np.full(N, np.nan)
    s = r[name]
    if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
    return pd.to_numeric(s, errors="coerce").values

# Baseline r1-r8
r1 = (col("runup_ma5_exhaustion") > 0.8) & (col("cpv_factor") <= -0.4)
r2 = (((col("prev_day_blowoff_risk") >= 2.0) & (col("sector_momentum_accel_rank") <= 0.30))
    | ((col("prev_day_blowoff_risk") > 5.0) & (col("sector_momentum_accel_rank") <= 0.50))
    | ((col("prev_day_blowoff_risk") > 1.0) & (col("sector_momentum_accel_rank") <= 0.15)))
r3 = (col("sector_leadership_persistence_v3") <= -0.18) & (col("return_20d_intra_z") <= 0.40)
r4 = (col("market_liquidity_change") < 0.94) & ((col("ulcer_index") < -0.80) | (col("momentum_peak_decay_3d") > 1.0) | (col("intra_sector_momentum_rank") < 0.67))
r5 = (col("mkt_dispersion_shock_v2") < 0.98) & (col("market_advance_decline_ratio") < 0.38)
r6 = np.nan_to_num(col("limit_down_3d"), nan=0) == 1
r7 = (col("top10_10d_strength_level_v3") >= 0.70) & (col("buy_sm_amount_rate") <= -1.25)
r8 = (col("prev_non_first_limit_up_performance_percentile_20d") >= 0.80) & (col("kdj_j_9") <= 0.75)
baseline = r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8
rets = col("actual_return")

print(f"Baseline r1-r8: {int(baseline.sum())}/{N} filtered")
print(f"Unfiltered: {int((~baseline).sum())} ({int((rets[~baseline]<=-0.04).sum())} BL, {int((rets[~baseline]>=0.08).sum())} BG)")

def gen_ts(vals, n=10):
    """Generate thresholds from fine-grained percentiles."""
    v = vals[~np.isnan(vals)]
    if len(v) < 10: return [round(float(np.median(v)), 6)]
    ps = np.percentile(v, np.linspace(5, 95, n))
    ts = [round(float(p), 6) for p in ps]
    # Add some key values around median
    med = float(np.median(v))
    for extra in [med * 0.5, med, med * 1.5, med * 2]:
        if extra > v.min() and extra < v.max():
            ts.append(round(extra, 6))
    return sorted(set(ts))

def compare(arr, op, v):
    s = np.array(arr, dtype=float).copy()
    s[np.isnan(s)] = -999 if op in (">=", ">") else 999
    if op == "<=": return s <= v
    if op == "<": return s < v
    if op == ">=": return s >= v
    if op == ">": return s > v
    raise ValueError(op)

good_rules = []  # bl>=2, bg=0, sg=0
ok_rules = []    # bl>=2, bg=0, sg<=1

total = len(OTHERS) * len(PRIMARY)
progress = 0
for primary_fac in PRIMARY:
    cand = col(primary_fac)
    cand_v = cand[~np.isnan(cand)]
    if len(cand_v) < 10: continue
    # Primary thresholds: percentiles
    cand_ts = gen_ts(cand_v, n=12)

    for other_fac in OTHERS:
        progress += 1
        if progress % 200 == 0:
            print(f"  ... {progress}/{total}, good={len(good_rules)}, ok={len(ok_rules)}")

        other_vals = col(other_fac)
        other_v = other_vals[~np.isnan(other_vals)]
        if len(other_v) < 5: continue

        ots = gen_ts(other_v, n=10)

        for ct in cand_ts:
            base = cand >= ct
            if base.sum() < 2: continue
            for ot in ots:
                for op in [">=", "<="]:
                    mask = base & compare(other_vals, op, ot)
                    inc = mask & ~baseline
                    inc_n = inc.sum()
                    if inc_n == 0: continue
                    inc_rets = rets[inc]
                    bl = int((inc_rets <= -0.04).sum())
                    bg = int((inc_rets >= 0.08).sum())
                    sg = int(((inc_rets >= 0.03) & (inc_rets < 0.08)).sum())

                    if bl >= 2 and bg == 0:
                        label = f"{primary_fac}>={ct:.4f} & {other_fac}{op}{ot:.4f}"
                        entry = {"rule": label, "inc_n": inc_n, "bl": bl, "bg": bg, "sg": sg,
                                 "inc_mean": float(inc_rets.mean()),
                                 "overlap": int((mask & baseline).sum()),
                                 "primary": primary_fac, "other": other_fac}
                        if sg == 0:
                            good_rules.append(entry)
                        else:
                            ok_rules.append(entry)

# Results
good_rules.sort(key=lambda x: (-x["bl"], -x["inc_n"]))
ok_rules.sort(key=lambda x: (-x["bl"], x["sg"], -x["inc_n"]))

print(f"\n{'='*70}")
print(f"STRICT RULES (bl>=2, bg=0, sg=0): {len(good_rules)} found")
print(f"{'='*70}")
for res in good_rules[:20]:
    print(f"  {res['rule']:90s} n={res['inc_n']:2d} bl={res['bl']} ov={res['overlap']:3d}")

if not good_rules:
    print("  NONE FOUND with strict sg=0")

print(f"\n{'='*70}")
print(f"RELAXED RULES (bl>=2, bg=0, sg<=1): top 20 of {len(ok_rules)}")
print(f"{'='*70}")
for res in ok_rules[:20]:
    print(f"  {res['rule']:90s} n={res['inc_n']:2d} bl={res['bl']} sg={res['sg']} ov={res['overlap']:3d}")

# Group by other factor for ok rules
from collections import Counter
if ok_rules:
    print(f"\n  Best factors (most strict rules):")
    fac_bl = {}
    for r in ok_rules:
        k = r["other"]
        if k not in fac_bl or fac_bl[k][1] < r["bl"]:
            fac_bl[k] = (r["bl"], r["sg"], r["rule"])
    for fac, (bl, sg, rule) in sorted(fac_bl.items(), key=lambda x: (-x[1][0], x[1][1]))[:15]:
        print(f"    {fac:50s} bl={bl} sg={sg}")

print(f"\nUnfiltered: {int((~baseline).sum())} ({int((rets[~baseline]<=-0.04).sum())} BL, {int((rets[~baseline]>=0.08).sum())} BG)")
