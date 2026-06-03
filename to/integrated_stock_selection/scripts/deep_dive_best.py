"""Deep-dive into the best price_ma5_ratio_raw combos found by full scan."""
import json, pandas as pd, numpy as np, pyarrow.parquet as pq
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
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

actual = load_all(f"walkforward_selections_{TIMESTAMP}.json", "actual")
shadow = load_all(f"walkforward_shadow_skipped_opportunities_{TIMESTAMP}.json", "shadow")
events = pd.concat([actual, shadow], ignore_index=True)
events.drop_duplicates(["source","date","stock_code","actual_return"], inplace=True)

V3 = ["runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","market_advance_decline_ratio",
    "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9"]
EXTRA = ["price_ma5_ratio_raw","ma_convergence_score","alpha_002",
    "ma_alignment_persistence_5d","ppo_strength_pct"]
ALL = V3 + EXTRA

schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
use = ["date","stock_code"] + [c for c in ALL if c in schema_cols]
factors = pd.read_parquet(FACTORS, columns=use)
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
for c in ALL:
    if c not in events.columns: events[c] = np.nan

r = events; N = len(r)
def col(name):
    if name not in r.columns: return np.full(N, np.nan)
    s = r[name]
    if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
    return pd.to_numeric(s, errors="coerce").values

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
rets = col("actual_return")
cand = col("price_ma5_ratio_raw")

# Factor distributions
print("Factor distributions on events:")
for fac in ["ma_convergence_score","alpha_002","ma_alignment_persistence_5d","ppo_strength_pct"]:
    v = col(fac); vv = v[~np.isnan(v)]
    print(f"  {fac}: mean={vv.mean():.4f} med={np.median(vv):.4f} std={vv.std():.4f} min={vv.min():.4f} max={vv.max():.4f}")
    for bkt, cond in [("BL", rets<=-0.04), ("BG", rets>=0.08)]:
        sub = v[cond & ~np.isnan(v)]
        print(f"    {bkt}: mean={sub.mean():.4f} med={np.median(sub):.4f}")

# === Best rule: ma_convergence_score ===
print("\n" + "="*70)
print("BEST: price_ma5_ratio_raw >= 9.2 & ma_convergence_score >= 0.1758")
print("="*70)

mcs = col("ma_convergence_score")
mcs_valid = mcs[~np.isnan(mcs)]
for p in [50,75,85,90,95,99]:
    print(f"  ma_convergence_score p{p}: {np.percentile(mcs_valid, p):.4f}")
print(f"  (threshold 0.1758 is around p{int(np.mean(mcs_valid <= 0.1758)*100)})")

mask = (cand >= 9.2) & (mcs >= 0.1758)
inc = mask & ~baseline
inc_idx = np.where(inc)[0]
inc_rets = rets[inc]
print(f"\n  Standalone: {int(mask.sum())}, inc: {len(inc_idx)}, overlap: {int((mask & baseline).sum())}")
print(f"  inc big_loss: {int((inc_rets<=-0.04).sum())}, big_gain: {int((inc_rets>=0.08).sum())}")
print(f"  inc small_gain(3-8%): {int(((inc_rets>=0.03)&(inc_rets<0.08)).sum())}")
print(f"  inc mean: {inc_rets.mean():+.4f}")

print("\n  All incremental hits:")
for idx in inc_idx[np.argsort(rets[inc_idx])]:
    tag = ""
    if rets[idx] <= -0.04: tag = " *** BIG LOSS"
    elif rets[idx] >= 0.08: tag = " *** BIG GAIN"
    elif rets[idx] >= 0.03: tag = " (small gain)"
    print(f"    {r.iloc[idx]['source']:6s} {pd.Timestamp(r.iloc[idx]['date']).date()} {r.iloc[idx]['stock_code']:12s} ret={rets[idx]:+.4f} ma5={cand[idx]:.2f} mcs={mcs[idx]:.4f}{tag}")

print("\n  Threshold stability:")
for raw_t in [7, 8, 9, 9.2, 9.5, 10, 11, 12]:
    for mcs_t in [0.10, 0.15, 0.1758, 0.20, 0.23, 0.25]:
        m = (cand >= raw_t) & (mcs >= mcs_t)
        i = m & ~baseline
        ir = rets[i]
        bl = int((ir<=-0.04).sum())
        bg = int((ir>=0.08).sum())
        if bl >= 2 and bg == 0:
            print(f"    raw>={raw_t} & mcs>={mcs_t:.4f}: inc_n={i.sum()} bl={bl} bg=0")

# === alpha_002 ===
print("\n" + "="*70)
print("RUNNER-UP: price_ma5_ratio_raw & alpha_002")
print("="*70)
a2 = col("alpha_002")
a2v = a2[~np.isnan(a2)]
print(f"  alpha_002: mean={a2v.mean():.4f} med={np.median(a2v):.4f}")
for p in [10,25,50,75,90]:
    print(f"  p{p}: {np.percentile(a2v, p):.4f}")

for raw_t in [0,1,2,3,5,7,9,10]:
    for a2_t in [-1.5, -1.167, -1.0, -0.8, -0.5]:
        m = (cand >= raw_t) & (a2 <= a2_t)
        i = m & ~baseline
        ir = rets[i]
        bl = int((ir<=-0.04).sum())
        bg = int((ir>=0.08).sum())
        if bl >= 2 and bg == 0:
            print(f"  raw>={raw_t} & alpha_002<={a2_t}: inc_n={i.sum()} bl={bl} bg=0")
            i_idx = np.where(i)[0]
            for idx in i_idx[np.argsort(rets[i_idx])]:
                print(f"    {r.iloc[idx]['source']:6s} {pd.Timestamp(r.iloc[idx]['date']).date()} {r.iloc[idx]['stock_code']:12s} ret={rets[idx]:+.4f}")

# === ma_alignment_persistence_5d ===
print("\n" + "="*70)
print("RUNNER-UP: price_ma5_ratio_raw & ma_alignment_persistence_5d")
print("="*70)
ap = col("ma_alignment_persistence_5d")
apv = ap[~np.isnan(ap)]
print(f"  mean={apv.mean():.4f} med={np.median(apv):.4f} min={apv.min():.4f} max={apv.max():.4f}")

for raw_t in [7,8,9,9.2,10,11]:
    for ap_t in sorted(set([round(float(p), 4) for p in np.percentile(apv, [50,70,80,85,90,95])])):
        m = (cand >= raw_t) & (ap <= ap_t)
        i = m & ~baseline
        ir = rets[i]
        bl = int((ir<=-0.04).sum())
        bg = int((ir>=0.08).sum())
        if bl >= 2 and bg == 0:
            print(f"  raw>={raw_t} & ap<={ap_t:.4f}: inc_n={i.sum()} bl={bl} bg=0")
            i_idx = np.where(i)[0]
            for idx in i_idx[np.argsort(rets[i_idx])]:
                print(f"    {r.iloc[idx]['source']:6s} {pd.Timestamp(r.iloc[idx]['date']).date()} {r.iloc[idx]['stock_code']:12s} ret={rets[idx]:+.4f}")

print(f"\n{'='*70}")
print(f"Unfiltered: {int((~baseline).sum())} ({int((rets[~baseline]<=-0.04).sum())} BL, {int((rets[~baseline]>=0.08).sum())} BG)")
