"""全量扫描: price_ma5_ratio_raw 与 factors.parquet 中所有其他因子组合。

对每个因子，基于其在事件上的分布自动生成阈值，尝试 >= 和 <= 两个方向。
只报告 0 误杀大涨 且 增量大亏 >= 2 的规则。
"""
import json, pandas as pd, numpy as np, pyarrow.parquet as pq
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"; FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
TIMESTAMP = "20260520_210454"

# Load events
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

# Load all factors
schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
V3 = ["runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","market_advance_decline_ratio",
    "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9"]

CANDIDATE = "price_ma5_ratio_raw"

# All other factors to try
ALL_OTHER = sorted([c for c in schema_cols if c not in {"date","stock_code"} and c != CANDIDATE])

print(f"Scanning {CANDIDATE} x {len(ALL_OTHER)} other factors...")

# Load all at once
use = ["date","stock_code", CANDIDATE] + ALL_OTHER
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
# Remove duplicate columns from merge
events = events.loc[:, ~events.columns.duplicated()]
for c in ALL_OTHER + [CANDIDATE] + V3:
    if c not in events.columns:
        events[c] = np.nan

r = events
N = len(r)

# v3 baseline
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
cand = col(CANDIDATE)

print(f"Baseline r1-r8: {int(baseline.sum())}/{N} filtered, {int((~baseline).sum())} unfiltered")

# Thresholds for candidate (use percentiles)
cand_valid = cand[~np.isnan(cand)]
MA5_TS = [round(float(p), 1) for p in np.percentile(cand_valid, [25,40,50,60,70,75,80,85,90,95])]
MA5_TS = sorted(set(MA5_TS + [0, 1, 2, 3, 5, 7, 10, 12]))

def gen_ts(vals, n=6):
    """Generate thresholds from percentile distribution."""
    v = vals[~np.isnan(vals)]
    if len(v) < 10: return [round(float(np.median(v)), 4)]
    ps = np.percentile(v, np.linspace(10, 95, n))
    return sorted(set([round(float(p), 4) for p in ps]))

def compare(arr, op, v):
    s = np.array(arr, dtype=float).copy()
    s[np.isnan(s)] = -999 if op in (">=", ">") else 999
    if op == "<=": return s <= v
    if op == "<": return s < v
    if op == ">=": return s >= v
    if op == ">": return s > v
    raise ValueError(op)

good_rules = []  # bl >= 2, bg == 0
ok_rules = []    # bl == 1, bg == 0

# Scan
total = len(ALL_OTHER)
for i, other_fac in enumerate(ALL_OTHER):
    if (i+1) % 20 == 0:
        print(f"  Progress: {i+1}/{total}, found {len(good_rules)} good, {len(ok_rules)} ok")

    other_vals = col(other_fac)
    other_valid = other_vals[~np.isnan(other_vals)]
    if len(other_valid) < 5: continue

    ots = gen_ts(other_valid, n=8)

    for mt in MA5_TS:
        base = cand >= mt
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
                    label = f"{CANDIDATE}>={mt} & {other_fac}{op}{ot:.4f}"
                    good_rules.append({
                        "rule": label, "inc_n": inc_n, "bl": bl, "bg": bg, "sg": sg,
                        "inc_mean": float(inc_rets.mean()),
                        "overlap": int((mask & baseline).sum()),
                        "other_fac": other_fac,
                    })
                elif bl == 1 and bg == 0:
                    label = f"{CANDIDATE}>={mt} & {other_fac}{op}{ot:.4f}"
                    ok_rules.append({
                        "rule": label, "inc_n": inc_n, "bl": bl, "bg": bg, "sg": sg,
                        "inc_mean": float(inc_rets.mean()),
                        "overlap": int((mask & baseline).sum()),
                        "other_fac": other_fac,
                    })

# Results
print(f"\n{'='*70}")
print(f"RESULTS: {CANDIDATE} x {total} factors")
print(f"{'='*70}")

good_rules.sort(key=lambda x: (-x["bl"], x["sg"], -x["overlap"]))
ok_rules.sort(key=lambda x: (-x["bl"], x["sg"], -x["overlap"]))

if good_rules:
    print(f"\n*** GOOD RULES (bl>=2, bg=0): {len(good_rules)} found ***")
    for res in good_rules[:30]:
        print(f"  {res['rule']:75s} n={res['inc_n']:2d} bl={res['bl']} bg=0 sg={res['sg']} mean={res['inc_mean']:+.4f} ov={res['overlap']:3d}")
else:
    print(f"\n*** NO RULES with bl>=2 and bg=0 found ***")

if ok_rules:
    print(f"\n--- OK RULES (bl=1, bg=0): {len(ok_rules)} found ---")
    # Group by other_fac
    from collections import Counter
    fac_counts = Counter(r["other_fac"] for r in ok_rules)
    print(f"  Factors with bl=1 bg=0 combinations:")
    for fac, cnt in fac_counts.most_common(20):
        rules_for_fac = [r for r in ok_rules if r["other_fac"] == fac]
        best = rules_for_fac[0]
        print(f"    {fac:50s} {cnt:3d} combos  best: n={best['inc_n']} bl=1 ov={best['overlap']}")

print(f"\nUnfiltered: {int((~baseline).sum())} ({int((rets[~baseline]<=-0.04).sum())} big_loss, {int((rets[~baseline]>=0.08).sum())} big_gain)")
