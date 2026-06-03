"""基于185日志(20260520_210454)全面扫描4个因子的组合规则。

候选因子: price_ma5_ratio_raw, price_ma5_ratio_raw_pct,
          reg_severe_up_10d_ratio, reg_severe_up_10d_ratio_pct

目标: 增量大亏拦截 >= 1, 增量大涨误杀 = 0
"""
import json, pandas as pd, numpy as np, pyarrow.parquet as pq
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
TIMESTAMP = "20260520_210454"

# ===== Load events =====
def load_all(source_file, label):
    obj = json.load(open(str(RESULTS / source_file), encoding="utf-8"))
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

# All existing v3 rule factors + the 4 candidates + many additional factors to try
V3_FACTORS = [
    "runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","market_advance_decline_ratio",
    "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9",
]
CANDIDATES = [
    "price_ma5_ratio_raw", "price_ma5_ratio_raw_pct",
    "reg_severe_up_10d_ratio", "reg_severe_up_10d_ratio_pct",
]
# Additional factors to try in combination
EXTRA_FACTORS = []
for c in schema_cols:
    if c in V3_FACTORS or c in CANDIDATES or c in ["date","stock_code"]:
        continue
    EXTRA_FACTORS.append(c)

ALL_NEEDED = V3_FACTORS + CANDIDATES + EXTRA_FACTORS
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
events = events.merge(snap, on=["factor_date","stock_code"], how="left")
for c in ALL_NEEDED:
    if c not in events.columns:
        events[c] = np.nan

r = events

# ===== Existing v3 rules (r1-r8) =====
r["r1"] = (r["runup_ma5_exhaustion"] > 0.8) & (r["cpv_factor"] <= -0.4)
r["r2"] = (((r["prev_day_blowoff_risk"] >= 2.0) & (r["sector_momentum_accel_rank"] <= 0.30))
    | ((r["prev_day_blowoff_risk"] > 5.0) & (r["sector_momentum_accel_rank"] <= 0.50))
    | ((r["prev_day_blowoff_risk"] > 1.0) & (r["sector_momentum_accel_rank"] <= 0.15)))
r["r3"] = (r["sector_leadership_persistence_v3"] <= -0.18) & (r["return_20d_intra_z"] <= 0.40)
r["r4"] = (r["market_liquidity_change"] < 0.94) & (
    (r["ulcer_index"] < -0.80) | (r["momentum_peak_decay_3d"] > 1.0)
    | (r["intra_sector_momentum_rank"] < 0.67))
r["r5"] = (r["mkt_dispersion_shock_v2"] < 0.98) & (r["market_advance_decline_ratio"] < 0.38)
r["r6"] = r["limit_down_3d"].fillna(0) == 1
r["r7"] = (r["top10_10d_strength_level_v3"] >= 0.70) & (r["buy_sm_amount_rate"] <= -1.25)
r["r8"] = (r["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80) & (r["kdj_j_9"] <= 0.75)

baseline = r[["r1","r2","r3","r4","r5","r6","r7","r8"]].any(axis=1)

# ===== Factor combo definitions =====
# For each candidate factor, define threshold sets to try
CANDIDATE_THRESHOLDS = {
    "price_ma5_ratio_raw": [
        (">=", 0.0), (">=", 1.0), (">=", 2.0), (">=", 3.0), (">=", 4.0),
        (">=", 5.0), (">=", 6.0), (">=", 7.0), (">=", 8.0), (">=", 9.0), (">=", 10.0),
        ("<=", -1.0), ("<=", -2.0), ("<=", -3.0), ("<=", -5.0),
    ],
    "price_ma5_ratio_raw_pct": [
        (">=", 0.55), (">=", 0.60), (">=", 0.65), (">=", 0.70),
        (">=", 0.75), (">=", 0.80), (">=", 0.85), (">=", 0.90),
        (">=", 0.92), (">=", 0.95), (">=", 0.97), (">=", 0.98), (">=", 0.99),
    ],
    "reg_severe_up_10d_ratio": [
        (">=", 0.10), (">=", 0.15), (">=", 0.18), (">=", 0.20), (">=", 0.25),
        (">=", 0.30), (">=", 0.35), (">=", 0.40), (">=", 0.45), (">=", 0.50),
        (">=", 0.55), (">=", 0.60), (">=", 0.65), (">=", 0.70),
    ],
    "reg_severe_up_10d_ratio_pct": [
        (">=", 0.55), (">=", 0.60), (">=", 0.65), (">=", 0.70),
        (">=", 0.75), (">=", 0.80), (">=", 0.85), (">=", 0.90),
        (">=", 0.92), (">=", 0.95), (">=", 0.97), (">=", 0.98), (">=", 0.99),
    ],
}

# For combo factors (second/third factors), define threshold sets
COMBO_THRESHOLDS = {
    "market_advance_decline_ratio": [
        ("<=", 0.30), ("<=", 0.35), ("<=", 0.38), ("<=", 0.45), ("<=", 0.50),
        (">=", 1.5), (">=", 2.0), (">=", 3.0),
    ],
    "top10_10d_strength_level_v3": [(">=", 0.55), (">=", 0.60), (">=", 0.65), (">=", 0.70), (">=", 0.75)],
    "market_liquidity_change": [("<", 0.90), ("<", 0.94), ("<", 0.98), ("<", 1.00), (">=", 1.05)],
    "mkt_dispersion_shock_v2": [("<", 0.90), ("<", 0.98), ("<", 1.05)],
    "runup_ma5_exhaustion": [(">", 0.6), (">", 0.7), (">", 0.8)],
    "cpv_factor": [("<=", -0.8), ("<=", -0.6), ("<=", -0.4), (">=", 0.4), (">=", 0.6)],
    "prev_day_blowoff_risk": [(">=", 1.0), (">=", 2.0), (">=", 3.0), (">=", 5.0)],
    "sector_momentum_accel_rank": [("<=", 0.15), ("<=", 0.30), ("<=", 0.50), (">=", 0.70)],
    "sector_leadership_persistence_v3": [("<=", -0.30), ("<=", -0.18), ("<=", 0.0)],
    "return_20d_intra_z": [("<=", 0.0), ("<=", 0.40), ("<=", 0.60), (">=", 1.0)],
    "ulcer_index": [("<", -1.2), ("<", -0.8), ("<", -0.4)],
    "momentum_peak_decay_3d": [(">", 0.5), (">", 1.0), (">", 2.0)],
    "intra_sector_momentum_rank": [("<", 0.30), ("<", 0.50), ("<", 0.67)],
    "buy_sm_amount_rate": [("<=", -1.5), ("<=", -1.25), ("<=", -1.0), (">=", 1.0)],
    "kdj_j_9": [("<=", 0.5), ("<=", 0.75), ("<=", 1.0), (">=", 1.5)],
}

def compare(series, op, value):
    s = series.astype(float)
    if op in ("<=", "<", ">=", ">"): s = s.fillna(-999 if op in (">=", ">") else 999)
    if op == "<=": return s <= value
    if op == "<": return s < value
    if op == ">=": return s >= value
    if op == ">": return s > value
    raise ValueError(op)

def eval_rule(mask, label):
    inc = r[mask & ~baseline]
    bl = int((inc["actual_return"] <= -0.04).sum())
    bg = int((inc["actual_return"] >= 0.08).sum())
    sg = int(((inc["actual_return"] >= 0.03) & (inc["actual_return"] < 0.08)).sum())
    return {
        "rule": label,
        "standalone_n": int(mask.sum()),
        "inc_n": len(inc),
        "inc_mean": float(inc["actual_return"].mean()) if len(inc) > 0 else 0,
        "inc_bl": bl, "inc_bg": bg, "inc_sg": sg,
        "inc_sl": int(((inc["actual_return"] > -0.04) & (inc["actual_return"] < 0)).sum()),
        "overlap": int((mask & baseline).sum()),
        "score": bl - 5.0 * bg - 2.0 * sg,
    }

def get_inc_bl_bg(mask):
    """Return (inc_bl_list, inc_bg_list) for detailed inspection."""
    inc = r[mask & ~baseline]
    bl = inc[inc["actual_return"] <= -0.04].sort_values("actual_return")
    bg = inc[inc["actual_return"] >= 0.08].sort_values("actual_return", ascending=False)
    return bl, bg

print(f"Events: {len(events)} ({len(actual)} actual + {len(shadow)} shadow)")
print(f"Baseline r1-r8: {int(baseline.sum())} hits, {int((~baseline).sum())} unfiltered")
print(f"Unfiltered big_loss: {int((r[~baseline]['actual_return']<=-0.04).sum())}, big_gain: {int((r[~baseline]['actual_return']>=0.08).sum())}")

# ====== PHASE 1: Single factor (each candidate alone) ======
print(f"\n{'='*70}")
print("PHASE 1: SINGLE FACTOR THRESHOLD SCAN")
for factor, thresholds in CANDIDATE_THRESHOLDS.items():
    print(f"\n  --- {factor} ---")
    results = []
    for op, t in thresholds:
        mask = compare(r[factor], op, t)
        res = eval_rule(mask, f"{factor}{op}{t}")
        if res["inc_n"] > 0:
            results.append(res)
    results.sort(key=lambda x: (x["inc_bg"] == 0, x["score"]), reverse=True)
    for res in results[:8]:
        star = " ***ZERO BG***" if res["inc_bg"] == 0 and res["inc_bl"] > 0 else ""
        print(f"    {res['rule']:50s} inc_n={res['inc_n']:3d} bl={res['inc_bl']:2d} bg={res['inc_bg']:2d} sg={res['inc_sg']} sl={res['inc_sl']} mean={res['inc_mean']:+.4f} ov={res['overlap']:3d}{star}")

# ====== PHASE 2: Two-factor combos (candidate + one other) ======
print(f"\n{'='*70}")
print("PHASE 2: TWO-FACTOR COMBOS (candidate + other factor)")

all_results_2f = []
for factor, thresholds in CANDIDATE_THRESHOLDS.items():
    for op, t in thresholds:
        base_mask = compare(r[factor], op, t)
        if base_mask.sum() == 0: continue
        for combo_factor, combo_thresholds in COMBO_THRESHOLDS.items():
            if combo_factor not in r.columns: continue
            for cop, ct in combo_thresholds:
                mask = base_mask & compare(r[combo_factor], cop, ct)
                res = eval_rule(mask, f"{factor}{op}{t} & {combo_factor}{cop}{ct}")
                if res["inc_n"] == 0: continue
                res["factor"] = factor
                res["combo_factor"] = combo_factor
                all_results_2f.append(res)

all_results_2f.sort(key=lambda x: (x["inc_bg"] == 0 and x["inc_bl"] > 0, x["score"]), reverse=True)

# Zero big-gain rules
zbg_2f = [r for r in all_results_2f if r["inc_bg"] == 0 and r["inc_bl"] > 0]
print(f"\nZERO BIG-GAIN TWO-FACTOR RULES ({len(zbg_2f)} found):")
for res in zbg_2f[:30]:
    print(f"  {res['rule']:70s} inc_n={res['inc_n']:3d} bl={res['inc_bl']} bg=0 sg={res['inc_sg']} sl={res['inc_sl']} mean={res['inc_mean']:+.4f} ov={res['overlap']:3d} score={res['score']:.0f}")

# Show detail for top zero-bg rules
for res in zbg_2f[:6]:
    print(f"\n  DETAIL: {res['rule']}")
    bl, bg = get_inc_bl_bg(eval(f"(r['{res['factor']}'] {CANDIDATE_THRESHOLDS[res['factor']][0][0]} {CANDIDATE_THRESHOLDS[res['factor']][0][1]})"))
    # Reconstruct mask properly
    factor_part = res["rule"].split(" & ")[0]
    # Parse the rule string to reconstruct mask
    parts = res["rule"].split(" & ")
    mask = pd.Series(True, index=r.index)
    for part in parts:
        part = part.strip()
        for fac_name in CANDIDATE_THRESHOLDS:
            if part.startswith(fac_name):
                rest = part[len(fac_name):]
                for op_test in [">=","<=","<",">"]:
                    if rest.startswith(op_test):
                        val = float(rest[len(op_test):])
                        mask &= compare(r[fac_name], op_test, val)
                        break
                break
        else:
            for fac_name in COMBO_THRESHOLDS:
                if part.startswith(fac_name):
                    rest = part[len(fac_name):]
                    for op_test in [">=","<=","<",">"]:
                        if rest.startswith(op_test):
                            val = float(rest[len(op_test):])
                            mask &= compare(r[fac_name], op_test, val)
                            break
                    break
    bl, bg = get_inc_bl_bg(mask)
    print(f"    Big losses caught:")
    for _, row in bl.head(8).iterrows():
        print(f"      {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}")
    if len(bg) > 0:
        print(f"    Big gains (SHOULD BE 0):")
        for _, row in bg.head(5).iterrows():
            print(f"      {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}")

# ====== PHASE 3: Three-factor combos (best two-factor + third) ======
print(f"\n{'='*70}")
print("PHASE 3: THREE-FACTOR COMBOS (best 2F + third factor)")

# Take top zero-bg two-factor rules and add a third condition
all_results_3f = []
for zf_res in zbg_2f[:10]:
    # Reconstruct the 2F mask
    parts = zf_res["rule"].split(" & ")
    mask_2f = pd.Series(True, index=r.index)
    for part in parts:
        part = part.strip()
        for fac_name in list(CANDIDATE_THRESHOLDS) + list(COMBO_THRESHOLDS):
            if part.startswith(fac_name):
                rest = part[len(fac_name):]
                for op_test in [">=","<=","<",">"]:
                    if rest.startswith(op_test):
                        val = float(rest[len(op_test):])
                        all_thresholds = {**CANDIDATE_THRESHOLDS, **COMBO_THRESHOLDS}
                        if fac_name in all_thresholds:
                            mask_2f &= compare(r[fac_name], op_test, val)
                        break
                break
    # Try adding a third factor
    for third_factor, third_thresholds in COMBO_THRESHOLDS.items():
        if third_factor not in r.columns: continue
        # Skip if already in the rule
        if third_factor in zf_res["rule"]: continue
        for top, tv in third_thresholds:
            mask_3f = mask_2f & compare(r[third_factor], top, tv)
            res = eval_rule(mask_3f, f"{zf_res['rule']} & {third_factor}{top}{tv}")
            if res["inc_n"] == 0: continue
            if res["inc_bg"] == 0 and res["inc_bl"] > 0:
                all_results_3f.append(res)

all_results_3f.sort(key=lambda x: x["score"], reverse=True)
print(f"ZERO BIG-GAIN THREE-FACTOR RULES ({len(all_results_3f)} found):")
for res in all_results_3f[:20]:
    print(f"  {res['rule']:90s} inc_n={res['inc_n']:3d} bl={res['inc_bl']} bg=0 sg={res['inc_sg']} mean={res['inc_mean']:+.4f} ov={res['overlap']:3d}")

# ====== SUMMARY ======
print(f"\n{'='*70}")
print("SUMMARY")
print(f"  Baseline r1-r8: {int(baseline.sum())}/{len(events)} filtered")
print(f"  Unfiltered: {int((~baseline).sum())} — {int((r[~baseline]['actual_return']<=-0.04).sum())} big_loss, {int((r[~baseline]['actual_return']>=0.08).sum())} big_gain")
print(f"  Best 2F zero-BG: {zbg_2f[0]['rule'] if zbg_2f else 'NONE'} (bl={zbg_2f[0]['inc_bl'] if zbg_2f else 0})")
print(f"  Best 3F zero-BG: {all_results_3f[0]['rule'] if all_results_3f else 'NONE'} (bl={all_results_3f[0]['inc_bl'] if all_results_3f else 0})")
