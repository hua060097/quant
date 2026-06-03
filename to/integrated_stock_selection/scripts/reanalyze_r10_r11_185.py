"""基于185日志正确文件 20260520_210454 重新分析 r10/r11"""
import json, pandas as pd, numpy as np, pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
TIMESTAMP = "20260520_210454"

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

ALL = ["runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","market_advance_decline_ratio",
    "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9",
    "reg_severe_up_10d_ratio","price_ma5_ratio_raw_pct"]

schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
use = ["date","stock_code"] + [c for c in ALL if c in schema_cols]
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
for c in ALL:
    if c not in events.columns:
        events[c] = np.nan

r = events
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
# r9 requires score_z_in_day which is runtime-computed, not in parquet
# Set r9=False: baseline uses r1-r8 only (this matches what was in 185 log)
r["r9"] = False

v3 = r[["r1","r2","r3","r4","r5","r6","r7","r8","r9"]].any(axis=1)

# R10: reg_severe_up_10d_ratio >= 0.18 & adr <= 0.45
r10_mask = (r["reg_severe_up_10d_ratio"] >= 0.18) & (r["market_advance_decline_ratio"] <= 0.45)
# R11: price_ma5_ratio_raw_pct >= 0.99
r11_mask = r["price_ma5_ratio_raw_pct"] >= 0.99

def B(df):
    x = df["actual_return"]
    return {"n":len(df), "<=-4%":int((x<=-0.04).sum()), "-4~0":int(((x>-0.04)&(x<0)).sum()),
            "3~8%":int(((x>=0.03)&(x<0.08)).sum()), ">=8%":int((x>=0.08).sum())}

print("="*70)
print(f"185 LOG CORRECT FILE: {TIMESTAMP}")
print(f"Events: actual={len(actual)}, shadow={len(shadow)}, total={len(events)}")
print(f"Date range: {events['date'].min().date()} ~ {events['date'].max().date()}")
print(f"Missing factors: reg_severe={r['reg_severe_up_10d_ratio'].isna().sum()}, "
      f"price_ma5_pct={r['price_ma5_ratio_raw_pct'].isna().sum()}")

print(f"\n{'='*70}")
print("BASE: r1-r9 (当前v3)")
print(f"  Hits: {int(v3.sum())} / {len(events)}")
print(f"  Buckets: {B(r[v3])}")
print(f"  Unfiltered: {int((~v3).sum())}")
print(f"  Unfiltered: {B(r[~v3])}")

# ===== R10 ANALYSIS =====
print(f"\n{'='*70}")
print("R10: reg_severe_up_10d_ratio >= 0.18 & market_advance_decline_ratio <= 0.45")
r10_standalone = r[r10_mask]
r10_inc = r[r10_mask & ~v3]
print(f"  Standalone: {B(r10_standalone)}")
print(f"  Incremental (over r1-r9): {B(r10_inc)}")
print(f"  Overlap with v3: {int((r10_mask & v3).sum())}")

print(f"\n  Incremental big losses ({len(r10_inc[r10_inc['actual_return']<=-0.04])}):")
for _, row in r10_inc[r10_inc["actual_return"]<=-0.04].sort_values("actual_return").iterrows():
    print(f"    {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}  reg={row['reg_severe_up_10d_ratio']:.3f} adr={row['market_advance_decline_ratio']:.3f}")

print(f"\n  Incremental big gains (>=8%) — FALSE KILLS ({len(r10_inc[r10_inc['actual_return']>=0.08])}):")
for _, row in r10_inc[r10_inc["actual_return"]>=0.08].sort_values("actual_return", ascending=False).iterrows():
    print(f"    {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}  reg={row['reg_severe_up_10d_ratio']:.3f} adr={row['market_advance_decline_ratio']:.3f}")

print(f"\n  Incremental small gains (3-8%) ({len(r10_inc[(r10_inc['actual_return']>=0.03)&(r10_inc['actual_return']<0.08)])}):")
for _, row in r10_inc[(r10_inc["actual_return"]>=0.03)&(r10_inc["actual_return"]<0.08)].sort_values("actual_return", ascending=False).iterrows():
    print(f"    {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}  reg={row['reg_severe_up_10d_ratio']:.3f} adr={row['market_advance_decline_ratio']:.3f}")

# ===== R11 ANALYSIS =====
print(f"\n{'='*70}")
print("R11: price_ma5_ratio_raw_pct >= 0.99")
r11_standalone = r[r11_mask]
r11_inc = r[r11_mask & ~v3]
print(f"  Standalone: {B(r11_standalone)}")
print(f"  Incremental (over r1-r9): {B(r11_inc)}")
print(f"  Overlap with v3: {int((r11_mask & v3).sum())}")

print(f"\n  Incremental big losses ({len(r11_inc[r11_inc['actual_return']<=-0.04])}):")
for _, row in r11_inc[r11_inc["actual_return"]<=-0.04].sort_values("actual_return").iterrows():
    print(f"    {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}  pct={row['price_ma5_ratio_raw_pct']:.4f}")

print(f"\n  Incremental big gains (>=8%) — FALSE KILLS ({len(r11_inc[r11_inc['actual_return']>=0.08])}):")
for _, row in r11_inc[r11_inc["actual_return"]>=0.08].sort_values("actual_return", ascending=False).iterrows():
    print(f"    {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}  pct={row['price_ma5_ratio_raw_pct']:.4f}")

print(f"\n  Incremental small gains (3-8%) ({len(r11_inc[(r11_inc['actual_return']>=0.03)&(r11_inc['actual_return']<0.08)])}):")
for _, row in r11_inc[(r11_inc["actual_return"]>=0.03)&(r11_inc["actual_return"]<0.08)].sort_values("actual_return", ascending=False).iterrows():
    print(f"    {row['source']:6s} {row['date'].date()} {row['stock_code']:12s} ret={row['actual_return']:+.4f}  pct={row['price_ma5_ratio_raw_pct']:.4f}")

# ===== ALTERNATIVE: try tighter R10 thresholds =====
print(f"\n{'='*70}")
print("ALTERNATIVE R10 THRESHOLDS (trying to reduce false kills)")
for reg_t in [0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65]:
    mask = (r["reg_severe_up_10d_ratio"] >= reg_t) & (r["market_advance_decline_ratio"] <= 0.45)
    inc = r[mask & ~v3]
    bl = int((inc["actual_return"]<=-0.04).sum())
    bg = int((inc["actual_return"]>=0.08).sum())
    sg = int(((inc["actual_return"]>=0.03)&(inc["actual_return"]<0.08)).sum())
    if bl > 0 or bg == 0:
        print(f"  reg>={reg_t:.2f} & adr<=0.45: inc_n={len(inc)} bl={bl} bg={bg} sg={sg}")

# ===== ALTERNATIVE: try R10 + additional condition =====
print(f"\n{'='*70}")
print("R10 TWO-FACTOR VARIANTS (adding a third condition)")
candidates = {
    "top10_10d_strength_level_v3 >= 0.65": (r["top10_10d_strength_level_v3"] >= 0.65),
    "top10_10d_strength_level_v3 >= 0.70": (r["top10_10d_strength_level_v3"] >= 0.70),
    "buy_sm_amount_rate <= -1.0": (r["buy_sm_amount_rate"] <= -1.0),
    "cpv_factor >= 0.4": (r["cpv_factor"] >= 0.4),
    "runup_ma5_exhaustion > 0.7": (r["runup_ma5_exhaustion"] > 0.7),
    "ulcer_index < -0.4": (r["ulcer_index"] < -0.4),
}
for label, extra in candidates.items():
    mask = r10_mask & extra
    inc = r[mask & ~v3]
    bl = int((inc["actual_return"]<=-0.04).sum())
    bg = int((inc["actual_return"]>=0.08).sum())
    if bl > 0 or bg == 0:
        print(f"  r10 & {label}: inc_n={len(inc)} bl={bl} bg={bg}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"  r1-r9: {int(v3.sum())}/{len(events)} filtered, {int((~v3).sum())} unfiltered")
print(f"  Unfiltered big losses: {int((r[~v3]['actual_return']<=-0.04).sum())}")
print(f"  Unfiltered big gains: {int((r[~v3]['actual_return']>=0.08).sum())}")
