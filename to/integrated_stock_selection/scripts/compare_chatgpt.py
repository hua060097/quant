"""用 ChatGPT 声称的同一文件验证两个规则的真实效果。"""
import json, pandas as pd, numpy as np, pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"

METRICS_FILE = RESULTS / "walkforward_metrics_20260520_210454.json"
SELECTIONS_FILE = RESULTS / "walkforward_selections_20260520_210454.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.json"

def load_records(path):
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

def normalize(df, source):
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

def bucket_counts(df):
    r = df["actual_return"]
    return {"n":int(len(df)), "mean(%)":round(float(r.mean())*100,2),
            "<=-4%":int((r<=-0.04).sum()),
            "-4~0%":int(((r>-0.04)&(r<0)).sum()),
            "3~8%":int(((r>=0.03)&(r<0.08)).sum()),
            ">=8%":int((r>=0.08).sum()),
            ">=10%":int((r>=0.10).sum())}

print("="*70)
print("Using ChatGPT's claimed files (20260520_210454)")
print("="*70)

selections = normalize(load_records(SELECTIONS_FILE), "actual")
shadow = normalize(load_records(SHADOW_FILE), "shadow")
events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(
    ["source","date","stock_code","actual_return"])
print(f"Events: actual={len(selections)}, shadow={len(shadow)}, total={len(events)}")

# Attach factors
schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
use = ["date","stock_code"] + [c for c in [
    "reg_severe_up_10d_ratio","market_advance_decline_ratio","price_ma5_ratio_raw_pct",
    "runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","limit_down_3d","top10_10d_strength_level_v3",
    "buy_sm_amount_rate","prev_non_first_limit_up_performance_percentile_20d",
    "kdj_j_9","score_z_in_day"] if c in schema_cols]
factors = pd.read_parquet(FACTORS, columns=use)
factors["date"] = pd.to_datetime(factors["date"])
for c in use[2:]:
    if c not in factors.columns:
        factors[c] = np.nan
# Also add to events after merge
for c in use[2:]:
    if c not in events.columns:
        events[c] = np.nan

trading_days = np.array(sorted(factors["date"].dropna().unique()))
prev_dates = []
for d in events["date"]:
    td = pd.Timestamp(d).normalize()
    pos = np.searchsorted(trading_days, np.datetime64(td), side="left") - 1
    prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)
events["factor_date"] = prev_dates
snap = factors.rename(columns={"date":"factor_date"})
events = events.merge(snap, on=["factor_date","stock_code"], how="left")
# Fill any columns that weren't in factors.parquet with NaN
ALL_RULE_COLS = [
    "runup_ma5_exhaustion","cpv_factor","prev_day_blowoff_risk",
    "sector_momentum_accel_rank","sector_leadership_persistence_v3",
    "return_20d_intra_z","market_liquidity_change","ulcer_index",
    "momentum_peak_decay_3d","intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2","market_advance_decline_ratio",
    "limit_down_3d","top10_10d_strength_level_v3","buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d","kdj_j_9",
    "score_z_in_day","reg_severe_up_10d_ratio","price_ma5_ratio_raw_pct"]
for c in ALL_RULE_COLS:
    if c not in events.columns:
        events[c] = np.nan

# Current v3 (r1-r9 only)
events["r1"] = (events["runup_ma5_exhaustion"] > 0.8) & (events["cpv_factor"] <= -0.4)
events["r2"] = (
    ((events["prev_day_blowoff_risk"] >= 2.0) & (events["sector_momentum_accel_rank"] <= 0.30))
    | ((events["prev_day_blowoff_risk"] > 5.0) & (events["sector_momentum_accel_rank"] <= 0.50))
    | ((events["prev_day_blowoff_risk"] > 1.0) & (events["sector_momentum_accel_rank"] <= 0.15)))
events["r3"] = (events["sector_leadership_persistence_v3"] <= -0.18) & (events["return_20d_intra_z"] <= 0.40)
events["r4"] = (events["market_liquidity_change"] < 0.94) & (
    (events["ulcer_index"] < -0.80) | (events["momentum_peak_decay_3d"] > 1.0)
    | (events["intra_sector_momentum_rank"] < 0.67))
events["r5"] = (events["mkt_dispersion_shock_v2"] < 0.98) & (events["market_advance_decline_ratio"] < 0.38)
events["r6"] = events["limit_down_3d"].fillna(0) == 1
events["r7"] = (events["top10_10d_strength_level_v3"] >= 0.70) & (events["buy_sm_amount_rate"] <= -1.25)
events["r8"] = (events["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80) & (events["kdj_j_9"] <= 0.75)
events["r9"] = events["score_z_in_day"].fillna(99) <= 1.50
events["r10_cand"] = (events["reg_severe_up_10d_ratio"] >= 0.18) & (events["market_advance_decline_ratio"] <= 0.45)
events["r11_cand"] = events["price_ma5_ratio_raw_pct"] >= 0.99

cur = events[["r1","r2","r3","r4","r5","r6","r7","r8","r9"]].any(axis=1)

# CHATGPT CLAIMS
print(f"\nChatGPT claims:")
print(f"  R10 standalone=28, inc=18 (2BL, 11BG)")
print(f"  R11 standalone=37, inc=25 (3BL, 17BG)")

print(f"\n{'='*70}")
print(f"BASE v3 (r1-r9)")
print(f"  Hits: {int(cur.sum())}")
print(f"  Buckets: {bucket_counts(events[cur])}")
print(f"  Unfiltered: {int((~cur).sum())}")
print(f"  Unfiltered buckets: {bucket_counts(events[~cur])}")

print(f"\n{'='*70}")
print(f"R10: reg_severe_up_10d_ratio>=0.18 & adr<=0.45")
s10 = events["r10_cand"]
i10 = s10 & ~cur
print(f"  standalone: {int(s10.sum())} hits, {bucket_counts(events[s10])}")
print(f"  incremental: {int(i10.sum())} hits, {bucket_counts(events[i10])}")
print(f"  overlap with v3: {int((s10 & cur).sum())}")
bl = events[i10 & (events["actual_return"] <= -0.04)].sort_values("actual_return")
bg = events[i10 & (events["actual_return"] >= 0.08)].sort_values("actual_return", ascending=False)
print(f"  inc big_loss ({len(bl)}):")
for _, r in bl.iterrows():
    print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f}")
print(f"  inc big_gain ({len(bg)}):")
for _, r in bg.iterrows():
    print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f}")

print(f"\n{'='*70}")
print(f"R11: price_ma5_ratio_raw_pct>=0.99")
s11 = events["r11_cand"]
i11 = s11 & ~cur
print(f"  standalone: {int(s11.sum())} hits, {bucket_counts(events[s11])}")
print(f"  incremental: {int(i11.sum())} hits, {bucket_counts(events[i11])}")
print(f"  overlap with v3: {int((s11 & cur).sum())}")
bl = events[i11 & (events["actual_return"] <= -0.04)].sort_values("actual_return")
bg = events[i11 & (events["actual_return"] >= 0.08)].sort_values("actual_return", ascending=False)
print(f"  inc big_loss ({len(bl)}):")
for _, r in bl.iterrows():
    print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f}")
print(f"  inc big_gain ({len(bg)}):")
for _, r in bg.iterrows():
    print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f}")

# R10+R11 incremental combined
cur_r10 = cur | events["r10_cand"]
i11b = events["r11_cand"] & ~cur_r10
print(f"\n{'='*70}")
print(f"R11 over base+r10:")
print(f"  incremental: {int(i11b.sum())} hits, {bucket_counts(events[i11b])}")
bl = events[i11b & (events["actual_return"] <= -0.04)].sort_values("actual_return")
bg = events[i11b & (events["actual_return"] >= 0.08)].sort_values("actual_return", ascending=False)
print(f"  inc big_loss ({len(bl)}):")
for _, r in bl.iterrows():
    print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f}")
print(f"  inc big_gain ({len(bg)}):")
for _, r in bg.iterrows():
    print(f"    {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f}")

# Show ChatGPT's claimed big gains: are they even in our events?
print(f"\n{'='*70}")
print("ChatGPT 声称被误杀的大涨，是否在我们的实际买入列表中?")
chatgpt_bg = ["603601.SH","605255.SH","600343.SH","688519.SH","688295.SH",
              "603618.SH","603929.SH","600151.SH"]
for code in chatgpt_bg:
    found = events[events["stock_code"] == code]
    if len(found) > 0:
        for _, r in found.iterrows():
            print(f"  {code} FOUND: source={r['source']} ret={r['actual_return']:+.4f} r10={r['r10_cand']} r11={r['r11_cand']}")
    else:
        print(f"  {code}: NOT IN OUR EVENTS (模型从未选中此股)")
