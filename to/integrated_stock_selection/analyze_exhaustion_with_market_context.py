"""
分析 exhaustion filter 与 TWO_BRANCH_GATE 市场因子的最佳组合

策略：只在市场环境不佳时才启用 exhaustion filter
- 市场强时：让模型自由发挥（被拦截的往往是好票）
- 市场弱时：启用过滤避开高位衰竭

TWO_BRANCH_GATE 因子：
- sector_momentum_accel_rank : 行业动量加速 (越低越弱)
- market_volatility_context  : 市场波动环境 (越高越危险)
- ma_alignment_persistence_5d: 均线排列持续性 (越低越弱)
"""

import re, os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq

LOG_98 = "stock_selection_test_mode_walkforward_分类模型_原因子98.log"
LOG_113 = "stock_selection_test_mode_walkforward_分类模型_原因子113.log"
FACTORS_PATH = "factor_lab_data/factors.parquet"

# ============================================================
# 1. Parse trades
# ============================================================
def parse_trades(log_path):
    buy_pat = re.compile(r"买入成功:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*([\d.]+),\s*数量:\s*\d+,\s*时间:\s*(\d{4}-\d{2}-\d{2})")
    sell_pat = re.compile(r"卖出:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*[\d.]+\s*,\s*收益:\s*([-\d.]+)\s*\(([-\d.]+)%\)")
    trades = {}; pending = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            bm = buy_pat.search(line)
            if bm: pending[bm.group(1)] = {"buy_date": bm.group(3), "stock": bm.group(1)}
            sm = sell_pat.search(line)
            if sm and sm.group(1) in pending:
                bi = pending.pop(sm.group(1))
                trades[f"{sm.group(1)}_{bi['buy_date']}"] = {
                    "stock_code": sm.group(1), "buy_date": bi["buy_date"],
                    "return_pct": float(sm.group(2)),
                }
    return pd.DataFrame(trades.values())

def load_factors(stocks, dates):
    schema = pq.read_schema(FACTORS_PATH).names
    needed = ["date", "stock_code",
              "close_below_ma5_after_runup", "low_break_prev_low_after_runup",
              "sector_momentum_accel_rank", "market_volatility_context",
              "ma_alignment_persistence_5d"]
    available = [c for c in needed if c in schema]
    missing = [c for c in needed if c not in schema]
    if missing: print(f"Missing factors: {missing}")

    df = pd.read_parquet(FACTORS_PATH, columns=available)
    df["date"] = pd.to_datetime(df["date"])
    min_d, max_d = pd.to_datetime(min(dates)), pd.to_datetime(max(dates))
    df = df[(df["date"] >= min_d) & (df["date"] <= max_d)]
    return df, available

# ============================================================
# 2. Evaluate a filter scheme
# ============================================================
def evaluate(returns, active_mask):
    """Compute metrics for active trades"""
    rets = returns[active_mask]
    n = len(rets)
    if n == 0: return {"cum_ret": 0, "max_dd": 0, "win_rate": 0, "n": 0, "big_loss": 0, "big_win": 0}
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum / peak - 1).min()
    return {
        "cum_ret": cum[-1] - 1, "max_dd": dd,
        "win_rate": np.mean(rets > 0), "n": n,
        "big_loss": int(np.sum(rets <= -0.05)),
        "big_win": int(np.sum(rets >= 0.15)),
        "cum_curve": cum,
        "mean_ret": np.mean(rets),
    }

# ============================================================
# 3. Main analysis
# ============================================================
print("=" * 100)
print("EXHAUSTION + TWO_BRANCH_GATE MARKET CONTEXT COMBINATION ANALYSIS")
print("=" * 100)

# Load trades from no-filter run (98 log - baseline performance)
trades = parse_trades(LOG_98)
print(f"\nBaseline (98 log, no filter): {len(trades)} trades")
base = evaluate(trades["return_pct"].values, np.ones(len(trades), dtype=bool))
print(f"  Cum: {base['cum_ret']:.2%}, MaxDD: {base['max_dd']:.2%}, Win: {base['win_rate']:.1%}, "
      f"BigLoss: {base['big_loss']}, BigWin: {base['big_win']}")

# Load factors
factors_df, available = load_factors(trades["stock_code"].tolist(), trades["buy_date"].tolist())
print(f"Available factors: {available}")

# Match factors to trades
trades["buy_date_dt"] = pd.to_datetime(trades["buy_date"])
merged = trades.merge(factors_df, left_on=["buy_date_dt", "stock_code"], right_on=["date", "stock_code"], how="left")

# Check which factors are available
has_close_below_ma5 = "close_below_ma5_after_runup" in merged.columns
has_sector_mom = "sector_momentum_accel_rank" in merged.columns
has_mkt_vol = "market_volatility_context" in merged.columns
has_ma_align = "ma_alignment_persistence_5d" in merged.columns

print(f"\nFactor availability: close_below_ma5={has_close_below_ma5}, "
      f"sector_mom={has_sector_mom}, mkt_vol={has_mkt_vol}, ma_align={has_ma_align}")

returns = merged["return_pct"].values
n = len(returns)

# ============================================================
# Define market context conditions
# ============================================================
def market_weak_mask(merged, factor, direction, pct):
    """Return boolean mask where market is weak"""
    vals = merged[factor].dropna()
    if len(vals) == 0: return pd.Series(False, index=merged.index)
    th = vals.quantile(pct / 100)
    if direction == "below":
        return merged[factor] < th
    else:
        return merged[factor] > th

# Test: exhaustion filter alone (for reference)
if has_close_below_ma5:
    exh_only_mask = merged["close_below_ma5_after_runup"] == 1
    exh_only = evaluate(returns, ~exh_only_mask)
    print(f"\n--- Exhaustion only (close_below_ma5) ---")
    print(f"  Blocked={int(exh_only_mask.sum())}, Cum={exh_only['cum_ret']:.2%}, "
          f"MaxDD={exh_only['max_dd']:.2%}, Win={exh_only['win_rate']:.1%}, "
          f"BigLoss={exh_only['big_loss']}, BigWin={exh_only['big_win']}")

# Test: TWO_BRANCH_GATE factors alone
print(f"\n--- Market context factors alone (weak market skip) ---")
for factor, direction, label in [
    ("sector_momentum_accel_rank", "below", "sector momentum weak"),
    ("market_volatility_context", "above", "market vol high"),
    ("ma_alignment_persistence_5d", "below", "MA alignment weak"),
]:
    if factor not in merged.columns:
        print(f"  {label}: factor not available")
        continue
    for pct in [30, 40, 50]:
        mask = market_weak_mask(merged, factor, direction, pct)
        if mask.sum() == 0: continue
        m = evaluate(returns, ~mask)
        blocked = int(mask.sum())
        bl_filtered = int(((returns <= -0.05) & mask).sum())
        bw_filtered = int(((returns >= 0.15) & mask).sum())
        print(f"  {label} P{pct}: blocked={blocked}, Cum={m['cum_ret']:.2%}, DD={m['max_dd']:.2%}, "
              f"BigLoss removed={bl_filtered}/{base['big_loss']}, BigWin removed={bw_filtered}/{base['big_win']}")

# ============================================================
# CORE: Exhaustion + Market context combination
# ============================================================
print(f"\n{'='*100}")
print("CORE ANALYSIS: Exhaustion filter ONLY when market is weak")
print(f"{'='*100}")

results = []
for mkt_factor, direction, mkt_label, mkt_pcts in [
    ("sector_momentum_accel_rank", "below", "sector_mom", [30, 40, 50]),
    ("market_volatility_context", "above", "mkt_vol", [50, 60, 70]),
    ("ma_alignment_persistence_5d", "below", "ma_align", [30, 40, 50]),
]:
    if mkt_factor not in merged.columns: continue
    for mkt_pct in mkt_pcts:
        # Market weak condition
        mkt_mask = market_weak_mask(merged, mkt_factor, direction, mkt_pct)
        if mkt_mask.sum() < 3: continue

        # Combined: exhaustion filter ONLY applies when market is weak
        exh_mask = merged["close_below_ma5_after_runup"] == 1
        combined_mask = exh_mask & mkt_mask

        m = evaluate(returns, ~combined_mask)
        blocked = int(combined_mask.sum())
        bl_filtered = int(((returns <= -0.05) & combined_mask).sum())
        bw_filtered = int(((returns >= 0.15) & combined_mask).sum())
        total_bl = int((returns <= -0.05).sum())

        results.append({
            "scheme": f"close_below_ma5 ONLY when {mkt_label} P{mkt_pct}",
            "mkt_factor": mkt_label,
            "mkt_pct": mkt_pct,
            "blocked": blocked,
            "n": m["n"],
            "cum_ret": m["cum_ret"],
            "max_dd": m["max_dd"],
            "win_rate": m["win_rate"],
            "mean_ret": m["mean_ret"],
            "big_loss": m["big_loss"],
            "big_win": m["big_win"],
            "bl_filtered": bl_filtered,
            "bw_filtered": bw_filtered,
            "total_bl": total_bl,
            "bl_reduction": bl_filtered / max(total_bl, 1),
        })

rd = pd.DataFrame(results).sort_values("cum_ret", ascending=False)
pd.set_option('display.max_columns', 20); pd.set_option('display.width', 250)
pd.set_option('display.float_format', '{:.3f}'.format)

print(f"\n{'scheme':<55} {'blocked':>7} {'cum_ret':>9} {'max_dd':>8} {'win':>6} "
      f"{'bigLoss':>8} {'bigWin':>7} {'bl_reduct':>9} {'bw_lost':>7}")
print("-" * 120)

for _, row in rd.iterrows():
    print(f"{row['scheme']:<55} {int(row['blocked']):>7} {row['cum_ret']:>8.2%} {row['max_dd']:>7.2%} "
          f"{row['win_rate']:>5.1%} {int(row['big_loss']):>3}/{base['big_loss']:<3} "
          f"{int(row['big_win']):>3}/{base['big_win']:<3} {row['bl_reduction']:>8.1%} "
          f"{int(row['bw_filtered']):>3}/{base['big_win']}")

# ============================================================
# Best vs baseline comparison
# ============================================================
print(f"\n{'='*100}")
print("BASELINE vs TOP COMBINATIONS")
print(f"{'='*100}")
print(f"{'Scheme':<55} {'blocked':>7} {'cum_ret':>9} {'max_dd':>8} {'win':>6} "
      f"{'mean_ret':>9} {'bigLoss':>8} {'bigWin':>7}")
print("-" * 120)
print(f"{'Baseline (no filter)':<55} {0:>7} {base['cum_ret']:>8.2%} {base['max_dd']:>7.2%} "
      f"{base['win_rate']:>5.1%} {base['mean_ret']:>8.4%} "
      f"{base['big_loss']:>3} {base['big_win']:>3}")

# Exhaustion only reference
if has_close_below_ma5:
    print(f"{'Exhaustion only (close_below_ma5)':<55} {int(exh_only_mask.sum()):>7} "
          f"{exh_only['cum_ret']:>8.2%} {exh_only['max_dd']:>7.2%} "
          f"{exh_only['win_rate']:>5.1%} {exh_only['mean_ret']:>8.4%} "
          f"{exh_only['big_loss']:>3} {exh_only['big_win']:>3}")

for _, row in rd.head(6).iterrows():
    print(f"{row['scheme']:<55} {int(row['blocked']):>7} {row['cum_ret']:>8.2%} "
          f"{row['max_dd']:>7.2%} {row['win_rate']:>5.1%} {row['mean_ret']:>8.4%} "
          f"{int(row['big_loss']):>3} {int(row['big_win']):>3}")

# ============================================================
# Segment stability for top schemes
# ============================================================
print(f"\n{'='*100}")
print("SEGMENT ANALYSIS FOR TOP SCHEMES")
print(f"{'='*100}")

seg_map = {
    "Seg1": ("2025-08-01", "2025-09-15"), "Seg2": ("2025-09-16", "2025-11-03"),
    "Seg3": ("2025-11-04", "2025-12-12"), "Seg4": ("2025-12-15", "2026-01-23"),
    "Seg5": ("2026-01-26", "2026-03-12"), "Seg6": ("2026-03-13", "2026-04-22"),
}
merged["segment"] = None
for seg, (s, e) in seg_map.items():
    mask = (merged["buy_date_dt"] >= pd.Timestamp(s)) & (merged["buy_date_dt"] <= pd.Timestamp(e))
    merged.loc[mask, "segment"] = seg

# Baseline segments
print(f"\n{'Scheme':<40} ", end="")
for s in seg_map: print(f"{s:>10} ", end="")
print(f"{'Cum':>10} {'DD':>8}")
print("-" * 120)

def print_segments(label, mask):
    print(f"{label:<40} ", end="")
    seg_rets = []
    for s in seg_map:
        seg_m = (merged["segment"] == s) & mask
        r = merged.loc[seg_m, "return_pct"].values
        if len(r) > 0:
            cr = np.prod(1+r)-1
            seg_rets.append(cr)
            print(f"{cr:>9.2%} ", end="")
        else:
            seg_rets.append(0)
            print(f"{'N/A':>10} ", end="")
    all_r = merged.loc[mask, "return_pct"].values
    cum = np.prod(1+all_r)-1 if len(all_r) > 0 else 0
    peak = np.maximum.accumulate(np.cumprod(1+all_r)) if len(all_r) > 0 else np.array([1])
    dd = min(np.cumprod(1+all_r) / peak - 1) if len(all_r) > 0 else 0
    print(f"{cum:>9.2%} {dd:>7.2%}")

print_segments("Baseline (no filter)", np.ones(n, dtype=bool))
print_segments("Exhaustion only (close_below_ma5)", ~exh_only_mask)

for _, row in rd.head(4).iterrows():
    mkt_factor_map = {
        "sector_mom": "sector_momentum_accel_rank",
        "mkt_vol": "market_volatility_context",
        "ma_align": "ma_alignment_persistence_5d",
    }
    direction_map = {"sector_mom": "below", "mkt_vol": "above", "ma_align": "below"}
    mf = mkt_factor_map.get(row["mkt_factor"])
    dr = direction_map.get(row["mkt_factor"])
    if mf and mf in merged.columns:
        mkt_mask = market_weak_mask(merged, mf, dr, row["mkt_pct"])
        exh_mask = merged["close_below_ma5_after_runup"] == 1
        cmb_mask = ~(exh_mask & mkt_mask)
        print_segments(row["scheme"], cmb_mask)

print(f"\n{'='*100}")
print("RECOMMENDATION")
print(f"{'='*100}")
best = rd.iloc[0]
print(f"Best combination: {best['scheme']}")
print(f"  Blocked: {int(best['blocked'])} trades, Cum: {best['cum_ret']:.2%}, DD: {best['max_dd']:.2%}")
print(f"  vs Baseline:  Cum {base['cum_ret']:.2%}->{best['cum_ret']:.2%}, DD {base['max_dd']:.2%}->{best['max_dd']:.2%}")
print(f"  vs Exhaustion only: Cum {exh_only['cum_ret']:.2%}->{best['cum_ret']:.2%}, DD {exh_only['max_dd']:.2%}->{best['max_dd']:.2%}")
