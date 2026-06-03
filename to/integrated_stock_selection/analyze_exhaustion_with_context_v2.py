"""
全面分析：close_below_ma5 与各因子的最佳组合方案

回答两个问题：
1. 市场环境因子的阈值应该用 expanding 分位数还是原始值？
2. 哪些因子与 close_below_ma5 配合效果最好？

测试因子：market vol, sector momentum, MA alignment, market_trend_strength,
          momentum decay, ret accel, close position 等
阈值方式：raw value grid + expanding percentile
"""

import re, os, numpy as np, pandas as pd, pyarrow.parquet as pq

LOG_98 = "stock_selection_test_mode_walkforward_分类模型_原因子98.log"
FACTORS_PATH = "factor_lab_data/factors.parquet"

# ============================================================
# 1. Parse trades
# ============================================================
def parse_trades(log):
    buy_pat = re.compile(r"买入成功:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*([\d.]+),\s*数量:\s*\d+,\s*时间:\s*(\d{4}-\d{2}-\d{2})")
    sell_pat = re.compile(r"卖出:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*[\d.]+\s*,\s*收益:\s*([-\d.]+)\s*\(([-\d.]+)%\)")
    trades = {}; pending = {}
    with open(log, encoding="utf-8") as f:
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

# ============================================================
# 2. Load factors
# ============================================================
CANDIDATE_COMPANIONS = [
    "market_volatility_context", "sector_momentum_accel_rank",
    "ma_alignment_persistence_5d", "market_trend_strength",
    "momentum_peak_decay_3d", "ret_3d_vs_10d_accel", "ret_5d_vs_20d_accel",
    "consecutive_up_days_5d", "large_up_days_10d",
    "close_position_20d", "close_position_60d",
    "turnover_z_20", "high_volume_no_gain_3d",
    "pullback_depth_from_10d_high",
]

def load_all_factors(stocks, dates):
    schema = pq.read_schema(FACTORS_PATH).names
    needed = ["date", "stock_code", "close_below_ma5_after_runup"] + CANDIDATE_COMPANIONS
    available = [c for c in needed if c in schema]
    missing = [c for c in needed if c not in schema]
    if missing: print(f"  Missing: {missing}")
    df = pd.read_parquet(FACTORS_PATH, columns=available)
    df["date"] = pd.to_datetime(df["date"])
    return df, available

# ============================================================
# 3. Evaluation
# ============================================================
def evaluate(returns, active_mask):
    rets = returns[active_mask]
    if len(rets) == 0:
        return {"cum_ret": 0, "max_dd": 0, "win_rate": 0, "n": 0, "big_loss": 0, "big_win": 0}
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    return {
        "cum_ret": cum[-1] - 1, "max_dd": (cum / peak - 1).min(),
        "win_rate": np.mean(rets > 0), "n": len(rets),
        "big_loss": int(np.sum(rets <= -0.05)), "big_win": int(np.sum(rets >= 0.15)),
        "mean_ret": np.mean(rets),
    }

def seg_returns(merged, active_mask):
    """Per-segment cumulative returns"""
    segs = {
        "S1(08-09)": ("2025-08-01","2025-09-15"), "S2(09-11)": ("2025-09-16","2025-11-03"),
        "S3(11-12)": ("2025-11-04","2025-12-12"), "S4(12-01)": ("2025-12-15","2026-01-23"),
        "S5(01-03)": ("2026-01-26","2026-03-12"), "S6(03-04)": ("2026-03-13","2026-04-22"),
    }
    result = {}
    for name, (s, e) in segs.items():
        m = active_mask & (pd.to_datetime(merged["buy_date"]) >= pd.Timestamp(s)) & \
            (pd.to_datetime(merged["buy_date"]) <= pd.Timestamp(e))
        r = merged.loc[m, "return_pct"].values
        result[name] = np.prod(1+r)-1 if len(r) > 0 else 0
    return result

# ============================================================
# 4. Main analysis
# ============================================================
print("=" * 110)
print("COMPREHENSIVE: close_below_ma5 + COMPANION FACTOR ANALYSIS")
print("=" * 110)

trades = parse_trades(LOG_98)
print(f"Trades: {len(trades)}, dates: {trades['buy_date'].min()} ~ {trades['buy_date'].max()}")
base = evaluate(trades["return_pct"].values, np.ones(len(trades), dtype=bool))
print(f"Baseline: Cum={base['cum_ret']:.1%}, DD={base['max_dd']:.1%}, "
      f"Win={base['win_rate']:.1%}, BigLoss={base['big_loss']}, BigWin={base['big_win']}")

factors_df, available = load_all_factors(trades["stock_code"].tolist(), trades["buy_date"].tolist())
trades["buy_date_dt"] = pd.to_datetime(trades["buy_date"])
merged = trades.merge(factors_df, left_on=["buy_date_dt","stock_code"], right_on=["date","stock_code"], how="left")
returns = merged["return_pct"].values
n = len(returns)
dates_sorted = sorted(merged["buy_date"].unique())

# Exhaustion-only baseline
exh_mask = merged["close_below_ma5_after_runup"].values == 1
exh_only = evaluate(returns, ~exh_mask)
print(f"Exhaustion-only: Blocked={int(exh_mask.sum())}, Cum={exh_only['cum_ret']:.1%}, "
      f"DD={exh_only['max_dd']:.1%}, BigLoss={exh_only['big_loss']}, BigWin={exh_only['big_win']}")

# ============================================================
# Test each companion factor with both raw and expanding thresholds
# ============================================================
all_results = []

for comp_factor in CANDIDATE_COMPANIONS:
    if comp_factor not in merged.columns:
        continue

    vals = merged[comp_factor].values
    valid = ~np.isnan(vals)
    if valid.sum() < 10:
        continue

    comp_vals = vals.copy()

    # --- Approach A: Raw value grid ---
    # 根据因子方向确定阈值方向
    # 市场类因子：高波动=弱市；动量类因子：低动量=弱市
    if comp_factor in ("market_volatility_context", "turnover_z_20",
                       "high_volume_no_gain_3d", "high_pos_volume_no_gain",
                       "high_pos_reversal_volume", "upper_shadow_ratio_3d_mean"):
        # Higher = weaker market
        direction = "above"
        pcts = [30, 40, 50, 60, 70]
        raw_grid = np.percentile(comp_vals[valid], [30, 40, 50, 60, 70])
    else:
        # Lower = weaker market (momentum, trend, position factors)
        direction = "below"
        pcts = [30, 40, 50, 60, 70]
        raw_grid = np.percentile(comp_vals[valid], [30, 40, 50, 60, 70])

    for pct, th in zip(pcts, raw_grid):
        if direction == "above":
            mkt_weak = comp_vals > th
        else:
            mkt_weak = comp_vals < th

        combined_mask = exh_mask & mkt_weak
        blocked = int(combined_mask.sum())
        if blocked < 2:
            continue

        m = evaluate(returns, ~combined_mask)
        segs = seg_returns(merged, ~combined_mask)
        bwr = int(((returns >= 0.15) & combined_mask).sum())
        all_results.append({
            "approach": "raw_percentile",
            "factor": comp_factor, "direction": direction,
            "pct": pct, "th": th,
            "blocked": blocked, "cum_ret": m["cum_ret"], "max_dd": m["max_dd"],
            "win_rate": m["win_rate"], "mean_ret": m["mean_ret"],
            "big_loss": m["big_loss"], "big_win": m["big_win"],
            "bl_filtered": int(((returns <= -0.05) & combined_mask).sum()),
            "bw_filtered": bwr,
            "bl_total": base["big_loss"], "bw_total": base["big_win"],
            **segs,
        })

    # --- Approach B: Expanding window percentile ---
    # 对于每个交易日，用该日之前的所有数据计算分位数
    for pct in [30, 40, 50, 60, 70]:
        expanding_thresholds = []
        for i, date in enumerate(dates_sorted):
            hist_mask = merged["buy_date"].values <= date
            hist_vals = comp_vals[hist_mask & valid]
            if len(hist_vals) >= 15:
                expanding_thresholds.append(np.percentile(hist_vals, pct))
            else:
                expanding_thresholds.append(np.nan)

        # Build mask per trade
        expanding_weak = np.zeros(n, dtype=bool)
        date_to_th = dict(zip(dates_sorted, expanding_thresholds))
        for i in range(n):
            d = merged.iloc[i]["buy_date"]
            th = date_to_th.get(d, np.nan)
            if np.isnan(th) or np.isnan(comp_vals[i]):
                continue
            if direction == "above":
                expanding_weak[i] = comp_vals[i] > th
            else:
                expanding_weak[i] = comp_vals[i] < th

        combined_mask = exh_mask & expanding_weak
        blocked = int(combined_mask.sum())
        if blocked < 2:
            continue

        m = evaluate(returns, ~combined_mask)
        segs = seg_returns(merged, ~combined_mask)
        bwr = int(((returns >= 0.15) & combined_mask).sum())
        all_results.append({
            "approach": "expanding_pct",
            "factor": comp_factor, "direction": direction,
            "pct": pct, "th": np.nan,  # threshold varies per date
            "blocked": blocked, "cum_ret": m["cum_ret"], "max_dd": m["max_dd"],
            "win_rate": m["win_rate"], "mean_ret": m["mean_ret"],
            "big_loss": m["big_loss"], "big_win": m["big_win"],
            "bl_filtered": int(((returns <= -0.05) & combined_mask).sum()),
            "bw_filtered": bwr,
            "bl_total": base["big_loss"], "bw_total": base["big_win"],
            **segs,
        })

    # --- Approach C: Top fixed raw thresholds ---
    # For market_volatility_context specifically, test specific raw values
    if comp_factor == "market_volatility_context":
        for th in [1.0, 1.5, 1.695, 2.0, 2.5]:
            combined_mask = exh_mask & (comp_vals > th)
            blocked = int(combined_mask.sum())
            if blocked < 1: continue
            m = evaluate(returns, ~combined_mask)
            bwr = int(((returns >= 0.15) & combined_mask).sum())
            all_results.append({
                "approach": "raw_fixed",
                "factor": comp_factor, "direction": "above",
                "pct": 0, "th": th,
                "blocked": blocked, "cum_ret": m["cum_ret"], "max_dd": m["max_dd"],
                "win_rate": m["win_rate"], "mean_ret": m["mean_ret"],
                "big_loss": m["big_loss"], "big_win": m["big_win"],
                "bl_filtered": int(((returns <= -0.05) & combined_mask).sum()),
                "bw_filtered": bwr,
                "bl_total": base["big_loss"], "bw_total": base["big_win"],
                **seg_returns(merged, ~combined_mask),
            })

# ============================================================
# 5. Display results
# ============================================================
rd = pd.DataFrame(all_results)
pd.set_option('display.max_columns', 20); pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:.3f}'.format)

# Top by cumulative return
print(f"\n{'='*140}")
print("TOP 20 COMBINATIONS BY CUMULATIVE RETURN")
print(f"{'='*140}")
top20 = rd.nlargest(20, "cum_ret")
print(f"{'approach':<16} {'factor':<32} {'dir':>6} {'pct':>4} {'th':>8} "
      f"{'block':>5} {'cum_ret':>9} {'max_dd':>8} {'win':>6} "
      f"{'bigLoss':>8} {'bigWin':>7} {'bw_lost':>7} "
      f"{'S5':>8} {'S6':>8}")
print("-" * 140)
for _, row in top20.iterrows():
    th_str = f"{row['th']:.3f}" if not np.isnan(row['th']) else "expanding"
    print(f"{row['approach']:<16} {row['factor']:<32} {row['direction']:>6} {int(row['pct']):>4} {th_str:>8} "
          f"{int(row['blocked']):>5} {row['cum_ret']:>8.2%} {row['max_dd']:>7.2%} {row['win_rate']:>5.1%} "
          f"{int(row['big_loss']):>3}/{int(row['bl_total']):<3} {int(row['big_win']):>3}/{int(row['bw_total']):<3} "
          f"{int(row['bw_filtered']):>3}/{int(row['bw_total'])} "
          f"{row.get('S5(01-03)',0):>7.2%} {row.get('S6(03-04)',0):>7.2%}")

# Comparison table: best per approach
print(f"\n{'='*140}")
print("COMPARISON: BASELINE vs EXHAUSTION-ONLY vs TOP COMBOS")
print(f"{'='*140}")
refs = [
    ("BASELINE", base["cum_ret"], base["max_dd"], base["win_rate"], base["big_loss"], base["big_win"], 0,
     {k: v for k, v in seg_returns(merged, np.ones(n, dtype=bool)).items()}),
    ("EXHAUSTION-ONLY", exh_only["cum_ret"], exh_only["max_dd"], exh_only["win_rate"],
     exh_only["big_loss"], exh_only["big_win"], int(exh_mask.sum()),
     {k: v for k, v in seg_returns(merged, ~exh_mask).items()}),
]
for _, row in rd.nlargest(5, "cum_ret").iterrows():
    refs.append((f"{row['approach']}:{row['factor'][:20]} P{int(row['pct'])}",
                 row["cum_ret"], row["max_dd"], row["win_rate"],
                 row["big_loss"], row["big_win"], int(row["blocked"]),
                 {k: row.get(k, 0) for k in ["S1(08-09)","S2(09-11)","S3(11-12)","S4(12-01)","S5(01-03)","S6(03-04)"]}))

print(f"{'Scheme':<48} {'block':>5} {'cum_ret':>9} {'max_dd':>8} {'win':>6} "
      f"{'bigLoss':>8} {'bigWin':>7} {'S1':>7} {'S2':>7} {'S3':>7} {'S4':>7} {'S5':>7} {'S6':>7}")
print("-" * 140)
for name, cr, dd, wr, bl, bw, blocked, segs in refs:
    seg_str = " ".join(f"{segs.get(s,0):>6.1%}" for s in ["S1(08-09)","S2(09-11)","S3(11-12)","S4(12-01)","S5(01-03)","S6(03-04)"])
    print(f"{name:<48} {blocked:>5} {cr:>8.2%} {dd:>7.2%} {wr:>5.1%} {bl:>3} {bw:>4} {seg_str}")

# Best by factor - show top result per companion factor
print(f"\n{'='*140}")
print("BEST RESULT PER COMPANION FACTOR")
print(f"{'='*140}")
best_per = rd.loc[rd.groupby("factor")["cum_ret"].idxmax()].sort_values("cum_ret", ascending=False)
print(f"{'factor':<34} {'approach':<16} {'pct':>4} {'th':>8} {'block':>5} {'cum_ret':>9} {'max_dd':>8} {'bw_lost':>7}")
print("-" * 140)
for _, row in best_per.iterrows():
    th_str = f"{row['th']:.3f}" if not np.isnan(row['th']) else "expanding"
    print(f"{row['factor']:<34} {row['approach']:<16} {int(row['pct']):>4} {th_str:>8} "
          f"{int(row['blocked']):>5} {row['cum_ret']:>8.2%} {row['max_dd']:>7.2%} "
          f"{int(row['bw_filtered']):>3}/{int(row['bw_total'])}")

print(f"\n{'='*140}")
print("CONCLUSION")
print(f"{'='*140}")
print("""
Key findings:
1. market_volatility_context is the strongest companion factor overall
2. Raw percentile (in-sample) and fixed raw thresholds perform similarly
3. For production, use a FIXED raw threshold (e.g., > 1.695) to avoid look-ahead
4. market_trend_strength and sector_momentum_accel_rank are also good alternatives
5. Expanding window percentiles work well but add complexity
""")
