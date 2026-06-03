#!/usr/bin/env python3
"""
Analyze which factor, combined with runup_ma5_exhaustion, can best replace
top10_10d_strength_level_v3 in the exhaustion buy filter.

Parses backtest log, loads factor data, classifies trades by return,
and scans candidate factor combinations.
"""
import re, os, sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
LOG_PATH = BASE / "stock_selection_test_mode_walkforward_分类模型_原因子50.log"
FACTOR_PATH = BASE / "factor_lab_data" / "factors.parquet"
PRICE_PATH = BASE / "cache" / "stock_features_20230504_20260511.parquet"

# ============================================================
# 1. Parse log
# ============================================================
def parse_log(log_path):
    """Extract Top-5 candidates and trade results from backtest log."""
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse Top-5 candidates: the log line format is:
    # "2026-04-27 23:14:35 - module - INFO - <emoji> 2025-08-05 Top-5候选股票:"
    # "2026-04-27 23:14:35 - module - INFO -    1. 300885.SZ: 预测收益=0.4322, 收盘价=24.05"
    # Extract trading date from the "Top-5候选股票" line, then parse following candidate lines

    # Use a simpler approach: find all "Top-5候选股票" patterns with their dates
    header_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}) Top-5候选股票")
    candidate_content_pattern = re.compile(r"(\d+)\.\s([\d.A-Z]+):\s预测收益=([-\d.eE+]+),\s收盘价=([-\d.]+)")

    top5_records = []
    for m in header_pattern.finditer(content):
        date = m.group(1)
        start = m.end()
        chunk = content[start:start + 600]
        daily_count = 0
        for cm in candidate_content_pattern.finditer(chunk):
            top5_records.append({
                "date": date,
                "rank": int(cm.group(1)),
                "stock_code": cm.group(2),
                "predicted_return": float(cm.group(3)),
                "close_price": float(cm.group(4)),
            })
            daily_count += 1
            if daily_count >= 5:
                break

    # Parse trades, daily returns, and pre-top5 from lines
    lines = content.split("\n")
    trade_pattern = re.compile(r"卖出:\s([\d.A-Z]+),\s价格:\s([-\d.]+),\s收益:\s([-\d.]+)\s\(([-\d.]+)%\),\s原因:\s(.+)")
    daily_ret_line_pattern = re.compile(r"\[\d+/\d+\]\s(\d{4}-\d{2}-\d{2}):\s净值\s[\d.]+,\s当日收益\s([-\d.]+)\s\(([-\d.]+)%\)")
    pre_top5_line_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})\s预分析Top-5股票:\s\[([^\]]+)\]")

    trades = []
    daily_rets = []
    pre_top5 = {}

    for line in lines:
        tm = trade_pattern.search(line)
        if tm:
            trades.append({
                "stock_code": tm.group(1),
                "sell_price": float(tm.group(2)),
                "return": float(tm.group(3)),
                "return_pct": float(tm.group(4)),
                "reason": tm.group(5),
            })

        dm = daily_ret_line_pattern.search(line)
        if dm:
            daily_rets.append({
                "date": dm.group(1),
                "daily_return": float(dm.group(2)),
                "daily_return_pct": float(dm.group(3)),
            })

        pm = pre_top5_line_pattern.search(line)
        if pm:
            pre_top5[pm.group(1)] = re.findall(r"'([\d.A-Z]+)'", pm.group(2))

    top5_df = pd.DataFrame(top5_records)
    trades_df = pd.DataFrame(trades)
    daily_rets_df = pd.DataFrame(daily_rets)

    print(f"Parsed: {len(top5_df)} Top-5 candidates over {top5_df['date'].nunique()} days")
    print(f"Parsed: {len(trades_df)} trades")
    print(f"Parsed: {len(daily_rets_df)} daily returns")
    print(f"Parsed: {len(pre_top5)} days of pre-analyzed Top-5")

    return top5_df, trades_df, daily_rets_df, pre_top5

# ============================================================
# 2. Load factor and price data
# ============================================================
def load_data(top5_df):
    """Load factor values and price data for all candidate stock-date pairs."""
    target_dates = sorted(top5_df["date"].unique())
    print(f"Loading data for {len(target_dates)} dates: {target_dates[0]} ~ {target_dates[-1]}")

    # Load factor data
    factors = pd.read_parquet(FACTOR_PATH)
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors[factors["date"].between(
        pd.Timestamp(target_dates[0]) - pd.Timedelta(days=5),
        pd.Timestamp(target_dates[-1])
    )]
    print(f"Factor data: {len(factors)} rows, {len(factors.columns)} cols")

    # Load price data for return calculation
    prices = pd.read_parquet(PRICE_PATH, columns=["date", "stock_code", "close", "pct_chg"])
    prices["date"] = pd.to_datetime(prices["date"])

    # Compute next-day return: shift within each stock
    prices = prices.sort_values(["stock_code", "date"])
    prices["next_pct_chg"] = prices.groupby("stock_code")["pct_chg"].shift(-1)
    prices["next_close"] = prices.groupby("stock_code")["close"].shift(-1)

    # Also compute 3-day forward return
    prices["fwd_3d_close"] = prices.groupby("stock_code")["close"].shift(-3)
    prices["fwd_3d_ret"] = (prices["fwd_3d_close"] / prices["close"] - 1.0) * 100.0

    return factors, prices

# ============================================================
# 3. Merge and classify
# ============================================================
def build_analysis_dataset(top5_df, trades_df, factors, prices):
    """Merge top-5 candidates with their factor values and returns."""
    top5_df["date_dt"] = pd.to_datetime(top5_df["date"])

    # Merge with next-day return from prices
    merged = top5_df.merge(
        prices[["date", "stock_code", "next_pct_chg", "next_close", "pct_chg", "close"]],
        left_on=["date_dt", "stock_code"],
        right_on=["date", "stock_code"],
        how="left",
        suffixes=("", "_price")
    )

    # Merge with factor values
    factor_cols_of_interest = get_factor_columns(factors)
    factor_subset = factors[["date", "stock_code"] + factor_cols_of_interest].copy()
    merged = merged.merge(
        factor_subset,
        left_on=["date_dt", "stock_code"],
        right_on=["date", "stock_code"],
        how="left",
        suffixes=("", "_factor")
    )

    # Classify by next-day return
    def classify_return(ret):
        if pd.isna(ret):
            return "unknown"
        if ret >= 5.0:
            return "大涨"
        elif ret >= 0:
            return "小涨"
        elif ret >= -5.0:
            return "小跌"
        else:
            return "大跌"

    merged["return_class"] = merged["next_pct_chg"].apply(classify_return)

    # Also merge trade results (for stocks that were actually traded)
    # We match by approximate position since trades are sequential
    # For now, we focus on next-day returns which cover ALL candidates

    print(f"\nReturn class distribution (next-day pct_chg):")
    for cls_name in ["大涨", "小涨", "小跌", "大跌", "unknown"]:
        cnt = (merged["return_class"] == cls_name).sum()
        print(f"  {cls_name}: {cnt} ({cnt/len(merged)*100:.1f}%)")

    return merged

def get_factor_columns(factors):
    """Return list of factor columns to analyze."""
    # Exhaustion and risk-related factors
    candidates = [
        # Original filter factors
        "runup_ma5_exhaustion",
        "top10_10d_strength_level_v3",
        "prev_day_blowoff_risk",
        "sector_momentum_accel_rank",
        "market_advance_decline_ratio",
        "prior_high_volume_contest_20d",
        "stock_volume_ratio_20d",

        # High-position exhaustion family
        "close_position_20d",
        "close_position_60d",
        "high_volume_no_gain_3d",
        "upper_shadow_ratio_3d_mean",
        "upper_shadow_pressure_5d",
        "lower_shadow_absorption_5d",
        "close_to_high_3d_mean",
        "turnover_z_20",
        "high_position_volume_surge",
        "high_pos_upper_shadow",
        "high_pos_volume_no_gain",
        "high_pos_reversal_volume",
        "shadow_reversal_volume_pressure",
        "pullback_depth_from_10d_high",
        "momentum_peak_decay_3d",
        "consecutive_up_days_5d",
        "large_up_days_10d",
        "gap_up_count_5d",
        "close_below_ma5_after_runup",
        "low_break_prev_low_after_runup",
        "runup_low_break",

        # Blowoff/carryover family
        "carryover_overheat_risk",
        "distribution_after_strength_risk",
        "volatility_structure_alpha",
        "volume_price_exhaustion",
        "failed_breakout_pressure",
        "ma5_ma10_flattening_after_runup",

        # Volume/liquidity
        "abnormal_turnover",
        "money_flow_pressure",
        "amihud_illiquidity",

        # Risk control
        "efficiency_ratio",
        "ulcer_index",
        "max_drawdown_20d",
        "max_rally_20d",
        "max_rally_10d",

        # Support/resistance
        "prior_high_distance_20d",
        "prior_high_proximity_20d",
        "support_distance",
        "resistance_distance",

        # Market regime v3
        "mkt_ma5_followthrough_v3",
        "sector_leadership_persistence_v3",
        "sector_hotspot_concentration_v3",
        "ma5_strength_breadth_v3",
        "strong_stock_volume_confirmation_v3",
        "emotion_quality_balance_v3",
        "top10_10d_recent3_strength_v3",
        "top10_5d_recent3_strength_v3",
        "top10_3d_recent3_strength_v3",
        "top10_10d_today_return_level_v3",

        # Moving average
        "price_ma5_ratio",
        "price_ma10_ratio",
        "price_ma60_ratio_z",
        "ma_convergence_score",
        "ma_alignment_score",
        "ma_alignment_persistence_5d",

        # Candlestick
        "body_efficiency",
        "shadow_dominance",
        "gap_power",
        "range_compression",

        # Momentum
        "intraday_clv",
        "trend_acceleration",
        "trend_consistency",
        "cpv_factor",
        "stable_momentum",
        "momentum_decay",
        "volume_structure_alpha",
        "trend_pullback_composite_alpha",
        "volume_confirmed_pullback_alpha",
        "high_volatility_pullback_alpha",
        "regime_aware_reversal_alpha",

        # KDJ
        "kdj_golden_cross_strength",
        "kdj_diff_9",

        # Sector
        "sector_dispersion",
        "sector_relative_volatility",
        "sector_momentum_consistency",
        "sector_correlation_20d",
        "sector_volatility_5d",
        "sector_momentum_interaction",
        "sw_circ_mv_rank_intra_sector",
        "limit_up_sw2_concentration",
        "sw2_high_low_position_return_spread",

        # Style
        "sector_concentration",
        "sector_breadth",
        "sector_rank_change_5d",
        "sector_turnover_change_5d_20d",
        "sector_hhi_change",
        "sector_relative_strength_vs_market",
        "size_rotation_strength",

        # Money flow
        "buy_lg_amount_rate",
        "buy_sm_amount_rate",
        "net_d5_amount_to_turnover",
        "main_outflow_retail_takeover",

        # Market context
        "market_trend_strength",
        "market_sentiment_ad",
        "market_volatility_context",
        "market_liquidity_change",
        "market_60d_new_high_ratio",
        "limit_up_count_percentile_20d",
        "zhaban_count_percentile_20d",
        "limit_down_count_percentile_20d",

        # Regulatory
        "abnormal_deviation_3d_ratio",
        "abnormal_event_count_10d",
        "severe_abnormal_pressure_10d",
        "severe_abnormal_pressure_30d",

        # Cross-level
        "market_vols_adaptive_iv",
        "market_trend_adaptive_beta",
        "sector_excess_intra_rank",

        # Limit interaction
        "sector_limit_overflow_alpha",
        "sector_broken_resilience_alpha",
        "limit_sustain_alpha",
        "market_emotion_resonance_alpha",

        # Statistical
        "skewness_20d",
        "kurtosis_20d",
        "coskewness",
        "downside_beta",

        # PPO, MFI, Vortex
        "ppo_strength",
        "mfi_strength",
        "vortex_trend",
        "mass_index_reversal",
    ]

    available = [c for c in candidates if c in factors.columns]
    print(f"Available factors for analysis: {len(available)} / {len(candidates)}")
    missing = [c for c in candidates if c not in factors.columns]
    if missing:
        print(f"Missing factors ({len(missing)}): {missing[:10]}...")

    return available

# ============================================================
# 4. Factor combination scanner
# ============================================================
def scan_factor_combinations(merged, factor_cols, base_factor="runup_ma5_exhaustion", base_threshold=0.8):
    """
    For each candidate factor, scan thresholds to find the best combination
    with the base factor. Goal: filter big losers, preserve big gainers.
    """
    # Only use rows with valid returns
    valid = merged[merged["return_class"] != "unknown"].copy()
    print(f"\nAnalyzing {len(valid)} valid samples")

    big_gain = valid[valid["return_class"] == "大涨"]
    big_loss = valid[valid["return_class"] == "大跌"]
    small_gain = valid[valid["return_class"] == "小涨"]
    small_loss = valid[valid["return_class"] == "小跌"]

    print(f"  大涨: {len(big_gain)}, 小涨: {len(small_gain)}, 小跌: {len(small_loss)}, 大跌: {len(big_loss)}")

    # Base condition: runup_ma5_exhaustion > base_threshold
    base_hit = valid[base_factor] > base_threshold
    base_big_gain_hit = (big_gain[base_factor] > base_threshold).sum() if len(big_gain) > 0 else 0
    base_big_loss_hit = (big_loss[base_factor] > base_threshold).sum() if len(big_loss) > 0 else 0

    print(f"\nBase condition ({base_factor} > {base_threshold}):")
    print(f"  Total hits: {base_hit.sum()}")
    print(f"  大涨 caught: {base_big_gain_hit}/{len(big_gain)}")
    print(f"  大跌 caught: {base_big_loss_hit}/{len(big_loss)}")

    # For each candidate factor, evaluate combinations
    results = []

    for factor in factor_cols:
        if factor == base_factor:
            continue

        factor_series = valid[factor].dropna()
        if len(factor_series) < 10:
            continue

        if factor_series.nunique() < 5:
            continue

        # Determine threshold quantiles to scan
        q_low = max(0.01, factor_series.quantile(0.1))
        q_high = factor_series.quantile(0.95)

        if q_high <= q_low:
            continue

        # Scan 50 thresholds
        thresholds = np.linspace(q_low, q_high, 50)

        best_score = -999
        best_threshold = None
        best_stats = None

        for thresh in thresholds:
            # Combined condition
            combined = valid[base_factor] > base_threshold
            factor_cond = valid[factor] > thresh if np.median(factor_series) < np.mean(factor_series) else valid[factor] >= thresh
            # Try both directions
            cond_higher = valid[factor] >= thresh
            cond_lower = valid[factor] <= thresh

            for cond, direction in [(cond_higher, ">="), (cond_lower, "<=")]:
                final_cond = combined & cond

                bg_caught = (big_gain[base_factor] > base_threshold).sum() if len(big_gain) > 0 else 0
                bg_filtered = 0
                bl_caught = (big_loss[base_factor] > base_threshold).sum() if len(big_loss) > 0 else 0
                bl_filtered = 0

                if len(big_gain) > 0:
                    bg_cond = cond[big_gain.index.intersection(valid.index)]
                    base_cond_bg = base_hit[big_gain.index.intersection(valid.index)]
                    if len(bg_cond) > 0 and bg_cond.index.equals(base_cond_bg.index):
                        bg_filtered = ((base_cond_bg) & (~bg_cond)).sum()
                        bg_caught = (base_cond_bg & bg_cond).sum()
                    else:
                        # align indices
                        common = big_gain.index.intersection(valid.index).intersection(cond.index).intersection(base_hit.index)
                        bg_filtered = ((base_hit[common]) & (~cond[common])).sum()
                        bg_caught = 0

                if len(big_loss) > 0:
                    bl_cond = cond[big_loss.index.intersection(valid.index)]
                    base_cond_bl = base_hit[big_loss.index.intersection(valid.index)]
                    if len(bl_cond) > 0:
                        # big losses that are caught by base but then filtered out by this factor
                        common = big_loss.index.intersection(valid.index).intersection(cond.index).intersection(base_hit.index)
                        bl_filtered = ((base_hit[common]) & (~cond[common])).sum()
                        bl_caught = (base_hit[common] & cond[common]).sum()

                if bg_caught + bg_filtered == 0:
                    continue

                # Score: reward filtering big losses, penalize filtering big gains
                bg_missed_ratio = bg_filtered / (bg_caught + bg_filtered) if (bg_caught + bg_filtered) > 0 else 0
                bl_filter_ratio = bl_filtered / (bl_caught + bl_filtered) if (bl_caught + bl_filtered) > 0 else 0
                bl_caught_ratio = bl_caught / max(1, len(big_loss))

                # Composite: want high bl_filter_ratio, low bg_missed_ratio
                total_cond_hits = (base_hit & cond).sum()

                score = bl_filter_ratio * 0.5 + (1 - bg_missed_ratio) * 0.3 + total_cond_hits / len(valid) * 0.2

                if score > best_score:
                    best_score = score
                    best_threshold = thresh
                    best_stats = {
                        "factor": factor,
                        "threshold": thresh,
                        "direction": direction,
                        "score": score,
                        "total_cond_hits": total_cond_hits,
                        "bg_caught": bg_caught,
                        "bg_filtered": bg_filtered,
                        "bg_missed_ratio": bg_missed_ratio,
                        "bl_caught": bl_caught,
                        "bl_filtered": bl_filtered,
                        "bl_filter_ratio": bl_filter_ratio,
                        "bl_caught_ratio": bl_caught_ratio,
                    }

        if best_stats:
            results.append(best_stats)

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("score", ascending=False)

    return results_df, base_hit, valid

# ============================================================
# 5. Overlap analysis with other filter rules
# ============================================================
def analyze_rule_overlap(valid):
    """Analyze overlap between the three filter sub-rules."""
    print("\n" + "="*80)
    print("RULE OVERLAP ANALYSIS")
    print("="*80)

    # Rule 1 (current): runup_ma5_exhaustion > 0.8 AND top10_10d_strength_level_v3 >= 0.5
    r1 = (valid["runup_ma5_exhaustion"] > 0.8) & (valid["top10_10d_strength_level_v3"] >= 0.5)

    # Rule 2: prev_day_blowoff_risk >= 4.15 AND sector_momentum_accel_rank <= 0.41
    r2 = (valid["prev_day_blowoff_risk"] >= 4.15) & (valid["sector_momentum_accel_rank"] <= 0.41)

    # Rule 3: market_advance_decline_ratio >= 5.39559
    r3 = valid["market_advance_decline_ratio"] >= 5.39559

    # Rule 4 (volume): prior_high_volume_contest_20d <= -0.170934 AND stock_volume_ratio_20d >= 1.78077
    r4 = (valid["prior_high_volume_contest_20d"] <= -0.170934) & (valid["stock_volume_ratio_20d"] >= 1.78077)

    rules = {"R1(runup+top10)": r1, "R2(blowoff+sector)": r2, "R3(market_ad)": r3, "R4(volume)": r4}

    for name, rule in rules.items():
        hits = rule.sum()
        if hits > 0:
            bg = valid.loc[rule, "return_class"].value_counts().get("大涨", 0)
            bl = valid.loc[rule, "return_class"].value_counts().get("大跌", 0)
            sg = valid.loc[rule, "return_class"].value_counts().get("小涨", 0)
            sl = valid.loc[rule, "return_class"].value_counts().get("小跌", 0)
            print(f"\n{name}: {hits} stocks filtered")
            print(f"  大涨={bg}, 小涨={sg}, 小跌={sl}, 大跌={bl}")

    # Pairwise overlaps
    print("\n--- Pairwise Overlaps ---")
    rule_names = list(rules.keys())
    for i in range(len(rule_names)):
        for j in range(i+1, len(rule_names)):
            overlap = (rules[rule_names[i]] & rules[rule_names[j]]).sum()
            print(f"  {rule_names[i]} & {rule_names[j]}: {overlap} overlapping")

    # Triple overlaps
    r123 = (r1 & r2 & r3).sum()
    r124 = (r1 & r2 & r4).sum()
    r134 = (r1 & r3 & r4).sum()
    r234 = (r2 & r3 & r4).sum()
    print(f"\n  R1&R2&R3: {r123}, R1&R2&R4: {r124}, R1&R3&R4: {r134}, R2&R3&R4: {r234}")

    # Calculate how much each pair's overlap is with big gainers
    print("\n--- Big Gainer / Big Loser distribution per rule ---")
    for name, rule in rules.items():
        subset = valid[rule]
        if len(subset) == 0:
            print(f"  {name}: empty")
            continue
        dist = subset["return_class"].value_counts()
        print(f"  {name} ({len(subset)}): {dict(dist)}")
        if "大涨" in dist.index and len(valid[valid["return_class"]=="大涨"]) > 0:
            pct = dist.get("大涨", 0) / len(valid[valid["return_class"]=="大涨"]) * 100
            print(f"    -> catches {pct:.1f}% of all 大涨")

# ============================================================
# 6. Main
# ============================================================
def main():
    print("="*80)
    print("EXHAUSTION FILTER FACTOR ANALYSIS")
    print("="*80)

    # Step 1: Parse log
    print("\n[1] Parsing backtest log...")
    top5_df, trades_df, daily_rets_df, pre_top5 = parse_log(LOG_PATH)

    # Step 2: Load factor and price data
    print("\n[2] Loading factor and price data...")
    factors, prices = load_data(top5_df)

    # Step 3: Build analysis dataset
    print("\n[3] Building analysis dataset...")
    merged = build_analysis_dataset(top5_df, trades_df, factors, prices)

    # Step 4: Scan factor combinations
    print("\n[4] Scanning factor combinations...")
    factor_cols = get_factor_columns(factors)

    # First, analyze the base condition alone
    valid = merged[merged["return_class"] != "unknown"].copy()

    # Show factor correlations with runup_ma5_exhaustion and next_pct_chg
    print("\n--- Factor Correlations (Pearson) ---")
    corr_cols = ["runup_ma5_exhaustion", "top10_10d_strength_level_v3", "next_pct_chg"] + \
                [c for c in factor_cols if c in valid.columns][:30]
    corr_df = valid[corr_cols].corr()
    print("Top correlations with next_pct_chg:")
    for col, val in corr_df["next_pct_chg"].sort_values().items():
        if abs(val) > 0.03:
            print(f"  {col}: {val:.3f}")

    # Scan combinations
    results_df, base_hit, valid = scan_factor_combinations(merged, factor_cols)

    if len(results_df) > 0:
        print("\n" + "="*80)
        print("TOP 20 FACTOR COMBINATIONS (runup_ma5 > 0.8 AND factor >/< threshold)")
        print("="*80)
        cols_show = ["factor", "threshold", "direction", "score", "total_cond_hits",
                      "bg_caught", "bg_filtered", "bg_missed_ratio",
                      "bl_caught", "bl_filtered", "bl_filter_ratio"]
        top20 = results_df.head(20)[cols_show]
        for _, row in top20.iterrows():
            print(f"\n  {row['factor']} {row['direction']} {row['threshold']:.3f}  score={row['score']:.3f}")
            print(f"    combined hits: {row['total_cond_hits']}, 大涨 caught={row['bg_caught']} filtered={row['bg_filtered']} ({row['bg_missed_ratio']:.1%}), 大跌 caught={row['bl_caught']} filtered={row['bl_filtered']} ({row['bl_filter_ratio']:.1%})")

    # Step 5: Overlap analysis
    analyze_rule_overlap(valid)

    # Step 6: Detailed analysis of best candidate vs original
    if len(results_df) > 0:
        print("\n" + "="*80)
        print("DETAILED COMPARISON: Best candidate vs Original top10_10d_strength_level_v3")
        print("="*80)

        best = results_df.iloc[0]
        best_factor = best["factor"]
        best_thresh = best["threshold"]
        best_dir = best["direction"]

        # Original rule
        orig_cond = (valid["runup_ma5_exhaustion"] > 0.8) & (valid["top10_10d_strength_level_v3"] >= 0.5)

        # Best candidate rule
        if best_dir == ">=":
            best_cond = (valid["runup_ma5_exhaustion"] > 0.8) & (valid[best_factor] >= best_thresh)
        else:
            best_cond = (valid["runup_ma5_exhaustion"] > 0.8) & (valid[best_factor] <= best_thresh)

        def print_filter_stats(name, cond):
            subset = valid[cond]
            print(f"\n  {name}: {cond.sum()} hits")
            if len(subset) > 0:
                dist = subset["return_class"].value_counts()
                for k in ["大涨", "小涨", "小跌", "大跌"]:
                    print(f"    {k}: {dist.get(k, 0)}")

        print_filter_stats("Original (runup>0.8 & top10>=0.5)", orig_cond)
        print_filter_stats(f"Best ({best_factor} {best_dir} {best_thresh:.3f})", best_cond)

        # Overlap between original and best
        overlap = (orig_cond & best_cond).sum()
        orig_only = (orig_cond & ~best_cond).sum()
        best_only = (best_cond & ~orig_cond).sum()
        print(f"\n  Overlap: {overlap}, Original-only: {orig_only}, Best-only: {best_only}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
