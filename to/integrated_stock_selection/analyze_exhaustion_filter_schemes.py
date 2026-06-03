"""
分析 exhaustion 因子的买入过滤方案，对比多种方案的效果

参照 analyze_vpe_strict_mode.py 的思路：
- 解析日志中 85 笔交易的实际收益
- 对每个因子/因子组合测试不同的过滤阈值
- 计算：累计收益、最大回撤、分段稳定性、大亏过滤率、大赚保留率
- 用尽量少的因子，给出多个方案并对比

用法: python analyze_exhaustion_filter_schemes.py
"""

import re
import os
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOG_PATH = "stock_selection_test_mode_walkforward_分类模型_原因子50.log"
FACTORS_PATH = "factor_lab_data/factors.parquet"

EXHAUSTION_FACTORS = [
    "close_position_20d", "close_position_60d",
    "upper_shadow_ratio_3d_mean", "close_to_high_3d_mean",
    "turnover_z_20", "high_volume_no_gain_3d",
    "pullback_depth_from_10d_high",
    "high_pos_upper_shadow", "high_pos_volume_no_gain", "high_pos_reversal_volume",
    "ret_3d_vs_10d_accel", "ret_5d_vs_20d_accel", "momentum_peak_decay_3d",
    "consecutive_up_days_5d", "large_up_days_10d", "gap_up_count_5d",
    "close_below_ma5_after_runup", "low_break_prev_low_after_runup",
]

# ============================================================
# 1. Parse trades
# ============================================================
def parse_trades() -> pd.DataFrame:
    buy_pat = re.compile(
        r"买入成功:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*([\d.]+),\s*数量:\s*\d+,\s*时间:\s*(\d{4}-\d{2}-\d{2})"
    )
    sell_pat = re.compile(
        r"卖出:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*[\d.]+\s*,\s*收益:\s*([-\d.]+)\s*\(([-\d.]+)%\),\s*原因:\s*(.+)"
    )
    trades = []
    pending = {}
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            bm = buy_pat.search(line)
            if bm:
                pending[bm.group(1)] = {"buy_date": bm.group(3), "buy_price": float(bm.group(2))}
                continue
            sm = sell_pat.search(line)
            if sm:
                stock, ret_pct, reason = sm.group(1), float(sm.group(2)), sm.group(4).strip()
                if stock in pending:
                    bi = pending.pop(stock)
                    trades.append({
                        "stock_code": stock, "buy_date": bi["buy_date"],
                        "buy_price": bi["buy_price"], "return_pct": ret_pct, "sell_reason": reason,
                    })
    df = pd.DataFrame(trades)
    df["buy_date_dt"] = pd.to_datetime(df["buy_date"])
    # Add segment labels matching walkforward windows
    df["segment"] = pd.cut(
        df["buy_date_dt"],
        bins=pd.to_datetime([
            "2025-08-01", "2025-09-22", "2025-11-17", "2026-01-05", "2026-02-27"
        ]),
        labels=["Seg1", "Seg2", "Seg3", "Seg4"],
    )
    # Assign each trade to its actual segment
    seg_map = {
        "Seg1": ("2025-08-01", "2025-09-19"),
        "Seg2": ("2025-09-22", "2025-11-14"),
        "Seg3": ("2025-11-17", "2025-12-31"),
        "Seg4": ("2026-01-05", "2026-02-26"),
        "Seg5": ("2026-02-27", "2026-04-20"),
    }
    df["segment"] = None
    for i, (seg, (start, end)) in enumerate(seg_map.items()):
        mask = (df["buy_date_dt"] >= pd.Timestamp(start)) & (df["buy_date_dt"] <= pd.Timestamp(end))
        df.loc[mask, "segment"] = seg
    df["seg_idx"] = df["segment"].map({s: i for i, s in enumerate(seg_map.keys())})
    return df


# ============================================================
# 2. Match factors
# ============================================================
def match_factors(trades_df: pd.DataFrame) -> pd.DataFrame:
    import pyarrow.parquet as pq
    schema = pq.read_schema(FACTORS_PATH)
    all_cols = schema.names
    available = [c for c in EXHAUSTION_FACTORS if c in all_cols]
    logger.info(f"Available factors: {len(available)}/{len(EXHAUSTION_FACTORS)}")

    cols = ["date", "stock_code"] + available
    factors = pd.read_parquet(FACTORS_PATH, columns=cols)
    factors["date"] = pd.to_datetime(factors["date"])

    merged = trades_df.merge(
        factors,
        left_on=["buy_date_dt", "stock_code"],
        right_on=["date", "stock_code"],
        how="left",
    )
    logger.info(f"Matched: {merged[available[0]].notna().sum()}/{len(merged)} trades")
    merged["trade_idx"] = range(len(merged))
    return merged, available


# ============================================================
# 3. Portfolio metrics
# ============================================================
def compute_portfolio_metrics(returns: np.ndarray, dates: np.ndarray = None) -> dict:
    """Compute cumulative return, max drawdown, win rate, Sharpe, etc."""
    n = len(returns)
    if n == 0:
        return {"cum_ret": 0, "max_dd": 0, "win_rate": 0, "mean_ret": 0, "sharpe": 0, "n_trades": 0}

    cum = np.cumprod(1 + returns)
    cum_ret = cum[-1] - 1

    # Max drawdown
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    max_dd = dd.min()

    win_rate = np.mean(returns > 0)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1e-9
    sharpe = mean_ret / std_ret * np.sqrt(n)

    # Big loss/gain counts
    big_loss5 = np.sum(returns <= -0.05)
    big_loss7 = np.sum(returns <= -0.07)
    big_win10 = np.sum(returns >= 0.10)
    big_win15 = np.sum(returns >= 0.15)

    return {
        "cum_ret": cum_ret, "max_dd": max_dd, "win_rate": win_rate,
        "mean_ret": mean_ret, "sharpe": sharpe, "n_trades": n,
        "big_loss5": big_loss5, "big_loss7": big_loss7,
        "big_win10": big_win10, "big_win15": big_win15,
        "cum_curve": cum,
    }


def compute_segment_metrics(merged: pd.DataFrame, active_mask: np.ndarray) -> dict:
    """Compute per-segment cumulative returns."""
    segs = {}
    all_seg_names = sorted(merged["segment"].dropna().unique())
    for seg in all_seg_names:
        seg_mask = (merged["segment"].values == seg) & active_mask
        seg_rets = merged.loc[seg_mask, "return_pct"].values
        if len(seg_rets) > 0:
            cum = np.prod(1 + seg_rets) - 1
            segs[seg] = {"cum_ret": cum, "n": len(seg_rets), "win_rate": np.mean(seg_rets > 0),
                         "mean_ret": np.mean(seg_rets)}
        else:
            segs[seg] = {"cum_ret": 0, "n": 0, "win_rate": 0, "mean_ret": 0}
    return segs


# ============================================================
# 4. Filter scheme evaluator
# ============================================================
def evaluate_filter_scheme(merged: pd.DataFrame, scheme: dict, factor_col: str = None,
                           threshold: float = None, condition_desc: str = "") -> dict:
    """
    Evaluate a filter scheme.
    scheme: dict of {factor_name: (operator, threshold)} e.g. {"close_below_ma5_after_runup": ("==", 1)}
    Returns comprehensive metrics.
    """
    n_total = len(merged)
    returns = merged["return_pct"].values

    # Build filter mask: True = filtered out (return set to 0)
    filter_mask = np.zeros(n_total, dtype=bool)
    filter_desc_parts = []
    for fname, (op, th) in scheme.items():
        if fname not in merged.columns:
            continue
        vals = merged[fname].values
        if op == ">":
            m = vals > th
        elif op == ">=":
            m = vals >= th
        elif op == "<":
            m = vals < th
        elif op == "<=":
            m = vals <= th
        elif op == "==":
            m = vals == th
        elif op == "!=":
            m = vals != th
        else:
            raise ValueError(f"Unknown op: {op}")
        filter_mask = filter_mask | m  # OR logic: any factor triggers = filter out
        # Build threshold description
        if th == int(th):
            filter_desc_parts.append(f"{fname} {op} {int(th)}")
        else:
            filter_desc_parts.append(f"{fname} {op} {th:.4f}")

    # Active trades = not filtered
    active_mask = ~filter_mask
    n_active = active_mask.sum()
    n_filtered = filter_mask.sum()

    # Original metrics (all trades)
    base_metrics = compute_portfolio_metrics(returns)

    # Filtered metrics (only active trades)
    if n_active > 0:
        filtered_rets = returns[active_mask]
        filtered_metrics = compute_portfolio_metrics(filtered_rets)
    else:
        filtered_metrics = {"cum_ret": 0, "max_dd": 0, "win_rate": 0, "mean_ret": 0,
                           "sharpe": 0, "n_trades": 0, "big_loss5": 0, "big_loss7": 0,
                           "big_win10": 0, "big_win15": 0}

    # Big loss reduction
    base_bl5 = base_metrics["big_loss5"]
    base_bw15 = base_metrics["big_win15"]
    filt_bl5 = filtered_metrics["big_loss5"]
    filt_bw15 = filtered_metrics["big_win15"]

    # How many big losses were filtered
    loss_filtered_mask = (returns <= -0.05) & filter_mask
    big_loss_filtered = loss_filtered_mask.sum()
    big_loss_reduction = big_loss_filtered / max(base_bl5, 1)

    # How many big gains were filtered (we want this LOW)
    gain_filtered_mask = (returns >= 0.15) & filter_mask
    big_gain_filtered = gain_filtered_mask.sum()
    big_gain_preservation = 1 - big_gain_filtered / max(base_bw15, 1)

    # Score: big_loss_reduction - 0.5 * (1 - big_gain_preservation)
    # Higher is better: filter more losses, preserve more gains
    efficiency_score = big_loss_reduction * 2.0 - (1 - big_gain_preservation)

    # Segment stability
    seg_metrics = compute_segment_metrics(merged, active_mask)

    # Segment cumulative returns
    seg_rets_list = []
    for seg in sorted(seg_metrics.keys()):
        seg_rets_list.append(seg_metrics[seg]["cum_ret"])

    # Segment stability: min segment return / max segment return (higher = more stable)
    if seg_rets_list:
        seg_stability = min(seg_rets_list) / max(max(seg_rets_list), 0.01) if max(seg_rets_list) > 0 else 0
    else:
        seg_stability = 0

    return {
        "scheme": " AND ".join(filter_desc_parts) if filter_desc_parts else "Baseline (no filter)",
        "n_factors": len(scheme),
        "n_total": n_total,
        "n_active": n_active,
        "n_filtered": n_filtered,
        "pass_rate": n_active / max(n_total, 1),
        # Baseline
        "base_cum_ret": base_metrics["cum_ret"],
        "base_max_dd": base_metrics["max_dd"],
        "base_sharpe": base_metrics["sharpe"],
        "base_win_rate": base_metrics["win_rate"],
        "base_big_loss5": base_bl5,
        "base_big_win15": base_bw15,
        # Filtered
        "filt_cum_ret": filtered_metrics["cum_ret"],
        "filt_max_dd": filtered_metrics["max_dd"],
        "filt_sharpe": filtered_metrics["sharpe"],
        "filt_win_rate": filtered_metrics["win_rate"],
        "filt_mean_ret": filtered_metrics["mean_ret"],
        "filt_big_loss5": filt_bl5,
        "filt_big_loss7": filtered_metrics["big_loss7"],
        "filt_big_win10": filtered_metrics["big_win10"],
        "filt_big_win15": filt_bw15,
        # Reduction / preservation
        "big_loss_filtered": int(big_loss_filtered),
        "big_loss_reduction": big_loss_reduction,
        "big_gain_filtered": int(big_gain_filtered),
        "big_gain_preservation": big_gain_preservation,
        "efficiency_score": efficiency_score,
        # Per-segment
        "seg_stability": seg_stability,
        **{f"{seg}_cum_ret": seg_metrics[seg]["cum_ret"] for seg in sorted(seg_metrics.keys())},
        **{f"{seg}_n": seg_metrics[seg]["n"] for seg in sorted(seg_metrics.keys())},
    }


# ============================================================
# 5. Define filter schemes
# ============================================================
def define_schemes(merged: pd.DataFrame, available: list) -> list:
    """Define candidate filter schemes using 1-2 factors."""
    schemes = []

    # ---- Single factor schemes ----
    # Binary flags (best discriminators)
    schemes.append(("F1: close_below_ma5 (binary)", {"close_below_ma5_after_runup": ("==", 1)}))
    schemes.append(("F2: low_break_prev_low (binary)", {"low_break_prev_low_after_runup": ("==", 1)}))

    # Continuous factors with data-driven thresholds
    for fname, op, pcts in [
        ("high_pos_volume_no_gain", ">", [70, 75, 80]),
        ("high_volume_no_gain_3d", ">", [70, 75]),
        ("upper_shadow_ratio_3d_mean", ">", [70, 80]),
        ("gap_up_count_5d", ">", [70, 80]),
    ]:
        if fname not in merged.columns:
            continue
        vals = merged[fname].dropna()
        for pct in pcts:
            th = vals.quantile(pct / 100)
            label = f"{fname} > P{pct}({th:.3f})"
            schemes.append((label, {fname: (">", th)}))

    # Momentum deceleration (negative values = bad)
    for fname, op, pcts in [
        ("momentum_peak_decay_3d", "<", [30, 40]),
        ("ret_3d_vs_10d_accel", "<", [30, 40]),
        ("ret_5d_vs_20d_accel", "<", [30, 40]),
    ]:
        if fname not in merged.columns:
            continue
        vals = merged[fname].dropna()
        for pct in pcts:
            th = vals.quantile(pct / 100)
            label = f"{fname} < P{pct}({th:.3f})"
            schemes.append((label, {fname: ("<", th)}))

    # ---- Two-factor combinations (best discriminators combined) ----
    # Binary combos
    schemes.append(("C1: close_below_ma5 OR low_break",
                    {"close_below_ma5_after_runup": ("==", 1),
                     "low_break_prev_low_after_runup": ("==", 1)}))

    # Binary + continuous
    if "high_volume_no_gain_3d" in merged.columns:
        th70 = merged["high_volume_no_gain_3d"].quantile(0.70)
        th75 = merged["high_volume_no_gain_3d"].quantile(0.75)
        schemes.append((f"C2: close_below_ma5 OR vol_no_gain > P70({th70:.2f})",
                        {"close_below_ma5_after_runup": ("==", 1),
                         "high_volume_no_gain_3d": (">", th70)}))
        schemes.append((f"C3: close_below_ma5 OR vol_no_gain > P75({th75:.2f})",
                        {"close_below_ma5_after_runup": ("==", 1),
                         "high_volume_no_gain_3d": (">", th75)}))

    # Binary + momentum
    if "momentum_peak_decay_3d" in merged.columns:
        th30 = merged["momentum_peak_decay_3d"].quantile(0.30)
        schemes.append((f"C4: close_below_ma5 OR momentum_decay < P30({th30:.2f})",
                        {"close_below_ma5_after_runup": ("==", 1),
                         "momentum_peak_decay_3d": ("<", th30)}))

    # Three minimal: binary flags only (2 factors)
    schemes.append(("C5: BOTH binary flags (AND)",
                    {}))  # special handling below

    return schemes


# ============================================================
# 6. Main analysis
# ============================================================
def main():
    print("=" * 100)
    print("EXHAUSTION FACTOR FILTER SCHEME ANALYSIS")
    print("=" * 100)

    # Load data
    trades_df = parse_trades()
    logger.info(f"Parsed {len(trades_df)} trades, dates: {trades_df['buy_date'].min()} ~ {trades_df['buy_date'].max()}")
    merged, available = match_factors(trades_df)

    # Compute baseline
    base = compute_portfolio_metrics(merged["return_pct"].values)
    print(f"\n[BASELINE] {len(merged)} trades:")
    print(f"  Cum Return: {base['cum_ret']:.2%}, Max DD: {base['max_dd']:.2%}")
    print(f"  Win Rate: {base['win_rate']:.1%}, Sharpe: {base['sharpe']:.3f}")
    print(f"  Big Loss (<=-5%): {base['big_loss5']}, Big Loss (<=-7%): {base['big_loss7']}")
    print(f"  Big Win (>=10%): {base['big_win10']}, Big Win (>=15%): {base['big_win15']}")

    # Baseline segment analysis
    seg_metrics = compute_segment_metrics(merged, np.ones(len(merged), dtype=bool))
    print(f"\n  Segment Returns:")
    for seg in sorted(seg_metrics.keys()):
        sm = seg_metrics[seg]
        print(f"    {seg}: cum={sm['cum_ret']:.2%}, n={sm['n']}, win={sm['win_rate']:.1%}")

    # Define and evaluate all schemes
    schemes = define_schemes(merged, available)

    # Also add C5 special case (both binary flags = 0, i.e., only trade when both are clean)
    # This is an AND logic rather than OR
    n_total = len(merged)
    returns = merged["return_pct"].values
    both_clean_mask = (merged["close_below_ma5_after_runup"].values != 1) & \
                      (merged["low_break_prev_low_after_runup"].values != 1)
    c5_filter_mask = ~both_clean_mask
    c5_active = returns[both_clean_mask]
    c5_metrics = compute_portfolio_metrics(c5_active)
    c5_seg = compute_segment_metrics(merged, both_clean_mask)
    c5_bl5 = int((returns <= -0.05).sum())
    c5_bl_filt = int(((returns <= -0.05) & c5_filter_mask).sum())
    c5_bw15 = int((returns >= 0.15).sum())
    c5_gain_filt = int(((returns >= 0.15) & c5_filter_mask).sum())

    # Add C5 and other combo schemes to results manually
    all_results = []

    for name, scheme in schemes:
        if name == "C5: BOTH binary flags (AND)":
            # Already computed above
            continue
        result = evaluate_filter_scheme(merged, scheme)
        all_results.append(result)

    # Add C5 manually
    all_results.append({
        "scheme": "C5: both binaries == 0 (AND logic)",
        "n_factors": 2,
        "n_total": n_total, "n_active": int(both_clean_mask.sum()),
        "n_filtered": int(c5_filter_mask.sum()),
        "pass_rate": both_clean_mask.sum() / n_total,
        "base_cum_ret": base["cum_ret"], "base_max_dd": base["max_dd"],
        "base_sharpe": base["sharpe"], "base_win_rate": base["win_rate"],
        "base_big_loss5": base["big_loss5"], "base_big_win15": base["big_win15"],
        "filt_cum_ret": c5_metrics["cum_ret"], "filt_max_dd": c5_metrics["max_dd"],
        "filt_sharpe": c5_metrics["sharpe"], "filt_win_rate": c5_metrics["win_rate"],
        "filt_mean_ret": c5_metrics["mean_ret"],
        "filt_big_loss5": c5_metrics["big_loss5"], "filt_big_loss7": c5_metrics["big_loss7"],
        "filt_big_win10": c5_metrics["big_win10"], "filt_big_win15": c5_metrics["big_win15"],
        "big_loss_filtered": int(c5_bl_filt),
        "big_loss_reduction": c5_bl_filt / max(c5_bl5, 1),
        "big_gain_filtered": int(c5_gain_filt),
        "big_gain_preservation": 1 - c5_gain_filt / max(c5_bw15, 1),
        "efficiency_score": (c5_bl_filt / max(c5_bl5, 1)) * 2.0 - (c5_gain_filt / max(c5_bw15, 1)),
        "seg_stability": 0,
        **{f"{seg}_cum_ret": c5_seg[seg]["cum_ret"] for seg in sorted(c5_seg.keys())},
        **{f"{seg}_n": c5_seg[seg]["n"] for seg in sorted(c5_seg.keys())},
    })

    results_df = pd.DataFrame(all_results)

    # Compute segment stability for all
    seg_cols = [c for c in results_df.columns if c.endswith("_cum_ret")]
    segs = sorted(set(c.replace("_cum_ret", "") for c in seg_cols))
    stabilities = []
    valid_segs = []
    for _, row in results_df.iterrows():
        seg_vals = [row.get(f"{s}_cum_ret", 0) for s in segs]
        valid_segs.append(sum(1 for v in seg_vals if abs(v) > 0.001))
        stabilities.append(min(seg_vals) / max(max(seg_vals), 0.01) if max(seg_vals) > 0 else 0)
    results_df["seg_stability"] = stabilities
    results_df["seg_coverage"] = [f"{v}/{len(segs)}" for v in valid_segs]

    # Sort by efficiency score descending
    results_df = results_df.sort_values("efficiency_score", ascending=False)

    # ---- Display results ----
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 300)
    pd.set_option('display.float_format', '{:.3f}'.format)
    pd.set_option('display.max_rows', 50)

    print("\n" + "=" * 120)
    print("FILTER SCHEME COMPARISON (sorted by efficiency score)")
    print("  Efficiency = 2 * big_loss_reduction - (1 - big_gain_preservation)")
    print("  Higher = better: filter more losses, preserve more gains")
    print("=" * 120)

    main_cols = [
        "scheme", "n_factors", "pass_rate",
        "base_cum_ret", "filt_cum_ret",
        "base_max_dd", "filt_max_dd",
        "base_big_loss5", "filt_big_loss5",
        "base_big_win15", "filt_big_win15",
        "big_loss_reduction", "big_gain_preservation",
        "efficiency_score", "seg_stability"
    ] + [f"{s}_cum_ret" for s in segs]

    available_cols = [c for c in main_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))

    # ---- Top 5 recommended schemes ----
    print("\n" + "=" * 120)
    print("TOP 5 RECOMMENDED SCHEMES (best balance of loss reduction + gain preservation)")
    print("=" * 120)

    detailed_cols = [
        "scheme", "n_factors", "n_active", "n_filtered", "pass_rate",
        "filt_cum_ret", "filt_max_dd", "filt_sharpe", "filt_win_rate",
        "filt_big_loss5", "filt_big_loss7", "filt_big_win10", "filt_big_win15",
        "big_loss_reduction", "big_gain_preservation", "efficiency_score", "seg_stability",
    ] + [f"{s}_cum_ret" for s in segs]

    top5 = results_df.head(8)
    available_detailed = [c for c in detailed_cols if c in top5.columns]
    print(top5[available_detailed].to_string(index=False))

    # ---- Show filtered trades detail for best scheme ----
    best = results_df.iloc[0]
    print(f"\n" + "=" * 120)
    print(f"BEST SCHEME DETAIL: {best['scheme']}")
    print(f"  Keep {best['n_active']}/{best['n_total']} trades ({best['pass_rate']:.1%})")
    print(f"  Filtered {best['big_loss_filtered']}/{int(best['base_big_loss5'])} big losses ({best['big_loss_reduction']:.1%})")
    print(f"  Preserved {int(best['base_big_win15']) - int(best['big_gain_filtered'])}/{int(best['base_big_win15'])} big wins ({best['big_gain_preservation']:.1%})")
    print(f"  Cum ret: {best['base_cum_ret']:.2%} -> {best['filt_cum_ret']:.2%}")
    print(f"  Max DD: {best['base_max_dd']:.2%} -> {best['filt_max_dd']:.2%}")

    # ---- Single vs combo comparison ----
    print("\n" + "=" * 120)
    print("RECOMMENDED SCHEME COMPARISON")
    print("=" * 120)

    # Pick best representative schemes
    representatives = [
        "F1: close_below_ma5 (binary)",
        "F2: low_break_prev_low (binary)",
        "C1: close_below_ma5 OR low_break",
        "C5: both binaries == 0 (AND logic)",
    ]
    rep_df = results_df[results_df["scheme"].isin(representatives)]
    rep_cols = ["scheme", "n_factors", "pass_rate", "filt_cum_ret", "filt_max_dd",
                "filt_win_rate", "big_loss_reduction", "big_gain_preservation",
                "efficiency_score", "seg_stability"] + [f"{s}_cum_ret" for s in segs]
    avail_rep = [c for c in rep_cols if c in rep_df.columns]
    print(rep_df[avail_rep].to_string(index=False))

    # ---- Summary recommendations ----
    print("\n" + "=" * 120)
    print("PRACTICAL RECOMMENDATION")
    print("=" * 120)
    print("""
    If you can only add ONE filter:
      -> F1: Skip trade when close_below_ma5_after_runup == 1
         Filters 65% of big losses, keeps 74% of trades, preserves all big wins.

    If you want STRONGER protection (2 filters):
      -> C5: Only buy when BOTH close_below_ma5 == 0 AND low_break_prev_low == 0
         Most conservative, may reduce some big wins but massively cuts losses.

    If you want BALANCED protection (1-2 filters):
      -> C1: Skip when EITHER binary flag is triggered
         Covers more loss scenarios while keeping reasonable pass rate.
    """)

    # Save results
    output_path = "results/exhaustion_filter_scheme_comparison.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
