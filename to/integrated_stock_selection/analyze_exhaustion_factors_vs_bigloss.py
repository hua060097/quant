"""
分析 high_position_exhaustion 因子与大亏/大赚股票的关系

从日志中提取所有交易记录，识别大亏(<-5%)和大赚(>15%)样本，
然后加载对应的因子数据，分析因子值与盈亏的相关性，
判断哪些因子可以作为买入过滤条件。

用法: python analyze_exhaustion_factors_vs_bigloss.py
"""

import re
import os
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 1. 从日志解析交易记录
# ============================================================
LOG_PATH = "stock_selection_test_mode_walkforward_分类模型_原因子50.log"
FACTORS_PATH = "factor_lab_data/factors.parquet"

# 新增的 high_position_exhaustion 因子列表
EXHAUSTION_FACTORS = [
    "close_position_20d",       # [0,1] 收盘价在近20日高低区间位置
    "close_position_60d",       # [0,1] 收盘价在近60日高低区间位置
    "upper_shadow_ratio_3d_mean", # [0,0.5] 近3日平均上影线比例
    "close_to_high_3d_mean",    # [-1,0] 近3日收盘相对最高价距离
    "turnover_z_20",            # [-6,6] 成交/换手异常z-score
    "high_volume_no_gain_3d",   # [-6,6] 近3日放量滞涨强度
    "pullback_depth_from_10d_high", # [-1,0] 相对10日最高回撤幅度
    "high_pos_upper_shadow",    # [-6,6] 高位×上影线交互
    "high_pos_volume_no_gain",  # [-6,6] 高位×放量滞涨交互
    "high_pos_reversal_volume", # [-6,6] 高位×上影线×放量三元交互
    "ret_3d_vs_10d_accel",      # [-6,6] 3日vs10日加速度
    "ret_5d_vs_20d_accel",      # [-6,6] 5日vs20日加速度
    "momentum_peak_decay_3d",   # [-6,6] 动量峰值衰减
    "consecutive_up_days_5d",   # [0,1] 近5日上涨天数占比
    "large_up_days_10d",        # [0,1] 近10日大涨(>5%)天数占比
    "gap_up_count_5d",          # [0,1] 近5日跳空高开>2%次数占比
    "close_below_ma5_after_runup", # 0/1 10日涨>10%后跌破MA5
    "low_break_prev_low_after_runup", # 0/1 10日涨>10%后破前低
]


def parse_trades_from_log(log_path: str) -> pd.DataFrame:
    """从日志解析所有交易记录，返回 (stock_code, buy_date, sell_date, return_pct, ...)"""
    buy_pattern = re.compile(
        r"买入成功:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*([\d.]+),\s*数量:\s*(\d+),\s*时间:\s*(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})"
    )
    sell_pattern = re.compile(
        r"卖出:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*([\d.]+),\s*收益:\s*([-\d.]+)\s*\(([-\d.]+)%\),\s*原因:\s*(.+)"
    )

    trades = []
    pending_buy = {}  # stock_code -> {buy_date, buy_price, buy_time}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            buy_m = buy_pattern.search(line)
            if buy_m:
                stock = buy_m.group(1)
                price = float(buy_m.group(2))
                date = buy_m.group(4)
                pending_buy[stock] = {"buy_date": date, "buy_price": price}
                continue

            sell_m = sell_pattern.search(line)
            if sell_m:
                stock = sell_m.group(1)
                ret_pct = float(sell_m.group(3))
                reason = sell_m.group(5).strip()
                if stock in pending_buy:
                    buy_info = pending_buy.pop(stock)
                    trades.append({
                        "stock_code": stock,
                        "buy_date": buy_info["buy_date"],
                        "buy_price": buy_info["buy_price"],
                        "return_pct": ret_pct,
                        "sell_reason": reason,
                    })

    df = pd.DataFrame(trades)
    logger.info(f"解析到 {len(df)} 条交易记录")
    return df


def classify_trades(df: pd.DataFrame) -> dict:
    """分类交易：大亏 / 大赚 / 普通"""
    big_loss = df[df["return_pct"] <= -0.05].sort_values("return_pct")
    big_gain = df[df["return_pct"] >= 0.15].sort_values("return_pct", ascending=False)
    normal = df[(df["return_pct"] > -0.05) & (df["return_pct"] < 0.15)]

    logger.info(f"大亏 (<= -5%): {len(big_loss)} 笔")
    logger.info(f"大赚 (>= 15%): {len(big_gain)} 笔")
    logger.info(f"普通: {len(normal)} 笔")

    return {"big_loss": big_loss, "big_gain": big_gain, "normal": normal}


# ============================================================
# 2. 加载因子数据并匹配
# ============================================================
def load_factors_for_dates(factors_path: str, stocks: list, dates: list) -> pd.DataFrame:
    """加载指定股票和日期的因子数据"""
    if not os.path.exists(factors_path):
        logger.error(f"因子文件不存在: {factors_path}")
        return pd.DataFrame()

    logger.info(f"加载因子数据: {factors_path}")
    # 先检查有哪些列可用
    all_columns = pd.read_parquet(factors_path, columns=['date', 'stock_code']).columns.tolist()
    # 追加检查
    import pyarrow.parquet as pq
    schema = pq.read_schema(factors_path)
    all_columns = schema.names

    # 找到可用的因子列
    available = [c for c in EXHAUSTION_FACTORS if c in all_columns]
    missing = [c for c in EXHAUSTION_FACTORS if c not in all_columns]
    if missing:
        logger.warning(f"缺失因子: {missing}")
    logger.info(f"可用因子: {len(available)}/{len(EXHAUSTION_FACTORS)}: {available}")

    if not available:
        logger.error("无可用因子")
        return pd.DataFrame(), []

    # 只加载需要的列，减少内存占用
    cols = ["date", "stock_code"] + available
    logger.info(f"读取 {len(cols)} 列...")
    all_factors = pd.read_parquet(factors_path, columns=cols)
    all_factors["date"] = pd.to_datetime(all_factors["date"])

    # 按日期范围过滤
    min_date = pd.to_datetime(min(dates))
    max_date = pd.to_datetime(max(dates))
    all_factors = all_factors[
        (all_factors["date"] >= min_date) & (all_factors["date"] <= max_date)
    ]

    return all_factors, available


def match_factors_to_trades(trades_df: pd.DataFrame, factors_df: pd.DataFrame,
                            factor_cols: list) -> pd.DataFrame:
    """将因子数据匹配到每笔交易的买入日期"""
    factors_df = factors_df.copy()
    factors_df["date"] = pd.to_datetime(factors_df["date"])

    # 转换交易日期
    trades_df = trades_df.copy()
    trades_df["buy_date_dt"] = pd.to_datetime(trades_df["buy_date"])

    # Merge
    merged = trades_df.merge(
        factors_df,
        left_on=["buy_date_dt", "stock_code"],
        right_on=["date", "stock_code"],
        how="left"
    )

    matched = merged.dropna(subset=factor_cols[:1])  # 至少有一个因子有值
    unmatched = len(trades_df) - len(matched)
    if unmatched > 0:
        logger.warning(f"{unmatched} 笔交易未能匹配到因子数据")

    return merged


# ============================================================
# 3. 分析函数
# ============================================================
def analyze_factor_correlation(merged_df: pd.DataFrame, factor_cols: list):
    """分析每个因子与收益的相关性，以及大亏/大赚组的因子分布差异"""
    results = []

    for col in factor_cols:
        valid = merged_df[col].dropna()
        if len(valid) == 0:
            continue

        # 与 return_pct 的相关系数
        corr = merged_df["return_pct"].corr(merged_df[col])

        # 分组统计
        big_loss_mask = merged_df["return_pct"] <= -0.05
        big_gain_mask = merged_df["return_pct"] >= 0.15
        normal_mask = ~big_loss_mask & ~big_gain_mask

        loss_vals = merged_df.loc[big_loss_mask, col]
        gain_vals = merged_df.loc[big_gain_mask, col]
        normal_vals = merged_df.loc[normal_mask, col]

        # 大亏组均值 - 大赚组均值：正数=大亏时因子更高
        diff = loss_vals.mean() - gain_vals.mean() if len(loss_vals) > 0 and len(gain_vals) > 0 else np.nan

        # 大亏组均值 - 全样本均值
        diff_vs_all = loss_vals.mean() - merged_df[col].mean() if len(loss_vals) > 0 else np.nan

        results.append({
            "factor": col,
            "corr_with_return": corr,
            "big_loss_mean": loss_vals.mean(),
            "big_gain_mean": gain_vals.mean(),
            "normal_mean": normal_vals.mean(),
            "all_mean": merged_df[col].mean(),
            "loss_vs_gain_diff": diff,         # >0 意味着大亏时因子更高（可以做反向过滤）
            "loss_vs_all_diff": diff_vs_all,
            "big_loss_std": loss_vals.std(),
            "big_gain_std": gain_vals.std(),
            "big_loss_n": len(loss_vals),
            "big_gain_n": len(gain_vals),
            "total_n": len(merged_df),
            "nan_rate": merged_df[col].isna().mean(),
        })

    result_df = pd.DataFrame(results).sort_values("loss_vs_gain_diff", ascending=False)
    return result_df


def find_filter_thresholds(merged_df: pd.DataFrame, factor_cols: list,
                           big_loss_threshold: float = -0.05):
    """对于有区分度的因子，找到合适的过滤阈值"""
    thresholds = []
    big_loss_mask = merged_df["return_pct"] <= big_loss_threshold

    for col in factor_cols:
        valid = merged_df[col].dropna()
        if len(valid) < 10:
            continue

        loss_vals = merged_df.loc[big_loss_mask, col].dropna()
        if len(loss_vals) < 3:
            continue

        all_vals = merged_df[col].dropna()

        # 尝试不同的分位数作为阈值
        for pct in [50, 60, 70, 75, 80, 85, 90, 95]:
            threshold = all_vals.quantile(pct / 100)
            # 高因子值被过滤掉（假设因子值越高越危险）
            filtered = merged_df[merged_df[col] <= threshold]
            filtered_loss = filtered[filtered["return_pct"] <= big_loss_threshold]
            kept = merged_df[merged_df[col] > threshold]
            kept_loss = kept[kept["return_pct"] <= big_loss_threshold]

            # 过滤后保留的大亏数量 vs 总大亏数量
            loss_capture_rate = len(filtered_loss) / max(len(loss_vals), 1)
            total_pass_rate = len(filtered) / max(len(merged_df), 1)
            # 过滤掉的大亏比例
            loss_filtered_rate = 1.0 - loss_capture_rate

            thresholds.append({
                "factor": col,
                "percentile": pct,
                "threshold": threshold,
                "total_trades": len(merged_df),
                "big_loss_total": len(loss_vals),
                "pass_through": len(filtered),
                "big_loss_remaining": len(filtered_loss),
                "loss_capture_rate": loss_capture_rate,
                "loss_filtered_rate": loss_filtered_rate,  # 成功过滤掉的大亏比例
                "total_pass_rate": total_pass_rate,
            })

    return pd.DataFrame(thresholds)


def print_big_trades_detail(merged_df: pd.DataFrame, factor_cols: list):
    """打印大亏和大赚的具体交易及其因子值"""
    big_loss = merged_df[merged_df["return_pct"] <= -0.05].sort_values("return_pct")
    big_gain = merged_df[merged_df["return_pct"] >= 0.15].sort_values("return_pct", ascending=False)

    print("\n" + "=" * 120)
    print("📉 大亏交易明细 (return <= -5%)")
    print("=" * 120)
    cols = ["stock_code", "buy_date", "return_pct", "sell_reason"] + [
        c for c in factor_cols if c in merged_df.columns
    ]
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(big_loss[cols].to_string(index=False))

    print("\n" + "=" * 120)
    print("📈 大赚交易明细 (return >= 15%)")
    print("=" * 120)
    print(big_gain[cols].to_string(index=False))


def print_factor_distribution_analysis(result_df: pd.DataFrame):
    """打印因子区分度分析"""
    print("\n" + "=" * 120)
    print("🔍 因子对大亏/大赚的区分度分析 (按 loss_vs_gain_diff 降序)")
    print("   loss_vs_gain_diff > 0: 因子值在大亏时更高 → 可做反向过滤")
    print("   loss_vs_gain_diff < 0: 因子值在大赚时更高 → 可做正向筛选")
    print("=" * 120)
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)

    display_cols = [
        "factor", "corr_with_return", "big_loss_mean", "big_gain_mean",
        "normal_mean", "loss_vs_gain_diff", "loss_vs_all_diff",
        "big_loss_n", "big_gain_n", "total_n", "nan_rate"
    ]
    print(result_df[display_cols].to_string(index=False))


def print_filter_recommendations(threshold_df: pd.DataFrame):
    """打印过滤建议"""
    if threshold_df.empty:
        print("\n无过滤阈值数据")
        return

    # 找出每个因子最优的阈值（过滤掉最多大亏且通过率合理的）
    best = threshold_df[
        (threshold_df["total_pass_rate"] >= 0.5)  # 至少保留50%的样本
    ].sort_values("loss_filtered_rate", ascending=False).groupby("factor").head(3)

    print("\n" + "=" * 120)
    print("🛡️ 推荐过滤阈值 (按过滤大亏效率排序，保留>=50%总样本)")
    print("=" * 120)
    display_cols = [
        "factor", "percentile", "threshold", "big_loss_total",
        "loss_filtered_rate", "total_pass_rate"
    ]
    print(best[display_cols].to_string(index=False))


# ============================================================
# 4. 主流程
# ============================================================
def main():
    print("=" * 80)
    print("高位衰竭因子 vs 大亏/大赚 分析")
    print("=" * 80)

    # Step 1: 解析日志
    print("\n[Step 1] 解析交易日志...")
    trades_df = parse_trades_from_log(LOG_PATH)
    if trades_df.empty:
        logger.error("未能解析到任何交易记录")
        return

    classified = classify_trades(trades_df)
    print(f"\n交易日期范围: {trades_df['buy_date'].min()} ~ {trades_df['buy_date'].max()}")
    print(f"\n大亏Top5:")
    print(classified["big_loss"][["stock_code", "buy_date", "return_pct", "sell_reason"]].head(10).to_string(index=False))
    print(f"\n大赚Top5:")
    print(classified["big_gain"][["stock_code", "buy_date", "return_pct", "sell_reason"]].head(10).to_string(index=False))

    # Step 2: 加载因子数据
    print("\n[Step 2] 加载因子数据...")
    all_dates = sorted(trades_df["buy_date"].unique())
    factors_df, available_factors = load_factors_for_dates(
        FACTORS_PATH,
        trades_df["stock_code"].tolist(),
        all_dates
    )

    if factors_df.empty:
        # 尝试使用替代路径
        alt_paths = [
            "factor_lab_data/factors.parquet",
            "../factor_lab_data/factors.parquet",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                factors_df, available_factors = load_factors_for_dates(
                    alt, trades_df["stock_code"].tolist(), all_dates
                )
                if not factors_df.empty:
                    break
        if factors_df.empty:
            logger.error("无法加载因子数据，请检查路径")
            return

    if not available_factors:
        logger.error("没有任何可用的 exhaustion 因子！")
        return

    # Step 3: 匹配因子到交易
    print("\n[Step 3] 匹配因子数据...")
    merged = match_factors_to_trades(trades_df, factors_df, available_factors)

    # Step 4: 分析相关性
    print("\n[Step 4] 分析因子区分度...")
    result_df = analyze_factor_correlation(merged, available_factors)
    print_factor_distribution_analysis(result_df)

    # Step 5: 找过滤阈值
    print("\n[Step 5] 搜索最优过滤阈值...")
    threshold_df = find_filter_thresholds(merged, available_factors)
    print_filter_recommendations(threshold_df)

    # Step 6: 打印详细交易（带因子值）
    print("\n[Step 6] 详细交易...")
    print_big_trades_detail(merged, available_factors)

    # Step 7: 总结
    print("\n" + "=" * 120)
    print("📋 总结建议")
    print("=" * 120)
    top_factors = result_df.dropna(subset=["loss_vs_gain_diff"]).head(5)
    for _, row in top_factors.iterrows():
        direction = "🚫 反向过滤" if row["loss_vs_gain_diff"] > 0 else "✅ 正向筛选"
        print(f"  {direction}: {row['factor']}")
        print(f"    大亏均值={row['big_loss_mean']:.4f}, 大赚均值={row['big_gain_mean']:.4f}, "
              f"差值={row['loss_vs_gain_diff']:.4f}, 相关系数={row['corr_with_return']:.4f}")

    if not threshold_df.empty:
        best_filter = threshold_df.sort_values("loss_filtered_rate", ascending=False).head(3)
        print(f"\n  推荐过滤规则:")
        for _, row in best_filter.iterrows():
            print(f"    {row['factor']} > {row['threshold']:.4f} 时过滤"
                  f" (分位P{row['percentile']}, 过滤{row['loss_filtered_rate']:.1%}大亏, "
                  f"通过率{row['total_pass_rate']:.1%})")


if __name__ == "__main__":
    main()
