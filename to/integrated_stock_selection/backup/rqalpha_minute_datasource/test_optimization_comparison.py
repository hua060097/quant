"""
性能优化对比测试

比较原版和优化版的5分钟回测性能，并验证交易逻辑修复

作者: Claude Code Assistant
日期: 2025-11-22
"""

import logging
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append('.')

def generate_test_data():
    """生成测试数据"""
    dates = pd.date_range('2024-10-08', '2024-10-15', freq='D')
    dates = [d for d in dates if d.weekday() < 5]

    stock_codes = [
        "000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "600519.SH",
        "000858.SZ", "002415.SZ", "300059.SZ", "002594.SZ", "000725.SZ"
    ]

    data = []
    for date in dates:
        for stock_code in stock_codes:
            record = {
                'date': date,
                'stock_code': stock_code,
                'predicted_return': np.random.normal(0, 0.02),
                'future_return_2d': np.random.normal(0, 0.03),
                'is_can_buy': 1,
                'pct_chg': np.random.normal(0, 0.02),
                'volume': np.random.randint(1000000, 10000000),
                'amount': np.random.randint(10000000, 100000000),
                'open': np.random.uniform(10, 100),
                'high': np.random.uniform(10, 100),
                'low': np.random.uniform(10, 100),
                'close': np.random.uniform(10, 100),

                # 技术指标
                'atr14': np.random.uniform(0.01, 0.05),
                'price_ema20_ratio': np.random.uniform(0.95, 1.05),
                'price_ema50_ratio': np.random.uniform(0.9, 1.1),
                'ema20_ema50_ratio': np.random.uniform(0.95, 1.05),
                'ema50_slope': np.random.normal(0, 0.01),
                'log_price_regression_slope': np.random.normal(0, 0.001),
                'log_price_regression_r2': np.random.uniform(0.1, 0.8),
                'adx_14': np.random.uniform(10, 50),
                'drawdown_from_20d_high_pct': np.random.uniform(-0.2, 0),
                'drawdown_atr_ratio': np.random.uniform(-3, 0),
                'atr20': np.random.uniform(0.01, 0.1),
                'negative_days_5d': np.random.randint(0, 5),
                'large_red_candle_5d': np.random.randint(0, 5),
                'distance_to_support_atr': np.random.uniform(0, 3),
                'bollinger_percent_b': np.random.uniform(0, 1),
                'rsi_14': np.random.uniform(20, 80),

                # 涨跌停状态
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_zhaban': 0,
                'limit_strength': np.random.uniform(0, 1),
                'limit_times': np.random.randint(0, 5),
                'consecutive_limit_up': 0,
                'consecutive_limit_down': 0,

                # 技术分析特征
                'near_support': np.random.choice([0, 1]),
                'near_resistance': np.random.choice([0, 1]),
                'short_reversal': np.random.choice([0, 1])
            }
            data.append(record)

    return pd.DataFrame(data)

def test_original_performance(test_data):
    """测试原版性能（备份版本）"""
    logger.info("=== 测试原版性能（备份版本） ===")

    try:
        from minute_backtest_original import MinuteLevelStockSelector as OriginalSelector

        minute_data_path = "../local_data/minute/"
        selector = OriginalSelector(minute_data_path, test_data)

        start_time = time.time()
        result = selector.run_minute_level_backtest(stock_counts=[1, 3], holding_days=2)
        end_time = time.time()

        execution_time = end_time - start_time

        # 提取交易信息检查逻辑
        trades = []
        if 'detailed_trades' in result:
            for stock_count, stock_trades in result['detailed_trades'].items():
                trades.extend(stock_trades)

        return {
            'execution_time': execution_time,
            'total_trades': len(trades),
            'result': result,
            'trades_sample': trades[:2] if trades else []  # 取前2笔交易检查逻辑
        }

    except Exception as e:
        logger.error(f"原版测试失败: {e}")
        return None

def test_optimized_performance(test_data):
    """测试优化版性能（当前主版本）"""
    logger.info("=== 测试优化版性能（当前主版本） ===")

    try:
        from minute_backtest import MinuteLevelStockSelector

        minute_data_path = "../local_data/minute/"
        selector = MinuteLevelStockSelector(minute_data_path, test_data)

        start_time = time.time()
        result = selector.run_minute_level_backtest(stock_counts=[1, 3], holding_days=2)
        end_time = time.time()

        execution_time = end_time - start_time

        # 提取交易信息检查逻辑
        trades = []
        if 'detailed_trades' in result:
            for stock_count, stock_trades in result['detailed_trades'].items():
                trades.extend(stock_trades)

        return {
            'execution_time': execution_time,
            'total_trades': len(trades),
            'result': result,
            'trades_sample': trades[:2] if trades else []  # 取前2笔交易检查逻辑
        }

    except Exception as e:
        logger.error(f"优化版测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_trading_logic(trades_sample, version_name):
    """分析交易逻辑"""
    logger.info(f"=== 分析{version_name}交易逻辑 ===")

    if not trades_sample:
        logger.warning("没有交易记录可供分析")
        return

    for i, trade in enumerate(trades_sample[:2]):  # 分析前2笔交易
        logger.info(f"交易{i+1}: {trade['stock_code']}")
        logger.info(f"  买入时间: {trade['buy_datetime']}, 价格: {trade['buy_price']:.4f}")
        logger.info(f"  卖出时间: {trade['sell_datetime']}, 价格: {trade['sell_price']:.4f}")
        logger.info(f"  收益率: {trade['actual_return']:.4f}")

        if 'price_logic' in trade:
            logger.info(f"  价格逻辑: {trade['price_logic']}")

def compare_performance(original_result, optimized_result):
    """性能对比"""
    logger.info("=== 性能对比结果 ===")

    if original_result and optimized_result:
        speedup = original_result['execution_time'] / optimized_result['execution_time']
        logger.info(f"原版执行时间: {original_result['execution_time']:.3f}秒")
        logger.info(f"优化版执行时间: {optimized_result['execution_time']:.3f}秒")
        logger.info(f"性能提升: {speedup:.2f}倍")

        logger.info(f"原版交易数: {original_result['total_trades']}")
        logger.info(f"优化版交易数: {optimized_result['total_trades']}")

        # 交易逻辑对比
        analyze_trading_logic(original_result['trades_sample'], "原版")
        analyze_trading_logic(optimized_result['trades_sample'], "优化版")

    else:
        logger.error("性能对比失败：缺少测试结果")

def main():
    """主测试函数"""
    logger.info("🚀 开始5分钟回测性能优化对比测试")
    logger.info("=" * 60)

    # 生成测试数据
    logger.info("生成测试数据...")
    test_data = generate_test_data()
    logger.info(f"测试数据: {len(test_data)}条记录")

    # 测试原版性能
    original_result = test_original_performance(test_data)

    # 测试优化版性能
    optimized_result = test_optimized_performance(test_data)

    # 性能对比
    compare_performance(original_result, optimized_result)

    logger.info("\n" + "=" * 60)
    logger.info("✅ 性能对比测试完成")
    logger.info("=" * 60)

    # 总结优化效果
    logger.info("\n📋 优化总结:")
    logger.info("1. 🔧 预处理股票数据映射 - 避免重复扫描，从O(n*m)降到O(1)")
    logger.info("2. ✅ 修复买入价格逻辑 - T+1使用open价格而非close")
    logger.info("3. 📦 批量数据获取 - 减少重复的日期数据加载")
    logger.info("4. ⚡ 向量化过滤 - 替代逐行循环提升性能")

    if original_result and optimized_result:
        speedup = original_result['execution_time'] / optimized_result['execution_time']
        logger.info(f"5. 🚀 整体性能提升: {speedup:.2f}倍")

if __name__ == "__main__":
    main()