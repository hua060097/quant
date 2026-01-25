"""
5分钟级别选股回测使用示例

演示如何使用新的5分钟级别回测功能。
支持Top-K选股（K=1,2,3,5）与原有日线回测的无缝切换。

作者: Claude Code Assistant
日期: 2025-11-22
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mock_prediction_data(start_date="2024-10-01", end_date="2024-10-10"):
    """
    生成模拟预测数据用于测试

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        pd.DataFrame: 模拟预测数据
    """
    # 生成日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # 过滤工作日
    dates = [d for d in dates if d.weekday() < 5]

    # 模拟股票代码（简化版）
    stock_codes = [
        "000001.SZ", "000002.SZ", "000858.SZ", "000895.SZ", "002415.SZ",
        "600000.SH", "600036.SH", "600519.SH", "600887.SH", "601318.SH"
    ]

    data = []
    for date in dates:
        for stock_code in stock_codes:
            # 生成随机预测值和技术指标
            predicted_return = np.random.normal(0, 0.02)  # 预测收益率
            future_return_2d = np.random.normal(0, 0.03)   # 未来2日收益

            # 生成模拟技术指标
            record = {
                'date': date,
                'stock_code': stock_code,
                'predicted_return': predicted_return,
                'future_return_2d': future_return_2d,
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
                'is_limit_up': np.random.choice([0, 0, 0, 1], p=[0.95, 0.03, 0.01, 0.01]),  # 1%涨停概率
                'is_limit_down': np.random.choice([0, 0, 0, 1], p=[0.95, 0.03, 0.01, 0.01]),  # 1%跌停概率
                'is_zhaban': 0,  # 炸板股
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


def example_daily_backtest():
    """
    示例：日线级别选股回测（原有功能）
    """
    logger.info("=" * 60)
    logger.info("🔍 示例1: 日线级别选股回测")
    logger.info("=" * 60)

    try:
        # 生成模拟数据
        prediction_data = generate_mock_prediction_data("2024-10-01", "2024-10-10")
        logger.info(f"生成模拟预测数据: {len(prediction_data)}条记录")

        # 调用日线回测
        from main import _run_stock_selection_backtest

        result = _run_stock_selection_backtest(
            backtest_data=prediction_data,
            stock_counts=[1, 2, 3, 5],
            window_name="daily_example",
            market_data=None,
            frequency='daily'  # 显式指定日线回测
        )

        # 输出结果
        logger.info("\n📊 日线回测结果:")
        for stock_count, stats in result.get('backtest_results', {}).items():
            logger.info(f"Top-{stock_count}: 收益={stats.get('total_return', 0):.4f}, "
                       f"夏普={stats.get('sharpe_ratio', 0):.4f}, "
                       f"胜率={stats.get('win_rate', 0):.4f}")

        return result

    except Exception as e:
        logger.error(f"日线回测示例失败: {e}")
        return None


def example_minute_backtest():
    """
    示例：5分钟级别选股回测（新功能）
    """
    logger.info("=" * 60)
    logger.info("⏰ 示例2: 5分钟级别选股回测")
    logger.info("=" * 60)

    try:
        # 生成模拟数据
        prediction_data = generate_mock_prediction_data("2024-10-08", "2024-10-10")  # 选择有5分钟数据的日期
        logger.info(f"生成模拟预测数据: {len(prediction_data)}条记录")

        # 检查5分钟数据是否存在
        minute_data_path = "../local_data/minute/202410/minute_data.parquet"
        if not os.path.exists(minute_data_path):
            logger.warning(f"⚠️ 未找到5分钟数据文件: {minute_data_path}")
            logger.info("   请确保5分钟数据已下载并解压到正确路径")
            logger.info("   将跳过5分钟回测示例")
            return None

        # 调用5分钟回测
        from main import _run_stock_selection_backtest

        result = _run_stock_selection_backtest(
            backtest_data=prediction_data,
            stock_counts=[1, 2, 3, 5],
            window_name="minute_example",
            market_data=None,
            frequency='minute',      # 指定分钟级回测
            holding_minutes=30       # 持仓30分钟
        )

        # 输出结果
        logger.info("\n📊 5分钟回测结果:")
        for stock_count, stats in result.get('backtest_results', {}).items():
            if 'total_trades' in stats and stats['total_trades'] > 0:
                logger.info(f"Top-{stock_count}: 收益={stats.get('total_return', 0):.4f}, "
                           f"年化收益={stats.get('annual_return', 0):.4f}, "
                           f"夏普={stats.get('sharpe_ratio', 0):.4f}, "
                           f"胜率={stats.get('win_rate', 0):.4f}, "
                           f"交易次数={stats.get('total_trades', 0)}")
            else:
                logger.info(f"Top-{stock_count}: 无交易产生")

        return result

    except Exception as e:
        logger.error(f"5分钟回测示例失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None


def example_rqalpha_integration():
    """
    示例：RQAlpha集成使用
    """
    logger.info("=" * 60)
    logger.info("🔗 示例3: RQAlpha 5分钟数据源集成")
    logger.info("=" * 60)

    try:
        # 检查数据路径
        minute_data_path = "../local_data/minute/"
        if not os.path.exists(minute_data_path):
            logger.warning(f"⚠️ 未找到5分钟数据目录: {minute_data_path}")
            logger.info("   将跳过RQAlpha集成示例")
            return None

        # 导入RQAlpha数据源
        from rqalpha_minute_datasource import MinuteKDataSource

        # 创建数据源实例
        data_source = MinuteKDataSource(
            data_path=minute_data_path,
            cache_size=50
        )

        logger.info("✅ 5分钟数据源创建成功")
        logger.info(f"   支持股票数量: {len(data_source.stock_list)}")
        logger.info(f"   数据时间范围: {data_source.date_range[0]} ~ {data_source.date_range[1]}")

        # 测试数据获取
        if data_source.stock_list:
            sample_stock = data_source.stock_list[0]
            sample_date = data_source.date_range[0] + timedelta(days=5)

            # 创建模拟的instrument对象
            class MockInstrument:
                def __init__(self, order_book_id):
                    self.order_book_id = order_book_id

            mock_instrument = MockInstrument(sample_stock)

            # 获取K线数据
            bar_data = data_source.get_bar(mock_instrument, sample_date, '5m')
            if bar_data:
                logger.info(f"✅ 成功获取 {sample_stock} 在 {sample_date} 的5分钟K线数据")
                logger.info(f"   开盘价: {bar_data['open']:.2f}, 收盘价: {bar_data['close']:.2f}")

            # 获取历史K线
            history_data = data_source.history_bars(
                mock_instrument, 10, '5m',
                ['open', 'close', 'volume'], sample_date
            )
            if history_data is not None:
                logger.info(f"✅ 成功获取 {sample_stock} 的历史5分钟K线数据")
                logger.info(f"   数据形状: {history_data.shape}")

        return data_source

    except Exception as e:
        logger.error(f"RQAlpha集成示例失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None


def example_performance_comparison():
    """
    示例：不同频率回测性能对比
    """
    logger.info("=" * 60)
    logger.info("📈 示例4: 不同频率回测性能对比")
    logger.info("=" * 60)

    try:
        # 生成模拟数据
        prediction_data = generate_mock_prediction_data("2024-10-08", "2024-10-10")
        logger.info(f"生成模拟预测数据: {len(prediction_data)}条记录")

        from main import _run_stock_selection_backtest

        # 配置参数
        stock_counts = [1, 3, 5]
        holding_minutes_options = [15, 30, 60]  # 不同持仓时间

        results = {}

        for holding_minutes in holding_minutes_options:
            logger.info(f"\n⏰ 测试5分钟回测 - 持仓时间: {holding_minutes}分钟")

            result = _run_stock_selection_backtest(
                backtest_data=prediction_data,
                stock_counts=stock_counts,
                window_name=f"minute_{holding_minutes}m",
                frequency='minute',
                holding_minutes=holding_minutes
            )

            results[holding_minutes] = result.get('backtest_results', {})

            # 输出结果
            for stock_count in stock_counts:
                stats = results[holding_minutes].get(stock_count, {})
                if stats.get('total_trades', 0) > 0:
                    logger.info(f"  Top-{stock_count}: 收益={stats.get('total_return', 0):.4f}, "
                               f"夏普={stats.get('sharpe_ratio', 0):.4f}")

        # 总结对比
        logger.info("\n📊 性能对比总结:")
        logger.info("持仓时间 | Top-1收益 | Top-3收益 | Top-5收益 | Top-1夏普 | Top-3夏普 | Top-5夏普")
        logger.info("-" * 70)

        for holding_minutes in holding_minutes_options:
            stats_dict = results[holding_minutes]
            line = f"{holding_minutes:6}分钟 | "
            for stock_count in stock_counts:
                stats = stats_dict.get(stock_count, {})
                return_val = stats.get('total_return', 0)
                sharpe_val = stats.get('sharpe_ratio', 0)
                line += f"{return_val:8.4f} | "

            for stock_count in stock_counts:
                stats = stats_dict.get(stock_count, {})
                sharpe_val = stats.get('sharpe_ratio', 0)
                line += f"{sharpe_val:8.4f} | "
                if stock_count < 5:  # 避免重复添加
                    break

            logger.info(line)

        return results

    except Exception as e:
        logger.error(f"性能对比示例失败: {e}")
        return None


def main():
    """
    主函数：运行所有示例
    """
    logger.info("🚀 开始5分钟选股回测系统示例演示")
    logger.info("=" * 60)

    # 示例1: 日线回测
    daily_result = example_daily_backtest()

    # 示例2: 5分钟回测
    minute_result = example_minute_backtest()

    # 示例3: RQAlpha集成
    data_source = example_rqalpha_integration()

    # 示例4: 性能对比
    comparison_results = example_performance_comparison()

    logger.info("\n" + "=" * 60)
    logger.info("✅ 所有示例演示完成")
    logger.info("=" * 60)

    # 输出总结
    logger.info("\n📋 功能总结:")
    logger.info("✅ 1. 日线级别选股回测 - 保持原有功能不变")
    logger.info("✅ 2. 5分钟级别选股回测 - 新增高频交易回测")
    logger.info("✅ 3. RQAlpha数据源集成 - 支持多种时间频率")
    logger.info("✅ 4. Top-K选股策略 - K=1,2,3,5多档位对比")
    logger.info("✅ 5. 可配置持仓时间 - 支持15/30/60分钟等")
    logger.info("✅ 6. 完整的选股过滤 - 复用原有风控逻辑")


if __name__ == "__main__":
    main()