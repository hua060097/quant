"""
简化的5分钟回测测试

不依赖main.py，直接测试数据源和回测逻辑。

作者: Claude Code Assistant
日期: 2025-11-22
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_mock_prediction_data(start_date="2024-10-08", end_date="2024-10-10"):
    """生成模拟预测数据用于测试"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # 只保留工作日

    # 只使用一些股票代码进行测试
    stock_codes = [
        "000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "600519.SH"
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

def test_minute_data_source():
    """测试5分钟数据源"""
    logger.info("=== 测试5分钟数据源 ===")

    from minute_data_source import MinuteKDataSource

    try:
        data_path = "../local_data/minute/"

        if not os.path.exists(data_path):
            logger.info(f"5分钟数据路径不存在: {data_path}")
            logger.info("创建模拟数据源进行测试...")
            # 创建一个空的目录结构
            os.makedirs(data_path, exist_ok=True)

        # 创建数据源
        data_source = MinuteKDataSource(data_path, cache_size=10)
        logger.info("✅ MinuteKDataSource创建成功")

        # 检查属性
        logger.info(f"   数据路径: {data_source.data_path}")
        logger.info(f"   缓存大小: {data_source.cache_size}")

        return data_source

    except Exception as e:
        logger.error(f"❌ 数据源测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_minute_backtest():
    """测试5分钟回测逻辑"""
    logger.info("=== 测试5分钟回测逻辑 ===")

    try:
        from minute_backtest import MinuteLevelStockSelector

        # 生成模拟数据
        prediction_data = generate_mock_prediction_data("2024-10-08", "2024-10-10")
        logger.info(f"生成模拟预测数据: {len(prediction_data)}条记录")

        # 创建5分钟数据路径（如果不存在）
        minute_data_path = "../local_data/minute/"
        os.makedirs(minute_data_path, exist_ok=True)

        # 创建选股器
        selector = MinuteLevelStockSelector(minute_data_path, prediction_data)
        logger.info("✅ MinuteLevelStockSelector创建成功")

        # 尝试运行回测
        logger.info("开始5分钟回测...")
        result = selector.run_minute_level_backtest(
            stock_counts=[1, 3],
            holding_minutes=30
        )

        logger.info("✅ 5分钟回测完成")

        # 显示结果
        if 'backtest_results' in result:
            for stock_count, stats in result['backtest_results'].items():
                if stats.get('total_trades', 0) > 0:
                    logger.info(f"Top-{stock_count}: 交易{stats.get('total_trades', 0)}次, "
                               f"收益{stats.get('total_return', 0):.4f}")
                else:
                    logger.info(f"Top-{stock_count}: 无交易")

        return result

    except Exception as e:
        logger.error(f"❌ 5分钟回测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_rqalpha_mod():
    """测试RQAlpha Mod集成"""
    logger.info("=== 测试RQAlpha Mod集成 ===")

    try:
        from mod import MinuteDataMod

        # 创建Mod实例
        mod = MinuteDataMod()
        logger.info("✅ MinuteDataMod创建成功")

        # 测试配置schema
        schema = mod.get_mod_config_schema()
        logger.info("✅ 配置schema获取成功")

        # 测试默认配置
        default_config = mod.get_default_mod_config()
        logger.info("✅ 默认配置获取成功")

        logger.info(f"   数据路径: {default_config.get('data_path')}")
        logger.info(f"   缓存大小: {default_config.get('cache_size')}")

        return mod

    except Exception as e:
        logger.error(f"❌ Mod集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    logger.info("🚀 开始5分钟回测系统简化测试")
    logger.info("=" * 60)

    # 测试1: 数据源
    data_source = test_minute_data_source()

    # 测试2: 回测逻辑
    backtest_result = test_minute_backtest()

    # 测试3: RQAlpha Mod
    mod = test_rqalpha_mod()

    logger.info("\n" + "=" * 60)
    logger.info("✅ 简化测试完成")
    logger.info("=" * 60)

    # 总结
    logger.info("\n📋 测试总结:")
    if data_source:
        logger.info("✅ 5分钟数据源 - 正常")
    else:
        logger.info("❌ 5分钟数据源 - 失败")

    if backtest_result:
        logger.info("✅ 5分钟回测逻辑 - 正常")
    else:
        logger.info("❌ 5分钟回测逻辑 - 失败")

    if mod:
        logger.info("✅ RQAlpha Mod集成 - 正常")
    else:
        logger.info("❌ RQAlpha Mod集成 - 失败")

if __name__ == "__main__":
    main()