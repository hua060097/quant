"""
测试市场状态仓位控制功能
Test Market Status Based Position Control
"""

import sys
import json
import pandas as pd
from datetime import datetime, date
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from to.integrated_stock_selection.rqalpha_minute_datasource.strategy_backtest1 import BacktestStrategy, StrategyConfig, StrategyType

def test_market_status_loading():
    """测试市场状态数据加载"""
    print("=" * 80)
    print("测试1: 市场状态数据加载")
    print("=" * 80)

    # 创建一个简单的策略配置
    config = StrategyConfig(
        strategy_type=StrategyType.TOP_1,
        stock_count=1
    )

    # 创建空的预测数据
    prediction_data = pd.DataFrame({
        'date': [date(2026, 2, 10)],
        'stock_code': ['000001.SZ'],
        'predicted_return': [0.01]
    })

    # 创建策略实例
    strategy = BacktestStrategy(
        config=config,
        prediction_data=prediction_data,
        minute_data_path=r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\minute\202602\minute_data.parquet"
    )

    # 调用加载方法
    strategy._load_market_status_data()

    # 验证数据已加载
    assert hasattr(strategy, 'market_status_data'), "市场状态数据未加载"
    assert len(strategy.market_status_data) > 0, "市场状态数据为空"

    print(f"数据加载成功!")
    print(f"  交易日数量: {len(strategy.market_status_data)}")
    print(f"  日期范围: {min(strategy.market_status_data.keys())} ~ {max(strategy.market_status_data.keys())}")

    return True


def test_market_status_judgment():
    """测试市场状态判断逻辑"""
    print("\n" + "=" * 80)
    print("测试2: 市场状态判断逻辑")
    print("=" * 80)

    # 创建策略实例
    config = StrategyConfig(strategy_type=StrategyType.TOP_1, stock_count=1)
    prediction_data = pd.DataFrame({
        'date': [date(2026, 2, 10)],
        'stock_code': ['000001.SZ'],
        'predicted_return': [0.01]
    })
    strategy = BacktestStrategy(
        config=config,
        prediction_data=prediction_data,
        minute_data_path=r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\minute\202602\minute_data.parquet"
    )

    # 加载市场状态数据
    strategy._load_market_status_data()

    # 测试几个日期
    test_dates = [
        date(2026, 2, 10),
        date(2026, 2, 5),
        date(2026, 2, 3),
    ]

    for test_date in test_dates:
        limit = strategy._get_position_limit_by_market_status(test_date)
        print(f"\n日期: {test_date}")
        print(f"  limit: {limit}")

        # 获取当日市场数据
        date_str = test_date.strftime('%Y%m%d')
        if date_str in strategy.market_status_data:
            try:
                tech_data = strategy.market_status_data[date_str]["individual_status"]["399006.SZ"]["technical_data"]
                close = tech_data["close"]
                ma20 = tech_data["ma20"]
                status = "弱势" if close < ma20 else "强势"
                print(f"  创业板指: close={close:.2f}, ma20={ma20:.2f}")
                print(f"  市场状态: {status}")
            except Exception as e:
                print(f"  获取市场数据失败: {e}")

    return True


def test_position_pct_calculation():
    """测试仓位比例计算"""
    print("\n" + "=" * 80)
    print("测试3: 仓位比例计算")
    print("=" * 80)

    # 创建策略实例
    config = StrategyConfig(strategy_type=StrategyType.TOP_1, stock_count=1)
    prediction_data = pd.DataFrame({
        'date': [date(2026, 2, 10)],
        'stock_code': ['000001.SZ'],
        'predicted_return': [0.01]
    })
    strategy = BacktestStrategy(
        config=config,
        prediction_data=prediction_data,
        minute_data_path=r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\minute\202602\minute_data.parquet"
    )

    # 加载市场状态数据
    strategy._load_market_status_data()

    # 测试场景
    test_cases = [
        # (lead, trade_date, expected_result_description)
        (0.015, date(2026, 2, 10), "高lead值"),
        (0.005, date(2026, 2, 10), "低lead值"),
        (0.015, date(2026, 2, 5), "不同日期"),
        (-0.005, date(2026, 2, 10), "负lead值"),
    ]

    for lead, trade_date, desc in test_cases:
        position_pct = strategy._position_pct_by_lead(lead, trade_date)
        print(f"\n{desc}")
        print(f"  lead: {lead:.4f}")
        print(f"  trade_date: {trade_date}")
        print(f"  position_pct: {position_pct:.2f} ({'满仓' if position_pct == 1.0 else '空仓'})")

        # 显示当日市场状态
        date_str = trade_date.strftime('%Y%m%d')
        if date_str in strategy.market_status_data:
            try:
                tech_data = strategy.market_status_data[date_str]["individual_status"]["399006.SZ"]["technical_data"]
                close = tech_data["close"]
                ma20 = tech_data["ma20"]
                limit = 0.01 if close < ma20 else 0.0
                print(f"  当日limit: {limit} (close={close:.2f}, ma20={ma20:.2f})")
            except:
                pass

    return True


def main():
    """运行所有测试"""
    print("\n" + "#" * 80)
    print("# 市场状态仓位控制功能测试")
    print("#" * 80)

    tests = [
        ("市场状态数据加载", test_market_status_loading),
        ("市场状态判断逻辑", test_market_status_judgment),
        ("仓位比例计算", test_position_pct_calculation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, True, None))
            print(f"\n✓ {test_name}: 通过")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name}: 失败 - {e}")

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for test_name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name}")
        if error:
            print(f"  错误: {error}")

    all_passed = all(passed for _, passed, _ in results)
    print(f"\n最终结果: {'全部通过 ✓' if all_passed else '部分失败 ✗'}")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
