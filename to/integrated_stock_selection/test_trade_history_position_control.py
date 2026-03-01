"""
测试基于历史交易结果的动态仓位控制功能
Test Trade History Based Position Control
"""

import sys
import pandas as pd
from datetime import date
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from to.integrated_stock_selection.rqalpha_minute_datasource.strategy_backtest1 import BacktestStrategy, StrategyConfig, StrategyType as ST


def create_strategy_with_trades():
    """创建一个带有预设交易历史的策略实例"""
    config = StrategyConfig(strategy_type=ST.TOP_1, stock_count=1, stop_loss_threshold=-0.05)

    # 创建空的预测数据
    pred_data = pd.DataFrame({
        'date': [date(2026, 2, 10)],
        'stock_code': ['000001.SZ'],
        'predicted_return': [0.01]
    })

    strategy = BacktestStrategy(
        config,
        pred_data,
        r'C:\Users\86151\.trae-cn\pythons\path\to\local_data\minute\202602\minute_data.parquet'
    )

    return strategy


def test_scenario_1_last_trade_stop_loss():
    """测试场景1: 上次交易触发止损"""
    print("=" * 80)
    print("场景1: 上次交易触发5%止损")
    print("=" * 80)

    strategy = create_strategy_with_trades()

    # 模拟历史交易：最后一次触发止损
    strategy.trades = [
        {
            'stock_code': '000001.SZ',
            'actual_return': 0.03,  # +3% 盈利
            'exit_reason': '到期'
        },
        {
            'stock_code': '000002.SZ',
            'actual_return': -0.06,  # -6% 止损！
            'exit_reason': '风控: 5%止损'
        }
    ]

    limit = strategy._get_position_limit_by_trade_history()

    print(f"\n预期: limit = 0.01 (防御模式)")
    print(f"实际: limit = {limit}")
    print(f"结果: {'✓ PASS' if limit == 0.01 else '✗ FAIL'}")

    return limit == 0.01


def test_scenario_2_consecutive_losses():
    """测试场景2: 连续3次亏损"""
    print("\n" + "=" * 80)
    print("场景2: 连续3次交易亏损")
    print("=" * 80)

    strategy = create_strategy_with_trades()

    # 模拟历史交易：连续3次亏损
    strategy.trades = [
        {
            'stock_code': '000001.SZ',
            'actual_return': -0.02,  # -2%
            'exit_reason': '到期'
        },
        {
            'stock_code': '000002.SZ',
            'actual_return': -0.01,  # -1%
            'exit_reason': '到期'
        },
        {
            'stock_code': '000003.SZ',
            'actual_return': -0.03,  # -3%
            'exit_reason': '到期'
        }
    ]

    limit = strategy._get_position_limit_by_trade_history()

    print(f"\n预期: limit = 0.01 (防御模式)")
    print(f"实际: limit = {limit}")
    print(f"结果: {'✓ PASS' if limit == 0.01 else '✗ FAIL'}")

    return limit == 0.01


def test_scenario_3_consecutive_profits():
    """测试场景3: 连续2次盈利"""
    print("\n" + "=" * 80)
    print("场景3: 连续2次交易盈利")
    print("=" * 80)

    strategy = create_strategy_with_trades()

    # 模拟历史交易：连续2次盈利
    strategy.trades = [
        {
            'stock_code': '000001.SZ',
            'actual_return': 0.03,  # +3%
            'exit_reason': '到期'
        },
        {
            'stock_code': '000002.SZ',
            'actual_return': 0.02,  # +2%
            'exit_reason': '到期'
        }
    ]

    limit = strategy._get_position_limit_by_trade_history()

    print(f"\n预期: limit = 0.0 (进攻模式)")
    print(f"实际: limit = {limit}")
    print(f"结果: {'✓ PASS' if limit == 0.0 else '✗ FAIL'}")

    return limit == 0.0


def test_scenario_4_no_history():
    """测试场景4: 无历史交易"""
    print("\n" + "=" * 80)
    print("场景4: 无历史交易（首次运行）")
    print("=" * 80)

    strategy = create_strategy_with_trades()

    # 空交易历史
    strategy.trades = []

    limit = strategy._get_position_limit_by_trade_history()

    print(f"\n预期: limit = 0.0 (默认模式)")
    print(f"实际: limit = {limit}")
    print(f"结果: {'✓ PASS' if limit == 0.0 else '✗ FAIL'}")

    return limit == 0.0


def test_scenario_5_mixed_results():
    """测试场景5: 混合结果（不触发任何规则）"""
    print("\n" + "=" * 80)
    print("场景5: 混合结果（不触发任何规则）")
    print("=" * 80)

    strategy = create_strategy_with_trades()

    # 模拟混合结果
    strategy.trades = [
        {
            'stock_code': '000001.SZ',
            'actual_return': 0.02,  # +2%
            'exit_reason': '到期'
        },
        {
            'stock_code': '000002.SZ',
            'actual_return': -0.01,  # -1%
            'exit_reason': '到期'
        },
        {
            'stock_code': '000003.SZ',
            'actual_return': 0.03,  # +3%
            'exit_reason': '到期'
        }
    ]

    limit = strategy._get_position_limit_by_trade_history()

    print(f"\n预期: limit = 0.0 (默认模式)")
    print(f"实际: limit = {limit}")
    print(f"结果: {'✓ PASS' if limit == 0.0 else '✗ FAIL'}")

    return limit == 0.0


def test_position_calculation():
    """测试仓位计算"""
    print("\n" + "=" * 80)
    print("测试: 仓位比例计算")
    print("=" * 80)

    strategy = create_strategy_with_trades()

    # 测试不同场景
    test_cases = [
        # (trade_history, lead, expected_pct, description)
        ([], 0.005, 1.0, "无历史，lead=0.005"),
        ([{'actual_return': -0.06, 'exit_reason': '止损', 'stock_code': 'A'}], 0.005, 0.0, "上次止损，lead=0.005"),
        ([{'actual_return': -0.06, 'exit_reason': '止损', 'stock_code': 'A'}], 0.015, 1.0, "上次止损，lead=0.015"),
        ([{'actual_return': 0.03, 'exit_reason': '到期', 'stock_code': 'A'},
          {'actual_return': 0.02, 'exit_reason': '到期', 'stock_code': 'B'}], 0.005, 1.0, "连续盈利，lead=0.005"),
    ]

    all_passed = True
    for i, (trades, lead, expected_pct, desc) in enumerate(test_cases, 1):
        strategy.trades = trades
        actual_pct = strategy._position_pct_by_lead(lead, date(2026, 2, 10))
        passed = actual_pct == expected_pct
        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"\n测试 {i}: {desc}")
        print(f"  预期: {expected_pct:.1f}, 实际: {actual_pct:.1f} - {status}")

        if not passed:
            all_passed = False

    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "#" * 80)
    print("# 基于历史交易结果的动态仓位控制功能测试")
    print("#" * 80)

    tests = [
        ("场景1: 上次交易止损", test_scenario_1_last_trade_stop_loss),
        ("场景2: 连续3次亏损", test_scenario_2_consecutive_losses),
        ("场景3: 连续2次盈利", test_scenario_3_consecutive_profits),
        ("场景4: 无历史交易", test_scenario_4_no_history),
        ("场景5: 混合结果", test_scenario_5_mixed_results),
        ("仓位计算", test_position_calculation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ {test_name}: 异常 - {e}")

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
