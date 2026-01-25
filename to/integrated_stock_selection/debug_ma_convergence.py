#!/usr/bin/env python3
"""
调试ma_convergence_score因子的问题
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.factor_lab.definitions.moving_average import ma_convergence_score
from factor_system.factor_lab.base import FactorContext


def generate_test_data_with_ma():
    """生成包含预计算均线的测试数据"""
    print("生成包含均线的测试数据...")

    # 生成较少的历史数据来模拟真实情况
    dates = pd.date_range(start='2025-01-01', end='2025-03-10', freq='D')
    dates = [d for d in dates if d.weekday() < 5][:50]  # 只取50个交易日

    # 生成股票数据
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']

    stock_data_list = []
    for stock in stocks:
        for i, date in enumerate(dates):
            base_price = 10 + hash(stock) % 20
            price_change = np.random.normal(0, 0.02)
            if i == 0:
                close_price = base_price
            else:
                prev_data = [x for x in stock_data_list if x['stock_code'] == stock and x['date'] == dates[i-1].strftime('%Y-%m-%d')]
                if prev_data:
                    prev_close = prev_data[0]['close']
                    close_price = prev_close * (1 + price_change)
                else:
                    close_price = base_price

            stock_data_list.append({
                'stock_code': stock,
                'date': date.strftime('%Y-%m-%d'),
                'close': round(close_price, 2),
                'pct_chg': round(price_change * 100, 2),
                'volume': np.random.randint(1000000, 10000000)
            })

    stock_data = pd.DataFrame(stock_data_list)
    stock_data = stock_data.sort_values(['stock_code', 'date']).reset_index(drop=True)

    # 计算均线并添加到数据中
    print("计算移动均线...")
    for stock in stocks:
        stock_mask = stock_data['stock_code'] == stock
        stock_subset = stock_data[stock_mask].copy()

        # 计算各期均线
        stock_subset['ma5'] = stock_subset['close'].rolling(window=5, min_periods=1).mean()
        stock_subset['ma10'] = stock_subset['close'].rolling(window=10, min_periods=1).mean()
        stock_subset['ma20'] = stock_subset['close'].rolling(window=20, min_periods=1).mean()
        stock_subset['ma60'] = stock_subset['close'].rolling(window=60, min_periods=1).mean()
        stock_subset['ma120'] = stock_subset['close'].rolling(window=120, min_periods=1).mean()

        # 更新原数据
        for col in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']:
            stock_data.loc[stock_mask, col] = stock_subset[col]

    print(f"股票数据形状: {stock_data.shape}")
    print(f"均线列: {[col for col in stock_data.columns if col.startswith('ma')]}")

    # 检查均线数据
    for ma_col in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']:
        non_null_count = stock_data[ma_col].notna().sum()
        print(f"{ma_col}: {non_null_count} 非空值")

    return stock_data


def debug_ma_convergence():
    """调试ma_convergence_score因子"""
    print("=" * 60)
    print("调试 ma_convergence_score 因子")
    print("=" * 60)

    # 生成测试数据
    stock_data = generate_test_data_with_ma()

    print(f"\n测试数据样本:")
    print(stock_data.head(10))

    print(f"\n计算ma_convergence_score因子...")

    try:
        context = FactorContext()
        result = ma_convergence_score.calculator(stock_data, context)

        print(f"\n因子计算结果:")
        print(f"  结果形状: {result.shape}")
        print(f"  非空值数量: {result.notna().sum()}")
        print(f"  唯一值数量: {result.nunique()}")
        print(f"  标准差: {result.std():.8f}")
        print(f"  最小值: {result.min():.6f}")
        print(f"  最大值: {result.max():.6f}")
        print(f"  均值: {result.mean():.6f}")
        print(f"  前10个值: {result.head(10).values}")

        # 检查是否为常数列
        is_constant = (result.nunique() <= 2 or result.std() < 1e-6 or (result.max() - result.min()) < 1e-4)
        print(f"  是否为常数列: {'是 - 有问题' if is_constant else '否 - 正常'}")

        if is_constant:
            print(f"\n[分析] 因子是常数列，可能原因:")

            # 分析具体原因
            if result.nunique() == 1:
                only_value = result.iloc[0]
                print(f"  1. 所有值都是相同的: {only_value}")
                if abs(only_value) < 1e-6:
                    print(f"     -> 可能是因为均线数据不足或计算结果全为0")
                elif abs(only_value - 1.0) < 1e-6:
                    print(f"     -> 可能是因为均线完全发散")

            # 检查均线数据
            print(f"\n  2. 检查均线数据质量:")
            for ma_col in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']:
                if ma_col in stock_data.columns:
                    valid_count = stock_data[ma_col].notna().sum()
                    total_count = len(stock_data)
                    print(f"     {ma_col}: {valid_count}/{total_count} 有效数据 ({valid_count/total_count:.1%})")
                    if valid_count < total_count * 0.8:  # 如果有效数据少于80%
                        print(f"        -> {ma_col} 数据不足，可能影响计算")

        return not is_constant

    except Exception as e:
        print(f"  计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ma_convergence_without_precomputed_ma():
    """测试没有预计算均线的情况"""
    print("\n" + "=" * 60)
    print("测试没有预计算均线的情况")
    print("=" * 60)

    # 生成只包含收盘价的数据
    dates = pd.date_range(start='2025-01-01', end='2025-03-10', freq='D')
    dates = [d for d in dates if d.weekday() < 5][:30]  # 只取30个交易日

    stock_data_list = []
    for stock in ['000001.SZ', '000002.SZ']:
        for i, date in enumerate(dates):
            base_price = 10 + hash(stock) % 20
            price_change = np.random.normal(0, 0.02)
            if i == 0:
                close_price = base_price
            else:
                prev_data = [x for x in stock_data_list if x['stock_code'] == stock and x['date'] == dates[i-1].strftime('%Y-%m-%d')]
                if prev_data:
                    prev_close = prev_data[0]['close']
                    close_price = prev_close * (1 + price_change)
                else:
                    close_price = base_price

            stock_data_list.append({
                'stock_code': stock,
                'date': date.strftime('%Y-%m-%d'),
                'close': round(close_price, 2)
            })

    stock_data = pd.DataFrame(stock_data_list)
    stock_data = stock_data.sort_values(['stock_code', 'date']).reset_index(drop=True)

    print(f"数据形状: {stock_data.shape}")
    print(f"列名: {list(stock_data.columns)}")

    try:
        context = FactorContext()
        result = ma_convergence_score.calculator(stock_data, context)

        print(f"\n因子计算结果:")
        print(f"  结果形状: {result.shape}")
        print(f"  唯一值数量: {result.nunique()}")
        print(f"  前5个值: {result.head(5).values}")

        return result.nunique() > 1

    except Exception as e:
        print(f"  计算失败: {e}")
        return False


def main():
    """主函数"""
    print("开始调试 ma_convergence_score 因子...\n")

    success1 = debug_ma_convergence()
    success2 = test_ma_convergence_without_precomputed_ma()

    print("\n" + "=" * 60)
    print("调试结果总结:")
    print("=" * 60)
    print(f"有预计算均线: {'正常' if success1 else '有问题'}")
    print(f"无预计算均线: {'正常' if success2 else '有问题'}")

    if success1 and success2:
        print("[成功] ma_convergence_score因子工作正常")
        return True
    else:
        print("[失败] ma_convergence_score因子存在问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)