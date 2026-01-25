#!/usr/bin/env python3
"""
简化版本的市场相关性因子测试
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.factor_lab.definitions.market_corr import corr_slope_20_120
from factor_system.factor_lab.definitions.low_volatility import beta_market
from factor_system.factor_lab.base import FactorContext


def generate_test_data():
    """生成测试数据"""
    print("生成测试数据...")

    # 生成日期范围
    dates = pd.date_range(start='2024-10-01', end='2025-03-10', freq='D')
    dates = [d for d in dates if d.weekday() < 5][:120]  # 取120个交易日

    # 生成股票数据
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']

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

            pct_chg = price_change * 100
            volume = np.random.randint(1000000, 10000000)

            stock_data_list.append({
                'stock_code': stock,
                'date': date.strftime('%Y-%m-%d'),
                'close': round(close_price, 2),
                'pct_chg': round(pct_chg, 2),
                'volume': volume
            })

    stock_data = pd.DataFrame(stock_data_list)

    # 生成市场数据
    market_data_list = []
    for date in dates:
        market_price_change = np.random.normal(0, 0.015)
        daily_change_pct = market_price_change * 100
        volume = np.random.randint(100000000, 1000000000)

        market_data_list.append({
            'date': date.strftime('%Y-%m-%d'),
            'daily_change_pct': round(daily_change_pct, 2),
            'volume': volume
        })

    market_data = pd.DataFrame(market_data_list)

    return stock_data, market_data


def test_factor_simple(factor_name, factor_def, stock_data, market_data):
    """简单测试单个因子"""
    print(f"\n测试因子: {factor_name}")

    try:
        context = FactorContext(market_data=market_data)
        result = factor_def.calculator(stock_data, context)

        if result.empty:
            print(f"  结果: 空数据")
            return False

        non_null_count = result.notna().sum()
        unique_count = result.nunique()
        std_val = result.std()
        min_val = result.min()
        max_val = result.max()

        is_constant = (unique_count <= 2 or std_val < 1e-6 or (max_val - min_val) < 1e-4)

        print(f"  非空值数量: {non_null_count:,}")
        print(f"  唯一值数量: {unique_count:,}")
        print(f"  标准差: {std_val:.8f}")
        print(f"  取值范围: [{min_val:.6f} ~ {max_val:.6f}]")
        print(f"  结果: {'常数列 - 有问题' if is_constant else '正常 - 修复成功'}")

        return not is_constant

    except Exception as e:
        print(f"  错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("市场相关性因子修复效果测试")
    print("=" * 60)

    # 生成测试数据
    stock_data, market_data = generate_test_data()
    print(f"股票数据: {stock_data.shape}")
    print(f"市场数据: {market_data.shape}")

    # 测试因子
    factors_to_test = [
        ('corr_slope_20_120', corr_slope_20_120),
        ('beta_market', beta_market),
    ]

    results = {}
    for factor_name, factor_def in factors_to_test:
        results[factor_name] = test_factor_simple(factor_name, factor_def, stock_data, market_data)

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)

    success_count = 0
    for factor_name, is_success in results.items():
        status = "✓ 修复成功" if is_success else "✗ 仍有问题"
        print(f"  {factor_name}: {status}")
        if is_success:
            success_count += 1

    print(f"\n总体结果: {success_count}/{len(results)} 个因子正常")

    if success_count == len(results):
        print("[成功] 所有测试的因子都已修复！")
        return True
    else:
        print("[失败] 仍有因子存在问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)