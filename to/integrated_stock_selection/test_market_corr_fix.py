#!/usr/bin/env python3
"""
测试市场相关性因子修复效果
验证corr_slope_20_120_z等因子不再为常数列
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.factor_lab.pipeline import FactorPipeline
from factor_system.factor_lab.definitions.market_corr import corr_slope_20_120
from factor_system.factor_lab.definitions.low_volatility import beta_market
from factor_system.factor_lab.definitions.statistical_moments import idiosyncratic_volatility


def test_market_correlation_factors():
    """测试市场相关性因子是否还有常数列问题"""
    print("=" * 80)
    print("测试市场相关性因子修复效果")
    print("=" * 80)

    try:
        # 生成模拟测试数据
        print("生成模拟测试数据...")

        # 生成日期范围
        dates = pd.date_range(start='2024-10-01', end='2025-03-10', freq='D')
        # 过滤出工作日（简化处理，实际应该用交易日历）
        dates = [d for d in dates if d.weekday() < 5][:120]  # 取120个交易日

        # 生成股票数据
        stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']

        stock_data_list = []
        for stock in stocks:
            for i, date in enumerate(dates):
                # 模拟股价数据，具有一定的随机性
                base_price = 10 + hash(stock) % 20  # 根据股票代码生成基础价格
                price_change = np.random.normal(0, 0.02)  # 2%的日波动率
                if i == 0:
                    close_price = base_price
                else:
                    # 找到前一日的价格
                    prev_data = [x for x in stock_data_list if x['stock_code'] == stock and x['date'] == dates[i-1]]
                    if prev_data:
                        prev_close = prev_data[0]['close']
                        close_price = prev_close * (1 + price_change)
                    else:
                        close_price = base_price

                pct_chg = price_change * 100  # 百分比变化
                volume = np.random.randint(1000000, 10000000)  # 成交量

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
        for i, date in enumerate(dates):
            market_price_change = np.random.normal(0, 0.015)  # 市场波动率1.5%
            daily_change_pct = market_price_change * 100  # 百分比变化
            volume = np.random.randint(100000000, 1000000000)  # 市场成交量

            market_data_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'daily_change_pct': round(daily_change_pct, 2),
                'volume': volume
            })

        market_data = pd.DataFrame(market_data_list)

        print(f"股票数据形状: {stock_data.shape}")
        print(f"市场数据形状: {market_data.shape}")

        # 创建因子上下文
        from factor_system.factor_lab.base import FactorContext
        context = FactorContext(
            market_data=market_data
        )

        # 测试市场相关性因子
        market_corr_factors = [
            ('corr_slope_20_120', corr_slope_20_120),
            ('beta_market', beta_market),
            ('idiosyncratic_volatility', idiosyncratic_volatility)
        ]

        print(f"\n测试 {len(market_corr_factors)} 个因子的常数列情况:")
        print("-" * 60)

        constant_factors = []
        normal_factors = []

        for factor_name, factor_def in market_corr_factors:
            try:
                print(f"\n计算因子: {factor_name}")

                # 调用因子计算函数
                factor_series = factor_def.calculator(stock_data, context)

                if factor_series.empty:
                    print(f"  错误：因子计算结果为空")
                    continue

                # 基本统计
                non_null_count = factor_series.notna().sum()
                unique_count = factor_series.nunique()
                std_val = factor_series.std()
                min_val = factor_series.min()
                max_val = factor_series.max()
                range_val = max_val - min_val

                # 判断是否为常数列
                is_constant = (
                    unique_count <= 2 or  # 唯一值太少
                    std_val < 1e-6 or     # 标准差太小
                    range_val < 1e-4      # 取值范围太小
                )

                print(f"  非空值数量: {non_null_count:,}")
                print(f"  唯一值数量: {unique_count:,}")
                print(f"  标准差: {std_val:.8f}")
                print(f"  取值范围: [{min_val:.6f} ~ {max_val:.6f}]")
                print(f"  范围宽度: {range_val:.6f}")
                print(f"  是否常数列: {'是 ⚠️' if is_constant else '否 ✅'}")

                if is_constant:
                    constant_factors.append(factor_name)
                else:
                    normal_factors.append(factor_name)

                    # 显示一些样本值
                    sample_values = factor_series.dropna().head(5)
                    print(f"  样本值: {sample_values.values}")

            except Exception as e:
                print(f"  计算失败: {e}")
                constant_factors.append(factor_name)

        # 总结报告
        print("\n" + "=" * 80)
        print("测试结果总结")
        print("=" * 80)

        if constant_factors:
            print(f"\n❌ 仍有 {len(constant_factors)} 个常数列因子:")
            for factor in constant_factors:
                print(f"   - {factor}")
        else:
            print("\n✅ 所有测试的因子都不是常数列！")

        if normal_factors:
            print(f"\n✅ {len(normal_factors)} 个因子正常:")
            for factor in normal_factors:
                print(f"   - {factor}")

        # 特别检查之前有问题的corr_slope_20_120
        if 'corr_slope_20_120' in [f[0] for f in market_corr_factors]:
            print(f"\n🔍 重点检查 corr_slope_20_120:")
            if 'corr_slope_20_120' in normal_factors:
                print(f"   ✅ corr_slope_20_120 已修复，不再是常数列！")
            else:
                print(f"   ⚠️  corr_slope_20_120 仍然是常数列！")

        return len(constant_factors) == 0

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始测试市场相关性因子修复效果...\n")

    success = test_market_correlation_factors()

    if success:
        print("\n[成功] 测试通过！市场相关性因子已成功修复，无常数列问题。")
        sys.exit(0)
    else:
        print("\n[失败] 测试失败！仍有因子存在常数列问题。")
        sys.exit(1)


if __name__ == "__main__":
    main()