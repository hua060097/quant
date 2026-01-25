#!/usr/bin/env python3
"""
检查真实的移动均线数据质量
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


def check_real_data_ma_quality():
    """检查真实数据中的移动均线质量"""
    print("=" * 60)
    print("检查真实数据中的移动均线质量")
    print("=" * 60)

    try:
        # 尝试加载真实数据
        from data_loaders import FeatureDataLoader

        loader = FeatureDataLoader()

        # 加载最近的数据
        end_date = "2025-03-10"
        start_date = "2024-10-01"  # 5个月数据

        print(f"加载时间范围: {start_date} ~ {end_date}")

        # 尝试加载包含均线的数据
        try:
            # 加载日线数据
            daily_data = loader.load_stock_data(
                start_date=start_date,
                end_date=end_date,
                columns=['stock_code', 'date', 'close', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120']
            )

            print(f"成功加载包含均线的日线数据: {daily_data.shape}")

            # 检查各条均线的质量
            ma_columns = ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']
            available_ma_cols = [col for col in ma_columns if col in daily_data.columns]

            print(f"\n均线质量检查:")
            for ma_col in available_ma_cols:
                total_count = len(daily_data)
                non_null_count = daily_data[ma_col].notna().sum()
                null_count = total_count - non_null_count

                print(f"  {ma_col}:")
                print(f"    总数据量: {total_count:,}")
                print(f"    非空值: {non_null_count:,} ({non_null_count/total_count:.1%})")
                print(f"    空值: {null_count:,} ({null_count/total_count:.1%})")

                if non_null_count > 0:
                    unique_count = daily_data[ma_col].nunique()
                    std_val = daily_data[ma_col].std()
                    print(f"    唯一值数量: {unique_count:,}")
                    print(f"    标准差: {std_val:.6f}")

                    # 检查是否为常数列
                    is_constant = unique_count <= 2 or std_val < 1e-6
                    print(f"    是否为常数列: {'是' if is_constant else '否'}")

                    if not is_constant:
                        min_val = daily_data[ma_col].min()
                        max_val = daily_data[ma_col].max()
                        print(f"    取值范围: [{min_val:.6f} ~ {max_val:.6f}]")

                print()

            # 测试ma_convergence_score
            if len(available_ma_cols) >= 2:
                print("测试ma_convergence_score因子...")

                # 取样本数据进行测试
                sample_data = daily_data.head(1000)  # 取前1000条记录

                context = FactorContext()
                result = ma_convergence_score.calculator(sample_data, context)

                print(f"  结果形状: {result.shape}")
                print(f"  非空值: {result.notna().sum()}")
                print(f"  唯一值: {result.nunique()}")
                print(f"  标准差: {result.std():.8f}")
                print(f"  取值范围: [{result.min():.6f} ~ {result.max():.6f}]")

                is_constant = (result.nunique() <= 2 or result.std() < 1e-6)
                print(f"  是否为常数列: {'是 - 有问题' if is_constant else '否 - 正常'}")

                if is_constant:
                    print(f"\n[问题分析] ma_convergence_score是常数列:")
                    if result.nunique() == 1:
                        only_value = result.iloc[0]
                        print(f"  所有值都是: {only_value}")
                        if abs(only_value) < 1e-6:
                            print(f"  -> 原因：计算结果全为0，可能是均线数据不足")
                        elif abs(only_value - 1.0) < 1e-6:
                            print(f"  -> 原因：计算结果全为1，可能是均线完全收敛或数据问题")

                    # 分析均线数据质量
                    print(f"\n  均线数据分析:")
                    for ma_col in available_ma_cols:
                        non_null_rate = sample_data[ma_col].notna().mean()
                        print(f"    {ma_col}: {non_null_rate:.1%} 非空率")
                        if non_null_rate < 0.5:
                            print(f"      -> {ma_col} 数据严重不足")

                return not is_constant
            else:
                print(f"可用的均线数量不足 ({len(available_ma_cols)} < 2)")
                return False

        except Exception as e:
            print(f"加载包含均线的数据失败: {e}")

            # 尝试只加载基础数据
            print("尝试加载基础数据（close价格）...")
            basic_data = loader.load_stock_data(
                start_date=start_date,
                end_date=end_date,
                columns=['stock_code', 'date', 'close']
            )

            print(f"基础数据: {basic_data.shape}")

            # 测试能否现场计算均线
            if len(basic_data) > 0:
                print("测试现场计算ma_convergence_score...")

                context = FactorContext()
                result = ma_convergence_score.calculator(basic_data, context)

                print(f"  结果形状: {result.shape}")
                print(f"  唯一值: {result.nunique()}")
                print(f"  标准差: {result.std():.8f}")

                is_constant = (result.nunique() <= 2 or result.std() < 1e-6)
                print(f"  是否为常数列: {'是' if is_constant else '否'}")

                return not is_constant

    except ImportError as e:
        print(f"无法导入数据加载器: {e}")
        return False
    except Exception as e:
        print(f"检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_ma_convergence_formula():
    """分析ma_convergence_score的计算公式"""
    print("=" * 60)
    print("分析ma_convergence_score计算公式")
    print("=" * 60)

    print("公式: score = 1.0 - (ma_max - ma_min) / ma_mean")
    print()

    # 创建一些测试场景
    scenarios = [
        {
            'name': '均线完全相同',
            'ma_values': [10.0, 10.0, 10.0, 10.0, 10.0],
        },
        {
            'name': '均线轻微发散',
            'ma_values': [9.8, 9.9, 10.0, 10.1, 10.2],
        },
        {
            'name': '均线中等发散',
            'ma_values': [8.0, 9.0, 10.0, 11.0, 12.0],
        },
        {
            'name': '均线严重发散',
            'ma_values': [5.0, 7.5, 10.0, 12.5, 15.0],
        },
        {
            'name': '均值为0的情况',
            'ma_values': [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ]

    for scenario in scenarios:
        ma_values = scenario['ma_values']
        ma_max = max(ma_values)
        ma_min = min(ma_values)
        ma_mean = np.mean(ma_values)

        if ma_mean > 0:
            score = 1.0 - (ma_max - ma_min) / ma_mean
            score = max(0.0, min(1.0, score))  # clip到[0,1]
        else:
            score = 0.0

        print(f"{scenario['name']}:")
        print(f"  均线值: {ma_values}")
        print(f"  MA_max: {ma_max}, MA_min: {ma_min}, MA_mean: {ma_mean}")
        print(f"  计算结果: {score:.6f}")
        print()


def main():
    """主函数"""
    print("开始分析ma_convergence_score因子问题...\n")

    # 分析计算公式
    analyze_ma_convergence_formula()

    # 检查真实数据
    success = check_real_data_ma_quality()

    print("\n" + "=" * 60)
    print("分析结果总结:")
    print("=" * 60)

    if success:
        print("[成功] ma_convergence_score因子工作正常")
    else:
        print("[问题] ma_convergence_score因子存在问题")
        print("\n可能原因:")
        print("1. 历史数据不足以计算长期均线（如MA120需要120天历史）")
        print("2. 均线数据中存在大量空值")
        print("3. 在特定市场环境下，多条均线趋向于相同值")
        print("4. 数据质量问题导致计算异常")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)