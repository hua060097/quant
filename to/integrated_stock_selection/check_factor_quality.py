#!/usr/bin/env python3
"""
检查factors.parquet文件中的因子质量

对比生成时和保存后的因子数据，找出导致因子变成NaN或常数的问题
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import logging

# 添加当前目录到路径
sys.path.append('.')

def check_factors_parquet_quality():
    """检查factors.parquet中的因子质量"""
    print("=== 检查factors.parquet文件中的因子质量 ===")

    # 检查文件路径
    factor_file = Path("factor_lab_data/factors.parquet")
    if not factor_file.exists():
        print(f"❌ 文件不存在: {factor_file}")
        return False

    try:
        # 读取文件基本信息
        print(f"\n📁 文件信息:")
        print(f"  文件路径: {factor_file}")
        file_size = factor_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  文件大小: {file_size:.2f} MB")

        # 读取数据（分块读取以节省内存）
        print(f"\n📊 读取因子数据...")
        try:
            # 先读取小样本检查结构
            sample_df = pd.read_parquet(factor_file, nrows=1000)
            print(f"  数据形状(样本): {sample_df.shape}")
            print(f"  列数: {len(sample_df.columns)}")
            print(f"  数据列: {list(sample_df.columns)}")

            # 检查数据类型
            print(f"\n📋 数据类型:")
            for col in sample_df.columns:
                dtype = sample_df[col].dtype
                print(f"  {col}: {dtype}")

            # 检查关键列
            required_cols = ['date', 'stock_code']
            missing_cols = [col for col in required_cols if col not in sample_df.columns]
            if missing_cols:
                print(f"❌ 缺少必要列: {missing_cols}")
                return False
            else:
                print(f"✅ 必要列检查通过")

        except Exception as e:
            print(f"❌ 读取数据失败: {e}")
            return False

        # 检查因子列的质量
        print(f"\n🔍 因子质量检查:")

        # 识别因子列（非date和stock_code的列）
        factor_columns = [col for col in sample_df.columns if col not in ['date', 'stock_code']]
        print(f"  因子列数: {len(factor_columns)}")
        print(f"  因子列: {factor_columns}")

        # 读取更多数据进行质量检查
        try:
            # 读取更多数据（但限制大小以避免内存问题）
            larger_df = pd.read_parquet(factor_file, nrows=100000)
            print(f"  扩展样本数据形状: {larger_df.shape}")

            # 检查每个因子列
            problematic_factors = []
            normal_factors = []

            for col in factor_columns:
                try:
                    series = larger_df[col]

                    # 基本统计
                    total_count = len(series)
                    nan_count = series.isna().sum()
                    valid_count = total_count - nan_count
                    valid_rate = valid_count / total_count if total_count > 0 else 0

                    if valid_count == 0:
                        print(f"❌ {col}: 全部为NaN (总数={total_count}, 有效=0)")
                        problematic_factors.append((col, "全部NaN"))
                        continue

                    # 检查是否为常数列
                    if series.dtype in ['object', 'string', 'category']:
                        # 对于分类型数据，检查唯一值
                        unique_count = series.nunique()
                        if unique_count == 1:
                            print(f"❌ {col}: 常数列 (唯一值={unique_count})")
                            problematic_factors.append((col, "常数列"))
                        else:
                            print(f"✅ {col}: 正常 (唯一值={unique_count})")
                            normal_factors.append(col)
                    else:
                        # 对于数值型数据，检查统计信息
                        if valid_count > 0:
                            valid_data = series.dropna()
                            mean_val = valid_data.mean()
                            std_val = valid_data.std()
                            min_val = valid_data.min()
                            max_val = valid_data.max()
                            unique_count = valid_data.nunique()

                            if np.isnan(mean_val) or np.isnan(std_val):
                                print(f"❌ {col}: 统计量为NaN")
                                problematic_factors.append((col, "统计量NaN"))
                            elif std_val == 0 or unique_count == 1:
                                print(f"❌ {col}: 常数列 (std={std_val}, 唯一值={unique_count})")
                                problematic_factors.append((col, "常数列"))
                            else:
                                print(f"✅ {col}: 正常 (均值={mean_val:.4f}, 标准差={std_val:.4f}, 范围=[{min_val:.4f}, {max_val:.4f}])")
                                normal_factors.append(col)

                except Exception as e:
                    print(f"❌ {col}: 检查失败 - {e}")
                    problematic_factors.append((col, f"检查失败: {e}"))

            # 总结
            print(f"\n📈 质量检查总结:")
            print(f"  总因子数: {len(factor_columns)}")
            print(f"  正常数: {len(normal_factors)}")
            print(f"  问题数: {len(problematic_factors)}")

            if problematic_factors:
                print(f"\n❌ 问题因子详情:")
                problem_types = {}
                for col, reason in problematic_factors:
                    if reason not in problem_types:
                        problem_types[reason] = []
                    problem_types[reason].append(col)

                for reason, cols in problem_types.items():
                    print(f"  {reason}: {len(cols)}个")
                    for col in cols[:5]:  # 只显示前5个
                        print(f"    - {col}")
                    if len(cols) > 5:
                        print(f"    ... 还有{len(cols)-5}个")

            return len(problematic_factors) == 0

        except Exception as e:
            print(f"❌ 质量检查失败: {e}")
            return False

    except Exception as e:
        print(f"❌ 文件检查失败: {e}")
        return False

def check_factor_generation_logs():
    """检查因子生成日志，对比生成时和保存后的数据"""
    print(f"\n=== 检查因子生成日志 ===")

    log_files = [
        "stock_selection_test_mode_walkforward_排序模型_lg57_因子计算.log",
        "stock_selection_test_mode_walkforward_排序模型_lg58_因子计算.log"
    ]

    for log_file in log_files:
        if Path(log_file).exists():
            print(f"\n📄 检查日志文件: {log_file}")
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 查找因子质量检查的日志
                quality_logs = []
                for line in lines:
                    if '质量检查' in line and 'alpha_' in line:
                        quality_logs.append(line.strip())

                print(f"  找到 {len(quality_logs)} 条质量检查日志")

                # 显示前10条
                for i, log in enumerate(quality_logs[:10], 1):
                    print(f"    {i}. {log}")

                if len(quality_logs) > 10:
                    print(f"    ... 还有 {len(quality_logs) - 10} 条")

            except Exception as e:
                print(f"  ❌ 读取日志失败: {e}")
        else:
            print(f"  ⚠️  日志文件不存在: {log_file}")

def check_factor_save_logic():
    """分析因子保存逻辑"""
    print(f"\n=== 分析因子保存逻辑 ===")

    # 检查因子实验室的保存代码
    try:
        from factor_lab.commands import FactorDatasetStore

        store = FactorDatasetStore("factor_lab_data")
        print(f"✅ 成功导入FactorDatasetStore")
        print(f"  存储路径: {store.path}")
        print(f"  文件存在: {store.exists()}")

        if store.exists():
            try:
                # 尝试使用store的方法读取
                data = store.load()
                print(f"✅ 使用store.load()读取成功")
                print(f"  数据形状: {data.shape}")
                print(f"  列: {list(data.columns)}")

                # 检查前几行数据
                print(f"  前5行数据:")
                print(data.head())

                return True
            except Exception as e:
                print(f"❌ store.load()失败: {e}")

                # 尝试直接读取parquet文件
                try:
                    direct_data = pd.read_parquet("factor_lab_data/factors.parquet", nrows=100)
                    print(f"✅ 直接读取parquet成功")
                    print(f"  数据形状: {direct_data.shape}")
                    print(f"  前5行数据:")
                    print(direct_data.head())
                except Exception as e2:
                    print(f"❌ 直接读取parquet也失败: {e2}")
        else:
            print(f"❌ store文件不存在")

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 检查失败: {e}")

    return False

if __name__ == "__main__":
    print("开始检查因子数据质量...")

    # 检查保存的因子文件
    quality_ok = check_factors_parquet_quality()

    # 检查生成日志
    check_factor_generation_logs()

    # 检查保存逻辑
    check_factor_save_logic()

    print(f"\n=== 检查完成 ===")
    if quality_ok:
        print("✅ 因子数据质量正常")
    else:
        print("❌ 发现因子数据质量问题，需要进一步调查保存逻辑")