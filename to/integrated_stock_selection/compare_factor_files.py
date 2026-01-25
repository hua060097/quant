#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较 factors.parquet 和 factors1.parquet 两个文件
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compare_factor_files():
    """详细比较两个因子文件的差异"""
    print("=" * 80)
    print("因子文件对比分析")
    print("=" * 80)

    factors1_path = Path("factor_lab_data/factors1.parquet")
    factors_path = Path("factor_lab_data/factors.parquet")

    if not factors1_path.exists():
        print(f"错误: {factors1_path} 不存在")
        return

    if not factors_path.exists():
        print(f"错误: {factors_path} 不存在")
        return

    # 获取文件大小
    size1 = factors1_path.stat().st_size / (1024 * 1024)  # MB
    size2 = factors_path.stat().st_size / (1024 * 1024)  # MB

    print(f"文件大小对比:")
    print(f"  factors1.parquet (原始): {size1:.2f} MB")
    print(f"  factors.parquet (新生): {size2:.2f} MB")
    print(f"  差异: {size2 - size1:.2f} MB ({((size2 - size1) / size1 * 100):+.1f}%)")
    print()

    # 加载数据
    print("正在加载数据...")
    try:
        df1 = pd.read_parquet(factors1_path)
        df2 = pd.read_parquet(factors_path)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    print(f"数据形状对比:")
    print(f"  factors1.parquet: {df1.shape}")
    print(f"  factors.parquet: {df2.shape}")
    print()

    # 获取索引列和因子列
    index_cols = ['date', 'stock_code']
    factor_cols1 = [col for col in df1.columns if col not in index_cols]
    factor_cols2 = [col for col in df2.columns if col not in index_cols]

    print(f"因子数量对比:")
    print(f"  factors1.parquet: {len(factor_cols1)} 个因子")
    print(f"  factors.parquet: {len(factor_cols2)} 个因子")
    print()

    # 找出新增的因子
    new_factors = set(factor_cols2) - set(factor_cols1)
    missing_factors = set(factor_cols1) - set(factor_cols2)

    print(f"因子变化:")
    if new_factors:
        print(f"  新增因子: {new_factors}")
    if missing_factors:
        print(f"  丢失因子: {missing_factors}")
    if not new_factors and not missing_factors:
        print("  因子列表相同")
    print()

    # 检查 drawdown_recovery_speed
    target_factor = "drawdown_recovery_speed"
    print(f"=== {target_factor} 因子检查 ===")
    if target_factor in factor_cols2:
        print(f"[SUCCESS] {target_factor} 已成功保存")

        # 统计信息
        factor_data = df2[target_factor]
        nan_count = factor_data.isna().sum()
        valid_count = factor_data.notna().sum()
        print(f"  总记录数: {len(factor_data):,}")
        print(f"  有效数据: {valid_count:,}")
        print(f"  缺失数据: {nan_count:,}")
        print(f"  缺失率: {nan_count / len(factor_data) * 100:.2f}%")
        print(f"  数据类型: {factor_data.dtype}")

        if valid_count > 0:
            valid_values = factor_data.dropna()
            print(f"  均值: {valid_values.mean():.6f}")
            print(f"  标准差: {valid_values.std():.6f}")
            print(f"  最小值: {valid_values.min():.6f}")
            print(f"  最大值: {valid_values.max():.6f}")
            print(f"  中位数: {valid_values.median():.6f}")
    else:
        print(f"[ERROR] {target_factor} 未找到")
    print()

    # 数据类型分析
    print("=== 数据类型对比 ===")
    common_factors = set(factor_cols1) & set(factor_cols2)
    dtype_changes = []

    for factor in sorted(common_factors):
        dtype1 = df1[factor].dtype
        dtype2 = df2[factor].dtype
        if dtype1 != dtype2:
            dtype_changes.append((factor, dtype1, dtype2))

    if dtype_changes:
        print("数据类型变化的因子:")
        for factor, dtype1, dtype2 in dtype_changes[:10]:  # 只显示前10个
            print(f"  {factor}: {dtype1} → {dtype2}")
        if len(dtype_changes) > 10:
            print(f"  ... 还有 {len(dtype_changes) - 10} 个因子的数据类型发生变化")
    else:
        print("所有因子数据类型保持一致")

    # 统计 float64 和 float32 的数量
    float64_count_1 = sum(1 for col in factor_cols1 if df1[col].dtype == 'float64')
    float32_count_1 = sum(1 for col in factor_cols1 if df1[col].dtype == 'float32')
    float64_count_2 = sum(1 for col in factor_cols2 if df2[col].dtype == 'float64')
    float32_count_2 = sum(1 for col in factor_cols2 if df2[col].dtype == 'float32')

    print(f"\n数据类型统计:")
    print(f"  factors1.parquet:")
    print(f"    float64: {float64_count_1} 个因子")
    print(f"    float32: {float32_count_1} 个因子")
    print(f"  factors.parquet:")
    print(f"    float64: {float64_count_2} 个因子")
    print(f"    float32: {float32_count_2} 个因子")

    if float32_count_2 > float32_count_1:
        print(f"  [OPTIMIZATION] 数据类型优化: {float32_count_2 - float32_count_1} 个因子从 float64 转为 float32")
        # 计算节省的存储空间
        saved_space = (float32_count_2 - float32_count_1) * len(df2) * (8 - 4) / (1024 * 1024)
        print(f"  [SPACE_SAVED] 预估节省空间: {saved_space:.2f} MB")
    print()

    # 内存使用分析
    print("=== 内存使用对比 ===")
    memory1 = df1.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    memory2 = df2.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

    print(f"内存占用:")
    print(f"  factors1.parquet: {memory1:.2f} MB")
    print(f"  factors.parquet: {memory2:.2f} MB")
    print(f"  内存节省: {memory1 - memory2:.2f} MB")

    # 检查数据完整性
    print("\n=== 数据完整性检查 ===")
    if df1.shape[0] == df2.shape[0] and df1.shape[1] + 1 == df2.shape[1]:
        print("[SUCCESS] 数据完整性正常: 新增了1列，行数保持一致")
    else:
        print("[WARNING] 数据形状发生变化")
        print(f"  行数变化: {df1.shape[0]} → {df2.shape[0]}")
        print(f"  列数变化: {df1.shape[1]} → {df2.shape[1]}")

    print("\n=== 总结 ===")
    if new_factors and target_factor in new_factors:
        print(f"[SUCCESS] {target_factor} 因子已成功添加")
    else:
        print(f"[ERROR] {target_factor} 因子添加失败")

    if float32_count_2 > float32_count_1:
        print("[SUCCESS] 数据类型优化成功，文件体积减小是正常的")

    if not missing_factors:
        print("[SUCCESS] 没有丢失任何原有因子")
    else:
        print(f"[WARNING] 警告: 丢失了 {len(missing_factors)} 个因子")

if __name__ == "__main__":
    compare_factor_files()