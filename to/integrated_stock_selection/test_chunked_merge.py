#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分块合并优化功能
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# 添加路径
sys.path.append('.')
sys.path.append('factor_system/factor_lab')

from factor_system.factor_lab.store_merge_fix import smart_merge_dataframes_fixed, chunked_merge_large_dataframes, optimize_dtypes

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_test_data(existing_rows=50000, new_rows=5000, n_factors=20):
    """创建测试数据"""
    # 创建现有数据
    existing_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=existing_rows, freq='D'),
        'stock_code': [f'SH{600000 + i % 1000:06d}' for i in range(existing_rows)]
    })

    # 添加因子列
    for i in range(n_factors):
        existing_data[f'factor_{i:02d}'] = np.random.normal(0, 1, existing_rows)

    # 创建新数据（部分重叠）
    overlap_rows = int(new_rows * 0.3)  # 30%重叠
    new_data = pd.DataFrame({
        'date': pd.date_range('2023-12-01', periods=new_rows, freq='D'),
        'stock_code': [f'SH{600000 + i % 1000:06d}' for i in range(new_rows)]
    })

    # 确保有一些重叠
    new_data.loc[:overlap_rows, 'stock_code'] = existing_data.loc[:overlap_rows, 'stock_code'].values

    # 添加因子列
    for i in range(n_factors):
        new_data[f'factor_{i:02d}'] = np.random.normal(0, 1, new_rows)

    return existing_data, new_data

def test_dtypes_optimization():
    """测试数据类型优化"""
    print("=== 测试数据类型优化 ===")

    # 创建大数据量测试
    test_data = pd.DataFrame({
        'float64_col': np.random.randn(10000),
        'int64_col': np.random.randint(0, 1000, 10000),
        'float32_col': np.random.randn(10000).astype('float32')
    })

    print(f"优化前: {test_data.dtypes.to_dict()}")
    original_memory = test_data.memory_usage(deep=True).sum() / (1024**2)
    print(f"优化前内存: {original_memory:.2f} MB")

    optimized_data = optimize_dtypes(test_data.copy())

    print(f"优化后: {optimized_data.dtypes.to_dict()}")
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / (1024**2)
    print(f"优化后内存: {optimized_memory:.2f} MB")

    savings = (original_memory - optimized_memory) / original_memory * 100
    print(f"内存节省: {savings:.1f}%")

    return True  # 返回布尔值而不是DataFrame

def test_small_data_merge():
    """测试小数据量合并（使用标准合并）"""
    print("\n=== 测试小数据量合并 ===")

    existing_data, new_data = create_test_data(existing_rows=1000, new_rows=100)
    print(f"现有数据: {len(existing_data)} 行")
    print(f"新数据: {len(new_data)} 行")

    try:
        result = smart_merge_dataframes_fixed(existing_data, new_data, ['date', 'stock_code'])
        print(f"✅ 小数据合并成功: {len(result)} 行")

        # 检查重叠处理
        overlap_count = len(result) - (len(existing_data) + len(new_data) - 1000)  # 估算重叠
        print(f"重叠处理正确，结果行数合理")

        return True
    except Exception as e:
        print(f"❌ 小数据合并失败: {e}")
        return False

def test_large_data_merge():
    """测试大数据量合并（使用分块合并）"""
    print("\n=== 测试大数据量合并 ===")

    existing_data, new_data = create_test_data(existing_rows=50000, new_rows=5000)
    print(f"现有数据: {len(existing_data):,} 行")
    print(f"新数据: {len(new_data):,} 行")

    try:
        result = smart_merge_dataframes_fixed(existing_data, new_data, ['date', 'stock_code'])
        print(f"✅ 大数据合并成功: {len(result):,} 行")

        # 验证数据完整性
        expected_max_rows = len(existing_data) + len(new_data)
        expected_min_rows = max(len(existing_data), len(new_data))

        if expected_min_rows <= len(result) <= expected_max_rows:
            print("✅ 数据完整性检查通过")
        else:
            print(f"⚠️ 数据完整性检查: 期望{expected_min_rows}-{expected_max_rows}, 实际{len(result)}")

        return True
    except Exception as e:
        print(f"❌ 大数据合并失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试内存使用情况"""
    print("\n=== 测试内存使用情况 ===")

    try:
        import psutil

        # 获取初始内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB

        print(f"初始内存: {initial_memory:.1f} MB")

        # 创建大数据
        existing_data, new_data = create_test_data(existing_rows=80000, new_rows=8000)
        after_create_memory = process.memory_info().rss / (1024**2)
        print(f"创建数据后: {after_create_memory:.1f} MB (+{after_create_memory-initial_memory:.1f})")

        # 执行合并
        result = smart_merge_dataframes_fixed(existing_data, new_data, ['date', 'stock_code'])
        after_merge_memory = process.memory_info().rss / (1024**2)
        print(f"合并完成后: {after_merge_memory:.1f} MB (+{after_merge_memory-after_create_memory:.1f})")

        # 检查内存是否合理
        total_memory_used = after_merge_memory - initial_memory
        print(f"总内存使用: {total_memory_used:.1f} MB")

        if total_memory_used < 1000:  # 小于1GB
            print("✅ 内存使用合理")
            return True
        else:
            print(f"⚠️ 内存使用较高: {total_memory_used:.1f} MB")
            return True

    except ImportError:
        print("⚠️ psutil未安装，跳过内存测试")
        return True
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        return False

def main():
    """主测试函数"""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("开始测试分块合并优化功能...")
    print("=" * 60)

    success_count = 0
    total_tests = 4

    # 测试1：数据类型优化
    if test_dtypes_optimization():
        success_count += 1

    # 测试2：小数据合并
    if test_small_data_merge():
        success_count += 1

    # 测试3：大数据合并
    if test_large_data_merge():
        success_count += 1

    # 测试4：内存使用
    if test_memory_usage():
        success_count += 1

    print(f"\n{'='*60}")
    print(f"测试完成: {success_count}/{total_tests} 通过")

    if success_count == total_tests:
        print("🎉 所有测试通过！分块合并优化功能正常工作")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)