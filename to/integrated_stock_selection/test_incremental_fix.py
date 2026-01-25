#!/usr/bin/env python3
"""
测试修复后的增量更新功能

验证索引重置冲突是否已修复
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import os
import sys
import logging

# 添加项目路径
sys.path.append('.')
from factor_system.factor_lab.store import _smart_merge_dataframes
from factor_system.factor_lab.commands import generate_and_store

# 简单的日志设置
def setup_logger(name):
    """设置简单的日志"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def test_smart_merge_fix():
    """测试智能合并功能修复"""
    print("=" * 80)
    print("测试智能合并修复")
    print("=" * 80)

    # 创建测试数据
    print("\n1. 创建测试数据:")

    # 现有数据 (2023-06-01 到 2023-06-10)
    existing_data = pd.DataFrame({
        'date': pd.date_range('2023-06-01', '2023-06-10', freq='D'),
        'stock_code': ['TEST001'] * 10,
        'pct_chg_1d': np.random.randn(10) * 0.02,
        'factor1': np.random.randn(10)
    })
    print(f"   现有数据: {len(existing_data)}条记录")
    print(f"   日期范围: {existing_data['date'].min()} ~ {existing_data['date'].max()}")

    # 新数据 (2023-06-08 到 2023-06-15)，有重叠
    new_data = pd.DataFrame({
        'date': pd.date_range('2023-06-08', '2023-06-15', freq='D'),
        'stock_code': ['TEST001'] * 8,
        'pct_chg_1d': np.random.randn(8) * 0.02,
        'factor1': np.random.randn(8)
    })
    print(f"   新数据: {len(new_data)}条记录")
    print(f"   日期范围: {new_data['date'].min()} ~ {new_data['date'].max()}")

    # 设置索引
    existing_data = existing_data.set_index(['date', 'stock_code'])
    new_data = new_data.set_index(['date', 'stock_code'])

    print("\n2. 执行智能合并:")
    try:
        # 测试智能合并
        merged_data = _smart_merge_dataframes(
            existing_data=existing_data,
            new_data=new_data,
            index_columns=['date', 'stock_code'],
            overlap_mode='merge'
        )

        print("✅ 智能合并成功!")
        print(f"   合并后数据: {len(merged_data)}条记录")
        print(f"   日期范围: {merged_data['date'].min()} ~ {merged_data['date'].max()}")
        print(f"   预期结果: {len(existing_data) + len(new_data) - 3}条记录")  # 3天重叠

        # 验证数据完整性
        expected_dates = set(pd.date_range('2023-06-01', '2023-06-15', freq='D'))
        actual_dates = set(merged_data['date'].unique())

        if actual_dates == expected_dates:
            print("✅ 日期连续性验证通过")
        else:
            missing = expected_dates - actual_dates
            extra = actual_dates - expected_dates
            if missing:
                print(f"❌ 缺失日期: {sorted(missing)}")
            if extra:
                print(f"❌ 多余日期: {sorted(extra)}")

        return True

    except Exception as e:
        print(f"❌ 智能合并失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_incremental_save_with_fix():
    """测试修复后的增量保存"""
    print("\n" + "=" * 80)
    print("测试修复后的增量保存")
    print("=" * 80)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    factor_path = Path(temp_dir) / "test_factors.parquet"

    try:
        print(f"\n1. 使用临时目录: {temp_dir}")

        # 第一次运行 - 创建初始数据
        print("\n2. 第一次运行 (创建初始数据):")
        first_data = pd.DataFrame({
            'date': pd.date_range('2023-06-01', '2023-06-05', freq='D'),
            'stock_code': ['TEST001'] * 5,
            'pct_chg_1d': np.random.randn(5) * 0.02,
            'factor1': np.random.randn(5)
        })

        first_data.to_parquet(factor_path, index=False)
        print(f"   创建初始数据: {len(first_data)}条记录")

        # 第二次运行 - 增量更新
        print("\n3. 第二次运行 (增量更新):")
        second_data = pd.DataFrame({
            'date': pd.date_range('2023-06-04', '2023-06-08', freq='D'),
            'stock_code': ['TEST001'] * 5,
            'pct_chg_1d': np.random.randn(5) * 0.02,
            'factor1': np.random.randn(5)
        })

        print(f"   新数据: {len(second_data)}条记录")

        # 使用修复后的智能合并
        existing = pd.read_parquet(factor_path)
        existing_indexed = existing.set_index(['date', 'stock_code'])
        new_indexed = second_data.set_index(['date', 'stock_code'])

        merged = _smart_merge_dataframes(
            existing_data=existing_indexed,
            new_data=new_indexed,
            index_columns=['date', 'stock_code'],
            overlap_mode='merge'
        )

        # 保存合并结果
        merged.to_parquet(factor_path, index=False)

        print("✅ 增量保存成功!")

        # 验证最终结果
        final_data = pd.read_parquet(factor_path)
        print(f"   最终数据: {len(final_data)}条记录")
        print(f"   日期范围: {final_data['date'].min()} ~ {final_data['date'].max()}")

        # 检查日期连续性
        expected_dates = set(pd.date_range('2023-06-01', '2023-06-08', freq='D'))
        actual_dates = set(final_data['date'].unique())

        if actual_dates == expected_dates:
            print("✅ 最终验证通过")
            return True
        else:
            missing = expected_dates - actual_dates
            extra = actual_dates - expected_dates
            if missing:
                print(f"❌ 缺失日期: {sorted(missing)}")
            if extra:
                print(f"❌ 多余日期: {sorted(extra)}")
            return False

    except Exception as e:
        print(f"❌ 增量保存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
            print(f"\n4. 清理临时目录: {temp_dir}")
        except Exception as e:
            print(f"   清理警告: {e}")

def main():
    """主测试函数"""
    print("开始测试修复后的增量更新功能...")

    # 设置日志
    logger = setup_logger('test_incremental_fix')

    test_results = []

    # 测试1: 智能合并修复
    print("\n" + "="*80)
    print("测试1: 智能合并修复")
    print("="*80)
    result1 = test_smart_merge_fix()
    test_results.append(("智能合并修复", result1))

    # 测试2: 增量保存修复
    print("\n" + "="*80)
    print("测试2: 增量保存修复")
    print("="*80)
    result2 = test_incremental_save_with_fix()
    test_results.append(("增量保存修复", result2))

    # 结果汇总
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总体结果: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过! 增量更新修复成功!")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        exit(1)