#!/usr/bin/env python3
"""
简化的增量更新修复测试

直接测试索引重置冲突修复
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import os

# 添加项目路径
import sys
sys.path.append('.')

def test_index_reset_fix():
    """测试索引重置修复"""
    print("=" * 80)
    print("Test Index Reset Fix")
    print("=" * 80)

    try:
        # 创建测试数据
        print("\n1. Creating test data:")

        # 现有数据
        existing_data = pd.DataFrame({
            'date': pd.date_range('2023-06-01', '2023-06-05', freq='D'),
            'stock_code': ['TEST001'] * 5,
            'pct_chg_1d': [0.01, -0.02, 0.03, -0.01, 0.02],
            'factor1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        print(f"   Existing data: {len(existing_data)} records")
        print(f"   Columns: {list(existing_data.columns)}")

        # 新数据
        new_data = pd.DataFrame({
            'date': pd.date_range('2023-06-04', '2023-06-08', freq='D'),
            'stock_code': ['TEST001'] * 5,
            'pct_chg_1d': [0.02, -0.01, 0.01, -0.02, 0.03],
            'factor1': [4.0, 5.0, 6.0, 7.0, 8.0]
        })
        print(f"   New data: {len(new_data)} records")
        print(f"   Columns: {list(new_data.columns)}")

        print("\n2. Testing merge:")

        # 直接调用修复后的合并函数
        from factor_system.factor_lab.store import _smart_merge_dataframes

        merged_data = _smart_merge_dataframes(
            existing_data=existing_data,
            new_data=new_data,
            index_columns=['date', 'stock_code'],
            overlap_mode='merge'
        )

        print("   SUCCESS: Merge completed!")
        print(f"   Merged data: {len(merged_data)} records")
        print(f"   Date range: {merged_data['date'].min()} ~ {merged_data['date'].max()}")

        # 验证数据完整性
        expected_dates = set(pd.date_range('2023-06-01', '2023-06-08', freq='D'))
        actual_dates = set(merged_data['date'].unique())

        if actual_dates == expected_dates:
            print("   SUCCESS: Date continuity verified")
        else:
            missing = expected_dates - actual_dates
            if missing:
                print(f"   WARNING: Missing dates: {sorted(missing)}")

        return True

    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_incremental_save_simulation():
    """测试增量保存模拟"""
    print("\n" + "=" * 80)
    print("Test Incremental Save Simulation")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp()
    factor_path = Path(temp_dir) / "test_factors.parquet"

    try:
        print(f"\n1. Using temp directory: {temp_dir}")

        # 第一次运行
        print("\n2. First run (create initial data):")
        first_data = pd.DataFrame({
            'date': pd.date_range('2023-06-01', '2023-06-05', freq='D'),
            'stock_code': ['TEST001'] * 5,
            'pct_chg_1d': [0.01, -0.02, 0.03, -0.01, 0.02],
            'factor1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        first_data.to_parquet(factor_path, index=False)
        print(f"   Created initial data: {len(first_data)} records")

        # 第二次运行 - 增量更新
        print("\n3. Second run (incremental update):")
        second_data = pd.DataFrame({
            'date': pd.date_range('2023-06-04', '2023-06-08', freq='D'),
            'stock_code': ['TEST001'] * 5,
            'pct_chg_1d': [0.02, -0.01, 0.01, -0.02, 0.03],
            'factor1': [4.0, 5.0, 6.0, 7.0, 8.0]
        })

        print(f"   New data: {len(second_data)} records")

        # 使用修复后的智能合并
        existing = pd.read_parquet(factor_path)

        from factor_system.factor_lab.store import _smart_merge_dataframes

        merged = _smart_merge_dataframes(
            existing_data=existing,
            new_data=second_data,
            index_columns=['date', 'stock_code'],
            overlap_mode='merge'
        )

        # 保存合并结果
        merged.to_parquet(factor_path, index=False)

        print("   SUCCESS: Incremental save completed!")

        # 验证最终结果
        final_data = pd.read_parquet(factor_path)
        print(f"   Final data: {len(final_data)} records")
        print(f"   Date range: {final_data['date'].min()} ~ {final_data['date'].max()}")

        # 检查日期连续性
        expected_dates = set(pd.date_range('2023-06-01', '2023-06-08', freq='D'))
        actual_dates = set(final_data['date'].unique())

        if actual_dates == expected_dates:
            print("   SUCCESS: Final verification passed")
            return True
        else:
            missing = expected_dates - actual_dates
            if missing:
                print(f"   WARNING: Missing dates: {sorted(missing)}")
            return False

    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
            print(f"\n4. Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"   Cleanup warning: {e}")

def main():
    """主测试函数"""
    print("Starting incremental update fix test...")
    print("Using simple test without Unicode characters")

    test_results = []

    # 测试1: 索引重置修复
    print("\n" + "="*80)
    print("Test 1: Index Reset Fix")
    print("="*80)
    result1 = test_index_reset_fix()
    test_results.append(("Index Reset Fix", result1))

    # 测试2: 增量保存修复
    print("\n" + "="*80)
    print("Test 2: Incremental Save Fix")
    print("="*80)
    result2 = test_incremental_save_simulation()
    test_results.append(("Incremental Save Fix", result2))

    # 结果汇总
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall Result: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: All tests passed! Incremental update fix is working!")
        return True
    else:
        print("WARNING: Some tests failed, need further investigation")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"FAILED: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        exit(1)