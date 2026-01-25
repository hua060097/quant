#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的合并逻辑测试
"""

import pandas as pd
import numpy as np
import sys

sys.path.append('.')

def test_merge():
    """测试合并逻辑"""
    print("=== 测试合并逻辑修复 ===")

    try:
        
        # 创建测试数据
        index_columns = ['date', 'stock_code']

        # 基础数据
        base_data = pd.DataFrame({
            'date': ['20240101', '20240102', '20240103'],
            'stock_code': ['SH600001', 'SH600002', 'SH600003'],
            'factor_a': [1.0, 2.0, 3.0]
        })

        # 新数据（相同索引，不同因子）
        new_data = pd.DataFrame({
            'date': ['20240101', '20240102', '20240103'],
            'stock_code': ['SH600001', 'SH600002', 'SH600003'],
            'factor_b': [4.0, 5.0, 6.0]
        })

        print("SUCCESS 测试数据创建完成")
        print(f"  基础数据: {base_data.shape}")
        print(f"  新数据: {new_data.shape}")

        # 合并测试
        from factor_system.factor_lab.store_merge_fix import smart_merge_dataframes_fixed
        result = smart_merge_dataframes_fixed(base_data, new_data, index_columns, 'merge')

        print(f"SUCCESS 合并完成: {result.shape}")
        print(f"  合并后列: {list(result.columns)}")

        # 检查结果
        if (len(result) == 3 and
            'factor_a' in result.columns and
            'factor_b' in result.columns and
            result['factor_a'].isna().sum() == 0 and
            result['factor_b'].isna().sum() == 0):
            print("SUCCESS 合并逻辑修复成功！")
            print("  factor_a 值:", result['factor_a'].tolist())
            print("  factor_b 值:", result['factor_b'].tolist())
            return True
        else:
            print("ERROR 合并逻辑仍有问题")
            print(f"  factor_a NaN数: {result['factor_a'].isna().sum()}")
            print(f"  factor_b NaN数: {result['factor_b'].isna().sum()}")
            return False

    except Exception as e:
        print(f"ERROR 测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_merge()
    if success:
        print("\nSUCCESS 修复验证通过，可以安全运行因子生成")
    else:
        print("\nERROR 修复验证失败")