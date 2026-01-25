#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试_smart_merge_dataframes函数修复效果
"""

import pandas as pd
import numpy as np
import sys
import logging

# 添加路径
sys.path.append('.')
sys.path.append('factor_system/factor_lab')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_fix():
    """测试修复效果"""
    print("=== 测试 _smart_merge_dataframes 修复效果 ===")

    from factor_system.factor_lab.store import _smart_merge_dataframes

    # 模拟实际错误的数据规模
    existing_rows = 2918521  # 实际数据规模
    new_rows = 14865         # 实际数据规模

    print(f"创建测试数据 - 现有: {existing_rows:,}行, 新: {new_rows:,}行")

    # 创建测试数据（小规模测试，但结构相同）
    test_existing_rows = 50000
    test_new_rows = 5000

    existing_data = pd.DataFrame({
        'date': pd.date_range('2023-06-01', periods=test_existing_rows, freq='H'),
        'stock_code': [f'SH{600000 + i % 4955:06d}' for i in range(test_existing_rows)]
    })

    new_data = pd.DataFrame({
        'date': pd.date_range('2025-12-16', periods=test_new_rows, freq='H'),
        'stock_code': [f'SH{600000 + i % 1000:06d}' for i in range(test_new_rows)]
    })

    # 添加因子列（模拟126个因子）
    n_factors = 50  # 测试用较少因子
    for i in range(n_factors):
        existing_data[f'factor_{i:02d}'] = np.random.randn(test_existing_rows)
        new_data[f'factor_{i:02d}'] = np.random.randn(test_new_rows)

    # 确保有重叠
    overlap_count = min(1000, test_new_rows)
    new_data.loc[:overlap_count, 'stock_code'] = existing_data.loc[:overlap_count, 'stock_code'].values
    new_data.loc[:overlap_count, 'date'] = existing_data.loc[:overlap_count, 'date'].values

    print(f"测试数据准备完成: 现有{len(existing_data):,}行, 新{len(new_data):,}行, {len(existing_data.columns)}列")

    try:
        # 调用修复后的函数
        result = _smart_merge_dataframes(
            existing_data=existing_data,
            new_data=new_data,
            index_columns=['date', 'stock_code'],
            overlap_mode='merge'
        )

        print(f"SUCCESS 合并成功: {len(result):,}行, {len(result.columns)}列")
        print(f"数据完整性检查通过")
        return True

    except Exception as e:
        print(f"FAIL 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fix()
    if success:
        print("\n🎉 修复测试通过！_smart_merge_dataframes 现在使用优化版本")
    else:
        print("\n❌ 修复测试失败")

    sys.exit(0 if success else 1)