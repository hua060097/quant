#!/usr/bin/env python3
"""
简单测试字符串字段修复
"""

import pandas as pd
import numpy as np
from generate_daily_features import optimize_dtypes

def test_fix():
    print('测试字符串字段修复...')

    # 创建包含问题数据的测试
    test_data = pd.DataFrame({
        'stock_code': ['000001', '000002'],
        'first_time': ['09:30:21', 123.45],  # 字符串和float混合
        'last_time': [np.nan, '14:30:00'],   # NaN和字符串混合
        'first_time_formatted': ['09:30:21', ''],
        'volume': [1000, 2000],
        'close': [10.0, 20.0]
    })

    print('原始数据类型:')
    for col in ['first_time', 'last_time']:
        print(f'  {col}: {test_data[col].dtype}')
        for i, val in enumerate(test_data[col]):
            print(f'    [{i}]: {val} (type: {type(val)})')

    # 优化数据类型
    optimized = optimize_dtypes(test_data)

    print('\n优化后数据类型:')
    for col in ['first_time', 'last_time']:
        print(f'  {col}: {optimized[col].dtype}')
        for i, val in enumerate(optimized[col]):
            print(f'    [{i}]: "{val}" (type: {type(val)})')

    # 测试保存
    try:
        optimized.to_parquet('test_fix.parquet', index=False)
        print('\n✅ 成功保存为 parquet 文件')
        
        # 读取验证
        loaded = pd.read_parquet('test_fix.parquet')
        print(f'✅ 成功读取，数据形状: {loaded.shape}')
        
        # 清理
        import os
        os.remove('test_fix.parquet')
        print('✅ 清理完成')
        
        return True
        
    except Exception as e:
        print(f'❌ 保存/读取失败: {e}')
        return False

if __name__ == "__main__":
    success = test_fix()
    if success:
        print('\n🎉 修复验证成功！')
    else:
        print('\n💥 修复验证失败！')
