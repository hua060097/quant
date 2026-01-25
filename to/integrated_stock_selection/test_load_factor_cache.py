#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试_load_factor_cache_data函数的输出
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导入main模块中的函数
from main import _load_factor_cache_data

print("测试_load_factor_cache_data函数...\n")

# 调用函数
factor_data = _load_factor_cache_data(include_label_generation_features=True)

if factor_data is None:
    print("❌ 函数返回None!")
    sys.exit(1)

print(f"✅ 成功加载数据: {len(factor_data)} 行, {len(factor_data.columns)} 列\n")

# 检查关键字段
required_fields = ['date', 'stock_code', 'pct_chg', 'is_can_buy', 'open', 'high', 'low', 'close', 'atr14']

print("=" * 60)
print("关键字段检查")
print("=" * 60)

for field in required_fields:
    if field in factor_data.columns:
        missing_count = factor_data[field].isna().sum()
        missing_pct = missing_count / len(factor_data) * 100
        zero_count = (factor_data[field] == 0).sum() if field not in ['date', 'stock_code'] else 0
        zero_pct = zero_count / len(factor_data) * 100 if field not in ['date', 'stock_code'] else 0
        
        status = "✅" if missing_count == 0 else "⚠️"
        print(f"{status} {field}:")
        print(f"  缺失: {missing_count:,} ({missing_pct:.2f}%)")
        if field not in ['date', 'stock_code']:
            print(f"  为0: {zero_count:,} ({zero_pct:.2f}%)")
    else:
        print(f"❌ {field}: 字段不存在")

print()

# 检查20250106
target_date = pd.to_datetime('2025-01-06')
print("=" * 60)
print(f"目标日期 {target_date.date()} 的数据")
print("=" * 60)

date_data = factor_data[factor_data['date'] == target_date]
print(f"\n该日期的记录数: {len(date_data)}")

if len(date_data) > 0:
    print(f"\n字段值统计:")
    for field in required_fields:
        if field in date_data.columns and field not in ['date', 'stock_code']:
            missing_count = date_data[field].isna().sum()
            zero_count = (date_data[field] == 0).sum()
            non_zero_count = len(date_data) - missing_count - zero_count
            
            print(f"  {field}:")
            print(f"    缺失: {missing_count}/{len(date_data)}")
            print(f"    为0: {zero_count}/{len(date_data)}")
            print(f"    非0: {non_zero_count}/{len(date_data)}")
            
            if non_zero_count > 0:
                non_zero_values = date_data[date_data[field] != 0][field].dropna()
                print(f"    非0值范围: [{non_zero_values.min():.4f}, {non_zero_values.max():.4f}]")
    
    # 显示示例数据
    print(f"\n示例数据 (前5行):")
    sample_cols = ['stock_code'] + [f for f in required_fields if f in date_data.columns and f not in ['date', 'stock_code']]
    print(date_data[sample_cols].head().to_string())
else:
    print("❌ 该日期没有数据!")

print("\n测试完成!")

