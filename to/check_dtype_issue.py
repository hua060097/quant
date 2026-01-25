#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据类型兼容性问题
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_dtype_compatibility():
    """检查数据类型兼容性问题"""
    
    print("🔍 检查数据类型兼容性问题")
    print("=" * 60)
    
    # 检查特征文件的数据类型
    features_dir = Path("to/local_data/features/daily")
    sample_file = sorted(features_dir.glob("*.parquet"))[-1]
    
    print(f"📂 检查文件: {sample_file}")
    
    df = pd.read_parquet(sample_file)
    
    # 检查市值相关字段的数据类型
    mv_fields = ['total_mv', 'circ_mv', 'turnover_rate']
    
    print(f"\n📊 当前数据类型:")
    for field in mv_fields:
        if field in df.columns:
            dtype = df[field].dtype
            print(f"   {field}: {dtype}")
            
            # 检查数值范围
            series = df[field]
            if series.dtype in ['uint32', 'float16', 'float32']:
                valid_data = series[series > 0]
                if len(valid_data) > 0:
                    print(f"      数值范围: {valid_data.min():.2e} ~ {valid_data.max():.2e}")
                    
                    # 检查是否超出数据类型范围
                    if dtype == 'uint32':
                        max_uint32 = np.iinfo(np.uint32).max  # 4,294,967,295
                        overflow_count = (valid_data > max_uint32).sum()
                        print(f"      uint32最大值: {max_uint32:,}")
                        print(f"      超出uint32范围的数量: {overflow_count}")
                        
                        if overflow_count > 0:
                            print(f"      ❌ 数据超出uint32范围！")
                        else:
                            print(f"      ✅ 数据在uint32范围内")
                    
                    elif dtype == 'float16':
                        max_float16 = np.finfo(np.float16).max  # 65504
                        overflow_count = (valid_data > max_float16).sum()
                        print(f"      float16最大值: {max_float16}")
                        print(f"      超出float16范围的数量: {overflow_count}")
                        
                        if overflow_count > 0:
                            print(f"      ❌ 数据超出float16范围！")
                        else:
                            print(f"      ✅ 数据在float16范围内")

def test_dtype_conversion():
    """测试数据类型转换方案"""
    
    print(f"\n🧪 测试数据类型转换方案")
    print("=" * 60)
    
    # 模拟问题数据
    test_values = [
        243156156416.0,  # 2431亿，超出uint32范围
        243151814656.0,  # 2431亿，超出uint32范围
        0.005710601806640625  # 换手率
    ]
    
    print(f"📊 测试数据:")
    for i, val in enumerate(test_values):
        print(f"   值{i+1}: {val}")
    
    # 测试不同的转换方案
    print(f"\n🔧 转换方案测试:")
    
    # 方案1：直接转换（当前方案，有警告）
    print(f"\n方案1：直接转换 np.float32/np.float16")
    for i, val in enumerate(test_values):
        if i < 2:  # 市值字段
            try:
                converted = np.float32(val)
                print(f"   {val} -> np.float32: {converted}")
            except Exception as e:
                print(f"   {val} -> np.float32: 错误 {e}")
        else:  # 换手率字段
            try:
                converted = np.float16(val)
                print(f"   {val} -> np.float16: {converted}")
            except Exception as e:
                print(f"   {val} -> np.float16: 错误 {e}")
    
    # 方案2：使用astype转换
    print(f"\n方案2：使用 pd.Series.astype 转换")
    for i, val in enumerate(test_values):
        if i < 2:  # 市值字段
            try:
                series = pd.Series([val])
                converted = series.astype('float32').iloc[0]
                print(f"   {val} -> astype('float32'): {converted}")
            except Exception as e:
                print(f"   {val} -> astype('float32'): 错误 {e}")
        else:  # 换手率字段
            try:
                series = pd.Series([val])
                converted = series.astype('float16').iloc[0]
                print(f"   {val} -> astype('float16'): {converted}")
            except Exception as e:
                print(f"   {val} -> astype('float16'): 错误 {e}")
    
    # 方案3：检查范围后选择合适的数据类型
    print(f"\n方案3：智能类型选择")
    
    uint32_max = np.iinfo(np.uint32).max
    float16_max = np.finfo(np.float16).max
    
    for i, val in enumerate(test_values):
        if i < 2:  # 市值字段
            if val <= uint32_max:
                target_type = 'uint32'
                converted = np.uint32(val)
            else:
                target_type = 'float64'  # 使用更大的类型
                converted = np.float64(val)
            print(f"   {val} -> {target_type}: {converted}")
        else:  # 换手率字段
            if val <= float16_max:
                target_type = 'float16'
                converted = np.float16(val)
            else:
                target_type = 'float32'
                converted = np.float32(val)
            print(f"   {val} -> {target_type}: {converted}")

def recommend_solution():
    """推荐解决方案"""
    
    print(f"\n💡 推荐解决方案")
    print("=" * 60)
    
    print("🎯 问题根源:")
    print("   - 市值数据（2431亿）超出了uint32的最大值（42.9亿）")
    print("   - pandas警告将来版本会严格检查类型兼容性")
    
    print(f"\n🔧 推荐方案：")
    print("   1. 市值字段使用 float64 类型（支持更大数值）")
    print("   2. 换手率字段使用 float32 类型（精度足够）")
    print("   3. 在赋值前先转换DataFrame列的数据类型")
    
    print(f"\n📝 修复代码示例:")
    print("   ```python")
    print("   # 在更新前先转换列类型")
    print("   if field in ['total_mv', 'circ_mv']:")
    print("       feature_df[field] = feature_df[field].astype('float64')")
    print("   elif field == 'turnover_rate':")
    print("       feature_df[field] = feature_df[field].astype('float32')")
    print("   ")
    print("   # 然后正常赋值")
    print("   feature_df.at[idx, field] = new_val")
    print("   ```")

if __name__ == "__main__":
    check_dtype_compatibility()
    test_dtype_conversion()
    recommend_solution()
