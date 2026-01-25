#!/usr/bin/env python3
"""
测试字符串字段修复
验证first_time和last_time字段的数据类型转换问题是否已解决
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_string_field_conversion():
    """测试字符串字段转换修复"""
    print("=" * 80)
    print("测试字符串字段转换修复")
    print("=" * 80)
    
    try:
        from generate_daily_features import optimize_dtypes, _extract_limit_features
        
        # 创建包含问题数据的测试DataFrame
        test_data = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003'],
            'close': [10.5, 20.3, 15.8],
            'volume': [1000000, 2000000, 1500000],
            'first_time': ['09:50:21', np.nan, 123.45],  # 混合类型：字符串、NaN、float
            'last_time': [np.nan, '14:30:00', 456.78],   # 混合类型：NaN、字符串、float
            'first_time_formatted': ['09:50:21', '', '10:15:30'],
            'last_time_formatted': ['14:30:00', '15:00:00', ''],
            'is_limit_up': [1.0, 0.0, 1.0],
            'limit_strength': [0.85, 0.0, 0.92],
            'date_str': ['20240102', '20240102', '20240102']
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        print("原始数据类型:")
        for col, dtype in test_data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        print("\n原始 first_time 值:")
        for i, val in enumerate(test_data['first_time']):
            print(f"  [{i}]: {val} (type: {type(val)})")
        
        print("\n原始 last_time 值:")
        for i, val in enumerate(test_data['last_time']):
            print(f"  [{i}]: {val} (type: {type(val)})")
        
        # 测试数据类型优化
        print("\n🔍 测试数据类型优化...")
        optimized_data = optimize_dtypes(test_data)
        
        print(f"✅ 数据类型优化完成: {optimized_data.shape}")
        print("优化后数据类型:")
        for col, dtype in optimized_data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        print("\n优化后 first_time 值:")
        for i, val in enumerate(optimized_data['first_time']):
            print(f"  [{i}]: '{val}' (type: {type(val)})")
        
        print("\n优化后 last_time 值:")
        for i, val in enumerate(optimized_data['last_time']):
            print(f"  [{i}]: '{val}' (type: {type(val)})")
        
        # 验证字符串字段是否正确转换
        string_fields = ['first_time', 'last_time', 'first_time_formatted', 'last_time_formatted', 'stock_code', 'date_str']
        
        all_string_correct = True
        for field in string_fields:
            if field in optimized_data.columns:
                if optimized_data[field].dtype == 'object':
                    # 检查所有值是否都是字符串
                    all_strings = all(isinstance(val, str) for val in optimized_data[field])
                    if all_strings:
                        print(f"✅ {field} 正确转换为字符串")
                    else:
                        print(f"❌ {field} 包含非字符串值")
                        all_string_correct = False
                else:
                    print(f"❌ {field} 类型错误: {optimized_data[field].dtype}")
                    all_string_correct = False
            else:
                print(f"⚠️  {field} 字段不存在")
        
        # 测试保存为parquet
        print("\n🔍 测试保存为 parquet...")
        try:
            test_file = Path("test_string_fix.parquet")
            optimized_data.to_parquet(test_file, index=False, compression='snappy')
            print("✅ 成功保存为 parquet 文件")
            
            # 读取验证
            loaded_data = pd.read_parquet(test_file)
            print(f"✅ 成功读取 parquet 文件: {loaded_data.shape}")
            
            # 清理测试文件
            test_file.unlink()
            print("✅ 清理测试文件")
            
        except Exception as e:
            print(f"❌ parquet 保存/读取失败: {e}")
            all_string_correct = False
        
        return all_string_correct
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_limit_features_extraction():
    """测试涨跌停特征提取"""
    print("\n" + "=" * 80)
    print("测试涨跌停特征提取")
    print("=" * 80)
    
    try:
        from generate_daily_features import _extract_limit_features
        
        # 创建模拟的涨跌停数据
        limit_row = {
            'fd_amount': 1000000.0,
            'open_times': 5.0,
            'limit_times': 3.0,
            'first_time': 93021.0,  # 数值格式的时间
            'last_time': 143000.0,  # 数值格式的时间
            'limit_amount': 500000.0,
            'limit_strength': 0.85,
            'time_to_limit_minutes': 30.5,
            'first_time_formatted': '09:30:21',
            'last_time_formatted': '14:30:00'
        }
        
        print(f"📊 创建模拟涨跌停数据")
        print("原始数据:")
        for key, val in limit_row.items():
            print(f"  {key}: {val} (type: {type(val)})")
        
        # 测试特征提取
        print("\n🔍 测试特征提取...")
        features = _extract_limit_features(limit_row)
        
        print(f"✅ 特征提取完成，共 {len(features)} 个特征")
        
        # 检查关键字段
        key_fields = ['first_time', 'last_time', 'first_time_formatted', 'last_time_formatted']
        for field in key_fields:
            if field in features:
                val = features[field]
                print(f"  {field}: '{val}' (type: {type(val)})")
                if not isinstance(val, str):
                    print(f"❌ {field} 应该是字符串类型")
                    return False
            else:
                print(f"❌ {field} 字段缺失")
                return False
        
        print("✅ 所有关键字段都是字符串类型")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("测试边界情况")
    print("=" * 80)
    
    try:
        from generate_daily_features import optimize_dtypes
        
        # 创建包含各种边界情况的数据
        edge_cases = pd.DataFrame({
            'stock_code': ['000001', '000002', '000003', '000004'],
            'first_time': [
                '09:30:21',    # 正常字符串
                '',            # 空字符串
                np.nan,        # NaN
                123.45         # float数值
            ],
            'last_time': [
                np.inf,        # 无穷大
                -np.inf,       # 负无穷大
                '14:30:00',    # 正常字符串
                None           # None值
            ],
            'volume': [1000, 2000, 3000, 4000],
            'close': [10.0, 20.0, 30.0, 40.0]
        })
        
        print(f"📊 创建边界情况测试数据: {edge_cases.shape}")
        print("原始数据:")
        for col in ['first_time', 'last_time']:
            print(f"  {col}:")
            for i, val in enumerate(edge_cases[col]):
                print(f"    [{i}]: {val} (type: {type(val)})")
        
        # 测试优化
        print("\n🔍 测试边界情况优化...")
        optimized = optimize_dtypes(edge_cases)
        
        print("优化后数据:")
        for col in ['first_time', 'last_time']:
            print(f"  {col}:")
            for i, val in enumerate(optimized[col]):
                print(f"    [{i}]: '{val}' (type: {type(val)})")
        
        # 验证所有值都是字符串
        for col in ['first_time', 'last_time']:
            all_strings = all(isinstance(val, str) for val in optimized[col])
            if all_strings:
                print(f"✅ {col} 所有值都是字符串")
            else:
                print(f"❌ {col} 包含非字符串值")
                return False
        
        # 测试保存
        try:
            test_file = Path("test_edge_cases.parquet")
            optimized.to_parquet(test_file, index=False)
            print("✅ 边界情况数据成功保存为 parquet")
            test_file.unlink()
        except Exception as e:
            print(f"❌ 边界情况保存失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始字符串字段修复测试...")
    
    tests = [
        ("字符串字段转换", test_string_field_conversion),
        ("涨跌停特征提取", test_limit_features_extraction),
        ("边界情况处理", test_edge_cases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        print("✅ 字符串字段转换错误已修复")
        print("✅ first_time 和 last_time 字段处理正确")
        print("✅ parquet 保存兼容性良好")
        print("✅ 边界情况处理稳定")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
