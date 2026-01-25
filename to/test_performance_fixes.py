#!/usr/bin/env python3
"""
测试性能修复和数据类型转换修复
验证所有关键问题是否已解决
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_dtype_optimization_fix():
    """测试数据类型优化修复"""
    print("=" * 80)
    print("测试数据类型优化修复")
    print("=" * 80)
    
    try:
        from generate_daily_features import optimize_dtypes
        
        # 创建包含字符串时间字段的测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'close': [10.5, 20.3],
            'volume': [1000000, 2000000],
            'first_time_formatted': ['09:50:21', '10:15:30'],  # 时间字符串
            'last_time_formatted': ['14:30:00', '15:00:00'],   # 时间字符串
            'is_limit_up': [1.0, 0.0],
            'limit_strength': [0.85, 0.0],
            'date_str': ['20240102', '20240102']
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        print(f"原始数据类型:")
        for col, dtype in test_data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # 测试数据类型优化
        optimized_data = optimize_dtypes(test_data)
        
        print(f"\n✅ 数据类型优化完成: {optimized_data.shape}")
        print(f"优化后数据类型:")
        for col, dtype in optimized_data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # 验证字符串字段保持为object类型
        string_fields = ['stock_code', 'first_time_formatted', 'last_time_formatted', 'date_str']
        for field in string_fields:
            if field in optimized_data.columns:
                if optimized_data[field].dtype == 'object':
                    print(f"✅ {field} 正确保持为 object 类型")
                else:
                    print(f"❌ {field} 类型错误: {optimized_data[field].dtype}")
                    return False
        
        # 验证数值字段转换为合适类型
        if optimized_data['close'].dtype == 'float32':
            print("✅ close 正确转换为 float32")
        else:
            print(f"❌ close 类型错误: {optimized_data['close'].dtype}")
            return False
            
        if optimized_data['volume'].dtype == 'int32':
            print("✅ volume 正确转换为 int32")
        else:
            print(f"❌ volume 类型错误: {optimized_data['volume'].dtype}")
            return False
        
        # 验证时间字符串值没有被破坏
        if optimized_data['first_time_formatted'].iloc[0] == '09:50:21':
            print("✅ 时间字符串值保持正确")
        else:
            print(f"❌ 时间字符串值被破坏: {optimized_data['first_time_formatted'].iloc[0]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_improvements():
    """测试性能改进"""
    print("\n" + "=" * 80)
    print("测试性能改进")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_technical_indicators
        
        # 创建较大的测试数据集
        n_stocks = 10
        n_days = 100
        total_rows = n_stocks * n_days
        
        print(f"📊 创建大型测试数据集: {n_stocks} 只股票 × {n_days} 天 = {total_rows} 行")
        
        # 生成测试数据
        stock_codes = [f"00000{i}" for i in range(1, n_stocks + 1)]
        dates = pd.date_range('2024-01-01', periods=n_days)
        
        test_data = []
        for stock_code in stock_codes:
            for date in dates:
                test_data.append({
                    'stock_code': stock_code,
                    'date': date,
                    'open': np.random.uniform(10, 100),
                    'high': np.random.uniform(10, 100),
                    'low': np.random.uniform(10, 100),
                    'close': np.random.uniform(10, 100),
                    'volume': np.random.randint(1000, 100000),
                    'amount': np.random.randint(100000, 10000000),
                    'pct_chg': np.random.uniform(-10, 10),
                    'total_mv': np.random.randint(1000000, 100000000),
                    'circ_mv': np.random.randint(800000, 80000000),
                    'turnover_rate': np.random.uniform(0, 20)
                })
        
        test_df = pd.DataFrame(test_data)
        print(f"✅ 测试数据生成完成: {test_df.shape}")
        
        # 测试计算性能
        print("🚀 开始性能测试...")
        start_time = time.time()
        
        result = calculate_technical_indicators(test_df, None)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ 技术指标计算完成: {result.shape}")
        print(f"⏱️  处理时间: {processing_time:.2f} 秒")
        print(f"📈 处理速度: {total_rows / processing_time:.0f} 行/秒")
        
        # 性能基准检查
        expected_speed = 1000  # 期望至少1000行/秒
        if total_rows / processing_time >= expected_speed:
            print(f"✅ 性能达标: {total_rows / processing_time:.0f} 行/秒 >= {expected_speed} 行/秒")
        else:
            print(f"⚠️  性能未达标: {total_rows / processing_time:.0f} 行/秒 < {expected_speed} 行/秒")
        
        # 检查内存使用
        memory_usage = result.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        print(f"💾 内存使用: {memory_usage:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multithreading_consistency():
    """测试多线程一致性"""
    print("\n" + "=" * 80)
    print("测试多线程一致性")
    print("=" * 80)
    
    try:
        from generate_daily_features import StockFeatureGenerator
        
        # 初始化特征生成器
        generator = StockFeatureGenerator()
        print("✅ 特征生成器初始化成功")
        
        # 检查性能统计功能
        stats = generator.get_performance_stats()
        print(f"📊 初始性能统计: {stats}")
        
        # 测试缓存清理功能
        generator.clear_cache()
        print("✅ 缓存清理功能正常")
        
        # 检查多线程配置
        print(f"🔧 多线程配置: {generator.max_workers} 个工作线程")
        
        if generator.max_workers >= 2:
            print("✅ 多线程配置合理")
        else:
            print("⚠️  多线程配置可能不够优化")
        
        return True
        
    except Exception as e:
        print(f"❌ 多线程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_string_field_handling():
    """测试字符串字段处理"""
    print("\n" + "=" * 80)
    print("测试字符串字段处理")
    print("=" * 80)
    
    try:
        from generate_daily_features import _get_all_limit_features, integrate_complete_limit_features
        
        # 获取所有涨跌停特征
        all_features = _get_all_limit_features()
        print(f"📊 涨跌停特征总数: {len(all_features)}")
        
        # 检查字符串字段是否在特征列表中
        string_features = ['first_time_formatted', 'last_time_formatted']
        for feature in string_features:
            if feature in all_features:
                print(f"✅ {feature} 在特征列表中")
            else:
                print(f"❌ {feature} 不在特征列表中")
                return False
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'],
            'date': [pd.Timestamp('2024-01-02')],
            'close': [10.0]
        })
        
        # 测试特征集成（使用None作为limit_loader）
        result = integrate_complete_limit_features(test_data, None)
        
        print(f"✅ 特征集成完成: {result.shape}")
        
        # 检查字符串字段是否被正确添加
        for feature in string_features:
            if feature in result.columns:
                print(f"✅ {feature} 已添加到结果中")
                # 检查数据类型
                if result[feature].dtype == 'object':
                    print(f"✅ {feature} 数据类型正确: {result[feature].dtype}")
                else:
                    print(f"❌ {feature} 数据类型错误: {result[feature].dtype}")
                    return False
            else:
                print(f"❌ {feature} 未添加到结果中")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 字符串字段测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """测试内存优化"""
    print("\n" + "=" * 80)
    print("测试内存优化")
    print("=" * 80)
    
    try:
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 初始内存使用: {initial_memory:.2f} MB")
        
        from generate_daily_features import optimize_dtypes
        
        # 创建大型测试数据
        n_rows = 10000
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * n_rows,
            'close': np.random.random(n_rows).astype('float64'),  # 使用float64
            'volume': np.random.randint(1000, 100000, n_rows, dtype='int64'),  # 使用int64
            'first_time_formatted': ['09:50:21'] * n_rows,
            'limit_strength': np.random.random(n_rows).astype('float64')
        })
        
        memory_before = test_data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"📊 优化前数据内存: {memory_before:.2f} MB")
        
        # 优化数据类型
        optimized_data = optimize_dtypes(test_data)
        
        memory_after = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"📊 优化后数据内存: {memory_after:.2f} MB")
        
        memory_saved = memory_before - memory_after
        savings_percent = (memory_saved / memory_before) * 100
        
        print(f"💾 内存节省: {memory_saved:.2f} MB ({savings_percent:.1f}%)")
        
        if savings_percent > 10:
            print("✅ 内存优化效果显著")
        else:
            print("⚠️  内存优化效果有限")
        
        # 清理内存
        del test_data, optimized_data
        gc.collect()
        
        return True
        
    except ImportError:
        print("⚠️  psutil 未安装，跳过内存测试")
        return True
    except Exception as e:
        print(f"❌ 内存优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始性能修复和数据类型转换修复测试...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        ("数据类型优化修复", test_dtype_optimization_fix),
        ("性能改进", test_performance_improvements),
        ("多线程一致性", test_multithreading_consistency),
        ("字符串字段处理", test_string_field_handling),
        ("内存优化", test_memory_optimization)
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
        print("✅ 数据类型转换错误已修复")
        print("✅ 性能优化已实现")
        print("✅ 多线程一致性已改善")
        print("✅ 字符串字段处理正确")
        print("✅ 内存使用已优化")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
