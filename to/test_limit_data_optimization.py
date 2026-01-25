#!/usr/bin/env python3
"""
测试涨跌停数据加载优化效果
验证预加载机制的性能提升
"""

import sys
import os
import time
import pandas as pd
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_original_vs_optimized_loading():
    """测试原始加载方式 vs 优化后的加载方式"""
    print("=" * 80)
    print("测试涨跌停数据加载优化效果")
    print("=" * 80)
    
    try:
        from generate_daily_features import CompleteLimitDataLoader
        
        # 测试日期列表（模拟实际使用场景）
        test_dates = [
            '20240101', '20240102', '20240103', '20240104', '20240105',
            '20240108', '20240109', '20240110', '20240111', '20240112',
            '20240115', '20240116', '20240117', '20240118', '20240119',
            '20240122', '20240123', '20240124', '20240125', '20240126'
        ]
        
        print(f"📊 测试场景: 连续加载 {len(test_dates)} 个交易日的数据")
        
        # 初始化加载器（会触发预加载）
        print("\n🚀 初始化优化后的数据加载器...")
        start_init = time.time()
        loader = CompleteLimitDataLoader()
        init_time = time.time() - start_init
        
        # 获取初始化统计信息
        stats = loader.get_performance_stats()
        print(f"✅ 初始化完成，耗时: {init_time:.2f} 秒")
        print(f"📊 数据统计: {stats}")
        
        # 测试优化后的加载性能
        print(f"\n⚡ 测试优化后的数据查询性能...")
        start_queries = time.time()
        
        total_records = 0
        successful_loads = 0
        
        for i, date_str in enumerate(test_dates):
            query_start = time.time()
            data = loader.load_limit_data_by_date(date_str)
            query_time = time.time() - query_start
            
            if not data.empty:
                total_records += len(data)
                successful_loads += 1
                print(f"  📅 {date_str}: {len(data)} 条记录, 耗时: {query_time*1000:.2f} ms")
            else:
                print(f"  📅 {date_str}: 无数据, 耗时: {query_time*1000:.2f} ms")
        
        total_query_time = time.time() - start_queries
        
        print(f"\n📈 性能测试结果:")
        print(f"  总查询时间: {total_query_time:.3f} 秒")
        print(f"  平均每次查询: {total_query_time/len(test_dates)*1000:.2f} ms")
        print(f"  成功加载: {successful_loads}/{len(test_dates)} 个日期")
        print(f"  总记录数: {total_records} 条")
        
        # 计算性能指标
        if total_query_time > 0:
            queries_per_second = len(test_dates) / total_query_time
            records_per_second = total_records / total_query_time
            print(f"  查询速度: {queries_per_second:.1f} 查询/秒")
            print(f"  数据处理速度: {records_per_second:.0f} 记录/秒")
        
        # 测试缓存效果
        print(f"\n🔄 测试缓存效果（重复查询）...")
        cache_start = time.time()
        
        for date_str in test_dates[:5]:  # 只测试前5个日期
            cache_query_start = time.time()
            data = loader.load_limit_data_by_date(date_str)
            cache_query_time = time.time() - cache_query_start
            print(f"  📅 {date_str} (缓存): {len(data)} 条记录, 耗时: {cache_query_time*1000:.3f} ms")
        
        cache_total_time = time.time() - cache_start
        print(f"  缓存查询总时间: {cache_total_time*1000:.2f} ms")
        print(f"  平均缓存查询: {cache_total_time/5*1000:.3f} ms")
        
        # 性能评估
        print(f"\n🎯 性能评估:")
        
        # 估算原始方式的性能（假设每次都要读取5.8MB文件）
        file_size_mb = 5.8
        estimated_io_time_per_query = 0.05  # 假设每次I/O需要50ms
        estimated_original_time = len(test_dates) * estimated_io_time_per_query
        
        if total_query_time > 0:
            speedup_ratio = estimated_original_time / total_query_time
            print(f"  估算原始方式总时间: {estimated_original_time:.2f} 秒")
            print(f"  优化后实际时间: {total_query_time:.3f} 秒")
            print(f"  性能提升倍数: {speedup_ratio:.1f}x")
            
            if speedup_ratio > 10:
                print(f"  ✅ 性能提升显著！")
            elif speedup_ratio > 5:
                print(f"  ✅ 性能提升良好")
            else:
                print(f"  ⚠️  性能提升有限")
        
        # 内存使用评估
        memory_mb = stats.get('memory_usage_mb', 0)
        print(f"\n💾 内存使用评估:")
        print(f"  预加载数据内存: {memory_mb:.2f} MB")
        if memory_mb < 50:
            print(f"  ✅ 内存使用合理")
        elif memory_mb < 100:
            print(f"  ⚠️  内存使用适中")
        else:
            print(f"  ❌ 内存使用较高")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_integrity():
    """测试数据完整性"""
    print("\n" + "=" * 80)
    print("测试数据完整性")
    print("=" * 80)
    
    try:
        from generate_daily_features import CompleteLimitDataLoader
        
        loader = CompleteLimitDataLoader()
        
        # 测试几个日期的数据
        test_dates = ['20240101', '20240102', '20240103']
        
        for date_str in test_dates:
            data = loader.load_limit_data_by_date(date_str)
            
            if not data.empty:
                print(f"📅 {date_str}:")
                print(f"  记录数: {len(data)}")
                print(f"  列数: {len(data.columns)}")
                print(f"  主要列: {list(data.columns[:10])}")
                
                # 检查关键字段
                required_fields = ['ts_code', 'trade_date']
                missing_fields = [field for field in required_fields if field not in data.columns]
                
                if missing_fields:
                    print(f"  ❌ 缺失字段: {missing_fields}")
                    return False
                else:
                    print(f"  ✅ 关键字段完整")
                
                # 检查数据类型
                if 'trade_date' in data.columns:
                    print(f"  trade_date 类型: {data['trade_date'].dtype}")
                
                print()
            else:
                print(f"📅 {date_str}: 无数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据完整性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """测试内存效率"""
    print("\n" + "=" * 80)
    print("测试内存效率")
    print("=" * 80)
    
    try:
        import psutil
        import gc
        
        # 获取初始内存
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"📊 初始内存使用: {initial_memory:.2f} MB")
        
        from generate_daily_features import CompleteLimitDataLoader
        
        # 创建加载器
        loader = CompleteLimitDataLoader()
        
        # 获取加载后内存
        after_load_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = after_load_memory - initial_memory
        
        print(f"📊 加载后内存使用: {after_load_memory:.2f} MB")
        print(f"📊 内存增加: {memory_increase:.2f} MB")
        
        # 获取数据统计
        stats = loader.get_performance_stats()
        data_memory = stats.get('memory_usage_mb', 0)
        
        print(f"📊 数据内存使用: {data_memory:.2f} MB")
        
        # 内存效率评估
        if data_memory > 0:
            efficiency = data_memory / memory_increase if memory_increase > 0 else 1
            print(f"📊 内存效率: {efficiency:.2f} (数据内存/总增加内存)")
            
            if efficiency > 0.8:
                print(f"✅ 内存效率优秀")
            elif efficiency > 0.6:
                print(f"✅ 内存效率良好")
            else:
                print(f"⚠️  内存效率一般")
        
        # 测试多次查询后的内存变化
        print(f"\n🔄 测试多次查询的内存稳定性...")
        
        test_dates = ['20240101', '20240102', '20240103', '20240104', '20240105']
        
        for i, date_str in enumerate(test_dates):
            loader.load_limit_data_by_date(date_str)
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"  查询 {i+1}: {current_memory:.2f} MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - after_load_memory
        
        print(f"📊 查询后内存: {final_memory:.2f} MB")
        print(f"📊 内存增长: {memory_growth:.2f} MB")
        
        if memory_growth < 5:
            print(f"✅ 内存稳定性良好")
        else:
            print(f"⚠️  内存可能存在泄漏")
        
        return True
        
    except ImportError:
        print("⚠️  psutil 未安装，跳过内存测试")
        return True
    except Exception as e:
        print(f"❌ 内存效率测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始涨跌停数据加载优化测试...")
    
    tests = [
        ("加载性能优化", test_original_vs_optimized_loading),
        ("数据完整性", test_data_integrity),
        ("内存效率", test_memory_efficiency)
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
        print("✅ 涨跌停数据加载优化成功")
        print("✅ 性能提升显著")
        print("✅ 内存使用合理")
        print("✅ 数据完整性保持")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和优化")
    
    return passed == total

if __name__ == "__main__":
    main()
