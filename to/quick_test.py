#!/usr/bin/env python3
"""
快速测试修改后的代码
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """测试导入"""
    print("🔧 测试导入...")
    
    try:
        from generate_daily_features import StockFeatureGenerator, LimitDataLoader
        print("✅ 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_limit_loader():
    """测试涨跌停数据加载器"""
    print("🔧 测试涨跌停数据加载器...")
    
    try:
        from generate_daily_features import LimitDataLoader
        
        loader = LimitDataLoader()
        print("✅ 涨跌停数据加载器初始化成功")
        
        # 测试加载数据
        test_date = "20240102"
        limit_data = loader.load_limit_data_by_date(test_date)
        
        if limit_data.empty:
            print(f"⚠️  {test_date} 没有涨跌停数据（正常）")
        else:
            print(f"✅ 成功加载 {len(limit_data)} 条涨跌停数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 涨跌停数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_generator_init():
    """测试特征生成器初始化"""
    print("🔧 测试特征生成器初始化...")
    
    try:
        from generate_daily_features import StockFeatureGenerator
        
        generator = StockFeatureGenerator()
        print("✅ 特征生成器初始化成功")
        
        # 检查涨跌停加载器是否正确初始化
        if hasattr(generator, 'limit_loader'):
            print("✅ 涨跌停加载器已集成")
        else:
            print("❌ 涨跌停加载器未集成")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征生成器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_infinite_value_fixes():
    """测试无穷值修复"""
    print("🔧 测试无穷值修复...")
    
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 100,
            'date': pd.date_range('2024-01-01', periods=100),
            'open': np.random.randn(100) * 5 + 100,
            'high': np.random.randn(100) * 5 + 105,
            'low': np.random.randn(100) * 5 + 95,
            'close': np.random.randn(100) * 5 + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'amount': np.random.randint(100000, 1000000, 100),
            'pct_chg': np.random.randn(100) * 2,
            'total_mv': np.random.randint(1000000, 10000000, 100),
            'circ_mv': np.random.randint(800000, 8000000, 100),
            'turnover_rate': np.random.rand(100) * 10
        })
        
        # 添加一些可能导致无穷值的数据
        test_data.loc[10, 'volume'] = 0  # 可能导致除零
        test_data.loc[20, 'close'] = 0   # 可能导致除零
        
        print(f"✅ 创建测试数据: {test_data.shape}")
        
        # 测试特征计算函数
        from generate_daily_features import calculate_technical_indicators
        
        result = calculate_technical_indicators(test_data)
        print(f"✅ 技术指标计算完成: {result.shape}")
        
        # 检查无穷值
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        
        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"❌ 发现无穷值:")
            for col, count in inf_counts.items():
                print(f"  {col}: {count} 个")
            return False
        else:
            print(f"✅ 无无穷值问题")
            return True
        
    except Exception as e:
        print(f"❌ 无穷值修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_features():
    """测试新增特征"""
    print("🔧 测试新增特征...")
    
    try:
        from generate_daily_features import calculate_technical_indicators
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 50,
            'date': pd.date_range('2024-01-01', periods=50),
            'open': np.random.randn(50) * 5 + 100,
            'high': np.random.randn(50) * 5 + 105,
            'low': np.random.randn(50) * 5 + 95,
            'close': np.random.randn(50) * 5 + 100,
            'volume': np.random.randint(1000, 10000, 50),
            'amount': np.random.randint(100000, 1000000, 50),
            'pct_chg': np.random.randn(50) * 2,
            'total_mv': np.random.randint(1000000, 10000000, 50),
            'circ_mv': np.random.randint(800000, 8000000, 50),
            'turnover_rate': np.random.rand(50) * 10
        })
        
        result = calculate_technical_indicators(test_data)
        
        # 检查新增的涨跌停特征
        new_limit_features = [
            'real_limit_strength', 'real_time_to_limit', 'real_open_times',
            'real_limit_times', 'real_fd_amount', 'limit_quality_score'
        ]
        
        missing_features = []
        for feature in new_limit_features:
            if feature not in result.columns:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ 缺失新增特征: {missing_features}")
            return False
        else:
            print(f"✅ 所有新增特征都存在")
            return True
        
    except Exception as e:
        print(f"❌ 新增特征测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始快速测试...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        ("导入测试", test_imports),
        ("涨跌停加载器测试", test_limit_loader),
        ("特征生成器初始化测试", test_feature_generator_init),
        ("无穷值修复测试", test_infinite_value_fixes),
        ("新增特征测试", test_new_features)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"{test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅ 通过' if result else '❌ 失败'}")
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("💥 部分测试失败！")
    
    return passed == total

if __name__ == "__main__":
    main()
