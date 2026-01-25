#!/usr/bin/env python3
"""
测试 limit_loader 为 None 时的错误修复
验证所有相关函数能够正确处理 limit_loader=None 的情况
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_integrate_complete_limit_features_with_none():
    """测试 integrate_complete_limit_features 函数处理 None 的情况"""
    print("=" * 80)
    print("测试 integrate_complete_limit_features 函数处理 None 的情况")
    print("=" * 80)
    
    try:
        from generate_daily_features import integrate_complete_limit_features, _get_all_limit_features
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001', '000002'],
            'date': pd.date_range('2024-01-02', periods=2),
            'close': [10.0, 20.0],
            'volume': [1000000, 2000000]
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 测试 limit_loader=None 的情况
        result = integrate_complete_limit_features(test_data, None)
        
        print(f"✅ 函数执行成功: {result.shape}")
        
        # 检查是否添加了所有涨跌停特征
        all_limit_features = _get_all_limit_features()
        missing_features = []
        
        for feature in all_limit_features:
            if feature not in result.columns:
                missing_features.append(feature)
            else:
                # 检查默认值是否为0
                if not all(result[feature] == 0.0):
                    print(f"⚠️  {feature} 的值不全为0: {result[feature].unique()}")
        
        if missing_features:
            print(f"❌ 缺失特征: {missing_features}")
            return False
        else:
            print(f"✅ 所有 {len(all_limit_features)} 个涨跌停特征都已添加，默认值为0")
            return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calculate_technical_indicators_with_none():
    """测试 calculate_technical_indicators 函数处理 None 的情况"""
    print("\n" + "=" * 80)
    print("测试 calculate_technical_indicators 函数处理 None 的情况")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_technical_indicators, _get_all_limit_features
        
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
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 测试 limit_loader=None 的情况
        result = calculate_technical_indicators(test_data, None)
        
        print(f"✅ 函数执行成功: {result.shape}")
        
        # 检查是否添加了涨跌停特征
        all_limit_features = _get_all_limit_features()
        added_features = []
        
        for feature in all_limit_features:
            if feature in result.columns:
                added_features.append(feature)
        
        print(f"📊 添加的涨跌停特征: {len(added_features)}/{len(all_limit_features)}")
        
        if len(added_features) == len(all_limit_features):
            print("✅ 所有涨跌停特征都已添加")
            return True
        else:
            missing = [f for f in all_limit_features if f not in result.columns]
            print(f"❌ 缺失特征: {missing}")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calculate_advanced_features_with_none():
    """测试 calculate_advanced_features 函数处理 None 的情况"""
    print("\n" + "=" * 80)
    print("测试 calculate_advanced_features 函数处理 None 的情况")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_advanced_features, _get_all_limit_features
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 30,
            'date': pd.date_range('2024-01-01', periods=30),
            'open': np.random.randn(30) * 5 + 100,
            'high': np.random.randn(30) * 5 + 105,
            'low': np.random.randn(30) * 5 + 95,
            'close': np.random.randn(30) * 5 + 100,
            'volume': np.random.randint(1000, 10000, 30),
            'ma5': np.random.randn(30) * 5 + 100,
            'ma20': np.random.randn(30) * 5 + 100,
            'ma60': np.random.randn(30) * 5 + 100,
            'turnover_rate': np.random.rand(30) * 10
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 测试 limit_loader=None 的情况
        result = calculate_advanced_features(test_data, None)
        
        print(f"✅ 函数执行成功: {result.shape}")
        
        # 检查是否添加了涨跌停特征
        all_limit_features = _get_all_limit_features()
        added_features = []
        
        for feature in all_limit_features:
            if feature in result.columns:
                added_features.append(feature)
        
        print(f"📊 添加的涨跌停特征: {len(added_features)}/{len(all_limit_features)}")
        
        if len(added_features) == len(all_limit_features):
            print("✅ 所有涨跌停特征都已添加")
            return True
        else:
            missing = [f for f in all_limit_features if f not in result.columns]
            print(f"❌ 缺失特征: {missing}")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stock_feature_generator_initialization():
    """测试 StockFeatureGenerator 初始化"""
    print("\n" + "=" * 80)
    print("测试 StockFeatureGenerator 初始化")
    print("=" * 80)
    
    try:
        from generate_daily_features import StockFeatureGenerator
        
        # 初始化特征生成器
        generator = StockFeatureGenerator()
        print("✅ StockFeatureGenerator 初始化成功")
        
        # 检查 limit_loader 是否正确初始化
        if hasattr(generator, 'limit_loader'):
            loader_type = type(generator.limit_loader).__name__
            print(f"📊 limit_loader 类型: {loader_type}")
            
            if loader_type == 'CompleteLimitDataLoader':
                print("✅ 使用完整涨跌停数据加载器")
                return True
            else:
                print(f"❌ 加载器类型错误: {loader_type}")
                return False
        else:
            print("❌ limit_loader 属性不存在")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_scenario():
    """测试原始错误场景"""
    print("\n" + "=" * 80)
    print("测试原始错误场景")
    print("=" * 80)
    
    try:
        from generate_daily_features import _process_stock_features_threaded
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 30,
            'date': pd.date_range('2024-01-01', periods=30),
            'open': np.random.randn(30) * 5 + 100,
            'high': np.random.randn(30) * 5 + 105,
            'low': np.random.randn(30) * 5 + 95,
            'close': np.random.randn(30) * 5 + 100,
            'volume': np.random.randint(1000, 10000, 30),
            'ma5': np.random.randn(30) * 5 + 100,
            'ma20': np.random.randn(30) * 5 + 100,
            'ma60': np.random.randn(30) * 5 + 100,
            'turnover_rate': np.random.rand(30) * 10
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 测试原始错误场景：limit_loader=None
        result = _process_stock_features_threaded('000001', test_data, None)
        
        print(f"✅ 函数执行成功: {result.shape}")
        print("✅ 原始错误已修复")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试 limit_loader=None 错误修复...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        ("integrate_complete_limit_features 处理 None", test_integrate_complete_limit_features_with_none),
        ("calculate_technical_indicators 处理 None", test_calculate_technical_indicators_with_none),
        ("calculate_advanced_features 处理 None", test_calculate_advanced_features_with_none),
        ("StockFeatureGenerator 初始化", test_stock_feature_generator_initialization),
        ("原始错误场景", test_error_scenario)
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
        print("✅ limit_loader=None 错误已修复")
        print("✅ 所有相关函数都能正确处理 None 参数")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
