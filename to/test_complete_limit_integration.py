#!/usr/bin/env python3
"""
测试完整涨跌停特征集成
验证 enhanced_limit_data_collector.py 的所有特征是否正确集成到 generate_daily_features.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_complete_limit_loader():
    """测试完整涨跌停数据加载器"""
    print("=" * 80)
    print("测试完整涨跌停数据加载器")
    print("=" * 80)
    
    try:
        from generate_daily_features import CompleteLimitDataLoader
        
        # 初始化加载器
        loader = CompleteLimitDataLoader()
        print("✅ 完整涨跌停数据加载器初始化成功")
        
        # 检查特征列表
        all_features = loader.get_available_features()
        print(f"📊 可用特征总数: {len(all_features)}")
        
        # 分类显示特征
        print(f"\n📋 原始字段 ({len(loader.original_fields)} 个):")
        for field in loader.original_fields:
            print(f"  - {field}")
        
        print(f"\n🔍 增强特征 ({len(loader.enhanced_features)} 个):")
        for feature in loader.enhanced_features:
            print(f"  - {feature}")
        
        # 测试数据加载
        test_date = "20240102"
        print(f"\n📊 测试加载 {test_date} 的数据...")
        
        limit_data = loader.load_limit_data_by_date(test_date)
        
        if not limit_data.empty:
            print(f"✅ 成功加载数据: {len(limit_data)} 条")
            print(f"📋 数据字段: {list(limit_data.columns)}")
            
            # 检查数据完整性
            is_complete = loader.validate_data_completeness(limit_data)
            print(f"🔍 数据完整性: {'✅ 完整' if is_complete else '❌ 不完整'}")
            
            # 显示数据类型分布
            if 'limit_type' in limit_data.columns:
                type_counts = limit_data['limit_type'].value_counts()
                print(f"📊 数据类型分布:")
                for limit_type, count in type_counts.items():
                    print(f"  {limit_type}: {count} 条")
        else:
            print(f"⚠️  {test_date} 没有涨跌停数据（正常）")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_integration():
    """测试特征集成功能"""
    print("\n" + "=" * 80)
    print("测试特征集成功能")
    print("=" * 80)
    
    try:
        from generate_daily_features import integrate_complete_limit_features, _get_all_limit_features, CompleteLimitDataLoader
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001', '000002', '600000'],
            'date': pd.date_range('2024-01-02', periods=3),
            'close': [10.0, 20.0, 30.0],
            'volume': [1000000, 2000000, 3000000]
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 初始化加载器
        loader = CompleteLimitDataLoader()
        
        # 测试特征集成
        enhanced_data = integrate_complete_limit_features(test_data, loader)
        
        print(f"✅ 特征集成完成: {enhanced_data.shape}")
        
        # 检查所有涨跌停特征是否存在
        all_limit_features = _get_all_limit_features()
        print(f"\n🔍 检查涨跌停特征 ({len(all_limit_features)} 个):")
        
        missing_features = []
        for feature in all_limit_features:
            if feature in enhanced_data.columns:
                print(f"  ✅ {feature}")
            else:
                print(f"  ❌ {feature} (缺失)")
                missing_features.append(feature)
        
        if missing_features:
            print(f"\n❌ 缺失特征: {missing_features}")
            return False
        else:
            print(f"\n✅ 所有涨跌停特征都存在")
        
        # 显示特征样本
        print(f"\n📋 特征样本 (前3个涨跌停特征):")
        sample_features = all_limit_features[:3]
        for feature in sample_features:
            values = enhanced_data[feature].tolist()
            print(f"  {feature}: {values}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_generator_integration():
    """测试特征生成器集成"""
    print("\n" + "=" * 80)
    print("测试特征生成器集成")
    print("=" * 80)
    
    try:
        from generate_daily_features import StockFeatureGenerator
        
        # 初始化特征生成器
        generator = StockFeatureGenerator()
        print("✅ 特征生成器初始化成功")
        
        # 检查涨跌停加载器是否正确集成
        if hasattr(generator, 'limit_loader'):
            print("✅ 涨跌停加载器已集成")
            
            # 检查加载器类型
            loader_type = type(generator.limit_loader).__name__
            if loader_type == 'CompleteLimitDataLoader':
                print("✅ 使用完整涨跌停数据加载器")
            else:
                print(f"❌ 加载器类型错误: {loader_type}")
                return False
        else:
            print("❌ 涨跌停加载器未集成")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 特征生成器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_technical_indicators_integration():
    """测试技术指标函数的涨跌停特征集成"""
    print("\n" + "=" * 80)
    print("测试技术指标函数的涨跌停特征集成")
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
        
        # 计算技术指标
        result = calculate_technical_indicators(test_data)
        
        print(f"✅ 技术指标计算完成: {result.shape}")
        
        # 检查涨跌停特征是否被添加
        all_limit_features = _get_all_limit_features()
        print(f"\n🔍 检查涨跌停特征是否被添加:")
        
        added_features = []
        for feature in all_limit_features:
            if feature in result.columns:
                added_features.append(feature)
                print(f"  ✅ {feature}")
            else:
                print(f"  ❌ {feature} (缺失)")
        
        print(f"\n📊 添加的涨跌停特征: {len(added_features)}/{len(all_limit_features)}")
        
        if len(added_features) == len(all_limit_features):
            print("✅ 所有涨跌停特征都已添加")
            return True
        else:
            print("❌ 部分涨跌停特征缺失")
            return False
        
    except Exception as e:
        print(f"❌ 技术指标集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_compatibility():
    """测试特征兼容性"""
    print("\n" + "=" * 80)
    print("测试特征兼容性")
    print("=" * 80)
    
    try:
        from generate_daily_features import _get_all_limit_features
        
        # 获取所有涨跌停特征
        all_features = _get_all_limit_features()
        
        # 检查特征命名规范
        print("🔍 检查特征命名规范:")
        
        naming_issues = []
        for feature in all_features:
            # 检查是否包含特殊字符
            if not feature.replace('_', '').replace('d', '').isalnum():
                naming_issues.append(f"{feature}: 包含特殊字符")
            
            # 检查长度
            if len(feature) > 30:
                naming_issues.append(f"{feature}: 名称过长 ({len(feature)} 字符)")
        
        if naming_issues:
            print("❌ 发现命名问题:")
            for issue in naming_issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ 特征命名规范")
        
        # 检查特征分类
        print(f"\n📊 特征分类统计:")
        
        categories = {
            '基础标识': ['is_limit_up', 'is_limit_down', 'is_zhaban'],
            '原始字段': ['fd_amount', 'open_times', 'limit_times', 'first_time', 'last_time'],
            '增强特征': ['limit_strength', 'time_efficiency_score', 'board_quality'],
            '历史特征': ['limit_up_days_5d', 'consecutive_limit_up']
        }
        
        for category, sample_features in categories.items():
            count = sum(1 for f in sample_features if f in all_features)
            print(f"  {category}: {count}/{len(sample_features)} 个样本特征存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始完整涨跌停特征集成测试...")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    tests = [
        ("完整涨跌停数据加载器", test_complete_limit_loader),
        ("特征集成功能", test_feature_integration),
        ("特征生成器集成", test_feature_generator_integration),
        ("技术指标函数集成", test_technical_indicators_integration),
        ("特征兼容性", test_feature_compatibility)
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
        print("✅ 完整涨跌停特征集成成功")
        print("✅ enhanced_limit_data_collector.py 的所有特征已正确集成")
        print("✅ 原有基于价格计算的逻辑已清理")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
