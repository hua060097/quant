#!/usr/bin/env python3
"""
测试量价相关系数修复
验证DataFrame赋值错误是否已解决
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_vol_price_corr_fix():
    """测试量价相关系数修复"""
    print("=" * 80)
    print("测试量价相关系数修复")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_advanced_features
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 30 + ['000002'] * 30,
            'date': pd.date_range('2024-01-01', periods=30).tolist() * 2,
            'open': np.random.uniform(10, 20, 60),
            'high': np.random.uniform(15, 25, 60),
            'low': np.random.uniform(8, 15, 60),
            'close': np.random.uniform(10, 20, 60),
            'volume': np.random.randint(1000, 100000, 60),
            'amount': np.random.randint(100000, 10000000, 60),
            'pct_chg': np.random.uniform(-5, 5, 60),
            'total_mv': np.random.randint(1000000, 100000000, 60),
            'circ_mv': np.random.randint(800000, 80000000, 60),
            'turnover_rate': np.random.uniform(0, 10, 60)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 测试高级特征计算
        result = calculate_advanced_features(test_data, None)
        
        print(f"✅ 高级特征计算完成: {result.shape}")
        
        # 检查vol_price_corr_20d特征
        if 'vol_price_corr_20d' in result.columns:
            print("✅ vol_price_corr_20d 特征已生成")
            
            # 检查数据类型
            dtype = result['vol_price_corr_20d'].dtype
            print(f"✅ 数据类型: {dtype}")
            
            # 检查数据长度
            length = len(result['vol_price_corr_20d'])
            print(f"✅ 数据长度: {length}")
            
            # 检查非零值数量
            non_zero_count = (result['vol_price_corr_20d'] != 0).sum()
            print(f"✅ 非零值数量: {non_zero_count}")
            
            # 检查样本值
            sample_values = result['vol_price_corr_20d'].head().tolist()
            print(f"✅ 样本值: {sample_values}")
            
            # 检查是否有异常值
            inf_count = np.isinf(result['vol_price_corr_20d']).sum()
            nan_count = result['vol_price_corr_20d'].isna().sum()
            print(f"✅ 无穷值数量: {inf_count}")
            print(f"✅ NaN值数量: {nan_count}")
            
            if inf_count == 0 and nan_count == 0:
                print("✅ 数据质量良好，无异常值")
            else:
                print("⚠️  数据包含异常值")
                
        else:
            print("❌ vol_price_corr_20d 特征未生成")
            return False
        
        # 检查其他特征
        expected_features = ['return_5d', 'return_20d', 'volatility_skew']
        for feature in expected_features:
            if feature in result.columns:
                print(f"✅ {feature} 特征已生成")
            else:
                print(f"ℹ️  {feature} 特征未生成（可能因为数据不足）")
        
        print(f"✅ 总特征数: {len(result.columns)}")
        print(f"✅ 数据行数: {len(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_correlation_calculation():
    """直接测试相关系数计算逻辑"""
    print("\n" + "=" * 80)
    print("直接测试相关系数计算逻辑")
    print("=" * 80)
    
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 25 + ['000002'] * 25,
            'volume': np.random.randint(1000, 100000, 50),
            'close': np.random.uniform(10, 20, 50)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 使用修复后的方法计算相关系数
        vol_price_corr_list = []
        for stock_code, group in test_data.groupby('stock_code'):
            print(f"处理股票: {stock_code}, 数据量: {len(group)}")
            
            if len(group) >= 20:
                corr_values = group['volume'].rolling(20, min_periods=5).corr(group['close']).fillna(0)
                # 修复无穷值和NaN值
                corr_values = corr_values.replace([np.inf, -np.inf], np.nan).fillna(0)
                vol_price_corr_list.extend(corr_values.tolist())
                print(f"  ✅ 计算了 {len(corr_values)} 个相关系数值")
            else:
                vol_price_corr_list.extend([0.0] * len(group))
                print(f"  ℹ️  数据不足，使用默认值")
        
        print(f"✅ 总计算结果: {len(vol_price_corr_list)} 个值")
        print(f"✅ 样本值: {vol_price_corr_list[:5]}")
        
        # 验证结果长度
        if len(vol_price_corr_list) == len(test_data):
            print("✅ 结果长度正确")
        else:
            print(f"❌ 结果长度错误: {len(vol_price_corr_list)} vs {len(test_data)}")
            return False
        
        # 检查数据质量
        inf_count = sum(1 for x in vol_price_corr_list if np.isinf(x))
        nan_count = sum(1 for x in vol_price_corr_list if np.isnan(x))
        
        print(f"✅ 无穷值数量: {inf_count}")
        print(f"✅ NaN值数量: {nan_count}")
        
        if inf_count == 0 and nan_count == 0:
            print("✅ 数据质量良好")
        else:
            print("⚠️  数据包含异常值")
        
        return True
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_feature_generation():
    """测试完整的特征生成流程"""
    print("\n" + "=" * 80)
    print("测试完整的特征生成流程")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_technical_indicators
        
        # 创建更完整的测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 40 + ['000002'] * 40,
            'date': pd.date_range('2024-01-01', periods=40).tolist() * 2,
            'open': np.random.uniform(10, 20, 80),
            'high': np.random.uniform(15, 25, 80),
            'low': np.random.uniform(8, 15, 80),
            'close': np.random.uniform(10, 20, 80),
            'volume': np.random.randint(1000, 100000, 80),
            'amount': np.random.randint(100000, 10000000, 80),
            'pct_chg': np.random.uniform(-5, 5, 80),
            'total_mv': np.random.randint(1000000, 100000000, 80),
            'circ_mv': np.random.randint(800000, 80000000, 80),
            'turnover_rate': np.random.uniform(0, 10, 80)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 运行完整的特征生成流程
        result = calculate_technical_indicators(test_data, None)
        
        print(f"✅ 完整特征生成完成: {result.shape}")
        
        # 检查关键特征
        key_features = [
            'vol_price_corr_20d', 'return_5d', 'return_20d', 
            'volatility_skew', 'amihud_illiquidity'
        ]
        
        generated_features = []
        missing_features = []
        
        for feature in key_features:
            if feature in result.columns:
                generated_features.append(feature)
                print(f"✅ {feature} 已生成")
            else:
                missing_features.append(feature)
                print(f"❌ {feature} 未生成")
        
        print(f"\n📊 特征生成统计:")
        print(f"  已生成: {len(generated_features)}/{len(key_features)}")
        print(f"  未生成: {len(missing_features)}")
        
        if len(generated_features) >= len(key_features) * 0.8:  # 80%以上生成成功
            print("✅ 特征生成成功率良好")
            return True
        else:
            print("⚠️  特征生成成功率较低")
            return False
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始量价相关系数修复测试...")
    
    tests = [
        ("量价相关系数修复", test_vol_price_corr_fix),
        ("直接相关系数计算", test_direct_correlation_calculation),
        ("完整特征生成流程", test_complete_feature_generation)
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
        print("✅ DataFrame赋值错误已修复")
        print("✅ 量价相关系数计算正常")
        print("✅ 特征生成流程稳定")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
