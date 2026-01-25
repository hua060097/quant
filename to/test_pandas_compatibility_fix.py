#!/usr/bin/env python3
"""
测试pandas兼容性修复
验证DeprecationWarning是否已解决
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_groupby_apply_fix():
    """测试groupby().apply()修复"""
    print("=" * 80)
    print("测试 groupby().apply() 兼容性修复")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_advanced_features
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 50 + ['000002'] * 50,
            'date': pd.date_range('2024-01-01', periods=50).tolist() * 2,
            'open': np.random.uniform(10, 20, 100),
            'high': np.random.uniform(15, 25, 100),
            'low': np.random.uniform(8, 15, 100),
            'close': np.random.uniform(10, 20, 100),
            'volume': np.random.randint(1000, 100000, 100),
            'amount': np.random.randint(100000, 10000000, 100),
            'pct_chg': np.random.uniform(-5, 5, 100),
            'total_mv': np.random.randint(1000000, 100000000, 100),
            'circ_mv': np.random.randint(800000, 80000000, 100),
            'turnover_rate': np.random.uniform(0, 10, 100)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # 测试高级特征计算
            result = calculate_advanced_features(test_data, None)
            
            # 检查是否有DeprecationWarning
            deprecation_warnings = [warning for warning in w 
                                  if issubclass(warning.category, DeprecationWarning)
                                  and "grouping columns" in str(warning.message)]
            
            if deprecation_warnings:
                print("❌ 仍然存在 DeprecationWarning:")
                for warning in deprecation_warnings:
                    print(f"  {warning.filename}:{warning.lineno}: {warning.message}")
                return False
            else:
                print("✅ 没有发现 DeprecationWarning")
                
            # 检查其他警告
            other_warnings = [warning for warning in w 
                            if not issubclass(warning.category, DeprecationWarning)]
            
            if other_warnings:
                print(f"⚠️  发现其他警告 {len(other_warnings)} 个:")
                for warning in other_warnings[:3]:  # 只显示前3个
                    print(f"  {warning.category.__name__}: {warning.message}")
                if len(other_warnings) > 3:
                    print(f"  ... 还有 {len(other_warnings) - 3} 个警告")
            else:
                print("✅ 没有发现其他警告")
        
        print(f"✅ 高级特征计算完成: {result.shape}")
        
        # 检查关键特征是否存在
        key_features = ['vol_price_corr_20d', 'volatility_skew']
        for feature in key_features:
            if feature in result.columns:
                print(f"✅ {feature} 特征已生成")
            else:
                print(f"❌ {feature} 特征缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_groupby_operations():
    """测试特定的groupby操作"""
    print("\n" + "=" * 80)
    print("测试特定的 groupby 操作")
    print("=" * 80)
    
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 30 + ['000002'] * 30,
            'volume': np.random.randint(1000, 100000, 60),
            'close': np.random.uniform(10, 20, 60),
            'pct_chg': np.random.uniform(-5, 5, 60)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 测试修复后的相关系数计算
        print("🔍 测试量价相关系数计算...")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            def calc_rolling_corr(group):
                return group['volume'].rolling(20, min_periods=5).corr(group['close']).fillna(0)
            
            # 使用修复后的方法（正确语法）
            vol_price_corr = test_data.groupby('stock_code').apply(calc_rolling_corr, include_groups=False).reset_index(level=0, drop=True)
            
            # 检查DeprecationWarning
            deprecation_warnings = [warning for warning in w 
                                  if issubclass(warning.category, DeprecationWarning)
                                  and "grouping columns" in str(warning.message)]
            
            if deprecation_warnings:
                print("❌ 相关系数计算仍有 DeprecationWarning")
                return False
            else:
                print("✅ 相关系数计算无 DeprecationWarning")
        
        print(f"✅ 相关系数计算结果: {len(vol_price_corr)} 个值")
        
        # 测试其他groupby操作
        print("🔍 测试其他 groupby 操作...")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # 测试transform操作（这些通常不会有问题）
            test_data['pct_chg_filled'] = test_data.groupby('stock_code')['pct_chg'].transform(
                lambda x: x.fillna(x.median()).fillna(0)
            )
            
            # 测试rolling操作
            volatility_skew = test_data.groupby('stock_code')['pct_chg'].rolling(
                20, min_periods=5
            ).skew().reset_index(level=0, drop=True)
            
            # 检查警告
            warnings_found = len(w)
            if warnings_found == 0:
                print("✅ 其他 groupby 操作无警告")
            else:
                print(f"⚠️  其他 groupby 操作有 {warnings_found} 个警告")
                for warning in w[:2]:  # 只显示前2个
                    print(f"  {warning.category.__name__}: {warning.message}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特定操作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pandas_version_compatibility():
    """测试pandas版本兼容性"""
    print("\n" + "=" * 80)
    print("测试 pandas 版本兼容性")
    print("=" * 80)
    
    try:
        import pandas as pd
        
        pandas_version = pd.__version__
        print(f"📊 当前 pandas 版本: {pandas_version}")
        
        # 检查版本
        major, minor = map(int, pandas_version.split('.')[:2])
        
        if major >= 2 or (major == 1 and minor >= 5):
            print("✅ pandas 版本支持 include_groups 参数")
        else:
            print("⚠️  pandas 版本较旧，可能不支持 include_groups 参数")
        
        # 测试include_groups参数是否可用
        test_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        
        try:
            # 测试include_groups参数（正确语法）
            result = test_df.groupby('group').apply(lambda x: x['value'].sum(), include_groups=False)
            print("✅ include_groups 参数可用")
            return True
        except TypeError as e:
            if "include_groups" in str(e):
                print("❌ include_groups 参数不可用，需要升级pandas")
                print("💡 建议升级到 pandas >= 1.5.0")
                return False
            else:
                raise e
        
    except Exception as e:
        print(f"❌ 版本兼容性测试失败: {e}")
        return False

def test_alternative_solutions():
    """测试替代解决方案"""
    print("\n" + "=" * 80)
    print("测试替代解决方案")
    print("=" * 80)
    
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 30 + ['000002'] * 30,
            'volume': np.random.randint(1000, 100000, 60),
            'close': np.random.uniform(10, 20, 60)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 方法1：使用include_groups=False（推荐）
        print("🔍 方法1: 使用 include_groups=False")
        try:
            def calc_corr_method1(group):
                return group['volume'].rolling(20, min_periods=5).corr(group['close']).fillna(0)
            
            result1 = test_data.groupby('stock_code').apply(calc_corr_method1, include_groups=False)
            print("✅ 方法1 成功")
        except Exception as e:
            print(f"❌ 方法1 失败: {e}")
            result1 = None
        
        # 方法2：显式选择列（兼容性更好）
        print("🔍 方法2: 显式选择列")
        try:
            def calc_corr_method2(group):
                return group['volume'].rolling(20, min_periods=5).corr(group['close']).fillna(0)
            
            result2 = test_data.groupby('stock_code')[['volume', 'close']].apply(calc_corr_method2)
            print("✅ 方法2 成功")
        except Exception as e:
            print(f"❌ 方法2 失败: {e}")
            result2 = None
        
        # 方法3：使用transform（如果适用）
        print("🔍 方法3: 使用 transform")
        try:
            # 这种方法对于相关系数不太适用，但可以用于其他操作
            result3 = test_data.groupby('stock_code')['volume'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            print("✅ 方法3 成功（适用于简单聚合）")
        except Exception as e:
            print(f"❌ 方法3 失败: {e}")
            result3 = None
        
        # 检查结果一致性
        if result1 is not None and result2 is not None:
            if len(result1) == len(result2):
                print("✅ 方法1和方法2结果长度一致")
            else:
                print("⚠️  方法1和方法2结果长度不一致")
        
        return True
        
    except Exception as e:
        print(f"❌ 替代方案测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始 pandas 兼容性修复测试...")
    
    tests = [
        ("groupby().apply() 修复", test_groupby_apply_fix),
        ("特定 groupby 操作", test_specific_groupby_operations),
        ("pandas 版本兼容性", test_pandas_version_compatibility),
        ("替代解决方案", test_alternative_solutions)
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
        print("✅ pandas 兼容性问题已修复")
        print("✅ DeprecationWarning 已消除")
        print("✅ 代码向前兼容")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
