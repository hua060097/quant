#!/usr/bin/env python3
"""
最终的DeprecationWarning修复验证测试
专门测试修复后的代码是否消除了警告
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_direct_groupby_apply_fix():
    """直接测试修复后的groupby().apply()代码"""
    print("=" * 80)
    print("直接测试修复后的 groupby().apply() 代码")
    print("=" * 80)
    
    try:
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 30 + ['000002'] * 30,
            'volume': np.random.randint(1000, 100000, 60),
            'close': np.random.uniform(10, 20, 60)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 捕获所有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # 测试修复前的代码（会产生警告）
            print("🔍 测试修复前的代码...")
            def calc_rolling_corr_old(group):
                return group['volume'].rolling(20, min_periods=5).corr(group['close']).fillna(0)
            
            # 这种写法会产生DeprecationWarning
            try:
                vol_price_corr_old = test_data.groupby('stock_code').apply(calc_rolling_corr_old).reset_index(level=0, drop=True)
                
                # 检查是否有DeprecationWarning
                old_warnings = [warning for warning in w 
                              if issubclass(warning.category, DeprecationWarning)
                              and "grouping columns" in str(warning.message)]
                
                if old_warnings:
                    print(f"❌ 修复前代码产生 {len(old_warnings)} 个 DeprecationWarning（预期）")
                    for warning in old_warnings[:1]:  # 只显示第一个
                        print(f"  警告内容: {warning.message}")
                else:
                    print("⚠️  修复前代码未产生 DeprecationWarning（意外）")
                    
            except Exception as e:
                print(f"❌ 修复前代码执行失败: {e}")
            
            # 清空警告记录
            w.clear()
            
            # 测试修复后的代码（不应产生警告）
            print("🔍 测试修复后的代码...")
            def calc_rolling_corr_new(group):
                return group['volume'].rolling(20, min_periods=5).corr(group['close']).fillna(0)
            
            # 使用修复后的语法
            vol_price_corr_new = test_data.groupby('stock_code').apply(calc_rolling_corr_new, include_groups=False).reset_index(level=0, drop=True)
            
            # 检查是否有DeprecationWarning
            new_warnings = [warning for warning in w 
                          if issubclass(warning.category, DeprecationWarning)
                          and "grouping columns" in str(warning.message)]
            
            if new_warnings:
                print(f"❌ 修复后代码仍产生 {len(new_warnings)} 个 DeprecationWarning")
                for warning in new_warnings:
                    print(f"  警告内容: {warning.message}")
                return False
            else:
                print("✅ 修复后代码无 DeprecationWarning")
            
            # 检查结果是否正确
            if len(vol_price_corr_new) == len(test_data):
                print("✅ 计算结果长度正确")
            else:
                print(f"❌ 计算结果长度错误: {len(vol_price_corr_new)} vs {len(test_data)}")
                return False
            
            # 检查结果是否有效
            valid_count = (~vol_price_corr_new.isna()).sum()
            print(f"✅ 有效结果数量: {valid_count}/{len(vol_price_corr_new)}")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actual_feature_calculation():
    """测试实际的特征计算函数"""
    print("\n" + "=" * 80)
    print("测试实际的特征计算函数")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_advanced_features
        
        # 创建足够的测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 60 + ['000002'] * 60,  # 增加数据量
            'date': pd.date_range('2024-01-01', periods=60).tolist() * 2,
            'open': np.random.uniform(10, 20, 120),
            'high': np.random.uniform(15, 25, 120),
            'low': np.random.uniform(8, 15, 120),
            'close': np.random.uniform(10, 20, 120),
            'volume': np.random.randint(1000, 100000, 120),
            'amount': np.random.randint(100000, 10000000, 120),
            'pct_chg': np.random.uniform(-5, 5, 120),
            'total_mv': np.random.randint(1000000, 100000000, 120),
            'circ_mv': np.random.randint(800000, 80000000, 120),
            'turnover_rate': np.random.uniform(0, 10, 120)
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        
        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # 调用实际的特征计算函数
            result = calculate_advanced_features(test_data, None)
            
            # 检查DeprecationWarning
            deprecation_warnings = [warning for warning in w 
                                  if issubclass(warning.category, DeprecationWarning)
                                  and "grouping columns" in str(warning.message)]
            
            if deprecation_warnings:
                print(f"❌ 特征计算产生 {len(deprecation_warnings)} 个 DeprecationWarning:")
                for warning in deprecation_warnings:
                    print(f"  {warning.filename}:{warning.lineno}: {warning.message}")
                return False
            else:
                print("✅ 特征计算无 DeprecationWarning")
            
            # 检查其他类型的警告
            other_warnings = [warning for warning in w 
                            if not issubclass(warning.category, DeprecationWarning)]
            
            if other_warnings:
                print(f"ℹ️  发现其他类型警告 {len(other_warnings)} 个:")
                for warning in other_warnings[:2]:  # 只显示前2个
                    print(f"  {warning.category.__name__}: {warning.message}")
            else:
                print("✅ 无其他类型警告")
        
        print(f"✅ 特征计算完成: {result.shape}")
        
        # 检查关键特征
        if 'vol_price_corr_20d' in result.columns:
            print("✅ vol_price_corr_20d 特征已生成")
            non_zero_count = (result['vol_price_corr_20d'] != 0).sum()
            print(f"✅ 非零值数量: {non_zero_count}")
        else:
            print("ℹ️  vol_price_corr_20d 特征未生成（可能因为数据不足）")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pandas_version_info():
    """显示pandas版本信息"""
    print("\n" + "=" * 80)
    print("pandas 版本信息")
    print("=" * 80)
    
    try:
        import pandas as pd
        
        print(f"📊 pandas 版本: {pd.__version__}")
        
        # 检查版本特性
        version_parts = pd.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        print(f"📊 版本解析: {major}.{minor}")
        
        if major >= 2:
            print("✅ pandas 2.x 版本，支持最新特性")
        elif major == 1 and minor >= 5:
            print("✅ pandas 1.5+ 版本，支持 include_groups 参数")
        else:
            print("⚠️  pandas 版本较旧，可能不支持某些新特性")
        
        # 测试include_groups参数
        test_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        
        try:
            result = test_df.groupby('group').apply(lambda x: x['value'].sum(), include_groups=False)
            print("✅ include_groups 参数可用")
        except Exception as e:
            print(f"❌ include_groups 参数不可用: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 版本信息获取失败: {e}")
        return False

def test_comprehensive_warning_check():
    """综合警告检查"""
    print("\n" + "=" * 80)
    print("综合警告检查")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_technical_indicators
        
        # 创建测试数据
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
        
        # 设置严格的警告捕获
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.filterwarnings("always", category=DeprecationWarning)
            
            # 运行完整的特征计算流程
            result = calculate_technical_indicators(test_data, None)
            
            # 分类统计警告
            deprecation_warnings = []
            future_warnings = []
            other_warnings = []
            
            for warning in w:
                if issubclass(warning.category, DeprecationWarning):
                    if "grouping columns" in str(warning.message):
                        deprecation_warnings.append(warning)
                elif issubclass(warning.category, FutureWarning):
                    future_warnings.append(warning)
                else:
                    other_warnings.append(warning)
            
            # 报告结果
            print(f"📊 警告统计:")
            print(f"  DeprecationWarning (grouping columns): {len(deprecation_warnings)}")
            print(f"  FutureWarning: {len(future_warnings)}")
            print(f"  其他警告: {len(other_warnings)}")
            
            # 详细显示关键警告
            if deprecation_warnings:
                print(f"\n❌ 发现 {len(deprecation_warnings)} 个关键 DeprecationWarning:")
                for warning in deprecation_warnings[:2]:  # 只显示前2个
                    print(f"  {warning.filename}:{warning.lineno}")
                    print(f"  {warning.message}")
                return False
            else:
                print(f"\n✅ 无关键 DeprecationWarning")
            
            if future_warnings:
                print(f"\nℹ️  发现 {len(future_warnings)} 个 FutureWarning（通常可忽略）")
            
            if other_warnings:
                print(f"\nℹ️  发现 {len(other_warnings)} 个其他警告")
        
        print(f"✅ 特征计算完成: {result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 综合检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始最终的 DeprecationWarning 修复验证...")
    
    tests = [
        ("pandas 版本信息", test_pandas_version_info),
        ("直接测试修复代码", test_direct_groupby_apply_fix),
        ("实际特征计算", test_actual_feature_calculation),
        ("综合警告检查", test_comprehensive_warning_check)
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
    print("最终测试总结")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        print("✅ DeprecationWarning 已完全修复")
        print("✅ pandas 兼容性问题已解决")
        print("✅ 代码可以正常运行")
        print("\n💡 修复要点:")
        print("  - 使用 groupby().apply(func, include_groups=False)")
        print("  - 而不是 groupby(include_groups=False).apply(func)")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
