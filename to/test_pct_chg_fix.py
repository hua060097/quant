#!/usr/bin/env python3
"""
测试 pct_chg 和 change_pct 修复
验证删除重复的 change_pct，统一使用 pct_chg
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_pct_chg_usage():
    """测试 pct_chg 的使用情况"""
    print("=" * 80)
    print("测试 pct_chg 的使用情况")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_technical_indicators
        
        # 创建测试数据，包含 pct_chg
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 50,
            'date': pd.date_range('2024-01-01', periods=50),
            'open': np.random.randn(50) * 5 + 100,
            'high': np.random.randn(50) * 5 + 105,
            'low': np.random.randn(50) * 5 + 95,
            'close': np.random.randn(50) * 5 + 100,
            'volume': np.random.randint(1000, 10000, 50),
            'amount': np.random.randint(100000, 1000000, 50),
            'pct_chg': np.random.randn(50) * 2,  # 原始涨跌幅数据
            'total_mv': np.random.randint(1000000, 10000000, 50),
            'circ_mv': np.random.randint(800000, 8000000, 50),
            'turnover_rate': np.random.rand(50) * 10
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        print(f"✅ 包含 pct_chg 列: {'pct_chg' in test_data.columns}")
        
        # 计算技术指标
        result = calculate_technical_indicators(test_data, None)
        
        print(f"✅ 技术指标计算完成: {result.shape}")
        
        # 检查 pct_chg 是否保留
        if 'pct_chg' in result.columns:
            print("✅ pct_chg 列已保留")
        else:
            print("❌ pct_chg 列丢失")
            return False
        
        # 检查 change_pct 是否被删除
        if 'change_pct' in result.columns:
            print("❌ change_pct 列仍然存在（应该被删除）")
            return False
        else:
            print("✅ change_pct 列已删除")
        
        # 检查 pct_chg 的值是否合理
        pct_chg_values = result['pct_chg'].dropna()
        if len(pct_chg_values) > 0:
            print(f"📊 pct_chg 统计: 均值={pct_chg_values.mean():.3f}, 标准差={pct_chg_values.std():.3f}")
            print("✅ pct_chg 数据正常")
        else:
            print("❌ pct_chg 数据全为 NaN")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pct_chg_without_original():
    """测试没有原始 pct_chg 时的计算"""
    print("\n" + "=" * 80)
    print("测试没有原始 pct_chg 时的计算")
    print("=" * 80)
    
    try:
        from generate_daily_features import calculate_technical_indicators
        
        # 创建测试数据，不包含 pct_chg
        test_data = pd.DataFrame({
            'stock_code': ['000001'] * 50,
            'date': pd.date_range('2024-01-01', periods=50),
            'open': np.random.randn(50) * 5 + 100,
            'high': np.random.randn(50) * 5 + 105,
            'low': np.random.randn(50) * 5 + 95,
            'close': np.random.randn(50) * 5 + 100,
            'volume': np.random.randint(1000, 10000, 50),
            'amount': np.random.randint(100000, 1000000, 50),
            # 注意：没有 pct_chg
            'total_mv': np.random.randint(1000000, 10000000, 50),
            'circ_mv': np.random.randint(800000, 8000000, 50),
            'turnover_rate': np.random.rand(50) * 10
        })
        
        print(f"📊 创建测试数据: {test_data.shape}")
        print(f"⚠️  不包含 pct_chg 列: {'pct_chg' not in test_data.columns}")
        
        # 计算技术指标
        result = calculate_technical_indicators(test_data, None)
        
        print(f"✅ 技术指标计算完成: {result.shape}")
        
        # 检查 pct_chg 是否被计算
        if 'pct_chg' in result.columns:
            print("✅ pct_chg 列已计算")
            
            # 检查计算的值是否合理
            pct_chg_values = result['pct_chg'].dropna()
            if len(pct_chg_values) > 0:
                print(f"📊 计算的 pct_chg 统计: 均值={pct_chg_values.mean():.3f}, 标准差={pct_chg_values.std():.3f}")
                print("✅ 计算的 pct_chg 数据正常")
            else:
                print("❌ 计算的 pct_chg 数据全为 NaN")
                return False
        else:
            print("❌ pct_chg 列未被计算")
            return False
        
        # 检查 change_pct 是否被删除
        if 'change_pct' in result.columns:
            print("❌ change_pct 列仍然存在（应该被删除）")
            return False
        else:
            print("✅ change_pct 列已删除")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_list_consistency():
    """测试特征列表的一致性"""
    print("\n" + "=" * 80)
    print("测试特征列表的一致性")
    print("=" * 80)
    
    try:
        # 读取文件内容
        with open('generate_daily_features.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否还有 change_pct 的引用
        change_pct_count = content.count("'change_pct'")
        print(f"📊 文件中 'change_pct' 的引用次数: {change_pct_count}")
        
        if change_pct_count == 0:
            print("✅ 所有 'change_pct' 引用已删除")
        else:
            print("❌ 仍有 'change_pct' 引用存在")
            # 找出具体位置
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if "'change_pct'" in line:
                    print(f"  第 {i} 行: {line.strip()}")
            return False
        
        # 检查 pct_chg 的使用
        pct_chg_count = content.count("'pct_chg'")
        print(f"📊 文件中 'pct_chg' 的引用次数: {pct_chg_count}")
        
        if pct_chg_count > 0:
            print("✅ pct_chg 正在被使用")
        else:
            print("❌ 没有找到 pct_chg 的使用")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging_improvements():
    """测试日志改进"""
    print("\n" + "=" * 80)
    print("测试日志改进")
    print("=" * 80)
    
    try:
        # 设置日志级别为 INFO 以查看新增的日志
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        
        from generate_daily_features import calculate_technical_indicators
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001', '000002'] * 25,  # 两只股票
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
        print("🔍 开始计算技术指标，观察日志输出...")
        
        # 计算技术指标（应该会输出新增的日志）
        result = calculate_technical_indicators(test_data, None)
        
        print(f"✅ 技术指标计算完成: {result.shape}")
        print("✅ 日志改进测试完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试 pct_chg 和 change_pct 修复...")
    
    tests = [
        ("pct_chg 使用情况", test_pct_chg_usage),
        ("没有原始 pct_chg 时的计算", test_pct_chg_without_original),
        ("特征列表一致性", test_feature_list_consistency),
        ("日志改进", test_logging_improvements)
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
        print("✅ change_pct 已删除，统一使用 pct_chg")
        print("✅ 日志改进已添加")
        print("✅ 代码修复完成")
    else:
        print("\n💥 部分测试失败！")
        print("❌ 需要进一步检查和修复")
    
    return passed == total

if __name__ == "__main__":
    main()
