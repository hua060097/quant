#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断市场乘子模型加载问题
"""

import pickle
import logging
from pathlib import Path

def diagnose_market_multiplier_issue():
    """诊断市场乘子模型问题"""
    
    print("=" * 60)
    print("🔍 市场乘子模型诊断")
    print("=" * 60)
    
    # 1. 查找最新的模型文件
    results_dir = Path("integrated_stock_selection/results")
    if not results_dir.exists():
        print(f"❌ results目录不存在: {results_dir}")
        return False
    
    model_files = list(results_dir.glob("xgboost_model_*.pkl"))
    if not model_files:
        print("❌ 未找到任何模型文件")
        return False
    
    # 按时间戳排序，获取最新的
    latest_model = sorted(model_files)[-1]
    print(f"📁 最新模型文件: {latest_model}")
    
    try:
        # 2. 加载模型文件
        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ 模型文件加载成功")
        print(f"📋 模型包含的字段: {list(model_data.keys())}")
        
        # 3. 检查market_multiplier字段
        market_multiplier = model_data.get('market_multiplier')
        
        if market_multiplier is None:
            print("\n❌ 问题诊断: market_multiplier字段为None")
            print("可能的原因:")
            print("  1. 训练时市场乘子模型训练失败")
            print("  2. 模型保存逻辑有问题")
            print("  3. 使用的是较旧的模型文件")
            
            # 检查是否有其他相关字段
            relevant_fields = ['scaler', 'feature_names', 'pca_info']
            for field in relevant_fields:
                if field in model_data:
                    print(f"  ✅ 找到字段: {field}")
                else:
                    print(f"  ❌ 缺失字段: {field}")
            return False
        
        print("\n✅ 找到market_multiplier字段")
        print(f"类型: {type(market_multiplier)}")
        
        if isinstance(market_multiplier, dict):
            print(f"组件: {list(market_multiplier.keys())}")
            
            # 4. 详细检查各个组件
            print("\n📊 组件详细检查:")
            
            # 检查fallback状态
            fallback = market_multiplier.get('fallback', True)
            print(f"  fallback状态: {fallback}")
            
            if fallback:
                print("  ⚠️ 市场乘子模型处于fallback状态（无效）")
                print("  这说明训练时市场乘子模型训练失败")
            else:
                print("  ✅ 市场乘子模型状态正常（有效）")
            
            # 检查核心组件
            required_components = ['scaler', 'pca', 'ridge']
            all_components_ok = True
            
            for comp in required_components:
                comp_value = market_multiplier.get(comp)
                if comp_value is not None:
                    print(f"  ✅ {comp}: {type(comp_value).__name__}")
                else:
                    print(f"  ❌ {comp}: None")
                    all_components_ok = False
            
            # 检查feature_names
            feature_names = market_multiplier.get('feature_names', [])
            print(f"  📋 市场因子数量: {len(feature_names)}")
            if len(feature_names) > 0:
                print(f"  📋 市场因子: {feature_names[:3]}{'...' if len(feature_names) > 3 else ''}")
            
            # 检查clip范围
            clip_range = market_multiplier.get('clip', None)
            print(f"  📊 Winsorize范围: {clip_range}")
            
            # 5. 综合诊断结果
            print("\n" + "=" * 60)
            print("📊 诊断结果总结:")
            print("=" * 60)
            
            if fallback:
                print("❌ 主要问题: 市场乘子模型处于fallback状态")
                print("解决方案:")
                print("  1. 重新训练模型，确保市场乘子训练成功")
                print("  2. 检查市场状态因子数据是否完整")
                print("  3. 检查验证集数据是否足够")
                return False
            elif not all_components_ok:
                print("❌ 主要问题: 市场乘子模型组件缺失")
                print("解决方案:")
                print("  1. 检查_fit_market_multiplier函数是否正常运行")
                print("  2. 重新训练模型")
                return False
            else:
                print("✅ 市场乘子模型状态正常")
                print("如果仍然无法加载，请检查:")
                print("  1. 模型加载逻辑是否正确")
                print("  2. 运行时的市场状态因子数据是否可用")
                return True
        else:
            print(f"❌ market_multiplier不是字典类型: {type(market_multiplier)}")
            return False
    
    except Exception as e:
        print(f"❌ 诊断过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_market_multiplier_issue()
    exit(0 if success else 1)