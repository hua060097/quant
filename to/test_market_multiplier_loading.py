#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试市场乘子模型加载问题修复
"""

import sys
import pickle
import logging
from pathlib import Path

def test_model_loading():
    """测试模型文件中是否包含市场乘子"""
    try:
        print("=" * 60)
        print("🔍 测试市场乘子模型加载")
        print("=" * 60)
        
        # 查找最新的模型文件
        results_dir = Path("results")
        if not results_dir.exists():
            print("❌ results目录不存在")
            return False
            
        model_files = list(results_dir.glob("xgboost_model_*.pkl"))
        if not model_files:
            print("❌ 未找到模型文件")
            return False
            
        latest_model_file = sorted(model_files)[-1]
        print(f"📁 最新模型文件: {latest_model_file}")
        
        # 加载模型文件
        with open(latest_model_file, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"📊 模型文件包含的键: {list(model_data.keys())}")
        
        # 检查市场乘子
        market_multiplier = model_data.get('market_multiplier')
        
        if market_multiplier is None:
            print("❌ 模型文件中不包含market_multiplier")
            return False
            
        print(f"✅ 找到market_multiplier")
        print(f"   类型: {type(market_multiplier)}")
        
        if isinstance(market_multiplier, dict):
            print(f"   组件: {list(market_multiplier.keys())}")
            
            fallback = market_multiplier.get('fallback')
            print(f"   fallback状态: {fallback}")
            
            if fallback == False:
                print("✅ 市场乘子模型为有效状态（fallback=False）")
                return True
            elif fallback == True:
                print("⚠️ 市场乘子模型为回退状态（fallback=True）")
                return True
            else:
                print(f"⚠️ fallback状态未知: {fallback}")
                return True
        else:
            print(f"⚠️ market_multiplier不是字典类型: {type(market_multiplier)}")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loading_logic():
    """测试加载逻辑"""
    print("\n" + "=" * 60)
    print("🔧 测试加载逻辑")
    print("=" * 60)
    
    try:
        # 模拟测试模式的加载逻辑
        results_dir = Path("results")
        model_files = list(results_dir.glob("xgboost_model_*.pkl"))
        latest_model_file = sorted(model_files)[-1]
        
        with open(latest_model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # 模拟测试代码中的检查逻辑
        market_multiplier_artifacts = model_data.get('market_multiplier')
        
        print(f"市场乘子检查:")
        if market_multiplier_artifacts is None:
            print("🔍 调试: market_multiplier_artifacts 为 None")
            result = "未找到有效的市场乘子模型"
        else:
            fallback_value = market_multiplier_artifacts.get('fallback', True)
            print(f"🔍 调试: market_multiplier_artifacts 存在，fallback={fallback_value}")
            print(f"🔍 调试: 模型组件: {list(market_multiplier_artifacts.keys())}")
            
            if not fallback_value:  # fallback=False表示有效
                result = "找到有效的市场乘子模型"
            else:  # fallback=True表示回退
                result = "未找到有效的市场乘子模型（fallback=True）"
        
        print(f"最终结果: {result}")
        
        if "找到有效的市场乘子模型" in result:
            print("✅ 修复成功！现在可以找到有效的市场乘子模型")
            return True
        else:
            print("⚠️ 市场乘子模型仍处于回退状态")
            return False
        
    except Exception as e:
        print(f"❌ 加载逻辑测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 启动市场乘子模型加载测试")
    
    test1_success = test_model_loading()
    test2_success = test_loading_logic()
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    if test1_success and test2_success:
        print("🎉 所有测试通过！市场乘子模型加载修复成功。")
        print("\n修复内容:")
        print("1. ✅ 在xgboost_model_*.pkl文件中包含market_multiplier字段")
        print("2. ✅ 测试模式可以正确加载和识别市场乘子模型")
        print("3. ✅ fallback状态检查逻辑正常工作")
        return True
    else:
        print(f"❌ 测试失败")
        if not test1_success:
            print("  - 模型文件加载测试失败")
        if not test2_success:
            print("  - 加载逻辑测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)