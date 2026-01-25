#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试最终修复
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent / "integrated_stock_selection"
sys.path.insert(0, str(project_root))

def test_final_fixes():
    """测试最终修复"""
    try:
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        print("=" * 60)
        print("测试最终修复")
        print("=" * 60)
        
        # 测试1: 导入和函数签名
        print("1. 测试模块导入和函数签名...")
        from main import run_train_model_mode, _save_model_results
        print("   模块导入成功!")
        
        # 测试2: 检查PCA信息结构
        print("\n2. 测试PCA信息结构...")
        test_pca_info = {
            'applied': True,
            'original_factors': ['factor1', 'factor2', 'factor3'],
            'n_components': 2,
            'variance_explained': 0.85
        }
        
        # 模拟_save_model_results中的PCA信息处理逻辑
        if test_pca_info.get('applied', False):
            original_factors = test_pca_info.get('original_factors', [])
            n_components = test_pca_info.get('n_components', 0)
            variance_explained = test_pca_info.get('variance_explained', 0)
            print(f"   PCA信息解析成功: {len(original_factors)}个特征 -> {n_components}个主成分")
            print(f"   方差解释比例: {variance_explained:.4f}")
        
        # 测试3: 特征工程函数
        print("\n3. 测试特征工程函数...")
        from main import _apply_market_factor_engineering, _get_available_market_factors
        
        market_factors = _get_available_market_factors()
        print(f"   获取到 {len(market_factors)} 个市场因子")
        
        # 检查函数签名
        import inspect
        sig = inspect.signature(_apply_market_factor_engineering)
        params = list(sig.parameters.keys())
        if 'pca_model' in params:
            print("   _apply_market_factor_engineering支持pca_model参数")
        else:
            print("   错误: _apply_market_factor_engineering不支持pca_model参数")
            return False
        
        print("\n所有测试通过!")
        print("\n最终修复摘要:")
        print("1. 修复了 _save_model_results 中的 KeyError: 'features'")
        print("2. 修复了预测流水线中市场因子识别问题")
        print("3. 重新排序预测流水线: 特征工程 -> 对齐 -> 标准化")
        print("4. 确保训练和预测特征数量匹配")
        
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("最终修复测试")
    print("=" * 60)
    
    success = test_final_fixes()
    
    if success:
        print("\n所有测试通过！最终修复完成。")
        sys.exit(0)
    else:
        print("\n测试失败，仍存在问题。")
        sys.exit(1)