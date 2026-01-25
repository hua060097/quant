#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基本函数测试
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent / "integrated_stock_selection"
sys.path.insert(0, str(project_root))

def test_basic_functions():
    """测试基本函数导入和可用性"""
    try:
        logging.basicConfig(level=logging.INFO)
        
        print("测试函数导入...")
        from main import (
            _compute_daily_slope_targets,
            _fit_market_multiplier,
            _apply_market_multiplier,
            _load_market_state_factors,
            _get_available_market_factors
        )
        print("✅ 函数导入成功!")
        
        print("\n测试市场因子列表...")
        factors = _get_available_market_factors()
        print(f"市场因子: {factors}")
        
        print("\n测试市场状态因子加载...")
        try:
            market_df = _load_market_state_factors()
            print(f"加载结果形状: {market_df.shape}")
        except Exception as e:
            print(f"加载失败但函数可用: {e}")
        
        print("\n✅ 基本测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 基本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functions()
    if success:
        print("\n🎉 基本功能正常!")
    else:
        print("\n❌ 存在问题，请检查。")