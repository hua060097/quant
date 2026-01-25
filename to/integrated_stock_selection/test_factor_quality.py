#!/usr/bin/env python3
"""
测试因子质量检查功能
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.factor_lab.pipeline import _validate_factor_quality
import logging

def test_factor_quality_validation():
    """测试因子质量验证功能"""
    print("=" * 60)
    print("测试因子质量检查功能")
    print("=" * 60)

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 测试1: 正常因子
    print("\n[测试1] 正常因子:")
    normal_factor = pd.Series(np.random.normal(0, 1, 1000))
    try:
        _validate_factor_quality(normal_factor, "normal_factor", logger)
        print("PASS: Normal factor validation passed")
    except Exception as e:
        print(f"FAIL: Normal factor validation failed: {e}")
        return False

    # 测试2: 常数列
    print("\n[测试2] 常数列:")
    constant_factor = pd.Series([0.5] * 1000)
    try:
        _validate_factor_quality(constant_factor, "constant_factor", logger)
        print("FAIL: Constant factor validation should have failed")
        return False
    except ValueError as e:
        print("PASS: Constant factor correctly detected as problematic")
        print(f"  Error info: {str(e)[:100]}...")

    # 测试3: 缺失数据过多
    print("\n[测试3] 缺失数据过多:")
    data = [1, 2, 3] + [np.nan] * 97  # 3%有效数据
    missing_factor = pd.Series(data)
    try:
        _validate_factor_quality(missing_factor, "missing_factor", logger)
        print("FAIL: Missing data validation should have failed")
        return False
    except ValueError as e:
        print("PASS: Missing data correctly detected as problematic")
        print(f"  Error info: {str(e)[:100]}...")

    # 测试4: 有效数据为0
    print("\n[测试4] 完全缺失数据:")
    all_nan_factor = pd.Series([np.nan] * 100)
    try:
        _validate_factor_quality(all_nan_factor, "all_nan_factor", logger)
        print("FAIL: All NaN data validation should have failed")
        return False
    except ValueError as e:
        print("PASS: All NaN data correctly detected as problematic")
        print(f"  Error info: {str(e)[:100]}...")

    print("\n" + "=" * 60)
    print("SUCCESS: All tests passed! Factor quality validation works correctly")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_factor_quality_validation()
    sys.exit(0 if success else 1)