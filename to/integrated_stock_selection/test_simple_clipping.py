#!/usr/bin/env python3
"""
简化测试智能裁剪功能
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factor_system.factor_lab.pipeline import IntelligentClipper

def test_basic_clipping():
    """测试基本裁剪功能"""
    print("Testing IntelligentClipper...")
    print("=" * 50)

    # 创建测试数据
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    # 添加一些极端值
    data[:50] = np.random.uniform(-8, 8, 50)

    series = pd.Series(data)
    clipper = IntelligentClipper()

    # 测试分布分析
    print("1. Testing distribution analysis...")
    dist_analysis = clipper.analyze_factor_distribution(series)
    print(f"   Sample size: {dist_analysis['sample_size']}")
    print(f"   Mean: {dist_analysis['mean']:.4f}")
    print(f"   Std: {dist_analysis['std']:.4f}")
    print(f"   Skewness: {dist_analysis['skewness']:.4f}")
    print(f"   Kurtosis: {dist_analysis['kurtosis']:.4f}")
    print(f"   |z|>3 extreme ratio: {dist_analysis['z_extreme_ratio']:.2%}")
    print(f"   Issues: {', '.join(dist_analysis['issues']) if dist_analysis['issues'] else 'None'}")

    # 测试策略推荐
    print("\n2. Testing strategy recommendation...")
    strategy = clipper.get_optimal_clipping_strategy("test_factor", series)
    print(f"   Strategy: {strategy['strategy']}")
    print(f"   Method: {strategy['method']}")
    print(f"   Parameters: {strategy['params']}")
    print(f"   Reason: {strategy['reason']}")

    # 测试裁剪应用
    print("\n3. Testing clipping application...")
    clipped_data, clip_info = clipper.apply_intelligent_clipping(series, "test_factor", strategy)
    print(f"   Original range: [{clip_info['original_range'][0]:.4f}, {clip_info['original_range'][1]:.4f}]")
    print(f"   Clipped range: [{clip_info['clipped_range'][0]:.4f}, {clip_info['clipped_range'][1]:.4f}]")
    print(f"   Clip ratio: {clip_info['clip_ratio']:.2%}")
    print(f"   Values clipped: {clip_info['n_clipped']}")

    print("\nTest completed successfully!")
    return True

def test_different_factor_types():
    """测试不同类型因子"""
    print("\nTesting different factor types...")
    print("=" * 50)

    clipper = IntelligentClipper()

    # 创建不同类型的测试数据
    test_cases = {
        'return_momentum': {
            'data': np.random.laplace(0, 1, 500),  # 厚尾分布
            'description': 'Return/Momentum factor (thick-tailed)'
        },
        'intra_sector_zscore': {
            'data': np.random.normal(0, 1, 200),  # 小样本
            'description': 'Intra-sector z-score (small sample)'
        },
        'technical_indicator': {
            'data': np.random.beta(2, 2, 800) * 2 - 1,  # 有界分布
            'description': 'Technical indicator (bounded)'
        },
        'extreme_case': {
            'data': np.concatenate([np.random.normal(0, 1, 400), np.random.uniform(-10, 10, 100)]),
            'description': 'Extreme case with many outliers'
        }
    }

    for factor_name, case in test_cases.items():
        print(f"\nTesting {factor_name} - {case['description']}")
        print("-" * 40)

        series = pd.Series(case['data'])
        strategy = clipper.get_optimal_clipping_strategy(factor_name, series)
        _, clip_info = clipper.apply_intelligent_clipping(series, factor_name, strategy)

        print(f"   Recommended strategy: {strategy['strategy']}")
        print(f"   Method: {strategy['method']}")
        print(f"   Clip ratio: {clip_info['clip_ratio']:.2%}")
        print(f"   Reason: {strategy['reason']}")

if __name__ == "__main__":
    try:
        # 测试基本功能
        test_basic_clipping()

        # 测试不同类型因子
        test_different_factor_types()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()