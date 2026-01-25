#!/usr/bin/env python3
"""
测试智能裁剪功能

Author: Claude Code Assistant
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factor_system.factor_lab.pipeline import IntelligentClipper, FactorPipeline

def create_test_data():
    """创建测试数据"""
    np.random.seed(42)

    # 创建测试因子数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stocks = [f'stock_{i:04d}' for i in range(100)]

    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'date': date,
                'stock_code': stock,
                'close': np.random.normal(100, 20),
                'volume': np.random.normal(1000000, 200000),
                'pct_chg_0d': np.random.normal(0, 0.03),  # 正常分布
                'momentum_reversal': np.random.laplace(0, 1),  # 厚尾分布
                'ma_ratio_5_10': np.random.beta(2, 2) * 2 - 1,  # 有界分布
            })

    df = pd.DataFrame(data)

    # 添加一些极端值
    extreme_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    df.loc[extreme_indices, 'momentum_reversal'] = np.random.uniform(-10, 10, size=len(extreme_indices))

    return df

def test_intelligent_clipper():
    """测试智能裁剪器"""
    print("[TEST] 测试智能裁剪器")
    print("=" * 50)

    # 创建测试数据
    df = create_test_data()

    # 初始化智能裁剪器
    clipper = IntelligentClipper()

    # 测试不同类型的因子
    test_factors = {
        'pct_chg_0d': df['pct_chg_0d'],
        'momentum_reversal': df['momentum_reversal'],
        'ma_ratio_5_10': df['ma_ratio_5_10'],
        'momentum_reversal_intra_z': df['momentum_reversal'],  # 模拟行业内z-score
    }

    for factor_name, factor_data in test_factors.items():
        print(f"\n📊 测试因子: {factor_name}")
        print("-" * 30)

        # 分析分布
        dist_analysis = clipper.analyze_factor_distribution(factor_data)
        if dist_analysis["status"] == "analyzed":
            print(f"  样本数: {dist_analysis['sample_size']}")
            print(f"  均值: {dist_analysis['mean']:.4f}, 标准差: {dist_analysis['std']:.4f}")
            print(f"  偏度: {dist_analysis['skewness']:.4f}, 峰度: {dist_analysis['kurtosis']:.4f}")
            print(f"  |z|>3 极端值占比: {dist_analysis['z_extreme_ratio']:.2%}")
            print(f"  问题: {', '.join(dist_analysis['issues']) if dist_analysis['issues'] else '无'}")

        # 获取裁剪策略
        strategy = clipper.get_optimal_clipping_strategy(factor_name, factor_data)
        print(f"\n🎯 推荐策略: {strategy['strategy']}")
        print(f"  方法: {strategy['method']}")
        print(f"  参数: {strategy['params']}")
        print(f"  原因: {strategy['reason']}")

        # 应用裁剪
        clipped_data, clip_info = clipper.apply_intelligent_clipping(factor_data, factor_name, strategy)
        print(f"\n✂️  裁剪结果:")
        print(f"  原始范围: [{clip_info['original_range'][0]:.4f}, {clip_info['original_range'][1]:.4f}]")
        print(f"  裁剪范围: [{clip_info['clipped_range'][0]:.4f}, {clip_info['clipped_range'][1]:.4f}]")
        print(f"  裁剪比例: {clip_info['clip_ratio']:.2%} ({clip_info['n_clipped']} 个值)")

def test_pipeline_integration():
    """测试与pipeline的集成"""
    print("\n🔧 测试Pipeline集成")
    print("=" * 50)

    # 创建简单的测试数据
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    stocks = ['stock_0001', 'stock_0002', 'stock_0003']

    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'date': date,
                'stock_code': stock,
                'close': np.random.normal(100, 10),
                'volume': np.random.normal(100000, 20000),
                'pct_chg_0d': np.random.normal(0, 0.02),
            })

    df = pd.DataFrame(data)

    # 创建pipeline
    pipeline = FactorPipeline()

    # 测试compute方法（使用简单的因子名称）
    try:
        result = pipeline.compute(
            factor_names=['pct_chg_0d'],
            data=df,
            start_date='2023-01-01',
            end_date='2023-01-10',
            enable_intelligent_clipping=True
        )

        print(f"✅ Pipeline计算成功")
        print(f"  因子数量: {len(result.factors)}")
        print(f"  数据形状: {result.data.shape}")
        if result.clipping_report:
            print(f"  裁剪报告: {len(result.clipping_report)} 个因子")
            for factor, info in result.clipping_report.items():
                print(f"    {factor}: {info.get('strategy', 'N/A')}")

    except Exception as e:
        print(f"❌ Pipeline计算失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 智能裁剪功能测试")
    print("=" * 60)

    # 测试智能裁剪器
    test_intelligent_clipper()

    # 测试Pipeline集成
    test_pipeline_integration()

    print("\n✨ 测试完成!")

if __name__ == "__main__":
    main()