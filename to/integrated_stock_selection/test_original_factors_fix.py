#!/usr/bin/env python3
"""
测试原始因子处理修复功能
验证original_factors不会被winsorize和归一化
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'factor_system', 'factor_lab'))

from pipeline import FactorPipeline
from base import FactorContext

def test_original_factors_processing():
    """测试原始因子处理逻辑"""
    print("=" * 60)
    print("测试原始因子处理修复")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=10)
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    # 创建数据框
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'stock_code': stock,
                'date': date,
                'pct_chg': np.random.normal(0, 2)
            })

    df = pd.DataFrame(data)

    # 创建测试因子数据
    # regular_factor: 会有一些极端值，会被winsorize
    # original_factor: rank类型的值，不应该被winsorize
    # extreme_original_factor: 包含极端值的original因子，应该只做Inf处理
    computed = {
        'regular_factor': pd.Series(np.random.normal(0, 1, len(df)), index=df.index),
        'original_factor': pd.Series(np.random.uniform(0, 1, len(df)), index=df.index),  # rank类型
        'extreme_original_factor': pd.Series(np.concatenate([
            np.random.uniform(0, 1, len(df)-2),
            [np.inf, -np.inf]  # 添加极端值
        ]), index=df.index),
    }

    # 配置
    config = {
        'original_factors': ['original_factor', 'extreme_original_factor'],
        'zscore_factors': ['regular_factor'],
        'rankpct_factors': [],
    }

    # 创建pipeline
    pipeline = FactorPipeline()
    logger = type('MockLogger', (), {'info': print, 'debug': print})()

    print(f"原始数据统计:")
    for name, series in computed.items():
        print(f"  {name}:")
        print(f"    范围: [{series.min():.4f}, {series.max():.4f}]")
        print(f"    包含Inf: {np.isinf(series).any()}")
        print(f"    包含NaN: {series.isna().any()}")

    # 执行后处理
    print(f"\n开始后处理...")
    processed = pipeline._apply_postprocessing(computed, df, config, logger)

    print(f"\n处理后数据统计:")
    for name, series in processed.items():
        print(f"  {name}:")
        print(f"    范围: [{series.min():.4f}, {series.max():.4f}]")
        print(f"    包含Inf: {np.isinf(series).any()}")
        print(f"    包含NaN: {series.isna().any()}")

        # 检查是否被修改
        original_series = computed.get(name.replace('_z', '').replace('_rank', ''))
        if original_series is not None:
            if name in computed:  # 原始因子
                if np.isinf(original_series).any() and not np.isinf(series).any():
                    print(f"    ✓ Inf值被正确处理")
                if series.min() >= 0 and series.max() <= 1:
                    print(f"    ✓ 值范围保持合理（未被winsorize）")
            elif name.endswith('_z'):
                print(f"    ✓ 生成z-score归一化因子")

def test_sector_factors():
    """测试sector.py中的因子是否被正确处理"""
    print(f"\n" + "=" * 60)
    print("测试sector因子处理")
    print("=" * 60)

    # 导入sector模块获取因子列表
    try:
        from factor_lab.definitions import sector
        all_sector_factors = []

        # 获取所有以sector_开头的因子
        for attr_name in dir(sector):
            if attr_name.startswith('sector_') and not attr_name.startswith('_'):
                all_sector_factors.append(attr_name)

        print(f"发现sector因子: {all_sector_factors}")

        # 检查这些因子中哪些应该是original_factors
        rank_factors = [f for f in all_sector_factors if 'rank' in f]
        non_rank_factors = [f for f in all_sector_factors if 'rank' not in f]

        print(f"\nRank类因子（应该跳过winsorize）: {rank_factors}")
        print(f"非Rank类因子（正常处理）: {non_rank_factors}")

        # 模拟归一化配置
        original_factors = set(_get_original_factors_from_main())

        print(f"\n当前_get_original_factors()中的因子:")
        original_rank_factors = [f for f in original_factors if 'rank' in f]
        original_non_rank_factors = [f for f in original_factors if 'rank' not in f]
        print(f"  Rank类: {original_rank_factors}")
        print(f"  非Rank类: {original_non_rank_factors}")

        # 检查一致性
        consistency_issues = []
        for factor in rank_factors:
            if factor not in original_factors:
                consistency_issues.append(f"警告: {factor} 是rank因子但不在original_factors中")

        for factor in original_rank_factors:
            if factor not in all_sector_factors:
                consistency_issues.append(f"警告: {factor} 在original_factors中但不存在于sector模块中")

        if consistency_issues:
            print(f"\n一致性检查:")
            for issue in consistency_issues:
                print(f"  - {issue}")
        else:
            print(f"\n一致性检查通过 ✓")

    except ImportError as e:
        print(f"无法导入sector模块: {e}")

def _get_original_factors_from_main():
    """从main.py获取original_factors列表"""
    try:
        # 简单模拟，实际使用时可以导入main.py
        return {
            'overnight_return', 'size_market_cap_interaction', 'rank_market_cap',
            'rank_momentum_20d', 'rank_rsi', 'rank_turnover_rate',
            'rank_volatility_20d', 'alpha_001', 'alpha_002',
            'alpha_017', 'returns_zscore', 'amount_rank',
            'amihud_illiquidity_enhanced'
        }
    except:
        return set()

if __name__ == "__main__":
    print("开始测试原始因子处理修复...")

    try:
        test_original_factors_processing()
        test_sector_factors()

        print(f"\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()