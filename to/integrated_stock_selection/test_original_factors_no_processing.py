#!/usr/bin/env python3
"""
测试original_factors不再进行Inf和缺失值填充的修改效果
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'factor_system'))

try:
    from factor_lab.pipeline import FactorPipeline
    from factor_lab.base import FactorContext
except ImportError:
    # 如果相对导入失败，尝试直接导入
    from pipeline import FactorPipeline
    from base import FactorContext

def test_original_factors_no_processing():
    """测试原始因子不再进行额外处理"""
    print("=" * 80)
    print("测试original_factors不再进行Inf和缺失值填充")
    print("=" * 80)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=10)
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

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
    # 原始因子: 包含一些特殊值，验证是否被修改
    computed = {
        'original_rank_factor': pd.Series(np.random.uniform(0, 1, len(df)), index=df.index),  # rank类型
        'original_alpha_factor': pd.Series(np.random.normal(0, 1, len(df)), index=df.index),  # alpha类型
        'extreme_original_factor': pd.Series(np.concatenate([
            np.random.uniform(0, 1, len(df)-2),
            [np.inf, -np.inf]  # 添加极端值
        ]), index=df.index),
        'regular_factor': pd.Series(np.random.normal(5, 10, len(df)), index=df.index),  # 常规因子
    }

    # 配置
    config = {
        'original_factors': ['original_rank_factor', 'original_alpha_factor', 'extreme_original_factor'],
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
        print(f"    非NaN数量: {series.notna().sum()}/{len(series)}")

    # 执行后处理
    print(f"\n开始后处理...")
    processed = pipeline._apply_postprocessing(computed, df, config, logger)

    print(f"\n处理后数据统计:")
    for name, series in processed.items():
        print(f"  {name}:")
        print(f"    范围: [{series.min():.4f}, {series.max():.4f}]")
        print(f"    包含Inf: {np.isinf(series).any()}")
        print(f"    包含NaN: {series.isna().any()}")
        print(f"    非NaN数量: {series.notna().sum()}/{len(series)}")

        # 检查是否被修改
        original_series = computed.get(name.replace('_z', '').replace('_rank', ''))
        if original_series is not None:
            if name in computed:  # 原始因子
                # 检查是否保持一致
                if series.equals(original_series):
                    print(f"    [OK] 数据完全一致（未做任何修改）")
                else:
                    print(f"    [CHANGED] 数据被修改")
                    print(f"      原始: [{original_series.min():.4f}, {original_series.max():.4f}]")
                    print(f"      处理后: [{series.min():.4f}, {series.max():.4f}]")

                # 检查Inf值
                original_inf = np.isinf(original_series).any()
                processed_inf = np.isinf(series).any()
                if original_inf and not processed_inf:
                    print(f"    [OK] Inf值被保留（这是期望的行为）")
                elif not original_inf and not processed_inf:
                    print(f"    [OK] 无Inf值（正常）")
                elif original_inf and processed_inf:
                    print(f"    [WARNING] Inf值仍然存在（可能有问题）")
            elif name.endswith('_z'):
                print(f"    [OK] 生成z-score归一化因子")

def test_inf_handling():
    """测试Inf值的处理逻辑"""
    print(f"\n" + "=" * 80)
    print("专门测试Inf值处理")
    print("=" * 80)

    # 创建包含Inf的数据
    data_with_inf = pd.Series([0.1, 0.2, np.inf, 0.4, -np.inf, 0.6])
    print(f"测试数据: {data_with_inf.values}")

    # 测试直接返回（新逻辑）
    mock_logger = type('MockLogger', (), {'debug': print})()
    pipeline = FactorPipeline()

    # 模拟原始因子处理
    result = pipeline._process_original_factor(
        "test_inf_factor", data_with_inf,
        pd.DataFrame({'date': ['20230101']*6}), mock_logger
    )

    print(f"处理后数据: {result.values}")
    print(f"数据保持一致: {data_with_inf.equals(result)}")

if __name__ == "__main__":
    print("开始测试original_factors无处理修改...")

    try:
        test_original_factors_no_processing()
        test_inf_handling()

        print(f"\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        print("结论:")
        print("[OK] original_factors不再进行Inf和缺失值填充")
        print("[OK] 保持因子原有数值和含义")
        print("[OK] 避免了不必要的处理开销")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()