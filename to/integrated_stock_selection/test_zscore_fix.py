#!/usr/bin/env python3
"""
测试修复后的z-score归一化功能
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.factor_lab.pipeline import FactorPipeline
import logging

def test_zscore_fix():
    """测试修复后的z-score归一化功能"""
    print("=" * 60)
    print("测试修复后的z-score归一化功能")
    print("=" * 60)

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    pipeline = FactorPipeline()

    # 创建测试数据
    print("\n[测试1] 模拟corr_slope_20_120类似数据:")

    # 创建多日的数据，其中一些日期会有相同的值（模拟常数列情况）
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    stocks = ['STOCK_1', 'STOCK_2', 'STOCK_3', 'STOCK_4', 'STOCK_5'] * 10

    # 创建因子值：某些日期所有股票值相同，某些日期有变化
    factor_values = []
    for i, date in enumerate(dates):
        if i in [2, 5, 8]:  # 第3、6、9天所有股票值相同（模拟标准差=0的情况）
            date_values = [0.12345] * 5
        else:
            # 其他日期有不同的值
            date_values = np.random.normal(0, 0.1, 5)
        factor_values.extend(date_values)

    # 创建DataFrame
    test_data = pd.DataFrame({
        'date': [date.strftime('%Y-%m-%d') for date in dates for _ in range(5)],
        'stock_code': stocks,
        'corr_slope_20_120': factor_values
    })

    print(f"测试数据形状: {test_data.shape}")
    print("前几行数据:")
    print(test_data.head(15))

    # 检查是否有常数列日期
    print("\n按日期分组统计:")
    daily_stats = test_data.groupby('date')['corr_slope_20_120'].agg(['mean', 'std', 'nunique'])
    print(daily_stats)

    # 提取因子序列
    factor_series = test_data['corr_slope_20_120']

    # 执行z-score归一化
    print("\n[测试] 执行z-score归一化...")
    try:
        zscore_result = pipeline._zscore_normalize(factor_series, test_data)
        print("✓ z-score归一化成功完成")

        # 检查结果
        print(f"\n结果统计:")
        print(f"  总数量: {len(zscore_result)}")
        print(f"  唯一值数量: {zscore_result.nunique()}")
        print(f"  是否为常数列: {zscore_result.nunique() == 1}")
        print(f"  缺失值数量: {zscore_result.isna().sum()}")
        print(f"  均值: {zscore_result.mean():.8f}")
        print(f"  标准差: {zscore_result.std():.8f}")
        print(f"  范围: [{zscore_result.min():.8f}, {zscore_result.max():.8f}]")

        # 按日期检查结果
        test_data['zscore'] = zscore_result.values
        daily_zscore_stats = test_data.groupby('date')['zscore'].agg(['mean', 'std', 'nunique'])
        print(f"\nz-score后按日期统计:")
        print(daily_zscore_stats)

        # 检查是否还有常数列
        constant_dates = daily_zscore_stats[daily_zscore_stats['nunique'] == 1]
        if not constant_dates.empty:
            print(f"\n警告：仍有{len(constant_dates)}个日期的z-score为常数:")
            print(constant_dates)
        else:
            print(f"\n✓ 成功修复！所有日期的z-score都有变异性")

        # 质量检查
        if zscore_result.nunique() > 10:  # 至少有10个不同值
            print(f"\n✓ 质量检查通过：z-score结果有足够的变异性")
            return True
        else:
            print(f"\n✗ 质量检查失败：z-score结果变异性不足")
            return False

    except Exception as e:
        print(f"✗ z-score归一化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_zscore_fix()
    sys.exit(0 if success else 1)