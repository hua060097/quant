#!/usr/bin/env python3
"""
测试因子质量检查修复是否正确
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from data_loader import IntegratedDataLoader
from factor_system.factor_lab.definitions.support import _institutional_activity
from factor_system.factor_lab.base import FactorContext

def test_quality_check_fix():
    """测试质量检查修复的效果"""

    print("=== 测试因子质量检查修复 ===")

    data_loader = IntegratedDataLoader()

    # 模拟用户的场景
    start_date = "20251021"
    end_date = "20251216"
    extend_history_days = 120

    extended_start_date = data_loader._calculate_extended_start_date(
        start_date, end_date, extend_history_days
    )

    print(f"测试场景: {start_date} - {end_date}")
    print(f"扩展开始日期: {extended_start_date}")

    # 加载数据
    stock_data = data_loader.load_stock_features(extended_start_date, end_date)

    if stock_data is None:
        print("无法加载数据")
        return

    print(f"加载数据: {len(stock_data):,} 条记录")
    print(f"股票数量: {stock_data['stock_code'].nunique():,}")

    # 准备数据
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # 创建target_mask
    target_mask = (stock_data['date'] >= start_ts) & (stock_data['date'] <= end_ts)
    print(f"目标范围内记录数: {target_mask.sum():,}")

    # 创建因子上下文
    ctx = FactorContext()
    ctx.extra['target_mask'] = target_mask

    # 测试institutional_activity计算
    print("\n=== 测试institutional_activity计算 ===")
    try:
        result_series = _institutional_activity(stock_data, ctx)

        print(f"计算结果统计:")
        print(f"  总记录数: {len(result_series):,}")
        print(f"  有效结果数: {result_series.notna().sum():,}")
        print(f"  缺失结果数: {result_series.isna().sum():,}")
        print(f"  整体有效率: {result_series.notna().sum() / len(result_series):.4f}")

        # 应用target_mask过滤（模拟修复后的行为）
        filtered_result = result_series[target_mask]
        print(f"\n应用target_mask后的统计:")
        print(f"  目标范围内记录数: {len(filtered_result):,}")
        print(f"  目标范围内有效数: {filtered_result.notna().sum():,}")
        print(f"  目标范围内缺失数: {filtered_result.isna().sum():,}")
        print(f"  目标范围内有效率: {filtered_result.notna().sum() / len(filtered_result):.4f}")

        # 检查质量检查是否会通过
        target_valid_ratio = filtered_result.notna().sum() / len(filtered_result)

        print(f"\n质量检查预期结果:")
        print(f"  检查的数据范围: 只检查目标日期范围 ({len(filtered_result):,} 条记录)")
        print(f"  有效率: {target_valid_ratio:.4f}")

        if target_valid_ratio >= 0.5:
            print("  ✅ 质量检查应该通过")
        elif target_valid_ratio >= 0.8:
            print("  ✅ 质量检查应该通过（无警告）")
        else:
            print(f"  ⚠️  质量检查会发出警告（有效率 < 0.80）")

        if target_valid_ratio < 0.5:
            print(f"  ❌ 质量检查会失败（有效率 < 0.50）")

        # 分析预期的有效率
        expected_per_stock = 0  # 每只股票在目标日期范围内应该有几个有效结果
        if len(filtered_result) > 0:
            # 计算每只股票在目标日期范围内的记录数
            target_data = stock_data[target_mask]
            records_per_stock_in_target = target_data.groupby('stock_code').size()
            print(f"\n目标日期范围内每只股票的记录数分布:")
            print(f"  平均每只股票记录数: {records_per_stock_in_target.mean():.2f}")
            print(f"  预期每只股票有效结果数: ~{expected_per_stock} (institutional_activity需要30天历史才能产生有效结果)")

        print(f"\n修复效果总结:")
        print(f"  修复前: 检查包含历史数据的完整结果，有效率很低")
        print(f"  修复后: 只检查目标日期范围，符合用户预期")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quality_check_fix()