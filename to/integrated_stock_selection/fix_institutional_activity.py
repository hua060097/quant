#!/usr/bin/env python3
"""
修复institutional_activity_score因子的正确方案
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from factor_system.factor_lab.base import FactorContext

def _institutional_activity_fixed(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    修复版本的institutional_activity函数

    关键改进：
    1. 从ctx.extra中获取完整的历史数据
    2. 对完整历史数据进行计算
    3. 只返回目标日期范围内的结果
    """

    # 检查是否有完整的历史数据可用
    if 'full_historical_data' in ctx.extra:
        # 使用完整历史数据进行计算
        full_data = ctx.extra['full_historical_data']
    else:
        # 回退到原始逻辑（可能数据不足）
        full_data = data

    sorted_info = _sort_by_stock_date(full_data, ctx)
    sorted_df = sorted_info.data
    result = np.full(len(sorted_df), 50.0, dtype=float)

    for _, idx in sorted_df.groupby('stock_code').indices.items():
        close = sorted_df.loc[idx, 'close'].to_numpy(dtype=float, copy=False)
        volume = sorted_df.loc[idx, 'volume'].to_numpy(dtype=float, copy=False)
        result[idx] = _institutional_activity_single(close, volume)

    # 关键：只对目标日期应用结果，其他日期设为NaN
    result = np.where(sorted_info.target_mask, result, np.nan)
    return _align_back(result, sorted_info)


def test_fix_approach():
    """测试修复方案"""

    print("=== 测试institutional_activity修复方案 ===")

    # 模拟场景
    from data_loader import IntegratedDataLoader

    data_loader = IntegratedDataLoader()
    start_date = "20251021"
    end_date = "20251216"
    extend_history_days = 120

    extended_start_date = data_loader._calculate_extended_start_date(
        start_date, end_date, extend_history_days
    )

    # 加载完整数据
    full_data = data_loader.load_stock_features(extended_start_date, end_date)

    if full_data is None:
        print("无法加载数据")
        return

    print(f"完整数据: {len(full_data):,} 条记录")
    print(f"股票数量: {full_data['stock_code'].nunique():,}")

    # 创建目标mask
    full_data['date'] = pd.to_datetime(full_data['date'])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    target_mask = (full_data['date'] >= start_ts) & (full_data['date'] <= end_ts)

    print(f"目标范围内记录: {target_mask.sum():,}")

    # 测试原始版本
    print("\n=== 原始版本测试 ===")
    try:
        ctx_original = FactorContext()
        ctx_original.extra['target_mask'] = target_mask

        from factor_system.factor_lab.definitions.support import _institutional_activity
        result_original = _institutional_activity(full_data, ctx_original)
        original_valid_rate = result_original.notna().sum() / len(result_original)
        print(f"原始版本有效率: {original_valid_rate:.4f}")

    except Exception as e:
        print(f"原始版本失败: {e}")

    # 建议的解决方案：在因子管道层面传递完整数据
    print("\n=== 建议的解决方案 ===")
    print("修改factor_system/factor_lab/pipeline.py中的compute方法：")
    print("1. 在line 525之后，添加完整历史数据到context：")
    print("   extra['full_historical_data'] = working.copy()")
    print("2. 时间序列因子检查'full_historical_data'并使用完整数据计算")
    print("3. 保持target_mask过滤逻辑不变，确保只保存目标日期数据")


if __name__ == "__main__":
    test_fix_approach()