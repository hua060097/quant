#!/usr/bin/env python3
"""
测试因子管道如何处理数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

from data_loader import IntegratedDataLoader
from factor_system.factor_lab.definitions.support import _institutional_activity
from factor_system.factor_lab.base import FactorContext

def test_factor_pipeline():
    """测试因子管道的数据流"""

    print("=== 测试因子管道数据流 ===")

    data_loader = IntegratedDataLoader()

    # 模拟用户场景：增量更新
    start_date = "20251021"
    end_date = "20251216"
    extend_history_days = 120

    extended_start_date = data_loader._calculate_extended_start_date(
        start_date, end_date, extend_history_days
    )

    print(f"扩展开始日期: {extended_start_date}")
    print(f"目标日期范围: {start_date} - {end_date}")

    # 加载股票数据（模拟整个范围）
    stock_data = data_loader.load_stock_features(extended_start_date, end_date)

    if stock_data is None:
        print("无法加载股票数据")
        return

    print(f"加载的数据: {len(stock_data):,} 条记录")
    print(f"股票数量: {stock_data['stock_code'].nunique():,}")
    print(f"日期范围: {stock_data['date'].min()} ~ {stock_data['date'].max()}")

    # 创建目标mask（模拟因子管道中的target_mask）
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    target_mask = (stock_data['date'] >= start_ts) & (stock_data['date'] <= end_ts)

    print(f"目标范围内的记录数: {target_mask.sum():,}")
    print(f"目标范围外的记录数: {(~target_mask).sum():,}")

    # 创建因子上下文
    ctx = FactorContext()
    ctx.extra['target_mask'] = target_mask
    ctx.extra['target_start_date'] = start_ts
    ctx.extra['target_end_date'] = end_ts

    # 测试 institutional_activity 函数
    print("\n=== 测试 institutional_activity 函数 ===")
    try:
        result_series = _institutional_activity(stock_data, ctx)
        print(f"函数返回的Series长度: {len(result_series)}")
        print(f"非NaN值数量: {result_series.notna().sum()}")
        print(f"NaN值数量: {result_series.isna().sum()}")
        print(f"有效率: {result_series.notna().sum() / len(result_series):.4f}")

        # 检查结果分布
        if result_series.notna().sum() > 0:
            valid_results = result_series[result_series.notna()]
            print(f"结果范围: {valid_results.min():.4f} ~ {valid_results.max():.4f}")
            print(f"结果均值: {valid_results.mean():.4f}")

    except Exception as e:
        print(f"调用 institutional_activity 函数失败: {e}")
        import traceback
        traceback.print_exc()

    # 进一步分析：检查数据结构对单只股票的处理
    print("\n=== 分析单只股票的数据结构 ===")
    sample_stock = stock_data['stock_code'].unique()[0]
    stock_sample = stock_data[stock_data['stock_code'] == sample_stock].copy()

    print(f"股票 {sample_stock}:")
    print(f"  总记录数: {len(stock_sample)}")
    print(f"  日期范围: {stock_sample['date'].min()} ~ {stock_sample['date'].max()}")
    print(f"  目标范围内记录数: {(stock_sample['date'] >= start_ts).sum() & (stock_sample['date'] <= end_ts).sum()}")
    print(f"  目标范围外记录数: {len(stock_sample) - ((stock_sample['date'] >= start_ts).sum() & (stock_sample['date'] <= end_ts).sum())}")

    # 检查为什么 institutional_activity_score 失败而 near_support_count_60d 成功
    print("\n=== 对比分析：为什么 near_support_count_60d 成功 ===")

    # 查找 near_support_count_60d 的定义
    from factor_system.factor_lab.definitions.support import _near_support_count_60d

    try:
        near_support_result = _near_support_count_60d(stock_data, ctx)
        print(f"near_support_count_60d 有效率: {near_support_result.notna().sum() / len(near_support_result):.4f}")
    except Exception as e:
        print(f"调用 near_support_count_60d 失败: {e}")

if __name__ == "__main__":
    test_factor_pipeline()