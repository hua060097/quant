#!/usr/bin/env python3
"""
追踪真正的数据流问题
"""

import pandas as pd
import numpy as np
from data_loader import IntegratedDataLoader

def trace_actual_issue():
    """追踪实际的数据流问题"""

    print("=== 重新分析数据流 ===")

    data_loader = IntegratedDataLoader()

    # 模拟当前失败的场景
    start_date = "20251021"
    end_date = "20251216"
    extend_history_days = 120

    extended_start_date = data_loader._calculate_extended_start_date(
        start_date, end_date, extend_history_days
    )

    print(f"1. 数据加载:")
    print(f"   扩展开始日期: {extended_start_date}")
    print(f"   目标日期范围: {start_date} - {end_date}")

    # 加载数据
    stock_data = data_loader.load_stock_features(extended_start_date, end_date)

    if stock_data is None:
        print("   无法加载数据")
        return

    print(f"   加载的记录数: {len(stock_data):,}")
    print(f"   股票数量: {stock_data['stock_code'].nunique():,}")

    # 检查每只股票的记录数
    records_per_stock = stock_data['stock_code'].value_counts()
    print(f"   平均每只股票记录数: {records_per_stock.mean():.1f}")
    print(f"   institutional_activity_score需要30天历史数据: {'足够' if records_per_stock.mean() >= 30 else '不足'}")

    # 检查数据结构
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    # 分析具体一只股票的数据
    sample_stock = stock_data['stock_code'].unique()[0]
    stock_sample = stock_data[stock_data['stock_code'] == sample_stock].copy()

    print(f"\n2. 单只股票数据分析 ({sample_stock}):")
    print(f"   总记录数: {len(stock_sample)}")
    print(f"   日期范围: {stock_sample['date'].min()} ~ {stock_sample['date'].max()}")

    # 分离目标范围和历史数据
    target_mask = (stock_sample['date'] >= start_ts) & (stock_sample['date'] <= end_ts)
    target_records = stock_sample[target_mask]
    historical_records = stock_sample[~target_mask]

    print(f"   目标范围记录: {len(target_records)}")
    print(f"   历史数据记录: {len(historical_records)}")
    print(f"   总数据是否足够30天: {'是' if len(stock_sample) >= 30 else '否'}")

    # 模拟institutional_activity_single的计算
    print(f"\n3. institutional_activity_single计算模拟:")
    if len(stock_sample) >= 30:
        print("   数据长度满足要求，应该能正常计算")
        print("   但为什么实际失败？")

        # 让我看看数据传递的实际结构
        print(f"\n4. 检查因子管道的数据传递:")
        print(f"   working数据形状: {stock_data.shape}")
        print(f"   target_mask的长度: {target_mask.sum() if hasattr(target_mask, 'sum') else len(target_mask)}")

        # 检查target_mask如何应用
        working_target_mask = (stock_data['date'] >= start_ts) & (stock_data['date'] <= end_ts)
        print(f"   working中目标范围记录数: {working_target_mask.sum()}")
        print(f"   working中历史范围记录数: {(~working_target_mask).sum()}")

        # 关键问题：看看_institutional_activity函数实际接收到什么数据
        print(f"\n5. 关键问题分析:")
        print(f"   institutional_activity函数接收的是完整的working数据")
        print(f"   但在line 198: result = np.where(sorted_info.target_mask, result, np.nan)")
        print(f"   这会将所有历史计算结果设为NaN")

        print(f"\n   所以问题不是数据不够，而是:")
        print(f"   1. 历史数据计算成功了")
        print(f"   2. 但被target_mask过滤掉了")
        print(f"   3. 只保留目标日期的结果")

        print(f"\n   这意味着每只股票只有目标日期的1个有效结果")
        print(f"   而不是每只股票所有历史日期都有有效结果")

    else:
        print("   数据长度不足，确实会导致失败")

if __name__ == "__main__":
    trace_actual_issue()