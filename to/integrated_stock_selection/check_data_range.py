#!/usr/bin/env python3
"""
检查数据加载器是否正确加载了目标日期范围的数据
"""

import pandas as pd
import sys
sys.path.append('..')

from data_loader import IntegratedDataLoader

def check_data_range():
    """检查数据加载的日期范围"""

    print("=== 检查数据加载范围 ===")

    data_loader = IntegratedDataLoader()

    # 模拟用户的参数
    start_date = "20251021"
    end_date = "20251216"
    extend_history_days = 120

    # 计算扩展开始日期
    extended_start_date = data_loader._calculate_extended_start_date(
        start_date, end_date, extend_history_days
    )

    print(f"目标日期范围: {start_date} - {end_date}")
    print(f"扩展开始日期: {extended_start_date}")
    print(f"完整数据范围: {extended_start_date} - {end_date}")

    # 加载股票特征数据
    stock_data = data_loader.load_stock_features(extended_start_date, end_date)

    if stock_data is None or stock_data.empty:
        print("无法加载股票数据")
        return

    print(f"\n加载数据统计:")
    print(f"总记录数: {len(stock_data):,}")
    print(f"股票数量: {stock_data['stock_code'].nunique():,}")

    # 检查日期范围
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    actual_start = stock_data['date'].min()
    actual_end = stock_data['date'].max()
    actual_start_str = actual_start.strftime('%Y%m%d')
    actual_end_str = actual_end.strftime('%Y%m%d')

    print(f"\n实际日期范围:")
    print(f"起始日期: {actual_start_str} ({actual_start})")
    print(f"结束日期: {actual_end_str} ({actual_end})")

    # 检查目标日期范围是否被包含
    target_start = pd.to_datetime(start_date)
    target_end = pd.to_datetime(end_date)

    target_data = stock_data[
        (stock_data['date'] >= target_start) & (stock_data['date'] <= target_end)
    ]

    print(f"\n目标日期范围数据:")
    print(f"目标范围记录数: {len(target_data):,}")
    print(f"目标范围日期数: {target_data['date'].nunique()}")

    if len(target_data) == 0:
        print("❌ 严重问题：目标日期范围内没有任何数据！")
        print("这就是为什么因子处理停在2025-10-21的原因")
    else:
        print("✅ 目标日期范围内有数据")

    # 检查具体哪些日期有数据
    print(f"\n目标日期范围分析:")
    target_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    available_dates = target_data['date'].unique()

    print(f"目标期间总交易日数: {len(target_dates)}")
    print(f"实际有数据的交易日数: {len(available_dates)}")

    if len(available_dates) > 0:
        print(f"最早有数据日期: {available_dates.min().strftime('%Y-%m-%d')}")
        print(f"最晚有数据日期: {available_dates.max().strftime('%Y-%m-%d')}")

        if len(available_dates) < len(target_dates):
            print(f"⚠️ 数据不完整：{len(available_dates)}/{len(target_dates)} 天有数据")

    # 检查文件是否存在
    print(f"\n检查关键日期的数据文件:")
    test_dates = ["20251021", "20251101", "20251201", "20251216"]

    for date_str in test_dates:
        file_path = f"../local_data/features/daily/{date_str}.parquet"
        try:
            import os
            if os.path.exists(file_path):
                print(f"✅ {date_str}.parquet 文件存在")
            else:
                print(f"❌ {date_str}.parquet 文件不存在")
        except Exception as e:
            print(f"检查 {date_str}.parquet 时出错: {e}")

if __name__ == "__main__":
    check_data_range()