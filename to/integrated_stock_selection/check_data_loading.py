#!/usr/bin/env python3
"""
检查数据加载器如何加载历史数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

from data_loader import IntegratedDataLoader

def check_data_loading():
    """检查数据加载器如何处理历史数据"""

    print("=== 检查数据加载器的行为 ===")

    data_loader = IntegratedDataLoader()

    # 测试日期范围
    start_date = "20251021"
    end_date = "20251216"
    extend_history_days = 120

    # 计算扩展开始日期
    extended_start_date = data_loader._calculate_extended_start_date(
        start_date, end_date, extend_history_days
    )

    print(f"目标日期范围: {start_date} - {end_date}")
    print(f"扩展历史天数: {extend_history_days}")
    print(f"扩展开始日期: {extended_start_date}")

    # 加载股票数据
    stock_data = data_loader.load_stock_features(extended_start_date, end_date)

    if stock_data is not None:
        print(f"\n加载的股票数据统计:")
        print(f"总记录数: {len(stock_data):,}")
        print(f"股票数量: {stock_data['stock_code'].nunique():,}")
        print(f"日期范围: {stock_data['date'].min()} ~ {stock_data['date'].max()}")

        # 检查记录数分布
        records_per_stock = stock_data['stock_code'].value_counts()
        print(f"平均每只股票记录数: {records_per_stock.mean():.2f}")
        print(f"中位数记录数: {records_per_stock.median():.2f}")
        print(f"最大记录数: {records_per_stock.max()}")

        # 检查具体某只股票的数据
        sample_stock = stock_data['stock_code'].unique()[0]
        stock_sample = stock_data[stock_data['stock_code'] == sample_stock]

        print(f"\n股票 {sample_stock} 的数据示例:")
        print(f"记录数: {len(stock_sample)}")
        if len(stock_sample) > 0:
            print(f"日期范围: {stock_sample['date'].min()} ~ {stock_sample['date'].max()}")
            print(f"前5条记录的日期:")
            for i, row in stock_sample.head().iterrows():
                print(f"  {row['date']}")

        # 检查数据结构是否按股票-日期排序
        print(f"\n数据排序检查:")
        print(f"是否按股票代码排序: {stock_data['stock_code'].is_monotonic_increasing}")
        print(f"是否按日期排序: {stock_data['date'].is_monotonic_increasing}")

        # 检查数据是否正确组织
        print(f"\n数据组织方式检查:")
        sample_stocks = stock_data['stock_code'].unique()[:3]
        for stock in sample_stocks:
            stock_data_subset = stock_data[stock_data['stock_code'] == stock]
            print(f"股票 {stock}: {len(stock_data_subset)}条记录")
            if len(stock_data_subset) > 1:
                dates = stock_data_subset['date'].dt.strftime('%Y-%m-%d').tolist()
                print(f"  日期序列: {dates[:5]}{'...' if len(dates) > 5 else ''}")

    else:
        print("无法加载股票数据")

if __name__ == "__main__":
    check_data_loading()