#!/usr/bin/env python3
"""
检查股票数据的记录数分布，理解数据结构
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def check_stock_data_structure():
    """检查股票数据的结构"""

    data_dir = Path("../local_data/features/daily")

    # 检查指定日期的数据
    test_dates = ["20251021", "20251022", "20251115", "20251216"]

    for date_str in test_dates:
        file_path = data_dir / f"{date_str}.parquet"

        if not file_path.exists():
            print(f"{date_str}: File not found")
            continue

        try:
            df = pd.read_parquet(file_path)
            print(f"\n=== {date_str} 数据结构分析 ===")
            print(f"总记录数: {len(df):,}")
            print(f"股票数量: {df['stock_code'].nunique():,}")

            # 检查记录数分布
            records_per_stock = df['stock_code'].value_counts()
            print(f"平均每只股票记录数: {records_per_stock.mean():.2f}")
            print(f"中位数记录数: {records_per_stock.median():.2f}")
            print(f"最大记录数: {records_per_stock.max()}")
            print(f"最小记录数: {records_per_stock.min()}")

            # 统计记录数分布
            distribution = records_per_stock.value_counts().sort_index()
            print(f"记录数分布:")
            for record_count, stock_count in distribution.head(10).items():
                print(f"  {record_count}条记录/股票: {stock_count}只股票")

            if len(distribution) > 10:
                print(f"  ... 还有{len(distribution) - 10}种其他记录数")

            # 检查具体几只股票的数据
            print(f"\n前5只股票的记录详情:")
            sample_stocks = df['stock_code'].unique()[:5]

            for stock_code in sample_stocks:
                stock_data = df[df['stock_code'] == stock_code]
                print(f"\n股票 {stock_code}:")
                print(f"  记录数: {len(stock_data)}")
                if len(stock_data) > 0:
                    print(f"  日期范围: {stock_data['date'].min()} ~ {stock_data['date'].max()}")
                    print(f"  有volume数据的记录: {stock_data['volume'].notna().sum()}")
                    print(f"  volume数据类型: {stock_data['volume'].dtype}")

        except Exception as e:
            print(f"Error reading {date_str}: {e}")

if __name__ == "__main__":
    check_stock_data_structure()