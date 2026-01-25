#!/usr/bin/env python3
"""
检查数据结构：每只股票有多少条记录
"""

import pandas as pd
from pathlib import Path

def check_data_structure():
    """检查每只股票的数据记录数"""

    data_dir = Path("../local_data/features/daily")
    test_date = "20251021"
    file_path = data_dir / f"{test_date}.parquet"

    if not file_path.exists():
        print(f"File not found: {test_date}")
        return

    df = pd.read_parquet(file_path)
    print(f"{test_date}: Total {len(df)} records, {df['stock_code'].nunique()} unique stocks")

    # 检查每只股票的记录数
    stock_record_counts = df['stock_code'].value_counts()
    print(f"\nStock record count distribution:")
    print(f"  Minimum records per stock: {stock_record_counts.min()}")
    print(f"  Maximum records per stock: {stock_record_counts.max()}")
    print(f"  Average records per stock: {stock_record_counts.mean():.2f}")

    # 统计不同记录数的股票数量
    count_distribution = stock_record_counts.value_counts().sort_index()
    print(f"\nRecord count distribution:")
    for count, stock_count in count_distribution.items():
        print(f"  {count} records: {stock_count} stocks")

    # 检查哪些股票有多条记录
    multi_record_stocks = stock_record_counts[stock_record_counts > 1]
    print(f"\nStocks with multiple records: {len(multi_record_stocks)}")
    if len(multi_record_stocks) > 0:
        print("First 10 stocks with multiple records:")
        for stock_code, count in multi_record_stocks.head(10).items():
            print(f"  {stock_code}: {count} records")

    # 查看股票ID格式
    print(f"\nStock ID patterns:")
    stock_codes = df['stock_code'].unique()
    for pattern in ['SZ', 'SH', 'BJ']:
        count = sum(1 for code in stock_codes if pattern in code)
        print(f"  {pattern}: {count} stocks")

if __name__ == "__main__":
    check_data_structure()
