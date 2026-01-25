#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查找12月数据行数少于20天的股票（可能是月中上市）
"""

import pandas as pd
from pathlib import Path

def find_new_listings():
    """查找新上市股票"""
    print("=" * 80)
    print("查找12月数据少于20天的股票（可能是月中上市）")
    print("=" * 80)

    # 读取12月数据
    daily_file = Path(__file__).parent.parent / "local_data" / "daily" / "202512" / "daily_data.parquet"

    df = pd.read_parquet(daily_file)
    print(f"[OK] 读取数据: {len(df):,} 行")

    # 统计每只股票的数据行数
    stock_counts = df.groupby('stock_code').size().sort_values()

    print(f"\n股票总数: {len(stock_counts)}")
    print(f"数据行数范围: {stock_counts.min()} ~ {stock_counts.max()}")

    # 找出数据行数少于20的股票
    new_listings = stock_counts[stock_counts < 20]

    print(f"\n数据行数少于20的股票数量: {len(new_listings)}")

    if len(new_listings) > 0:
        print(f"\n前10只股票及其数据范围:")
        for i, (stock_code, count) in enumerate(new_listings.head(10).items()):
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('date')

            first_date = stock_df.iloc[0]['date']
            last_date = stock_df.iloc[-1]['date']

            print(f"\n  {i+1}. {stock_code}: {count}天")
            print(f"     首个交易日: {first_date.strftime('%Y-%m-%d')}")
            print(f"     最后交易日: {last_date.strftime('%Y-%m-%d')}")

            # 检查12月1日是否存在
            target_date = pd.to_datetime('2025-12-01')
            has_target = target_date in stock_df['date'].values

            print(f"     12-01数据存在: {has_target}")

            if not has_target:
                print(f"     -> 不会保存到2025-12-01的特征文件")
            else:
                target_record = stock_df[stock_df['date'] == target_date].iloc[0]
                if pd.isna(target_record['open']):
                    print(f"     -> 12-01数据存在但价格为NaN")
                else:
                    print(f"     -> 12-01数据正常，会保存")

        # 选择一只进行详细分析
        if len(new_listings) > 0:
            test_stock = new_listings.index[0]
            print(f"\n{'=' * 80}")
            print(f"详细分析: {test_stock}")
            print(f"{'=' * 80}")

            stock_df = df[df['stock_code'] == test_stock].copy()
            stock_df = stock_df.sort_values('date')

            print(f"\n完整数据:")
            print(stock_df[['date', 'open', 'close', 'volume']].to_string(index=False))

    else:
        print("\n所有股票都有完整的12月数据（>=20天）")


if __name__ == "__main__":
    find_new_listings()
