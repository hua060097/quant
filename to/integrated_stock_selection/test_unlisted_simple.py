#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试：查询一只较新的股票，看看早期日期是否有数据
"""

import pandas as pd
from pathlib import Path

def test_unlisted_stock():
    """测试未上市股票的数据情况"""
    print("=" * 80)
    print("测试：查询新上市股票在早期日期的数据")
    print("=" * 80)

    # 读取12月数据
    daily_file = Path(__file__).parent.parent / "local_data" / "daily" / "202512" / "daily_data.parquet"

    if not daily_file.exists():
        print(f"[ERROR] 文件不存在: {daily_file}")
        return

    df = pd.read_parquet(daily_file)
    print(f"[OK] 读取数据: {len(df):,} 行")

    # 选择一只较新的股票（比如300开头的，通常是创业板）
    test_stocks = ['301001.SZ', '301000.SZ', '300999.SZ', '301678.SZ']

    for stock_code in test_stocks[:2]:  # 只测试前2个
        print(f"\n{'=' * 80}")
        print(f"测试股票: {stock_code}")
        print(f"{'=' * 80}")

        stock_df = df[df['stock_code'] == stock_code].copy()

        if len(stock_df) == 0:
            print(f"  未找到该股票的数据")
            continue

        stock_df = stock_df.sort_values('date')

        print(f"  数据行数: {len(stock_df)}")
        print(f"  最早日期: {stock_df.iloc[0]['date']}")
        print(f"  最晚日期: {stock_df.iloc[-1]['date']}")

        # 检查12月1日是否存在
        target_date = pd.to_datetime('2025-12-01')
        has_target = target_date in stock_df['date'].values

        print(f"\n  2025-12-01是否存在: {has_target}")

        if has_target:
            target_record = stock_df[stock_df['date'] == target_date].iloc[0]
            print(f"    open: {target_record['open']}")
            print(f"    close: {target_record['close']}")
            print(f"    是否为NaN: {pd.isna(target_record['open'])}")
        else:
            print(f"    -> 该股票不会保存到2025-12-01的特征文件中")

        # 显示前5条和后5条数据
        print(f"\n  前5条数据:")
        print(stock_df[['date', 'open', 'close']].head().to_string(index=False))

        print(f"\n  后5条数据:")
        print(stock_df[['date', 'open', 'close']].tail().to_string(index=False))


if __name__ == "__main__":
    test_unlisted_stock()
