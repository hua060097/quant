#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查688188.SH在2025-06-09~2025-06-11的开盘价和收盘价数据
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def check_stock_data():
    """检查指定股票和日期范围的数据"""

    data_file = Path('../local_data/daily/202511/daily_data.parquet')
    stock_code = '002759.SZ'
    start_date = '2025-11-11'
    end_date = '2025-11-12'

    print('=' * 100)
    print(f'股票数据检查')
    print('=' * 100)
    print(f'数据文件: {data_file}')
    print(f'股票代码: {stock_code}')
    print(f'日期范围: {start_date} ~ {end_date}')
    print('=' * 100)
    print()

    # 检查文件是否存在
    if not data_file.exists():
        print(f'错误: 数据文件不存在: {data_file}')
        return

    try:
        # 读取parquet文件
        df = pd.read_parquet(data_file)

        print(f'文件读取成功，共 {len(df)} 行数据')
        print(f'包含股票数量: {df["stock_code"].nunique()}')
        print()

        # 筛选指定股票
        stock_df = df[df['stock_code'] == stock_code]

        if len(stock_df) == 0:
            print(f'警告: 未找到股票 {stock_code} 的数据')
            print(f'文件中的股票代码示例: {df["stock_code"].unique()[:10]}')
            return

        print(f'找到股票 {stock_code} 的数据: {len(stock_df)} 行')
        print(f'日期范围: {stock_df["date"].min()} ~ {stock_df["date"].max()}')
        print()

        # 转换日期格式
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        # 筛选日期范围
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        filtered_df = stock_df[
            (stock_df['date'] >= start_date_dt) &
            (stock_df['date'] <= end_date_dt)
        ]

        if len(filtered_df) == 0:
            print(f'警告: 在 {start_date} ~ {end_date} 日期范围内未找到数据')
            print(f'该股票可用的日期范围: {stock_df["date"].min()} ~ {stock_df["date"].max()}')
            return

        # 显示结果
        print('=' * 100)
        print(f'找到 {len(filtered_df)} 条记录')
        print('=' * 100)
        print()

        # 选择并格式化显示列
        display_df = filtered_df[['date', 'open', 'close', 'volume', 'amount']].copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        display_df = display_df.sort_values('date')

        print(display_df.to_string(index=False))

        print()
        print('=' * 100)
        print('数据统计')
        print('=' * 100)
        print(f'开盘价范围: {display_df["open"].min():.2f} ~ {display_df["open"].max():.2f}')
        print(f'收盘价范围: {display_df["close"].min():.2f} ~ {display_df["close"].max():.2f}')
        print(f'平均开盘价: {display_df["open"].mean():.2f}')
        print(f'平均收盘价: {display_df["close"].mean():.2f}')

    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_stock_data()
