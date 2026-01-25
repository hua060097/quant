#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证300486.SZ在2025-07-31的future_return_3d计算是否正确
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def verify_future_return_calculation(stock_code='300486.SZ', start_date='2025-07-31'):
    """
    验证future_return_3d的计算
    """
    print(f"验证股票 {stock_code} 从 {start_date} 开始的future_return_3d计算...")

    # 计算需要检查的日期（5个交易日后）
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')

    # 查找接下来的交易日期
    dates_to_check = []
    current_dt = start_dt + timedelta(days=1)

    while len(dates_to_check) < 5:  # 找接下来的5个交易日
        if current_dt.weekday() < 5:  # 工作日
            date_str = current_dt.strftime('%Y%m%d')
            file_path = f"../local_data/features/daily/{date_str}.parquet"
            if os.path.exists(file_path):
                dates_to_check.append((current_dt.strftime('%Y-%m-%d'), file_path))
        current_dt += timedelta(days=1)

    print(f"找到接下来的交易日: {dates_to_check}")

    # 获取起始日的数据
    start_file = f"../local_data/features/daily/{start_dt.strftime('%Y%m%d')}.parquet"
    start_df = pd.read_parquet(start_file)
    start_stock = start_df[start_df['stock_code'] == stock_code].iloc[0]
    start_price = start_stock['close']
    recorded_return = start_stock['future_return_3d']

    print(f"\n起始日 {start_date}:")
    print(f"  收盘价: {start_price:.2f}")
    print(f"  记录的future_return_3d: {recorded_return:.6f} ({recorded_return*100:.2f}%)")

    # 查找3个交易日后的价格
    if len(dates_to_check) >= 3:
        third_day_date, third_day_file = dates_to_check[2]  # 第3个交易日
        third_df = pd.read_parquet(third_day_file)
        third_stock = third_df[third_df['stock_code'] == stock_code].iloc[0]
        third_day_price = third_stock['close']

        print(f"\n第3个交易日 {third_day_date}:")
        print(f"  收盘价: {third_day_price:.2f}")

        # 计算实际收益
        actual_return = (third_day_price - start_price) / start_price
        expected_price = start_price * (1 + recorded_return)

        print(f"\n收益验证:")
        print(f"  起始价格: {start_price:.2f}")
        print(f"  第3日价格: {third_day_price:.2f}")
        print(f"  实际收益: {actual_return:.6f} ({actual_return*100:.2f}%)")
        print(f"  记录收益: {recorded_return:.6f} ({recorded_return*100:.2f}%)")
        print(f"  预期价格: {expected_price:.2f}")
        print(f"  差异: {abs(actual_return - recorded_return):.6f}")

        if abs(actual_return - recorded_return) < 0.001:
            print(f"  ✅ 计算正确！")
        else:
            print(f"  ❌ 计算有误！")

        # 查看每日价格变化
        print(f"\n每日价格变化:")
        print(f"  {start_date}: {start_price:.2f}")
        for i, (date_str, file_path) in enumerate(dates_to_check[:3]):
            df = pd.read_parquet(file_path)
            stock = df[df['stock_code'] == stock_code].iloc[0]
            price = stock['close']
            pct_change = (price - start_price) / start_price * 100
            daily_change = stock['pct_chg']
            print(f"  {date_str}: {price:.2f} (累计: {pct_change:+.2f}%, 当日: {daily_change:+.2f}%)")

    else:
        print("❌ 找不到足够的交易日数据进行验证")

if __name__ == "__main__":
    verify_future_return_calculation('300486.SZ', '2025-07-31')