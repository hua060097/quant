#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查股票300486.SZ在2025年7月31日前后的价格数据
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def check_stock_data(stock_code='300486.SZ', target_date='2025-07-31'):
    """
    检查指定股票在目标日期前后的数据
    """
    print(f"检查股票 {stock_code} 在 {target_date} 前后的数据...")

    # 计算需要检查的日期范围（前后10个交易日）
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')

    # 检查前后日期的数据
    dates_to_check = []
    for i in range(-15, 16):  # 前15天到后15天
        check_dt = target_dt + timedelta(days=i)
        # 只检查工作日
        if check_dt.weekday() < 5:  # 0-4是周一到周五
            date_str = check_dt.strftime('%Y%m%d')
            file_path = f"../local_data/features/daily/{date_str}.parquet"
            if os.path.exists(file_path):
                dates_to_check.append((check_dt.strftime('%Y-%m-%d'), file_path))

    print(f"找到 {len(dates_to_check)} 个交易日的数据")

    stock_data = []

    for date_str, file_path in dates_to_check:
        try:
            # 读取数据
            df = pd.read_parquet(file_path)

            # 查找目标股票
            stock_mask = df['stock_code'] == stock_code
            stock_row = df[stock_mask]

            if len(stock_row) > 0:
                row = stock_row.iloc[0]
                stock_data.append({
                    'date': date_str,
                    'stock_code': stock_code,
                    'close': row.get('close', np.nan),
                    'future_return_3d': row.get('future_return_3d', np.nan),
                    'pct_chg': row.get('pct_chg', np.nan),
                    'volume': row.get('volume', np.nan),
                    'amount': row.get('amount', np.nan)
                })

                if date_str == target_date:
                    print(f"\n目标日期 {target_date} 的数据:")
                    print(f"  收盘价: {row.get('close', 'N/A')}")
                    print(f"  future_return_3d: {row.get('future_return_3d', 'N/A')}")
                    print(f"  涨跌幅: {row.get('pct_chg', 'N/A')}%")
                    print(f"  成交量: {row.get('volume', 'N/A')}")
                    print(f"  成交额: {row.get('amount', 'N/A')}")

        except Exception as e:
            print(f"处理 {date_str} 数据时出错: {e}")
            continue

    if stock_data:
        df_stock = pd.DataFrame(stock_data)
        df_stock = df_stock.sort_values('date')

        print(f"\n股票 {stock_code} 在 {target_date} 前后的价格数据:")
        print("=" * 80)

        for _, row in df_stock.iterrows():
            print(f"{row['date']} | 收盘: {row['close']:8.2f} | 涨跌: {row['pct_chg']:6.2f}% | 3日收益: {row['future_return_3d']:8.4f} | 成交量: {row['volume']:10.0f}")

        # 验证77.3%的收益是否合理
        target_row = df_stock[df_stock['date'] == target_date]
        if len(target_row) > 0:
            future_return = target_row.iloc[0]['future_return_3d']
            close_price = target_row.iloc[0]['close']

            if not pd.isna(future_return) and not pd.isna(close_price):
                expected_3d_later_price = close_price * (1 + future_return)
                print(f"\n收益验证:")
                print(f"  {target_date} 收盘价: {close_price:.2f}")
                print(f"  future_return_3d: {future_return:.4f} ({future_return*100:.2f}%)")
                print(f"  预期3日后价格: {expected_3d_later_price:.2f}")

        # 保存详细数据
        output_file = f"stock_{stock_code}_analysis.csv"
        df_stock.to_csv(output_file, index=False)
        print(f"\n详细数据已保存到: {output_file}")

    else:
        print(f"未找到股票 {stock_code} 在指定时间范围内的数据")

if __name__ == "__main__":
    check_stock_data('300486.SZ', '2025-07-31')