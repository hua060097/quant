#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Factor Lab中688188.SH的6月数据
"""

import pandas as pd
from pathlib import Path

def check_factor_lab():
    """检查Factor Lab数据"""
    print("=" * 80)
    print("检查Factor Lab中688188.SH的6月数据")
    print("=" * 80)

    factor_file = Path("factor_lab_data/factors.parquet")
    factor_df = pd.read_parquet(factor_file)

    # 筛选688188.SH在6月的数据
    stock_688188 = factor_df[
        (factor_df['stock_code'] == '688188.SH') &
        (factor_df['date'] >= '2025-06-01') &
        (factor_df['date'] <= '2025-06-30')
    ].copy().sort_values('date')

    print(f"\n总记录数: {len(stock_688188)}")

    if len(stock_688188) == 0:
        print("Factor Lab中688188.SH在6月没有数据！")
        return

    print(f"\n6月数据:")
    for _, row in stock_688188.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        print(f"  {date_str}")

    # 检查关键日期
    print(f"\n关键日期检查:")
    for date_str in ['2025-06-06', '2025-06-09']:
        target_date = pd.to_datetime(date_str)
        date_data = stock_688188[stock_688188['date'] == target_date]

        if len(date_data) > 0:
            print(f"  {date_str}: 有数据")
        else:
            print(f"  {date_str}: 无数据 -> 会导致merge后open/close为0.0")

    # 检查是否有open/close字段（确认）
    if 'open' in factor_df.columns or 'close' in factor_df.columns:
        print(f"\nFactor Lab有open/close字段")
    else:
        print(f"\nFactor Lab没有open/close字段（已确认）")

if __name__ == "__main__":
    check_factor_lab()
