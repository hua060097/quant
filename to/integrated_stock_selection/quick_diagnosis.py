#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速诊断：直接检查Factor Lab中688188.SH的open和close数据
"""

import pandas as pd
import numpy as np
from pathlib import Path

def quick_diagnosis():
    """快速诊断"""
    print("=" * 80)
    print("快速诊断688188.SH的Factor Lab数据")
    print("=" * 80)

    factor_file = Path("factor_lab_data/factors.parquet")
    df = pd.read_parquet(factor_file)

    # 检查688188.SH
    stock = df[df['stock_code'] == '688188.SH'].copy().sort_values('date')
    print(f"\n总记录数: {len(stock)}")
    print(f"日期范围: {stock['date'].min()} ~ {stock['date'].max()}")

    # 检查是否有open和close列
    print(f"\n可用的列: {stock.columns.tolist()[:20]}")

    if 'open' in stock.columns and 'close' in stock.columns:
        # 检查关键日期
        target_dates = ['2025-06-06', '2025-06-09']
        for date_str in target_dates:
            target_date = pd.to_datetime(date_str)
            date_data = stock[stock['date'] == target_date]

            if len(date_data) > 0:
                row = date_data.iloc[0]
                print(f"\n{date_str}:")
                print(f"  open: {row['open']} (isna: {pd.isna(row['open'])})")
                print(f"  close: {row['close']} (isna: {pd.isna(row['close'])})")

                if pd.notna(row['open']) and pd.notna(row['close']):
                    gap = (row['open'] - row['close']) / row['close']
                    print(f"  gap_ratio: {gap:.6f} ({gap:.2%})")
            else:
                print(f"\n{date_str}: 无数据")

        # 检查是否有异常值
        print(f"\n数据质量检查:")
        print(f"  open=0或NaN: {((stock['open'] == 0) | stock['open'].isna()).sum()} / {len(stock)}")
        print(f"  close=0或NaN: {((stock['close'] == 0) | stock['close'].isna()).sum()} / {len(stock)}")

        # 查找2025年6月的数据
        june_data = stock[(stock['date'] >= '2025-06-01') & (stock['date'] <= '2025-06-30')].copy()
        print(f"\n2025年6月数据 ({len(june_data)} 条):")
        for _, row in june_data.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: open={row['open']:.2f}, close={row['close']:.2f}")
    else:
        print("\n[ERROR] Factor Lab中没有open或close列")

if __name__ == "__main__":
    quick_diagnosis()
