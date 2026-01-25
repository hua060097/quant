#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查688188.SH在Daily Features中的6月数据
"""

import pandas as pd
from pathlib import Path

def check_june_data():
    """检查6月数据"""
    print("=" * 80)
    print("检查688188.SH在Daily Features中的6月数据")
    print("=" * 80)

    features_dir = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily")

    # 检查关键日期
    dates = ['20250606', '20250609']

    for date_str in dates:
        file_path = features_dir / f"{date_str}.parquet"

        print(f"\n[{date_str}]")
        print(f"  文件路径: {file_path}")
        print(f"  文件存在: {file_path.exists()}")

        if file_path.exists():
            df = pd.read_parquet(file_path)
            print(f"  总记录数: {len(df):,}")

            # 检查688188.SH
            stock_688188 = df[df['stock_code'] == '688188.SH']

            if len(stock_688188) > 0:
                row = stock_688188.iloc[0]
                print(f"  688188.SH数据:")
                print(f"    date: {row['date']}")
                print(f"    open: {row['open']} (isna: {pd.isna(row['open'])})")
                print(f"    close: {row['close']} (isna: {pd.isna(row['close'])})")
                print(f"    volume: {row.get('volume', 'N/A')}")
            else:
                print(f"  688188.SH: 无数据")
                print(f"  -> 这会导致merge后open/close被填充为0.0")

if __name__ == "__main__":
    check_june_data()
