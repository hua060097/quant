#!/usr/bin/env python3
"""检查原始日度数据中是否存在1913.573332的异常值"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_extreme_in_daily_data():
    """检查原始日度数据中的极端涨跌幅"""
    print("=== 检查原始日度数据中的极端涨跌幅 ===\n")

    target_value = 1913.573332
    daily_data_path = Path("C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/to/local_data/daily")

    if not daily_data_path.exists():
        print("日度数据目录不存在")
        return

    # 查找所有月度目录
    monthly_dirs = [d for d in daily_data_path.iterdir() if d.is_dir()]
    monthly_dirs.sort()

    print(f"找到 {len(monthly_dirs)} 个月度目录")

    extreme_found = False
    extreme_records = []

    # 检查最近几个月的数据
    for month_dir in monthly_dirs[-6:]:  # 检查最近6个月
        print(f"\n检查 {month_dir.name}...")

        parquet_file = month_dir / "daily_data.parquet"
        if not parquet_file.exists():
            print(f"  文件不存在: {parquet_file}")
            continue

        try:
            # 读取月度数据
            df = pd.read_parquet(parquet_file)
            print(f"  数据形状: {df.shape}")

            if 'pct_chg' not in df.columns:
                print(f"  缺少pct_chg列")
                continue

            # 检查极端值
            extreme_mask = df['pct_chg'] > 1000  # 超过1000%的涨跌幅
            extreme_count = extreme_mask.sum()

            if extreme_count > 0:
                print(f"  找到 {extreme_count} 个超过1000的极端值")
                extreme_data = df[extreme_mask][['stock_code', 'date', 'pct_chg', 'open', 'high', 'low', 'close', 'amount']]
                extreme_records.append(extreme_data)
                extreme_found = True

                # 检查是否包含目标值
                target_mask = abs(df['pct_chg'] - target_value) < 0.001
                target_count = target_mask.sum()
                if target_count > 0:
                    target_data = df[target_mask][['stock_code', 'date', 'pct_chg', 'open', 'high', 'low', 'close', 'amount']]
                    print(f"  *** 找到目标值 {target_value} ***")
                    print(target_data)
            else:
                print(f"  无超过1000的极端值")
                print(f"  pct_chg范围: [{df['pct_chg'].min():.6f}, {df['pct_chg'].max():.6f}]")

        except Exception as e:
            print(f"  读取失败: {e}")

    # 汇总所有极端值
    if extreme_found:
        print(f"\n=== 极端值汇总 ===")
        all_extremes = pd.concat(extreme_records, ignore_index=True)
        print(f"总共找到 {len(all_extremes)} 个极端值:")
        print(all_extremes.sort_values('pct_chg', ascending=False).head(20))

        # 按股票代码统计
        print(f"\n按股票代码统计:")
        stock_counts = all_extremes['stock_code'].value_counts().head(10)
        for stock, count in stock_counts.items():
            print(f"  {stock}: {count} 次")

        # 按日期统计
        print(f"\n按日期统计:")
        date_counts = all_extremes['date'].value_counts().head(10)
        for date, count in date_counts.items():
            print(f"  {date}: {count} 次")
    else:
        print("\n未发现超过1000的极端涨跌幅")

if __name__ == "__main__":
    check_extreme_in_daily_data()