#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查2025-06-08和2025-06-09的688188.SH数据

回测日志：
- buy_date=2025-06-09
- 需要检查前一交易日（应该是2025-06-06，因为08可能是周日）
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def check_gap_data():
    """检查跳空数据"""
    print("=" * 80)
    print("检查688188.SH的跳空数据（2025-06-06 -> 2025-06-09）")
    print("=" * 80)

    features_dir = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily")

    # 检查2025-06-08（周日）和2025-06-09（周一）
    dates_to_check = {
        '20250606': '2025-06-06 (周五)',
        '20250608': '2025-06-08 (周日)',
        '20250609': '2025-06-09 (周一)'
    }

    data = {}

    for date_str, desc in dates_to_check.items():
        file_path = features_dir / f"{date_str}.parquet"
        print(f"\n[{desc}]")
        print(f"  文件存在: {file_path.exists()}")

        if file_path.exists():
            df = pd.read_parquet(file_path)
            stock_688188 = df[df['stock_code'] == '688188.SH']

            if len(stock_688188) > 0:
                row = stock_688188.iloc[0]
                print(f"  688188.SH: open={row['open']:.2f}, close={row['close']:.2f}")
                data[date_str] = {
                    'open': row['open'],
                    'close': row['close'],
                    'exists': True
                }
            else:
                print(f"  688188.SH: 无数据")
                data[date_str] = {'exists': False}
        else:
            print(f"  文件不存在")
            data[date_str] = {'exists': False}

    # 计算gap_ratio
    print("\n" + "=" * 80)
    print("计算跳空幅度")
    print("=" * 80)

    # 2025-06-08是周日，不是交易日，所以前一交易日是2025-06-06
    prev_date = '20250606'
    curr_date = '20250609'

    if data.get(prev_date, {}).get('exists') and data.get(curr_date, {}).get('exists'):
        prev_close = data[prev_date]['close']
        curr_open = data[curr_date]['open']

        gap_ratio = (curr_open - prev_close) / prev_close

        print(f"\n前一交易日 (2025-06-06, 周五):")
        print(f"  close: {prev_close:.6f}")
        print(f"\n买入日 (2025-06-09, 周一):")
        print(f"  open: {curr_open:.6f}")
        print(f"\n跳空幅度:")
        print(f"  gap_ratio = ({curr_open:.6f} - {prev_close:.6f}) / {prev_close:.6f}")
        print(f"  gap_ratio = {gap_ratio:.8f}")
        print(f"  gap_ratio = {gap_ratio:.4%}")

        if abs(gap_ratio + 0.2934) < 0.01:  # -29.34%
            print(f"\n>>> 发现-29.34%的跌幅！")
        elif abs(gap_ratio + 0.0003) < 0.0001:  # -0.03%
            print(f"\n>>> 这是正常的-0.03%跳空，不是-29.34%")
            print(f">>> 说明回测时的数据已经修复或更新")
    else:
        print(f"\n数据缺失，无法计算")

if __name__ == "__main__":
    check_gap_data()
