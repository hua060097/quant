#!/usr/bin/env python3
"""简化检查新股过滤对异常值的影响"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simple_new_stock_debug():
    """简化检查新股过滤影响"""
    print("=== 检查新股过滤对异常值的影响 ===\n")

    # 1. 基本信息
    ipo_date = datetime(2024, 10, 11)  # 上市日期
    target_date = datetime(2024, 11, 15)  # 目标日期
    prev_date = datetime(2024, 11, 14)   # 前一日

    days_since_ipo = (target_date - ipo_date).days
    print(f"上市日期: {ipo_date.strftime('%Y-%m-%d')}")
    print(f"目标日期: {target_date.strftime('%Y-%m-%d')}")
    print(f"前一日: {prev_date.strftime('%Y-%m-%d')}")
    print(f"交易天数: {days_since_ipo}")

    # 2. 新股过滤判断
    new_stock_threshold = 25
    print(f"新股过滤标准: 交易天数 < {new_stock_threshold}")

    if days_since_ipo < new_stock_threshold:
        print(f"结果: 001279.SZ会被新股过滤掉 (交易天数: {days_since_ipo})")
    else:
        print(f"结果: 001279.SZ不会被新股过滤")

    # 3. 检查关键日期的数据
    print(f"\n检查关键数据:")

    # 检查2024-11-14数据
    try:
        df_20241114 = pd.read_parquet("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/20241114.parquet")
        stock_001279 = df_20241114[df_20241114['stock_code'] == '001279.SZ']

        if len(stock_001279) > 0:
            info = stock_001279.iloc[0]
            print(f"2024-11-14 001279.SZ: pct_chg = {info['pct_chg']:.6f}")
        else:
            print("2024-11-14: 未找到001279.SZ数据")
    except Exception as e:
        print(f"2024-11-14: 读取失败 - {e}")

    # 检查2024-11-15数据
    try:
        df_20241115 = pd.read_parquet("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/20241115.parquet")
        stock_001279_15 = df_20241115[df_20241115['stock_code'] == '001279.SZ']

        if len(stock_001279_15) > 0:
            info = stock_001279_15.iloc[0]
            print(f"2024-11-15 001279.SZ: pct_chg = {info['pct_chg']:.6f}")
        else:
            print("2024-11-15: 未找到001279.SZ数据")
    except Exception as e:
        print(f"2024-11-15: 读取失败 - {e}")

    # 4. 检查时间序列中的异常值
    print(f"\n检查001279.SZ时间序列异常值:")

    # 检查多个日期文件
    dates_to_check = []
    for i in range(30):  # 上市前后的数据
        check_date = ipo_date + timedelta(days=i)
        date_str = check_date.strftime('%Y%m%d')
        dates_to_check.append(date_str)

    all_data = []
    for date_str in dates_to_check:
        try:
            df = pd.read_parquet(f"C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/{date_str}.parquet")
            stock_data = df[df['stock_code'] == '001279.SZ']
            if len(stock_data) > 0:
                stock_data_copy = stock_data.copy()
                stock_data_copy['date_str'] = date_str
                all_data.append(stock_data_copy)
        except:
            continue

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('date_str')

        print(f"001279.SZ时间序列数据 ({len(combined_df)}条记录):")
        extreme_count = 0

        for _, row in combined_df.iterrows():
            pct_chg = row.get('pct_chg', 'N/A')
            date_str = row['date_str']

            if pct_chg != 'N/A':
                if abs(pct_chg) > 100:
                    extreme_count += 1
                    print(f"  {date_str}: {pct_chg:.6f} [异常值!]")

                    # 计算shift影响
                    current_date = datetime.strptime(date_str, '%Y%m%d')
                    shift_target = current_date + timedelta(days=1)
                    print(f"    -> 会被shift到: {shift_target.strftime('%Y-%m-%d')}")
                else:
                    print(f"  {date_str}: {pct_chg:.6f}")

        print(f"\n异常值统计: 发现 {extreme_count} 个异常涨跌幅")

    # 5. 结论
    print(f"\n分析结论:")
    print(f"1. 001279.SZ上市日期: 2024-10-11")
    print(f"2. 到2024-11-15交易天数: {days_since_ipo} 天")
    print(f"3. 新股过滤阈值: {new_stock_threshold} 天")

    if days_since_ipo < new_stock_threshold:
        print(f"4. 在2024-11-15会被新股过滤掉")
        print(f"5. shift(1)操作可能会因为数据不完整而产生异常值")
        print(f"6. 建议检查pct_chg数据源的质量和时间对齐逻辑")
    else:
        print(f"4. 在2024-11-15不会被新股过滤")
        print(f"5. 异常值可能来自其他数据质量问题")

if __name__ == "__main__":
    simple_new_stock_debug()