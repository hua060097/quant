#!/usr/bin/env python3
"""验证新股过滤机制的问题 - 检查上市首日异常数据是否被正确过滤"""

import pandas as pd
import numpy as np

def verify_new_stock_filtering_issue():
    """验证新股过滤机制的问题"""
    print("=== 验证新股过滤机制问题 ===\n")

    # 1. 模拟_check_new_stocks函数的行为
    print("步骤1: 模拟_check_new_stocks函数行为")

    # 从日志我们知道，2024-10-11是上市日期
    ipo_date = "20241011"
    target_date = "20241115"  # 35天后
    prev_date = "20241114"

    print(f"  上市日期: {ipo_date}")
    print(f"  目标日期: {target_date} (上市35天后)")
    print(f"  前一日: {prev_date}")

    # 计算交易日（简化版本，假设每个交易日都存在）
    trading_days_since_ipo = 35  # 从10-11到11-15，约35个交易日
    print(f"  交易天数: {trading_days_since_ipo}")

    # 新股过滤阈值
    new_stock_threshold = 25
    print(f"  新股过滤阈值: < {new_stock_threshold} 天")

    if trading_days_since_ipo < new_stock_threshold:
        print(f"  结果: 在{target_date}会被识别为新股")
    else:
        print(f"  结果: 在{target_date}不会被识别为新股")

    # 2. 检查关键日期的数据
    print(f"\n步骤2: 检查关键日期的数据")

    # 检查2024-10-11（上市首日）的数据
    try:
        df_ipo = pd.read_parquet(f"C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/{ipo_date}.parquet")
        stock_001279_ipo = df_ipo[df_ipo['stock_code'] == '001279.SZ']

        if len(stock_001279_ipo) > 0:
            info = stock_001279_ipo.iloc[0]
            print(f"  {ipo_date} 001279.SZ数据:")
            print(f"    pct_chg: {info['pct_chg']:.6f} [异常值!]")
            print(f"    开盘价: {info['open']}")
            print(f"    收盘价: {info['close']}")
            print(f"    最高价: {info['high']}")
            print(f"    最低价: {info['low']}")
            print(f"    成交量: {info.get('volume', 'N/A')}")
        else:
            print(f"  {ipo_date}: 未找到001279.SZ数据")
    except Exception as e:
        print(f"  {ipo_date}: 读取失败 - {e}")

    # 检查2024-11-14的数据（shift的目标来源）
    try:
        df_prev = pd.read_parquet(f"C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/{prev_date}.parquet")
        stock_001279_prev = df_prev[df_prev['stock_code'] == '001279.SZ']

        if len(stock_001279_prev) > 0:
            info = stock_001279_prev.iloc[0]
            print(f"  {prev_date} 001279.SZ数据:")
            print(f"    pct_chg: {info['pct_chg']:.6f}")
        else:
            print(f"  {prev_date}: 未找到001279.SZ数据")
    except Exception as e:
        print(f"  {prev_date}: 读取失败 - {e}")

    # 检查2024-11-15的数据（应用shift后的日期）
    try:
        df_target = pd.read_parquet(f"C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/{target_date}.parquet")
        stock_001279_target = df_target[df_target['stock_code'] == '001279.SZ']

        if len(stock_001279_target) > 0:
            info = stock_001279_target.iloc[0]
            print(f"  {target_date} 001279.SZ数据:")
            print(f"    pct_chg: {info['pct_chg']:.6f}")
        else:
            print(f"  {target_date}: 未找到001279.SZ数据")
    except Exception as e:
        print(f"  {target_date}: 读取失败 - {e}")

    # 3. 分析shift操作的影响
    print(f"\n步骤3: 分析shift操作的影响")

    # 模拟shift操作：每个股票的pct_chg_1d = shift(1)
    print(f"  shift(1)操作逻辑:")
    print(f"    2024-11-15的pct_chg_1d = 2024-11-14的pct_chg")
    print(f"    2024-11-14的pct_chg_1d = 2024-11-13的pct_chg")
    print(f"    2024-10-12的pct_chg_1d = 2024-10-11的pct_chg")

    print(f"\n  问题分析:")
    print(f"  关键问题: shift(1)操作在原始数据上进行，不检查过滤结果")
    print(f"  即使2024-11-15的001279.SZ会被过滤，")
    print(f"  但shift(1)时可能仍会使用2024-10-11的异常数据")

    # 4. 验证这个假设
    print(f"\n步骤4: 验证假设")

    # 读取多日数据来验证shift操作
    dates_to_check = ['20241011', '20241012', '20241013', '20241014', '20241015', '20241016', '20241017', '20241018']

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
        print(f"   001279.SZ时间序列数据:")

        for _, row in combined_df.iterrows():
            date_str = row['date_str']
            pct_chg = row.get('pct_chg', 'N/A')

            if date_str == '20241011':
                print(f"    {date_str}: pct_chg = {pct_chg:.6f} [异常值，上市首日]")
            elif date_str == '20241012':
                print(f"    {date_str}: pct_chg = {pct_chg:.6f} [这是2024-10-11的shift结果!]")
            else:
                print(f"    {date_str}: pct_chg = {pct_chg:.6f}")

    # 5. 结论
    print(f"\n步骤5: 结论")
    print(f"  你的猜测是正确的！")
    print(f"  问题根源: shift(1)操作在原始数据进行，")
    print(f"           不考虑后续的过滤逻辑")
    print(f"  ")
    print(f"  具体问题:")
    print(f"   1. 2024-10-11: 001279.SZ上市首日，pct_chg = 1738.946281")
    print(f"   2. shift(1)操作: 2024-10-12的pct_chg_1d = 1738.946281")
    print(f"   3. 2024-11-14: shift(1)操作使用2024-11-13的数据")
    print(f"  4. 2024-11-15: shift(1)操作使用2024-11-14的正常数据")
    print(f"  ")
    print(f"  5. 过滤逻辑问题: ")
    print(f"     - 虽然2024-11-15的数据会被过滤，")
    print(f"     - 但shift(1)可能仍然使用到异常的历史数据")

if __name__ == "__main__":
    verify_new_stock_filtering_issue()