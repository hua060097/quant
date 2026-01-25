#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查缓存文件stock_features_20230601_20251226.parquet中688188.SH的数据
"""

import pandas as pd
from pathlib import Path

def check_cached_data():
    """检查缓存文件中的数据"""
    print("=" * 100)
    print("检查缓存文件 stock_features_20230601_20251226.parquet 中688188.SH的数据")
    print("=" * 100)

    cache_file = Path("cache/stock_features_20230601_20251226.parquet")

    print(f"\n[1] 缓存文件路径:")
    print(f"    {cache_file.absolute()}")
    print(f"    文件存在: {cache_file.exists()}")

    if not cache_file.exists():
        print(f"\n[ERROR] 缓存文件不存在！")
        return

    # 读取缓存文件
    print(f"\n[2] 读取缓存文件...")
    try:
        df = pd.read_parquet(cache_file)
        print(f"    [OK] 成功读取: {len(df):,} 条记录")
        print(f"    列数: {len(df.columns)}")
        print(f"    日期范围: {df['date'].min()} ~ {df['date'].max()}")
    except Exception as e:
        print(f"    [ERROR] 读取失败: {e}")
        return

    # 筛选688188.SH
    print(f"\n[3] 筛选688188.SH数据...")
    stock_688188 = df[df['stock_code'] == '688188.SH'].copy()

    if len(stock_688188) == 0:
        print(f"    [ERROR] 缓存文件中没有688188.SH的数据！")
        print(f"    这说明在生成缓存时（2023-06-01之前），688188.SH可能还未上市")
        return

    print(f"    [OK] 找到688188.SH: {len(stock_688188)} 条记录")
    print(f"    日期范围: {stock_688188['date'].min()} ~ {stock_688188['date'].max()}")

    # 检查关键日期
    print(f"\n[4] 检查关键日期数据...")
    target_dates = ['2025-06-06', '2025-06-09']

    for date_str in target_dates:
        target_date = pd.to_datetime(date_str)
        date_data = stock_688188[stock_688188['date'] == target_date]

        print(f"\n    {date_str} ({'周五' if '06' in date_str else '周一'}):")

        if len(date_data) == 0:
            print(f"      [ERROR] 该日期没有数据！")
            print(f"      -> 这会导致merge后open/close被填充为0.0")
            continue

        row = date_data.iloc[0]

        print(f"      open: {row['open']} (isna: {pd.isna(row['open'])})")
        print(f"      close: {row['close']} (isna: {pd.isna(row['close'])})")
        print(f"      high: {row['high']}")
        print(f"      low: {row['low']}")
        print(f"      volume: {row['volume']}")

    # 计算跳空幅度
    print(f"\n[5] 计算隔夜跳空幅度...")
    date_0606_data = stock_688188[stock_688188['date'] == pd.to_datetime('2025-06-06')]
    date_0609_data = stock_688188[stock_688188['date'] == pd.to_datetime('2025-06-09')]

    if len(date_0606_data) > 0 and len(date_0609_data) > 0:
        close_0606 = date_0606_data['close'].iloc[0]
        open_0609 = date_0609_data['open'].iloc[0]

        print(f"\n    2025-06-06 (周五) 收盘: {close_0606:.6f}")
        print(f"    2025-06-09 (周一) 开盘: {open_0609:.6f}")

        if pd.notna(close_0606) and pd.notna(open_0609) and close_0606 > 0:
            gap_ratio = (open_0609 - close_0606) / close_0606
            print(f"\n    隔夜跳空幅度:")
            print(f"      gap_ratio = ({open_0609:.6f} - {close_0606:.6f}) / {close_0606:.6f}")
            print(f"      gap_ratio = {gap_ratio:.8f}")
            print(f"      gap_ratio = {gap_ratio:.4%}")

            if abs(gap_ratio + 0.2934) < 0.01:  # -29.34%
                print(f"\n    >>> 发现-29.34%的跌幅！")
                print(f"    >>> 缓存文件中的数据已经出现问题！")
            elif abs(gap_ratio + 0.0003) < 0.0001:  # -0.03%
                print(f"\n    >>> 这是正常的-0.03%跳空")
                print(f"    >>> 缓存文件中的数据是正确的")
                print(f"    >>> 问题可能在merge过程或回测代码中")
            else:
                print(f"\n    >>> 跳空幅度异常: {gap_ratio:.4%}")
        else:
            print(f"\n    [ERROR] 无法计算跳空幅度:")
            print(f"      close_0606 isna: {pd.isna(close_0606)}")
            print(f"      open_0609 isna: {pd.isna(open_0609)}")
            print(f"      close_0606 > 0: {close_0606 > 0 if pd.notna(close_0606) else 'N/A'}")
    else:
        print(f"\n    [ERROR] 数据缺失，无法计算跳空幅度")
        if len(date_0606_data) == 0:
            print(f"      2025-06-06数据缺失")
        if len(date_0609_data) == 0:
            print(f"      2025-06-09数据缺失")

    # 检查2025年6月所有数据
    print(f"\n[6] 检查688188.SH在2025年6月的所有数据...")
    june_data = stock_688188[
        (stock_688188['date'] >= '2025-06-01') &
        (stock_688188['date'] <= '2025-06-30')
    ].copy().sort_values('date')

    print(f"    总记录数: {len(june_data)}")

    if len(june_data) > 0:
        print(f"\n    6月所有交易日:")
        for _, row in june_data.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"      {date_str}: open={row['open']:.2f}, close={row['close']:.2f}, volume={row['volume']}")

        # 检查是否有异常值
        print(f"\n    数据质量检查:")
        zero_open = (june_data['open'] == 0) | june_data['open'].isna()
        zero_close = (june_data['close'] == 0) | june_data['close'].isna()
        zero_volume = (june_data['volume'] == 0) | june_data['volume'].isna()

        print(f"      open=0或NaN: {zero_open.sum()} / {len(june_data)}")
        print(f"      close=0或NaN: {zero_close.sum()} / {len(june_data)}")
        print(f"      volume=0或NaN: {zero_volume.sum()} / {len(june_data)}")

        if zero_open.sum() > 0 or zero_close.sum() > 0:
            print(f"\n      [WARNING] 发现异常值!")
            for idx in june_data[zero_open | zero_close].index:
                row = june_data.loc[idx]
                date_str = row['date'].strftime('%Y-%m-%d')
                print(f"        {date_str}: open={row['open']}, close={row['close']}")
    else:
        print(f"    [WARNING] 6月完全没有数据！")

    # 结论
    print(f"\n" + "=" * 100)
    print("[结论]")
    print("=" * 100)

    if len(date_0606_data) > 0 and len(date_0609_data) > 0:
        close_0606 = date_0606_data['close'].iloc[0]
        open_0609 = date_0609_data['open'].iloc[0]

        if pd.notna(close_0606) and pd.notna(open_0609) and close_0606 > 0:
            gap_ratio = (open_0609 - close_0606) / close_0606

            if abs(gap_ratio) < 0.001:  # 正常跳空范围
                print(f"[OK] 缓存文件中688188.SH的数据正常:")
                print(f"     2025-06-06 close: {close_0606:.6f}")
                print(f"     2025-06-09 open: {open_0609:.6f}")
                print(f"     跳空幅度: {gap_ratio:.4%}")
                print(f"\n如果回测日志显示-29.34%，说明:")
                print(f"  1. 问题不在缓存文件")
                print(f"  2. 可能是merge过程中数据被错误处理")
                print(f"  3. 或者回测代码使用了错误的数据源")
            else:
                print(f"[ERROR] 缓存文件中数据异常:")
                print(f"       跳空幅度: {gap_ratio:.4%}")
                print(f"       需要重新生成缓存文件")

if __name__ == "__main__":
    check_cached_data()
