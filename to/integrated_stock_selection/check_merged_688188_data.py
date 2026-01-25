#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟merge过程，检查merge后的数据是否正确
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_merged_data():
    """检查merge后的数据"""
    print("=" * 100)
    print("模拟merge过程，检查merge后688188.SH的数据")
    print("=" * 100)

    # 1. 加载Factor Lab
    print(f"\n[1] 加载Factor Lab数据...")
    factor_file = project_root / "factor_lab_data" / "factors.parquet"

    if not factor_file.exists():
        print(f"    [ERROR] Factor Lab文件不存在")
        return

    factor_df = pd.read_parquet(factor_file)
    print(f"    [OK] 加载完成: {len(factor_df):,} 条记录")
    print(f"    日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")

    # 检查是否有open/close
    has_ohlc = 'open' in factor_df.columns and 'close' in factor_df.columns
    print(f"    是否包含OHLC字段: {has_ohlc}")

    # 2. 筛选688188.SH
    print(f"\n[2] 筛选688188.SH在Factor Lab中的数据...")
    factor_688188 = factor_df[factor_df['stock_code'] == '688188.SH'].copy()
    print(f"    总记录数: {len(factor_688188)}")

    if len(factor_688188) == 0:
        print(f"    [ERROR] Factor Lab中没有688188.SH")
        return

    # 检查关键日期
    target_dates = ['2025-06-06', '2025-06-09']
    for date_str in target_dates:
        target_date = pd.to_datetime(date_str)
        date_data = factor_688188[factor_688188['date'] == target_date]
        print(f"    {date_str}: {len(date_data)} 条记录")

    # 3. 加载Daily Features缓存
    print(f"\n[3] 加载Daily Features缓存...")
    cache_file = project_root / "cache" / "stock_features_20230601_20251226.parquet"

    if not cache_file.exists():
        print(f"    [ERROR] 缓存文件不存在")
        return

    daily_df = pd.read_parquet(cache_file)
    print(f"    [OK] 加载完成: {len(daily_df):,} 条记录")

    # 筛选688188.SH
    print(f"\n[4] 筛选688188.SH在Daily Features中的数据...")
    daily_688188 = daily_df[daily_df['stock_code'] == '688188.SH'].copy()
    print(f"    总记录数: {len(daily_688188)}")

    # 检查关键日期的OHLC数据
    print(f"\n[5] 检查Daily Features中关键日期的OHLC数据:")
    for date_str in target_dates:
        target_date = pd.to_datetime(date_str)
        date_data = daily_688188[daily_688188['date'] == target_date]

        if len(date_data) > 0:
            row = date_data.iloc[0]
            print(f"\n    {date_str}:")
            print(f"      open: {row['open']} (isna: {pd.isna(row['open'])})")
            print(f"      close: {row['close']} (isna: {pd.isna(row['close'])})")
        else:
            print(f"\n    {date_str}: 无数据")

    # 6. 模拟merge操作（关键步骤）
    print(f"\n[6] 模拟merge操作 (how='left')...")

    # 准备merge数据
    factor_merge = factor_688188[['date', 'stock_code']].copy()
    daily_merge = daily_df[['date', 'stock_code', 'open', 'close', 'high', 'low', 'volume']].copy()

    print(f"    merge前Factor Lab: {len(factor_merge)} 条")
    print(f"    merge前Daily Features: {len(daily_merge)} 条")

    # 执行merge
    merged = factor_merge.merge(daily_merge, on=['date', 'stock_code'], how='left')

    print(f"    merge后: {len(merged)} 条")

    # 检查merge后的数据质量
    print(f"\n[7] 检查merge后的数据质量:")
    print(f"    open为NaN或0: {((merged['open'].isna()) | (merged['open'] == 0)).sum()} / {len(merged)}")
    print(f"    close为NaN或0: {((merged['close'].isna()) | (merged['close'] == 0)).sum()} / {len(merged)}")

    # 检查关键日期的merge结果
    print(f"\n[8] 检查关键日期的merge结果:")
    for date_str in target_dates:
        target_date = pd.to_datetime(date_str)
        date_data = merged[merged['date'] == target_date]

        if len(date_data) > 0:
            row = date_data.iloc[0]
            print(f"\n    {date_str}:")
            print(f"      open: {row['open']} (isna: {pd.isna(row['open'])})")
            print(f"      close: {row['close']} (isna: {pd.isna(row['close'])})")
        else:
            print(f"\n    {date_str}: merge后无数据")

    # 9. 计算merge后的跳空幅度
    print(f"\n[9] 计算merge后的跳空幅度:")

    merged_0606 = merged[merged['date'] == pd.to_datetime('2025-06-06')]
    merged_0609 = merged[merged['date'] == pd.to_datetime('2025-06-09')]

    if len(merged_0606) > 0 and len(merged_0609) > 0:
        close_0606 = merged_0606['close'].iloc[0]
        open_0609 = merged_0609['open'].iloc[0]

        print(f"\n    merge后 2025-06-06 close: {close_0606}")
        print(f"    merge后 2025-06-09 open: {open_0609}")

        if pd.notna(close_0606) and pd.notna(open_0609) and close_0606 > 0:
            gap_ratio = (open_0609 - close_0606) / close_0606
            print(f"\n    merge后跳空幅度: {gap_ratio:.6f} ({gap_ratio:.4%})")

            if abs(gap_ratio + 0.2934) < 0.01:
                print(f"\n    >>> 发现-29.34%的异常！")
                print(f"    >>> merge过程导致了数据错误！")
            elif abs(gap_ratio + 0.00003) < 0.00001:
                print(f"\n    >>> merge后数据正常 (-0.003%)")
                print(f"    >>> 问题可能在回测代码的其他部分")
        else:
            print(f"\n    [ERROR] merge后数据异常:")
            print(f"      close_0606 isna: {pd.isna(close_0606)}")
            print(f"      open_0609 isna: {pd.isna(open_0609)}")
            print(f"      close_0606 > 0: {close_0606 > 0 if pd.notna(close_0606) else 'N/A'}")

    # 10. 检查Window 3 test_data范围
    print(f"\n[10] 检查Window 3 test_data范围 (2025-06-05 ~ 2025-12-15):")
    window_3_test = merged[
        (merged['date'] >= pd.to_datetime('2025-06-05')) &
        (merged['date'] <= pd.to_datetime('2025-12-15'))
    ].copy()

    print(f"    test_data记录数: {len(window_3_test)}")

    # 检查数据质量
    invalid_open = ((window_3_test['open'].isna()) | (window_3_test['open'] == 0))
    invalid_close = ((window_3_test['close'].isna()) | (window_3_test['close'] == 0))

    print(f"    open为NaN或0: {invalid_open.sum()} / {len(window_3_test)}")
    print(f"    close为NaN或0: {invalid_close.sum()} / {len(window_3_test)}")

    if invalid_open.sum() > 0 or invalid_close.sum() > 0:
        print(f"\n    [WARNING] test_data中发现异常值!")
        invalid_dates = window_3_test[invalid_open | invalid_close][['date', 'stock_code', 'open', 'close']].head(10)
        print(f"    异常数据示例:")
        for _, row in invalid_dates.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"      {date_str}: open={row['open']}, close={row['close']}")

    # 结论
    print(f"\n" + "=" * 100)
    print("[结论]")
    print("=" * 100)

    if len(merged_0606) > 0 and len(merged_0609) > 0:
        close_0606 = merged_0606['close'].iloc[0]
        open_0609 = merged_0609['open'].iloc[0]

        if pd.notna(close_0606) and pd.notna(open_0609) and close_0606 > 0:
            gap_ratio = (open_0609 - close_0606) / close_0606

            if abs(gap_ratio) < 0.001:
                print(f"[OK] merge后数据正确:")
                print(f"     跳空幅度: {gap_ratio:.4%}")
                print(f"\n如果回测日志显示-29.34%，说明:")
                print(f"  1. 缓存文件数据正确")
                print(f"  2. merge过程正确")
                print(f"  3. 问题在回测引擎(strategy_backtest.py)")
                print(f"     - 可能使用了错误的数据源")
                print(f"     - 或者计算逻辑有bug")
                print(f"     - 或者gap_info_map记录错误")
            else:
                print(f"[ERROR] merge后数据异常:")
                print(f"       跳空幅度: {gap_ratio:.4%}")
                print(f"       问题在merge过程")

if __name__ == "__main__":
    check_merged_data()
