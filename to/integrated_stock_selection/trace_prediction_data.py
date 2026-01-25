#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追踪prediction_data在传递给回测引擎时的实际数据

模拟完整的_run_stock_selection_backtest流程，查看prediction_data的内容
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def trace_prediction_data():
    """追踪prediction_data"""
    print("=" * 100)
    print("追踪prediction_data在传递给回测引擎时的实际数据")
    print("=" * 100)

    # 1. 模拟_load_factor_cache_data
    print("\n[1] 模拟_load_factor_cache_data...")
    from data_loader import IntegratedDataLoader

    FACTOR_LAB_STORE_PATH = project_root / "factor_lab_data" / "factors.parquet"

    factor_df = pd.read_parquet(FACTOR_LAB_STORE_PATH)
    factor_df['date'] = pd.to_datetime(factor_df['date'])

    print(f"    Factor Lab: {len(factor_df):,} 条记录")

    # 筛选Window 3的test数据
    window_3_test = factor_df[
        (factor_df['date'] >= pd.to_datetime('2025-06-05')) &
        (factor_df['date'] <= pd.to_datetime('2025-12-15'))
    ].copy()

    print(f"    Window 3 test_data: {len(window_3_test):,} 条记录")

    # 2. 加载Daily Features
    print("\n[2] 加载Daily Features...")
    loader = IntegratedDataLoader()
    stock_start = '20230601'
    stock_end = '20251226'
    stock_data = loader.load_stock_features(stock_start, stock_end)

    print(f"    Daily Features: {len(stock_data):,} 条记录")

    # 3. 执行merge（模拟main.py:1217）
    print("\n[3] 执行merge操作...")
    base_columns = ['open', 'high', 'low', 'close', 'volume', 'amount',
                   'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate',
                   'open_front', 'high_front', 'low_front', 'close_front']

    # 筛选基础列
    base_merge_columns = [
        col for col in base_columns if col in stock_data.columns and col not in {'date', 'stock_code'}
    ]

    if base_merge_columns:
        merge_source = stock_data[['date', 'stock_code'] + base_merge_columns].copy()
    else:
        merge_source = stock_data[['date', 'stock_code']].copy()

    # 执行merge
    factor_df_merged = factor_df.merge(merge_source, on=['date', 'stock_code'], how='left')

    print(f"    merge后: {len(factor_df_merged):,} 条记录")

    # 4. 筛选Window 3 test_data
    print("\n[4] 筛选Window 3 test_data...")
    test_data_merged = factor_df_merged[
        (factor_df_merged['date'] >= pd.to_datetime('2025-06-05')) &
        (factor_df_merged['date'] <= pd.to_datetime('2025-12-15'))
    ].copy()

    print(f"    Window 3 test_data: {len(test_data_merged):,} 条记录")

    # 5. 检查688188.SH的数据
    print("\n[5] 检查688188.SH在test_data中的数据...")
    stock_688188 = test_data_merged[test_data_merged['stock_code'] == '688188.SH'].copy()

    if len(stock_688188) == 0:
        print("    [ERROR] 688188.SH不在test_data中")
        return

    print(f"    总记录数: {len(stock_688188)}")

    # 检查关键日期
    target_dates = ['2025-06-06', '2025-06-09']
    for date_str in target_dates:
        target_date = pd.to_datetime(date_str)
        date_data = stock_688188[stock_688188['date'] == target_date]

        if len(date_data) > 0:
            row = date_data.iloc[0]
            print(f"\n    {date_str}:")
            print(f"      open: {row['open']} (isna: {pd.isna(row['open'])})")
            print(f"      close: {row['close']} (isna: {pd.isna(row['close'])})")
        else:
            print(f"\n    {date_str}: 无数据")

    # 6. 计算跳空幅度
    print("\n[6] 计算test_data中的跳空幅度...")
    data_0606 = stock_688188[stock_688188['date'] == pd.to_datetime('2025-06-06')]
    data_0609 = stock_688188[stock_688188['date'] == pd.to_datetime('2025-06-09')]

    if len(data_0606) > 0 and len(data_0609) > 0:
        close_0606 = data_0606['close'].iloc[0]
        open_0609 = data_0609['open'].iloc[0]

        print(f"    2025-06-06 close: {close_0606}")
        print(f"    2025-06-09 open: {open_0609}")

        if pd.notna(close_0606) and pd.notna(open_0609) and close_0606 > 0:
            gap_ratio = (open_0609 - close_0606) / close_0606
            print(f"    跳空幅度: {gap_ratio:.6f} ({gap_ratio:.4%})")

            if abs(gap_ratio + 0.2934) < 0.01:
                print(f"\n    [ERROR] test_data中就是-29.34%！")
                print(f"    问题在merge操作或Factor Lab！")
            elif abs(gap_ratio + 0.00003) < 0.00001:
                print(f"\n    [OK] test_data中是正常的-0.003%")
                print(f"    问题在回测引擎！")

    # 7. 模拟strategy_backtest的索引构建逻辑
    print("\n[7] 模拟strategy_backtest的索引构建逻辑...")
    print("    模拟lines 263-277的merge操作...")

    # 准备T日和T+1日数据
    data_t = test_data_merged[test_data_merged['date'] == pd.to_datetime('2025-06-06')][['stock_code', 'close']]
    data_t1 = test_data_merged[test_data_merged['date'] == pd.to_datetime('2025-06-09')][['stock_code', 'open']]

    print(f"    T日(2025-06-06)数据: {len(data_t)} 条")
    print(f"    T+1日(2025-06-09)数据: {len(data_t1)} 条")

    # Inner merge（strategy_backtest.py:263-268）
    merged = pd.merge(
        data_t[['stock_code', 'close']],
        data_t1[['stock_code', 'open']],
        on='stock_code',
        how='inner'
    )

    print(f"    inner merge后: {len(merged)} 条")

    # 检查688188.SH
    merged_688188 = merged[merged['stock_code'] == '688188.SH']

    if len(merged_688188) > 0:
        close_val = merged_688188['close'].iloc[0]
        open_val = merged_688188['open'].iloc[0]

        print(f"\n    688188.SH:")
        print(f"      T日close: {close_val}")
        print(f"      T+1日open: {open_val}")

        # 过滤异常值（line 271）
        filtered = merged[(merged['close'] > 0) & (merged['open'] > 0)]
        print(f"\n    过滤后记录数: {len(filtered)}")

        if len(filtered) > 0 and '688188.SH' in filtered['stock_code'].values:
            # 计算gap_ratio（line 277）
            filtered_688188 = filtered[filtered['stock_code'] == '688188.SH'].iloc[0]
            gap_ratio_calc = (filtered_688188['open'] - filtered_688188['close']) / filtered_688188['close']

            print(f"\n    计算的gap_ratio:")
            print(f"      gap_ratio = ({filtered_688188['open']} - {filtered_688188['close']}) / {filtered_688188['close']}")
            print(f"      gap_ratio = {gap_ratio_calc:.8f}")
            print(f"      gap_ratio = {gap_ratio_calc:.4%}")

            if abs(gap_ratio_calc + 0.2934) < 0.01:
                print(f"\n    >>> 确认是-29.34%！")
                print(f"    >>> test_data中的open/close值异常！")
            elif abs(gap_ratio_calc + 0.00003) < 0.00001:
                print(f"\n    >>> 确认是-0.003%（正常）")
                print(f"    >>> 回测引擎应该看到这个值")
                print(f"    >>> 如果日志显示-29.34%，说明回测引擎使用了不同的数据源")
        else:
            print(f"\n    [WARNING] 688188.SH被过滤掉了")
            print(f"    原因: open或close <= 0")
            print(f"    688188.SH数据: close={close_val}, open={open_val}")
    else:
        print(f"\n    [WARNING] 688188.SH不在inner merge结果中")
        print(f"    可能原因: T日或T+1日数据缺失")

    # 8. 检查test_data中的所有数据质量
    print("\n[8] 检查test_data中的数据质量...")
    invalid_open = ((test_data_merged['open'].isna()) | (test_data_merged['open'] == 0))
    invalid_close = ((test_data_merged['close'].isna()) | (test_data_merged['close'] == 0))

    print(f"    open为NaN或0: {invalid_open.sum()} / {len(test_data_merged)}")
    print(f"    close为NaN或0: {invalid_close.sum()} / {len(test_data_merged)}")

    if invalid_open.sum() > 0 or invalid_close.sum() > 0:
        print(f"\n    发现异常值:")
        invalid_stocks = test_data_merged[invalid_open | invalid_close][['date', 'stock_code', 'open', 'close']].head(20)
        for _, row in invalid_stocks.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"      {date_str} {row['stock_code']}: open={row['open']}, close={row['close']}")

        # 检查2025-06-09的异常数据
        print(f"\n    检查2025-06-09的异常数据...")
        date_0609_invalid = test_data_merged[
            (test_data_merged['date'] == pd.to_datetime('2025-06-09')) &
            (invalid_open | invalid_close)
        ]

        if len(date_0609_invalid) > 0:
            print(f"      2025-06-09有{len(date_0609_invalid)}只股票数据异常")
            for _, row in date_0609_invalid.head(10).iterrows():
                print(f"        {row['stock_code']}: open={row['open']}, close={row['close']}")
        else:
            print(f"      2025-06-09没有异常数据")

if __name__ == "__main__":
    trace_prediction_data()
