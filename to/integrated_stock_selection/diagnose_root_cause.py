#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断根本原因：检查prediction_data在传递给回测引擎前后的数据一致性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def diagnose_root_cause():
    """诊断根本原因"""
    print("=" * 100)
    print("诊断：prediction_data在传递给回测引擎前后的数据一致性")
    print("=" * 100)

    # 1. 加载并模拟_load_factor_cache_data
    print("\n[步骤1] 加载Factor Lab...")
    factor_file = project_root / "factor_lab_data" / "factors.parquet"
    factor_df = pd.read_parquet(factor_file)
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    print(f"    Factor Lab: {len(factor_df):,} 条记录")

    # 2. 加载Daily Features
    print("\n[步骤2] 加载Daily Features...")
    from data_loader import IntegratedDataLoader
    loader = IntegratedDataLoader()
    stock_data = loader.load_stock_features('20230601', '20251226')
    print(f"    Daily Features: {len(stock_data):,} 条记录")

    # 3. 执行merge（模拟main.py的_load_factor_cache_data）
    print("\n[步骤3] 执行merge...")
    base_columns = ['open', 'high', 'low', 'close', 'volume', 'amount',
                   'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate']

    base_merge_columns = [
        col for col in base_columns if col in stock_data.columns and col not in {'date', 'stock_code'}
    ]

    merge_source = stock_data[['date', 'stock_code'] + base_merge_columns].copy()
    factor_df_merged = factor_df.merge(merge_source, on=['date', 'stock_code'], how='left')

    print(f"    merge后: {len(factor_df_merged):,} 条记录")

    # 4. 筛选Window 3 test_data（模拟main.py:15310-15313）
    print("\n[步骤4] 筛选Window 3 test_data...")
    test_data = factor_df_merged[
        (factor_df_merged['date'] >= pd.to_datetime('2025-06-05')) &
        (factor_df_merged['date'] <= pd.to_datetime('2025-12-15'))
    ].copy()

    print(f"    Window 3 test_data: {len(test_data):,} 条记录")

    # 5. 检查688188.SH的数据
    print("\n[步骤5] 检查688188.SH...")
    stock_688188 = test_data[test_data['stock_code'] == '688188.SH']

    if len(stock_688188) == 0:
        print("    [ERROR] 688188.SH不在test_data中")
        return

    # 检查关键日期
    data_0606 = stock_688188[stock_688188['date'] == pd.to_datetime('2025-06-06')]
    data_0609 = stock_688188[stock_688188['date'] == pd.to_datetime('2025-06-09')]

    if len(data_0606) > 0 and len(data_0609) > 0:
        close_0606 = data_0606['close'].iloc[0]
        open_0609 = data_0609['open'].iloc[0]
        gap_ratio = (open_0609 - close_0606) / close_0606

        print(f"    2025-06-06 close: {close_0606}")
        print(f"    2025-06-09 open: {open_0609}")
        print(f"    计算的gap_ratio: {gap_ratio:.6f} ({gap_ratio:.4%})")

        # 6. 检查prediction_data中的所有数据
        print("\n[步骤6] 检查prediction_data数据质量...")

        # 统计异常值
        invalid_open = ((test_data['open'].isna()) | (test_data['open'] == 0))
        invalid_close = ((test_data['close'].isna()) | (test_data['close'] == 0))

        print(f"    open为NaN或0: {invalid_open.sum()} / {len(test_data)}")
        print(f"    close为NaN或0: {invalid_close.sum()} / {len(test_data)}")

        # 7. 模拟strategy_backtest的构建索引逻辑
        print("\n[步骤7] 模拟回测引擎的索引构建...")

        # 筛选T日和T+1日
        trading_dates = sorted(test_data['date'].unique())
        date_t = pd.to_datetime('2025-06-06').date()
        date_t1 = pd.to_datetime('2025-06-09').date()

        # 标准化为datetime.date（strategy_backtest.py:66）
        test_data_for_backtest = test_data.copy()
        test_data_for_backtest['date'] = pd.to_datetime(test_data_for_backtest['date']).dt.date

        # 获取T日和T+1日数据
        data_t = test_data_for_backtest[test_data_for_backtest['date'] == date_t]
        data_t1 = test_data_for_backtest[test_data_for_backtest['date'] == date_t1]

        print(f"    T日数据: {len(data_t)} 条")
        print(f"    T+1日数据: {len(data_t1)} 条")

        # Inner merge
        merged = pd.merge(
            data_t[['stock_code', 'close']],
            data_t1[['stock_code', 'open']],
            on='stock_code',
            how='inner'
        )

        # 过滤异常值
        merged = merged[(merged['close'] > 0) & (merged['open'] > 0)]

        print(f"    过滤后: {len(merged)} 条")

        # 检查688188.SH
        merged_688188 = merged[merged['stock_code'] == '688188.SH']

        if len(merged_688188) > 0:
            final_close = merged_688188['close'].iloc[0]
            final_open = merged_688188['open'].iloc[0]
            final_gap = (final_open - final_close) / final_close

            print(f"\n    688188.SH:")
            print(f"      最终T日close: {final_close}")
            print(f"      最终T+1日open: {final_open}")
            print(f"      最终gap_ratio: {final_gap:.6f} ({final_gap:.4%})")

            if abs(final_gap + 0.2934) < 0.01:
                print(f"\n    [ERROR] 确认是-29.34%！")
                print(f"    [ERROR] test_data中已经存在错误数据！")
            elif abs(final_gap + 0.00003) < 0.00001:
                print(f"\n    [OK] 确认是-0.003%（正常）")
                print(f"    [OK] test_data中的数据是正确的")
                print(f"\n    [结论] 如果回测日志显示-29.34%，问题可能在:")
                print(f"      1. 前两次回测使用了不同版本的test_data")
                print(f"      2. 回测引擎在某个地方修改了prediction_data")
                print(f"      3. 日志记录的gap_info_map有误")

        # 8. 检查是否有其他688188.SH的数据
        print(f"\n[步骤8] 检查688188.SH在test_data中的所有记录...")
        stock_688188_all = test_data_for_backtest[test_data_for_backtest['stock_code'] == '688188.SH'].copy()
        stock_688188_all = stock_688188_all.sort_values('date')

        print(f"    总记录数: {len(stock_688188_all)}")

        # 检查是否有open或close为0的记录
        invalid = stock_688188_all[
            (stock_688188_all['open'] == 0) |
            (stock_688188_all['close'] == 0) |
            stock_688188_all['open'].isna() |
            stock_688188_all['close'].isna()
        ]

        if len(invalid) > 0:
            print(f"\n    [WARNING] 发现异常记录:")
            for _, row in invalid.iterrows():
                print(f"      {row['date']}: open={row['open']}, close={row['close']}")

        # 显示6月的数据
        june_data = stock_688188_all[
            (stock_688188_all['date'] >= pd.to_datetime('2025-06-01').date()) &
            (stock_688188_all['date'] <= pd.to_datetime('2025-06-30').date())
        ]

        print(f"\n    6月数据 ({len(june_data)} 条):")
        for _, row in june_data.iterrows():
            print(f"      {row['date']}: open={row['open']:.2f}, close={row['close']:.2f}")

if __name__ == "__main__":
    diagnose_root_cause()
