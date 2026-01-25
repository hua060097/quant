#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Factor Lab merge后的数据质量问题

关键发现：
1. Factor Lab只有109个因子 + date + stock_code，**没有** open/close
2. main.py通过merge Daily Features来添加open/close
3. 如果merge失败，open/close会被填充为0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def analyze_merge_issue():
    """分析merge导致的数据问题"""
    print("=" * 100)
    print("[核心发现] Factor Lab不包含open/close字段")
    print("=" * 100)

    # 1. 加载Factor Lab
    factor_file = project_root / "factor_lab_data" / "factors.parquet"
    factor_df = pd.read_parquet(factor_file)

    print(f"\n[1] Factor Lab数据:")
    print(f"    总记录数: {len(factor_df):,}")
    print(f"    总列数: {len(factor_df.columns)}")
    print(f"    列名: {factor_df.columns.tolist()[:10]}...")

    has_ohlc = any(col in factor_df.columns for col in ['open', 'close', 'high', 'low'])
    print(f"    是否包含OHLC字段: {has_ohlc}")

    if not has_ohlc:
        print(f"    WARNING: Factor Lab **不包含** open/close字段！")
        print(f"    WARNING: 必须通过merge Daily Features来获取这些字段")

    # 2. 检查Daily Features目录
    features_dir = project_root / "local_data" / "features" / "daily"

    print(f"\n[2] 检查Daily Features目录:")
    print(f"    路径: {features_dir}")

    if not features_dir.exists():
        print(f"    ERROR: 目录不存在！")
        print(f"    ERROR: 这会导致merge失败，open/close被填充为0.0")
        return

    # 列出所有文件
    parquet_files = list(features_dir.glob("*.parquet"))
    print(f"    文件数量: {len(parquet_files)}")

    if len(parquet_files) == 0:
        print(f"    ERROR: 没有任何parquet文件！")
        print(f"    ERROR: 这会导致merge失败，open/close被填充为0.0")
        return

    # 检查是否有6月的文件
    june_files = [f for f in parquet_files if '202506' in f.name]
    print(f"    2025年6月文件数量: {len(june_files)}")

    if len(june_files) == 0:
        print(f"    WARNING: 没有2025年6月的文件！")
        print(f"    WARNING: 这会导致688188.SH在6月的数据merge失败")

        # 检查最近的文件
        recent_files = sorted(parquet_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        print(f"\n    最近修改的5个文件:")
        for f in recent_files:
            print(f"      - {f.name} ({f.stat().st_mtime})")

        return

    # 3. 如果有6月文件，检查688188.SH的数据
    print(f"\n[3] 检查688188.SH在6月的数据:")

    # 加载6月所有文件
    june_data = []
    for june_file in june_files:
        df = pd.read_parquet(june_file)
        june_data.append(df)

    if june_data:
        june_df = pd.concat(june_data, ignore_index=True)
        stock_688188 = june_df[june_df['stock_code'] == '688188.SH']

        print(f"    6月总记录数: {len(june_df):,}")
        print(f"    688188.SH记录数: {len(stock_688188)}")

        if len(stock_688188) > 0:
            print(f"\n    688188.SH的6月数据:")
            for _, row in stock_688188.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                print(f"      {date_str}: open={row['open']:.2f}, close={row['close']:.2f}")
        else:
            print(f"    ERROR: 688188.SH在6月完全没有数据！")
            print(f"    ERROR: 这会导致merge后open/close被填充为0.0")

            # 检查是否有其他股票的数据
            sample_stock = june_df['stock_code'].iloc[0] if len(june_df) > 0 else None
            if sample_stock:
                print(f"\n    对比：{sample_stock}在6月有数据:")
                sample_data = june_df[june_df['stock_code'] == sample_stock].head(5)
                for _, row in sample_data.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    print(f"      {date_str}: open={row['open']:.2f}, close={row['close']:.2f}")

    # 4. 模拟merge操作
    print(f"\n[4] 模拟merge操作（how='left'）:")

    # 获取688188.SH在Factor Lab中的6月数据
    stock_factor = factor_df[
        (factor_df['stock_code'] == '688188.SH') &
        (factor_df['date'] >= '2025-06-01') &
        (factor_df['date'] <= '2025-06-30')
    ].copy()

    print(f"    Factor Lab中688188.SH的6月记录: {len(stock_factor)}")

    if len(june_data) > 0:
        # 准备merge数据
        daily_subset = june_df[['date', 'stock_code', 'open', 'close']].copy()

        # 执行merge
        merged = stock_factor[['date', 'stock_code']].merge(
            daily_subset,
            on=['date', 'stock_code'],
            how='left'
        )

        print(f"    merge后记录数: {len(merged)}")
        print(f"    open为NaN或0的记录: {((merged['open'].isna()) | (merged['open'] == 0)).sum()}")
        print(f"    close为NaN或0的记录: {((merged['close'].isna()) | (merged['close'] == 0)).sum()}")

        # 检查关键日期
        for date_str in ['2025-06-06', '2025-06-09']:
            target_date = pd.to_datetime(date_str)
            date_data = merged[merged['date'] == target_date]

            if len(date_data) > 0:
                open_val = date_data['open'].iloc[0]
                close_val = date_data['close'].iloc[0]
                print(f"\n    {date_str}:")
                print(f"      open: {open_val} (isna: {pd.isna(open_val)})")
                print(f"      close: {close_val} (isna: {pd.isna(close_val)})")

    # 5. 结论
    print(f"\n" + "=" * 100)
    print("[结论]")
    print("=" * 100)

    if not has_ohlc:
        print("[OK] Factor Lab确实不包含open/close字段")
        print("[OK] 必须通过merge Daily Features来获取这些字段")

        if len(june_files) == 0:
            print("\n[ROOT CAUSE] Daily Features目录下没有2025年6月的文件！")
            print("[ROOT CAUSE] 导致688188.SH在6月的open/close被填充为默认值0.0")
            print("[ROOT CAUSE] strategy_backtest.py计算gap_ratio时使用这些0值，导致异常结果")
        elif len(stock_688188) == 0:
            print("\n[ROOT CAUSE] Daily Features中688188.SH在6月没有数据！")
            print("[ROOT CAUSE] 导致merge后open/close被填充为默认值0.0")
            print("[ROOT CAUSE] strategy_backtest.py计算gap_ratio时使用这些0值，导致异常结果")
        else:
            print("\n[OK] Daily Features中688188.SH有数据")
            print("[OK] 需要进一步检查merge过程")

if __name__ == "__main__":
    analyze_merge_issue()
