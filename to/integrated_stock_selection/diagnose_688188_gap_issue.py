#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断688188.SH显示-29.34%的根本原因

模拟回测时的数据加载和合并过程
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def diagnose_gap_issue():
    """诊断开盘跌幅计算问题"""
    print("=" * 100)
    print("诊断688188.SH开盘跌幅显示-29.34%的根本原因")
    print("=" * 100)

    # 1. 加载Factor Lab数据（模拟_load_factor_cache_data）
    factor_file = project_root / "factor_lab_data" / "factors.parquet"
    print(f"\n[1] 加载Factor Lab数据: {factor_file}")

    if not factor_file.exists():
        print(f"   [ERROR] 文件不存在")
        return

    factor_df = pd.read_parquet(factor_file)
    print(f"   [OK] 加载完成: {len(factor_df):,} 条记录")
    print(f"   日期范围: {factor_df['date'].min()} ~ {factor_df['date'].max()}")
    print(f"   股票数量: {factor_df['stock_code'].nunique()}")

    # 2. 检查688188.SH在Factor Lab中的数据
    print("\n[2] 检查688188.SH在Factor Lab中的数据")
    stock_688188 = factor_df[factor_df['stock_code'] == '688188.SH'].copy()
    print(f"   总记录数: {len(stock_688188)}")

    if len(stock_688188) == 0:
        print(f"   [WARNING] Factor Lab中没有688188.SH的数据")
        return

    # 检查关键日期的数据
    target_dates = ['2025-06-06', '2025-06-09']
    for date_str in target_dates:
        target_date = pd.to_datetime(date_str)
        date_data = stock_688188[stock_688188['date'] == target_date]

        if len(date_data) > 0:
            print(f"\n   {date_str}:")
            print(f"     记录数: {len(date_data)}")
            if 'open' in date_data.columns:
                open_val = date_data['open'].iloc[0]
                close_val = date_data['close'].iloc[0] if 'close' in date_data.columns else None
                print(f"     open: {open_val} (类型: {type(open_val).__name__})")
                if close_val is not None:
                    print(f"     close: {close_val} (类型: {type(close_val).__name__})")
                    print(f"     是否为NaN: {pd.isna(open_val)}, {pd.isna(close_val)}")
        else:
            print(f"\n   {date_str}: 无数据")

    # 3. 加载Daily Features（模拟load_stock_features）
    print("\n[3] 加载Daily Features数据")
    features_dir = project_root / "local_data" / "features" / "daily"

    daily_data_list = []
    for date_str in ['20250606', '20250609']:
        file_path = features_dir / f"{date_str}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            print(f"   [OK] {date_str}: {len(df):,} 条记录")
            daily_data_list.append(df)
        else:
            print(f"   [ERROR] {date_str}: 文件不存在")

    if not daily_data_list:
        print("   [ERROR] 未能加载任何Daily Features数据")
        return

    daily_data = pd.concat(daily_data_list, ignore_index=True)
    print(f"   合并后: {len(daily_data):,} 条记录")

    # 4. 检查688188.SH在Daily Features中的数据
    print("\n[4] 检查688188.SH在Daily Features中的数据")
    daily_688188 = daily_data[daily_data['stock_code'] == '688188.SH']

    if len(daily_688188) == 0:
        print(f"   [WARNING] Daily Features中没有688188.SH的数据")
        print(f"   -> 这会导致merge后open/close被填充为默认值0.0")
    else:
        print(f"   总记录数: {len(daily_688188)}")
        for _, row in daily_688188.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'NaT'
            print(f"\n   {date_str}:")
            print(f"     open: {row['open']} (是否为NaN: {pd.isna(row['open'])})")
            print(f"     close: {row['close']} (是否为NaN: {pd.isna(row['close'])})")
            print(f"     volume: {row.get('volume', 'N/A')}")

    # 5. 模拟merge操作（关键步骤）
    print("\n[5] 模拟merge操作（how='left'）")

    # 准备merge数据
    factor_subset = stock_688188[['date', 'stock_code']].copy()
    daily_subset = daily_data[['date', 'stock_code', 'open', 'close']].copy()

    # 执行merge（模拟main.py line 1217）
    merged = factor_subset.merge(daily_subset, on=['date', 'stock_code'], how='left')

    print(f"   merge前Factor Lab记录数: {len(factor_subset)}")
    print(f"   merge后记录数: {len(merged)}")

    # 检查merge后的数据
    print("\n[6] 检查merge后的数据")
    for date_str in ['2025-06-06', '2025-06-09']:
        target_date = pd.to_datetime(date_str)
        date_data = merged[merged['date'] == target_date]

        if len(date_data) > 0:
            print(f"\n   {date_str}:")
            open_val = date_data['open'].iloc[0]
            close_val = date_data['close'].iloc[0]
            print(f"     open: {open_val} (类型: {type(open_val).__name__}, 是否NaN: {pd.isna(open_val)})")
            print(f"     close: {close_val} (类型: {type(close_val).__name__}, 是否NaN: {pd.isna(close_val)})")

            # 检查是否被填充为默认值
            if pd.isna(open_val) or open_val == 0.0:
                print(f"     -> open被填充为默认值或NaN！")
            if pd.isna(close_val) or close_val == 0.0:
                print(f"     -> close被填充为默认值或NaN！")

    # 7. 模拟strategy_backtest的gap_ratio计算
    print("\n[7] 模拟strategy_backtest的gap_ratio计算")

    # 准备T日和T+1日数据（模拟lines 263-268）
    data_t = merged[merged['date'] == pd.to_datetime('2025-06-06')][['stock_code', 'close']]
    data_t1 = merged[merged['date'] == pd.to_datetime('2025-06-09')][['stock_code', 'open']]

    print(f"   T日(2025-06-06)数据: {len(data_t)} 条")
    print(f"   T+1日(2025-06-09)数据: {len(data_t1)} 条")

    if len(data_t) > 0 and len(data_t1) > 0:
        print(f"\n   T日 close值: {data_t['close'].iloc[0]}")
        print(f"   T+1日 open值: {data_t1['open'].iloc[0]}")

        # Inner merge（模拟line 263-268）
        merged_calc = pd.merge(
            data_t[['stock_code', 'close']],
            data_t1[['stock_code', 'open']],
            on='stock_code',
            how='inner'
        )

        print(f"\n   inner merge后记录数: {len(merged_calc)}")

        if len(merged_calc) > 0:
            close_val = merged_calc['close'].iloc[0]
            open_val = merged_calc['open'].iloc[0]

            print(f"   close: {close_val}")
            print(f"   open: {open_val}")

            # 过滤异常值（模拟line 271）
            filtered = merged_calc[(merged_calc['close'] > 0) & (merged_calc['open'] > 0)]
            print(f"   过滤close>0 and open>0后记录数: {len(filtered)}")

            if len(filtered) > 0:
                # 计算gap_ratio（模拟line 277）
                gap_ratio = (filtered['open'].iloc[0] - filtered['close'].iloc[0]) / filtered['close'].iloc[0]
                print(f"\n   [计算结果] gap_ratio = {gap_ratio:.6f} ({gap_ratio:.2%})")
            else:
                print(f"\n   [被过滤] close或open <= 0，无法计算gap_ratio")
        else:
            print(f"\n   [ERROR] inner merge后无数据")
    else:
        print(f"\n   [ERROR] T日或T+1日数据缺失")

    # 8. 关键发现总结
    print("\n" + "=" * 100)
    print("[诊断结论]")
    print("=" * 100)

    # 检查关键问题
    issues = []

    # 检查1: Daily Features中是否有688188.SH
    if len(daily_688188) == 0:
        issues.append("❌ Daily Features中完全没有688188.SH的数据")
        issues.append("   -> 这会导致merge后open/close被填充为0.0")
        issues.append("   -> 过滤条件 (close > 0) & (open > 0) 会过滤掉这些记录")
        issues.append("   -> 导致无法计算gap_ratio")

    # 检查2: merge后是否有0或NaN
    merged_0606 = merged[merged['date'] == pd.to_datetime('2025-06-06')]
    merged_0609 = merged[merged['date'] == pd.to_datetime('2025-06-09')]

    if len(merged_0606) > 0:
        close_val = merged_0606['close'].iloc[0]
        if pd.isna(close_val) or close_val == 0.0:
            issues.append(f"❌ 2025-06-06的close值为: {close_val} (异常)")

    if len(merged_0609) > 0:
        open_val = merged_0609['open'].iloc[0]
        if pd.isna(open_val) or open_val == 0.0:
            issues.append(f"❌ 2025-06-09的open值为: {open_val} (异常)")

    if issues:
        for issue in issues:
            print(issue)
        print("\n🔍 推测的根本原因:")
        print("   回测时(2026-01-02 21:xx)，Daily Features中可能缺少688188.SH的数据")
        print("   或者数据加载失败，导致merge后open/close为0或NaN")
        print("   这些记录被过滤掉，但其他股票的gap_info_map可能记录了错误的值")
    else:
        print("✓ 当前数据显示正常，推测原因:")
        print("   1. 回测时的数据已被修复/更新")
        print("   2. 或者回测代码有其他bug导致计算错误")
        print("   3. 建议检查是否有其他数据源覆盖了prediction_data")

if __name__ == "__main__":
    diagnose_gap_issue()
