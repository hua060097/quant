#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析CSV中缺失日期的原因
"""
import pandas as pd
from pathlib import Path
import sys

print("分析CSV中缺失日期的原因...\n")

# 1. 加载因子数据
factor_file = Path("factor_lab_data/factors.parquet")
if not factor_file.exists():
    print(f"❌ 因子文件不存在: {factor_file}")
    sys.exit(1)

factors = pd.read_parquet(factor_file)
factors['date'] = pd.to_datetime(factors['date'])
print(f"✅ 因子数据: {len(factors)} 行, 日期范围 {factors['date'].min()} ~ {factors['date'].max()}\n")

# 2. 加载CSV文件
results_dir = Path("results")
csv_files = list(results_dir.glob("detailed_trades_1stocks_*.csv"))
if not csv_files:
    print("❌ 未找到CSV文件")
    sys.exit(1)

# 使用最新的CSV文件
csv_file = sorted(csv_files)[-1]
print(f"📄 分析文件: {csv_file.name}\n")

trades = pd.read_csv(csv_file)
trades['date'] = pd.to_datetime(trades['date'])
print(f"✅ 交易记录: {len(trades)} 行, 日期范围 {trades['date'].min()} ~ {trades['date'].max()}\n")

# 3. 检查20250106
target_date = pd.to_datetime('2025-01-06')
print("=" * 60)
print(f"检查目标日期: {target_date.date()}")
print("=" * 60)

# 3.1 因子数据中是否有该日期
factor_data_on_date = factors[factors['date'] == target_date]
print(f"\n因子数据中该日期的记录数: {len(factor_data_on_date)}")
if len(factor_data_on_date) > 0:
    print(f"  股票数: {factor_data_on_date['stock_code'].nunique()}")
    print(f"  示例股票: {factor_data_on_date['stock_code'].head(5).tolist()}")
else:
    print("  ❌ 因子数据中没有该日期!")

# 3.2 CSV中是否有该日期
trades_on_date = trades[trades['date'] == target_date]
print(f"\nCSV中该日期的记录数: {len(trades_on_date)}")
if len(trades_on_date) > 0:
    print(f"  股票数: {trades_on_date['stock_code'].nunique()}")
    print(f"  示例股票: {trades_on_date['stock_code'].head(5).tolist()}")
else:
    print("  ❌ CSV中没有该日期!")

# 4. 分析可能的原因
print("\n" + "=" * 60)
print("原因分析")
print("=" * 60)

if len(factor_data_on_date) == 0:
    print("❌ 原因1: 因子数据中缺少该日期")
else:
    print("✅ 因子数据中有该日期")
    
    # 检查必要字段
    required_fields = ['future_return_2d', 'is_can_buy', 'pct_chg', 'atr14']
    missing_fields = [f for f in required_fields if f not in factor_data_on_date.columns]
    if missing_fields:
        print(f"❌ 原因2: 缺少必要字段: {missing_fields}")
    else:
        print("✅ 必要字段完整")
        
        # 检查is_can_buy
        can_buy_count = (factor_data_on_date['is_can_buy'] == 1).sum()
        print(f"\n可买入股票数: {can_buy_count} / {len(factor_data_on_date)}")
        
        if can_buy_count == 0:
            print("❌ 原因3: 所有股票都不可买入(is_can_buy=0)")
        else:
            print(f"✅ 有 {can_buy_count} 只股票可买入")
            
            # 检查future_return_2d
            can_buy_stocks = factor_data_on_date[factor_data_on_date['is_can_buy'] == 1]
            has_future_return = can_buy_stocks['future_return_2d'].notna().sum()
            print(f"\n有future_return_2d的股票数: {has_future_return} / {can_buy_count}")
            
            if has_future_return == 0:
                print("❌ 原因4: 所有可买入股票都没有future_return_2d")
            else:
                print(f"✅ 有 {has_future_return} 只股票有future_return_2d")

# 5. 检查前后日期
print("\n" + "=" * 60)
print("前后日期对比")
print("=" * 60)

# 获取所有交易日
all_dates = sorted(factors['date'].unique())
target_idx = None
for i, d in enumerate(all_dates):
    if d == target_date:
        target_idx = i
        break

if target_idx is not None:
    print(f"\n目标日期在因子数据中的位置: {target_idx + 1} / {len(all_dates)}")
    
    # 前一个交易日
    if target_idx > 0:
        prev_date = all_dates[target_idx - 1]
        prev_trades = trades[trades['date'] == prev_date]
        print(f"\n前一交易日 {prev_date.date()}:")
        print(f"  CSV记录数: {len(prev_trades)}")
    
    # 后一个交易日
    if target_idx < len(all_dates) - 1:
        next_date = all_dates[target_idx + 1]
        next_trades = trades[trades['date'] == next_date]
        print(f"\n后一交易日 {next_date.date()}:")
        print(f"  CSV记录数: {len(next_trades)}")

# 6. 检查CSV中的日期分布
print("\n" + "=" * 60)
print("CSV日期分布")
print("=" * 60)

csv_dates = sorted(trades['date'].unique())
print(f"\nCSV中的唯一日期数: {len(csv_dates)}")
print(f"日期范围: {csv_dates[0].date()} ~ {csv_dates[-1].date()}")

# 检查是否有连续缺失
factor_dates = sorted(factors['date'].unique())
factor_dates_in_range = [d for d in factor_dates if csv_dates[0] <= d <= csv_dates[-1]]
missing_dates = [d for d in factor_dates_in_range if d not in csv_dates]

print(f"\n因子数据在CSV日期范围内的交易日数: {len(factor_dates_in_range)}")
print(f"CSV中缺失的交易日数: {len(missing_dates)}")

if len(missing_dates) > 0:
    print(f"\n缺失的日期 (前10个):")
    for d in missing_dates[:10]:
        print(f"  {d.date()}")

print("\n分析完成!")

