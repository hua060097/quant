#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证因子文件的完整性
"""
import pandas as pd
from pathlib import Path

print("验证因子文件...\n")

# 加载因子文件
factor_file = Path("factor_lab_data/factors.parquet")
if not factor_file.exists():
    print(f"❌ 因子文件不存在: {factor_file}")
    exit(1)

df = pd.read_parquet(factor_file)
print(f"✅ 成功加载因子文件")
print(f"总行数: {len(df):,}")
print(f"总列数: {len(df.columns)}")
print(f"因子数: {len(df.columns) - 2} (date, stock_code除外)\n")

# 检查日期范围
df['date'] = pd.to_datetime(df['date'])
print("=" * 60)
print("日期范围检查")
print("=" * 60)
print(f"最早日期: {df['date'].min()}")
print(f"最晚日期: {df['date'].max()}")
print(f"唯一日期数: {df['date'].nunique()}\n")

# 检查关键日期
print("=" * 60)
print("关键日期检查")
print("=" * 60)

# 检查20250106
date_20250106 = pd.to_datetime('2025-01-06')
has_20250106 = (df['date'] == date_20250106).any()
print(f"包含2025-01-06: {'✅ 是' if has_20250106 else '❌ 否'}")
if has_20250106:
    count_20250106 = len(df[df['date'] == date_20250106])
    print(f"  股票数: {count_20250106}")

# 检查20250814
date_20250814 = pd.to_datetime('2025-08-14')
has_20250814 = (df['date'] == date_20250814).any()
print(f"包含2025-08-14: {'✅ 是' if has_20250814 else '❌ 否'}")
if has_20250814:
    count_20250814 = len(df[df['date'] == date_20250814])
    print(f"  股票数: {count_20250814}")

# 检查20250930
date_20250930 = pd.to_datetime('2025-09-30')
has_20250930 = (df['date'] == date_20250930).any()
print(f"包含2025-09-30: {'✅ 是' if has_20250930 else '❌ 否'}")
if has_20250930:
    count_20250930 = len(df[df['date'] == date_20250930])
    print(f"  股票数: {count_20250930}")

print()

# 检查2025年8月14日之后的数据
print("=" * 60)
print("2025-08-14之后数据检查")
print("=" * 60)
after_20250814 = df[df['date'] > date_20250814]
print(f"2025-08-14之后的数据行数: {len(after_20250814):,}")
print(f"2025-08-14之后的唯一日期数: {after_20250814['date'].nunique()}")
if len(after_20250814) > 0:
    print(f"日期范围: {after_20250814['date'].min()} ~ {after_20250814['date'].max()}")
    print(f"平均每日股票数: {len(after_20250814) / after_20250814['date'].nunique():.0f}")

print()

# 检查数据完整性
print("=" * 60)
print("数据完整性检查")
print("=" * 60)

# 检查缺失值
factor_cols = [col for col in df.columns if col not in ['date', 'stock_code']]
missing_stats = []
for col in factor_cols[:10]:  # 只检查前10个因子
    missing_count = df[col].isna().sum()
    missing_pct = missing_count / len(df) * 100
    if missing_count > 0:
        missing_stats.append((col, missing_count, missing_pct))

if missing_stats:
    print("前10个因子的缺失值统计:")
    for col, count, pct in missing_stats:
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("✅ 前10个因子无缺失值")

print()

# 检查每日股票数
print("=" * 60)
print("每日股票数统计")
print("=" * 60)
daily_counts = df.groupby('date').size()
print(f"平均每日股票数: {daily_counts.mean():.0f}")
print(f"最少每日股票数: {daily_counts.min()}")
print(f"最多每日股票数: {daily_counts.max()}")

# 检查最近10个交易日
print(f"\n最近10个交易日的股票数:")
recent_dates = sorted(df['date'].unique())[-10:]
for date in recent_dates:
    count = len(df[df['date'] == date])
    print(f"  {date.date()}: {count}")

print("\n验证完成!")

