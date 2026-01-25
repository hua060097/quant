#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查因子数据中缺失字段的详细情况
"""
import pandas as pd
from pathlib import Path
import sys

print("检查因子数据中缺失字段的详细情况...\n")

# 加载因子数据
factor_file = Path("factor_lab_data/factors.parquet")
if not factor_file.exists():
    print(f"❌ 因子文件不存在: {factor_file}")
    sys.exit(1)

factors = pd.read_parquet(factor_file)
factors['date'] = pd.to_datetime(factors['date'])
print(f"✅ 因子数据: {len(factors)} 行, {len(factors.columns)} 列\n")

# 检查必要字段
required_fields = ['future_return_2d', 'is_can_buy', 'pct_chg', 'atr14', 'open', 'high', 'low', 'close']

print("=" * 60)
print("字段存在性检查")
print("=" * 60)

for field in required_fields:
    exists = field in factors.columns
    status = "✅" if exists else "❌"
    print(f"{status} {field}: {'存在' if exists else '不存在'}")

print()

# 检查每个字段的缺失情况
print("=" * 60)
print("字段缺失值统计")
print("=" * 60)

existing_fields = [f for f in required_fields if f in factors.columns]
for field in existing_fields:
    missing_count = factors[field].isna().sum()
    missing_pct = missing_count / len(factors) * 100
    print(f"{field}:")
    print(f"  缺失数: {missing_count:,} ({missing_pct:.2f}%)")
    
    if missing_count > 0:
        # 检查缺失值的日期分布
        missing_dates = factors[factors[field].isna()]['date'].unique()
        print(f"  缺失日期数: {len(missing_dates)}")
        if len(missing_dates) <= 10:
            print(f"  缺失日期: {[d.date() for d in sorted(missing_dates)]}")
        else:
            print(f"  缺失日期 (前10个): {[d.date() for d in sorted(missing_dates)[:10]]}")

print()

# 检查20250106的详细情况
target_date = pd.to_datetime('2025-01-06')
print("=" * 60)
print(f"目标日期 {target_date.date()} 的详细情况")
print("=" * 60)

date_data = factors[factors['date'] == target_date]
print(f"\n该日期的记录数: {len(date_data)}")

if len(date_data) > 0:
    print(f"\n字段缺失情况:")
    for field in required_fields:
        if field in date_data.columns:
            missing_count = date_data[field].isna().sum()
            missing_pct = missing_count / len(date_data) * 100
            status = "❌" if missing_count == len(date_data) else "⚠️" if missing_count > 0 else "✅"
            print(f"  {status} {field}: {missing_count}/{len(date_data)} ({missing_pct:.1f}%) 缺失")
        else:
            print(f"  ❌ {field}: 字段不存在")
    
    # 显示示例数据
    print(f"\n示例数据 (前5行):")
    sample_cols = ['stock_code'] + [f for f in required_fields if f in date_data.columns]
    print(date_data[sample_cols].head().to_string())

# 检查所有缺失日期
print("\n" + "=" * 60)
print("所有缺失必要字段的日期")
print("=" * 60)

# 找出至少有一个必要字段完全缺失的日期
problematic_dates = set()
for field in existing_fields:
    # 找出该字段完全缺失的日期
    field_missing_dates = factors.groupby('date')[field].apply(lambda x: x.isna().all())
    problematic_dates.update(field_missing_dates[field_missing_dates].index)

problematic_dates = sorted(problematic_dates)
print(f"\n至少有一个必要字段完全缺失的日期数: {len(problematic_dates)}")

if len(problematic_dates) > 0:
    print(f"\n这些日期 (前20个):")
    for d in problematic_dates[:20]:
        date_data = factors[factors['date'] == d]
        missing_fields = []
        for field in existing_fields:
            if date_data[field].isna().all():
                missing_fields.append(field)
        print(f"  {d.date()}: 缺失 {missing_fields}")

print("\n检查完成!")

