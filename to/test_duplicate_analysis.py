"""
深度分析：为什么每只股票每天有3条记录？

检查数据结构，找出重复的真正原因
"""

import tushare as ts
import pandas as pd

# 设置token
ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000')

print("=" * 80)
print("深度分析：数据重复原因")
print("=" * 80)

# 测试一只股票，一天
stock = "000001.SZ"
start_date = "20260109"
end_date = "20260109"

print(f"\n测试: {stock}, 单日 {start_date}")

df = ts.pro_bar(
    ts_code=stock,
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"\n返回行数: {len(df)}")
print(f"列名: {list(df.columns)}")

print(f"\n所有数据:")
print(df)

# 分析重复
print(f"\n[分析]")
print(f"数据行数: {len(df)}")
print(f"预期: 1 行（每天1条）")

if len(df) > 1:
    print(f"\n发现 {len(df)} 条记录，可能原因:")

    # 检查是否有重复列
    if 'trade_date' in df.columns:
        dates = df['trade_date'].unique()
        print(f"  1. trade_date唯一值: {dates}")

    # 检查calender列
    if 'cal_date' in df.columns:
        print(f"  2. 发现 cal_date 列！")
        print(f"  这可能意味着API返回了额外的日历信息")

    # 检查所有唯一值
    print(f"\n各字段的唯一值:")
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals > 1 and unique_vals <= 10:
            print(f"  {col}: {unique_vals} 个唯一值 - {list(df[col].unique())}")
        elif unique_vals == 1:
            print(f"  {col}: {unique_vals} 个唯一值 - {df[col].iloc[0]}")
        else:
            print(f"  {col}: {unique_vals} 个唯一值")

    # 检查是否有完全相同的行
    dup_rows = df.duplicated(keep=False).sum()
    if dup_rows > 0:
        print(f"\n  有 {dup_rows} 行完全重复的数据")
    else:
        print(f"\n  没有完全相同的行")

print(f"\n" + "=" * 80)
