"""
简单测试：验证 ts.pro_bar() 日期参数是否有效

只测试：
1. 单个股票下载指定日期
2. 批量下载指定日期
3. 检查是否返回超出日期范围的数据
"""

import tushare as ts
import pandas as pd

# 初始化
pro = ts.pro_api()

print("=" * 80)
print("ts.pro_bar() 日期参数测试")
print("=" * 80)

# 测试参数
stock1 = "000001.SZ"
stock2 = "000002.SZ"
start_date = "20240103"
end_date = "20240105"

print(f"\n测试参数:")
print(f"  日期范围: {start_date} ~ {end_date} (3天)")
print(f"  预期: 每只股票 3 条数据")

# 测试1：单个股票
print(f"\n" + "=" * 80)
print(f"测试1: 单个股票下载 ({stock1})")
print("=" * 80)

df_single = ts.pro_bar(
    ts_code=stock1,
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"返回行数: {len(df_single)}")
print(f"日期范围: {df_single['trade_date'].min()} ~ {df_single['trade_date'].max()}")
print(f"日期唯一值: {df_single['trade_date'].nunique()}")
print(f"数据样本:")
print(df_single[['trade_date', 'open', 'close']])

# 测试2：批量下载（2个股票）
print(f"\n" + "=" * 80)
print(f"测试2: 批量下载 ({stock1}, {stock2})")
print("=" * 80)

df_batch = ts.pro_bar(
    ts_code=f"{stock1},{stock2}",
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"返回行数: {len(df_batch)}")
print(f"日期范围: {df_batch['trade_date'].min()} ~ {df_batch['trade_date'].max()}")
print(f"日期唯一值: {df_batch['trade_date'].nunique()}")

# 每只股票的数据
print(f"\n每只股票的记录数:")
for stock in [stock1, stock2]:
    count = len(df_batch[df_batch['ts_code'] == stock])
    print(f"  {stock}: {count} 条")

print(f"\n数据样本:")
print(df_batch[['ts_code', 'trade_date', 'open', 'close']].head(10))

# 结论
print(f"\n" + "=" * 80)
print(f"结论:")
print("=" * 80)

print(f"\n单个股票:")
if len(df_single) <= 5:
    print(f"  [OK] 返回 {len(df_single)} 行，日期参数正常")
else:
    print(f"  [BUG] 返回 {len(df_single)} 行，远超预期的 3 行！")
    print(f"  [原因] start_date/end_date 参数失效")

print(f"\n批量下载:")
if len(df_batch) <= 10:
    print(f"  [OK] 返回 {len(df_batch)} 行，日期参数正常")
else:
    print(f"  [BUG] 返回 {len(df_batch)} 行，远超预期的 6 行！")
    print(f"  [原因] 批量模式下 start_date/end_date 参数失效")
