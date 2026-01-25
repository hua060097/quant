"""
测试 ts.pro_bar() 日期参数是否有效
使用 ts.set_token() 初始化
"""

import tushare as ts
import pandas as pd

# 设置token
ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000')

print("=" * 80)
print("ts.pro_bar() 日期参数有效性测试")
print("=" * 80)

# 测试参数
stock1 = "000001.SZ"
stock2 = "000002.SZ"
start_date = "20240103"
end_date = "20240105"

print(f"\n测试参数:")
print(f"  股票: {stock1}, {stock2}")
print(f"  日期: {start_date} ~ {end_date} (3天)")
print(f"  预期: 每只股票 3 条数据")

# ============================================================================
# 测试1: 单个股票下载
# ============================================================================
print(f"\n" + "=" * 80)
print(f"测试1: 单个股票下载 ({stock1})")
print("=" * 80)

df_single = ts.pro_bar(
    ts_code=stock1,
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"\n返回行数: {len(df_single)}")

if len(df_single) > 0:
    print(f"日期范围: {df_single['trade_date'].min()} ~ {df_single['trade_date'].max()}")
    print(f"日期唯一值: {df_single['trade_date'].nunique()}")
    print(f"所有日期: {sorted(df_single['trade_date'].unique())}")

    # 检查是否有超出日期范围的数据
    out_of_range = df_single[
        (df_single['trade_date'] < start_date) |
        (df_single['trade_date'] > end_date)
    ]

    if len(out_of_range) > 0:
        print(f"\n[警告] 发现 {len(out_of_range)} 条超出指定日期的数据!")
        print(f"超出日期的示例: {sorted(out_of_range['trade_date'].unique())[:10]}")
    else:
        print(f"\n[OK] 所有数据都在指定日期范围内")

    print(f"\n数据样本:")
    print(df_single[['trade_date', 'open', 'high', 'low', 'close']].head(10))

# 判断
if len(df_single) <= 10:
    print(f"\n[结论] 单个股票下载: 正常 - 日期参数生效")
elif len(df_single) > 100:
    print(f"\n[结论] 单个股票下载: BUG - 日期参数失效!")
    print(f"  返回了 {len(df_single)} 行而不是预期的 3-5 行")
else:
    print(f"\n[结论] 单个股票下载: 基本正常 (可能有少量停牌)")

# ============================================================================
# 测试2: 批量下载（2个股票）
# ============================================================================
print(f"\n" + "=" * 80)
print(f"测试2: 批量下载 ({stock1}, {stock2})")
print("=" * 80)

df_batch = ts.pro_bar(
    ts_code=f"{stock1},{stock2}",
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"\n返回行数: {len(df_batch)}")

if len(df_batch) > 0:
    print(f"日期范围: {df_batch['trade_date'].min()} ~ {df_batch['trade_date'].max()}")
    print(f"日期唯一值: {df_batch['trade_date'].nunique()}")
    print(f"所有日期: {sorted(df_batch['trade_date'].unique())}")

    # 每只股票的数据
    print(f"\n每只股票的记录数:")
    for stock in [stock1, stock2]:
        stock_data = df_batch[df_batch['ts_code'] == stock]
        print(f"  {stock}: {len(stock_data)} 条")
        if len(stock_data) > 0:
            dates = sorted(stock_data['trade_date'].unique())
            print(f"    日期: {dates}")

    # 检查是否有超出日期范围的数据
    out_of_range = df_batch[
        (df_batch['trade_date'] < start_date) |
        (df_batch['trade_date'] > end_date)
    ]

    if len(out_of_range) > 0:
        print(f"\n[警告] 发现 {len(out_of_range)} 条超出指定日期的数据!")
        print(f"超出日期的示例: {sorted(out_of_range['trade_date'].unique())[:10]}")
    else:
        print(f"\n[OK] 所有数据都在指定日期范围内")

    print(f"\n数据样本:")
    print(df_batch[['ts_code', 'trade_date', 'open', 'close']].head(10))

# 判断
if len(df_batch) <= 20:
    print(f"\n[结论] 批量下载: 正常 - 日期参数生效")
elif len(df_batch) > 200:
    print(f"\n[结论] 批量下载: BUG - 日期参数失效!")
    print(f"  返回了 {len(df_batch)} 行而不是预期的 6 行")
else:
    print(f"\n[结论] 批量下载: 基本正常 (可能有停牌)")

# ============================================================================
# 最终总结
# ============================================================================
print(f"\n" + "=" * 80)
print(f"[最终总结]")
print("=" * 80)

print(f"\n单个股票: {len(df_single)} 行")
if len(df_single) > 100:
    print(f"  状态: [BUG] 日期参数失效")
    print(f"  说明: start_date/end_date 参数在单个股票下载时无效")
elif len(df_single) <= 10:
    print(f"  状态: [正常] 日期参数生效")
else:
    print(f"  状态: [部分正常]")

print(f"\n批量下载: {len(df_batch)} 行")
if len(df_batch) > 200:
    print(f"  状态: [BUG] 日期参数失效")
    print(f"  说明: start_date/end_date 参数在批量下载时无效")
elif len(df_batch) <= 20:
    print(f"  状态: [正常] 日期参数生效")
else:
    print(f"  状态: [部分正常]")

# 关键判断
if len(df_single) > 100 and len(df_batch) > 200:
    print(f"\n[关键结论] API本身存在严重bug!")
    print(f"  无论是单个还是批量下载，日期参数都失效")
    print(f"  返回全部历史数据而不是指定日期范围")
    print(f"  建议: 必须使用其他API或自己过滤日期")
elif len(df_single) <= 10 and len(df_batch) > 200:
    print(f"\n[关键结论] 批量模式存在bug!")
    print(f"  单个股票下载正常，但批量下载时日期参数失效")
    print(f"  建议: 回退到逐股票下载")
elif len(df_single) <= 10 and len(df_batch) <= 20:
    print(f"\n[关键结论] API正常工作!")
    print(f"  日期参数在所有模式下都生效")
    print(f"  建议: 继续使用批量下载")

print(f"\n" + "=" * 80)
