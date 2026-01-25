"""
使用用户实际使用的参数测试
20260101 ~ 20260109，批量下载
"""

import tushare as ts
import pandas as pd

# 设置token
ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000')

print("=" * 80)
print("模拟用户实际场景测试")
print("=" * 80)

# 用户实际使用的参数
test_stocks = ["000001.SZ", "000002.SZ", "600000.SH"]  # 先测试3只
start_date = "20260101"
end_date = "20260109"

print(f"\n参数:")
print(f"  股票: {len(test_stocks)} 只")
print(f"  日期: {start_date} ~ {end_date} (9天)")
print(f"  预期: {len(test_stocks)} × 9 = {len(test_stocks) * 9} 行")

# 批量下载
print(f"\n开始批量下载...")
df_batch = ts.pro_bar(
    ts_code=','.join(test_stocks),
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"\n[结果]")
print(f"返回行数: {len(df_batch)}")

if len(df_batch) > 0:
    print(f"日期范围: {df_batch['trade_date'].min()} ~ {df_batch['trade_date'].max()}")
    print(f"日期唯一值: {df_batch['trade_date'].nunique()}")
    print(f"所有日期: {sorted(df_batch['trade_date'].unique())}")

    # 每只股票的统计
    print(f"\n每只股票的统计:")
    for stock in test_stocks:
        stock_data = df_batch[df_batch['ts_code'] == stock]
        print(f"\n  {stock}:")
        print(f"    记录数: {len(stock_data)}")
        print(f"    日期数: {stock_data['trade_date'].nunique()}")
        print(f"    所有日期: {sorted(stock_data['trade_date'].unique())}")

    # 检查超出范围的数据
    out_of_range = df_batch[
        (df_batch['trade_date'] < start_date) |
        (df_batch['trade_date'] > end_date)
    ]

    if len(out_of_range) > 0:
        print(f"\n[警告] 发现 {len(out_of_range)} 条超出日期范围的数据!")
        print(f"超出日期的示例: {sorted(out_of_range['trade_date'].unique())[:10]}")
    else:
        print(f"\n[OK] 所有数据都在指定日期范围内")

    print(f"\n数据样本:")
    print(df_batch[['ts_code', 'trade_date', 'open', 'close']].head(20))

# 判断
expected_rows = len(test_stocks) * 9
ratio = len(df_batch) / expected_rows if expected_rows > 0 else 0

print(f"\n[分析]")
print(f"  预期: {expected_rows} 行")
print(f"  实际: {len(df_batch)} 行")
print(f"  比值: {ratio:.2f}x")

if len(df_batch) == expected_rows:
    print(f"  [结论] API正常工作!")
elif len(df_batch) > expected_rows * 10:
    print(f"  [BUG] 日期参数失效，返回了 {ratio:.1f} 倍的数据!")
else:
    print(f"  [结论] 数据量基本正常（可能有停牌）")

print(f"\n" + "=" * 80)
