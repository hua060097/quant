"""
模拟真实场景测试：下载20260101-20260109，测试批量下载是否正常
"""

import tushare as ts
import pandas as pd
from datetime import datetime

# 初始化
pro = ts.pro_api()

print("=" * 80)
print("模拟真实场景测试")
print("=" * 80)

# 测试参数（模拟 download_daily.py 的调用方式）
test_stocks = ["000001.SZ", "000002.SZ", "000004.SZ", "000005.SZ", "000006.SZ",  # 5只股票
               "000007.SZ", "000008.SZ", "000009.SZ", "000010.SZ", "000011.SZ"]
adj = 'qfq'
start_date = '20260101'
end_date = '20260109'

print(f"\n测试参数:")
print(f"  股票数量: {len(test_stocks)}")
print(f"  日期范围: {start_date} ~ {end_date}")
print(f"  预期行数: {len(test_stocks) * 9} (每只股票9天)")

print(f"\n开始批量下载...")

# 调用 ts.pro_bar（与 download_daily.py 完全一致）
batch_df_raw = ts.pro_bar(
    ts_code=','.join(test_stocks),
    adj=adj,
    start_date=start_date,
    end_date=end_date
)

print(f"\n[结果]")
print(f"  返回行数: {len(batch_df_raw)}")
print(f"  ts_code唯一值: {batch_df_raw['ts_code'].nunique()}")

if 'trade_date' in batch_df_raw.columns:
    date_range = f"{batch_df_raw['trade_date'].min()} ~ {batch_df_raw['trade_date'].max()}"
    print(f"  trade_date范围: {date_range}")

# 检查重复
print(f"\n[重复检查]")
print(f"  预期行数: {len(test_stocks) * 9}")
print(f"  实际行数: {len(batch_df_raw)}")
print(f"  比值: {len(batch_df_raw) / (len(test_stocks) * 9):.2f}x")

if len(batch_df_raw) > len(test_stocks) * 9 * 1.5:  # 允许50%误差
    print(f"  [警告] 数据行数异常！")
    print(f"\n[每只股票的记录数]")
    stock_counts = batch_df_raw.groupby('ts_code').size()
    print(stock_counts)

    print(f"\n[样本数据 - 000001.SZ]")
    sample = batch_df_raw[batch_df_raw['ts_code'] == '000001.SZ']
    print(sample[['ts_code', 'trade_date', 'open', 'close']].head(20))
else:
    print(f"  [OK] 数据行数正常")

# 测试逐股票下载（对比）
print(f"\n" + "=" * 80)
print("对比测试：逐股票下载")
print("=" * 80)

individual_data = []
for stock in test_stocks[:3]:  # 只测试前3只
    df = ts.pro_bar(
        ts_code=stock,
        adj=adj,
        start_date=start_date,
        end_date=end_date
    )
    if df is not None and len(df) > 0:
        individual_data.append(df)
        print(f"  {stock}: {len(df)} 行")

if individual_data:
    individual_df = pd.concat(individual_data, ignore_index=True)
    print(f"\n[逐股票下载结果]")
    print(f"  总行数: {len(individual_df)}")
    print(f"  预期: {3 * 9} = 27 行")
