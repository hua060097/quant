"""
深度诊断 ts.pro_bar 批量下载的真实行为

分析为什么会返回 1,247,500 行而不是预期的 4,500 行
"""

import tushare as ts
import pandas as pd
from datetime import datetime
import sys

# 初始化（与 download_daily.py 相同的方式）
pro = ts.pro_api()
print(f"Tushare初始化成功")
print(f"=" * 100)

print("=" * 100)
print("深度诊断：ts.pro_bar 批量下载行为分析")
print("=" * 100)

# 模拟用户的实际场景
batch_stocks = ["000001.SZ", "000002.SZ", "000004.SZ", "000005.SZ", "000006.SZ",
               "000007.SZ", "000008.SZ", "000009.SZ", "000010.SZ", "000011.SZ"]
adj = 'qfq'
start_date = '20260101'
end_date = '20260109'

print(f"\n[测试参数]")
print(f"  股票数量: {len(batch_stocks)}")
print(f"  日期范围: {start_date} ~ {end_date}")
print(f"  复权类型: {adj}")
print(f"  预期行数: {len(batch_stocks)} * 9 = {len(batch_stocks) * 9}")

print(f"\n" + "=" * 100)
print(f"[测试1] 批量下载 - 使用 ts.pro_bar()")
print("=" * 100)

batch_df_raw = ts.pro_bar(
    ts_code=','.join(batch_stocks),
    adj=adj,
    start_date=start_date,
    end_date=end_date
)

print(f"\n[基本信息]")
print(f"  返回行数: {len(batch_df_raw):,}")
print(f"  列名: {list(batch_df_raw.columns)}")
print(f"  ts_code唯一值: {batch_df_raw['ts_code'].nunique()}")

# 检查日期列
if 'trade_date' in batch_df_raw.columns:
    print(f"\n[日期分析]")
    date_range = f"{batch_df_raw['trade_date'].min()} ~ {batch_df_raw['trade_date'].max()}"
    print(f"  trade_date范围: {date_range}")
    print(f"  trade_date唯一值: {batch_df_raw['trade_date'].nunique()}")

    # 显示所有日期
    all_dates = sorted(batch_df_raw['trade_date'].unique())
    print(f"  所有日期: {all_dates}")

# 每只股票的数据量
print(f"\n[每只股票的记录数]")
stock_counts = batch_df_raw.groupby('ts_code').size()
print(stock_counts)

avg_records = stock_counts.mean()
print(f"\n[关键统计]")
print(f"  平均每只股票记录数: {avg_records:.1f}")
print(f"  预期每只股票记录数: 9")
print(f"  比值: {avg_records / 9:.2f}x")

# 检查是否有超出日期范围的数据
print(f"\n[日期范围检查]")
if 'trade_date' in batch_df_raw.columns:
    out_of_range = batch_df_raw[
        (batch_df_raw['trade_date'] < start_date) |
        (batch_df_raw['trade_date'] > end_date)
    ]

    if len(out_of_range) > 0:
        print(f"  [警告] 发现 {len(out_of_range)} 条超出日期范围的数据！")
        print(f"  日期范围: {out_of_range['trade_date'].min()} ~ {out_of_range['trade_date'].max()}")
    else:
        print(f"  [OK] 所有数据都在指定日期范围内")

# 详细分析某一只股票
print(f"\n" + "=" * 100)
print(f"[详细分析] 000001.SZ 的数据")
print("=" * 100)

sample_stock = batch_df_raw[batch_df_raw['ts_code'] == '000001.SZ'].copy()
print(f"  记录数: {len(sample_stock)}")

if 'trade_date' in sample_stock.columns:
    print(f"\n  按日期分组的记录数:")
    date_counts = sample_stock.groupby('trade_date').size()
    print(date_counts)

    print(f"\n  完整数据样本:")
    print(sample_stock[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']].head(20))

# 检查数据结构
print(f"\n" + "=" * 100)
print(f"[数据结构分析]")
print("=" * 100)

print(f"\n数据类型:")
print(batch_df_raw.dtypes)

print(f"\n数据统计:")
print(batch_df_raw.describe())

# 对比逐股票下载
print(f"\n" + "=" * 100)
print(f"[测试2] 逐股票下载对比")
print("=" * 100)

individual_data = []
for stock in batch_stocks[:3]:  # 只测试前3只
    df = ts.pro_bar(
        ts_code=stock,
        adj=adj,
        start_date=start_date,
        end_date=end_date
    )
    if df is not None and len(df) > 0:
        individual_data.append(df)
        print(f"  {stock}: {len(df)} 条记录")

if individual_data:
    individual_df = pd.concat(individual_data, ignore_index=True)
    print(f"\n[逐股票下载结果]")
    print(f"  总行数: {len(individual_df)}")
    print(f"  预期: {3 * 9} = 27 行")
    print(f"  比值: {len(individual_df) / 27:.2f}x")

# 测试3：检查 adj 参数的影响
print(f"\n" + "=" * 100)
print(f"[测试3] 检查 adj 参数的影响")
print("=" * 100)

test_stock = '000001.SZ'

print(f"\n测试 adj=None (默认):")
df_none = ts.pro_bar(
    ts_code=test_stock,
    start_date=start_date,
    end_date=end_date
)
print(f"  返回行数: {len(df_none)}")

print(f"\n测试 adj='qfq' (前复权):")
df_qfq = ts.pro_bar(
    ts_code=test_stock,
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)
print(f"  返回行数: {len(df_qfq)}")

print(f"\n测试 adj='hfq' (后复权):")
df_hfq = ts.pro_bar(
    ts_code=test_stock,
    adj='hfq',
    start_date=start_date,
    end_date=end_date
)
print(f"  返回行数: {len(df_hfq)}")

print(f"\n测试 adj=None (不复权):")
df_no_adj = ts.pro_bar(
    ts_code=test_stock,
    adj=None,
    start_date=start_date,
    end_date=end_date
)
print(f"  返回行数: {len(df_no_adj)}")

# 结论
print(f"\n" + "=" * 100)
print(f"[诊断结论]")
print("=" * 100)

if len(batch_df_raw) > len(batch_stocks) * 9 * 2:
    print(f"[发现异常] 批量下载返回了 {len(batch_df_raw):,} 行数据")
    print(f"  预期: {len(batch_stocks) * 9} 行")
    print(f"  比值: {len(batch_df_raw) / (len(batch_stocks) * 9):.2f}x")

    if 'trade_date' in batch_df_raw.columns:
        actual_days = batch_df_raw['trade_date'].nunique()
        print(f"\n  实际日期数: {actual_days}")
        print(f"  预期日期数: 9")
        print(f"  结论: API返回了 {actual_days} 天的数据，而不是 9 天！")

        if actual_days > 100:
            print(f"\n  [根本原因] ts.pro_bar() 的 start_date/end_date 参数在批量模式下没有生效！")
            print(f"  API返回了所有历史数据，而不是指定日期范围的数据。")
else:
    print(f"[正常] 批量下载返回 {len(batch_df_raw)} 行，符合预期")

print(f"\n" + "=" * 100)
