"""
测试批量下载时的数据结构
验证 ts.pro_bar 是否正确返回不同股票的数据
"""

import tushare as ts
import pandas as pd

# 设置token（需要从环境变量或配置文件读取）
ts.set_token('2d87b56abd0b44349cb9237945139f8e4b2298e1fe9a0f6f3c8a6e2b')
pro = ts.pro_api()

# 测试批量下载3只股票
test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']

print("=" * 80)
print("测试批量下载数据结构")
print("=" * 80)

print(f"\n测试股票: {test_stocks}")

# 批量下载
batch_df_raw = pro.daily(
    ts_code=','.join(test_stocks),
    start_date='20260109',
    end_date='20260109'
)

print(f"\n[批量下载结果]")
print(f"总行数: {len(batch_df_raw)}")
print(f"列名: {list(batch_df_raw.columns)}")

# 检查每只股票的记录数
print(f"\n[每只股票的记录数]")
for stock in test_stocks:
    stock_df = batch_df_raw[batch_df_raw['ts_code'] == stock]
    print(f"{stock}: {len(stock_df)} 条")

# 检查是否有数据混乱
print(f"\n[数据样本]")
print(batch_df_raw[['ts_code', 'trade_date', 'open', 'close', 'vol']].head(10))

# 关键检查：查看ts_code字段的唯一值
print(f"\n[关键检查] ts_code字段的唯一值:")
unique_codes = batch_df_raw['ts_code'].unique()
print(f"唯一值数量: {len(unique_codes)}")
print(f"唯一值列表: {list(unique_codes)}")

# 检查是否存在数据混乱（同一行的数据来自不同股票）
print(f"\n[验证] 检查数据是否正确分组:")
for stock in test_stocks:
    stock_data = batch_df_raw[batch_df_raw['ts_code'] == stock]
    if len(stock_data) > 0:
        print(f"\n{stock}:")
        print(f"  open: {stock_data['open'].iloc[0]}")
        print(f"  close: {stock_data['close'].iloc[0]}")
        print(f"  vol: {stock_data['vol'].iloc[0]}")
