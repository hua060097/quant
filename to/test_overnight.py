"""
测试脚本：检查隔夜涨跌幅计算是否正确
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\features\daily")

# 加载3天的数据
date_list = ['20251222', '20251223', '20251224']
all_data = []
for date_str in date_list:
    file_path = DATA_DIR / f"{date_str}.parquet"
    if file_path.exists():
        df = pd.read_parquet(file_path)
        all_data.append(df)

df = pd.concat(all_data, ignore_index=True)
df = df.sort_values(['stock_code', 'date'])

# 查看几只股票的数据
print("=" * 80)
print("测试数据检查")
print("=" * 80)

# 随机选择几只股票查看
test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']

for stock_code in test_stocks:
    stock_df = df[df['stock_code'] == stock_code]
    if len(stock_df) > 0:
        print(f"\n股票: {stock_code}")
        print(stock_df[['date', 'open', 'close']].head(10))

        print(f"\n隔夜涨跌幅计算:")
        for i in range(len(stock_df) - 1):
            current_row = stock_df.iloc[i]
            next_row = stock_df.iloc[i + 1]

            close = current_row['close']
            open_next = next_row['open']
            overnight_ret = (open_next - close) / close * 100

            print(f"  {current_row['date']} -> {next_row['date']}: "
                  f"收盘={close:.2f}, 开盘={open_next:.2f}, 涨跌幅={overnight_ret:.2f}%")

# 检查所有股票的最大隔夜涨跌幅
print("\n" + "=" * 80)
print("统计所有股票的隔夜涨跌幅分布")
print("=" * 80)

all_returns = []
for stock_code in df['stock_code'].unique():
    stock_df = df[df['stock_code'] == stock_code].sort_values('date')
    for i in range(len(stock_df) - 1):
        close = stock_df.iloc[i]['close']
        open_next = stock_df.iloc[i + 1]['open']
        if close > 0 and not pd.isna(close) and not pd.isna(open_next):
            overnight_ret = (open_next - close) / close * 100
            all_returns.append({
                'stock_code': stock_code,
                'date': stock_df.iloc[i]['date'],
                'return': overnight_ret
            })

returns_df = pd.DataFrame(all_returns)
print(f"\n总共计算了 {len(returns_df)} 个隔夜涨跌幅")
print(f"\n涨跌幅统计:")
print(returns_df['return'].describe())
print(f"\n最大涨幅: {returns_df['return'].max():.2f}%")
print(f"最大跌幅: {returns_df['return'].min():.2f}%")
print(f"\n涨跌幅绝对值 > 10% 的数量: {(returns_df['return'].abs() > 10).sum()}")
print(f"涨跌幅绝对值 > 20% 的数量: {(returns_df['return'].abs() > 20).sum()}")
print(f"涨跌幅绝对值 > 25% 的数量: {(returns_df['return'].abs() > 25).sum()}")

# 显示一些涨跌幅较大的记录
if (returns_df['return'].abs() > 10).sum() > 0:
    print(f"\n涨跌幅绝对值 > 10% 的记录:")
    large_moves = returns_df[returns_df['return'].abs() > 10].sort_values('return', key=abs, ascending=False)
    for _, row in large_moves.head(20).iterrows():
        print(f"  {row['stock_code']} | {row['date']} | {row['return']:.2f}%")
