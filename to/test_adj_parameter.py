"""
测试 adj 参数的影响

对比批量下载时，不同 adj 参数的行为
"""

import tushare as ts
import pandas as pd

# 设置token
ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000')

print("=" * 80)
print("测试 adj 参数对批量下载的影响")
print("=" * 80)

test_stocks = "000001.SZ,000002.SZ"
start_date = "20260109"
end_date = "20260109"

# 测试不同的 adj 参数
for adj_value in ['qfq', 'hfq', None]:
    print(f"\n" + "=" * 80)
    print(f"测试 adj='{adj_value}'")
    print("=" * 80)

    df = ts.pro_bar(
        ts_code=test_stocks,
        adj=adj_value,
        start_date=start_date,
        end_date=end_date
    )

    if df is not None and len(df) > 0:
        print(f"返回行数: {len(df)}")

        # 每只股票的记录数
        for stock in ["000001.SZ", "000002.SZ"]:
            stock_data = df[df['ts_code'] == stock]
            print(f"\n  {stock}: {len(stock_data)} 条记录")
            if len(stock_data) > 0:
                print(f"    open价格: {list(stock_data['open'].round(2))}")
                print(f"    close价格: {list(stock_data['close'].round(2))}")

                # 如果有多条记录，检查是否所有trade_date都相同
                unique_dates = stock_data['trade_date'].nunique()
                if unique_dates == 1:
                    print(f"    trade_date: {stock_data['trade_date'].iloc[0]} (唯一)")
                else:
                    print(f"    trade_date: {sorted(stock_data['trade_date'].unique())}")

        # 显示所有数据
        print(f"\n  所有数据:")
        print(df[['ts_code', 'trade_date', 'open', 'close']])

print(f"\n" + "=" * 80)
