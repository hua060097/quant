"""
简单测试：验证 ts.pro_bar() 日期参数是否有效
查看返回数据的实际日期分布
"""

import tushare as ts
import pandas as pd

# 初始化（与 download_daily.py 完全相同）
print("初始化 Tushare...")
pro = ts.pro_api()
print("Tushare 初始化成功")

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

try:
    df_single = ts.pro_bar(
        ts_code=stock1,
        adj='qfq',
        start_date=start_date,
        end_date=end_date
    )

    print(f"返回行数: {len(df_single)}")

    if len(df_single) > 0:
        print(f"日期范围: {df_single['trade_date'].min()} ~ {df_single['trade_date'].max()}")
        print(f"日期唯一值数量: {df_single['trade_date'].nunique()}")
        print(f"\n所有日期分布:")
        date_counts = df_single.groupby('trade_date').size()
        print(date_counts)

        # 显示前10条和后10条
        print(f"\n前10条数据:")
        print(df_single[['trade_date', 'open', 'close']].head(10))
        print(f"\n后10条数据:")
        print(df_single[['trade_date', 'open', 'close']].tail(10))

        # 检查是否有超出指定日期的数据
        out_of_range = df_single[
            (df_single['trade_date'] < start_date) |
            (df_single['trade_date'] > end_date)
        ]
        if len(out_of_range) > 0:
            print(f"\n[警告] 发现 {len(out_of_range)} 条超出指定日期的数据!")
            print(f"超出日期的样本:")
            print(out_of_range[['trade_date', 'open', 'close']].head(10))

except Exception as e:
    print(f"错误: {e}")
    df_single = None

# 测试2：批量下载（2个股票）
print(f"\n" + "=" * 80)
print(f"测试2: 批量下载 ({stock1}, {stock2})")
print("=" * 80)

try:
    df_batch = ts.pro_bar(
        ts_code=f"{stock1},{stock2}",
        adj='qfq',
        start_date=start_date,
        end_date=end_date
    )

    print(f"返回行数: {len(df_batch)}")

    if len(df_batch) > 0:
        print(f"日期范围: {df_batch['trade_date'].min()} ~ {df_batch['trade_date'].max()}")
        print(f"日期唯一值数量: {df_batch['trade_date'].nunique()}")

        # 每只股票的统计
        print(f"\n每只股票的记录数:")
        for stock in [stock1, stock2]:
            stock_data = df_batch[df_batch['ts_code'] == stock]
            print(f"  {stock}: {len(stock_data)} 条")
            if len(stock_data) > 0:
                dates = stock_data['trade_date'].unique()
                print(f"    日期: {list(dates)}")

        # 显示前20条数据
        print(f"\n前20条数据:")
        print(df_batch[['ts_code', 'trade_date', 'open', 'close']].head(20))

        # 检查是否有超出指定日期的数据
        out_of_range = df_batch[
            (df_batch['trade_date'] < start_date) |
            (df_batch['trade_date'] > end_date)
        ]
        if len(out_of_range) > 0:
            print(f"\n[警告] 发现 {len(out_of_range)} 条超出指定日期的数据!")
            print(f"超出日期的样本:")
            print(out_of_range[['ts_code', 'trade_date', 'open', 'close']].head(10))

except Exception as e:
    print(f"错误: {e}")
    df_batch = None

# 结论
print(f"\n" + "=" * 80)
print(f"结论:")
print("=" * 80)

if df_single is not None:
    print(f"\n单个股票 ({stock1}):")
    print(f"  返回行数: {len(df_single)}")
    print(f"  日期数: {df_single['trade_date'].nunique()}")

    # 判断日期参数是否生效
    single_dates = set(df_single['trade_date'].unique())
    expected_dates = {'20240103', '20240104', '20240105'}

    if single_dates.issubset(expected_dates) and len(df_single) <= 10:
        print(f"  [正常] 日期参数生效，只返回指定日期的数据")
    elif len(single_dates) > 100:
        print(f"  [BUG] 日期参数失效！返回了 {len(single_dates)} 天的数据")
        print(f"  说明: start_date/end_date 参数在单个股票下载时无效")
    else:
        print(f"  [部分异常] 返回了 {len(single_dates)} 天的数据")

if df_batch is not None:
    print(f"\n批量下载 ({stock1}, {stock2}):")
    print(f"  返回行数: {len(df_batch)}")
    print(f"  日期数: {df_batch['trade_date'].nunique()}")

    # 判断日期参数是否生效
    batch_dates = set(df_batch['trade_date'].unique())
    expected_dates = {'20240103', '20240104', '20240105'}

    if batch_dates.issubset(expected_dates) and len(df_batch) <= 20:
        print(f"  [正常] 日期参数生效，只返回指定日期的数据")
    elif len(batch_dates) > 100:
        print(f"  [BUG] 日期参数失效！返回了 {len(batch_dates)} 天的数据")
        print(f"  说明: start_date/end_date 参数在批量下载时无效")
    else:
        print(f"  [部分异常] 返回了 {len(batch_dates)} 天的数据")

print(f"\n" + "=" * 80)
