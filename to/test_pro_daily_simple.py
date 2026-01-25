"""
使用 pro.daily() API 进行测试
"""

import tushare as ts
import pandas as pd

# 初始化
print("初始化 Tushare...")
pro = ts.pro_api()
print("Tushare 初始化成功\n")

# 测试参数
stock1 = "000001.SZ"
stock2 = "000002.SZ"
start_date = "20240103"
end_date = "20240105"

print("=" * 80)
print("测试 pro.daily() API")
print("=" * 80)
print(f"股票: {stock1}")
print(f"日期: {start_date} ~ {end_date} (3天)")

# 测试单个股票
print(f"\n测试1: 单个股票下载")
try:
    df_single = pro.daily(
        ts_code=stock1,
        start_date=start_date,
        end_date=end_date
    )

    if df_single is not None and len(df_single) > 0:
        print(f"返回行数: {len(df_single)}")
        print(f"日期范围: {df_single['trade_date'].min()} ~ {df_single['trade_date'].max()}")
        print(f"日期数: {df_single['trade_date'].nunique()}")
        print(f"\n数据样本:")
        print(df_single[['trade_date', 'open', 'close']])

        # 检查是否超出范围
        out = df_single[
            (df_single['trade_date'] < start_date) |
            (df_single['trade_date'] > end_date)
        ]
        if len(out) > 0:
            print(f"\n[警告] 有 {len(out)} 条超出日期范围的数据")
        else:
            print(f"\n[OK] 日期参数生效")
    else:
        print(f"返回为空")
except Exception as e:
    print(f"错误: {e}")

# 测试批量下载
print(f"\n\n测试2: 批量下载 ({stock1}, {stock2})")
try:
    df_batch = pro.daily(
        ts_code=f"{stock1},{stock2}",
        start_date=start_date,
        end_date=end_date
    )

    if df_batch is not None and len(df_batch) > 0:
        print(f"返回行数: {len(df_batch)}")
        print(f"日期范围: {df_batch['trade_date'].min()} ~ {df_batch['trade_date'].max()}")
        print(f"日期数: {df_batch['trade_date'].nunique()}")

        print(f"\n每只股票的记录数:")
        for stock in [stock1, stock2]:
            count = len(df_batch[df_batch['ts_code'] == stock])
            print(f"  {stock}: {count} 条")

        print(f"\n数据样本:")
        print(df_batch[['ts_code', 'trade_date', 'open', 'close']].head(10))

        # 检查是否超出范围
        out = df_batch[
            (df_batch['trade_date'] < start_date) |
            (df_batch['trade_date'] > end_date)
        ]
        if len(out) > 0:
            print(f"\n[警告] 有 {len(out)} 条超出日期范围的数据")
        else:
            print(f"\n[OK] 日期参数生效")
    else:
        print(f"返回为空")
except Exception as e:
    print(f"错误: {e}")

print(f"\n" + "=" * 80)
