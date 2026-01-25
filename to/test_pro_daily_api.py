"""
测试 pro.daily() API 是否支持批量下载且日期参数有效

与 ts.pro_bar() 进行对比
"""

import tushare as ts
import pandas as pd
from datetime import datetime

# 初始化
pro = ts.pro_api()

print("=" * 80)
print("测试 pro.daily() API 批量下载")
print("=" * 80)

# 测试参数：使用真实的2024年数据
test_stocks = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH"]
start_date = "20240101"
end_date = "20240105"  # 5天

print(f"\n[测试参数]")
print(f"  股票数量: {len(test_stocks)}")
print(f"  日期范围: {start_date} ~ {end_date} (5天)")
print(f"  预期行数: {len(test_stocks)} × 5 = {len(test_stocks) * 5}")

# 测试1：pro.daily() 批量下载
print(f"\n" + "=" * 80)
print(f"[测试1] pro.daily() 批量下载")
print("=" * 80)

try:
    df_daily = pro.daily(
        ts_code=','.join(test_stocks),
        start_date=start_date,
        end_date=end_date
    )

    if df_daily is not None and len(df_daily) > 0:
        print(f"\n  [成功]")
        print(f"  返回行数: {len(df_daily)}")
        print(f"  ts_code唯一值: {df_daily['ts_code'].nunique()}")
        print(f"  trade_date范围: {df_daily['trade_date'].min()} ~ {df_daily['trade_date'].max()}")
        print(f"  trade_date唯一值: {df_daily['trade_date'].nunique()}")

        # 每只股票的记录数
        print(f"\n  每只股票的记录数:")
        for stock in test_stocks:
            count = len(df_daily[df_daily['ts_code'] == stock])
            print(f"    {stock}: {count} 条")

        # 验证
        print(f"\n  [验证]")
        if len(df_daily) == len(test_stocks) * 5:
            print(f"  [OK] 数据量完全正确！")
        elif len(df_daily) < len(test_stocks) * 5 * 2:
            print(f"  [可接受] 数据量接近预期（可能有股票停牌）")
        else:
            print(f"  [异常] 数据量过大！")
            print(f"    预期: {len(test_stocks) * 5}")
            print(f"    实际: {len(df_daily)}")
            print(f"    比值: {len(df_daily) / (len(test_stocks) * 5):.2f}x")

        # 显示样本
        print(f"\n  数据样本:")
        print(df_daily[['ts_code', 'trade_date', 'open', 'close', 'vol']].head(10))

    else:
        print(f"  [失败] 返回数据为空")
        df_daily = None

except Exception as e:
    print(f"  [错误] {e}")
    import traceback
    traceback.print_exc()
    df_daily = None

# 测试2：pro.daily() 逐股票下载（对比）
print(f"\n" + "=" * 80)
print(f"[测试2] pro.daily() 逐股票下载（对比）")
print("=" * 80)

individual_list = []
for stock in test_stocks:
    try:
        df = pro.daily(
            ts_code=stock,
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and len(df) > 0:
            print(f"  {stock}: {len(df)} 条")
            individual_list.append(df)
        else:
            print(f"  {stock}: 0 条")
    except Exception as e:
        print(f"  {stock}: 错误 - {e}")

if individual_list:
    individual_df = pd.concat(individual_list, ignore_index=True)
    print(f"\n  总行数: {len(individual_df)}")
    print(f"  预期: {len(test_stocks) * 5} = {len(test_stocks) * 5}")

# 测试3：ts.pro_bar() 对比（确认问题）
print(f"\n" + "=" * 80)
print(f"[测试3] ts.pro_bar() 批量下载（对比）")
print("=" * 80)

try:
    df_bar = ts.pro_bar(
        ts_code=','.join(test_stocks),
        adj='qfq',
        start_date=start_date,
        end_date=end_date
    )

    if df_bar is not None and len(df_bar) > 0:
        print(f"\n  返回行数: {len(df_bar)}")
        print(f"  trade_date范围: {df_bar['trade_date'].min()} ~ {df_bar['trade_date'].max()}")
        print(f"  trade_date唯一值: {df_bar['trade_date'].nunique()}")

        print(f"\n  每只股票的记录数:")
        for stock in test_stocks:
            count = len(df_bar[df_bar['ts_code'] == stock])
            print(f"    {stock}: {count} 条")

        if len(df_bar) > len(test_stocks) * 5 * 10:
            print(f"\n  [确认问题] ts.pro_bar() 返回了 {len(df_bar)} 行，远超预期的 {len(test_stocks) * 5} 行！")
            print(f"  比值: {len(df_bar) / (len(test_stocks) * 5):.2f}x")

    else:
        print(f"  返回数据为空")
        df_bar = None

except Exception as e:
    print(f"  [错误] {e}")
    df_bar = None

# 结论
print(f"\n" + "=" * 80)
print(f"[结论]")
print("=" * 80)

if df_daily is not None:
    daily_rows = len(df_daily)
    print(f"\npro.daily() 批量下载:")
    print(f"  返回行数: {daily_rows}")
    print(f"  预期行数: {len(test_stocks) * 5}")

    if daily_rows == len(test_stocks) * 5 or daily_rows < len(test_stocks) * 5 * 2:
        print(f"\n[结论] pro.daily() API 支持批量下载，且日期参数正常工作！")
        print(f"[建议] 修改 download_daily.py，将 ts.pro_bar() 替换为 pro.daily()")
    else:
        print(f"\n[结论] pro.daily() 也有问题")

if df_bar is not None:
    bar_rows = len(df_bar)
    print(f"\nts.pro_bar() 批量下载:")
    print(f"  返回行数: {bar_rows}")
    print(f"  预期行数: {len(test_stocks) * 5}")

    if bar_rows > len(test_stocks) * 5 * 10:
        print(f"\n[结论] ts.pro_bar() 确实存在严重bug，返回了 {bar_rows / (len(test_stocks) * 5):.2f}x 的数据！")

print(f"\n" + "=" * 80)
