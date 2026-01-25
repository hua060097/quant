"""
简单测试：对比批量下载和逐股票下载的差异

使用2024年的真实数据测试
"""

import tushare as ts
import pandas as pd

# 初始化
pro = ts.pro_api()

print("=" * 80)
print("批量下载 vs 逐股票下载对比测试")
print("=" * 80)

# 测试参数：使用2024年的真实数据
test_stocks = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH"]  # 4只股票
start_date = "20240101"
end_date = "20240110"  # 10天

print(f"\n测试参数:")
print(f"  股票: {test_stocks}")
print(f"  日期: {start_date} ~ {end_date} (10天)")
print(f"  预期行数: {len(test_stocks)} × 10 = {len(test_stocks) * 10}")

# 测试1：批量下载
print(f"\n" + "=" * 80)
print(f"[测试1] 批量下载")
print("=" * 80)

try:
    batch_df = ts.pro_bar(
        ts_code=','.join(test_stocks),
        adj='qfq',
        start_date=start_date,
        end_date=end_date
    )

    if batch_df is not None and len(batch_df) > 0:
        print(f"  返回行数: {len(batch_df)}")
        print(f"  ts_code唯一值: {batch_df['ts_code'].nunique()}")

        if 'trade_date' in batch_df.columns:
            date_range = f"{batch_df['trade_date'].min()} ~ {batch_df['trade_date'].max()}"
            unique_dates = batch_df['trade_date'].nunique()
            print(f"  trade_date范围: {date_range}")
            print(f"  trade_date唯一值: {unique_dates}")

            # 分析每只股票的数据
            print(f"\n  每只股票的记录数:")
            for stock in test_stocks:
                count = len(batch_df[batch_df['ts_code'] == stock])
                print(f"    {stock}: {count} 条")

            # 检查日期分布
            print(f"\n  日期分布（所有股票）:")
            date_counts = batch_df.groupby('trade_date').size()
            print(date_counts)

        # 检查是否有问题
        if len(batch_df) > len(test_stocks) * 10 * 2:
            print(f"\n  [警告] 数据量异常！")
            print(f"    预期: {len(test_stocks) * 10} 行")
            print(f"    实际: {len(batch_df)} 行")
            print(f"    比值: {len(batch_df) / (len(test_stocks) * 10):.2f}x")
        else:
            print(f"\n  [OK] 数据量正常")

except Exception as e:
    print(f"  [错误] {e}")
    batch_df = None

# 测试2：逐股票下载
print(f"\n" + "=" * 80)
print(f"[测试2] 逐股票下载（对比）")
print("=" * 80)

individual_list = []
for stock in test_stocks:
    try:
        df = ts.pro_bar(
            ts_code=stock,
            adj='qfq',
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and len(df) > 0:
            print(f"  {stock}: {len(df)} 条")
            individual_list.append(df)
        else:
            print(f"  {stock}: 0 条（无数据）")
    except Exception as e:
        print(f"  {stock}: 错误 - {e}")

if individual_list:
    individual_df = pd.concat(individual_list, ignore_index=True)
    print(f"\n  总行数: {len(individual_df)}")
    print(f"  预期: {len(test_stocks) * 10} = {len(test_stocks) * 10} 行")

# 结论
print(f"\n" + "=" * 80)
print(f"[结论]")
print("=" * 80)

if batch_df is not None and individual_list:
    batch_rows = len(batch_df)
    individual_rows = len(individual_df)

    print(f"  批量下载: {batch_rows} 行")
    print(f"  逐个下载: {individual_rows} 行")

    if batch_rows == individual_rows:
        print(f"\n  [结果] 两种方式返回的数据量一致，批量下载正常")
    elif batch_rows > individual_rows * 2:
        print(f"\n  [结果] 批量下载返回了 {batch_rows / individual_rows:.2f}x 的数据！")
        print(f"  [问题] 批量下载存在严重bug，返回了过多数据")
        print(f"  [原因] 可能是 start_date/end_date 参数在批量模式下失效")
    else:
        print(f"\n  [结果] 数据量有差异，需要进一步分析")
