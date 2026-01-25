"""
关键诊断：测试 ts.pro_bar() 下载单个股票时，日期参数是否有效

验证：
1. 单个股票下载时，start_date/end_date 是否生效
2. 批量下载时，start_date/end_date 是否生效
3. 对比两者的差异

这将帮助我们确定是 API 本身的 bug，还是批量模式的 bug
"""

import tushare as ts
import pandas as pd
from datetime import datetime, timedelta

# 初始化
pro = ts.pro_api()

print("=" * 100)
print("关键诊断：ts.pro_bar() 日期参数有效性测试")
print("=" * 100)

# 测试股票
test_stock = "000001.SZ"  # 平安银行
start_date = "20240101"
end_date = "20240105"

print(f"\n[测试参数]")
print(f"  股票: {test_stock}")
print(f"  日期范围: {start_date} ~ {end_date} (5天)")
print(f"  预期行数: 5 行")

# ============================================================================
# 测试1：单个股票下载（指定日期范围）
# ============================================================================
print(f"\n" + "=" * 100)
print(f"[测试1] 单个股票下载 - 指定日期范围")
print("=" * 100)

df_single = ts.pro_bar(
    ts_code=test_stock,
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"\n[结果]")
print(f"  返回行数: {len(df_single)}")
print(f"  预期行数: 5")

if 'trade_date' in df_single.columns:
    date_range = f"{df_single['trade_date'].min()} ~ {df_single['trade_date'].max()}"
    unique_dates = df_single['trade_date'].nunique()
    print(f"  trade_date范围: {date_range}")
    print(f"  trade_date唯一值: {unique_dates}")

    # 检查是否超出指定日期范围
    out_of_range = df_single[
        (df_single['trade_date'] < start_date) |
        (df_single['trade_date'] > end_date)
    ]

    if len(out_of_range) > 0:
        print(f"  [警告] 发现 {len(out_of_range)} 条超出日期范围的数据！")
        print(f"  超出日期示例: {out_of_range['trade_date'].unique()[:5]}")
    else:
        print(f"  [OK] 所有数据都在指定日期范围内")

    # 显示数据样本
    print(f"\n  数据样本:")
    print(df_single[['ts_code', 'trade_date', 'open', 'close']].head(10))

# 分析
if len(df_single) == 5:
    print(f"\n  [结论] 单个股票下载时，日期参数正常工作！✅")
elif len(df_single) > 100:
    print(f"\n  [结论] 单个股票下载时，日期参数失效！返回了 {len(df_single)} 行而不是 5 行！❌")
    print(f"  [问题] API 本身存在 bug，start_date/end_date 参数无效")
else:
    print(f"\n  [结论] 返回 {len(df_single)} 行，可能有股票停牌等原因")

# ============================================================================
# 测试2：单个股票下载（不指定日期范围）
# ============================================================================
print(f"\n" + "=" * 100)
print(f"[测试2] 单个股票下载 - 不指定日期范围（获取全部历史）")
print("=" * 100)

df_all = ts.pro_bar(
    ts_code=test_stock,
    adj='qfq'
    # 注意：不指定 start_date 和 end_date
)

print(f"\n[结果]")
print(f"  返回行数: {len(df_all)}")

if 'trade_date' in df_all.columns:
    date_range = f"{df_all['trade_date'].min()} ~ {df_all['trade_date'].max()}"
    unique_dates = df_all['trade_date'].nunique()
    print(f"  trade_date范围: {date_range}")
    print(f"  trade_date唯一值: {unique_dates}")
    print(f"  数据覆盖天数: {(pd.to_datetime(df_all['trade_date'].max()) - pd.to_datetime(df_all['trade_date'].min())).days + 1} 天")

print(f"\n  [结论] 不指定日期范围时，API 返回了 {len(df_all)} 行历史数据")

# ============================================================================
# 测试3：对比指定日期和不指定日期的差异
# ============================================================================
print(f"\n" + "=" * 100)
print(f"[测试3] 对比分析")
print("=" * 100)

if 'trade_date' in df_single.columns and 'trade_date' in df_all.columns:
    single_dates = set(df_single['trade_date'].unique())
    all_dates = set(df_all['trade_date'].unique())

    print(f"\n  指定日期范围({start_date}~{end_date})获取到的日期:")
    print(f"    {sorted(single_dates)}")

    print(f"\n  全部历史数据中，该日期范围的数据:")
    all_in_range = df_all[
        (df_all['trade_date'] >= start_date) &
        (df_all['trade_date'] <= end_date)
    ]
    print(f"    行数: {len(all_in_range)}")
    print(f"    日期: {sorted(all_in_range['trade_date'].unique())}")

    # 验证
    if len(df_single) == len(all_in_range):
        print(f"\n  [验证通过] 指定日期范围的数据量正确")
    elif len(df_single) > len(all_in_range):
        print(f"\n  [异常] 指定日期范围返回了更多数据！")
        print(f"    指定日期: {len(df_single)} 行")
        print(f"    全部数据中该范围: {len(all_in_range)} 行")
        print(f"    差异: {len(df_single) - len(all_in_range)} 行")

# ============================================================================
# 测试4：批量下载（2只股票）
# ============================================================================
print(f"\n" + "=" * 100)
print(f"[测试4] 批量下载（2只股票）")
print("=" * 100)

test_stocks = [test_stock, "000002.SZ"]

df_batch = ts.pro_bar(
    ts_code=','.join(test_stocks),
    adj='qfq',
    start_date=start_date,
    end_date=end_date
)

print(f"\n[结果]")
print(f"  返回行数: {len(df_batch)}")
print(f"  预期行数: 2 × 5 = 10")

if 'trade_date' in df_batch.columns:
    date_range = f"{df_batch['trade_date'].min()} ~ {df_batch['trade_date'].max()}"
    unique_dates = df_batch['trade_date'].nunique()
    print(f"  trade_date范围: {date_range}")
    print(f"  trade_date唯一值: {unique_dates}")

    # 每只股票的数据
    print(f"\n  每只股票的记录数:")
    for stock in test_stocks:
        count = len(df_batch[df_batch['ts_code'] == stock])
        print(f"    {stock}: {count} 条")

    # 检查超出范围
    out_of_range = df_batch[
        (df_batch['trade_date'] < start_date) |
        (df_batch['trade_date'] > end_date)
    ]

    if len(out_of_range) > 0:
        print(f"\n  [警告] 发现 {len(out_of_range)} 条超出日期范围的数据！")
    else:
        print(f"\n  [OK] 所有数据都在指定日期范围内")

# 分析批量下载
if len(df_batch) == 10 or len(df_batch) < 20:  # 允许少量停牌
    print(f"\n  [结论] 批量下载时，日期参数正常工作！✅")
elif len(df_batch) > 100:
    print(f"\n  [结论] 批量下载时，日期参数失效！返回了 {len(df_batch)} 行而不是 10 行！❌")
    print(f"  [问题] 批量模式存在 bug")
else:
    print(f"\n  [结论] 返回 {len(df_batch)} 行")

# ============================================================================
# 最终结论
# ============================================================================
print(f"\n" + "=" * 100)
print(f"[最终诊断结论]")
print("=" * 100)

single_ok = (len(df_single) <= 10)  # 单股票下载正常
batch_ok = (len(df_batch) <= 20)   # 批量下载正常

print(f"\n单个股票下载: {'✅ 正常' if single_ok else '❌ 日期参数失效'}")
print(f"  返回行数: {len(df_single)}")
print(f"  预期行数: 5")

print(f"\n批量下载: {'✅ 正常' if batch_ok else '❌ 日期参数失效'}")
print(f"  返回行数: {len(df_batch)}")
print(f"  预期行数: 10")

if not single_ok and not batch_ok:
    print(f"\n[诊断结果] API 本身存在 bug！")
    print(f"  说明: ts.pro_bar() 的 start_date/end_date 参数在所有模式下都失效")
    print(f"  影响: 无论单个还是批量下载，都会返回全部历史数据")
    print(f"  建议: 向 Tushare 官方反馈，或使用 pro.daily() API")
elif single_ok and not batch_ok:
    print(f"\n[诊断结果] 批量模式存在 bug！")
    print(f"  说明: ts.pro_bar() 只在批量模式下日期参数失效")
    print(f"  影响: 使用逗号分隔多股票时会出现问题")
    print(f"  建议: 回退到逐股票下载，或使用 pro.daily() API")
elif single_ok and batch_ok:
    print(f"\n[诊断结果] API 正常工作！")
    print(f"  说明: 日期参数在单个和批量模式下都正常")
    print(f"  建议: 检查用户的实际调用参数和返回数据")
else:
    print(f"\n[诊断结果] 需要进一步分析")

# ============================================================================
# 详细数据样本（如果有问题）
# ============================================================================
if not single_ok or not batch_ok:
    print(f"\n" + "=" * 100)
    print(f"[问题详情]")
    print("=" * 100)

    if not single_ok:
        print(f"\n单个股票下载异常样本:")
        print(f"  前20行:")
        print(df_single[['ts_code', 'trade_date', 'open', 'close']].head(20))
        print(f"  最后10行:")
        print(df_single[['ts_code', 'trade_date', 'open', 'close']].tail(10))

    if not batch_ok:
        print(f"\n批量下载异常样本:")
        print(f"  前20行:")
        print(df_batch[['ts_code', 'trade_date', 'open', 'close']].head(20))
        print(f"  最后10行:")
        print(df_batch[['ts_code', 'trade_date', 'open', 'close']].tail(10))

print(f"\n" + "=" * 100)
