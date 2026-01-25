"""
诊断 ts.pro_bar 批量下载的问题

分析为什么批量下载会返回重复数据
"""

import tushare as ts
import pandas as pd

# 设置token
ts.set_token('2d87b56abd0b44349cb9237945139f8e4b2298e1fe9a0f6f3c8a6e2b')
pro = ts.pro_api()

print("=" * 80)
print("诊断 ts.pro_bar 批量下载问题")
print("=" * 80)

# 测试1：下载1只股票，1天数据
print("\n[测试1] 1只股票 x 1天")
print("-" * 80)
df1 = pro.daily(
    ts_code='000001.SZ',
    start_date='20260109',
    end_date='20260109'
)
print(f"返回行数: {len(df1)}")
print(f"ts_code唯一值: {df1['ts_code'].nunique()}")
print(f"trade_date唯一值: {df1['trade_date'].nunique()}")
if len(df1) > 1:
    print(f"警告: 预期1行，实际{len(df1)}行")
    print(df1)

# 测试2：下载2只股票，1天数据
print("\n[测试2] 2只股票 x 1天（逗号分隔）")
print("-" * 80)
df2 = pro.daily(
    ts_code='000001.SZ,000002.SZ',
    start_date='20260109',
    end_date='20260109'
)
print(f"返回行数: {len(df2)}")
print(f"ts_code唯一值: {df2['ts_code'].nunique()}")
print(f"预期行数: 2")
if len(df2) > 2:
    print(f"警告: 预期2行，实际{len(df2)}行")
    print(f"\n重复分析:")
    for code in ['000001.SZ', '000002.SZ']:
        count = (df2['ts_code'] == code).sum()
        print(f"  {code}: {count} 条记录")

# 测试3：下载100只股票，1天数据（模拟实际场景）
print("\n[测试3] 100只股票 x 1天")
print("-" * 80)

# 获取前100只股票
stock_list = []
for i in range(1, 101):
    if i <= 30:
        stock_list.append(f"{i:06d}.SZ")
    else:
        stock_list.append(f"{i-30+600000:06d}.SH")

test_codes = ','.join(stock_list[:10])  # 只测试10只
print(f"测试股票: {test_codes}")

df3 = pro.daily(
    ts_code=test_codes,
    start_date='20260109',
    end_date='20260109'
)
print(f"返回行数: {len(df3)}")
print(f"ts_code唯一值: {df3['ts_code'].nunique()}")
print(f"预期行数: 10")
if len(df3) > 10:
    print(f"警告: 预期10行，实际{len(df3)}行（{len(df3)/10:.1f}倍）")
    print(f"\nts_code分布:")
    print(df3['ts_code'].value_counts().head(10))

# 测试4：下载1只股票，9天数据
print("\n[测试4] 1只股票 x 9天（20260101-20260109）")
print("-" * 80)
df4 = pro.daily(
    ts_code='000001.SZ',
    start_date='20260101',
    end_date='20260109'
)
print(f"返回行数: {len(df4)}")
print(f"预期行数: 9")
print(f"trade_date范围: {df4['trade_date'].min()} ~ {df4['trade_date'].max()}")
if len(df4) > 9:
    print(f"警告: 预期9行，实际{len(df4)}行")

# 测试5：检查 adj 参数的影响
print("\n[测试5] 测试 adj 参数（前复权）")
print("-" * 80)
df5 = pro.daily(
    ts_code='000001.SZ,000002.SZ',
    start_date='20260109',
    end_date='20260109'
)
print(f"adj=None (默认): {len(df5)} 行")

df5_qfq = pro.daily(
    ts_code='000001.SZ,000002.SZ',
    start_date='20260109',
    end_date='20260109',
    adj='qfq'
)
print(f"adj='qfq' (前复权): {len(df5_qfq)} 行")

# 结论
print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

if len(df2) == 2 and len(df3) == 10:
    print("[结论] 批量下载正常，没有重复数据")
    print("问题可能在于 download_daily.py 的其他逻辑")
else:
    print("[结论] ts.pro_bar 批量下载存在重复数据问题！")
    print("建议: 回退到逐股票下载模式")
