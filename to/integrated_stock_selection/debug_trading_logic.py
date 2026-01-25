#!/usr/bin/env python3
"""
调试卖出时机问题
"""
import pandas as pd
from datetime import datetime, timedelta

# 模拟trading_dates数组
trading_dates = [
    datetime(2025, 7, 1),
    datetime(2025, 7, 2),
    datetime(2025, 7, 3),
    datetime(2025, 7, 4),
]

# 模拟交易逻辑
selection_date = datetime(2025, 7, 1)  # T日：选股
buy_date = datetime(2025, 7, 2)       # T+1日：买入
holding_days = 2

# 按照代码逻辑计算
buy_idx = trading_dates.index(buy_date)
sell_idx = min(buy_idx + holding_days - 1, len(trading_dates) - 1)
sell_date = trading_dates[sell_idx]

print(f"选股日 (T): {selection_date.strftime('%Y-%m-%d')}")
print(f"买入日 (T+1): {buy_date.strftime('%Y-%m-%d')}")
print(f"持仓天数: {holding_days}")
print(f"买入索引: {buy_idx}")
print(f"卖出索引计算: buy_idx + holding_days - 1 = {buy_idx} + {holding_days} - 1 = {sell_idx}")
print(f"计算卖出日: {sell_date.strftime('%Y-%m-%d')}")

print("\n预期分析:")
print("T+1买入，T+2收盘卖出应该是：")
print("- T日: 2025-07-01 (选股)")
print("- T+1日: 2025-07-02 (买入)")
print("- T+2日: 2025-07-03 (卖出)")

print(f"\n实际计算结果: {sell_date.strftime('%Y-%m-%d')}")
print(f"是否正确: {'✓' if sell_date.strftime('%Y-%m-%d') == '2025-07-03' else '✗'}")

if sell_date.strftime('%Y-%m-%d') != '2025-07-03':
    print("\n❌ 计算逻辑有误!")
    print("正确逻辑应该是: sell_idx = buy_idx + (holding_days - 1)")
    correct_sell_idx = buy_idx + (holding_days - 1)
    correct_sell_date = trading_dates[correct_sell_idx]
    print(f"修正后卖出日: {correct_sell_date.strftime('%Y-%m-%d')}")