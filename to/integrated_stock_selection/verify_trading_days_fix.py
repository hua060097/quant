#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证持仓天数修复 - 测试交易日计算逻辑
"""

from datetime import datetime, date, timedelta
from typing import List

def add_trading_days(start_date: date, n: int, trading_dates: List[date]):
    """
    从start_date开始，计算n个交易日后的日期

    Args:
        start_date: 起始日期
        n: 要增加的交易日数量
        trading_dates: 完整的交易日期列表

    Returns:
        n个交易日后的日期，如果start_date不在trading_dates中则返回None
    """
    try:
        start_index = trading_dates.index(start_date)
        end_index = start_index + n

        # 边界检查：确保索引不超过列表长度
        if end_index < len(trading_dates):
            return trading_dates[end_index]
        else:
            # 如果超出范围，返回最后一个交易日
            print(f"[!] 到期日超出交易日期范围，使用最后一个交易日: {trading_dates[-1]}")
            return trading_dates[-1]
    except ValueError:
        # 如果买入日期不在trading_dates中，返回None
        print(f"[X] 买入日期 {start_date} 不在交易日期列表中")
        return None

def test_case_1():
    """测试案例1：周五买入，持有3个交易日"""
    print("=" * 100)
    print("测试案例1：周五买入（2025-12-05），持有3个交易日")
    print("=" * 100)

    # 构造交易日期（模拟2025年12月的交易日）
    trading_dates = [
        date(2025, 12, 1),  # 周一
        date(2025, 12, 2),  # 周二
        date(2025, 12, 3),  # 周三
        date(2025, 12, 4),  # 周四
        date(2025, 12, 5),  # 周五
        date(2025, 12, 8),  # 周一
        date(2025, 12, 9),  # 周二
        date(2025, 12, 10), # 周三
        date(2025, 12, 11), # 周四
        date(2025, 12, 12), # 周五
    ]

    buy_date = date(2025, 12, 5)  # 周五
    holding_days = 3

    # 旧逻辑（自然日）- 有BUG
    old_end_date = buy_date + timedelta(days=holding_days - 1)

    # 新逻辑（交易日）- 修复后
    new_end_date = add_trading_days(buy_date, holding_days - 1, trading_dates)

    print(f"\n[日期] 买入日期: {buy_date} (周五)")
    print(f"[配置] 持有天数配置: {holding_days}个交易日")
    print()
    print(f"[X] 旧逻辑（自然日）:")
    print(f"   到期日: {old_end_date} (周日，非交易日)")
    print(f"   问题: 2025-12-08(周一) >= 2025-12-07(周日)，提前一天卖出 [X]")
    print()
    print(f"[OK] 新逻辑（交易日）:")
    print(f"   到期日: {new_end_date} (周二)")
    print(f"   持仓时间线:")
    print(f"     - 2025-12-05 (周五): 买入日，交易日1")
    print(f"     - 2025-12-06/07 (周末): 非交易日，不计入")
    print(f"     - 2025-12-08 (周一): 交易日2")
    print(f"     - 2025-12-09 (周二): 交易日3，正确到期日 [OK]")
    print()

    expected = date(2025, 12, 9)
    if new_end_date == expected:
        print(f"[OK] 测试通过！到期日正确: {new_end_date}")
    else:
        print(f"[X] 测试失败！期望: {expected}, 实际: {new_end_date}")

def test_case_2():
    """测试案例2：周一买入，持有3个交易日"""
    print("\n" + "=" * 100)
    print("测试案例2：周一买入（2025-12-08），持有3个交易日")
    print("=" * 100)

    trading_dates = [
        date(2025, 12, 8),  # 周一
        date(2025, 12, 9),  # 周二
        date(2025, 12, 10), # 周三
        date(2025, 12, 11), # 周四
        date(2025, 12, 12), # 周五
    ]

    buy_date = date(2025, 12, 8)  # 周一
    holding_days = 3

    old_end_date = buy_date + timedelta(days=holding_days - 1)
    new_end_date = add_trading_days(buy_date, holding_days - 1, trading_dates)

    print(f"\n[日期] 买入日期: {buy_date} (周一)")
    print(f"[配置] 持有天数配置: {holding_days}个交易日")
    print()
    print(f"[X] 旧逻辑（自然日）: {old_end_date} (周三)")
    print(f"[OK] 新逻辑（交易日）: {new_end_date} (周三)")
    print()
    print(f"   持仓时间线:")
    print(f"     - 2025-12-08 (周一): 交易日1，买入日")
    print(f"     - 2025-12-09 (周二): 交易日2")
    print(f"     - 2025-12-10 (周三): 交易日3，到期日")
    print()

    expected = date(2025, 12, 10)
    if new_end_date == expected:
        print(f"[OK] 测试通过！到期日正确: {new_end_date}")
    else:
        print(f"[X] 测试失败！期望: {expected}, 实际: {new_end_date}")

def test_case_3():
    """测试案例3：跨越春节"""
    print("\n" + "=" * 100)
    print("测试案例3：春节前买入，跨越春节假期")
    print("=" * 100)

    # 模拟2025年春节前后的交易日
    trading_dates = [
        date(2025, 1, 24),  # 周五（春节前最后一天）
        date(2025, 2, 5),   # 周三（春节后第一天）
        date(2025, 2, 6),   # 周四
        date(2025, 2, 7),   # 周五
    ]

    buy_date = date(2025, 1, 24)  # 春节前最后一天
    holding_days = 3

    old_end_date = buy_date + timedelta(days=holding_days - 1)
    new_end_date = add_trading_days(buy_date, holding_days - 1, trading_dates)

    print(f"\n[日期] 买入日期: {buy_date} (春节前最后交易日)")
    print(f"[配置] 持有天数配置: {holding_days}个交易日")
    print()
    print(f"[X] 旧逻辑（自然日）: {old_end_date} (2025-01-26，春节期间，非交易日)")
    print(f"[OK] 新逻辑（交易日）: {new_end_date} (春节后第2个交易日)")
    print()
    print(f"   持仓时间线:")
    print(f"     - 2025-01-24 (周五): 交易日1，买入日")
    print(f"     - 2025-01-25~02-04 (春节假期): 非交易日，不计入")
    print(f"     - 2025-02-05 (周三): 交易日2")
    print(f"     - 2025-02-06 (周四): 交易日3，到期日")
    print()

    expected = date(2025, 2, 6)
    if new_end_date == expected:
        print(f"[OK] 测试通过！到期日正确: {new_end_date}")
    else:
        print(f"[X] 测试失败！期望: {expected}, 实际: {new_end_date}")

def test_case_4():
    """测试案例4：002565.SZ的实际案例"""
    print("\n" + "=" * 100)
    print("测试案例4：002565.SZ在2025-12-05买入")
    print("=" * 100)

    # 构造2025年12月的完整交易日
    trading_dates = [
        date(2025, 12, 1),  # 周一
        date(2025, 12, 2),  # 周二
        date(2025, 12, 3),  # 周三
        date(2025, 12, 4),  # 周四
        date(2025, 12, 5),  # 周五
        date(2025, 12, 8),  # 周一
        date(2025, 12, 9),  # 周二
        date(2025, 12, 10), # 周三
        date(2025, 12, 11), # 周四
        date(2025, 12, 12), # 周五
    ]

    buy_date = date(2025, 12, 5)  # 周五
    holding_days = 3

    old_end_date = buy_date + timedelta(days=holding_days - 1)
    new_end_date = add_trading_days(buy_date, holding_days - 1, trading_dates)

    print(f"\n股票代码: 002565.SZ")
    print(f"[日期] 买入日期: {buy_date} (周五)")
    print(f"[配置] 持有天数配置: {holding_days}个交易日")
    print()
    print(f"[X] 旧逻辑（自然日）:")
    print(f"   holding_end_date = {old_end_date} (周日)")
    print(f"   2025-12-08(周一) >= 2025-12-07(周日) → 触发卖出 [X]")
    print(f"   问题: 只持有1个交易日就卖出了！")
    print()
    print(f"[OK] 新逻辑（交易日）:")
    print(f"   holding_end_date = {new_end_date} (周二)")
    print(f"   2025-12-08(周一) < 2025-12-09(周二) → 继续持有")
    print(f"   2025-12-09(周二) >= 2025-12-09(周二) → 到期卖出 [OK]")
    print(f"   正确: 持有3个交易日（周五、周一、周二）")
    print()

    expected = date(2025, 12, 9)
    if new_end_date == expected:
        print(f"[OK] 测试通过！到期日正确: {new_end_date}")
        print(f"[OK] 002565.SZ的BUG已修复！")
    else:
        print(f"[X] 测试失败！期望: {expected}, 实际: {new_end_date}")

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("持仓天数修复验证 - 交易日计算逻辑测试")
    print("=" * 100)

    test_case_1()  # 周五买入
    test_case_2()  # 周一买入
    test_case_3()  # 跨越春节
    test_case_4()  # 002565.SZ实际案例

    print("\n" + "=" * 100)
    print("所有测试完成")
    print("=" * 100)
