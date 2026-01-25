#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版验证脚本 - 持仓天数修复
"""

from datetime import datetime, date, timedelta
from typing import List

def add_trading_days(start_date: date, n: int, trading_dates: List[date]):
    """从start_date开始，计算n个交易日后的日期"""
    try:
        start_index = trading_dates.index(start_date)
        end_index = start_index + n

        if end_index < len(trading_dates):
            return trading_dates[end_index]
        else:
            return trading_dates[-1]
    except ValueError:
        return None

def main():
    print("=" * 80)
    print("持仓天数修复验证")
    print("=" * 80)

    # 测试案例：002565.SZ 在 2025-12-05（周五）买入
    trading_dates = [
        date(2025, 12, 1),  # 周一
        date(2025, 12, 2),  # 周二
        date(2025, 12, 3),  # 周三
        date(2025, 12, 4),  # 周四
        date(2025, 12, 5),  # 周五
        date(2025, 12, 8),  # 周一
        date(2025, 12, 9),  # 周二
        date(2025, 12, 10), # 周三
    ]

    buy_date = date(2025, 12, 5)  # 周五
    holding_days = 3

    # 旧逻辑
    old_end_date = buy_date + timedelta(days=holding_days - 1)

    # 新逻辑
    new_end_date = add_trading_days(buy_date, holding_days - 1, trading_dates)

    print("\n案例：002565.SZ 在 2025-12-05（周五）买入，持有3个交易日")
    print("-" * 80)
    print(f"买入日期: {buy_date.strftime('%Y-%m-%d (%A)')}")
    print(f"持有配置: {holding_days}个交易日")
    print()
    print("旧逻辑（自然日）:")
    print(f"  到期日: {old_end_date.strftime('%Y-%m-%d (%A)')}")
    print(f"  问题: 2025-12-08(周一) >= 2025-12-07(周日)，提前卖出！")
    print()
    print("新逻辑（交易日）:")
    print(f"  到期日: {new_end_date.strftime('%Y-%m-%d (%A)')}")
    print(f"  时间线:")
    print(f"    1. 2025-12-05 (周五) - 买入日，交易日1")
    print(f"    2. 2025-12-06/07 (周末) - 非交易日")
    print(f"    3. 2025-12-08 (周一) - 交易日2")
    print(f"    4. 2025-12-09 (周二) - 交易日3，到期卖出")
    print()

    if new_end_date == date(2025, 12, 9):
        print("[OK] 测试通过！修复成功")
    else:
        print("[X] 测试失败！")

    print("=" * 80)

if __name__ == "__main__":
    main()
