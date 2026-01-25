#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试stock_trading_days字典 - 检查301551.SZ是否在其中
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
current_file_dir = Path(__file__).parent
sys.path.append(str(current_file_dir))

def debug_stock_trading_days():
    """调试stock_trading_days字典"""

    # 目标股票和日期
    target_stock = "301551.SZ"
    target_date = "20240926"

    print("=" * 80)
    print(f"调试stock_trading_days: 股票 {target_stock}")
    print("=" * 80)

    # 1. 手动复现_build_new_stock_list_for_date的逻辑
    print(f"\n1. 手动复现新股识别逻辑:")

    try:
        from main import _get_previous_trading_dates

        # 获取历史交易日
        trading_dates = _get_previous_trading_dates(target_date, 30)
        print(f"   历史交易日数量: {len(trading_dates)}")
        print(f"   前5个交易日: {trading_dates[:5]}")

        # 收集所有股票的交易天数
        features_dir = Path(__file__).parent.parent / "local_data" / "features" / "daily"
        stock_trading_days = {}

        print(f"   开始分析历史交易日...")
        for i, date_str in enumerate(trading_dates):
            file_path = features_dir / f"{date_str}.parquet"
            if file_path.exists():
                try:
                    daily_data = pd.read_parquet(file_path)

                    # 获取有有效成交量的股票
                    valid_stocks = daily_data[
                        (~np.isnan(daily_data['volume'])) & (daily_data['volume'] > 0)
                    ]['stock_code'].unique()

                    # 检查目标股票是否在当日数据中
                    target_in_date = target_stock in valid_stocks
                    if i < 10:  # 只显示前10天的检查结果
                        print(f"     {date_str}: 包含目标股票且有有效成交量 = {target_in_date}")

                    # 统计所有股票的交易天数
                    for stock_code in valid_stocks:
                        if stock_code not in stock_trading_days:
                            stock_trading_days[stock_code] = 0
                        stock_trading_days[stock_code] += 1

                except Exception as e:
                    print(f"     {date_str}: 读取失败 - {e}")
            else:
                print(f"     {date_str}: 文件不存在")

        print(f"   总统计股票数: {len(stock_trading_days)}")
        print(f"   目标股票在字典中: {target_stock in stock_trading_days}")

        if target_stock in stock_trading_days:
            target_days = stock_trading_days[target_stock]
            print(f"   目标股票交易天数: {target_days}")
            is_new_stock = target_days < 20
            print(f"   应该被标记为新股: {is_new_stock}")
        else:
            print(f"   目标股票不在字典中！这意味着什么？")

        # 2. 检查新股识别结果
        new_stocks = {
            stock_code for stock_code, days in stock_trading_days.items()
            if days < 20
        }

        print(f"\n2. 新股识别结果:")
        print(f"   识别出的新股数量: {len(new_stocks)}")
        print(f"   目标股票是否在新股中: {target_stock in new_stocks}")

        # 显示部分交易天数的股票
        print(f"\n3. 按交易天数分组的股票统计:")
        from collections import defaultdict
        days_groups = defaultdict(list)
        for stock_code, days in stock_trading_days.items():
            days_groups[days].append(stock_code)

        for days in sorted(days_groups.keys())[:10]:  # 显示前10个交易天数
            stocks_count = len(days_groups[days])
            if days == 0:
                print(f"   {days}天: {stocks_count}只股票")
            else:
                print(f"   {days}天: {stocks_count}只股票")

        # 检查是否有0天交易的股票
        if 0 in days_groups:
            zero_day_stocks = days_groups[0]
            print(f"\n4. 0天交易的股票（前10只）:")
            for stock in zero_day_stocks[:10]:
                print(f"   {stock}")

        # 5. 检查目标股票在哪些日期出现过（即使成交量无效）
        print(f"\n5. 检查目标股票在历史日期的出现情况:")
        target_appearances = []

        for date_str in trading_dates[:10]:  # 只检查前10天
            file_path = features_dir / f"{date_str}.parquet"
            if file_path.exists():
                try:
                    daily_data = pd.read_parquet(file_path)
                    target_data = daily_data[daily_data['stock_code'] == target_stock]
                    if len(target_data) > 0:
                        volume = target_data['volume'].iloc[0]
                        target_appearances.append((date_str, volume))
                        print(f"     {date_str}: 存在，成交量={volume}")
                    else:
                        print(f"     {date_str}: 不存在")
                except Exception as e:
                    print(f"     {date_str}: 读取失败 - {e}")

        print(f"   目标股票出现次数: {len(target_appearances)}")

    except Exception as e:
        print(f"   调试失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 80)
    print(f"调试完成")
    print("=" * 80)

if __name__ == "__main__":
    debug_stock_trading_days()