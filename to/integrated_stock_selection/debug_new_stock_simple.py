#!/usr/bin/env python3
"""调试001279.SZ新股识别问题 - 简化版本"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_new_stock_simple():
    """简化版调试脚本"""
    print("=== 调试001279.SZ新股识别问题 ===\n")

    # 检查交易日历文件
    print("=== 检查交易日历文件 ===")
    calendar_path = Path("local_data/calendar/trading_calendar.parquet")

    if calendar_path.exists():
        trading_calendar = pd.read_parquet(calendar_path)
        print(f"✅ 交易日历文件存在，包含 {len(trading_calendar)} 行")
        print(f"交易日历列: {list(trading_calendar.columns)}")

        if 'date_str' in trading_calendar.columns:
            all_trading_dates = trading_calendar['date_str'].tolist()
            print(f"总交易日数: {len(all_trading_dates)}")
            print(f"第一个交易日: {all_trading_dates[0]}")
            print(f"最后一个交易日: {all_trading_dates[-1]}")

            # 检查关键日期
            key_dates = ['20241011', '20241115']
            for date_str in key_dates:
                if date_str in all_trading_dates:
                    idx = all_trading_dates.index(date_str)
                    print(f"✅ {date_str} 在交易日历中 (索引: {idx})")
                else:
                    print(f"❌ {date_str} 不在交易日历中")

                    # 查找最接近的日期
                    close_dates = [d for d in all_trading_dates if d.startswith('202410') or d.startswith('202411')]
                    print(f"   接近的日期: {close_dates[:10]}")
        else:
            print("❌ 交易日历文件没有date_str列")
    else:
        print("❌ 交易日历文件不存在")

    # 检查实际的因子数据
    print(f"\n=== 检查因子数据 ===")

    # 检查2024-11-15的因子文件
    factor_file = Path("local_data/features/daily/20241114.parquet")  # 注意：11月15日计算的是11月14日的因子

    if factor_file.exists():
        factor_data = pd.read_parquet(factor_file)
        print(f"✅ 20241114因子文件存在，包含 {len(factor_data)} 行")
        print(f"因子数据列: {list(factor_data.columns)}")

        # 检查是否有001279.SZ
        if 'stock_code' in factor_data.columns:
            stock_001279 = factor_data[factor_data['stock_code'] == '001279.SZ']
            print(f"001279.SZ在20241114的数据: {len(stock_001279)} 行")

            if len(stock_001279) > 0:
                print("001279.SZ因子数据:")
                for _, row in stock_001279.iterrows():
                    print(f"  {row}")
            else:
                print("❌ 001279.SZ在20241114的因子数据中不存在")

        # 检查所有股票代码
        if 'stock_code' in factor_data.columns:
            all_stocks = factor_data['stock_code'].unique().tolist()
            print(f"20241114因子文件中的股票数量: {len(all_stocks)}")

            # 查找是否包含类似001279的股票
            similar_stocks = [s for s in all_stocks if '001' in s and len(s) == 9]
            print(f"类似001xxx的股票: {similar_stocks[:10]}")

    else:
        print("❌ 20241114因子文件不存在")

    # 检查原始日线数据
    print(f"\n=== 检查原始日线数据 ===")
    daily_files = [
        "local_data/daily/202410/202410.parquet",
        "local_data/daily/202411/202411.parquet"
    ]

    for daily_file in daily_files:
        daily_path = Path(daily_file)
        if daily_path.exists():
            daily_data = pd.read_parquet(daily_path)
            print(f"✅ {daily_file} 存在，包含 {len(daily_data)} 行")

            # 检查001279.SZ
            if 'symbol' in daily_data.columns:
                stock_001279 = daily_data[daily_data['symbol'] == '001279.SZ']
                print(f"001279.SZ在{daily_file}中的数据: {len(stock_001279)} 行")

                if len(stock_001279) > 0:
                    print("001279.SZ日线数据样例:")
                    for _, row in stock_001279.head(5).iterrows():
                        print(f"  日期: {row.get('date', 'N/A')}, 涨跌幅: {row.get('pct_chg', 'N/A')}%")
                else:
                    print("❌ 001279.SZ在日线数据中不存在")

            # 检查列名
            print(f"  列名: {list(daily_data.columns)}")
        else:
            print(f"❌ {daily_file} 不存在")

    # 模拟新股过滤逻辑的关键步骤
    print(f"\n=== 手动验证新股过滤逻辑 ===")

    # 关键问题：新股过滤逻辑依赖于第一个交易日的股票列表
    # 如果001279.SZ在数据的第一个交易日就存在，那么它不会被识别为新股

    # 假设因子生成的第一个日期
    first_dates_to_check = ['20241008', '20241009', '20241010']

    for check_date in first_dates_to_check:
        factor_file = Path(f"local_data/features/daily/{check_date}.parquet")
        if factor_file.exists():
            factor_data = pd.read_parquet(factor_file)

            if 'stock_code' in factor_data.columns:
                if '001279.SZ' in factor_data['stock_code'].values:
                    print(f"❌ 问题发现: {check_date} 的因子数据中已经包含001279.SZ")
                    print(f"   这意味着新股过滤逻辑不会将其识别为新股票")
                    print(f"   因为它存在于数据的第一个交易日")
                else:
                    print(f"✅ {check_date} 的因子数据中没有001279.SZ")

if __name__ == "__main__":
    debug_new_stock_simple()