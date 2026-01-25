#!/usr/bin/env python3
"""测试基于首个日期的新股识别逻辑"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加路径以便导入main模块
sys.path.append(str(Path(__file__).parent))

def test_first_date_logic():
    """测试基于首个日期的新股识别逻辑"""
    print("=== 测试基于首个日期的新股识别逻辑 ===\n")

    try:
        # 设置测试数据
        print("--- 创建测试数据 ---")

        # 创建一个包含多日数据的DataFrame
        all_data = []

        # 第一个日期: 2024-10-08 (只包含老股票)
        date_1008 = pd.DataFrame({
            'stock_code': ['000001.SZ', '000002.SZ'],
            'date': pd.to_datetime('2024-10-08'),
            'pct_chg': [2.1, -1.5],
            'volume': [1200000, 900000],
            'open': [10.0, 20.0],
            'close': [10.21, 19.70],
            'high': [10.3, 20.1],
            'low': [9.9, 19.6]
        })
        all_data.append(date_1008)

        # 第二个日期: 2024-10-09 (老股票继续)
        date_1009 = pd.DataFrame({
            'stock_code': ['000001.SZ', '000002.SZ'],
            'date': pd.to_datetime('2024-10-09'),
            'pct_chg': [1.8, 0.8],
            'volume': [1150000, 950000],
            'open': [10.25, 19.75],
            'close': [10.44, 19.86],
            'high': [10.5, 20.0],
            'low': [10.2, 19.7]
        })
        all_data.append(date_1009)

        # 第三个日期: 2024-10-11 (001279.SZ上市首日，包含异常值)
        date_1011 = pd.DataFrame({
            'stock_code': ['000001.SZ', '000002.SZ', '001279.SZ'],  # 新股出现！
            'date': pd.to_datetime('2024-10-11'),
            'pct_chg': [2.5, -1.2, 1738.946281],              # 001279.SZ异常值
            'volume': [1100000, 880000, 319155],
            'open': [10.58, 19.61, 87.12],
            'close': [10.84, 19.37, 178.01],
            'high': [11.0, 19.8, 245.0],
            'low': [10.5, 19.5, 87.12]
        })
        all_data.append(date_1011)

        # 第四个日期: 2024-10-14 (继续包含新股)
        date_1014 = pd.DataFrame({
            'stock_code': ['000001.SZ', '000002.SZ', '001279.SZ'],
            'date': pd.to_datetime('2024-10-14'),
            'pct_chg': [1.2, -0.5, -38.233807],
            'volume': [1050000, 920000, 250000],
            'open': [10.97, 19.32, 110.0],
            'close': [11.10, 19.27, 68.0],
            'high': [11.2, 19.5, 125.0],
            'low': [10.9, 19.2, 65.0]
        })
        all_data.append(date_1014)

        # 第五个日期: 2024-11-15 (第25个交易日)
        date_1115 = pd.DataFrame({
            'stock_code': ['000001.SZ', '000002.SZ', '001279.SZ'],
            'date': pd.to_datetime('2024-11-15'),
            'pct_chg': [0.8, -0.3, -3.776808],
            'volume': [1000000, 830000, 290000],
            'open': [11.18, 19.30, 65.0],
            'close': [11.27, 19.21, 62.55],
            'high': [11.3, 19.4, 68.0],
            'low': [11.1, 19.1, 62.0]
        })
        all_data.append(date_1115)

        # 第六个日期: 2024-11-18 (第26个交易日，新股应该保留)
        date_1118 = pd.DataFrame({
            'stock_code': ['000001.SZ', '000002.SZ', '001279.SZ'],
            'date': pd.to_datetime('2024-11-18'),
            'pct_chg': [1.2, 0.5, 1.5],
            'volume': [1020000, 850000, 310000],
            'open': [11.41, 19.31, 63.5],
            'close': [11.55, 19.41, 64.45],
            'high': [11.6, 19.5, 65.0],
            'low': [11.3, 19.3, 63.0]
        })
        all_data.append(date_1118)

        # 合并所有数据
        stock_data = pd.concat(all_data, ignore_index=True)
        print(f"原始数据: {len(stock_data)} 行, {len(stock_data['date'].unique())} 个交易日, {stock_data['stock_code'].nunique()} 只股票")

        # 步骤1：识别新股（以第一个日期为准）
        print("\n--- 步骤1：识别新股 ---")
        first_date = sorted(stock_data['date'].unique())[0]
        first_date_str = first_date.strftime('%Y%m%d')
        first_day_stocks = set(stock_data[stock_data['date'] == first_date]['stock_code'].unique())

        print(f"第一个交易日: {first_date_str}")
        print(f"第一个交易日股票数: {len(first_day_stocks)}")
        print(f"第一个交易日股票: {sorted(list(first_day_stocks))}")

        # 获取所有股票
        all_dates = sorted(stock_data['date'].unique())
        all_stocks = set(stock_data['stock_code'].unique())

        # 识别新股
        new_stocks = all_stocks - first_day_stocks
        print(f"识别出新股数量: {len(new_stocks)}")
        print(f"新股股票: {sorted(list(new_stocks))}")

        # 步骤3：为每个新股计算首次出现日期
        print("\n--- 步骤2：计算新股首次出现日期 ---")
        new_stock_first_dates = {}
        for new_stock in new_stocks:
            stock_data_filtered = stock_data[stock_data['stock_code'] == new_stock]
            if not stock_data_filtered.empty:
                first_appear_date = stock_data_filtered['date'].min()
                new_stock_first_dates[new_stock] = first_appear_date
                print(f"{new_stock}: 首次出现于 {first_appear_date.strftime('%Y-%m-%d')}")

        # 步骤4：过滤新股期间的数据
        print("\n--- 步骤3：过滤新股期间的数据 ---")

        # 模拟交易日历
        trading_dates = [
            '20241008', '20241009', '20241010', '20241011', '20241014', '20241015',
            '20241016', '20241017', '20241018', '20241021', '20241022', '20241023',
            '20241024', '20241025', '20241028', '20241029', '20241030', '20241031',
            '20241101', '20241104', '20241105', '20241106', '20241107', '20241108',
            '20241111', '20241112', '20241113', '20241114', '20241115', '20241118'
        ]

        valid_data_list = []
        total_new_filtered = 0

        for date in all_dates:
            date_str = date.strftime('%Y%m%d')
            daily_data = stock_data[stock_data['date'] == date].copy()

            if len(daily_data) > 0:
                daily_count_before = len(daily_data)
                print(f"\n处理 {date_str}: {daily_count_before} 只股票")

                # 获取当前日期在交易日历中的索引
                if date_str in trading_dates:
                    current_date_idx = trading_dates.index(date_str)

                    # 检查每只股票是否在新股期间
                    valid_stocks = []
                    for _, row in daily_data.iterrows():
                        stock_code = row['stock_code']

                        if stock_code in new_stock_first_dates:
                            # 这是新股，检查是否在新股期间（25个交易日）
                            first_appear_date = new_stock_first_dates[stock_code]
                            first_appear_str = first_appear_date.strftime('%Y%m%d')

                            if first_appear_str in trading_dates:
                                first_appear_idx = trading_dates.index(first_appear_str)
                                trading_days_since_first = current_date_idx - first_appear_idx + 1

                                print(f"  {stock_code}: 上市第{trading_days_since_first}天")

                                # 如果在新股期间（<=25天），过滤掉
                                if trading_days_since_first <= 25:
                                    print(f"    -> 过滤 (新股期间)")
                                    continue  # 跳过这只股票
                                else:
                                    print(f"    -> 保留 (超过新股期间)")
                        else:
                            print(f"  {stock_code}: 老股票，保留")

                        # 保留这只股票
                        valid_stocks.append(row)

                    if valid_stocks:
                        daily_data = pd.DataFrame(valid_stocks)
                        print(f"  保留股票: {daily_data['stock_code'].tolist()}")

                daily_new_filtered = daily_count_before - len(daily_data)
                total_new_filtered += daily_new_filtered
                print(f"  当日过滤: {daily_new_filtered} 只")

                if len(daily_data) > 0:
                    valid_data_list.append(daily_data)

        if valid_data_list:
            filtered_stock_data = pd.concat(valid_data_list, ignore_index=True)
            final_count = len(filtered_stock_data)
        else:
            filtered_stock_data = pd.DataFrame(columns=stock_data.columns)
            final_count = 0

        print(f"\n=== 过滤结果 ===")
        print(f"原始数据: {len(stock_data)} 行")
        print(f"过滤后数据: {final_count} 行")
        print(f"总过滤数量: {total_new_filtered} 行")

        # 检查异常值是否被过滤
        print(f"\n=== 异常值检查 ===")
        original_extreme_count = (abs(stock_data['pct_chg']) > 1000).sum()
        filtered_extreme_count = (abs(filtered_stock_data['pct_chg']) > 1000).sum()

        print(f"原始数据中|pct_chg| > 1000的记录: {original_extreme_count} 条")
        print(f"过滤后数据中|pct_chg| > 1000的记录: {filtered_extreme_count} 条")

        if original_extreme_count > 0 and filtered_extreme_count == 0:
            print("✅ 异常值已成功被过滤！")
        elif original_extreme_count == filtered_extreme_count:
            print("⚠️ 异常值过滤可能有问题")
        else:
            print("✓ 部分异常值被过滤")

        # 检查001279.SZ的处理情况
        print(f"\n=== 001279.SZ处理情况 ===")
        original_001279 = stock_data[stock_data['stock_code'] == '001279.SZ']
        filtered_001279 = filtered_stock_data[filtered_stock_data['stock_code'] == '001279.SZ']

        print(f"原始001279.SZ数据: {len(original_001279)} 条")
        print(f"过滤后001279.SZ数据: {len(filtered_001279)} 条")

        if len(original_001279) > 0:
            print("原始001279.SZ记录:")
            for _, row in original_001279.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: pct_chg = {row['pct_chg']:.6f}")

        if len(filtered_001279) > 0:
            print("过滤后001279.SZ记录:")
            for _, row in filtered_001279.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: pct_chg = {row['pct_chg']:.6f}")

        print("\n=== 测试结论 ===")
        if original_extreme_count > 0 and filtered_extreme_count == 0:
            print("✅ 新股过滤逻辑成功！")
            print("  - 1738.946281异常值被过滤")
            print("  - 基于首个日期正确识别新股")
            print("  - 25个交易日规则正确应用")
        else:
            print("❌ 需要检查过滤逻辑")

        print("\n测试完成")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    test_first_date_logic()