#!/usr/bin/env python3
"""检查新股过滤机制对pct_chg_1d异常值的影响"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def debug_new_stock_filtering():
    """调试新股过滤机制对pct_chg_1d的影响"""
    print("=== 检查新股过滤机制对异常值的影响 ===\n")

    try:
        # 1. 检查001279.SZ的上市日期
        print("步骤1: 检查001279.SZ的基本信息")

        # 从日志中我们知道上市时间是20241011
        ipo_date = datetime(2024, 10, 11)
        target_date = datetime(2024, 11, 15)
        prev_date = datetime(2024, 11, 14)

        print(f"  上市日期: {ipo_date.strftime('%Y-%m-%d')}")
        print(f"  目标日期: {target_date.strftime('%Y-%m-%d')}")
        print(f"  前一日: {prev_date.strftime('%Y-%m-%d')}")

        # 计算交易天数
        days_since_ipo = (target_date - ipo_date).days
        print(f"  从上市到目标日期的天数: {days_since_ipo}")

        # 2. 检查新股过滤逻辑
        print(f"\n步骤2: 检查新股过滤逻辑")
        print(f"  新股过滤标准: 有效交易天数 < 25天")
        print(f"  2024-11-15时的交易天数: {days_since_ipo}")

        if days_since_ipo < 25:
            print(f"  ⚠️  001279.SZ在2024-11-15会被新股过滤掉!")
        else:
            print(f"  ✅ 001279.SZ在2024-11-15不会被新股过滤")

        # 3. 读取20241114的数据（shift操作的数据源）
        print(f"\n步骤3: 检查2024-11-14的原始数据")
        try:
            df_20241114 = pd.read_parquet("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/20241114.parquet")
            print(f"  2024-11-14数据形状: {df_20241114.shape}")

            # 查找001279.SZ
            stock_001279 = df_20241114[df_20241114['stock_code'] == '001279.SZ']
            if len(stock_001279) > 0:
                stock_info = stock_001279.iloc[0]
                print(f"  001279.SZ在2024-11-14的数据:")
                print(f"    stock_code: {stock_info['stock_code']}")
                print(f"    date: {stock_info['date']}")
                print(f"    pct_chg: {stock_info['pct_chg']}")
                print(f"    open: {stock_info['open']}")
                print(f"    close: {stock_info['close']}")
                print(f"    high: {stock_info['high']}")
                print(f"    low: {stock_info['low']}")
            else:
                print(f"  ❌ 未找到001279.SZ在2024-11-14的数据")

        except Exception as e:
            print(f"  ❌ 读取2024-11-14数据失败: {e}")

        # 4. 读取2024-11-15的数据
        print(f"\n步骤4: 检查2024-11-15的原始数据")
        try:
            df_20241115 = pd.read_parquet("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/20241115.parquet")
            print(f"  2024-11-15数据形状: {df_20241115.shape}")

            # 查找001279.SZ
            stock_001279_15 = df_20241115[df_20241115['stock_code'] == '001279.SZ']
            if len(stock_001279_15) > 0:
                stock_info = stock_001279_15.iloc[0]
                print(f"  001279.SZ在2024-11-15的数据:")
                print(f"    stock_code: {stock_info['stock_code']}")
                print(f"    date: {stock_info['date']}")
                print(f"    pct_chg: {stock_info['pct_chg']}")
                print(f"    open: {stock_info['open']}")
                print(f"    close: {stock_info['close']}")
                print(f"    high: {stock_info['high']}")
                print(f"    low: {stock_info['low']}")
            else:
                print(f"  ❌ 未找到001279.SZ在2024-11-15的数据")

        except Exception as e:
            print(f"  ❌ 读取2024-11-15数据失败: {e}")

        # 5. 模拟pct_chg_1d的shift操作
        print(f"\n步骤5: 模拟shift操作和潜在问题")

        # 读取多日数据来模拟真实的shift操作
        dates_to_check = []
        for i in range(30):  # 检查上市前后的30天
            check_date = ipo_date + timedelta(days=i)
            dates_to_check.append(check_date.strftime('%Y%m%d'))

        print(f"  检查日期范围: {dates_to_check[0]} 到 {dates_to_check[-1]}")

        all_data = []
        for date_str in dates_to_check:
            try:
                df = pd.read_parquet(f"C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/{date_str}.parquet")
                df['date_str'] = date_str
                all_data.append(df)
            except:
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"  合并后数据形状: {combined_df.shape}")

            # 检查001279.SZ的所有数据
            stock_all = combined_df[combined_df['stock_code'] == '001279.SZ'].copy()
            if len(stock_all) > 0:
                print(f"  001279.SZ的完整时间序列数据:")
                print(f"    数据行数: {len(stock_all)}")
                print(f"    日期范围: {stock_all['date_str'].min()} 到 {stock_all['date_str'].max()}")

                # 按日期排序
                stock_all = stock_all.sort_values('date_str')

                print(f"  时间序列数据:")
                for _, row in stock_all.iterrows():
                    pct_chg_val = row.get('pct_chg', 'N/A')
                    if pct_chg_val != 'N/A':
                        print(f"    {row['date_str']}: pct_chg = {pct_chg_val:.6f}")

                        # 检查是否有异常值
                        if abs(pct_chg_val) > 100:
                            print(f"      ⚠️  发现异常值!")

                            # 计算这个日期对应的shift后的位置
                            current_date = datetime.strptime(row['date_str'], '%Y%m%d')
                            shift_target_date = current_date + timedelta(days=1)
                            print(f"      这个异常值会被shift到: {shift_target_date.strftime('%Y-%m-%d')}")
                else:
                    print(f"  ❌ 未找到001279.SZ的任何数据")

        # 6. 分析潜在的数据质量问题
        print(f"\n步骤6: 分析潜在的数据质量问题")
        print(f"  可能的问题源:")
        print(f"    1. 数据文件中某个日期的pct_chg值录入错误")
        print(f"    2. 股票复权计算错误")
        print(f"    3. 数据文件损坏或错误合并")
        print(f"    4. 时间戳错误导致数据错位")

        # 7. 检查是否有其他股票在相同日期也有异常值
        print(f"\n步骤7: 检查其他股票在相关日期的异常值")

        # 检查2024-11-15前后几天的数据
        check_dates = ['20241114', '20241115', '20241113', '20241116']

        for date_str in check_dates:
            try:
                df = pd.read_parquet(f"C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/{date_str}.parquet")

                # 检查极端值
                extreme_mask = abs(df['pct_chg']) > 100
                extreme_count = extreme_mask.sum()

                if extreme_count > 0:
                    print(f"  {date_str}: 发现 {extreme_count} 个极端涨跌幅")
                    extreme_stocks = df[extreme_mask][['stock_code', 'pct_chg']].nlargest(5, 'pct_chg')
                    for _, row in extreme_stocks.iterrows():
                        print(f"    {row['stock_code']}: {row['pct_chg']:.6f}")

            except Exception as e:
                print(f"  {date_str}: 读取失败 - {e}")

    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_new_stock_filtering()