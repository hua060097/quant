#!/usr/bin/env python3
"""测试简化后的新股过滤逻辑集成"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加路径以便导入main模块
sys.path.append(str(Path(__file__).parent))

def test_simplified_integration():
    """测试简化后的新股过滤逻辑集成"""
    print("=== 测试简化后的新股过滤逻辑集成 ===\n")

    try:
        from main import _filter_new_stocks, _get_trading_dates_cache, _save_trading_dates_cache
        import logging

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # 清理缓存文件
        cache_files = [
            Path(__file__).parent / "cache" / "stock_first_appearance_cache.pkl",
            Path(__file__).parent / "cache" / "trading_dates_cache.pkl"
        ]

        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                print(f"已清理缓存文件: {cache_file.name}")

        # 设置测试交易日历
        trading_dates = []
        test_dates = [
            '20241008', '20241009', '20241010', '20241011', '20241014', '20241015',
            '20241016', '20241017', '20241018', '20241021', '20241022', '20241023',
            '20241024', '20241025', '20241028', '20241029', '20241030', '20241031',
            '20241101', '20241104', '20241105', '20241106', '20241107', '20241108',
            '20241111', '20241112', '20241113', '20241114', '20241115', '20241118'
        ]

        # 保存交易日历缓存
        trading_dates_cache = {'all_trading_dates': test_dates}
        _save_trading_dates_cache(trading_dates_cache)

        print(f"已设置测试交易日历，包含 {len(test_dates)} 个交易日")

        # 模拟完整的数据处理流程
        print("\n--- 模拟数据处理流程 ---")

        # 创建一个包含多日数据的DataFrame（模拟stock_data）
        all_data = []

        # 日期1: 2024-10-11 (001279.SZ上市首日，包含异常值)
        date_1011 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'date': pd.to_datetime('2024-10-11'),
            'pct_chg': [1738.946281, 2.5, -1.2],  # 001279.SZ异常值
            'volume': [319155, 1000000, 800000],
            'open': [87.12, 10.0, 20.0],
            'close': [178.01, 10.25, 19.76],
            'high': [245.0, 10.5, 21.0],
            'low': [87.12, 9.8, 19.0]
        })
        all_data.append(date_1011)

        # 日期2: 2024-10-14 (001279.SZ上市后几天)
        date_1014 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ', '600001.SH'],  # 新增600001.SH
            'date': pd.to_datetime('2024-10-14'),
            'pct_chg': [-38.233807, 1.8, -0.8, 2.1],
            'volume': [250000, 1200000, 900000, 1500000],
            'open': [110.0, 10.43, 19.61, 15.2],
            'close': [68.0, 10.62, 19.45, 15.52],
            'high': [125.0, 10.8, 20.0, 15.8],
            'low': [65.0, 10.3, 19.2, 15.0]
        })
        all_data.append(date_1014)

        # 日期3: 2024-11-15 (001279.SZ第25个交易日)
        date_1115 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ', '600001.SH'],
            'date': pd.to_datetime('2024-11-15'),
            'pct_chg': [-3.776808, 0.8, -0.3, 1.5],
            'volume': [290000, 1050000, 820000, 1400000],
            'open': [65.0, 10.65, 19.39, 15.75],
            'close': [62.55, 10.74, 19.33, 15.98],
            'high': [68.0, 10.9, 19.8, 16.2],
            'low': [62.0, 10.5, 19.1, 15.5]
        })
        all_data.append(date_1115)

        # 日期4: 2024-11-18 (001279.SZ第26个交易日，应该保留)
        date_1118 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ', '600001.SH'],
            'date': pd.to_datetime('2024-11-18'),
            'pct_chg': [1.5, 1.2, 0.5, 0.8],
            'volume': [310000, 1080000, 830000, 1350000],
            'open': [63.5, 10.87, 19.43, 16.1],
            'close': [64.45, 11.0, 19.53, 16.23],
            'high': [65.0, 11.2, 19.8, 16.5],
            'low': [63.0, 10.7, 19.3, 15.9]
        })
        all_data.append(date_1118)

        # 合并所有数据
        stock_data = pd.concat(all_data, ignore_index=True)
        print(f"原始数据: {len(stock_data)} 行, {len(stock_data['date'].unique())} 个交易日, {stock_data['stock_code'].nunique()} 只股票")

        # 应用新的新股过滤逻辑（模拟run_generate_factors_mode的逻辑）
        print("\n--- 应用新股过滤逻辑 ---")

        valid_data_list = []
        total_new_filtered = 0

        # 按日期分组进行新股过滤
        all_dates = sorted(stock_data['date'].unique())
        print(f"需处理 {len(all_dates)} 个交易日")

        for date in all_dates:
            date_str = date.strftime('%Y%m%d')
            daily_data = stock_data[stock_data['date'] == date].copy()

            if len(daily_data) > 0:
                daily_count_before = len(daily_data)
                print(f"\n处理 {date_str}: {daily_count_before} 只股票")

                # 使用新的新股过滤逻辑
                new_stock_filter = _filter_new_stocks(daily_data, date_str, logger)
                daily_data = daily_data[new_stock_filter].copy()

                if len(daily_data) > 0:
                    valid_data_list.append(daily_data)
                    print(f"  保留: {len(daily_data)} 只股票 - {daily_data['stock_code'].tolist()}")
                else:
                    print(f"  过滤: 全部被过滤")

                daily_new_filtered = daily_count_before - len(daily_data)
                total_new_filtered += daily_new_filtered
                print(f"  当日过滤: {daily_new_filtered} 只")

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
            print("✅ 新股过滤逻辑成功！异常值被过滤，shift操作安全。")
        else:
            print("❌ 需要检查过滤逻辑")

        print("\n测试完成")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    test_simplified_integration()