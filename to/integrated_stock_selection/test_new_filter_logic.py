#!/usr/bin/env python3
"""测试新的新股过滤逻辑"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

# 添加路径以便导入main模块
sys.path.append(str(Path(__file__).parent))

def test_new_stock_filtering():
    """测试新的新股过滤逻辑"""
    print("=== 测试新的新股过滤逻辑 ===\n")

    try:
        from main import _filter_new_stocks, _get_trading_dates_cache, _save_trading_dates_cache
        import logging

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # 模拟交易日历数据
        trading_dates = []
        for year in [2024, 2025]:
            for month in range(1, 13):
                for day in range(1, 32):
                    if month == 2:
                        max_day = 29 if year == 2024 else 28
                    elif month in [4, 6, 9, 11]:
                        max_day = 30
                    else:
                        max_day = 31

                    if day <= max_day:
                        # 简单跳过周末（仅作示例）
                        if month == 10 and day >= 11 and day <= 15:  # 包含我们的测试日期
                            trading_dates.extend([f"2024{month:02d}{day:02d}"])
                        elif month == 11 and day >= 13 and day <= 15:
                            trading_dates.extend([f"2024{month:02d}{day:02d}"])

        # 确保包含关键测试日期
        test_dates = ['20241011', '20241012', '20241014', '20241015', '20241114', '20241115']
        for date in test_dates:
            if date not in trading_dates:
                trading_dates.append(date)

        trading_dates.sort()

        # 保存交易日历缓存
        trading_dates_cache = {'20241115': trading_dates}
        _save_trading_dates_cache(trading_dates_cache)

        print(f"已设置测试交易日历，包含 {len(trading_dates)} 个交易日")
        print(f"关键日期包含: {test_dates}")

        # 清理股票交易天数缓存
        cache_file = Path(__file__).parent / "cache" / "stock_trading_days_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
            print("已清理旧的股票交易天数缓存")

        # 测试数据1：2024-10-11（上市首日）
        print("\n--- 测试1：2024-10-11（001279.SZ上市首日）---")
        data_1011 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'pct_chg': [1738.946281, 2.5, -1.2],
            'volume': [319155, 1000000, 800000],
            'date': '2024-10-11'
        })

        filter_result_1011 = _filter_new_stocks(data_1011, '20241011', logger)
        print(f"过滤结果: {filter_result_1011.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1011.iloc[0]}")

        # 测试数据2：2024-10-12（上市第二天）
        print("\n--- 测试2：2024-10-12（上市第二天）---")
        data_1012 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'pct_chg': [-38.233807, 1.8, -0.8],
            'volume': [250000, 1200000, 900000],
            'date': '2024-10-12'
        })

        filter_result_1012 = _filter_new_stocks(data_1012, '20241012', logger)
        print(f"过滤结果: {filter_result_1012.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1012.iloc[0]}")

        # 模拟历史数据（为了让001279.SZ积累足够的历史天数）
        print("\n--- 模拟历史数据积累 ---")
        dates_to_simulate = ['20241013', '20241014', '20241015', '20241016', '20241017', '20241018']

        for i, date in enumerate(dates_to_simulate):
            print(f"模拟 {date}: 001279.SZ 历史天数 = {i+3}")
            data_sim = pd.DataFrame({
                'stock_code': ['001279.SZ'],
                'pct_chg': [np.random.uniform(-10, 10)],
                'volume': [300000],
                'date': f'2024-{date[4:6]}-{date[6:8]}'
            })
            _filter_new_stocks(data_sim, date, logger)

        # 测试数据3：2024-11-14（接近25个历史天数）
        print("\n--- 测试3：2024-11-14（接近25个历史天数）---")
        data_1114 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'pct_chg': [-4.805781, 1.2, -0.5],
            'volume': [280000, 1100000, 850000],
            'date': '2024-11-14'
        })

        filter_result_1114 = _filter_new_stocks(data_1114, '20241114', logger)
        print(f"过滤结果: {filter_result_1114.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1114.iloc[0]}")

        # 测试数据4：2024-11-15（25个历史天数后）
        print("\n--- 测试4：2024-11-15（25个历史天数后）---")
        data_1115 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'pct_chg': [-3.776808, 0.8, -0.3],
            'volume': [290000, 1050000, 820000],
            'date': '2024-11-15'
        })

        filter_result_1115 = _filter_new_stocks(data_1115, '20241115', logger)
        print(f"过滤结果: {filter_result_1115.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1115.iloc[0]}")

        # 分析结果
        print("\n--- 结果分析 ---")
        print(f"2024-10-11 (上市首日): 001279.SZ {'被过滤' if not filter_result_1011.iloc[0] else '未过滤'}")
        print(f"2024-10-12 (上市第二天): 001279.SZ {'被过滤' if not filter_result_1012.iloc[0] else '未过滤'}")
        print(f"2024-11-14 (接近25天): 001279.SZ {'被过滤' if not filter_result_1114.iloc[0] else '未过滤'}")
        print(f"2024-11-15 (25天后): 001279.SZ {'被过滤' if not filter_result_1115.iloc[0] else '未过滤'}")

        # 检查缓存状态
        print("\n--- 缓存状态检查 ---")
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                stock_trading_days = pickle.load(f)

            if '001279.SZ' in stock_trading_days:
                dates_001279 = stock_trading_days['001279.SZ']
                print(f"001279.SZ交易日期记录: {len(dates_001279)} 个")
                print(f"  最后5个交易日: {dates_001279[-5:]}")
            else:
                print("001279.SZ 无交易记录")
        else:
            print("缓存文件不存在")

        print("\n✅ 测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    test_new_stock_filtering()