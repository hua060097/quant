#!/usr/bin/env python3
"""测试基于首次出现日期的新股过滤逻辑"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

# 添加路径以便导入main模块
sys.path.append(str(Path(__file__).parent))

def test_first_appearance_filter():
    """测试基于首次出现日期的新股过滤逻辑"""
    print("=== 测试基于首次出现日期的新股过滤逻辑 ===\n")

    try:
        from main import _filter_new_stocks, _get_trading_dates_cache, _save_trading_dates_cache
        import logging

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # 清理缓存文件
        cache_file = Path(__file__).parent / "cache" / "stock_first_appearance_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
            print("已清理旧的股票首次出现缓存")

        # 模拟真实的交易日历（基于实际数据）
        trading_dates = []
        # 生成2024年10月到11月的交易日
        for date_str in ['20241008', '20241009', '20241010', '20241011', '20241014', '20241015',
                         '20241016', '20241017', '20241018', '20241021', '20241022', '20241023',
                         '20241024', '20241025', '20241028', '20241029', '20241030', '20241031',
                         '20241101', '20241104', '20241105', '20241106', '20241107', '20241108',
                         '20241111', '20241112', '20241113', '20241114', '20241115', '20241118']:
            trading_dates.append(date_str)

        # 保存交易日历缓存
        trading_dates_cache = {'all_trading_dates': trading_dates}
        _save_trading_dates_cache(trading_dates_cache)

        print(f"已设置测试交易日历，包含 {len(trading_dates)} 个交易日")
        print(f"关键日期包含: 20241011(001279上市), 20241115(25个交易日后)")

        # 测试数据1：2024-10-11（001279.SZ上市首日）
        print("\n--- 测试1：2024-10-11（001279.SZ上市首日）---")
        data_1011 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],  # 001279.SZ首次出现
            'pct_chg': [1738.946281, 2.5, -1.2],                 # 包含异常值
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
            'pct_chg': [-38.233807, 1.8, -0.8],                   # 001279.SZ的第二天数据
            'volume': [250000, 1200000, 900000],
            'date': '2024-10-12'
        })

        filter_result_1012 = _filter_new_stocks(data_1012, '20241012', logger)
        print(f"过滤结果: {filter_result_1012.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1012.iloc[0]}")

        # 测试数据3：2024-10-20（上市第10个交易日左右）
        print("\n--- 测试3：2024-10-21（上市第10个交易日）---")
        data_1021 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'pct_chg': [3.2, 2.1, -1.5],                           # 001279.SZ第10天左右
            'volume': [300000, 1100000, 850000],
            'date': '2024-10-21'
        })

        filter_result_1021 = _filter_new_stocks(data_1021, '20241021', logger)
        print(f"过滤结果: {filter_result_1021.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1021.iloc[0]}")

        # 测试数据4：2024-11-15（上市第25个交易日）
        print("\n--- 测试4：2024-11-15（上市第25个交易日）---")
        data_1115 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'pct_chg': [-3.776808, 0.8, -0.3],                    # 001279.SZ第25天
            'volume': [290000, 1050000, 820000],
            'date': '2024-11-15'
        })

        filter_result_1115 = _filter_new_stocks(data_1115, '20241115', logger)
        print(f"过滤结果: {filter_result_1115.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1115.iloc[0]}")

        # 测试数据5：2024-11-18（上市第26个交易日，应该不再被过滤）
        print("\n--- 测试5：2024-11-18（上市第26个交易日，应该保留）---")
        data_1118 = pd.DataFrame({
            'stock_code': ['001279.SZ', '000001.SZ', '000002.SZ'],
            'pct_chg': [1.5, 1.2, 0.5],                           # 001279.SZ第26天，正常数据
            'volume': [310000, 1080000, 830000],
            'date': '2024-11-18'
        })

        filter_result_1118 = _filter_new_stocks(data_1118, '20241118', logger)
        print(f"过滤结果: {filter_result_1118.tolist()}")
        print(f"001279.SZ被过滤: {not filter_result_1118.iloc[0]}")

        # 分析结果
        print("\n--- 结果分析 ---")
        results = [
            ("2024-10-11 (上市首日)", filter_result_1011.iloc[0]),
            ("2024-10-12 (上市第二天)", filter_result_1012.iloc[0]),
            ("2024-10-21 (第10个交易日)", filter_result_1021.iloc[0]),
            ("2024-11-15 (第25个交易日)", filter_result_1115.iloc[0]),
            ("2024-11-18 (第26个交易日)", filter_result_1118.iloc[0])
        ]

        for desc, result in results:
            status = "被过滤" if not result else "未过滤"
            print(f"{desc}: 001279.SZ {status}")

        # 检查缓存状态
        print("\n--- 缓存状态检查 ---")
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                stock_first_dates = pickle.load(f)

            if '001279.SZ' in stock_first_dates:
                first_date = stock_first_dates['001279.SZ']
                print(f"001279.SZ首次出现日期: {first_date}")

                # 验证日期计算
                if first_date in trading_dates:
                    first_idx = trading_dates.index(first_date)
                    print(f"首次出现索引: {first_idx}")

                    # 检查各个测试日期的交易日数
                    test_dates = ['20241011', '20241012', '20241021', '20241115', '20241118']
                    for test_date in test_dates:
                        if test_date in trading_dates:
                            test_idx = trading_dates.index(test_date)
                            trading_days = test_idx - first_idx + 1
                            print(f"  {test_date}: 第{trading_days}个交易日")
            else:
                print("001279.SZ 无首次出现记录")
        else:
            print("缓存文件不存在")

        print("\n=== 预期结果 ===")
        print("✅ 2024-10-11: 被过滤 (上市第1天)")
        print("✅ 2024-10-12: 被过滤 (上市第2天)")
        print("✅ 2024-10-21: 被过滤 (上市第10天)")
        print("✅ 2024-11-15: 被过滤 (上市第25天)")
        print("✅ 2024-11-18: 未过滤 (上市第26天，超过25天)")

        print("\n=== 逻辑验证 ===")
        print("如果符合预期结果，说明:")
        print("1. 1738.946281异常值被正确过滤")
        print("2. shift(1)操作不会访问到IPO首日异常数据")
        print("3. 新股上市25个交易日后正常使用")

        print("\n测试完成")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    test_first_appearance_filter()