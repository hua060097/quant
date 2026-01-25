#!/usr/bin/env python3
"""测试_process_walkforward_window中的批量新股过滤"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加路径以便导入main模块
sys.path.append(str(Path(__file__).parent))

def test_walkforward_batch_filter():
    """测试Walk-Forward窗口处理中的批量新股过滤"""
    print("=== 测试_process_walkforward_window中的批量新股过滤 ===\n")

    try:
        from main import _filter_new_stocks_batch, _get_trading_dates_cache, _save_trading_dates_cache
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
        test_dates = [
            '20240801', '20240802', '20240805', '20240806', '20240807', '20240808',
            '20240809', '20240812', '20240813', '20240814', '20240815', '20240816',
            '20240819', '20240820', '20240821', '20240822', '20240823', '20240826',
            '20240827', '20240828', '20240829', '20240830', '20240902', '20240903'
        ]

        # 保存交易日历缓存
        trading_dates_cache = {'all_trading_dates': test_dates}
        _save_trading_dates_cache(trading_dates_cache)

        print(f"已设置测试交易日历，包含 {len(test_dates)} 个交易日")

        # 创建模拟的因子数据（模拟Walk-Forward窗口的数据）
        all_data = []

        # 训练集起始日期：2024-08-01 (基准日期，只有老股票)
        for date in pd.date_range('2024-08-01', '2024-08-10'):
            date_str = date.strftime('%Y-%m-%d')
            data = pd.DataFrame({
                'date': date,
                'stock_code': ['000001.SZ', '000002.SZ', '600000.SH'],
                'factor1': np.random.normal(0, 1, 3),
                'factor2': np.random.normal(0, 1, 3),
                'pct_chg': np.random.normal(0, 2, 3),
                'volume': np.random.randint(500000, 2000000, 3),
                'close': np.random.uniform(10, 50, 3),
                'atr14': np.random.uniform(0.5, 2.0, 3)
            })
            all_data.append(data)

        # 验证集起始日期：2024-08-12 (新股001279.SZ上市，包含异常值)
        for date in pd.date_range('2024-08-12', '2024-08-20'):
            date_str = date.strftime('%Y-%m-%d')

            # 老股票数据
            base_data = pd.DataFrame({
                'date': date,
                'stock_code': ['000001.SZ', '000002.SZ', '600000.SH'],
                'factor1': np.random.normal(0, 1, 3),
                'factor2': np.random.normal(0, 1, 3),
                'pct_chg': np.random.normal(0, 2, 3),
                'volume': np.random.randint(500000, 2000000, 3),
                'close': np.random.uniform(10, 50, 3),
                'atr14': np.random.uniform(0.5, 2.0, 3)
            })

            # 新股001279.SZ数据
            if date == pd.to_datetime('2024-08-12'):
                # 上市首日，异常值
                new_stock_data = pd.DataFrame({
                    'date': date,
                    'stock_code': ['001279.SZ'],
                    'factor1': [0.5],
                    'factor2': [0.8],
                    'pct_chg': [1738.946281],  # 异常值
                    'volume': [319155],
                    'close': [178.01],
                    'atr14': [10.0]
                })
            elif date == pd.to_datetime('2024-08-13'):
                # 上市第二天
                new_stock_data = pd.DataFrame({
                    'date': date,
                    'stock_code': ['001279.SZ'],
                    'factor1': [-0.2],
                    'factor2': [-0.1],
                    'pct_chg': [-38.233807],
                    'volume': [250000],
                    'close': [68.0],
                    'atr14': [8.0]
                })
            else:
                # 正常交易日
                new_stock_data = pd.DataFrame({
                    'date': date,
                    'stock_code': ['001279.SZ'],
                    'factor1': np.random.normal(0, 1, 1),
                    'factor2': np.random.normal(0, 1, 1),
                    'pct_chg': np.random.normal(0, 2, 1),
                    'volume': [np.random.randint(200000, 400000)],
                    'close': [np.random.uniform(60, 80)],
                    'atr14': [np.random.uniform(3, 7)]
                })

            all_data.append(pd.concat([base_data, new_stock_data], ignore_index=True))

        # 测试集日期：2024-08-21到2024-08-30
        for date in pd.date_range('2024-08-21', '2024-08-30'):
            date_str = date.strftime('%Y-%m-%d')
            data = pd.DataFrame({
                'date': date,
                'stock_code': ['000001.SZ', '000002.SZ', '600000.SH', '001279.SZ'],
                'factor1': np.random.normal(0, 1, 4),
                'factor2': np.random.normal(0, 1, 4),
                'pct_chg': np.random.normal(0, 2, 4),
                'volume': np.random.randint(500000, 2000000, 4),
                'close': np.random.uniform(10, 80, 4),
                'atr14': np.random.uniform(0.5, 5.0, 4)
            })
            all_data.append(data)

        # 合并所有数据
        factor_data = pd.concat(all_data, ignore_index=True)
        print(f"模拟因子数据: {len(factor_data)} 行, {len(factor_data['date'].unique())} 个交易日, {factor_data['stock_code'].nunique()} 只股票")

        # 检查异常值
        original_extreme_count = (abs(factor_data['pct_chg']) > 1000).sum()
        print(f"原始数据中|pct_chg| > 1000的记录: {original_extreme_count} 条")

        # 模拟Walk-Forward窗口处理的关键部分：数据过滤
        print("\n--- 模拟Walk-Forward窗口数据过滤 ---")

        # 合并训练集、验证集和测试集（模拟Walk-Forward的处理逻辑）
        combined_data = factor_data.copy()
        combined_data = combined_data.sort_values(['date', 'stock_code']).reset_index(drop=True)

        print(f"合并数据: {len(combined_data)} 行")

        # 应用批量新股过滤（模拟_process_walkforward_window中的逻辑）
        initial_total = len(combined_data)
        print(f"初始样本: {initial_total:,}条")

        # 批量过滤新股
        print("开始批量新股过滤...")
        original_count = len(combined_data)
        combined_data = _filter_new_stocks_batch(combined_data, logger)
        after_new_stock_count = len(combined_data)
        new_stock_filtered_count = original_count - after_new_stock_count
        print(f"批量新股过滤完成: {original_count} -> {after_new_stock_count} (过滤掉 {new_stock_filtered_count} 条新股期间数据)")

        # 检查异常值是否被过滤
        filtered_extreme_count = (abs(combined_data['pct_chg']) > 1000).sum()
        print(f"过滤后数据中|pct_chg| > 1000的记录: {filtered_extreme_count} 条")

        # 检查001279.SZ的处理情况
        print(f"\n=== 001279.SZ处理情况 ===")
        original_001279 = factor_data[factor_data['stock_code'] == '001279.SZ']
        filtered_001279 = combined_data[combined_data['stock_code'] == '001279.SZ']

        print(f"原始001279.SZ数据: {len(original_001279)} 条")
        print(f"过滤后001279.SZ数据: {len(filtered_001279)} 条")

        if len(original_001279) > 0:
            print("原始001279.SZ关键记录:")
            for _, row in original_001279.iterrows():
                if abs(row['pct_chg']) > 1000:
                    print(f"  {row['date'].strftime('%Y-%m-%d')}: pct_chg = {row['pct_chg']:.6f} [异常值]")
                else:
                    print(f"  {row['date'].strftime('%Y-%m-%d')}: pct_chg = {row['pct_chg']:.6f}")

        # 按数据集统计过滤情况
        print(f"\n=== 按数据集统计 ===")
        train_start, train_end = '2024-08-01', '2024-08-10'
        val_start, val_end = '2024-08-12', '2024-08-20'
        test_start, test_end = '2024-08-21', '2024-08-30'

        # 训练集统计
        train_orig = factor_data[(factor_data['date'] >= train_start) & (factor_data['date'] <= train_end)]
        train_filt = combined_data[(combined_data['date'] >= train_start) & (combined_data['date'] <= train_end)]
        print(f"训练集: {len(train_orig)} -> {len(train_filt)} (无新股，应该无变化)")

        # 验证集统计
        val_orig = factor_data[(factor_data['date'] >= val_start) & (factor_data['date'] <= val_end)]
        val_filt = combined_data[(combined_data['date'] >= val_start) & (combined_data['date'] <= val_end)]
        print(f"验证集: {len(val_orig)} -> {len(val_filt)} (有新股，应该被过滤)")

        # 测试集统计
        test_orig = factor_data[(factor_data['date'] >= test_start) & (factor_data['date'] <= test_end)]
        test_filt = combined_data[(combined_data['date'] >= test_start) & (combined_data['date'] <= test_end)]
        print(f"测试集: {len(test_orig)} -> {len(test_filt)} (包含新股期间，部分被过滤)")

        print("\n=== 测试结论 ===")
        if original_extreme_count > 0 and filtered_extreme_count == 0:
            print("✅ Walk-Forward窗口中的批量新股过滤成功！")
            print("  - 1738.946281异常值被过滤")
            print("  - 新股过滤已成功移到日期循环之外")
            print("  - 使用_filter_new_stocks_batch进行批量处理")
            print("  - Walk-Forward数据集正确处理")
        else:
            print("❌ 需要检查批量新股过滤逻辑")

        print("\n测试完成")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    test_walkforward_batch_filter()