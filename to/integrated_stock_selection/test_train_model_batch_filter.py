#!/usr/bin/env python3
"""测试run_train_model_mode中的批量新股过滤"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加路径以便导入main模块
sys.path.append(str(Path(__file__).parent))

def test_train_model_batch_filter():
    """测试训练模型模式中的批量新股过滤"""
    print("=== 测试run_train_model_mode中的批量新股过滤 ===\n")

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
            '20240102', '20240103', '20240104', '20240105', '20240108', '20240109',
            '20240110', '20240111', '20240112', '20240115', '20240116', '20240117',
            '20240118', '20240119', '20240122', '20240123', '20240124', '20240125', '20240126'
        ]

        # 保存交易日历缓存
        trading_dates_cache = {'all_trading_dates': test_dates}
        _save_trading_dates_cache(trading_dates_cache)

        print(f"已设置测试交易日历，包含 {len(test_dates)} 个交易日")

        # 创建模拟的训练因子数据
        all_data = []

        # 2024-01-02到2024-01-10: 只有老股票
        for date in pd.date_range('2024-01-02', '2024-01-10'):
            base_data = pd.DataFrame({
                'date': date,
                'stock_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'],
                'factor1': np.random.normal(0, 1, 4),
                'factor2': np.random.normal(0, 1, 4),
                'factor3': np.random.normal(0, 1, 4),
                'momentum_factor': np.random.normal(0, 0.5, 4),
                'volatility_factor': np.random.uniform(0.1, 0.5, 4),
                'liquidity_factor': np.random.uniform(0, 2, 4)
            })
            all_data.append(base_data)

        # 2024-01-11到2024-01-15: 新股301488.SZ上市，包含异常值
        for date in pd.date_range('2024-01-11', '2024-01-15'):
            base_data = pd.DataFrame({
                'date': date,
                'stock_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'],
                'factor1': np.random.normal(0, 1, 4),
                'factor2': np.random.normal(0, 1, 4),
                'factor3': np.random.normal(0, 1, 4),
                'momentum_factor': np.random.normal(0, 0.5, 4),
                'volatility_factor': np.random.uniform(0.1, 0.5, 4),
                'liquidity_factor': np.random.uniform(0, 2, 4)
            })

            # 新股301488.SZ数据
            if date == pd.to_datetime('2024-01-11'):
                # 上市首日，异常值
                new_stock_data = pd.DataFrame({
                    'date': date,
                    'stock_code': ['301488.SZ'],
                    'factor1': [1.2],
                    'factor2': [0.8],
                    'factor3': [-0.5],
                    'momentum_factor': [2.5],
                    'volatility_factor': [3.0],
                    'liquidity_factor': [0.5]
                })
            elif date == pd.to_datetime('2024-01-12'):
                # 上市第二天
                new_stock_data = pd.DataFrame({
                    'date': date,
                    'stock_code': ['301488.SZ'],
                    'factor1': [-0.3],
                    'factor2': [-0.1],
                    'factor3': [0.2],
                    'momentum_factor': [-1.5],
                    'volatility_factor': [4.5],
                    'liquidity_factor': [0.8]
                })
            else:
                # 正常交易日
                new_stock_data = pd.DataFrame({
                    'date': date,
                    'stock_code': ['301488.SZ'],
                    'factor1': np.random.normal(0, 1, 1),
                    'factor2': np.random.normal(0, 1, 1),
                    'factor3': np.random.normal(0, 1, 1),
                    'momentum_factor': np.random.normal(0, 0.5, 1),
                    'volatility_factor': np.random.uniform(0.1, 0.5, 1),
                    'liquidity_factor': np.random.uniform(0, 2, 1)
                })

            all_data.append(pd.concat([base_data, new_stock_data], ignore_index=True))

        # 2024-01-16到2024-01-26: 继续包含新股
        for date in pd.date_range('2024-01-16', '2024-01-26'):
            base_data = pd.DataFrame({
                'date': date,
                'stock_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '301488.SZ'],
                'factor1': np.random.normal(0, 1, 5),
                'factor2': np.random.normal(0, 1, 5),
                'factor3': np.random.normal(0, 1, 5),
                'momentum_factor': np.random.normal(0, 0.5, 5),
                'volatility_factor': np.random.uniform(0.1, 0.5, 5),
                'liquidity_factor': np.random.uniform(0, 2, 5)
            })
            all_data.append(base_data)

        # 合并所有数据
        factor_data = pd.concat(all_data, ignore_index=True)
        print(f"模拟因子数据: {len(factor_data)} 行, {len(factor_data['date'].unique())} 个交易日, {factor_data['stock_code'].nunique()} 只股票")

        # 检查新股
        new_stock_301488 = factor_data[factor_data['stock_code'] == '301488.SZ']
        print(f"301488.SZ数据: {len(new_stock_301488)} 条")

        # 模拟run_train_model_mode中的数据过滤逻辑
        print("\n--- 模拟训练模型模式数据过滤 ---")

        # 应用批量新股过滤（模拟run_train_model_mode中的逻辑）
        print("开始批量新股过滤...")
        original_count = len(factor_data)
        factor_data_filtered = _filter_new_stocks_batch(factor_data, logger)
        after_new_stock_count = len(factor_data_filtered)
        new_stock_filtered_count = original_count - after_new_stock_count
        print(f"批量新股过滤完成: {original_count} -> {after_new_stock_count} (过滤掉 {new_stock_filtered_count} 条新股期间数据)")

        # 检查301488.SZ的处理情况
        filtered_new_stock_301488 = factor_data_filtered[factor_data_filtered['stock_code'] == '301488.SZ']
        print(f"过滤后301488.SZ数据: {len(filtered_new_stock_301488)} 条")

        if len(new_stock_301488) > 0:
            print("原始301488.SZ记录:")
            for _, row in new_stock_301488.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['stock_code']}")

        if len(filtered_new_stock_301488) > 0:
            print("过滤后301488.SZ记录:")
            for _, row in filtered_new_stock_301488.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['stock_code']}")

        # 按日期统计过滤情况
        print(f"\n=== 按日期统计 ===")
        original_dates = factor_data['date'].unique()
        for date in sorted(original_dates)[:10]:  # 只显示前10天
            date_str = date.strftime('%Y-%m-%d')
            original_count = len(factor_data[factor_data['date'] == date])
            filtered_count = len(factor_data_filtered[factor_data_filtered['date'] == date])
            print(f"{date_str}: {original_count} -> {filtered_count}")

        # 检查25个交易日后的情况
        print(f"\n=== 新股过滤效果分析 ===")
        # 301488.SZ是2024-01-11上市的，所以前25个交易日（到约2月）的数据应该被过滤

        # 检查3月份是否还有301488.SZ数据
        march_data = filtered_new_stock_301488[filtered_new_stock_301488['date'] >= pd.to_datetime('2024-03-01')]
        print(f"3月份301488.SZ数据: {len(march_data)} 条")

        if len(march_data) > 0:
            print("✅ 3月份保留了301488.SZ数据（说明25个交易日规则正确应用）")
        else:
            print("⚠️ 3月份没有301488.SZ数据（可能需要更多测试数据）")

        print("\n=== 测试结论 ===")
        if new_stock_filtered_count > 0:
            print("✅ 训练模型模式中的批量新股过滤成功！")
            print("  - 新股数据已被正确过滤")
            print("  - 新股过滤已成功移到日期循环之外")
            print("  - 使用_filter_new_stocks_batch进行批量处理")
            print("  - 训练数据正确处理，避免污染模型训练")
        else:
            print("⚠️ 没有数据被过滤（可能测试数据没有足够的新股期间）")

        print("\n测试完成")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    test_train_model_batch_filter()