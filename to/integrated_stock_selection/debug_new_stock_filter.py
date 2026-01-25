#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试新股过滤逻辑 - 以股票301551.SZ在2024-09-26为例
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
current_file_dir = Path(__file__).parent
sys.path.append(str(current_file_dir))

# 导入main.py中的函数
from main import (
    _filter_new_stocks,
    _build_new_stock_list_for_date,
    _get_previous_trading_dates,
    _get_new_stock_cache,
    _save_new_stock_cache
)

def debug_new_stock_filter():
    """调试新股过滤逻辑"""

    # 目标股票和日期
    target_stock = "301551.SZ"
    target_date = "20240926"
    target_stock_list_date = "20240926"  # 上市日期

    print("=" * 80)
    print(f"调试新股过滤: 股票 {target_stock} 在日期 {target_date}")
    print("=" * 80)

    # 1. 检查目标日期的因子文件是否存在
    features_dir = Path(__file__).parent / "local_data" / "features" / "daily"
    target_file = features_dir / f"{target_date}.parquet"

    print(f"\n1. 检查因子文件:")
    print(f"   文件路径: {target_file}")
    print(f"   文件存在: {target_file.exists()}")

    if target_file.exists():
        try:
            daily_data = pd.read_parquet(target_file)
            print(f"   数据形状: {daily_data.shape}")
            print(f"   股票数量: {daily_data['stock_code'].nunique()}")

            # 检查目标股票是否存在
            target_stock_data = daily_data[daily_data['stock_code'] == target_stock]
            print(f"   目标股票存在: {len(target_stock_data) > 0}")

            if len(target_stock_data) > 0:
                print(f"   目标股票数据:")
                print(f"     stock_code: {target_stock_data['stock_code'].iloc[0]}")
                print(f"     date: {target_stock_data['date'].iloc[0]}")
                print(f"     close: {target_stock_data.get('close', [np.nan])[0]}")
                print(f"     volume: {target_stock_data.get('volume', [np.nan])[0]}")
                print(f"     amount: {target_stock_data.get('amount', [np.nan])[0]}")

        except Exception as e:
            print(f"   读取文件失败: {e}")
            return

    # 2. 检查新股列表构建
    print(f"\n2. 检查新股列表构建:")
    try:
        new_stocks = _build_new_stock_list_for_date(target_date)
        print(f"   识别的新股数量: {len(new_stocks)}")
        print(f"   目标股票是否在新股列表中: {target_stock in new_stocks}")

        if target_stock in new_stocks:
            print(f"   [OK] 目标股票被正确识别为新股")
        else:
            print(f"   [FAIL] 目标股票没有被识别为新股")

        # 显示前20个新股
        new_stocks_list = list(new_stocks)[:20]
        print(f"   前20个新股: {new_stocks_list}")

    except Exception as e:
        print(f"   构建新股列表失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 检查过滤逻辑
    print(f"\n3. 检查过滤逻辑:")
    if target_file.exists():
        try:
            # 创建简单的logger
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            # 执行过滤
            filter_result = _filter_new_stocks(daily_data, target_date, logger)

            print(f"   过滤结果形状: {filter_result.shape}")
            print(f"   目标股票过滤结果: {filter_result[daily_data['stock_code'] == target_stock].iloc[0] if len(daily_data[daily_data['stock_code'] == target_stock]) > 0 else 'N/A'}")

            # 统计过滤情况
            total_stocks = len(filter_result)
            remaining_stocks = filter_result.sum()
            filtered_stocks = total_stocks - remaining_stocks

            print(f"   总股票数: {total_stocks}")
            print(f"   剩余股票数: {remaining_stocks}")
            print(f"   过滤股票数: {filtered_stocks}")

        except Exception as e:
            print(f"   执行过滤失败: {e}")
            import traceback
            traceback.print_exc()

    # 4. 分析历史交易天数
    print(f"\n4. 分析历史交易天数:")
    try:
        trading_dates = _get_previous_trading_dates(target_date, 30)
        print(f"   获取到 {len(trading_dates)} 个历史交易日")
        print(f"   前10个交易日: {trading_dates[:10]}")

        # 统计目标股票的交易天数
        stock_trading_days = 0
        trading_dates_with_data = []

        for date_str in trading_dates:
            file_path = features_dir / f"{date_str}.parquet"
            if file_path.exists():
                try:
                    date_data = pd.read_parquet(file_path)
                    target_in_date = date_data[date_data['stock_code'] == target_stock]

                    if len(target_in_date) > 0:
                        volume = target_in_date.get('volume', [np.nan])[0]
                        if not np.isnan(volume) and volume > 0:
                            stock_trading_days += 1
                            trading_dates_with_data.append(date_str)

                except Exception as e:
                    print(f"     读取 {date_str} 失败: {e}")

        print(f"   目标股票有效交易天数: {stock_trading_days}")
        print(f"   有数据的前10个交易日: {trading_dates_with_data[:10]}")

        if stock_trading_days < 20:
            print(f"   [OK] 应该被过滤（少于20个有效交易日）")
        else:
            print(f"   [FAIL] 不应该被过滤（有{stock_trading_days}个有效交易日）")

    except Exception as e:
        print(f"   分析交易天数失败: {e}")
        import traceback
        traceback.print_exc()

    # 5. 检查缓存情况
    print(f"\n5. 检查缓存情况:")
    try:
        cache = _get_new_stock_cache()
        print(f"   缓存中的日期数量: {len(cache)}")
        print(f"   目标日期在缓存中: {target_date in cache}")

        if target_date in cache:
            cached_new_stocks = cache[target_date]
            print(f"   缓存中的新股数量: {len(cached_new_stocks)}")
            print(f"   目标股票在缓存新股中: {target_stock in cached_new_stocks}")

    except Exception as e:
        print(f"   检查缓存失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 80)
    print(f"调试完成")
    print("=" * 80)

if __name__ == "__main__":
    debug_new_stock_filter()