#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的新股过滤调试 - 使用正确路径分析301551.SZ问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
current_file_dir = Path(__file__).parent
sys.path.append(str(current_file_dir))

def debug_new_stock_filter_correct():
    """使用正确路径调试新股过滤"""

    # 目标股票和日期
    target_stock = "301551.SZ"
    target_date = "20240926"

    print("=" * 80)
    print(f"正确路径调试: 股票 {target_stock} 在日期 {target_date}")
    print("=" * 80)

    # 1. 确认因子文件存在且包含目标股票
    features_dir = Path(__file__).parent.parent / "local_data" / "features" / "daily"
    target_file = features_dir / f"{target_date}.parquet"

    print(f"\n1. 确认因子文件:")
    print(f"   文件路径: {target_file}")
    print(f"   文件存在: {target_file.exists()}")

    daily_data = None
    if target_file.exists():
        daily_data = pd.read_parquet(target_file)
        target_stock_data = daily_data[daily_data['stock_code'] == target_stock]
        print(f"   总股票数: {daily_data['stock_code'].nunique()}")
        print(f"   目标股票存在: {len(target_stock_data) > 0}")

        if len(target_stock_data) > 0:
            print(f"   目标股票数据:")
            print(f"     volume: {target_stock_data['volume'].iloc[0]}")
            print(f"     close: {target_stock_data['close'].iloc[0]}")

    # 2. 测试新股过滤逻辑
    print(f"\n2. 测试新股过滤逻辑:")

    # 导入main.py中的函数
    try:
        from main import _filter_new_stocks

        # 创建logger
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # 执行过滤
        if daily_data is not None:
            filter_result = _filter_new_stocks(daily_data, target_date, logger)
        else:
            print(f"   跳过过滤测试，数据文件不存在")
            return

        print(f"   过滤结果形状: {filter_result.shape}")

        # 检查目标股票的过滤结果
        target_mask = daily_data['stock_code'] == target_stock
        if target_mask.any():
            target_filter_result = filter_result[target_mask].iloc[0]
            print(f"   目标股票过滤结果: {target_filter_result} (True=保留, False=过滤)")

            if target_filter_result:
                print(f"   ❌ 目标股票没有被过滤")
            else:
                print(f"   ✅ 目标股票被正确过滤")

        # 统计整体过滤情况
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

    # 3. 检查新股列表构建
    print(f"\n3. 检查新股列表构建:")

    try:
        from main import _build_new_stock_list_for_date

        new_stocks = _build_new_stock_list_for_date(target_date)
        print(f"   识别的新股数量: {len(new_stocks)}")
        print(f"   目标股票是否在新股列表中: {target_stock in new_stocks}")

        if target_stock in new_stocks:
            print(f"   [OK] 目标股票被正确识别为新股")
        else:
            print(f"   [FAIL] 目标股票没有被识别为新股")

        # 显示部分新股
        new_stocks_list = list(new_stocks)
        print(f"   前10个新股: {new_stocks_list[:10]}")

    except Exception as e:
        print(f"   构建新股列表失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. 手动计算历史交易天数
    print(f"\n4. 手动计算历史交易天数:")

    try:
        from main import _get_previous_trading_dates

        trading_dates = _get_previous_trading_dates(target_date, 30)
        print(f"   获取到 {len(trading_dates)} 个历史交易日")
        print(f"   前5个交易日: {trading_dates[:5]}")

        # 统计目标股票的交易天数
        stock_trading_days = 0
        first_trading_date = None

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
                            if first_trading_date is None:
                                first_trading_date = date_str

                except Exception as e:
                    print(f"     读取 {date_str} 失败: {e}")

        print(f"   目标股票有效交易天数: {stock_trading_days}")
        if first_trading_date:
            print(f"   第一个有效交易日: {first_trading_date}")

        if stock_trading_days < 20:
            print(f"   应该被过滤（少于20个有效交易日）")
        else:
            print(f"   不应该被过滤（有{stock_trading_days}个有效交易日）")

    except Exception as e:
        print(f"   计算交易天数失败: {e}")
        import traceback
        traceback.print_exc()

    # 5. 检查当日成交量过滤
    print(f"\n5. 检查当日成交量过滤:")

    try:
        target_data = daily_data[daily_data['stock_code'] == target_stock]
        volume = target_data['volume'].iloc[0] if len(target_data) > 0 else 0

        print(f"   当日成交量: {volume}")
        print(f"   成交量是否为0: {volume == 0}")
        print(f"   成交量是否为NaN: {np.isnan(volume)}")

        if volume == 0 or np.isnan(volume):
            print(f"   会因当日数据异常被过滤")
        else:
            print(f"   通过当日数据异常检查")

    except Exception as e:
        print(f"   检查成交量失败: {e}")

    print(f"\n" + "=" * 80)
    print(f"调试完成")
    print("=" * 80)

if __name__ == "__main__":
    debug_new_stock_filter_correct()