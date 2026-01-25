#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试交易数据 - 检查301551.SZ的交易数据情况
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
current_file_dir = Path(__file__).parent
sys.path.append(str(current_file_dir))

def debug_trading_data():
    """调试交易数据"""

    # 目标股票和日期
    target_stock = "301551.SZ"
    target_date = "20240926"

    print("=" * 80)
    print(f"调试交易数据: 股票 {target_stock}")
    print("=" * 80)

    # 检查因子文件目录
    features_dir = Path(__file__).parent / "local_data" / "features" / "daily"
    print(f"\n因子文件目录: {features_dir}")
    print(f"目录存在: {features_dir.exists()}")

    # 检查目标日期前后的文件
    dates_to_check = ["20240924", "20240925", "20240926", "20240927", "20240930"]

    print(f"\n检查因子文件:")
    for date in dates_to_check:
        file_path = features_dir / f"{date}.parquet"
        exists = file_path.exists()
        print(f"   {date}: {'存在' if exists else '不存在'}")

        if exists:
            try:
                df = pd.read_parquet(file_path)
                target_data = df[df['stock_code'] == target_stock]
                has_target = len(target_data) > 0

                print(f"     总股票数: {df['stock_code'].nunique()}")
                print(f"     包含目标股票: {has_target}")

                if has_target:
                    volume = target_data.get('volume', [np.nan])[0]
                    close = target_data.get('close', [np.nan])[0]
                    print(f"     成交量: {volume}")
                    print(f"     收盘价: {close}")

            except Exception as e:
                print(f"     读取失败: {e}")

    # 检查基础数据目录
    print(f"\n检查基础数据目录:")
    base_dirs = [
        "local_data/daily",
        "local_data/minute",
        "local_data/market_status"
    ]

    for dir_name in base_dirs:
        dir_path = Path(__file__).parent / dir_name
        exists = dir_path.exists()
        print(f"   {dir_name}: {'存在' if exists else '不存在'}")

        if exists and "daily" in dir_name:
            # 检查daily目录下的2024年9月文件
            try:
                daily_files = list(dir_path.glob("202409*"))
                print(f"     202409文件数量: {len(daily_files)}")

                # 检查特定日期
                for date in dates_to_check:
                    date_file = dir_path / f"{date}.parquet"
                    if date_file.exists():
                        print(f"     {date}: 存在")

                        # 尝试读取并检查目标股票
                        try:
                            df = pd.read_parquet(date_file)
                            target_data = df[df['stock_code'] == target_stock]
                            if len(target_data) > 0:
                                print(f"       包含目标股票数据")
                            else:
                                print(f"       不包含目标股票")
                        except Exception as e:
                            print(f"       读取失败: {e}")
            except Exception as e:
                print(f"     检查失败: {e}")

    # 检查trading calendar
    print(f"\n检查交易日历:")
    calendar_file = Path(__file__).parent / "local_data" / "calendar" / "trading_calendar.parquet"
    if calendar_file.exists():
        try:
            calendar = pd.read_parquet(calendar_file)
            print(f"   交易日历文件存在，记录数: {len(calendar)}")

            # 检查目标日期是否为交易日
            if 'date' in calendar.columns:
                is_trading_day = target_date in calendar['date'].values
                print(f"   {target_date} 是否为交易日: {is_trading_day}")

                # 显示前后几个交易日
                if is_trading_day:
                    date_list = calendar['date'].tolist()
                    target_idx = date_list.index(target_date)
                    print(f"   前一个交易日: {date_list[target_idx-1] if target_idx > 0 else 'N/A'}")
                    print(f"   后一个交易日: {date_list[target_idx+1] if target_idx < len(date_list)-1 else 'N/A'}")

        except Exception as e:
            print(f"   读取交易日历失败: {e}")
    else:
        print(f"   交易日历文件不存在")

    # 检查metadata中的股票信息
    print(f"\n检查股票元数据:")
    metadata_file = Path(__file__).parent / "local_data" / "metadata" / "metadata.json"
    if metadata_file.exists():
        try:
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            print(f"   元数据文件存在")
            print(f"   股票总数: {len(metadata.get('stocks', {}))}")

            if target_stock in metadata.get('stocks', {}):
                stock_info = metadata['stocks'][target_stock]
                print(f"   目标股票信息存在:")
                print(f"     名称: {stock_info.get('name', 'N/A')}")
                print(f"     上市日期: {stock_info.get('list_date', 'N/A')}")
                print(f"     状态: {stock_info.get('status', 'N/A')}")
            else:
                print(f"   目标股票信息不存在")

        except Exception as e:
            print(f"   读取元数据失败: {e}")
    else:
        print(f"   元数据文件不存在")

    print(f"\n" + "=" * 80)
    print(f"调试完成")
    print("=" * 80)

if __name__ == "__main__":
    debug_trading_data()