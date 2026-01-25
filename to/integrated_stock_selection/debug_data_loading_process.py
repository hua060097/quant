#!/usr/bin/env python3
"""
调试数据加载过程，找出为什么缓存文件内容不完整
"""

import pandas as pd
import sys
sys.path.append('..')

from data_loader import IntegratedDataLoader

def debug_data_loading():
    """调试数据加载过程"""

    print("=== 调试数据加载过程 ===")

    data_loader = IntegratedDataLoader()

    # 模拟用户参数
    start_date = "20251021"
    end_date = "20251216"
    extend_history_days = 120

    print(f"用户请求范围: {start_date} - {end_date}")
    print(f"扩展历史天数: {extend_history_days}")

    # 步骤1：检查扩展开始日期计算
    print(f"\n步骤1：计算扩展开始日期")
    try:
        extended_start_date = data_loader._calculate_extended_start_date(
            start_date, end_date, extend_history_days
        )
        print(f"扩展开始日期: {extended_start_date}")
    except Exception as e:
        print(f"计算扩展开始日期失败: {e}")
        return

    # 步骤2：检查交易日历
    print(f"\n步骤2：检查交易日历")
    try:
        trading_dates = data_loader.get_trading_dates(extended_start_date, end_date)
        print(f"交易日数量: {len(trading_dates)}")
        if len(trading_dates) > 0:
            print(f"交易日范围: {trading_dates[0]} - {trading_dates[-1]}")
            print(f"最后10个交易日: {trading_dates[-10:]}")
        else:
            print("❌ 没有找到交易日")
            return
    except Exception as e:
        print(f"获取交易日失败: {e}")
        return

    # 步骤3：检查每个交易日对应的文件是否存在
    print(f"\n步骤3：检查数据文件存在性")
    missing_files = []
    found_files = []

    from pathlib import Path
    current_file_dir = Path(__file__).parent
    features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

    for date_str in trading_dates[-20:]:  # 检查最后20个交易日
        file_path = features_dir / f"{date_str}.parquet"
        if file_path.exists():
            found_files.append(date_str)
        else:
            missing_files.append(date_str)

    print(f"检查的最后20个交易日:")
    print(f"  文件存在: {len(found_files)}个")
    print(f"  文件缺失: {len(missing_files)}个")

    if missing_files:
        print(f"缺失的文件: {missing_files}")
        print("❌ 这可能是缓存文件数据不完整的原因！")
    else:
        print("✅ 所有文件都存在")

    # 步骤4：检查缓存逻辑
    print(f"\n步骤4：检查缓存逻辑")
    cache_key = f"stock_features_{extended_start_date}_{end_date}"
    cache_file = data_loader.cache_dir / f"{cache_key}.parquet"
    print(f"缓存文件路径: {cache_file}")
    print(f"缓存文件存在: {cache_file.exists()}")

    # 步骤5：手动加载几个文件验证数据完整性
    print(f"\n步骤5：验证源文件数据完整性")
    test_dates = ["20251210", "20251211", "20251212", "20251215", "20251216"]

    for date_str in test_dates:
        file_path = features_dir / f"{date_str}.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                print(f"{date_str}.parquet: {len(df):,}条记录, 股票数={df['stock_code'].nunique():,}")
            except Exception as e:
                print(f"{date_str}.parquet: 读取失败 - {e}")
        else:
            print(f"{date_str}.parquet: 文件不存在")

    # 步骤6：分析为什么缓存只到2025-10-21
    print(f"\n步骤6：分析缓存不完整的原因")

    if missing_files:
        print("可能原因1：部分源数据文件缺失，导致加载过程中断")

    # 检查加载过程是否在中途出错
    print("可能原因2：数据加载过程中出现异常，导致部分数据未被保存到缓存")
    print("可能原因3：缓存生成过程中只处理了部分文件")

if __name__ == "__main__":
    debug_data_loading()