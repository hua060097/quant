#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试301551.SZ调试日志 - 验证新添加的调试信息
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# 添加项目根目录到路径
current_file_dir = Path(__file__).parent
sys.path.append(str(current_file_dir))

def test_debug_301551_logs():
    """测试301551.SZ的调试日志"""

    # 目标股票和日期
    target_stock = "301551.SZ"
    target_date = "20240926"

    print("=" * 80)
    print(f"测试301551.SZ调试日志: 股票 {target_stock} 在日期 {target_date}")
    print("=" * 80)

    try:
        from main import _filter_new_stocks, _get_new_stock_cache, _save_new_stock_cache

        # 设置日志级别为INFO以查看调试信息
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)

        # 1. 清除缓存以确保重新计算
        print(f"\n1. 清除缓存并准备测试:")
        cache = _get_new_stock_cache()
        if target_date in cache:
            del cache[target_date]
            _save_new_stock_cache(cache)
            print(f"   已清除 {target_date} 的缓存")

        # 2. 读取数据并执行过滤
        print(f"\n2. 执行新股过滤（查看调试日志）:")

        features_dir = Path(__file__).parent.parent / "local_data" / "features" / "daily"
        target_file = features_dir / f"{target_date}.parquet"

        if target_file.exists():
            daily_data = pd.read_parquet(target_file)
            print(f"   数据文件存在，股票数量: {daily_data['stock_code'].nunique()}")
            print(f"   目标股票在数据中: {target_stock in daily_data['stock_code'].values}")

            # 执行过滤（会触发调试日志）
            print("\n   执行过滤函数...")
            filter_result = _filter_new_stocks(daily_data, target_date, logger)

            # 检查最终结果
            target_mask = daily_data['stock_code'] == target_stock
            if target_mask.any():
                target_final_result = filter_result[target_mask].iloc[0]
                print(f"\n   最终结果: {'过滤掉' if not target_final_result else '保留'} ({target_final_result})")
            else:
                print(f"\n   错误: 目标股票不在数据中")

        else:
            print(f"   错误: 数据文件不存在")

    except Exception as e:
        print(f"   测试失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 80)
    print(f"测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_debug_301551_logs()