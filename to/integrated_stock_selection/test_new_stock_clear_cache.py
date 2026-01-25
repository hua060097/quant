#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新股过滤修复（清除缓存） - 验证修复后的效果
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
current_file_dir = Path(__file__).parent
sys.path.append(str(current_file_dir))

def test_new_stock_clear_cache():
    """测试新股过滤修复效果（清除缓存）"""

    # 目标股票和日期
    target_stock = "301551.SZ"
    target_date = "20240926"

    print("=" * 80)
    print(f"测试新股过滤修复（清除缓存）: 股票 {target_stock} 在日期 {target_date}")
    print("=" * 80)

    try:
        from main import _get_new_stock_cache, _save_new_stock_cache, _filter_new_stocks

        # 1. 清除缓存
        print(f"\n1. 清除新股缓存:")
        cache = _get_new_stock_cache()

        print(f"   缓存中是否有目标日期数据: {target_date in cache}")
        if target_date in cache:
            cached_new_stocks = cache[target_date]
            print(f"   缓存中目标日期的新股数量: {len(cached_new_stocks)}")
            print(f"   目标股票在缓存新股中: {target_stock in cached_new_stocks}")

        # 删除目标日期的缓存
        if target_date in cache:
            del cache[target_date]
            _save_new_stock_cache(cache)
            print(f"   已删除 {target_date} 的缓存数据")
        else:
            print(f"   缓存中没有 {target_date} 的数据")

        # 2. 读取数据并测试过滤
        print(f"\n2. 测试修复后的过滤逻辑:")

        # 创建logger
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # 读取目标日期数据
        features_dir = Path(__file__).parent.parent / "local_data" / "features" / "daily"
        target_file = features_dir / f"{target_date}.parquet"

        if target_file.exists():
            daily_data = pd.read_parquet(target_file)
            print(f"   数据文件存在，股票数量: {daily_data['stock_code'].nunique()}")

            # 执行过滤（现在会重新构建新股列表）
            filter_result = _filter_new_stocks(daily_data, target_date, logger)

            # 检查目标股票的过滤结果
            target_mask = daily_data['stock_code'] == target_stock
            if target_mask.any():
                target_filter_result = filter_result[target_mask].iloc[0]
                print(f"   目标股票过滤结果: {target_filter_result} (True=保留, False=过滤)")

                if not target_filter_result:
                    print(f"   [SUCCESS] 目标股票现在被正确过滤掉了!")
                else:
                    print(f"   [FAIL] 目标股票仍然没有被过滤")

                # 统计整体情况
                total_stocks = len(filter_result)
                remaining_stocks = filter_result.sum()
                filtered_stocks = total_stocks - remaining_stocks

                print(f"   总股票数: {total_stocks}")
                print(f"   剩余股票数: {remaining_stocks}")
                print(f"   过滤股票数: {filtered_stocks}")

            else:
                print(f"   [ERROR] 目标股票在数据中不存在")
        else:
            print(f"   [ERROR] 数据文件不存在: {target_file}")

        # 3. 验证缓存是否更新
        print(f"\n3. 验证缓存更新:")
        cache_after = _get_new_stock_cache()

        if target_date in cache_after:
            new_cached_stocks = cache_after[target_date]
            print(f"   缓存中目标日期的新股数量: {len(new_cached_stocks)}")
            print(f"   目标股票在更新后的缓存新股中: {target_stock in new_cached_stocks}")
        else:
            print(f"   缓存中仍未有目标日期的数据")

    except Exception as e:
        print(f"   测试失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 80)
    print(f"测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_new_stock_clear_cache()