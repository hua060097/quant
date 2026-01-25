#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新股过滤修复 - 验证301551.SZ是否现在能被正确识别为新股
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
current_file_dir = Path(__file__).parent
sys.path.append(str(current_file_dir))

def test_new_stock_fix():
    """测试新股过滤修复效果"""

    # 目标股票和日期
    target_stock = "301551.SZ"
    target_date = "20240926"

    print("=" * 80)
    print(f"测试新股过滤修复: 股票 {target_stock} 在日期 {target_date}")
    print("=" * 80)

    try:
        # 1. 测试修复后的新股识别逻辑
        from main import _build_new_stock_list_for_date

        print(f"\n1. 测试修复后的新股识别逻辑:")
        new_stocks = _build_new_stock_list_for_date(target_date)

        print(f"   识别的新股数量: {len(new_stocks)}")
        print(f"   目标股票是否在新股列表中: {target_stock in new_stocks}")

        if target_stock in new_stocks:
            print(f"   [SUCCESS] 目标股票现在被正确识别为新股!")
        else:
            print(f"   [FAIL] 目标股票仍然没有被识别为新股")

        # 显示前20只新股
        new_stocks_list = list(new_stocks)[:20]
        print(f"   前20只新股: {new_stocks_list}")

    except Exception as e:
        print(f"   测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 2. 测试完整的新股过滤流程
    print(f"\n2. 测试完整的新股过滤流程:")

    try:
        from main import _filter_new_stocks

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

            # 执行过滤
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

    except Exception as e:
        print(f"   测试过滤流程失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "=" * 80)
    print(f"测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_new_stock_fix()