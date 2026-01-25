#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试优化后的分钟数据下载
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_quant_data_fixed import ParquetDataStorage, logger
import pandas as pd

# 创建存储实例
storage = ParquetDataStorage()

# 测试参数 - 只下载2只股票的数据
test_stocks = ['600051.SH', '605090.SH']  # 日志中提到的两只股票
start_date = pd.to_datetime('20241101')  # 测试一天的数据
end_date = pd.to_datetime('20241101')

print(f"测试下载 {len(test_stocks)} 只股票的5分钟数据")
print(f"时间范围: {start_date} - {end_date}")

# 执行下载
try:
    downloaded_count = storage.download_minute_data(
        stock_list=test_stocks,
        start_date=start_date,
        end_date=end_date,
        batch_size=2,  # 小批次
        max_workers=1  # 单线程
    )
    print(f"\n成功下载 {downloaded_count} 只股票的数据")
except Exception as e:
    print(f"\n下载失败: {e}")
    import traceback
    traceback.print_exc()