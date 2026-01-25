#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试优化后的性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_quant_data_fixed import ParquetDataStorage
import pandas as pd
import time

# 创建存储实例
storage = ParquetDataStorage()

# 测试更大批次
test_stocks = [
    '600051.SH', '605090.SH', '000001.SZ', '000002.SZ',
    '000858.SZ', '002415.SZ', '600036.SH', '000725.SZ',
    '600519.SH', '000858.SZ', '002594.SZ', '601318.SH'
]
start_date = pd.to_datetime('20241101')
end_date = pd.to_datetime('20241102')  # 测试2天

print(f"测试下载 {len(test_stocks)} 只股票的5分钟数据")
print(f"时间范围: {start_date} - {end_date}")

# 执行下载
start_time = time.time()
try:
    downloaded_count = storage.download_minute_data(
        stock_list=test_stocks,
        start_date=start_date,
        end_date=end_date,
        batch_size=4,  # 4只股票一批
        max_workers=1  # 单线程
    )
    total_time = time.time() - start_time
    print(f"\n成功下载 {downloaded_count} 只股票的数据")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每只股票: {total_time/downloaded_count:.2f}秒")
except Exception as e:
    print(f"\n下载失败: {e}")
    import traceback
    traceback.print_exc()