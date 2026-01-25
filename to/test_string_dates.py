#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试字符串日期格式的下载
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_quant_data_fixed import ParquetDataStorage

# 创建存储实例
storage = ParquetDataStorage()

# 测试字符串日期格式
test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']
start_date = '20241101'  # 字符串格式 YYYYMMDD
end_date = '20241102'

print(f"测试字符串日期格式")
print(f"股票: {test_stocks}")
print(f"开始日期: {start_date} (类型: {type(start_date)})")
print(f"结束日期: {end_date} (类型: {type(end_date)})")
print("-" * 60)

# 执行下载
try:
    downloaded = storage.download_minute_data(
        stock_list=test_stocks,
        start_date=start_date,
        end_date=end_date,
        batch_size=3,
        max_workers=1
    )
    print(f"\n下载成功: {downloaded}")
except Exception as e:
    print(f"\n下载失败: {e}")
    import traceback
    traceback.print_exc()