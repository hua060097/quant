#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试大批量数据的性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_quant_data_fixed import ParquetDataStorage
import pandas as pd
import time
import random

# 创建存储实例
storage = ParquetDataStorage()

# 生成更多股票用于测试
# 从现有股票中随机选择
all_stocks = [
    '600051.SH', '605090.SH', '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ',
    '600036.SH', '000725.SZ', '600519.SH', '000858.SZ', '002594.SZ', '601318.SH',
    '000063.SZ', '002027.SZ', '600276.SH', '000768.SZ', '600585.SH', '000596.SZ',
    '002304.SZ', '600104.SH', '000623.SZ', '600887.SH', '000778.SZ', '600030.SH'
]

# 随机选择50只股票
random.shuffle(all_stocks)
test_stocks = all_stocks[:50]

start_date = pd.to_datetime('20241101')
end_date = pd.to_datetime('20241103')  # 3天数据

print(f"测试下载 {len(test_stocks)} 只股票的5分钟数据")
print(f"时间范围: {start_date} - {end_date}")
print(f"预估数据量: ~{len(test_stocks) * 3 * 48 * 0.8:.0f} 条记录 (假设80%有效)")
print("-" * 60)

# 执行下载
start_time = time.time()
try:
    downloaded_count = storage.download_minute_data(
        stock_list=test_stocks,
        start_date=start_date,
        end_date=end_date,
        batch_size=10,  # 每批10只
        max_workers=1  # 单线程
    )
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"性能报告:")
    print(f"  成功下载: {downloaded_count} 批次")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均每只股票: {total_time/len(test_stocks):.2f}秒")
    print(f"  每秒处理: {len(test_stocks)/total_time:.1f} 只股票")

except Exception as e:
    print(f"\n下载失败: {e}")
    import traceback
    traceback.print_exc()