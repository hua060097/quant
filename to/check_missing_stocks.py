#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查2023年1月分区中哪些股票没有5分钟数据
"""

import sys
import logging
import os
from pathlib import Path

# 添加当前目录到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入ParquetDataStorage类
from advanced_quant_data import ParquetDataStorage, logger

def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建数据存储对象
    storage = ParquetDataStorage()
    
    # 检查2023年1月分区中的股票数据
    result = storage.check_missing_stocks_in_minute_data('202301')
    
    # 如果要检查2023年其他月份，可以使用以下代码：
    # result_202302 = storage.check_missing_stocks_in_minute_data('202302')  # 检查2月份
    # result_202303 = storage.check_missing_stocks_in_minute_data('202303')  # 检查3月份
    # result_202304 = storage.check_missing_stocks_in_minute_data('202304')  # 检查4月份
    
    # 保存缺失股票列表到文件
    missing_stocks = result.get('missing_stocks', [])
    available_stocks = result.get('available_stocks', [])
    
    if missing_stocks:
        output_file = 'missing_stocks_202301.txt'
        with open(output_file, 'w') as f:
            for stock in missing_stocks:
                f.write(f"{stock}\n")
        print(f"已将缺失股票列表保存到文件: {output_file}，共 {len(missing_stocks)} 只股票")
    
    # 如果想保存有数据的股票列表，可以使用以下代码
    if available_stocks:
        output_file = 'available_stocks_202301.txt'
        with open(output_file, 'w') as f:
            for stock in available_stocks:
                f.write(f"{stock}\n")
        print(f"已将有数据的股票列表保存到文件: {output_file}，共 {len(available_stocks)} 只股票")
    
    print("检查完成！")
    
    # 特别检查000001.SZ是否在可用股票列表中
    if '000001.SZ' in available_stocks:
        print("000001.SZ 在2023年1月的数据中")
    else:
        print("000001.SZ 不在2023年1月的数据中")

if __name__ == "__main__":
    main() 