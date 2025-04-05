#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复后的训练模型脚本
使用模拟的股票和特征数据进行测试
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import random
import time

# 设置日志 - 确保输出到控制台
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 清除现有的handler
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

# 添加控制台handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 添加项目根目录到搜索路径
current_dir = Path(__file__).resolve().parent 
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 强制打印一条消息，检查日志是否工作
print("测试脚本开始执行...")
logger.info("日志系统已初始化")

# 导入训练函数
from train_models import train_and_save_sector, load_features_for_training

def create_simulated_data():
    """创建模拟的股票和特征数据"""
    # 创建模拟的股票代码
    stock_codes = [f'00000{i}.SZ' for i in range(1, 21)]
    
    # 创建模拟的行业
    sector_name = "TGN测试行业"
    
    # 确保目录存在
    os.makedirs(f"models/{sector_name}", exist_ok=True)
    
    # 创建日期
    test_date = "20230401"
    
    return sector_name, test_date, stock_codes

def test_train_and_save_sector():
    """测试train_and_save_sector函数"""
    logger.info("开始测试train_and_save_sector函数")
    print("开始测试train_and_save_sector函数")
    
    # 创建模拟数据
    sector_name, test_date, stock_codes = create_simulated_data()
    logger.info(f"创建了模拟数据: 板块={sector_name}, 日期={test_date}, 股票数量={len(stock_codes)}")
    
    # 调用训练函数
    start_time = time.time()
    result = train_and_save_sector(
        sector_name=sector_name, 
        date=test_date, 
        sector_stocks=stock_codes, 
        lookback_short=30, 
        lookback_long=90
    )
    end_time = time.time()
    
    # 检查结果
    logger.info(f"训练结果: {'成功' if result else '失败'}")
    logger.info(f"训练耗时: {end_time - start_time:.2f} 秒")
    
    # 检查模型文件是否存在
    model_file = f"models/{sector_name}/{test_date}.pkl"
    if os.path.exists(model_file):
        logger.info(f"模型文件已创建: {model_file}")
        file_size = os.path.getsize(model_file) / 1024  # KB
        logger.info(f"模型文件大小: {file_size:.2f} KB")
    else:
        logger.error(f"模型文件未创建: {model_file}")

def main():
    """主函数"""
    print("=" * 50)
    print("开始测试修复后的训练模型脚本")
    print("=" * 50)
    
    logger.info("=" * 50)
    logger.info("开始测试修复后的训练模型脚本")
    logger.info("=" * 50)
    
    # 测试训练函数
    test_train_and_save_sector()
    
    logger.info("=" * 50)
    logger.info("测试完成")
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 