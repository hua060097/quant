#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试stock_sectors_mapping.py文件能否正确加载数据
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mapping")

# 添加当前目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # 尝试导入stock_sectors_mapping模块
    from stock_sectors_mapping import get_tng_sectors, get_sector_stocks, load_data

    # 检查数据加载
    logger.info("尝试加载数据...")
    data = load_data()
    
    if data is None:
        logger.error("数据加载失败")
    else:
        logger.info("数据加载成功")
        logger.info(f"数据更新时间: {data.get('update_time', '未知')}")
        logger.info(f"数据包含以下键: {list(data.keys())}")
        
        # 检查TGN板块列表
        sectors = get_tng_sectors()
        logger.info(f"TGN板块数量: {len(sectors)}")
        if sectors:
            logger.info(f"样例板块: {sectors[:5]}")
            
            # 尝试获取第一个板块的股票
            first_sector = sectors[0]
            stocks = get_sector_stocks(first_sector)
            logger.info(f"板块 '{first_sector}' 包含 {len(stocks)} 只股票")
            if stocks:
                logger.info(f"样例股票: {stocks[:5]}")
            else:
                logger.warning(f"板块 '{first_sector}' 没有股票")
        else:
            logger.warning("未找到任何TGN板块")
            
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
except Exception as e:
    logger.error(f"测试过程中发生错误: {e}")
    logger.exception(e)
    
logger.info("测试完成") 