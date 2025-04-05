#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试加载真实训练的模型功能
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型加载相关功能
from advanced_quant_model import SimpleAdvancedQuantModel, _load_models_for_date

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_load_trained_models():
    """测试加载真实训练好的模型"""
    # 测试模型目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    logger.info(f"模型目录: {model_dir}")
    
    # 列出模型目录下的所有板块文件夹
    sector_dirs = [d for d in os.listdir(model_dir) 
                  if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("TGN")]
    
    logger.info(f"发现 {len(sector_dirs)} 个板块目录:")
    for sector in sector_dirs:
        logger.info(f"  - {sector}")
    
    # 测试日期 - 使用我们之前训练的模型日期
    test_dates = ["20230401", "20230402"]
    
    for date_str in test_dates:
        logger.info(f"\n测试加载日期 {date_str} 的模型")
        
        # 遍历每个板块目录，查找对应日期的模型文件
        found_models = []
        for sector in sector_dirs:
            model_path = os.path.join(model_dir, sector, f"{date_str}.pkl")
            if os.path.exists(model_path):
                found_models.append((sector, model_path))
        
        logger.info(f"日期 {date_str} 找到 {len(found_models)} 个模型文件:")
        for sector, path in found_models:
            logger.info(f"  - {sector}: {path}")
        
        # 使用_load_models_for_date函数加载模型
        logger.info(f"\n使用_load_models_for_date加载日期 {date_str} 的模型")
        models_dict = _load_models_for_date(date_str, model_dir)
        
        if models_dict:
            logger.info(f"成功加载 {len(models_dict)} 个板块模型:")
            for sector_name in models_dict:
                logger.info(f"  - {sector_name}")
                # 打印模型的基本信息
                try:
                    model_data = models_dict[sector_name]
                    if hasattr(model_data, 'get'):
                        model_type = model_data.get('type', 'Unknown')
                        train_score = model_data.get('train_score', None)
                        test_score = model_data.get('test_score', None)
                        logger.info(f"    类型: {model_type}, 训练得分: {train_score}, 测试得分: {test_score}")
                    else:
                        logger.info(f"    模型类型: {type(model_data)}")
                except Exception as e:
                    logger.error(f"    提取模型信息时出错: {e}")
        else:
            logger.warning(f"未能加载日期 {date_str} 的任何模型")
    
    # 测试使用SimpleAdvancedQuantModel加载模型
    logger.info(f"\n测试通过SimpleAdvancedQuantModel加载模型")
    model = SimpleAdvancedQuantModel()
    if model.load_models(test_dates[0]):
        loaded_sectors = model.get_available_sectors()
        logger.info(f"可用板块数量: {len(loaded_sectors)}")
        if loaded_sectors:
            logger.info("可用板块列表:")
            for sector in loaded_sectors:
                logger.info(f"  - {sector}")
    else:
        logger.warning(f"SimpleAdvancedQuantModel未能加载日期 {test_dates[0]} 的模型")

if __name__ == "__main__":
    logger.info("开始测试真实训练的模型加载功能")
    test_load_trained_models()
    logger.info("模型加载测试完成") 