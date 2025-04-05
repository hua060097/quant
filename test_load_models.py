#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试加载模型功能
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
from advanced_quant_model import _load_models_for_date

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_load_models():
    """测试加载训练好的模型"""
    # 测试模型目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    logger.info(f"模型目录: {model_dir}")
    
    # 直接列出 TGN区块链 目录下的模型文件
    sector_dir = os.path.join(model_dir, "TGN区块链")
    if os.path.exists(sector_dir):
        logger.info(f"TGN区块链目录存在: {sector_dir}")
        model_files = os.listdir(sector_dir)
        logger.info(f"目录中的文件: {model_files}")
    else:
        logger.warning(f"TGN区块链目录不存在: {sector_dir}")
    
    # 测试日期 - 使用我们之前训练的模型日期
    test_dates = ["20230401", "20230402"]
    
    for date_str in test_dates:
        logger.info(f"\n测试加载日期 {date_str} 的模型")
        
        # 检查特定文件是否存在
        model_path = os.path.join(model_dir, "TGN区块链", f"{date_str}.pkl")
        if os.path.exists(model_path):
            logger.info(f"模型文件存在: {model_path}")
            logger.info(f"文件大小: {os.path.getsize(model_path)} 字节")
            
            # 尝试加载这个模型文件
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"模型加载成功，类型: {type(model)}")
                if hasattr(model, 'keys'):
                    logger.info(f"模型键: {list(model.keys())}")
            except Exception as e:
                logger.error(f"加载模型文件时出错: {e}")
        else:
            logger.warning(f"模型文件不存在: {model_path}")
        
        # 使用_load_models_for_date函数加载模型
        logger.info(f"尝试使用_load_models_for_date加载日期为 {date_str} 的模型...")
        models_dict = _load_models_for_date(date_str, model_dir)
        
        if models_dict:
            logger.info(f"成功加载 {len(models_dict)} 个板块模型:")
            for sector_name in models_dict:
                logger.info(f"  - {sector_name}")
        else:
            logger.warning(f"未能加载日期 {date_str} 的任何模型")

if __name__ == "__main__":
    logger.info("开始测试模型加载功能")
    test_load_models()
    logger.info("模型加载测试完成") 