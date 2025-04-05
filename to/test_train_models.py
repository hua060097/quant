#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本，用于验证train_models.py的功能

该脚本运行train_models.py中的run_offline_training函数，
使用最简单的参数测试模型训练和保存功能。
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import time

# 设置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到搜索路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入train_models中的run_offline_training函数
try:
    from train_models import run_offline_training
    logger.info("成功导入run_offline_training函数")
except ImportError as e:
    logger.error(f"导入run_offline_training函数失败: {e}")
    sys.exit(1)

def test_train_models():
    """测试train_models.py中的run_offline_training函数"""
    logger.info("开始测试train_models.py脚本")
    
    # 测试参数
    start_date = "20230401"
    end_date = "20230402"
    interval = 1
    sectors = ["TGN区块链", "TGN人工智能", "TGN云计算"]  # 只测试这三个板块
    workers = 1  # 使用单进程便于调试
    
    # 运行离线训练
    start_time = time.time()
    results = run_offline_training(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        sectors=sectors,
        workers=workers
    )
    duration = time.time() - start_time
    
    # 输出结果
    logger.info(f"测试完成，耗时: {duration:.2f}秒")
    logger.info(f"训练结果: {results}")
    
    # 检查模型文件是否已创建
    model_dir = Path("models")
    if not model_dir.exists():
        logger.error("模型目录未创建")
        return False
    
    for sector in sectors:
        sector_dir = model_dir / sector
        if not sector_dir.exists():
            logger.error(f"板块 {sector} 的模型目录未创建")
            continue
        
        model_files = list(sector_dir.glob("*.pkl"))
        if not model_files:
            logger.error(f"未找到板块 {sector} 的模型文件")
        else:
            logger.info(f"板块 {sector} 有 {len(model_files)} 个模型文件")
            for model_file in model_files:
                logger.info(f"  - {model_file.name} ({model_file.stat().st_size} 字节)")
    
    return results["success"] > 0

if __name__ == "__main__":
    success = test_train_models()
    sys.exit(0 if success else 1) 