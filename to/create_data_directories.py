#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建系统所需的数据目录

确保系统所需的所有数据目录都存在，避免在运行时出现找不到目录的错误
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到系统路径
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

# 导入路径工具
try:
    from to.path_utils import get_project_path, get_data_path, ensure_dir
except ImportError:
    # 如果path_utils.py不存在，则提供兼容性处理
    def get_project_path(relative_path=None):
        """返回相对于项目根目录的绝对路径"""
        project_root = Path(__file__).resolve().parent.parent
        if relative_path:
            return project_root / relative_path
        return project_root
    
    def get_data_path(relative_path=None):
        """返回相对于数据目录的绝对路径"""
        data_root = get_project_path('to/local_data')
        if relative_path:
            return data_root / relative_path
        return data_root
    
    def ensure_dir(path):
        """确保目录存在，如果不存在则创建"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_data_directories():
    """创建系统所需的所有数据目录"""
    logger.info("开始创建系统所需的数据目录...")
    
    # 主要数据目录
    data_root = get_data_path()
    ensure_dir(data_root)
    logger.info(f"创建数据根目录: {data_root}")
    
    # 创建基本数据目录
    basic_dirs = [
        'calendar',    # 交易日历
        'daily',       # 日线数据
        'minute',      # 分钟线数据
        'metadata',    # 元数据
        'market_status'  # 市场状态
    ]
    
    for dir_name in basic_dirs:
        sub_path = get_data_path(dir_name)
        ensure_dir(sub_path)
        logger.info(f"创建数据子目录: {sub_path}")
    
    # 创建特征相关目录
    feature_dirs = [
        'features',         # 特征数据
        'sector_features',  # 板块特征
        'market_features'   # 市场特征
    ]
    
    for dir_name in feature_dirs:
        sub_path = get_data_path(dir_name)
        ensure_dir(sub_path)
        logger.info(f"创建特征子目录: {sub_path}")
    
    # 创建模型和预测相关目录
    model_dirs = [
        'models',       # 模型文件
        'predictions'   # 预测结果
    ]
    
    for dir_name in model_dirs:
        sub_path = get_data_path(dir_name)
        ensure_dir(sub_path)
        logger.info(f"创建模型子目录: {sub_path}")
    
    # 创建板块特殊数据目录
    sector_dirs = [
        'sector_data',   # 板块数据
        'index_weight'   # 指数权重
    ]
    
    for dir_name in sector_dirs:
        sub_path = get_data_path(dir_name)
        ensure_dir(sub_path)
        logger.info(f"创建板块相关子目录: {sub_path}")
    
    # 创建其他必要目录
    other_dirs = [
        'to/log',       # 日志目录
        'to/models',    # 模型定义目录
        'to/results'    # 结果目录
    ]
    
    for dir_path in other_dirs:
        full_path = get_project_path(dir_path)
        ensure_dir(full_path)
        logger.info(f"创建其他目录: {full_path}")
    
    logger.info("所有数据目录创建完成")

if __name__ == "__main__":
    create_data_directories()
    print("\n所有数据目录已创建完成，系统已准备就绪。")
    print("\n要查看目录结构的详细信息，请查阅README_PATH.md文件。")