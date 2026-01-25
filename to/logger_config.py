#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志配置模块
为整个项目提供统一的日志配置
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

def configure_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件名
        level: 日志级别
        console_output: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，不重复添加
    if logger.handlers:
        return logger
    
    logger.propagate = False  # 新增此行
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 如果指定了日志文件
    if log_file:
        # 获取项目根目录
        project_root = Path(__file__).resolve().parent
        
        # 构建日志目录
        log_dir = os.path.join(project_root, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        # # 构建日志文件路径
        log_path = os.path.join(log_dir, log_file)
        print(log_path)
        # 创建文件处理器
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # 添加到日志记录器
        logger.addHandler(file_handler)
    
    # 如果需要控制台输出
    if console_output:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # 添加到日志记录器
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name):
    """
    获取已配置的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name) 