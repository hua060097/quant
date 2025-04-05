#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路径处理工具

提供项目路径处理相关的工具函数，避免路径混乱和冗余目录结构。
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional

# 获取logger
logger = logging.getLogger(__name__)

# 缓存项目根目录路径，避免重复计算
_PROJECT_ROOT = None


def get_project_path(relative_path: str = '') -> Path:
    """获取项目根目录下的路径
    
    Args:
        relative_path: 相对于项目根目录的路径
        
    Returns:
        完整路径
    """
    global _PROJECT_ROOT
    
    if _PROJECT_ROOT is None:
        # 从当前文件所在目录出发，找到项目根目录
        current_dir = Path(__file__).parent
        
        # 如果当前目录名为'to'，向上一级找到项目根目录
        if current_dir.name == 'to':
            _PROJECT_ROOT = current_dir.parent
        else:
            # 否则，当前目录可能已经是项目根目录
            _PROJECT_ROOT = current_dir
    
    # 合并路径
    if not relative_path:
        return _PROJECT_ROOT
    
    return _PROJECT_ROOT / relative_path


def get_data_path(relative_path: str = '') -> Path:
    """获取数据目录下的路径
    
    Args:
        relative_path: 相对于数据目录的路径
        
    Returns:
        完整路径
    """
    data_dir = get_project_path('local_data')
    
    # 确保目录存在
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not relative_path:
        return data_dir
    
    return data_dir / relative_path


def get_model_path(relative_path: str = '') -> Path:
    """获取模型目录下的路径
    
    Args:
        relative_path: 相对于模型目录的路径
        
    Returns:
        完整路径
    """
    model_dir = get_project_path('models')
    
    # 确保目录存在
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not relative_path:
        return model_dir
    
    return model_dir / relative_path


def get_log_path(relative_path: str = '') -> Path:
    """获取日志目录下的路径
    
    Args:
        relative_path: 相对于日志目录的路径
        
    Returns:
        完整路径
    """
    log_dir = get_project_path('log')
    
    # 确保目录存在
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if not relative_path:
        return log_dir
    
    return log_dir / relative_path


def get_results_path(relative_path: str = '') -> Path:
    """获取结果目录下的路径
    
    Args:
        relative_path: 相对于结果目录的路径
        
    Returns:
        完整路径
    """
    results_dir = get_project_path('results')
    
    # 确保目录存在
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not relative_path:
        return results_dir
    
    return results_dir / relative_path


def normalize_path(path_str: str) -> str:
    """标准化路径字符串
    
    检测并修复路径中的冗余目录结构。
    
    Args:
        path_str: 路径字符串
        
    Returns:
        标准化后的路径字符串
    """
    # 如果路径以'to/to/'开头，去掉一个'to/'
    if path_str.startswith('to/to/') or path_str.startswith('to\\to\\'):
        logger.warning(f"检测到冗余路径: {path_str}")
        path_str = path_str[3:]  # 去掉开头的'to/'
        logger.info(f"修复为: {path_str}")
    
    return path_str


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def check_and_warn_path(path_str: str, context: Optional[str] = None) -> str:
    """检查路径并发出警告
    
    检查路径是否存在潜在问题，如冗余目录结构。
    
    Args:
        path_str: 路径字符串
        context: 上下文信息，用于日志
        
    Returns:
        原始路径字符串，不做修改
    """
    # 检查是否有冗余目录结构
    if 'to/to/' in path_str or 'to\\to\\' in path_str:
        if context:
            logger.warning(f"{context}: 检测到冗余路径结构 'to/to/' in '{path_str}'")
        else:
            logger.warning(f"检测到冗余路径结构 'to/to/' in '{path_str}'")
    
    return path_str


if __name__ == "__main__":
    # 简单测试
    print(f"项目根目录: {get_project_path()}")
    print(f"数据目录: {get_data_path()}")
    print(f"模型目录: {get_model_path()}")
    print(f"日志目录: {get_log_path()}")
    print(f"结果目录: {get_results_path()}")
    
    # 测试路径标准化
    test_path = "to/to/local_data/features"
    print(f"原始路径: {test_path}")
    print(f"标准化后: {normalize_path(test_path)}") 