#!/usr/bin/env python3
"""
因子系统模块

基于2020-2025年顶尖量化研究的模块化因子系统架构
支持灵活的因子生成、测试和管理
"""

from .factor_manager import FactorManager
from .factor_cache import FactorCacheManager
from .factor_config import FactorConfig
from .factor_validator import FactorValidator

__all__ = [
    'FactorManager',
    'FactorCacheManager', 
    'FactorConfig',
    'FactorValidator'
]

__version__ = '1.0.0'
__author__ = 'Quantitative Research Team'
