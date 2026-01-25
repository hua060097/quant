#!/usr/bin/env python3
"""
因子组模块

包含各种类型的因子组实现
"""

from .base_factor_group import BaseFactorGroup
# 重新启用已修复的因子组
from .technical_factors import TechnicalFactors
from .fundamental_factors import FundamentalFactors
from .kline_pattern_factors import KlinePatternFactors
from .microstructure_factors import MicrostructureFactors
from .risk_factors import RiskFactors
from .momentum_factors import MomentumFactors
from .volume_factors import VolumeFactors
from .cross_sectional_factors import CrossSectionalFactors
# from .correlation_factors import CorrelationFactors  # 已删除，因子迁移到EnhancedCorrelationFactors
from .limit_up_quality_factors import LimitUpQualityFactors  # 重构为直接加载模式（v2.0）
from .enhanced_correlation_factors import EnhancedCorrelationFactors
from .comprehensive_ma_factors import ComprehensiveMAFactors
from .price_position_factors import PricePositionFactors
from .advanced_technical_factors import AdvancedTechnicalFactors

# 只启用已修复的因子组
from .sector_relative_factors import SectorRelativeFactors

# 新增：顶级支撑因子组
from .elite_support_factors import EliteSupportFactors

# 新增：市场相关性因子组
from .market_corr_factors import MarketCorrFactors

# 新增：市场相对因子组
from .market_relative_factors import MarketRelativeFactors

__all__ = [
    'BaseFactorGroup',
    # 重新启用已修复的因子组
    'TechnicalFactors',
    'FundamentalFactors',
    'KlinePatternFactors',
    'MicrostructureFactors',
    'RiskFactors',
    'MomentumFactors',
    'VolumeFactors',
    'CrossSectionalFactors',
    # 'CorrelationFactors',  # 已删除，因子迁移到EnhancedCorrelationFactors
    'LimitUpQualityFactors',  # 重构为直接加载模式（v2.0）
    'EnhancedCorrelationFactors',
    'ComprehensiveMAFactors',
    'PricePositionFactors',
    'AdvancedTechnicalFactors',

    # 只启用已修复的因子组
    'SectorRelativeFactors',

    # 新增：顶级支撑因子组
    'EliteSupportFactors',

    # 新增：市场相关性因子组
    'MarketCorrFactors',

    # 新增：市场相对因子组
    'MarketRelativeFactors'
]
