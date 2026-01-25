#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
序列特征配置模块

此模块定义了序列特征处理的默认配置参数，包括特征类型、尺度设置和数据处理选项
"""

# 基本序列配置
DEFAULT_SEQUENCE_LENGTH = 10  # 默认序列长度
DEFAULT_TARGET_DAYS = 2  # 默认目标天数

# 多尺度特征配置
MULTI_SCALE_CONFIG = {
    'enabled': True,          # 是否启用多尺度特征
    'scopes': (5, 10, 20),    # 短期、中期、长期时间尺度
    'scales': ['short_term', 'mid_term', 'long_term'],  # 尺度名称
    'fusion_method': 'attention',  # 特征融合方法: 'attention', 'weighted', 'concat'
}

# 特征组配置
FEATURE_GROUPS = {
    # 趋势特征组
    'trend': {
        'enabled': True,
        'features': [
            'slope',
            'acceleration',
            'consistency',
            'momentum'
        ]
    },
    
    # 波动性特征组
    'volatility': {
        'enabled': True,
        'features': [
            'volatility',
            'range_ratio',
            'turbulence',
            'return_volatility',
            'volume_price_correlation'
        ]
    },
    
    # 周期性特征组
    'cyclical': {
        'enabled': True,
        'features': [
            'autocorr_lag1',
            'autocorr_lag2',
            'cycle_strength',
            'cycle_length'
        ]
    },
    
    # 位置特征组
    'position': {
        'enabled': True,
        'features': [
            'percentile',
            'z_score',
            'bb_position',
            'rsi'
        ]
    }
}

# 时间权重配置
TIME_WEIGHT_CONFIG = {
    'enabled': True,           # 是否启用时间权重
    'method': 'exponential',   # 权重计算方法: 'exponential', 'linear', 'sigmoid'
    'decay_rate': 0.1,         # 衰减率 (指数衰减)
    'recency_boost': 1.5       # 最近数据的权重提升因子
}

# 缓存配置
CACHE_CONFIG = {
    'enabled': True,          # 是否启用特征缓存
    'expire_days': 7,         # 缓存过期天数
    'compression': True,      # 是否压缩缓存
    'verbose': True           # 是否显示缓存信息
}

# 获取完整配置
def get_full_config():
    """获取完整配置字典"""
    return {
        'sequence': {
            'length': DEFAULT_SEQUENCE_LENGTH,
            'target_days': DEFAULT_TARGET_DAYS
        },
        'multi_scale': MULTI_SCALE_CONFIG,
        'feature_groups': FEATURE_GROUPS,
        'time_weight': TIME_WEIGHT_CONFIG,
        'cache': CACHE_CONFIG
    }

# 获取特定尺度的所有特征名
def get_scale_features(scale, selected_groups=None):
    """
    获取特定尺度下的所有特征名
    
    参数:
        scale: 尺度名称，如'short', 'mid', 'long'
        selected_groups: 选择的特征组列表，不指定则使用所有启用的组
        
    返回:
        特征名列表
    """
    features = []
    
    # 如果未指定组，使用所有启用的组
    if selected_groups is None:
        selected_groups = [group for group, config in FEATURE_GROUPS.items() 
                          if config['enabled']]
    
    # 收集所有组的特征
    for group in selected_groups:
        if group in FEATURE_GROUPS and FEATURE_GROUPS[group]['enabled']:
            for feature in FEATURE_GROUPS[group]['features']:
                features.append(f'{scale}_{feature}')
    
    return features 