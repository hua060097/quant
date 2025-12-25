#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级板块状态分类器
基于真实的板块特征和A股涨跌停特色，实现动态阈值的板块状态分类
参考ml_sector_strategy_features_documentation.md中的141个板块特征
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedSectorClassifier:
    """高级板块状态分类器 - 基于真实特征和动态阈值"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 基于实战炒股经验的A股板块子状态定义（11个状态）
        self.sub_states = {
            # Rising 上涨状态 - 基于涨停规律
            'rising_first_board_surge': '首板涨停潮',      # 情况1: 涨停数量高于历史中低位，3板以上为0，昨日溢价小于历史中位
            'rising_mid_board_leading': '中位连板带动',     # 情况2: 中位连板带动批量首板涨停跟风
            'rising_tier_formation': '涨停梯队形成',       # 情况3: 涨停形成梯队，高位连板，昨日溢价大于历史中高位
            'rising_gentle_warming': '温和上涨',          # 情况9: 无涨停或涨停少，但上涨比例高于历史中位

            # Differentiation 分化状态 - 基于连板分化
            'differentiation_high_board_weakening': '高位连板走弱',  # 情况4: 连板数量比首板多，高位连板开始走弱

            # Retreat 回调状态 - 基于跌停和断板
            'retreat_high_break_negative': '高位断板(负溢价)',    # 情况5: 高位断板，yesterday_max_limit_up_premium为负
            'retreat_high_break_positive': '高位断板(正溢价)',    # 情况6: 高位断板，yesterday_max_limit_up_premium为正
            'retreat_limit_down_tide': '跌停退潮',              # 情况7: 批量跌停，跌停数量高于历史中高位
            'retreat_limit_down_reducing': '跌停减少',          # 情况8: 跌停减少，跌停数量小于历史中位
            'retreat_gentle_decline': '温和下跌',               # 情况10: 无涨停无跌停，跌幅小于历史中低位
            'retreat_major_decline': '板块大跌'                 # 情况11: 跌幅大于等于历史中位
        }
        
        # 动态阈值存储
        self.dynamic_thresholds = {}
        
        # 关键特征列表（基于真实的板块特征和A股特色）
        self.key_features = {
            # 价格动量特征
            'price_momentum': ['avg_pct_chg', 'change_percent', 'momentum_5d', 'momentum_10d', 'momentum_20d'],

            # 涨跌停特征（A股特色）
            'limit_features': ['limit_up_count', 'limit_down_count', 'limit_up_ratio', 'limit_down_ratio',
                             'one_word_limit_up_count', 'broken_limit_up_count', 'max_consecutive_limit_up',
                             'near_limit_up_count', 'near_limit_down_count', 'limit_open_count', 'high_elasticity_count'],

            # 连板阶梯特征（A股特色）
            'board_tier_features': ['board_counts_1板', 'board_counts_2板', 'board_counts_3板', 'board_counts_4板',
                                  'board_counts_5板', 'board_counts_6板', 'board_counts_7板', 'board_counts_8板',
                                  'board_counts_9板', 'board_counts_10板及以上'],

            # 溢价特征（A股特色）
            'premium_features': ['yesterday_limit_up_premium', 'yesterday_max_limit_up_premium',
                               'yesterday_broken_limit_premium'],

            # 质量评价特征
            'quality_features': ['heat_index', 'one_word_limit_ratio', 'broken_ratio',
                                'sector_momentum_score', 'sector_strength'],

            # 市值分布特征
            'market_cap_features': ['market_cap_distribution_large', 'market_cap_distribution_medium',
                                  'market_cap_distribution_small', 'limit_board_distribution_large',
                                  'limit_board_distribution_medium', 'limit_board_distribution_small'],
            
            # 成交量特征
            'volume_features': ['volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20', 'avg_turnover_rate',
                              'volume_change_ratio', 'volume_trend'],
            
            # 波动率特征
            'volatility_features': ['volatility', 'volatility_5d', 'volatility_20d', 'avg_amplitude'],
            
            # 市值分布特征
            'market_cap_features': ['market_cap_distribution_large', 'market_cap_distribution_medium', 
                                  'market_cap_distribution_small'],
            
            # 技术指标特征
            'technical_features': ['ma5_slope', 'ma20_slope', 'trend_strength', 'rsi_macd_bull_signal',
                                 'rsi_macd_bear_signal'],
            
            # 板块结构特征
            'structure_features': ['stock_count', 'up_count', 'down_count', 'heat_index']
        }
        
    def classify_sectors(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        对板块数据进行状态分类
        
        Args:
            sector_data: 板块特征数据
            
        Returns:
            pd.DataFrame: 包含状态分类结果的数据
        """
        self.logger.info(f"开始对 {len(sector_data)} 条板块数据进行高级状态分类")
        
        # 计算动态阈值
        self._calculate_dynamic_thresholds(sector_data)

        # 计算A股特色的特殊阈值
        self._calculate_limit_thresholds(sector_data)
        
        result_data = sector_data.copy()
        
        # 添加状态列
        result_data['main_state'] = ''
        result_data['sub_state'] = ''
        result_data['confidence'] = 0.0
        result_data['state_strength'] = 0.0
        
        # 批量分类
        self._batch_classify_sectors(result_data)
        
        # 统计结果
        self._log_classification_stats(result_data)
        
        return result_data
    
    def _calculate_dynamic_thresholds(self, data: pd.DataFrame):
        """
        基于A股特色的精细化动态阈值计算
        参考顶尖量化研究：不同特征需要不同的分位数设置
        """
        self.logger.info("计算基于A股特色的精细化动态阈值...")

        # A股特色特征分类及其专业阈值设置
        feature_configs = {
            # 涨跌幅类特征：A股日内波动特点，需要更精细的分位数
            'price_features': {
                'features': ['avg_pct_chg', 'volatility', 'amplitude', 'change_percent'],
                'quantiles': [0.05, 0.15, 0.35, 0.65, 0.85, 0.95],  # 更密集的分位数
                'min_threshold': 0.005  # 最小阈值差异0.5%
            },
            # 涨跌停特征：A股独有，需要考虑10%涨跌停制度
            'limit_features': {
                'features': ['limit_up_count', 'limit_down_count', 'broken_limit_up_count', 'one_word_limit_ratio', 'broken_ratio'],
                'quantiles': [0.0, 0.1, 0.3, 0.7, 0.9, 1.0],  # 偏向低值，因为涨跌停是稀有事件
                'min_threshold': 0.1  # 最小阈值差异
            },
            # 溢价特征：游资关注的核心指标
            'premium_features': {
                'features': ['yesterday_limit_up_premium', 'yesterday_max_limit_up_premium', 'yesterday_broken_limit_premium'],
                'quantiles': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],  # 均匀分布，溢价变化敏感
                'min_threshold': 0.002  # 最小阈值差异0.2%
            },
            # 连板特征：A股特色，高连板稀有但重要
            'board_features': {
                'features': ['board_counts_1板', 'board_counts_2板', 'board_counts_3板', 'board_counts_4板',
                           'board_counts_5板', 'board_counts_6板', 'board_counts_7板', 'board_counts_8板',
                           'board_counts_9板', 'board_counts_10板及以上'],
                'quantiles': [0.0, 0.05, 0.2, 0.8, 0.95, 1.0],  # 极度偏向低值，高连板极稀有
                'min_threshold': 0.1  # 最小阈值差异
            },
            # 比例特征：相对稳定的指标
            'ratio_features': {
                'features': ['up_ratio', 'down_ratio', 'volume_ratio_5', 'limit_up_ratio', 'limit_down_ratio'],
                'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9, 0.95],  # 标准分位数
                'min_threshold': 0.05  # 最小阈值差异5%
            },
            # 热度特征：市场情绪指标
            'sentiment_features': {
                'features': ['heat_index', 'participation_entropy'],
                'quantiles': [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],  # 情绪变化敏感
                'min_threshold': 0.1  # 最小阈值差异
            }
        }

        # 获取所有可用的特征
        all_features = []
        for config in feature_configs.values():
            all_features.extend(config['features'])

        # 只使用数据中存在的特征
        available_features = [f for f in all_features if f in data.columns]

        if not available_features:
            self.logger.warning("没有找到关键特征，使用默认阈值")
            self._set_default_thresholds()
            return

        # 按板块分别计算阈值
        self.sector_thresholds = {}
        unique_sectors = data['sector_code'].unique() if 'sector_code' in data.columns else ['default']

        for sector in unique_sectors:
            if 'sector_code' in data.columns:
                sector_data = data[data['sector_code'] == sector]
            else:
                sector_data = data

            if len(sector_data) < 10:  # 数据太少，使用全局阈值
                continue

            self.sector_thresholds[sector] = {}

            # 为每类特征计算专业化阈值
            for feature_type, config in feature_configs.items():
                for feature in config['features']:
                    if feature in sector_data.columns:
                        feature_data = sector_data[feature].dropna()
                        if len(feature_data) > 5:
                            # 检查数据质量
                            non_zero_ratio = (feature_data != 0).sum() / len(feature_data)

                            if non_zero_ratio < 0.1:  # 处理全零数据
                                self._handle_zero_data_threshold(data, sector, feature, feature_type, config)
                            else:
                                # 使用专业化分位数计算阈值
                                quantiles = config['quantiles']
                                threshold_values = [feature_data.quantile(q) for q in quantiles]

                                # 确保阈值的合理性和单调性
                                threshold_values = self._ensure_threshold_monotonicity(
                                    threshold_values, config['min_threshold'])

                                self.sector_thresholds[sector][feature] = {
                                    'very_low': threshold_values[0],
                                    'low': threshold_values[1],
                                    'medium_low': threshold_values[2],
                                    'medium_high': threshold_values[3],
                                    'high': threshold_values[4],
                                    'very_high': threshold_values[5]
                                }


        # 计算全局阈值作为备用
        self.dynamic_thresholds = {}
        for feature in available_features:
            feature_data = data[feature].dropna()
            if len(feature_data) > 0:
                self.dynamic_thresholds[feature] = {
                    'very_low': feature_data.quantile(0.10),
                    'low': feature_data.quantile(0.25),
                    'medium': feature_data.quantile(0.50),
                    'high': feature_data.quantile(0.75),
                    'very_high': feature_data.quantile(0.90),
                    'extreme_high': feature_data.quantile(0.95)
                }

        # 计算A股特色的涨跌停阈值
        self._calculate_limit_thresholds(data)

        # 调试信息：检查数据质量
        self.logger.info("=== 数据质量检查 ===")
        sample_data = data.head(5)
        for idx, row in sample_data.iterrows():
            sector = row.get('sector_code', 'unknown')
            change_percent = row.get('change_percent', 'missing')
            limit_up = row.get('limit_up_count', 'missing')
            volatility = row.get('volatility', 'missing')
            self.logger.info(f"样本数据 - 板块:{sector}, change_percent:{change_percent}, limit_up:{limit_up}, volatility:{volatility}")

        # 统计非零数据
        non_zero_pct = (data['change_percent'] != 0).sum() if 'change_percent' in data.columns else 0
        total_count = len(data)
        self.logger.info(f"change_percent: {non_zero_pct}/{total_count} ({non_zero_pct/total_count*100:.1f}%)")

        # 显示部分板块的阈值
        sample_sectors = list(self.sector_thresholds.keys())[:3]  # 显示前3个板块
        for sector in sample_sectors:
            if 'change_percent' in self.sector_thresholds[sector]:
                thresholds = self.sector_thresholds[sector]['change_percent']
                self.logger.info(f"板块{sector}change_percent: low={thresholds['low']:.4f}, high={thresholds['high']:.4f}")

        self.logger.info(f"按板块动态阈值计算完成，覆盖 {len(unique_sectors)} 个板块，{len(available_features)} 个特征")
        self.logger.info(f"成功计算阈值的板块数: {len(self.sector_thresholds)}")

    def _handle_zero_data_threshold(self, data, sector, feature, feature_type, config):
        """处理全零数据的阈值计算"""
        # 使用全局数据计算阈值
        global_feature_data = data[feature].dropna()
        if len(global_feature_data) > 0:
            global_non_zero_ratio = (global_feature_data != 0).sum() / len(global_feature_data)
            if global_non_zero_ratio > 0.1:  # 全局数据有足够的非零值
                quantiles = config['quantiles']
                threshold_values = [global_feature_data.quantile(q) for q in quantiles]
                threshold_values = self._ensure_threshold_monotonicity(
                    threshold_values, config['min_threshold'])

                self.sector_thresholds[sector][feature] = {
                    'very_low': threshold_values[0],
                    'low': threshold_values[1],
                    'medium_low': threshold_values[2],
                    'medium_high': threshold_values[3],
                    'high': threshold_values[4],
                    'very_high': threshold_values[5]
                }
            else:
                # 如果全局数据也主要是零，使用基于特征类型的固定阈值
                self._set_fixed_thresholds_by_type(sector, feature, feature_type)

    def _set_fixed_thresholds_by_type(self, sector, feature, feature_type):
        """根据特征类型设置固定的合理阈值"""
        if feature_type == 'price_features':
            # 价格类特征：基于A股日内波动特点
            self.sector_thresholds[sector][feature] = {
                'very_low': -0.08, 'low': -0.03, 'medium_low': -0.01,
                'medium_high': 0.01, 'high': 0.03, 'very_high': 0.08
            }
        elif feature_type == 'limit_features':
            # 涨跌停特征：基于A股10%涨跌停制度
            self.sector_thresholds[sector][feature] = {
                'very_low': 0, 'low': 0, 'medium_low': 1,
                'medium_high': 3, 'high': 8, 'very_high': 20
            }
        elif feature_type == 'premium_features':
            # 溢价特征：基于游资操作特点
            self.sector_thresholds[sector][feature] = {
                'very_low': -0.05, 'low': -0.02, 'medium_low': -0.005,
                'medium_high': 0.005, 'high': 0.02, 'very_high': 0.05
            }
        elif feature_type == 'board_features':
            # 连板特征：基于A股连板分布
            self.sector_thresholds[sector][feature] = {
                'very_low': 0, 'low': 0, 'medium_low': 0,
                'medium_high': 1, 'high': 3, 'very_high': 10
            }
        elif feature_type == 'ratio_features':
            # 比例特征：基于市场参与度
            self.sector_thresholds[sector][feature] = {
                'very_low': 0.1, 'low': 0.25, 'medium_low': 0.4,
                'medium_high': 0.6, 'high': 0.75, 'very_high': 0.9
            }
        else:  # sentiment_features
            # 情绪特征：基于市场热度
            self.sector_thresholds[sector][feature] = {
                'very_low': 0, 'low': 0.2, 'medium_low': 0.4,
                'medium_high': 0.6, 'high': 0.8, 'very_high': 1.0
            }

    def _ensure_threshold_monotonicity(self, threshold_values, min_threshold):
        """确保阈值的单调性和合理的最小差异"""
        # 确保单调递增
        for i in range(1, len(threshold_values)):
            if threshold_values[i] <= threshold_values[i-1]:
                threshold_values[i] = threshold_values[i-1] + min_threshold

        # 确保最小差异
        for i in range(1, len(threshold_values)):
            if threshold_values[i] - threshold_values[i-1] < min_threshold:
                threshold_values[i] = threshold_values[i-1] + min_threshold

        return threshold_values
    
    def _calculate_limit_thresholds(self, data: pd.DataFrame):
        """计算A股涨跌停相关的特殊阈值"""
        
        # 涨停相关阈值
        if 'limit_up_count' in data.columns:
            limit_up_data = data['limit_up_count'].dropna()
            if len(limit_up_data) > 0:
                self.dynamic_thresholds['limit_cascade'] = {
                    'single_limit': 1,
                    'multiple_limit': limit_up_data.quantile(0.7),
                    'cascade_threshold': limit_up_data.quantile(0.9)
                }
        
        # 板块效应阈值
        if 'limit_up_ratio' in data.columns:
            ratio_data = data['limit_up_ratio'].dropna()
            if len(ratio_data) > 0:
                self.dynamic_thresholds['board_effect'] = {
                    'weak': ratio_data.quantile(0.5),
                    'strong': ratio_data.quantile(0.8),
                    'very_strong': ratio_data.quantile(0.95)
                }
        
        # 连板阈值
        if 'max_consecutive_limit_up' in data.columns:
            consecutive_data = data['max_consecutive_limit_up'].dropna()
            if len(consecutive_data) > 0:
                self.dynamic_thresholds['consecutive'] = {
                    'low': consecutive_data.quantile(0.6),
                    'high': consecutive_data.quantile(0.85)
                }
    
    def _set_default_thresholds(self):
        """设置默认阈值"""
        self.dynamic_thresholds = {
            'avg_pct_chg': {'very_low': -0.05, 'low': -0.02, 'medium': 0, 'high': 0.02, 'very_high': 0.05, 'extreme_high': 0.08},
            'volatility': {'very_low': 0.01, 'low': 0.02, 'medium': 0.03, 'high': 0.05, 'very_high': 0.08, 'extreme_high': 0.12},
            'volume_ratio_5': {'very_low': 0.5, 'low': 0.8, 'medium': 1.0, 'high': 1.5, 'very_high': 2.0, 'extreme_high': 3.0},
            'limit_cascade': {'single_limit': 1, 'multiple_limit': 3, 'cascade_threshold': 5},
            'board_effect': {'weak': 0.1, 'strong': 0.3, 'very_strong': 0.5},
            'consecutive': {'low': 2, 'high': 4}
        }
    
    def _batch_classify_sectors(self, data: pd.DataFrame):
        """批量分类板块状态"""
        
        for idx, row in data.iterrows():
            try:
                main_state, sub_state, confidence, strength = self._classify_single_sector(row)
                data.at[idx, 'main_state'] = main_state
                data.at[idx, 'sub_state'] = sub_state
                data.at[idx, 'confidence'] = confidence
                data.at[idx, 'state_strength'] = strength
            except Exception as e:
                self.logger.warning(f"分类失败 idx={idx}: {e}")
                # 使用默认分类
                data.at[idx, 'main_state'] = 'oscillation'
                data.at[idx, 'sub_state'] = 'oscillation_range_bound'
                data.at[idx, 'confidence'] = 0.5
                data.at[idx, 'state_strength'] = 0.0
    
    def _classify_single_sector(self, row: pd.Series) -> Tuple[str, str, float, float]:
        """
        对单个板块进行状态分类
        
        Args:
            row: 板块数据行
            
        Returns:
            tuple: (主状态, 子状态, 置信度, 状态强度)
        """
        # 提取关键特征
        features = self._extract_sector_features(row)
        
        # self.logger.info(f"features columns: {features.columns}")
        
        # 基于A股特色的分类逻辑
        main_state, sub_state, confidence, strength = self._classify_with_real_features(features)
        
        return main_state, sub_state, confidence, strength
    
    def _extract_sector_features(self, row: pd.Series) -> Dict[str, float]:
        """提取板块特征"""
        features = {}

        # 板块标识 - 关键修复！
        features['sector_code'] = row.get('sector_code', 'default')
        features['date'] = row.get('date', '')

        # 价格动量特征 - 使用实际存在的字段
        # features['avg_pct_chg'] = row.get('change_percent', 0)  # 使用change_percent作为avg_pct_chg
        features['change_percent'] = row.get('change_percent', 0)
        features['momentum_5d'] = row.get('momentum_5d', 0)
        features['momentum_20d'] = row.get('momentum_20d', 0)
        
        # A股特色：涨跌停特征 - 使用实际存在的丰富字段
        features['limit_up_count'] = row.get('limit_up_count', 0)
        features['limit_down_count'] = row.get('limit_down_count', 0)
        features['limit_up_ratio'] = row.get('limit_up_ratio', 0)
        features['limit_down_ratio'] = row.get('limit_down_ratio', 0)
        features['near_limit_up_count'] = row.get('near_limit_up_count', 0)
        features['near_limit_down_count'] = row.get('near_limit_down_count', 0)
        features['near_limit_up_ratio'] = row.get('near_limit_up_ratio', 0)
        features['one_word_limit_up_count'] = row.get('one_word_limit_up_count', 0)
        features['broken_limit_up_count'] = row.get('broken_limit_up_count', 0)
        features['limit_open_count'] = row.get('limit_open_count', 0)
        features['max_consecutive_limit_up'] = row.get('max_consecutive_limit_up', 0)
        features['high_elasticity_count'] = row.get('high_elasticity_count', 0)

        # A股特色：连板阶梯统计特征
        features['board_counts_1板'] = row.get('board_counts_1板', 0)
        features['board_counts_2板'] = row.get('board_counts_2板', 0)
        features['board_counts_3板'] = row.get('board_counts_3板', 0)
        features['board_counts_4板'] = row.get('board_counts_4板', 0)
        features['board_counts_5板'] = row.get('board_counts_5板', 0)
        features['board_counts_6板'] = row.get('board_counts_6板', 0)
        features['board_counts_7板'] = row.get('board_counts_7板', 0)
        features['board_counts_8板'] = row.get('board_counts_8板', 0)
        features['board_counts_9板'] = row.get('board_counts_9板', 0)
        features['board_counts_10板及以上'] = row.get('board_counts_10板及以上', 0)

        # A股特色：溢价特征（龙头、炸板溢价）
        features['yesterday_limit_up_premium'] = row.get('yesterday_limit_up_premium', 0)
        features['yesterday_max_limit_up_premium'] = row.get('yesterday_max_limit_up_premium', 0)
        features['yesterday_broken_limit_premium'] = row.get('yesterday_broken_limit_premium', 0)

        # A股特色：市值分布特征
        features['market_cap_distribution_large'] = row.get('market_cap_distribution_large', 0)
        features['market_cap_distribution_medium'] = row.get('market_cap_distribution_medium', 0)
        features['market_cap_distribution_small'] = row.get('market_cap_distribution_small', 0)
        features['limit_board_distribution_large'] = row.get('limit_board_distribution_large', 0)
        features['limit_board_distribution_medium'] = row.get('limit_board_distribution_medium', 0)
        features['limit_board_distribution_small'] = row.get('limit_board_distribution_small', 0)

        # A股特色：质量评价特征
        features['heat_index'] = row.get('heat_index', 0)
        features['one_word_limit_ratio'] = row.get('one_word_limit_ratio', 0)
        features['broken_ratio'] = row.get('broken_ratio', 0)
        features['sector_momentum_score'] = row.get('sector_momentum_score', 0)
        features['sector_strength'] = row.get('sector_strength', 0)
        
        # 成交量特征
        features['volume_ratio_5'] = row.get('volume_ratio_5', 1)
        features['volume_ratio_20'] = row.get('volume_ratio_20', 1)
        features['avg_turnover_rate'] = row.get('avg_turnover_rate', 0)
        features['volume_change_ratio'] = row.get('volume_change_ratio', 1)
        
        # 波动率特征
        features['volatility'] = row.get('volatility', 0)
        features['volatility_20d'] = row.get('volatility_20d', 0)
        features['avg_amplitude'] = row.get('avg_amplitude', 0)
        
        # 板块结构特征 - 使用实际存在的字段
        features['stock_count'] = row.get('stock_count', 0)
        features['up_count'] = row.get('up_count', 0)
        features['down_count'] = row.get('down_count', 0)
        features['flat_count'] = row.get('flat_count', 0)
        features['up_ratio'] = row.get('up_ratio', 0)
        features['down_ratio'] = row.get('down_ratio', 0)
        features['flat_ratio'] = row.get('flat_ratio', 0)
        features['heat_index'] = row.get('heat_index', 0)
        
        # 市值分布特征
        features['market_cap_distribution_large'] = row.get('market_cap_distribution_large', 0)
        features['market_cap_distribution_small'] = row.get('market_cap_distribution_small', 0)
        
        # 技术指标特征
        features['trend_strength'] = row.get('trend_strength', 0)
        features['ma5_slope'] = row.get('ma5_slope', 0)
        
        return features

    def _classify_with_real_features(self, features: Dict[str, float]) -> Tuple[str, str, float, float]:
        """基于真实特征进行分类 - 使用按板块的动态阈值"""

        # 获取关键指标
        change_percent = features['change_percent']
        limit_up_count = features['limit_up_count']
        limit_down_count = features['limit_down_count']
        up_count = features['up_count']
        down_count = features['down_count']
        stock_count = max(features['stock_count'], 1)

        # A股特色指标
        yesterday_limit_up_premium = features.get('yesterday_limit_up_premium', 0)
        yesterday_max_limit_up_premium = features.get('yesterday_max_limit_up_premium', 0)

        # 连板阶梯特征
        board_1 = features.get('board_counts_1板', 0)
        board_2 = features.get('board_counts_2板', 0)
        board_3_plus = sum([features.get(f'board_counts_{i}板', 0) for i in range(3, 11)]) + features.get('board_counts_10板及以上', 0)

        # 计算连板梯队数量（有多少个不同连板高度）
        board_tier_count = sum([1 for i in range(1, 11) if features.get(f'board_counts_{i}板', 0) > 0])
        if features.get('board_counts_10板及以上', 0) > 0:
            board_tier_count += 1

        # 计算关键比例（防止除零错误）
        up_ratio = up_count / max(stock_count, 1)
        down_ratio = down_count / max(stock_count, 1)

        # 获取板块特定的历史阈值
        sector_code = features.get('sector_code', 'default')
        sector_thresholds = self.sector_thresholds.get(sector_code, {})

        # 获取历史分位数阈值
        limit_up_thresholds = sector_thresholds.get('limit_up_count', {})
        limit_down_thresholds = sector_thresholds.get('limit_down_count', {})
        premium_thresholds = sector_thresholds.get('yesterday_limit_up_premium', {})
        up_ratio_thresholds = sector_thresholds.get('up_ratio', {})
        change_thresholds = sector_thresholds.get('change_percent', {})

        # 历史分位数（如果没有历史数据，使用经验值，防止除零）
        limit_up_low = max(limit_up_thresholds.get('low', 1), 0.1)        # 历史中低位
        limit_up_median = max(limit_up_thresholds.get('medium', 3), 0.1)   # 历史中位数
        limit_up_high = max(limit_up_thresholds.get('high', 6), 0.1)      # 历史中高位

        limit_down_low = max(limit_down_thresholds.get('low', 1), 0.1)     # 历史中低位
        limit_down_median = max(limit_down_thresholds.get('medium', 2), 0.1) # 历史中位数
        limit_down_high = max(limit_down_thresholds.get('high', 4), 0.1)   # 历史中高位

        premium_median = premium_thresholds.get('medium', 0.01)  # 历史中位数
        premium_high = premium_thresholds.get('high', 0.03)      # 历史中高位

        up_ratio_median = max(up_ratio_thresholds.get('medium', 0.5), 0.01) # 历史中位数

        change_low = change_thresholds.get('low', -0.02)         # 历史中低位
        change_median = change_thresholds.get('medium', 0.0)     # 历史中位数

        # 防止除零的安全检查
        if premium_median == 0:
            premium_median = 0.01
        if change_low == 0:
            change_low = -0.01
        if change_median == 0:
            change_median = 0.01

        # === 基于实战炒股经验的A股板块状态分类逻辑 ===

        # 情况1: 首板涨停潮 - 涨停数量高于历史中低位，3板以上为0，昨日溢价小于历史中位
        if (limit_up_count > limit_up_low and
            board_3_plus == 0 and
            yesterday_limit_up_premium < premium_median):
            score = (limit_up_count / limit_up_low) + (1 - abs(yesterday_limit_up_premium) / abs(premium_median))
            return 'rising', 'rising_first_board_surge', 0.9, score

        # 情况2: 中位连板带动 - 中位连板带动批量首板涨停跟风
        elif (board_2 > 0 and
              board_1 > limit_up_median):
            score = (board_2 * 2) + (board_1 / limit_up_median)
            return 'rising', 'rising_mid_board_leading', 0.85, score

        # 情况3: 涨停梯队形成 - 涨停形成梯队，高位连板，昨日溢价大于历史中高位
        elif (board_tier_count > 3 and
              board_3_plus > 0 and
              board_1 <= limit_up_median and
              yesterday_limit_up_premium > premium_high):
            score = board_tier_count + (board_3_plus * 2) + (yesterday_limit_up_premium / premium_high)
            return 'rising', 'rising_tier_formation', 0.95, score

        # 情况4: 高位连板走弱 - 连板数量比首板多，高位连板开始走弱（昨日溢价降低）
        elif (board_2 + board_3_plus > board_1 and
              board_3_plus > 0 and
              yesterday_limit_up_premium < premium_median):
            score = ((board_2 + board_3_plus) / max(board_1, 1)) + (1 - abs(yesterday_limit_up_premium) / abs(premium_median))
            return 'differentiation', 'differentiation_high_board_weakening', 0.8, score

        # 情况5: 高位断板(负溢价) - yesterday_max_limit_up_premium为负，板块跌停数量大于历史中低位
        elif (yesterday_max_limit_up_premium < 0 and
              limit_down_count > limit_down_low):
            score = abs(yesterday_max_limit_up_premium) * 10 + (limit_down_count / limit_down_low)
            return 'retreat', 'retreat_high_break_negative', 0.9, score

        # 情况6: 高位断板(正溢价) - yesterday_max_limit_up_premium为正，板块涨停数量小于等于历史中位数
        elif (yesterday_max_limit_up_premium > 0 and
              limit_up_count <= limit_up_median):
            score = yesterday_max_limit_up_premium * 5 + (1 - limit_up_count / limit_up_median)
            return 'retreat', 'retreat_high_break_positive', 0.75, score

        # 情况7: 跌停退潮 - 批量跌停，跌停数量高于历史中高位
        elif limit_down_count > limit_down_high:
            score = limit_down_count / limit_down_high
            return 'retreat', 'retreat_limit_down_tide', 0.95, score

        # 情况8: 跌停减少 - 跌停数量小于历史中位
        elif (limit_down_count > 0 and
              limit_down_count < limit_down_median):
            score = 1 - (limit_down_count / limit_down_median)
            return 'retreat', 'retreat_limit_down_reducing', 0.7, score

        # 情况9: 温和上涨 - 无涨停或涨停少，但上涨比例高于历史中位
        elif (limit_up_count <= limit_up_low and
              up_ratio > up_ratio_median):
            score = (up_ratio / up_ratio_median) + (1 - limit_up_count / limit_up_low)
            return 'rising', 'rising_gentle_warming', 0.7, score

        # 情况10: 温和下跌 - 无涨停无跌停，跌幅小于历史中低位
        elif (limit_up_count == 0 and
              limit_down_count == 0 and
              change_percent < change_median and
              change_percent > change_low):
            score = 1 - abs(change_percent / change_low)
            return 'retreat', 'retreat_gentle_decline', 0.6, score

        # 情况11: 板块大跌 - 跌幅大于等于历史中位
        elif change_percent <= change_median:
            score = abs(change_percent / abs(change_median))
            return 'retreat', 'retreat_major_decline', 0.8, score

        # 默认情况：如果不符合以上任何情况，根据基本指标判断
        else:
            # 根据涨跌幅和涨停情况进行基本分类
            if change_percent > 0 and limit_up_count > 0:
                score = change_percent * 5 + limit_up_count
                return 'rising', 'rising_gentle_warming', 0.5, score
            elif change_percent < 0 and limit_down_count > 0:
                score = abs(change_percent) * 5 + limit_down_count
                return 'retreat', 'retreat_gentle_decline', 0.5, score
            elif change_percent > 0:
                score = change_percent * 3
                return 'rising', 'rising_gentle_warming', 0.4, score
            elif change_percent < 0:
                score = abs(change_percent) * 3
                return 'retreat', 'retreat_gentle_decline', 0.4, score
            else:
                # 完全平盘，归为温和下跌
                return 'retreat', 'retreat_gentle_decline', 0.3, 0.1

    def _log_classification_stats(self, data: pd.DataFrame):
        """记录分类统计信息"""
        self.logger.info("=== 高级板块状态分类统计 ===")

        # 主状态分布
        main_state_counts = data['main_state'].value_counts()
        total_count = len(data)

        self.logger.info("主状态分布:")
        for state, count in main_state_counts.items():
            percentage = count / total_count * 100
            self.logger.info(f"  {state}: {count}条 ({percentage:.1f}%)")

        # 子状态分布
        sub_state_counts = data['sub_state'].value_counts()
        self.logger.info("子状态分布:")
        for state, count in sub_state_counts.items():
            percentage = count / total_count * 100
            self.logger.info(f"  {state}: {count}条 ({percentage:.1f}%)")

        # 置信度统计
        avg_confidence = data['confidence'].mean()
        self.logger.info(f"平均置信度: {avg_confidence:.3f}")

        # 检查子状态多样性
        unique_sub_states = len(sub_state_counts)
        total_possible_states = len(self.sub_states)
        coverage = unique_sub_states / total_possible_states * 100

        self.logger.info(f"子状态覆盖率: {unique_sub_states}/{total_possible_states} ({coverage:.1f}%)")

        if unique_sub_states < 8:  # 少于一半的子状态
            self.logger.warning("⚠️  子状态多样性不足，可能需要调整分类阈值")
        else:
            self.logger.info("✅ 子状态分布良好")
