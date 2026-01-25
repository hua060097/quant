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
        
        # 基于板块竞争对比分析的子状态定义（15个状态）
        # 设计理念：结合同日其他板块的竞争态势，体现A股板块轮动和竞争特征
        self.sub_states = {
            # Rising 上涨状态 - 6个状态（增加竞争维度）
            'rising_dominant_leader': '绝对龙头',                   # R1: 连板梯队+无其他板块竞争
            'rising_competitive_leader': '竞争龙头',               # R2: 连板梯队+其他板块首板竞争
            'rising_cascade_battle': '连板对抗',                   # R3: 连板梯队+其他板块连板对抗
            'rising_first_board_advantage': '首板优势',            # R4: 首板优势+其他板块无连板
            'rising_balanced_competition': '均衡竞争',             # R5: 多板块首板均衡态势
            'rising_independent_breakout': '独立突破',              # R6: 市场弱势下的板块强势

            # Differentiation 分化状态 - 4个状态（细化竞争分化）
            'differentiation_sector_rotation': '板块轮动',         # D1: 板块间明显轮动
            'differentiation_theme_switching': '题材切换',         # D2: 题材热点切换
            'differentiation_volume_divergence': '量价分化',        # D3: 成交量与价格分化
            'differentiation_internal_split': '内部分化',          # D4: 板块内部分化

            # Retreat 回调状态 - 5个状态（增加竞争压力维度）
            'retreat_isolated_decline': '孤立下跌',                # T1: 单板块下跌+其他板块强势
            'retreat_market_wide_decline': '全面下跌',             # T2: 多板块同步下跌
            'retreat_competitive_pressure': '竞争压力',            # T3: 其他板块强势挤压
            'retreat_gentle_correction': '温和回调',                # T4: 小幅回调
            'retreat_consolidation': '整理企稳'                     # T5: 整理企稳
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
        
    def classify_sectors(self, sector_data: pd.DataFrame, market_features: pd.DataFrame) -> pd.DataFrame:
        """
        对板块数据进行状态分类

        Args:
            sector_data: 板块特征数据
            market_features: 市场特征数据

        Returns:
            pd.DataFrame: 包含状态分类结果的数据
        """
        self.logger.info(f"开始对 {len(sector_data)} 条板块数据进行高级状态分类")

        # 计算动态阈值
        self._calculate_dynamic_thresholds(sector_data)

        # 计算A股特色的特殊阈值
        self._calculate_limit_thresholds(sector_data)

        # 计算市场状态阈值
        self._calculate_market_thresholds(market_features)

        result_data = sector_data.copy()

        # 添加状态列
        result_data['main_state'] = ''
        result_data['sub_state'] = ''
        result_data['confidence'] = 0.0
        result_data['state_strength'] = 0.0

        # 批量分类（传入市场特征）
        self._batch_classify_sectors(result_data, market_features)

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

    def _calculate_market_thresholds(self, market_features: pd.DataFrame):
        """
        计算市场状态特征的动态阈值

        Args:
            market_features: 市场特征数据，包含volume和daily_change_pct等字段
        """
        self.logger.info("计算市场状态特征的动态阈值...")

        if market_features is None or market_features.empty:
            self.logger.warning("市场特征数据为空，使用默认市场阈值")
            self._set_default_market_thresholds()
            return

        # 初始化市场阈值字典
        self.market_thresholds = {}

        # 成交量阈值计算 - 划分为5个等级：极低、中低、中、中高、极高
        if 'volume' in market_features.columns:
            volume_data = market_features['volume'].dropna()
            if len(volume_data) > 0:
                volume_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
                volume_thresholds = [volume_data.quantile(q) for q in volume_quantiles]

                self.market_thresholds['volume'] = {
                    'very_low': volume_thresholds[0],      # 极低：10%分位数以下
                    'medium_low': volume_thresholds[1],    # 中低：10%-30%分位数
                    'medium': volume_thresholds[2],        # 中：30%-50%分位数
                    'medium_high': volume_thresholds[3],   # 中高：50%-70%分位数
                    'very_high': volume_thresholds[4]      # 极高：70%分位数以上
                }

                self.logger.info(f"成交量阈值: 极低<{volume_thresholds[0]:.0f}, 中低<{volume_thresholds[1]:.0f}, "
                               f"中<{volume_thresholds[2]:.0f}, 中高<{volume_thresholds[3]:.0f}, 极高>={volume_thresholds[4]:.0f}")

        # 涨跌幅阈值计算 - 划分为7个等级：大跌、中跌、小跌、平、小涨、中涨、大涨
        if 'daily_change_pct' in market_features.columns:
            change_data = market_features['daily_change_pct'].dropna()
            if len(change_data) > 0:
                # 使用对称的分位数设置，考虑A股涨跌幅特点
                change_quantiles = [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]
                change_thresholds = [change_data.quantile(q) for q in change_quantiles]

                self.market_thresholds['daily_change_pct'] = {
                    'big_fall': change_thresholds[0],      # 大跌：5%分位数以下
                    'medium_fall': change_thresholds[1],   # 中跌：5%-15%分位数
                    'small_fall': change_thresholds[2],    # 小跌：15%-35%分位数
                    'flat': change_thresholds[3],          # 平：35%-50%分位数（中位数附近）
                    'small_rise': change_thresholds[4],    # 小涨：50%-65%分位数
                    'medium_rise': change_thresholds[5],   # 中涨：65%-85%分位数
                    'big_rise': change_thresholds[6]       # 大涨：85%分位数以上
                }

                self.logger.info(f"涨跌幅阈值: 大跌<{change_thresholds[0]:.2%}, 中跌<{change_thresholds[1]:.2%}, "
                               f"小跌<{change_thresholds[2]:.2%}, 平<{change_thresholds[3]:.2%}, "
                               f"小涨<{change_thresholds[4]:.2%}, 中涨<{change_thresholds[5]:.2%}, 大涨>={change_thresholds[6]:.2%}")

        # 其他市场特征阈值（可选）
        market_score_features = ['trend_score', 'total_score', 'trend_strength']
        for feature in market_score_features:
            if feature in market_features.columns:
                feature_data = market_features[feature].dropna()
                if len(feature_data) > 0:
                    # 使用标准三分位数
                    feature_quantiles = [0.25, 0.5, 0.75]
                    feature_thresholds = [feature_data.quantile(q) for q in feature_quantiles]

                    self.market_thresholds[feature] = {
                        'low': feature_thresholds[0],
                        'medium': feature_thresholds[1],
                        'high': feature_thresholds[2]
                    }

        self.logger.info(f"市场状态阈值计算完成，共处理 {len(self.market_thresholds)} 个市场特征")

    def _set_default_market_thresholds(self):
        """设置默认的市场阈值"""
        self.market_thresholds = {
            'volume': {
                'very_low': 1000000,
                'medium_low': 5000000,
                'medium': 10000000,
                'medium_high': 20000000,
                'very_high': 50000000
            },
            'daily_change_pct': {
                'big_fall': -0.03,      # -3%
                'medium_fall': -0.015,  # -1.5%
                'small_fall': -0.005,   # -0.5%
                'flat': 0.0,            # 0%
                'small_rise': 0.005,    # 0.5%
                'medium_rise': 0.015,   # 1.5%
                'big_rise': 0.03        # 3%
            }
        }

    def _batch_classify_sectors(self, data: pd.DataFrame, market_features: pd.DataFrame = None):
        """批量分类板块状态 - 增加跨板块竞争分析"""

        # 按日期分组处理，确保每个日期使用对应的市场状态
        if 'date' in data.columns:
            for current_date, group in data.groupby('date'):
                # 获取该日期对应的市场特征
                current_market_state = self._get_current_market_state(group, market_features)

                # 添加市场状态字段到该日期的所有行
                data.loc[group.index, 'market_volume_level'] = current_market_state.get('volume_level', 'medium')
                data.loc[group.index, 'market_change_level'] = current_market_state.get('change_level', 'flat')
                data.loc[group.index, 'market_volume'] = current_market_state.get('volume', 0)
                data.loc[group.index, 'market_daily_change_pct'] = current_market_state.get('daily_change_pct', 0)

                # 分析该日期所有板块的竞争态势
                competition_analysis = self._analyze_sector_competition(group, current_market_state)

                # 对该日期的每个板块进行分类（加入竞争分析）
                for idx, row in group.iterrows():
                    try:
                        main_state, sub_state, confidence, strength = self._classify_single_sector_with_competition(
                            row, current_market_state, competition_analysis
                        )
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
        else:
            # 如果没有日期字段，使用原来的逻辑
            current_market_state = self._get_current_market_state(data, market_features)

            # 添加市场状态字段到所有行
            data['market_volume_level'] = current_market_state.get('volume_level', 'medium')
            data['market_change_level'] = current_market_state.get('change_level', 'flat')
            data['market_volume'] = current_market_state.get('volume', 0)
            data['market_daily_change_pct'] = current_market_state.get('daily_change_pct', 0)

            for idx, row in data.iterrows():
                try:
                    main_state, sub_state, confidence, strength = self._classify_single_sector(row, current_market_state)
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

    def _get_current_market_state(self, data: pd.DataFrame, market_features: pd.DataFrame) -> Dict[str, str]:
        """
        获取当前日期对应的市场状态

        Args:
            data: 板块数据
            market_features: 市场特征数据

        Returns:
            Dict[str, str]: 市场状态字典，包含volume_level和change_level
        """
        if market_features is None or market_features.empty:
            return {'volume_level': 'medium', 'change_level': 'flat'}

        # 优先使用市场数据中已经计算好的分级字段
        if 'market_volume_level' in market_features.columns and 'market_change_level' in market_features.columns:
            # 获取当前日期对应的市场数据
            if 'date' in data.columns and len(data) > 0:
                current_date = data['date'].iloc[0]

                # 在市场特征中查找对应日期的数据
                if 'date' in market_features.columns:
                    market_data = market_features[market_features['date'] == current_date]
                elif 'date_str' in market_features.columns:
                    current_date_str = current_date.strftime('%Y%m%d') if hasattr(current_date, 'strftime') else str(current_date)
                    market_data = market_features[market_features['date_str'] == current_date_str]
                else:
                    market_data = market_features.tail(1)
            else:
                market_data = market_features.tail(1)

            if not market_data.empty:
                market_row = market_data.iloc[0]
                return {
                    'volume_level': market_row.get('market_volume_level', 'medium'),
                    'change_level': market_row.get('market_change_level', 'flat'),
                    'volume': market_row.get('volume', 0),
                    'daily_change_pct': market_row.get('daily_change_pct', 0)
                }

        # 如果没有预计算的分级字段，则动态计算
        # 获取当前日期（假设板块数据中有date字段）
        if 'date' in data.columns and len(data) > 0:
            current_date = data['date'].iloc[0]

            # 在市场特征中查找对应日期的数据
            if 'date' in market_features.columns:
                market_data = market_features[market_features['date'] == current_date]
            elif 'date_str' in market_features.columns:
                # 如果是字符串格式的日期，需要转换
                current_date_str = current_date.strftime('%Y%m%d') if hasattr(current_date, 'strftime') else str(current_date)
                market_data = market_features[market_features['date_str'] == current_date_str]
            else:
                # 如果没有日期字段，使用最新的市场数据
                market_data = market_features.tail(1)
        else:
            # 如果没有日期信息，使用最新的市场数据
            market_data = market_features.tail(1)

        if market_data.empty:
            return {'volume_level': 'medium', 'change_level': 'flat'}

        # 获取市场数据
        market_row = market_data.iloc[0]
        volume = market_row.get('volume', 0)
        daily_change_pct = market_row.get('daily_change_pct', 0)

        # 根据阈值判断成交量等级
        volume_level = self._classify_volume_level(volume)

        # 根据阈值判断涨跌幅等级
        change_level = self._classify_change_level(daily_change_pct)

        return {
            'volume_level': volume_level,
            'change_level': change_level,
            'volume': volume,
            'daily_change_pct': daily_change_pct
        }

    def _classify_volume_level(self, volume: float) -> str:
        """
        根据成交量阈值分类成交量等级

        Args:
            volume: 成交量

        Returns:
            str: 成交量等级 (very_low, medium_low, medium, medium_high, very_high)
        """
        if not hasattr(self, 'market_thresholds') or 'volume' not in self.market_thresholds:
            return 'medium'

        volume_thresholds = self.market_thresholds['volume']

        if volume <= volume_thresholds['very_low']:
            return 'very_low'
        elif volume <= volume_thresholds['medium_low']:
            return 'medium_low'
        elif volume <= volume_thresholds['medium']:
            return 'medium'
        elif volume <= volume_thresholds['medium_high']:
            return 'medium_high'
        else:
            return 'very_high'

    def _classify_change_level(self, daily_change_pct: float) -> str:
        """
        根据涨跌幅阈值分类涨跌幅等级

        Args:
            daily_change_pct: 日涨跌幅

        Returns:
            str: 涨跌幅等级 (big_fall, medium_fall, small_fall, flat, small_rise, medium_rise, big_rise)
        """
        if not hasattr(self, 'market_thresholds') or 'daily_change_pct' not in self.market_thresholds:
            return 'flat'

        change_thresholds = self.market_thresholds['daily_change_pct']

        if daily_change_pct <= change_thresholds['big_fall']:
            return 'big_fall'
        elif daily_change_pct <= change_thresholds['medium_fall']:
            return 'medium_fall'
        elif daily_change_pct <= change_thresholds['small_fall']:
            return 'small_fall'
        elif daily_change_pct <= change_thresholds['flat']:
            return 'flat'
        elif daily_change_pct <= change_thresholds['small_rise']:
            return 'small_rise'
        elif daily_change_pct <= change_thresholds['medium_rise']:
            return 'medium_rise'
        else:
            return 'big_rise'

    def _classify_single_sector(self, row: pd.Series, market_state: Dict[str, str] = None) -> Tuple[str, str, float, float]:
        """
        对单个板块进行状态分类
        
        Args:
            row: 板块数据行
            
        Returns:
            tuple: (主状态, 子状态, 置信度, 状态强度)
        """
        # 提取关键特征
        features = self._extract_sector_features(row)

        # 添加市场状态特征到features中
        if market_state:
            features['market_volume_level'] = market_state.get('volume_level', 'medium')
            features['market_change_level'] = market_state.get('change_level', 'flat')
            features['market_volume'] = market_state.get('volume', 0)
            features['market_daily_change_pct'] = market_state.get('daily_change_pct', 0)

        # 基于A股特色的分类逻辑（传入市场状态）
        main_state, sub_state, confidence, strength = self._classify_with_real_features(features, market_state)

        return main_state, sub_state, confidence, strength

    def _analyze_sector_competition(self, daily_sectors: pd.DataFrame, market_state: Dict[str, str]) -> Dict[str, Any]:
        """
        分析同日所有板块的竞争态势

        Args:
            daily_sectors: 同一日期的所有板块数据
            market_state: 市场状态

        Returns:
            竞争态势分析结果
        """
        try:
            competition_analysis = {
                'total_sectors': len(daily_sectors),
                'cascade_sectors': [],      # 连板梯队板块
                'first_board_sectors': [],  # 首板爆发板块
                'strong_sectors': [],       # 强势板块
                'weak_sectors': [],         # 弱势板块
                'market_competition_level': 'low'  # 市场竞争激烈程度
            }

            # 计算各板块的关键指标
            for idx, row in daily_sectors.iterrows():
                sector_code = row.get('sector_code', f'sector_{idx}')

                # 连板梯队识别
                board_2_plus = row.get('board_counts_2板', 0) + row.get('board_counts_3板', 0) + \
                              row.get('board_counts_4板', 0) + row.get('board_counts_5板', 0)
                board_1 = row.get('board_counts_1板', 0)
                limit_up_count = row.get('limit_up_count', 0)
                change_percent = row.get('change_percent', 0)

                # 连板梯队判断：有2板以上（降低门槛）
                if board_2_plus > 0:
                    competition_analysis['cascade_sectors'].append({
                        'sector_code': sector_code,
                        'cascade_strength': board_2_plus * 2 + board_1,
                        'board_2_plus': board_2_plus,
                        'board_1': board_1
                    })

                # 首板爆发判断：首板较多但无连板（降低门槛）
                elif board_1 >= 2 and board_2_plus == 0:
                    competition_analysis['first_board_sectors'].append({
                        'sector_code': sector_code,
                        'first_board_count': board_1,
                        'limit_up_count': limit_up_count
                    })

                # 强势板块判断：涨停多或涨幅大（降低门槛）
                if limit_up_count >= 2 or change_percent >= 0.015:
                    competition_analysis['strong_sectors'].append({
                        'sector_code': sector_code,
                        'strength_score': limit_up_count * 2 + change_percent * 100
                    })

                # 弱势板块判断：跌幅大或跌停多（降低门槛）
                elif change_percent <= -0.015 or row.get('limit_down_count', 0) >= 1:
                    competition_analysis['weak_sectors'].append({
                        'sector_code': sector_code,
                        'weakness_score': abs(change_percent) * 100 + row.get('limit_down_count', 0) * 2
                    })

            # 计算市场竞争激烈程度
            cascade_count = len(competition_analysis['cascade_sectors'])
            first_board_count = len(competition_analysis['first_board_sectors'])
            strong_count = len(competition_analysis['strong_sectors'])

            if cascade_count >= 3:
                competition_analysis['market_competition_level'] = 'very_high'
            elif cascade_count >= 2 or (cascade_count >= 1 and first_board_count >= 2):
                competition_analysis['market_competition_level'] = 'high'
            elif cascade_count >= 1 or first_board_count >= 3 or strong_count >= 4:
                competition_analysis['market_competition_level'] = 'medium'
            else:
                competition_analysis['market_competition_level'] = 'low'

            self.logger.debug(f"竞争分析: 连板{cascade_count}个, 首板{first_board_count}个, "
                            f"强势{strong_count}个, 竞争等级{competition_analysis['market_competition_level']}")

            return competition_analysis

        except Exception as e:
            self.logger.warning(f"竞争分析失败: {e}")
            return {
                'total_sectors': len(daily_sectors),
                'cascade_sectors': [],
                'first_board_sectors': [],
                'strong_sectors': [],
                'weak_sectors': [],
                'market_competition_level': 'low'
            }

    def _classify_single_sector_with_competition(self, row: pd.Series, market_state: Dict[str, str],
                                               competition_analysis: Dict[str, Any]) -> Tuple[str, str, float, float]:
        """
        基于竞争分析的单板块分类

        Args:
            row: 板块数据行
            market_state: 市场状态
            competition_analysis: 竞争态势分析

        Returns:
            (主状态, 子状态, 置信度, 强度得分)
        """
        try:
            # 提取板块特征
            features = self._extract_sector_features(row)
            sector_code = row.get('sector_code', 'unknown')

            # 判断当前板块在竞争中的位置
            is_cascade_sector = any(s['sector_code'] == sector_code for s in competition_analysis['cascade_sectors'])
            is_first_board_sector = any(s['sector_code'] == sector_code for s in competition_analysis['first_board_sectors'])
            is_strong_sector = any(s['sector_code'] == sector_code for s in competition_analysis['strong_sectors'])
            is_weak_sector = any(s['sector_code'] == sector_code for s in competition_analysis['weak_sectors'])

            # 基于竞争态势进行分类
            return self._classify_with_competition_features(
                features, market_state, competition_analysis,
                is_cascade_sector, is_first_board_sector, is_strong_sector, is_weak_sector
            )

        except Exception as e:
            self.logger.warning(f"竞争分类失败: {e}")
            # 回退到原始分类方法
            features = self._extract_sector_features(row)
            return self._classify_with_real_features(features, market_state)

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

    def _classify_with_competition_features(self, features: Dict[str, float], market_state: Dict[str, str],
                                          competition_analysis: Dict[str, Any], is_cascade_sector: bool,
                                          is_first_board_sector: bool, is_strong_sector: bool,
                                          is_weak_sector: bool) -> Tuple[str, str, float, float]:
        """
        基于竞争态势的板块分类逻辑

        Args:
            features: 板块特征
            market_state: 市场状态
            competition_analysis: 竞争分析结果
            is_cascade_sector: 是否为连板梯队板块
            is_first_board_sector: 是否为首板爆发板块
            is_strong_sector: 是否为强势板块
            is_weak_sector: 是否为弱势板块

        Returns:
            (主状态, 子状态, 置信度, 强度得分)
        """
        try:
            # 提取关键指标
            change_percent = features.get('change_percent', 0)
            limit_up_count = features.get('limit_up_count', 0)
            limit_down_count = features.get('limit_down_count', 0)
            board_2_plus = features.get('board_counts_2板', 0) + features.get('board_counts_3板', 0)
            board_1 = features.get('board_counts_1板', 0)

            # 获取竞争环境信息
            cascade_count = len(competition_analysis['cascade_sectors'])
            first_board_count = len(competition_analysis['first_board_sectors'])
            competition_level = competition_analysis['market_competition_level']

            # 市场环境调整因子
            market_adjustment = self._get_market_adjustment_factor(market_state)

            # === 上涨状态分类（基于竞争态势） ===
            if change_percent > 0 or limit_up_count > 0:

                # R1: 绝对龙头 - 连板梯队且无其他强竞争
                if is_cascade_sector and cascade_count == 1 and first_board_count <= 1:
                    score = board_2_plus * 3 + board_1 * 1.5 + change_percent * 50
                    confidence = min(0.95 * market_adjustment, 1.0)
                    return 'rising', 'rising_dominant_leader', confidence, score * market_adjustment

                # R2: 竞争龙头 - 连板梯队但有其他首板竞争
                elif is_cascade_sector and first_board_count >= 1:
                    score = board_2_plus * 2.5 + board_1 * 1.2 + change_percent * 40
                    confidence = min(0.90 * market_adjustment, 1.0)
                    return 'rising', 'rising_competitive_leader', confidence, score * market_adjustment

                # R3: 连板对抗 - 连板梯队且有其他连板梯队竞争
                elif is_cascade_sector and cascade_count >= 2:
                    score = board_2_plus * 2 + board_1 + change_percent * 30
                    confidence = min(0.85 * market_adjustment, 1.0)
                    return 'rising', 'rising_cascade_battle', confidence, score * market_adjustment

                # R4: 首板优势 - 首板爆发且其他板块无连板
                elif is_first_board_sector and cascade_count == 0:
                    score = board_1 * 2 + limit_up_count * 1.5 + change_percent * 25
                    confidence = min(0.80 * market_adjustment, 1.0)
                    return 'rising', 'rising_first_board_advantage', confidence, score * market_adjustment

                # R5: 均衡竞争 - 多板块首板均衡，无明显连板优势
                elif is_first_board_sector and first_board_count >= 2 and cascade_count <= 1:
                    score = board_1 * 1.5 + limit_up_count + change_percent * 20
                    confidence = min(0.75 * market_adjustment, 1.0)
                    return 'rising', 'rising_balanced_competition', confidence, score * market_adjustment

                # R6: 独立突破 - 强势板块或市场弱势下的板块强势
                elif is_strong_sector or (change_percent > 0.005 and competition_level in ['low', 'medium']):
                    score = limit_up_count * 2 + change_percent * 30
                    confidence = min(0.85 * market_adjustment, 1.0)
                    return 'rising', 'rising_independent_breakout', confidence, score * market_adjustment

            # === 分化状态分类 ===
            elif abs(change_percent) <= 0.02 and (limit_up_count > 0 or limit_down_count > 0):

                # D1: 板块轮动 - 有强势板块和弱势板块并存
                if len(competition_analysis['strong_sectors']) >= 1 and len(competition_analysis['weak_sectors']) >= 1:
                    score = abs(change_percent) * 10 + (limit_up_count + limit_down_count) * 0.5
                    confidence = min(0.70 * market_adjustment, 1.0)
                    return 'differentiation', 'differentiation_sector_rotation', confidence, score * market_adjustment

                # D2: 题材切换 - 竞争激烈但无明显优势
                elif competition_level in ['high', 'medium'] and not (is_cascade_sector or is_first_board_sector):
                    score = (limit_up_count + limit_down_count) * 0.8 + abs(change_percent) * 15
                    confidence = min(0.65 * market_adjustment, 1.0)
                    return 'differentiation', 'differentiation_theme_switching', confidence, score * market_adjustment

                # D3: 量价分化 - 成交量与价格表现不匹配
                elif limit_up_count > 0 and change_percent < 0.008:
                    score = limit_up_count * 1.5 + abs(change_percent) * 20
                    confidence = min(0.60 * market_adjustment, 1.0)
                    return 'differentiation', 'differentiation_volume_divergence', confidence, score * market_adjustment

                # D4: 内部分化 - 板块内部分化明显
                else:
                    score = (limit_up_count + limit_down_count) + abs(change_percent) * 10
                    confidence = min(0.55 * market_adjustment, 1.0)
                    return 'differentiation', 'differentiation_internal_split', confidence, score * market_adjustment

            # === 回调状态分类 ===
            else:

                # T1: 孤立下跌 - 单板块下跌，其他板块强势
                if is_weak_sector and len(competition_analysis['strong_sectors']) >= 1:
                    score = abs(change_percent) * 20 + limit_down_count * 2
                    confidence = min(0.80 * market_adjustment, 1.0)
                    return 'retreat', 'retreat_isolated_decline', confidence, score * market_adjustment

                # T2: 全面下跌 - 多板块同步下跌
                elif len(competition_analysis['weak_sectors']) >= 2:
                    score = abs(change_percent) * 15 + limit_down_count * 1.5
                    confidence = min(0.85 * market_adjustment, 1.0)
                    return 'retreat', 'retreat_market_wide_decline', confidence, score * market_adjustment

                # T3: 竞争压力 - 其他板块强势挤压
                elif not is_weak_sector and len(competition_analysis['strong_sectors']) >= 2:
                    score = abs(change_percent) * 10 + limit_down_count
                    confidence = min(0.70 * market_adjustment, 1.0)
                    return 'retreat', 'retreat_competitive_pressure', confidence, score * market_adjustment

                # T4: 温和回调 - 小幅回调
                elif change_percent > -0.025:
                    score = abs(change_percent) * 12 + limit_down_count * 0.5
                    confidence = min(0.60 * market_adjustment, 1.0)
                    return 'retreat', 'retreat_gentle_correction', confidence, score * market_adjustment

                # T5: 整理企稳 - 基本平盘
                else:
                    score = abs(change_percent) * 8 + (limit_up_count + limit_down_count) * 0.3
                    confidence = min(0.50 * market_adjustment, 1.0)
                    return 'retreat', 'retreat_consolidation', confidence, score * market_adjustment

            # 如果没有匹配到任何条件，使用默认分类
            self.logger.debug("未匹配到竞争分类条件，使用默认分类")
            return self._classify_with_real_features(features, market_state)

        except Exception as e:
            self.logger.warning(f"竞争分类失败: {e}")
            # 回退到原始分类
            return self._classify_with_real_features(features, market_state)

    def _get_market_adjustment_factor(self, market_state: Dict[str, str]) -> float:
        """
        根据市场状态计算调整因子

        Args:
            market_state: 市场状态

        Returns:
            调整因子 (0.7-1.3)
        """
        try:
            volume_level = market_state.get('volume_level', 'medium')
            change_level = market_state.get('change_level', 'flat')

            # 成交量调整
            volume_factor = {
                'very_low': 0.8,
                'medium_low': 0.9,
                'medium': 1.0,
                'medium_high': 1.1,
                'very_high': 1.2
            }.get(volume_level, 1.0)

            # 涨跌幅调整
            change_factor = {
                'big_fall': 0.7,
                'medium_fall': 0.8,
                'small_fall': 0.9,
                'flat': 1.0,
                'small_rise': 1.1,
                'medium_rise': 1.2,
                'big_rise': 1.3
            }.get(change_level, 1.0)

            # 综合调整因子
            adjustment = (volume_factor + change_factor) / 2
            return max(0.7, min(1.3, adjustment))

        except Exception as e:
            self.logger.warning(f"市场调整因子计算失败: {e}")
            return 1.0

    def _classify_with_real_features(self, features: Dict[str, float], market_state: Dict[str, str] = None) -> Tuple[str, str, float, float]:
        """
        基于2020-2025年顶尖量化金融研究和A股游资操作策略的板块状态分类

        设计理念：
        1. 市场环境驱动：将市场整体状况作为分类的首要条件
        2. 板块特征细分：在市场环境基础上，通过板块特征进行精细化分类
        3. 游资偏好导向：重点识别游资偏好的市场环境和板块特征组合
        4. 收益预测优化：确保状态间有明显的future_return_1d差异特征
        """

        # === 1. 基础数据提取 ===
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
        yesterday_broken_limit_premium = features.get('yesterday_broken_limit_premium', 0)

        # 连板阶梯特征
        board_1 = features.get('board_counts_1板', 0)
        board_2 = features.get('board_counts_2板', 0)
        board_3 = features.get('board_counts_3板', 0)
        board_4_plus = sum([features.get(f'board_counts_{i}板', 0) for i in range(4, 11)]) + features.get('board_counts_10板及以上', 0)
        board_3_plus = board_3 + board_4_plus

        # 计算连板梯队数量和强度
        board_tier_count = sum([1 for i in range(1, 11) if features.get(f'board_counts_{i}板', 0) > 0])
        if features.get('board_counts_10板及以上', 0) > 0:
            board_tier_count += 1

        # 高位连板强度（3板以上的总数）
        high_board_strength = board_3_plus

        # 计算关键比例（防止除零错误）
        up_ratio = up_count / max(stock_count, 1)
        down_ratio = down_count / max(stock_count, 1)
        limit_up_ratio = limit_up_count / max(stock_count, 1)

        # === 2. 历史阈值获取 ===
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

        # === 3. 市场状态分析 ===
        # 获取市场状态特征
        market_volume_level = features.get('market_volume_level', 'medium')
        market_change_level = features.get('market_change_level', 'flat')

        # 市场状态调整因子
        market_adjustment = self._calculate_market_adjustment(market_volume_level, market_change_level)

        # 市场强弱判断
        is_market_strong = market_change_level in ['medium_rise', 'big_rise']
        is_market_weak = market_change_level in ['medium_fall', 'big_fall']
        is_volume_active = market_volume_level in ['medium_high', 'very_high']
        is_volume_low = market_volume_level in ['very_low', 'medium_low']

        self.logger.debug(f"市场状态: 成交量={market_volume_level}, 涨跌幅={market_change_level}, 调整因子={market_adjustment}")
        self.logger.debug(f"市场判断: 强势={is_market_strong}, 弱势={is_market_weak}, 活跃={is_volume_active}, 低迷={is_volume_low}")

        # === 4. 基于市场环境+板块特征的简化分类逻辑 ===
        # 重新设计：简化条件，提高覆盖率，确保12个子状态都能被合理触发

        # ========== RISING 状态：上涨趋势 (5个状态) ==========

        # R2: 连板接力 - 有连板梯队形成（优先级提前，避免被强势爆发覆盖）
        if (board_tier_count >= 2 and (board_2 > 0 or board_3_plus > 0) and
            limit_up_count <= limit_up_median * 1.5):  # 限制涨停数量，避免与强势爆发重叠
            score = board_tier_count + (board_2 * 1.5) + (board_3_plus * 2)
            confidence = min(0.85 * market_adjustment, 1.0)
            return 'rising', 'rising_cascade_formation', confidence, score * market_adjustment

        # R1: 强势爆发 - 涨停数量多或市场强势下的板块表现
        elif (limit_up_count > limit_up_median or
              (is_market_strong and change_percent > 0.02)):
            score = (limit_up_count / max(limit_up_median, 1)) + (change_percent * 20)
            confidence = min(0.90 * market_adjustment, 1.0)
            return 'rising', 'rising_strong_surge', confidence, score * market_adjustment

        # R3: 独立突破 - 市场中性或弱势下的板块强势（降低门槛）
        elif ((not is_market_strong) and change_percent > 0.005 and
              up_ratio > up_ratio_median * 0.8):
            score = (change_percent * 15) + (up_ratio / up_ratio_median)
            confidence = min(0.80 * market_adjustment, 1.0)
            return 'rising', 'rising_independent_breakout', confidence, score * market_adjustment

        # R4: 温和上涨 - 小幅上涨但上涨面较好
        elif (change_percent > 0 and change_percent <= 0.01 and
              up_ratio > up_ratio_median):
            score = (change_percent * 10) + (up_ratio / up_ratio_median)
            confidence = min(0.75 * market_adjustment, 1.0)
            return 'rising', 'rising_gentle_accumulation', confidence, score * market_adjustment

        # R5: 首板爆发 - 首板涨停多，高位板少
        elif (board_1 > limit_up_low and board_3_plus <= 1):
            score = (board_1 / max(limit_up_low, 1)) + (2 - board_3_plus)
            confidence = min(0.80 * market_adjustment, 1.0)
            return 'rising', 'rising_first_board_surge', confidence, score * market_adjustment

        # ========== DIFFERENTIATION 状态：分化趋势 (3个状态) ==========

        # D1: 高低分化 - 既有涨停又有跌停，或连板分化明显
        elif ((limit_up_count > 0 and limit_down_count > 0) or
              (board_2 + board_3_plus > board_1 and board_1 > 0)):
            score = (limit_up_count + limit_down_count) / max(stock_count, 1) * 10
            if board_2 + board_3_plus > board_1:
                score += (board_2 + board_3_plus) / max(board_1, 1)
            confidence = min(0.75 * market_adjustment, 1.0)
            return 'differentiation', 'differentiation_high_low_split', confidence, score * market_adjustment

        # D2: 量价分化 - 成交量活跃但价格表现平淡
        elif (is_volume_active and abs(change_percent) < 0.01):
            score = 1 + (1 / max(abs(change_percent) + 0.001, 0.001))
            confidence = min(0.70 * market_adjustment, 1.0)
            return 'differentiation', 'differentiation_volume_divergence', confidence, score * market_adjustment

        # D3: 题材轮动 - 有一定涨停但整体表现平淡
        elif (limit_up_count > 0 and limit_up_count <= limit_up_median and
              abs(change_percent) < 0.015):
            score = (limit_up_count / max(limit_up_median, 1)) + (1 / max(abs(change_percent) + 0.001, 0.001))
            confidence = min(0.70 * market_adjustment, 1.0)
            return 'differentiation', 'differentiation_theme_rotation', confidence, score * market_adjustment

        # ========== RETREAT 状态：回调趋势 (4个状态) ==========

        # T1: 重度下跌 - 大幅下跌或市场弱势下的下跌
        elif (change_percent < change_low or
              (is_market_weak and change_percent < -0.01)):
            score = abs(change_percent / max(abs(change_low), 0.01)) + (limit_down_count / max(limit_down_median, 1))
            confidence = min(0.90 * market_adjustment, 1.0)
            return 'retreat', 'retreat_heavy_decline', confidence, score * market_adjustment

        # T2: 跌停压力 - 跌停数量较多
        elif (limit_down_count > limit_down_low):
            score = (limit_down_count / max(limit_down_low, 1)) + abs(change_percent) * 10
            confidence = min(0.85 * market_adjustment, 1.0)
            return 'retreat', 'retreat_limit_down_pressure', confidence, score * market_adjustment

        # T3: 温和回调 - 小幅下跌（扩大范围）
        elif (change_percent < -0.001 and change_percent > change_low and
              limit_down_count <= limit_down_low):
            score = abs(change_percent) * 10 + (down_ratio / max(up_ratio, 0.1))
            confidence = min(0.70 * market_adjustment, 1.0)
            return 'retreat', 'retreat_gentle_correction', confidence, score * market_adjustment

        # T4: 整理企稳 - 基本平盘或小幅波动（扩大范围）
        elif (abs(change_percent) <= 0.008 and limit_up_count == 0 and limit_down_count == 0):
            score = 1 + (1 / max(abs(change_percent) + 0.001, 0.001))
            confidence = min(0.60 * market_adjustment, 1.0)
            return 'retreat', 'retreat_consolidation', confidence, score * market_adjustment

        # ========== 默认分类逻辑 ==========

        # 基于基本指标的兜底分类
        else:
            if change_percent > 0:
                score = change_percent * 10
                confidence = min(0.50 * market_adjustment, 1.0)
                return 'rising', 'rising_gentle_accumulation', confidence, score * market_adjustment
            else:
                score = abs(change_percent) * 10
                confidence = min(0.50 * market_adjustment, 1.0)
                return 'retreat', 'retreat_gentle_correction', confidence, score * market_adjustment

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

    def _calculate_market_adjustment(self, volume_level: str, change_level: str) -> float:
        """
        根据市场状态计算调整因子

        Args:
            volume_level: 成交量等级 (very_low, medium_low, medium, medium_high, very_high)
            change_level: 涨跌幅等级 (big_fall, medium_fall, small_fall, flat, small_rise, medium_rise, big_rise)

        Returns:
            float: 调整因子 (0.5-1.5)，用于调整板块分类的置信度和强度
        """
        # 成交量调整因子
        volume_factors = {
            'very_low': 0.7,     # 极低成交量，降低置信度
            'medium_low': 0.85,  # 中低成交量，略微降低置信度
            'medium': 1.0,       # 中等成交量，正常
            'medium_high': 1.15, # 中高成交量，提高置信度
            'very_high': 1.3     # 极高成交量，显著提高置信度
        }

        # 涨跌幅调整因子
        change_factors = {
            'big_fall': 0.6,     # 大跌，板块分化明显，降低置信度
            'medium_fall': 0.8,  # 中跌，市场偏弱
            'small_fall': 0.9,   # 小跌，市场略弱
            'flat': 1.0,         # 平盘，正常
            'small_rise': 1.1,   # 小涨，市场略强
            'medium_rise': 1.25, # 中涨，市场较强，提高置信度
            'big_rise': 1.4      # 大涨，市场强势，显著提高置信度
        }

        volume_factor = volume_factors.get(volume_level, 1.0)
        change_factor = change_factors.get(change_level, 1.0)

        # 综合调整因子：成交量和涨跌幅的加权平均
        # 涨跌幅权重更高，因为它更直接反映市场情绪
        adjustment = (volume_factor * 0.3) + (change_factor * 0.7)

        # 限制调整因子范围在0.5-1.5之间
        adjustment = max(0.5, min(1.5, adjustment))

        return adjustment
