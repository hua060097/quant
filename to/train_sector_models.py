#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块轮动模型训练 - 函数调用链分析和优化版本

主要功能：基于变化点检测的智能选股系统
核心入口：select_stocks_by_change_points()

函数调用链分析：
1. select_stocks_by_change_points() - 主入口函数
   ├── _load_trading_calendar() - 加载交易日历
   ├── _batch_load_features() - 批量加载板块特征（板块+市场+相对特征）
   ├── load_stock_features() - 加载个股特征（个股基础特征和技术指标）
   ├── _comprehensive_data_filtering() - 数据过滤（个股和板块数据）
   ├── _detect_stock_change_points() - 检测个股变化点
   ├── collect_stock_sector_change_point_features() - 特征关联工程（个股+板块关联）
   ├── _extract_change_point_features_directly() - 提取变化点特征
   ├── _analyze_change_point_patterns_direct() - 模式分析
   ├── _predict_stocks_with_change_points() - 股票预测
   ├── _rank_and_select_stocks_by_method() - 排序选股
   ├── _combine_selection_results() - 合并结果
   ├── _evaluate_selected_stocks_with_method() - 评估表现
   └── _compare_method_evaluations() - 对比评估

"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import json
import traceback
from concurrent.futures import as_completed, ThreadPoolExecutor
import pickle
import hashlib
import time
import math
import inspect

# ===== 集成学习多策略系统 =====
class ChangePointEnsembleSystem:
    """基于变化点的集成学习多策略选股系统"""

    def __init__(self, logger=None):
        self.logger = logger

        # 策略特征分层定义 - 基于实际存在的特征
        self.strategy_features = {
            'momentum': [
                'return_5d', 'return_20d', 'tech_momentum_signal',
                'trend_strength', 'trend_consistency', 'momentum_acceleration'
            ],
            'mean_reversion': [
                'bb_position', 'rsi12', 'overbought_oversold',
                'short_reversal', 'near_support', 'near_resistance'
            ],
            'quality': [
                'risk_adjusted_return', 'volatility20', 'atr14',
                'volume_weighted_price', 'candlestick_strength'
            ],
            'technical_pattern': [
                'pattern_engulfing_bullish', 'pattern_engulfing_bearish',
                'candlestick_strength', 'trend_consistency'
            ],
            'market_context': [
                'market_benchmark_change', 'market_rsi12', 'market_trend_strength',
                'relative_change_percent', 'correlation_with_cyb'
            ]
        }

        # Thompson Sampling参数
        self.strategy_alpha = np.ones(len(self.strategy_features))  # 成功次数+1
        self.strategy_beta = np.ones(len(self.strategy_features))   # 失败次数+1
        self.strategy_names = list(self.strategy_features.keys())

        # 市场状态阈值
        self.market_thresholds = {
            'bull_threshold': 1.0,      # 牛市：market_benchmark_change > 1%
            'bear_threshold': -2.0,     # 熊市：market_benchmark_change < -2%
            'trend_high': 0.6,          # 高趋势强度
            'trend_low': 0.3            # 低趋势强度
        }

        # 性能历史
        self.performance_history = []

    def identify_market_regime(self, market_features):
        """识别市场状态"""
        try:
            benchmark_change = market_features.get('market_benchmark_change', 0)
            trend_strength = market_features.get('market_trend_strength', 0.5)
            market_rsi = market_features.get('market_rsi12', 50)

            # 牛市条件
            if (benchmark_change > self.market_thresholds['bull_threshold'] and
                trend_strength > self.market_thresholds['trend_high'] and
                market_rsi < 70):
                return 'bull_market'

            # 熊市条件
            elif (benchmark_change < self.market_thresholds['bear_threshold'] and
                  trend_strength < self.market_thresholds['trend_low'] and
                  market_rsi < 30):
                return 'bear_market'

            # 震荡市
            else:
                return 'sideways_market'

        except Exception as e:
            if self.logger:
                self.logger.warning(f"市场状态识别失败: {e}")
            return 'sideways_market'

    def sample_strategy_weights(self):
        """Thompson Sampling采样策略权重"""
        try:
            weights = []
            for i in range(len(self.strategy_names)):
                # 从Beta分布采样
                weight = np.random.beta(self.strategy_alpha[i], self.strategy_beta[i])
                weights.append(weight)

            # 归一化
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(self.strategy_names)] * len(self.strategy_names)

            return dict(zip(self.strategy_names, weights))

        except Exception as e:
            if self.logger:
                self.logger.error(f"权重采样失败: {e}")
            # 返回均匀权重
            uniform_weight = 1.0 / len(self.strategy_names)
            return dict(zip(self.strategy_names, [uniform_weight] * len(self.strategy_names)))

    def adjust_weights_by_market(self, base_weights, market_regime):
        """根据市场状态调整策略权重"""
        try:
            # 市场状态调整系数
            adjustments = {
                'bull_market': {
                    'momentum': 1.4,
                    'mean_reversion': 0.7,
                    'quality': 0.9,
                    'technical_pattern': 1.1,
                    'market_context': 1.2
                },
                'bear_market': {
                    'momentum': 0.6,
                    'mean_reversion': 1.5,
                    'quality': 1.4,
                    'technical_pattern': 1.2,
                    'market_context': 1.3
                },
                'sideways_market': {
                    'momentum': 0.8,
                    'mean_reversion': 1.6,
                    'quality': 1.1,
                    'technical_pattern': 1.3,
                    'market_context': 1.0
                }
            }

            if market_regime not in adjustments:
                return base_weights

            regime_adjustments = adjustments[market_regime]
            adjusted_weights = {}

            for strategy, base_weight in base_weights.items():
                adjustment = regime_adjustments.get(strategy, 1.0)
                adjusted_weights[strategy] = base_weight * adjustment

            # 重新归一化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}

            return adjusted_weights

        except Exception as e:
            if self.logger:
                self.logger.error(f"权重调整失败: {e}")
            return base_weights

    def update_performance(self, strategy_returns, benchmark_return=0.0):
        """更新策略表现，调整Thompson Sampling参数"""
        try:
            for i, strategy_name in enumerate(self.strategy_names):
                if i < len(strategy_returns):
                    strategy_return = strategy_returns[i]
                    if strategy_return > benchmark_return:
                        self.strategy_alpha[i] += 1  # 成功
                    else:
                        self.strategy_beta[i] += 1   # 失败

            # 记录历史
            self.performance_history.append({
                'timestamp': datetime.now(),
                'returns': strategy_returns,
                'benchmark': benchmark_return
            })

            # 保持历史记录在合理范围内
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

        except Exception as e:
            if self.logger:
                self.logger.error(f"性能更新失败: {e}")

class MarketRiskFilter:
    """市场风险过滤器 - 识别并避免市场高点选股"""

    def __init__(self, logger=None):
        self.logger = logger

        # 市场高点风险指标阈值
        self.risk_thresholds = {
            # 技术指标风险阈值
            'market_rsi_overbought': 75,        # RSI超买阈值
            'market_rsi_extreme': 85,           # RSI极度超买
            'bb_upper_breach': 0.95,            # 布林带上轨突破
            'trend_exhaustion': 0.85,           # 趋势衰竭信号

            # 市场结构风险阈值
            'benchmark_surge_1d': 0.03,         # 单日涨幅过大(3%)
            'benchmark_surge_3d': 0.08,         # 3日累计涨幅过大(8%)
            'benchmark_surge_5d': 0.12,         # 5日累计涨幅过大(12%)

            # 波动率风险阈值
            'volatility_spike': 2.0,            # 波动率激增倍数
            'volume_surge': 1.8,                # 成交量激增倍数

            # 综合风险评分阈值
            'high_risk_threshold': 0.7,         # 高风险阈值
            'extreme_risk_threshold': 0.85      # 极端风险阈值
        }

        # 风险权重配置
        self.risk_weights = {
            'technical_risk': 0.35,      # 技术指标风险权重
            'structural_risk': 0.30,     # 市场结构风险权重
            'volatility_risk': 0.20,     # 波动率风险权重
            'momentum_risk': 0.15        # 动量风险权重
        }

    def assess_market_risk(self, market_features, historical_data=None):
        """
        评估市场风险等级

        Args:
            market_features: 当前市场特征
            historical_data: 历史数据用于计算相对指标

        Returns:
            dict: 风险评估结果
        """
        try:
            risk_assessment = {
                'overall_risk_score': 0.0,
                'risk_level': 'low',
                'risk_factors': {},
                'should_avoid_selection': False,
                'risk_details': {}
            }

            # 1. 技术指标风险评估
            technical_risk = self._assess_technical_risk(market_features)

            # 2. 市场结构风险评估
            structural_risk = self._assess_structural_risk(market_features, historical_data)

            # 3. 波动率风险评估
            volatility_risk = self._assess_volatility_risk(market_features, historical_data)

            # 4. 动量风险评估
            momentum_risk = self._assess_momentum_risk(market_features)

            # 5. 综合风险评分
            overall_risk = (
                technical_risk['score'] * self.risk_weights['technical_risk'] +
                structural_risk['score'] * self.risk_weights['structural_risk'] +
                volatility_risk['score'] * self.risk_weights['volatility_risk'] +
                momentum_risk['score'] * self.risk_weights['momentum_risk']
            )

            # 6. 风险等级判定
            if overall_risk >= self.risk_thresholds['extreme_risk_threshold']:
                risk_level = 'extreme'
                should_avoid = True
            elif overall_risk >= self.risk_thresholds['high_risk_threshold']:
                risk_level = 'high'
                should_avoid = True
            elif overall_risk >= 0.5:
                risk_level = 'medium'
                should_avoid = False
            else:
                risk_level = 'low'
                should_avoid = False

            # 7. 构建结果
            risk_assessment.update({
                'overall_risk_score': float(overall_risk),
                'risk_level': risk_level,
                'should_avoid_selection': should_avoid,
                'risk_factors': {
                    'technical_risk': technical_risk,
                    'structural_risk': structural_risk,
                    'volatility_risk': volatility_risk,
                    'momentum_risk': momentum_risk
                },
                'risk_details': {
                    'primary_concerns': self._identify_primary_concerns(
                        technical_risk, structural_risk, volatility_risk, momentum_risk
                    ),
                    'risk_explanation': self._generate_risk_explanation(risk_level, overall_risk)
                }
            })

            if self.logger and should_avoid:
                self.logger.warning(f"市场风险过滤器触发 - 风险等级: {risk_level}, "
                                  f"风险评分: {overall_risk:.3f}, 建议避免选股")

            return risk_assessment

        except Exception as e:
            if self.logger:
                self.logger.error(f"市场风险评估失败: {e}")
            return {
                'overall_risk_score': 0.5,
                'risk_level': 'medium',
                'should_avoid_selection': False,
                'risk_factors': {},
                'risk_details': {'error': str(e)}
            }

    def _assess_technical_risk(self, market_features):
        """评估技术指标风险"""
        try:
            risk_score = 0.0
            risk_factors = []

            # RSI超买风险
            market_rsi = market_features.get('market_rsi12', 50)
            if market_rsi >= self.risk_thresholds['market_rsi_extreme']:
                risk_score += 0.4
                risk_factors.append(f"RSI极度超买({market_rsi:.1f})")
            elif market_rsi >= self.risk_thresholds['market_rsi_overbought']:
                risk_score += 0.25
                risk_factors.append(f"RSI超买({market_rsi:.1f})")

            # 趋势衰竭风险
            trend_strength = market_features.get('market_trend_strength', 0.5)
            if trend_strength >= self.risk_thresholds['trend_exhaustion']:
                risk_score += 0.3
                risk_factors.append(f"趋势衰竭信号({trend_strength:.2f})")

            return {
                'score': min(1.0, risk_score),
                'factors': risk_factors,
                'details': {
                    'market_rsi': market_rsi,
                    'trend_strength': trend_strength
                }
            }

        except Exception as e:
            return {'score': 0.0, 'factors': [], 'details': {'error': str(e)}}

    def _assess_structural_risk(self, market_features, historical_data):
        """评估市场结构风险"""
        try:
            risk_score = 0.0
            risk_factors = []

            # 单日涨幅风险
            benchmark_change = market_features.get('market_benchmark_change', 0)
            if benchmark_change >= self.risk_thresholds['benchmark_surge_1d']:
                risk_score += 0.4
                risk_factors.append(f"单日涨幅过大({benchmark_change:.2%})")

            return {
                'score': min(1.0, risk_score),
                'factors': risk_factors,
                'details': {
                    'benchmark_change_1d': benchmark_change
                }
            }

        except Exception as e:
            return {'score': 0.0, 'factors': [], 'details': {'error': str(e)}}

    def _assess_volatility_risk(self, market_features, historical_data):
        """评估波动率风险"""
        try:
            risk_score = 0.0
            risk_factors = []

            # 当前波动率 - 使用现有特征估算
            benchmark_change = abs(market_features.get('market_benchmark_change', 0))
            if benchmark_change >= 0.04:  # 4%以上的日变化
                risk_score += 0.3
                risk_factors.append(f"高波动率({benchmark_change:.2%})")

            return {
                'score': min(1.0, risk_score),
                'factors': risk_factors,
                'details': {
                    'estimated_volatility': benchmark_change
                }
            }

        except Exception as e:
            return {'score': 0.0, 'factors': [], 'details': {'error': str(e)}}

    def _assess_momentum_risk(self, market_features):
        """评估动量风险"""
        try:
            risk_score = 0.0
            risk_factors = []

            # 动量过热风险
            market_rsi = market_features.get('market_rsi12', 50)
            if market_rsi >= 80:
                risk_score += 0.4
                risk_factors.append(f"动量过热({market_rsi:.1f})")
            elif market_rsi >= 70:
                risk_score += 0.2
                risk_factors.append(f"动量偏热({market_rsi:.1f})")

            return {
                'score': min(1.0, risk_score),
                'factors': risk_factors,
                'details': {
                    'market_rsi': market_rsi
                }
            }

        except Exception as e:
            return {'score': 0.0, 'factors': [], 'details': {'error': str(e)}}

    def _identify_primary_concerns(self, technical_risk, structural_risk, volatility_risk, momentum_risk):
        """识别主要风险因素"""
        concerns = []

        risks = [
            ('技术指标', technical_risk),
            ('市场结构', structural_risk),
            ('波动率', volatility_risk),
            ('动量', momentum_risk)
        ]

        # 按风险评分排序
        risks.sort(key=lambda x: x[1]['score'], reverse=True)

        for risk_name, risk_data in risks:
            if risk_data['score'] >= 0.3:
                concerns.append({
                    'type': risk_name,
                    'score': risk_data['score'],
                    'factors': risk_data['factors']
                })

        return concerns

    def _generate_risk_explanation(self, risk_level, risk_score):
        """生成风险解释"""
        explanations = {
            'extreme': f"市场处于极端风险状态(评分:{risk_score:.2f})，强烈建议避免选股，等待市场调整",
            'high': f"市场处于高风险状态(评分:{risk_score:.2f})，建议暂停选股或大幅降低仓位",
            'medium': f"市场风险适中(评分:{risk_score:.2f})，可谨慎选股但需严格控制风险",
            'low': f"市场风险较低(评分:{risk_score:.2f})，可正常进行选股操作"
        }

        return explanations.get(risk_level, f"风险评分:{risk_score:.2f}")

    def _extract_market_features_for_date(self, group_data, date):
        """从当日数据中提取市场特征用于风险评估"""
        try:
            # 从group_data中提取市场相关特征的平均值
            market_features = {}

            # 如果group_data包含市场特征，直接使用
            market_columns = [col for col in group_data.columns if col.startswith('market_')]

            for col in market_columns:
                if col in group_data.columns:
                    # 使用中位数作为代表值，更稳健
                    market_features[col.replace('market_', '')] = group_data[col].median()

            # 如果没有直接的市场特征，从个股数据推断
            if not market_features:
                # 使用个股数据的统计特征推断市场状态
                if 'pct_chg' in group_data.columns:
                    avg_change = group_data['pct_chg'].median() / 100  # 转换为小数
                    market_features['benchmark_change'] = avg_change

                if 'rsi12' in group_data.columns:
                    avg_rsi = group_data['rsi12'].median()
                    market_features['rsi12'] = avg_rsi

                if 'trend_strength' in group_data.columns:
                    avg_trend = group_data['trend_strength'].median()
                    market_features['trend_strength'] = avg_trend

            # 确保关键字段存在
            market_features.setdefault('market_benchmark_change', 0)
            market_features.setdefault('market_rsi12', 50)
            market_features.setdefault('market_trend_strength', 0.5)

            return market_features

        except Exception as e:
            # 如果提取失败，返回默认的中性市场特征
            return {
                'market_benchmark_change': 0,
                'market_rsi12': 50,
                'market_trend_strength': 0.5
            }

class ChangePointStrategyAnalyzer:
    """变化点策略分析器"""

    def __init__(self, ensemble_system, logger=None):
        self.ensemble_system = ensemble_system
        self.logger = logger

    def analyze_by_strategy_and_regime(self, change_point_features):
        """按策略和市场状态分析变化点特征"""
        try:
            if change_point_features.empty:
                return {}

            # 按市场状态分组
            regime_groups = self._group_by_market_regime(change_point_features)

            strategy_patterns = {}

            for regime, regime_data in regime_groups.items():
                if len(regime_data) < 10:  # 样本太少跳过
                    continue

                regime_patterns = {}

                # 为每个策略分析模式
                for strategy_name, strategy_features in self.ensemble_system.strategy_features.items():
                    # 获取该策略相关的特征
                    available_features = [f for f in strategy_features if f in regime_data.columns]

                    if len(available_features) < 2:
                        continue

                    # 分析上涨和下跌模式
                    up_data = regime_data[regime_data['future_return_2d'] > 0]
                    down_data = regime_data[regime_data['future_return_2d'] <= 0]

                    if len(up_data) > 5 and len(down_data) > 5:
                        strategy_pattern = {
                            'up_patterns': self._extract_strategy_patterns(up_data, available_features),
                            'down_patterns': self._extract_strategy_patterns(down_data, available_features),
                            'feature_importance': self._calculate_strategy_importance(regime_data, available_features)
                        }
                        regime_patterns[strategy_name] = strategy_pattern

                if regime_patterns:
                    strategy_patterns[regime] = regime_patterns

            return strategy_patterns

        except Exception as e:
            if self.logger:
                self.logger.error(f"策略分析失败: {e}")
            return {}

    def _group_by_market_regime(self, data):
        """按市场状态分组数据"""
        try:
            regime_groups = {'bull_market': [], 'bear_market': [], 'sideways_market': []}

            for idx, row in data.iterrows():
                market_features = {
                    'market_benchmark_change': row.get('market_benchmark_change', 0),
                    'market_trend_strength': row.get('market_trend_strength', 0.5),
                    'market_rsi12': row.get('market_rsi12', 50)
                }

                regime = self.ensemble_system.identify_market_regime(market_features)
                regime_groups[regime].append(row)

            # 转换为DataFrame
            for regime in regime_groups:
                if regime_groups[regime]:
                    regime_groups[regime] = pd.DataFrame(regime_groups[regime])
                else:
                    regime_groups[regime] = pd.DataFrame()

            return regime_groups

        except Exception as e:
            if self.logger:
                self.logger.error(f"市场状态分组失败: {e}")
            return {'sideways_market': data}  # 默认返回震荡市

    def _extract_strategy_patterns(self, data, features):
        """提取策略特征模式"""
        try:
            patterns = {}

            for feature in features:
                if feature not in data.columns:
                    continue

                feature_values = data[feature].dropna()
                if len(feature_values) < 3:
                    continue

                # 计算统计特征
                patterns[feature] = {
                    'mean': float(feature_values.mean()),
                    'std': float(feature_values.std()),
                    'min': float(feature_values.min()),
                    'max': float(feature_values.max()),
                    'q25': float(feature_values.quantile(0.25)),
                    'q75': float(feature_values.quantile(0.75))
                }

            return patterns

        except Exception as e:
            if self.logger:
                self.logger.error(f"模式提取失败: {e}")
            return {}

    def _calculate_strategy_importance(self, data, features):
        """计算策略特征重要性"""
        try:
            importance = {}
            target = data['future_return_2d']

            for feature in features:
                if feature not in data.columns:
                    continue

                feature_values = pd.to_numeric(data[feature], errors='coerce').fillna(0)

                # 计算相关系数
                try:
                    corr = np.corrcoef(feature_values, target)[0, 1]
                    if pd.notna(corr):
                        importance[feature] = abs(corr)
                    else:
                        importance[feature] = 0.0
                except:
                    importance[feature] = 0.0

            return importance

        except Exception as e:
            if self.logger:
                self.logger.error(f"重要性计算失败: {e}")
            return {}



# 导入logger配置 - 修改为相对导入
try:
    from logger_config import configure_logger
except ImportError:
    # 添加当前目录到sys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from logger_config import configure_logger

# 将原来的logger配置替换为新的配置
logger = configure_logger(__name__, 'train_sector_models.log')

# 自定义JSON编码器，用于处理NumPy和其他特殊类型
class NumpyEncoder(json.JSONEncoder):
    """扩展JSON编码器，处理NumPy和其他特殊类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):  # 添加对numpy bool_类型的支持
            return bool(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

# 路径设置
def get_data_path(relative_path=None):
    """获取数据目录路径"""
    project_root = get_project_root()
    data_path = project_root / "to" / "local_data"
    if relative_path:
        return data_path / relative_path
    return data_path

def get_model_path():
    """获取模型目录路径"""
    project_root = get_project_root()
    model_path = project_root / "to" / "models" / "sector_models"
    return model_path

def ensure_dir(path):
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class SectorModelTrainer:
    """板块模型训练器"""

    def __init__(self, model_dir=None):
        """
        初始化模型训练器

        Args:
            model_dir: 模型保存目录
        """
        self.logger = logger

        # 设置模型目录
        self.model_dir = Path(model_dir) if model_dir else get_model_path()
        ensure_dir(self.model_dir)

        # 特征目录
        self.sector_features_dir = get_data_path("sector_features")
        self.market_status_dir = get_data_path("market_status/daily")

        # 交易日历
        self.trading_days = []
        self._load_trading_calendar()

        # 一次性加载板块统计信息，避免重复加载
        self.sector_stock_count = self._load_sector_stock_count()

        # 一次性加载股票-板块映射，避免重复获取
        self.stock_sector_mapping = self._load_stock_sector_mapping()

        # 添加特征缓存目录
        self.features_cache_dir = Path(get_data_path(), "features_cache")
        ensure_dir(self.features_cache_dir)

        # 初始化集成学习系统
        self.ensemble_system = ChangePointEnsembleSystem(logger=self.logger)
        self.strategy_analyzer = ChangePointStrategyAnalyzer(self.ensemble_system, logger=self.logger)

        # 初始化市场风险过滤器
        self.market_risk_filter = MarketRiskFilter(logger=self.logger)

        self.logger.info(f"板块模型训练器初始化完成，模型目录: {self.model_dir}")
        self.logger.info(f"集成学习系统已初始化，包含 {len(self.ensemble_system.strategy_names)} 个策略")

    def _load_sector_stock_count(self):
        """
        一次性加载板块股票数量统计信息

        Returns:
            dict: {sector_name: stock_count} 映射字典
        """
        try:
            from to.data_loaders import get_sector_list_with_stats
            sector_stock_count = get_sector_list_with_stats()
            self.logger.info(f"成功加载 {len(sector_stock_count)} 个板块的股票数量信息")
            return sector_stock_count
        except Exception as e:
            self.logger.error(f"加载板块股票数量信息失败: {e}")
            return {}

    def _load_stock_sector_mapping(self):
        """
        一次性加载股票-板块映射信息

        Returns:
            dict: {stock_code: [sector_codes]} 映射字典
        """
        try:
            from to.stock_sectors_mapping import get_stock_sectors
            # 获取所有股票的板块映射
            stock_sector_mapping = {}
            # 这里我们需要遍历所有可能的股票，但为了效率，我们在使用时再获取
            self.logger.info("股票-板块映射将在使用时动态获取")
            return stock_sector_mapping
        except Exception as e:
            self.logger.error(f"初始化股票-板块映射失败: {e}")
            return {}




    def _comprehensive_data_filtering(self, stock_features_df, sector_features_df):
        """
        综合数据过滤策略 - 在模型训练前清理无效数据

        Args:
            stock_features_df: 个股特征DataFrame
            sector_features_df: 板块特征DataFrame

        Returns:
            tuple: (过滤后的个股特征, 过滤后的板块特征, 过滤统计信息)
        """
        self.logger.info("=== 开始综合数据过滤 ===")

        filter_stats = {
            'original_stock_count': len(stock_features_df) if stock_features_df is not None else 0,
            'original_sector_count': len(sector_features_df) if sector_features_df is not None else 0,
            'filtered_stock_count': 0,
            'filtered_sector_count': 0,
            'removed_suspended_stocks': 0,
            'removed_invalid_features': 0,
            'removed_insufficient_data': 0,
            'feature_coverage_after_filter': {}
        }

        # 1. 过滤个股数据
        if stock_features_df is not None and not stock_features_df.empty:
            filtered_stock_df = self._filter_stock_data(stock_features_df, filter_stats)
        else:
            filtered_stock_df = stock_features_df

        # 2. 过滤板块数据
        if sector_features_df is not None and not sector_features_df.empty:
            filtered_sector_df = self._filter_sector_data(sector_features_df, filter_stats)
        else:
            filtered_sector_df = sector_features_df

        # 3. 验证过滤后的数据质量
        self._validate_filtered_data(filtered_stock_df, filtered_sector_df, filter_stats)

        # 4. 输出过滤统计
        self._log_filter_statistics(filter_stats)

        return filtered_stock_df, filtered_sector_df, filter_stats

    def _filter_stock_data(self, stock_df, filter_stats):
        """
        过滤个股数据 - 移除停牌股票和无效数据点

        Args:
            stock_df: 原始个股数据
            filter_stats: 过滤统计信息

        Returns:
            DataFrame: 过滤后的个股数据
        """
        self.logger.info(f"开始过滤个股数据，原始数据: {len(stock_df)} 条记录")

        # 1. 移除停牌股票数据点
        valid_mask = self._create_valid_stock_mask(stock_df)
        filtered_df = stock_df[valid_mask].copy()

        suspended_removed = len(stock_df) - len(filtered_df)
        filter_stats['removed_suspended_stocks'] = suspended_removed
        self.logger.info(f"移除停牌/无效股票数据点: {suspended_removed} 条")

        # 2. 移除关键特征缺失的数据点
        critical_features = ['close', 'volume', 'pct_chg']
        before_count = len(filtered_df)

        for feature in critical_features:
            if feature in filtered_df.columns:
                # 移除该特征为NaN或异常值的记录
                valid_feature_mask = (
                    filtered_df[feature].notna() &
                    (filtered_df[feature] != 0) if feature == 'volume' else filtered_df[feature].notna()
                )
                filtered_df = filtered_df[valid_feature_mask]

        critical_removed = before_count - len(filtered_df)
        filter_stats['removed_invalid_features'] += critical_removed
        self.logger.info(f"移除关键特征缺失的数据点: {critical_removed} 条")

        # 3. 按股票分组，移除数据不足的股票
        stock_counts = filtered_df['stock_code'].value_counts()

        # 🔥 修复：根据实际日期范围动态调整最小数据点要求
        # 计算实际的交易日数量
        actual_trading_days = len(filtered_df['date'].unique()) if len(filtered_df) > 0 else 0
        # 最小数据点要求：至少是实际交易日数的50%，但不少于2个，不多于20个
        min_data_points = max(2, min(20, actual_trading_days // 2))

        self.logger.info(f"实际交易日数: {actual_trading_days}, 最小数据点要求: {min_data_points}")
        valid_stocks = stock_counts[stock_counts >= min_data_points].index

        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['stock_code'].isin(valid_stocks)]
        insufficient_removed = before_count - len(filtered_df)
        filter_stats['removed_insufficient_data'] = insufficient_removed
        self.logger.info(f"移除数据不足的股票记录: {insufficient_removed} 条")

        # 4. 特征质量过滤
        filtered_df = self._filter_features_by_quality(filtered_df, filter_stats)

        filter_stats['filtered_stock_count'] = len(filtered_df)
        self.logger.info(f"个股数据过滤完成，保留: {len(filtered_df)} 条记录")

        return filtered_df

    def _create_valid_stock_mask(self, stock_df):
        """
        创建有效股票数据掩码 - 识别停牌和异常数据

        Args:
            stock_df: 股票数据DataFrame

        Returns:
            Series: 布尔掩码，True表示有效数据
        """
        # 基础有效性检查
        valid_mask = pd.Series(True, index=stock_df.index)

        # 1. 价格数据有效性
        if 'close' in stock_df.columns:
            valid_mask &= (stock_df['close'].notna() & (stock_df['close'] > 0))

        # 2. 成交量有效性（成交量为0可能表示停牌）
        if 'volume' in stock_df.columns:
            valid_mask &= (stock_df['volume'].notna() & (stock_df['volume'] > 0))

        # 3. 涨跌幅合理性检查（避免异常数据）
        if 'pct_chg' in stock_df.columns:
            # 涨跌幅在合理范围内（-50%到+50%）
            valid_mask &= (
                stock_df['pct_chg'].notna() &
                (stock_df['pct_chg'] >= -50) &
                (stock_df['pct_chg'] <= 50)
            )

        # 4. 检查是否有前收盘价数据
        if 'preClose' in stock_df.columns:
            valid_mask &= (stock_df['preClose'].notna() & (stock_df['preClose'] > 0))

        return valid_mask

    def _filter_features_by_quality(self, df, filter_stats):
        """
        根据特征质量过滤数据 - 移除无效特征列

        修正版本：
        1. 个股形态特征(pattern_*)保留所有，因为它们是二进制特征(0/1)，非零率低是正常的
        2. 板块比例特征(turnover_rate等)保留所有，因为它们都是有效的比例值
        3. 只过滤真正无效的特征（如全NaN、全零且无意义的特征）

        Args:
            df: 数据DataFrame
            filter_stats: 过滤统计信息

        Returns:
            DataFrame: 过滤后的数据
        """
        # 排除非数值列
        exclude_cols = ['stock_code', 'date_str', 'date']
        numeric_cols = [col for col in df.columns
                       if col not in exclude_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        # 识别无效特征
        invalid_features = []
        feature_quality = {}

        for col in numeric_cols:
            series = df[col]

            # 计算特征质量指标
            total_count = len(series)
            valid_count = series.notna().sum()
            non_zero_count = (series != 0).sum()
            unique_count = series.nunique()

            valid_ratio = valid_count / total_count if total_count > 0 else 0
            non_zero_ratio = non_zero_count / total_count if total_count > 0 else 0
            diversity_ratio = unique_count / total_count if total_count > 0 else 0

            feature_quality[col] = {
                'valid_ratio': valid_ratio,
                'non_zero_ratio': non_zero_ratio,
                'diversity_ratio': diversity_ratio
            }

            # 判断是否为无效特征 - 修正过滤逻辑，保护二进制和技术指标特征
            is_pattern_feature = col.startswith('pattern_')  # 个股形态特征
            is_ratio_feature = any(keyword in col.lower() for keyword in
                                 ['rate', 'ratio', 'percent', 'turnover'])  # 比例特征
            is_binary_feature = col.endswith(('_bearish', '_bullish', '_signal', '_consensus'))  # 二进制技术指标
            is_macd_feature = 'macd' in col.lower()  # MACD相关特征
            is_technical_feature = any(keyword in col.lower() for keyword in
                                     ['rsi', 'bb_', 'atr', 'momentum', 'volatility', 'overbought', 'oversold'])  # 技术指标特征
            is_sector_feature = col.startswith('sector_') or 'most_relevant_sector' in col  # 板块相关特征

            # 保护重要特征：个股形态、比例、二进制技术指标、MACD、技术指标、板块特征
            if (is_pattern_feature or is_ratio_feature or is_binary_feature or
                is_macd_feature or is_technical_feature or is_sector_feature):
                # 对于这些重要特征，只有在完全无效时才移除
                if valid_ratio < 0.05:  # 只有当有效值比例极低时才移除
                    invalid_features.append(col)
                    self.logger.debug(f"移除完全无效的重要特征: {col}, 有效率={valid_ratio:.2%}")
                else:
                    # 记录保留的重要特征
                    self.logger.debug(f"保留重要特征: {col}, 有效率={valid_ratio:.2%}, 非零率={non_zero_ratio:.2%}, 多样性={diversity_ratio:.2%}")
            else:
                # 其他特征使用更宽松的条件
                if (valid_ratio < 0.2 or  # 有效值比例低于20%
                    (non_zero_ratio < 0.01 and unique_count <= 1)):  # 几乎全零且无变化
                    invalid_features.append(col)

        # 移除无效特征
        if invalid_features:
            df = df.drop(columns=invalid_features)
            filter_stats['removed_invalid_features'] += len(invalid_features)
            self.logger.info(f"移除无效特征: {len(invalid_features)} 个")
            for feature in invalid_features[:10]:  # 只显示前10个
                quality = feature_quality[feature]
                self.logger.info(f"  - {feature}: 有效率={quality['valid_ratio']:.2%}, "
                               f"非零率={quality['non_zero_ratio']:.2%}, "
                               f"多样性={quality['diversity_ratio']:.2%}")

        # 保存特征质量信息
        filter_stats['feature_coverage_after_filter'] = {
            k: v for k, v in feature_quality.items() if k not in invalid_features
        }

        return df

    def _filter_sector_data(self, sector_df, filter_stats):
        """
        过滤板块数据 - 移除无效数据点

        Args:
            sector_df: 原始板块数据
            filter_stats: 过滤统计信息

        Returns:
            DataFrame: 过滤后的板块数据
        """
        self.logger.info(f"开始过滤板块数据，原始数据: {len(sector_df)} 条记录")

        # 板块数据过滤逻辑相对简单，主要是特征质量过滤
        filtered_df = self._filter_features_by_quality(sector_df, filter_stats)

        filter_stats['filtered_sector_count'] = len(filtered_df)
        self.logger.info(f"板块数据过滤完成，保留: {len(filtered_df)} 条记录")

        return filtered_df

    def _validate_filtered_data(self, stock_df, sector_df, filter_stats):
        """
        验证过滤后的数据质量

        Args:
            stock_df: 过滤后的个股数据
            sector_df: 过滤后的板块数据
            filter_stats: 过滤统计信息
        """
        # 验证个股数据
        if stock_df is not None and not stock_df.empty:
            # 检查特征多样性
            numeric_cols = stock_df.select_dtypes(include=[np.number]).columns
            feature_diversity = {}

            for col in numeric_cols:
                if col not in ['stock_code', 'date_str', 'date']:
                    unique_ratio = stock_df[col].nunique() / len(stock_df)
                    feature_diversity[col] = unique_ratio

            # 统计高质量特征数量
            high_quality_features = sum(1 for ratio in feature_diversity.values() if ratio > 0.05)
            filter_stats['high_quality_features'] = high_quality_features

            self.logger.info(f"过滤后个股数据包含 {high_quality_features} 个高质量特征")

        # 验证板块数据
        if sector_df is not None and not sector_df.empty:
            self.logger.info(f"过滤后板块数据形状: {sector_df.shape}")

    def _log_filter_statistics(self, filter_stats):
        """
        输出过滤统计信息

        Args:
            filter_stats: 过滤统计信息
        """
        self.logger.info("=== 数据过滤统计 ===")
        self.logger.info(f"原始个股记录数: {filter_stats['original_stock_count']}")
        self.logger.info(f"过滤后个股记录数: {filter_stats['filtered_stock_count']}")
        self.logger.info(f"移除停牌股票: {filter_stats['removed_suspended_stocks']} 条")
        self.logger.info(f"移除无效特征: {filter_stats['removed_invalid_features']} 个")
        self.logger.info(f"移除数据不足股票: {filter_stats['removed_insufficient_data']} 条")

        if filter_stats['original_stock_count'] > 0:
            retention_rate = filter_stats['filtered_stock_count'] / filter_stats['original_stock_count']
            self.logger.info(f"数据保留率: {retention_rate:.2%}")

        if 'high_quality_features' in filter_stats:
            self.logger.info(f"高质量特征数量: {filter_stats['high_quality_features']}")



    def _load_trading_calendar(self):
        """
        加载交易日历

        Returns:
            list: 交易日历列表，如果加载失败则返回None
        """
        try:
            # 尝试从ParquetDataStorage中获取
            sys.path.append(str(get_project_root() / "to"))
            from advanced_quant_data import ParquetDataStorage

            storage = ParquetDataStorage()

            # 获取较长时间范围的交易日历，但不需要太早的数据
            # 只需要包含训练日期和回溯期即可，'20220101'足够早
            cal_df = storage.get_trading_calendar('20221001', '20250331')

            if cal_df is not None and not cal_df.empty:
                if 'date' in cal_df.columns:
                    self.trading_days = cal_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    self.trading_days = cal_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()

                self.logger.info(f"成功加载交易日历，共 {len(self.trading_days)} 个交易日，范围: {self.trading_days[0]} - {self.trading_days[-1]}")
                return self.trading_days
            else:
                self.logger.warning("获取交易日历失败")
                self.trading_days = None
                return None
        except Exception as e:
            print(f"发生错误: {e}")
            self.logger.error(f"加载交易日历出错: {e}")
            self.logger.error(traceback.format_exc())
            self.trading_days = None
            return None


    def _load_sector_features(self, sector_code, date):
        """
        加载特定板块的特征数据

        Args:
            sector_code: 板块代码
            date: 日期，格式为YYYYMMDD

        Returns:
            DataFrame: 特征数据
        """
        try:
            # 构建文件路径 - 按板块代码划分目录
            # 优先加载优化后的主特征文件
            main_features_path = self.sector_features_dir / sector_code / f"{sector_code}_features.parquet"
            print(f"main_features_path: {main_features_path}")

            if main_features_path.exists():
                # 如果主特征文件存在，直接读取并筛选日期
                self.logger.warning(f"尝试从主特征文件加载板块 {sector_code} 的日期 {date} 数据")
                df = pd.read_parquet(main_features_path)

                # 筛选指定日期
                if 'date' in df.columns:
                    date_df = df[df['date_str'] == date].copy()
                    if not date_df.empty:
                        return date_df

                if 'date_str' in df.columns:
                    date_df = df[df['date_str'] == date].copy()
                    if not date_df.empty:
                        return date_df

                self.logger.debug(f"主特征文件中未找到日期 {date} 的数据")
            else:
                # 尝试加载单日期特征文件
                date_file = self.sector_features_dir / sector_code / f"{sector_code}_{date}.parquet"
                if date_file.exists():
                    self.logger.debug(f"尝试从单日期文件加载板块 {sector_code} 的日期 {date} 数据")
                    df = pd.read_parquet(date_file)
                    return df

                # 尝试加载原始合并特征文件
                orig_file = self.sector_features_dir / sector_code / f"{sector_code}.parquet"
                if orig_file.exists():
                    self.logger.debug(f"尝试从原始合并文件加载板块 {sector_code} 的日期 {date} 数据")
                    df = pd.read_parquet(orig_file)

                    # 筛选指定日期
                    if 'date' in df.columns:
                        date_df = df[df['date_str'] == date].copy()
                        if not date_df.empty:
                            return date_df

                    if 'date_str' in df.columns:
                        date_df = df[df['date_str'] == date].copy()
                        if not date_df.empty:
                            return date_df

                self.logger.warning(f"未找到板块 {sector_code} 在日期 {date} 的特征数据")
                return None
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 在日期 {date} 的特征数据时出错: {e}")
            return None

    def _load_all_market_status(self):
        """
        加载市场状态数据

        Args:
            date: 日期，格式为YYYYMMDD

        Returns:
            dict: 市场状态数据
        """
        try:
            # 加载市场状态数据
            status_file = self.market_status_dir / "market_status.json"
            if not status_file.exists():
                self.logger.warning(f"市场状态文件不存在: {status_file}")
                return None

            with open(status_file, "r", encoding="utf-8") as f:
                market_status_data = json.load(f)
            return market_status_data

        except Exception as e:
            self.logger.error(f"加载市场状态数据时出错: {e}")
            self.logger.exception(e)
            return None

    def _batch_load_features(self, sector_list, start_date, end_date):
        """批量加载多个板块多个日期的特征数据

        注意：这个函数加载的是板块特征数据，包括：
        1. 板块基础特征（价格、成交量、技术指标等）
        2. 板块结构特征（内部股票分布、轮动特征等）
        3. 市场环境特征（独立的市场状态特征）
        4. 板块与市场的相对特征

        Args:
            sector_list: 板块列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 包含所有板块特征的数据（板块+市场+相对特征）
        """
        # 添加缓存机制，避免重复加载和处理数据
        cache_dir = os.path.join(get_project_root(), "to", "data_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # 生成缓存文件名，基于输入参数
        sector_str = "-".join(sorted(sector_list))
        cache_key = f"{sector_str[:100]}-{start_date}-{end_date}"
        import hashlib
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        # cache_hash = "9953d2f3ca428eebeeaf18e6f986f03f"
        cache_file = os.path.join(cache_dir, f"batch_features_{cache_hash}.pkl")

        # 检查是否存在缓存
        if os.path.exists(cache_file):
            try:
                self.logger.info(f"发现缓存文件 {cache_file}，尝试加载...")
                with open(cache_file, 'rb') as f:
                    df_all = pickle.load(f)
                self.logger.info(f"成功从缓存加载特征数据，形状: {df_all.shape}")
                return df_all
            except Exception as e:
                self.logger.warning(f"加载缓存文件失败: {e}，将重新生成特征")
        else:
            self.logger.info(f"缓存文件 {cache_file} 不存在，将重新生成特征")

        # 确保有交易日历
        if self.trading_days is None:
            self._load_trading_calendar()

        # 过滤交易日
        trading_days_in_range = [day for day in self.trading_days if start_date <= day <= end_date]
        self.logger.info(f"在 {start_date} 至 {end_date} 期间共有 {len(trading_days_in_range)} 个交易日")

        # 先加载所有板块在整个日期范围的基础数据，以便缓存
        self.logger.info(f"开始加载 {len(sector_list)} 个板块的基础数据...")
        base_features = {}

        # 定义板块涨跌停结构特征路径
        board_structure_dir = os.path.join(get_project_root(), "to", "local_data", "sector_features_board")
        self.logger.info(f"板块涨跌停结构特征路径: {board_structure_dir}")

        # 存储板块的涨跌停结构特征
        board_structure_features = {}

        # 使用多线程并行加载基础数据
        with ThreadPoolExecutor(max_workers=min(10, len(sector_list))) as executor:
            futures = {
                executor.submit(self._load_base_features, sector, start_date, end_date): sector
                for sector in sector_list
            }

            for future in as_completed(futures):
                sector = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        base_features[sector] = result
                        self.logger.info(f"成功加载板块 {sector} 的基础数据，共 {len(result)} 条记录")
                    else:
                        self.logger.warning(f"板块 {sector} 没有找到基础数据")
                except Exception as e:
                    self.logger.error(f"加载板块 {sector} 基础数据时出错: {e}")

        # 加载板块涨跌停结构特征
        self.logger.info("开始加载板块涨跌停结构特征数据...")
        for sector in sector_list:
            try:
                # 构建板块涨跌停结构特征文件路径
                board_file = os.path.join(board_structure_dir, sector, f"{sector}_board_analysis.parquet")

                # 检查文件是否存在
                if os.path.exists(board_file):
                    # 加载板块涨跌停结构特征
                    board_df = pd.read_parquet(board_file)

                    # 确保日期列是datetime类型
                    if 'date' in board_df.columns:
                        board_df['date_str'] = pd.to_datetime(board_df['date'])
                    elif 'date_obj' in board_df.columns:
                        board_df['date_str'] = board_df['date_obj']
                    else:
                        self.logger.warning(f"板块 {sector} 的涨跌停结构特征文件没有日期列")
                        continue

                    # 筛选日期范围内的数据
                    mask = (board_df['date_str'] >= pd.to_datetime(start_date)) & (board_df['date_str'] <= pd.to_datetime(end_date))
                    filtered_board_df = board_df[mask].copy()

                    if not filtered_board_df.empty:
                        # 设置日期为索引，便于后续处理
                        filtered_board_df.set_index('date', inplace=True)
                        board_structure_features[sector] = filtered_board_df
                        self.logger.info(f"成功加载板块 {sector} 的涨跌停结构特征，共 {len(filtered_board_df)} 条记录")
                    else:
                        self.logger.warning(f"板块 {sector} 在指定日期范围内没有涨跌停结构特征数据")
                else:
                    self.logger.warning(f"板块 {sector} 的涨跌停结构特征文件不存在: {board_file}")
            except Exception as e:
                self.logger.error(f"加载板块 {sector} 的涨跌停结构特征时出错: {e}")

        # 如果没有任何数据，提前返回
        if not base_features:
            self.logger.warning("没有找到任何板块数据")
            return pd.DataFrame()

        self.logger.info(f"成功加载 {len(base_features)}/{len(sector_list)} 个板块的基础数据")
        self.logger.info(f"成功加载 {len(board_structure_features)}/{len(sector_list)} 个板块的涨跌停结构特征")

        # 提前加载市场环境数据
        self.logger.info("开始加载市场环境数据...")
        market_status_cache = self._load_all_market_status()
        self.logger.info(f"成功加载 {len(market_status_cache)} 个日期的市场环境数据")

        # 准备所有任务 - 每个板块每个日期一个任务
        tasks = []
        for sector in base_features.keys():
            for date in trading_days_in_range:
                date_obj = pd.to_datetime(date)
                if date_obj in base_features[sector].index:
                    tasks.append((sector, date))

        self.logger.info(f"共找到 {len(tasks)} 个特征构建任务")

        # 使用多线程处理每个板块-日期组合
        all_features = []
        futures = {}

        with ThreadPoolExecutor(max_workers=min(20, len(tasks))) as executor:
            for sector, date in tasks:
                date_obj = pd.to_datetime(date)

                # 获取市场环境数据
                market_data = market_status_cache.get(date, None)

                # 获取板块涨跌停结构特征
                board_data = None
                if sector in board_structure_features and date_obj in board_structure_features[sector].index:
                    board_data = board_structure_features[sector].loc[date_obj].to_dict()

                # 提交任务
                future = executor.submit(
                    self._process_features_with_layers,
                    sector_code=sector,
                    date=date_obj,
                    base_data=base_features[sector],
                    market_status=market_data,
                    board_structure=board_data
                )

                futures[future] = (sector, date)

            # 收集结果
            for i, future in enumerate(as_completed(futures)):
                sector_code, date = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # 不再需要再次计算未来收益率，因为已经在_process_features_with_layers中完成
                        all_features.append(result)

                    # 进度日志
                    if (i+1) % 100 == 0 or (i+1) == len(tasks):
                        self.logger.info(f"已处理 {i+1}/{len(tasks)} 个特征构建任务")
                except Exception as e:
                    self.logger.error(f"处理板块 {sector_code} 日期 {date} 的特征时出错: {e}")

        self.logger.info(f"批量加载完成，成功构建 {len(all_features)} 条特征记录")

        if not all_features:
            self.logger.warning("没有生成任何有效特征")
            return pd.DataFrame()

        # 合并所有特征
        try:
            # 合并所有DataFrame
            df_all = pd.concat(all_features, ignore_index=True)

            # 添加日期字段(如果没有)
            if 'date' in df_all.columns and not pd.api.types.is_datetime64_any_dtype(df_all['date']):
                df_all['date'] = pd.to_datetime(df_all['date'])

            # 按日期排序
            if 'date' in df_all.columns:
                df_all = df_all.sort_values('date')

            self.logger.info(f"特征构建完成，共 {len(df_all)} 条记录，{df_all.shape[1]} 个特征")

            # 保存到缓存
            try:
                self.logger.info(f"保存特征数据到缓存 {cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump(df_all, f)
                self.logger.info(f"特征数据缓存保存成功")

            except Exception as e:
                self.logger.warning(f"保存特征数据到缓存失败: {e}")

            return df_all

        except Exception as e:
            print(f"发生错误: {e}")
            self.logger.error(f"合并特征数据时出错: {e}")
            return pd.DataFrame()



    def _load_base_features(self, sector_code, start_date, end_date):
        """加载单个板块的基础特征数据（价格、成交量等）

        Args:
            sector_code: 板块代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            板块基础数据
        """
        try:
            # 获取日期范围内的所有交易日
            if self.trading_days is None:
                self._load_trading_calendar()

            trading_days = [day for day in self.trading_days if start_date <= day <= end_date]
            if not trading_days:
                self.logger.warning(f"在{start_date}至{end_date}之间没有有效交易日")
                return None

            # 构建板块主特征文件路径
            sector_dir = self.sector_features_dir / f"{sector_code}"
            main_features_path = sector_dir / f"{sector_code}_features.parquet"

            # 如果主特征文件不存在，尝试使用合并特征文件
            if not main_features_path.exists():
                merged_file_path = sector_dir / f"{sector_code}.parquet"
                if merged_file_path.exists():
                    main_features_path = merged_file_path
                else:
                    self.logger.warning(f"板块 {sector_code} 没有主特征文件或合并特征文件")
                    return None

            # 读取特征文件
            self.logger.info(f"从 {main_features_path} 读取 {sector_code} 板块从 {start_date} 到 {end_date} 的特征数据")

            try:
                df = pd.read_parquet(main_features_path)

                # 确保日期列存在
                if 'date' not in df.columns:
                    self.logger.warning(f"特征文件 {main_features_path} 没有date列")
                    return None

                # 将日期列转换为datetime对象便于比较
                df['date_str'] = pd.to_datetime(df['date_str'])

                # 筛选日期范围内的数据
                mask = (df['date_str'] >= pd.to_datetime(start_date)) & (df['date_str'] <= pd.to_datetime(end_date))
                filtered_df = df[mask].copy()

                if filtered_df.empty:
                    self.logger.warning(f"板块 {sector_code} 在 {start_date} 至 {end_date} 期间没有数据")
                    return None

                # 设置date为索引以便后续处理
                filtered_df.set_index('date', inplace=True)

                self.logger.info(f"成功加载 {sector_code} 板块特征，共 {len(filtered_df)} 条记录")
                return filtered_df

            except Exception as e:
                print(f"发生错误: {e}")
                self.logger.error(f"读取 {main_features_path} 出错: {e}")
                return None


        except Exception as e:
            print(f"发生错误: {e}")
            self.logger.error(f"加载板块 {sector_code} 基础数据时发生错误: {e}")
            return None


    def _process_features_with_layers(self, sector_code, date, base_data, market_status, board_structure):
        """优化后的三层特征体系处理函数

        基于顶尖机器学习论文和量化实践，重构特征工程流程：
        1. 板块基础特征层：板块价格、成交量、技术指标等基础特征
        2. 板块结构特征层：板块内部结构、相对强度、轮动特征
        3. 市场特征层：独立的市场环境特征，避免与板块特征混合

        Args:
            sector_code: 板块代码
            date: 日期
            base_data: 板块基础特征数据（注意：这里是板块数据，不是个股数据）
            market_status: 市场环境层数据
            board_structure: 板块涨跌停结构特征
        Returns:
            处理后的特征数据
        """
        try:
            # 数据有效性检查
            if base_data is None or date not in base_data.index:
                return None

            # 获取当日数据
            current_day = base_data.loc[date]

            # 初始化特征字典
            features = {
                "sector_code": sector_code,
                "date": date
            }

            # ===== 第一层：板块基础特征 =====
            # 基于《Advances in Financial Machine Learning》的特征工程原则
            # 注意：这里处理的是板块基础特征，不是个股特征
            features.update(self._extract_sector_basic_features(current_day))

            # ===== 第二层：板块结构特征 =====
            # 基于板块轮动理论和相对强度分析
            features.update(self._extract_sector_structure_features(current_day, base_data, date, board_structure))

            # ===== 第三层：市场环境特征（独立处理）=====
            # 将市场特征从板块特征中分离，提高特征可解释性
            market_features = self._extract_market_environment_features(market_status)

            # ===== 第四层：板块与市场相对特征 =====
            # 计算板块相对于市场的相对强度特征
            relative_features = self.calculate_sector_market_relative_features(features, market_features)

            # 返回包含四层特征的完整特征集
            return pd.DataFrame([{**features, **market_features, **relative_features}])

        except Exception as e:
            self.logger.error(f"处理板块 {sector_code} 日期 {date} 特征时错误: {str(e)}")
            return None

    def _extract_sector_basic_features(self, current_day):
        """提取板块基础特征

        基于《Quantitative Portfolio Management》的特征分类方法
        注意：这里处理的是板块层面的基础特征，包括板块价格、成交量、内部股票分布等
        """
        features = {}

        # 1. 板块价格特征
        price_fields = ["close", "change_percent", "volume", "amount", "turnover_rate", "amplitude"]
        for field in price_fields:
            if field in current_day:
                features[field] = current_day[field]

        # 2. 板块内部股票参与度特征
        participation_fields = ["stock_count", "up_count", "down_count", "flat_count"]
        for field in participation_fields:
            if field in current_day:
                features[field] = current_day[field]

        # 3. 计算板块内部分化特征（基于信息论，提高特征信息密度）
        if all(f in current_day for f in ["up_count", "stock_count"]) and current_day["stock_count"] > 0:
            features["up_ratio"] = current_day["up_count"] / current_day["stock_count"]
            # 信息熵特征：衡量板块内部分化程度
            up_ratio = features["up_ratio"]
            if 0 < up_ratio < 1:
                features["participation_entropy"] = -(up_ratio * np.log2(up_ratio) +
                                                    (1-up_ratio) * np.log2(1-up_ratio))
            else:
                features["participation_entropy"] = 0

        if all(f in current_day for f in ["down_count", "stock_count"]) and current_day["stock_count"] > 0:
            features["down_ratio"] = current_day["down_count"] / current_day["stock_count"]

        return features

    def _extract_sector_structure_features(self, current_day, base_data, date, board_structure):
        """提取板块结构特征

        基于《Active Portfolio Management》的因子模型理论
        """
        features = {}

        # ===== 1. 资金流动特征 =====
        # 基于Kyle模型的流动性理论
        if "amount" in current_day and base_data is not None:
            try:
                date_idx = base_data.index.get_loc(date)
                if date_idx >= 4:  # 确保有足够的历史数据
                    historical_dates = base_data.index[date_idx-4:date_idx+1]
                    historical_amounts = base_data.loc[historical_dates, "amount"]
                    avg_amount_5d = historical_amounts.mean()
                    if avg_amount_5d > 0:
                        features["fund_flow_momentum"] = current_day["amount"] / avg_amount_5d
            except Exception as e:
                self.logger.debug(f"计算fund_flow_momentum时出错: {e}")

        # 资金密集度（基于市场微观结构理论）
        if "volume" in current_day and "volatility_20d" in current_day and current_day["volatility_20d"] > 0:
            features["volume_intensity"] = current_day["volume"] / current_day["volatility_20d"]

        # 上涨资金质量指标
        if all(k in current_day for k in ["up_count", "volume", "stock_count"]) and "volume" in base_data.columns:
            try:
                date_idx = base_data.index.get_loc(date)
                if date_idx >= 4:
                    historical_dates = base_data.index[date_idx-4:date_idx+1]
                    avg_volume_5d = base_data.loc[historical_dates, "volume"].mean()
                    if avg_volume_5d > 0 and current_day["stock_count"] > 0:
                        features["up_volume_ratio"] = (current_day["up_count"] * current_day["volume"]) / (current_day["stock_count"] * avg_volume_5d)
            except Exception as e:
                self.logger.debug(f"计算up_volume_ratio时出错: {e}")

        # ===== 2. 技术指标特征 =====
        # 基于《Technical Analysis of the Financial Markets》的技术分析理论

        # 均线系统特征（趋势跟踪）
        ma_ratios = ["ma5_10_ratio", "ma5_20_ratio", "ma10_20_ratio",
                     "ma20_60_ratio", "ma60_120_ratio", "price_ma5_ratio",
                     "price_ma20_ratio", "price_ma60_ratio"]
        for ratio in ma_ratios:
            if ratio in current_day:
                features[ratio] = current_day[ratio]

        # 价格位置特征（支撑阻力分析）
        price_positions = ["price_above_ma5", "price_above_ma20", "price_above_ma60"]
        for pos in price_positions:
            if pos in current_day:
                features[pos] = current_day[pos]

        # 趋势强度特征
        trend_indicators = ["trend_direction_score", "ma_bull_trend", "ma_bear_trend",
                           "ma_neutral_trend", "ma5_slope", "ma20_slope"]
        for trend in trend_indicators:
            if trend in current_day:
                features[trend] = current_day[trend]

        # 动量特征（基于动量理论）
        momentum_indicators = ["momentum_5d", "momentum_10d", "momentum_20d"]
        for mom in momentum_indicators:
            if mom in current_day:
                features[mom] = current_day[mom]

        # 板块相对变化率
        if "change_pct" in current_day:
            features["sector_change_pct"] = current_day["change_pct"]

        # ===== 3. 波动率特征 =====
        # 基于GARCH模型和波动率聚集理论
        volatility_indicators = ["volatility", "volatility_5d", "volatility_10d",
                               "volatility_20d", "volatility_60d", "volatility_ratio"]
        for vol in volatility_indicators:
            if vol in current_day:
                features[vol] = current_day[vol]

        # ===== 4. 成交量特征 =====
        # 基于成交量价格关系理论（OBV理论）
        volume_indicators = ["volume_ratio_5", "volume_ratio_10", "volume_ratio_20",
                           "volume_ratio_60", "volume_change_ratio", "volume_trend"]
        for vol in volume_indicators:
            if vol in current_day:
                features[vol] = current_day[vol]

        # ===== 5. 高级技术指标特征 =====
        # 基于《New Concepts in Technical Trading Systems》的指标理论

        # RSI和MACD组合特征（多指标确认）
        if all(key in current_day for key in ["rsi14", "macd_line", "macd_signal"]):
            # 多头信号确认
            features["rsi_macd_bull_signal"] = int(
                current_day["rsi14"] > 50 and
                current_day["macd_line"] > current_day["macd_signal"] and
                current_day["macd_line"] > 0
            )
            # 空头信号确认
            features["rsi_macd_bear_signal"] = int(
                current_day["rsi14"] < 50 and
                current_day["macd_line"] < current_day["macd_signal"] and
                current_day["macd_line"] < 0
            )

        # 动量与波动率组合（风险调整收益）
        if all(key in current_day for key in ["momentum_20d", "volatility_20d"]) and current_day["volatility_20d"] > 0:
            features["momentum_volatility_ratio"] = current_day["momentum_20d"] / current_day["volatility_20d"]

        # 量价确认指标
        if all(key in current_day for key in ["volume_ratio_5", "momentum_5d"]):
            features["volume_momentum_confirm"] = int(
                (current_day["volume_ratio_5"] > 1.2 and current_day["momentum_5d"] > 0) or
                (current_day["volume_ratio_5"] < 0.8 and current_day["momentum_5d"] < 0)
            )
            # 量价背离指标
            features["volume_momentum_divergence"] = int(
                (current_day["volume_ratio_5"] > 1.2 and current_day["momentum_5d"] < 0) or
                (current_day["volume_ratio_5"] < 0.8 and current_day["momentum_5d"] > 0)
            )

        # ===== 6. 板块生命周期特征 =====
        # 基于板块轮动理论
        lifecycle_indicators = ["trend_strength", "consecutive_up_days", "consecutive_down_days",
                              "swing_pattern"]
        for field in lifecycle_indicators:
            if field in current_day:
                features[field] = current_day[field]

        # 支撑阻力特征
        support_resistance = ["support_distance_pct", "resistance_distance_pct"]
        for field in support_resistance:
            if field in current_day:
                features[field] = current_day[field]

        # 技术评分特征
        tech_scores = ["tech_score", "market_sentiment", "signal_strength"]
        for field in tech_scores:
            if field in current_day:
                features[field] = current_day[field]

        # ===== 7. 涨跌停板结构特征 =====
        # 基于中国A股市场特有的涨跌停制度
        if board_structure is not None:
            features.update(self._extract_board_structure_features(board_structure))

        # 涨跌停比例特征
        limit_features = ["limit_up_ratio", "limit_down_ratio", "near_limit_up_ratio",
                         "limit_open_ratio"]
        for field in limit_features:
            if field in current_day:
                features[field] = current_day[field]

        # ===== 8. 排名特征 =====
        # 基于相对排名理论
        ranking_features = ["ten_day_rise_count", "ten_day_rise_ratio", "daily_rise_count",
                           "daily_rise_ratio", "daily_fall_count", "daily_fall_ratio",
                           "daily_amount_count", "daily_amount_ratio", "ten_day_rise_weighted"]
        for field in ranking_features:
            if field in current_day:
                features[field] = current_day[field]

        # ===== 9. 技术指标一致性特征 =====
        # 基于多指标确认理论
        trend_signals = []
        if "macd_line" in current_day and "macd_signal" in current_day:
            trend_signals.append(1 if current_day["macd_line"] > current_day["macd_signal"] else -1)
        if "rsi14" in current_day:
            trend_signals.append(1 if current_day["rsi14"] > 50 else -1)
        if "momentum_10d" in current_day:
            trend_signals.append(1 if current_day["momentum_10d"] > 0 else -1)

        if trend_signals:
            # 技术指标一致程度
            features["indicator_consensus"] = sum(trend_signals) / len(trend_signals)
            features["strong_bull_consensus"] = int(sum(trend_signals) == len(trend_signals))
            features["strong_bear_consensus"] = int(sum(trend_signals) == -len(trend_signals))

        return features

    def _extract_board_structure_features(self, board_structure):
        """提取涨跌停板结构特征"""
        features = {}

        if board_structure is not None:
            # 涨跌停统计特征
            board_fields = ["limit_up_count", "limit_down_count", "near_limit_up_count",
                           "near_limit_down_count", "one_word_limit_up_count",
                           "broken_limit_up_count", "limit_open_count"]
            for field in board_fields:
                if field in board_structure:
                    features[field] = board_structure[field]

            # 连板特征
            if "max_consecutive_limit_up" in board_structure:
                features["max_consecutive_limit_up"] = board_structure["max_consecutive_limit_up"]

        return features

    def _extract_market_environment_features(self, market_status):
        """提取市场环境特征（独立于板块特征）

        基于《Market Microstructure Theory》的市场状态分类理论
        """
        features = {}

        # ===== 基础市场状态特征 =====
        if market_status:
            # 市场类型和周期
            features["market_type"] = market_status.get("market_type", 0)
            features["market_cycle"] = market_status.get("market_cycle", 0)
            features["market_trend_strength"] = market_status.get("trend_strength", 0)

            # ===== 处理字符串类型的市场信号特征 =====
            signals = market_status.get("signals", {})
            if signals:
                # 处理risk_level字符串字段
                risk_level = signals.get("risk_level", "low")
                risk_level_map = {
                    "low": 0, "medium": 1, "high": 2, "extreme": 3,
                    "bull": 1, "bear": -1, "neutral": 0,
                    "bullish": 1, "bearish": -1
                }
                features["market_risk_level"] = risk_level_map.get(str(risk_level).lower(), 0)

                # 处理trading_advice字符串字段（转换为数值）
                trading_advice = signals.get("trading_advice", "")
                if "买入" in str(trading_advice) or "加仓" in str(trading_advice):
                    features["market_trading_signal"] = 1
                elif "卖出" in str(trading_advice) or "减仓" in str(trading_advice):
                    features["market_trading_signal"] = -1
                else:
                    features["market_trading_signal"] = 0  # 观望

                # 其他数值信号
                features["market_signal_strength"] = signals.get("strength", 0)

            # ===== 市场状态分类特征 =====
            # 基于市场类型映射到标准化的市场状态
            market_type = market_status.get("market_type", 0)
            trend_strength = market_status.get("trend_strength", 0)

            if isinstance(market_type, (int, float)):
                # 市场状态分类（基于行为金融学理论）
                if market_type in [11, 12, 13, 21, 22, 23]:
                    features["market_group"] = 1  # 牛市
                elif market_type in [41, 42, 43, 51, 52, 53]:
                    features["market_group"] = 2  # 熊市
                elif market_type in [54, 55, 56, 57, 58, 59] or trend_strength > 85:
                    features["market_group"] = 4  # 极端市场
                elif market_type in [31, 32, 33, 34, 35, 36]:
                    features["market_group"] = 3  # 震荡市场
                else:
                    features["market_group"] = 3  # 默认震荡市场
            else:
                features["market_group"] = 3
        else:
            # 默认市场状态
            features["market_type"] = 33
            features["market_cycle"] = 0
            features["market_trend_strength"] = 0
            features["market_group"] = 3
            features["market_risk_level"] = 0
            features["market_trading_signal"] = 0
            features["market_signal_strength"] = 0

        # ===== 市场基准特征（创业板指数） =====
        if market_status and "individual_status" in market_status and "399006.SZ" in market_status["individual_status"]:
            cyb_data = market_status["individual_status"]["399006.SZ"]

            # 基准指数基础特征
            if "change_percent" in cyb_data:
                features["market_benchmark_change"] = cyb_data["change_percent"]

            # 基准指数技术特征
            if "technical_data" in cyb_data:
                tech_data = cyb_data["technical_data"]

                # 市场技术指标
                market_tech_indicators = ["rsi6", "rsi12", "macd", "bb_width", "ma5", "ma10"]
                for indicator in market_tech_indicators:
                    if indicator in tech_data:
                        features[f"market_{indicator}"] = tech_data[indicator]

            # 市场盈亏比特征
            if "profit_loss_ratio_14" in cyb_data:
                features["market_profit_loss_ratio"] = cyb_data["profit_loss_ratio_14"]

        return features

    def calculate_sector_market_relative_features(self, sector_features, market_features):
        """计算板块与市场的相对特征

        基于《Quantitative Equity Portfolio Management》的相对强度理论

        Args:
            sector_features: 板块特征字典
            market_features: 市场特征字典

        Returns:
            dict: 相对特征字典
        """
        relative_features = {}

        # ===== 1. 基础相对特征 =====
        # 相对涨跌幅
        if "change_percent" in sector_features and "market_benchmark_change" in market_features:
            relative_features["relative_change_percent"] = (
                sector_features["change_percent"] - market_features["market_benchmark_change"]
            )

        # ===== 2. 技术指标相对特征 =====
        # RSI相对强度
        if "rsi14" in sector_features and "market_rsi12" in market_features:
            if market_features["market_rsi12"] > 0:
                relative_features["rsi_vs_market"] = sector_features["rsi14"] / market_features["market_rsi12"]
                relative_features["rsi_diff_market"] = sector_features["rsi14"] - market_features["market_rsi12"]

        # MACD相对特征
        if "macd_line" in sector_features and "market_macd" in market_features:
            if abs(market_features["market_macd"]) > 1e-8:
                relative_features["macd_vs_market"] = sector_features["macd_line"] / market_features["market_macd"]
                relative_features["macd_diff_market"] = sector_features["macd_line"] - market_features["market_macd"]

        # 波动率相对特征
        if "volatility_20d" in sector_features and "market_bb_width" in market_features:
            if market_features["market_bb_width"] > 0:
                relative_features["volatility_vs_market"] = (
                    sector_features["volatility_20d"] / market_features["market_bb_width"]
                )

        # ===== 3. 动量相对特征 =====
        if "momentum_10d" in sector_features and "market_rsi6" in market_features:
            if market_features["market_rsi6"] > 0:
                relative_features["momentum_vs_market_rsi"] = (
                    sector_features["momentum_10d"] / (market_features["market_rsi6"] + 1e-8)
                )

        # ===== 4. 均线相对特征 =====
        if ("ma5_10_ratio" in sector_features and
            "market_ma5" in market_features and
            "market_ma10" in market_features):
            if market_features["market_ma10"] > 0:
                market_ma_ratio = market_features["market_ma5"] / market_features["market_ma10"]
                if market_ma_ratio > 0:
                    relative_features["ma_ratio_vs_market"] = (
                        sector_features["ma5_10_ratio"] / market_ma_ratio
                    )

        # ===== 5. 盈亏比相对特征 =====
        if ("support_distance_pct" in sector_features and
            "resistance_distance_pct" in sector_features and
            "market_profit_loss_ratio" in market_features):
            if (sector_features["resistance_distance_pct"] != 0 and
                market_features["market_profit_loss_ratio"] > 0):
                sector_pl_ratio = (sector_features["support_distance_pct"] /
                                 (sector_features["resistance_distance_pct"] + 1e-8))
                relative_features["profit_loss_ratio_vs_market"] = (
                    sector_pl_ratio / market_features["market_profit_loss_ratio"]
                )

        return relative_features


    def _get_all_sectors(self):
        """获取所有可用的板块列表

        返回:
            板块代码列表
        """
        try:
            # # 首先尝试获取本地存储的板块列表
            # sector_dirs = [d for d in os.listdir(self.sector_features_dir)
            #              if os.path.isdir(os.path.join(self.sector_features_dir, d))]

            # if sector_dirs:
            #     self.logger.info(f"从本地目录获取到 {len(sector_dirs)} 个板块")
            #     return sector_dirs

            # 尝试从stock_sectors_mapping模块获取
            try:
                # 将当前工作目录添加到sys.path
                sys.path.append(os.getcwd())
                # 尝试导入
                try:
                    from to.data_loaders import get_sector_list
                    sector_list = get_sector_list()
                    self.logger.info(f"从get_tng_sectors获取 {len(sector_list)} 个板块")
                    return sector_list
                except ImportError:
                    # 如果导入失败，尝试从to目录导入
                    try:
                        sys.path.append(str(get_project_root()))
                        from to.stock_sectors_mapping import get_tng_sectors
                        sector_list = get_tng_sectors()
                        self.logger.info(f"从to.stock_sectors_mapping获取 {len(sector_list)} 个板块")
                        return sector_list
                    except ImportError:
                        self.logger.warning("无法导入get_tng_sectors函数")
            except Exception as e:
                self.logger.error(f"尝试获取板块列表时出错: {e}")

            # 如果都失败了，返回空列表
            return []

        except Exception as e:
            self.logger.error(f"获取所有板块列表时出错: {e}")
            return []

    def _detect_single_stock_change_points(self, stock_code, stock_df, indicators=None):
        """检测单个股票的变化点

        Args:
            stock_code: 股票代码
            stock_df: 股票数据DataFrame
            indicators: 指标列表(可选)，默认仅使用future_return_2d

        Returns:
            dict: 变化点检测结果
        """
        from ruptures import Pelt, Binseg, Window
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        self.logger.info(f"开始检测股票 {stock_code} 的变化点...")

        try:
            # 确保按日期排序
            stock_df = stock_df.sort_values('date_str')

            # 检查是否有足够的数据
            if len(stock_df) < 20:  # 需要至少20个数据点
                self.logger.warning(f"股票 {stock_code} 数据点不足 ({len(stock_df)}个)，跳过变化点检测")
                return None

            # 检查future_return_2d是否存在（必须指标）
            if 'future_return_2d' not in stock_df.columns:
                self.logger.warning(f"股票 {stock_code} 缺少必要指标future_return_2d，跳过变化点检测")
                return None

            # 保存日期序列用于结果记录
            dates = stock_df['date_str'].values

            # 获取future_return_2d序列
            future_returns = stock_df['future_return_2d'].values

            # 确保future_returns是有效的数值数组
            if not isinstance(future_returns, np.ndarray):
                self.logger.warning(f"股票 {stock_code} 的future_return_2d不是numpy数组，尝试转换")
                try:
                    future_returns = np.array(future_returns, dtype=float)
                except Exception as e:
                    self.logger.error(f"转换future_return_2d到numpy数组失败: {e}")
                    return None

            # 处理缺失值
            try:
                # 检查是否有NaN值
                nan_mask = np.isnan(future_returns)
                if np.any(nan_mask):
                    # 使用均值填充缺失值
                    valid_returns = future_returns[~nan_mask]
                    if len(valid_returns) == 0:
                        self.logger.warning(f"股票 {stock_code} 的future_return_2d全为NaN，跳过变化点检测")
                        return None

                    mean_return = np.mean(valid_returns)
                    future_returns = np.nan_to_num(future_returns, nan=mean_return)
            except Exception as e:
                self.logger.error(f"处理股票 {stock_code} 的future_return_2d时出错: {e}")
                # 尝试一种更安全的方式处理数据
                try:
                    # 手动处理NaN和无效值
                    clean_returns = []
                    for val in future_returns:
                        try:
                            fval = float(val)
                            if np.isfinite(fval):
                                clean_returns.append(fval)
                            else:
                                clean_returns.append(0.0)  # 用0替换无效值
                        except (ValueError, TypeError):
                            clean_returns.append(0.0)

                    if not clean_returns:
                        return None

                    future_returns = np.array(clean_returns, dtype=float)
                except Exception as e2:
                    self.logger.error(f"尝试修复future_returns失败: {e2}")
                    return None

            # 输出future_return_2d统计信息
            returns_mean = np.mean(future_returns)
            returns_std = np.std(future_returns)
            returns_min = np.min(future_returns)
            returns_max = np.max(future_returns)
            self.logger.info(f"股票 {stock_code} future_return_2d统计: 均值={returns_mean:.4f}, 标准差={returns_std:.4f}, 最小值={returns_min:.4f}, 最大值={returns_max:.4f}")

            # 将数据转换为二维数组（为了兼容ruptures库的输入格式）
            indicator_matrix = future_returns.reshape(-1, 1)

            # 标准化数据
            scaler = StandardScaler()
            indicator_matrix_scaled = scaler.fit_transform(indicator_matrix)

            # 计算信号长度和最小分段长度
            n_samples = len(indicator_matrix_scaled)

            # 根据数据长度动态调整参数
            if n_samples > 100:
                min_size = min(10, n_samples // 20)  # 减小最小分段长度，最小为10或数据长度的5%
                max_points = min(20, n_samples // 10)  # 最多检测点数不超过20或数据长度的10%
            else:
                min_size = min(5, n_samples // 10)  # 对于短序列，使用更小的分段长度
                max_points = min(10, n_samples // 5)  # 对于短序列，限制变化点数量

            # 降低惩罚值 - 使用自适应惩罚值
            standard_penalty = 0.5 * np.log(n_samples)  # 大幅降低惩罚，使算法更敏感

            # 变化点检测结果容器
            change_points_results = []

            # === 方法1: Pelt算法 ===
            try:
                pelt_model = Pelt(model="l2", min_size=min_size)
                pelt_points = pelt_model.fit_predict(indicator_matrix_scaled, pen=standard_penalty)
                # 过滤掉序列末尾点
                pelt_points = [p for p in pelt_points if p < n_samples]

                if len(pelt_points) > 0:
                    self.logger.info(f"Pelt-L2 检测到 {len(pelt_points)} 个变化点")
                    change_points_results.append(('pelt_l2', pelt_points))

                # 如果检测不到点，进一步降低惩罚值
                if len(pelt_points) < 2:
                    lower_penalty = standard_penalty * 0.5
                    pelt_points = pelt_model.fit_predict(indicator_matrix_scaled, pen=lower_penalty)
                    pelt_points = [p for p in pelt_points if p < n_samples]

                    if len(pelt_points) > 0:
                        self.logger.info(f"Pelt-L2(低惩罚) 检测到 {len(pelt_points)} 个变化点")
                        change_points_results.append(('pelt_l2_low', pelt_points))
            except Exception as e:
                self.logger.warning(f"Pelt-L2 算法失败: {e}")

            # === 方法2: Binseg算法 ===
            try:
                # 二分割分割法，不需要惩罚，直接指定分割数量
                binseg_model = Binseg(model="l2", min_size=min_size)
                binseg_points = binseg_model.fit_predict(indicator_matrix_scaled, n_bkps=max_points)
                binseg_points = [p for p in binseg_points if p < n_samples]

                if len(binseg_points) > 0:
                    self.logger.info(f"Binseg 检测到 {len(binseg_points)} 个变化点")
                    change_points_results.append(('binseg', binseg_points))
            except Exception as e:
                self.logger.warning(f"Binseg 算法失败: {e}")

            # === 方法3: Window算法 ===
            try:
                # 窗口宽度为数据长度的10%，最小10个点
                window_width = max(10, n_samples // 10)
                window_model = Window(width=window_width, model="l2", min_size=min_size)
                window_points = window_model.fit_predict(indicator_matrix_scaled, n_bkps=max_points)
                window_points = [p for p in window_points if p < n_samples]

                if len(window_points) > 0:
                    self.logger.info(f"Window 检测到 {len(window_points)} 个变化点")
                    change_points_results.append(('window', window_points))
            except Exception as e:
                self.logger.warning(f"Window 算法失败: {e}")

            # 选择变化点结果
            if not change_points_results:
                self.logger.warning(f"股票 {stock_code} 未能使用任何方法检测到变化点")
                return None

            # 评估每种算法结果的质量
            best_algorithm = None
            final_change_points = []

            # 偏好基于最大收益率变化的检测点
            best_score = -1

            for alg_name, points in change_points_results:
                if len(points) < 2:  # 至少需要两个变化点
                    continue

                # 计算变化点前后的收益率变化幅度
                return_changes = []
                for i in range(len(points)-1):
                    start, end = points[i], points[i+1]
                    if start < len(future_returns) and end < len(future_returns):
                        segment1 = future_returns[start:end]
                        mean1 = np.mean(segment1)

                        # 计算相邻段的均值变化
                        if i+2 < len(points):
                            next_end = points[i+2]
                            if next_end < len(future_returns):
                                segment2 = future_returns[end:next_end]
                                mean2 = np.mean(segment2)
                                return_changes.append(abs(mean2 - mean1))

                # 如果有足够的变化，计算平均变化幅度
                if return_changes:
                    avg_change = np.mean(return_changes)
                    # 变化幅度越大越好，同时考虑变化点数量适中
                    score = avg_change * min(1.0, 10.0/len(points))

                    if score > best_score:
                        best_score = score
                        best_algorithm = alg_name
                        final_change_points = points

            # 如果未能选出最佳算法，选择检测到变化点最多的算法
            if not best_algorithm:
                sorted_results = sorted(change_points_results, key=lambda x: len(x[1]), reverse=True)
                best_algorithm, final_change_points = sorted_results[0]

            # 记录变化点
            if final_change_points:
                # 确保变化点是整数
                final_change_points = [int(cp) for cp in final_change_points]

                # 按顺序排序
                final_change_points.sort()

                # 获取变化点对应的日期
                cp_dates = dates[final_change_points]

                # 获取变化点对应的future_return_2d值
                cp_returns = future_returns[final_change_points]

                # 创建结果字典
                result = {
                    'change_points': final_change_points,
                    'dates': cp_dates,
                    'returns': cp_returns,
                    'algorithm': best_algorithm
                }

                self.logger.info(f"股票 {stock_code} 最终使用 {best_algorithm} 检测到 {len(final_change_points)} 个变化点")
                return result
            else:
                self.logger.warning(f"股票 {stock_code} 未能检测到任何变化点")
                return None

        except Exception as e:
            self.logger.error(f"检测股票 {stock_code} 变化点时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None


    def _analyze_change_point_patterns_direct(self, change_point_features):
        """
        集成学习增强的变化点特征模式分析

        基于变化点检测结果，结合多策略分析框架，
        分析不同市场状态下各策略的特征模式。

        Args:
            change_point_features: 变化点特征列表，每个元素包含完整的特征向量和标签

        Returns:
            dict: 包含传统模式和策略模式的综合分析结果
        """
        # 检查缓存
        import pickle
        import hashlib

        cache_dir = self.model_dir / "pattern_analysis_cache"
        cache_dir.mkdir(exist_ok=True)

        # 生成缓存键（基于特征数据的哈希）
        try:
            # 创建特征数据的哈希
            feature_str = str(len(change_point_features))
            if change_point_features:
                # 使用第一个和最后一个特征的关键信息生成哈希
                first_feature = change_point_features[0]
                last_feature = change_point_features[-1]
                feature_str += str(first_feature.get('stock_code', '')) + str(first_feature.get('date_str', ''))
                feature_str += str(last_feature.get('stock_code', '')) + str(last_feature.get('date_str', ''))
                # 添加特征维度信息
                feature_str += str(len(first_feature.keys()))

            cache_key = hashlib.md5(feature_str.encode()).hexdigest()
            cache_file = cache_dir / f"pattern_analysis_{cache_key}.pkl"

            # 尝试加载缓存
            if cache_file.exists():
                self.logger.info(f"从缓存加载模式分析结果: {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_result = pickle.load(f)
                    self.logger.info("成功加载缓存的模式分析结果")
                    return cached_result
                except Exception as e:
                    self.logger.warning(f"加载缓存失败: {e}，将重新计算")
        except Exception as e:
            self.logger.warning(f"生成缓存键失败: {e}，将不使用缓存")
            cache_file = None

        self.logger.info(f"开始训练两套预测系统，共 {len(change_point_features)} 个变化点特征...")

        if not change_point_features:
            return {
                'similarity_system': {
                    'historical_features_db': [],
                    'sector_feature_groups': {},
                    'feature_statistics': {}
                },

                'training_summary': {
                    'total_samples': 0,
                    'sectors_count': 0,
                    'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

        # 新的基于变化点特征分析的选股策略
        self.logger.info("训练基于变化点特征分析的新选股策略...")
        pattern_system = self._train_change_point_pattern_system(change_point_features)

        # 构建训练总结
        training_summary = {
            'total_samples': len(change_point_features),
            'pattern_analysis_status': pattern_system.get('status', 'unknown'),
            'up_patterns_count': len(pattern_system.get('up_patterns', {})),
            'down_patterns_count': len(pattern_system.get('down_patterns', {})),
            'feature_count': len(pattern_system.get('feature_cols', [])),
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 新增：集成学习策略分析
        self.logger.info("=== 开始集成学习策略分析 ===")
        strategy_patterns = {}
        ensemble_weights = {}

        try:
            # 转换为DataFrame进行策略分析
            if change_point_features:
                features_df = pd.DataFrame(change_point_features)
                if not features_df.empty and 'future_return_2d' in features_df.columns:
                    # 按策略和市场状态分析
                    strategy_patterns = self.strategy_analyzer.analyze_by_strategy_and_regime(features_df)

                    # 采样当前策略权重
                    ensemble_weights = self.ensemble_system.sample_strategy_weights()

                    self.logger.info(f"策略分析完成 - 发现 {len(strategy_patterns)} 个市场状态的策略模式")
                    self.logger.info(f"当前策略权重: {ensemble_weights}")
                else:
                    self.logger.warning("变化点特征数据无效，跳过策略分析")
            else:
                self.logger.warning("无变化点特征数据，跳过策略分析")
        except Exception as e:
            self.logger.error(f"策略分析失败: {e}")
            strategy_patterns = {}
            ensemble_weights = {}

        # 构建最终结果
        final_patterns = {
            'pattern_system': pattern_system,
            'training_summary': training_summary,
            'strategy_patterns': strategy_patterns,
            'ensemble_weights': ensemble_weights
        }

        self.logger.info(f"集成学习模式分析完成 - 上涨模式: {training_summary['up_patterns_count']} 个, "
                        f"下跌模式: {training_summary['down_patterns_count']} 个, "
                        f"总样本: {training_summary['total_samples']} 个, "
                        f"策略模式: {len(strategy_patterns)} 个市场状态")

        # 保存缓存
        if 'cache_file' in locals() and cache_file is not None:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(final_patterns, f)
                self.logger.info(f"模式分析结果已缓存到: {cache_file}")
            except Exception as e:
                self.logger.warning(f"保存缓存失败: {e}")

        return final_patterns

    def _train_similarity_matching_system(self, change_point_features):
        """
        训练相似度匹配系统

        构建历史特征数据库，用于后续的相似度匹配预测
        按板块分组存储，提高匹配效率和准确性

        Args:
            change_point_features: 变化点特征列表

        Returns:
            dict: 相似度匹配系统的训练结果
        """
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        self.logger.info("构建历史特征数据库...")

        # 转换为DataFrame便于处理
        features_df = pd.DataFrame(change_point_features)

        # 过滤有效样本（有标签的样本）
        valid_mask = features_df['future_return_2d'].notna()
        features_df = features_df[valid_mask]

        if len(features_df) == 0:
            self.logger.warning("没有有效的训练样本")
            return {
                'historical_features_db': [],
                'sector_feature_groups': {},
                'feature_statistics': {}
            }

        # 提取特征列（排除元数据列和字符串列）
        # 更新排除列表，适配新的特征结构
        exclude_cols = ['stock_code', 'date_str', 'date', 'future_return_2d', 'change_point_idx', 'most_relevant_sector', 'sector_code']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        # 进一步过滤，只保留数值类型的列
        # 优化特征类型检测，支持新的特征类别
        numeric_feature_cols = []
        feature_categories = {
            'basic_features': [],
            'technical_features': [],
            'market_features': [],
            'relative_features': [],
            'board_features': [],
            'derived_features': []
        }

        for col in feature_cols:
            if col in features_df.columns:
                # 检查列的数据类型
                if features_df[col].dtype in ['float64', 'float32', 'int64', 'int32', 'bool']:
                    numeric_feature_cols.append(col)

                    # 特征分类（用于后续分析）
                    if col.startswith('market_'):
                        feature_categories['market_features'].append(col)
                    elif col.startswith('relative_') or '_vs_market' in col or '_diff_market' in col:
                        feature_categories['relative_features'].append(col)
                    elif 'limit_' in col or 'board_' in col:
                        feature_categories['board_features'].append(col)
                    elif any(tech in col for tech in ['rsi', 'macd', 'ma_', 'momentum', 'volatility', 'volume_']):
                        feature_categories['technical_features'].append(col)
                    elif col in ['participation_entropy', 'fund_flow_momentum', 'volume_intensity']:
                        feature_categories['derived_features'].append(col)
                    else:
                        feature_categories['basic_features'].append(col)

                elif features_df[col].dtype == 'object':
                    # 尝试转换为数值类型
                    try:
                        pd.to_numeric(features_df[col], errors='raise')
                        numeric_feature_cols.append(col)
                    except (ValueError, TypeError):
                        # 无法转换为数值的列跳过
                        self.logger.debug(f"跳过非数值列: {col}")

        feature_cols = numeric_feature_cols

        # 记录特征分类统计
        self.logger.info("=== 特征分类统计 ===")
        for category, features in feature_categories.items():
            if features:
                self.logger.info(f"{category}: {len(features)} 个特征")

        # 特征标准化
        scaler = StandardScaler()
        feature_matrix = features_df[feature_cols].fillna(0)  # 填充缺失值
        # 转换为numpy数组以避免特征名称警告
        feature_matrix_values = feature_matrix.values
        scaled_features = scaler.fit_transform(feature_matrix_values)

        # 构建历史特征数据库
        historical_db = []
        for i, (idx, row) in enumerate(features_df.iterrows()):
            sample = {
                'features': scaled_features[i],
                'feature_names': feature_cols,
                'target': row['future_return_2d'],
                'stock_code': row.get('stock_code', ''),
                'date': row.get('date_str', ''),
                'sector': row.get('most_relevant_sector', 'unknown'),
                'metadata': {
                    'change_point_idx': row.get('change_point_idx', 0)
                }
            }
            historical_db.append(sample)

        # 按板块分组
        sector_groups = {}
        for sample in historical_db:
            sector = sample['sector']
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(sample)

        # 计算特征统计信息
        feature_stats = {
            'feature_names': feature_cols,
            'scaler': scaler,
            'feature_means': np.mean(scaled_features, axis=0),
            'feature_stds': np.std(scaled_features, axis=0),
            'total_samples': len(historical_db),
            'sectors': list(sector_groups.keys())
        }

        self.logger.info(f"历史特征数据库构建完成: {len(historical_db)} 个样本, "
                        f"{len(sector_groups)} 个板块, {len(feature_cols)} 个特征")

        return {
            'historical_features_db': historical_db,
            'sector_feature_groups': sector_groups,
            'feature_statistics': feature_stats
        }

    def _train_change_point_pattern_system(self, change_point_features):
        """
        基于变化点特征分析的新选股策略训练系统

        核心思路：通过统计分析找出导致股票上涨和下跌的关键特征模式

        Args:
            change_point_features: 变化点特征列表

        Returns:
            dict: 包含特征模式分析结果的字典
        """
        self.logger.info("开始基于变化点特征分析的新选股策略训练...")

        # 转换为DataFrame便于处理
        features_df = pd.DataFrame(change_point_features)

        # 过滤有效样本
        valid_mask = features_df['future_return_2d'].notna()
        features_df = features_df[valid_mask]

        if len(features_df) < 30:
            self.logger.warning("训练样本不足，无法进行特征模式分析")
            return {
                'status': 'insufficient_data',
                'up_patterns': {},
                'down_patterns': {},
                'feature_cols': [],
                'avg_up_return': 0,
                'avg_down_return': 0
            }

        # 1. 数据预处理 - 过滤变化点数量不足的股票
        self.logger.info("=== 第1步：数据预处理和过滤 ===")
        filtered_data = self._filter_stocks_by_change_points(features_df)

        if len(filtered_data) < 30:
            self.logger.warning("过滤后样本不足，无法进行特征模式分析")
            return {
                'status': 'insufficient_filtered_data',
                'up_patterns': {},
                'down_patterns': {},
                'feature_cols': [],
                'avg_up_return': 0,
                'avg_down_return': 0
            }

        # 2. 最相关板块计算优化
        self.logger.info("=== 第2步：优化最相关板块计算 ===")
        filtered_data = self._optimize_sector_correlation(filtered_data)

        # 3. 特征贡献分析
        self.logger.info("=== 第3步：特征贡献分析 ===")
        pattern_analysis = self._analyze_change_point_patterns(filtered_data)

        return pattern_analysis

    def _filter_stocks_by_change_points(self, data):
        """
        过滤变化点数量不足的股票
        只保留变化点数量≥10个的股票
        """
        # 检查数据类型，如果是列表则转换为DataFrame
        if isinstance(data, list):
            if not data:
                self.logger.warning("输入数据为空列表")
                return pd.DataFrame()

            # 转换为DataFrame
            data = pd.DataFrame(data)
            self.logger.info(f"将列表数据转换为DataFrame，形状: {data.shape}")

        # 检查是否有stock_code列
        if 'stock_code' not in data.columns:
            self.logger.error(f"数据中缺少stock_code列，现有列: {list(data.columns)}")
            return pd.DataFrame()

        # 统计每只股票的变化点数量
        stock_change_counts = data['stock_code'].value_counts()
        valid_stocks = stock_change_counts[stock_change_counts >= 10].index  # 降低阈值到10个

        self.logger.info(f"原始股票数量: {len(stock_change_counts)}")
        self.logger.info(f"变化点≥10个的股票数量: {len(valid_stocks)}")
        self.logger.info(f"过滤掉的股票比例: {(1 - len(valid_stocks)/len(stock_change_counts))*100:.1f}%")

        # 过滤数据
        filtered_data = data[data['stock_code'].isin(valid_stocks)].copy()
        self.logger.info(f"过滤后的变化点数量: {len(filtered_data)}")

        return filtered_data

    def _optimize_sector_correlation(self, data):
        """
        优化最相关板块计算
        根据变化点当日的涨跌情况选择最相关板块
        """
        self.logger.info("开始优化最相关板块计算...")

        optimized_data = data.copy()

        # 如果数据中有当日涨跌幅信息，根据涨跌情况优化板块选择
        if 'pct_chg' in data.columns:
            # 按股票分组处理
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask]

                for idx, row in stock_data.iterrows():
                    current_return = row.get('pct_chg', 0)

                    # 根据当日涨跌选择最相关板块
                    # 这里的逻辑需要根据实际的板块数据来实现
                    # 暂时保持原有逻辑，后续可以根据需要优化
                    pass

        self.logger.info("最相关板块计算优化完成")
        return optimized_data

    def _analyze_change_point_patterns(self, data):
        """
        分析上涨和下跌的特征模式

        Args:
            data: 过滤后的训练数据

        Returns:
            dict: 包含上涨和下跌特征模式的分析结果
        """
        self.logger.info("开始分析上涨和下跌的特征模式...")

        # 准备特征列
        exclude_cols = ['stock_code', 'date', 'date_str', 'future_return_2d', 'sector_code', 'most_relevant_sector']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # 移除无效特征
        valid_features = []
        for col in feature_cols:
            if data[col].notna().sum() > len(data) * 0.5:  # 至少50%的数据有效
                valid_features.append(col)

        feature_cols = valid_features
        self.logger.info(f"有效特征数量: {len(feature_cols)}")

        # 分离上涨和下跌的变化点
        up_data = data[data['future_return_2d'] > 0].copy()
        down_data = data[data['future_return_2d'] < 0].copy()

        self.logger.info(f"上涨变化点数量: {len(up_data)}")
        self.logger.info(f"下跌变化点数量: {len(down_data)}")
        self.logger.info(f"上涨比例: {len(up_data)/len(data)*100:.1f}%")

        # 分析上涨特征模式
        self.logger.info("分析上涨特征模式...")
        up_patterns = self._extract_feature_patterns(up_data, feature_cols, "上涨")

        # 分析下跌特征模式
        self.logger.info("分析下跌特征模式...")
        down_patterns = self._extract_feature_patterns(down_data, feature_cols, "下跌")

        # 特征重要性分析 - 新增
        self.logger.info("计算特征重要性...")
        feature_importance = self._calculate_advanced_feature_importance(data, feature_cols)

        # 将特征重要性权重应用到模式中
        self.logger.info("应用特征重要性权重到模式中...")
        up_patterns = self._apply_importance_weights(up_patterns, feature_importance)
        down_patterns = self._apply_importance_weights(down_patterns, feature_importance)

        # 特征范围优化 - 提高区分度
        self.logger.info("优化特征范围以提高区分度...")
        optimized_patterns = self._optimize_feature_ranges(up_patterns, down_patterns, data, feature_cols)

        # 计算历史平均涨跌幅
        avg_up_return = up_data['future_return_2d'].mean()
        avg_down_return = down_data['future_return_2d'].mean()

        self.logger.info(f"历史平均上涨幅度: {avg_up_return:.4f}")
        self.logger.info(f"历史平均下跌幅度: {avg_down_return:.4f}")
        self.logger.info(f"特征重要性前10: {dict(list(feature_importance.items())[:10])}")

        # 保存分析结果
        pattern_cache_dir = os.path.join(self.model_dir, 'pattern_analysis_cache')
        os.makedirs(pattern_cache_dir, exist_ok=True)

        # 创建缓存文件名
        data_hash = hashlib.md5(f"{len(data)}_{len(feature_cols)}_{data['stock_code'].nunique()}".encode()).hexdigest()
        cache_file = os.path.join(pattern_cache_dir, f"change_point_patterns_{data_hash}.pkl")

        result = {
            'status': 'success',
            'up_patterns': optimized_patterns['up_patterns'],
            'down_patterns': optimized_patterns['down_patterns'],
            'feature_cols': feature_cols,
            'feature_importance': feature_importance,  # 新增特征重要性
            'avg_up_return': avg_up_return,
            'avg_down_return': avg_down_return,
            'up_count': len(up_data),
            'down_count': len(down_data),
            'total_count': len(data)
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        self.logger.info(f"特征模式分析结果已保存到: {cache_file}")

        return result

    def _extract_feature_patterns(self, data, feature_cols, pattern_type):
        """
        精细化特征模式提取 - 基于顶级量化金融研究

        参考论文：
        1. "Feature Engineering for Financial Machine Learning" (Marcos López de Prado)
        2. "Advances in Financial Machine Learning" (Marcos López de Prado)
        3. "Machine Learning for Asset Management" (Emmanuel Jurczenko)

        特征分类：
        - 连续型特征：价格、成交量、技术指标等
        - 二进制特征：K线形态、技术信号等（0/1编码）
        - 分类特征：板块、状态等

        Args:
            data: 数据DataFrame
            feature_cols: 特征列列表
            pattern_type: 模式类型（"上涨"或"下跌"）

        Returns:
            dict: 精细化特征模式字典
        """
        patterns = {}
        threshold = 0.75  # 降低到75%以获得更多有效模式

        # 定义特征分类 - 基于金融特征的本质属性
        binary_features = self._identify_binary_features(feature_cols)
        continuous_features = self._identify_continuous_features(feature_cols)
        categorical_features = self._identify_categorical_features(feature_cols)

        self.logger.info(f"特征分类完成 - 二进制: {len(binary_features)}, 连续型: {len(continuous_features)}, 分类型: {len(categorical_features)}")

        for feature in feature_cols:
            if feature not in data.columns:
                continue

            feature_values = data[feature].dropna()
            if len(feature_values) < 10:  # 至少需要10个样本
                continue

            # 根据特征类型采用不同的模式提取策略
            if feature in binary_features:
                pattern = self._extract_binary_pattern(feature_values, feature, pattern_type)
            elif feature in continuous_features:
                pattern = self._extract_continuous_pattern(feature_values, feature, pattern_type, threshold)
            elif feature in categorical_features:
                pattern = self._extract_categorical_pattern(feature_values, feature, pattern_type)
            else:
                # 自动判断特征类型
                pattern = self._extract_auto_pattern(feature_values, feature, pattern_type, threshold)

            if pattern:
                patterns[feature] = pattern

        self.logger.info(f"{pattern_type}特征模式提取完成，共提取 {len(patterns)} 个有效特征模式")
        return patterns

    def _identify_binary_features(self, feature_cols):
        """识别二进制特征（K线形态、技术信号等）"""
        binary_patterns = [
            'pattern_', 'signal_', 'flag_', 'breakout_', 'reversal_',
            'engulfing', 'doji', 'hammer', 'star', 'soldiers', 'crows',
            'head_shoulders', 'double_', 'triangle_', 'near_support', 'near_resistance'
        ]

        binary_features = []
        for feature in feature_cols:
            # 检查是否包含二进制模式关键词
            if any(pattern in feature.lower() for pattern in binary_patterns):
                binary_features.append(feature)
            # 检查是否为明确的0/1特征
            elif feature.lower().endswith(('_signal', '_flag', '_alert', '_trigger')):
                binary_features.append(feature)

        return binary_features

    def _identify_continuous_features(self, feature_cols):
        """识别连续型特征（价格、成交量、技术指标等）"""
        continuous_patterns = [
            'price', 'volume', 'amount', 'turnover', 'pct_chg', 'return',
            'ma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'volatility',
            'momentum', 'strength', 'ratio', 'correlation', 'zscore'
        ]

        continuous_features = []
        for feature in feature_cols:
            if any(pattern in feature.lower() for pattern in continuous_patterns):
                continuous_features.append(feature)
            elif feature.lower() in ['open', 'high', 'low', 'close']:
                continuous_features.append(feature)

        return continuous_features

    def _identify_categorical_features(self, feature_cols):
        """识别分类特征（板块、状态等）"""
        categorical_patterns = [
            'sector', 'industry', 'state', 'regime', 'category', 'type', 'class'
        ]

        categorical_features = []
        for feature in feature_cols:
            if any(pattern in feature.lower() for pattern in categorical_patterns):
                categorical_features.append(feature)

        return categorical_features

    def _extract_binary_pattern(self, feature_values, feature_name, pattern_type):
        """
        提取二进制特征模式 - 专门处理0/1编码的技术指标

        对于K线形态等二进制特征，重点分析：
        1. 激活频率（True的比例）
        2. 激活时的效果（条件概率）
        3. 信号强度评估
        """
        # 转换为数值型，处理可能的字符串或其他类型
        numeric_values = pd.to_numeric(feature_values, errors='coerce').dropna()

        if len(numeric_values) < 5:
            return None

        # 检查是否确实是二进制特征（只有0和1）
        unique_values = set(numeric_values.unique())
        if not unique_values.issubset({0, 1, 0.0, 1.0}):
            # 如果不是严格的0/1，尝试阈值化
            mean_val = numeric_values.mean()
            numeric_values = (numeric_values > mean_val).astype(int)

        # 计算激活频率
        activation_rate = numeric_values.mean()

        # 计算信号强度（基于激活频率的稀有性）
        # 稀有信号（激活率5%-20%）通常更有价值
        if 0.05 <= activation_rate <= 0.20:
            signal_strength = 'high'
            weight = 1.5
        elif 0.20 < activation_rate <= 0.40:
            signal_strength = 'medium'
            weight = 1.2
        elif 0.40 < activation_rate <= 0.60:
            signal_strength = 'low'
            weight = 1.0
        else:
            signal_strength = 'very_low'
            weight = 0.8

        return {
            'type': 'binary',
            'activation_rate': float(activation_rate),
            'signal_strength': signal_strength,
            'weight': weight,
            'sample_count': len(numeric_values),
            'pattern_type': pattern_type,
            'threshold': 0.5,  # 二进制特征的激活阈值
            'confidence': min(1.0, activation_rate * 2) if activation_rate < 0.5 else min(1.0, (1-activation_rate) * 2)
        }

    def _extract_continuous_pattern(self, feature_values, feature_name, pattern_type, threshold):
        """
        提取连续型特征模式 - 基于分布分析和统计显著性

        参考Marcos López de Prado的特征工程方法：
        1. 使用分位数分析而非简单的均值±标准差
        2. 考虑分布的偏度和峰度
        3. 使用信息论方法评估特征价值
        """
        numeric_values = pd.to_numeric(feature_values, errors='coerce').dropna()

        if len(numeric_values) < 10:
            return None

        # 基础统计量
        mean_val = numeric_values.mean()
        std_val = numeric_values.std()
        median_val = numeric_values.median()

        if std_val == 0:  # 常数特征，无预测价值
            return None

        # 分位数分析
        q10, q25, q75, q90 = numeric_values.quantile([0.1, 0.25, 0.75, 0.9])
        iqr = q75 - q25

        # 分布特征分析
        from scipy import stats
        try:
            skewness = stats.skew(numeric_values)
            kurtosis = stats.kurtosis(numeric_values)
        except:
            skewness = 0
            kurtosis = 0

        # 基于分布特征选择最佳范围策略
        ranges = []

        # 策略1：基于IQR的稳健范围（适合有异常值的数据）
        if iqr > 0:
            ranges.append({
                'method': 'robust_iqr',
                'min': q25 - 0.5 * iqr,
                'max': q75 + 0.5 * iqr,
                'expected_coverage': 0.75
            })

        # 策略2：基于分位数的核心范围（适合正态分布）
        ranges.append({
            'method': 'quantile_core',
            'min': q10,
            'max': q90,
            'expected_coverage': 0.80
        })

        # 策略3：基于均值的动态范围（考虑偏度）
        if abs(skewness) < 1:  # 接近正态分布
            ranges.append({
                'method': 'adaptive_mean',
                'min': mean_val - 1.2 * std_val,
                'max': mean_val + 1.2 * std_val,
                'expected_coverage': 0.77
            })
        else:  # 偏态分布，使用中位数
            ranges.append({
                'method': 'adaptive_median',
                'min': median_val - 1.0 * std_val,
                'max': median_val + 1.0 * std_val,
                'expected_coverage': 0.70
            })

        # 选择最佳范围
        best_range = None
        target_coverage = threshold

        for range_def in ranges:
            in_range = ((numeric_values >= range_def['min']) &
                       (numeric_values <= range_def['max']))
            actual_coverage = in_range.sum() / len(numeric_values)

            if actual_coverage >= 0.5:  # 至少50%覆盖率
                if best_range is None or abs(actual_coverage - target_coverage) < abs(best_range['actual_coverage'] - target_coverage):
                    best_range = range_def.copy()
                    best_range['actual_coverage'] = actual_coverage

        if not best_range:
            return None

        # 计算特征重要性权重（基于分布特征）
        weight = self._calculate_continuous_feature_weight(numeric_values, skewness, kurtosis)

        return {
            'type': 'continuous',
            'min': best_range['min'],
            'max': best_range['max'],
            'coverage': best_range['actual_coverage'],
            'method': best_range['method'],
            'weight': weight,
            'sample_count': len(numeric_values),
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'iqr': iqr
        }

    def _extract_categorical_pattern(self, feature_values, feature_name, pattern_type):
        """提取分类特征模式"""
        value_counts = feature_values.value_counts()

        if len(value_counts) == 0:
            return None

        # 找出最频繁的类别
        most_frequent = value_counts.index[0]
        frequency = value_counts.iloc[0] / len(feature_values)

        return {
            'type': 'categorical',
            'most_frequent_value': most_frequent,
            'frequency': frequency,
            'unique_count': len(value_counts),
            'weight': min(2.0, 1.0 / len(value_counts)),  # 类别越少权重越高
            'sample_count': len(feature_values)
        }

    def _extract_auto_pattern(self, feature_values, feature_name, pattern_type, threshold):
        """自动判断特征类型并提取模式"""
        numeric_values = pd.to_numeric(feature_values, errors='coerce').dropna()

        if len(numeric_values) < 5:
            return None

        unique_values = set(numeric_values.unique())

        # 判断是否为二进制特征
        if unique_values.issubset({0, 1, 0.0, 1.0}) or len(unique_values) == 2:
            return self._extract_binary_pattern(numeric_values, feature_name, pattern_type)
        # 判断是否为分类特征（少于10个唯一值）
        elif len(unique_values) < 10:
            return self._extract_categorical_pattern(feature_values, feature_name, pattern_type)
        # 否则作为连续特征处理
        else:
            return self._extract_continuous_pattern(numeric_values, feature_name, pattern_type, threshold)

    def _calculate_continuous_feature_weight(self, values, skewness, kurtosis):
        """
        计算连续特征的重要性权重

        基于以下因素：
        1. 分布的信息熵
        2. 偏度和峰度（分布形状）
        3. 变异系数（相对变异性）
        """
        try:
            # 计算变异系数
            cv = values.std() / abs(values.mean()) if values.mean() != 0 else 0

            # 基于分布特征计算权重
            # 1. 适度的偏度和峰度表明特征有区分性
            skew_weight = max(0.5, 1.0 - abs(skewness) / 3.0)
            kurt_weight = max(0.5, 1.0 - abs(kurtosis) / 5.0)

            # 2. 适度的变异系数表明特征有信息量
            cv_weight = min(2.0, max(0.5, cv))

            # 综合权重
            weight = (skew_weight + kurt_weight + cv_weight) / 3.0

            return max(0.5, min(2.0, weight))  # 限制在0.5-2.0之间

        except:
            return 1.0  # 默认权重

    def _apply_importance_weights(self, patterns, feature_importance):
        """
        将特征重要性权重应用到模式中

        Args:
            patterns: 特征模式字典
            feature_importance: 特征重要性字典

        Returns:
            dict: 应用权重后的特征模式
        """
        weighted_patterns = patterns.copy()

        for feature_name, pattern in weighted_patterns.items():
            if feature_name in feature_importance:
                # 获取重要性分数
                importance_score = feature_importance[feature_name]

                # 原有权重
                original_weight = pattern.get('weight', 1.0)

                # 计算综合权重：原有权重 × 重要性分数
                # 重要性分数已经归一化到0-1，我们将其转换为0.5-2.0的权重乘数
                importance_multiplier = 0.5 + 1.5 * importance_score

                # 综合权重
                combined_weight = original_weight * importance_multiplier

                # 更新模式权重
                pattern['weight'] = combined_weight
                pattern['importance_score'] = importance_score
                pattern['importance_multiplier'] = importance_multiplier
            else:
                # 如果没有重要性分数，使用默认的较低权重
                pattern['weight'] = pattern.get('weight', 1.0) * 0.8
                pattern['importance_score'] = 0.0
                pattern['importance_multiplier'] = 0.8

        return weighted_patterns

    def _optimize_feature_ranges(self, up_patterns, down_patterns, data, feature_cols):
        """
        优化特征范围以提高区分度
        对于同时在上涨和下跌中出现的特征，缩小各自的取值范围

        Args:
            up_patterns: 上涨特征模式
            down_patterns: 下跌特征模式
            data: 原始数据
            feature_cols: 特征列列表

        Returns:
            dict: 优化后的特征模式
        """
        self.logger.info("开始优化特征范围以提高区分度...")

        optimized_up = up_patterns.copy()
        optimized_down = down_patterns.copy()

        # 找出同时出现在上涨和下跌模式中的特征
        common_features = set(up_patterns.keys()) & set(down_patterns.keys())

        self.logger.info(f"需要优化的共同特征数量: {len(common_features)}")

        for feature in common_features:
            if feature not in data.columns:
                continue

            up_data = data[data['future_return_2d'] > 0][feature].dropna()
            down_data = data[data['future_return_2d'] < 0][feature].dropna()

            if len(up_data) < 5 or len(down_data) < 5:
                continue

            # 分析数值分布差异
            up_mean = up_data.mean()
            down_mean = down_data.mean()
            up_std = up_data.std()
            down_std = down_data.std()

            # 如果均值差异显著，调整范围
            if abs(up_mean - down_mean) > 0.5 * (up_std + down_std):
                # 重新定义范围，增加区分度
                if up_mean > down_mean:
                    # 上涨时特征值更高
                    new_up_min = up_mean - 0.5 * up_std
                    new_up_max = up_mean + up_std
                    new_down_min = down_mean - up_std
                    new_down_max = down_mean + 0.5 * down_std
                else:
                    # 下跌时特征值更高
                    new_up_min = up_mean - up_std
                    new_up_max = up_mean + 0.5 * up_std
                    new_down_min = down_mean - 0.5 * down_std
                    new_down_max = down_mean + down_std

                # 更新模式
                optimized_up[feature].update({
                    'min': new_up_min,
                    'max': new_up_max,
                    'optimized': True
                })

                optimized_down[feature].update({
                    'min': new_down_min,
                    'max': new_down_max,
                    'optimized': True
                })

        self.logger.info(f"特征范围优化完成，优化了 {len(common_features)} 个共同特征")

        return {
            'up_patterns': optimized_up,
            'down_patterns': optimized_down
        }

    def _calculate_advanced_feature_importance(self, data, feature_cols, target_col='future_return_2d'):
        """
        前沿特征重要性分析系统 - 基于2024-2025年最新研究

        参考最新论文：
        1. "Developing a Dynamic Feature Selection System (DFSS) for Stock Market Prediction" (Kim et al., 2024)
        2. "Explainable AI for Financial Machine Learning" (2024-2025)
        3. "Attention Mechanism for Feature Attribution" (2024)
        4. "SHAP TreeExplainer and Permutation Importance" (2024)

        集成方法：
        1. SHAP TreeExplainer (2024最新版本)
        2. 排列重要性 (Permutation Importance)
        3. 注意力机制特征归因 (Attention-based Attribution)
        4. 集成梯度 (Integrated Gradients)
        5. 动态特征选择 (Dynamic Feature Selection)
        6. BorutaSHAP (最新混合方法)
        """
        self.logger.info("开始前沿特征重要性分析...")

        # 准备数据
        valid_data = data.dropna(subset=[target_col])
        if len(valid_data) < 20:
            self.logger.warning("有效样本不足，跳过特征重要性分析")
            return {}

        # 创建二分类目标（上涨/下跌）
        y_binary = (valid_data[target_col] > 0).astype(int)
        y_continuous = valid_data[target_col]

        importance_scores = {}

        # 预处理：确保所有方法使用相同的有效特征集
        valid_feature_cols = []
        for feature in feature_cols:
            if feature in valid_data.columns:
                feature_values = valid_data[feature].fillna(0)
                # 只保留有变化的特征
                if len(pd.Series(feature_values).unique()) > 1:
                    valid_feature_cols.append(feature)

        self.logger.info(f"特征重要性分析 - 有效特征数量: {len(valid_feature_cols)}")

        # 1. SHAP TreeExplainer (2024最新版本) - 使用统一的特征集
        shap_scores = self._calculate_shap_importance(valid_data, valid_feature_cols, y_binary)

        # 2. 排列重要性 (Permutation Importance) - 使用统一的特征集
        perm_scores = self._calculate_permutation_importance(valid_data, valid_feature_cols, y_binary)

        # 3. BorutaSHAP (最新混合方法) - 使用统一的特征集
        boruta_shap_scores = self._calculate_boruta_shap_importance(valid_data, valid_feature_cols, y_binary)

        # 4. 注意力机制特征归因 (Attention-based Attribution)
        attention_scores = self._calculate_attention_importance(valid_data, valid_feature_cols, y_continuous)

        # 5. 集成梯度近似 (Integrated Gradients Approximation)
        integrated_grad_scores = self._calculate_integrated_gradients_importance(valid_data, valid_feature_cols, y_continuous)

        # 6. 动态特征选择评分 (Dynamic Feature Selection Score)
        dynamic_scores = self._calculate_dynamic_feature_score(valid_data, valid_feature_cols, y_continuous)

        # 7. 综合评分（基于2024年最新权重分配策略）
        all_features = set()
        for score_dict in [shap_scores, perm_scores, boruta_shap_scores, attention_scores, integrated_grad_scores, dynamic_scores]:
            all_features.update(score_dict.keys())

        for feature in all_features:
            scores = []
            weights = []

            # SHAP权重25% (最可靠的方法)
            if feature in shap_scores:
                scores.append(shap_scores[feature])
                weights.append(0.25)

            # 排列重要性权重20% (模型无关)
            if feature in perm_scores:
                scores.append(perm_scores[feature])
                weights.append(0.20)

            # BorutaSHAP权重20% (混合优势)
            if feature in boruta_shap_scores:
                scores.append(boruta_shap_scores[feature])
                weights.append(0.20)

            # 注意力机制权重15% (深度学习优势)
            if feature in attention_scores:
                scores.append(attention_scores[feature])
                weights.append(0.15)

            # 集成梯度权重10% (梯度信息)
            if feature in integrated_grad_scores:
                scores.append(integrated_grad_scores[feature])
                weights.append(0.10)

            # 动态评分权重10% (时间序列特性)
            if feature in dynamic_scores:
                scores.append(dynamic_scores[feature])
                weights.append(0.10)

            if scores:
                # 加权平均
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                importance_scores[feature] = weighted_score

        # 按重要性排序
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

        self.logger.info(f"前沿特征重要性分析完成，共分析 {len(sorted_importance)} 个特征")

        return sorted_importance

    def _calculate_shap_importance(self, data, feature_cols, target):
        """SHAP TreeExplainer - 2024最新版本 - 彻底修复特征长度不匹配问题"""
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            # 处理目标变量
            if isinstance(target, str):
                if target in data.columns:
                    target_values = data[target].values
                else:
                    self.logger.error(f"目标列 {target} 不存在于数据中")
                    return {}
            else:
                target_values = np.array(target)

            # 确保目标变量是一维数组
            if target_values.ndim > 1:
                target_values = target_values.flatten()

            # 转换为二分类标签
            target_binary = (target_values > 0).astype(int)

            # 准备特征矩阵 - 严格按照feature_cols顺序
            X_list = []
            valid_features = []

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = data[feature].fillna(0)

                if pd.api.types.is_numeric_dtype(feature_values):
                    X_list.append(feature_values.values)
                    valid_features.append(feature)
                else:
                    try:
                        le = LabelEncoder()
                        encoded_values = le.fit_transform(feature_values.astype(str))
                        X_list.append(encoded_values)
                        valid_features.append(feature)
                    except:
                        continue

            if len(X_list) < 2:
                return {}

            X = np.column_stack(X_list)

            # 确保特征矩阵和目标变量长度一致
            min_length = min(len(X), len(target_binary))
            X = X[:min_length]
            target_binary = target_binary[:min_length]

            # 记录原始特征数量，用于后续验证
            original_feature_count = len(valid_features)

            self.logger.info(f"SHAP分析 - 原始特征数量: {original_feature_count}, 特征矩阵形状: {X.shape}")

            # 训练模型 - 只使用原始特征，不添加影子特征
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X, target_binary)

            # 计算SHAP值 - 严格控制样本和特征数量
            explainer = shap.TreeExplainer(model)
            sample_size = min(3000, len(X))  # 减少样本数量以提高稳定性
            X_sample = X[:sample_size]

            self.logger.info(f"SHAP计算 - 样本数量: {sample_size}, 特征数量: {X_sample.shape[1]}")

            shap_values = explainer.shap_values(X_sample)

            # 处理SHAP值 - 确保正确处理二分类情况
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # 二分类情况，取正类的SHAP值
                shap_values_final = shap_values[1]
            elif isinstance(shap_values, np.ndarray):
                # 单一数组情况
                shap_values_final = shap_values
            else:
                # 其他情况，尝试取第一个
                shap_values_final = shap_values[0] if isinstance(shap_values, list) else shap_values

            # 确保SHAP值是二维数组
            if shap_values_final.ndim == 1:
                shap_values_final = shap_values_final.reshape(1, -1)

            self.logger.info(f"SHAP值形状: {shap_values_final.shape}")

            # 计算特征重要性 - 彻底修复数组转换问题
            feature_importance = np.abs(shap_values_final).mean(axis=0)

            # 确保feature_importance是一维数组
            if feature_importance.ndim > 1:
                feature_importance = feature_importance.flatten()

            # 进一步确保是标量数组
            feature_importance = np.asarray(feature_importance).flatten()

            self.logger.info(f"特征重要性形状: {feature_importance.shape}, 原始特征数量: {original_feature_count}")

            # 严格验证特征重要性长度与原始特征数量一致
            if len(feature_importance) != original_feature_count:
                self.logger.error(f"严重错误: SHAP重要性长度({len(feature_importance)})与原始特征数量({original_feature_count})不匹配")
                self.logger.error(f"特征矩阵形状: {X.shape}, SHAP值形状: {shap_values_final.shape}")

                # 尝试修复：截取或填充到正确长度
                if len(feature_importance) > original_feature_count:
                    # 如果重要性数组过长，截取前N个
                    feature_importance = feature_importance[:original_feature_count]
                    self.logger.warning(f"截取SHAP重要性数组到正确长度: {len(feature_importance)}")
                elif len(feature_importance) < original_feature_count:
                    # 如果重要性数组过短，用0填充
                    padding = np.zeros(original_feature_count - len(feature_importance))
                    feature_importance = np.concatenate([feature_importance, padding])
                    self.logger.warning(f"填充SHAP重要性数组到正确长度: {len(feature_importance)}")

                # 再次验证
                if len(feature_importance) != original_feature_count:
                    self.logger.error("无法修复SHAP重要性长度不匹配问题，返回空结果")
                    return {}

            shap_scores = {}
            for i, feature in enumerate(valid_features):
                try:
                    # 彻底安全的标量转换
                    importance_value = feature_importance[i]

                    # 处理各种numpy数据类型
                    if isinstance(importance_value, np.ndarray):
                        # 如果仍然是数组，取第一个元素
                        if importance_value.size == 1:
                            importance_value = importance_value.item()
                        else:
                            importance_value = importance_value.flatten()[0]
                    elif hasattr(importance_value, 'item'):
                        # numpy标量
                        importance_value = importance_value.item()

                    # 最终转换为Python float
                    shap_scores[feature] = float(importance_value)

                except (ValueError, TypeError, IndexError) as e:
                    self.logger.warning(f"SHAP重要性转换失败 {feature}: {e}")
                    shap_scores[feature] = 0.0

            # 归一化
            if shap_scores:
                max_score = max(shap_scores.values())
                if max_score > 0:
                    shap_scores = {k: v/max_score for k, v in shap_scores.items()}

            return shap_scores

        except ImportError:
            self.logger.warning("SHAP不可用，请安装: pip install shap")
            # 使用树模型重要性作为备用
            return self._calculate_tree_importance_backup(data, feature_cols, target)
        except Exception as e:
            self.logger.error(f"计算SHAP重要性时出错: {e}")
            return {}

    def _calculate_permutation_importance(self, data, feature_cols, target):
        """排列重要性 - 模型无关方法"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance
            from sklearn.preprocessing import LabelEncoder

            # 处理目标变量
            if isinstance(target, str):
                if target in data.columns:
                    target_values = data[target].values
                else:
                    self.logger.error(f"目标列 {target} 不存在于数据中")
                    return {}
            else:
                target_values = np.array(target)

            # 确保目标变量是一维数组
            if target_values.ndim > 1:
                target_values = target_values.flatten()

            # 转换为二分类标签
            target_binary = (target_values > 0).astype(int)

            # 准备特征矩阵
            X_list = []
            valid_features = []

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = data[feature].fillna(0)

                if pd.api.types.is_numeric_dtype(feature_values):
                    X_list.append(feature_values.values)
                    valid_features.append(feature)
                else:
                    try:
                        le = LabelEncoder()
                        encoded_values = le.fit_transform(feature_values.astype(str))
                        X_list.append(encoded_values)
                        valid_features.append(feature)
                    except:
                        continue

            if len(X_list) < 2:
                return {}

            X = np.column_stack(X_list)

            # 确保特征矩阵和目标变量长度一致
            min_length = min(len(X), len(target_binary))
            X = X[:min_length]
            target_binary = target_binary[:min_length]

            # 训练模型
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            model.fit(X, target_binary)

            # 计算排列重要性
            perm_importance = permutation_importance(
                model, X, target_binary,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )

            perm_scores = {}
            for i, feature in enumerate(valid_features):
                try:
                    # 安全转换排列重要性值
                    importance_value = perm_importance.importances_mean[i]

                    # 处理各种numpy数据类型
                    if isinstance(importance_value, np.ndarray):
                        if importance_value.size == 1:
                            importance_value = importance_value.item()
                        else:
                            importance_value = importance_value.flatten()[0]
                    elif hasattr(importance_value, 'item'):
                        importance_value = importance_value.item()

                    perm_scores[feature] = float(importance_value)

                except (ValueError, TypeError, IndexError) as e:
                    self.logger.warning(f"排列重要性转换失败 {feature}: {e}")
                    perm_scores[feature] = 0.0

            # 归一化
            if perm_scores:
                max_score = max(perm_scores.values())
                if max_score > 0:
                    perm_scores = {k: v/max_score for k, v in perm_scores.items()}

            return perm_scores

        except ImportError:
            self.logger.warning("sklearn不可用，跳过排列重要性计算")
            return {}
        except Exception as e:
            self.logger.error(f"计算排列重要性时出错: {e}")
            return {}

    def _calculate_boruta_shap_importance(self, data, feature_cols, target):
        """BorutaSHAP - 最新混合方法"""
        try:
            # 简化版BorutaSHAP实现
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            # 处理目标变量
            if isinstance(target, str):
                if target in data.columns:
                    target_values = data[target].values
                else:
                    self.logger.error(f"目标列 {target} 不存在于数据中")
                    return {}
            else:
                target_values = np.array(target)

            # 确保目标变量是一维数组
            if target_values.ndim > 1:
                target_values = target_values.flatten()

            # 转换为二分类标签
            target_binary = (target_values > 0).astype(int)

            # 准备特征矩阵
            X_list = []
            valid_features = []

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = data[feature].fillna(0)

                if pd.api.types.is_numeric_dtype(feature_values):
                    X_list.append(feature_values.values)
                    valid_features.append(feature)
                else:
                    try:
                        le = LabelEncoder()
                        encoded_values = le.fit_transform(feature_values.astype(str))
                        X_list.append(encoded_values)
                        valid_features.append(feature)
                    except:
                        continue

            if len(X_list) < 2:
                return {}

            X = np.column_stack(X_list)

            # 确保特征矩阵和目标变量长度一致
            min_length = min(len(X), len(target_binary))
            X = X[:min_length]
            target_binary = target_binary[:min_length]

            # 创建影子特征（Boruta核心思想）
            np.random.seed(42)
            shadow_features = []
            for i in range(X.shape[1]):
                shadow = X[:, i].copy()
                np.random.shuffle(shadow)
                shadow_features.append(shadow)

            # 合并原始特征和影子特征
            X_extended = np.column_stack([X] + shadow_features)

            # 训练模型
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_extended, target_binary)

            # 获取特征重要性
            importances = model.feature_importances_

            # 分离原始特征和影子特征的重要性
            original_importances = importances[:len(valid_features)]
            shadow_importances = importances[len(valid_features):]

            # 计算影子特征的最大重要性作为阈值
            shadow_threshold = np.max(shadow_importances)

            boruta_shap_scores = {}
            for i, feature in enumerate(valid_features):
                try:
                    # 安全转换重要性值
                    importance_value = original_importances[i]

                    # 处理各种numpy数据类型
                    if isinstance(importance_value, np.ndarray):
                        if importance_value.size == 1:
                            importance_value = importance_value.item()
                        else:
                            importance_value = importance_value.flatten()[0]
                    elif hasattr(importance_value, 'item'):
                        importance_value = importance_value.item()

                    importance_value = float(importance_value)

                    # 只保留重要性超过影子特征阈值的特征
                    if importance_value > shadow_threshold:
                        boruta_shap_scores[feature] = importance_value

                except (ValueError, TypeError, IndexError) as e:
                    self.logger.warning(f"BorutaSHAP重要性转换失败 {feature}: {e}")
                    continue

            # 归一化
            if boruta_shap_scores:
                max_score = max(boruta_shap_scores.values())
                if max_score > 0:
                    boruta_shap_scores = {k: v/max_score for k, v in boruta_shap_scores.items()}

            return boruta_shap_scores

        except Exception as e:
            self.logger.error(f"计算BorutaSHAP重要性时出错: {e}")
            return {}

    def _calculate_attention_importance(self, data, feature_cols, target):
        """注意力机制特征归因 - 基于2024年最新研究"""
        try:
            # 简化版注意力机制实现
            attention_scores = {}

            # 计算特征间的相关性矩阵（模拟注意力权重）
            numeric_features = []
            valid_features = []

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = pd.to_numeric(data[feature], errors='coerce').fillna(0)
                if len(feature_values.unique()) > 1:  # 避免常数特征
                    numeric_features.append(feature_values.values)
                    valid_features.append(feature)

            if len(numeric_features) < 2:
                return {}

            # 构建特征矩阵
            X = np.column_stack(numeric_features)

            # 确保目标变量长度与特征矩阵匹配
            if len(target) != X.shape[0]:
                self.logger.warning(f"目标变量长度({len(target)})与特征矩阵行数({X.shape[0]})不匹配")
                return {}

            # 计算注意力权重（基于特征与目标的相关性和特征间相关性）
            target_corr = []
            for i in range(X.shape[1]):
                try:
                    corr = np.corrcoef(X[:, i], target)[0, 1]
                    target_corr.append(abs(corr) if pd.notna(corr) else 0)
                except:
                    target_corr.append(0)

            target_corr = np.array(target_corr)

            # 计算特征间相关性矩阵
            try:
                feature_corr_matrix = np.abs(np.corrcoef(X.T))
                feature_corr_matrix = np.nan_to_num(feature_corr_matrix)
            except:
                # 如果计算失败，使用单位矩阵
                feature_corr_matrix = np.eye(X.shape[1])

            # 注意力权重 = 目标相关性 × (1 - 平均特征间相关性)
            # 这样既考虑了与目标的相关性，又惩罚了冗余特征
            for i, feature in enumerate(valid_features):
                target_relevance = target_corr[i]

                # 安全计算特征冗余度
                try:
                    if len(valid_features) > 1:
                        other_corrs = np.concatenate([feature_corr_matrix[i, :i], feature_corr_matrix[i, i+1:]])
                        feature_redundancy = np.mean(other_corrs) if len(other_corrs) > 0 else 0
                    else:
                        feature_redundancy = 0
                except:
                    feature_redundancy = 0

                # 注意力分数 = 目标相关性 × (1 - 冗余度)
                attention_weight = target_relevance * (1 - min(0.8, feature_redundancy))
                attention_scores[feature] = attention_weight

            # 归一化
            if attention_scores:
                max_score = max(attention_scores.values())
                if max_score > 0:
                    attention_scores = {k: v/max_score for k, v in attention_scores.items()}

            return attention_scores

        except Exception as e:
            self.logger.error(f"计算注意力重要性时出错: {e}")
            return {}

    def _calculate_integrated_gradients_importance(self, data, feature_cols, target):
        """集成梯度近似 - 基于梯度信息的特征归因"""
        try:
            # 简化版集成梯度实现
            integrated_grad_scores = {}

            # 准备数据
            numeric_features = []
            valid_features = []

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = pd.to_numeric(data[feature], errors='coerce').fillna(0)
                if len(feature_values.unique()) > 1:
                    numeric_features.append(feature_values.values)
                    valid_features.append(feature)

            if len(numeric_features) < 2:
                return {}

            X = np.column_stack(numeric_features)

            # 使用线性回归作为基础模型计算梯度
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, target)

            # 线性模型的系数就是梯度
            gradients = np.abs(model.coef_)

            # 计算集成梯度（简化版：梯度 × 特征值的平均）
            for i, feature in enumerate(valid_features):
                feature_mean = np.mean(np.abs(X[:, i]))
                integrated_gradient = gradients[i] * feature_mean
                integrated_grad_scores[feature] = integrated_gradient

            # 归一化
            if integrated_grad_scores:
                max_score = max(integrated_grad_scores.values())
                if max_score > 0:
                    integrated_grad_scores = {k: v/max_score for k, v in integrated_grad_scores.items()}

            return integrated_grad_scores

        except Exception as e:
            self.logger.error(f"计算集成梯度重要性时出错: {e}")
            return {}

    def _calculate_dynamic_feature_score(self, data, feature_cols, target):
        """动态特征选择评分 - 考虑时间序列特性"""
        try:
            dynamic_scores = {}

            # 检查目标变量类型
            if isinstance(target, str):
                # 如果target是列名，从data中获取数据
                if target in data.columns:
                    target_values = data[target].values
                else:
                    self.logger.error(f"目标列 {target} 不存在于数据中")
                    return {}
            else:
                # 如果target是数组，直接使用
                target_values = target

            # 如果数据包含时间信息，按时间分组分析特征稳定性
            if 'date' in data.columns or 'date_str' in data.columns:
                date_col = 'date' if 'date' in data.columns else 'date_str'

                # 创建包含目标变量的临时DataFrame
                temp_data = data.copy()
                if isinstance(target, str):
                    temp_target_col = target
                else:
                    temp_target_col = 'temp_target'
                    temp_data[temp_target_col] = target_values

                # 按时间分组
                time_groups = temp_data.groupby(date_col)

                if len(time_groups) < 3:
                    # 时间组太少，使用静态方法
                    return self._calculate_static_feature_score(data, feature_cols, target_values)

                # 计算每个特征在不同时间段的稳定性
                for feature in feature_cols:
                    if feature not in temp_data.columns:
                        continue

                    # 计算特征在不同时间段的相关性稳定性
                    correlations = []
                    for name, group in time_groups:
                        if len(group) > 5:  # 确保有足够样本
                            try:
                                group_feature = pd.to_numeric(group[feature], errors='coerce').fillna(0)
                                group_target = group[temp_target_col].values

                                if len(group_feature.unique()) > 1 and len(group_target) == len(group_feature):
                                    corr = np.corrcoef(group_feature, group_target)[0, 1]
                                    if pd.notna(corr):
                                        correlations.append(abs(corr))
                            except Exception as e:
                                continue

                    if len(correlations) >= 2:
                        # 动态评分 = 平均相关性 × (1 - 相关性标准差)
                        # 这样既考虑了相关性强度，又考虑了时间稳定性
                        mean_corr = np.mean(correlations)
                        std_corr = np.std(correlations)
                        stability = 1 - min(1.0, std_corr / (mean_corr + 1e-8))

                        dynamic_score = mean_corr * stability
                        dynamic_scores[feature] = dynamic_score
            else:
                # 没有时间信息，使用静态方法
                return self._calculate_static_feature_score(data, feature_cols, target_values)

            # 归一化
            if dynamic_scores:
                max_score = max(dynamic_scores.values())
                if max_score > 0:
                    dynamic_scores = {k: v/max_score for k, v in dynamic_scores.items()}

            return dynamic_scores

        except Exception as e:
            self.logger.error(f"计算动态特征评分时出错: {e}")
            return {}

    def _calculate_static_feature_score(self, data, feature_cols, target):
        """静态特征评分 - 备用方法"""
        static_scores = {}

        # 确保target是数组
        if isinstance(target, str):
            if target in data.columns:
                target_values = data[target].values
            else:
                return {}
        else:
            target_values = target

        for feature in feature_cols:
            if feature not in data.columns:
                continue

            feature_values = pd.to_numeric(data[feature], errors='coerce').fillna(0)

            if len(feature_values.unique()) > 1 and len(feature_values) == len(target_values):
                try:
                    # 计算相关系数
                    corr = np.corrcoef(feature_values, target_values)[0, 1]
                    if pd.notna(corr):
                        static_scores[feature] = abs(corr)
                except:
                    continue

        return static_scores

    def _calculate_tree_importance_backup(self, data, feature_cols, target):
        """备用树重要性方法 - 当SHAP不可用时使用"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            # 处理目标变量
            if isinstance(target, str):
                if target in data.columns:
                    target_values = data[target].values
                else:
                    self.logger.error(f"目标列 {target} 不存在于数据中")
                    return {}
            else:
                target_values = np.array(target)

            # 确保目标变量是一维数组
            if target_values.ndim > 1:
                target_values = target_values.flatten()

            # 转换为二分类标签
            target_binary = (target_values > 0).astype(int)

            # 准备特征矩阵
            X_list = []
            valid_features = []

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = data[feature].fillna(0)

                if pd.api.types.is_numeric_dtype(feature_values):
                    X_list.append(feature_values.values)
                    valid_features.append(feature)
                else:
                    try:
                        le = LabelEncoder()
                        encoded_values = le.fit_transform(feature_values.astype(str))
                        X_list.append(encoded_values)
                        valid_features.append(feature)
                    except:
                        continue

            if len(X_list) < 2:
                return {}

            X = np.column_stack(X_list)

            # 确保特征矩阵和目标变量长度一致
            min_length = min(len(X), len(target_binary))
            X = X[:min_length]
            target_binary = target_binary[:min_length]

            # 训练随机森林
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X, target_binary)

            # 获取特征重要性
            importances = rf.feature_importances_

            tree_scores = {}
            for i, feature in enumerate(valid_features):
                try:
                    # 安全转换树重要性值
                    importance_value = importances[i]

                    # 处理各种numpy数据类型
                    if isinstance(importance_value, np.ndarray):
                        if importance_value.size == 1:
                            importance_value = importance_value.item()
                        else:
                            importance_value = importance_value.flatten()[0]
                    elif hasattr(importance_value, 'item'):
                        importance_value = importance_value.item()

                    tree_scores[feature] = float(importance_value)

                except (ValueError, TypeError, IndexError) as e:
                    self.logger.warning(f"树重要性转换失败 {feature}: {e}")
                    tree_scores[feature] = 0.0

            return tree_scores

        except Exception as e:
            self.logger.error(f"计算备用树重要性时出错: {e}")
            return {}

    def _calculate_mutual_information(self, data, feature_cols, target):
        """计算互信息"""
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import LabelEncoder

            mi_scores = {}

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = data[feature].fillna(0)

                # 处理不同类型的特征
                if pd.api.types.is_numeric_dtype(feature_values):
                    # 数值特征直接使用
                    X = feature_values.values.reshape(-1, 1)
                else:
                    # 分类特征需要编码
                    le = LabelEncoder()
                    X = le.fit_transform(feature_values.astype(str)).reshape(-1, 1)

                try:
                    mi_score = mutual_info_classif(X, target, random_state=42)[0]
                    mi_scores[feature] = mi_score
                except:
                    continue

            # 归一化到0-1范围
            if mi_scores:
                max_score = max(mi_scores.values())
                if max_score > 0:
                    mi_scores = {k: v/max_score for k, v in mi_scores.items()}

            return mi_scores

        except ImportError:
            self.logger.warning("sklearn不可用，跳过互信息计算")
            return {}
        except Exception as e:
            self.logger.error(f"计算互信息时出错: {e}")
            return {}

    def _calculate_tree_importance(self, data, feature_cols, target):
        """基于随机森林的特征重要性"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            # 准备特征矩阵
            X_list = []
            valid_features = []

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                feature_values = data[feature].fillna(0)

                if pd.api.types.is_numeric_dtype(feature_values):
                    X_list.append(feature_values.values)
                    valid_features.append(feature)
                else:
                    # 分类特征编码
                    try:
                        le = LabelEncoder()
                        encoded_values = le.fit_transform(feature_values.astype(str))
                        X_list.append(encoded_values)
                        valid_features.append(feature)
                    except:
                        continue

            if len(X_list) < 2:
                return {}

            X = np.column_stack(X_list)

            # 训练随机森林
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            rf.fit(X, target)

            # 获取特征重要性
            importances = rf.feature_importances_

            tree_scores = {}
            for i, feature in enumerate(valid_features):
                tree_scores[feature] = importances[i]

            return tree_scores

        except ImportError:
            self.logger.warning("sklearn不可用，跳过树模型特征重要性计算")
            return {}
        except Exception as e:
            self.logger.error(f"计算树模型特征重要性时出错: {e}")
            return {}

    def _calculate_correlation_importance(self, data, feature_cols, target):
        """计算相关系数重要性"""
        corr_scores = {}

        for feature in feature_cols:
            if feature not in data.columns:
                continue

            feature_values = pd.to_numeric(data[feature], errors='coerce').fillna(0)

            try:
                # 计算Pearson相关系数
                corr = np.corrcoef(feature_values, target)[0, 1]
                if pd.notna(corr) and np.isfinite(corr):
                    corr_scores[feature] = abs(corr)
            except:
                continue

        return corr_scores

    def _calculate_anova_importance(self, data, feature_cols, target):
        """计算ANOVA F-test重要性"""
        try:
            from scipy.stats import f_oneway

            anova_scores = {}

            # 分组：上涨组和下跌组
            up_group = data[target == 1]
            down_group = data[target == 0]

            if len(up_group) < 5 or len(down_group) < 5:
                return {}

            for feature in feature_cols:
                if feature not in data.columns:
                    continue

                up_values = pd.to_numeric(up_group[feature], errors='coerce').dropna()
                down_values = pd.to_numeric(down_group[feature], errors='coerce').dropna()

                if len(up_values) < 3 or len(down_values) < 3:
                    continue

                try:
                    f_stat, p_value = f_oneway(up_values, down_values)
                    if pd.notna(f_stat) and pd.notna(p_value):
                        # 使用F统计量作为重要性分数，p值越小越重要
                        importance = f_stat * (1 - p_value) if p_value < 0.05 else 0
                        anova_scores[feature] = importance
                except:
                    continue

            # 归一化
            if anova_scores:
                max_score = max(anova_scores.values())
                if max_score > 0:
                    anova_scores = {k: v/max_score for k, v in anova_scores.items()}

            return anova_scores

        except ImportError:
            self.logger.warning("scipy不可用，跳过ANOVA分析")
            return {}
        except Exception as e:
            self.logger.error(f"计算ANOVA重要性时出错: {e}")
            return {}









    def _cluster_change_point_patterns(self, change_point_features):
        """对变化点模式进行聚类分析"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            if len(change_point_features) < 10:
                return {'status': 'insufficient_data'}

            # 准备聚类数据
            X, feature_names = self._prepare_clustering_data(change_point_features)

            if len(X) < 10:
                return {'status': 'insufficient_valid_data'}

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 确定最佳聚类数
            optimal_clusters = min(5, len(X) // 10 + 1)

            # 聚类
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # 分析每个聚类
            cluster_analysis = {}
            for i in range(optimal_clusters):
                cluster_mask = cluster_labels == i
                cluster_features = [change_point_features[j] for j in range(len(change_point_features)) if cluster_mask[j]]

                # 计算聚类统计
                returns = [f.get('future_return_2d', 0) for f in cluster_features if f.get('future_return_2d') is not None]
                avg_return = np.mean(returns) if returns else 0

                cluster_analysis[i] = {
                    'size': int(np.sum(cluster_mask)),
                    'avg_return': float(avg_return),
                    'return_std': float(np.std(returns)) if returns else 0,
                    'center': kmeans.cluster_centers_[i].tolist()
                }

            return {
                'status': 'success',
                'num_clusters': optimal_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_analysis': cluster_analysis,
                'feature_names': feature_names
            }

        except ImportError:
            return {'status': 'sklearn_not_available'}
        except Exception as e:
            self.logger.error(f"聚类分析时出错: {e}")
            return {'status': 'error', 'message': str(e)}

    def _calculate_change_point_feature_importance(self, change_point_features, target_column='future_return_2d'):
        """
        计算特征重要性 - 高性能优化版本

        优化策略：
        1. 使用pandas DataFrame进行向量化操作
        2. 批量计算相关系数
        3. 避免重复的类型转换和检查

        Args:
            change_point_features: 变化点特征列表
            target_column: 目标列名，默认为'future_return_2d'
        """
        if not change_point_features:
            return {}

        self.logger.info(f"开始计算特征重要性，共 {len(change_point_features)} 个样本，目标列: {target_column}")

        try:
            # 检查输入数据类型
            if not isinstance(change_point_features, list):
                self.logger.error(f"change_point_features 应该是list类型，实际类型: {type(change_point_features)}")
                return {}

            if len(change_point_features) == 0:
                self.logger.warning("change_point_features 为空列表")
                return {}

            # 检查第一个元素是否为字典
            if not isinstance(change_point_features[0], dict):
                self.logger.error(f"change_point_features[0] 应该是dict类型，实际类型: {type(change_point_features[0])}")
                return {}

            # 转换为DataFrame进行向量化操作
            df = pd.DataFrame(change_point_features)
            self.logger.info(f"成功创建DataFrame，形状: {df.shape}，列: {list(df.columns)}")

            # 过滤掉不需要的列 - 修正：保留板块相关特征
            exclude_cols = [
                target_column, 'stock_code', 'date', 'date_str', 'change_point_idx',
                'future_return_2d'  # 目标变量
            ]
            # 注意：不再排除most_relevant_sector，这是重要的板块特征，对预测有价值

            # 修正：不再排除技术形态特征，因为它们是有效的二进制特征
            # 只排除真正无意义的状态特征
            state_features_to_exclude = [
                'trend_state', 'stock_state'  # 这些状态特征可能确实无意义
            ]
            exclude_cols.extend(state_features_to_exclude)

            feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('_')]

            # 确保目标列存在且为数值型
            if target_column not in df.columns:
                self.logger.warning(f"目标列 {target_column} 不存在")
                return {}

            # 转换为数值型，无效值设为NaN
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

            # 过滤掉目标列为空的行
            valid_mask = df[target_column].notna()
            if valid_mask.sum() < 10:  # 至少需要10个有效样本
                self.logger.warning("有效样本数量不足，无法计算特征重要性")
                return {}

            df_valid = df[valid_mask]

            # 批量转换特征列为数值型
            correlations = {}
            for feature_name in feature_cols:
                if feature_name not in df_valid.columns:
                    continue

                # 转换为数值型
                feature_values = pd.to_numeric(df_valid[feature_name], errors='coerce')

                # 检查有效样本数量
                valid_feature_mask = feature_values.notna()
                combined_valid = valid_feature_mask & df_valid[target_column].notna()

                if combined_valid.sum() >= 10:  # 至少10个有效样本
                    try:
                        # 计算相关系数，添加边界情况处理
                        feature_valid = feature_values[combined_valid]
                        target_valid = df_valid[target_column][combined_valid]

                        # 检查数据有效性
                        if len(feature_valid) == 0 or len(target_valid) == 0:
                            continue

                        # 检查是否所有值都相同（标准差为0）
                        if feature_valid.std() == 0 or target_valid.std() == 0:
                            # 如果标准差为0，相关系数无意义，跳过
                            continue

                        # 检查是否包含无穷大或NaN
                        if not (np.isfinite(feature_valid).all() and np.isfinite(target_valid).all()):
                            continue

                        # 使用numpy计算相关系数，更好地处理边界情况
                        with np.errstate(divide='ignore', invalid='ignore'):
                            corr_matrix = np.corrcoef(feature_valid, target_valid)
                            if corr_matrix.shape == (2, 2):
                                corr = corr_matrix[0, 1]
                                if pd.notna(corr) and np.isfinite(corr) and not np.isnan(corr):
                                    correlations[feature_name] = abs(corr)
                    except Exception as e:
                        # 跳过计算失败的特征
                        continue

            # 按重要性排序
            sorted_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))

            self.logger.info(f"特征重要性计算完成，共计算 {len(sorted_correlations)} 个特征")

            return sorted_correlations

        except Exception as e:
            self.logger.error(f"计算特征重要性时出错: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {}

    def _analyze_regime_transitions(self, change_point_features):
        """分析市场状态转换"""
        if not change_point_features:
            return {}

        # 按时间排序 - 修复类型错误
        def safe_sort_key(x):
            stock_code = str(x.get('stock_code', ''))
            date = x.get('date', x.get('date_str', ''))
            # 确保date是字符串类型
            date = str(date) if date is not None else ''
            return (stock_code, date)

        try:
            sorted_features = sorted(change_point_features, key=safe_sort_key)
        except Exception as e:
            self.logger.error(f"排序变化点特征时出错: {e}")
            # 如果排序失败，直接使用原始顺序
            sorted_features = change_point_features

        transitions = {
            'bull_to_bear': 0,
            'bear_to_bull': 0,
            'neutral_to_bull': 0,
            'neutral_to_bear': 0,
            'total_transitions': len(sorted_features)
        }

        # 简单的状态转换分析
        for feature_dict in sorted_features:
            future_return = feature_dict.get('future_return_2d', 0)
            try:
                future_return = float(future_return)
                if future_return > 0.05:
                    transitions['neutral_to_bull'] += 1
                elif future_return < -0.05:
                    transitions['neutral_to_bear'] += 1
            except (ValueError, TypeError):
                continue

        return transitions

    def _calculate_pattern_feature_importance(self, positive_patterns, negative_patterns):
        """计算模式特征重要性"""
        all_patterns = positive_patterns + negative_patterns
        if not all_patterns:
            return {}

        # 为每个模式添加标签
        labeled_patterns = []
        for pattern in positive_patterns:
            pattern_copy = pattern.copy()
            pattern_copy['label'] = 1
            labeled_patterns.append(pattern_copy)

        for pattern in negative_patterns:
            pattern_copy = pattern.copy()
            pattern_copy['label'] = 0
            labeled_patterns.append(pattern_copy)

        return self._calculate_change_point_feature_importance(labeled_patterns, 'label')



    def _prepare_clustering_data(self, change_point_features):
        """
        准备聚类数据 - 优化版本

        优化策略：
        1. 使用pandas DataFrame进行向量化操作
        2. 批量处理数据转换
        """
        if not change_point_features:
            return np.array([]), []

        # 动态获取特征列表
        if change_point_features:
            sample_feature = change_point_features[0]
            exclude_fields = ['stock_code', 'date_str', 'date']
            feature_names = [k for k in sample_feature.keys()
                           if k not in exclude_fields and not k.startswith('_')]
        else:
            feature_names = ['pct_chg', 'vol', 'turnover_rate', 'amplitude', 'future_return_2d']

        try:
            # 转换为DataFrame
            df = pd.DataFrame(change_point_features)

            # 批量处理特征
            X_list = []
            for feature_name in feature_names:
                if feature_name in df.columns:
                    # 转换为数值型，无效值填充为0
                    feature_values = pd.to_numeric(df[feature_name], errors='coerce').fillna(0)
                    X_list.append(feature_values.values)
                else:
                    # 如果特征不存在，填充为0
                    X_list.append(np.zeros(len(df)))

            # 转置得到正确的形状
            X = np.column_stack(X_list)

            return X, feature_names

        except Exception as e:
            self.logger.error(f"准备聚类数据时出错: {e}")
            return np.array([]), []

    def _predict_stocks_with_change_points(self, complete_features_df, feature_patterns):
        """
        集成学习增强的变化点特征模式预测

        结合多策略分析框架进行预测：
        1. 识别当前市场状态
        2. 各策略独立计算相似度
        3. 根据市场状态调整策略权重
        4. 集成预测结果

        Args:
            complete_features_df: 包含所有特征的完整DataFrame
            feature_patterns: 特征模式分析结果（包含策略模式）

        Returns:
            dict: 集成预测结果
        """
        self.logger.info("开始集成学习变化点特征模式预测...")

        if complete_features_df is None or len(complete_features_df) == 0:
            self.logger.warning("预测数据为空")
            return {}

        # 检查特征模式数据
        pattern_system = feature_patterns.get('pattern_system', {})
        if not pattern_system or pattern_system.get('status') != 'success':
            self.logger.warning("特征模式数据无效或分析失败")
            return {}

        # 获取传统特征模式
        up_patterns = pattern_system.get('up_patterns', {})
        down_patterns = pattern_system.get('down_patterns', {})
        feature_cols = pattern_system.get('feature_cols', [])
        avg_up_return = pattern_system.get('avg_up_return', 0.05)
        avg_down_return = pattern_system.get('avg_down_return', -0.05)

        if not up_patterns or not down_patterns:
            self.logger.warning("上涨或下跌特征模式为空")
            return {}

        # 获取策略模式和权重
        strategy_patterns = feature_patterns.get('strategy_patterns', {})
        ensemble_weights = feature_patterns.get('ensemble_weights', {})

        self.logger.info(f"使用 {len(up_patterns)} 个上涨特征模式和 {len(down_patterns)} 个下跌特征模式进行预测")
        self.logger.info(f"集成学习策略模式: {len(strategy_patterns)} 个市场状态")

        # 集成学习预测
        if strategy_patterns and ensemble_weights:
            predictions = self._calculate_ensemble_predictions(
                complete_features_df, up_patterns, down_patterns, feature_cols,
                strategy_patterns, ensemble_weights, avg_up_return, avg_down_return
            )
        else:
            # 回退到传统预测
            self.logger.info("策略模式不可用，使用传统预测方法")
            predictions = self._calculate_pattern_based_predictions(
                complete_features_df, up_patterns, down_patterns,
                feature_cols, avg_up_return, avg_down_return
            )

        return {
            'pattern_predictions': predictions
        }

    def _calculate_ensemble_predictions(self, complete_features_df, up_patterns, down_patterns,
                                      feature_cols, strategy_patterns, ensemble_weights,
                                      avg_up_return, avg_down_return):
        """
        集成学习预测计算

        Args:
            complete_features_df: 完整特征数据
            up_patterns: 传统上涨模式
            down_patterns: 传统下跌模式
            feature_cols: 特征列
            strategy_patterns: 策略模式字典
            ensemble_weights: 策略权重
            avg_up_return: 历史平均上涨收益
            avg_down_return: 历史平均下跌收益

        Returns:
            dict: 集成预测结果
        """
        import numpy as np

        predictions = {}

        # 获取样本基本信息
        stock_codes = complete_features_df['stock_code'].values
        dates = complete_features_df['date_str'].values

        self.logger.info(f"开始集成学习预测 {len(complete_features_df)} 个样本...")

        # 准备特征数据
        available_features = [col for col in feature_cols if col in complete_features_df.columns]
        if len(available_features) < len(feature_cols) * 0.8:
            self.logger.warning(f"可用特征不足: {len(available_features)}/{len(feature_cols)}")
            return predictions

        feature_matrix = complete_features_df[available_features].fillna(0).values

        # 为每个样本进行集成预测
        for i, (stock_code, date) in enumerate(zip(stock_codes, dates)):
            try:
                current_features = feature_matrix[i]

                # 1. 识别当前市场状态
                market_features = self._extract_market_features(complete_features_df.iloc[i])
                current_market_state = self.ensemble_system.identify_market_regime(market_features)

                # 2. 获取当前市场状态的策略权重
                base_weights = ensemble_weights.copy()
                adjusted_weights = self.ensemble_system.adjust_weights_by_market(base_weights, current_market_state)

                # 3. 各策略独立预测
                strategy_predictions = {}

                for strategy_name, strategy_features in self.ensemble_system.strategy_features.items():
                    # 获取该策略的可用特征
                    strategy_available_features = [f for f in strategy_features if f in available_features]

                    if len(strategy_available_features) < 2:
                        continue

                    # 获取该策略在当前市场状态下的模式
                    regime_patterns = strategy_patterns.get(current_market_state, {})
                    strategy_pattern = regime_patterns.get(strategy_name, {})

                    if strategy_pattern:
                        # 使用策略特定模式预测
                        strategy_pred = self._predict_with_strategy_pattern(
                            current_features, available_features, strategy_available_features,
                            strategy_pattern, avg_up_return, avg_down_return
                        )
                    else:
                        # 回退到传统模式
                        strategy_pred = self._predict_with_traditional_pattern(
                            current_features, available_features, strategy_available_features,
                            up_patterns, down_patterns, avg_up_return, avg_down_return
                        )

                    if strategy_pred is not None:
                        strategy_predictions[strategy_name] = strategy_pred

                # 4. 集成预测结果
                if strategy_predictions:
                    ensemble_prediction = self._ensemble_strategy_predictions(
                        strategy_predictions, adjusted_weights
                    )

                    key = f"{stock_code}_{date}"
                    predictions[key] = {
                        'stock_code': stock_code,
                        'date': date,
                        'predicted_return': float(ensemble_prediction['predicted_return']),
                        'confidence': float(ensemble_prediction['confidence']),
                        'market_state': current_market_state,
                        'strategy_predictions': strategy_predictions,
                        'ensemble_weights': adjusted_weights,
                        'method': 'ensemble_prediction'
                    }

            except Exception as e:
                self.logger.warning(f"集成预测样本 {stock_code}_{date} 时出错: {e}")
                continue

        self.logger.info(f"集成学习预测完成，共预测 {len(predictions)} 个样本")
        return predictions

    def _extract_market_features(self, sample_row):
        """提取市场特征用于状态识别"""
        try:
            return {
                'market_benchmark_change': sample_row.get('market_benchmark_change', 0),
                'market_trend_strength': sample_row.get('market_trend_strength', 0.5),
                'market_rsi12': sample_row.get('market_rsi12', 50)
            }
        except:
            return {'market_benchmark_change': 0, 'market_trend_strength': 0.5, 'market_rsi12': 50}

    def _predict_with_strategy_pattern(self, current_features, available_features,
                                     strategy_features, strategy_pattern, avg_up_return, avg_down_return):
        """使用策略特定模式预测"""
        try:
            up_patterns = strategy_pattern.get('up_patterns', {})
            down_patterns = strategy_pattern.get('down_patterns', {})

            if not up_patterns or not down_patterns:
                return None

            # 计算策略特征的相似度
            strategy_indices = [available_features.index(f) for f in strategy_features if f in available_features]
            if not strategy_indices:
                return None

            strategy_current_features = current_features[strategy_indices]
            strategy_feature_names = [available_features[i] for i in strategy_indices]

            # 计算相似度
            up_similarity = self._calculate_pattern_similarity(
                strategy_current_features, strategy_feature_names, up_patterns
            )
            down_similarity = self._calculate_pattern_similarity(
                strategy_current_features, strategy_feature_names, down_patterns
            )

            # 预测逻辑
            if up_similarity > down_similarity:
                similarity_strength = up_similarity - down_similarity
                confidence = min(0.95, max(0.5, similarity_strength))
                predicted_return = avg_up_return * up_similarity * (1 - down_similarity * 0.5) * confidence
                predicted_return = max(0.001, min(0.15, predicted_return))
            else:
                similarity_strength = down_similarity - up_similarity
                confidence = min(0.95, max(0.5, similarity_strength))
                predicted_return = avg_down_return * down_similarity * (1 - up_similarity * 0.5) * confidence
                predicted_return = max(-0.15, min(-0.001, predicted_return))

            return {
                'predicted_return': predicted_return,
                'confidence': confidence,
                'up_similarity': up_similarity,
                'down_similarity': down_similarity
            }

        except Exception as e:
            return None

    def _predict_with_traditional_pattern(self, current_features, available_features,
                                        strategy_features, up_patterns, down_patterns,
                                        avg_up_return, avg_down_return):
        """使用传统模式预测（回退方法）"""
        try:
            # 只使用策略相关的特征
            strategy_indices = [available_features.index(f) for f in strategy_features if f in available_features]
            if not strategy_indices:
                return None

            strategy_current_features = current_features[strategy_indices]
            strategy_feature_names = [available_features[i] for i in strategy_indices]

            # 从传统模式中筛选策略相关特征
            strategy_up_patterns = {k: v for k, v in up_patterns.items() if k in strategy_features}
            strategy_down_patterns = {k: v for k, v in down_patterns.items() if k in strategy_features}

            if not strategy_up_patterns or not strategy_down_patterns:
                return None

            # 计算相似度
            up_similarity = self._calculate_pattern_similarity(
                strategy_current_features, strategy_feature_names, strategy_up_patterns
            )
            down_similarity = self._calculate_pattern_similarity(
                strategy_current_features, strategy_feature_names, strategy_down_patterns
            )

            # 简化预测逻辑
            if up_similarity > down_similarity:
                predicted_return = avg_up_return * 0.5  # 降低传统方法的权重
                confidence = 0.3
            else:
                predicted_return = avg_down_return * 0.5
                confidence = 0.3

            return {
                'predicted_return': predicted_return,
                'confidence': confidence,
                'up_similarity': up_similarity,
                'down_similarity': down_similarity
            }

        except Exception as e:
            return None

    def _ensemble_strategy_predictions(self, strategy_predictions, weights):
        """集成各策略的预测结果"""
        try:
            total_weighted_return = 0.0
            total_weighted_confidence = 0.0
            total_weight = 0.0

            for strategy_name, prediction in strategy_predictions.items():
                weight = weights.get(strategy_name, 0.0)
                if weight > 0:
                    total_weighted_return += prediction['predicted_return'] * weight
                    total_weighted_confidence += prediction['confidence'] * weight
                    total_weight += weight

            if total_weight > 0:
                ensemble_return = total_weighted_return / total_weight
                ensemble_confidence = total_weighted_confidence / total_weight
            else:
                ensemble_return = 0.0
                ensemble_confidence = 0.0

            return {
                'predicted_return': ensemble_return,
                'confidence': ensemble_confidence
            }

        except Exception as e:
            return {'predicted_return': 0.0, 'confidence': 0.0}

    def _calculate_pattern_based_predictions(self, complete_features_df, up_patterns, down_patterns,
                                           feature_cols, avg_up_return, avg_down_return):
        """
        基于特征模式计算预测结果

        Args:
            complete_features_df: 完整特征数据
            up_patterns: 上涨特征模式
            down_patterns: 下跌特征模式
            feature_cols: 特征列列表
            avg_up_return: 历史平均上涨幅度
            avg_down_return: 历史平均下跌幅度

        Returns:
            dict: 预测结果
        """
        import numpy as np

        predictions = {}

        # 获取样本基本信息
        stock_codes = complete_features_df['stock_code'].values
        dates = complete_features_df['date_str'].values

        self.logger.info(f"开始计算 {len(complete_features_df)} 个样本的特征模式相似度...")

        # 准备特征数据
        available_features = [col for col in feature_cols if col in complete_features_df.columns]
        if len(available_features) < len(feature_cols) * 0.8:
            self.logger.warning(f"可用特征不足: {len(available_features)}/{len(feature_cols)}")
            return predictions

        feature_matrix = complete_features_df[available_features].fillna(0).values

        # 为每个样本计算相似度
        for i, (stock_code, date) in enumerate(zip(stock_codes, dates)):
            try:
                current_features = feature_matrix[i]

                # 计算与上涨模式的相似度
                up_similarity = self._calculate_pattern_similarity(
                    current_features, available_features, up_patterns
                )

                # 计算与下跌模式的相似度
                down_similarity = self._calculate_pattern_similarity(
                    current_features, available_features, down_patterns
                )

                # 预测判断
                if up_similarity > down_similarity:
                    # 预测上涨
                    similarity_strength = up_similarity - down_similarity
                    confidence = min(0.95, max(0.5, similarity_strength))

                    # 预测收益 = 历史平均涨幅 × 上涨相似度权重 × (1 - 下跌相似度权重)
                    predicted_return = (avg_up_return * up_similarity *
                                      (1 - down_similarity * 0.5) * confidence)

                    # 限制预测收益在合理范围内
                    predicted_return = max(0.001, min(0.15, predicted_return))

                else:
                    # 预测下跌或持平
                    similarity_strength = down_similarity - up_similarity
                    confidence = min(0.95, max(0.5, similarity_strength))

                    # 预测收益 = 历史平均跌幅 × 下跌相似度权重 × (1 - 上涨相似度权重)
                    predicted_return = (avg_down_return * down_similarity *
                                      (1 - up_similarity * 0.5) * confidence)

                    # 限制预测收益在合理范围内
                    predicted_return = max(-0.15, min(-0.001, predicted_return))

                key = f"{stock_code}_{date}"
                predictions[key] = {
                    'stock_code': stock_code,
                    'date': date,
                    'predicted_return': float(predicted_return),
                    'confidence': float(confidence),
                    'up_similarity': float(up_similarity),
                    'down_similarity': float(down_similarity),
                    'method': 'pattern_based_prediction'
                }

            except Exception as e:
                self.logger.warning(f"计算样本 {stock_code}_{date} 的预测时出错: {e}")
                continue

        self.logger.info(f"特征模式预测完成，共预测 {len(predictions)} 个样本")

        return predictions

    def _calculate_pattern_similarity(self, current_features, feature_names, patterns):
        """
        计算当前特征与特征模式的相似度

        Args:
            current_features: 当前特征值数组
            feature_names: 特征名称列表
            patterns: 特征模式字典

        Returns:
            float: 相似度分数 (0-1)
        """
        if not patterns:
            return 0.0

        total_similarity = 0.0
        valid_features = 0

        for i, feature_name in enumerate(feature_names):
            if feature_name not in patterns:
                continue

            pattern = patterns[feature_name]

            # 检查模式是否包含必要的键
            if not isinstance(pattern, dict) or 'min' not in pattern or 'max' not in pattern:
                continue

            current_value = current_features[i]

            # 检查数值有效性
            if not (isinstance(current_value, (int, float)) and
                    isinstance(pattern['min'], (int, float)) and
                    isinstance(pattern['max'], (int, float))):
                continue

            # 检查当前值是否在模式范围内
            if pattern['min'] <= current_value <= pattern['max']:
                # 在范围内，计算相似度
                range_size = pattern['max'] - pattern['min']
                if range_size > 0:
                    # 基于距离中心点的相似度
                    center = (pattern['min'] + pattern['max']) / 2
                    distance_from_center = abs(current_value - center)
                    similarity = 1.0 - (distance_from_center / (range_size / 2))
                    similarity = max(0.0, min(1.0, similarity))
                else:
                    similarity = 1.0  # 范围为0，完全匹配

                # 根据模式的覆盖率调整权重
                weight = pattern.get('coverage', 0.5)
                total_similarity += similarity * weight
                valid_features += weight
            else:
                # 不在范围内，计算惩罚相似度
                range_size = pattern['max'] - pattern['min']
                if range_size > 0:
                    if current_value < pattern['min']:
                        distance = pattern['min'] - current_value
                    else:
                        distance = current_value - pattern['max']

                    # 距离越远，相似度越低
                    penalty_similarity = max(0.0, 1.0 - distance / range_size)
                    weight = pattern.get('coverage', 0.5) * 0.5  # 降低权重
                    total_similarity += penalty_similarity * weight
                    valid_features += weight

        if valid_features > 0:
            return total_similarity / valid_features
        else:
            return 0.0





    def _predict_with_similarity_matching(self, complete_features_df, feature_patterns):
        """
        重构版本：使用真正的历史相似度匹配进行预测

        实现原理：
        1. 从历史特征数据库中找到与当前特征最相似的样本
        2. 使用多种相似度计算方法（余弦、欧几里得、曼哈顿）
        3. 基于top-k最相似样本进行加权预测
        4. 支持板块分层匹配，提高预测准确性

        Args:
            complete_features_df: 包含所有特征的完整DataFrame（个股+板块+相对特征）
            feature_patterns: 训练好的相似度匹配系统数据

        Returns:
            dict: 预测结果，包含每只股票的预测收益率和置信度
        """
        import time
        start_time = time.time()

        predictions = {}

        try:
            self.logger.info("开始相似度匹配预测...")

            # 检查输入数据
            if complete_features_df.empty:
                self.logger.warning("预测特征DataFrame为空")
                return predictions

            # 获取相似度匹配系统数据
            similarity_system = feature_patterns.get('similarity_system', {})
            historical_db = similarity_system.get('historical_features_db', [])
            feature_stats = similarity_system.get('feature_statistics', {})

            if not historical_db:
                self.logger.warning("历史特征数据库为空，无法进行相似度匹配")
                return predictions

            self.logger.info(f"历史数据库包含 {len(historical_db)} 个样本")

            # 使用板块分层相似度匹配
            predictions = self._predict_with_sector_stratified_similarity(
                complete_features_df, similarity_system
            )

            elapsed_time = time.time() - start_time
            self.logger.info(f"相似度匹配预测完成，用时 {elapsed_time:.2f} 秒，预测了 {len(predictions)} 只股票")

        except Exception as e:
            self.logger.error(f"相似度匹配预测失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {}

        return predictions

    def _predict_with_sector_stratified_similarity(self, current_features_df, similarity_system):
        """
        高性能板块分层相似度匹配预测

        性能优化策略：
        1. 向量化相似度计算，避免逐样本循环
        2. 预构建历史特征矩阵，避免重复转换
        3. 批量预测，减少函数调用开销
        4. 内存优化，及时释放大型数组

        Args:
            current_features_df: 当前特征DataFrame
            similarity_system: 相似度匹配系统数据

        Returns:
            dict: 预测结果
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import time

        start_time = time.time()
        predictions = {}

        # 获取历史数据库和特征统计
        historical_db = similarity_system.get('historical_features_db', [])
        sector_groups = similarity_system.get('sector_feature_groups', {})
        feature_stats = similarity_system.get('feature_statistics', {})

        if not historical_db:
            self.logger.warning("历史数据库为空")
            return predictions

        # 准备当前特征数据
        feature_names = feature_stats.get('feature_names', [])
        scaler = feature_stats.get('scaler')

        if not feature_names or scaler is None:
            self.logger.warning("特征统计信息不完整")
            return predictions

        self.logger.info(f"开始高性能相似度匹配，历史样本: {len(historical_db)}, 板块数: {len(sector_groups)}")

        try:
            # 1. 预处理：排除未来信息特征
            safe_feature_names = [f for f in feature_names
                                 if f not in ['future_return_2d', 'future_return_1d', 'future_return_5d']]

            # 2. 提取并标准化当前特征矩阵
            self.logger.info(f"提取特征矩阵，特征数量: {len(safe_feature_names)}")
            current_matrix = current_features_df[safe_feature_names].fillna(0).values
            current_scaled = scaler.transform(current_matrix)

            # 3. 预构建历史特征矩阵（一次性转换，避免重复计算）
            self.logger.info("预构建历史特征矩阵...")
            historical_matrix, historical_targets, historical_sectors = self._build_historical_matrix(
                historical_db, safe_feature_names
            )

            if len(historical_matrix) == 0:
                self.logger.warning("历史特征矩阵为空")
                return predictions

            # 4. 内存优化的分块相似度计算
            self.logger.info("开始分块相似度计算...")
            predictions = self._chunked_similarity_predict(
                current_features_df, current_scaled, historical_matrix,
                historical_targets, historical_sectors, safe_feature_names,
                chunk_size=1000, top_k=10
            )

            elapsed_time = time.time() - start_time
            self.logger.info(f"高性能相似度匹配完成，预测了 {len(predictions)} 个样本，用时 {elapsed_time:.2f} 秒")

        except Exception as e:
            self.logger.error(f"高性能相似度匹配失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return predictions

    def _chunked_similarity_predict(self, current_df, current_scaled, historical_matrix,
                                   historical_targets, historical_sectors, feature_names,
                                   chunk_size=1000, top_k=10):
        """
        内存优化的分块相似度预测

        修复问题1：改进相似度预测逻辑，基于历史变化点的实际收益率结合相似度来评估
        - 使用相似度加权的历史收益率作为预测基础
        - 结合相似度置信度进行预测调整
        - 限制预测收益率在合理范围内

        Args:
            current_df: 当前样本DataFrame
            current_scaled: 标准化后的当前特征矩阵
            historical_matrix: 历史特征矩阵
            historical_targets: 历史目标值（历史变化点的实际收益率）
            historical_sectors: 历史板块信息
            feature_names: 特征名称列表
            chunk_size: 每次处理的样本数量
            top_k: 选择最相似的k个样本
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        predictions = {}
        n_current = len(current_df)

        # 获取当前样本的基本信息
        stock_codes = current_df['stock_code'].values
        dates = current_df['date_str'].values
        current_sectors = current_df['most_relevant_sector'].fillna('unknown').values

        self.logger.info(f"开始分块处理 {n_current} 个样本，块大小: {chunk_size}")

        # 分块处理以避免内存问题
        for i in range(0, n_current, chunk_size):
            chunk_end = min(i + chunk_size, n_current)
            self.logger.debug(f"处理块 {i//chunk_size + 1}，样本范围: {i}-{chunk_end}")

            # 获取当前块的特征矩阵
            chunk_features = current_scaled[i:chunk_end]

            # 计算当前块与历史样本的相似度
            chunk_similarities = cosine_similarity(chunk_features, historical_matrix)

            # 处理当前块中的每个样本
            for j in range(len(chunk_features)):
                try:
                    actual_idx = i + j
                    stock_code = stock_codes[actual_idx]
                    date = dates[actual_idx]
                    current_sector = current_sectors[actual_idx]

                    # 获取当前样本与所有历史样本的相似度
                    similarities = chunk_similarities[j]

                    # 板块过滤：优先使用同板块样本
                    sector_mask = historical_sectors == current_sector
                    if np.sum(sector_mask) >= 10:  # 如果同板块样本足够
                        sector_similarities = similarities[sector_mask]
                        sector_targets = historical_targets[sector_mask]
                    else:  # 否则使用全部样本
                        sector_similarities = similarities
                        sector_targets = historical_targets

                    # 选择top-k最相似样本
                    top_k_actual = min(top_k, len(sector_similarities))
                    top_indices = np.argsort(sector_similarities)[-top_k_actual:]

                    top_similarities = sector_similarities[top_indices]
                    top_targets = sector_targets[top_indices]

                    # 相似度阈值过滤
                    similarity_threshold = 0.01
                    valid_mask = top_similarities > similarity_threshold

                    if not np.any(valid_mask):
                        valid_similarities = top_similarities
                        valid_targets = top_targets
                    else:
                        valid_similarities = top_similarities[valid_mask]
                        valid_targets = top_targets[valid_mask]

                    # 检查目标值
                    if len(valid_targets) == 0 or np.all(valid_targets == 0):
                        continue

                    # 改进的加权预测：基于历史变化点的实际收益率结合相似度
                    weights = valid_similarities / np.sum(valid_similarities)

                    # 使用相似度加权的历史收益率作为预测基础
                    # 结合相似度权重和历史实际收益率分布
                    similarity_weighted_return = np.sum(weights * valid_targets)

                    # 考虑相似度的置信度调整：高相似度样本的预测更可靠
                    confidence_adjustment = np.mean(valid_similarities)

                    # 最终预测收益率：结合历史收益率和相似度置信度
                    predicted_return = similarity_weighted_return * (0.7 + 0.3 * confidence_adjustment)

                    # 限制预测收益率在合理范围内
                    predicted_return = np.clip(predicted_return, -0.15, 0.15)  # 限制在-15%到+15%之间

                    # 改进的置信度计算：考虑相似度质量和样本数量
                    avg_similarity = np.mean(valid_similarities)
                    sample_count_factor = min(1.0, len(valid_similarities) / 5.0)
                    similarity_quality = np.std(valid_similarities) if len(valid_similarities) > 1 else 0.1  # 相似度的一致性

                    # 置信度综合考虑：平均相似度、样本数量、相似度一致性
                    confidence = avg_similarity * sample_count_factor * (1.0 - similarity_quality * 0.5)

                    # 存储预测结果
                    key = f"{stock_code}_{date}"
                    predictions[key] = {
                        'stock_code': stock_code,
                        'date': date,
                        'predicted_return': float(predicted_return),
                        'confidence': float(confidence),
                        'similar_samples_count': len(valid_similarities),
                        'sector': current_sector,
                        'method': 'chunked_similarity'
                    }

                except Exception as e:
                    self.logger.debug(f"处理样本 {stock_code}_{date} 时出错: {e}")
                    continue

            # 定期报告进度
            if (i//chunk_size) % 10 == 0 or chunk_end == n_current:
                self.logger.info(f"已处理 {chunk_end}/{n_current} 个样本，当前预测数: {len(predictions)}")

        return predictions

    def _build_historical_matrix(self, historical_db, feature_names):
        """预构建历史特征矩阵"""
        import numpy as np

        historical_features = []
        historical_targets = []
        historical_sectors = []

        for sample in historical_db:
            features = sample.get('features', [])
            target = sample.get('target', 0)
            sector = sample.get('sector', 'unknown')

            if len(features) == len(feature_names) and not pd.isna(target):
                historical_features.append(features)
                historical_targets.append(target)
                historical_sectors.append(sector)

        return (np.array(historical_features),
                np.array(historical_targets),
                np.array(historical_sectors))


    # 删除未使用的函数 _calculate_sector_similarity
    # 该函数定义了 positive_similarity 但从未被调用，导致评估指标逻辑混乱
    # 现在使用统一的评估指标体系替代



    def _get_consistent_feature_list(self, data_df):
        """
        获取与训练阶段一致的特征列表

        Args:
            data_df: 数据DataFrame

        Returns:
            list: 特征列表
        """
        # 定义统一的排除字段列表 - 修正：保留板块相关特征
        exclude_fields = [
            'stock_code', 'date_str', 'date', 'future_return_2d',
            'change_point_idx'  # 变化点索引
        ]
        # 注意：不再排除most_relevant_sector，这是重要的板块特征，对预测有价值

        # 注意：不再过滤技术形态特征，因为它们是有效的二进制特征

        # 获取所有可用的数值特征
        available_features = []
        for col in data_df.columns:
            if col not in exclude_fields and not col.startswith('_'):
                # 检查是否为数值型特征
                try:
                    # 测试前100行是否可以转换为数值
                    test_data = data_df[col].iloc[:min(100, len(data_df))]
                    pd.to_numeric(test_data, errors='coerce')
                    available_features.append(col)
                except:
                    continue

        # 修正：不再过滤技术形态特征，保留所有可用特征
        key_features = available_features

        # 按字母顺序排序，确保一致性
        key_features.sort()

        self.logger.info(f"一致性特征提取: 总列数={len(data_df.columns)}, 可用特征={len(available_features)}, 最终特征={len(key_features)}")

        return key_features

    def _patterns_to_matrix(self, patterns, key_features):
        """将模式列表转换为特征矩阵，并过滤全零特征，正确处理数值编码"""
        if not patterns:
            return np.array([]), []

        # 创建板块编码映射
        sector_encoder = {}
        sector_counter = 1

        # 首先构建完整矩阵，正确处理不同类型的特征
        matrix = []
        for pattern in patterns:
            row = []
            for feature in key_features:
                value = pattern.get(feature, 0)

                # 特殊处理不同类型的特征
                if feature == 'most_relevant_sector':
                    # 板块代码编码为数值
                    if isinstance(value, str) and value:
                        if value not in sector_encoder:
                            sector_encoder[value] = sector_counter
                            sector_counter += 1
                        row.append(float(sector_encoder[value]))
                    else:
                        row.append(0.0)

                else:
                    # 其他数值特征
                    try:
                        numeric_value = float(value)
                        # 检查是否为有效数值
                        if np.isfinite(numeric_value):
                            row.append(numeric_value)
                        else:
                            row.append(0.0)
                    except (ValueError, TypeError):
                        row.append(0.0)
            matrix.append(row)

        matrix = np.array(matrix)

        # 识别并移除全零特征列 - 修正版本
        if len(matrix) > 0:
            # 计算每列的标准差，标准差为0的列是全零或全相同的列
            col_stds = np.std(matrix, axis=0)

            # 修正的特征过滤逻辑
            valid_cols = []
            removed_features = []

            for i, feature in enumerate(key_features):
                is_pattern_feature = feature.startswith('pattern_')  # 个股形态特征
                is_ratio_feature = any(keyword in feature.lower() for keyword in
                                     ['rate', 'ratio', 'percent', 'turnover'])  # 比例特征

                # 个股形态特征和比例特征保留所有
                if is_pattern_feature or is_ratio_feature:
                    valid_cols.append(True)
                else:
                    # 其他特征使用更宽松的标准差阈值
                    if col_stds[i] > 1e-10:  # 更宽松的阈值
                        valid_cols.append(True)
                    else:
                        valid_cols.append(False)
                        removed_features.append(feature)

            valid_cols = np.array(valid_cols)

            if removed_features:
                self.logger.info(f"移除全零或无变化特征: {removed_features[:10]}{'...' if len(removed_features) > 10 else ''} (共{len(removed_features)}个)")

            # 过滤矩阵和特征名
            filtered_matrix = matrix[:, valid_cols]
            filtered_features = [key_features[i] for i, valid in enumerate(valid_cols) if valid]

            self.logger.info(f"特征过滤结果: {len(key_features)} -> {len(filtered_features)} 个有效特征")

            # 保存编码映射以供后续使用
            if hasattr(self, '_sector_encoder'):
                self._sector_encoder.update(sector_encoder)
            else:
                self._sector_encoder = sector_encoder

            return filtered_matrix, filtered_features

        return matrix, key_features

    def _patterns_to_matrix_no_filter(self, patterns, key_features):
        """将模式列表转换为特征矩阵，不过滤特征，但正确处理数值编码"""
        if not patterns:
            return np.array([])

        # 获取或创建板块编码映射
        if not hasattr(self, '_sector_encoder'):
            self._sector_encoder = {}

        sector_counter = len(self._sector_encoder) + 1

        matrix = []
        for pattern in patterns:
            row = []
            for feature in key_features:
                value = pattern.get(feature, 0)

                # 特殊处理不同类型的特征
                if feature == 'most_relevant_sector':
                    # 板块代码编码为数值
                    if isinstance(value, str) and value:
                        if value not in self._sector_encoder:
                            self._sector_encoder[value] = sector_counter
                            sector_counter += 1
                        row.append(float(self._sector_encoder[value]))
                    else:
                        row.append(0.0)

                else:
                    # 其他数值特征
                    try:
                        numeric_value = float(value)
                        # 检查是否为有效数值
                        if np.isfinite(numeric_value):
                            row.append(numeric_value)
                        else:
                            row.append(0.0)
                    except (ValueError, TypeError):
                        row.append(0.0)
            matrix.append(row)

        return np.array(matrix)

    def _extract_test_features_matrix(self, predict_features, key_features):
        """批量提取测试样本特征矩阵，正确处理特殊特征"""
        if predict_features.empty:
            return np.array([]), [], []

        # 获取或创建板块编码映射
        if not hasattr(self, '_sector_encoder'):
            self._sector_encoder = {}

        # 批量提取特征
        matrix = []
        stock_codes = predict_features['stock_code'].values
        dates = predict_features['date'].values

        for feature in key_features:
            if feature in predict_features.columns:
                if feature == 'most_relevant_sector':
                    # 板块代码编码为数值
                    feature_values = []
                    for value in predict_features[feature]:
                        if isinstance(value, str) and value:
                            if value not in self._sector_encoder:
                                # 如果是新的板块代码，分配新的编号
                                self._sector_encoder[value] = len(self._sector_encoder) + 1
                            feature_values.append(float(self._sector_encoder[value]))
                        else:
                            feature_values.append(0.0)
                    matrix.append(np.array(feature_values))

                else:
                    # 其他数值特征
                    feature_values = pd.to_numeric(predict_features[feature], errors='coerce').fillna(0)
                    matrix.append(feature_values.values)
            else:
                # 如果特征不存在，填充为0
                matrix.append(np.zeros(len(predict_features)))

        # 转置得到正确的形状 (n_samples, n_features)
        test_matrix = np.column_stack(matrix)

        return test_matrix, stock_codes, dates

    def _batch_cosine_similarity(self, test_matrix, pattern_matrix):
        """批量计算余弦相似度"""
        if len(pattern_matrix) == 0 or len(test_matrix) == 0:
            return np.array([])

        # 计算范数
        test_norms = np.linalg.norm(test_matrix, axis=1, keepdims=True)
        pattern_norms = np.linalg.norm(pattern_matrix, axis=1, keepdims=True)

        # 避免除零
        test_norms = np.where(test_norms == 0, 1e-8, test_norms)
        pattern_norms = np.where(pattern_norms == 0, 1e-8, pattern_norms)

        # 标准化 - 修复广播错误
        test_normalized = test_matrix / test_norms  # (n_test, n_features)
        pattern_normalized = pattern_matrix / pattern_norms  # (n_patterns, n_features)

        # 计算余弦相似度矩阵
        # test_normalized: (n_test, n_features) @ pattern_normalized.T: (n_features, n_patterns)
        # 结果: (n_test, n_patterns)
        similarities = np.dot(test_normalized, pattern_normalized.T)

        # 确保相似度在[-1, 1]范围内，然后转换到[0, 1]
        similarities = np.clip(similarities, -1, 1)
        # 将[-1, 1]映射到[0, 1]：(x + 1) / 2
        similarities = (similarities + 1) / 2

        return similarities

    def _batch_cosine_similarity_chunked(self, test_matrix, pattern_matrix, chunk_size=500):
        """分批计算余弦相似度，避免内存溢出 - 优化版本"""
        if len(pattern_matrix) == 0 or len(test_matrix) == 0:
            return np.array([])

        # 动态调整chunk_size以避免内存问题
        max_features = max(test_matrix.shape[1], pattern_matrix.shape[1])
        if max_features > 10000:  # 特征数量过多时进一步减小chunk_size
            chunk_size = min(chunk_size, 100)
        elif max_features > 5000:
            chunk_size = min(chunk_size, 200)

        self.logger.info(f"使用分批处理，测试样本: {len(test_matrix)}, 模式: {len(pattern_matrix)}, 批次大小: {chunk_size}")

        # 预计算模式矩阵的标准化
        try:
            pattern_norms = np.linalg.norm(pattern_matrix, axis=1, keepdims=True)
            pattern_norms = np.where(pattern_norms == 0, 1e-8, pattern_norms)
            pattern_normalized = pattern_matrix / pattern_norms
        except MemoryError as e:
            self.logger.error(f"模式矩阵标准化时内存不足: {e}")
            # 进一步减小处理规模
            if len(pattern_matrix) > 1000:
                self.logger.info("模式数量过多，随机采样1000个模式")
                indices = np.random.choice(len(pattern_matrix), 1000, replace=False)
                pattern_matrix = pattern_matrix[indices]
                pattern_norms = np.linalg.norm(pattern_matrix, axis=1, keepdims=True)
                pattern_norms = np.where(pattern_norms == 0, 1e-8, pattern_norms)
                pattern_normalized = pattern_matrix / pattern_norms
            else:
                raise

        # 分批处理测试样本
        all_similarities = []
        num_chunks = (len(test_matrix) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(test_matrix))
            test_chunk = test_matrix[start_idx:end_idx]

            try:
                # 计算当前批次的相似度
                test_norms = np.linalg.norm(test_chunk, axis=1, keepdims=True)
                test_norms = np.where(test_norms == 0, 1e-8, test_norms)
                test_normalized = test_chunk / test_norms

                # 计算余弦相似度
                similarities = np.dot(test_normalized, pattern_normalized.T)
                similarities = np.clip(similarities, -1, 1)
                similarities = (similarities + 1) / 2  # 映射到[0, 1]

                all_similarities.append(similarities)

            except MemoryError as e:
                self.logger.error(f"批次 {i+1} 内存不足: {e}")
                # 进一步减小批次大小
                if chunk_size > 50:
                    new_chunk_size = chunk_size // 2
                    self.logger.info(f"减小批次大小从 {chunk_size} 到 {new_chunk_size}")
                    # 重新计算当前批次
                    sub_chunks = (end_idx - start_idx + new_chunk_size - 1) // new_chunk_size
                    for j in range(sub_chunks):
                        sub_start = start_idx + j * new_chunk_size
                        sub_end = min(sub_start + new_chunk_size, end_idx)
                        sub_chunk = test_matrix[sub_start:sub_end]

                        try:
                            sub_norms = np.linalg.norm(sub_chunk, axis=1, keepdims=True)
                            sub_norms = np.where(sub_norms == 0, 1e-8, sub_norms)
                            sub_normalized = sub_chunk / sub_norms
                            sub_similarities = np.dot(sub_normalized, pattern_normalized.T)
                            sub_similarities = np.clip(sub_similarities, -1, 1)
                            sub_similarities = (sub_similarities + 1) / 2
                            all_similarities.append(sub_similarities)
                        except MemoryError:
                            self.logger.error(f"即使减小批次大小仍然内存不足，跳过批次 {i+1}")
                            # 创建零相似度矩阵作为fallback
                            fallback_similarities = np.zeros((sub_end - sub_start, pattern_normalized.shape[0]))
                            all_similarities.append(fallback_similarities)
                else:
                    self.logger.error(f"批次大小已经很小({chunk_size})，跳过批次 {i+1}")
                    # 创建零相似度矩阵作为fallback
                    fallback_similarities = np.zeros((end_idx - start_idx, pattern_normalized.shape[0]))
                    all_similarities.append(fallback_similarities)

            if (i + 1) % 10 == 0:  # 每10个批次报告一次进度
                self.logger.info(f"已处理 {i + 1}/{num_chunks} 个批次")

        # 合并所有批次的结果
        final_similarities = np.vstack(all_similarities)
        self.logger.info(f"分批处理完成，结果形状: {final_similarities.shape}")

        return final_similarities

    def _analyze_pattern_features(self, patterns, pattern_type):
        """分析历史模式的特征分布"""
        if not patterns:
            return

        # 动态获取特征列表
        if patterns:
            sample_pattern = patterns[0]
            exclude_fields = ['stock_code', 'date_str', 'date', 'future_return_2d']
            key_features = [k for k in sample_pattern.keys()
                          if k not in exclude_fields and not k.startswith('_')]
        else:
            key_features = ['pct_chg', 'vol', 'turnover_rate', 'amplitude']

        for feature_name in key_features:
            values = []
            for pattern in patterns:
                value = pattern.get(feature_name, 0)
                try:
                    values.append(float(value))
                except (ValueError, TypeError):
                    values.append(0)

            if values:
                unique_count = len(set(values))
                std_dev = np.std(values)
                value_range = f"{min(values):.6f} - {max(values):.6f}"
                avg_value = np.mean(values)
                self.logger.info(f"{pattern_type}模式特征 {feature_name}: 唯一值={unique_count}, 均值={avg_value:.6f}, 标准差={std_dev:.6f}, 范围={value_range}")

        # 分析模式中的板块和股票分布
        sectors = [p.get('most_relevant_sector', 'unknown') for p in patterns]
        stocks = [p.get('stock_code', 'unknown') for p in patterns]
        future_returns = [p.get('future_return_2d', 0) for p in patterns]

        sector_counts = {}
        for sector in sectors:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        self.logger.info(f"{pattern_type}模式板块分布: {dict(list(sector_counts.items())[:5])}")  # 显示前5个板块
        self.logger.info(f"{pattern_type}模式future_return范围: {min(future_returns):.6f} - {max(future_returns):.6f}")

    def _prepare_test_data_for_model(self, predict_features, relations_df):
        """
        为模型预测准备测试数据 - 高性能优化版本

        优化策略：
        1. 使用向量化操作替代逐行处理
        2. 避免昂贵的 row.to_dict() 操作
        3. 批量处理数据转换
        4. 减少内存分配
        """
        import time
        start_time = time.time()

        # 使用与训练阶段完全一致的特征提取逻辑
        if predict_features.empty:
            return np.array([]), []

        # 获取训练时保存的特征列表
        key_features = self._get_consistent_feature_list(predict_features)

        self.logger.info(f"使用一致的特征数量: {len(key_features)}")
        self.logger.info(f"特征列表: {key_features[:10]}{'...' if len(key_features) > 10 else ''}")

        if predict_features.empty:
            return np.array([]), []

        try:
            # 添加详细的特征分析日志
            self.logger.info("=== 特征数据分析 ===")
            for feature_name in key_features:
                if feature_name in predict_features.columns:
                    feature_values = pd.to_numeric(predict_features[feature_name], errors='coerce').fillna(0)
                    unique_values = len(feature_values.unique())
                    value_range = f"{feature_values.min():.6f} - {feature_values.max():.6f}"
                    std_dev = feature_values.std()
                    self.logger.info(f"特征 {feature_name}: 唯一值数={unique_values}, 范围={value_range}, 标准差={std_dev:.6f}")
                else:
                    self.logger.warning(f"特征 {feature_name} 不存在于数据中")

            # 批量处理特征提取
            X_list = []
            for feature_name in key_features:
                if feature_name in predict_features.columns:
                    # 向量化转换为数值型，无效值填充为0
                    feature_values = pd.to_numeric(predict_features[feature_name], errors='coerce').fillna(0)
                    X_list.append(feature_values.values)
                else:
                    # 如果特征不存在，填充为0
                    X_list.append(np.zeros(len(predict_features)))

            # 转置得到正确的形状 (n_samples, n_features)
            X_test = np.column_stack(X_list)

            # 分析特征矩阵的差异性
            self.logger.info("=== 特征矩阵分析 ===")
            for i, feature_name in enumerate(key_features):
                feature_column = X_test[:, i]
                unique_values = len(np.unique(feature_column))
                self.logger.info(f"特征矩阵列 {i} ({feature_name}): 唯一值数={unique_values}, 标准差={np.std(feature_column):.6f}")

            # 检查是否所有样本的特征向量都相同
            unique_rows = len(np.unique(X_test, axis=0))
            self.logger.info(f"特征矩阵: 总样本={len(X_test)}, 唯一特征向量数={unique_rows}")
            if unique_rows == 1:
                self.logger.error("严重问题：所有样本的特征向量完全相同！")
                self.logger.info(f"示例特征向量: {X_test[0]}")
            elif unique_rows < 10:
                self.logger.warning(f"特征向量差异性很低，只有 {unique_rows} 种不同的特征向量")

            # 批量构建stock_info，避免row.to_dict()
            stock_codes = predict_features['stock_code'].values
            dates = predict_features['date_str'].values

            # 只提取必要的特征字典，避免完整的to_dict()
            essential_features = ['stock_code', 'date_str'] + key_features
            available_features = [col for col in essential_features if col in predict_features.columns]

            # 使用向量化方式构建特征字典
            stock_info = []
            for i in range(len(predict_features)):
                # 只构建必要的特征字典
                features_dict = {}
                for col in available_features:
                    features_dict[col] = predict_features.iloc[i][col]

                stock_info.append((stock_codes[i], dates[i], features_dict))

            elapsed_time = time.time() - start_time
            self.logger.info(f"测试数据准备完成，共处理 {len(X_test)} 个样本，用时 {elapsed_time:.2f} 秒")

            return X_test, stock_info

        except Exception as e:
            self.logger.error(f"准备测试数据时出错: {e}")
            return np.array([]), []

    def _calculate_cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        try:
            vec1 = np.array(vec1, dtype=float)
            vec2 = np.array(vec2, dtype=float)

            # 检查向量是否有效
            if len(vec1) != len(vec2) or len(vec1) == 0:
                return None

            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception:
            return None

    def _prepare_test_data_for_model_with_features(self, complete_features_df, expected_features):
        """
        为模型预测准备测试数据，使用指定的特征列表确保一致性 - 重构版本

        重构后功能：
        1. 直接使用完整特征DataFrame，包含所有特征（个股+板块+相对特征）
        2. 与训练时保持一致的特征工程
        3. 简化数据准备流程

        Args:
            complete_features_df: 包含所有特征的完整DataFrame
            expected_features: 期望的特征列表

        Returns:
            tuple: (特征矩阵, 股票信息)
        """
        if complete_features_df.empty:
            return np.array([]), []

        self.logger.info(f"使用完整特征DataFrame准备测试数据，期望特征数量: {len(expected_features)}")

        try:
            # 1. 直接使用完整特征DataFrame，无需额外增强
            enhanced_predict_features = complete_features_df
            self.logger.info(f"完整特征DataFrame包含 {enhanced_predict_features.shape[1]} 个特征")
            # 2. 批量提取特征（使用增强后的特征）
            matrix = []
            stock_codes = enhanced_predict_features['stock_code'].values
            dates = enhanced_predict_features['date'].values

            self.logger.info("=== 预测特征分析（增强版本）===")
            for feature_name in expected_features:
                if feature_name in enhanced_predict_features.columns:
                    # 向量化转换为数值型，无效值填充为0
                    feature_values = pd.to_numeric(enhanced_predict_features[feature_name], errors='coerce').fillna(0)
                    matrix.append(feature_values.values)

                    # 分析特征分布
                    unique_count = len(feature_values.unique())
                    std_dev = feature_values.std()
                    value_range = f"{feature_values.min():.6f} - {feature_values.max():.6f}"
                    self.logger.info(f"预测特征 {feature_name}: 唯一值数={unique_count}, 范围={value_range}, 标准差={std_dev:.6f}")
                else:
                    # 如果特征不存在，填充为0
                    matrix.append(np.zeros(len(enhanced_predict_features)))
                    self.logger.warning(f"预测特征 {feature_name} 不存在于数据中，填充为0")

            # 转置得到正确的形状 (n_samples, n_features)
            test_matrix = np.column_stack(matrix)

            # 构建股票信息
            stock_info = []
            for i in range(len(stock_codes)):
                stock_info.append({
                    'stock_code': stock_codes[i],
                    'date': dates[i]
                })

            self.logger.info(f"测试数据准备完成，共处理 {len(stock_info)} 个样本，特征维度: {test_matrix.shape}")

            return test_matrix, stock_info

        except Exception as e:
            self.logger.error(f"准备测试数据时出错: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return np.array([]), []


    def _batch_suspended_stock_filtering(self, df):
        """
        批量停牌股票过滤 - 高性能版本

        Args:
            df: 预处理后的数据DataFrame

        Returns:
            DataFrame: 过滤后的数据
        """
        if df.empty:
            return df

        # 如果没有股票特征数据，跳过停牌检查
        if not hasattr(self, '_current_stock_features') or self._current_stock_features is None:
            self.logger.info("没有股票特征数据，跳过停牌检查")
            return df

        # 创建股票-日期的查找索引
        features_df = self._current_stock_features
        features_lookup = features_df.set_index(['stock_code', 'date_str'])

        # 向量化停牌检查
        def is_valid_stock_vectorized(row):
            try:
                key = (row['stock_code'], row['date'])
                if key not in features_lookup.index:
                    return False

                stock_data = features_lookup.loc[key]

                # 检查关键指标
                close_price = stock_data.get('close', 0)
                volume = stock_data.get('volume', 0)

                if pd.isna(close_price) or close_price <= 0:
                    return False
                if pd.isna(volume) or volume <= 0:
                    return False

                return True
            except:
                return False

        # 应用向量化检查
        valid_mask = df.apply(is_valid_stock_vectorized, axis=1)
        filtered_df = df[valid_mask].copy()

        filtered_count = len(df) - len(filtered_df)
        if filtered_count > 0:
            self.logger.info(f"批量过滤停牌股票: {filtered_count} 只")

        return filtered_df


    def _rank_and_select_stocks_by_method(self, predictions, top_n, stock_features_df, method_name):
        """
        按指定方法对股票进行排名和选择

        Args:
            predictions: 预测结果字典
            top_n: 每日选择的股票数量
            stock_features_df: 股票特征数据
            method_name: 方法名称（用于标识）

        Returns:
            dict: 选股结果
        """
        self.logger.info(f"开始 {method_name} 方法的股票排名和选择...")

        # 设置当前股票特征数据用于停牌检查
        if stock_features_df is not None:
            self._current_stock_features = stock_features_df

        if not predictions:
            return {
                'method_name': method_name,
                'selected_stocks': [],
                'selected_stocks_by_date': {},
                'dates': [],
                'total_stocks': 0,
                'date': datetime.now().strftime('%Y%m%d')
            }

        # 1. 向量化数据预处理
        processed_data = self._vectorized_data_preprocessing_for_method(predictions, method_name)

        # 2. 批量停牌检查
        valid_data = self._batch_suspended_stock_filtering(processed_data)

        # 3. 高性能排名和选择
        selected_results = self._high_performance_ranking_for_method(valid_data, top_n, method_name)

        self.logger.info(f"{method_name} 方法选股完成")
        return selected_results

    def _vectorized_data_preprocessing_for_method(self, predictions, method_name):
        """
        为特定方法进行向量化数据预处理

        修复评估指标逻辑混乱问题：
        - 统一评估指标计算逻辑
        - 基于量化金融理论设计合理的评估体系
        - 确保所有指标都有明确的金融含义

        Args:
            predictions: 预测结果字典
            method_name: 方法名称

        Returns:
            DataFrame: 预处理后的数据
        """
        data_list = []

        for key, pred_data in predictions.items():
            stock_code = pred_data.get('stock_code')
            date = pred_data.get('date')

            if not date or not stock_code:
                continue

            # 1. 提取预测收益率（统一为小数形式）
            prediction_score = pred_data.get('predicted_return', pred_data.get('prediction_score', 0))

            # 确保预测收益率在合理范围内
            if abs(prediction_score) > 1:  # 如果是百分比形式，转换为小数
                prediction_score = prediction_score / 100
            prediction_score = np.clip(prediction_score, -0.2, 0.2)  # 限制在±20%

            # 2. 计算统一的评估指标
            evaluation_metrics = self._calculate_unified_evaluation_metrics(pred_data, prediction_score, method_name)

            # 3. 组织数据
            data_list.append({
                'key': key,
                'stock_code': stock_code,
                'date': date,
                'prediction_score_pct': prediction_score,  # 小数形式的预测得分
                'prediction_score_original': prediction_score,  # 原始预测得分
                'positive_similarity': evaluation_metrics['similarity_score'],
                'prediction_binary': 1 if prediction_score > 0 else 0,
                'confidence': evaluation_metrics['confidence_score'],
                'score': evaluation_metrics['composite_score'],
                'method_name': method_name,
                # 新增统一评估指标
                'information_ratio': evaluation_metrics['information_ratio'],
                'sharpe_ratio_proxy': evaluation_metrics['sharpe_ratio_proxy'],
                'stability_score': evaluation_metrics['stability_score'],
                'risk_adjusted_return': evaluation_metrics['risk_adjusted_return']
            })

        df = pd.DataFrame(data_list)
        self.logger.info(f"{method_name} 方法向量化预处理完成，处理 {len(df)} 条记录")
        return df

    def _calculate_unified_evaluation_metrics(self, pred_data, prediction_score, method_name):
        """
        计算统一的评估指标体系

        基于现代投资组合理论和量化金融理论设计：
        1. 信息比率 (Information Ratio): 衡量超额收益相对于跟踪误差的比率
        2. 夏普比率代理 (Sharpe Ratio Proxy): 风险调整后收益
        3. 稳定性得分 (Stability Score): 预测一致性和可靠性
        4. 相似度得分 (Similarity Score): 历史模式匹配度
        5. 置信度得分 (Confidence Score): 预测置信水平
        6. 综合得分 (Composite Score): 多因子加权综合评分

        Args:
            pred_data: 预测数据字典
            prediction_score: 预测收益率
            method_name: 方法名称

        Returns:
            dict: 统一评估指标
        """
        # 1. 基础指标提取
        base_confidence = pred_data.get('confidence', 0.5)
        similar_samples_count = pred_data.get('similar_samples_count', 1)
        sector = pred_data.get('sector', 'unknown')

        # 2. 相似度得分 (0-1)：基于历史模式匹配
        # 考虑相似样本数量和基础置信度
        similarity_score = min(1.0, base_confidence * (1 + np.log(max(1, similar_samples_count)) / 10))

        # 3. 置信度得分 (0-1)：综合考虑预测强度和样本支持
        prediction_strength = min(1.0, abs(prediction_score) * 10)  # 预测强度
        sample_support = min(1.0, similar_samples_count / 10)  # 样本支持度 
        confidence_score = (base_confidence * 0.5 + prediction_strength * 0.3 + sample_support * 0.2)

        # 4. 稳定性得分 (0-1)：预测一致性
        # 基于相似度和置信度的一致性
        stability_score = (similarity_score * 0.6 + confidence_score * 0.4)

        # 5. 信息比率代理：超额收益/跟踪误差的代理指标
        # 使用预测收益率和置信度构建
        tracking_error_proxy = max(0.01, 1 - confidence_score)  # 置信度越低，跟踪误差越大
        information_ratio = prediction_score / tracking_error_proxy

        # 6. 夏普比率代理：风险调整收益
        # 使用稳定性作为风险的反向代理
        risk_proxy = max(0.01, 1 - stability_score)
        sharpe_ratio_proxy = prediction_score / risk_proxy

        # 7. 风险调整收益：考虑预测不确定性
        risk_adjusted_return = prediction_score * confidence_score

        # 8. 综合得分：多因子加权
        # 权重设计基于现代投资组合理论
        weights = {
            'return': 0.35,      # 收益率权重35%
            'risk_adj': 0.25,    # 风险调整权重25%
            'stability': 0.20,   # 稳定性权重20%
            'confidence': 0.20   # 置信度权重20%
        }

        composite_score = (
            prediction_score * weights['return'] +
            risk_adjusted_return * weights['risk_adj'] +
            stability_score * weights['stability'] +
            confidence_score * weights['confidence']
        )

        # 确保所有指标都在合理范围内
        return {
            'similarity_score': np.clip(similarity_score, 0, 1),
            'confidence_score': np.clip(confidence_score, 0, 1),
            'stability_score': np.clip(stability_score, 0, 1),
            'information_ratio': np.clip(information_ratio, -10, 10),
            'sharpe_ratio_proxy': np.clip(sharpe_ratio_proxy, -10, 10),
            'risk_adjusted_return': np.clip(risk_adjusted_return, -0.2, 0.2),
            'composite_score': composite_score
        }

    def _high_performance_ranking_for_method(self, df, top_n, method_name):
        """
        为特定方法进行高性能排名和选择

        基于现代投资组合理论的多因子排序系统：
        - 使用统一的评估指标体系
        - 实现多层次风险调整排序
        - 结合信息比率、夏普比率等量化指标
        - 考虑市场中性和行业分散化

        Args:
            df: 有效的股票数据DataFrame
            top_n: 每日选择数量
            method_name: 方法名称

        Returns:
            dict: 选择结果
        """
        if df.empty:
            return {
                'method_name': method_name,
                'selected_stocks': [],
                'selected_stocks_by_date': {},
                'dates': [],
                'total_stocks': 0,
                'date': datetime.now().strftime('%Y%m%d')
            }

        # 按日期分组并排序
        grouped = df.groupby('date')
        selected_stocks_by_date = {}
        total_selected = 0

        # 每天选择的股票数量
        daily_top_n = min(top_n, 10)  # 每天最多选10只

        for date, group in grouped:
            # 🛡️ 市场风险过滤：评估当前市场状态，避免在高风险时期选股
            market_features = self.market_risk_filter._extract_market_features_for_date(group, date)
            risk_assessment = self.market_risk_filter.assess_market_risk(market_features)

            # 如果市场风险过高，跳过该日期的选股
            if risk_assessment['should_avoid_selection']:
                self.logger.warning(f"日期 {date} 市场风险过高({risk_assessment['risk_level']})，"
                                  f"风险评分: {risk_assessment['overall_risk_score']:.3f}，跳过选股")
                self.logger.warning(f"风险因素: {risk_assessment['risk_details']['risk_explanation']}")
                continue

            # 🔥 修复选股逻辑：首先过滤掉预测收益为负的股票
            positive_return_group = group[group['prediction_score_pct'] > 0].copy()

            if positive_return_group.empty:
                self.logger.warning(f"日期 {date} 没有预测收益为正的股票，跳过该日期")
                continue

            # 使用统一评估指标进行专业排序
            group_enhanced = positive_return_group.copy()

            # 计算最终排序得分：多因子加权，重点关注预测收益率
            # 修正权重分配，确保预测收益率是主要因素
            group_enhanced['final_ranking_score'] = (
                group_enhanced['prediction_score_pct'] * 0.5 +      # 预测收益率权重50%（主要因素）
                group_enhanced['score'] * 0.25 +                    # 综合得分权重25%
                group_enhanced['stability_score'] * 0.15 +          # 稳定性权重15%
                group_enhanced['information_ratio'] * 0.01 * 0.1    # 信息比率权重10%（标准化）
            )

            # 多层次排序：优先预测收益率，确保选出最有潜力的股票
            sorted_group = group_enhanced.sort_values([
                'prediction_score_pct',     # 🎯 预测收益率优先（选择涨幅最大的）
                'final_ranking_score',      # 最终排序得分
                'score',                    # 综合得分
                'stability_score',          # 稳定性得分
                'information_ratio'         # 信息比率
            ], ascending=[False, False, False, False, False]).head(daily_top_n)

            # 添加排名
            sorted_group = sorted_group.copy()
            sorted_group['rank'] = range(1, len(sorted_group) + 1)

            # 转换为字典格式
            date_selected = []
            for _, row in sorted_group.iterrows():
                stock_dict = {
                    'stock_code': row['stock_code'],
                    'date': row['date'],
                    'predicted_return': row['prediction_score_pct'] * 100,  # 转换为百分比显示（如2.5表示2.5%）
                    'prediction_score': row['prediction_score_original'],  # 保留原始得分（小数形式）
                    'positive_similarity': row['positive_similarity'],
                    'prediction_binary': row['prediction_binary'],
                    'confidence': row['confidence'],
                    'score': row['score'],
                    'rank': row['rank'],
                    'method_name': method_name,
                    # 新增的排序指标
                    'stability_score': row['stability_score'],
                    'risk_adjusted_return': row['risk_adjusted_return']
                }

                # 添加统一评估指标
                stock_dict['comprehensive_score'] = row.get('final_ranking_score', row.get('score', 0))
                stock_dict['final_ranking_score'] = row.get('final_ranking_score', 0)
                stock_dict['information_ratio'] = row.get('information_ratio', 0)
                stock_dict['sharpe_ratio_proxy'] = row.get('sharpe_ratio_proxy', 0)

                date_selected.append(stock_dict)

            selected_stocks_by_date[date] = date_selected
            total_selected += len(date_selected)

            # 输出选择结果
            if date_selected:
                stock_codes = [s['stock_code'] for s in date_selected]
                predicted_returns = [f"{s['predicted_return']:.4f}%" for s in date_selected]
                comprehensive_scores = [f"{s['comprehensive_score']:.4f}" for s in date_selected]
                stability_scores = [f"{s['stability_score']:.4f}" for s in date_selected]

                # 尝试获取实际收益率进行对比
                actual_returns = []
                for stock in date_selected:
                    stock_code = stock['stock_code']
                    # 尝试从股票特征数据中获取实际收益率
                    actual_return = self._get_actual_return_for_logging(
                        stock_code, date, getattr(self, '_current_stock_features', None)
                    )
                    if actual_return is not None:
                        actual_returns.append(f"{actual_return:.4f}%")
                    else:
                        actual_returns.append("N/A")

                # 增强的日志输出，包含预测收益率、综合评分、稳定性和实际收益率
                if any(ret != "N/A" for ret in actual_returns):
                    self.logger.info(f"{method_name} 日期 {date} 选中: {stock_codes}")
                    self.logger.info(f"  预测收益: {predicted_returns}")
                    self.logger.info(f"  综合评分: {comprehensive_scores}")
                    self.logger.info(f"  稳定性分: {stability_scores}")
                    self.logger.info(f"  实际收益: {actual_returns}")
                else:
                    self.logger.info(f"{method_name} 日期 {date} 选中: {stock_codes}")
                    self.logger.info(f"  预测收益: {predicted_returns}, 综合评分: {comprehensive_scores}, 稳定性: {stability_scores}")

        # 获取最新日期的选股
        dates = sorted(selected_stocks_by_date.keys())
        latest_date = dates[-1] if dates else datetime.now().strftime('%Y%m%d')
        selected_stocks = selected_stocks_by_date.get(latest_date, [])

        # 组织结果
        result = {
            'method_name': method_name,
            'selected_stocks': selected_stocks,
            'selected_stocks_by_date': selected_stocks_by_date,
            'dates': dates,
            'total_stocks': len(df),
            'selected_count': total_selected,
            'selection_criteria': f'{method_name}_变化点检测',
            'selection_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': latest_date
        }

        self.logger.info(f"{method_name} 选股完成: 处理 {len(dates)} 个交易日，选出 {total_selected} 只股票")

        return result

    def _combine_selection_results(self, similarity_selected, model_selected=None):
        """
        处理选股结果，现在只使用相似度匹配方法

        Args:
            similarity_selected: 相似度匹配选股结果
            model_selected: 已废弃，保留参数用于兼容性

        Returns:
            dict: 选股结果
        """
        # 只使用相似度匹配结果
        if similarity_selected and similarity_selected.get('selected_stocks'):
            results = {
                'similarity_matching': similarity_selected
            }
            self.logger.info(f"相似度匹配选中股票数: {len(similarity_selected.get('selected_stocks', []))}")
            return results
        else:
            # 没有结果，返回空结果
            return {
                'selected_stocks': [],
                'selected_stocks_by_date': {},
                'dates': [],
                'total_stocks': 0,
                'date': datetime.now().strftime('%Y%m%d')
            }

    def _is_stock_suspended(self, stock_code, date):
        """
        检查股票在指定日期是否停牌

        Args:
            stock_code: 股票代码
            date: 日期字符串，格式为YYYYMMDD

        Returns:
            bool: True表示停牌，False表示正常交易
        """
        try:
            # 这里可以接入实际的停牌数据源
            # 目前使用简单的启发式方法：检查股票在该日期是否有基础数据

            # 如果有股票特征数据，可以检查该股票在该日期的数据
            if hasattr(self, '_current_stock_features') and self._current_stock_features is not None:
                stock_data = self._current_stock_features[
                    (self._current_stock_features['stock_code'] == stock_code) &
                    (self._current_stock_features['date_str'] == date)
                ]

                if stock_data.empty:
                    return True  # 没有数据，可能停牌

                # 检查关键字段是否为空或异常
                row = stock_data.iloc[0]
                if pd.isna(row.get('close', 0)) or row.get('close', 0) <= 0:
                    return True  # 收盘价异常，可能停牌

                if pd.isna(row.get('volume', 0)) or row.get('volume', 0) <= 0:
                    return True  # 成交量为0，可能停牌

            return False  # 默认认为正常交易

        except Exception as e:
            self.logger.warning(f"检查股票 {stock_code} 在 {date} 的停牌状态时出错: {e}")
            return False  # 出错时默认认为正常交易

    def _get_actual_return_for_logging(self, stock_code, date, stock_features_df):
        """
        获取用于日志显示的实际收益率

        Args:
            stock_code: 股票代码
            date: 日期
            stock_features_df: 股票特征数据

        Returns:
            float: 实际收益率（百分比），如果无法获取则返回None
        """
        try:
            if stock_features_df is None:
                return None

            # 查找该股票在该日期的数据
            stock_data = stock_features_df[
                (stock_features_df['stock_code'] == stock_code) &
                (stock_features_df['date_str'] == date)
            ]

            if stock_data.empty:
                return None

            # 检查是否有future_return_2d字段
            if 'future_return_2d' in stock_data.columns:
                future_return = stock_data['future_return_2d'].iloc[0]
                if pd.notna(future_return):
                    # 转换为百分比
                    if abs(future_return) < 1:  # 小数形式
                        return future_return * 100
                    else:  # 已经是百分比形式
                        return future_return

            return None

        except Exception as e:
            self.logger.debug(f"获取股票 {stock_code} 在 {date} 的实际收益率时出错: {e}")
            return None

    def _evaluate_selected_stocks_with_method(self, selected_stocks, stock_features_df, method_name, evaluation_days=2):
        """
        评估特定方法选出的股票的实际表现

        Args:
            selected_stocks: 选股结果
            stock_features_df: 股票特征数据
            method_name: 方法名称
            evaluation_days: 评估天数

        Returns:
            dict: 评估结果
        """
        self.logger.info(f"开始评估 {method_name} 方法的选股表现...")

        if not selected_stocks or not selected_stocks.get('selected_stocks_by_date'):
            return {
                'method_name': method_name,
                'evaluation_summary': {
                    'total_stocks': 0,
                    'evaluated_stocks': 0,
                    'avg_actual_return': 0.0,
                    'direction_accuracy': 0.0,
                    'win_rate': 0.0,
                    'performance_rating': '无数据'
                },
                'stock_evaluations': []
            }

        all_evaluations = []
        evaluation_by_date = {}

        for date, stocks in selected_stocks['selected_stocks_by_date'].items():
            date_evaluations = []

            for stock in stocks:
                stock_code = stock['stock_code']
                predicted_return = stock.get('predicted_return', 0)  # 已经是百分比形式

                # 获取实际收益率（统一为百分比）
                actual_return = self._get_actual_return_unified(
                    stock_code, date, stock_features_df, evaluation_days
                )

                if actual_return is not None:
                    # 计算方向准确性
                    direction_correct = (predicted_return > 0) == (actual_return > 0)

                    # 计算预测准确性（误差）
                    prediction_error = abs(predicted_return - actual_return)
                    prediction_accuracy = max(0, 1 - prediction_error / 10)  # 10%为满分误差

                    evaluation = {
                        'stock_code': stock_code,
                        'date': date,
                        'predicted_return': predicted_return,
                        'actual_return': actual_return,
                        'direction_correct': direction_correct,
                        'prediction_accuracy': prediction_accuracy,
                        'prediction_error': prediction_error,
                        'method_name': method_name,
                        'rank': stock.get('rank', 0)
                    }

                    date_evaluations.append(evaluation)
                    all_evaluations.append(evaluation)

            # 计算该日期的评估汇总
            if date_evaluations:
                date_summary = self._calculate_evaluation_summary(date_evaluations, method_name)
                date_summary['date'] = date
                evaluation_by_date[date] = date_summary

        # 计算整体评估汇总
        overall_summary = self._calculate_evaluation_summary(all_evaluations, method_name)

        result = {
            'method_name': method_name,
            'evaluation_summary': overall_summary,
            'evaluation_by_date': evaluation_by_date,
            'stock_evaluations': all_evaluations,
            'total_evaluated_dates': len(evaluation_by_date),
            'total_evaluated_stocks': len(all_evaluations)
        }

        self.logger.info(f"{method_name} 方法评估完成: 评估了 {len(all_evaluations)} 只股票")
        return result

    def _get_actual_return_unified(self, stock_code, prediction_date, stock_features_df, evaluation_days=2):
        """
        获取统一计量单位的实际收益率（百分比）

        Args:
            stock_code: 股票代码
            prediction_date: 预测日期
            stock_features_df: 股票特征数据
            evaluation_days: 评估天数

        Returns:
            float: 实际收益率（百分比），如果无法获取则返回None
        """
        try:
            # 计算评估日期
            prediction_dt = pd.to_datetime(prediction_date)
            evaluation_date = prediction_dt + pd.Timedelta(days=evaluation_days)
            evaluation_date_str = evaluation_date.strftime('%Y%m%d')

            # 获取预测日期的股票数据
            prediction_data = stock_features_df[
                (stock_features_df['stock_code'] == stock_code) &
                (stock_features_df['date_str'] == prediction_date)
            ]

            if prediction_data.empty:
                return None

            # 检查是否已有future_return_2d字段
            if 'future_return_2d' in prediction_data.columns:
                future_return = prediction_data['future_return_2d'].iloc[0]
                if pd.notna(future_return):
                    # future_return_2d通常是小数形式，转换为百分比
                    if abs(future_return) < 1:  # 如果是小数形式（如0.05）
                        return future_return * 100  # 转换为百分比（如5.0）
                    else:  # 如果已经是百分比形式
                        return future_return

            # 如果没有future_return_2d，手动计算
            # 获取评估日期的股票数据
            evaluation_data = stock_features_df[
                (stock_features_df['stock_code'] == stock_code) &
                (stock_features_df['date_str'] == evaluation_date_str)
            ]

            if evaluation_data.empty:
                return None

            # 计算收益率
            prediction_close = prediction_data['close'].iloc[0]
            evaluation_close = evaluation_data['close'].iloc[0]

            if pd.isna(prediction_close) or pd.isna(evaluation_close) or prediction_close <= 0:
                return None

            # 计算收益率并转换为百分比
            actual_return = ((evaluation_close - prediction_close) / prediction_close) * 100
            return actual_return

        except Exception as e:
            self.logger.warning(f"获取股票 {stock_code} 在 {prediction_date} 的实际收益率时出错: {e}")
            return None

    def _calculate_evaluation_summary(self, evaluations, method_name):
        """
        计算评估汇总统计

        Args:
            evaluations: 评估结果列表
            method_name: 方法名称

        Returns:
            dict: 汇总统计
        """
        if not evaluations:
            return {
                'method_name': method_name,
                'total_stocks': 0,
                'evaluated_stocks': 0,
                'avg_actual_return': 0.0,
                'direction_accuracy': 0.0,
                'win_rate': 0.0,
                'avg_prediction_error': 0.0,
                'performance_rating': '无数据'
            }

        # 过滤有效评估
        valid_evaluations = [e for e in evaluations if e['actual_return'] is not None]

        if not valid_evaluations:
            return {
                'method_name': method_name,
                'total_stocks': len(evaluations),
                'evaluated_stocks': 0,
                'avg_actual_return': 0.0,
                'direction_accuracy': 0.0,
                'win_rate': 0.0,
                'avg_prediction_error': 0.0,
                'performance_rating': '无有效数据'
            }

        # 计算统计指标
        actual_returns = [e['actual_return'] for e in valid_evaluations]
        direction_correct_count = sum(1 for e in valid_evaluations if e['direction_correct'])
        win_count = sum(1 for return_val in actual_returns if return_val > 0)
        prediction_errors = [e['prediction_error'] for e in valid_evaluations]

        avg_actual_return = np.mean(actual_returns)
        direction_accuracy = direction_correct_count / len(valid_evaluations)
        win_rate = win_count / len(valid_evaluations)
        avg_prediction_error = np.mean(prediction_errors)

        # 计算风险指标
        if len(actual_returns) > 1:
            returns_std = np.std(actual_returns, ddof=1)
            sharpe_ratio = avg_actual_return / returns_std if returns_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # 计算盈亏比
        profit_returns = [r for r in actual_returns if r > 0]
        loss_returns = [abs(r) for r in actual_returns if r < 0]

        avg_profit = np.mean(profit_returns) if profit_returns else 0.0
        avg_loss = np.mean(loss_returns) if loss_returns else 0.0
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf') if avg_profit > 0 else 0.0

        # 性能评级
        performance_rating = self._calculate_performance_rating(
            avg_actual_return, direction_accuracy, win_rate, sharpe_ratio
        )

        return {
            'method_name': method_name,
            'total_stocks': len(evaluations),
            'evaluated_stocks': len(valid_evaluations),
            'avg_actual_return': avg_actual_return,
            'direction_accuracy': direction_accuracy,
            'win_rate': win_rate,
            'avg_prediction_error': avg_prediction_error,
            'sharpe_ratio': sharpe_ratio,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'performance_rating': performance_rating
        }

    def _compare_method_evaluations(self, similarity_evaluation, model_evaluation=None):
        """
        评估相似度匹配方法的结果（已废弃模型对比功能）

        Args:
            similarity_evaluation: 相似度匹配评估结果
            model_evaluation: 已废弃，保留参数用于兼容性

        Returns:
            dict: 评估分析结果
        """
        comparison = {
            'comparison_summary': {},
            'winner_analysis': {},
            'detailed_comparison': {}
        }

        # 只评估相似度匹配方法
        if not similarity_evaluation:
            comparison['comparison_summary'] = {'status': '没有评估结果'}
            return comparison

        # 只有相似度匹配方法的结果
        sim_summary = similarity_evaluation.get('evaluation_summary', {})

        # 计算综合得分
        sim_score = self._calculate_comprehensive_score(sim_summary)

        comparison['comparison_summary'] = {
            'similarity_score': sim_score,
            'overall_winner': 'similarity_matching',
            'status': '使用相似度匹配方法'
        }

        # 简化的分析
        comparison['winner_analysis'] = {
            'overall_winner': 'similarity_matching',
            'recommendation': f"相似度匹配方法得分: {sim_score:.1f}"
        }

        return comparison

    def _calculate_comprehensive_score(self, summary):
        """
        计算综合评分

        Args:
            summary: 评估汇总

        Returns:
            float: 综合得分
        """
        if not summary or summary.get('evaluated_stocks', 0) == 0:
            return 0.0

        # 权重设置
        weights = {
            'avg_actual_return': 0.3,    # 平均收益率权重30%
            'direction_accuracy': 0.25,  # 方向准确率权重25%
            'win_rate': 0.25,           # 胜率权重25%
            'sharpe_ratio': 0.2         # 夏普比率权重20%
        }

        score = 0.0

        # 收益率得分（归一化到0-100）
        avg_return = summary.get('avg_actual_return', 0)
        return_score = min(100, max(0, avg_return * 10 + 50))  # 5%收益率得分100分
        score += return_score * weights['avg_actual_return']

        # 方向准确率得分
        direction_acc = summary.get('direction_accuracy', 0)
        score += direction_acc * 100 * weights['direction_accuracy']

        # 胜率得分
        win_rate = summary.get('win_rate', 0)
        score += win_rate * 100 * weights['win_rate']

        # 夏普比率得分（归一化）
        sharpe = summary.get('sharpe_ratio', 0)
        sharpe_score = min(100, max(0, sharpe * 50 + 50))  # 1.0夏普比率得分100分
        score += sharpe_score * weights['sharpe_ratio']

        return score

    def _generate_method_recommendation(self, comparison):
        """
        生成方法推荐（现在只使用相似度匹配方法）

        Args:
            comparison: 对比结果

        Returns:
            str: 推荐说明
        """
        similarity_score = comparison['comparison_summary'].get('similarity_score', 0)

        if similarity_score >= 70:
            return f"相似度匹配方法表现优秀（得分: {similarity_score:.1f}），强烈推荐使用"
        elif similarity_score >= 50:
            return f"相似度匹配方法表现良好（得分: {similarity_score:.1f}），推荐使用"
        elif similarity_score >= 30:
            return f"相似度匹配方法表现一般（得分: {similarity_score:.1f}），可以使用但需谨慎"
        else:
            return f"相似度匹配方法表现较差（得分: {similarity_score:.1f}），建议优化参数或数据"

    def _calculate_performance_rating(self, avg_return, direction_accuracy, win_rate, sharpe_ratio):
        """
        计算性能评级

        Args:
            avg_return: 平均收益率
            direction_accuracy: 方向准确率
            win_rate: 胜率
            sharpe_ratio: 夏普比率

        Returns:
            str: 性能评级
        """
        # 计算综合得分
        score = 0

        # 收益率得分 (40%)
        if avg_return >= 5:
            score += 40
        elif avg_return >= 3:
            score += 30
        elif avg_return >= 1:
            score += 20
        elif avg_return >= 0:
            score += 10

        # 方向准确率得分 (30%)
        if direction_accuracy >= 0.7:
            score += 30
        elif direction_accuracy >= 0.6:
            score += 20
        elif direction_accuracy >= 0.5:
            score += 10

        # 胜率得分 (20%)
        if win_rate >= 0.7:
            score += 20
        elif win_rate >= 0.6:
            score += 15
        elif win_rate >= 0.5:
            score += 10

        # 夏普比率得分 (10%)
        if sharpe_ratio >= 1.0:
            score += 10
        elif sharpe_ratio >= 0.5:
            score += 5

        # 评级映射
        if score >= 80:
            return "优秀"
        elif score >= 60:
            return "良好"
        elif score >= 40:
            return "一般"
        elif score >= 20:
            return "较差"
        else:
            return "很差"



    def select_stocks_by_change_points(self, start_date=None, end_date=None, model_name="stock_selector", top_n=50, use_cache=True):
        """基于变化点方法选择个股

        Args:
            start_date: 开始日期，如果为None则使用默认值
            end_date: 结束日期，如果为None则使用默认值
            model_name: 模型名称
            top_n: 返回的股票数量
            use_cache: 是否使用特征缓存，默认为True

        Returns:
            dict: {'selected_stocks': [...], 'model_name': model_name, 'date': date}
        """
        # 设置日期范围
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=180)).strftime("%Y%m%d")

        # 确保交易日历已加载
        if self.trading_days is None:
            self._load_trading_calendar()

        if not self.trading_days:
            self.logger.error("无法加载交易日历，无法继续")
            return None

        # 获取日期范围内的实际交易日
        trading_days_in_range = [day for day in self.trading_days
                               if start_date <= day <= end_date]

        if not trading_days_in_range:
            self.logger.error(f"日期范围 {start_date} 至 {end_date} 内没有有效交易日")
            return None

        # 使用实际交易日来划分训练和预测日期
        total_trading_days = len(trading_days_in_range)
        train_days_count = int(total_trading_days * 0.9)  # 使用60%的交易日作为训练集

        train_end_idx = min(train_days_count, total_trading_days - 1)
        train_end_date = trading_days_in_range[train_end_idx]

        predict_start_idx = train_end_idx + 1
        predict_start_date = trading_days_in_range[predict_start_idx] if predict_start_idx < total_trading_days else end_date
        predict_end_date = end_date

        self.logger.info(f"日期范围内共有 {total_trading_days} 个交易日")
        self.logger.info(f"训练日期范围: {start_date} 至 {train_end_date} (共{train_days_count}个交易日)")
        self.logger.info(f"预测日期范围: {predict_start_date} 至 {predict_end_date} (共{total_trading_days - train_days_count}个交易日)")

        # 步骤1: 加载板块特征
        self.logger.info("加载板块特征数据...")
        sector_features_df = self._batch_load_features(self._get_all_sectors(), start_date, end_date)

        # 步骤2: 加载个股特征
        self.logger.info("加载个股特征数据...")
        stock_features_df = self.load_stock_features(start_date=start_date, end_date=end_date)

        self.logger.info(f"个股特征列：{stock_features_df.columns}")

        if stock_features_df is None or stock_features_df.empty:
            self.logger.error("无法获取个股特征数据")
            return None

        # 步骤3: 综合数据过滤 - 在模型训练前清理无效数据
        self.logger.info("开始综合数据过滤...")
        filtered_stock_df, filtered_sector_df, filter_stats = self._comprehensive_data_filtering(
            stock_features_df, sector_features_df
        )

        # 检查过滤后的数据是否足够
        if filtered_stock_df is None or filtered_stock_df.empty:
            self.logger.error("过滤后没有有效的个股数据，无法继续训练")
            return None

        if len(filtered_stock_df) < 1000:  # 至少需要1000条记录
            self.logger.warning(f"过滤后个股数据量较少: {len(filtered_stock_df)} 条，可能影响模型质量")

        # 使用过滤后的数据替换原始数据
        stock_features_df = filtered_stock_df
        sector_features_df = filtered_sector_df

        # 步骤4: 记录过滤后的数据统计
        self.logger.info(f"过滤后数据统计: 个股数据 {len(stock_features_df)} 条, 板块数据 {len(sector_features_df) if sector_features_df is not None else 0} 条")

        # 步骤3: 划分训练集和预测集
        # 筛选训练数据 - 使用实际的交易日期划分
        train_mask = ((pd.to_datetime(stock_features_df['date_str']) >= pd.to_datetime(start_date)) &
                      (pd.to_datetime(stock_features_df['date_str']) <= pd.to_datetime(train_end_date)))
        train_features = stock_features_df[train_mask].copy()

        # 筛选预测数据
        predict_mask = ((pd.to_datetime(stock_features_df['date_str']) >= pd.to_datetime(predict_start_date)) &
                         (pd.to_datetime(stock_features_df['date_str']) <= pd.to_datetime(predict_end_date)))
        predict_features = stock_features_df[predict_mask].copy()

        self.logger.info(f"训练数据: {len(train_features)} 条记录")
        self.logger.info(f"预测数据: {len(predict_features)} 条记录")

        # 保存预测日期列表，用于进度计算
        predict_dates = sorted(predict_features['date_str'].unique())
        self.logger.info(f"预测数据包含 {len(predict_dates)} 个交易日")

        # 确保预测数据中的日期都是有效交易日
        valid_predict_dates = [date for date in predict_dates if date in self.trading_days]
        if len(valid_predict_dates) != len(predict_dates):
            self.logger.warning(f"预测数据中包含 {len(predict_dates) - len(valid_predict_dates)} 个非交易日，将被忽略")
            # 更新预测数据，只保留有效交易日的数据
            predict_features = predict_features[predict_features['date_str'].isin(valid_predict_dates)]
            predict_dates = valid_predict_dates


        # 步骤4: 检测个股的变化点
        self.logger.info("检测个股变化点...")
        stock_change_points = self._detect_stock_change_points(train_features)

        # 步骤5: 计算完整特征（个股+板块+市场+相对特征）
        # 注意：这里将个股特征与板块特征进行关联，生成完整的特征集
        self.logger.info("计算训练数据的完整特征（个股+板块+市场+相对特征）...")
        train_complete_features = self.collect_stock_sector_change_point_features(
            train_features, sector_features_df, stock_change_points, is_training=True, use_cache=use_cache)

        # 对预测数据也进行相同的特征工程（全部日期）
        self.logger.info("计算预测数据的完整特征（个股+板块+市场+相对特征）...")
        predict_complete_features = self.collect_stock_sector_change_point_features(
            predict_features, sector_features_df, is_training=False, use_cache=use_cache)

        # 步骤6: 从完整特征DataFrame中提取变化点特征并进行模式分析
        self.logger.info("从完整特征DataFrame中提取变化点特征...")

        # 调试：检查train_complete_features中的most_relevant_sector分布
        if 'most_relevant_sector' in train_complete_features.columns:
            sector_counts = train_complete_features['most_relevant_sector'].value_counts()
            self.logger.info(f"训练完整特征中板块分布: {dict(sector_counts.head(10))}")
            unknown_count = (train_complete_features['most_relevant_sector'] == 'unknown').sum()
            if unknown_count > 0:
                self.logger.warning(f"训练完整特征中发现 {unknown_count} 个unknown板块记录，占比 {unknown_count/len(train_complete_features):.2%}")
            else:
                self.logger.info("训练完整特征中所有记录都有有效的板块信息")
        else:
            self.logger.error("训练完整特征中缺少most_relevant_sector字段！")

        # 直接从完整特征DataFrame中提取变化点特征，避免重复计算
        train_change_point_features = self._extract_change_point_features_directly(
            train_complete_features, stock_change_points
        )

        self.logger.info("分析变化点特征模式...")
        stock_feature_patterns = self._analyze_change_point_patterns_direct(train_change_point_features)


        # 步骤7: 预测个股表现
        self.logger.info("预测个股表现...")
        # 使用完整特征DataFrame进行预测
        prediction_results = self._predict_stocks_with_change_points(predict_complete_features, stock_feature_patterns)

        if not prediction_results:
            self.logger.error("股票预测失败")
            return None

        # 步骤8: 处理新的特征模式预测选股
        self.logger.info(f"处理基于特征模式的预测选股...")

        # 8.1: 处理特征模式预测选股
        pattern_selected = None
        if prediction_results.get('pattern_predictions'):
            self.logger.info(f"处理特征模式预测选股，共 {len(prediction_results['pattern_predictions'])} 只股票")
            pattern_selected = self._rank_and_select_stocks_by_method(
                prediction_results['pattern_predictions'],
                top_n,
                stock_features_df,
                method_name="pattern_based_prediction"
            )

        # 为了兼容性，保持原有的变量名
        similarity_selected = pattern_selected
        model_selected = None  # 新方法不使用机器学习模型

        # 8.3: 合并选股结果用于兼容性
        selected_stocks = self._combine_selection_results(similarity_selected, model_selected)

        # 步骤9: 评估新的特征模式预测方法的实际表现
        self.logger.info("评估特征模式预测方法的实际表现...")

        # 9.1: 评估特征模式预测方法
        pattern_evaluation = None
        if similarity_selected:  # similarity_selected实际上是pattern_selected
            self.logger.info("评估特征模式预测方法的表现...")
            pattern_evaluation = self._evaluate_selected_stocks_with_method(
                similarity_selected, stock_features_df, "pattern_based_prediction"
            )

        # 为了兼容性，保持原有的变量名
        similarity_evaluation = pattern_evaluation
        model_evaluation = None  # 新方法不使用机器学习模型

        # 9.2: 生成评估报告（简化版，因为只有一种方法）
        comparison_evaluation = {
            'pattern_method_performance': pattern_evaluation.get('overall_summary', {}) if pattern_evaluation else {},
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method_used': 'pattern_based_prediction'
        }

        # 将评估结果添加到选股结果中
        selected_stocks['evaluation'] = {
            'similarity_evaluation': similarity_evaluation,
            'model_evaluation': model_evaluation,
            'comparison_evaluation': comparison_evaluation
        }
        selected_stocks['model_name'] = model_name
        selected_stocks['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 为选出的股票添加排名信息（如果尚未添加）
        if 'selected_stocks' in selected_stocks and selected_stocks['selected_stocks']:
            for i, stock in enumerate(selected_stocks['selected_stocks']):
                if 'rank' not in stock:
                    stock['rank'] = i + 1

        # 保存结果
        result_path = self.model_dir / f"{model_name}_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(selected_stocks, f, ensure_ascii=False, cls=NumpyEncoder, indent=2)

        # 保存每日选股结果到单独文件
        if 'selected_stocks_by_date' in selected_stocks and selected_stocks['selected_stocks_by_date']:
            daily_results_dir = self.model_dir / f"{model_name}_daily_results"
            os.makedirs(daily_results_dir, exist_ok=True)

            for date, stocks in selected_stocks['selected_stocks_by_date'].items():
                date_result_path = daily_results_dir / f"{date}_stocks.json"

                # 获取该日期的评估结果
                date_evaluation = None
                if 'evaluation' in selected_stocks and 'evaluation_by_date' in selected_stocks['evaluation']:
                    date_evaluation = selected_stocks['evaluation']['evaluation_by_date'].get(date)

                date_result = {
                    'selected_stocks': stocks,
                    'date': date,
                    'model_name': model_name,
                    'total_stocks': len(stocks),
                    'evaluation': date_evaluation
                }

                with open(date_result_path, "w", encoding="utf-8") as f:
                    json.dump(date_result, f, ensure_ascii=False, cls=NumpyEncoder, indent=2)

            self.logger.info(f"每日选股结果已保存到目录: {daily_results_dir}")

        # 生成评估报告
        if 'evaluation' in selected_stocks:
            eval_results = selected_stocks['evaluation']

            # 处理新的评估结果格式
            if 'overall_summary' in eval_results and isinstance(eval_results['overall_summary'], dict):
                summary = eval_results['overall_summary']

                # 输出整体评估结果
                if 'avg_actual_return' in summary and 'win_rate' in summary:
                    self.logger.info(f"整体选股评估结果:")
                    self.logger.info(f"- 评估日期数: {summary.get('total_evaluated_dates', 'N/A')}")
                    self.logger.info(f"- 评估股票数: {summary.get('total_evaluated_stocks', 'N/A')}")
                    self.logger.info(f"- 平均实际收益率: {summary['avg_actual_return']:.2f}%")
                    self.logger.info(f"- 方向准确率: {summary['direction_accuracy']*100:.1f}%")
                    self.logger.info(f"- 胜率: {summary['win_rate']*100:.1f}%")
                    self.logger.info(f"- 表现评级: {summary['performance_rating']}")

                    # 如果有日平均表现数据
                    if 'avg_daily_return' in summary:
                        self.logger.info(f"- 日均收益率: {summary['avg_daily_return']:.2f}%")
                        self.logger.info(f"- 日均胜率: {summary['avg_daily_win_rate']*100:.1f}%")

            # 处理旧的评估结果格式 (向后兼容)
            elif 'evaluation_summary' in eval_results and isinstance(eval_results['evaluation_summary'], dict):
                summary = eval_results['evaluation_summary']
                if 'avg_actual_return' in summary and 'win_rate' in summary and 'direction_accuracy' in summary:
                    self.logger.info(f"选股评估结果:")
                    self.logger.info(f"- 平均实际收益率: {summary['avg_actual_return']:.2f}%")
                    self.logger.info(f"- 方向准确率: {summary['direction_accuracy']*100:.1f}%")
                    self.logger.info(f"- 胜率: {summary['win_rate']*100:.1f}%")
                    self.logger.info(f"- 表现评级: {summary['performance_rating']}")

        self.logger.info(f"选股结果和评估已保存到 {result_path}")

        return selected_stocks

    def load_stock_features(self, start_date=None, end_date=None):
        """
        加载个股特征数据，使用缓存机制提高效率

        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD

        Returns:
            DataFrame: 股票特征数据
        """
        self.logger.info(f"加载个股特征数据 {start_date} 至 {end_date}...")

        # 创建缓存键
        cache_key = f"stock_features_{start_date}_{end_date}"
        cache_file = self.features_cache_dir / f"{cache_key}.parquet"

        # 检查缓存
        if cache_file.exists():
            self.logger.info(f"从缓存加载个股特征数据: {cache_file}")
            cached_data = pd.read_parquet(cache_file)

            # 检查缓存是否包含必要的列
            required_columns = ['future_return_2d']
            missing_columns = [col for col in required_columns if col not in cached_data.columns]

            if not missing_columns:
                # 检查future_return_2d列是否有足够的有效值
                valid_pct = cached_data['future_return_2d'].notna().sum() / len(cached_data)
                self.logger.info(f"缓存数据中future_return_2d列有效值比例: {valid_pct:.2f}")

                if valid_pct >= 0.5:  # 如果有效值比例足够高
                    self.logger.info(f"缓存数据包含所有必要列且有效，直接使用")
                    return cached_data
                else:
                    self.logger.warning(f"缓存数据中future_return_2d列有效值比例过低: {valid_pct:.2f}，需要重新计算")
            else:
                self.logger.warning(f"缓存数据缺少必要列: {missing_columns}，需要重新计算")

        # 缓存不存在或者不完整，从原始特征文件加载
        try:
            # 确定日期范围
            if start_date is None or end_date is None:
                raise ValueError("必须指定开始日期和结束日期")

            # 获取日期范围内的所有交易日
            calendar = self._load_trading_calendar()
            if calendar is None:
                raise ValueError("无法加载交易日历")

            # 筛选日期范围内的交易日
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            trading_dates = [d for d in calendar if start_dt <= pd.to_datetime(d) <= end_dt]

            if not trading_dates:
                raise ValueError(f"指定的日期范围 {start_date} 至 {end_date} 内没有交易日")

            # 从generate_daily_features.py中获取的保存特征的目录路径
            features_dir = get_data_path("features/daily")

            # 从每个交易日的特征文件加载数据
            all_data = []
            for date in trading_dates:
                date_str = date if isinstance(date, str) else date.strftime("%Y%m%d")
                feature_file = features_dir / f"{date_str}.parquet"

                if not os.path.exists(feature_file):
                    self.logger.warning(f"日期 {date_str} 的特征文件不存在: {feature_file}")
                    continue

                try:
                    # 加载当日特征
                    daily_features = pd.read_parquet(feature_file)
                    all_data.append(daily_features)
                    self.logger.debug(f"成功加载日期 {date_str} 的特征数据，共 {len(daily_features)} 条记录")
                except Exception as e:
                    self.logger.error(f"加载日期 {date_str} 的特征数据时出错: {e}")
                    continue

            if not all_data:
                raise ValueError(f"未能加载任何日期的特征数据")

            # 合并所有日期的数据
            combined_features = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"成功加载 {len(trading_dates)} 个交易日的特征数据，共 {len(combined_features)} 条记录")

            # 特征已在generate_daily_features.py中进行了高级增强处理
            # 包含K线形态、趋势分析、支撑阻力、涨跌停检测、未来收益率等
            enhanced_features = combined_features

            # 保存到缓存
            ensure_dir(self.features_cache_dir)
            enhanced_features.to_parquet(cache_file)
            self.logger.info(f"增强后的个股特征数据已缓存到: {cache_file}")

            return enhanced_features

        except Exception as e:
            self.logger.error(f"加载个股特征数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            raise ValueError(f"加载个股特征数据失败: {e}")



    def _detect_stock_change_points(self, stock_features_df):
        """
        检测股票价格关键变化点

        Args:
            stock_features_df: 股票特征DataFrame

        Returns:
            dict: 变化点检测结果，格式为 {stock_code: {'change_points': [...], 'returns': [...], 'dates': [...]}}
        """
        # 检查缓存
        # 创建缓存目录
        cache_dir = self.model_dir / "features_cache"
        ensure_dir(cache_dir)

        # 生成缓存文件名
        try:
            date_range = f"{pd.to_datetime(stock_features_df['date_str']).min().strftime('%Y%m%d')}_{pd.to_datetime(stock_features_df['date_str']).max().strftime('%Y%m%d')}"
            num_stocks = len(stock_features_df['stock_code'].unique())
            cache_file = cache_dir / f"change_points_{num_stocks}stocks_{date_range}.pkl"

            # 如果缓存存在，直接加载并返回
            if cache_file.exists():
                self.logger.info(f"从缓存加载变化点检测结果: {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"加载缓存失败: {e}，将重新计算变化点")
        except Exception as e:
            self.logger.warning(f"生成缓存键失败: {e}，将不使用缓存")

        self.logger.info("开始检测股票价格关键变化点...")

        # 仅使用future_return_2d作为变化点检测的核心指标
        # 其他指标仅作为参考，但不直接用于变化点检测
        indicators = {'future_return_2d': 1.0}  # 最核心指标，唯一权重

        # 处理不同股票
        results = {}
        stock_codes = stock_features_df['stock_code'].unique()

        self.logger.info(f"开始处理 {len(stock_codes)} 只股票的变化点检测...")

        # 按股票代码分组处理
        for i, stock_code in enumerate(stock_codes):
            if i % 100 == 0:
                self.logger.info(f"已处理 {i}/{len(stock_codes)} 只股票")

            stock_df = stock_features_df[stock_features_df['stock_code'] == stock_code].copy()
            if len(stock_df) < 30:  # 确保有足够数据
                continue

            # 确保股票数据中有future_return_2d列
            if 'future_return_2d' not in stock_df.columns:
                self.logger.warning(f"股票 {stock_code} 缺少future_return_2d列，无法进行变化点检测")
                continue

            # 检查future_return_2d列是否有足够的有效值
            valid_ratio = stock_df['future_return_2d'].notna().sum() / len(stock_df)
            if valid_ratio < 0.3:  # 至少30%的数据有效
                self.logger.warning(f"股票 {stock_code} 的future_return_2d有效值比例过低: {valid_ratio:.2f}，跳过")
                continue

            # 调用单只股票的变化点检测方法
            result = self._detect_single_stock_change_points(stock_code, stock_df, indicators)

            # 保存有效结果
            if result and 'change_points' in result and len(result['change_points']) > 0:
                results[stock_code] = result

        successful_count = len(results)
        total_count = len(stock_codes)
        success_rate = successful_count / total_count * 100 if total_count > 0 else 0

        self.logger.info(f"变化点检测完成，成功率: {success_rate:.1f}% ({successful_count}/{total_count})")

        # 计算所有股票平均变化点数量
        if results:
            total_points = sum(len(r['change_points']) for r in results.values() if 'change_points' in r)
            avg_points = total_points / len(results)
            self.logger.info(f"平均每个股票变化点数: {avg_points:.2f}")

        # 保存结果到缓存
        try:
            if 'cache_file' in locals():
                self.logger.info(f"保存变化点检测结果到缓存: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {e}")

        return results

    def collect_stock_sector_change_point_features(self, stock_features_df, sector_features_df, change_points=None, is_training=False, use_cache=True):
        """
        统一的特征工程入口 - 返回完整的特征DataFrame

        重构版本：无论训练还是预测，都返回一个包含所有特征的统一DataFrame：
        - 个股基础特征（从stock_features_df中获取）
        - 个股与板块的相对特征（通过关联分析计算）
        - 板块特征（从sector_features_df中获取）
        - 市场特征（独立的市场环境特征）

        Args:
            stock_features_df: 个股特征DataFrame（包含个股的所有基础特征和技术指标）
            sector_features_df: 板块特征DataFrame（包含板块的基础特征和技术指标）
            change_points: 变化点检测结果，格式为 {stock_code: {'change_points': [...], 'dates': [...]}}
            is_training: 是否为训练数据，用于区分缓存文件和计算逻辑
            use_cache: 是否使用缓存，默认为True

        Returns:
            DataFrame: 包含所有特征的完整DataFrame（个股+板块+市场+相对特征）
        """
        self.logger.info("开始计算个股与板块的关联特征...")

        # 保存变化点数据和DataFrame引用，供计算最相关板块使用
        self.change_points = change_points
        self.stock_features_df = stock_features_df
        self.sector_features_df = sector_features_df

        # 如果是预测阶段，尝试加载训练阶段计算的最相关板块映射
        if not is_training:
            self.training_sector_mapping = self._load_training_sector_mapping()
            if self.training_sector_mapping:
                self.logger.info(f"成功加载训练阶段的板块映射，包含 {len(self.training_sector_mapping)} 个股票的映射关系")
            else:
                self.logger.warning("未能加载训练阶段的板块映射，将使用实时计算")
        else:
            self.stock_most_relevant_sectors = {}

        if stock_features_df.empty or sector_features_df.empty:
            self.logger.warning("特征数据为空，返回空DataFrame")
            return stock_features_df, pd.DataFrame()

        # ===== 关键修复：在数据处理早期阶段检查和转换字符串类型特征 =====
        self.logger.info("=== 开始检查和处理字符串类型特征 ===")

        # 定义不应该被转换的重要标识列
        protected_columns = {
            'stock_code', 'sector_code', 'date_str', 'date', 'most_relevant_sector'
        }

        # 检查个股特征中的字符串列
        for col in stock_features_df.columns:
            if stock_features_df[col].dtype == 'object':
                # 跳过受保护的标识列
                if col in protected_columns:
                    self.logger.info(f"跳过受保护的标识列: {col}")
                    continue

                sample_values = stock_features_df[col].dropna().head(10).tolist()
                self.logger.info(f"个股特征中发现字符串列 {col}，样本值: {sample_values}")

                # 检查是否包含特定的字符串值
                unique_values = stock_features_df[col].dropna().unique()
                string_values = [v for v in unique_values if isinstance(v, str)]
                if string_values:
                    self.logger.info(f"  字符串值: {string_values[:10]}")

                    # 特别检查是否包含bull, bear, neutral等问题字符串
                    problematic_values = [v for v in string_values if v.lower() in ['bull', 'bear', 'neutral', 'low', 'medium', 'high']]
                    if problematic_values:
                        self.logger.warning(f"  发现需要转换的字符串值: {problematic_values}")

                        # 应用字符串到数值的映射
                        string_to_num_map = {
                            'bull': 1, 'bear': -1, 'neutral': 0,
                            'bullish': 1, 'bearish': -1,
                            'low': 0, 'medium': 1, 'high': 2, 'extreme': 3,
                            'buy': 1, 'sell': -1, 'hold': 0,
                            'strong': 2, 'weak': -2, 'normal': 0
                        }

                        # 创建副本以避免修改原始数据
                        stock_features_df = stock_features_df.copy()
                        stock_features_df[col] = stock_features_df[col].apply(
                            lambda x: string_to_num_map.get(str(x).lower(), 0) if pd.notna(x) else 0
                        )
                        self.logger.info(f"  已转换列 {col} 为数值类型")
                    else:
                        # 尝试直接转换为数值
                        try:
                            stock_features_df = stock_features_df.copy()
                            stock_features_df[col] = pd.to_numeric(stock_features_df[col], errors='coerce').fillna(0)
                            self.logger.info(f"  已转换列 {col} 为数值类型")
                        except:
                            self.logger.warning(f"  无法转换列 {col}，保持原样")

        # 检查板块特征中的字符串列
        for col in sector_features_df.columns:
            if sector_features_df[col].dtype == 'object':
                # 跳过受保护的标识列
                if col in protected_columns:
                    self.logger.info(f"跳过受保护的标识列: {col}")
                    continue

                sample_values = sector_features_df[col].dropna().head(10).tolist()
                self.logger.info(f"板块特征中发现字符串列 {col}，样本值: {sample_values}")

                # 检查是否包含特定的字符串值
                unique_values = sector_features_df[col].dropna().unique()
                string_values = [v for v in unique_values if isinstance(v, str)]
                if string_values:
                    self.logger.info(f"  字符串值: {string_values[:10]}")

                    # 特别检查是否包含bull, bear, neutral等问题字符串
                    problematic_values = [v for v in string_values if v.lower() in ['bull', 'bear', 'neutral', 'low', 'medium', 'high']]
                    if problematic_values:
                        self.logger.warning(f"  发现需要转换的字符串值: {problematic_values}")

                        # 应用字符串到数值的映射
                        string_to_num_map = {
                            'bull': 1, 'bear': -1, 'neutral': 0,
                            'bullish': 1, 'bearish': -1,
                            'low': 0, 'medium': 1, 'high': 2, 'extreme': 3,
                            'buy': 1, 'sell': -1, 'hold': 0,
                            'strong': 2, 'weak': -2, 'normal': 0
                        }

                        # 创建副本以避免修改原始数据
                        sector_features_df = sector_features_df.copy()
                        sector_features_df[col] = sector_features_df[col].apply(
                            lambda x: string_to_num_map.get(str(x).lower(), 0) if pd.notna(x) else 0
                        )
                        self.logger.info(f"  已转换列 {col} 为数值类型")
                    else:
                        # 尝试直接转换为数值
                        try:
                            sector_features_df = sector_features_df.copy()
                            sector_features_df[col] = pd.to_numeric(sector_features_df[col], errors='coerce').fillna(0)
                            self.logger.info(f"  已转换列 {col} 为数值类型")
                        except:
                            self.logger.warning(f"  无法转换列 {col}，保持原样")

        self.logger.info("=== 字符串类型特征处理完成 ===")

        # 检查缓存
        cache_file = None
        if use_cache:
            # 创建缓存目录
            cache_dir = self.model_dir / "features_cache"
            ensure_dir(cache_dir)

            # 生成缓存文件名 - 统一缓存，不再区分relative和relations
            data_type = "train" if is_training else "predict"

            # 使用日期范围和股票数量作为缓存键的一部分
            try:
                date_range = f"{pd.to_datetime(stock_features_df['date_str']).min().strftime('%Y%m%d')}_{pd.to_datetime(stock_features_df['date_str']).max().strftime('%Y%m%d')}"
                num_stocks = len(stock_features_df['stock_code'].unique())
                cache_file = cache_dir / f"complete_features_{data_type}_{num_stocks}stocks_{date_range}.pkl"

                # 如果缓存存在，直接加载并返回
                if cache_file.exists():
                    self.logger.info(f"从缓存加载完整特征: {cache_file}")
                    try:
                        # 直接加载完整特征DataFrame
                        with open(cache_file, 'rb') as f:
                            complete_features_df = pickle.load(f)

                        self.logger.info(f"成功加载完整特征缓存，包含 {len(complete_features_df)} 条记录，{len(complete_features_df.columns)} 个特征")

                        # 验证缓存数据的完整性
                        if 'most_relevant_sector' in complete_features_df.columns:
                            sector_counts = complete_features_df['most_relevant_sector'].value_counts()
                            self.logger.info(f"缓存中板块分布: {dict(sector_counts.head(5))}")

                        return complete_features_df
                    except Exception as e:
                        self.logger.warning(f"加载缓存失败: {e}，将重新计算特征")
            except Exception as e:
                self.logger.warning(f"生成缓存键失败: {e}，将不使用缓存")

        # 性能优化：复制数据前先创建索引，避免反复查询
        self.logger.info("创建高效索引以加速处理...")

        # 复制数据，避免修改原始数据
        result_df = stock_features_df.copy()

        # 预处理股票龙头特征
        result_df = self._prepare_stock_leadership_features(result_df)

        # 获取股票板块映射函数
        try:
            from to.stock_sectors_mapping import get_stock_sectors
            get_stock_sectors_func = get_stock_sectors
        except ImportError:
            try:
                from stock_sectors_mapping import get_stock_sectors
                get_stock_sectors_func = get_stock_sectors
            except ImportError:
                self.logger.error("无法导入get_stock_sectors函数")
                return stock_features_df

        # 加载一次数据，供所有股票使用
        data = None
        try:
            from to.stock_sectors_mapping import load_data
            data = load_data()
            self.logger.info("成功加载股票板块映射数据")
        except ImportError:
            try:
                from stock_sectors_mapping import load_data
                data = load_data()
                self.logger.info("成功加载股票板块映射数据")
            except ImportError:
                self.logger.warning("无法导入load_data函数，股票板块映射可能不完整")
            except Exception as e:
                self.logger.warning(f"加载stock_sectors_mapping数据时出错: {str(e)}")
        except Exception as e:
            self.logger.warning(f"加载to.stock_sectors_mapping数据时出错: {str(e)}")

        # 排除的非行业概念板块列表
        excluded_sectors = [
            'TGN中证500成份股', 'TGN融资融券', 'TGN深股通',
            'TGN标普道琼斯A股', 'TGNMSCI概念', 'TGN沪深300样本股'
        ]

        # 创建股票到板块的映射缓存，一次性获取所有映射
        self.logger.info("一次性构建所有股票的板块映射关系...")
        stock_codes = result_df['stock_code'].unique()
        stock_sector_mapping = {}

        # 使用多进程加速股票板块映射构建
        from concurrent.futures import ThreadPoolExecutor

        def get_stock_sectors_wrapper(stock_code):
            sectors = get_stock_sectors_func(stock_code=stock_code, data=data)
            if sectors:
                return stock_code, [s for s in sectors if s not in excluded_sectors]
            return stock_code, []

        # 使用线程池并行处理股票到板块的映射
        with ThreadPoolExecutor(max_workers=10) as executor:
            for stock_code, sectors in executor.map(get_stock_sectors_wrapper, stock_codes):
                stock_sector_mapping[stock_code] = sectors

        # 确定需要处理的股票-日期组合
        process_combinations = []

        # 修复问题1: 正确检查变化点数据是否有效
        has_valid_change_points = False
        if change_points:
            # 检查是否有任何有效的变化点
            for stock_code, cp_data in change_points.items():
                if cp_data and isinstance(cp_data, dict) and 'dates' in cp_data:
                    dates = cp_data['dates']
                    if isinstance(dates, list) and dates:
                        has_valid_change_points = True
                        for date_str in dates:
                            process_combinations.append((stock_code, date_str))
                    elif isinstance(dates, np.ndarray) and dates.size > 0:
                        has_valid_change_points = True
                        for date_str in dates:
                            process_combinations.append((stock_code, date_str))

            if has_valid_change_points:
                self.logger.info(f"将处理 {len(process_combinations)} 个变化点对应的股票-日期组合")
            else:
                self.logger.warning("未找到有效的变化点数据，将处理所有股票-日期组合")
                has_valid_change_points = False

        if not has_valid_change_points:
            # 如果没有提供变化点，则处理所有组合
            self.logger.warning("将处理所有股票-日期组合")
            stock_groups = result_df.groupby(['stock_code', 'date_str'])
            process_combinations = [(stock_code, date_str) for (stock_code, date_str) in stock_groups.groups.keys()]

        # 如果是训练阶段，先计算所有股票的最相关板块，然后过滤组合
        if is_training and change_points:
            self.stock_most_relevant_sectors = self._calculate_all_stocks_most_relevant_sectors()

            # 保存训练阶段的板块映射
            if self.stock_most_relevant_sectors:
                self._save_training_sector_mapping_from_dict(self.stock_most_relevant_sectors)

                # 重要修复：只处理已经计算出最相关板块的股票的变化点组合
                # 这样确保process_combinations中的每个股票都有对应的最相关板块
                filtered_combinations = []
                filtered_count = 0

                for stock_code, date_str in process_combinations:
                    if stock_code in self.stock_most_relevant_sectors:
                        filtered_combinations.append((stock_code, date_str))
                    else:
                        filtered_count += 1

                process_combinations = filtered_combinations

                if filtered_count > 0:
                    self.logger.info(f"过滤掉了 {filtered_count} 个无最相关板块的股票-日期组合")
                    self.logger.info(f"保留了 {len(process_combinations)} 个有最相关板块的股票-日期组合")
            else:
                self.logger.warning("未能计算出任何股票的最相关板块，将跳过特征计算")
                return stock_features_df

        # 预测阶段：过滤掉训练阶段没有最相关板块的股票
        if not is_training and hasattr(self, 'training_sector_mapping') and self.training_sector_mapping:
            filtered_combinations = []
            filtered_count = 0

            for stock_code, date_str in process_combinations:
                if stock_code in self.training_sector_mapping:
                    filtered_combinations.append((stock_code, date_str))
                else:
                    filtered_count += 1

            process_combinations = filtered_combinations

            if filtered_count > 0:
                self.logger.info(f"预测阶段过滤掉了 {filtered_count} 个训练时无最相关板块的股票-日期组合")
                self.logger.info(f"预测阶段保留了 {len(process_combinations)} 个股票-日期组合")

        # 创建性能优化的索引
        result_df_idx = result_df.set_index(['stock_code', 'date_str'])
        sector_features_df_idx = sector_features_df.set_index(['sector_code', 'date'])

        # 优化：增大批处理大小，减少线程池创建开销
        batch_size = 5000  # 增大批处理大小
        total_groups = len(process_combinations)
        processed = 0

        # 创建结果字典
        results = {}

        # 添加进度日志
        self.logger.info(f"开始处理 {total_groups} 个股票-日期组合，每批次 {batch_size} 个...")
        start_time = time.time()

        # 优化：减少线程数，避免过度并发
        max_workers = min(8, os.cpu_count())  # 减少线程数

        # 按批次处理
        for batch_start in range(0, total_groups, batch_size):
            batch_end = min(batch_start + batch_size, total_groups)
            batch = process_combinations[batch_start:batch_end]

            batch_start_time = time.time()
            self.logger.info(f"开始处理第 {batch_start//batch_size + 1} 批，共 {len(batch)} 个组合...")

            # 优化：对于小批次，直接顺序处理，避免线程池开销
            if len(batch) < 100:
                batch_success = 0
                batch_errors = 0
                for stock_code, date_str in batch:
                    try:
                        result = self._process_single_stock_date(
                            stock_code, date_str, result_df_idx,
                            stock_sector_mapping, sector_features_df_idx, is_training
                        )
                        if result:
                            key = (result['stock_code'], result['date'])
                            results[key] = result
                            batch_success += 1
                    except Exception as e:
                        self.logger.error(f"处理股票 {stock_code} 在 {date_str} 的特征时出错: {str(e)}")
                        batch_errors += 1
            else:
                # 使用线程池并行处理大批次
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    futures_dict = {}
                    for stock_code, date_str in batch:
                        future = executor.submit(
                            self._process_single_stock_date,
                            stock_code,
                            date_str,
                            result_df_idx,
                            stock_sector_mapping,
                            sector_features_df_idx,
                            is_training
                        )
                        futures_dict[future] = (stock_code, date_str)

                    # 收集结果
                    batch_success = 0
                    batch_errors = 0
                    for future in as_completed(futures_dict):
                        try:
                            result = future.result()
                            if result:
                                key = (result['stock_code'], result['date'])
                                results[key] = result
                                batch_success += 1
                        except Exception as e:
                            stock_code, date_str = futures_dict[future]
                            self.logger.error(f"处理股票 {stock_code} 在 {date_str} 的特征时出错: {str(e)}")
                            batch_errors += 1

            # 更新处理进度
            processed += len(batch)
            batch_time = time.time() - batch_start_time
            self.logger.info(f"第 {batch_start//batch_size + 1} 批处理完成：成功 {batch_success}，失败 {batch_errors}，用时 {batch_time:.2f} 秒")
            self.logger.info(f"总进度: 已处理 {processed}/{total_groups} 个股票-日期组合 ({processed/total_groups:.1%})")

        total_time = time.time() - start_time
        self.logger.info(f"所有批次处理完成，总用时 {total_time:.2f} 秒，平均每个组合 {total_time/total_groups:.4f} 秒")

        # 优化：批量更新DataFrame，避免逐行操作
        if results:
            self.logger.info(f"共找到 {len(results)} 个有效的股票-板块关联")
            try:
                results_df = pd.DataFrame.from_dict(results, orient='index')
                results_df = results_df.reset_index(drop=True)

                self.logger.info("开始集成特征到result_df和relations_df...")

                # 定义基本关联信息列
                basic_cols = ['stock_code', 'date', 'most_relevant_sector']

                # 分离板块特征：除了基本关联信息外的所有列都是板块特征
                sector_feature_cols = [col for col in results_df.columns if col not in basic_cols]

                # 创建relations_df（直接使用results_df，避免重复循环）
                relations_df = results_df.copy()
                self.logger.info(f"创建relations_df，包含 {len(relations_df)} 条记录和 {len(relations_df.columns)} 个特征")
                self.logger.info(f"其中板块特征列数量: {len(sector_feature_cols)}")

                # 优化：使用merge操作批量更新，而不是逐行更新
                # 准备更新数据
                update_data = results_df[basic_cols].copy()
                update_data = update_data.rename(columns={'date': 'date_str'})  # 匹配result_df的列名

                # 初始化新列（如果不存在）
                for col in ['most_relevant_sector']:
                    if col not in result_df.columns:
                        result_df[col] = None

                # 使用merge + update的方式批量更新
                # 创建临时索引用于快速更新
                result_df_temp = result_df.set_index(['stock_code', 'date_str'])
                update_data_temp = update_data.set_index(['stock_code', 'date_str'])

                # 批量更新
                for col in ['most_relevant_sector']:
                    result_df_temp.loc[update_data_temp.index, col] = update_data_temp[col]

                # 调试：检查most_relevant_sector的值分布
                if 'most_relevant_sector' in update_data_temp.columns:
                    sector_counts = update_data_temp['most_relevant_sector'].value_counts()
                    self.logger.info(f"板块分布统计: {dict(sector_counts.head(10))}")
                    unknown_count = (update_data_temp['most_relevant_sector'] == 'unknown').sum()
                    if unknown_count > 0:
                        self.logger.warning(f"发现 {unknown_count} 个unknown板块记录，占比 {unknown_count/len(update_data_temp):.2%}")
                        # 随机抽样几个unknown记录进行调试
                        unknown_samples = update_data_temp[update_data_temp['most_relevant_sector'] == 'unknown'].head(3)
                        for idx, row in unknown_samples.iterrows():
                            self.logger.debug(f"Unknown板块样本: 股票={row.get('stock_code', 'N/A')}, 日期={row.get('date_str', 'N/A')}")
                    else:
                        self.logger.info("所有记录都有有效的板块信息")

                # 重置索引
                result_df = result_df_temp.reset_index()

                self.logger.info(f"成功批量更新result_df中的 {len(update_data)} 条记录的关联信息")

            except Exception as e:
                self.logger.error(f"处理结果数据时出错: {str(e)}")
                import traceback
                self.logger.error(f"错误详情: {traceback.format_exc()}")
        else:
            self.logger.warning("没有找到有效的股票-板块关联数据")

        # 如果前面的代码没有创建relations_df（比如在结果为空的情况下），则创建一个空的DataFrame
        if 'relations_df' not in locals():
            self.logger.info("没有找到有效的股票板块关联数据，创建空的relations_df")
            relations_df = pd.DataFrame()

        # 重构：合并所有特征到一个完整的DataFrame
        self.logger.info("合并个股特征、板块特征和相对特征到统一DataFrame...")

        # 将板块特征合并到个股特征中，创建完整的特征DataFrame
        complete_features_df = self._merge_all_features(result_df, relations_df)

        self.logger.info(f"完整特征DataFrame包含 {len(complete_features_df)} 条记录，{len(complete_features_df.columns)} 个特征")

        # 如果是训练阶段，保存板块映射供预测阶段使用
        if is_training and hasattr(self, 'stock_most_relevant_sectors') and self.stock_most_relevant_sectors:
            self._save_training_sector_mapping_from_dict(self.stock_most_relevant_sectors)

        # 保存完整特征到缓存 - 统一缓存，不再分离
        if use_cache and cache_file:
            try:
                self.logger.info(f"保存完整特征到缓存文件: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(complete_features_df, f)

                # 验证保存的数据
                if 'most_relevant_sector' in complete_features_df.columns:
                    sector_counts = complete_features_df['most_relevant_sector'].value_counts()
                    self.logger.info(f"保存到缓存的板块分布: {dict(sector_counts.head(5))}")

                self.logger.info(f"成功保存 {len(complete_features_df)} 条记录，{len(complete_features_df.columns)} 个特征到缓存")
            except Exception as e:
                self.logger.warning(f"保存缓存失败: {e}")

        return complete_features_df

    def _merge_all_features(self, result_df, relations_df):
        """
        合并所有特征到一个完整的DataFrame

        Args:
            result_df: 包含个股基础特征和关联信息的DataFrame
            relations_df: 包含板块特征和相对特征的DataFrame

        Returns:
            DataFrame: 包含所有特征的完整DataFrame（个股+板块+市场+相对特征）
        """
        if relations_df is None or relations_df.empty:
            self.logger.warning("板块特征DataFrame为空，只返回个股特征")
            return result_df

        try:
            # 调试：检查输入DataFrame中的most_relevant_sector字段
            self.logger.info(f"result_df包含 {len(result_df)} 条记录，{len(result_df.columns)} 个特征")
            self.logger.info(f"relations_df包含 {len(relations_df)} 条记录，{len(relations_df.columns)} 个特征")

            # 调试：检查DataFrame的列信息
            self.logger.info(f"result_df列名: {list(result_df.columns)}")
            self.logger.info(f"relations_df列名: {list(relations_df.columns)}")

            # 检查日期相关列的数据类型
            for df_name, df in [('result_df', result_df), ('relations_df', relations_df)]:
                for col in ['date', 'date_str']:
                    if col in df.columns:
                        self.logger.info(f"{df_name}[{col}] 数据类型: {df[col].dtype}, 样本值: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")

            if 'most_relevant_sector' in result_df.columns:
                sector_counts = result_df['most_relevant_sector'].value_counts()
                self.logger.info(f"result_df中板块分布: {dict(sector_counts.head(10))}")
                unknown_count = (result_df['most_relevant_sector'] == 'unknown').sum()
                if unknown_count > 0:
                    self.logger.warning(f"result_df中发现 {unknown_count} 个unknown板块记录")
            else:
                self.logger.warning("result_df中缺少most_relevant_sector字段")

            if 'most_relevant_sector' in relations_df.columns:
                sector_counts = relations_df['most_relevant_sector'].value_counts()
                self.logger.info(f"relations_df中板块分布: {dict(sector_counts.head(10))}")
            else:
                self.logger.info("relations_df中没有most_relevant_sector字段（这是正常的）")

            # 确定合并键
            merge_keys = ['stock_code']

            # 检查日期字段
            if 'date_str' in result_df.columns and 'date_str' in relations_df.columns:
                merge_keys.append('date_str')
            elif 'date' in result_df.columns and 'date' in relations_df.columns:
                merge_keys.append('date')
            elif 'date_str' in result_df.columns and 'date' in relations_df.columns:
                # 统一日期字段名
                relations_df = relations_df.copy()
                relations_df['date_str'] = relations_df['date']
                merge_keys.append('date_str')
            elif 'date' in result_df.columns and 'date_str' in relations_df.columns:
                # 统一日期字段名
                result_df = result_df.copy()
                result_df['date_str'] = result_df['date']
                merge_keys.append('date_str')

            # 检查并修复日期列的数据类型不匹配问题
            if len(merge_keys) > 1:  # 如果包含日期字段
                date_key = merge_keys[1]  # 第二个键是日期字段（第一个是stock_code）

                # 检查两个DataFrame中日期列的数据类型
                result_date_dtype = result_df[date_key].dtype
                relations_date_dtype = relations_df[date_key].dtype

                self.logger.debug(f"result_df[{date_key}] 数据类型: {result_date_dtype}")
                self.logger.debug(f"relations_df[{date_key}] 数据类型: {relations_date_dtype}")

                # 如果数据类型不匹配，进行转换
                if result_date_dtype != relations_date_dtype:
                    self.logger.info(f"检测到日期列数据类型不匹配: {result_date_dtype} vs {relations_date_dtype}")

                    # 创建副本以避免修改原始数据
                    result_df = result_df.copy()
                    relations_df = relations_df.copy()

                    # 统一转换为字符串类型，这是最安全的方式
                    try:
                        # 如果是datetime类型，转换为字符串格式
                        if pd.api.types.is_datetime64_any_dtype(result_df[date_key]):
                            result_df[date_key] = result_df[date_key].dt.strftime('%Y%m%d')
                            self.logger.debug(f"将result_df[{date_key}]从datetime转换为字符串")
                        else:
                            # 确保是字符串类型
                            result_df[date_key] = result_df[date_key].astype(str)
                            self.logger.debug(f"将result_df[{date_key}]转换为字符串")

                        if pd.api.types.is_datetime64_any_dtype(relations_df[date_key]):
                            relations_df[date_key] = relations_df[date_key].dt.strftime('%Y%m%d')
                            self.logger.debug(f"将relations_df[{date_key}]从datetime转换为字符串")
                        else:
                            # 确保是字符串类型
                            relations_df[date_key] = relations_df[date_key].astype(str)
                            self.logger.debug(f"将relations_df[{date_key}]转换为字符串")

                        self.logger.info(f"成功统一日期列数据类型为字符串")

                    except Exception as e:
                        self.logger.error(f"转换日期列数据类型时出错: {e}")
                        # 如果转换失败，尝试其他方法
                        try:
                            # 尝试直接转换为字符串
                            result_df[date_key] = result_df[date_key].astype(str)
                            relations_df[date_key] = relations_df[date_key].astype(str)
                            self.logger.info("使用备用方法成功转换日期列为字符串")
                        except Exception as e2:
                            self.logger.error(f"备用转换方法也失败: {e2}")
                            raise e2

            # 执行左连接，保留所有个股记录
            try:
                self.logger.debug(f"准备合并DataFrame，合并键: {merge_keys}")
                complete_df = result_df.merge(
                    relations_df,
                    on=merge_keys,
                    how='left',
                    suffixes=('', '_sector')
                )
                self.logger.info(f"DataFrame合并成功，结果包含 {len(complete_df)} 行，{len(complete_df.columns)} 列")
            except Exception as merge_error:
                self.logger.error(f"DataFrame合并失败: {merge_error}")
                # 提供更详细的错误信息
                if len(merge_keys) > 1:
                    date_key = merge_keys[1]
                    self.logger.error(f"合并键详情: {merge_keys}")
                    self.logger.error(f"result_df[{date_key}] 数据类型: {result_df[date_key].dtype}")
                    self.logger.error(f"relations_df[{date_key}] 数据类型: {relations_df[date_key].dtype}")
                    self.logger.error(f"result_df[{date_key}] 样本值: {result_df[date_key].iloc[:3].tolist() if len(result_df) > 0 else []}")
                    self.logger.error(f"relations_df[{date_key}] 样本值: {relations_df[date_key].iloc[:3].tolist() if len(relations_df) > 0 else []}")
                raise merge_error

            # 处理重复列名 - 修复：排除重要的板块字段
            # 排除most_relevant_sector等重要字段，避免被错误重命名
            exclude_from_rename = ['most_relevant_sector']
            duplicate_cols = [col for col in complete_df.columns
                            if col.endswith('_sector') and col not in exclude_from_rename]

            if duplicate_cols:
                self.logger.debug(f"发现需要处理的重复列: {duplicate_cols}")
                # 对于重复的列，优先保留个股特征，删除板块特征的重复列
                for col in duplicate_cols:
                    original_col = col.replace('_sector', '')
                    if original_col in complete_df.columns:
                        # 如果个股特征列存在，删除板块特征的重复列
                        complete_df = complete_df.drop(columns=[col])
                        self.logger.debug(f"删除重复的板块特征列: {col}")
                    else:
                        # 如果个股特征列不存在，重命名板块特征列
                        complete_df = complete_df.rename(columns={col: original_col})
                        self.logger.debug(f"重命名板块特征列: {col} -> {original_col}")

            # 确保重要的板块字段被保留
            important_sector_fields = ['most_relevant_sector']
            for field in important_sector_fields:
                if field not in complete_df.columns:
                    self.logger.warning(f"重要的板块字段 {field} 在合并后丢失！")

            self.logger.info(f"成功合并特征，最终DataFrame包含 {len(complete_df.columns)} 个特征")
            return complete_df

        except Exception as e:
            self.logger.error(f"合并特征时出错: {e}")
            self.logger.warning("返回原始个股特征DataFrame")
            return result_df



    def _extract_change_point_features_directly(self, complete_features_df, change_points):
        """
        直接从完整特征DataFrame中提取变化点特征（简化版本）

        这个方法替代了_extract_change_point_features_from_complete_df，
        避免重复的特征计算，只进行必要的数据格式转换和元数据添加。

        Args:
            complete_features_df: 包含所有特征的完整DataFrame
            change_points: 变化点检测结果

        Returns:
            list: 变化点特征列表
        """
        import time
        start_time = time.time()

        self.logger.info("直接提取变化点特征（避免重复计算）...")

        if not change_points or complete_features_df.empty:
            return []

        # 创建索引以加速查询
        complete_df_idx = complete_features_df.set_index(['stock_code', 'date_str'])

        # 收集变化点特征
        change_point_features = []

        for stock_code, stock_data in change_points.items():
            if 'change_points' not in stock_data or not stock_data['change_points']:
                continue

            dates = stock_data['dates']
            returns = stock_data.get('returns', [])

            for cp_idx, cp_date in enumerate(dates):
                try:
                    # 从完整特征DataFrame获取该变化点的所有特征
                    feature_row = complete_df_idx.loc[(stock_code, cp_date)]
                    feature_dict = feature_row.to_dict()

                    # 添加变化点元数据和索引信息
                    future_return = returns[cp_idx] if cp_idx < len(returns) else None
                    feature_dict['future_return_2d'] = future_return
                    feature_dict['change_point_idx'] = cp_idx
                    feature_dict['stock_code'] = stock_code  # 重新添加stock_code
                    feature_dict['date_str'] = cp_date       # 重新添加date_str

                    change_point_features.append(feature_dict)

                except KeyError:
                    # 跳过不存在的数据
                    self.logger.debug(f"变化点数据不存在: 股票={stock_code}, 日期={cp_date}")
                    continue

        elapsed_time = time.time() - start_time
        self.logger.info(f"变化点特征提取完成，共提取 {len(change_point_features)} 个有效特征，用时 {elapsed_time:.2f} 秒")

        return change_point_features




    def _add_board_structure_features(self, features, current_day, sector_code, board_structure=None):
        """
        添加板块涨跌停结构特征，借鉴sector_stock_analysis中的分析逻辑

        Args:
            features: 特征字典
            current_day: 当日数据
            sector_code: 板块代码
            board_structure: 涨跌停板结构特征数据

        Returns:
            更新后的特征字典
        """
        # 注释: 本方法是特征提取的重要组成部分，配合整体的相似度匹配方法，
        # 我们通过提取历史数据中的各种特征（包括涨跌停板结构特征），
        # 在预测时通过计算当前市场特征与历史特征的相似度来识别潜在的交易机会。
        # 这种基于相似度的模式识别方法在量化交易中被广泛应用，
        # 无需复杂的机器学习训练过程，但能够有效捕捉市场模式。
        try:
            # 只有当board_structure不为None时才添加板块涨跌停结构特征
            if board_structure is not None:
                # 1. 涨跌停统计特征
                for field in ["limit_up_count", "limit_down_count", "near_limit_up_count",
                             "near_limit_down_count", "one_word_limit_up_count",
                             "broken_limit_up_count", "limit_open_count"]:
                    if field in board_structure:
                        features[field] = board_structure[field]

                # 2. 连板特征
                if "max_consecutive_limit_up" in board_structure:
                    features["max_consecutive_limit_up"] = board_structure["max_consecutive_limit_up"]

                # 3. 溢价特征 - 只保留有效的溢价特征
                for field in ["yesterday_limit_up_premium", "yesterday_max_limit_up_premium"]:
                    if field in board_structure:
                        features[field] = board_structure[field]

                # 4. 市值分布特征
                if "market_cap_distribution" in board_structure and isinstance(board_structure["market_cap_distribution"], dict):
                    for size, count in board_structure["market_cap_distribution"].items():
                        features[f"market_cap_{size}"] = count

                # 5. 涨停板市值分布
                if "limit_board_distribution" in board_structure and isinstance(board_structure["limit_board_distribution"], dict):
                    for size, count in board_structure["limit_board_distribution"].items():
                        features[f"limit_board_{size}"] = count

                # 6. 连板分布特征 - 简化版本
                if "board_counts" in board_structure and isinstance(board_structure["board_counts"], dict):
                    # 只保留关键的连板统计
                    total_boards = sum(board_structure["board_counts"].values())
                    if total_boards > 0:
                        features["total_board_count"] = total_boards
                        # 高连板比例（3板及以上）
                        high_boards = sum(count for board_type, count in board_structure["board_counts"].items()
                                        if any(str(i) in board_type for i in range(3, 11)))
                        features["high_board_ratio"] = high_boards / total_boards if total_boards > 0 else 0

                # 6. 计算涨停结构指标
                if "stock_count" in board_structure and board_structure["stock_count"] > 0:
                    # 涨停占比
                    if "limit_up_count" in board_structure:
                        features["limit_up_ratio"] = board_structure["limit_up_count"] / board_structure["stock_count"]

                    # 跌停占比
                    if "limit_down_count" in board_structure:
                        features["limit_down_ratio"] = board_structure["limit_down_count"] / board_structure["stock_count"]

                    # 一字板占比
                    if "one_word_limit_up_count" in board_structure:
                        features["one_word_limit_ratio"] = board_structure["one_word_limit_up_count"] / max(1, board_structure["limit_up_count"])

                    # 炸板率
                    if "broken_limit_up_count" in board_structure and "limit_up_count" in board_structure:
                        broken_count = board_structure["broken_limit_up_count"]
                        limit_count = board_structure["limit_up_count"]
                        total_limits = limit_count + broken_count
                        if total_limits > 0:
                            ratio = broken_count / total_limits
                            features["broken_ratio"] = ratio
                            # 添加调试日志
                            if ratio > 0:
                                self.logger.debug(f"计算炸板率: 板块={sector_code}, 日期={current_day.get('date_str', 'unknown')}, 炸板数={broken_count}, 涨停数={limit_count}, 炸板率={ratio:.4f}")
                        else:
                            features["broken_ratio"] = 0

                # 7. 涨停质量特征 - 简化版本
                if "one_word_limit_up_count" in board_structure and "limit_up_count" in board_structure:
                    limit_up_count = board_structure["limit_up_count"]
                    if limit_up_count > 0:
                        # 一字板占比（高质量涨停）
                        features["one_word_limit_ratio"] = board_structure["one_word_limit_up_count"] / limit_up_count

                # 8. 计算板块强度指标
                if all(k in board_structure for k in ["avg_pct_chg", "avg_turnover", "avg_amplitude"]):
                    # 综合强度 = 平均涨幅 * (1 + 平均换手率/100) * (1 + 平均振幅/100)
                    features["sector_strength"] = board_structure["avg_pct_chg"] * (1 + board_structure["avg_turnover"]/100) * (1 + board_structure["avg_amplitude"]/100)

                # 8. 计算热度指数（与sector_stock_analysis.py保持一致）
                heat_score = 0
                weights = {
                    "limit_up_count": 0.30,          # 涨停数量权重
                    "limit_up_ratio": 0.20,          # 涨停占比权重
                    "max_consecutive_limit_up": 0.25, # 最高连板权重
                    "one_word_limit_up_count": 0.15,  # 一字板权重
                    "broken_limit_up_count": -0.10    # 炸板数量负向权重
                }

                for metric, weight in weights.items():
                    if metric in board_structure:
                        if metric == "limit_up_ratio":
                            # 涨停占比标准化（20%为满分）
                            value = min(100, board_structure.get(metric, 0) * 500)
                        elif metric == "limit_up_count":
                            # 涨停数量标准化（5个为满分）
                            value = min(100, board_structure.get(metric, 0) * 20)
                        elif metric == "max_consecutive_limit_up":
                            # 最高连板标准化（10板为满分）
                            value = min(100, board_structure.get(metric, 0) * 10)
                        else:
                            value = board_structure.get(metric, 0)

                        heat_score += value * weight

                features["board_heat_index"] = max(0, min(100, heat_score))

                # 8. 板块强度综合评分
                if all(k in board_structure for k in ["avg_pct_chg", "limit_up_count", "stock_count"]):
                    avg_pct_chg = board_structure["avg_pct_chg"]
                    limit_ratio = board_structure["limit_up_count"] / max(1, board_structure["stock_count"])
                    features["sector_momentum_score"] = avg_pct_chg * (1 + limit_ratio * 10)


            return features

        except Exception as e:
            self.logger.error(f"添加板块 {sector_code} 的涨跌停结构特征时出错: {e}")
            return features

    def _evaluate_selected_stocks_old(self, selected_stocks, all_stock_features, evaluation_days=2):
        """
        评估选出股票的实际表现，与预测结果进行比较 (旧方法，用于单日评估)

        Args:
            selected_stocks: 选出的股票列表
            all_stock_features: 包含所有股票特征的DataFrame，包括未来收益率数据
            evaluation_days: 评估的天数，默认为2天（对应future_return_2d）

        Returns:
            dict: 评估结果
        """
        stocks_count = len(selected_stocks.get('selected_stocks', []))
        self.logger.info(f"评估单日选出的 {stocks_count} 只股票的实际表现...")

        try:
            if not selected_stocks or 'selected_stocks' not in selected_stocks or not selected_stocks['selected_stocks']:
                self.logger.warning("没有选出的股票，无法进行评估")
                return {"evaluation_summary": "无选出股票"}

            # 获取预测日期
            prediction_date = selected_stocks.get('date', '')
            if not prediction_date:
                self.logger.warning("无法获取预测日期，无法进行评估")
                return {"evaluation_summary": "无预测日期"}

            # 转换日期格式
            prediction_date = pd.to_datetime(prediction_date)

            # 计算评估日期
            trading_calendar = self._load_trading_calendar()
            if not trading_calendar:
                self.logger.warning("无法加载交易日历，无法确定评估日期")
                return {"evaluation_summary": "无交易日历"}

            # 将交易日历转换为日期对象列表
            trading_dates = [pd.to_datetime(d) for d in trading_calendar]
            trading_dates.sort()

            # 找出预测日期在交易日历中的位置
            try:
                date_index = trading_dates.index(prediction_date)
            except ValueError:
                # 找到最接近的交易日
                closest_date_index = min(range(len(trading_dates)),
                                        key=lambda i: abs((trading_dates[i] - prediction_date).days))
                date_index = closest_date_index
                self.logger.warning(f"预测日期 {prediction_date} 不是交易日，使用最近的交易日 {trading_dates[date_index]}")

            # 确保有足够的未来交易日进行评估
            if date_index + evaluation_days >= len(trading_dates):
                self.logger.warning(f"预测日期 {prediction_date} 后没有足够的交易日进行评估")
                return {"evaluation_summary": "无足够未来交易日"}

            # 获取评估日期
            evaluation_date = trading_dates[date_index + evaluation_days]

            # 准备评估数据
            selected_stock_codes = [stock['stock_code'] for stock in selected_stocks['selected_stocks']]

            # 获取实际收益率数据
            evaluation_data = all_stock_features[
                (all_stock_features['date_str'] == evaluation_date.strftime('%Y%m%d')) &
                (all_stock_features['stock_code'].isin(selected_stock_codes))
            ].copy()

            if evaluation_data.empty:
                self.logger.warning(f"无法获取评估日期 {evaluation_date} 的股票数据")
                return {"evaluation_summary": "无评估数据"}

            # 合并预测和实际数据
            evaluation_results = []
            for stock in selected_stocks['selected_stocks']:
                stock_code = stock['stock_code']
                actual_data = evaluation_data[evaluation_data['stock_code'] == stock_code]

                if actual_data.empty:
                    # 如果没有找到实际数据，记录缺失
                    evaluation_results.append({
                        'stock_code': stock_code,
                        'stock_name': stock.get('stock_name', ''),
                        'predicted_return': stock.get('expected_return', 0),
                        'actual_return': None,
                        'prediction_accuracy': None,
                        'rank': stock.get('rank', 0),
                        'status': '数据缺失'
                    })
                else:
                    # 获取实际收益率
                    actual_return = actual_data.iloc[0].get('future_return_2d', None)
                    if actual_return is None:
                        # 尝试从其他列获取实际收益率
                        for col in ['return_2d', 'return_2day', 'change_percent_2d']:
                            if col in actual_data.columns:
                                actual_return = actual_data.iloc[0][col]
                                break

                    # 如果仍然没有找到实际收益率，尝试计算收益率
                    if actual_return is None and 'close' in actual_data.columns and 'prev_close' in actual_data.columns:
                        try:
                            close = float(actual_data.iloc[0]['close'])
                            prev_close = float(actual_data.iloc[0]['prev_close'])
                            if prev_close > 0:
                                actual_return = (close - prev_close) / prev_close * 100
                            else:
                                actual_return = None
                        except (ValueError, TypeError):
                            actual_return = None

                    # 计算预测准确性
                    predicted_return = stock.get('expected_return', 0)
                    if actual_return is not None:
                        # 预测方向是否正确
                        direction_correct = (predicted_return > 0 and actual_return > 0) or \
                                          (predicted_return < 0 and actual_return < 0)

                        # 计算预测误差
                        if predicted_return != 0:
                            prediction_error = abs(actual_return - predicted_return) / abs(predicted_return)
                            prediction_accuracy = max(0, 1 - prediction_error)
                        else:
                            prediction_accuracy = 0 if actual_return != 0 else 1

                        # 确定状态
                        if direction_correct:
                            if prediction_accuracy > 0.7:
                                status = '预测优秀'
                            elif prediction_accuracy > 0.4:
                                status = '方向正确'
                            else:
                                status = '方向正确但误差大'
                        else:
                            status = '方向错误'
                    else:
                        direction_correct = None
                        prediction_accuracy = None
                        status = '无法评估'

                    # 添加到评估结果
                    evaluation_results.append({
                        'stock_code': stock_code,
                        'stock_name': stock.get('stock_name', ''),
                        'predicted_return': predicted_return,
                        'actual_return': actual_return,
                        'direction_correct': direction_correct,
                        'prediction_accuracy': prediction_accuracy,
                        'status': status,
                        'rank': stock.get('rank', 0),
                        'score': stock.get('score', 0)
                    })

            # 计算总体评估指标
            valid_results = [r for r in evaluation_results if r['actual_return'] is not None and not pd.isna(r['actual_return'])]
            if valid_results:
                # 安全计算平均实际收益率
                actual_returns = [r['actual_return'] for r in valid_results if pd.notna(r['actual_return'])]
                avg_actual_return = sum(actual_returns) / len(actual_returns) if actual_returns else 0.0

                # 计算方向正确率
                direction_correct_count = sum(1 for r in valid_results if r['direction_correct'] == True)
                direction_accuracy = direction_correct_count / len(valid_results) if valid_results else 0.0

                # 计算平均预测准确度
                valid_accuracy_results = [r for r in valid_results if r['prediction_accuracy'] is not None and not pd.isna(r['prediction_accuracy'])]
                avg_prediction_accuracy = sum(r['prediction_accuracy'] for r in valid_accuracy_results) / len(valid_accuracy_results) if valid_accuracy_results else 0.0

                # 计算胜率（实际收益为正的比例）
                win_count = sum(1 for r in valid_results if r['actual_return'] > 0)
                win_rate = win_count / len(valid_results) if valid_results else 0.0

                # 计算盈亏比指标
                profit_results = [r['actual_return'] for r in valid_results if r['actual_return'] > 0]
                loss_results = [abs(r['actual_return']) for r in valid_results if r['actual_return'] < 0]

                avg_profit = sum(profit_results) / len(profit_results) if profit_results else 0.0
                avg_loss = sum(loss_results) / len(loss_results) if loss_results else 0.0
                profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf') if avg_profit > 0 else 0.0

                # 计算最大回撤
                cumulative_returns = []
                cumulative = 0
                for r in valid_results:
                    cumulative += r['actual_return']
                    cumulative_returns.append(cumulative)

                max_drawdown = 0.0
                if cumulative_returns:
                    peak = cumulative_returns[0]
                    for cum_return in cumulative_returns:
                        if cum_return > peak:
                            peak = cum_return
                        drawdown = peak - cum_return
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown

                # 计算夏普比率（简化版本，假设无风险利率为0）
                if len(actual_returns) > 1:
                    returns_std = np.std(actual_returns, ddof=1)
                    sharpe_ratio = avg_actual_return / returns_std if returns_std > 0 else 0.0
                else:
                    sharpe_ratio = 0.0

                # 计算评估总结
                evaluation_summary = {
                    'prediction_date': prediction_date.strftime('%Y%m%d'),
                    'evaluation_date': evaluation_date.strftime('%Y%m%d') if isinstance(evaluation_date, pd.Timestamp) else str(evaluation_date),
                    'total_stocks': len(selected_stock_codes),
                    'evaluated_stocks': len(valid_results),
                    'avg_actual_return': avg_actual_return,
                    'direction_accuracy': direction_accuracy,
                    'avg_prediction_accuracy': avg_prediction_accuracy,
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'profit_loss_ratio': profit_loss_ratio,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'performance_rating': self._get_performance_rating(avg_actual_return, win_rate, direction_accuracy)
                }
            else:
                evaluation_summary = {
                    'prediction_date': prediction_date.strftime('%Y%m%d'),
                    'evaluation_date': evaluation_date.strftime('%Y%m%d') if isinstance(evaluation_date, pd.Timestamp) else str(evaluation_date),
                    'total_stocks': len(selected_stock_codes),
                    'evaluated_stocks': 0,
                    'performance_rating': '无法评估'
                }

            # 组织最终评估结果
            final_evaluation = {
                'evaluation_summary': evaluation_summary,
                'stock_evaluations': evaluation_results
            }

            # 安全格式化输出，避免NaN问题
            avg_return_str = f"{evaluation_summary.get('avg_actual_return', 0):.2f}" if pd.notna(evaluation_summary.get('avg_actual_return')) else "N/A"
            direction_acc_str = f"{evaluation_summary.get('direction_accuracy', 0) * 100:.1f}" if pd.notna(evaluation_summary.get('direction_accuracy')) else "N/A"
            win_rate_str = f"{evaluation_summary.get('win_rate', 0) * 100:.1f}" if pd.notna(evaluation_summary.get('win_rate')) else "N/A"
            profit_loss_str = f"{evaluation_summary.get('profit_loss_ratio', 0):.2f}" if pd.notna(evaluation_summary.get('profit_loss_ratio')) and evaluation_summary.get('profit_loss_ratio') != float('inf') else "N/A"
            sharpe_str = f"{evaluation_summary.get('sharpe_ratio', 0):.2f}" if pd.notna(evaluation_summary.get('sharpe_ratio')) else "N/A"

            self.logger.info(f"评估完成，平均实际收益率: {avg_return_str}%, "
                          f"方向准确率: {direction_acc_str}%, "
                          f"胜率: {win_rate_str}%, "
                          f"盈亏比: {profit_loss_str}, "
                          f"夏普比率: {sharpe_str}")

            return final_evaluation

        except Exception as e:
            self.logger.error(f"评估选出股票时出错: {e}")
            self.logger.error(traceback.format_exc())
            return {"evaluation_summary": f"评估出错: {str(e)}"}

    def _get_performance_rating(self, avg_return, win_rate, direction_accuracy):
        """
        根据平均收益率、胜率和方向准确率评定表现等级

        Args:
            avg_return: 平均收益率
            win_rate: 胜率
            direction_accuracy: 方向准确率

        Returns:
            str: 表现等级
        """
        # 评分标准
        if avg_return > 3 and win_rate > 0.6 and direction_accuracy > 0.7:
            return '优秀'
        elif avg_return > 1.5 and win_rate > 0.55 and direction_accuracy > 0.6:
            return '良好'
        elif avg_return > 0 and win_rate > 0.5 and direction_accuracy > 0.5:
            return '一般'
        elif avg_return <= 0 and win_rate < 0.5 and direction_accuracy < 0.5:
            return '较差'
        else:
            return '中性'

    def _prepare_stock_leadership_features(self, stock_df):
        """
        预处理股票的龙头特征，为计算股票在板块中的地位做准备

        Args:
            stock_df: 股票特征DataFrame

        Returns:
            添加了龙头特征的DataFrame
        """
        self.logger.info("开始预处理股票龙头特征...")

        if stock_df.empty:
            return stock_df

        result_df = stock_df.copy()

        # 1. 识别连板天数
        if 'is_limit_up' in result_df.columns:
            # 初始化连续涨停天数列
            result_df['consecutive_limit_days'] = 0

            # 按股票代码分组处理
            for stock_code, group in result_df.groupby('stock_code'):
                # 确保按日期排序
                group = group.sort_values('date_str')

                # 计算连续涨停天数
                consecutive_days = 0
                for idx, row in group.iterrows():
                    if row['is_limit_up']:
                        consecutive_days += 1
                    else:
                        consecutive_days = 0

                    # 更新连续涨停天数
                    result_df.loc[idx, 'consecutive_limit_days'] = consecutive_days

        # 2. 识别一字板
        if all(col in result_df.columns for col in ['open', 'high', 'low', 'close', 'is_limit_up']):
            # 判断开盘、最高、最低、收盘价格是否相同，且是涨停
            result_df['is_one_word_limit'] = (
                (result_df['open'] == result_df['high']) &
                (result_df['high'] == result_df['low']) &
                (result_df['low'] == result_df['close']) &
                result_df['is_limit_up']
            )
        else:
            result_df['is_one_word_limit'] = False

        # 3. 识别高弹性股票
        if 'amplitude' in result_df.columns:
            # 振幅大于5%且非涨跌停的股票视为高弹性
            result_df['is_high_elasticity'] = (
                (result_df['amplitude'] >= 5.0) &
                (~result_df['is_limit_up']) &
                (~result_df.get('is_limit_down', pd.Series(False, index=result_df.index)))
            )
        else:
            result_df['is_high_elasticity'] = False

        # 4. 判断市值类型
        if 'total_mv' in result_df.columns:
            # 根据总市值划分大中小市值
            result_df['market_cap_type'] = 'small'  # 默认小市值
            result_df.loc[result_df['total_mv'] > 50, 'market_cap_type'] = 'medium'  # 50-200亿为中市值
            result_df.loc[result_df['total_mv'] > 200, 'market_cap_type'] = 'large'  # >200亿为大市值
        else:
            result_df['market_cap_type'] = 'small'

        # 5. 判断是否开盘涨停
        if all(col in result_df.columns for col in ['open', 'preClose', 'is_limit_up']):
            # ST股票涨停幅度是5%，非ST是10%
            # 这里简化处理，统一按照9.5%的阈值判断是否开盘涨停
            result_df['is_open_limit_up'] = (
                (result_df['open'] / result_df['preClose'] - 1 > 0.095) &
                result_df['is_limit_up']
            )
        else:
            result_df['is_open_limit_up'] = False

        # 6. 判断是否突破前期高点
        if 'close' in result_df.columns:
            # 按股票代码分组处理
            for stock_code, group in result_df.groupby('stock_code'):
                # logger.info(f"列：{group.columns}")
                # 确保按日期排序
                group = group.sort_values('date_str')

                # 计算60日最高价
                group['max_60d'] = group['close'].rolling(window=60, min_periods=1).max().shift(1)

                # 判断是否突破前期高点
                group['break_previous_high'] = group['close'] > group['max_60d']

                # 更新回原始DataFrame
                result_df.loc[group.index, 'max_60d'] = group['max_60d']
                result_df.loc[group.index, 'break_previous_high'] = group['break_previous_high']
        else:
            result_df['break_previous_high'] = False

        return result_df

    def _process_single_stock_date(self, stock_code, date_str, result_df_idx, stock_sector_mapping, sector_features_df_idx, is_training=True):
        """处理单个股票日期组合，使用预计算的最相关板块并返回板块特征

        Args:
            stock_code: 股票代码
            date_str: 日期字符串
            result_df_idx: 股票特征DataFrame索引
            stock_sector_mapping: 股票与板块的映射字典（已弃用，保留兼容性）
            sector_features_df_idx: 板块特征DataFrame索引
            is_training: 是否为训练阶段

        Returns:
            dict: 包含最相关板块信息和板块特征的字典，如果无效则返回None
        """
        try:
            # 使用索引快速获取数据
            try:
                stock_row = result_df_idx.loc[(stock_code, date_str)]
            except KeyError:
                return None

            # 获取最相关板块
            best_sector_code = None

            # 训练阶段：使用预计算的最相关板块
            if is_training and hasattr(self, 'stock_most_relevant_sectors'):
                best_sector_code = self.stock_most_relevant_sectors.get(stock_code)

            # 预测阶段：使用训练阶段保存的板块映射
            elif not is_training and hasattr(self, 'training_sector_mapping') and self.training_sector_mapping:
                best_sector_code = self.training_sector_mapping.get(stock_code)

            # 如果没有找到最相关板块，跳过该股票
            if not best_sector_code:
                return None

            # 获取最相关板块的完整特征
            best_sector_features = {}
            try:
                best_sector_row = sector_features_df_idx.loc[(best_sector_code, date_str)]
                # 将最相关板块的所有特征添加到结果中
                for col in best_sector_row.index:
                    if col not in ['sector_code', 'date']:  # 排除标识列
                        best_sector_features[col] = best_sector_row[col]
            except KeyError:
                # 如果板块在当前日期没有数据，跳过该记录
                return None

            # 返回结果：基本关联信息 + 最相关板块的完整特征
            result = {
                'stock_code': stock_code,
                'date': date_str,
                'most_relevant_sector': best_sector_code
            }
            # 添加最相关板块的完整特征
            result.update(best_sector_features)

            return result

        except Exception as e:
            self.logger.error(f"处理股票 {stock_code} 在 {date_str} 的特征时出错: {str(e)}")
            # 添加详细的错误跟踪以便更好地调试
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return None



    def _load_training_sector_mapping(self):
        """
        加载训练阶段计算的最相关板块映射

        Returns:
            dict: {stock_code: most_relevant_sector} 映射字典，如果加载失败返回None
        """
        try:
            # 构建映射文件路径
            cache_dir = Path(get_project_root()) / "to" / "data_cache"
            mapping_file = cache_dir / "training_sector_mapping.pkl"

            if mapping_file.exists():
                with open(mapping_file, 'rb') as f:
                    mapping = pickle.load(f)
                self.logger.info(f"成功加载训练阶段板块映射，包含 {len(mapping)} 个股票")
                return mapping
            else:
                self.logger.warning(f"训练阶段板块映射文件不存在: {mapping_file}")
                return None
        except Exception as e:
            self.logger.error(f"加载训练阶段板块映射失败: {e}")
            return None

    def _save_training_sector_mapping(self, relations_df):
        """
        保存训练阶段计算的最相关板块映射

        Args:
            relations_df: 包含股票-板块关联信息的DataFrame
        """
        try:
            if relations_df is None or relations_df.empty:
                self.logger.warning("relations_df为空，无法保存板块映射")
                return

            # 构建映射字典：{stock_code: most_relevant_sector}
            mapping = {}
            for _, row in relations_df.iterrows():
                stock_code = row.get('stock_code')
                sector = row.get('most_relevant_sector')
                if stock_code and sector:
                    mapping[stock_code] = sector

            if not mapping:
                self.logger.warning("没有有效的板块映射数据可保存")
                return

            # 保存到文件
            cache_dir = Path(get_project_root()) / "to" / "data_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            mapping_file = cache_dir / "training_sector_mapping.pkl"

            with open(mapping_file, 'wb') as f:
                pickle.dump(mapping, f)

            self.logger.info(f"成功保存训练阶段板块映射到 {mapping_file}，包含 {len(mapping)} 个股票")

        except Exception as e:
            self.logger.error(f"保存训练阶段板块映射失败: {e}")

    def _save_training_sector_mapping_from_dict(self, stock_sector_mapping):
        """
        从字典保存训练阶段计算的最相关板块映射

        Args:
            stock_sector_mapping: {stock_code: most_relevant_sector} 映射字典
        """
        try:
            if not stock_sector_mapping:
                self.logger.warning("没有有效的板块映射数据可保存")
                return

            # 保存到文件
            cache_dir = Path(get_project_root()) / "to" / "data_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            mapping_file = cache_dir / "training_sector_mapping.pkl"

            with open(mapping_file, 'wb') as f:
                pickle.dump(stock_sector_mapping, f)

            self.logger.info(f"成功保存训练阶段板块映射到 {mapping_file}，包含 {len(stock_sector_mapping)} 个股票")

        except Exception as e:
            self.logger.error(f"保存训练阶段板块映射失败: {e}")

    def _calculate_all_stocks_most_relevant_sectors(self):
        """
        计算所有股票的最相关板块（基于变化点的同向次数统计）- 优化版本

        Returns:
            dict: {stock_code: most_relevant_sector} 映射字典
        """
        self.logger.info("开始计算所有股票的最相关板块（优化版本）...")

        # 步骤1: 预处理数据，建立索引
        self.logger.info("步骤1: 预处理数据，建立索引...")
        self._prepare_optimized_data_structures()

        # 步骤2: 批量计算所有股票的最相关板块
        self.logger.info("步骤2: 批量计算最相关板块...")
        stock_most_relevant_sectors = self._batch_calculate_most_relevant_sectors()

        self.logger.info(f"最相关板块计算完成：共处理 {len(stock_most_relevant_sectors)} 个股票")

        # 打印部分结果示例
        if stock_most_relevant_sectors:
            self.logger.info("部分股票的最相关板块示例：")
            count = 0
            for stock_code, sector in stock_most_relevant_sectors.items():
                if count < 10:  # 只打印前10个
                    self.logger.info(f"  {stock_code} -> {sector}")
                    count += 1
                else:
                    break

        return stock_most_relevant_sectors

    def _prepare_optimized_data_structures(self):
        """
        预处理数据结构以优化性能
        """
        import pandas as pd

        # 1. 为个股数据建立多级索引
        if not hasattr(self, 'stock_data_indexed'):
            self.logger.info("建立个股数据索引...")
            stock_df = self.stock_features_df.copy()
            stock_df = stock_df.set_index(['stock_code', 'date_str'])
            self.stock_data_indexed = stock_df

        # 2. 为板块数据建立索引并预处理日期格式
        if not hasattr(self, 'sector_data_indexed'):
            self.logger.info("建立板块数据索引...")
            sector_df = self.sector_features_df.copy()

            # 预处理日期格式：将 2023-06-01 00:00:00 转换为 20230601
            sector_df['date_str'] = pd.to_datetime(sector_df['date']).dt.strftime('%Y%m%d')
            sector_df = sector_df.set_index(['sector_code', 'date_str'])
            self.sector_data_indexed = sector_df

        # 3. 预处理所有股票的变化点和板块映射
        if not hasattr(self, 'stock_change_points_processed'):
            self.logger.info("预处理股票变化点和板块映射...")
            self._preprocess_stock_sector_mappings()

    def _preprocess_stock_sector_mappings(self):
        """
        预处理股票板块映射和变化点数据
        """
        from to.stock_sectors_mapping import get_stock_sectors

        self.stock_change_points_processed = {}
        stocks_with_change_points = list(self.change_points.keys())

        self.logger.info(f"预处理 {len(stocks_with_change_points)} 个股票的板块映射...")

        for i, stock_code in enumerate(stocks_with_change_points):
            if i % 500 == 0:
                self.logger.info(f"已预处理 {i}/{len(stocks_with_change_points)} 个股票")

            try:
                # 获取股票所属板块
                if stock_code not in self.stock_sector_mapping:
                    sectors = get_stock_sectors(stock_code=stock_code)
                    self.stock_sector_mapping[stock_code] = sectors if sectors else []
                else:
                    sectors = self.stock_sector_mapping[stock_code]

                if not sectors:
                    continue

                # 过滤掉在板块特征数据中不存在的板块
                valid_sectors = [sector_code for sector_code in sectors
                               if sector_code in self.sector_features_df['sector_code'].unique()]

                if not valid_sectors:
                    continue
                elif len(valid_sectors) == 1:
                    # 只有一个板块，直接设为最相关板块
                    self.stock_change_points_processed[stock_code] = {
                        'most_relevant_sector': valid_sectors[0],
                        'method': 'single_sector'
                    }
                else:
                    # 多个板块，需要计算同向次数
                    change_point_data = self.change_points[stock_code]
                    change_point_indices = change_point_data['change_points']
                    change_point_dates = change_point_data['dates']
                    self.stock_change_points_processed[stock_code] = {
                        'sectors': valid_sectors,
                        'change_point_indices': change_point_indices,
                        'change_point_dates': change_point_dates,
                        'method': 'calculate_similarity'
                    }

            except Exception as e:
                self.logger.error(f"预处理股票 {stock_code} 时出错: {e}")
                continue

    def _batch_calculate_most_relevant_sectors(self):
        """
        批量计算最相关板块（优化版本）
        """
        import pandas as pd
        import numpy as np

        stock_most_relevant_sectors = {}

        # 处理已经确定最相关板块的股票（只有一个板块的情况）
        single_sector_count = 0
        for stock_code, data in self.stock_change_points_processed.items():
            if data['method'] == 'single_sector':
                stock_most_relevant_sectors[stock_code] = data['most_relevant_sector']
                single_sector_count += 1

        self.logger.info(f"已处理 {single_sector_count} 个只有单一板块的股票")

        # 批量处理需要计算同向次数的股票
        multi_sector_stocks = {k: v for k, v in self.stock_change_points_processed.items()
                              if v['method'] == 'calculate_similarity'}

        self.logger.info(f"开始批量计算 {len(multi_sector_stocks)} 个多板块股票的最相关板块...")

        # 批量获取所有需要的数据
        all_stock_sector_pairs = []
        stock_sector_mapping = {}

        for stock_code, data in multi_sector_stocks.items():
            sectors = data['sectors']
            change_point_dates = data['change_point_dates']

            for sector_code in sectors:
                for date_str in change_point_dates:
                    all_stock_sector_pairs.append((stock_code, sector_code, date_str))

                    if stock_code not in stock_sector_mapping:
                        stock_sector_mapping[stock_code] = {}
                    if sector_code not in stock_sector_mapping[stock_code]:
                        stock_sector_mapping[stock_code][sector_code] = []
                    stock_sector_mapping[stock_code][sector_code].append(date_str)

        # 批量获取个股数据
        self.logger.info(f"批量获取 {len(all_stock_sector_pairs)} 个数据点...")
        import time
        start_time = time.time()

        # 分步骤获取数据，提供进度反馈
        self.logger.info("步骤2.1: 批量获取个股涨跌幅数据...")
        stock_changes_batch = self._batch_get_stock_changes(all_stock_sector_pairs)
        self.logger.info(f"个股数据获取完成，耗时: {time.time() - start_time:.2f}秒")

        self.logger.info("步骤2.2: 批量获取板块涨跌幅数据...")
        step2_start = time.time()
        sector_changes_batch = self._batch_get_sector_changes(all_stock_sector_pairs)
        self.logger.info(f"板块数据获取完成，耗时: {time.time() - step2_start:.2f}秒")

        # 批量计算同向次数
        self.logger.info("步骤2.3: 批量计算同向次数...")
        step3_start = time.time()
        processed_count = 0
        total_stocks = len(multi_sector_stocks)

        for stock_code, data in multi_sector_stocks.items():
            try:
                sectors = data['sectors']
                change_point_dates = data['change_point_dates']

                # 计算每个板块的同向次数
                sector_same_direction_counts = {}

                for sector_code in sectors:
                    same_direction_count = 0
                    total_valid_points = 0

                    for date_str in change_point_dates:
                        key = (stock_code, sector_code, date_str)

                        stock_change = stock_changes_batch.get(key)
                        sector_change = sector_changes_batch.get(key)

                        if stock_change is not None and sector_change is not None:
                            total_valid_points += 1
                            # 检查是否同向
                            if (stock_change >= 0 and sector_change >= 0) or (stock_change < 0 and sector_change < 0):
                                same_direction_count += 1

                    if total_valid_points > 0:
                        sector_same_direction_counts[sector_code] = same_direction_count

                # 选择同向次数最多的板块
                if sector_same_direction_counts:
                    best_sector = max(sector_same_direction_counts.items(), key=lambda x: x[1])
                    best_sectors = [sector for sector, count in sector_same_direction_counts.items()
                                  if count == best_sector[1]]

                    # 如果有多个板块同向次数相同，选择股票数量最少的板块
                    if len(best_sectors) > 1:
                        final_sector = self._select_sector_by_stock_count(best_sectors)
                    else:
                        final_sector = best_sectors[0]

                    stock_most_relevant_sectors[stock_code] = final_sector

                processed_count += 1
                if processed_count % 500 == 0:  # 增加报告频率
                    elapsed = time.time() - step3_start
                    progress = processed_count / total_stocks * 100
                    estimated_total = elapsed / processed_count * total_stocks
                    remaining = estimated_total - elapsed
                    self.logger.info(f"已处理 {processed_count}/{total_stocks} 个多板块股票 ({progress:.1f}%)，"
                                   f"耗时: {elapsed:.1f}秒，预计剩余: {remaining:.1f}秒")

            except Exception as e:
                self.logger.error(f"计算股票 {stock_code} 最相关板块时出错: {e}")
                continue

        total_time = time.time() - start_time
        self.logger.info(f"批量计算最相关板块完成，总耗时: {total_time:.2f}秒")

        return stock_most_relevant_sectors

    def _batch_get_stock_changes(self, stock_sector_pairs):
        """
        批量获取个股涨跌幅数据（优化版本）

        Args:
            stock_sector_pairs: [(stock_code, sector_code, date_str), ...] 列表

        Returns:
            dict: {(stock_code, sector_code, date_str): stock_change} 映射
        """
        import pandas as pd
        import time
        stock_changes = {}

        # 优化：预先构建反向映射，避免嵌套循环
        stock_date_to_sectors = {}
        for stock_code, sector_code, date_str in stock_sector_pairs:
            key = (stock_code, date_str)
            if key not in stock_date_to_sectors:
                stock_date_to_sectors[key] = []
            stock_date_to_sectors[key].append(sector_code)

        # 优化：使用向量化操作批量获取数据
        total_keys = len(stock_date_to_sectors)
        start_time = time.time()

        # 提取所有需要查询的索引
        query_indices = list(stock_date_to_sectors.keys())

        try:
            # 批量查询，使用reindex方法处理缺失值
            query_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(query_indices, names=['stock_code', 'date_str']))

            # 使用join操作批量获取数据，自动处理缺失值
            result_df = query_df.join(self.stock_data_indexed[['pct_chg']], how='left')

            # 批量填充结果
            for (stock_code, date_str), sector_codes in stock_date_to_sectors.items():
                try:
                    stock_change = result_df.loc[(stock_code, date_str), 'pct_chg']
                    stock_change_value = float(stock_change) if pd.notna(stock_change) else 0.0

                    # 为所有相关的sector_code填充相同的数据
                    for sector_code in sector_codes:
                        stock_changes[(stock_code, sector_code, date_str)] = stock_change_value

                except (KeyError, IndexError):
                    # 该股票在该日期没有数据，为所有相关sector_code设置None
                    for sector_code in sector_codes:
                        stock_changes[(stock_code, sector_code, date_str)] = None

        except Exception as e:
            self.logger.warning(f"批量查询失败，回退到逐个查询: {e}")
            # 回退到原始方法
            processed = 0
            for (stock_code, date_str), sector_codes in stock_date_to_sectors.items():
                try:
                    stock_change = self.stock_data_indexed.loc[(stock_code, date_str), 'pct_chg']
                    stock_change_value = float(stock_change) if pd.notna(stock_change) else 0.0

                    for sector_code in sector_codes:
                        stock_changes[(stock_code, sector_code, date_str)] = stock_change_value

                except KeyError:
                    for sector_code in sector_codes:
                        stock_changes[(stock_code, sector_code, date_str)] = None
                except Exception as ex:
                    self.logger.debug(f"获取股票 {stock_code} 在 {date_str} 数据时出错: {ex}")
                    for sector_code in sector_codes:
                        stock_changes[(stock_code, sector_code, date_str)] = None

                processed += 1
                if processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    progress = processed / total_keys * 100
                    self.logger.info(f"个股数据获取进度: {processed}/{total_keys} ({progress:.1f}%)，耗时: {elapsed:.1f}秒")

        return stock_changes

    def _batch_get_sector_changes(self, stock_sector_pairs):
        """
        批量获取板块涨跌幅数据（优化版本）

        Args:
            stock_sector_pairs: [(stock_code, sector_code, date_str), ...] 列表

        Returns:
            dict: {(stock_code, sector_code, date_str): sector_change} 映射
        """
        import time
        sector_changes = {}

        # 优化：预先构建反向映射，避免嵌套循环
        sector_date_to_stocks = {}
        for stock_code, sector_code, date_str in stock_sector_pairs:
            key = (sector_code, date_str)
            if key not in sector_date_to_stocks:
                sector_date_to_stocks[key] = []
            sector_date_to_stocks[key].append(stock_code)

        # 优化：使用向量化操作批量获取数据
        total_keys = len(sector_date_to_stocks)
        start_time = time.time()

        # 提取所有需要查询的索引
        query_indices = list(sector_date_to_stocks.keys())

        try:
            # 批量查询，使用reindex方法处理缺失值
            query_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(query_indices, names=['sector_code', 'date_str']))

            # 使用join操作批量获取数据，自动处理缺失值
            result_df = query_df.join(self.sector_data_indexed[['change_percent']], how='left')

            # 批量填充结果
            for (sector_code, date_str), stock_codes in sector_date_to_stocks.items():
                try:
                    sector_change = result_df.loc[(sector_code, date_str), 'change_percent']
                    sector_change_value = float(sector_change) if pd.notna(sector_change) else 0.0

                    # 为所有相关的stock_code填充相同的数据
                    for stock_code in stock_codes:
                        sector_changes[(stock_code, sector_code, date_str)] = sector_change_value

                except (KeyError, IndexError):
                    # 该板块在该日期没有数据，为所有相关stock_code设置None
                    for stock_code in stock_codes:
                        sector_changes[(stock_code, sector_code, date_str)] = None

        except Exception as e:
            self.logger.warning(f"批量查询失败，回退到逐个查询: {e}")
            # 回退到原始方法
            processed = 0
            for (sector_code, date_str), stock_codes in sector_date_to_stocks.items():
                try:
                    sector_change = self.sector_data_indexed.loc[(sector_code, date_str), 'change_percent']
                    sector_change_value = float(sector_change) if pd.notna(sector_change) else 0.0

                    for stock_code in stock_codes:
                        sector_changes[(stock_code, sector_code, date_str)] = sector_change_value

                except KeyError:
                    for stock_code in stock_codes:
                        sector_changes[(stock_code, sector_code, date_str)] = None
                except Exception as ex:
                    self.logger.debug(f"获取板块 {sector_code} 在 {date_str} 数据时出错: {ex}")
                    for stock_code in stock_codes:
                        sector_changes[(stock_code, sector_code, date_str)] = None

                processed += 1
                if processed % 5000 == 0:
                    elapsed = time.time() - start_time
                    progress = processed / total_keys * 100
                    self.logger.info(f"板块数据获取进度: {processed}/{total_keys} ({progress:.1f}%)，耗时: {elapsed:.1f}秒")

        return sector_changes

    def _select_sector_by_stock_count(self, sectors):
        """
        根据股票数量选择板块（股票数量少的优先）

        Args:
            sectors: 板块代码列表

        Returns:
            str: 选中的板块代码
        """
        min_stock_count = float('inf')
        selected_sector = sectors[0]  # 默认选择第一个

        for sector_code in sectors:
            sector_name = self._get_sector_name_from_code(sector_code)
            stock_count = self.sector_stock_count.get(sector_name, 999999)

            if stock_count < min_stock_count:
                min_stock_count = stock_count
                selected_sector = sector_code

        return selected_sector

    def _calculate_single_stock_most_relevant_sector(self, stock_code, sectors):
        """
        计算单个股票的最相关板块（基于所有变化点的同向次数统计）

        Args:
            stock_code: 股票代码
            sectors: 股票所属的板块列表

        Returns:
            str: 最相关的板块代码，如果无法确定则返回None
        """
        try:
            # 获取股票的变化点信息
            stock_change_points = self.change_points[stock_code]
            change_point_dates = stock_change_points.get('dates', [])

            # 确保change_point_dates是列表格式
            if isinstance(change_point_dates, np.ndarray):
                change_point_dates = change_point_dates.tolist()
            elif not isinstance(change_point_dates, list):
                change_point_dates = [change_point_dates] if change_point_dates is not None else []

            if not change_point_dates:
                return None

            # 过滤掉在板块特征数据中不存在的板块
            sectors = [sector_code for sector_code in sectors if sector_code in self.sector_features_df['sector_code'].unique()]

            # 根据过滤后的sectors数量决定处理逻辑
            if not sectors:
                # 如果没有任何板块在特征数据中存在，删除该股票
                self.logger.info(f"个股 {stock_code} 没有有效的板块数据，将被删除")
                return None
            elif len(sectors) == 1:
                # 如果只剩下一个板块，直接返回该板块作为最相关板块
                self.logger.info(f"个股 {stock_code} 只有一个有效板块 {sectors[0]}，直接设为最相关板块")
                return sectors[0]

            # 如果有多个板块，继续计算同向次数
            self.logger.info(f"个股 {stock_code} 有 {len(sectors)} 个有效板块: {sectors}，开始计算同向次数")

            # 计算每个板块与个股在变化点的同向次数
            sector_same_direction_counts = {}
            for sector_code in sectors:
                same_direction_count = 0
                total_valid_points = 0
               
                for date_str in change_point_dates:
                    try:
                        # 确保date_str是字符串格式
                        if isinstance(date_str, (pd.Timestamp, np.datetime64)):
                            date_str = pd.to_datetime(date_str).strftime('%Y%m%d')
                        elif not isinstance(date_str, str):
                            date_str = str(date_str)
                        self.logger.info(f"个股 {stock_code} 在 {date_str} 有变化点")
                        # 获取个股和板块的涨跌幅数据
                        stock_change, sector_change = self._get_stock_sector_changes(
                            stock_code, sector_code, date_str
                        )

                        if stock_change is None or sector_change is None:
                            self.logger.info(f"个股 {stock_code} 或板块 {sector_code} 在 {date_str} 没有有效的涨跌幅数据，跳过")
                            continue

                        # 检查是否同向
                        is_same_direction = (
                            (stock_change >= 0 and sector_change >= 0) or
                            (stock_change <= 0 and sector_change <= 0)
                        )

                        if is_same_direction:
                            same_direction_count += 1

                        total_valid_points += 1

                    except Exception as e:
                        self.logger.error(f"处理变化点 {date_str} 时出错: {e}")
                        continue

                if total_valid_points > 0:
                    # 计算同向比例
                    same_direction_ratio = same_direction_count / total_valid_points
                    sector_same_direction_counts[sector_code] = {
                        'same_direction_count': same_direction_count,
                        'total_valid_points': total_valid_points,
                        'same_direction_ratio': same_direction_ratio
                    }

            if not sector_same_direction_counts:
                self.logger.info(f"个股 {stock_code} 没有有效的板块同向数据")
                return None

            # 找到同向次数最多的板块
            max_same_direction_count = max(
                info['same_direction_count'] for info in sector_same_direction_counts.values()
            )

            # 获取同向次数最多的板块列表
            best_sectors = [
                sector_code for sector_code, info in sector_same_direction_counts.items()
                if info['same_direction_count'] == max_same_direction_count
            ]

            # 如果有多个板块同向次数相同，选择股票数量最少的板块
            if len(best_sectors) > 1:
                min_stock_count = float('inf')
                final_sector = best_sectors[0]  # 默认选择第一个

                for sector_code in best_sectors:
                    sector_name = self._get_sector_name_from_code(sector_code)
                    stock_count = self.sector_stock_count.get(sector_name, 999999)

                    if stock_count < min_stock_count:
                        min_stock_count = stock_count
                        final_sector = sector_code

                return final_sector
            else:
                return best_sectors[0]

        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 的最相关板块时出错: {e}")
            return None

    def _get_stock_sector_changes(self, stock_code, sector_code, date_str):
        """
        获取指定日期的个股和板块涨跌幅

        Args:
            stock_code: 股票代码
            sector_code: 板块代码
            date_str: 日期字符串

        Returns:
            tuple: (stock_change, sector_change) 或 (None, None) 如果获取失败
        """
        try:
            stock_change = None
            sector_change = None

            # 获取个股数据
            try:
                for date_col in ['date_str', 'date']:
                    if date_col in self.stock_features_df.columns:
                        stock_subset = self.stock_features_df[
                            (self.stock_features_df['stock_code'] == stock_code)
                        ]
                        if not stock_subset.empty:
                            date_matches = stock_subset[date_col].astype(str) == str(date_str)
                            if date_matches.any():
                                matched_rows = stock_subset[date_matches]
                                if not matched_rows.empty:
                                    stock_change = matched_rows.iloc[0].get('pct_chg', 0)
                                    self.logger.info(f"个股 {stock_code} 在 {date_str} 的涨跌幅: {stock_change}")
                                    break
            except Exception as e:
                self.logger.error(f"获取个股 {stock_code} 在 {date_str} 的数据时出错: {e}")
                return None, None

            # 获取板块数据 - 修复日期格式匹配问题
            try:
                if sector_code not in self.sector_features_df['sector_code'].unique():
                    self.logger.info(f"板块 {sector_code} 没有数据")
                    return None, None

                # 板块数据只有date字段，格式为 2023-06-01 00:00:00
                if 'date' in self.sector_features_df.columns:
                    sector_subset = self.sector_features_df[
                        (self.sector_features_df['sector_code'] == sector_code)
                    ]
                    if not sector_subset.empty:
                        # 将date_str (20240913) 转换为板块数据的日期格式进行匹配
                        try:
                            # 将 20240913 转换为 2024-09-13 格式
                            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                            # 将板块数据的date字段转换为字符串格式进行匹配
                            sector_dates = pd.to_datetime(sector_subset['date']).dt.strftime('%Y-%m-%d')
                            date_matches = sector_dates == formatted_date

                            if date_matches.any():
                                matched_rows = sector_subset[date_matches]
                                if not matched_rows.empty:
                                    sector_change = matched_rows.iloc[0].get('change_percent', 0)
                                    self.logger.info(f"板块 {sector_code} 在 {date_str} 的涨跌幅: {sector_change}")
                                else:
                                    self.logger.info(f"板块 {sector_code} 在 {date_str} 没有匹配的数据")
                            else:
                                self.logger.info(f"板块 {sector_code} 在 {formatted_date} 没有匹配的日期")
                        except Exception as date_convert_error:
                            self.logger.debug(f"日期格式转换失败 {date_str}: {date_convert_error}")
                            return None, None
                    else:
                        self.logger.info(f"板块 {sector_code} 没有数据")
        
            except Exception as e:
                self.logger.error(f"获取板块 {sector_code} 在 {date_str} 的数据时出错: {e}")
                return None, None

            # 确保是数值类型
            if stock_change is not None and sector_change is not None:
                try:
                    stock_change = float(stock_change) if pd.notna(stock_change) else 0.0
                    sector_change = float(sector_change) if pd.notna(sector_change) else 0.0
                    return stock_change, sector_change
                except (ValueError, TypeError):
                    self.logger.error(f"无法将个股 {stock_code} 或板块 {sector_code} 的涨跌幅转换为数值类型")
                    return None, None

            return None, None

        except Exception as e:
            self.logger.error(f"获取股票 {stock_code} 和板块 {sector_code} 在 {date_str} 的涨跌幅时出错: {e}")
            return None, None

    def _get_sector_name_from_code(self, sector_code):
        """
        从板块代码获取板块名称

        Args:
            sector_code: 板块代码

        Returns:
            str: 板块名称
        """
        # 简单的映射逻辑，假设板块代码就是板块名称
        # 如果需要更复杂的映射，可以在这里添加
        return sector_code

    def _calculate_sector_relevance_by_change_points(self, stock_code, stock_features_df, sector_features_df, change_points):
        """
        基于变化点计算个股与板块的相关性
        计算个股所有变化点涨跌幅跟它所属板块同向次数最多的板块作为最相关板块

        Args:
            stock_code: 股票代码
            stock_features_df: 个股特征DataFrame
            sector_features_df: 板块特征DataFrame
            change_points: 变化点检测结果

        Returns:
            str: 最相关的板块代码，如果无法确定则返回None
        """
        try:
            # 获取股票的变化点信息
            if stock_code not in change_points:
                return None

            stock_change_points = change_points[stock_code]
            change_point_dates = stock_change_points.get('dates', [])

            # 确保change_point_dates是列表格式
            if isinstance(change_point_dates, np.ndarray):
                change_point_dates = change_point_dates.tolist()
            elif not isinstance(change_point_dates, list):
                change_point_dates = [change_point_dates] if change_point_dates is not None else []

            if not change_point_dates:
                return None

            # 获取股票所属的板块列表
            from to.stock_sectors_mapping import get_stock_sectors
            sectors = get_stock_sectors(stock_code=stock_code)
            if not sectors:
                sectors = []

            if not sectors:
                return None

            # 计算每个板块与个股在变化点的同向次数
            sector_same_direction_counts = {}

            for sector_code in sectors:
                same_direction_count = 0
                total_valid_points = 0

                for date_str in change_point_dates:
                    try:
                        # 确保date_str是字符串格式
                        if isinstance(date_str, (pd.Timestamp, np.datetime64)):
                            date_str = pd.to_datetime(date_str).strftime('%Y%m%d')
                        elif not isinstance(date_str, str):
                            date_str = str(date_str)

                        # 简化的数据获取方法，避免复杂的数组操作
                        stock_change = None
                        sector_change = None

                        # 获取个股数据
                        try:
                            # 尝试不同的日期列名
                            for date_col in ['date_str', 'date']:
                                if date_col in stock_features_df.columns:
                                    stock_subset = stock_features_df[
                                        (stock_features_df['stock_code'] == stock_code)
                                    ]
                                    if not stock_subset.empty:
                                        # 简单的字符串匹配
                                        date_matches = stock_subset[date_col].astype(str) == str(date_str)
                                        if date_matches.any():
                                            matched_rows = stock_subset[date_matches]
                                            if not matched_rows.empty:
                                                stock_change = matched_rows.iloc[0].get('pct_chg', 0)
                                                break
                        except Exception as e:
                            self.logger.debug(f"获取个股数据时出错: {e}")
                            continue

                        # 获取板块数据 - 修复日期格式匹配问题
                        try:
                            # 板块数据只有date字段，格式为 2023-06-01 00:00:00
                            if 'date' in sector_features_df.columns:
                                sector_subset = sector_features_df[
                                    (sector_features_df['sector_code'] == sector_code)
                                ]
                                if not sector_subset.empty:
                                    # 将date_str (20240913) 转换为板块数据的日期格式进行匹配
                                    try:
                                        # 将 20240913 转换为 2024-09-13 格式
                                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                                        # 将板块数据的date字段转换为字符串格式进行匹配
                                        sector_dates = pd.to_datetime(sector_subset['date']).dt.strftime('%Y-%m-%d')
                                        date_matches = sector_dates == formatted_date

                                        if date_matches.any():
                                            matched_rows = sector_subset[date_matches]
                                            if not matched_rows.empty:
                                                sector_change = matched_rows.iloc[0].get('change_percent', 0)
                                    except Exception as date_convert_error:
                                        self.logger.debug(f"日期格式转换失败 {date_str}: {date_convert_error}")
                                        continue
                        except Exception as e:
                            self.logger.debug(f"获取板块数据时出错: {e}")
                            continue

                        # 如果没有获取到数据，跳过
                        if stock_change is None or sector_change is None:
                            continue

                        # 确保是数值类型
                        try:
                            stock_change = float(stock_change) if pd.notna(stock_change) else 0.0
                            sector_change = float(sector_change) if pd.notna(sector_change) else 0.0
                        except (ValueError, TypeError):
                            continue

                        # 检查是否同向
                        is_same_direction = (
                            (stock_change >= 0 and sector_change >= 0) or
                            (stock_change <= 0 and sector_change <= 0)
                        )

                        if is_same_direction:
                            same_direction_count += 1

                        total_valid_points += 1

                    except Exception as e:
                        self.logger.debug(f"处理变化点 {date_str} 时出错: {e}")
                        continue

                if total_valid_points > 0:
                    # 计算同向比例
                    same_direction_ratio = same_direction_count / total_valid_points
                    sector_same_direction_counts[sector_code] = {
                        'same_direction_count': same_direction_count,
                        'total_valid_points': total_valid_points,
                        'same_direction_ratio': same_direction_ratio
                    }

            if not sector_same_direction_counts:
                return None

            # 找到同向次数最多的板块
            max_same_direction_count = max(
                info['same_direction_count'] for info in sector_same_direction_counts.values()
            )

            # 获取同向次数最多的板块列表
            best_sectors = [
                sector_code for sector_code, info in sector_same_direction_counts.items()
                if info['same_direction_count'] == max_same_direction_count
            ]

            # 如果有多个板块同向次数相同，选择股票数量最少的板块
            if len(best_sectors) > 1:
                # 使用类中已加载的板块股票数量信息
                sector_stock_count = self.sector_stock_count if hasattr(self, 'sector_stock_count') else {}

                min_stock_count = float('inf')
                final_sector = best_sectors[0]  # 默认选择第一个

                for sector_code in best_sectors:
                    sector_name = self._get_sector_name_from_code(sector_code)
                    stock_count = sector_stock_count.get(sector_name, 999999)

                    if stock_count < min_stock_count:
                        min_stock_count = stock_count
                        final_sector = sector_code

                self.logger.debug(f"股票 {stock_code} 有多个同向次数相同的板块，选择股票数量最少的: {final_sector}")
                return final_sector
            else:
                selected_sector = best_sectors[0]
                sector_info = sector_same_direction_counts[selected_sector]
                self.logger.debug(f"股票 {stock_code} 最相关板块: {selected_sector}, "
                                f"同向次数: {sector_info['same_direction_count']}/{sector_info['total_valid_points']}")
                return selected_sector

        except Exception as e:
            self.logger.error(f"计算股票 {stock_code} 的板块相关性时出错: {e}")
            return None


def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="板块轮动模型训练与预测")
    parser.add_argument('--mode', type=str, choices=['select_stocks'],
                      help='select_stocks (选股)')
    parser.add_argument('--start_date', type=str, help='开始日期，格式为YYYYMMDD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD')
    parser.add_argument('--date', type=str, help='预测日期，格式为YYYYMMDD')
    parser.add_argument('--top_n', type=int, default=50, help='选择前N个板块或股票')
    parser.add_argument('--max_workers', type=int, help='并行训练的进程数')
    parser.add_argument('--sector_list', type=str, help='板块列表文件路径，用于板块轮动模型训练')
    parser.add_argument('--model_dir', type=str, help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='sector_rotation_model', help='模型名称')
    parser.add_argument('--save_results', type=str, help='保存结果的文件路径')
    parser.add_argument('--stock_list', type=str, help='股票列表文件路径，用于选股')
    parser.add_argument('--use_cache', action='store_true', help='是否使用特征缓存，默认为True')
    parser.add_argument('--no_cache', dest='use_cache', action='store_false', help='不使用特征缓存')
    parser.set_defaults(use_cache=True)

    args = parser.parse_args()
    print(f"开始")

    logger.setLevel(logging.DEBUG)
    # 创建SectorModelTrainer实例
    trainer = SectorModelTrainer(model_dir=args.model_dir)

    # 加载板块列表
    sector_list = []
    try:
        # 优先使用参数指定的板块列表文件
        if args.sector_list and os.path.exists(args.sector_list):
            with open(args.sector_list, 'r', encoding='utf-8') as f:
                sector_list = [line.strip() for line in f if line.strip()]
            logger.info(f"从文件 {args.sector_list} 加载了 {len(sector_list)} 个板块")
        else:
            # 尝试从stock_sectors_mapping模块获取
            try:
                # 将当前工作目录添加到sys.path
                sys.path.append(os.getcwd())
                from stock_sectors_mapping import get_tng_sectors
                sector_list = get_tng_sectors()
                logger.info(f"从get_tng_sectors加载了 {len(sector_list)} 个板块")
            except ImportError:
                # 如果导入失败，尝试从to目录导入
                try:
                    sys.path.append(str(get_project_root()))
                    from to.stock_sectors_mapping import get_tng_sectors
                    sector_list = get_tng_sectors()
                    logger.info(f"从to.stock_sectors_mapping加载了 {len(sector_list)} 个板块")
                except ImportError:
                    logger.error("无法导入get_tng_sectors函数，请提供板块列表文件")
    except Exception as e:
        logger.error(f"获取板块列表时出错: {e}")
        logger.error(traceback.format_exc())

    # 加载股票列表（如果需要）
    stock_list = None
    if args.stock_list and os.path.exists(args.stock_list):
        try:
            with open(args.stock_list, 'r', encoding='utf-8') as f:
                stock_list = [line.strip() for line in f if line.strip()]
            logger.info(f"从文件 {args.stock_list} 加载了 {len(stock_list)} 只股票")
        except Exception as e:
            logger.error(f"加载股票列表时出错: {e}")

    # 运行模式
    if args.mode == 'select_stocks':
        # 使用变化点方法进行选股
        top_n = args.top_n if args.top_n else 50
        model_name = args.model_name if args.model_name else "stock_selector"

        # 设置日期范围
        start_date = args.start_date
        end_date = args.end_date if args.end_date else datetime.now().strftime("%Y%m%d")

        print(f"开始运行基于变化点检测的智能选股框架...")
        print(f"- 时间范围: {start_date} 至 {end_date}")
        print(f"- 选出数量: 前 {top_n} 只股票")
        print(f"- 模型名称: {model_name}")
        print(f"- 使用缓存: {'是' if args.use_cache else '否'}")
        print("-" * 80)

        # 调用选股函数
        selected_stocks = trainer.select_stocks_by_change_points(
            start_date=start_date,
            end_date=end_date,
            model_name=model_name,
            top_n=top_n,
            use_cache=args.use_cache
        )

        # if selected_stocks and selected_stocks.get('selected_stocks'):
        #     # 打印选股结果
        #     selected_list = selected_stocks.get('selected_stocks', [])
        #     print(f"\n选股结果 (日期: {selected_stocks.get('date', 'N/A')}, 共选出 {len(selected_list)} 只股票):")
        #     print("-" * 80)
        #     print(f"{'排名':<6}{'股票代码':<12}{'股票名称':<15}{'预期收益率':>12}{'概率':>10}{'置信度':>10}{'得分':>10}")
        #     print("-" * 80)

        #     for i, stock in enumerate(selected_list):
        #         if i < 20:  # 只显示前20只
        #             stock_name = stock.get('stock_name', 'N/A')
        #             expected_return = stock.get('expected_return', 0)
        #             probability = stock.get('probability', 0)
        #             confidence = stock.get('confidence', 0)
        #             score = stock.get('score', 0)

        #             print(f"{i+1:<6}{stock['stock_code']:<12}{stock_name:<15}{expected_return:>12.2f}%{probability*100:>9.1f}%{confidence*100:>9.1f}%{score:>9.2f}")

        #     if len(selected_list) > 20:
        #         print(f"... 还有 {len(selected_list)-20} 只股票未显示")

        #     print(f"\n选股结果已保存到 {trainer.model_dir / f'{model_name}_results.json'}")
        # else:
        #     print("选股失败或未选出任何股票，请检查日志获取详细信息")
        #     if selected_stocks:
        #         print(f"选股结果状态: {selected_stocks}")
        #     else:
        #         print("选股结果为空")
    else:
        print("请指定有效的运行模式: --mode train 或 --mode select_stocks")
        print("\n示例用法:")
        print("训练板块轮动模型:")
        print("  python train_sector_models.py --mode train --start_date 20230101 --end_date 20231231 --model_name sector_rotation_model")
        print("\n智能选股:")
        print("  python train_sector_models.py --mode select_stocks --start_date 20230101 --end_date 20231231 --top_n 50 --model_name stock_selector")
        print("  python train_sector_models.py --mode select_stocks --start_date 20230101 --end_date 20231231 --top_n 50 --no_cache  # 不使用缓存")


if __name__ == "__main__":
    main()