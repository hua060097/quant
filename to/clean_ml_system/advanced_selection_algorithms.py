"""
先进选股算法模块
实现多任务学习+注意力机制和遗传算法+强化学习两套并行选股方案
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

logger = logging.getLogger(__name__)

@dataclass
class SelectionResult:
    """选股结果数据类"""
    algorithm_name: str
    selected_stocks: List[str]
    confidence_scores: Dict[str, float]
    feature_importance: Dict[str, float]
    performance_metrics: Dict[str, float]
    sub_state: str
    sector: str

class BaseSelectionAlgorithm(ABC):
    """选股算法基类"""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.logger = logger
        self.is_trained = False
        
    @abstractmethod
    def train(self, data: pd.DataFrame, target_cols: List[str]) -> None:
        """训练算法"""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> SelectionResult:
        """预测选股"""
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, actual_returns: pd.Series) -> Dict[str, float]:
        """评估算法性能"""
        pass

class MultiTaskAttentionAlgorithm(BaseSelectionAlgorithm):
    """方案A: 多任务学习 + 注意力机制选股算法 - 子状态约束版本"""

    def __init__(self):
        super().__init__("MultiTask_Attention")
        self.sub_state_models = {}  # 🔥 每个子状态独立的模型
        self.sub_state_attention_weights = {}  # 🔥 每个子状态的注意力权重
        self.feature_columns = None
        # 🔥 恢复：多任务学习架构，同时预测收益、波动率、胜率
        self.target_columns = ['future_return_1d', 'volatility', 'win_rate']

    def train(self, data: pd.DataFrame, target_cols: List[str] = None) -> None:
        """训练多任务学习模型 - 按子状态分别训练（带缓存）"""
        import pickle
        import hashlib
        import os
        from pathlib import Path

        # 🔥 新增：缓存逻辑
        cache_dir = Path("training_cache")
        cache_dir.mkdir(exist_ok=True)

        # 生成缓存键
        data_hash = hashlib.md5(str(data.shape).encode() + str(data.columns.tolist()).encode()).hexdigest()[:8]
        cache_key = f"multitask_attention_train_{data_hash}"
        cache_file = cache_dir / f"{cache_key}.pkl"

        # 检查缓存
        if cache_file.exists():
            try:
                self.logger.info(f"发现{self.algorithm_name}训练缓存，正在加载...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.sub_state_models = cached_data['sub_state_models']
                    self.feature_columns = cached_data['feature_columns']
                    self.is_trained = cached_data['is_trained']
                    # 🔥 修复：加载注意力权重
                    self.sub_state_attention_weights = cached_data.get('sub_state_attention_weights', {})
                self.logger.info(f"✅ {self.algorithm_name}训练缓存加载成功，跳过训练")
                return
            except Exception as e:
                self.logger.warning(f"缓存加载失败: {e}，重新训练")

        self.logger.info(f"开始训练{self.algorithm_name}算法（子状态约束版本）")

        if target_cols is None:
            target_cols = self.target_columns

        # 🔥 核心修改：按子状态分组训练
        if 'sub_state' not in data.columns:
            raise ValueError("训练数据必须包含sub_state列")

        # 🔥 优化：增强特征工程
        enhanced_data = self._enhance_features(data)

        # 准备特征列
        feature_cols = self._get_feature_columns(enhanced_data)
        if not feature_cols:
            raise ValueError("没有可用的特征列")
        self.feature_columns = feature_cols

        # 🔥 按子状态分别训练模型
        sub_states = data['sub_state'].unique()
        self.logger.info(f"发现 {len(sub_states)} 个子状态: {list(sub_states)}")

        for sub_state in sub_states:
            self.logger.info(f"训练子状态 {sub_state} 的多任务模型")

            # 获取该子状态的数据
            sub_state_data = enhanced_data[enhanced_data['sub_state'] == sub_state].copy()

            if len(sub_state_data) < 50:  # 最小样本要求
                self.logger.warning(f"子状态 {sub_state} 样本量不足({len(sub_state_data)})，跳过训练")
                continue

            # 准备特征和目标
            X = sub_state_data[feature_cols].fillna(0)

            # 🔥 恢复：多任务学习，同时预测收益、波动率、胜率
            if 'future_return_1d' not in sub_state_data.columns:
                self.logger.error(f"数据中缺少future_return_1d字段！可用字段: {list(sub_state_data.columns)}")
                raise KeyError(f"future_return_1d字段不存在于训练数据中")

            # 构建多任务目标变量
            y_return = sub_state_data['future_return_1d'].fillna(0)
            y_volatility = self._calculate_volatility(sub_state_data)
            y_win_rate = self._calculate_win_rate(sub_state_data)

            # 组合多任务目标
            y_multi = pd.DataFrame({
                'future_return_1d': y_return,
                'volatility': y_volatility,
                'win_rate': y_win_rate
            })

            # 🔥 优化：高性能RandomForest参数 - 针对高胜率、高收益、低回撤
            from sklearn.multioutput import MultiOutputRegressor
            base_model = RandomForestRegressor(
                n_estimators=200,        # 🔥 增加树数量提升稳定性
                max_depth=15,           # 🔥 适度增加深度捕捉复杂模式
                min_samples_split=10,   # 🔥 降低分割要求提升敏感性
                min_samples_leaf=5,     # 🔥 降低叶子节点要求
                max_features=0.6,       # 🔥 使用60%特征平衡偏差和方差
                bootstrap=True,         # 🔥 启用bootstrap提升泛化
                oob_score=True,         # 🔥 启用OOB评分
                random_state=42,
                n_jobs=-1
            )
            model = MultiOutputRegressor(base_model)

            try:
                model.fit(X, y_multi)
                self.sub_state_models[sub_state] = model

                # 计算该子状态的注意力权重
                attention_weights = self._calculate_attention_weights(X, y_return, model)
                self.sub_state_attention_weights[sub_state] = attention_weights

                self.logger.info(f"子状态 {sub_state} 模型训练完成，样本量: {len(sub_state_data)}")

            except Exception as e:
                self.logger.error(f"子状态 {sub_state} 模型训练失败: {e}")
                continue

        if self.sub_state_models:
            self.is_trained = True
            self.logger.info(f"{self.algorithm_name}算法训练完成，共训练 {len(self.sub_state_models)} 个子状态模型")

            # 🔥 新增：保存训练缓存
            try:
                cache_data = {
                    'sub_state_models': self.sub_state_models,
                    'feature_columns': self.feature_columns,
                    'is_trained': self.is_trained,
                    'sub_state_attention_weights': getattr(self, 'sub_state_attention_weights', {})
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                self.logger.info(f"✅ {self.algorithm_name}训练缓存已保存")
            except Exception as e:
                self.logger.warning(f"保存训练缓存失败: {e}")
        else:
            self.logger.error(f"{self.algorithm_name}算法训练失败，没有成功训练的子状态模型")
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """获取特征列"""
        exclude_cols = {
            'stock_code', 'date', 'date_str', 'trade_date', 'future_return_1d', 'future_return_2d',
            'is_can_buy',  # 🔥 关键修复：排除 is_can_buy 字段
            'target_sector', 'sub_state', 'sector_code', 'sector_name',
            'first_time_formatted', 'last_time_formatted', 'first_time', 'last_time'
        }
        
        feature_cols = []
        for col in data.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col]):
                feature_cols.append(col)
        
        return feature_cols
    
    def _calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """计算波动率指标"""
        if 'pct_chg' in data.columns:
            # 使用价格变化的绝对值作为波动率代理
            return data['pct_chg'].abs()
        else:
            # 默认返回零
            return pd.Series(0, index=data.index)
    
    def _calculate_win_rate(self, data: pd.DataFrame) -> pd.Series:
        """计算胜率指标"""
        if 'future_return_1d' in data.columns:
            # 胜率 = 正收益的概率
            return (data['future_return_1d'] > 0).astype(float)
        else:
            return pd.Series(0.5, index=data.index)
    
    def _calculate_attention_weights(self, X: pd.DataFrame, y: pd.Series, model) -> Dict[str, float]:
        """🔥 优化：真正的动态注意力机制 - 基于特征相关性和市场状态"""
        attention_weights = {}

        # 🔥 第一层：基础特征重要性（来自模型）
        if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
            base_importance = model.estimators_[0].feature_importances_
        else:
            base_importance = model.feature_importances_

        # 🔥 第二层：特征与目标的相关性权重
        correlation_weights = {}
        for i, feature in enumerate(X.columns):
            if feature in X.columns:
                corr = abs(X[feature].corr(y))
                correlation_weights[feature] = corr if not pd.isna(corr) else 0

        # 🔥 第三层：A股特色特征增强权重
        a_share_weights = self._calculate_a_share_feature_weights(X)

        # 🔥 第四层：市场状态适应性权重
        market_adaptive_weights = self._calculate_market_adaptive_weights(X)

        # 🔥 动态融合四层权重
        for i, feature in enumerate(X.columns):
            base_w = base_importance[i] if i < len(base_importance) else 0
            corr_w = correlation_weights.get(feature, 0)
            a_share_w = a_share_weights.get(feature, 0)
            market_w = market_adaptive_weights.get(feature, 0)

            # 加权融合：基础40% + 相关性30% + A股特色20% + 市场适应10%
            attention_weights[feature] = (0.4 * base_w + 0.3 * corr_w +
                                        0.2 * a_share_w + 0.1 * market_w)

        # 归一化权重
        total_weight = sum(attention_weights.values())
        if total_weight > 0:
            attention_weights = {k: v/total_weight for k, v in attention_weights.items()}
        else:
            uniform_weight = 1.0 / len(X.columns)
            attention_weights = {feature: uniform_weight for feature in X.columns}

        return attention_weights

    def _calculate_a_share_feature_weights(self, X: pd.DataFrame) -> Dict[str, float]:
        """🔥 新增：A股特色特征权重计算"""
        weights = {}

        # 🔥 A股特色特征组及其权重
        a_share_features = {
            # 涨跌停特征组 - A股独有
            'limit_features': ['is_limit_up', 'is_limit_down', 'limit_times', 'limit_strength',
                             'time_to_limit_minutes', 'limit_quality_score', 'limit_up_days_5d',
                             'limit_down_days_5d', 'limit_up_days_20d', 'consecutive_limit_up',
                             'consecutive_limit_down'],
            # 板块轮动特征
            'board_features': ['board_quality', 'continuity_factor', 'is_zhaban'],
            # 资金流向特征
            'capital_features': ['fd_amount', 'limit_amount', 'turnover_efficiency',
                               'limit_up_volume_ratio', 'limit_up_turnover_ratio'],
            # 技术形态特征
            'pattern_features': ['pattern_score', 'candlestick_strength', 'pattern_engulfing_bullish',
                               'pattern_engulfing_bearish']
        }

        # 为每个特征分配权重
        for feature in X.columns:
            base_weight = 0.1  # 默认权重

            # A股特色特征增强
            if feature in a_share_features['limit_features']:
                base_weight = 0.8  # 涨跌停特征最重要
            elif feature in a_share_features['board_features']:
                base_weight = 0.6  # 板块特征次重要
            elif feature in a_share_features['capital_features']:
                base_weight = 0.5  # 资金特征
            elif feature in a_share_features['pattern_features']:
                base_weight = 0.4  # 形态特征

            weights[feature] = base_weight

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        return weights

    def _calculate_market_adaptive_weights(self, X: pd.DataFrame) -> Dict[str, float]:
        """🔥 新增：市场状态适应性权重计算"""
        weights = {}

        # 计算市场整体状态指标
        market_volatility = X.get('volatility20', pd.Series(0.02)).mean()
        market_trend = X.get('trend_strength', pd.Series(0)).mean()
        limit_up_activity = X.get('is_limit_up', pd.Series(0)).mean()

        for feature in X.columns:
            base_weight = 0.1

            # 🔥 根据市场状态调整特征权重
            if market_volatility > 0.05:  # 高波动市场
                if 'volatility' in feature or 'risk' in feature:
                    base_weight = 0.8
                elif 'limit' in feature:
                    base_weight = 0.6
            elif market_trend > 0.6:  # 强趋势市场
                if 'trend' in feature or 'momentum' in feature:
                    base_weight = 0.8
                elif 'ma' in feature:
                    base_weight = 0.6
            elif limit_up_activity > 0.1:  # 活跃涨停市场
                if 'limit' in feature:
                    base_weight = 0.9
                elif 'volume' in feature or 'turnover' in feature:
                    base_weight = 0.7

            weights[feature] = base_weight

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        return weights

    def _calculate_dynamic_fusion_weights(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        """🔥 新增：基于股票特征动态计算多任务融合权重"""
        # 默认权重
        default_weights = {'return': 0.5, 'win_rate': 0.3, 'risk_adjusted': 0.2}

        if len(stock_data) == 0:
            return default_weights

        # 🔥 基于涨跌停特征调整权重
        limit_up_ratio = stock_data.get('is_limit_up', pd.Series(0)).mean()
        limit_strength = stock_data.get('limit_strength', pd.Series(0)).mean()
        volatility_level = stock_data.get('volatility20', pd.Series(0.02)).mean()

        # 🔥 基于技术指标调整权重
        rsi_level = stock_data.get('rsi14', pd.Series(50)).mean()
        trend_strength = stock_data.get('trend_strength', pd.Series(0)).mean()

        # 🔥 动态权重调整逻辑
        weights = default_weights.copy()

        # 高涨停比例时，增加胜率权重，降低收益权重（防止追高）
        if limit_up_ratio > 0.1:  # 10%以上涨停比例
            weights['win_rate'] += 0.15
            weights['return'] -= 0.1
            weights['risk_adjusted'] -= 0.05

        # 高波动时，增加风险调整权重
        if volatility_level > 0.05:  # 5%以上波动率
            weights['risk_adjusted'] += 0.1
            weights['return'] -= 0.05
            weights['win_rate'] -= 0.05

        # 强趋势时，增加收益权重
        if trend_strength > 0.6:
            weights['return'] += 0.1
            weights['risk_adjusted'] -= 0.05
            weights['win_rate'] -= 0.05

        # 超买超卖调整
        if rsi_level > 70:  # 超买
            weights['risk_adjusted'] += 0.1
            weights['return'] -= 0.1
        elif rsi_level < 30:  # 超卖
            weights['return'] += 0.1
            weights['risk_adjusted'] -= 0.1

        # 确保权重和为1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = default_weights

        return weights

    def _enhance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """🔥 新增：增强特征工程 - 基于105个特征构建更有效的组合特征"""
        enhanced_data = data.copy()

        try:
            # 🔥 A股特色组合特征
            # 1. 涨跌停强度综合指标
            if all(col in data.columns for col in ['is_limit_up', 'limit_times', 'time_to_limit_minutes']):
                enhanced_data['limit_strength_composite'] = (
                    enhanced_data['is_limit_up'] * 2 +
                    enhanced_data['limit_times'] * 0.5 +
                    (1 / (enhanced_data['time_to_limit_minutes'] + 1)) * 0.3
                ).fillna(0)

            # 2. 连板质量评分
            if all(col in data.columns for col in ['consecutive_limit_up', 'limit_quality_score', 'fd_amount']):
                enhanced_data['board_quality_enhanced'] = (
                    enhanced_data['consecutive_limit_up'] * 0.4 +
                    enhanced_data['limit_quality_score'] * 0.4 +
                    np.log1p(enhanced_data['fd_amount'].fillna(0)) * 0.2
                ).fillna(0)

            # 3. 技术指标综合强度
            if all(col in data.columns for col in ['rsi14', 'macd_hist', 'trend_strength']):
                enhanced_data['technical_momentum_enhanced'] = (
                    (enhanced_data['rsi14'] - 50) / 50 * 0.3 +
                    enhanced_data['macd_hist'] * 0.4 +
                    enhanced_data['trend_strength'] * 0.3
                ).fillna(0)

            # 4. 价量配合度
            if all(col in data.columns for col in ['pct_chg', 'volume_zscore', 'turnover_rate']):
                enhanced_data['price_volume_synergy'] = (
                    enhanced_data['pct_chg'] * enhanced_data['volume_zscore'] * 0.5 +
                    enhanced_data['pct_chg'] * enhanced_data['turnover_rate'] * 0.5
                ).fillna(0)

            # 5. 多周期趋势一致性增强
            if all(col in data.columns for col in ['ma5', 'ma20', 'ma60', 'close']):
                ma5_trend = (enhanced_data['close'] > enhanced_data['ma5']).astype(int)
                ma20_trend = (enhanced_data['close'] > enhanced_data['ma20']).astype(int)
                ma60_trend = (enhanced_data['close'] > enhanced_data['ma60']).astype(int)
                enhanced_data['multi_ma_alignment'] = (ma5_trend + ma20_trend + ma60_trend) / 3

            # 6. 风险收益比增强
            if all(col in data.columns for col in ['return_20d', 'volatility20', 'risk_adjusted_return']):
                enhanced_data['enhanced_risk_reward'] = (
                    enhanced_data['return_20d'] / (enhanced_data['volatility20'] + 0.01) * 0.6 +
                    enhanced_data['risk_adjusted_return'] * 0.4
                ).fillna(0)

            # 7. 市场地位指标
            if all(col in data.columns for col in ['total_mv', 'turnover_rate', 'relative_strength_20d']):
                enhanced_data['market_position_score'] = (
                    np.log1p(enhanced_data['total_mv'].fillna(0)) * 0.3 +
                    enhanced_data['turnover_rate'] * 0.3 +
                    enhanced_data['relative_strength_20d'] * 0.4
                ).fillna(0)

            # 8. 突破信号综合
            if all(col in data.columns for col in ['bb_breakout_signal', 'pattern_score', 'momentum_score']):
                enhanced_data['breakout_signal_composite'] = (
                    enhanced_data['bb_breakout_signal'] * 0.4 +
                    enhanced_data['pattern_score'] * 0.3 +
                    enhanced_data['momentum_score'] * 0.3
                ).fillna(0)

            self.logger.info(f"特征增强完成，新增 {len(enhanced_data.columns) - len(data.columns)} 个组合特征")

        except Exception as e:
            self.logger.warning(f"特征增强过程中出现错误: {e}，使用原始特征")
            enhanced_data = data.copy()

        return enhanced_data

    def predict(self, data: pd.DataFrame, target_sub_states: List[str] = None) -> SelectionResult:
        """预测选股 - 子状态约束版本"""
        if not self.is_trained:
            raise ValueError("模型未训练")

        if not self.sub_state_models:
            raise ValueError("没有训练好的子状态模型")

        # 🔥 优化：对预测数据也应用特征增强
        enhanced_data = self._enhance_features(data)

        # 🔥 核心修改：按子状态分别预测
        all_selected_stocks = []
        all_confidence_scores = {}
        combined_feature_importance = {}

        # 🔥 修复：确保 target_sub_states 是列表类型
        if target_sub_states is None:
            target_sub_states = list(self.sub_state_models.keys())
        elif isinstance(target_sub_states, str):
            # 如果传入的是字符串，转换为单元素列表
            target_sub_states = [target_sub_states]
            self.logger.warning(f"检测到传入字符串参数，已转换为列表: {target_sub_states}")
        elif not isinstance(target_sub_states, list):
            # 如果是其他可迭代类型，转换为列表
            target_sub_states = list(target_sub_states)

        self.logger.info(f"在子状态 {target_sub_states} 中进行预测")

        for sub_state in target_sub_states:
            if sub_state not in self.sub_state_models:
                self.logger.warning(f"子状态 {sub_state} 没有训练好的模型，跳过")
                continue

            # 🔥 关键：只在该子状态对应的股票中预测
            # 这里需要外部传入子状态对应的股票范围，或者从数据中筛选
            sub_state_data = enhanced_data.copy()  # 使用增强后的数据

            if len(sub_state_data) == 0:
                continue

            # 🔥 修复：使用训练时保存的特征列，确保一致性
            if self.feature_columns is None:
                self.logger.error(f"算法 {self.algorithm_name} 没有保存的特征列信息")
                continue

            # 检查数据中是否包含所需的特征列
            available_features = [col for col in self.feature_columns if col in sub_state_data.columns]
            missing_features = [col for col in self.feature_columns if col not in sub_state_data.columns]

            if len(available_features) < len(self.feature_columns) * 0.8:  # 至少80%的特征可用
                self.logger.warning(f"子状态 {sub_state} 数据中缺少过多训练时的特征列")
                self.logger.warning(f"需要 {len(self.feature_columns)} 个特征，只找到 {len(available_features)} 个")
                self.logger.debug(f"缺失的特征列: {missing_features[:5]}...")  # 只显示前5个
                continue

            X = sub_state_data[available_features].fillna(0)
            model = self.sub_state_models[sub_state]
            attention_weights = self.sub_state_attention_weights[sub_state]

            # 🔥 修复：多任务预测
            predictions = model.predict(X)

            # 多任务输出：[收益预测, 波动率预测, 胜率预测]
            if predictions.ndim == 2 and predictions.shape[1] >= 3:
                return_pred = predictions[:, 0]    # 收益预测
                volatility_pred = predictions[:, 1]  # 波动率预测
                win_rate_pred = predictions[:, 2]   # 胜率预测

                # 🔥 优化：动态多任务融合 - 基于市场状态和股票特征
                risk_adjusted_return = return_pred / (volatility_pred + 0.01)  # 避免除零

                # 🔥 动态权重计算：基于股票特征调整融合权重
                dynamic_weights = self._calculate_dynamic_fusion_weights(sub_state_data.iloc[range(len(return_pred))])

                # 应用动态权重融合
                composite_scores = (dynamic_weights['return'] * return_pred +
                                  dynamic_weights['win_rate'] * win_rate_pred +
                                  dynamic_weights['risk_adjusted'] * risk_adjusted_return)
            else:
                # 回退到单任务模式
                composite_scores = predictions.flatten() if predictions.ndim > 1 else predictions

            # 应用注意力权重到特征
            weighted_features = X.copy()
            for feature, weight in attention_weights.items():
                if feature in weighted_features.columns:
                    weighted_features[feature] *= weight

            # 选择该子状态的top股票（每个子状态选择3-5只）
            n_select = min(5, len(composite_scores))
            top_indices = np.argsort(composite_scores)[-n_select:]

            sub_state_stocks = sub_state_data.iloc[top_indices]['stock_code'].tolist()
            all_selected_stocks.extend(sub_state_stocks)

            # 记录置信度
            for i, stock_code in enumerate(sub_state_stocks):
                score = float(composite_scores[top_indices[i]])
                all_confidence_scores[stock_code] = score

            # 合并特征重要性
            for feature, weight in attention_weights.items():
                if feature not in combined_feature_importance:
                    combined_feature_importance[feature] = 0
                combined_feature_importance[feature] += weight

        # 去重并限制总数
        unique_stocks = list(dict.fromkeys(all_selected_stocks))[:10]
        final_confidence_scores = {stock: all_confidence_scores[stock] for stock in unique_stocks}

        return SelectionResult(
            algorithm_name=self.algorithm_name,
            selected_stocks=unique_stocks,
            confidence_scores=final_confidence_scores,
            feature_importance=combined_feature_importance,
            performance_metrics={},
            sub_state=",".join(target_sub_states),
            sector="multi_sub_state"
        )
    
    def evaluate(self, data: pd.DataFrame, actual_returns: pd.Series) -> Dict[str, float]:
        """评估算法性能"""
        if not self.is_trained:
            return {}
        
        X = data[self.feature_columns].fillna(0)
        predictions = self.multi_task_model.predict(X)
        
        # 评估收益预测
        predicted_returns = predictions[:, 0]
        mse = mean_squared_error(actual_returns, predicted_returns)
        r2 = r2_score(actual_returns, predicted_returns)
        
        return {
            'mse': mse,
            'r2_score': r2,
            'mean_predicted_return': np.mean(predicted_returns),
            'std_predicted_return': np.std(predicted_returns)
        }

class GeneticRLAlgorithm(BaseSelectionAlgorithm):
    """方案B: 遗传算法 + 强化学习优化因子组合"""
    
    def __init__(self):
        super().__init__("Genetic_RL")
        self.sub_state_factor_combinations = {}  # 🔥 每个子状态的因子组合
        self.sub_state_rl_weights = {}          # 🔥 每个子状态的强化学习权重
        self.sub_state_models = {}              # 🔥 修复：添加模型字典，与MultiTask_Attention保持一致
        self.feature_columns = None
        self.population_size = 50
        self.generations = 20

    def train(self, data: pd.DataFrame, target_cols: List[str] = None) -> None:
        """训练遗传算法优化的因子组合 - 按子状态分别训练（带缓存）"""
        import pickle
        import hashlib
        from pathlib import Path

        # 🔥 新增：缓存逻辑
        cache_dir = Path("training_cache")
        cache_dir.mkdir(exist_ok=True)

        # 生成缓存键
        data_hash = hashlib.md5(str(data.shape).encode() + str(data.columns.tolist()).encode()).hexdigest()[:8]
        cache_key = f"genetic_rl_train_{data_hash}"
        cache_file = cache_dir / f"{cache_key}.pkl"

        # 检查缓存
        if cache_file.exists():
            try:
                self.logger.info(f"发现{self.algorithm_name}训练缓存，正在加载...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.sub_state_models = cached_data.get('sub_state_models', {})
                    self.sub_state_factor_combinations = cached_data.get('sub_state_factor_combinations', {})
                    self.feature_columns = cached_data['feature_columns']
                    self.is_trained = cached_data['is_trained']
                    # 🔥 修复：加载强化学习权重
                    self.sub_state_rl_weights = cached_data.get('sub_state_rl_weights', {})

                    # 🔥 修复：如果缓存中没有sub_state_models但有factor_combinations，重建sub_state_models
                    if not self.sub_state_models and self.sub_state_factor_combinations:
                        for sub_state in self.sub_state_factor_combinations:
                            self.sub_state_models[sub_state] = {
                                'factor_combinations': self.sub_state_factor_combinations[sub_state],
                                'rl_weights': self.sub_state_rl_weights.get(sub_state, {}),
                                'trained': True
                            }
                self.logger.info(f"✅ {self.algorithm_name}训练缓存加载成功，跳过训练")
                return
            except Exception as e:
                self.logger.warning(f"缓存加载失败: {e}，重新训练")

        self.logger.info(f"开始训练{self.algorithm_name}算法（子状态约束版本）")

        # 🔥 核心修改：按子状态分组训练
        if 'sub_state' not in data.columns:
            raise ValueError("训练数据必须包含sub_state列")

        # 准备特征
        feature_cols = self._get_feature_columns(data)
        if not feature_cols:
            raise ValueError("没有可用的特征列")
        self.feature_columns = feature_cols

        # 🔥 按子状态分别训练
        sub_states = data['sub_state'].unique()
        self.logger.info(f"发现 {len(sub_states)} 个子状态: {list(sub_states)}")

        for sub_state in sub_states:
            self.logger.info(f"训练子状态 {sub_state} 的遗传强化学习模型")

            # 获取该子状态的数据
            sub_state_data = data[data['sub_state'] == sub_state].copy()

            if len(sub_state_data) < 100:  # 遗传算法需要更多样本
                self.logger.warning(f"子状态 {sub_state} 样本量不足({len(sub_state_data)})，跳过训练")
                continue

            X = sub_state_data[feature_cols].fillna(0)
            y = sub_state_data['future_return_1d'].fillna(0)

            try:
                # 遗传算法优化因子组合
                factor_combinations = self._genetic_optimization(X, y)
                self.sub_state_factor_combinations[sub_state] = factor_combinations

                # 强化学习优化权重
                rl_weights = self._reinforcement_learning_weights(X, y, factor_combinations)
                self.sub_state_rl_weights[sub_state] = rl_weights

                # 🔥 修复：设置sub_state_models以保持与MultiTask_Attention一致的接口
                # 对于Genetic_RL，我们用一个标记对象表示该子状态已训练
                self.sub_state_models[sub_state] = {
                    'factor_combinations': factor_combinations,
                    'rl_weights': rl_weights,
                    'trained': True
                }

                self.logger.info(f"子状态 {sub_state} 模型训练完成，样本量: {len(sub_state_data)}")

            except Exception as e:
                self.logger.error(f"子状态 {sub_state} 模型训练失败: {e}")
                continue

        if self.sub_state_factor_combinations:
            self.is_trained = True
            self.logger.info(f"{self.algorithm_name}算法训练完成，共训练 {len(self.sub_state_factor_combinations)} 个子状态模型")

            # 🔥 新增：保存训练缓存
            try:
                cache_data = {
                    'sub_state_models': self.sub_state_models,
                    'sub_state_factor_combinations': self.sub_state_factor_combinations,
                    'feature_columns': self.feature_columns,
                    'is_trained': self.is_trained,
                    # 🔥 修复：保存强化学习权重
                    'sub_state_rl_weights': self.sub_state_rl_weights
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                self.logger.info(f"✅ {self.algorithm_name}训练缓存已保存")
            except Exception as e:
                self.logger.warning(f"保存训练缓存失败: {e}")
        else:
            self.logger.error(f"{self.algorithm_name}算法训练失败，没有成功训练的子状态模型")
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """获取特征列"""
        exclude_cols = {
            'stock_code', 'date', 'date_str', 'trade_date', 'future_return_1d', 'future_return_2d',
            'is_can_buy',  # 🔥 关键修复：排除 is_can_buy 字段
            'target_sector', 'sub_state', 'sector_code', 'sector_name',
            'first_time_formatted', 'last_time_formatted', 'first_time', 'last_time'
        }
        
        feature_cols = []
        for col in data.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col]):
                feature_cols.append(col)
        
        return feature_cols
    
    def _genetic_optimization(self, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """遗传算法优化因子组合"""
        # 简化的遗传算法实现
        population = []
        
        # 初始化种群
        for _ in range(self.population_size):
            # 随机选择特征组合
            selected_features = np.random.choice(
                X.columns, 
                size=min(20, len(X.columns)), 
                replace=False
            ).tolist()
            
            # 随机权重
            weights = np.random.random(len(selected_features))
            weights = weights / weights.sum()
            
            individual = {
                'features': selected_features,
                'weights': weights,
                'fitness': 0
            }
            population.append(individual)
        
        # 进化过程
        for generation in range(self.generations):
            # 评估适应度
            for individual in population:
                individual['fitness'] = self._evaluate_fitness(X, y, individual)
            
            # 选择、交叉、变异（简化实现）
            population = sorted(population, key=lambda x: x['fitness'], reverse=True)
            population = population[:self.population_size//2]  # 保留前50%
            
            # 生成新个体（简化）
            while len(population) < self.population_size:
                parent = np.random.choice(population)
                child = self._mutate(parent)
                population.append(child)
        
        # 返回最佳个体
        best_individuals = sorted(population, key=lambda x: x['fitness'], reverse=True)[:5]
        return best_individuals
    
    def _evaluate_fitness(self, X: pd.DataFrame, y: pd.Series, individual: Dict) -> float:
        """评估个体适应度"""
        try:
            features = individual['features']
            weights = individual['weights']
            
            # 计算加权特征得分
            weighted_score = np.zeros(len(X))
            for i, feature in enumerate(features):
                if feature in X.columns:
                    weighted_score += X[feature].values * weights[i]
            
            # 计算与目标的相关性作为适应度
            correlation = np.corrcoef(weighted_score, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        except:
            return 0
    
    def _mutate(self, parent: Dict) -> Dict:
        """变异操作"""
        child = {
            'features': parent['features'].copy(),
            'weights': parent['weights'].copy(),
            'fitness': 0
        }
        
        # 随机变异权重
        mutation_rate = 0.1
        for i in range(len(child['weights'])):
            if np.random.random() < mutation_rate:
                child['weights'][i] += np.random.normal(0, 0.1)
        
        # 归一化权重
        child['weights'] = np.abs(child['weights'])
        child['weights'] = child['weights'] / child['weights'].sum()
        
        return child
    
    def _reinforcement_learning_weights(self, X: pd.DataFrame, y: pd.Series, factor_combinations: List[Dict]) -> Dict[str, float]:
        """专业版强化学习权重优化 - 基于2020-2025顶级量化研究"""
        self.logger.info("开始强化学习权重优化")

        # 🔥 专业RL实现：基于Q-Learning的动态权重优化
        n_features = len(self.feature_columns)
        n_episodes = 100  # RL训练轮数
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 0.1  # 探索率

        # 初始化Q表 - 状态空间：市场状态，动作空间：权重调整
        n_states = 10  # 简化的市场状态数量
        n_actions = 5  # 权重调整动作：大幅增加、小幅增加、保持、小幅减少、大幅减少
        q_table = np.random.uniform(0, 0.1, (n_states, n_actions, n_features))

        # 初始权重
        weights = np.ones(n_features) / n_features

        # 🔥 专注个股特征：提取个股特征分组用于组合优化
        feature_groups = self._extract_individual_stock_features(X)

        # 🔥 生成个股特征组合而非市场特征
        factor_combinations = self._generate_stock_factor_combinations(feature_groups, X.columns)

        best_weights = weights.copy()
        best_reward = float('-inf')

        for episode in range(n_episodes):
            # 重置环境
            current_weights = weights.copy()
            episode_reward = 0

            # 时间序列分段训练（模拟在线学习）
            n_segments = 10
            segment_size = len(X) // n_segments

            for segment in range(n_segments):
                start_idx = segment * segment_size
                end_idx = min((segment + 1) * segment_size, len(X))

                if end_idx - start_idx < 10:
                    continue

                X_segment = X.iloc[start_idx:end_idx]
                y_segment = y.iloc[start_idx:end_idx]

                # 🔥 状态识别：基于个股特征而非市场特征
                market_state = self._identify_stock_state(X_segment)

                # 🔥 动作选择：epsilon-greedy策略
                if np.random.random() < epsilon:
                    actions = np.random.randint(0, n_actions, n_features)  # 探索
                else:
                    actions = np.argmax(q_table[market_state], axis=0)  # 利用

                # 🔥 执行动作：调整权重
                new_weights = self._apply_weight_actions(current_weights, actions)

                # 🔥 计算奖励：基于A股量化策略的多维度评估
                reward = self._calculate_rl_reward(X_segment, y_segment, new_weights, factor_combinations)

                # 🔥 Q-Learning更新
                next_market_state = self._identify_stock_state(X_segment)
                for i in range(n_features):
                    current_q = q_table[market_state, actions[i], i]
                    max_next_q = np.max(q_table[next_market_state, :, i])
                    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
                    q_table[market_state, actions[i], i] = new_q

                current_weights = new_weights
                episode_reward += reward

            # 记录最佳权重
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_weights = current_weights.copy()

            # 衰减探索率
            epsilon = max(0.01, epsilon * 0.995)

        # 转换为字典格式
        final_weights = {}
        for i, feature in enumerate(self.feature_columns):
            final_weights[feature] = float(best_weights[i])

        self.logger.info(f"强化学习权重优化完成，最佳奖励: {best_reward:.4f}")
        return final_weights

    def _extract_individual_stock_features(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """提取个股特征分组 - 专注于个股层面的特征组合优化"""
        feature_groups = {}

        # 🔥 价格特征组（10个）
        price_features = ['open', 'high', 'low', 'close', 'pct_chg', 'pre_close',
                         'change', 'price_limit_up', 'price_limit_down', 'adj_factor']
        feature_groups['price'] = [col for col in price_features if col in X.columns]

        # 🔥 成交量特征组（15个）
        volume_features = ['volume', 'amount', 'turnover', 'free_turn', 'amplitude',
                          'vol_ma5', 'vol_ma10', 'vol_ma20', 'volume_ratio',
                          'volume_weighted_price', 'vol_price_corr_5d', 'vol_price_corr_20d',
                          'volume_spike', 'volume_trend', 'amount_per_share']
        feature_groups['volume'] = [col for col in volume_features if col in X.columns]

        # 🔥 技术指标特征组（25个）
        technical_features = ['ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma_ratio_5_20', 'ma_ratio_10_30',
                             'rsi6', 'rsi12', 'rsi24', 'macd', 'macd_signal', 'macd_hist',
                             'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                             'kdj_k', 'kdj_d', 'kdj_j', 'cci', 'williams_r', 'stoch_rsi', 'atr']
        feature_groups['technical'] = [col for col in technical_features if col in X.columns]

        # 🔥 涨跌停特征组（20个）- 个股层面
        limit_features = ['is_limit_up', 'is_limit_down', 'is_st', 'limit_up_days',
                         'consecutive_limit_up', 'limit_up_strength', 'limit_down_strength',
                         'yesterday_limit_up_premium', 'yesterday_max_limit_up_premium',
                         'yesterday_broken_limit_premium', 'limit_up_time', 'limit_up_volume',
                         'broken_limit_up', 'one_word_limit_up', 'limit_open',
                         'near_limit_up', 'near_limit_down', 'limit_up_probability',
                         'limit_down_probability', 'price_to_limit_ratio']
        feature_groups['limit'] = [col for col in limit_features if col in X.columns]

        # 🔥 动量特征组（15个）
        momentum_features = ['momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_10d', 'momentum_20d',
                           'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
                           'volatility_5d', 'volatility_10d', 'volatility_20d', 'sharpe_5d', 'sharpe_20d']
        feature_groups['momentum'] = [col for col in momentum_features if col in X.columns]

        # 🔥 波动率特征组（10个）
        volatility_features = ['realized_volatility', 'garch_volatility', 'parkinson_volatility',
                              'garman_klass_volatility', 'rogers_satchell_volatility',
                              'volatility_ratio', 'volatility_trend', 'volatility_mean_reversion',
                              'volatility_skew', 'volatility_kurtosis']
        feature_groups['volatility'] = [col for col in volatility_features if col in X.columns]

        # 🔥 排名特征组（10个）
        ranking_features = ['rank_pct_chg', 'rank_volume', 'rank_turnover', 'rank_amplitude',
                           'rank_momentum_5d', 'rank_momentum_20d', 'rank_rsi', 'rank_macd',
                           'rank_volatility', 'rank_amount']
        feature_groups['ranking'] = [col for col in ranking_features if col in X.columns]

        # 过滤掉空的特征组
        feature_groups = {k: v for k, v in feature_groups.items() if len(v) > 0}

        total_features = sum(len(v) for v in feature_groups.values())
        self.logger.info(f"个股特征分组完成，共{len(feature_groups)}组，{total_features}个特征")
        for group, features in feature_groups.items():
            self.logger.debug(f"{group}组: {len(features)}个特征")

        return feature_groups

    def _generate_stock_factor_combinations(self, feature_groups: Dict[str, List[str]],
                                          all_columns: List[str]) -> List[Dict]:
        """生成个股特征组合用于强化学习优化"""
        factor_combinations = []

        # 🔥 策略1：单特征组合（每组选择最重要的3-5个特征）
        for group_name, features in feature_groups.items():
            if len(features) >= 3:
                # 选择该组前5个特征作为一个组合
                combination = {
                    'type': 'single_group',
                    'group': group_name,
                    'features': features[:5],
                    'weight_range': (0.1, 1.0)
                }
                factor_combinations.append(combination)

        # 🔥 策略2：跨组合特征组合（技术指标+成交量）
        if 'technical' in feature_groups and 'volume' in feature_groups:
            tech_vol_combo = {
                'type': 'cross_group',
                'groups': ['technical', 'volume'],
                'features': feature_groups['technical'][:3] + feature_groups['volume'][:2],
                'weight_range': (0.2, 0.8)
            }
            factor_combinations.append(tech_vol_combo)

        # 🔥 策略3：涨跌停+动量组合（A股特色）
        if 'limit' in feature_groups and 'momentum' in feature_groups:
            limit_momentum_combo = {
                'type': 'limit_momentum',
                'groups': ['limit', 'momentum'],
                'features': feature_groups['limit'][:3] + feature_groups['momentum'][:3],
                'weight_range': (0.3, 1.0)
            }
            factor_combinations.append(limit_momentum_combo)

        # 🔥 策略4：价格+波动率组合
        if 'price' in feature_groups and 'volatility' in feature_groups:
            price_vol_combo = {
                'type': 'price_volatility',
                'groups': ['price', 'volatility'],
                'features': feature_groups['price'][:4] + feature_groups['volatility'][:2],
                'weight_range': (0.1, 0.6)
            }
            factor_combinations.append(price_vol_combo)

        # 🔥 策略5：综合排名组合
        if 'ranking' in feature_groups:
            ranking_combo = {
                'type': 'ranking',
                'groups': ['ranking'],
                'features': feature_groups['ranking'][:6],
                'weight_range': (0.2, 0.9)
            }
            factor_combinations.append(ranking_combo)

        self.logger.info(f"生成{len(factor_combinations)}个个股特征组合用于强化学习")
        return factor_combinations

    def _load_market_status_data(self, X: pd.DataFrame) -> Dict:
        """加载真实市场状态数据"""
        try:
            # 🔥 尝试从train_sector_models.py的方法加载市场状态
            import json
            from pathlib import Path

            # 构建市场状态文件路径
            market_status_dir = Path("to/local_data/market_status")
            status_file = market_status_dir / "market_status.json"

            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    market_status_data = json.load(f)
                self.logger.info(f"成功加载市场状态数据，包含 {len(market_status_data)} 个日期")
                return market_status_data
            else:
                self.logger.warning(f"市场状态文件不存在: {status_file}")
                return {}

        except Exception as e:
            self.logger.warning(f"加载市场状态数据失败: {e}")
            return {}

    def _identify_stock_state(self, X_segment: pd.DataFrame) -> int:
        """基于个股特征识别当前状态 - 专注于个股层面"""
        if len(X_segment) == 0:
            return 0

        try:
            # 🔥 方法1：基于个股技术指标状态
            if 'rsi6' in X_segment.columns and 'macd' in X_segment.columns:
                rsi_mean = X_segment['rsi6'].mean()
                macd_mean = X_segment['macd'].mean()

                if rsi_mean > 70 and macd_mean > 0:
                    return 0  # 强势多头
                elif rsi_mean < 30 and macd_mean < 0:
                    return 1  # 强势空头
                elif 50 <= rsi_mean <= 70 and macd_mean > 0:
                    return 2  # 温和多头
                elif 30 <= rsi_mean <= 50 and macd_mean < 0:
                    return 3  # 温和空头
                else:
                    return 4  # 震荡

            # 🔥 方法2：基于个股涨跌停特征状态
            limit_features = ['is_limit_up', 'is_limit_down', 'consecutive_limit_up']
            available_limit = [col for col in limit_features if col in X_segment.columns]

            if available_limit:
                if 'is_limit_up' in X_segment.columns:
                    limit_up_ratio = X_segment['is_limit_up'].mean()
                    if limit_up_ratio > 0.1:  # 10%以上涨停
                        return 0  # 强势状态

                if 'is_limit_down' in X_segment.columns:
                    limit_down_ratio = X_segment['is_limit_down'].mean()
                    if limit_down_ratio > 0.05:  # 5%以上跌停
                        return 1  # 弱势状态

                if 'consecutive_limit_up' in X_segment.columns:
                    consecutive_mean = X_segment['consecutive_limit_up'].mean()
                    if consecutive_mean > 1:
                        return 2  # 连板状态

            # 🔥 方法3：基于个股价格动量状态
            momentum_features = ['pct_chg', 'momentum_5d', 'momentum_10d']
            available_momentum = [col for col in momentum_features if col in X_segment.columns]

            if available_momentum:
                momentum_mean = X_segment[available_momentum].mean().mean()
                if momentum_mean > 0.05:  # 5%以上涨幅
                    return 0  # 上涨趋势
                elif momentum_mean < -0.05:  # 5%以上跌幅
                    return 1  # 下跌趋势
                else:
                    return 4  # 震荡趋势

            # 🔥 方法4：基于个股成交量状态
            volume_features = ['turnover', 'volume_ratio', 'amplitude']
            available_volume = [col for col in volume_features if col in X_segment.columns]

            if available_volume:
                volume_activity = X_segment[available_volume].mean().mean()
                if volume_activity > 0.1:  # 高活跃度
                    return 5  # 活跃交易
                else:
                    return 4  # 平静交易

            # 默认震荡状态
            return 4

        except Exception as e:
            self.logger.warning(f"个股状态识别失败: {e}")
            return 4  # 默认震荡状态

    def _identify_market_state(self, X_segment: pd.DataFrame, market_features: Dict[str, np.ndarray]) -> int:
        """识别当前市场状态 - 基于真实市场数据"""
        if len(X_segment) == 0:
            return 0

        # 🔥 使用真实市场特征进行状态识别
        try:
            # 方法1：如果有市场状态特征，直接使用
            market_state_features = [
                'market_type', 'market_group', 'market_trend_strength',
                'trend_score', 'momentum_score', 'volatility_score'
            ]

            # 检查是否有市场状态特征
            available_market_features = [col for col in market_state_features if col in X_segment.columns]

            if available_market_features:
                # 使用真实市场状态特征
                market_values = X_segment[available_market_features].mean()

                # 基于market_group进行状态映射
                if 'market_group' in market_values:
                    market_group = market_values['market_group']
                    if market_group == 1:  # 牛市
                        return 0
                    elif market_group == 2:  # 熊市
                        return 1
                    elif market_group == 3:  # 震荡
                        return 4
                    elif market_group == 4:  # 极端
                        return 5

                # 基于trend_score和momentum_score进行状态映射
                if 'trend_score' in market_values and 'momentum_score' in market_values:
                    trend_score = market_values['trend_score']
                    momentum_score = market_values['momentum_score']

                    if trend_score > 0.7 and momentum_score > 0.6:
                        return 0  # 强势上涨
                    elif trend_score < 0.3 and momentum_score < 0.4:
                        return 1  # 强势下跌
                    elif 0.4 <= trend_score <= 0.6:
                        return 4  # 震荡
                    elif trend_score > 0.6:
                        return 2  # 温和上涨
                    else:
                        return 3  # 温和下跌

            # 方法2：基于技术指标计算市场状态
            technical_features = ['rsi6', 'macd', 'bb_position', 'ma_ratio_5_20']
            available_technical = [col for col in technical_features if col in X_segment.columns]

            if available_technical:
                tech_values = X_segment[available_technical].mean()

                # RSI状态判断
                rsi_state = 0
                if 'rsi6' in tech_values:
                    rsi = tech_values['rsi6']
                    if rsi > 70:
                        rsi_state = 1  # 超买
                    elif rsi < 30:
                        rsi_state = -1  # 超卖

                # MACD状态判断
                macd_state = 0
                if 'macd' in tech_values:
                    macd = tech_values['macd']
                    if macd > 0:
                        macd_state = 1  # 多头
                    else:
                        macd_state = -1  # 空头

                # 综合状态判断
                if rsi_state == 1 and macd_state == 1:
                    return 0  # 强势多头
                elif rsi_state == -1 and macd_state == -1:
                    return 1  # 强势空头
                elif rsi_state == 0 or macd_state == 0:
                    return 4  # 震荡
                else:
                    return 2  # 温和趋势

            # 方法3：基于价格和成交量特征
            price_volume_features = ['close', 'volume', 'turnover', 'amplitude']
            available_pv = [col for col in price_volume_features if col in X_segment.columns]

            if available_pv:
                pv_values = X_segment[available_pv].mean()

                # 基于振幅和换手率判断
                amplitude = pv_values.get('amplitude', 0)
                turnover = pv_values.get('turnover', 0)

                if amplitude > 0.08 and turnover > 0.05:  # 高振幅高换手
                    return 5  # 极端波动
                elif amplitude > 0.05:  # 中等振幅
                    return 2  # 活跃交易
                else:  # 低振幅
                    return 4  # 平静交易

            # 默认状态
            return 4  # 震荡

        except Exception as e:
            self.logger.warning(f"市场状态识别失败: {e}")
            return 4  # 默认震荡状态

    def _apply_weight_actions(self, current_weights: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """应用权重调整动作"""
        new_weights = current_weights.copy()

        for i, action in enumerate(actions):
            if action == 0:  # 大幅增加
                new_weights[i] *= 1.2
            elif action == 1:  # 小幅增加
                new_weights[i] *= 1.05
            elif action == 2:  # 保持
                pass
            elif action == 3:  # 小幅减少
                new_weights[i] *= 0.95
            elif action == 4:  # 大幅减少
                new_weights[i] *= 0.8

        # 归一化
        new_weights = np.abs(new_weights)
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            new_weights = np.ones(len(new_weights)) / len(new_weights)

        return new_weights

    def _calculate_rl_reward(self, X_segment: pd.DataFrame, y_segment: pd.Series,
                           weights: np.ndarray, factor_combinations: List[Dict]) -> float:
        """计算强化学习奖励 - 基于A股量化策略的多维度评估"""
        if len(X_segment) == 0 or len(y_segment) == 0:
            return 0

        # 🔥 专注个股特征组合的奖励函数
        reward = 0

        # 1. 个股收益预测准确性 (45%) - 最重要
        try:
            # 计算个股特征加权得分
            weighted_features = np.zeros(len(X_segment))
            valid_features = 0

            for i, weight in enumerate(weights):
                if i < X_segment.shape[1]:
                    feature_values = X_segment.iloc[:, i].fillna(0)
                    weighted_features += feature_values * weight
                    valid_features += 1

            if valid_features > 0:
                correlation = np.corrcoef(weighted_features, y_segment)[0, 1]
                if not np.isnan(correlation):
                    reward += correlation * 0.45
        except:
            pass

        # 2. 个股特征稳定性 (25%) - 减少噪音
        try:
            if len(weights) > 0:
                feature_stability = 1.0 / (1.0 + np.std(weights))
                reward += feature_stability * 0.25
        except:
            pass

        # 3. 🔥 A股个股涨停预测能力 (20%)
        try:
            limit_features = ['is_limit_up', 'limit_up_probability', 'consecutive_limit_up']
            available_limit = [col for col in limit_features if col in X_segment.columns]

            if available_limit:
                limit_signals = X_segment[available_limit].mean(axis=1)
                weighted_features = np.zeros(len(X_segment))
                for i, weight in enumerate(weights):
                    if i < X_segment.shape[1]:
                        weighted_features += X_segment.iloc[:, i].fillna(0) * weight

                limit_correlation = np.corrcoef(weighted_features, limit_signals)[0, 1]
                if not np.isnan(limit_correlation):
                    reward += abs(limit_correlation) * 0.2
        except:
            pass

        # 4. 特征多样性奖励 (10%) - 避免过度依赖单一特征
        try:
            if len(weights) > 0 and np.sum(np.abs(weights)) > 0:
                feature_diversity = 1.0 - np.max(np.abs(weights)) / np.sum(np.abs(weights))
                reward += feature_diversity * 0.1
        except:
            pass

        # 🔥 惩罚机制：避免极端权重和负相关
        try:
            # 惩罚极端权重
            if np.max(np.abs(weights)) > 5:
                reward -= 0.1

            # 惩罚负相关过多
            if reward < -0.1:
                reward -= 0.1
        except:
            pass

        return np.clip(reward, -1.0, 1.0)
    
    def predict(self, data: pd.DataFrame, target_sub_states: List[str] = None) -> SelectionResult:
        """预测选股 - 子状态约束版本"""
        if not self.is_trained:
            raise ValueError("模型未训练")

        if not self.sub_state_factor_combinations:
            raise ValueError("没有训练好的子状态模型")

        # 🔥 核心修改：按子状态分别预测
        all_selected_stocks = []
        all_confidence_scores = {}
        combined_feature_importance = {}

        # 🔥 修复：确保 target_sub_states 是列表类型
        if target_sub_states is None:
            target_sub_states = list(self.sub_state_factor_combinations.keys())
        elif isinstance(target_sub_states, str):
            # 如果传入的是字符串，转换为单元素列表
            target_sub_states = [target_sub_states]
            self.logger.warning(f"检测到传入字符串参数，已转换为列表: {target_sub_states}")
        elif not isinstance(target_sub_states, list):
            # 如果是其他可迭代类型，转换为列表
            target_sub_states = list(target_sub_states)

        self.logger.info(f"在子状态 {target_sub_states} 中进行遗传RL预测")

        for sub_state in target_sub_states:
            if sub_state not in self.sub_state_factor_combinations:
                self.logger.warning(f"子状态 {sub_state} 没有训练好的模型，跳过")
                continue

            # 🔥 关键：只在该子状态对应的股票中预测
            sub_state_data = data.copy()  # 暂时使用全部数据，实际应该筛选

            if len(sub_state_data) == 0:
                continue

            # 🔥 修复：使用训练时保存的特征列，确保一致性
            if self.feature_columns is None:
                self.logger.error(f"算法 {self.algorithm_name} 没有保存的特征列信息")
                continue

            # 检查数据中是否包含所需的特征列
            available_features = [col for col in self.feature_columns if col in sub_state_data.columns]
            missing_features = [col for col in self.feature_columns if col not in sub_state_data.columns]

            if len(available_features) < len(self.feature_columns) * 0.8:  # 至少80%的特征可用
                self.logger.warning(f"子状态 {sub_state} 数据中缺少过多训练时的特征列")
                self.logger.warning(f"需要 {len(self.feature_columns)} 个特征，只找到 {len(available_features)} 个")
                self.logger.debug(f"缺失的特征列: {missing_features[:5]}...")  # 只显示前5个
                continue

            X = sub_state_data[available_features].fillna(0)
            factor_combinations = self.sub_state_factor_combinations[sub_state]
            rl_weights = self.sub_state_rl_weights[sub_state]

            # 使用该子状态的因子组合计算得分
            composite_scores = np.zeros(len(X))

            for combination in factor_combinations:
                features = combination['features']
                weights = combination['weights']
                fitness = combination['fitness']

                # 计算该组合的得分
                combo_score = np.zeros(len(X))
                for i, feature in enumerate(features):
                    if feature in X.columns:
                        combo_score += X[feature].values * weights[i]

                # 按适应度加权
                composite_scores += combo_score * fitness

            # 应用强化学习权重
            for feature, weight in rl_weights.items():
                if feature in X.columns:
                    feature_idx = list(X.columns).index(feature)
                    composite_scores += X[feature].values * weight

            # 选择该子状态的top股票（每个子状态选择3-5只）
            n_select = min(5, len(composite_scores))
            top_indices = np.argsort(composite_scores)[-n_select:]

            sub_state_stocks = sub_state_data.iloc[top_indices]['stock_code'].tolist()
            all_selected_stocks.extend(sub_state_stocks)

            # 记录置信度
            for i, stock_code in enumerate(sub_state_stocks):
                score = float(composite_scores[top_indices[i]])
                all_confidence_scores[stock_code] = score

            # 合并特征重要性
            for feature, weight in rl_weights.items():
                if feature not in combined_feature_importance:
                    combined_feature_importance[feature] = 0
                combined_feature_importance[feature] += weight

        # 去重并限制总数
        unique_stocks = list(dict.fromkeys(all_selected_stocks))[:10]
        final_confidence_scores = {stock: all_confidence_scores[stock] for stock in unique_stocks}

        return SelectionResult(
            algorithm_name=self.algorithm_name,
            selected_stocks=unique_stocks,
            confidence_scores=final_confidence_scores,
            feature_importance=combined_feature_importance,
            performance_metrics={},
            sub_state=",".join(target_sub_states),
            sector="multi_sub_state"
        )
    
    def evaluate(self, data: pd.DataFrame, actual_returns: pd.Series) -> Dict[str, float]:
        """评估算法性能"""
        if not self.is_trained:
            return {}
        
        # 简化的评估逻辑
        result = self.predict(data)
        selected_indices = data[data['stock_code'].isin(result.selected_stocks)].index
        
        if len(selected_indices) > 0:
            selected_returns = actual_returns.iloc[selected_indices]
            return {
                'mean_return': selected_returns.mean(),
                'std_return': selected_returns.std(),
                'sharpe_ratio': selected_returns.mean() / selected_returns.std() if selected_returns.std() > 0 else 0,
                'win_rate': (selected_returns > 0).mean()
            }
        
        return {}
