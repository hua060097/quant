#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专业选股系统
基于A股特色和顶级量化研究的选股规则训练与预测系统
"""

import pandas as pd
import numpy as np
import logging
import pickle
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StockCandidate:
    """选股候选对象"""
    stock_code: str
    sector: str
    sub_state: str
    predicted_return: float
    confidence: float
    features: Dict[str, float]
    selection_reason: str

@dataclass
class SelectionRule:
    """选股规则"""
    sub_state: str
    model: Any
    feature_importance: Dict[str, float]
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    sample_count: int
    selection_criteria: Dict[str, Any]

@dataclass
class DailySelectionResult:
    """每日选股结果"""
    date: str
    selected_stocks: List[StockCandidate]
    sector_states: Dict[str, str]
    market_condition: str
    selection_confidence: float
    expected_return: float
    actual_return: Optional[float] = None
    win_count: int = 0
    total_count: int = 0

class ProfessionalStockSelectionSystem:
    """专业选股系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 最佳子状态（将在训练后动态确定）
        self.best_sub_states = []
        
        # 训练好的规则
        self.trained_rules: Dict[str, SelectionRule] = {}
        
        # A股特色特征权重
        self.a_share_features = {
            'limit_up_features': ['limit_up_count', 'limit_up_ratio', 'consecutive_limit_up'],
            'momentum_features': ['momentum_5d', 'momentum_20d', 'rsi14'],
            'volume_features': ['turnover_ratio', 'volume_ratio', 'amount_ratio'],
            'technical_features': ['macd_signal', 'bollinger_position', 'ma_trend'],
            'market_structure': ['market_cap_rank', 'industry_rank', 'sector_strength']
        }
        
    def train_all_sub_state_rules(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """训练所有子状态的选股规则"""
        
        self.logger.info("开始训练专业选股规则...")
        
        all_rules = {}
        successful_rules = {}
        
        for sub_state, data in training_data.items():
            self.logger.info(f"训练子状态 {sub_state} 的选股规则")
            
            try:
                rule = self._train_single_sub_state_rule(sub_state, data)
                all_rules[sub_state] = rule
                
                # 只保留高质量规则
                if rule and rule.win_rate > 0.35 and rule.sample_count >= 100:
                    successful_rules[sub_state] = rule
                    self.logger.info(f"子状态 {sub_state} 规则训练成功 - "
                                   f"胜率: {rule.win_rate:.2%}, 平均收益: {rule.avg_return:.4f}, "
                                   f"夏普比率: {rule.sharpe_ratio:.3f}")
                else:
                    self.logger.warning(f"子状态 {sub_state} 规则效果不佳，跳过")
                    
            except Exception as e:
                self.logger.error(f"训练子状态 {sub_state} 失败: {e}")
                continue
        
        self.trained_rules = successful_rules

        # 根据回测表现选择最佳2个子状态
        best_sub_states = self._select_best_sub_states_by_backtest(successful_rules, training_data)
        self.best_sub_states = best_sub_states

        self.logger.info(f"完成训练，共 {len(successful_rules)} 个子状态有有效规则")
        self.logger.info(f"选择的最佳子状态: {best_sub_states}")

        return {
            'all_rules': all_rules,
            'successful_rules': successful_rules,
            'best_sub_states': best_sub_states
        }
    
    def _train_single_sub_state_rule(self, sub_state: str, data: pd.DataFrame) -> Optional[SelectionRule]:
        """训练单个子状态的选股规则 - 内存优化版本"""

        if len(data) < 100:
            return None

        # 准备特征和目标变量
        feature_cols = self._get_feature_columns(data)
        if not feature_cols:
            return None

        X = data[feature_cols].fillna(0)
        y = data['future_return_2d'].fillna(0)

        # 内存优化：如果数据量过大，使用智能采样
        if len(data) > 50000:
            self.logger.info(f"子状态 {sub_state} 数据量较大 ({len(data)} 样本)，使用智能采样优化内存")
            X, y = self._smart_sampling(X, y, max_samples=50000)

        # 训练模型 - 使用内存优化的参数
        model = RandomForestRegressor(
            n_estimators=50,        # 减少树的数量以节省内存
            max_depth=8,           # 限制深度避免过深的树
            min_samples_split=20,
            min_samples_leaf=10,
            max_samples=0.8,       # 每棵树只使用80%的样本
            random_state=42,
            n_jobs=1               # 使用单线程避免内存冲突
        )

        try:
            model.fit(X, y)

            # 预测和评估
            y_pred = model.predict(X)
        except MemoryError as e:
            self.logger.error(f"训练子状态 {sub_state} 时内存不足: {e}")
            # 进一步减少数据量重试
            if len(X) > 10000:
                self.logger.info(f"进一步减少样本量到10000进行重试")
                X_small, y_small = self._smart_sampling(X, y, max_samples=10000)

                # 使用更保守的模型参数
                model_small = RandomForestRegressor(
                    n_estimators=20,
                    max_depth=6,
                    min_samples_split=50,
                    min_samples_leaf=20,
                    max_samples=0.7,
                    random_state=42,
                    n_jobs=1
                )

                try:
                    model_small.fit(X_small, y_small)
                    y_pred = model_small.predict(X_small)
                    model = model_small
                    X, y = X_small, y_small
                except MemoryError:
                    self.logger.error(f"即使减少样本量仍然内存不足，跳过子状态 {sub_state}")
                    return None
            else:
                return None
        
        # 计算性能指标
        performance = self._calculate_performance_metrics(y, y_pred)
        
        # 获取特征重要性
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        
        # 生成选股标准
        selection_criteria = self._generate_selection_criteria(sub_state, feature_importance, data)
        
        return SelectionRule(
            sub_state=sub_state,
            model=model,
            feature_importance=feature_importance,
            win_rate=performance['win_rate'],
            avg_return=performance['avg_return'],
            sharpe_ratio=performance['sharpe_ratio'],
            max_drawdown=performance['max_drawdown'],
            sample_count=len(y),  # 使用实际训练样本数量
            selection_criteria=selection_criteria
        )

    def _smart_sampling(self, X: pd.DataFrame, y: pd.Series, max_samples: int) -> tuple:
        """
        智能采样方法，保持数据分布的同时减少样本量

        Args:
            X: 特征数据
            y: 目标变量
            max_samples: 最大样本数

        Returns:
            采样后的 (X, y)
        """
        if len(X) <= max_samples:
            return X, y

        # 使用分层采样保持收益率分布
        try:
            # 将收益率分为5个分位数
            y_quantiles = pd.qcut(y, q=5, labels=False, duplicates='drop')

            # 计算每个分位数的采样数量
            samples_per_quantile = max_samples // len(y_quantiles.unique())

            sampled_indices = []
            for quantile in y_quantiles.unique():
                quantile_indices = y_quantiles[y_quantiles == quantile].index

                # 从每个分位数中随机采样
                sample_size = min(len(quantile_indices), samples_per_quantile)
                if sample_size > 0:
                    sampled_quantile_indices = np.random.choice(
                        quantile_indices, size=sample_size, replace=False
                    )
                    sampled_indices.extend(sampled_quantile_indices)

            # 如果采样数量不足，随机补充
            if len(sampled_indices) < max_samples:
                remaining_indices = list(set(X.index) - set(sampled_indices))
                additional_samples = min(len(remaining_indices), max_samples - len(sampled_indices))
                if additional_samples > 0:
                    additional_indices = np.random.choice(
                        remaining_indices, size=additional_samples, replace=False
                    )
                    sampled_indices.extend(additional_indices)

            return X.loc[sampled_indices], y.loc[sampled_indices]

        except Exception as e:
            self.logger.warning(f"分层采样失败: {e}，使用随机采样")
            # 回退到简单随机采样
            sampled_indices = np.random.choice(X.index, size=max_samples, replace=False)
            return X.loc[sampled_indices], y.loc[sampled_indices]
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """获取可用的特征列"""
        
        # 基础特征
        basic_features = [
            'pct_chg', 'turnover_ratio', 'volume_ratio', 'amount_ratio',
            'pe_ratio', 'pb_ratio', 'market_cap', 'float_market_cap'
        ]
        
        # A股特色特征
        a_share_features = [
            'limit_up_count', 'limit_up_ratio', 'consecutive_limit_up',
            'momentum_5d', 'momentum_20d', 'rsi14', 'macd_signal'
        ]
        
        # 技术指标特征
        technical_features = [
            'ma5', 'ma20', 'ma60', 'bollinger_upper', 'bollinger_lower',
            'atr14', 'cci14', 'williams_r'
        ]
        
        all_potential_features = basic_features + a_share_features + technical_features
        
        # 只返回数据中存在的特征
        available_features = [f for f in all_potential_features if f in data.columns]
        
        # 至少需要5个特征
        if len(available_features) < 5:
            # 使用数值型列作为特征
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['future_return_2d', 'date', 'stock_code']
            available_features = [c for c in numeric_cols if c not in exclude_cols][:20]
        
        return available_features
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算性能指标"""
        
        # 胜率（预测方向正确的比例）
        win_rate = np.mean((y_true > 0) == (y_pred > 0))
        
        # 平均收益
        avg_return = np.mean(y_true)
        
        # 夏普比率
        returns_std = np.std(y_true)
        sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
        
        # 最大回撤
        cumulative_returns = np.cumsum(y_true)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
        max_drawdown = np.min(drawdowns)
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown)
        }
    
    def _generate_selection_criteria(self, sub_state: str, feature_importance: Dict[str, float], 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """生成选股标准"""
        
        # 获取最重要的特征
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        criteria = {
            'top_features': [f[0] for f in top_features],
            'feature_thresholds': {},
            'min_confidence': 0.6,
            'max_selections': 10
        }
        
        # 为每个重要特征设置阈值
        for feature, importance in top_features:
            if feature in data.columns:
                feature_data = data[feature].dropna()
                if len(feature_data) > 0:
                    criteria['feature_thresholds'][feature] = {
                        'min': feature_data.quantile(0.3),
                        'max': feature_data.quantile(0.9),
                        'optimal': feature_data.quantile(0.7)
                    }
        
        return criteria

    def _select_best_sub_states(self, successful_rules: Dict[str, SelectionRule]) -> List[str]:
        """根据综合表现选择最佳的2个子状态 - 优化评分逻辑"""

        if len(successful_rules) <= 2:
            return list(successful_rules.keys())

        # 计算综合评分 - 重新设计评分逻辑
        rule_scores = []
        for sub_state, rule in successful_rules.items():
            # 计算盈亏比
            win_avg = 0
            loss_avg = 0
            # 这里简化处理，实际应该从训练数据中计算
            if rule.avg_return > 0:
                profit_loss_ratio = 2.0  # 假设盈亏比
            else:
                profit_loss_ratio = abs(rule.avg_return) / 0.01 if rule.avg_return < 0 else 0.5

            # 期望收益 = 胜率 * 平均盈利 - (1-胜率) * 平均亏损
            expected_return = rule.win_rate * abs(rule.avg_return) if rule.avg_return > 0 else rule.avg_return

            # 综合评分 = 期望收益 * 0.5 + 胜率 * 0.2 + 夏普比率 * 0.2 + 样本数权重 * 0.1
            sample_weight = min(rule.sample_count / 1000, 1.0)

            # 优化评分逻辑：降低阈值，更合理的评分机制
            min_expected_return = 0.001  # 降低最小期望收益阈值
            min_win_rate = 0.52  # 降低最小胜率阈值

            # 基础评分：只要期望收益为正就给分
            if expected_return > 0:
                base_score = expected_return * 0.5 + rule.win_rate * 0.3

                # 额外奖励：满足更高标准的规则
                if expected_return > min_expected_return and rule.win_rate > min_win_rate:
                    stability_bonus = min(rule.win_rate - 0.5, 0.2) * 0.5
                    sharpe_bonus = max(rule.sharpe_ratio, 0) * 0.1
                    composite_score = base_score + stability_bonus + sharpe_bonus + sample_weight * 0.1
                else:
                    # 基础评分，但有一定折扣
                    composite_score = base_score * 0.7 + sample_weight * 0.05
            else:
                composite_score = 0  # 期望收益为负的规则评分为0

            rule_scores.append((sub_state, composite_score, rule))

            self.logger.info(f"子状态 {sub_state}: 胜率={rule.win_rate:.2%}, "
                           f"平均收益={rule.avg_return:.4f}, 期望收益={expected_return:.4f}, "
                           f"夏普比率={rule.sharpe_ratio:.3f}, 综合评分={composite_score:.4f}")

        # 按综合评分排序，选择前2个
        rule_scores.sort(key=lambda x: x[1], reverse=True)

        # 只选择评分大于0的规则
        valid_rules = [item for item in rule_scores if item[1] > 0]

        if len(valid_rules) >= 2:
            best_sub_states = [item[0] for item in valid_rules[:2]]
        elif len(valid_rules) == 1:
            best_sub_states = [valid_rules[0][0]]
            self.logger.warning("只有1个子状态有正期望收益")
        else:
            # 如果没有正期望收益的规则，选择胜率最高的2个
            rule_scores.sort(key=lambda x: x[2].win_rate, reverse=True)
            best_sub_states = [item[0] for item in rule_scores[:2]]
            self.logger.warning("没有正期望收益的规则，按胜率选择")

        self.logger.info(f"根据优化评分选择的最佳子状态: {best_sub_states}")

        return best_sub_states

    def _select_best_sub_states_by_backtest(self, successful_rules: Dict[str, SelectionRule],
                                          training_data: Dict[str, pd.DataFrame]) -> List[str]:
        """基于回测表现选择最佳的2个子状态"""

        if len(successful_rules) <= 2:
            return list(successful_rules.keys())

        self.logger.info("开始回测评估各子状态表现...")

        backtest_results = {}

        for sub_state, rule in successful_rules.items():
            if sub_state not in training_data:
                continue

            self.logger.info(f"回测子状态: {sub_state}")

            # 获取该子状态的训练数据
            data = training_data[sub_state]

            # 随机选择30%的数据进行回测
            sample_size = int(len(data) * 0.3)
            if sample_size < 100:
                sample_size = min(len(data), 100)

            backtest_data = data.sample(n=sample_size, random_state=42)

            # 进行回测
            backtest_result = self._perform_backtest(rule, backtest_data, sub_state)
            backtest_results[sub_state] = backtest_result

            self.logger.info(f"子状态 {sub_state} 回测结果: "
                           f"总收益={backtest_result['total_return']:.4f}, "
                           f"最大回撤={backtest_result['max_drawdown']:.4f}, "
                           f"胜率={backtest_result['win_rate']:.2%}, "
                           f"夏普比率={backtest_result['sharpe_ratio']:.3f}")

        # 根据回测结果选择最佳子状态
        best_sub_states = self._rank_sub_states_by_backtest(backtest_results)

        self.logger.info(f"基于回测选择的最佳子状态: {best_sub_states}")

        return best_sub_states

    def _perform_backtest(self, rule: SelectionRule, data: pd.DataFrame, sub_state: str = None) -> Dict[str, float]:
        """对单个子状态进行回测"""

        # 准备特征
        feature_cols = [f for f in rule.feature_importance.keys() if f in data.columns]
        if not feature_cols:
            return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0}

        X = data[feature_cols].fillna(0)

        # 获取实际收益（预测收益在这里不需要，我们直接用实际收益评估）
        actual_returns = data['future_return_2d'].fillna(0)

        # 计算置信度
        confidences = self._calculate_prediction_confidence(X, rule)

        # 选择高置信度的预测
        min_confidence = rule.selection_criteria.get('min_confidence', 0.6)
        high_conf_mask = confidences >= min_confidence

        if high_conf_mask.sum() == 0:
            return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0}

        # 获取高置信度的实际收益
        selected_returns = actual_returns[high_conf_mask]

        # 计算回测指标
        total_return = selected_returns.sum()
        win_rate = (selected_returns > 0).mean()

        # 计算最大回撤
        cumulative_returns = selected_returns.cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        # 计算夏普比率
        if len(selected_returns) > 1:
            sharpe_ratio = selected_returns.mean() / (selected_returns.std() + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'sample_count': len(selected_returns)
        }

    def _rank_sub_states_by_backtest(self, backtest_results: Dict[str, Dict[str, float]]) -> List[str]:
        """根据回测结果排序子状态"""

        # 计算综合评分
        scores = []
        for sub_state, result in backtest_results.items():
            # 综合评分 = 总收益 * 0.4 + (1-最大回撤) * 0.3 + 胜率 * 0.2 + 夏普比率 * 0.1
            score = (
                result['total_return'] * 0.4 +
                (1 - result['max_drawdown']) * 0.3 +
                result['win_rate'] * 0.2 +
                max(result['sharpe_ratio'], 0) * 0.1
            )
            scores.append((sub_state, score, result))

        # 按评分排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 选择前2个
        best_sub_states = [item[0] for item in scores[:2]]

        return best_sub_states

    def predict_and_select_stocks(self, sector_states: pd.DataFrame,
                                stock_features: pd.DataFrame,
                                target_date: str) -> List[StockCandidate]:
        """预测并选择股票 - 从子状态对应的板块中选股"""

        if not self.trained_rules:
            self.logger.warning("没有训练好的规则")
            return []

        self.logger.info(f"开始预测选股 - 日期: {target_date}")

        # 获取当日数据
        daily_sector_states = sector_states[sector_states['date'] == target_date]
        daily_stock_features = stock_features[stock_features['date'] == target_date]

        if daily_sector_states.empty or daily_stock_features.empty:
            self.logger.warning(f"日期 {target_date} 没有数据")
            return []

        candidates = []

        # 为每个最佳子状态寻找股票
        for sub_state in self.best_sub_states:
            if sub_state not in self.trained_rules:
                continue

            rule = self.trained_rules[sub_state]

            # 获取该子状态对应的板块
            target_sectors = daily_sector_states[
                daily_sector_states['sub_state'] == sub_state
            ]['sector_code'].tolist()

            if not target_sectors:
                self.logger.info(f"子状态 {sub_state} 在当日没有出现")
                continue

            self.logger.info(f"子状态 {sub_state} 对应板块: {target_sectors}")

            # 从对应板块中选择股票
            stock_candidates = self._select_stocks_from_sectors(
                target_sectors, daily_stock_features, rule, sub_state
            )

            candidates.extend(stock_candidates)

        # 过滤未上市股票
        candidates = self._filter_valid_trading_stocks(candidates, daily_stock_features, target_date)

        # 最终筛选和排序
        final_candidates = self._final_selection(candidates)

        self.logger.info(f"最终选择 {len(final_candidates)} 只股票")

        return final_candidates
    
    def _select_stocks_from_sectors(self, sectors: List[str], stock_data: pd.DataFrame,
                                  rule: SelectionRule, sub_state: str) -> List[StockCandidate]:
        """从指定板块中选股 - 使用股票-板块映射"""

        candidates = []

        for sector in sectors:
            # 获取板块内股票 - 使用股票-板块映射
            sector_stocks = self._get_stocks_in_sector(sector, stock_data)

            if sector_stocks.empty:
                self.logger.debug(f"板块 {sector} 没有找到股票")
                continue

            self.logger.debug(f"板块 {sector} 找到 {len(sector_stocks)} 只股票")

            # 应用选股规则
            sector_candidates = self._apply_selection_rule(sector_stocks, rule, sector, sub_state)
            candidates.extend(sector_candidates)

        return candidates

    def _get_stocks_in_sector(self, sector_code: str, stock_data: pd.DataFrame) -> pd.DataFrame:
        """获取指定板块的股票数据"""
        try:
            # 导入股票板块映射函数
            from data_loaders import get_stock_list_in_sector

            # 获取板块内的股票列表
            sector_stocks = get_stock_list_in_sector(sector_code)

            if not sector_stocks:
                return pd.DataFrame()

            # 过滤出在当日数据中存在的股票
            available_stocks = stock_data[
                stock_data['stock_code'].isin(sector_stocks)
            ]

            return available_stocks

        except Exception as e:
            self.logger.warning(f"获取板块 {sector_code} 股票失败: {e}")
            return pd.DataFrame()
    
    def _apply_selection_rule(self, stocks: pd.DataFrame, rule: SelectionRule,
                            sector: str, sub_state: str) -> List[StockCandidate]:
        """应用选股规则 - 从指定板块的股票中选择"""

        candidates = []
        feature_cols = [f for f in rule.feature_importance.keys() if f in stocks.columns]

        if not feature_cols:
            self.logger.warning(f"板块 {sector} 子状态 {sub_state} 没有可用特征")
            return candidates

        # 准备特征数据
        X = stocks[feature_cols].fillna(0)

        # 预测收益
        predicted_returns = rule.model.predict(X)

        # 计算置信度
        confidences = self._calculate_prediction_confidence(X, rule)

        # 生成候选股票
        for idx, (_, stock_row) in enumerate(stocks.iterrows()):
            if confidences[idx] >= rule.selection_criteria.get('min_confidence', 0.6):
                stock_code = stock_row.get('stock_code', '')

                candidate = StockCandidate(
                    stock_code=stock_code,
                    sector=sector,  # 使用传入的板块信息
                    sub_state=sub_state,
                    predicted_return=predicted_returns[idx],
                    confidence=confidences[idx],
                    features={col: stock_row.get(col, 0) for col in feature_cols[:5]},
                    selection_reason=f"{sub_state}规则选择(板块:{sector})"
                )
                candidates.append(candidate)

        # 按预测收益排序，取前几名
        candidates.sort(key=lambda x: x.predicted_return, reverse=True)
        return candidates[:5]  # 每个板块最多5只

    def _filter_valid_trading_stocks(self, candidates: List[StockCandidate],
                                   stock_features: pd.DataFrame, target_date: str) -> List[StockCandidate]:
        """过滤出有效交易的股票（排除未上市股票）"""

        valid_candidates = []
        filtered_count = 0

        for candidate in candidates:
            # 检查股票在当日是否有交易数据
            stock_data = stock_features[stock_features['stock_code'] == candidate.stock_code]

            if not stock_data.empty:
                volume = stock_data['volume'].iloc[0]

                # 只选择有交易量的股票（已上市）
                if volume > 0:
                    valid_candidates.append(candidate)
                else:
                    filtered_count += 1
                    self.logger.warning(f"过滤未上市股票: {candidate.stock_code} 在 {target_date} (volume=0)")
            else:
                filtered_count += 1
                self.logger.warning(f"过滤无数据股票: {candidate.stock_code} 在 {target_date}")

        if filtered_count > 0:
            self.logger.info(f"共过滤了 {filtered_count} 只未上市/无数据股票")

        return valid_candidates

    def _calculate_prediction_confidence(self, X: pd.DataFrame, rule: SelectionRule) -> np.ndarray:
        """计算预测置信度"""
        
        # 基于特征重要性和数据质量计算置信度
        confidences = np.ones(len(X)) * 0.5
        
        for i, (_, row) in enumerate(X.iterrows()):
            confidence = 0.5
            
            # 基于重要特征的置信度
            for feature, importance in rule.feature_importance.items():
                if feature in row.index:
                    feature_value = row[feature]
                    if not pd.isna(feature_value):
                        confidence += importance * 0.5
            
            confidences[i] = min(confidence, 0.95)
        
        return confidences
    
    def _final_selection(self, candidates: List[StockCandidate]) -> List[StockCandidate]:
        """最终选股"""
        
        if not candidates:
            return []
        
        # 按综合评分排序
        for candidate in candidates:
            candidate.score = (
                candidate.predicted_return * 0.4 +
                candidate.confidence * 0.3 +
                (1.0 if 'limit_up' in candidate.sub_state else 0.5) * 0.3
            )
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # 确保每个子状态至少选1只，最多选2只
        final_candidates = []
        sub_state_counts = {}
        
        for candidate in candidates:
            current_count = sub_state_counts.get(candidate.sub_state, 0)
            if current_count < 1:  # 每个子状态最多1只
                final_candidates.append(candidate)
                sub_state_counts[candidate.sub_state] = current_count + 1
                
                if len(final_candidates) >= 2:  # 总共最多2只
                    break
        
        return final_candidates

    def _get_stock_real_sector(self, stock_code: str) -> str:
        """获取股票真实的板块信息"""
        try:
            # 导入股票板块映射函数
            from data_loaders import get_stock_sectors

            # 获取股票所属的板块列表
            sectors = get_stock_sectors(stock_code)

            if sectors:
                # 优先选择TGN概念板块
                tgn_sectors = [s for s in sectors if s.startswith('TGN')]
                if tgn_sectors:
                    return tgn_sectors[0]  # 返回第一个TGN板块
                else:
                    return sectors[0]  # 返回第一个板块
            else:
                return 'unknown'

        except Exception as e:
            self.logger.warning(f"获取股票 {stock_code} 板块信息失败: {e}")
            return 'unknown'

    def evaluate_final_performance(self, selected_stocks: List[StockCandidate],
                                 actual_returns: Dict[str, float]) -> Dict[str, Any]:
        """评估最终选股表现"""
        
        if not selected_stocks:
            return {
                'total_return': 0.0,
                'avg_return': 0.0,
                'win_rate': 0.0,
                'selected_stocks': []
            }
        
        total_return = 0.0
        win_count = 0
        stock_results = []
        
        for stock in selected_stocks:
            actual_return = actual_returns.get(stock.stock_code, 0.0)
            total_return += actual_return
            
            is_win = actual_return > 0
            if is_win:
                win_count += 1
            
            stock_result = {
                'stock_code': stock.stock_code,
                'sector': stock.sector,
                'sub_state': stock.sub_state,
                'predicted_return': stock.predicted_return,
                'actual_return': actual_return,
                'confidence': stock.confidence,
                'is_win': is_win,
                'selection_reason': stock.selection_reason
            }
            stock_results.append(stock_result)
        
        avg_return = total_return / len(selected_stocks)
        win_rate = win_count / len(selected_stocks)
        
        return {
            'total_return': total_return,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'win_count': win_count,
            'total_count': len(selected_stocks),
            'selected_stocks': stock_results
        }
    
    def save_system(self, filename: str = None):
        """保存完整的系统状态，支持独立预测"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"professional_system_{timestamp}.pkl"

        # 保存完整的系统状态
        save_data = {
            'trained_rules': self.trained_rules,
            'best_sub_states': self.best_sub_states,
            'a_share_features': self.a_share_features,
            'system_version': '2.0',  # 版本标识
            'save_timestamp': datetime.now().isoformat(),
            # 添加板块-股票映射信息
            'sector_stock_mapping': getattr(self, 'sector_stock_mapping', {}),
            # 添加特征列信息
            'feature_columns': getattr(self, 'feature_columns', [])
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

        self.logger.info(f"完整系统已保存到: {filename}")
        self.logger.info(f"包含 {len(self.trained_rules)} 个训练规则")
        self.logger.info(f"最佳子状态: {self.best_sub_states}")
        return filename

    def load_system(self, filename: str):
        """加载完整的系统状态，支持独立预测"""
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # 加载核心组件
        self.trained_rules = save_data['trained_rules']
        self.best_sub_states = save_data['best_sub_states']
        self.a_share_features = save_data.get('a_share_features', self.a_share_features)

        # 加载扩展信息
        self.sector_stock_mapping = save_data.get('sector_stock_mapping', {})
        self.feature_columns = save_data.get('feature_columns', [])

        version = save_data.get('system_version', '1.0')
        save_time = save_data.get('save_timestamp', 'unknown')

        self.logger.info(f"系统已从 {filename} 加载 (版本: {version})")
        self.logger.info(f"保存时间: {save_time}")
        self.logger.info(f"加载了 {len(self.trained_rules)} 个训练规则")
        self.logger.info(f"最佳子状态: {self.best_sub_states}")

        return True
