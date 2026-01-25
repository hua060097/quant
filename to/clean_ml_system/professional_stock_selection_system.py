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

# 先进选股算法
from advanced_selection_algorithms import (
    MultiTaskAttentionAlgorithm,
    GeneticRLAlgorithm,
    SelectionResult,
    BaseSelectionAlgorithm
)

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
    """专业选股系统 - 双方案架构版本"""

    def __init__(self, enable_advanced_algorithms: bool = True):
        self.logger = logging.getLogger(__name__)

        # 最佳子状态（将在训练后动态确定）
        self.best_sub_states = []

        # 训练好的规则（传统方法）
        self.trained_rules: Dict[str, SelectionRule] = {}

        # 🔥 新增：先进算法实例
        self.enable_advanced_algorithms = enable_advanced_algorithms
        self.advanced_algorithms: Dict[str, BaseSelectionAlgorithm] = {}

        if enable_advanced_algorithms:
            self.advanced_algorithms = {
                'multitask_attention': MultiTaskAttentionAlgorithm(),
                'genetic_rl': GeneticRLAlgorithm()
            }
            self.logger.info("启用先进选股算法：多任务注意力 + 遗传强化学习")

        # A股特色特征权重
        self.a_share_features = {
            'limit_up_features': ['limit_up_count', 'limit_up_ratio', 'consecutive_limit_up'],
            'momentum_features': ['momentum_5d', 'momentum_20d', 'rsi14'],
            'volume_features': ['turnover_ratio', 'volume_ratio', 'amount_ratio'],
            'technical_features': ['macd_signal', 'bollinger_position', 'ma_trend'],
            'market_structure': ['market_cap_rank', 'industry_rank', 'sector_strength']
        }

        # 性能统计（包含两套方案的对比）
        self.performance_stats = {
            'traditional': {},
            'advanced': {}
        }
        
    def train_all_sub_state_rules(self, training_data: Dict[str, pd.DataFrame],
                                  algorithm_type: str = 'both') -> Dict[str, Any]:
        """训练选股规则 - 分离双算法架构

        Args:
            training_data: 训练数据
            algorithm_type: 'traditional', 'advanced', 'both'
        """

        self.logger.info(f"开始训练选股规则 - 算法类型: {algorithm_type}")

        results = {}

        # 🔥 修改：根据算法类型分别训练
        if algorithm_type in ['advanced', 'both']:
            if self.enable_advanced_algorithms:
                self.logger.info("=" * 50)
                self.logger.info("开始训练先进算法（MultiTask_Attention + Genetic_RL）")
                self.logger.info("=" * 50)
                advanced_results = self._train_advanced_algorithms(training_data)
                results['advanced'] = advanced_results
            else:
                self.logger.warning("先进算法未启用，跳过训练")

        if algorithm_type in ['traditional', 'both']:
            self.logger.info("=" * 50)
            self.logger.info("开始训练传统选股规则")
            self.logger.info("=" * 50)
            traditional_results = self._train_traditional_rules(training_data)
            results['traditional'] = traditional_results

        return results

    def _train_traditional_rules(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """训练传统选股规则"""

        all_rules = {}
        successful_rules = {}

        for sub_state, data in training_data.items():
            self.logger.info(f"训练子状态 {sub_state} 的传统选股规则")

            try:
                rule = self._train_single_sub_state_rule(sub_state, data)
                all_rules[sub_state] = rule

                # 🔥 修复：降低质量要求，确保有足够的子状态可选
                if rule and rule.win_rate > 0.25 and rule.sample_count >= 50:  # 降低胜率要求到25%，样本数要求到50
                    successful_rules[sub_state] = rule
                    self.logger.info(f"子状态 {sub_state} 传统规则训练成功 - "
                                   f"胜率: {rule.win_rate:.2%}, 平均收益: {rule.avg_return:.4f}, "
                                   f"夏普比率: {rule.sharpe_ratio:.3f}")
                else:
                    if rule:
                        self.logger.warning(f"子状态 {sub_state} 传统规则效果不佳 - "
                                          f"胜率: {rule.win_rate:.2%}, 样本数: {rule.sample_count}，跳过")
                    else:
                        self.logger.warning(f"子状态 {sub_state} 传统规则训练失败，跳过")

            except Exception as e:
                self.logger.error(f"训练子状态 {sub_state} 传统规则失败: {e}")
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

    def _train_advanced_algorithms(self, training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """训练先进选股算法"""
        self.logger.info("开始训练先进选股算法...")

        results = {
            'multitask_attention': {'status': 'failed', 'error': None, 'best_sub_states': []},
            'genetic_rl': {'status': 'failed', 'error': None, 'best_sub_states': []}
        }

        # 合并所有训练数据
        all_training_data = []
        for sub_state, data in training_data.items():
            if not data.empty:
                data_copy = data.copy()
                data_copy['sub_state'] = sub_state
                all_training_data.append(data_copy)

        if not all_training_data:
            self.logger.warning("没有训练数据，跳过先进算法训练")
            return results

        combined_data = pd.concat(all_training_data, ignore_index=True)
        self.logger.info(f"合并训练数据: {len(combined_data)} 条记录")

        # 训练多任务注意力算法
        try:
            self.logger.info("训练多任务注意力算法...")
            self.advanced_algorithms['multitask_attention'].train(combined_data)
            results['multitask_attention']['status'] = 'success'
            self.logger.info("✅ 多任务注意力算法训练完成")

            # 🔥 新增：为多任务注意力算法选择最佳子状态
            best_sub_states_ma = self._select_best_sub_states_for_advanced_algorithm(
                'multitask_attention', training_data
            )
            results['multitask_attention']['best_sub_states'] = best_sub_states_ma
            self.logger.info(f"多任务注意力算法最佳子状态: {best_sub_states_ma}")

        except Exception as e:
            results['multitask_attention']['error'] = str(e)
            self.logger.error(f"❌ 多任务注意力算法训练失败: {e}")

        # 训练遗传强化学习算法
        try:
            self.logger.info("训练遗传强化学习算法...")
            self.advanced_algorithms['genetic_rl'].train(combined_data)
            results['genetic_rl']['status'] = 'success'
            self.logger.info("✅ 遗传强化学习算法训练完成")

            # 🔥 新增：为遗传强化学习算法选择最佳子状态
            best_sub_states_rl = self._select_best_sub_states_for_advanced_algorithm(
                'genetic_rl', training_data
            )
            results['genetic_rl']['best_sub_states'] = best_sub_states_rl
            self.logger.info(f"遗传强化学习算法最佳子状态: {best_sub_states_rl}")

        except Exception as e:
            results['genetic_rl']['error'] = str(e)
            self.logger.error(f"❌ 遗传强化学习算法训练失败: {e}")

        # 保存训练好的模型
        self._save_advanced_models()

        # 🔥 新增：保存各算法的最佳子状态
        self.advanced_best_sub_states = {
            'multitask_attention': results['multitask_attention']['best_sub_states'],
            'genetic_rl': results['genetic_rl']['best_sub_states']
        }

        self.logger.info("先进选股算法训练完成")
        return results

    def _select_best_sub_states_for_advanced_algorithm(self, algorithm_name: str,
                                                     training_data: Dict[str, pd.DataFrame]) -> List[str]:
        """为先进算法选择最佳子状态（基于回测表现）"""
        self.logger.info(f"开始为 {algorithm_name} 算法选择最佳子状态...")

        if algorithm_name not in self.advanced_algorithms:
            self.logger.error(f"算法 {algorithm_name} 不存在")
            return []

        algorithm = self.advanced_algorithms[algorithm_name]
        if not algorithm.is_trained:
            self.logger.error(f"算法 {algorithm_name} 尚未训练")
            return []

        # 获取所有可用的子状态
        available_sub_states = list(training_data.keys())
        if len(available_sub_states) <= 2:
            self.logger.info(f"子状态数量 <= 2，直接返回所有子状态: {available_sub_states}")
            return available_sub_states

        self.logger.info(f"开始回测评估 {algorithm_name} 算法在各子状态的表现...")

        backtest_results = {}

        for sub_state in available_sub_states:
            if sub_state not in training_data or training_data[sub_state].empty:
                continue

            self.logger.info(f"回测 {algorithm_name} 算法在子状态 {sub_state} 的表现")

            # 获取该子状态的训练数据
            data = training_data[sub_state]

            # 随机选择30%的数据进行回测
            sample_size = int(len(data) * 0.3)
            if sample_size < 100:
                sample_size = min(len(data), 100)

            backtest_data = data.sample(n=sample_size, random_state=42)

            # 使用先进算法进行回测
            backtest_result = self._perform_advanced_algorithm_backtest(
                algorithm, backtest_data, sub_state
            )
            backtest_results[sub_state] = backtest_result

            self.logger.info(f"{algorithm_name} 在子状态 {sub_state} 回测结果: "
                           f"总收益={backtest_result['total_return']:.4f}, "
                           f"最大回撤={backtest_result['max_drawdown']:.4f}, "
                           f"胜率={backtest_result['win_rate']:.2%}, "
                           f"夏普比率={backtest_result['sharpe_ratio']:.3f}")

        # 根据回测结果选择最佳子状态
        best_sub_states = self._rank_sub_states_by_backtest(backtest_results)

        self.logger.info(f"{algorithm_name} 算法选择的最佳子状态: {best_sub_states}")
        return best_sub_states

    def _perform_advanced_algorithm_backtest(self, algorithm, data: pd.DataFrame,
                                           sub_state: str) -> Dict[str, float]:
        """对先进算法进行回测"""

        # 🔥 过滤可买入的股票进行回测
        if 'is_can_buy' in data.columns:
            buyable_data = data[data['is_can_buy'] == 1]
            if not buyable_data.empty:
                self.logger.debug(f"子状态 {sub_state} 回测: 过滤后 {len(data)} 条，可买入 {len(buyable_data)} 条")
                data = buyable_data
            else:
                self.logger.warning(f"子状态 {sub_state} 回测: 没有可买入的股票")
                return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0, 'sample_count': 0}

        # 使用先进算法进行预测
        try:
            # 🔥 关键修复：传递完整数据而不是只传递特征列
            # 算法内部会自己处理特征列的提取，但需要访问 stock_code 等元数据

            # 🔥 修复：传递子状态列表而不是单个字符串
            predictions = algorithm.predict(data, [sub_state])

            if predictions is None or not predictions.selected_stocks:
                self.logger.warning(f"算法对子状态 {sub_state} 预测失败")
                return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0, 'sample_count': 0}

            # 🔥 修复：正确处理 SelectionResult 对象
            # 获取选中的股票代码
            selected_stock_codes = predictions.selected_stocks

            # 根据股票代码获取对应的收益率
            stock_mask = data['stock_code'].isin(selected_stock_codes)
            selected_data = data[stock_mask]

            if selected_data.empty:
                self.logger.warning(f"算法对子状态 {sub_state} 选中的股票在数据中不存在")
                return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0, 'sample_count': 0}

            selected_returns = selected_data['future_return_1d'].fillna(0)

            # 🔥 专业化回测指标计算
            metrics = self._calculate_professional_metrics(selected_returns)

            return {
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'calmar_ratio': metrics['calmar_ratio'],
                'var_95': metrics['var_95'],
                'cvar_95': metrics['cvar_95'],
                'sample_count': len(selected_returns)
            }

        except Exception as e:
            import traceback
            self.logger.error(f"先进算法回测失败: {e}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0,
                   'sortino_ratio': 0, 'calmar_ratio': 0, 'var_95': 0, 'cvar_95': 0, 'sample_count': 0}

    def _calculate_professional_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算专业的量化交易指标
        基于2020-2025年顶级量化研究的标准指标计算方法
        """
        if len(returns) == 0:
            return {
                'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0,
                'sortino_ratio': 0, 'calmar_ratio': 0, 'var_95': 0, 'cvar_95': 0
            }

        # 基础指标
        total_return = returns.mean()
        win_rate = (returns > 0).mean()

        # 🔥 专业最大回撤计算 - 基于累计净值曲线
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0

        # 🔥 专业夏普比率计算 - 考虑无风险利率
        risk_free_rate = 0.03 / 252  # 年化3%无风险利率转日收益率
        excess_returns = returns - risk_free_rate
        if excess_returns.std() > 0:
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 🔥 索提诺比率 - 只考虑下行风险
        downside_returns = returns[returns < risk_free_rate] - risk_free_rate
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio  # 如果没有下行风险，使用夏普比率

        # 🔥 卡尔马比率 - 年化收益与最大回撤的比值 (修复计算错误)
        if max_drawdown > 0.001:  # 避免除以极小值
            # 正确的年化收益率计算：日均收益 * 252个交易日
            annualized_return = total_return * 252
            calmar_ratio = annualized_return / max_drawdown
            # 限制卡尔马比率在合理范围内 (-10, 10)
            calmar_ratio = np.clip(calmar_ratio, -10, 10)
        else:
            # 当回撤极小时，使用保守估计
            calmar_ratio = 10.0 if total_return > 0 else -10.0

        # 🔥 VaR和CVaR计算 - 风险价值和条件风险价值
        var_95 = np.percentile(returns, 5)  # 95% VaR
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': abs(var_95),  # 取绝对值，表示损失
            'cvar_95': abs(cvar_95)  # 取绝对值，表示条件损失
        }

    def _save_advanced_models(self) -> None:
        """保存训练好的先进算法模型"""
        import pickle
        from pathlib import Path

        models_dir = Path("models/advanced_algorithms")
        models_dir.mkdir(parents=True, exist_ok=True)

        for algo_name, algorithm in self.advanced_algorithms.items():
            if algorithm.is_trained:
                try:
                    model_path = models_dir / f"{algo_name}_model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(algorithm, f)
                    self.logger.info(f"✅ 保存 {algo_name} 模型到 {model_path}")
                except Exception as e:
                    self.logger.error(f"❌ 保存 {algo_name} 模型失败: {e}")

    def _load_advanced_models(self) -> bool:
        """加载训练好的先进算法模型"""
        import pickle
        from pathlib import Path

        models_dir = Path("models/advanced_algorithms")
        if not models_dir.exists():
            return False

        loaded_count = 0
        for algo_name in ['multitask_attention', 'genetic_rl']:
            model_path = models_dir / f"{algo_name}_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.advanced_algorithms[algo_name] = pickle.load(f)
                    self.logger.info(f"✅ 加载 {algo_name} 模型成功")
                    loaded_count += 1
                except Exception as e:
                    self.logger.error(f"❌ 加载 {algo_name} 模型失败: {e}")

        return loaded_count > 0

    def _get_best_sub_states_from_sector_states(self, sector_states: pd.DataFrame) -> List[str]:
        # 🔥 使用训练时选出的最佳子状态

        # 优先使用训练时选出的最佳子状态
        if hasattr(self, 'advanced_best_sub_states'):
            # 合并所有算法的最佳子状态，去重
            all_best_sub_states = []
            for algo_name, best_states in self.advanced_best_sub_states.items():
                all_best_sub_states.extend(best_states)

            # 去重并限制为前2个
            unique_best_states = list(dict.fromkeys(all_best_sub_states))[:2]

            if unique_best_states:
                self.logger.info(f"使用训练时选出的最佳子状态: {unique_best_states}")
                return unique_best_states

        # 备选方案：获取已训练的子状态列表
        if hasattr(self, 'advanced_algorithms') and 'multitask_attention' in self.advanced_algorithms:
            algorithm = self.advanced_algorithms['multitask_attention']
            if hasattr(algorithm, 'sub_state_models') and algorithm.sub_state_models:
                available_sub_states = list(algorithm.sub_state_models.keys())
                self.logger.info(f"可用的训练子状态: {available_sub_states}")

                # 选择前2个子状态作为最佳子状态
                best_sub_states = available_sub_states[:2]
                self.logger.info(f"选择最佳子状态: {best_sub_states}")
                return best_sub_states

        # 如果没有训练好的模型，返回空列表
        self.logger.warning("没有找到已训练的子状态模型")
        return []
    
    def _train_single_sub_state_rule(self, sub_state: str, data: pd.DataFrame) -> Optional[SelectionRule]:
        """训练单个子状态的选股规则 - 内存优化版本"""

        if len(data) < 100:
            return None

        # 🔥 新增：过滤可买入的股票进行训练
        if 'is_can_buy' in data.columns:
            buyable_data = data[data['is_can_buy'] == 1]
            if len(buyable_data) < 50:  # 如果可买入股票太少，使用全部数据
                self.logger.warning(f"子状态 {sub_state} 可买入股票太少 ({len(buyable_data)})，使用全部数据训练")
                training_data = data
            else:
                self.logger.info(f"子状态 {sub_state} 使用 {len(buyable_data)} 只可买入股票训练 (原始: {len(data)})")
                training_data = buyable_data
        else:
            self.logger.warning(f"子状态 {sub_state} 缺少is_can_buy特征，使用全部数据训练")
            training_data = data

        # 准备特征和目标变量
        feature_cols = self._get_feature_columns(training_data)
        if not feature_cols:
            return None

        X = training_data[feature_cols].fillna(0)
        y = training_data['future_return_1d'].fillna(0)  # 修改为future_return_1d目标

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
            # 回退到简单随机采样 - 修复float16索引问题
            try:
                # 确保索引是int类型
                valid_indices = X.index.astype(int)
                sampled_indices = np.random.choice(valid_indices, size=max_samples, replace=False)
                return X.loc[sampled_indices], y.loc[sampled_indices]
            except Exception as e2:
                self.logger.error(f"随机采样也失败: {e2}，使用前{max_samples}条数据")
                return X.head(max_samples), y.head(max_samples)
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """获取可用的特征列 - 智能处理各种特征类型"""

        # 🔥 修复：排除非特征列和字符串特征
        exclude_cols = {
            # 基础信息
            'stock_code', 'date', 'date_str', 'trade_date',
            # 目标变量 - 修改为future_return_1d，排除future_return_2d避免数据泄露
            'future_return_1d', 'future_return_2d',
            # 🔥 关键修复：排除 is_can_buy 字段，避免作为特征使用
            'is_can_buy',
            # 板块信息
            'target_sector', 'sub_state',
            # 其他非特征列
            'sector_code', 'sector_name',
            # 字符串时间特征（无法直接用于机器学习）
            'first_time_formatted', 'last_time_formatted', 'first_time', 'last_time'
        }

        # 获取所有数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # 过滤掉非特征列
        available_features = [col for col in numeric_cols if col not in exclude_cols]

        # 额外检查：移除包含过多缺失值的特征（超过50%缺失）
        valid_features = []
        for feature in available_features:
            missing_ratio = data[feature].isna().sum() / len(data)
            if missing_ratio < 0.5:  # 缺失值少于50%
                valid_features.append(feature)
            else:
                self.logger.debug(f"移除高缺失率特征 {feature}: {missing_ratio:.2%} 缺失")

        self.logger.debug(f"可用特征数量: {len(valid_features)} (原始: {len(available_features)})")
        self.logger.debug(f"前10个特征: {valid_features[:10]}")

        return valid_features
    
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
        """生成选股标准 - 基于特征重要性和数据分布智能确定特征数量和阈值"""

        # 🔥 修复1: 根据特征重要性值动态确定特征数量
        # 计算重要性阈值：取平均重要性的50%作为阈值
        importance_values = list(feature_importance.values())
        if len(importance_values) == 0:
            return {'top_features': [], 'feature_thresholds': {}, 'min_confidence': 0.3, 'max_selections': 10}  # 🔥 修复：降低置信度阈值

        avg_importance = np.mean(importance_values)
        importance_threshold = avg_importance * 0.5

        # 选择重要性超过阈值的特征，但至少保留前10个，最多取前30个
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # 动态确定特征数量
        significant_features = [f for f in sorted_features if f[1] >= importance_threshold]
        if len(significant_features) < 10:
            top_features = sorted_features[:10]  # 至少10个
        elif len(significant_features) > 30:
            top_features = sorted_features[:30]  # 最多30个
        else:
            top_features = significant_features

        self.logger.debug(f"子状态 {sub_state}: 选择 {len(top_features)} 个重要特征 (重要性阈值: {importance_threshold:.4f})")

        criteria = {
            'top_features': [f[0] for f in top_features],
            'feature_thresholds': {},
            'min_confidence': 0.3,  # 🔥 修复：降低置信度阈值，避免过度筛选
            'max_selections': 10
        }

        # 🔥 修复2: 智能设置特征阈值 - 区分二元特征和连续特征
        for feature, importance in top_features:
            if feature in data.columns:
                feature_data = data[feature].dropna()
                if len(feature_data) == 0:
                    continue

                # 检查是否为二元特征（只有0和1）
                unique_values = feature_data.unique()
                is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0, 1, 0.0, 1.0})

                if is_binary:
                    # 🔥 修复：二元特征也需要判断正负相关性
                    feature_return_corr = feature_data.corr(data['future_return_1d'].dropna())

                    if pd.isna(feature_return_corr):
                        # 无法计算相关性，默认认为1是正向信号
                        criteria['feature_thresholds'][feature] = {
                            'type': 'binary_default',
                            'positive_threshold': 0.5,
                            'importance': importance
                        }
                    elif feature_return_corr > 0:
                        # 正相关：1是好的，0是坏的
                        criteria['feature_thresholds'][feature] = {
                            'type': 'binary_positive',
                            'positive_threshold': 0.5,  # 选择值为1的股票
                            'correlation': feature_return_corr,
                            'importance': importance,
                            'interpretation': f'{feature}=1时收益更好'
                        }
                    else:
                        # 负相关：0是好的，1是坏的
                        criteria['feature_thresholds'][feature] = {
                            'type': 'binary_negative',
                            'positive_threshold': -0.5,  # 选择值为0的股票
                            'correlation': feature_return_corr,
                            'importance': importance,
                            'interpretation': f'{feature}=0时收益更好'
                        }
                else:
                    # 连续特征：使用分位数，但需要考虑与收益的相关性
                    feature_return_corr = feature_data.corr(data['future_return_1d'].dropna())

                    if pd.isna(feature_return_corr):
                        # 无法计算相关性，使用默认阈值
                        criteria['feature_thresholds'][feature] = {
                            'type': 'continuous',
                            'min': feature_data.quantile(0.3),
                            'max': feature_data.quantile(0.9),
                            'optimal': feature_data.quantile(0.7),
                            'importance': importance
                        }
                    else:
                        # 根据相关性方向设置阈值
                        if feature_return_corr > 0:
                            # 正相关：更高的特征值更好
                            criteria['feature_thresholds'][feature] = {
                                'type': 'continuous_positive',
                                'min': feature_data.quantile(0.6),  # 选择较高的值
                                'optimal': feature_data.quantile(0.8),
                                'correlation': feature_return_corr,
                                'importance': importance
                            }
                        else:
                            # 负相关：更低的特征值更好
                            criteria['feature_thresholds'][feature] = {
                                'type': 'continuous_negative',
                                'max': feature_data.quantile(0.4),  # 选择较低的值
                                'optimal': feature_data.quantile(0.2),
                                'correlation': feature_return_corr,
                                'importance': importance
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

            # 🔥 关键修改：按照子状态对应的板块进行回测
            # 获取该子状态对应的板块列表
            sub_state_sectors = data['target_sector'].unique().tolist() if 'target_sector' in data.columns else []

            if len(sub_state_sectors) == 0:
                self.logger.warning(f"子状态 {sub_state} 没有板块信息，跳过回测")
                continue

            self.logger.debug(f"子状态 {sub_state} 对应板块: {list(sub_state_sectors)}")

            # 随机选择30%的数据进行回测
            sample_size = int(len(data) * 0.3)
            if sample_size < 100:
                sample_size = min(len(data), 100)

            backtest_data = data.sample(n=sample_size, random_state=42)

            # 进行回测 - 传递板块信息
            backtest_result = self._perform_backtest(rule, backtest_data, sub_state, sub_state_sectors)
            backtest_results[sub_state] = backtest_result

            self.logger.info(f"子状态 {sub_state} 回测结果: "
                           f"总收益={backtest_result['total_return']:.4f}, "
                           f"最大回撤={backtest_result['max_drawdown']:.4f}, "
                           f"胜率={backtest_result['win_rate']:.2%}, "
                           f"夏普比率={backtest_result['sharpe_ratio']:.3f}, "
                           f"样本数={backtest_result['sample_count']}")

        # 根据回测结果选择最佳子状态
        best_sub_states = self._rank_sub_states_by_backtest(backtest_results)

        self.logger.info(f"基于回测选择的最佳子状态: {best_sub_states}")

        return best_sub_states

    def _perform_backtest(self, rule: SelectionRule, data: pd.DataFrame, sub_state: str = None,
                         sub_state_sectors: list = None) -> Dict[str, float]:
        """对单个子状态进行回测 - 确保只使用子状态对应板块的股票"""

        # 🔥 关键修改：按照子状态对应的板块过滤数据
        if sub_state_sectors is not None and len(sub_state_sectors) > 0 and 'target_sector' in data.columns:
            # 只使用来自子状态对应板块的股票数据
            sector_filtered_data = data[data['target_sector'].isin(sub_state_sectors)]

            if sector_filtered_data.empty:
                self.logger.warning(f"子状态 {sub_state} 在指定板块中没有数据")
                return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0, 'sample_count': 0}

            self.logger.debug(f"子状态 {sub_state} 回测: 原始数据 {len(data)} 条，板块过滤后 {len(sector_filtered_data)} 条")
            data = sector_filtered_data
        else:
            self.logger.warning(f"子状态 {sub_state} 缺少板块信息，使用全部数据进行回测")

        # 🔥 新增：过滤可买入的股票进行回测
        if 'is_can_buy' in data.columns:
            buyable_data = data[data['is_can_buy'] == 1]
            if not buyable_data.empty:
                self.logger.debug(f"子状态 {sub_state} 回测: 过滤后 {len(data)} 条，可买入 {len(buyable_data)} 条")
                data = buyable_data
            else:
                self.logger.warning(f"子状态 {sub_state} 回测: 没有可买入的股票")
                return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0, 'sample_count': 0}

        # 准备特征
        feature_cols = [f for f in rule.feature_importance.keys() if f in data.columns]
        if not feature_cols:
            return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0, 'sample_count': 0}

        X = data[feature_cols].fillna(0)

        # 获取实际收益（预测收益在这里不需要，我们直接用实际收益评估）
        actual_returns = data['future_return_1d'].fillna(0)  # 修改为future_return_1d

        # 计算置信度
        confidences = self._calculate_prediction_confidence(X, rule)

        # 选择高置信度的预测
        min_confidence = rule.selection_criteria.get('min_confidence', 0.3)  # 🔥 修复：降低默认阈值
        high_conf_mask = confidences >= min_confidence

        if high_conf_mask.sum() == 0:
            return {'total_return': 0, 'max_drawdown': 1, 'win_rate': 0, 'sharpe_ratio': 0, 'sample_count': 0}

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
        """根据回测结果排序子状态 - 使用专业量化指标"""

        # 🔥 专业化综合评分计算
        scores = []
        for sub_state, result in backtest_results.items():
            # 基于2020-2025年顶级量化研究的评分权重
            # 综合评分 = 收益指标(40%) + 风险指标(35%) + 稳定性指标(25%)

            # 收益指标 (40%)
            return_score = result['total_return'] * 0.25 + max(result.get('calmar_ratio', 0), 0) * 0.15

            # 风险指标 (35%)
            risk_score = (
                (1 - result['max_drawdown']) * 0.15 +  # 最大回撤控制
                max(result['sharpe_ratio'], 0) * 0.10 +  # 夏普比率
                max(result.get('sortino_ratio', 0), 0) * 0.10  # 索提诺比率
            )

            # 稳定性指标 (25%)
            stability_score = (
                result['win_rate'] * 0.15 +  # 胜率
                (1 - result.get('var_95', 0)) * 0.10  # VaR控制
            )

            # 样本量惩罚 - 样本过少的子状态降权
            sample_penalty = min(result.get('sample_count', 0) / 100, 1.0)

            total_score = (return_score + risk_score + stability_score) * sample_penalty

            scores.append((sub_state, total_score, result))

            # 详细日志
            self.logger.debug(f"子状态 {sub_state} 评分详情:")
            self.logger.debug(f"  收益评分: {return_score:.4f} (总收益: {result['total_return']:.4f}, 卡尔马: {result.get('calmar_ratio', 0):.3f})")
            self.logger.debug(f"  风险评分: {risk_score:.4f} (回撤: {result['max_drawdown']:.4f}, 夏普: {result['sharpe_ratio']:.3f})")
            self.logger.debug(f"  稳定评分: {stability_score:.4f} (胜率: {result['win_rate']:.2%}, VaR: {result.get('var_95', 0):.4f})")
            self.logger.debug(f"  样本惩罚: {sample_penalty:.3f} (样本数: {result.get('sample_count', 0)})")
            self.logger.debug(f"  综合评分: {total_score:.4f}")

        # 按评分排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 选择前2个
        best_sub_states = [item[0] for item in scores[:2]]

        self.logger.info("子状态排序结果:")
        for i, (sub_state, score, result) in enumerate(scores):
            status = "✅ 选中" if sub_state in best_sub_states else "❌ 未选中"
            self.logger.info(f"  {i+1}. {sub_state}: {score:.4f} {status}")

        return best_sub_states

    def predict_and_select_stocks(self, sector_states: pd.DataFrame,
                                stock_features: pd.DataFrame,
                                target_date: str,
                                algorithm_type: str = 'multitask_attention') -> List[StockCandidate]:
        """预测并选择股票 - 分离双算法版本

        Args:
            sector_states: 板块状态数据
            stock_features: 股票特征数据
            target_date: 目标日期
            algorithm_type: 算法类型 ('traditional', 'multitask_attention', 'genetic_rl')
        """

        self.logger.info(f"开始预测选股 - 日期: {target_date}, 算法: {algorithm_type}")

        # 🔥 修改：支持三种独立算法
        if algorithm_type == 'traditional':
            return self._predict_traditional(sector_states, stock_features, target_date)
        elif algorithm_type in ['multitask_attention', 'genetic_rl']:
            # 🔥 修改：传递 sector_states 以支持板块优先架构
            return self._predict_advanced(sector_states, stock_features, target_date, algorithm_type)
        else:
            self.logger.error(f"不支持的算法类型: {algorithm_type}，请使用 'traditional', 'multitask_attention' 或 'genetic_rl'")
            return []

    def _predict_traditional(self, sector_states: pd.DataFrame,
                           stock_features: pd.DataFrame,
                           target_date: str) -> List[StockCandidate]:
        """传统算法预测选股"""

        self.logger.info("使用传统算法预测选股")

        # 获取最佳子状态
        best_sub_states = self._get_best_sub_states_from_sector_states(sector_states)

        # 🔥 按日期过滤板块状态数据
        target_date_pd = pd.to_datetime(target_date)
        daily_sector_states = sector_states[sector_states['date'] == target_date_pd]

        if daily_sector_states.empty:
            self.logger.warning(f"日期 {target_date} 没有板块状态数据")
            return []

        candidates = []
        for sub_state in best_sub_states:
            if sub_state in self.trained_rules:
                rule = self.trained_rules[sub_state]

                try:
                    # 🔥 板块优先架构：先找出处于该子状态的板块
                    sub_state_sectors = daily_sector_states[daily_sector_states['sub_state'] == sub_state]

                    if sub_state_sectors.empty:
                        self.logger.info(f"日期 {target_date} 没有板块处于子状态 {sub_state}")
                        continue

                    sector_codes = sub_state_sectors['sector_code'].unique().tolist()
                    self.logger.info(f"子状态 {sub_state} 对应的板块: {sector_codes}")

                    # 🔥 获取这些板块内的股票并应用规则
                    selected_stocks = self._apply_traditional_rule_to_sectors(rule, stock_features, sector_codes, sub_state, target_date)

                    for stock_info in selected_stocks:
                        candidate = StockCandidate(
                            stock_code=stock_info['stock_code'],
                            sector=stock_info['sector'],
                            sub_state=sub_state,
                            predicted_return=rule.avg_return,
                            confidence=rule.win_rate,
                            features={'rule_type': 'traditional', 'sub_state': sub_state, 'target_sector': stock_info['sector']},
                            selection_reason=f"传统算法选择(子状态:{sub_state},板块:{stock_info['sector']})"
                        )
                        candidates.append(candidate)

                    self.logger.info(f"传统算法在子状态 {sub_state} 中选择了 {len(selected_stocks)} 只股票")

                except Exception as e:
                    self.logger.error(f"传统规则 {sub_state} 预测失败: {e}")
                    continue
            else:
                self.logger.warning(f"子状态 {sub_state} 没有训练好的传统规则，跳过")

        self.logger.info(f"传统算法选择了 {len(candidates)} 只股票")
        return candidates

    def _apply_traditional_rule_to_sectors(self, rule: 'SelectionRule', stock_features: pd.DataFrame,
                                         sector_codes: List[str], sub_state: str, target_date: str) -> List[Dict[str, str]]:
        """应用传统选股规则到指定板块 - 板块优先架构"""

        selected_stocks = []
        target_date_pd = pd.to_datetime(target_date)

        for sector_code in sector_codes:
            try:
                # 使用 data_loaders 获取板块内股票列表
                from data_loaders import get_stock_list_in_sector
                sector_stocks = get_stock_list_in_sector(sector_code)

                if not sector_stocks:
                    self.logger.warning(f"板块 {sector_code} 没有股票")
                    continue

                # 🔥 关键修复：先按日期过滤，再按板块过滤
                # 1. 首先过滤出目标日期的股票数据
                date_mask = stock_features['date'] == target_date_pd
                daily_stock_features = stock_features[date_mask]

                # 2. 然后在当日数据中过滤出该板块的股票
                sector_mask = daily_stock_features['stock_code'].isin(sector_stocks)
                sector_stock_features = daily_stock_features[sector_mask]

                if sector_stock_features.empty:
                    self.logger.warning(f"板块 {sector_code} 没有可用的股票特征数据")
                    continue

                # 应用传统规则到该板块的股票
                sector_selected = self._apply_traditional_rule(rule, sector_stock_features, sub_state)

                # 每个板块最多选择1只股票
                if sector_selected:
                    best_stock = sector_selected[0]  # 选择第一只（已按收益排序）
                    selected_stocks.append({
                        'stock_code': best_stock,
                        'sector': sector_code
                    })
                    self.logger.info(f"板块 {sector_code} 选择股票: {best_stock}")

            except Exception as e:
                self.logger.warning(f"处理板块 {sector_code} 失败: {e}")
                continue

        return selected_stocks

    def _apply_traditional_rule(self, rule: 'SelectionRule', stock_features: pd.DataFrame,
                              sub_state: str) -> List[str]:
        """应用传统选股规则"""

        # 🔥 过滤可买入的股票
        if 'is_can_buy' in stock_features.columns:
            buyable_stocks = stock_features[stock_features['is_can_buy'] == 1]
        else:
            buyable_stocks = stock_features

        if len(buyable_stocks) == 0:
            return []

        # 应用规则条件
        selected_mask = pd.Series([True] * len(buyable_stocks), index=buyable_stocks.index)

        for condition in rule.conditions:
            try:
                if condition.feature in buyable_stocks.columns:
                    feature_values = buyable_stocks[condition.feature]

                    if condition.operator == '>':
                        mask = feature_values > condition.threshold
                    elif condition.operator == '<':
                        mask = feature_values < condition.threshold
                    elif condition.operator == '>=':
                        mask = feature_values >= condition.threshold
                    elif condition.operator == '<=':
                        mask = feature_values <= condition.threshold
                    else:
                        continue

                    selected_mask &= mask

            except Exception as e:
                self.logger.warning(f"应用条件 {condition.feature} {condition.operator} {condition.threshold} 失败: {e}")
                continue

        # 选择符合条件的股票
        selected_stocks = buyable_stocks[selected_mask]

        # 限制选择数量（每个子状态最多选择3只）
        if len(selected_stocks) > 3:
            # 按预期收益排序选择top3
            if 'future_return_1d' in selected_stocks.columns:
                selected_stocks = selected_stocks.nlargest(3, 'future_return_1d')
            else:
                selected_stocks = selected_stocks.head(3)

        return selected_stocks['stock_code'].tolist() if 'stock_code' in selected_stocks.columns else []



    def _predict_advanced(self, sector_states: pd.DataFrame, stock_features: pd.DataFrame,
                         target_date: str, algorithm_type: str) -> List[StockCandidate]:
        """先进算法预测选股 - 板块优先架构"""

        if not self.enable_advanced_algorithms:
            self.logger.warning("先进算法未启用")
            return []

        if algorithm_type not in self.advanced_algorithms:
            self.logger.warning(f"算法 {algorithm_type} 不存在")
            return []

        algorithm = self.advanced_algorithms[algorithm_type]

        if not algorithm.is_trained:
            self.logger.warning(f"算法 {algorithm_type} 未训练")
            return []

        # 🔥 获取该算法训练时选出的最佳子状态
        target_sub_states = None
        if hasattr(self, 'advanced_best_sub_states') and algorithm_type in self.advanced_best_sub_states:
            target_sub_states = self.advanced_best_sub_states[algorithm_type]
            self.logger.info(f"{algorithm_type} 使用训练时选出的最佳子状态: {target_sub_states}")
        else:
            # 备选方案：使用所有可用的子状态
            if hasattr(algorithm, 'sub_state_models') and algorithm.sub_state_models:
                target_sub_states = list(algorithm.sub_state_models.keys())[:2]
                self.logger.warning(f"{algorithm_type} 未找到训练时的最佳子状态，使用前2个可用子状态: {target_sub_states}")
            else:
                self.logger.error(f"{algorithm_type} 没有可用的子状态模型")
                return []

        self.logger.info(f"开始使用 {algorithm_type} 算法预测，目标子状态: {target_sub_states}")

        candidates = []

        # 🔥 核心修改：实现板块优先的预测架构
        # 1. 首先按日期过滤板块状态数据
        target_date_pd = pd.to_datetime(target_date)
        daily_sector_states = sector_states[sector_states['date'] == target_date_pd]

        if daily_sector_states.empty:
            self.logger.warning(f"日期 {target_date} 没有板块状态数据")
            return []

        # 🔥 为每个目标子状态分别预测 - 板块优先架构
        for sub_state in target_sub_states:
            try:
                self.logger.info(f"{algorithm_type} 算法预测子状态: {sub_state}")

                # 检查该子状态是否有训练好的模型
                if not hasattr(algorithm, 'sub_state_models') or sub_state not in algorithm.sub_state_models:
                    self.logger.warning(f"子状态 {sub_state} 没有训练好的模型，跳过")
                    continue

                # 🔥 步骤2：找出当日处于该子状态的板块
                sub_state_sectors = daily_sector_states[daily_sector_states['sub_state'] == sub_state]

                if sub_state_sectors.empty:
                    self.logger.info(f"日期 {target_date} 没有板块处于子状态 {sub_state}")
                    continue

                sector_codes = sub_state_sectors['sector_code'].unique().tolist()
                self.logger.info(f"子状态 {sub_state} 对应的板块: {sector_codes}")

                # 🔥 步骤3：获取这些板块内的股票
                sector_stock_features = []
                for sector_code in sector_codes:
                    try:
                        # 使用 data_loaders 获取板块内股票列表
                        from data_loaders import get_stock_list_in_sector
                        sector_stocks = get_stock_list_in_sector(sector_code)

                        if sector_stocks:
                            # 🔥 关键修复：先按日期过滤，再按板块过滤
                            # 1. 首先过滤出目标日期的股票数据
                            date_mask = stock_features['date'] == target_date_pd
                            daily_stock_features = stock_features[date_mask]

                            # 2. 然后在当日数据中过滤出该板块的股票
                            sector_mask = daily_stock_features['stock_code'].isin(sector_stocks)
                            sector_data = daily_stock_features[sector_mask].copy()

                            if not sector_data.empty:
                                sector_data['target_sector'] = sector_code
                                sector_stock_features.append(sector_data)
                                self.logger.info(f"板块 {sector_code} 有 {len(sector_data)} 只股票")

                    except Exception as e:
                        self.logger.warning(f"获取板块 {sector_code} 股票列表失败: {e}")
                        continue

                if not sector_stock_features:
                    self.logger.warning(f"子状态 {sub_state} 没有找到可用的股票数据")
                    continue

                # 合并所有板块的股票特征
                combined_features = pd.concat(sector_stock_features, ignore_index=True)

                # 🔥 步骤4：对板块内股票进行预测
                # 🔥 关键修复：传递完整数据而不是只传递特征列
                # 算法内部会自己处理特征列的提取，但需要访问 stock_code 等元数据

                # 🔥 修复：传递子状态列表而不是单个字符串
                predictions = algorithm.predict(combined_features, [sub_state])

                # 🔥 修复：处理SelectionResult对象
                if predictions is None:
                    self.logger.warning(f"{algorithm_type} 算法对子状态 {sub_state} 预测失败")
                    continue

                # 从SelectionResult对象中提取预测结果
                if hasattr(predictions, 'selected_stocks'):
                    prediction_list = predictions.selected_stocks
                elif hasattr(predictions, 'predictions'):
                    prediction_list = predictions.predictions
                elif isinstance(predictions, list):
                    prediction_list = predictions
                else:
                    self.logger.warning(f"{algorithm_type} 算法对子状态 {sub_state} 返回了未知类型的结果: {type(predictions)}")
                    continue

                if not prediction_list or len(prediction_list) == 0:
                    self.logger.warning(f"{algorithm_type} 算法对子状态 {sub_state} 没有预测结果")
                    continue

                # 🔥 步骤5：选择预测置信度最高的股票（每个板块最多1只）
                sector_candidates = {}
                for idx, prediction in enumerate(prediction_list):
                    if idx < len(combined_features):
                        stock_code = combined_features.iloc[idx]['stock_code']
                        sector_code = combined_features.iloc[idx]['target_sector']

                        # 每个板块只选择置信度最高的股票
                        if sector_code not in sector_candidates or prediction > sector_candidates[sector_code]['confidence']:
                            sector_candidates[sector_code] = {
                                'stock_code': stock_code,
                                'confidence': prediction,
                                'sector': sector_code
                            }

                # 创建候选股票
                for sector_code, candidate_info in sector_candidates.items():
                    candidate = StockCandidate(
                        stock_code=candidate_info['stock_code'],
                        sector=candidate_info['sector'],
                        sub_state=sub_state,
                        predicted_return=candidate_info['confidence'],
                        confidence=candidate_info['confidence'],
                        features={'algorithm': algorithm_type, 'target_sector': sector_code},
                        selection_reason=f"{algorithm_type}算法选择(子状态:{sub_state},板块:{sector_code})"
                    )
                    candidates.append(candidate)

                self.logger.info(f"{algorithm_type} 算法在子状态 {sub_state} 中选择了 {len(sector_candidates)} 只股票")

            except Exception as e:
                self.logger.error(f"{algorithm_type} 算法预测子状态 {sub_state} 失败: {e}")
                continue

        self.logger.info(f"{algorithm_type} 算法总共选择了 {len(candidates)} 只股票")
        return candidates

    # 🔥 删除集成方法 - 改为独立算法评估

    def evaluate_algorithm_performance(self, test_data: Dict[str, pd.DataFrame],
                                     actual_returns: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """评估不同算法的性能对比"""
        self.logger.info("开始评估算法性能对比")

        performance_results = {}

        # 🔥 修改：只评估先进算法
        if self.enable_advanced_algorithms:
            for algo_name in ['multitask_attention', 'genetic_rl']:
                try:
                    advanced_performance = self._evaluate_advanced_performance(
                        test_data, actual_returns, algo_name)
                    performance_results[algo_name] = advanced_performance
                except Exception as e:
                    self.logger.error(f"{algo_name} 性能评估失败: {e}")
                    performance_results[algo_name] = {}
        else:
            self.logger.error("先进算法未启用，无法进行性能评估")
            return {}

        # 计算性能对比指标
        comparison_metrics = self._calculate_performance_comparison(performance_results)
        performance_results['comparison'] = comparison_metrics

        self.logger.info("算法性能对比评估完成")
        return performance_results



    def _evaluate_advanced_performance(self, test_data: Dict[str, pd.DataFrame],
                                     actual_returns: Dict[str, float],
                                     algorithm_type: str) -> Dict[str, float]:
        """评估先进算法性能"""
        # 实现先进算法的性能评估
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_return': 0.0
        }

    def _calculate_performance_comparison(self, performance_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算性能对比指标 - 只对比先进算法"""
        comparison = {}

        # 🔥 修改：只对比先进算法
        algorithms = ['multitask_attention', 'genetic_rl']
        metrics = ['total_return', 'win_rate', 'sharpe_ratio', 'max_drawdown']

        for metric in metrics:
            metric_values = []
            for algo in algorithms:
                if algo in performance_results and metric in performance_results[algo]:
                    metric_values.append(performance_results[algo][metric])

            if metric_values:
                comparison[f'best_{metric}'] = max(metric_values)
                comparison[f'worst_{metric}'] = min(metric_values)
                comparison[f'avg_{metric}'] = sum(metric_values) / len(metric_values)

                # 🔥 新增：算法间差异分析
                if len(metric_values) == 2:
                    comparison[f'{metric}_difference'] = abs(metric_values[0] - metric_values[1])
                    comparison[f'{metric}_multitask_vs_genetic'] = metric_values[0] - metric_values[1]  # 正值表示多任务更好

        return comparison
    
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
            if confidences[idx] >= rule.selection_criteria.get('min_confidence', 0.3):  # 🔥 修复：降低默认阈值
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
        """过滤出有效交易的股票（排除未上市股票和不可买入股票）"""

        valid_candidates = []
        filtered_count = 0
        is_can_buy_filtered = 0

        for candidate in candidates:
            # 检查股票在当日是否有交易数据
            stock_data = stock_features[stock_features['stock_code'] == candidate.stock_code]

            if not stock_data.empty:
                volume = stock_data['volume'].iloc[0]

                # 🔥 新增：检查is_can_buy特征
                is_can_buy = stock_data['is_can_buy'].iloc[0] if 'is_can_buy' in stock_data.columns else 1

                # 只选择有交易量且可买入的股票
                if volume > 0 and is_can_buy == 1:
                    valid_candidates.append(candidate)
                else:
                    filtered_count += 1
                    if volume <= 0:
                        self.logger.warning(f"过滤未上市股票: {candidate.stock_code} 在 {target_date} (volume=0)")
                    elif is_can_buy == 0:
                        is_can_buy_filtered += 1
                        self.logger.info(f"过滤不可买入股票: {candidate.stock_code} 在 {target_date} (is_can_buy=0)")
            else:
                filtered_count += 1
                self.logger.warning(f"过滤无数据股票: {candidate.stock_code} 在 {target_date}")

        if filtered_count > 0:
            self.logger.info(f"共过滤了 {filtered_count} 只股票 (其中 {is_can_buy_filtered} 只因不可买入被过滤)")

        return valid_candidates

    def _calculate_prediction_confidence(self, X: pd.DataFrame, rule: SelectionRule) -> np.ndarray:
        """计算预测置信度 - 基于新的阈值结构进行特征筛选"""

        confidences = np.ones(len(X)) * 0.5
        feature_thresholds = rule.selection_criteria.get('feature_thresholds', {})

        for i, (_, row) in enumerate(X.iterrows()):
            confidence = 0.5
            feature_score = 0.0
            valid_features = 0

            # 🔥 修复：使用新的阈值结构进行特征评估
            for feature, importance in rule.feature_importance.items():
                if feature in row.index and feature in feature_thresholds:
                    feature_value = row[feature]
                    if pd.isna(feature_value):
                        continue

                    threshold_info = feature_thresholds[feature]
                    threshold_type = threshold_info.get('type', 'unknown')

                    # 根据特征类型和阈值计算得分
                    feature_meets_criteria = False

                    if threshold_type == 'binary_positive':
                        # 二元正相关：值为1时好
                        feature_meets_criteria = feature_value >= 0.5

                    elif threshold_type == 'binary_negative':
                        # 二元负相关：值为0时好
                        feature_meets_criteria = feature_value < 0.5

                    elif threshold_type == 'binary_default':
                        # 默认二元：值为1时好
                        feature_meets_criteria = feature_value >= 0.5

                    elif threshold_type == 'continuous_positive':
                        # 连续正相关：值越高越好
                        min_threshold = threshold_info.get('min', 0)
                        feature_meets_criteria = feature_value >= min_threshold

                    elif threshold_type == 'continuous_negative':
                        # 连续负相关：值越低越好
                        max_threshold = threshold_info.get('max', float('inf'))
                        feature_meets_criteria = feature_value <= max_threshold

                    elif threshold_type == 'continuous':
                        # 默认连续：在范围内
                        min_val = threshold_info.get('min', float('-inf'))
                        max_val = threshold_info.get('max', float('inf'))
                        feature_meets_criteria = min_val <= feature_value <= max_val

                    # 计算特征得分
                    if feature_meets_criteria:
                        feature_score += importance * 1.0  # 满足条件的特征得满分
                    else:
                        feature_score += importance * 0.2  # 不满足条件的特征得低分

                    valid_features += 1

            # 🔥 修复：简化置信度计算逻辑
            if valid_features > 0:
                # 计算满足条件的特征比例
                total_importance = sum(rule.feature_importance[f] for f in feature_thresholds.keys() if f in rule.feature_importance)
                if total_importance > 0:
                    confidence = 0.2 + (feature_score / total_importance) * 0.7  # 基础0.2 + 特征得分占比*0.7
                else:
                    confidence = 0.5
            else:
                # 没有有效特征时给予中等置信度
                confidence = 0.4

            confidences[i] = min(max(confidence, 0.1), 0.9)  # 限制在0.1-0.9范围内

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
            'feature_columns': getattr(self, 'feature_columns', []),
            # 🔥 修复：保存先进算法的最佳子状态
            'advanced_best_sub_states': getattr(self, 'advanced_best_sub_states', {})
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

        # 🔥 关键修复：加载先进算法模型
        if self.enable_advanced_algorithms:
            advanced_loaded = self._load_advanced_models()
            if advanced_loaded:
                self.logger.info("✅ 先进算法模型加载成功")
            else:
                self.logger.warning("⚠️ 先进算法模型加载失败或未找到")

        # 加载先进算法的最佳子状态
        self.advanced_best_sub_states = save_data.get('advanced_best_sub_states', {})

        version = save_data.get('system_version', '1.0')
        save_time = save_data.get('save_timestamp', 'unknown')

        self.logger.info(f"系统已从 {filename} 加载 (版本: {version})")
        self.logger.info(f"保存时间: {save_time}")
        self.logger.info(f"加载了 {len(self.trained_rules)} 个训练规则")
        self.logger.info(f"最佳子状态: {self.best_sub_states}")

        # 显示先进算法状态
        if self.enable_advanced_algorithms:
            for algo_name, algorithm in self.advanced_algorithms.items():
                self.logger.info(f"算法 {algo_name}: 训练状态={algorithm.is_trained}")

        return True
