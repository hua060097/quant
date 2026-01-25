#!/usr/bin/env python3
"""
遗传编程因子进化模块

使用gplearn库通过遗传编程进化出新的超级因子组合
基于现有因子构建更强的预测因子
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time
import warnings
import multiprocessing as mp
from datetime import datetime

# 抑制gplearn的警告
warnings.filterwarnings('ignore', category=UserWarning, module='gplearn')
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from gplearn.genetic import SymbolicRegressor
    from gplearn.fitness import make_fitness
    from gplearn.functions import make_function
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    logging.warning("gplearn未安装，遗传编程功能不可用")

class GeneticFactorEvolution:
    """遗传编程因子进化器"""
    
    def __init__(self, output_dir: str = "./results"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not GPLEARN_AVAILABLE:
            raise ImportError("请先安装gplearn: pip install gplearn")
        
        # 遗传编程参数
        self.population_size = 1000
        self.generations = 20
        self.max_samples = 0.8  # 使用80%的样本进行训练
        self.n_jobs = min(mp.cpu_count(), 8)  # 限制最大并行数
        
        # 风险控制参数
        self.max_drawdown_threshold = 0.15  # 最大回撤阈值15%
        self.min_sharpe_threshold = 0.5     # 最小夏普比率阈值
        
        # 结果存储
        self.best_factors = []
        self.evolution_history = []
        
    def _prepare_data(self, factor_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备遗传编程所需的数据
        
        Args:
            factor_data: 因子数据
            
        Returns:
            X: 特征矩阵
            y: 目标变量（未来收益率）
            feature_names: 特征名称列表
        """
        self.logger.info("开始准备遗传编程数据...")
        
        # 定义需要的基础因子
        required_factors = [
            # 技术因子
            'rsi6', 'rsi12', 'rsi14', 'macd', 'volatility20',
            'return_5d', 'return_20d', 'ma5', 'ma10', 'ma20', 'ma60',
            
            # 动量因子
            'momentum_5d', 'momentum_20d', 'momentum_60d',
            'price_momentum', 'volume_momentum',
            
            # 成交量因子
            'volume_ratio_5d', 'volume_ratio_20d', 'turnover_rate',
            'volume_price_trend', 'volume_ma_ratio',
            
            # 价格位置因子
            'price_position_5d', 'price_position_20d', 'price_position_60d',
            
            # 基本面因子（如果有）
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa',
            
            # 涨停质量因子
            'is_limit_up', 'limit_strength', 'limit_up_days_5d',
            'consecutive_limit_up', 'limit_up_volume_ratio'
        ]
        
        # 筛选可用的因子
        available_factors = [f for f in required_factors if f in factor_data.columns]
        self.logger.info(f"可用因子: {len(available_factors)}/{len(required_factors)}")
        
        if len(available_factors) < 5:
            raise ValueError(f"可用因子数量不足，至少需要5个，当前只有{len(available_factors)}个")
        
        # 准备特征矩阵
        X = factor_data[available_factors].values
        
        # 准备目标变量（使用future_return_2d或计算T+2日收益率）
        if 'future_return_2d' in factor_data.columns:
            y = factor_data['future_return_2d'].values
        else:
            # 如果没有future_return_2d，计算T+2日的收益率
            factor_data_sorted = factor_data.sort_values(['stock_code', 'date'])
            y = factor_data_sorted.groupby('stock_code')['pct_chg'].shift(-2).values
        
        # 数据清洗
        self.logger.info("开始数据清洗...")
        
        # 处理无穷大值
        X = np.where(np.isinf(X), np.nan, X)
        y = np.where(np.isinf(y), np.nan, y)
        
        # 处理异常值（超过5个标准差的值）
        for i in range(X.shape[1]):
            col_data = X[:, i]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.sum() > 0:
                mean_val = np.mean(col_data[valid_mask])
                std_val = np.std(col_data[valid_mask])
                outlier_mask = np.abs(col_data - mean_val) > 5 * std_val
                X[outlier_mask, i] = np.nan
        
        # 处理缺失值 - 使用中位数填充
        for i in range(X.shape[1]):
            col_data = X[:, i]
            valid_mask = ~np.isnan(col_data)
            if valid_mask.sum() > 0:
                median_val = np.median(col_data[valid_mask])
                X[np.isnan(col_data), i] = median_val
            else:
                X[:, i] = 0  # 如果整列都是NaN，填充为0
        
        # 处理目标变量的缺失值
        y_valid_mask = ~np.isnan(y)
        if y_valid_mask.sum() == 0:
            raise ValueError("目标变量全部为NaN")
        
        # 只保留目标变量有效的样本
        X = X[y_valid_mask]
        y = y[y_valid_mask]
        
        # 数据标准化（Z-score标准化）
        self.logger.info("开始数据标准化...")
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.where(X_std == 0, 1, X_std)  # 避免除零
        X = (X - X_mean) / X_std
        
        # 限制数据范围，避免极端值
        X = np.clip(X, -5, 5)
        y = np.clip(y, -0.2, 0.2)  # 限制收益率在-20%到20%之间
        
        self.logger.info(f"数据准备完成: X形状{X.shape}, y形状{y.shape}")
        self.logger.info(f"目标变量(T+2日收益率)统计: 均值{np.mean(y):.4f}, 标准差{np.std(y):.4f}")
        
        return X, y, available_factors
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or np.std(returns) == 0:
            return -10.0
        
        # 假设无风险利率为3%年化
        risk_free_rate = 0.03 / 252  # 日化无风险利率
        excess_returns = returns - risk_free_rate
        
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 1.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _custom_fitness_function(self, y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
        """
        自定义适应度函数：基于多空组合的夏普比率，带有否决门槛

        Args:
            y_true: 真实收益率（T+2日收益率）
            y_pred: 预测的因子值
            w: 样本权重

        Returns:
            适应度分数（越高越好）
        """
        try:
            # 数据验证
            if len(y_true) == 0 or len(y_pred) == 0:
                return -100.0
            
            # 处理异常值
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            if valid_mask.sum() < 10:  # 至少需要10个有效样本
                return -100.0
            
            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]
            
            # 构建多空组合：根据因子值排序，做多前20%，做空后20%
            n_samples = len(y_pred_clean)
            n_long = max(1, int(n_samples * 0.2))
            n_short = max(1, int(n_samples * 0.2))
            
            # 按因子值排序
            sorted_indices = np.argsort(y_pred_clean)
            
            # 选择多空股票
            short_indices = sorted_indices[:n_short]  # 因子值最小的做空
            long_indices = sorted_indices[-n_long:]   # 因子值最大的做多
            
            # 计算多空组合收益
            long_returns = y_true_clean[long_indices]
            short_returns = y_true_clean[short_indices]
            
            # 多空组合收益 = 做多收益 - 做空收益
            portfolio_returns = np.concatenate([long_returns, -short_returns])
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # 否决门槛：最大回撤超过15%时给予极低分数
            if max_drawdown > self.max_drawdown_threshold:
                return -100.0
            
            # 否决门槛：夏普比率过低时给予极低分数
            if sharpe_ratio < self.min_sharpe_threshold:
                return -50.0
            
            # 综合评分：主要基于夏普比率，同时考虑回撤控制
            fitness_score = sharpe_ratio * (1 - max_drawdown * 0.5)
            
            return fitness_score
            
        except Exception as e:
            self.logger.debug(f"适应度函数计算错误: {e}")
            return -100.0
    
    def _create_custom_functions(self) -> List:
        """创建自定义数学函数"""
        custom_functions = []
        
        # 保护性对数函数
        def _protected_log(x1):
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)
        
        # 保护性平方根函数
        def _protected_sqrt(x1):
            with np.errstate(invalid='ignore'):
                return np.where(x1 > 0, np.sqrt(x1), 0.)
        
        # 保护性指数函数
        def _protected_exp(x1):
            with np.errstate(over='ignore'):
                return np.where(np.abs(x1) < 10, np.exp(x1), 0.)
        
        # 逻辑函数：if x1 > x2 then x3 else x4
        def _logical(x1, x2, x3, x4):
            return np.where(x1 > x2, x3, x4)
        
        # 最大值函数
        def _max_func(x1, x2):
            return np.maximum(x1, x2)
        
        # 最小值函数
        def _min_func(x1, x2):
            return np.minimum(x1, x2)
        
        # 创建gplearn函数
        try:
            custom_functions.extend([
                make_function(function=_protected_log, name='log', arity=1),
                make_function(function=_protected_sqrt, name='sqrt', arity=1),
                make_function(function=_protected_exp, name='exp', arity=1),
                make_function(function=_logical, name='if', arity=4),
                make_function(function=_max_func, name='max', arity=2),
                make_function(function=_min_func, name='min', arity=2),
            ])
        except Exception as e:
            self.logger.warning(f"创建自定义函数失败: {e}")
        
        return custom_functions

    def evolve_factors(self, factor_data: pd.DataFrame, n_best_factors: int = 10) -> Dict[str, Any]:
        """
        进化生成超级因子

        Args:
            factor_data: 因子数据
            n_best_factors: 返回最佳因子的数量

        Returns:
            包含最佳因子信息的字典
        """
        self.logger.info("=" * 80)
        self.logger.info("开始遗传编程因子进化")
        self.logger.info("=" * 80)

        start_time = time.time()

        # 准备数据
        X, y, feature_names = self._prepare_data(factor_data)

        # 创建自定义适应度函数
        self.logger.info("创建自定义适应度函数...")

        # 创建一个闭包来访问self的方法
        def fitness_wrapper(y_true, y_pred, w):
            return self._custom_fitness_function(y_true, y_pred, w)

        custom_fitness = make_fitness(
            function=fitness_wrapper,
            greater_is_better=True,
            wrap=True
        )

        # 创建自定义函数
        custom_functions = self._create_custom_functions()

        # 定义函数集合
        function_set = ['add', 'sub', 'mul', 'div'] + custom_functions

        self.logger.info(f"遗传编程参数:")
        self.logger.info(f"  种群大小: {self.population_size}")
        self.logger.info(f"  进化代数: {self.generations}")
        self.logger.info(f"  并行进程: {self.n_jobs}")
        self.logger.info(f"  采样比例: {self.max_samples}")
        self.logger.info(f"  函数集合: {len(function_set)} 个函数")
        self.logger.info(f"  特征数量: {len(feature_names)}")
        self.logger.info(f"  训练样本: {len(X)}")

        # 创建遗传编程模型
        gp = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=self.max_samples,
            verbose=1,
            parsimony_coefficient=0.01,
            random_state=42,
            n_jobs=self.n_jobs,
            function_set=function_set,
            metric=custom_fitness,
            const_range=(-1.0, 1.0),
            init_depth=(2, 6),
            init_method='half and half',
            tournament_size=20,
            feature_names=feature_names
        )

        # 开始进化
        self.logger.info("开始遗传编程进化...")
        try:
            gp.fit(X, y)

            evolution_time = time.time() - start_time
            self.logger.info(f"进化完成，耗时: {evolution_time:.2f}秒")

            # 分析结果
            results = self._analyze_results(gp, X, y, feature_names, n_best_factors)

            # 保存结果
            self._save_results(results)

            return results

        except Exception as e:
            self.logger.error(f"遗传编程进化失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            raise

    def _analyze_results(self, gp: SymbolicRegressor, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str], n_best: int) -> Dict[str, Any]:
        """分析进化结果"""
        self.logger.info("分析进化结果...")

        results = {
            'best_program': None,
            'best_factors': [],
            'evolution_stats': {},
            'feature_names': feature_names,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # 获取最佳程序
            best_program = gp._program
            results['best_program'] = {
                'formula': str(best_program),
                'fitness': float(best_program.fitness_),
                'raw_fitness': float(best_program.raw_fitness_),
                'depth': int(best_program.depth_),
                'length': int(best_program.length_)
            }

            self.logger.info(f"最佳因子公式: {best_program}")
            self.logger.info(f"最佳适应度: {best_program.fitness_:.4f}")
            self.logger.info(f"原始适应度: {best_program.raw_fitness_:.4f}")

            # 计算最佳因子的详细指标
            y_pred = gp.predict(X)
            detailed_metrics = self._calculate_detailed_metrics(y, y_pred)
            results['best_program'].update(detailed_metrics)

            # 获取进化历史中的优秀程序
            if hasattr(gp, '_programs') and gp._programs:
                self.logger.info("分析进化历史中的优秀程序...")

                all_programs = []
                for generation_idx, generation in enumerate(gp._programs):
                    for program_idx, program in enumerate(generation):
                        if program is not None and hasattr(program, 'fitness_'):
                            try:
                                program_pred = program.execute(X)
                                if not np.any(np.isnan(program_pred)) and not np.any(np.isinf(program_pred)):
                                    metrics = self._calculate_detailed_metrics(y, program_pred)

                                    all_programs.append({
                                        'formula': str(program),
                                        'fitness': float(program.fitness_),
                                        'raw_fitness': float(program.raw_fitness_),
                                        'depth': int(program.depth_),
                                        'length': int(program.length_),
                                        'generation': generation_idx,
                                        'program_idx': program_idx,
                                        **metrics
                                    })
                            except Exception as e:
                                self.logger.debug(f"程序执行失败: {e}")
                                continue

                # 按适应度排序，选择最佳的n个
                all_programs.sort(key=lambda x: x['fitness'], reverse=True)
                results['best_factors'] = all_programs[:n_best]

                self.logger.info(f"找到 {len(all_programs)} 个有效程序，选择前 {n_best} 个")

            # 进化统计信息
            results['evolution_stats'] = {
                'total_generations': self.generations,
                'population_size': self.population_size,
                'final_fitness': float(best_program.fitness_),
                'convergence_generation': self._find_convergence_generation(gp),
                'diversity_score': self._calculate_diversity_score(gp)
            }

        except Exception as e:
            self.logger.error(f"结果分析失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")

        return results

    def _calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算详细的性能指标"""
        try:
            # 数据清洗
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
            if valid_mask.sum() < 10:
                return {'sharpe_ratio': -10.0, 'max_drawdown': 1.0, 'annual_return': -1.0, 'win_rate': 0.0}

            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]

            # 构建多空组合
            n_samples = len(y_pred_clean)
            n_long = max(1, int(n_samples * 0.2))
            n_short = max(1, int(n_samples * 0.2))

            sorted_indices = np.argsort(y_pred_clean)
            short_indices = sorted_indices[:n_short]
            long_indices = sorted_indices[-n_long:]

            long_returns = y_true_clean[long_indices]
            short_returns = y_true_clean[short_indices]
            portfolio_returns = np.concatenate([long_returns, -short_returns])

            # 计算各项指标
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            annual_return = np.mean(portfolio_returns) * 252
            win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns)

            # 计算信息比率（IC）
            ic = np.corrcoef(y_true_clean, y_pred_clean)[0, 1] if len(y_true_clean) > 1 else 0.0

            return {
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'annual_return': float(annual_return),
                'win_rate': float(win_rate),
                'information_coefficient': float(ic) if not np.isnan(ic) else 0.0
            }

        except Exception as e:
            self.logger.debug(f"指标计算失败: {e}")
            return {'sharpe_ratio': -10.0, 'max_drawdown': 1.0, 'annual_return': -1.0, 'win_rate': 0.0}

    def _find_convergence_generation(self, gp: SymbolicRegressor) -> int:
        """找到收敛的代数"""
        try:
            if not hasattr(gp, '_programs') or not gp._programs:
                return self.generations

            best_fitness_history = []
            for generation in gp._programs:
                generation_fitness = []
                for program in generation:
                    if program is not None and hasattr(program, 'fitness_'):
                        generation_fitness.append(program.fitness_)

                if generation_fitness:
                    best_fitness_history.append(max(generation_fitness))

            # 找到适应度不再显著提升的代数
            if len(best_fitness_history) < 3:
                return len(best_fitness_history)

            for i in range(2, len(best_fitness_history)):
                recent_improvement = best_fitness_history[i] - best_fitness_history[i-2]
                if recent_improvement < 0.001:  # 改进小于0.001认为收敛
                    return i

            return len(best_fitness_history)

        except Exception:
            return self.generations

    def _calculate_diversity_score(self, gp: SymbolicRegressor) -> float:
        """计算种群多样性分数"""
        try:
            if not hasattr(gp, '_programs') or not gp._programs or not gp._programs[-1]:
                return 0.0

            final_generation = gp._programs[-1]
            unique_formulas = set()

            for program in final_generation:
                if program is not None:
                    unique_formulas.add(str(program))

            total_programs = sum(1 for p in final_generation if p is not None)
            if total_programs == 0:
                return 0.0

            return len(unique_formulas) / total_programs

        except Exception:
            return 0.0

    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存进化结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存详细结果
            import json
            results_file = self.output_dir / f"genetic_factors_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"详细结果已保存: {results_file}")

            # 保存简化的因子公式
            formulas_file = self.output_dir / f"best_factor_formulas_{timestamp}.txt"
            with open(formulas_file, 'w', encoding='utf-8') as f:
                f.write("遗传编程进化的最佳因子公式\n")
                f.write("=" * 50 + "\n\n")

                if results['best_program']:
                    f.write("最佳因子:\n")
                    f.write(f"公式: {results['best_program']['formula']}\n")
                    f.write(f"夏普比率: {results['best_program'].get('sharpe_ratio', 'N/A'):.4f}\n")
                    f.write(f"最大回撤: {results['best_program'].get('max_drawdown', 'N/A'):.4f}\n")
                    f.write(f"年化收益: {results['best_program'].get('annual_return', 'N/A'):.4f}\n")
                    f.write(f"胜率: {results['best_program'].get('win_rate', 'N/A'):.4f}\n\n")

                f.write("前10个最佳因子:\n")
                f.write("-" * 30 + "\n")

                for i, factor in enumerate(results['best_factors'][:10], 1):
                    f.write(f"\n{i}. 夏普比率: {factor.get('sharpe_ratio', 'N/A'):.4f}\n")
                    f.write(f"   公式: {factor['formula']}\n")
                    f.write(f"   最大回撤: {factor.get('max_drawdown', 'N/A'):.4f}\n")
                    f.write(f"   年化收益: {factor.get('annual_return', 'N/A'):.4f}\n")
                    f.write(f"   胜率: {factor.get('win_rate', 'N/A'):.4f}\n")

            self.logger.info(f"因子公式已保存: {formulas_file}")

        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")

    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """打印结果摘要"""
        print("\n" + "=" * 80)
        print("遗传编程因子进化结果摘要")
        print("=" * 80)

        if results['best_program']:
            best = results['best_program']
            print(f"\n🏆 最佳因子:")
            print(f"   公式: {best['formula']}")
            print(f"   夏普比率: {best.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"   最大回撤: {best.get('max_drawdown', 'N/A'):.4f}")
            print(f"   年化收益: {best.get('annual_return', 'N/A'):.4f}")
            print(f"   胜率: {best.get('win_rate', 'N/A'):.4f}")
            print(f"   信息系数: {best.get('information_coefficient', 'N/A'):.4f}")

        print(f"\n📊 进化统计:")
        stats = results['evolution_stats']
        print(f"   总代数: {stats.get('total_generations', 'N/A')}")
        print(f"   种群大小: {stats.get('population_size', 'N/A')}")
        print(f"   收敛代数: {stats.get('convergence_generation', 'N/A')}")
        print(f"   种群多样性: {stats.get('diversity_score', 'N/A'):.4f}")

        print(f"\n🔝 前5个最佳因子:")
        for i, factor in enumerate(results['best_factors'][:5], 1):
            print(f"\n   {i}. 夏普比率: {factor.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"      公式: {factor['formula']}")
            print(f"      最大回撤: {factor.get('max_drawdown', 'N/A'):.4f}")

        print("\n" + "=" * 80)


def run_genetic_factor_evolution(factor_data: pd.DataFrame,
                                output_dir: str = "./results",
                                population_size: int = 1000,
                                generations: int = 20,
                                n_best_factors: int = 10) -> Dict[str, Any]:
    """
    运行遗传编程因子进化

    Args:
        factor_data: 因子数据
        output_dir: 输出目录
        population_size: 种群大小
        generations: 进化代数
        n_best_factors: 返回最佳因子数量

    Returns:
        进化结果字典
    """
    # 创建进化器
    evolver = GeneticFactorEvolution(output_dir=output_dir)
    evolver.population_size = population_size
    evolver.generations = generations

    # 开始进化
    results = evolver.evolve_factors(factor_data, n_best_factors)

    # 打印结果摘要
    evolver.print_results_summary(results)

    return results


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("遗传编程因子进化模块测试")
    print("请在main.py中调用此模块进行因子进化")
