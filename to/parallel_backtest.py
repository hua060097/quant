#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行回测引擎模块

此模块提供了高性能的并行回测功能，可以在多核系统上加速回测过程。
支持多种并行化方式：
1. Ray分布式计算框架
2. Python multiprocessing进程池
3. 线程池（适用于I/O密集型任务）

此模块与现有回测框架无缝集成，可以加速信号生成、回测执行和参数优化等过程。
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import pickle
import copy

# 设置日志
logger = logging.getLogger("parallel_backtest")
if not logger.handlers:
    # 配置日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 配置文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, "parallel_backtest.log"))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器到日志
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

# 检查是否可以导入Ray
try:
    import ray
    HAS_RAY = True
    if not ray.is_initialized():
        try:
            # 尝试初始化Ray
            ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)
            logger.info("Ray已成功初始化")
        except Exception as e:
            logger.warning(f"Ray初始化失败: {e}")
            HAS_RAY = False
except ImportError:
    HAS_RAY = False
    logger.warning("未找到Ray库，将使用进程池作为替代")

# 工作函数：生成单个区间的交易信号
def _generate_signals_for_interval(model_class, start_date, end_date):
    """为特定日期区间生成交易信号"""
    try:
        start_time = time.time()
        model = model_class()
        model.load_data()
        signals = model.generate_signals(start_date, end_date)
        elapsed = time.time() - start_time
        logger.info(f"为区间 {start_date} 到 {end_date} 生成信号完成，耗时: {elapsed:.2f}秒")
        return signals
    except Exception as e:
        logger.error(f"生成区间 {start_date} 到 {end_date} 的信号时出错: {e}")
        return {}

# Ray版本的信号生成函数
if HAS_RAY:
    @ray.remote
    def _ray_generate_signals(model_class, start_date, end_date):
        return _generate_signals_for_interval(model_class, start_date, end_date)

# 工作函数：运行单个区间的回测
def _run_backtest_for_interval(engine, start_date, end_date, signals):
    """为特定日期区间运行回测"""
    try:
        start_time = time.time()
        # 克隆回测引擎以避免共享状态
        sub_engine = copy.deepcopy(engine)
        sub_engine.start_date = start_date
        sub_engine.end_date = end_date
        sub_engine.trading_signals = {k: v for k, v in signals.items() if start_date <= k <= end_date}
        
        # 运行回测
        sub_engine.run_backtest()
        
        elapsed = time.time() - start_time
        logger.info(f"为区间 {start_date} 到 {end_date} 运行回测完成，耗时: {elapsed:.2f}秒")
        return sub_engine
    except Exception as e:
        logger.error(f"运行区间 {start_date} 到 {end_date} 的回测时出错: {e}")
        return None

# Ray版本的回测函数
if HAS_RAY:
    @ray.remote
    def _ray_run_backtest(engine, start_date, end_date, signals):
        return _run_backtest_for_interval(engine, start_date, end_date, signals)

class ParallelBacktestEngine:
    """并行回测引擎类
    
    提供高性能的并行回测功能，可以加速信号生成、回测执行和参数优化等过程。
    支持多种并行化方式：Ray、进程池、线程池。
    """
    
    def __init__(self, backend="auto", workers=None):
        """初始化并行回测引擎
        
        参数:
            backend: 并行化后端类型，可选 'ray', 'process', 'thread', 'auto'
            workers: 工作进程/线程数，默认为CPU核心数
        """
        self.backend = backend
        self.workers = workers or mp.cpu_count()
        
        # 自动选择最佳后端
        if self.backend == "auto":
            if HAS_RAY:
                self.backend = "ray"
                logger.info(f"自动选择Ray作为并行后端, 使用{self.workers}个工作进程")
            else:
                self.backend = "process"
                logger.info(f"自动选择进程池作为并行后端, 使用{self.workers}个工作进程")
        
        logger.info(f"初始化并行回测引擎, 后端: {self.backend}, 工作进程/线程数: {self.workers}")
        
        # 进程池/线程池
        self._pool = None
        
        # 性能统计
        self.performance_stats = {
            "signal_generation_time": 0,
            "backtest_time": 0,
            "speedup_ratio": 0
        }
    
    def _get_pool(self):
        """获取适当的进程/线程池"""
        if self.backend == "process" and self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        elif self.backend == "thread" and self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
        return self._pool
    
    def parallel_generate_signals(self, model, date_ranges):
        """并行生成交易信号
        
        参数:
            model: 量化模型对象或类
            date_ranges: 日期区间列表，每个元素为(start_date, end_date)元组
            
        返回:
            合并后的交易信号字典
        """
        start_time = time.time()
        model_class = model if isinstance(model, type) else model.__class__
        signals = {}
        
        # 检查日期区间列表是否有效
        if not date_ranges:
            logger.error("日期区间列表为空")
            return {}
        
        logger.info(f"开始并行生成交易信号，使用{self.backend}后端, 区间数量: {len(date_ranges)}")
        
        try:
            # 使用Ray进行并行计算
            if self.backend == "ray" and HAS_RAY:
                # 提交所有任务
                futures = [
                    _ray_generate_signals.remote(model_class, start_date, end_date)
                    for start_date, end_date in date_ranges
                ]
                
                # 获取结果
                results = ray.get(futures)
                
                # 合并结果
                for result in results:
                    signals.update(result)
            
            # 使用进程池或线程池
            elif self.backend in ["process", "thread"]:
                pool = self._get_pool()
                futures = []
                
                # 提交所有任务
                for start_date, end_date in date_ranges:
                    future = pool.submit(
                        _generate_signals_for_interval,
                        model_class, start_date, end_date
                    )
                    futures.append(future)
                
                # 获取结果
                for future in futures:
                    result = future.result()
                    signals.update(result)
            
            else:
                logger.warning(f"不支持的后端类型: {self.backend}，使用顺序执行")
                # 顺序执行
                for start_date, end_date in date_ranges:
                    result = _generate_signals_for_interval(model_class, start_date, end_date)
                    signals.update(result)
        
        except Exception as e:
            logger.error(f"并行生成交易信号时出错: {e}")
            return {}
        
        elapsed = time.time() - start_time
        self.performance_stats["signal_generation_time"] = elapsed
        logger.info(f"并行生成交易信号完成，总耗时: {elapsed:.2f}秒, 信号数量: {len(signals)}")
        
        return signals
    
    def run_parallel_backtest(self, main_engine, date_ranges):
        """并行运行回测
        
        参数:
            main_engine: 回测引擎实例
            date_ranges: 日期区间列表，每个元素为(start_date, end_date)元组
            
        返回:
            子回测引擎列表
        """
        start_time = time.time()
        results = []
        
        # 检查日期区间列表是否有效
        if not date_ranges:
            logger.error("日期区间列表为空")
            return []
        
        # 确保引擎有交易信号
        if not hasattr(main_engine, 'trading_signals') or not main_engine.trading_signals:
            logger.error("回测引擎没有交易信号")
            return []
        
        logger.info(f"开始并行执行回测，使用{self.backend}后端, 区间数量: {len(date_ranges)}")
        
        try:
            # 使用Ray进行并行计算
            if self.backend == "ray" and HAS_RAY:
                # 提交所有任务
                futures = [
                    _ray_run_backtest.remote(
                        main_engine, start_date, end_date, main_engine.trading_signals
                    )
                    for start_date, end_date in date_ranges
                ]
                
                # 获取结果
                results = ray.get(futures)
                
            # 使用进程池或线程池
            elif self.backend in ["process", "thread"]:
                pool = self._get_pool()
                futures = []
                
                # 提交所有任务
                for start_date, end_date in date_ranges:
                    future = pool.submit(
                        _run_backtest_for_interval,
                        main_engine, start_date, end_date, main_engine.trading_signals
                    )
                    futures.append(future)
                
                # 获取结果
                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
            
            else:
                logger.warning(f"不支持的后端类型: {self.backend}，使用顺序执行")
                # 顺序执行
                for start_date, end_date in date_ranges:
                    result = _run_backtest_for_interval(
                        main_engine, start_date, end_date, main_engine.trading_signals
                    )
                    if result:
                        results.append(result)
        
        except Exception as e:
            logger.error(f"并行执行回测时出错: {e}")
            return []
        
        # 过滤无效结果
        results = [r for r in results if r is not None]
        
        elapsed = time.time() - start_time
        self.performance_stats["backtest_time"] = elapsed
        logger.info(f"并行执行回测完成，总耗时: {elapsed:.2f}秒, 有效子回测数量: {len(results)}")
        
        return results
    
    def merge_backtest_results(self, backtest_results):
        """合并多个回测结果
        
        参数:
            backtest_results: 子回测引擎列表
            
        返回:
            (合并后的回测引擎, 性能统计字典)
        """
        if not backtest_results:
            logger.error("没有有效的回测结果可合并")
            return None, {}
        
        try:
            logger.info(f"开始合并{len(backtest_results)}个回测结果")
            
            # 使用第一个结果作为基础
            merged_engine = backtest_results[0]
            
            # 合并交易历史
            for engine in backtest_results[1:]:
                if hasattr(engine, 'trade_history') and engine.trade_history:
                    merged_engine.trade_history.extend(engine.trade_history)
            
            # 按日期排序交易历史
            if hasattr(merged_engine, 'trade_history') and merged_engine.trade_history:
                merged_engine.trade_history.sort(key=lambda x: x.get('date', ''))
            
            # 合并持仓历史
            position_history = {}
            for engine in backtest_results:
                if hasattr(engine, 'position_history') and engine.position_history:
                    for date, positions in engine.position_history.items():
                        position_history[date] = positions
            
            # 按日期排序持仓历史
            merged_engine.position_history = {
                k: position_history[k] for k in sorted(position_history.keys())
            }
            
            # 合并股票池
            stock_pool = set()
            for engine in backtest_results:
                if hasattr(engine, 'stock_pool') and engine.stock_pool:
                    stock_pool.update(engine.stock_pool)
            merged_engine.stock_pool = list(stock_pool)
            
            # 合并账户历史
            account_history = {}
            for engine in backtest_results:
                if hasattr(engine, 'account_history') and engine.account_history:
                    for date, account in engine.account_history.items():
                        account_history[date] = account
            
            # 按日期排序账户历史
            merged_engine.account_history = {
                k: account_history[k] for k in sorted(account_history.keys())
            }
            
            # 更新引擎的开始和结束日期
            all_dates = list(merged_engine.account_history.keys())
            if all_dates:
                merged_engine.start_date = min(all_dates)
                merged_engine.end_date = max(all_dates)
            
            # 计算性能统计
            stats = self._calculate_performance_stats(merged_engine)
            
            logger.info("合并回测结果完成")
            return merged_engine, stats
            
        except Exception as e:
            logger.error(f"合并回测结果时出错: {e}")
            return backtest_results[0] if backtest_results else None, {}
    
    def _calculate_performance_stats(self, engine):
        """计算回测性能统计
        
        参数:
            engine: 回测引擎
            
        返回:
            性能统计字典
        """
        stats = {}
        
        try:
            # 计算基本指标
            if hasattr(engine, 'account_history') and engine.account_history:
                # 提取净值序列
                dates = sorted(engine.account_history.keys())
                values = [engine.account_history[d]['total_value'] for d in dates]
                
                # 计算总收益率
                initial_value = values[0] if values else 0
                final_value = values[-1] if values else 0
                total_return = (final_value / initial_value - 1) if initial_value > 0 else 0
                
                # 计算最大回撤
                max_drawdown = 0
                peak = values[0] if values else 0
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak if peak > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                # 计算年化收益率
                if len(dates) > 1:
                    start_date = datetime.strptime(dates[0], "%Y%m%d")
                    end_date = datetime.strptime(dates[-1], "%Y%m%d")
                    years = (end_date - start_date).days / 365.0
                    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
                else:
                    annual_return = 0
                
                # 计算交易次数和胜率
                win_count = 0
                if hasattr(engine, 'trade_history') and engine.trade_history:
                    for trade in engine.trade_history:
                        if trade.get('profit', 0) > 0:
                            win_count += 1
                    
                    win_rate = win_count / len(engine.trade_history) if engine.trade_history else 0
                else:
                    win_rate = 0
                
                # 填充统计字典
                stats = {
                    "total_return": total_return,
                    "annual_return": annual_return,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "trade_count": len(engine.trade_history) if hasattr(engine, 'trade_history') else 0,
                    "processing_time": self.performance_stats["signal_generation_time"] + self.performance_stats["backtest_time"]
                }
        
        except Exception as e:
            logger.error(f"计算性能统计时出错: {e}")
        
        return stats
    
    def run_parameter_optimization(self, model_class, param_grid, start_date, end_date, eval_metric='annual_return'):
        """并行运行参数优化
        
        参数:
            model_class: 量化模型类
            param_grid: 参数网格，格式为{参数名:[参数值列表]}
            start_date: 回测开始日期
            end_date: 回测结束日期
            eval_metric: 评估指标，可选'total_return', 'annual_return', 'max_drawdown', 'win_rate'
            
        返回:
            (最佳参数字典, 所有参数评估结果列表)
        """
        try:
            from itertools import product
            
            # 生成所有参数组合
            param_names = list(param_grid.keys())
            param_values = list(product(*[param_grid[name] for name in param_names]))
            param_dicts = [dict(zip(param_names, values)) for values in param_values]
            
            logger.info(f"开始参数优化，参数组合数: {len(param_dicts)}")
            
            results = []
            best_score = float('-inf') if eval_metric != 'max_drawdown' else float('inf')
            best_params = None
            
            # 定义评估函数
            def evaluate_params(params):
                try:
                    # 初始化模型和回测引擎
                    model = model_class(**params)
                    
                    # 生成信号
                    signals = model.generate_signals(start_date, end_date)
                    
                    # 创建回测引擎
                    from backtest_engine import BacktestEngine
                    engine = BacktestEngine(
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=1000000,
                        position_size=0.1,
                        max_positions=10
                    )
                    
                    # 设置信号并运行回测
                    engine.trading_signals = signals
                    engine.run_backtest()
                    
                    # 计算性能指标
                    stats = self._calculate_performance_stats(engine)
                    
                    # 添加参数
                    stats.update(params)
                    
                    return stats
                except Exception as e:
                    logger.error(f"评估参数组合时出错: {e}, 参数: {params}")
                    return None
            
            # 使用Ray进行并行计算
            if self.backend == "ray" and HAS_RAY:
                @ray.remote
                def ray_evaluate_params(params):
                    return evaluate_params(params)
                
                # 提交所有任务
                futures = [ray_evaluate_params.remote(params) for params in param_dicts]
                
                # 获取结果
                eval_results = ray.get(futures)
                results = [r for r in eval_results if r is not None]
            
            # 使用进程池或线程池
            elif self.backend in ["process", "thread"]:
                pool = self._get_pool()
                futures = [pool.submit(evaluate_params, params) for params in param_dicts]
                
                # 获取结果
                eval_results = [future.result() for future in futures]
                results = [r for r in eval_results if r is not None]
            
            else:
                # 顺序执行
                for params in param_dicts:
                    result = evaluate_params(params)
                    if result:
                        results.append(result)
            
            # 找到最佳参数
            for result in results:
                score = result.get(eval_metric, 0)
                if eval_metric == 'max_drawdown':
                    # 最大回撤越小越好
                    if score < best_score:
                        best_score = score
                        best_params = {k: result[k] for k in param_names}
                else:
                    # 其他指标越大越好
                    if score > best_score:
                        best_score = score
                        best_params = {k: result[k] for k in param_names}
            
            logger.info(f"参数优化完成，评估了{len(results)}个有效参数组合")
            logger.info(f"最佳参数: {best_params}, {eval_metric}: {best_score}")
            
            return best_params, results
            
        except Exception as e:
            logger.error(f"运行参数优化时出错: {e}")
            return {}, []
    
    def preprocess_data_parallel(self, data_loader, process_func, chunk_size=1000):
        """并行预处理数据
        
        参数:
            data_loader: 数据加载函数或可迭代对象
            process_func: 处理函数，接收数据块并返回处理结果
            chunk_size: 数据块大小
            
        返回:
            处理后的数据列表
        """
        try:
            # 加载数据
            if callable(data_loader):
                data = data_loader()
            else:
                data = data_loader
            
            # 分块
            chunks = []
            if isinstance(data, list):
                for i in range(0, len(data), chunk_size):
                    chunks.append(data[i:i+chunk_size])
            elif isinstance(data, dict):
                keys = list(data.keys())
                for i in range(0, len(keys), chunk_size):
                    chunk_keys = keys[i:i+chunk_size]
                    chunks.append({k: data[k] for k in chunk_keys})
            elif isinstance(data, pd.DataFrame):
                for i in range(0, len(data), chunk_size):
                    chunks.append(data.iloc[i:i+chunk_size])
            else:
                logger.error(f"不支持的数据类型: {type(data)}")
                return []
            
            logger.info(f"数据分块完成，共{len(chunks)}个块")
            
            # 并行处理
            if self.backend == "ray" and HAS_RAY:
                @ray.remote
                def ray_process_chunk(chunk):
                    return process_func(chunk)
                
                # 提交所有任务
                futures = [ray_process_chunk.remote(chunk) for chunk in chunks]
                
                # 获取结果
                results = ray.get(futures)
            
            elif self.backend in ["process", "thread"]:
                pool = self._get_pool()
                futures = [pool.submit(process_func, chunk) for chunk in chunks]
                
                # 获取结果
                results = [future.result() for future in futures]
            
            else:
                # 顺序执行
                results = [process_func(chunk) for chunk in chunks]
            
            # 合并结果
            if all(isinstance(r, list) for r in results):
                merged_results = []
                for result in results:
                    merged_results.extend(result)
                return merged_results
            elif all(isinstance(r, dict) for r in results):
                merged_results = {}
                for result in results:
                    merged_results.update(result)
                return merged_results
            elif all(isinstance(r, pd.DataFrame) for r in results):
                return pd.concat(results, ignore_index=True)
            else:
                return results
            
        except Exception as e:
            logger.error(f"并行预处理数据时出错: {e}")
            return []
    
    def shutdown(self):
        """关闭并行引擎，释放资源"""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        
        if self.backend == "ray" and HAS_RAY and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray已关闭")
            except Exception as e:
                logger.warning(f"关闭Ray时出错: {e}")
        
        logger.info("并行回测引擎已关闭")
        
        return self.performance_stats

if __name__ == "__main__":
    # 示例使用
    logger.info("并行回测引擎模块")
    logger.info("请通过导入此模块使用，而不是直接运行")
    
    # 检查Ray是否可用
    if HAS_RAY:
        logger.info("Ray分布式计算框架可用")
    else:
        logger.info("Ray分布式计算框架不可用，将使用进程池")
    
    # 检查CPU核心数
    logger.info(f"检测到{mp.cpu_count()}个CPU核心") 