#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
执行回测并分析结果的主脚本
1. 运行回测获取模型生成信号的表现
2. 分析回测结果并与基准对比
3. 生成综合分析报告
"""

import os
import sys
import logging
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from run_backtest import run_backtest_with_params
from backtest_signals import BacktestEngine  # 导入BacktestEngine
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import ray
from backtest_analyzer import BacktestAnalyzer
import json
import time
import traceback
import numpy as np
import pandas as pd
import argparse

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/run_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_analysis")

# 尝试导入缓存集成模块
try:
    from cache_integration import integrate_cache_system, restore_original_functions
    CACHE_AVAILABLE = True
    logger.info("缓存系统可用，将自动优化数据加载性能")
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("未找到缓存系统，将使用标准数据加载")
    integrate_cache_system = None
    restore_original_functions = None

# 导入增量训练集成模块并自动激活
try:
    # 尝试从to目录导入
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from incremental_integration import get_incremental_integration
    
    # 初始化增量训练集成
    incremental_integration = get_incremental_integration()
    logger.info("增量训练集成模块已加载")
except ImportError as e:
    logger.warning(f"无法导入增量训练集成模块: {e}，将使用标准训练方法")
    incremental_integration = None
except Exception as e:
    logger.warning(f"初始化增量训练集成模块时出错: {e}，将使用标准训练方法")
    incremental_integration = None

# 导入向量化特征提取集成模块并自动激活
try:
    # 尝试从to目录导入
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from vectorized_integration import get_vectorized_integration
    
    # 初始化向量化特征提取集成 - 确保创建新实例
    vectorized_integration = get_vectorized_integration(create_new=True)
    logger.info("向量化特征提取集成模块已加载")
except ImportError as e:
    logger.warning(f"无法导入向量化特征提取集成模块: {e}，将使用标准特征提取方法")
    vectorized_integration = None
except Exception as e:
    logger.warning(f"初始化向量化特征提取集成模块时出错: {e}，将使用标准特征提取方法")
    vectorized_integration = None

# 导入并行回测引擎
try:
    from parallel_backtest import ParallelBacktestEngine
    logger.info("并行回测引擎已加载")
    HAS_PARALLEL_ENGINE = True
except ImportError as e:
    logger.warning(f"无法导入并行回测引擎: {e}，将使用标准回测方式")
    HAS_PARALLEL_ENGINE = False
except Exception as e:
    logger.warning(f"初始化并行回测引擎时出错: {e}，将使用标准回测方式")
    HAS_PARALLEL_ENGINE = False

# 初始化缓存系统
cache_integration = None
if CACHE_AVAILABLE:
    try:
        cache_integration = integrate_cache_system()
        if cache_integration:
            logger.info("数据缓存系统已成功集成，将提升数据访问性能")
        else:
            logger.warning("数据缓存系统集成失败，将使用标准数据访问方式")
    except Exception as e:
        logger.warning(f"初始化数据缓存系统时出错: {e}，将使用标准数据访问方式")

def init_ray():
    """初始化Ray分布式计算框架"""
    try:
        # 检测操作系统类型
        import platform
        if platform.system() == 'Windows':
            logger.warning("检测到Windows系统，不初始化Ray以避免兼容性问题")
            return False
            
        if not ray.is_initialized():
            logger.info("尝试初始化Ray...")
            
            # 使用最小化的配置来确保成功初始化
            # 首先尝试本地模式
            logger.info("尝试本地模式初始化Ray...")
            ray.init(
                local_mode=True,
                include_dashboard=False,
                logging_level=logging.ERROR
            )
            logger.info("Ray本地模式初始化成功")
            ray_initialized = True
            
            # 测试Ray是否正常工作
            @ray.remote
            def test_ray():
                import socket
                import os
                return {"hostname": socket.gethostname(), "pid": os.getpid()}
            
            result = ray.get(test_ray.remote())
            logger.info(f"Ray测试成功: {result}")
            
        else:
            logger.info("Ray已经初始化")
            ray_initialized = True
        return ray_initialized
    except Exception as e:
        logger.warning(f"Ray初始化失败: {e}, 将使用标准并行计算")
        return False

def run_backtest(start_date, end_date, initial_capital):
    """运行回测并返回回测引擎"""
    logger.info(f"开始回测，时间范围: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,} 元")
    
    # 运行回测
    backtest_engine = run_backtest_with_params(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # 保存回测结果
    try:
        with open('backtest_results.pkl', 'wb') as f:
            pickle.dump(backtest_engine, f)
        logger.info("回测结果已保存至 backtest_results.pkl")
    except Exception as e:
        logger.error(f"保存回测结果时出错: {e}")
    
    return backtest_engine

def load_benchmark_and_calculate_random(analyzer, benchmarks, num_trials):
    """加载基准数据并计算随机投资组合表现"""
    # 使用线程池同时执行这两个任务
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交两个任务
        load_benchmark_future = executor.submit(
            analyzer.load_benchmark_data, benchmarks
        )
        calculate_random_future = executor.submit(
            analyzer.calculate_random_portfolio, num_trials
        )
        
        # 获取结果
        benchmark_result = load_benchmark_future.result()
        random_result = calculate_random_future.result()
        
        logger.info(f"加载基准数据: {'成功' if benchmark_result else '失败'}")
        logger.info(f"计算随机投资组合: {'成功' if random_result else '失败'}")
        
        return benchmark_result, random_result

def generate_analysis_reports(analyzer):
    """并行生成各种分析报告"""
    with ProcessPoolExecutor(max_workers=3) as executor:
        # 并行提交任务
        futures = []
        # 1. 绘制表现对比图
        futures.append(executor.submit(analyzer.plot_performance_comparison))
        # 2. 计算统计指标
        futures.append(executor.submit(analyzer.calculate_statistics))
        # 3. 分析信号有效性
        futures.append(executor.submit(analyzer.analyze_signal_effectiveness))
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"生成分析报告时出错: {e}")
    
    # 生成综合报告（需要前面的结果，单独执行）
    report = analyzer.generate_comprehensive_report()
    
    return report

def train_model(model, start_date, end_date, is_parameter_optimization=False):
    """训练模型并保存
    
    参数:
        model: 量化模型对象
        start_date: 开始日期
        end_date: 结束日期
        is_parameter_optimization: 是否进行参数优化
    """
    logger.info(f"开始训练模型: {start_date} 至 {end_date}, 参数优化: {is_parameter_optimization}")
    
    # 由于Ray存在初始化问题，直接使用进程池模式
    logger.info("系统将使用进程池模式进行模型训练")
    
    try:
        # 获取行业数据
        sector_names = model._load_sector_data()
        if not sector_names:
            logger.error("没有可用的行业数据，无法训练模型")
            return False
        
        # 使用进程池模式进行训练
        result = model.train_sector_models(
            date=end_date,
            lookback_short=30,  # 短期模型回溯30天
            lookback_long=90,   # 长期模型回溯90天
            forward_days=5,     # 预测5天后的涨跌幅
            parallel="process_pool"  # 明确指定使用进程池，不尝试Ray
        )
        
        if result:
            success_count = result.get("success", 0)
            error_count = result.get("error", 0)
            logger.info(f"模型训练完成，成功: {success_count}，失败: {error_count}")
            
            # 保存模型
            model._save_models()
            logger.info("模型已保存")
            
            return True
        else:
            logger.error("模型训练失败")
            return False
            
    except Exception as e:
        logger.error(f"训练模型时出错: {e}")
        logger.error(traceback.format_exc())
        return False

def evaluate_model_performance(model):
    """评估模型性能，返回综合评分"""
    try:
        performance_metrics = {}
        
        # 汇总所有板块模型的性能
        for sector, model_data in model.sector_models.items():
            if 'model' in model_data and isinstance(model_data['model'], dict):
                metrics = model_data['model']
                for metric, value in metrics.items():
                    if metric not in performance_metrics:
                        performance_metrics[metric] = []
                    if isinstance(value, (int, float)):
                        performance_metrics[metric].append(value)
        
        # 计算平均性能
        avg_metrics = {}
        for metric, values in performance_metrics.items():
            if values:
                avg_metrics[metric] = sum(values) / len(values)
        
        # 计算综合得分
        score = 0.0
        weights = {
            'accuracy': 0.4,
            'precision': 0.3,
            'recall': 0.3
        }
        
        for metric, weight in weights.items():
            if metric in avg_metrics:
                score += avg_metrics[metric] * weight
        
        return {
            'metrics': avg_metrics,
            'score': score
        }
    except Exception as e:
        logger.error(f"评估模型性能时出错: {e}")
        return {'metrics': {}, 'score': 0.0}

def run_backtest_with_rolling_training(start_date, end_date, initial_capital=1000000, 
                                     long_cycle_update=10, short_cycle_update=1):
    """运行带有滚动训练的回测
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资本
        long_cycle_update: 长周期模型更新间隔(天)
        short_cycle_update: 短周期模型更新间隔(天)
    """
    logger.info(f"开始带滚动训练的回测: {start_date} 至 {end_date}")
    
    # 创建回测引擎
    engine = None
    try:
        # 加载最佳参数 (如果有)
        best_params = {}
        if os.path.exists('best_params.pkl'):
            try:
                with open('best_params.pkl', 'rb') as f:
                    loaded_params = pickle.load(f)
                    # 参数名称映射
                    param_mapping = {
                        'short': 'lookback_short',
                        'long': 'lookback_long'
                    }
                    # 转换参数名称
                    best_params = {}
                    for key, value in loaded_params.items():
                        # 如果有对应的映射名称，使用映射名称
                        mapped_key = param_mapping.get(key, key)
                        # 确保数字类型参数被正确转换
                        if mapped_key in ['lookback_short', 'lookback_long', 'forward_days']:
                            best_params[mapped_key] = int(value) if isinstance(value, str) else value
                        elif mapped_key == 'test_size':
                            best_params[mapped_key] = float(value) if isinstance(value, str) else value
                        else:
                            best_params[mapped_key] = value
                logger.info(f"加载最佳参数: {best_params}")
            except Exception as e:
                logger.warning(f"加载最佳参数失败: {e}")
        
        # 创建高级量化模型实例
        from advanced_quant_model import AdvancedQuantModel
        model = AdvancedQuantModel()
        
        # 集成增量训练功能(如果可用)
        if incremental_integration is not None:
            try:
                logger.info("正在集成增量训练功能到量化模型...")
                integration_success = incremental_integration.integrate(model.__class__)
                if integration_success:
                    logger.info("增量训练功能已成功集成到量化模型中")
                else:
                    logger.warning("增量训练功能集成失败，将使用标准训练方法")
            except Exception as e:
                logger.warning(f"集成增量训练功能时出错: {e}，将使用标准训练方法")
        
        # 集成向量化特征提取功能(如果可用)
        if vectorized_integration is not None:
            try:
                logger.info("正在集成向量化特征提取功能到量化模型...")
                vectorized_success = vectorized_integration.integrate(
                    target_class=None,
                    target_methods=['_extract_features'],
                    debug_mode=True
                )
                if vectorized_success:
                    logger.info("向量化特征提取功能已成功集成到量化模型中")
                    logger.info("现在使用向量化计算替代传统循环，提高特征提取速度")
                else:
                    logger.warning("向量化特征提取功能集成失败，将使用标准特征提取方法")
            except Exception as e:
                logger.warning(f"集成向量化特征提取功能时出错: {e}，将使用标准特征提取方法")
        
        # 加载交易日历
        logger.info("加载交易日历数据...")
        model.load_trading_dates()
        
        # 获取交易日历
        trading_dates = model.trading_dates
        
        # 找到开始和结束日期的索引
        if start_date not in trading_dates or end_date not in trading_dates:
            # 验证日期是否为交易日
            if start_date not in trading_dates:
                logger.error(f"开始日期 {start_date} 不是交易日")
            if end_date not in trading_dates:
                logger.error(f"结束日期 {end_date} 不是交易日")
            logger.error("回测日期必须是实际交易日，请检查输入日期")
            return None
        
        # 获取日期索引
        start_idx = trading_dates.index(start_date)
        end_idx = trading_dates.index(end_date)
        
        # 确定第一次长周期和短周期更新的日期
        last_long_update = None
        last_short_update = None
        
        # 批量处理大小
        batch_size = 20
        
        # 计算总批次数
        total_batches = (end_idx - start_idx + 1) // batch_size + (1 if (end_idx - start_idx + 1) % batch_size > 0 else 0)
        logger.info(f"回测将分 {total_batches} 批次执行")
        
        # 创建信号字典
        all_signals = {}
        
        # 分批处理交易日期
        for batch_idx in range(total_batches):
            batch_start_idx = start_idx + batch_idx * batch_size
            batch_end_idx = min(batch_start_idx + batch_size - 1, end_idx)
            
            batch_start_date = trading_dates[batch_start_idx]
            batch_end_date = trading_dates[batch_end_idx]
            
            logger.info(f"处理批次 {batch_idx+1}/{total_batches}: {batch_start_date} 至 {batch_end_date}")
            
            # 为该批次创建回测引擎
            if engine is None:
                # 第一次创建引擎
                engine = BacktestEngine(
                    start_date=start_date, 
                    end_date=end_date,
                    initial_capital=initial_capital
                )
            
            # 处理这个批次中的每一天
            for day_idx in range(batch_start_idx, batch_end_idx + 1):
                current_date = trading_dates[day_idx]
                
                # 跟踪已经处理过的交易日数量
                processed_trading_days = day_idx - start_idx + 1
                
                # 检查是否需要进行长周期模型更新
                days_since_long_update = ((day_idx - trading_dates.index(last_long_update)) 
                                         if last_long_update else long_cycle_update)
                
                if days_since_long_update >= long_cycle_update:
                    logger.info(f"执行长周期模型更新，日期: {current_date}，处理进度: {processed_trading_days}/{end_idx-start_idx+1}")
                    
                    # 使用最佳参数或默认参数
                    lookback_long = int(best_params.get('lookback_long', 90))
                    lookback_short = int(best_params.get('lookback_short', 30))
                    parallel = best_params.get('parallel', 'process_pool')
                    
                    # 长周期训练使用较大的数据窗口
                    success = model.train_sector_models(
                        date=current_date,
                        lookback_short=lookback_short,
                        lookback_long=lookback_long,
                        parallel=parallel
                    )
                    
                    if success:
                        logger.info(f"长周期模型更新成功")
                        last_long_update = current_date
                        last_short_update = current_date  # 长周期更新也算作短周期更新
                    else:
                        logger.error(f"长周期模型更新失败")
                
                # 检查是否需要进行短周期模型更新
                elif (last_short_update is None) or \
                     ((day_idx - trading_dates.index(last_short_update)) >= short_cycle_update):
                    logger.info(f"执行短周期模型更新，日期: {current_date}，处理进度: {processed_trading_days}/{end_idx-start_idx+1}")
                    
                    # 短周期训练只更新短周期参数
                    lookback_short = int(best_params.get('lookback_short', 30))
                    
                    # 短周期训练使用较小的数据窗口
                    success = model.train_sector_models(
                        date=current_date,
                        lookback_short=lookback_short,
                        lookback_long=0,  # 不使用长周期数据
                        parallel="sequential"  # 使用简单的顺序处理即可
                    )
                    
                    if success:
                        logger.info(f"短周期模型更新成功")
                        last_short_update = current_date
                    else:
                        logger.warning(f"短周期模型更新失败")
                
                # 生成当天的交易信号
                signals = model.generate_trading_signals(date=current_date)
                if signals:
                    logger.info(f"日期 {current_date} 生成 {len(signals)} 个交易信号")
                    all_signals[current_date] = signals
                else:
                    logger.warning(f"日期 {current_date} 未生成交易信号")
        
        # 回测引擎加载所有信号
        if engine and all_signals:
            # 设置信号字典
            engine.trading_signals = all_signals
            
            # 运行回测
            logger.info("开始执行回测...")
            if engine.run_backtest():
                logger.info("回测执行成功")
                
                # 保存回测结果
                with open('backtest_results.pkl', 'wb') as f:
                    pickle.dump(engine, f)
                logger.info("回测结果已保存至 backtest_results.pkl")
                
                # 生成回测报告
                report = engine.generate_report()
                
                return engine, report
            else:
                logger.error("回测执行失败")
                return None
        else:
            logger.error("回测引擎创建失败或未生成交易信号")
            return None
        
    except Exception as e:
        logger.exception(f"执行回测过程中出现异常: {e}")
        return None

def run_analysis_parallel(start_date, end_date, initial_capital=1000000, params=None):
    """
    并行运行分析流程
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        params: 参数字典
    """
    logger.info(f"开始并行分析: {start_date} 到 {end_date}, 初始资金: {initial_capital:,.2f}")
    
    # 默认参数
    if params is None:
        params = {}
    
    # 创建并行回测引擎
    parallel_engine = ParallelBacktestEngine(backend="auto")
    
    # 加载模型
    logger.info("加载量化模型")
    try:
        # 导入高级量化模型
        from advanced_quant_model import AdvancedQuantModel
        model = AdvancedQuantModel()
        
        # 如果缓存系统可用，预加载关键数据
        if CACHE_AVAILABLE and cache_integration:
            logger.info("使用缓存系统预加载关键市场数据...")
            try:
                # 获取所有股票代码
                stock_universe = model.get_stock_universe()
                
                # 预加载市场数据以提高性能
                fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
                cache_integration.parallel_preload(
                    stock_list=stock_universe[:300],  # 限制预加载的股票数量以避免内存溢出
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields,
                    periods=['1d'],
                    batch_size=50
                )
                
                # 预加载交易日历数据
                cache_integration.preload_data(
                    stock_list=[],
                    start_date="20220101",
                    end_date="20250401",
                    fields=[],
                    periods=[]
                )
                
                # 生成性能报告
                cache_stats = cache_integration.get_performance_report().get('cache_stats', {})
                logger.info(f"缓存预加载完成。内存使用: {cache_stats.get('memory_usage_mb', 0):.2f}MB")
            except Exception as e:
                logger.warning(f"缓存预加载过程中出错: {e}，将继续使用动态缓存")
        
        model.load_data()
        model.load_trading_dates()
    except Exception as e:
        logger.error(f"加载量化模型时出错: {e}")
        return None
    
    # 集成增量训练功能(如果可用)
    if incremental_integration is not None:
        try:
            logger.info("正在集成增量训练功能到量化模型...")
            integration_success = incremental_integration.integrate(model.__class__)
            if integration_success:
                logger.info("增量训练功能已成功集成到量化模型中")
            else:
                logger.warning("增量训练功能集成失败，将使用标准训练方法")
        except Exception as e:
            logger.warning(f"集成增量训练功能时出错: {e}，将使用标准训练方法")
    
    # 集成向量化特征提取功能(如果可用)
    if vectorized_integration is not None:
        try:
            logger.info("正在集成向量化特征提取功能到量化模型...")
            vectorized_success = vectorized_integration.integrate(
                target_class=None,
                target_methods=['_extract_features'],
                debug_mode=True
            )
            if vectorized_success:
                logger.info("向量化特征提取功能已成功集成到量化模型中")
            else:
                logger.warning("向量化特征提取功能集成失败，将使用标准特征提取方法")
        except Exception as e:
            logger.warning(f"集成向量化特征提取功能时出错: {e}，将使用标准特征提取方法")
    
    # 分段处理
    if start_date not in model.trading_dates or end_date not in model.trading_dates:
        logger.error("开始或结束日期不在交易日历中")
        return None
    
    start_idx = model.trading_dates.index(start_date)
    end_idx = model.trading_dates.index(end_date)
    
    # 将整个回测区间分成几个子区间
    interval_days = params.get('interval_days', 30)
    date_ranges = []
    for i in range(start_idx, end_idx, interval_days):
        sub_end_idx = min(i + interval_days - 1, end_idx)
        date_ranges.append((
            model.trading_dates[i],
            model.trading_dates[sub_end_idx]
        ))
    
    logger.info(f"将分析区间分成 {len(date_ranges)} 个子区间进行并行处理")
    
    # 并行生成交易信号
    logger.info("开始并行生成交易信号")
    signals = parallel_engine.parallel_generate_signals(model, date_ranges)
    
    if not signals:
        logger.error("无法生成交易信号，退出分析")
        return None
    
    logger.info(f"成功生成 {len(signals)} 个日期的交易信号")
    
    # 创建回测引擎
    from backtest_engine import BacktestEngine
    main_engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        position_size=params.get('position_size', 0.1),
        max_positions=params.get('max_positions', 10)
    )
    
    # 设置交易信号
    main_engine.trading_signals = signals
    
    # 运行并行回测
    logger.info("开始执行并行回测")
    backtest_results = parallel_engine.run_parallel_backtest(main_engine, date_ranges)
    
    # 合并回测结果
    logger.info("合并并行回测结果")
    merged_engine, stats = parallel_engine.merge_backtest_results(backtest_results)
    
    # 生成分析报告
    logger.info("生成分析报告")
    if merged_engine:
        report = merged_engine.generate_report()
        logger.info("\n=== 回测性能统计 ===")
        logger.info(f"总收益率: {stats.get('total_return', 0):.4f}")
        logger.info(f"年化收益率: {stats.get('annual_return', 0):.4f}")
        logger.info(f"最大回撤: {stats.get('max_drawdown', 0):.4f}")
        logger.info(f"胜率: {stats.get('win_rate', 0):.4f}")
        logger.info(f"交易次数: {stats.get('trade_count', 0)}")
        logger.info(f"处理时间: {stats.get('processing_time', 0):.2f}秒")
    else:
        logger.error("合并回测结果失败，无法生成分析报告")
    
    # 关闭并行引擎
    parallel_engine.shutdown()
    
    return merged_engine

def test_ray():
    """测试Ray的初始化和基本功能"""
    print("==== 开始测试Ray初始化和序列化 ====")
    logger = logging.getLogger("test_ray")

    # 设置日志格式为输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    
    # 确保Ray正确初始化
    try:
        import ray
        
        # 检查Ray是否已经初始化
        if ray.is_initialized():
            logger.info("Ray已经初始化，正在关闭...")
            ray.shutdown()
            logger.info("Ray已关闭")
        
        # 初始化Ray
        logger.info("正在初始化Ray (local_mode=True)...")
        ray.init(
            local_mode=True,
            include_dashboard=False,
            ignore_reinit_error=True,
            logging_level=logging.ERROR
        )
        logger.info("Ray初始化成功")
        
        # 测试基本远程函数
        @ray.remote
        def add(a, b):
            import os
            import socket
            host = socket.gethostname()
            pid = os.getpid()
            logger.info(f"add函数在主机 {host} 的进程 {pid} 中执行")
            return a + b
        
        logger.info("测试基本远程函数...")
        result = ray.get(add.remote(1, 2))
        logger.info(f"基本远程函数测试结果: 1 + 2 = {result}")
        
        # 测试对象序列化
        logger.info("测试对象序列化...")
        
        # 测试复杂对象序列化 - 使用sklearn模型
        @ray.remote
        def create_model():
            try:
                from sklearn.ensemble import RandomForestClassifier
                import numpy as np
                
                # 创建一个简单的数据集
                X = np.random.rand(100, 4)
                y = np.random.randint(0, 2, 100)
                
                # 创建并训练模型
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(X, y)
                
                logger.info("模型训练成功，尝试返回模型...")
                return model, X, y
            except Exception as e:
                logger.error(f"创建模型时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None, None, None
        
        try:
            logger.info("调用创建模型的远程函数...")
            model_ref = create_model.remote()
            logger.info("等待模型结果...")
            model, X, y = ray.get(model_ref, timeout=30)
            
            if model is not None:
                logger.info("模型序列化成功")
                # 测试模型功能
                predictions = model.predict(X)
                accuracy = (predictions == y).mean()
                logger.info(f"模型准确率: {accuracy:.2f}")
            else:
                logger.error("无法获取模型")
        except Exception as e:
            logger.error(f"模型序列化测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 关闭Ray
        logger.info("测试完成，关闭Ray...")
        ray.shutdown()
        logger.info("Ray已关闭")
        
        print("==== Ray初始化和序列化测试完成 ====")
        return True
    
    except Exception as e:
        logger.error(f"Ray测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("==== Ray测试失败 ====")
        return False

def main():
    """主函数"""
    logger = logging.getLogger("run_analysis")
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="量化回测分析系统")
    parser.add_argument("--test-ray", action="store_true", help="测试Ray的初始化和序列化功能")
    args = parser.parse_args()
    
    # 如果指定了测试Ray，运行测试并返回
    if args.test_ray:
        print("开始测试Ray初始化...")
        success = test_ray()
        if success:
            print("Ray初始化和序列化测试成功")
            return 0
        else:
            print("Ray初始化和序列化测试失败")
            return 1
    
    logger.info("===== 开始量化回测分析 =====")
    
    try:
        # 初始化并行计算环境
        ray_initialized = init_ray()
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"设置并行计算工作线程数: {num_workers}")
        
        # 添加缓存配置选项
        enable_cache = False
        if CACHE_AVAILABLE:
            choice = input("是否启用数据缓存系统加速回测(y/n，默认y): ").strip().lower()
            if choice != 'n':
                enable_cache = True
                logger.info("启用数据缓存系统，将提高数据加载性能")
                
                # 如果之前没有初始化缓存，现在初始化
                global cache_integration
                if not cache_integration:
                    cache_integration = integrate_cache_system()
            else:
                logger.info("不使用数据缓存系统")
                # 如果之前启用了缓存，现在关闭它
                if cache_integration:
                    restore_original_functions()
                    cache_integration = None
        
        # 添加并行处理选项
        enable_parallel = False
        if HAS_PARALLEL_ENGINE:
            choice = input("是否启用并行处理加速分析(y/n，默认n): ").strip().lower()
            if choice == 'y':
                enable_parallel = True
                logger.info("启用并行处理加速分析")
            else:
                logger.info("使用标准顺序处理方式")
        
        # 创建高级量化模型实例
        from advanced_quant_model import AdvancedQuantModel
        model = AdvancedQuantModel()
        
        # 集成增量训练功能(如果可用)
        if incremental_integration is not None:
            try:
                logger.info("正在集成增量训练功能到量化模型...")
                integration_success = incremental_integration.integrate(model.__class__)
                if integration_success:
                    logger.info("增量训练功能已成功集成到量化模型中")
                else:
                    logger.warning("增量训练功能集成失败，将使用标准训练方法")
            except Exception as e:
                logger.warning(f"集成增量训练功能时出错: {e}，将使用标准训练方法")
        
        # 集成向量化特征提取功能(如果可用)
        if vectorized_integration is not None:
            try:
                logger.info("正在集成向量化特征提取功能到量化模型...")
                vectorized_success = vectorized_integration.integrate(
                    target_class=None,
                    target_methods=['_extract_features'],
                    debug_mode=True
                )
                if vectorized_success:
                    logger.info("向量化特征提取功能已成功集成到量化模型中")
                else:
                    logger.warning("向量化特征提取功能集成失败，将使用标准特征提取方法")
            except Exception as e:
                logger.warning(f"集成向量化特征提取功能时出错: {e}，将使用标准特征提取方法")
        
        # 加载交易日历
        logger.info("加载交易日历数据...")
        model.load_trading_dates()
        
        # 检查交易日历是否可用
        if not model.trading_dates or len(model.trading_dates) == 0:
            logger.error("交易日历为空，无法继续执行")
            return
        
        # 输出系统状态
        logger.info(f"数据加载完成，共 {len(model.trading_dates)} 个交易日")
        logger.info(f"可用交易日期范围: {model.trading_dates[0]} 至 {model.trading_dates[-1]}")
        
        # 计划阶段日期范围
        # 注意：实际回测将使用实际存在的交易日，跳过非交易日（如节假日、周末）
        plan_dates = {
            "训练数据积累期": {
                "开始": "20230601",
                "结束": "20230930"
            },
            "参数优化期": {
                "开始": "20231001", 
                "结束": "20231231"
            },
            "主要回测期": {
                "开始": "20240101",
                "结束": "20240930"
            },
            "未来验证期": {
                "开始": "20241001",
                "结束": "20250301"
            }
        }
        
        # 调整日期为有效交易日（向后调整到下一个交易日）
        trading_periods = {}
        for period_name, period_dates in plan_dates.items():
            start_date = period_dates["开始"]
            end_date = period_dates["结束"]
            
            # 确保使用有效的交易日
            valid_start = None
            valid_end = None
            
            # 找到大于等于计划开始日期的第一个交易日
            for date in model.trading_dates:
                if date >= start_date and valid_start is None:
                    valid_start = date
                    
                # 找到小于等于计划结束日期的最后一个交易日    
                if date <= end_date:
                    valid_end = date
                elif valid_end:
                    # 已经找到结束日期且当前日期已超过计划结束日期，可以退出循环
                    break
            
            # 存储有效的交易期间
            if valid_start and valid_end:
                trading_periods[period_name] = {
                    "开始": valid_start,
                    "结束": valid_end
                }
                logger.info(f"{period_name}: 计划日期 {start_date}-{end_date}，调整为实际交易日 {valid_start}-{valid_end}")
            else:
                logger.error(f"{period_name}: 计划日期 {start_date}-{end_date} 没有对应的有效交易日")
                return
        
        # 时间规划说明
        logger.info("\n======= 时间规划说明 =======")
        logger.info(f"可用数据范围: {model.trading_dates[0]} 至 {model.trading_dates[-1]}")
        logger.info(f"阶段1 - 训练数据积累期: {trading_periods['训练数据积累期']['开始']} 至 {trading_periods['训练数据积累期']['结束']}")
        logger.info(f"阶段2 - 参数优化期: {trading_periods['参数优化期']['开始']} 至 {trading_periods['参数优化期']['结束']}")
        logger.info(f"阶段3 - 主要回测期: {trading_periods['主要回测期']['开始']} 至 {trading_periods['主要回测期']['结束']}")
        logger.info(f"阶段4 - 未来验证期: {trading_periods['未来验证期']['开始']} 至 {trading_periods['未来验证期']['结束']}")
        
        # 阶段1: 训练数据积累期
        logger.info("\n===== 阶段1: 训练数据积累期 =====")
        train_stage1_result = train_model(
            model, 
            start_date=trading_periods['训练数据积累期']['开始'], 
            end_date=trading_periods['训练数据积累期']['结束']
        )
        
        if not train_stage1_result or not train_stage1_result.get('success', False):
            logger.error("训练数据积累期失败，无法继续后续阶段")
            return
        
        # 阶段2: 参数优化期
        logger.info("\n===== 阶段2: 参数优化期 =====")
        best_params = train_model(
            model, 
            start_date=trading_periods['参数优化期']['开始'], 
            end_date=trading_periods['参数优化期']['结束'],
            is_parameter_optimization=True
        )
        
        if not best_params:
            logger.warning("参数优化未找到最佳参数，将使用默认参数")
        
        # 阶段3: 主要回测期
        logger.info("\n===== 阶段3: 主要回测期 =====")
        initial_capital = 1000000  # 初始资金100万
        
        # 运行带有滚动训练的回测
        if enable_parallel:
            # 使用并行处理
            logger.info("使用并行处理加速回测分析")
            parallel_params = {
                'interval_days': 30,
                'position_size': 0.1,
                'max_positions': 10
            }
            backtest_engine = run_analysis_parallel(
                start_date=trading_periods['主要回测期']['开始'],
                end_date=trading_periods['主要回测期']['结束'],
                initial_capital=initial_capital, 
                params=parallel_params
            )
            if backtest_engine is None:
                logger.error("并行回测执行失败，无法继续后续阶段")
                return
            
            report = backtest_engine.generate_report() if hasattr(backtest_engine, 'generate_report') else None
        else:
            # 使用原有流程
            backtest_result = run_backtest_with_rolling_training(
                start_date=trading_periods['主要回测期']['开始'],
                end_date=trading_periods['主要回测期']['结束'],
                initial_capital=initial_capital,
                long_cycle_update=10,  # 每10天更新一次长周期模型
                short_cycle_update=1   # 每天更新短周期模型
            )
        
            if not backtest_result:
                logger.error("主要回测期执行失败，无法继续后续阶段")
                return
            
            backtest_engine, report = backtest_result
        
        # 步骤3: 分析回测结果
        logger.info("\n===== 分析回测结果 =====")
        
        # 创建分析器实例
        analyzer = BacktestAnalyzer('backtest_results.pkl')
        
        # 并行加载基准数据和计算随机投资组合表现
        logger.info("并行加载基准数据和计算随机投资组合...")
        benchmarks = ['000001.SH', '399001.SZ', '399006.SZ']
        load_benchmark_and_calculate_random(analyzer, benchmarks, num_trials=5)
        
        # 并行生成分析图表和报告
        logger.info("并行生成分析图表和报告...")
        report = generate_analysis_reports(analyzer)
        
        # 阶段4: 未来验证期(可选)
        logger.info("\n===== 阶段4: 未来验证期(可选) =====")
        run_future = input("是否执行未来验证期回测? (y/n): ").strip().lower()
        
        if run_future == 'y':
            # 运行未来验证期回测
            if enable_parallel:
                future_engine = run_analysis_parallel(
                    start_date=trading_periods['未来验证期']['开始'],
                    end_date=trading_periods['未来验证期']['结束'],
                    initial_capital=initial_capital, 
                    params=parallel_params
                )
                future_report = future_engine.generate_report() if future_engine else None
            else:
                future_result = run_backtest_with_rolling_training(
                    start_date=trading_periods['未来验证期']['开始'],
                    end_date=trading_periods['未来验证期']['结束'],
                    initial_capital=initial_capital,
                    long_cycle_update=10,
                    short_cycle_update=1
                )
                
                if future_result:
                    future_engine, future_report = future_result
            
            if future_engine:
                # 比较两个时期的性能
                logger.info("\n===== 两个回测时期表现对比 =====")
                logger.info(f"主要回测期总收益率: {backtest_engine.total_return:.2%}")
                logger.info(f"未来验证期总收益率: {future_engine.total_return:.2%}")
                logger.info(f"主要回测期年化收益率: {backtest_engine.annual_return:.2%}")
                logger.info(f"未来验证期年化收益率: {future_engine.annual_return:.2%}")
                logger.info(f"主要回测期夏普比率: {backtest_engine.sharpe_ratio:.2f}")
                logger.info(f"未来验证期夏普比率: {future_engine.sharpe_ratio:.2f}")
        
        logger.info("===============================")
        logger.info("量化交易研究流程完成")
        logger.info("===============================")
        
        # 输出摘要结果
        if backtest_engine:
            logger.info("\n=== 回测结果摘要 ===")
            logger.info(f"总收益率: {backtest_engine.total_return:.2%}")
            logger.info(f"年化收益率: {backtest_engine.annual_return:.2%}")
            logger.info(f"最大回撤: {backtest_engine.max_drawdown:.2%}")
            logger.info(f"夏普比率: {backtest_engine.sharpe_ratio:.2f}")
            logger.info(f"胜率: {backtest_engine.win_rate:.2%}")
        
        # 生成增量训练性能报告(如果可用)
        if incremental_integration is not None:
            try:
                logger.info("\n=== 增量训练性能报告 ===")
                perf_stats = incremental_integration.generate_performance_report()
                
                if perf_stats.get('avg_full_time', 0) > 0 and perf_stats.get('avg_incremental_time', 0) > 0:
                    speedup = perf_stats['avg_full_time'] / perf_stats['avg_incremental_time']
                    logger.info(f"训练速度提升: {speedup:.2f}倍")
                    logger.info(f"节省训练时间: {perf_stats.get('time_saved', 0):.2f}秒 ({perf_stats.get('time_saved_pct', 0):.2f}%)")
                
                logger.info("增量训练系统使用说明请参考: to/增量训练系统使用说明.md")
            except Exception as e:
                logger.warning(f"生成增量训练性能报告时出错: {e}")
        
        # 生成向量化特征提取性能报告(如果可用)
        if vectorized_integration is not None:
            try:
                logger.info("\n=== 向量化特征提取性能报告 ===")
                vec_perf_stats = vectorized_integration.generate_performance_report()
                
                if vec_perf_stats.get('avg_standard_time', 0) > 0 and vec_perf_stats.get('avg_vectorized_time', 0) > 0:
                    vec_speedup = vec_perf_stats['avg_standard_time'] / vec_perf_stats['avg_vectorized_time']
                    logger.info(f"特征提取速度提升: {vec_speedup:.2f}倍")
                    logger.info(f"节省特征提取时间: {vec_perf_stats.get('time_saved', 0):.2f}秒 ({vec_perf_stats.get('time_saved_pct', 0):.2f}%)")
                
                logger.info("向量化特征提取系统使用说明请参考: to/向量化特征提取使用说明.md")
            except Exception as e:
                logger.warning(f"生成向量化特征提取性能报告时出错: {e}")
        
        # 生成并行回测性能报告(如果可用)
        if HAS_PARALLEL_ENGINE and enable_parallel:
            try:
                logger.info("\n=== 并行回测性能报告 ===")
                # 确保parallel_engine存在或创建一个新的实例
                parallel_engine = None
                if 'parallel_engine' not in locals() or parallel_engine is None:
                    try:
                        from parallel_backtest import ParallelBacktestEngine
                        parallel_engine = ParallelBacktestEngine(backend="auto")
                    except Exception as e:
                        logger.warning(f"无法初始化并行回测引擎: {e}")
                        # 跳过后续处理
                        pass
                
                # 只有当parallel_engine成功初始化时才继续处理
                if parallel_engine:
                    logger.info(f"并行回测引擎类型: {parallel_engine.backend if hasattr(parallel_engine, 'backend') else '未知'}")
                    logger.info(f"工作进程/线程数: {parallel_engine.workers if hasattr(parallel_engine, 'workers') else multiprocessing.cpu_count()}")
                    
                    if hasattr(parallel_engine, 'performance_stats'):
                        speedup = parallel_engine.performance_stats.get('speedup_ratio', 0)
                        if speedup > 0:
                            logger.info(f"回测速度提升: {speedup:.2f}倍")
                            logger.info(f"信号生成时间: {parallel_engine.performance_stats.get('signal_generation_time', 0):.2f}秒")
                            logger.info(f"回测执行时间: {parallel_engine.performance_stats.get('backtest_time', 0):.2f}秒")
                
            except Exception as e:
                logger.warning(f"生成并行回测性能报告时出错: {e}")
        
        # 在处理结束前生成缓存性能报告
        if CACHE_AVAILABLE and cache_integration:
            try:
                logger.info("\n=== 数据缓存性能报告 ===")
                cache_report = cache_integration.generate_report()
                
                # 提取关键统计信息
                cache_stats = cache_integration.get_performance_report().get('cache_stats', {})
                api_perf = cache_integration.get_performance_report().get('function_performance', {})
                
                # 打印缓存效率信息
                hit_rate = cache_stats.get('hit_rate', 0)
                memory_usage = cache_stats.get('memory_usage_mb', 0)
                items_count = cache_stats.get('items_count', 0)
                
                logger.info(f"缓存命中率: {hit_rate:.2f}%")
                logger.info(f"内存使用: {memory_usage:.2f}MB")
                logger.info(f"缓存项数量: {items_count}")
                
                # 计算时间节省
                if 'get_market_data' in api_perf:
                    calls = api_perf['get_market_data'].get('calls', 0)
                    avg_time = api_perf['get_market_data'].get('avg_time', 0)
                    if calls > 0:
                        # 假设原始API调用平均需要200ms
                        estimated_saved = calls * (0.2 - avg_time)
                        if estimated_saved > 0:
                            logger.info(f"估计节省API调用时间: {estimated_saved:.2f}秒")
                
                logger.info("数据缓存系统使用说明请参考: to/数据缓存系统使用说明.md")
            except Exception as e:
                logger.warning(f"生成缓存性能报告时出错: {e}")
        
        # 在函数结束前关闭缓存系统
        if CACHE_AVAILABLE and cache_integration:
            try:
                restore_original_functions()
                logger.info("数据缓存系统已关闭")
            except Exception as e:
                logger.warning(f"关闭缓存系统时出错: {e}")
        
        # 关闭Ray (如果已初始化)
        if ray_initialized:
            ray.shutdown()
            
    except Exception as e:
        logger.exception(f"执行过程中出错: {e}")
        
        # 确保关闭缓存系统
        if CACHE_AVAILABLE and cache_integration:
            try:
                restore_original_functions()
            except:
                pass
        
        # 关闭Ray (如果已初始化)
        if ray_initialized:
            ray.shutdown()

if __name__ == "__main__":
    main() 