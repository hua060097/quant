#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行回测系统脚本
用于启动回测系统和分析结果
"""

import os
import sys
import logging
import pickle
from datetime import datetime, timedelta
from backtest_signals import BacktestEngine
import time
import argparse
import pandas as pd
import numpy as np

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/run_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_backtest")

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

# 尝试导入并行回测引擎
try:
    from parallel_backtest import ParallelBacktestEngine
    PARALLEL_AVAILABLE = True
    logger.info("并行回测引擎可用，将自动选择最优回测模式")
except ImportError:
    PARALLEL_AVAILABLE = False
    logger.warning("未找到并行回测引擎，将使用标准回测流程")
    ParallelBacktestEngine = None

# 导入增量训练集成模块
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

# 导入向量化特征提取集成模块
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

def run_parallel_backtest(start_date, end_date, initial_capital=1000000, interval_days=30, **kwargs):
    """
    并行运行分段回测，最后合并结果
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        interval_days: 分段间隔天数
        **kwargs: 其他参数
    """
    logger.info(f"启动并行回测: {start_date} 到 {end_date}, 初始资金: {initial_capital:,.2f}")
    
    # 创建回测引擎
    main_engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        position_size=kwargs.get('position_size', 0.1),
        max_positions=kwargs.get('max_positions', 10)
    )
    
    # 先检查缓存集成是否可用，先进行集成再获取交易日历
    if CACHE_AVAILABLE:
        try:
            logger.info("初始化缓存系统集成...")
            cache_integration = integrate_cache_system()
            if cache_integration and cache_integration.patched:
                logger.info("缓存系统已成功集成")
            else:
                logger.warning("缓存系统集成失败，将使用直接API调用方式")
        except Exception as e:
            logger.error(f"初始化缓存系统时出错: {e}")
    
    # 获取交易日历
    from advanced_quant_model import AdvancedQuantModel
    model = AdvancedQuantModel()
    
    # 确保交易日期获取成功
    try:
        logger.info("加载交易日历...")
        model.load_trading_dates()
        
        if not model.trading_dates or len(model.trading_dates) == 0:
            logger.error("无法获取交易日历，将尝试直接从API获取")
            
            # 尝试直接从API获取
            end_date_obj = datetime.now().strftime('%Y%m%d')
            start_date_obj = (datetime.now() - timedelta(days=365*5)).strftime('%Y%m%d')
            
            try:
                import xtquant.xtdata as xtdata
                xtdata.connect()
                trading_dates = xtdata.get_trading_dates(market='SH', start_time=start_date_obj, end_time=end_date_obj)
                
                if trading_dates and len(trading_dates) > 0:
                    # 转换时间戳为日期字符串
                    model.trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
                    logger.info(f"成功直接从API获取了 {len(model.trading_dates)} 个交易日期")
                else:
                    logger.error("无法获取交易日历，退出回测")
                    return None, None
            except Exception as e:
                logger.error(f"尝试直接从API获取交易日历时出错: {e}")
                return None, None
    except Exception as e:
        logger.error(f"加载交易日历时出错: {e}")
        return None, None
        
    logger.info(f"成功加载交易日历，包含 {len(model.trading_dates)} 个交易日期")
    
    # 分段处理
    if start_date not in model.trading_dates or end_date not in model.trading_dates:
        logger.error(f"开始或结束日期不在交易日历中，可用范围: {model.trading_dates[0]} - {model.trading_dates[-1]}")
        return None, None
    
    start_idx = model.trading_dates.index(start_date)
    end_idx = model.trading_dates.index(end_date)
    
    # 初始化并行回测引擎
    parallel_engine = ParallelBacktestEngine(backend="auto")
    
    # 将整个回测区间分成几个子区间
    date_ranges = []
    for i in range(start_idx, end_idx, interval_days):
        sub_end_idx = min(i + interval_days - 1, end_idx)
        date_ranges.append((
            model.trading_dates[i],
            model.trading_dates[sub_end_idx]
        ))
    
    logger.info(f"将回测区间分成 {len(date_ranges)} 个子区间进行并行处理")
    
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
                target_class=model.__class__, 
                target_methods=['_extract_features'],
                debug_mode=True
            )
            if vectorized_success:
                logger.info("向量化特征提取功能已成功集成到量化模型中")
            else:
                logger.warning("向量化特征提取功能集成失败，将使用标准特征提取方法")
        except Exception as e:
            logger.warning(f"集成向量化特征提取功能时出错: {e}，将使用标准特征提取方法")
    
    # 并行生成交易信号
    signals = {}
    
    # 检查是否需要生成信号或加载已有信号
    if kwargs.get('signals_file'):
        # 从文件加载信号
        try:
            with open(kwargs['signals_file'], 'rb') as f:
                signals = pickle.load(f)
            logger.info(f"已从文件 {kwargs['signals_file']} 加载交易信号")
        except Exception as e:
            logger.error(f"加载信号文件时出错: {e}")
            return None, None
    elif kwargs.get('enable_rolling_training', False):
        # 使用滚动训练方式生成信号，调用回测引擎的方法
        logger.info("使用滚动训练方式生成信号")
        # 确保并行生成信号
        long_cycle_update = kwargs.get('long_cycle_update', 10)
        short_cycle_update = kwargs.get('short_cycle_update', 1)
        
        if not main_engine.load_rolling_signals(long_cycle_update, short_cycle_update):
            logger.error("滚动训练模式下生成交易信号失败，退出回测")
            return None, None
    else:
        # 并行生成所有日期的信号
        logger.info("开始并行生成所有日期的交易信号")
        signals = parallel_engine.parallel_generate_signals(model, date_ranges)
        if signals:
            main_engine.trading_signals = signals
            logger.info(f"成功生成 {len(signals)} 个日期的交易信号")
        else:
            logger.error("无法生成交易信号，退出回测")
            return None, None
    
    # 运行并行回测
    logger.info("开始执行并行回测")
    backtest_results = parallel_engine.run_parallel_backtest(main_engine, date_ranges)
    
    # 合并回测结果
    logger.info("合并并行回测结果")
    merged_engine, stats = parallel_engine.merge_backtest_results(backtest_results)
    
    # 生成回测报告
    logger.info("生成回测报告")
    report = merged_engine.generate_report()
    
    # 关闭并行引擎
    parallel_engine.shutdown()
    
    return merged_engine, report

def run_backtest_with_params(start_date, end_date, initial_capital=1000000, signals_file=None,
                          enable_rolling_training=False, long_cycle_update=10, short_cycle_update=1,
                          enable_parallel=True, parallel_interval_days=30):
    """
    使用指定参数运行回测
    
    参数:
        start_date: 回测开始日期，格式'YYYYMMDD'
        end_date: 回测结束日期，格式'YYYYMMDD'
        initial_capital: 初始资金
        signals_file: 信号文件路径，如果为None则由模型生成
        enable_rolling_training: 是否启用滚动训练模式
        long_cycle_update: 长周期模型更新间隔(天)
        short_cycle_update: 短周期模型更新间隔(天)
        enable_parallel: 是否启用并行回测
        parallel_interval_days: 并行回测的区间天数
    """
    logger.info(f"启动回测: {start_date} 到 {end_date}, 初始资金: {initial_capital:,.2f}")
    
    # 检查是否支持并行回测
    can_use_parallel = HAS_PARALLEL_ENGINE and enable_parallel
    
    if can_use_parallel:
        # 使用并行回测
        logger.info("使用并行回测引擎进行回测")
        engine, report = run_parallel_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            interval_days=parallel_interval_days,
            signals_file=signals_file,
            enable_rolling_training=enable_rolling_training,
            long_cycle_update=long_cycle_update,
            short_cycle_update=short_cycle_update,
            position_size=0.1,
            max_positions=10
        )
        
        return engine
    else:
        # 使用原来的顺序回测
        if not HAS_PARALLEL_ENGINE and enable_parallel:
            logger.warning("未能加载并行回测引擎，使用标准回测方式")
            
        # 原有的回测逻辑
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_size=0.1,  # 单个股票仓位比例
            max_positions=10  # 最大持仓数量
        )
        
        # 加载或生成交易信号
        logger.info("加载交易信号")
        if enable_rolling_training:
            logger.info("启用滚动训练模式")
            
            # 为BacktestEngine的模型集成增量训练功能
            if incremental_integration is not None:
                try:
                    from advanced_quant_model import AdvancedQuantModel
                    logger.info("正在集成增量训练功能到量化模型...")
                    integration_success = incremental_integration.integrate(AdvancedQuantModel)
                    if integration_success:
                        logger.info("增量训练功能已成功集成到量化模型中")
                    else:
                        logger.warning("增量训练功能集成失败，将使用标准训练方法")
                except Exception as e:
                    logger.warning(f"集成增量训练功能时出错: {e}，将使用标准训练方法")
            
            # 集成向量化特征提取功能
            if vectorized_integration is not None:
                try:
                    from advanced_quant_model import AdvancedQuantModel
                    logger.info("正在集成向量化特征提取功能到量化模型...")
                    # 确保集成特别处理_extract_features方法
                    vectorized_success = vectorized_integration.integrate(
                        target_class=AdvancedQuantModel,
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
            
            if not engine.load_rolling_signals(long_cycle_update, short_cycle_update):
                logger.error("滚动训练模式下生成交易信号失败，退出回测")
                return None
        elif not engine.load_trading_signals(signals_file):
            logger.error("无法加载或生成交易信号，退出回测")
            return None
        
        # 运行回测
        logger.info("开始执行回测")
        if not engine.run_backtest():
            logger.error("回测执行失败")
            return None
        
        # 生成报告和图表
        logger.info("生成回测报告")
        report = engine.generate_report()
        
        return engine

def compare_signals_effectiveness():
    """比较不同信号类型的有效性"""
    logger.info("开始分析不同信号类型的有效性")
    
    # 检查是否有保存的回测结果
    if not os.path.exists('backtest_results.pkl'):
        logger.error("找不到回测结果文件，请先运行回测")
        return
    
    # 加载回测结果
    with open('backtest_results.pkl', 'rb') as f:
        engine = pickle.load(f)
    
    # 分析不同信号类型的表现
    signal_types = {}
    
    for trade in engine.trade_history:
        if trade['action'] == 'buy':
            # 记录买入信号
            stock_code = trade['stock_code']
            reason = trade['reason']
            
            if reason not in signal_types:
                signal_types[reason] = {
                    'count': 0,
                    'win': 0,
                    'loss': 0,
                    'total_profit': 0,
                    'total_loss': 0
                }
            
            signal_types[reason]['count'] += 1
            
            # 查找对应的卖出记录
            sell_trade = None
            for t in engine.trade_history:
                if t['action'] == 'sell' and t['stock_code'] == stock_code and t['date'] > trade['date']:
                    sell_trade = t
                    break
            
            if sell_trade:
                # 计算盈亏
                buy_amount = trade['amount']
                sell_amount = sell_trade['amount']
                profit = sell_amount - buy_amount
                
                if profit > 0:
                    signal_types[reason]['win'] += 1
                    signal_types[reason]['total_profit'] += profit
                else:
                    signal_types[reason]['loss'] += 1
                    signal_types[reason]['total_loss'] += -profit
    
    # 计算每种信号类型的胜率和盈亏比
    logger.info("\n===== 信号类型分析 =====")
    for signal_type, stats in signal_types.items():
        win_rate = stats['win'] / stats['count'] if stats['count'] > 0 else 0
        avg_profit = stats['total_profit'] / stats['win'] if stats['win'] > 0 else 0
        avg_loss = stats['total_loss'] / stats['loss'] if stats['loss'] > 0 else 1
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        
        logger.info(f"\n信号类型: {signal_type}")
        logger.info(f"  交易次数: {stats['count']}")
        logger.info(f"  胜率: {win_rate:.2%}")
        logger.info(f"  平均盈利: {avg_profit:.2f}")
        logger.info(f"  平均亏损: {avg_loss:.2f}")
        logger.info(f"  盈亏比: {profit_loss_ratio:.2f}")
    
    logger.info("\n分析完成")

def run_future_validation(start_date='20241001', end_date='20250301', initial_capital=1000000,
                          enable_rolling_training=True, long_cycle_update=10, short_cycle_update=1):
    """
    执行未来验证期回测
    
    参数:
        start_date: 验证开始日期
        end_date: 验证结束日期
        initial_capital: 初始资金
        enable_rolling_training: 是否启用滚动训练
        long_cycle_update: 长周期更新间隔(天)
        short_cycle_update: 短周期更新间隔(天)
    """
    logger.info(f"执行未来验证期回测: {start_date} 至 {end_date}")
    
    # 运行未来验证期回测
    future_engine, future_report = run_backtest_with_params(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        enable_rolling_training=enable_rolling_training,
        long_cycle_update=long_cycle_update,
        short_cycle_update=short_cycle_update
    )
    
    if future_engine and future_report:
        # 保存回测结果
        with open('future_validation_results.pkl', 'wb') as f:
            pickle.dump(future_engine, f)
        logger.info("未来验证期回测结果已保存至 future_validation_results.pkl")
        
        # 对比两个时期的表现
        if os.path.exists('backtest_results.pkl'):
            logger.info("\n===== 两个回测时期表现对比 =====")
            
            with open('backtest_results.pkl', 'rb') as f:
                main_engine = pickle.load(f)
                
            logger.info(f"主要回测期总收益率: {main_engine.total_return:.2%}")
            logger.info(f"未来验证期总收益率: {future_engine.total_return:.2%}")
            logger.info(f"主要回测期年化收益率: {main_engine.annual_return:.2%}")
            logger.info(f"未来验证期年化收益率: {future_engine.annual_return:.2%}")
            logger.info(f"主要回测期夏普比率: {main_engine.sharpe_ratio:.2f}")
            logger.info(f"未来验证期夏普比率: {future_engine.sharpe_ratio:.2f}")
            
            # 计算性能差异百分比
            return_diff = (future_engine.total_return - main_engine.total_return) / abs(main_engine.total_return) if main_engine.total_return != 0 else float('inf')
            annual_return_diff = (future_engine.annual_return - main_engine.annual_return) / abs(main_engine.annual_return) if main_engine.annual_return != 0 else float('inf')
            sharpe_diff = (future_engine.sharpe_ratio - main_engine.sharpe_ratio) / abs(main_engine.sharpe_ratio) if main_engine.sharpe_ratio != 0 else float('inf')
            
            logger.info(f"\n性能变化:")
            logger.info(f"总收益率变化: {return_diff:.2%}")
            logger.info(f"年化收益率变化: {annual_return_diff:.2%}")
            logger.info(f"夏普比率变化: {sharpe_diff:.2%}")
            
            # 比较信号有效性变化
            logger.info("\n主要回测期与未来验证期信号有效性对比分析将保存到signal_comparison_periods.csv")
            
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
            
            return future_engine, future_report, {
                'return_diff': return_diff,
                'annual_return_diff': annual_return_diff,
                'sharpe_diff': sharpe_diff
            }
        else:
            logger.warning("找不到主要回测期的结果文件，无法进行对比分析")
            return future_engine, future_report, None
    else:
        logger.error("未来验证期回测失败")
        return None, None, None

def main():
    """主函数"""
    logger.info("===== 开始回测分析 =====")
    
    # 初始化缓存系统(如果可用)
    if CACHE_AVAILABLE:
        try:
            logger.info("初始化缓存系统...")
            integration = integrate_cache_system()
            if integration and integration.patched:
                logger.info("缓存系统已成功集成，将优化数据加载性能")
            else:
                logger.warning("缓存系统集成失败或未完成，将使用直接API调用方式")
        except Exception as e:
            logger.error(f"初始化缓存系统时出错: {e}")
            logger.exception(e)
    
    # 数据有效范围: 2023年6月1日至2025年3月1日
    # 基于可用数据，优化规划回测时间段
    
    # 1. 训练数据积累期: 2023年6月1日至2023年9月30日
    # 2. 参数优化期: 2023年10月1日至2023年12月31日
    # 3. 主要回测期: 2024年1月1日至2024年9月30日
    # 4. 未来验证期: 2024年10月1日至2025年3月1日
    
    # 确认是否需要运行完整流程
    logger.info("\n是否运行完整的量化交易研究流程? 这将经历训练、优化、回测和验证四个阶段(y/n):")
    run_full_process = input("").strip().lower() == 'y'
    
    if run_full_process:
        logger.info("将运行完整研究流程，这可能需要较长时间...")
        # 完整流程将由run_analysis.py处理
        from run_analysis import main as run_full_analysis
        run_full_analysis()
        return
    
    # 用户选择单独运行回测
    logger.info("\n请选择要运行的回测时期:")
    logger.info("1. 主要回测期 (20240101-20240930)")
    logger.info("2. 未来验证期 (20241001-20250301)")
    logger.info("3. 自定义时期")
    
    choice = input("请输入选项编号(1/2/3): ").strip()
    
    # 设置回测参数
    initial_capital = 1000000
    enable_rolling = input("是否启用滚动训练模式(y/n): ").strip().lower() == 'y'
    
    if choice == '1':
        # 执行主要回测期的分析
        start_date = '20240101'
        end_date = '20240930'
        
        logger.info(f"执行主要回测期分析: {start_date} 至 {end_date}")
        logger.info("该时间段有足够的历史数据进行模型训练，且留有足够的未来数据进行表现验证")
        
    elif choice == '2':
        # 执行未来验证期回测
        start_date = '20241001'
        end_date = '20250301'
        
        logger.info(f"执行未来验证期回测: {start_date} 至 {end_date}")
        
    elif choice == '3':
        # 自定义时期
        start_date = input("请输入开始日期(YYYYMMDD格式): ").strip()
        end_date = input("请输入结束日期(YYYYMMDD格式): ").strip()
        
        logger.info(f"执行自定义时期回测: {start_date} 至 {end_date}")
        
    else:
        logger.error("无效的选项，退出回测")
        return
    
    if enable_rolling:
        long_cycle = int(input("请输入长周期更新间隔(天数): ").strip() or "10")
        short_cycle = int(input("请输入短周期更新间隔(天数): ").strip() or "1")
        logger.info(f"滚动训练参数: 长周期更新={long_cycle}天, 短周期更新={short_cycle}天")
    else:
        long_cycle = 10
        short_cycle = 1
    
    # 添加并行回测选项
    if choice in ['1', '2', '3']:
        enable_parallel = input("是否启用并行回测加速(y/n): ").strip().lower() == 'y'
        
        if enable_parallel and HAS_PARALLEL_ENGINE:
            parallel_interval = int(input("请输入并行回测的区间天数(默认30天): ").strip() or "30")
            logger.info(f"启用并行回测，区间天数: {parallel_interval}")
        elif enable_parallel and not HAS_PARALLEL_ENGINE:
            logger.warning("未能加载并行回测引擎，将使用标准回测方式")
            enable_parallel = False
        else:
            logger.info("使用标准顺序回测方式")
            
        # 运行回测
        engine, report = run_backtest_with_params(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            enable_rolling_training=enable_rolling,
            long_cycle_update=long_cycle,
            short_cycle_update=short_cycle,
            enable_parallel=enable_parallel,
            parallel_interval_days=parallel_interval if enable_parallel else 30
        )
    
    if engine and report:
        # 保存回测引擎以便后续分析
        results_file = 'future_validation_results.pkl' if choice == '2' else 'backtest_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(engine, f)
        logger.info(f"回测结果已保存至 {results_file}")
        
        # 分析不同信号类型的有效性
        compare_signals_effectiveness()
        
        # 如果是未来验证期，询问是否与主要回测期进行对比
        if choice == '2' and os.path.exists('backtest_results.pkl'):
            if input("\n是否与主要回测期进行对比分析?(y/n): ").strip().lower() == 'y':
                run_future_validation(start_date, end_date, initial_capital, enable_rolling, long_cycle, short_cycle)
    
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
    
    logger.info("===== 回测分析结束 =====")

if __name__ == "__main__":
    main() 