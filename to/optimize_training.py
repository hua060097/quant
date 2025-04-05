#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化训练脚本
演示如何使用增量训练加速回测和模型训练过程
"""

import os
import sys
import time
import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/optimize_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optimize_training")

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入增量训练集成模块
try:
    from incremental_integration import get_incremental_integration
    logger.info("成功导入增量训练集成模块")
except ImportError as e:
    logger.error(f"导入增量训练集成模块失败: {e}")
    logger.error("请确保incremental_integration.py文件位于当前目录下")
    raise

def run_benchmark_test(start_date, end_date, sector_count=5, benchmark_mode=True):
    """
    运行基准测试，比较原始训练和增量训练的性能
    
    参数:
        start_date: 开始日期，格式'YYYYMMDD'
        end_date: 结束日期，格式'YYYYMMDD'
        sector_count: 测试行业数量
        benchmark_mode: 是否启用基准测试模式
    """
    logger.info("\n===== 开始模型训练基准测试 =====")
    logger.info(f"日期范围: {start_date} 至 {end_date}")
    logger.info(f"测试行业数量: {sector_count}个")
    
    # 导入高级量化模型
    try:
        from advanced_quant_model import AdvancedQuantModel
        
        # 创建模型实例
        model = AdvancedQuantModel()
        
        # 先进行一次原始训练作为基准
        logger.info("\n1. 使用原始训练方法...")
        original_start = time.time()
        
        # 训练模型
        model.train_sector_models(
            date=end_date,
            lookback_short=30,
            lookback_long=90,
            test_size=0.2,
            forward_days=5,
            parallel="auto"  # 自动选择并行方式
        )
        
        original_time = time.time() - original_start
        logger.info(f"原始训练方法总耗时: {original_time:.2f}秒")
        
        # 重置模型
        model = AdvancedQuantModel()
        
        # 创建并集成增量训练
        logger.info("\n2. 集成增量训练...")
        integration = get_incremental_integration(benchmark_mode=benchmark_mode)
        integration.integrate()
        logger.info("增量训练集成完成")
        
        # 使用增量训练方法
        logger.info("\n3. 使用增量训练方法...")
        incremental_start = time.time()
        
        # 训练模型（将自动使用增量训练）
        model.train_sector_models(
            date=end_date,
            lookback_short=30,
            lookback_long=90,
            test_size=0.2,
            forward_days=5,
            parallel="auto"
        )
        
        incremental_time = time.time() - incremental_start
        logger.info(f"增量训练方法总耗时: {incremental_time:.2f}秒")
        
        # 第二次增量训练（这次应该更快，因为利用了第一次的模型）
        logger.info("\n4. 第二次增量训练（应该更快）...")
        second_incremental_start = time.time()
        
        # 更改日期，模拟向前推进
        next_date = _get_next_trading_date(end_date)
        model.train_sector_models(
            date=next_date,
            lookback_short=30,
            lookback_long=90,
            test_size=0.2,
            forward_days=5,
            parallel="auto"
        )
        
        second_incremental_time = time.time() - second_incremental_start
        logger.info(f"第二次增量训练方法总耗时: {second_incremental_time:.2f}秒")
        
        # 恢复原始方法
        integration.restore()
        logger.info("\n5. 已恢复原始训练方法")
        
        # 计算加速比
        first_speedup = original_time / incremental_time if incremental_time > 0 else float('inf')
        second_speedup = original_time / second_incremental_time if second_incremental_time > 0 else float('inf')
        
        logger.info("\n===== 性能对比结果 =====")
        logger.info(f"原始训练方法耗时: {original_time:.2f}秒")
        logger.info(f"首次增量训练耗时: {incremental_time:.2f}秒")
        logger.info(f"第二次增量训练耗时: {second_incremental_time:.2f}秒")
        logger.info(f"首次加速比: {first_speedup:.2f}倍")
        logger.info(f"第二次加速比: {second_speedup:.2f}倍")
        
        # 打印详细的性能报告
        integration.print_performance_report()
        
        # 绘制性能对比图
        _plot_training_comparison(original_time, incremental_time, second_incremental_time)
        
        return {
            'original_time': original_time,
            'incremental_time': incremental_time,
            'second_incremental_time': second_incremental_time,
            'first_speedup': first_speedup,
            'second_speedup': second_speedup
        }
        
    except Exception as e:
        logger.error(f"基准测试过程中出现异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    finally:
        logger.info("===== 模型训练基准测试结束 =====")

def run_rolling_backtest_with_incremental(start_date, end_date, initial_capital=1000000,
                                        long_cycle_update=10, short_cycle_update=1):
    """
    使用增量训练运行滚动回测
    
    参数:
        start_date: 回测开始日期，格式'YYYYMMDD'
        end_date: 回测结束日期，格式'YYYYMMDD'
        initial_capital: 初始资金
        long_cycle_update: 长周期模型更新间隔(天)
        short_cycle_update: 短周期模型更新间隔(天)
    """
    logger.info("\n===== 开始增量训练滚动回测 =====")
    logger.info(f"回测时间范围: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,}")
    logger.info(f"长周期更新间隔: {long_cycle_update}天, 短周期更新间隔: {short_cycle_update}天")
    
    # 计时开始
    start_time_total = time.time()
    
    # 集成增量训练
    try:
        integration = get_incremental_integration(benchmark_mode=False)
        integration.integrate()
        logger.info("增量训练集成成功")
        
        # 导入必要的模块
        from run_backtest import run_backtest_with_params
        
        # 运行带有滚动训练的回测
        logger.info("\n运行滚动回测...")
        start_time_backtest = time.time()
        
        engine, report = run_backtest_with_params(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            enable_rolling_training=True,  # 启用滚动训练
            long_cycle_update=long_cycle_update,
            short_cycle_update=short_cycle_update
        )
        
        elapsed_backtest = time.time() - start_time_backtest
        logger.info(f"滚动回测执行耗时: {elapsed_backtest:.2f}秒")
        
        # 显示回测结果
        if engine:
            logger.info("\n回测结果:")
            logger.info(f"总收益率: {engine.total_return:.2%}")
            logger.info(f"年化收益率: {engine.annual_return:.2%}")
            logger.info(f"最大回撤: {engine.max_drawdown:.2%}")
            logger.info(f"夏普比率: {engine.sharpe_ratio:.2f}")
        else:
            logger.error("回测执行失败")
        
        # 显示增量训练性能报告
        logger.info("\n增量训练性能报告:")
        integration.print_performance_report()
        
        return engine, report, elapsed_backtest
        
    except Exception as e:
        logger.error(f"增量训练滚动回测过程中出现异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, 0
    
    finally:
        # 恢复原始方法
        if 'integration' in locals() and integration._patched:
            integration.restore()
            logger.info("已恢复原始训练方法")
        
        # 计算总耗时
        elapsed_total = time.time() - start_time_total
        logger.info(f"总耗时: {elapsed_total:.2f}秒")
        logger.info("===== 增量训练滚动回测结束 =====")

def compare_training_strategies(start_date, end_date, initial_capital=1000000):
    """
    比较不同训练策略的效果和性能
    
    参数:
        start_date: 回测开始日期，格式'YYYYMMDD'
        end_date: 回测结束日期，格式'YYYYMMDD'
        initial_capital: 初始资金
    """
    logger.info("\n===== 开始训练策略对比 =====")
    logger.info(f"回测时间范围: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,}")
    
    results = {}
    
    try:
        # 导入必要的模块
        from run_backtest import run_backtest_with_params
        
        # 1. 常规训练（每次全量训练）
        logger.info("\n1. 常规训练（每次全量训练）...")
        regular_start = time.time()
        
        regular_engine, regular_report = run_backtest_with_params(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            enable_rolling_training=True,
            long_cycle_update=10,
            short_cycle_update=5  # 较长的间隔，减少训练次数
        )
        
        regular_time = time.time() - regular_start
        logger.info(f"常规训练回测耗时: {regular_time:.2f}秒")
        
        if regular_engine:
            results['regular'] = {
                'time': regular_time,
                'total_return': regular_engine.total_return,
                'annual_return': regular_engine.annual_return,
                'max_drawdown': regular_engine.max_drawdown,
                'sharpe_ratio': regular_engine.sharpe_ratio
            }
        
        # 2. 集成增量训练
        logger.info("\n2. 集成增量训练...")
        integration = get_incremental_integration(benchmark_mode=False)
        integration.integrate()
        
        incremental_start = time.time()
        
        incremental_engine, incremental_report = run_backtest_with_params(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            enable_rolling_training=True,
            long_cycle_update=10,
            short_cycle_update=1  # 更短的间隔，增加训练次数
        )
        
        incremental_time = time.time() - incremental_start
        logger.info(f"增量训练回测耗时: {incremental_time:.2f}秒")
        
        if incremental_engine:
            results['incremental'] = {
                'time': incremental_time,
                'total_return': incremental_engine.total_return,
                'annual_return': incremental_engine.annual_return,
                'max_drawdown': incremental_engine.max_drawdown,
                'sharpe_ratio': incremental_engine.sharpe_ratio
            }
        
        # 恢复原始方法
        integration.restore()
        
        # 打印结果对比
        if 'regular' in results and 'incremental' in results:
            logger.info("\n===== 训练策略对比结果 =====")
            logger.info(f"常规训练回测耗时: {results['regular']['time']:.2f}秒")
            logger.info(f"增量训练回测耗时: {results['incremental']['time']:.2f}秒")
            
            speedup = results['regular']['time'] / results['incremental']['time'] if results['incremental']['time'] > 0 else float('inf')
            logger.info(f"时间节省: {speedup:.2f}倍")
            
            logger.info("\n性能对比:")
            logger.info(f"总收益率: 常规={results['regular']['total_return']:.2%}, 增量={results['incremental']['total_return']:.2%}")
            logger.info(f"年化收益率: 常规={results['regular']['annual_return']:.2%}, 增量={results['incremental']['annual_return']:.2%}")
            logger.info(f"最大回撤: 常规={results['regular']['max_drawdown']:.2%}, 增量={results['incremental']['max_drawdown']:.2%}")
            logger.info(f"夏普比率: 常规={results['regular']['sharpe_ratio']:.2f}, 增量={results['incremental']['sharpe_ratio']:.2f}")
            
            # 绘制性能对比图
            _plot_strategy_comparison(results)
        
        return results
        
    except Exception as e:
        logger.error(f"训练策略对比过程中出现异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    finally:
        logger.info("===== 训练策略对比结束 =====")

def _get_next_trading_date(date_str):
    """模拟获取下一个交易日，实际应该使用交易日历"""
    from datetime import datetime, timedelta
    
    # 将字符串转换为日期
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    
    # 简单地加1天，实际应该查询真实的交易日历
    next_date = date_obj + timedelta(days=1)
    
    # 避免周末
    while next_date.weekday() > 4:  # 5和6为周六和周日
        next_date += timedelta(days=1)
    
    # 转回字符串格式
    return next_date.strftime('%Y%m%d')

def _plot_training_comparison(original_time, incremental_time, second_incremental_time):
    """绘制训练时间对比图"""
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    labels = ['原始训练', '首次增量训练', '第二次增量训练']
    times = [original_time, incremental_time, second_incremental_time]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    bars = plt.bar(labels, times, color=colors)
    
    # 添加数据标签
    for bar, time in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time:.2f}秒', ha='center', va='bottom')
    
    # 计算和显示加速比
    first_speedup = original_time / incremental_time if incremental_time > 0 else float('inf')
    second_speedup = original_time / second_incremental_time if second_incremental_time > 0 else float('inf')
    
    # 设置图表标题和标签
    plt.title('训练方法性能对比')
    plt.ylabel('执行时间 (秒)')
    plt.ylim(0, max(times) * 1.2)  # 留出标签空间
    
    # 添加加速比文本
    plt.figtext(0.5, 0.01, 
                f"首次加速比: {first_speedup:.2f}倍 | 第二次加速比: {second_speedup:.2f}倍",
                ha="center", fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
    
    # 保存图表
    output_dir = 'to/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/training_comparison_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"训练性能对比图已保存至 {output_file}")
    
    plt.close()

def _plot_strategy_comparison(results):
    """绘制训练策略对比图"""
    # 确保有结果
    if 'regular' not in results or 'incremental' not in results:
        logger.warning("缺少对比数据，无法绘制图表")
        return
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 时间性能对比
    labels = ['常规训练', '增量训练']
    times = [results['regular']['time'], results['incremental']['time']]
    speedup = times[0] / times[1] if times[1] > 0 else float('inf')
    
    bars1 = ax1.bar(labels, times, color=['#1f77b4', '#2ca02c'])
    ax1.set_title('训练时间对比')
    ax1.set_ylabel('执行时间 (秒)')
    
    # 添加数据标签
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time:.2f}秒', ha='center', va='bottom')
    
    # 投资性能对比
    metrics = ['总收益率', '年化收益率', '夏普比率']
    regular_values = [
        results['regular']['total_return'] * 100,  # 转为百分比
        results['regular']['annual_return'] * 100,
        results['regular']['sharpe_ratio']
    ]
    incremental_values = [
        results['incremental']['total_return'] * 100,
        results['incremental']['annual_return'] * 100,
        results['incremental']['sharpe_ratio']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars2_1 = ax2.bar(x - width/2, regular_values, width, label='常规训练', color='#1f77b4')
    bars2_2 = ax2.bar(x + width/2, incremental_values, width, label='增量训练', color='#2ca02c')
    
    ax2.set_title('投资性能对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # 添加数据标签
    for bars, values in [(bars2_1, regular_values), (bars2_2, incremental_values)]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # 区分百分比和夏普比率的显示
            if i < 2:  # 总收益率和年化收益率
                label = f'{values[i]:.2f}%'
            else:  # 夏普比率
                label = f'{values[i]:.2f}'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    label, ha='center', va='bottom', fontsize=9)
    
    # 添加信息文本
    plt.figtext(0.5, 0.01, 
                f"时间节省: {speedup:.2f}倍",
                ha="center", fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 保存图表
    output_dir = 'to/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/strategy_comparison_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"策略对比图已保存至 {output_file}")
    
    plt.close()

def main():
    """主函数"""
    print("\n请选择要执行的测试:")
    print("1. 基准性能测试 (比较原始训练和增量训练)")
    print("2. 使用增量训练运行滚动回测")
    print("3. 比较不同训练策略的效果和性能")
    print("4. 全部测试")
    
    choice = input("\n请输入选项编号(1/2/3/4): ").strip()
    
    # 设置测试参数
    start_date = '20240101'  # 开始日期
    end_date = '20240131'    # 结束日期（使用一个月的时间减少测试时间）
    
    if choice == '1' or choice == '4':
        # 运行基准性能测试
        logger.info("\n执行基准性能测试...")
        benchmark_results = run_benchmark_test(
            start_date=start_date,
            end_date=end_date,
            sector_count=5,  # 测试5个行业
            benchmark_mode=True  # 启用基准测试模式
        )
    
    if choice == '2' or choice == '4':
        # 使用增量训练运行滚动回测
        logger.info("\n使用增量训练运行滚动回测...")
        engine, report, elapsed = run_rolling_backtest_with_incremental(
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000,  # 初始资金100万
            long_cycle_update=10,     # 长周期每10天更新
            short_cycle_update=1      # 短周期每天更新
        )
    
    if choice == '3' or choice == '4':
        # 比较不同训练策略的效果和性能
        logger.info("\n比较不同训练策略...")
        strategy_results = compare_training_strategies(
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000  # 初始资金100万
        )
    
    logger.info("\n所有测试完成")

if __name__ == "__main__":
    main() 