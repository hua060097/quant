#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特征提取优化脚本
演示如何使用向量化特征提取加速回测和模型训练过程
"""

import os
import sys
import time
import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/optimize_features.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optimize_features")

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入向量化集成模块
try:
    from vectorized_integration import get_vectorized_integration
    logger.info("成功导入向量化集成模块")
except ImportError as e:
    logger.error(f"导入向量化集成模块失败: {e}")
    logger.error("请确保vectorized_integration.py文件位于当前目录下")
    raise

def run_benchmark_test(start_date, end_date, stock_count=100, lookback_days=30, iterations=5):
    """
    运行基准测试，比较原始特征提取和向量化特征提取的性能
    
    参数:
        start_date: 开始日期，格式'YYYYMMDD'
        end_date: 结束日期，格式'YYYYMMDD'
        stock_count: 测试股票数量
        lookback_days: 回溯天数
        iterations: 测试迭代次数
    """
    logger.info("\n===== 开始特征提取基准测试 =====")
    logger.info(f"日期范围: {start_date} 至 {end_date}")
    logger.info(f"测试股票数量: {stock_count}只")
    logger.info(f"回溯天数: {lookback_days}天")
    logger.info(f"测试迭代次数: {iterations}次")
    
    # 导入高级量化模型
    try:
        from advanced_quant_model import AdvancedQuantModel
        
        # 创建模型实例并加载基础数据
        model = AdvancedQuantModel()
        model.load_trading_dates()
        
        # 获取股票列表
        all_stocks = model.get_stock_universe()
        logger.info(f"获取股票列表成功，共 {len(all_stocks)} 只股票")
        
        # 选择测试股票
        test_stocks = all_stocks[:stock_count] if len(all_stocks) > stock_count else all_stocks
        logger.info(f"选择测试股票: {len(test_stocks)} 只")
        
        # 获取一个有效的交易日期
        closest_date = model._get_closest_trading_date(end_date)
        logger.info(f"使用交易日期: {closest_date}")
        
        # 原始方法测试
        original_times = []
        logger.info("\n1. 测试原始特征提取方法...")
        for i in range(iterations):
            start_time = time.time()
            features, labels = model._extract_features(test_stocks, closest_date, lookback_days)
            elapsed = time.time() - start_time
            original_times.append(elapsed)
            logger.info(f"  迭代 {i+1}/{iterations}: 耗时 {elapsed:.4f}秒, 特征形状: {features.shape if features is not None else None}")
        
        mean_original_time = np.mean(original_times)
        logger.info(f"原始方法平均耗时: {mean_original_time:.4f}秒")
        
        # 集成向量化特征提取
        logger.info("\n2. 集成向量化特征提取...")
        vectorized_integration = get_vectorized_integration(benchmark_mode=True)
        vectorized_integration.integrate(
            target_class=None,
            target_methods=['_extract_features'], 
            debug_mode=True
        )
        
        # 向量化方法测试
        vectorized_times = []
        logger.info("\n3. 测试向量化特征提取方法...")
        for i in range(iterations):
            start_time = time.time()
            features, labels = model._extract_features(test_stocks, closest_date, lookback_days)
            elapsed = time.time() - start_time
            vectorized_times.append(elapsed)
            logger.info(f"  迭代 {i+1}/{iterations}: 耗时 {elapsed:.4f}秒, 特征形状: {features.shape if features is not None else None}")
        
        mean_vectorized_time = np.mean(vectorized_times)
        logger.info(f"向量化方法平均耗时: {mean_vectorized_time:.4f}秒")
        
        # 恢复原始方法
        logger.info("\n4. 恢复原始方法...")
        vectorized_integration.restore()
        
        # 计算加速比
        speedup = mean_original_time / mean_vectorized_time if mean_vectorized_time > 0 else float('inf')
        time_saved_pct = (1 - mean_vectorized_time / mean_original_time) * 100 if mean_original_time > 0 else 0
        
        logger.info("\n===== 性能对比结果 =====")
        logger.info(f"原始方法平均耗时: {mean_original_time:.4f}秒")
        logger.info(f"向量化方法平均耗时: {mean_vectorized_time:.4f}秒")
        logger.info(f"加速比: {speedup:.2f}倍")
        logger.info(f"节省时间: {time_saved_pct:.2f}%")
        
        # 绘制性能对比图
        _plot_performance_comparison(original_times, vectorized_times)
        
        return {
            'original_times': original_times,
            'vectorized_times': vectorized_times,
            'mean_original_time': mean_original_time,
            'mean_vectorized_time': mean_vectorized_time,
            'speedup': speedup,
            'time_saved_pct': time_saved_pct
        }
        
    except Exception as e:
        logger.error(f"基准测试过程中出现异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    finally:
        logger.info("===== 特征提取基准测试结束 =====")

def run_scaling_test(start_date, end_date, stock_counts=[50, 100, 200, 500]):
    """
    运行扩展性测试，测试随着股票数量增加的性能变化
    
    参数:
        start_date: 开始日期，格式'YYYYMMDD'
        end_date: 结束日期，格式'YYYYMMDD'
        stock_counts: 要测试的股票数量列表
    """
    logger.info("\n===== 开始特征提取扩展性测试 =====")
    logger.info(f"日期范围: {start_date} 至 {end_date}")
    logger.info(f"测试股票数量: {stock_counts}只")
    
    # 存储测试结果
    results = {
        'stock_counts': stock_counts,
        'original_times': [],
        'vectorized_times': [],
        'speedups': []
    }
    
    # 导入高级量化模型
    try:
        from advanced_quant_model import AdvancedQuantModel
        
        # 创建模型实例并加载基础数据
        model = AdvancedQuantModel()
        model.load_trading_dates()
        
        # 获取股票列表
        all_stocks = model.get_stock_universe()
        logger.info(f"获取股票列表成功，共 {len(all_stocks)} 只股票")
        
        # 获取一个有效的交易日期
        closest_date = model._get_closest_trading_date(end_date)
        logger.info(f"使用交易日期: {closest_date}")
        
        # 集成向量化特征提取
        vectorized_integration = get_vectorized_integration(benchmark_mode=False)
        
        # 对每个股票数量进行测试
        for stock_count in stock_counts:
            # 选择测试股票
            test_stocks = all_stocks[:stock_count] if len(all_stocks) >= stock_count else all_stocks
            actual_count = len(test_stocks)
            logger.info(f"\n测试 {actual_count} 只股票:")
            
            # 测试原始方法
            logger.info("1. 测试原始特征提取方法...")
            start_time = time.time()
            
            # 恢复原始方法
            if vectorized_integration._patched:
                vectorized_integration.restore()
                
            # 执行特征提取
            features, labels = model._extract_features(test_stocks, closest_date, 30)
            original_time = time.time() - start_time
            results['original_times'].append(original_time)
            logger.info(f"原始方法耗时: {original_time:.4f}秒")
            
            # 测试向量化方法
            logger.info("2. 测试向量化特征提取方法...")
            
            # 集成向量化方法
            vectorized_integration.integrate(
                target_class=None,
                target_methods=['_extract_features'], 
                debug_mode=True
            )
            
            start_time = time.time()
            # 执行特征提取
            features, labels = model._extract_features(test_stocks, closest_date, 30)
            vectorized_time = time.time() - start_time
            results['vectorized_times'].append(vectorized_time)
            logger.info(f"向量化方法耗时: {vectorized_time:.4f}秒")
            
            # 计算加速比
            speedup = original_time / vectorized_time if vectorized_time > 0 else float('inf')
            results['speedups'].append(speedup)
            logger.info(f"加速比: {speedup:.2f}倍")
        
        # 恢复原始方法
        if vectorized_integration._patched:
            vectorized_integration.restore()
        
        # 绘制扩展性测试结果
        _plot_scaling_test(results)
        
        return results
        
    except Exception as e:
        logger.error(f"扩展性测试过程中出现异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    finally:
        logger.info("===== 特征提取扩展性测试结束 =====")

def run_backtest_with_vectorized_features(start_date, end_date, initial_capital=1000000):
    """
    使用向量化特征提取运行回测
    
    参数:
        start_date: 回测开始日期，格式'YYYYMMDD'
        end_date: 回测结束日期，格式'YYYYMMDD'
        initial_capital: 初始资金
    """
    logger.info("\n===== 开始向量化特征提取回测 =====")
    logger.info(f"回测时间范围: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,}")
    
    # 计时开始
    start_time_total = time.time()
    
    # 集成向量化特征提取
    try:
        vectorized_integration = get_vectorized_integration(benchmark_mode=False)
        vectorized_integration.integrate(
            target_class=None,
            target_methods=['_extract_features'], 
            debug_mode=True
        )
        logger.info("向量化特征提取集成成功")
        
        # 导入必要的模块
        from run_backtest import run_backtest_with_params
        
        # 运行回测
        logger.info("\n运行回测...")
        start_time_backtest = time.time()
        
        engine, report = run_backtest_with_params(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            enable_rolling_training=True,
            long_cycle_update=10,
            short_cycle_update=1
        )
        
        elapsed_backtest = time.time() - start_time_backtest
        logger.info(f"回测执行耗时: {elapsed_backtest:.2f}秒")
        
        # 显示回测结果
        if engine:
            logger.info("\n回测结果:")
            logger.info(f"总收益率: {engine.total_return:.2%}")
            logger.info(f"年化收益率: {engine.annual_return:.2%}")
            logger.info(f"最大回撤: {engine.max_drawdown:.2%}")
            logger.info(f"夏普比率: {engine.sharpe_ratio:.2f}")
        else:
            logger.error("回测执行失败")
        
        # 显示向量化特征提取性能报告
        logger.info("\n向量化特征提取性能报告:")
        vectorized_integration.print_performance_report()
        
        return engine, report, elapsed_backtest
        
    except Exception as e:
        logger.error(f"向量化特征提取回测过程中出现异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, 0
    
    finally:
        # 恢复原始方法
        if 'vectorized_integration' in locals() and vectorized_integration._patched:
            vectorized_integration.restore()
            logger.info("已恢复原始特征提取方法")
        
        # 计算总耗时
        elapsed_total = time.time() - start_time_total
        logger.info(f"总耗时: {elapsed_total:.2f}秒")
        logger.info("===== 向量化特征提取回测结束 =====")

def _plot_performance_comparison(original_times, vectorized_times):
    """绘制性能对比图"""
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    labels = ['原始方法', '向量化方法']
    mean_times = [np.mean(original_times), np.mean(vectorized_times)]
    bars = plt.bar(labels, mean_times, color=['#1f77b4', '#2ca02c'])
    
    # 添加标准差误差线
    std_times = [np.std(original_times), np.std(vectorized_times)]
    plt.errorbar(labels, mean_times, yerr=std_times, fmt='none', capsize=5, color='black')
    
    # 添加数据标签
    for bar, time in zip(bars, mean_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time:.4f}秒', ha='center', va='bottom')
    
    # 设置图表标题和标签
    plt.title('特征提取方法性能对比')
    plt.ylabel('执行时间 (秒)')
    plt.ylim(0, max(mean_times) * 1.2)  # 留出标签空间
    
    # 添加加速比文本
    speedup = mean_times[0] / mean_times[1] if mean_times[1] > 0 else float('inf')
    time_saved_pct = (1 - mean_times[1] / mean_times[0]) * 100 if mean_times[0] > 0 else 0
    plt.figtext(0.5, 0.01, 
                f"加速比: {speedup:.2f}倍 | 节省时间: {time_saved_pct:.2f}%", 
                ha="center", fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
    
    # 保存图表
    output_dir = 'to/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/feature_extraction_comparison_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"性能对比图已保存至 {output_file}")
    
    plt.close()

def _plot_scaling_test(results):
    """绘制扩展性测试结果图"""
    plt.figure(figsize=(12, 8))
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制执行时间对比
    ax1.plot(results['stock_counts'], results['original_times'], 'o-', label='原始方法', color='#1f77b4')
    ax1.plot(results['stock_counts'], results['vectorized_times'], 'o-', label='向量化方法', color='#2ca02c')
    ax1.set_title('特征提取方法扩展性测试 - 执行时间')
    ax1.set_xlabel('股票数量')
    ax1.set_ylabel('执行时间 (秒)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 绘制加速比
    ax2.plot(results['stock_counts'], results['speedups'], 'o-', color='#ff7f0e')
    ax2.set_title('特征提取方法扩展性测试 - 加速比')
    ax2.set_xlabel('股票数量')
    ax2.set_ylabel('加速比')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 添加横向线表示1倍速
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = 'to/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/feature_extraction_scaling_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"扩展性测试图已保存至 {output_file}")
    
    plt.close()

def main():
    """主函数"""
    print("\n请选择要执行的测试:")
    print("1. 基准性能测试 (对比原始方法和向量化方法)")
    print("2. 扩展性测试 (测试不同股票数量的性能变化)")
    print("3. 使用向量化特征提取运行完整回测")
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
            stock_count=200,  # 测试200只股票
            lookback_days=30,  # 30天的回溯期
            iterations=3       # 执行3次来取平均值
        )
    
    if choice == '2' or choice == '4':
        # 运行扩展性测试
        logger.info("\n执行扩展性测试...")
        scaling_results = run_scaling_test(
            start_date=start_date,
            end_date=end_date,
            stock_counts=[50, 100, 200, 400, 800]  # 测试不同数量的股票
        )
    
    if choice == '3' or choice == '4':
        # 使用向量化特征提取运行完整回测
        logger.info("\n使用向量化特征提取运行完整回测...")
        engine, report, elapsed = run_backtest_with_vectorized_features(
            start_date=start_date,
            end_date=end_date,
            initial_capital=1000000  # 初始资金100万
        )
    
    logger.info("\n所有测试完成")

if __name__ == "__main__":
    main() 