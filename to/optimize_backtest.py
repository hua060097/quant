#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测优化脚本
演示如何使用缓存系统优化回测过程，加速数据获取
"""

import os
import sys
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/optimize_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optimize_backtest")

# 导入缓存集成模块
from cache_integration import (
    get_cache_integration, 
    patch_backtest_engine, 
    patch_quant_model,
    restore_all_patches
)

def run_optimized_backtest(start_date, end_date, initial_capital=1000000, 
                          preload_data=True, memory_limit_gb=4):
    """
    运行优化后的回测
    
    参数:
        start_date: 回测开始日期，格式'YYYYMMDD'
        end_date: 回测结束日期，格式'YYYYMMDD'
        initial_capital: 初始资金
        preload_data: 是否预加载数据
        memory_limit_gb: 缓存内存限制(GB)
    """
    logger.info("\n===== 开始优化回测 =====")
    logger.info(f"回测时间范围: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,}")
    
    # 计时开始
    start_time_total = time.time()
    
    # 初始化缓存系统
    logger.info("\n1. 初始化缓存系统...")
    cache_integration = get_cache_integration(memory_limit_gb=memory_limit_gb)
    cache_integration.integrate()
    
    # 应用补丁到回测引擎和量化模型
    logger.info("\n2. 应用缓存补丁...")
    patch_backtest_engine()
    patch_quant_model()
    
    try:
        # 导入必要的模块
        from backtest_signals import BacktestEngine
        from advanced_quant_model import AdvancedQuantModel
        from run_backtest import run_backtest_with_params
        
        if preload_data:
            logger.info("\n3. 预加载数据...")
            # 创建模型实例以获取股票列表
            model = AdvancedQuantModel()
            model.load_trading_dates()
            
            # 获取股票列表
            stock_universe = model.get_stock_universe()
            logger.info(f"股票列表获取成功，共 {len(stock_universe)} 只股票")
            
            # 预加载常用字段的数据
            fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
            cache_integration.parallel_preload(
                stock_list=stock_universe,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
                periods=['1d']
            )
        
        # 运行回测
        logger.info("\n4. 运行回测...")
        start_time_backtest = time.time()
        
        engine, report = run_backtest_with_params(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        elapsed_backtest = time.time() - start_time_backtest
        logger.info(f"回测执行耗时: {elapsed_backtest:.2f}秒")
        
        # 显示回测结果
        if engine:
            logger.info("\n5. 回测结果:")
            logger.info(f"总收益率: {engine.total_return:.2%}")
            logger.info(f"年化收益率: {engine.annual_return:.2%}")
            logger.info(f"最大回撤: {engine.max_drawdown:.2%}")
            logger.info(f"夏普比率: {engine.sharpe_ratio:.2f}")
        else:
            logger.error("回测执行失败")
        
        # 显示缓存性能报告
        logger.info("\n6. 缓存性能报告:")
        cache_integration._report_performance()
        
        return engine, report, elapsed_backtest
        
    except Exception as e:
        logger.exception(f"优化回测过程中出现异常: {e}")
        return None, None, 0
    
    finally:
        # 恢复原始系统
        logger.info("\n7. 恢复原始系统...")
        restore_all_patches()
        
        # 计算总耗时
        elapsed_total = time.time() - start_time_total
        logger.info(f"总耗时: {elapsed_total:.2f}秒")
        logger.info("===== 优化回测结束 =====")

def run_standard_backtest(start_date, end_date, initial_capital=1000000):
    """
    运行标准回测（不使用缓存）
    
    参数:
        start_date: 回测开始日期，格式'YYYYMMDD'
        end_date: 回测结束日期，格式'YYYYMMDD'
        initial_capital: 初始资金
    """
    logger.info("\n===== 开始标准回测 (无缓存) =====")
    logger.info(f"回测时间范围: {start_date} 至 {end_date}")
    logger.info(f"初始资金: {initial_capital:,}")
    
    # 计时开始
    start_time_backtest = time.time()
    
    try:
        # 导入必要的模块
        from run_backtest import run_backtest_with_params
        
        # 运行回测
        logger.info("运行回测...")
        
        engine, report = run_backtest_with_params(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
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
        
        return engine, report, elapsed_backtest
        
    except Exception as e:
        logger.exception(f"标准回测过程中出现异常: {e}")
        return None, None, 0
    
    finally:
        logger.info("===== 标准回测结束 =====")

def compare_performance(standard_time, optimized_time):
    """比较标准回测和优化回测的性能差异"""
    if standard_time <= 0 or optimized_time <= 0:
        logger.warning("无效的时间数据，无法比较性能")
        return
    
    speedup = standard_time / optimized_time
    time_saved_pct = (standard_time - optimized_time) / standard_time * 100
    
    logger.info("\n===== 性能对比 =====")
    logger.info(f"标准回测耗时: {standard_time:.2f}秒")
    logger.info(f"优化回测耗时: {optimized_time:.2f}秒")
    logger.info(f"加速比: {speedup:.2f}x")
    logger.info(f"节省时间: {time_saved_pct:.2f}%")
    
    # 绘制性能对比图
    labels = ['标准回测', '优化回测']
    times = [standard_time, optimized_time]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['#1f77b4', '#2ca02c'])
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}秒', ha='center', va='bottom')
    
    plt.title('回测性能对比')
    plt.xlabel('回测类型')
    plt.ylabel('执行时间 (秒)')
    plt.ylim(0, max(times) * 1.2)  # 留出标签空间
    
    # 添加加速比和节省时间文本
    plt.figtext(0.5, 0.01, 
                f"优化后加速比: {speedup:.2f}倍 | 节省时间: {time_saved_pct:.2f}%", 
                ha="center", fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
    
    # 保存图表
    output_dir = 'to/results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/performance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    logger.info(f"性能对比图已保存至 {output_dir}/performance_comparison_{timestamp}.png")
    
    plt.close()

def main():
    """主函数"""
    # 可选项
    logger.info("\n是否进行性能比较测试(y/n)?")
    compare_test = input("").strip().lower() == 'y'
    
    # 设置回测参数
    initial_capital = 1000000  # 100万初始资金
    
    # 使用标准时间段
    # 基于用户笔记中的时间规划，选择一个主要回测区间
    start_date = '20240101'
    end_date = '20240401'  # 使用前3个月
    
    if compare_test:
        # 运行标准回测
        _, _, standard_time = run_standard_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        # 运行优化回测
        _, _, optimized_time = run_optimized_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            preload_data=True,  # 启用数据预加载
            memory_limit_gb=4   # 4GB内存缓存
        )
        
        # 比较性能
        compare_performance(standard_time, optimized_time)
    else:
        # 只运行优化版本
        run_optimized_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            preload_data=True,
            memory_limit_gb=4
        )
    
    logger.info("测试完成")

if __name__ == "__main__":
    main() 