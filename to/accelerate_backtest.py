#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测系统Numba加速接口

使用方法：
python accelerate_backtest.py --bench   # 运行性能测试
python accelerate_backtest.py --run     # 使用加速运行回测
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/accelerate_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("accelerate_backtest")

def check_dependencies():
    """检查依赖项
    
    Returns:
        bool: 是否所有依赖都可用
    """
    missing_deps = []
    
    # 检查核心依赖
    try:
        import numpy
        logger.info(f"NumPy版本: {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
        logger.info(f"Pandas版本: {pandas.__version__}")
    except ImportError:
        missing_deps.append("pandas")
    
    # 检查Numba
    try:
        import numba
        logger.info(f"Numba版本: {numba.__version__}")
    except ImportError:
        missing_deps.append("numba")
        logger.warning("未安装Numba，无法使用加速功能")
    
    # 检查加速器模块
    try:
        from numba_accelerator import get_numba_accelerator
        logger.info("Numba加速器模块可用")
    except ImportError:
        missing_deps.append("numba_accelerator")
        logger.warning("未找到Numba加速器模块，请确保numba_accelerator.py在Python路径中")
    
    # 检查量化模型模块
    try:
        from advanced_quant_model import AdvancedQuantModel
        logger.info("高级量化模型模块可用")
    except ImportError:
        missing_deps.append("advanced_quant_model")
        logger.warning("未找到高级量化模型模块")
    
    if missing_deps:
        logger.error(f"缺少以下依赖: {', '.join(missing_deps)}")
        return False
    
    return True

def run_benchmark(test_size=100000, trials=3):
    """运行性能测试
    
    Args:
        test_size: 测试数据大小
        trials: 测试重复次数
    
    Returns:
        bool: 测试是否成功完成
    """
    logger.info(f"开始性能测试 (数据大小: {test_size}, 重复次数: {trials})")
    
    try:
        from advanced_quant_model import AdvancedQuantModel
        
        # 初始化模型，启用Numba加速
        model = AdvancedQuantModel(use_numba=True)
        
        # 运行性能测试
        results = model.benchmark_performance(test_size=test_size, trials=trials)
        
        # 打印测试结果
        print("\n" + "=" * 60)
        print("Numba加速性能测试结果")
        print("=" * 60)
        
        if "error" in results:
            logger.error(f"性能测试失败: {results['error']}")
            print(f"测试失败: {results['error']}")
            return False
        
        print("\n计算日内涨跌幅:")
        print(f"  原始版本: {results['intraday_range']['原始版本']:.6f} 秒")
        print(f"  Numba版本: {results['intraday_range']['Numba版本']:.6f} 秒")
        print(f"  加速比: {results['speedup'].get('intraday_range', 0):.2f}x")
        
        # 生成图表
        try:
            plt.figure(figsize=(10, 6))
            
            # 条形图对比
            labels = ['不使用Numba', '使用Numba']
            times = [
                results['intraday_range']['原始版本'],
                results['intraday_range']['Numba版本']
            ]
            
            bars = plt.bar(labels, times, color=['#3498db', '#2ecc71'])
            
            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height * 0.9,
                    f"{height:.6f}s",
                    ha='center', va='center',
                    color='white', fontweight='bold'
                )
            
            # 添加加速比
            speedup = results['speedup'].get('intraday_range', 0)
            plt.text(
                0.5, 0.5, 
                f"加速比: {speedup:.2f}x",
                ha='center', va='center',
                transform=plt.gca().transAxes,
                fontsize=18, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#f39c12", ec="black", alpha=0.7)
            )
            
            plt.title('Numba加速性能对比 - 日内涨跌幅计算')
            plt.ylabel('执行时间 (秒)')
            plt.tight_layout()
            
            # 保存图表
            plt.savefig('numba_performance_test.png')
            print("\n性能对比图表已保存为: numba_performance_test.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"生成图表时出错: {e}")
        
        logger.info("性能测试完成")
        return True
        
    except Exception as e:
        logger.error(f"运行性能测试时出错: {e}")
        print(f"测试失败: {e}")
        return False

def run_accelerated_backtest():
    """运行加速版回测
    
    Returns:
        bool: 回测是否成功完成
    """
    logger.info("开始运行加速版回测")
    
    try:
        from advanced_quant_model import AdvancedQuantModel
        
        # 初始化模型，启用Numba加速
        start_time = time.time()
        model = AdvancedQuantModel(use_numba=True)
        
        # 这里应该调用您的回测主函数
        # 例如：model.run_backtest(...)
        
        # 模拟回测执行
        print("模拟回测执行中...")
        # TODO: 替换为真实回测代码
        time.sleep(2)
        
        print(f"回测完成，总用时: {time.time() - start_time:.2f} 秒")
        logger.info(f"回测完成，总用时: {time.time() - start_time:.2f} 秒")
        
        return True
        
    except Exception as e:
        logger.error(f"运行加速版回测时出错: {e}")
        print(f"回测失败: {e}")
        return False

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="回测系统Numba加速接口")
    parser.add_argument("--bench", action="store_true", help="运行性能测试")
    parser.add_argument("--run", action="store_true", help="使用加速运行回测")
    parser.add_argument("--test-size", type=int, default=100000, help="性能测试数据大小")
    parser.add_argument("--trials", type=int, default=3, help="性能测试重复次数")
    
    args = parser.parse_args()
    
    # 检查依赖项
    if not check_dependencies():
        print("缺少必要依赖，请安装所需包后重试")
        return 1
    
    # 根据参数执行不同功能
    if args.bench:
        success = run_benchmark(test_size=args.test_size, trials=args.trials)
        return 0 if success else 1
    
    if args.run:
        success = run_accelerated_backtest()
        return 0 if success else 1
    
    # 如果没有提供参数，显示帮助信息
    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 