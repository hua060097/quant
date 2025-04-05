#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numba加速器示例

演示如何在回测系统中使用Numba加速器加速计算
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import wraps

try:
    from numba_accelerator import get_numba_accelerator, NumbaAccelerator
    HAS_ACCELERATOR = True
except ImportError:
    print("无法导入Numba加速器，将使用原始计算方法")
    HAS_ACCELERATOR = False

# 假设的AdvancedQuantModel类
class MockAdvancedQuantModel:
    """模拟的高级量化模型类"""
    
    def __init__(self):
        self.name = "MockModel"
        self.data = {}
        
    def calculate_intraday_range(self, open_price, high_price, low_price, prev_close=None):
        """计算日内区间涨跌幅
        
        原始版本，用于对比性能
        """
        if prev_close is None or prev_close <= 0:
            prev_close = open_price
            
        if open_price <= 0 or high_price <= 0 or low_price <= 0:
            return 0.0, 0.0
                
        up_range = (high_price - open_price) / prev_close * 100.0
        down_range = (low_price - open_price) / prev_close * 100.0
            
        return up_range, down_range
    
    def _extract_features(self, prices, lookback=10):
        """提取价格特征
        
        原始版本，用于对比性能
        """
        n = len(prices)
        features = np.zeros((n, 5))
        
        # 计算简单移动平均
        for i in range(n):
            if i < lookback - 1:
                features[i, 0] = np.mean(prices[:i+1])
            else:
                features[i, 0] = np.mean(prices[i-lookback+1:i+1])
        
        # 计算每日价格变化
        for i in range(1, n):
            features[i, 1] = (prices[i] - prices[i-1]) / prices[i-1] * 100
        
        # 计算波动率 (10日标准差)
        for i in range(lookback, n):
            features[i, 2] = np.std(features[i-lookback+1:i+1, 1])
        
        # 计算RSI指标
        gains = np.zeros(n)
        losses = np.zeros(n)
        
        for i in range(1, n):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains[i] = change
            else:
                losses[i] = -change
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        
        for i in range(lookback, n):
            avg_gain[i] = np.mean(gains[i-lookback+1:i+1])
            avg_loss[i] = np.mean(losses[i-lookback+1:i+1])
            
            if avg_loss[i] == 0:
                features[i, 3] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                features[i, 3] = 100 - (100 / (1 + rs))
        
        # 计算MACD简化版
        ema12 = np.zeros(n)
        ema26 = np.zeros(n)
        
        if n > 0:
            ema12[0] = prices[0]
            ema26[0] = prices[0]
        
        alpha12 = 2.0 / (12 + 1.0)
        alpha26 = 2.0 / (26 + 1.0)
        
        for i in range(1, n):
            ema12[i] = prices[i] * alpha12 + ema12[i-1] * (1.0 - alpha12)
            ema26[i] = prices[i] * alpha26 + ema26[i-1] * (1.0 - alpha26)
            features[i, 4] = ema12[i] - ema26[i]
            
        return features


# 性能对比函数
def benchmark_calculation(model, size=100000, trials=3):
    """对比计算性能
    
    Args:
        model: 量化模型实例
        size: 测试数据大小
        trials: 重复测试次数
    
    Returns:
        dict: 测试结果统计
    """
    # 生成测试数据
    np.random.seed(42)
    open_prices = np.random.rand(size) * 100 + 50
    high_prices = open_prices * (1 + np.random.rand(size) * 0.1)
    low_prices = open_prices * (1 - np.random.rand(size) * 0.1)
    prev_closes = open_prices * (1 + (np.random.rand(size) * 0.06 - 0.03))
    
    prices = np.random.rand(size) * 100 + 50
    
    results = {
        "intraday_range": {"原始版本": 0, "Numba版本": 0},
        "extract_features": {"原始版本": 0, "Numba版本": 0},
        "speedup": {"intraday_range": 0, "extract_features": 0}
    }
    
    # 测试日内涨跌幅计算
    # 原始版本
    start_time = time.time()
    for _ in range(trials):
        for i in range(size):
            model.calculate_intraday_range(
                open_prices[i], high_prices[i], low_prices[i], prev_closes[i]
            )
    results["intraday_range"]["原始版本"] = (time.time() - start_time) / trials
    
    # 如果有加速器，测试加速版本
    if HAS_ACCELERATOR:
        # 获取并应用加速器
        accelerator = get_numba_accelerator()
        accelerator.enable_benchmark_mode(True)
        accelerator.patch_advanced_quant_model(MockAdvancedQuantModel)
        
        # 等待JIT编译完成（第一次运行较慢）
        model.calculate_intraday_range(100, 110, 95, 100)
        
        # 批量计算测试
        start_time = time.time()
        for _ in range(trials):
            model.calculate_intraday_range(
                open_prices, high_prices, low_prices, prev_closes
            )
        results["intraday_range"]["Numba版本"] = (time.time() - start_time) / trials
        
        # 计算加速比
        if results["intraday_range"]["原始版本"] > 0:
            results["speedup"]["intraday_range"] = (
                results["intraday_range"]["原始版本"] / 
                results["intraday_range"]["Numba版本"]
            )
    
    # 测试特征提取计算
    # 原始版本 - 只测试一小部分数据，否则太慢
    subset_size = min(10000, size)
    start_time = time.time()
    for _ in range(trials):
        model._extract_features(prices[:subset_size])
    results["extract_features"]["原始版本"] = (time.time() - start_time) / trials
    
    # TODO: 完整的特征提取加速等后续实现
    
    return results


# 主函数
def main():
    """主函数"""
    print("=" * 60)
    print("Numba加速器示例 - 性能对比")
    print("=" * 60)
    
    # 创建模型实例
    model = MockAdvancedQuantModel()
    
    # 运行性能测试
    print("\n1. 小规模测试 (1,000数据点)")
    results_small = benchmark_calculation(model, size=1000, trials=5)
    print_benchmark_results(results_small)
    
    print("\n2. 中等规模测试 (10,000数据点)")
    results_medium = benchmark_calculation(model, size=10000, trials=3)
    print_benchmark_results(results_medium)
    
    print("\n3. 大规模测试 (100,000数据点)")
    results_large = benchmark_calculation(model, size=100000, trials=2)
    print_benchmark_results(results_large)
    
    # 可视化性能对比
    if HAS_ACCELERATOR:
        plot_benchmark_results([
            ("小规模", results_small),
            ("中等规模", results_medium),
            ("大规模", results_large)
        ])


def print_benchmark_results(results):
    """打印性能测试结果
    
    Args:
        results: 测试结果字典
    """
    print("\n计算日内涨跌幅:")
    print(f"  原始版本: {results['intraday_range']['原始版本']:.6f} 秒")
    if HAS_ACCELERATOR:
        print(f"  Numba版本: {results['intraday_range']['Numba版本']:.6f} 秒")
        print(f"  加速比: {results['speedup']['intraday_range']:.2f}x")
    
    print("\n特征提取计算:")
    print(f"  原始版本: {results['extract_features']['原始版本']:.6f} 秒")
    if results['extract_features'].get('Numba版本', 0) > 0:
        print(f"  Numba版本: {results['extract_features']['Numba版本']:.6f} 秒")
        print(f"  加速比: {results['speedup'].get('extract_features', 0):.2f}x")


def plot_benchmark_results(all_results):
    """绘制性能测试结果图表
    
    Args:
        all_results: 元组列表(标签,结果)
    """
    try:
        import matplotlib.pyplot as plt
        
        labels = [r[0] for r in all_results]
        intraday_original = [r[1]['intraday_range']['原始版本'] for r in all_results]
        intraday_numba = [r[1]['intraday_range']['Numba版本'] for r in all_results]
        
        x = range(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, intraday_original, width, label='原始版本')
        ax.bar([i + width for i in x], intraday_numba, width, label='Numba版本')
        
        ax.set_ylabel('执行时间 (秒)')
        ax.set_title('Numba加速性能对比 - 日内涨跌幅计算')
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(labels)
        ax.legend()
        
        # 添加加速比标签
        for i in range(len(labels)):
            speedup = all_results[i][1]['speedup']['intraday_range']
            ax.text(i + width/2, intraday_original[i]/2, 
                   f"{speedup:.1f}x",
                   ha='center', va='center', 
                   fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig('numba_performance_comparison.png')
        print("\n性能对比图表已保存为: numba_performance_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"无法绘制图表: {e}")


if __name__ == "__main__":
    main() 