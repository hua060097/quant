#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试稳健评估功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入main.py中的函数
import numpy as np
from scipy import stats

def calculate_robust_statistics(returns, confidence_level=0.95):
    """
    计算稳健统计指标

    Args:
        returns: 收益率数组
        confidence_level: 置信水平

    Returns:
        dict: 稳健统计指标
    """
    returns = np.array(returns)
    n = len(returns)

    if n == 0:
        return {}

    # 基础统计
    mean_return = np.mean(returns)
    median_return = np.median(returns)

    # 稳健的中心趋势指标
    trimmed_mean_10 = stats.trim_mean(returns, 0.1)  # 去除两端10%后的均值
    trimmed_mean_20 = stats.trim_mean(returns, 0.2)  # 去除两端20%后的均值

    # Winsorized均值 - 将极端值限制在指定分位数
    winsorized_10 = stats.mstats.winsorize(returns, limits=[0.1, 0.1])
    winsorized_mean_10 = np.mean(winsorized_10)

    winsorized_20 = stats.mstats.winsorize(returns, limits=[0.2, 0.2])
    winsorized_mean_20 = np.mean(winsorized_20)

    # 风险指标
    std_return = np.std(returns, ddof=1)

    # 夏普比率
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0

    # 下行风险
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0

    # 最大回撤
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

    # 分位数
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = {f'p{p}': np.percentile(returns, p) for p in percentiles}

    # IQR (四分位距)
    iqr = np.percentile(returns, 75) - np.percentile(returns, 25)

    # MAD (中位数绝对偏差)
    mad = stats.median_abs_deviation(returns)

    # 异常值检测 (IQR方法)
    q1, q3 = np.percentile(returns, [25, 75])
    iqr_outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    outliers = returns[(returns < iqr_outlier_bounds[0]) | (returns > iqr_outlier_bounds[1])]

    # 置信区间
    sem = std_return / np.sqrt(n)
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, n-1)
    ci_lower = mean_return - t_critical * sem
    ci_upper = mean_return + t_critical * sem

    return {
        'n': n,
        'mean': mean_return,
        'median': median_return,
        'trimmed_mean_10': trimmed_mean_10,
        'trimmed_mean_20': trimmed_mean_20,
        'winsorized_mean_10': winsorized_mean_10,
        'winsorized_mean_20': winsorized_mean_20,
        'std': std_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'range': np.max(returns) - np.min(returns),
        'iqr': iqr,
        'mad': mad,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / n * 100,
        **percentile_values
    }

def test_robust_evaluation():
    """
    测试稳健评估功能
    """
    print("="*60)
    print("测试稳健评估功能")
    print("="*60)

    # 模拟验证集数据 (53天，没有极端值)
    np.random.seed(42)
    val_returns = np.random.normal(0.0045, 0.02, 265)  # Top-5: 0.45%

    # 模拟测试集数据 (89天，包含极端值)
    test_returns_normal = np.random.normal(0.008, 0.025, 400)  # 正常收益
    test_returns_extreme = np.array([0.773])  # 极端值 (来自300486.SZ)
    test_returns = np.concatenate([test_returns_normal, test_returns_extreme])

    print(f"\n模拟数据集:")
    print(f"验证集: {len(val_returns)}个样本，传统均值={np.mean(val_returns):.4f}")
    print(f"测试集: {len(test_returns)}个样本，传统均值={np.mean(test_returns):.4f}")

    print(f"\n验证集稳健评估:")
    val_stats = calculate_robust_statistics(val_returns)
    print(f"  传统均值: {val_stats['mean']:.4f}")
    print(f"  Winsorized均值(10%): {val_stats['winsorized_mean_10']:.4f}")
    print(f"  去尾均值(20%): {val_stats['trimmed_mean_20']:.4f}")
    print(f"  夏普比率: {val_stats['sharpe_ratio']:.4f}")
    print(f"  异常值比例: {val_stats['outlier_percentage']:.1f}%")
    print(f"  95%分位数: {val_stats['p95']:.4f}")

    print(f"\n测试集稳健评估:")
    test_stats = calculate_robust_statistics(test_returns)
    print(f"  传统均值: {test_stats['mean']:.4f}")
    print(f"  Winsorized均值(10%): {test_stats['winsorized_mean_10']:.4f}")
    print(f"  去尾均值(20%): {test_stats['trimmed_mean_20']:.4f}")
    print(f"  夏普比率: {test_stats['sharpe_ratio']:.4f}")
    print(f"  异常值比例: {test_stats['outlier_percentage']:.1f}%")
    print(f"  95%分位数: {test_stats['p95']:.4f}")

    print(f"\n对比分析:")
    print(f"传统均值差异: {test_stats['mean'] - val_stats['mean']:.4f} ({abs(test_stats['mean'] - val_stats['mean'])*100:.1f}%)")
    print(f"Winsorized均值差异: {test_stats['winsorized_mean_10'] - val_stats['winsorized_mean_10']:.4f} ({abs(test_stats['winsorized_mean_10'] - val_stats['winsorized_mean_10'])*100:.1f}%)")
    print(f"去尾均值差异: {test_stats['trimmed_mean_20'] - val_stats['trimmed_mean_20']:.4f} ({abs(test_stats['trimmed_mean_20'] - val_stats['trimmed_mean_20'])*100:.1f}%)")

    print(f"\n结论:")
    print(f"- 传统均值差异最大，容易被极端值影响")
    print(f"- Winsorized均值和去尾均值更稳健，差异较小")
    print(f"- 使用稳健指标可以更准确地评估模型真实表现")

if __name__ == "__main__":
    test_robust_evaluation()