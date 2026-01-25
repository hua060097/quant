#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更稳健的Top-k收益评估方法
提供比均值和中位数更合适的统计指标
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def robust_return_statistics(returns, confidence_level=0.95):
    """
    提供更稳健的收益统计指标

    Args:
        returns: 收益率数组
        confidence_level: 置信水平 (默认95%)

    Returns:
        dict: 包含各种稳健统计指标的字典
    """
    returns = np.array(returns)

    # 基础统计
    n = len(returns)
    if n == 0:
        return {}

    # 1. 中心趋势指标
    mean_return = np.mean(returns)
    median_return = np.median(returns)

    # 2. 稳健的中心趋势指标
    trimmed_mean_10 = stats.trim_mean(returns, 0.1)  # 去除两端10%后的均值
    trimmed_mean_20 = stats.trim_mean(returns, 0.2)  # 去除两端20%后的均值
    huber_m = stats.huberstats(returns)[0]  # Huber M估计量

    # 3. 离散程度指标
    std_return = np.std(returns, ddof=1)
    mad = stats.median_abs_deviation(returns)  # 中位数绝对偏差
    iqr = np.percentile(returns, 75) - np.percentile(returns, 25)  # 四分位距

    # 4. 分位数指标
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = {f'p{p}': np.percentile(returns, p) for p in percentiles}

    # 5. 置信区间
    sem = std_return / np.sqrt(n)  # 标准误
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, n-1)
    ci_lower = mean_return - t_critical * sem
    ci_upper = mean_return + t_critical * sem

    # 6. 风险调整指标
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    sortino_ratio = mean_return / np.std(np.minimum(returns, 0)) if np.std(np.minimum(returns, 0)) > 0 else 0
    max_drawdown = calculate_max_drawdown(returns)
    calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

    # 7. 异常值检测
    q1, q3 = np.percentile(returns, [25, 75])
    iqr_outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    outliers = returns[(returns < iqr_outlier_bounds[0]) | (returns > iqr_outlier_bounds[1])]

    # 8. 分布检验
    _, p_value_normal = stats.normaltest(returns)
    is_normal = p_value_normal > 0.05

    return {
        'sample_size': n,
        'mean': mean_return,
        'median': median_return,
        'trimmed_mean_10': trimmed_mean_10,
        'trimmed_mean_20': trimmed_mean_20,
        'huber_m': huber_m,
        'std': std_return,
        'mad': mad,
        'iqr': iqr,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'range': np.max(returns) - np.min(returns),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / n * 100,
        'is_normal_distribution': is_normal,
        'normal_test_p_value': p_value_normal,
        **percentile_values
    }

def calculate_max_drawdown(returns):
    """
    计算最大回撤
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

def compare_return_distributions(val_returns, test_returns, k_value):
    """
    比较验证集和测试集的收益分布

    Args:
        val_returns: 验证集收益率
        test_returns: 测试集收益率
        k_value: Top-k的k值
    """
    print(f"\n=== Top-{k_value} 收益分布比较 ===")

    val_stats = robust_return_statistics(val_returns)
    test_stats = robust_return_statistics(test_returns)

    print(f"\n验证集 (样本数: {val_stats.get('sample_size', 0)}):")
    print_stats(val_stats, "val")

    print(f"\n测试集 (样本数: {test_stats.get('sample_size', 0)}):")
    print_stats(test_stats, "test")

    # 统计显著性检验
    if len(val_returns) > 0 and len(test_returns) > 0:
        # Mann-Whitney U检验 (非参数检验)
        u_stat, u_p_value = stats.mannwhitneyu(val_returns, test_returns, alternative='two-sided')

        # t检验 (参数检验，假设正态分布)
        t_stat, t_p_value = stats.ttest_ind(val_returns, test_returns)

        print(f"\n统计显著性检验:")
        print(f"  Mann-Whitney U检验: p-value = {u_p_value:.6f}")
        print(f"    {'显著差异' if u_p_value < 0.05 else '无显著差异'} (α=0.05)")
        print(f"  t检验: p-value = {t_p_value:.6f}")
        print(f"    {'显著差异' if t_p_value < 0.05 else '无显著差异'} (α=0.05)")

    # 效应量计算
    if len(val_returns) > 0 and len(test_returns) > 0:
        # Cohen's d
        pooled_std = np.sqrt(((len(val_returns) - 1) * np.var(val_returns, ddof=1) +
                              (len(test_returns) - 1) * np.var(test_returns, ddof=1)) /
                             (len(val_returns) + len(test_returns) - 2))
        cohens_d = (np.mean(test_returns) - np.mean(val_returns)) / pooled_std

        print(f"\n效应量 (Cohen's d): {cohens_d:.4f}")
        effect_size_interpretation = interpret_cohens_d(cohens_d)
        print(f"效应量解释: {effect_size_interpretation}")

def print_stats(stats_dict, prefix=""):
    """
    格式化打印统计结果
    """
    if not stats_dict:
        return

    print(f"  均值: {stats_dict.get('mean', 0):.4f}")
    print(f"  中位数: {stats_dict.get('median', 0):.4f}")
    print(f"  去尾均值(10%): {stats_dict.get('trimmed_mean_10', 0):.4f}")
    print(f"  去尾均值(20%): {stats_dict.get('trimmed_mean_20', 0):.4f}")
    print(f"  Huber M估计量: {stats_dict.get('huber_m', 0):.4f}")
    print(f"  标准差: {stats_dict.get('std', 0):.4f}")
    print(f"  MAD: {stats_dict.get('mad', 0):.4f}")
    print(f"  IQR: {stats_dict.get('iqr', 0):.4f}")
    print(f"  夏普比率: {stats_dict.get('sharpe_ratio', 0):.4f}")
    print(f"  索提诺比率: {stats_dict.get('sortino_ratio', 0):.4f}")
    print(f"  最大回撤: {stats_dict.get('max_drawdown', 0):.4f}")
    print(f"  范围: [{stats_dict.get('min_return', 0):.4f}, {stats_dict.get('max_return', 0):.4f}]")
    print(f"  异常值数量: {stats_dict.get('outlier_count', 0)} ({stats_dict.get('outlier_percentage', 0):.1f}%)")
    print(f"  正态分布: {'是' if stats_dict.get('is_normal_distribution', False) else '否'} (p={stats_dict.get('normal_test_p_value', 0):.4f})")
    print(f"  95%置信区间: [{stats_dict.get('ci_lower', 0):.4f}, {stats_dict.get('ci_upper', 0):.4f}]")

def interpret_cohens_d(cohens_d):
    """
    解释Cohen's d效应量
    """
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "微小效应"
    elif abs_d < 0.5:
        return "小效应"
    elif abs_d < 0.8:
        return "中等效应"
    else:
        return "大效应"

def generate_robust_evaluation_report(val_data, test_data):
    """
    生成稳健的评估报告
    """
    print("="*60)
    print("稳健的Top-k收益评估报告")
    print("="*60)

    # 假设val_data和test_data是包含不同k值收益的字典
    # 格式: {k: [returns]}

    for k in sorted(val_data.keys()):
        if k in test_data:
            compare_return_distributions(val_data[k], test_data[k], k)
            print("\n" + "-"*50)

def bootstrap_confidence_interval(returns, n_bootstrap=10000, confidence_level=0.95):
    """
    使用bootstrap方法计算置信区间

    Args:
        returns: 收益率数组
        n_bootstrap: bootstrap次数
        confidence_level: 置信水平
    """
    returns = np.array(returns)
    n = len(returns)
    if n == 0:
        return None, None, None

    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(returns, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    mean_est = np.mean(bootstrap_means)

    return mean_est, lower, upper

def analyze_return_distribution_properties(returns):
    """
    分析收益分布的详细属性
    """
    returns = np.array(returns)

    print("收益分布详细分析:")
    print(f"  偏度 (Skewness): {stats.skew(returns):.4f}")
    print(f"  峰度 (Kurtosis): {stats.kurtosis(returns):.4f}")
    print(f"  Jarque-Bera检验: p-value = {stats.jarque_bera(returns)[1]:.6f}")
    print(f"    {'符合正态分布' if stats.jarque_bera(returns)[1] > 0.05 else '不符合正态分布'}")

# 示例使用
if __name__ == "__main__":
    # 模拟一些数据
    np.random.seed(42)

    # 模拟验证集数据 (较小的样本，较少的极端值)
    val_returns_1 = np.random.normal(0.005, 0.02, 100)  # Top-1
    val_returns_5 = np.random.normal(0.004, 0.015, 200)  # Top-5

    # 模拟测试集数据 (更大的样本，包含极端值)
    test_returns_1 = np.concatenate([
        np.random.normal(0.008, 0.025, 400),
        np.array([0.773])  # 极端值
    ])  # Top-1
    test_returns_5 = np.concatenate([
        np.random.normal(0.006, 0.02, 800),
        np.array([0.353, 0.445, 0.382])  # 极端值
    ])  # Top-5

    val_data = {1: val_returns_1, 5: val_returns_5}
    test_data = {1: test_returns_1, 5: test_returns_5}

    generate_robust_evaluation_report(val_data, test_data)

    print("\n=== Bootstrap置信区间示例 ===")
    for k in [1, 5]:
        val_mean, val_lower, val_upper = bootstrap_confidence_interval(val_data[k])
        test_mean, test_lower, test_upper = bootstrap_confidence_interval(test_data[k])

        print(f"\nTop-{k}:")
        print(f"  验证集: 均值={val_mean:.4f}, 95%CI=[{val_lower:.4f}, {val_upper:.4f}]")
        print(f"  测试集: 均值={test_mean:.4f}, 95%CI=[{test_lower:.4f}, {test_upper:.4f}]")