#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的Top-k收益评估方法 - 解决极端值问题
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    n = len(returns)
    if n == 0:
        return {}

    # 1. 中心趋势指标
    mean_return = np.mean(returns)
    median_return = np.median(returns)

    # 2. 稳健的中心趋势指标
    trimmed_mean_10 = stats.trim_mean(returns, 0.1)  # 去除两端10%后的均值
    trimmed_mean_20 = stats.trim_mean(returns, 0.2)  # 去除两端20%后的均值

    # 使用winsorized均值作为稳健估计
    winsorized_mean_10 = stats.mstats.winsorize(returns, limits=[0.1, 0.1]).mean()
    winsorized_mean_20 = stats.mstats.winsorize(returns, limits=[0.2, 0.2]).mean()

    # 3. 离散程度指标
    std_return = np.std(returns, ddof=1)
    mad = stats.median_abs_deviation(returns)  # 中位数绝对偏差
    iqr = np.percentile(returns, 75) - np.percentile(returns, 25)  # 四分位距

    # 4. 分位数指标
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = {f'p{p}': np.percentile(returns, p) for p in percentiles}

    # 5. 置信区间
    sem = std_return / np.sqrt(n) if n > 0 else 0
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, n-1) if n > 1 else 0
    ci_lower = mean_return - t_critical * sem
    ci_upper = mean_return + t_critical * sem

    # 6. 风险调整指标
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0

    # 计算下行标准差
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0

    max_drawdown = calculate_max_drawdown(returns)
    calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

    # 7. 异常值检测
    q1, q3 = np.percentile(returns, [25, 75])
    iqr_outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    outliers = returns[(returns < iqr_outlier_bounds[0]) | (returns > iqr_outlier_bounds[1])]

    # 8. 分布检验
    if n >= 8:  # normaltest需要至少8个样本
        try:
            _, p_value_normal = stats.normaltest(returns)
            is_normal = p_value_normal > 0.05
        except:
            p_value_normal = 1.0
            is_normal = True
    else:
        p_value_normal = 1.0
        is_normal = True

    return {
        'sample_size': n,
        'mean': mean_return,
        'median': median_return,
        'trimmed_mean_10': trimmed_mean_10,
        'trimmed_mean_20': trimmed_mean_20,
        'winsorized_mean_10': winsorized_mean_10,
        'winsorized_mean_20': winsorized_mean_20,
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
    return np.min(drawdown) if len(drawdown) > 0 else 0

def bootstrap_confidence_interval(returns, n_bootstrap=10000, confidence_level=0.95):
    """
    使用bootstrap方法计算置信区间
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

def compare_return_distributions(val_returns, test_returns, k_value):
    """
    比较验证集和测试集的收益分布
    """
    print(f"\n=== Top-{k_value} 收益分布比较 ===")

    val_stats = robust_return_statistics(val_returns)
    test_stats = robust_return_statistics(test_returns)

    print(f"\n验证集 (样本数: {val_stats.get('sample_size', 0)}):")
    print_detailed_stats(val_stats)

    print(f"\n测试集 (样本数: {test_stats.get('sample_size', 0)}):")
    print_detailed_stats(test_stats)

    # 统计显著性检验
    if len(val_returns) > 0 and len(test_returns) > 0:
        try:
            # Mann-Whitney U检验 (非参数检验)
            u_stat, u_p_value = stats.mannwhitneyu(val_returns, test_returns, alternative='two-sided')

            # t检验 (参数检验，假设正态分布)
            t_stat, t_p_value = stats.ttest_ind(val_returns, test_returns)

            print(f"\n统计显著性检验:")
            print(f"  Mann-Whitney U检验: p-value = {u_p_value:.6f}")
            print(f"    {'显著差异' if u_p_value < 0.05 else '无显著差异'} (α=0.05)")
            print(f"  t检验: p-value = {t_p_value:.6f}")
            print(f"    {'显著差异' if t_p_value < 0.05 else '无显著差异'} (α=0.05)")
        except Exception as e:
            print(f"统计检验出错: {e}")

    # 效应量计算
    if len(val_returns) > 0 and len(test_returns) > 0:
        try:
            # Cohen's d
            pooled_std = np.sqrt(((len(val_returns) - 1) * np.var(val_returns, ddof=1) +
                                  (len(test_returns) - 1) * np.var(test_returns, ddof=1)) /
                                 (len(val_returns) + len(test_returns) - 2))
            if pooled_std > 0:
                cohens_d = (np.mean(test_returns) - np.mean(val_returns)) / pooled_std
                print(f"\n效应量 (Cohen's d): {cohens_d:.4f}")
                effect_size_interpretation = interpret_cohens_d(cohens_d)
                print(f"效应量解释: {effect_size_interpretation}")
        except Exception as e:
            print(f"效应量计算出错: {e}")

    # Bootstrap置信区间
    print(f"\nBootstrap 95%置信区间:")
    val_boot_mean, val_boot_lower, val_boot_upper = bootstrap_confidence_interval(val_returns, n_bootstrap=1000)
    test_boot_mean, test_boot_lower, test_boot_upper = bootstrap_confidence_interval(test_returns, n_bootstrap=1000)

    print(f"  验证集: {val_boot_mean:.4f} [{val_boot_lower:.4f}, {val_boot_upper:.4f}]")
    print(f"  测试集: {test_boot_mean:.4f} [{test_boot_lower:.4f}, {test_boot_upper:.4f}]")

def print_detailed_stats(stats_dict):
    """
    格式化打印统计结果
    """
    if not stats_dict:
        return

    print(f"  均值: {stats_dict.get('mean', 0):.4f}")
    print(f"  中位数: {stats_dict.get('median', 0):.4f}")
    print(f"  去尾均值(10%): {stats_dict.get('trimmed_mean_10', 0):.4f}")
    print(f"  去尾均值(20%): {stats_dict.get('trimmed_mean_20', 0):.4f}")
    print(f"  Winsorized均值(10%): {stats_dict.get('winsorized_mean_10', 0):.4f}")
    print(f"  Winsorized均值(20%): {stats_dict.get('winsorized_mean_20', 0):.4f}")
    print(f"  标准差: {stats_dict.get('std', 0):.4f}")
    print(f"  MAD: {stats_dict.get('mad', 0):.4f}")
    print(f"  IQR: {stats_dict.get('iqr', 0):.4f}")
    print(f"  夏普比率: {stats_dict.get('sharpe_ratio', 0):.4f}")
    print(f"  索提诺比率: {stats_dict.get('sortino_ratio', 0):.4f}")
    print(f"  最大回撤: {stats_dict.get('max_drawdown', 0):.4f}")
    print(f"  范围: [{stats_dict.get('min_return', 0):.4f}, {stats_dict.get('max_return', 0):.4f}]")
    print(f"  95%分位数: {stats_dict.get('p95', 0):.4f}")
    print(f"  99%分位数: {stats_dict.get('p99', 0):.4f}")
    print(f"  异常值数量: {stats_dict.get('outlier_count', 0)} ({stats_dict.get('outlier_percentage', 0):.1f}%)")
    print(f"  正态分布: {'是' if stats_dict.get('is_normal_distribution', False) else '否'} (p={stats_dict.get('normal_test_p_value', 0):.4f})")

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

def comprehensive_evaluation_summary(val_data, test_data):
    """
    综合评估摘要 - 提供最重要的指标
    """
    print("="*80)
    print("综合评估摘要 - 建议使用的稳健指标")
    print("="*80)

    recommendation_text = """
针对验证集vs测试集收益差异的问题，建议使用以下稳健的评估指标：

1. 📊 主要推荐指标：
   • Winsorized均值(10%): 去除极端值影响，保持大部分数据
   • 95%分位数: 关注大多数情况下的表现，避免极端值
   • 夏普比率: 风险调整后收益，更全面的评估
   • Bootstrap置信区间: 提供更可靠的统计推断

2. 🚫 需要谨慎使用的指标：
   • 简单均值: 容易被极端值拉高
   • 最大值: 偶然事件，不具备代表性
   • 标准差: 对极端值敏感

3. 📈 决策建议：
   • 如果Winsorized均值差异不大，说明模型稳定
   • 如果夏普比率测试集>验证集，说明风险调整后表现更好
   • 如果置信区间重叠，说明差异可能不显著
"""

    print(recommendation_text)

    print("\n" + "="*80)
    print("实际数据分析")
    print("="*80)

    for k in sorted(val_data.keys()):
        if k in test_data:
            compare_return_distributions(val_data[k], test_data[k], k)
            print("\n" + "-"*60)

# 实际使用示例
if __name__ == "__main__":
    # 根据你的日志数据模拟
    np.random.seed(42)

    # 模拟验证集数据 (基于你的日志)
    val_returns_1 = np.random.normal(0.0029, 0.01, 53)  # Top-1: 0.29%
    val_returns_5 = np.random.normal(0.0045, 0.015, 265)  # Top-5: 0.45%

    # 模拟测试集数据 (包含极端值)
    test_returns_1 = np.concatenate([
        np.random.normal(0.01, 0.02, 80),  # 正常收益
        np.array([0.773])  # 极端值
    ])
    test_returns_5 = np.concatenate([
        np.random.normal(0.008, 0.018, 400),  # 正常收益
        np.array([0.353, 0.281, 0.412, 0.398, 0.376])  # 极端值
    ])

    val_data = {1: val_returns_1, 5: val_returns_5}
    test_data = {1: test_returns_1, 5: test_returns_5}

    comprehensive_evaluation_summary(val_data, test_data)