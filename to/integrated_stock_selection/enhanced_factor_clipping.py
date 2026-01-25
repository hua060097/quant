#!/usr/bin/env python3
"""
顶尖量化级别的因子极端值处理策略

Author: Claude Code Assistant
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings

class AdvancedFactorClipper:
    """
    高级因子裁剪器 - 采用多重策略
    """

    def __init__(self):
        self.clip_history = {}

    def calculate_adaptive_thresholds(self, factor_series: pd.Series,
                                     method: str = "adaptive") -> Tuple[float, float]:
        """
        计算自适应阈值

        Args:
            factor_series: 因子序列
            method: 方法类型

        Returns:
            Tuple[float, float]: (下阈值, 上阈值)
        """
        # 移除缺失值
        data = factor_series.dropna()

        if len(data) < 100:  # 样本太小，使用保守策略
            return -3.0, 3.0

        if method == "adaptive":
            # 策略1: 基于实际分布的百分位数
            p99_5 = np.percentile(data, 99.5)
            p0_5 = np.percentile(data, 0.5)

            # 策略2: 基于MAD (Median Absolute Deviation) - 更鲁棒
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            mad_threshold = 3 * 1.4826 * mad  # 1.4826是正态分布的MAD转标准差系数
            mad_upper = median + mad_threshold
            mad_lower = median - mad_threshold

            # 策略3: 基于滚动窗口的动态标准差
            rolling_std = data.rolling(window=min(60, len(data)//4)).std()
            dynamic_std_threshold = 3 * rolling_std.median()
            dynamic_mean = data.rolling(window=min(60, len(data)//4)).mean().median()
            dynamic_upper = dynamic_mean + dynamic_std_threshold
            dynamic_lower = dynamic_mean - dynamic_std_threshold

            # 策略4: 修正的Z-Score (考虑偏度和峰度)
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)

            # 根据偏度和峰度调整阈值
            if abs(skewness) > 0.5:  # 有偏态
                if skewness > 0:  # 右偏
                    adjusted_upper = min(p99_5, mad_upper, dynamic_upper)
                    adjusted_lower = max(p0_5, mad_lower, dynamic_lower)
                else:  # 左偏
                    adjusted_upper = min(p99_5, mad_upper, dynamic_upper) * 0.9  # 更保守
                    adjusted_lower = max(p0_5, mad_lower, dynamic_lower) * 1.1
            else:  # 相对对称
                adjusted_upper = min(p99_5, mad_upper, dynamic_upper)
                adjusted_lower = max(p0_5, mad_lower, dynamic_lower)

            # 如果峰度过高，使用更保守的阈值
            if kurtosis > 3:  # 厚尾
                adjusted_upper *= 0.8
                adjusted_lower *= 0.8

            return adjusted_lower, adjusted_upper

        elif method == "quantile":
            # 纯分位数方法
            return np.percentile(data, 1), np.percentile(data, 99)

        else:  # "standard"
            return -3.0, 3.0

    def detect_distribution_issues(self, factor_series: pd.Series) -> Dict:
        """
        检测分布问题

        Args:
            factor_series: 因子序列

        Returns:
            Dict: 分布问题报告
        """
        data = factor_series.dropna()

        if len(data) < 50:
            return {"status": "insufficient_data"}

        # 基本统计
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()

        # 分布特征
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        # 正态性检验
        try:
            _, p_value = stats.normaltest(data)
            is_normal = p_value > 0.05
        except:
            is_normal = False
            p_value = 0

        # 极端值检测 (多种方法)
        z_extreme = np.abs((data - mean_val) / std_val) > 3
        iqr = data.quantile(0.75) - data.quantile(0.25)
        iqr_extreme = (data < data.quantile(0.25) - 1.5*iqr) | (data > data.quantile(0.75) + 1.5*iqr)

        # 集中度检测
        concentration_ratio = (data.abs() < 0.1).mean()  # 接近0的值比例

        return {
            "status": "analyzed",
            "sample_size": len(data),
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "is_normal": is_normal,
            "normal_test_p": p_value,
            "z_extreme_ratio": z_extreme.mean(),
            "iqr_extreme_ratio": iqr_extreme.mean(),
            "concentration_ratio": concentration_ratio,
            "issues": self._identify_issues(skewness, kurtosis, is_normal, concentration_ratio)
        }

    def _identify_issues(self, skewness: float, kurtosis: float,
                        is_normal: bool, concentration_ratio: float) -> list:
        """识别具体问题"""
        issues = []

        if abs(skewness) > 1.0:
            issues.append(f"严重偏态 (skew={skewness:.3f})")
        elif abs(skewness) > 0.5:
            issues.append(f"中等偏态 (skew={skewness:.3f})")

        if kurtosis > 5.0:
            issues.append(f"严重厚尾 (kurtosis={kurtosis:.3f})")
        elif kurtosis > 3.0:
            issues.append(f"中等厚尾 (kurtosis={kurtosis:.3f})")

        if not is_normal:
            issues.append("非正态分布")

        if concentration_ratio > 0.3:
            issues.append(f"过度集中 ({concentration_ratio:.1%}在[-0.1,0.1])")
        elif concentration_ratio > 0.2:
            issues.append(f"较高集中 ({concentration_ratio:.1%}在[-0.1,0.1])")

        return issues

    def recommend_clip_strategy(self, factor_name: str,
                               factor_series: pd.Series) -> Dict:
        """
        为因子推荐裁剪策略

        Args:
            factor_name: 因子名称
            factor_series: 因子序列

        Returns:
            Dict: 推荐策略
        """
        # 检测分布问题
        dist_issues = self.detect_distribution_issues(factor_series)

        if dist_issues["status"] != "analyzed":
            return {
                "strategy": "standard",
                "threshold": (-3.0, 3.0),
                "reason": "数据不足，使用标准阈值"
            }

        # 基于因子类型和分布问题推荐策略
        issues = dist_issues["issues"]

        # 行业内因子特殊处理
        if factor_name.endswith("_intra_z"):
            # 行内因子样本量更小，需要更保守
            if "严重厚尾" in issues or "严重偏态" in issues:
                return {
                    "strategy": "conservative_quantile",
                    "threshold": (np.percentile(factor_series.dropna(), 2),
                                np.percentile(factor_series.dropna(), 98)),
                    "reason": "行业内因子存在严重分布问题，使用保守分位数裁剪"
                }
            else:
                return {
                    "strategy": "adaptive",
                    "threshold": self.calculate_adaptive_thresholds(factor_series, "adaptive"),
                    "reason": "行业内因子使用自适应裁剪"
                }

        # 收益率类因子特殊处理
        if any(keyword in factor_name.lower() for keyword in ['return', 'pct_chg', 'momentum']):
            if "厚尾" in issues:
                return {
                    "strategy": "tail_aware",
                    "threshold": (-4.0, 4.0),  # 收益率因子允许更大的极端值
                    "reason": "收益率因子天然厚尾，使用宽松阈值"
                }

        # 根据严重程度选择策略
        if len(issues) >= 3 or "严重" in str(issues):
            return {
                "strategy": "conservative",
                "threshold": (np.percentile(factor_series.dropna(), 1),
                            np.percentile(factor_series.dropna(), 99)),
                "reason": f"因子存在严重问题: {', '.join(issues)}"
            }
        elif len(issues) >= 1:
            return {
                "strategy": "adaptive",
                "threshold": self.calculate_adaptive_thresholds(factor_series, "adaptive"),
                "reason": f"因子存在中等问题: {', '.join(issues)}"
            }
        else:
            return {
                "strategy": "standard",
                "threshold": (-3.0, 3.0),
                "reason": "因子分布相对正常"
            }

    def apply_intelligent_clipping(self, factor_df: pd.DataFrame,
                                 factor_columns: list) -> pd.DataFrame:
        """
        应用智能裁剪

        Args:
            factor_df: 因子数据框
            factor_columns: 需要裁剪的因子列

        Returns:
            pd.DataFrame: 裁剪后的数据框
        """
        result_df = factor_df.copy()
        clipping_report = {}

        for factor in factor_columns:
            if factor not in factor_df.columns:
                continue

            # 获取推荐策略
            strategy_info = self.recommend_clip_strategy(factor, factor_df[factor])
            lower_thresh, upper_thresh = strategy_info["threshold"]

            # 应用裁剪
            original_data = factor_df[factor].copy()
            clipped_data = original_data.clip(lower=lower_thresh, upper=upper_thresh)

            # 记录裁剪情况
            n_clipped = ((original_data != clipped_data) &
                        (original_data.notna()) & (clipped_data.notna())).sum()

            result_df[factor] = clipped_data

            clipping_report[factor] = {
                "strategy": strategy_info["strategy"],
                "threshold": strategy_info["threshold"],
                "reason": strategy_info["reason"],
                "clipped_count": n_clipped,
                "clipped_ratio": n_clipped / len(original_data.dropna()),
                "original_range": (original_data.min(), original_data.max()),
                "clipped_range": (clipped_data.min(), clipped_data.max())
            }

        self.clip_history = clipping_report
        return result_df, clipping_report


def generate_enhanced_recommendations(clipping_report: Dict) -> pd.DataFrame:
    """
    生成增强的裁剪建议

    Args:
        clipping_report: 裁剪报告

    Returns:
        pd.DataFrame: 建议数据框
    """
    recommendations = []

    for factor, info in clipping_report.items():
        clipped_ratio = info["clipped_ratio"]
        strategy = info["strategy"]

        # 根据裁剪比例和策略给出建议
        if clipped_ratio > 0.05:  # 超过5%被裁剪
            priority = "HIGH"
            action = "重新检查因子计算逻辑"
        elif clipped_ratio > 0.02:  # 2-5%
            priority = "MEDIUM"
            action = "监控因子稳定性"
        else:
            priority = "LOW"
            action = "保持当前策略"

        recommendations.append({
            "factor": factor,
            "priority": priority,
            "clipped_ratio": clipped_ratio,
            "strategy": strategy,
            "action": action,
            "threshold": info["threshold"],
            "reason": info["reason"]
        })

    return pd.DataFrame(recommendations).sort_values(['priority', 'clipped_ratio'],
                                                   ascending=[True, False])


# 使用示例
if __name__ == "__main__":
    # 示例代码
    clipper = AdvancedFactorClipper()

    print("高级因子裁剪器已准备就绪")
    print("主要策略:")
    print("1. 自适应阈值 - 基于实际分布特性")
    print("2. 多重检测 - Z-score, IQR, MAD, 分位数")
    print("3. 分类处理 - 行业内因子、收益率因子特殊处理")
    print("4. 动态调整 - 根据偏度、峰度调整阈值")