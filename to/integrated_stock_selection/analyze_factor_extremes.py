#!/usr/bin/env python3
"""
因子极端值分析脚本

分析生成的因子中所有全局z-score和行业内z-score因子的极端值分布情况，
包括极端值占比、边界值统计等，用于判断是否需要进一步的clip处理。

Author: Claude Code Assistant
Date: 2025-01-09
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from main import FACTOR_LAB_STORE_PATH
from factor_system.factor_lab.store import FactorDatasetStore

class FactorExtremeAnalyzer:
    """因子极端值分析器"""

    def __init__(self, store_path: str = FACTOR_LAB_STORE_PATH):
        """
        初始化分析器

        Args:
            store_path: 因子数据存储路径
        """
        self.store_path = store_path
        self.factor_df = None
        self.extreme_stats = {}

    def load_factor_data(self) -> bool:
        """
        加载因子数据

        Returns:
            bool: 是否成功加载数据
        """
        try:
            logger.info(f"正在加载因子数据: {self.store_path}")
            store = FactorDatasetStore(self.store_path)

            if not store.exists():
                logger.error(f"因子数据文件不存在: {self.store_path}")
                return False

            self.factor_df = store.load()

            if self.factor_df is None or self.factor_df.empty:
                logger.error("因子数据为空")
                return False

            logger.info(f"成功加载因子数据: {len(self.factor_df):,} 行, {len(self.factor_df.columns)} 列")
            logger.info(f"数据时间范围: {self.factor_df['date'].min()} 到 {self.factor_df['date'].max()}")
            logger.info(f"股票数量: {self.factor_df['stock_code'].nunique():,}")
            logger.info(f"交易日数量: {self.factor_df['date'].nunique():,}")

            return True

        except Exception as e:
            logger.error(f"加载因子数据失败: {e}")
            return False

    def identify_zscore_factors(self) -> Tuple[List[str], List[str]]:
        """
        识别全局z-score和行业内z-score因子

        Returns:
            Tuple[List[str], List[str]]: (全局z-score因子列表, 行业内z-score因子列表)
        """
        if self.factor_df is None:
            logger.error("请先加载因子数据")
            return [], []

        global_zscore_factors = []
        intra_zscore_factors = []

        # 获取所有因子列
        factor_columns = [col for col in self.factor_df.columns
                         if col not in ['date', 'stock_code', 'date_str']]

        logger.info(f"总共找到 {len(factor_columns)} 个因子列")

        for factor in factor_columns:
            if factor.endswith('_z'):
                if factor.endswith('_intra_z'):
                    intra_zscore_factors.append(factor)
                else:
                    global_zscore_factors.append(factor)

        logger.info(f"全局z-score因子: {len(global_zscore_factors)} 个")
        logger.info(f"行业内z-score因子: {len(intra_zscore_factors)} 个")

        return global_zscore_factors, intra_zscore_factors

    def calculate_extreme_stats(self, factor_series: pd.Series,
                               extreme_thresholds: List[float] = [3.0, 2.5, 2.0]) -> Dict:
        """
        计算单个因子的极端值统计

        Args:
            factor_series: 因子数据序列
            extreme_thresholds: 极端值阈值列表

        Returns:
            Dict: 极端值统计信息
        """
        # 移除缺失值
        valid_data = factor_series.dropna()

        if len(valid_data) == 0:
            return {}

        stats = {
            'total_count': len(valid_data),
            'mean': valid_data.mean(),
            'std': valid_data.std(),
            'min': valid_data.min(),
            'max': valid_data.max(),
            'q1': valid_data.quantile(0.25),
            'median': valid_data.median(),
            'q3': valid_data.quantile(0.75),
            'extreme_stats': {}
        }

        # 计算不同阈值下的极端值统计
        for threshold in extreme_thresholds:
            upper_extreme_mask = valid_data > threshold
            lower_extreme_mask = valid_data < -threshold

            stats['extreme_stats'][f'threshold_{threshold}'] = {
                'upper_count': upper_extreme_mask.sum(),
                'lower_count': lower_extreme_mask.sum(),
                'total_extreme_count': (upper_extreme_mask | lower_extreme_mask).sum(),
                'upper_pct': upper_extreme_mask.mean() * 100,
                'lower_pct': lower_extreme_mask.mean() * 100,
                'total_extreme_pct': (upper_extreme_mask | lower_extreme_mask).mean() * 100,
                'upper_max': valid_data[upper_extreme_mask].max() if upper_extreme_mask.any() else None,
                'lower_min': valid_data[lower_extreme_mask].min() if lower_extreme_mask.any() else None,
                'upper_max_abs': valid_data[upper_extreme_mask].abs().max() if upper_extreme_mask.any() else None,
                'lower_min_abs': valid_data[lower_extreme_mask].abs().min() if lower_extreme_mask.any() else None
            }

        return stats

    def analyze_factor_extremes(self, factors: List[str],
                               factor_type: str = "global_zscore") -> Dict[str, Dict]:
        """
        分析多个因子的极端值情况

        Args:
            factors: 因子列表
            factor_type: 因子类型描述

        Returns:
            Dict[str, Dict]: 因子极端值统计字典
        """
        logger.info(f"开始分析 {len(factors)} 个 {factor_type} 因子的极端值...")

        factor_stats = {}

        for i, factor in enumerate(factors, 1):
            logger.info(f"  [{i}/{len(factors)}] 分析因子: {factor}")

            if factor not in self.factor_df.columns:
                logger.warning(f"    因子 {factor} 不存在于数据中")
                continue

            factor_series = self.factor_df[factor]
            stats = self.calculate_extreme_stats(factor_series)

            if stats:
                factor_stats[factor] = stats
                # 打印简要统计
                extreme_3_stats = stats['extreme_stats']['threshold_3.0']
                logger.info(f"    均值: {stats['mean']:.3f}, 标准差: {stats['std']:.3f}")
                logger.info(f"    范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
                logger.info(f"    |z|>3 极端值: {extreme_3_stats['total_extreme_pct']:.3f}% "
                          f"(上: {extreme_3_stats['upper_pct']:.3f}%, 下: {extreme_3_stats['lower_pct']:.3f}%)")
            else:
                logger.warning(f"    因子 {factor} 统计失败")

        return factor_stats

    def generate_summary_report(self, global_stats: Dict, intra_stats: Dict) -> pd.DataFrame:
        """
        生成极端值统计汇总报告

        Args:
            global_stats: 全局z-score因子统计
            intra_stats: 行业内z-score因子统计

        Returns:
            pd.DataFrame: 汇总报告
        """
        report_data = []

        # 处理全局z-score因子
        for factor, stats in global_stats.items():
            extreme_3 = stats['extreme_stats']['threshold_3.0']
            extreme_2 = stats['extreme_stats']['threshold_2.0']

            report_data.append({
                'factor': factor,
                'type': 'Global Z-Score',
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'extreme_3_pct': extreme_3['total_extreme_pct'],
                'extreme_3_upper_pct': extreme_3['upper_pct'],
                'extreme_3_lower_pct': extreme_3['lower_pct'],
                'extreme_2_pct': extreme_2['total_extreme_pct'],
                'extreme_2_upper_pct': extreme_2['upper_pct'],
                'extreme_2_lower_pct': extreme_2['lower_pct'],
                'extreme_3_upper_max': extreme_3['upper_max'],
                'extreme_3_lower_min': extreme_3['lower_min']
            })

        # 处理行业内z-score因子
        for factor, stats in intra_stats.items():
            extreme_3 = stats['extreme_stats']['threshold_3.0']
            extreme_2 = stats['extreme_stats']['threshold_2.0']

            report_data.append({
                'factor': factor,
                'type': 'Intra-Sector Z-Score',
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'extreme_3_pct': extreme_3['total_extreme_pct'],
                'extreme_3_upper_pct': extreme_3['upper_pct'],
                'extreme_3_lower_pct': extreme_3['lower_pct'],
                'extreme_2_pct': extreme_2['total_extreme_pct'],
                'extreme_2_upper_pct': extreme_2['upper_pct'],
                'extreme_2_lower_pct': extreme_2['lower_pct'],
                'extreme_3_upper_max': extreme_3['upper_max'],
                'extreme_3_lower_min': extreme_3['lower_min']
            })

        return pd.DataFrame(report_data)

    def plot_extreme_distribution(self, global_stats: Dict, intra_stats: Dict,
                                 save_path: str = None):
        """
        绘制极端值分布图

        Args:
            global_stats: 全局z-score因子统计
            intra_stats: 行业内z-score因子统计
            save_path: 图片保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('因子极端值分布分析', fontsize=16)

        # 提取数据
        global_factors = list(global_stats.keys())
        intra_factors = list(intra_stats.keys())

        # 1. 全局z-score因子 |z|>3 极端值占比
        global_extreme_3 = [stats['extreme_stats']['threshold_3.0']['total_extreme_pct']
                           for stats in global_stats.values()]

        axes[0, 0].barh(range(len(global_factors)), global_extreme_3)
        axes[0, 0].set_yticks(range(len(global_factors)))
        axes[0, 0].set_yticklabels(global_factors, fontsize=8)
        axes[0, 0].set_xlabel('极端值占比 (%)')
        axes[0, 0].set_title('全局Z-Score因子 |z|>3 极端值占比')
        axes[0, 0].axvline(x=0.27, color='red', linestyle='--',
                          label='理论值 (0.27%)')
        axes[0, 0].legend()

        # 2. 行业内z-score因子 |z|>3 极端值占比
        intra_extreme_3 = [stats['extreme_stats']['threshold_3.0']['total_extreme_pct']
                          for stats in intra_stats.values()]

        axes[0, 1].barh(range(len(intra_factors)), intra_extreme_3)
        axes[0, 1].set_yticks(range(len(intra_factors)))
        axes[0, 1].set_yticklabels(intra_factors, fontsize=8)
        axes[0, 1].set_xlabel('极端值占比 (%)')
        axes[0, 1].set_title('行业内Z-Score因子 |z|>3 极端值占比')
        axes[0, 1].axvline(x=0.27, color='red', linestyle='--',
                          label='理论值 (0.27%)')
        axes[0, 1].legend()

        # 3. 极端值占比分布对比
        all_extreme_3 = global_extreme_3 + intra_extreme_3
        labels = ['Global'] * len(global_extreme_3) + ['Intra'] * len(intra_extreme_3)

        df_violin = pd.DataFrame({
            'extreme_pct': all_extreme_3,
            'type': labels
        })

        sns.violinplot(data=df_violin, x='type', y='extreme_pct', ax=axes[1, 0])
        axes[1, 0].set_ylabel('极端值占比 (%)')
        axes[1, 0].set_title('极端值占比分布对比')
        axes[1, 0].axhline(y=0.27, color='red', linestyle='--',
                          label='理论值 (0.27%)')
        axes[1, 0].legend()

        # 4. 极端值边界值分布
        global_max_values = [stats['max'] for stats in global_stats.values()]
        global_min_values = [stats['min'] for stats in global_stats.values()]
        intra_max_values = [stats['max'] for stats in intra_stats.values()]
        intra_min_values = [stats['min'] for stats in intra_stats.values()]

        axes[1, 1].scatter(global_max_values, global_min_values,
                          alpha=0.6, label='Global Z-Score')
        axes[1, 1].scatter(intra_max_values, intra_min_values,
                          alpha=0.6, label='Intra-Sector Z-Score')
        axes[1, 1].set_xlabel('最大值')
        axes[1, 1].set_ylabel('最小值')
        axes[1, 1].set_title('因子值边界分布')
        axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[1, 1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存至: {save_path}")

        plt.show()

    def generate_clip_recommendations(self, report_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成clip建议

        Args:
            report_df: 汇总报告DataFrame

        Returns:
            pd.DataFrame: 包含clip建议的DataFrame
        """
        recommendations = []

        for _, row in report_df.iterrows():
            factor = row['factor']
            factor_type = row['type']
            extreme_3_pct = row['extreme_3_pct']
            extreme_2_pct = row['extreme_2_pct']
            max_val = row['max']
            min_val = row['min']

            # 判断是否需要clip
            needs_clip = False
            clip_threshold = None
            reason = []

            if extreme_3_pct > 0.5:  # 超过理论值约2倍
                needs_clip = True
                reason.append(f"|z|>3极端值占比过高: {extreme_3_pct:.3f}%")

            if extreme_2_pct > 10.0:  # 超过理论值约5倍
                needs_clip = True
                reason.append(f"|z|>2极端值占比过高: {extreme_2_pct:.3f}%")

            if abs(max_val) > 5.0 or abs(min_val) > 5.0:
                needs_clip = True
                reason.append(f"极端边界值: [{min_val:.3f}, {max_val:.3f}]")
                clip_threshold = max(abs(max_val), abs(min_val), 3.0)

            # 根据因子类型调整建议
            if factor_type == 'Intra-Sector Z-Score':
                # 行业内z-score通常应该更接近标准正态分布
                if extreme_3_pct > 0.4:
                    needs_clip = True
                    reason.append("行业内z-score极端值过多")

            recommendations.append({
                'factor': factor,
                'type': factor_type,
                'needs_clip': needs_clip,
                'suggested_threshold': clip_threshold,
                'current_extreme_3_pct': extreme_3_pct,
                'current_extreme_2_pct': extreme_2_pct,
                'current_range': f"[{min_val:.3f}, {max_val:.3f}]",
                'reason': "; ".join(reason) if reason else "正常范围"
            })

        return pd.DataFrame(recommendations)

    def run_analysis(self, save_results: bool = True,
                    output_dir: str = "factor_extreme_analysis") -> Dict:
        """
        运行完整的极端值分析

        Args:
            save_results: 是否保存结果
            output_dir: 输出目录

        Returns:
            Dict: 分析结果
        """
        logger.info("=" * 60)
        logger.info("开始因子极端值分析")
        logger.info("=" * 60)

        # 1. 加载数据
        if not self.load_factor_data():
            return {}

        # 2. 识别因子
        global_factors, intra_factors = self.identify_zscore_factors()

        if not global_factors and not intra_factors:
            logger.error("未找到z-score因子")
            return {}

        # 3. 分析极端值
        global_stats = {}
        intra_stats = {}

        if global_factors:
            global_stats = self.analyze_factor_extremes(global_factors, "全局z-score")

        if intra_factors:
            intra_stats = self.analyze_factor_extremes(intra_factors, "行业内z-score")

        # 4. 生成报告
        report_df = self.generate_summary_report(global_stats, intra_stats)

        # 5. 生成clip建议
        clip_recommendations = self.generate_clip_recommendations(report_df)

        # 6. 打印汇总
        logger.info("=" * 60)
        logger.info("分析结果汇总")
        logger.info("=" * 60)

        if global_stats:
            global_extreme_3 = report_df[report_df['type'] == 'Global Z-Score']['extreme_3_pct']
            logger.info(f"全局z-score因子 |z|>3 极端值平均占比: {global_extreme_3.mean():.3f}% "
                       f"(理论值: 0.27%)")

        if intra_stats:
            intra_extreme_3 = report_df[report_df['type'] == 'Intra-Sector Z-Score']['extreme_3_pct']
            logger.info(f"行业内z-score因子 |z|>3 极端值平均占比: {intra_extreme_3.mean():.3f}% "
                       f"(理论值: 0.27%)")

        # 需要clip的因子数量
        need_clip_count = clip_recommendations['needs_clip'].sum()
        total_count = len(clip_recommendations)
        logger.info(f"建议进行clip的因子: {need_clip_count}/{total_count} "
                   f"({need_clip_count/total_count*100:.1f}%)")

        # 7. 保存结果
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存汇总报告
            report_path = os.path.join(output_dir, f"factor_extreme_report_{timestamp}.csv")
            report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
            logger.info(f"汇总报告已保存至: {report_path}")

            # 保存clip建议
            clip_path = os.path.join(output_dir, f"clip_recommendations_{timestamp}.csv")
            clip_recommendations.to_csv(clip_path, index=False, encoding='utf-8-sig')
            logger.info(f"Clip建议已保存至: {clip_path}")

            # 绘制图表
            plot_path = os.path.join(output_dir, f"extreme_distribution_{timestamp}.png")
            self.plot_extreme_distribution(global_stats, intra_stats, plot_path)

        return {
            'global_stats': global_stats,
            'intra_stats': intra_stats,
            'report_df': report_df,
            'clip_recommendations': clip_recommendations,
            'global_factors': global_factors,
            'intra_factors': intra_factors
        }


def main():
    """主函数"""
    logger.info("因子极端值分析工具")
    logger.info("=" * 60)

    # 创建分析器
    analyzer = FactorExtremeAnalyzer()

    # 运行分析
    results = analyzer.run_analysis(save_results=True)

    if results:
        logger.info("=" * 60)
        logger.info("分析完成!")
        logger.info(f"分析了 {len(results['global_factors'])} 个全局z-score因子")
        logger.info(f"分析了 {len(results['intra_factors'])} 个行业内z-score因子")

        # 打印需要特别注意的因子
        clip_rec = results['clip_recommendations']
        high_risk_factors = clip_rec[clip_rec['needs_clip']].sort_values(
            'current_extreme_3_pct', ascending=False)

        if not high_risk_factors.empty:
            logger.info("\n🚨 需要重点关注的因子 (建议进行clip):")
            for _, row in high_risk_factors.iterrows():
                logger.info(f"  - {row['factor']} ({row['type']})")
                logger.info(f"    极端值占比: |z|>3 ({row['current_extreme_3_pct']:.3f}%), "
                           f"|z|>2 ({row['current_extreme_2_pct']:.3f}%)")
                logger.info(f"    建议阈值: {row['suggested_threshold']}")
                logger.info(f"    原因: {row['reason']}")
        else:
            logger.info("✅ 所有因子极端值都在合理范围内")

    else:
        logger.error("分析失败!")


if __name__ == "__main__":
    main()