#!/usr/bin/env python3
"""
正确的智能裁剪修复脚本

功能：
1. 从原始因子数据重新进行归一化处理
2. 使用改进的智能裁剪算法v2.0
3. 重新生成所有标准化因子
4. 保存修复后的因子数据

使用方法：
python fix_intelligent_clipping_correct.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, List, Set

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.factor_lab.pipeline import FactorPipeline
from main import load_sw_sector_mapping


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fix_intelligent_clipping_correct.log')
        ]
    )
    return logging.getLogger(__name__)


def extract_failed_factors_from_log(log_file: str) -> List[str]:
    """从日志文件中提取智能裁剪失败的因子列表"""
    logger = logging.getLogger(__name__)

    # 从修复报告中读取失败因子列表
    failed_factors = [
        'consecutive_up_severity', 'corr_20d_mom5', 'corr_5d', 'corr_slope_20_120',
        'distance_from_52w_high', 'distance_from_52w_low', 'idiosyncratic_volatility_sector',
        'institutional_activity_score', 'lower_shadow_ratio', 'ma60_slope', 'ma_convergence_score',
        'ma_ratio_5_10', 'market_correlation', 'near_support_count_60d', 'obv_momentum_factor',
        'open_to_high_ratio', 'pct_chg_1d', 'pct_chg_2d', 'price_impact', 'price_ma60_ratio',
        'pullback_volume_ratio', 'resistance_distance', 'return_3d', 'support_distance',
        'support_risk_reward_ratio', 'turnover_rate_strength', 'volatility_change_rate',
        'volatility_correlation', 'volume_price_divergence'
    ]

    logger.info(f"需要修复的因子数量: {len(failed_factors)}")
    return failed_factors


def load_original_factor_data(factors_file: str) -> pd.DataFrame:
    """加载原始因子数据"""
    logger = logging.getLogger(__name__)

    try:
        df = pd.read_parquet(factors_file)
        logger.info(f"成功加载因子数据: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"加载因子数据失败: {e}")
        return pd.DataFrame()


def identify_factor_processing_type(factor_name: str) -> str:
    """识别因子的处理类型"""
    # 基于因子名称判断应该使用哪种归一化方法
    if factor_name in ['corr_slope_20_120', 'ma_convergence_score']:
        return 'rankpct_only'  # 只使用rankpct
    elif factor_name.startswith('return_') or factor_name.startswith('pct_chg_'):
        return 'dual'  # 双重归一化
    elif factor_name in ['turnover_rate_strength', 'obv_momentum_factor']:
        return 'zscore_only'  # 只使用z-score
    else:
        return 'zscore_only'  # 默认z-score


def regenerate_factors_from_original(df: pd.DataFrame, failed_factors: List[str]) -> Dict[str, pd.Series]:
    """从原始因子数据重新生成归一化因子"""
    logger = logging.getLogger(__name__)

    pipeline = FactorPipeline()
    sw_sector_mapping = load_sw_sector_mapping()
    regenerated_factors = {}

    logger.info("开始从原始数据重新生成因子...")

    for factor_name in failed_factors:
        logger.info(f"处理因子: {factor_name}")

        # 检查原始因子是否存在
        if factor_name not in df.columns:
            logger.warning(f"  原始因子 {factor_name} 不存在，跳过")
            continue

        try:
            original_series = df[factor_name].copy()

            # 确保series有正确的name
            if original_series.name is None:
                original_series.name = factor_name

            # 确定处理类型
            processing_type = identify_factor_processing_type(factor_name)
            logger.info(f"  处理类型: {processing_type}")

            # 创建包含stock_code和date的DataFrame
            factor_df = pd.DataFrame({
                'stock_code': df['stock_code'],
                'date': df['date'],
                factor_name: original_series
            })

            # 根据处理类型进行归一化
            if processing_type == 'rankpct_only':
                # 只生成rankpct
                rank_factor = pipeline._rankpct_normalize(original_series, factor_df)
                regenerated_factors[f'{factor_name}_rank'] = rank_factor
                logger.info(f"  生成 {factor_name}_rank")

            elif processing_type == 'dual':
                # 双重归一化：z-score + rankpct + 行业内版本
                # z-score
                z_factor = pipeline._zscore_normalize_with_clipping(
                    original_series, factor_df, enable_intelligent_clipping=True
                )
                regenerated_factors[f'{factor_name}_z'] = z_factor
                logger.info(f"  生成 {factor_name}_z")

                # rankpct
                rank_factor = pipeline._rankpct_normalize(original_series, factor_df)
                regenerated_factors[f'{factor_name}_rank'] = rank_factor
                logger.info(f"  生成 {factor_name}_rank")

                # 行业内z-score
                intra_z_factor = pipeline._intra_sector_zscore_normalize_with_clipping(
                    original_series, factor_df, sw_sector_mapping, enable_intelligent_clipping=True
                )
                regenerated_factors[f'{factor_name}_intra_z'] = intra_z_factor
                logger.info(f"  生成 {factor_name}_intra_z")

                # 行业内rankpct
                intra_rank_factor = pipeline._intra_sector_rank_normalize(
                    original_series, factor_df, sw_sector_mapping
                )
                regenerated_factors[f'{factor_name}_intra_rank'] = intra_rank_factor
                logger.info(f"  生成 {factor_name}_intra_rank")

            elif processing_type == 'zscore_only':
                # 只生成z-score
                z_factor = pipeline._zscore_normalize_with_clipping(
                    original_series, factor_df, enable_intelligent_clipping=True
                )
                regenerated_factors[f'{factor_name}_z'] = z_factor
                logger.info(f"  生成 {factor_name}_z")

            # 删除旧的原始因子，保留重新生成的
            if factor_name in df.columns:
                logger.info(f"  删除原始因子 {factor_name}")

        except Exception as e:
            logger.error(f"  处理因子 {factor_name} 失败: {e}")
            # 保留原始数据作为备份
            regenerated_factors[factor_name] = df[factor_name]

    return regenerated_factors


def save_corrected_factors(original_df: pd.DataFrame, regenerated_factors: Dict[str, pd.Series], output_file: str):
    """保存修复后的因子数据"""
    logger = logging.getLogger(__name__)

    try:
        # 创建修复后的数据副本
        corrected_df = original_df.copy()

        # 首先删除需要修复的原始因子和它们的归一化版本
        factors_to_remove = set()
        for factor_name in regenerated_factors.keys():
            base_factor = factor_name.replace('_z', '').replace('_rank', '').replace('_intra_z', '').replace('_intra_rank', '')
            # 删除所有相关版本
            for suffix in ['', '_z', '_rank', '_intra_z', '_intra_rank']:
                col_to_remove = base_factor + suffix
                if col_to_remove in corrected_df.columns:
                    factors_to_remove.add(col_to_remove)

        logger.info(f"删除 {len(factors_to_remove)} 个旧因子列")
        corrected_df = corrected_df.drop(columns=factors_to_remove)

        # 添加重新生成的因子
        logger.info(f"添加 {len(regenerated_factors)} 个重新生成的因子")
        for factor_name, series in regenerated_factors.items():
            corrected_df[factor_name] = series.values

        # 保存到文件
        corrected_df.to_parquet(output_file)
        logger.info(f"修复后的因子数据已保存到: {output_file}")

        # 保存统计信息
        stats_file = output_file.replace('.parquet', '_stats.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== 智能裁剪修复统计 ===\n")
            f.write(f"原始数据形状: {original_df.shape}\n")
            f.write(f"修复后形状: {corrected_df.shape}\n")
            f.write(f"重新生成因子数: {len(regenerated_factors)}\n")
            f.write(f"删除旧因子数: {len(factors_to_remove)}\n\n")

            f.write("重新生成的因子列表:\n")
            for factor_name in sorted(regenerated_factors.keys()):
                series = regenerated_factors[factor_name]
                stats = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max()
                }
                extreme_ratio = ((series.abs() > 3).sum() / len(series)) if 'z' in factor_name else 0
                f.write(f"  {factor_name}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 范围=[{stats['min']:.4f}, {stats['max']:.4f}]")
                if extreme_ratio > 0:
                    f.write(f", 极端值比例={extreme_ratio:.2%}")
                f.write("\n")

        logger.info(f"修复统计信息已保存到: {stats_file}")

    except Exception as e:
        logger.error(f"保存修复数据失败: {e}")


def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始正确的智能裁剪修复流程")

    # 配置文件路径
    factors_file = 'C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors.parquet'
    output_file = 'C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors_corrected.parquet'

    # 1. 加载原始因子数据
    logger.info("步骤1: 加载原始因子数据...")
    original_df = load_original_factor_data(factors_file)

    if original_df.empty:
        logger.error("无法加载因子数据，退出修复流程")
        return

    # 2. 识别需要修复的因子
    logger.info("步骤2: 识别需要修复的因子...")
    failed_factors = extract_failed_factors_from_log("stock_selection_test_mode_walkforward_排序模型_lg52_因子计算.log")

    # 3. 从原始数据重新生成因子
    logger.info("步骤3: 从原始数据重新生成因子...")
    regenerated_factors = regenerate_factors_from_original(original_df, failed_factors)

    if not regenerated_factors:
        logger.error("没有成功生成任何因子，退出修复流程")
        return

    # 4. 保存修复结果
    logger.info("步骤4: 保存修复结果...")
    save_corrected_factors(original_df, regenerated_factors, output_file)

    logger.info("正确的智能裁剪修复流程完成!")
    logger.info(f"修复后的文件: {output_file}")
    logger.info("建议使用 factors_corrected.parquet 作为新的因子数据文件")


if __name__ == "__main__":
    main()