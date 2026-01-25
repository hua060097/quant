#!/usr/bin/env python3
"""
智能裁剪修复脚本

功能：
1. 解析日志文件，识别智能裁剪失败的因子
2. 加载已生成的因子数据
3. 对失败的因子重新应用智能裁剪
4. 保存修复后的因子数据

使用方法：
python fix_intelligent_clipping.py --log_file stock_selection_test_mode_walkforward_排序模型_lg52_因子计算.log
"""

import re
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.factor_lab.pipeline import FactorPipeline
from factor_system.factor_lab.store import FactorDatasetStore
from main import load_sw_sector_mapping


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fix_intelligent_clipping.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_failed_factors_from_log(log_file: str) -> List[str]:
    """从日志文件中解析智能裁剪失败的因子列表"""
    logger = logging.getLogger(__name__)
    failed_factors = set()

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析智能裁剪失败的因子
        # 模式1: [FACTOR] factor_name -> Z-score 后面跟着智能裁剪失败
        pattern1 = r'\[FACTOR\]\s+(.+?)\s+->.*?智能裁剪失败'
        matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)

        # 模式2: 更宽松的匹配，寻找因子名称
        pattern2 = r'智能裁剪失败，回退到标准裁剪.*?expected string'
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if '智能裁剪失败，回退到标准裁剪' in line:
                # 向上查找最近的因子记录
                for j in range(max(0, i-10), i):
                    if '[FACTOR]' in lines[j]:
                        factor_match = re.search(r'\[FACTOR\]\s+(.+?)\s+->', lines[j])
                        if factor_match:
                            failed_factors.add(factor_match.group(1).strip())
                            break

        failed_factors.update(matches1)

        logger.info(f"从日志中识别到 {len(failed_factors)} 个智能裁剪失败的因子")
        logger.info(f"失败因子: {sorted(failed_factors)}")

        return sorted(failed_factors)

    except FileNotFoundError:
        logger.error(f"日志文件未找到: {log_file}")
        return []
    except Exception as e:
        logger.error(f"解析日志文件失败: {e}")
        return []


def load_factor_data(factors_file: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """加载因子数据"""
    logger = logging.getLogger(__name__)

    try:
        # 尝试作为parquet文件加载
        if factors_file.endswith('.parquet'):
            df = pd.read_parquet(factors_file)
        else:
            # 尝试自动检测文件类型
            if Path(factors_file + '.parquet').exists():
                df = pd.read_parquet(factors_file + '.parquet')
            else:
                raise FileNotFoundError(f"找不到因子文件: {factors_file}")

        logger.info(f"加载因子数据成功: {df.shape}")
        logger.info(f"因子列: {list(df.columns)[:10]}...")

        # 分离因子数据
        factor_series = {}
        for col in df.columns:
            if col not in ['stock_code', 'date']:
                factor_series[col] = df[col]

        return df, factor_series

    except Exception as e:
        logger.error(f"加载因子数据失败: {e}")
        return pd.DataFrame(), {}


def apply_intelligent_clipping_to_factors(
    factor_series: Dict[str, pd.Series],
    data_df: pd.DataFrame,
    failed_factors: List[str]
) -> Dict[str, pd.Series]:
    """对指定的因子应用智能裁剪"""
    logger = logging.getLogger(__name__)

    pipeline = FactorPipeline()
    fixed_factors = {}

    # 加载SW行业映射（用于行业内处理）
    sw_sector_mapping = load_sw_sector_mapping()

    for factor_name in failed_factors:
        if factor_name not in factor_series:
            logger.warning(f"因子 {factor_name} 不在数据中，跳过")
            continue

        logger.info(f"开始修复因子: {factor_name}")

        try:
            series = factor_series[factor_name]

            # 确保series有正确的name
            if series.name is None:
                series.name = factor_name

            # 应用智能裁剪
            if factor_name.endswith('_z'):
                # z-score因子，应用智能裁剪
                clipped_series = pipeline._zscore_normalize_with_clipping(
                    series, data_df, enable_intelligent_clipping=True
                )
            elif factor_name.endswith('_rank'):
                # rank因子，不需要裁剪
                clipped_series = series.copy()
                logger.info(f"  {factor_name} 是rank因子，跳过裁剪")
            elif 'intra' in factor_name:
                # 行业内因子
                if factor_name.endswith('_intra_z'):
                    clipped_series = pipeline._intra_sector_zscore_normalize_with_clipping(
                        series, data_df, sw_sector_mapping, enable_intelligent_clipping=True
                    )
                elif factor_name.endswith('_intra_rank'):
                    clipped_series = pipeline._intra_sector_rank_normalize(
                        series, data_df, sw_sector_mapping
                    )
                else:
                    clipped_series = series.copy()
            else:
                # 普通因子，应用标准处理
                logger.info(f"  {factor_name} 应用标准z-score处理")
                clipped_series = pipeline._zscore_normalize_with_clipping(
                    series, data_df, enable_intelligent_clipping=True
                )

            fixed_factors[factor_name] = clipped_series
            logger.info(f"  ✅ {factor_name} 修复成功")

        except Exception as e:
            logger.error(f"  ❌ {factor_name} 修复失败: {e}")
            # 保留原始数据
            fixed_factors[factor_name] = factor_series[factor_name]

    return fixed_factors


def save_fixed_factors(
    original_df: pd.DataFrame,
    fixed_factors: Dict[str, pd.Series],
    output_file: str
):
    """保存修复后的因子数据"""
    logger = logging.getLogger(__name__)

    try:
        # 创建修复后的数据副本
        fixed_df = original_df.copy()

        # 更新修复的因子
        for factor_name, series in fixed_factors.items():
            if factor_name in fixed_df.columns:
                fixed_df[factor_name] = series.values
                logger.info(f"更新因子: {factor_name}")
            else:
                logger.warning(f"因子 {factor_name} 不在原始数据中")

        # 保存到文件
        if output_file.endswith('.parquet'):
            fixed_df.to_parquet(output_file)
        else:
            fixed_df.to_parquet(output_file + '.parquet')

        logger.info(f"修复后的因子数据已保存到: {output_file}")

        # 保存备份
        backup_file = output_file.replace('.parquet', '_backup.parquet')
        original_df.to_parquet(backup_file)
        logger.info(f"原始数据备份已保存到: {backup_file}")

    except Exception as e:
        logger.error(f"保存修复数据失败: {e}")


def generate_fix_report(failed_factors: List[str], fixed_factors: Dict[str, pd.Series]):
    """生成修复报告"""
    logger = logging.getLogger(__name__)

    report = []
    report.append("# 智能裁剪修复报告")
    report.append(f"修复时间: {pd.Timestamp.now()}")
    report.append("")

    report.append("## 修复统计")
    report.append(f"- 检测到的失败因子: {len(failed_factors)}")
    report.append(f"- 成功修复因子: {len(fixed_factors)}")
    report.append(f"- 修复成功率: {len(fixed_factors)/len(failed_factors)*100:.1f}%")
    report.append("")

    report.append("## 失败因子列表")
    for factor in failed_factors:
        status = "✅ 已修复" if factor in fixed_factors else "❌ 修复失败"
        report.append(f"- {factor}: {status}")

    report.append("")
    report.append("## 因子数据统计")
    for factor_name, series in fixed_factors.items():
        if hasattr(series, 'name') and series.name:
            stats = {
                'count': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'clipped_ratio': ((series.abs() > 3).sum() / len(series)) if len(series) > 0 else 0
            }
            report.append(f"### {factor_name}")
            report.append(f"- 数量: {stats['count']:,}")
            report.append(f"- 均值: {stats['mean']:.6f}")
            report.append(f"- 标准差: {stats['std']:.6f}")
            report.append(f"- 范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
            report.append(f"- 裁剪比例: {stats['clipped_ratio']:.2%}")
            report.append("")

    # 保存报告
    report_content = "\n".join(report)
    with open('intelligent_clipping_fix_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info("修复报告已保存到: intelligent_clipping_fix_report.md")
    print(report_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能裁剪修复脚本')
    parser.add_argument('--log_file', required=True, help='因子计算日志文件')
    parser.add_argument('--factors_file',
                       default='C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors.parquet',
                       help='因子数据文件')
    parser.add_argument('--output_file',
                       default='C:/Users/86151/.trae-cn/pythons/path/to/integrated_stock_selection/factor_lab_data/factors_fixed.parquet',
                       help='输出文件')

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("开始智能裁剪修复流程")

    # 1. 解析失败因子
    logger.info("步骤1: 解析日志中的失败因子...")
    failed_factors = parse_failed_factors_from_log(args.log_file)

    if not failed_factors:
        logger.info("未发现智能裁剪失败的因子，无需修复")
        return

    # 2. 加载因子数据
    logger.info("步骤2: 加载因子数据...")
    factor_df, factor_series = load_factor_data(args.factors_file)

    if factor_df.empty:
        logger.error("无法加载因子数据，退出修复流程")
        return

    # 3. 应用智能裁剪
    logger.info("步骤3: 应用智能裁剪修复...")
    fixed_factors = apply_intelligent_clipping_to_factors(
        factor_series, factor_df, failed_factors
    )

    # 4. 保存修复结果
    logger.info("步骤4: 保存修复结果...")
    save_fixed_factors(factor_df, fixed_factors, args.output_file)

    # 5. 生成报告
    logger.info("步骤5: 生成修复报告...")
    generate_fix_report(failed_factors, fixed_factors)

    logger.info("智能裁剪修复流程完成!")


if __name__ == "__main__":
    main()