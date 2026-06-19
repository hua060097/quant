#!/usr/bin/env python3
"""
改进的因子生成逻辑：批次级全流程处理

避免两阶段处理的风险，每10个因子完成完整流程后直接保存
"""

import pandas as pd
import numpy as np
import logging
from typing import Iterable, Optional, List
from pathlib import Path

from .pipeline import FactorPipeline, PipelineResult
from .store import FactorDatasetStore


def generate_and_store_batch_processing(
    factor_names: Iterable[str],
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    store: FactorDatasetStore,
    pipeline: Optional[FactorPipeline] = None,
    market_data: Optional[pd.DataFrame] = None,
    sector_data: Optional[pd.DataFrame] = None,
    context_extra: Optional[dict] = None,
    batch_size: int = 10,
    normalization_config: Optional[dict] = None,
    skip_existing: bool = True,
) -> PipelineResult:
    """
    改进的因子生成：批次级全流程处理

    每计算batch_size个因子，立即进行完整的后处理并保存到正式文件

    Args:
        factor_names: 因子名称列表
        data: 原始数据
        start_date: 开始日期
        end_date: 结束日期
        store: 因子存储对象
        pipeline: 因子管道对象
        market_data: 市场数据
        sector_data: 行业数据
        context_extra: 额外上下文
        batch_size: 批次大小（默认10）
        normalization_config: 归一化配置
        skip_existing: 是否跳过已存在的因子

    Returns:
        PipelineResult: 包含最终结果和裁剪报告
    """
    import logging
    logger = logging.getLogger(__name__)

    pipe = pipeline or FactorPipeline()

    # 🆕 断点续算：检查已存在的因子
    factor_list = list(factor_names)
    existing_factors = set()

    if skip_existing and store.exists():
        existing_factors = set(store.get_factor_names())
        logger.info(f"📊 正式文件中已存在 {len(existing_factors)} 个因子，将跳过这些因子的计算")

    # 过滤出需要计算的因子
    remaining_factors = [f for f in factor_list if f not in existing_factors]

    if not remaining_factors:
        logger.info("✅ 所有因子已存在，无需计算")
        all_factors = store.load()
        return PipelineResult(factors=list(existing_factors), data=all_factors, clipping_report={})

    logger.info(f"🔄 开始批次级全流程处理，共 {len(remaining_factors)} 个待计算因子")

    # 🔧 准备sector mapping
    if sector_data is not None and normalization_config and 'sw_sector_mapping' not in normalization_config:
        if isinstance(sector_data, pd.DataFrame):
            if sector_data.index.name == 'stock_code' and 'SW' in sector_data.columns:
                sw_mapping = sector_data['SW'].to_dict()
                normalization_config['sw_sector_mapping'] = sw_mapping
                logger.info(f"从sector_data构建SW行业映射: {len(sw_mapping)}个股票")
            elif 'stock_code' in sector_data.columns and 'SW' in sector_data.columns:
                sw_mapping = dict(zip(sector_data['stock_code'], sector_data['SW']))
                normalization_config['sw_sector_mapping'] = sw_mapping
                logger.info(f"从sector_data构建SW行业映射: {len(sw_mapping)}个股票")

    # 🆕 初始化结果数据结构
    final_data = data[['stock_code', 'date']].copy()
    final_data = final_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
    final_data['date'] = pd.to_datetime(final_data['date'])

    # 确保在目标日期范围内
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    target_mask = (final_data['date'] >= start_ts) & (final_data['date'] <= end_ts)
    final_data = final_data.loc[target_mask].copy().reset_index(drop=True)

    all_clipping_reports = {}

    # 🆕 批次级处理循环
    total_batches = (len(remaining_factors) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_factors))
        batch_factors = remaining_factors[start_idx:end_idx]

        logger.info(f"🔄 处理批次 [{batch_idx + 1}/{total_batches}]: {batch_factors}")

        try:
            # 🔸 步骤1：计算原始因子
            batch_raw_result = pipe.compute(
                batch_factors,
                data,
                start_date,
                end_date,
                market_data=market_data,
                sector_data=sector_data,
                context_extra=context_extra,
                normalization_config=None,  # 不进行归一化
            )

            logger.info(f"  ✓ 原始因子计算完成: {len(batch_factors)} 个")

            # 🔸 步骤2：立即进行后处理
            if normalization_config:
                logger.info(f"  🔄 开始批次后处理...")

                # 准备批次数据
                batch_working = batch_raw_result.data.copy()
                batch_working['date'] = pd.to_datetime(batch_working['date'])

                # 提取因子数据
                batch_factor_cols = [c for c in batch_working.columns if c not in ['stock_code', 'date']]
                batch_computed = {name: batch_working[name] for name in batch_factor_cols}

                # 应用后处理
                enable_intelligent_clipping = normalization_config.get('enable_intelligent_clipping', True)
                processed, clipping_report = pipe._apply_postprocessing(
                    batch_computed,
                    batch_working,
                    normalization_config,
                    logger,
                    enable_intelligent_clipping
                )

                # 记录裁剪报告
                if clipping_report:
                    all_clipping_reports.update(clipping_report)
                    clipped_count = len([k for k, v in clipping_report.items() if v.get('clip_ratio', 0) > 0])
                    if clipped_count > 0:
                        logger.info(f"  ✂️ 批次裁剪完成: {clipped_count} 个因子被裁剪")

                # 🔸 步骤3：合并到最终数据
                for name, series in processed.items():
                    final_data[name] = series.values

                logger.info(f"  ✓ 后处理完成，合并到最终数据: {len(processed)} 个因子")
            else:
                # 不进行归一化，直接合并原始因子
                for name in batch_factors:
                    if name in batch_raw_result.data.columns:
                        final_data[name] = batch_raw_result.data[name].values
                logger.info(f"  ✓ 直接合并原始因子: {len(batch_factors)} 个")

            # 🔸 步骤4：立即保存到正式文件（增量保存）
            store.save(final_data)
            logger.info(f"  💾 批次保存完成: {store.path}")

        except Exception as e:
            logger.error(f"  ❌ 批次 [{batch_idx + 1}] 处理失败: {e}")
            logger.error(f"  📍 失败因子: {batch_factors}")

            # 🆕 快速失败：可以选择立即终止或继续
            raise RuntimeError(f"批次处理失败，已处理 {batch_idx} 个批次，失败因子: {batch_factors}") from e

    # 🎯 最终结果
    final_factor_names = [c for c in final_data.columns if c not in ['stock_code', 'date']]

    logger.info(f"🎉 批次级全流程处理完成!")
    logger.info(f"  总因子数: {len(final_factor_names)}")
    logger.info(f"  数据形状: {final_data.shape}")
    logger.info(f"  裁剪统计: {len(all_clipping_reports)} 个因子有裁剪记录")

    return PipelineResult(
        factors=final_factor_names,
        data=final_data,
        clipping_report=all_clipping_reports
    )


def compare_approaches():
    """
    对比两种处理方式的优缺点
    """

    comparison = """
    ## 📊 两种因子生成策略对比

    ### 🔴 当前方式：两阶段处理
    **流程：**
    ```
    阶段1: 计算所有原始因子 → 保存临时文件
    阶段2: 读取临时文件 → 统一后处理 → 保存正式文件
    ```

    **优点：**
    - 内存效率高（理论上）
    - 统一的后处理标准

    **缺点：**
    - ❌ 风险集中：归一化错误导致全盘失败
    - ❌ 延迟错误发现：Bug在最后阶段暴露
    - ❌ 双重I/O：写临时 + 读临时 + 写正式
    - ❌ 存储浪费：临时文件占用空间
    - ❌ 调试困难：无法批次级测试

    ### 🟢 改进方式：批次级全流程处理
    **流程：**
    ```
    批次1: 计算10个因子 → 后处理10个因子 → 保存正式文件
    批次2: 计算10个因子 → 后处理10个因子 → 保存正式文件
    ...
    ```

    **优点：**
    - ✅ 风险分散：单个批次失败不影响其他批次
    - ✅ 快速失败：Bug立即暴露，快速定位
    - ✅ 单次I/O：直接写入正式文件
    - ✅ 存储高效：无需临时文件
    - ✅ 易于调试：批次级别测试和验证
    - ✅ 断点续算：已处理的批次不会丢失

    **缺点：**
    - 内存占用稍高（但可控制批次大小）
    - 需要仔细设计增量合并逻辑

    ## 🎯 推荐：立即改进为批次级全流程处理！
    """

    return comparison


if __name__ == "__main__":
    print(compare_approaches())