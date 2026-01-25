"""
分块合并优化函数

解决大数据合并时的内存分配问题
"""

import pandas as pd
import numpy as np
import logging
from typing import Sequence

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化DataFrame的数据类型以减少内存使用

    Args:
        df: 输入的DataFrame

    Returns:
        优化后的DataFrame
    """
    logger = logging.getLogger(__name__)
    original_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB

    # float64 -> float32 (节省50%内存)
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype('float32')
        logger.debug(f"数据类型优化: {len(float_cols)}个float64列转换为float32")

    # int64 -> int32 (节省50%内存，如果数据范围允许)
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype('int32')

    optimized_memory = df.memory_usage(deep=True).sum() / (1024**2)  # MB
    memory_savings = (original_memory - optimized_memory) / original_memory * 100

    logger.debug(f"内存优化: {original_memory:.1f}MB -> {optimized_memory:.1f}MB (节省 {memory_savings:.1f}%)")
    return df


def chunked_merge_large_dataframes(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    index_columns: Sequence[str],
    chunk_size: int = 50000,
    overlap_mode: str = 'merge',  # 🆕 添加重叠模式参数
    add_factor_mode: bool = False  # ✅ 添加add_factor_mode参数
) -> pd.DataFrame:
    """
    🆕 正确的因子数据分块合并函数

    支持两种不同的合并场景：
    1. 分批保存模式（列扩展）：相同索引，不同因子列，不去重
    2. 增量时间范围模式（行扩展）：可能重叠，需要去重

    Args:
        overlap_mode: 'merge'（增量模式，需要去重）或 'batch'（分批模式，不去重）
        add_factor_mode: 是否为新增因子模式
            - True: 列合并（因子维度增量），删除重复因子列
            - False: 行合并（时间维度增量），保留所有因子列
    """
    logger = logging.getLogger(__name__)

    logger.info(f"[分块合并] 开始因子数据合并 - 现有: {len(existing_data):,}行, 新: {len(new_data):,}行")

    # 1. 数据类型优化
    logger.debug(f"[分块合并] 优化数据类型...")
    existing_data = optimize_dtypes(existing_data.copy())
    new_data = optimize_dtypes(new_data.copy())

    # 2. 处理空数据情况
    if len(existing_data) == 0:
        logger.debug(f"[分块合并] 现有数据为空，直接返回新数据")
        return new_data.copy()

    if len(new_data) == 0:
        logger.debug(f"[分块合并] 新数据为空，直接返回现有数据")
        return existing_data.copy()

    # 3. 分析因子列
    existing_factors = set(existing_data.columns) - set(index_columns)
    new_factors = set(new_data.columns) - set(index_columns)
    factor_overlap = existing_factors & new_factors

    logger.info(f"[分块合并] 因子分析 - 现有因子: {len(existing_factors)}个, 新因子: {len(new_factors)}个, 重叠因子: {len(factor_overlap)}个")

    # ⭐ 核心修复：只有在add_factor_mode时才删除重复列
    if factor_overlap and add_factor_mode:
        # 新增因子模式：删除重复的列
        logger.warning(f"[新增因子] 发现重复因子列: {len(factor_overlap)}个，新数据将覆盖旧数据")
        existing_data = existing_data.drop(columns=list(factor_overlap))
    elif factor_overlap and not add_factor_mode:
        # ⭐ 增量更新模式：因子列重复是正常的，不删除任何列
        logger.info(f"[增量更新] 检测到因子列重叠: {len(factor_overlap)}个（正常，保留所有列）")
    # 无重叠的情况不需要处理

    # 4. 🆕 根据重叠模式选择不同的合并策略
    if overlap_mode == 'batch':
        # 分批保存模式：列扩展，不需要去重
        logger.debug(f"[分块合并] 执行分批保存模式（列扩展）...")

        # 检查是否为分批保存场景（相同索引，不同因子列）
        existing_factors = set(existing_data.columns) - set(index_columns)
        new_factors = set(new_data.columns) - set(index_columns)
        factor_overlap = existing_factors & new_factors

        if factor_overlap:
            # 有因子列重叠，新数据覆盖旧数据
            logger.warning(f"[分块合并] 检测到因子列重叠: {factor_overlap}，新数据将覆盖")
            # 移除重叠的旧因子列
            existing_data = existing_data.drop(columns=list(factor_overlap))

        # 直接按列合并（不是行合并！）
        logger.debug(f"[分块合并] 按列合并因子数据...")
        result_df = existing_data.copy()

        # 为每个新因子列添加数据（基于索引匹配）
        for col in new_factors:
            if col in new_data.columns:
                # 创建映射字典
                new_data_indexed = new_data.set_index(index_columns)[col]
                result_data_indexed = result_df.set_index(index_columns)
                result_data_indexed[col] = new_data_indexed
                result_df = result_data_indexed.reset_index()

        logger.info(f"[分块合并] 分批保存完成 - 结果: {len(result_df):,}行, {len(result_df.columns)}列")

    else:
        # 增量时间范围模式：行扩展，需要去重
        logger.debug(f"[分块合并] 执行增量时间范围模式（行扩展）...")

        # 直接拼接数据（行方向合并）
        logger.debug(f"[分块合并] 执行行方向拼接...")
        result_df = pd.concat([existing_data, new_data], axis=0, ignore_index=True)

        # 去重处理（按索引列去重，保留最新的数据）
        original_count = len(result_df)
        result_df = result_df.drop_duplicates(subset=index_columns, keep='last')
        removed_count = original_count - len(result_df)

        if removed_count > 0:
            logger.info(f"[分块合并] 去重完成: 移除了 {removed_count:,} 行重复索引，保留 {len(result_df):,} 行")

    # 6. 数据完整性验证
    if len(result_df) == 0:
        logger.error(f"[分块合并] 警告：合并结果为空！这可能导致NaN问题")
    else:
        # 检查NaN率
        total_cells = len(result_df) * len(result_df.columns)
        nan_count = result_df.isna().sum().sum()
        nan_ratio = nan_count / total_cells if total_cells > 0 else 0

        if nan_ratio > 0.3:  # 降低阈值，30%就警告
            logger.warning(f"[分块合并] 检测到高NaN率: {nan_ratio:.1%} ({nan_count:,}/{total_cells:,})")
            # 显示哪些列有较多NaN
            for col in result_df.columns:
                col_nan_count = result_df[col].isna().sum()
                if col_nan_count > 0:
                    col_nan_ratio = col_nan_count / len(result_df)
                    logger.warning(f"[分块合并] 列 '{col}' NaN率: {col_nan_ratio:.1%} ({col_nan_count:,}/{len(result_df):,})")
        else:
            logger.debug(f"[分块合并] 数据完整性检查通过: NaN率 {nan_ratio:.1%}")

    # 7. 排序以确保一致性
    result_df = result_df.sort_values(index_columns, kind='mergesort')
    result_df = result_df.reset_index(drop=True)

    logger.info(f"[分块合并] 因子数据合并完成: {len(result_df):,}行, {len(result_df.columns)}列")
    return result_df


def smart_merge_dataframes_fixed(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    index_columns: Sequence[str],
    overlap_mode: str = 'merge',
    add_factor_mode: bool = False  # ✅ 添加add_factor_mode参数
) -> pd.DataFrame:
    """
    🆕 优化的内存安全合并函数

    使用分块合并策略，避免大数据合并时的内存分配错误

    Args:
        existing_data: 现有数据
        new_data: 新数据
        index_columns: 索引列（如['date', 'stock_code']）
        overlap_mode: 重叠模式
        add_factor_mode: 是否为新增因子模式（True=列合并，False=行合并）
    """
    logger = logging.getLogger(__name__)

    # 参数验证
    if existing_data is None or new_data is None:
        raise ValueError("数据不能为None")
    if not index_columns:
        raise ValueError("索引列不能为空")

    # 处理空数据情况
    if existing_data.empty:
        logger.debug("现有数据为空，返回新数据")
        return new_data.copy()
    if new_data.empty:
        logger.debug("新数据为空，返回现有数据")
        return existing_data.copy()

    # 估算内存需求，选择合适的合并策略
    total_rows = len(existing_data) + len(new_data)
    estimated_memory_gb = total_rows * len(existing_data.columns) * 8 / (1024**3)  # 估算

    logger.debug(f"合并分析 - 现有: {len(existing_data):,}行, 新: {len(new_data):,}行, 预估内存: {estimated_memory_gb:.2f}GB")

    # 如果数据量很大或预计内存需求较高，使用分块合并
    if total_rows > 100000 or estimated_memory_gb > 1.0:
        logger.info(f"[内存优化] 检测到大数据量，使用分块合并策略")
        return chunked_merge_large_dataframes(
            existing_data, new_data, index_columns,
            overlap_mode=overlap_mode,
            add_factor_mode=add_factor_mode  # ✅ 传递参数
        )
    else:
        logger.debug(f"[常规合并] 数据量适中，使用标准合并")
        return _standard_merge_dataframes(
            existing_data, new_data, index_columns,
            overlap_mode,
            add_factor_mode=add_factor_mode  # ✅ 传递参数
        )


def _standard_merge_dataframes(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    index_columns: Sequence[str],
    overlap_mode: str = 'merge',
    add_factor_mode: bool = False  # ✅ 添加add_factor_mode参数
) -> pd.DataFrame:
    """
    🆕 正确的因子数据合并函数（用于小数据量）

    使用行方向合并，适合因子数据的特点：
    - 相同的股票和日期索引
    - 不同的因子列
    - 通过去重处理重复的索引

    Args:
        add_factor_mode: 是否为新增因子模式
            - True: 列合并（因子维度增量），删除重复因子列
            - False: 行合并（时间维度增量），保留所有因子列
    """
    logger = logging.getLogger(__name__)

    # 创建数据的副本并优化数据类型
    existing_copy = optimize_dtypes(existing_data.copy())
    new_copy = optimize_dtypes(new_data.copy())

    try:
        logger.debug("使用正确的因子数据行方向合并")

        # 验证索引列存在
        for col in index_columns:
            if col not in existing_copy.columns or col not in new_copy.columns:
                raise ValueError(f"索引列 {col} 在数据中不存在")

        # 分析因子列
        existing_factors = set(existing_copy.columns) - set(index_columns)
        new_factors = set(new_copy.columns) - set(index_columns)

        logger.debug(f"因子分析 - 现有因子: {len(existing_factors)}个, 新因子: {len(new_factors)}个")

        # 检查因子列重复
        factor_overlap = existing_factors & new_factors

        # ⭐ 核心修复：只有在add_factor_mode时才处理列重复
        # 在增量更新模式（incremental_save）下，因子列重复是正常的（时间维度扩展）
        if factor_overlap and add_factor_mode:
            # 新增因子模式：删除重复的列，添加新列
            logger.warning(f"[新增因子] 发现重复的因子列: {len(factor_overlap)}个，新数据将覆盖旧数据")
            overlap_mode = 'batch'  # 强制使用列合并模式
        elif factor_overlap and not add_factor_mode:
            # ⭐ 增量更新模式：因子列重复是正常的（时间维度扩展）
            # 不修改overlap_mode，继续使用行合并策略
            logger.info(f"[增量更新] 检测到因子列重叠: {len(factor_overlap)}个（正常，将使用行合并扩展时间范围）")
            # 不修改overlap_mode，保持原值（'merge', 'append', 'prepend'等）
        else:
            logger.debug(f"[合并检查] 无因子列重叠")

        # 🆕 根据重叠模式选择不同的合并策略
        if overlap_mode == 'batch':
            # 分批保存模式：列扩展，不去重
            logger.debug("执行分批保存模式（列扩展）")

            # 如果有因子列重叠，新数据覆盖旧数据
            if factor_overlap:
                logger.debug("检测到因子列重复，执行覆盖合并")
                existing_copy = existing_copy.drop(columns=list(factor_overlap))

            # 直接按列合并（类似分块合并的逻辑）
            result_df = existing_copy.copy()
            new_factors_clean = set(new_copy.columns) - set(index_columns)

            # 为每个新因子列添加数据（基于索引匹配）
            for col in new_factors_clean:
                if col in new_copy.columns:
                    # 创建映射字典
                    new_data_indexed = new_copy.set_index(index_columns)[col]
                    result_data_indexed = result_df.set_index(index_columns)
                    result_data_indexed[col] = new_data_indexed
                    result_df = result_data_indexed.reset_index()

            logger.debug(f"分批保存合并完成 - 结果: {len(result_df):,}行, {len(result_df.columns)}列")

        else:
            # 增量时间范围模式：行扩展，需要去重
            logger.debug("执行增量时间范围模式（行扩展）")

            # ⭐ 核心修复：在增量模式下，永远不删除因子列
            # 只在add_factor_mode且overlap_mode='batch'时才删除重复列
            if add_factor_mode and overlap_mode == 'batch' and factor_overlap:
                # 新增因子模式且有重复列：删除旧列，添加新列
                logger.debug("[新增因子] 检测到因子列重复，执行列扩展")
                existing_copy = existing_copy.drop(columns=list(factor_overlap))
            elif not add_factor_mode and factor_overlap:
                # ⭐ 增量更新模式：因子列重复是正常的，不删除任何列
                logger.debug(f"[增量更新] 检测到{len(factor_overlap)}个因子列重叠（正常，保留所有列）")
            # 其他情况不处理

            # 行方向合并（保持所有行和所有列）
            # 拼接数据
            result_df = pd.concat([existing_copy, new_copy], axis=0, ignore_index=True)

            # 去重处理：相同的索引组合保留最新的数据
            original_count = len(result_df)
            result_df = result_df.drop_duplicates(subset=index_columns, keep='last')
            removed_count = original_count - len(result_df)

            if removed_count > 0:
                logger.debug(f"去重处理: 移除了 {removed_count:,} 行重复数据")

        # 排序以确保一致性
        result_df = result_df.sort_values(index_columns, kind='mergesort')
        result_df = result_df.reset_index(drop=True)

        logger.debug(f"标准合并完成 - 最终数据: {len(result_df):,}行, {len(result_df.columns)}列")
        return result_df

    except Exception as e:
        logger.error(f"标准合并失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 如果标准合并失败，降级到分块合并
        logger.warning("降级使用分块合并策略")
        return chunked_merge_large_dataframes(existing_data, new_data, index_columns)