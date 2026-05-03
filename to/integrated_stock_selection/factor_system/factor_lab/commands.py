"""因子生成与管理命令接口。"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .pipeline import FactorPipeline, PipelineResult
from .store import FactorDatasetStore


def generate_factors(
    factor_names: Iterable[str],
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    pipeline: Optional[FactorPipeline] = None,
    market_data: Optional[pd.DataFrame] = None,
    sector_data: Optional[pd.DataFrame] = None,
    context_extra: Optional[dict] = None,
    batch_callback: Optional[callable] = None,
    batch_size: int = 10,
    normalization_config: Optional[dict] = None,
    existing_factors: Optional[set] = None,
) -> PipelineResult:
    pipe = pipeline or FactorPipeline()
    return pipe.compute(
        factor_names,
        data,
        start_date,
        end_date,
        market_data=market_data,
        sector_data=sector_data,
        context_extra=context_extra,
        batch_callback=batch_callback,
        batch_size=batch_size,
        normalization_config=normalization_config,
        existing_factors=existing_factors,
    )


def generate_and_store(
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
    enable_batch_save: bool = True,
    batch_size: int = 10,
    normalization_config: Optional[dict] = None,
    skip_existing: bool = True,
    incremental_mode: bool = False,
    add_factor_mode: bool = False,
) -> PipelineResult:
    """
    两阶段生成因子并保存到存储

    阶段1：计算原始因子并保存到临时文件（factors_raw.parquet）
    阶段2：对所有原始因子进行后处理（缩尾→清洗→归一化）并保存到正式文件（factors.parquet）

    Args:
        enable_batch_save: 是否启用分批保存（默认True）
        batch_size: 批次大小，每计算N个因子保存一次（默认10）
        normalization_config: 归一化配置
        skip_existing: 是否跳过已存在的因子（默认True，支持断点续算）
        incremental_mode: 是否启用增量保存模式（默认False）
                         - True: 保留现有数据，智能合并新数据
                         - False: 传统全量覆盖模式
        add_factor_mode: 是否启用新增因子模式（默认False）
                        - True: 新增因子模式，保留原有因子不变，仅新增指定因子（列扩展）
                        - False: 正常增量模式，可能触发overwrite

    Returns:
        PipelineResult: 包含生成因子信息和最终数据的结果对象

    Raises:
        ValueError: 当输入参数无效时
        RuntimeError: 当数据处理过程中出现错误时
    """
    import logging
    logger = logging.getLogger(__name__)

    # ========== 增量模式：分析现有数据范围 ==========
    overlap_mode = 'full'
    logger.info(f"[DEBUG] 增量更新检查 - incremental_mode: {incremental_mode}, store.exists(): {store.exists()}")

    if incremental_mode and store.exists():
        try:
            existing_data = store.load()
            logger.info(f"[DEBUG] 加载现有数据成功 - 数据形状: {existing_data.shape}, 是否为空: {existing_data.empty}")

            if not existing_data.empty:
                # 分析现有数据的日期范围
                if 'date' in existing_data.columns:
                    existing_date_range = {
                        'min_date': existing_data['date'].min(),
                        'max_date': existing_data['date'].max(),
                        'count': len(existing_data)
                    }
                    logger.info(f"[DEBUG] 增量模式 - 现有数据日期范围: {existing_date_range['min_date']} ~ {existing_date_range['max_date']}, 共{existing_date_range['count']:,}条记录")

                    # 智能确定需要生成的日期范围和重叠模式
                    requested_start = pd.to_datetime(start_date)
                    requested_end = pd.to_datetime(end_date)
                    existing_start = existing_date_range['min_date']
                    existing_end = existing_date_range['max_date']

                    logger.info(f"[DEBUG] 日期范围对比:")
                    logger.info(f"[DEBUG]   请求范围: {requested_start} ~ {requested_end}")
                    logger.info(f"[DEBUG]   现有范围: {existing_start} ~ {existing_end}")
                    logger.info(f"[DEBUG]   请求开始 >= 现有开始: {requested_start >= existing_start}")
                    logger.info(f"[DEBUG]   请求结束 <= 现有结束: {requested_end <= existing_end}")
                    logger.info(f"[DEBUG]   请求开始 > 现有结束: {requested_start > existing_end}")
                    logger.info(f"[DEBUG]   请求结束 < 现有开始: {requested_end < existing_start}")

                    # 计算需要生成的日期范围和重叠模式
                    if requested_start >= existing_start and requested_end <= existing_end:
                        # 完全在现有数据范围内，仍然生成（用于覆盖更新）
                        generation_needed = True
                        # 🆕 新增因子模式优先：强制使用列扩展模式，避免overwrite
                        if add_factor_mode:
                            overlap_mode = 'batch'
                            logger.info(f"[DEBUG] 新增因子模式 - 重叠模式: {overlap_mode} (新增因子，强制列扩展)")
                        else:
                            overlap_mode = 'overwrite'
                            logger.info(f"[DEBUG] 增量模式 - 重叠模式: {overlap_mode} (完全重叠，将覆盖更新)")
                    elif requested_start > existing_end:
                        # 新数据在现有数据之后，纯增量
                        generation_needed = True
                        # 🆕 新增因子模式：统一使用列扩展
                        if add_factor_mode:
                            overlap_mode = 'batch'
                            logger.info(f"[DEBUG] 新增因子模式 - 重叠模式: {overlap_mode} (新增因子，列扩展)")
                        else:
                            overlap_mode = 'append'
                            logger.info(f"[DEBUG] 增量模式 - 重叠模式: {overlap_mode} (纯增量追加)")
                    elif requested_end < existing_start:
                        # 新数据在现有数据之前，纯前置
                        generation_needed = True
                        # 🆕 新增因子模式：统一使用列扩展
                        if add_factor_mode:
                            overlap_mode = 'batch'
                            logger.info(f"[DEBUG] 新增因子模式 - 重叠模式: {overlap_mode} (新增因子，列扩展)")
                        else:
                            overlap_mode = 'prepend'
                            logger.info(f"[DEBUG] 增量模式 - 重叠模式: {overlap_mode} (纯前置增量)")
                    else:
                        # 部分重叠，需要合并
                        generation_needed = True
                        # 🆕 新增因子模式：统一使用列扩展
                        if add_factor_mode:
                            overlap_mode = 'batch'
                            logger.info(f"[DEBUG] 新增因子模式 - 重叠模式: {overlap_mode} (新增因子，列扩展)")
                        else:
                            overlap_mode = 'merge'
                            logger.info(f"[DEBUG] 增量模式 - 重叠模式: {overlap_mode} (部分重叠，智能合并)")

                    logger.info(f"[DEBUG] 最终确定的overlap_mode: {overlap_mode}")

                    # 详细的重叠分析
                    overlap_start = max(requested_start, existing_start)
                    overlap_end = min(requested_end, existing_end)
                    if overlap_start <= overlap_end:
                        overlap_days = (overlap_end - overlap_start).days + 1
                        logger.info(f"[DEBUG] 增量模式 - 重叠日期: {overlap_start} ~ {overlap_end}, 共{overlap_days}天")
                    else:
                        logger.info("[DEBUG] 增量模式 - 无重叠日期")
                else:
                    logger.warning("现有数据缺少date列，无法进行日期范围分析")
                    overlap_mode = 'full'
            else:
                logger.info("增量模式 - 现有数据文件为空，将执行全量生成")
                overlap_mode = 'full'
        except Exception as e:
            logger.error(f"增量模式 - 分析现有数据失败: {e}")
            logger.warning("将回退到全量生成模式")
            overlap_mode = 'full'
    else:
        if incremental_mode:
            logger.info("增量模式 - 现有数据文件不存在，将执行全量生成")
        else:
            logger.info("全量模式 - 传统覆盖保存")
        overlap_mode = 'full'

    # ========== 阶段1：计算原始因子 ==========

    # 🆕 断点续算：检查临时文件中已存在的原始因子
    existing_raw_factors = None
    if skip_existing and store.raw_exists():
        existing_raw_factors = store.get_raw_factor_names()
        if existing_raw_factors:
            logger.info(f"📊 临时文件中已存在 {len(existing_raw_factors)} 个原始因子，将跳过这些因子的计算")

    def batch_save_raw_callback(batch_data: pd.DataFrame, batch_factors: list):
        """分批保存原始因子到临时文件"""
        try:
            store.merge_raw_incremental(batch_data)
            logger.info(f"💾 已保存原始因子批次到临时文件")
        except Exception as e:
            import traceback
            logger.error(f"分批保存失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    # 根据配置决定是否使用分批保存
    callback = batch_save_raw_callback if enable_batch_save else None

    # 计算原始因子（不进行归一化）
    logger.info("🔄 阶段1：开始计算原始因子")
    result = generate_factors(
        factor_names,
        data,
        start_date,
        end_date,
        pipeline=pipeline,
        market_data=market_data,
        sector_data=sector_data,
        context_extra=context_extra,
        batch_callback=callback,
        batch_size=batch_size,
        normalization_config=None,  # 阶段1不进行归一化
        existing_factors=existing_raw_factors,
    )

    if not enable_batch_save and not result.data.empty:
        try:
            store.save_raw(result.data)
            logger.info("💾 已保存所有原始因子到临时文件")
        except Exception as e:
            import traceback
            logger.error(f"保存原始因子失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise

    # 如果没有计算任何因子（全部跳过），检查是否需要进入阶段2
    if result.data.empty and not store.raw_exists():
        logger.info("✅ 无需计算任何因子")
        return result

    # ========== 阶段2：后处理与正式保存 ==========

    logger.info("🔄 阶段2：开始后处理（缩尾→清洗→归一化）")

    # 加载所有原始因子
    try:
        all_raw_factors = store.load_raw()
        if all_raw_factors.empty:
            logger.warning("⚠️ 临时文件为空，无法进行后处理")
            return result
    except Exception as e:
        import traceback
        logger.error(f"加载临时文件失败: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        return result

    factor_count = len(all_raw_factors.columns) - len(store.index_columns)
    logger.info(f"📊 加载了 {factor_count} 个原始因子")

    factor_cols = [c for c in all_raw_factors.columns if c not in store.index_columns]

    # 对所有原始因子进行后处理
    pipe = pipeline or FactorPipeline()

    # 准备数据（需要包含date列用于cross-sectional处理）
    working = all_raw_factors.copy()
    working['date'] = pd.to_datetime(working['date'])

    # 提取因子数据（排除索引列）
    factor_cols = [c for c in working.columns if c not in store.index_columns]
    computed = {name: working[name] for name in factor_cols}
    expanding_history = None

    if normalization_config:
        expanding_factors = list(normalization_config.get('expanding_factors', []))
        expanding_anchor_start = normalization_config.get('expanding_anchor_start')
        if expanding_factors and store.exists():
            available_history_cols = [c for c in expanding_factors if c in factor_cols]
            if available_history_cols:
                try:
                    history_columns = store.index_columns + available_history_cols
                    history_source = pd.read_parquet(store.path, columns=history_columns)
                    if not history_source.empty:
                        history_source['date'] = pd.to_datetime(history_source['date'])
                        current_start_date = working['date'].min()
                        history_mask = history_source['date'] < current_start_date
                        if expanding_anchor_start:
                            history_mask &= history_source['date'] >= pd.to_datetime(expanding_anchor_start)
                        expanding_history = history_source.loc[history_mask].copy()
                        logger.info(
                            f"Expanding历史基线加载完成: {len(expanding_history):,}行, "
                            f"锚点={expanding_anchor_start}, 当前起点={current_start_date.strftime('%Y-%m-%d')}"
                        )
                except Exception as exc:
                    logger.warning(f"加载Expanding历史基线失败，将退回当前窗口计算: {exc}")

    # 应用后处理流程
    if normalization_config:
        logger.info("开始应用后处理流程：winsorize + 清洗 + 归一化")
        # 从normalization_config中获取智能裁剪设置
        enable_intelligent_clipping = normalization_config.get('enable_intelligent_clipping', True)

        # SW 行业映射现在通过 normalization_config['sw_sector_mapping'] 直接传入，不再从 sector_data 提取
        logger.debug(f"[DEBUG] SW行业映射: normalization_config['sw_sector_mapping'] = {len(normalization_config.get('sw_sector_mapping', {}))}个股票")

        processed, clipping_report = pipe._apply_postprocessing(
            computed,
            working,
            normalization_config,
            logger,
            enable_intelligent_clipping,
            expanding_history=expanding_history,
        )
        logger.info(f"后处理完成，最终因子数量: {len(processed)}")

        # 记录智能裁剪统计
        if clipping_report and enable_intelligent_clipping:
            clipped_factors = {k: v for k, v in clipping_report.items() if v.get('clip_ratio', 0) > 0}
            if clipped_factors:
                logger.info(f"智能裁剪统计: {len(clipped_factors)} 个因子被裁剪")
                for factor, info in clipped_factors.items():
                    logger.info(f"  {factor}: {info.get('strategy', 'N/A')} - 裁剪{info.get('clip_ratio', 0):.1%}")
    else:
        processed = computed
        logger.info("⚠️ 未提供归一化配置，跳过归一化步骤")
        clipping_report = {}

    # 构建最终数据
    final_data = working[store.index_columns].copy()

    # 复制数据
    logger.info(f"[调试] 开始复制处理后的因子到最终数据")
    for name, series in processed.items():
        try:
            final_data[name] = series.values
            if len(series) % 1000 == 0:  # 每复制1000个因子记录一次进度
                logger.info(f"[调试] 已复制因子 {name}: {len(series)} 个值")
        except Exception as e:
            logger.error(f"[调试] 复制因子 {name} 失败: {e}")
            logger.error(f"[调试] 因子类型: {type(series)}, 形状: {series.shape if hasattr(series, 'shape') else 'N/A'}")
            raise

    logger.info(f"[调试] 最终数据构建完成 - 形状: {final_data.shape}, 列数: {len(final_data.columns)}")
    # 🆕 根据模式选择保存策略
    logger.info(f"[DEBUG] 进入保存策略选择 - incremental_mode: {incremental_mode}, store.exists(): {store.exists()}, overlap_mode: {overlap_mode}")

    try:
        if incremental_mode and store.exists() and overlap_mode != 'full':
            # 增量模式：智能合并
            logger.info(f"[DEBUG] 增量模式 - 使用 {overlap_mode} 模式智能合并数据")

            # 加载现有数据
            existing_data = store.load()
            logger.info(f"[DEBUG] 加载现有数据用于合并 - 形状: {existing_data.shape}")

            # 导入智能合并函数
            from .store import _smart_merge_dataframes

            # 合并数据（重合日期用新数据覆盖）
            logger.info(f"[DEBUG] 开始智能合并 - 现有数据: {len(existing_data):,}条, 新数据: {len(final_data):,}条")
            merged_data = _smart_merge_dataframes(
                existing_data=existing_data,
                new_data=final_data,
                index_columns=store.index_columns,
                overlap_mode=overlap_mode,
                add_factor_mode=add_factor_mode  # ✅ 传递参数
            )
            logger.info(f"[DEBUG] 智能合并完成 - 合并后数据: {len(merged_data):,}条记录")

            # 保存合并后的数据
            store.save(merged_data)
            logger.info(f"[DEBUG] 保存合并数据完成 - 保存到: {store.path}")

            # 更新最终结果数据
            final_data = merged_data
        else:
            # 全量模式：直接覆盖保存
            if incremental_mode:
                logger.info(f"[DEBUG] 增量模式 - 无现有数据或重叠模式为full({overlap_mode})，执行全量保存")
            else:
                logger.info(f"[DEBUG] 全量模式 - 传统覆盖保存")

            # [调试] 保存前检查最终数据质量
            logger.info(f"[调试] 保存前最终数据检查 - 形状: {final_data.shape}")
            logger.info(f"[调试] 保存前最终数据列数: {len(final_data.columns)}")

            # 检查即将保存的数据质量
            final_factor_cols = [c for c in final_data.columns if c not in store.index_columns]
            logger.info(f"[调试] 即将保存 {len(final_factor_cols)} 个因子")

            save_problematic = []
            for col in final_factor_cols[:10]:  # 检查前10个
                try:
                    series = final_data[col]
                    nan_count = series.isna().sum()
                    total_count = len(series)
                    unique_count = series.nunique()

                    if nan_count == total_count:
                        logger.error(f"[调试] 即将保存的因子 {col}: 全部NaN ({nan_count}/{total_count})")
                        save_problematic.append(col)
                    elif unique_count == 1:
                        logger.error(f"[调试] 即将保存的因子 {col}: 常数列 (唯一值={unique_count})")
                        save_problematic.append(col)
                    else:
                        valid_data = series.dropna()
                        if len(valid_data) > 0:
                            mean_val = valid_data.mean()
                            std_val = valid_data.std()
                            logger.info(f"[调试] 即将保存的因子 {col}: 正常 (均值={mean_val:.6f}, 标准差={std_val:.6f}, 有效={len(valid_data)}/{total_count})")
                        else:
                            logger.error(f"[调试] 即将保存的因子 {col}: 无有效数据")
                            save_problematic.append(col)
                except Exception as e:
                    logger.error(f"[调试] 即将保存的因子 {col}: 检查失败 - {e}")
                    save_problematic.append(col)

            if save_problematic:
                logger.error(f"[调试] 即将保存的数据中发现 {len(save_problematic)} 个问题因子: {save_problematic}")
                logger.error(f"[调试] 保存后这些因子将保持NaN或常数状态")
            else:
                logger.info(f"[调试] 即将保存的数据质量检查通过")

            # 检查保存路径和权限
            try:
                import os
                logger.info(f"[调试] 检查保存路径: {store.path}")
                logger.info(f"[调试] 目录存在: {store.path.parent.exists()}")
                logger.info(f"[调试] 目录权限: {os.access(store.path.parent, os.W_OK)}")
            except Exception as e:
                logger.error(f"[调试] 检查保存路径失败: {e}")

            # 🆕 智能保存：支持增量模式和分批保存
            logger.info(f"[调试] 开始保存最终数据到正式文件: {store.path}")
            logger.info(f"[调试] 增量模式: {incremental_mode}, 重叠模式: {overlap_mode}")

            try:
                if incremental_mode and store.exists():
                    # 增量保存模式：智能合并现有数据和新数据
                    logger.info(f"[增量保存] 使用 {overlap_mode} 模式合并数据")

                    # 加载现有数据
                    existing_data = store.load()
                    logger.info(f"[增量保存] 加载现有数据: {existing_data.shape}")

                    # 使用修复后的合并函数
                    from .store_merge_fix import smart_merge_dataframes_fixed
                    merged_data = smart_merge_dataframes_fixed(
                        existing_data, final_data, store.index_columns, overlap_mode,
                        add_factor_mode=add_factor_mode  # ✅ 传递参数
                    )

                    # 保存合并后的数据
                    store.save(merged_data)

                    logger.info(f"[调试] 增量保存成功 - 正式文件大小: {store.path.stat().st_size / (1024*1024):.2f} MB")
                    logger.info(f"[DEBUG] 增量保存完成 - 现有: {len(existing_data):,}条 + 新: {len(final_data):,}条 → 合并: {len(merged_data):,}条")

                    # 更新 final_data 用于后续验证
                    final_data = merged_data

                else:
                    # 全量保存模式（包括分批保存的第一批，或明确不使用增量模式）
                    save_mode = "全量覆盖" if not incremental_mode else "首次保存"
                    logger.info(f"[全量保存] 使用 {save_mode} 模式")

                    store.save(final_data)
                    logger.info(f"[调试] 保存成功 - 正式文件大小: {store.path.stat().st_size / (1024*1024):.2f} MB")
                    logger.info(f"[DEBUG] {save_mode}完成 - 保存了 {len(final_data):,}条记录到正式文件: {store.path}")

                # 验证保存后的数据
                logger.info(f"[调试] 验证保存后的数据...")
                try:
                    verify_data = store.load()
                    logger.info(f"[调试] 验证读取成功 - 形状: {verify_data.shape}")

                    # 检查几个因子
                    verify_factor_cols = [c for c in verify_data.columns if c not in store.index_columns]
                    logger.info(f"[调试] 验证因子数量: {len(verify_factor_cols)}")

                    for col in verify_factor_cols[:5]:  # 验证前5个
                        series = verify_data[col]
                        nan_count = series.isna().sum()
                        if nan_count == len(series):
                            logger.error(f"[调试] 验证发现保存后的因子 {col}: 全部NaN")
                        else:
                            logger.info(f"[调试] 验证保存后的因子 {col}: 正常 (NaN: {nan_count}/{len(series)})")

                except Exception as e:
                    logger.error(f"[调试] 验证保存后的数据失败: {e}")

            except Exception as e:
                logger.error(f"[调试] 保存最终数据失败: {e}")
                logger.error(f"[调试] 错误类型: {type(e)}")
                import traceback
                logger.error(f"[调试] 错误详情: {traceback.format_exc()}")
                raise
    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        # 如果增量保存失败，尝试直接覆盖保存
        logger.warning("尝试回退到直接覆盖保存...")
        store.save(final_data)
        logger.info("回退保存完成")

    # 清理临时文件
    try:
        store.cleanup_raw()
        logger.debug("临时文件清理完成")
    except Exception as e:
        logger.warning(f"清理临时文件失败: {e}")

    # 返回最终结果
    return PipelineResult(factors=list(processed.keys()), data=final_data, clipping_report=clipping_report)


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
    incremental_mode: bool = False,  # 🆕 增量保存模式参数
    add_factor_mode: bool = False,  # 🆕 新增因子模式参数
) -> PipelineResult:
    """
    改进的因子生成：批次级全流程处理

    每计算batch_size个因子，立即进行完整的后处理并保存到正式文件
    避免两阶段处理的风险，实现快速失败和减少I/O开销

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

    # ========== 增量模式：分析现有数据范围 ==========
    overlap_mode = 'merge'  # 分批保存默认使用合并模式
    logger.info(f"[DEBUG] 批次处理增量更新检查 - incremental_mode: {incremental_mode}, store.exists(): {store.exists()}")

    if incremental_mode and store.exists():
        try:
            existing_data = store.load()
            logger.info(f"[DEBUG] 批次处理加载现有数据成功 - 数据形状: {existing_data.shape}, 是否为空: {existing_data.empty}")

            if not existing_data.empty and 'date' in existing_data.columns:
                # 分析现有数据的日期范围
                existing_date_range = {
                    'min_date': existing_data['date'].min(),
                    'max_date': existing_data['date'].max(),
                    'count': len(existing_data)
                }
                logger.info(f"[DEBUG] 批次处理增量模式 - 现有数据日期范围: {existing_date_range['min_date']} ~ {existing_date_range['max_date']}, 共{existing_date_range['count']:,}条记录")

                # 智能确定需要生成的日期范围和重叠模式
                requested_start = pd.to_datetime(start_date)
                requested_end = pd.to_datetime(end_date)
                existing_start = existing_date_range['min_date']
                existing_end = existing_date_range['max_date']

                logger.info(f"[DEBUG] 批次处理日期范围对比:")
                logger.info(f"[DEBUG]   请求范围: {requested_start} ~ {requested_end}")
                logger.info(f"[DEBUG]   现有范围: {existing_start} ~ {existing_end}")

                # 计算重叠模式
                if requested_start >= existing_start and requested_end <= existing_end:
                    # 🆕 新增因子模式优先：强制使用列扩展模式，避免overwrite
                    if add_factor_mode:
                        overlap_mode = 'batch'
                        logger.info(f"[DEBUG] 批次处理新增因子模式 - 重叠模式: {overlap_mode} (新增因子，强制列扩展)")
                    else:
                        overlap_mode = 'overwrite'  # 完全重叠
                        logger.info(f"[DEBUG] 批次处理增量模式 - 重叠模式: {overlap_mode} (完全重叠，将覆盖更新)")
                elif requested_start > existing_end:
                    # 🆕 新增因子模式：统一使用列扩展
                    if add_factor_mode:
                        overlap_mode = 'batch'
                        logger.info(f"[DEBUG] 批次处理新增因子模式 - 重叠模式: {overlap_mode} (新增因子，列扩展)")
                    else:
                        overlap_mode = 'append'     # 纯增量追加
                        logger.info(f"[DEBUG] 批次处理增量模式 - 重叠模式: {overlap_mode} (纯增量追加)")
                elif requested_end < existing_start:
                    # 🆕 新增因子模式：统一使用列扩展
                    if add_factor_mode:
                        overlap_mode = 'batch'
                        logger.info(f"[DEBUG] 批次处理新增因子模式 - 重叠模式: {overlap_mode} (新增因子，列扩展)")
                    else:
                        overlap_mode = 'prepend'    # 纯前置增量
                        logger.info(f"[DEBUG] 批次处理增量模式 - 重叠模式: {overlap_mode} (纯前置增量)")
                else:
                    # 🆕 新增因子模式：统一使用列扩展
                    if add_factor_mode:
                        overlap_mode = 'batch'
                        logger.info(f"[DEBUG] 批次处理新增因子模式 - 重叠模式: {overlap_mode} (新增因子，列扩展)")
                    else:
                        overlap_mode = 'merge'      # 部分重叠
                        logger.info(f"[DEBUG] 批次处理增量模式 - 重叠模式: {overlap_mode} (部分重叠，智能合并)")

                logger.info(f"[DEBUG] 批次处理最终确定的overlap_mode: {overlap_mode}")
            else:
                logger.warning("批次处理现有数据缺少date列或为空，将使用默认merge模式")
                overlap_mode = 'merge'
        except Exception as e:
            logger.error(f"批次处理增量模式 - 分析现有数据失败: {e}")
            logger.warning("将回退到默认merge模式")
            overlap_mode = 'merge'

    pipe = pipeline or FactorPipeline()

    # 断点续算：检查已存在的因子
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

    # SW 行业映射现在通过 normalization_config['sw_sector_mapping'] 直接传入，不再从 sector_data 提取
    logger.debug(f"[DEBUG] SW行业映射: normalization_config['sw_sector_mapping'] = {len(normalization_config.get('sw_sector_mapping', {}))}个股票")

    # 🆕 修复：初始化完整的结果数据结构
    logger.info(f"[DEBUG] 初始化完整数据结构 - 原始数据: {data.shape}")

    # 确保在目标日期范围内
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    target_mask = (data['date'] >= start_ts) & (data['date'] <= end_ts)
    filtered_data = data.loc[target_mask].copy().reset_index(drop=True)

    # 创建基础数据结构
    final_data = filtered_data[['stock_code', 'date']].copy()
    final_data = final_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
    final_data['date'] = pd.to_datetime(final_data['date'])

    logger.info(f"[DEBUG] 基础数据结构: {final_data.shape}")

    # 🆕 修复：加载已存在的因子到完整结构中
    if store.exists():
        logger.info(f"[DEBUG] 检测到现有数据文件，加载已存在的因子")
        existing_data = store.load()
        logger.info(f"[DEBUG] 已存在数据形状: {existing_data.shape}")

        # 确保索引列一致
        existing_data['date'] = pd.to_datetime(existing_data['date'])

        # 识别已存在的因子
        existing_factors_in_file = [c for c in existing_data.columns if c not in ['stock_code', 'date']]
        logger.info(f"[DEBUG] 已存在因子: {len(existing_factors_in_file)}个")

        # 安全合并已存在的因子
        for factor in existing_factors_in_file:
            if factor in existing_data.columns:
                # 使用索引匹配，确保数据对齐
                try:
                    # 创建临时DataFrame进行安全合并
                    temp_factor_df = existing_data[['stock_code', 'date', factor]].copy()

                    # 移除同名列（如果存在）
                    if factor in final_data.columns:
                        final_data = final_data.drop(columns=[factor])

                    # 合并新因子
                    final_data = pd.merge(
                        final_data,
                        temp_factor_df,
                        on=['stock_code', 'date'],
                        how='left'
                    )

                    logger.debug(f"[DEBUG] 成功加载已存在因子: {factor}")

                except Exception as e:
                    logger.error(f"[DEBUG] 加载已存在因子 {factor} 失败: {e}")
                    # 如果合并失败，跳过该因子
                    continue

        logger.info(f"📊 已加载 {len(existing_factors_in_file)} 个已存在的因子，当前总列数: {len(final_data.columns)}")
    else:
        logger.info(f"[DEBUG] 无现有数据文件，从空白开始")

    # 验证最终数据结构
    logger.info(f"[DEBUG] 初始化后数据结构: {final_data.shape}")
    logger.info(f"[DEBUG] 初始化后列数: {len(final_data.columns)}")
    logger.info(f"[DEBUG] 列名: {[c for c in final_data.columns[:10]]}{'...' if len(final_data.columns) > 10 else ''}")

    all_clipping_reports = {}

    # 批次级处理循环
    total_batches = (len(remaining_factors) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_factors))
        batch_factors = remaining_factors[start_idx:end_idx]

        logger.info(f"🔄 处理批次 [{batch_idx + 1}/{total_batches}]: {batch_factors}")

        try:
            # 步骤1：计算原始因子
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

            # 步骤2：立即进行后处理
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

                # 步骤3：安全合并到最终数据
            if normalization_config:
                # 后处理后的数据合并
                logger.info(f"  🔄 开始安全合并后处理因子...")
                for name, series in processed.items():
                    # 使用索引匹配进行安全合并
                    try:
                        # 创建临时DataFrame进行索引匹配
                        temp_df = pd.DataFrame({
                            'stock_code': final_data['stock_code'],
                            'date': final_data['date'],
                            name: series
                        })

                        # 移除同名列（如果存在）
                        if name in final_data.columns:
                            final_data = final_data.drop(columns=[name])

                        # 合并新因子
                        final_data = pd.merge(
                            final_data,
                            temp_df,
                            on=['stock_code', 'date'],
                            how='left'
                        )

                        logger.debug(f"  ✓ 安全合并后处理因子: {name}")

                    except Exception as e:
                        logger.error(f"  ❌ 合并后处理因子 {name} 失败: {e}")
                        continue

                logger.info(f"  ✓ 后处理完成，安全合并到最终数据: {len(processed)} 个因子")
            else:
                # 不进行归一化，直接合并原始因子
                logger.info(f"  🔄 开始安全合并原始因子...")
                for name in batch_factors:
                    if name in batch_raw_result.data.columns:
                        try:
                            # 创建临时DataFrame进行索引匹配
                            temp_df = batch_raw_result.data[['stock_code', 'date', name]].copy()

                            # 移除同名列（如果存在）
                            if name in final_data.columns:
                                final_data = final_data.drop(columns=[name])

                            # 合并新因子
                            final_data = pd.merge(
                                final_data,
                                temp_df,
                                on=['stock_code', 'date'],
                                how='left'
                            )

                            logger.debug(f"  ✓ 安全合并原始因子: {name}")

                        except Exception as e:
                            logger.error(f"  ❌ 合并原始因子 {name} 失败: {e}")
                            continue

                logger.info(f"  ✓ 直接合并原始因子: {len(batch_factors)} 个")

            # 验证合并后的数据质量
            logger.info(f"  🔍 验证批次 {batch_idx + 1} 合并后数据质量...")
            batch_factor_cols = [c for c in final_data.columns if c not in ['stock_code', 'date']]
            logger.info(f"  📊 批次合并后总列数: {len(final_data.columns)}，因子列数: {len(batch_factor_cols)}")

            # 检查新增因子的质量
            new_factor_cols = []
            if normalization_config:
                new_factor_cols = [name for name in processed.keys() if name in final_data.columns]
            else:
                new_factor_cols = [name for name in batch_factors if name in final_data.columns]

            for col in new_factor_cols[:5]:  # 检查前5个新因子
                if col in final_data.columns:
                    series = final_data[col]
                    nan_count = series.isna().sum()
                    total_count = len(series)
                    if nan_count > 0:
                        logger.warning(f"  ⚠️ 新因子 {col}: {nan_count}/{total_count} NaN值")
                    else:
                        logger.info(f"  ✅ 新因子 {col}: 全部有效")

            # 步骤4：智能保存到正式文件（方案2）
            logger.info(f"  🔄 开始批次 {batch_idx + 1} 智能保存...")
            try:
                if batch_idx == 0:
                    # 第一批：直接保存（因为是最初的完整结构）
                    store.save(final_data)
                    logger.info(f"  💾 第一批直接保存: {len(final_data.columns)} 列")
                else:
                    # 后续批次：智能合并保存
                    logger.info(f"  🔍 开始智能合并保存策略...")

                    # 加载现有数据
                    existing_data = store.load()
                    logger.info(f"  📊 加载现有数据: {existing_data.shape}")

                    # 确保日期类型一致
                    existing_data['date'] = pd.to_datetime(existing_data['date'])
                    final_data['date'] = pd.to_datetime(final_data['date'])

                    # 识别新因子
                    existing_cols = set(existing_data.columns)
                    current_cols = set(final_data.columns)
                    new_cols = current_cols - existing_cols

                    logger.info(f"  🆕 新增因子: {len(new_cols)} 个")
                    if new_cols:
                        logger.info(f"  📝 新增因子列表: {list(new_cols)[:5]}{'...' if len(new_cols) > 5 else ''}")

                    # 使用优化的合并函数
                    from .store import _smart_merge_dataframes
                    merged_data = _smart_merge_dataframes(
                        existing_data=existing_data,
                        new_data=final_data,
                        index_columns=store.index_columns,
                        overlap_mode='merge',  # 使用merge确保保留所有列
                        add_factor_mode=add_factor_mode  # ✅ 传递参数
                    )

                    # 验证合并结果
                    logger.info(f"  ✅ 智能合并完成: {merged_data.shape}")

                    # 检查合并后的NaN数量
                    total_cells = merged_data.shape[0] * merged_data.shape[1]
                    nan_cells = merged_data.isna().sum().sum()
                    if nan_cells > 0:
                        logger.warning(f"  ⚠️ 合并后存在 {nan_cells:,} 个NaN值，NaN率: {nan_cells/total_cells*100:.2f}%")

                        # 检查具体哪些列有NaN
                        nan_cols = []
                        for col in merged_data.columns:
                            col_nan_count = merged_data[col].isna().sum()
                            if col_nan_count > 0:
                                nan_cols.append(f"{col}({col_nan_count})")

                        if nan_cols:
                            logger.warning(f"  📊 NaN因子列: {', '.join(nan_cols[:10])}{'...' if len(nan_cols) > 10 else ''}")
                    else:
                        logger.info(f"  ✅ 合并后数据完整，无NaN值")

                    # 🆕 智能保存：支持增量模式和分批保存
                    if incremental_mode and store.exists():
                        # 增量保存模式：智能合并现有数据和新数据（时间范围扩展）
                        logger.info(f"  [增量保存] 使用 {overlap_mode} 模式合并数据")

                        # 加载现有数据
                        existing_data_in_store = store.load()
                        logger.info(f"  [增量保存] 加载现有数据: {existing_data_in_store.shape}")

                        # 使用修复后的合并函数（增量时间范围模式）
                        from .store_merge_fix import smart_merge_dataframes_fixed
                        final_merged_data = smart_merge_dataframes_fixed(
                            existing_data_in_store, merged_data, store.index_columns, overlap_mode,
                            add_factor_mode=add_factor_mode  # ✅ 传递参数
                        )

                        # 保存合并后的数据
                        store.save(final_merged_data)

                        logger.info(f"  💾 增量智能合并保存完成: {store.path}")
                        logger.info(f"  [DEBUG] 批次增量保存完成 - 现有: {len(existing_data_in_store):,}条 + 新: {len(merged_data):,}条 → 合并: {len(final_merged_data):,}条")

                        # 更新final_data为合并后的数据
                        final_data = final_merged_data
                    else:
                        # 分批保存模式：列扩展，不需要去重
                        save_mode = "全量覆盖" if not incremental_mode else "首次保存"
                        logger.info(f"  [分批保存] 使用 {save_mode} 模式（列扩展）")

                        if store.exists():
                            # 非首次分批保存：需要与现有数据合并（列扩展）
                            existing_data_in_store = store.load()
                            logger.info(f"  [分批保存] 加载现有数据: {existing_data_in_store.shape}")

                            # 使用修复后的合并函数（分批保存模式）
                            from .store_merge_fix import smart_merge_dataframes_fixed
                            final_merged_data = smart_merge_dataframes_fixed(
                                existing_data_in_store, merged_data, store.index_columns, overlap_mode='batch',
                                add_factor_mode=add_factor_mode  # ✅ 传递参数
                            )

                            # 保存合并后的数据
                            store.save(final_merged_data)

                            logger.info(f"  💾 分批合并保存完成: {store.path}")
                            logger.info(f"  [DEBUG] 批次保存完成 - 现有: {len(existing_data_in_store)}列 + 新: {len(merged_data)}列 → 合并: {len(final_merged_data)}列")

                            # 更新final_data为合并后的数据
                            final_data = final_merged_data
                        else:
                            # 首次分批保存：直接保存
                            store.save(merged_data)
                            logger.info(f"  💾 {save_mode}保存完成: {store.path}")
                            logger.info(f"  [DEBUG] {save_mode}完成: {len(merged_data)}列")

                            # 更新final_data为合并后的数据
                            final_data = merged_data

                logger.info(f"  ✅ 批次 {batch_idx + 1} 保存完成，当前总列数: {len(final_data.columns)}")

            except Exception as save_error:
                logger.error(f"  ❌ 批次 {batch_idx + 1} 保存失败: {save_error}")
                logger.error(f"  🔍 尝试备用保存策略...")

                # 备用保存策略：根据模式选择保存方式
                try:
                    if incremental_mode and store.exists():
                        # 增量保存备用策略
                        logger.info(f"  [增量保存] 使用备用增量保存策略")
                        existing_data_backup = store.load()
                        from .store_merge_fix import smart_merge_dataframes_fixed
                        backup_merged = smart_merge_dataframes_fixed(
                            existing_data_backup, final_data, store.index_columns, overlap_mode,
                            add_factor_mode=add_factor_mode  # ✅ 传递参数
                        )
                        store.save(backup_merged)
                        logger.info(f"  🔧 备用增量保存成功: {len(backup_merged)} 列")
                    else:
                        # 分批保存备用策略
                        logger.info(f"  [分批保存] 使用备用分批保存策略")
                        if store.exists():
                            existing_data_backup = store.load()
                            from .store_merge_fix import smart_merge_dataframes_fixed
                            backup_merged = smart_merge_dataframes_fixed(
                                existing_data_backup, final_data, store.index_columns, overlap_mode='batch',
                                add_factor_mode=add_factor_mode  # ✅ 传递参数
                            )
                            store.save(backup_merged)
                            logger.info(f"  🔧 备用分批合并保存成功: {len(backup_merged)} 列")
                        else:
                            # 首次保存
                            store.save(final_data)
                            logger.info(f"  🔧 备用首次保存成功: {len(final_data.columns)} 列")
                except Exception as backup_error:
                    logger.error(f"  ❌ 备用保存也失败: {backup_error}")
                    raise RuntimeError(f"批次 {batch_idx + 1} 保存完全失败") from save_error

        except Exception as e:
            logger.error(f"  ❌ 批次 [{batch_idx + 1}] 处理失败: {e}")
            logger.error(f"  📍 失败因子: {batch_factors}")

            # 快速失败：立即终止，避免浪费时间
            raise RuntimeError(f"批次处理失败，已处理 {batch_idx} 个批次，失败因子: {batch_factors}") from e

    # 🆕 添加最终数据验证逻辑
    logger.info(f"🔍 开始最终数据完整性验证...")

    final_factor_names = [c for c in final_data.columns if c not in ['stock_code', 'date']]

    # 1. 基础结构验证
    logger.info(f"📊 最终数据结构验证:")
    logger.info(f"  - 数据形状: {final_data.shape}")
    logger.info(f"  - 总因子数: {len(final_factor_names)}")
    logger.info(f"  - 索引列: stock_code, date")

    # 2. 索引完整性验证
    logger.info(f"🔍 索引完整性验证:")
    if 'stock_code' in final_data.columns and 'date' in final_data.columns:
        stock_unique = final_data['stock_code'].nunique()
        date_unique = final_data['date'].nunique()
        total_rows = len(final_data)
        logger.info(f"  - 唯一股票数: {stock_unique:,}")
        logger.info(f"  - 唯一日期数: {date_unique}")
        logger.info(f"  - 总记录数: {total_rows:,}")

        # 检查缺失值
        stock_missing = final_data['stock_code'].isna().sum()
        date_missing = final_data['date'].isna().sum()
        if stock_missing > 0 or date_missing > 0:
            logger.warning(f"  ⚠️ 索引列存在缺失值: stock_code({stock_missing}), date({date_missing})")
        else:
            logger.info(f"  ✅ 索引列无缺失值")
    else:
        logger.error(f"  ❌ 缺少必要的索引列")

    # 3. 因子数据质量验证
    logger.info(f"🔍 因子数据质量验证:")

    total_cells = len(final_data) * len(final_factor_names)
    total_nan_cells = 0
    fully_nan_factors = []
    constant_factors = []

    for col in final_factor_names:
        if col in final_data.columns:
            series = final_data[col]
            nan_count = series.isna().sum()
            total_nan_cells += nan_count

            # 检查是否全为NaN
            if nan_count == len(series):
                fully_nan_factors.append(col)
                logger.error(f"  ❌ 因子 {col}: 全部NaN ({nan_count}/{len(series)})")
            else:
                # 检查是否为常数列
                unique_count = series.dropna().nunique()
                if unique_count == 1:
                    constant_factors.append(col)
                    logger.warning(f"  ⚠️ 因子 {col}: 常数列 (唯一值={unique_count})")
                else:
                    valid_data = series.dropna()
                    if len(valid_data) > 0:
                        mean_val = valid_data.mean()
                        std_val = valid_data.std()
                        logger.info(f"  ✅ 因子 {col}: 正常 (均值={mean_val:.6f}, 标准差={std_val:.6f}, 有效={len(valid_data)}/{len(series)})")
        else:
            logger.error(f"  ❌ 因子 {col}: 列不存在")

    # 4. 整体质量统计
    logger.info(f"📊 整体质量统计:")
    logger.info(f"  - 总数据单元: {total_cells:,}")
    logger.info(f"  - NaN单元: {total_nan_cells:,}")
    logger.info(f"  - NaN率: {total_nan_cells/total_cells*100:.2f}%")
    logger.info(f"  - 全NaN因子数: {len(fully_nan_factors)}")
    logger.info(f"  - 常数因子数: {len(constant_factors)}")
    logger.info(f"  - 正常因子数: {len(final_factor_names) - len(fully_nan_factors)}")

    # 5. 保存前最后验证
    logger.info(f"💾 最终验证 - 即将保存到: {store.path}")
    try:
        # 验证文件路径
        store.path.parent.mkdir(parents=True, exist_ok=True)

        # 模拟保存验证
        temp_path = store.path.with_suffix('.verify.parquet')
        final_data.head(1000).to_parquet(temp_path, index=False)

        # 验证读取
        verify_data = pd.read_parquet(temp_path)
        if verify_data.shape == final_data.shape:
            logger.info(f"  ✅ 保存验证通过: {verify_data.shape}")
            temp_path.unlink()  # 删除临时文件
        else:
            logger.error(f"  ❌ 保存验证失败: 期望{final_data.shape}, 实际{verify_data.shape}")

    except Exception as verify_error:
        logger.error(f"  ❌ 保存验证异常: {verify_error}")

    logger.info(f"🎉 批次级全流程处理完成!")
    logger.info(f"  总因子数: {len(final_factor_names)}")
    logger.info(f"  数据形状: {final_data.shape}")
    logger.info(f"  裁剪统计: {len(all_clipping_reports)} 个因子有裁剪记录")

    # 6. 数据完整性总结
    if len(fully_nan_factors) == 0 and len(constant_factors) == 0:
        logger.info(f"  🎉 数据质量验证通过: 所有因子数据正常")
    elif len(fully_nan_factors) > 0:
        logger.warning(f"  ⚠️ 警告: 发现 {len(fully_nan_factors)} 个全NaN因子，需要检查处理逻辑")
    elif len(constant_factors) > 0:
        logger.warning(f"  ⚠️ 警告: 发现 {len(constant_factors)} 个常数列因子，可能影响模型性能")

    return PipelineResult(
        factors=final_factor_names,
        data=final_data,
        clipping_report=all_clipping_reports
    )


def generate_and_store_optimized(
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
    enable_batch_save: bool = True,
    normalization_config: Optional[dict] = None,
    skip_existing: bool = True,
    use_batch_processing: bool = True,
    incremental_mode: bool = False,  # 🆕 增量保存模式参数
    add_factor_mode: bool = False,  # 🆕 新增因子模式参数
) -> PipelineResult:
    """
    优化的因子生成函数，自动选择最佳处理策略

    Args:
        use_batch_processing: 是否使用批次级全流程处理（推荐True）
        其他参数与 generate_and_store_batch_processing 或 generate_and_store 相同

    Returns:
        PipelineResult: 包含最终结果和裁剪报告
    """
    import logging
    logger = logging.getLogger(__name__)

    if use_batch_processing:
        logger.info("🚀 使用批次级全流程处理模式（推荐）")
        return generate_and_store_batch_processing(
            factor_names=factor_names,
            data=data,
            start_date=start_date,
            end_date=end_date,
            store=store,
            pipeline=pipeline,
            market_data=market_data,
            sector_data=sector_data,
            context_extra=context_extra,
            batch_size=batch_size,
            normalization_config=normalization_config,
            skip_existing=skip_existing,
            incremental_mode=incremental_mode,  # 🆕 传递增量保存模式参数
            add_factor_mode=add_factor_mode,  # 🆕 传递新增因子模式参数
        )
    else:
        logger.info("🔄 使用传统两阶段处理模式")
        return generate_and_store(
            factor_names=factor_names,
            data=data,
            start_date=start_date,
            end_date=end_date,
            store=store,
            pipeline=pipeline,
            market_data=market_data,
            sector_data=sector_data,
            context_extra=context_extra,
            enable_batch_save=enable_batch_save,
            batch_size=batch_size,
            normalization_config=normalization_config,
            skip_existing=skip_existing,
            incremental_mode=incremental_mode,  # 🆕 传递增量保存模式参数
            add_factor_mode=add_factor_mode,  # 🆕 传递新增因子模式参数
        )


def drop_factors(factor_names: Iterable[str], *, store: FactorDatasetStore) -> pd.DataFrame:
    return store.drop_factors(factor_names)


__all__ = [
    "generate_factors",
    "generate_and_store",  # 传统两阶段处理（保留作为备选）
    "generate_and_store_batch_processing",  # 新的批次级全流程处理（推荐）
    "generate_and_store_optimized",  # 智能选择最佳策略
    "drop_factors",
]
