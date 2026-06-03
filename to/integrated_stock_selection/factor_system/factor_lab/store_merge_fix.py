"""
分块合并优化函数

解决大数据合并时的内存分配问题
"""

import pandas as pd
import numpy as np
import logging
import os
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


def _normalise_key_frame(df: pd.DataFrame, index_columns: Sequence[str]) -> pd.DataFrame:
    """Return a small frame with comparable merge keys."""
    key_df = df[list(index_columns)].copy()
    for col in index_columns:
        if col == 'date':
            key_df[col] = pd.to_datetime(key_df[col]).dt.normalize()
        else:
            key_df[col] = key_df[col].astype('string')
    return key_df


def _key_index(df: pd.DataFrame, index_columns: Sequence[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(_normalise_key_frame(df, index_columns))


def _align_to_schema(df: pd.DataFrame, schema) -> pd.DataFrame:
    """Align DataFrame columns to an Arrow schema without changing row order."""
    aligned = pd.DataFrame(index=df.index)
    for name in schema.names:
        if name in df.columns:
            aligned[name] = df[name]
        else:
            aligned[name] = pd.NA
    return aligned


def _align_arrow_table_to_schema(table, schema):
    """Align an Arrow table to the target schema without converting to pandas."""
    import pyarrow as pa

    source_names = set(table.column_names)
    arrays = []
    for field in schema:
        if field.name in source_names:
            column = table[field.name]
            if not column.type.equals(field.type):
                column = column.cast(field.type, safe=False)
            arrays.append(column)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _mark_stream_merge_completed(
    result_df: pd.DataFrame,
    *,
    total_rows: int,
    replaced_rows: int = 0,
    updated_rows: int = 0,
    appended_rows: int = 0,
) -> pd.DataFrame:
    result_df.attrs['_stream_merged'] = True
    result_df.attrs['_already_saved'] = True
    result_df.attrs['_stream_total_rows'] = total_rows
    result_df.attrs['_stream_replaced_rows'] = replaced_rows
    result_df.attrs['_stream_updated_rows'] = updated_rows
    result_df.attrs['_stream_appended_rows'] = appended_rows
    return result_df


def chunked_merge_large_dataframes(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    index_columns: Sequence[str],
    chunk_size: int = 50000,
    overlap_mode: str = 'merge',  # 🆕 添加重叠模式参数
    add_factor_mode: bool = False,  # ✅ 添加add_factor_mode参数
    store_path: str = None  # ⭐ parquet文件路径，用于按日期过滤加载
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

    logger.info(f"[分块合并] 开始因子数据合并 - 现有: {len(existing_data):,}行, 新: {len(new_data):,}行, store_path: {store_path}")

    # ⭐ 如果有store_path，走流式合并路径。
    # batch模式（列扩展）也必须走这里；调用方可能只传入空schema壳，
    # 需要从parquet逐批读取旧数据并写回旧列+新列，避免覆盖丢失既有因子。
    if store_path and 'date' in new_data.columns:
        logger.info(f"[分块合并] 使用索引级流式合并 (低内存模式, overlap_mode={overlap_mode})")

        missing_keys = [c for c in index_columns if c not in new_data.columns]
        if missing_keys:
            raise ValueError(f"新数据缺少索引列: {missing_keys}")

        new_min_date = new_data['date'].min()
        new_max_date = new_data['date'].max()
        new_min_ts = pd.Timestamp(new_min_date)
        new_max_ts = pd.Timestamp(new_max_date)
        logger.info(f"[分块合并] 新数据日期范围: {new_min_date} ~ {new_max_date}")

        import gc, os
        writer = None
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa

            temp_path = str(store_path) + ".tmp_merge"

            pf = pq.ParquetFile(store_path)
            old_schema = pf.schema_arrow
            old_names = set(old_schema.names)
            fields = list(old_schema)
            for col in new_data.columns:
                if col not in old_names:
                    inferred = pa.Schema.from_pandas(new_data[[col]], preserve_index=False).field(col)
                    fields.append(pa.field(col, inferred.type))
            schema = pa.schema(fields)

            new_data_unique = new_data.drop_duplicates(subset=list(index_columns), keep='last').copy()
            duplicate_new_rows = len(new_data) - len(new_data_unique)
            if duplicate_new_rows:
                logger.warning(f"[分块合并] 新数据存在重复索引，已保留最后一条: {duplicate_new_rows:,}行")

            new_key_index = _key_index(new_data_unique, index_columns)
            new_key_set = set(new_key_index.tolist())
            existing_factor_cols = set(old_schema.names) - set(index_columns)
            new_payload_cols = [c for c in new_data_unique.columns if c not in index_columns]
            # 新数据含完整旧schema时，可安全整行替换；否则只更新新数据提供的因子列，避免丢失旧因子。
            full_row_replace = set(old_schema.names).issubset(set(new_data_unique.columns))
            if full_row_replace:
                logger.info("[分块合并] 新数据包含完整schema，执行索引级整行替换")
            else:
                missing_payload = sorted(existing_factor_cols - set(new_payload_cols))
                logger.warning(
                    f"[分块合并] 新数据缺少 {len(missing_payload)} 个既有因子列，"
                    "执行索引级部分列更新以保留旧因子"
                )

            # batch模式（列扩展）：构建查找表避免每个row group做全表merge
            # 将new_data_unique按索引列设为dict，key为(index_0, index_1, ...)
            # 这样每个row group只需查表，无需merge 348万行
            is_batch_mode = (overlap_mode == 'batch')
            lookup_table = None
            if is_batch_mode and not full_row_replace and new_payload_cols:
                # 构建查找表：MultiIndex -> 因子值元组
                lookup_index = _key_index(new_data_unique, index_columns)
                lookup_values = new_data_unique[new_payload_cols].values
                lookup_table = dict(zip(lookup_index.tolist(), lookup_values))
                logger.info(f"[分块合并] batch模式查找表构建完成: {len(lookup_table):,}条, {len(new_payload_cols)}列")

            kept_rows = 0
            replaced_rows = 0
            updated_rows = 0
            matched_key_set = set()
            writer = pq.ParquetWriter(temp_path, schema, compression='snappy',
                                       use_dictionary=True)
            date_col_index = old_schema.names.index('date') if 'date' in old_schema.names else None
            new_min_norm = new_min_ts.normalize()
            new_max_norm = new_max_ts.normalize()

            for row_group_idx in range(pf.num_row_groups):
                row_group_meta = pf.metadata.row_group(row_group_idx)
                date_overlap = True
                if date_col_index is not None:
                    stats = row_group_meta.column(date_col_index).statistics
                    if stats is not None and stats.has_min_max:
                        rg_min = pd.Timestamp(stats.min).normalize()
                        rg_max = pd.Timestamp(stats.max).normalize()
                        # batch模式下所有row group都需要更新（列扩展不按日期排除）
                        if not is_batch_mode:
                            date_overlap = not (rg_max < new_min_norm or rg_min > new_max_norm)

                table = pf.read_row_group(row_group_idx)
                if not date_overlap:
                    aligned_table = _align_arrow_table_to_schema(table, schema)
                    writer.write_table(aligned_table, row_group_size=max(aligned_table.num_rows, 1))
                    kept_rows += aligned_table.num_rows
                    del table, aligned_table
                    continue

                df_batch = table.to_pandas()
                batch_match_mask = pd.Series(False, index=df_batch.index)

                # batch模式下所有行都是候选匹配，无需按日期过滤
                if is_batch_mode:
                    batch_keys = _key_index(df_batch[list(index_columns)], index_columns)
                    candidate_match = pd.Series(batch_keys.isin(new_key_index), index=df_batch.index)
                    if candidate_match.any():
                        batch_match_mask.loc[candidate_match.index[candidate_match]] = True
                        matched_key_set.update(batch_keys[candidate_match.values].tolist())
                else:
                    if 'date' in index_columns:
                        batch_dates = pd.to_datetime(df_batch['date']).dt.normalize()
                        date_mask = (batch_dates >= new_min_ts.normalize()) & (batch_dates <= new_max_ts.normalize())
                    else:
                        date_mask = pd.Series(True, index=df_batch.index)

                    if date_mask.any():
                        candidate_keys = _key_index(df_batch.loc[date_mask, list(index_columns)], index_columns)
                        candidate_match = candidate_keys.isin(new_key_index)
                        if candidate_match.any():
                            candidate_positions = df_batch.index[date_mask][candidate_match]
                            batch_match_mask.loc[candidate_positions] = True
                            matched_key_set.update(candidate_keys[candidate_match].tolist())

                if full_row_replace:
                    filtered = df_batch.loc[~batch_match_mask].copy()
                    replaced_rows += int(batch_match_mask.sum())
                else:
                    filtered = df_batch.copy()
                    if batch_match_mask.any() and new_payload_cols:
                        if lookup_table is not None:
                            # batch模式：用查找表逐行更新，避免全表merge
                            batch_keys = _key_index(filtered[list(index_columns)], index_columns)
                            for col_idx, col_name in enumerate(new_payload_cols):
                                col_values = []
                                for key in batch_keys.tolist():
                                    if key in lookup_table:
                                        col_values.append(lookup_table[key][col_idx])
                                    else:
                                        col_values.append(np.nan)
                                if col_name in filtered.columns:
                                    # 覆盖旧值
                                    new_series = pd.Series(col_values, index=filtered.index, dtype='float32')
                                    filtered[col_name] = new_series.combine_first(filtered[col_name])
                                else:
                                    filtered[col_name] = pd.Series(col_values, index=filtered.index, dtype='float32')
                            updated_rows += int(batch_match_mask.sum())
                        else:
                            # 非batch模式：用merge更新（数据量较小，可以接受）
                            updates = new_data_unique[list(index_columns) + new_payload_cols]
                            filtered = filtered.merge(
                                updates,
                                on=list(index_columns),
                                how='left',
                                suffixes=('', '__new'),
                                sort=False,
                            )
                            for col in new_payload_cols:
                                new_col = f"{col}__new"
                                if new_col in filtered.columns:
                                    if col in filtered.columns:
                                        filtered[col] = filtered[new_col].combine_first(filtered[col])
                                    else:
                                        filtered[col] = filtered[new_col]
                                    filtered = filtered.drop(columns=[new_col])
                            updated_rows += int(batch_match_mask.sum())

                if len(filtered) > 0:
                    tbl = pa.Table.from_pandas(
                        _align_to_schema(filtered, schema),
                        schema=schema,
                        preserve_index=False,
                        safe=False,
                    )
                    writer.write_table(tbl, row_group_size=max(len(filtered), 1))
                    kept_rows += len(filtered)
                del table, df_batch, filtered
            gc.collect()

            # batch模式不需要追加新行（所有行已在旧文件中），否则追加旧文件中不存在的新日期行
            if is_batch_mode:
                appended_rows = 0
            else:
                if full_row_replace:
                    append_data = new_data_unique
                else:
                    append_mask = [key not in matched_key_set for key in new_key_index.tolist()]
                    append_data = new_data_unique.loc[append_mask]

                appended_rows = len(append_data)
                if appended_rows > 0:
                    tbl_new = pa.Table.from_pandas(
                        _align_to_schema(append_data, schema),
                        schema=schema,
                        preserve_index=False,
                        safe=False,
                    )
                    writer.write_table(tbl_new, row_group_size=max(len(append_data), 1))
            writer.close()
            writer = None

            pf = None  # 关闭旧文件句柄
            gc.collect()

            logger.info(
                f"[分块合并] 索引级流式完成: 保留/更新后{kept_rows:,}行, "
                f"替换{replaced_rows:,}行, 更新{updated_rows:,}行, 追加{appended_rows:,}行"
            )

            # 用新文件替换旧文件
            if os.path.exists(store_path):
                os.replace(temp_path, store_path)

            total_rows = kept_rows + appended_rows
            logger.info(f"[分块合并] 文件已替换: {store_path}, 总行数约 {total_rows:,}")
            # 标记此DataFrame为流式合并结果，调用方不应再次save覆盖
            return _mark_stream_merge_completed(
                new_data_unique,
                total_rows=total_rows,
                replaced_rows=replaced_rows,
                updated_rows=updated_rows,
                appended_rows=appended_rows,
            )

        except Exception as e:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            logger.error(f"[分块合并] 索引级流式合并失败，已保留原文件不变: {e}")
            raise

    # 1. 数据类型优化（in-place，调用方不再复用原数据，无需copy）
    logger.debug(f"[分块合并] 优化数据类型...")
    optimize_dtypes(existing_data)
    optimize_dtypes(new_data)

    # 2. 处理空数据情况
    if len(existing_data) == 0:
        logger.debug(f"[分块合并] 现有数据为空，直接返回新数据")
        return new_data

    if len(new_data) == 0:
        logger.debug(f"[分块合并] 新数据为空，直接返回现有数据")
        return existing_data

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

        # 直接按列合并：只对索引列做merge（2列），避开set_index在206列上的全量block合并
        logger.debug(f"[分块合并] 按列合并因子数据...")
        new_cols_to_add = [c for c in new_factors if c in new_data.columns and c not in existing_data.columns]
        if new_cols_to_add:
            # 仅在2列索引+新因子列上merge → 内存仅 ~60MB，不会触发206列的全量拷贝
            aligned = existing_data[list(index_columns)].merge(
                new_data[list(index_columns) + new_cols_to_add],
                on=list(index_columns), how='left'
            )
            for col in new_cols_to_add:
                existing_data[col] = aligned[col].values
        result_df = existing_data

        logger.info(f"[分块合并] 分批保存完成 - 结果: {len(result_df):,}行, {len(result_df.columns)}列")

    else:
        # 增量时间范围模式：行扩展，需要去重
        logger.debug(f"[分块合并] 执行增量时间范围模式（行扩展）...")

        # ⭐ 内存优化：用pyarrow按日期过滤直接从parquet读取非重叠行
        # 避免全量加载3.4M行到内存再过滤（内存中同时存在全量+过滤后两份数据）
        if 'date' in new_data.columns and store_path:
            new_min_date = new_data['date'].min()
            new_max_date = new_data['date'].max()
            logger.info(f"[分块合并] 新数据日期范围: {new_min_date} ~ {new_max_date}")

            # 释放全量existing_data，改用pyarrow过滤加载
            del existing_data
            import gc; gc.collect()

            try:
                import pyarrow.parquet as pq
                table = pq.read_table(
                    store_path,
                    filters=[
                        ('date', '<', pd.Timestamp(new_min_date)),
                        ('date', '>', pd.Timestamp(new_max_date)),
                    ],
                )
                existing_non_overlap = table.to_pandas()
                logger.info(f"[分块合并] pyarrow过滤加载: {len(existing_non_overlap):,}行 (排除{new_min_date}~{new_max_date})")
            except Exception as e:
                logger.warning(f"[分块合并] pyarrow过滤加载失败，回退全量加载: {e}")
                existing_non_overlap = pd.read_parquet(store_path)
                mask = (existing_non_overlap['date'] < new_min_date) | (existing_non_overlap['date'] > new_max_date)
                existing_non_overlap = existing_non_overlap[mask]
                logger.info(f"[分块合并] 回退过滤加载: {len(existing_non_overlap):,}行")

            # concat 非重叠现有数据 + 新数据（无需去重）
            result_df = pd.concat([existing_non_overlap, new_data], axis=0, ignore_index=True)
            del existing_non_overlap; gc.collect()
            logger.info(f"[分块合并] 日期过滤合并完成: {len(result_df):,}行 (无需去重)")
        elif 'date' in new_data.columns and 'date' in existing_data.columns:
            # 无store_path时的回退逻辑
            new_min_date = new_data['date'].min()
            new_max_date = new_data['date'].max()
            logger.info(f"[分块合并] 新数据日期范围: {new_min_date} ~ {new_max_date}")

            new_data = new_data.drop_duplicates(subset=list(index_columns), keep='last')
            new_index = _key_index(new_data, index_columns)
            existing_index = _key_index(existing_data, index_columns)
            overlap_mask = existing_index.isin(new_index)
            overlap_count = overlap_mask.sum()
            existing_non_overlap = existing_data[~overlap_mask]
            del existing_data; import gc; gc.collect()

            logger.info(f"[分块合并] 现有数据中重叠索引行数: {overlap_count:,}, 保留非重叠行: {len(existing_non_overlap):,}")

            result_df = pd.concat([existing_non_overlap, new_data], axis=0, ignore_index=True)
            del existing_non_overlap; gc.collect()
            logger.info(f"[分块合并] 日期过滤合并完成: {len(result_df):,}行 (无需去重)")
        else:
            # 无date列时回退到原逻辑
            logger.debug(f"[分块合并] 无date列，使用concat+去重方式")
            result_df = pd.concat([existing_data, new_data], axis=0, ignore_index=True)

            # 去重处理（按索引列去重，保留最新的数据）
            original_count = len(result_df)
            result_df = result_df.drop_duplicates(subset=index_columns, keep='last')
            removed_count = original_count - len(result_df)

            if removed_count > 0:
                logger.info(f"[分块合并] 去重完成: 移除了 {removed_count:,} 行重复索引，保留 {len(result_df):,} 行")

    # 6. 数据完整性验证（内存优化：只对新增数据做NaN检查，避免全量isna()分配巨大内存）
    if len(result_df) == 0:
        logger.error(f"[分块合并] 警告：合并结果为空！这可能导致NaN问题")
    else:
        # 只检查新增数据的NaN率（而非全量3.4M行），大幅减少内存开销
        if len(new_data) > 0:
            new_factor_cols = [c for c in new_data.columns if c not in index_columns]
            if new_factor_cols:
                new_nan_count = new_data[new_factor_cols].isna().sum().sum()
                new_total = len(new_data) * len(new_factor_cols)
                new_nan_ratio = new_nan_count / new_total if new_total > 0 else 0
                if new_nan_ratio > 0.3:
                    logger.warning(f"[分块合并] 新数据NaN率: {new_nan_ratio:.1%} ({new_nan_count:,}/{new_total:,})")
                else:
                    logger.debug(f"[分块合并] 数据完整性检查通过: 新数据NaN率 {new_nan_ratio:.1%}")

    # 7. 排序以确保一致性（内存优化：避免reset_index触发Block合并OOM）
    # batch模式（列合并）：merge left join保持行顺序不变，索引已是RangeIndex，无需reset
    # merge模式（行合并）：日期过滤合并时，现有数据保持原序+新数据追加，跳过sort避免OOM
    if overlap_mode == 'batch':
        if not isinstance(result_df.index, pd.RangeIndex):
            result_df.index = pd.RangeIndex(len(result_df))
    else:
        # 增量合并时：现有数据已有序，新数据追加到末尾，跳过全量排序避免内存峰值
        # 仅在确实需要排序（无date列的回退路径）时执行sort
        if 'date' in result_df.columns and 'date' in new_data.columns:
            # 日期过滤路径合并的结果：现有数据有序 + 新数据有序，整体已有序
            if not isinstance(result_df.index, pd.RangeIndex):
                result_df.index = pd.RangeIndex(len(result_df))
            logger.debug(f"[分块合并] 跳过全量排序（数据已按日期有序）")
        else:
            result_df = result_df.sort_values(list(index_columns), kind='quicksort')
            if not isinstance(result_df.index, pd.RangeIndex):
                result_df.index = pd.RangeIndex(len(result_df))

    logger.info(f"[分块合并] 因子数据合并完成: {len(result_df):,}行, {len(result_df.columns)}列")
    return result_df


def smart_merge_dataframes_fixed(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    index_columns: Sequence[str],
    overlap_mode: str = 'merge',
    add_factor_mode: bool = False,  # ✅ 添加add_factor_mode参数
    store_path: str = None  # ⭐ parquet文件路径，用于按日期过滤加载
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
        store_path: parquet文件路径，增量合并时用pyarrow按日期过滤加载，避免全量加载
    """
    logger = logging.getLogger(__name__)

    # 参数验证
    if existing_data is None or new_data is None:
        raise ValueError("数据不能为None")
    if not index_columns:
        raise ValueError("索引列不能为空")

    # 处理空数据情况
    # ⭐ 如果有store_path，existing_data可能只是空壳（用于列信息），实际数据从parquet读取
    if existing_data.empty and (not store_path or not os.path.exists(store_path)):
        if store_path and not os.path.exists(store_path):
            logger.info(f"store_path不存在，按首次保存处理: {store_path}")
        logger.debug("现有数据为空，返回新数据")
        return new_data.copy()
    if new_data.empty:
        if store_path:
            logger.debug("新数据为空，从store_path加载现有数据")
            return pd.read_parquet(store_path)
        logger.debug("新数据为空，返回现有数据")
        return existing_data.copy()

    # 日志：合并入口关键信息
    logger.info(
        f"[合并入口] store_path={'SET' if store_path else 'None'}, "
        f"has_date={'date' in new_data.columns if hasattr(new_data, 'columns') else 'N/A'}, "
        f"overlap_mode={overlap_mode}, "
        f"existing={len(existing_data):,}r×{len(existing_data.columns)}c, "
        f"new={len(new_data):,}r×{len(new_data.columns)}c"
    )

    # 估算内存需求，选择合适的合并策略
    # ⭐ 有store_path时，直接走分块合并（内部用pyarrow流式处理，支持所有模式）
    if store_path:
        total_rows = len(new_data)  # 无法预知existing行数，保守估计
        estimated_memory_gb = 1.5  # 强制走分块合并
    else:
        total_rows = len(existing_data) + len(new_data)
        estimated_memory_gb = total_rows * len(existing_data.columns) * 8 / (1024**3)  # 估算

    logger.debug(f"合并分析 - 现有: {len(existing_data):,}行, 新: {len(new_data):,}行, 预估内存: {estimated_memory_gb:.2f}GB")

    # 如果数据量很大或预计内存需求较高，使用分块合并
    if total_rows > 100000 or estimated_memory_gb > 1.0:
        logger.info(f"[内存优化] 检测到大数据量，使用分块合并策略")
        return chunked_merge_large_dataframes(
            existing_data, new_data, index_columns,
            overlap_mode=overlap_mode,
            add_factor_mode=add_factor_mode,
            store_path=store_path  # ✅ 传递store_path
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

    # 创建数据的副本并优化数据类型（in-place优化，调用方不再复用）
    existing_copy = existing_data.copy()
    new_copy = new_data.copy()
    optimize_dtypes(existing_copy)
    optimize_dtypes(new_copy)

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

            # 直接按列合并（使用merge避免逐列set_index的内存开销）
            new_factors_clean = set(new_copy.columns) - set(index_columns)
            new_cols_to_add = [c for c in new_factors_clean if c in new_copy.columns]
            if new_cols_to_add:
                right_df = new_copy[list(index_columns) + new_cols_to_add]
                existing_copy = existing_copy.merge(right_df, on=list(index_columns), how='left')
            result_df = existing_copy

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
        # batch模式（列合并）：merge left join保持行顺序，跳过sort避免OOM
        if overlap_mode == 'batch':
            if not isinstance(result_df.index, pd.RangeIndex):
                result_df.index = pd.RangeIndex(len(result_df))
        else:
            result_df = result_df.sort_values(list(index_columns), kind='quicksort')
            if not isinstance(result_df.index, pd.RangeIndex):
                result_df.index = pd.RangeIndex(len(result_df))

        logger.debug(f"标准合并完成 - 最终数据: {len(result_df):,}行, {len(result_df.columns)}列")
        return result_df

    except Exception as e:
        logger.error(f"标准合并失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        # 如果标准合并失败，降级到分块合并
        logger.warning("降级使用分块合并策略")
        return chunked_merge_large_dataframes(existing_data, new_data, index_columns)
