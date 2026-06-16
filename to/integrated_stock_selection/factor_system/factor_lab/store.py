"""因子数据存储管理。"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


class FactorDatasetStore:
    """管理单文件因子数据的存储。"""

    def __init__(self, path: str | Path, index_columns: Sequence[str] = ('date', 'stock_code')) -> None:
        self.path = Path(path)
        self.index_columns = list(index_columns)
        # 临时文件路径（用于两阶段保存）
        self.raw_path = self.path.parent / f"{self.path.stem}_raw{self.path.suffix}"

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> pd.DataFrame:
        if not self.exists():
            return pd.DataFrame(columns=self.index_columns)
        return pd.read_parquet(self.path)

    def load_excluding_dates(self, start_date, end_date) -> pd.DataFrame:
        """加载排除指定日期范围的数据，使用pyarrow过滤器避免全量加载到内存再过滤。

        Args:
            start_date: 排除范围的起始日期
            end_date: 排除范围的结束日期
        """
        import logging
        logger = logging.getLogger(__name__)
        if not self.exists():
            return pd.DataFrame(columns=self.index_columns)
        try:
            import pyarrow.parquet as pq
            start_pd = pd.Timestamp(start_date)
            end_pd = pd.Timestamp(end_date)
            table = pq.read_table(
                self.path,
                filters=[
                    [('date', '<', start_pd)],
                    [('date', '>', end_pd)],
                ],
            )
            result = table.to_pandas()
            logger.info(f"[按日期过滤加载] 排除 {start_pd} ~ {end_pd}, 读取 {len(result):,} 行")
            return result
        except Exception as e:
            logger.warning(f"[按日期过滤加载] pyarrow过滤失败，回退全量加载: {e}")
            df = self.load()
            if 'date' in df.columns:
                mask = (df['date'] < pd.Timestamp(start_date)) | (df['date'] > pd.Timestamp(end_date))
                return df[mask]
            return df

    def save(self, df: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        try:
            df.to_parquet(temp_path, index=False)
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def save_factor_columns_streaming(
        self,
        index_data: pd.DataFrame,
        factor_columns: dict,
        *,
        batch_rows: int = 100_000,
    ) -> None:
        """按行分块写入宽因子表，避免 pandas 一次性构造连续二维数组。"""
        import gc
        import logging
        import os

        import pyarrow as pa
        import pyarrow.parquet as pq

        missing_index = [c for c in self.index_columns if c not in index_data.columns]
        if missing_index:
            raise ValueError(f"index_data 缺少索引列: {missing_index}")

        row_count = len(index_data)
        logger = logging.getLogger(__name__)
        bad_lengths = [name for name, values in factor_columns.items() if len(values) != row_count]
        if bad_lengths:
            raise ValueError(f"因子列长度与索引不一致: {bad_lengths[:5]}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        writer = None
        try:
            for start in range(0, row_count, batch_rows):
                end = min(start + batch_rows, row_count)
                batch = {
                    name: index_data[name].iloc[start:end]
                    for name in self.index_columns
                }
                batch.update({
                    name: values.iloc[start:end]
                    if isinstance(values, pd.Series)
                    else values[start:end]
                    for name, values in factor_columns.items()
                })
                table = pa.Table.from_pydict(batch)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path,
                        table.schema,
                        compression='snappy',
                        use_dictionary=True,
                    )
                writer.write_table(table, row_group_size=max(len(table), 1))
                del batch, table
                gc.collect()
                if end == row_count or end % 1_000_000 == 0:
                    logger.info(f"[低内存全量保存] 已写入 {end:,}/{row_count:,} 行")

            if writer is None:
                empty = pa.Table.from_pydict({
                    name: index_data[name] for name in self.index_columns
                })
                writer = pq.ParquetWriter(temp_path, empty.schema, compression='snappy')
                writer.write_table(empty)
            writer.close()
            writer = None
            os.replace(temp_path, self.path)
        finally:
            if writer is not None:
                writer.close()
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def merge_incremental(self, new_data: pd.DataFrame) -> pd.DataFrame:
        if new_data is None or new_data.empty:
            return self.load()

        base = self.load()
        combined = _merge_frames(base, new_data, self.index_columns)
        self.save(combined)
        return combined

    def drop_factors(self, factor_names: Iterable[str]) -> pd.DataFrame:
        dataset = self.load()
        drop_set = set(factor_names)
        columns_to_drop = [c for c in dataset.columns if c in drop_set]
        if columns_to_drop:
            dataset = dataset.drop(columns=columns_to_drop)
            self.save(dataset)
        return dataset

    # ========== 两阶段保存支持 ==========

    def raw_exists(self) -> bool:
        """检查临时原始因子文件是否存在"""
        return self.raw_path.exists()

    def load_raw(self) -> pd.DataFrame:
        """加载临时原始因子数据"""
        if not self.raw_exists():
            return pd.DataFrame(columns=self.index_columns)
        return pd.read_parquet(self.raw_path)

    def save_raw(self, df: pd.DataFrame) -> None:
        """保存临时原始因子数据"""
        import logging
        logger = logging.getLogger(__name__)

        if df is None or df.empty:
            logger.warning("[保存临时文件] 数据为空，跳过保存")
            return

        logger.debug(f"[保存临时文件] 开始保存 - 数据形状: {df.shape}, 列数: {len(df.columns)}")

        # 检查数据质量
        factor_cols = [c for c in df.columns if c not in self.index_columns]
        if factor_cols:
            nan_count = df[factor_cols].isna().sum().sum()
            total_count = len(df) * len(factor_cols)
            nan_ratio = nan_count / total_count if total_count > 0 else 0
            logger.debug(f"[保存临时文件] 数据质量 - NaN比率: {nan_ratio:.2%} ({nan_count}/{total_count})")

            if nan_ratio > 0.5:  # 如果超过50%的数据是NaN，发出警告
                logger.warning(f"[保存临时文件] 警告：数据质量较差，NaN比率: {nan_ratio:.2%}")

        # 创建目录
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建临时备份（如果文件已存在）
        backup_path = None
        if self.raw_path.exists():
            backup_path = self.raw_path.with_suffix('.parquet.bak')
            try:
                import shutil
                shutil.copy2(self.raw_path, backup_path)
                logger.debug(f"[保存临时文件] 创建备份: {backup_path}")
            except Exception as e:
                logger.warning(f"[保存临时文件] 备份失败: {e}")

        try:
            # 保存数据
            df.to_parquet(self.raw_path, index=False)

            # 验证保存结果
            file_size = self.raw_path.stat().st_size / (1024 * 1024)  # MB
            logger.debug(f"[保存临时文件] 保存成功 - 文件大小: {file_size:.2f} MB")

            # 尝试读取验证
            try:
                # 使用pyarrow引擎进行验证，它支持行数限制
                test_read = pd.read_parquet(self.raw_path, engine='pyarrow')
                if len(test_read) > 10:
                    test_read = test_read.head(10)

                if test_read.shape[1] != df.shape[1]:
                    logger.error(f"[保存临时文件] 验证失败：列数不匹配 {test_read.shape[1]} != {df.shape[1]}")
                else:
                    logger.debug(f"[保存临时文件] 验证通过：读取{len(test_read)}行测试成功")
            except Exception as verify_e:
                logger.error(f"[保存临时文件] 验证失败：{verify_e}")
                # 如果验证失败，恢复备份
                if backup_path and backup_path.exists():
                    try:
                        import shutil
                        shutil.move(backup_path, self.raw_path)
                        logger.warning(f"[保存临时文件] 已恢复备份文件")
                    except Exception as restore_e:
                        logger.error(f"[保存临时文件] 恢复备份失败: {restore_e}")
                raise

            # 清理备份文件
            if backup_path and backup_path.exists():
                try:
                    backup_path.unlink()
                    logger.debug(f"[保存临时文件] 清理备份文件")
                except Exception as e:
                    logger.debug(f"[保存临时文件] 清理备份失败: {e}")

        except Exception as e:
            logger.error(f"[保存临时文件] 保存失败: {e}")
            logger.error(f"[保存临时文件] 错误类型: {type(e)}")
            import traceback
            logger.error(f"[保存临时文件] 错误详情: {traceback.format_exc()}")

            # 如果保存失败，恢复备份
            if backup_path and backup_path.exists():
                try:
                    import shutil
                    shutil.move(backup_path, self.raw_path)
                    logger.warning(f"[保存临时文件] 已恢复备份文件")
                except Exception as restore_e:
                    logger.error(f"[保存临时文件] 恢复备份失败: {restore_e}")

            raise

    def merge_raw_incremental(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """增量合并到临时原始因子文件"""
        import logging
        logger = logging.getLogger(__name__)

        if new_data is None or new_data.empty:
            logger.debug("[增量合并] 新数据为空，返回现有数据")
            return self.load_raw()

        logger.debug(f"[增量合并] 开始合并 - 新数据形状: {new_data.shape}, 列: {list(new_data.columns)}")

        # 检查新数据质量
        new_factor_cols = [c for c in new_data.columns if c not in self.index_columns]
        new_nan_count = new_data[new_factor_cols].isna().sum().sum()
        logger.debug(f"[增量合并] 新数据质量检查 - NaN总数: {new_nan_count}")

        # 加载现有数据
        base = self.load_raw()
        logger.debug(f"[增量合并] 加载现有数据 - 形状: {base.shape}")

        if base.empty and not self.raw_exists():
            logger.info("[增量合并] 临时原始因子文件不存在，直接创建首批 raw 数据")
            self.save_raw(new_data)
            logger.debug(f"[增量合并] 首批保存成功 - 文件大小: {self.raw_path.stat().st_size / (1024*1024):.2f} MB")
            return new_data

        if not base.empty:
            # 检查现有数据质量
            base_factor_cols = [c for c in base.columns if c not in self.index_columns]
            base_nan_count = base[base_factor_cols].isna().sum().sum()
            logger.debug(f"[增量合并] 现有数据质量检查 - NaN总数: {base_nan_count}")

        # 合并数据
        try:
            combined = _merge_frames(base, new_data, self.index_columns, store_path=str(self.raw_path))
            logger.debug(f"[增量合并] 合并成功 - 结果形状: {combined.shape}")

            # 检查合并后数据质量
            if not combined.empty:
                combined_factor_cols = [c for c in combined.columns if c not in self.index_columns]
                combined_nan_count = combined[combined_factor_cols].isna().sum().sum()
                logger.debug(f"[增量合并] 合并后质量检查 - NaN总数: {combined_nan_count}")

                # 检查是否有新的NaN产生
                if combined_nan_count > new_nan_count:
                    logger.warning(f"[增量合并] 警告：合并后NaN数量增加 ({new_nan_count} -> {combined_nan_count})")

                    # 检查具体哪些因子出现问题
                    problematic_factors = []
                    for col in combined_factor_cols:
                        if col in new_data.columns and col in base.columns:
                            new_nan = new_data[col].isna().sum()
                            base_nan = base[col].isna().sum()
                            combined_nan = combined[col].isna().sum()

                            if combined_nan > max(new_nan, base_nan):
                                problematic_factors.append((col, new_nan, base_nan, combined_nan))
                                logger.error(f"[增量合并] 因子 {col} 出现问题: 新数据NaN={new_nan}, 现有数据NaN={base_nan}, 合并后NaN={combined_nan}")

                    if problematic_factors:
                        logger.error(f"[增量合并] 发现 {len(problematic_factors)} 个问题因子: {[f[0] for f in problematic_factors[:5]]}")

            # 流式合并已直接替换磁盘文件，不能再用返回的批次壳覆盖 raw 文件。
            if hasattr(combined, 'attrs') and (
                combined.attrs.get('_already_saved') or combined.attrs.get('_stream_merged')
            ):
                logger.debug("[增量合并] 流式合并已写入磁盘，跳过二次保存")
            else:
                self.save_raw(combined)
                logger.debug(f"[增量合并] 保存成功 - 文件大小: {self.raw_path.stat().st_size / (1024*1024):.2f} MB")
        except Exception as e:
            logger.error(f"[增量合并] 合并失败: {e}")
            logger.error(f"[增量合并] 错误类型: {type(e)}")
            import traceback
            logger.error(f"[增量合并] 错误详情: {traceback.format_exc()}")
            raise

        return combined

    def cleanup_raw(self) -> None:
        """清理临时原始因子文件"""
        if self.raw_exists():
            import logging
            logger = logging.getLogger(__name__)
            keep_raw = os.environ.get("KEEP_FACTOR_RAW", "").strip().lower()
            if keep_raw in {"1", "true", "yes", "y"}:
                logger.info(f"🧪 保留临时文件用于调试: {self.raw_path}")
                return
            snapshot_raw = os.environ.get("SNAPSHOT_FACTOR_RAW", "").strip().lower()
            if snapshot_raw in {"1", "true", "yes", "y"}:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = self.raw_path.with_name(f"{self.raw_path.stem}_{stamp}{self.raw_path.suffix}")
                try:
                    shutil.copy2(self.raw_path, snapshot_path)
                    logger.info(f"📸 临时文件快照已保存: {snapshot_path}")
                except Exception as exc:
                    logger.warning(f"临时文件快照保存失败: {exc}")
            logger.info(f"🗑️ 清理临时文件: {self.raw_path}")
            self.raw_path.unlink()

    def get_raw_factor_names(self) -> set:
        """获取临时文件中的因子名称"""
        if not self.raw_exists():
            return set()
        raw_data = self.load_raw()
        if raw_data.empty:
            return set()
        return set(raw_data.columns) - set(self.index_columns)

    def get_factor_names(self) -> set:
        """获取正式文件中的因子名称"""
        if not self.exists():
            return set()
        data = self.load()
        if data.empty:
            return set()
        return set(data.columns) - set(self.index_columns)


def _smart_merge_dataframes(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    index_columns: Sequence[str],
    overlap_mode: str = 'merge',
    add_factor_mode: bool = False,  # ✅ 添加add_factor_mode参数
    store_path: str = None  # ⭐ parquet文件路径
) -> pd.DataFrame:
    """
    🆕 优化的内存安全合并函数

    使用分块合并策略，避免大数据合并时的内存分配错误
    保持与原函数完全相同的接口和行为

    Args:
        add_factor_mode: 是否为新增因子模式
            - True: 列合并（因子维度增量），删除重复因子列
            - False: 行合并（时间维度增量），保留所有因子列
        store_path: parquet文件路径，增量合并时用pyarrow按日期过滤加载
    """
    import logging
    logger = logging.getLogger(__name__)

    # 保持与原函数相同的参数验证
    if existing_data is None:
        raise ValueError("existing_data cannot be None")
    if new_data is None:
        raise ValueError("new_data cannot be None")
    if not index_columns:
        raise ValueError("index_columns cannot be empty")

    # 保持与原函数相同的空数据处理
    # ⭐ 有store_path时，existing_data可能只是空壳（仅用于列信息），不能直接返回新数据
    if existing_data.empty and not store_path:
        logger.info("现有数据为空，直接返回新数据副本")
        return new_data.copy()

    if new_data.empty:
        if store_path:
            logger.info("新数据为空，从store_path加载现有数据")
            return pd.read_parquet(store_path)
        logger.info("新数据为空，直接返回现有数据副本")
        return existing_data.copy()

    # 统计信息记录
    if store_path and existing_data.empty:
        logger.info(f"合并前统计 - 现有数据从store_path加载, 新数据: {len(new_data):,}条")
    else:
        logger.info(f"合并前统计 - 现有数据: {len(existing_data):,}条, 新数据: {len(new_data):,}条")

    if 'date' in new_data.columns:
        new_min_date = new_data['date'].min()
        new_max_date = new_data['date'].max()
        logger.info(f"新数据日期范围: {new_min_date} ~ {new_max_date}")
        if not existing_data.empty and 'date' in existing_data.columns:
            existing_min_date = existing_data['date'].min()
            existing_max_date = existing_data['date'].max()
            logger.info(f"现有数据日期范围: {existing_min_date} ~ {existing_max_date}")

    # 🎯 核心优化：调用优化版本的合并函数
    try:
        from .store_merge_fix import smart_merge_dataframes_fixed

        logger.debug(f"[优化] 调用分块优化合并函数 - 数据量: {len(existing_data):,} + {len(new_data):,}")
        result_df = smart_merge_dataframes_fixed(
            existing_data, new_data, index_columns, overlap_mode,
            add_factor_mode=add_factor_mode,
            store_path=store_path  # ✅ 传递store_path
        )

        # 保持与原函数相同的最终统计记录
        if hasattr(result_df, 'attrs') and result_df.attrs.get('_stream_merged'):
            total_rows = result_df.attrs.get('_stream_total_rows', len(result_df))
            logger.info(f"合并完成 - 最终数据: {total_rows:,}条记录 (流式合并，文件已直接写入)")
            # 流式合并已在磁盘完成，不需要再次save
            result_df.attrs['_already_saved'] = True
        else:
            logger.info(f"合并完成 - 最终数据: {len(result_df):,}条记录")

        if 'date' in result_df.columns:
            final_min_date = result_df['date'].min()
            final_max_date = result_df['date'].max()
            logger.info(f"最终数据日期范围: {final_min_date} ~ {final_max_date}")

        return result_df

    except Exception as e:
        logger.error(f"[优化] 分块合并失败，降级到简单处理: {e}")

        if store_path and not add_factor_mode and 'date' in new_data.columns:
            raise RuntimeError(
                "索引级流式增量合并失败。为避免按日期范围粗粒度覆盖导致因子数据损坏，"
                "已中止保存，原 parquet 文件保持不变。"
            ) from e

        # 降级处理：根据合并模式选择不同策略
        try:
            if add_factor_mode:
                # 列合并模式：只在索引列上merge避免全量拷贝
                logger.warning("[降级] add_factor模式：使用merge进行列合并")
                existing_factors = set(existing_data.columns) - set(index_columns)
                new_factors = set(new_data.columns) - set(index_columns)
                new_cols_to_add = [c for c in new_factors if c in new_data.columns]
                if new_cols_to_add:
                    aligned = existing_data[list(index_columns)].merge(
                        new_data[list(index_columns) + new_cols_to_add],
                        on=list(index_columns), how='left'
                    )
                    for col in new_cols_to_add:
                        existing_data[col] = aligned[col].values
                    result_df = existing_data
                else:
                    result_df = existing_data
            else:
                # 行合并模式：分批流式合并，避免全量加载到内存
                if 'date' in new_data.columns and store_path:
                    new_min_date = new_data['date'].min()
                    new_max_date = new_data['date'].max()
                    new_min_ts = pd.Timestamp(new_min_date)
                    new_max_ts = pd.Timestamp(new_max_date)
                    try:
                        import pyarrow.parquet as pq, pyarrow as pa, os, gc
                        from .store_merge_fix import optimize_dtypes
                        temp_path = str(store_path) + ".tmp_fallback"
                        pf = pq.ParquetFile(store_path)
                        # 从原文件schema派生float32版本
                        orig_schema = pf.schema_arrow
                        fields_32 = []
                        for field in orig_schema:
                            if field.type == pa.float64() or str(field.type) == 'float':
                                fields_32.append(pa.field(field.name, pa.float32()))
                            else:
                                fields_32.append(field)
                        schema_32 = pa.schema(fields_32)
                        kept_rows = 0; first_batch = True
                        for batch in pf.iter_batches(batch_size=200000):
                            df_batch = batch.to_pandas()
                            optimize_dtypes(df_batch)
                            mask = (df_batch['date'] < new_min_ts) | (df_batch['date'] > new_max_ts)
                            filtered = df_batch[mask]
                            if len(filtered) > 0:
                                rb = pa.RecordBatch.from_pandas(filtered, schema=schema_32)
                                if first_batch:
                                    writer = pq.ParquetWriter(temp_path, schema_32, compression='snappy')
                                    first_batch = False
                                writer.write_batch(rb)
                                kept_rows += len(filtered)
                            del df_batch, filtered
                        gc.collect()
                        new_data_opt = new_data.copy(); optimize_dtypes(new_data_opt)
                        if first_batch:
                            writer = pq.ParquetWriter(temp_path, schema_32, compression='snappy')
                            first_batch = False
                        writer.write_batch(pa.RecordBatch.from_pandas(new_data_opt, schema=schema_32))
                        writer.close(); del new_data_opt
                        pf = None; gc.collect()
                        os.replace(temp_path, store_path)
                        logger.warning(f"[降级] 分批流式合并完成: 保留{kept_rows:,}行 + 新增{len(new_data):,}行")
                        result_df = new_data.copy()
                    except Exception as pa_err:
                        logger.warning(f"[降级] 流式合并失败: {pa_err}")
                        if 'date' in existing_data.columns and not existing_data.empty:
                            from .store_merge_fix import _key_index
                            new_dedup = new_data.drop_duplicates(subset=list(index_columns), keep='last')
                            new_index = _key_index(new_dedup, index_columns)
                            existing_index = _key_index(existing_data, index_columns)
                            overlap_mask = existing_index.isin(new_index)
                            existing_non_overlap = existing_data[~overlap_mask]
                            logger.warning(f"[降级] 回退索引过滤+concat: 排除{overlap_mask.sum():,}行")
                            result_df = pd.concat([existing_non_overlap, new_dedup], axis=0, ignore_index=True)
                            result_df = result_df.reset_index(drop=True)
                        else:
                            result_df = pd.concat([existing_data, new_data], axis=0, ignore_index=True)
                            result_df = result_df.drop_duplicates(subset=index_columns, keep='last')
                            logger.warning(f"[降级] concat+去重: {len(result_df):,}行")
                elif 'date' in new_data.columns and 'date' in existing_data.columns and not existing_data.empty:
                    from .store_merge_fix import _key_index
                    new_dedup = new_data.drop_duplicates(subset=list(index_columns), keep='last')
                    new_index = _key_index(new_dedup, index_columns)
                    existing_index = _key_index(existing_data, index_columns)
                    overlap_mask = existing_index.isin(new_index)
                    existing_non_overlap = existing_data[~overlap_mask]
                    logger.warning(f"[降级] 使用索引过滤+concat方式: 排除{overlap_mask.sum():,}行重叠数据")
                    result_df = pd.concat([existing_non_overlap, new_dedup], axis=0, ignore_index=True)
                    result_df = result_df.reset_index(drop=True)
                else:
                    # 无date列时回退到concat+去重
                    logger.warning("[降级] 使用简单的concat+去重方式作为备选方案")
                    result_df = pd.concat([existing_data, new_data], axis=0, ignore_index=True)
                    result_df = result_df.drop_duplicates(subset=index_columns, keep='last')
                    if len(index_columns) > 1:
                        sort_columns = index_columns
                    else:
                        sort_columns = index_columns[0]
                    result_df = result_df.sort_values(sort_columns, kind='quicksort')
                    result_df = result_df.reset_index(drop=True)

            logger.info(f"[降级] 简单合并完成: {len(result_df):,}条记录")
            return result_df

        except Exception as fallback_e:
            logger.error(f"[降级] 简单合并也失败: {fallback_e}")
            raise ValueError(f"所有合并策略都失败了: 原始错误={e}, 降级错误={fallback_e}")


def _merge_frames(base: pd.DataFrame, new: pd.DataFrame, index_cols: Sequence[str], overlap_mode: str = 'auto', store_path: str = None) -> pd.DataFrame:
    """
    🆕 智能合并函数，自动判断合并模式

    Args:
        overlap_mode: 'auto'（自动判断）, 'merge'（增量合并）, 'batch'（分批保存）
        store_path: parquet文件路径，增量合并时用pyarrow按日期过滤加载
    """
    from .store_merge_fix import smart_merge_dataframes_fixed
    import logging
    logger = logging.getLogger(__name__)

    if overlap_mode == 'auto':
        # 自动判断合并模式
        if len(base) == len(new):
            # 检查是否为分批保存场景（相同行数，不同因子列）
            base_factors = set(base.columns) - set(index_cols)
            new_factors = set(new.columns) - set(index_cols)
            factor_overlap = base_factors & new_factors

            if len(base_factors) > 0 and len(new_factors) > 0 and len(factor_overlap) == 0:
                # 相同行数，不同因子列，无重叠 → 分批保存模式
                overlap_mode = 'batch'
                logger.debug(f"[智能合并] 自动判断为分批保存模式 - 现有因子: {len(base_factors)}, 新因子: {len(new_factors)}")
            else:
                # 其他情况 → 增量合并模式
                overlap_mode = 'merge'
                logger.debug(f"[智能合并] 自动判断为增量合并模式 - 重叠因子: {len(factor_overlap)}")
        else:
            # 行数不同 → 增量合并模式
            overlap_mode = 'merge'
            logger.debug(f"[智能合并] 自动判断为增量合并模式 - 行数不同: {len(base)} vs {len(new)}")

    logger.debug(f"[智能合并] 使用 {overlap_mode} 模式进行合并")
    return smart_merge_dataframes_fixed(base, new, index_cols, overlap_mode, store_path=store_path)
