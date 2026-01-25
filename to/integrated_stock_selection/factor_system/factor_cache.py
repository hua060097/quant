#!/usr/bin/env python3
"""
智能因子缓存管理器

提供因子组级别的精细化缓存管理，支持版本控制和增量更新
"""

import pandas as pd
import numpy as np
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

class FactorCacheManager:
    """智能因子缓存管理器"""
    
    def __init__(self, cache_dir: str = "./factor_cache", max_workers: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

        # 线程锁，用于并发安全
        self._lock = threading.RLock()

        # 内存缓存，用于热数据快速访问
        self._memory_cache = {}
        self._memory_cache_size = 0
        self._max_memory_cache_size = 500 * 1024 * 1024  # 500MB

        # 缓存索引文件
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存索引"""
        try:
            if self.index_file.exists():
                import json
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                self.logger.debug(f"加载缓存索引: {len(index)} 个因子组")
                return index
            else:
                self.logger.debug("缓存索引文件不存在，创建新索引")
                return {}
        except Exception as e:
            self.logger.warning(f"加载缓存索引失败: {e}，创建新索引")
            return {}

    def _save_cache_index(self):
        """保存缓存索引"""
        try:
            import json
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存缓存索引失败: {e}")
    
    def _get_cache_file_path(self, group_name: str, start_date: str, end_date: str) -> Path:
        """获取缓存文件路径（基于日期范围）"""
        filename = f"{group_name}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename
    
    def _generate_data_key(self, stock_code: str, date: str) -> str:
        """生成数据键"""
        return f"{stock_code}_{date}"
    
    def _is_cache_valid(self, group_name: str, start_date: str, end_date: str, ttl_days: int = 30) -> bool:
        """检查缓存是否有效（基于日期范围）"""
        cache_file = self._get_cache_file_path(group_name, start_date, end_date)

        if not cache_file.exists():
            return False

        # 检查文件修改时间
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            current_time = datetime.now()
            time_diff = current_time - file_time

            return time_diff.total_seconds() < ttl_days * 24 * 3600
        except Exception:
            return False
    
    def get_cached_features_incremental(self, group_name: str, start_date: str, end_date: str,
                                      trading_dates: List[str], force_recalc: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        增量缓存逻辑：按因子组和交易日分片缓存，支持增量更新

        Args:
            group_name: 因子组名称
            start_date: 开始日期
            end_date: 结束日期
            trading_dates: 交易日列表
            force_recalc: 是否强制重新计算

        Returns:
            (已缓存数据, 需要计算的交易日列表)
        """
        try:
            if force_recalc:
                self.logger.info(f"强制重新计算因子组 {group_name}")
                return pd.DataFrame(), trading_dates

            # 创建因子组专用缓存目录
            group_cache_dir = self.cache_dir / group_name
            group_cache_dir.mkdir(exist_ok=True)

            # 高效检查：批量检查文件存在性
            cached_dates = []
            missing_dates = []

            for date_str in trading_dates:
                date_simple = self._format_date_simple(date_str)
                cache_file = group_cache_dir / f"{date_simple}.parquet"

                if cache_file.exists():
                    cached_dates.append(date_str)
                else:
                    missing_dates.append(date_str)

            self.logger.info(f"因子组 {group_name} 缓存检查: {len(cached_dates)}/{len(trading_dates)} 已缓存")

            # 如果没有缓存数据，直接返回
            if not cached_dates:
                return pd.DataFrame(), missing_dates

            # 快速加载已缓存的数据
            cached_data_list = []
            for date_str in cached_dates:
                date_simple = self._format_date_simple(date_str)
                cache_file = group_cache_dir / f"{date_simple}.parquet"

                try:
                    daily_data = pd.read_parquet(cache_file)
                    if not daily_data.empty:
                        cached_data_list.append(daily_data)

                except Exception as e:
                    self.logger.warning(f"加载日期 {date_str} 缓存失败: {e}")
                    missing_dates.append(date_str)

            # 合并缓存数据
            if cached_data_list:
                cached_data = pd.concat(cached_data_list, ignore_index=True)
                self.logger.info(f"成功加载缓存: {len(cached_data)} 条记录")
            else:
                cached_data = pd.DataFrame()

            return cached_data, missing_dates

        except Exception as e:
            self.logger.error(f"增量缓存检查失败 {group_name}: {e}")
            return pd.DataFrame(), trading_dates

    def _format_date_simple(self, date_str: str) -> str:
        """将日期格式转换为YYMMDD"""
        try:
            # 处理各种日期格式
            if isinstance(date_str, str):
                # 移除时间部分
                date_part = date_str.split(' ')[0]
                # 解析日期
                if '-' in date_part:
                    year, month, day = date_part.split('-')
                    return f"{year[2:]}{month.zfill(2)}{day.zfill(2)}"
                elif len(date_part) == 8:  # YYYYMMDD
                    return date_part[2:]  # 取后6位
            return date_str
        except Exception:
            return str(date_str).replace('-', '').replace(':', '').replace(' ', '')[:6]

    def _get_factor_group_features(self, factor_group_name: str, available_columns: list,
                                 excluded_columns: list) -> list:
        """
        动态获取因子组的特征定义

        Args:
            factor_group_name: 因子组名称
            available_columns: 数据中可用的列
            excluded_columns: 需要排除的列（基础列、时间列等）

        Returns:
            该因子组应该保留的特征列表
        """
        try:
            # 尝试从因子管理器获取因子组实例
            factor_group_instance = self._get_factor_group_instance(factor_group_name)

            if factor_group_instance is not None:
                # 获取因子组定义的特征名称
                defined_features = factor_group_instance.feature_names
                self.logger.debug(f"从因子组 {factor_group_name} 获取到 {len(defined_features)} 个定义特征")

                # 只保留在数据中实际存在的特征
                existing_features = [col for col in defined_features if col in available_columns]
                self.logger.debug(f"因子组 {factor_group_name} 中实际存在 {len(existing_features)} 个特征")

                return existing_features
            else:
                # 如果无法获取因子组实例，使用备用策略：保留所有非排除的列
                self.logger.warning(f"无法获取因子组 {factor_group_name} 的实例，使用备用策略")
                return [col for col in available_columns if col not in excluded_columns]

        except Exception as e:
            self.logger.error(f"获取因子组 {factor_group_name} 特征定义失败: {e}")
            # 出错时使用备用策略
            return [col for col in available_columns if col not in excluded_columns]

    def _get_factor_group_instance(self, factor_group_name: str):
        """
        获取因子组实例

        Args:
            factor_group_name: 因子组名称

        Returns:
            因子组实例或None
        """
        try:
            # 导入因子管理器
            from .factor_manager import FactorManager

            # 创建临时的因子管理器实例来获取因子组
            temp_manager = FactorManager()

            if factor_group_name in temp_manager.factor_groups:
                return temp_manager.factor_groups[factor_group_name]
            else:
                self.logger.warning(f"因子组 {factor_group_name} 未在因子管理器中注册")
                return None

        except Exception as e:
            self.logger.error(f"创建因子组实例失败: {e}")
            return None

    def _prepare_data_for_parquet(self, data: pd.DataFrame, factor_group_name: str = None) -> pd.DataFrame:
        """准备数据以确保parquet格式兼容性，并优化存储"""
        try:
            data_clean = data.copy()

            # 基础必需列（包括时间维度）
            essential_columns = ['stock_code', 'date']

            # 必要的时间列（用于时间序列分析和兼容性）
            necessary_time_columns = ['date_str']

            # 冗余的时间列（可以删除以节省空间）
            redundant_time_columns = [
                'first_time', 'last_time', 'first_time_formatted', 'last_time_formatted',
                'time_to_limit_minutes', 'time_efficiency_score'
            ]

            # 动态获取因子组的特征定义（替代硬编码的字典）
            factor_columns = self._get_factor_group_features(factor_group_name, data_clean.columns,
                                                           essential_columns + necessary_time_columns + redundant_time_columns)

            # 构建最终要保留的列
            columns_to_keep = essential_columns.copy()

            # 添加必要的时间列（如果存在）
            for col in necessary_time_columns:
                if col in data_clean.columns:
                    columns_to_keep.append(col)

            # 添加因子列（如果存在）
            for col in factor_columns:
                if col in data_clean.columns:
                    columns_to_keep.append(col)

            # 只保留必要的列
            data_clean = data_clean[columns_to_keep]

            # 优化数据类型
            for col in data_clean.columns:
                if col in essential_columns:
                    # 确保基础列类型正确
                    if col == 'stock_code':
                        data_clean[col] = data_clean[col].astype(str)
                    elif col == 'date':
                        # 保持datetime类型
                        if not pd.api.types.is_datetime64_any_dtype(data_clean[col]):
                            data_clean[col] = pd.to_datetime(data_clean[col])
                elif col in necessary_time_columns:
                    # 时间字符串列
                    data_clean[col] = data_clean[col].astype(str)
                elif data_clean[col].dtype == 'float64':
                    # 将float64优化为float32
                    data_clean[col] = data_clean[col].astype('float32')

            # 确保date_str列存在（如果原数据没有，从date列生成）
            if 'date_str' not in data_clean.columns and 'date' in data_clean.columns:
                data_clean['date_str'] = data_clean['date'].dt.strftime('%Y%m%d')

            self.logger.debug(f"数据优化: {data.shape} -> {data_clean.shape}, 节省 {(1-data_clean.memory_usage(deep=True).sum()/data.memory_usage(deep=True).sum()):.1%} 内存")
            self.logger.debug(f"保留列: {list(data_clean.columns)}")

            return data_clean

        except Exception as e:
            self.logger.error(f"准备parquet数据失败: {e}")
            return data



    
    def update_group_cache(self, group_name: str, version_hash: str, new_data: pd.DataFrame,
                          merge_with_existing: bool = True):
        """
        更新因子组缓存
        
        Args:
            group_name: 因子组名称
            version_hash: 版本哈希
            new_data: 新数据
            merge_with_existing: 是否与现有数据合并
        """
        try:
            if new_data.empty:
                return
            
            cache_file = self._get_cache_file_path(group_name, version_hash)
            
            # 如果需要合并且缓存文件存在
            if merge_with_existing and cache_file.exists():
                try:
                    # 读取现有缓存
                    existing_data = pd.read_parquet(cache_file)
                    
                    if not existing_data.empty:
                        # 合并数据，新数据优先
                        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                        
                        # 去重，保留最新的数据
                        if 'stock_code' in combined_data.columns and 'date' in combined_data.columns:
                            combined_data = combined_data.drop_duplicates(
                                subset=['stock_code', 'date'], keep='last'
                            )
                        
                        final_data = combined_data
                    else:
                        final_data = new_data
                        
                except Exception as e:
                    self.logger.warning(f"合并缓存数据失败: {e}，使用新数据")
                    final_data = new_data
            else:
                final_data = new_data
            
            # 保存缓存数据
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # 处理数据以确保parquet兼容性
            final_data_clean = self._prepare_data_for_parquet(final_data, group_name)
            final_data_clean.to_parquet(cache_file, engine='pyarrow', compression='brotli')
            
            # 更新缓存索引
            self.cache_index[group_name] = {
                'version_hash': version_hash,
                'last_update': datetime.now().isoformat(),
                'record_count': len(final_data),
                'file_path': str(cache_file),
                'features': list(final_data.columns) if not final_data.empty else []
            }
            
            self._save_cache_index()
            
            self.logger.debug(f"因子组 {group_name} 缓存已更新: {len(final_data)} 条记录")
            
        except Exception as e:
            self.logger.error(f"更新因子组 {group_name} 缓存失败: {e}")
    
    def invalidate_group_cache(self, group_name: str):
        """使因子组缓存失效"""
        try:
            if group_name in self.cache_index:
                # 删除缓存文件
                cache_info = self.cache_index[group_name]
                cache_file = Path(cache_info.get('file_path', ''))
                
                if cache_file.exists():
                    cache_file.unlink()
                
                # 从索引中移除
                del self.cache_index[group_name]
                self._save_cache_index()
                
                self.logger.info(f"因子组 {group_name} 缓存已失效")
            
        except Exception as e:
            self.logger.error(f"使因子组 {group_name} 缓存失效失败: {e}")
    
    def clear_all_cache(self):
        """清空所有缓存"""
        try:
            # 删除所有缓存文件
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            
            # 清空索引
            self.cache_index = {}
            self._save_cache_index()
            
            self.logger.info("所有因子缓存已清空")
            
        except Exception as e:
            self.logger.error(f"清空缓存失败: {e}")
    
    def get_cache_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有因子组的缓存状态"""
        status = {}
        
        for group_name, cache_info in self.cache_index.items():
            cache_file = Path(cache_info.get('file_path', ''))
            
            status[group_name] = {
                'version_hash': cache_info.get('version_hash', ''),
                'last_update': cache_info.get('last_update', ''),
                'record_count': cache_info.get('record_count', 0),
                'file_exists': cache_file.exists(),
                'file_size_mb': cache_file.stat().st_size / 1024 / 1024 if cache_file.exists() else 0,
                'features': cache_info.get('features', [])
            }
        
        return status
    
    def cleanup_expired_cache(self, max_age_days: int = 30):
        """清理过期缓存"""
        try:
            current_time = datetime.now()
            expired_groups = []

            for group_name, cache_info in self.cache_index.items():
                last_update = cache_info.get('last_update') or cache_info.get('timestamp')
                if last_update:
                    last_update_time = datetime.fromisoformat(last_update)
                    if current_time - last_update_time > timedelta(days=max_age_days):
                        expired_groups.append(group_name)

            for group_name in expired_groups:
                self.invalidate_group_cache(group_name)

            if expired_groups:
                self.logger.info(f"清理了 {len(expired_groups)} 个过期缓存: {expired_groups}")

        except Exception as e:
            self.logger.error(f"清理过期缓存失败: {e}")

    def cleanup_duplicate_hash_caches(self):
        """清理重复的hash格式缓存文件，保留最新的"""
        try:
            self.logger.info("🧹 开始清理重复的hash格式缓存文件...")

            # 扫描所有缓存文件
            cache_files = list(self.cache_dir.glob("*.parquet"))

            # 按因子组分组
            group_files = {}
            for cache_file in cache_files:
                filename = cache_file.name

                # 识别hash格式的文件（包含16位hex字符）
                if '_' in filename:
                    parts = filename.replace('.parquet', '').split('_')
                    if len(parts) >= 2:
                        # 检查最后一部分是否是hash（16位hex）
                        potential_hash = parts[-1]
                        if len(potential_hash) == 16 and all(c in '0123456789abcdef' for c in potential_hash.lower()):
                            group_name = '_'.join(parts[:-1])
                            if group_name not in group_files:
                                group_files[group_name] = []
                            group_files[group_name].append(cache_file)

            # 清理重复文件
            cleaned_count = 0
            for group_name, files in group_files.items():
                if len(files) > 1:
                    # 按修改时间排序，保留最新的
                    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    latest_file = files[0]

                    # 删除其他文件
                    for old_file in files[1:]:
                        try:
                            old_file.unlink()
                            cleaned_count += 1
                            self.logger.debug(f"删除重复缓存: {old_file.name}")
                        except Exception as e:
                            self.logger.warning(f"删除文件失败 {old_file.name}: {e}")

                    self.logger.info(f"因子组 {group_name}: 保留最新缓存 {latest_file.name}，删除 {len(files)-1} 个重复文件")

            if cleaned_count > 0:
                self.logger.info(f"✅ 清理完成，删除了 {cleaned_count} 个重复的hash格式缓存文件")
            else:
                self.logger.info("✅ 没有发现重复的hash格式缓存文件")

        except Exception as e:
            self.logger.error(f"清理重复缓存失败: {e}")

    def batch_load_cache(self, group_names: List[str], data_hash: str) -> Dict[str, Optional[pd.DataFrame]]:
        """并行批量加载缓存"""
        try:
            results = {}

            # 先检查内存缓存
            memory_hits = {}
            remaining_groups = []

            with self._lock:
                for group_name in group_names:
                    cache_key = f"{group_name}_{data_hash}"
                    if cache_key in self._memory_cache:
                        memory_hits[group_name] = self._memory_cache[cache_key]
                        self.logger.debug(f"内存缓存命中: {group_name}")
                    else:
                        remaining_groups.append(group_name)

            results.update(memory_hits)

            if not remaining_groups:
                return results

            # 并行加载磁盘缓存
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_group = {
                    executor.submit(self.load_cache, group_name, data_hash): group_name
                    for group_name in remaining_groups
                }

                for future in as_completed(future_to_group):
                    group_name = future_to_group[future]
                    try:
                        cached_data = future.result()
                        results[group_name] = cached_data

                        # 添加到内存缓存
                        if cached_data is not None:
                            self._add_to_memory_cache(f"{group_name}_{data_hash}", cached_data)

                    except Exception as e:
                        self.logger.error(f"并行加载缓存失败 {group_name}: {e}")
                        results[group_name] = None

            return results

        except Exception as e:
            self.logger.error(f"批量加载缓存失败: {e}")
            return {group_name: None for group_name in group_names}

    def batch_save_cache(self, cache_data: Dict[str, Tuple[pd.DataFrame, str, str]]) -> Dict[str, bool]:
        """并行批量保存缓存

        Args:
            cache_data: {group_name: (data, data_hash, version)}
        """
        try:
            results = {}

            # 并行保存缓存
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_group = {
                    executor.submit(self.save_cache, group_name, data, data_hash, version): group_name
                    for group_name, (data, data_hash, version) in cache_data.items()
                }

                for future in as_completed(future_to_group):
                    group_name = future_to_group[future]
                    try:
                        success = future.result()
                        results[group_name] = success

                        # 添加到内存缓存
                        if success:
                            data, data_hash, version = cache_data[group_name]
                            self._add_to_memory_cache(f"{group_name}_{data_hash}", data)

                    except Exception as e:
                        self.logger.error(f"并行保存缓存失败 {group_name}: {e}")
                        results[group_name] = False

            return results

        except Exception as e:
            self.logger.error(f"批量保存缓存失败: {e}")
            return {group_name: False for group_name in cache_data.keys()}

    def _add_to_memory_cache(self, cache_key: str, data: pd.DataFrame):
        """添加数据到内存缓存"""
        try:
            with self._lock:
                # 估算数据大小
                data_size = data.memory_usage(deep=True).sum()

                # 如果数据太大，不缓存
                if data_size > self._max_memory_cache_size // 4:
                    return

                # 如果缓存满了，清理最旧的数据
                while (self._memory_cache_size + data_size > self._max_memory_cache_size and
                       self._memory_cache):
                    # 简单的FIFO策略
                    oldest_key = next(iter(self._memory_cache))
                    old_data = self._memory_cache.pop(oldest_key)
                    self._memory_cache_size -= old_data.memory_usage(deep=True).sum()

                # 添加新数据
                self._memory_cache[cache_key] = data.copy()
                self._memory_cache_size += data_size

                self.logger.debug(f"添加到内存缓存: {cache_key}, 当前缓存大小: {self._memory_cache_size / 1024 / 1024:.1f}MB")

        except Exception as e:
            self.logger.error(f"添加内存缓存失败: {e}")

    def clear_memory_cache(self):
        """清空内存缓存"""
        with self._lock:
            self._memory_cache.clear()
            self._memory_cache_size = 0
            self.logger.info("内存缓存已清空")

    def get_memory_cache_stats(self) -> Dict[str, Any]:
        """获取内存缓存统计信息"""
        with self._lock:
            return {
                'cache_count': len(self._memory_cache),
                'cache_size_mb': self._memory_cache_size / 1024 / 1024,
                'max_size_mb': self._max_memory_cache_size / 1024 / 1024,
                'usage_percent': (self._memory_cache_size / self._max_memory_cache_size) * 100
            }

    def load_cache(self, group_name: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载单个因子组的缓存（基于日期范围）"""
        try:
            cache_key = f"{group_name}_{start_date}_{end_date}"

            # 先检查内存缓存
            with self._lock:
                if cache_key in self._memory_cache:
                    self.logger.debug(f"内存缓存命中: {group_name}")
                    return self._memory_cache[cache_key].copy()

            # 检查磁盘缓存
            cache_file = self._get_cache_file_path(group_name, start_date, end_date)

            if cache_file.exists():
                # 检查缓存是否过期
                if self._is_cache_valid(group_name, start_date, end_date):
                    try:
                        cached_data = pd.read_parquet(cache_file)

                        # 添加到内存缓存
                        self._add_to_memory_cache(cache_key, cached_data)

                        self.logger.debug(f"磁盘缓存命中: {group_name} ({start_date}-{end_date})")
                        return cached_data
                    except Exception as e:
                        self.logger.error(f"读取缓存文件失败: {e}")
                        cache_file.unlink()
                else:
                    # 删除过期缓存
                    cache_file.unlink()
                    self.logger.debug(f"删除过期缓存: {group_name}")

            return None

        except Exception as e:
            self.logger.error(f"加载缓存失败 {group_name}: {e}")
            return None

    def save_cache(self, group_name: str, data: pd.DataFrame, start_date: str, end_date: str, version: str = "1.0") -> bool:
        """保存单个因子组的缓存（基于日期范围）"""
        try:
            cache_key = f"{group_name}_{start_date}_{end_date}"
            cache_file = self._get_cache_file_path(group_name, start_date, end_date)

            # 🚀 修复：添加详细的保存过程日志
            self.logger.info(f"🔧 开始保存因子组 {group_name} 到缓存...")
            self.logger.info(f"  缓存文件路径: {cache_file}")
            self.logger.info(f"  数据形状: {data.shape}")

            if data.empty:
                self.logger.error(f"❌ 数据为空，无法保存 {group_name}")
                return False

            # 保存到磁盘
            self.logger.info(f"  准备parquet数据...")
            data_clean = self._prepare_data_for_parquet(data, group_name)
            self.logger.info(f"  清理后数据形状: {data_clean.shape}")

            self.logger.info(f"  写入parquet文件...")
            data_clean.to_parquet(cache_file, engine='pyarrow', compression='brotli')

            # 验证文件是否成功创建
            if cache_file.exists():
                file_size = cache_file.stat().st_size / (1024*1024)  # MB
                self.logger.info(f"  ✅ 文件保存成功: {cache_file.name}, 大小: {file_size:.2f}MB")
            else:
                self.logger.error(f"  ❌ 文件保存失败: {cache_file}")
                return False

            # 更新缓存索引
            with self._lock:
                self.cache_index[group_name] = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'version': version,
                    'timestamp': datetime.now().isoformat(),
                    'file_path': str(cache_file),
                    'record_count': len(data)
                }
                self._save_cache_index()

            # 添加到内存缓存
            self._add_to_memory_cache(cache_key, data)

            self.logger.debug(f"缓存保存成功: {group_name} ({start_date}-{end_date})")
            return True

        except Exception as e:
            self.logger.error(f"保存缓存失败 {group_name}: {e}")
            return False

    def _is_cache_file_valid(self, cache_file: Path) -> bool:
        """检查缓存文件是否有效"""
        try:
            # 检查文件是否存在且不为空
            if not cache_file.exists() or cache_file.stat().st_size == 0:
                return False

            # 检查文件修改时间（24小时内有效）
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            current_time = datetime.now()
            time_diff = current_time - file_time

            return time_diff.total_seconds() < 24 * 3600  # 24小时

        except Exception:
            return False
