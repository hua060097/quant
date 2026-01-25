#!/usr/bin/env python3
"""
因子管理器

统一管理所有因子组，提供因子生成、缓存管理和依赖解析功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import gc
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .factor_config import FactorConfig
from .factor_cache import FactorCacheManager
from .factor_validator import FactorValidator
from .factor_groups import (
    # 重新启用已修复的因子组
    TechnicalFactors,  # 重新启用，已优化性能
    FundamentalFactors,
    KlinePatternFactors,
    MicrostructureFactors,
    RiskFactors,
    MomentumFactors,
    VolumeFactors,
    CrossSectionalFactors,
    # CorrelationFactors,  # 已删除，因子迁移到EnhancedCorrelationFactors
    LimitUpQualityFactors,  # 重构为直接加载模式（v2.0）
    EnhancedCorrelationFactors,
    ComprehensiveMAFactors,
    PricePositionFactors,
    AdvancedTechnicalFactors,
    # LimitUpQualityFactors, EnhancedCorrelationFactors, ComprehensiveMAFactors,
    # PricePositionFactors,

    # 只启用已修复的因子组
    SectorRelativeFactors,

    # 新增：顶级支撑因子组
    EliteSupportFactors,

    # 新增：市场相关性因子组
    MarketCorrFactors,

    # 新增：市场相对因子组
    MarketRelativeFactors
)

class FactorManager:
    """因子管理器"""
    
    def __init__(self, config_file: Optional[str] = None, cache_dir: str = "./factor_cache"):
        self.logger = logging.getLogger(__name__)
        
        # 初始化配置和缓存管理器
        self.config = FactorConfig(config_file)
        self.cache_manager = FactorCacheManager(cache_dir)
        self.validator = FactorValidator()
        
        # 初始化因子组
        self.factor_groups = self._initialize_factor_groups()
        
        # 性能统计
        self.performance_stats = {}

        # 内存优化配置
        self.enable_memory_optimization = self.config.get_global_setting('enable_memory_optimization', True)
        self.enable_batch_save = self.config.get_global_setting('enable_batch_save', True)
        self.memory_monitor_interval = self.config.get_global_setting('memory_monitor_interval', 10)  # 每10个因子组监控一次
        self.enable_chunked_processing = self.config.get_global_setting('enable_chunked_processing', True)  # 启用分块处理
        self.chunk_size = self.config.get_global_setting('chunk_size', 50000)  # 分块大小

        self.logger.info(f"因子管理器初始化完成，共加载 {len(self.factor_groups)} 个因子组")
        self.logger.info(f"内存优化: {'启用' if self.enable_memory_optimization else '禁用'}")
        self.logger.info(f"分批保存: {'启用' if self.enable_batch_save else '禁用'}")
    
    def _initialize_factor_groups(self) -> Dict[str, Any]:
        """初始化因子组"""
        factor_groups = {}
        
        # 因子组类映射（临时只启用已修复的因子组）
        factor_group_classes = {
            # 重新启用已修复的因子组
            'technical_factors': TechnicalFactors,  # 重新启用，已优化性能
            'fundamental_factors': FundamentalFactors,
            'kline_pattern_factors': KlinePatternFactors,
            'microstructure_factors': MicrostructureFactors,
            'risk_factors': RiskFactors,
            'momentum_factors': MomentumFactors,
            'volume_factors': VolumeFactors,
            'cross_sectional_factors': CrossSectionalFactors,
            # 'correlation_factors': CorrelationFactors,  # 已删除，因子迁移到EnhancedCorrelationFactors
            'limit_up_quality_factors': LimitUpQualityFactors,  # 重构为直接加载模式（v2.0）
            'enhanced_correlation_factors': EnhancedCorrelationFactors,
            'comprehensive_ma_factors': ComprehensiveMAFactors,
            'price_position_factors': PricePositionFactors,
            'advanced_technical_factors': AdvancedTechnicalFactors,

            # 只启用已修复的因子组
            'sector_relative_factors': SectorRelativeFactors,

            # 新增：顶级支撑因子组
            'elite_support_factors': EliteSupportFactors,

            # 新增：市场相关性因子组
            'market_corr_factors': MarketCorrFactors,

            # 新增：市场相对因子组
            'market_relative_factors': MarketRelativeFactors
        }
        
        # 只初始化启用的因子组
        enabled_groups = self.config.get_enabled_factor_groups()
        
        for group_name, group_config in enabled_groups.items():
            if group_name in factor_group_classes:
                try:
                    factor_group = factor_group_classes[group_name]()
                    factor_groups[group_name] = factor_group
                    self.logger.debug(f"初始化因子组: {group_name}")
                except Exception as e:
                    self.logger.error(f"初始化因子组 {group_name} 失败: {e}")
            else:
                self.logger.warning(f"未找到因子组类: {group_name}")
        
        return factor_groups
    
    def get_available_factor_groups(self) -> Dict[str, Dict[str, Any]]:
        """获取可用的因子组信息"""
        available_groups = {}
        
        for group_name, factor_group in self.factor_groups.items():
            available_groups[group_name] = factor_group.get_summary()
        
        return available_groups
    
    def get_factor_group_features(self, group_name: str) -> List[str]:
        """获取指定因子组的特征列表"""
        if group_name in self.factor_groups:
            return self.factor_groups[group_name].feature_names
        else:
            return []
    
    def get_all_features(self) -> List[str]:
        """获取所有因子组的特征列表"""
        all_features = []
        for factor_group in self.factor_groups.values():
            all_features.extend(factor_group.feature_names)
        return list(set(all_features))  # 去重
    
    def calculate_factor_group(self, group_name: str, data: pd.DataFrame,
                             use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """
        计算指定因子组
        
        Args:
            group_name: 因子组名称
            data: 输入数据
            use_cache: 是否使用缓存
            
        Returns:
            包含新特征的数据
        """
        try:
            print(f"📍 进入calculate_factor_group: {group_name}")
            if group_name not in self.factor_groups:
                self.logger.error(f"因子组 {group_name} 不存在")
                return data

            factor_group = self.factor_groups[group_name]
            start_time = time.time()

            # 获取数据日期范围和交易日列表
            if 'date' in data.columns:
                # 🔧 修复：优先使用用户指定的目标日期范围作为文件命名，而不是数据的实际日期范围
                target_start_date = kwargs.get('target_start_date')
                target_end_date = kwargs.get('target_end_date')

                if target_start_date and target_end_date:
                    # 使用用户指定的目标日期范围作为文件命名
                    start_date = target_start_date
                    end_date = target_end_date
                    self.logger.info(f"✅ 使用用户指定的目标日期范围作为文件命名: {start_date} - {end_date}")
                    print(f"📍 文件命名使用目标日期范围: {start_date} - {end_date}")
                else:
                    # 回退到数据的实际日期范围
                    start_date = data['date'].min().strftime('%Y%m%d')
                    end_date = data['date'].max().strftime('%Y%m%d')
                    self.logger.warning(f"⚠️ 未找到目标日期范围，使用数据实际日期范围: {start_date} - {end_date}")
                    print(f"⚠️ 文件命名使用数据实际日期范围: {start_date} - {end_date}")

                # 获取实际数据的日期范围用于日志记录
                actual_start_date = data['date'].min().strftime('%Y%m%d')
                actual_end_date = data['date'].max().strftime('%Y%m%d')
                trading_dates = sorted(data['date'].dt.strftime('%Y-%m-%d').unique().tolist())

                self.logger.info(f"开始计算因子组 {group_name}: 数据实际范围 {actual_start_date} - {actual_end_date} ({len(trading_dates)} 个交易日)")
                self.logger.info(f"文件命名使用范围: {start_date} - {end_date}")
                print(f"🔧 开始计算因子组 {group_name}: {len(data)} 条记录, {len(trading_dates)} 个交易日")
                print(f"📊 数据实际范围: {actual_start_date} - {actual_end_date}")
                print(f"📁 文件命名范围: {start_date} - {end_date}")
            else:
                start_date = kwargs.get('target_start_date', '20230601')
                end_date = kwargs.get('target_end_date', '20250110')
                trading_dates = []
                self.logger.info(f"开始计算因子组 {group_name}: 数据量 {len(data)} 条记录")
                print(f"🔧 开始计算因子组 {group_name}: {len(data)} 条记录")
                print(f"⚠️ 数据中没有date列，使用默认日期范围: {start_date}-{end_date}")

            # 检查是否强制重新计算
            force_recalc = getattr(self, '_force_recalc', False)

            if use_cache and self.config.get_global_setting('enable_caching', True) and trading_dates:
                print(f"📍 缓存检查（基于日期范围: {start_date}-{end_date}）...")

                # 检查基于日期范围的缓存
                cached_data = self.cache_manager.load_cache(group_name, start_date, end_date)

                if cached_data is not None and not cached_data.empty:
                    missing_dates = []  # 完全缓存命中
                else:
                    missing_dates = trading_dates  # 缓存未命中

                if not missing_dates:
                    # 完全缓存命中
                    print(f"✅ 完全缓存命中，直接使用缓存数据")
                    result_data = cached_data
                else:
                    # 缓存未命中，全量计算
                    self.logger.info(f"缓存未命中，开始全量计算因子组 {group_name}")
                    print(f"🔧 缓存未命中，开始全量计算因子组 {group_name}...")

                    calc_start_time = time.time()
                    factor_group = self.factor_groups[group_name]
                    result_data = factor_group.calculate_with_validation(data, **kwargs)
                    calc_elapsed = time.time() - calc_start_time

                    self.logger.info(f"因子组 {group_name} 计算完成，耗时 {calc_elapsed:.2f}s")
                    print(f"✅ 因子组 {group_name} 计算完成，耗时 {calc_elapsed:.2f}s")

                    # 🚀 修复：保存前验证数据质量
                    self._validate_factor_data_before_save(group_name, result_data)

                    # 保存到缓存（基于日期范围）
                    print(f"💾 保存计算结果到缓存...")
                    cache_success = self.cache_manager.save_cache(
                        group_name, result_data, start_date, end_date, factor_group.version
                    )
                    if cache_success:
                        self.logger.debug(f"因子组 {group_name} 计算结果已保存到缓存")
                        print(f"✅ 缓存保存成功: {group_name}_{start_date}_{end_date}.parquet")
            else:
                # 不使用缓存，直接计算
                cache_reason = "缓存已禁用" if not use_cache else "强制重新计算" if force_recalc else "无交易日数据"
                self.logger.info(f"直接计算因子组 {group_name}（{cache_reason}）")
                print(f"🔧 直接计算因子组 {group_name}（{cache_reason}）...")

                calc_start_time = time.time()
                result_data = factor_group.calculate_with_validation(data, **kwargs)
                calc_elapsed = time.time() - calc_start_time

                self.logger.info(f"因子组 {group_name} 计算完成，耗时 {calc_elapsed:.2f}s")
                print(f"✅ 因子组 {group_name} 计算完成，耗时 {calc_elapsed:.2f}s")
            
            # 记录性能统计
            elapsed_time = time.time() - start_time
            self.performance_stats[group_name] = {
                'last_calculation_time': elapsed_time,
                'records_processed': len(data),
                'features_generated': len(factor_group.feature_names)
            }
            
            self.logger.debug(f"因子组 {group_name} 计算完成，耗时 {elapsed_time:.2f}s")
            return result_data
            
        except Exception as e:
            self.logger.error(f"计算因子组 {group_name} 失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return data
    
    def calculate_selected_groups(self, group_names: List[str], data: pd.DataFrame,
                                sector_data: pd.DataFrame = None, market_data: pd.DataFrame = None,
                                use_cache: bool = True, parallel: bool = True, **kwargs) -> pd.DataFrame:
        """
        计算选定的因子组

        Args:
            group_names: 因子组名称列表
            data: 股票数据
            sector_data: 板块数据
            market_data: 市场数据
            use_cache: 是否使用缓存
            parallel: 是否并行计算
            **kwargs: 额外的上下文数据（如stock_sector_mapping等）

        Returns:
            包含所有新特征的数据
        """
        try:
            # 将sector_data和market_data添加到kwargs中
            if sector_data is not None:
                kwargs['sector_data'] = sector_data
                self.logger.info(f"传入板块数据: {sector_data.shape}")

            if market_data is not None:
                kwargs['market_data'] = market_data
                self.logger.info(f"传入市场数据: {market_data.shape}")

            # 解析依赖关系
            resolved_groups = self.config.resolve_dependencies(group_names)
            self.logger.info(f"计算因子组: {resolved_groups}")
            print(f"🔧 开始计算因子组: {resolved_groups}")

            result_data = data.copy()
            print(f"📊 输入数据形状: {result_data.shape}")

            if parallel and len(resolved_groups) > 1:
                # 并行计算（注意依赖关系）
                result_data = self._calculate_groups_parallel(resolved_groups, result_data, use_cache, **kwargs)
            else:
                # 串行计算
                self.logger.info(f"开始串行计算 {len(resolved_groups)} 个因子组")
                for i, group_name in enumerate(resolved_groups):
                    self.logger.info(f"计算因子组 {i+1}/{len(resolved_groups)}: {group_name}")
                    print(f"🔧 计算因子组 {i+1}/{len(resolved_groups)}: {group_name}")

                    group_start_time = time.time()
                    result_data = self.calculate_factor_group(group_name, result_data, use_cache, **kwargs)
                    group_elapsed = time.time() - group_start_time

                    self.logger.info(f"因子组 {group_name} 计算完成，耗时 {group_elapsed:.2f}秒")
                    print(f"✅ 因子组 {group_name} 计算完成")

            self.logger.info(f"因子组计算完成，共生成 {len(result_data.columns) - len(data.columns)} 个新特征")
            return result_data
            
        except Exception as e:
            self.logger.error(f"计算选定因子组失败: {e}")
            return data
    
    def calculate_all_factors(self, data: pd.DataFrame, sector_data: pd.DataFrame = None,
                            market_data: pd.DataFrame = None, use_cache: bool = True,
                            parallel: bool = True, **kwargs) -> pd.DataFrame:
        """
        计算所有启用的因子组（内存优化版本）

        Args:
            data: 股票数据
            sector_data: 板块数据
            market_data: 市场数据
            use_cache: 是否使用缓存
            parallel: 是否并行计算

        Returns:
            包含所有特征的数据
        """
        import time

        # 记录初始内存使用
        self._log_memory_usage("开始计算所有因子组前")

        all_group_names = list(self.factor_groups.keys())
        total_groups = len(all_group_names)

        # 记录开始计算所有因子组
        self.logger.info(f"开始计算所有因子组: 共 {total_groups} 个因子组")
        self.logger.info(f"因子组列表: {all_group_names}")
        self.logger.info(f"计算模式: {'并行' if parallel and total_groups > 1 else '串行'}")
        self.logger.info(f"内存优化模式: {'启用' if self.enable_memory_optimization else '禁用'}")
        self.logger.info(f"分批保存模式: {'启用' if self.enable_batch_save else '禁用'}")

        start_time = time.time()

        result_data = self.calculate_selected_groups(all_group_names, data, sector_data, market_data, use_cache, parallel, **kwargs)

        # 记录总体完成信息
        total_elapsed = time.time() - start_time
        original_features = len(data.columns)
        total_features = len(result_data.columns)
        new_features = total_features - original_features

        self.logger.info(f"所有因子组计算完成!")
        self.logger.info(f"总耗时: {total_elapsed:.2f}秒")
        self.logger.info(f"特征统计: 原始{original_features}个 -> 总计{total_features}个 (新增{new_features}个)")
        self.logger.info(f"平均每个因子组耗时: {total_elapsed/total_groups:.2f}秒")

        # 记录最终内存使用和优化效果
        self._log_memory_usage("所有因子组计算完成后")

        if self.enable_memory_optimization:
            self.logger.info(f"内存优化策略: 数据引用、分批保存、定期垃圾回收")

        # 最终垃圾回收
        self._force_garbage_collection()

        return result_data
    
    def _calculate_groups_parallel(self, group_names: List[str], data: pd.DataFrame,
                                 use_cache: bool, **kwargs) -> pd.DataFrame:
        """并行计算因子组（考虑依赖关系）- 内存优化版本"""
        try:
            # 记录初始内存使用
            self._log_memory_usage("并行计算开始前")

            # 内存优化：不立即复制完整数据，而是使用引用
            if self.enable_memory_optimization:
                result_data = data  # 使用引用而不是复制
                self.logger.info("内存优化: 使用数据引用而非复制")
            else:
                result_data = data.copy()

            max_workers = self.config.get_global_setting('max_parallel_groups', 4)

            # 获取日期范围用于缓存
            start_date = kwargs.get('target_start_date', '20230601')
            end_date = kwargs.get('target_end_date', '20250110')

            # 批量检查缓存
            if use_cache:
                cache_hits = {}
                cache_misses = []

                for group_name in group_names:
                    cached_data = self.cache_manager.load_cache(group_name, start_date, end_date)
                    if cached_data is not None and not cached_data.empty:
                        cache_hits[group_name] = cached_data
                    else:
                        cache_misses.append(group_name)

                # 合并缓存命中的结果
                for group_name, cached_data in cache_hits.items():
                    factor_group = self.factor_groups[group_name]
                    for feature in factor_group.feature_names:
                        if feature in cached_data.columns:
                            result_data[feature] = cached_data[feature]

                self.logger.info(f"缓存命中: {len(cache_hits)}/{len(group_names)} 个因子组")

                # 如果全部命中缓存，直接返回
                if not cache_misses:
                    return result_data

                # 只计算缓存未命中的因子组
                group_names = cache_misses

            # 按依赖层级分组
            dependency_levels = self._group_by_dependency_level(group_names)

            self.logger.info(f"开始并行计算 {len(group_names)} 个因子组，分为 {len(dependency_levels)} 个依赖层级")
            for level_idx, level_groups in enumerate(dependency_levels):
                self.logger.info(f"依赖层级 {level_idx + 1}: {level_groups}")

            # 用于批量保存缓存的数据
            cache_save_data = {}
            total_processed = 0

            for level_idx, level_groups in enumerate(dependency_levels):
                self.logger.info(f"处理依赖层级 {level_idx + 1}/{len(dependency_levels)}: {level_groups}")

                if len(level_groups) == 1:
                    # 单个因子组，直接计算
                    group_name = level_groups[0]
                    total_processed += 1

                    self.logger.info(f"计算因子组 {total_processed}/{len(group_names)}: {group_name} (串行)")

                    start_time = time.time()
                    group_result = self._calculate_single_group_optimized(group_name, result_data, **kwargs)
                    calc_time = time.time() - start_time

                    # 更新结果
                    factor_group = self.factor_groups[group_name]
                    for feature in factor_group.feature_names:
                        if feature in group_result.columns:
                            result_data[feature] = group_result[feature]

                    # 准备缓存数据
                    if use_cache:
                        cache_save_data[group_name] = (
                            group_result, start_date, end_date, factor_group.version
                        )

                    # 记录性能统计
                    self.performance_stats[group_name] = {
                        'calculation_time': calc_time,
                        'data_size': len(result_data),
                        'feature_count': len(factor_group.feature_names)
                    }

                    self.logger.info(f"因子组 {group_name} 计算完成，耗时 {calc_time:.2f}秒")

                else:
                    # 多个因子组，并行计算
                    self.logger.info(f"并行计算 {len(level_groups)} 个因子组: {level_groups}")

                    with ThreadPoolExecutor(max_workers=min(max_workers, len(level_groups))) as executor:
                        # 提交任务（修复：传递kwargs参数）
                        future_to_group = {
                            executor.submit(self._calculate_single_group_optimized, group_name, result_data, **kwargs): group_name
                            for group_name in level_groups
                        }

                        # 内存优化：分批处理结果，避免累积所有结果
                        if self.enable_batch_save and use_cache:
                            # 分批保存模式：每个因子组完成后立即保存和合并
                            completed_count = 0

                            for future in as_completed(future_to_group):
                                group_name = future_to_group[future]
                                completed_count += 1
                                total_processed += 1

                                try:
                                    start_time = time.time()
                                    group_result = future.result()
                                    calc_time = time.time() - start_time

                                    self.logger.info(f"因子组 {group_name} 并行计算完成 ({completed_count}/{len(level_groups)})，耗时 {calc_time:.2f}秒")

                                    # 内存优化：优化数据类型
                                    group_result = self._optimize_data_types(group_result)

                                    # 立即保存到缓存
                                    factor_group = self.factor_groups[group_name]
                                    cache_success = self.cache_manager.save_cache(
                                        group_name, group_result, start_date, end_date, factor_group.version
                                    )

                                    if cache_success:
                                        self.logger.debug(f"因子组 {group_name} 已保存到缓存")

                                    # 立即合并特征到结果数据
                                    features_merged = 0
                                    for feature in factor_group.feature_names:
                                        if feature in group_result.columns:
                                            # 内存优化：确保合并时使用相同的数据类型
                                            if self.enable_memory_optimization and feature not in result_data.columns:
                                                result_data[feature] = group_result[feature].astype('float32', errors='ignore')
                                            else:
                                                result_data[feature] = group_result[feature]
                                            features_merged += 1

                                    self.logger.debug(f"因子组 {group_name} 合并了 {features_merged} 个特征")

                                    # 记录性能统计
                                    self.performance_stats[group_name] = {
                                        'calculation_time': calc_time,
                                        'data_size': len(group_result),
                                        'feature_count': len(factor_group.feature_names)
                                    }

                                    # 内存优化：立即释放group_result
                                    del group_result

                                    # 定期进行垃圾回收
                                    if completed_count % self.memory_monitor_interval == 0:
                                        self._force_garbage_collection()
                                        self._log_memory_usage(f"完成 {completed_count}/{len(level_groups)} 个因子组后")

                                except Exception as e:
                                    self.logger.error(f"并行计算因子组 {group_name} 失败: {e}")

                        else:
                            # 传统模式：收集所有结果后批量处理
                            group_results = {}
                            completed_count = 0

                            for future in as_completed(future_to_group):
                                group_name = future_to_group[future]
                                completed_count += 1
                                total_processed += 1

                                try:
                                    start_time = time.time()
                                    group_result = future.result()
                                    calc_time = time.time() - start_time

                                    group_results[group_name] = group_result

                                    self.logger.info(f"因子组 {group_name} 并行计算完成 ({completed_count}/{len(level_groups)})，耗时 {calc_time:.2f}秒")

                                    # 记录性能统计
                                    factor_group = self.factor_groups[group_name]
                                    self.performance_stats[group_name] = {
                                        'calculation_time': calc_time,
                                        'data_size': len(group_result),
                                        'feature_count': len(factor_group.feature_names)
                                    }

                                except Exception as e:
                                    self.logger.error(f"并行计算因子组 {group_name} 失败: {e}")
                                    group_results[group_name] = result_data

                            # 传统模式：合并结果并准备缓存数据
                            self.logger.info(f"合并 {len(level_groups)} 个并行计算结果")

                            for group_name in level_groups:
                                if group_name in group_results:
                                    group_result = group_results[group_name]
                                    factor_group = self.factor_groups[group_name]

                                    # 合并新特征
                                    features_merged = 0
                                    for feature in factor_group.feature_names:
                                        if feature in group_result.columns:
                                            result_data[feature] = group_result[feature]
                                            features_merged += 1

                                    self.logger.debug(f"因子组 {group_name} 合并了 {features_merged} 个特征")

                                    # 准备缓存数据
                                    if use_cache:
                                        cache_save_data[group_name] = (
                                            group_result, start_date, end_date, factor_group.version
                                        )

                            # 内存优化：清理group_results
                            if self.enable_memory_optimization:
                                del group_results
                                self._force_garbage_collection()

            # 批量保存缓存（仅在传统模式下）
            if use_cache and cache_save_data and not self.enable_batch_save:
                successful_saves = 0
                for group_name, (group_result, start_date, end_date, version) in cache_save_data.items():
                    cache_success = self.cache_manager.save_cache(
                        group_name, group_result, start_date, end_date, version
                    )
                    if cache_success:
                        successful_saves += 1
                self.logger.info(f"批量保存缓存: {successful_saves}/{len(cache_save_data)} 个因子组")

                # 内存优化：清理缓存数据
                if self.enable_memory_optimization:
                    del cache_save_data
                    self._force_garbage_collection()

            # 记录最终内存使用
            self._log_memory_usage("并行计算完成后")

            return result_data

        except Exception as e:
            self.logger.error(f"并行计算因子组失败: {e}")
            # 回退到串行计算（修复：传递kwargs）
            for group_name in group_names:
                result_data = self.calculate_factor_group(group_name, result_data, use_cache, **kwargs)
            return result_data

    def _calculate_single_group_optimized(self, group_name: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """优化的单个因子组计算（修复版：正确传递kwargs）"""
        try:
            if group_name not in self.factor_groups:
                self.logger.warning(f"因子组 {group_name} 不存在")
                return data

            factor_group = self.factor_groups[group_name]

            # 数据验证
            validation_result = self.validator.validate_input_data(data, factor_group.dependencies)
            if not validation_result['is_valid']:
                self.logger.warning(f"因子组 {group_name} 输入数据验证失败: {validation_result['errors']}")
                return data

            # 计算因子（修复：传递kwargs）
            result_data = factor_group.calculate_with_validation(data, **kwargs)

            return result_data

        except Exception as e:
            self.logger.error(f"计算因子组 {group_name} 失败: {e}")
            return data
    
    def _group_by_dependency_level(self, group_names: List[str]) -> List[List[str]]:
        """按依赖层级分组"""
        levels = []
        remaining_groups = set(group_names)
        processed_groups = set()
        
        while remaining_groups:
            current_level = []
            
            for group_name in list(remaining_groups):
                dependencies = self.config.get_factor_group_dependencies(group_name)
                # 检查依赖是否都已处理
                if all(dep in processed_groups or dep not in group_names for dep in dependencies):
                    current_level.append(group_name)
            
            if not current_level:
                # 可能存在循环依赖，强制处理剩余的
                current_level = list(remaining_groups)
                self.logger.warning(f"检测到可能的循环依赖，强制处理: {current_level}")
            
            levels.append(current_level)
            remaining_groups -= set(current_level)
            processed_groups.update(current_level)
        
        return levels

    def _get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存使用 (MB)
                'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存使用 (MB)
                'percent': process.memory_percent()        # 内存使用百分比
            }
        except Exception as e:
            self.logger.warning(f"获取内存使用情况失败: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}

    def _log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        if self.enable_memory_optimization:
            memory_info = self._get_memory_usage()
            self.logger.info(f"内存使用 {context}: 物理内存 {memory_info['rss_mb']:.1f}MB, "
                           f"虚拟内存 {memory_info['vms_mb']:.1f}MB, "
                           f"使用率 {memory_info['percent']:.1f}%")

    def _force_garbage_collection(self):
        """强制垃圾回收"""
        if self.enable_memory_optimization:
            collected = gc.collect()
            self.logger.debug(f"垃圾回收: 清理了 {collected} 个对象")

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """生成数据哈希值"""
        try:
            import hashlib
            # 使用数据形状、列名和部分数据内容生成哈希
            content = f"{data.shape}_{list(data.columns)}_{data.head().to_string()}"
            return hashlib.md5(content.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.warning(f"生成数据哈希失败: {e}")
            return "default_hash"

    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        if not self.enable_memory_optimization:
            return data

        try:
            original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024

            # 优化数值列的数据类型
            for col in data.columns:
                if data[col].dtype == 'float64':
                    # 检查是否可以安全转换为float32
                    if data[col].notna().any():
                        min_val = data[col].min()
                        max_val = data[col].max()
                        # float32的范围大约是-3.4e38到3.4e38
                        if abs(min_val) < 1e30 and abs(max_val) < 1e30:
                            data[col] = data[col].astype('float32')
                elif data[col].dtype == 'int64':
                    # 尝试转换为更小的整数类型
                    if data[col].notna().any():
                        min_val = data[col].min()
                        max_val = data[col].max()
                        if min_val >= -2147483648 and max_val <= 2147483647:
                            data[col] = data[col].astype('int32')

            optimized_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            memory_saved = original_memory - optimized_memory

            if memory_saved > 0:
                self.logger.info(f"数据类型优化: 节省内存 {memory_saved:.1f}MB ({memory_saved/original_memory*100:.1f}%)")

            return data

        except Exception as e:
            self.logger.warning(f"数据类型优化失败: {e}")
            return data

    def _should_use_chunked_processing(self, data: pd.DataFrame) -> bool:
        """判断是否应该使用分块处理"""
        if not self.enable_chunked_processing:
            return False

        # 计算数据大小（MB）
        data_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024

        # 如果数据大于100MB，使用分块处理
        return data_size_mb > 100

    def _process_data_in_chunks(self, data: pd.DataFrame, process_func, *args, **kwargs):
        """分块处理数据"""
        if not self._should_use_chunked_processing(data):
            return process_func(data, *args, **kwargs)

        self.logger.info(f"启用分块处理: 数据大小 {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")

        # 按股票代码分组进行分块处理
        if 'stock_code' in data.columns:
            unique_stocks = data['stock_code'].unique()
            chunk_size = min(self.chunk_size, len(unique_stocks) // 4)  # 至少分4块

            results = []
            for i in range(0, len(unique_stocks), chunk_size):
                chunk_stocks = unique_stocks[i:i + chunk_size]
                chunk_data = data[data['stock_code'].isin(chunk_stocks)]

                self.logger.debug(f"处理分块 {i//chunk_size + 1}: {len(chunk_stocks)} 只股票")

                try:
                    chunk_result = process_func(chunk_data, *args, **kwargs)
                    results.append(chunk_result)

                    # 强制垃圾回收
                    del chunk_data, chunk_result
                    self._force_garbage_collection()

                except Exception as e:
                    self.logger.error(f"分块处理失败: {e}")
                    # 如果分块处理失败，尝试整体处理
                    return process_func(data, *args, **kwargs)

            # 合并结果
            if results:
                final_result = pd.concat(results, ignore_index=True)
                self.logger.info(f"分块处理完成: {len(results)} 个分块")
                return final_result
            else:
                return data
        else:
            # 如果没有stock_code列，按行数分块
            chunk_size = self.chunk_size
            results = []

            for i in range(0, len(data), chunk_size):
                chunk_data = data.iloc[i:i + chunk_size]

                self.logger.debug(f"处理分块 {i//chunk_size + 1}: {len(chunk_data)} 行")

                try:
                    chunk_result = process_func(chunk_data, *args, **kwargs)
                    results.append(chunk_result)

                    # 强制垃圾回收
                    del chunk_data, chunk_result
                    self._force_garbage_collection()

                except Exception as e:
                    self.logger.error(f"分块处理失败: {e}")
                    return process_func(data, *args, **kwargs)

            # 合并结果
            if results:
                final_result = pd.concat(results, ignore_index=True)
                self.logger.info(f"分块处理完成: {len(results)} 个分块")
                return final_result
            else:
                return data

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """
        生成数据哈希值用于缓存

        策略：使用简化但稳定的哈希算法，兼容现有缓存
        """
        try:
            import hashlib

            # 使用简化的稳定特征生成哈希
            hash_components = [
                str(len(data)),                    # 数据行数
                str(data.shape[1]),               # 数据列数
            ]

            # 添加关键的数据范围特征
            if 'date' in data.columns:
                dates = data['date'].dt.strftime('%Y-%m-%d')
                hash_components.extend([
                    str(dates.min()),              # 最早日期
                    str(dates.max()),              # 最晚日期
                ])

            if 'stock_code' in data.columns:
                hash_components.append(str(data['stock_code'].nunique()))  # 股票数量

            # 生成MD5哈希（保持与现有缓存兼容）
            hash_string = '|'.join(hash_components)
            return hashlib.md5(hash_string.encode()).hexdigest()[:16]

        except Exception as e:
            self.logger.error(f"生成数据哈希失败: {e}")
            return "default_hash"

    def _generate_data_keys(self, data: pd.DataFrame) -> List[str]:
        """生成数据键"""
        if 'stock_code' in data.columns and 'date' in data.columns:
            return [f"{row['stock_code']}_{row['date']}" for _, row in data[['stock_code', 'date']].iterrows()]
        else:
            return []

    def _filter_data_by_keys(self, data: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        """根据键过滤数据"""
        if not keys or 'stock_code' not in data.columns or 'date' not in data.columns:
            return pd.DataFrame()

        # 解析键
        key_pairs = []
        for key in keys:
            parts = key.split('_', 1)
            if len(parts) == 2:
                key_pairs.append((parts[0], parts[1]))

        if not key_pairs:
            return pd.DataFrame()

        # 创建过滤条件
        filter_conditions = []
        for stock_code, date in key_pairs:
            try:
                # 安全地转换日期类型
                if 'date' in data.columns:
                    if data['date'].dtype == 'object':
                        condition = (data['stock_code'] == stock_code) & (data['date'] == date)
                    else:
                        condition = (data['stock_code'] == stock_code) & (data['date'].astype(str) == str(date))
                else:
                    self.logger.warning("数据中缺少date列")
                    continue
                filter_conditions.append(condition)
            except Exception as e:
                self.logger.error(f"创建过滤条件失败 {stock_code}-{date}: {e}")
                continue

        # 合并条件
        if filter_conditions:
            combined_condition = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_condition |= condition

            return data[combined_condition].copy()
        else:
            return pd.DataFrame()

    def _merge_factor_data(self, cached_data: pd.DataFrame, new_data: pd.DataFrame,
                          original_data: pd.DataFrame) -> pd.DataFrame:
        """合并因子数据"""
        try:
            if cached_data.empty:
                return new_data
            if new_data.empty:
                return self._align_cached_data_with_input(cached_data, original_data)

            # 合并缓存数据和新数据
            combined_data = pd.concat([cached_data, new_data], ignore_index=True)

            # 去重，保留最新的数据
            if 'stock_code' in combined_data.columns and 'date' in combined_data.columns:
                combined_data = combined_data.drop_duplicates(
                    subset=['stock_code', 'date'], keep='last'
                )

            return self._align_cached_data_with_input(combined_data, original_data)

        except Exception as e:
            self.logger.error(f"合并因子数据失败: {e}")
            return new_data

    def _align_cached_data_with_input(self, cached_data: pd.DataFrame,
                                    original_data: pd.DataFrame) -> pd.DataFrame:
        """将缓存数据与输入数据对齐"""
        try:
            if cached_data.empty or original_data.empty:
                return original_data

            # 基于stock_code和date进行合并
            if 'stock_code' in cached_data.columns and 'date' in cached_data.columns:
                # 确保日期格式一致
                cached_data['date'] = pd.to_datetime(cached_data['date']).dt.strftime('%Y%m%d')
                original_data_copy = original_data.copy()
                original_data_copy['date'] = pd.to_datetime(original_data_copy['date']).dt.strftime('%Y%m%d')

                # 合并数据
                result_data = original_data_copy.merge(
                    cached_data,
                    on=['stock_code', 'date'],
                    how='left',
                    suffixes=('', '_cached')
                )

                # 移除重复的列
                duplicate_cols = [col for col in result_data.columns if col.endswith('_cached')]
                result_data = result_data.drop(columns=duplicate_cols)

                return result_data
            else:
                return original_data

        except Exception as e:
            self.logger.error(f"对齐缓存数据失败: {e}")
            return original_data

    def invalidate_factor_group_cache(self, group_name: str):
        """使指定因子组的缓存失效"""
        self.cache_manager.invalidate_group_cache(group_name)
        self.logger.info(f"因子组 {group_name} 缓存已失效")

    def clear_all_cache(self):
        """清空所有缓存"""
        self.cache_manager.clear_all_cache()
        self.logger.info("所有因子缓存已清空")

    def get_cache_status(self) -> Dict[str, Dict[str, Any]]:
        """获取缓存状态"""
        return self.cache_manager.get_cache_status()

    def get_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取性能统计"""
        return self.performance_stats.copy()

    def validate_factor_data(self, data: pd.DataFrame, factor_names: List[str] = None) -> Dict[str, Any]:
        """验证因子数据"""
        if factor_names is None:
            factor_names = self.get_all_features()

        return self.validator.validate_factor_data(data, factor_names)

    def get_factor_metadata(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """获取因子元数据"""
        for group_name, factor_group in self.factor_groups.items():
            if factor_name in factor_group.feature_names:
                metadata = factor_group.get_feature_metadata().get(factor_name, {})
                metadata['group_name'] = group_name
                return metadata
        return None

    def cleanup_expired_cache(self, max_age_days: int = 30):
        """清理过期缓存"""
        self.cache_manager.cleanup_expired_cache(max_age_days)

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        return {
            'total_factor_groups': len(self.factor_groups),
            'total_features': len(self.get_all_features()),
            'enabled_groups': list(self.factor_groups.keys()),
            'cache_status': self.get_cache_status(),
            'performance_stats': self.get_performance_stats(),
            'config_summary': {
                'enable_caching': self.config.get_global_setting('enable_caching'),
                'cache_compression': self.config.get_global_setting('cache_compression'),
                'max_parallel_groups': self.config.get_global_setting('max_parallel_groups'),
                'validation_enabled': self.config.get_global_setting('validation_enabled')
            }
        }

    def _validate_factor_data_before_save(self, group_name: str, data: pd.DataFrame):
        """
        🚀 修复：保存前验证因子数据质量
        """
        try:
            self.logger.info(f"🔍 验证因子组 {group_name} 数据质量...")

            if data.empty:
                self.logger.error(f"❌ {group_name} 数据为空，无法保存")
                return

            self.logger.info(f"  数据形状: {data.shape}")

            # 检查是否包含因子列
            factor_columns = [col for col in data.columns if col not in ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            self.logger.info(f"  因子列数: {len(factor_columns)}")

            if not factor_columns:
                self.logger.error(f"❌ {group_name} 没有因子列")
                return

            # 检查关键因子（针对kline_pattern_factors）
            if group_name == 'kline_pattern_factors':
                key_factors = ['lower_shadow_ratio', 'upper_shadow_ratio', 'body_ratio']
                for factor in key_factors:
                    if factor in data.columns:
                        factor_data = data[factor]
                        unique_count = factor_data.nunique()
                        non_zero_count = (factor_data != 0).sum()

                        self.logger.info(f"  {factor}: 唯一值{unique_count}, 非零值{non_zero_count:,}")

                        if unique_count == 1 and (factor_data == 0).all():
                            self.logger.error(f"  ❌ {factor} 全为0")
                        else:
                            self.logger.info(f"  ✅ {factor} 数据正常")

            self.logger.info(f"✅ {group_name} 数据质量检查完成")

        except Exception as e:
            self.logger.error(f"数据质量检查失败: {e}")
