#!/usr/bin/env python3
"""
板块相对特征因子组

计算个股相对于其所属板块的相对表现特征
核心因子：个股涨幅与板块平均涨幅的相对涨幅
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from .base_factor_group import BaseFactorGroup


class SectorRelativeFactors(BaseFactorGroup):
    """板块相对特征因子组"""
    
    def __init__(self):
        self.name = "sector_relative_factors"
        self.description = "板块相对特征因子：计算个股相对于板块的表现"
        self.logger = logging.getLogger(__name__)

        # 因子列表（在super().__init__()之前定义）
        self.factor_names = [
            # 基础相对收益因子
            'sector_relative_return_1d',    # 1日相对板块涨幅
            'sector_relative_return_5d',    # 5日相对板块涨幅
            'sector_relative_return_20d',   # 20日相对板块涨幅
            'sector_relative_strength',     # 相对强度指标
            'sector_outperform_ratio_20d',  # 20日跑赢板块比例
            'sector_correlation_20d',       # 与板块相关性

            # 板块成交量相对排名因子
            'sector_volume_rank_3d',        # 板块3日平均成交量排名分位数
            'sector_volume_rank_5d',        # 板块5日平均成交量排名分位数
            'sector_volume_rank_10d',       # 板块10日平均成交量排名分位数
            'sector_volume_rank_20d',       # 板块20日平均成交量排名分位数

            # 板块涨幅相对排名因子
            'sector_return_rank_3d',        # 板块3日累计涨幅排名分位数
            'sector_return_rank_5d',        # 板块5日累计涨幅排名分位数
            'sector_return_rank_10d',       # 板块10日累计涨幅排名分位数
            'sector_return_rank_20d',       # 板块20日累计涨幅排名分位数

            # 板块动量强度因子
            'sector_momentum_strength',     # 基于多时间窗口涨幅排名的动量强度得分
            'sector_momentum_consistency',  # 动量一致性指标
            'sector_momentum_acceleration', # 动量加速度指标

            # 板块流动性因子
            'sector_liquidity_score',       # 基于成交量排名的流动性得分
            'sector_liquidity_stability',   # 流动性稳定性指标
            'sector_relative_liquidity',    # 相对流动性指标

            # 新增：紧急优先级因子
            'risk_adjusted_relative_return_5d',   # 5日风险调整相对收益
            'risk_adjusted_relative_return_20d',  # 20日风险调整相对收益
            'risk_adjusted_relative_return_60d',  # 60日风险调整相对收益
            'sector_beta_20d',                    # 20日板块Beta系数
            'sector_beta_60d',                    # 60日板块Beta系数
            'sector_beta_120d',                   # 120日板块Beta系数

            # 从correlation_factors.py迁移的板块相关因子
            'sector_correlation'                  # 板块相关性
        ]

        super().__init__()
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算板块相对特征因子

        Args:
            data: 股票数据，包含 pct_chg, close 等字段
            **kwargs: 额外参数，期望包含：
                - sector_data: 板块数据
                - stock_sector_mapping: 股票-板块映射

        Returns:
            包含板块相对特征的数据
        """
        try:
            self.logger.info("开始计算板块相对特征因子")

            # 清理之前的缓存（如果存在）
            self._clear_calculation_cache()

            result_data = data.copy()
            
            # 获取必要的上下文数据
            sector_data = kwargs.get('sector_data')
            stock_sector_mapping = kwargs.get('stock_sector_mapping')
            
            if sector_data is None or sector_data.empty:
                self.logger.warning("未提供板块数据或板块数据为空，使用默认值")
                return self._set_default_values(result_data)

            if stock_sector_mapping is None:
                self.logger.warning("未提供股票板块映射，使用默认值")
                return self._set_default_values(result_data)

            # 检查板块数据是否包含必要字段
            if 'sector_code' not in sector_data.columns:
                self.logger.error(f"板块数据缺少sector_code字段，现有字段: {list(sector_data.columns)}")
                return self._set_default_values(result_data)

            # 确保数据按股票和日期排序
            result_data = result_data.sort_values(['stock_code', 'date'])
            sector_data = sector_data.sort_values(['sector_code', 'date'])
            
            # 重新设计：正确识别目标计算日期和历史支撑数据
            target_dates, historical_dates = self._identify_target_and_historical_dates(
                result_data, sector_data, kwargs
            )

            if not target_dates:
                self.logger.warning("无法识别目标计算日期，使用默认值")
                return self._set_default_values(result_data)

            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_sector_relative_features_optimized(
                result_data, sector_data, stock_sector_mapping, target_dates, historical_dates
            )

            # 计算新增的紧急优先级因子
            result_data = self._calculate_risk_adjusted_relative_return_features(
                result_data, sector_data, stock_sector_mapping
            )
            result_data = self._calculate_sector_beta_features(
                result_data, sector_data, stock_sector_mapping
            )

            # 计算从correlation_factors.py迁移的板块相关因子
            result_data = self._calculate_migrated_sector_features(
                result_data, sector_data, stock_sector_mapping
            )

            self.logger.info(f"板块相对特征因子计算完成，新增 {len(self.factor_names)} 个特征")

            # 清理计算缓存，释放内存
            self._clear_calculation_cache()

            return result_data

        except Exception as e:
            self.logger.error(f"板块相对特征因子计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")

            # 异常情况下也要清理缓存
            self._clear_calculation_cache()

            return self._set_default_values(data.copy())
    
    def _set_default_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """设置默认值"""
        for factor_name in self.factor_names:
            data[factor_name] = 0.0
        return data

    def _clear_calculation_cache(self):
        """清理计算过程中的缓存，释放内存"""
        cache_items = [
            '_stock_multi_sector_returns_cache',
            '_sector_returns_df_cache',
            '_volume_ranking_cache',
            '_return_ranking_cache',
            '_sector_statistics_cache',
            '_ranking_results_cache'
        ]

        cleared_count = 0
        for cache_item in cache_items:
            if hasattr(self, cache_item):
                delattr(self, cache_item)
                cleared_count += 1

        self.logger.debug(f"计算缓存已清理，清理了 {cleared_count} 个缓存项")

    def _generate_cache_key(self, data: pd.DataFrame,
                           sector_returns_df: pd.DataFrame,
                           stock_sector_mapping: Dict[str, List[str]]) -> str:
        """生成缓存键"""
        try:
            import hashlib

            # 基于关键数据生成哈希
            data_shape = f"{data.shape[0]}_{data.shape[1]}"
            sector_shape = f"{sector_returns_df.shape[0]}_{sector_returns_df.shape[1]}"
            mapping_size = len(stock_sector_mapping)

            # 获取数据的日期范围
            data_dates = f"{data['date'].min()}_{data['date'].max()}"

            # 组合关键信息
            key_components = f"{data_shape}_{sector_shape}_{mapping_size}_{data_dates}"

            # 生成哈希
            cache_key = hashlib.md5(key_components.encode()).hexdigest()
            return cache_key

        except Exception as e:
            # 如果生成缓存键失败，返回一个基于时间的键（确保不会使用缓存）
            import time
            return f"fallback_{time.time()}"

    def _generate_cache_key_for_sector_data(self, sector_data: pd.DataFrame) -> str:
        """生成板块数据的缓存键"""
        try:
            import hashlib

            # 基于板块数据的关键信息生成哈希
            data_shape = f"{sector_data.shape[0]}_{sector_data.shape[1]}"

            # 获取数据的日期范围
            if 'date' in sector_data.columns:
                data_dates = f"{sector_data['date'].min()}_{sector_data['date'].max()}"
            else:
                data_dates = "no_date"

            # 获取板块数量
            if 'sector_code' in sector_data.columns:
                sector_count = sector_data['sector_code'].nunique()
            else:
                sector_count = 0

            # 组合关键信息
            key_components = f"sector_{data_shape}_{data_dates}_{sector_count}"

            # 生成哈希
            cache_key = hashlib.md5(key_components.encode()).hexdigest()
            return cache_key

        except Exception as e:
            # 如果生成缓存键失败，返回一个基于时间的键
            import time
            return f"sector_fallback_{time.time()}"

    def _handle_sector_avg_return_merge(self, data: pd.DataFrame,
                                       stock_sector_returns: pd.DataFrame) -> pd.DataFrame:
        """
        统一处理sector_avg_return字段的merge冲突问题

        Args:
            data: 原始股票数据
            stock_sector_returns: 包含sector_avg_return的板块收益率数据

        Returns:
            合并后的数据，确保包含正确的sector_avg_return字段
        """
        try:
            # 检查是否存在字段冲突
            if 'sector_avg_return' in data.columns:
                self.logger.info("检测到数据中已存在 'sector_avg_return' 字段，先删除以避免merge冲突")
                data = data.drop(columns=['sector_avg_return'])

            # 执行merge操作
            merged_data = data.merge(
                stock_sector_returns,
                on=['stock_code', 'date'],
                how='left'
            )

            # 验证必要字段并处理merge后的字段名
            if 'sector_avg_return' not in merged_data.columns:
                # 检查是否有带后缀的字段
                if 'sector_avg_return_y' in merged_data.columns:
                    self.logger.info("使用 'sector_avg_return_y' 字段作为板块平均收益率")
                    merged_data['sector_avg_return'] = merged_data['sector_avg_return_y']
                    # 清理重复字段
                    if 'sector_avg_return_x' in merged_data.columns:
                        merged_data = merged_data.drop(columns=['sector_avg_return_x'])
                    merged_data = merged_data.drop(columns=['sector_avg_return_y'])
                elif 'sector_avg_return_x' in merged_data.columns:
                    self.logger.info("使用 'sector_avg_return_x' 字段作为板块平均收益率")
                    merged_data['sector_avg_return'] = merged_data['sector_avg_return_x']
                    merged_data = merged_data.drop(columns=['sector_avg_return_x'])
                else:
                    self.logger.error("合并后的数据中缺少 'sector_avg_return' 相关字段")
                    self.logger.error(f"可用字段: {list(merged_data.columns)}")
                    # 设置默认值
                    merged_data['sector_avg_return'] = 0.0

            return merged_data

        except Exception as e:
            self.logger.error(f"处理sector_avg_return字段merge失败: {e}")
            # 返回原始数据并设置默认值
            data['sector_avg_return'] = 0.0
            return data

    def _identify_target_and_historical_dates(self, stock_data: pd.DataFrame,
                                             sector_data: pd.DataFrame,
                                             kwargs: dict) -> tuple:
        """
        精确识别目标计算日期和历史支撑数据

        修复逻辑：
        1. 优先使用kwargs中传递的target_start_date和target_end_date
        2. 根据因子需要的历史数据长度，智能确定历史支撑数据范围
        3. 只为目标日期保存计算结果
        """
        try:
            # 1. 从kwargs中获取精确的目标日期范围
            target_start_date = kwargs.get('target_start_date')
            target_end_date = kwargs.get('target_end_date')

            if target_start_date and target_end_date:
                # 使用传递的精确目标日期
                target_dates = self._get_trading_dates_in_range(
                    stock_data, target_start_date, target_end_date
                )
                self.logger.info(f"使用传递的目标日期范围: {target_start_date} - {target_end_date}")
            else:
                # 回退到启发式方法
                self.logger.warning("未传递目标日期，使用启发式方法推断")
                target_dates, _ = self._infer_target_dates_from_data(stock_data, sector_data)

            # 使用所有可用数据作为历史数据（已通过extend_history_days=120预加载）
            historical_dates = sorted(stock_data['date'].dt.strftime('%Y%m%d').unique())

            self.logger.info(f"目标日期识别完成: {len(target_dates)} 个目标日期, {len(historical_dates)} 个历史日期")

            return target_dates, historical_dates

        except Exception as e:
            self.logger.error(f"识别目标和历史日期失败: {e}")
            return [], []

    def _infer_target_dates_from_data(self, stock_data: pd.DataFrame,
                                    sector_data: pd.DataFrame) -> tuple:
        """
        从数据中推断目标日期（启发式方法）

        启发式逻辑：
        1. 如果板块数据明显少于股票数据，板块数据日期为目标日期
        2. 否则，使用最后20%的日期作为目标日期
        """
        try:
            stock_dates = sorted(stock_data['date'].dt.strftime('%Y%m%d').unique())
            sector_dates = sorted(sector_data['date'].dt.strftime('%Y%m%d').unique())

            # 启发式1：板块数据明显少于股票数据
            if len(sector_dates) < len(stock_dates) * 0.5:
                target_dates = sector_dates
                historical_dates = [d for d in stock_dates if d not in target_dates]
                self.logger.info(f"启发式1：板块数据较少，使用板块日期作为目标日期")
            else:
                # 启发式2：使用最后20%的日期作为目标日期
                target_count = max(1, int(len(stock_dates) * 0.2))
                target_dates = stock_dates[-target_count:]
                historical_dates = stock_dates[:-target_count]
                self.logger.info(f"启发式2：使用最后{target_count}个日期作为目标日期")

            return target_dates, historical_dates

        except Exception as e:
            self.logger.error(f"推断目标日期失败: {e}")
            return [], []

    def _get_trading_dates_in_range(self, stock_data: pd.DataFrame,
                                  start_date: str, end_date: str) -> List[str]:
        """获取指定范围内的交易日期（修复版：支持增量计算模式）"""
        try:
            # 获取数据中的所有日期
            data_dates = stock_data['date'].dt.strftime('%Y%m%d').unique()

            # 生成目标日期范围（即使数据中不包含这些日期）
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            target_dates = []
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime('%Y%m%d')
                target_dates.append(date_str)
                current_dt += timedelta(days=1)

            # 过滤出实际存在的交易日（与数据中的日期交集）
            actual_target_dates = [d for d in target_dates if d in data_dates]

            # 如果数据中没有目标日期，返回指定的目标日期范围（用于增量计算模式）
            if not actual_target_dates:
                self.logger.info(f"数据中不包含目标日期，返回指定范围: {start_date} - {end_date}")
                return [start_date] if start_date == end_date else [start_date, end_date]

            return sorted(actual_target_dates)
        except Exception as e:
            self.logger.error(f"获取交易日期失败: {e}")
            return []



    def _get_optimal_historical_dates(self, stock_data: pd.DataFrame,
                                    target_dates: List[str],
                                    max_history_needed: int) -> List[str]:
        """
        获取最优的历史支撑数据范围

        逻辑：
        1. 从最早的目标日期向前推max_history_needed个交易日
        2. 确保历史数据足够支撑所有目标日期的计算
        3. 避免加载过多冗余的历史数据
        """
        try:
            if not target_dates:
                return []

            # 获取所有可用的交易日期
            all_dates = sorted(stock_data['date'].dt.strftime('%Y%m%d').unique())

            # 找到最早的目标日期在所有日期中的位置
            earliest_target_date = min(target_dates)
            try:
                earliest_target_index = all_dates.index(earliest_target_date)
            except ValueError:
                self.logger.warning(f"目标日期 {earliest_target_date} 不在数据中")
                return []

            # 计算需要的历史数据起始位置
            history_start_index = max(0, earliest_target_index - max_history_needed)
            history_end_index = earliest_target_index

            # 获取历史支撑数据日期
            historical_dates = all_dates[history_start_index:history_end_index]

            self.logger.info(f"历史数据优化: 从{len(all_dates)}个日期中选择{len(historical_dates)}个历史日期")
            self.logger.info(f"历史数据范围: {historical_dates[0] if historical_dates else 'N/A'} - {historical_dates[-1] if historical_dates else 'N/A'}")

            return historical_dates

        except Exception as e:
            self.logger.error(f"获取最优历史数据失败: {e}")
            return []

    def _calculate_sector_relative_features_optimized(self, stock_data: pd.DataFrame,
                                                    sector_data: pd.DataFrame,
                                                    stock_sector_mapping: Dict[str, List[str]],
                                                    target_dates: List[str],
                                                    historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化板块相对特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        """
        try:
            self.logger.info("开始正确优化的板块相对特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            # 转换目标日期为datetime格式以避免警告
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = stock_data['date'].isin(target_date_objects)
            result_data = stock_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(stock_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(stock_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(stock_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_sector_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_sector_features_with_history(
                stock_data, result_data, sector_data, stock_sector_mapping, target_dates
            )

            self.logger.info("正确优化的板块相对特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的板块相对特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return stock_data

    def _initialize_target_sector_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化板块相对特征列（内存优化）"""
        try:
            # 基础相对收益因子
            result_data['sector_relative_return_1d'] = 0.0
            result_data['sector_relative_return_5d'] = 0.0
            result_data['sector_relative_return_20d'] = 0.0
            result_data['sector_relative_strength'] = 0.0
            result_data['sector_outperform_ratio_20d'] = 0.0
            result_data['sector_correlation_20d'] = 0.0

            # 板块成交量相对排名因子
            result_data['sector_volume_rank_3d'] = 0.5
            result_data['sector_volume_rank_5d'] = 0.5
            result_data['sector_volume_rank_10d'] = 0.5
            result_data['sector_volume_rank_20d'] = 0.5

            # 板块涨幅相对排名因子
            result_data['sector_return_rank_3d'] = 0.5
            result_data['sector_return_rank_5d'] = 0.5
            result_data['sector_return_rank_10d'] = 0.5
            result_data['sector_return_rank_20d'] = 0.5

            # 板块动量强度因子
            result_data['sector_momentum_strength'] = 0.0
            result_data['sector_momentum_consistency'] = 0.0
            result_data['sector_momentum_acceleration'] = 0.0

            # 其他因子
            for factor_name in self.factor_names:
                if factor_name not in result_data.columns:
                    if 'rank' in factor_name or 'percentile' in factor_name:
                        result_data[factor_name] = 0.5  # 排名类特征默认为中位数
                    else:
                        result_data[factor_name] = 0.0

        except Exception as e:
            self.logger.error(f"初始化目标板块特征失败: {e}")

    def _calculate_target_sector_features_with_history(self, full_data: pd.DataFrame,
                                                     target_data: pd.DataFrame,
                                                     sector_data: pd.DataFrame,
                                                     stock_sector_mapping: Dict[str, List[str]],
                                                     target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的板块相对特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的板块相对特征（使用历史数据支撑）...")

            # 准备股票-板块映射数据
            stock_sector_df = self._prepare_stock_sector_mapping(stock_sector_mapping, sector_data)
            if stock_sector_df.empty:
                self.logger.warning("股票板块映射为空，使用默认值")
                return target_data

            # 预处理板块收益率数据
            sector_returns_df = self._prepare_sector_returns_lookup(sector_data)

            # 计算相对收益率特征（只为目标日期）
            target_data = self._calculate_target_relative_returns(
                full_data, target_data, sector_returns_df, stock_sector_mapping, target_dates
            )

            # 计算板块排名特征（只为目标日期）
            target_data = self._calculate_target_sector_ranking_features(
                full_data, target_data, sector_data, stock_sector_mapping, target_dates
            )

            self.logger.info("目标日期板块相对特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期板块相对特征计算失败: {e}")
            return target_data

    def _calculate_target_relative_returns(self, full_data: pd.DataFrame,
                                         target_data: pd.DataFrame,
                                         sector_returns_df: pd.DataFrame,
                                         stock_sector_mapping: Dict[str, List[str]],
                                         target_dates: List[str]) -> pd.DataFrame:
        """为目标日期计算相对收益率特征（使用历史数据支撑）"""
        try:
            self.logger.info("计算目标日期相对收益率特征...")

            # 为每个目标日期逐一计算相对收益率
            for target_date in target_dates:
                # 获取该目标日期的数据
                target_day_mask = target_data['date'] == target_date
                target_day_data = target_data[target_day_mask]

                if len(target_day_data) == 0:
                    continue

                # 为该目标日期计算相对收益率特征
                for idx, row in target_day_data.iterrows():
                    stock_code = row['stock_code']

                    # 获取该股票的历史数据（用于计算多期收益率）
                    historical_mask = (full_data['date'] <= target_date) & (full_data['stock_code'] == stock_code)
                    stock_historical = full_data[historical_mask].sort_values('date')

                    if len(stock_historical) < 2:
                        continue

                    # 计算1日相对收益率
                    rel_return_1d = self._calculate_relative_return_for_date(
                        stock_historical, sector_returns_df, stock_sector_mapping, stock_code, target_date, 1
                    )
                    target_data.loc[idx, 'sector_relative_return_1d'] = rel_return_1d

                    # 计算5日相对收益率
                    rel_return_5d = self._calculate_relative_return_for_date(
                        stock_historical, sector_returns_df, stock_sector_mapping, stock_code, target_date, 5
                    )
                    target_data.loc[idx, 'sector_relative_return_5d'] = rel_return_5d

                    # 计算20日相对收益率
                    rel_return_20d = self._calculate_relative_return_for_date(
                        stock_historical, sector_returns_df, stock_sector_mapping, stock_code, target_date, 20
                    )
                    target_data.loc[idx, 'sector_relative_return_20d'] = rel_return_20d

                    # 计算相对强度指标
                    rel_strength = self._calculate_relative_strength_for_date(
                        stock_historical, sector_returns_df, stock_sector_mapping, stock_code, target_date
                    )
                    target_data.loc[idx, 'sector_relative_strength'] = rel_strength

                    # 计算跑赢比例
                    outperform_ratio = self._calculate_outperform_ratio_for_date(
                        stock_historical, sector_returns_df, stock_sector_mapping, stock_code, target_date
                    )
                    target_data.loc[idx, 'sector_outperform_ratio_20d'] = outperform_ratio

                    # 计算相关性
                    correlation = self._calculate_correlation_for_date(
                        stock_historical, sector_returns_df, stock_sector_mapping, stock_code, target_date
                    )
                    target_data.loc[idx, 'sector_correlation_20d'] = correlation

            self.logger.info("目标日期相对收益率特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期相对收益率特征计算失败: {e}")
            return target_data

    def _calculate_target_sector_ranking_features(self, full_data: pd.DataFrame,
                                                target_data: pd.DataFrame,
                                                sector_data: pd.DataFrame,
                                                stock_sector_mapping: Dict[str, List[str]],
                                                target_dates: List[str]) -> pd.DataFrame:
        """为目标日期计算板块排名特征（使用历史数据支撑）"""
        try:
            self.logger.info("计算目标日期板块排名特征...")

            # 为每个目标日期计算板块排名
            for target_date in target_dates:
                # 计算该日期的板块排名
                sector_rankings = self._calculate_sector_rankings_for_date(
                    full_data, sector_data, target_date
                )

                # 将排名应用到目标日期的股票
                target_day_mask = target_data['date'] == target_date
                for idx, row in target_data[target_day_mask].iterrows():
                    stock_code = row['stock_code']

                    # 获取股票所属板块
                    stock_sectors = stock_sector_mapping.get(stock_code, [])
                    if not stock_sectors:
                        continue

                    # 计算平均排名
                    for period in [3, 5, 10, 20]:
                        volume_ranks = []
                        return_ranks = []

                        for sector in stock_sectors:
                            if sector in sector_rankings:
                                volume_ranks.append(sector_rankings[sector].get(f'volume_rank_{period}d', 0.5))
                                return_ranks.append(sector_rankings[sector].get(f'return_rank_{period}d', 0.5))

                        if volume_ranks:
                            target_data.loc[idx, f'sector_volume_rank_{period}d'] = np.mean(volume_ranks)
                        if return_ranks:
                            target_data.loc[idx, f'sector_return_rank_{period}d'] = np.mean(return_ranks)

            self.logger.info("目标日期板块排名特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期板块排名特征计算失败: {e}")
            return target_data

    def _calculate_relative_return_for_date(self, stock_historical: pd.DataFrame,
                                          sector_returns_df: pd.DataFrame,
                                          stock_sector_mapping: Dict[str, List[str]],
                                          stock_code: str, target_date: str, period: int) -> float:
        """为单个目标日期计算相对收益率"""
        try:
            # 获取目标日期的股票收益率
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 0.0

            if period == 1:
                # 1日相对收益率
                stock_return = target_row['pct_chg'].iloc[0] / 100.0

                # 获取板块收益率
                stock_sectors = stock_sector_mapping.get(stock_code, [])
                if not stock_sectors:
                    return 0.0

                sector_returns = []
                for sector in stock_sectors:
                    sector_row = sector_returns_df[
                        (sector_returns_df['sector_code'] == sector) &
                        (sector_returns_df['date'] == target_date)
                    ]
                    if len(sector_row) > 0:
                        sector_returns.append(sector_row['pct_chg'].iloc[0] / 100.0)

                if sector_returns:
                    sector_avg_return = np.mean(sector_returns)
                    return stock_return - sector_avg_return

                return 0.0

            else:
                # 多日相对收益率
                window_data = stock_historical.tail(period)
                if len(window_data) < period:
                    return 0.0

                # 计算股票累计收益率
                stock_cumret = (window_data['pct_chg'] / 100.0).sum()

                # 计算板块累计收益率
                stock_sectors = stock_sector_mapping.get(stock_code, [])
                if not stock_sectors:
                    return 0.0

                sector_cumrets = []
                for sector in stock_sectors:
                    sector_window = sector_returns_df[
                        (sector_returns_df['sector_code'] == sector) &
                        (sector_returns_df['date'].isin(window_data['date']))
                    ]
                    if len(sector_window) > 0:
                        sector_cumret = (sector_window['pct_chg'] / 100.0).sum()
                        sector_cumrets.append(sector_cumret)

                if sector_cumrets:
                    sector_avg_cumret = np.mean(sector_cumrets)
                    return stock_cumret - sector_avg_cumret

                return 0.0

        except Exception as e:
            self.logger.error(f"计算单日期相对收益率失败: {e}")
            return 0.0

    def _calculate_relative_strength_for_date(self, stock_historical: pd.DataFrame,
                                            sector_returns_df: pd.DataFrame,
                                            stock_sector_mapping: Dict[str, List[str]],
                                            stock_code: str, target_date: str) -> float:
        """为单个目标日期计算相对强度指标"""
        try:
            # 获取20日窗口数据
            window_data = stock_historical.tail(20)
            if len(window_data) < 10:
                return 0.0

            # 计算股票20日平均收益率
            stock_avg_return = (window_data['pct_chg'] / 100.0).mean()

            # 计算板块20日平均收益率
            stock_sectors = stock_sector_mapping.get(stock_code, [])
            if not stock_sectors:
                return 0.0

            sector_avg_returns = []
            for sector in stock_sectors:
                sector_window = sector_returns_df[
                    (sector_returns_df['sector_code'] == sector) &
                    (sector_returns_df['date'].isin(window_data['date']))
                ]
                if len(sector_window) > 0:
                    sector_avg_return = (sector_window['pct_chg'] / 100.0).mean()
                    sector_avg_returns.append(sector_avg_return)

            if sector_avg_returns:
                sector_overall_avg = np.mean(sector_avg_returns)
                if sector_overall_avg != 0:
                    return stock_avg_return / sector_overall_avg

            return 1.0

        except Exception as e:
            self.logger.error(f"计算单日期相对强度失败: {e}")
            return 1.0

    def _calculate_outperform_ratio_for_date(self, stock_historical: pd.DataFrame,
                                           sector_returns_df: pd.DataFrame,
                                           stock_sector_mapping: Dict[str, List[str]],
                                           stock_code: str, target_date: str) -> float:
        """为单个目标日期计算跑赢比例"""
        try:
            # 获取20日窗口数据
            window_data = stock_historical.tail(20)
            if len(window_data) < 10:
                return 0.0

            # 计算每日相对收益率
            outperform_count = 0
            total_count = 0

            for _, day_data in window_data.iterrows():
                day_date = day_data['date']
                stock_return = day_data['pct_chg'] / 100.0

                # 获取当日板块收益率
                stock_sectors = stock_sector_mapping.get(stock_code, [])
                if not stock_sectors:
                    continue

                sector_returns = []
                for sector in stock_sectors:
                    sector_row = sector_returns_df[
                        (sector_returns_df['sector_code'] == sector) &
                        (sector_returns_df['date'] == day_date)
                    ]
                    if len(sector_row) > 0:
                        sector_returns.append(sector_row['pct_chg'].iloc[0] / 100.0)

                if sector_returns:
                    sector_avg_return = np.mean(sector_returns)
                    if stock_return > sector_avg_return:
                        outperform_count += 1
                    total_count += 1

            if total_count > 0:
                return outperform_count / total_count

            return 0.0

        except Exception as e:
            self.logger.error(f"计算单日期跑赢比例失败: {e}")
            return 0.0

    def _calculate_correlation_for_date(self, stock_historical: pd.DataFrame,
                                      sector_returns_df: pd.DataFrame,
                                      stock_sector_mapping: Dict[str, List[str]],
                                      stock_code: str, target_date: str) -> float:
        """为单个目标日期计算相关性"""
        try:
            # 获取20日窗口数据
            window_data = stock_historical.tail(20)
            if len(window_data) < 10:
                return 0.0

            stock_returns = window_data['pct_chg'] / 100.0

            # 获取对应的板块收益率
            stock_sectors = stock_sector_mapping.get(stock_code, [])
            if not stock_sectors:
                return 0.0

            sector_returns_list = []
            for _, day_data in window_data.iterrows():
                day_date = day_data['date']
                sector_returns = []

                for sector in stock_sectors:
                    sector_row = sector_returns_df[
                        (sector_returns_df['sector_code'] == sector) &
                        (sector_returns_df['date'] == day_date)
                    ]
                    if len(sector_row) > 0:
                        sector_returns.append(sector_row['pct_chg'].iloc[0] / 100.0)

                if sector_returns:
                    sector_returns_list.append(np.mean(sector_returns))
                else:
                    sector_returns_list.append(0.0)

            if len(sector_returns_list) == len(stock_returns):
                correlation = np.corrcoef(stock_returns, sector_returns_list)[0, 1]
                if np.isnan(correlation):
                    return 0.0
                return correlation

            return 0.0

        except Exception as e:
            self.logger.error(f"计算单日期相关性失败: {e}")
            return 0.0

    def _calculate_sector_rankings_for_date(self, full_data: pd.DataFrame,
                                          sector_data: pd.DataFrame,
                                          target_date: str) -> Dict:
        """为单个目标日期计算板块排名"""
        try:
            # 简化的板块排名计算
            rankings = {}

            # 获取所有板块
            sectors = sector_data['sector_code'].unique()

            for period in [3, 5, 10, 20]:
                # 计算各板块的成交量和收益率排名
                sector_metrics = []

                for sector in sectors:
                    sector_window = sector_data[
                        (sector_data['sector_code'] == sector) &
                        (sector_data['date'] <= target_date)
                    ].tail(period)

                    if len(sector_window) > 0:
                        avg_volume = sector_window['volume'].mean() if 'volume' in sector_window.columns else 0
                        avg_return = sector_window['pct_chg'].mean() if 'pct_chg' in sector_window.columns else 0

                        sector_metrics.append({
                            'sector': sector,
                            'volume': avg_volume,
                            'return': avg_return
                        })

                if sector_metrics:
                    # 计算排名分位数
                    volumes = [m['volume'] for m in sector_metrics]
                    returns = [m['return'] for m in sector_metrics]

                    for i, metric in enumerate(sector_metrics):
                        sector = metric['sector']
                        if sector not in rankings:
                            rankings[sector] = {}

                        # 计算分位数排名
                        volume_rank = (np.sum(np.array(volumes) <= metric['volume']) - 1) / (len(volumes) - 1) if len(volumes) > 1 else 0.5
                        return_rank = (np.sum(np.array(returns) <= metric['return']) - 1) / (len(returns) - 1) if len(returns) > 1 else 0.5

                        rankings[sector][f'volume_rank_{period}d'] = volume_rank
                        rankings[sector][f'return_rank_{period}d'] = return_rank

            return rankings

        except Exception as e:
            self.logger.error(f"计算单日期板块排名失败: {e}")
            return {}

    def _calculate_sector_relative_features_redesigned(self, stock_data: pd.DataFrame,
                                                     sector_data: pd.DataFrame,
                                                     stock_sector_mapping: Dict[str, List[str]],
                                                     target_dates: List[str],
                                                     historical_dates: List[str]) -> pd.DataFrame:
        """
        重新设计的高效板块相对特征计算

        核心设计原则：
        1. 使用所有历史数据进行准确的滚动计算
        2. 只为目标日期保存计算结果
        3. 使用合适的min_periods确保计算准确性
        4. 保持高效的向量化计算
        """
        try:
            self.logger.info("开始重新设计的板块相对特征计算")

            # 1. 初始化所有因子为默认值
            for factor_name in self.factor_names:
                stock_data[factor_name] = 0.0

            # 2. 准备股票-板块映射数据
            stock_sector_df = self._prepare_stock_sector_mapping(stock_sector_mapping, sector_data)
            if stock_sector_df.empty:
                self.logger.warning("股票板块映射为空，使用默认值")
                return stock_data

            # 3. 预处理板块收益率数据
            sector_returns_df = self._prepare_sector_returns_lookup(sector_data)

            # 4. 计算相对收益率特征（使用增强版向量化方法，内部会计算板块平均收益率）
            # 注意：_calculate_relative_returns_vectorized 内部会调用 _calculate_stock_multi_sector_returns
            # 所以不需要单独调用 _calculate_stock_sector_average_returns
            stock_data = self._calculate_relative_returns_vectorized(
                stock_data, sector_returns_df, stock_sector_mapping
            )

            # 5. 计算板块排名特征（使用所有历史数据，但只为目标日期计算）
            stock_data = self._calculate_sector_ranking_features_efficient(
                stock_data, sector_data, stock_sector_mapping, target_dates
            )

            # 6. 只保留目标日期的计算结果，历史日期设为默认值
            stock_data = self._filter_results_to_target_dates(stock_data, target_dates)

            # 7. 保存目标日期信息到数据中，供缓存系统使用
            stock_data.attrs['target_dates'] = target_dates

            self.logger.info("重新设计的板块相对特征计算完成")
            return stock_data

        except Exception as e:
            self.logger.error(f"重新设计的板块相对特征计算失败: {e}")
            return stock_data





    def _calculate_sector_ranking_features_efficient(self, stock_data: pd.DataFrame,
                                                   sector_data: pd.DataFrame,
                                                   stock_sector_mapping: Dict[str, List[str]],
                                                   target_dates: List[str]) -> pd.DataFrame:
        """
        专业高效的板块排名特征计算

        计算板块在各个维度的排名分位数：
        1. 成交量排名因子（3日、5日、10日、20日）
        2. 涨幅排名因子（3日、5日、10日、20日）
        3. 只为目标日期计算和保存结果
        """
        try:
            self.logger.info("开始专业板块排名特征计算")

            # 1. 计算板块成交量排名因子
            stock_data = self._calculate_volume_ranking_factors_optimized(
                stock_data, sector_data, stock_sector_mapping, target_dates
            )

            # 2. 计算板块涨幅排名因子
            stock_data = self._calculate_return_ranking_factors_optimized(
                stock_data, sector_data, stock_sector_mapping, target_dates
            )

            # 3. 计算基于排名的动量因子
            stock_data = self._calculate_momentum_factors(
                stock_data, sector_data, stock_sector_mapping
            )

            # 4. 计算基于排名的流动性因子
            stock_data = self._calculate_liquidity_factors(
                stock_data, sector_data, stock_sector_mapping
            )

            self.logger.info("专业板块排名特征计算完成")
            return stock_data

        except Exception as e:
            self.logger.error(f"计算专业板块排名特征失败: {e}")
            return stock_data

    def _calculate_volume_ranking_factors_optimized(self, stock_data: pd.DataFrame,
                                                  sector_data: pd.DataFrame,
                                                  stock_sector_mapping: Dict[str, List[str]],
                                                  target_dates: List[str]) -> pd.DataFrame:
        """
        优化版板块成交量排名因子计算

        关键优化：
        1. 只为目标日期计算排名因子
        2. 使用正确的min_periods确保计算准确性
        3. 高效的向量化计算
        """
        try:
            # 检查缓存
            cache_key = f"volume_ranking_{self._generate_cache_key_for_sector_data(sector_data)}_{len(target_dates)}"
            if hasattr(self, '_volume_ranking_cache'):
                cached_result = self._volume_ranking_cache.get(cache_key)
                if cached_result is not None:
                    self.logger.info("使用缓存的成交量排名因子数据")
                    # 将缓存结果合并到stock_data
                    for period in [3, 5, 10, 20]:
                        factor_name = f'sector_volume_rank_{period}d'
                        if factor_name in cached_result.columns:
                            stock_data[factor_name] = cached_result[factor_name]
                    return stock_data

            if 'volume' not in sector_data.columns:
                self.logger.warning("板块数据中没有成交量字段，跳过成交量排名因子计算")
                # 设置默认值
                periods = [3, 5, 10, 20]
                for period in periods:
                    stock_data[f'sector_volume_rank_{period}d'] = 0.5
                return stock_data

            self.logger.info("开始优化版板块成交量排名因子计算（支持缓存）")

            # 1. 检查并使用已有的成交量移动平均数据，避免重复计算
            sector_data_sorted = sector_data.sort_values(['sector_code', 'date']).copy()
            periods = [3, 5, 10, 20]

            for period in periods:
                existing_field = f'vol_ma{period}'  # 板块数据中的标准字段名
                target_field = f'volume_ma_{period}d'

                if existing_field in sector_data_sorted.columns:
                    self.logger.info(f"使用已有的{period}日成交量移动平均数据: {existing_field}")
                    sector_data_sorted[target_field] = sector_data_sorted[existing_field]
                else:
                    self.logger.info(f"计算{period}日成交量滚动平均（数据中不存在{existing_field}）")
                    sector_data_sorted[target_field] = (
                        sector_data_sorted.groupby('sector_code')['volume']
                        .rolling(window=period, min_periods=period)
                        .mean()
                        .reset_index(0, drop=True)
                    )

            # 2. 只为目标日期计算排名分位数
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_sector_data = sector_data_sorted[sector_data_sorted['date'].isin(target_date_objects)].copy()

            for period in periods:
                self.logger.info(f"计算{period}日成交量排名分位数（仅目标日期）")
                target_sector_data[f'volume_rank_{period}d'] = (
                    target_sector_data.groupby('date')[f'volume_ma_{period}d']
                    .rank(pct=True, method='min')
                    .fillna(0.5)
                )

            # 3. 计算个股的板块平均排名（仅目标日期）
            stock_data = self._calculate_stock_sector_rankings_for_target_dates(
                stock_data, target_sector_data, stock_sector_mapping,
                [f'volume_rank_{p}d' for p in periods], 'volume', target_dates
            )

            # 缓存计算结果
            if not hasattr(self, '_volume_ranking_cache'):
                self._volume_ranking_cache = {}

            # 提取排名因子列进行缓存
            ranking_factors = [f'sector_volume_rank_{p}d' for p in periods]
            cache_data = stock_data[['stock_code', 'date'] + ranking_factors].copy()
            self._volume_ranking_cache[cache_key] = cache_data

            self.logger.info("优化版板块成交量排名因子计算完成（已缓存）")
            return stock_data

        except Exception as e:
            self.logger.error(f"计算优化版板块成交量排名因子失败: {e}")
            # 设置默认值
            periods = [3, 5, 10, 20]
            for period in periods:
                stock_data[f'sector_volume_rank_{period}d'] = 0.5
            return stock_data

    def _calculate_return_ranking_factors_optimized(self, stock_data: pd.DataFrame,
                                                  sector_data: pd.DataFrame,
                                                  stock_sector_mapping: Dict[str, List[str]],
                                                  target_dates: List[str]) -> pd.DataFrame:
        """
        优化版板块涨幅排名因子计算

        关键优化：
        1. 只为目标日期计算排名因子
        2. 使用正确的min_periods确保计算准确性
        3. 高效的向量化计算
        """
        try:
            # 使用统一的方法获取收益率字段
            try:
                return_field = self._get_sector_return_field(sector_data)
                self.logger.info(f"涨幅排名计算使用字段: {return_field}")
            except ValueError as e:
                self.logger.warning(f"{e}，跳过涨幅排名因子计算")
                # 设置默认值
                periods = [3, 5, 10, 20]
                for period in periods:
                    stock_data[f'sector_return_rank_{period}d'] = 0.5
                return stock_data

            self.logger.info("开始优化版板块涨幅排名因子计算")

            # 1. 直接使用板块数据中的change_percent，避免重复计算累计涨幅
            sector_data_sorted = sector_data.sort_values(['sector_code', 'date']).copy()
            periods = [3, 5, 10, 20]

            # 检查是否有预计算的累计收益率数据
            for period in periods:
                existing_field = f'return_{period}d'  # 可能的预计算字段
                target_field = f'cumret_{period}d'

                if existing_field in sector_data_sorted.columns:
                    self.logger.info(f"使用已有的{period}日累计收益率数据: {existing_field}")
                    sector_data_sorted[target_field] = sector_data_sorted[existing_field]
                else:
                    self.logger.info(f"计算{period}日涨幅滚动累计（数据中不存在{existing_field}）")
                    sector_data_sorted[target_field] = (
                        sector_data_sorted.groupby('sector_code')[return_field]
                        .rolling(window=period, min_periods=period)
                        .sum()
                        .reset_index(0, drop=True)
                    )

            # 2. 只为目标日期计算排名分位数
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_sector_data = sector_data_sorted[sector_data_sorted['date'].isin(target_date_objects)].copy()

            for period in periods:
                self.logger.info(f"计算{period}日涨幅排名分位数（仅目标日期）")
                target_sector_data[f'return_rank_{period}d'] = (
                    target_sector_data.groupby('date')[f'cumret_{period}d']
                    .rank(pct=True, method='min')
                    .fillna(0.5)
                )

            # 3. 计算个股的板块平均排名（仅目标日期）
            stock_data = self._calculate_stock_sector_rankings_for_target_dates(
                stock_data, target_sector_data, stock_sector_mapping,
                [f'return_rank_{p}d' for p in periods], 'return', target_dates
            )

            self.logger.info("优化版板块涨幅排名因子计算完成")
            return stock_data

        except Exception as e:
            self.logger.error(f"计算优化版板块涨幅排名因子失败: {e}")
            # 设置默认值
            periods = [3, 5, 10, 20]
            for period in periods:
                stock_data[f'sector_return_rank_{period}d'] = 0.5
            return stock_data

    def _calculate_stock_sector_rankings_for_target_dates(self, stock_data: pd.DataFrame,
                                                        target_sector_data: pd.DataFrame,
                                                        stock_sector_mapping: Dict[str, List[str]],
                                                        ranking_columns: List[str],
                                                        factor_type: str,
                                                        target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算个股的板块平均排名（超高效向量化版本）

        关键优化：
        1. 完全向量化计算，消除嵌套循环
        2. 使用DataFrame merge替代字典查找
        3. 批量处理所有排名因子
        4. 内存优化的数据结构
        """
        try:
            self.logger.info(f"开始超高效计算{factor_type}排名因子（仅目标日期）")

            # 1. 转换目标日期为datetime格式
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_stock_data = stock_data[stock_data['date'].isin(target_date_objects)].copy()

            if target_stock_data.empty:
                self.logger.warning("目标日期没有股票数据")
                return stock_data

            # 2. 构建股票-板块展开数据（向量化）
            stock_sector_expanded = []
            for stock_code, sectors in stock_sector_mapping.items():
                for sector in sectors:
                    stock_sector_expanded.append({
                        'stock_code': stock_code,
                        'sector_code': sector
                    })

            if not stock_sector_expanded:
                self.logger.warning("股票板块映射为空")
                return stock_data

            stock_sector_df = pd.DataFrame(stock_sector_expanded)

            # 3. 向量化计算所有排名因子
            for col in ranking_columns:
                period = col.split('_')[-1]  # 提取周期（如'3d', '5d'等）
                factor_name = f'sector_{factor_type}_rank_{period}'

                # 初始化因子列
                stock_data[factor_name] = 0.0

                # 步骤A：创建股票-日期-板块的完整组合
                stock_date_sector = target_stock_data[['stock_code', 'date']].merge(
                    stock_sector_df, on='stock_code', how='inner'
                )

                # 步骤B：向量化merge获取板块排名
                stock_with_rankings = stock_date_sector.merge(
                    target_sector_data[['sector_code', 'date', col]],
                    on=['sector_code', 'date'],
                    how='left'
                )

                # 步骤C：向量化计算每只股票的平均排名
                if not stock_with_rankings.empty:
                    avg_rankings = (
                        stock_with_rankings.groupby(['stock_code', 'date'])[col]
                        .mean()
                        .reset_index()
                        .rename(columns={col: factor_name})
                    )

                    # 步骤D：超高效向量化更新stock_data
                    # 使用merge方式更新，避免循环
                    stock_data = stock_data.merge(
                        avg_rankings[['stock_code', 'date', factor_name]],
                        on=['stock_code', 'date'],
                        how='left',
                        suffixes=('', '_new')
                    )

                    # 更新非空值
                    new_col = f'{factor_name}_new'
                    if new_col in stock_data.columns:
                        stock_data[factor_name] = stock_data[new_col].fillna(stock_data[factor_name])
                        stock_data = stock_data.drop(columns=[new_col])

            self.logger.info(f"{factor_type}排名因子计算完成（超高效向量化版）")
            return stock_data

        except Exception as e:
            self.logger.error(f"计算{factor_type}排名因子失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return stock_data

    def _filter_results_to_target_dates(self, stock_data: pd.DataFrame, target_dates: List[str]) -> pd.DataFrame:
        """
        只保留目标日期的计算结果，历史日期设为默认值
        """
        try:
            self.logger.info(f"过滤结果到目标日期: {len(target_dates)} 个日期")

            # 转换目标日期为datetime格式
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]

            # 创建目标日期掩码
            is_target_date = stock_data['date'].isin(target_date_objects)

            # 对于非目标日期，将因子值设为默认值
            for factor_name in self.factor_names:
                if factor_name in stock_data.columns:
                    stock_data.loc[~is_target_date, factor_name] = 0.0

            target_count = is_target_date.sum()
            non_target_count = (~is_target_date).sum()

            self.logger.info(f"过滤完成: {target_count}条目标记录保留, {non_target_count}条历史记录设为默认值")

            return stock_data

        except Exception as e:
            self.logger.error(f"过滤结果到目标日期失败: {e}")
            return stock_data






    
    def _calculate_sector_relative_features(self, stock_data: pd.DataFrame, 
                                          sector_data: pd.DataFrame,
                                          stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        计算板块相对特征（向量化实现）
        """
        try:
            self.logger.debug("开始向量化计算板块相对特征")
            
            # 1. 准备股票-板块映射数据
            stock_sector_df = self._prepare_stock_sector_mapping(stock_sector_mapping, sector_data)
            
            if stock_sector_df.empty:
                self.logger.warning("没有有效的股票-板块映射数据")
                return self._set_default_values(stock_data)
            
            # 2. 预处理板块收益率数据
            sector_returns_df = self._prepare_sector_returns_lookup(sector_data)

            # 3. 向量化计算相对特征（使用增强版方法，内部会计算板块平均收益率）
            # 注意：_calculate_relative_returns_vectorized 内部会调用 _calculate_stock_multi_sector_returns
            # 所以不需要单独调用 _calculate_stock_sector_average_returns
            stock_data = self._calculate_relative_returns_vectorized(
                stock_data, sector_returns_df, stock_sector_mapping
            )

            # 5. 计算板块排名因子（使用高效版本）
            target_dates = []  # 对于这个方法，计算所有日期
            stock_data = self._calculate_sector_ranking_features_efficient(
                stock_data, sector_data, stock_sector_mapping, target_dates
            )

            self.logger.info("板块相对特征向量化计算完成")
            return stock_data

        except Exception as e:
            self.logger.error(f"向量化计算板块相对特征失败: {e}")
            return self._set_default_values(stock_data)

    def _calculate_stock_sector_average_returns(self, stock_data: pd.DataFrame,
                                              sector_data: pd.DataFrame,
                                              stock_sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每只股票所属板块的平均涨幅（完全向量化版本）

        性能优化：
        1. 完全避免嵌套循环
        2. 使用pandas的merge和groupby操作
        3. 向量化计算，复杂度从O(n³)降到O(n log n)
        """
        try:
            self.logger.info("开始计算个股所属板块平均涨幅（完全向量化版）")
            import time
            start_time = time.time()

            # 使用统一的方法获取收益率字段
            try:
                return_field = self._get_sector_return_field(sector_data)
                self.logger.info(f"使用板块收益率字段: {return_field}")
            except ValueError as e:
                self.logger.warning(f"{e}")
                stock_data['sector_avg_return'] = 0.0
                return stock_data

            # 步骤1：展开股票-板块映射为长格式
            self.logger.info("步骤1: 展开股票-板块映射...")
            stock_sector_expanded = []
            for _, row in stock_sector_df.iterrows():
                stock_code = row['stock_code']
                for sector in row['all_sectors']:
                    stock_sector_expanded.append({
                        'stock_code': stock_code,
                        'sector_code': sector
                    })

            if not stock_sector_expanded:
                self.logger.warning("股票-板块映射为空")
                stock_data['sector_avg_return'] = 0.0
                return stock_data

            stock_sector_mapping = pd.DataFrame(stock_sector_expanded)
            self.logger.info(f"展开完成: {len(stock_sector_mapping)} 条股票-板块关系")

            # 步骤2：向量化合并操作
            self.logger.info("步骤2: 向量化合并股票数据和板块映射...")

            # 只保留必要的列，减少内存使用
            stock_subset = stock_data[['stock_code', 'date']].copy()

            # 合并股票数据和股票-板块映射
            stock_with_sectors = stock_subset.merge(
                stock_sector_mapping,
                on='stock_code',
                how='inner'  # 只保留有板块映射的股票
            )
            self.logger.info(f"合并后数据量: {len(stock_with_sectors)} 条记录")

            # 步骤3：合并板块收益率数据
            self.logger.info("步骤3: 合并板块收益率数据...")
            sector_subset = sector_data[['date', 'sector_code', return_field]].copy()

            stock_with_returns = stock_with_sectors.merge(
                sector_subset,
                on=['date', 'sector_code'],
                how='left'
            )

            # 填充缺失的板块收益率
            stock_with_returns[return_field] = stock_with_returns[return_field].fillna(0.0)
            self.logger.info(f"合并板块收益率完成: {len(stock_with_returns)} 条记录")

            # 步骤4：向量化计算平均收益率
            self.logger.info("步骤4: 向量化计算板块平均收益率...")
            sector_avg_returns = (
                stock_with_returns
                .groupby(['stock_code', 'date'])[return_field]
                .mean()
                .reset_index()
                .rename(columns={return_field: 'sector_avg_return'})
            )
            self.logger.info(f"计算完成: {len(sector_avg_returns)} 条平均收益率记录")

            # 步骤5：将结果合并回原始数据
            self.logger.info("步骤5: 合并结果到原始数据...")
            stock_data = stock_data.merge(
                sector_avg_returns,
                on=['stock_code', 'date'],
                how='left'
            )

            # 填充缺失值（没有板块映射的股票）
            stock_data['sector_avg_return'] = stock_data['sector_avg_return'].fillna(0.0)

            elapsed_time = time.time() - start_time
            self.logger.info(f"个股所属板块平均涨幅计算完成（向量化版），耗时 {elapsed_time:.2f}s")

            # 验证结果
            non_zero_count = (stock_data['sector_avg_return'] != 0).sum()
            total_count = len(stock_data)
            self.logger.info(f"结果验证: {non_zero_count}/{total_count} ({non_zero_count/total_count*100:.1f}%) 条记录有非零板块平均收益率")

            return stock_data

        except Exception as e:
            self.logger.error(f"计算个股所属板块平均涨幅失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            stock_data['sector_avg_return'] = 0.0
            return stock_data


    
    def _prepare_stock_sector_mapping(self, stock_sector_mapping: Dict[str, List[str]],
                                    sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备股票-板块映射数据框，支持多板块平均计算
        """
        try:
            # 获取板块数据中存在的板块列表
            available_sectors = set(sector_data['sector_code'].unique()) if 'sector_code' in sector_data.columns else set()

            mapping_records = []
            for stock_code, sectors in stock_sector_mapping.items():
                # 过滤出在sector_data中存在的板块
                valid_sectors = [s for s in sectors if s in available_sectors]

                if valid_sectors:
                    mapping_records.append({
                        'stock_code': stock_code,
                        'all_sectors': valid_sectors,
                        'sector_count': len(valid_sectors)
                    })

            if not mapping_records:
                self.logger.warning("没有找到有效的股票-板块映射")
                return pd.DataFrame()

            mapping_df = pd.DataFrame(mapping_records)
            self.logger.info(f"准备股票板块映射: {len(mapping_df)} 只股票，{len(available_sectors)} 个板块")

            return mapping_df

        except Exception as e:
            self.logger.error(f"准备股票板块映射失败: {e}")
            return pd.DataFrame()
    



    


    def _calculate_stock_sector_rankings_vectorized(self, stock_data: pd.DataFrame,
                                                  sector_ranking_data: pd.DataFrame,
                                                  stock_sector_mapping: Dict[str, List[str]],
                                                  ranking_columns: List[str],
                                                  factor_type: str) -> pd.DataFrame:
        """
        内存优化版：计算个股所属板块的平均排名（支持多板块平均，内存高效版本）
        """
        try:
            self.logger.info(f"开始计算{factor_type}排名因子（内存优化版）")

            # 初始化结果列
            for col in ranking_columns:
                stock_data[f'sector_{col}'] = 0.5

            # 创建板块排名查找表（内存高效）
            ranking_lookup = {}
            for col in ranking_columns:
                ranking_lookup[col] = sector_ranking_data.set_index(['sector_code', 'date'])[col].to_dict()

            # 按日期分批处理，避免内存爆炸
            unique_dates = stock_data['date'].unique()
            total_dates = len(unique_dates)

            for i, date in enumerate(unique_dates):
                if i % 20 == 0 or i == total_dates - 1:
                    progress = (i + 1) / total_dates * 100
                    self.logger.info(f"{factor_type}排名计算进度: {i+1}/{total_dates} ({progress:.1f}%)")

                # 获取当日股票数据
                date_mask = stock_data['date'] == date
                date_stocks = stock_data[date_mask]

                # 为每只股票计算板块平均排名
                for idx, row in date_stocks.iterrows():
                    stock_code = row['stock_code']

                    if stock_code in stock_sector_mapping:
                        sectors = stock_sector_mapping[stock_code]

                        # 计算每个排名列的平均值
                        for col in ranking_columns:
                            sector_rankings = []
                            for sector in sectors:
                                key = (sector, date)
                                if key in ranking_lookup[col]:
                                    sector_rankings.append(ranking_lookup[col][key])

                            if sector_rankings:
                                avg_ranking = np.mean(sector_rankings)
                                stock_data.loc[idx, f'sector_{col}'] = avg_ranking

            self.logger.info(f"{factor_type}排名因子计算完成（内存优化版）")
            return stock_data

        except Exception as e:
            self.logger.error(f"计算{factor_type}排名因子失败: {e}")
            # 设置默认值
            for col in ranking_columns:
                stock_data[f'sector_{col}'] = 0.5
            return stock_data



    def _calculate_momentum_factors(self, stock_data: pd.DataFrame,
                                  sector_data: pd.DataFrame,
                                  stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        计算板块动量强度因子

        基于板块在不同时间窗口的涨幅排名计算动量强度得分
        """
        try:
            # 使用已计算的涨幅排名来计算动量因子
            periods = [3, 5, 10, 20]

            # 1. 动量强度：多时间窗口排名的加权平均
            weights = [0.4, 0.3, 0.2, 0.1]  # 短期权重更高

            momentum_strength = np.zeros(len(stock_data))
            for i, period in enumerate(periods):
                rank_col = f'sector_return_rank_{period}d'
                if rank_col in stock_data.columns:
                    momentum_strength += weights[i] * stock_data[rank_col].fillna(0.5)

            stock_data['sector_momentum_strength'] = momentum_strength

            # 2. 动量一致性：不同时间窗口排名的标准差（越小越一致）
            rank_cols = [f'sector_return_rank_{p}d' for p in periods]
            existing_rank_cols = [col for col in rank_cols if col in stock_data.columns]

            if existing_rank_cols and len(existing_rank_cols) >= 2:
                rank_std = stock_data[existing_rank_cols].std(axis=1, skipna=True)
                # 转换为一致性得分（标准差越小，一致性越高）
                # 使用更敏感的转换函数增加变异性
                max_std = rank_std.max() if rank_std.max() > 0 else 1.0
                stock_data['sector_momentum_consistency'] = 1 - (rank_std / max_std).fillna(0.5)
            else:
                stock_data['sector_momentum_consistency'] = 0.5

            # 3. 动量加速度：短期排名 - 长期排名
            if ('sector_return_rank_3d' in stock_data.columns and
                'sector_return_rank_20d' in stock_data.columns):
                stock_data['sector_momentum_acceleration'] = (
                    stock_data['sector_return_rank_3d'] - stock_data['sector_return_rank_20d']
                ).fillna(0)
            else:
                stock_data['sector_momentum_acceleration'] = 0.0

            return stock_data

        except Exception as e:
            self.logger.error(f"计算板块动量因子失败: {e}")
            return stock_data

    def _calculate_liquidity_factors(self, stock_data: pd.DataFrame,
                                   sector_data: pd.DataFrame,
                                   stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        计算板块流动性因子

        基于板块成交量排名计算流动性得分
        """
        try:
            # 1. 流动性得分：多时间窗口成交量排名的加权平均
            periods = [3, 5, 10, 20]
            weights = [0.3, 0.3, 0.2, 0.2]  # 中短期权重较高

            liquidity_score = np.zeros(len(stock_data))
            for i, period in enumerate(periods):
                vol_rank_col = f'sector_volume_rank_{period}d'
                if vol_rank_col in stock_data.columns:
                    liquidity_score += weights[i] * stock_data[vol_rank_col].fillna(0.5)

            stock_data['sector_liquidity_score'] = liquidity_score

            # 2. 流动性稳定性：不同时间窗口成交量排名的标准差
            vol_rank_cols = [f'sector_volume_rank_{p}d' for p in periods]
            existing_vol_rank_cols = [col for col in vol_rank_cols if col in stock_data.columns]

            if existing_vol_rank_cols and len(existing_vol_rank_cols) >= 2:
                vol_rank_std = stock_data[existing_vol_rank_cols].std(axis=1, skipna=True)
                # 转换为稳定性得分（标准差越小，稳定性越高）
                # 使用更敏感的转换函数增加变异性
                max_std = vol_rank_std.max() if vol_rank_std.max() > 0 else 1.0
                stock_data['sector_liquidity_stability'] = 1 - (vol_rank_std / max_std).fillna(0.5)
            else:
                stock_data['sector_liquidity_stability'] = 0.5

            # 3. 相对流动性：短期成交量排名相对于长期的变化
            if ('sector_volume_rank_5d' in stock_data.columns and
                'sector_volume_rank_20d' in stock_data.columns):
                stock_data['sector_relative_liquidity'] = (
                    stock_data['sector_volume_rank_5d'] / (stock_data['sector_volume_rank_20d'] + 1e-6)
                ).fillna(1.0)
            else:
                stock_data['sector_relative_liquidity'] = 1.0

            return stock_data

        except Exception as e:
            self.logger.error(f"计算板块流动性因子失败: {e}")
            return stock_data

    def get_factor_names(self) -> List[str]:
        """获取因子名称列表"""
        return self.factor_names.copy()

    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """验证输入数据"""
        required_columns = ['stock_code', 'date', 'pct_chg']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            self.logger.error(f"输入数据缺少必要列: {missing_columns}")
            return False

        return True

    def _get_feature_names(self) -> List[str]:
        """获取特征名称列表（实现抽象方法）"""
        return self.factor_names.copy()

    def _calculate_risk_adjusted_relative_return_features(self, data: pd.DataFrame,
                                                        sector_data: pd.DataFrame,
                                                        stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        计算Risk-Adjusted Relative Return特征（高性能优化版）

        风险调整相对收益不仅看股票比板块涨得多，还要看承担了多少额外风险
        这帮助区分"真正的超额收益"和"高风险换来的收益"

        计算公式：Risk-Adjusted Relative Return = (个股收益率 - 板块收益率) / 跟踪误差
        其中跟踪误差 = std(个股收益率 - 板块收益率)

        性能优化：
        1. 向量化计算，避免嵌套循环
        2. 预处理板块数据，避免重复查找
        3. 批量处理，减少内存分配
        """
        try:
            self.logger.info("开始计算Risk-Adjusted Relative Return特征（高性能版）")
            import time
            start_time = time.time()

            # 初始化风险调整相对收益特征
            periods = [5, 20, 60]
            for period in periods:
                data[f'risk_adjusted_relative_return_{period}d'] = 0.0

            if sector_data is None or sector_data.empty:
                self.logger.warning("未提供板块数据，Risk-Adjusted Relative Return特征使用默认值")
                return data

            if not stock_sector_mapping:
                self.logger.warning("未提供股票板块映射，Risk-Adjusted Relative Return特征使用默认值")
                return data

            # 性能优化1：预处理板块收益率数据，创建高效查找表
            self.logger.info("预处理板块收益率数据...")
            sector_returns_df = self._prepare_sector_returns_lookup(sector_data)

            if sector_returns_df.empty:
                self.logger.warning("预处理数据为空，使用默认值")
                return data

            # 性能优化2：向量化计算相对收益率（基于多板块平均）
            self.logger.info("向量化计算相对收益率...")
            data_with_relative = self._calculate_relative_returns_vectorized(
                data, sector_returns_df, stock_sector_mapping
            )

            # 性能优化4：批量计算风险调整指标
            self.logger.info("批量计算风险调整指标...")
            for period in periods:
                self.logger.info(f"计算{period}日风险调整相对收益...")
                data_with_relative = self._calculate_risk_adjusted_returns_batch(
                    data_with_relative, period
                )

            # 复制结果到原数据
            for period in periods:
                factor_name = f'risk_adjusted_relative_return_{period}d'
                if factor_name in data_with_relative.columns:
                    data[factor_name] = data_with_relative[factor_name]

            elapsed_time = time.time() - start_time
            self.logger.info(f"Risk-Adjusted Relative Return特征计算完成，耗时 {elapsed_time:.2f}s")
            return data

        except Exception as e:
            self.logger.error(f"Risk-Adjusted Relative Return特征计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return data

    def _prepare_sector_returns_lookup(self, sector_data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理板块收益率数据，创建高效查找表（增强版：支持缓存）

        优化说明：
        1. 优先使用板块数据中已计算好的change_percent字段
        2. 避免重复计算板块收益率
        3. 确保数据一致性和计算效率
        4. 支持缓存机制，避免重复预处理
        """
        try:
            # 检查缓存
            cache_key = self._generate_cache_key_for_sector_data(sector_data)
            if hasattr(self, '_sector_returns_df_cache'):
                cached_result = self._sector_returns_df_cache.get(cache_key)
                if cached_result is not None:
                    self.logger.info("使用缓存的板块收益率查找表")
                    return cached_result

            # 使用统一的方法获取收益率字段
            try:
                return_field = self._get_sector_return_field(sector_data)
                self.logger.info(f"使用板块数据中的{return_field}字段（已预计算的板块涨跌幅）")
            except ValueError as e:
                self.logger.warning(f"{e}")
                return pd.DataFrame()

            # 验证必要字段
            required_fields = ['date', 'sector_code', return_field]
            missing_fields = [field for field in required_fields if field not in sector_data.columns]
            if missing_fields:
                self.logger.warning(f"板块数据缺少必要字段: {missing_fields}")
                return pd.DataFrame()

            # 创建高效的查找表
            sector_returns = sector_data[required_fields].copy()

            # 数据清理：移除缺失值
            initial_count = len(sector_returns)
            sector_returns = sector_returns.dropna()
            if len(sector_returns) < initial_count:
                self.logger.info(f"移除了 {initial_count - len(sector_returns)} 条包含缺失值的记录")

            # 转换收益率为小数形式（如果原始数据是百分比形式）
            # change_percent通常已经是百分比形式，需要转换为小数
            if return_field == 'change_percent':
                sector_returns[return_field] = sector_returns[return_field] / 100.0
            elif return_field == 'pct_chg':
                # pct_chg可能已经是小数形式，检查数据范围
                max_val = sector_returns[return_field].abs().max()
                if max_val > 1.0:  # 如果最大值大于1，说明是百分比形式
                    sector_returns[return_field] = sector_returns[return_field] / 100.0

            # 透视表格式，便于快速查找
            sector_pivot = sector_returns.pivot(index='date', columns='sector_code', values=return_field)
            sector_pivot = sector_pivot.fillna(0.0)

            # 缓存计算结果
            if not hasattr(self, '_sector_returns_df_cache'):
                self._sector_returns_df_cache = {}
            self._sector_returns_df_cache[cache_key] = sector_pivot.copy()

            self.logger.info(f"板块收益率查找表创建完成: {len(sector_pivot)} 个日期, {len(sector_pivot.columns)} 个板块（已缓存）")
            self.logger.info(f"收益率数据范围: [{sector_pivot.min().min():.4f}, {sector_pivot.max().max():.4f}]")

            return sector_pivot

        except Exception as e:
            self.logger.error(f"预处理板块收益率数据失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return pd.DataFrame()

    def _get_sector_return_field(self, sector_data: pd.DataFrame) -> str:
        """
        获取板块收益率字段名，统一处理逻辑

        Returns:
            str: 收益率字段名，优先返回change_percent
        """
        if 'change_percent' in sector_data.columns:
            return 'change_percent'
        elif 'pct_chg' in sector_data.columns:
            return 'pct_chg'
        else:
            raise ValueError("板块数据中没有找到收益率字段（change_percent或pct_chg）")



    def _calculate_relative_returns_vectorized(self, data: pd.DataFrame,
                                             sector_returns_df: pd.DataFrame,
                                             stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        向量化计算相对收益率（增强版：完整的相对收益率因子计算）

        功能增强：
        1. 使用多板块平均收益率替代主板块概念
        2. 计算完整的相对收益率因子（1日、5日、20日）
        3. 计算相对强度、跑赢比例、相关性等高级因子
        4. 保持向量化特性，提高计算效率
        """
        try:
            # 步骤1：计算每只股票的多板块平均收益率
            stock_sector_returns = self._calculate_stock_multi_sector_returns(
                data, sector_returns_df, stock_sector_mapping
            )

            # 调试信息：检查返回的数据
            if stock_sector_returns.empty:
                self.logger.error("_calculate_stock_multi_sector_returns 返回空DataFrame")
                return data

            self.logger.debug(f"stock_sector_returns字段: {list(stock_sector_returns.columns)}")
            self.logger.debug(f"stock_sector_returns形状: {stock_sector_returns.shape}")

            # 步骤2：使用统一的字段处理函数合并数据
            data_with_mapping = self._handle_sector_avg_return_merge(data, stock_sector_returns)

            data_with_mapping['stock_return'] = data_with_mapping['pct_chg'] / 100.0
            data_with_mapping['sector_return'] = data_with_mapping['sector_avg_return'].fillna(0)

            # 1日相对收益率
            data_with_mapping['sector_relative_return_1d'] = (
                data_with_mapping['stock_return'] - data_with_mapping['sector_return']
            )

            # 添加通用的relative_return字段，供风险调整计算使用
            data_with_mapping['relative_return'] = data_with_mapping['sector_relative_return_1d']

            # 步骤4：计算多期相对收益率（5日、20日）
            data_with_mapping = data_with_mapping.sort_values(['stock_code', 'date'])

            for period in [5, 20]:
                # 计算个股累计收益率
                stock_cumret = (
                    data_with_mapping.groupby('stock_code')['stock_return']
                    .rolling(period, min_periods=period)
                    .sum()
                    .reset_index(0, drop=True)
                )

                # 计算板块累计收益率
                sector_cumret = (
                    data_with_mapping.groupby('stock_code')['sector_return']
                    .rolling(period, min_periods=period)
                    .sum()
                    .reset_index(0, drop=True)
                )

                # 计算相对收益率
                data_with_mapping[f'sector_relative_return_{period}d'] = (
                    stock_cumret - sector_cumret
                ).fillna(0)

            # 步骤5：计算相对强度指标（20日）
            stock_strength = (
                data_with_mapping.groupby('stock_code')['stock_return']
                .rolling(20, min_periods=10)
                .mean()
                .reset_index(0, drop=True)
            )

            sector_strength = (
                data_with_mapping.groupby('stock_code')['sector_return']
                .rolling(20, min_periods=10)
                .mean()
                .reset_index(0, drop=True)
            )

            # 避免除零错误
            data_with_mapping['sector_relative_strength'] = np.where(
                sector_strength.abs() > 1e-6,
                stock_strength / sector_strength,
                1.0
            )

            # 步骤6：计算跑赢板块比例（20日）
            outperform_signals = (data_with_mapping['stock_return'] > data_with_mapping['sector_return']).astype(int)
            outperform_ratio = (
                outperform_signals.groupby(data_with_mapping['stock_code'])
                .rolling(20, min_periods=10)
                .mean()
                .reset_index(0, drop=True)
            )
            data_with_mapping['sector_outperform_ratio_20d'] = outperform_ratio.fillna(0.5)

            # 步骤7：计算与板块相关性（20日）
            def rolling_corr(group):
                stock_returns = group['stock_return']
                sector_returns = group['sector_return']
                return stock_returns.rolling(20, min_periods=10).corr(sector_returns)

            correlation = (
                data_with_mapping.groupby('stock_code')
                .apply(rolling_corr)
                .reset_index(0, drop=True)
            )
            data_with_mapping['sector_correlation_20d'] = correlation.fillna(0)

            self.logger.info("增强版向量化相对收益率计算完成（多板块平均版）")
            return data_with_mapping

        except Exception as e:
            self.logger.error(f"向量化计算相对收益率失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return data

    def _calculate_risk_adjusted_returns_batch(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        批量计算风险调整相对收益
        """
        try:
            factor_name = f'risk_adjusted_relative_return_{period}d'
            data[factor_name] = 0.0

            if 'relative_return' not in data.columns:
                return data

            # 确保数据按股票和日期排序
            data = data.sort_values(['stock_code', 'date'])

            # 向量化计算滚动统计量
            grouped = data.groupby('stock_code')['relative_return']

            # 计算滚动均值和标准差
            rolling_mean = grouped.rolling(window=period, min_periods=period).mean()
            rolling_std = grouped.rolling(window=period, min_periods=period).std()

            # 计算风险调整相对收益，改进除零处理
            # 当标准差为0或很小时，使用均值本身
            risk_adjusted = rolling_mean / (rolling_std + 1e-8)  # 添加小常数避免除零

            # 处理异常值：限制在合理范围内
            risk_adjusted = risk_adjusted.clip(-5, 5).fillna(0)

            # 当标准差过小时（说明收益率变化很小），直接使用均值
            small_std_mask = rolling_std < 1e-6
            risk_adjusted = risk_adjusted.where(~small_std_mask, rolling_mean)

            # 重置索引并赋值
            data[factor_name] = risk_adjusted.reset_index(level=0, drop=True)

            return data

        except Exception as e:
            self.logger.error(f"批量计算{period}日风险调整相对收益失败: {e}")
            return data

    def _calculate_sector_beta_features(self, data: pd.DataFrame,
                                      sector_data: pd.DataFrame,
                                      stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        计算Sector Beta特征（高性能优化版）

        板块Beta衡量个股相对于所属板块的敏感程度，类似市场Beta但使用板块收益率作为基准
        识别行业内的"领头羊"（高Beta）和"独行侠"（低Beta）
        在行业轮动时，高板块Beta的股票涨跌幅更大

        计算公式：Sector Beta = Cov(Stock_Return, Sector_Return) / Var(Sector_Return)

        性能优化：
        1. 复用已预处理的板块数据
        2. 向量化计算协方差和方差
        3. 批量处理多个周期
        """
        try:
            self.logger.info("开始计算Sector Beta特征（高性能版）")
            import time
            start_time = time.time()

            # 初始化板块Beta特征
            periods = [20, 60, 120]
            for period in periods:
                data[f'sector_beta_{period}d'] = 1.0  # 默认Beta为1

            if sector_data is None or sector_data.empty:
                self.logger.warning("未提供板块数据，Sector Beta特征使用默认值")
                return data

            if not stock_sector_mapping:
                self.logger.warning("未提供股票板块映射，Sector Beta特征使用默认值")
                return data

            # 复用预处理的数据（如果已存在）
            if hasattr(self, '_sector_returns_df'):
                sector_returns_df = self._sector_returns_df
            else:
                # 预处理数据
                sector_returns_df = self._prepare_sector_returns_lookup(sector_data)
                # 缓存预处理结果
                self._sector_returns_df = sector_returns_df

            if sector_returns_df.empty:
                self.logger.warning("预处理数据为空，使用默认值")
                return data

            # 向量化计算Beta（基于多板块平均）
            self.logger.info("向量化计算板块Beta...")
            data_with_beta = self._calculate_sector_beta_vectorized(
                data, sector_returns_df, stock_sector_mapping, periods
            )

            # 复制结果到原数据
            for period in periods:
                factor_name = f'sector_beta_{period}d'
                if factor_name in data_with_beta.columns:
                    data[factor_name] = data_with_beta[factor_name]

            elapsed_time = time.time() - start_time
            self.logger.info(f"Sector Beta特征计算完成，耗时 {elapsed_time:.2f}s")
            return data

        except Exception as e:
            self.logger.error(f"Sector Beta特征计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return data

    def _calculate_sector_beta_vectorized(self, data: pd.DataFrame,
                                        sector_returns_df: pd.DataFrame,
                                        stock_sector_mapping: Dict[str, List[str]],
                                        periods: List[int]) -> pd.DataFrame:
        """
        向量化计算板块Beta（重构版：基于多板块平均）

        新设计：
        1. 使用多板块平均收益率替代主板块概念
        2. 计算股票收益率与其所属所有板块平均收益率的Beta系数
        3. 保持向量化特性，提高计算效率
        """
        try:
            # 步骤1：计算每只股票的多板块平均收益率
            stock_sector_returns = self._calculate_stock_multi_sector_returns(
                data, sector_returns_df, stock_sector_mapping
            )

            # 步骤2：使用统一的字段处理函数合并数据
            data_with_mapping = self._handle_sector_avg_return_merge(data, stock_sector_returns)

            # 步骤4：准备收益率数据
            data_with_mapping['stock_return'] = data_with_mapping['pct_chg'] / 100.0
            data_with_mapping['sector_return'] = data_with_mapping['sector_avg_return'].fillna(0)

            # 确保数据按股票和日期排序
            data_with_mapping = data_with_mapping.sort_values(['stock_code', 'date'])

            # 为每个周期计算Beta
            for period in periods:
                factor_name = f'sector_beta_{period}d'
                data_with_mapping[factor_name] = 1.0  # 默认值

                # 向量化计算滚动协方差和方差
                grouped = data_with_mapping.groupby('stock_code')

                # 计算滚动协方差
                def rolling_cov(group):
                    stock_ret = group['stock_return']
                    sector_ret = group['sector_return']
                    return stock_ret.rolling(window=period, min_periods=period).cov(sector_ret)

                rolling_covariance = grouped.apply(rolling_cov).reset_index(level=0, drop=True)

                # 计算板块收益率的滚动方差
                rolling_variance = (
                    grouped['sector_return']
                    .rolling(window=period, min_periods=period)
                    .var()
                    .reset_index(level=0, drop=True)
                )

                # 计算Beta = Cov / Var
                beta = rolling_covariance / rolling_variance.replace(0, np.nan)
                beta = beta.clip(-3, 3).fillna(1.0)

                data_with_mapping[factor_name] = beta

            return data_with_mapping

        except Exception as e:
            self.logger.error(f"向量化计算板块Beta失败: {e}")
            return data

    def _calculate_migrated_sector_features(self, data: pd.DataFrame,
                                          sector_data: pd.DataFrame,
                                          stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        计算从correlation_factors.py迁移的板块相关因子（高性能优化版）
        使用专业的板块数据进行计算

        性能优化：
        1. 预计算板块股票分组，避免重复查找
        2. 向量化计算相关性
        3. 批量处理，减少循环次数
        """
        try:
            self.logger.info("开始计算迁移的板块相关因子（高性能版）")
            import time
            start_time = time.time()

            # 初始化特征
            data['sector_correlation'] = 0.0

            # 检查必要数据
            if sector_data is None or sector_data.empty:
                self.logger.warning("未提供板块数据，使用默认值")
                return data

            if stock_sector_mapping is None:
                self.logger.warning("未提供股票板块映射，使用默认值")
                return data

            # 性能优化1：向量化计算板块相关性
            self.logger.info("向量化计算板块相关性...")
            data = self._calculate_sector_correlation_vectorized(data, stock_sector_mapping)



            # 数据清理
            data['sector_correlation'] = data['sector_correlation'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            data['sector_correlation'] = np.clip(data['sector_correlation'], -1, 1)

            elapsed_time = time.time() - start_time
            self.logger.info(f"迁移的板块相关因子计算完成，耗时 {elapsed_time:.2f}s")
            return data

        except Exception as e:
            self.logger.error(f"迁移的板块相关因子计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return data



    def _calculate_sector_correlation_vectorized(self, data: pd.DataFrame,
                                               stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        向量化计算板块相关性（重构版：基于多板块平均收益率）

        新设计：
        1. 对每只股票，计算其所属所有板块的平均收益率
        2. 使用板块数据中的change_percent，避免重复计算
        3. 计算股票收益率与其板块平均收益率的相关性
        4. 完全向量化操作，提高计算效率
        """
        try:
            # 检查是否有板块收益率数据可用
            if not hasattr(self, '_sector_returns_df') or self._sector_returns_df.empty:
                self.logger.warning("没有可用的板块收益率数据，跳过板块相关性计算")
                data['sector_correlation'] = 0.0
                return data

            # 使用已预处理的板块收益率数据（来自sector_data的change_percent）
            sector_returns_df = self._sector_returns_df

            self.logger.info("计算股票与多板块平均收益率的相关性...")

            # 步骤1：为每只股票计算其所属板块的平均收益率
            stock_sector_returns = self._calculate_stock_multi_sector_returns(
                data, sector_returns_df, stock_sector_mapping
            )

            # 步骤2：使用统一的字段处理函数合并数据
            data_with_sector = self._handle_sector_avg_return_merge(data, stock_sector_returns)

            # 步骤4：准备收益率数据
            data_with_sector['stock_return'] = data_with_sector['pct_chg'] / 100.0
            data_with_sector['sector_avg_return'] = data_with_sector['sector_avg_return'].fillna(0)

            # 按股票分组计算滚动相关性
            def calculate_rolling_correlation(group):
                if len(group) < 20:
                    return pd.Series(0.0, index=group.index)

                stock_ret = group['stock_return']
                sector_ret = group['sector_avg_return']

                # 计算20日滚动相关性
                rolling_corr = stock_ret.rolling(20, min_periods=10).corr(sector_ret)
                return rolling_corr.fillna(0.0)

            # 向量化计算所有股票的相关性
            data_with_sector = data_with_sector.sort_values(['stock_code', 'date'])
            correlations = (
                data_with_sector
                .groupby('stock_code')
                .apply(calculate_rolling_correlation)
                .reset_index(level=0, drop=True)
            )

            # 将相关性结果合并回原始数据
            data['sector_correlation'] = correlations.reindex(data.index).fillna(0.0)

            self.logger.info("板块相关性计算完成（多板块平均版）")
            return data

        except Exception as e:
            self.logger.error(f"向量化计算板块相关性失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            data['sector_correlation'] = 0.0
            return data

    def _calculate_stock_multi_sector_returns(self, data: pd.DataFrame,
                                            sector_returns_df: pd.DataFrame,
                                            stock_sector_mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """
        计算每只股票所属多个板块的平均收益率（高效向量化版本 + 缓存优化）

        Args:
            data: 股票数据
            sector_returns_df: 板块收益率透视表（日期×板块）
            stock_sector_mapping: 股票-板块映射 {股票代码: [板块列表]}

        Returns:
            DataFrame: 包含stock_code, date, sector_avg_return的数据框
        """
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(data, sector_returns_df, stock_sector_mapping)
            if hasattr(self, '_stock_multi_sector_returns_cache'):
                cached_result = self._stock_multi_sector_returns_cache.get(cache_key)
                if cached_result is not None:
                    # 验证缓存数据的完整性
                    if 'sector_avg_return' in cached_result.columns:
                        self.logger.info("使用缓存的股票多板块平均收益率数据")
                        return cached_result
                    else:
                        self.logger.warning("缓存数据缺少 'sector_avg_return' 字段，重新计算")
                        # 清理损坏的缓存
                        del self._stock_multi_sector_returns_cache[cache_key]

            self.logger.info("计算股票多板块平均收益率（高效向量化版）...")

            # 获取所有股票-日期组合
            stock_date_combinations = data[['stock_code', 'date']].drop_duplicates()

            # 步骤1：展开股票-板块映射为长格式
            stock_sector_expanded = []
            for stock_code, sectors in stock_sector_mapping.items():
                for sector in sectors:
                    stock_sector_expanded.append({
                        'stock_code': stock_code,
                        'sector_code': sector
                    })

            if not stock_sector_expanded:
                self.logger.warning("股票-板块映射为空")
                stock_date_combinations['sector_avg_return'] = 0.0
                return stock_date_combinations

            stock_sector_df = pd.DataFrame(stock_sector_expanded)

            # 步骤2：向量化合并操作
            # 合并股票-日期组合和股票-板块映射
            stock_with_sectors = stock_date_combinations.merge(
                stock_sector_df,
                on='stock_code',
                how='left'
            )

            # 步骤3：合并板块收益率数据
            # 将透视表转换为长格式
            sector_returns_melted = sector_returns_df.reset_index().melt(
                id_vars='date',
                var_name='sector_code',
                value_name='sector_return'
            )

            stock_with_returns = stock_with_sectors.merge(
                sector_returns_melted,
                on=['date', 'sector_code'],
                how='left'
            )

            # 填充缺失的板块收益率
            stock_with_returns['sector_return'] = stock_with_returns['sector_return'].fillna(0.0)

            # 步骤4：向量化计算平均收益率
            sector_avg_returns = (
                stock_with_returns
                .groupby(['stock_code', 'date'])['sector_return']
                .mean()
                .reset_index()
                .rename(columns={'sector_return': 'sector_avg_return'})
            )

            # 步骤5：确保所有股票-日期组合都有结果
            result_df = stock_date_combinations.merge(
                sector_avg_returns,
                on=['stock_code', 'date'],
                how='left'
            )

            # 填充缺失值（没有板块映射的股票）
            result_df['sector_avg_return'] = result_df['sector_avg_return'].fillna(0.0)

            # 缓存计算结果
            if not hasattr(self, '_stock_multi_sector_returns_cache'):
                self._stock_multi_sector_returns_cache = {}
            self._stock_multi_sector_returns_cache[cache_key] = result_df.copy()

            self.logger.info(f"股票多板块平均收益率计算完成: {len(result_df)} 条记录（已缓存）")
            return result_df

        except Exception as e:
            self.logger.error(f"计算股票多板块平均收益率失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")

            # 返回默认值
            stock_date_combinations = data[['stock_code', 'date']].drop_duplicates()
            stock_date_combinations['sector_avg_return'] = 0.0
            return stock_date_combinations


