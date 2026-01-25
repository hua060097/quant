#!/usr/bin/env python3
"""
增强相关性因子组

基于CAPM模型和多因子模型理论，计算股票与市场的相关性和Beta系数
从feature_calculator.py的_calculate_correlation_features方法提取

现已扩展支持全市场状态相关因子，包括：
1. 市场波动率和趋势因子
2. 涨停/跌停/炸板个数排名分位数
3. 最高连板排名分位数
4. 最高涨幅股票排名
5. 大盘成交量排名分位数
6. 涨跌幅比率排名分位数

数据源结构说明：
- market_status.json: 包含市场指数的OHLCV数据、趋势强度等
  * individual_status['399006.SZ']['market_data']: {close, open, high, low, volume, daily_change_pct}
  * trend_strength: 市场趋势强度评分
- limit_data/*.parquet: 包含涨停、跌停、炸板的详细数据
  * trade_date: 交易日期
  * limit_times: 连板数
  * limit_type: 涨停类型
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from .base_factor_group import BaseFactorGroup

class EnhancedCorrelationFactors(BaseFactorGroup):
    """增强相关性因子组"""
    
    def __init__(self):
        super().__init__("enhanced_correlation_factors")
    
    def _get_version(self) -> str:
        return "v1.0"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接使用原始数据计算相关性特征
    
    def _get_feature_names(self) -> List[str]:
        """返回所有因子：原有3个相关性因子 + 新增的市场状态因子"""
        return [
            # 原有的3个相关性因子
            'market_correlation',      # 市场相关性（20日滚动相关性）
            'corr_slope_20_120',      # 相关性期限结构（120日相关性 - 20日相关性）
            'corr_20d_mom5',          # 20日相关性5日动量

            # 新增的市场状态因子
            # 1. 市场波动率和趋势因子
            'market_volatility_atr20d',    # 市场20日ATR波动率
            'market_trend_factor',         # 市场趋势因子

            # 2. 涨停个数排名分位数
            'limit_up_count_rank_60d',     # 涨停个数60日排名分位数
            'limit_up_count_rank_20d',     # 涨停个数20日排名分位数
            'limit_up_count_rank_10d',     # 涨停个数10日排名分位数
            'limit_up_count_rank_5d',      # 涨停个数5日排名分位数

            # 3. 跌停个数排名分位数
            'limit_down_count_rank_60d',   # 跌停个数60日排名分位数
            'limit_down_count_rank_20d',   # 跌停个数20日排名分位数
            'limit_down_count_rank_10d',   # 跌停个数10日排名分位数
            'limit_down_count_rank_5d',    # 跌停个数5日排名分位数

            # 4. 炸板个数排名分位数
            'broken_limit_count_rank_60d', # 炸板个数60日排名分位数
            'broken_limit_count_rank_20d', # 炸板个数20日排名分位数
            'broken_limit_count_rank_10d', # 炸板个数10日排名分位数
            'broken_limit_count_rank_5d',  # 炸板个数5日排名分位数

            # 5. 最高连板排名分位数
            'max_consecutive_limit_rank_30d', # 最高连板30日排名分位数
            'max_consecutive_limit_rank_20d', # 最高连板20日排名分位数
            'max_consecutive_limit_rank_10d', # 最高连板10日排名分位数
            'max_consecutive_limit_rank_5d',  # 最高连板5日排名分位数

            # 6. 最高涨幅股票排名
            'max_gain_10d_rank_60d',       # 近10日最高涨幅60日排名分位数
            'max_gain_5d_rank_30d',        # 近5日最高涨幅30日排名分位数

            # 7. 大盘成交量排名分位数
            'market_volume_rank_60d',      # 大盘成交量60日排名分位数
            'market_volume_rank_20d',      # 大盘成交量20日排名分位数
            'market_volume_rank_10d',      # 大盘成交量10日排名分位数
            'market_volume_rank_5d',       # 大盘成交量5日排名分位数

            # 8. 涨跌幅比率排名分位数
            'gain_loss_ratio_rank_60d',    # 涨跌幅比率60日排名分位数
            'gain_loss_ratio_rank_20d',    # 涨跌幅比率20日排名分位数
            'gain_loss_ratio_rank_10d',    # 涨跌幅比率10日排名分位数
            'gain_loss_ratio_rank_5d',     # 涨跌幅比率5日排名分位数
        ]
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算增强相关性因子（基于feature_calculator.py的专业实现）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return data
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            result_data = data.copy()

            # 获取传入的市场数据并进行验证
            market_data = kwargs.get('market_data')

            # 验证市场数据
            market_data_valid = self._validate_market_data(market_data)
            if not market_data_valid:
                self.logger.warning("市场数据无效，相关性因子将使用默认值")
                # 创建空的市场数据以避免计算错误
                market_data = None

            # 保存数据引用供市场状态因子计算使用
            self.full_data = data
            self.market_data = market_data

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_enhanced_correlation_features_optimized(
                data, target_dates, historical_dates, market_data
            )

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            return result_data

        except Exception as e:
            self.logger.error(f"增强相关性因子计算失败: {e}")
            return data

    def _calculate_enhanced_correlation_features_optimized(self, full_data: pd.DataFrame,
                                                         target_dates: List[str],
                                                         historical_dates: List[str],
                                                         market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        正确的优化增强相关性特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意相关性计算中的时间窗口边界，确保不使用未来信息
        """
        try:
            self.logger.info("开始正确优化的增强相关性特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_correlation_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_correlation_features_with_history(
                full_data, result_data, target_dates, market_data
            )

            self.logger.info("正确优化的增强相关性特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的增强相关性特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_correlation_features(self, result_data: pd.DataFrame):
        """为目标日期初始化所有相关性和市场状态特征列"""
        try:
            # 原有的3个相关性因子
            result_data['market_correlation'] = 0.0      # 市场相关性
            result_data['corr_slope_20_120'] = 0.0       # 相关性期限结构
            result_data['corr_20d_mom5'] = 0.0           # 20日相关性5日动量

            # 新增的市场状态因子
            # 1. 市场波动率和趋势因子
            result_data['market_volatility_atr20d'] = 0.0    # 市场20日ATR波动率
            result_data['market_trend_factor'] = 0.0         # 市场趋势因子

            # 2. 涨停个数排名分位数
            result_data['limit_up_count_rank_60d'] = 0.5     # 涨停个数60日排名分位数
            result_data['limit_up_count_rank_20d'] = 0.5     # 涨停个数20日排名分位数
            result_data['limit_up_count_rank_10d'] = 0.5     # 涨停个数10日排名分位数
            result_data['limit_up_count_rank_5d'] = 0.5      # 涨停个数5日排名分位数

            # 3. 跌停个数排名分位数
            result_data['limit_down_count_rank_60d'] = 0.5   # 跌停个数60日排名分位数
            result_data['limit_down_count_rank_20d'] = 0.5   # 跌停个数20日排名分位数
            result_data['limit_down_count_rank_10d'] = 0.5   # 跌停个数10日排名分位数
            result_data['limit_down_count_rank_5d'] = 0.5    # 跌停个数5日排名分位数

            # 4. 炸板个数排名分位数
            result_data['broken_limit_count_rank_60d'] = 0.5 # 炸板个数60日排名分位数
            result_data['broken_limit_count_rank_20d'] = 0.5 # 炸板个数20日排名分位数
            result_data['broken_limit_count_rank_10d'] = 0.5 # 炸板个数10日排名分位数
            result_data['broken_limit_count_rank_5d'] = 0.5  # 炸板个数5日排名分位数

            # 5. 最高连板排名分位数
            result_data['max_consecutive_limit_rank_30d'] = 0.5 # 最高连板30日排名分位数
            result_data['max_consecutive_limit_rank_20d'] = 0.5 # 最高连板20日排名分位数
            result_data['max_consecutive_limit_rank_10d'] = 0.5 # 最高连板10日排名分位数
            result_data['max_consecutive_limit_rank_5d'] = 0.5  # 最高连板5日排名分位数

            # 6. 最高涨幅股票排名
            result_data['max_gain_10d_rank_60d'] = 0.5       # 近10日最高涨幅60日排名分位数
            result_data['max_gain_5d_rank_30d'] = 0.5        # 近5日最高涨幅30日排名分位数

            # 7. 大盘成交量排名分位数
            result_data['market_volume_rank_60d'] = 0.5      # 大盘成交量60日排名分位数
            result_data['market_volume_rank_20d'] = 0.5      # 大盘成交量20日排名分位数
            result_data['market_volume_rank_10d'] = 0.5      # 大盘成交量10日排名分位数
            result_data['market_volume_rank_5d'] = 0.5       # 大盘成交量5日排名分位数

            # 8. 涨跌幅比率排名分位数
            result_data['gain_loss_ratio_rank_60d'] = 0.5    # 涨跌幅比率60日排名分位数
            result_data['gain_loss_ratio_rank_20d'] = 0.5    # 涨跌幅比率20日排名分位数
            result_data['gain_loss_ratio_rank_10d'] = 0.5    # 涨跌幅比率10日排名分位数
            result_data['gain_loss_ratio_rank_5d'] = 0.5     # 涨跌幅比率5日排名分位数

        except Exception as e:
            self.logger.error(f"初始化目标相关性和市场状态特征失败: {e}")

    def _calculate_target_correlation_features_with_history(self, full_data: pd.DataFrame,
                                                          target_data: pd.DataFrame,
                                                          target_dates: List[str],
                                                          market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的增强相关性特征（多线程优化版本）
        关键原则：历史数据只作支撑，只为目标日期计算指标

        性能优化：
        1. 预计算所有市场状态因子（避免重复计算）
        2. 多线程并行处理股票数据
        3. 线程安全的结果合并
        """
        try:
            self.logger.info("开始计算目标日期的增强相关性特征（多线程优化版本）...")

            # 验证市场数据
            if market_data is None or market_data.empty:
                self.logger.warning("市场数据无效，相关性因子将使用默认值")
                return target_data

            if 'daily_change_pct' not in market_data.columns:
                self.logger.warning("市场数据中缺少daily_change_pct字段")
                return target_data

            # 🚀 性能优化1：预计算所有目标日期的市场状态因子
            self.logger.info("预计算市场状态因子...")
            market_state_cache = self._precompute_market_state_features(
                target_dates, full_data, market_data
            )

            # 🚀 性能优化2：预处理市场数据（只处理一次）
            self.logger.info("预处理市场数据...")
            market_returns_cache = self._precompute_market_returns(market_data)

            # 🚀 性能优化3：多线程并行处理股票
            target_stocks = target_data['stock_code'].unique()
            total_stocks = len(target_stocks)

            self.logger.info(f"准备使用多线程处理 {total_stocks} 只股票的增强相关性特征")

            # 使用多线程处理股票
            result_data = self._process_stocks_multithreaded(
                full_data, target_data, target_dates, target_stocks,
                market_returns_cache, market_state_cache
            )

            self.logger.info("目标日期增强相关性特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"目标日期增强相关性特征计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return target_data

    def _process_stocks_multithreaded(self, full_data: pd.DataFrame,
                                    target_data: pd.DataFrame,
                                    target_dates: List[str],
                                    target_stocks: np.ndarray,
                                    market_returns_cache: dict,
                                    market_state_cache: dict) -> pd.DataFrame:
        """
        使用多线程并行处理股票数据（优化版本）

        优化点：
        1. 单股票级别并行，提高负载均衡
        2. 批量向量化合并，提升合并性能
        3. 详细的过滤统计日志

        Args:
            full_data: 完整的股票数据
            target_data: 目标日期的数据
            target_dates: 目标日期列表
            target_stocks: 目标股票列表
            market_returns_cache: 市场收益率缓存
            market_state_cache: 市场状态因子缓存

        Returns:
            pd.DataFrame: 更新后的目标数据
        """
        try:
            total_stocks = len(target_stocks)

            # 确定线程数量：基于CPU核心数，但不超过股票数量
            max_workers = min(os.cpu_count() or 4, total_stocks, 20)  # 增加到20个线程
            self.logger.info(f"使用 {max_workers} 个线程并行处理 {total_stocks} 只股票")

            # 线程安全的统计计数器
            progress_lock = threading.Lock()
            completed_count = [0]
            filtered_count = [0]  # 被过滤的股票数量

            def process_single_stock(stock_code):
                """处理单只股票的工作函数（优化版本）"""
                try:
                    # 获取该股票的完整历史数据
                    stock_full_data = full_data[full_data['stock_code'] == stock_code].sort_values('date')

                    # 检查历史数据是否足够
                    if len(stock_full_data) < 120:
                        with progress_lock:
                            filtered_count[0] += 1
                        return None

                    # 计算该股票的相关性特征（使用缓存数据）
                    stock_target_features = self._calculate_stock_target_correlation_features_optimized(
                        stock_full_data, target_dates, market_returns_cache, market_state_cache
                    )

                    # 获取该股票的目标日期数据掩码
                    stock_target_mask = target_data['stock_code'] == stock_code

                    # 线程安全的进度更新
                    with progress_lock:
                        completed_count[0] += 1
                        if completed_count[0] % max(1, total_stocks // 50) == 0 or completed_count[0] == total_stocks:
                            progress_pct = completed_count[0] / total_stocks * 100
                            self.logger.info(f"多线程处理进度: {completed_count[0]}/{total_stocks} ({progress_pct:.1f}%)")

                    return (stock_code, stock_target_features, stock_target_mask)

                except Exception as e:
                    self.logger.error(f"处理股票 {stock_code} 失败: {e}")
                    return None

            # 使用线程池执行并行处理（单股票级别）
            self.logger.info("开始多线程并行计算...")
            all_results = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有股票任务
                future_to_stock = {executor.submit(process_single_stock, stock): stock for stock in target_stocks}

                # 收集结果
                for future in as_completed(future_to_stock):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        stock = future_to_stock[future]
                        self.logger.error(f"股票 {stock} 处理失败: {e}")

            # 统计信息
            successful_count = len(all_results)
            filtered_count_final = filtered_count[0]
            self.logger.info(f"多线程计算完成统计:")
            self.logger.info(f"  • 输入股票总数: {total_stocks}")
            self.logger.info(f"  • 成功处理股票: {successful_count}")
            self.logger.info(f"  • 历史数据不足被过滤: {filtered_count_final}")
            self.logger.info(f"  • 处理失败股票: {total_stocks - successful_count - filtered_count_final}")

            # 🚀 优化的批量合并阶段
            self.logger.info(f"开始批量合并 {successful_count} 只股票的计算结果...")
            self._batch_update_target_features(target_data, all_results)

            self.logger.info(f"多线程处理完成，成功处理 {successful_count} 只股票")
            return target_data

        except Exception as e:
            self.logger.error(f"多线程处理失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return target_data

    def _batch_update_target_features(self, target_data: pd.DataFrame, all_results: list):
        """
        批量向量化更新目标数据特征（性能优化版本）

        优化策略：
        1. 预构建所有更新操作的索引映射
        2. 批量向量化更新，避免逐行操作
        3. 减少重复的掩码计算
        4. 使用DataFrame.update()进行高效合并

        Args:
            target_data: 目标数据DataFrame
            all_results: 所有股票的计算结果列表
        """
        try:
            if not all_results:
                return

            # 获取所有特征名称
            feature_names = self._get_feature_names()

            # 🚀 优化1：批量构建索引映射
            self.logger.info("构建批量更新索引映射...")

            # 预计算所有股票和日期的索引映射
            stock_date_index_map = {}
            for idx, row in target_data.iterrows():
                key = (row['stock_code'], row['date'])
                stock_date_index_map[key] = idx

            # 🚀 优化2：批量收集所有更新操作
            update_operations = []
            for stock_code, stock_target_features, stock_target_mask in all_results:
                for target_date_obj, features in stock_target_features.items():
                    key = (stock_code, target_date_obj)
                    if key in stock_date_index_map:
                        idx = stock_date_index_map[key]
                        update_operations.append((idx, features))

            self.logger.info(f"准备执行 {len(update_operations)} 个更新操作...")

            # 🚀 优化3：批量向量化更新
            for feature_name in feature_names:
                if feature_name in target_data.columns:
                    # 收集该特征的所有更新值
                    indices_to_update = []
                    values_to_update = []

                    for idx, features in update_operations:
                        if feature_name in features:
                            indices_to_update.append(idx)
                            values_to_update.append(features[feature_name])

                    # 批量更新
                    if indices_to_update:
                        target_data.loc[indices_to_update, feature_name] = values_to_update

            self.logger.info("批量合并完成")

        except Exception as e:
            self.logger.error(f"批量更新目标特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")

            # 降级到原始方法
            self.logger.info("降级使用原始合并方法...")
            for stock_code, stock_target_features, stock_target_mask in all_results:
                self._update_target_correlation_features(target_data, stock_target_features, stock_target_mask)

    def _precompute_market_state_features(self, target_dates: List[str],
                                        full_data: pd.DataFrame,
                                        market_data: pd.DataFrame) -> dict:
        """
        预计算所有目标日期的市场状态因子（避免重复计算）

        Returns:
            dict: {date_str: market_state_features_dict}
        """
        try:
            market_state_cache = {}

            # 预处理全市场数据（只处理一次）
            daily_limit_stats = self._precompute_daily_limit_stats(full_data)
            daily_gain_stats = self._precompute_daily_gain_stats(full_data)
            daily_ratio_stats = self._precompute_daily_ratio_stats(full_data)

            for target_date in target_dates:
                target_date_obj = pd.to_datetime(target_date, format='%Y%m%d')

                # 计算市场状态因子
                market_features = {}

                # 1. 市场波动率和趋势因子
                market_features['market_volatility_atr20d'] = self._calculate_market_volatility_atr(
                    target_date_obj, market_data
                )
                market_features['market_trend_factor'] = self._calculate_market_trend_factor(
                    target_date_obj, market_data
                )

                # 2-4. 涨停/跌停/炸板排名（使用预计算的统计数据）
                limit_features = self._calculate_limit_ranking_features_cached(
                    target_date_obj, daily_limit_stats
                )
                market_features.update(limit_features)

                # 5. 连板排名
                consecutive_features = self._calculate_consecutive_limit_ranking_features_cached(
                    target_date_obj, daily_limit_stats
                )
                market_features.update(consecutive_features)

                # 6. 最高涨幅排名
                gain_features = self._calculate_max_gain_ranking_features_cached(
                    target_date_obj, daily_gain_stats
                )
                market_features.update(gain_features)

                # 7. 成交量排名
                volume_features = self._calculate_market_volume_ranking_features(
                    target_date_obj, market_data
                )
                market_features.update(volume_features)

                # 8. 涨跌比率排名
                ratio_features = self._calculate_gain_loss_ratio_ranking_features_cached(
                    target_date_obj, daily_ratio_stats
                )
                market_features.update(ratio_features)

                market_state_cache[target_date] = market_features

            return market_state_cache

        except Exception as e:
            self.logger.error(f"预计算市场状态因子失败: {e}")
            return {}

    def _precompute_market_returns(self, market_data: pd.DataFrame) -> dict:
        """预计算市场收益率映射（避免重复处理）"""
        try:
            market_returns_cache = {}
            for _, row in market_data.iterrows():
                date_key = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                market_returns_cache[date_key] = row['daily_change_pct'] / 100.0
            return market_returns_cache
        except Exception as e:
            self.logger.error(f"预计算市场收益率失败: {e}")
            return {}

    def _calculate_stock_target_correlation_features_optimized(self, stock_full_data: pd.DataFrame,
                                                             target_dates: List[str],
                                                             market_returns_cache: dict,
                                                             market_state_cache: dict) -> dict:
        """为单只股票的目标日期计算增强相关性特征（性能优化版本）"""
        try:
            target_features = {}

            # 🚀 性能优化：使用预计算的市场收益率
            market_returns = self._prepare_market_returns_cached(stock_full_data, market_returns_cache)
            if market_returns is None:
                return {}

            # 为每个目标日期计算特征
            for target_date in target_dates:
                target_date_obj = pd.to_datetime(target_date, format='%Y%m%d')

                # 找到目标日期在数据中的位置
                date_mask = stock_full_data['date'] == target_date_obj
                if not date_mask.any():
                    continue

                target_idx = stock_full_data[date_mask].index[0]
                data_idx = stock_full_data.index.get_loc(target_idx)

                # 🚀 性能优化：只计算相关性因子，市场状态因子从缓存获取
                correlation_features = self._calculate_correlation_features_only(
                    stock_full_data, data_idx, market_returns
                )

                # 🚀 性能优化：从缓存获取市场状态因子
                market_state_features = market_state_cache.get(target_date, {})

                # 合并特征
                features = {**correlation_features, **market_state_features}
                target_features[target_date_obj] = features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标相关性特征失败: {e}")
            return {}

    def _calculate_stock_target_correlation_features(self, stock_full_data: pd.DataFrame,
                                                   target_dates: List[str],
                                                   market_data: pd.DataFrame) -> dict:
        """为单只股票的目标日期计算增强相关性特征（保留原方法以兼容）"""
        try:
            target_features = {}

            # 预处理市场数据
            market_returns = self._prepare_market_returns(stock_full_data, market_data)
            if market_returns is None:
                return {}

            # 为每个目标日期计算特征
            for target_date in target_dates:
                target_date_obj = pd.to_datetime(target_date, format='%Y%m%d')

                # 找到目标日期在数据中的位置
                date_mask = stock_full_data['date'] == target_date_obj
                if not date_mask.any():
                    continue

                target_idx = stock_full_data[date_mask].index[0]
                data_idx = stock_full_data.index.get_loc(target_idx)

                # 计算该日期的特征
                features = self._calculate_correlation_features_for_single_date(
                    stock_full_data, data_idx, market_returns
                )

                target_features[target_date_obj] = features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标相关性特征失败: {e}")
            return {}

    def _prepare_market_returns_cached(self, stock_data: pd.DataFrame,
                                     market_returns_cache: dict) -> pd.Series:
        """使用缓存的市场收益率准备与股票数据对齐的市场收益率（性能优化版本）"""
        try:
            # 为股票数据的每个日期匹配市场收益率
            market_returns = []
            for _, row in stock_data.iterrows():
                date_key = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                market_return = market_returns_cache.get(date_key, 0.0)
                market_returns.append(market_return)

            return pd.Series(market_returns, index=stock_data.index)

        except Exception as e:
            self.logger.error(f"准备缓存市场收益率失败: {e}")
            return None

    def _calculate_correlation_features_only(self, stock_data: pd.DataFrame,
                                           data_idx: int,
                                           market_returns: pd.Series) -> dict:
        """只计算相关性因子（不包括市场状态因子）"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 120:
                return {
                    'market_correlation': 0.0,
                    'corr_slope_20_120': 0.0,
                    'corr_20d_mom5': 0.0
                }

            # 获取股票收益率
            stock_returns = stock_data['pct_chg'] / 100.0

            # 计算20日和120日相关性
            corr_20d = 0.0
            corr_120d = 0.0

            # 计算20日相关性
            if data_idx >= 19:
                end_idx = data_idx
                start_idx = max(0, end_idx - 19)
                stock_window = stock_returns.iloc[start_idx:end_idx+1]
                market_window = market_returns.iloc[start_idx:end_idx+1]
                correlation = stock_window.corr(market_window)
                corr_20d = correlation if not pd.isna(correlation) else 0.0

            # 计算120日相关性
            if data_idx >= 119:
                end_idx = data_idx
                start_idx = max(0, end_idx - 119)
                stock_window = stock_returns.iloc[start_idx:end_idx+1]
                market_window = market_returns.iloc[start_idx:end_idx+1]
                correlation = stock_window.corr(market_window)
                corr_120d = correlation if not pd.isna(correlation) else 0.0

            # 计算相关性因子
            features['market_correlation'] = corr_20d
            features['corr_slope_20_120'] = corr_120d - corr_20d

            # 计算20日相关性5日动量
            if data_idx >= 24:
                prev_idx = data_idx - 5
                if prev_idx >= 19:
                    prev_end_idx = prev_idx
                    prev_start_idx = max(0, prev_end_idx - 19)
                    prev_stock_window = stock_returns.iloc[prev_start_idx:prev_end_idx+1]
                    prev_market_window = market_returns.iloc[prev_start_idx:prev_end_idx+1]
                    prev_correlation = prev_stock_window.corr(prev_market_window)
                    prev_corr_20d = prev_correlation if not pd.isna(prev_correlation) else 0.0
                    features['corr_20d_mom5'] = corr_20d - prev_corr_20d
                else:
                    features['corr_20d_mom5'] = 0.0
            else:
                features['corr_20d_mom5'] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"计算相关性因子失败: {e}")
            return {
                'market_correlation': 0.0,
                'corr_slope_20_120': 0.0,
                'corr_20d_mom5': 0.0
            }

    def _prepare_market_returns(self, stock_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.Series:
        """准备与股票数据对齐的市场收益率"""
        try:
            # 确保市场数据有正确的字段
            if 'daily_change_pct' not in market_data.columns:
                return None

            # 创建日期到市场收益率的映射
            market_dict = {}
            for _, row in market_data.iterrows():
                date_key = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                market_dict[date_key] = row['daily_change_pct'] / 100.0

            # 为股票数据的每个日期匹配市场收益率
            market_returns = []
            for _, row in stock_data.iterrows():
                date_key = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                market_return = market_dict.get(date_key, 0.0)
                market_returns.append(market_return)

            return pd.Series(market_returns, index=stock_data.index)

        except Exception as e:
            self.logger.error(f"准备市场收益率失败: {e}")
            return None

    def _calculate_correlation_features_for_single_date(self, stock_data: pd.DataFrame,
                                                      data_idx: int,
                                                      market_returns: pd.Series) -> dict:
        """为单个目标日期计算所有相关性和市场状态特征"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 120:
                return self._get_default_all_features()

            # 获取股票收益率
            stock_returns = stock_data['pct_chg'] / 100.0

            # ========== 第一部分：原有的相关性因子 ==========
            # 1. 计算20日和120日相关性（用于market_correlation和corr_slope_20_120）
            corr_20d = 0.0
            corr_120d = 0.0

            # 计算20日相关性
            if data_idx >= 19:  # 20日窗口需要至少20个数据点
                end_idx = data_idx
                start_idx = max(0, end_idx - 19)  # 20日窗口：[start_idx, end_idx]

                stock_window = stock_returns.iloc[start_idx:end_idx+1]
                market_window = market_returns.iloc[start_idx:end_idx+1]

                correlation = stock_window.corr(market_window)
                corr_20d = correlation if not pd.isna(correlation) else 0.0

            # 计算120日相关性
            if data_idx >= 119:  # 120日窗口需要至少120个数据点
                end_idx = data_idx
                start_idx = max(0, end_idx - 119)  # 120日窗口：[start_idx, end_idx]

                stock_window = stock_returns.iloc[start_idx:end_idx+1]
                market_window = market_returns.iloc[start_idx:end_idx+1]

                correlation = stock_window.corr(market_window)
                corr_120d = correlation if not pd.isna(correlation) else 0.0

            # 2. 计算原有的三个相关性因子
            # market_correlation: 使用20日相关性作为市场相关性
            features['market_correlation'] = corr_20d

            # corr_slope_20_120: 相关性期限结构（120日相关性 - 20日相关性）
            features['corr_slope_20_120'] = corr_120d - corr_20d

            # corr_20d_mom5: 20日相关性的5日动量
            if data_idx >= 24:  # 需要5日前的20日相关性，所以至少需要25个数据点
                # 计算5日前的20日相关性
                prev_idx = data_idx - 5
                if prev_idx >= 19:  # 确保5日前也有足够数据计算20日相关性
                    prev_end_idx = prev_idx
                    prev_start_idx = max(0, prev_end_idx - 19)

                    prev_stock_window = stock_returns.iloc[prev_start_idx:prev_end_idx+1]
                    prev_market_window = market_returns.iloc[prev_start_idx:prev_end_idx+1]

                    prev_correlation = prev_stock_window.corr(prev_market_window)
                    prev_corr_20d = prev_correlation if not pd.isna(prev_correlation) else 0.0

                    features['corr_20d_mom5'] = corr_20d - prev_corr_20d
                else:
                    features['corr_20d_mom5'] = 0.0
            else:
                features['corr_20d_mom5'] = 0.0

            # ========== 第二部分：新增的市场状态因子 ==========
            # 获取当前日期
            current_date = stock_data.iloc[data_idx]['date']

            # 计算市场状态因子（这些因子对所有股票在同一日期都相同）
            # 需要传入完整的数据以便进行全市场统计计算
            market_state_features = self._calculate_market_state_features_for_date(
                current_date, data_idx, stock_data, self.full_data, self.market_data
            )

            # 合并市场状态因子
            features.update(market_state_features)

            return features

        except Exception as e:
            self.logger.error(f"计算单日期相关性和市场状态特征失败: {e}")
            return self._get_default_all_features()

    def _precompute_daily_limit_stats(self, full_data: pd.DataFrame) -> pd.DataFrame:
        """预计算每日涨停/跌停/炸板统计数据"""
        try:
            required_cols = ['date', 'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_times']
            missing_cols = [col for col in required_cols if col not in full_data.columns]
            if missing_cols:
                self.logger.warning(f"缺少涨停统计所需列: {missing_cols}")
                return pd.DataFrame()

            # 按日期聚合统计
            daily_stats = full_data.groupby('date').agg({
                'is_limit_up': 'sum',
                'is_limit_down': 'sum',
                'is_zhaban': 'sum',
                'limit_times': 'max'
            }).reset_index()

            daily_stats.columns = ['date', 'limit_up_count', 'limit_down_count',
                                 'broken_limit_count', 'max_consecutive_limit']

            return daily_stats.sort_values('date').reset_index(drop=True)

        except Exception as e:
            self.logger.error(f"预计算涨停统计失败: {e}")
            return pd.DataFrame()

    def _precompute_daily_gain_stats(self, full_data: pd.DataFrame) -> pd.DataFrame:
        """预计算每日最高涨幅统计数据"""
        try:
            if 'pct_chg' not in full_data.columns:
                return pd.DataFrame()

            daily_gain_stats = full_data.groupby('date')['pct_chg'].max().reset_index()
            daily_gain_stats.columns = ['date', 'max_gain']

            return daily_gain_stats.sort_values('date').reset_index(drop=True)

        except Exception as e:
            self.logger.error(f"预计算涨幅统计失败: {e}")
            return pd.DataFrame()

    def _precompute_daily_ratio_stats(self, full_data: pd.DataFrame) -> pd.DataFrame:
        """预计算每日涨跌比率统计数据"""
        try:
            if 'pct_chg' not in full_data.columns:
                return pd.DataFrame()

            def calculate_daily_ratio(group):
                gains = (group['pct_chg'] > 0).sum()
                losses = (group['pct_chg'] < 0).sum()
                total_active = gains + losses
                return gains / total_active if total_active > 0 else 0.5

            daily_ratios = full_data.groupby('date').apply(calculate_daily_ratio).reset_index()
            daily_ratios.columns = ['date', 'gain_loss_ratio']

            return daily_ratios.sort_values('date').reset_index(drop=True)

        except Exception as e:
            self.logger.error(f"预计算涨跌比率失败: {e}")
            return pd.DataFrame()

    def _get_default_all_features(self) -> dict:
        """获取所有因子的默认值"""
        return {
            # 原有的3个相关性因子
            'market_correlation': 0.0,      # 市场相关性默认值
            'corr_slope_20_120': 0.0,       # 相关性期限结构默认值
            'corr_20d_mom5': 0.0,           # 20日相关性5日动量默认值

            # 新增的市场状态因子
            # 1. 市场波动率和趋势因子
            'market_volatility_atr20d': 0.0,    # 市场20日ATR波动率
            'market_trend_factor': 0.0,         # 市场趋势因子

            # 2. 涨停个数排名分位数
            'limit_up_count_rank_60d': 0.5,     # 涨停个数60日排名分位数
            'limit_up_count_rank_20d': 0.5,     # 涨停个数20日排名分位数
            'limit_up_count_rank_10d': 0.5,     # 涨停个数10日排名分位数
            'limit_up_count_rank_5d': 0.5,      # 涨停个数5日排名分位数

            # 3. 跌停个数排名分位数
            'limit_down_count_rank_60d': 0.5,   # 跌停个数60日排名分位数
            'limit_down_count_rank_20d': 0.5,   # 跌停个数20日排名分位数
            'limit_down_count_rank_10d': 0.5,   # 跌停个数10日排名分位数
            'limit_down_count_rank_5d': 0.5,    # 跌停个数5日排名分位数

            # 4. 炸板个数排名分位数
            'broken_limit_count_rank_60d': 0.5, # 炸板个数60日排名分位数
            'broken_limit_count_rank_20d': 0.5, # 炸板个数20日排名分位数
            'broken_limit_count_rank_10d': 0.5, # 炸板个数10日排名分位数
            'broken_limit_count_rank_5d': 0.5,  # 炸板个数5日排名分位数

            # 5. 最高连板排名分位数
            'max_consecutive_limit_rank_30d': 0.5, # 最高连板30日排名分位数
            'max_consecutive_limit_rank_20d': 0.5, # 最高连板20日排名分位数
            'max_consecutive_limit_rank_10d': 0.5, # 最高连板10日排名分位数
            'max_consecutive_limit_rank_5d': 0.5,  # 最高连板5日排名分位数

            # 6. 最高涨幅股票排名
            'max_gain_10d_rank_60d': 0.5,       # 近10日最高涨幅60日排名分位数
            'max_gain_5d_rank_30d': 0.5,        # 近5日最高涨幅30日排名分位数

            # 7. 大盘成交量排名分位数
            'market_volume_rank_60d': 0.5,      # 大盘成交量60日排名分位数
            'market_volume_rank_20d': 0.5,      # 大盘成交量20日排名分位数
            'market_volume_rank_10d': 0.5,      # 大盘成交量10日排名分位数
            'market_volume_rank_5d': 0.5,       # 大盘成交量5日排名分位数

            # 8. 涨跌幅比率排名分位数
            'gain_loss_ratio_rank_60d': 0.5,    # 涨跌幅比率60日排名分位数
            'gain_loss_ratio_rank_20d': 0.5,    # 涨跌幅比率20日排名分位数
            'gain_loss_ratio_rank_10d': 0.5,    # 涨跌幅比率10日排名分位数
            'gain_loss_ratio_rank_5d': 0.5,     # 涨跌幅比率5日排名分位数
        }

    def _update_target_correlation_features(self, target_data: pd.DataFrame,
                                          stock_target_features: dict,
                                          stock_target_mask: pd.Series):
        """更新目标数据中的相关性和市场状态特征"""
        try:
            for target_date_obj, features in stock_target_features.items():
                # 找到对应的行
                date_mask = target_data['date'] == target_date_obj
                combined_mask = stock_target_mask & date_mask

                if combined_mask.any():
                    for feature_name, feature_value in features.items():
                        if feature_name in target_data.columns:
                            target_data.loc[combined_mask, feature_name] = feature_value

        except Exception as e:
            self.logger.error(f"更新目标相关性和市场状态特征失败: {e}")

    def _calculate_market_state_features_for_date(self, current_date: pd.Timestamp,
                                                data_idx: int,
                                                stock_data: pd.DataFrame,
                                                full_data: pd.DataFrame,
                                                market_data: pd.DataFrame = None) -> dict:
        """
        计算指定日期的市场状态因子

        注意：这些因子对所有股票在同一日期都相同，是全市场层面的状态指标

        Args:
            current_date: 当前计算日期
            data_idx: 当前数据索引（在单只股票数据中的位置）
            stock_data: 单只股票的数据（用于获取日期信息）
            full_data: 完整的所有股票数据
            market_data: 市场指数数据

        Returns:
            dict: 包含所有市场状态因子的字典
        """
        try:
            # 初始化市场状态因子
            market_features = {}

            # 1. 市场波动率和趋势因子（基于market_data）
            market_features['market_volatility_atr20d'] = self._calculate_market_volatility_atr(
                current_date, market_data
            )
            market_features['market_trend_factor'] = self._calculate_market_trend_factor(
                current_date, market_data
            )

            # 2-4. 涨停/跌停/炸板个数排名分位数（基于full_data中的个股数据）
            limit_features = self._calculate_limit_ranking_features(
                current_date, full_data
            )
            market_features.update(limit_features)

            # 5. 最高连板排名分位数
            consecutive_features = self._calculate_consecutive_limit_ranking_features(
                current_date, full_data
            )
            market_features.update(consecutive_features)

            # 6. 最高涨幅股票排名
            gain_features = self._calculate_max_gain_ranking_features(
                current_date, full_data
            )
            market_features.update(gain_features)

            # 7. 大盘成交量排名分位数
            volume_features = self._calculate_market_volume_ranking_features(
                current_date, market_data
            )
            market_features.update(volume_features)

            # 8. 涨跌幅比率排名分位数
            ratio_features = self._calculate_gain_loss_ratio_ranking_features(
                current_date, full_data
            )
            market_features.update(ratio_features)

            return market_features

        except Exception as e:
            self.logger.error(f"计算市场状态因子失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            # 返回默认值
            return self._get_default_market_state_features()

    def _get_default_market_state_features(self) -> dict:
        """获取市场状态因子的默认值"""
        return {
            'market_volatility_atr20d': 0.0,
            'market_trend_factor': 0.0,
            'limit_up_count_rank_60d': 0.5, 'limit_up_count_rank_20d': 0.5,
            'limit_up_count_rank_10d': 0.5, 'limit_up_count_rank_5d': 0.5,
            'limit_down_count_rank_60d': 0.5, 'limit_down_count_rank_20d': 0.5,
            'limit_down_count_rank_10d': 0.5, 'limit_down_count_rank_5d': 0.5,
            'broken_limit_count_rank_60d': 0.5, 'broken_limit_count_rank_20d': 0.5,
            'broken_limit_count_rank_10d': 0.5, 'broken_limit_count_rank_5d': 0.5,
            'max_consecutive_limit_rank_30d': 0.5, 'max_consecutive_limit_rank_20d': 0.5,
            'max_consecutive_limit_rank_10d': 0.5, 'max_consecutive_limit_rank_5d': 0.5,
            'max_gain_10d_rank_60d': 0.5, 'max_gain_5d_rank_30d': 0.5,
            'market_volume_rank_60d': 0.5, 'market_volume_rank_20d': 0.5,
            'market_volume_rank_10d': 0.5, 'market_volume_rank_5d': 0.5,
            'gain_loss_ratio_rank_60d': 0.5, 'gain_loss_ratio_rank_20d': 0.5,
            'gain_loss_ratio_rank_10d': 0.5, 'gain_loss_ratio_rank_5d': 0.5,
        }

    def _calculate_market_volatility_atr(self, current_date: pd.Timestamp,
                                       market_data: pd.DataFrame = None) -> float:
        """
        计算市场20日ATR波动率

        基于market_data中的市场指数数据计算ATR（平均真实波动范围）
        ATR = Average True Range，衡量市场波动性的经典指标
        """
        try:
            if market_data is None or market_data.empty:
                return 0.0

            # 检查必需的列
            required_cols = ['date', 'high', 'low', 'close']
            if not all(col in market_data.columns for col in required_cols):
                self.logger.warning(f"市场数据缺少ATR计算所需列: {required_cols}")
                return 0.0

            # 按日期排序并找到当前日期的位置
            market_data_sorted = market_data.sort_values('date').reset_index(drop=True)
            market_data_sorted['date_str'] = market_data_sorted['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 找到当前日期在市场数据中的位置
            current_idx = None
            for idx, date_str in enumerate(market_data_sorted['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None or current_idx < 20:  # 需要至少20天数据计算ATR
                return 0.0

            # 计算True Range
            high = market_data_sorted['high'].values
            low = market_data_sorted['low'].values
            close = market_data_sorted['close'].values

            # 计算前一日收盘价
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]  # 第一天使用当天收盘价

            # 计算True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))

            # 计算20日ATR（使用当前日期及之前的20天数据）
            start_idx = max(0, current_idx - 19)
            end_idx = current_idx + 1
            atr_window = true_range[start_idx:end_idx]

            atr_20d = np.mean(atr_window)

            # 标准化ATR（除以当前价格）
            current_price = close[current_idx]
            if current_price > 0:
                normalized_atr = atr_20d / current_price
            else:
                normalized_atr = 0.0

            return float(normalized_atr)

        except Exception as e:
            self.logger.error(f"计算市场ATR波动率失败: {e}")
            return 0.0

    def _calculate_market_trend_factor(self, current_date: pd.Timestamp,
                                     market_data: pd.DataFrame = None) -> float:
        """
        计算市场趋势因子

        基于量化金融最佳实践，结合多个技术指标计算综合趋势强度：
        1. 价格趋势：多周期移动平均线排列
        2. 动量趋势：基于价格变化的动量计算
        3. 成交量趋势：量价配合度
        """
        try:
            if market_data is None or market_data.empty:
                return 0.0

            # 检查必需的列
            required_cols = ['date', 'close', 'volume']
            if not all(col in market_data.columns for col in required_cols):
                self.logger.warning(f"市场数据缺少趋势计算所需列: {required_cols}")
                return 0.0

            # 按日期排序并找到当前日期的位置
            market_data_sorted = market_data.sort_values('date').reset_index(drop=True)
            market_data_sorted['date_str'] = market_data_sorted['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 找到当前日期在市场数据中的位置
            current_idx = None
            for idx, date_str in enumerate(market_data_sorted['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None or current_idx < 20:  # 需要至少20天数据
                return 0.0

            close_prices = market_data_sorted['close'].values
            volumes = market_data_sorted['volume'].values

            # 1. 价格趋势：计算多周期移动平均线排列
            # 计算5日、10日、20日移动平均线（只使用当前日期及之前的数据）
            ma5 = np.mean(close_prices[max(0, current_idx-4):current_idx+1])
            ma10 = np.mean(close_prices[max(0, current_idx-9):current_idx+1])
            ma20 = np.mean(close_prices[max(0, current_idx-19):current_idx+1])

            # 均线排列得分：多头排列为正，空头排列为负
            ma_trend_score = 0.0
            if ma5 > ma10 > ma20:
                ma_trend_score = 1.0  # 完美多头排列
            elif ma5 > ma10:
                ma_trend_score = 0.5  # 部分多头排列
            elif ma10 > ma20:
                ma_trend_score = 0.25  # 轻微多头排列
            elif ma5 < ma10 < ma20:
                ma_trend_score = -1.0  # 完美空头排列
            elif ma5 < ma10:
                ma_trend_score = -0.5  # 部分空头排列
            elif ma10 < ma20:
                ma_trend_score = -0.25  # 轻微空头排列

            # 2. 动量趋势：计算价格动量
            if current_idx >= 5:
                current_price = close_prices[current_idx]
                price_5d_ago = close_prices[current_idx - 5]
                momentum_5d = (current_price - price_5d_ago) / price_5d_ago if price_5d_ago > 0 else 0.0
            else:
                momentum_5d = 0.0

            # 标准化动量到[-1, 1]区间
            momentum_score = np.tanh(momentum_5d * 10)  # 使用tanh函数进行软标准化

            # 3. 成交量趋势：量价配合度
            if current_idx >= 5:
                recent_volume = np.mean(volumes[max(0, current_idx-4):current_idx+1])
                past_volume = np.mean(volumes[max(0, current_idx-9):max(1, current_idx-4)])
                volume_ratio = recent_volume / past_volume if past_volume > 0 else 1.0

                # 量价配合：价格上涨且成交量放大为正，价格下跌且成交量放大为负
                if momentum_5d > 0 and volume_ratio > 1.2:
                    volume_score = 0.5  # 量价配合上涨
                elif momentum_5d < 0 and volume_ratio > 1.2:
                    volume_score = -0.5  # 量价配合下跌
                else:
                    volume_score = 0.0  # 量价不配合
            else:
                volume_score = 0.0

            # 综合趋势得分（加权平均）
            trend_factor = (ma_trend_score * 0.5 + momentum_score * 0.3 + volume_score * 0.2)

            # 确保结果在[-1, 1]区间内
            trend_factor = np.clip(trend_factor, -1.0, 1.0)

            return float(trend_factor)

        except Exception as e:
            self.logger.error(f"计算市场趋势因子失败: {e}")
            return 0.0

    def _calculate_limit_ranking_features(self, current_date: pd.Timestamp,
                                         full_data: pd.DataFrame) -> dict:
        """
        计算涨停、跌停、炸板个数的排名分位数

        基于full_data中的个股数据，聚合计算当日各类涨停数量在历史上的排名分位数
        """
        try:
            if full_data is None or full_data.empty:
                return self._get_default_limit_ranking_features()

            # 检查必需的列
            required_cols = ['date', 'is_limit_up', 'is_limit_down', 'is_zhaban']
            missing_cols = [col for col in required_cols if col not in full_data.columns]
            if missing_cols:
                self.logger.warning(f"个股数据缺少涨停计算所需列: {missing_cols}")
                return self._get_default_limit_ranking_features()

            # 按日期聚合计算每日的涨停、跌停、炸板个数
            daily_stats = full_data.groupby('date').agg({
                'is_limit_up': 'sum',      # 每日涨停个数
                'is_limit_down': 'sum',    # 每日跌停个数
                'is_zhaban': 'sum'         # 每日炸板个数
            }).reset_index()

            # 按日期排序
            daily_stats = daily_stats.sort_values('date').reset_index(drop=True)
            daily_stats['date_str'] = daily_stats['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 找到当前日期在统计数据中的位置
            current_idx = None
            for idx, date_str in enumerate(daily_stats['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_limit_ranking_features()

            # 计算各个时间窗口的排名分位数
            windows = [60, 20, 10, 5]
            result = {}

            # 涨停个数排名分位数
            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_stats['is_limit_up'], window, current_idx
                )
                result[f'limit_up_count_rank_{window}d'] = rank_percentile

            # 跌停个数排名分位数
            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_stats['is_limit_down'], window, current_idx
                )
                result[f'limit_down_count_rank_{window}d'] = rank_percentile

            # 炸板个数排名分位数
            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_stats['is_zhaban'], window, current_idx
                )
                result[f'broken_limit_count_rank_{window}d'] = rank_percentile

            return result

        except Exception as e:
            self.logger.error(f"计算涨停排名特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return self._get_default_limit_ranking_features()

    def _get_default_limit_ranking_features(self) -> dict:
        """获取涨停排名特征的默认值"""
        return {
            'limit_up_count_rank_60d': 0.5, 'limit_up_count_rank_20d': 0.5,
            'limit_up_count_rank_10d': 0.5, 'limit_up_count_rank_5d': 0.5,
            'limit_down_count_rank_60d': 0.5, 'limit_down_count_rank_20d': 0.5,
            'limit_down_count_rank_10d': 0.5, 'limit_down_count_rank_5d': 0.5,
            'broken_limit_count_rank_60d': 0.5, 'broken_limit_count_rank_20d': 0.5,
            'broken_limit_count_rank_10d': 0.5, 'broken_limit_count_rank_5d': 0.5,
        }

    def _calculate_consecutive_limit_ranking_features(self, current_date: pd.Timestamp,
                                                    full_data: pd.DataFrame) -> dict:
        """
        计算最高连板数的排名分位数

        基于full_data中的limit_times字段，计算当日最高连板数在历史上的排名分位数
        """
        try:
            if full_data is None or full_data.empty:
                return self._get_default_consecutive_limit_features()

            # 检查必需的列
            if 'limit_times' not in full_data.columns:
                self.logger.warning("个股数据缺少limit_times列，无法计算连板排名")
                return self._get_default_consecutive_limit_features()

            # 按日期聚合计算每日的最高连板数
            daily_max_consecutive = full_data.groupby('date')['limit_times'].max().reset_index()
            daily_max_consecutive.columns = ['date', 'max_consecutive_limit']

            # 按日期排序
            daily_max_consecutive = daily_max_consecutive.sort_values('date').reset_index(drop=True)
            daily_max_consecutive['date_str'] = daily_max_consecutive['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 找到当前日期在统计数据中的位置
            current_idx = None
            for idx, date_str in enumerate(daily_max_consecutive['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_consecutive_limit_features()

            # 计算各个时间窗口的排名分位数
            windows = [30, 20, 10, 5]
            result = {}

            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_max_consecutive['max_consecutive_limit'], window, current_idx
                )
                result[f'max_consecutive_limit_rank_{window}d'] = rank_percentile

            return result

        except Exception as e:
            self.logger.error(f"计算连板排名特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return self._get_default_consecutive_limit_features()

    def _get_default_consecutive_limit_features(self) -> dict:
        """获取连板排名特征的默认值"""
        return {
            'max_consecutive_limit_rank_30d': 0.5,
            'max_consecutive_limit_rank_20d': 0.5,
            'max_consecutive_limit_rank_10d': 0.5,
            'max_consecutive_limit_rank_5d': 0.5,
        }

    def _calculate_max_gain_ranking_features(self, current_date: pd.Timestamp,
                                           full_data: pd.DataFrame) -> dict:
        """
        计算最高涨幅股票的排名分位数

        基于full_data中的股票数据，计算当日近N日内最高涨幅股票的涨幅在历史上的排名分位数
        """
        try:
            if full_data is None or full_data.empty:
                return self._get_default_max_gain_features()

            # 检查必需的列
            if 'pct_chg' not in full_data.columns:
                self.logger.warning("个股数据缺少pct_chg列，无法计算最高涨幅排名")
                return self._get_default_max_gain_features()

            # 按日期聚合计算每日的最高涨幅
            daily_max_gain = full_data.groupby('date')['pct_chg'].max().reset_index()
            daily_max_gain.columns = ['date', 'max_gain']

            # 按日期排序
            daily_max_gain = daily_max_gain.sort_values('date').reset_index(drop=True)
            daily_max_gain['date_str'] = daily_max_gain['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 找到当前日期在统计数据中的位置
            current_idx = None
            for idx, date_str in enumerate(daily_max_gain['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_max_gain_features()

            result = {}

            # 计算近10日最高涨幅的60日排名分位数
            if current_idx >= 9:  # 需要至少10天数据
                # 计算近10日内的最高涨幅
                start_idx = max(0, current_idx - 9)
                end_idx = current_idx + 1
                max_gain_10d = daily_max_gain['max_gain'].iloc[start_idx:end_idx].max()

                # 创建一个包含这个最高涨幅的序列来计算排名
                max_gain_series = daily_max_gain['max_gain'].copy()
                max_gain_series.iloc[current_idx] = max_gain_10d

                # 计算60日排名分位数
                rank_percentile = self._calculate_rolling_rank_percentile(
                    max_gain_series, 60, current_idx
                )
                result['max_gain_10d_rank_60d'] = rank_percentile
            else:
                result['max_gain_10d_rank_60d'] = 0.5

            # 计算近5日最高涨幅的30日排名分位数
            if current_idx >= 4:  # 需要至少5天数据
                # 计算近5日内的最高涨幅
                start_idx = max(0, current_idx - 4)
                end_idx = current_idx + 1
                max_gain_5d = daily_max_gain['max_gain'].iloc[start_idx:end_idx].max()

                # 创建一个包含这个最高涨幅的序列来计算排名
                max_gain_series = daily_max_gain['max_gain'].copy()
                max_gain_series.iloc[current_idx] = max_gain_5d

                # 计算30日排名分位数
                rank_percentile = self._calculate_rolling_rank_percentile(
                    max_gain_series, 30, current_idx
                )
                result['max_gain_5d_rank_30d'] = rank_percentile
            else:
                result['max_gain_5d_rank_30d'] = 0.5

            return result

        except Exception as e:
            self.logger.error(f"计算最高涨幅排名特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return self._get_default_max_gain_features()

    def _get_default_max_gain_features(self) -> dict:
        """获取最高涨幅排名特征的默认值"""
        return {
            'max_gain_10d_rank_60d': 0.5,
            'max_gain_5d_rank_30d': 0.5,
        }

    def _calculate_market_volume_ranking_features(self, current_date: pd.Timestamp,
                                                 market_data: pd.DataFrame = None) -> dict:
        """
        计算大盘成交量的排名分位数

        基于market_data中的成交量数据，计算当日成交量在历史上的排名分位数
        """
        try:
            if market_data is None or market_data.empty:
                return self._get_default_volume_ranking_features()

            # 检查必需的列
            if 'volume' not in market_data.columns:
                self.logger.warning("市场数据缺少volume列，无法计算成交量排名")
                return self._get_default_volume_ranking_features()

            # 按日期排序并找到当前日期的位置
            market_data_sorted = market_data.sort_values('date').reset_index(drop=True)
            market_data_sorted['date_str'] = market_data_sorted['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 找到当前日期在市场数据中的位置
            current_idx = None
            for idx, date_str in enumerate(market_data_sorted['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_volume_ranking_features()

            # 计算各个时间窗口的排名分位数
            windows = [60, 20, 10, 5]
            result = {}

            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    market_data_sorted['volume'], window, current_idx
                )
                result[f'market_volume_rank_{window}d'] = rank_percentile

            return result

        except Exception as e:
            self.logger.error(f"计算成交量排名特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return self._get_default_volume_ranking_features()

    def _get_default_volume_ranking_features(self) -> dict:
        """获取成交量排名特征的默认值"""
        return {
            'market_volume_rank_60d': 0.5,
            'market_volume_rank_20d': 0.5,
            'market_volume_rank_10d': 0.5,
            'market_volume_rank_5d': 0.5,
        }

    def _calculate_gain_loss_ratio_ranking_features(self, current_date: pd.Timestamp,
                                                  full_data: pd.DataFrame) -> dict:
        """
        计算涨跌幅比率的排名分位数

        基于full_data中的全市场股票数据，计算当日涨跌家数比率在历史上的排名分位数
        涨跌幅比率 = 上涨股票数 / (上涨股票数 + 下跌股票数)
        """
        try:
            if full_data is None or full_data.empty:
                return self._get_default_gain_loss_ratio_features()

            # 检查必需的列
            if 'pct_chg' not in full_data.columns:
                self.logger.warning("个股数据缺少pct_chg列，无法计算涨跌比率排名")
                return self._get_default_gain_loss_ratio_features()

            # 按日期聚合计算每日的涨跌家数比率
            def calculate_daily_gain_loss_ratio(group):
                """计算单日的涨跌家数比率"""
                gains = (group['pct_chg'] > 0).sum()  # 上涨股票数
                losses = (group['pct_chg'] < 0).sum()  # 下跌股票数
                total_active = gains + losses  # 总的有涨跌的股票数（排除平盘）

                if total_active > 0:
                    ratio = gains / total_active
                else:
                    ratio = 0.5  # 如果没有涨跌股票，设为中性值

                return ratio

            daily_ratios = full_data.groupby('date').apply(calculate_daily_gain_loss_ratio).reset_index()
            daily_ratios.columns = ['date', 'gain_loss_ratio']

            # 按日期排序
            daily_ratios = daily_ratios.sort_values('date').reset_index(drop=True)
            daily_ratios['date_str'] = daily_ratios['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # 找到当前日期在统计数据中的位置
            current_idx = None
            for idx, date_str in enumerate(daily_ratios['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_gain_loss_ratio_features()

            # 计算各个时间窗口的排名分位数
            windows = [60, 20, 10, 5]
            result = {}

            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_ratios['gain_loss_ratio'], window, current_idx
                )
                result[f'gain_loss_ratio_rank_{window}d'] = rank_percentile

            return result

        except Exception as e:
            self.logger.error(f"计算涨跌幅比率排名特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return self._get_default_gain_loss_ratio_features()

    def _get_default_gain_loss_ratio_features(self) -> dict:
        """获取涨跌比率排名特征的默认值"""
        return {
            'gain_loss_ratio_rank_60d': 0.5,
            'gain_loss_ratio_rank_20d': 0.5,
            'gain_loss_ratio_rank_10d': 0.5,
            'gain_loss_ratio_rank_5d': 0.5,
        }

    def _calculate_rolling_rank_percentile(self, values: pd.Series, window: int,
                                         current_idx: int) -> float:
        """
        计算滚动窗口内的排名分位数（确保不使用未来信息）

        Args:
            values: 时间序列数据
            window: 滚动窗口大小
            current_idx: 当前数据索引

        Returns:
            float: 排名分位数 [0, 1]
        """
        try:
            if current_idx < window - 1:
                return 0.5  # 数据不足时返回中位数

            # 获取历史窗口数据（不包含未来信息）
            start_idx = max(0, current_idx - window + 1)
            end_idx = current_idx + 1  # 包含当前值

            window_values = values.iloc[start_idx:end_idx]
            current_value = values.iloc[current_idx]

            # 计算排名分位数
            rank = (window_values < current_value).sum()
            percentile = rank / (len(window_values) - 1) if len(window_values) > 1 else 0.5

            return np.clip(percentile, 0.0, 1.0)

        except Exception as e:
            self.logger.error(f"计算滚动排名分位数失败: {e}")
            return 0.5











    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> tuple:
        """识别目标计算日期和历史支撑数据"""
        try:
            target_start_date = kwargs.get('target_start_date')
            target_end_date = kwargs.get('target_end_date')
            if target_start_date and target_end_date:
                target_dates = self._get_trading_dates_in_range(data, target_start_date, target_end_date)
                self.logger.info(f"使用传递的目标日期范围: {target_start_date} - {target_end_date}")
            else:
                self.logger.error("必须提供target_start_date和target_end_date参数")
                return [], []

            # 获取目标日期范围
            target_dates = self._get_trading_dates_in_range(data, target_start_date, target_end_date)

            if not target_dates:
                self.logger.error(f"数据中不包含指定的目标日期范围: {target_start_date} - {target_end_date}")
                return [], []

            # 使用所有可用数据作为历史数据
            historical_dates = sorted(data['date'].dt.strftime('%Y%m%d').unique())
            self.logger.info(f"目标日期识别完成: {len(target_dates)} 个目标日期, {len(historical_dates)} 个历史日期")
            return target_dates, historical_dates
        except Exception as e:
            self.logger.error(f"识别目标和历史日期失败: {e}")
            return [], []

    def _validate_no_future_leakage(self, target_dates: List[str], current_date: str = None) -> bool:
        """验证目标日期不包含未来数据，防止时间泄露"""
        try:
            from datetime import datetime
            
            if current_date is None:
                current_date = datetime.now().strftime('%Y%m%d')
            
            future_dates = [d for d in target_dates if d > current_date]
            if future_dates:
                self.logger.error(f"❌ 检测到未来日期泄露: {future_dates}")
                self.logger.error(f"   当前日期: {current_date}")
                return False
            
            self.logger.debug(f"✅ 时间边界检查通过，所有目标日期都在 {current_date} 之前")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 时间泄露验证失败: {e}")
            return False

    def _get_trading_dates_in_range(self, stock_data: pd.DataFrame, start_date: str, end_date: str) -> List[str]:
        """获取指定范围内的交易日期"""
        try:
            data_dates = stock_data['date'].dt.strftime('%Y%m%d').unique()
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            target_dates = []
            current_dt = start_dt
            while current_dt <= end_dt:
                target_dates.append(current_dt.strftime('%Y%m%d'))
                current_dt += timedelta(days=1)
            actual_target_dates = [d for d in target_dates if d in data_dates]
            if not actual_target_dates:
                self.logger.warning(f"⚠️ 数据中不包含任何目标日期: {start_date} - {end_date}")
                return []
            return sorted(actual_target_dates)
        except Exception as e:
            self.logger.error(f"获取交易日期失败: {e}")
            return []





    def _validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """验证市场数据的有效性"""
        try:
            if market_data is None or market_data.empty:
                return False

            # 检查必需的列
            required_columns = ['date', 'daily_change_pct']
            missing_columns = [col for col in required_columns if col not in market_data.columns]

            if missing_columns:
                self.logger.warning(f"市场数据缺少必需列: {missing_columns}")
                return False

            # 检查数据量
            if len(market_data) < 10:
                self.logger.warning(f"市场数据量不足: {len(market_data)} < 10")
                return False

            return True

        except Exception as e:
            self.logger.error(f"市场数据验证失败: {e}")
            return False

    # ========== 缓存版本的计算方法（性能优化） ==========

    def _calculate_limit_ranking_features_cached(self, current_date: pd.Timestamp,
                                               daily_limit_stats: pd.DataFrame) -> dict:
        """使用预计算统计数据计算涨停排名特征（性能优化版本）"""
        try:
            if daily_limit_stats.empty:
                return self._get_default_limit_ranking_features()

            # 找到当前日期在统计数据中的位置
            daily_limit_stats['date_str'] = daily_limit_stats['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            current_idx = None
            for idx, date_str in enumerate(daily_limit_stats['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_limit_ranking_features()

            # 计算各个时间窗口的排名分位数
            windows = [60, 20, 10, 5]
            result = {}

            # 涨停个数排名分位数
            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_limit_stats['limit_up_count'], window, current_idx
                )
                result[f'limit_up_count_rank_{window}d'] = rank_percentile

            # 跌停个数排名分位数
            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_limit_stats['limit_down_count'], window, current_idx
                )
                result[f'limit_down_count_rank_{window}d'] = rank_percentile

            # 炸板个数排名分位数
            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_limit_stats['broken_limit_count'], window, current_idx
                )
                result[f'broken_limit_count_rank_{window}d'] = rank_percentile

            return result

        except Exception as e:
            self.logger.error(f"计算缓存涨停排名特征失败: {e}")
            return self._get_default_limit_ranking_features()

    def _calculate_consecutive_limit_ranking_features_cached(self, current_date: pd.Timestamp,
                                                           daily_limit_stats: pd.DataFrame) -> dict:
        """使用预计算统计数据计算连板排名特征（性能优化版本）"""
        try:
            if daily_limit_stats.empty or 'max_consecutive_limit' not in daily_limit_stats.columns:
                return self._get_default_consecutive_limit_features()

            # 找到当前日期在统计数据中的位置
            daily_limit_stats['date_str'] = daily_limit_stats['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            current_idx = None
            for idx, date_str in enumerate(daily_limit_stats['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_consecutive_limit_features()

            # 计算各个时间窗口的排名分位数
            windows = [30, 20, 10, 5]
            result = {}

            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_limit_stats['max_consecutive_limit'], window, current_idx
                )
                result[f'max_consecutive_limit_rank_{window}d'] = rank_percentile

            return result

        except Exception as e:
            self.logger.error(f"计算缓存连板排名特征失败: {e}")
            return self._get_default_consecutive_limit_features()

    def _calculate_max_gain_ranking_features_cached(self, current_date: pd.Timestamp,
                                                  daily_gain_stats: pd.DataFrame) -> dict:
        """使用预计算统计数据计算最高涨幅排名特征（性能优化版本）"""
        try:
            if daily_gain_stats.empty:
                return self._get_default_max_gain_features()

            # 找到当前日期在统计数据中的位置
            daily_gain_stats['date_str'] = daily_gain_stats['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            current_idx = None
            for idx, date_str in enumerate(daily_gain_stats['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_max_gain_features()

            result = {}

            # 计算近10日最高涨幅的60日排名分位数
            if current_idx >= 9:
                start_idx = max(0, current_idx - 9)
                end_idx = current_idx + 1
                max_gain_10d = daily_gain_stats['max_gain'].iloc[start_idx:end_idx].max()

                max_gain_series = daily_gain_stats['max_gain'].copy()
                max_gain_series.iloc[current_idx] = max_gain_10d

                rank_percentile = self._calculate_rolling_rank_percentile(
                    max_gain_series, 60, current_idx
                )
                result['max_gain_10d_rank_60d'] = rank_percentile
            else:
                result['max_gain_10d_rank_60d'] = 0.5

            # 计算近5日最高涨幅的30日排名分位数
            if current_idx >= 4:
                start_idx = max(0, current_idx - 4)
                end_idx = current_idx + 1
                max_gain_5d = daily_gain_stats['max_gain'].iloc[start_idx:end_idx].max()

                max_gain_series = daily_gain_stats['max_gain'].copy()
                max_gain_series.iloc[current_idx] = max_gain_5d

                rank_percentile = self._calculate_rolling_rank_percentile(
                    max_gain_series, 30, current_idx
                )
                result['max_gain_5d_rank_30d'] = rank_percentile
            else:
                result['max_gain_5d_rank_30d'] = 0.5

            return result

        except Exception as e:
            self.logger.error(f"计算缓存最高涨幅排名特征失败: {e}")
            return self._get_default_max_gain_features()

    def _calculate_gain_loss_ratio_ranking_features_cached(self, current_date: pd.Timestamp,
                                                         daily_ratio_stats: pd.DataFrame) -> dict:
        """使用预计算统计数据计算涨跌比率排名特征（性能优化版本）"""
        try:
            if daily_ratio_stats.empty:
                return self._get_default_gain_loss_ratio_features()

            # 找到当前日期在统计数据中的位置
            daily_ratio_stats['date_str'] = daily_ratio_stats['date'].dt.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            current_idx = None
            for idx, date_str in enumerate(daily_ratio_stats['date_str']):
                if date_str == current_date_str:
                    current_idx = idx
                    break

            if current_idx is None:
                return self._get_default_gain_loss_ratio_features()

            # 计算各个时间窗口的排名分位数
            windows = [60, 20, 10, 5]
            result = {}

            for window in windows:
                rank_percentile = self._calculate_rolling_rank_percentile(
                    daily_ratio_stats['gain_loss_ratio'], window, current_idx
                )
                result[f'gain_loss_ratio_rank_{window}d'] = rank_percentile

            return result

        except Exception as e:
            self.logger.error(f"计算缓存涨跌比率排名特征失败: {e}")
            return self._get_default_gain_loss_ratio_features()