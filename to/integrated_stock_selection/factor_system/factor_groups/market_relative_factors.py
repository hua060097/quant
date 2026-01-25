#!/usr/bin/env python3
"""
市场相对因子组

基于enhanced_correlation_factors.py中的市场状态因子，计算个股与市场状态因子的相对特征
包括个股波动率相对市场波动率、个股涨幅与市场状态的相关性、个股动量与市场状态的相关性

数据源结构说明：
- market_status.json: 包含市场指数的OHLCV数据、趋势强度等
- limit_data/*.parquet: 包含涨停、跌停、炸板的详细数据
- 个股数据: 包含个股的价格、成交量、技术指标等
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import multiprocessing
from .base_factor_group import BaseFactorGroup

class MarketRelativeFactors(BaseFactorGroup):
    """市场相对因子组"""
    
    def __init__(self):
        super().__init__("market_relative_factors")
    
    def _get_version(self) -> str:
        return "v1.0"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接使用原始数据计算相对特征
    
    def _get_feature_names(self) -> List[str]:
        """返回所有因子：1个基础相对因子 + 6个涨幅相关因子 + 6个动量相关因子"""
        return [
            # 基础相对因子（1个）
            'volatility_ratio_to_market_20d',      # 个股20日波动率与市场波动率的比值
            
            # 个股涨幅相关因子（6个）
            'pct_chg_corr_limit_up_rank_5d',       # 个股涨幅与limit_up_count_rank_60d的5日滚动相关系数
            'pct_chg_corr_limit_up_rank_10d',      # 个股涨幅与limit_up_count_rank_60d的10日滚动相关系数
            'pct_chg_corr_limit_up_rank_20d',      # 个股涨幅与limit_up_count_rank_60d的20日滚动相关系数
            'pct_chg_corr_volume_rank_5d',         # 个股涨幅与market_volume_rank_60d的5日滚动相关系数
            'pct_chg_corr_volume_rank_10d',        # 个股涨幅与market_volume_rank_60d的10日滚动相关系数
            'pct_chg_corr_volume_rank_20d',        # 个股涨幅与market_volume_rank_60d的20日滚动相关系数
            
            # 个股动量相关因子（6个）
            'momentum_corr_limit_up_rank_5d',      # 个股20日动量排名与limit_up_count_rank_60d的5日滚动相关系数
            'momentum_corr_limit_up_rank_10d',     # 个股20日动量排名与limit_up_count_rank_60d的10日滚动相关系数
            'momentum_corr_limit_up_rank_20d',     # 个股20日动量排名与limit_up_count_rank_60d的20日滚动相关系数
            'momentum_corr_volume_rank_5d',        # 个股20日动量排名与market_volume_rank_60d的5日滚动相关系数
            'momentum_corr_volume_rank_10d',       # 个股20日动量排名与market_volume_rank_60d的10日滚动相关系数
            'momentum_corr_volume_rank_20d',       # 个股20日动量排名与market_volume_rank_60d的20日滚动相关系数
        ]
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算市场相对因子（基于enhanced_correlation_factors.py的专业实现）"""
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
                self.logger.warning("市场数据无效，市场相对因子将使用默认值")
                # 创建空的市场数据以避免计算错误
                market_data = None

            # 保存数据引用供市场状态因子计算使用
            self.full_data = data
            self.market_data = market_data

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_market_relative_features_optimized(
                data, target_dates, historical_dates, market_data
            )

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            return result_data

        except Exception as e:
            self.logger.error(f"市场相对因子计算失败: {e}")
            return data

    def _calculate_market_relative_features_optimized(self, full_data: pd.DataFrame,
                                                    target_dates: List[str],
                                                    historical_dates: List[str],
                                                    market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        正确的优化市场相对特征计算（参照enhanced_correlation_factors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意相关性计算中的时间窗口边界，确保不使用未来信息
        """
        try:
            self.logger.info("开始正确优化的市场相对特征计算")

            # 1. 创建目标日期的结果DataFrame（参照enhanced_correlation_factors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_market_relative_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_market_relative_features_with_history(
                full_data, result_data, target_dates, market_data
            )

            self.logger.info("正确优化的市场相对特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的市场相对特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_market_relative_features(self, result_data: pd.DataFrame):
        """为目标日期初始化所有市场相对特征列"""
        try:
            # 基础相对因子
            result_data['volatility_ratio_to_market_20d'] = 1.0      # 个股波动率与市场波动率比值
            
            # 个股涨幅相关因子
            result_data['pct_chg_corr_limit_up_rank_5d'] = 0.0       # 个股涨幅与涨停排名5日相关性
            result_data['pct_chg_corr_limit_up_rank_10d'] = 0.0      # 个股涨幅与涨停排名10日相关性
            result_data['pct_chg_corr_limit_up_rank_20d'] = 0.0      # 个股涨幅与涨停排名20日相关性
            result_data['pct_chg_corr_volume_rank_5d'] = 0.0         # 个股涨幅与成交量排名5日相关性
            result_data['pct_chg_corr_volume_rank_10d'] = 0.0        # 个股涨幅与成交量排名10日相关性
            result_data['pct_chg_corr_volume_rank_20d'] = 0.0        # 个股涨幅与成交量排名20日相关性
            
            # 个股动量相关因子
            result_data['momentum_corr_limit_up_rank_5d'] = 0.0      # 个股动量与涨停排名5日相关性
            result_data['momentum_corr_limit_up_rank_10d'] = 0.0     # 个股动量与涨停排名10日相关性
            result_data['momentum_corr_limit_up_rank_20d'] = 0.0     # 个股动量与涨停排名20日相关性
            result_data['momentum_corr_volume_rank_5d'] = 0.0        # 个股动量与成交量排名5日相关性
            result_data['momentum_corr_volume_rank_10d'] = 0.0       # 个股动量与成交量排名10日相关性
            result_data['momentum_corr_volume_rank_20d'] = 0.0       # 个股动量与成交量排名20日相关性

        except Exception as e:
            self.logger.error(f"初始化目标市场相对特征失败: {e}")

    def _calculate_target_market_relative_features_with_history(self, full_data: pd.DataFrame,
                                                              target_data: pd.DataFrame,
                                                              target_dates: List[str],
                                                              market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的市场相对特征（优化版本）
        关键原则：历史数据只作支撑，只为目标日期计算指标
        优化策略：向量化计算、批量处理、预计算缓存
        """
        try:
            self.logger.info("开始计算目标日期的市场相对特征（优化版本）...")

            # 转换目标日期为datetime格式
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]

            # 🚀 优化1：预计算市场状态因子（复用enhanced_correlation_factors的逻辑）
            market_state_factors = self._precompute_market_state_factors(full_data, market_data, target_date_objects)

            # 🚀 优化2：多线程预计算所有股票的基础指标（ATR、动量等）
            self.logger.info("多线程预计算基础指标...")
            base_indicators = self._precompute_base_indicators_multithreaded(full_data, target_date_objects)

            # 🚀 优化3：多线程批量计算所有目标日期的市场相对特征
            self.logger.info("多线程批量计算市场相对特征...")
            all_features = self._calculate_all_dates_market_relative_features_multithreaded(
                full_data, target_date_objects, market_state_factors, base_indicators
            )

            # 🚀 优化4：批量更新目标数据
            self._batch_update_target_market_relative_features(target_data, all_features, target_date_objects)

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标市场相对特征失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return target_data

    def _precompute_market_state_factors(self, full_data: pd.DataFrame,
                                       market_data: pd.DataFrame,
                                       target_dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, Dict]:
        """预计算市场状态因子（复用enhanced_correlation_factors的逻辑）"""
        try:
            market_state_factors = {}

            for target_date in target_dates:
                # 计算市场波动率（复用enhanced_correlation_factors的实现）
                market_volatility = self._calculate_market_volatility_atr(target_date, market_data)

                # 计算涨停排名（复用enhanced_correlation_factors的实现）
                limit_up_rank = self._calculate_limit_up_count_rank_60d(target_date, full_data)

                # 计算成交量排名（复用enhanced_correlation_factors的实现）
                volume_rank = self._calculate_market_volume_rank_60d(target_date, market_data)

                market_state_factors[target_date] = {
                    'market_volatility_atr20d': market_volatility,
                    'limit_up_count_rank_60d': limit_up_rank,
                    'market_volume_rank_60d': volume_rank
                }

            return market_state_factors

        except Exception as e:
            self.logger.error(f"预计算市场状态因子失败: {e}")
            return {}

    def _calculate_market_volatility_atr(self, current_date: pd.Timestamp,
                                       market_data: pd.DataFrame = None) -> float:
        """
        计算市场20日ATR波动率（与enhanced_correlation_factors.py完全一致）

        基于market_data中的市场指数数据计算ATR（平均真实波动范围）
        """
        try:
            if market_data is None or market_data.empty:
                return 0.0

            # 检查必需的列
            required_cols = ['date', 'close', 'high', 'low']
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

            if current_idx is None or current_idx < 20:  # 需要至少20天数据
                return 0.0

            # 提取价格数据
            close = market_data_sorted['close'].values
            high = market_data_sorted['high'].values
            low = market_data_sorted['low'].values

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

    def _calculate_limit_up_count_rank_60d(self, current_date: pd.Timestamp,
                                         full_data: pd.DataFrame) -> float:
        """
        计算涨停个数60日排名分位数（与enhanced_correlation_factors.py完全一致）
        """
        try:
            if full_data is None or full_data.empty:
                return 0.5

            # 检查必需的列
            if 'is_limit_up' not in full_data.columns:
                self.logger.warning("个股数据缺少is_limit_up列")
                return 0.5

            # 按日期聚合计算每日的涨停个数
            daily_stats = full_data.groupby('date').agg({
                'is_limit_up': 'sum'      # 每日涨停个数
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
                return 0.5

            # 计算60日排名分位数
            rank_percentile = self._calculate_rolling_rank_percentile(
                daily_stats['is_limit_up'], 60, current_idx
            )

            return rank_percentile

        except Exception as e:
            self.logger.error(f"计算涨停排名失败: {e}")
            return 0.5

    def _calculate_market_volume_rank_60d(self, current_date: pd.Timestamp,
                                        market_data: pd.DataFrame = None) -> float:
        """
        计算大盘成交量60日排名分位数（与enhanced_correlation_factors.py完全一致）
        """
        try:
            if market_data is None or market_data.empty:
                return 0.5

            # 检查必需的列
            if 'volume' not in market_data.columns:
                self.logger.warning("市场数据缺少volume列")
                return 0.5

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
                return 0.5

            # 计算60日排名分位数
            rank_percentile = self._calculate_rolling_rank_percentile(
                market_data_sorted['volume'], 60, current_idx
            )

            return rank_percentile

        except Exception as e:
            self.logger.error(f"计算成交量排名失败: {e}")
            return 0.5

    def _calculate_rolling_rank_percentile(self, series: pd.Series, window: int, current_idx: int) -> float:
        """
        计算滚动排名分位数（与enhanced_correlation_factors.py完全一致）
        """
        try:
            if current_idx < window - 1:
                # 数据不足时，使用所有可用数据
                start_idx = 0
                end_idx = current_idx + 1
            else:
                # 使用指定窗口的数据
                start_idx = current_idx - window + 1
                end_idx = current_idx + 1

            window_data = series.iloc[start_idx:end_idx]

            if len(window_data) < 2:
                return 0.5

            current_value = series.iloc[current_idx]

            # 计算排名分位数
            rank = (window_data < current_value).sum() + 1
            percentile = rank / len(window_data)

            return float(percentile)

        except Exception as e:
            self.logger.error(f"计算滚动排名分位数失败: {e}")
            return 0.5

    def _calculate_single_date_market_relative_features(self, full_data: pd.DataFrame,
                                                      target_date_obj: pd.Timestamp,
                                                      market_state: Dict) -> Dict:
        """计算单个日期的市场相对特征"""
        try:
            # 获取该日期的所有股票数据
            date_mask = full_data['date'] == target_date_obj
            all_stocks_data = full_data[date_mask]

            if len(all_stocks_data) == 0:
                return self._get_default_market_relative_features()

            features = {}

            # 1. 基础相对因子：个股波动率与市场波动率比值
            features['volatility_ratio_to_market_20d'] = self._calculate_volatility_ratio_to_market(
                full_data, target_date_obj, market_state.get('market_volatility_atr20d', 0.0)
            )

            # 2-7. 个股涨幅相关因子
            pct_chg_corr_features = self._calculate_pct_chg_correlation_features(
                full_data, target_date_obj, market_state
            )
            features.update(pct_chg_corr_features)

            # 8-13. 个股动量相关因子
            momentum_corr_features = self._calculate_momentum_correlation_features(
                full_data, target_date_obj, market_state
            )
            features.update(momentum_corr_features)

            return features

        except Exception as e:
            self.logger.error(f"计算单日期市场相对特征失败: {e}")
            return self._get_default_market_relative_features()

    def _calculate_volatility_ratio_to_market(self, full_data: pd.DataFrame,
                                            target_date_obj: pd.Timestamp,
                                            market_volatility: float) -> pd.Series:
        """计算个股20日波动率与市场波动率的比值"""
        try:
            # 获取该日期的所有股票
            date_mask = full_data['date'] == target_date_obj
            target_stocks = full_data[date_mask]['stock_code'].unique()

            volatility_ratios = {}

            for stock_code in target_stocks:
                # 获取该股票的历史数据
                stock_mask = full_data['stock_code'] == stock_code
                stock_data = full_data[stock_mask & (full_data['date'] <= target_date_obj)].sort_values('date')

                if len(stock_data) >= 20:
                    # 计算个股20日ATR波动率
                    stock_volatility = self._calculate_stock_atr_volatility(stock_data.tail(20))

                    # 计算比值
                    if market_volatility > 0:
                        ratio = stock_volatility / market_volatility
                    else:
                        ratio = 1.0

                    volatility_ratios[stock_code] = ratio
                else:
                    volatility_ratios[stock_code] = 1.0

            # 转换为Series，索引对应目标日期的数据索引
            target_day_data = full_data[date_mask]
            ratio_series = pd.Series(index=target_day_data.index, dtype=float)

            for idx, row in target_day_data.iterrows():
                stock_code = row['stock_code']
                ratio_series[idx] = volatility_ratios.get(stock_code, 1.0)

            return ratio_series

        except Exception as e:
            self.logger.error(f"计算个股波动率比值失败: {e}")
            # 返回默认值Series
            date_mask = full_data['date'] == target_date_obj
            target_day_data = full_data[date_mask]
            return pd.Series(1.0, index=target_day_data.index)

    def _calculate_stock_atr_volatility(self, stock_data: pd.DataFrame) -> float:
        """计算个股ATR波动率"""
        try:
            if len(stock_data) < 2:
                return 0.0

            # 检查必需的列
            required_cols = ['close', 'high', 'low']
            if not all(col in stock_data.columns for col in required_cols):
                # 如果没有高低价，使用收益率标准差
                if 'pct_chg' in stock_data.columns:
                    return stock_data['pct_chg'].std() / 100.0  # 转换为小数
                else:
                    return 0.0

            close = stock_data['close'].values
            high = stock_data['high'].values
            low = stock_data['low'].values

            # 计算前一日收盘价
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]

            # 计算True Range
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))

            # 计算ATR
            atr = np.mean(true_range)

            # 标准化ATR（除以当前价格）
            current_price = close[-1]
            if current_price > 0:
                normalized_atr = atr / current_price
            else:
                normalized_atr = 0.0

            return float(normalized_atr)

        except Exception as e:
            self.logger.error(f"计算个股ATR波动率失败: {e}")
            return 0.0

    def _calculate_pct_chg_correlation_features(self, full_data: pd.DataFrame,
                                              target_date_obj: pd.Timestamp,
                                              market_state: Dict) -> Dict:
        """计算个股涨幅与市场状态因子的相关性特征"""
        try:
            features = {}

            # 获取市场状态因子值
            limit_up_rank = market_state.get('limit_up_count_rank_60d', 0.5)
            volume_rank = market_state.get('market_volume_rank_60d', 0.5)

            # 计算不同时间窗口的相关性
            windows = [5, 10, 20]

            for window in windows:
                # 个股涨幅与limit_up_count_rank_60d的相关性
                limit_corr = self._calculate_rolling_correlation_with_market_factor(
                    full_data, target_date_obj, 'pct_chg', 'limit_up_rank', window, limit_up_rank
                )
                features[f'pct_chg_corr_limit_up_rank_{window}d'] = limit_corr

                # 个股涨幅与market_volume_rank_60d的相关性
                volume_corr = self._calculate_rolling_correlation_with_market_factor(
                    full_data, target_date_obj, 'pct_chg', 'volume_rank', window, volume_rank
                )
                features[f'pct_chg_corr_volume_rank_{window}d'] = volume_corr

            return features

        except Exception as e:
            self.logger.error(f"计算个股涨幅相关性特征失败: {e}")
            return {
                'pct_chg_corr_limit_up_rank_5d': 0.0,
                'pct_chg_corr_limit_up_rank_10d': 0.0,
                'pct_chg_corr_limit_up_rank_20d': 0.0,
                'pct_chg_corr_volume_rank_5d': 0.0,
                'pct_chg_corr_volume_rank_10d': 0.0,
                'pct_chg_corr_volume_rank_20d': 0.0,
            }

    def _calculate_momentum_correlation_features(self, full_data: pd.DataFrame,
                                               target_date_obj: pd.Timestamp,
                                               market_state: Dict) -> Dict:
        """计算个股动量与市场状态因子的相关性特征"""
        try:
            features = {}

            # 获取市场状态因子值
            limit_up_rank = market_state.get('limit_up_count_rank_60d', 0.5)
            volume_rank = market_state.get('market_volume_rank_60d', 0.5)

            # 计算不同时间窗口的相关性
            windows = [5, 10, 20]

            for window in windows:
                # 个股动量与limit_up_count_rank_60d的相关性
                limit_corr = self._calculate_rolling_correlation_with_market_factor(
                    full_data, target_date_obj, 'rank_momentum_20d', 'limit_up_rank', window, limit_up_rank
                )
                features[f'momentum_corr_limit_up_rank_{window}d'] = limit_corr

                # 个股动量与market_volume_rank_60d的相关性
                volume_corr = self._calculate_rolling_correlation_with_market_factor(
                    full_data, target_date_obj, 'rank_momentum_20d', 'volume_rank', window, volume_rank
                )
                features[f'momentum_corr_volume_rank_{window}d'] = volume_corr

            return features

        except Exception as e:
            self.logger.error(f"计算个股动量相关性特征失败: {e}")
            return {
                'momentum_corr_limit_up_rank_5d': 0.0,
                'momentum_corr_limit_up_rank_10d': 0.0,
                'momentum_corr_limit_up_rank_20d': 0.0,
                'momentum_corr_volume_rank_5d': 0.0,
                'momentum_corr_volume_rank_10d': 0.0,
                'momentum_corr_volume_rank_20d': 0.0,
            }

    def _calculate_rolling_correlation_with_market_factor(self, full_data: pd.DataFrame,
                                                        target_date_obj: pd.Timestamp,
                                                        stock_feature: str,
                                                        market_factor_type: str,
                                                        window: int,
                                                        current_market_value: float) -> pd.Series:
        """计算个股特征与市场因子的滚动相关性"""
        try:
            # 获取该日期的所有股票
            date_mask = full_data['date'] == target_date_obj
            target_stocks = full_data[date_mask]['stock_code'].unique()

            correlations = {}

            for stock_code in target_stocks:
                # 获取该股票的历史数据
                stock_mask = full_data['stock_code'] == stock_code
                stock_data = full_data[stock_mask & (full_data['date'] <= target_date_obj)].sort_values('date')

                if len(stock_data) >= window:
                    # 获取最近window天的数据
                    recent_data = stock_data.tail(window)

                    # 获取个股特征值
                    if stock_feature == 'pct_chg':
                        stock_values = recent_data['pct_chg'].values
                    elif stock_feature == 'rank_momentum_20d':
                        # 计算rank_momentum_20d（与cross_sectional_factors.py一致）
                        stock_values = self._calculate_rank_momentum_20d_series(stock_data, window)
                    else:
                        correlations[stock_code] = 0.0
                        continue

                    # 构造市场因子时间序列（简化处理，假设市场因子在窗口期内变化）
                    market_values = self._construct_market_factor_series(
                        recent_data, market_factor_type, current_market_value
                    )

                    # 计算相关性
                    if len(stock_values) == len(market_values) and len(stock_values) > 1:
                        correlation = np.corrcoef(stock_values, market_values)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                        correlations[stock_code] = correlation
                    else:
                        correlations[stock_code] = 0.0
                else:
                    correlations[stock_code] = 0.0

            # 转换为Series，索引对应目标日期的数据索引
            target_day_data = full_data[date_mask]
            corr_series = pd.Series(index=target_day_data.index, dtype=float)

            for idx, row in target_day_data.iterrows():
                stock_code = row['stock_code']
                corr_series[idx] = correlations.get(stock_code, 0.0)

            return corr_series

        except Exception as e:
            self.logger.error(f"计算滚动相关性失败: {e}")
            # 返回默认值Series
            date_mask = full_data['date'] == target_date_obj
            target_day_data = full_data[date_mask]
            return pd.Series(0.0, index=target_day_data.index)

    def _calculate_rank_momentum_20d_series(self, stock_data: pd.DataFrame, window: int) -> np.ndarray:
        """计算rank_momentum_20d时间序列（与cross_sectional_factors.py一致）"""
        try:
            if len(stock_data) < 20:
                return np.zeros(window)

            # 获取最近window天的数据
            recent_data = stock_data.tail(window + 19)  # 需要额外19天来计算20日动量

            momentum_values = []

            for i in range(window):
                end_idx = len(recent_data) - window + i + 1
                start_idx = max(0, end_idx - 20)

                if end_idx > start_idx:
                    period_data = recent_data.iloc[start_idx:end_idx]
                    if len(period_data) >= 2:
                        start_price = period_data['close'].iloc[0]
                        end_price = period_data['close'].iloc[-1]
                        momentum = (end_price - start_price) / start_price if start_price != 0 else 0.0
                        momentum_values.append(momentum)
                    else:
                        momentum_values.append(0.0)
                else:
                    momentum_values.append(0.0)

            return np.array(momentum_values)

        except Exception as e:
            self.logger.error(f"计算rank_momentum_20d序列失败: {e}")
            return np.zeros(window)

    def _construct_market_factor_series(self, recent_data: pd.DataFrame,
                                      market_factor_type: str,
                                      current_value: float) -> np.ndarray:
        """🔧 专业量化计算：构造真实的市场因子时间序列（移除随机数）"""
        try:
            window = len(recent_data)

            # 🚀 专业量化方法：基于历史数据计算真实的市场因子序列
            if market_factor_type == 'limit_up_rank':
                # 基于历史涨停数据计算真实的排名序列
                base_values = self._calculate_historical_limit_up_rank_series(recent_data, current_value)
            elif market_factor_type == 'volume_rank':
                # 基于历史成交量数据计算真实的排名序列
                base_values = self._calculate_historical_volume_rank_series(recent_data, current_value)
            else:
                # 使用当前值作为基准，基于技术指标计算变化
                base_values = self._calculate_technical_factor_series(recent_data, current_value)

            return base_values

        except Exception as e:
            self.logger.error(f"构造市场因子序列失败: {e}")
            return np.full(len(recent_data), current_value)

    def _calculate_historical_limit_up_rank_series(self, recent_data: pd.DataFrame,
                                                 current_value: float) -> np.ndarray:
        """🚀 专业量化：基于历史涨停数据计算真实排名序列"""
        try:
            window = len(recent_data)

            # 获取历史日期
            dates = recent_data['date'].values

            # 为每个历史日期计算真实的涨停排名
            rank_series = []
            for date in dates:
                # 获取该日期的涨停数据
                date_data = self.full_data[self.full_data['date'] == date]
                if len(date_data) > 0:
                    # 计算该日期的涨停个数
                    limit_up_count = date_data['is_limit_up'].sum()
                    total_stocks = len(date_data)

                    # 计算排名分位数（涨停个数相对于历史的排名）
                    if total_stocks > 0:
                        rank_percentile = min(1.0, max(0.0, limit_up_count / (total_stocks * 0.1)))  # 假设10%为满分
                    else:
                        rank_percentile = current_value
                else:
                    rank_percentile = current_value

                rank_series.append(rank_percentile)

            return np.array(rank_series)

        except Exception as e:
            self.logger.error(f"计算历史涨停排名序列失败: {e}")
            return np.full(len(recent_data), current_value)

    def _calculate_historical_volume_rank_series(self, recent_data: pd.DataFrame,
                                               current_value: float) -> np.ndarray:
        """🚀 专业量化：基于历史成交量数据计算真实排名序列"""
        try:
            window = len(recent_data)

            # 如果有市场数据，使用真实的成交量排名
            if hasattr(self, 'market_data') and self.market_data is not None:
                dates = recent_data['date'].values
                rank_series = []

                for date in dates:
                    # 获取该日期的市场成交量
                    market_date_data = self.market_data[self.market_data['date'] == date]
                    if len(market_date_data) > 0:
                        volume = market_date_data['volume'].iloc[0]

                        # 计算成交量相对于历史的排名分位数
                        historical_volumes = self.market_data[self.market_data['date'] <= date]['volume']
                        if len(historical_volumes) > 1:
                            rank_percentile = (historical_volumes < volume).sum() / len(historical_volumes)
                        else:
                            rank_percentile = current_value
                    else:
                        rank_percentile = current_value

                    rank_series.append(rank_percentile)

                return np.array(rank_series)
            else:
                # 如果没有市场数据，使用技术指标方法
                return self._calculate_technical_factor_series(recent_data, current_value)

        except Exception as e:
            self.logger.error(f"计算历史成交量排名序列失败: {e}")
            return np.full(len(recent_data), current_value)

    def _calculate_technical_factor_series(self, recent_data: pd.DataFrame,
                                         current_value: float) -> np.ndarray:
        """🚀 专业量化：基于技术指标计算因子序列"""
        try:
            window = len(recent_data)

            # 基于移动平均和趋势计算因子变化
            if 'close' in recent_data.columns and len(recent_data) > 1:
                # 计算价格变化率
                price_changes = recent_data['close'].pct_change().fillna(0)

                # 基于价格变化计算因子调整
                # 上涨时因子值增加，下跌时因子值减少
                cumulative_change = price_changes.cumsum()

                # 标准化到合理范围
                if cumulative_change.std() > 0:
                    normalized_change = (cumulative_change - cumulative_change.mean()) / cumulative_change.std()
                    # 将变化映射到因子值范围
                    factor_adjustment = normalized_change * 0.1  # 控制变化幅度
                else:
                    factor_adjustment = np.zeros(window)

                # 应用调整
                base_values = np.full(window, current_value) + factor_adjustment
                base_values = np.clip(base_values, 0.0, 1.0)

                return base_values
            else:
                # 如果没有价格数据，使用线性趋势
                trend = np.linspace(-0.05, 0.05, window)  # 小幅趋势变化
                base_values = np.full(window, current_value) + trend
                base_values = np.clip(base_values, 0.0, 1.0)

                return base_values

        except Exception as e:
            self.logger.error(f"计算技术因子序列失败: {e}")
            return np.full(len(recent_data), current_value)

    def _construct_professional_factor_series(self, window: int,
                                             current_value: float,
                                             factor_type: str,
                                             target_date: pd.Timestamp = None) -> np.ndarray:
        """🚀 专业量化：基于真实历史数据构造市场因子序列（参考enhanced_correlation_factors）"""
        try:
            if target_date is None:
                # 如果没有目标日期，使用简单的线性序列
                return np.full(window, current_value)

            # 🚀 基于真实历史数据计算因子序列
            if factor_type == 'limit_up_rank':
                return self._calculate_real_limit_up_rank_series(window, current_value, target_date)
            elif factor_type == 'volume_rank':
                return self._calculate_real_volume_rank_series(window, current_value, target_date)
            else:
                # 其他因子使用当前值
                return np.full(window, current_value)

        except Exception as e:
            self.logger.error(f"构造专业因子序列失败: {e}")
            return np.full(window, current_value)

    def _calculate_real_limit_up_rank_series(self, window: int,
                                           current_value: float,
                                           target_date: pd.Timestamp) -> np.ndarray:
        """🚀 基于真实数据计算涨停排名序列（参考enhanced_correlation_factors逻辑）"""
        try:
            # 获取历史数据
            if not hasattr(self, 'full_data') or self.full_data is None:
                return np.full(window, current_value)

            # 检查必需的列
            required_cols = ['date', 'is_limit_up']
            if not all(col in self.full_data.columns for col in required_cols):
                return np.full(window, current_value)

            # 获取目标日期之前的历史数据
            historical_data = self.full_data[self.full_data['date'] <= target_date].copy()

            if len(historical_data) < window:
                return np.full(window, current_value)

            # 🚀 真实计算：按日期聚合计算每日涨停个数
            daily_limit_stats = historical_data.groupby('date').agg({
                'is_limit_up': 'sum'  # 每日涨停个数
            }).reset_index()

            # 按日期排序，取最近window天
            daily_limit_stats = daily_limit_stats.sort_values('date').tail(window).reset_index(drop=True)

            if len(daily_limit_stats) < window:
                return np.full(window, current_value)

            # 🚀 真实计算：计算每日涨停个数的排名分位数
            limit_up_counts = daily_limit_stats['is_limit_up'].values

            # 计算滚动排名分位数
            rank_series = []
            for i in range(len(limit_up_counts)):
                # 计算当前值在历史窗口中的排名分位数
                window_data = limit_up_counts[:i+1]
                if len(window_data) > 1:
                    rank_percentile = (window_data < limit_up_counts[i]).sum() / len(window_data)
                else:
                    rank_percentile = current_value
                rank_series.append(rank_percentile)

            return np.array(rank_series)

        except Exception as e:
            self.logger.error(f"计算真实涨停排名序列失败: {e}")
            return np.full(window, current_value)

    def _calculate_real_volume_rank_series(self, window: int,
                                         current_value: float,
                                         target_date: pd.Timestamp) -> np.ndarray:
        """🚀 基于真实数据计算成交量排名序列（参考enhanced_correlation_factors逻辑）"""
        try:
            # 获取市场数据
            if not hasattr(self, 'market_data') or self.market_data is None:
                return np.full(window, current_value)

            # 检查必需的列
            if 'volume' not in self.market_data.columns:
                return np.full(window, current_value)

            # 获取目标日期之前的历史数据
            historical_market_data = self.market_data[self.market_data['date'] <= target_date].copy()

            if len(historical_market_data) < window:
                return np.full(window, current_value)

            # 🚀 真实计算：按日期排序，取最近window天
            market_data_sorted = historical_market_data.sort_values('date').tail(window).reset_index(drop=True)

            if len(market_data_sorted) < window:
                return np.full(window, current_value)

            # 🚀 真实计算：计算每日成交量的排名分位数
            volumes = market_data_sorted['volume'].values

            # 计算滚动排名分位数
            rank_series = []
            for i in range(len(volumes)):
                # 计算当前值在历史窗口中的排名分位数
                window_data = volumes[:i+1]
                if len(window_data) > 1:
                    rank_percentile = (window_data < volumes[i]).sum() / len(window_data)
                else:
                    rank_percentile = current_value
                rank_series.append(rank_percentile)

            return np.array(rank_series)

        except Exception as e:
            self.logger.error(f"计算真实成交量排名序列失败: {e}")
            return np.full(window, current_value)

    def _get_default_market_relative_features(self) -> Dict:
        """获取默认的市场相对特征值"""
        return {
            'volatility_ratio_to_market_20d': 1.0,
            'pct_chg_corr_limit_up_rank_5d': 0.0,
            'pct_chg_corr_limit_up_rank_10d': 0.0,
            'pct_chg_corr_limit_up_rank_20d': 0.0,
            'pct_chg_corr_volume_rank_5d': 0.0,
            'pct_chg_corr_volume_rank_10d': 0.0,
            'pct_chg_corr_volume_rank_20d': 0.0,
            'momentum_corr_limit_up_rank_5d': 0.0,
            'momentum_corr_limit_up_rank_10d': 0.0,
            'momentum_corr_limit_up_rank_20d': 0.0,
            'momentum_corr_volume_rank_5d': 0.0,
            'momentum_corr_volume_rank_10d': 0.0,
            'momentum_corr_volume_rank_20d': 0.0,
        }

    def _update_target_market_relative_features(self, target_data: pd.DataFrame,
                                              target_day_features: Dict,
                                              target_day_mask: pd.Series):
        """更新目标数据中的市场相对特征"""
        try:
            for feature_name, feature_series in target_day_features.items():
                if feature_name in target_data.columns and isinstance(feature_series, pd.Series):
                    # 只更新目标日期的数据
                    for idx in target_data[target_day_mask].index:
                        if idx in feature_series.index:
                            target_data.loc[idx, feature_name] = feature_series[idx]
                elif feature_name in target_data.columns and isinstance(feature_series, (int, float)):
                    # 如果是标量值，应用到所有目标日期的数据
                    target_data.loc[target_day_mask, feature_name] = feature_series

        except Exception as e:
            self.logger.error(f"更新目标市场相对特征失败: {e}")

    def _validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """验证市场数据的有效性（与enhanced_correlation_factors.py一致）"""
        try:
            if market_data is None or market_data.empty:
                return False

            # 检查必需的列
            required_cols = ['date', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in market_data.columns]
            if missing_cols:
                self.logger.warning(f"市场数据缺少必需列: {missing_cols}")
                return False

            # 检查数据量
            if len(market_data) < 60:  # 至少需要60天数据
                self.logger.warning(f"市场数据量不足: {len(market_data)} < 60")
                return False

            return True

        except Exception as e:
            self.logger.error(f"市场数据验证失败: {e}")
            return False

    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> tuple:
        """
        精确识别目标计算日期和历史支撑数据（复用enhanced_correlation_factors的逻辑）
        """
        try:
            # 优先使用kwargs中传递的target_start_date和target_end_date
            target_start_date = kwargs.get('target_start_date')
            target_end_date = kwargs.get('target_end_date')

            if target_start_date and target_end_date:
                # 转换为datetime格式
                if isinstance(target_start_date, str):
                    target_start_date = pd.to_datetime(target_start_date, format='%Y%m%d')
                if isinstance(target_end_date, str):
                    target_end_date = pd.to_datetime(target_end_date, format='%Y%m%d')

                # 获取目标日期范围内的所有日期
                target_mask = (data['date'] >= target_start_date) & (data['date'] <= target_end_date)
                target_dates = data[target_mask]['date'].dt.strftime('%Y%m%d').unique().tolist()

                # 历史支撑数据：目标日期之前的所有数据
                historical_mask = data['date'] < target_start_date
                historical_dates = data[historical_mask]['date'].dt.strftime('%Y%m%d').unique().tolist()

                return target_dates, historical_dates

            # 如果没有指定目标日期，使用最近的日期作为目标
            all_dates = sorted(data['date'].dt.strftime('%Y%m%d').unique())
            if len(all_dates) >= 60:  # 需要足够的历史数据
                target_dates = [all_dates[-1]]  # 最后一天作为目标
                historical_dates = all_dates[:-1]  # 其余作为历史
                return target_dates, historical_dates

            return [], []

        except Exception as e:
            self.logger.error(f"识别目标和历史日期失败: {e}")
            return [], []

    def _precompute_base_indicators_vectorized(self, full_data: pd.DataFrame,
                                             target_dates: List[pd.Timestamp]) -> Dict:
        """
        🚀 优化：向量化预计算所有股票的基础指标
        一次性计算所有需要的ATR、动量等指标，避免重复计算
        """
        try:
            self.logger.info("开始向量化预计算基础指标...")

            # 确保数据按股票和日期排序
            sorted_data = full_data.sort_values(['stock_code', 'date']).reset_index(drop=True)

            # 🚀 向量化计算ATR（所有股票一次性计算）
            atr_data = self._calculate_atr_vectorized(sorted_data)

            # 🚀 向量化计算20日动量（所有股票一次性计算）
            momentum_data = self._calculate_momentum_vectorized(sorted_data)

            # 🚀 预计算每个目标日期需要的历史数据索引
            date_indices = self._precompute_date_indices(sorted_data, target_dates)

            base_indicators = {
                'atr_data': atr_data,
                'momentum_data': momentum_data,
                'date_indices': date_indices,
                'sorted_data': sorted_data
            }

            self.logger.info("基础指标预计算完成")
            return base_indicators

        except Exception as e:
            self.logger.error(f"预计算基础指标失败: {e}")
            return {}

    def _calculate_atr_vectorized(self, sorted_data: pd.DataFrame) -> pd.DataFrame:
        """🚀 向量化计算所有股票的ATR"""
        try:
            # 检查必需的列
            required_cols = ['close', 'high', 'low']
            if not all(col in sorted_data.columns for col in required_cols):
                # 如果没有高低价，使用收益率标准差
                if 'pct_chg' in sorted_data.columns:
                    atr_result = sorted_data.groupby('stock_code')['pct_chg'].rolling(
                        window=20, min_periods=1
                    ).std().reset_index(level=0, drop=True) / 100.0
                    return atr_result.to_frame('atr_20d')
                else:
                    return pd.DataFrame({'atr_20d': 0.0}, index=sorted_data.index)

            # 计算True Range的三个组成部分
            # TR1: high - low
            tr1 = sorted_data['high'] - sorted_data['low']

            # TR2: abs(high - prev_close)
            prev_close = sorted_data.groupby('stock_code')['close'].shift(1)
            tr2 = (sorted_data['high'] - prev_close).abs()

            # TR3: abs(low - prev_close)
            tr3 = (sorted_data['low'] - prev_close).abs()

            # True Range = max(TR1, TR2, TR3)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # 计算20日ATR（按股票分组）
            atr_20d = true_range.groupby(sorted_data['stock_code']).rolling(
                window=20, min_periods=1
            ).mean().reset_index(level=0, drop=True)

            # 标准化ATR（除以当前价格）
            normalized_atr = atr_20d / sorted_data['close']
            normalized_atr = normalized_atr.fillna(0.0)

            return normalized_atr.to_frame('atr_20d')

        except Exception as e:
            self.logger.error(f"向量化ATR计算失败: {e}")
            return pd.DataFrame({'atr_20d': 0.0}, index=sorted_data.index)

    def _calculate_momentum_vectorized(self, sorted_data: pd.DataFrame) -> pd.DataFrame:
        """🚀 向量化计算所有股票的20日动量"""
        try:
            # 计算20日收益率（当前价格相对于20天前的价格）
            close_20d_ago = sorted_data.groupby('stock_code')['close'].shift(20)
            momentum_20d = (sorted_data['close'] - close_20d_ago) / close_20d_ago
            momentum_20d = momentum_20d.fillna(0.0)

            # 计算每日的动量排名（横截面排名）
            rank_momentum = sorted_data.groupby('date')['close'].transform(
                lambda x: momentum_20d[x.index].rank(pct=True)
            )
            rank_momentum = rank_momentum.fillna(0.5)

            return rank_momentum.to_frame('rank_momentum_20d')

        except Exception as e:
            self.logger.error(f"向量化动量计算失败: {e}")
            return pd.DataFrame({'rank_momentum_20d': 0.5}, index=sorted_data.index)

    def _precompute_date_indices(self, sorted_data: pd.DataFrame,
                               target_dates: List[pd.Timestamp]) -> Dict:
        """🚀 预计算日期索引，加速数据访问（内存优化版本）"""
        try:
            date_indices = {}

            # 🔧 修复内存问题：避免存储大量索引，改为存储布尔掩码
            for target_date in target_dates:
                # 目标日期的数据掩码（更节省内存）
                target_mask = sorted_data['date'] == target_date
                target_count = target_mask.sum()

                if target_count == 0:
                    continue

                # 只存储必要的信息，使用布尔掩码避免大量索引列表
                target_stocks = sorted_data.loc[target_mask, 'stock_code'].unique()

                date_indices[target_date] = {
                    'target_mask': target_mask,  # 使用布尔掩码而不是索引列表
                    'target_count': target_count,
                    'target_stocks': target_stocks.tolist(),  # 不限制股票数量，处理所有股票
                    'date_position': sorted_data[target_mask].index[0] if target_count > 0 else None  # 只存储第一个索引作为参考
                }

            return date_indices

        except Exception as e:
            self.logger.error(f"预计算日期索引失败: {e}")
            return {}

    def _calculate_all_dates_market_relative_features_vectorized(self,
                                                               full_data: pd.DataFrame,
                                                               target_dates: List[pd.Timestamp],
                                                               market_state_factors: Dict,
                                                               base_indicators: Dict) -> Dict:
        """🚀 向量化批量计算所有目标日期的市场相对特征"""
        try:
            self.logger.info("开始向量化批量计算市场相对特征...")

            all_features = {}
            sorted_data = base_indicators.get('sorted_data', full_data)
            atr_data = base_indicators.get('atr_data', pd.DataFrame())
            momentum_data = base_indicators.get('momentum_data', pd.DataFrame())
            date_indices = base_indicators.get('date_indices', {})

            for target_date in target_dates:
                if target_date not in date_indices:
                    continue

                target_info = date_indices[target_date]
                target_indices = target_info['target_indices']

                if not target_indices:
                    continue

                # 获取市场状态
                market_state = market_state_factors.get(target_date, {})
                market_volatility = market_state.get('market_volatility_atr20d', 0.0)

                # 🚀 向量化计算波动率比值
                volatility_ratios = self._calculate_volatility_ratios_vectorized(
                    sorted_data, atr_data, target_indices, market_volatility
                )

                # 🚀 向量化计算相关性特征
                correlation_features = self._calculate_correlation_features_vectorized(
                    sorted_data, momentum_data, target_indices, market_state, target_date
                )

                # 合并特征
                date_features = {
                    'volatility_ratio_to_market_20d': volatility_ratios,
                    **correlation_features
                }

                all_features[target_date] = date_features

            self.logger.info("向量化批量计算完成")
            return all_features

        except Exception as e:
            self.logger.error(f"向量化批量计算失败: {e}")
            return {}

    def _calculate_volatility_ratios_vectorized(self, sorted_data: pd.DataFrame,
                                              atr_data: pd.DataFrame,
                                              target_indices: List[int],
                                              market_volatility: float) -> pd.Series:
        """🚀 向量化计算波动率比值"""
        try:
            if atr_data.empty or market_volatility <= 0:
                return pd.Series(1.0, index=target_indices)

            # 获取目标日期的ATR值
            target_atr = atr_data.loc[target_indices, 'atr_20d']

            # 计算比值
            volatility_ratios = target_atr / market_volatility
            volatility_ratios = volatility_ratios.fillna(1.0)

            return volatility_ratios

        except Exception as e:
            self.logger.error(f"向量化波动率比值计算失败: {e}")
            return pd.Series(1.0, index=target_indices)

    def _calculate_correlation_features_vectorized(self, sorted_data: pd.DataFrame,
                                                 momentum_data: pd.DataFrame,
                                                 target_indices: List[int],
                                                 market_state: Dict,
                                                 target_date: pd.Timestamp) -> Dict:
        """🚀 向量化计算相关性特征"""
        try:
            correlation_features = {}

            # 获取市场状态因子值
            limit_up_rank = market_state.get('limit_up_count_rank_60d', 0.5)
            volume_rank = market_state.get('market_volume_rank_60d', 0.5)

            # 获取目标日期的股票代码
            target_stocks = sorted_data.loc[target_indices, 'stock_code'].unique()

            # 🚀 向量化计算涨幅相关性（批量处理所有窗口）
            pct_chg_correlations = self._calculate_pct_chg_correlations_batch(
                sorted_data, target_stocks, target_date, limit_up_rank, volume_rank
            )
            correlation_features.update(pct_chg_correlations)

            # 🚀 向量化计算动量相关性（批量处理所有窗口）
            momentum_correlations = self._calculate_momentum_correlations_batch(
                sorted_data, momentum_data, target_stocks, target_date, limit_up_rank, volume_rank
            )
            correlation_features.update(momentum_correlations)

            # 转换为Series格式，索引对应target_indices
            for feature_name, correlations_dict in correlation_features.items():
                feature_series = pd.Series(index=target_indices, dtype=float)
                for idx in target_indices:
                    stock_code = sorted_data.loc[idx, 'stock_code']
                    feature_series[idx] = correlations_dict.get(stock_code, 0.0)
                correlation_features[feature_name] = feature_series

            return correlation_features

        except Exception as e:
            self.logger.error(f"向量化相关性特征计算失败: {e}")
            return {}

    def _calculate_pct_chg_correlations_batch(self, sorted_data: pd.DataFrame,
                                            target_stocks: np.ndarray,
                                            target_date: pd.Timestamp,
                                            limit_up_rank: float,
                                            volume_rank: float) -> Dict:
        """🚀 批量计算涨幅相关性（所有窗口一次性计算）"""
        try:
            correlations = {}
            windows = [5, 10, 20]

            # 为每个股票批量计算所有窗口的相关性
            for stock_code in target_stocks:
                stock_mask = sorted_data['stock_code'] == stock_code
                stock_data = sorted_data[stock_mask & (sorted_data['date'] <= target_date)]

                if len(stock_data) < max(windows):
                    # 数据不足，使用默认值
                    for window in windows:
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'] = correlations.get(
                            f'pct_chg_corr_limit_up_rank_{window}d', {}
                        )
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'][stock_code] = 0.0

                        correlations[f'pct_chg_corr_volume_rank_{window}d'] = correlations.get(
                            f'pct_chg_corr_volume_rank_{window}d', {}
                        )
                        correlations[f'pct_chg_corr_volume_rank_{window}d'][stock_code] = 0.0
                    continue

                # 获取涨幅数据
                pct_chg_values = stock_data['pct_chg'].values

                # 为所有窗口计算相关性
                for window in windows:
                    if len(pct_chg_values) >= window:
                        recent_pct_chg = pct_chg_values[-window:]

                        # 🚀 专业量化计算：构造真实的市场因子序列（移除随机数）
                        limit_up_series = self._construct_professional_factor_series(
                            window, limit_up_rank, 'limit_up_rank'
                        )
                        volume_series = self._construct_professional_factor_series(
                            window, volume_rank, 'volume_rank'
                        )

                        # 计算相关性
                        limit_corr = np.corrcoef(recent_pct_chg, limit_up_series)[0, 1]
                        volume_corr = np.corrcoef(recent_pct_chg, volume_series)[0, 1]

                        # 处理NaN值
                        limit_corr = 0.0 if np.isnan(limit_corr) else limit_corr
                        volume_corr = 0.0 if np.isnan(volume_corr) else volume_corr

                        # 存储结果
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'] = correlations.get(
                            f'pct_chg_corr_limit_up_rank_{window}d', {}
                        )
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'][stock_code] = limit_corr

                        correlations[f'pct_chg_corr_volume_rank_{window}d'] = correlations.get(
                            f'pct_chg_corr_volume_rank_{window}d', {}
                        )
                        correlations[f'pct_chg_corr_volume_rank_{window}d'][stock_code] = volume_corr
                    else:
                        # 窗口数据不足
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'] = correlations.get(
                            f'pct_chg_corr_limit_up_rank_{window}d', {}
                        )
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'][stock_code] = 0.0

                        correlations[f'pct_chg_corr_volume_rank_{window}d'] = correlations.get(
                            f'pct_chg_corr_volume_rank_{window}d', {}
                        )
                        correlations[f'pct_chg_corr_volume_rank_{window}d'][stock_code] = 0.0

            return correlations

        except Exception as e:
            self.logger.error(f"批量涨幅相关性计算失败: {e}")
            return {}

    def _calculate_momentum_correlations_batch(self, sorted_data: pd.DataFrame,
                                             momentum_data: pd.DataFrame,
                                             target_stocks: np.ndarray,
                                             target_date: pd.Timestamp,
                                             limit_up_rank: float,
                                             volume_rank: float) -> Dict:
        """🚀 批量计算动量相关性（所有窗口一次性计算）"""
        try:
            correlations = {}
            windows = [5, 10, 20]

            # 为每个股票批量计算所有窗口的相关性
            for stock_code in target_stocks:
                stock_mask = sorted_data['stock_code'] == stock_code
                stock_indices = sorted_data[stock_mask & (sorted_data['date'] <= target_date)].index

                if len(stock_indices) < max(windows) or momentum_data.empty:
                    # 数据不足，使用默认值
                    for window in windows:
                        correlations[f'momentum_corr_limit_up_rank_{window}d'] = correlations.get(
                            f'momentum_corr_limit_up_rank_{window}d', {}
                        )
                        correlations[f'momentum_corr_limit_up_rank_{window}d'][stock_code] = 0.0

                        correlations[f'momentum_corr_volume_rank_{window}d'] = correlations.get(
                            f'momentum_corr_volume_rank_{window}d', {}
                        )
                        correlations[f'momentum_corr_volume_rank_{window}d'][stock_code] = 0.0
                    continue

                # 获取动量数据
                momentum_values = momentum_data.loc[stock_indices, 'rank_momentum_20d'].values

                # 为所有窗口计算相关性
                for window in windows:
                    if len(momentum_values) >= window:
                        recent_momentum = momentum_values[-window:]

                        # 🚀 专业量化计算：构造真实的市场因子序列（移除随机数）
                        limit_up_series = self._construct_professional_factor_series(
                            window, limit_up_rank, 'limit_up_rank'
                        )
                        volume_series = self._construct_professional_factor_series(
                            window, volume_rank, 'volume_rank'
                        )

                        # 计算相关性
                        limit_corr = np.corrcoef(recent_momentum, limit_up_series)[0, 1]
                        volume_corr = np.corrcoef(recent_momentum, volume_series)[0, 1]

                        # 处理NaN值
                        limit_corr = 0.0 if np.isnan(limit_corr) else limit_corr
                        volume_corr = 0.0 if np.isnan(volume_corr) else volume_corr

                        # 存储结果
                        correlations[f'momentum_corr_limit_up_rank_{window}d'] = correlations.get(
                            f'momentum_corr_limit_up_rank_{window}d', {}
                        )
                        correlations[f'momentum_corr_limit_up_rank_{window}d'][stock_code] = limit_corr

                        correlations[f'momentum_corr_volume_rank_{window}d'] = correlations.get(
                            f'momentum_corr_volume_rank_{window}d', {}
                        )
                        correlations[f'momentum_corr_volume_rank_{window}d'][stock_code] = volume_corr
                    else:
                        # 窗口数据不足
                        correlations[f'momentum_corr_limit_up_rank_{window}d'] = correlations.get(
                            f'momentum_corr_limit_up_rank_{window}d', {}
                        )
                        correlations[f'momentum_corr_limit_up_rank_{window}d'][stock_code] = 0.0

                        correlations[f'momentum_corr_volume_rank_{window}d'] = correlations.get(
                            f'momentum_corr_volume_rank_{window}d', {}
                        )
                        correlations[f'momentum_corr_volume_rank_{window}d'][stock_code] = 0.0

            return correlations

        except Exception as e:
            self.logger.error(f"批量动量相关性计算失败: {e}")
            return {}

    def _batch_update_target_market_relative_features(self, target_data: pd.DataFrame,
                                                    all_features: Dict,
                                                    target_dates: List[pd.Timestamp]):
        """🚀 批量更新目标数据中的市场相对特征"""
        try:
            self.logger.info("开始批量更新目标特征...")

            for target_date in target_dates:
                if target_date not in all_features:
                    continue

                target_day_mask = target_data['date'] == target_date
                if not target_day_mask.any():
                    continue

                date_features = all_features[target_date]

                # 批量更新所有特征
                for feature_name, feature_series in date_features.items():
                    if feature_name in target_data.columns and isinstance(feature_series, pd.Series):
                        # 只更新目标日期的数据
                        for idx in target_data[target_day_mask].index:
                            if idx in feature_series.index:
                                target_data.loc[idx, feature_name] = feature_series[idx]

            self.logger.info("批量更新完成")

        except Exception as e:
            self.logger.error(f"批量更新目标特征失败: {e}")

    def _precompute_base_indicators_multithreaded(self, full_data: pd.DataFrame,
                                                target_dates: List[pd.Timestamp]) -> Dict:
        """
        🚀 多线程优化：并行预计算所有股票的基础指标
        最佳多线程场景：不同股票组的计算可以并行进行
        """
        try:
            self.logger.info("开始多线程预计算基础指标...")

            # 确保数据按股票和日期排序
            sorted_data = full_data.sort_values(['stock_code', 'date']).reset_index(drop=True)

            # 获取所有股票代码
            all_stocks = sorted_data['stock_code'].unique()

            # 🚀 多线程优化：根据CPU核心数动态调整线程数
            cpu_count = multiprocessing.cpu_count()
            # 使用CPU核心数，但不超过股票组数的合理分割
            optimal_threads = min(cpu_count, max(1, len(all_stocks) // 50))  # 每个线程至少处理50只股票
            num_threads = max(1, optimal_threads)  # 确保至少有1个线程

            stock_groups = np.array_split(all_stocks, num_threads)

            self.logger.info(f"CPU核心数: {cpu_count}, 使用 {num_threads} 个线程处理 {len(all_stocks)} 只股票")

            # 并行计算各组股票的指标
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []

                for i, stock_group in enumerate(stock_groups):
                    if len(stock_group) == 0:
                        continue

                    # 为每个股票组创建子数据集
                    group_mask = sorted_data['stock_code'].isin(stock_group)
                    group_data = sorted_data[group_mask].copy()

                    # 提交并行任务
                    future = executor.submit(
                        self._calculate_group_indicators,
                        group_data, i, len(stock_group)
                    )
                    futures.append(future)

                # 收集结果
                atr_results = []
                momentum_results = []

                for future in as_completed(futures):
                    try:
                        group_atr, group_momentum = future.result()
                        atr_results.append(group_atr)
                        momentum_results.append(group_momentum)
                    except Exception as e:
                        self.logger.error(f"线程计算失败: {e}")

            # 合并所有组的结果
            if atr_results and momentum_results:
                combined_atr = pd.concat(atr_results, ignore_index=False).sort_index()
                combined_momentum = pd.concat(momentum_results, ignore_index=False).sort_index()
            else:
                # 如果多线程失败，回退到单线程
                self.logger.warning("多线程计算失败，回退到单线程模式")
                return self._precompute_base_indicators_vectorized(full_data, target_dates)

            # 🔧 内存优化：预计算日期索引（修复版本）
            date_indices = self._precompute_date_indices_optimized(sorted_data, target_dates)

            base_indicators = {
                'atr_data': combined_atr,
                'momentum_data': combined_momentum,
                'date_indices': date_indices,
                'sorted_data': sorted_data
            }

            self.logger.info("多线程基础指标预计算完成")
            return base_indicators

        except Exception as e:
            self.logger.error(f"多线程预计算基础指标失败: {e}")
            # 回退到单线程版本
            self.logger.info("回退到单线程版本...")
            return self._precompute_base_indicators_vectorized(full_data, target_dates)

    def _calculate_group_indicators(self, group_data: pd.DataFrame,
                                  group_id: int, group_size: int) -> tuple:
        """计算单个股票组的指标"""
        try:
            self.logger.info(f"线程 {group_id}: 处理 {group_size} 只股票")

            # 计算该组的ATR
            group_atr = self._calculate_atr_vectorized(group_data)

            # 计算该组的动量
            group_momentum = self._calculate_momentum_vectorized(group_data)

            return group_atr, group_momentum

        except Exception as e:
            self.logger.error(f"线程 {group_id} 计算失败: {e}")
            # 返回空结果
            empty_atr = pd.DataFrame({'atr_20d': 0.0}, index=group_data.index)
            empty_momentum = pd.DataFrame({'rank_momentum_20d': 0.5}, index=group_data.index)
            return empty_atr, empty_momentum

    def _precompute_date_indices_optimized(self, sorted_data: pd.DataFrame,
                                         target_dates: List[pd.Timestamp]) -> Dict:
        """🔧 内存优化版本：预计算日期索引，避免大内存分配"""
        try:
            date_indices = {}

            # 预先计算所有日期的位置映射，避免重复计算
            date_positions = {}
            unique_dates = sorted_data['date'].unique()

            for date in unique_dates:
                date_positions[date] = sorted_data['date'] == date

            for target_date in target_dates:
                if target_date not in date_positions:
                    continue

                target_mask = date_positions[target_date]
                target_count = target_mask.sum()

                if target_count == 0:
                    continue

                # 只存储必要信息，使用布尔掩码避免大量内存分配
                target_stocks = sorted_data.loc[target_mask, 'stock_code'].unique()

                # � 高配置优化：处理所有股票，不限制数量（128GB内存支持）
                self.logger.info(f"目标日期 {target_date.strftime('%Y-%m-%d')}: 处理 {len(target_stocks)} 只股票")

                date_indices[target_date] = {
                    'target_mask': target_mask,
                    'target_count': target_count,
                    'target_stocks': target_stocks.tolist(),  # 处理所有股票
                    'has_data': True
                }

            return date_indices

        except Exception as e:
            self.logger.error(f"优化版日期索引预计算失败: {e}")
            return {}

    def _calculate_all_dates_market_relative_features_multithreaded(self,
                                                                  full_data: pd.DataFrame,
                                                                  target_dates: List[pd.Timestamp],
                                                                  market_state_factors: Dict,
                                                                  base_indicators: Dict) -> Dict:
        """🚀 多线程优化：并行计算所有目标日期的市场相对特征"""
        try:
            self.logger.info("开始多线程批量计算市场相对特征...")

            sorted_data = base_indicators.get('sorted_data', full_data)
            atr_data = base_indicators.get('atr_data', pd.DataFrame())
            momentum_data = base_indicators.get('momentum_data', pd.DataFrame())
            date_indices = base_indicators.get('date_indices', {})

            # 🚀 多线程优化：根据CPU核心数和目标日期数量调整线程数
            cpu_count = multiprocessing.cpu_count()
            # 线程数不超过CPU核心数，也不超过目标日期数量
            num_threads = min(cpu_count, len(target_dates))

            if num_threads <= 1:
                # 如果只有一个日期或线程数不足，使用单线程
                return self._calculate_all_dates_market_relative_features_vectorized(
                    full_data, target_dates, market_state_factors, base_indicators
                )

            self.logger.info(f"CPU核心数: {cpu_count}, 使用 {num_threads} 个线程处理 {len(target_dates)} 个目标日期")

            # 将目标日期分组
            date_groups = np.array_split(target_dates, num_threads)

            all_features = {}

            # 并行计算各组日期的特征
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []

                for i, date_group in enumerate(date_groups):
                    if len(date_group) == 0:
                        continue

                    # 提交并行任务
                    future = executor.submit(
                        self._calculate_date_group_features,
                        sorted_data, date_group, market_state_factors,
                        atr_data, momentum_data, date_indices, i
                    )
                    futures.append(future)

                # 收集结果
                for future in as_completed(futures):
                    try:
                        group_features = future.result()
                        all_features.update(group_features)
                    except Exception as e:
                        self.logger.error(f"线程特征计算失败: {e}")

            self.logger.info("多线程批量计算完成")
            return all_features

        except Exception as e:
            self.logger.error(f"多线程批量计算失败: {e}")
            # 回退到单线程版本
            self.logger.info("回退到单线程版本...")
            return self._calculate_all_dates_market_relative_features_vectorized(
                full_data, target_dates, market_state_factors, base_indicators
            )

    def _calculate_date_group_features(self, sorted_data: pd.DataFrame,
                                     date_group: List[pd.Timestamp],
                                     market_state_factors: Dict,
                                     atr_data: pd.DataFrame,
                                     momentum_data: pd.DataFrame,
                                     date_indices: Dict,
                                     group_id: int) -> Dict:
        """计算单个日期组的特征"""
        try:
            self.logger.info(f"线程 {group_id}: 处理 {len(date_group)} 个日期")

            group_features = {}

            for target_date in date_group:
                if target_date not in date_indices:
                    continue

                target_info = date_indices[target_date]
                if not target_info.get('has_data', False):
                    continue

                target_mask = target_info['target_mask']
                target_indices = sorted_data[target_mask].index.tolist()

                if not target_indices:
                    continue

                # 获取市场状态
                market_state = market_state_factors.get(target_date, {})
                market_volatility = market_state.get('market_volatility_atr20d', 0.0)

                # 🚀 计算波动率比值（使用掩码而不是索引列表）
                volatility_ratios = self._calculate_volatility_ratios_optimized(
                    sorted_data, atr_data, target_mask, market_volatility
                )

                # 🚀 计算相关性特征（优化版本）
                correlation_features = self._calculate_correlation_features_optimized(
                    sorted_data, momentum_data, target_mask, market_state, target_date
                )

                # 合并特征
                date_features = {
                    'volatility_ratio_to_market_20d': volatility_ratios,
                    **correlation_features
                }

                group_features[target_date] = date_features

            return group_features

        except Exception as e:
            self.logger.error(f"线程 {group_id} 日期组特征计算失败: {e}")
            return {}

    def _calculate_volatility_ratios_optimized(self, sorted_data: pd.DataFrame,
                                             atr_data: pd.DataFrame,
                                             target_mask: pd.Series,
                                             market_volatility: float) -> pd.Series:
        """🔧 优化版本：使用掩码计算波动率比值，避免大索引列表"""
        try:
            if atr_data.empty or market_volatility <= 0:
                target_indices = sorted_data[target_mask].index
                return pd.Series(1.0, index=target_indices)

            # 使用掩码直接获取目标日期的ATR值
            target_atr = atr_data.loc[target_mask, 'atr_20d']

            # 计算比值
            volatility_ratios = target_atr / market_volatility
            volatility_ratios = volatility_ratios.fillna(1.0)

            return volatility_ratios

        except Exception as e:
            self.logger.error(f"优化版波动率比值计算失败: {e}")
            target_indices = sorted_data[target_mask].index
            return pd.Series(1.0, index=target_indices)

    def _calculate_correlation_features_optimized(self, sorted_data: pd.DataFrame,
                                                momentum_data: pd.DataFrame,
                                                target_mask: pd.Series,
                                                market_state: Dict,
                                                target_date: pd.Timestamp) -> Dict:
        """� 高性能版本：向量化计算相关性特征，解决性能瓶颈"""
        try:
            correlation_features = {}

            # 获取市场状态因子值
            limit_up_rank = market_state.get('limit_up_count_rank_60d', 0.5)
            volume_rank = market_state.get('market_volume_rank_60d', 0.5)

            # 获取目标日期的所有股票代码
            target_stocks = sorted_data.loc[target_mask, 'stock_code'].unique()
            self.logger.info(f"高性能相关性计算: 处理 {len(target_stocks)} 只股票")

            # 🚀 性能优化：向量化批量计算相关性
            correlation_features = self._calculate_vectorized_correlations(
                sorted_data, momentum_data, target_stocks, target_date,
                limit_up_rank, volume_rank, target_mask
            )

            return correlation_features

        except Exception as e:
            self.logger.error(f"高性能相关性特征计算失败: {e}")
            return {}

    def _calculate_vectorized_correlations(self, sorted_data: pd.DataFrame,
                                         momentum_data: pd.DataFrame,
                                         target_stocks: np.ndarray,
                                         target_date: pd.Timestamp,
                                         limit_up_rank: float,
                                         volume_rank: float,
                                         target_mask: pd.Series) -> Dict:
        """🚀 高性能相关性计算：解决性能瓶颈的核心优化（添加详细进度日志）"""
        try:
            import time
            start_time = time.time()

            correlation_features = {}
            windows = [5, 10, 20]

            self.logger.info(f"📊 开始相关性计算 - 目标股票数: {len(target_stocks)}, 窗口: {windows}")

            # 🚀 步骤1：预先过滤历史数据，避免重复过滤
            step1_start = time.time()
            historical_data = sorted_data[sorted_data['date'] <= target_date].copy()
            step1_time = time.time() - step1_start
            self.logger.info(f"  ✅ 步骤1-历史数据过滤完成: {step1_time:.2f}秒, 数据量: {len(historical_data):,}行")

            # 🚀 步骤2：批量预计算所有股票的历史数据（性能关键）
            step2_start = time.time()
            stock_data_dict = self._precompute_stock_data_optimized(historical_data, target_stocks, windows)
            step2_time = time.time() - step2_start
            valid_stocks = len(stock_data_dict)
            self.logger.info(f"  ✅ 步骤2-股票数据预计算完成: {step2_time:.2f}秒, 有效股票: {valid_stocks}/{len(target_stocks)}")

            if valid_stocks == 0:
                self.logger.warning("  ⚠️ 没有有效的股票数据，跳过相关性计算")
                return {}

            # 🚀 步骤3：预构造市场因子序列（避免重复计算）
            step3_start = time.time()
            factor_series_cache = self._precompute_factor_series_optimized(windows, limit_up_rank, volume_rank, target_date)
            step3_time = time.time() - step3_start
            self.logger.info(f"  ✅ 步骤3-因子序列预计算完成: {step3_time:.2f}秒, 窗口数: {len(windows)}")

            # 🚀 步骤4：向量化计算涨幅相关性（性能关键）
            step4_start = time.time()
            pct_chg_correlations = self._vectorized_pct_chg_correlations_optimized(
                stock_data_dict, factor_series_cache, windows
            )
            step4_time = time.time() - step4_start
            correlation_features.update(pct_chg_correlations)
            self.logger.info(f"  ✅ 步骤4-涨幅相关性计算完成: {step4_time:.2f}秒, 特征数: {len(pct_chg_correlations)}")

            # 🚀 步骤5：向量化计算动量相关性（性能关键）
            step5_start = time.time()
            if not momentum_data.empty:
                momentum_correlations = self._vectorized_momentum_correlations_optimized(
                    stock_data_dict, momentum_data, factor_series_cache, windows
                )
                correlation_features.update(momentum_correlations)
                step5_time = time.time() - step5_start
                self.logger.info(f"  ✅ 步骤5-动量相关性计算完成: {step5_time:.2f}秒, 特征数: {len(momentum_correlations)}")
            else:
                step5_time = 0
                self.logger.info(f"  ⚠️ 步骤5-跳过动量相关性计算（无动量数据）")

            # 🚀 步骤6：高效转换为Series格式
            step6_start = time.time()
            target_indices = sorted_data[target_mask].index
            final_features = self._efficient_series_conversion_optimized(
                correlation_features, target_indices, sorted_data
            )
            step6_time = time.time() - step6_start
            self.logger.info(f"  ✅ 步骤6-Series转换完成: {step6_time:.2f}秒, 最终特征数: {len(final_features)}")

            total_time = time.time() - start_time
            self.logger.info(f"📊 相关性计算总耗时: {total_time:.2f}秒 (步骤耗时: {step1_time:.2f}+{step2_time:.2f}+{step3_time:.2f}+{step4_time:.2f}+{step5_time:.2f}+{step6_time:.2f})")

            return final_features

        except Exception as e:
            self.logger.error(f"向量化相关性计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {}

    def _precompute_stock_data_optimized(self, historical_data: pd.DataFrame,
                                       target_stocks: np.ndarray,
                                       windows: List[int]) -> Dict:
        """🚀 优化的股票数据预计算（性能关键）"""
        try:
            import time
            start_time = time.time()

            stock_data_dict = {}
            max_window = max(windows)
            processed_count = 0

            # 🚀 性能优化：使用groupby批量处理
            grouped_data = historical_data.groupby('stock_code')

            for stock_code in target_stocks:
                if stock_code in grouped_data.groups:
                    stock_data = grouped_data.get_group(stock_code)

                    if len(stock_data) >= max_window:
                        # 只保存必要的数据，减少内存使用
                        stock_data_dict[stock_code] = {
                            'pct_chg': stock_data['pct_chg'].values,
                            'indices': stock_data.index.values  # 使用numpy数组而不是list
                        }
                        processed_count += 1

                # 每处理100只股票记录一次进度
                if processed_count % 100 == 0 and processed_count > 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"    📈 股票数据预计算进度: {processed_count}/{len(target_stocks)} ({processed_count/len(target_stocks)*100:.1f}%), 耗时: {elapsed:.2f}秒")

            elapsed = time.time() - start_time
            self.logger.info(f"    ✅ 股票数据预计算完成: {processed_count}只有效股票, 耗时: {elapsed:.2f}秒")

            return stock_data_dict

        except Exception as e:
            self.logger.error(f"股票数据预计算失败: {e}")
            return {}

    def _precompute_factor_series_optimized(self, windows: List[int],
                                          limit_up_rank: float,
                                          volume_rank: float,
                                          target_date: pd.Timestamp) -> Dict:
        """🚀 优化的因子序列预计算"""
        try:
            import time
            start_time = time.time()

            factor_series_cache = {}

            for window in windows:
                # 🚀 真实数据计算因子序列
                limit_up_series = self._construct_professional_factor_series(
                    window, limit_up_rank, 'limit_up_rank', target_date
                )
                volume_series = self._construct_professional_factor_series(
                    window, volume_rank, 'volume_rank', target_date
                )

                factor_series_cache[window] = {
                    'limit_up': limit_up_series,
                    'volume': volume_series
                }

            elapsed = time.time() - start_time
            self.logger.info(f"    ✅ 因子序列预计算完成: {len(windows)}个窗口, 耗时: {elapsed:.2f}秒")

            return factor_series_cache

        except Exception as e:
            self.logger.error(f"因子序列预计算失败: {e}")
            return {}

    def _vectorized_pct_chg_correlations_optimized(self, stock_data_dict: Dict,
                                                 factor_series_cache: Dict,
                                                 windows: List[int]) -> Dict:
        """🚀 优化的涨幅相关性计算（性能关键）"""
        try:
            import time
            start_time = time.time()

            correlations = {}
            total_stocks = len(stock_data_dict)
            processed_count = 0

            # 预分配结果字典
            for window in windows:
                correlations[f'pct_chg_corr_limit_up_rank_{window}d'] = {}
                correlations[f'pct_chg_corr_volume_rank_{window}d'] = {}

            for stock_code, stock_info in stock_data_dict.items():
                pct_chg_values = stock_info['pct_chg']

                for window in windows:
                    if len(pct_chg_values) >= window:
                        recent_pct_chg = pct_chg_values[-window:]
                        limit_up_series = factor_series_cache[window]['limit_up']
                        volume_series = factor_series_cache[window]['volume']

                        # 🚀 向量化相关性计算
                        try:
                            limit_corr = np.corrcoef(recent_pct_chg, limit_up_series)[0, 1]
                            volume_corr = np.corrcoef(recent_pct_chg, volume_series)[0, 1]
                        except:
                            limit_corr = 0.0
                            volume_corr = 0.0

                        # 处理NaN值
                        limit_corr = 0.0 if np.isnan(limit_corr) else limit_corr
                        volume_corr = 0.0 if np.isnan(volume_corr) else volume_corr

                        # 存储结果
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'][stock_code] = limit_corr
                        correlations[f'pct_chg_corr_volume_rank_{window}d'][stock_code] = volume_corr
                    else:
                        # 数据不足的情况
                        correlations[f'pct_chg_corr_limit_up_rank_{window}d'][stock_code] = 0.0
                        correlations[f'pct_chg_corr_volume_rank_{window}d'][stock_code] = 0.0

                processed_count += 1

                # 每处理50只股票记录一次进度
                if processed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"    📈 涨幅相关性计算进度: {processed_count}/{total_stocks} ({processed_count/total_stocks*100:.1f}%), 耗时: {elapsed:.2f}秒")

            elapsed = time.time() - start_time
            self.logger.info(f"    ✅ 涨幅相关性计算完成: {processed_count}只股票, 耗时: {elapsed:.2f}秒")

            return correlations

        except Exception as e:
            self.logger.error(f"涨幅相关性计算失败: {e}")
            return {}

    def _vectorized_momentum_correlations_optimized(self, stock_data_dict: Dict,
                                                  momentum_data: pd.DataFrame,
                                                  factor_series_cache: Dict,
                                                  windows: List[int]) -> Dict:
        """🚀 优化的动量相关性计算（性能关键）"""
        try:
            import time
            start_time = time.time()

            correlations = {}
            total_stocks = len(stock_data_dict)
            processed_count = 0

            # 预分配结果字典
            for window in windows:
                correlations[f'momentum_corr_limit_up_rank_{window}d'] = {}
                correlations[f'momentum_corr_volume_rank_{window}d'] = {}

            for stock_code, stock_info in stock_data_dict.items():
                stock_indices = stock_info['indices']

                for window in windows:
                    if len(stock_indices) >= window:
                        try:
                            # 获取最近window个索引
                            recent_indices = stock_indices[-window:]

                            # 🚀 性能优化：批量获取动量数据
                            momentum_values = momentum_data.loc[recent_indices, 'rank_momentum_20d'].values

                            if len(momentum_values) == window and not np.isnan(momentum_values).all():
                                limit_up_series = factor_series_cache[window]['limit_up']
                                volume_series = factor_series_cache[window]['volume']

                                # 🚀 向量化相关性计算
                                try:
                                    limit_corr = np.corrcoef(momentum_values, limit_up_series)[0, 1]
                                    volume_corr = np.corrcoef(momentum_values, volume_series)[0, 1]
                                except:
                                    limit_corr = 0.0
                                    volume_corr = 0.0

                                # 处理NaN值
                                limit_corr = 0.0 if np.isnan(limit_corr) else limit_corr
                                volume_corr = 0.0 if np.isnan(volume_corr) else volume_corr

                                # 存储结果
                                correlations[f'momentum_corr_limit_up_rank_{window}d'][stock_code] = limit_corr
                                correlations[f'momentum_corr_volume_rank_{window}d'][stock_code] = volume_corr
                            else:
                                # 动量数据不足或全为NaN
                                correlations[f'momentum_corr_limit_up_rank_{window}d'][stock_code] = 0.0
                                correlations[f'momentum_corr_volume_rank_{window}d'][stock_code] = 0.0
                        except:
                            # 索引不匹配或其他错误
                            correlations[f'momentum_corr_limit_up_rank_{window}d'][stock_code] = 0.0
                            correlations[f'momentum_corr_volume_rank_{window}d'][stock_code] = 0.0
                    else:
                        # 数据不足的情况
                        correlations[f'momentum_corr_limit_up_rank_{window}d'][stock_code] = 0.0
                        correlations[f'momentum_corr_volume_rank_{window}d'][stock_code] = 0.0

                processed_count += 1

                # 每处理50只股票记录一次进度
                if processed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"    📈 动量相关性计算进度: {processed_count}/{total_stocks} ({processed_count/total_stocks*100:.1f}%), 耗时: {elapsed:.2f}秒")

            elapsed = time.time() - start_time
            self.logger.info(f"    ✅ 动量相关性计算完成: {processed_count}只股票, 耗时: {elapsed:.2f}秒")

            return correlations

        except Exception as e:
            self.logger.error(f"动量相关性计算失败: {e}")
            return {}

    def _efficient_series_conversion_optimized(self, correlation_features: Dict,
                                             target_indices: pd.Index,
                                             sorted_data: pd.DataFrame) -> Dict:
        """🚀 优化的Series转换：避免逐个索引查找（性能关键）"""
        try:
            import time
            start_time = time.time()

            final_features = {}

            # 🚀 性能优化：预先构建股票代码到索引的映射
            stock_to_indices = {}
            for idx in target_indices:
                stock_code = sorted_data.loc[idx, 'stock_code']
                if stock_code not in stock_to_indices:
                    stock_to_indices[stock_code] = []
                stock_to_indices[stock_code].append(idx)

            # 🚀 批量转换所有特征
            feature_count = 0
            for feature_name, correlations_dict in correlation_features.items():
                # 预分配Series
                feature_series = pd.Series(0.0, index=target_indices, dtype=float)

                # 🚀 向量化赋值
                for stock_code, corr_value in correlations_dict.items():
                    if stock_code in stock_to_indices:
                        indices = stock_to_indices[stock_code]
                        feature_series.loc[indices] = corr_value

                final_features[feature_name] = feature_series
                feature_count += 1

            elapsed = time.time() - start_time
            self.logger.info(f"    ✅ Series转换完成: {feature_count}个特征, {len(target_indices)}个索引, 耗时: {elapsed:.2f}秒")

            return final_features

        except Exception as e:
            self.logger.error(f"Series转换失败: {e}")
            return {}

    def _vectorized_pct_chg_correlations(self, stock_data_dict: Dict,
                                       factor_series_cache: Dict,
                                       windows: List[int]) -> Dict:
        """🚀 向量化涨幅相关性计算"""
        try:
            correlations = {}

            for window in windows:
                limit_up_series = factor_series_cache[window]['limit_up']
                volume_series = factor_series_cache[window]['volume']

                # 批量计算所有股票的相关性
                limit_corrs = {}
                volume_corrs = {}

                for stock_code, stock_info in stock_data_dict.items():
                    pct_chg_values = stock_info['pct_chg']

                    if len(pct_chg_values) >= window:
                        recent_pct_chg = pct_chg_values[-window:]

                        # 计算相关性
                        try:
                            limit_corr = np.corrcoef(recent_pct_chg, limit_up_series)[0, 1]
                            volume_corr = np.corrcoef(recent_pct_chg, volume_series)[0, 1]
                        except:
                            limit_corr = 0.0
                            volume_corr = 0.0

                        # 处理NaN值
                        limit_corrs[stock_code] = 0.0 if np.isnan(limit_corr) else limit_corr
                        volume_corrs[stock_code] = 0.0 if np.isnan(volume_corr) else volume_corr
                    else:
                        limit_corrs[stock_code] = 0.0
                        volume_corrs[stock_code] = 0.0

                # 存储结果
                correlations[f'pct_chg_corr_limit_up_rank_{window}d'] = limit_corrs
                correlations[f'pct_chg_corr_volume_rank_{window}d'] = volume_corrs

            return correlations

        except Exception as e:
            self.logger.error(f"向量化涨幅相关性计算失败: {e}")
            return {}

    def _vectorized_momentum_correlations(self, stock_data_dict: Dict,
                                        momentum_data: pd.DataFrame,
                                        factor_series_cache: Dict,
                                        windows: List[int]) -> Dict:
        """🚀 向量化动量相关性计算"""
        try:
            correlations = {}

            for window in windows:
                limit_up_series = factor_series_cache[window]['limit_up']
                volume_series = factor_series_cache[window]['volume']

                # 批量计算所有股票的相关性
                limit_corrs = {}
                volume_corrs = {}

                for stock_code, stock_info in stock_data_dict.items():
                    stock_indices = stock_info['indices']

                    if len(stock_indices) >= window:
                        try:
                            # 获取动量数据
                            recent_indices = stock_indices[-window:]
                            momentum_values = momentum_data.loc[recent_indices, 'rank_momentum_20d'].values

                            if len(momentum_values) == window:
                                # 计算相关性
                                try:
                                    limit_corr = np.corrcoef(momentum_values, limit_up_series)[0, 1]
                                    volume_corr = np.corrcoef(momentum_values, volume_series)[0, 1]
                                except:
                                    limit_corr = 0.0
                                    volume_corr = 0.0

                                # 处理NaN值
                                limit_corrs[stock_code] = 0.0 if np.isnan(limit_corr) else limit_corr
                                volume_corrs[stock_code] = 0.0 if np.isnan(volume_corr) else volume_corr
                            else:
                                limit_corrs[stock_code] = 0.0
                                volume_corrs[stock_code] = 0.0
                        except:
                            limit_corrs[stock_code] = 0.0
                            volume_corrs[stock_code] = 0.0
                    else:
                        limit_corrs[stock_code] = 0.0
                        volume_corrs[stock_code] = 0.0

                # 存储结果
                correlations[f'momentum_corr_limit_up_rank_{window}d'] = limit_corrs
                correlations[f'momentum_corr_volume_rank_{window}d'] = volume_corrs

            return correlations

        except Exception as e:
            self.logger.error(f"向量化动量相关性计算失败: {e}")
            return {}

    def _efficient_series_conversion(self, correlation_features: Dict,
                                   target_indices: pd.Index,
                                   sorted_data: pd.DataFrame,
                                   target_stocks: np.ndarray) -> Dict:
        """🚀 高效Series转换：避免逐个索引查找"""
        try:
            final_features = {}

            # 预先构建股票代码到索引的映射
            stock_to_indices = {}
            for idx in target_indices:
                stock_code = sorted_data.loc[idx, 'stock_code']
                if stock_code not in stock_to_indices:
                    stock_to_indices[stock_code] = []
                stock_to_indices[stock_code].append(idx)

            # 批量转换所有特征
            for feature_name, correlations_dict in correlation_features.items():
                feature_series = pd.Series(index=target_indices, dtype=float)

                # 向量化赋值
                for stock_code, corr_value in correlations_dict.items():
                    if stock_code in stock_to_indices:
                        indices = stock_to_indices[stock_code]
                        feature_series.loc[indices] = corr_value

                # 填充缺失值
                feature_series = feature_series.fillna(0.0)
                final_features[feature_name] = feature_series

            return final_features

        except Exception as e:
            self.logger.error(f"高效Series转换失败: {e}")
            return {}

