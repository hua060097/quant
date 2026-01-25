#!/usr/bin/env python3
"""
市场相关性因子组

基于CAPM模型和多因子模型理论，计算股票与市场的新型相关性因子
参考enhanced_correlation_factors.py的实现架构

新增因子：
1. volume_correlation: 计算股票成交量与市场成交量的相关性
2. corr_5d: 计算5日短期价格相关性
3. volatility_correlation: 计算股票波动率与市场波动率的相关性

数据源结构说明：
- market_data: 包含市场指数的OHLCV数据、日收益率等
  * date: 交易日期
  * close: 收盘价
  * volume: 成交量
  * daily_change_pct: 日收益率
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from .base_factor_group import BaseFactorGroup

class MarketCorrFactors(BaseFactorGroup):
    """市场相关性因子组"""
    
    def __init__(self):
        super().__init__("market_corr_factors")
    
    def _get_version(self) -> str:
        return "v1.0"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接使用原始数据计算相关性特征
    
    def _get_feature_names(self) -> List[str]:
        """返回新增的3个市场相关性因子"""
        return [
            'volume_correlation',      # 股票成交量与市场成交量的相关性
            'corr_5d',                # 5日短期价格相关性
            'volatility_correlation'   # 股票波动率与市场波动率的相关性
        ]
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算市场相关性因子（基于enhanced_correlation_factors.py的专业实现）"""
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

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_market_corr_features_optimized(
                data, target_dates, historical_dates, market_data
            )

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            return result_data

        except Exception as e:
            self.logger.error(f"市场相关性因子计算失败: {e}")
            return data

    def _calculate_market_corr_features_optimized(self, full_data: pd.DataFrame,
                                                 target_dates: List[str],
                                                 historical_dates: List[str],
                                                 market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        高性能优化的市场相关性特征计算

        性能优化策略：
        1. 预计算和缓存所有市场数据
        2. 向量化股票数据预处理
        3. 批量计算相关性指标
        4. 减少重复的数据访问和转换
        5. 优化内存使用模式
        """
        try:
            self.logger.info("开始高性能优化的市场相关性特征计算")

            # 1. 创建目标日期的结果DataFrame
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列
            self._initialize_target_market_corr_features(result_data)

            # 3. 🚀 新的高性能计算逻辑
            result_data = self._calculate_target_market_corr_features_vectorized(
                full_data, result_data, target_dates, market_data
            )

            self.logger.info("高性能优化的市场相关性特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"高性能优化的市场相关性特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_market_corr_features(self, result_data: pd.DataFrame):
        """为目标日期初始化所有市场相关性特征列"""
        try:
            # 新增的3个市场相关性因子
            result_data['volume_correlation'] = 0.0      # 成交量相关性
            result_data['corr_5d'] = 0.0                 # 5日价格相关性
            result_data['volatility_correlation'] = 0.0  # 波动率相关性

        except Exception as e:
            self.logger.error(f"初始化目标市场相关性特征失败: {e}")

    def _calculate_target_market_corr_features_vectorized(self, full_data: pd.DataFrame,
                                                        target_data: pd.DataFrame,
                                                        target_dates: List[str],
                                                        market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        高性能向量化计算市场相关性特征

        性能优化策略：
        1. 预计算所有股票的基础数据（收益率、波动率等）
        2. 向量化市场数据对齐
        3. 批量计算相关性指标
        4. 减少循环和重复计算
        """
        try:
            self.logger.info("开始向量化计算市场相关性特征...")

            # 验证市场数据
            if market_data is None or market_data.empty:
                self.logger.warning("市场数据无效，相关性因子将使用默认值")
                return target_data

            # 🚀 优化1：预计算市场数据映射（向量化版本）
            market_cache = self._precompute_market_data_vectorized(market_data, full_data)
            if not market_cache:
                self.logger.warning("市场数据预计算失败")
                return target_data

            # 🚀 优化2：预计算所有股票的基础特征（向量化版本）
            stock_features_cache = self._precompute_stock_features_vectorized(full_data)

            # 🚀 优化3：批量计算相关性特征
            target_data = self._batch_calculate_correlations_vectorized(
                target_data, target_dates, market_cache, stock_features_cache
            )

            self.logger.info("向量化计算市场相关性特征完成")
            return target_data

        except Exception as e:
            self.logger.error(f"向量化计算市场相关性特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return target_data

    def _precompute_market_data_vectorized(self, market_data: pd.DataFrame,
                                          full_data: pd.DataFrame) -> dict:
        """向量化预计算市场数据映射"""
        try:
            self.logger.info("开始预计算市场数据...")

            # 获取所有需要的日期
            all_dates = pd.to_datetime(full_data['date'].unique())
            self.logger.info(f"需要处理 {len(all_dates)} 个交易日的市场数据")

            # 确保市场数据的日期列是datetime类型
            market_data = market_data.copy()
            market_data['date'] = pd.to_datetime(market_data['date'])
            self.logger.info(f"原始市场数据: {len(market_data)} 行")

            # 🚀 向量化对齐：使用merge而不是循环
            self.logger.info("对齐市场数据与交易日期...")
            date_df = pd.DataFrame({'date': all_dates})
            aligned_market = date_df.merge(market_data, on='date', how='left')

            # 检查对齐结果
            missing_count = aligned_market['daily_change_pct'].isnull().sum()
            if missing_count > 0:
                self.logger.warning(f"发现 {missing_count} 个交易日缺少市场数据，将使用默认值填充")

            # 填充缺失值
            self.logger.info("填充缺失的市场数据...")
            aligned_market['daily_change_pct'] = aligned_market['daily_change_pct'].fillna(0.0)
            aligned_market['volume'] = aligned_market['volume'].fillna(0.0)

            # 计算市场波动率（20日滚动标准差）
            self.logger.info("计算市场波动率...")
            aligned_market['market_volatility'] = (
                aligned_market['daily_change_pct'].rolling(window=20, min_periods=1).std()
            )

            # 创建日期到数据的映射
            self.logger.info("创建市场数据映射...")
            market_cache = {
                'returns': dict(zip(aligned_market['date'], aligned_market['daily_change_pct'] / 100.0)),
                'volumes': dict(zip(aligned_market['date'], aligned_market['volume'])),
                'volatilities': dict(zip(aligned_market['date'], aligned_market['market_volatility'] / 100.0))
            }

            self.logger.info(f"市场数据预计算完成: {len(market_cache['returns'])} 个交易日")
            return market_cache

        except Exception as e:
            self.logger.error(f"向量化预计算市场数据失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {}

    def _precompute_stock_features_vectorized(self, full_data: pd.DataFrame) -> dict:
        """向量化预计算股票特征"""
        try:
            self.logger.info("预计算股票特征...")

            # 获取股票分组
            grouped = full_data.groupby('stock_code')
            total_stocks = len(grouped)
            self.logger.info(f"需要预计算 {total_stocks} 只股票的特征")

            # 🚀 向量化计算所有股票的收益率和波动率
            stock_features = {}
            processed_count = 0

            # 按股票分组，向量化计算
            for stock_code, stock_data in grouped:
                stock_data = stock_data.sort_values('date').reset_index(drop=True)

                # 计算收益率
                returns = stock_data['pct_chg'] / 100.0

                # 计算成交量（对数变换）
                volumes_log = np.log1p(stock_data['volume'])

                # 计算波动率（20日滚动标准差）
                volatilities = returns.rolling(window=20, min_periods=1).std()

                stock_features[stock_code] = {
                    'data': stock_data,
                    'returns': returns,
                    'volumes_log': volumes_log,
                    'volatilities': volatilities,
                    'dates': stock_data['date'].values
                }

                processed_count += 1

                # 🚀 添加进度日志
                if processed_count % max(1, total_stocks // 10) == 0 or processed_count == total_stocks:
                    progress_pct = processed_count / total_stocks * 100
                    self.logger.info(f"预计算进度: {processed_count}/{total_stocks} ({progress_pct:.1f}%) - "
                                   f"当前处理 {stock_code}, 数据点: {len(stock_data)}")

            self.logger.info(f"预计算完成，成功处理了 {len(stock_features)} 只股票")
            return stock_features

        except Exception as e:
            self.logger.error(f"向量化预计算股票特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {}

    def _batch_calculate_correlations_vectorized(self, target_data: pd.DataFrame,
                                               target_dates: List[str],
                                               market_cache: dict,
                                               stock_features_cache: dict) -> pd.DataFrame:
        """批量向量化计算相关性特征"""
        try:
            self.logger.info("开始批量向量化计算相关性...")

            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            unique_stocks = target_data['stock_code'].unique()
            total_stocks = len(unique_stocks)

            self.logger.info(f"计算范围: {total_stocks} 只股票 × {len(target_date_objects)} 个交易日")

            # 🚀 优化：批量处理所有股票和日期组合
            update_data = []
            processed_stocks = 0
            skipped_stocks = 0

            for stock_code in unique_stocks:
                if stock_code not in stock_features_cache:
                    skipped_stocks += 1
                    continue

                stock_features = stock_features_cache[stock_code]

                # 🚀 向量化查找目标日期的索引
                date_indices = {}
                for i, date in enumerate(stock_features['dates']):
                    if pd.to_datetime(date) in target_date_objects:
                        date_indices[pd.to_datetime(date)] = i

                # 为每个目标日期计算特征
                stock_data_points = 0
                for target_date_obj in target_date_objects:
                    if target_date_obj not in date_indices:
                        continue

                    data_idx = date_indices[target_date_obj]

                    # 🚀 向量化计算相关性特征
                    features = self._calculate_correlations_vectorized_single(
                        stock_features, data_idx, market_cache
                    )

                    if features:
                        update_data.append({
                            'stock_code': stock_code,
                            'date': target_date_obj,
                            **features
                        })
                        stock_data_points += 1

                processed_stocks += 1

                # 🚀 添加进度日志
                if processed_stocks % max(1, total_stocks // 20) == 0 or processed_stocks == total_stocks:
                    progress_pct = processed_stocks / total_stocks * 100
                    self.logger.info(f"向量化计算进度: {processed_stocks}/{total_stocks} ({progress_pct:.1f}%) - "
                                   f"当前股票 {stock_code} 处理了 {stock_data_points} 个数据点")

            self.logger.info(f"股票处理统计: 成功 {processed_stocks} 只, 跳过 {skipped_stocks} 只")

            # 🚀 批量更新target_data
            if update_data:
                self.logger.info(f"开始批量更新 {len(update_data)} 个数据点到目标数据...")
                self._batch_update_vectorized(target_data, update_data)
                self.logger.info("批量更新完成")
            else:
                self.logger.warning("没有数据需要更新")

            self.logger.info(f"批量向量化计算完成，处理了 {len(update_data)} 个数据点")
            return target_data

        except Exception as e:
            self.logger.error(f"批量向量化计算相关性失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return target_data

    def _calculate_correlations_vectorized_single(self, stock_features: dict,
                                                data_idx: int,
                                                market_cache: dict) -> dict:
        """向量化计算单个股票单个日期的相关性特征"""
        try:
            # 确保有足够的历史数据
            if data_idx < 20:
                return {
                    'volume_correlation': 0.0,
                    'corr_5d': 0.0,
                    'volatility_correlation': 0.0
                }

            features = {}

            # 获取预计算的股票特征
            stock_returns = stock_features['returns']
            stock_volumes_log = stock_features['volumes_log']
            stock_volatilities = stock_features['volatilities']
            stock_dates = stock_features['dates']

            # 🚀 向量化获取对应的市场数据
            market_returns_list = []
            market_volumes_list = []
            market_volatilities_list = []

            for i in range(max(0, data_idx - 19), data_idx + 1):
                if i < len(stock_dates):
                    date = pd.to_datetime(stock_dates[i])
                    market_returns_list.append(market_cache['returns'].get(date, 0.0))
                    market_volumes_list.append(market_cache['volumes'].get(date, 0.0))
                    market_volatilities_list.append(market_cache['volatilities'].get(date, 0.0))

            market_returns_array = np.array(market_returns_list)
            market_volumes_array = np.log1p(np.array(market_volumes_list))  # 对数变换
            market_volatilities_array = np.array(market_volatilities_list)

            # 1. volume_correlation: 20日成交量相关性
            if data_idx >= 19:
                stock_vol_window = stock_volumes_log.iloc[max(0, data_idx - 19):data_idx + 1]
                vol_correlation = np.corrcoef(stock_vol_window, market_volumes_array)[0, 1]
                features['volume_correlation'] = vol_correlation if not np.isnan(vol_correlation) else 0.0
            else:
                features['volume_correlation'] = 0.0

            # 2. corr_5d: 5日价格相关性
            if data_idx >= 4:
                stock_ret_window = stock_returns.iloc[max(0, data_idx - 4):data_idx + 1]
                market_ret_window = market_returns_array[-5:] if len(market_returns_array) >= 5 else market_returns_array

                if len(stock_ret_window) == len(market_ret_window) and len(stock_ret_window) > 1:
                    price_correlation = np.corrcoef(stock_ret_window, market_ret_window)[0, 1]
                    features['corr_5d'] = price_correlation if not np.isnan(price_correlation) else 0.0
                else:
                    features['corr_5d'] = 0.0
            else:
                features['corr_5d'] = 0.0

            # 3. volatility_correlation: 20日波动率相关性
            if data_idx >= 19:
                stock_vol_window = stock_volatilities.iloc[max(0, data_idx - 19):data_idx + 1]
                volatility_correlation = np.corrcoef(stock_vol_window, market_volatilities_array)[0, 1]
                features['volatility_correlation'] = volatility_correlation if not np.isnan(volatility_correlation) else 0.0
            else:
                features['volatility_correlation'] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"向量化计算单个相关性特征失败: {e}")
            return {
                'volume_correlation': 0.0,
                'corr_5d': 0.0,
                'volatility_correlation': 0.0
            }

    def _batch_update_vectorized(self, target_data: pd.DataFrame, update_data: list):
        """高效批量更新目标数据"""
        try:
            if not update_data:
                self.logger.warning("没有数据需要更新")
                return

            self.logger.info(f"开始批量更新 {len(update_data)} 个数据点...")

            # 🚀 转换为DataFrame进行高效合并
            update_df = pd.DataFrame(update_data)
            self.logger.info(f"更新数据转换完成: {len(update_df)} 行 x {len(update_df.columns)} 列")

            # 创建合并键
            self.logger.info("创建合并键...")
            target_data['_merge_key'] = target_data['stock_code'].astype(str) + '_' + target_data['date'].dt.strftime('%Y%m%d')
            update_df['_merge_key'] = update_df['stock_code'].astype(str) + '_' + update_df['date'].dt.strftime('%Y%m%d')

            # 🚀 使用pandas的高效更新机制
            feature_names = ['volume_correlation', 'corr_5d', 'volatility_correlation']

            for i, feature in enumerate(feature_names):
                if feature in update_df.columns:
                    self.logger.info(f"更新特征 {feature} ({i+1}/{len(feature_names)})...")

                    # 创建映射字典
                    feature_map = dict(zip(update_df['_merge_key'], update_df[feature]))

                    # 批量更新
                    mask = target_data['_merge_key'].isin(feature_map.keys())
                    matched_count = mask.sum()

                    if matched_count > 0:
                        target_data.loc[mask, feature] = target_data.loc[mask, '_merge_key'].map(feature_map)
                        self.logger.info(f"  {feature}: 成功更新 {matched_count} 个数据点")
                    else:
                        self.logger.warning(f"  {feature}: 没有匹配的数据点")
                else:
                    self.logger.warning(f"特征 {feature} 不在更新数据中")

            # 清理临时列
            self.logger.info("清理临时列...")
            target_data.drop('_merge_key', axis=1, inplace=True)

            self.logger.info("批量更新完成")

        except Exception as e:
            self.logger.error(f"批量向量化更新失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")

            # 降级到逐行更新
            self.logger.info("降级使用逐行更新方法...")
            updated_count = 0
            for item in update_data:
                try:
                    mask = (target_data['stock_code'] == item['stock_code']) & (target_data['date'] == item['date'])
                    if mask.any():
                        for feature in ['volume_correlation', 'corr_5d', 'volatility_correlation']:
                            if feature in item:
                                target_data.loc[mask, feature] = item[feature]
                        updated_count += 1
                except Exception as item_error:
                    self.logger.error(f"逐行更新失败: {item_error}")

            self.logger.info(f"逐行更新完成，成功更新 {updated_count} 个数据点")

    def _calculate_target_market_corr_features_with_history(self, full_data: pd.DataFrame,
                                                          target_data: pd.DataFrame,
                                                          target_dates: List[str],
                                                          market_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的市场相关性特征（多线程优化版本）
        关键原则：历史数据只作支撑，只为目标日期计算指标

        性能优化：
        1. 预计算市场数据（避免重复计算）
        2. 多线程并行处理股票数据
        3. 线程安全的结果合并
        """
        try:
            self.logger.info("开始计算目标日期的市场相关性特征（多线程优化版本）...")

            # 验证市场数据
            if market_data is None or market_data.empty:
                self.logger.warning("市场数据无效，相关性因子将使用默认值")
                return target_data

            required_cols = ['daily_change_pct', 'volume']
            missing_cols = [col for col in required_cols if col not in market_data.columns]
            if missing_cols:
                self.logger.warning(f"市场数据中缺少必需字段: {missing_cols}")
                return target_data

            # 🚀 性能优化1：预处理市场数据（只处理一次）
            self.logger.info("预处理市场数据...")
            market_cache = self._precompute_market_data(market_data)

            # 🚀 性能优化2：多线程并行处理股票
            target_stocks = target_data['stock_code'].unique()
            total_stocks = len(target_stocks)

            self.logger.info(f"准备使用多线程处理 {total_stocks} 只股票的市场相关性特征")

            # 使用多线程处理股票
            result_data = self._process_stocks_multithreaded(
                full_data, target_data, target_dates, target_stocks, market_cache
            )

            self.logger.info("目标日期市场相关性特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"目标日期市场相关性特征计算失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return target_data

    def _precompute_market_data(self, market_data: pd.DataFrame) -> dict:
        """预计算市场数据映射（避免重复处理）"""
        try:
            market_cache = {
                'returns': {},
                'volumes': {},
                'volatilities': {}
            }
            
            # 按日期排序
            market_data_sorted = market_data.sort_values('date').reset_index(drop=True)
            
            # 计算市场波动率（20日滚动标准差）
            market_data_sorted['market_volatility'] = (
                market_data_sorted['daily_change_pct'].rolling(window=20, min_periods=1).std()
            )
            
            for _, row in market_data_sorted.iterrows():
                date_key = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                market_cache['returns'][date_key] = row['daily_change_pct'] / 100.0
                market_cache['volumes'][date_key] = row['volume']
                market_cache['volatilities'][date_key] = row.get('market_volatility', 0.0) / 100.0
                
            return market_cache

        except Exception as e:
            self.logger.error(f"预计算市场数据失败: {e}")
            return {'returns': {}, 'volumes': {}, 'volatilities': {}}

    def _process_stocks_multithreaded(self, full_data: pd.DataFrame,
                                    target_data: pd.DataFrame,
                                    target_dates: List[str],
                                    target_stocks: np.ndarray,
                                    market_cache: dict) -> pd.DataFrame:
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
            market_cache: 市场数据缓存

        Returns:
            pd.DataFrame: 更新后的目标数据
        """
        try:
            total_stocks = len(target_stocks)

            # 确定线程数量：基于CPU核心数，但不超过股票数量
            max_workers = min(os.cpu_count() or 4, total_stocks, 20)  # 最多20个线程
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

                    # 检查历史数据是否足够（至少需要20天数据计算相关性）
                    if len(stock_full_data) < 20:
                        with progress_lock:
                            filtered_count[0] += 1
                        return None

                    # 计算该股票的市场相关性特征（使用缓存数据）
                    stock_target_features = self._calculate_stock_target_market_corr_features_optimized(
                        stock_full_data, target_dates, market_cache
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

    def _calculate_stock_target_market_corr_features_optimized(self, stock_full_data: pd.DataFrame,
                                                             target_dates: List[str],
                                                             market_cache: dict) -> dict:
        """为单只股票的目标日期计算市场相关性特征（性能优化版本）"""
        try:
            target_features = {}

            # 🚀 性能优化：使用预计算的市场数据
            market_returns = self._prepare_market_data_cached(stock_full_data, market_cache, 'returns')
            market_volumes = self._prepare_market_data_cached(stock_full_data, market_cache, 'volumes')
            market_volatilities = self._prepare_market_data_cached(stock_full_data, market_cache, 'volatilities')

            if market_returns is None or market_volumes is None or market_volatilities is None:
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

                # 🚀 计算市场相关性因子
                correlation_features = self._calculate_market_corr_features_only(
                    stock_full_data, data_idx, market_returns, market_volumes, market_volatilities
                )

                target_features[target_date_obj] = correlation_features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标市场相关性特征失败: {e}")
            return {}

    def _prepare_market_data_cached(self, stock_data: pd.DataFrame,
                                  market_cache: dict, data_type: str) -> pd.Series:
        """使用缓存的市场数据准备与股票数据对齐的市场数据（性能优化版本）"""
        try:
            # 为股票数据的每个日期匹配市场数据
            market_data_list = []
            cache_dict = market_cache.get(data_type, {})

            for _, row in stock_data.iterrows():
                date_key = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                market_value = cache_dict.get(date_key, 0.0)
                market_data_list.append(market_value)

            return pd.Series(market_data_list, index=stock_data.index)

        except Exception as e:
            self.logger.error(f"准备缓存市场数据失败: {e}")
            return None

    def _calculate_market_corr_features_only(self, stock_data: pd.DataFrame,
                                           data_idx: int,
                                           market_returns: pd.Series,
                                           market_volumes: pd.Series,
                                           market_volatilities: pd.Series) -> dict:
        """计算市场相关性因子（核心算法）"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 20:
                return {
                    'volume_correlation': 0.0,
                    'corr_5d': 0.0,
                    'volatility_correlation': 0.0
                }

            # 获取股票数据
            stock_returns = stock_data['pct_chg'] / 100.0
            stock_volumes = stock_data['volume']

            # 计算股票波动率（20日滚动标准差）
            stock_volatilities = stock_returns.rolling(window=20, min_periods=1).std()

            # 1. volume_correlation: 计算股票成交量与市场成交量的相关性（20日窗口）
            if data_idx >= 19:
                end_idx = data_idx
                start_idx = max(0, end_idx - 19)

                stock_vol_window = stock_volumes.iloc[start_idx:end_idx+1]
                market_vol_window = market_volumes.iloc[start_idx:end_idx+1]

                # 使用对数变换减少异常值影响
                stock_vol_log = np.log1p(stock_vol_window)
                market_vol_log = np.log1p(market_vol_window)

                vol_correlation = stock_vol_log.corr(market_vol_log)
                features['volume_correlation'] = vol_correlation if not pd.isna(vol_correlation) else 0.0
            else:
                features['volume_correlation'] = 0.0

            # 2. corr_5d: 计算5日短期价格相关性
            if data_idx >= 4:
                end_idx = data_idx
                start_idx = max(0, end_idx - 4)

                stock_ret_window = stock_returns.iloc[start_idx:end_idx+1]
                market_ret_window = market_returns.iloc[start_idx:end_idx+1]

                price_correlation = stock_ret_window.corr(market_ret_window)
                features['corr_5d'] = price_correlation if not pd.isna(price_correlation) else 0.0
            else:
                features['corr_5d'] = 0.0

            # 3. volatility_correlation: 计算股票波动率与市场波动率的相关性（20日窗口）
            if data_idx >= 19:
                end_idx = data_idx
                start_idx = max(0, end_idx - 19)

                stock_vol_window = stock_volatilities.iloc[start_idx:end_idx+1]
                market_vol_window = market_volatilities.iloc[start_idx:end_idx+1]

                volatility_correlation = stock_vol_window.corr(market_vol_window)
                features['volatility_correlation'] = volatility_correlation if not pd.isna(volatility_correlation) else 0.0
            else:
                features['volatility_correlation'] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"计算市场相关性因子失败: {e}")
            return {
                'volume_correlation': 0.0,
                'corr_5d': 0.0,
                'volatility_correlation': 0.0
            }

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
                self._update_target_market_corr_features(target_data, stock_target_features, stock_target_mask)

    def _update_target_market_corr_features(self, target_data: pd.DataFrame,
                                          stock_target_features: dict,
                                          stock_target_mask: pd.Series):
        """更新目标数据中的市场相关性特征"""
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
            self.logger.error(f"更新目标市场相关性特征失败: {e}")

    def _validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """验证市场数据的有效性"""
        try:
            if market_data is None or market_data.empty:
                self.logger.warning("市场数据为空")
                return False

            # 检查必需的列
            required_cols = ['date', 'daily_change_pct', 'volume']
            missing_cols = [col for col in required_cols if col not in market_data.columns]

            if missing_cols:
                self.logger.warning(f"市场数据缺少必需列: {missing_cols}")
                return False

            # 检查数据量
            if len(market_data) < 20:
                self.logger.warning(f"市场数据量不足: {len(market_data)} < 20")
                return False

            # 检查数据质量
            null_counts = market_data[required_cols].isnull().sum()
            if null_counts.any():
                self.logger.warning(f"市场数据存在空值: {null_counts.to_dict()}")

            return True

        except Exception as e:
            self.logger.error(f"验证市场数据失败: {e}")
            return False

    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> tuple:
        """识别目标日期和历史数据范围（参照enhanced_correlation_factors实现）"""
        try:
            # 🔧 修复：正确处理target_start_date和target_end_date参数
            target_start_date = kwargs.get('target_start_date')
            target_end_date = kwargs.get('target_end_date')

            if target_start_date and target_end_date:
                # 使用指定的目标日期范围
                target_dates = self._get_trading_dates_in_range(data, target_start_date, target_end_date)
                self.logger.info(f"使用传递的目标日期范围: {target_start_date} - {target_end_date}")
            else:
                # 从kwargs中获取目标日期列表
                target_dates = kwargs.get('target_dates', [])

                if not target_dates:
                    # 如果没有指定目标日期，使用数据中的所有日期
                    unique_dates = data['date'].unique()
                    target_dates = [pd.to_datetime(date).strftime('%Y%m%d') for date in unique_dates]
                    target_dates = sorted(target_dates)
                    self.logger.info("使用数据中的所有日期作为目标日期")

            if not target_dates:
                self.logger.error(f"数据中不包含指定的目标日期范围: {target_start_date} - {target_end_date}")
                return [], []

            # 🔧 修复：历史数据范围是所有可用日期（包括目标日期之前的历史数据）
            all_dates = data['date'].unique()
            historical_dates = [pd.to_datetime(date).strftime('%Y%m%d') for date in all_dates]
            historical_dates = sorted(historical_dates)

            self.logger.info(f"目标日期识别完成: {len(target_dates)} 个目标日期, {len(historical_dates)} 个历史日期")
            return target_dates, historical_dates

        except Exception as e:
            self.logger.error(f"识别目标和历史日期失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return [], []

    def _get_trading_dates_in_range(self, data: pd.DataFrame, start_date: str, end_date: str) -> List[str]:
        """获取指定日期范围内的交易日"""
        try:
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            end_dt = pd.to_datetime(end_date, format='%Y%m%d')

            # 过滤数据中在指定范围内的日期
            mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
            trading_dates = data[mask]['date'].unique()

            # 转换为字符串格式并排序
            trading_dates_str = [pd.to_datetime(date).strftime('%Y%m%d') for date in trading_dates]
            return sorted(trading_dates_str)

        except Exception as e:
            self.logger.error(f"获取交易日期范围失败: {e}")
            return []

    def get_default_features(self) -> dict:
        """获取所有因子的默认值"""
        return {
            'volume_correlation': 0.0,      # 成交量相关性默认值
            'corr_5d': 0.0,                 # 5日价格相关性默认值
            'volatility_correlation': 0.0   # 波动率相关性默认值
        }
