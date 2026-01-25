#!/usr/bin/env python3
"""
成交量因子组
"""

import pandas as pd
import numpy as np
from typing import List
from .base_factor_group import BaseFactorGroup

class VolumeFactors(BaseFactorGroup):
    def __init__(self):
        super().__init__("volume_factors")
    
    def _get_version(self) -> str:
        return "v1.9"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接使用原始数据计算成交量特征
    
    def _get_feature_names(self) -> List[str]:
        return [
            'volume_ratio', 'turnover_rate', 'volume_price_trend',
            'volume_weighted_return', 'volume_momentum'
        ]

    def _get_max_history_window_needed(self) -> int:
        """返回计算成交量因子所需的最大历史窗口长度"""
        return 60  # 成交量因子需要60日历史数据
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算成交量因子（基于feature_calculator.py的专业实现）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return self._set_default_values(data)
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")
            
            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_volume_features_optimized(data, target_dates, historical_dates)

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            return result_data

        except Exception as e:
            self.logger.error(f"成交量因子计算失败: {e}")
            return self._set_default_values(data)

    def _calculate_volume_features_optimized(self, full_data: pd.DataFrame,
                                           target_dates: List[str],
                                           historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化成交量特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意成交量计算中的时间窗口边界，确保不使用未来信息
        """
        try:
            self.logger.info("开始正确优化的成交量特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_volume_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_volume_features_with_history(
                full_data, result_data, target_dates
            )

            self.logger.info("正确优化的成交量特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的成交量特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_volume_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化成交量特征列（内存优化）"""
        try:
            # 成交量特征
            result_data['volume_ratio'] = 1.0
            result_data['turnover_rate'] = 0.0
            result_data['volume_price_trend'] = 0.0
            result_data['volume_weighted_return'] = 0.0
            result_data['volume_momentum'] = 0.0

        except Exception as e:
            self.logger.error(f"初始化目标成交量特征失败: {e}")

    def _calculate_target_volume_features_with_history(self, full_data: pd.DataFrame,
                                                     target_data: pd.DataFrame,
                                                     target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的成交量特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的成交量特征（使用历史数据支撑）...")

            # 1. 计算简单成交量特征（只为目标日期，基于当日数据）
            target_data = self._calculate_target_simple_volume_features(target_data)

            # 2. 计算复杂成交量特征（只为目标日期，使用历史数据支撑）
            target_data = self._calculate_target_complex_volume_features(
                full_data, target_data, target_dates
            )

            self.logger.info("目标日期成交量特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期成交量特征计算失败: {e}")
            return target_data

    def _calculate_target_simple_volume_features(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """为目标日期计算简单成交量特征（基于当日数据）"""
        try:
            # 1. turnover_rate: 换手率（基于当日数据）
            if 'volume' in target_data.columns and 'circulating_market_cap' in target_data.columns:
                # 换手率 = 成交量 / 流通股本
                circulating_cap_safe = np.where(
                    target_data['circulating_market_cap'] > 0,
                    target_data['circulating_market_cap'],
                    1e9  # 默认10亿流通市值
                )
                target_data['turnover_rate'] = target_data['volume'] / circulating_cap_safe
                target_data['turnover_rate'] = np.clip(target_data['turnover_rate'], 0, 1)
                self.logger.info("✅ turnover_rate计算完成（基于当日数据）")
            else:
                target_data['turnover_rate'] = 0.0
                self.logger.warning("⚠️ 缺少成交量或流通市值字段，turnover_rate使用默认值0.0")

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标简单成交量特征失败: {e}")
            return target_data

    def _calculate_target_complex_volume_features(self, full_data: pd.DataFrame,
                                                target_data: pd.DataFrame,
                                                target_dates: List[str]) -> pd.DataFrame:
        """为目标日期计算复杂成交量特征（使用历史数据支撑）"""
        try:
            # 按股票分组处理
            target_stocks = target_data['stock_code'].unique()
            total_stocks = len(target_stocks)

            self.logger.info(f"准备处理 {total_stocks} 只股票的复杂成交量特征")

            for idx, stock_code in enumerate(target_stocks):
                if idx % max(1, total_stocks // 10) == 0 or idx == total_stocks - 1:
                    progress_pct = (idx + 1) / total_stocks * 100
                    self.logger.info(f"处理进度: {idx + 1}/{total_stocks} ({progress_pct:.1f}%)")

                # 获取该股票的完整历史数据
                stock_full_data = full_data[full_data['stock_code'] == stock_code].sort_values('date')

                # 获取该股票的目标日期数据
                stock_target_mask = target_data['stock_code'] == stock_code

                if len(stock_full_data) < 60:  # 需要足够的历史数据计算成交量特征
                    continue

                # 为该股票的目标日期计算复杂成交量特征
                stock_target_features = self._calculate_stock_target_volume_features(
                    stock_full_data, target_dates
                )

                # 更新目标数据
                self._update_target_volume_features(target_data, stock_target_features, stock_target_mask)

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标复杂成交量特征失败: {e}")
            return target_data

    def _calculate_stock_target_volume_features(self, stock_full_data: pd.DataFrame,
                                              target_dates: List[str]) -> dict:
        """为单只股票的目标日期计算复杂成交量特征"""
        try:
            target_features = {}

            # 预处理数据
            required_columns = ['volume', 'close', 'pct_chg']
            if not all(col in stock_full_data.columns for col in required_columns):
                return {}

            volume = stock_full_data['volume'].values
            close = stock_full_data['close'].values
            pct_chg = stock_full_data['pct_chg'].values / 100.0  # 转换为小数

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
                features = self._calculate_volume_features_for_single_date(
                    volume, close, pct_chg, data_idx
                )

                target_features[target_date_obj] = features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标成交量特征失败: {e}")
            return {}

    def _calculate_volume_features_for_single_date(self, volume: np.ndarray,
                                                 close: np.ndarray,
                                                 pct_chg: np.ndarray,
                                                 data_idx: int) -> dict:
        """为单个目标日期计算所有复杂成交量特征"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 60:
                return self._get_default_volume_features()

            # 1. volume_ratio: 成交量比率（相对于历史平均）
            features['volume_ratio'] = self._calculate_volume_ratio_for_date(volume, data_idx)

            # 2. volume_price_trend: 量价趋势
            features['volume_price_trend'] = self._calculate_volume_price_trend_for_date(
                volume, close, data_idx
            )

            # 3. volume_weighted_return: 成交量加权收益率
            features['volume_weighted_return'] = self._calculate_volume_weighted_return_for_date(
                volume, pct_chg, data_idx
            )

            # 4. volume_momentum: 成交量动量
            features['volume_momentum'] = self._calculate_volume_momentum_for_date(
                volume, data_idx
            )

            return features

        except Exception as e:
            self.logger.error(f"计算单日期成交量特征失败: {e}")
            return self._get_default_volume_features()

    def _calculate_volume_ratio_for_date(self, volume: np.ndarray, data_idx: int) -> float:
        """为单个目标日期计算成交量比率"""
        try:
            # 获取20日窗口数据（确保不使用未来信息）
            window_size = 20
            start_idx = max(0, data_idx - window_size + 1)
            end_idx = data_idx + 1  # 包含目标日期

            if end_idx - start_idx < window_size:
                return 1.0

            # 计算历史平均成交量（不包含当日）
            historical_volume = volume[start_idx:data_idx]
            if len(historical_volume) > 0:
                avg_volume = np.mean(historical_volume)
                if avg_volume > 0:
                    ratio = volume[data_idx] / avg_volume
                    return float(np.clip(ratio, 0.1, 10.0))

            return 1.0

        except Exception as e:
            self.logger.error(f"计算成交量比率失败: {e}")
            return 1.0

    def _calculate_volume_price_trend_for_date(self, volume: np.ndarray,
                                             close: np.ndarray,
                                             data_idx: int) -> float:
        """为单个目标日期计算量价趋势"""
        try:
            # 获取10日窗口数据（确保不使用未来信息）
            window_size = 10
            start_idx = max(0, data_idx - window_size + 1)
            end_idx = data_idx + 1  # 包含目标日期

            if end_idx - start_idx < window_size:
                return 0.0

            # 计算价格和成交量的相关性
            price_window = close[start_idx:end_idx]
            volume_window = volume[start_idx:end_idx]

            if len(price_window) >= 5:
                # 计算价格变化和成交量变化的相关性
                price_changes = np.diff(price_window)
                volume_changes = np.diff(volume_window)

                if len(price_changes) > 0 and np.std(price_changes) > 0 and np.std(volume_changes) > 0:
                    correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                    if not np.isnan(correlation):
                        return float(np.clip(correlation, -1, 1))

            return 0.0

        except Exception as e:
            self.logger.error(f"计算量价趋势失败: {e}")
            return 0.0

    def _calculate_volume_weighted_return_for_date(self, volume: np.ndarray,
                                                 pct_chg: np.ndarray,
                                                 data_idx: int) -> float:
        """为单个目标日期计算成交量加权收益率"""
        try:
            # 获取5日窗口数据（确保不使用未来信息）
            window_size = 5
            start_idx = max(0, data_idx - window_size + 1)
            end_idx = data_idx + 1  # 包含目标日期

            if end_idx - start_idx < window_size:
                return 0.0

            # 计算成交量加权收益率
            volume_window = volume[start_idx:end_idx]
            return_window = pct_chg[start_idx:end_idx]

            total_volume = np.sum(volume_window)
            if total_volume > 0:
                weights = volume_window / total_volume
                weighted_return = np.sum(weights * return_window)
                return float(np.clip(weighted_return, -0.2, 0.2))

            return 0.0

        except Exception as e:
            self.logger.error(f"计算成交量加权收益率失败: {e}")
            return 0.0

    def _calculate_volume_momentum_for_date(self, volume: np.ndarray, data_idx: int) -> float:
        """为单个目标日期计算成交量动量"""
        try:
            # 获取20日窗口数据（确保不使用未来信息）
            window_size = 20
            start_idx = max(0, data_idx - window_size + 1)
            end_idx = data_idx + 1  # 包含目标日期

            if end_idx - start_idx < window_size:
                return 0.0

            # 计算成交量动量（近期vs远期平均成交量比率）
            volume_window = volume[start_idx:end_idx]

            # 分为前半段和后半段
            mid_point = len(volume_window) // 2
            early_avg = np.mean(volume_window[:mid_point])
            recent_avg = np.mean(volume_window[mid_point:])

            if early_avg > 0:
                momentum = (recent_avg - early_avg) / early_avg
                return float(np.clip(momentum, -2, 2))

            return 0.0

        except Exception as e:
            self.logger.error(f"计算成交量动量失败: {e}")
            return 0.0

    def _get_default_volume_features(self) -> dict:
        """获取默认的成交量特征值"""
        return {
            'volume_ratio': 1.0,
            'volume_price_trend': 0.0,
            'volume_weighted_return': 0.0,
            'volume_momentum': 0.0
        }

    def _update_target_volume_features(self, target_data: pd.DataFrame,
                                     stock_target_features: dict,
                                     stock_target_mask: pd.Series):
        """更新目标数据中的成交量特征"""
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
            self.logger.error(f"更新目标成交量特征失败: {e}")

    def _calculate_volume_relative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量相对特征（从feature_calculator.py迁移）"""
        try:
            self.logger.debug("开始计算成交量相对特征")

            # 初始化特征
            data['volume_ratio_5d'] = 1.0
            data['volume_ratio_20d'] = 1.0
            data['volume_percentile_60d'] = 0.5
            data['volatility_zscore_20d'] = 0.0
            data['volatility_percentile_60d'] = 0.5

            # 向量化计算（替代逐股票循环）
            self.logger.debug("开始向量化计算成交量和波动率特征")

            # 确保数据按股票和日期排序
            data = data.sort_values(['stock_code', 'date'])

            # 1. 向量化计算成交量比率特征
            if 'volume' in data.columns:
                # 5日平均成交量比率
                volume_ma5 = data.groupby('stock_code')['volume'].rolling(window=5, min_periods=5).mean().reset_index(0, drop=True)
                data['volume_ratio_5d'] = (data['volume'] / volume_ma5).fillna(1.0)

                # 20日平均成交量比率
                volume_ma20 = data.groupby('stock_code')['volume'].rolling(window=20, min_periods=20).mean().reset_index(0, drop=True)
                data['volume_ratio_20d'] = (data['volume'] / volume_ma20).fillna(1.0)

                # 60日成交量百分位排名
                data['volume_percentile_60d'] = (
                    data.groupby('stock_code')['volume']
                    .rolling(window=60, min_periods=60)
                    .rank(pct=True)
                    .reset_index(0, drop=True)
                    .fillna(0.5)
                )

            # 2. 向量化计算波动率特征
            if 'pct_chg' in data.columns:
                # 计算滚动波动率
                rolling_volatility = (
                    data.groupby('stock_code')['pct_chg']
                    .rolling(window=20, min_periods=20)
                    .std()
                    .reset_index(0, drop=True)
                )

                # 20日波动率Z-score
                vol_mean = rolling_volatility.groupby(data['stock_code']).rolling(window=20, min_periods=20).mean().reset_index(0, drop=True)
                vol_std = rolling_volatility.groupby(data['stock_code']).rolling(window=20, min_periods=20).std().reset_index(0, drop=True)
                data['volatility_zscore_20d'] = ((rolling_volatility - vol_mean) / vol_std.fillna(1)).fillna(0)

                # 60日波动率百分位排名
                data['volatility_percentile_60d'] = (
                    rolling_volatility.groupby(data['stock_code'])
                    .rolling(window=60, min_periods=60)
                    .rank(pct=True)
                    .reset_index(0, drop=True)
                    .fillna(0.5)
                )

            # 计算其他成交量相关特征
            self._calculate_additional_volume_features(data)

            self.logger.debug("成交量相对特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"成交量相对特征计算失败: {e}")
            return self._set_default_values(data)

    def _calculate_additional_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算额外的成交量特征"""
        try:
            # 1. volume_ratio: 当前成交量与平均成交量的比率
            if 'volume' in data.columns:
                avg_volume = data.groupby('stock_code')['volume'].transform(
                    lambda x: x.rolling(20, min_periods=20).mean()
                )
                # 改进除零处理
                data['volume_ratio'] = np.where(
                    avg_volume > 0,
                    data['volume'] / avg_volume,
                    1.0
                )
                data['volume_ratio'] = data['volume_ratio'].fillna(1.0)
            else:
                data['volume_ratio'] = 1.0

            # 2. turnover_rate: 换手率（如果不存在）
            if 'turnover_rate' not in data.columns:
                if 'volume' in data.columns and 'circ_mv' in data.columns:
                    # 改进的换手率计算：成交量/流通市值
                    data['turnover_rate'] = np.where(
                        (data['circ_mv'] > 0) & (data['volume'] > 0),
                        data['volume'] / data['circ_mv'] * 100,
                        0.1  # 使用小的非零值而不是1.0
                    )
                else:
                    data['turnover_rate'] = 0.1

            # 确保换手率字段存在且有合理值
            if 'turnover_rate' in data.columns:
                data['turnover_rate'] = data['turnover_rate'].fillna(0.1)
                # 限制异常值
                data['turnover_rate'] = data['turnover_rate'].clip(0, 100)
            else:
                data['turnover_rate'] = 0.1

            # 3. volume_price_trend: 量价趋势（向量化优化版）
            if 'volume' in data.columns and 'close' in data.columns:
                self.logger.debug("开始向量化计算量价趋势")

                # 计算量价相关性
                volume_change = data.groupby('stock_code')['volume'].pct_change()
                price_change = data.groupby('stock_code')['close'].pct_change()

                # 向量化计算5日滚动相关性
                def rolling_corr(group):
                    vol_chg = volume_change[group.index]
                    price_chg = price_change[group.index]
                    return vol_chg.rolling(5, min_periods=5).corr(price_chg)

                data['volume_price_trend'] = (
                    data.groupby('stock_code')
                    .apply(rolling_corr)
                    .reset_index(0, drop=True)
                    .fillna(0)
                )
            else:
                data['volume_price_trend'] = 0.0

            # 4. volume_weighted_return: 成交量加权收益率（修复版：解决索引对齐问题）
            if 'pct_chg' in data.columns and 'volume' in data.columns:
                self.logger.debug("开始计算成交量加权收益率（修复版）")

                def calculate_vw_return_fixed(group):
                    """修复版：确保返回正确的Series"""
                    returns = group['pct_chg'].values
                    volumes = group['volume'].values

                    # 计算5日滚动加权平均收益率
                    vw_returns = []
                    for i in range(len(returns)):
                        if i >= 4:  # 有足够的历史数据
                            window_returns = returns[i-4:i+1]
                            window_volumes = volumes[i-4:i+1]

                            # 计算加权平均
                            weighted_sum = np.sum(window_returns * window_volumes)
                            volume_sum = np.sum(window_volumes)

                            if volume_sum > 0:
                                vw_return = weighted_sum / volume_sum
                            else:
                                vw_return = returns[i]
                        else:
                            # 历史数据不足，使用当期收益率
                            vw_return = returns[i]

                        vw_returns.append(vw_return)

                    # 返回与原索引对齐的Series
                    return pd.Series(vw_returns, index=group.index)

                # 使用transform确保索引对齐
                data['volume_weighted_return'] = (
                    data.groupby('stock_code', group_keys=False)
                    .apply(calculate_vw_return_fixed)
                    .fillna(0)
                )
            else:
                data['volume_weighted_return'] = 0.0

            # 5. volume_momentum: 成交量动量
            if 'volume' in data.columns:
                data['volume_momentum'] = data.groupby('stock_code')['volume'].pct_change(5).fillna(0)
            else:
                data['volume_momentum'] = 0.0

            return data

        except Exception as e:
            self.logger.error(f"额外成交量特征计算失败: {e}")
            return self._set_default_values(data)

    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> tuple:
        """
        精确识别目标计算日期和历史支撑数据（完全移植sector_relative_factors的成功逻辑）

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
                    data, target_start_date, target_end_date
                )
                self.logger.info(f"使用传递的目标日期范围: {target_start_date} - {target_end_date}")
            else:
                self.logger.error("❌ 必须提供target_start_date和target_end_date参数")
                return [], []

            # 2. 时间边界检查 - 确保不包含未来日期
            if not self._validate_no_future_leakage([target_start_date, target_end_date]):
                self.logger.error("❌ 检测到时间泄露风险，目标日期包含未来日期")
                return [], []

            # 3. 获取目标日期范围
            target_dates = self._get_trading_dates_in_range(
                data, target_start_date, target_end_date
            )
            
            if not target_dates:
                self.logger.error(f"❌ 数据中不包含指定的目标日期范围: {target_start_date} - {target_end_date}")
                return [], []

            self.logger.info(f"✅ 使用明确指定的目标日期范围: {target_start_date} - {target_end_date}")

            # 使用所有可用数据作为历史数据（已通过extend_history_days=120预加载）
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

    def _get_trading_dates_in_range(self, stock_data: pd.DataFrame,
                                  start_date: str, end_date: str) -> List[str]:
        """获取指定范围内的交易日期（严格模式：只返回数据中实际存在的日期）"""
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

            # 如果数据中没有目标日期，返回空列表
            if not actual_target_dates:
                self.logger.warning(f"⚠️ 数据中不包含任何目标日期: {start_date} - {end_date}")
                return []

            return sorted(actual_target_dates)
        except Exception as e:
            self.logger.error(f"获取交易日期失败: {e}")
            return []





    def _get_optimal_historical_dates(self, data: pd.DataFrame, target_dates: List[str], max_history_needed: int) -> List[str]:
        """获取最优的历史支撑数据日期范围"""
        try:
            if not target_dates:
                return []

            all_dates = sorted(data['date'].dt.strftime('%Y%m%d').unique())
            earliest_target_index = all_dates.index(min(target_dates))
            history_start_index = max(0, earliest_target_index - max_history_needed)
            historical_dates = all_dates[history_start_index:earliest_target_index]

            return historical_dates
        except Exception as e:
            self.logger.error(f"获取历史日期失败: {e}")
            return []

    def _filter_results_to_target_dates(self, data: pd.DataFrame, target_dates: List[str]) -> pd.DataFrame:
        """只保留目标日期的计算结果"""
        try:
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            is_target_date = data['date'].isin(target_date_objects)

            # 对于非目标日期，将因子值设为默认值
            for factor_name in self.feature_names:
                if factor_name in data.columns:
                    data.loc[~is_target_date, factor_name] = 0.0

            return data
        except Exception as e:
            self.logger.error(f"过滤结果失败: {e}")
            return data

    def _set_default_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """设置默认值"""
        for factor_name in self.feature_names:
            data[factor_name] = 0.0
        return data
