#!/usr/bin/env python3
"""
横截面因子组
"""

import pandas as pd
import numpy as np
from typing import List
from .base_factor_group import BaseFactorGroup

class CrossSectionalFactors(BaseFactorGroup):
    def __init__(self):
        super().__init__("cross_sectional_factors")
    
    def _get_version(self) -> str:
        return "v1.6"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接使用原始数据计算横截面排名特征
    
    def _get_feature_names(self) -> List[str]:
        return [
            'rank_pct_chg', 'rank_volume', 'rank_turnover', 'rank_amplitude',
            'rank_momentum_5d', 'rank_momentum_20d', 'rank_rsi', 'rank_macd'
        ]
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算横截面因子（基于feature_calculator.py的专业实现）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return self._set_default_values(data)
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_cross_sectional_features_optimized(data, target_dates, historical_dates)

            # 4. 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            self.logger.info("横截面因子计算完成（修复版）")
            return result_data

        except Exception as e:
            self.logger.error(f"横截面因子计算失败: {e}")
            return data

    def _calculate_cross_sectional_features_optimized(self, full_data: pd.DataFrame,
                                                    target_dates: List[str],
                                                    historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化横截面特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        """
        try:
            self.logger.info("开始正确优化的横截面特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_cross_sectional_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_cross_sectional_features_with_history(
                full_data, result_data, target_dates
            )

            self.logger.info("正确优化的横截面特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的横截面特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_cross_sectional_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化横截面特征列（内存优化）"""
        try:
            # 横截面排名特征
            result_data['rank_pct_chg'] = 0.5
            result_data['rank_volume'] = 0.5
            result_data['rank_turnover'] = 0.5
            result_data['rank_amplitude'] = 0.5
            result_data['rank_momentum_5d'] = 0.5
            result_data['rank_momentum_20d'] = 0.5
            result_data['rank_rsi'] = 0.5
            result_data['rank_macd'] = 0.5

        except Exception as e:
            self.logger.error(f"初始化目标横截面特征失败: {e}")

    def _calculate_target_cross_sectional_features_with_history(self, full_data: pd.DataFrame,
                                                              target_data: pd.DataFrame,
                                                              target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的横截面特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的横截面特征（使用历史数据支撑）...")

            # 转换目标日期为datetime格式
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]

            # 为每个目标日期逐一计算横截面排名
            for target_date_obj in target_date_objects:
                # 获取该目标日期的数据
                target_day_mask = target_data['date'] == target_date_obj
                target_day_data = target_data[target_day_mask]

                if len(target_day_data) < 5:  # 至少需要5只股票进行排名
                    self.logger.warning(f"目标日期 {target_date_obj} 股票数量不足: {len(target_day_data)}")
                    continue

                # 为该目标日期计算横截面排名特征
                target_day_features = self._calculate_cross_sectional_ranks_for_single_date(
                    full_data, target_day_data, target_date_obj
                )

                # 更新目标数据
                self._update_target_cross_sectional_features(target_data, target_day_features, target_day_mask)

            self.logger.info("目标日期横截面特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期横截面特征计算失败: {e}")
            return target_data

    def _calculate_cross_sectional_ranks_for_single_date(self, full_data: pd.DataFrame,
                                                       target_day_data: pd.DataFrame,
                                                       target_date_obj: pd.Timestamp) -> dict:
        """为单个目标日期计算横截面排名特征"""
        try:
            features = {}

            # 获取该日期的所有股票数据（用于横截面排名）
            date_mask = full_data['date'] == target_date_obj
            all_stocks_data = full_data[date_mask]

            if len(all_stocks_data) < 5:
                self.logger.warning(f"日期 {target_date_obj} 股票数量不足: {len(all_stocks_data)}")
                return self._get_default_cross_sectional_features()

            # 1. rank_pct_chg: 涨跌幅排名
            if 'pct_chg' in all_stocks_data.columns:
                features['rank_pct_chg'] = all_stocks_data['pct_chg'].rank(pct=True, method='min')
            else:
                features['rank_pct_chg'] = pd.Series(0.5, index=all_stocks_data.index)

            # 2. rank_volume: 成交量排名
            if 'volume' in all_stocks_data.columns:
                features['rank_volume'] = all_stocks_data['volume'].rank(pct=True, method='min')
            else:
                features['rank_volume'] = pd.Series(0.5, index=all_stocks_data.index)

            # 3. rank_turnover: 换手率排名
            if 'turnover_rate' in all_stocks_data.columns:
                features['rank_turnover'] = all_stocks_data['turnover_rate'].rank(pct=True, method='min')
            else:
                features['rank_turnover'] = pd.Series(0.5, index=all_stocks_data.index)

            # 4. rank_amplitude: 振幅排名
            if all(col in all_stocks_data.columns for col in ['high', 'low', 'close']):
                amplitude = (all_stocks_data['high'] - all_stocks_data['low']) / all_stocks_data['close']
                features['rank_amplitude'] = amplitude.rank(pct=True, method='min')
            else:
                features['rank_amplitude'] = pd.Series(0.5, index=all_stocks_data.index)

            # 5. rank_momentum_5d: 5日动量排名（需要历史数据支撑）
            momentum_5d = self._calculate_momentum_for_date(full_data, target_date_obj, 5)
            if momentum_5d is not None and len(momentum_5d) > 0:
                features['rank_momentum_5d'] = momentum_5d.rank(pct=True, method='min')
            else:
                features['rank_momentum_5d'] = pd.Series(0.5, index=all_stocks_data.index)

            # 6. rank_momentum_20d: 20日动量排名（需要历史数据支撑）
            momentum_20d = self._calculate_momentum_for_date(full_data, target_date_obj, 20)
            if momentum_20d is not None and len(momentum_20d) > 0:
                features['rank_momentum_20d'] = momentum_20d.rank(pct=True, method='min')
            else:
                features['rank_momentum_20d'] = pd.Series(0.5, index=all_stocks_data.index)

            # 7. rank_rsi: RSI排名
            if 'rsi6' in all_stocks_data.columns:
                features['rank_rsi'] = all_stocks_data['rsi6'].rank(pct=True, method='min')
            elif 'rsi14' in all_stocks_data.columns:
                features['rank_rsi'] = all_stocks_data['rsi14'].rank(pct=True, method='min')
            else:
                features['rank_rsi'] = pd.Series(0.5, index=all_stocks_data.index)

            # 8. rank_macd: MACD排名
            if 'macd' in all_stocks_data.columns:
                features['rank_macd'] = all_stocks_data['macd'].rank(pct=True, method='min')
            else:
                features['rank_macd'] = pd.Series(0.5, index=all_stocks_data.index)

            return features

        except Exception as e:
            self.logger.error(f"计算单日期横截面排名失败: {e}")
            return self._get_default_cross_sectional_features()

    def _calculate_momentum_for_date(self, full_data: pd.DataFrame, target_date_obj: pd.Timestamp, period: int) -> pd.Series:
        """为指定日期计算动量指标（使用历史数据支撑）"""
        try:
            # 获取该日期的所有股票
            date_mask = full_data['date'] == target_date_obj
            target_stocks = full_data[date_mask]['stock_code'].unique()

            momentum_values = {}

            for stock_code in target_stocks:
                # 获取该股票的历史数据
                stock_mask = full_data['stock_code'] == stock_code
                stock_data = full_data[stock_mask & (full_data['date'] <= target_date_obj)].sort_values('date')

                if len(stock_data) >= period:
                    # 计算period日收益率
                    recent_data = stock_data.tail(period)
                    if len(recent_data) >= 2:
                        start_price = recent_data['close'].iloc[0]
                        end_price = recent_data['close'].iloc[-1]
                        momentum = (end_price - start_price) / start_price if start_price != 0 else 0.0
                        momentum_values[stock_code] = momentum
                    else:
                        momentum_values[stock_code] = 0.0
                else:
                    momentum_values[stock_code] = 0.0

            # 转换为Series，索引对应目标日期的数据索引
            target_day_data = full_data[date_mask]
            momentum_series = pd.Series(index=target_day_data.index, dtype=float)

            for idx, row in target_day_data.iterrows():
                stock_code = row['stock_code']
                momentum_series[idx] = momentum_values.get(stock_code, 0.0)

            return momentum_series

        except Exception as e:
            self.logger.error(f"计算动量指标失败: {e}")
            return None

    def _get_default_cross_sectional_features(self) -> dict:
        """获取默认的横截面特征值"""
        return {
            'rank_pct_chg': 0.5,
            'rank_volume': 0.5,
            'rank_turnover': 0.5,
            'rank_amplitude': 0.5,
            'rank_momentum_5d': 0.5,
            'rank_momentum_20d': 0.5,
            'rank_rsi': 0.5,
            'rank_macd': 0.5
        }

    def _update_target_cross_sectional_features(self, target_data: pd.DataFrame,
                                              target_day_features: dict,
                                              target_day_mask: pd.Series):
        """更新目标数据中的横截面特征"""
        try:
            for feature_name, feature_series in target_day_features.items():
                if feature_name in target_data.columns and isinstance(feature_series, pd.Series):
                    # 只更新目标日期的数据
                    for idx in target_data[target_day_mask].index:
                        if idx in feature_series.index:
                            target_data.loc[idx, feature_name] = feature_series[idx]

        except Exception as e:
            self.logger.error(f"更新目标横截面特征失败: {e}")

    def _calculate_cross_sectional_ranks(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算横截面排名特征（内存优化版）"""
        try:
            self.logger.debug("开始计算横截面排名特征（内存优化）")

            result_data = data.copy()

            # 初始化所有排名特征
            rank_features = ['rank_pct_chg', 'rank_volume', 'rank_turnover', 'rank_amplitude',
                           'rank_momentum_5d', 'rank_momentum_20d', 'rank_rsi', 'rank_macd']
            for feature in rank_features:
                result_data[feature] = 0.5  # 默认值

            # 内存优化：分批处理日期
            unique_dates = result_data['date'].unique()
            batch_size = 50  # 每批处理50个日期

            for i in range(0, len(unique_dates), batch_size):
                batch_dates = unique_dates[i:i + batch_size]

                for date in batch_dates:
                    date_mask = result_data['date'] == date
                    date_data = result_data[date_mask]

                    if len(date_data) < 10:  # 至少需要10只股票
                        continue

                    # 1. rank_pct_chg: 涨跌幅排名
                    if 'pct_chg' in date_data.columns:
                        rank_pct_chg = date_data['pct_chg'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_pct_chg'] = rank_pct_chg

                    # 2. rank_volume: 成交量排名
                    if 'volume' in date_data.columns:
                        rank_volume = date_data['volume'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_volume'] = rank_volume

                    # 3. rank_turnover: 换手率排名
                    if 'turnover_rate' in date_data.columns:
                        rank_turnover = date_data['turnover_rate'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_turnover'] = rank_turnover

                    # 4. rank_amplitude: 振幅排名
                    if all(col in date_data.columns for col in ['high', 'low', 'close']):
                        amplitude = (date_data['high'] - date_data['low']) / date_data['close']
                        rank_amplitude = amplitude.rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_amplitude'] = rank_amplitude

                    # 5. rank_momentum_5d: 5日动量排名
                    if 'cumulative_return_5d' in date_data.columns:
                        rank_momentum_5d = date_data['cumulative_return_5d'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_momentum_5d'] = rank_momentum_5d
                    elif 'return_5d' in date_data.columns:
                        rank_momentum_5d = date_data['return_5d'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_momentum_5d'] = rank_momentum_5d

                    # 6. rank_momentum_20d: 20日动量排名
                    if 'return_20d' in date_data.columns:
                        rank_momentum_20d = date_data['return_20d'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_momentum_20d'] = rank_momentum_20d

                    # 7. rank_rsi: RSI排名
                    if 'rsi6' in date_data.columns:
                        rank_rsi = date_data['rsi6'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_rsi'] = rank_rsi
                    elif 'rsi14' in date_data.columns:
                        rank_rsi = date_data['rsi14'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_rsi'] = rank_rsi

                    # 8. rank_macd: MACD排名
                    if 'macd' in date_data.columns:
                        rank_macd = date_data['macd'].rank(pct=True, method='min')
                        result_data.loc[date_mask, 'rank_macd'] = rank_macd

                # 定期垃圾回收
                if i % 100 == 0:  # 每处理100个日期批次进行一次垃圾回收
                    import gc
                    gc.collect()

            # 填充缺失值
            rank_features = [
                'rank_pct_chg', 'rank_volume', 'rank_turnover', 'rank_amplitude',
                'rank_momentum_5d', 'rank_momentum_20d', 'rank_rsi', 'rank_macd'
            ]

            for feature in rank_features:
                if feature in result_data.columns:
                    result_data[feature] = result_data[feature].fillna(0.5)  # 用中位数填充

            self.logger.debug("横截面排名特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"横截面排名特征计算失败: {e}")
            return data

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
            if not target_dates:
                return data

            # 转换日期格式进行过滤
            date_mask = data['date'].dt.strftime('%Y%m%d').isin(target_dates)
            filtered_data = data[date_mask].copy()

            self.logger.info(f"过滤结果: {len(data)} -> {len(filtered_data)} 条记录")
            return filtered_data
        except Exception as e:
            self.logger.error(f"过滤结果失败: {e}")
            return data

    def _set_default_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """设置默认值"""
        for factor_name in self.feature_names:
            data[factor_name] = 0.0
        return data
