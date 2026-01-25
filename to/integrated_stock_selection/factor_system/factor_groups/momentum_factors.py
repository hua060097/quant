#!/usr/bin/env python3
"""
动量因子组
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from .base_factor_group import BaseFactorGroup

class MomentumFactors(BaseFactorGroup):
    def __init__(self):
        super().__init__("momentum_factors")
    
    def _get_version(self) -> str:
        return "v2.5"
    
    def _get_dependencies(self) -> List[str]:
        return []
    
    def _get_feature_names(self) -> List[str]:
        return [
            'momentum_quality_score', 'cumulative_return_3d', 'cumulative_return_5d',
            'cumulative_return_10d', 'price_momentum'
        ]

    def _get_max_history_window_needed(self) -> int:
        """返回计算动量因子所需的最大历史窗口长度"""
        return 20  # 动量因子需要20日历史数据
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算动量因子（基于feature_calculator.py的专业实现）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return self._set_default_values(data)
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_momentum_features_optimized(data, target_dates, historical_dates)

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            return result_data

        except Exception as e:
            self.logger.error(f"动量因子计算失败: {e}")
            return self._set_default_values(data)

    def _calculate_momentum_features_optimized(self, full_data: pd.DataFrame,
                                             target_dates: List[str],
                                             historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化动量特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意动量计算中的时间窗口边界，确保不使用未来信息
        """
        try:
            self.logger.info("开始正确优化的动量特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_momentum_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_momentum_features_with_history(
                full_data, result_data, target_dates
            )

            self.logger.info("正确优化的动量特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的动量特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_momentum_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化动量特征列（内存优化）"""
        try:
            # 动量特征
            result_data['cumulative_return_3d'] = 0.0
            result_data['cumulative_return_5d'] = 0.0
            result_data['cumulative_return_10d'] = 0.0
            result_data['price_momentum'] = 0.0
            result_data['momentum_quality_score'] = 50.0

        except Exception as e:
            self.logger.error(f"初始化目标动量特征失败: {e}")

    def _calculate_target_momentum_features_with_history(self, full_data: pd.DataFrame,
                                                       target_data: pd.DataFrame,
                                                       target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的动量特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的动量特征（使用历史数据支撑）...")

            # 按股票分组处理
            target_stocks = target_data['stock_code'].unique()
            total_stocks = len(target_stocks)

            self.logger.info(f"准备处理 {total_stocks} 只股票的动量特征")

            for idx, stock_code in enumerate(target_stocks):
                if idx % max(1, total_stocks // 10) == 0 or idx == total_stocks - 1:
                    progress_pct = (idx + 1) / total_stocks * 100
                    self.logger.info(f"处理进度: {idx + 1}/{total_stocks} ({progress_pct:.1f}%)")

                # 获取该股票的完整历史数据
                stock_full_data = full_data[full_data['stock_code'] == stock_code].sort_values('date')

                # 获取该股票的目标日期数据
                stock_target_mask = target_data['stock_code'] == stock_code

                if len(stock_full_data) < 20:  # 需要足够的历史数据计算动量特征
                    continue

                # 为该股票的目标日期计算动量特征
                stock_target_features = self._calculate_stock_target_momentum_features(
                    stock_full_data, target_dates
                )

                # 更新目标数据
                self._update_target_momentum_features(target_data, stock_target_features, stock_target_mask)

            self.logger.info("目标日期动量特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期动量特征计算失败: {e}")
            return target_data

    def _calculate_stock_target_momentum_features(self, stock_full_data: pd.DataFrame,
                                                target_dates: List[str]) -> dict:
        """为单只股票的目标日期计算动量特征"""
        try:
            target_features = {}

            # 预处理数据
            if 'pct_chg' not in stock_full_data.columns or 'close' not in stock_full_data.columns:
                return {}

            pct_chg = stock_full_data['pct_chg'].values / 100.0  # 转换为小数
            close = stock_full_data['close'].values

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
                features = self._calculate_momentum_features_for_single_date(
                    pct_chg, close, data_idx
                )

                target_features[target_date_obj] = features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标动量特征失败: {e}")
            return {}

    def _calculate_momentum_features_for_single_date(self, pct_chg: np.ndarray,
                                                   close: np.ndarray,
                                                   data_idx: int) -> dict:
        """为单个目标日期计算所有动量特征"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 20:
                return self._get_default_momentum_features()

            # 1. 累计收益率特征（使用历史数据支撑）
            for period in [3, 5, 10]:
                cumulative_return = self._calculate_cumulative_return_for_date(pct_chg, data_idx, period)
                features[f'cumulative_return_{period}d'] = cumulative_return

            # 2. 价格动量（20日价格变化率）
            if data_idx >= 20:
                price_momentum = (close[data_idx] - close[data_idx - 20]) / close[data_idx - 20]
                features['price_momentum'] = price_momentum
            else:
                features['price_momentum'] = 0.0

            # 3. 动量质量评分
            momentum_quality = self._calculate_momentum_quality_for_date(features)
            features['momentum_quality_score'] = momentum_quality

            return features

        except Exception as e:
            self.logger.error(f"计算单日期动量特征失败: {e}")
            return self._get_default_momentum_features()

    def _calculate_cumulative_return_for_date(self, pct_chg: np.ndarray,
                                            data_idx: int,
                                            period: int) -> float:
        """为单个目标日期计算累计收益率"""
        try:
            # 获取窗口数据（确保不使用未来信息）
            start_idx = max(0, data_idx - period + 1)
            end_idx = data_idx + 1  # 包含目标日期

            if end_idx - start_idx < period:
                return 0.0

            # 计算累计收益率
            returns_window = pct_chg[start_idx:end_idx]
            cumulative_return = np.prod(1 + returns_window) - 1

            return float(cumulative_return)

        except Exception as e:
            self.logger.error(f"计算累计收益率失败: {e}")
            return 0.0

    def _calculate_momentum_quality_for_date(self, features: dict) -> float:
        """为单个目标日期计算动量质量评分"""
        try:
            # 基于多个动量指标的综合评分
            momentum_components = [
                ('cumulative_return_3d', features.get('cumulative_return_3d', 0.0), 0.2),
                ('cumulative_return_5d', features.get('cumulative_return_5d', 0.0), 0.3),
                ('cumulative_return_10d', features.get('cumulative_return_10d', 0.0), 0.3),
                ('price_momentum', features.get('price_momentum', 0.0), 0.2)
            ]

            weighted_score = 0
            total_weight = 0

            for _, values, weight in momentum_components:
                # 将收益率转换为评分（0-100）
                score = 50 + values * 100  # 基础分50，收益率影响±50分
                score = max(0, min(100, score))  # 限制在0-100范围

                weighted_score += score * weight
                total_weight += weight

            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 50.0

            return float(final_score)

        except Exception as e:
            self.logger.error(f"计算动量质量评分失败: {e}")
            return 50.0

    def _get_default_momentum_features(self) -> dict:
        """获取默认的动量特征值"""
        return {
            'cumulative_return_3d': 0.0,
            'cumulative_return_5d': 0.0,
            'cumulative_return_10d': 0.0,
            'price_momentum': 0.0,
            'momentum_quality_score': 50.0
        }

    def _update_target_momentum_features(self, target_data: pd.DataFrame,
                                       stock_target_features: dict,
                                       stock_target_mask: pd.Series):
        """更新目标数据中的动量特征"""
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
            self.logger.error(f"更新目标动量特征失败: {e}")

    def _calculate_cumulative_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算累计涨幅特征（从feature_calculator.py迁移）"""
        try:
            self.logger.debug("开始计算累计涨幅特征")

            # 向量化计算累计涨幅（替代逐股票循环）
            self.logger.debug("开始向量化计算累计涨幅特征")

            # 确保数据按股票和日期排序
            data = data.sort_values(['stock_code', 'date'])

            # 计算累计涨幅
            if 'pct_chg' in data.columns:
                pct_chg_decimal = data['pct_chg'].fillna(0) / 100  # 转换为小数

                # 内存优化：分批计算不同周期的累计收益率
                periods = [3, 5, 10]

                # 按股票分组，避免创建大数组
                unique_stocks = data['stock_code'].unique()

                for period in periods:
                    cumulative_returns = []

                    # 分批处理股票，避免内存溢出
                    batch_size = 1000  # 每批处理1000只股票
                    for i in range(0, len(unique_stocks), batch_size):
                        batch_stocks = unique_stocks[i:i + batch_size]
                        batch_mask = data['stock_code'].isin(batch_stocks)
                        batch_data = data[batch_mask].copy()

                        # 计算该批次的累计收益率
                        batch_pct_chg = pct_chg_decimal[batch_mask]

                        def calculate_cumulative_return_batch(group):
                            returns = group + 1  # 转换为(1+r)形式
                            # 使用更内存友好的计算方式
                            cum_returns = returns.rolling(window=period, min_periods=period).apply(
                                lambda x: (x.prod() - 1) if len(x) == period else 0, raw=True
                            )
                            return cum_returns.fillna(0)

                        batch_result = (
                            batch_pct_chg.groupby(batch_data['stock_code'])
                            .apply(calculate_cumulative_return_batch)
                            .reset_index(0, drop=True)
                        )

                        cumulative_returns.append(batch_result)

                        # 清理内存
                        del batch_data, batch_pct_chg, batch_result

                    # 合并所有批次的结果
                    if cumulative_returns:
                        data[f'cumulative_return_{period}d'] = pd.concat(cumulative_returns, ignore_index=False)
                    else:
                        data[f'cumulative_return_{period}d'] = 0

            # 计算价格动量和盈利动量
            if 'close' in data.columns:
                data['price_momentum'] = data.groupby('stock_code')['close'].pct_change(20).fillna(0)
            else:
                data['price_momentum'] = 0

            self.logger.debug("累计涨幅特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"累计涨幅特征计算失败: {e}")
            return self._set_default_values(data)


    def _calculate_momentum_quality_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算动量质量评分"""
        try:
            # 初始化动量质量评分
            data['momentum_quality_score'] = 50.0  # 默认中性评分

            # 基于多个动量指标的综合评分
            momentum_components = []

            # 1. 短期动量（权重30%）
            if 'cumulative_return_3d' in data.columns:
                short_momentum = np.clip(data['cumulative_return_3d'] * 100, -10, 10)  # 转换为百分比并限制范围
                momentum_components.append(('short_momentum', short_momentum, 0.3))

            # 2. 中期动量（权重40%）
            if 'cumulative_return_10d' in data.columns:
                medium_momentum = np.clip(data['cumulative_return_10d'] * 100, -20, 20)
                momentum_components.append(('medium_momentum', medium_momentum, 0.4))

            # 3. 价格动量（权重20%）
            if 'price_momentum' in data.columns:
                price_mom = np.clip(data['price_momentum'] * 100, -30, 30)
                momentum_components.append(('price_momentum', price_mom, 0.2))

            # 4. 动量稳定性（权重10%）
            if 'cumulative_return_3d' in data.columns and 'cumulative_return_10d' in data.columns:
                # 短期和中期动量的一致性
                momentum_consistency = np.where(
                    (data['cumulative_return_3d'] * data['cumulative_return_10d']) > 0,
                    5,  # 方向一致，加分
                    -5  # 方向不一致，减分
                )
                momentum_components.append(('momentum_consistency', momentum_consistency, 0.1))

            # 计算综合动量评分
            if momentum_components:
                weighted_score = 0
                total_weight = 0

                for name, values, weight in momentum_components:
                    weighted_score += values * weight
                    total_weight += weight

                if total_weight > 0:
                    # 标准化到0-100分
                    normalized_score = 50 + weighted_score  # 基准分50分
                    data['momentum_quality_score'] = np.clip(normalized_score, 0, 100)

            self.logger.debug("动量质量评分计算完成")
            return data

        except Exception as e:
            self.logger.error(f"动量质量评分计算失败: {e}")
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

            # 2. 智能确定历史支撑数据范围（基于因子需要的最大历史窗口）
            max_history_needed = self._get_max_history_window_needed()
            historical_dates = self._get_optimal_historical_dates(
                data, target_dates, max_history_needed
            )

            self.logger.info(f"目标日期识别完成: {len(target_dates)} 个目标日期, {len(historical_dates)} 个历史日期")
            self.logger.info(f"最大历史窗口需求: {max_history_needed} 个交易日")

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
