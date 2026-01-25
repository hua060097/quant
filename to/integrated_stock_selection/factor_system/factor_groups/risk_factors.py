#!/usr/bin/env python3
"""
风险因子组
"""

import pandas as pd
import numpy as np
from typing import List
from .base_factor_group import BaseFactorGroup

class RiskFactors(BaseFactorGroup):
    def __init__(self):
        super().__init__("risk_factors")
    
    def _get_version(self) -> str:
        return "v2.0"  # 新增滞后收益率、滚动聚合、行为金融学和路径依赖因子
    
    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接使用原始数据计算风险特征
    
    def _get_feature_names(self) -> List[str]:
        return [
            # 原有风险因子 (删除vol_term_structure)
            'risk_adjusted_return', 'returns_zscore',
            'price_vol_change_ratio', 'high_volatility',

            # 1. 滞后收益率因子 (Lagged Return Factors)
            'pct_chg_lag1', 'pct_chg_lag2', 'pct_chg_lag3', 'pct_chg_lag4',
            'pct_chg_lag5', 'pct_chg_lag10', 'pct_chg_lag20',

            # 1.1 滞后收益率因子的全市场排名分位数
            'pct_chg_lag1_rank', 'pct_chg_lag2_rank', 'pct_chg_lag3_rank', 'pct_chg_lag4_rank',
            'pct_chg_lag5_rank', 'pct_chg_lag10_rank', 'pct_chg_lag20_rank',

            # 2. 滚动聚合特征 (Rolling Aggregates) - 删除ma_return_5d, std_return_5d, std_return_20d
            'ma_return_3d',

            # 4. 路径依赖特征 (Path-Dependent Features)
            'consecutive_positive_days', 'consecutive_negative_days',

            # 4.1 路径依赖特征的全市场排名分位数
            'consecutive_positive_days_rank', 'consecutive_negative_days_rank'
        ]

    def _get_max_history_window_needed(self) -> int:
        """返回计算风险因子所需的最大历史窗口长度"""
        return 20  # 风险因子需要20日历史数据
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算风险因子（基于feature_calculator.py的专业实现）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return self._set_default_values(data)
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_risk_features_optimized(data, target_dates, historical_dates)

            # 4. 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            self.logger.info("风险因子计算完成（修复版）")
            return result_data

        except Exception as e:
            self.logger.error(f"风险因子计算失败: {e}")
            return data

    def _calculate_risk_features_optimized(self, full_data: pd.DataFrame,
                                         target_dates: List[str],
                                         historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化风险特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意风险计算中的时间窗口边界，确保不使用未来信息
        """
        try:
            self.logger.info("开始正确优化的风险特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_risk_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_risk_features_with_history(
                full_data, result_data, target_dates
            )

            self.logger.info("正确优化的风险特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的风险特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_risk_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化风险特征列（内存优化）"""
        try:
            # 原有风险因子 (删除vol_term_structure)
            result_data['risk_adjusted_return'] = 0.0
            result_data['returns_zscore'] = 0.0
            result_data['price_vol_change_ratio'] = 0.0
            result_data['high_volatility'] = 0.0

            # 滞后收益率因子
            for lag in [1, 2, 3, 4, 5, 10, 20]:
                result_data[f'pct_chg_lag{lag}'] = 0.0
                result_data[f'pct_chg_lag{lag}_rank'] = 0.0  # 新增排名分位数

            # 滚动聚合特征 (删除ma_return_5d, std_return_5d, std_return_20d)
            result_data['ma_return_3d'] = 0.0

            # 路径依赖特征
            result_data['consecutive_positive_days'] = 0.0
            result_data['consecutive_negative_days'] = 0.0
            result_data['consecutive_positive_days_rank'] = 0.0  # 新增排名分位数
            result_data['consecutive_negative_days_rank'] = 0.0  # 新增排名分位数

        except Exception as e:
            self.logger.error(f"初始化目标风险特征失败: {e}")

    def _calculate_target_risk_features_with_history(self, full_data: pd.DataFrame,
                                                   target_data: pd.DataFrame,
                                                   target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的风险特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的风险特征（使用历史数据支撑）...")

            # 1. 计算基础风险特征（只为目标日期）
            target_data = self._calculate_target_basic_risk_features(target_data)

            # 2. 计算复杂风险特征（只为目标日期，使用历史数据支撑）
            if 'pct_chg' in full_data.columns:
                self.logger.info("✅ 检测到pct_chg字段，开始计算复杂风险因子")
                target_data = self._calculate_target_complex_risk_features(
                    full_data, target_data, target_dates
                )
                self.logger.info("✅ 复杂风险因子计算完成")
            else:
                self.logger.error("❌ 缺少pct_chg字段，无法计算复杂风险因子")

            self.logger.info("目标日期风险特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期风险特征计算失败: {e}")
            return target_data

    def _calculate_target_basic_risk_features(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """为目标日期计算基础风险特征（不需要历史数据）"""
        try:
            # 这些特征基于当日数据，不需要历史数据支撑
            # 设置为默认值，实际计算逻辑可以根据需要添加
            target_data['risk_adjusted_return'] = 0.0
            target_data['returns_zscore'] = 0.0
            target_data['price_vol_change_ratio'] = 0.0
            target_data['high_volatility'] = 0.0

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标基础风险特征失败: {e}")
            return target_data

    def _calculate_target_complex_risk_features(self, full_data: pd.DataFrame,
                                              target_data: pd.DataFrame,
                                              target_dates: List[str]) -> pd.DataFrame:
        """为目标日期计算复杂风险特征（向量化优化版本）"""
        try:
            import time
            start_time = time.time()

            target_stocks = target_data['stock_code'].unique()
            total_stocks = len(target_stocks)

            self.logger.info(f"🚀 开始向量化计算 {total_stocks} 只股票的复杂风险特征")
            self.logger.info(f"📊 性能优化: 使用向量化计算替代逐股票循环")

            # 预处理：过滤目标股票数据并确保有足够历史数据
            preprocess_start = time.time()
            target_stock_data = full_data[full_data['stock_code'].isin(target_stocks)].copy()

            # 检查数据量
            stock_counts = target_stock_data.groupby('stock_code').size()
            valid_stocks = stock_counts[stock_counts >= 20].index
            target_stock_data = target_stock_data[target_stock_data['stock_code'].isin(valid_stocks)]

            preprocess_time = time.time() - preprocess_start
            self.logger.info(f"✅ 数据预处理完成，有效股票: {len(valid_stocks)}/{total_stocks} (耗时: {preprocess_time:.2f}秒)")

            # 向量化计算所有风险特征
            calc_start = time.time()
            target_data = self._calculate_vectorized_risk_features(target_stock_data, target_data, target_dates)
            calc_time = time.time() - calc_start

            # 计算排名分位数因子
            rank_start = time.time()
            target_data = self._calculate_ranking_factors(target_data)
            rank_time = time.time() - rank_start

            total_time = time.time() - start_time

            # 性能报告
            self.logger.info("🎯 性能统计报告:")
            self.logger.info(f"  📊 总处理时间: {total_time:.2f}秒")
            self.logger.info(f"  📊 数据预处理: {preprocess_time:.2f}秒 ({preprocess_time/total_time*100:.1f}%)")
            self.logger.info(f"  📊 特征计算: {calc_time:.2f}秒 ({calc_time/total_time*100:.1f}%)")
            self.logger.info(f"  📊 排名计算: {rank_time:.2f}秒 ({rank_time/total_time*100:.1f}%)")
            self.logger.info(f"  📊 平均每股处理时间: {total_time/len(valid_stocks)*1000:.2f}毫秒")
            self.logger.info(f"✅ 向量化风险特征计算完成 - 性能提升显著!")

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标复杂风险特征失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return target_data

    def _calculate_vectorized_risk_features(self, full_stock_data: pd.DataFrame,
                                          target_data: pd.DataFrame,
                                          target_dates: List[str]) -> pd.DataFrame:
        """向量化计算所有风险特征（高性能版本）"""
        try:
            self.logger.info("🔄 开始向量化特征计算...")

            # 1. 预处理数据
            if 'pct_chg' not in full_stock_data.columns:
                self.logger.error("缺少pct_chg列，无法计算风险特征")
                return target_data

            # 转换pct_chg为小数形式并排序
            full_stock_data = full_stock_data.sort_values(['stock_code', 'date']).copy()
            full_stock_data['pct_chg_decimal'] = full_stock_data['pct_chg'] / 100.0

            # 2. 向量化计算滞后收益率因子
            self.logger.info("📊 计算滞后收益率因子...")
            for lag in [1, 2, 3, 4, 5, 10, 20]:
                full_stock_data[f'pct_chg_lag{lag}'] = (
                    full_stock_data.groupby('stock_code')['pct_chg_decimal']
                    .shift(lag)
                    .fillna(0.0)
                    .clip(-0.2, 0.2)  # 异常值处理
                )

            # 3. 向量化计算滚动平均
            self.logger.info("📊 计算滚动聚合特征...")
            full_stock_data['ma_return_3d'] = (
                full_stock_data.groupby('stock_code')['pct_chg_decimal']
                .rolling(window=3, min_periods=2)
                .mean()
                .fillna(0.0)
                .clip(-0.15, 0.15)
                .reset_index(level=0, drop=True)
            )

            # 4. 向量化计算连续涨跌天数
            self.logger.info("📊 计算路径依赖特征...")
            full_stock_data = self._calculate_vectorized_consecutive_days(full_stock_data)

            # 5. 向量化计算基础风险因子（补全遗漏的四个因子）
            self.logger.info("📊 计算基础风险因子...")
            full_stock_data = self._calculate_vectorized_basic_risk_features(full_stock_data)

            # 6. 提取目标日期的结果
            self.logger.info("📊 提取目标日期结果...")
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            result_mask = full_stock_data['date'].isin(target_date_objects)
            result_features = full_stock_data[result_mask]

            # 7. 更新目标数据
            feature_columns = [f'pct_chg_lag{lag}' for lag in [1, 2, 3, 4, 5, 10, 20]] + [
                'ma_return_3d', 'consecutive_positive_days', 'consecutive_negative_days',
                'risk_adjusted_return', 'returns_zscore', 'price_vol_change_ratio', 'high_volatility'
            ]

            for col in feature_columns:
                if col in result_features.columns:
                    # 创建合并键
                    target_data['merge_key'] = target_data['stock_code'] + '_' + target_data['date'].dt.strftime('%Y%m%d')
                    result_features['merge_key'] = result_features['stock_code'] + '_' + result_features['date'].dt.strftime('%Y%m%d')

                    # 合并数据
                    merge_dict = result_features.set_index('merge_key')[col].to_dict()
                    target_data[col] = target_data['merge_key'].map(merge_dict).fillna(0.0)

            # 清理临时列
            if 'merge_key' in target_data.columns:
                target_data.drop('merge_key', axis=1, inplace=True)

            self.logger.info("✅ 向量化特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"向量化特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return target_data

    def _calculate_vectorized_basic_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算基础风险因子（补全遗漏的四个因子）"""
        try:
            self.logger.info("🔄 向量化计算基础风险因子...")

            # 1. high_volatility: 高波动率标识
            if 'volatility20' in data.columns:
                # 基于20日波动率的分位数，前20%为高波动率
                vol_threshold = data['volatility20'].quantile(0.8)
                data['high_volatility'] = (data['volatility20'] > vol_threshold).astype(int)
                self.logger.info(f"✅ high_volatility计算完成，阈值: {vol_threshold:.4f}")
            else:
                data['high_volatility'] = 0
                self.logger.warning("⚠️ 缺少volatility20列，high_volatility设为默认值0")

            # 2. risk_adjusted_return: 风险调整收益率
            if 'return_20d' in data.columns and 'volatility20' in data.columns:
                # 使用夏普比率的简化版本
                data['risk_adjusted_return'] = np.where(
                    data['volatility20'] > 0,
                    data['return_20d'] / data['volatility20'],
                    0
                )
                # 限制在合理范围内
                data['risk_adjusted_return'] = np.clip(data['risk_adjusted_return'], -5, 5)
                self.logger.info("✅ risk_adjusted_return计算完成")
            else:
                data['risk_adjusted_return'] = 0
                self.logger.warning("⚠️ 缺少return_20d或volatility20列，risk_adjusted_return设为默认值0")

            # 3. returns_zscore: 收益率Z-score
            if 'pct_chg' in data.columns:
                # 按股票分组计算收益率的Z-score
                data['returns_zscore'] = data.groupby('stock_code')['pct_chg'].transform(
                    lambda x: (x - x.rolling(60, min_periods=10).mean()) /
                             x.rolling(60, min_periods=10).std().fillna(1)
                ).fillna(0)
                # 限制在合理范围内
                data['returns_zscore'] = np.clip(data['returns_zscore'], -5, 5)
                self.logger.info("✅ returns_zscore计算完成")
            else:
                data['returns_zscore'] = 0
                self.logger.warning("⚠️ 缺少pct_chg列，returns_zscore设为默认值0")

            # 4. price_vol_change_ratio: 价格波动变化比率
            if 'pct_chg' in data.columns and 'volume' in data.columns:
                # 价格变化与成交量变化的比率
                price_change = np.abs(data['pct_chg'])
                volume_change = data.groupby('stock_code')['volume'].pct_change().abs()

                data['price_vol_change_ratio'] = np.where(
                    volume_change > 0,
                    price_change / (volume_change + 0.01),  # 避免除零
                    price_change
                )
                # 限制在合理范围内
                data['price_vol_change_ratio'] = np.clip(data['price_vol_change_ratio'], 0, 10)
                self.logger.info("✅ price_vol_change_ratio计算完成")
            else:
                data['price_vol_change_ratio'] = 1.0
                self.logger.warning("⚠️ 缺少pct_chg或volume列，price_vol_change_ratio设为默认值1.0")

            self.logger.info("✅ 基础风险因子向量化计算完成")
            return data

        except Exception as e:
            self.logger.error(f"基础风险因子向量化计算失败: {e}")
            # 设置默认值
            data['high_volatility'] = 0
            data['risk_adjusted_return'] = 0
            data['returns_zscore'] = 0
            data['price_vol_change_ratio'] = 1.0
            return data

    def _calculate_vectorized_consecutive_days(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算连续涨跌天数"""
        try:
            self.logger.info("🔄 向量化计算连续涨跌天数...")

            # 初始化结果列
            data['consecutive_positive_days'] = 0
            data['consecutive_negative_days'] = 0

            # 按股票分组处理
            for stock_code, group in data.groupby('stock_code'):
                if len(group) < 2:
                    continue

                # 获取收益率数组
                returns = group['pct_chg_decimal'].values
                indices = group.index

                # 向量化计算连续天数
                pos_days, neg_days = self._vectorized_consecutive_calculation(returns)

                # 更新数据
                data.loc[indices, 'consecutive_positive_days'] = np.minimum(pos_days, 10)
                data.loc[indices, 'consecutive_negative_days'] = np.minimum(neg_days, 10)

            return data

        except Exception as e:
            self.logger.error(f"向量化连续天数计算失败: {e}")
            return data

    def _vectorized_consecutive_calculation(self, returns: np.ndarray) -> tuple:
        """向量化计算连续涨跌天数的核心算法"""
        try:
            n = len(returns)
            if n == 0:
                return np.array([]), np.array([])

            pos_days = np.zeros(n, dtype=int)
            neg_days = np.zeros(n, dtype=int)

            # 创建涨跌标记
            is_positive = returns > 0
            is_negative = returns < 0

            # 向量化计算连续天数
            for i in range(n):
                if is_positive[i]:
                    # 当日上涨，计算连续上涨天数
                    if i == 0:
                        pos_days[i] = 1
                    else:
                        pos_days[i] = pos_days[i-1] + 1 if is_positive[i-1] else 1
                    neg_days[i] = 0
                elif is_negative[i]:
                    # 当日下跌，计算连续下跌天数
                    if i == 0:
                        neg_days[i] = 1
                    else:
                        neg_days[i] = neg_days[i-1] + 1 if is_negative[i-1] else 1
                    pos_days[i] = 0
                else:
                    # 平盘，重置计数
                    pos_days[i] = 0
                    neg_days[i] = 0

            return pos_days, neg_days

        except Exception as e:
            self.logger.error(f"向量化连续天数核心计算失败: {e}")
            return np.zeros(len(returns), dtype=int), np.zeros(len(returns), dtype=int)

    def _calculate_stock_target_risk_features(self, stock_full_data: pd.DataFrame,
                                            target_dates: List[str]) -> dict:
        """为单只股票的目标日期计算复杂风险特征"""
        try:
            target_features = {}

            # 预处理数据
            if 'pct_chg' not in stock_full_data.columns:
                return {}

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
                features = self._calculate_risk_features_for_single_date(
                    stock_full_data, data_idx, pct_chg
                )

                target_features[target_date_obj] = features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标风险特征失败: {e}")
            return {}

    def _calculate_risk_features_for_single_date(self, stock_data: pd.DataFrame,
                                               data_idx: int,
                                               pct_chg: np.ndarray) -> dict:
        """为单个目标日期计算所有复杂风险特征"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 20:
                return self._get_default_risk_features()

            # 1. 滞后收益率因子
            for lag in [1, 2, 3, 4, 5, 10, 20]:
                if data_idx >= lag:
                    features[f'pct_chg_lag{lag}'] = float(pct_chg[data_idx - lag])
                else:
                    features[f'pct_chg_lag{lag}'] = 0.0

            # 2. 滚动聚合特征 (删除ma_return_5d, std_return_5d, std_return_20d)
            features['ma_return_3d'] = self._calculate_rolling_mean_for_date(pct_chg, data_idx, 3)

            # 3. 路径依赖特征
            features['consecutive_positive_days'] = self._calculate_consecutive_days_for_date(
                pct_chg, data_idx, positive=True
            )
            features['consecutive_negative_days'] = self._calculate_consecutive_days_for_date(
                pct_chg, data_idx, positive=False
            )

            return features

        except Exception as e:
            self.logger.error(f"计算单日期风险特征失败: {e}")
            return self._get_default_risk_features()

    def _calculate_rolling_mean_for_date(self, pct_chg: np.ndarray, data_idx: int, window: int) -> float:
        """为单个目标日期计算滚动平均收益率"""
        try:
            if data_idx < window - 1:
                return 0.0

            start_idx = max(0, data_idx - window + 1)
            end_idx = data_idx + 1

            window_data = pct_chg[start_idx:end_idx]
            if len(window_data) >= 2:
                mean_return = np.mean(window_data)
                return float(np.clip(mean_return, -0.15, 0.15))

            return 0.0

        except Exception as e:
            self.logger.error(f"计算滚动平均收益率失败: {e}")
            return 0.0



    def _calculate_consecutive_days_for_date(self, pct_chg: np.ndarray, data_idx: int, positive: bool) -> float:
        """为单个目标日期计算连续涨跌天数"""
        try:
            if data_idx < 1:
                return 0.0

            count = 0
            for i in range(data_idx, -1, -1):
                if positive:
                    if pct_chg[i] > 0:
                        count += 1
                    else:
                        break
                else:
                    if pct_chg[i] < 0:
                        count += 1
                    else:
                        break

            return float(min(count, 10))  # 限制最大值

        except Exception as e:
            self.logger.error(f"计算连续涨跌天数失败: {e}")
            return 0.0

    def _get_default_risk_features(self) -> dict:
        """获取默认的风险特征值"""
        features = {}

        # 滞后收益率因子
        for lag in [1, 2, 3, 4, 5, 10, 20]:
            features[f'pct_chg_lag{lag}'] = 0.0

        # 滚动聚合特征 (删除ma_return_5d, std_return_5d, std_return_20d)
        features['ma_return_3d'] = 0.0

        # 路径依赖特征
        features['consecutive_positive_days'] = 0.0
        features['consecutive_negative_days'] = 0.0

        return features

    def _update_target_risk_features(self, target_data: pd.DataFrame,
                                   stock_target_features: dict,
                                   stock_target_mask: pd.Series):
        """更新目标数据中的风险特征"""
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
            self.logger.error(f"更新目标风险特征失败: {e}")

    def _calculate_ranking_factors(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算排名分位数因子

        为pct_chg_lag系列因子和连续涨跌天数因子添加全市场排名分位数
        排名逻辑：对每个交易日，计算该因子在全部股票中的分位数排名（0-1之间，1表示最高）
        """
        try:
            self.logger.info("开始计算排名分位数因子...")

            # 按日期分组计算排名分位数
            for date_group in target_data.groupby('date'):
                date_value, date_data = date_group
                date_mask = target_data['date'] == date_value

                # 1. 计算pct_chg_lag系列因子的排名分位数
                for lag in [1, 2, 3, 4, 5, 10, 20]:
                    factor_name = f'pct_chg_lag{lag}'
                    rank_name = f'pct_chg_lag{lag}_rank'

                    if factor_name in date_data.columns:
                        # 计算分位数排名 (0-1之间，1表示最高)
                        ranks = date_data[factor_name].rank(method='min', pct=True)
                        target_data.loc[date_mask, rank_name] = ranks.fillna(0.5)

                # 2. 计算连续涨跌天数因子的排名分位数
                for factor_name in ['consecutive_positive_days', 'consecutive_negative_days']:
                    rank_name = f'{factor_name}_rank'

                    if factor_name in date_data.columns:
                        # 计算分位数排名 (0-1之间，1表示最高)
                        ranks = date_data[factor_name].rank(method='min', pct=True)
                        target_data.loc[date_mask, rank_name] = ranks.fillna(0.5)

            self.logger.info("排名分位数因子计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"计算排名分位数因子失败: {e}")
            # 设置默认值
            for lag in [1, 2, 3, 4, 5, 10, 20]:
                target_data[f'pct_chg_lag{lag}_rank'] = 0.5
            target_data['consecutive_positive_days_rank'] = 0.5
            target_data['consecutive_negative_days_rank'] = 0.5
            return target_data

    def _calculate_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算风险相关特征（从feature_calculator.py迁移）"""
        try:
            # 1. high_volatility: 高波动率标识
            if 'volatility20' in data.columns:
                # 基于20日波动率的分位数，前20%为高波动率
                vol_threshold = data['volatility20'].quantile(0.8)
                data['high_volatility'] = (data['volatility20'] > vol_threshold).astype(int)
            else:
                data['high_volatility'] = 0


            # 3. risk_adjusted_return: 风险调整收益率
            if 'return_20d' in data.columns and 'volatility20' in data.columns:
                # 使用夏普比率的简化版本
                data['risk_adjusted_return'] = np.where(
                    data['volatility20'] > 0,
                    data['return_20d'] / data['volatility20'],
                    0
                )
                # 限制在合理范围内
                data['risk_adjusted_return'] = np.clip(data['risk_adjusted_return'], -5, 5)
            else:
                data['risk_adjusted_return'] = 0

            # 4. returns_zscore: 收益率Z-score
            if 'pct_chg' in data.columns:
                # 按股票分组计算收益率的Z-score
                data['returns_zscore'] = data.groupby('stock_code')['pct_chg'].transform(
                    lambda x: (x - x.rolling(60, min_periods=10).mean()) /
                             x.rolling(60, min_periods=10).std().fillna(1)
                ).fillna(0)
                # 限制在合理范围内
                data['returns_zscore'] = np.clip(data['returns_zscore'], -5, 5)
            else:
                data['returns_zscore'] = 0



            # 6. price_vol_change_ratio: 价格波动变化比率
            if 'pct_chg' in data.columns and 'volume' in data.columns:
                # 价格变化与成交量变化的比率
                price_change = np.abs(data['pct_chg'])
                volume_change = data.groupby('stock_code')['volume'].pct_change().abs()

                data['price_vol_change_ratio'] = np.where(
                    volume_change > 0,
                    price_change / (volume_change + 0.01),  # 避免除零
                    price_change
                )
                # 限制在合理范围内
                data['price_vol_change_ratio'] = np.clip(data['price_vol_change_ratio'], 0, 10)
            else:
                data['price_vol_change_ratio'] = 1.0

            self.logger.debug("风险相关特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"风险特征计算失败: {e}")
            return data

    def _calculate_lagged_return_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算滞后收益率因子 (Lagged Return Factors)

        这些因子捕捉股票的历史收益率模式，用于：
        1. 动量效应分析：连续正收益可能预示动量延续
        2. 反转效应分析：极端收益后可能出现反转
        3. 序列相关性：识别收益率的时间序列依赖关系

        金融意义：
        - pct_chg_lag1~5: 短期动量/反转信号
        - pct_chg_lag10: 中短期趋势信号
        - pct_chg_lag20: 中期趋势信号
        """
        try:
            if 'pct_chg' not in data.columns:
                self.logger.warning("缺少pct_chg列，无法计算滞后收益率因子")
                # 设置默认值
                for lag in [1, 2, 3, 4, 5, 10, 20]:
                    data[f'pct_chg_lag{lag}'] = 0.0
                return data

            # 按股票分组计算滞后收益率
            lag_periods = [1, 2, 3, 4, 5, 10, 20]

            for lag in lag_periods:
                try:
                    # 计算滞后收益率，按股票分组避免跨股票污染
                    data[f'pct_chg_lag{lag}'] = data.groupby('stock_code')['pct_chg'].shift(lag)

                    # 处理缺失值：新上市股票或数据不足的情况
                    data[f'pct_chg_lag{lag}'] = data[f'pct_chg_lag{lag}'].fillna(0.0)

                    # 异常值处理：限制在合理范围内（±20%）
                    data[f'pct_chg_lag{lag}'] = np.clip(data[f'pct_chg_lag{lag}'], -0.2, 0.2)

                except Exception as e:
                    self.logger.warning(f"计算pct_chg_lag{lag}失败: {e}")
                    data[f'pct_chg_lag{lag}'] = 0.0

            self.logger.debug("滞后收益率因子计算完成")
            return data

        except Exception as e:
            self.logger.error(f"滞后收益率因子计算失败: {e}")
            # 设置默认值
            for lag in [1, 2, 3, 4, 5, 10, 20]:
                data[f'pct_chg_lag{lag}'] = 0.0
            return data



    def _calculate_path_dependent_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算路径依赖特征 (Path-Dependent Features)

        这些因子捕捉股票价格的路径依赖特征，基于连续涨跌模式：

        金融意义：
        - consecutive_positive_days: 连续上涨天数
          * 0-2天: 正常波动
          * 3-5天: 短期动量，可能延续
          * 6-10天: 强势动量，但需警惕反转风险
          * >10天: 极端情况，高反转风险

        - consecutive_negative_days: 连续下跌天数
          * 0-2天: 正常调整
          * 3-5天: 弱势调整，可能反弹
          * 6-10天: 深度调整，反弹机会增加
          * >10天: 极端弱势，需谨慎

        应用场景：
        1. 动量策略：连续上涨3-5天的股票，动量延续概率高
        2. 反转策略：连续下跌5-8天的股票，反弹概率增加
        3. 风险控制：连续单边运行超过8天，需降低仓位
        """
        try:
            if 'pct_chg' not in data.columns:
                self.logger.warning("缺少pct_chg列，无法计算路径依赖因子")
                data['consecutive_positive_days'] = 0
                data['consecutive_negative_days'] = 0
                return data

            # 初始化结果列
            data['consecutive_positive_days'] = 0
            data['consecutive_negative_days'] = 0

            # 按股票分组计算连续涨跌天数
            for stock_code in data['stock_code'].unique():
                try:
                    stock_mask = data['stock_code'] == stock_code
                    stock_data = data[stock_mask].sort_values('date')

                    if len(stock_data) == 0:
                        continue

                    # 计算连续上涨和下跌天数
                    pos_days, neg_days = self._calculate_consecutive_days(
                        stock_data['pct_chg'].values
                    )

                    # 更新结果（限制最大值为10天）
                    data.loc[stock_mask, 'consecutive_positive_days'] = np.minimum(pos_days, 10)
                    data.loc[stock_mask, 'consecutive_negative_days'] = np.minimum(neg_days, 10)

                except Exception as e:
                    self.logger.warning(f"计算股票{stock_code}的连续涨跌天数失败: {e}")
                    continue

            self.logger.debug("路径依赖因子计算完成")
            return data

        except Exception as e:
            self.logger.error(f"路径依赖因子计算失败: {e}")
            # 设置默认值
            data['consecutive_positive_days'] = 0
            data['consecutive_negative_days'] = 0
            return data

    def _calculate_consecutive_days(self, returns: np.ndarray) -> tuple:
        """
        计算连续涨跌天数的核心算法

        Args:
            returns: 收益率数组，按时间顺序排列

        Returns:
            tuple: (连续上涨天数数组, 连续下跌天数数组)
        """
        try:
            n = len(returns)
            if n == 0:
                return np.array([]), np.array([])

            pos_days = np.zeros(n, dtype=int)
            neg_days = np.zeros(n, dtype=int)

            for i in range(n):
                if i == 0:
                    # 第一天
                    pos_days[i] = 1 if returns[i] > 0 else 0
                    neg_days[i] = 1 if returns[i] < 0 else 0
                else:
                    # 后续天数
                    if returns[i] > 0:
                        # 当日上涨
                        pos_days[i] = pos_days[i-1] + 1 if returns[i-1] > 0 else 1
                        neg_days[i] = 0
                    elif returns[i] < 0:
                        # 当日下跌
                        neg_days[i] = neg_days[i-1] + 1 if returns[i-1] < 0 else 1
                        pos_days[i] = 0
                    else:
                        # 当日平盘，重置计数
                        pos_days[i] = 0
                        neg_days[i] = 0

            return pos_days, neg_days

        except Exception as e:
            self.logger.error(f"计算连续天数失败: {e}")
            return np.zeros(len(returns), dtype=int), np.zeros(len(returns), dtype=int)

    def _set_default_new_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """为新增因子设置默认值"""
        try:
            # 滞后收益率因子
            for lag in [1, 2, 3, 4, 5, 10, 20]:
                data[f'pct_chg_lag{lag}'] = 0.0
                data[f'pct_chg_lag{lag}_rank'] = 0.5  # 排名分位数默认值

            # 滚动聚合特征 (删除ma_return_5d, std_return_5d, std_return_20d)
            data['ma_return_3d'] = 0.0

            # 路径依赖特征
            data['consecutive_positive_days'] = 0
            data['consecutive_negative_days'] = 0
            data['consecutive_positive_days_rank'] = 0.5  # 排名分位数默认值
            data['consecutive_negative_days_rank'] = 0.5  # 排名分位数默认值

            self.logger.info("✅ 已为新增因子设置默认值")
            return data

        except Exception as e:
            self.logger.error(f"设置新增因子默认值失败: {e}")
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
