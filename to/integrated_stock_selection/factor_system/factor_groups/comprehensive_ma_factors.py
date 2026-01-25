#!/usr/bin/env python3
"""
综合均线因子组

整合所有均线相关特征，包括均线粘合度、交叉强度、相对位置等
修复重复逻辑并添加专业量化指标，如MACD、EMA交叉、均线乖离率等
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
from typing import List
from .base_factor_group import BaseFactorGroup

class ComprehensiveMAFactors(BaseFactorGroup):
    """综合均线因子组"""
    
    def __init__(self):
        super().__init__("comprehensive_ma_factors")

        # 高质量上升趋势评分的权重参数
        self.uptrend_weights = {
            'ma_slope_weight': 0.4,      # 均线斜率权重（最重要）
            'trend_quality_weight': 0.3,  # 趋势质量权重
            'ma_alignment_weight': 0.2,   # 均线排列权重
            'price_action_weight': 0.1    # 价格行为权重
        }
    
    def _get_version(self) -> str:
        return "v1.0"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 均线因子组不依赖其他因子组
    
    def _get_feature_names(self) -> List[str]:
        return [
            # 均线粘合度特征
            'ma_support_strength',

            # 均线相对特征
            'ma_alignment_score', 'ma_convergence_score',

            # 专业量化指标（删除重复的MACD特征，使用generate_daily_features.py中的）
            'ma_deviation_rate_5d', 'ma_deviation_rate_20d',
            'ma_trend_strength', 'ma_volatility_ratio',

            # 趋势间关系特征 (Trend vs. Trend)
            'ma_ratio_5_10', 'ma_ratio_5_20', 'ma_ratio_20_60', 'ma_ratio_10_30', 'ma_ratio_5_60',

            # 价格与趋势关系特征 (Price vs. Trend)
            'price_ma_ratio_5', 'price_ma_ratio_10', 'price_ma_ratio_20',
            'price_ma_ratio_60', 'price_ma_ratio_120',

            # 长期趋势向上特征
            'ma_ratio_60_120',           # 60日均线与120日均线的比率
            'ma_alignment_score_extended', # 扩展的均线排列得分

            # 趋势缓慢与健康特征
            'trend_quality_score',       # 趋势质量评分：ma60_slope / volatility_120d
            'max_drawdown_in_uptrend_120d', # 120天内最大回撤

            # 回调到关键支撑特征
            'price_position_vs_120d_high', # 相对120日高点的位置（替代52周高点）
            'BIAS_60D',                  # 60日均线乖离率

            # 高质量上升趋势评分特征
            'high_quality_uptrend_score' # 综合评估股票的高质量上升趋势强度
        ]

    def _validate_no_data_leakage(self, data: pd.DataFrame) -> bool:
        """验证均线计算是否存在数据泄露

        检查均线是否包含当日数据，这是导致准确率异常达到1.0000的根本原因

        Args:
            data: 包含均线数据的DataFrame

        Returns:
            bool: True表示无数据泄露，False表示存在数据泄露
        """
        try:
            self.logger.info("🔍 开始数据泄露检测...")

            # 检查均线是否包含当日数据
            ma_columns = ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']
            available_mas = [col for col in ma_columns if col in data.columns]

            if not available_mas:
                self.logger.warning("⚠️ 未找到均线数据，跳过数据泄露检测")
                return True

            # 随机选择一只股票进行检测
            sample_stocks = data['stock_code'].unique()[:2]  # 检测前2只股票

            for stock_code in sample_stocks:
                stock_data = data[data['stock_code'] == stock_code].sort_values('date').reset_index(drop=True)

                if len(stock_data) < 120:  # 需要足够的历史数据
                    continue

                # 检查ma120的计算逻辑
                if 'ma120' in stock_data.columns and 'close' in stock_data.columns:
                    # 检查第120个数据点（索引119）
                    i = 119
                    if i < len(stock_data):
                        # 手动计算ma120（不包含当日）
                        manual_ma120_no_today = stock_data['close'].iloc[i-119:i].mean()  # 不包含当日
                        manual_ma120_with_today = stock_data['close'].iloc[i-119:i+1].mean()  # 包含当日
                        actual_ma120 = stock_data['ma120'].iloc[i]

                        if not pd.isna(actual_ma120):
                            diff_without_today = abs(actual_ma120 - manual_ma120_no_today)
                            diff_with_today = abs(actual_ma120 - manual_ma120_with_today)

                            # 如果与包含当日的计算更接近，说明存在数据泄露
                            if diff_with_today < diff_without_today and diff_with_today < 0.001:
                                self.logger.error(f"❌ 发现数据泄露！股票{stock_code}的ma120包含当日数据")
                                self.logger.error(f"   实际ma120: {actual_ma120:.6f}")
                                self.logger.error(f"   不含当日: {manual_ma120_no_today:.6f} (差异: {diff_without_today:.6f})")
                                self.logger.error(f"   含当日:   {manual_ma120_with_today:.6f} (差异: {diff_with_today:.6f})")
                                return False

                break  # 只检查第一只有足够数据的股票

            self.logger.info("✅ 数据泄露检测通过")
            return True

        except Exception as e:
            self.logger.error(f"❌ 数据泄露检测失败: {e}")
            return False

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算综合均线因子（优化版：参照KlinePatternFactors模式）"""
        try:
            # 🚀 性能监控初始化
            start_time = time.time()
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            self.logger.info("🚀 开始计算综合均线因子（优化版）")
            self.logger.info(f"📊 输入数据规模: {len(data)} 条记录, {len(data.columns)} 个字段")
            self.logger.info(f"💾 初始内存使用: {initial_memory:.1f} MB")

            # 1. 识别目标日期和历史数据范围
            step_start = time.time()
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return data

            self.logger.info(f"📅 目标计算日期: {len(target_dates)} 个")
            self.logger.info(f"📈 历史支撑数据: {len(historical_dates)} 个交易日")
            self.logger.info(f"⏱️ 日期识别耗时: {time.time() - step_start:.2f}s")

            # 🚀 关键优化：使用优化的计算策略
            result_data = self._calculate_ma_features_optimized(data, target_dates, historical_dates)

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            # 🎉 总体性能统计
            total_duration = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory

            self.logger.info("🎉 综合均线因子计算完成（优化版）！")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 性能统计:")
            self.logger.info(f"    ⏱️ 总耗时: {total_duration:.2f}s")
            self.logger.info(f"    💾 内存使用: {memory_usage:+.1f} MB")
            self.logger.info(f"    📋 输出记录: {len(result_data)} 条")
            self.logger.info("=" * 60)

            return result_data

        except Exception as e:
            self.logger.error(f"❌ 综合均线因子计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return data

    def _calculate_ma_features_optimized(self, full_data: pd.DataFrame, target_dates: list, historical_dates: list) -> pd.DataFrame:
        """
        优化的均线特征计算（参照KlinePatternFactors模式）
        只对目标日期进行计算，避免为扩展历史日期计算不需要的特征
        """
        try:
            self.logger.info("开始优化版均线特征计算")

            # 0. 缓存完整数据供需要历史数据的特征使用
            self._full_data_cache = full_data.copy()

            # 1. 使用完整数据计算滚动均线（这些是必需的基础数据）
            full_data_with_ma = self._calculate_rolling_ma_on_full_data(full_data)

            # 2. 只对目标日期计算复杂的衍生特征
            target_data = self._calculate_target_ma_features(full_data_with_ma, target_dates)

            self.logger.info("优化版均线特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"优化版均线特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _calculate_rolling_ma_on_full_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """在完整数据上计算滚动均线（必需的基础数据）"""
        try:
            self.logger.info("计算基础滚动均线...")

            # 确保数据按股票和日期排序
            data = data.sort_values(['stock_code', 'date'])

            # 计算基础均线（如果不存在的话）
            ma_periods = [5, 10, 20, 30, 60, 120]
            for period in ma_periods:
                ma_col = f'ma{period}'
                if ma_col not in data.columns:
                    data[ma_col] = data.groupby('stock_code')['close'].rolling(period).mean().reset_index(0, drop=True)

            self.logger.info("基础滚动均线计算完成")
            return data

        except Exception as e:
            self.logger.error(f"基础滚动均线计算失败: {e}")
            return data

    def _calculate_target_ma_features(self, full_data: pd.DataFrame, target_dates: list) -> pd.DataFrame:
        """计算均线衍生特征（正确的优化策略：历史数据只作支撑，只为目标日期计算指标）"""
        try:
            self.logger.info("开始计算均线衍生特征（正确优化策略）...")

            # 转换目标日期为datetime格式以避免警告
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)

            if target_mask.sum() == 0:
                self.logger.warning("目标日期数据为空")
                return full_data

            self.logger.info(f"目标日期数据: {target_mask.sum()} 条记录")
            self.logger.info(f"总数据量: {len(full_data)} 条记录")
            self.logger.info(f"历史数据: {len(full_data) - target_mask.sum()} 条记录（仅作数据支撑）")

            # 🚀 正确策略：只对目标日期计算所有衍生特征，历史数据仅作支撑
            target_data = full_data[target_mask].copy()

            # 计算简单特征（不需要历史数据的特征）
            target_data = self._calculate_target_ma_convergence_features(target_data)
            target_data = self._calculate_target_ma_relative_features(target_data)
            target_data = self._calculate_target_professional_ma_indicators(target_data)
            target_data = self._calculate_target_trend_relationship_features(target_data)
            target_data = self._calculate_target_price_trend_relationship_features(target_data)

            # 计算复杂特征（需要历史数据的特征）- 关键修复：只为目标日期计算
            target_data = self._calculate_target_complex_features_with_history(target_data, full_data, target_date_objects)

            self.logger.info("正确优化策略均线衍生特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"正确优化策略均线特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _calculate_target_complex_features_with_history(self, target_data: pd.DataFrame, full_data: pd.DataFrame, target_date_objects: list) -> pd.DataFrame:
        """
        正确计算需要历史数据的复杂特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算需要历史数据的复杂特征（只为目标日期）...")

            # 为每个目标日期逐一计算需要历史数据的特征
            result_list = []

            for target_date in target_date_objects:
                # 获取该目标日期的数据
                target_day_mask = target_data['date'] == target_date
                target_day_data = target_data[target_day_mask].copy()

                if len(target_day_data) == 0:
                    continue

                # 为该目标日期计算复杂特征
                target_day_data = self._calculate_complex_features_for_single_date(
                    target_day_data, full_data, target_date
                )

                result_list.append(target_day_data)

            if result_list:
                result_data = pd.concat(result_list, ignore_index=True)
                self.logger.info(f"复杂特征计算完成，处理了 {len(result_list)} 个目标日期")
                return result_data
            else:
                self.logger.warning("没有有效的目标日期数据")
                return target_data

        except Exception as e:
            self.logger.error(f"复杂特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return target_data

    def _calculate_complex_features_for_single_date(self, target_day_data: pd.DataFrame, full_data: pd.DataFrame, target_date) -> pd.DataFrame:
        """为单个目标日期计算需要历史数据的复杂特征"""
        try:
            # 获取该日期及之前的历史数据（用于计算120日特征）
            historical_mask = full_data['date'] <= target_date
            historical_data = full_data[historical_mask].copy()

            # 为目标日期的每只股票计算复杂特征
            for idx, row in target_day_data.iterrows():
                stock_code = row['stock_code']

                # 获取该股票的历史数据
                stock_historical = historical_data[historical_data['stock_code'] == stock_code].copy()
                stock_historical = stock_historical.sort_values('date')

                if len(stock_historical) < 2:
                    # 历史数据不足，设置默认值
                    target_day_data.loc[idx, 'max_drawdown_in_uptrend_120d'] = 0.0
                    target_day_data.loc[idx, 'trend_quality_score'] = 0.0
                    target_day_data.loc[idx, 'price_position_vs_120d_high'] = 0.0
                    target_day_data.loc[idx, 'BIAS_60D'] = 0.0
                    target_day_data.loc[idx, 'high_quality_uptrend_score'] = 0.0
                    target_day_data.loc[idx, 'ma_ratio_60_120'] = 1.0
                    target_day_data.loc[idx, 'ma_alignment_score_extended'] = 0.0
                    continue

                # 计算max_drawdown_in_uptrend_120d（只为目标日期计算）
                max_drawdown = self._calculate_max_drawdown_for_date(stock_historical, target_date)
                target_day_data.loc[idx, 'max_drawdown_in_uptrend_120d'] = max_drawdown

                # 计算其他复杂特征
                target_day_data.loc[idx, 'trend_quality_score'] = self._calculate_trend_quality_for_date(stock_historical, target_date)
                target_day_data.loc[idx, 'price_position_vs_120d_high'] = self._calculate_price_position_for_date(stock_historical, target_date)
                target_day_data.loc[idx, 'BIAS_60D'] = self._calculate_bias_60d_for_date(stock_historical, target_date)
                target_day_data.loc[idx, 'high_quality_uptrend_score'] = self._calculate_uptrend_score_for_date(stock_historical, target_date)
                target_day_data.loc[idx, 'ma_ratio_60_120'] = self._calculate_ma_ratio_60_120_for_date(stock_historical, target_date)
                target_day_data.loc[idx, 'ma_alignment_score_extended'] = self._calculate_ma_alignment_extended_for_date(stock_historical, target_date)

            return target_day_data

        except Exception as e:
            self.logger.error(f"单日期复杂特征计算失败: {e}")
            return target_day_data

    def _calculate_max_drawdown_for_date(self, stock_historical: pd.DataFrame, target_date) -> float:
        """为单个目标日期计算120日最大回撤"""
        try:
            # 获取目标日期及之前120天的数据
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 0.0

            # 获取120天窗口的数据
            window_data = stock_historical.tail(120)  # 最近120天
            if len(window_data) < 2:
                return 0.0

            closes = window_data['close'].values

            # 计算最大回撤
            peak = closes[0]
            max_drawdown = 0.0

            for price in closes:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak if peak > 0 else 0.0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            return -max_drawdown  # 返回负值表示回撤

        except Exception as e:
            self.logger.error(f"计算单日期最大回撤失败: {e}")
            return 0.0

    def _calculate_trend_quality_for_date(self, stock_historical: pd.DataFrame, target_date) -> float:
        """为单个目标日期计算趋势质量评分"""
        try:
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 0.0

            # 简化的趋势质量计算
            if 'ma60' in target_row.columns and 'close' in target_row.columns:
                ma60_val = target_row['ma60'].iloc[0]
                close_val = target_row['close'].iloc[0]
                if close_val > 0:
                    return (ma60_val / close_val) - 1.0

            return 0.0

        except Exception as e:
            self.logger.error(f"计算单日期趋势质量失败: {e}")
            return 0.0

    def _calculate_price_position_for_date(self, stock_historical: pd.DataFrame, target_date) -> float:
        """为单个目标日期计算相对120日高点位置"""
        try:
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 0.0

            # 获取120天窗口的最高价（不包含当日）
            historical_only = stock_historical[stock_historical['date'] < target_date].tail(120)
            if len(historical_only) == 0:
                return 0.0

            high_120d = historical_only['high'].max()
            current_close = target_row['close'].iloc[0]

            if high_120d > 0:
                return (current_close - high_120d) / high_120d

            return 0.0

        except Exception as e:
            self.logger.error(f"计算单日期价格位置失败: {e}")
            return 0.0

    def _calculate_bias_60d_for_date(self, stock_historical: pd.DataFrame, target_date) -> float:
        """为单个目标日期计算60日均线乖离率"""
        try:
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 0.0

            if 'ma60' in target_row.columns and 'close' in target_row.columns:
                ma60_val = target_row['ma60'].iloc[0]
                close_val = target_row['close'].iloc[0]
                if ma60_val > 0:
                    return (close_val - ma60_val) / ma60_val

            return 0.0

        except Exception as e:
            self.logger.error(f"计算单日期BIAS_60D失败: {e}")
            return 0.0

    def _calculate_uptrend_score_for_date(self, stock_historical: pd.DataFrame, target_date) -> float:
        """为单个目标日期计算高质量上升趋势评分"""
        try:
            # 简化的上升趋势评分
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 0.0

            # 基于均线排列的简单评分
            score = 0.0
            if all(col in target_row.columns for col in ['ma5', 'ma10', 'ma20', 'ma60']):
                ma5 = target_row['ma5'].iloc[0]
                ma10 = target_row['ma10'].iloc[0]
                ma20 = target_row['ma20'].iloc[0]
                ma60 = target_row['ma60'].iloc[0]

                if ma5 > ma10 > ma20 > ma60:
                    score = 1.0
                elif ma5 > ma10 > ma20:
                    score = 0.7
                elif ma5 > ma10:
                    score = 0.4
                else:
                    score = 0.1

            return score

        except Exception as e:
            self.logger.error(f"计算单日期上升趋势评分失败: {e}")
            return 0.0

    def _calculate_ma_ratio_60_120_for_date(self, stock_historical: pd.DataFrame, target_date) -> float:
        """为单个目标日期计算60日与120日均线比率"""
        try:
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 1.0

            if 'ma60' in target_row.columns and 'ma120' in target_row.columns:
                ma60_val = target_row['ma60'].iloc[0]
                ma120_val = target_row['ma120'].iloc[0]
                if ma120_val > 0:
                    return ma60_val / ma120_val

            return 1.0

        except Exception as e:
            self.logger.error(f"计算单日期MA比率失败: {e}")
            return 1.0

    def _calculate_ma_alignment_extended_for_date(self, stock_historical: pd.DataFrame, target_date) -> float:
        """为单个目标日期计算扩展均线排列评分"""
        try:
            target_row = stock_historical[stock_historical['date'] == target_date]
            if len(target_row) == 0:
                return 0.0

            # 扩展的均线排列评分
            score = 0.0
            if all(col in target_row.columns for col in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']):
                mas = [
                    target_row['ma5'].iloc[0],
                    target_row['ma10'].iloc[0],
                    target_row['ma20'].iloc[0],
                    target_row['ma60'].iloc[0],
                    target_row['ma120'].iloc[0]
                ]

                # 检查是否按顺序排列
                ascending_count = 0
                for i in range(len(mas) - 1):
                    if mas[i] > mas[i + 1]:
                        ascending_count += 1

                score = ascending_count / (len(mas) - 1)  # 归一化到[0,1]

            return score

        except Exception as e:
            self.logger.error(f"计算单日期扩展均线排列失败: {e}")
            return 0.0

    def _calculate_target_ma_convergence_features(self, result_data: pd.DataFrame) -> pd.DataFrame:
        """只为目标日期计算均线粘合度特征"""
        try:
            self.logger.info("计算目标日期均线粘合度特征...")

            # 检查必要的均线数据
            ma_columns = ['ma5', 'ma10', 'ma20', 'ma60']
            missing_mas = [col for col in ma_columns if col not in result_data.columns]
            if missing_mas:
                self.logger.warning(f"缺少均线数据: {missing_mas}")
                result_data['ma_support_strength'] = 0.0
                return result_data

            # 向量化计算均线粘合度（只对目标日期）
            result_data = self._calculate_ma_convergence_vectorized(result_data)

            self.logger.info("目标日期均线粘合度特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"目标日期均线粘合度特征计算失败: {e}")
            result_data['ma_support_strength'] = 0.0
            return result_data

    def _calculate_target_ma_relative_features(self, result_data: pd.DataFrame) -> pd.DataFrame:
        """只为目标日期计算均线相对特征"""
        try:
            self.logger.info("计算目标日期均线相对特征...")

            # 向量化计算均线相对特征（只对目标日期）
            result_data = self._calculate_ma_relative_vectorized(result_data)

            self.logger.info("目标日期均线相对特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"目标日期均线相对特征计算失败: {e}")
            result_data['ma_alignment_score'] = 0.0
            result_data['ma_convergence_score'] = 0.0
            return result_data

    def _calculate_target_professional_ma_indicators(self, result_data: pd.DataFrame) -> pd.DataFrame:
        """只为目标日期计算专业量化指标"""
        try:
            self.logger.info("计算目标日期专业量化指标...")

            # 向量化计算专业量化指标（只对目标日期）
            result_data = self._calculate_professional_ma_indicators_vectorized(result_data)

            self.logger.info("目标日期专业量化指标计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"目标日期专业量化指标计算失败: {e}")
            # 设置默认值
            for feature in ['ma_deviation_rate_5d', 'ma_deviation_rate_20d', 'ma_trend_strength', 'ma_volatility_ratio']:
                result_data[feature] = 0.0
            return result_data

    def _calculate_target_trend_relationship_features(self, result_data: pd.DataFrame) -> pd.DataFrame:
        """只为目标日期计算趋势间关系特征"""
        try:
            self.logger.info("计算目标日期趋势间关系特征...")

            # 向量化计算趋势间关系特征（只对目标日期）
            result_data = self._calculate_trend_relationship_vectorized(result_data)

            self.logger.info("目标日期趋势间关系特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"目标日期趋势间关系特征计算失败: {e}")
            # 设置默认值
            for feature in ['ma_ratio_5_10', 'ma_ratio_5_20', 'ma_ratio_20_60', 'ma_ratio_10_30', 'ma_ratio_5_60']:
                result_data[feature] = 1.0
            return result_data

    def _calculate_target_price_trend_relationship_features(self, result_data: pd.DataFrame) -> pd.DataFrame:
        """只为目标日期计算价格趋势关系特征"""
        try:
            self.logger.info("计算目标日期价格趋势关系特征...")

            # 向量化计算价格趋势关系特征（只对目标日期）
            result_data = self._calculate_price_trend_relationship_vectorized(result_data)

            self.logger.info("目标日期价格趋势关系特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"目标日期价格趋势关系特征计算失败: {e}")
            # 设置默认值
            for feature in ['price_ma_ratio_5', 'price_ma_ratio_10', 'price_ma_ratio_20', 'price_ma_ratio_60', 'price_ma_ratio_120']:
                result_data[feature] = 1.0
            return result_data

    # 注意：_calculate_target_advanced_ma_features 和 _calculate_target_high_quality_uptrend_score
    # 函数已删除，因为这些特征需要历史数据，现在在 _calculate_complex_features_on_full_data 中处理

    def _calculate_professional_ma_indicators_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算专业量化指标（目标日期版本）"""
        try:
            # 调用现有的向量化函数
            return self._calculate_professional_indicators_vectorized(data)
        except Exception as e:
            self.logger.error(f"专业量化指标向量化计算失败: {e}")
            return data

    def _calculate_trend_relationship_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算趋势间关系特征（目标日期版本）"""
        try:
            # 调用现有的向量化函数
            return self._calculate_trend_ratios_vectorized(data)
        except Exception as e:
            self.logger.error(f"趋势间关系特征向量化计算失败: {e}")
            return data

    def _calculate_price_trend_relationship_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算价格趋势关系特征（目标日期版本）"""
        try:
            # 调用现有的向量化函数
            return self._calculate_price_ma_ratios_vectorized(data)
        except Exception as e:
            self.logger.error(f"价格趋势关系特征向量化计算失败: {e}")
            return data

    def _calculate_advanced_ma_features_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算高级MA特征（目标日期版本）"""
        try:
            # 调用现有的高级特征计算函数
            data = self._calculate_advanced_ma_features(data)
            return data
        except Exception as e:
            self.logger.error(f"高级MA特征向量化计算失败: {e}")
            return data

    def _calculate_high_quality_uptrend_score_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算高质量上升趋势评分（目标日期版本）"""
        try:
            # 调用现有的向量化函数
            return self._calculate_uptrend_score_vectorized(data)
        except Exception as e:
            self.logger.error(f"高质量上升趋势评分向量化计算失败: {e}")
            return data

    def _calculate_ma_convergence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算均线粘合度特征（高性能监控版本）"""
        try:
            start_time = time.time()
            self.logger.info("    🔄 计算均线粘合度特征...")

            # 检查必要的均线数据
            ma_columns = ['ma5', 'ma10', 'ma20', 'ma60']
            missing_mas = [col for col in ma_columns if col not in data.columns]
            if missing_mas:
                self.logger.warning(f"    ⚠️ 缺少均线数据: {missing_mas}")
                # 初始化缺失特征为0
                features_to_init = ['ma_support_strength']
                for feature in features_to_init:
                    data[feature] = 0.0
                self.logger.info(f"    ✅ 已初始化 {len(features_to_init)} 个特征为默认值")
                return data

            # 数据质量检查
            total_records = len(data)
            valid_ma_records = data[ma_columns].notna().all(axis=1).sum()
            self.logger.info(f"    📊 数据质量: {valid_ma_records}/{total_records} 条记录有完整均线数据")

            # 向量化计算均线粘合度
            data = self._calculate_ma_convergence_vectorized(data)

            # 验证计算结果
            support_stats = data['ma_support_strength'].describe()
            self.logger.info(f"    📈 ma_support_strength: 均值={support_stats['mean']:.4f}, "
                            f"范围=[{support_stats['min']:.4f}, {support_stats['max']:.4f}]")

            total_duration = time.time() - start_time
            self.logger.info(f"    ✅ 均线粘合度特征完成，耗时: {total_duration:.3f}s")
            return data

        except Exception as e:
            self.logger.error(f"    ❌ 均线粘合度特征计算失败: {e}")
            import traceback
            self.logger.error(f"    错误详情: {traceback.format_exc()}")
            return data

    def _calculate_ma_convergence_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算均线粘合度特征"""
        try:
            # 1. 均线相对距离（衡量均线粘合程度）- 向量化计算
            ma5 = data['ma5'].values
            ma10 = data['ma10'].values
            ma20 = data['ma20'].values
            ma60 = data['ma60'].values
            close = data['close'].values


            # 3. 均线支撑强度（股价在所有均线之上的程度）- 向量化计算
            # 计算股价高于多少条均线
            above_ma5 = (close > ma5) & (ma5 > 0)
            above_ma10 = (close > ma10) & (ma10 > 0)
            above_ma20 = (close > ma20) & (ma20 > 0)
            above_ma60 = (close > ma60) & (ma60 > 0)

            # 有效均线数量
            valid_ma_count = (ma5 > 0).astype(int) + (ma10 > 0).astype(int) + \
                           (ma20 > 0).astype(int) + (ma60 > 0).astype(int)

            # 高于均线的数量
            above_count = above_ma5.astype(int) + above_ma10.astype(int) + \
                         above_ma20.astype(int) + above_ma60.astype(int)

            data['ma_support_strength'] = np.where(
                valid_ma_count > 0,
                above_count / valid_ma_count,
                0
            )

            return data

        except Exception as e:
            self.logger.error(f"向量化计算均线粘合度特征失败: {e}")
            return data

    # 旧方法已被向量化版本替代，删除以提升性能







    def _calculate_atr(self, close: np.ndarray, n: int, period: int = 14) -> np.ndarray:
        """计算ATR（Average True Range）"""
        try:
            # 转换为pandas Series进行向量化计算
            close_series = pd.Series(close)

            # 计算真实波动范围（简化版：基于收盘价变化）
            true_range = close_series.diff().abs()

            # 使用pandas rolling计算ATR
            atr_series = true_range.rolling(period, min_periods=1).mean()

            # 填充第一个值为0
            atr_series.iloc[0] = 0

            return atr_series.values

        except Exception as e:
            self.logger.error(f"计算ATR失败: {e}")
            return np.ones(n) * 0.01  # 返回默认值避免除零错误

    def _calculate_ma_relative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算均线相对特征（高性能向量化版本）"""
        try:
            self.logger.info("开始计算均线相对特征（向量化）")

            # 向量化计算均线相对特征
            data = self._calculate_ma_relative_vectorized(data)

            self.logger.info("均线相对特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"均线相对特征计算失败: {e}")
            return data

    def _calculate_ma_relative_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算均线相对特征"""
        try:
            # 检查必要的均线列
            ma_cols = ['ma5', 'ma10', 'ma20', 'ma60']
            available_mas = [col for col in ma_cols if col in data.columns]

            if len(available_mas) < 3:
                # 初始化为0
                data['ma_alignment_score'] = 0.0
                data['ma_convergence_score'] = 0.0
                return data

            # 获取均线数据
            ma5 = data['ma5'].values if 'ma5' in data.columns else np.zeros(len(data))
            ma10 = data['ma10'].values if 'ma10' in data.columns else np.zeros(len(data))
            ma20 = data['ma20'].values if 'ma20' in data.columns else np.zeros(len(data))
            ma60 = data['ma60'].values if 'ma60' in data.columns else np.zeros(len(data))
            ma120 = data['ma120'].values if 'ma120' in data.columns else np.zeros(len(data))

            # 1. 均线排列评分（向量化计算）
            alignment_score = np.zeros(len(data))

            # 计算各均线对的符号差异
            if 'ma5' in available_mas and 'ma10' in available_mas:
                alignment_score += np.sign(ma5 - ma10)
            if 'ma10' in available_mas and 'ma20' in available_mas:
                alignment_score += np.sign(ma10 - ma20)
            if 'ma20' in available_mas and 'ma60' in available_mas:
                alignment_score += np.sign(ma20 - ma60)
            if 'ma60' in data.columns and 'ma120' in data.columns:
                alignment_score += np.sign(ma60 - ma120)

            data['ma_alignment_score'] = alignment_score

            # 2. 均线收敛度评分（向量化计算）
            # 创建均线矩阵
            ma_matrix = np.column_stack([data[col].values for col in available_mas])

            # 计算每行的最大值、最小值和均值
            ma_max = np.nanmax(ma_matrix, axis=1)
            ma_min = np.nanmin(ma_matrix, axis=1)
            ma_mean = np.nanmean(ma_matrix, axis=1)

            # 收敛度 = 1 - (最大值 - 最小值) / 均值
            data['ma_convergence_score'] = np.where(
                ma_mean > 0,
                1.0 - (ma_max - ma_min) / ma_mean,
                0.0
            )



            return data

        except Exception as e:
            self.logger.error(f"向量化计算均线相对特征失败: {e}")
            # 初始化为0
            data['ma_alignment_score'] = 0.0
            data['ma_convergence_score'] = 0.0
            return data

        except Exception as e:
            self.logger.error(f"均线相对特征计算失败: {e}")
            return data

    def _calculate_professional_ma_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算专业量化均线指标（高性能监控版本）"""
        try:
            start_time = time.time()
            self.logger.info("    🔄 计算专业量化均线指标...")

            # 检查必要的列
            required_cols = ['close', 'ma5', 'ma20']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"    ⚠️ 缺少必要列: {missing_cols}")
                # 初始化缺失特征
                features_to_init = ['ma_deviation_rate_5d', 'ma_deviation_rate_20d',
                                   'ma_trend_strength', 'ma_volatility_ratio']
                for feature in features_to_init:
                    data[feature] = 0.0
                self.logger.info(f"    ✅ 已初始化 {len(features_to_init)} 个专业指标为默认值")
                return data

            # 数据质量检查
            total_records = len(data)
            valid_records = data[required_cols].notna().all(axis=1).sum()
            self.logger.info(f"    📊 数据质量: {valid_records}/{total_records} 条记录有完整专业指标数据")

            # 向量化计算专业指标
            data = self._calculate_professional_indicators_vectorized(data)

            # 验证计算结果 - 只显示关键指标
            key_indicators = ['ma_trend_strength']
            indicator_summary = []
            for indicator in key_indicators:
                if indicator in data.columns:
                    nonzero_count = (data[indicator] != 0).sum()
                    indicator_summary.append(f"{indicator}({nonzero_count}个)")

            self.logger.info(f"    📈 专业指标: {', '.join(indicator_summary)}")

            total_duration = time.time() - start_time
            self.logger.info(f"    ✅ 专业量化均线指标完成，耗时: {total_duration:.3f}s")
            return data

        except Exception as e:
            self.logger.error(f"    ❌ 专业量化均线指标计算失败: {e}")
            import traceback
            self.logger.error(f"    错误详情: {traceback.format_exc()}")
            return data

    def _calculate_professional_indicators_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算专业量化指标"""
        try:
            # 获取数据
            close = data['close'].values
            ma5 = data['ma5'].values if 'ma5' in data.columns else np.zeros(len(data))
            ma20 = data['ma20'].values if 'ma20' in data.columns else np.zeros(len(data))

            # 1. 均线乖离率（向量化计算）
            data['ma_deviation_rate_5d'] = np.where(
                ma5 > 0,
                (close - ma5) / ma5,
                0.0
            )

            data['ma_deviation_rate_20d'] = np.where(
                ma20 > 0,
                (close - ma20) / ma20,
                0.0
            )



            # 2. 均线趋势强度（向量化计算）
            # 使用MA20的变化率作为趋势强度
            ma20_prev = np.roll(ma20, 1)
            trend_strength = np.where(
                ma20_prev > 0,
                (ma20 - ma20_prev) / ma20_prev,
                0.0
            )
            trend_strength[0] = 0  # 第一个值设为0
            data['ma_trend_strength'] = trend_strength

            # 3. 均线波动率比率（向量化计算）
            # 计算MA5和MA20的相对波动率
            ma5_volatility = pd.Series(ma5).rolling(window=5, min_periods=5).std().values
            ma20_volatility = pd.Series(ma20).rolling(window=5, min_periods=5).std().values

            volatility_ratio = np.where(
                ma20_volatility > 1e-8,
                ma5_volatility / ma20_volatility,
                1.0
            )
            data['ma_volatility_ratio'] = volatility_ratio

            return data

        except Exception as e:
            self.logger.error(f"向量化计算专业指标失败: {e}")
            # 初始化为0（删除重复的MACD特征）
            for feature in ['ma_deviation_rate_5d', 'ma_deviation_rate_20d',
                           'ma_trend_strength', 'ma_volatility_ratio']:
                data[feature] = 0.0
            return data

    def _calculate_trend_relationship_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势间关系特征 (Trend vs. Trend)（高性能监控版本）
        量化不同时间周期趋势之间的关系
        """
        try:
            start_time = time.time()
            self.logger.info("    🔄 计算趋势间关系特征...")

            # 向量化计算趋势关系特征
            data = self._calculate_trend_ratios_vectorized(data)

            # 验证计算结果
            trend_features = ['ma_ratio_5_20', 'ma_ratio_20_60', 'ma_ratio_10_30', 'ma_ratio_5_60']
            valid_features = [f for f in trend_features if f in data.columns]
            self.logger.info(f"    📈 生成趋势比率特征: {len(valid_features)} 个")

            total_duration = time.time() - start_time
            self.logger.info(f"    ✅ 趋势间关系特征完成，耗时: {total_duration:.3f}s")
            return data

        except Exception as e:
            self.logger.error(f"    ❌ 趋势间关系特征计算失败: {e}")
            import traceback
            self.logger.error(f"    错误详情: {traceback.format_exc()}")
            return data

    def _calculate_trend_ratios_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算趋势比率特征"""
        try:
            # 获取均线数据
            ma5 = data['ma5'].values if 'ma5' in data.columns else np.ones(len(data))
            ma10 = data['ma10'].values if 'ma10' in data.columns else np.ones(len(data))
            ma20 = data['ma20'].values if 'ma20' in data.columns else np.ones(len(data))
            ma60 = data['ma60'].values if 'ma60' in data.columns else np.ones(len(data))

            # 1. MA5/MA10比率 (短期vs短中期趋势) - 向量化计算
            data['ma_ratio_5_10'] = np.where(
                (ma10 > 0) & (~np.isnan(ma10)) & (~np.isnan(ma5)),
                ma5 / ma10,
                1.0
            )

            # 2. MA5/MA20比率 (短期vs中期趋势) - 向量化计算
            data['ma_ratio_5_20'] = np.where(
                (ma20 > 0) & (~np.isnan(ma20)) & (~np.isnan(ma5)),
                ma5 / ma20,
                1.0
            )

            # 3. MA20/MA60比率 (中期vs长期趋势) - 向量化计算
            data['ma_ratio_20_60'] = np.where(
                (ma60 > 0) & (~np.isnan(ma60)) & (~np.isnan(ma20)),
                ma20 / ma60,
                1.0
            )

            # 4. MA10/MA30比率 (短中期vs中期趋势) - 向量化计算
            # 如果没有MA30，使用MA20作为替代
            if 'ma30' in data.columns:
                ma30 = data['ma30'].values
                data['ma_ratio_10_30'] = np.where(
                    (ma30 > 0) & (~np.isnan(ma30)) & (~np.isnan(ma10)),
                    ma10 / ma30,
                    1.0
                )
            else:
                # 使用MA20作为MA30的替代
                data['ma_ratio_10_30'] = np.where(
                    (ma20 > 0) & (~np.isnan(ma20)) & (~np.isnan(ma10)),
                    ma10 / ma20,
                    1.0
                )

            # 5. MA5/MA60比率 (短期vs长期趋势) - 向量化计算
            data['ma_ratio_5_60'] = np.where(
                (ma60 > 0) & (~np.isnan(ma60)) & (~np.isnan(ma5)),
                ma5 / ma60,
                1.0
            )

            return data

        except Exception as e:
            self.logger.error(f"向量化计算趋势比率失败: {e}")
            # 初始化为1.0
            data['ma_ratio_5_10'] = 1.0
            data['ma_ratio_5_20'] = 1.0
            data['ma_ratio_20_60'] = 1.0
            data['ma_ratio_10_30'] = 1.0
            data['ma_ratio_5_60'] = 1.0
            return data

    def _calculate_price_trend_relationship_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格与趋势关系特征 (Price vs. Trend)（高性能监控版本）
        量化当前价格相对于不同周期趋势的位置
        """
        try:
            start_time = time.time()
            self.logger.info("    🔄 计算价格与趋势关系特征...")

            # 检查必要的列
            if 'close' not in data.columns:
                self.logger.warning("    ⚠️ 缺少close列")
                # 初始化为1.0
                features_to_init = ['price_ma_ratio_5', 'price_ma_ratio_10', 'price_ma_ratio_20',
                                   'price_ma_ratio_60', 'price_ma_ratio_120']
                for feature in features_to_init:
                    data[feature] = 1.0
                self.logger.info(f"    ✅ 已初始化 {len(features_to_init)} 个价格比率特征为默认值")
                return data

            # 向量化计算价格趋势关系特征
            data = self._calculate_price_ma_ratios_vectorized(data)

            # 验证计算结果
            price_features = ['price_ma_ratio_5', 'price_ma_ratio_10', 'price_ma_ratio_20', 'price_ma_ratio_60', 'price_ma_ratio_120']
            valid_features = [f for f in price_features if f in data.columns]

            total_duration = time.time() - start_time
            self.logger.info(f"    ✅ 价格与趋势关系特征完成，生成 {len(valid_features)} 个特征，耗时: {total_duration:.3f}s")
            return data

        except Exception as e:
            self.logger.error(f"价格与趋势关系特征计算失败: {e}")
            return data

    def _calculate_price_ma_ratios_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算价格与均线比率特征"""
        try:
            # 获取价格数据
            close = data['close'].values

            # 获取均线数据
            ma5 = data['ma5'].values if 'ma5' in data.columns else np.ones(len(data))
            ma10 = data['ma10'].values if 'ma10' in data.columns else np.ones(len(data))
            ma20 = data['ma20'].values if 'ma20' in data.columns else np.ones(len(data))
            ma60 = data['ma60'].values if 'ma60' in data.columns else np.ones(len(data))
            ma120 = data['ma120'].values if 'ma120' in data.columns else np.ones(len(data))

            # 向量化计算价格与各均线的比率
            data['price_ma_ratio_5'] = np.where(
                (ma5 > 0) & (~np.isnan(ma5)) & (~np.isnan(close)),
                close / ma5,
                1.0
            )

            data['price_ma_ratio_10'] = np.where(
                (ma10 > 0) & (~np.isnan(ma10)) & (~np.isnan(close)),
                close / ma10,
                1.0
            )

            data['price_ma_ratio_20'] = np.where(
                (ma20 > 0) & (~np.isnan(ma20)) & (~np.isnan(close)),
                close / ma20,
                1.0
            )

            data['price_ma_ratio_60'] = np.where(
                (ma60 > 0) & (~np.isnan(ma60)) & (~np.isnan(close)),
                close / ma60,
                1.0
            )

            data['price_ma_ratio_120'] = np.where(
                (ma120 > 0) & (~np.isnan(ma120)) & (~np.isnan(close)),
                close / ma120,
                1.0
            )

            return data

        except Exception as e:
            self.logger.error(f"向量化计算价格均线比率失败: {e}")
            # 初始化为1.0
            for feature in ['price_ma_ratio_5', 'price_ma_ratio_10', 'price_ma_ratio_20',
                           'price_ma_ratio_60', 'price_ma_ratio_120']:
                data[feature] = 1.0
            return data

    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> tuple:
        """
        精确识别目标计算日期和历史支撑数据（强制要求明确日期参数）

        严格逻辑：
        1. 强制要求kwargs中提供target_start_date和target_end_date
        2. 添加时间边界检查，确保不包含未来日期
        3. 根据因子需要的历史数据长度，智能确定历史支撑数据范围
        4. 只为目标日期保存计算结果
        """
        try:
            # 1. 强制要求明确的目标日期范围
            target_start_date = kwargs.get('target_start_date')
            target_end_date = kwargs.get('target_end_date')

            if not target_start_date or not target_end_date:
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

            self.logger.info(f"📊 目标日期识别完成: {len(target_dates)} 个目标日期, {len(historical_dates)} 个历史日期")

            return target_dates, historical_dates

        except Exception as e:
            self.logger.error(f"❌ 识别目标和历史日期失败: {e}")
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

    def _calculate_advanced_ma_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算高级移动平均线特征（高性能监控版本）
        包括长期趋势、趋势质量、回调支撑等9个新特征
        """
        try:
            start_time = time.time()
            self.logger.info("    🔄 计算高级MA特征...")

            # 首先计算所需的基础特征
            data = self._calculate_missing_base_features(data)

            # 1. 长期趋势向上特征
            data = self._calculate_long_term_trend_features(data)

            # 2. 趋势缓慢与健康特征
            data = self._calculate_trend_quality_features(data)

            # 3. 回调到关键支撑特征
            data = self._calculate_pullback_support_features(data)

            # 统计生成的高级特征
            advanced_features = [
                'long_term_trend_up', 'trend_slow_and_healthy', 'pullback_to_key_support',
                'ma60_slope', 'ma120_slope', 'ma_trend_consistency', 'ma_trend_acceleration',
                'ma_support_level', 'ma_resistance_level'
            ]
            generated_count = sum(1 for f in advanced_features if f in data.columns)

            total_duration = time.time() - start_time
            self.logger.info(f"    ✅ 高级MA特征完成，生成 {generated_count} 个特征，耗时: {total_duration:.3f}s")
            return data

        except Exception as e:
            self.logger.error(f"高级MA特征计算失败: {e}")
            self._set_advanced_features_default(data)
            return data

    def _calculate_missing_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算缺失的基础特征"""
        try:
            # 计算volatility_60d（60日波动率）
            if 'volatility_60d' not in data.columns and 'close' in data.columns:
                returns = data.groupby('stock_code')['close'].pct_change(fill_method=None)
                data['volatility_60d'] = returns.groupby(data['stock_code']).rolling(60, min_periods=20).std().reset_index(0, drop=True)
                data['volatility_60d'] = data['volatility_60d'].fillna(0.01)  # 避免除零错误
                self.logger.info("计算volatility_60d完成")

            # 计算volatility_120d（120日波动率）
            if 'volatility_120d' not in data.columns and 'close' in data.columns:
                returns = data.groupby('stock_code')['close'].pct_change(fill_method=None)
                data['volatility_120d'] = returns.groupby(data['stock_code']).rolling(120, min_periods=30).std().reset_index(0, drop=True)
                self.logger.info("计算volatility_120d完成")

            # 🚀 计算ma20_slope（20日均线斜率）
            if 'ma20_slope' not in data.columns and 'ma20' in data.columns:
                # 计算ma20的5日变化率作为斜率
                data['ma20_slope'] = data.groupby('stock_code')['ma20'].pct_change(5, fill_method=None).reset_index(0, drop=True)
                data['ma20_slope'] = data['ma20_slope'].fillna(0.0)
                self.logger.info("计算ma20_slope完成")
            elif 'ma20_slope' not in data.columns and 'close' in data.columns:
                # 如果没有ma20，使用close计算20日均线再计算斜率
                self.logger.info("ma20不存在，使用close计算20日均线和斜率")
                ma20_temp = data.groupby('stock_code')['close'].rolling(20, min_periods=10).mean().reset_index(0, drop=True)
                data['ma20_slope'] = ma20_temp.groupby(data['stock_code']).pct_change(5, fill_method=None).reset_index(0, drop=True)
                data['ma20_slope'] = data['ma20_slope'].fillna(0.0)
                self.logger.info("基于close计算ma20_slope完成")

            # 🚀 计算ma60_slope（60日均线斜率）
            if 'ma60_slope' not in data.columns and 'ma60' in data.columns:
                # 计算ma60的5日变化率作为斜率
                data['ma60_slope'] = data.groupby('stock_code')['ma60'].pct_change(5, fill_method=None).reset_index(0, drop=True)
                data['ma60_slope'] = data['ma60_slope'].fillna(0.0)
                self.logger.info("计算ma60_slope完成")
            elif 'ma60_slope' not in data.columns and 'close' in data.columns:
                # 如果没有ma60，使用close计算60日均线再计算斜率
                self.logger.info("ma60不存在，使用close计算60日均线和斜率")
                ma60_temp = data.groupby('stock_code')['close'].rolling(60, min_periods=30).mean().reset_index(0, drop=True)
                data['ma60_slope'] = ma60_temp.groupby(data['stock_code']).pct_change(5, fill_method=None).reset_index(0, drop=True)
                data['ma60_slope'] = data['ma60_slope'].fillna(0.0)
                self.logger.info("基于close计算ma60_slope完成")

            # 🚀 计算ma120_slope（120日均线斜率）
            if 'ma120_slope' not in data.columns and 'ma120' in data.columns:
                # 计算ma120的5日变化率作为斜率
                data['ma120_slope'] = data.groupby('stock_code')['ma120'].pct_change(5, fill_method=None).reset_index(0, drop=True)
                data['ma120_slope'] = data['ma120_slope'].fillna(0.0)
                self.logger.info("计算ma120_slope完成")
            elif 'ma120_slope' not in data.columns and 'close' in data.columns:
                # 如果没有ma120，使用close计算120日均线再计算斜率
                self.logger.info("ma120不存在，使用close计算120日均线和斜率")
                ma120_temp = data.groupby('stock_code')['close'].rolling(120, min_periods=60).mean().reset_index(0, drop=True)
                data['ma120_slope'] = ma120_temp.groupby(data['stock_code']).pct_change(5, fill_method=None).reset_index(0, drop=True)
                data['ma120_slope'] = data['ma120_slope'].fillna(0.0)
                self.logger.info("基于close计算ma120_slope完成")

            # 计算high_120d（120日最高价）
            if 'high_120d' not in data.columns and 'high' in data.columns:
                # ✅ 正确逻辑：T日收盘后使用T日数据是合理的
                # 时间逻辑：T日收盘后计算特征，T+1日开盘前选股，无数据泄露
                # 对于第t日，high_120d是第(t-119)日到第t日的120天最高价
                data['high_120d'] = data.groupby('stock_code')['high'].rolling(120, min_periods=60).max().reset_index(0, drop=True)
                self.logger.info("✅ 计算high_120d完成（时间逻辑正确）")

            return data

        except Exception as e:
            self.logger.error(f"❌ 计算基础特征失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return data

    def _calculate_long_term_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算长期趋势向上特征"""
        try:
            # 1. ma_ratio_60_120 = ma60 / ma120
            if all(col in data.columns for col in ['ma60', 'ma120']):
                data['ma_ratio_60_120'] = data['ma60'] / data['ma120'].replace(0, np.nan)
                data['ma_ratio_60_120'] = data['ma_ratio_60_120'].fillna(1.0)
            else:
                data['ma_ratio_60_120'] = 1.0
                self.logger.warning("缺少ma60或ma120，ma_ratio_60_120设为1.0")

            # 2. ma_alignment_score_extended（扩展版均线排列得分）
            data = self._calculate_extended_ma_alignment(data)

            return data

        except Exception as e:
            self.logger.error(f"计算长期趋势特征失败: {e}")
            data['ma_ratio_60_120'] = 1.0
            data['ma_alignment_score_extended'] = 0.0
            return data

    def _calculate_extended_ma_alignment(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算扩展的均线排列得分（完全向量化高性能版）"""
        try:
            start_time = time.time()
            ma_cols = ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']
            available_mas = [col for col in ma_cols if col in data.columns]

            if len(available_mas) < 4:
                data['ma_alignment_score_extended'] = 0.0
                return data

            # 🚀 完全向量化计算均线排列得分
            ma_matrix = data[available_mas].values
            n_rows = ma_matrix.shape[0]

            # 处理NaN和零值
            ma_matrix = np.where((np.isnan(ma_matrix)) | (ma_matrix <= 0), np.nan, ma_matrix)

            # 计算每行的有效均线数量
            valid_counts = np.sum(~np.isnan(ma_matrix), axis=1)
            valid_mask = valid_counts >= 4

            # 初始化得分数组
            alignment_scores = np.zeros(n_rows)

            if np.any(valid_mask):
                # 🚀 向量化优化：预计算所有相邻对的比较
                # 创建相邻对比较矩阵 (n_rows, n_cols-1)
                # ma_matrix[:, :-1] >= ma_matrix[:, 1:] 检查每个相邻对是否满足多头排列
                adjacent_comparisons = ma_matrix[:, :-1] >= ma_matrix[:, 1:]

                # 处理NaN：如果任一均线为NaN，则该对比较无效
                valid_pairs = (~np.isnan(ma_matrix[:, :-1])) & (~np.isnan(ma_matrix[:, 1:]))
                adjacent_comparisons = adjacent_comparisons & valid_pairs

                # 🚀 向量化计算：每行正确排列的对数
                correct_pairs_count = np.sum(adjacent_comparisons, axis=1)

                # 🚀 向量化计算：每行有效的对数
                valid_pairs_count = np.sum(valid_pairs, axis=1)

                # 🚀 向量化计算得分
                # 避免除零：当valid_pairs_count为0时，得分为0
                with np.errstate(divide='ignore', invalid='ignore'):
                    partial_scores = np.where(
                        valid_pairs_count > 0,
                        correct_pairs_count / valid_pairs_count,
                        0.0
                    )

                # 🚀 向量化检查完美排列
                # 完美排列：所有有效对都正确排列
                perfect_alignment = (correct_pairs_count == valid_pairs_count) & (valid_pairs_count > 0)

                # 🚀 向量化计算最终得分
                # 完美排列得分 = 有效均线数量，部分排列得分 = 比例得分
                final_scores = np.where(
                    perfect_alignment,
                    valid_counts,  # 完美排列：得分等于有效均线数量
                    partial_scores  # 部分排列：得分等于正确比例
                )

                # 只保留有足够数据的行的得分
                alignment_scores = np.where(valid_mask, final_scores, 0.0)

            data['ma_alignment_score_extended'] = alignment_scores

            calc_time = time.time() - start_time
            valid_count = np.sum(valid_mask)
            self.logger.info(f"🚀 完全向量化均线排列计算完成，有效数据点: {valid_count}, 耗时: {calc_time:.4f}s")
            return data

        except Exception as e:
            self.logger.error(f"❌ 计算扩展均线排列得分失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            data['ma_alignment_score_extended'] = 0.0
            return data

    def _calculate_trend_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算趋势缓慢与健康特征（优化版）"""
        try:
            # 1. trend_quality_score = ma60_slope / volatility_120d
            if all(col in data.columns for col in ['ma60_slope', 'volatility_120d']):
                # 🚀 优化的趋势质量评分计算
                # 避免除零和异常值
                volatility_safe = data['volatility_120d'].replace(0, np.nan)
                volatility_safe = np.where(volatility_safe < 1e-6, np.nan, volatility_safe)  # 过滤极小值

                # 计算趋势质量评分
                trend_quality = data['ma60_slope'] / volatility_safe

                # 处理异常值：限制在合理范围内
                trend_quality = np.where(np.abs(trend_quality) > 100, np.sign(trend_quality) * 100, trend_quality)

                data['trend_quality_score'] = pd.Series(trend_quality).fillna(0.0)

                # 统计信息
                valid_scores = data['trend_quality_score'][data['trend_quality_score'] != 0]
                if len(valid_scores) > 0:
                    self.logger.info(f"✅ trend_quality_score计算完成，有效值: {len(valid_scores)}, "
                                   f"范围: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
                else:
                    self.logger.warning("⚠️ trend_quality_score计算完成，但无有效值")

            elif 'ma60_slope' in data.columns and 'volatility_120d' not in data.columns:
                # 🚀 备选方案1：只有ma60_slope，使用简化评分
                self.logger.info("🔄 使用ma60_slope简化计算trend_quality_score")
                # 使用ma60_slope的标准化值作为趋势质量评分
                ma60_slope_std = data['ma60_slope'].std()
                if ma60_slope_std > 0:
                    data['trend_quality_score'] = data['ma60_slope'] / ma60_slope_std
                else:
                    data['trend_quality_score'] = data['ma60_slope']
                self.logger.info("✅ 基于ma60_slope的简化trend_quality_score计算完成")

            elif 'volatility_120d' in data.columns and 'ma60_slope' not in data.columns:
                # 🚀 备选方案2：只有volatility_120d，使用波动率倒数
                self.logger.info("🔄 使用volatility_120d简化计算trend_quality_score")
                volatility_safe = data['volatility_120d'].replace(0, np.nan)
                data['trend_quality_score'] = (1 / volatility_safe).fillna(0.0)
                # 标准化到合理范围
                max_score = data['trend_quality_score'].quantile(0.95)
                if max_score > 0:
                    data['trend_quality_score'] = np.minimum(data['trend_quality_score'], max_score)
                self.logger.info("✅ 基于volatility_120d的简化trend_quality_score计算完成")

            else:
                # 🚀 备选方案3：都没有，使用基础数据计算
                if 'ma60' in data.columns and 'close' in data.columns:
                    self.logger.info("🔄 使用基础数据计算trend_quality_score")
                    # 计算简化的趋势质量：ma60相对于close的稳定性
                    ma60_close_ratio = data['ma60'] / data['close'].replace(0, np.nan)
                    data['trend_quality_score'] = ma60_close_ratio.fillna(1.0) - 1.0  # 转换为偏离度
                    self.logger.info("✅ 基于基础数据的trend_quality_score计算完成")
                else:
                    data['trend_quality_score'] = 0.0
                    self.logger.warning("⚠️ 缺少必要数据，trend_quality_score设为0.0")

            # 2. max_drawdown_in_uptrend_120d
            data = self._calculate_max_drawdown_120d(data)

            return data

        except Exception as e:
            self.logger.error(f"❌ 计算趋势质量特征失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            data['trend_quality_score'] = 0.0
            data['max_drawdown_in_uptrend_120d'] = 0.0
            return data

    def _calculate_max_drawdown_120d(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算120天内最大回撤（完全向量化高性能版）"""
        try:
            start_time = time.time()

            if 'close' not in data.columns:
                data['max_drawdown_in_uptrend_120d'] = 0.0
                return data

            # 🚀 完全向量化的最大回撤计算
            if 'stock_code' in data.columns:
                # 多股票情况：按股票分组处理
                result_list = []

                for _, stock_group in data.groupby('stock_code'):
                    stock_df = stock_group.sort_values('date').copy()
                    stock_df = self._calculate_single_stock_max_drawdown_vectorized(stock_df)
                    result_list.append(stock_df)

                # 合并结果
                result_data = pd.concat(result_list, ignore_index=True)
                # 按原始顺序恢复
                data = data.merge(
                    result_data[['date', 'stock_code', 'max_drawdown_in_uptrend_120d']],
                    on=['date', 'stock_code'],
                    how='left'
                )
                data['max_drawdown_in_uptrend_120d'] = data['max_drawdown_in_uptrend_120d'].fillna(0.0)
            else:
                # 单股票情况：直接计算
                data = self._calculate_single_stock_max_drawdown_vectorized(data)

            calc_time = time.time() - start_time
            valid_count = (data['max_drawdown_in_uptrend_120d'] != 0).sum()
            self.logger.info(f"🚀 完全向量化120日最大回撤计算完成，有效数据点: {valid_count}, 耗时: {calc_time:.4f}s")
            return data

        except Exception as e:
            self.logger.error(f"❌ 计算120日最大回撤失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            data['max_drawdown_in_uptrend_120d'] = 0.0
            return data

    def _calculate_single_stock_max_drawdown_vectorized(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """单股票完全向量化最大回撤计算（超高性能版）"""
        try:
            stock_df = stock_df.sort_values('date').copy()
            closes = stock_df['close'].values
            n = len(closes)

            if n <= 1:
                stock_df['max_drawdown_in_uptrend_120d'] = 0.0
                return stock_df

            # 🚀 超高性能：使用pandas内置的rolling操作
            closes_series = pd.Series(closes)

            # 方法1：使用pandas rolling + apply（针对复杂逻辑）
            def fast_max_drawdown(window_values):
                """快速计算窗口内最大回撤"""
                if len(window_values) <= 1:
                    return 0.0

                # 使用numpy的高效操作
                cummax = np.maximum.accumulate(window_values)
                drawdowns = (window_values - cummax) / cummax
                return np.min(drawdowns)

            # 使用pandas的高效rolling操作
            max_drawdowns = closes_series.rolling(
                window=120,
                min_periods=2
            ).apply(fast_max_drawdown, raw=True)  # raw=True使用numpy数组，更快

            # 填充NaN值
            max_drawdowns = max_drawdowns.fillna(0.0)

            stock_df['max_drawdown_in_uptrend_120d'] = max_drawdowns.values
            return stock_df

        except Exception as e:
            self.logger.error(f"❌ 单股票最大回撤计算失败: {e}")
            stock_df['max_drawdown_in_uptrend_120d'] = 0.0
            return stock_df

    def _calculate_pullback_support_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算回调到关键支撑特征（已修复数据泄露问题）"""
        try:
            # 🚀 1. price_position_vs_120d_high = (close - high_120d) / high_120d
            # 量化含义：当日收盘价相对于过去120日最高价的位置
            # 修复说明：high_120d现在只使用历史数据（不包含当日），避免数据泄露
            # 时间对齐：close(t) vs high_120d(t-120 to t-1)
            if all(col in data.columns for col in ['close', 'high_120d']):
                high_safe = data['high_120d'].replace(0, np.nan)
                data['price_position_vs_120d_high'] = (data['close'] - data['high_120d']) / high_safe
                data['price_position_vs_120d_high'] = data['price_position_vs_120d_high'].fillna(0.0)
                self.logger.info("✅ 计算price_position_vs_120d_high完成（无数据泄露）")
            else:
                data['price_position_vs_120d_high'] = 0.0
                self.logger.warning("⚠️ 缺少close或high_120d，price_position_vs_120d_high设为0.0")

            # 2. BIAS_60D = (close - ma60) / ma60
            if all(col in data.columns for col in ['close', 'ma60']):
                ma60_safe = data['ma60'].replace(0, np.nan)
                data['BIAS_60D'] = (data['close'] - data['ma60']) / ma60_safe
                data['BIAS_60D'] = data['BIAS_60D'].fillna(0.0)
                self.logger.info("计算BIAS_60D完成")
            else:
                data['BIAS_60D'] = 0.0
                self.logger.info("缺少close或ma60，BIAS_60D设为0.0")

            return data

        except Exception as e:
            self.logger.error(f"计算回调支撑特征失败: {e}")
            data['price_position_vs_120d_high'] = 0.0
            data['BIAS_60D'] = 0.0
            return data

    def _set_advanced_features_default(self, data: pd.DataFrame):
        """设置高级特征的默认值"""
        advanced_features = [
            'ma_ratio_60_120', 'ma_alignment_score_extended',
            'trend_quality_score', 'max_drawdown_in_uptrend_120d',
            'price_position_vs_120d_high', 'BIAS_60D',
            'high_quality_uptrend_score'
        ]

        for feature in advanced_features:
            if feature in ['ma_ratio_60_120']:
                data[feature] = 1.0  # 比率类特征默认为1
            else:
                data[feature] = 0.0  # 其他特征默认为0

    def _set_default_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """设置默认值"""
        for factor_name in self.feature_names:
            data[factor_name] = 0.0
        return data

    def _calculate_high_quality_uptrend_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算高质量上升趋势评分特征（高性能监控版本）

        该特征综合评估股票的高质量上升趋势强度，包括：
        1. 均线斜率得分：ma20和ma60的斜率是否向上
        2. 趋势质量得分：趋势斜率相对于波动率的稳定性
        3. 均线排列得分：短期均线是否在长期均线之上
        4. 价格行为得分：当前价格是否在关键均线之上

        Returns:
            pd.DataFrame: 包含high_quality_uptrend_score特征的数据
        """
        try:
            start_time = time.time()
            self.logger.info("    🔄 计算高质量上升趋势评分特征...")

            # 确保所需的基础特征存在
            data = self._ensure_required_features_for_uptrend_score(data)

            # 向量化计算高质量上升趋势评分
            data = self._calculate_uptrend_score_vectorized(data)

            # 性能统计
            duration = time.time() - start_time
            valid_scores = data['high_quality_uptrend_score'].dropna()

            if len(valid_scores) > 0:
                self.logger.info(f"    ✅ 高质量上升趋势评分计算完成，耗时: {duration:.2f}s")
                self.logger.info(f"    📊 有效评分: {len(valid_scores)} 个")
                self.logger.info(f"    📊 评分范围: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
                self.logger.info(f"    📊 平均评分: {valid_scores.mean():.4f}")
            else:
                self.logger.warning("    ⚠️ 高质量上升趋势评分计算完成，但无有效值")

            return data

        except Exception as e:
            self.logger.error(f"    ❌ 高质量上升趋势评分特征计算失败: {e}")
            import traceback
            self.logger.error(f"    错误详情: {traceback.format_exc()}")
            data['high_quality_uptrend_score'] = 0.0
            return data

    def _ensure_required_features_for_uptrend_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保计算高质量上升趋势评分所需的特征存在"""
        try:
            # 确保基础特征存在
            data = self._calculate_missing_base_features(data)

            # 检查所需特征
            required_features = ['ma5', 'ma20', 'ma60', 'ma20_slope', 'ma60_slope', 'volatility_60d', 'close']
            missing_features = [f for f in required_features if f not in data.columns]

            if missing_features:
                self.logger.warning(f"    ⚠️ 缺少特征: {missing_features}，将使用默认值")

                # 为缺失特征设置默认值
                for feature in missing_features:
                    if feature in ['ma5', 'ma20', 'ma60', 'close']:
                        data[feature] = data.get('close', 1.0)  # 使用收盘价作为默认值
                    elif feature in ['ma20_slope', 'ma60_slope']:
                        data[feature] = 0.0  # 斜率默认为0
                    elif feature == 'volatility_60d':
                        data[feature] = 0.01  # 波动率默认为0.01，避免除零

            return data

        except Exception as e:
            self.logger.error(f"确保上升趋势评分所需特征失败: {e}")
            return data

    def _calculate_uptrend_score_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """向量化计算高质量上升趋势评分"""
        try:
            # 获取权重参数
            w1 = self.uptrend_weights['ma_slope_weight']
            w2 = self.uptrend_weights['trend_quality_weight']
            w3 = self.uptrend_weights['ma_alignment_weight']
            w4 = self.uptrend_weights['price_action_weight']

            # 1. 均线斜率得分 (0或1的二进制评分)
            ma20_slope_positive = (data['ma20_slope'] > 0).astype(int)
            ma60_slope_positive = (data['ma60_slope'] > 0).astype(int)
            ma_slope_score = (ma20_slope_positive & ma60_slope_positive).astype(float)

            # 2. 趋势质量得分 (0或1的二进制评分)
            # 计算趋势质量：ma60_slope / volatility_60d
            volatility_safe = np.where(data['volatility_60d'] > 1e-6, data['volatility_60d'], 1e-6)
            trend_quality_ratio = data['ma60_slope'] / volatility_safe

            # 动态阈值：使用数据分布的75分位数作为阈值
            valid_ratios = trend_quality_ratio[np.isfinite(trend_quality_ratio)]
            if len(valid_ratios) > 0:
                threshold1 = np.percentile(valid_ratios, 75)
                if np.isnan(threshold1) or threshold1 <= 0:
                    threshold1 = 0.1  # 默认阈值
            else:
                threshold1 = 0.1  # 默认阈值

            trend_quality_score = (trend_quality_ratio > threshold1).astype(float)

            # 3. 均线排列得分 (0或1的二进制评分)
            ma5_above_ma20 = (data['ma5'] > data['ma20']).astype(int)
            ma20_above_ma60 = (data['ma20'] > data['ma60']).astype(int)
            ma_alignment_score = (ma5_above_ma20 & ma20_above_ma60).astype(float)

            # 4. 价格行为得分 (0或1的二进制评分)
            price_action_score = (data['close'] > data['ma20']).astype(float)

            # 5. 计算加权综合得分
            high_quality_uptrend_score = (
                w1 * ma_slope_score +
                w2 * trend_quality_score +
                w3 * ma_alignment_score +
                w4 * price_action_score
            )

            # 处理异常值和边界情况
            high_quality_uptrend_score = np.where(
                np.isfinite(high_quality_uptrend_score),
                high_quality_uptrend_score,
                0.0
            )

            # 确保评分在[0, 1]范围内
            high_quality_uptrend_score = np.clip(high_quality_uptrend_score, 0.0, 1.0)

            data['high_quality_uptrend_score'] = high_quality_uptrend_score

            # 记录计算统计信息
            self.logger.info(f"    📊 趋势质量阈值: {threshold1:.6f}")
            self.logger.info(f"    📊 各维度得分统计:")
            self.logger.info(f"        均线斜率得分平均值: {ma_slope_score.mean():.4f}")
            self.logger.info(f"        趋势质量得分平均值: {trend_quality_score.mean():.4f}")
            self.logger.info(f"        均线排列得分平均值: {ma_alignment_score.mean():.4f}")
            self.logger.info(f"        价格行为得分平均值: {price_action_score.mean():.4f}")

            return data

        except Exception as e:
            self.logger.error(f"向量化计算高质量上升趋势评分失败: {e}")
            data['high_quality_uptrend_score'] = 0.0
            return data
