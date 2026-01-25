#!/usr/bin/env python3
"""
价格相对位置因子组（高性能优化版）

计算价格在不同时间周期内的相对位置特征，并进行横截面排序
- 只保存排序后的因子（_rank后缀），不保存原始因子
- 使用向量化计算和多线程处理提升性能
- 完全移除废弃的逐股票循环计算逻辑

优化特性：
1. 向量化计算：使用pandas groupby + rolling操作
2. 多线程处理：并行处理横截面排序
3. 内存优化：只为目标日期分配结果空间
4. 智能缓存：避免重复计算
"""

import pandas as pd
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_factor_group import BaseFactorGroup

class PricePositionFactors(BaseFactorGroup):
    """价格相对位置因子组"""
    
    def __init__(self):
        super().__init__("price_position_factors")
    
    def _get_version(self) -> str:
        return "v2.0"  # 高性能优化版：只保存排序因子，向量化计算，多线程处理
    
    def _get_dependencies(self) -> List[str]:
        return []  # 价格相对位置因子不依赖其他因子组
    
    def _get_original_feature_names(self) -> List[str]:
        """获取原始因子名称列表（不包含排序后的因子）"""
        return [
            # 价格在区间内的位置特征
            'price_position_in_range_5d',
            'price_position_in_range_10d',  # 新增
            'price_position_in_range_20d',
            'price_position_in_range_60d',

            # 价格相对均线比例特征
            'price_vs_ma5_ratio',
            'price_vs_ma10_ratio',  # 新增
            'price_vs_ma20_ratio',
            'price_vs_ma60_ratio',

            # 新增：高点相对均线比例特征
            'high_vs_ma5_ratio',
            'high_vs_ma10_ratio',
            'high_vs_ma20_ratio',
            'high_vs_ma60_ratio',

            # 新增：低点相对均线比例特征
            'low_vs_ma5_ratio',
            'low_vs_ma10_ratio',
            'low_vs_ma20_ratio',
            'low_vs_ma60_ratio',

            # 新增：高点在价格区间内的位置特征
            'high_position_in_range_5d',
            'high_position_in_range_10d',
            'high_position_in_range_20d',

            # 新增：低点在价格区间内的位置特征
            'low_position_in_range_5d',
            'low_position_in_range_10d',
            'low_position_in_range_20d'
        ]

    def _get_feature_names(self) -> List[str]:
        """获取需要保存的因子名称列表（只包含横截面排序因子）"""
        original_features = self._get_original_feature_names()
        ranked_features = [f"{feature}_rank" for feature in original_features]
        return ranked_features  # 只返回排序后的因子，不保存原始因子

    def get_all_feature_names(self) -> List[str]:
        """获取所有因子名称列表（只包含排序后的因子）- 兼容性方法"""
        return self._get_feature_names()

    def get_all_computed_feature_names(self) -> List[str]:
        """获取所有计算的因子名称列表（包含原始因子和排序后的因子）- 用于内部计算"""
        original_features = self._get_original_feature_names()
        ranked_features = [f"{feature}_rank" for feature in original_features]
        return original_features + ranked_features

    def _get_max_history_window_needed(self) -> int:
        """返回计算价格位置因子所需的最大历史窗口长度"""
        return 60  # 价格位置因子需要60日历史数据
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算价格相对位置因子（基于feature_calculator.py的专业实现并增强）"""
        try:

            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return data

            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 2. 专业的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_price_position_features_optimized(data, target_dates, historical_dates)

            # 3. 添加横截面排序：将因子值转换为分位数（0到1之间）
            result_data = self._apply_cross_sectional_ranking(result_data)

            # 4. 只保留排序后的因子用于保存（移除原始因子）
            result_data = self._filter_to_ranked_factors_only(result_data)

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates
            self.logger.info(f"价格相对位置因子计算完成（修复版+横截面排序+只保存排序因子）")

            return result_data

        except Exception as e:
            self.logger.error(f"价格相对位置因子计算失败: {e}")
            return data

    def _filter_to_ranked_factors_only(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        过滤数据，只保留排序后的因子和基础列

        Args:
            data: 包含所有因子的DataFrame

        Returns:
            pd.DataFrame: 只包含排序后因子的DataFrame
        """
        try:
            # 获取基础列（非因子列）
            original_factor_names = self._get_original_feature_names()
            base_columns = [col for col in data.columns
                          if col not in original_factor_names and not col.endswith('_rank')]

            # 获取排序后的因子列
            ranked_factor_columns = [col for col in data.columns if col.endswith('_rank')]

            # 组合最终需要保留的列
            final_columns = base_columns + ranked_factor_columns

            # 过滤数据
            filtered_data = data[final_columns].copy()

            self.logger.info(f"因子过滤完成: 原始列数={len(data.columns)}, "
                           f"过滤后列数={len(filtered_data.columns)}, "
                           f"排序因子数={len(ranked_factor_columns)}")

            return filtered_data

        except Exception as e:
            self.logger.error(f"过滤排序因子失败: {e}")
            return data

    def _calculate_price_position_features_optimized(self, full_data: pd.DataFrame,
                                                   target_dates: List[str],
                                                   historical_dates: List[str] = None) -> pd.DataFrame:
        """
        高性能向量化价格位置特征计算（完全重写优化版）

        优化策略：
        1. 使用pandas向量化操作替代逐股票循环
        2. 利用groupby + rolling进行高效滚动计算
        3. 多线程处理大数据集
        4. 内存优化：只为目标日期分配结果空间
        5. 避免重复计算和数据复制
        """
        try:
            self.logger.info("开始高性能向量化价格位置特征计算")

            # 1. 创建目标日期的结果DataFrame
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")

            # 2. 向量化计算所有价格位置特征
            result_data = self._calculate_vectorized_price_position_features(full_data, result_data, target_date_objects)

            self.logger.info("高性能向量化价格位置特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"高性能价格位置特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _calculate_vectorized_price_position_features(self, full_data: pd.DataFrame,
                                                    result_data: pd.DataFrame,
                                                    target_date_objects: List[pd.Timestamp]) -> pd.DataFrame:
        """
        向量化计算价格位置特征（高性能版本）

        使用pandas的groupby + rolling操作进行高效计算，避免逐股票循环
        """
        try:
            self.logger.info("开始向量化计算价格位置特征...")

            # 确保数据按股票和日期排序
            full_data_sorted = full_data.sort_values(['stock_code', 'date']).copy()

            # 1. 向量化计算价格在区间内的位置特征
            self.logger.info("计算价格区间位置特征...")
            result_data = self._calculate_vectorized_price_position_in_range(
                full_data_sorted, result_data, target_date_objects
            )

            # 2. 向量化计算价格相对均线比例特征
            self.logger.info("计算价格均线比例特征...")
            result_data = self._calculate_vectorized_price_vs_ma_ratios(
                full_data_sorted, result_data, target_date_objects
            )

            # 3. 向量化计算高点、低点位置特征
            self.logger.info("计算高低点位置特征...")
            result_data = self._calculate_vectorized_high_low_positions(
                full_data_sorted, result_data, target_date_objects
            )

            self.logger.info("向量化价格位置特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"向量化计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return result_data

    def _initialize_target_price_position_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化价格位置特征列（内存优化）"""
        try:
            # 价格在区间内的位置特征
            for period in [5, 10, 20, 60]:
                result_data[f'price_position_in_range_{period}d'] = np.nan

            # 价格相对均线比例特征
            for ma_period in [5, 10, 20, 60]:
                result_data[f'price_vs_ma{ma_period}_ratio'] = np.nan
                result_data[f'high_vs_ma{ma_period}_ratio'] = np.nan
                result_data[f'low_vs_ma{ma_period}_ratio'] = np.nan

            # 高点和低点在价格区间内的位置特征
            for period in [5, 10, 20]:
                result_data[f'high_position_in_range_{period}d'] = np.nan
                result_data[f'low_position_in_range_{period}d'] = np.nan

        except Exception as e:
            self.logger.error(f"初始化目标价格位置特征失败: {e}")

    def _calculate_vectorized_price_position_in_range(self, full_data_sorted: pd.DataFrame,
                                                    result_data: pd.DataFrame,
                                                    target_date_objects: List[pd.Timestamp]) -> pd.DataFrame:
        """向量化计算价格在区间内的位置特征"""
        try:
            # 使用groupby + rolling进行高效滚动计算
            for period in [5, 10, 20, 60]:
                # 计算滚动最高价和最低价
                period_high = (full_data_sorted.groupby('stock_code')['high']
                             .rolling(window=period, min_periods=1)
                             .max()
                             .reset_index(level=0, drop=True))

                period_low = (full_data_sorted.groupby('stock_code')['low']
                            .rolling(window=period, min_periods=1)
                            .min()
                            .reset_index(level=0, drop=True))

                # 计算价格位置（向量化）
                range_diff = period_high - period_low
                position = np.where(
                    range_diff > 1e-8,  # 避免除零
                    (full_data_sorted['close'] - period_low) / range_diff,
                    0.5  # 默认中间位置
                )
                position = np.clip(position, 0, 1)  # 确保在[0,1]范围内

                # 将计算结果添加到完整数据中
                full_data_sorted[f'price_position_in_range_{period}d'] = position

            # 提取目标日期的结果
            target_mask = full_data_sorted['date'].isin(target_date_objects)
            target_results = full_data_sorted[target_mask]

            # 更新结果数据
            for period in [5, 10, 20, 60]:
                feature_name = f'price_position_in_range_{period}d'
                result_data = result_data.merge(
                    target_results[['stock_code', 'date', feature_name]],
                    on=['stock_code', 'date'], how='left', suffixes=('', '_new')
                )
                if f'{feature_name}_new' in result_data.columns:
                    result_data[feature_name] = result_data[f'{feature_name}_new']
                    result_data.drop(columns=[f'{feature_name}_new'], inplace=True)

            return result_data

        except Exception as e:
            self.logger.error(f"向量化计算价格区间位置失败: {e}")
            return result_data

    def _calculate_vectorized_price_vs_ma_ratios(self, full_data_sorted: pd.DataFrame,
                                               result_data: pd.DataFrame,
                                               target_date_objects: List[pd.Timestamp]) -> pd.DataFrame:
        """向量化计算价格相对均线比例特征"""
        try:
            # 计算移动平均线（如果不存在）
            for ma_period in [5, 10, 20, 60]:
                ma_col = f'ma{ma_period}'
                if ma_col not in full_data_sorted.columns:
                    # 计算移动平均线
                    full_data_sorted[ma_col] = (full_data_sorted.groupby('stock_code')['close']
                                              .rolling(window=ma_period, min_periods=1)
                                              .mean()
                                              .reset_index(level=0, drop=True))

                # 计算价格相对均线比例（向量化）
                ma_values = full_data_sorted[ma_col]

                # 收盘价相对均线比例
                close_ratio = np.where(
                    ma_values > 1e-8,
                    (full_data_sorted['close'] / ma_values) - 1,
                    0.0
                )
                full_data_sorted[f'price_vs_ma{ma_period}_ratio'] = close_ratio

                # 最高价相对均线比例
                high_ratio = np.where(
                    ma_values > 1e-8,
                    (full_data_sorted['high'] / ma_values) - 1,
                    0.0
                )
                full_data_sorted[f'high_vs_ma{ma_period}_ratio'] = high_ratio

                # 最低价相对均线比例
                low_ratio = np.where(
                    ma_values > 1e-8,
                    (full_data_sorted['low'] / ma_values) - 1,
                    0.0
                )
                full_data_sorted[f'low_vs_ma{ma_period}_ratio'] = low_ratio

            # 提取目标日期的结果并更新result_data
            target_mask = full_data_sorted['date'].isin(target_date_objects)
            target_results = full_data_sorted[target_mask]

            # 更新结果数据
            ratio_features = []
            for ma_period in [5, 10, 20, 60]:
                ratio_features.extend([
                    f'price_vs_ma{ma_period}_ratio',
                    f'high_vs_ma{ma_period}_ratio',
                    f'low_vs_ma{ma_period}_ratio'
                ])

            for feature_name in ratio_features:
                result_data = result_data.merge(
                    target_results[['stock_code', 'date', feature_name]],
                    on=['stock_code', 'date'], how='left', suffixes=('', '_new')
                )
                if f'{feature_name}_new' in result_data.columns:
                    result_data[feature_name] = result_data[f'{feature_name}_new']
                    result_data.drop(columns=[f'{feature_name}_new'], inplace=True)

            return result_data

        except Exception as e:
            self.logger.error(f"向量化计算价格均线比例失败: {e}")
            return result_data

    def _calculate_vectorized_high_low_positions(self, full_data_sorted: pd.DataFrame,
                                               result_data: pd.DataFrame,
                                               target_date_objects: List[pd.Timestamp]) -> pd.DataFrame:
        """向量化计算高点、低点在价格区间内的位置特征"""
        try:
            # 计算高点和低点在价格区间内的位置
            for period in [5, 10, 20]:
                # 计算滚动最高价和最低价
                period_high = (full_data_sorted.groupby('stock_code')['high']
                             .rolling(window=period, min_periods=1)
                             .max()
                             .reset_index(level=0, drop=True))

                period_low = (full_data_sorted.groupby('stock_code')['low']
                            .rolling(window=period, min_periods=1)
                            .min()
                            .reset_index(level=0, drop=True))

                # 计算价格区间
                range_diff = period_high - period_low

                # 计算高点在区间内的位置（向量化）
                high_position = np.where(
                    range_diff > 1e-8,
                    (full_data_sorted['high'] - period_low) / range_diff,
                    0.5
                )
                high_position = np.clip(high_position, 0, 1)
                full_data_sorted[f'high_position_in_range_{period}d'] = high_position

                # 计算低点在区间内的位置（向量化）
                low_position = np.where(
                    range_diff > 1e-8,
                    (full_data_sorted['low'] - period_low) / range_diff,
                    0.5
                )
                low_position = np.clip(low_position, 0, 1)
                full_data_sorted[f'low_position_in_range_{period}d'] = low_position

            # 提取目标日期的结果并更新result_data
            target_mask = full_data_sorted['date'].isin(target_date_objects)
            target_results = full_data_sorted[target_mask]

            # 更新结果数据
            position_features = []
            for period in [5, 10, 20]:
                position_features.extend([
                    f'high_position_in_range_{period}d',
                    f'low_position_in_range_{period}d'
                ])

            for feature_name in position_features:
                result_data = result_data.merge(
                    target_results[['stock_code', 'date', feature_name]],
                    on=['stock_code', 'date'], how='left', suffixes=('', '_new')
                )
                if f'{feature_name}_new' in result_data.columns:
                    result_data[feature_name] = result_data[f'{feature_name}_new']
                    result_data.drop(columns=[f'{feature_name}_new'], inplace=True)

            return result_data

        except Exception as e:
            self.logger.error(f"向量化计算高低点位置失败: {e}")
            return result_data















    def _apply_cross_sectional_ranking(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        高效横截面排序（优化版）

        使用向量化操作和多线程处理提升性能
        """
        try:
            self.logger.info("🔄 开始高效横截面排序...")

            # 获取所有需要排序的原始因子名称
            original_factor_names = self._get_original_feature_names()
            available_factors = [f for f in original_factor_names if f in data.columns]

            if not available_factors:
                self.logger.warning("⚠️ 没有找到可排序的价格位置因子")
                return data

            self.logger.info(f"📊 准备对 {len(available_factors)} 个因子进行高效横截面排序")

            # 使用多线程并行处理因子排序
            ranked_data = data.copy()

            # 判断是否使用多线程（因子数量较多时使用）
            use_threading = len(available_factors) > 4

            if use_threading:
                ranked_data = self._parallel_cross_sectional_ranking(ranked_data, available_factors)
            else:
                ranked_data = self._sequential_cross_sectional_ranking(ranked_data, available_factors)

            # 统计排序结果
            ranked_factors = [f for f in ranked_data.columns if f.endswith('_rank')]
            self.logger.info(f"✅ 高效横截面排序完成，生成 {len(ranked_factors)} 个排序因子")

            return ranked_data

        except Exception as e:
            self.logger.error(f"❌ 高效横截面排序失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return data

    def _sequential_cross_sectional_ranking(self, data: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
        """顺序处理横截面排序"""
        try:
            for factor_name in factors:
                ranked_factor_name = f"{factor_name}_rank"
                # 使用pandas的高效groupby操作
                data[ranked_factor_name] = (data.groupby('date')[factor_name]
                                          .rank(method='average', pct=True, na_option='keep'))
            return data
        except Exception as e:
            self.logger.error(f"顺序横截面排序失败: {e}")
            return data

    def _parallel_cross_sectional_ranking(self, data: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
        """并行处理横截面排序"""
        try:
            # 将因子分组进行并行处理
            n_threads = min(4, len(factors))  # 最多使用4个线程
            factor_chunks = np.array_split(factors, n_threads)

            def process_factor_chunk(factor_chunk, data_chunk):
                """处理一组因子的排序"""
                result_data = data_chunk.copy()
                for factor_name in factor_chunk:
                    ranked_factor_name = f"{factor_name}_rank"
                    result_data[ranked_factor_name] = (result_data.groupby('date')[factor_name]
                                                     .rank(method='average', pct=True, na_option='keep'))
                return result_data

            # 并行处理
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(process_factor_chunk, chunk, data) for chunk in factor_chunks]
                results = [future.result() for future in as_completed(futures)]

            # 合并结果
            final_data = data.copy()
            for result in results:
                rank_columns = [col for col in result.columns if col.endswith('_rank')]
                for col in rank_columns:
                    final_data[col] = result[col]

            return final_data

        except Exception as e:
            self.logger.error(f"并行横截面排序失败: {e}")
            # 回退到顺序处理
            return self._sequential_cross_sectional_ranking(data, factors)

    def _log_ranking_statistics(self, data: pd.DataFrame, ranked_factors: List[str]):
        """记录排序统计信息"""
        try:
            self.logger.info("📈 横截面排序统计信息:")

            for factor in ranked_factors[:5]:  # 只显示前5个因子的统计
                if factor in data.columns:
                    factor_data = data[factor].dropna()
                    if len(factor_data) > 0:
                        stats = {
                            'count': len(factor_data),
                            'min': factor_data.min(),
                            'max': factor_data.max(),
                            'mean': factor_data.mean()
                        }
                        self.logger.info(f"  {factor}: 数量={stats['count']}, "
                                       f"范围=[{stats['min']:.3f}, {stats['max']:.3f}], "
                                       f"均值={stats['mean']:.3f}")

        except Exception as e:
            self.logger.warning(f"记录排序统计信息失败: {e}")



    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据格式"""
        try:
            if not super().validate_input_data(data):
                return False
            
            # 检查必需的价格数据列
            required_price_columns = ['close', 'high', 'low']
            missing_columns = [col for col in required_price_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"价格相对位置因子需要价格数据列: {missing_columns}")
                return False
            
            # 检查数据有效性
            if data['close'].isna().all() or data['high'].isna().all() or data['low'].isna().all():
                self.logger.error("价格数据全部为空")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证价格相对位置因子输入数据失败: {e}")
            return False

    def get_feature_metadata(self) -> dict:
        """获取特征元数据（包含原始因子和排序后的因子）"""
        metadata = super().get_feature_metadata()

        # 添加原始特征描述
        feature_descriptions = {
            # 价格在区间内的位置特征
            'price_position_in_range_5d': '价格在近5日价格区间的相对位置 (0-1)',
            'price_position_in_range_10d': '价格在近10日价格区间的相对位置 (0-1)',
            'price_position_in_range_20d': '价格在近20日价格区间的相对位置 (0-1)',
            'price_position_in_range_60d': '价格在近60日价格区间的相对位置 (0-1)',

            # 价格相对均线比例特征
            'price_vs_ma5_ratio': '价格相对MA5的比例偏差',
            'price_vs_ma10_ratio': '价格相对MA10的比例偏差',
            'price_vs_ma20_ratio': '价格相对MA20的比例偏差',
            'price_vs_ma60_ratio': '价格相对MA60的比例偏差',

            # 高点相对均线比例特征
            'high_vs_ma5_ratio': '最高价相对MA5的比例偏差',
            'high_vs_ma10_ratio': '最高价相对MA10的比例偏差',
            'high_vs_ma20_ratio': '最高价相对MA20的比例偏差',
            'high_vs_ma60_ratio': '最高价相对MA60的比例偏差',

            # 低点相对均线比例特征
            'low_vs_ma5_ratio': '最低价相对MA5的比例偏差',
            'low_vs_ma10_ratio': '最低价相对MA10的比例偏差',
            'low_vs_ma20_ratio': '最低价相对MA20的比例偏差',
            'low_vs_ma60_ratio': '最低价相对MA60的比例偏差',

            # 高点在价格区间内的位置特征
            'high_position_in_range_5d': '最高价在近5日价格区间的相对位置 (0-1)',
            'high_position_in_range_10d': '最高价在近10日价格区间的相对位置 (0-1)',
            'high_position_in_range_20d': '最高价在近20日价格区间的相对位置 (0-1)',

            # 低点在价格区间内的位置特征
            'low_position_in_range_5d': '最低价在近5日价格区间的相对位置 (0-1)',
            'low_position_in_range_10d': '最低价在近10日价格区间的相对位置 (0-1)',
            'low_position_in_range_20d': '最低价在近20日价格区间的相对位置 (0-1)'
        }

        # 添加排序后的特征描述
        for original_feature, description in feature_descriptions.items():
            ranked_feature = f"{original_feature}_rank"
            feature_descriptions[ranked_feature] = f"{description} [横截面排序分位数]"

        # 更新元数据
        for feature, description in feature_descriptions.items():
            if feature in metadata:
                metadata[feature]['description'] = description

        return metadata

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




