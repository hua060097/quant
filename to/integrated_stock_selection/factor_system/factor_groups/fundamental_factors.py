#!/usr/bin/env python3
"""
基本面因子组

包含基本面相关因子的计算
"""

import pandas as pd
import numpy as np
from typing import List
from .base_factor_group import BaseFactorGroup

class FundamentalFactors(BaseFactorGroup):
    """基本面因子组"""
    
    def __init__(self):
        super().__init__("fundamental_factors")
    
    def _get_version(self) -> str:
        return "v1.5"
    
    def _get_dependencies(self) -> List[str]:
        return []
    
    def _get_feature_names(self) -> List[str]:
        return [
            # 基本面衍生特征
            'float_ratio', 'mv_factor',
            # 市值相关特征
            'log_market_cap', 'market_cap_percentile', 'market_cap_zscore',
            'size_factor_score', 'liquidity_adjusted_mv', 'market_cap_momentum',
            'sector_relative_mv'
        ]
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算基本面因子（基于feature_calculator.py的专业实现）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return self._set_default_values(data)
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_fundamental_features_optimized(data, target_dates, historical_dates)

            # 4. 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            self.logger.info("基本面因子计算完成（修复版）")
            return result_data

        except Exception as e:
            self.logger.error(f"基本面因子计算失败: {e}")
            return self._set_default_values(data)

    def _calculate_fundamental_features_optimized(self, full_data: pd.DataFrame,
                                                target_dates: List[str],
                                                historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化基本面特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意基本面数据的时间滞后性，确保使用正确的财务数据时点
        """
        try:
            self.logger.info("开始正确优化的基本面特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_fundamental_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_fundamental_features_with_history(
                full_data, result_data, target_dates
            )

            self.logger.info("正确优化的基本面特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的基本面特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_fundamental_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化基本面特征列（内存优化）"""
        try:
            # 基本面衍生特征
            result_data['float_ratio'] = 1.0
            result_data['mv_factor'] = 1.0

            # 市值相关特征
            result_data['log_market_cap'] = 0.0
            result_data['market_cap_percentile'] = 0.5
            result_data['market_cap_zscore'] = 0.0
            result_data['size_factor_score'] = 50.0
            result_data['liquidity_adjusted_mv'] = 0.0
            result_data['market_cap_momentum'] = 0.0
            result_data['sector_relative_mv'] = 0.0

        except Exception as e:
            self.logger.error(f"初始化目标基本面特征失败: {e}")

    def _calculate_target_fundamental_features_with_history(self, full_data: pd.DataFrame,
                                                          target_data: pd.DataFrame,
                                                          target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的基本面特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的基本面特征（使用历史数据支撑）...")

            # 转换目标日期为datetime格式
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]

            # 1. 计算基本面衍生特征（只为目标日期）
            target_data = self._calculate_target_fundamental_derived_features(target_data)

            # 2. 计算市值相关特征（只为目标日期，使用历史数据支撑）
            target_data = self._calculate_target_market_cap_features(full_data, target_data, target_date_objects)

            self.logger.info("目标日期基本面特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期基本面特征计算失败: {e}")
            return target_data

    def _calculate_target_fundamental_derived_features(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """为目标日期计算基本面衍生特征（不需要历史数据）"""
        try:
            # 1. float_ratio: 流通比例
            if 'circ_mv' in target_data.columns and 'total_mv' in target_data.columns:
                total_mv_valid = (target_data['total_mv'] > 0).any()
                circ_mv_valid = (target_data['circ_mv'] > 0).any()

                if total_mv_valid and circ_mv_valid:
                    target_data['float_ratio'] = np.where(
                        target_data['total_mv'] > 0,
                        target_data['circ_mv'] / target_data['total_mv'],
                        1.0
                    )
                    self.logger.info("✅ float_ratio计算完成（基于circ_mv/total_mv）")
                else:
                    target_data['float_ratio'] = 1.0
                    self.logger.warning("⚠️ 市值数据无效，float_ratio使用默认值1.0")
            else:
                target_data['float_ratio'] = 1.0
                self.logger.warning("⚠️ 缺少市值字段，float_ratio使用默认值1.0")

            # 2. mv_factor: 市值因子（简化版）
            if 'total_mv' in target_data.columns:
                # 使用对数变换减少极值影响
                target_data['mv_factor'] = np.log(target_data['total_mv'].fillna(1) + 1)
                # 标准化到合理范围
                mv_factor_mean = target_data['mv_factor'].mean()
                if mv_factor_mean > 0:
                    target_data['mv_factor'] = target_data['mv_factor'] / mv_factor_mean
                self.logger.info("✅ mv_factor计算完成（基于total_mv对数变换）")
            else:
                target_data['mv_factor'] = 1.0
                self.logger.warning("⚠️ 缺少total_mv字段，mv_factor使用默认值1.0")

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标基本面衍生特征失败: {e}")
            return target_data

    def _calculate_target_market_cap_features(self, full_data: pd.DataFrame,
                                            target_data: pd.DataFrame,
                                            target_date_objects: List) -> pd.DataFrame:
        """为目标日期计算市值相关特征（使用历史数据支撑）"""
        try:
            # 确定市值列
            mv_column = self._determine_market_cap_column(full_data)
            if mv_column is None:
                self.logger.warning("无法确定有效的市值列，使用默认值")
                return target_data

            # 为每个目标日期逐一计算市值特征
            for target_date_obj in target_date_objects:
                # 获取该目标日期的数据
                target_day_mask = target_data['date'] == target_date_obj
                target_day_data = target_data[target_day_mask]

                if len(target_day_data) == 0:
                    continue

                # 为该目标日期计算市值特征
                target_day_features = self._calculate_market_cap_features_for_single_date(
                    full_data, target_day_data, target_date_obj, mv_column
                )

                # 更新目标数据
                self._update_target_market_cap_features(target_data, target_day_features, target_day_mask)

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标市值特征失败: {e}")
            return target_data

    def _determine_market_cap_column(self, data: pd.DataFrame) -> str:
        """确定有效的市值列"""
        try:
            mv_column = None

            # 优先检查total_mv
            if 'total_mv' in data.columns:
                total_mv_positive = (data['total_mv'] > 0).sum()
                if total_mv_positive > 0:
                    mv_column = 'total_mv'
                    self.logger.info(f"使用total_mv作为市值指标，有效数据: {total_mv_positive}")
                    return mv_column

            # 如果total_mv无效，检查circ_mv
            if 'circ_mv' in data.columns:
                circ_mv_positive = (data['circ_mv'] > 0).sum()
                if circ_mv_positive > 0:
                    mv_column = 'circ_mv'
                    self.logger.info(f"使用circ_mv作为市值指标，有效数据: {circ_mv_positive}")
                    return mv_column

            # 临时方案：使用收盘价作为市值代理
            if 'close' in data.columns:
                self.logger.warning("⚠️ 市值数据无效，使用收盘价作为市值代理")
                return 'close'

            return None

        except Exception as e:
            self.logger.error(f"确定市值列失败: {e}")
            return None

    def _calculate_market_cap_features_for_single_date(self, full_data: pd.DataFrame,
                                                     target_day_data: pd.DataFrame,
                                                     target_date_obj: pd.Timestamp,
                                                     mv_column: str) -> dict:
        """为单个目标日期计算市值特征"""
        try:
            features = {}

            # 获取该日期的所有股票数据（用于横截面排名）
            date_mask = full_data['date'] == target_date_obj
            all_stocks_data = full_data[date_mask]

            if len(all_stocks_data) < 5:
                return self._get_default_market_cap_features()

            # 1. 对数市值
            mv_values = all_stocks_data[mv_column].fillna(1)
            log_mv = np.log(mv_values + 1)
            features['log_market_cap'] = pd.Series(log_mv.values, index=all_stocks_data.index)

            # 2. 市值百分位排名（横截面）
            mv_rank = mv_values.rank(pct=True, method='min')
            features['market_cap_percentile'] = pd.Series(mv_rank.values, index=all_stocks_data.index)

            # 3. 市值Z-score标准化（需要历史数据）
            zscore_values = []
            for idx, row in all_stocks_data.iterrows():
                stock_code = row['stock_code']
                zscore = self._calculate_market_cap_zscore_for_stock(
                    full_data, stock_code, target_date_obj, mv_column
                )
                zscore_values.append(zscore)

            features['market_cap_zscore'] = pd.Series(zscore_values, index=all_stocks_data.index)

            # 4. 规模因子评分
            size_scores = mv_rank.apply(lambda x: 80 if x < 0.3 else (60 if x < 0.7 else 40))
            features['size_factor_score'] = pd.Series(size_scores.values, index=all_stocks_data.index)

            # 5. 其他特征设为默认值
            features['liquidity_adjusted_mv'] = pd.Series(0.0, index=all_stocks_data.index)
            features['market_cap_momentum'] = pd.Series(0.0, index=all_stocks_data.index)
            features['sector_relative_mv'] = pd.Series(0.0, index=all_stocks_data.index)

            return features

        except Exception as e:
            self.logger.error(f"计算单日期市值特征失败: {e}")
            return self._get_default_market_cap_features()

    def _calculate_market_cap_zscore_for_stock(self, full_data: pd.DataFrame,
                                             stock_code: str,
                                             target_date_obj: pd.Timestamp,
                                             mv_column: str) -> float:
        """为单只股票计算市值Z-score（使用历史数据）"""
        try:
            # 获取该股票的历史数据
            stock_mask = (full_data['stock_code'] == stock_code) & (full_data['date'] <= target_date_obj)
            stock_data = full_data[stock_mask].sort_values('date')

            if len(stock_data) < 10:
                return 0.0

            # 获取最近60个交易日的市值数据
            recent_data = stock_data.tail(60)
            mv_series = recent_data[mv_column]

            if len(mv_series) < 10:
                return 0.0

            # 计算Z-score
            mv_mean = mv_series.mean()
            mv_std = mv_series.std()

            if mv_std > 0:
                current_mv = mv_series.iloc[-1]  # 目标日期的市值
                zscore = (current_mv - mv_mean) / mv_std
                return zscore

            return 0.0

        except Exception as e:
            self.logger.error(f"计算股票市值Z-score失败: {e}")
            return 0.0

    def _get_default_market_cap_features(self) -> dict:
        """获取默认的市值特征值"""
        return {
            'log_market_cap': 0.0,
            'market_cap_percentile': 0.5,
            'market_cap_zscore': 0.0,
            'size_factor_score': 50.0,
            'liquidity_adjusted_mv': 0.0,
            'market_cap_momentum': 0.0,
            'sector_relative_mv': 0.0
        }

    def _update_target_market_cap_features(self, target_data: pd.DataFrame,
                                         target_day_features: dict,
                                         target_day_mask: pd.Series):
        """更新目标数据中的市值特征"""
        try:
            for feature_name, feature_series in target_day_features.items():
                if feature_name in target_data.columns and isinstance(feature_series, pd.Series):
                    # 只更新目标日期的数据
                    for idx in target_data[target_day_mask].index:
                        if idx in feature_series.index:
                            target_data.loc[idx, feature_name] = feature_series[idx]

        except Exception as e:
            self.logger.error(f"更新目标市值特征失败: {e}")

    def _calculate_fundamental_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算基本面衍生特征（从feature_calculator.py迁移）"""
        try:
            # 1. float_ratio: 流通比例
            # 🔧 检查市值数据是否有效
            if 'circ_mv' in data.columns and 'total_mv' in data.columns:
                total_mv_valid = (data['total_mv'] > 0).any()
                circ_mv_valid = (data['circ_mv'] > 0).any()

                if total_mv_valid and circ_mv_valid:
                    data['float_ratio'] = np.where(
                        data['total_mv'] > 0,
                        data['circ_mv'] / data['total_mv'],
                        0.7  # 默认流通比例
                    )
                    data['float_ratio'] = np.clip(data['float_ratio'], 0, 1)
                    self.logger.info("使用有效市值数据计算流通比例")
                else:
                    data['float_ratio'] = 0.7
                    self.logger.warning("⚠️ 市值数据无效，使用默认流通比例0.7")
            else:
                data['float_ratio'] = 0.7
                self.logger.info("缺少市值字段，使用默认流通比例0.7")

            # 2. mv_factor: 市值因子
            if 'total_mv' in data.columns and (data['total_mv'] > 0).any():
                # 对数市值的标准化，用于质量评分
                log_mv = np.log(data['total_mv'] + 1)
                mv_mean = log_mv.mean()
                mv_std = log_mv.std()
                data['mv_factor'] = np.where(
                    mv_std > 0,
                    (log_mv - mv_mean) / mv_std,
                    0
                )
                # 标准化到0-1区间，用于质量评分
                data['mv_factor'] = (data['mv_factor'] + 3) / 6  # 假设±3σ范围
                data['mv_factor'] = np.clip(data['mv_factor'], 0, 1)
                self.logger.info("使用有效total_mv计算市值因子")
            elif 'close' in data.columns:
                # 🔧 临时方案：使用收盘价计算市值因子
                log_price = np.log(data['close'] + 1)
                price_mean = log_price.mean()
                price_std = log_price.std()
                data['mv_factor'] = np.where(
                    price_std > 0,
                    (log_price - price_mean) / price_std,
                    0
                )
                # 标准化到0-1区间
                data['mv_factor'] = (data['mv_factor'] + 3) / 6
                data['mv_factor'] = np.clip(data['mv_factor'], 0, 1)
                self.logger.warning("⚠️ 使用收盘价作为市值因子的替代计算")
            else:
                data['mv_factor'] = 0.5  # 中性值
                self.logger.warning("⚠️ 无可用数据，使用默认市值因子0.5")

            return data

        except Exception as e:
            self.logger.error(f"基本面衍生特征计算失败: {e}")
            return self._set_default_values(data)

    def _calculate_market_cap_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市值相关特征 - 基于专业量化研究（从feature_calculator.py迁移）

        参考文献：
        - Fama & French (1992) "The Cross-Section of Expected Stock Returns"
        - Banz (1981) "The relationship between return and market value"
        - 专业量化机构实践：使用相对特征而非绝对市值
        """
        try:
            # 检查是否有市值数据
            if 'total_mv' not in data.columns and 'circ_mv' not in data.columns:
                self.logger.warning("数据中缺少市值字段，跳过市值特征计算")
                return self._set_default_values(data)

            # 🔧 临时修复：检查市值数据质量并使用替代方案
            mv_column = None

            # 优先检查total_mv
            if 'total_mv' in data.columns:
                total_mv_positive = (data['total_mv'] > 0).sum()
                if total_mv_positive > 0:
                    mv_column = 'total_mv'
                    self.logger.info(f"使用total_mv作为市值指标，有效数据: {total_mv_positive}")

            # 如果total_mv无效，检查circ_mv
            if mv_column is None and 'circ_mv' in data.columns:
                circ_mv_positive = (data['circ_mv'] > 0).sum()
                if circ_mv_positive > 0:
                    mv_column = 'circ_mv'
                    self.logger.info(f"使用circ_mv作为市值指标，有效数据: {circ_mv_positive}")

            # 🔧 临时方案：如果市值数据全为0，使用股价×流通股本作为替代
            if mv_column is None:
                self.logger.warning("⚠️ 市值数据全为0，尝试使用替代方案计算市值")

                # 方案1：使用close价格作为市值代理（临时方案）
                if 'close' in data.columns:
                    # 使用收盘价的相对排名作为市值代理
                    data['_temp_mv'] = data['close'].fillna(data['close'].median())
                    mv_column = '_temp_mv'
                    self.logger.info("使用收盘价作为市值代理指标")
                else:
                    self.logger.warning("没有可用的市值替代数据，跳过市值特征计算")
                    return self._set_default_values(data)

            # 过滤有效市值数据
            valid_mv_mask = (data[mv_column].notna()) & (data[mv_column] > 0)

            if not valid_mv_mask.any():
                self.logger.warning("没有有效的市值数据，跳过市值特征计算")
                return self._set_default_values(data)

            # 1. 对数市值（已存在，确保计算正确）
            if 'log_market_cap' not in data.columns:
                data['log_market_cap'] = np.log(data[mv_column].fillna(1))

            # 2. 市值百分位排名（横截面）
            data['market_cap_percentile'] = data.groupby('date')[mv_column].rank(pct=True)

            # 3. 市值Z-score标准化（时间序列）
            data['market_cap_zscore'] = data.groupby('stock_code')[mv_column].transform(
                lambda x: (x - x.rolling(60, min_periods=10).mean()) /
                         x.rolling(60, min_periods=10).std().fillna(1)
            ).fillna(0)

            # 4. 规模因子评分（基于Fama-French研究）
            data['size_factor_score'] = data['market_cap_percentile'].apply(
                lambda x: 80 if x < 0.3 else (60 if x < 0.7 else 40)  # 小盘股溢价
            )

            # 5. 流动性调整市值（市值 × 换手率）
            if 'turnover_rate' in data.columns:
                data['liquidity_adjusted_mv'] = (
                    data[mv_column] * data['turnover_rate'].fillna(data['turnover_rate'].median())
                )
            else:
                data['liquidity_adjusted_mv'] = data[mv_column]

            # 6. 市值动量（相对变化）
            data['market_cap_momentum'] = data.groupby('stock_code')[mv_column].pct_change(20).fillna(0)

            # 7. 板块内相对市值排名
            if 'sector_code' in data.columns:
                data['sector_relative_mv'] = data.groupby(['date', 'sector_code'])[mv_column].rank(pct=True)
            else:
                data['sector_relative_mv'] = data['market_cap_percentile']  # 使用全市场排名作为替代

            self.logger.debug("市值相对特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"市值特征计算失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
