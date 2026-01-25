#!/usr/bin/env python3
"""
涨停质量因子组

直接加载原始股票数据中的涨停相关特征
不再进行复杂计算，而是直接提取已有的13个涨停质量特征字段
"""

import pandas as pd
import numpy as np
from typing import List
from .base_factor_group import BaseFactorGroup

class LimitUpQualityFactors(BaseFactorGroup):
    """涨停质量因子组 - 直接加载原始特征版本"""

    def __init__(self):
        super().__init__("limit_up_quality_factors")

    def _get_version(self) -> str:
        return "v2.0"  # 版本升级，表示改为直接加载模式

    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接从原始数据加载

    def _get_max_history_window_needed(self) -> int:
        """返回所需的最大历史窗口天数"""
        return 1  # 直接加载模式不需要历史窗口

    def _get_feature_names(self) -> List[str]:
        """返回需要从原始数据中加载的13个涨停质量特征"""
        return [
            'is_limit_up',
            'is_limit_down',
            'is_zhaban',
            'limit_strength',
            'limit_up_days_5d',
            'limit_down_days_5d',
            'limit_up_days_20d',
            'consecutive_limit_up',
            'consecutive_limit_down',
            'limit_up_volume_ratio',
            'limit_up_turnover_ratio',
            'amihud_illiquidity',
            'near_support',
            'short_reversal'
        ]
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """直接加载原始股票数据中的涨停质量特征"""
        try:
            self.logger.info("开始加载涨停质量特征（直接加载模式）")

            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return self._set_default_values(data.copy())

            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            result_data = data.copy()

            # 直接加载原始特征
            result_data = self._load_original_features(result_data)

            # 过滤结果到目标日期
            result_data = self._filter_results_to_target_dates(result_data, target_dates)

            # 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            return result_data

        except Exception as e:
            self.logger.error(f"涨停质量因子加载失败: {e}")
            return self._set_default_values(data.copy())

    def _load_original_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        直接从原始股票数据中加载涨停质量特征
        不进行复杂计算，只是提取已有的特征字段
        """
        try:
            self.logger.info("开始加载原始涨停质量特征...")

            result_data = data.copy()

            # 获取需要加载的特征列表
            required_features = self._get_feature_names()

            # 检查哪些特征在原始数据中存在
            available_features = []
            missing_features = []

            for feature in required_features:
                if feature in data.columns:
                    available_features.append(feature)
                    self.logger.debug(f"✅ 特征 {feature} 在原始数据中存在")
                else:
                    missing_features.append(feature)
                    self.logger.warning(f"⚠️ 特征 {feature} 在原始数据中缺失，将设置为默认值")

            # 记录加载统计
            self.logger.info(f"特征加载统计:")
            self.logger.info(f"  可用特征: {len(available_features)}/{len(required_features)}")
            self.logger.info(f"  缺失特征: {len(missing_features)}")

            if available_features:
                self.logger.info(f"  可用特征列表: {available_features}")

            if missing_features:
                self.logger.warning(f"  缺失特征列表: {missing_features}")

                # 为缺失的特征设置默认值
                for feature in missing_features:
                    default_value = self._get_default_value_for_feature(feature)
                    result_data[feature] = default_value
                    self.logger.debug(f"为缺失特征 {feature} 设置默认值: {default_value}")

            # 验证数据质量
            self._validate_feature_data_quality(result_data, required_features)

            self.logger.info("✅ 涨停质量特征加载完成")
            return result_data

        except Exception as e:
            self.logger.error(f"加载原始涨停质量特征失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return self._set_default_values(data.copy())

    def _get_default_value_for_feature(self, feature_name: str) -> float:
        """为不同的特征返回合适的默认值"""
        # 布尔型特征的默认值
        boolean_features = ['is_limit_up', 'is_limit_down', 'is_zhaban', 'near_support']
        if feature_name in boolean_features:
            return 0

        # 比例型特征的默认值
        ratio_features = ['limit_strength', 'limit_up_volume_ratio', 'limit_up_turnover_ratio']
        if feature_name in ratio_features:
            return 1.0

        # 计数型特征的默认值
        count_features = ['limit_up_days_5d', 'limit_down_days_5d', 'limit_up_days_20d',
                         'consecutive_limit_up', 'consecutive_limit_down']
        if feature_name in count_features:
            return 0

        # 流动性特征的默认值
        if feature_name == ['amihud_illiquidity','short_reversal']:
            return 0.0

        # 其他特征的默认值
        return 0.0

    def _validate_feature_data_quality(self, data: pd.DataFrame, feature_names: List[str]) -> None:
        """验证特征数据质量"""
        try:
            for feature in feature_names:
                if feature in data.columns:
                    # 检查NaN值
                    nan_count = data[feature].isna().sum()
                    if nan_count > 0:
                        self.logger.warning(f"特征 {feature} 包含 {nan_count} 个NaN值")

                    # 检查无穷值
                    inf_count = np.isinf(data[feature]).sum()
                    if inf_count > 0:
                        self.logger.warning(f"特征 {feature} 包含 {inf_count} 个无穷值")
                        # 将无穷值替换为默认值
                        default_val = self._get_default_value_for_feature(feature)
                        data[feature] = data[feature].replace([np.inf, -np.inf], default_val)

                    # 记录特征值范围
                    if data[feature].dtype in ['int64', 'float64']:
                        min_val = data[feature].min()
                        max_val = data[feature].max()
                        self.logger.debug(f"特征 {feature} 值范围: [{min_val:.4f}, {max_val:.4f}]")

        except Exception as e:
            self.logger.error(f"特征数据质量验证失败: {e}")

    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> tuple:
        """精确识别目标计算日期和历史支撑数据"""
        try:
            target_start_date = kwargs.get('target_start_date')
            target_end_date = kwargs.get('target_end_date')
            if target_start_date and target_end_date:
                target_dates = self._get_trading_dates_in_range(data, target_start_date, target_end_date)
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
            max_history_needed = self._get_max_history_window_needed()
            historical_dates = self._get_optimal_historical_dates(data, target_dates, max_history_needed)
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

    def _get_optimal_historical_dates(self, data: pd.DataFrame, target_dates: List[str], max_history_needed: int) -> List[str]:
        """获取最优的历史支撑数据日期范围"""
        try:
            if not target_dates:
                return []

            all_dates = sorted(data['date'].dt.strftime('%Y%m%d').unique())
            earliest_target_index = all_dates.index(min(target_dates))
            history_start_index = max(0, earliest_target_index - max_history_needed)
            historical_dates = all_dates[history_start_index:earliest_target_index]

            self.logger.debug(f"历史数据范围: {len(historical_dates)} 个交易日")
            return historical_dates
        except Exception as e:
            self.logger.error(f"获取历史日期失败: {e}")
            return []





    def _filter_results_to_target_dates(self, data: pd.DataFrame, target_dates: List[str]) -> pd.DataFrame:
        """只保留目标日期的计算结果"""
        try:
            if not target_dates:
                return data
            date_mask = data['date'].dt.strftime('%Y%m%d').isin(target_dates)
            filtered_data = data[date_mask].copy()
            self.logger.info(f"过滤结果: {len(data)} -> {len(filtered_data)} 条记录")
            return filtered_data
        except Exception as e:
            self.logger.error(f"过滤结果到目标日期失败: {e}")
            return data

    def _set_default_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """为所有特征设置默认值"""
        try:
            result_data = data.copy()
            feature_names = self._get_feature_names()

            for feature_name in feature_names:
                default_value = self._get_default_value_for_feature(feature_name)
                result_data[feature_name] = default_value
                self.logger.debug(f"为特征 {feature_name} 设置默认值: {default_value}")

            self.logger.info(f"已为 {len(feature_names)} 个特征设置默认值")
            return result_data

        except Exception as e:
            self.logger.error(f"设置默认值失败: {e}")
            return data
