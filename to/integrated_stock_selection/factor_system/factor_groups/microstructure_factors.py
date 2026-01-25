#!/usr/bin/env python3
"""
微观结构因子组
"""

import pandas as pd
import numpy as np
from typing import List
from .base_factor_group import BaseFactorGroup

class MicrostructureFactors(BaseFactorGroup):
    def __init__(self):
        super().__init__("microstructure_factors")
    
    def _get_version(self) -> str:
        return "v2.3"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 不依赖其他因子组，直接使用原始数据计算微观结构特征
    
    def _get_feature_names(self) -> List[str]:
        return [
            'bid_ask_spread', 'order_flow_imbalance', 'price_impact',
            'volume_weighted_price', 'tick_direction', 'trade_intensity'
        ]

    def _get_max_history_window_needed(self) -> int:
        """返回计算微观结构因子所需的最大历史窗口长度"""
        return 20  # 微观结构因子需要20日历史数据
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算微观结构因子（基于专业市场微观结构理论）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return data
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_microstructure_features_optimized(data, target_dates, historical_dates)

            # 4. 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            self.logger.info("微观结构因子计算完成（修复版）")
            return result_data

        except Exception as e:
            self.logger.error(f"微观结构因子计算失败: {e}")
            return data

    def _calculate_microstructure_features_optimized(self, full_data: pd.DataFrame,
                                                   target_dates: List[str],
                                                   historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化微观结构特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意微观结构数据的高频特性，确保时间窗口边界的正确性
        """
        try:
            self.logger.info("开始正确优化的微观结构特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_microstructure_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_microstructure_features_with_history(
                full_data, result_data, target_dates
            )

            self.logger.info("正确优化的微观结构特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的微观结构特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_microstructure_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化微观结构特征列（内存优化）"""
        try:
            # 微观结构特征
            result_data['bid_ask_spread'] = 0.01
            result_data['order_flow_imbalance'] = 0.0
            result_data['price_impact'] = 1.0
            result_data['volume_weighted_price'] = 0.0
            result_data['tick_direction'] = 0.0
            result_data['trade_intensity'] = 1.0

        except Exception as e:
            self.logger.error(f"初始化目标微观结构特征失败: {e}")

    def _calculate_target_microstructure_features_with_history(self, full_data: pd.DataFrame,
                                                             target_data: pd.DataFrame,
                                                             target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的微观结构特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的微观结构特征（使用历史数据支撑）...")

            # 转换目标日期为datetime格式
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]

            # 1. 计算简单微观结构特征（只为目标日期，不需要历史数据）
            target_data = self._calculate_target_simple_microstructure_features(target_data)

            # 2. 计算复杂微观结构特征（只为目标日期，使用历史数据支撑）
            target_data = self._calculate_target_complex_microstructure_features(
                full_data, target_data, target_date_objects
            )

            self.logger.info("目标日期微观结构特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期微观结构特征计算失败: {e}")
            return target_data

    def _calculate_target_simple_microstructure_features(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """为目标日期计算简单微观结构特征（不需要历史数据）"""
        try:
            # 1. bid_ask_spread: 买卖价差（基于日内价差）
            if 'high' in target_data.columns and 'low' in target_data.columns and 'close' in target_data.columns:
                close_safe = np.where(target_data['close'] > 1e-10, target_data['close'], 1.0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    target_data['bid_ask_spread'] = (target_data['high'] - target_data['low']) / close_safe
                target_data['bid_ask_spread'] = np.clip(target_data['bid_ask_spread'], 0, 0.2)
                self.logger.info("✅ bid_ask_spread计算完成（基于日内价差）")
            else:
                target_data['bid_ask_spread'] = 0.01
                self.logger.warning("⚠️ 缺少价格字段，bid_ask_spread使用默认值0.01")

            # 2. tick_direction: 价格变动方向
            if 'pct_chg' in target_data.columns:
                target_data['tick_direction'] = np.sign(target_data['pct_chg'])
                self.logger.info("✅ tick_direction计算完成（基于涨跌幅）")
            else:
                target_data['tick_direction'] = 0.0
                self.logger.warning("⚠️ 缺少pct_chg字段，tick_direction使用默认值0.0")

            # 3. volume_weighted_price: 成交量加权价格（简化版）
            if 'close' in target_data.columns and 'volume' in target_data.columns:
                # 使用收盘价作为VWAP的代理
                target_data['volume_weighted_price'] = target_data['close']
                self.logger.info("✅ volume_weighted_price计算完成（使用收盘价代理）")
            else:
                target_data['volume_weighted_price'] = 0.0
                self.logger.warning("⚠️ 缺少价格或成交量字段，volume_weighted_price使用默认值0.0")

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标简单微观结构特征失败: {e}")
            return target_data

    def _calculate_target_complex_microstructure_features(self, full_data: pd.DataFrame,
                                                        target_data: pd.DataFrame,
                                                        target_date_objects: List) -> pd.DataFrame:
        """为目标日期计算复杂微观结构特征（使用历史数据支撑）"""
        try:
            # 按股票分组处理
            target_stocks = target_data['stock_code'].unique()
            total_stocks = len(target_stocks)

            self.logger.info(f"准备处理 {total_stocks} 只股票的复杂微观结构特征")

            for idx, stock_code in enumerate(target_stocks):
                if idx % max(1, total_stocks // 10) == 0 or idx == total_stocks - 1:
                    progress_pct = (idx + 1) / total_stocks * 100
                    self.logger.info(f"处理进度: {idx + 1}/{total_stocks} ({progress_pct:.1f}%)")

                # 获取该股票的完整历史数据
                stock_full_data = full_data[full_data['stock_code'] == stock_code].sort_values('date')

                # 获取该股票的目标日期数据
                stock_target_mask = target_data['stock_code'] == stock_code

                if len(stock_full_data) < 20:  # 需要足够的历史数据计算微观结构特征
                    continue

                # 为该股票的目标日期计算复杂微观结构特征
                stock_target_features = self._calculate_stock_target_microstructure_features(
                    stock_full_data, target_date_objects
                )

                # 更新目标数据
                self._update_target_microstructure_features(target_data, stock_target_features, stock_target_mask)

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标复杂微观结构特征失败: {e}")
            return target_data

    def _calculate_stock_target_microstructure_features(self, stock_full_data: pd.DataFrame,
                                                      target_date_objects: List) -> dict:
        """为单只股票的目标日期计算复杂微观结构特征"""
        try:
            target_features = {}

            # 为每个目标日期计算特征
            for target_date_obj in target_date_objects:
                # 找到目标日期在数据中的位置
                date_mask = stock_full_data['date'] == target_date_obj
                if not date_mask.any():
                    continue

                target_idx = stock_full_data[date_mask].index[0]
                data_idx = stock_full_data.index.get_loc(target_idx)

                # 计算该日期的特征
                features = self._calculate_microstructure_features_for_single_date(
                    stock_full_data, data_idx
                )

                target_features[target_date_obj] = features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标微观结构特征失败: {e}")
            return {}

    def _calculate_microstructure_features_for_single_date(self, stock_data: pd.DataFrame,
                                                         data_idx: int) -> dict:
        """为单个目标日期计算所有复杂微观结构特征"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 20:
                return self._get_default_microstructure_features()

            # 获取必要的数据列
            volume = stock_data['volume'].values if 'volume' in stock_data.columns else None
            pct_chg = stock_data['pct_chg'].values if 'pct_chg' in stock_data.columns else None

            if volume is None or pct_chg is None:
                return self._get_default_microstructure_features()

            # 1. order_flow_imbalance: 订单流不平衡（使用历史数据支撑）
            features['order_flow_imbalance'] = self._calculate_order_flow_imbalance_for_date(
                volume, pct_chg, data_idx
            )

            # 2. price_impact: 价格冲击（使用历史数据支撑）
            features['price_impact'] = self._calculate_price_impact_for_date(
                volume, pct_chg, data_idx
            )

            # 3. trade_intensity: 交易强度（使用历史数据支撑）
            features['trade_intensity'] = self._calculate_trade_intensity_for_date(
                volume, data_idx
            )

            return features

        except Exception as e:
            self.logger.error(f"计算单日期微观结构特征失败: {e}")
            return self._get_default_microstructure_features()

    def _calculate_order_flow_imbalance_for_date(self, volume: np.ndarray,
                                               pct_chg: np.ndarray,
                                               data_idx: int) -> float:
        """为单个目标日期计算订单流不平衡"""
        try:
            # 获取20日窗口数据（确保不使用未来信息）
            window_size = 20
            start_idx = max(0, data_idx - window_size + 1)
            end_idx = data_idx + 1  # 包含目标日期

            volume_window = volume[start_idx:end_idx]
            pct_chg_window = pct_chg[start_idx:end_idx]

            if len(volume_window) < 5:
                return 0.0

            # 计算成交量标准化
            volume_mean = np.mean(volume_window[:-1])  # 不包含当日
            volume_std = np.std(volume_window[:-1])

            if volume_std > 0:
                volume_normalized = (volume[data_idx] - volume_mean) / volume_std
            else:
                volume_normalized = 0.0

            # 计算价格方向
            price_direction = np.sign(pct_chg[data_idx])

            # 订单流不平衡
            order_flow_imbalance = volume_normalized * price_direction
            return float(np.clip(order_flow_imbalance, -3, 3))

        except Exception as e:
            self.logger.error(f"计算订单流不平衡失败: {e}")
            return 0.0

    def _calculate_price_impact_for_date(self, volume: np.ndarray,
                                       pct_chg: np.ndarray,
                                       data_idx: int) -> float:
        """为单个目标日期计算价格冲击"""
        try:
            # 获取20日窗口数据（确保不使用未来信息）
            window_size = 20
            start_idx = max(0, data_idx - window_size + 1)
            end_idx = data_idx + 1  # 包含目标日期

            volume_window = volume[start_idx:end_idx]

            if len(volume_window) < 5:
                return 1.0

            # 计算成交量比率
            volume_mean = np.mean(volume_window[:-1])  # 不包含当日
            if volume_mean > 0:
                volume_ratio = volume[data_idx] / volume_mean
            else:
                volume_ratio = 1.0

            # 计算价格冲击
            price_change = abs(pct_chg[data_idx])
            price_impact = price_change / (volume_ratio + 0.1)

            return float(np.clip(price_impact, 0, 10))

        except Exception as e:
            self.logger.error(f"计算价格冲击失败: {e}")
            return 1.0

    def _calculate_trade_intensity_for_date(self, volume: np.ndarray, data_idx: int) -> float:
        """为单个目标日期计算交易强度"""
        try:
            # 获取20日窗口数据（确保不使用未来信息）
            window_size = 20
            start_idx = max(0, data_idx - window_size + 1)
            end_idx = data_idx + 1  # 包含目标日期

            volume_window = volume[start_idx:end_idx]

            if len(volume_window) < 5:
                return 1.0

            # 计算交易强度（当日成交量相对于历史平均的倍数）
            volume_mean = np.mean(volume_window[:-1])  # 不包含当日
            if volume_mean > 0:
                trade_intensity = volume[data_idx] / volume_mean
            else:
                trade_intensity = 1.0

            return float(np.clip(trade_intensity, 0.1, 10))

        except Exception as e:
            self.logger.error(f"计算交易强度失败: {e}")
            return 1.0

    def _get_default_microstructure_features(self) -> dict:
        """获取默认的微观结构特征值"""
        return {
            'order_flow_imbalance': 0.0,
            'price_impact': 1.0,
            'trade_intensity': 1.0
        }

    def _update_target_microstructure_features(self, target_data: pd.DataFrame,
                                             stock_target_features: dict,
                                             stock_target_mask: pd.Series):
        """更新目标数据中的微观结构特征"""
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
            self.logger.error(f"更新目标微观结构特征失败: {e}")

    def _calculate_microstructure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算市场微观结构指标"""
        try:
            # 1. bid_ask_spread: 买卖价差（简化实现）
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
                # 使用日内价差作为买卖价差的代理，避免除零警告
                close_safe = np.where(data['close'] > 1e-10, data['close'], 1.0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    data['bid_ask_spread'] = (data['high'] - data['low']) / close_safe
                data['bid_ask_spread'] = np.clip(data['bid_ask_spread'], 0, 0.2)
            else:
                data['bid_ask_spread'] = 0.01

            # 2. order_flow_imbalance: 订单流不平衡（内存优化版）
            if 'volume' in data.columns and 'pct_chg' in data.columns:
                # 分批计算以避免内存溢出
                data['order_flow_imbalance'] = 0.0
                unique_stocks = data['stock_code'].unique()
                batch_size = 500  # 每批处理500只股票

                for i in range(0, len(unique_stocks), batch_size):
                    batch_stocks = unique_stocks[i:i + batch_size]
                    batch_mask = data['stock_code'].isin(batch_stocks)
                    batch_data = data[batch_mask].copy()

                    # 计算该批次的订单流不平衡
                    volume_normalized = batch_data.groupby('stock_code')['volume'].transform(
                        lambda x: (x - x.rolling(20, min_periods=5).mean()) / x.rolling(20, min_periods=5).std().fillna(1)
                    )
                    price_direction = np.sign(batch_data['pct_chg'])
                    batch_result = volume_normalized * price_direction
                    batch_result = np.clip(batch_result, -3, 3)

                    # 更新结果
                    data.loc[batch_mask, 'order_flow_imbalance'] = batch_result

                    # 清理内存
                    del batch_data, volume_normalized, batch_result
            else:
                data['order_flow_imbalance'] = 0.0

            # 3. price_impact: 价格冲击（内存优化版）
            if 'volume' in data.columns and 'pct_chg' in data.columns:
                # 分批计算以避免内存溢出
                data['price_impact'] = 1.0
                unique_stocks = data['stock_code'].unique()
                batch_size = 500  # 每批处理500只股票

                for i in range(0, len(unique_stocks), batch_size):
                    batch_stocks = unique_stocks[i:i + batch_size]
                    batch_mask = data['stock_code'].isin(batch_stocks)
                    batch_data = data[batch_mask].copy()

                    # 计算该批次的价格冲击，避免除零警告
                    volume_ratio = batch_data.groupby('stock_code')['volume'].transform(
                        lambda x: x / (x.rolling(20, min_periods=5).mean() + 1e-10)
                    )
                    with np.errstate(divide='ignore', invalid='ignore'):
                        batch_result = np.abs(batch_data['pct_chg']) / (volume_ratio + 0.1)
                    batch_result = np.clip(batch_result, 0, 10)

                    # 更新结果
                    data.loc[batch_mask, 'price_impact'] = batch_result

                    # 清理内存
                    del batch_data, volume_ratio, batch_result
            else:
                data['price_impact'] = 1.0

            # 4. volume_weighted_price: 成交量加权价格偏离
            if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns and 'volume' in data.columns:
                # 简化的VWAP计算，避免除零警告
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                typical_price_safe = np.where(typical_price > 1e-10, typical_price, data['close'])
                with np.errstate(divide='ignore', invalid='ignore'):
                    data['volume_weighted_price'] = (data['close'] - typical_price) / typical_price_safe
                data['volume_weighted_price'] = np.clip(data['volume_weighted_price'], -0.1, 0.1)
            else:
                data['volume_weighted_price'] = 0.0

            # 5. tick_direction: 价格变动方向
            if 'close' in data.columns:
                data['tick_direction'] = data.groupby('stock_code')['close'].transform(
                    lambda x: np.sign(x.diff())
                ).fillna(0)
            else:
                data['tick_direction'] = 0.0

            # 6. trade_intensity: 交易强度（内存优化版）
            if 'volume' in data.columns and 'turnover_rate' in data.columns:
                # 分批计算以避免内存溢出
                data['trade_intensity'] = 0.0
                unique_stocks = data['stock_code'].unique()
                batch_size = 500  # 每批处理500只股票

                for i in range(0, len(unique_stocks), batch_size):
                    batch_stocks = unique_stocks[i:i + batch_size]
                    batch_mask = data['stock_code'].isin(batch_stocks)
                    batch_data = data[batch_mask].copy()

                    # 计算该批次的交易强度
                    volume_zscore = batch_data.groupby('stock_code')['volume'].transform(
                        lambda x: (x - x.rolling(20, min_periods=5).mean()) / x.rolling(20, min_periods=5).std().fillna(1)
                    )
                    turnover_zscore = batch_data.groupby('stock_code')['turnover_rate'].transform(
                        lambda x: (x - x.rolling(20, min_periods=5).mean()) / x.rolling(20, min_periods=5).std().fillna(1)
                    )
                    batch_result = (volume_zscore + turnover_zscore) / 2
                    batch_result = np.clip(batch_result, -3, 3)

                    # 更新结果
                    data.loc[batch_mask, 'trade_intensity'] = batch_result

                    # 清理内存
                    del batch_data, volume_zscore, turnover_zscore, batch_result
            else:
                data['trade_intensity'] = 0.0

            return data

        except Exception as e:
            self.logger.error(f"微观结构指标计算失败: {e}")
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
