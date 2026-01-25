#!/usr/bin/env python3
"""
技术因子组

包含技术指标衍生特征的计算
保留原有feature_calculator.py中的所有技术指标计算逻辑
"""

import pandas as pd
import numpy as np
from typing import List
from .base_factor_group import BaseFactorGroup

class TechnicalFactors(BaseFactorGroup):
    """技术因子组"""
    
    def __init__(self):
        super().__init__("technical_factors")
    
    def _get_version(self) -> str:
        return "v2.1"
    
    def _get_dependencies(self) -> List[str]:
        return []  # 技术因子组不依赖其他因子组
    
    def _get_feature_names(self) -> List[str]:
        return [
            'trend_consistency',
            'momentum_acceleration',
            'bb_position',
            'rsi_divergence',
            'macd_signal_strength',
            'price_channel_position',
            'volatility_regime',
            'volume_price_divergence',
            'support_resistance_strength',
            'breakout_momentum',
            # 新增的专业支撑阻力特征
            'distance_to_poc_pct',
            'distance_to_nearest_support_atr',
            'support_strength_score',
            'last_reaction_at_poc'
        ]

    def _get_max_history_window_needed(self) -> int:
        """返回计算技术因子所需的最大历史窗口长度"""
        return 60  # 技术因子需要60日历史数据
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算技术指标衍生特征（修复版：支持目标日期和历史数据优化）"""
        try:
            
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return data
            
            self.logger.info(f"目标计算日期: {target_dates}")
            self.logger.info(f"历史支撑数据: {len(historical_dates)} 个交易日")

            # 🚀 正确的优化计算逻辑：历史数据只作支撑，只为目标日期计算指标
            result_data = self._calculate_technical_features_optimized(data, target_dates, historical_dates)

            # 12. 保存目标日期信息供缓存系统使用
            result_data.attrs['target_dates'] = target_dates

            self.logger.info("技术指标衍生特征计算完成（修复版）")
            return result_data
            
        except Exception as e:
            self.logger.error(f"技术指标衍生特征计算失败（修复版）: {e}")
            return data

    def _calculate_technical_features_optimized(self, full_data: pd.DataFrame,
                                              target_dates: List[str],
                                              historical_dates: List[str]) -> pd.DataFrame:
        """
        正确的优化技术特征计算（参照KlinePatternFactors模式）

        核心原则：
        1. 历史数据只作支撑，不计算指标
        2. 只为目标日期计算衍生指标
        3. 确保需要历史数据的指标能正确访问足够的历史数据
        4. 避免为历史日期计算不必要的指标值
        5. 特别注意技术指标计算中的时间窗口边界，确保不使用未来信息
        """
        try:
            self.logger.info("开始正确优化的技术特征计算")

            # 1. 创建目标日期的结果DataFrame（参照KlinePatternFactors）
            target_date_objects = [pd.to_datetime(date, format='%Y%m%d') for date in target_dates]
            target_mask = full_data['date'].isin(target_date_objects)
            result_data = full_data[target_mask].copy()

            self.logger.info(f"目标日期数据量: {len(result_data)} 行，原始数据量: {len(full_data)} 行")
            self.logger.info(f"计算效率提升: {(1 - len(result_data)/len(full_data))*100:.1f}% 减少计算量")
            self.logger.info(f"历史数据: {len(full_data) - len(result_data)} 条记录（仅作数据支撑）")

            # 2. 初始化目标日期的特征列（只为目标日期分配内存）
            self._initialize_target_technical_features(result_data)

            # 3. 使用完整数据计算指标，但只保留目标日期的结果
            result_data = self._calculate_target_technical_features_with_history(
                full_data, result_data, target_dates
            )

            self.logger.info("正确优化的技术特征计算完成")
            return result_data

        except Exception as e:
            self.logger.error(f"正确优化的技术特征计算失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return full_data

    def _initialize_target_technical_features(self, result_data: pd.DataFrame):
        """只为目标日期初始化技术特征列（内存优化）"""
        try:
            # 技术指标特征
            result_data['trend_consistency'] = 0.5
            result_data['momentum_acceleration'] = 0.0
            result_data['bb_position'] = 0.5
            result_data['rsi_divergence'] = 0.0
            result_data['macd_signal_strength'] = 0.0
            result_data['price_channel_position'] = 0.5
            result_data['volatility_regime'] = 0.0
            result_data['volume_price_divergence'] = 0.0
            result_data['support_resistance_strength'] = 0.0
            result_data['breakout_momentum'] = 0.0

            # 专业支撑阻力特征
            result_data['distance_to_poc_pct'] = 0.0
            result_data['distance_to_nearest_support_atr'] = 0.0
            result_data['support_strength_score'] = 0.0
            result_data['last_reaction_at_poc'] = 0.0

        except Exception as e:
            self.logger.error(f"初始化目标技术特征失败: {e}")

    def _calculate_target_technical_features_with_history(self, full_data: pd.DataFrame,
                                                        target_data: pd.DataFrame,
                                                        target_dates: List[str]) -> pd.DataFrame:
        """
        为目标日期计算需要历史数据的技术特征
        关键原则：历史数据只作支撑，只为目标日期计算指标
        """
        try:
            self.logger.info("开始计算目标日期的技术特征（使用历史数据支撑）...")

            # 1. 计算简单技术特征（只为目标日期，基于当日数据）
            target_data = self._calculate_target_simple_technical_features(target_data)

            # 2. 计算复杂技术特征（只为目标日期，使用历史数据支撑）
            target_data = self._calculate_target_complex_technical_features(
                full_data, target_data, target_dates
            )

            self.logger.info("目标日期技术特征计算完成")
            return target_data

        except Exception as e:
            self.logger.error(f"目标日期技术特征计算失败: {e}")
            return target_data

    def _calculate_target_simple_technical_features(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """为目标日期计算简单技术特征（基于当日数据）"""
        try:
            # 1. trend_consistency: 趋势一致性（基于当日均线数据）
            if all(col in target_data.columns for col in ['ma5', 'ma20', 'ma60', 'close']):
                above_ma5 = (target_data['close'] > target_data['ma5']).astype(int)
                above_ma20 = (target_data['close'] > target_data['ma20']).astype(int)
                above_ma60 = (target_data['close'] > target_data['ma60']).astype(int)
                ma_alignment = (target_data['ma5'] > target_data['ma20']).astype(int) + (target_data['ma20'] > target_data['ma60']).astype(int)
                target_data['trend_consistency'] = (above_ma5 + above_ma20 + above_ma60 + ma_alignment) / 5.0
                self.logger.info("✅ trend_consistency计算完成（基于当日均线数据）")
            else:
                target_data['trend_consistency'] = 0.5
                self.logger.warning("⚠️ 缺少均线字段，trend_consistency使用默认值0.5")

            # 2. bb_position: 布林带位置（基于当日布林带数据）
            if 'bb_lower' in target_data.columns and 'close' in target_data.columns:
                if 'ma20' in target_data.columns and 'volatility20' in target_data.columns:
                    ma20_safe = np.where(np.isfinite(target_data['ma20']), target_data['ma20'], target_data['close'])
                    volatility20_safe = np.where(np.isfinite(target_data['volatility20']), target_data['volatility20'], 0.02)
                    close_safe = np.where(np.isfinite(target_data['close']), target_data['close'], ma20_safe)
                    bb_lower_safe = np.where(np.isfinite(target_data['bb_lower']), target_data['bb_lower'], ma20_safe * 0.98)

                    bb_upper = ma20_safe + 2 * volatility20_safe * close_safe
                    bb_width = bb_upper - bb_lower_safe

                    safe_mask = (bb_width > 1e-10) & np.isfinite(bb_width) & np.isfinite(close_safe) & np.isfinite(bb_lower_safe)
                    bb_position = np.full(len(target_data), 0.5)

                    if np.any(safe_mask):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            bb_position[safe_mask] = (close_safe[safe_mask] - bb_lower_safe[safe_mask]) / bb_width[safe_mask]

                    target_data['bb_position'] = np.clip(bb_position, 0, 1)
                    self.logger.info("✅ bb_position计算完成（基于当日布林带数据）")
                else:
                    target_data['bb_position'] = 0.5
                    self.logger.warning("⚠️ 缺少布林带计算所需字段，bb_position使用默认值0.5")
            else:
                target_data['bb_position'] = 0.5
                self.logger.warning("⚠️ 缺少布林带字段，bb_position使用默认值0.5")

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标简单技术特征失败: {e}")
            return target_data

    def _calculate_target_complex_technical_features(self, full_data: pd.DataFrame,
                                                   target_data: pd.DataFrame,
                                                   target_dates: List[str]) -> pd.DataFrame:
        """为目标日期计算复杂技术特征（使用历史数据支撑）"""
        try:
            # 按股票分组处理
            target_stocks = target_data['stock_code'].unique()
            total_stocks = len(target_stocks)

            self.logger.info(f"准备处理 {total_stocks} 只股票的复杂技术特征")

            for idx, stock_code in enumerate(target_stocks):
                if idx % max(1, total_stocks // 10) == 0 or idx == total_stocks - 1:
                    progress_pct = (idx + 1) / total_stocks * 100
                    self.logger.info(f"处理进度: {idx + 1}/{total_stocks} ({progress_pct:.1f}%)")

                # 获取该股票的完整历史数据
                stock_full_data = full_data[full_data['stock_code'] == stock_code].sort_values('date')

                # 获取该股票的目标日期数据
                stock_target_mask = target_data['stock_code'] == stock_code

                if len(stock_full_data) < 60:  # 需要足够的历史数据计算技术特征
                    continue

                # 为该股票的目标日期计算复杂技术特征
                stock_target_features = self._calculate_stock_target_technical_features(
                    stock_full_data, target_dates
                )

                # 更新目标数据
                self._update_target_technical_features(target_data, stock_target_features, stock_target_mask)

            return target_data

        except Exception as e:
            self.logger.error(f"计算目标复杂技术特征失败: {e}")
            return target_data

    def _calculate_stock_target_technical_features(self, stock_full_data: pd.DataFrame,
                                                 target_dates: List[str]) -> dict:
        """为单只股票的目标日期计算复杂技术特征"""
        try:
            target_features = {}

            # 预处理数据
            required_columns = ['close', 'high', 'low', 'volume']
            if not all(col in stock_full_data.columns for col in required_columns):
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

                # 计算该日期的特征
                features = self._calculate_technical_features_for_single_date(
                    stock_full_data, data_idx
                )

                target_features[target_date_obj] = features

            return target_features

        except Exception as e:
            self.logger.error(f"计算单股票目标技术特征失败: {e}")
            return {}

    def _calculate_technical_features_for_single_date(self, stock_data: pd.DataFrame,
                                                    data_idx: int) -> dict:
        """为单个目标日期计算所有复杂技术特征"""
        try:
            features = {}

            # 确保有足够的历史数据
            if data_idx < 60:
                return self._get_default_technical_features()

            # 获取必要的数据
            close = stock_data['close'].values
            high = stock_data['high'].values
            low = stock_data['low'].values
            volume = stock_data['volume'].values

            # 1. momentum_acceleration: 动量加速度（需要历史数据）
            features['momentum_acceleration'] = self._calculate_momentum_acceleration_for_date(
                close, data_idx
            )

            # 2. rsi_divergence: RSI背离（需要历史数据）
            features['rsi_divergence'] = self._calculate_rsi_divergence_for_date(
                stock_data, data_idx
            )

            # 3. macd_signal_strength: MACD信号强度（需要历史数据）
            features['macd_signal_strength'] = self._calculate_macd_signal_strength_for_date(
                stock_data, data_idx
            )

            # 4. price_channel_position: 价格通道位置（需要历史数据）
            features['price_channel_position'] = self._calculate_price_channel_position_for_date(
                high, low, close, data_idx
            )

            # 5. volatility_regime: 波动率状态（需要历史数据）
            features['volatility_regime'] = self._calculate_volatility_regime_for_date(
                stock_data, data_idx
            )

            # 6. volume_price_divergence: 量价背离（需要历史数据）
            features['volume_price_divergence'] = self._calculate_volume_price_divergence_for_date(
                close, volume, data_idx
            )

            # 7. support_resistance_strength: 支撑阻力强度（需要历史数据）
            features['support_resistance_strength'] = self._calculate_support_resistance_strength_for_date(
                high, low, close, data_idx
            )

            # 8. breakout_momentum: 突破动量（需要历史数据）
            features['breakout_momentum'] = self._calculate_breakout_momentum_for_date(
                high, low, close, volume, data_idx
            )

            # 9. 专业支撑阻力特征（需要历史数据）
            support_features = self._calculate_support_resistance_features_for_date(
                stock_data, data_idx
            )
            features.update(support_features)

            return features

        except Exception as e:
            self.logger.error(f"计算单日期技术特征失败: {e}")
            return self._get_default_technical_features()

    def _calculate_momentum_acceleration_for_date(self, close: np.ndarray, data_idx: int) -> float:
        """为单个目标日期计算动量加速度"""
        try:
            if data_idx < 20:
                return 0.0

            # 计算短期和长期动量
            short_momentum = (close[data_idx] - close[data_idx - 5]) / close[data_idx - 5]
            long_momentum = (close[data_idx] - close[data_idx - 20]) / close[data_idx - 20]

            # 动量加速度
            acceleration = short_momentum - long_momentum
            return float(np.clip(acceleration, -0.2, 0.2))

        except Exception as e:
            self.logger.error(f"计算动量加速度失败: {e}")
            return 0.0

    def _calculate_rsi_divergence_for_date(self, stock_data: pd.DataFrame, data_idx: int) -> float:
        """为单个目标日期计算RSI背离"""
        try:
            # 检查是否有预计算的RSI
            rsi_columns = ['rsi6', 'rsi12', 'rsi14']
            available_rsi = [col for col in rsi_columns if col in stock_data.columns]

            if available_rsi:
                # 使用预计算的RSI
                rsi_col = available_rsi[0]
                if data_idx >= 10:
                    current_rsi = stock_data.iloc[data_idx][rsi_col]
                    prev_rsi = stock_data.iloc[data_idx - 10][rsi_col]

                    if pd.notna(current_rsi) and pd.notna(prev_rsi):
                        rsi_change = current_rsi - prev_rsi
                        return float(np.clip(rsi_change / 100.0, -0.5, 0.5))

            return 0.0

        except Exception as e:
            self.logger.error(f"计算RSI背离失败: {e}")
            return 0.0

    def _calculate_macd_signal_strength_for_date(self, stock_data: pd.DataFrame, data_idx: int) -> float:
        """为单个目标日期计算MACD信号强度"""
        try:
            # 检查是否有预计算的MACD
            if 'macd' in stock_data.columns and 'macd_signal' in stock_data.columns:
                current_macd = stock_data.iloc[data_idx]['macd']
                current_signal = stock_data.iloc[data_idx]['macd_signal']

                if pd.notna(current_macd) and pd.notna(current_signal):
                    signal_strength = current_macd - current_signal
                    return float(np.clip(signal_strength, -1.0, 1.0))

            return 0.0

        except Exception as e:
            self.logger.error(f"计算MACD信号强度失败: {e}")
            return 0.0

    def _get_default_technical_features(self) -> dict:
        """获取默认的技术特征值"""
        return {
            'momentum_acceleration': 0.0,
            'rsi_divergence': 0.0,
            'macd_signal_strength': 0.0,
            'price_channel_position': 0.5,
            'volatility_regime': 0.0,
            'volume_price_divergence': 0.0,
            'support_resistance_strength': 0.0,
            'breakout_momentum': 0.0,
            'distance_to_poc_pct': 0.0,
            'distance_to_nearest_support_atr': 0.0,
            'support_strength_score': 0.0,
            'last_reaction_at_poc': 0.0
        }

    def _update_target_technical_features(self, target_data: pd.DataFrame,
                                        stock_target_features: dict,
                                        stock_target_mask: pd.Series):
        """更新目标数据中的技术特征"""
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
            self.logger.error(f"更新目标技术特征失败: {e}")

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

    def _calculate_trend_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算趋势一致性"""
        try:
            if all(col in data.columns for col in ['ma5', 'ma20', 'ma60', 'close']):
                # 价格与多条均线的一致性
                above_ma5 = (data['close'] > data['ma5']).astype(int)
                above_ma20 = (data['close'] > data['ma20']).astype(int)
                above_ma60 = (data['close'] > data['ma60']).astype(int)
                ma_alignment = (data['ma5'] > data['ma20']).astype(int) + (data['ma20'] > data['ma60']).astype(int)
                
                # 趋势一致性评分 (0-1)
                data['trend_consistency'] = (above_ma5 + above_ma20 + above_ma60 + ma_alignment) / 5.0
            else:
                data['trend_consistency'] = 0.5  # 中性值

            return data
        except Exception as e:
            self.logger.error(f"计算趋势一致性失败: {e}")
            data['trend_consistency'] = 0.5
            return data
    
    def _calculate_momentum_acceleration(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算动量加速度"""
        try:
            if all(col in data.columns for col in ['return_5d', 'return_20d']):
                # 短期动量相对长期动量的加速度
                data['momentum_acceleration'] = np.where(
                    data['return_20d'] != 0,
                    (data['return_5d'] - data['return_20d'] / 4) / (np.abs(data['return_20d']) + 0.01),
                    0
                )
                # 限制在合理范围内
                data['momentum_acceleration'] = np.clip(data['momentum_acceleration'], -2, 2)
            else:
                data['momentum_acceleration'] = 0

            return data
        except Exception as e:
            self.logger.error(f"计算动量加速度失败: {e}")
            data['momentum_acceleration'] = 0
            return data
    
    def _calculate_bb_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带位置"""
        try:
            if 'bb_lower' in data.columns and 'close' in data.columns:
                # 需要计算布林带上轨，基于20日均线和标准差
                if 'ma20' in data.columns and 'volatility20' in data.columns:
                    # 安全计算，避免Inf值
                    ma20_safe = np.where(np.isfinite(data['ma20']), data['ma20'], data['close'])
                    volatility20_safe = np.where(np.isfinite(data['volatility20']), data['volatility20'], 0.02)
                    close_safe = np.where(np.isfinite(data['close']), data['close'], ma20_safe)
                    bb_lower_safe = np.where(np.isfinite(data['bb_lower']), data['bb_lower'], ma20_safe * 0.98)

                    # 计算布林带上轨
                    bb_upper = ma20_safe + 2 * volatility20_safe * close_safe
                    bb_width = bb_upper - bb_lower_safe

                    # 安全计算布林带位置，避免除零警告
                    # 首先创建安全的除法掩码
                    safe_mask = (bb_width > 1e-10) & np.isfinite(bb_width) & np.isfinite(close_safe) & np.isfinite(bb_lower_safe)

                    # 初始化结果数组
                    bb_position = np.full(len(data), 0.5)

                    # 只对安全的数据进行除法计算
                    if np.any(safe_mask):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            bb_position[safe_mask] = (close_safe[safe_mask] - bb_lower_safe[safe_mask]) / bb_width[safe_mask]

                    data['bb_position'] = bb_position

                    # 确保结果有限且在合理范围内
                    data['bb_position'] = np.clip(data['bb_position'], 0, 1)
                    data['bb_position'] = np.where(np.isfinite(data['bb_position']), data['bb_position'], 0.5)
                else:
                    data['bb_position'] = 0.5
            else:
                data['bb_position'] = 0.5
            
            return data
        except Exception as e:
            self.logger.error(f"计算布林带位置失败: {e}")
            data['bb_position'] = 0.5
            return data
    
    def _calculate_rsi_divergence(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算RSI背离（向量化优化版）"""
        try:
            if all(col in data.columns for col in ['rsi6', 'close']):
                self.logger.debug("开始向量化计算RSI背离特征")

                # 向量化计算价格和RSI的5日变化
                data = data.sort_values(['stock_code', 'date'])
                price_change_5d = data.groupby('stock_code')['close'].pct_change(5)
                rsi_change_5d = data.groupby('stock_code')['rsi6'].diff(5)

                # 向量化计算RSI背离
                data['rsi_divergence'] = np.where(
                    (price_change_5d > 0.02) & (rsi_change_5d < -5),  # 价格涨但RSI跌
                    -1,  # 顶背离
                    np.where(
                        (price_change_5d < -0.02) & (rsi_change_5d > 5),  # 价格跌但RSI涨
                        1,   # 底背离
                        0    # 无背离
                    )
                )

                self.logger.debug("RSI背离特征向量化计算完成")
            else:
                data['rsi_divergence'] = 0

            return data
        except Exception as e:
            self.logger.error(f"计算RSI背离失败: {e}")
            data['rsi_divergence'] = 0
            return data
    
    def _calculate_macd_signal_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD信号强度"""
        try:
            if 'macd' in data.columns:
                # 向量化计算MACD信号强度
                self.logger.debug("开始向量化计算MACD信号强度")

                data = data.sort_values(['stock_code', 'date'])
                macd_change = data.groupby('stock_code')['macd'].diff()
                macd_acceleration = macd_change.groupby(data['stock_code']).diff()

                # 向量化计算信号强度
                data['macd_signal_strength'] = (
                    np.sign(data['macd']) * 0.4 +  # MACD方向
                    np.sign(macd_change) * 0.4 +   # MACD变化方向
                    np.sign(macd_acceleration) * 0.2  # MACD加速度方向
                ).fillna(0)

                self.logger.debug("MACD信号强度向量化计算完成")
            else:
                data['macd_signal_strength'] = 0
            
            return data
        except Exception as e:
            self.logger.error(f"计算MACD信号强度失败: {e}")
            data['macd_signal_strength'] = 0
            return data
    
    def _calculate_price_channel_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算价格通道位置"""
        try:
            if all(col in data.columns for col in ['high', 'low', 'close']):
                # 向量化计算20日价格通道
                self.logger.debug("开始向量化计算价格通道位置")

                data = data.sort_values(['stock_code', 'date'])
                high_20d = data.groupby('stock_code')['high'].rolling(window=20, min_periods=20).max().reset_index(0, drop=True)
                low_20d = data.groupby('stock_code')['low'].rolling(window=20, min_periods=20).min().reset_index(0, drop=True)

                # 向量化计算价格在通道中的位置，避免除零警告
                channel_range = high_20d - low_20d

                # 安全的除法计算
                safe_range_mask = channel_range > 1e-10
                channel_position = np.full(len(data), 0.5)

                if np.any(safe_range_mask):
                    with np.errstate(divide='ignore', invalid='ignore'):
                        channel_position[safe_range_mask] = (
                            (data['close'][safe_range_mask] - low_20d[safe_range_mask]) /
                            channel_range[safe_range_mask]
                        )

                data['price_channel_position'] = np.clip(channel_position, 0, 1)
                self.logger.debug("价格通道位置向量化计算完成")
            else:
                data['price_channel_position'] = 0.5
            
            return data
        except Exception as e:
            self.logger.error(f"计算价格通道位置失败: {e}")
            data['price_channel_position'] = 0.5
            return data

    def _calculate_volatility_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率状态"""
        try:
            if 'volatility20' in data.columns:
                # 按股票分组计算波动率状态
                for stock_code in data['stock_code'].unique():
                    stock_mask = data['stock_code'] == stock_code
                    stock_data = data[stock_mask].copy()

                    if len(stock_data) >= 60:
                        stock_data = stock_data.sort_values('date')

                        # 60日波动率分位数
                        vol_60d_quantile = stock_data['volatility20'].rolling(window=60, min_periods=60).rank(pct=True)

                        # 波动率状态：低波动(0-0.3)、中波动(0.3-0.7)、高波动(0.7-1.0)
                        volatility_regime = np.where(
                            vol_60d_quantile <= 0.3, 0,  # 低波动
                            np.where(vol_60d_quantile <= 0.7, 1, 2)  # 中波动、高波动
                        )

                        data.loc[stock_mask, 'volatility_regime'] = volatility_regime
                    else:
                        data.loc[stock_mask, 'volatility_regime'] = 1  # 默认中波动
            else:
                data['volatility_regime'] = 1

            return data
        except Exception as e:
            self.logger.error(f"计算波动率状态失败: {e}")
            data['volatility_regime'] = 1
            return data

    def _calculate_volume_price_divergence(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算量价背离"""
        try:
            if all(col in data.columns for col in ['volume', 'close']):
                # 按股票分组计算量价背离
                for stock_code in data['stock_code'].unique():
                    stock_mask = data['stock_code'] == stock_code
                    stock_data = data[stock_mask].copy()

                    if len(stock_data) >= 10:
                        stock_data = stock_data.sort_values('date')

                        # 5日价格和成交量变化
                        price_change_5d = stock_data['close'].pct_change(5)
                        volume_change_5d = stock_data['volume'].pct_change(5)

                        # 量价背离：价格涨量缩，或价格跌量增
                        divergence = np.where(
                            (price_change_5d > 0.02) & (volume_change_5d < -0.1),  # 价格涨量缩
                            -1,  # 负背离
                            np.where(
                                (price_change_5d < -0.02) & (volume_change_5d > 0.1),  # 价格跌量增
                                1,   # 正背离
                                0    # 无背离
                            )
                        )

                        data.loc[stock_mask, 'volume_price_divergence'] = divergence
                    else:
                        data.loc[stock_mask, 'volume_price_divergence'] = 0
            else:
                data['volume_price_divergence'] = 0

            return data
        except Exception as e:
            self.logger.error(f"计算量价背离失败: {e}")
            data['volume_price_divergence'] = 0
            return data

    def _calculate_support_resistance_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算专业的支撑阻力强度 - 高性能优化版本
        基于成交量分布和价格行为分析，实现Volume Profile方法

        优化策略：
        1. 向量化计算减少循环
        2. 增量式Volume Profile更新
        3. 预计算和缓存重用
        4. 内存优化的数据结构
        """
        try:
            # 初始化特征列
            feature_columns = [
                'support_resistance_strength', 'distance_to_poc_pct',
                'distance_to_nearest_support_atr', 'support_strength_score', 'last_reaction_at_poc'
            ]
            for col in feature_columns:
                data[col] = 0.0

            if not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                self.logger.warning("缺少必要字段，使用默认值")
                return data

            # 按股票分组计算 - 使用向量化处理
            unique_stocks = data['stock_code'].unique()
            batch_size = 200  # 增大批次大小，减少循环开销

            for i in range(0, len(unique_stocks), batch_size):
                batch_stocks = unique_stocks[i:i + batch_size]

                # 批量处理股票
                for stock_code in batch_stocks:
                    stock_mask = data['stock_code'] == stock_code
                    stock_data = data[stock_mask].copy().sort_values('date')

                    if len(stock_data) >= 30:  # 至少需要30个交易日
                        # 使用高性能的特征计算方法
                        features = self._calculate_volume_profile_features_optimized(stock_data)

                        # 批量更新数据
                        for feature_name, values in features.items():
                            if len(values) == len(stock_data):
                                data.loc[stock_mask, feature_name] = values

                # 减少垃圾回收频率
                if i % 1000 == 0:
                    import gc
                    gc.collect()

            return data

        except Exception as e:
            self.logger.error(f"计算支撑阻力强度失败: {e}")
            # 确保所有特征列都存在
            for col in feature_columns:
                if col not in data.columns:
                    data[col] = 0.0
            return data

    def _calculate_volume_profile_features_optimized(self, stock_data: pd.DataFrame) -> dict:
        """
        高性能优化的Volume Profile特征计算

        优化策略：
        1. 预计算价格区间和ATR
        2. 向量化的Volume Profile构建
        3. 增量式滚动窗口更新
        4. 缓存重复计算结果
        """
        try:
            window_size = min(30, len(stock_data))
            n_days = len(stock_data)

            # 预计算ATR - 向量化实现
            atr_values = self._calculate_atr_vectorized(stock_data, period=14)

            # 初始化结果数组 - 使用numpy数组提高性能
            features = {
                'support_resistance_strength': np.zeros(n_days, dtype=np.float32),
                'distance_to_poc_pct': np.zeros(n_days, dtype=np.float32),
                'distance_to_nearest_support_atr': np.zeros(n_days, dtype=np.float32),
                'support_strength_score': np.zeros(n_days, dtype=np.float32),
                'last_reaction_at_poc': np.zeros(n_days, dtype=np.float32)
            }

            # 预计算价格和成交量数组
            prices = stock_data[['high', 'low', 'close']].values.astype(np.float32)
            volumes = stock_data['volume'].values.astype(np.float32)

            # 批量计算滚动窗口特征
            start_idx = window_size - 1
            if start_idx >= n_days:
                return {k: v.astype(float) for k, v in features.items()}

            # 使用滑动窗口批量处理
            for i in range(start_idx, n_days, 5):  # 每5天计算一次，然后插值
                end_idx = min(i + 5, n_days)

                for j in range(i, end_idx):
                    # 获取窗口数据
                    window_start = max(0, j - window_size + 1)
                    window_prices = prices[window_start:j + 1]
                    window_volumes = volumes[window_start:j + 1]

                    current_price = prices[j, 2]  # close price
                    current_atr = atr_values[j] if j < len(atr_values) else 1.0

                    # 高性能Volume Profile构建
                    poc_price, support_levels = self._build_volume_profile_fast(
                        window_prices, window_volumes
                    )

                    if poc_price > 0:
                        # 计算特征
                        features['distance_to_poc_pct'][j] = (
                            (current_price - poc_price) / current_price if current_price > 0 else 0
                        )

                        # 找到最近支撑位
                        nearest_support = self._find_nearest_support_fast(current_price, support_levels)
                        features['distance_to_nearest_support_atr'][j] = (
                            (current_price - nearest_support) / current_atr if current_atr > 0 else 0
                        )

                        # 简化的支撑强度评分
                        features['support_strength_score'][j] = self._calculate_support_strength_fast(
                            current_price, support_levels, window_volumes
                        )

                        # 简化的POC反应
                        features['last_reaction_at_poc'][j] = self._calculate_poc_reaction_fast(
                            poc_price, window_prices, current_price
                        )

                        # 综合支撑阻力强度
                        poc_distance = abs(features['distance_to_poc_pct'][j])
                        support_factor = features['support_strength_score'][j] / 100.0

                        if current_price > poc_price:
                            features['support_resistance_strength'][j] = np.exp(-poc_distance * 20) * (1 + support_factor)
                        else:
                            features['support_resistance_strength'][j] = -np.exp(-poc_distance * 20) * (1 + support_factor)

                        # 限制范围
                        features['support_resistance_strength'][j] = np.clip(
                            features['support_resistance_strength'][j], -2, 2
                        )

            # 对稀疏计算的结果进行线性插值
            for feature_name, values in features.items():
                if np.any(values[start_idx:] != 0):  # 如果有非零值
                    features[feature_name] = self._interpolate_sparse_values(values, start_idx)

            return {k: v.astype(float) for k, v in features.items()}

        except Exception as e:
            self.logger.debug(f"优化的Volume Profile特征计算失败: {e}")
            # 回退到原始方法
            return self._calculate_volume_profile_features_fallback(stock_data)

    def _build_volume_profile_fast(self, window_prices: np.ndarray, window_volumes: np.ndarray, num_bins: int = 30) -> tuple:
        """
        高性能Volume Profile构建 - 向量化实现

        返回: (poc_price, support_levels)
        """
        try:
            if len(window_prices) == 0:
                return 0.0, []

            # 获取价格范围
            price_min = np.min(window_prices[:, 1])  # low prices
            price_max = np.max(window_prices[:, 0])  # high prices

            if price_max <= price_min:
                return 0.0, []

            # 创建价格区间 - 减少区间数量提高性能
            price_bins = np.linspace(price_min, price_max, num_bins + 1)
            price_centers = (price_bins[:-1] + price_bins[1:]) / 2

            # 向量化的成交量分配
            volume_by_bin = np.zeros(num_bins, dtype=np.float32)

            # 批量处理价格区间分配
            for i in range(len(window_prices)):
                low, high = window_prices[i, 1], window_prices[i, 0]
                volume = window_volumes[i]

                if high > low:
                    # 向量化的区间查找
                    low_bin = np.searchsorted(price_bins, low, side='right') - 1
                    high_bin = np.searchsorted(price_bins, high, side='left')

                    low_bin = max(0, min(low_bin, num_bins - 1))
                    high_bin = max(0, min(high_bin, num_bins - 1))

                    # 快速成交量分配
                    if low_bin == high_bin:
                        volume_by_bin[low_bin] += volume
                    else:
                        # 向量化分配
                        bin_count = high_bin - low_bin + 1
                        volume_by_bin[low_bin:high_bin + 1] += volume / bin_count
                else:
                    bin_idx = np.searchsorted(price_bins, low, side='right') - 1
                    bin_idx = max(0, min(bin_idx, num_bins - 1))
                    volume_by_bin[bin_idx] += volume

            # 找到POC
            poc_idx = np.argmax(volume_by_bin)
            poc_price = price_centers[poc_idx]

            # 快速识别支撑位 - 使用阈值方法
            volume_threshold = np.percentile(volume_by_bin[volume_by_bin > 0], 70)
            support_indices = np.where(volume_by_bin >= volume_threshold)[0]
            support_levels = price_centers[support_indices].tolist()

            return float(poc_price), support_levels

        except Exception as e:
            return 0.0, []

    def _find_nearest_support_fast(self, current_price: float, support_levels: list) -> float:
        """快速找到最近支撑位"""
        if not support_levels:
            return current_price * 0.95

        support_array = np.array(support_levels)
        valid_supports = support_array[support_array < current_price]

        if len(valid_supports) > 0:
            return float(np.max(valid_supports))
        else:
            return float(np.min(support_array)) if len(support_array) > 0 else current_price * 0.95

    def _calculate_support_strength_fast(self, current_price: float, support_levels: list, window_volumes: np.ndarray) -> float:
        """快速计算支撑强度评分"""
        if not support_levels:
            return 0.0

        # 简化的强度计算
        nearest_support = self._find_nearest_support_fast(current_price, support_levels)
        distance_ratio = abs(current_price - nearest_support) / current_price if current_price > 0 else 1.0

        # 基于距离和成交量的简化评分
        volume_factor = np.mean(window_volumes[-10:]) / np.mean(window_volumes) if len(window_volumes) > 10 else 1.0
        strength = (1 - distance_ratio) * volume_factor * 100

        return float(np.clip(strength, 0, 100))

    def _calculate_poc_reaction_fast(self, poc_price: float, window_prices: np.ndarray, current_price: float) -> float:
        """快速计算POC反应"""
        if len(window_prices) < 5:
            return 0.0

        tolerance = poc_price * 0.02

        # 简化的触及检测
        for i in range(len(window_prices) - 3, -1, -1):
            low, high = window_prices[i, 1], window_prices[i, 0]
            close = window_prices[i, 2]

            if low <= poc_price + tolerance and high >= poc_price - tolerance:
                # 计算后续反应
                if i < len(window_prices) - 2:
                    future_close = window_prices[i + 1, 2]
                    reaction = (future_close - close) / close if close > 0 else 0
                    return float(np.clip(reaction * 10, -1, 1))
                break

        return 0.0

    def _interpolate_sparse_values(self, values: np.ndarray, start_idx: int) -> np.ndarray:
        """对稀疏计算的值进行线性插值"""
        if start_idx >= len(values):
            return values

        # 找到非零值的位置
        nonzero_indices = np.nonzero(values[start_idx:])[0] + start_idx

        if len(nonzero_indices) < 2:
            return values

        # 对中间的零值进行线性插值
        for i in range(len(nonzero_indices) - 1):
            start_pos = nonzero_indices[i]
            end_pos = nonzero_indices[i + 1]

            if end_pos - start_pos > 1:
                # 线性插值
                start_val = values[start_pos]
                end_val = values[end_pos]

                for j in range(start_pos + 1, end_pos):
                    ratio = (j - start_pos) / (end_pos - start_pos)
                    values[j] = start_val + (end_val - start_val) * ratio

        return values

    def _calculate_atr_vectorized(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """向量化的ATR计算"""
        try:
            if len(data) < period:
                return np.ones(len(data), dtype=np.float32)

            # 向量化计算真实波幅
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values

            high_low = high - low
            high_close_prev = np.abs(high[1:] - close[:-1])
            low_close_prev = np.abs(low[1:] - close[:-1])

            # 计算真实波幅
            true_range = np.zeros(len(data), dtype=np.float32)
            true_range[0] = high_low[0]
            true_range[1:] = np.maximum(high_low[1:], np.maximum(high_close_prev, low_close_prev))

            # 向量化的移动平均
            atr = np.zeros(len(data), dtype=np.float32)
            atr[:period] = np.mean(true_range[:period])

            # 使用指数移动平均进行快速计算
            alpha = 1.0 / period
            for i in range(period, len(data)):
                atr[i] = atr[i-1] * (1 - alpha) + true_range[i] * alpha

            return atr

        except Exception as e:
            return np.ones(len(data), dtype=np.float32)

    def _calculate_volume_profile_features_fallback(self, stock_data: pd.DataFrame) -> dict:
        """回退方法 - 使用原始实现"""
        return self._calculate_volume_profile_features_original(stock_data)

    def _calculate_volume_profile_features_original(self, stock_data: pd.DataFrame) -> dict:
        """
        计算基于成交量分布的Volume Profile特征 - 原始实现
        """
        try:
            # 使用最近30个交易日的数据
            window_size = min(30, len(stock_data))
            recent_data = stock_data.tail(window_size).copy()

            # 计算ATR用于标准化距离
            atr = self._calculate_atr(recent_data, period=14)

            # 初始化结果
            features = {
                'support_resistance_strength': np.zeros(len(stock_data)),
                'distance_to_poc_pct': np.zeros(len(stock_data)),
                'distance_to_nearest_support_atr': np.zeros(len(stock_data)),
                'support_strength_score': np.zeros(len(stock_data)),
                'last_reaction_at_poc': np.zeros(len(stock_data))
            }

            # 对每个交易日计算特征（滚动窗口）
            for i in range(len(stock_data)):
                if i < window_size - 1:
                    # 数据不足，使用默认值
                    continue

                # 获取当前窗口数据
                window_data = stock_data.iloc[max(0, i - window_size + 1):i + 1].copy()
                current_price = stock_data.iloc[i]['close']
                current_atr = atr[i] if i < len(atr) else 1.0

                # 1. 构建Volume Profile
                volume_profile = self._build_volume_profile(window_data)

                if volume_profile is None or len(volume_profile) == 0:
                    continue

                # 2. 找到POC (Point of Control) - 成交量最大的价格区间
                poc_price = volume_profile['price'].iloc[volume_profile['volume'].idxmax()]

                # 3. 识别支撑阻力位
                support_resistance_levels = self._identify_support_resistance_levels(volume_profile, window_data)

                # 4. 计算标准化距离特征
                # 距离POC的百分比距离
                distance_to_poc_pct = (current_price - poc_price) / current_price if current_price > 0 else 0

                # 距离最近支撑位的ATR标准化距离
                nearest_support = self._find_nearest_support(current_price, support_resistance_levels['support'])
                distance_to_support_atr = (current_price - nearest_support) / current_atr if current_atr > 0 else 0

                # 5. 计算支撑强度评分
                support_strength = self._calculate_support_strength_score(
                    current_price, support_resistance_levels, window_data
                )

                # 6. 计算历史反应特征
                last_reaction = self._calculate_last_reaction_at_poc(
                    poc_price, window_data, tolerance_pct=0.02
                )

                # 7. 计算综合支撑阻力强度
                # 基于距离POC的远近和支撑强度
                poc_strength = np.exp(-abs(distance_to_poc_pct) * 20)  # 距离POC越近强度越大
                support_factor = support_strength / 100.0  # 标准化到0-1

                # 综合强度：正值表示支撑，负值表示阻力
                if current_price > poc_price:
                    # 价格在POC上方，POC作为支撑
                    total_strength = poc_strength * (1 + support_factor)
                else:
                    # 价格在POC下方，POC作为阻力
                    total_strength = -poc_strength * (1 + support_factor)

                # 更新特征
                features['support_resistance_strength'][i] = np.clip(total_strength, -2, 2)
                features['distance_to_poc_pct'][i] = np.clip(distance_to_poc_pct, -0.1, 0.1)
                features['distance_to_nearest_support_atr'][i] = np.clip(distance_to_support_atr, -5, 5)
                features['support_strength_score'][i] = np.clip(support_strength, 0, 100)
                features['last_reaction_at_poc'][i] = np.clip(last_reaction, -1, 1)

            return features

        except Exception as e:
            self.logger.debug(f"计算Volume Profile特征失败: {e}")
            # 返回默认值
            return {
                'support_resistance_strength': np.zeros(len(stock_data)),
                'distance_to_poc_pct': np.zeros(len(stock_data)),
                'distance_to_nearest_support_atr': np.zeros(len(stock_data)),
                'support_strength_score': np.zeros(len(stock_data)),
                'last_reaction_at_poc': np.zeros(len(stock_data))
            }

    def _build_volume_profile(self, window_data: pd.DataFrame, num_bins: int = 50) -> pd.DataFrame:
        """
        构建Volume Profile - 将价格范围分成区间，统计每个区间的成交量
        """
        try:
            if len(window_data) == 0:
                return None

            # 获取价格范围
            price_min = window_data['low'].min()
            price_max = window_data['high'].max()

            if price_max <= price_min:
                return None

            # 创建价格区间
            price_bins = np.linspace(price_min, price_max, num_bins + 1)
            price_centers = (price_bins[:-1] + price_bins[1:]) / 2

            # 统计每个区间的成交量
            volume_by_bin = np.zeros(num_bins)

            for _, row in window_data.iterrows():
                # 假设成交量在high-low范围内均匀分布
                low, high, volume = row['low'], row['high'], row['volume']

                if high > low:
                    # 找到价格范围对应的区间
                    low_bin = np.searchsorted(price_bins, low, side='right') - 1
                    high_bin = np.searchsorted(price_bins, high, side='left')

                    low_bin = max(0, min(low_bin, num_bins - 1))
                    high_bin = max(0, min(high_bin, num_bins - 1))

                    # 将成交量分配到相关区间
                    if low_bin == high_bin:
                        volume_by_bin[low_bin] += volume
                    else:
                        # 按比例分配成交量
                        for bin_idx in range(low_bin, high_bin + 1):
                            if bin_idx < num_bins:
                                volume_by_bin[bin_idx] += volume / (high_bin - low_bin + 1)
                else:
                    # 价格没有波动，分配到最近的区间
                    bin_idx = np.searchsorted(price_bins, low, side='right') - 1
                    bin_idx = max(0, min(bin_idx, num_bins - 1))
                    volume_by_bin[bin_idx] += volume

            # 创建Volume Profile DataFrame
            volume_profile = pd.DataFrame({
                'price': price_centers,
                'volume': volume_by_bin
            })

            # 过滤掉成交量为0的区间
            volume_profile = volume_profile[volume_profile['volume'] > 0].reset_index(drop=True)

            return volume_profile

        except Exception as e:
            self.logger.debug(f"构建Volume Profile失败: {e}")
            return None

    def _identify_support_resistance_levels(self, volume_profile: pd.DataFrame, window_data: pd.DataFrame) -> dict:
        """
        识别支撑阻力位
        """
        try:
            if volume_profile is None or len(volume_profile) == 0:
                return {'support': [], 'resistance': []}

            # 找到高成交量节点 (High Volume Nodes)
            volume_threshold = volume_profile['volume'].quantile(0.7)  # 前30%的成交量区间
            hvn_levels = volume_profile[volume_profile['volume'] >= volume_threshold]['price'].tolist()

            # 基于历史价格行为识别支撑阻力
            current_price = window_data['close'].iloc[-1]

            # 支撑位：当前价格下方的高成交量区域
            support_levels = [price for price in hvn_levels if price < current_price]

            # 阻力位：当前价格上方的高成交量区域
            resistance_levels = [price for price in hvn_levels if price > current_price]

            # 添加传统的支撑阻力位（最近的高点和低点）
            recent_highs = window_data['high'].rolling(5).max().dropna().unique()
            recent_lows = window_data['low'].rolling(5).min().dropna().unique()

            # 合并并去重
            all_resistance = list(set(resistance_levels + [h for h in recent_highs if h > current_price]))
            all_support = list(set(support_levels + [l for l in recent_lows if l < current_price]))

            return {
                'support': sorted(all_support, reverse=True),  # 从高到低排序
                'resistance': sorted(all_resistance)  # 从低到高排序
            }

        except Exception as e:
            self.logger.debug(f"识别支撑阻力位失败: {e}")
            return {'support': [], 'resistance': []}

    def _find_nearest_support(self, current_price: float, support_levels: list) -> float:
        """找到最近的支撑位"""
        if not support_levels:
            return current_price * 0.95  # 默认支撑位为当前价格的95%

        # 找到最接近且低于当前价格的支撑位
        valid_supports = [s for s in support_levels if s < current_price]
        if valid_supports:
            return max(valid_supports)  # 最高的支撑位
        else:
            return min(support_levels) if support_levels else current_price * 0.95

    def _calculate_support_strength_score(self, current_price: float,
                                        support_resistance_levels: dict,
                                        window_data: pd.DataFrame) -> float:
        """
        计算支撑强度评分
        综合考虑：成交量、测试次数、时间远近
        """
        try:
            support_levels = support_resistance_levels.get('support', [])
            if not support_levels:
                return 0.0

            nearest_support = self._find_nearest_support(current_price, support_levels)

            # 1. 成交量权重 (40%)
            volume_score = 0.0
            tolerance = abs(current_price * 0.02)  # 2%的价格容忍度

            for _, row in window_data.iterrows():
                if abs(row['low'] - nearest_support) <= tolerance or abs(row['high'] - nearest_support) <= tolerance:
                    volume_score += row['volume']

            # 标准化成交量评分
            avg_volume = window_data['volume'].mean()
            volume_score = min(100, (volume_score / avg_volume) * 20) if avg_volume > 0 else 0

            # 2. 测试次数权重 (40%)
            test_count = 0
            for _, row in window_data.iterrows():
                if abs(row['low'] - nearest_support) <= tolerance:
                    test_count += 1

            test_score = min(100, test_count * 10)

            # 3. 时间远近权重 (20%)
            # 最近的测试更重要
            recent_tests = 0
            recent_window = window_data.tail(10)  # 最近10个交易日
            for _, row in recent_window.iterrows():
                if abs(row['low'] - nearest_support) <= tolerance:
                    recent_tests += 1

            time_score = min(100, recent_tests * 20)

            # 综合评分
            total_score = volume_score * 0.4 + test_score * 0.4 + time_score * 0.2

            return total_score

        except Exception as e:
            self.logger.debug(f"计算支撑强度评分失败: {e}")
            return 0.0

    def _calculate_last_reaction_at_poc(self, poc_price: float,
                                      window_data: pd.DataFrame,
                                      tolerance_pct: float = 0.02) -> float:
        """
        计算上次触及POC时的市场反应

        专业量化因子实现：
        1. 找到最近一次价格触及POC的时点
        2. 计算该时点后1-2天的实际收益率
        3. 编码反应：+1(强力反弹), -1(跌破), 0(无明确反应)

        时间序列安全性：
        - 在T日计算时，使用T-k日触及POC后T-k+1, T-k+2日的收益
        - 这些收益对T-k日是"未来"，但对T日是"历史"，符合因果性

        返回值：+1表示上次反弹，-1表示上次跌破，0表示无明确反应
        """
        try:
            tolerance = abs(poc_price * tolerance_pct)

            # 确保有足够的历史数据进行分析
            if len(window_data) < 5:
                return 0.0

            # 找到最近一次明确的POC触及事件
            # 策略：找到一个触及POC且后续有明显价格变化的时点
            last_touch_idx = -1

            # 从前往后搜索，找到最后一个有效的触及点
            # 确保触及后有至少2天的观察期
            for i in range(len(window_data) - 2):  # 保留最后2天作为观察期
                row = window_data.iloc[i]

                # 检查是否触及POC（价格区间与POC重叠）
                if (row['low'] <= poc_price + tolerance and
                    row['high'] >= poc_price - tolerance):

                    # 检查后续是否有足够的价格变化（避免选择震荡中的触及）
                    if i < len(window_data) - 2:
                        # 检查后续2天是否有明显的价格变化
                        touch_price = row['close']
                        future_prices = []

                        for j in range(1, min(3, len(window_data) - i)):
                            future_idx = i + j
                            if future_idx < len(window_data):
                                future_prices.append(window_data.iloc[future_idx]['close'])

                        # 如果后续有价格变化，则认为是有效触及
                        if future_prices:
                            max_change = max(abs(p - touch_price) / touch_price for p in future_prices)
                            if max_change > 0.005:  # 至少0.5%的价格变化
                                last_touch_idx = i  # 更新为最新的有效触及点

            if last_touch_idx == -1:
                return 0.0  # 没有找到触及POC的历史记录

            # 计算触及POC后1-2天的实际收益率
            touch_price = window_data.iloc[last_touch_idx]['close']

            # 收集后续1-2天的收益率
            future_returns = []
            max_observation_days = min(2, len(window_data) - last_touch_idx - 1)

            for j in range(1, max_observation_days + 1):
                future_idx = last_touch_idx + j
                if future_idx < len(window_data):
                    future_price = window_data.iloc[future_idx]['close']
                    daily_return = (future_price - touch_price) / touch_price
                    future_returns.append(daily_return)

            if not future_returns:
                return 0.0  # 没有后续数据可以观察

            # 计算平均反应强度
            avg_reaction = np.mean(future_returns)

            # 专业的反应分类逻辑
            # 基于量化交易中常用的阈值设定
            strong_reaction_threshold = 0.02  # 2%
            weak_reaction_threshold = 0.005   # 0.5%

            # 计算反应的一致性（所有观察日的方向一致性）
            positive_days = sum(1 for r in future_returns if r > 0)
            negative_days = sum(1 for r in future_returns if r < 0)
            total_days = len(future_returns)

            # 强力反弹：平均收益>2%且方向一致性>80%
            if (avg_reaction > strong_reaction_threshold and
                positive_days / total_days >= 0.8):
                return 1.0

            # 明显反弹：平均收益>0.5%且方向一致性>70%
            elif (avg_reaction > weak_reaction_threshold and
                  positive_days / total_days >= 0.7):
                return min(0.8, avg_reaction * 20)  # 按强度缩放

            # 强力跌破：平均收益<-2%且方向一致性>80%
            elif (avg_reaction < -strong_reaction_threshold and
                  negative_days / total_days >= 0.8):
                return -1.0

            # 明显跌破：平均收益<-0.5%且方向一致性>70%
            elif (avg_reaction < -weak_reaction_threshold and
                  negative_days / total_days >= 0.7):
                return max(-0.8, avg_reaction * 20)  # 按强度缩放

            # 震荡或无明确方向
            else:
                return 0.0

        except Exception as e:
            self.logger.debug(f"计算POC反应失败: {e}")
            return 0.0

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """计算平均真实波幅(ATR)"""
        try:
            if len(data) < period:
                return np.ones(len(data))

            # 计算真实波幅
            high_low = data['high'] - data['low']
            high_close_prev = np.abs(data['high'] - data['close'].shift(1))
            low_close_prev = np.abs(data['low'] - data['close'].shift(1))

            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))

            # 计算ATR
            atr = np.zeros(len(data))
            atr[:period] = np.mean(true_range[:period])  # 前period个值使用简单平均

            for i in range(period, len(data)):
                atr[i] = (atr[i-1] * (period - 1) + true_range.iloc[i]) / period

            return atr

        except Exception as e:
            self.logger.debug(f"计算ATR失败: {e}")
            return np.ones(len(data))

    def _calculate_breakout_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算突破动量"""
        try:
            if all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                # 按股票分组计算突破动量
                for stock_code in data['stock_code'].unique():
                    stock_mask = data['stock_code'] == stock_code
                    stock_data = data[stock_mask].copy()

                    if len(stock_data) >= 20:
                        stock_data = stock_data.sort_values('date')

                        # 20日最高价和最低价
                        high_20d = stock_data['high'].rolling(window=20, min_periods=20).max()
                        low_20d = stock_data['low'].rolling(window=20, min_periods=20).min()

                        # 平均成交量
                        avg_volume_20d = stock_data['volume'].rolling(window=20, min_periods=20).mean()

                        # 向上突破
                        upward_breakout = (
                            (stock_data['close'] > high_20d.shift(1)) &  # 突破前期高点
                            (stock_data['volume'] > avg_volume_20d * 1.5)  # 放量突破
                        ).astype(int)

                        # 向下突破
                        downward_breakout = (
                            (stock_data['close'] < low_20d.shift(1)) &  # 跌破前期低点
                            (stock_data['volume'] > avg_volume_20d * 1.5)  # 放量突破
                        ).astype(int)

                        # 突破动量：向上为正，向下为负
                        breakout_momentum = upward_breakout - downward_breakout

                        data.loc[stock_mask, 'breakout_momentum'] = breakout_momentum
                    else:
                        data.loc[stock_mask, 'breakout_momentum'] = 0
            else:
                data['breakout_momentum'] = 0

            return data
        except Exception as e:
            self.logger.error(f"计算突破动量失败: {e}")
            data['breakout_momentum'] = 0
            return data

    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> tuple:
        """识别目标日期和历史数据范围"""
        try:
            # 从kwargs中获取目标日期
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

    def _set_default_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """设置默认值"""
        for factor_name in self.feature_names:
            data[factor_name] = 0.0
        return data
