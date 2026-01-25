#!/usr/bin/env python3
"""
特征计算器 - 计算multi_factor_scorer.py中使用但在原始数据中不存在的特征

基于2020-2025年顶尖量化研究论文的特征工程方法
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import warnings
import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

class FeatureCalculator:
    """
    高效特征计算器
    
    实现multi_factor_scorer.py中使用但在ml_sector_strategy_features_documentation.md中
    不存在的特征，使用向量化计算提升性能
    """
    
    def __init__(self, cache_dir: str = "./feature_cache"):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "feature_cache.pkl"
        self._cache = self._load_cache()
        self.current_feature_version = self._get_current_feature_version()

    def _get_current_feature_version(self) -> str:
        """获取当前特征版本号"""
        # 基于新增的K线形态特征生成版本号
        single_day_patterns = [
            'pattern_bare_head_yang', 'pattern_bare_head_yin', 'pattern_hammer',
            'pattern_hanging_man', 'pattern_doji', 'pattern_morning_star',
            'pattern_evening_star', 'pattern_engulfing_bull', 'pattern_engulfing_bear',
            'pattern_piercing', 'pattern_dark_cloud'
        ]

        ma_convergence_features = [
            'ma_convergence', 'price_ma5_distance', 'price_ma10_distance',
            'price_ma20_distance', 'price_ma60_distance', 'ma_support_strength'
        ]

        # 新增MA交叉特征
        ma_crossover_features = [
            'ma5_above_ma10_strength', 'ma5_below_ma10_strength'
        ]

        price_distance_features = [
            'price_low_distance_3d', 'price_low_distance_5d', 'price_low_distance_10d',
            'price_low_distance_20d', 'price_low_distance_60d', 'price_low_distance_weighted'
        ]

        # 累计收益率特征
        cumulative_return_features = [
            'cumulative_return_3d', 'cumulative_return_5d', 'cumulative_return_10d'
        ]

        all_new_features = (single_day_patterns + ma_convergence_features + price_distance_features +
                           ma_crossover_features + cumulative_return_features)

        # 使用特征列表生成版本哈希
        import hashlib
        feature_string = '|'.join(sorted(all_new_features))
        feature_hash = hashlib.md5(feature_string.encode()).hexdigest()[:8]
        return f"v3.0_kline_patterns_{feature_hash}"
        
    def calculate_missing_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有缺失的特征（带缓存功能）

        Args:
            stock_data: 包含基础特征的股票数据

        Returns:
            添加了缺失特征的股票数据
        """
        try:
            self.logger.info(f"开始计算特征，输入数据: {len(stock_data)} 条记录")

            # 检查缓存
            cached_data, missing_data = self._check_cache(stock_data)

            if missing_data.empty:
                self.logger.info("所有数据都已缓存，直接返回")
                return cached_data

            self.logger.info(f"需要计算 {len(missing_data)} 条新记录")

            # 计算缺失数据的特征
            new_data = self._calculate_features_for_data(missing_data)

            # 更新缓存
            self._update_cache(new_data)

            # 合并缓存数据和新计算数据
            if not cached_data.empty:
                # 修复：使用列的并集而不是交集，避免新特征丢失
                result_data = self._merge_cached_and_new_data(cached_data, new_data)
            else:
                result_data = new_data

            # 按原始数据顺序排序
            if 'stock_code' in result_data.columns and 'date' in result_data.columns:
                result_data = result_data.sort_values(['stock_code', 'date'])

            self.logger.info(f"特征计算完成，总计: {len(result_data)} 条记录")
            return result_data

        except Exception as e:
            self.logger.error(f"特征计算失败: {e}")
            return stock_data

    def _calculate_features_for_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        为指定数据计算特征（不使用缓存）
        """
        try:
            # 1. 风险相关特征
            data = self._calculate_risk_features(data)

            # 2. 技术指标衍生特征
            data = self._calculate_technical_derived_features(data)

            # 3. 基本面衍生特征
            data = self._calculate_fundamental_derived_features(data)

            # 4. 流动性相关特征
            data = self._calculate_liquidity_features(data)

            # 5. 质量因子特征
            data = self._calculate_quality_features(data)

            # 6. 涨停质量特征（新增）- 暂时禁用：稀疏性问题导致IR计算异常（99%+零值）
            # data = self._calculate_limit_up_quality_features(data)

            # 7. K线形态识别（新增）
            data = self._calculate_kline_patterns(data)

            # 8. 累计涨幅特征（新增）
            data = self._calculate_cumulative_return_features(data)

            # 9. 均线粘合度指标（新增）
            data = self._calculate_ma_convergence_features(data)

            # 10. 股价低点距离指标（新增）
            data = self._calculate_price_distance_features(data)

            # 11. MA交叉强度特征（新增）
            data = self._calculate_ma_crossover_features(data)

            # 12. 高级技术指标特征（新增）
            data = self._calculate_advanced_technical_features(data)

            return data

        except Exception as e:
            self.logger.error(f"特征计算失败: {e}")
            return data

    def _calculate_advanced_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算高级技术指标特征"""
        try:
            # 导入AdvancedTechnicalFactors
            from ..factor_system.factor_groups.advanced_technical_factors import AdvancedTechnicalFactors

            # 创建高级技术指标计算器
            advanced_factors = AdvancedTechnicalFactors()

            # 计算高级技术指标
            enhanced_data = advanced_factors.calculate(data)

            self.logger.debug("高级技术指标特征计算完成")
            return enhanced_data

        except Exception as e:
            self.logger.warning(f"计算高级技术指标特征失败: {e}")
            # 如果计算失败，返回原始数据
            return data

    def _load_cache(self) -> pd.DataFrame:
        """加载缓存数据"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.logger.info(f"加载缓存数据: {len(cache_data)} 条记录")
                return cache_data
            else:
                self.logger.info("缓存文件不存在，创建新缓存")
                return pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}，创建新缓存")
            return pd.DataFrame()

    def _save_cache(self, data: pd.DataFrame):
        """保存缓存数据"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"缓存已保存: {len(data)} 条记录")
        except Exception as e:
            self.logger.error(f"保存缓存失败: {e}")

    def _get_cache_key(self, stock_code: str, date: str) -> str:
        """生成缓存键，统一日期格式"""
        # 统一日期格式为 YYYY-MM-DD
        try:
            if isinstance(date, str):
                # 如果是字符串，尝试解析并格式化
                if ' ' in date:  # 包含时间部分
                    date_part = date.split(' ')[0]
                else:
                    date_part = date
            else:
                # 如果是其他类型，转换为字符串后处理
                date_str = str(date)
                if ' ' in date_str:
                    date_part = date_str.split(' ')[0]
                else:
                    date_part = date_str

            return f"{stock_code}_{date_part}"
        except Exception:
            # 如果处理失败，使用原始字符串
            return f"{stock_code}_{date}"

    def _check_cache(self, stock_data: pd.DataFrame) -> tuple:
        """
        检查缓存，返回(已缓存数据, 需要计算的数据)
        支持特征版本控制
        """
        try:
            if self._cache.empty:
                return pd.DataFrame(), stock_data

            # 创建缓存键
            if 'stock_code' not in stock_data.columns or 'date' not in stock_data.columns:
                return pd.DataFrame(), stock_data

            # 检查缓存版本
            cache_version = None
            if not self._cache.empty and '_feature_version' in self._cache.columns:
                version_series = self._cache['_feature_version'].dropna()
                if not version_series.empty:
                    cache_version = version_series.iloc[0]

            if cache_version != self.current_feature_version:
                self.logger.info(f"特征版本不匹配 (缓存: {cache_version}, 当前: {self.current_feature_version})，清空缓存")
                self.clear_cache()
                return pd.DataFrame(), stock_data

            # 为输入数据生成新格式的缓存键
            stock_data['cache_key'] = stock_data.apply(
                lambda row: self._get_cache_key(row['stock_code'], str(row['date'])), axis=1
            )

            # 为缓存数据生成两种格式的键进行匹配
            # 新格式键（统一日期格式）
            self._cache['cache_key_new'] = self._cache.apply(
                lambda row: self._get_cache_key(row['stock_code'], str(row['date'])), axis=1
            )
            # 旧格式键（保持原始时间戳格式）
            self._cache['cache_key_old'] = self._cache.apply(
                lambda row: f"{row['stock_code']}_{str(row['date'])}", axis=1
            )

            # 找到已缓存的数据（支持新旧两种键格式）
            cached_keys_new = set(self._cache['cache_key_new'])
            cached_keys_old = set(self._cache['cache_key_old'])
            input_keys = set(stock_data['cache_key'])

            # 尝试匹配新格式键，如果不匹配再尝试旧格式键
            cached_mask_new = stock_data['cache_key'].isin(cached_keys_new)
            cached_mask_old = stock_data['cache_key'].isin(cached_keys_old)
            cached_mask = cached_mask_new | cached_mask_old
            missing_mask = ~cached_mask

            # 获取已缓存的数据
            cached_condition = (self._cache['cache_key_new'].isin(input_keys)) | (self._cache['cache_key_old'].isin(input_keys))
            cached_data = self._cache[cached_condition].copy()

            # 获取需要计算的数据
            missing_data = stock_data[missing_mask].copy()

            # 清理临时列
            if 'cache_key' in stock_data.columns:
                stock_data.drop('cache_key', axis=1, inplace=True)
            if not cached_data.empty:
                # 清理所有临时键列
                temp_cols = ['cache_key_new', 'cache_key_old']
                for col in temp_cols:
                    if col in cached_data.columns:
                        cached_data.drop(col, axis=1, inplace=True)
                # 移除版本信息列
                if '_feature_version' in cached_data.columns:
                    cached_data.drop('_feature_version', axis=1, inplace=True)
            if not missing_data.empty and 'cache_key' in missing_data.columns:
                missing_data.drop('cache_key', axis=1, inplace=True)

            self.logger.info(f"缓存命中: {len(cached_data)} 条，需计算: {len(missing_data)} 条")

            return cached_data, missing_data

        except Exception as e:
            self.logger.error(f"检查缓存失败: {e}")
            return pd.DataFrame(), stock_data

    def _update_cache(self, new_data: pd.DataFrame):
        """更新缓存（支持版本控制）"""
        try:
            if new_data.empty:
                return

            # 为新数据添加版本信息
            new_data_with_version = new_data.copy()
            new_data_with_version['_feature_version'] = self.current_feature_version

            if self._cache.empty:
                self._cache = new_data_with_version.copy()
            else:
                # 合并新数据到缓存
                self._cache = pd.concat([self._cache, new_data_with_version], ignore_index=True)

                # 去重（基于stock_code和date）
                if 'stock_code' in self._cache.columns and 'date' in self._cache.columns:
                    self._cache = self._cache.drop_duplicates(subset=['stock_code', 'date'], keep='last')

            # 保存到文件
            self._save_cache(self._cache)

            self.logger.info(f"缓存已更新: 新增 {len(new_data)} 条，总计 {len(self._cache)} 条，版本 {self.current_feature_version}")

        except Exception as e:
            self.logger.error(f"更新缓存失败: {e}")

    def _merge_cached_and_new_data(self, cached_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        修复后的缓存数据合并逻辑
        使用列的并集而不是交集，避免新特征丢失
        """
        try:
            if cached_data.empty:
                return new_data
            if new_data.empty:
                return cached_data

            # 获取所有列的并集
            all_columns = list(set(cached_data.columns) | set(new_data.columns))

            # 为缺失列填充默认值
            for col in all_columns:
                if col not in cached_data.columns:
                    # 根据列名确定默认值
                    if col.endswith('_strength') or col.startswith('pattern_'):
                        cached_data[col] = 0.0  # 强度和形态特征默认为0
                    elif col in ['_feature_version']:
                        cached_data[col] = self.current_feature_version
                    else:
                        cached_data[col] = np.nan  # 其他特征默认为NaN

                if col not in new_data.columns:
                    if col.endswith('_strength') or col.startswith('pattern_'):
                        new_data[col] = 0.0
                    elif col in ['_feature_version']:
                        new_data[col] = self.current_feature_version
                    else:
                        new_data[col] = np.nan

            # 确保列顺序一致
            cached_data = cached_data[all_columns]
            new_data = new_data[all_columns]

            # 合并数据
            result_data = pd.concat([cached_data, new_data], ignore_index=True)

            self.logger.info(f"缓存合并完成: 缓存{len(cached_data)}条 + 新增{len(new_data)}条 = 总计{len(result_data)}条")
            self.logger.info(f"合并后列数: {len(result_data.columns)}")

            # 验证关键特征
            key_features = ['ma5_above_ma10_strength', 'ma5_below_ma10_strength']
            for feature in key_features:
                if feature in result_data.columns:
                    non_zero_count = (result_data[feature] != 0).sum()
                    self.logger.info(f"特征验证 {feature}: {non_zero_count} 条非零记录")
                else:
                    self.logger.warning(f"特征验证失败: {feature} 列不存在")

            return result_data

        except Exception as e:
            self.logger.error(f"缓存合并失败: {e}")
            # 出错时返回新数据
            return new_data

    def clear_cache(self):
        """清空缓存"""
        try:
            self._cache = pd.DataFrame()
            if self.cache_file.exists():
                self.cache_file.unlink()
            self.logger.info("缓存已清空")
        except Exception as e:
            self.logger.error(f"清空缓存失败: {e}")
    
    def _calculate_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算风险相关特征"""
        try:
            # 1. high_volatility: 高波动率标识
            if 'volatility20' in data.columns:
                # 基于20日波动率的分位数，前20%为高波动率
                vol_threshold = data['volatility20'].quantile(0.8)
                data['high_volatility'] = (data['volatility20'] > vol_threshold).astype(int)
            else:
                data['high_volatility'] = 0
            
            # 2. recent_limit_down: 近期跌停标识
            if 'limit_down_days_5d' in data.columns:
                # 近5日有跌停记录
                data['recent_limit_down'] = (data['limit_down_days_5d'] > 0).astype(int)
            elif 'is_limit_down' in data.columns:
                # 当日跌停
                data['recent_limit_down'] = data['is_limit_down']
            else:
                data['recent_limit_down'] = 0
            
            # 3. low_quality_limit: 低质量涨停标识
            if 'limit_quality_score' in data.columns:
                # 涨停质量评分低于30分视为低质量
                data['low_quality_limit'] = (data['limit_quality_score'] < 30).astype(int)
            elif 'is_limit_up' in data.columns and 'time_to_limit_minutes' in data.columns:
                # 涨停但封板时间较晚（超过60分钟）
                data['low_quality_limit'] = (
                    (data['is_limit_up'] == 1) & 
                    (data['time_to_limit_minutes'] > 60)
                ).astype(int)
            else:
                data['low_quality_limit'] = 0
            
            return data
            
        except Exception as e:
            self.logger.error(f"风险特征计算失败: {e}")
            return data
    
    def _calculate_technical_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标衍生特征"""
        try:
            # 1. trend_consistency: 趋势一致性
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
            
            # 2. momentum_acceleration: 动量加速度
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
            
            # 3. bb_position: 布林带位置（安全计算，避免Inf值）
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

                    # 安全计算布林带位置
                    data['bb_position'] = np.where(
                        (bb_width > 0) & np.isfinite(bb_width) & np.isfinite(close_safe) & np.isfinite(bb_lower_safe),
                        (close_safe - bb_lower_safe) / bb_width,
                        0.5
                    )

                    # 确保结果有限且在合理范围内
                    data['bb_position'] = np.where(
                        np.isfinite(data['bb_position']),
                        np.clip(data['bb_position'], 0, 1),
                        0.5
                    )
                else:
                    data['bb_position'] = 0.5
            else:
                data['bb_position'] = 0.5

            # 4. technical_consensus: 技术信号一致性（安全计算）
            if all(x in data.columns for x in ['rsi12', 'macd', 'bb_position']):
                # 安全处理各个组件，避免Inf值

                # RSI组件：安全归一化，处理异常值
                rsi_component = np.where(
                    np.isfinite(data['rsi12']) & (data['rsi12'] > 0),
                    np.clip((data['rsi12'] - 50) / 50, -2, 2),  # 限制在[-2, 2]范围
                    0.0
                )

                # MACD组件：安全处理符号
                macd_component = np.where(
                    np.isfinite(data['macd']),
                    np.sign(data['macd']),
                    0.0
                )

                # 布林带位置组件：安全归一化
                bb_component = np.where(
                    np.isfinite(data['bb_position']),
                    np.clip((data['bb_position'] - 0.5) * 2, -2, 2),  # 限制在[-2, 2]范围
                    0.0
                )

                # 综合计算，确保结果有限
                data['technical_consensus'] = (rsi_component + macd_component + bb_component) / 3

                # 最终安全检查
                data['technical_consensus'] = np.where(
                    np.isfinite(data['technical_consensus']),
                    np.clip(data['technical_consensus'], -1, 1),  # 限制在[-1, 1]范围
                    0.0
                )
            else:
                # 如果缺少必要特征，设为中性值
                data['technical_consensus'] = 0.0

            return data
            
        except Exception as e:
            self.logger.error(f"技术衍生特征计算失败: {e}")
            return data
    
    def _calculate_fundamental_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算基本面衍生特征"""
        try:
            # 1. float_ratio: 流通比例
            if 'circ_mv' in data.columns and 'total_mv' in data.columns:
                data['float_ratio'] = np.where(
                    data['total_mv'] > 0,
                    data['circ_mv'] / data['total_mv'],
                    0.7  # 默认流通比例
                )
                data['float_ratio'] = np.clip(data['float_ratio'], 0, 1)
            else:
                data['float_ratio'] = 0.7
            
            # 2. mv_factor: 市值因子
            if 'total_mv' in data.columns:
                # 对数市值的标准化，用于质量评分
                log_mv = np.log(data['total_mv'] + 1)
                mv_mean = log_mv.mean()
                mv_std = log_mv.std()
                data['mv_factor'] = np.where(
                    mv_std > 0,
                    (log_mv - mv_mean) / mv_std,
                    0
                )
                # 标准化到0-1范围
                data['mv_factor'] = (data['mv_factor'] + 3) / 6  # 假设3个标准差范围
                data['mv_factor'] = np.clip(data['mv_factor'], 0, 1)
            else:
                data['mv_factor'] = 0.5
            
            # 3. capital_intensity: 资金强度
            if 'amount' in data.columns and 'total_mv' in data.columns:
                # 成交额相对市值的比例，反映资金关注度
                data['capital_intensity'] = np.where(
                    data['total_mv'] > 0,
                    data['amount'] / data['total_mv'],
                    0
                )
                # 对数变换平滑极值
                data['capital_intensity'] = np.log(data['capital_intensity'] + 1e-10)
                # 标准化到0-1范围
                intensity_max = data['capital_intensity'].quantile(0.95)
                data['capital_intensity'] = np.clip(data['capital_intensity'] / intensity_max, 0, 1)
            else:
                data['capital_intensity'] = 0.1
            
            return data
            
        except Exception as e:
            self.logger.error(f"基本面衍生特征计算失败: {e}")
            return data
    
    def _calculate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算流动性相关特征"""
        try:
            # amihud_illiquidity: Amihud非流动性指标
            # 参考: Amihud (2002) "Illiquidity and stock returns"
            if 'pct_chg' in data.columns and 'amount' in data.columns:
                # Amihud = |return| / volume (这里用成交额代替成交量)
                data['amihud_illiquidity'] = np.where(
                    data['amount'] > 0,
                    np.abs(data['pct_chg']) / (data['amount'] / 1e8),  # 成交额以亿为单位
                    1.0  # 默认高非流动性
                )
                # 对数变换处理极值
                data['amihud_illiquidity'] = np.log(data['amihud_illiquidity'] + 1e-10)
                # 标准化
                amihud_max = data['amihud_illiquidity'].quantile(0.95)
                data['amihud_illiquidity'] = data['amihud_illiquidity'] / amihud_max
                data['amihud_illiquidity'] = np.clip(data['amihud_illiquidity'], 0, 2)
            else:
                data['amihud_illiquidity'] = 0.5
            
            return data
            
        except Exception as e:
            self.logger.error(f"流动性特征计算失败: {e}")
            return data
    
    def _calculate_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算质量因子特征"""
        try:
            # 这里可以添加更多质量相关的特征
            # 目前主要特征已在其他方法中计算
            
            return data

        except Exception as e:
            self.logger.error(f"质量特征计算失败: {e}")
            return data

    def _calculate_cumulative_return_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算累计涨幅特征"""
        try:
            self.logger.debug("开始计算累计涨幅特征")

            # 按股票分组计算累计涨幅
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy()

                if len(stock_data) < 10:  # 至少需要10个交易日
                    continue

                # 按日期排序
                stock_data = stock_data.sort_values('date')

                # 计算累计涨幅
                if 'pct_chg' in stock_data.columns:
                    pct_chg = stock_data['pct_chg'].fillna(0) / 100  # 转换为小数

                    # 计算不同周期的累计涨幅
                    cumulative_returns = self._calculate_rolling_cumulative_returns(pct_chg)

                    # 将结果赋值回原数据
                    for period, returns in cumulative_returns.items():
                        data.loc[stock_mask, f'cumulative_return_{period}d'] = returns
                else:
                    # 如果没有pct_chg，设置默认值
                    for period in [3, 5, 10]:
                        data.loc[stock_mask, f'cumulative_return_{period}d'] = 0

            self.logger.debug("累计涨幅特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"累计涨幅特征计算失败: {e}")
            return data

    def _calculate_rolling_cumulative_returns(self, pct_chg: pd.Series) -> dict:
        """计算滚动累计涨幅"""
        try:
            cumulative_returns = {}

            # 计算3日、5日、10日累计涨幅
            for period in [3, 5, 10]:
                if len(pct_chg) >= period:
                    # 计算滚动累计收益率
                    rolling_returns = []
                    for i in range(len(pct_chg)):
                        if i < period - 1:
                            rolling_returns.append(0)  # 前几天数据不足，设为0
                        else:
                            # 计算过去period天的累计收益率
                            period_returns = pct_chg.iloc[i-period+1:i+1]
                            cumulative_return = (1 + period_returns).prod() - 1
                            rolling_returns.append(cumulative_return)

                    cumulative_returns[period] = rolling_returns
                else:
                    # 数据不足，全部设为0
                    cumulative_returns[period] = [0] * len(pct_chg)

            return cumulative_returns

        except Exception as e:
            self.logger.error(f"计算滚动累计涨幅失败: {e}")
            return {3: [0] * len(pct_chg), 5: [0] * len(pct_chg), 10: [0] * len(pct_chg)}

    def _calculate_ma_convergence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算均线粘合度指标"""
        try:
            self.logger.debug("开始计算均线粘合度指标")

            # 按股票分组计算
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy()

                if len(stock_data) < 60:  # 至少需要60个交易日
                    continue

                # 按日期排序
                stock_data = stock_data.sort_values('date')

                # 检查必要的均线数据
                ma_columns = ['ma5', 'ma10', 'ma20', 'ma60']
                missing_mas = [col for col in ma_columns if col not in stock_data.columns]
                if missing_mas:
                    self.logger.warning(f"股票 {stock_code} 缺少均线数据: {missing_mas}")
                    continue

                # 计算均线粘合度特征
                ma_features = self._calculate_ma_convergence_for_stock(stock_data)

                # 将结果赋值回原数据
                for feature_name, values in ma_features.items():
                    data.loc[stock_mask, feature_name] = values

            self.logger.debug("均线粘合度指标计算完成")
            return data

        except Exception as e:
            self.logger.error(f"均线粘合度指标计算失败: {e}")
            return data

    def _calculate_ma_convergence_for_stock(self, stock_data: pd.DataFrame) -> dict:
        """为单只股票计算均线粘合度特征"""
        try:
            features = {}

            # 获取价格和均线数据
            close = stock_data['close'].values
            ma5 = stock_data['ma5'].values
            ma10 = stock_data['ma10'].values
            ma20 = stock_data['ma20'].values
            ma60 = stock_data['ma60'].values

            # 1. 均线相对距离（衡量均线粘合程度）
            ma_distances = []
            for i in range(len(close)):
                if pd.isna(ma5[i]) or pd.isna(ma10[i]) or pd.isna(ma20[i]) or pd.isna(ma60[i]):
                    ma_distances.append(0)
                    continue

                # 计算各均线之间的相对距离
                mas = [ma5[i], ma10[i], ma20[i], ma60[i]]
                max_ma = max(mas)
                min_ma = min(mas)

                # 均线粘合度 = 1 - (最大均线 - 最小均线) / 最大均线
                if max_ma > 0:
                    convergence = 1 - (max_ma - min_ma) / max_ma
                else:
                    convergence = 0

                ma_distances.append(convergence)

            features['ma_convergence'] = ma_distances

            # 2. 股价与各均线的距离
            price_ma5_distance = []
            price_ma10_distance = []
            price_ma20_distance = []
            price_ma60_distance = []

            for i in range(len(close)):
                if pd.isna(close[i]):
                    price_ma5_distance.append(0)
                    price_ma10_distance.append(0)
                    price_ma20_distance.append(0)
                    price_ma60_distance.append(0)
                    continue

                # 计算股价与各均线的相对距离
                price_ma5_distance.append((close[i] - ma5[i]) / ma5[i] if not pd.isna(ma5[i]) and ma5[i] > 0 else 0)
                price_ma10_distance.append((close[i] - ma10[i]) / ma10[i] if not pd.isna(ma10[i]) and ma10[i] > 0 else 0)
                price_ma20_distance.append((close[i] - ma20[i]) / ma20[i] if not pd.isna(ma20[i]) and ma20[i] > 0 else 0)
                price_ma60_distance.append((close[i] - ma60[i]) / ma60[i] if not pd.isna(ma60[i]) and ma60[i] > 0 else 0)

            features['price_ma5_distance'] = price_ma5_distance
            features['price_ma10_distance'] = price_ma10_distance
            features['price_ma20_distance'] = price_ma20_distance
            features['price_ma60_distance'] = price_ma60_distance

            # 3. 均线支撑强度（股价在所有均线之上的程度）
            ma_support_strength = []
            for i in range(len(close)):
                if pd.isna(close[i]):
                    ma_support_strength.append(0)
                    continue

                # 计算股价高于多少条均线
                above_count = 0
                total_count = 0

                for ma in [ma5[i], ma10[i], ma20[i], ma60[i]]:
                    if not pd.isna(ma) and ma > 0:
                        total_count += 1
                        if close[i] > ma:
                            above_count += 1

                support_strength = above_count / total_count if total_count > 0 else 0
                ma_support_strength.append(support_strength)

            features['ma_support_strength'] = ma_support_strength

            return features

        except Exception as e:
            self.logger.error(f"计算单股均线粘合度特征失败: {e}")
            return {}

    def _calculate_price_distance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算股价低点距离指标"""
        try:
            self.logger.debug("开始计算股价低点距离指标")

            # 按股票分组计算
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy()

                if len(stock_data) < 60:  # 至少需要60个交易日
                    continue

                # 按日期排序
                stock_data = stock_data.sort_values('date')

                # 检查必要的价格数据
                if 'low' not in stock_data.columns or 'close' not in stock_data.columns:
                    self.logger.warning(f"股票 {stock_code} 缺少价格数据")
                    continue

                # 计算股价低点距离特征
                distance_features = self._calculate_price_distance_for_stock(stock_data)

                # 将结果赋值回原数据
                for feature_name, values in distance_features.items():
                    data.loc[stock_mask, feature_name] = values

            self.logger.debug("股价低点距离指标计算完成")
            return data

        except Exception as e:
            self.logger.error(f"股价低点距离指标计算失败: {e}")
            return data

    def _calculate_price_distance_for_stock(self, stock_data: pd.DataFrame) -> dict:
        """为单只股票计算股价低点距离特征"""
        try:
            features = {}

            # 获取价格数据
            close = stock_data['close'].values
            low = stock_data['low'].values

            # 计算不同周期的低点距离
            periods = [3, 5, 10, 20, 60]

            for period in periods:
                distance_values = []

                for i in range(len(close)):
                    if i < period - 1:
                        # 数据不足，设为0
                        distance_values.append(0)
                        continue

                    # 计算过去period天的最低价
                    period_lows = low[max(0, i-period+1):i+1]

                    # 过滤NaN值
                    valid_lows = period_lows[~pd.isna(period_lows)]

                    if len(valid_lows) == 0 or pd.isna(close[i]):
                        distance_values.append(0)
                        continue

                    min_low = np.min(valid_lows)

                    # 计算当前价格与最低价的距离（涨跌幅）
                    if min_low > 0:
                        distance = (close[i] - min_low) / min_low
                    else:
                        distance = 0

                    distance_values.append(distance)

                features[f'price_low_distance_{period}d'] = distance_values

            # 计算综合低点距离指标
            # 加权平均：近期权重更高
            weights = {3: 0.4, 5: 0.3, 10: 0.2, 20: 0.1}
            weighted_distance = []

            for i in range(len(close)):
                weighted_sum = 0
                weight_sum = 0

                for period, weight in weights.items():
                    if f'price_low_distance_{period}d' in features:
                        value = features[f'price_low_distance_{period}d'][i]
                        weighted_sum += value * weight
                        weight_sum += weight

                if weight_sum > 0:
                    weighted_distance.append(weighted_sum / weight_sum)
                else:
                    weighted_distance.append(0)

            features['price_low_distance_weighted'] = weighted_distance

            return features

        except Exception as e:
            self.logger.error(f"计算单股价格低点距离特征失败: {e}")
            return {}

    def validate_features(self, data: pd.DataFrame, required_features: list) -> Dict[str, bool]:
        """
        验证特征是否存在
        
        Args:
            data: 数据DataFrame
            required_features: 需要的特征列表
            
        Returns:
            特征存在性字典
        """
        validation_result = {}
        for feature in required_features:
            validation_result[feature] = feature in data.columns
        
        missing_features = [f for f, exists in validation_result.items() if not exists]
        if missing_features:
            self.logger.warning(f"缺失特征: {missing_features}")
        
        return validation_result

    def _calculate_limit_up_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算涨停质量特征
        基于A股市场涨停特色，计算涨停相关的质量指标
        """
        try:
            # 按股票代码分组进行时间序列计算
            data_sorted = data.sort_values(['stock_code', 'date'])

            # 1. limit_up_count_60d: 近60个交易日涨停次数
            # 当交易日不足60天时，使用实际可用天数
            if 'is_limit_up' in data.columns:
                data['limit_up_count_60d'] = data_sorted.groupby('stock_code')['is_limit_up'].rolling(
                    window=60, min_periods=1  # min_periods=1确保有数据就计算
                ).sum().reset_index(level=0, drop=True)
            else:
                data['limit_up_count_60d'] = 0

            # 2. limit_up_next_day_positive_rate: 涨停次日红盘率
            if 'is_limit_up' in data.columns and 'pct_chg' in data.columns:
                # 计算次日涨跌幅
                next_day_change = data_sorted.groupby('stock_code')['pct_chg'].shift(-1)
                # 涨停日标记
                limit_up_mask = data['is_limit_up'] == 1

                # 计算涨停次日红盘率（滚动60日窗口）
                limit_up_next_positive = (limit_up_mask & (next_day_change > 0)).astype(int)

                # 添加临时列到原数据
                data['temp_limit_up_next_positive'] = limit_up_next_positive
                data_sorted = data.sort_values(['stock_code', 'date'])

                limit_up_count_rolling = data_sorted.groupby('stock_code')['is_limit_up'].rolling(
                    window=60, min_periods=1
                ).sum().values

                positive_count_rolling = data_sorted.groupby('stock_code')['temp_limit_up_next_positive'].rolling(
                    window=60, min_periods=1
                ).sum().values

                data['limit_up_next_day_positive_rate'] = np.where(
                    limit_up_count_rolling > 0,
                    positive_count_rolling / limit_up_count_rolling,
                    0.5  # 默认50%
                )

                # 清理临时列
                data.drop('temp_limit_up_next_positive', axis=1, inplace=True)
            else:
                data['limit_up_next_day_positive_rate'] = 0.5

            # 3. first_board_seal_rate: 首板涨停封板率
            # 使用is_zhaban字段：首板涨停且未炸板为封板成功
            if 'is_limit_up' in data.columns and 'is_zhaban' in data.columns:
                data_sorted = data.sort_values(['stock_code', 'date'])
                prev_limit_up = data_sorted.groupby('stock_code')['is_limit_up'].shift(1).fillna(0)

                # 首板：当日涨停且前一日未涨停
                first_board_mask = (data['is_limit_up'] == 1) & (prev_limit_up == 0)
                # 封板成功：首板且未炸板
                sealed_mask = first_board_mask & (data['is_zhaban'] == 0)

                data['temp_first_board'] = first_board_mask.astype(int)
                data['temp_sealed'] = sealed_mask.astype(int)
                data_sorted = data.sort_values(['stock_code', 'date'])

                first_board_count = data_sorted.groupby('stock_code')['temp_first_board'].rolling(
                    window=60, min_periods=1
                ).sum().values
                sealed_count = data_sorted.groupby('stock_code')['temp_sealed'].rolling(
                    window=60, min_periods=1
                ).sum().values

                data['first_board_seal_rate'] = np.where(
                    first_board_count > 0,
                    sealed_count / first_board_count,
                    0.8  # 默认80%
                )

                # 清理临时列
                data.drop(['temp_first_board', 'temp_sealed'], axis=1, inplace=True)
            else:
                data['first_board_seal_rate'] = 0.8

            # 4. consecutive_limit_rate: 连板率（首板后次日再涨停概率）
            if 'limit_times' in data.columns and 'is_limit_up' in data.columns:
                first_board_mask = (data['limit_times'] == 1) & (data['is_limit_up'] == 1)
                data_sorted = data.sort_values(['stock_code', 'date'])
                next_day_limit = data_sorted.groupby('stock_code')['is_limit_up'].shift(-1).fillna(0)

                consecutive_mask = first_board_mask & (next_day_limit == 1)

                data['temp_first_board_2'] = first_board_mask.astype(int)
                data['temp_consecutive'] = consecutive_mask.astype(int)
                data_sorted = data.sort_values(['stock_code', 'date'])

                first_board_count = data_sorted.groupby('stock_code')['temp_first_board_2'].rolling(
                    window=60, min_periods=1
                ).sum().values
                consecutive_count = data_sorted.groupby('stock_code')['temp_consecutive'].rolling(
                    window=60, min_periods=1
                ).sum().values

                data['consecutive_limit_rate'] = np.where(
                    first_board_count > 0,
                    consecutive_count / first_board_count,
                    0.3  # 默认30%
                )

                # 清理临时列
                data.drop(['temp_first_board_2', 'temp_consecutive'], axis=1, inplace=True)
            else:
                data['consecutive_limit_rate'] = 0.3

            # 5. limit_up_premium_5pct_count: 涨停次日溢价≥5%的次数
            if 'is_limit_up' in data.columns and 'pct_chg' in data.columns:
                data_sorted = data.sort_values(['stock_code', 'date'])
                next_day_change = data_sorted.groupby('stock_code')['pct_chg'].shift(-1)
                limit_up_mask = data['is_limit_up'] == 1
                premium_5pct_mask = limit_up_mask & (next_day_change >= 5.0)

                data['temp_premium_5pct'] = premium_5pct_mask.astype(int)
                data_sorted = data.sort_values(['stock_code', 'date'])

                data['limit_up_premium_5pct_count'] = data_sorted.groupby('stock_code')['temp_premium_5pct'].rolling(
                    window=60, min_periods=1
                ).sum().values

                # 清理临时列
                data.drop('temp_premium_5pct', axis=1, inplace=True)
            else:
                data['limit_up_premium_5pct_count'] = 0

            # 6. 计算综合涨停质量评分 (0-100分)
            data['limit_up_quality_score'] = self._calculate_limit_up_quality_score(data)

            self.logger.debug("涨停质量特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"涨停质量特征计算失败: {e}")
            return data

    def _calculate_limit_up_quality_score(self, data: pd.DataFrame) -> pd.Series:
        """
        计算综合涨停质量评分 (0-100分)
        简化版本：涨停50%权重，首板封板率40%权重，涨停次日红盘率10%权重
        """
        try:
            # 初始化评分
            quality_score = pd.Series(0.0, index=data.index)

            # 1. 涨停权重 (50%) - 有涨停给最高分
            if 'limit_up_count_60d' in data.columns:
                # 有涨停就给满分，无涨停给0分
                limit_up_score = np.where(data['limit_up_count_60d'] > 0, 50.0, 0.0)
                quality_score += limit_up_score

            # 2. 首板封板率权重 (40%)
            if 'first_board_seal_rate' in data.columns:
                # 封板率直接转换为0-40分
                seal_rate_score = data['first_board_seal_rate'] * 40
                quality_score += seal_rate_score

            # 3. 涨停次日红盘率权重 (10%)
            if 'limit_up_next_day_positive_rate' in data.columns:
                # 红盘率直接转换为0-10分
                positive_rate_score = data['limit_up_next_day_positive_rate'] * 10
                quality_score += positive_rate_score

            # 确保评分在0-100范围内
            quality_score = np.clip(quality_score, 0, 100)

            return quality_score

        except Exception as e:
            self.logger.error(f"涨停质量评分计算失败: {e}")
            return pd.Series(50.0, index=data.index)  # 返回默认评分

    def _calculate_kline_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算K线形态特征
        基于顶尖量化研究的经典看涨形态识别

        参考文献：
        1. Lo et al. (2000) "Foundations of Technical Analysis"
        2. Park & Irwin (2007) "What Do We Know About the Profitability of Technical Analysis?"
        3. Zhu & Zhou (2009) "Technical analysis: An asset allocation perspective"
        """
        try:
            self.logger.debug("开始计算K线形态特征")

            # 按股票分组排序（用于后续处理）

            # 初始化多日形态特征
            data['pattern_golden_spoon'] = 0      # 金勺底
            data['pattern_w_bottom'] = 0          # W底
            data['pattern_n_shape'] = 0           # N字型
            data['pattern_ascending_triangle'] = 0  # 上升三角形
            data['pattern_cup_handle'] = 0        # 杯柄形态
            data['pattern_flag'] = 0              # 旗形整理
            data['pattern_breakout'] = 0          # 突破形态

            # 初始化单日K线形态特征
            data['pattern_bare_head_yang'] = 0    # 光头阳线
            data['pattern_bare_head_yin'] = 0     # 光头阴线
            data['pattern_hammer'] = 0            # 锤子线
            data['pattern_hanging_man'] = 0       # 吊颈线
            data['pattern_doji'] = 0              # 十字星
            data['pattern_morning_star'] = 0      # 启明星
            data['pattern_evening_star'] = 0      # 黄昏星
            data['pattern_engulfing_bull'] = 0    # 看涨吞没
            data['pattern_engulfing_bear'] = 0    # 看跌吞没
            data['pattern_piercing'] = 0          # 刺透形态
            data['pattern_dark_cloud'] = 0        # 乌云盖顶
            data['pattern_long_upper_shadow'] = 0 # 长上影线（新增）

            # 按股票分组计算形态
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy()

                if len(stock_data) < 20:  # 至少需要20个交易日
                    continue

                # 计算各种形态
                patterns = self._identify_patterns_for_stock(stock_data)

                # 将结果赋值回原数据
                for pattern_name, pattern_values in patterns.items():
                    data.loc[stock_mask, pattern_name] = pattern_values

            self.logger.debug("K线形态特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"K线形态计算失败: {e}")
            return data

    def _calculate_ma_crossover_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算MA5和MA10交叉强度特征

        基于量化研究的移动平均线交叉信号强度计算：
        1. 交叉角度：基于均线斜率差异
        2. 交叉持续性：交叉后的维持时间
        3. 价格确认：价格是否配合交叉方向
        4. 成交量确认：成交量是否放大

        参考文献：
        - Brock et al. (1992) "Simple Technical Trading Rules"
        - Lo et al. (2000) "Foundations of Technical Analysis"

        Returns:
            添加了MA交叉特征的数据
        """
        try:
            self.logger.debug("开始计算MA交叉强度特征")

            # 初始化特征列
            data['ma5_above_ma10_strength'] = 0.0
            data['ma5_below_ma10_strength'] = 0.0

            # 按股票分组计算
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy()

                if len(stock_data) < 10:  # 至少需要10个交易日
                    continue

                # 按日期排序
                stock_data = stock_data.sort_values('date')

                # 检查必要的列
                required_cols = ['ma5', 'ma10', 'close', 'volume']
                if not all(col in stock_data.columns for col in required_cols):
                    continue

                # 计算MA交叉强度
                ma5_above_strength, ma5_below_strength = self._calculate_crossover_strength(
                    stock_data['ma5'].values,
                    stock_data['ma10'].values,
                    stock_data['close'].values,
                    stock_data['volume'].values
                )

                # 更新数据
                data.loc[stock_mask, 'ma5_above_ma10_strength'] = ma5_above_strength
                data.loc[stock_mask, 'ma5_below_ma10_strength'] = ma5_below_strength

            self.logger.debug("MA交叉强度特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"计算MA交叉强度特征失败: {e}")
            return data

    def _calculate_crossover_strength(self, ma5: np.ndarray, ma10: np.ndarray,
                                    close: np.ndarray, volume: np.ndarray) -> tuple:
        """
        计算MA5和MA10交叉强度

        基于量化研究的交叉强度算法：
        1. 交叉角度强度：基于均线斜率差异
        2. 持续性强度：交叉后的维持时间
        3. 价格确认强度：价格是否配合交叉方向
        4. 成交量确认强度：成交量是否放大

        Args:
            ma5: MA5数组
            ma10: MA10数组
            close: 收盘价数组
            volume: 成交量数组

        Returns:
            (ma5_above_strength, ma5_below_strength): 两个强度数组
        """
        try:
            n = len(ma5)
            ma5_above_strength = np.zeros(n)
            ma5_below_strength = np.zeros(n)

            if n < 5:
                return ma5_above_strength, ma5_below_strength

            # 计算均线斜率（3日斜率）
            ma5_slope = np.zeros(n)
            ma10_slope = np.zeros(n)

            for i in range(3, n):
                ma5_slope[i] = (ma5[i] - ma5[i-3]) / ma5[i-3] if ma5[i-3] != 0 else 0
                ma10_slope[i] = (ma10[i] - ma10[i-3]) / ma10[i-3] if ma10[i-3] != 0 else 0

            # 计算成交量相对强度（与20日均量比较）
            volume_strength = np.ones(n)
            for i in range(20, n):
                avg_volume = np.mean(volume[i-20:i])
                if avg_volume > 0:
                    volume_strength[i] = min(volume[i] / avg_volume, 3.0)  # 限制在3倍以内

            for i in range(5, n):
                # 1. 计算MA5高于MA10的强度（黄金交叉）
                if ma5[i] > ma10[i]:
                    strength = 0.0

                    # 1.1 交叉角度强度 (0-0.4)
                    slope_diff = ma5_slope[i] - ma10_slope[i]
                    angle_strength = min(max(slope_diff * 10, 0), 0.4)

                    # 1.2 持续性强度 (0-0.3)
                    persistence_days = 0
                    for j in range(max(0, i-10), i+1):
                        if j < n and ma5[j] > ma10[j]:
                            persistence_days += 1
                    persistence_strength = min(persistence_days / 10.0, 0.3)

                    # 1.3 价格确认强度 (0-0.2)
                    price_above_ma5 = 1.0 if close[i] > ma5[i] else 0.0
                    price_strength = price_above_ma5 * 0.2

                    # 1.4 成交量确认强度 (0-0.1)
                    vol_strength = min((volume_strength[i] - 1.0) / 2.0, 0.1)

                    strength = angle_strength + persistence_strength + price_strength + vol_strength
                    ma5_above_strength[i] = min(strength, 1.0)

                # 2. 计算MA5低于MA10的强度（死亡交叉）
                if ma5[i] < ma10[i]:
                    strength = 0.0

                    # 2.1 交叉角度强度 (0-0.4)
                    slope_diff = ma10_slope[i] - ma5_slope[i]  # 注意顺序相反
                    angle_strength = min(max(slope_diff * 10, 0), 0.4)

                    # 2.2 持续性强度 (0-0.3)
                    persistence_days = 0
                    for j in range(max(0, i-10), i+1):
                        if j < n and ma5[j] < ma10[j]:
                            persistence_days += 1
                    persistence_strength = min(persistence_days / 10.0, 0.3)

                    # 2.3 价格确认强度 (0-0.2)
                    price_below_ma5 = 1.0 if close[i] < ma5[i] else 0.0
                    price_strength = price_below_ma5 * 0.2

                    # 2.4 成交量确认强度 (0-0.1)
                    vol_strength = min((volume_strength[i] - 1.0) / 2.0, 0.1)

                    strength = angle_strength + persistence_strength + price_strength + vol_strength
                    ma5_below_strength[i] = min(strength, 1.0)

            return ma5_above_strength, ma5_below_strength

        except Exception as e:
            self.logger.error(f"计算交叉强度失败: {e}")
            return np.zeros(len(ma5)), np.zeros(len(ma5))

    def _identify_patterns_for_stock(self, stock_data: pd.DataFrame) -> dict:
        """
        为单只股票识别各种K线形态
        """
        patterns = {
            # 多日形态
            'pattern_golden_spoon': [0] * len(stock_data),
            'pattern_w_bottom': [0] * len(stock_data),
            'pattern_n_shape': [0] * len(stock_data),
            'pattern_ascending_triangle': [0] * len(stock_data),
            'pattern_cup_handle': [0] * len(stock_data),
            'pattern_flag': [0] * len(stock_data),
            'pattern_breakout': [0] * len(stock_data),
            # 单日形态
            'pattern_bare_head_yang': [0] * len(stock_data),
            'pattern_bare_head_yin': [0] * len(stock_data),
            'pattern_hammer': [0] * len(stock_data),
            'pattern_hanging_man': [0] * len(stock_data),
            'pattern_doji': [0] * len(stock_data),
            'pattern_morning_star': [0] * len(stock_data),
            'pattern_evening_star': [0] * len(stock_data),
            'pattern_engulfing_bull': [0] * len(stock_data),
            'pattern_engulfing_bear': [0] * len(stock_data),
            'pattern_piercing': [0] * len(stock_data),
            'pattern_dark_cloud': [0] * len(stock_data)
        }

        try:
            # 计算技术指标
            opens = stock_data['open'].values
            closes = stock_data['close'].values
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            volumes = stock_data.get('volume', pd.Series([1] * len(stock_data))).values

            # 计算移动平均线
            ma5 = self._calculate_ma(closes, 5)
            ma10 = self._calculate_ma(closes, 10)
            ma20 = self._calculate_ma(closes, 20)

            # 1. 金勺底形态 (Golden Spoon Bottom)
            # 特征：价格跌破MA20后快速拉回，形成勺子形状
            patterns['pattern_golden_spoon'] = self._identify_golden_spoon(
                closes, lows, ma20, volumes
            )

            # 2. W底形态 (Double Bottom)
            # 特征：两个相近的低点，中间有一个高点
            patterns['pattern_w_bottom'] = self._identify_w_bottom(
                closes, lows, highs
            )

            # 3. N字型形态 (N-Shape)
            # 特征：上涨-回调-再上涨，形成N字
            patterns['pattern_n_shape'] = self._identify_n_shape(
                closes, highs, lows
            )

            # 4. 上升三角形 (Ascending Triangle)
            # 特征：高点平行，低点逐步抬高
            patterns['pattern_ascending_triangle'] = self._identify_ascending_triangle(
                closes, highs, lows
            )

            # 5. 杯柄形态 (Cup and Handle)
            # 特征：U型底部 + 小幅回调
            patterns['pattern_cup_handle'] = self._identify_cup_handle(
                closes, highs, lows, volumes
            )

            # 6. 旗形整理 (Flag Pattern)
            # 特征：急涨后小幅整理
            patterns['pattern_flag'] = self._identify_flag_pattern(
                closes, highs, lows, volumes
            )

            # 7. 突破形态 (Breakout Pattern)
            # 特征：突破重要阻力位
            patterns['pattern_breakout'] = self._identify_breakout_pattern(
                closes, highs, ma5, ma10, ma20, volumes
            )

            # === 单日K线形态识别 ===

            # 8. 光头阳线 (Bare Head Yang Line)
            patterns['pattern_bare_head_yang'] = self._identify_bare_head_yang(
                opens, closes, highs, lows
            )

            # 9. 光头阴线 (Bare Head Yin Line)
            patterns['pattern_bare_head_yin'] = self._identify_bare_head_yin(
                opens, closes, highs, lows
            )

            # 10. 锤子线 (Hammer)
            patterns['pattern_hammer'] = self._identify_hammer(
                opens, closes, highs, lows
            )

            # 11. 吊颈线 (Hanging Man)
            patterns['pattern_hanging_man'] = self._identify_hanging_man(
                opens, closes, highs, lows
            )

            # 12. 十字星 (Doji)
            patterns['pattern_doji'] = self._identify_doji(
                opens, closes, highs, lows
            )

            # 13. 启明星 (Morning Star) - 需要3日数据
            patterns['pattern_morning_star'] = self._identify_morning_star(
                opens, closes, highs, lows
            )

            # 14. 黄昏星 (Evening Star) - 需要3日数据
            patterns['pattern_evening_star'] = self._identify_evening_star(
                opens, closes, highs, lows
            )

            # 15. 看涨吞没 (Bullish Engulfing) - 需要2日数据
            patterns['pattern_engulfing_bull'] = self._identify_engulfing_bull(
                opens, closes, highs, lows
            )

            # 16. 看跌吞没 (Bearish Engulfing) - 需要2日数据
            patterns['pattern_engulfing_bear'] = self._identify_engulfing_bear(
                opens, closes, highs, lows
            )

            # 17. 刺透形态 (Piercing Pattern) - 需要2日数据
            patterns['pattern_piercing'] = self._identify_piercing(
                opens, closes, highs, lows
            )

            # 18. 乌云盖顶 (Dark Cloud Cover) - 需要2日数据
            patterns['pattern_dark_cloud'] = self._identify_dark_cloud(
                opens, closes, highs, lows
            )

            # 19. 长上影线 (Long Upper Shadow) - 单日形态
            patterns['pattern_long_upper_shadow'] = self._identify_long_upper_shadow(
                opens, closes, highs, lows
            )

        except Exception as e:
            self.logger.error(f"股票形态识别失败: {e}")

        return patterns

    def _calculate_ma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """计算移动平均线"""
        ma = np.full_like(prices, np.nan)
        for i in range(period - 1, len(prices)):
            ma[i] = np.mean(prices[i - period + 1:i + 1])
        return ma

    def _identify_golden_spoon(self, closes: np.ndarray, lows: np.ndarray,
                              ma20: np.ndarray, volumes: np.ndarray) -> list:
        """
        识别金勺底形态
        特征：
        1. 价格跌破MA20
        2. 快速反弹回MA20上方
        3. 成交量放大确认
        """
        pattern = [0] * len(closes)

        for i in range(20, len(closes) - 2):
            if np.isnan(ma20[i]):
                continue

            # 检查过去10天是否有跌破MA20
            broke_ma20 = False
            break_day = -1
            for j in range(max(0, i - 10), i):
                if lows[j] < ma20[j] * 0.98:  # 跌破MA20的2%
                    broke_ma20 = True
                    break_day = j
                    break

            if broke_ma20 and break_day >= 0:
                # 检查是否快速反弹
                days_since_break = i - break_day
                if days_since_break <= 5:  # 5天内反弹
                    # 当前价格回到MA20上方
                    if closes[i] > ma20[i] * 1.01:  # 超过MA20的1%
                        # 成交量确认（相对前期放大）
                        avg_volume = np.mean(volumes[max(0, i-10):i])
                        if volumes[i] > avg_volume * 1.2:  # 成交量放大20%
                            pattern[i] = 1

        return pattern

    def _identify_w_bottom(self, closes: np.ndarray, lows: np.ndarray,
                          highs: np.ndarray) -> list:
        """
        识别W底形态
        特征：
        1. 两个相近的低点
        2. 中间有一个明显的高点
        3. 第二个低点后开始上涨
        """
        pattern = [0] * len(closes)

        for i in range(20, len(closes) - 5):
            # 寻找第一个低点（过去15天的最低点）
            start_idx = max(0, i - 15)
            first_low_idx = start_idx + np.argmin(lows[start_idx:i])
            first_low = lows[first_low_idx]

            # 寻找中间高点（第一个低点后的最高点）
            if i - first_low_idx < 5:
                continue
            mid_high_idx = first_low_idx + np.argmax(highs[first_low_idx:i])
            mid_high = highs[mid_high_idx]

            # 寻找第二个低点（当前位置附近）
            second_low_start = max(mid_high_idx + 2, i - 5)
            if second_low_start >= i:
                continue
            second_low_idx = second_low_start + np.argmin(lows[second_low_start:i+1])
            second_low = lows[second_low_idx]

            # W底条件检查
            if (abs(second_low - first_low) / first_low < 0.05 and  # 两个低点相近（5%以内）
                mid_high > first_low * 1.03 and  # 中间高点明显高于低点
                mid_high > second_low * 1.03 and
                i == second_low_idx and  # 当前就是第二个低点
                len(closes) > i + 2 and closes[i + 1] > closes[i]):  # 开始反弹
                pattern[i] = 1

        return pattern

    def _identify_n_shape(self, closes: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray) -> list:
        """
        识别N字型形态
        特征：上涨-回调-再上涨，形成N字
        """
        pattern = [0] * len(closes)

        for i in range(15, len(closes) - 3):
            # 寻找第一段上涨的起点和终点
            start_idx = max(0, i - 15)

            # 第一段上涨：寻找低点到高点
            first_low_idx = start_idx + np.argmin(lows[start_idx:i-8])
            first_high_idx = first_low_idx + 3 + np.argmax(highs[first_low_idx+3:i-5])

            if first_high_idx <= first_low_idx + 3:
                continue

            # 回调低点
            pullback_low_idx = first_high_idx + 1 + np.argmin(lows[first_high_idx+1:i-2])

            if pullback_low_idx <= first_high_idx + 1:
                continue

            # N字型条件检查
            first_rise = (highs[first_high_idx] - lows[first_low_idx]) / lows[first_low_idx]
            pullback = (highs[first_high_idx] - lows[pullback_low_idx]) / highs[first_high_idx]
            current_rise = (closes[i] - lows[pullback_low_idx]) / lows[pullback_low_idx]

            if (first_rise > 0.08 and  # 第一段上涨超过8%
                pullback > 0.03 and pullback < 0.15 and  # 回调3%-15%
                current_rise > 0.05 and  # 第二段上涨超过5%
                closes[i] > highs[first_high_idx] * 0.98):  # 接近或超过前高
                pattern[i] = 1

        return pattern

    def _identify_ascending_triangle(self, closes: np.ndarray, highs: np.ndarray,
                                   lows: np.ndarray) -> list:
        """
        识别上升三角形形态
        特征：高点平行，低点逐步抬高
        """
        pattern = [0] * len(closes)

        for i in range(20, len(closes) - 2):
            # 寻找过去20天的高点和低点
            window_highs = highs[i-20:i+1]
            window_lows = lows[i-20:i+1]

            # 找到最近的几个高点
            high_indices = []
            for j in range(2, len(window_highs) - 2):
                if (window_highs[j] > window_highs[j-1] and
                    window_highs[j] > window_highs[j-2] and
                    window_highs[j] > window_highs[j+1] and
                    window_highs[j] > window_highs[j+2]):
                    high_indices.append(j)

            # 找到最近的几个低点
            low_indices = []
            for j in range(2, len(window_lows) - 2):
                if (window_lows[j] < window_lows[j-1] and
                    window_lows[j] < window_lows[j-2] and
                    window_lows[j] < window_lows[j+1] and
                    window_lows[j] < window_lows[j+2]):
                    low_indices.append(j)

            # 检查上升三角形条件
            if len(high_indices) >= 2 and len(low_indices) >= 2:
                # 高点相对平行（变化小于3%）
                recent_highs = [window_highs[idx] for idx in high_indices[-2:]]
                high_variation = abs(recent_highs[1] - recent_highs[0]) / recent_highs[0]

                # 低点逐步抬高
                recent_lows = [window_lows[idx] for idx in low_indices[-2:]]
                low_rising = recent_lows[1] > recent_lows[0] * 1.02  # 低点抬高2%

                if high_variation < 0.03 and low_rising and closes[i] > recent_highs[0] * 0.99:
                    pattern[i] = 1

        return pattern

    def _identify_cup_handle(self, closes: np.ndarray, highs: np.ndarray,
                           lows: np.ndarray, volumes: np.ndarray) -> list:
        """
        识别杯柄形态
        特征：U型底部 + 小幅回调（手柄）
        """
        pattern = [0] * len(closes)

        for i in range(30, len(closes) - 5):
            # 寻找杯子的左边缘（高点）
            left_high_idx = max(0, i - 30) + np.argmax(highs[max(0, i-30):i-20])
            left_high = highs[left_high_idx]

            # 寻找杯底（最低点）
            cup_bottom_idx = left_high_idx + np.argmin(lows[left_high_idx:i-10])
            cup_bottom = lows[cup_bottom_idx]

            # 寻找杯子右边缘（回到接近左边缘的高度）
            right_high_idx = cup_bottom_idx + np.argmax(highs[cup_bottom_idx:i-5])
            right_high = highs[right_high_idx]

            # 当前位置应该是手柄的回调
            handle_low = min(lows[right_high_idx:i+1])

            # 杯柄形态条件
            cup_depth = (left_high - cup_bottom) / left_high
            right_height = right_high / left_high
            handle_depth = (right_high - handle_low) / right_high

            if (cup_depth > 0.12 and cup_depth < 0.35 and  # 杯子深度12%-35%
                right_height > 0.95 and  # 右边缘接近左边缘
                handle_depth > 0.08 and handle_depth < 0.20 and  # 手柄回调8%-20%
                closes[i] > right_high * 0.98):  # 接近突破
                pattern[i] = 1

        return pattern

    def _identify_flag_pattern(self, closes: np.ndarray, highs: np.ndarray,
                             lows: np.ndarray, volumes: np.ndarray) -> list:
        """
        识别旗形整理形态
        特征：急涨后小幅整理
        """
        pattern = [0] * len(closes)

        for i in range(15, len(closes) - 3):
            # 检查前期是否有急涨
            lookback = min(10, i)
            price_rise = (closes[i-1] - closes[i-lookback]) / closes[i-lookback]

            if price_rise < 0.15:  # 前期涨幅至少15%
                continue

            # 检查最近几天是否在整理（小幅波动）
            recent_high = max(highs[i-5:i+1])
            recent_low = min(lows[i-5:i+1])
            consolidation_range = (recent_high - recent_low) / recent_low

            # 检查成交量是否萎缩
            pre_surge_volume = np.mean(volumes[i-lookback:i-5])
            recent_volume = np.mean(volumes[i-5:i+1])
            volume_ratio = recent_volume / pre_surge_volume if pre_surge_volume > 0 else 1

            if (consolidation_range < 0.08 and  # 整理幅度小于8%
                volume_ratio < 0.8 and  # 成交量萎缩
                closes[i] > recent_low * 1.02):  # 价格在整理区间上方
                pattern[i] = 1

        return pattern

    def _identify_breakout_pattern(self, closes: np.ndarray, highs: np.ndarray,
                                 ma5: np.ndarray, ma10: np.ndarray, ma20: np.ndarray,
                                 volumes: np.ndarray) -> list:
        """
        识别突破形态
        特征：突破重要阻力位，均线多头排列
        """
        pattern = [0] * len(closes)

        for i in range(20, len(closes) - 1):
            if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]):
                continue

            # 检查均线多头排列
            ma_bullish = ma5[i] > ma10[i] > ma20[i]

            # 检查是否突破前期高点
            lookback_period = min(20, i)
            resistance_level = max(highs[i-lookback_period:i])
            breakout = closes[i] > resistance_level * 1.01  # 突破前高1%

            # 检查成交量放大
            avg_volume = np.mean(volumes[max(0, i-10):i])
            volume_surge = volumes[i] > avg_volume * 1.5  # 成交量放大50%

            # 检查价格强度
            price_strength = closes[i] > ma5[i] * 1.02  # 价格明显高于MA5

            if ma_bullish and breakout and volume_surge and price_strength:
                pattern[i] = 1

        return pattern

    # === 单日K线形态识别方法 ===

    def _identify_bare_head_yang(self, opens: np.ndarray, closes: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别光头阳线（强度评分版本）
        特征：
        1. 收盘价 > 开盘价 (阳线)
        2. 收盘价 = 最高价 (无上影线)
        3. 实体较长，根据涨幅程度给出强度评分

        返回值：0-1之间的强度评分，接近涨停的给予更高分数
        """
        pattern = [0.0] * len(closes)

        for i in range(len(closes)):
            if closes[i] > opens[i]:  # 阳线
                # 计算涨幅百分比
                rise_pct = (closes[i] - opens[i]) / opens[i]

                # 检查是否无上影线（允许微小误差）
                upper_shadow_ratio = (highs[i] - closes[i]) / opens[i]
                has_no_upper_shadow = upper_shadow_ratio < 0.002  # 0.2%以内认为无上影线

                # 检查实体是否足够长
                has_significant_body = rise_pct > 0.01  # 至少1%涨幅

                if has_no_upper_shadow and has_significant_body:
                    # 根据涨幅程度计算强度评分
                    if rise_pct >= 0.095:  # 接近涨停（9.5%以上）
                        strength = 1.0  # 最高强度
                    elif rise_pct >= 0.07:  # 大涨（7%以上）
                        strength = 0.8
                    elif rise_pct >= 0.05:  # 中等涨幅（5%以上）
                        strength = 0.6
                    elif rise_pct >= 0.03:  # 小涨（3%以上）
                        strength = 0.4
                    else:  # 轻微上涨（1-3%）
                        strength = 0.2

                    pattern[i] = strength

        return pattern

    def _identify_bare_head_yin(self, opens: np.ndarray, closes: np.ndarray,
                               highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别光头阴线（恢复原始0-1强度评分，让模型自主学习）
        特征：
        1. 收盘价 < 开盘价 (阴线)
        2. 开盘价 = 最高价 (无上影线)
        3. 实体较长，根据跌幅程度给出强度评分

        返回值：0-1之间的强度评分，让模型通过历史数据学习这是风险信号
        """
        pattern = [0.0] * len(closes)

        for i in range(len(closes)):
            if closes[i] < opens[i]:  # 阴线
                # 计算跌幅百分比
                decline_pct = (opens[i] - closes[i]) / opens[i]

                # 检查是否无上影线（允许微小误差）
                upper_shadow_ratio = (highs[i] - opens[i]) / opens[i]
                has_no_upper_shadow = upper_shadow_ratio < 0.002  # 0.2%以内认为无上影线

                # 检查实体是否足够长
                has_significant_body = decline_pct > 0.01  # 至少1%跌幅

                if has_no_upper_shadow and has_significant_body:
                    # 根据跌幅程度计算强度评分（让模型学习强度与风险的关系）
                    if decline_pct >= 0.095:  # 接近跌停（9.5%以上）
                        strength = 1.0  # 最高强度（模型需学习这意味着高风险）
                    elif decline_pct >= 0.07:  # 大跌（7%以上）
                        strength = 0.8
                    elif decline_pct >= 0.05:  # 中等跌幅（5%以上）
                        strength = 0.6
                    elif decline_pct >= 0.03:  # 小跌（3%以上）
                        strength = 0.4
                    else:  # 轻微下跌（1-3%）
                        strength = 0.2

                    pattern[i] = strength

        return pattern

    def _identify_hammer(self, opens: np.ndarray, closes: np.ndarray,
                        highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别锤子线
        特征：
        1. 实体较小
        2. 下影线长度 >= 2倍实体长度
        3. 上影线很短或没有
        4. 出现在下跌趋势中
        """
        pattern = [0] * len(closes)

        for i in range(1, len(closes)):  # 从第2天开始，需要判断趋势
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]

            # 判断是否在下跌趋势中
            is_downtrend = closes[i-1] < closes[max(0, i-5):i].mean() if i >= 5 else closes[i-1] < opens[i-1]

            if (body_size > 0 and  # 有实体
                lower_shadow >= 2 * body_size and  # 下影线长
                upper_shadow <= 0.5 * body_size and  # 上影线短
                is_downtrend):  # 在下跌趋势中
                pattern[i] = 1

        return pattern

    def _identify_hanging_man(self, opens: np.ndarray, closes: np.ndarray,
                             highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别吊颈线（负值强度评分版本）
        特征：
        1. 实体较小
        2. 下影线长度 >= 2倍实体长度
        3. 上影线很短或没有
        4. 出现在上涨趋势中

        返回值：-1到0之间的强度评分，吊颈线是看跌信号
        """
        pattern = [0.0] * len(closes)

        for i in range(1, len(closes)):  # 从第2天开始，需要判断趋势
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            total_range = highs[i] - lows[i]

            if total_range == 0:
                continue

            # 判断是否在上涨趋势中
            is_uptrend = closes[i-1] > closes[max(0, i-5):i].mean() if i >= 5 else closes[i-1] > opens[i-1]

            if (body_size > 0 and  # 有实体
                lower_shadow >= 2 * body_size and  # 下影线长
                upper_shadow <= 0.5 * body_size and  # 上影线短
                is_uptrend):  # 在上涨趋势中

                # 根据下影线长度和实体大小计算负值强度
                shadow_ratio = lower_shadow / total_range
                body_ratio = body_size / total_range

                if shadow_ratio >= 0.7 and body_ratio <= 0.1:  # 极长下影线，极小实体
                    strength = -0.8  # 强烈看跌信号
                elif shadow_ratio >= 0.6 and body_ratio <= 0.2:  # 长下影线，小实体
                    strength = -0.6
                elif shadow_ratio >= 0.5 and body_ratio <= 0.3:  # 中等下影线
                    strength = -0.4
                else:
                    strength = -0.2

                pattern[i] = strength

        return pattern

    def _identify_doji(self, opens: np.ndarray, closes: np.ndarray,
                      highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别十字星
        特征：
        1. 开盘价 ≈ 收盘价 (实体很小)
        2. 上下影线相对较长
        """
        pattern = [0] * len(closes)

        for i in range(len(closes)):
            body_size = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]

            if (total_range > 0 and
                body_size <= 0.001 * opens[i] and  # 实体很小
                total_range >= 0.01 * opens[i]):  # 总体波动范围足够
                pattern[i] = 1

        return pattern

    def _identify_morning_star(self, opens: np.ndarray, closes: np.ndarray,
                              highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别启明星 (3日形态)
        特征：
        1. 第1日：长阴线
        2. 第2日：小实体（十字星或小阳/阴线），向下跳空
        3. 第3日：长阳线，向上跳空，收盘价进入第1日实体
        """
        pattern = [0] * len(closes)

        for i in range(2, len(closes)):  # 需要3日数据
            # 第1日：长阴线
            day1_body = opens[i-2] - closes[i-2]
            day1_is_bear = closes[i-2] < opens[i-2]
            day1_is_long = day1_body > 0.015 * opens[i-2]

            # 第2日：小实体，向下跳空
            day2_body = abs(closes[i-1] - opens[i-1])
            day2_is_small = day2_body < 0.005 * opens[i-1]
            day2_gap_down = highs[i-1] < lows[i-2]

            # 第3日：长阳线，向上跳空
            day3_body = closes[i] - opens[i]
            day3_is_bull = closes[i] > opens[i]
            day3_is_long = day3_body > 0.015 * opens[i]
            day3_gap_up = lows[i] > highs[i-1]
            day3_penetrate = closes[i] > (opens[i-2] + closes[i-2]) / 2

            if (day1_is_bear and day1_is_long and
                day2_is_small and day2_gap_down and
                day3_is_bull and day3_is_long and day3_gap_up and day3_penetrate):
                pattern[i] = 1

        return pattern

    def _identify_evening_star(self, opens: np.ndarray, closes: np.ndarray,
                              highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别黄昏星 (3日形态，负值强度评分版本)
        特征：
        1. 第1日：长阳线
        2. 第2日：小实体，向上跳空
        3. 第3日：长阴线，向下跳空，收盘价进入第1日实体

        返回值：-1到0之间的强度评分，黄昏星是强烈的看跌信号
        """
        pattern = [0.0] * len(closes)

        for i in range(2, len(closes)):  # 需要3日数据
            # 第1日：长阳线
            day1_body = closes[i-2] - opens[i-2]
            day1_is_bull = closes[i-2] > opens[i-2]
            day1_is_long = day1_body > 0.015 * opens[i-2]

            # 第2日：小实体，向上跳空
            day2_body = abs(closes[i-1] - opens[i-1])
            day2_is_small = day2_body < 0.005 * opens[i-1]
            day2_gap_up = lows[i-1] > highs[i-2]

            # 第3日：长阴线，向下跳空
            day3_body = opens[i] - closes[i]
            day3_is_bear = closes[i] < opens[i]
            day3_is_long = day3_body > 0.015 * opens[i]
            day3_gap_down = highs[i] < lows[i-1]
            day3_penetrate = closes[i] < (opens[i-2] + closes[i-2]) / 2

            if (day1_is_bull and day1_is_long and
                day2_is_small and day2_gap_up and
                day3_is_bear and day3_is_long and day3_gap_down and day3_penetrate):

                # 根据第3日跌幅和穿透程度计算负值强度
                day3_decline = day3_body / opens[i]
                penetration_depth = (opens[i-2] + closes[i-2]) / 2 - closes[i]
                penetration_ratio = penetration_depth / day1_body if day1_body > 0 else 0

                if day3_decline >= 0.08 and penetration_ratio >= 0.7:  # 大跌且深度穿透
                    strength = -0.9  # 强烈看跌信号
                elif day3_decline >= 0.06 and penetration_ratio >= 0.5:  # 中等跌幅
                    strength = -0.7
                elif day3_decline >= 0.04 and penetration_ratio >= 0.3:  # 轻微跌幅
                    strength = -0.5
                else:
                    strength = -0.3

                pattern[i] = strength

        return pattern

    def _identify_engulfing_bull(self, opens: np.ndarray, closes: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别看涨吞没 (2日形态)
        特征：
        1. 第1日：阴线
        2. 第2日：阳线，完全吞没第1日实体
        """
        pattern = [0] * len(closes)

        for i in range(1, len(closes)):  # 需要2日数据
            # 第1日：阴线
            day1_is_bear = closes[i-1] < opens[i-1]

            # 第2日：阳线，完全吞没
            day2_is_bull = closes[i] > opens[i]
            day2_engulfs = (opens[i] < closes[i-1] and closes[i] > opens[i-1])

            if day1_is_bear and day2_is_bull and day2_engulfs:
                pattern[i] = 1

        return pattern

    def _identify_engulfing_bear(self, opens: np.ndarray, closes: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别看跌吞没 (2日形态，负值强度评分版本)
        特征：
        1. 第1日：阳线
        2. 第2日：阴线，完全吞没第1日实体

        返回值：-1到0之间的强度评分，看跌吞没是强烈的看跌信号
        """
        pattern = [0.0] * len(closes)

        for i in range(1, len(closes)):  # 需要2日数据
            # 第1日：阳线
            day1_is_bull = closes[i-1] > opens[i-1]
            day1_body = closes[i-1] - opens[i-1]

            # 第2日：阴线，完全吞没
            day2_is_bear = closes[i] < opens[i]
            day2_body = opens[i] - closes[i]
            day2_engulfs = (opens[i] > closes[i-1] and closes[i] < opens[i-1])

            if day1_is_bull and day2_is_bear and day2_engulfs and day1_body > 0:
                # 根据吞没程度和第2日跌幅计算负值强度
                engulf_ratio = day2_body / day1_body  # 吞没比例
                day2_decline = day2_body / opens[i]   # 第2日跌幅

                if engulf_ratio >= 2.0 and day2_decline >= 0.06:  # 大幅吞没且大跌
                    strength = -0.9  # 强烈看跌信号
                elif engulf_ratio >= 1.5 and day2_decline >= 0.04:  # 中等吞没
                    strength = -0.7
                elif engulf_ratio >= 1.2 and day2_decline >= 0.02:  # 轻微吞没
                    strength = -0.5
                else:
                    strength = -0.3

                pattern[i] = strength

        return pattern

    def _identify_piercing(self, opens: np.ndarray, closes: np.ndarray,
                          highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别刺透形态 (2日形态)
        特征：
        1. 第1日：长阴线
        2. 第2日：阳线，开盘价低于第1日最低价，收盘价进入第1日实体上半部
        """
        pattern = [0] * len(closes)

        for i in range(1, len(closes)):  # 需要2日数据
            # 第1日：长阴线
            day1_body = opens[i-1] - closes[i-1]
            day1_is_bear = closes[i-1] < opens[i-1]
            day1_is_long = day1_body > 0.01 * opens[i-1]

            # 第2日：阳线，刺透条件
            day2_is_bull = closes[i] > opens[i]
            day2_gap_down = opens[i] < lows[i-1]
            day2_penetrate = closes[i] > (opens[i-1] + closes[i-1]) / 2
            day2_not_engulf = closes[i] < opens[i-1]

            if (day1_is_bear and day1_is_long and
                day2_is_bull and day2_gap_down and day2_penetrate and day2_not_engulf):
                pattern[i] = 1

        return pattern

    def _identify_dark_cloud(self, opens: np.ndarray, closes: np.ndarray,
                            highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别乌云盖顶 (2日形态，负值强度评分版本)
        特征：
        1. 第1日：长阳线
        2. 第2日：阴线，开盘价高于第1日最高价，收盘价进入第1日实体下半部

        返回值：-1到0之间的强度评分，乌云盖顶是强烈的看跌信号
        """
        pattern = [0.0] * len(closes)

        for i in range(1, len(closes)):  # 需要2日数据
            # 第1日：长阳线
            day1_body = closes[i-1] - opens[i-1]
            day1_is_bull = closes[i-1] > opens[i-1]
            day1_is_long = day1_body > 0.01 * opens[i-1]

            # 第2日：阴线，乌云盖顶条件
            day2_body = opens[i] - closes[i]
            day2_is_bear = closes[i] < opens[i]
            day2_gap_up = opens[i] > highs[i-1]
            day2_penetrate = closes[i] < (opens[i-1] + closes[i-1]) / 2
            day2_not_engulf = closes[i] > opens[i-1]

            if (day1_is_bull and day1_is_long and
                day2_is_bear and day2_gap_up and day2_penetrate and day2_not_engulf and day1_body > 0):

                # 根据穿透深度和第2日跌幅计算负值强度
                penetration_depth = (opens[i-1] + closes[i-1]) / 2 - closes[i]
                penetration_ratio = penetration_depth / day1_body
                day2_decline = day2_body / opens[i]
                gap_size = (opens[i] - highs[i-1]) / highs[i-1]

                if penetration_ratio >= 0.7 and day2_decline >= 0.06 and gap_size >= 0.02:  # 深度穿透且大跌
                    strength = -0.9  # 强烈看跌信号
                elif penetration_ratio >= 0.5 and day2_decline >= 0.04:  # 中等穿透
                    strength = -0.7
                elif penetration_ratio >= 0.3 and day2_decline >= 0.02:  # 轻微穿透
                    strength = -0.5
                else:
                    strength = -0.3

                pattern[i] = strength

        return pattern

    def _identify_morning_star(self, opens: np.ndarray, closes: np.ndarray,
                              highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别启明星 (3日形态)
        特征：
        1. 第1日：长阴线
        2. 第2日：小实体（十字星或小阳/阴线），向下跳空
        3. 第3日：长阳线，向上跳空，收盘价进入第1日实体
        """
        pattern = [0] * len(closes)

        for i in range(2, len(closes)):  # 需要3日数据
            # 第1日：长阴线
            day1_body = opens[i-2] - closes[i-2]
            day1_is_bear = closes[i-2] < opens[i-2]
            day1_is_long = day1_body > 0.015 * opens[i-2]

            # 第2日：小实体，向下跳空
            day2_body = abs(closes[i-1] - opens[i-1])
            day2_is_small = day2_body < 0.005 * opens[i-1]
            day2_gap_down = highs[i-1] < lows[i-2]

            # 第3日：长阳线，向上跳空
            day3_body = closes[i] - opens[i]
            day3_is_bull = closes[i] > opens[i]
            day3_is_long = day3_body > 0.015 * opens[i]
            day3_gap_up = lows[i] > highs[i-1]
            day3_penetrate = closes[i] > (opens[i-2] + closes[i-2]) / 2

            if (day1_is_bear and day1_is_long and
                day2_is_small and day2_gap_down and
                day3_is_bull and day3_is_long and day3_gap_up and day3_penetrate):
                pattern[i] = 1

        return pattern

    def _identify_evening_star(self, opens: np.ndarray, closes: np.ndarray,
                              highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别黄昏星 (3日形态)
        特征：
        1. 第1日：长阳线
        2. 第2日：小实体，向上跳空
        3. 第3日：长阴线，向下跳空，收盘价进入第1日实体
        """
        pattern = [0] * len(closes)

        for i in range(2, len(closes)):  # 需要3日数据
            # 第1日：长阳线
            day1_body = closes[i-2] - opens[i-2]
            day1_is_bull = closes[i-2] > opens[i-2]
            day1_is_long = day1_body > 0.015 * opens[i-2]

            # 第2日：小实体，向上跳空
            day2_body = abs(closes[i-1] - opens[i-1])
            day2_is_small = day2_body < 0.005 * opens[i-1]
            day2_gap_up = lows[i-1] > highs[i-2]

            # 第3日：长阴线，向下跳空
            day3_body = opens[i] - closes[i]
            day3_is_bear = closes[i] < opens[i]
            day3_is_long = day3_body > 0.015 * opens[i]
            day3_gap_down = highs[i] < lows[i-1]
            day3_penetrate = closes[i] < (opens[i-2] + closes[i-2]) / 2

            if (day1_is_bull and day1_is_long and
                day2_is_small and day2_gap_up and
                day3_is_bear and day3_is_long and day3_gap_down and day3_penetrate):
                pattern[i] = 1

        return pattern

    def _identify_long_upper_shadow(self, opens: np.ndarray, closes: np.ndarray,
                                   highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别长上影线（负值强度评分版本）
        特征：
        1. 上影线长度 >= 2倍实体长度
        2. 上影线长度 >= 总体波动的50%
        3. 通常出现在高位，是看跌信号

        返回值：-1到0之间的强度评分，长上影线是看跌信号
        """
        pattern = [0.0] * len(closes)

        for i in range(len(closes)):
            # 计算实体、上影线、总体波动
            body_size = abs(closes[i] - opens[i])
            upper_shadow = highs[i] - max(opens[i], closes[i])
            total_range = highs[i] - lows[i]

            if total_range == 0:
                continue

            # 长上影线的判断条件
            has_long_upper_shadow = (
                upper_shadow >= 2 * body_size and  # 上影线至少是实体的2倍
                upper_shadow >= 0.5 * total_range  # 上影线占总体波动的50%以上
            )

            if has_long_upper_shadow:
                # 根据上影线长度和实体大小计算负值强度
                shadow_ratio = upper_shadow / total_range
                body_ratio = body_size / total_range

                # 上影线越长，实体越小，看跌信号越强
                if shadow_ratio >= 0.8 and body_ratio <= 0.1:  # 极长上影线，极小实体
                    strength = -0.9  # 强烈看跌信号（如您提到的高位长上影线）
                elif shadow_ratio >= 0.7 and body_ratio <= 0.2:  # 长上影线，小实体
                    strength = -0.7
                elif shadow_ratio >= 0.6 and body_ratio <= 0.3:  # 中等上影线
                    strength = -0.5
                elif shadow_ratio >= 0.5 and body_ratio <= 0.4:  # 较短上影线
                    strength = -0.3
                else:
                    strength = -0.1

                pattern[i] = strength

        return pattern

    def _identify_engulfing_bull(self, opens: np.ndarray, closes: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray) -> list:
        """
        识别看涨吞没 (2日形态)
        特征：
        1. 第1日：阴线
        2. 第2日：阳线，完全吞没第1日实体
        """
        pattern = [0] * len(closes)

        for i in range(1, len(closes)):  # 需要2日数据
            # 第1日：阴线
            day1_is_bear = closes[i-1] < opens[i-1]

            # 第2日：阳线，完全吞没
            day2_is_bull = closes[i] > opens[i]
            day2_engulfs = (opens[i] < closes[i-1] and closes[i] > opens[i-1])

            if day1_is_bear and day2_is_bull and day2_engulfs:
                pattern[i] = 1

        return pattern
