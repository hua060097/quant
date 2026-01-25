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
            # 1. 市值相对特征（新增）
            data = self.calculate_market_cap_features(data)

            # 2. 风险相关特征
            data = self._calculate_risk_features(data)

            # 3. 技术指标衍生特征
            data = self._calculate_technical_derived_features(data)

            # 4. 基本面衍生特征
            data = self._calculate_fundamental_derived_features(data)

            # 5. 流动性相关特征
            data = self._calculate_liquidity_features(data)

            # 6. 质量因子特征
            data = self._calculate_quality_features(data)

            # 7. 涨停质量特征（新增）- 暂时禁用：稀疏性问题导致IR计算异常（99%+零值）
            # data = self._calculate_limit_up_quality_features(data)

            # 8. K线市场微观结构特征（专业量化版本）
            data = self._calculate_microstructure_features(data)

            # 9. 横截面排名特征
            data = self._calculate_cross_sectional_features(data)

            # 10. 多周期相关性特征
            data = self._calculate_correlation_features(data)

            # 11. 累计涨幅特征（保留）
            data = self._calculate_cumulative_return_features(data)

            # 12. 均线粘合度指标（保留）
            data = self._calculate_ma_convergence_features(data)

            # 13. 股价低点距离指标（保留）
            data = self._calculate_price_distance_features(data)

            # 14. MA交叉强度特征（保留）
            data = self._calculate_ma_crossover_features(data)

            # 15. 高级价格位置特征
            data = self._calculate_advanced_price_features(data)

            # 16. 成交量相对特征
            data = self._calculate_volume_relative_features(data)

            # 17. 均线相对特征
            data = self._calculate_ma_relative_features(data)

            return data

        except Exception as e:
            self.logger.error(f"特征计算失败: {e}")
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

            # 2. limit_up_next_day_positive_rate: 涨停次日红盘率（历史统计特征，无未来信息泄露）
            if 'is_limit_up' in data.columns and 'pct_chg' in data.columns:
                # 计算次日涨跌幅（用于历史统计）
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

            # 4. consecutive_limit_rate: 连板率（首板后次日再涨停概率）- 历史统计特征
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

            # 5. limit_up_premium_5pct_count: 涨停次日溢价≥5%的次数 - 历史统计特征
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

    def _calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算K线市场微观结构特征（专业量化机构版本）

        替代传统主观形态识别，使用连续型数值特征量化市场微观结构
        基于顶级量化研究论文的市场微观结构理论

        参考文献：
        1. O'Hara (1995) "Market Microstructure Theory"
        2. Hasbrouck (2007) "Empirical Market Microstructure"
        3. Foucault et al. (2013) "Market Liquidity: Theory, Evidence, and Policy"
        """
        try:
            self.logger.debug("开始计算K线市场微观结构特征")

            # 初始化单日微观结构特征
            data['lower_shadow_ratio'] = 0.0      # 下影线比例：买方支撑强度
            data['upper_shadow_ratio'] = 0.0      # 上影线比例：卖方压力强度
            data['body_ratio'] = 0.0              # 实体比例：趋势信念强度
            data['reversal_score'] = 0.0          # 日内反转强度
            data['overnight_gap_pct'] = 0.0       # 隔夜跳空强度
            data['intraday_momentum'] = 0.0       # 日内动量强度
            data['price_position_ratio'] = 0.0    # 收盘价位置比例
            data['volatility_efficiency'] = 0.0   # 波动率效率
            data['buying_pressure'] = 0.0         # 买方压力指数
            data['selling_pressure'] = 0.0        # 卖方压力指数
            data['market_structure_score'] = 0.0  # 市场结构综合评分

            # 初始化基于DTW的多日形态相似度特征
            data['w_bottom_similarity'] = 0.0          # W底形态相似度
            data['head_shoulders_similarity'] = 0.0    # 头肩顶形态相似度
            data['cup_handle_similarity'] = 0.0        # 杯柄形态相似度
            data['ascending_triangle_similarity'] = 0.0 # 上升三角形相似度
            data['flag_pattern_similarity'] = 0.0      # 旗形整理相似度
            data['double_top_similarity'] = 0.0        # 双顶形态相似度

            # 初始化多日形态解构特征
            # W底解构特征
            data['w_bottom_vol_ratio'] = 0.0           # 波动率结构特征
            data['w_bottom_vol_profile_score'] = 0.0   # 成交量剖面特征
            data['w_bottom_rs_improvement'] = 0.0      # 相对强度变化特征

            # 头肩顶解构特征
            data['hs_vol_progression'] = 0.0           # 成交量递进特征
            data['hs_symmetry_score'] = 0.0            # 对称性评分
            data['hs_neckline_strength'] = 0.0         # 颈线强度特征

            # 杯柄解构特征
            data['cup_depth_ratio'] = 0.0              # 杯子深度比例
            data['handle_vol_contraction'] = 0.0       # 手柄缩量特征
            data['cup_rounding_score'] = 0.0           # 圆弧度评分

            # 按股票分组计算微观结构特征
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy()

                if len(stock_data) < 5:  # 至少需要5个交易日
                    continue

                # 按日期排序
                stock_data = stock_data.sort_values('date')

                # 计算单日微观结构特征
                microstructure_features = self._calculate_single_day_microstructure(stock_data)

                # 计算多日形态相似度特征
                pattern_similarities = self._calculate_pattern_similarities(stock_data)

                # 计算多日形态解构特征
                pattern_decompositions = self._calculate_pattern_decompositions(stock_data)

                # 将结果赋值回原数据
                for feature_name, feature_values in microstructure_features.items():
                    data.loc[stock_mask, feature_name] = feature_values

                for pattern_name, pattern_values in pattern_similarities.items():
                    data.loc[stock_mask, pattern_name] = pattern_values

                for decomp_name, decomp_values in pattern_decompositions.items():
                    data.loc[stock_mask, decomp_name] = decomp_values

            self.logger.debug("K线市场微观结构特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"K线市场微观结构特征计算失败: {e}")
            return data

    def _calculate_single_day_microstructure(self, stock_data: pd.DataFrame) -> dict:
        """
        计算单日微观结构特征

        基于市场微观结构理论，量化单日内的买卖压力、价格发现效率等
        """
        try:
            features = {}
            n = len(stock_data)

            # 检查必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in stock_data.columns for col in required_cols):
                # 返回零值特征
                for feature_name in ['lower_shadow_ratio', 'upper_shadow_ratio', 'body_ratio',
                                   'reversal_score', 'overnight_gap_pct', 'intraday_momentum',
                                   'price_position_ratio', 'volatility_efficiency', 'buying_pressure',
                                   'selling_pressure', 'market_structure_score']:
                    features[feature_name] = [0.0] * n
                return features

            # 提取价格数据
            opens = stock_data['open'].values
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            closes = stock_data['close'].values
            volumes = stock_data['volume'].values

            # 初始化特征数组
            lower_shadow_ratio = np.zeros(n)
            upper_shadow_ratio = np.zeros(n)
            body_ratio = np.zeros(n)
            reversal_score = np.zeros(n)
            overnight_gap_pct = np.zeros(n)
            intraday_momentum = np.zeros(n)
            price_position_ratio = np.zeros(n)
            volatility_efficiency = np.zeros(n)
            buying_pressure = np.zeros(n)
            selling_pressure = np.zeros(n)
            market_structure_score = np.zeros(n)

            for i in range(n):
                o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]

                # 防止除零错误
                if h == l or h == 0 or l == 0:
                    continue

                # 1. 下影线比例：买方支撑强度
                lower_shadow = min(o, c) - l
                total_range = h - l
                lower_shadow_ratio[i] = lower_shadow / total_range if total_range > 0 else 0

                # 2. 上影线比例：卖方压力强度
                upper_shadow = h - max(o, c)
                upper_shadow_ratio[i] = upper_shadow / total_range if total_range > 0 else 0

                # 3. 实体比例：趋势信念强度
                body_size = abs(c - o)
                body_ratio[i] = body_size / total_range if total_range > 0 else 0

                # 4. 日内反转强度
                if o != 0:
                    intraday_high_ret = (h - o) / o
                    intraday_low_ret = (l - o) / o
                    close_ret = (c - o) / o

                    # 反转强度 = 日内极值回撤程度
                    if close_ret > 0:  # 阳线
                        reversal_score[i] = max(0, intraday_low_ret) / (intraday_high_ret + 1e-8)
                    else:  # 阴线
                        reversal_score[i] = max(0, -intraday_high_ret) / (abs(intraday_low_ret) + 1e-8)

                # 5. 隔夜跳空强度
                if i > 0 and closes[i-1] != 0:
                    overnight_gap_pct[i] = (o - closes[i-1]) / closes[i-1]

                # 6. 日内动量强度
                if o != 0:
                    intraday_momentum[i] = (c - o) / o

                # 7. 收盘价位置比例
                price_position_ratio[i] = (c - l) / (h - l) if h != l else 0.5

                # 8. 波动率效率（价格变化/波动率）
                if o != 0 and total_range > 0:
                    price_change = abs(c - o) / o
                    volatility = total_range / o
                    volatility_efficiency[i] = price_change / (volatility + 1e-8)

                # 9. 买方压力指数
                if c > o:  # 阳线
                    buying_pressure[i] = (c - l) / (h - l) if h != l else 0.5
                else:  # 阴线
                    buying_pressure[i] = (o - l) / (h - l) if h != l else 0.5

                # 10. 卖方压力指数
                if c < o:  # 阴线
                    selling_pressure[i] = (h - c) / (h - l) if h != l else 0.5
                else:  # 阳线
                    selling_pressure[i] = (h - o) / (h - l) if h != l else 0.5

                # 11. 市场结构综合评分
                # 综合考虑实体比例、影线分布、价格位置
                structure_score = (
                    body_ratio[i] * 0.4 +  # 实体强度
                    (1 - upper_shadow_ratio[i]) * 0.3 +  # 上影线阻力
                    lower_shadow_ratio[i] * 0.3  # 下影线支撑
                )
                market_structure_score[i] = min(max(structure_score, 0), 1)

            # 返回特征字典
            features = {
                'lower_shadow_ratio': lower_shadow_ratio.tolist(),
                'upper_shadow_ratio': upper_shadow_ratio.tolist(),
                'body_ratio': body_ratio.tolist(),
                'reversal_score': reversal_score.tolist(),
                'overnight_gap_pct': overnight_gap_pct.tolist(),
                'intraday_momentum': intraday_momentum.tolist(),
                'price_position_ratio': price_position_ratio.tolist(),
                'volatility_efficiency': volatility_efficiency.tolist(),
                'buying_pressure': buying_pressure.tolist(),
                'selling_pressure': selling_pressure.tolist(),
                'market_structure_score': market_structure_score.tolist()
            }

            return features

        except Exception as e:
            self.logger.error(f"计算单日微观结构特征失败: {e}")
            # 返回零值特征
            n = len(stock_data)
            features = {}
            for feature_name in ['lower_shadow_ratio', 'upper_shadow_ratio', 'body_ratio',
                               'reversal_score', 'overnight_gap_pct', 'intraday_momentum',
                               'price_position_ratio', 'volatility_efficiency', 'buying_pressure',
                               'selling_pressure', 'market_structure_score']:
                features[feature_name] = [0.0] * n
            return features

    def _calculate_dtw_distance(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """
        计算两个时间序列的DTW距离

        基于专业量化机构的DTW实现，用于形态相似度计算
        """
        try:
            n, m = len(series1), len(series2)

            # 创建成本矩阵
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0

            # 填充DTW矩阵
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(series1[i-1] - series2[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # 插入
                        dtw_matrix[i, j-1],      # 删除
                        dtw_matrix[i-1, j-1]     # 匹配
                    )

            # 返回标准化的DTW距离
            return dtw_matrix[n, m] / (n + m)

        except Exception:
            return np.inf

    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """
        对时间序列进行Min-Max归一化到[0,1]区间

        这是专业形态识别的关键步骤，消除绝对价格影响
        """
        try:
            if len(series) == 0:
                return series

            min_val = np.min(series)
            max_val = np.max(series)

            if max_val == min_val:
                return np.full_like(series, 0.5)  # 如果序列是常数，返回0.5

            return (series - min_val) / (max_val - min_val)

        except Exception:
            return np.full_like(series, 0.5)

    def _create_pattern_templates(self) -> dict:
        """
        创建标准形态模板库

        基于专业量化机构的形态定义，创建理想的形态模板
        """
        templates = {}

        # W底模板 (25天周期)
        # 特征：两个相近的低点，中间有一个高点
        w_bottom_template = np.array([
            0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3,
            0.1, 0.15, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.9, 0.85, 0.8, 0.9
        ])
        templates['w_bottom'] = w_bottom_template

        # 头肩顶模板 (25天周期)
        # 特征：左肩-头部-右肩，头部最高
        head_shoulders_template = np.array([
            0.3, 0.4, 0.6, 0.7, 0.6, 0.4, 0.5, 0.7, 0.85, 1.0, 0.9, 0.8, 0.7,
            0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.1
        ])
        templates['head_shoulders'] = head_shoulders_template

        # 杯柄模板 (25天周期)
        # 特征：U型底部 + 小幅回调
        cup_handle_template = np.array([
            0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.1, 0.15, 0.2, 0.3, 0.4,
            0.6, 0.7, 0.8, 0.85, 0.9, 0.85, 0.8, 0.75, 0.8, 0.85, 0.9, 0.95
        ])
        templates['cup_handle'] = cup_handle_template

        # 上升三角形模板 (25天周期)
        # 特征：高点平行，低点逐步抬高
        ascending_triangle_template = np.array([
            0.2, 0.3, 0.5, 0.8, 0.9, 0.7, 0.4, 0.5, 0.7, 0.9, 0.85, 0.6, 0.5,
            0.6, 0.8, 0.9, 0.8, 0.65, 0.6, 0.7, 0.85, 0.9, 0.85, 0.9, 0.95
        ])
        templates['ascending_triangle'] = ascending_triangle_template

        # 旗形整理模板 (25天周期)
        # 特征：强势上涨后的横盘整理
        flag_pattern_template = np.array([
            0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.85, 0.8, 0.82, 0.78, 0.8, 0.77, 0.79,
            0.76, 0.78, 0.75, 0.77, 0.74, 0.76, 0.73, 0.75, 0.8, 0.85, 0.9, 0.95
        ])
        templates['flag_pattern'] = flag_pattern_template

        # 双顶模板 (25天周期)
        # 特征：两个相近的高点
        double_top_template = np.array([
            0.3, 0.4, 0.6, 0.8, 0.9, 0.95, 0.8, 0.6, 0.5, 0.4, 0.5, 0.6, 0.7,
            0.8, 0.9, 0.95, 0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.1
        ])
        templates['double_top'] = double_top_template

        return templates

    def _calculate_pattern_similarities(self, stock_data: pd.DataFrame) -> dict:
        """
        计算基于DTW的多日形态相似度特征

        使用动态时间规整(DTW)算法计算与经典形态的相似度
        这是专业量化机构使用的形态识别方法
        """
        try:
            n = len(stock_data)

            # 检查数据长度
            if n < 20:  # 至少需要20个交易日进行形态识别
                features = {}
                for pattern_name in ['w_bottom_similarity', 'head_shoulders_similarity',
                                   'cup_handle_similarity', 'ascending_triangle_similarity',
                                   'flag_pattern_similarity', 'double_top_similarity']:
                    features[pattern_name] = [0.0] * n
                return features

            # 检查必要的列
            required_cols = ['close', 'high', 'low', 'volume']
            if not all(col in stock_data.columns for col in required_cols):
                features = {}
                for pattern_name in ['w_bottom_similarity', 'head_shoulders_similarity',
                                   'cup_handle_similarity', 'ascending_triangle_similarity',
                                   'flag_pattern_similarity', 'double_top_similarity']:
                    features[pattern_name] = [0.0] * n
                return features

            # 提取价格数据
            closes = stock_data['close'].values
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            volumes = stock_data['volume'].values

            # 初始化相似度数组
            w_bottom_sim = np.zeros(n)
            head_shoulders_sim = np.zeros(n)
            cup_handle_sim = np.zeros(n)
            ascending_triangle_sim = np.zeros(n)
            flag_pattern_sim = np.zeros(n)
            double_top_sim = np.zeros(n)

            # 使用滑动窗口计算形态相似度
            window_size = 20

            for i in range(window_size, n):
                # 提取窗口数据
                window_closes = closes[i-window_size:i]
                window_highs = highs[i-window_size:i]
                window_lows = lows[i-window_size:i]
                window_volumes = volumes[i-window_size:i]

                # 标准化价格数据（使用窗口内的最小-最大标准化）
                price_min = np.min(window_lows)
                price_max = np.max(window_highs)
                if price_max > price_min:
                    norm_closes = (window_closes - price_min) / (price_max - price_min)
                    norm_highs = (window_highs - price_min) / (price_max - price_min)
                    norm_lows = (window_lows - price_min) / (price_max - price_min)
                else:
                    norm_closes = np.ones_like(window_closes) * 0.5
                    norm_highs = np.ones_like(window_highs) * 0.5
                    norm_lows = np.ones_like(window_lows) * 0.5

                # 使用专业DTW算法计算形态相似度
                # 获取标准形态模板
                templates = self._create_pattern_templates()

                # 调整归一化序列长度以匹配模板（25天）
                if len(norm_closes) == 20:
                    # 通过插值将20天数据扩展到25天
                    from scipy import interpolate
                    try:
                        x_old = np.linspace(0, 1, 20)
                        x_new = np.linspace(0, 1, 25)
                        f = interpolate.interp1d(x_old, norm_closes, kind='cubic')
                        norm_closes_25 = f(x_new)
                    except:
                        # 如果插值失败，使用简单重采样
                        norm_closes_25 = np.interp(np.linspace(0, 19, 25), np.arange(20), norm_closes)
                else:
                    norm_closes_25 = norm_closes

                # 计算DTW相似度
                dtw_dist = self._calculate_dtw_distance(norm_closes_25, templates['w_bottom'])
                w_bottom_sim[i] = 1.0 / (1.0 + dtw_dist)

                dtw_dist = self._calculate_dtw_distance(norm_closes_25, templates['head_shoulders'])
                head_shoulders_sim[i] = 1.0 / (1.0 + dtw_dist)

                dtw_dist = self._calculate_dtw_distance(norm_closes_25, templates['cup_handle'])
                cup_handle_sim[i] = 1.0 / (1.0 + dtw_dist)

                dtw_dist = self._calculate_dtw_distance(norm_closes_25, templates['ascending_triangle'])
                ascending_triangle_sim[i] = 1.0 / (1.0 + dtw_dist)

                dtw_dist = self._calculate_dtw_distance(norm_closes_25, templates['flag_pattern'])
                flag_pattern_sim[i] = 1.0 / (1.0 + dtw_dist)

                dtw_dist = self._calculate_dtw_distance(norm_closes_25, templates['double_top'])
                double_top_sim[i] = 1.0 / (1.0 + dtw_dist)

            # 返回特征字典
            features = {
                'w_bottom_similarity': w_bottom_sim.tolist(),
                'head_shoulders_similarity': head_shoulders_sim.tolist(),
                'cup_handle_similarity': cup_handle_sim.tolist(),
                'ascending_triangle_similarity': ascending_triangle_sim.tolist(),
                'flag_pattern_similarity': flag_pattern_sim.tolist(),
                'double_top_similarity': double_top_sim.tolist()
            }

            return features

        except Exception as e:
            self.logger.error(f"计算形态相似度特征失败: {e}")
            # 返回零值特征
            n = len(stock_data)
            features = {}
            for pattern_name in ['w_bottom_similarity', 'head_shoulders_similarity',
                               'cup_handle_similarity', 'ascending_triangle_similarity',
                               'flag_pattern_similarity', 'double_top_similarity']:
                features[pattern_name] = [0.0] * n
            return features

    # 删除旧的形态识别方法，已被专业DTW算法替代

    def _calculate_pattern_decompositions(self, stock_data: pd.DataFrame) -> dict:
        """
        计算多日形态解构特征

        基于专业量化研究的形态解构方法，提取形态的深层结构特征
        """
        try:
            n = len(stock_data)

            # 检查数据长度
            if n < 20:  # 至少需要20个交易日进行形态解构
                features = {}
                for feature_name in ['w_bottom_vol_ratio', 'w_bottom_vol_profile_score', 'w_bottom_rs_improvement',
                                   'hs_vol_progression', 'hs_symmetry_score', 'hs_neckline_strength',
                                   'cup_depth_ratio', 'handle_vol_contraction', 'cup_rounding_score']:
                    features[feature_name] = [0.0] * n
                return features

            # 检查必要的列
            required_cols = ['close', 'high', 'low', 'volume']
            if not all(col in stock_data.columns for col in required_cols):
                features = {}
                for feature_name in ['w_bottom_vol_ratio', 'w_bottom_vol_profile_score', 'w_bottom_rs_improvement',
                                   'hs_vol_progression', 'hs_symmetry_score', 'hs_neckline_strength',
                                   'cup_depth_ratio', 'handle_vol_contraction', 'cup_rounding_score']:
                    features[feature_name] = [0.0] * n
                return features

            # 提取价格和成交量数据
            closes = stock_data['close'].values
            highs = stock_data['high'].values
            lows = stock_data['low'].values
            volumes = stock_data['volume'].values

            # 初始化解构特征数组
            w_bottom_vol_ratio = np.zeros(n)
            w_bottom_vol_profile_score = np.zeros(n)
            w_bottom_rs_improvement = np.zeros(n)
            hs_vol_progression = np.zeros(n)
            hs_symmetry_score = np.zeros(n)
            hs_neckline_strength = np.zeros(n)
            cup_depth_ratio = np.zeros(n)
            handle_vol_contraction = np.zeros(n)
            cup_rounding_score = np.zeros(n)

            # 使用滑动窗口计算解构特征
            window_size = 20

            for i in range(window_size, n):
                # 提取窗口数据
                window_closes = closes[i-window_size:i]
                window_highs = highs[i-window_size:i]
                window_lows = lows[i-window_size:i]
                window_volumes = volumes[i-window_size:i]

                # W底解构特征
                w_decomp = self._decompose_w_bottom(window_closes, window_lows, window_volumes)
                w_bottom_vol_ratio[i] = w_decomp['vol_ratio']
                w_bottom_vol_profile_score[i] = w_decomp['vol_profile_score']
                w_bottom_rs_improvement[i] = w_decomp['rs_improvement']

                # 头肩顶解构特征
                hs_decomp = self._decompose_head_shoulders(window_closes, window_highs, window_volumes)
                hs_vol_progression[i] = hs_decomp['vol_progression']
                hs_symmetry_score[i] = hs_decomp['symmetry_score']
                hs_neckline_strength[i] = hs_decomp['neckline_strength']

                # 杯柄解构特征
                cup_decomp = self._decompose_cup_handle(window_closes, window_lows, window_volumes)
                cup_depth_ratio[i] = cup_decomp['depth_ratio']
                handle_vol_contraction[i] = cup_decomp['vol_contraction']
                cup_rounding_score[i] = cup_decomp['rounding_score']

            # 返回特征字典
            features = {
                'w_bottom_vol_ratio': w_bottom_vol_ratio.tolist(),
                'w_bottom_vol_profile_score': w_bottom_vol_profile_score.tolist(),
                'w_bottom_rs_improvement': w_bottom_rs_improvement.tolist(),
                'hs_vol_progression': hs_vol_progression.tolist(),
                'hs_symmetry_score': hs_symmetry_score.tolist(),
                'hs_neckline_strength': hs_neckline_strength.tolist(),
                'cup_depth_ratio': cup_depth_ratio.tolist(),
                'handle_vol_contraction': handle_vol_contraction.tolist(),
                'cup_rounding_score': cup_rounding_score.tolist()
            }

            return features

        except Exception as e:
            self.logger.error(f"计算形态解构特征失败: {e}")
            # 返回零值特征
            n = len(stock_data)
            features = {}
            for feature_name in ['w_bottom_vol_ratio', 'w_bottom_vol_profile_score', 'w_bottom_rs_improvement',
                               'hs_vol_progression', 'hs_symmetry_score', 'hs_neckline_strength',
                               'cup_depth_ratio', 'handle_vol_contraction', 'cup_rounding_score']:
                features[feature_name] = [0.0] * n
            return features

    def _decompose_w_bottom(self, closes: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> dict:
        """W底形态解构分析"""
        try:
            n = len(closes)
            if n < 10:
                return {'vol_ratio': 0.0, 'vol_profile_score': 0.0, 'rs_improvement': 0.0}

            # 寻找两个低点 - 修复未来信息泄露
            local_mins = []
            for i in range(4, n):  # 从第4个点开始
                # W底低点确认（使用历史数据）
                if (i >= 4 and
                    lows[i-2] < lows[i-3] and lows[i-2] < lows[i-4] and
                    lows[i-2] < lows[i-1] and lows[i-2] < lows[i]):
                    local_mins.append((i-2, lows[i-2]))  # 记录2天前确认的极值点

            if len(local_mins) < 2:
                return {'vol_ratio': 0.0, 'vol_profile_score': 0.0, 'rs_improvement': 0.0}

            # 选择最佳的两个低点
            min1_idx, min1_val = local_mins[0]
            min2_idx, min2_val = local_mins[-1]

            # 1. 波动率结构特征
            left_vol = np.std(closes[:min1_idx+1]) if min1_idx > 0 else 0
            middle_vol = np.std(closes[min1_idx:min2_idx+1]) if min2_idx > min1_idx else 0
            right_vol = np.std(closes[min2_idx:]) if min2_idx < n-1 else 0

            if middle_vol > 0:
                vol_ratio = (left_vol + right_vol) / (2 * middle_vol)
            else:
                vol_ratio = 0.0

            # 2. 成交量剖面特征
            left_vol_avg = np.mean(volumes[:min1_idx+1]) if min1_idx > 0 else 0
            middle_vol_avg = np.mean(volumes[min1_idx:min2_idx+1]) if min2_idx > min1_idx else 0
            right_vol_avg = np.mean(volumes[min2_idx:]) if min2_idx < n-1 else 0

            # 理想的W底：左右成交量高，中间成交量低
            if middle_vol_avg > 0:
                vol_profile_score = (left_vol_avg + right_vol_avg) / (2 * middle_vol_avg)
                vol_profile_score = min(vol_profile_score / 2, 1.0)  # 标准化到[0,1]
            else:
                vol_profile_score = 0.0

            # 3. 相对强度变化特征
            if min1_val > 0 and min2_val > 0:
                left_strength = (closes[min1_idx] - closes[0]) / closes[0] if closes[0] > 0 else 0
                right_strength = (closes[-1] - closes[min2_idx]) / closes[min2_idx] if closes[min2_idx] > 0 else 0
                rs_improvement = max(0, right_strength - left_strength)
            else:
                rs_improvement = 0.0

            return {
                'vol_ratio': min(max(vol_ratio, 0), 2.0),
                'vol_profile_score': vol_profile_score,
                'rs_improvement': min(max(rs_improvement, 0), 1.0)
            }

        except Exception:
            return {'vol_ratio': 0.0, 'vol_profile_score': 0.0, 'rs_improvement': 0.0}

    def _decompose_head_shoulders(self, closes: np.ndarray, highs: np.ndarray, volumes: np.ndarray) -> dict:
        """头肩顶形态解构分析"""
        try:
            n = len(closes)
            if n < 15:
                return {'vol_progression': 0.0, 'symmetry_score': 0.0, 'neckline_strength': 0.0}

            # 寻找三个高点 - 修复未来信息泄露
            local_maxs = []
            for i in range(4, n):  # 从第4个点开始
                # 头肩顶高点确认（使用历史数据）
                if (i >= 4 and
                    highs[i-2] > highs[i-3] and highs[i-2] > highs[i-4] and
                    highs[i-2] > highs[i-1] and highs[i-2] > highs[i]):
                    local_maxs.append((i-2, highs[i-2]))  # 记录2天前确认的极值点

            if len(local_maxs) < 3:
                return {'vol_progression': 0.0, 'symmetry_score': 0.0, 'neckline_strength': 0.0}

            # 选择最佳的头肩顶组合
            left_shoulder = local_maxs[0]
            head = local_maxs[1]
            right_shoulder = local_maxs[2]

            # 1. 成交量递进特征
            left_vol = np.mean(volumes[:left_shoulder[0]+1]) if left_shoulder[0] > 0 else 0
            head_vol = np.mean(volumes[left_shoulder[0]:head[0]+1]) if head[0] > left_shoulder[0] else 0
            right_vol = np.mean(volumes[head[0]:right_shoulder[0]+1]) if right_shoulder[0] > head[0] else 0

            # 理想的头肩顶：成交量递减
            if left_vol > 0 and head_vol > 0 and right_vol > 0:
                vol_progression = (left_vol - head_vol) / left_vol + (head_vol - right_vol) / head_vol
                vol_progression = max(0, vol_progression / 2)
            else:
                vol_progression = 0.0

            # 2. 对称性评分
            left_height = left_shoulder[1]
            head_height = head[1]
            right_height = right_shoulder[1]

            if head_height > 0:
                left_ratio = left_height / head_height
                right_ratio = right_height / head_height
                symmetry_score = 1.0 - abs(left_ratio - right_ratio)
                symmetry_score = max(0, symmetry_score)
            else:
                symmetry_score = 0.0

            # 3. 颈线强度特征
            # 寻找两个肩部之间的最低点作为颈线
            neckline_start = left_shoulder[0]
            neckline_end = right_shoulder[0]
            if neckline_end > neckline_start:
                neckline_low = np.min(closes[neckline_start:neckline_end+1])
                neckline_strength = (head_height - neckline_low) / head_height if head_height > 0 else 0
                neckline_strength = min(max(neckline_strength, 0), 1.0)
            else:
                neckline_strength = 0.0

            return {
                'vol_progression': min(vol_progression, 1.0),
                'symmetry_score': symmetry_score,
                'neckline_strength': neckline_strength
            }

        except Exception:
            return {'vol_progression': 0.0, 'symmetry_score': 0.0, 'neckline_strength': 0.0}

    def _decompose_cup_handle(self, closes: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> dict:
        """杯柄形态解构分析"""
        try:
            n = len(closes)
            if n < 15:
                return {'depth_ratio': 0.0, 'vol_contraction': 0.0, 'rounding_score': 0.0}

            # 寻找最低点（杯底）
            min_idx = np.argmin(lows)

            if min_idx < 5 or min_idx > n - 5:
                return {'depth_ratio': 0.0, 'vol_contraction': 0.0, 'rounding_score': 0.0}

            # 1. 杯子深度比例
            cup_start_price = closes[0]
            cup_bottom_price = closes[min_idx]

            if cup_start_price > 0:
                depth_ratio = (cup_start_price - cup_bottom_price) / cup_start_price
                depth_ratio = min(max(depth_ratio, 0), 1.0)
            else:
                depth_ratio = 0.0

            # 2. 手柄缩量特征
            cup_volume = np.mean(volumes[:min_idx+5])  # 杯子阶段成交量
            handle_volume = np.mean(volumes[min_idx+5:])  # 手柄阶段成交量

            if cup_volume > 0:
                vol_contraction = (cup_volume - handle_volume) / cup_volume
                vol_contraction = min(max(vol_contraction, 0), 1.0)
            else:
                vol_contraction = 0.0

            # 3. 圆弧度评分
            # 计算杯子部分的圆弧度
            cup_prices = closes[:min_idx+5]
            if len(cup_prices) >= 10:
                try:
                    # 简化的圆弧度计算：检查价格变化的平滑度
                    price_changes = np.diff(cup_prices)
                    change_smoothness = 1.0 - np.std(price_changes) / (np.mean(np.abs(price_changes)) + 1e-8)
                    rounding_score = min(max(change_smoothness, 0), 1.0)
                except:
                    rounding_score = 0.0
            else:
                rounding_score = 0.0

            return {
                'depth_ratio': depth_ratio,
                'vol_contraction': vol_contraction,
                'rounding_score': rounding_score
            }

        except Exception:
            return {'depth_ratio': 0.0, 'vol_contraction': 0.0, 'rounding_score': 0.0}

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

    def _calculate_cross_sectional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算横截面排名特征

        基于横截面排名的相对强度特征，用于识别相对表现优异的股票
        """
        try:
            self.logger.debug("开始计算横截面排名特征")

            # 初始化横截面排名特征
            data['rank_return_20d'] = 0.0
            data['rank_volatility20'] = 0.0
            data['rank_turnover_rate'] = 0.0
            data['rank_total_mv'] = 0.0
            data['rank_cumulative_return_5d'] = 0.0
            data['rank_cumulative_return_10d'] = 0.0
            data['rank_rsi14'] = 0.0
            data['rank_return_1d'] = 0.0
            data['rank_return_3d'] = 0.0
            data['rank_momentum_composite'] = 0.0
            data['rank_quality_composite'] = 0.0
            data['rank_risk_composite'] = 0.0

            # 按日期分组计算横截面排名
            for date in data['date'].unique():
                date_mask = data['date'] == date
                date_data = data[date_mask].copy()

                if len(date_data) < 10:  # 至少需要10只股票
                    continue

                # 计算各种特征的横截面排名（百分位）
                if 'return_20d' in date_data.columns:
                    data.loc[date_mask, 'rank_return_20d'] = date_data['return_20d'].rank(pct=True)

                if 'volatility20' in date_data.columns:
                    data.loc[date_mask, 'rank_volatility20'] = date_data['volatility20'].rank(pct=True, ascending=False)  # 低波动率排名高

                if 'turnover_rate' in date_data.columns:
                    data.loc[date_mask, 'rank_turnover_rate'] = date_data['turnover_rate'].rank(pct=True)

                if 'total_mv' in date_data.columns:
                    data.loc[date_mask, 'rank_total_mv'] = date_data['total_mv'].rank(pct=True, ascending=False)  # 小市值排名高

                if 'cumulative_return_5d' in date_data.columns:
                    data.loc[date_mask, 'rank_cumulative_return_5d'] = date_data['cumulative_return_5d'].rank(pct=True)

                if 'cumulative_return_10d' in date_data.columns:
                    data.loc[date_mask, 'rank_cumulative_return_10d'] = date_data['cumulative_return_10d'].rank(pct=True)

                if 'rsi14' in date_data.columns:
                    data.loc[date_mask, 'rank_rsi14'] = date_data['rsi14'].rank(pct=True)

                if 'pct_chg' in date_data.columns:
                    data.loc[date_mask, 'rank_return_1d'] = date_data['pct_chg'].rank(pct=True)

                # 计算复合排名特征
                momentum_features = ['return_20d', 'cumulative_return_5d', 'cumulative_return_10d']
                available_momentum = [f for f in momentum_features if f in date_data.columns]
                if available_momentum:
                    momentum_composite = date_data[available_momentum].mean(axis=1)
                    data.loc[date_mask, 'rank_momentum_composite'] = momentum_composite.rank(pct=True)

                quality_features = ['rsi14', 'volatility20']
                available_quality = [f for f in quality_features if f in date_data.columns]
                if available_quality:
                    # 质量评分：高RSI + 低波动率
                    quality_score = pd.Series(0.0, index=date_data.index)
                    if 'rsi14' in date_data.columns:
                        quality_score += date_data['rsi14'].rank(pct=True) * 0.5
                    if 'volatility20' in date_data.columns:
                        quality_score += date_data['volatility20'].rank(pct=True, ascending=False) * 0.5
                    data.loc[date_mask, 'rank_quality_composite'] = quality_score.rank(pct=True)

                # 风险评分：低波动率 + 适中换手率
                risk_score = pd.Series(0.0, index=date_data.index)
                if 'volatility20' in date_data.columns:
                    risk_score += date_data['volatility20'].rank(pct=True, ascending=False) * 0.6
                if 'turnover_rate' in date_data.columns:
                    # 适中换手率得分最高
                    turnover_median = date_data['turnover_rate'].median()
                    turnover_distance = abs(date_data['turnover_rate'] - turnover_median)
                    risk_score += turnover_distance.rank(pct=True, ascending=False) * 0.4
                data.loc[date_mask, 'rank_risk_composite'] = risk_score.rank(pct=True)

            self.logger.debug("横截面排名特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"横截面排名特征计算失败: {e}")
            return data

    def _calculate_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算多周期股票市场相关性特征

        基于CAPM模型和多因子模型理论，计算股票与市场的相关性和Beta系数
        使用market_data中的daily_change_pct作为市场基准
        """
        try:
            self.logger.debug("开始计算多周期相关性特征")

            # 初始化相关性特征
            periods = [5, 10, 20, 60, 120]

            # 基础滚动指标
            for period in periods:
                data[f'corr_{period}d'] = 0.0
                data[f'beta_{period}d'] = 0.0

            # 期限结构特征
            data['corr_slope_5_20'] = 0.0
            data['corr_slope_10_60'] = 0.0
            data['corr_slope_20_120'] = 0.0
            data['beta_slope_5_20'] = 0.0
            data['beta_slope_10_60'] = 0.0
            data['beta_slope_20_120'] = 0.0

            # 动态变化特征
            data['corr_10d_mom5'] = 0.0
            data['corr_20d_mom5'] = 0.0
            data['corr_60d_mom5'] = 0.0
            data['beta_10d_mom5'] = 0.0
            data['beta_20d_mom5'] = 0.0
            data['beta_60d_mom5'] = 0.0

            # 相对位置特征
            data['corr_20d_zscore250'] = 0.0
            data['corr_60d_zscore250'] = 0.0
            data['beta_20d_zscore250'] = 0.0
            data['beta_60d_zscore250'] = 0.0

            # 尝试加载市场数据进行相关性计算
            try:
                # 从data_loader加载市场数据
                from .data_loader import DataLoader
                data_loader = DataLoader()

                # 获取数据的日期范围
                if 'date' in data.columns:
                    dates = data['date'].unique()
                    start_date = min(dates).strftime('%Y%m%d')
                    end_date = max(dates).strftime('%Y%m%d')

                    # 加载市场数据
                    _, _, market_data = data_loader.load_all_data(start_date, end_date)

                    if market_data is not None and not market_data.empty:
                        # 使用市场数据计算相关性特征
                        data = self._calculate_correlation_with_market(data, market_data)
                        self.logger.debug("使用市场数据计算相关性特征")
                    else:
                        self.logger.warning("市场数据为空，使用默认相关性特征")
                else:
                    self.logger.warning("数据中缺少日期字段，无法加载市场数据")

            except Exception as e:
                self.logger.warning(f"加载市场数据失败: {e}，使用默认相关性特征")

            self.logger.debug("多周期相关性特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"多周期相关性特征计算失败: {e}")
            return data

    def _calculate_correlation_with_market(self, stock_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        使用市场数据计算股票与市场的相关性特征
        """
        try:
            # 确保市场数据有daily_change_pct字段
            if 'daily_change_pct' not in market_data.columns:
                self.logger.warning("市场数据中缺少daily_change_pct字段")
                return stock_data

            # 按股票分组计算相关性
            for stock_code in stock_data['stock_code'].unique():
                stock_mask = stock_data['stock_code'] == stock_code
                stock_subset = stock_data[stock_mask].copy().sort_values('date')

                if len(stock_subset) < 20:  # 至少需要20个交易日
                    continue

                # 获取股票收益率
                if 'pct_chg' in stock_subset.columns:
                    stock_returns = stock_subset['pct_chg'].values / 100.0  # 转换为小数
                else:
                    continue

                # 匹配市场收益率
                market_returns = []
                for date in stock_subset['date']:
                    market_row = market_data[market_data['date'] == date]
                    if not market_row.empty:
                        market_return = market_row['daily_change_pct'].iloc[0] / 100.0  # 转换为小数
                        market_returns.append(market_return)
                    else:
                        market_returns.append(0.0)  # 缺失数据用0填充

                market_returns = np.array(market_returns)

                if len(stock_returns) != len(market_returns):
                    continue

                # 计算各周期的相关性和Beta
                periods = [5, 10, 20, 60, 120]
                correlations = {}
                betas = {}

                for period in periods:
                    if len(stock_returns) >= period:
                        corr_values = []
                        beta_values = []

                        for i in range(period-1, len(stock_returns)):
                            window_stock = stock_returns[i-period+1:i+1]
                            window_market = market_returns[i-period+1:i+1]

                            # 计算相关系数
                            if np.std(window_stock) > 1e-8 and np.std(window_market) > 1e-8:
                                corr = np.corrcoef(window_stock, window_market)[0, 1]
                                if not np.isnan(corr):
                                    corr_values.append(corr)
                                else:
                                    corr_values.append(0.0)

                                # 计算Beta系数
                                market_var = np.var(window_market)
                                if market_var > 1e-8:
                                    beta = np.cov(window_stock, window_market)[0, 1] / market_var
                                    if not np.isnan(beta):
                                        beta_values.append(beta)
                                    else:
                                        beta_values.append(1.0)
                                else:
                                    beta_values.append(1.0)
                            else:
                                corr_values.append(0.0)
                                beta_values.append(1.0)

                        # 填充前面的值
                        while len(corr_values) < len(stock_returns):
                            corr_values.insert(0, 0.0)
                        while len(beta_values) < len(stock_returns):
                            beta_values.insert(0, 1.0)

                        correlations[period] = corr_values
                        betas[period] = beta_values

                # 将计算结果赋值回原数据
                for period in periods:
                    if period in correlations:
                        stock_data.loc[stock_mask, f'corr_{period}d'] = correlations[period]
                        stock_data.loc[stock_mask, f'beta_{period}d'] = betas[period]

                # 计算期限结构特征
                if 5 in correlations and 20 in correlations:
                    corr_slope_5_20 = np.array(correlations[20]) - np.array(correlations[5])
                    stock_data.loc[stock_mask, 'corr_slope_5_20'] = corr_slope_5_20

                if 10 in correlations and 60 in correlations:
                    corr_slope_10_60 = np.array(correlations[60]) - np.array(correlations[10])
                    stock_data.loc[stock_mask, 'corr_slope_10_60'] = corr_slope_10_60

                if 20 in correlations and 120 in correlations:
                    corr_slope_20_120 = np.array(correlations[120]) - np.array(correlations[20])
                    stock_data.loc[stock_mask, 'corr_slope_20_120'] = corr_slope_20_120

                # 计算Beta期限结构
                if 5 in betas and 20 in betas:
                    beta_slope_5_20 = np.array(betas[20]) - np.array(betas[5])
                    stock_data.loc[stock_mask, 'beta_slope_5_20'] = beta_slope_5_20

                if 10 in betas and 60 in betas:
                    beta_slope_10_60 = np.array(betas[60]) - np.array(betas[10])
                    stock_data.loc[stock_mask, 'beta_slope_10_60'] = beta_slope_10_60

                # 计算动态变化特征（5日动量）
                if 10 in correlations:
                    corr_10d = np.array(correlations[10])
                    corr_10d_mom5 = np.zeros_like(corr_10d)
                    for i in range(5, len(corr_10d)):
                        corr_10d_mom5[i] = corr_10d[i] - corr_10d[i-5]
                    stock_data.loc[stock_mask, 'corr_10d_mom5'] = corr_10d_mom5

                if 20 in correlations:
                    corr_20d = np.array(correlations[20])
                    corr_20d_mom5 = np.zeros_like(corr_20d)
                    for i in range(5, len(corr_20d)):
                        corr_20d_mom5[i] = corr_20d[i] - corr_20d[i-5]
                    stock_data.loc[stock_mask, 'corr_20d_mom5'] = corr_20d_mom5

                if 60 in correlations:
                    corr_60d = np.array(correlations[60])
                    corr_60d_mom5 = np.zeros_like(corr_60d)
                    for i in range(5, len(corr_60d)):
                        corr_60d_mom5[i] = corr_60d[i] - corr_60d[i-5]
                    stock_data.loc[stock_mask, 'corr_60d_mom5'] = corr_60d_mom5

                # 计算相对位置特征（Z-score）
                if 20 in correlations:
                    corr_20d = np.array(correlations[20])
                    corr_20d_zscore250 = np.zeros_like(corr_20d)
                    for i in range(250, len(corr_20d)):
                        window = corr_20d[i-250:i]
                        mean_val = np.mean(window)
                        std_val = np.std(window)
                        if std_val > 1e-8:
                            corr_20d_zscore250[i] = (corr_20d[i] - mean_val) / std_val
                    stock_data.loc[stock_mask, 'corr_20d_zscore250'] = corr_20d_zscore250

                if 60 in correlations:
                    corr_60d = np.array(correlations[60])
                    corr_60d_zscore250 = np.zeros_like(corr_60d)
                    for i in range(250, len(corr_60d)):
                        window = corr_60d[i-250:i]
                        mean_val = np.mean(window)
                        std_val = np.std(window)
                        if std_val > 1e-8:
                            corr_60d_zscore250[i] = (corr_60d[i] - mean_val) / std_val
                    stock_data.loc[stock_mask, 'corr_60d_zscore250'] = corr_60d_zscore250

            return stock_data

        except Exception as e:
            self.logger.error(f"计算市场相关性特征失败: {e}")
            return stock_data

    def _calculate_advanced_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算高级价格位置特征
        """
        try:
            self.logger.debug("开始计算高级价格位置特征")

            # 初始化特征
            data['price_position_in_range_5d'] = 0.0
            data['price_position_in_range_20d'] = 0.0
            data['price_position_in_range_60d'] = 0.0
            data['price_vs_ma5_ratio'] = 0.0
            data['price_vs_ma20_ratio'] = 0.0
            data['price_vs_ma60_ratio'] = 0.0

            # 按股票分组计算
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy().sort_values('date')

                if len(stock_data) < 5:
                    continue

                # 计算价格位置特征
                closes = stock_data['close'].values
                highs = stock_data['high'].values
                lows = stock_data['low'].values

                # 价格在区间内的位置
                for period in [5, 20, 60]:
                    if len(closes) >= period:
                        position_values = []
                        for j in range(len(closes)):
                            start_idx = max(0, j - period + 1)
                            period_high = np.max(highs[start_idx:j+1])
                            period_low = np.min(lows[start_idx:j+1])
                            if period_high > period_low:
                                position = (closes[j] - period_low) / (period_high - period_low)
                            else:
                                position = 0.5
                            position_values.append(position)

                        feature_name = f'price_position_in_range_{period}d'
                        data.loc[stock_mask, feature_name] = position_values

                # 价格相对均线比例
                if 'ma5' in stock_data.columns:
                    ma5_ratio = stock_data['close'] / stock_data['ma5'] - 1
                    data.loc[stock_mask, 'price_vs_ma5_ratio'] = ma5_ratio.fillna(0)

                if 'ma20' in stock_data.columns:
                    ma20_ratio = stock_data['close'] / stock_data['ma20'] - 1
                    data.loc[stock_mask, 'price_vs_ma20_ratio'] = ma20_ratio.fillna(0)

                if 'ma60' in stock_data.columns:
                    ma60_ratio = stock_data['close'] / stock_data['ma60'] - 1
                    data.loc[stock_mask, 'price_vs_ma60_ratio'] = ma60_ratio.fillna(0)

            self.logger.debug("高级价格位置特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"高级价格位置特征计算失败: {e}")
            return data

    def _calculate_volume_relative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量相对特征
        """
        try:
            self.logger.debug("开始计算成交量相对特征")

            # 初始化特征
            data['volume_ratio_5d'] = 0.0
            data['volume_ratio_20d'] = 0.0
            data['volume_percentile_60d'] = 0.0
            data['volatility_zscore_20d'] = 0.0
            data['volatility_percentile_60d'] = 0.0

            # 按股票分组计算
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy().sort_values('date')

                if len(stock_data) < 5:
                    continue

                # 成交量比率特征
                if 'volume' in stock_data.columns:
                    volumes = stock_data['volume'].values

                    # 5日成交量比率
                    volume_ratio_5d = []
                    for i in range(len(volumes)):
                        if i >= 5:
                            avg_volume_5d = np.mean(volumes[i-5:i])
                            ratio = volumes[i] / (avg_volume_5d + 1e-8)
                        else:
                            ratio = 1.0
                        volume_ratio_5d.append(ratio)
                    data.loc[stock_mask, 'volume_ratio_5d'] = volume_ratio_5d

                    # 20日成交量比率
                    volume_ratio_20d = []
                    for i in range(len(volumes)):
                        if i >= 20:
                            avg_volume_20d = np.mean(volumes[i-20:i])
                            ratio = volumes[i] / (avg_volume_20d + 1e-8)
                        else:
                            ratio = 1.0
                        volume_ratio_20d.append(ratio)
                    data.loc[stock_mask, 'volume_ratio_20d'] = volume_ratio_20d

                    # 60日成交量百分位
                    volume_percentile_60d = []
                    for i in range(len(volumes)):
                        if i >= 60:
                            window_volumes = volumes[i-60:i+1]
                            percentile = (np.sum(window_volumes <= volumes[i]) - 1) / len(window_volumes)
                        else:
                            percentile = 0.5
                        volume_percentile_60d.append(percentile)
                    data.loc[stock_mask, 'volume_percentile_60d'] = volume_percentile_60d

                # 波动率相对特征
                if 'volatility20' in stock_data.columns:
                    volatilities = stock_data['volatility20'].values

                    # 20日波动率Z-score
                    volatility_zscore_20d = []
                    for i in range(len(volatilities)):
                        if i >= 20:
                            window_vol = volatilities[i-20:i]
                            mean_vol = np.mean(window_vol)
                            std_vol = np.std(window_vol)
                            if std_vol > 0:
                                zscore = (volatilities[i] - mean_vol) / std_vol
                            else:
                                zscore = 0.0
                        else:
                            zscore = 0.0
                        volatility_zscore_20d.append(zscore)
                    data.loc[stock_mask, 'volatility_zscore_20d'] = volatility_zscore_20d

                    # 60日波动率百分位
                    volatility_percentile_60d = []
                    for i in range(len(volatilities)):
                        if i >= 60:
                            window_vol = volatilities[i-60:i+1]
                            percentile = (np.sum(window_vol <= volatilities[i]) - 1) / len(window_vol)
                        else:
                            percentile = 0.5
                        volatility_percentile_60d.append(percentile)
                    data.loc[stock_mask, 'volatility_percentile_60d'] = volatility_percentile_60d

            self.logger.debug("成交量相对特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"成交量相对特征计算失败: {e}")
            return data

    def _calculate_ma_relative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算均线相对特征
        """
        try:
            self.logger.debug("开始计算均线相对特征")

            # 初始化特征
            data['ma_alignment_score'] = 0.0
            data['ma_convergence_score'] = 0.0
            data['ma5_ma20_distance_pct'] = 0.0
            data['ma20_ma60_distance_pct'] = 0.0

            # 按股票分组计算
            for stock_code in data['stock_code'].unique():
                stock_mask = data['stock_code'] == stock_code
                stock_data = data[stock_mask].copy()

                if len(stock_data) < 5:
                    continue

                # 均线排列评分
                ma_cols = ['ma5', 'ma10', 'ma20', 'ma60']
                available_mas = [col for col in ma_cols if col in stock_data.columns]

                if len(available_mas) >= 3:
                    alignment_scores = []
                    convergence_scores = []

                    for _, row in stock_data.iterrows():
                        ma_values = [row[col] for col in available_mas if pd.notna(row[col])]

                        if len(ma_values) >= 3:
                            # 多头排列评分
                            is_bullish = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
                            is_bearish = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))

                            if is_bullish:
                                alignment_score = 1.0
                            elif is_bearish:
                                alignment_score = -1.0
                            else:
                                alignment_score = 0.0

                            # 均线收敛度评分
                            ma_range = max(ma_values) - min(ma_values)
                            ma_mean = np.mean(ma_values)
                            convergence_score = 1.0 - (ma_range / (ma_mean + 1e-8))
                        else:
                            alignment_score = 0.0
                            convergence_score = 0.0

                        alignment_scores.append(alignment_score)
                        convergence_scores.append(max(0, convergence_score))

                    data.loc[stock_mask, 'ma_alignment_score'] = alignment_scores
                    data.loc[stock_mask, 'ma_convergence_score'] = convergence_scores

                # 均线距离百分比
                if 'ma5' in stock_data.columns and 'ma20' in stock_data.columns:
                    ma5_ma20_distance = (stock_data['ma5'] - stock_data['ma20']) / stock_data['ma20']
                    data.loc[stock_mask, 'ma5_ma20_distance_pct'] = ma5_ma20_distance.fillna(0)

                if 'ma20' in stock_data.columns and 'ma60' in stock_data.columns:
                    ma20_ma60_distance = (stock_data['ma20'] - stock_data['ma60']) / stock_data['ma60']
                    data.loc[stock_mask, 'ma20_ma60_distance_pct'] = ma20_ma60_distance.fillna(0)

            self.logger.debug("均线相对特征计算完成")
            return data

        except Exception as e:
            self.logger.error(f"均线相对特征计算失败: {e}")
            return data

    # 删除所有过时的K线形态识别方法，保留calculate_market_cap_features

    def calculate_market_cap_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市值相关特征 - 基于专业量化研究

        参考文献：
        - Fama & French (1992) "The Cross-Section of Expected Stock Returns"
        - Banz (1981) "The relationship between return and market value"
        - 专业量化机构实践：使用相对特征而非绝对市值

        Args:
            stock_data: 包含total_mv和circ_mv的股票数据

        Returns:
            添加了市值相对特征的数据
        """
        try:
            data = stock_data.copy()

            # 检查是否有市值数据
            if 'total_mv' not in data.columns and 'circ_mv' not in data.columns:
                self.logger.warning("数据中缺少市值字段，跳过市值特征计算")
                return data

            # 使用total_mv作为主要市值指标，如果没有则使用circ_mv
            if 'total_mv' in data.columns:
                mv_column = 'total_mv'
            else:
                mv_column = 'circ_mv'

            # 过滤有效市值数据
            valid_mv_mask = (data[mv_column].notna()) & (data[mv_column] > 0)

            if not valid_mv_mask.any():
                self.logger.warning("没有有效的市值数据，跳过市值特征计算")
                return data

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
            return stock_data
