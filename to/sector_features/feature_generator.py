#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征生成器模块

计算板块的各种特征，包括价格指标、成交量指标、技术指标等。
"""

import os
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
# 导入技术指标计算类
from to.sector_features.technical_indicators import TechnicalIndicators

try:
    from to.path_utils import get_data_path, ensure_dir
except ImportError:
    # 如果导入失败，提供简单的替代实现
    def get_data_path(relative_path=None):
        from pathlib import Path
        data_root = Path('to') / 'local_data'
        if relative_path:
            return data_root / relative_path
        return data_root
    
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

logger = logging.getLogger(__name__)

class SectorFeatureGenerator:
    """板块特征生成器
    
    生成板块的各种基础特征，包括价格均值、成交量、成交额、活跃股票数等。
    """
    
    def __init__(self, data_provider, sector_mapping, output_dir=None, 
                weighting_method='equal'):
        """初始化特征生成器
        
        Args:
            data_provider: 数据提供者，需要实现get_daily_data和get_trading_days方法
            sector_mapping: 板块映射字典 {板块名称: [股票列表]}
            output_dir: 特征输出目录，默认使用get_data_path('sector_features')
            weighting_method: 权重计算方法，可选值有equal(等权重)、market_cap(市值加权)等
        """
        self.data_provider = data_provider
        self.sector_mapping = sector_mapping
        # 使用path_utils来获取正确的路径
        self.output_dir = output_dir if output_dir is not None else get_data_path('sector_features')
        self.weighting_method = weighting_method
        
        # 创建输出目录
        ensure_dir(self.output_dir)
        
        logger.info(f"初始化板块特征生成器，权重方法: {weighting_method}, 输出目录: {self.output_dir}")
        
    def _get_sector_stocks(self, sector):
        """获取板块成分股
        
        Args:
            sector: 板块名称
            
        Returns:
            股票列表
        """
        # 从sector_mapping中获取
        stocks = self.sector_mapping.get(sector, [])
        
        # 如果为空，尝试从stock_sectors_mapping中获取
        if not stocks:
            try:
                # 兼容性代码，允许从外部获取股票列表
                from to.stock_sectors_mapping import get_sector_stocks
                stocks = get_sector_stocks(sector_name=sector)
            except Exception as e:
                logger.error(f"获取板块 {sector} 的成分股失败: {str(e)}")
                stocks = []
                
        logger.debug(f"板块 {sector} 包含 {len(stocks)} 只股票")
        return stocks
        
    def _get_weights(self, data, method='equal'):
        """计算权重
        
        Args:
            data: DataFrame，包含要计算权重的数据
            method: 权重计算方法，可选值有equal、market_cap等
            
        Returns:
            权重Series
        """
        if method == 'equal':
            # 等权重
            return pd.Series(1.0 / len(data), index=data.index)
            
        elif method == 'market_cap':
            # 市值加权
            if 'market_cap' not in data.columns:
                logger.warning("数据中没有市值列，无法进行市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            # 计算市值权重
            market_cap = data['market_cap']
            total_market_cap = market_cap.sum()
            
            if total_market_cap == 0:
                logger.warning("总市值为0，无法进行市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            return market_cap / total_market_cap
            
        elif method == 'float_market_cap':
            # 流通市值加权
            if 'float_market_cap' not in data.columns:
                logger.warning("数据中没有流通市值列，无法进行流通市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            # 计算流通市值权重
            float_market_cap = data['float_market_cap']
            total_float_market_cap = float_market_cap.sum()
            
            if total_float_market_cap == 0:
                logger.warning("总流通市值为0，无法进行流通市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            return float_market_cap / total_float_market_cap
            
        else:
            logger.warning(f"未知的权重计算方法 {method}，将使用等权重")
            return pd.Series(1.0 / len(data), index=data.index)
            
    def _calculate_weighted_average(self, data, column, weights):
        """计算加权平均值
        
        Args:
            data: 包含数据的DataFrame
            column: 要计算的列名
            weights: 权重列名
            
        Returns:
            加权平均值
        """
        if data.empty:
            return 0.0
        
        # 移除任何权重为NaN或无穷的行
        valid_data = data.copy()
        valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna(subset=[weights, column])
        
        if valid_data.empty:
            return 0.0
            
        # 计算加权平均
        try:
            result = np.average(valid_data[column], weights=valid_data[weights])
            # 四舍五入到两位小数
            return round(result, 2)
        except Exception as e:
            logger.error(f"计算加权平均值时出错: {e}")
            return 0.0
        
    def _calculate_basic_features(self, sector_data, date, stocks):
        """
        计算板块的基本特征
        
        Args:
            sector_data: 板块股票的行情数据
            date: 日期
            stocks: 板块的股票列表
            
        Returns:
            包含基本特征的DataFrame
        """
        if sector_data is None or sector_data.empty:
            logger.warning(f"日期 {date} 没有有效数据")
            return None
            
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in required_columns:
            if col not in sector_data.columns:
                logger.error(f"数据缺少必要的列: {col}")
                return None
        
        # 筛选出有效的数据（收盘价大于0）
        valid_data = sector_data[sector_data['close'] > 0].copy()
        
        if valid_data.empty:
            logger.warning(f"日期 {date} 没有有效数据（收盘价>0）")
            return None
            
        # 如果没有change_percent列，尝试计算
        if 'change_percent' not in valid_data.columns:
            logger.info(f"数据源中没有change_percent列，将计算该值")
            try:
                # 获取前一个交易日的数据来计算涨跌幅
                prev_date = self._get_previous_trading_date(date)
                if prev_date:
                    logger.info(f"获取到前一个交易日: {prev_date}")
                    prev_data = self._load_sector_stocks(stocks, prev_date)
                    if prev_data is not None and not prev_data.empty:
                        # 创建前一日收盘价的映射
                        prev_close_map = dict(zip(prev_data['stock_code'], prev_data['close']))
                        
                        # 添加前一日收盘价
                        valid_data['prev_close'] = valid_data['stock_code'].map(prev_close_map)
                        
                        # 计算涨跌幅，并处理缺失值
                        valid_data['change_percent'] = valid_data.apply(
                            lambda x: round((x['close'] / x['prev_close'] - 1) * 100, 2) 
                            if pd.notna(x.get('prev_close', None)) and x.get('prev_close', 0) > 0 
                            else 0, 
                            axis=1
                        )
                        logger.info(f"成功计算涨跌幅，样例: {valid_data['change_percent'].head()}")
                    else:
                        logger.warning(f"无法获取前一交易日数据，将使用默认涨跌幅0")
                        valid_data['change_percent'] = 0
                else:
                    logger.warning(f"无法获取前一交易日，将使用默认涨跌幅0")
                    valid_data['change_percent'] = 0
            except Exception as e:
                logger.error(f"计算涨跌幅时出错: {e}")
                valid_data['change_percent'] = 0
        
        # 计算其他可能缺失的列
        if 'turnover_rate' not in valid_data.columns:
            valid_data['turnover_rate'] = 0  # 使用默认值
            
        if 'amplitude' not in valid_data.columns:
            # 振幅 = (最高价 - 最低价) / 收盘价 * 100%
            valid_data['amplitude'] = ((valid_data['high'] - valid_data['low']) / valid_data['close'] * 100).round(2)
            
        # 计算特征
        features = {}
        
        # 日期
        features['date'] = date
        features['date_str'] = date.strftime('%Y%m%d') if isinstance(date, (pd.Timestamp, datetime)) else str(date)
        
        # 股票数量
        features['stock_count'] = len(valid_data)
        
        # 使用valid_data计算平均值，保留两位小数
        features['close'] = round(valid_data['close'].mean(), 2)
        features['open'] = round(valid_data['open'].mean(), 2)
        features['high'] = round(valid_data['high'].mean(), 2)
        features['low'] = round(valid_data['low'].mean(), 2)
        features['volume'] = round(valid_data['volume'].mean(), 2)
        features['amount'] = round(valid_data['amount'].mean(), 2)
        features['change_percent'] = round(valid_data['change_percent'].mean(), 2)
        
        if 'turnover_rate' in valid_data.columns:
            features['turnover_rate'] = round(valid_data['turnover_rate'].mean(), 2)
        else:
            features['turnover_rate'] = 0
            
        if 'amplitude' in valid_data.columns:
            features['amplitude'] = round(valid_data['amplitude'].mean(), 2)
        else:
            features['amplitude'] = round(((valid_data['high'] - valid_data['low']) / valid_data['close'] * 100).mean(), 2)
        
        # 计算波动性指标
        features['volatility'] = round(valid_data['change_percent'].std(), 2)
        
        # 计算涨跌家数
        features['up_count'] = len(valid_data[valid_data['change_percent'] > 0])
        features['down_count'] = len(valid_data[valid_data['change_percent'] < 0])
        features['flat_count'] = len(valid_data[valid_data['change_percent'] == 0])
        
        # 涨停和跌停计数 (假设涨跌停为9.8%)
        features['limit_up_count'] = len(valid_data[valid_data['change_percent'] >= 9.8])
        features['limit_down_count'] = len(valid_data[valid_data['change_percent'] <= -9.8])
        
        # 创建DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df
        
    def _calculate_technical_features(self, sector_data, date, features, historical_data=None):
        """计算技术指标特征
        
        利用技术指标计算类，计算各种技术指标，如MA、MACD、RSI等
        增加了高级特征计算，更适合机器学习模型训练
        
        Args:
            sector_data: 板块股票的当日行情数据
            date: 当前日期
            features: 包含基本特征的DataFrame
            historical_data: 历史数据DataFrame，包含足够的历史记录用于计算指标
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 如果没有历史数据，无法计算技术指标
        if sector_data is None or sector_data.empty:
            return features
            
        # 使用板块的均价数据计算技术指标
        try:
            # 创建技术指标计算器
            indicator = TechnicalIndicators()
            
            # 检查历史数据是否足够
            if historical_data is None or len(historical_data) < 60:
                logger.warning(f"历史数据不足，无法准确计算技术指标，使用默认值")
                # 使用默认值
                ma_values = {
                    'ma5': features['close'].values[0],
                    'ma10': features['close'].values[0],
                    'ma20': features['close'].values[0],
                    'ma30': features['close'].values[0],
                    'ma60': features['close'].values[0]
                }
                
                # 添加MA值到特征中
                for ma_key, ma_value in ma_values.items():
                    features[ma_key] = ma_value
                    
                return features
            
            # 使用历史数据计算技术指标
            logger.debug(f"使用 {len(historical_data)} 天的历史数据计算技术指标")
            
            # 获取历史数据
            historical_close = historical_data['close'].values
            historical_open = historical_data['open'].values if 'open' in historical_data.columns else None
            historical_high = historical_data['high'].values if 'high' in historical_data.columns else None
            historical_low = historical_data['low'].values if 'low' in historical_data.columns else None
            historical_volume = historical_data['volume'].values if 'volume' in historical_data.columns else None
            
            # 1. 基础均线指标
            features['ma5'] = round(self._calculate_ma(historical_close, 5)[-1], 2)
            features['ma10'] = round(self._calculate_ma(historical_close, 10)[-1], 2)
            features['ma20'] = round(self._calculate_ma(historical_close, 20)[-1], 2)
            features['ma30'] = round(self._calculate_ma(historical_close, 30)[-1], 2)
            features['ma60'] = round(self._calculate_ma(historical_close, 60)[-1], 2)
            features['ma120'] = round(self._calculate_ma(historical_close, 120)[-1], 2) if len(historical_close) >= 120 else features['ma60']
            
            # 成交量均线
            if historical_volume is not None:
                features['vol_ma5'] = round(self._calculate_ma(historical_volume, 5)[-1], 2)
                features['vol_ma10'] = round(self._calculate_ma(historical_volume, 10)[-1], 2)
                features['vol_ma20'] = round(self._calculate_ma(historical_volume, 20)[-1], 2)
            
            # 2. 高级均线指标 - 均线关系指标
            features['ma5_10_ratio'] = round(features['ma5'] / features['ma10'], 4) if features['ma10'] > 0 else 1.0
            features['ma5_20_ratio'] = round(features['ma5'] / features['ma20'], 4) if features['ma20'] > 0 else 1.0
            features['ma10_20_ratio'] = round(features['ma10'] / features['ma20'], 4) if features['ma20'] > 0 else 1.0
            features['ma20_60_ratio'] = round(features['ma20'] / features['ma60'], 4) if features['ma60'] > 0 else 1.0
            features['ma60_120_ratio'] = round(features['ma60'] / features['ma120'], 4) if features['ma120'] > 0 else 1.0
            
            # 当前价格与均线关系
            current_close = features['close'].values[0]
            features['price_ma5_ratio'] = round(current_close / features['ma5'], 4) if features['ma5'] > 0 else 1.0
            features['price_ma20_ratio'] = round(current_close / features['ma20'], 4) if features['ma20'] > 0 else 1.0
            features['price_ma60_ratio'] = round(current_close / features['ma60'], 4) if features['ma60'] > 0 else 1.0
            
            # 2.1 均线斜率特征 - 衡量均线的变化速度
            if len(historical_close) >= 25:
                ma5_series = self._calculate_ma(historical_close, 5)
                ma20_series = self._calculate_ma(historical_close, 20)
                # 计算最近5天均线的斜率
                features['ma5_slope'] = round((ma5_series[-1] - ma5_series[-6]) / ma5_series[-6] * 100, 2) if ma5_series[-6] > 0 else 0
                features['ma20_slope'] = round((ma20_series[-1] - ma20_series[-6]) / ma20_series[-6] * 100, 2) if ma20_series[-6] > 0 else 0

            # 3. 趋势指标 - 均线多头/空头排列
            features['ma_bull_trend'] = 1 if (features['ma5'] > features['ma10'] > features['ma20'] > features['ma60']) else 0
            features['ma_bear_trend'] = 1 if (features['ma5'] < features['ma10'] < features['ma20'] < features['ma60']) else 0
            features['ma_neutral_trend'] = 1 if (not features['ma_bull_trend'] and not features['ma_bear_trend']) else 0
            
            # 3.1 价格趋势方向 - 根据价格与关键均线的位置判断趋势
            features['price_above_ma5'] = 1 if current_close > features['ma5'] else 0
            features['price_above_ma20'] = 1 if current_close > features['ma20'] else 0
            features['price_above_ma60'] = 1 if current_close > features['ma60'] else 0
            features['trend_direction_score'] = features['price_above_ma5'] + features['price_above_ma20'] + features['price_above_ma60']  # 0-3分，越高表示上升趋势越明显
            
            # 4. 动量指标
            # 计算MACD
            if len(historical_close) >= 26:
                macd, signal, hist = self._calculate_macd(historical_close)
                features['macd'] = round(macd[-1], 4)
                features['macd_signal'] = round(signal[-1], 4)
                features['macd_hist'] = round(hist[-1], 4)
                
                # MACD趋势
                if len(hist) > 5:
                    features['macd_hist_diff'] = round(hist[-1] - hist[-6], 4)  # 5日MACD柱状图变化
                    features['macd_hist_slope'] = round(np.mean(np.diff(hist[-5:])), 4)  # MACD柱状图斜率
                    
                    # MACD零线穿越判断
                    features['macd_cross_up'] = 1 if (hist[-2] <= 0 and hist[-1] > 0) else 0
                    features['macd_cross_down'] = 1 if (hist[-2] >= 0 and hist[-1] < 0) else 0
                    
                    # MACD能量指标 - 柱状图能量比率
                    positive_energy = np.sum(np.where(hist[-20:] > 0, hist[-20:], 0))
                    negative_energy = np.sum(np.where(hist[-20:] < 0, -hist[-20:], 0))
                    
                    if negative_energy > 0:
                        features['macd_energy_ratio'] = round(positive_energy / negative_energy, 4)
                    else:
                        features['macd_energy_ratio'] = 10.0  # 没有负能量时设置一个较大值
                        
                    # MACD背离探测 (简化版)
                    price_higher_high = historical_close[-1] > np.max(historical_close[-10:-1])
                    macd_higher_high = hist[-1] > np.max(hist[-10:-1])
                    features['macd_bullish_divergence'] = 1 if (not price_higher_high and macd_higher_high) else 0
                    
                    price_lower_low = historical_close[-1] < np.min(historical_close[-10:-1])
                    macd_lower_low = hist[-1] < np.min(hist[-10:-1])
                    features['macd_bearish_divergence'] = 1 if (price_lower_low and not macd_lower_low) else 0
            
            # 4.1 CCI指标 - 顺势指标
            if historical_high is not None and historical_low is not None and len(historical_close) >= 20:
                typical_price = (historical_high + historical_low + historical_close) / 3
                cci = self._calculate_cci(typical_price, 20)
                features['cci'] = round(cci[-1], 2)
                
                # CCI状态
                features['cci_overbought'] = 1 if cci[-1] > 100 else 0
                features['cci_oversold'] = 1 if cci[-1] < -100 else 0
                features['cci_extreme_overbought'] = 1 if cci[-1] > 200 else 0
                features['cci_extreme_oversold'] = 1 if cci[-1] < -200 else 0
            
            # 5. RSI指标
            if len(historical_close) >= 14:
                rsi6 = self._calculate_rsi(historical_close, 6)
                rsi14 = self._calculate_rsi(historical_close, 14)
                
                features['rsi6'] = round(rsi6[-1], 2)
                features['rsi14'] = round(rsi14[-1], 2)
                
                # RSI超买超卖
                features['rsi_overbought'] = 1 if rsi14[-1] > 70 else 0
                features['rsi_oversold'] = 1 if rsi14[-1] < 30 else 0
                
                # RSI趋势
                if len(rsi14) > 5:
                    features['rsi_slope'] = round(np.mean(np.diff(rsi14[-5:])), 4)  # RSI斜率
                    
                    # RSI背离(简化版)
                    price_higher_high = historical_close[-1] > np.max(historical_close[-10:-1])
                    rsi_higher_high = rsi14[-1] > np.max(rsi14[-10:-1])
                    features['rsi_bearish_divergence'] = 1 if (price_higher_high and not rsi_higher_high) else 0
                
                # RSI与MACD协同指标
                if 'macd_hist' in features:
                    features['rsi_macd_bull_confirm'] = 1 if (rsi14[-1] > 50 and features['macd_hist'] > 0) else 0
                    features['rsi_macd_bear_confirm'] = 1 if (rsi14[-1] < 50 and features['macd_hist'] < 0) else 0
            
            # 6. 波动率指标
            if historical_high is not None and historical_low is not None:
                # ATR - 平均真实波幅
                if len(historical_close) >= 14:
                    atr = self._calculate_atr(historical_high, historical_low, historical_close, 14)
                    features['atr'] = round(atr[-1], 4)
                    # ATR占比
                    features['atr_percent'] = round(atr[-1] / current_close * 100, 2) if current_close > 0 else 0
                    
                    # ATR波动率变化 - 衡量波动率的放大或收缩
                    if len(atr) > 5 and atr[-6] > 0:
                        features['atr_change_ratio'] = round((atr[-1] / atr[-6] - 1) * 100, 2)
                
                # 波动率
                returns = np.diff(np.log(historical_close))
                if len(returns) >= 20:
                    features['volatility_20d'] = round(np.std(returns[-20:]) * np.sqrt(252) * 100, 2)  # 年化波动率
                    
                    # 计算更多时间窗口的波动率
                    if len(returns) >= 60:
                        features['volatility_60d'] = round(np.std(returns[-60:]) * np.sqrt(252) * 100, 2)
                        # 波动率比率 - 短期vs长期
                        if features['volatility_60d'] > 0:
                            features['volatility_ratio'] = round(features['volatility_20d'] / features['volatility_60d'], 2)
                            
                    # 波动率分位数 - 当前波动率在历史波动率分布中的位置
                    if len(returns) >= 120:
                        rolling_vol = [np.std(returns[i-20:i]) * np.sqrt(252) * 100 for i in range(20, len(returns))]
                        current_vol = features['volatility_20d']
                        vol_rank = sum(v < current_vol for v in rolling_vol) / len(rolling_vol)
                        features['volatility_rank'] = round(vol_rank, 2)
            
            # 7. 布林带指标
            if len(historical_close) >= 20:
                upper, middle, lower = self._calculate_bollinger_bands(historical_close, 20, 2)
                features['bb_upper'] = round(upper[-1], 2)
                features['bb_middle'] = round(middle[-1], 2)
                features['bb_lower'] = round(lower[-1], 2)
                
                # 计算布林带宽度和位置指标
                if middle[-1] > 0:
                    features['bb_width'] = round((upper[-1] - lower[-1]) / middle[-1], 4)
                    features['bb_position'] = round((current_close - lower[-1]) / (upper[-1] - lower[-1]), 4) if (upper[-1] - lower[-1]) > 0 else 0.5
                    
                    # 布林带信号
                    features['bb_upper_break'] = 1 if current_close > upper[-1] else 0
                    features['bb_lower_break'] = 1 if current_close < lower[-1] else 0
                    
                    # 布林带宽度变化 - 衡量市场波动性的扩张或收缩
                    if len(upper) > 20 and middle[-21] > 0:
                        bb_width_past = (upper[-21] - lower[-21]) / middle[-21]
                        features['bb_width_change'] = round((features['bb_width'] / bb_width_past - 1) * 100, 2)
                        
                    # 布林带压力/支撑比率 - 衡量当前价格所受压力与支撑的关系
                    upside_room = (upper[-1] - current_close) / current_close
                    downside_room = (current_close - lower[-1]) / current_close
                    if downside_room > 0:
                        features['bb_pressure_support_ratio'] = round(upside_room / downside_room, 2)
                    else:
                        features['bb_pressure_support_ratio'] = 10.0  # 下方空间极小时设置较大值
            
            # 8. KDJ随机指标
            if historical_high is not None and historical_low is not None:
                if len(historical_close) >= 9:
                    k, d, j = self._calculate_kdj(historical_high, historical_low, historical_close)
                    features['kdj_k'] = round(k[-1], 2)
                    features['kdj_d'] = round(d[-1], 2)
                    features['kdj_j'] = round(j[-1], 2)
                    
                    # KDJ信号
                    features['kdj_golden_cross'] = 1 if (k[-2] <= d[-2] and k[-1] > d[-1]) else 0
                    features['kdj_dead_cross'] = 1 if (k[-2] >= d[-2] and k[-1] < d[-1]) else 0
                    
                    # KDJ超买超卖
                    features['kdj_overbought'] = 1 if k[-1] > 80 else 0
                    features['kdj_oversold'] = 1 if k[-1] < 20 else 0
                    
                    # J线值域分析 - 极值信号
                    features['kdj_j_extreme_high'] = 1 if j[-1] > 100 else 0
                    features['kdj_j_extreme_low'] = 1 if j[-1] < 0 else 0
            
            # 9. 板块轮动特征
            # 相对强度指标 - 计算最近20天的涨幅与最近60天的涨幅比
            if len(historical_close) >= 60:
                gain_20d = (historical_close[-1] / historical_close[-20]) - 1 if historical_close[-20] > 0 else 0
                gain_60d = (historical_close[-1] / historical_close[-60]) - 1 if historical_close[-60] > 0 else 0
                features['gain_20d'] = round(gain_20d * 100, 2)  # 转换为百分比
                features['gain_60d'] = round(gain_60d * 100, 2)  # 转换为百分比
                
                # 短期与长期涨幅比 - 板块轮动核心指标
                features['momentum_ratio'] = round(gain_20d / gain_60d, 4) if abs(gain_60d) > 0.001 else 0
                
                # 计算最近5天、10天、20天的收益率
                features['return_5d'] = round((historical_close[-1] / historical_close[-5] - 1) * 100, 2) if historical_close[-5] > 0 else 0
                features['return_10d'] = round((historical_close[-1] / historical_close[-10] - 1) * 100, 2) if historical_close[-10] > 0 else 0
                features['return_20d'] = round((historical_close[-1] / historical_close[-20] - 1) * 100, 2) if historical_close[-20] > 0 else 0
                
                # 计算收益率加速度 - 衡量动量变化
                features['return_acceleration'] = features['return_5d'] - (features['return_20d'] / 4)  # 5日收益-平均日收益
                
                # 收益率排名百分位 - 当前收益在历史收益分布中的位置
                if len(historical_close) >= 120:
                    rolling_returns = [(historical_close[i] / historical_close[i-20] - 1) * 100 for i in range(20, len(historical_close))]
                    current_return = features['return_20d']
                    return_rank = sum(r < current_return for r in rolling_returns) / len(rolling_returns)
                    features['return_20d_rank'] = round(return_rank, 2)
                    
                # 计算Alpha - 超额收益能力(简化版，假设市场基准收益为0)
                for period in [5, 20, 60]:
                    if len(historical_close) >= period:
                        sector_return = (historical_close[-1] / historical_close[-period] - 1) * 100
                        features[f'alpha_{period}d'] = round(sector_return, 2)  # 简化版Alpha
            
            # 10. 成交量分析指标
            if historical_volume is not None:
                # 成交量变化率
                avg_vol_5 = np.mean(historical_volume[-5:])
                avg_vol_20 = np.mean(historical_volume[-20:]) if len(historical_volume) >= 20 else avg_vol_5
                
                features['volume_change_5d'] = round(historical_volume[-1] / avg_vol_5 - 1, 4) if avg_vol_5 > 0 else 0
                features['volume_change_20d'] = round(historical_volume[-1] / avg_vol_20 - 1, 4) if avg_vol_20 > 0 else 0
                
                # 成交量比率
                features['volume_ratio_5_20'] = round(avg_vol_5 / avg_vol_20, 4) if avg_vol_20 > 0 else 1
                
                # 量价背离指标
                if len(historical_close) >= 5 and len(historical_volume) >= 5:
                    price_up = historical_close[-1] > historical_close[-5]
                    volume_up = historical_volume[-1] > historical_volume[-5]
                    
                    features['vol_price_confirm'] = 1 if (price_up and volume_up) or (not price_up and not volume_up) else 0
                    features['vol_price_diverge'] = 1 if (price_up and not volume_up) or (not price_up and volume_up) else 0
                    
                # 成交量爆发指标 - 检测成交量突然放大
                if len(historical_volume) >= 20:
                    vol_std = np.std(historical_volume[-20:])
                    vol_mean = np.mean(historical_volume[-20:])
                    
                    if vol_std > 0 and vol_mean > 0:
                        vol_z_score = (historical_volume[-1] - vol_mean) / vol_std
                        features['volume_surge'] = 1 if vol_z_score > 2.0 else 0  # 当日成交量超过正常波动2个标准差
                        features['volume_z_score'] = round(vol_z_score, 2)  # 成交量Z得分
                
                # 成交量均衡指数 - 衡量买入和卖出压力的平衡(简化版)
                if len(historical_close) >= 20 and len(historical_volume) >= 20:
                    up_volume = 0
                    down_volume = 0
                    
                    for i in range(1, 20):
                        idx = -i
                        prev_idx = -(i+1)
                        
                        if historical_close[idx] > historical_close[prev_idx]:
                            up_volume += historical_volume[idx]
                        else:
                            down_volume += historical_volume[idx]
                    
                    if down_volume > 0:
                        features['money_flow_ratio'] = round(up_volume / down_volume, 2)
                    else:
                        features['money_flow_ratio'] = 10.0  # 下跌成交量极小时设置较大值
            
            # 11. 趋势强度指标
            if len(historical_close) >= 20:
                # 趋势强度 - 最近N天收盘价高于/低于均线的天数比例
                days_above_ma20 = sum(historical_close[-20:] > self._calculate_ma(historical_close, 20)[-20:])
                features['trend_strength'] = round(days_above_ma20 / 20, 4)  # 0-1之间，越接近1表示上升趋势越强
                
                # 趋势一致性 - 连续上涨或下跌天数
                up_days = 0
                for i in range(1, min(11, len(historical_close))):
                    if historical_close[-i] > historical_close[-(i+1)]:
                        up_days += 1
                    else:
                        break
                        
                down_days = 0
                for i in range(1, min(11, len(historical_close))):
                    if historical_close[-i] < historical_close[-(i+1)]:
                        down_days += 1
                    else:
                        break
                
                features['consecutive_up_days'] = up_days
                features['consecutive_down_days'] = down_days
                
                # 趋势稳定性 - 价格振幅与总体涨幅的比值
                if len(historical_close) >= 20 and historical_close[-20] > 0:
                    total_gain = abs(historical_close[-1] / historical_close[-20] - 1)
                    
                    if historical_high is not None and historical_low is not None:
                        highest = np.max(historical_high[-20:])
                        lowest = np.min(historical_low[-20:])
                        total_range = (highest - lowest) / lowest
                        
                        if total_range > 0:
                            features['trend_efficiency'] = round(total_gain / total_range, 2)  # 趋势效率，越高越好
                
                # 趋势分形分析 - 计算最近N天的高点和低点数
                if len(historical_close) >= 10:
                    swing_high_count = 0
                    swing_low_count = 0
                    
                    for i in range(2, 8):  # 检查倒数第2到第8个数据点
                        idx = -i
                        # 检测摆动高点
                        if (historical_close[idx] > historical_close[idx-1] and 
                            historical_close[idx] > historical_close[idx+1]):
                            swing_high_count += 1
                        # 检测摆动低点
                        if (historical_close[idx] < historical_close[idx-1] and 
                            historical_close[idx] < historical_close[idx+1]):
                            swing_low_count += 1
                    
                    features['swing_high_count'] = swing_high_count
                    features['swing_low_count'] = swing_low_count
                    features['swing_pattern'] = 1 if swing_high_count > swing_low_count else (-1 if swing_low_count > swing_high_count else 0)
            
            # 12. 综合技术评分
            # 基于多个技术指标计算综合评分 (0-100分)
            tech_score = 50  # 中性起点
            score_factors = 0  # 计算了多少个因子
            
            # 均线多头排列加分
            if features.get('ma_bull_trend', 0) == 1:
                tech_score += 10
                score_factors += 1
            elif features.get('ma_bear_trend', 0) == 1:
                tech_score -= 10
                score_factors += 1
                
            # MACD柱状图方向
            if 'macd_hist' in features and 'macd_hist_diff' in features:
                if features['macd_hist'] > 0 and features['macd_hist_diff'] > 0:
                    tech_score += 10
                    score_factors += 1
                elif features['macd_hist'] < 0 and features['macd_hist_diff'] < 0:
                    tech_score -= 10
                    score_factors += 1
            
            # RSI方向
            if 'rsi14' in features and 'rsi_slope' in features:
                if features['rsi14'] > 50 and features['rsi_slope'] > 0:
                    tech_score += 10
                    score_factors += 1
                elif features['rsi14'] < 50 and features['rsi_slope'] < 0:
                    tech_score -= 10
                    score_factors += 1
                    
            # 布林带位置
            if 'bb_position' in features:
                if features['bb_position'] > 0.8:
                    tech_score += 5
                    score_factors += 0.5
                elif features['bb_position'] < 0.2:
                    tech_score -= 5
                    score_factors += 0.5
            
            # 趋势强度
            if 'trend_strength' in features:
                # 将0-1的趋势强度映射到-10到+10的得分
                trend_score = (features['trend_strength'] - 0.5) * 20
                tech_score += trend_score
                score_factors += 1
            
            # 量价关系
            if 'vol_price_confirm' in features and features['vol_price_confirm'] == 1:
                if 'return_5d' in features and features['return_5d'] > 0:
                    tech_score += 5  # 价量配合上涨
                    score_factors += 0.5
                elif 'return_5d' in features and features['return_5d'] < 0:
                    tech_score -= 5  # 价量配合下跌
                    score_factors += 0.5
            
            # KDJ指标
            if 'kdj_golden_cross' in features and features['kdj_golden_cross'] == 1:
                tech_score += 5
                score_factors += 0.5
            elif 'kdj_dead_cross' in features and features['kdj_dead_cross'] == 1:
                tech_score -= 5
                score_factors += 0.5
            
            # 波动率贡献
            if 'volatility_rank' in features:
                # 低波动率在牛市中加分，高波动率在熊市中加分
                if features['volatility_rank'] < 0.3 and features.get('return_20d', 0) > 0:
                    tech_score += 5  # 低波动率上涨，稳健上升趋势
                    score_factors += 0.5
                elif features['volatility_rank'] > 0.7 and features.get('return_20d', 0) < 0:
                    tech_score -= 5  # 高波动率下跌，剧烈下跌趋势
                    score_factors += 0.5
            
            # 动量因子
            if 'momentum_ratio' in features:
                if features['momentum_ratio'] > 1.5:  # 短期动能强于长期
                    tech_score += 10
                    score_factors += 1
                elif features['momentum_ratio'] < 0.5 and features['momentum_ratio'] > 0:  # 短期动能弱于长期
                    tech_score -= 10
                    score_factors += 1
            
            # 计算平均分，确保不同情况下的可比性
            if score_factors > 0:
                tech_score = 50 + (tech_score - 50) * (5 / score_factors)  # 标准化为基于5个因子的得分
            
            # 限制分数范围
            tech_score = max(0, min(100, tech_score))
            features['tech_score'] = round(tech_score, 0)
            
            # 13. 市场情绪指标
            features['market_sentiment'] = 'bullish' if tech_score > 65 else ('bearish' if tech_score < 35 else 'neutral')
            
            # 14. 交易信号强度
            signal_strength = 0
            
            # 技术指标一致性 - 多指标共振信号
            bullish_signals = 0
            bearish_signals = 0
            
            # 计算看涨信号数量
            for signal in ['ma_bull_trend', 'price_above_ma20', 'macd_cross_up', 'rsi_oversold', 
                         'bb_lower_break', 'kdj_golden_cross']:
                if features.get(signal, 0) == 1:
                    bullish_signals += 1
            
            # 计算看跌信号数量
            for signal in ['ma_bear_trend', 'price_above_ma20', 'macd_cross_down', 'rsi_overbought', 
                         'bb_upper_break', 'kdj_dead_cross']:
                if features.get(signal, 0) == 1:
                    bearish_signals += 1
            
            # 根据信号数量设置强度
            total_signals = 6  # 总共检查的信号数量
            bull_percent = bullish_signals / total_signals
            bear_percent = bearish_signals / total_signals
            
            if bull_percent > 0.5:
                signal_strength = round(bull_percent * 100)  # 0-100之间的看涨信号强度
            elif bear_percent > 0.5:
                signal_strength = -round(bear_percent * 100)  # 0-(-100)之间的看跌信号强度
            else:
                signal_strength = round((bull_percent - bear_percent) * 100)  # 模糊区域
            
            features['signal_strength'] = signal_strength
            
            return features
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return features
            
    def _calculate_ma(self, data, n):
        """计算移动平均线
        
        Args:
            data: 数据数组
            n: 周期
            
        Returns:
            移动平均线数组
        """
        if len(data) < n:
            # 数据不足，返回全为0的数组
            return np.zeros_like(data)
            
        # 使用卷积计算移动平均
        weights = np.ones(n) / n
        ma = np.convolve(data, weights, mode='full')[:len(data)]
        
        # 开始的n-1个值不准确，设为NaN
        ma[:n-1] = np.nan
        
        return ma
            
    def generate_sector_features_for_date(self, sector, date, preloaded_data=None):
        """计算指定日期的板块特征
        
        Args:
            sector: 板块名称
            date: 日期
            preloaded_data: 预加载的数据，如果提供，则不会再次加载数据
            
        Returns:
            tuple: (成功标志, 特征DataFrame)，如果失败则第二个元素为None
        """
        try:
            # 转换日期为pandas datetime
            if isinstance(date, str):
                if len(date) == 8 and date.isdigit():  # 判断是否为YYYYMMDD格式
                    date_obj = pd.to_datetime(
                        datetime.strptime(date, '%Y%m%d')
                    )
                else:
                    date_obj = pd.to_datetime(date)
            else:
                date_obj = pd.to_datetime(date)
                
            # 获取板块股票
            stocks = self._get_sector_stocks(sector)
            if not stocks:
                logger.warning(f"板块 {sector} 没有成分股")
                return False, None
                
            # 加载当日数据
            date_str = date_obj.strftime('%Y%m%d')
            if preloaded_data is not None:
                sector_data = preloaded_data
            else:
                # 加载这个板块所有股票在指定日期的数据
                sector_data = self._load_sector_stocks(stocks, date_str)
                
            if sector_data is None or sector_data.empty:
                logger.warning(f"日期 {date} 没有数据，板块: {sector}")
                return False, None
                
            # 过滤有效数据
            sector_data = sector_data.replace([np.inf, -np.inf], np.nan)
            sector_data = sector_data.dropna(subset=['close'])
            
            if sector_data.empty:
                logger.warning(f"日期 {date} 没有有效数据，板块: {sector}")
                return False, None
            
            # 加载历史数据用于计算技术指标(获取前180天数据)
            historical_data = None
            try:
                history_start_date = (date_obj - timedelta(days=180)).strftime('%Y%m%d')
                # 获取历史数据
                history_end_date = date_str
                logger.debug(f"正在加载历史数据，从 {history_start_date} 到 {history_end_date}")
                
                # 获取历史交易日列表
                trading_days = self.data_provider.get_trading_days(history_start_date, history_end_date)
                if isinstance(trading_days, pd.DataFrame):
                    if 'date' in trading_days.columns:
                        trading_days = trading_days['date'].tolist()
                    else:
                        trading_days = trading_days.index.tolist()
                
                # 确保交易日按时间排序
                trading_days = sorted(trading_days)
                
                # 收集历史日期的基本特征数据
                historical_data_list = []
                
                for day in trading_days:
                    # 转换日期格式
                    if isinstance(day, pd.Timestamp):
                        day_str = day.strftime('%Y%m%d')
                    else:
                        day_str = str(day)
                        
                    # 加载数据
                    day_data = self._load_sector_stocks(stocks, day_str)
                    
                    if day_data is not None and not day_data.empty:
                        # 计算基本特征
                        success, day_features = self._calculate_basic_features(day_data, day, stocks)
                        
                        if success and day_features is not None and not day_features.empty:
                            # 只保留需要的列
                            keep_cols = ['date', 'close', 'open', 'high', 'low', 'volume']
                            keep_features = {col: day_features[col].values[0] for col in keep_cols if col in day_features.columns}
                            historical_data_list.append(keep_features)
                
                # 转换为DataFrame
                if historical_data_list:
                    historical_data = pd.DataFrame(historical_data_list)
                    # 确保日期列的类型一致
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                    # 按日期排序
                    historical_data = historical_data.sort_values('date')
                    logger.info(f"成功加载 {len(historical_data)} 天的历史数据")
                else:
                    logger.warning(f"没有获取到历史数据")
            
            except Exception as e:
                logger.error(f"加载历史数据时出错: {str(e)}")
                logger.error(traceback.format_exc())
            
            # 计算基本特征
            try:
                basic_features = self._calculate_basic_features(sector_data, date_obj, stocks)
                
                if basic_features is None:
                    logger.error(f"为行业 {sector} 日期 {date} 生成特征时出错")
                    return False, None
                    
                # 转换为Series以兼容其他功能（如果是DataFrame）
                if isinstance(basic_features, pd.DataFrame):
                    if not basic_features.empty:
                        # 使用第一行数据
                        basic_features_row = basic_features.iloc[0]
                    else:
                        logger.warning(f"基本特征DataFrame为空，板块: {sector}, 日期: {date}")
                        return False, None
                else:
                    # 已经是Series
                    basic_features_row = basic_features
                
                # 添加板块名称
                basic_features_row['sector'] = sector
                
                # 确保date列的类型一致
                if 'date' in basic_features_row:
                    basic_features_row['date'] = pd.to_datetime(basic_features_row['date'])
                
                # 创建返回的DataFrame
                features_df = pd.DataFrame([basic_features_row])
                
                # 计算技术指标（传入历史数据）
                features_df = self._calculate_technical_features(sector_data, date_obj, features_df, historical_data)
                
                # 将所有可能的浮点数特征转换为最多两位小数
                numeric_cols = features_df.select_dtypes(include=['float', 'int']).columns
                for col in numeric_cols:
                    features_df[col] = features_df[col].round(2)
                
                return True, features_df
                
            except Exception as e:
                logger.error(f"为行业 {sector} 日期 {date} 生成特征时出错: {str(e)}")
                logger.error(traceback.format_exc())
                return False, None
                
        except Exception as e:
            logger.error(f"为行业 {sector} 日期 {date} 生成特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
            
    def generate_and_save_features(self, sector, start_date, end_date):
        """生成并保存板块特征
        
        Args:
            sector: 板块名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            表示操作是否成功的布尔值
        """
        try:
            logger.info(f"正在生成板块 {sector} 从 {start_date} 到 {end_date} 的特征数据")
            
            # 获取交易日列表
            trading_days = self.data_provider.get_trading_days(start_date, end_date)
            
            # 确保trading_days是列表类型
            if isinstance(trading_days, pd.DataFrame):
                if 'date' in trading_days.columns:
                    trading_days = trading_days['date'].tolist()
                else:
                    trading_days = trading_days.index.tolist()
            
            if not trading_days:
                logger.warning(f"从 {start_date} 到 {end_date} 没有交易日")
                return False
                
            # 创建一个空的DataFrame来存储所有日期的特征
            all_features = []
            
            # 对每个交易日生成特征
            for day in trading_days:
                # 转换日期格式
                if isinstance(day, pd.Timestamp):
                    date_str = day.strftime('%Y%m%d')
                else:
                    date_str = str(day)
                    
                logger.debug(f"正在计算板块 {sector} 在日期 {date_str} 的特征")
                
                # 生成特征并保存
                success, features = self.generate_sector_features_for_date(sector, date_str)
                
                if success and features is not None and not features.empty:
                    all_features.append(features)
                else:
                    logger.warning(f"板块 {sector} 在日期 {date_str} 没有生成有效特征")
            
            if not all_features:
                logger.warning(f"板块 {sector} 在日期范围 {start_date} - {end_date} 没有生成任何特征")
                return False
                
            # 合并所有日期的特征
            features_df = pd.concat(all_features, ignore_index=True)
            
            if features_df.empty:
                logger.warning(f"板块 {sector} 在日期范围 {start_date} - {end_date} 没有有效特征")
                return False
                
            # 对所有列做类型转换和四舍五入
            for col in features_df.columns:
                if col not in ['date', 'sector', 'date_str']:
                    try:
                        # 尝试转换为浮点数并四舍五入到两位小数
                        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                        if not pd.isna(features_df[col]).all():
                            features_df[col] = features_df[col].round(2)
                    except Exception as e:
                        logger.warning(f"无法将列 {col} 转换为数值型: {e}")
            
            # 按日期排序
            if 'date' in features_df.columns:
                features_df = features_df.sort_values('date')
                
            # 保存特征
            self._save_features(sector, features_df)
            
            logger.info(f"成功为板块 {sector} 生成并保存了 {len(features_df)} 天的特征数据")
            return True
            
        except Exception as e:
            logger.error(f"为板块 {sector} 生成特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _save_features(self, sector, features_df):
        """保存行业特征数据
        
        Args:
            sector: 行业代码
            features_df: 特征DataFrame
            
        Returns:
            None
        """
        try:
            # 确保输出目录存在
            sector_dir = os.path.join(self.output_dir, sector)
            os.makedirs(sector_dir, exist_ok=True)
            
            # 确保所有浮点数列都四舍五入到两位小数
            numeric_cols = features_df.select_dtypes(include=['float64']).columns
            for col in numeric_cols:
                features_df[col] = features_df[col].round(2)
            
            # 保存为Parquet文件
            parquet_path = os.path.join(sector_dir, f"{sector}_features.parquet")
            features_df.to_parquet(parquet_path, index=False)
            logger.info(f"已保存行业 {sector} 特征数据到 {parquet_path}，共 {len(features_df)} 行")
            
            # 保存最新特征为CSV文件
            if not features_df.empty:
                # 确保日期列为正确格式
                if 'date' in features_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                        features_df['date'] = pd.to_datetime(features_df['date'])
                    
                # 获取最新日期的数据
                latest_date = features_df['date'].max()
                latest_features = features_df[features_df['date'] == latest_date].copy()
                
                if not latest_features.empty:
                    latest_csv_path = os.path.join(sector_dir, f"{sector}_latest.csv")
                    latest_features.to_csv(latest_csv_path, index=False, encoding='utf-8')
                    logger.info(f"已保存行业 {sector} 最新日期 {latest_date.strftime('%Y-%m-%d')} 的特征数据到 {latest_csv_path}")
        
        except Exception as e:
            logger.error(f"保存行业 {sector} 特征数据时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _get_previous_trading_date(self, date):
        """
        获取指定日期的前一个交易日
        
        Args:
            date: 当前日期，可以是字符串、datetime或Timestamp
            
        Returns:
            前一个交易日，格式与输入相同
        """
        try:
            # 统一转换为datetime对象进行处理
            if isinstance(date, str):
                if len(date) == 8 and date.isdigit():  # YYYYMMDD格式
                    date_obj = datetime.strptime(date, '%Y%m%d')
                else:
                    # 尝试解析其他格式
                    try:
                        date_obj = datetime.strptime(date, '%Y-%m-%d')
                    except ValueError:
                        try:
                            date_obj = pd.to_datetime(date).to_pydatetime()
                        except:
                            logger.error(f"无法解析日期格式: {date}")
                            return None
            elif isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            elif isinstance(date, datetime):
                date_obj = date
            else:
                logger.error(f"不支持的日期类型: {type(date)}")
                return None
                
            # 获取所有交易日
            try:
                trading_days = self.data_provider.get_trading_days(
                    start_date=(date_obj - timedelta(days=30)).strftime('%Y%m%d'),
                    end_date=date_obj.strftime('%Y%m%d')
                )
                
                # 确保trading_days是列表或数组
                if isinstance(trading_days, pd.DataFrame):
                    if 'date' in trading_days.columns:
                        trading_days = trading_days['date'].tolist()
                    else:
                        trading_days = list(trading_days.index)
                
                # 排序交易日
                trading_days = sorted(trading_days)
                
                # 将date_obj转换为相同格式进行比较
                date_str = date_obj.strftime('%Y%m%d')
                
                # 查找当前日期的索引
                for i, day in enumerate(trading_days):
                    # 转换交易日为字符串格式以便比较
                    day_str = day.strftime('%Y%m%d') if hasattr(day, 'strftime') else str(day)
                    if day_str == date_str:
                        # 找到当前日期，返回前一个交易日
                        if i > 0:
                            prev_day = trading_days[i-1]
                            
                            # 与输入类型保持一致
                            if isinstance(date, str):
                                if hasattr(prev_day, 'strftime'):
                                    return prev_day.strftime('%Y%m%d')
                                else:
                                    return str(prev_day)
                            else:
                                return prev_day
                        else:
                            logger.warning(f"日期 {date} 是第一个交易日，没有前一个交易日")
                            return None
                
                logger.warning(f"在交易日历中找不到日期 {date}")
                return None
                
            except Exception as e:
                logger.error(f"获取交易日历时出错: {e}")
                return None
                
        except Exception as e:
            logger.error(f"获取前一个交易日时出错: {e}")
            return None
            
    def _load_sector_stocks(self, stocks, date):
        """
        加载指定日期的板块股票数据
        
        Args:
            stocks: 股票列表
            date: 日期
            
        Returns:
            包含股票数据的DataFrame
        """
        try:
            # 将date转换为字符串格式
            if isinstance(date, (datetime, pd.Timestamp)):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)
                
            # 调用data_provider获取数据
            # 注意：使用 stocks 而不是 stock_codes 作为参数名
            data = self.data_provider.get_daily_data(
                stocks,
                date_str,
                date_str
            )
            
            if data is not None and not data.empty:
                return data
            else:
                logger.warning(f"日期 {date_str} 的数据为空")
                return None
                
        except Exception as e:
            logger.error(f"加载板块股票数据时出错: {e}")
            return None

    def _calculate_macd(self, data, fast=12, slow=26, signal=9):
        """计算MACD指标
        
        Args:
            data: 价格数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            tuple: (MACD, 信号线, 柱状图)
        """
        # 计算快线和慢线的EMA
        ema_fast = self._calculate_ema(data, fast)
        ema_slow = self._calculate_ema(data, slow)
        
        # 计算MACD线 = 快线EMA - 慢线EMA
        macd = ema_fast - ema_slow
        
        # 计算信号线 = MACD的EMA
        signal_line = self._calculate_ema(macd, signal)
        
        # 计算柱状图 = MACD线 - 信号线
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
        
    def _calculate_ema(self, data, n):
        """计算指数移动平均线
        
        Args:
            data: 数据数组
            n: 周期
            
        Returns:
            EMA数组
        """
        if len(data) < n:
            return np.zeros_like(data)
            
        ema = np.zeros_like(data)
        # 初始值为简单平均
        ema[n-1] = np.mean(data[:n])
        
        # 计算后续值
        multiplier = 2 / (n + 1)
        for i in range(n, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
            
        # 设置前n-1个值为NaN
        ema[:n-1] = np.nan
        
        return ema
        
    def _calculate_rsi(self, data, n=14):
        """计算相对强弱指数（RSI）
        
        Args:
            data: 价格数据
            n: 周期
            
        Returns:
            RSI数组
        """
        if len(data) <= n:
            return np.zeros_like(data)
            
        # 计算价格变化
        deltas = np.diff(data)
        deltas = np.append([0], deltas)  # 添加一个0作为第一个元素，保持数组长度一致
        
        # 分离上涨和下跌
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        # 初始化平均值
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        
        # 第一个值是简单平均
        avg_gain[n] = np.mean(gain[1:n+1])
        avg_loss[n] = np.mean(loss[1:n+1])
        
        # 计算后续值
        for i in range(n+1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (n-1) + gain[i]) / n
            avg_loss[i] = (avg_loss[i-1] * (n-1) + loss[i]) / n
        
        # 计算RS和RSI
        rs = np.zeros_like(data)
        rsi = np.zeros_like(data)
        
        for i in range(n, len(data)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
        
        # 设置前n个值为NaN
        rsi[:n] = np.nan
        
        return rsi
        
    def _calculate_bollinger_bands(self, data, n=20, k=2):
        """计算布林带
        
        Args:
            data: 价格数据
            n: 周期
            k: 标准差倍数
            
        Returns:
            tuple: (上轨, 中轨, 下轨)
        """
        if len(data) < n:
            return np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)
            
        # 计算中轨（简单移动平均线）
        ma = self._calculate_ma(data, n)
        
        # 计算标准差
        std = np.zeros_like(data)
        for i in range(n-1, len(data)):
            std[i] = np.std(data[i-n+1:i+1])
        
        # 计算上下轨
        upper = ma + k * std
        lower = ma - k * std
        
        return upper, ma, lower
        
    def _calculate_kdj(self, high, low, close, n=9, m1=3, m2=3):
        """计算KDJ随机指标
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            n: 周期
            m1: K值平滑因子
            m2: D值平滑因子
            
        Returns:
            tuple: (K, D, J)
        """
        if len(close) < n:
            return np.zeros_like(close), np.zeros_like(close), np.zeros_like(close)
            
        # 初始化数组
        k = np.zeros_like(close)
        d = np.zeros_like(close)
        j = np.zeros_like(close)
        
        # 计算RSV值
        rsv = np.zeros_like(close)
        
        for i in range(n-1, len(close)):
            period_low = np.min(low[i-n+1:i+1])
            period_high = np.max(high[i-n+1:i+1])
            
            if period_high - period_low > 0:
                rsv[i] = 100 * (close[i] - period_low) / (period_high - period_low)
            else:
                rsv[i] = 50  # 如果最高价等于最低价，取中间值
        
        # 计算K值（RSV的m1日移动平均）
        k[n-1] = rsv[n-1]
        for i in range(n, len(close)):
            k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
        
        # 计算D值（K的m2日移动平均）
        d[n-1] = k[n-1]
        for i in range(n, len(close)):
            d[i] = (2/3) * d[i-1] + (1/3) * k[i]
        
        # 计算J值（3*K-2*D）
        for i in range(n-1, len(close)):
            j[i] = 3 * k[i] - 2 * d[i]
        
        # 设置前n-1个值为NaN
        k[:n-1] = np.nan
        d[:n-1] = np.nan
        j[:n-1] = np.nan
        
        return k, d, j
        
    def _calculate_atr(self, high, low, close, n=14):
        """计算平均真实波幅（ATR）
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            n: 周期
            
        Returns:
            ATR数组
        """
        if len(close) <= 1:
            return np.zeros_like(close)
            
        # 计算真实波幅（TR）
        tr = np.zeros_like(close)
        
        # 第一个TR就是当天的价格范围
        tr[0] = high[0] - low[0]
        
        # 计算剩余的TR值
        for i in range(1, len(close)):
            # TR是以下三者的最大值：
            # 1. 当日最高价 - 当日最低价
            # 2. |当日最高价 - 昨日收盘价|
            # 3. |当日最低价 - 昨日收盘价|
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # 计算ATR
        atr = np.zeros_like(close)
        
        # 第一个ATR是前n天TR的简单平均
        if len(tr) >= n:
            atr[n-1] = np.mean(tr[:n])
            
            # 使用Wilder平滑法计算后续的ATR值
            for i in range(n, len(close)):
                atr[i] = (atr[i-1] * (n-1) + tr[i]) / n
        
        # 设置前n-1个值为NaN
        atr[:n-1] = np.nan
        
        return atr

    def _calculate_cci(self, typical_price, n=20):
        """计算商品通道指数（CCI）
        
        Args:
            typical_price: 典型价格，通常是 (high + low + close) / 3
            n: 周期
            
        Returns:
            CCI数组
        """
        if len(typical_price) < n:
            return np.zeros_like(typical_price)
            
        # 初始化CCI数组
        cci = np.zeros_like(typical_price)
        
        # 计算移动平均和绝对偏差
        for i in range(n-1, len(typical_price)):
            # 计算周期内典型价格的简单移动平均
            sma = np.mean(typical_price[i-n+1:i+1])
            
            # 计算平均绝对偏差
            mad = np.mean(np.abs(typical_price[i-n+1:i+1] - sma))
            
            # 计算CCI
            if mad > 0:
                cci[i] = (typical_price[i] - sma) / (0.015 * mad)
            else:
                cci[i] = 0  # 避免除以零
        
        # 设置前n-1个值为NaN
        cci[:n-1] = np.nan
        
        return cci