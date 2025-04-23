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
from pathlib import Path

# 导入logger配置
from to.logger_config import configure_logger

# 使用共享的logger配置
logger = configure_logger(__name__)

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

class SectorFeatureGenerator:
    """板块特征生成器
    
    生成板块的各种基础特征，包括价格均值、成交量、成交额、活跃股票数等。
    """
    
    def __init__(self, data_provider, sector_mapping, output_dir=None, 
                weighting_method='equal'):
        """初始化特征生成器
        
        Args:
            data_provider: 数据提供者，用于获取股票数据
            sector_mapping: 板块股票映射关系
            output_dir: 输出目录
            weighting_method: 加权方法，默认为equal
        """
        self.data_provider = data_provider
        self.sector_mapping = sector_mapping
        self.weighting_method = weighting_method
        self.output_dir = output_dir
        
        # 添加历史特征缓存
        self._historical_features_cache = {}  # {sector: {date_str: feature_dict}}
        self._cache_max_size = 250  # 大约一年的交易日数量
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(get_data_path()) / 'sector_features'
            
        ensure_dir(self.output_dir)
        
        logger.info(f"初始化板块特征生成器，权重方法: {weighting_method}, 输出目录: {self.output_dir}")
        
    def _get_sector_stocks(self, sector):
        """获取板块成分股
        
        Args:
            sector: 板块名称
            
        Returns:
            股票列表
        """
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

        # 优先使用pct_chg计算平均涨跌幅
        if 'pct_chg' in valid_data.columns:
            logger.info(f"使用数据源中的pct_chg列计算平均涨跌幅")
            valid_data['change_percent'] = valid_data['pct_chg']
        # 如果没有pct_chg列，尝试计算change_percent
        elif 'change_percent' not in valid_data.columns:
            logger.info(f"数据源中没有pct_chg和change_percent列，将计算涨跌幅")
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
        
        # # 计算其他可能缺失的列
        # if 'turnover_rate' not in valid_data.columns:
        #     valid_data['turnover_rate'] = 0  # 使用默认值
            
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
        
        # 添加平均换手率
        if 'turnover_rate' in valid_data.columns:
            features['turnover_rate'] = round(valid_data['turnover_rate'].mean(), 2)
            
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
        
        # 涨停和跌停计数 (假设涨跌停为9.8%) enhanced_feature_generator.py 中计算了
        # features['limit_up_count'] = len(valid_data[valid_data['change_percent'] >= 9.8])
        # features['limit_down_count'] = len(valid_data[valid_data['change_percent'] <= -9.8])
        
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
        try:
            logger.info(f"计算技术指标开始")
            # 如果没有历史数据，无法计算技术指标
            if sector_data is None or sector_data.empty:
                logger.info(f"板块数据为空，无法计算技术指标")
                return features
                
            # 使用板块的均价数据计算技术指标
            # 创建技术指标计算器
            indicator = TechnicalIndicators()
            
            # 检查历史数据是否足够
            if historical_data is None or len(historical_data) < 20:  # 修改为至少需要20天数据
                logger.info(f"历史数据不足，无法准确计算技术指标，使用默认值")
                # 使用默认值
                default_close = features['close'] if isinstance(features['close'], (int, float, np.number)) else features['close'].iloc[0] if not features['close'].empty else 0.0
                
                ma_values = {
                    'ma5': default_close,
                    'ma10': default_close,
                    'ma20': default_close,
                    'ma30': default_close,
                    'ma60': default_close,
                    'ma120': default_close,
                    'rsi6': 50,
                    'rsi14': 50,
                    'rsi_overbought': 0,
                    'rsi_oversold': 0,
                    'rsi_neutral': 1,
                    'ema12': default_close,
                    'macd_line': 0,
                    'macd_signal': 0,
                    'macd_histogram': 0,
                    'macd_trend': 0,
                    'trend_strength': 0.5,
                    'swing_pattern': 0,
                    'consecutive_up_days': 0,
                    'consecutive_down_days': 0,
                    'volume_ratio_5': 1.0,
                    'volume_ratio_10': 1.0,
                    'volume_ratio_20': 1.0,
                    'volume_ratio_60': 1.0
                }
                
                # 批量添加值到特征中
                for key, value in ma_values.items():
                    features[key] = value

                logger.info(f"计算技术指标完成，返回默认特征")
                return features
            
            # 获取历史数据并确保是numpy数组
            logger.info(f"使用 {len(historical_data)} 天的历史数据计算技术指标")
            
            # 预处理历史数据，确保正确的格式
            historical_close = np.array(historical_data['close'].values) if 'close' in historical_data.columns else None
            historical_open = np.array(historical_data['open'].values) if 'open' in historical_data.columns else None
            historical_high = np.array(historical_data['high'].values) if 'high' in historical_data.columns else None
            historical_low = np.array(historical_data['low'].values) if 'low' in historical_data.columns else None
            historical_volume = np.array(historical_data['volume'].values) if 'volume' in historical_data.columns else None
            
            # 获取当前价格，确保是标量值
            current_close = features['close']
            if isinstance(current_close, pd.Series):
                current_close = current_close.iloc[0] if not current_close.empty else 0.0
            
            # 异常值检查 - 确保没有NaN或无穷值
            if historical_close is not None:
                historical_close = np.nan_to_num(historical_close, nan=current_close)
            if historical_volume is not None:
                historical_volume = np.nan_to_num(historical_volume, nan=0.0)
            
            # 批量计算多个周期的移动平均线
            ma_periods = [5, 10, 20, 30, 60, 120]
            ma_values = {}
            
            for period in ma_periods:
                if historical_close is not None and len(historical_close) >= period:
                    ma = np.mean(historical_close[-period:])
                    ma_values[f'ma{period}'] = round(ma, 2)
                else:
                    ma_values[f'ma{period}'] = current_close
            
            # 批量添加移动平均线到特征中
            for key, value in ma_values.items():
                features[key] = value
            
            # 提取MA值作为变量，确保是标量值
            ma5 = ma_values['ma5']
            ma10 = ma_values['ma10']
            ma20 = ma_values['ma20']
            ma60 = ma_values['ma60']
            ma120 = ma_values['ma120']
            
            # 计算均线比率，确保分母大于0
            features['ma5_10_ratio'] = round(ma5 / ma10, 4) if ma10 > 0 else 1.0
            features['ma5_20_ratio'] = round(ma5 / ma20, 4) if ma20 > 0 else 1.0
            features['ma10_20_ratio'] = round(ma10 / ma20, 4) if ma20 > 0 else 1.0
            features['ma20_60_ratio'] = round(ma20 / ma60, 4) if ma60 > 0 else 1.0
            features['ma60_120_ratio'] = round(ma60 / ma120, 4) if ma120 > 0 else 1.0
            
            # 当前价格与均线比率
            features['price_ma5_ratio'] = round(current_close / ma5, 4) if ma5 > 0 else 1.0
            features['price_ma20_ratio'] = round(current_close / ma20, 4) if ma20 > 0 else 1.0
            features['price_ma60_ratio'] = round(current_close / ma60, 4) if ma60 > 0 else 1.0
            
            # 计算价格趋势方向指标 - 使用标量值比较
            features['price_above_ma5'] = 1 if current_close > ma5 else 0
            features['price_above_ma20'] = 1 if current_close > ma20 else 0
            features['price_above_ma60'] = 1 if current_close > ma60 else 0
            
            # 计算趋势方向得分 - 显式转换为标量
            above_ma5 = features['price_above_ma5']
            above_ma20 = features['price_above_ma20']
            above_ma60 = features['price_above_ma60']
            
            if isinstance(above_ma5, pd.Series):
                above_ma5 = above_ma5.iloc[0] if not above_ma5.empty else 0
            if isinstance(above_ma20, pd.Series):
                above_ma20 = above_ma20.iloc[0] if not above_ma20.empty else 0
            if isinstance(above_ma60, pd.Series):
                above_ma60 = above_ma60.iloc[0] if not above_ma60.empty else 0
                
            features['trend_direction_score'] = above_ma5 + above_ma20 + above_ma60
            
            # 趋势指标 - 均线多头/空头排列
            features['ma_bull_trend'] = 1 if (ma5 > ma10 > ma20 > ma60) else 0
            features['ma_bear_trend'] = 1 if (ma5 < ma10 < ma20 < ma60) else 0
            
            # 确保使用标量值进行比较，避免Series的布尔运算歧义
            bull_trend = features['ma_bull_trend']
            bear_trend = features['ma_bear_trend']
            
            if isinstance(bull_trend, pd.Series):
                bull_trend = bull_trend.iloc[0] if not bull_trend.empty else 0
            if isinstance(bear_trend, pd.Series):
                bear_trend = bear_trend.iloc[0] if not bear_trend.empty else 0
                
            features['ma_neutral_trend'] = 1 if (bull_trend == 0 and bear_trend == 0) else 0
            
            # 批量计算均线斜率
            if len(historical_close) >= 10:
                try:
                    # 使用向量化操作计算斜率
                    ma5_current = ma5
                    ma5_prev = np.mean(historical_close[-10:-5])
                    ma20_current = ma20
                    ma20_prev = np.mean(historical_close[-25:-5]) if len(historical_close) >= 25 else ma20
                    
                    features['ma5_slope'] = round((ma5_current - ma5_prev) / ma5_prev * 100, 2) if ma5_prev > 0 else 0
                    features['ma20_slope'] = round((ma20_current - ma20_prev) / ma20_prev * 100, 2) if ma20_prev > 0 else 0
                except Exception as e:
                    logger.error(f"计算均线斜率出错: {e}")
                    features['ma5_slope'] = 0
                    features['ma20_slope'] = 0
            else:
                features['ma5_slope'] = 0
                features['ma20_slope'] = 0
            
            # 批量计算历史N天前收盘价
            for days in [5, 10, 20]:
                if len(historical_close) >= days:
                    features[f'close_{days}d_ago'] = historical_close[-days]
                else:
                    features[f'close_{days}d_ago'] = current_close
            
            # 计算EMA12
            if len(historical_close) >= 12:
                ema12 = self._calculate_ema(historical_close, 12)
                features['ema12'] = round(ema12, 2)
            else:
                features['ema12'] = current_close
            
            # 计算动量指标 - 确保使用标量值
            for days in [5, 10, 20]:
                close_nd_ago = features[f'close_{days}d_ago']
                # 确保close_nd_ago是标量
                if isinstance(close_nd_ago, pd.Series):
                    close_nd_ago = close_nd_ago.iloc[0] if not close_nd_ago.empty else 0
                
                # 使用转换后的标量值进行比较和计算
                features[f'momentum_{days}d'] = round(((current_close / close_nd_ago) - 1) * 100, 2) if close_nd_ago > 0 else 0
            
            # 计算相对变化率 - 确保使用标量值
            ema12_val = features['ema12']
            if isinstance(ema12_val, pd.Series):
                ema12_val = ema12_val.iloc[0] if not ema12_val.empty else current_close
                
            features['change_pct'] = round(((current_close / ema12_val) - 1) * 100, 2) if ema12_val > 0 else 0
            
            # 计算RSI指标
            if len(historical_close) >= 14:
                try:
                    # 使用向量化操作计算RSI
                    deltas = np.diff(historical_close[-15:])
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    # RSI6
                    if len(gains) >= 6 and len(losses) >= 6:
                        avg_gain_6 = np.mean(gains[-6:])
                        avg_loss_6 = np.mean(losses[-6:])
                        rs_6 = avg_gain_6 / avg_loss_6 if avg_loss_6 > 0 else 9999
                        rsi6 = 100.0 - (100.0 / (1.0 + rs_6))
                        features['rsi6'] = round(rsi6, 2)
                    else:
                        features['rsi6'] = 50
                    
                    # RSI14
                    avg_gain_14 = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                    avg_loss_14 = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                    rs_14 = avg_gain_14 / avg_loss_14 if avg_loss_14 > 0 else 9999
                    rsi14 = 100.0 - (100.0 / (1.0 + rs_14))
                    features['rsi14'] = round(rsi14, 2)
                    
                    # RSI超买超卖指标 - 确保使用标量值
                    rsi14_val = features['rsi14']
                    if isinstance(rsi14_val, pd.Series):
                        rsi14_val = rsi14_val.iloc[0] if not rsi14_val.empty else 50
                    
                    if rsi14_val > 70:
                        features['rsi_overbought'] = 1
                        features['rsi_oversold'] = 0
                        features['rsi_neutral'] = 0
                    elif rsi14_val < 30:
                        features['rsi_overbought'] = 0
                        features['rsi_oversold'] = 1
                        features['rsi_neutral'] = 0
                    else:
                        features['rsi_overbought'] = 0
                        features['rsi_oversold'] = 0
                        features['rsi_neutral'] = 1
                except Exception as e:
                    logger.error(f"计算RSI指标出错: {e}")
                    features['rsi6'] = 50
                    features['rsi14'] = 50
                    features['rsi_overbought'] = 0
                    features['rsi_oversold'] = 0
                    features['rsi_neutral'] = 1
            else:
                features['rsi6'] = 50
                features['rsi14'] = 50
                features['rsi_overbought'] = 0
                features['rsi_oversold'] = 0
                features['rsi_neutral'] = 1
            
            # 计算MACD指标
            try:
                if len(historical_close) >= 26:
                    # 调用优化后的MACD函数
                    macd_line, signal_line, macd_histogram = self._calculate_macd(historical_close, 12, 26, 9)
                    
                    features['macd_line'] = round(macd_line, 4)
                    features['macd_signal'] = round(signal_line, 4)
                    features['macd_histogram'] = round(macd_histogram, 4)
                    
                    # MACD趋势判断 - 确保使用标量值
                    macd = macd_line
                    signal = signal_line
                    
                    if isinstance(macd, pd.Series):
                        macd = macd.iloc[0] if not macd.empty else 0
                    if isinstance(signal, pd.Series):
                        signal = signal.iloc[0] if not signal.empty else 0
                    
                    if macd > signal:
                        if macd > 0:
                            features['macd_trend'] = 2  # 强势上涨
                        else:
                            features['macd_trend'] = 1  # 弱势上涨
                    else:
                        if macd < 0:
                            features['macd_trend'] = -2  # 强势下跌
                        else:
                            features['macd_trend'] = -1  # 弱势下跌
                else:
                    features['macd_line'] = 0
                    features['macd_signal'] = 0
                    features['macd_histogram'] = 0
                    features['macd_trend'] = 0
            except Exception as e:
                logger.error(f"计算MACD指标时出错: {e}")
                features['macd_line'] = 0
                features['macd_signal'] = 0
                features['macd_histogram'] = 0
                features['macd_trend'] = 0
                
            # 计算波动率指标 - 向量化批量计算
            volatility_periods = [5, 10, 20, 60]
            for period in volatility_periods:
                try:
                    if len(historical_close) >= period + 1:
                        # 使用对数收益率
                        returns = np.diff(np.log(historical_close))[-period:]
                        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
                        features[f'volatility_{period}d'] = round(volatility * 100, 2)  # 转换为百分比
                    else:
                        features[f'volatility_{period}d'] = 0
                except Exception as e:
                    logger.error(f"计算波动率指标 {period}日 出错: {e}")
                    features[f'volatility_{period}d'] = 0
            
            # 波动率比率
            if 'volatility_20d' in features and 'volatility_60d' in features:
                # 处理Series类型
                if isinstance(features['volatility_20d'], pd.Series):
                    vol_20d = features['volatility_20d'].iloc[0] if len(features['volatility_20d']) > 0 else 0
                else:
                    vol_20d = features['volatility_20d']
                    
                if isinstance(features['volatility_60d'], pd.Series):
                    vol_60d = features['volatility_60d'].iloc[0] if len(features['volatility_60d']) > 0 else 0
                else:
                    vol_60d = features['volatility_60d']
                
                # 检查NaN值    
                vol_20d = 0 if pd.isna(vol_20d) else float(vol_20d)
                vol_60d = 0 if pd.isna(vol_60d) else float(vol_60d)
                    
                features['volatility_ratio'] = float(vol_20d / vol_60d) if vol_60d > 0 else 1.0
            else:
                features['volatility_ratio'] = 1.0
            
            # 计算成交量特征
            if historical_volume is not None and len(historical_volume) > 0:
                try:
                    # 获取当前成交量
                    current_volume = historical_volume[-1]
                    
                    # 向量化批量计算不同周期的平均成交量
                    vol_periods = [5, 10, 20, 60]
                    for period in vol_periods:
                        if len(historical_volume) >= period:
                            avg_vol = np.mean(historical_volume[-period:])
                            # 确保使用标量值
                            if isinstance(avg_vol, pd.Series):
                                avg_vol = avg_vol.iloc[0] if not avg_vol.empty else 1.0
                            ratio = round(current_volume / avg_vol, 2) if avg_vol > 0 else 1.0
                            features[f'volume_ratio_{period}'] = ratio
                        else:
                            features[f'volume_ratio_{period}'] = 1.0
                    
                    # 成交量变化率 - 确保使用标量值
                    if len(historical_volume) >= 2:
                        prev_volume = historical_volume[-2]
                        # 确保使用标量值
                        if isinstance(prev_volume, pd.Series):
                            prev_volume = prev_volume.iloc[0] if not prev_volume.empty else current_volume
                        features['volume_change_ratio'] = round((current_volume - prev_volume) / prev_volume, 2) if prev_volume > 0 else 0.0
                    else:
                        features['volume_change_ratio'] = 0.0
                    
                    # 成交量趋势 - 使用安全的索引方法
                    if len(historical_volume) >= 5:
                        try:
                            # 使用numpy操作前确保数据类型正确
                            volume_data = np.array(historical_volume[-5:])
                            vol_changes = np.sign(np.diff(volume_data))
                            features['volume_trend'] = np.sum(vol_changes)
                        except Exception as inner_e:
                            logger.error(f"计算成交量趋势时出错: {inner_e}")
                            # 退回到手动计算方法
                            volume_trend = 0
                            for i in range(1, min(5, len(historical_volume))):
                                curr_vol = historical_volume[-(i)]
                                prev_vol = historical_volume[-(i+1)]
                                if isinstance(curr_vol, pd.Series):
                                    curr_vol = curr_vol.iloc[0] if not curr_vol.empty else 0
                                if isinstance(prev_vol, pd.Series):
                                    prev_vol = prev_vol.iloc[0] if not prev_vol.empty else 0
                                
                                if curr_vol > prev_vol:
                                    volume_trend += 1
                                elif curr_vol < prev_vol:
                                    volume_trend -= 1
                            features['volume_trend'] = volume_trend
                    else:
                        features['volume_trend'] = 0
                    
                except Exception as e:
                    logger.error(f"计算成交量特征时出错: {e}")
                    for period in [5, 10, 20, 60]:
                        features[f'volume_{period}d_ratio'] = 1.0
                    features['volume_change_ratio'] = 0.0
                    features['volume_trend'] = 0
            else:
                # 默认值
                for period in [5, 10, 20, 60]:
                    features[f'volume_{period}d_ratio'] = 1.0
                features['volume_change_ratio'] = 0.0
                features['volume_trend'] = 0
            
            # 计算趋势强度指标
            try:
                if len(historical_close) >= 20:
                    # 使用向量化计算价格变化方向
                    price_changes = np.diff(historical_close[-20:])
                    up_days = np.sum(price_changes > 0)
                    down_days = np.sum(price_changes < 0)
                    
                    # 趋势强度
                    features['trend_strength'] = round(up_days / 19, 2)  # 19 = 20天中的变化次数
                    
                    # 连续上涨/下跌天数 - 确保使用标量值
                    consecutive_up = 0
                    consecutive_down = 0
                    
                    # 优化连续天数计算
                    for i in range(1, min(10, len(historical_close))):
                        if i >= len(historical_close) - 1:
                            break
                            
                        curr_price = historical_close[-i]
                        prev_price = historical_close[-(i+1)]
                        
                        # 确保使用标量值
                        if isinstance(curr_price, pd.Series):
                            curr_price = curr_price.iloc[0] if not curr_price.empty else 0
                        if isinstance(prev_price, pd.Series):
                            prev_price = prev_price.iloc[0] if not prev_price.empty else 0
                        
                        change = curr_price - prev_price
                        
                        if change > 0:
                            if consecutive_down == 0:  # 只有当没有连续下跌时才增加
                                consecutive_up += 1
                            else:
                                break  # 如果已经有下跌记录，则结束计数
                        elif change < 0:
                            if consecutive_up == 0:  # 只有当没有连续上涨时才增加
                                consecutive_down += 1
                            else:
                                break  # 如果已经有上涨记录，则结束计数
                        else:
                            break  # 价格不变，结束计数
                    
                    features['consecutive_up_days'] = consecutive_up
                    features['consecutive_down_days'] = consecutive_down
                    
                    # 摆动模式 - 优化计算
                    if len(historical_close) >= 10:
                        price_diffs = np.diff(historical_close[-10:])
                        signs = np.sign(price_diffs)
                        sign_changes = np.diff(signs)
                        features['swing_pattern'] = np.sum(np.abs(sign_changes) > 0)  # 计算符号变化的次数
                    else:
                        features['swing_pattern'] = 0
                else:
                    # 数据不足时设置默认值
                    features['trend_strength'] = 0.5
                    features['consecutive_up_days'] = 0
                    features['consecutive_down_days'] = 0
                    features['swing_pattern'] = 0
            except Exception as e:
                logger.error(f"计算趋势强度指标时出错: {e}")
                features['trend_strength'] = 0.5
                features['consecutive_up_days'] = 0
                features['consecutive_down_days'] = 0
                features['swing_pattern'] = 0
            
            # 计算价格支撑位和压力位
            if len(historical_close) >= 20:
                try:
                    support, resistance = self._calculate_support_resistance(historical_close, n=20)
                    features['support_level'] = round(support, 2)
                    features['resistance_level'] = round(resistance, 2)
                    
                    # 确保使用标量值进行比较和计算
                    support_val = support
                    resistance_val = resistance
                    
                    if isinstance(support_val, pd.Series):
                        support_val = support_val.iloc[0] if not support_val.empty else current_close * 0.95
                    if isinstance(resistance_val, pd.Series):
                        resistance_val = resistance_val.iloc[0] if not resistance_val.empty else current_close * 1.05
                    
                    features['support_distance_pct'] = round((current_close - support_val) / support_val * 100, 2) if support_val > 0 else 0
                    features['resistance_distance_pct'] = round((resistance_val - current_close) / current_close * 100, 2) if current_close > 0 else 0
                except Exception as e:
                    logger.error(f"计算价格支撑压力位出错: {e}")
                    features['support_level'] = current_close * 0.95
                    features['resistance_level'] = current_close * 1.05
                    features['support_distance_pct'] = 5.0
                    features['resistance_distance_pct'] = 5.0
            else:
                features['support_level'] = current_close * 0.95
                features['resistance_level'] = current_close * 1.05
                features['support_distance_pct'] = 5.0
                features['resistance_distance_pct'] = 5.0
                
            # 技术综合评分计算
            tech_score = 50  # 初始中性评分
            score_factors = 0  # 计入评分的因子数量
            
            # 按不同指标调整评分
            # 趋势指标
            bull_trend_val = bull_trend  # 使用前面已经提取的标量值
            bear_trend_val = bear_trend  # 使用前面已经提取的标量值
            
            if bull_trend_val == 1:
                tech_score += 15
                score_factors += 1
            elif bear_trend_val == 1:
                tech_score -= 15
                score_factors += 1
            
            # 价格位置 - 使用前面提取的标量值
            if above_ma5 == 1:
                tech_score += 5
                score_factors += 1
            if above_ma20 == 1:
                tech_score += 5
                score_factors += 1
            if above_ma60 == 1:
                tech_score += 5
                score_factors += 1
            
            # RSI - 使用前面提取的标量值
            rsi14_for_score = rsi14_val  # 使用前面已经提取的标量值
            
            if rsi14_for_score > 70:
                tech_score += 10  # 超买
                score_factors += 1
            elif rsi14_for_score < 30:
                tech_score -= 10  # 超卖
                score_factors += 1
            elif rsi14_for_score > 60:
                tech_score += 5   # 偏强
                score_factors += 1
            elif rsi14_for_score < 40:
                tech_score -= 5   # 偏弱
                score_factors += 1
            
            # MACD - 使用前面提取的标量值
            macd_histogram_val = features['macd_histogram']
            macd_trend_val = features['macd_trend']
            
            if isinstance(macd_histogram_val, pd.Series):
                macd_histogram_val = macd_histogram_val.iloc[0] if not macd_histogram_val.empty else 0
            if isinstance(macd_trend_val, pd.Series):
                macd_trend_val = macd_trend_val.iloc[0] if not macd_trend_val.empty else 0
            
            if macd_histogram_val > 0 and macd_trend_val > 0:
                tech_score += 10
                score_factors += 1
            elif macd_histogram_val < 0 and macd_trend_val < 0:
                tech_score -= 10
                score_factors += 1
            
            # 动量
            momentum_5d_val = features['momentum_5d']
            if isinstance(momentum_5d_val, pd.Series):
                momentum_5d_val = momentum_5d_val.iloc[0] if not momentum_5d_val.empty else 0
                
            if momentum_5d_val > 3:
                tech_score += 8
                score_factors += 1
            elif momentum_5d_val < -3:
                tech_score -= 8
                score_factors += 1
            
            # 连续天数
            consecutive_up_val = features['consecutive_up_days']
            consecutive_down_val = features['consecutive_down_days']
            
            if isinstance(consecutive_up_val, pd.Series):
                consecutive_up_val = consecutive_up_val.iloc[0] if not consecutive_up_val.empty else 0
            if isinstance(consecutive_down_val, pd.Series):
                consecutive_down_val = consecutive_down_val.iloc[0] if not consecutive_down_val.empty else 0
            
            if consecutive_up_val >= 3:
                tech_score += 5
                score_factors += 1
            elif consecutive_down_val >= 3:
                tech_score -= 5
                score_factors += 1
                
            # 计算最终技术评分，确保在0-100范围内
            if score_factors > 0:
                tech_score = min(max(tech_score, 0), 100)
            else:
                tech_score = 50  # 没有足够信息时保持中性
            
            features['tech_score'] = tech_score
            
            # 市场情绪 - 基于技术评分
            tech_score_val = tech_score  # 已经是标量值
            
            if tech_score_val >= 70:
                features['market_sentiment'] = 2  # 看涨
            elif tech_score_val <= 30:
                features['market_sentiment'] = 0  # 看跌
            else:
                features['market_sentiment'] = 1  # 中性
            
            # 交易信号强度
            signal_strength = 0
            
            # 基于技术评分 - 已经是标量值
            if tech_score_val >= 80:
                signal_strength += 3
            elif tech_score_val >= 65:
                signal_strength += 2
            elif tech_score_val >= 55:
                signal_strength += 1
            elif tech_score_val <= 20:
                signal_strength -= 3
            elif tech_score_val <= 35:
                signal_strength -= 2
            elif tech_score_val <= 45:
                signal_strength -= 1
            
            # 基于MACD
            macd_trend_for_signal = macd_trend_val  # 使用前面提取的标量值
            if macd_trend_for_signal != 0:
                signal_strength += macd_trend_for_signal  # -2到2的贡献
            
            # 基于RSI - 使用前面提取的标量值
            if rsi14_for_score > 80:
                signal_strength -= 1  # 严重超买
            elif rsi14_for_score < 20:
                signal_strength += 1  # 严重超卖
            
            features['signal_strength'] = signal_strength
            
            logger.info(f"计算技术指标完成")
            return features
        
        except Exception as e:
            logger.error(f"计算技术指标时发生错误: {e}")
            logger.error(traceback.format_exc())
            # 出错时返回原特征，不修改
            return features

    def generate_sector_features_for_date(self, sector, date, preloaded_data=None):
        """为指定板块和日期生成特征
        
        Args:
            sector: 板块名称
            date: 日期，可以是字符串或日期对象
            preloaded_data: 预加载的数据，包含历史数据
            
        Returns:
            (success, features_df): 成功标志和特征DataFrame
        """
        try:
            # 将日期转为字符串格式
            date_str = date if isinstance(date, str) else date.strftime('%Y%m%d')
            date_obj = pd.to_datetime(date)
            
            # 获取股票列表 - 无论是否使用预加载数据都需要获取
            stocks = self._get_sector_stocks(sector)
            if not stocks:
                logger.warning(f"板块 {sector} 没有成分股")
                return False, None
            
            if preloaded_data is not None:
                logger.debug("使用预加载的数据")
                sector_data = preloaded_data
                
                # 筛选当前日期的数据
                if 'date' in sector_data.columns:
                    # 将日期列统一为datetime格式，便于比较
                    if not pd.api.types.is_datetime64_dtype(sector_data['date']):
                        sector_data['date'] = pd.to_datetime(sector_data['date'])
                    
                    # 获取当前日期的数据
                    current_date_data = sector_data[sector_data['date'] == date_obj].copy()
                    
                    if current_date_data.empty:
                        logger.warning(f"预加载数据中找不到日期 {date_str} 的数据")
                        # 尝试按照字符串匹配
                        if 'date_str' in sector_data.columns:
                            current_date_data = sector_data[sector_data['date_str'] == date_str].copy()
                    
                    # 准备历史数据（过滤出日期小于等于当前日期的所有数据）
                    historical_data = sector_data[sector_data['date'] <= date_obj].copy()
                    
                    # 使用当前日期数据作为sector_data
                    if not current_date_data.empty:
                        sector_data = current_date_data
                    else:
                        logger.warning(f"无法从预加载数据中提取日期 {date_str} 的数据")
                        # 尝试加载股票数据
                        sector_data = self._load_sector_stocks(stocks, date_str)
            else:
                # 加载股票数据
                sector_data = self._load_sector_stocks(stocks, date_str) 
                # 没有预加载数据时，historical_data设为None
                historical_data = None
            
            if sector_data is None or sector_data.empty:
                logger.warning(f"板块 {sector} 在 {date_str} 没有数据")
                return False, None
            
            # 计算基本特征
            features_df = self._calculate_basic_features(sector_data, date_obj, stocks)
            
            if features_df is None or features_df.empty:
                logger.warning(f"板块 {sector} 在 {date_str} 无法计算基本特征")
                return False, None
                
            # 如果没有预加载的历史数据，按照原来的方式获取历史数据
            if preloaded_data is None or 'historical_data' not in locals() or historical_data is None or historical_data.empty:
                try:
                    # 获取前180天的日期范围
                    history_start_date = (date_obj - timedelta(days=180)).strftime('%Y%m%d')
                    # 获取历史数据
                    history_end_date = date_str
                    logger.info(f"正在加载历史数据，从 {history_start_date} 到 {history_end_date}")
                    
                    # 获取历史交易日列表
                    trading_days = self.data_provider.get_trading_days(history_start_date, history_end_date)
                    if isinstance(trading_days, pd.DataFrame):
                        if 'date' in trading_days.columns:
                            trading_days = trading_days['date'].tolist()
                        else:
                            trading_days = trading_days.index.tolist()
                    
                    # 确保交易日按时间排序
                    trading_days = sorted(trading_days)
                    
                    # 初始化板块缓存（如果不存在）
                    if sector not in self._historical_features_cache:
                        self._historical_features_cache[sector] = {}
                    
                    # 收集历史日期的基本特征数据，使用缓存
                    historical_data_list = []
                    days_to_calculate = []
                    
                    for day in trading_days:
                        # 转换日期格式
                        if isinstance(day, pd.Timestamp):
                            day_str = day.strftime('%Y%m%d')
                        else:
                            day_str = str(day)
                        
                        # 检查缓存中是否有该日期的数据
                        if day_str in self._historical_features_cache[sector]:
                            historical_data_list.append(self._historical_features_cache[sector][day_str])
                        else:
                            days_to_calculate.append((day, day_str))
                    
                    # 计算缺失的日期数据
                    for day, day_str in days_to_calculate:
                        # 加载数据
                        day_data = self._load_sector_stocks(stocks, day_str)
                        
                        if day_data is not None and not day_data.empty:
                            # 计算基本特征
                            day_features = self._calculate_basic_features(day_data, day, stocks)
                            
                            if day_features is not None and not day_features.empty:
                                # 只保留需要的列
                                keep_cols = ['date', 'close', 'open', 'high', 'low', 'volume']
                                keep_features = {col: day_features[col].values[0] for col in keep_cols if col in day_features.columns}
                                
                                # 添加到缓存
                                self._historical_features_cache[sector][day_str] = keep_features
                                historical_data_list.append(keep_features)
                    
                    # 转换为DataFrame
                    if historical_data_list:
                        historical_data = pd.DataFrame(historical_data_list)
                    else:
                        historical_data = pd.DataFrame()
                        
                    # 清理缓存，防止内存占用过大
                    self._trim_historical_cache(sector)
                    logger.info(f"历史数据缓存大小: {len(self._historical_features_cache[sector])}")
                    
                except Exception as e:
                    logger.error(f"计算历史数据时出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    # 如果获取历史数据失败，仍然返回基本特征
                    return True, features_df
            
            # 如果有历史数据，则处理为适合技术指标计算的格式
            if historical_data is not None and not historical_data.empty:
                # 提取计算技术指标所需的历史数据
                # 确保历史数据包含必要的列
                required_cols = ['date', 'close', 'open', 'high', 'low', 'volume']
                missing_cols = [col for col in required_cols if col not in historical_data.columns]
                
                if missing_cols:
                    logger.warning(f"历史数据缺少必要的列: {missing_cols}")
                    # 为缺失的列添加默认值
                    for col in missing_cols:
                        historical_data[col] = 0.0
                
                # 聚合数据，为每个日期计算板块的平均价格和成交量
                agg_functions = {
                    'close': 'mean',
                    'open': 'mean',
                    'high': 'mean',
                    'low': 'mean',
                    'volume': 'sum'
                }
                
                # 在聚合前确保date列的格式统一
                if not pd.api.types.is_datetime64_dtype(historical_data['date']):
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                
                # 按日期聚合
                try:
                    aggregated_data = historical_data.groupby('date').agg(agg_functions).reset_index()
                    logger.debug(f"聚合后的历史数据有 {len(aggregated_data)} 行")
                    
                    # 计算技术指标
                    features_df = self._calculate_technical_features(sector_data, date_obj, features_df, aggregated_data)
                except Exception as e:
                    logger.error(f"聚合历史数据或计算技术指标时出错: {e}")
                    logger.error(traceback.format_exc())
            else:
                # 无历史数据，仍然尝试计算技术指标（将使用默认值）
                features_df = self._calculate_technical_features(sector_data, date_obj, features_df)
            
            return True, features_df
                
        except Exception as e:
            logger.error(f"为板块 {sector} 在日期 {date} 生成特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _trim_historical_cache(self, sector):
        """控制历史数据缓存的大小，避免内存占用过大"""
        if sector in self._historical_features_cache:
            cache = self._historical_features_cache[sector]
            if len(cache) > self._cache_max_size:
                # 按日期排序，只保留最近的数据
                dates = sorted(cache.keys())
                dates_to_remove = dates[:-self._cache_max_size]
                for old_date in dates_to_remove:
                    del cache[old_date]
                logger.debug(f"清理了 {len(dates_to_remove)} 个旧缓存项，当前缓存大小：{len(cache)}")
                
    def clear_historical_cache(self):
        """清除所有历史数据缓存"""
        self._historical_features_cache = {}
        logger.info("已清除所有历史数据缓存")

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
        try:
            # 检查输入数据类型和长度
            if isinstance(data, (float, int, np.float64, np.int64)):
                # 如果输入是单个数值，返回默认值
                return 0.0, 0.0, 0.0
                
            if isinstance(data, pd.Series):
                data = data.values
                
            if len(data) < slow:
                # 数据不足时返回全零值
                return 0.0, 0.0, 0.0
                
            # 计算快线和慢线的EMA
            ema_fast = self._calculate_ema(data, fast)
            ema_slow = self._calculate_ema(data, slow)
            
            # 计算MACD线 = 快线EMA - 慢线EMA
            macd = ema_fast - ema_slow
            
            # 计算信号线 = MACD的EMA
            # 确保macd是数组类型
            if isinstance(macd, (float, int, np.float64, np.int64)):
                signal_line = macd * 0.2  # 近似计算，信号线通常滞后于MACD
                histogram = macd * 0.8    # 近似计算，柱状图
            else:
                signal_line = self._calculate_ema(np.array([macd]), signal)
                histogram = macd - signal_line
                
            return macd, signal_line, histogram
        except Exception as e:
            logger.error(f"计算MACD时出错: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_ema(self, data, window):
        """
        计算给定窗口大小的指数移动平均线(EMA)
        
        Args:
            data: 价格数据的numpy数组或pandas Series
            window: EMA的窗口大小
            
        Returns:
            float: 最新的EMA值
        """
        try:
            # 检查输入数据类型
            if isinstance(data, (float, int, np.float64, np.int64)):
                return data  # 如果是单个数值，直接返回
                
            if isinstance(data, pd.Series):
                data = data.values
                
            # 确保数据是数组类型
            if not isinstance(data, (list, np.ndarray)):
                data = np.array([data])
                
            if len(data) < window:
                return data[-1] if len(data) > 0 else 0  # 数据不足时返回最新价格或0
                
            # 计算权重系数
            alpha = 2 / (window + 1)
            
            # 初始EMA为简单平均
            start_idx = max(0, len(data) - (window*2))
            end_idx = max(start_idx + 1, len(data) - window)
            
            if start_idx == end_idx:  # 处理边界情况
                ema = data[start_idx]
            else:
                ema = np.mean(data[start_idx:end_idx])
            
            # 计算EMA
            for i in range(window):
                idx = len(data) - window + i
                if 0 <= idx < len(data):  # 确保索引有效
                    ema = alpha * data[idx] + (1 - alpha) * ema
                
            return ema
        except Exception as e:
            logger.error(f"计算EMA时出错: {e}")
            return 0.0
        
    def _calculate_rsi(self, data, n=14):
        """计算相对强弱指数（RSI）
        
        Args:
            data: 价格数据的numpy数组或pandas Series
            n: 周期
            
        Returns:
            float: 最新的RSI值
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        if len(data) <= n:
            return 50.0  # 数据不足时返回中性值
            
        # 计算价格变化
        deltas = np.diff(data[-n-1:])
        
        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均上涨和下跌
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # 计算RS和RSI
        if avg_loss == 0:
            return 100.0  # 避免除以零
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
        return round(rsi, 2)
        
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
            float: 最新的CCI值
        """
        if isinstance(typical_price, pd.Series):
            typical_price = typical_price.values
            
        if len(typical_price) < n:
            return 0.0  # 数据不足时返回中性值
            
        # 计算n周期的简单移动平均
        sma = np.mean(typical_price[-n:])
        
        # 计算n周期内典型价格的平均绝对偏差
        mad = np.mean(np.abs(typical_price[-n:] - sma))
        
        # 计算CCI
        if mad > 0:
            cci = (typical_price[-1] - sma) / (0.015 * mad)
        else:
            cci = 0  # 避免除以零
            
        return round(cci, 2)

    def _calculate_support_resistance(self, historical_data, n=20, sensitivity=1.5, volatility_adjust=True):
        """
        计算专业的支撑位和压力位
        
        参数:
            historical_data (DataFrame/Series/ndarray): 包含high, low, close的历史数据或仅close的历史数据
            n (int): 寻找局部极值的窗口大小
            sensitivity (float): 灵敏度系数，用于调整支撑/压力位的距离
            volatility_adjust (bool): 是否根据波动率调整支撑/压力位
            
        返回:
            tuple: (支撑位, 压力位)
        """
        # 检查输入类型并转换为DataFrame
        if isinstance(historical_data, np.ndarray):
            # 如果是numpy数组，假设它只包含close数据
            if historical_data.ndim == 1:
                historical_data = pd.DataFrame({'close': historical_data})
            else:
                # 如果是2D数组，假设按列顺序为high, low, close
                if historical_data.shape[1] >= 3:
                    historical_data = pd.DataFrame({
                        'high': historical_data[:, 0],
                        'low': historical_data[:, 1],
                        'close': historical_data[:, 2]
                    })
                else:
                    # 如果不足3列，只使用close
                    historical_data = pd.DataFrame({'close': historical_data[:, -1]})
        elif isinstance(historical_data, pd.Series):
            # 如果是Series，转换为只有close的DataFrame
            historical_data = pd.DataFrame({'close': historical_data})
        
        # 确保historical_data包含必要的列
        # 对于只有close的DataFrame，添加high和low列
        if 'high' not in historical_data.columns and 'close' in historical_data.columns:
            historical_data['high'] = historical_data['close']
        if 'low' not in historical_data.columns and 'close' in historical_data.columns:
            historical_data['low'] = historical_data['close']
        
        if historical_data is None or len(historical_data) < n + 2 or not all(col in historical_data.columns for col in ['high', 'low', 'close']):
            # 数据不足或列缺失，使用默认方法
            current_close = historical_data['close'].iloc[-1] if historical_data is not None and 'close' in historical_data.columns and len(historical_data) > 0 else 0
            return current_close * 0.95, current_close * 1.05
        
        # 提取相关数据
        high = historical_data['high'].values
        low = historical_data['low'].values
        close = historical_data['close'].values
        
        # 当前收盘价
        current_close = close[-1]
        
        try:
            # 计算ATR作为波动参考（多时间周期考虑）
            atr14 = self._calculate_atr(high, low, close, 14)[-1]
            atr21 = self._calculate_atr(high, low, close, 21)[-1] if len(close) >= 21 else atr14
            
            # 使用加权平均ATR增强稳定性
            atr = 0.7 * atr14 + 0.3 * atr21
            
            # 计算多个时间周期的支撑位和压力位候选点
            support_candidates = []
            resistance_candidates = []
            
            # 短期支撑压力位 (近10天)
            short_term_period = min(10, len(close) - 1)
            if short_term_period >= 5:
                recent_low = np.min(low[-short_term_period:])
                recent_high = np.max(high[-short_term_period:])
                if recent_low < current_close:
                    support_candidates.append((recent_low, 1.0))  # 短期支撑，权重1.0
                if recent_high > current_close:
                    resistance_candidates.append((recent_high, 1.0))  # 短期压力，权重1.0
            
            # 中期支撑压力位 (近30天局部极值)
            medium_term_period = min(30, len(close) - 1)
            if medium_term_period >= 10:
                # 使用滑动窗口找出局部极值
                window_size = 5  # 局部极值判断窗口
                
                for i in range(window_size, medium_term_period):
                    idx = len(close) - medium_term_period + i
                    # 检查是否为局部低点
                    if all(low[idx] <= low[idx-j] for j in range(1, window_size+1)) and all(low[idx] <= low[idx+j] for j in range(1, min(window_size, medium_term_period-i))):
                        if low[idx] < current_close and low[idx] > current_close * 0.75:
                            # 计算该点被触及次数（价格在该点附近波动次数）
                            touch_count = np.sum(np.abs(low - low[idx]) < atr * 0.3)
                            importance = touch_count / 5  # 根据触及次数确定重要性
                            support_candidates.append((low[idx], 1.5 * importance))  # 中期支撑，基础权重1.5
                    
                    # 检查是否为局部高点
                    if all(high[idx] >= high[idx-j] for j in range(1, window_size+1)) and all(high[idx] >= high[idx+j] for j in range(1, min(window_size, medium_term_period-i))):
                        if high[idx] > current_close and high[idx] < current_close * 1.25:
                            # 计算该点被触及次数
                            touch_count = np.sum(np.abs(high - high[idx]) < atr * 0.3)
                            importance = touch_count / 5  # 根据触及次数确定重要性
                            resistance_candidates.append((high[idx], 1.5 * importance))  # 中期压力，基础权重1.5
            
            # 长期支撑压力位 (全部历史数据中的主要水平)
            if len(close) >= 60:
                # 将价格区间分成若干区间，统计每个区间的价格出现频率
                price_min = np.min(low)
                price_max = np.max(high)
                
                if price_max > price_min:
                    # 创建10个价格区间
                    bin_count = 10
                    bins = np.linspace(price_min, price_max, bin_count + 1)
                    
                    # 统计收盘价在各区间的频率
                    hist, _ = np.histogram(close, bins=bins)
                    
                    # 找出频率较高的区间作为潜在的支撑/压力区
                    for i in range(bin_count):
                        bin_center = (bins[i] + bins[i+1]) / 2
                        if hist[i] > np.mean(hist) * 1.5:  # 高于平均频率的1.5倍
                            if bin_center < current_close and bin_center > current_close * 0.7:
                                support_candidates.append((bin_center, 2.0))  # 长期支撑，权重2.0
                            elif bin_center > current_close and bin_center < current_close * 1.3:
                                resistance_candidates.append((bin_center, 2.0))  # 长期压力，权重2.0
            
            # 如果没有找到支撑位或压力位，使用ATR计算
            if not support_candidates:
                support_level = current_close - atr * sensitivity
            else:
                # 计算加权平均支撑位
                total_weight = sum(weight for _, weight in support_candidates)
                support_level = sum(level * weight for level, weight in support_candidates) / total_weight
            
            if not resistance_candidates:
                resistance_level = current_close + atr * sensitivity
            else:
                # 计算加权平均压力位
                total_weight = sum(weight for _, weight in resistance_candidates)
                resistance_level = sum(level * weight for level, weight in resistance_candidates) / total_weight
            
            # 基于波动率的动态调整
            if volatility_adjust and not np.isnan(atr):
                price_volatility = atr / current_close
                vol_factor = min(max(price_volatility * 100, 1.0), 5.0)  # 限制波动率影响在1-5倍之间
                
                # 高波动率环境中，支撑位和压力位范围更大
                support_level = min(support_level, current_close * (1 - 0.02 * vol_factor))
                resistance_level = max(resistance_level, current_close * (1 + 0.02 * vol_factor))
            else:
                # 确保支撑位低于当前价格，压力位高于当前价格（最小间距2%）
                support_level = min(support_level, current_close * 0.98)
                resistance_level = max(resistance_level, current_close * 1.02)
            
            # 避免极端值
            support_level = max(support_level, current_close * 0.7)  # 支撑位不低于当前价格的70%
            resistance_level = min(resistance_level, current_close * 1.3)  # 压力位不高于当前价格的130%
            
            return support_level, resistance_level
            
        except Exception as e:
            logger.error(f"计算支撑/压力位失败: {e}")
            logger.error(traceback.format_exc())
            # 发生错误时使用默认计算方法
            return current_close * 0.95, current_close * 1.05

    def _calculate_volume_features(self, historical_volume, features):
        """计算成交量相关特征
        
        Args:
            historical_volume: 历史成交量数据，可以是Series或列表
            features: 特征字典，用于存储计算结果
            
        Returns:
            更新后的特征字典
        """
        # 初始化默认值
        default_features = {
            'volume_5d_avg': 0,
            'volume_10d_avg': 0,
            'volume_20d_avg': 0,
            'volume_60d_avg': 0,
            'volume_5d_ratio': 1.0,
            'volume_10d_ratio': 1.0,
            'volume_20d_ratio': 1.0,
            'volume_5d_change_ratio': 0,
            'volume_trend': 0
        }
        
        try:
            # 检查历史成交量数据
            if historical_volume is None or len(historical_volume) == 0:
                logger.warning("历史成交量数据为空，无法计算成交量特征")
                features.update(default_features)
                return features
            
            # 确保historical_volume是Series类型
            if not isinstance(historical_volume, pd.Series):
                try:
                    historical_volume = pd.Series(historical_volume)
                except Exception as e:
                    logger.error(f"转换成交量数据为Series时出错: {e}")
                    features.update(default_features)
                    return features
            
            # 检查Series是否为空
            if historical_volume.empty:
                logger.warning("成交量数据为空Series，无法计算特征")
                features.update(default_features)
                return features
                
            # 获取当前成交量（最近一天）
            try:
                current_volume = historical_volume.iloc[-1]
                # 检查NaN
                if pd.isna(current_volume):
                    current_volume = 0
            except IndexError:
                logger.warning("获取当前成交量时出错（索引错误）")
                current_volume = 0
            
            # 计算各周期平均成交量，确保处理数据不足的情况
            periods = [5, 10, 20, 60]
            avg_volumes = {}
            
            for period in periods:
                try:
                    if len(historical_volume) >= period:
                        period_avg = historical_volume.iloc[-period:].mean()
                    else:
                        period_avg = historical_volume.mean()
                    
                    # 检查NaN
                    if pd.isna(period_avg):
                        period_avg = 0
                        
                    avg_volumes[period] = float(period_avg)
                except Exception as e:
                    logger.error(f"计算{period}日平均成交量时出错: {e}")
                    avg_volumes[period] = 0
            
            # 存储平均成交量
            features['volume_5d_avg'] = avg_volumes[5]
            features['volume_10d_avg'] = avg_volumes[10]
            features['volume_20d_avg'] = avg_volumes[20]
            features['volume_60d_avg'] = avg_volumes[60]
            
            # 计算成交量比率（当前/平均），确保除数不为零
            for period in [5, 10, 20]:
                try:
                    if avg_volumes[period] > 0:
                        ratio = float(current_volume / avg_volumes[period])
                        # 限制极端值
                        ratio = min(max(ratio, 0.1), 10.0)
                    else:
                        ratio = 1.0
                    features[f'volume_{period}d_ratio'] = ratio
                except Exception as e:
                    logger.error(f"计算成交量{period}日比率时出错: {e}")
                    features[f'volume_{period}d_ratio'] = 1.0
            
            # 计算5日成交量变化率
            try:
                if len(historical_volume) >= 6:
                    # 当前5日均量
                    current_5d_avg = avg_volumes[5]
                    
                    # 前5日均量 (6天前到2天前)
                    prev_5d_data = historical_volume.iloc[-6:-1]
                    if len(prev_5d_data) >= 5:
                        prev_5d_avg = prev_5d_data.mean()
                        if pd.isna(prev_5d_avg):
                            prev_5d_avg = current_5d_avg if current_5d_avg > 0 else 1.0
                            
                        # 计算变化率
                        if prev_5d_avg > 0:
                            change_ratio = float((current_5d_avg - prev_5d_avg) / prev_5d_avg)
                            # 限制极端值
                            change_ratio = min(max(change_ratio, -0.9), 9.0)
                        else:
                            change_ratio = 0
                    else:
                        change_ratio = 0
                else:
                    change_ratio = 0
                    
                features['volume_5d_change_ratio'] = change_ratio
            except Exception as e:
                logger.error(f"计算5日成交量变化率时出错: {e}")
                features['volume_5d_change_ratio'] = 0
            
            # 计算成交量趋势（最近5天）
            try:
                if len(historical_volume) >= 5:
                    # 使用numpy操作更安全高效
                    recent_volumes = historical_volume.iloc[-5:].values
                    if len(recent_volumes) >= 2:
                        # 计算日间变化
                        diffs = np.diff(recent_volumes)
                        # 获取变化方向 (+1 上升, -1 下降, 0 不变)
                        signs = np.sign(diffs)
                        # 净方向和
                        volume_trend = int(np.sum(signs))
                    else:
                        volume_trend = 0
                else:
                    volume_trend = 0
                    
                features['volume_trend'] = volume_trend
            except Exception as e:
                logger.error(f"计算成交量趋势时出错: {e}")
                features['volume_trend'] = 0
            
            return features
        
        except Exception as e:
            logger.error(f"计算成交量特征时出错: {e}")
            logger.error(traceback.format_exc())
            # 发生异常时使用默认值
            features.update(default_features)
            return features

 
