#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征技术指标计算模块

提供各种技术指标的高性能计算实现
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional
import numba
from numba import jit

class TechnicalIndicators:
    """技术指标计算类
    
    提供各种技术指标的计算方法，包括：
    - 移动平均线 (SMA, EMA, WMA)
    - 布林带
    - RSI
    - MACD
    - ATR
    - OBV
    - 资金流向指标
    - 波动率指标
    等
    """
    
    def __init__(self):
        """初始化技术指标计算器"""
        pass
    
    def calculate_ma(self, series: Union[pd.Series, np.ndarray], window: int = 5) -> np.ndarray:
        """计算简单移动平均线
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            MA值数组
        """
        if isinstance(series, pd.Series):
            return series.rolling(window=window).mean().values
        else:
            return pd.Series(series).rolling(window=window).mean().values
    
    def calculate_ema(self, series: Union[pd.Series, np.ndarray], window: int = 12) -> np.ndarray:
        """计算指数移动平均线
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            EMA值数组
        """
        if isinstance(series, pd.Series):
            return series.ewm(span=window, adjust=False).mean().values
        else:
            return pd.Series(series).ewm(span=window, adjust=False).mean().values
    
    def calculate_wma(self, series: Union[pd.Series, np.ndarray], window: int = 10) -> np.ndarray:
        """计算加权移动平均线
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            WMA值数组
        """
        if not isinstance(series, np.ndarray):
            series = np.array(series)
        
        n = len(series)
        weights = np.arange(1, window + 1)
        result = np.full_like(series, np.nan, dtype=np.float64)
        
        for i in range(window - 1, n):
            # 使用向量化操作计算加权平均
            window_slice = series[i - window + 1:i + 1]
            result[i] = np.sum(window_slice * weights) / np.sum(weights)
        
        return result
    
    def calculate_bollinger_bands(
        self, series: Union[pd.Series, np.ndarray], window: int = 20, std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算布林带
        
        Args:
            series: 价格序列
            window: 窗口大小
            std_dev: 标准差倍数
            
        Returns:
            (上轨, 中轨, 下轨) 元组
        """
        if isinstance(series, pd.Series):
            ma = series.rolling(window=window).mean()
            std = series.rolling(window=window).std(ddof=0)
        else:
            s = pd.Series(series)
            ma = s.rolling(window=window).mean()
            std = s.rolling(window=window).std(ddof=0)
        
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        
        return upper.values, ma.values, lower.values
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """使用Numba加速的RSI计算
        
        Args:
            prices: 价格数组
            period: 周期
            
        Returns:
            RSI值数组
        """
        n = len(prices)
        rsi = np.full(n, np.nan)
        
        if n <= period:
            return rsi
        
        # 计算每日变化
        deltas = np.zeros(n)
        for i in range(1, n):
            deltas[i] = prices[i] - prices[i-1]
        
        # 初始化平均涨跌幅
        avg_gain = 0
        avg_loss = 0
        
        # 第一个周期的平均涨跌幅
        for i in range(1, period + 1):
            if deltas[i] > 0:
                avg_gain += deltas[i]
            elif deltas[i] < 0:
                avg_loss += abs(deltas[i])
        
        avg_gain /= period
        avg_loss /= period
        
        # 计算第一个RSI值
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        
        # 计算后续RSI值
        for i in range(period + 1, n):
            delta = deltas[i]
            
            if delta > 0:
                avg_gain = (avg_gain * (period - 1) + delta) / period
                avg_loss = (avg_loss * (period - 1)) / period
            elif delta < 0:
                avg_gain = (avg_gain * (period - 1)) / period
                avg_loss = (avg_loss * (period - 1) + abs(delta)) / period
            else:
                avg_gain = (avg_gain * (period - 1)) / period
                avg_loss = (avg_loss * (period - 1)) / period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_rsi(self, series: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
        """计算RSI (相对强弱指标)
        
        Args:
            series: 价格序列
            period: 周期
            
        Returns:
            RSI值数组
        """
        # 确保输入数据是numpy数组
        prices = np.array(series, dtype=np.float64)
        
        # 使用Numba加速计算
        return self._calculate_rsi_numba(prices, period)
    
    def calculate_macd(
        self, series: Union[pd.Series, np.ndarray], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算MACD (移动平均收敛/发散)
        
        Args:
            series: 价格序列
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            (MACD, 信号线, 柱状图) 元组
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        
        # 计算EMA
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return macd_line.values, signal_line.values, histogram.values
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """使用Numba加速的ATR计算
        
        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            period: 周期
            
        Returns:
            ATR值数组
        """
        n = len(high)
        tr = np.zeros(n)
        atr = np.full(n, np.nan)
        
        if n <= period:
            return atr
        
        # 计算真实波幅 (TR)
        for i in range(1, n):
            tr1 = high[i] - low[i]  # 当日振幅
            tr2 = abs(high[i] - close[i-1])  # 当日最高与昨收差值
            tr3 = abs(low[i] - close[i-1])  # 当日最低与昨收差值
            tr[i] = max(tr1, tr2, tr3)
        
        # 计算第一个ATR值
        atr[period] = 0
        for i in range(1, period + 1):
            atr[period] += tr[i]
        atr[period] /= period
        
        # 计算后续ATR值
        for i in range(period + 1, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def calculate_atr(
        self, high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
        close: Union[pd.Series, np.ndarray], period: int = 14
    ) -> np.ndarray:
        """计算ATR (平均真实波幅)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期
            
        Returns:
            ATR值数组
        """
        # 确保输入数据是numpy数组
        high_arr = np.array(high, dtype=np.float64)
        low_arr = np.array(low, dtype=np.float64)
        close_arr = np.array(close, dtype=np.float64)
        
        # 使用Numba加速计算
        return self._calculate_atr_numba(high_arr, low_arr, close_arr, period)
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_obv_numba(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """使用Numba加速的OBV计算
        
        Args:
            close: 收盘价数组
            volume: 成交量数组
            
        Returns:
            OBV值数组
        """
        n = len(close)
        obv = np.zeros(n)
        
        if n <= 1:
            return obv
        
        # 第一个OBV值设为成交量
        obv[0] = volume[0]
        
        # 计算后续OBV值
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def calculate_obv(
        self, close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """计算OBV (能量潮)
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
            
        Returns:
            OBV值数组
        """
        # 确保输入数据是numpy数组
        close_arr = np.array(close, dtype=np.float64)
        volume_arr = np.array(volume, dtype=np.float64)
        
        # 使用Numba加速计算
        return self._calculate_obv_numba(close_arr, volume_arr)
    
    def calculate_momentum(self, series: Union[pd.Series, np.ndarray], period: int = 10) -> np.ndarray:
        """计算动量指标
        
        Args:
            series: 价格序列
            period: 周期
            
        Returns:
            动量值数组
        """
        if isinstance(series, pd.Series):
            result = series.diff(period)
            return result.values
        else:
            return pd.Series(series).diff(period).values
    
    def calculate_rate_of_change(self, series: Union[pd.Series, np.ndarray], period: int = 10) -> np.ndarray:
        """计算变化率 (ROC)
        
        Args:
            series: 价格序列
            period: 周期
            
        Returns:
            ROC值数组
        """
        if isinstance(series, pd.Series):
            result = series.pct_change(period) * 100
            return result.values
        else:
            return pd.Series(series).pct_change(period).values * 100
    
    def calculate_volatility(self, series: Union[pd.Series, np.ndarray], period: int = 20) -> np.ndarray:
        """计算波动率
        
        Args:
            series: 价格序列
            period: 周期
            
        Returns:
            波动率值数组
        """
        if isinstance(series, pd.Series):
            # 计算对数收益率
            log_returns = np.log(series / series.shift(1))
            # 计算移动标准差
            volatility = log_returns.rolling(window=period).std() * np.sqrt(252)  # 年化
            return volatility.values
        else:
            s = pd.Series(series)
            log_returns = np.log(s / s.shift(1))
            volatility = log_returns.rolling(window=period).std() * np.sqrt(252)  # 年化
            return volatility.values
    
    def calculate_trend_strength(self, series: Union[pd.Series, np.ndarray], window: int = 10) -> np.ndarray:
        """计算趋势强度
        
        Args:
            series: 价格序列
            window: 窗口大小
            
        Returns:
            趋势强度值数组
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # 计算线性回归斜率
        def slope(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                # 使用最小二乘法计算斜率
                m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                return m
            except:
                return np.nan
        
        # 计算窗口内的斜率
        result = series.rolling(window=window).apply(slope, raw=True)
        
        # 标准化结果
        max_val = result.max()
        min_val = result.min()
        if max_val == min_val:
            return np.zeros_like(result)
        
        normalized = (result - min_val) / (max_val - min_val) * 2 - 1
        return normalized.values
    
    def calculate_distance_to_high(
        self, close: Union[pd.Series, np.ndarray], high: Union[pd.Series, np.ndarray], period: int = 20
    ) -> np.ndarray:
        """计算到周期高点的距离
        
        Args:
            close: 收盘价序列
            high: 最高价序列
            period: 周期
            
        Returns:
            到高点距离的百分比数组
        """
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        
        # 计算滚动最高价
        rolling_high = high.rolling(window=period).max()
        
        # 计算当前价格与最高价的差距百分比
        result = (rolling_high - close) / rolling_high * 100
        
        return result.values
    
    def calculate_distance_to_low(
        self, close: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], period: int = 20
    ) -> np.ndarray:
        """计算到周期低点的距离
        
        Args:
            close: 收盘价序列
            low: 最低价序列
            period: 周期
            
        Returns:
            到低点距离的百分比数组
        """
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        
        # 计算滚动最低价
        rolling_low = low.rolling(window=period).min()
        
        # 计算当前价格与最低价的差距百分比
        result = (close - rolling_low) / close * 100
        
        return result.values
    
    def calculate_volume_price_divergence(
        self, close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray], period: int = 14
    ) -> np.ndarray:
        """计算量价背离指标
        
        Args:
            close: 收盘价序列
            volume: 成交量序列
            period: 周期
            
        Returns:
            量价背离指标数组
        """
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        if not isinstance(volume, pd.Series):
            volume = pd.Series(volume)
        
        # 计算价格变化率
        price_change = close.pct_change(period)
        
        # 计算成交量变化率
        volume_change = volume.pct_change(period)
        
        # 计算量价背离 (价格上涨但成交量下降，或价格下跌但成交量上升)
        divergence = np.zeros_like(close.values)
        
        for i in range(period, len(close)):
            # 价格上涨但成交量下降
            if price_change.iloc[i] > 0 and volume_change.iloc[i] < 0:
                divergence[i] = -1
            # 价格下跌但成交量上升
            elif price_change.iloc[i] < 0 and volume_change.iloc[i] > 0:
                divergence[i] = 1
        
        return divergence
    
    def calculate_money_flow_index(
        self, high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
        close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray], period: int = 14
    ) -> np.ndarray:
        """计算资金流向指标 (MFI)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
            period: 周期
            
        Returns:
            MFI值数组
        """
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        if not isinstance(volume, pd.Series):
            volume = pd.Series(volume)
        
        # 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 计算原始资金流 (典型价格 * 成交量)
        raw_money_flow = typical_price * volume
        
        # 决定资金流向
        money_flow_positive = pd.Series(0, index=close.index)
        money_flow_negative = pd.Series(0, index=close.index)
        
        # 价格上涨时为正向资金流，下跌时为负向资金流
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                money_flow_positive.iloc[i] = raw_money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                money_flow_negative.iloc[i] = raw_money_flow.iloc[i]
        
        # 计算周期内的正向和负向资金流总和
        positive_sum = money_flow_positive.rolling(window=period).sum()
        negative_sum = money_flow_negative.rolling(window=period).sum()
        
        # 计算资金比率 (Money Ratio)
        money_ratio = pd.Series(np.nan, index=close.index)
        valid_indices = (negative_sum > 0)
        money_ratio[valid_indices] = positive_sum[valid_indices] / negative_sum[valid_indices]
        
        # 计算MFI
        mfi = 100 - (100 / (1 + money_ratio))
        
        # 处理特殊情况
        mfi[negative_sum == 0] = 100
        
        return mfi.values
    
    def calculate_chaikin_money_flow(
        self, high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
        close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray], period: int = 20
    ) -> np.ndarray:
        """计算佳庆资金流量指标 (CMF)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
            period: 周期
            
        Returns:
            CMF值数组
        """
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        if not isinstance(volume, pd.Series):
            volume = pd.Series(volume)
        
        # 计算货币流量乘数
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfm = mfm.fillna(0)
        
        # 计算货币流量成交量
        mfv = mfm * volume
        
        # 计算CMF
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf.values 