#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场分析器的Numba加速版本
用于加速计算密集型的市场状态分析函数
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional, Union

# 导入Numba
try:
    import numba
    from numba import njit, jit, prange
    HAS_NUMBA = True
    logger = logging.getLogger("numba_market_analyzer")
    logger.info(f"Numba加速已启用，版本: {numba.__version__}")
except ImportError:
    HAS_NUMBA = False
    logger = logging.getLogger("numba_market_analyzer")
    logger.warning("Numba未安装，将使用原始计算方法")

# 配置参数
CACHE_JIT = True  # 是否缓存编译结果
ENABLE_PARALLEL = True  # 是否启用并行计算

# ----- 加速版技术指标计算函数 -----

@njit(cache=CACHE_JIT)
def calculate_returns_numba(prices: np.ndarray) -> np.ndarray:
    """计算收益率的Numba加速版本
    
    Args:
        prices: 价格数组
        
    Returns:
        returns: 收益率数组
    """
    n = len(prices)
    returns = np.zeros(n, dtype=np.float64)
    returns[0] = 0.0  # 首个元素无法计算收益率
    
    for i in range(1, n):
        if prices[i-1] != 0:  # 避免除以零
            returns[i] = (prices[i] / prices[i-1]) - 1
    
    return returns

@njit(cache=CACHE_JIT)
def calculate_ma_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """计算移动平均线的Numba加速版本
    
    Args:
        prices: 价格数组
        window: 窗口大小
        
    Returns:
        ma: 移动平均线数组
    """
    n = len(prices)
    ma = np.zeros(n, dtype=np.float64)
    
    # 对于前window-1个元素，使用可用数据计算
    for i in range(window-1):
        ma[i] = np.mean(prices[:i+1])
    
    # 对于其余元素，使用窗口内数据计算
    for i in range(window-1, n):
        ma[i] = np.mean(prices[i-window+1:i+1])
    
    return ma

@njit(cache=CACHE_JIT)
def calculate_ema_numba(prices: np.ndarray, span: int) -> np.ndarray:
    """计算指数移动平均线的Numba加速版本
    
    Args:
        prices: 价格数组
        span: EMA周期
        
    Returns:
        ema: 指数移动平均线数组
    """
    n = len(prices)
    ema = np.zeros(n, dtype=np.float64)
    
    # 计算平滑系数
    alpha = 2.0 / (span + 1.0)
    
    # 第一个值使用简单均值
    if n > 0:
        ema[0] = prices[0]
    
    # 计算其余值
    for i in range(1, n):
        ema[i] = prices[i] * alpha + ema[i-1] * (1.0 - alpha)
    
    return ema

@njit(cache=CACHE_JIT)
def calculate_macd_numba(prices: np.ndarray, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算MACD指标的Numba加速版本
    
    Args:
        prices: 价格数组
        fast_span: 快线周期
        slow_span: 慢线周期
        signal_span: 信号线周期
        
    Returns:
        (macd_line, signal_line, histogram): MACD线，信号线和柱状图
    """
    n = len(prices)
    macd_line = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    histogram = np.zeros(n, dtype=np.float64)
    
    # 计算EMA
    fast_ema = calculate_ema_numba(prices, fast_span)
    slow_ema = calculate_ema_numba(prices, slow_span)
    
    # 计算MACD线
    for i in range(n):
        macd_line[i] = fast_ema[i] - slow_ema[i]
    
    # 计算信号线 (MACD的EMA)
    alpha = 2.0 / (signal_span + 1.0)
    
    if n > 0:
        signal_line[0] = macd_line[0]
    
    for i in range(1, n):
        signal_line[i] = macd_line[i] * alpha + signal_line[i-1] * (1.0 - alpha)
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram

@njit(cache=CACHE_JIT)
def calculate_rsi_numba(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """计算RSI指标的Numba加速版本
    
    Args:
        prices: 价格数组
        window: 窗口大小
        
    Returns:
        rsi: RSI指标数组
    """
    n = len(prices)
    rsi = np.zeros(n, dtype=np.float64)
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    # 计算收益和损失
    for i in range(1, n):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change
    
    # 计算初始平均收益和损失
    avg_gain = np.sum(gains[1:window+1]) / window
    avg_loss = np.sum(losses[1:window+1]) / window
    
    # 填充前window个元素为NaN等效值
    for i in range(window+1):
        rsi[i] = 50.0  # 默认中性值
    
    # 计算其余RSI值
    for i in range(window+1, n):
        avg_gain = ((avg_gain * (window-1)) + gains[i]) / window
        avg_loss = ((avg_loss * (window-1)) + losses[i]) / window
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0  # 如果没有损失，RSI为100
    
    return rsi

@njit(cache=CACHE_JIT)
def calculate_bollinger_bands_numba(prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算布林带的Numba加速版本
    
    Args:
        prices: 价格数组
        window: 窗口大小
        num_std: 标准差倍数
        
    Returns:
        (upper_band, middle_band, lower_band): 上轨，中轨和下轨
    """
    n = len(prices)
    upper_band = np.zeros(n, dtype=np.float64)
    middle_band = np.zeros(n, dtype=np.float64)
    lower_band = np.zeros(n, dtype=np.float64)
    
    # 计算移动平均线
    for i in range(n):
        if i < window - 1:
            middle_band[i] = np.mean(prices[:i+1])
        else:
            window_slice = prices[i-window+1:i+1]
            middle_band[i] = np.mean(window_slice)
            std = np.std(window_slice)
            upper_band[i] = middle_band[i] + num_std * std
            lower_band[i] = middle_band[i] - num_std * std
    
    return upper_band, middle_band, lower_band

@njit(cache=CACHE_JIT)
def calculate_kdj_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算KDJ指标的Numba加速版本
    
    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        n: 周期
        m1: K值平滑系数
        m2: D值平滑系数
        
    Returns:
        (k, d, j): K, D, J值数组
    """
    data_len = len(close)
    k = np.zeros(data_len, dtype=np.float64)
    d = np.zeros(data_len, dtype=np.float64)
    j = np.zeros(data_len, dtype=np.float64)
    
    # 计算rsv
    rsv = np.zeros(data_len, dtype=np.float64)
    
    for i in range(n-1, data_len):
        period_high = np.max(high[i-n+1:i+1])
        period_low = np.min(low[i-n+1:i+1])
        
        if period_high == period_low:
            rsv[i] = 50
        else:
            rsv[i] = (close[i] - period_low) / (period_high - period_low) * 100
    
    # 初始化k、d值
    k[n-1] = rsv[n-1]
    d[n-1] = k[n-1]
    
    # 计算k、d、j值
    for i in range(n, data_len):
        k[i] = (m1 * k[i-1] + rsv[i]) / (m1 + 1)
        d[i] = (m2 * d[i-1] + k[i]) / (m2 + 1)
        j[i] = 3 * k[i] - 2 * d[i]
    
    return k, d, j

@njit(cache=CACHE_JIT, parallel=ENABLE_PARALLEL)
def calculate_trend_scores_batch_numba(
    prices: np.ndarray, 
    ma5: np.ndarray, 
    ma10: np.ndarray, 
    ma20: np.ndarray, 
    ma60: np.ndarray, 
    macd: np.ndarray, 
    macd_signal: np.ndarray
) -> np.ndarray:
    """批量计算趋势得分的Numba加速版本
    
    Args:
        prices: 价格数组
        ma5: 5日移动平均线
        ma10: 10日移动平均线
        ma20: 20日移动平均线
        ma60: 60日移动平均线
        macd: MACD线
        macd_signal: MACD信号线
        
    Returns:
        scores: 趋势得分数组
    """
    n = len(prices)
    scores = np.zeros(n, dtype=np.float64)
    
    # 并行计算每个点的趋势得分
    for i in prange(n):
        if i < 60:  # 不足60个点时，使用中性得分
            scores[i] = 0.5
            continue
        
        score = 0.0
        count = 0
        
        # 短期趋势 (MA5与MA10)
        if not np.isnan(ma5[i]) and not np.isnan(ma10[i]):
            if ma5[i] > ma10[i]:
                score += 0.6
            else:
                score += 0.4
            count += 1
        
        # 中期趋势 (MA10与MA20)
        if not np.isnan(ma10[i]) and not np.isnan(ma20[i]):
            if ma10[i] > ma20[i]:
                score += 0.6
            else:
                score += 0.4
            count += 1
        
        # 长期趋势 (MA20与MA60)
        if not np.isnan(ma20[i]) and not np.isnan(ma60[i]):
            if ma20[i] > ma60[i]:
                score += 0.7
            else:
                score += 0.3
            count += 1
        
        # MACD趋势
        if not np.isnan(macd[i]) and not np.isnan(macd_signal[i]):
            if macd[i] > macd_signal[i]:
                score += 0.65
            else:
                score += 0.35
            count += 1
        
        # 计算平均得分
        if count > 0:
            scores[i] = score / count
        else:
            scores[i] = 0.5  # 默认中性得分
    
    return scores

@njit(cache=CACHE_JIT)
def calculate_volume_profile_numba(volumes: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """计算成交量特性的Numba加速版本
    
    Args:
        volumes: 成交量数组
        window: 窗口大小
        
    Returns:
        (volume_ratio, volume_ma): 成交量比率和移动平均
    """
    n = len(volumes)
    volume_ratio = np.zeros(n, dtype=np.float64)
    volume_ma = np.zeros(n, dtype=np.float64)
    
    # 计算成交量移动平均
    for i in range(n):
        if i < window:
            volume_ma[i] = np.mean(volumes[:i+1])
        else:
            volume_ma[i] = np.mean(volumes[i-window+1:i+1])
    
    # 计算成交量比率
    for i in range(window, n):
        if volume_ma[i] > 0:
            volume_ratio[i] = volumes[i] / volume_ma[i]
        else:
            volume_ratio[i] = 1.0  # 默认中性值
    
    return volume_ratio, volume_ma

# ----- 工具函数 -----

def get_numba_accelerator():
    """获取Numba加速器实例"""
    if not HAS_NUMBA:
        logger.warning("Numba未安装，无法获取加速器")
        return None
    
    return {
        'has_numba': HAS_NUMBA,
        'calculate_returns': calculate_returns_numba,
        'calculate_ma': calculate_ma_numba,
        'calculate_ema': calculate_ema_numba,
        'calculate_macd': calculate_macd_numba,
        'calculate_rsi': calculate_rsi_numba,
        'calculate_bollinger_bands': calculate_bollinger_bands_numba,
        'calculate_kdj': calculate_kdj_numba,
        'calculate_trend_scores_batch': calculate_trend_scores_batch_numba,
        'calculate_volume_profile': calculate_volume_profile_numba
    }

def patch_market_analyzer(analyzer_class):
    """将Numba加速版本的函数应用到市场分析器类"""
    if not HAS_NUMBA:
        logger.warning("Numba未安装，无法应用加速")
        return False
    
    try:
        # 保存原始方法
        original_methods = {
            '_calculate_ma': getattr(analyzer_class, '_calculate_ma', None),
            '_calculate_macd': getattr(analyzer_class, '_calculate_macd', None),
            '_calculate_rsi': getattr(analyzer_class, '_calculate_rsi', None),
            '_calculate_bollinger_bands': getattr(analyzer_class, '_calculate_bollinger_bands', None),
            '_calculate_kdj': getattr(analyzer_class, '_calculate_kdj', None)
        }
        
        # 应用加速方法
        analyzer_class._calculate_ma_original = original_methods['_calculate_ma']
        analyzer_class._calculate_ma = lambda self, prices, window: calculate_ma_numba(np.array(prices, dtype=np.float64), window)
        
        analyzer_class._calculate_macd_original = original_methods['_calculate_macd']
        analyzer_class._calculate_macd = lambda self, prices, fast=12, slow=26, signal=9: calculate_macd_numba(np.array(prices, dtype=np.float64), fast, slow, signal)
        
        analyzer_class._calculate_rsi_original = original_methods['_calculate_rsi']
        analyzer_class._calculate_rsi = lambda self, prices, window=14: calculate_rsi_numba(np.array(prices, dtype=np.float64), window)
        
        analyzer_class._calculate_bollinger_bands_original = original_methods['_calculate_bollinger_bands']
        analyzer_class._calculate_bollinger_bands = lambda self, prices, window=20, num_std=2.0: calculate_bollinger_bands_numba(np.array(prices, dtype=np.float64), window, num_std)
        
        analyzer_class._calculate_kdj_original = original_methods['_calculate_kdj']
        analyzer_class._calculate_kdj = lambda self, high, low, close, n=9, m1=3, m2=3: calculate_kdj_numba(np.array(high, dtype=np.float64), np.array(low, dtype=np.float64), np.array(close, dtype=np.float64), n, m1, m2)
        
        # 添加加速器属性
        analyzer_class._numba_accelerator = get_numba_accelerator()
        analyzer_class._has_numba_accelerator = True
        
        logger.info(f"成功应用Numba加速到{analyzer_class.__name__}类")
        return True
    except Exception as e:
        logger.error(f"应用Numba加速失败: {e}")
        return False

# ----- 测试代码 -----

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据
    test_size = 1000
    prices = np.random.rand(test_size) * 100 + 50
    high = prices * 1.05
    low = prices * 0.95
    volumes = np.random.rand(test_size) * 10000
    
    # 测试指标计算
    print("\n测试Numba加速版指标计算:")
    
    # 测试MA
    import time
    start = time.time()
    ma20 = calculate_ma_numba(prices, 20)
    print(f"计算MA20耗时: {time.time() - start:.6f}秒")
    
    # 测试RSI
    start = time.time()
    rsi14 = calculate_rsi_numba(prices, 14)
    print(f"计算RSI14耗时: {time.time() - start:.6f}秒")
    
    # 测试MACD
    start = time.time()
    macd, signal, hist = calculate_macd_numba(prices)
    print(f"计算MACD耗时: {time.time() - start:.6f}秒")
    
    # 测试KDJ
    start = time.time()
    k, d, j = calculate_kdj_numba(high, low, prices)
    print(f"计算KDJ耗时: {time.time() - start:.6f}秒")
    
    # 测试布林带
    start = time.time()
    upper, middle, lower = calculate_bollinger_bands_numba(prices)
    print(f"计算布林带耗时: {time.time() - start:.6f}秒")
    
    # 测试趋势得分
    start = time.time()
    trend_scores = calculate_trend_scores_batch_numba(prices, ma20, ma20, ma20, ma20, macd, signal)
    print(f"计算趋势得分耗时: {time.time() - start:.6f}秒")
    
    print("\n加速完成!") 