#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numba加速模块
用于加速回测系统中的计算密集型操作
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import numba
from numba import jit, njit, prange
from functools import wraps

# 配置日志
logger = logging.getLogger("numba_accelerator")

# 检查Numba版本
try:
    numba_version = numba.__version__
    logger.info(f"Numba版本: {numba_version}")
    HAS_NUMBA = True
except:
    logger.warning("无法获取Numba版本信息，部分优化可能不可用")
    HAS_NUMBA = False

# 加速配置
ENABLE_PARALLEL = True  # 是否启用并行计算
CACHE_JIT = True        # 是否缓存JIT编译结果

# 用于性能对比
BENCHMARK_MODE = False  # 性能对比模式
performance_stats = {
    "original_time": {},
    "accelerated_time": {},
    "speedup_ratio": {},
    "call_count": {}
}

# 为Numba提供兼容的类型定义
float64_array = numba.float64[:]
float32_array = numba.float32[:]

@njit(cache=CACHE_JIT)
def calculate_ma(prices, window):
    """计算移动平均线 (JIT加速版)"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if i < window - 1:
            # 不足window个数据点时，使用可用数据的平均值
            result[i] = np.mean(prices[:i+1])
        else:
            # 使用window个数据点的平均值
            result[i] = np.mean(prices[i-window+1:i+1])
    
    return result

@njit(cache=CACHE_JIT)
def calculate_rsi(prices, window=14):
    """计算RSI指标 (JIT加速版)"""
    n = len(prices)
    result = np.zeros(n, dtype=np.float64)
    
    if n <= window:
        return result
    
    # 计算价格变化
    changes = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        changes[i] = prices[i] - prices[i-1]
    
    # 计算涨跌幅
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if changes[i] > 0:
            gains[i] = changes[i]
        else:
            losses[i] = -changes[i]
    
    # 计算平均涨跌幅
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    # 第一个窗口使用简单平均
    if window < n:
        avg_gain[window] = np.sum(gains[1:window+1]) / window
        avg_loss[window] = np.sum(losses[1:window+1]) / window
    
    # 其余使用递推公式
    for i in range(window+1, n):
        avg_gain[i] = (avg_gain[i-1] * (window-1) + gains[i]) / window
        avg_loss[i] = (avg_loss[i-1] * (window-1) + losses[i]) / window
    
    # 计算RSI
    for i in range(window, n):
        if avg_loss[i] == 0:
            result[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result

@njit(cache=CACHE_JIT)
def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标 (JIT加速版)"""
    n = len(prices)
    macd_line = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    histogram = np.zeros(n, dtype=np.float64)
    
    # 计算EMA
    fast_ema = np.zeros(n, dtype=np.float64)
    slow_ema = np.zeros(n, dtype=np.float64)
    
    # 初始化第一个EMA值
    if n > 0:
        fast_ema[0] = prices[0]
        slow_ema[0] = prices[0]
    
    # 计算系数
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)
    signal_alpha = 2.0 / (signal + 1.0)
    
    # 计算EMA
    for i in range(1, n):
        fast_ema[i] = prices[i] * fast_alpha + fast_ema[i-1] * (1.0 - fast_alpha)
        slow_ema[i] = prices[i] * slow_alpha + slow_ema[i-1] * (1.0 - slow_alpha)
        macd_line[i] = fast_ema[i] - slow_ema[i]
    
    # 计算信号线
    if n > 0:
        signal_line[0] = macd_line[0]
    
    for i in range(1, n):
        signal_line[i] = macd_line[i] * signal_alpha + signal_line[i-1] * (1.0 - signal_alpha)
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram

@njit(cache=CACHE_JIT)
def calculate_kdj(high, low, close, n=9, m1=3, m2=3):
    """计算KDJ指标 (JIT加速版)"""
    length = len(close)
    k = np.zeros(length, dtype=np.float64)
    d = np.zeros(length, dtype=np.float64)
    j = np.zeros(length, dtype=np.float64)
    
    if length < n:
        return k, d, j
    
    # 计算RSV
    rsv = np.zeros(length, dtype=np.float64)
    
    for i in range(n-1, length):
        period_low = np.min(low[i-n+1:i+1])
        period_high = np.max(high[i-n+1:i+1])
        
        if period_high == period_low:
            rsv[i] = 50.0
        else:
            rsv[i] = 100.0 * (close[i] - period_low) / (period_high - period_low)
    
    # 计算K值
    k[n-1] = rsv[n-1]
    
    # 计算剩余的K、D、J值
    for i in range(n, length):
        k[i] = (m1 * k[i-1] + rsv[i]) / (m1 + 1.0)
        
    # 计算D值
    d[n-1] = k[n-1]
    
    for i in range(n, length):
        d[i] = (m2 * d[i-1] + k[i]) / (m2 + 1.0)
    
    # 计算J值
    for i in range(n-1, length):
        j[i] = 3.0 * k[i] - 2.0 * d[i]
    
    return k, d, j

@njit(cache=CACHE_JIT)
def calculate_bollinger_bands(prices, window=20, num_std=2.0):
    """计算布林带 (JIT加速版)"""
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
def calculate_intraday_range_numba(open_price, high_price, low_price, prev_close=None):
    """计算日内区间涨跌幅 (JIT加速版)"""
    if prev_close is None or prev_close <= 0:
        prev_close = open_price
        
    if open_price <= 0 or high_price <= 0 or low_price <= 0:
        return 0.0, 0.0
            
    up_range = (high_price - open_price) / prev_close * 100.0
    down_range = (low_price - open_price) / prev_close * 100.0
        
    return up_range, down_range

@njit(parallel=ENABLE_PARALLEL, cache=CACHE_JIT)
def batch_calculate_intraday_range(open_prices, high_prices, low_prices, prev_closes=None):
    """批量计算日内区间涨跌幅 (JIT加速版)"""
    n = len(open_prices)
    up_ranges = np.zeros(n, dtype=np.float64)
    down_ranges = np.zeros(n, dtype=np.float64)
    
    use_prev = prev_closes is not None
    
    for i in prange(n):
        prev_close = prev_closes[i] if use_prev else open_prices[i]
        
        if prev_close <= 0 or open_prices[i] <= 0 or high_prices[i] <= 0 or low_prices[i] <= 0:
            up_ranges[i] = 0.0
            down_ranges[i] = 0.0
            continue
            
        up_ranges[i] = (high_prices[i] - open_prices[i]) / prev_close * 100.0
        down_ranges[i] = (low_prices[i] - open_prices[i]) / prev_close * 100.0
    
    return up_ranges, down_ranges

@njit(cache=CACHE_JIT)
def calculate_volatility(returns, window=20):
    """计算波动率 (JIT加速版)"""
    n = len(returns)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(window-1, n):
        result[i] = np.std(returns[i-window+1:i+1]) * np.sqrt(252)  # 年化
    
    return result

def performance_timer(func=None, name=None):
    """函数执行时间计时器装饰器"""
    def decorator(f):
        func_name = name or f.__name__
        
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not BENCHMARK_MODE:
                return f(*args, **kwargs)
            
            # 记录调用次数
            if func_name not in performance_stats["call_count"]:
                performance_stats["call_count"][func_name] = 0
            performance_stats["call_count"][func_name] += 1
            
            # 计时执行
            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # 记录执行时间
            if func_name not in performance_stats["accelerated_time"]:
                performance_stats["accelerated_time"][func_name] = 0
            performance_stats["accelerated_time"][func_name] += elapsed
            
            return result
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

class NumbaAccelerator:
    """Numba加速器
    
    提供对回测系统关键计算的加速支持
    """
    
    def __init__(self):
        """初始化加速器"""
        self.patched = False
        self.original_functions = {}
        self.benchmark_mode = False
        self.patch_complete = False
    
    def enable_benchmark_mode(self, enable=True):
        """启用/禁用性能对比模式"""
        global BENCHMARK_MODE
        BENCHMARK_MODE = enable
        self.benchmark_mode = enable
        logger.info(f"性能对比模式: {'启用' if enable else '禁用'}")
    
    def patch_advanced_quant_model(self, model_class):
        """将Numba加速版本的函数集成到量化模型中"""
        if not HAS_NUMBA:
            logger.warning("Numba库不可用，无法进行加速")
            return False
            
        if self.patched:
            logger.warning("已经应用过加速补丁，请勿重复应用")
            return True
        
        try:
            # 保存原始函数引用
            self.original_functions["calculate_intraday_range"] = model_class.calculate_intraday_range
            
            # 应用加速版本函数
            model_class.calculate_intraday_range = self._accelerated_calculate_intraday_range
            
            logger.info("成功应用Numba加速补丁到量化模型")
            self.patched = True
            return True
        except Exception as e:
            logger.error(f"应用Numba加速补丁失败: {e}")
            self.restore_original_functions(model_class)
            return False
    
    @performance_timer(name="accelerated_calculate_intraday_range")
    def _accelerated_calculate_intraday_range(self, open_price, high_price, low_price, prev_close=None):
        """加速版的日内区间涨跌幅计算函数"""
        # 单值计算使用标量函数
        if isinstance(open_price, (int, float)):
            return calculate_intraday_range_numba(open_price, high_price, low_price, prev_close)
        
        # 批量计算使用向量函数
        if isinstance(open_price, (list, np.ndarray, pd.Series)):
            open_arr = np.asarray(open_price, dtype=np.float64)
            high_arr = np.asarray(high_price, dtype=np.float64)
            low_arr = np.asarray(low_price, dtype=np.float64)
            
            prev_arr = None
            if prev_close is not None:
                prev_arr = np.asarray(prev_close, dtype=np.float64)
            
            return batch_calculate_intraday_range(open_arr, high_arr, low_arr, prev_arr)
        
        # 其他类型使用原始函数
        if self.benchmark_mode and "calculate_intraday_range" in self.original_functions:
            start_time = time.time()
            result = self.original_functions["calculate_intraday_range"](self, open_price, high_price, low_price, prev_close)
            elapsed = time.time() - start_time
            
            if "calculate_intraday_range" not in performance_stats["original_time"]:
                performance_stats["original_time"]["calculate_intraday_range"] = 0
            performance_stats["original_time"]["calculate_intraday_range"] += elapsed
            
            return result
        
        # 无法识别的类型或未保存原始函数
        return (0, 0)
    
    def create_accelerated_feature_extractors(self):
        """创建加速版的特征提取函数"""
        # 这些函数将在后续整合进特征提取流程
        extractors = {
            "calculate_ma": calculate_ma,
            "calculate_rsi": calculate_rsi,
            "calculate_macd": calculate_macd,
            "calculate_kdj": calculate_kdj,
            "calculate_bollinger_bands": calculate_bollinger_bands,
            "calculate_volatility": calculate_volatility
        }
        return extractors
    
    def restore_original_functions(self, model_class):
        """恢复原始函数"""
        if not self.patched:
            logger.info("未应用过加速补丁，无需恢复")
            return
            
        try:
            # 恢复原始函数
            for name, func in self.original_functions.items():
                setattr(model_class, name, func)
                
            logger.info("成功恢复原始函数")
            self.patched = False
        except Exception as e:
            logger.error(f"恢复原始函数失败: {e}")
    
    def generate_performance_report(self):
        """生成性能对比报告"""
        if not self.benchmark_mode or not BENCHMARK_MODE:
            logger.warning("未启用性能对比模式，无法生成报告")
            return {}
            
        report = {
            "function_stats": {},
            "overall_speedup": 0.0,
            "max_speedup": 0.0,
            "accelerated_functions": len(performance_stats["accelerated_time"])
        }
        
        total_original_time = 0
        total_accelerated_time = 0
        
        for func_name in performance_stats["accelerated_time"]:
            acc_time = performance_stats["accelerated_time"].get(func_name, 0)
            org_time = performance_stats["original_time"].get(func_name, 0)
            calls = performance_stats["call_count"].get(func_name, 0)
            
            if org_time > 0 and acc_time > 0:
                speedup = org_time / acc_time
                performance_stats["speedup_ratio"][func_name] = speedup
                
                report["function_stats"][func_name] = {
                    "original_time": org_time,
                    "accelerated_time": acc_time,
                    "speedup": speedup,
                    "calls": calls
                }
                
                total_original_time += org_time
                total_accelerated_time += acc_time
                
                if speedup > report["max_speedup"]:
                    report["max_speedup"] = speedup
        
        if total_original_time > 0 and total_accelerated_time > 0:
            report["overall_speedup"] = total_original_time / total_accelerated_time
        
        return report

# 创建全局实例
numba_accelerator = NumbaAccelerator()

def get_numba_accelerator():
    """获取Numba加速器实例"""
    return numba_accelerator 