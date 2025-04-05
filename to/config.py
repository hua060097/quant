#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件: 定义项目中使用的所有关键路径和常量
"""

import os
from pathlib import Path

# --- 基础路径 ---
# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
# 数据根目录
DATA_ROOT = PROJECT_ROOT / "local_data"
# 模型目录
MODEL_DIR = PROJECT_ROOT / "models"
# 日志目录
LOG_DIR = PROJECT_ROOT / "log"
# 结果目录
RESULTS_DIR = PROJECT_ROOT / "results"

# --- 特征相关路径 ---
# 特征根目录
FEATURE_DIR = DATA_ROOT / "features"
# 日线特征目录
FEATURE_DIR_DAILY = FEATURE_DIR / "daily"
# 分钟线特征目录
FEATURE_DIR_MINUTE = FEATURE_DIR / "minute"

# --- 原始数据相关路径 ---
# 日线数据目录
DAILY_DATA_DIR = DATA_ROOT / "daily"
# 分钟线数据目录
MINUTE_DATA_DIR = DATA_ROOT / "minute"
# 交易日历目录
CALENDAR_DIR = DATA_ROOT / "calendar"
# 元数据目录
METADATA_DIR = DATA_ROOT / "metadata"
# 市场状态数据目录
MARKET_STATUS_DIR = DATA_ROOT / "market_status"

# --- 板块数据路径 ---
SECTOR_DATA_DIR = PROJECT_ROOT / "sector_analysis"
# 同花顺概念板块数据文件
TNG_SECTORS_FILE = SECTOR_DATA_DIR / "tng_sectors_stats.json"

# --- 回测相关设置 ---
# 回测结果目录
BACKTEST_RESULTS_DIR = RESULTS_DIR / "backtest"
# 训练数据积累期
TRAINING_ACCUMULATION_START = "20230601"
TRAINING_ACCUMULATION_END = "20230930"
# 参数优化期
PARAMETER_OPTIMIZATION_START = "20231001"
PARAMETER_OPTIMIZATION_END = "20231231"
# 主要回测期
MAIN_BACKTEST_START = "20240101"
MAIN_BACKTEST_END = "20240930"
# 未来验证期
FUTURE_VALIDATION_START = "20241001"
FUTURE_VALIDATION_END = "20250301"

# --- 模型训练参数 ---
# 模型更新频率
LONG_CYCLE_UPDATE_INTERVAL = 10  # 长周期模型每10天更新一次
SHORT_CYCLE_UPDATE_INTERVAL = 1   # 短周期模型每天更新
# 回溯天数
LOOKBACK_SHORT = 30  # 短期回溯天数
LOOKBACK_LONG = 90   # 长期回溯天数
# 预测未来天数
FORWARD_DAYS = 5     # 预测未来5天的收益

# --- 系统设置 ---
# 默认并行处理数量 (若不指定则使用CPU核心数的一半)
DEFAULT_WORKERS = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
# 最大并行处理数量
MAX_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 8
# 缓存设置
ENABLE_CACHE = True
CACHE_DIR = PROJECT_ROOT / "data_cache"
MAX_CACHE_SIZE_GB = 8  # 最大缓存大小 (GB)

# --- 自动创建目录 ---
for directory in [
    MODEL_DIR, LOG_DIR, RESULTS_DIR, FEATURE_DIR_DAILY, FEATURE_DIR_MINUTE,
    DAILY_DATA_DIR, MINUTE_DATA_DIR, CALENDAR_DIR, METADATA_DIR, MARKET_STATUS_DIR,
    BACKTEST_RESULTS_DIR, CACHE_DIR, SECTOR_DATA_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# 打印所有目录的创建状态
print("配置路径初始化完成:")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"数据根目录: {DATA_ROOT}")
print(f"模型目录: {MODEL_DIR}")
print(f"特征目录: {FEATURE_DIR_DAILY}")
print(f"日志目录: {LOG_DIR}")
print(f"结果目录: {RESULTS_DIR}")
print(f"回测结果目录: {BACKTEST_RESULTS_DIR}")

def get_config():
    """获取系统配置"""
    return {
        'data_dirs': {
            'features': os.path.join('local_data', 'features'),
            'sector_features': os.path.join('local_data', 'sector_features'),
            'market_features': os.path.join('local_data', 'market_features'),
            'models': os.path.join('local_data', 'models'),
            'signals': os.path.join('signals')
        },
        'model_params': {
            'lookback_short': 30,
            'lookback_long': 90,
            'forward_days': 5
        },
        'sector_rotation': {
            'top_sectors': 3,
            'stocks_per_sector': 3
        }
    } 