#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Manager, Lock, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, date
import xtquant.xtdata as xtdata
import matplotlib.pyplot as plt
import time
import warnings
import talib
import joblib
from tqdm import tqdm
from functools import partial
from itertools import product
import ray
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import calendar
from stock_daily_rankings import get_ten_day_rise_ranking, get_daily_rise_ranking, get_daily_fall_ranking, get_daily_amount_ranking
from stock_sectors_mapping import load_data as load_sectors_data, get_stock_sectors, get_tng_sectors, create_stock_sector_mapping, get_sector_stocks
from sector_analysis import read_rankings as read_sector_rankings
import json
import threading
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import re
from typing import Dict, List, Tuple, Union, Optional, Any  # 添加这行导入语句
import hashlib
import sys
import concurrent.futures
import psutil
from pathlib import Path
import traceback

# 配置日志记录器
logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("警告: imblearn包未安装，将使用基本方法处理样本不平衡")

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False
    print("警告: hyperopt包未安装，将使用默认参数")

# 禁止警告消息
warnings.filterwarnings('ignore')

# 基础配置
START_DATE = '20230601'  # 训练起始日期
END_DATE = '20250301'    # 训练截止日期
TRAINING_DAYS = 30       # 训练使用前N个交易日
MAX_THREADS = min(32, cpu_count())  # 最大线程数
MAX_POSITION = 0.8       # 最大仓位比例
MIN_POSITION = 0.2       # 最小仓位比例
STOP_LOSS = 0.05         # 止损比例
DYNAMIC_STOP_PROFIT = True  # 是否使用动态止盈
STOP_PROFIT_MIN = 0.05   # 最小止盈比例
STOP_PROFIT_MAX = 0.15   # 最大止盈比例
MODEL_DIR = 'models'     # 模型保存目录
RESULT_DIR = 'results'   # 结果保存目录
MAX_STOCKS_PER_SECTOR = 3  # 每个板块最多选择的股票数量

# 确保目录存在
for dir_path in [MODEL_DIR, RESULT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 尝试导入缓存系统
try:
    from data_cache_manager import get_data_cache_manager
    HAS_CACHE = True
    logger = logging.getLogger(__name__)
    logger.info("数据缓存系统已加载，将优化数据访问性能")
except ImportError:
    HAS_CACHE = False
    logger = logging.getLogger(__name__)
    logger.warning("未找到数据缓存系统，将使用标准数据加载方式")

# 尝试导入Numba加速器
try:
    from numba_accelerator import get_numba_accelerator
    HAS_NUMBA_ACCELERATOR = True
    logger.info("成功导入Numba加速器")
except ImportError:
    HAS_NUMBA_ACCELERATOR = False
    logger.warning("无法导入Numba加速器，将使用原始计算方法")

class LiquidTimeCell(nn.Module):
    """液体时间单元(Liquid Time Cell)"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 定义网络参数
        self.W = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, h):
        """前向传播"""
        # 计算输入和隐藏状态的线性组合
        dh = torch.tanh(torch.mm(x, self.W.t()) + torch.mm(h, self.U.t()) + self.bias)
        # 应用时间常数
        h_new = h + (dh - h) / self.tau.unsqueeze(0)
        return h_new

class LiquidNeuralNetwork(nn.Module):
    """液体神经网络模型"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 创建多层LTC
        self.ltc_layers = nn.ModuleList([
            LiquidTimeCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None):
        """前向传播"""
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if h is None:
            h = [torch.zeros(batch_size, self.hidden_size).to(x.device) 
                 for _ in range(self.num_layers)]
        
        # 通过LTC层
        for i, ltc in enumerate(self.ltc_layers):
            h[i] = ltc(x if i == 0 else h[i-1], h[i])
        
        # 输出层
        output = self.output_layer(h[-1])
        return output, h

class EnsembleModel:
    """集成模型，结合XGBoost、LightGBM、CatBoost和LNN"""
    def __init__(self, input_size, hidden_size=64, output_size=5):
        self.models = {
            'xgboost': None,
            'lightgbm': None,
            'catboost': None,
            'lnn': None
        }
        self.weights = {
            'xgboost': 0.3,
            'lightgbm': 0.3,
            'catboost': 0.2,
            'lnn': 0.2
        }
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
    def train(self, X, y, validation_data=None):
        """训练所有模型"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # 训练XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgboost'].fit(X, y)
        
        # 训练LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        self.models['lightgbm'].fit(X, y)
        
        # 训练CatBoost
        from catboost import CatBoostClassifier
        self.models['catboost'] = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False
        )
        self.models['catboost'].fit(X, y)
        
        # 训练LNN
        self.models['lnn'] = LiquidNeuralNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(self.models['lnn'].parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 训练LNN
        self.models['lnn'].train()
        num_epochs = 50
        batch_size = 32
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output, _ = self.models['lnn'](batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        
    def predict(self, X):
        """集成预测"""
        X_tensor = torch.FloatTensor(X)
        
        # 获取各个模型的预测
        predictions = {}
        
        # XGBoost预测
        predictions['xgboost'] = self.models['xgboost'].predict_proba(X)
        
        # LightGBM预测
        predictions['lightgbm'] = self.models['lightgbm'].predict_proba(X)
        
        # CatBoost预测
        predictions['catboost'] = self.models['catboost'].predict_proba(X)
        
        # LNN预测
        self.models['lnn'].eval()
        with torch.no_grad():
            lnn_output, _ = self.models['lnn'](X_tensor)
            predictions['lnn'] = F.softmax(lnn_output, dim=1).numpy()
        
        # 加权融合预测结果
        final_prediction = np.zeros_like(predictions['xgboost'])
        for model_name, weight in self.weights.items():
            final_prediction += predictions[model_name] * weight
        
        return final_prediction

class AdvancedQuantModel:
    """高级量化回测模型"""
    logger = logging.getLogger(__name__)
    
    # 将EnsembleModel类移到外部，作为AdvancedQuantModel的嵌套类
    class EnsembleModelWrapper:
        def __init__(self, short_model, long_model, weights=(0.7, 0.3)):
            # 存储模型和权重信息
            self.short_model = short_model
            self.long_model = long_model
            self.weights = weights
            # 添加标志以避免序列化问题
            self.is_serializable = True
            
        def predict(self, X):
            """预测分类结果"""
            try:
                # 获取两个模型的预测概率
                short_proba = self.short_model.get('model', {}).get('predict_proba', lambda x: np.zeros((len(x), 2)))(X) if isinstance(self.short_model, dict) else self.short_model.predict_proba(X)
                long_proba = self.long_model.get('model', {}).get('predict_proba', lambda x: np.zeros((len(x), 2)))(X) if isinstance(self.long_model, dict) else self.long_model.predict_proba(X)
                
                # 加权组合概率
                combined_proba = (self.weights[0] * short_proba + 
                                self.weights[1] * long_proba)
                
                # 返回预测结果
                return np.argmax(combined_proba, axis=1)
            except Exception as e:
                # 出错时返回默认预测
                return np.zeros(len(X), dtype=int)
            
        def predict_proba(self, X):
            """预测概率分布"""
            try:
                # 获取两个模型的预测概率
                short_proba = self.short_model.get('model', {}).get('predict_proba', lambda x: np.zeros((len(x), 2)))(X) if isinstance(self.short_model, dict) else self.short_model.predict_proba(X)
                long_proba = self.long_model.get('model', {}).get('predict_proba', lambda x: np.zeros((len(x), 2)))(X) if isinstance(self.long_model, dict) else self.long_model.predict_proba(X)
                
                # 加权组合概率
                return (self.weights[0] * short_proba + 
                        self.weights[1] * long_proba)
            except Exception as e:
                # 出错时返回默认概率
                return np.zeros((len(X), 2))
        
        def __getattr__(self, attr):
            """获取属性（优先短期模型）"""
            # 修复递归问题：检查self.short_model是否存在，避免循环引用
            if attr == 'short_model' or not hasattr(self, 'short_model') or self.short_model is None:
                raise AttributeError(f"{self.__class__.__name__} 对象没有 '{attr}' 属性")
            
            # 适配字典格式的模型
            if isinstance(self.short_model, dict):
                if attr in self.short_model:
                    return self.short_model[attr]
                elif 'model' in self.short_model and hasattr(self.short_model['model'], attr):
                    return getattr(self.short_model['model'], attr)
                else:
                    raise AttributeError(f"{self.__class__.__name__} 的short_model中没有 '{attr}' 属性")
            else:
                # 默认使用短周期模型的属性
                if hasattr(self.short_model, attr):
                    return getattr(self.short_model, attr)
                else:
                    raise AttributeError(f"{self.__class__.__name__} 和 short_model 都没有 '{attr}' 属性")
                    
        def to_dict(self):
            """转换为纯字典表示，确保可序列化"""
            return {
                'class': 'EnsembleModelWrapper',
                'weights': self.weights,
                'short_model': self.short_model if isinstance(self.short_model, dict) else {'model_type': str(type(self.short_model).__name__)},
                'long_model': self.long_model if isinstance(self.long_model, dict) else {'model_type': str(type(self.long_model).__name__)},
                'is_serializable': True
            }
    
    def __init__(self, 
                 config_file: str = None, 
                 cache_dir: str = './data_cache',
                 use_cache: bool = True,
                 use_numba: bool = True) -> None:
        """初始化高级量化模型
        
        Args:
            config_file: 配置文件路径
            cache_dir: 缓存数据目录
            use_cache: 是否使用缓存
            use_numba: 是否使用Numba加速计算
        """
        self.config_file = config_file
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # 使用默认配置，不再调用_load_config()
        self.config = {
            "model": {
                "max_lookback_days": 60,
                "forward_days": 5,
                "feature_cols": ["ma_cross", "volume_ratio", "rsi", "intraday_range"]
            },
            "backtest": {
                "initial_capital": 1000000,
                "commission_rate": 0.0003,
                "slippage": 0.0002,
                "position_limit": 0.10
            },
            "data": {
                "market": "CN",
                "cache_dir": cache_dir,
                "use_cache": use_cache
            }
        }
        
        # 初始化日志
        self._setup_logging()
        
        # 初始化数据缓存
        self.cache_manager = None
        if use_cache:
            try:
                from data_cache_manager import DataCacheManager
                self.cache_manager = DataCacheManager(cache_dir=cache_dir)
                logger.info(f"成功初始化数据缓存，缓存目录: {cache_dir}")
            except Exception as e:
                logger.error(f"初始化数据缓存失败: {e}")
        
        # 初始化Numba加速器
        self.numba_accelerator = None
        if use_numba and HAS_NUMBA_ACCELERATOR:
            try:
                self.numba_accelerator = get_numba_accelerator()
                # 应用Numba加速到当前类
                success = self.numba_accelerator.patch_advanced_quant_model(self.__class__)
                if success:
                    logger.info("成功应用Numba加速到量化模型")
                else:
                    logger.warning("应用Numba加速失败，将使用原始计算方法")
            except Exception as e:
                logger.error(f"初始化Numba加速器失败: {e}")
        
        # 初始化其他属性
        self.trading_dates = []
        self.models = {}
        self.features = {}
        self.stock_pools = {}
        self.market_data = {}
        
        logger.info("高级量化模型初始化完成")
        
        # 初始化属性
        self.sectors = {}  # 板块数据
        self.sector_models = {}  # 板块模型字典
        self.stocks = {}  # 股票信息
        self.market_data = {}  # 市场数据
        self.trading_dates = []  # 交易日历
        self.sector_performance = {}  # 板块表现排名
        self.api = None  # API接口
        self.data_source = None  # 数据源
        self.ray_initialized = False  # Ray分布式计算标记

    def load_trading_dates(self):
        """加载交易日期
        """
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
            # 优先从缓存加载交易日期
            if HAS_CACHE and self.cache_manager:
                logger.info("尝试从缓存加载交易日期")
                try:
                    trading_dates = self.cache_manager.get_trading_dates(market='SH', start_time=start_date, end_time=end_date)
                    logger.debug(f"缓存返回交易日期类型: {type(trading_dates)}")
                    
                    if trading_dates is not None and len(trading_dates) > 0:
                        logger.debug(f"交易日期第一项: {trading_dates[0]}, 类型: {type(trading_dates[0])}")
                        self.trading_dates = trading_dates
                        logger.info(f"从缓存成功加载了 {len(self.trading_dates)} 个交易日")
                        if self.trading_dates:
                            logger.info(f"日期范围: {self.trading_dates[0]} 至 {self.trading_dates[-1]}")
                        return
                    else:
                        logger.info("缓存中没有交易日期数据，将从API获取")
                except Exception as e:
                    logger.error(f"加载交易日期时出错: {e}")
                    logger.exception(e)  # 打印完整堆栈
                    logger.info("尝试直接从API获取交易日期")
                    trading_dates = []
            
            if trading_dates and len(trading_dates) > 0:
                # 检查日期类型，进行必要的格式转换
                if isinstance(trading_dates[0], (int, float)):
                    # 转换时间戳（毫秒）为字符串格式 YYYYMMDD
                    self.trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
                elif isinstance(trading_dates[0], str):
                    # 如果已经是字符串，确保格式统一
                    # 检查格式是否是'YYYYMMDD'
                    if len(trading_dates[0]) == 8 and trading_dates[0].isdigit():
                        self.trading_dates = trading_dates
                    else:
                        # 尝试转换其他字符串格式
                        try:
                            self.trading_dates = [datetime.strptime(d, '%Y-%m-%d').strftime('%Y%m%d') for d in trading_dates]
                        except:
                            logger.error(f"无法识别的日期格式: {trading_dates[0]}")
                            raise ValueError(f"无法识别的日期格式: {trading_dates[0]}")
                elif isinstance(trading_dates[0], datetime):
                    # 如果是datetime对象，转换为字符串
                    self.trading_dates = [d.strftime('%Y%m%d') for d in trading_dates]
                else:
                    logger.error(f"未知的日期类型: {type(trading_dates[0])}")
                    raise TypeError(f"未知的日期类型: {type(trading_dates[0])}")
                
                # 确保日期是按时间顺序排列的
                self.trading_dates = sorted(self.trading_dates) if self.trading_dates else []
                
                # 如果有缓存管理器，将交易日期保存到缓存
                if HAS_CACHE and self.cache_manager:
                    try:
                        # 确保传递正确格式的参数，避免递归调用
                        logger.debug("尝试保存交易日期到缓存")
                        success = self.cache_manager.save_trading_dates(market='SH', trading_dates=self.trading_dates)
                        if success:
                            logger.info("交易日期已保存到缓存")
                        else:
                            logger.warning("保存交易日期到缓存失败")
                    except Exception as e:
                        logger.warning(f"保存交易日期到缓存时出错: {e}")
                        logger.exception(e)
                
                logger.info(f"成功加载了 {len(self.trading_dates)} 个交易日")
                if self.trading_dates:
                    logger.info(f"日期范围: {self.trading_dates[0]} 至 {self.trading_dates[-1]}")
            else:
                logger.warning("未能获取交易日期数据")
                self.trading_dates = []
        except Exception as e:
            logger.error(f"加载交易日期时出错: {e}")
            logger.exception(e)
            self.trading_dates = []
    
    

    def _get_start_date(self, date, lookback_days):
        """获取向前的日期
        
        参数:
            date: 当前日期
            lookback_days: 向前天数
            
        返回:
            开始日期，如果无法获取则返回None
        """
        if not self.trading_dates:
            self.load_trading_dates()
        
        try:
            current_idx = self.trading_dates.index(date)
            if current_idx - lookback_days >= 0:
                return self.trading_dates[current_idx - lookback_days]
            else:
                return self.trading_dates[0]  # 返回最早的交易日
        except ValueError:
            # 如果日期不在交易日中，找到最接近的交易日
            closest_date = self._get_closest_trading_date(date)
            try:
                closest_idx = self.trading_dates.index(closest_date)
                if closest_idx - lookback_days >= 0:
                    return self.trading_dates[closest_idx - lookback_days]
                else:
                    return self.trading_dates[0]
            except ValueError:
                return None
        except Exception as e:
            logger.error(f"获取开始日期时出错: {e}")
            return None
    
    def _load_models_for_date(self, date: str) -> bool:
        """
        为指定日期加载合适的预训练模型。

        会查找每个板块在指定日期或之前训练的最新模型。
        假设模型存储在 to/models/{sector_name}/{train_date}.pkl

        Args:
            date: 目标日期 (YYYYMMDD)

        Returns:
            bool: 如果成功为至少一个板块加载了模型，则返回True，否则False。
        """
        self.logger.info(f"开始为日期 {date} 加载预训练模型...")
        loaded_count = 0
        error_count = 0
        
        # 修改模型基础目录路径，确保与train_models.py中的保存路径一致
        model_base_dir = Path(os.path.join(os.path.dirname(__file__), "models"))
        self.logger.info(f"使用模型目录: {model_base_dir}")

        if not hasattr(self, 'sectors') or not self.sectors:
            self.logger.error("板块数据未加载 (self.sectors)，无法加载模型。")
            self._load_sector_data()  # 尝试加载板块数据
            if not hasattr(self, 'sectors') or not self.sectors:
                self.logger.error("尝试加载板块数据失败，仍无法加载模型。")
                return False

        # 清空现有模型字典
        self.sector_models = {}

        # 遍历所有已知板块，优先处理TGN开头的同花顺概念板块
        tng_sectors = {k: v for k, v in self.sectors.items() if k.startswith('TGN')}
        if not tng_sectors:
            self.logger.warning("未找到同花顺概念板块(TGN开头)，将使用所有板块")
            sectors_to_load = self.sectors
        else:
            self.logger.info(f"找到 {len(tng_sectors)} 个同花顺概念板块，将优先加载这些板块模型")
            sectors_to_load = tng_sectors

        for sector_name in sectors_to_load.keys():
            sector_model_dir = model_base_dir / sector_name
            if not sector_model_dir.exists():
                self.logger.warning(f"板块 {sector_name} 的模型目录不存在: {sector_model_dir}")
                error_count += 1
                continue

            # 查找所有该板块的模型文件，并提取日期
            available_models = []
            for model_file in sector_model_dir.glob("*.pkl"):
                try:
                    # 假设文件名是 YYYYMMDD.pkl 格式
                    train_date_str = model_file.stem
                    # 验证日期格式
                    try:
                        datetime.strptime(train_date_str, '%Y%m%d')
                        available_models.append((train_date_str, model_file))
                    except ValueError:
                        self.logger.warning(f"无法从文件名 {model_file.name} 解析日期，跳过。")
                except Exception as e:
                    self.logger.warning(f"处理模型文件 {model_file} 时出错: {e}")

            if not available_models:
                self.logger.warning(f"板块 {sector_name} 没有找到任何有效的模型文件。")
                error_count += 1
                continue

            # 筛选出日期 <= target_date 的模型
            relevant_models = [(d, f) for d, f in available_models if d <= date]

            if not relevant_models:
                self.logger.warning(f"板块 {sector_name} 没有找到在日期 {date} 或之前的可用模型。")
                error_count += 1
                continue

            # 找到最新的一个相关模型
            latest_model_date, latest_model_path = max(relevant_models, key=lambda item: item[0])

            # 加载模型
            try:
                with open(latest_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.sector_models[sector_name] = model_data
                self.logger.info(f"为板块 {sector_name} 加载了模型 (训练日期: {latest_model_date})，来自 {latest_model_path}")
                loaded_count += 1
            except Exception as e:
                self.logger.error(f"加载板块 {sector_name} 的模型 {latest_model_path} 失败: {e}")
                self.logger.debug(traceback.format_exc())
                error_count += 1

        self.logger.info(f"模型加载完成。成功加载 {loaded_count} 个板块模型，遇到 {error_count} 个错误。")
        return loaded_count > 0
    
    
    def train_sector_models(self, date=None, lookback_short=30, lookback_long=90, test_size=0.2, forward_days=5, parallel="process_pool"):
        """
        加载指定日期的预训练行业模型 (原训练逻辑已移除)

        参数:
            date: 目标日期，默认为最近交易日
            lookback_short, lookback_long, test_size, forward_days, parallel: 参数保留以兼容接口，但不再用于训练

        返回:
            bool: 如果成功为至少一个板块加载了模型，则返回True
        """
        if date is None:
            date = self._get_latest_date()
        
        logger.info(f"调用 train_sector_models 为日期 {date} 加载预训练模型 (训练逻辑已移除)。")
        # 忽略其他参数 lookback_short, lookback_long 等

        # 确保行业数据已加载，以便知道要为哪些板块加载模型
        if not hasattr(self, 'sectors') or not self.sectors:
            logger.info("板块数据未加载，尝试加载...")
            try:
                self._load_sector_data()
                if not self.sectors:
                    logger.error("加载板块数据后仍为空，无法加载模型。")
                    return False    
            except Exception as e:
                logger.error(f"加载板块数据时出错: {e}")
                return False

        # 调用 _load_models_for_date 方法加载模型
        return self._load_models_for_date(date)


    # 使用正则表达式计算开盘后的区间涨跌幅
    def calculate_intraday_range(self, open_price, high_price, low_price, prev_close=None):
        """计算日内区间涨跌幅
        
        参数:
            open_price: 开盘价
            high_price: 最高价
            low_price: 最低价
            prev_close: 昨收价，如果为None则使用开盘价计算
            
        返回:
            上涨区间幅度和下跌区间幅度
        """
        if prev_close is None:
            prev_close = open_price
            
        if prev_close <= 0 or open_price <= 0 or high_price <= 0 or low_price <= 0:
            return 0, 0
            
        up_range = (high_price - open_price) / prev_close * 100
        down_range = (low_price - open_price) / prev_close * 100
        
        return up_range, down_range
        
    def _get_closest_trading_date(self, date):
        """获取最接近的交易日
        
        参数:
            date: 日期字符串（YYYYMMDD格式）或datetime对象或时间戳
            
        返回:
            最接近的交易日，如果无法获取则返回None
        """
        # 确保交易日期已加载
        if not self.trading_dates:
            self.load_trading_dates()
            
        if not self.trading_dates:
            logger.warning("交易日期列表为空，无法查找最接近的交易日")
            return None
            
        try:
            # 确保date是字符串格式的YYYYMMDD
            if isinstance(date, (int, float)):
                # 如果是时间戳（秒），转换为YYYYMMDD格式
                date = datetime.fromtimestamp(date).strftime('%Y%m%d')
            elif isinstance(date, datetime):
                # 如果是datetime对象，转换为YYYYMMDD格式
                date = date.strftime('%Y%m%d')
            elif isinstance(date, str):
                # 如果是字符串，确保格式为YYYYMMDD
                if len(date) != 8 or not date.isdigit():
                    # 尝试解析其他常见格式
                    try:
                        if '-' in date or '/' in date:
                            # 尝试解析YYYY-MM-DD或YYYY/MM/DD格式
                            parsed_date = datetime.strptime(date, '%Y-%m-%d' if '-' in date else '%Y/%m/%d')
                            date = parsed_date.strftime('%Y%m%d')
                        else:
                            logger.warning(f"无法识别的日期格式: {date}，期望格式为YYYYMMDD")
                            return None
                    except ValueError:
                        logger.warning(f"无法解析日期: {date}")
                        return None
            else:
                logger.warning(f"不支持的日期类型: {type(date)}")
                return None
                
            # 如果日期已经在交易日列表中，直接返回
            if date in self.trading_dates:
                return date
                
            # 将日期转换为datetime对象进行比较
            target_date = datetime.strptime(date, '%Y%m%d')
            
            # 将交易日期也转换为datetime对象
            trading_dates_dt = [datetime.strptime(d, '%Y%m%d') for d in self.trading_dates]
            
            # 找到最接近的日期
            closest_date = min(trading_dates_dt, key=lambda d: abs((d - target_date).days))
            
            # 转回字符串格式
            return closest_date.strftime('%Y%m%d')
        except Exception as e:
            logger.error(f"寻找最接近的交易日期时出错: {e}")
            return None

    def _load_sector_data(self):
        """加载板块数据
        
        从库中加载股票行业板块映射数据，构建板块股票映射关系
        """
        logger.info("开始加载板块数据...")
        # 初始化板块数据结构
        self.sectors = {}
        
        # 获取股票列表
        stocks = self.get_stock_universe()
        if not stocks:
            logger.error("股票列表为空，无法加载板块数据")
            raise ValueError("股票列表为空，无法加载板块数据")
            
        logger.info(f"正在获取 {len(stocks)} 只股票的概念板块信息...")
        
        # 获取所有概念板块
        # 使用get_tng_sectors()获取所有可用的概念板块
        self.sector_stocks = get_tng_sectors()
        all_sectors = self.sector_stocks
        if not all_sectors:
            logger.error("获取概念板块列表失败")
            raise ValueError("获取概念板块列表失败")
            
        logger.info(f"成功获取 {len(all_sectors)} 个概念板块")
        
        # 获取股票-板块映射关系
        stock_sector_mapping = {}
        
        # 使用get_stock_sectors获取每只股票所属的概念板块
        for stock in stocks:
            # 获取该股票所属的概念板块
            stock_sectors = get_stock_sectors(stock)
            if not stock_sectors:
                logger.warning(f"股票 {stock} 没有概念板块数据")
                continue
                
            stock_sector_mapping[stock] = {
                'sector_names': stock_sectors,
                'sector_type': 'concept'
            }
        
        # 检查是否获取到足够的映射数据
        if not stock_sector_mapping:
            logger.error("未能获取到任何股票的板块信息")
            raise ValueError("未能获取到任何股票的板块信息")
        
        # 按板块组织股票
        for stock, sector_info in stock_sector_mapping.items():
            sector_names = sector_info.get('sector_names', [])
            for sector_name in sector_names:
                if sector_name not in self.sectors:
                    self.sectors[sector_name] = []
                if stock not in self.sectors[sector_name]:
                    self.sectors[sector_name].append(stock)
        
        # 检查是否成功构建板块数据
        if not self.sectors:
            logger.error("未能构建任何板块数据")
            raise ValueError("未能构建任何板块数据")
            
        # 读取板块表现排名（如果有的话）
        self.sector_performance = {}
        
        logger.info(f"成功加载了 {len(self.sectors)} 个板块的数据")
        return True

    def get_stock_universe(self):
        """获取股票池
        
        返回当前模型使用的股票池列表
        
        返回:
            list: 股票代码列表
        """
        try:
            # 如果已有预设的股票池，直接返回
            if hasattr(self, 'stock_pool') and self.stock_pool:
                return self.stock_pool
                
            # 获取A股主板股票列表
            a_share_stocks = []
            
            try:
                # 使用缓存系统获取行业股票（如果可用）
                if HAS_CACHE and self.cache_manager:
                    cached = False
                    
                    # 尝试从缓存获取股票池
                    logger.info("尝试从缓存获取股票池")
                    cached_stocks = self.cache_manager.get_stock_pool()
                    if cached_stocks and len(cached_stocks) > 0:
                        a_share_stocks = cached_stocks
                        logger.info(f"从缓存获取了 {len(a_share_stocks)} 只股票")
                        cached = True
                    
                    # 如果缓存中没有，则从API获取并保存到缓存
                    if not cached:
                        # 使用sector_list获取板块股票
                        sectors_to_get = ['上证主板', '深证主板', '中小板', '创业板', '科创板']
                        logger.info(f"从API获取股票池，目标板块: {sectors_to_get}")
                        
                        for sector in sectors_to_get:
                            try:
                                # 检查是否可以从缓存获取行业股票
                                sector_stocks = self.cache_manager.get_sector_stocks(sector)
                                if not sector_stocks:
                                    # 如果缓存中没有，则从API获取并保存到缓存
                                    sector_stocks = xtdata.get_stock_list_in_sector(sector)
                                    if sector_stocks:
                                        # 保存到缓存
                                        self.cache_manager.save_sector_stocks(sector, sector_stocks)
                                
                                if sector_stocks:
                                    a_share_stocks.extend(sector_stocks)
                                    logger.info(f"获取{sector}股票 {len(sector_stocks)} 只")
                            except Exception as e:
                                logger.error(f"获取{sector}股票时出错: {e}")
                        
                        if a_share_stocks:
                            # 去重
                            a_share_stocks = list(set(a_share_stocks))
                            # 保存到缓存
                            self.cache_manager.save_stock_pool(a_share_stocks)
                            logger.info(f"已将 {len(a_share_stocks)} 只股票保存到缓存")
                else:
                    # 不使用缓存系统，直接获取
                    # 使用sector_list获取板块股票，而不是get_stock_list_info
                    # 尝试获取上证主板、深证主板、中小板和创业板
                    for sector in ['上证主板', '深证主板', '中小板', '创业板', '科创板']:
                        try:
                            sector_stocks = xtdata.get_stock_list_in_sector(sector)
                            if sector_stocks:
                                a_share_stocks.extend(sector_stocks)
                        except:
                            pass
                
                if a_share_stocks:
                    # 去重
                    a_share_stocks = list(set(a_share_stocks))
                    logger.info(f"从板块获取了 {len(a_share_stocks)} 只A股股票")
                
            except Exception as e:
                logger.error(f"使用板块获取股票列表时出错: {e}")
                
            # 如果板块获取失败，使用备用方法
            if not a_share_stocks:
                # 尝试从配置文件或其他来源加载股票列表
                logger.warning("板块获取股票列表失败，尝试使用备用方法...")
                
                # 尝试获取某些重要指数的成分股
                try:
                    # 尝试获取上证50、沪深300或中证500的成分股
                    for index_code in ['000016.SH', '000300.SH', '000905.SH']:
                        try:
                            # 检查是否可以从缓存获取指数成分股
                            if HAS_CACHE and self.cache_manager:
                                index_stocks = self.cache_manager.get_sector_stocks(index_code)
                                if not index_stocks:
                                    # 如果缓存中没有，则从API获取并保存到缓存
                                    index_stocks = xtdata.get_stock_list_in_sector(index_code)
                                    if index_stocks:
                                        # 保存到缓存
                                        self.cache_manager.save_sector_stocks(index_code, index_stocks)
                            else:
                                index_stocks = xtdata.get_stock_list_in_sector(index_code)
                                
                            if index_stocks:
                                a_share_stocks.extend(index_stocks)
                                logger.info(f"获取到{index_code}指数成分股 {len(index_stocks)} 只")
                        except Exception as e:
                            logger.error(f"获取{index_code}成分股时出错: {e}")
                    
                    if a_share_stocks:
                        # 去重
                        a_share_stocks = list(set(a_share_stocks))
                        logger.info(f"从指数成分股获取了共 {len(a_share_stocks)} 只股票")
                        
                        # 如果使用缓存且之前没保存，现在保存
                        if HAS_CACHE and self.cache_manager and not hasattr(self, 'stock_pool'):
                            self.cache_manager.save_stock_pool(a_share_stocks)
                            logger.info(f"已将 {len(a_share_stocks)} 只股票保存到缓存")
                except Exception as e:
                    logger.error(f"获取指数成分股时出错: {e}")
                    
            # 如果仍然无法获取股票列表，使用硬编码的备用列表
            if not a_share_stocks:
                logger.warning("所有获取股票列表的方法都失败，使用默认股票列表")
                # 使用一些常见的大盘股作为备用
                a_share_stocks = ['600519', '601318', '600036', '600276', '000858', '601888']
                
            # 缓存股票池以供后续使用
            self.stock_pool = a_share_stocks
            
            return a_share_stocks
        except Exception as e:
            logger.error(f"获取股票池时出错: {e}")
            # 返回一个小的默认股票池
            return ['600519', '601318', '600036', '600276', '000858', '601888']

    def _train_with_ray(self, task_func, sectors=None, 
                       date=None, stock_data=None, indus=None, 
                       feature_cols=None, param_dict=None,
                       **kwargs):
        """使用Ray进行并行训练
        
        Args:
            task_func: 训练任务函数
            sectors: 待训练的行业列表，默认为全部
            date: 训练使用的日期
            stock_data: 股票数据
            indus: 行业数据
            feature_cols: 特征列名列表
            param_dict: 训练参数字典
            kwargs: 其他参数
            
        Returns:
            dict: 行业模型字典 {sector_name: model_dict}
        """
        if sectors is None and hasattr(self, 'sectors') and self.sectors:
            sectors = list(self.sectors.keys())
        elif sectors is None:
            self.logger.warning("未指定行业列表且未加载行业数据，无法进行训练")
            return {}
            
        # 尝试导入Ray
        try:
            import ray
        except ImportError:
            self.logger.error("无法导入Ray，将使用顺序处理")
            return self._train_sequential(task_func, sectors, date, stock_data, indus, feature_cols, param_dict, **kwargs)
            
        # 检查Ray是否已初始化
        if not ray.is_initialized():
            try:
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False,
                    local_mode=True,
                    logging_level="error"
                )
                self.logger.info("Ray初始化成功")
            except Exception as e:
                self.logger.error(f"Ray初始化失败: {e}")
                self.logger.warning("将使用顺序处理替代Ray")
                return self._train_sequential(task_func, sectors, date, stock_data, indus, feature_cols, param_dict, **kwargs)
                
        try:
            # 定义远程训练函数
            @ray.remote
            def train_sector_remote(sector, model_instance, feature_data=None, training_date=None, params=None, **task_kwargs):
                """远程执行的板块训练函数
                
                Args:
                    sector: 板块名称
                    model_instance: AdvancedQuantModel实例的序列化版本
                    feature_data: 特征数据
                    training_date: 训练日期
                    params: 训练参数
                    task_kwargs: 其他参数
                    
                Returns:
                    tuple: (板块名称, 训练结果)
                """
                try:
                    # 记录执行信息
                    import os
                    import socket
                    host = socket.gethostname()
                    pid = os.getpid()
                    print(f"在主机 {host} 的进程 {pid} 中训练板块 {sector}")
                    
                    # 提取该板块的特征和标签
                    # 注意：直接使用传入的model_instance，因为Ray不会序列化self引用
                    features_short, features_long, labels = model_instance._extract_features_and_labels(
                        sector, training_date, 
                        params.get('lookback_short', 60), 
                        params.get('lookback_long', 120),
                        params.get('forward_days', 20)
                    )
                    
                    if features_short is None or labels is None:
                        print(f"板块 {sector} 无法提取有效特征或标签")
                        return sector, None
                        
                    # 训练模型
                    model_dict = model_instance._train_model(
                        sector, features_short, features_long, labels, 
                        params.get('test_size', 0.3)
                    )
                    
                    # 确保模型可序列化
                    if model_dict is not None:
                        # 使用自定义序列化方法确保结果可序列化
                        serializable_model = model_instance._make_serializable_model(model_dict)
                        print(f"板块 {sector} 训练成功")
                        return sector, serializable_model
                    else:
                        print(f"板块 {sector} 训练失败，返回None")
                        return sector, None
                        
                except Exception as e:
                    import traceback
                    print(f"训练板块 {sector} 时出错: {e}")
                    print(traceback.format_exc())
                    return sector, None
                    
            # 提交任务
            self.logger.info(f"使用Ray并行训练 {len(sectors)} 个板块模型")
            futures = []
            
            # 处理参数
            safe_param_dict = self._make_serializable(param_dict or {})
            
            # 创建任务
            for sector in sectors:
                # 注意：不要传递self引用，因为可能包含不可序列化的对象
                # 相反，我们传递一个新创建的干净对象或必要的方法
                task = train_sector_remote.remote(
                    sector, 
                    self,  # Ray会自动序列化这个对象
                    feature_cols, 
                    date, 
                    safe_param_dict, 
                    **kwargs
                )
                futures.append(task)
                
            # 获取结果，设置超时避免挂起
            try:
                # 使用超时获取结果
                results = ray.get(futures, timeout=1800)  # 30分钟超时
                
                # 处理结果
                sector_models = {}
                for sector, model in results:
                    if model is not None:
                        sector_models[sector] = model
                        
                self.logger.info(f"Ray训练完成，成功训练了 {len(sector_models)}/{len(sectors)} 个板块模型")
                
                # 确保返回的字典是可序列化的
                return self._make_serializable(sector_models)
                
            except (ray.exceptions.RayTimeoutError, ray.exceptions.RayTaskError) as e:
                self.logger.error(f"Ray任务执行出错: {e}")
                # 尝试获取已完成的任务结果
                try:
                    completed_ids, _ = ray.wait(futures, num_returns=len(futures), timeout=0)
                    if completed_ids:
                        self.logger.info(f"尝试获取 {len(completed_ids)} 个已完成的任务结果")
                        partial_results = ray.get(completed_ids)
                        sector_models = {sector: model for sector, model in partial_results if model is not None}
                        self.logger.info(f"获取到 {len(sector_models)} 个有效结果")
                        return self._make_serializable(sector_models)
                except Exception as inner_e:
                    self.logger.error(f"获取部分结果时出错: {inner_e}")
                
        except Exception as e:
            self.logger.error(f"Ray训练过程中出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        finally:
            # 确保关闭Ray
            try:
                ray.shutdown()
                self.logger.info("Ray已关闭")
            except:
                pass
                
        # 出错时回退到顺序处理
        self.logger.warning("Ray训练失败，回退到顺序处理")
        return self._train_sequential(task_func, sectors, date, stock_data, indus, feature_cols, param_dict, **kwargs)

    def _train_sector_process(self, sector_name, date, lookback_short, lookback_long, 
                           test_size=0.3, forward_days=20):
        """在单独进程中训练单个板块的模型
        
        Args:
            sector_name: 板块名称
            date: 截止日期
            lookback_short: 短期回溯天数
            lookback_long: 长期回溯天数
            test_size: 测试集比例
            forward_days: 预测天数
            
        Returns:
            dict: 训练好的模型字典，包含模型和训练信息
        """
        try:
            self.logger.info(f"开始训练板块 '{sector_name}' 模型")
            
            # 提取特征和标签
            features_short, features_long, labels = self._extract_features_and_labels(
                sector_name, date, lookback_short, lookback_long, forward_days
            )
            
            # 如果特征或标签为空，记录原因并返回
            if features_short is None or features_long is None or labels is None:
                self.logger.warning(f"板块 '{sector_name}' 无法提取有效特征或标签")
                return None
                
            # 模型训练
            model_dict = self._train_model(sector_name, features_short, features_long, 
                                       labels, test_size)
            
            # 确保模型可序列化
            serializable_model = self._make_serializable_model(model_dict)
            
            self.logger.info(f"板块 '{sector_name}' 训练完成")
            return serializable_model
            
        except MemoryError:
            self.logger.error(f"训练板块 '{sector_name}' 时内存不足")
            return None
        except Exception as e:
            self.logger.error(f"训练板块 '{sector_name}' 时发生错误: {e}")
            return None
            
    def _make_serializable_model(self, model_dict):
        """确保模型字典完全可序列化，处理所有不可序列化的对象
        
        Args:
            model_dict: 模型字典
            
        Returns:
            dict: 可序列化的模型字典
        """
        if model_dict is None:
            return None
            
        # 创建一个新的可序列化字典
        serializable_dict = {}
        
        try:
            # 遍历模型字典中的所有项
            for key, value in model_dict.items():
                # 处理嵌套字典
                if isinstance(value, dict):
                    serializable_dict[key] = self._make_serializable_model(value)
                # 处理模型对象
                elif key == 'model':
                    # 根据模型类型进行不同处理
                    model_type = type(value).__name__
                    
                    # 对于特定类型的模型，我们保存其类型信息和特定属性
                    if model_type in ['RandomForestClassifier', 'ExtraTreesClassifier', 
                                      'GradientBoostingClassifier']:
                        # 保存基本信息
                        model_info = {
                            'type': model_type,
                            'n_estimators': getattr(value, 'n_estimators', 100),
                            'max_depth': getattr(value, 'max_depth', None),
                            'criterion': getattr(value, 'criterion', 'gini')
                        }
                        serializable_dict[key] = model_info
                    elif model_type in ['XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']:
                        # 保存基本信息
                        model_info = {
                            'type': model_type,
                            'n_estimators': getattr(value, 'n_estimators', 100),
                            'max_depth': getattr(value, 'max_depth', 3),
                            'learning_rate': getattr(value, 'learning_rate', 0.1)
                        }
                        serializable_dict[key] = model_info
                    elif model_type == 'EnsembleModelWrapper':
                        # 使用模型自己的序列化方法
                        if hasattr(value, 'to_dict'):
                            serializable_dict[key] = value.to_dict()
                        else:
                            # 基本信息
                            serializable_dict[key] = {
                                'type': 'EnsembleModelWrapper',
                                'weights': getattr(value, 'weights', (0.7, 0.3))
                            }
                    else:
                        # 对于其他模型，简单记录类型
                        serializable_dict[key] = {'type': model_type}
                # 处理其他可能的不可序列化对象
                elif isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                    # 转换为列表
                    serializable_dict[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
                else:
                    # 尝试直接添加，如果不可序列化，则转为字符串
                    try:
                        # 测试是否可序列化
                        json.dumps({key: value})
                        serializable_dict[key] = value
                    except (TypeError, OverflowError):
                        # 不可序列化，转为字符串
                        serializable_dict[key] = str(value)
        except Exception as e:
            self.logger.error(f"模型序列化处理失败: {e}")
            return {'error': str(e)}
            
        return serializable_dict

    def _train_with_process_pool(self, sector_names, date, lookback_short, lookback_long, 
                             test_size=0.3, forward_days=20):
        """使用进程池并行训练所有板块模型
        Args:
            sector_names: 板块名称列表
            date: 截止日期
            lookback_short: 短期回溯天数
            lookback_long: 长期回溯天数
            test_size: 测试集比例
            forward_days: 预测天数
        Returns:
            dict: 板块名称到模型字典的映射
        """
        if not sector_names:
            self.logger.warning(f"没有可训练的板块，跳过训练")
            return {}
        
        # 计算进程数
        max_processes = min(cpu_count(), len(sector_names), 8)  # 限制最大进程数
        
        # 检查可用内存 - 如果内存不足，减少进程数
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            if available_memory < 4:  # 如果可用内存小于4GB
                max_processes = min(max_processes, 2)  # 减少进程数
                self.logger.warning(f"可用内存不足 ({available_memory:.2f}GB)，减少进程数至 {max_processes}")
        except ImportError:
            self.logger.warning("无法导入psutil模块，无法监控内存使用情况")
        
        # 将板块分成批次以减少内存压力
        batch_size = min(max(1, max_processes), 5)  # 每批处理的板块数，最大为5
        num_batches = (len(sector_names) + batch_size - 1) // batch_size
        
        self.logger.info(f"开始并行训练 {len(sector_names)} 个板块模型，分为 {num_batches} 批次，"
                         f"每批次最多 {batch_size} 个板块，使用 {max_processes} 个进程")
        
        # 创建一个字典来存储结果
        model_dict = {}
        failed_sectors = []
        
        # 按批次处理
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sector_names))
            batch_sectors = sector_names[start_idx:end_idx]
            
            self.logger.info(f"处理批次 {batch_idx+1}/{num_batches}，"
                           f"包含 {len(batch_sectors)} 个板块")
            
            # 使用进程池执行训练
            results = {}
            failed_in_batch = []
            
            try:
                with ProcessPoolExecutor(max_workers=max_processes) as executor:
                    # 创建训练任务，使用独立顶级函数
                    futures = {
                        executor.submit(
                            _process_safe_train_sector, 
                            sector_name, 
                            date, 
                            lookback_short, 
                            lookback_long, 
                            test_size, 
                            forward_days
                        ): sector_name for sector_name in batch_sectors
                    }
                    
                    # 等待所有任务完成，使用超时防止挂起
                    for future in as_completed(futures, timeout=600):
                        sector_name = futures[future]
                        try:
                            # 获取训练结果
                            result_sector, model = future.result()
                            if model is not None:
                                # 结果已经序列化，直接使用
                                results[result_sector] = model
                                self.logger.info(f"板块 '{result_sector}' 训练成功")
                            else:
                                self.logger.warning(f"板块 '{sector_name}' 训练返回空结果")
                                failed_in_batch.append(sector_name)
                        except Exception as e:
                            self.logger.error(f"板块 '{sector_name}' 训练失败: {e}")
                            failed_in_batch.append(sector_name)
                
            except (TimeoutError, concurrent.futures.TimeoutError) as e:
                self.logger.error(f"批次 {batch_idx+1} 训练超时: {e}")
                failed_in_batch.extend([s for s in batch_sectors if s not in results])
            except Exception as e:
                self.logger.error(f"批次 {batch_idx+1} 训练出错: {e}")
                failed_in_batch.extend([s for s in batch_sectors if s not in results])
                
            # 更新结果
            model_dict.update(results)
            failed_sectors.extend(failed_in_batch)
        
        # 顺序处理失败的板块
        if failed_sectors:
            self.logger.warning(f"{len(failed_sectors)} 个板块在并行处理中失败，尝试顺序执行...")
            sequential_results = self._train_sequential(failed_sectors, date, 
                                                        lookback_short, lookback_long, 
                                                        test_size, forward_days)
            if sequential_results:
                model_dict.update(sequential_results)
                self.logger.info(f"顺序执行成功修复 {len(sequential_results)} 个板块")
                
                # 更新失败列表
                final_failed = [s for s in failed_sectors if s not in sequential_results]
                if final_failed:
                    self.logger.warning(f"最终有 {len(final_failed)} 个板块训练失败: {final_failed[:5]}"
                                      f"{' 等' if len(final_failed) > 5 else ''}")
        
        return model_dict

    def _train_sequential(self, sector_names, date, lookback_short, lookback_long, test_size, forward_days):
        """顺序执行各个板块的训练
        
        参数:
            sector_names: 板块名称列表
            date: 当前训练截止日期
            lookback_short: 短期回溯天数
            lookback_long: 长期回溯天数
            test_size: 测试集比例
            forward_days: 预测未来天数
            
        返回:
            训练成功返回True，否则返回False
        """
        try:
            # 顺序训练各个板块
            success_count = 0
            
            for sector_name in tqdm(sector_names, desc="训练板块"):
                if sector_name in self.sectors:
                    stocks = self.sectors[sector_name]
                    logger.info(f"开始训练板块: {sector_name}, 股票数量: {len(stocks)}")
                    
                    # 提取特征和标签
                    features, labels = self._extract_features(stocks, date, lookback_short, forward_days)
                    
                    if features is None or labels is None:
                        logger.warning(f"板块 {sector_name} 无法提取有效特征，跳过训练")
                        continue
                    
                    # 训练模型
                    model = self._train_model(features, labels, test_size)
                    
                    if model is None:
                        logger.warning(f"板块 {sector_name} 模型训练失败")
                        continue
                    
                    # 提取长期特征
                    if lookback_long > lookback_short:
                        long_features, long_labels = self._extract_features(stocks, date, lookback_long, forward_days)
                        if long_features is not None and long_labels is not None:
                            # 训练长期模型
                            long_model = self._train_model(long_features, long_labels, test_size)
                            if long_model is not None:
                                # 组合长短期模型
                                model = self._combine_models(model, long_model)
                    
                    # 保存模型
                    self.sector_models[sector_name] = model
                    success_count += 1
                    logger.info(f"板块 {sector_name} 训练成功")
            
            success_rate = success_count / len(sector_names) if sector_names else 0
            logger.info(f"顺序训练完成，成功率: {success_rate:.2%} ({success_count}/{len(sector_names)})")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"顺序训练时出错: {e}")
            return False
            
    def _train_model(self, features, labels, test_size=0.2):
        """训练模型
        
        参数:
            features: 特征数据
            labels: 标签数据
            test_size: 测试集比例
            
        返回:
            训练好的模型，失败返回None
        """
        try:
            if features.shape[0] < 10:
                logger.warning(f"样本数量过少 ({features.shape[0]}), 无法训练模型")
                return None
                
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42
            )
            
            # 创建模型
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # 改为单线程训练，避免创建线程锁
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"模型训练完成，测试集准确率: {accuracy:.4f}")
            
            # 确保返回的模型是可序列化的
            model_dict = {
                'model': model,
                'accuracy': accuracy,
                'n_features': features.shape[1],
                'n_samples': features.shape[0],
                'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
            }
            
            # 处理模型确保可序列化
            serializable_model = self._make_serializable(model_dict)
            
            return serializable_model
            
        except Exception as e:
            logger.error(f"训练模型时出错: {e}")
            return None
            
    def _combine_models(self, short_model, long_model, weights=(0.7, 0.3)):
        """
        组合短期和长期模型
        
        参数:
            short_model: 短期模型
            long_model: 长期模型
            weights: 模型权重, 默认短期模型权重0.7，长期模型权重0.3
        
        返回:
            组合后的模型，确保可序列化
        """
        try:
            # 使用外部定义的类而不是局部类
            combined_model = self.EnsembleModelWrapper(short_model, long_model, weights)
            
            # 创建包含组合模型信息的字典
            model_info = {
                'type': 'EnsembleModelWrapper',
                'short_model': short_model,
                'long_model': long_model,
                'weights': weights,
                'combined_model': combined_model
            }
            
            # 确保可序列化
            serializable_model = self._make_serializable(model_info)
            return serializable_model
            
        except Exception as e:
            logger.error(f"组合模型时出错: {e}，将返回短期模型")
            # 如果组合失败，返回短期模型
            return short_model
        
    def _save_models(self, sector_models, model_dir=None):
        """保存板块模型到磁盘
        
        Args:
            sector_models: 板块模型字典
            model_dir: 模型保存目录，默认为self.model_dir
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        if sector_models is None or len(sector_models) == 0:
            self.logger.warning("没有模型需要保存")
            return False
            
        if model_dir is None:
            model_dir = self.model_dir
            
        # 确保模型目录存在
        try:
            os.makedirs(model_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"创建模型目录 '{model_dir}' 失败: {e}")
            return False
            
        # 将模型转换为可序列化的形式
        self.logger.info(f"开始保存 {len(sector_models)} 个板块模型到 '{model_dir}'")
        serializable_models = self._make_serializable(sector_models)
        
        # 将模型保存到文件
        try:
            model_path = os.path.join(model_dir, f"sector_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(serializable_models, f)
                
            self.logger.info(f"成功保存模型到 '{model_path}'")
            
            # 保存模型元数据
            success_count = sum(1 for model in serializable_models.values() if model.get('training_success', False))
            metadata = {
                'saved_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_sectors': len(serializable_models),
                'success_sectors': success_count,
                'model_path': model_path
            }
            
            meta_path = os.path.join(model_dir, f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"成功保存模型元数据到 '{meta_path}'")
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _load_models(self, model_path=None):
        """加载训练好的模型
        
        参数:
            model_path: 模型文件路径，如果为None则加载最新的模型
            
        返回:
            成功返回True，失败返回False
        """
        try:
            # 如果未指定模型路径，加载最新的模型
            if model_path is None:
                if not os.path.exists(MODEL_DIR):
                    logger.warning(f"模型目录不存在: {MODEL_DIR}")
                    return False
                    
                # 获取最新的模型文件
                model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('sector_models_') and f.endswith('.pkl')]
                if not model_files:
                    logger.warning("没有找到可用的模型文件")
                    return False
                    
                # 按时间排序
                model_files.sort(reverse=True)
                model_path = os.path.join(MODEL_DIR, model_files[0])
                
            # 加载模型
            with open(model_path, 'rb') as f:
                self.sector_models = pickle.load(f)
                
            logger.info(f"成功加载模型: {model_path}, 共 {len(self.sector_models)} 个板块模型")
            return True
            
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            return False




    def _extract_features(self, stocks, date, lookback_days, forward_days=5):
        """
        使用预计算的特征文件提取特征和标签

        Args:
            stocks (list): 股票代码列表
            date (str): 当前日期 (YYYYMMDD)
            lookback_days (int): 历史回溯天数 (在这个版本中不再直接使用，因为特征已包含历史信息)
            forward_days (int): 预测未来天数，用于计算标签

        Returns:
            tuple: (features_df, labels)
                   features_df (pd.DataFrame): 包含所选股票在指定日期的预计算特征, index为stock_code
                   labels (pd.Series): 包含所选股票的未来收益率标签, index为stock_code
        """
        self.logger.info(f"开始使用预计算特征提取: 日期={date}, 股票数={len(stocks)}")
        # 导入必要的库
        import pandas as pd
        import numpy as np
        import json
        from pathlib import Path
        import traceback # 确保导入 traceback

        if not stocks:
            self.logger.warning("股票列表为空，无法提取特征")
            return None, None
                    
        # --- 路径定义 ---
        # 假设 ParquetDataStorage 实例存储在 self.storage
        # 并且 base_dir 指向 'to/local_data'
        if not hasattr(self, 'storage') or self.storage is None:
            self.logger.error("数据存储对象 (self.storage) 未初始化")
            # 尝试创建一个默认实例，但这可能不是最佳做法
            try:
                from advanced_quant_data import ParquetDataStorage
                self.storage = ParquetDataStorage()
                base_data_dir = Path(self.storage.base_dir) # 使用 storage 的 base_dir
                self.logger.warning("自动初始化了一个 ParquetDataStorage 实例")
            except Exception as storage_init_err:
                self.logger.error(f"自动初始化 ParquetDataStorage 失败: {storage_init_err}. 无法确定数据路径。")
                return None, None
        else:
             base_data_dir = Path(self.storage.base_dir) # 使用 storage 的 base_dir

        feature_dir = base_data_dir / "features" / "daily"
        market_status_dir = base_data_dir / "market_status" / "daily"

        feature_file = feature_dir / f"{date}.parquet"
        market_status_file = market_status_dir / f"{date}.json"

        # --- 1. 加载预计算的股票特征 ---
        if not feature_file.exists():
            self.logger.error(f"特征文件不存在: {feature_file}")
            # 可以选择尝试加载最近的可用文件，或直接返回错误
            # TODO: 实现寻找最近可用特征文件的逻辑 (self._find_closest_available_feature_date)
            return None, None # 如果无法找到特征文件，则无法继续

        try:
            daily_features_all = pd.read_parquet(feature_file)
            # 根据请求的股票列表筛选特征
            features_df = daily_features_all[daily_features_all['stock_code'].isin(stocks)].copy()
            if features_df.empty:
                 self.logger.warning(f"在特征文件 {feature_file} 中未找到请求的股票 {stocks} 的数据")
                 return None, None
            # 将 stock_code 设置为索引，方便后续合并和标签对齐
            features_df = features_df.set_index('stock_code')
            self.logger.info(f"成功加载并筛选了 {len(features_df)} 只股票的特征，来自 {feature_file}")
        except Exception as e:
            self.logger.error(f"加载或处理特征文件 {feature_file} 时出错: {e}")
            self.logger.debug(traceback.format_exc())
                return None, None
                
        # --- 2. 加载市场状态数据 ---
        market_status_data = {}
        if not market_status_file.exists():
            self.logger.warning(f"市场状态文件不存在: {market_status_file}，将不包含市场状态特征")
            # 可以选择回退或忽略
        else:
            try:
                with open(market_status_file, 'r', encoding='utf-8') as f:
                    market_status_raw = json.load(f)
                # 提取需要的市场状态指标
                # 假设 market_status_processor.py 保存的json直接是结果字典
                # 如果有嵌套，例如 'analysis_result'，则需要调整
                # market_status_data = market_status_raw.get('analysis_result', {}) # 更安全的访问方式
                market_status_data = market_status_raw # 直接使用顶层字典
                if market_status_data:
                     self.logger.info(f"成功加载市场状态数据，来自 {market_status_file}")
                else:
                     self.logger.warning(f"市场状态文件 {market_status_file} 加载后为空或结构不符。")

            except json.JSONDecodeError as jde:
                 self.logger.error(f"解析市场状态文件 {market_status_file} 时出错 (JSON格式错误): {jde}")
            except Exception as e:
                self.logger.error(f"加载或处理市场状态文件 {market_status_file} 时出错: {e}")
                self.logger.debug(traceback.format_exc())
                # 出错时不添加市场状态特征

        # --- 3. 合并市场状态到特征DataFrame ---
        if market_status_data:
             # 将市场状态字典的每个键值对作为新列添加到 features_df
             added_market_cols = 0
             for key, value in market_status_data.items():
                  # 确保列名唯一，例如加上 "market_" 前缀
                  market_feature_name = f"market_{key}"
                  # 检查列名是否已存在，避免覆盖
                  if market_feature_name in features_df.columns:
                      self.logger.warning(f"市场状态特征列 '{market_feature_name}' 已存在于特征中，将被覆盖。")
                  features_df[market_feature_name] = value # 将单一值广播到所有行
                  added_market_cols += 1
             if added_market_cols > 0:
                  self.logger.info(f"已将 {added_market_cols} 个市场状态指标合并到特征中")
             else:
                  self.logger.info("市场状态数据已加载但未找到可合并的指标")
        else:
             self.logger.info("未加载或合并市场状态指标")


        # --- 4. 计算未来收益率标签 (Label) ---
        # 这部分仍然需要查询未来的价格数据
        labels = None # 初始化标签变量
        try:
            # --- 4a. 确定标签计算所需的结束日期 ---
            # 确保 self.trading_dates 存在且已加载
            if not hasattr(self, 'trading_dates') or not self.trading_dates:
                # 尝试加载交易日历
                self.load_trading_dates()
                if not self.trading_dates:
                    self.logger.error("交易日历 (self.trading_dates) 未加载且无法自动加载，无法计算标签")
                    return None, None # 必须有交易日历

            trading_dates = self.trading_dates # 假设 self.trading_dates 已加载

            # 健壮性检查：确保 date 在 trading_dates 中
            if date not in trading_dates:
                 # 尝试查找最接近的过去交易日
                 past_dates = [td for td in trading_dates if td < date]
                 if not past_dates:
                     self.logger.error(f"指定日期 {date} 不在交易日历中，且没有更早的交易日。无法计算标签。")
                     return None, None
                 closest_past_date = max(past_dates)
                 self.logger.warning(f"指定日期 {date} 不是交易日，使用最近的过去交易日 {closest_past_date} 代替进行标签计算。")
                 date_for_label = closest_past_date # 使用找到的最近日期继续
            else:
                 date_for_label = date

            current_date_index = trading_dates.index(date_for_label)
            label_end_index = current_date_index + forward_days

            # 检查结束日期是否超出日历范围
            if label_end_index >= len(trading_dates):
                self.logger.warning(f"交易日历数据不足以计算未来 {forward_days} 天的收益率，当前日期: {date_for_label}, 需要日期索引: {label_end_index}, 最大索引: {len(trading_dates)-1}")
                # 返回特征和全NaN的标签，由调用者处理
                labels = pd.Series(np.nan, index=features_df.index, name='label')
                return features_df, labels

            label_end_date = trading_dates[label_end_index]

            # --- 4b. 获取未来收盘价 ---
            # 确保 self.storage 存在且已初始化
            if not hasattr(self, 'storage') or self.storage is None:
                 self.logger.error("数据存储对象 (self.storage) 未初始化，无法获取未来价格。")
                 # 返回特征，标签为None或NaN
                 labels = pd.Series(np.nan, index=features_df.index, name='label')
                 return features_df, labels

            # 获取计算标签所需的未来收盘价 (只需'close'字段)
            future_prices_data = self.storage.get_daily_data(
                stock_list=features_df.index.tolist(), # 只获取已有特征的股票
                start_date=label_end_date,
                end_date=label_end_date, # 只需这一天的数据
                fields=['close']
            )

            if future_prices_data is None or future_prices_data.empty:
                self.logger.warning(f"无法获取日期 {label_end_date} 的收盘价数据来计算标签")
                # 返回特征和全NaN的标签
                labels = pd.Series(np.nan, index=features_df.index, name='label')
                return features_df, labels


            # --- 4c. 计算标签 ---
            # 将未来价格数据整理成 Series (stock_code -> future_close)
            future_prices = future_prices_data.set_index('stock_code')['close']

            # 获取当前日期的收盘价 (从已加载的特征中获取)
            if 'close' not in features_df.columns:
                self.logger.error(f"预计算的特征缺少 'close' 列，无法计算标签。特征文件: {feature_file}")
                # 返回特征和全NaN的标签
                labels = pd.Series(np.nan, index=features_df.index, name='label')
                return features_df, labels
            current_prices = features_df['close']

            # 计算未来收益率 (确保对齐索引)
            # 使用 .loc 对齐，避免潜在的顺序问题
            future_prices_aligned = future_prices.reindex(features_df.index)
            current_prices_aligned = current_prices.reindex(features_df.index) # 虽然已经是Series，对齐更保险

            # 避免除以零或NaN
            current_prices_safe = current_prices_aligned.replace(0, np.nan)

            labels = (future_prices_aligned - current_prices_aligned) / current_prices_safe
            labels = labels.reindex(features_df.index) # 再次确保与 features_df 的索引完全一致
            labels.name = 'label' # 命名Series

            # 处理计算过程中可能出现的NaN或inf (例如，当前价格为0或NaN, 未来价格为NaN)
            labels = labels.replace([np.inf, -np.inf], np.nan)
            # 注意：这里不填充NaN，让模型自己处理或在训练前处理

            # 检查标签计算结果
            valid_label_count = labels.notna().sum()
            if valid_label_count == 0:
                 self.logger.warning(f"计算出的未来收益率标签全部为NaN，日期: {date}")
                 # 这里可能需要更详细的检查，看是未来价格问题还是当前价格问题
                 self.logger.debug(f"Current prices head (first 5 non-NaN):\n{current_prices_aligned.dropna().head()}")
                 self.logger.debug(f"Future prices head (first 5 non-NaN):\n{future_prices_aligned.dropna().head()}")
                 # 即使全NaN，也返回，让调用者决定如何处理
                            else:
                self.logger.info(f"成功计算了 {valid_label_count} 个有效标签，日期: {date}")

        except ValueError as ve:
             # 处理 date 不在 trading_dates 中的情况
             self.logger.error(f"日期 {date} 或 {date_for_label} 在交易日历中查找失败: {ve}")
             return None, None  # 修复缩进，确保与except块对齐
                except Exception as e:
            self.logger.error(f"计算未来收益率标签时出错: {e}")
            self.logger.debug(traceback.format_exc())
            labels = pd.Series(np.nan, index=features_df.index, name='label') # 返回NaN标签

        # --- 5. 返回结果 ---
        # 确保返回的 features_df 不包含可能意外加入的标签列
        if labels is not None and labels.name in features_df.columns:
             features_df = features_df.drop(columns=[labels.name])

        # 如果标签未能成功计算（例如出错），labels 可能是 None 或全 NaN Series
        if labels is None:
            # 创建一个全 NaN 的 Series 以保持返回类型一致
            labels = pd.Series(np.nan, index=features_df.index, name='label')

        return features_df, labels
    
            
    def _check_stock_data_availability(self, market_data, stocks, fields):
        """检查股票数据可用性
        
        参数:
            market_data: 市场数据
            stocks: 股票列表
            fields: 数据字段列表
            
        返回:
            有效股票列表
        """
        valid_stocks = []
        
        # 记录统计信息，用于诊断
        total_checked = 0
        missing_field_count = 0
        insufficient_data_count = 0
        all_nan_count = 0
        
        for stock in stocks:
            total_checked += 1
            is_valid = True
            reason = None
            
            # 检查每个字段
            for field in fields:
                # 检查字段是否存在
                if field not in market_data:
                    reason = f"缺少字段 {field}"
                    missing_field_count += 1
                    is_valid = False
                    break
                
                # 检查股票是否在该字段的DataFrame中
                if stock not in market_data[field].index:
                    reason = f"在字段 {field} 中不存在"
                    missing_field_count += 1
                    is_valid = False
                    break
                
                # 获取该股票的数据
                stock_data = market_data[field].loc[stock]
                
                # 检查数据点是否足够
                if len(stock_data) < 20:  # 至少需要20个数据点来计算技术指标
                    reason = f"字段 {field} 数据点不足: {len(stock_data)}"
                    insufficient_data_count += 1
                    is_valid = False
                    break
                
                # 检查是否全部为NaN
                if stock_data.isna().all():
                    reason = f"字段 {field} 数据全为NaN"
                    all_nan_count += 1
                    is_valid = False
                    break
                
                # 检查非NaN值的比例，确保大部分数据有效
                valid_ratio = stock_data.count() / len(stock_data)
                if valid_ratio < 0.8:  # 要求至少80%的数据有效
                    reason = f"字段 {field} 有效数据比例过低: {valid_ratio:.2f}"
                    insufficient_data_count += 1
                    is_valid = False
                    break
            
            if is_valid:
                valid_stocks.append(stock)
            else:
                logger.debug(f"股票 {stock} 数据无效: {reason}")
        
        # 记录统计信息
        logger.info(f"数据可用性检查: 总计 {total_checked} 只股票, 有效 {len(valid_stocks)} 只")
        logger.info(f"无效原因统计: 缺少字段 {missing_field_count}, 数据点不足 {insufficient_data_count}, 全NaN {all_nan_count}")
        
        return valid_stocks

    def _check_system_resources(self):
        """检查系统资源状态，返回系统资源信息
        
        返回:
            dict: 包含CPU、内存、磁盘等资源信息的字典
        """
        resources = {
            "cpu_count": cpu_count(),
            "memory_available_gb": None,
            "memory_total_gb": None,
            "disk_free_gb": None,
            "system": None,
            "python_version": None,
            "can_use_multiprocessing": True
        }
        
        # 检测系统类型
        try:
            import platform
            resources["system"] = platform.system()
            resources["python_version"] = platform.python_version()
        except:
            pass
            
        # 检测内存状态
        try:
            import psutil
            vm = psutil.virtual_memory()
            resources["memory_available_gb"] = vm.available / (1024 * 1024 * 1024)
            resources["memory_total_gb"] = vm.total / (1024 * 1024 * 1024)
            
            # 检测磁盘状态
            disk = psutil.disk_usage('/')
            resources["disk_free_gb"] = disk.free / (1024 * 1024 * 1024)
            
            # 检测CPU使用率
            resources["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            # 内存不足时发出警告
            if resources["memory_available_gb"] < 2.0:  # 少于2GB可用内存
                logger.warning(f"系统可用内存不足: {resources['memory_available_gb']:.1f}GB，可能影响训练性能")
                if resources["memory_available_gb"] < 1.0:  # 少于1GB时建议不使用多进程
                    logger.warning("可用内存严重不足，建议不使用多进程训练")
                    resources["can_use_multiprocessing"] = False
        except ImportError:
            logger.warning("无法导入psutil模块，无法检查系统资源状态")
        except Exception as e:
            logger.warning(f"检查系统资源时出错: {e}")
            
        # 记录系统资源信息
        resource_info = ", ".join([f"{k}: {v}" for k, v in resources.items() if v is not None])
        logger.info(f"系统资源状态: {resource_info}")
        
        return resources

    def preload_data(self, start_date=None, end_date=None, stock_list=None):
        """预加载常用数据到缓存
        
        参数:
            start_date: 开始日期，默认为最近30天
            end_date: 结束日期，默认为今天
            stock_list: 股票列表，默认为全部股票
            
        返回:
            成功返回True，否则返回False
        """
        if not HAS_CACHE or not self.cache_manager:
            logger.warning("缓存系统未启用，无法预加载数据")
            return False
        
        try:
            # 如果未指定股票列表，使用全部股票
            if stock_list is None:
                stock_list = self.get_stock_universe()
                # 限制预加载的股票数量以避免内存溢出
                if len(stock_list) > 300:
                    logger.info(f"限制预加载的股票数量为前300只，原数量: {len(stock_list)}")
                    stock_list = stock_list[:300]
            
            # 如果未指定日期范围，使用默认值
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            # 预加载交易日期
            logger.info("预加载交易日期...")
            self.load_trading_dates()
            
            # 预加载日线行情数据
            logger.info(f"预加载日线行情数据，股票数量: {len(stock_list)}, 日期范围: {start_date}-{end_date}")
            fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
            
            # 使用并行预加载以提高性能
            self.cache_manager.parallel_preload(
                stock_list=stock_list,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
                periods=['1d'],
                batch_size=50
            )
            
            logger.info("数据预加载完成")
            return True
        
        except Exception as e:
            logger.error(f"预加载数据时出错: {e}")
            return False

    def benchmark_performance(self, 
                             test_size: int = 10000, 
                             trials: int = 3, 
                             feature_subset_size: int = None) -> Dict:
        """性能测试函数
        
        测试各种计算密集型操作的性能，比较原始实现和Numba加速版本
        
        Args:
            test_size: 测试数据大小
            trials: 重复测试次数
            feature_subset_size: 特征提取测试的子集大小，None时使用完整数据
        
        Returns:
            Dict: 性能测试结果
        """
        if not HAS_NUMBA_ACCELERATOR or self.numba_accelerator is None:
            logger.warning("Numba加速器未启用，无法进行性能对比")
            return {"error": "Numba加速器未启用"}
        
        # 开启性能测试模式
        self.numba_accelerator.enable_benchmark_mode(True)
        
        # 生成测试数据
        np.random.seed(42)
        open_prices = np.random.rand(test_size) * 100 + 50
        high_prices = open_prices * (1 + np.random.rand(test_size) * 0.1)
        low_prices = open_prices * (1 - np.random.rand(test_size) * 0.1)
        prev_closes = open_prices * (1 + (np.random.rand(test_size) * 0.06 - 0.03))
        
        prices = np.random.rand(test_size) * 100 + 50
        if feature_subset_size is None:
            feature_subset_size = min(10000, test_size)
        
        results = {
            "intraday_range": {"原始版本": 0, "Numba版本": 0},
            "speedup": {}
        }
        
        # 保存原始函数
        if not hasattr(self.numba_accelerator, '_original_calculate_intraday_range'):
            logger.error("无法获取原始函数引用，性能测试中止")
            return {"error": "无法获取原始函数引用"}
        
        original_func = self.numba_accelerator.original_functions.get("calculate_intraday_range")
        if original_func is None:
            logger.error("无法获取原始函数引用，性能测试中止")
            return {"error": "无法获取原始函数引用"}
        
        # 测试原始版本
        logger.info(f"开始测试原始版本，数据大小: {test_size}, 重复次数: {trials}")
        
        # 临时恢复原始函数
        temp_func = self.__class__.calculate_intraday_range
        self.__class__.calculate_intraday_range = original_func
        
        start_time = time.time()
        for _ in range(trials):
            for i in range(test_size):
                self.calculate_intraday_range(
                    open_prices[i], high_prices[i], low_prices[i], prev_closes[i]
                )
        results["intraday_range"]["原始版本"] = (time.time() - start_time) / trials
        
        # 恢复Numba版本
        self.__class__.calculate_intraday_range = temp_func
        
        # 测试Numba版本 - 先编译
        logger.info("编译Numba函数...")
        self.calculate_intraday_range(100, 110, 95, 100)
        
        # 然后测试批量计算
        logger.info(f"开始测试Numba版本，数据大小: {test_size}, 重复次数: {trials}")
        start_time = time.time()
        for _ in range(trials):
            self.calculate_intraday_range(
                open_prices, high_prices, low_prices, prev_closes
            )
        results["intraday_range"]["Numba版本"] = (time.time() - start_time) / trials
        
        # 计算加速比
        if results["intraday_range"]["原始版本"] > 0:
            results["speedup"]["intraday_range"] = (
                results["intraday_range"]["原始版本"] / 
                results["intraday_range"]["Numba版本"]
            )
        
        # 获取Numba加速器的性能报告
        numba_report = self.numba_accelerator.generate_performance_report()
        results["numba_report"] = numba_report
        
        # 关闭性能测试模式
        self.numba_accelerator.enable_benchmark_mode(False)
        
        logger.info(f"性能测试完成，加速比: {results['speedup'].get('intraday_range', 0):.2f}x")
        return results

    def _setup_logging(self):
        """设置日志系统"""
        os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("to/log/advanced_quant_model.log"),
                logging.StreamHandler()
            ]
        )
        logger.info("初始化高级量化模型")

    def _load_config(self):
        """加载配置文件
        
        如果配置文件不存在，则返回默认配置
        """
        default_config = {
            "model": {
                "max_lookback_days": 60,
                "forward_days": 5,
                "feature_cols": ["ma_cross", "volume_ratio", "rsi", "intraday_range"]
            },
            "backtest": {
                "initial_capital": 1000000,
                "commission_rate": 0.0003,
                "slippage": 0.0002,
                "position_limit": 0.10
            },
            "data": {
                "market": "CN",
                "cache_dir": self.cache_dir,
                "use_cache": self.use_cache
            }
        }
        
        if self.config_file is None or not os.path.exists(self.config_file):
            logger.warning(f"未找到配置文件: {self.config_file}，使用默认配置")
            return default_config
        
        try:
            import json
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件时出错: {e}，使用默认配置")
            return default_config

    def _get_latest_date(self):
        """获取最新交易日期"""
        if not self.trading_dates:
            self.load_trading_dates()
        
        if self.trading_dates:
            return self.trading_dates[-1]
        else:
            # 如果无法获取交易日历，使用当前日期
            today = datetime.now().strftime('%Y%m%d')
            logger.warning(f"无法获取交易日历，使用当前日期: {today}")
            return today

    def _make_serializable(self, obj):
        """确保对象可序列化，将不可序列化的对象转换为可序列化的字典表示
        
        Args:
            obj: 任意Python对象
            
        Returns:
            object: 可序列化的对象
        """
        # 如果对象为None，直接返回
        if obj is None:
            return None
            
        # 尝试测试对象是否可序列化
        try:
            # 尝试使用json序列化对象
            json.dumps(obj)
            # 如果成功，直接返回原对象
            return obj
        except (TypeError, OverflowError, ValueError):
            # 对象不能被json序列化，继续处理
            pass
            
        # 处理不同类型的对象
        if isinstance(obj, dict):
            # 递归处理字典中的每个值
            return {k: self._make_serializable(v) for k, v in obj.items()}
            
        elif isinstance(obj, list):
            # 递归处理列表中的每个元素
            return [self._make_serializable(item) for item in obj]
            
        elif isinstance(obj, tuple):
            # 将元组转换为列表后再处理
            return [self._make_serializable(item) for item in obj]
            
        elif isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
            # 处理numpy数组和pandas对象
            try:
                # 尝试转换为列表
                return obj.tolist() if hasattr(obj, 'tolist') else obj.values.tolist()
            except:
                # 如果转换失败，尝试转换为字符串
                return str(obj)
                
        elif isinstance(obj, (np.integer, np.floating)):
            # 处理numpy标量类型
            return obj.item()
            
        elif isinstance(obj, (datetime, date)):
            # 处理日期时间对象
            return obj.isoformat()
            
        elif hasattr(obj, '__dict__'):
            # 处理自定义类对象
            # 首先检查对象是否有to_dict方法
            if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                try:
                    return obj.to_dict()
                except:
                    pass
                    
            # 尝试转换为字典表示
            try:
                # 获取对象的类名
                class_name = obj.__class__.__name__
                
                # 处理常见的机器学习模型
                if class_name in ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier']:
                    # RF和其他树模型的基本信息
                    return {
                        'type': class_name,
                        'n_estimators': getattr(obj, 'n_estimators', 100),
                        'max_depth': getattr(obj, 'max_depth', None),
                        'criterion': getattr(obj, 'criterion', 'gini'),
                        'is_serialized_model': True
                    }
                    
                elif class_name in ['XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']:
                    # XGB和其他梯度提升模型的基本信息
                    return {
                        'type': class_name,
                        'n_estimators': getattr(obj, 'n_estimators', 100),
                        'max_depth': getattr(obj, 'max_depth', 3),
                        'learning_rate': getattr(obj, 'learning_rate', 0.1),
                        'is_serialized_model': True
                    }
                    
                elif class_name == 'EnsembleModelWrapper':
                    # 处理自定义的集成模型
                    return {
                        'type': 'EnsembleModelWrapper',
                        'short_model': self._make_serializable(getattr(obj, 'short_model', None)),
                        'long_model': self._make_serializable(getattr(obj, 'long_model', None)),
                        'weights': getattr(obj, 'weights', (0.7, 0.3)),
                        'is_serialized_model': True
                    }
                    
                else:
                    # 其他类对象，提取一般属性
                    serializable_dict = {'type': class_name, 'is_serialized_object': True}
                    
                    # 添加对象的属性，排除私有属性和方法
                    for attr, value in vars(obj).items():
                        if not attr.startswith('_') and not callable(value):
                            try:
                                serializable_dict[attr] = self._make_serializable(value)
                            except:
                                serializable_dict[attr] = str(value)
                                
                    return serializable_dict
            except:
                # 如果无法转换为有意义的字典，则转为字符串
                return str(obj)
                
        # 对于其他类型，转换为字符串
        return str(obj)

    def _extract_features_and_labels(self, sector_name, date, lookback_short, lookback_long, forward_days):
        """提取特定板块的短期和长期特征及标签
        
        Args:
            sector_name: 板块名称
            date: 截止日期
            lookback_short: 短期回溯天数
            lookback_long: 长期回溯天数
            forward_days: 预测天数
            
        Returns:
            tuple: (短期特征, 长期特征, 标签)，提取失败时返回None
        """
        try:
            # 确保板块数据已加载
            if not hasattr(self, 'sectors') or not self.sectors:
                self.logger.warning("板块数据未加载，尝试加载...")
                self._load_sector_data()
                
            # 检查板块是否存在
            if sector_name not in self.sectors:
                self.logger.warning(f"板块 '{sector_name}' 不在已加载的板块列表中")
                return None, None, None
                
            # 获取板块股票列表
            stocks = self.sectors[sector_name]
            if not stocks or len(stocks) == 0:
                self.logger.warning(f"板块 '{sector_name}' 没有股票")
                return None, None, None
                
            self.logger.info(f"板块 '{sector_name}' 包含 {len(stocks)} 只股票，开始提取特征")
            
            # 提取短期特征和标签
            features_short, labels = self._extract_features(stocks, date, lookback_short, forward_days)
            if features_short is None or labels is None:
                self.logger.warning(f"板块 '{sector_name}' 无法提取短期特征或标签")
                return None, None, None
                
            # 提取长期特征（如果需要）
            features_long = None
            if lookback_long > lookback_short:
                features_long, _ = self._extract_features(stocks, date, lookback_long, forward_days)
                if features_long is None:
                    self.logger.warning(f"板块 '{sector_name}' 无法提取长期特征，将只使用短期特征")
                else:
                    self.logger.info(f"板块 '{sector_name}' 成功提取长期特征，形状: {features_long.shape}")
            
            # 记录特征形状
            self.logger.info(f"板块 '{sector_name}' 特征提取完成，短期特征形状: {features_short.shape}, 标签形状: {labels.shape}")
            return features_short, features_long, labels
            
        except Exception as e:
            self.logger.error(f"提取板块 '{sector_name}' 特征时发生错误: {e}")
            return None, None, None
            
    def _train_model(self, sector_name, features_short, features_long, labels, test_size=0.3):
        """训练模型，包括短期和长期模型的训练与组合
        
        Args:
            sector_name: 板块名称
            features_short: 短期特征
            features_long: 长期特征（可为None）
            labels: 标签
            test_size: 测试集比例
            
        Returns:
            dict: 包含模型和训练信息的字典
        """
        try:
            # 检查特征和标签
            if features_short is None or labels is None:
                self.logger.warning(f"板块 '{sector_name}' 的特征或标签为空，无法训练模型")
                return None
                
            # 记录模型训练信息
            model_info = {
                'sector': sector_name,
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'feature_shape': features_short.shape,
                'positive_ratio': np.mean(labels),
                'training_success': False
            }
            
            # 分割训练集和测试集
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    features_short, labels, test_size=test_size, random_state=42, stratify=labels
                )
                model_info['train_samples'] = len(X_train)
                model_info['test_samples'] = len(X_test)
            except Exception as e:
                self.logger.error(f"分割训练集和测试集时出错: {e}")
                return None
                
            # 训练短期模型
            self.logger.info(f"开始训练板块 '{sector_name}' 的短期模型")
            short_model = self._train_single_model(X_train, y_train)
            if short_model is None:
                self.logger.warning(f"板块 '{sector_name}' 的短期模型训练失败")
                return None
                
            # 评估短期模型
            try:
                y_pred = short_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                model_info['short_metrics'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                self.logger.info(f"短期模型评估: 准确率={accuracy:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}")
            except Exception as e:
                self.logger.error(f"评估短期模型时出错: {e}")
                model_info['short_metrics'] = {'error': str(e)}
            
            # 如果有长期特征，训练长期模型
            long_model = None
            if features_long is not None:
                try:
                    # 分割长期特征的训练集和测试集
                    X_long_train, X_long_test, y_long_train, y_long_test = train_test_split(
                        features_long, labels, test_size=test_size, random_state=42, stratify=labels
                    )
                    
                    # 训练长期模型
                    self.logger.info(f"开始训练板块 '{sector_name}' 的长期模型")
                    long_model = self._train_single_model(X_long_train, y_long_train)
                    
                    if long_model is not None:
                        # 评估长期模型
                        y_long_pred = long_model.predict(X_long_test)
                        long_accuracy = accuracy_score(y_long_test, y_long_pred)
                        long_precision = precision_score(y_long_test, y_long_pred, zero_division=0)
                        long_recall = recall_score(y_long_test, y_long_pred, zero_division=0)
                        long_f1 = f1_score(y_long_test, y_long_pred, zero_division=0)
                        
                        model_info['long_metrics'] = {
                            'accuracy': long_accuracy,
                            'precision': long_precision,
                            'recall': long_recall,
                            'f1_score': long_f1
                        }
                        
                        self.logger.info(f"长期模型评估: 准确率={long_accuracy:.4f}, 精确率={long_precision:.4f}, "
                                      f"召回率={long_recall:.4f}, F1={long_f1:.4f}")
                except Exception as e:
                    self.logger.error(f"训练或评估长期模型时出错: {e}")
                    model_info['long_metrics'] = {'error': str(e)}
                    long_model = None
            
            # 组合模型（如果有长期模型）
            final_model = short_model
            if long_model is not None:
                try:
                    self.logger.info(f"组合板块 '{sector_name}' 的长短期模型")
                    ensemble_model = self.EnsembleModelWrapper(short_model, long_model)
                    
                    # 评估组合模型
                    y_ensemble_pred = ensemble_model.predict(X_test)
                    ensemble_accuracy = accuracy_score(y_test, y_ensemble_pred)
                    ensemble_precision = precision_score(y_test, y_ensemble_pred, zero_division=0)
                    ensemble_recall = recall_score(y_test, y_ensemble_pred, zero_division=0)
                    ensemble_f1 = f1_score(y_test, y_ensemble_pred, zero_division=0)
                    
                    model_info['ensemble_metrics'] = {
                        'accuracy': ensemble_accuracy,
                        'precision': ensemble_precision,
                        'recall': ensemble_recall,
                        'f1_score': ensemble_f1
                    }
                    
                    self.logger.info(f"组合模型评估: 准确率={ensemble_accuracy:.4f}, 精确率={ensemble_precision:.4f}, "
                                  f"召回率={ensemble_recall:.4f}, F1={ensemble_f1:.4f}")
                    
                    # 如果组合模型性能更好，使用组合模型
                    if ensemble_f1 >= f1:
                        final_model = ensemble_model
                        self.logger.info("使用组合模型作为最终模型")
                    else:
                        self.logger.info("组合模型性能不如短期模型，使用短期模型作为最终模型")
                except Exception as e:
                    self.logger.error(f"组合模型时出错: {e}")
                    model_info['ensemble_error'] = str(e)
            
            # 构建最终的模型字典
            model_dict = {
                'info': model_info,
                'model': final_model,
                'training_success': True
            }
            
            model_info['training_success'] = True
            self.logger.info(f"板块 '{sector_name}' 模型训练成功")
            return model_dict
            
        except Exception as e:
            self.logger.error(f"训练板块 '{sector_name}' 模型时发生错误: {e}")
            return None
            
    def _train_single_model(self, X_train, y_train):
        """训练单个模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            object: 训练好的模型，失败时返回None
        """
        try:
            # 使用随机森林分类器
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=1  # 使用单线程，避免多线程导致的序列化问题
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.logger.error(f"训练单个模型时出错: {e}")
            return None

# 在类定义之前，添加顶级函数
def _process_safe_train_sector(sector_name, date, lookback_short, lookback_long, test_size, forward_days):
    """
    进程安全的板块训练函数，作为顶级函数以确保可序列化
    完全独立于类方法，避免传递任何类实例
    """
    try:
        import logging
        import os
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

                # 定义辅助函数
        def calculate_ema(prices, period):
            """计算指数移动平均线"""
            if len(prices) < period:
                return np.array([prices[-1]])
            
            ema = np.zeros_like(prices)
            ema[0] = prices[0]
            
            # 计算平滑系数
            multiplier = 2 / (period + 1)
            
            # 计算EMA
            for i in range(1, len(prices)):
                ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            
            return ema
        
        def calculate_tr(high, low, close):
            """计算真实波动范围(True Range)"""
            if len(high) <= 1:
                return np.array([high[0] - low[0]])
                
            tr = np.zeros(len(high))
            tr[0] = high[0] - low[0]  # 首日TR
            
            # 计算TR: max(high-low, |high-prevClose|, |low-prevClose|)
            for i in range(1, len(high)):
                hl = high[i] - low[i]
                hpc = abs(high[i] - close[i-1])
                lpc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hpc, lpc)
                
            return tr
        
        def is_doji(candle):
            """判断是否是十字星形态"""
            open_price, high, low, close = candle
            body_size = abs(close - open_price)
            range_size = high - low
            
            if range_size == 0:
                return False
                
            # 如果实体很小，上下影线较长，则为十字星
            return body_size / range_size < 0.1               
        
        def is_hammer(candle):
            """判断是否是锤子形态"""
            open_price, high, low, close = candle
            body_size = abs(close - open_price)
            
            if body_size == 0:
                return False
                
            # 获取下影线长度
            if close >= open_price:  # 阳线
                lower_shadow = open_price - low
            else:  # 阴线
                lower_shadow = close - low
                
            # 获取上影线长度
            if close >= open_price:  # 阳线
                upper_shadow = high - close
            else:  # 阴线
                upper_shadow = high - open_price
                
            # 锤子形态：下影线长度是实体长度的2倍以上，且上影线较短
            return (lower_shadow > 2 * body_size) and (upper_shadow < 0.5 * body_size)
        
        def is_engulfing(candles):
            """判断是否是吞没形态"""
            if len(candles) < 2:
                return 0
                
            prev_open, prev_high, prev_low, prev_close = candles[0]
            curr_open, curr_high, curr_low, curr_close = candles[1]
            
            prev_body_size = abs(prev_close - prev_open)
            curr_body_size = abs(curr_close - curr_open)
            
            # 无实体则非吞没形态
            if prev_body_size == 0 or curr_body_size == 0:
                return 0
                
            # 判断是否是看涨吞没：前一天阴线，当天阳线，且当天实体完全覆盖前一天
            bullish_engulfing = (prev_close < prev_open) and (curr_close > curr_open) and \
                                (curr_open <= prev_close) and (curr_close >= prev_open)
                                
            # 判断是否是看跌吞没：前一天阳线，当天阴线，且当天实体完全覆盖前一天
            bearish_engulfing = (prev_close > prev_open) and (curr_close < curr_open) and \
                                (curr_open >= prev_close) and (curr_close <= prev_open)
                                
            if bullish_engulfing:
                return 1  # 看涨吞没
            elif bearish_engulfing:
                return -1  # 看跌吞没
            else:
                return 0  # 非吞没形态
        
        
        # 导入数据缓存管理器
        try:
            from data_cache_manager import get_data_cache_manager
            cache_manager = get_data_cache_manager()
        except Exception as e:
            cache_manager = None
            
        # 创建新的logger进行日志记录
        proc_logger = logging.getLogger(f"process_{os.getpid()}")
        if not proc_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            proc_logger.addHandler(handler)
            proc_logger.setLevel(logging.INFO)
        
        proc_logger.info(f"开始训练板块 '{sector_name}'")
        
        # 在这里实现特征提取和模型训练逻辑
        # 注意：不依赖于原类的方法，避免序列化问题
        
            
        # 获取板块内股票
        try:
            stocks = get_sector_stocks(sector_name)
            if not stocks or len(stocks) < 5:  # 至少需要5只股票
                proc_logger.warning(f"板块 '{sector_name}' 包含的股票数量不足: {len(stocks) if stocks else 0}")
                return sector_name, None
        except Exception as e:
            proc_logger.error(f"获取板块 '{sector_name}' 的股票列表失败: {e}")
            return sector_name, None
            
        # 提取特征(简化版本，仅用于演示)
        try:
            # 提取短期特征
            fields = ['close', 'open', 'high', 'low', 'volume']
            end_date = date
            
            # 计算起始日期 - 为了简化，我们使用固定的日期偏移
            from datetime import datetime, timedelta
            end_date_obj = datetime.strptime(end_date, '%Y%m%d')
            
            # 短期起始日期
            short_start_date_obj = end_date_obj - timedelta(days=lookback_short*2)  # 多取一些数据以计算指标
            short_start_date = short_start_date_obj.strftime('%Y%m%d')
            
            # 长期起始日期
            long_start_date_obj = end_date_obj - timedelta(days=lookback_long*2)  # 多取一些数据以计算指标
            long_start_date = long_start_date_obj.strftime('%Y%m%d')
            
            # 获取市场数据
            try:
                # 确保API连接
                if not xtdata.get_client().is_connected():
                    proc_logger.info("尝试连接xtdata API...")
                    xtdata.get_client().connect()
                
                # 分批获取数据，防止大量股票一次请求导致API问题
                batch_size = 500  # 每批最多处理50只股票
                num_batches = (len(stocks) + batch_size - 1) // batch_size  # 计算需要的批次数
                
                proc_logger.info(f"将分{num_batches}批获取数据，每批最多{batch_size}只股票")
                
                # 初始化空数据结构
                short_data = {field: pd.DataFrame() for field in fields}
                long_data = {field: pd.DataFrame() for field in fields}
                
                # 分批获取数据
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(stocks))
                    batch_stocks = stocks[start_idx:end_idx]
                    
                    proc_logger.info(f"获取第{batch_idx+1}/{num_batches}批数据，包含{len(batch_stocks)}只股票")
                    
                    # 获取短期数据
                    proc_logger.info(f"开始获取第{batch_idx+1}批短期数据...")
                    batch_short_data = xtdata.get_market_data(
                        stock_list=batch_stocks,
                        field_list=fields,
                        period='1d',
                        start_time=short_start_date,
                        end_time=end_date
                    )
                    
                    if batch_short_data:
                        # 详细记录批次数据结构
                        proc_logger.info(f"第{batch_idx+1}批短期数据包含字段: {list(batch_short_data.keys())}")
                        
                        # 合并数据
                        for field in fields:
                            if field in batch_short_data and isinstance(batch_short_data[field], pd.DataFrame) and not batch_short_data[field].empty:
                                if short_data[field].empty:
                                    short_data[field] = batch_short_data[field]
                                else:
                                    # 合并DataFrame
                                    short_data[field] = pd.concat([short_data[field], batch_short_data[field]])
                    else:
                        proc_logger.warning(f"第{batch_idx+1}批短期数据获取失败或为空")
                    
                    # 获取长期数据
                    proc_logger.info(f"开始获取第{batch_idx+1}批长期数据...")
                    batch_long_data = xtdata.get_market_data(
                        stock_list=batch_stocks,
                        field_list=fields,
                        period='1d',
                        start_time=long_start_date,
                        end_time=end_date
                    )
                    
                    if batch_long_data:
                        # 详细记录批次数据结构
                        proc_logger.info(f"第{batch_idx+1}批长期数据包含字段: {list(batch_long_data.keys())}")
                        
                        # 合并数据
                        for field in fields:
                            if field in batch_long_data and isinstance(batch_long_data[field], pd.DataFrame) and not batch_long_data[field].empty:
                                if long_data[field].empty:
                                    long_data[field] = batch_long_data[field]
                                else:
                                    # 合并DataFrame
                                    long_data[field] = pd.concat([long_data[field], batch_long_data[field]])
                    else:
                        proc_logger.warning(f"第{batch_idx+1}批长期数据获取失败或为空")
                
                # 检查合并后的数据
                proc_logger.info("检查合并后的数据...")
                for field in fields:
                    if field in short_data:
                        proc_logger.info(f"短期数据字段 '{field}': {'非空' if not short_data[field].empty else '为空'}")
                    if field in long_data:
                        proc_logger.info(f"长期数据字段 '{field}': {'非空' if not long_data[field].empty else '为空'}")
                
                # 详细记录短期数据结构
                if short_data:
                    proc_logger.info(f"合并后的短期数据结构:")
                    for field in fields:
                        if field in short_data:
                            data = short_data[field]
                            proc_logger.info(f"字段 '{field}':")
                            proc_logger.info(f"  - 类型: {type(data)}")
                            proc_logger.info(f"  - 形状: {data.shape if hasattr(data, 'shape') else '无形状'}")
                            if not data.empty and hasattr(data, 'index'):
                                proc_logger.info(f"  - 索引: {list(data.index)[:5]}... (共{len(data.index)}个)")
                            if not data.empty and hasattr(data, 'columns'):
                                proc_logger.info(f"  - 列: {list(data.columns)[:5]}... (共{len(data.columns)}个)")
                            proc_logger.info(f"  - 是否为空: {data.empty if hasattr(data, 'empty') else '未知'}")
                        else:
                            proc_logger.warning(f"字段 '{field}' 不存在于短期数据中")
                else:
                    proc_logger.error("合并后的短期数据为空")
                    
                # 验证获取的数据不为None
                if short_data is None:
                    proc_logger.error("获取短期市场数据失败：返回None")
                    return sector_name, None
                
                if long_data is None:
                    proc_logger.error("获取长期市场数据失败：返回None")
                    return sector_name, None
                
                # 检查必需的字段是否存在
                required_fields = ['close', 'open', 'high', 'low']
                missing_short_fields = [f for f in required_fields if f not in short_data]
                if missing_short_fields:
                    proc_logger.error(f"短期数据缺少必需字段: {missing_short_fields}")
                    return sector_name, None
                    
                missing_long_fields = [f for f in required_fields if f not in long_data]
                if missing_long_fields:
                    proc_logger.error(f"长期数据缺少必需字段: {missing_long_fields}")
                    return sector_name, None
                    
            except Exception as e:
                proc_logger.error(f"获取市场数据失败: {e}")
                import traceback
                proc_logger.error(f"详细错误信息: {traceback.format_exc()}")
                return sector_name, None
                
            # 专业量化特征工程与高级标签设计
            features_short = {}
            features_long = {}
            labels = {}
            
            # 对每只股票进行处理
            for stock in stocks:
                # 提取数据并进行特征工程
                try:
                    # 检查股票在数据中是否存在
                    if 'close' not in short_data or stock not in short_data['close']:
                        proc_logger.warning(f"股票 {stock} 在短期数据中不存在11111: {short_data['close']} {short_data}")
                        continue
                    
                    if 'close' not in long_data or stock not in long_data['close']:
                        proc_logger.warning(f"股票 {stock} 在长期数据中不存在22222")
                        continue

                    print(f"股票 {stock} 在长期数据中存在333333: {long_data['close'][stock]}")
                    
                    # 获取基础价格和交易量数据
                    stock_close_short = np.array(short_data['close'][stock])
                    stock_open_short = np.array(short_data['open'][stock])
                    stock_high_short = np.array(short_data['high'][stock])
                    stock_low_short = np.array(short_data['low'][stock])
                    stock_volume_short = np.array(short_data['volume'][stock])
                    
                    stock_close_long = np.array(long_data['close'][stock])
                    stock_open_long = np.array(long_data['open'][stock])
                    stock_high_long = np.array(long_data['high'][stock])
                    stock_low_long = np.array(long_data['low'][stock])
                    stock_volume_long = np.array(long_data['volume'][stock])
                    
                    # 检查数据是否为None或空
                    if stock_close_short is None or len(stock_close_short) == 0:
                        proc_logger.warning(f"股票 {stock} 短期收盘价数据为空")
                        continue
                    
                    if stock_close_long is None or len(stock_close_long) == 0:
                        proc_logger.warning(f"股票 {stock} 长期收盘价数据为空")
                        continue
                    
                    # 短期特征计算 - 使用专业指标
                    if len(stock_close_short) >= lookback_short:
                        # 1. 趋势指标
                        # 计算移动平均线(SMA)
                        sma5 = np.mean(stock_close_short[-5:]) if len(stock_close_short) >= 5 else stock_close_short[-1]
                        sma10 = np.mean(stock_close_short[-10:]) if len(stock_close_short) >= 10 else stock_close_short[-1]
                        sma20 = np.mean(stock_close_short[-20:]) if len(stock_close_short) >= 20 else stock_close_short[-1]
                        
                        # 计算指数移动平均线(EMA)
                        ema12 = calculate_ema(stock_close_short, 12)[-1]
                        ema26 = calculate_ema(stock_close_short, 26)[-1]
                        
                        # MACD指标
                        macd = ema12 - ema26
                        macd_signal = calculate_ema(np.array([ema12 - ema26 for ema12, ema26 in zip(
                            calculate_ema(stock_close_short, 12)[-9:], 
                            calculate_ema(stock_close_short, 26)[-9:])]), 9)[-1]
                        macd_hist = macd - macd_signal
                        
                        # 2. 动量指标
                        # 计算收益率
                        returns = np.diff(stock_close_short) / stock_close_short[:-1]
                        
                        # RSI计算
                        if len(returns) >= 14:
                            up_returns = np.where(returns[-14:] > 0, returns[-14:], 0)
                            down_returns = np.where(returns[-14:] < 0, -returns[-14:], 0)
                            avg_up = np.mean(up_returns)
                            avg_down = np.mean(down_returns)
                            rs = avg_up / avg_down if avg_down != 0 else float('inf')
                            rsi = 100 - (100 / (1 + rs))
                        else:
                            rsi = 50  # 默认中性值
                        
                        # 3. 波动率指标
                        # 计算波动率(年化)
                        volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else np.std(returns) * np.sqrt(252)
                        
                        # ATR计算
                        tr_values = calculate_tr(stock_high_short, stock_low_short, stock_close_short)
                        atr = np.mean(tr_values[-14:]) if len(tr_values) >= 14 else np.mean(tr_values)
                        
                        # 4. 布林带指标
                        bb_std = np.std(stock_close_short[-20:]) if len(stock_close_short) >= 20 else np.std(stock_close_short)
                        bb_upper = sma20 + 2 * bb_std
                        bb_lower = sma20 - 2 * bb_std
                        bb_width = (bb_upper - bb_lower) / sma20 if sma20 != 0 else 0
                        bb_position = (stock_close_short[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
                        
                        # 5. K线形态识别
                        recent_candles = np.column_stack((
                            stock_open_short[-5:],
                            stock_high_short[-5:],
                            stock_low_short[-5:],
                            stock_close_short[-5:]
                        ))
                        doji = 1 if is_doji(recent_candles[-1]) else 0
                        hammer = 1 if is_hammer(recent_candles[-1]) else 0
                        engulfing = is_engulfing(recent_candles[-2:]) if len(recent_candles) >= 2 else 0
                        
                        # 6. 成交量指标
                        volume_sma5 = np.mean(stock_volume_short[-5:]) if len(stock_volume_short) >= 5 else stock_volume_short[-1]
                        volume_ratio = stock_volume_short[-1] / volume_sma5 if volume_sma5 != 0 else 1.0
                        
                        # 存储短期特征
                        features_short[stock] = {
                            # 趋势指标
                            'sma5': sma5,
                            'sma10': sma10,
                            'sma20': sma20,
                            'sma5_10_ratio': sma5/sma10 if sma10 != 0 else 1.0,
                            'sma5_20_ratio': sma5/sma20 if sma20 != 0 else 1.0,
                            'price_to_sma20': stock_close_short[-1]/sma20 if sma20 != 0 else 1.0,
                            'macd': macd,
                            'macd_signal': macd_signal,
                            'macd_hist': macd_hist,
                            
                            # 动量指标
                            'rsi': rsi,
                            'momentum_5d': np.sum(returns[-5:]) if len(returns) >= 5 else np.nan,
                            'momentum_10d': np.sum(returns[-10:]) if len(returns) >= 10 else np.nan,
                            
                            # 波动率指标
                            'volatility': volatility,
                            'atr': atr,
                            'atr_ratio': atr / stock_close_short[-1] if stock_close_short[-1] != 0 else 0,
                            
                            # 布林带指标
                            'bb_width': bb_width,
                            'bb_position': bb_position,
                            'bb_breakout_up': 1 if stock_close_short[-1] > bb_upper else 0,
                            'bb_breakout_down': 1 if stock_close_short[-1] < bb_lower else 0,
                            
                            # K线形态
                            'doji': doji,
                            'hammer': hammer,
                            'engulfing': engulfing,
                            
                            # 成交量指标
                            'volume_ratio': volume_ratio,
                            'volume_trend': (stock_volume_short[-1] - volume_sma5) / volume_sma5 if volume_sma5 != 0 else 0,
                            
                            # 价格波动
                            'close_to_high': (stock_high_short[-1] - stock_close_short[-1]) / stock_high_short[-1] if stock_high_short[-1] != 0 else 0,
                            'close_to_low': (stock_close_short[-1] - stock_low_short[-1]) / stock_close_short[-1] if stock_close_short[-1] != 0 else 0,
                            'high_low_range': (stock_high_short[-1] - stock_low_short[-1]) / stock_close_short[-1] if stock_close_short[-1] != 0 else 0,
                        }
                    else:
                        proc_logger.warning(f"股票 {stock} 短期数据不足: {len(stock_close_short)}/{lookback_short}")
                        continue
                    
                    # 长期特征计算 - 使用专业指标
                    if len(stock_close_long) >= lookback_long:
                        # 1. 长期趋势指标
                        sma30 = np.mean(stock_close_long[-30:]) if len(stock_close_long) >= 30 else stock_close_long[-1]
                        sma60 = np.mean(stock_close_long[-60:]) if len(stock_close_long) >= 60 else stock_close_long[-1]
                        
                        # 2. 长期回报和波动率
                        long_returns = np.diff(stock_close_long) / stock_close_long[:-1]
                        cum_return = np.prod(1 + long_returns[-60:]) - 1 if len(long_returns) >= 60 else np.prod(1 + long_returns) - 1
                        long_volatility = np.std(long_returns[-60:]) * np.sqrt(252) if len(long_returns) >= 60 else np.std(long_returns) * np.sqrt(252)
                        
                        # 3. 长期夏普比率
                        risk_free_rate = 0.02 / 252  # 年化2%
                        excess_returns = long_returns[-60:] - risk_free_rate if len(long_returns) >= 60 else long_returns - risk_free_rate
                        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
                        
                        # 4. 长期成交量趋势
                        volume_sma30 = np.mean(stock_volume_long[-30:]) if len(stock_volume_long) >= 30 else np.mean(stock_volume_long)
                        volume_sma60 = np.mean(stock_volume_long[-60:]) if len(stock_volume_long) >= 60 else volume_sma30
                        volume_trend = volume_sma30 / volume_sma60 if volume_sma60 != 0 else 1.0
                        
                        # 5. 历史极值分析
                        highest_high = np.max(stock_high_long[-60:]) if len(stock_high_long) >= 60 else np.max(stock_high_long)
                        lowest_low = np.min(stock_low_long[-60:]) if len(stock_low_long) >= 60 else np.min(stock_low_long)
                        price_to_high = stock_close_long[-1] / highest_high if highest_high != 0 else 1.0
                        price_to_low = stock_close_long[-1] / lowest_low if lowest_low != 0 else 1.0
                        
                        # 存储长期特征
                        features_long[stock] = {
                            # 长期趋势指标
                            'sma30': sma30,
                            'sma60': sma60,
                            'sma30_60_ratio': sma30/sma60 if sma60 != 0 else 1.0,
                            'price_to_sma60': stock_close_long[-1]/sma60 if sma60 != 0 else 1.0,
                            
                            # 长期回报和波动率
                            'cum_return': cum_return,
                            'long_volatility': long_volatility,
                            
                            # 长期夏普比率
                            'sharpe_ratio': sharpe_ratio,
                            
                            # 长期成交量趋势
                            'volume_trend': volume_trend,
                            
                            # 历史极值分析
                            'price_to_high': price_to_high,
                            'price_to_low': price_to_low,
                            'high_low_range': (highest_high - lowest_low) / sma60 if sma60 != 0 else 0,
                            
                            # 趋势强度
                            'trend_strength': np.abs(sma30 - sma60) / sma60 if sma60 != 0 else 0,
                            'trend_direction': 1 if sma30 > sma60 else -1 if sma30 < sma60 else 0,
                        }
                    else:
                        proc_logger.warning(f"股票 {stock} 长期数据不足: {len(stock_close_long)}/{lookback_long}")
                        continue
                    
                    # 高级标签设计 - 使用多因子评分
                    if len(stock_close_short) >= 5 + forward_days:
                        # 计算未来收益
                        future_prices = stock_close_short[-forward_days-1:]
                        future_returns = np.diff(future_prices) / future_prices[:-1]
                        
                        # 总收益率
                        total_return = (future_prices[-1] / future_prices[0]) - 1
                        
                        # 计算未来波动率
                        future_volatility = np.std(future_returns) * np.sqrt(252) if len(future_returns) > 0 else 0
                        
                        # 计算夏普比率
                        daily_risk_free = 0.02 / 252  # 假设无风险利率为2%
                        excess_return = total_return - (daily_risk_free * forward_days)
                        sharpe_ratio = excess_return / (future_volatility * np.sqrt(forward_days/252)) if future_volatility != 0 else 0
                        
                        # 计算最大回撤
                        cumulative_returns = np.cumprod(1 + future_returns)
                        peak = np.maximum.accumulate(cumulative_returns)
                        drawdown = (peak - cumulative_returns) / peak
                        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                        
                        # 计算胜率
                        win_rate = np.sum(future_returns > 0) / len(future_returns) if len(future_returns) > 0 else 0.5
                        
                        # 多目标评分系统 - 结合收益、风险、夏普比率和胜率
                        return_score = min(max(total_return * 10, -1), 1)  # 收益评分，范围[-1,1]
                        sharpe_score = min(max(sharpe_ratio / 2, -1), 1)   # 夏普比率评分，范围[-1,1]
                        drawdown_score = 1 - min(max(max_drawdown * 5, 0), 1)  # 回撤评分，越小越好
                        win_rate_score = min(max((win_rate - 0.5) * 2, -1), 1)  # 胜率评分，相对于50%的偏离
                        
                        # 综合评分 - 加权汇总
                        total_score = (0.4 * return_score + 
                                      0.3 * sharpe_score + 
                                      0.2 * drawdown_score + 
                                      0.1 * win_rate_score)
                        
                        # 多分类标签生成 - 五分类系统
                        if total_score > 0.6:
                            label = 2      # 强烈买入
                        elif total_score > 0.2:
                            label = 1      # 买入
                        elif total_score > -0.2:
                            label = 0      # 持有
                        elif total_score > -0.6:
                            label = -1     # 卖出
                        else:
                            label = -2     # 强烈卖出
                        
                        # 存储标签
                        labels[stock] = label
                        
                        # 记录预测相关信息
                        pred_info = {
                            'total_return': total_return,
                            'sharpe_ratio': sharpe_ratio,
                            'max_drawdown': max_drawdown,
                            'win_rate': win_rate,
                            'total_score': total_score,
                            'label': label
                        }
                        proc_logger.debug(f"股票 {stock} 预测信息: {pred_info}")
                    else:
                        proc_logger.warning(f"股票 {stock} 数据不足以计算未来收益: {len(stock_close_short)}/{5 + forward_days}")
                        continue
                except Exception as e:
                    proc_logger.warning(f"处理股票 {stock} 数据时出错: {e}")
                    import traceback
                    proc_logger.warning(f"详细错误: {traceback.format_exc()}")
                    continue
            
            # 转换为DataFrame
            features_short_df = pd.DataFrame.from_dict(features_short, orient='index')
            features_long_df = pd.DataFrame.from_dict(features_long, orient='index')
            labels_series = pd.Series(labels)
            
            # 合并数据
            merged_data = features_short_df.join(features_long_df, how='inner')
            merged_data = merged_data.join(labels_series.rename('label'), how='inner')
            
            # 检查是否有足够的数据
            if len(merged_data) < 5:  # 至少需要5个样本
                proc_logger.warning(f"板块 '{sector_name}' 合并后的有效样本数不足: {len(merged_data)}")
                return sector_name, None
                
            # 分离特征和标签
            X = merged_data.drop('label', axis=1)
            y = merged_data['label']
            
            # 训练测试集分割
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # 训练模型
            # 1. 短期模型
            model_short = None
            try:
                # XGBoost模型
                model_short = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    random_state=42
                )
                model_short.fit(X_train, y_train)
                # 评估
                short_accuracy = model_short.score(X_test, y_test)
                proc_logger.info(f"短期模型准确率: {short_accuracy:.4f}")
            except Exception as e:
                proc_logger.error(f"训练短期模型失败: {e}")
                model_short = None
            
            # 2. 长期模型
            model_long = None
            try:
                # RandomForest模型
                model_long = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
                model_long.fit(X_train, y_train)
                # 评估
                long_accuracy = model_long.score(X_test, y_test)
                proc_logger.info(f"长期模型准确率: {long_accuracy:.4f}")
            except Exception as e:
                proc_logger.error(f"训练长期模型失败: {e}")
                model_long = None
            
            # 3. 组合模型
            if model_short is not None and model_long is not None:
                # 创建一个模型字典
                model_dict = {
                    'sector_name': sector_name,
                    'short_model': {
                        'type': 'XGBClassifier',
                        'accuracy': short_accuracy if 'short_accuracy' in locals() else 0.0,
                        'feature_importance': dict(zip(X.columns, model_short.feature_importances_)) if hasattr(model_short, 'feature_importances_') else {},
                        'model': model_short
                    },
                    'long_model': {
                        'type': 'RandomForestClassifier',
                        'accuracy': long_accuracy if 'long_accuracy' in locals() else 0.0,
                        'feature_importance': dict(zip(X.columns, model_long.feature_importances_)) if hasattr(model_long, 'feature_importances_') else {},
                        'model': model_long
                    },
                    'weights': (0.7, 0.3),  # 短期模型权重更高
                    'feature_names': list(X.columns),
                    'created_date': date,
                    'sample_count': len(merged_data)
                }
                
                # 将模型转换为可序列化形式
                serializable_model = _make_serializable_dict(model_dict)
                
                proc_logger.info(f"板块 '{sector_name}' 训练完成")
                return sector_name, serializable_model
            else:
                proc_logger.warning(f"板块 '{sector_name}' 模型训练失败")
                return sector_name, None
                
        except Exception as e:
            proc_logger.error(f"提取特征时出错: {e}")
            return sector_name, None
        
    except MemoryError:
        if 'proc_logger' in locals():
            proc_logger.error(f"训练板块 '{sector_name}' 时内存不足")
        return sector_name, None
    except Exception as e:
        if 'proc_logger' in locals():
            proc_logger.error(f"训练板块 '{sector_name}' 时发生错误: {e}")
        return sector_name, None

def _make_serializable_dict(model_dict):
    """将模型字典转换为可序列化形式的辅助函数"""
    import json
    import numpy as np
    import pandas as pd
    import pickle
    import base64
    
    serializable_dict = {}
    
    try:
        for key, value in model_dict.items():
            # 处理嵌套字典
            if isinstance(value, dict):
                serializable_dict[key] = _make_serializable_dict(value)
            # 处理模型对象
            elif key == 'model':
                try:
                    # 将模型序列化为字节，然后转为base64字符串
                    model_bytes = pickle.dumps(value)
                    serializable_dict[key] = {
                        'type': type(value).__name__,
                        'model_bytes': base64.b64encode(model_bytes).decode('utf-8')
                    }
                except Exception as e:
                    # 序列化失败时，记录模型类型和错误
                    serializable_dict[key] = {
                        'type': type(value).__name__,
                        'error': str(e)
                    }
            # 处理numpy数组和pandas数据结构
            elif isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                serializable_dict[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
            else:
                # 尝试直接添加，如果不可序列化，则转为字符串
                try:
                    # 测试是否可序列化
                    json.dumps({key: value})
                    serializable_dict[key] = value
                except (TypeError, OverflowError):
                    # 不可序列化，转为字符串
                    serializable_dict[key] = str(value)
    except Exception as e:
        return {'error': str(e)}
        
    return serializable_dict
