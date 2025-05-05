#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块轮动模型训练
用于训练基于板块特征的板块轮动预测模型，结合市场状态特征，优化板块选择策略。
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import json
import traceback
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
import copy
import pickle
import inspect
import bisect
import joblib
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import logging
import datetime
import os
import time
import math
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取项目根目录
def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

# 路径设置
def get_data_path(relative_path=None):
    """获取数据目录路径"""
    project_root = get_project_root()
    data_path = project_root / "to" / "local_data"
    if relative_path:
        return data_path / relative_path
    return data_path

def get_model_path():
    """获取模型目录路径"""
    project_root = get_project_root()
    model_path = project_root / "to" / "models" / "sector_models"
    return model_path

def ensure_dir(path):
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class SectorModelTrainer:
    """板块模型训练器"""
    
    def __init__(self, model_dir=None, use_lnn=True):
        """
        初始化模型训练器
        
        Args:
            model_dir: 模型保存目录
            use_lnn: 是否使用液体神经网络
        """
        self.logger = logger
        
        # 设置模型目录
        self.model_dir = Path(model_dir) if model_dir else get_model_path()
        ensure_dir(self.model_dir)
        
        # 特征目录
        self.sector_features_dir = get_data_path("sector_features")
        self.market_status_dir = get_data_path("market_status/daily")
        
        # 是否使用液体神经网络
        self.use_lnn = use_lnn
        if use_lnn:
            try:
                # 尝试导入液体神经网络
                sys.path.append(str(get_project_root() / "to"))
                from Liquid.nn.liquid.liquid_network import LiquidNetwork, LiquidNetModel
                self.LiquidNetModel = LiquidNetModel
                self.logger.info("成功导入液体神经网络模型")
            except ImportError:
                self.logger.warning("导入液体神经网络失败，将使用传统模型")
                self.use_lnn = False
        
        # 交易日历
        self.trading_days = []
        self._load_trading_calendar()
        
        # 设置默认参数
        self.lookback_short = 20  # 短期回溯天数
        self.lookback_long = 60   # 长期回溯天数
        self.forward_days = 5     # 预测未来几天
        
        # 添加特征缓存目录
        self.features_cache_dir = Path(get_data_path(), "features_cache")
        ensure_dir(self.features_cache_dir)
        
        # 长期特征缓存刷新频率(天)
        self.long_feature_refresh_days = 30
        
        # 短期特征缓存刷新频率(天)
        self.short_feature_refresh_days = 3
        
        self.logger.info(f"板块模型训练器初始化完成，模型目录: {self.model_dir}")

    def _load_trading_calendar(self):
        """
        加载交易日历
        
        Returns:
            list: 交易日历列表，如果加载失败则返回None
        """
        try:
            # 尝试从ParquetDataStorage中获取
            sys.path.append(str(get_project_root() / "to"))
            from advanced_quant_data import ParquetDataStorage
            
            storage = ParquetDataStorage()
            
            # 获取较长时间范围的交易日历，但不需要太早的数据
            # 只需要包含训练日期和回溯期即可，'20220101'足够早
            cal_df = storage.get_trading_calendar('20221001', '20250331')
            
            if cal_df is not None and not cal_df.empty:
                if 'date' in cal_df.columns:
                    self.trading_days = cal_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    self.trading_days = cal_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
                    
                self.logger.info(f"成功加载交易日历，共 {len(self.trading_days)} 个交易日，范围: {self.trading_days[0]} - {self.trading_days[-1]}")
                return self.trading_days
            else:
                self.logger.warning("获取交易日历失败")
                self.trading_days = None
                return None
        except Exception as e:
            self.logger.error(f"加载交易日历出错: {e}")
            self.logger.error(traceback.format_exc())
            self.trading_days = None
            return None
    
    def _load_sector_features(self, sector_code, date):
        """
        加载特定板块的特征数据
        
        Args:
            sector_code: 板块代码
            date: 日期，格式为YYYYMMDD
            
        Returns:
            DataFrame: 特征数据
        """
        try:
            # 构建文件路径 - 按板块代码划分目录
            # 优先加载优化后的主特征文件
            main_features_path = self.sector_features_dir / sector_code / f"{sector_code}_features.parquet"
            print(f"main_features_path: {main_features_path}")
            
            if main_features_path.exists():
                # 如果主特征文件存在，直接读取并筛选日期
                self.logger.warning(f"尝试从主特征文件加载板块 {sector_code} 的日期 {date} 数据")
                df = pd.read_parquet(main_features_path)
                
                # 筛选指定日期
                if 'date' in df.columns:
                    date_df = df[df['date'] == date].copy()
                    if not date_df.empty:
                        return date_df
                    
                if 'date_str' in df.columns:
                    date_df = df[df['date_str'] == date].copy()
                    if not date_df.empty:
                        return date_df
                
                self.logger.debug(f"主特征文件中未找到日期 {date} 的数据")
            else:
                # 尝试加载单日期特征文件
                date_file = self.sector_features_dir / sector_code / f"{sector_code}_{date}.parquet"
                if date_file.exists():
                    self.logger.debug(f"尝试从单日期文件加载板块 {sector_code} 的日期 {date} 数据")
                    df = pd.read_parquet(date_file)
                    return df
                
                # 尝试加载原始合并特征文件
                orig_file = self.sector_features_dir / sector_code / f"{sector_code}.parquet"
                if orig_file.exists():
                    self.logger.debug(f"尝试从原始合并文件加载板块 {sector_code} 的日期 {date} 数据")
                    df = pd.read_parquet(orig_file)
                    
                    # 筛选指定日期
                    if 'date' in df.columns:
                        date_df = df[df['date'] == date].copy()
                        if not date_df.empty:
                            return date_df
                        
                    if 'date_str' in df.columns:
                        date_df = df[df['date_str'] == date].copy()
                        if not date_df.empty:
                            return date_df
                
                self.logger.warning(f"未找到板块 {sector_code} 在日期 {date} 的特征数据")
                return None
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 在日期 {date} 的特征数据时出错: {e}")
            return None
    
    def _load_market_status(self, date):
        """
        加载市场状态数据
        
        Args:
            date: 日期，格式为YYYYMMDD
            
        Returns:
            dict: 市场状态数据
        """
        try:
            # 加载市场状态数据
            status_file = self.market_status_dir / "market_status.json"
            if not status_file.exists():
                self.logger.warning(f"市场状态文件不存在: {status_file}")
                return None
                
            with open(status_file, "r", encoding="utf-8") as f:
                market_status_data = json.load(f)
                
            # 获取特定日期的市场状态
            if date in market_status_data:
                return market_status_data[date]
            else:
                self.logger.warning(f"日期 {date} 的市场状态数据不存在")
                return None
                
        except Exception as e:
            self.logger.error(f"加载日期 {date} 的市场状态数据时出错: {e}")
            self.logger.exception(e)
            return None

    def _build_features_for_date_range(self, sector_code, end_date, lookback_days=60, forward_days=5):
        """构建指定日期范围内的特征数据
        
        Args:
            sector_code: 板块代码
            end_date: 结束日期
            lookback_days: 回溯天数，默认60天
            forward_days: 预测未来天数，默认5天
            
        Returns:
            特征DataFrame，包含标签
        """
        # 确定日期范围
        end_idx = self.trading_days.index(end_date) if end_date in self.trading_days else -1
        if end_idx == -1:
            self.logger.warning(f"结束日期 {end_date} 不在交易日历中")
            return None
            
        # 确保有足够的数据用于特征和标签
        if end_idx < lookback_days + forward_days:
            self.logger.warning(f"日期 {end_date} 前没有足够的数据用于特征回溯和前向预测")
            return None
            
        # 计算所需的日期
        feature_end_idx = end_idx
        feature_start_idx = max(0, end_idx - lookback_days + 1)
        
        # 获取特征日期列表
        feature_dates = self.trading_days[feature_start_idx:feature_end_idx+1]
        
        # 计算标签日期
        if end_idx + forward_days < len(self.trading_days):
            label_date = self.trading_days[end_idx + forward_days]
        else:
            self.logger.warning(f"无法获取未来 {forward_days} 天的数据作为标签")
            return None
            
        # 加载特征数据
        feature_dfs = []
        for date in feature_dates:
            sector_df = self._load_sector_features(sector_code, date)
            if sector_df is not None:
                # 添加日期标记
                sector_df['feature_date'] = date
                # 添加市场状态特征
                market_status = self._load_market_status(date)
                if market_status:
                    # 提取市场整体特征
                    sector_df['market_type'] = market_status.get('market_type', '32')  # 默认为中性盘整
                    sector_df['market_cycle'] = market_status.get('market_cycle', '2')  # 默认为上升期
                    sector_df['trend_strength'] = market_status.get('trend_strength', 0.0)
                    
                    # 处理个股指标 - 使用创业板指数作为市场代表
                    if 'individual_status' in market_status and '399006.SZ' in market_status['individual_status']:
                        cyb_data = market_status['individual_status']['399006.SZ']
                        
                        # 市场评分指标
                        sector_df['trend_score'] = cyb_data.get('trend_score', 0.0)
                        sector_df['position_score'] = cyb_data.get('position_score', 0.0)
                        sector_df['volume_score'] = cyb_data.get('volume_score', 0.0)
                        sector_df['momentum_score'] = cyb_data.get('momentum_score', 0.0)
                        sector_df['volatility_score'] = cyb_data.get('volatility_score', 0.0)
                        sector_df['structure_score'] = cyb_data.get('structure_score', 0.0)
                        sector_df['cycle_score'] = cyb_data.get('cycle_score', 0.0)
                        sector_df['total_score'] = cyb_data.get('total_score', 0.0)
                        
                        # 市场类型和周期编码
                        if 'market_status_info' in cyb_data:
                            ms_info = cyb_data['market_status_info']
                            sector_df['market_type_code'] = ms_info.get('type_code', 32)
                            sector_df['market_cycle_code'] = ms_info.get('cycle_code', 2)
                        
                        # 技术指标派生特征
                        if 'technical_data' in cyb_data:
                            tech_data = cyb_data['technical_data']
                            
                            # 价格与均线关系
                            close = tech_data.get('close', 0.0)
                            ma5 = tech_data.get('ma5', 0.0)
                            ma20 = tech_data.get('ma20', 0.0)
                            ma60 = tech_data.get('ma60', 0.0)
                            
                            if ma5 > 0:
                                sector_df['price_to_ma5'] = close / ma5
                            if ma20 > 0:
                                sector_df['price_to_ma20'] = close / ma20
                            if ma60 > 0:
                                sector_df['price_to_ma60'] = close / ma60
                            if ma5 > 0 and ma20 > 0:
                                sector_df['ma5_to_ma20'] = ma5 / ma20
                            
                            # 动量指标
                            sector_df['macd_hist'] = tech_data.get('macd_hist', 0.0)
                            
                            rsi6 = tech_data.get('rsi6', 50.0)
                            sector_df['rsi6_norm'] = rsi6 / 50.0 - 1.0  # 归一化到[-1,1]区间
                            
                            bb_pos = tech_data.get('bb_position', 0.5)
                            sector_df['bb_position_norm'] = bb_pos * 2 - 1  # 归一化到[-1,1]区间
                            
                            kdj_k = tech_data.get('kdj_k', 50.0)
                            kdj_d = tech_data.get('kdj_d', 50.0)
                            sector_df['kdj_diff'] = kdj_k - kdj_d
                            
                        # 价格支撑压力特征
                        support = cyb_data.get('support', 0.0)
                        resistance = cyb_data.get('resistance', 0.0)
                        change_percent = cyb_data.get('change_percent', 0.0)
                        profit_loss_ratio_14 = cyb_data.get('profit_loss_ratio_14', 0.0)
                        profit_loss_ratio_30 = cyb_data.get('profit_loss_ratio_30', 0.0)

                        sector_df['market_profit_loss_ratio_14'] = profit_loss_ratio_14
                        sector_df['market_profit_loss_ratio_30'] = profit_loss_ratio_30
                        
                        # 市场涨跌幅(创业板)
                        sector_df['market_change_percent'] = change_percent
                        
                        # 计算板块相对于创业板的相对涨幅
                        if 'change_percent' in sector_df.columns:
                            # 如果板块数据已经有涨跌幅，计算相对涨幅(差值)
                            sector_df['relative_change_diff'] = sector_df['change_percent'] - change_percent
                            
                                
                            # 添加相对强度指标 - 计算相对于创业板的累积相对强度
                            # 使用指数平滑方式计算，避免单日异常值的影响
                            alpha = 0.3  # 平滑系数
                            if 'relative_strength_index' in sector_df.columns:
                                # 如果已有相对强度指标，进行更新
                                current_rsi = sector_df['relative_strength_index'].iloc[0]
                                # 使用EMA更新相对强度
                                sector_df['relative_strength_index'] = alpha * sector_df['relative_change_diff'] + (1 - alpha) * current_rsi
                            else:
                                # 首次计算，直接使用相对涨幅差
                                sector_df['relative_strength_index'] = sector_df['relative_change_diff']
                                
                            # 计算相对于大盘的Beta和Alpha指标
                            # Beta表示板块波动与大盘相关性，Alpha表示超额收益
                            if 'volatility' in sector_df.columns and 'market_volatility' in sector_df.columns:
                                market_vol = sector_df['market_volatility'].iloc[0]
                                sector_vol = sector_df['volatility'].iloc[0]
                                if market_vol > 0:
                                    # 计算Beta系数(近似) - 波动率比值
                                    sector_df['beta_estimate'] = sector_vol / market_vol
                                    
                                    # 计算风险调整后的相对收益 (Alpha近似)
                                    sector_df['alpha_estimate'] = sector_df['change_percent'] - sector_df['beta_estimate'] * change_percent
                                    
                            # 添加相对强度区间划分
                            # 创建区间类别：强于大盘、弱于大盘、与大盘同步
                            threshold = 0.5  # 设置相对差异阈值
                            conditions = [
                                sector_df['relative_change_diff'] > threshold,  # 强于大盘
                                sector_df['relative_change_diff'] < -threshold,  # 弱于大盘
                            ]
                            choices = ['stronger', 'weaker']
                            default = 'neutral'  # 与大盘同步
                            sector_df['relative_strength_category'] = np.select(conditions, choices, default=default)
                        
                        if support > 0 and close > 0:
                            sector_df['price_to_support'] = close / support - 1.0
                        if resistance > 0 and close > 0:
                            sector_df['resistance_to_price'] = resistance / close - 1.0
                        
                        # 市场数据特征
                        if 'market_data' in cyb_data:
                            mkt_data = cyb_data['market_data']
                            open_price = mkt_data.get('open', 0.0)
                            high = mkt_data.get('high', 0.0)
                            low = mkt_data.get('low', 0.0)
                            volume = mkt_data.get('volume', 0.0)
                            
                            # 当日波动率
                            if close > 0 and high > low:
                                sector_df['intraday_volatility'] = (high - low) / close
                            
                            # 日内涨跌
                            if open_price > 0:
                                sector_df['intraday_return'] = (close - open_price) / open_price
                            
                            # 成交量指标已通过volume_score体现
                            sector_df['volume'] = volume
                        
                feature_dfs.append(sector_df)
                
        if not feature_dfs:
            self.logger.warning(f"板块 {sector_code} 在日期范围内没有可用的特征数据")
            return None
            
        # 合并特征
        features_df = pd.concat(feature_dfs, ignore_index=True)
        
        # 加载标签数据
        label_df = self._load_sector_features(sector_code, label_date)
        if label_df is None:
            self.logger.warning(f"无法获取标签日期 {label_date} 的数据")
            return None
            
        # 添加标签
        features_df['label_date'] = label_date
        features_df['future_return'] = label_df['sector_pct_chg'].values[0] if 'sector_pct_chg' in label_df.columns else 0.0
        
        # 在现有特征基础上添加时序动态特征
        # 获取历史数据用于计算时序特征
        feature_dates = []
        end_idx = self.trading_days.index(end_date)
        history_lookback = min(60, lookback_days)  # 最大60天历史数据
        start_idx = max(0, end_idx - history_lookback)
        feature_dates = self.trading_days[start_idx:end_idx+1]
        
        if len(feature_dates) >= 5 and 'sector_df' in locals() and sector_df is not None:
            # 收集历史数据
            historical_dfs = []
            for date in feature_dates[-10:]:  # 使用最近10天数据
                df = self._load_sector_features(sector_code, date)
                if df is not None and not df.empty:
                    # 加载该日期的市场状态数据
                    market_status = self._load_market_status(date)
                    if market_status and 'individual_status' in market_status and '399006.SZ' in market_status['individual_status']:
                        cyb_data = market_status['individual_status']['399006.SZ']
                        if 'change_percent' in df.columns:
                            df['date'] = date
                            df['market_change_percent'] = cyb_data.get('change_percent', 0.0)
                            df['relative_change_diff'] = df['change_percent'] - df['market_change_percent']
                            historical_dfs.append(df)
        
            if len(historical_dfs) >= 5:  # 至少需要5天数据
                # 合并历史数据
                historical_df = pd.concat(historical_dfs).sort_values('date')
                
                # 1. 相对强度时序特征
                if 'relative_change_diff' in historical_df.columns:
                    # 计算相对强度变化趋势（斜率）
                    recent_values = historical_df['relative_change_diff'].tail(5).values
                    if len(recent_values) >= 3:
                        slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                        sector_df['rel_strength_trend'] = slope * 5  # 标准化为5日总变化量
                    
                    # 相对强度加速度
                    if len(historical_df) >= 5:
                        recent_change = historical_df['relative_change_diff'].tail(3).mean()
                        previous_change = historical_df['relative_change_diff'].tail(5).head(2).mean()
                        sector_df['rel_strength_acceleration'] = recent_change - previous_change
                    
                    # 相对强度一致性（5天中相对强度为正的天数比例）
                    sector_df['rel_strength_consistency'] = (historical_df['relative_change_diff'].tail(5) > 0).mean()
                
                # 2. 价格动量特征增强
                if 'change_percent' in historical_df.columns:
                    # 短期动量
                    sector_df['momentum_3d'] = historical_df['change_percent'].tail(3).sum()
                    # 与市场短期动量对比
                    if 'market_change_percent' in historical_df.columns:
                        market_momentum_3d = historical_df['market_change_percent'].tail(3).sum()
                        sector_df['relative_momentum_3d'] = sector_df['momentum_3d'] - market_momentum_3d
                
                # 3. 波动性特征增强
                if 'volatility' in historical_df.columns and len(historical_df['volatility']) >= 5:
                    recent_vol = historical_df['volatility'].tail(3).mean()
                    prev_vol = historical_df['volatility'].tail(5).head(2).mean()
                    if prev_vol > 0:
                        sector_df['volatility_change_ratio'] = recent_vol / prev_vol
                
                # 4. 季节性特征
                if end_date:
                    month = pd.to_datetime(end_date).month
                    sector_df['month'] = month
                    # A股市场季节性轮动特征
                    sector_df['quarter_start'] = 1 if month in [1, 4, 7, 10] else 0
                    sector_df['year_end'] = 1 if month in [11, 12] else 0
        
        return sector_df


    def _add_dynamic_features(self, features_df, market_status):
        """基于市场环境动态添加或增强特征
        
        Args:
            features_df: 特征DataFrame
            market_status: 市场状态数据
        """
        if features_df is None or features_df.empty:
            return features_df
        
        if market_status is None:
            return features_df
        
        # 获取市场类型
        market_type = market_status.get('market_type', 0)
        market_cycle = market_status.get('market_cycle', 0)
        
        # 根据市场环境构建组合特征
        # 1. 牛市环境 - 强化动量和相对强度特征
        if market_type in [11, 12, 13, 21, 22, 23]:  # 上涨趋势
            # 动量和相对强度综合
            if 'short_momentum_3d' in features_df.columns and 'short_rel_strength_trend' in features_df.columns:
                features_df['bull_momentum_strength'] = features_df['short_momentum_3d'] * features_df['short_rel_strength_trend']
    
        # 2. 震荡市场 - 强化相对强度和波动性特征
        elif market_type in [31, 32, 33]:  # 盘整趋势
            # 相对强度和波动性综合
            if 'short_rel_strength_consistency' in features_df.columns and 'short_volatility' in features_df.columns:
                vol = features_df['short_volatility']
                consistency = features_df['short_rel_strength_consistency']
                if not (vol.isnull().all() or consistency.isnull().all()):
                    vol_norm = vol / vol.max() if vol.max() > 0 else 0
                    features_df['range_rel_strength_stability'] = consistency * (1 - vol_norm)
    
        # 3. 熊市环境 - 强化波动率比率和趋势效率特征
        elif market_type in [41, 42, 43, 51, 52, 53]:  # 下跌趋势
            # 防守性指标
            if 'short_volatility_change_ratio' in features_df.columns and 'long_volatility' in features_df.columns:
                vol_change = features_df['short_volatility_change_ratio']
                long_vol = features_df['long_volatility']
                if not (vol_change.isnull().all() or long_vol.isnull().all()):
                    vol_change.fillna(1, inplace=True)
                    long_vol.fillna(long_vol.mean(), inplace=True)
                    features_df['bear_defensive_score'] = (1 / vol_change) * (1 / long_vol)
    
        return features_df

    def _calculate_time_decay_weights(self, df):
        """
        计算时间衰减权重
        
        Args:
            df: DataFrame，包含特征数据
            
        Returns:
            Series: 权重
        """
        # 将日期转换为时间戳
        if 'feature_date' in df.columns:
            dates = pd.to_datetime(df['feature_date'])
            latest_date = dates.max()
            
            # 计算与最新日期的天数差
            days_diff = (latest_date - dates).dt.days
            
            # 使用指数衰减计算权重
            decay_rate = 0.05  # 衰减率
            weights = np.exp(-decay_rate * days_diff)
            
            return weights
        else:
            return pd.Series(1.0, index=df.index)

    def _save_models(self, sector_code, models, feature_columns):
        """
        保存模型
        
        Args:
            sector_code: 板块代码
            models: 模型字典
            feature_columns: 特征列名
        """
        # 创建板块模型目录，确保路径正确
        model_dir = self.model_dir
        print(f"创建板块模型目录11: {model_dir}")
        
        # 如果是板块轮动模型（通常通过sector_code是否等于"sector_rotation"来判断）
        if sector_code == "sector_rotation":
            # 确保目录不重复添加sector_rotation
            if model_dir.name != "sector_rotation":
                sector_dir = model_dir / "sector_rotation"
            else:
                sector_dir = model_dir
        else:
            # 普通板块模型
            sector_dir = model_dir / sector_code
        print(f"创建板块模型目录22: {sector_dir}")
        ensure_dir(sector_dir)
        print(f"创建板块模型目录33: {sector_dir}")
        
        # 记录模型保存位置，方便调试
        self.logger.info(f"保存板块 {sector_code} 的模型到目录: {sector_dir}")
        
        # 保存特征列名
        feature_file = sector_dir / "feature_columns.json"
        with open(feature_file, "w") as f:
            json.dump(feature_columns, f)
        
        # 保存每个模型
        for model_name, model in models.items():
            if model_name == 'xgboost':
                model_file = sector_dir / f"{model_name}_model.json"
                model.save_model(str(model_file))
            elif model_name == 'lightgbm':
                model_file = sector_dir / f"{model_name}_model.txt"
                model.booster_.save_model(str(model_file))
            elif model_name == 'lnn':
                # 创建LNN模型目录
                lnn_model_dir = sector_dir / "lnn_models"
                ensure_dir(lnn_model_dir)
                
                # 保存模型到文件而非目录
                model_file = lnn_model_dir / "lnn_model.pt"
                model.save(model_file)
                
                # 保存模型参数到JSON文件
                params_file = lnn_model_dir / "model_params.json"
                with open(params_file, 'w') as f:
                    json.dump({
                        'input_size': model.input_size,
                        'hidden_size': model.hidden_size,
                        'output_size': model.output_size,
                        'num_layers': model.num_layers
                    }, f)
                
                self.logger.info(f"成功保存LNN模型到 {model_file}，参数文件：{params_file}")
                
        self.logger.info(f"成功保存板块 {sector_code} 的模型到 {sector_dir}")

    def _load_multiple_sectors_features(self, sector_list, date):
        """
        加载多个板块的特征数据并合并
        
        Args:
            sector_list: 板块代码列表
            date: 日期，格式为YYYYMMDD
            
        Returns:
            dict: 包含各板块特征数据的字典，键为板块代码
        """
        if not sector_list:
            self.logger.warning("板块列表为空")
            return None
            
        # 将日期统一格式为字符串
        date = str(date)
        
        self.logger.info(f"加载日期 {date} 的 {len(sector_list)} 个板块特征")
        
        # 逐个加载板块特征
        features_dict = {}
        
        for sector_code in sector_list:
            try:
                # 构建文件路径 - 按板块名称划分目录
                sector_dir = self.sector_features_dir / f"{sector_code}"
                
                # 尝试多种格式的文件
                # 1. 优先加载优化后的主特征文件
                main_features_path = sector_dir / f"{sector_code}_features.parquet"
                
                print(f"main_features_path: {main_features_path}")
                
                # 2. 加载单日期特征文件
                single_date_path = sector_dir / f"{sector_code}_{date}.parquet"
                
                # 3. 加载原始合并特征文件
                merged_file_path = sector_dir / f"{sector_code}.parquet"
                # print(f"main_features_path: {main_features_path}")
                
                # 优先从主特征文件加载
                if main_features_path.exists():
                    try:
                        # self.logger.warning(f"尝试从主特征文件加载板块 {sector_code} 的数据 {date}")
                        df = pd.read_parquet(main_features_path)
                        
                        # 筛选指定日期
                        date_df = None
                        
                        # 尝试不同列名的日期列
                        for date_col in ['date', 'date_str', '日期']:
                            if date_col in df.columns:
                                date_df = df[df[date_col] == date].copy()
                                if not date_df.empty:
                                    break
                        
                        if date_df is not None and not date_df.empty:
                            row_dict = date_df.iloc[0].to_dict()
                            # 确保字典的键是字符串而不是数字
                            cleaned_dict = {}
                            for k, v in row_dict.items():
                                # 如果键是数字或者索引，将其转换为字符串类型的名称
                                if isinstance(k, (int, np.integer)):
                                    cleaned_dict[f"feature_{k}"] = v
                                else:
                                    cleaned_dict[k] = v
                            
                            features_dict[sector_code] = cleaned_dict
                            self.logger.debug(f"从主特征文件成功加载板块 {sector_code} 日期 {date} 的数据")
                            continue
                    except Exception as e:
                        self.logger.warning(f"从主特征文件加载板块 {sector_code} 数据时出错: {e}")
                
                # 如果主特征文件不存在或者未找到指定日期，尝试单日期文件
                elif single_date_path.exists():
                    try:
                        self.logger.debug(f"尝试从单日期文件加载板块 {sector_code} 的数据")
                        df = pd.read_parquet(single_date_path)
                        if not df.empty:
                            row_dict = df.iloc[0].to_dict()
                            # 确保字典的键是字符串而不是数字
                            cleaned_dict = {}
                            for k, v in row_dict.items():
                                if isinstance(k, (int, np.integer)):
                                    cleaned_dict[f"feature_{k}"] = v
                                else:
                                    cleaned_dict[k] = v
                                    
                            features_dict[sector_code] = cleaned_dict
                            self.logger.debug(f"从单日期文件成功加载板块 {sector_code} 日期 {date} 的数据")
                            continue
                    except Exception as e:
                        self.logger.warning(f"从单日期文件加载板块 {sector_code} 数据时出错: {e}")
                
                # 尝试加载原始合并特征文件
                elif merged_file_path.exists():
                    try:
                        self.logger.debug(f"尝试从原始合并文件加载板块 {sector_code} 的数据")
                        df = pd.read_parquet(merged_file_path)
                        
                        # 筛选指定日期
                        date_df = None
                        
                        # 尝试不同列名的日期列
                        for date_col in ['date', 'date_str', '日期']:
                            if date_col in df.columns:
                                date_df = df[df[date_col] == date].copy()
                                if not date_df.empty:
                                    break
                        
                        if date_df is not None and not date_df.empty:
                            row_dict = date_df.iloc[0].to_dict()
                            # 确保字典的键是字符串而不是数字
                            cleaned_dict = {}
                            for k, v in row_dict.items():
                                if isinstance(k, (int, np.integer)):
                                    cleaned_dict[f"feature_{k}"] = v
                                else:
                                    cleaned_dict[k] = v
                                    
                            features_dict[sector_code] = cleaned_dict
                            self.logger.debug(f"从原始合并文件成功加载板块 {sector_code} 日期 {date} 的数据")
                            continue
                    except Exception as e:
                        self.logger.warning(f"从原始合并文件加载板块 {sector_code} 数据时出错: {e}")
                
                self.logger.warning(f"未找到板块 {sector_code} 日期 {date} 的特征数据")
                
            except Exception as e:
                self.logger.error(f"加载板块 {sector_code} 日期 {date} 的特征时出错: {e}")
                self.logger.exception(e)
        
        return features_dict

    def _build_rotation_features_dataset(self, sector_list, start_date, end_date, forward_days=2):
        """
        构建板块轮动特征数据集 - 高效优化版
        
        Args:
            sector_list: 板块列表
            start_date: 开始日期
            end_date: 结束日期
            forward_days: 预测未来天数，默认2天
            
        Returns:
            DataFrame: 板块轮动特征数据集
        """
        self.logger.info(f"开始构建板块轮动特征数据集，板块数量: {len(sector_list)}")
        
        # 初始化结果
        all_features = []
        
        # 确保交易日历已加载
        if self.trading_days is None:
            self._load_trading_calendar()
        
        # 如果交易日历加载失败，无法继续
        if self.trading_days is None:
            self.logger.error("交易日历加载失败，无法构建板块轮动特征数据集")
            return None
        
        # 确保日期格式一致
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # 获取日期范围内的所有交易日
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        print(f"date_range: {date_range}")
        date_range = [d.strftime('%Y%m%d') for d in date_range if d.strftime('%Y%m%d') in self.trading_days]
        
        # 计算总日期数量
        total_dates = len(date_range)
        processed_dates = 0
        
        # 预计算所有板块的长期特征缓存
        self.logger.info("预计算长期特征...")
        long_term_features = {}
        seasonal_patterns = {}  # 用于缓存季节性模式
        
        # 预先检查哪些日期需要更新长期特征
        update_dates = []
        for current_date in date_range:
            if self._check_if_long_features_need_update(current_date):
                update_dates.append(current_date)
        
        self.logger.info(f"在{total_dates}个日期中，有{len(update_dates)}个日期需要更新长期特征")
        
        # 对日期进行处理
        for current_date in date_range:
            try:
                # 日志进度
                processed_dates += 1
                if processed_dates % 5 == 0 or processed_dates == total_dates:
                    self.logger.info(f"正在处理日期 {current_date} ({processed_dates}/{total_dates})")
                
                # 确保有足够的未来交易日
                current_idx = self.trading_days.index(current_date)
                if current_idx + forward_days >= len(self.trading_days):
                    self.logger.warning(f"日期 {current_date} 没有足够的未来交易日，跳过")
                    continue
                
                # 获取标签日期
                label_date = self.trading_days[current_idx + forward_days]
                
                # 获取市场状态
                market_status = self._load_market_status(current_date)
                if not market_status:
                    self.logger.warning(f"日期 {current_date} 无法获取市场状态，跳过")
                    continue
                
                # 检查是否需要更新长期特征缓存
                need_update_long = current_date in update_dates
                
                # 保存当日所有板块的特征
                date_features = []
                
                # 处理每个板块
                for sector_code in sector_list:
                    try:
                        # 获取或计算长期特征 - 长期特征只在特定时间更新
                        if need_update_long or sector_code not in long_term_features:
                            long_features = self._build_or_load_features(
                                sector_code, current_date, feature_type='long'
                            )
                            
                            # 只有第一次计算或特定日期才计算季节性模式
                            if sector_code not in seasonal_patterns or need_update_long:
                                # 加载板块的季节性模式
                                sector_pattern = self._compute_seasonal_patterns(sector_code)
                                seasonal_patterns[sector_code] = sector_pattern
                            
                            if long_features is not None:
                                # 添加历史季节性模式特征
                                long_features = self._enrich_with_historical_patterns(
                                    long_features, sector_code, current_date, 
                                    seasonal_data=seasonal_patterns.get(sector_code)
                                )
                                long_term_features[sector_code] = long_features
                        else:
                            long_features = long_term_features[sector_code]
                        
                        # 获取或计算短期特征 - 每天都更新
                        short_features = self._build_or_load_features(
                            sector_code, current_date, feature_type='short'
                        )
                        
                        # 检查特征是否有效
                        if short_features is None or long_features is None:
                            self.logger.warning(f"板块 {sector_code} 日期 {current_date} 无法获取有效特征，跳过")
                            continue
                        
                        # 为区分特征来源，添加前缀
                        short_cols = {}
                        long_cols = {}
                        
                        # 排除不需要加前缀的通用列
                        exclude_prefix = ['feature_date', 'label_date', 'future_return', 'sector_code', 'date']
                        
                        for col in short_features.columns:
                            if col not in exclude_prefix:
                                short_cols[col] = f'short_{col}'
                        
                        for col in long_features.columns:
                            if col not in exclude_prefix:
                                long_cols[col] = f'long_{col}'
                        
                        # 重命名列添加前缀
                        short_features.rename(columns=short_cols, inplace=True)
                        long_features.rename(columns=long_cols, inplace=True)
                        
                        # 确保合并的键存在
                        if 'date' not in short_features.columns:
                            short_features['date'] = current_date
                        
                        if 'date' not in long_features.columns:
                            long_features['date'] = current_date
                        
                        # 合并特征
                        merged_features = pd.merge(
                            short_features,
                            long_features,
                            on='date',
                            how='inner',
                            suffixes=('', '_long')
                        )
                        
                        if merged_features.empty:
                            continue
                        
                        # 添加市场环境特征
                        if market_status:
                            # 动态特征工程 - 基于市场环境
                            self._add_dynamic_features(merged_features, market_status)
                        
                        # 获取未来收益率和成交额变化
                        label_features = self._load_sector_features(sector_code, label_date)
                        
                        if label_features is not None:
                            # 获取收盘价
                            try:
                                # 获取当前收盘价
                                if 'close' in merged_features:
                                    close_value = merged_features.get('close')
                                    if isinstance(close_value, pd.Series):
                                        current_close = float(close_value.iloc[0]) if not close_value.empty else 0
                                    else:
                                        current_close = float(close_value) if close_value is not None else 0
                                elif '收盘价' in merged_features:
                                    close_value = merged_features.get('收盘价')
                                    if isinstance(close_value, pd.Series):
                                        current_close = float(close_value.iloc[0]) if not close_value.empty else 0
                                    else:
                                        current_close = float(close_value) if close_value is not None else 0
                                else:
                                    current_close = 0
                                    self.logger.warning(f"板块 {sector_code} 日期 {current_date} 无法获取当前收盘价")
                                
                                # 同样处理label收盘价
                                if 'close' in label_features:
                                    close_value = label_features.get('close')
                                    if isinstance(close_value, pd.Series):
                                        label_close = float(close_value.iloc[0]) if not close_value.empty else 0
                                    else:
                                        label_close = float(close_value) if close_value is not None else 0
                                elif '收盘价' in label_features:
                                    close_value = label_features.get('收盘价')
                                    if isinstance(close_value, pd.Series):
                                        label_close = float(close_value.iloc[0]) if not close_value.empty else 0
                                    else:
                                        label_close = float(close_value) if close_value is not None else 0
                                else:
                                    label_close = 0
                                    self.logger.warning(f"板块 {sector_code} 日期 {label_date} 无法获取标签收盘价")
            
                            except Exception as e:
                                self.logger.warning(f"处理 {sector_code} 日期 {current_date} 的收盘价时出错: {str(e)}")
                                current_close = 0
                                label_close = 0
                            
                            if current_close > 0 and label_close > 0:
                                # 计算未来收益率
                                future_return = (label_close - current_close) / current_close
                                merged_features['future_return'] = future_return
                                
                                # 添加板块标签
                                merged_features['sector_code'] = sector_code
                                merged_features['label_date'] = label_date
                                
                                # 加入结果列表
                                # all_features.append(merged_features)
                                date_features.append(merged_features)
                    
                    except Exception as e:
                        self.logger.error(f"处理板块 {sector_code} 特征时出错: {e}")
                
                # 每天批量添加所有板块特征，减少内存碎片
                if date_features:
                    # 优化：一次合并同一日期的所有板块特征，减少列表操作开销
                    all_features.extend(date_features)
                    # 清理临时列表释放内存
                    date_features = []
                    
                # 优化：如果特征列表过大，进行中间结果合并并释放内存
                if len(all_features) > 500:  # 每500个样本合并一次
                    self.logger.info(f"合并中间结果并清理内存... 当前样本数: {len(all_features)}")
                    if all_features:
                        # 将列表中的所有DataFrame合并成一个
                        temp_df = pd.concat(all_features, ignore_index=True)
                        # 清空列表释放内存
                        all_features = []
                        # 将合并后的DataFrame重新添加到列表中
                        all_features.append(temp_df)
            
            except Exception as e:
                self.logger.error(f"处理日期 {current_date} 时出错: {e}")
                self.logger.exception(e)
                continue
        
        # 合并所有特征
        if not all_features:
            self.logger.warning("没有生成任何特征数据")
            return None
            
        # 处理all_features可能是DataFrame或DataFrame列表的情况
        if isinstance(all_features[0], pd.DataFrame):
            features_df = pd.concat(all_features, ignore_index=True)
        else:
            # 已经是合并好的DataFrame
            features_df = all_features[0]
            
        # 确保所有列都存在
        if 'future_return' not in features_df.columns:
            self.logger.warning("特征数据中缺少未来收益率列，添加默认值0")
            features_df['future_return'] = 0
            
        # 删除重复行
        features_df.drop_duplicates(subset=['sector_code', 'date'], keep='first', inplace=True)
        
        # 重新排序列，将关键列放在前面
        key_columns = ['sector_code', 'date', 'label_date', 'future_return']
        other_columns = [col for col in features_df.columns if col not in key_columns]
        features_df = features_df[key_columns + other_columns]
        
        # 返回结果
        self.logger.info(f"特征数据构建完成，样本数量: {len(features_df)}")
        return features_df

    def train_rotation_model(self, sector_list=None, start_date=None, end_date=None, features_df=None, model_name="sector_rotation", save_recent_days=0, use_lnn=False):
        """训练板块轮动模型
        
        Args:
            sector_list: 板块列表，如果提供了features_df则可为None
            start_date: 训练开始日期，如果提供了features_df则可为None
            end_date: 训练结束日期，如果提供了features_df则可为None
            features_df: 包含多个板块特征的DataFrame，如提供则直接使用
            model_name: 模型名称，默认为"sector_rotation"
            save_recent_days: 保存最近n个交易日的模型，默认为0（仅保存最终模型）
            use_lnn: 是否使用液体神经网络模型，默认为False
            
        Returns:
            训练好的模型字典
        """
        self.logger.info(f"开始训练板块轮动模型 {model_name}...")
        
        # 确保model_name不包含重复的sector_rotation
        if model_name == "sector_rotation/sector_rotation":
            model_name = "sector_rotation"
        elif model_name.endswith("/sector_rotation"):
            model_name = model_name[:-len("/sector_rotation")]
        
        # 如果未提供features_df，则根据sector_list、start_date和end_date构建
        if features_df is None:
            if sector_list is None or start_date is None or end_date is None:
                self.logger.error("未提供特征数据集，且sector_list、start_date或end_date有缺失，无法训练模型")
                return None
                
            # 批量数据加载 - 根据优化方案进行一次性数据加载处理
            self.logger.info(f"批量加载板块轮动特征数据集...")
            all_features = self._batch_load_features(
                sector_list=sector_list,
                start_date=start_date,
                end_date=end_date
            )
            
            if not all_features:
                self.logger.error("批量加载特征数据失败")
                return None
                
            # 合并特征数据
            features_df = pd.concat(all_features, ignore_index=True)
        
        if features_df is None or len(features_df) == 0:
            self.logger.error("特征数据集为空，无法训练模型")
            return None
            
        # 添加标签列：是否是表现最好的板块
        if 'future_return' in features_df.columns and 'sector_code' in features_df.columns and 'date' in features_df.columns:
            # 按日期分组，找出每天表现最好的板块
            top_performers = features_df.groupby('date').apply(
                lambda x: x.nlargest(5, 'future_return')['sector_code'].tolist()
            ).to_dict()
            
            # 标记是否是表现最好的板块
            features_df['is_top_performer'] = features_df.apply(
                lambda row: 1 if row['sector_code'] in top_performers.get(row['date'], []) else 0, 
                axis=1
            )
        else:
            self.logger.warning("缺少必要的列，无法创建'is_top_performer'标记")
            features_df['is_top_performer'] = 0
        
        # 拆分训练/验证/测试集
        # 按时间顺序拆分，保持时间序列的完整性
        unique_dates = sorted(features_df['date'].unique())
        n_dates = len(unique_dates)
        
        # 70%训练，10%验证，20%测试
        train_dates = unique_dates[:int(n_dates * 0.7)]
        val_dates = unique_dates[int(n_dates * 0.7):int(n_dates * 0.8)]
        test_dates = unique_dates[int(n_dates * 0.8):]
        
        train_df = features_df[features_df['date'].isin(train_dates)]
        val_df = features_df[features_df['date'].isin(val_dates)]
        test_df = features_df[features_df['date'].isin(test_dates)]
        
        self.logger.info(f"数据集拆分: 训练集 {len(train_df)} 样本, 验证集 {len(val_df)} 样本, 测试集 {len(test_df)} 样本")
        
        # 多时间尺度融合 - 构建短期、中期、长期特征
        # 筛选短期特征（1-5日）
        short_term_cols = [col for col in features_df.columns if 'short_' in col or '_5d' in col or '_3d' in col]
        # 筛选中期特征（5-20日）
        mid_term_cols = [col for col in features_df.columns if '_10d' in col or '_15d' in col or '_20d' in col]
        # 筛选长期特征（20-60日）
        long_term_cols = [col for col in features_df.columns if 'long_' in col or '_30d' in col or '_60d' in col]
        
        # 共享特征（基础信息、市场环境特征等）
        base_cols = ['date', 'sector_code', 'market_type', 'market_cycle']
        target_cols = ['future_return', 'is_top_performer']
        
        # 分层模型架构实现
        # 1. 市场环境分类模型（第一层）
        market_env_features = [col for col in features_df.columns if 'market_' in col or 'trend_' in col]
        X_market = train_df[market_env_features].fillna(0)
        y_market = train_df['market_type'].astype('category').cat.codes
        
        # 训练市场环境模型
        self.logger.info("训练第一层模型：市场环境分类器...")
        market_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
                max_depth=5,
            num_leaves=32,
            random_state=42
        )
        market_model.fit(X_market, y_market)
        
        # 2. 板块相对强度预测模型（第二层）
        # 结合市场环境预测和板块特征
        self.logger.info("训练第二层模型：板块相对强度预测模型...")
        
        # 获取市场环境预测
        market_pred = market_model.predict_proba(X_market)
        
        # 为每个市场环境类型添加预测概率作为特征
        for i in range(market_pred.shape[1]):
            train_df[f'market_prob_{i}'] = market_pred[:, i]
        
        # 构建板块相对强度预测模型特征
        strength_features = [col for col in train_df.columns if 
                              'momentum' in col or 'relative_' in col or 
                              'strength' in col or 'market_prob_' in col]
        
        X_strength = train_df[strength_features].fillna(0)
        y_strength = train_df['future_return']
        
        # 训练板块相对强度模型
        strength_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
            random_state=42
        )
        strength_model.fit(X_strength, y_strength)
        
        # 3. 深度学习增强层（第三层）- 使用液体神经网络
        # 准备最终模型特征和标签
        # 结合前两层模型的预测结果作为特征
        train_df['predicted_market'] = market_model.predict(X_market)
        train_df['predicted_strength'] = strength_model.predict(X_strength)
        
        # 构建最终特征，包括短中长期特征的融合
        final_features = (
            base_cols + 
            ['predicted_market', 'predicted_strength'] + 
            short_term_cols[:10] +  # 最重要的10个短期特征 
            mid_term_cols[:10] +    # 最重要的10个中期特征
            long_term_cols[:10]     # 最重要的10个长期特征
        )
        
        # 移除重复特征
        final_features = list(dict.fromkeys(final_features))
        
        # 清理特征列表，确保所有列都存在
        final_features = [col for col in final_features if col in train_df.columns]
        
        # 提取最终特征和标签
        X_final = train_df[final_features].fillna(0)
        y_final = train_df['is_top_performer']
        
        # 使用StandardScaler标准化特征
        scaler = StandardScaler()
        X_final_scaled = scaler.fit_transform(X_final.select_dtypes(include=np.number))
        
        # 转换为适合深度学习的格式
        X_tensor = torch.FloatTensor(X_final_scaled)
        y_tensor = torch.FloatTensor(y_final.values)
        
        # 初始化并训练LNN模型（如果启用）
        lnn_model = None
        if use_lnn and torch is not None:
            self.logger.info("训练第三层模型：液体神经网络...")
            try:
                sys.path.append(str(get_project_root() / "to"))
                from Liquid.nn.liquid.liquid_network import LiquidNetModel
                
                # 初始化LNN模型
                input_size = X_final_scaled.shape[1]
                hidden_size = 128  # 较大的隐藏层能力更强
                output_size = 1
                
                lnn_model = LiquidNetModel(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    num_layers=2
                )
                
                # 设置训练参数
                batch_size = 32
                epochs = 50
                
                # 创建数据集和数据加载器
                train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor.unsqueeze(1))
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
            
            # 训练模型
                lnn_model.fit(
                    X=X_final_scaled,
                    y=y_final.values.reshape(-1, 1),
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(
                        scaler.transform(val_df[final_features].select_dtypes(include=np.number).fillna(0)),
                        val_df['is_top_performer'].values.reshape(-1, 1)
                    ),
                    early_stopping=True,
                    patience=10
                )
            except Exception as e:
                self.logger.error(f"训练LNN模型时出错: {e}")
                self.logger.error(traceback.format_exc())
                lnn_model = None
        
        # 包装并保存所有模型
        models = {
            'xgboost': strength_model,
            'lightgbm': market_model,
        }
        
        if lnn_model is not None:
            models['lnn'] = lnn_model
        
        # 保存模型
        model_dir = os.path.join(self.model_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存特征列名
        with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
            json.dump(final_features, f)
        
        # 保存StandardScaler
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        
        # 保存模型融合权重
        weights = [0.5, 0.5, 0.0]  # 默认权重
        if lnn_model is not None:
            weights = [0.3, 0.3, 0.4]  # 如果有LNN，调整权重
        
        with open(os.path.join(model_dir, "weights.json"), "w") as f:
            json.dump(weights, f)
        
        # 保存完整模型包
        model_package = {
            'feature_columns': final_features,
            'xgb_model': strength_model,
            'lgb_model': market_model,
            'lnn_model': lnn_model,
            'scaler': scaler,
            'weights': weights,
            'model_name': model_name,
            'training_date': datetime.now().strftime('%Y%m%d')
        }
        
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model_package, f)
        
        self._save_models(model_name, models, final_features)
        
        self.logger.info(f"板块轮动模型训练完成，已保存到 {model_dir}")
        
        # 测试集评估 - 新增代码
        self.logger.info("开始在测试集上评估模型性能...")
        
        # 确保测试集不为空
        if not test_df.empty:
            # 评估测试集的性能
            test_results = self._evaluate_on_test_set(
                test_df=test_df,
                model_package=model_package
            )
            
            # 将评估结果添加到模型包中
            model_package['test_evaluation'] = test_results
            
            # 更新保存的模型包
            with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
                pickle.dump(model_package, f)
            
            # 保存测试结果到单独文件，便于查看
            test_results_file = os.path.join(model_dir, "test_evaluation.json")
            with open(test_results_file, "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"测试集评估完成，结果已保存到 {test_results_file}")
        else:
            self.logger.warning("测试集为空，无法进行评估")
        
        # 返回完整的模型包
        return model_package

    def _batch_load_features(self, sector_list, start_date, end_date):
        """批量加载多个板块多个日期的特征数据
        
        根据优化训练方案实现一次性批量加载和特征层次化处理：
        1. 基础特征层：所有板块的价格、成交量等原始数据
        2. 市场环境层：市场状态分类、大盘趋势特征
        3. 板块特有层：相对强度、季节性表现
        
        Args:
            sector_list: 板块列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            处理后的特征数据
        """
        self.logger.info(f"开始批量加载 {len(sector_list)} 个板块从 {start_date} 到 {end_date} 的特征数据")
        
        # 获取日期范围内的所有交易日
        if self.trading_days is None:
            self._load_trading_calendar()
    
        trading_days = [day for day in self.trading_days if start_date <= day <= end_date]
        if not trading_days:
            self.logger.warning(f"在{start_date}至{end_date}之间没有有效交易日")
            return pd.DataFrame()
        
        # 步骤1: 加载基础特征层 - 所有板块的基本数据
        # --------------------------------------------------------------------------
        self.logger.info("第1阶段: 加载所有板块的基础数据...")
        # 为了效率，使用并行处理
        base_features = {}
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
            futures = {}
            for sector_code in sector_list:
                future = executor.submit(self._load_base_features, sector_code, start_date, end_date)
                futures[future] = sector_code
            
            for future in as_completed(futures):
                sector_code = futures[future]
                try:
                    sector_base_data = future.result()
                    if sector_base_data is not None:
                        base_features[sector_code] = sector_base_data
                except Exception as e:
                    self.logger.error(f"加载板块 {sector_code} 基础数据时出错: {e}")
        
        if not base_features:
            self.logger.error("没有成功加载任何板块的基础数据")
            return pd.DataFrame()
        
        self.logger.info(f"成功加载 {len(base_features)} 个板块的基础数据")
        
        # 步骤2: 加载市场环境层 - 一次性加载所有日期的市场状态
        # --------------------------------------------------------------------------
        self.logger.info("第2阶段: 加载市场环境数据...")
        market_status_data = {}
        for date in trading_days:
            try:
                market_status = self._load_market_status(date)
                if market_status is not None:
                    market_status_data[date] = market_status
            except Exception as e:
                self.logger.error(f"加载日期 {date} 的市场状态时出错: {e}")
        
        if not market_status_data:
            self.logger.error("没有成功加载任何日期的市场状态数据")
            return pd.DataFrame()
        
        self.logger.info(f"成功加载 {len(market_status_data)} 个交易日的市场状态数据")
        
        # 步骤3: 处理板块特有层 - 结合基础数据和市场状态计算高级特征
        # --------------------------------------------------------------------------
        self.logger.info("第3阶段: 构建板块特有特征...")
        all_features = []
        
        # 计算板块季节性表现（低频计算）- 直接使用已加载的基础数据
        seasonal_patterns = {}
        for sector_code in sector_list:
            try:
                # 使用已加载的基础数据计算季节性模式
                if sector_code in base_features and not base_features[sector_code].empty:
                    seasonal_patterns[sector_code] = self._compute_seasonal_patterns(
                        sector_code, 
                        base_data=base_features[sector_code]
                    )
                else:
                    seasonal_patterns[sector_code] = self._compute_seasonal_patterns(sector_code)
            except Exception as e:
                self.logger.error(f"计算板块 {sector_code} 季节性表现时出错: {e}")
        
        # 一次性批量生成所有板块的所有日期的特征数据
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
            futures = {}
            tasks = []
            
            for sector_code in sector_list:
                if sector_code not in base_features:
                    continue
                
                for date in trading_days:
                    if date not in market_status_data:
                        continue
                    
                    # 提交任务
                    task = (sector_code, date)
                    tasks.append(task)
                    future = executor.submit(
                        self._process_features_with_layers,
                        sector_code=sector_code,
                        date=date,
                        base_data=base_features.get(sector_code),
                        market_status=market_status_data.get(date),
                        seasonal_data=seasonal_patterns.get(sector_code)
                    )
                    futures[future] = task
            
            # 收集结果
            for i, future in enumerate(as_completed(futures)):
                sector_code, date = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # 在将结果添加到all_features前，计算未来收益率
                        # 使用已有的全部数据计算，避免重复IO
                        if sector_code in base_features:
                            # 确保日期格式一致
                            date_obj = pd.to_datetime(date)
                            # 传递已加载的数据给_calculate_future_returns方法
                            future_returns = self._calculate_future_returns(
                                sector_code=sector_code,
                                date=date_obj,
                                days_list=[1, 2, 3, 5, 10],
                                all_data=base_features[sector_code]
                            )
                            # 将未来收益率添加到结果中
                            if future_returns and not result.empty:
                                for days, ret in future_returns.items():
                                    result[f"future_return_{days}d"] = ret
                    
                        all_features.append(result)
                    
                    # 进度日志
                    if (i+1) % 100 == 0 or (i+1) == len(tasks):
                        self.logger.info(f"已处理 {i+1}/{len(tasks)} 个特征构建任务")
                except Exception as e:
                    self.logger.error(f"处理板块 {sector_code} 日期 {date} 的特征时出错: {e}")
        
        self.logger.info(f"批量加载完成，成功构建 {len(all_features)} 条特征记录")
        
        # 步骤4: 划分训练集、验证集和测试集
        # --------------------------------------------------------------------------
        if all_features:
            all_features_df = pd.concat(all_features, ignore_index=True)
            self.logger.info(f"合并后特征数据集大小: {all_features_df.shape}")
            
            # 将数据集划分为训练集(70%)、验证集(10%)和测试集(20%)
            all_features_df = all_features_df.sort_values("date")
            total_days = len(all_features_df["date"].unique())
            
            train_cutoff = int(total_days * 0.7)
            valid_cutoff = int(total_days * 0.8)
            
            train_dates = sorted(all_features_df["date"].unique())[:train_cutoff]
            valid_dates = sorted(all_features_df["date"].unique())[train_cutoff:valid_cutoff]
            test_dates = sorted(all_features_df["date"].unique())[valid_cutoff:]
            
            # 添加数据集标识列
            all_features_df["dataset_type"] = "unknown"
            all_features_df.loc[all_features_df["date"].isin(train_dates), "dataset_type"] = "train"
            all_features_df.loc[all_features_df["date"].isin(valid_dates), "dataset_type"] = "valid"
            all_features_df.loc[all_features_df["date"].isin(test_dates), "dataset_type"] = "test"
            
            train_ratio = len(all_features_df[all_features_df["dataset_type"] == "train"]) / len(all_features_df)
            valid_ratio = len(all_features_df[all_features_df["dataset_type"] == "valid"]) / len(all_features_df)
            test_ratio = len(all_features_df[all_features_df["dataset_type"] == "test"]) / len(all_features_df)
            
            self.logger.info(f"数据集划分: 训练集 {train_ratio:.1%}, 验证集 {valid_ratio:.1%}, 测试集 {test_ratio:.1%}")
            
            return all_features_df
        else:
            self.logger.error("没有成功构建任何特征数据")
            return pd.DataFrame()

    def _load_base_features(self, sector_code, start_date, end_date):
        """加载单个板块的基础特征数据（价格、成交量等）
        
        Args:
            sector_code: 板块代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            板块基础数据
        """
        try:
            # 获取日期范围内的所有交易日
            if self.trading_days is None:
                self._load_trading_calendar()
                
            trading_days = [day for day in self.trading_days if start_date <= day <= end_date]
            if not trading_days:
                self.logger.warning(f"在{start_date}至{end_date}之间没有有效交易日")
                return None
                
            # 构建板块主特征文件路径
            sector_dir = self.sector_features_dir / f"{sector_code}"
            main_features_path = sector_dir / f"{sector_code}_features.parquet"
            
            # 如果主特征文件不存在，尝试使用合并特征文件
            if not main_features_path.exists():
                merged_file_path = sector_dir / f"{sector_code}.parquet"
                if merged_file_path.exists():
                    main_features_path = merged_file_path
                else:
                    self.logger.warning(f"板块 {sector_code} 没有主特征文件或合并特征文件")
                    return None
            
            # 读取特征文件
            self.logger.info(f"从 {main_features_path} 读取 {sector_code} 板块从 {start_date} 到 {end_date} 的特征数据")
            
            try:
                df = pd.read_parquet(main_features_path)
                
                # 确保日期列存在
                if 'date' not in df.columns:
                    self.logger.warning(f"特征文件 {main_features_path} 没有date列")
                    return None
                
                # 将日期列转换为datetime对象便于比较
                df['date'] = pd.to_datetime(df['date'])
                
                # 筛选日期范围内的数据
                mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
                filtered_df = df[mask].copy()
                
                if filtered_df.empty:
                    self.logger.warning(f"板块 {sector_code} 在 {start_date} 至 {end_date} 期间没有数据")
                    return None
                
                # 设置date为索引以便后续处理
                filtered_df.set_index('date', inplace=True)
                
                self.logger.info(f"成功加载 {sector_code} 板块特征，共 {len(filtered_df)} 条记录")
                return filtered_df
                
            except Exception as e:
                self.logger.error(f"读取 {main_features_path} 出错: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 基础数据时发生错误: {e}")
            return None

    def _process_features_with_layers(self, sector_code, date, base_data, market_status, seasonal_data):
        """使用三层特征体系处理单个板块单个日期的特征
        
        Args:
            sector_code: 板块代码
            date: 日期
            base_data: 基础层数据
            market_status: 市场环境层数据
            seasonal_data: 季节性特征数据
            
        Returns:
            处理后的特征数据
        """
        try:
            # 确保基础数据中包含当前日期
            if base_data is None or date not in base_data.index:
                return None
                
            # 获取当日数据
            current_day = base_data.loc[date]
            
            # 构建特征字典
            features = {}
            
            # 1. 基础层特征 - 直接使用预计算的基础特征
            # ------------------------------------------------------------
            features["sector_code"] = sector_code
            features["date"] = date
            
            # 添加基础价格类指标
            for field in ["close", "open", "high", "low", "volume", "amount", 
                        "change_percent", "turnover_rate", "amplitude"]:
                if field in current_day:
                    features[field] = current_day[field]
            
            # 添加股票统计信息
            for field in ["stock_count", "up_count", "down_count", "flat_count"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 提取上涨/下跌比例
            if all(f in current_day for f in ["up_count", "stock_count"]) and current_day["stock_count"] > 0:
                features["up_ratio"] = current_day["up_count"] / current_day["stock_count"]
                
            if all(f in current_day for f in ["down_count", "stock_count"]) and current_day["stock_count"] > 0:
                features["down_ratio"] = current_day["down_count"] / current_day["stock_count"]
                
            # 2. 技术指标层 - 直接使用预计算的技术指标
            # ------------------------------------------------------------
            # 添加均线数据
            for ma in ["ma5", "ma10", "ma20", "ma30", "ma60", "ma120"]:
                if ma in current_day:
                    features[ma] = current_day[ma]
                    
            # 添加均线比率指标
            for ratio in ["ma5_10_ratio", "ma5_20_ratio", "ma10_20_ratio", 
                         "ma20_60_ratio", "ma60_120_ratio", "price_ma5_ratio", 
                         "price_ma20_ratio", "price_ma60_ratio"]:
                if ratio in current_day:
                    features[ratio] = current_day[ratio]
                    
            # 添加价格位置指标
            for pos in ["price_above_ma5", "price_above_ma20", "price_above_ma60"]:
                if pos in current_day:
                    features[pos] = current_day[pos]
                    
            # 添加趋势指标
            for trend in ["trend_direction_score", "ma_bull_trend", "ma_bear_trend", 
                         "ma_neutral_trend", "ma5_slope", "ma20_slope"]:
                if trend in current_day:
                    features[trend] = current_day[trend]
                    
            # 添加动量指标
            for mom in ["momentum_5d", "momentum_10d", "momentum_20d", "change_pct"]:
                if mom in current_day:
                    features[mom] = current_day[mom]
                    
            # 添加RSI指标
            for rsi in ["rsi6", "rsi14", "rsi_overbought", "rsi_oversold", "rsi_neutral"]:
                if rsi in current_day:
                    features[rsi] = current_day[rsi]
                    
            # 添加MACD指标
            for macd in ["macd_line", "macd_signal", "macd_histogram", "macd_trend"]:
                if macd in current_day:
                    features[macd] = current_day[macd]
                    
            # 添加波动率指标
            for vol in ["volatility", "volatility_5d", "volatility_10d", 
                       "volatility_20d", "volatility_60d", "volatility_ratio"]:
                if vol in current_day:
                    features[vol] = current_day[vol]
                    
            # 添加成交量指标
            for vol in ["volume_ratio_5", "volume_ratio_10", "volume_ratio_20", 
                       "volume_ratio_60", "volume_change_ratio", "volume_trend"]:
                if vol in current_day:
                    features[vol] = current_day[vol]
                    
            # 添加趋势强度和连续天数指标
            for field in ["trend_strength", "consecutive_up_days", "consecutive_down_days", 
                         "swing_pattern"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 添加支撑阻力位指标
            for field in ["support_level", "resistance_level", "support_distance_pct", 
                         "resistance_distance_pct"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 添加技术评分和市场情绪指标
            for field in ["tech_score", "market_sentiment", "signal_strength"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 添加涨跌停相关指标
            for field in ["limit_up_ratio", "limit_down_ratio", "near_limit_up_ratio", 
                         "limit_open_ratio"]:
                if field in current_day:
                    features[field] = current_day[field]
                    
            # 添加排名相关指标
            for field in ["ten_day_rise_count", "ten_day_rise_ratio", "daily_rise_count", 
                         "daily_rise_ratio", "daily_fall_count", "daily_fall_ratio", 
                         "daily_amount_count", "daily_amount_ratio", "ten_day_rise_weighted"]:
                if field in current_day:
                    features[field] = current_day[field]
            
            # 3. 市场环境层 - 直接使用提供的市场状态数据
            # ------------------------------------------------------------
            if market_status:
                # 添加市场类型和周期指标
                features["market_type"] = market_status.get("market_type", "unknown")
                features["market_cycle"] = market_status.get("market_cycle", 0)
                features["market_trend_strength"] = market_status.get("trend_strength", 0)
                
                # 添加市场信号
                signals = market_status.get("signals", {})
                features["market_risk_level"] = signals.get("risk_level", "medium")
                features["market_signal_strength"] = signals.get("strength", 0)
                
                # 添加创业板指数数据(399006.SZ)
                if "individual_status" in market_status and "399006.SZ" in market_status["individual_status"]:
                    cyb_data = market_status["individual_status"]["399006.SZ"]
                    
                    # 提取评分
                    for score in ["trend_score", "position_score", "volume_score", 
                                 "momentum_score", "volatility_score", "structure_score", 
                                 "cycle_score", "total_score"]:
                        if score in cyb_data:
                            features[f"cyb_{score}"] = cyb_data[score]
                    
                    # 提取技术数据
                    if "technical_data" in cyb_data:
                        tech_data = cyb_data["technical_data"]
                        for tech_field in ["rsi6", "rsi12", "macd", "bb_width", "bb_position"]:
                            if tech_field in tech_data:
                                features[f"cyb_{tech_field}"] = tech_data[tech_field]
                    
                    # 提取盈亏比
                    if "profit_loss_ratio_14" in cyb_data:
                        features["cyb_profit_loss_ratio_14"] = cyb_data["profit_loss_ratio_14"]
                    if "profit_loss_ratio_30" in cyb_data:
                        features["cyb_profit_loss_ratio_30"] = cyb_data["profit_loss_ratio_30"]
            
            # 4. 板块特有层 - 添加板块的季节性特征
            # ------------------------------------------------------------
            if seasonal_data:
                # 添加月度和季度季节性表现
                date_obj = pd.to_datetime(date)
                current_month = date_obj.month
                current_quarter = (date_obj.month - 1) // 3 + 1
                
                # 月度季节性表现
                if "monthly" in seasonal_data and current_month in seasonal_data["monthly"]:
                    features["seasonal_month_return"] = seasonal_data["monthly"][current_month]
                    
                # 季度季节性表现
                if "quarterly" in seasonal_data and current_quarter in seasonal_data["quarterly"]:
                    features["seasonal_quarter_return"] = seasonal_data["quarterly"][current_quarter]
                    
                # 市场环境适应性
                if "market_fit" in seasonal_data:
                    market_type = str(features.get("market_type", "unknown"))
                    if market_type in seasonal_data["market_fit"]:
                        features["market_fit_score"] = seasonal_data["market_fit"][market_type]
            
            # 5. 日期相关特征 - 添加日历特征
            # ------------------------------------------------------------
            date_obj = pd.to_datetime(date)
            features["month"] = date_obj.month
            features["day_of_month"] = date_obj.day
            features["day_of_week"] = date_obj.dayofweek
            features["is_month_start"] = 1 if date_obj.is_month_start else 0
            features["is_month_end"] = 1 if date_obj.is_month_end else 0
            features["is_quarter_start"] = 1 if date_obj.is_quarter_start else 0
            features["is_quarter_end"] = 1 if date_obj.is_quarter_end else 0
            
            # 6. 标签数据
            # ------------------------------------------------------------
            # 注意：未来收益率现在在_batch_load_features方法中计算并添加
            # 这里只添加分类标签，如果已经有未来收益率
            if "future_return_2d" in current_day:
                features["future_return_2d"] = current_day["future_return_2d"]
                features["is_top_performer"] = 1 if current_day["future_return_2d"] > 0.03 else 0
            elif "change_percent" in current_day:
                # 如果没有未来收益率，但有涨跌幅，使用涨跌幅作为替代
                if current_day["change_percent"] > 3:  # 3%的涨幅作为高表现阈值
                    features["is_top_performer"] = 1
                else:
                    features["is_top_performer"] = 0
            
            return pd.DataFrame([features])
        except Exception as e:
            self.logger.error(f"处理板块 {sector_code} 日期 {date} 特征时错误: {str(e)}")
            traceback.print_exc()
            return None

    def backtest_rotation_model(self, sector_list, start_date, end_date, model_version="latest", top_n=5, capital=1000000):
        """对板块轮动模型进行回测
        
        Args:
            sector_list: 板块代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            model_version: 模型版本，"latest"表示使用最新模型，或指定日期格式为"YYYYMMDD"
            top_n: 每个交易日选择表现最好的前N个板块
            capital: 初始资金
            
        Returns:
            回测结果字典
        """
        self.logger.info(f"开始回测 {start_date} 至 {end_date}，使用模型版本: {model_version}")
        
        # 1. 获取回测日期范围内的所有交易日
        calendar = self._load_trading_calendar()
        
        # 确保日期格式统一
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 筛选回测日期范围内的交易日
        backtest_dates = calendar[(calendar >= start_date) & (calendar <= end_date)].sort_values()
        
        if len(backtest_dates) == 0:
            self.logger.error(f"回测日期范围 {start_date} 至 {end_date} 内无有效交易日")
            return None
        
        self.logger.info(f"回测日期范围内共有 {len(backtest_dates)} 个交易日")
        
        # 2. 初始化回测结果数据结构
        backtest_results = {
            'dates': [],
            'predictions': [],
            'selected_sectors': [],
            'daily_returns': [],
            'portfolio_values': [],
            'market_types': [],
            'capital': capital,
            'initial_capital': capital
        }
        
        # 3. 对每个交易日进行预测和模拟交易
        positions = {}  # 当前持仓情况
        prev_date = None
        
        for date_idx, current_date in enumerate(backtest_dates):
            date_str = current_date.strftime("%Y%m%d")
            self.logger.info(f"处理回测日期: {date_str} ({date_idx+1}/{len(backtest_dates)})")
            
            try:
                # 获取市场状态
                market_status = self._load_market_status(current_date)
                market_type = market_status.get('market_type', 'normal') if market_status else 'normal'
                
                # 确定使用哪个日期的模型
                if model_version == "latest":
                    # 查找小于等于当前日期的最大日期作为模型日期
                    model_dir = os.path.join(self.model_dir, "sector_rotation")
                    if not os.path.exists(model_dir):
                        self.logger.error(f"模型目录 {model_dir} 不存在")
                        continue
                    
                    # 获取所有日期文件夹
                    available_models = [d for d in os.listdir(model_dir) 
                                       if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()]
                    
                    # 筛选小于等于当前日期的模型
                    available_models = [d for d in available_models if d <= date_str]
                    
                    if not available_models:
                        self.logger.warning(f"无法找到 {date_str} 之前的可用模型，跳过该日期")
                        continue
                    
                        # 使用最近的模型
                        model_date = max(available_models)
                        model_path = os.path.join("sector_rotation", model_date)
                    else:
                        # 使用指定版本的模型
                        model_path = os.path.join("sector_rotation", model_version)
                
                # 为当前日期准备特征数据
                features_data = {}
                for sector_code in sector_list:
                    # 构建特征
                    sector_features = self._build_features_for_date_range(
                        sector_code=sector_code,
                        end_date=current_date,
                        lookback_days=60,
                        forward_days=self.forward_days
                    )
                    
                    if sector_features is not None and not sector_features.empty:
                        # 获取最新日期的特征数据
                        latest_features = sector_features.iloc[-1:]
                        features_data[sector_code] = latest_features
                
                if not features_data:
                    self.logger.warning(f"日期 {date_str} 没有可用的特征数据")
                    continue
                
                # 加载模型
                model_file = os.path.join(self.model_dir, model_path, "model.pkl")
                if not os.path.exists(model_file):
                    self.logger.error(f"模型文件 {model_file} 不存在")
                    continue
                    
                with open(model_file, "rb") as f:
                    model_data = pickle.load(f)
                
                feature_columns = model_data['feature_columns']
                xgb_model = model_data['xgb_model']
                lgb_model = model_data['lgb_model']
                weights = model_data['weights']
                
                # 预测每个板块的表现
                sector_predictions = {}
                
                for sector_code, features in features_data.items():
                    if features is None or features.empty:
                        continue
                    
                    # 确保特征顺序一致
                    X = features[feature_columns].values
                    
                    # XGBoost预测
                    xgb_pred = xgb_model.predict_proba(X)[:, 1]
                    
                    # LightGBM预测
                    lgb_pred = lgb_model.predict_proba(X)[:, 1]
                    
                    # 如果有LNN模型，也进行预测
                    if 'lnn_model' in model_data and 'lnn_scaler' in model_data:
                        lnn_model = model_data['lnn_model']
                        lnn_scaler = model_data['lnn_scaler']
                        
                        # 标准化特征
                        X_scaled = lnn_scaler.transform(X)
                        
                        # LNN预测
                        if torch is not None:
                            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                            X_tensor = torch.FloatTensor(X_scaled).to(device)
                            with torch.no_grad():
                                lnn_pred = lnn_model(X_tensor).cpu().numpy().flatten()
                        else:
                            lnn_pred = np.zeros_like(xgb_pred)
                            
                        # 融合预测结果
                        if len(weights) == 3:  # 如果有3个权重，说明包含LNN
                            ensemble_pred = (
                                weights[0] * xgb_pred + 
                                weights[1] * lgb_pred + 
                                weights[2] * lnn_pred
                            )
                        else:
                            ensemble_pred = weights[0] * xgb_pred + weights[1] * lgb_pred
                    else:
                        # 只融合XGBoost和LightGBM
                        ensemble_pred = weights[0] * xgb_pred + weights[1] * lgb_pred
                    
                    # 保存预测结果
                    sector_predictions[sector_code] = ensemble_pred[0]
                
                # 将预测结果按得分从高到低排序
                sorted_sectors = sorted(
                    sector_predictions.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # 根据市场环境调整板块选择
                adapted_sectors = self._adapt_sectors_to_market(
                    sorted_sectors=sorted_sectors,
                    market_type=market_type,
                    market_status=market_status
                )
                
                # 选择前top_n个板块
                selected_sectors = [sector for sector, _ in adapted_sectors[:top_n]]
                
                # 模拟交易逻辑
                if prev_date is not None:
                    # 计算持仓收益
                    portfolio_return = 0.0
                    
                    for sector_code, position in positions.items():
                        # 获取该板块从上一交易日到当前日的收益率
                        sector_features = self._load_sector_features(sector_code, current_date)
                        prev_features = self._load_sector_features(sector_code, prev_date)
                        
                        if sector_features is not None and prev_features is not None:
                            prev_close = prev_features.get('close', None)
                            curr_close = sector_features.get('close', None)
                            
                            if prev_close is not None and curr_close is not None and prev_close > 0:
                                sector_return = (curr_close - prev_close) / prev_close
                                portfolio_return += sector_return * position / capital
                    
                    # 更新资金
                    capital *= (1 + portfolio_return)
                    
                    # 记录当日回报
                    backtest_results['daily_returns'].append(portfolio_return)
                    backtest_results['portfolio_values'].append(capital)
                else:
                    # 回测第一天，初始化数据
                    backtest_results['daily_returns'].append(0.0)
                    backtest_results['portfolio_values'].append(capital)
                
                # 更新持仓
                positions = {}
                if selected_sectors:
                    # 平均分配资金
                    position_value = capital / len(selected_sectors)
                    for sector_code in selected_sectors:
                        positions[sector_code] = position_value
                
                # 记录结果
                backtest_results['dates'].append(date_str)
                backtest_results['predictions'].append(sector_predictions)
                backtest_results['selected_sectors'].append(selected_sectors)
                backtest_results['market_types'].append(market_type)
                
                # 更新prev_date
                prev_date = current_date
            
            except Exception as e:
                self.logger.error(f"处理日期 {date_str} 时出错: {e}")
                traceback.print_exc()
        
        # 4. 计算回测指标
        if len(backtest_results['daily_returns']) > 0:
            # 总回报
            total_return = (capital / backtest_results['initial_capital']) - 1.0
            
            # 年化回报
            days = len(backtest_results['daily_returns'])
            if days > 1:
                annual_return = ((1 + total_return) ** (252 / days)) - 1
            else:
                annual_return = 0.0
            
            # 计算最大回撤
            portfolio_values = np.array(backtest_results['portfolio_values'])
            max_drawdown = 0.0
            peak = portfolio_values[0]
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # 计算夏普比率
            daily_returns = np.array(backtest_results['daily_returns'])
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # 胜率
            win_days = sum(1 for r in daily_returns if r > 0)
            win_rate = win_days / len(daily_returns) if len(daily_returns) > 0 else 0.0
            
            # 添加指标到结果
            backtest_results['metrics'] = {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'num_trades': len(backtest_results['dates']),
            }
            
            # 打印主要回测指标
            self.logger.info("\n======= 回测结果 =======")
            self.logger.info(f"回测期间: {start_date} 至 {end_date}")
            self.logger.info(f"交易天数: {days}")
            self.logger.info(f"总收益率: {total_return*100:.2f}%")
            self.logger.info(f"年化收益率: {annual_return*100:.2f}%")
            self.logger.info(f"最大回撤: {max_drawdown*100:.2f}%")
            self.logger.info(f"夏普比率: {sharpe_ratio:.2f}")
            self.logger.info(f"胜率: {win_rate*100:.2f}%")
            
            # 保存回测结果
            result_path = os.path.join(get_project_root(), "to", "results", "backtest")
            ensure_dir(result_path)
            
            result_file = os.path.join(
                result_path, 
                f"sector_rotation_backtest_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
            )
            
            try:
                with open(result_file, "wb") as f:
                    pickle.dump(backtest_results, f)
                self.logger.info(f"回测结果已保存至 {result_file}")
            except Exception as e:
                self.logger.error(f"保存回测结果失败: {e}")
        
        return backtest_results

    def _adapt_sectors_to_market(self, sorted_sectors, market_type, market_status):
        """根据市场环境适应性选择板块
        
        Args:
            sorted_sectors: 按预测收益排序的板块
            market_type: 市场类型
            market_status: 市场状态信息
            
        Returns:
            适应市场环境的板块列表
        """
        # 从排序后的板块中提取基本信息
        sector_list = [s[0] for s in sorted_sectors]
        if not sector_list:
            return []
        
        # 根据市场类型调整选择策略
        if market_type in [11, 12, 13]:  # 强势上涨
            # 选择预测收益最高的板块，偏向进攻性策略
            return sector_list[:5]
        elif market_type in [21, 22, 23]:  # 普通上涨
            # 选择预测收益较高且相对稳定的板块
            return sector_list[:4]
        elif market_type in [31, 32, 33]:  # 盘整
            # 选择预测收益适中、波动较小的板块
            return sector_list[1:4]
        elif market_type in [41, 42, 43]:  # 普通下跌
            # 选择预测收益相对较高的防御型板块
            return sector_list[:3]
        elif market_type in [51, 52, 53]:  # 强势下跌
            # 集中选择预测收益最高的1-2个板块或考虑空仓
            return sector_list[:2]
        else:
            # 默认策略
            return sector_list[:3]

    def analyze_feature_importance(self, sector_code, X, y, model=None):
        """分析特征重要性，支持回测期间动态特征选择
        
        Args:
            sector_code: 板块代码
            X: 特征数据
            y: 标签数据
            model: 已训练的模型，如果为None则训练一个新模型
            
        Returns:
            特征重要性字典
        """
        if model is None:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X, y)
        
        # 提取特征重要性
        importance_dict = {}
        for i, col in enumerate(X.columns):
            importance_dict[col] = model.feature_importances_[i]
        
        # 按重要性排序
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 分类特征
        feature_categories = {
            "trend": [],
            "momentum": [],
            "volatility": [],
            "relative_strength": [],
            "seasonal": [],
            "volume": [],
            "other": []
        }
        
        # 分类特征
        for feature, importance in sorted_importance:
            if "trend" in feature or "ma" in feature:
                feature_categories["trend"].append((feature, importance))
            elif "momentum" in feature or "acceleration" in feature:
                feature_categories["momentum"].append((feature, importance))
            elif "volatility" in feature or "atr" in feature:
                feature_categories["volatility"].append((feature, importance))
            elif "relative" in feature or "rel_strength" in feature:
                feature_categories["relative_strength"].append((feature, importance))
            elif "month" in feature or "quarter" in feature:
                feature_categories["seasonal"].append((feature, importance))
            elif "volume" in feature:
                feature_categories["volume"].append((feature, importance))
        else:
                feature_categories["other"].append((feature, importance))
        
        # 获取最近的市场状态
        try:
            latest_date = max(self.trading_days)
            market_status = self._load_market_status(latest_date)
            if market_status and 'market_type' in market_status:
                market_type = market_status['market_type']
                
                # 根据市场状态推荐特征权重
                recommended_weights = {}
                if market_type in [11, 12, 13]:  # 强势上涨
                    recommended_weights = {
                        "momentum": 0.3,
                        "relative_strength": 0.3,
                        "trend": 0.2,
                        "volume": 0.1,
                        "volatility": 0.05,
                        "seasonal": 0.05
                    }
                elif market_type in [31, 32, 33]:  # 盘整
                    recommended_weights = {
                        "relative_strength": 0.35,
                        "volatility": 0.2,
                        "volume": 0.2,
                        "trend": 0.1,
                        "momentum": 0.1,
                        "seasonal": 0.05
                    }
                elif market_type in [51, 52, 53]:  # 下跌
                    recommended_weights = {
                        "volatility": 0.3,
                        "relative_strength": 0.25,
                        "volume": 0.15,
                        "trend": 0.15,
                        "momentum": 0.1,
                        "seasonal": 0.05
                    }
                else:  # 默认
                    recommended_weights = {
                        "relative_strength": 0.25,
                        "momentum": 0.20,
                        "trend": 0.20,
                        "volatility": 0.15,
                        "volume": 0.10,
                        "seasonal": 0.10
                    }
                
                feature_categories["recommended_weights"] = recommended_weights
        except Exception as e:
            self.logger.error(f"分析特征重要性时出错: {e}")
        
        return {
            "sorted_importance": sorted_importance[:30],  # 前30个最重要特征
            "feature_categories": feature_categories,
            "sector_code": sector_code
        }

    def prepare_backtest_data(self, sector_codes, start_date, end_date, model_version="latest"):
        """准备回测数据
        
        Args:
            sector_codes: 板块代码列表
            start_date: 回测开始日期
            end_date: 回测结束日期
            model_version: 模型版本，默认使用最新
            
        Returns:
            回测数据字典，包含每个日期的预测结果
        """
        backtest_results = {}
        
        # 获取交易日范围
        if start_date not in self.trading_days:
            start_idx = bisect.bisect_left(self.trading_days, start_date)
            start_date = self.trading_days[start_idx] if start_idx < len(self.trading_days) else None
        
        if end_date not in self.trading_days:
            end_idx = bisect.bisect_right(self.trading_days, end_date) - 1
            end_date = self.trading_days[end_idx] if end_idx >= 0 else None
        
        if not start_date or not end_date or start_date > end_date:
            self.logger.error(f"无效的回测日期范围: {start_date} - {end_date}")
            return None
        
        # 确定回测日期列表
        backtest_range = []
        start_idx = self.trading_days.index(start_date)
        end_idx = self.trading_days.index(end_date)
        backtest_range = self.trading_days[start_idx:end_idx+1]
        
        # 加载模型
        models = {}
        for sector_code in sector_codes:
            model_dir = self.model_dir / sector_code
            if not model_dir.exists():
                self.logger.warning(f"板块 {sector_code} 没有找到模型")
                continue
                
            try:
                # 加载特征列
                with open(model_dir / "feature_columns.json", "r") as f:
                    feature_columns = json.load(f)
                    
                # 加载模型
                model_files = sorted(model_dir.glob("*_model.*"))
                if not model_files:
                    self.logger.warning(f"板块 {sector_code} 没有找到模型文件")
                    continue
                    
                # 选择模型文件
                model_path = None
                for mf in model_files:
                    if "xgboost" in str(mf):
                        model_path = mf
                        break
                
                if not model_path:
                    self.logger.warning(f"板块 {sector_code} 没有找到XGBoost模型文件")
                    continue
                    
                # 加载模型
                model = xgb.XGBRegressor()
                model.load_model(str(model_path))
                
                # 加载缩放器
                scaler_path = model_dir / "scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, "rb") as f:
                        import pickle
                        scaler = pickle.load(f)
                else:
                    scaler = StandardScaler()  # 默认缩放器
                    
                models[sector_code] = {
                    "model": model,
                    "scaler": scaler,
                    "features": feature_columns
                }
                self.logger.info(f"成功加载板块 {sector_code} 的模型")
            except Exception as e:
                self.logger.error(f"加载板块 {sector_code} 模型时出错: {e}")
                continue
        
        # 执行回测
        for date in backtest_range:
            date_results = {
                "date": date,
                "predictions": {}
            }
            
            # 获取当前市场状态
            market_status = self._load_market_status(date)
            if not market_status:
                self.logger.warning(f"日期 {date} 没有市场状态数据")
                continue
                
            date_results["market_status"] = {
                "market_type": market_status.get("market_type", "unknown"),
                "market_cycle": market_status.get("market_cycle", "unknown"),
                "trend_strength": market_status.get("trend_strength", 0.0)
            }
            
            # 获取各板块预测
            sector_predictions = {}
            for sector_code, model_info in models.items():
                # 构建特征
                short_features = self._build_features_for_date_range(
                    sector_code, date,
                    lookback_days=self.lookback_short,
                    forward_days=self.forward_days
                )
                
                long_features = self._build_features_for_date_range(
                    sector_code, date,
                    lookback_days=self.lookback_long,
                    forward_days=self.forward_days
                )
                
                if short_features is None or long_features is None:
                    self.logger.warning(f"日期 {date} 板块 {sector_code} 无法构建完整特征")
                    continue
                    
                # 合并特征
                # 为区分特征来源，添加前缀
                short_cols = {}
                long_cols = {}
                
                # 排除不需要加前缀的通用列
                exclude_prefix = ['feature_date', 'label_date', 'future_return', 'sector_code', 'date']
                
                for col in short_features.columns:
                    if col not in exclude_prefix:
                        short_cols[col] = f'short_{col}'
                
                for col in long_features.columns:
                    if col not in exclude_prefix:
                        long_cols[col] = f'long_{col}'
                
                # 重命名列添加前缀
                short_features.rename(columns=short_cols, inplace=True)
                long_features.rename(columns=long_cols, inplace=True)
                
                # 确保合并的键存在
                if 'date' not in short_features.columns:
                    short_features['date'] = date
                
                if 'date' not in long_features.columns:
                    long_features['date'] = date
                
                # 合并特征
                merged_features = pd.merge(
                    short_features,
                    long_features,
                    on='date',
                    how='inner',
                    suffixes=('', '_long')
                )
                
                if merged_features.empty:
                    self.logger.warning(f"日期 {date} 板块 {sector_code} 合并特征为空")
                    continue
                
                # 添加市场环境特征和动态特征
                self._add_dynamic_features(merged_features, market_status)
                
                # 准备预测用特征
                req_features = model_info["features"]
                avail_features = merged_features.columns
                
                # 检查特征匹配情况
                missing_features = set(req_features) - set(avail_features)
                if missing_features:
                    self.logger.warning(f"板块 {sector_code} 缺少特征: {missing_features}")
                    # 为缺失特征填充0值
                    for feat in missing_features:
                        merged_features[feat] = 0
                
                X_pred = merged_features[req_features].fillna(0)
                
                # 标准化特征
                X_pred_scaled = model_info["scaler"].transform(X_pred)
                
                # 预测
                try:
                    prediction = model_info["model"].predict(X_pred_scaled)[0]
                    
                    # 存储预测结果
                    sector_predictions[sector_code] = {
                        "predicted_return": float(prediction),
                        "features": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                   for k, v in merged_features.iloc[0].items()
                                   if k in req_features[:10]}  # 只保存部分重要特征
                    }
                except Exception as e:
                    self.logger.error(f"预测板块 {sector_code} 在日期 {date} 的收益时出错: {e}")
            
            # 记录预测结果
            date_results["predictions"] = sector_predictions
            
            # 按预测收益排序
            sorted_sectors = sorted(
                sector_predictions.items(),
                key=lambda x: x[1]["predicted_return"],
                reverse=True
            )
            
            # 记录排名结果
            date_results["ranked_sectors"] = [s[0] for s in sorted_sectors]
            date_results["top_sectors"] = date_results["ranked_sectors"][:5]  # 取前5名
            
            # 市场环境适应型板块选择
            market_type = date_results["market_status"]["market_type"]
            market_adapted_sectors = self._adapt_sectors_to_market(
                sorted_sectors, market_type, date_results["market_status"]
            )
            date_results["market_adapted_sectors"] = market_adapted_sectors
            
            # 保存当日回测结果
            backtest_results[date] = date_results
        
        return backtest_results

    # 添加特征缓存函数
    def _get_cache_path(self, sector_code, date, feature_type='short'):
        """
        获取特征缓存文件路径
        
        Args:
            sector_code: 板块代码
            date: 日期
            feature_type: 特征类型，'short'或'long'
            
        Returns:
            Path: 缓存文件路径
        """
        cache_dir = self.features_cache_dir / feature_type
        ensure_dir(cache_dir)
        return cache_dir / f"{sector_code}_{date}.pkl"
    
    def _check_if_long_features_need_update(self, date):
        """
        检查长期特征是否需要更新
        通常在月初、季度初或市场状态变化显著时更新
        
        Args:
            date: 当前日期
            
        Returns:
            bool: 是否需要更新
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # 判断是否为月初(1-3号)
        if date.day <= 3:
            return True
            
        # 检查最近的市场状态变化
        try:
            # 获取最近几天的市场状态
            prev_days = 5
            market_types = []
            
            # 使用类的交易日历属性，避免重复加载
            date_str = date.strftime('%Y%m%d')
            
            # 确保交易日历不为None
            if self.trading_days is None:
                self.logger.warning("交易日历为None，无法检查市场状态变化")
                return True
            
            if date_str in self.trading_days:
                idx = self.trading_days.index(date_str)
                if idx >= prev_days:
                    for i in range(prev_days):
                        prev_date = self.trading_days[idx-i-1]
                        status = self._load_market_status(prev_date)
                        if status and 'market_type' in status:
                            market_types.append(status['market_type'])
            
            # 如果市场状态变化显著，更新长期特征
            if len(market_types) >= 3 and len(set(market_types[:2])) > 1:
                return True
        except Exception as e:
            self.logger.warning(f"检查市场状态变化时出错: {e}")
            return True  # 出错时默认需要更新
            
        # 检查缓存文件是否已过期
        cache_files = list(self.features_cache_dir.glob("long/*.pkl"))
        if not cache_files:
            return True
            
        # 检查最新的缓存文件日期
        try:
            latest_cache = max(cache_files, key=lambda x: os.path.getmtime(x))
            cache_time = datetime.fromtimestamp(os.path.getmtime(latest_cache))
            days_diff = (date - cache_time.replace(tzinfo=None)).days
            
            return days_diff >= self.long_feature_refresh_days
        except Exception as e:
            self.logger.warning(f"检查缓存文件日期时出错: {e}")
            return True
            
        return False
    
    def _build_or_load_features(self, sector_code, date, feature_type='short'):
        """
        构建或加载特征
        
        Args:
            sector_code: 板块代码
            date: 日期
            feature_type: 特征类型，'short'或'long'
            
        Returns:
            DataFrame: 特征数据
        """
        # 获取缓存路径
        cache_path = self._get_cache_path(sector_code, date, feature_type)
        
        # 检查缓存是否存在且有效
        if cache_path.exists():
            # 检查缓存是否已过期
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            now = datetime.now()
            days_diff = (now - cache_time).days
            
            refresh_days = self.long_feature_refresh_days if feature_type == 'long' else self.short_feature_refresh_days
            
            if days_diff < refresh_days:
                try:
                    # 加载缓存
                    return pd.read_pickle(cache_path)
                except Exception as e:
                    self.logger.warning(f"加载特征缓存时出错: {e}")
        
        # 构建新特征 - 使用预定义的回溯天数
        # 不同特征类型的回溯期不同
        if feature_type == 'long':
            lookback_days = 60  # 固定长期特征回溯60天
        else:
            lookback_days = 20  # 固定短期特征回溯20天
            
        features = self._build_features_for_date_range(
            sector_code, date, lookback_days=lookback_days, forward_days=self.forward_days
        )
        
        # 如果成功构建，保存到缓存
        if features is not None:
            try:
                features.to_pickle(cache_path)
                self.logger.debug(f"成功缓存{feature_type}特征: {cache_path}")
            except Exception as e:
                self.logger.warning(f"保存特征缓存时出错: {e}")
                
        return features
    
    def _compute_seasonal_patterns(self, sector_code, base_data=None, years=3):
        """计算板块的季节性模式
    
    Args:
            sector_code: 板块代码
            base_data: 预加载的板块历史数据，如果提供则直接使用
            years: 回溯年数
        
    Returns:
            dict: 季节性模式特征
        """
        self.logger.info(f"计算板块 {sector_code} 的季节性模式，回溯{years}年")
        
        try:
            # 如果未提供预加载数据，从文件加载数据
            if base_data is None or base_data.empty:
                self.logger.info(f"未提供预加载数据，从文件加载板块 {sector_code} 历史数据")
                # 获取当前日期并回溯年数
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y%m%d')
                
                # 加载板块的历史数据
                features_df = self._build_features_for_date_range(
                    sector_code=sector_code,
                    end_date=end_date,
                    lookback_days=365 * years,
                    forward_days=0
                )
                
                if features_df is None or features_df.empty:
                    self.logger.warning(f"板块 {sector_code} 历史数据不足，无法计算季节性模式")
                    return {}
                
                self.logger.info(f"使用预加载数据计算板块 {sector_code} 季节性模式")
                # 确保日期列存在并格式正确
                features_df = base_data.copy()
                if 'date' not in features_df.columns and isinstance(features_df.index, pd.DatetimeIndex):
                    features_df['date'] = features_df.index
                    features_df.reset_index(drop=True, inplace=True)
                
                # 确保日期是datetime类型
                if 'date' in features_df.columns and not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                    features_df['date'] = pd.to_datetime(features_df['date'])
                    
                # 确保数据足够多
                if len(features_df) < 30:
                    self.logger.warning(f"板块 {sector_code} 历史数据不足，无法计算季节性模式")
                    return {}
            
            # 添加日期相关列
            features_df['month'] = features_df['date'].dt.month
            features_df['quarter'] = features_df['date'].dt.quarter
            features_df['year'] = features_df['date'].dt.year
            
            # 确保有收益率数据
            if 'future_return_5d' not in features_df.columns and 'change_percent' in features_df.columns:
                features_df['future_return_5d'] = features_df['change_percent']  # 如果没有未来收益，使用当日涨跌幅
            
            # 月度表现 - 计算每个月的平均收益率
            monthly_returns = {}
            for month in range(1, 13):
                month_data = features_df[features_df['month'] == month]
                if len(month_data) > 0:
                    avg_return = month_data['future_return_5d'].mean()
                    monthly_returns[month] = float(avg_return)
            
            # 季度表现 - 计算每个季度的平均收益率
            quarterly_returns = {}
            for quarter in range(1, 5):
                quarter_data = features_df[features_df['quarter'] == quarter]
                if len(quarter_data) > 0:
                    avg_return = quarter_data['future_return_5d'].mean()
                    quarterly_returns[quarter] = float(avg_return)
            
            # 市场适应性 - 如果有市场类型数据，计算不同市场类型下的表现
            market_fit = {}
            if 'market_type' in features_df.columns:
                market_types = features_df['market_type'].unique()
                for market_type in market_types:
                    market_data = features_df[features_df['market_type'] == market_type]
                    if len(market_data) > 3:  # 确保有足够的样本
                        avg_return = market_data['future_return_5d'].mean()
                        market_fit[str(market_type)] = float(avg_return)
            
            # 整合结果
            seasonal_patterns = {
                'monthly': monthly_returns,
                'quarterly': quarterly_returns,
                'market_fit': market_fit
            }
            
            return seasonal_patterns
        except Exception as e:
            self.logger.error(f"计算板块 {sector_code} 季节性模式时出错: {e}")
            return {}

    def _calculate_future_returns(self, sector_code, date, days_list, all_data=None):
        """计算未来收益率
        
        Args:
            sector_code: 板块代码
            date: 当前日期
            days_list: 未来天数列表，如[1, 2, 3, 5, 10]
            all_data: 包含所有日期数据的DataFrame，如果提供则直接使用
            
        Returns:
            dict: 未来收益率字典，键为天数，值为收益率
        """
        try:
            # 将日期转换为datetime对象
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            # 如果未提供预加载的全部数据，则加载数据
            if all_data is None:
                # 确保交易日历已加载
                if self.trading_days is None:
                    self._load_trading_calendar()
                
                # 获取当前日期对应的交易日索引
                if date.strftime('%Y%m%d') not in self.trading_days:
                    return {}
                    
                current_idx = self.trading_days.index(date.strftime('%Y%m%d'))
                
                # 初始化结果字典
                future_returns = {}
                
                # 计算指定天数的未来收益率
                for days in days_list:
                    if current_idx + days < len(self.trading_days):
                        future_date = self.trading_days[current_idx + days]
                        
                        # 加载当前日期和未来日期的收盘价
                        current_features = self._load_sector_features(sector_code, date.strftime('%Y%m%d'))
                        future_features = self._load_sector_features(sector_code, future_date)
                        
                        if current_features is not None and future_features is not None:
                            # 获取收盘价
                            current_close = current_features.get('close', 0)
                            future_close = future_features.get('close', 0)
                            
                            # 计算收益率
                            if current_close > 0 and future_close > 0:
                                future_returns[days] = (future_close - current_close) / current_close
                
                return future_returns
            else:
                # 使用预加载的数据计算未来收益率
                # 确保数据索引是日期
                df = all_data.copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    else:
                        return {}
                
                # 确保当前日期存在于数据中
                if date not in df.index:
                    return {}
                
                # 获取当前收盘价
                current_close = df.loc[date, 'close']
                
                # 初始化结果字典
                future_returns = {}
                
                # 计算指定天数的未来收益率
                for days in days_list:
                    # 获取未来日期
                    future_date = None
                    
                    # 如果使用交易日历
                    if self.trading_days is not None:
                        try:
                            current_idx = self.trading_days.index(date.strftime('%Y%m%d'))
                            if current_idx + days < len(self.trading_days):
                                future_date_str = self.trading_days[current_idx + days]
                                future_date = pd.to_datetime(future_date_str)
                        except:
                            pass
                    
                    # 如果未找到有效的未来日期，使用日历日
                    if future_date is None:
                        future_date = date + pd.Timedelta(days=days)
                    
                    # 检查未来日期是否存在于数据中
                    if future_date in df.index:
                        future_close = df.loc[future_date, 'close']
                        
                        # 计算收益率
                        if current_close > 0 and future_close > 0:
                            future_returns[days] = (future_close - current_close) / current_close
                
                return future_returns
        except Exception as e:
            self.logger.error(f"计算板块 {sector_code} 日期 {date} 的未来收益率时出错: {e}")
            return {}

    def _enrich_with_historical_patterns(self, features_df, sector_code, date, seasonal_data=None):
        """
        用历史模式丰富特征
        
        Args:
            features_df: 特征DataFrame
            sector_code: 板块代码
            date: 当前日期
            seasonal_data: 预先计算好的季节性数据，如果为None，则重新计算
            
        Returns:
            DataFrame: 丰富后的特征
        """
        # 转换日期格式
        if isinstance(date, str):
            date_obj = pd.to_datetime(date)
        else:
            date_obj = date
            
        # 获取季节性和市场状态
        month = date_obj.month
        quarter = (month - 1) // 3 + 1
        
        # 获取市场状态
        market_status = self._load_market_status(date)
        market_type = "neutral"
        if market_status and 'market_type' in market_status:
            if market_status['market_type'] in [11, 12, 13, 21, 22, 23]:
                market_type = "bull"
            elif market_status['market_type'] in [41, 42, 43, 51, 52, 53]:
                market_type = "bear"
        
        # 如果没有预先计算的季节性数据，则从缓存加载或重新计算
        if seasonal_data is None:
            # 计算或加载板块的季节性模式
            patterns_cache_path = self.features_cache_dir / 'patterns' / f"{sector_code}_patterns.pkl"
            ensure_dir(self.features_cache_dir / 'patterns')
            
            patterns = None
            if patterns_cache_path.exists():
                # 检查缓存是否过期(30天更新一次)
                cache_time = datetime.fromtimestamp(os.path.getmtime(patterns_cache_path))
                days_diff = (datetime.now() - cache_time).days
                
                if days_diff < 30:
                    try:
                        patterns = pd.read_pickle(patterns_cache_path)
                    except Exception as e:
                        self.logger.warning(f"加载季节性模式缓存时出错: {e}")
            
            if patterns is None:
                patterns = self._compute_seasonal_patterns(sector_code)
                try:
                    with open(patterns_cache_path, 'wb') as f:
                        pickle.dump(patterns, f)
                except Exception as e:
                    self.logger.warning(f"保存季节性模式缓存时出错: {e}")
        else:
            # 使用传入的预先计算好的季节性数据
            patterns = seasonal_data
        
        # 添加季节性特征
        if patterns and 'monthly_performance' in patterns and month in patterns['monthly_performance']:
            monthly_perf = patterns['monthly_performance'][month]
            features_df['historical_month_return_5d'] = monthly_perf.get(5, 0)
            features_df['historical_month_return_10d'] = monthly_perf.get(10, 0)
            features_df['historical_month_return_20d'] = monthly_perf.get(20, 0)
            
            # 添加月份强弱排名特征
            month_rankings = {}
            for m, perf in patterns['monthly_performance'].items():
                if 5 in perf:
                    month_rankings[m] = perf[5]
            
            sorted_months = sorted(month_rankings.items(), key=lambda x: x[1], reverse=True)
            month_rank = 0
            for i, (m, _) in enumerate(sorted_months):
                if m == month:
                    month_rank = i + 1
                    break
            
            # 月份表现排名（1-12，1表示历史上该月份表现最好）
            features_df['historical_month_rank'] = month_rank
            # 月份表现百分比排名（0-1，越接近1表示该月份历史表现越好）
            features_df['historical_month_rank_pct'] = 1 - (month_rank - 1) / max(1, len(sorted_months) - 1) if month_rank > 0 else 0
            features_df['historical_month_return_5d'] = 0
            features_df['historical_month_return_10d'] = 0
            features_df['historical_month_return_20d'] = 0
            features_df['historical_month_rank'] = 0
            features_df['historical_month_rank_pct'] = 0
            
        if patterns and 'quarterly_performance' in patterns and quarter in patterns['quarterly_performance']:
            quarterly_perf = patterns['quarterly_performance'][quarter]
            features_df['historical_quarter_return_5d'] = quarterly_perf.get(5, 0)
            features_df['historical_quarter_return_10d'] = quarterly_perf.get(10, 0)
            features_df['historical_quarter_return_20d'] = quarterly_perf.get(20, 0)
            
            # 添加季度强弱排名特征
            quarter_rankings = {}
            for q, perf in patterns['quarterly_performance'].items():
                if 5 in perf:
                    quarter_rankings[q] = perf[5]
            
            sorted_quarters = sorted(quarter_rankings.items(), key=lambda x: x[1], reverse=True)
            quarter_rank = 0
            for i, (q, _) in enumerate(sorted_quarters):
                if q == quarter:
                    quarter_rank = i + 1
                    break
            
            # 季度表现排名（1-4，1表示历史上该季度表现最好）
            features_df['historical_quarter_rank'] = quarter_rank
            # 季度表现百分比排名（0-1，越接近1表示该季度历史表现越好）
            features_df['historical_quarter_rank_pct'] = 1 - (quarter_rank - 1) / max(1, len(sorted_quarters) - 1) if quarter_rank > 0 else 0
            features_df['historical_quarter_return_5d'] = 0
            features_df['historical_quarter_return_10d'] = 0
            features_df['historical_quarter_return_20d'] = 0
            features_df['historical_quarter_rank'] = 0
            features_df['historical_quarter_rank_pct'] = 0
            
        if patterns and 'market_correlation' in patterns and market_type in patterns['market_correlation']:
            market_perf = patterns['market_correlation'][market_type]
            features_df['historical_market_return_5d'] = market_perf.get(5, 0)
            features_df['historical_market_return_10d'] = market_perf.get(10, 0)
            features_df['historical_market_return_20d'] = market_perf.get(20, 0)
            
            # 添加市场环境适应性分数
            market_types = ["bull", "bear", "neutral"]
            market_avg_returns = {}
            for mt in market_types:
                if mt in patterns['market_correlation'] and 5 in patterns['market_correlation'][mt]:
                    market_avg_returns[mt] = patterns['market_correlation'][mt][5]
            
            # 当前市场环境的适应性分数 (与其他市场环境对比)
            if market_type in market_avg_returns and len(market_avg_returns) > 1:
                current_return = market_avg_returns[market_type]
                other_returns = [v for k, v in market_avg_returns.items() if k != market_type]
                avg_other_return = sum(other_returns) / len(other_returns)
                features_df['market_adaptation_score'] = current_return - avg_other_return
        else:
            features_df['historical_market_return_5d'] = 0
            features_df['historical_market_return_10d'] = 0
            features_df['historical_market_return_20d'] = 0
            features_df['market_adaptation_score'] = 0
            
        # 添加年度特殊月份标记
        is_month_start = date_obj.day <= 5
        is_month_end = date_obj.day >= 25
        is_quarter_start = is_month_start and month in [1, 4, 7, 10]
        is_quarter_end = is_month_end and month in [3, 6, 9, 12]
        is_year_start = is_month_start and month == 1
        is_year_end = is_month_end and month == 12
        
        features_df['is_month_start'] = 1 if is_month_start else 0
        features_df['is_month_end'] = 1 if is_month_end else 0
        features_df['is_quarter_start'] = 1 if is_quarter_start else 0
        features_df['is_quarter_end'] = 1 if is_quarter_end else 0
        features_df['is_year_start'] = 1 if is_year_start else 0
        features_df['is_year_end'] = 1 if is_year_end else 0
        
        return features_df

    def _evaluate_on_test_set(self, test_df, model_package):
        """在测试集上评估模型性能
        
        Args:
            test_df: 测试数据集
            model_package: 训练好的模型包含所有组件
            
        Returns:
            评估指标字典
        """
        self.logger.info(f"在 {len(test_df)} 个测试样本上评估模型性能")
        
        # 提取模型组件
        feature_columns = model_package['feature_columns']
        xgb_model = model_package['xgb_model']
        lgb_model = model_package['lgb_model']
        lnn_model = model_package['lnn_model']
        scaler = model_package['scaler']
        weights = model_package['weights']
        
        # 准备测试数据
        X_test = test_df[feature_columns].select_dtypes(include=np.number).fillna(0)
        y_test = test_df['is_top_performer']
        
        # 存储预测结果
        predictions = []
        
        # 按日期对测试数据进行分组
        date_groups = test_df.groupby('date')
        
        # 跟踪每个日期的预测性能
        daily_metrics = {}
        
        # 整体板块预测与实际情况
        all_sector_predictions = []
        top_n_hits = []
        
        # 处理每个日期
        for date, group in date_groups:
            self.logger.info(f"评估日期: {date}")
            
            # 获取当日特征数据
            X_date = group[feature_columns].select_dtypes(include=np.number).fillna(0)
            
            # 1. 市场环境分类预测
            market_cols = [col for col in X_date.columns if 'market_' in col or 'trend_' in col]
            if market_cols and lgb_model is not None:
                market_X = X_date[market_cols].fillna(0)
                market_pred = lgb_model.predict_proba(market_X)
                
                # 添加市场预测结果作为特征
                for i in range(market_pred.shape[1]):
                    X_date[f'market_prob_{i}'] = market_pred[:, i]
            
            # 2. 板块相对强度预测
            strength_cols = [col for col in X_date.columns if 
                            'momentum' in col or 'relative_' in col or 
                            'strength' in col or 'market_prob_' in col]
            if strength_cols and xgb_model is not None:
                strength_X = X_date[strength_cols].fillna(0)
                strength_pred = xgb_model.predict(strength_X)
                X_date['predicted_strength'] = strength_pred
            else:
                strength_pred = np.zeros(len(X_date))
            
            # 3. LNN最终预测
            final_pred = None
            if lnn_model is not None and scaler is not None:
                try:
                    X_scaled = scaler.transform(X_date.select_dtypes(include=np.number).fillna(0))
                    final_pred = lnn_model.predict(X_scaled).flatten()
                except Exception as e:
                    self.logger.error(f"LNN预测出错: {e}")
                    final_pred = np.zeros(len(X_date))
            else:
                final_pred = np.zeros(len(X_date))
            
            # 4. 融合预测结果
            ensemble_pred = np.zeros(len(X_date))
            if len(weights) == 3:  # XGB, LGB, LNN权重
                for i in range(len(X_date)):
                    ensemble_pred[i] = (weights[0] * strength_pred[i] + 
                                        weights[2] * final_pred[i])
            else:
                ensemble_pred = strength_pred  # 默认只使用XGBoost
            
            # 收集每个板块的预测
            date_predictions = []
            for i, (idx, row) in enumerate(group.iterrows()):
                sector_code = row['sector_code']
                actual = row['is_top_performer']
                future_return = row.get('future_return', 0)
                
                date_predictions.append({
                    'sector_code': sector_code,
                    'prediction': float(ensemble_pred[i]),
                    'actual': int(actual),
                    'future_return': float(future_return)
                })
                
                # 添加到整体预测列表
                all_sector_predictions.append({
                    'date': date,
                    'sector_code': sector_code,
                    'prediction': float(ensemble_pred[i]),
                    'actual': int(actual)
                })
            
            # 排序预测，获取预测得分最高的板块
            sorted_predictions = sorted(date_predictions, key=lambda x: x['prediction'], reverse=True)
            
            # 获取实际表现最好的板块
            actual_top = [p['sector_code'] for p in sorted(date_predictions, 
                                                        key=lambda x: x['actual'], 
                                                        reverse=True) if p['actual'] == 1]
            
            # 获取预测的前5名板块
            pred_top_5 = [p['sector_code'] for p in sorted_predictions[:5]]
            
            # 计算Top-N准确率 - 预测的前N名中有多少是实际前N名
            overlap = len(set(pred_top_5) & set(actual_top))
            top_n_accuracy = overlap / 5 if pred_top_5 and actual_top else 0
            
            # 记录命中情况
            top_n_hits.append(top_n_accuracy)
            
            # 记录当天的评估指标
            daily_metrics[str(date)] = {
                'predicted_top5': pred_top_5,
                'actual_top': actual_top,
                'overlap': overlap,
                'top_n_accuracy': top_n_accuracy
            }
        
        # 计算整体评估指标
        if all_sector_predictions:
            df_pred = pd.DataFrame(all_sector_predictions)
            
            # 计算预测分数阈值（使得正例比例与实际相同）
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            
            # 总体ROC AUC
            try:
                roc_auc = roc_auc_score(df_pred['actual'], df_pred['prediction'])
            except:
                roc_auc = 0
            
            # 计算选择前5名板块的平均命中率
            avg_top_n_accuracy = sum(top_n_hits) / len(top_n_hits) if top_n_hits else 0
            
            # 按日期统计正确预测率
            dates_count = len(daily_metrics)
            dates_with_hit = sum(1 for date_metric in daily_metrics.values() if date_metric['overlap'] > 0)
            date_hit_rate = dates_with_hit / dates_count if dates_count > 0 else 0
            
            # 计算日均命中数量
            avg_daily_hits = sum(metric['overlap'] for metric in daily_metrics.values()) / dates_count if dates_count > 0 else 0
            roc_auc = 0
            avg_top_n_accuracy = 0
            date_hit_rate = 0
            avg_daily_hits = 0
        
        # 创建具有可视化的评估指标
        evaluation_results = {
            'overall': {
                'roc_auc': float(roc_auc),
                'avg_top_n_accuracy': float(avg_top_n_accuracy),
                'date_hit_rate': float(date_hit_rate),
                'avg_daily_hits': float(avg_daily_hits),
                'test_dates_count': len(daily_metrics),
                'test_samples_count': len(test_df)
            },
            'daily_metrics': daily_metrics
        }
        
        # 控制台输出关键指标，便于直观查看
        self.logger.info("\n========== 测试集评估结果 ==========")
        self.logger.info(f"测试日期数量: {len(daily_metrics)}")
        self.logger.info(f"测试样本数量: {len(test_df)}")
        self.logger.info(f"ROC AUC: {roc_auc:.4f}")
        self.logger.info(f"平均Top-5命中率: {avg_top_n_accuracy:.4f}")
        self.logger.info(f"有命中的日期比例: {date_hit_rate:.4f}")
        self.logger.info(f"日均命中板块数: {avg_daily_hits:.2f}")
        self.logger.info("======================================")
        
        # 输出按命中率排序的前10个日期
        top_dates = sorted(daily_metrics.items(), key=lambda x: x[1]['top_n_accuracy'], reverse=True)[:10]
        self.logger.info("\n前10个预测效果最好的日期:")
        for date, metrics in top_dates:
            self.logger.info(f"日期: {date}, 命中数: {metrics['overlap']}/5, 命中率: {metrics['top_n_accuracy']:.2f}")
            self.logger.info(f"  预测前5: {', '.join(metrics['predicted_top5'])}")
            self.logger.info(f"  实际前5: {', '.join(metrics['actual_top'])}")
        
        return evaluation_results


def predict_sector_rotation(model_dir, date, sector_list=None, forward_days=2, top_n=5):
    """预测板块轮动
        
        Args:
            date: 日期
            sector_list: 板块列表，默认为None，表示使用全部板块
            forward_days: 预测未来天数
            top_n: 返回前N个板块
            
        Returns:
            预测结果字典
    """
    rotation_model_dir = model_dir / "sector_rotation"
    # 确保不会出现重复的sector_rotation路径
    if not rotation_model_dir.exists():
        logger.warning(f"板块轮动模型目录 {rotation_model_dir} 不存在")
        return None
    
    # 初始化训练器
    trainer = SectorModelTrainer(model_dir=rotation_model_dir)
    
    # 如果sector_list为空，尝试获取所有可用板块
    if not sector_list:
        sector_list = []
        try:
            from stock_sectors_mapping import get_tng_sectors, get_sector_stocks
            sector_list = get_tng_sectors()
        except Exception as e:
            logger.error(f"获取板块列表时出错: {e}")
            logger.error(traceback.format_exc())
    
    if not sector_list:
        logger.error("未能获取到任何板块数据")
        return None
    
    # 加载当前日期的板块特征
    current_features = trainer._load_multiple_sectors_features(sector_list, date)
    
    # 调试：打印加载的特征
    if current_features is None:
        logger.warning(f"日期 {date} 无法加载板块特征")
        return None
    
    logger.info(f"加载的特征类型: {type(current_features)}")
    logger.info(f"加载的板块数量: {len(current_features) if isinstance(current_features, dict) else 'N/A'}")
    
    if isinstance(current_features, dict) and len(current_features) > 1:
        logger.info(f"current_features包含多个板块的特征，将使用所有板块进行预测")
    
    # 加载市场状态特征 - 所有板块共享
    market_status = trainer._load_market_status(date)
    market_features = {}
    
    if market_status:
        # 提取市场环境特征到单独的字典
        market_features['market_type'] = market_status.get('market_type', '32')  # 默认为中性盘整
        market_features['market_cycle'] = market_status.get('market_cycle', '2')  # 默认为上升期
        market_features['trend_strength'] = market_status.get('trend_strength', 0.0)
        
        # 处理个股指标 - 使用创业板指数作为市场代表
        if 'individual_status' in market_status and '399006.SZ' in market_status['individual_status']:
            cyb_data = market_status['individual_status']['399006.SZ']
            
            # 市场评分指标
            market_features['trend_score'] = cyb_data.get('trend_score', 0.0)
            market_features['position_score'] = cyb_data.get('position_score', 0.0)
            market_features['volume_score'] = cyb_data.get('volume_score', 0.0)
            market_features['momentum_score'] = cyb_data.get('momentum_score', 0.0)
            market_features['volatility_score'] = cyb_data.get('volatility_score', 0.0)
            market_features['structure_score'] = cyb_data.get('structure_score', 0.0)
            market_features['cycle_score'] = cyb_data.get('cycle_score', 0.0)
            market_features['total_score'] = cyb_data.get('total_score', 0.0)
    
    # 添加当前日期
    market_features['current_date'] = date
    
    # 尝试加载模型包
    try:
        model_file = os.path.join(rotation_model_dir, "model.pkl")
        if not os.path.exists(model_file):
            logger.error(f"模型文件不存在: {model_file}")
            return None
            
        with open(model_file, "rb") as f:
            model_package = pickle.load(f)
            
        feature_columns = model_package.get('feature_columns', [])
        xgb_model = model_package.get('xgb_model')
        lgb_model = model_package.get('lgb_model')
        lnn_model = model_package.get('lnn_model')
        scaler = model_package.get('scaler')
        weights = model_package.get('weights', [0.4, 0.4, 0.2])
            
    except Exception as e:
        logger.error(f"加载模型包时出错: {e}")
        logger.error(traceback.format_exc())
        return None
    
    # 处理每个板块的特征并生成预测
    all_sector_predictions = {}
    all_sector_scores = []
    
    # 将所有板块特征统一到一个DataFrame
    all_features_df = []
    
    # 批量特征处理 - 优化版
    if isinstance(current_features, dict):
        for sector_code, sector_features in current_features.items():
            try:
                # 合并板块特征和市场特征
                features = sector_features.copy()
                if isinstance(features, dict):
                    # 添加市场特征
                    for k, v in market_features.items():
                        if k not in features:
                            features[k] = v
                    
                    # 将字典转换为DataFrame
                    features_df = pd.DataFrame([features])
                    # 添加板块代码
                    features_df['sector_code'] = sector_code
                    
                    # 添加到合并列表
                    all_features_df.append(features_df)
            except Exception as e:
                logger.error(f"处理板块 {sector_code} 特征时出错: {e}")
                continue
    
    # 合并所有板块特征
    if not all_features_df:
        logger.error("没有有效的板块特征数据")
        return None

    # 合并为一个DataFrame
    combined_df = pd.concat(all_features_df, ignore_index=True)
    
    # 确保所有需要的特征列都存在
    for col in feature_columns:
        if col not in combined_df.columns:
            combined_df[col] = 0
    
    # 选择模型所需的特征列
    X = combined_df[feature_columns].fillna(0)
    
    # 检查NaN和无穷值
    if X.isna().any().any() or np.isinf(X.values).any():
        logger.warning(f"特征数据中包含NaN或无穷值，将进行填充")
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
    
    # 实现分层预测
    try:
        # 1. 使用市场环境模型预测
        if lgb_model is not None:
            market_cols = [col for col in feature_columns if 'market' in col or 'trend' in col]
            market_X = X[market_cols].fillna(0)
            market_pred = lgb_model.predict_proba(market_X)
            
            # 添加市场预测结果到特征
            for i in range(market_pred.shape[1]):
                X[f'market_prob_{i}'] = market_pred[:, i]
        
        # 2. 使用相对强度模型预测
        strength_cols = [col for col in X.columns if 
                          'momentum' in col or 'relative_' in col or 
                          'strength' in col or 'market_prob_' in col]
        strength_X = X[strength_cols].fillna(0)
        
        if xgb_model is not None:
            strength_pred = xgb_model.predict(strength_X)
            X['predicted_strength'] = strength_pred
            
        # 3. 使用液体神经网络进行最终预测
        final_pred = None
        
        if lnn_model is not None and scaler is not None:
            # 标准化特征
            try:
                X_scaled = scaler.transform(X.select_dtypes(include=np.number).fillna(0))
                X_tensor = torch.FloatTensor(X_scaled)
                
                # LNN预测
                lnn_pred = lnn_model.predict(X_scaled)
                final_pred = lnn_pred.flatten()
            except Exception as e:
                logger.error(f"LNN预测出错: {e}")
                logger.error(traceback.format_exc())
        
        # 如果LNN预测失败，使用XGBoost预测
        if final_pred is None and xgb_model is not None:
            final_pred = strength_pred
            
        # 合并预测结果到原始DataFrame
        combined_df['prediction_score'] = final_pred
        
        # 按板块和预测分数分组
        predictions_by_sector = combined_df.groupby('sector_code')['prediction_score'].mean().reset_index()
        
        # 对板块按预测分数排序
        sorted_sectors = predictions_by_sector.sort_values('prediction_score', ascending=False)
        
        # 获取前N个板块
        top_sectors = sorted_sectors.head(top_n)
        
        # 构建结果字典
        results = {
            'date': date,
            'top_sectors': top_sectors['sector_code'].tolist(),
            'top_scores': top_sectors['prediction_score'].tolist(),
            'all_sectors': sorted_sectors['sector_code'].tolist(),
            'all_scores': sorted_sectors['prediction_score'].tolist(),
            'market_status': market_status
        }
        
        # 适应市场状态的板块选择
        if hasattr(trainer, '_adapt_sectors_to_market'):
            market_type = int(market_features.get('market_type', 32))
            sector_tuples = [(code, score) for code, score in zip(top_sectors['sector_code'], top_sectors['prediction_score'])]
            adapted_sectors = trainer._adapt_sectors_to_market(sector_tuples, market_type, market_status)
            
            results['adapted_sectors'] = [code for code, _ in adapted_sectors]
            results['adapted_scores'] = [score for _, score in adapted_sectors]
        
        logger.info(f"预测结果: 日期 {date} 未来 {forward_days} 天表现最好的前 {top_n} 个板块:")
        for i, sector_code in enumerate(results['top_sectors']):
            logger.info(f"  {i+1}. {sector_code}: 得分 {results['top_scores'][i]:.6f}")
        
        if 'adapted_sectors' in results:
            logger.info(f"适应市场状态后的板块选择:")
            for i, sector_code in enumerate(results['adapted_sectors']):
                logger.info(f"  {i+1}. {sector_code}: 得分 {results['adapted_scores'][i]:.6f}")
        
        return results
        
    except Exception as e:
        logger.error(f"预测时出错: {e}")
        logger.error(traceback.format_exc())
        return None


def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="板块轮动模型训练与预测")
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'backtest'], 
                      help='操作模式：train (训练模型), predict (预测), backtest (回测)')
    parser.add_argument('--sector', type=str, help='板块代码，用于训练单个板块模型')
    parser.add_argument('--start_date', type=str, help='开始日期，格式为YYYYMMDD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD')
    parser.add_argument('--date', type=str, help='预测日期，格式为YYYYMMDD')
    parser.add_argument('--top_n', type=int, default=5, help='选择前N个板块')
    parser.add_argument('--max_workers', type=int, help='并行训练的进程数')
    parser.add_argument('--use_lnn', action='store_true', help='是否使用液体神经网络模型')
    parser.add_argument('--sector_list', type=str, help='板块列表文件路径，用于板块轮动模型训练')
    parser.add_argument('--model_dir', type=str, help='模型保存目录')
    parser.add_argument('--model_version', type=str, default='latest', help='用于预测或回测的模型版本，默认使用最新版本')
    parser.add_argument('--save_recent_days', type=int, default=0, help='保存最近N个交易日的模型，用于回测，默认为0表示只保存最终模型')
    
    args = parser.parse_args()
    
    # 创建SectorModelTrainer实例
    trainer = SectorModelTrainer(model_dir=args.model_dir, use_lnn=args.use_lnn)
    
    # 加载板块列表
    sector_list = []
    try:
        from stock_sectors_mapping import get_tng_sectors
    except Exception as e:
        logger.error(f"获取板块列表时出错: {e}")
        logger.error(traceback.format_exc())
    sector_list = get_tng_sectors()
    # 运行模式
    if args.mode == 'train':
        # 训练板块轮动模型
        if not sector_list:
            print("错误: 训练板块轮动模型需要提供板块列表")
            return
                    
        models = trainer.train_rotation_model(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
            use_lnn=args.use_lnn,
            save_recent_days=args.save_recent_days
            )
        print(f"板块轮动模型训练结果: {'成功' if models else '失败'}")
            
    
    elif args.mode == 'predict':
        # 预测模式
        if not args.date:
            args.date = datetime.now().strftime('%Y%m%d')
            print(f"未指定预测日期，使用当前日期: {args.date}")
            
        # 预测特定日期的板块排名
        result = predict_sector_rotation(
            trainer.model_dir,
            args.date,
            sector_list=sector_list,
            forward_days=2,
            top_n=args.top_n
        )
        
        if result:
            print(f"\n预测日期: {args.date}")
            print(f"市场环境: {result.get('market_type', '未知')}")
            print("\n推荐板块:")
            for i, (sector, score) in enumerate(result.get('ranked_sectors', [])[:args.top_n]):
                print(f"{i+1}. {sector}: {score:.4f}")
        else:
            print(f"预测失败，未能获取有效结果")
    
    elif args.mode == 'backtest':
        # 回测模式
        if not sector_list:
            print("错误: 回测需要提供板块列表")
            return
            
        if not args.start_date or not args.end_date:
            print("错误: 回测需要指定--start_date和--end_date参数")
            return
            
        backtest_results = trainer.backtest_rotation_model(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
            model_version=args.model_version,
            top_n=args.top_n
        )
        
        if backtest_results:
            metrics = backtest_results['performance_metrics']
            print("\n=== 回测结果 ===")
            print(f"回测区间: {args.start_date} 至 {args.end_date}")
            print(f"交易天数: {metrics['total_trading_days']}")
            print(f"总收益率: {metrics['total_return']:.2%}")
            print(f"年化收益率: {metrics['annual_return']:.2%}")
            print(f"最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"胜率: {metrics['win_rate']:.2%}")
            print(f"日均收益: {metrics['avg_daily_return']:.4%}")
            print(f"年化波动率: {metrics['volatility']:.2%}")
            
            # 打印板块选择频率前10
            top_sectors = sorted(
                backtest_results['sector_selection_count'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            print("\n最常选择的板块:")
            for sector, count in top_sectors:
                print(f"{sector}: {count}次")
                
            # 保存回测结果
            try:
                result_dir = Path(get_data_path()) / "results" / "backtest"
                ensure_dir(result_dir)
                result_file = result_dir / f"rotation_backtest_{args.start_date}_to_{args.end_date}.pkl"
                
                with open(result_file, 'wb') as f:
                    pickle.dump(backtest_results, f)
                print(f"\n回测结果已保存到: {result_file}")
                
                # 绘制净值曲线
                try:
                    import matplotlib.pyplot as plt
                    
                    portfolio_values = pd.Series(backtest_results['portfolio_values'])
                    portfolio_values = portfolio_values / portfolio_values.iloc[0]  # 归一化
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(portfolio_values.index, portfolio_values.values)
                    plt.title('板块轮动策略净值曲线')
                    plt.xlabel('日期')
                    plt.ylabel('净值')
                    plt.grid(True)
                    
                    chart_file = result_dir / f"rotation_backtest_{args.start_date}_to_{args.end_date}.png"
                    plt.savefig(chart_file)
                    print(f"净值曲线图已保存到: {chart_file}")
                except Exception as e:
                    print(f"绘制净值曲线失败: {e}")
            except Exception as e:
                print(f"保存回测结果失败: {e}")
        else:
            print("回测失败，未能获取有效结果")
    
    else:
        print("请指定有效的运行模式: --mode train|predict|backtest")


if __name__ == "__main__":
    main() 