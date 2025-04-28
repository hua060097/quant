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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    
    def __init__(self, model_dir=None, use_lnn=False):
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
        
        self.logger.info(f"板块模型训练器初始化完成，模型目录: {self.model_dir}")

    def _load_trading_calendar(self):
        """加载交易日历"""
        try:
            # 尝试从ParquetDataStorage中获取
            sys.path.append(str(get_project_root() / "to"))
            from advanced_quant_data import ParquetDataStorage
            
            storage = ParquetDataStorage()
            
            # 获取较长时间范围的交易日历，但不需要太早的数据
            # 只需要包含训练日期和回溯期即可，'20220101'足够早
            cal_df = storage.get_trading_calendar('20220101', '20250331')
            
            if cal_df is not None and not cal_df.empty:
                if 'date' in cal_df.columns:
                    self.trading_days = cal_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    self.trading_days = cal_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
                    
                self.logger.info(f"成功加载交易日历，共 {len(self.trading_days)} 个交易日，范围: {self.trading_days[0]} - {self.trading_days[-1]}")
            else:
                self.logger.warning("获取交易日历失败")
        except Exception as e:
            self.logger.error(f"加载交易日历出错: {e}")
            self.logger.error(traceback.format_exc())
    
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

    def _prepare_training_data_enhanced(self, sector_code, start_date, end_date):
        """增强版训练数据准备，优化特征组合并支持回测
        
        Args:
            sector_code: 板块代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            X: 特征矩阵
            y: 标签向量
            weights: 样本权重
        """
        # 基本日期处理
        start_idx = self.trading_days.index(start_date) if start_date in self.trading_days else 0
        end_idx = self.trading_days.index(end_date) if end_date in self.trading_days else -1
        
        if start_idx > end_idx or end_idx == -1:
            self.logger.warning(f"日期范围 {start_date} - {end_date} 无效")
            return None, None, None
            
        trading_range = self.trading_days[start_idx:end_idx+1]
        
        # 收集所有日期的特征
        all_features = []
        feature_dates = []  # 记录特征生成的日期，用于回测
        self.logger.info(f"为板块 {sector_code} 构建 {len(trading_range)} 天的特征数据")
        
        # 增加进度跟踪
        processed = 0
        total = len(trading_range)
        
        for date in trading_range:
            # 显示进度
            processed += 1
            if processed % 20 == 0 or processed == total:
                self.logger.info(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")
            
            # 短期特征
            short_features = self._build_features_for_date_range(
                sector_code, date, 
                lookback_days=self.lookback_short, 
                forward_days=self.forward_days
            )
            
            # 长期特征
            long_features = self._build_features_for_date_range(
                sector_code, date, 
                lookback_days=self.lookback_long, 
                forward_days=self.forward_days
            )
            
            # 合并特征与标记日期
            if short_features is not None and long_features is not None:
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
                
                if not merged_features.empty:
                    # 添加回测需要的信息
                    merged_features['prediction_date'] = date  # 特征生成日期
                    if 'label_date' in merged_features.columns:
                        merged_features['target_date'] = merged_features['label_date']  # 预测目标日期
                    elif 'label_date' in short_features.columns:
                        merged_features['target_date'] = short_features['label_date']
                    
                    # 添加市场环境特征
                    market_status = self._load_market_status(date)
                    if market_status and 'market_type' in market_status:
                        merged_features['market_type'] = market_status['market_type']
                        merged_features['market_cycle'] = market_status.get('market_cycle', 0)
                        merged_features['trend_strength'] = market_status.get('trend_strength', 0)
                    
                    # 动态特征工程 - 基于市场环境
                    self._add_dynamic_features(merged_features, market_status)
                    
                    feature_dates.append(date)
                    all_features.append(merged_features)
        
        # 合并数据
        if all_features:
            training_df = pd.concat(all_features, ignore_index=True)
            
            # 保存回测元数据
            training_df['sector_code'] = sector_code
            training_df['is_training_data'] = True
            
            # 准备特征与标签
            label_col = 'future_return'
            if label_col not in training_df.columns and 'future_return_x' in training_df.columns:
                training_df[label_col] = training_df['future_return_x']
            
            # 排除不作为特征的列
            exclude_cols = ['date', 'feature_date', 'label_date', 'future_return', 
                            'future_return_x', 'future_return_y', 'prediction_date', 
                            'target_date', 'sector_code', 'is_training_data']
            
            feature_cols = [c for c in training_df.columns if c not in exclude_cols]
            
            # 添加时间衰减权重
            training_df['time_weight'] = self._calculate_time_decay_weights(training_df)
            
            X = training_df[feature_cols].copy()
            y = training_df[label_col].copy()
            weights = training_df['time_weight'].copy()
            
            # 处理缺失值
            X = X.fillna(0)
            
            # 保存完整DataFrame以用于回测
            self.feature_columns = feature_cols  # 保存特征列名
            self.backtest_features = training_df  # 保存完整数据用于回测
            
            return X, y, weights
        else:
            return None, None, None

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
    
    def train_sector_model(self, sector_code, start_date, end_date):
        """训练单个板块的预测模型
        
        使用增强版的数据准备函数处理数据
        
        Args:
            sector_code: 板块代码
            start_date: 训练开始日期
            end_date: 训练结束日期
            
        Returns:
            训练好的模型dict
        """
        # 确保板块目录没有重复部分
        if "sector_rotation" in sector_code and sector_code != "sector_rotation":
            sector_code = "sector_rotation"
            
        self.logger.info(f"开始训练板块 {sector_code} 的预测模型，日期范围: {start_date} - {end_date}")
        
        # 使用增强版数据准备
        X, y, weights = self._prepare_training_data_enhanced(sector_code, start_date, end_date)
        
        if X is None or y is None:
            self.logger.error(f"无法获取板块 {sector_code} 的训练数据")
            return None
        
        self.logger.info(f"成功获取 {len(X)} 条训练数据")
        
        # 特征标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 训练XGBoost模型
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.logger.info(f"开始训练XGBoost模型...")
        try:
            xgb_model.fit(X_scaled, y, sample_weight=weights)
            self.logger.info(f"XGBoost模型训练完成")
            
            # 获取特征重要性
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = importance.head(20)
            self.logger.info(f"Top 20 特征重要性:\n{top_features}")
            
            # 训练LightGBM模型
            lgb_model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.logger.info(f"开始训练LightGBM模型...")
            lgb_model.fit(X_scaled, y, sample_weight=weights)
            self.logger.info(f"LightGBM模型训练完成")
            
            # 返回模型和相关数据
            models = {
                'xgb': xgb_model,
                'lgb': lgb_model,
                'scaler': scaler,
                'feature_importance': importance.to_dict('records'),
                'feature_columns': X.columns.tolist()
            }
            
            # 保存模型
            self._save_models(sector_code, models, X.columns.tolist())
            
            return models
        except Exception as e:
            self.logger.error(f"训练模型时出错: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

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

    def train_all_sectors(self, sector_list, start_date, end_date, max_workers=None):
        """
        训练所有板块的模型
        
        Args:
            sector_list: 板块列表
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            max_workers: 最大进程数
            
        Returns:
            dict: 训练结果
        """
        self.logger.info(f"开始训练 {len(sector_list)} 个板块的模型，日期范围: {start_date} - {end_date}")
        
        results = {}
        
        # 判断是否使用多进程
        if max_workers and max_workers > 1 and len(sector_list) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for sector_code in sector_list:
                    future = executor.submit(self.train_sector_model, sector_code, start_date, end_date)
                    futures[future] = sector_code
                
                for future in as_completed(futures):
                    sector_code = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results[sector_code] = result
                            self.logger.info(f"板块 {sector_code} 模型训练完成")
                        else:
                            self.logger.warning(f"板块 {sector_code} 模型训练失败")
                    except Exception as e:
                        self.logger.error(f"处理板块 {sector_code} 的结果时出错: {e}")
        else:
            # 顺序训练
            for sector_code in sector_list:
                try:
                    result = self.train_sector_model(sector_code, start_date, end_date)
                    if result:
                        results[sector_code] = result
                        self.logger.info(f"板块 {sector_code} 模型训练完成")
                    else:
                        self.logger.warning(f"板块 {sector_code} 模型训练失败")
                except Exception as e:
                    self.logger.error(f"训练板块 {sector_code} 的模型时出错: {e}")
                    self.logger.error(traceback.format_exc())
                
                # 释放内存
                gc.collect()
        
        # 保存总结果
        if results:
            summary_file = self.model_dir / "training_summary.json"
            with open(summary_file, "w") as f:
                json.dump(results, f, indent=2)
                
            self.logger.info(f"训练完成，成功训练 {len(results)} 个板块的模型，结果保存在 {summary_file}")
        else:
            self.logger.warning("没有成功训练任何板块的模型")
            
        return results

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
        构建板块轮动特征数据集
        
        Args:
            sector_list: 板块列表
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            forward_days: 向前预测天数
            
        Returns:
            DataFrame: 轮动特征数据集
        """
        # 确保交易日历已加载
        if not hasattr(self, 'trading_days') or not self.trading_days:
            self._load_trading_calendar()
        
        # 将日期转换为字符串格式
        start_date = str(start_date)
        end_date = str(end_date)
        
        # 日期参数校验和日志
        if start_date not in self.trading_days:
            self.logger.warning(f"开始日期 {start_date} 不是交易日，将选择最近的下一个交易日")
            # 寻找大于start_date的最小交易日
            valid_days = [d for d in self.trading_days if d > start_date]
            if valid_days:
                start_date = min(valid_days)
                self.logger.info(f"选择 {start_date} 作为新的开始日期")
            else:
                self.logger.error(f"无法找到 {start_date} 之后的交易日")
                return None
        
        if end_date not in self.trading_days:
            self.logger.warning(f"结束日期 {end_date} 不是交易日，将选择最近的前一个交易日")
            # 寻找小于end_date的最大交易日
            valid_days = [d for d in self.trading_days if d < end_date]
            if valid_days:
                end_date = max(valid_days)
                self.logger.info(f"选择 {end_date} 作为新的结束日期")
            else:
                self.logger.error(f"无法找到 {end_date} 之前的交易日")
                return None
        
        # 获取交易日期范围
        trading_days = self.trading_days
        start_idx = trading_days.index(start_date)
        end_idx = trading_days.index(end_date)
        
        # 修正：计算实际需要的起始日期，考虑了lookback_long参数
        # 以确保我们只取需要的日期数据
        actual_start_idx = max(0, start_idx - self.lookback_long)
        actual_start_date = trading_days[actual_start_idx]
        
        date_range = trading_days[actual_start_idx:end_idx+1]
        self.logger.info(f"构建板块轮动特征：实际日期范围 {actual_start_date} - {end_date}，共 {len(date_range)} 天")
        
        # 用于存储所有日期的特征
        all_features = []
        
        # 记录处理进度
        total_dates = len(date_range)
        processed_dates = 0
        
        # 对每个日期构建特征
        for current_date in date_range:
            try:
                # 获取当前日期在交易日历中的索引
                current_idx = trading_days.index(current_date)
                
                # 确保有足够的未来交易日来计算标签
                if current_idx + forward_days >= len(trading_days):
                    self.logger.debug(f"日期 {current_date} 后没有足够的交易日计算标签，跳过")
                    continue
                
                # 获取标签日期
                label_date = trading_days[current_idx + forward_days]
                
                # 获取当前日期的所有板块特征
                sector_features = self._load_multiple_sectors_features(sector_list, current_date)
                if sector_features is None or len(sector_features) == 0:
                    self.logger.debug(f"日期 {current_date} 没有板块特征数据，跳过")
                    continue
                
                # 获取标签日期的所有板块特征用于计算标签
                label_features = self._load_multiple_sectors_features(sector_list, label_date)
                if label_features is None or len(label_features) == 0:
                    self.logger.debug(f"标签日期 {label_date} 没有板块特征数据，跳过")
                    continue
                
                # 计算标签 - 未来收益率和收益率排名
                for sector_name, features in sector_features.items():
                    if sector_name in label_features:
                        try:
                            # 创建特征副本，防止修改原始数据
                            features_copy = features.copy()
                            
                            # 添加当前日期和标签日期
                            features_copy['current_date'] = current_date
                            features_copy['label_date'] = label_date
                            
                            # 获取收盘价
                            current_close = float(features.get('close', features.get('收盘价', 0)))
                            label_close = float(label_features[sector_name].get('close', label_features[sector_name].get('收盘价', 0)))
                            
                            # 计算未来收益率
                            future_return = (label_close - current_close) / current_close if current_close > 0 else 0
                            features_copy['future_return'] = future_return
                            
                            # 计算成交量变化
                            current_amount = float(features.get('amount', features.get('成交额', 0)))
                            label_amount = float(label_features[sector_name].get('amount', label_features[sector_name].get('成交额', 0)))
                            amount_change = (label_amount - current_amount) / current_amount if current_amount > 0 else 0
                            features_copy['future_amount_change'] = amount_change
                            
                            # 将板块名称添加为特征
                            features_copy['sector_code'] = sector_name
                            
                            # 添加到特征列表
                            all_features.append(features_copy)
                        except Exception as e:
                            self.logger.warning(f"处理板块 {sector_name} 日期 {current_date} 的特征时出错: {e}")
                
                # 更新进度
                processed_dates += 1
                if processed_dates % 10 == 0 or processed_dates == total_dates:
                    self.logger.info(f"已处理 {processed_dates}/{total_dates} 天的数据 ({processed_dates/total_dates*100:.1f}%)")
            
            except Exception as e:
                self.logger.error(f"处理日期 {current_date} 时出错: {e}")
                self.logger.error(traceback.format_exc())
        
        if not all_features:
            self.logger.warning("未能构建任何特征数据")
            return None
            
        # 将所有特征合并为一个DataFrame
        try:
            # 首先检查所有特征的键是否一致
            all_keys = set()
            for feature_dict in all_features:
                all_keys.update(feature_dict.keys())
            
            # 确保每个特征字典都有所有的键，如果没有则补充为None
            standardized_features = []
            for feature_dict in all_features:
                standardized_dict = {key: feature_dict.get(key, None) for key in all_keys}
                standardized_features.append(standardized_dict)
            
            # 检查并打印特征字典的键
            if standardized_features:
                self.logger.debug(f"特征字典键样例: {list(standardized_features[0].keys())[:10]}...")
            
            # 使用标准化的特征字典创建DataFrame
            features_df = pd.DataFrame(standardized_features)
            
            # 检查DataFrame的列名
            self.logger.debug(f"DataFrame列名: {features_df.columns[:10]}...")
            
        except Exception as e:
            self.logger.error(f"将特征列表转换为DataFrame时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None
            
        # 计算未来收益率排名
        date_groups = features_df.groupby('current_date')
        
        # 用于存储带有排名的数据
        ranked_features = []
        
        for date, group in date_groups:
            try:
                # 计算收益率排名
                group['future_return_rank'] = group['future_return'].rank(ascending=False)
                group['future_return_rank_pct'] = group['future_return'].rank(pct=True, ascending=False)
                
                # 标记表现最好的板块
                top_n = 5  # 标记前5个板块
                group['is_top_performer'] = group['future_return_rank'] <= top_n
                
                ranked_features.append(group)
            except Exception as e:
                self.logger.warning(f"计算日期 {date} 的排名时出错: {e}")
        
        if not ranked_features:
            self.logger.warning("计算排名后无数据")
            return None
            
        # 合并所有排名后的特征
        result_df = pd.concat(ranked_features, ignore_index=True)
        
        # 筛选掉训练开始日期前的数据，因为我们仅收集它们用于构建回溯特征
        result_df = result_df[pd.to_datetime(result_df['current_date']) >= pd.to_datetime(start_date)]
        
        self.logger.info(f"成功构建板块轮动特征数据集，共 {len(result_df)} 条记录，涵盖 {result_df['current_date'].nunique()} 个交易日")
        
        return result_df

    def train_rotation_model(self, features_df, model_name="sector_rotation"):
        """训练板块轮动模型
        
        Args:
            features_df: 包含多个板块特征的DataFrame
            model_name: 模型名称，默认为"sector_rotation"
            
        Returns:
            训练好的模型字典
        """
        self.logger.info(f"开始训练板块轮动模型 {model_name}...")
        
        # 确保model_name不包含重复的sector_rotation
        if model_name == "sector_rotation/sector_rotation":
            model_name = "sector_rotation"
        elif model_name.endswith("/sector_rotation"):
            model_name = model_name[:-len("/sector_rotation")]
        
        if features_df is None or len(features_df) == 0:
            self.logger.error("特征数据集为空，无法训练模型")
            return None
            
        # 检查必要的特征和标签是否存在
        required_cols = ['future_return', 'is_top_performer']
        for col in required_cols:
            if col not in features_df.columns:
                self.logger.error(f"缺少必要的列 {col}，无法训练模型")
                return None
                
        # 打印数据集情况
        self.logger.info(f"数据集情况: 总数据量 {len(features_df)}, 日期数 {features_df['current_date'].nunique()}")
        self.logger.info(f"特征列: {list(features_df.columns)}")
        
        # 检查顶尖表现板块的比例
        top_ratio = features_df['is_top_performer'].mean()
        self.logger.info(f"顶尖表现板块的比例: {top_ratio:.2%}")
        
        # 过滤掉有缺失值的行
        has_missing = features_df.isnull().any(axis=1)
        if has_missing.any():
            missing_count = has_missing.sum()
            self.logger.warning(f"发现 {missing_count} 行数据包含缺失值，将被过滤")
            features_df = features_df[~has_missing].copy()
            
        if len(features_df) == 0:
            self.logger.error("过滤缺失值后数据集为空，无法训练模型")
            return None
            
        # 分离特征和标签
        # 排除非特征列
        exclude_cols = ['current_date', 'label_date', 'future_return', 'future_return_rank', 
                        'future_return_rank_pct', 'is_top_performer', 'sector_code', 'future_amount_change']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 检查特征列是否都是数值型
        non_numeric_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(features_df[col])]
        if non_numeric_cols:
            self.logger.warning(f"发现非数值型特征列: {non_numeric_cols}，将被排除")
            feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
            
        if not feature_cols:
            self.logger.error("没有有效的特征列，无法训练模型")
            return None
            
        self.logger.info(f"最终使用的特征列数量: {len(feature_cols)}")
        
        X = features_df[feature_cols].copy()
        y_reg = features_df['future_return'].copy()
        y_cls = features_df['is_top_performer'].copy()
        
        # 数据集划分
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X, y_reg, y_cls, test_size=0.2, random_state=42, shuffle=True
        )
        
        self.logger.info(f"训练集数量: {len(X_train)}, 测试集数量: {len(X_test)}")
        
        # 训练模型
        models = {}
        
        # 1. XGBoost回归器 - 预测未来收益率
        self.logger.info("开始训练XGBoost回归器...")
        try:
            xgb_reg = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                tree_method='hist',
                random_state=42,
                callbacks=[xgb.callback.EarlyStopping(
                    rounds=10,
                    metric_name='rmse',
                    data_name='validation_0'
                )]
            )
            
            # 使用eval_set参数而非早停回合
            eval_set = [(X_test, y_reg_test)]
            
            # 训练模型
            xgb_reg.fit(
                X_train, y_reg_train,
                eval_set=eval_set,
                verbose=False
            )
            
            y_reg_pred = xgb_reg.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
            r2 = r2_score(y_reg_test, y_reg_pred)
            self.logger.info(f"XGBoost回归器 - RMSE: {rmse:.6f}, R2: {r2:.6f}")
            
            models['xgb_regressor'] = xgb_reg
        except Exception as e:
            self.logger.error(f"训练XGBoost回归器时出错: {e}")
            self.logger.error(traceback.format_exc())
        
        # 2. XGBoost分类器 - 预测是否为顶尖表现板块
        self.logger.info("开始训练XGBoost分类器...")
        try:
            # 计算类别权重以应对不平衡数据（避免除以零）
            if np.sum(y_cls_train == 0) > 0 and np.sum(y_cls_train == 1) > 0:
                scale = np.sum(y_cls_train == 0) / np.sum(y_cls_train == 1)
            else:
                scale = 1.0  # 避免除以零
            
            # 创建XGBoost分类器
            xgb_clf = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=scale,
                random_state=42,
                use_label_encoder=False,
                eval_metric='auc',
                callbacks=[xgb.callback.EarlyStopping(
                    rounds=10,
                    metric_name='auc',
                    data_name='validation_0'
                )]
            )

            
            # 使用回调进行早停
            eval_set = [(X_test, y_cls_test)]
            
            # 训练模型
            xgb_clf.fit(
                X_train, 
                y_cls_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # 计算性能指标
            y_pred_clf = xgb_clf.predict(X_test)
            y_pred_prob_clf = xgb_clf.predict_proba(X_test)[:, 1]
            
            perf_clf = {
                'accuracy': accuracy_score(y_cls_test, y_pred_clf),
                'precision': precision_score(y_cls_test, y_pred_clf, zero_division=0),
                'recall': recall_score(y_cls_test, y_pred_clf, zero_division=0),
                'f1': f1_score(y_cls_test, y_pred_clf, zero_division=0),
                'auc': roc_auc_score(y_cls_test, y_pred_prob_clf) if len(np.unique(y_cls_test)) > 1 else 0
            }
            
            self.logger.info(f"XGBoost分类器性能: "
                            f"准确率={perf_clf['accuracy']:.4f}, "
                            f"精确率={perf_clf['precision']:.4f}, "
                            f"召回率={perf_clf['recall']:.4f}, "
                            f"F1={perf_clf['f1']:.4f}, "
                            f"AUC={perf_clf['auc']:.4f}")
            
            models['xgb_classifier'] = xgb_clf
        except Exception as e:
            self.logger.error(f"训练XGBoost分类器失败: {e}")
            self.logger.error(traceback.format_exc())
        
        # 3. LightGBM回归器
        self.logger.info("开始训练LightGBM回归器...")
        try:
            # 创建LightGBM回归器
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='regression'
            )
            
            # 准备评估集
            eval_set = [(X_test, y_reg_test)] if X_test is not None else None
            callbacks = [lgb.early_stopping(stopping_rounds=10)]
            
            # 训练模型
            lgb_reg.fit(
                X_train, 
                y_reg_train,
                eval_set=eval_set,
                callbacks=callbacks
                # 移除verbose参数
            )
            
            # 计算性能指标
            y_pred_lgb = lgb_reg.predict(X_test)
            mse_lgb = mean_squared_error(y_reg_test, y_pred_lgb)
            rmse_lgb = np.sqrt(mse_lgb)
            r2_lgb = r2_score(y_reg_test, y_pred_lgb)
            
            self.logger.info(f"LightGBM回归器性能: RMSE={rmse_lgb:.4f}, R2={r2_lgb:.4f}")
            
            models['lgb_regressor'] = lgb_reg
        except Exception as e:
            self.logger.error(f"训练LightGBM回归器失败: {e}")
            self.logger.error(traceback.format_exc())
        
        # 4. 神经网络回归器
        self.logger.info("开始训练神经网络回归器...")
        try:
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 保存scaler
            models['scaler'] = scaler
            
            # 构建神经网络
            input_size = X_train_scaled.shape[1]
            hidden_size = [128, 64]
            output_size = 1
            
            # 导入LiquidNetModel
            try:
                sys.path.append(str(get_project_root() / "to"))
                from Liquid.nn.liquid.liquid_network import LiquidNetModel
                
                # 添加一个更健壮的参数调整机制
                try:
                    # 首先尝试获取LiquidNetModel的构造函数参数
                    import inspect
                    model_params = inspect.signature(LiquidNetModel.__init__).parameters.keys()
                    
                    # 根据实际API参数构建参数字典
                    net_params = {}
                    if 'input_size' in model_params:
                        net_params['input_size'] = input_size
                    elif 'input_dim' in model_params:
                        net_params['input_dim'] = input_size
                        
                    if 'output_size' in model_params:
                        net_params['output_size'] = output_size
                    elif 'output_dim' in model_params:
                        net_params['output_dim'] = output_size
                        
                    if 'hidden_size' in model_params:
                        net_params['hidden_size'] = hidden_size[0] if isinstance(hidden_size, list) else hidden_size
                    elif 'hidden_dim' in model_params or 'hidden_dims' in model_params:
                        param_name = 'hidden_dim' if 'hidden_dim' in model_params else 'hidden_dims'
                        net_params[param_name] = hidden_size
                        
                    # 添加其他通用参数
                    net_params['num_layers'] = len(hidden_size) if isinstance(hidden_size, list) else 2
                    net_params['learning_rate'] = 0.0005  # 降低学习率
                    # 删除dropout参数 - LiquidNetModel不支持此参数
                    
                    # 使用参数字典初始化模型
                    net = LiquidNetModel(**net_params)
                    
                except Exception as param_error:
                    self.logger.warning(f"使用自动参数调整初始化LiquidNetModel失败: {param_error}")
                    
                    # 回退到标准参数初始化
                    net = LiquidNetModel(
                        input_size=input_size,
                        output_size=output_size,
                        hidden_size=hidden_size[0] if isinstance(hidden_size, list) else hidden_size,
                        num_layers=len(hidden_size) if isinstance(hidden_size, list) else 2,
                        learning_rate=0.0005
                        # 删除dropout参数 - LiquidNetModel不支持此参数
                    )
                
                # 使用LiquidNetModel的fit方法训练
                net.fit(
                    X_train_scaled, y_reg_train.values.reshape(-1, 1),
                    validation_data=(X_test_scaled, y_reg_test.values.reshape(-1, 1)),
                    batch_size=32,
                    epochs=100,
                    patience=15,
                    verbose=1
                    # 移除l2_reg参数 - 不支持
                )
                
                # 评估
                y_reg_pred = net.predict(X_test_scaled).flatten()
                
                rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
                r2 = r2_score(y_reg_test, y_reg_pred)
                self.logger.info(f"液体神经网络回归器 - RMSE: {rmse:.6f}, R2: {r2:.6f}")
                
                models['nn_regressor'] = net
            except Exception as e:
                self.logger.error(f"导入或初始化LiquidNetModel时出错: {e}")
                self.logger.error(traceback.format_exc())
                
                # 如果LiquidNetModel不可用，回退到标准PyTorch实现
                self.logger.info("尝试使用标准PyTorch神经网络...")
                
                class SimpleNN(nn.Module):
                    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
                        super(SimpleNN, self).__init__()
                        layers = []
                        prev_size = input_size
                        
                        for hidden_size in hidden_sizes:
                            layers.append(nn.Linear(prev_size, hidden_size))
                            layers.append(nn.ReLU())
                            layers.append(nn.Dropout(dropout_rate))
                            prev_size = hidden_size
                            
                        layers.append(nn.Linear(prev_size, output_size))
                        self.model = nn.Sequential(*layers)
                        
                    def forward(self, x):
                        return self.model(x)
                
                net = SimpleNN(input_size, hidden_size, output_size)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=0.001)
                
                # 训练循环
                batch_size = 64
                num_epochs = 50
                
                # 转换为PyTorch张量
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.FloatTensor(y_reg_train.values).reshape(-1, 1)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                y_test_tensor = torch.FloatTensor(y_reg_test.values).reshape(-1, 1)
                
                # 创建数据加载器
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # 训练循环
                best_loss = float('inf')
                patience = 10
                patience_counter = 0
                best_model = None
                
                for epoch in range(num_epochs):
                    net.train()
                    running_loss = 0.0
                    
                    for inputs, targets in train_loader:
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    
                    # 验证
                    net.eval()
                    with torch.no_grad():
                        test_outputs = net(X_test_tensor)
                        test_loss = criterion(test_outputs, y_test_tensor).item()
                    
                    if (epoch + 1) % 5 == 0:
                        self.logger.info(f"Epoch {epoch+1}/{num_epochs}, 训练损失: {running_loss/len(train_loader):.6f}, 验证损失: {test_loss:.6f}")
                    
                    # 早停
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_model = copy.deepcopy(net.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            self.logger.info(f"早停在 epoch {epoch+1}")
                            break
                
                # 恢复最佳模型
                if best_model is not None:
                    net.load_state_dict(best_model)
                
                # 评估
                net.eval()
                with torch.no_grad():
                    y_reg_pred = net(X_test_tensor).numpy().flatten()
                    
                rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
                r2 = r2_score(y_reg_test, y_reg_pred)
                self.logger.info(f"标准神经网络回归器 - RMSE: {rmse:.6f}, R2: {r2:.6f}")
                
                models['nn_regressor'] = net
        except Exception as e:
            self.logger.error(f"训练神经网络回归器时出错: {e}")
            self.logger.error(traceback.format_exc())
            
        # 5. 集成预测器
        self.logger.info("构建集成预测器...")
        try:
            # 训练堆叠模型
            if all(k in models for k in ['xgb_regressor', 'lgb_regressor']):
                # 获取每个基模型在验证集上的预测
                xgb_pred = models['xgb_regressor'].predict(X_test)
                lgb_pred = models['lgb_regressor'].predict(X_test)
                
                # 初始化nn_pred为None
                nn_pred = None
                
                # 如果神经网络模型也存在
                if 'nn_regressor' in models:
                    net = models['nn_regressor']
                    # 检查是否为PyTorch模型还是LiquidNetModel
                    if isinstance(net, torch.nn.Module):
                        # PyTorch模型
                        net.eval()
                        
                        # 确保X_test_tensor已定义
                        X_test_tensor = torch.FloatTensor(X_test_scaled)
                            
                        with torch.no_grad():
                            nn_pred = net(X_test_tensor).numpy().flatten()
                    else:
                        # LiquidNetModel
                        nn_pred = net.predict(X_test_scaled).flatten()
                
                # 确保nn_pred已定义
                if nn_pred is None:
                    # 如果没有神经网络预测，使用零向量
                    nn_pred = np.zeros_like(xgb_pred)
                
                # 创建堆叠特征
                stacking_features = np.column_stack([xgb_pred, lgb_pred, nn_pred])
                
                # 训练元模型
                meta_model = Ridge(alpha=1.0)
                meta_model.fit(stacking_features, y_reg_test)
                
                # 评估堆叠模型
                stacked_pred = meta_model.predict(stacking_features)
                rmse = np.sqrt(mean_squared_error(y_reg_test, stacked_pred))
                r2 = r2_score(y_reg_test, stacked_pred)
                self.logger.info(f"堆叠集成模型 - RMSE: {rmse:.6f}, R2: {r2:.6f}")
                
                models['stacking_model'] = meta_model
        except Exception as e:
            self.logger.error(f"构建集成预测器时出错: {e}")
        
        # 保存模型
        if models:
            # 包括特征列信息
            models['feature_columns'] = feature_cols
            
            # 包括缩放器
            if 'scaler' not in models:
                models['scaler'] = StandardScaler().fit(X)
            
            # 包括元数据
            models['metadata'] = {
                'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'top_performer_ratio': top_ratio
            }
            
            self.logger.info(f"成功训练板块轮动模型 {model_name}，共包含 {len(models) - 2} 个子模型")
            
            # 保存训练好的模型
            model_base_dir = self.model_dir
            print(f"模型保存路径: {model_base_dir}")
            ensure_dir(model_base_dir)
            print(f"模型保存路径: {model_base_dir}")
            
            # 保存特征列
            feature_file = model_base_dir / "feature_columns.json"
            with open(feature_file, "w") as f:
                json.dump(feature_cols, f)
                
            # 保存各个子模型
            if 'xgb_regressor' in models:
                model_file = model_base_dir / "xgb_return_model.json"
                models['xgb_regressor'].save_model(str(model_file))
                self.logger.info(f"XGBoost回归器已保存到 {model_file}")
                
            if 'xgb_classifier' in models:
                model_file = model_base_dir / "xgb_clf_model.json"
                models['xgb_classifier'].save_model(str(model_file))
                self.logger.info(f"XGBoost分类器已保存到 {model_file}")
                
            if 'lgb_regressor' in models:
                model_file = model_base_dir / "lgb_return_model.txt"
                models['lgb_regressor'].booster_.save_model(str(model_file))
                self.logger.info(f"LightGBM回归器已保存到 {model_file}")
                
            if 'nn_regressor' in models:
                try:
                    # 创建神经网络模型目录
                    print(f"创建神经网络模型目录: {model_base_dir / 'lnn_models'}")
                    model_dir = model_base_dir / "lnn_models"
                    ensure_dir(model_dir)
                    print(f"神经网络模型目录: {model_dir}")
                    
                    net = models['nn_regressor']
                    
                    # 判断是LiquidNetModel还是PyTorch模型
                    if hasattr(net, 'save'):
                        # LiquidNetModel有save方法，需要保存到一个文件而非目录
                        model_file = model_dir / "lnn_model.pt"
                        net.save(model_file)
                        self.logger.info(f"神经网络回归器已保存到 {model_file}")
                        
                        # 同时保存模型参数到JSON文件，方便未来加载
                        params_file = model_dir / "model_params.json"
                        with open(params_file, 'w') as f:
                            json.dump({
                                'input_size': net.input_size,
                                'output_size': net.output_size,
                                'hidden_size': net.hidden_size,
                                'num_layers': net.num_layers
                            }, f)
                        self.logger.info(f"保存了模型参数到 {params_file}")
                        
                        # 额外确保scaler也被保存（虽然LNN的save方法应该已经保存了）
                        if hasattr(net, 'scaler') and net.scaler is not None:
                            try:
                                import joblib
                                scaler_file = model_dir / "scaler.pkl"
                                joblib.dump(net.scaler, scaler_file)
                                self.logger.info(f"神经网络的StandardScaler已额外保存到 {scaler_file}")
                            except Exception as scaler_error:
                                self.logger.warning(f"保存神经网络的StandardScaler时出错: {scaler_error}")
                    else:
                        # PyTorch模型
                        model_file = model_dir / "model.pt"
                        torch.save(net.state_dict(), model_file)
                        # 保存一些额外信息
                        arch_file = model_dir / "architecture.json"
                        with open(arch_file, "w") as f:
                            json.dump({
                                "input_size": input_size,
                                "hidden_size": hidden_size,
                                "output_size": output_size,
                                "type": "SimpleNN"
                            }, f)
                        self.logger.info(f"PyTorch神经网络已保存到 {model_file}")
                except Exception as e:
                    self.logger.error(f"保存神经网络模型时出错: {e}")
                    
            if 'stacking_model' in models:
                try:
                    model_file = model_base_dir / "stacking_model.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump(models['stacking_model'], f)
                    self.logger.info(f"堆叠模型已保存到 {model_file}")
                except Exception as e:
                    self.logger.error(f"保存堆叠模型时出错: {e}")
                    
            if 'scaler' in models:
                try:
                    scaler_file = model_base_dir / "scaler.pkl"
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(models['scaler'], f)
                    self.logger.info(f"特征缩放器已保存到 {scaler_file}")
                except Exception as e:
                    self.logger.error(f"保存特征缩放器时出错: {e}")
                    
            # 保存元数据
            metadata_file = model_base_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                metadata = {
                    'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'feature_count': len(feature_cols),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'top_performer_ratio': float(top_ratio),
                    'models': [k for k in models.keys() if k not in ['feature_columns', 'scaler', 'metadata']]
                }
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"板块轮动模型 {model_name} 训练成功，模型已保存到 {model_base_dir}")
            return models
        else:
            self.logger.error(f"训练板块轮动模型 {model_name} 失败，未能成功训练任何子模型")
            return None

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
    
    # 打印current_features的所有键（为了调试）
    if isinstance(current_features, dict):
            logger.info(f"特征字典的键: {list(current_features.keys())}")
            
            # 如果只有一个板块，打印该板块的所有特征
            if len(current_features) == 1:
                sector_code = list(current_features.keys())[0]
                sector_features = current_features[sector_code]
                logger.info(f"板块 {sector_code} 的特征数量: {len(sector_features) if isinstance(sector_features, dict) else 'N/A'}")
                logger.info(f"板块 {sector_code} 的特征列: {list(sector_features.keys()) if isinstance(sector_features, dict) else 'N/A'}")
    else:
            # 如果不是字典，打印其结构
            logger.info(f"非字典特征的结构: {type(current_features)}")
            if hasattr(current_features, 'keys'):
                logger.info(f"特征的键: {list(current_features.keys())}")
            if hasattr(current_features, 'columns'):
                logger.info(f"特征的列: {list(current_features.columns)}")
    
    # 添加市场状态特征
    market_status = trainer._load_market_status(date)
    if market_status:
            # 为所有板块添加相同的市场特征
            current_features['market_type'] = market_status.get('market_type', '32')  # 默认为中性盘整
            current_features['market_cycle'] = market_status.get('market_cycle', '2')  # 默认为上升期
            current_features['trend_strength'] = market_status.get('trend_strength', 0.0)
            
            # 处理个股指标 - 使用创业板指数作为市场代表
            if 'individual_status' in market_status and '399006.SZ' in market_status['individual_status']:
                cyb_data = market_status['individual_status']['399006.SZ']
                
                # 市场评分指标
                current_features['trend_score'] = cyb_data.get('trend_score', 0.0)
                current_features['position_score'] = cyb_data.get('position_score', 0.0)
                current_features['volume_score'] = cyb_data.get('volume_score', 0.0)
                current_features['momentum_score'] = cyb_data.get('momentum_score', 0.0)
                current_features['volatility_score'] = cyb_data.get('volatility_score', 0.0)
                current_features['structure_score'] = cyb_data.get('structure_score', 0.0)
                current_features['cycle_score'] = cyb_data.get('cycle_score', 0.0)
                current_features['total_score'] = cyb_data.get('total_score', 0.0)
                current_features['profit_loss_ratio_14'] = cyb_data.get('profit_loss_ratio_14', 0.0)
                current_features['profit_loss_ratio_30'] = cyb_data.get('profit_loss_ratio_30', 0.0)
                
                # 技术指标派生特征
                if 'technical_data' in cyb_data:
                    tech_data = cyb_data['technical_data']
                    
                    # 价格与均线关系
                    close = tech_data.get('close', 0.0)
                    ma5 = tech_data.get('ma5', 0.0)
                    ma20 = tech_data.get('ma20', 0.0)
                    ma60 = tech_data.get('ma60', 0.0)
                    
                    if ma5 > 0:
                        current_features['price_to_ma5'] = close / ma5
                    if ma20 > 0:
                        current_features['price_to_ma20'] = close / ma20
                    if ma60 > 0:
                        current_features['price_to_ma60'] = close / ma60
                    if ma5 > 0 and ma20 > 0:
                        current_features['ma5_to_ma20'] = ma5 / ma20
                    
                    # 动量指标
                    current_features['macd_hist'] = tech_data.get('macd_hist', 0.0)
                    current_features['rsi6_norm'] = tech_data.get('rsi6', 50.0) / 50.0 - 1.0  # 归一化到[-1,1]区间
    
    # 添加当前日期
    current_features['current_date'] = date
    
    # 再次打印：查看添加市场状态后的特征（为了调试）
    logger.info(f"添加市场状态后的特征数量: {len(current_features) if isinstance(current_features, dict) else 'N/A'}")
    logger.info(f"添加市场状态后的特征列: {list(current_features.keys())[:20] if isinstance(current_features, dict) else 'N/A'} {'...' if len(current_features) > 20 else ''}")
    
    # 加载模型
    model_types = ["xgb_return", "lgb_return", "xgb_rank", "lgb_rank", "lnn_return"]
    # if hasattr(trainer, 'use_lnn') and trainer.use_lnn:
    #         model_types.append("lnn_return")
    
    loaded_models = {}
    try:      
            # 加载特征列
            with open(rotation_model_dir / "feature_columns.json", "r") as f:
                feature_columns = json.load(f)
    except Exception as e:
            logger.error(f"加载特征列时出错: {e}")
            logger.error(traceback.format_exc())
            feature_columns = {}

    try:
            sys.path.append(str(get_project_root() / "to"))
            from Liquid.nn.liquid.liquid_network import LiquidNetModel
    except Exception as e:
            logger.error(f"加载LiquidNetModel时出错: {e}")
            logger.error(traceback.format_exc())
    
    # 尝试加载不同类型的模型
    for model_type in model_types:
        if model_type.startswith('xgb'):
            model_file = rotation_model_dir / f"{model_type}_model.json"
            logger.info(f"检查XGBoost模型文件: {model_file}")
            if model_file.exists():
                try:
                    import xgboost as xgb
                    model = xgb.XGBRegressor() if 'return' in model_type or 'rank' in model_type else xgb.XGBClassifier()
                    model.load_model(model_file)
                    loaded_models[model_type] = model
                    logger.info(f"成功加载模型 {model_type}")
                except Exception as e:
                    logger.error(f"加载模型 {model_type} 出错: {e}")
                    logger.error(traceback.format_exc())
        elif model_type.startswith('lgb'):
            model_file = rotation_model_dir / f"{model_type}_model.txt"
            logger.info(f"检查LightGBM模型文件: {model_file}")
            if model_file.exists():
                try:
                    import lightgbm as lgb
                    model = lgb.Booster(model_file=str(model_file))
                    loaded_models[model_type] = model
                    logger.info(f"成功加载模型 {model_type}")
                except Exception as e:
                    logger.error(f"加载模型 {model_type} 出错: {e}")
                    logger.error(traceback.format_exc())
        elif model_type.startswith('lnn'):
            # 使用新的模型目录结构
            model_dir = rotation_model_dir / "lnn_models"
            logger.info(f"检查LNN模型目录: {model_dir}")
            if model_dir.exists():
                try:
                    # 尝试导入液体神经网络
                    sys.path.append(str(get_project_root() / "to"))
                    from Liquid.nn.liquid.liquid_network import LiquidNetModel
                    
                    # 检查是否有模型参数文件
                    params_file = model_dir / "model_params.json"
                    weights_loaded = False  # 标记是否成功加载了权重
                    
                    if params_file.exists():
                        # 从参数文件加载参数
                        with open(params_file, 'r') as f:
                            params = json.load(f)
                        input_size = params.get('input_size', len(feature_columns))
                        output_size = params.get('output_size', 1)
                        hidden_size = params.get('hidden_size', 64)
                        num_layers = params.get('num_layers', 2)
                    else:
                        # 找不到参数文件，使用默认参数
                        logger.warning(f"找不到LNN模型参数文件: {params_file}，将使用默认参数")
                        input_size = len(feature_columns)
                        output_size = 1
                        hidden_size = 64
                        num_layers = 2
                            
                    # 初始化模型
                    model = LiquidNetModel(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        num_layers=num_layers
                    )
                    
                    logger.info(f"LNN模型初始化: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}")
                    
                    # 尝试直接加载模型文件
                    model_file = model_dir / "lnn_model.pt"
                    if model_file.exists():
                        try:
                            logger.info(f"准备加载LNN模型文件: {model_file}")
                            # 尝试直接加载完整模型（添加weights_only=False解决PyTorch 2.6兼容性问题）
                            checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
                            logger.info(f"成功读取checkpoint，类型: {type(checkpoint)}，包含键: {checkpoint.keys() if isinstance(checkpoint, dict) else '非字典类型'}")
                            if isinstance(checkpoint, dict):
                                # 检查是否是LNN save方法保存的格式
                                if 'model_state_dict' in checkpoint:
                                    model.model.load_state_dict(checkpoint['model_state_dict'])
                                    
                                    # 如果checkpoint包含optimzer，也加载它
                                    if 'optimizer_state_dict' in checkpoint and hasattr(model, 'optimizer'):
                                        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                    
                                    # 如果checkpoint包含scaler，也加载它
                                    if 'scaler' in checkpoint:
                                        model.scaler = checkpoint['scaler']
                                        
                                    logger.info(f"成功从{model_file}加载模型")
                                    weights_loaded = True
                        except Exception as e:
                            logger.warning(f"加载模型文件失败: {e}")
                    else:
                        logger.warning(f"LNN模型文件不存在: {model_file}")
                    
                    # 如果没有成功加载模型，尝试加载单独的scaler
                    if not hasattr(model, 'scaler') or model.scaler is None:
                        scaler_file = model_dir / "scaler.pkl"
                        if scaler_file.exists():
                            try:
                                import joblib
                                model.scaler = joblib.load(scaler_file)
                                logger.info(f"成功加载StandardScaler: {scaler_file}")
                            except Exception as e:
                                logger.warning(f"加载StandardScaler时出错: {e}")
                                # 创建一个新的scaler
                                from sklearn.preprocessing import StandardScaler
                                model.scaler = StandardScaler()
                                logger.info("已创建一个新的StandardScaler实例")
                        else:
                            logger.warning(f"StandardScaler文件不存在: {scaler_file}，将创建新实例")
                            from sklearn.preprocessing import StandardScaler
                            model.scaler = StandardScaler()
                            logger.info("已创建一个新的StandardScaler实例")
                    
                    # 即使模型权重没有加载成功，我们也添加模型实例
                    loaded_models[model_type] = model
                    if weights_loaded:
                        logger.info(f"成功加载模型 {model_type}")
                    else:
                        logger.warning(f"未能加载模型 {model_type} 的权重，将使用随机初始化的模型")
                    
                except Exception as e:
                    logger.error(f"加载模型 {model_type} 出错: {e}")
                    logger.error(traceback.format_exc())
            else:
                # 如果新的模型目录不存在，尝试向后兼容旧的目录结构
                old_model_dir = rotation_model_dir / f"{model_type}_model"
                logger.warning(f"新的LNN模型目录不存在: {model_dir}，尝试旧目录: {old_model_dir}")
                
                if old_model_dir.exists():
                    try:
                        # 尝试导入液体神经网络
                        sys.path.append(str(get_project_root() / "to"))
                        from Liquid.nn.liquid.liquid_network import LiquidNetModel
                        
                        # 检查是否有模型配置文件
                        params_file = old_model_dir / "model_params.json"
                        weights_loaded = False  # 标记是否成功加载了权重
                        
                        if params_file.exists():
                            # 从配置文件加载参数
                            with open(params_file, 'r') as f:
                                params = json.load(f)
                            input_size = params.get('input_size', len(feature_columns))
                            output_size = params.get('output_size', 1)
                            hidden_size = params.get('hidden_size', 64)
                            num_layers = params.get('num_layers', 2)
                        else:
                            # 使用默认参数
                            input_size = len(feature_columns)
                            output_size = 1
                            hidden_size = 64
                            num_layers = 2
                        
                        # 初始化模型
                        model = LiquidNetModel(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=output_size,
                            num_layers=num_layers
                        )
                        
                        # 尝试加载旧格式的模型权重
                        complete_model_file = old_model_dir / "complete_model.pt"
                        if complete_model_file.exists():
                            try:
                                checkpoint = torch.load(complete_model_file, map_location="cpu", weights_only=False)
                                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                                    model.model.load_state_dict(checkpoint['model_state_dict'])
                                    
                                    # 如果checkpoint包含scaler，也加载它
                                    if 'scaler' in checkpoint:
                                        model.scaler = checkpoint['scaler']
                                    
                                    weights_loaded = True
                                    logger.info("成功从旧格式的complete_model.pt加载模型")
                            except Exception as e:
                                logger.warning(f"加载旧格式模型文件失败: {e}")
                        
                        # 即使模型权重没有加载成功，我们也添加模型实例
                        loaded_models[model_type] = model
                        if weights_loaded:
                            logger.info(f"成功从旧格式加载模型 {model_type}")
                        else:
                            logger.warning(f"未能从旧格式加载模型 {model_type} 的权重，将使用随机初始化的模型")
                        
                    except Exception as e:
                        logger.error(f"加载旧格式模型 {model_type} 出错: {e}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"LNN模型目录不存在: {model_dir}，旧格式目录也不存在: {old_model_dir}")
    
    if not loaded_models:
            logger.error(f"未能加载任何模型")
            return None
    
    logger.info(f"成功加载 {len(loaded_models)} 个模型: {list(loaded_models.keys())}")
    
    # 准备特征数据
    # 将特征字典转换为DataFrame
    try:
        # 先检查current_features的结构
        if isinstance(current_features, dict):
                # 如果current_features是字典但不是板块特征的集合
                if not any(isinstance(v, dict) for v in current_features.values()):
                    # 这种情况下current_features已经是一个特征字典，可以直接转换为DataFrame
                    features_df = pd.DataFrame([current_features])
                else:
                    # 这种情况下current_features是板块特征的集合，需要处理每个板块
                    logger.info(f"处理 {len(current_features)} 个板块的特征数据进行预测")
                    
                    # 创建一个新的DataFrame来存储所有板块的特征和预测结果
                    all_sectors_features = []
                    
                    # 遍历所有板块特征
                    for sector_code, sector_features in current_features.items():
                        if isinstance(sector_features, dict):
                            # 将市场环境特征添加到板块特征
                            for k in ['market_type', 'market_cycle', 'trend_strength', 'current_date', 
                                    'trend_score', 'position_score', 'volume_score', 'momentum_score', 
                                    'volatility_score', 'structure_score', 'cycle_score', 'total_score',
                                    'profit_loss_ratio_14', 'profit_loss_ratio_30', 'price_to_ma5', 
                                    'price_to_ma20', 'price_to_ma60', 'ma5_to_ma20', 'macd_hist', 'rsi6_norm']:
                                if k in current_features and k not in sector_features:
                                    sector_features[k] = current_features[k]
                        
                            # 添加板块代码
                            sector_features['sector_code'] = sector_code
                            all_sectors_features.append(sector_features)
                    
                    # 将所有板块特征转换为DataFrame
                    features_df = pd.DataFrame(all_sectors_features)
        else:
            # 如果current_features已经是DataFrame
            features_df = current_features
        
        logger.info(f"特征DataFrame的形状: {features_df.shape}")
        logger.info(f"特征DataFrame的列: {list(features_df.columns)}...")
        
        # 确保所有需要的特征列都存在
        missing_cols = [col for col in feature_columns if col not in features_df.columns]
        if missing_cols:
            logger.warning(f"特征数据中缺少以下列: {missing_cols}")
            # 为缺失的特征列添加默认值0
            for col in missing_cols:
                features_df[col] = 0
                
        # 只选择模型所需的特征列，并确保行的sector_code信息保留
        sector_codes = features_df['sector_code'].values if 'sector_code' in features_df.columns else None
        X = features_df[feature_columns].fillna(0)
        
        logger.info(f"成功准备特征数据，形状: {X.shape}")
    except Exception as e:
        logger.error(f"准备特征数据时出错: {e}")
        logger.error(traceback.format_exc())
        return None
    
    # 在适当位置添加特征数据形状的日志记录（在"使用模型进行预测"上方）
    logger.info(f"特征数据形状: {X.shape if 'X' in locals() else 'X not defined'}")

    # 使用模型进行预测
    predictions = {}
    for model_type, model in loaded_models.items():
        try:
                # 确保特征数据符合要求
                features = X.copy()
                
                # 检查NaN和无穷值
                import numpy as np  # 确保numpy已导入
                if features.isna().any().any() or np.isinf(features.values).any():
                    logger.warning(f"特征数据中包含NaN或无穷值，将进行填充")
                    features = features.fillna(0)
                    features = features.replace([np.inf, -np.inf], 0)
                
                if model_type.startswith('xgb'):
                    preds = model.predict(features)
                    predictions[model_type] = preds
                elif model_type.startswith('lgb'):
                    # LightGBM模型需要特殊处理
                    if hasattr(model, 'predict'):
                        # 使用predict方法
                        preds = model.predict(features)
                    else:
                        # 兼容其他情况
                        logger.warning(f"LightGBM模型没有predict方法，尝试其他方式")
                        continue
                    predictions[model_type] = preds
                elif model_type.startswith('lnn'):
                    # 尝试使用LiquidNetModel预测
                    try:
                        logger.info(f"LNN模型预测开始: 特征形状={features.shape}, 非零值比例={np.count_nonzero(features.values)/features.size:.4f}")
                        logger.info(f"LNN模型参数状态: {sum(p.numel() for p in model.model.parameters())}个参数, 非零参数比例: {sum(torch.count_nonzero(p) for p in model.model.parameters())/sum(p.numel() for p in model.model.parameters()):.4f}")
                        # 使用numpy数组预测
                        preds = model.predict(features.values).flatten()
                        logger.info(f"LNN预测结果统计: 最小值={preds.min():.6f}, 最大值={preds.max():.6f}, 平均值={preds.mean():.6f}, 标准差={preds.std():.6f}")
                        logger.info(f"LNN预测非零值数量: {np.count_nonzero(preds)}/{len(preds)}, 比例: {np.count_nonzero(preds)/len(preds) if len(preds)>0 else 0:.4f}")
                    except Exception as lnn_err:
                        logger.error(f"使用模型 {model_type} 预测出错: {lnn_err}")
                        
                        # 检查是否是StandardScaler未初始化的错误
                        if 'StandardScaler instance is not fitted yet' in str(lnn_err):
                            logger.warning(f"模型的StandardScaler未初始化，尝试创建新的标准化器")
                            try:
                                # 使用当前特征构建一个新的StandardScaler
                                from sklearn.preprocessing import StandardScaler
                                
                                # 如果模型已经有scaler但未初始化，先尝试用当前数据拟合它
                                if hasattr(model, 'scaler') and model.scaler is not None:
                                    model.scaler.fit(features.values)
                                    logger.info(f"已使用当前特征拟合现有的StandardScaler")
                                else:
                                    # 创建新的StandardScaler实例
                                    model.scaler = StandardScaler()
                                    model.scaler.fit(features.values)
                                    logger.info(f"已创建新的StandardScaler，使用当前特征进行拟合")
                                
                                # 重新尝试预测
                                preds = model.predict(features.values).flatten()
                                logger.info(f"使用新拟合的StandardScaler预测成功")
                            except Exception as new_err:
                                logger.error(f"使用新的StandardScaler预测仍然失败: {new_err}")
                                logger.error(traceback.format_exc())
                                # 跳过此模型
                                continue
                        else:
                            # 尝试不使用StandardScaler直接预测
                            try:
                                logger.warning("尝试跳过StandardScaler直接使用模型预测")
                                # 获取原始模型进行预测
                                raw_output = model.model(torch.FloatTensor(features.values).to(model.device))
                                preds = raw_output.detach().cpu().numpy().flatten()
                                logger.info("直接使用模型预测成功")
                            except Exception as raw_err:
                                logger.error(f"直接使用模型预测也失败: {raw_err}")
                                logger.error(traceback.format_exc())
                                # 其他错误，跳过此模型
                                continue
                    
                    predictions[model_type] = preds
        except Exception as e:
            logger.error(f"使用模型 {model_type} 预测出错: {e}")
            logger.error(traceback.format_exc())
            
    if not predictions:
        logger.error("没有生成任何预测结果")
        return None
    
    # 将预测结果添加到特征中
    for model_type, pred in predictions.items():
        col_name = f'pred_{model_type}'
        if isinstance(pred, np.ndarray):
                if len(pred) == len(features_df):
                    features_df[col_name] = pred
                else:
                    logger.warning(f"预测结果 {model_type} 的长度与特征数量不匹配: {len(pred)} vs {len(features_df)}")
                    # 尝试扩展或截断预测结果以匹配特征数量
                    if len(pred) > 0:
                        if len(pred) > len(features_df):
                            features_df[col_name] = pred[:len(features_df)]
                        else:
                            # 扩展预测结果
                            extended_pred = np.zeros(len(features_df))
                            extended_pred[:len(pred)] = pred
                            features_df[col_name] = extended_pred
                    else:
                        features_df[col_name] = 0.0
        else:
            # 如果只有一个预测值，扩展到所有行
            features_df[col_name] = pred
    
    # 计算综合得分（简单加权平均）
    features_df['composite_score'] = 0.0
    
    # 根据模型类型赋予权重
    weights = {}
    if 'pred_xgb_return' in features_df.columns:
            weights['pred_xgb_return'] = 0.35
    if 'pred_lgb_return' in features_df.columns:
            weights['pred_lgb_return'] = 0.25
    if 'pred_lnn_return' in features_df.columns:  # 添加LNN权重
            weights['pred_lnn_return'] = 0.2
    if 'pred_xgb_clf' in features_df.columns:
            weights['pred_xgb_clf'] = 0.1
    if 'pred_xgb_rank' in features_df.columns:
            weights['pred_xgb_rank'] = 0.1
        
    # 如果没有权重，使用平均值
    if not weights:
        pred_cols = [col for col in features_df.columns if col.startswith('pred_')]
        for col in pred_cols:
            weights[col] = 1.0 / len(pred_cols)
            
    # 计算加权得分
    for col, weight in weights.items():
        if col in features_df.columns:
            features_df['composite_score'] += features_df[col] * weight
    
    # 计算排名，按composite_score降序排序
    features_df = features_df.sort_values('composite_score', ascending=False)
    features_df['rank'] = range(1, len(features_df) + 1)
                    
    # 确保'sector_code'列存在
    if 'sector_code' not in features_df.columns:
        # 如果没有sector_code列，但有板块列表，使用板块列表
        if sector_list and len(sector_list) == len(features_df):
            features_df['sector_code'] = sector_list
        else:
            # 否则使用默认值
            features_df['sector_code'] = [f"sector_{i}" for i in range(len(features_df))]
    
    # 获取前N名板块
    top_sectors = features_df.head(top_n)
    
    result = {
        'date': date,
        'forward_days': forward_days,
        'top_sectors': []
        }
    
    # 将结果转换为字典列表
    for _, row in top_sectors.iterrows():
        sector_info = {
            'sector_code': row['sector_code'],
            'composite_score': float(row['composite_score']),
            'rank': int(row['rank'])
        }
        
        # 添加各模型的预测结果
        for model_type in loaded_models.keys():
            pred_col = f'pred_{model_type}'
            if pred_col in row:
                sector_info[model_type.replace('_', '_pred_')] = float(row[pred_col])
        
        result['top_sectors'].append(sector_info)
    
    logger.info(f"预测结果: 日期 {date} 未来 {forward_days} 天表现最好的前 {top_n} 个板块:")
    for sector in result['top_sectors']:
        sector_info = f"  {sector['sector_code']}: 综合分数 {sector['composite_score']:.4f}, 排名 {sector['rank']}"
        # 添加各模型预测值
        model_preds = [f"{k.split('_pred_')[1]}={v:.4f}" for k, v in sector.items() if '_pred_' in k]
        if model_preds:
            sector_info += f" ({', '.join(model_preds)})"
        logger.info(sector_info)
    
    return result


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="板块轮动模型训练与预测")
    
    # 运行模式
    parser.add_argument("--mode", type=str, choices=['train', 'predict'], default='train',
                     help="运行模式：train表示训练模型，predict表示使用已训练的模型进行预测")
    
    # 模型训练子模式
    parser.add_argument("--train_mode", type=str, choices=['individual', 'rotation'], default='rotation',
                     help="训练模式：individual表示为每个板块单独训练模型，rotation表示训练统一的板块轮动模型")
    
    # 日期相关参数
    parser.add_argument("--date", type=str, help="指定日期，格式为YYYYMMDD，用于预测模式")
    parser.add_argument("--start_date", type=str, help="开始日期，格式为YYYYMMDD，用于训练模式")
    parser.add_argument("--end_date", type=str, help="结束日期，格式为YYYYMMDD，用于训练模式")
    
    # 预测相关参数
    parser.add_argument("--top_n", type=int, default=2, help="预测模式下返回的板块数量")
    
    # 训练参数
    parser.add_argument("--lookback_short", type=int, default=20, help="短期回溯天数")
    parser.add_argument("--lookback_long", type=int, default=60, help="长期回溯天数")
    parser.add_argument("--forward_days", type=int, default=2, help="预测未来几天")
    parser.add_argument("--sector", type=str, help="指定板块代码（不指定则训练/预测所有板块）")
    parser.add_argument("--max_workers", type=int, default=None, help="最大进程数")
    parser.add_argument("--use_lnn", action="store_true", default=True, help="是否使用液体神经网络")
    
    # 模型与输出相关
    parser.add_argument("--model_dir", type=str, help="模型目录")
    parser.add_argument("--model_name", type=str, default="sector_rotation", help="模型名称")
    
    args = parser.parse_args()
    
    # 设置模型目录
    try:
        model_dir = args.model_dir if args.model_dir else get_model_path()
        logger.info(f"原始模型目录: {model_dir}")
        
        # 如果是预测模式，确保路径的一致性处理
        # predict_sector_rotation内部会处理sector_rotation子路径
        if args.mode == 'predict':
            # 预测模式不需要在这里修改model_dir，因为predict_sector_rotation函数会处理
            pass
        elif args.train_mode == 'rotation':
            # 训练轮动模型时，确保模型目录的最后一级不是sector_rotation
            model_dir_path = Path(model_dir)
            if model_dir_path.name == 'sector_rotation':
                # 如果已经是sector_rotation，就不再添加
                logger.info(f"模型目录已经包含sector_rotation，保持原样: {model_dir}")
            else:
                # 原本的模式，添加model_name作为子目录
                old_path = str(model_dir_path)
                model_dir = model_dir_path / args.model_name
                logger.info(f"为轮动模型添加子目录: {old_path} -> {model_dir}")
                
        # 确保模型目录存在
        ensure_dir(Path(model_dir))
        logger.info(f"最终使用的模型目录: {model_dir}")
    except Exception as e:
        logger.error(f"处理模型目录时出错: {e}")
        logger.error(traceback.format_exc())
        return
    
    # 根据运行模式选择功能
    if args.mode == 'train':
        # 训练模式
        # 设置默认日期
        if not args.start_date:
            args.start_date = '20240501'
        if not args.end_date:
            args.end_date = '20241231'
        
        # 创建训练器
        trainer = SectorModelTrainer(model_dir=model_dir, use_lnn=args.use_lnn)
        
        # 设置参数
        trainer.lookback_short = args.lookback_short
        trainer.lookback_long = args.lookback_long
        trainer.forward_days = args.forward_days
        
        # 获取板块列表
        if args.sector:
            sector_list = [args.sector]
        else:
            # 从本地特征文件中获取所有板块
            sector_list = []
            try:
               from stock_sectors_mapping import get_tng_sectors, get_sector_stocks
               sector_list = get_tng_sectors()
            except Exception as e:
                logger.error(f"获取板块列表时出错: {e}")
                logger.error(traceback.format_exc())
        if not sector_list:
            logger.error("未能获取到任何板块数据")
            raise ValueError("板块列表为空，无法继续训练")
        
        logger.info(f"将处理 {len(sector_list)} 个板块: {sector_list}")
        
        # 根据模式选择训练方法
        if args.train_mode == 'rotation':
            # 训练统一的板块轮动模型
            logger.info("使用板块轮动模式训练模型")
            
            # 确保交易日历已加载
            if not hasattr(trainer, 'trading_days') or not trainer.trading_days:
                trainer._load_trading_calendar()
            
            # 构建板块轮动特征数据集 - 直接使用用户指定的日期范围
            # 这不会导致要求early日期的数据，因为_build_rotation_features_dataset方法
            # 会自己处理日期范围和特征回溯的逻辑
            logger.info(f"使用日期范围 {args.start_date} - {args.end_date} 构建特征数据集")
            features_df = trainer._build_rotation_features_dataset(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
                forward_days=args.forward_days
            )
            
            if features_df is None or features_df.empty:
                logger.error("无法构建有效的板块轮动特征数据集，训练失败")
                return
                
            # 训练模型
            result = trainer.train_rotation_model(
                features_df=features_df,
                model_name=args.model_name
            )
            
            # 输出结果
            if result:
                logger.info(f"板块轮动模型 {args.model_name} 训练成功")
                for model_key, perf in result.get('performance', {}).items():
                    metrics = ", ".join([f"{k}={v:.4f}" for k, v in perf.items()])
                    logger.info(f"  {model_key}: {metrics}")
            else:
                logger.warning("板块轮动模型训练失败")
        else:
            # 为每个板块单独训练模型
            logger.info("使用独立板块模式训练模型")
            
            # 训练模型
            results = trainer.train_all_sectors(
                sector_list=sector_list,
                start_date=args.start_date,
                end_date=args.end_date,
                max_workers=args.max_workers
            )
            
            # 输出结果
            if results:
                logger.info(f"成功训练 {len(results)} 个板块的模型")
                for sector, result in results.items():
                    logger.info(f"板块 {sector} 性能:")
                    for model, perf in result.get('performance', {}).items():
                        logger.info(f"  {model}: MSE={perf.get('mse', 'N/A'):.6f}, R²={perf.get('r2', 'N/A'):.4f}")
            else:
                logger.warning("没有成功训练任何板块的模型")
    
    elif args.mode == 'predict':
        # 预测模式
        if not args.date:
            logger.error("预测模式需要指定日期 (--date)")
            return
        
        # 确定使用的板块
        sector_list = None
        if args.sector:
            sector_list = [args.sector]
        
        # 修复预测模式下的路径传递
        # 运行预测
        logger.info(f"使用板块轮动模型进行预测，日期: {args.date}")
        result = predict_sector_rotation(
            model_dir=model_dir,
            date=args.date,
            sector_list=sector_list,
            forward_days=args.forward_days,
            top_n=args.top_n
        )
        
        if result:
            logger.info(f"预测完成，发现 {len(result['top_sectors'])} 个潜在强势板块")
            
            # 输出详细结果
            print(f"\n预测日期: {args.date}, 未来 {args.forward_days} 天的板块轮动预测:")
            print("-" * 60)
            print(f"{'排名':^5}|{'板块代码':^20}|{'综合得分':^10}|{'预测涨幅':^10}")
            print("-" * 60)
            
            for i, sector in enumerate(result['top_sectors']):
                predicted_return = sector.get('predicted_return', float('nan'))
                print(f"{i+1:^5}|{sector['sector_code']:^20}|{sector['composite_score']:^10.4f}|{predicted_return:^10.2f}")
                
            print("-" * 60)
            print("注: 结果仅供参考，实际投资决策请结合其他因素综合考虑。")
        else:
            logger.error("预测失败")

if __name__ == "__main__":
    main() 