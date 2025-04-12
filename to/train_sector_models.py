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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
import copy

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

def get_model_path(relative_path=None):
    """获取模型目录路径"""
    project_root = get_project_root()
    model_path = project_root / "to" / "models" / "sector_models"
    if relative_path:
        return model_path / relative_path
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
            
            if main_features_path.exists():
                # 如果主特征文件存在，直接读取并筛选日期
                self.logger.debug(f"尝试从主特征文件加载板块 {sector_code} 的日期 {date} 数据")
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
                
            with open(status_file, "r") as f:
                market_status_data = json.load(f)
                
            # 获取特定日期的市场状态
            if date in market_status_data:
                return market_status_data[date]
            else:
                self.logger.warning(f"日期 {date} 的市场状态数据不存在")
                return None
        except Exception as e:
            self.logger.error(f"加载市场状态数据时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def _build_features_for_date_range(self, sector_code, end_date, lookback_days=60, forward_days=5):
        """
        为指定日期范围构建特征数据
        
        Args:
            sector_code: 板块代码
            end_date: 结束日期，格式为YYYYMMDD
            lookback_days: 回溯天数
            forward_days: 预测未来几天
            
        Returns:
            DataFrame: 特征数据
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
                        
                        sector_df['change_percent'] = change_percent
                        
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
        
        return features_df

    def _prepare_training_data(self, sector_code, start_date, end_date):
        """
        准备训练数据
        
        Args:
            sector_code: 板块代码
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            tuple: (X, y) 特征和标签
        """
        # 确定日期范围
        try:
            start_idx = self.trading_days.index(start_date) if start_date in self.trading_days else 0
            end_idx = self.trading_days.index(end_date) if end_date in self.trading_days else -1
            
            if start_idx > end_idx or end_idx == -1:
                self.logger.warning(f"日期范围 {start_date} - {end_date} 无效")
                return None, None
                
            trading_range = self.trading_days[start_idx:end_idx+1]
            
            # 收集所有日期的特征
            all_features = []
            
            # 使用短期和长期两种回溯
            for date in trading_range:
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
                
                # 合并短期和长期特征
                if short_features is not None and long_features is not None:
                    # 为区分特征来源，添加前缀
                    for col in short_features.columns:
                        if col not in ['feature_date', 'label_date', 'future_return', 'sector_code']:
                            short_features.rename(columns={col: f'short_{col}'}, inplace=True)
                            
                    for col in long_features.columns:
                        if col not in ['feature_date', 'label_date', 'future_return', 'sector_code']:
                            long_features.rename(columns={col: f'long_{col}'}, inplace=True)
                    
                    # 合并特征
                    merged_features = pd.merge(
                        short_features[['feature_date', 'label_date', 'future_return'] + [c for c in short_features.columns if c.startswith('short_')]],
                        long_features[['feature_date'] + [c for c in long_features.columns if c.startswith('long_')]],
                        on='feature_date',
                        how='inner'
                    )
                    
                    if not merged_features.empty:
                        all_features.append(merged_features)
                
            if not all_features:
                self.logger.warning(f"板块 {sector_code} 在日期范围 {start_date} - {end_date} 内没有可用的特征数据")
                return None, None
                
            # 合并所有日期的特征
            training_df = pd.concat(all_features, ignore_index=True)
            
            # 添加时间衰减权重
            training_df['time_weight'] = self._calculate_time_decay_weights(training_df)
            
            # 准备特征和标签
            feature_cols = [c for c in training_df.columns if (c.startswith('short_') or c.startswith('long_')) 
                          and c not in ['feature_date', 'label_date', 'future_return', 'time_weight']]
            
            X = training_df[feature_cols].copy()
            y = training_df['future_return'].copy()
            weights = training_df['time_weight'].copy()
            
            # 处理缺失值
            X = X.fillna(0)
            
            return X, y, weights
            
        except Exception as e:
            self.logger.error(f"准备板块 {sector_code} 的训练数据出错: {e}")
            self.logger.error(traceback.format_exc())
            return None, None, None

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
        """
        训练特定板块的模型
        
        Args:
            sector_code: 板块代码
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            dict: 训练结果
        """
        self.logger.info(f"开始训练板块 {sector_code} 的模型，日期范围: {start_date} - {end_date}")
        
        # 准备训练数据
        X, y, weights = self._prepare_training_data(sector_code, start_date, end_date)
        
        if X is None or y is None or X.empty or len(y) == 0:
            self.logger.warning(f"板块 {sector_code} 没有足够的训练数据")
            return None
            
        self.logger.info(f"准备好板块 {sector_code} 的训练数据，特征数: {X.shape[1]}，样本数: {len(y)}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, test_size=0.2, shuffle=False
        )
        
        models = {}
        predictions = {}
        
        # 训练 XGBoost 模型
        try:
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            xgb_model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            models['xgboost'] = xgb_model
            predictions['xgboost'] = xgb_model.predict(X_test)
            
            self.logger.info(f"XGBoost 模型训练完成，测试集 MSE: {mean_squared_error(y_test, predictions['xgboost']):.6f}")
            
        except Exception as e:
            self.logger.error(f"训练 XGBoost 模型出错: {e}")
            self.logger.error(traceback.format_exc())
        
        # 训练 LightGBM 模型
        try:
            lgb_model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            lgb_model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            models['lightgbm'] = lgb_model
            predictions['lightgbm'] = lgb_model.predict(X_test)
            
            self.logger.info(f"LightGBM 模型训练完成，测试集 MSE: {mean_squared_error(y_test, predictions['lightgbm']):.6f}")
            
        except Exception as e:
            self.logger.error(f"训练 LightGBM 模型出错: {e}")
            self.logger.error(traceback.format_exc())
        
        # 训练液体神经网络
        if self.use_lnn:
            try:
                # 构建液体神经网络模型
                lnn_model = self.LiquidNetModel(
                    input_dim=X.shape[1],
                    hidden_dims=[64, 32],
                    output_dim=1,
                    lr=0.001
                )
                
                lnn_model.fit(
                    X_train.values, y_train.values.reshape(-1, 1),
                    batch_size=32,
                    epochs=50,
                    validation_data=(X_test.values, y_test.values.reshape(-1, 1)),
                    verbose=0
                )
                
                models['lnn'] = lnn_model
                preds = lnn_model.predict(X_test.values)
                predictions['lnn'] = preds.flatten()
                
                self.logger.info(f"液体神经网络模型训练完成，测试集 MSE: {mean_squared_error(y_test, predictions['lnn']):.6f}")
                
            except Exception as e:
                self.logger.error(f"训练液体神经网络模型出错: {e}")
                self.logger.error(traceback.format_exc())
        
        # 保存模型
        if models:
            self._save_models(sector_code, models, X.columns.tolist())
            
            # 返回训练结果
            result = {
                'sector_code': sector_code,
                'start_date': start_date,
                'end_date': end_date,
                'feature_count': X.shape[1],
                'sample_count': len(y),
                'models': list(models.keys()),
                'performance': {}
            }
            
            # 计算每个模型的性能指标
            for model_name, preds in predictions.items():
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                result['performance'][model_name] = {
                    'mse': mse,
                    'r2': r2
                }
                
            return result
        else:
            self.logger.warning(f"板块 {sector_code} 没有成功训练任何模型")
            return None

    def _save_models(self, sector_code, models, feature_columns):
        """
        保存模型
        
        Args:
            sector_code: 板块代码
            models: 模型字典
            feature_columns: 特征列名
        """
        # 创建板块模型目录
        sector_dir = self.model_dir / sector_code
        ensure_dir(sector_dir)
        
        # 保存特征列名
        feature_file = sector_dir / "feature_columns.json"
        with open(feature_file, "w") as f:
            json.dump(feature_columns, f)
        
        # 保存每个模型
        for model_name, model in models.items():
            if model_name == 'xgboost':
                model_file = sector_dir / f"{model_name}_model.json"
                model.save_model(model_file)
            elif model_name == 'lightgbm':
                model_file = sector_dir / f"{model_name}_model.txt"
                model.booster_.save_model(model_file)
            elif model_name == 'lnn':
                model_file = sector_dir / f"{model_name}_model"
                model.save(model_file)
                
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
        features_dict = {}
        
        for sector_code in sector_list:
            try:
                # 构建文件路径 - 按板块名称划分目录
                # 首选加载优化后的主特征文件
                main_features_path = self.sector_features_dir / f"{sector_code}" / f"{sector_code}_features.parquet"
                single_date_path = self.sector_features_dir / f"{sector_code}" / f"{sector_code}_{date}.parquet"
                merged_file_path = self.sector_features_dir / f"{sector_code}" / f"{sector_code}.parquet"
                
                # 尝试从主特征文件加载
                if main_features_path.exists():
                    self.logger.debug(f"尝试从主特征文件加载板块 {sector_code} 的数据")
                    df = pd.read_parquet(main_features_path)
                    
                    # 筛选指定日期
                    date_df = None
                    if 'date' in df.columns:
                        date_df = df[df['date'] == date].copy()
                    elif 'date_str' in df.columns:
                        date_df = df[df['date_str'] == date].copy()
                    
                    if date_df is not None and not date_df.empty:
                        features_dict[sector_code] = date_df.iloc[0].to_dict()
                        self.logger.debug(f"从主特征文件成功加载板块 {sector_code} 日期 {date} 的数据")
                        continue
                
                # 尝试加载单日期特征文件
                elif single_date_path.exists():
                    self.logger.debug(f"尝试从单日期文件加载板块 {sector_code} 的数据")
                    df = pd.read_parquet(single_date_path)
                    if not df.empty:
                        features_dict[sector_code] = df.iloc[0].to_dict()
                        self.logger.debug(f"从单日期文件成功加载板块 {sector_code} 日期 {date} 的数据")
                        continue
                
                # 尝试加载原始合并特征文件
                elif merged_file_path.exists():
                    self.logger.debug(f"尝试从原始合并文件加载板块 {sector_code} 的数据")
                    df = pd.read_parquet(merged_file_path)
                    
                    # 筛选指定日期
                    date_df = None
                    if 'date' in df.columns:
                        date_df = df[df['date'] == date].copy()
                    elif 'date_str' in df.columns:
                        date_df = df[df['date_str'] == date].copy()
                    
                    if date_df is not None and not date_df.empty:
                        features_dict[sector_code] = date_df.iloc[0].to_dict()
                        self.logger.debug(f"从原始合并文件成功加载板块 {sector_code} 日期 {date} 的数据")
                        continue
                
                self.logger.warning(f"未找到板块 {sector_code} 日期 {date} 的特征数据: 尝试路径 {main_features_path}, {single_date_path}, {merged_file_path}")
            except Exception as e:
                self.logger.error(f"加载板块 {sector_code} 日期 {date} 的特征时出错: {e}")
        
        # 返回特征字典
        if not features_dict:
            self.logger.warning(f"日期 {date} 未找到任何板块数据")
            return None
        
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
        features_df = pd.DataFrame(all_features)
        
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
        result_df = result_df[result_df['current_date'] >= start_date]
        
        self.logger.info(f"成功构建板块轮动特征数据集，共 {len(result_df)} 条记录，涵盖 {result_df['current_date'].nunique()} 个交易日")
        
        return result_df

    def train_rotation_model(self, features_df, model_name="sector_rotation"):
        """
        训练板块轮动模型
        
        Args:
            features_df: 轮动特征数据集
            model_name: 模型名称
            
        Returns:
            dict: 包含训练后模型的字典
        """
        self.logger.info(f"开始训练板块轮动模型 {model_name}")
        
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
        self.logger.info(f"特征列: {sorted(features_df.columns)}")
        
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
                random_state=42
            )
            
            # 使用eval_set参数而非早停回合
            eval_set = [(X_test, y_reg_test)]
            
            xgb_reg.fit(
                X_train, y_reg_train,
                eval_set=eval_set,
                eval_metric='rmse',
                verbose=False
            )
            
            # 如需使用早停，通过回调实现
            callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True)]
            
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
            # 计算类别权重
            class_count = y_cls_train.value_counts()
            total = len(y_cls_train)
            
            # 确保不会出现除零错误
            weight_0 = total / (2 * class_count.get(False, 1)) if class_count.get(False, 0) > 0 else 1.0
            weight_1 = total / (2 * class_count.get(True, 1)) if class_count.get(True, 0) > 0 else 1.0
            
            scale = 1 / (weight_0 + weight_1)  # 缩放以使权重和为1
            weight_0 *= scale
            weight_1 *= scale
            
            xgb_cls = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42,
                scale_pos_weight=weight_1/weight_0 if weight_0 > 0 else 1.0
            )
            
            # 使用回调函数设置早停
            xgb_cls.fit(
                X_train, y_cls_train,
                eval_set=[(X_test, y_cls_test)],
                callbacks=[xgb.callback.EarlyStopping(rounds=10, metric_name='logloss', save_best=True)]
            )
            
            # 评估
            y_cls_pred = xgb_cls.predict(X_test)
            accuracy = accuracy_score(y_cls_test, y_cls_pred)
            precision = precision_score(y_cls_test, y_cls_pred)
            recall = recall_score(y_cls_test, y_cls_pred)
            f1 = f1_score(y_cls_test, y_cls_pred)
            self.logger.info(f"XGBoost分类器 - 准确率: {accuracy:.6f}, 精确率: {precision:.6f}, 召回率: {recall:.6f}, F1: {f1:.6f}")
            
            models['xgb_classifier'] = xgb_cls
        except Exception as e:
            self.logger.error(f"训练XGBoost分类器时出错: {e}")
            
        # 3. LightGBM回归器
        self.logger.info("开始训练LightGBM回归器...")
        try:
            lgb_reg = lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # 使用eval_set参数传递验证集
            eval_set = [(X_test, y_reg_test)]
            
            lgb_reg.fit(
                X_train, y_reg_train,
                eval_set=eval_set,
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                verbose=False
            )
            
            y_reg_pred = lgb_reg.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
            r2 = r2_score(y_reg_test, y_reg_pred)
            self.logger.info(f"LightGBM回归器 - RMSE: {rmse:.6f}, R2: {r2:.6f}")
            
            models['lgb_regressor'] = lgb_reg
        except Exception as e:
            self.logger.error(f"训练LightGBM回归器时出错: {e}")
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
            hidden_size = [64, 32]
            output_size = 1
            
            # 导入LiquidNetModel
            try:
                sys.path.append(str(get_project_root() / "to"))
                from Liquid.nn.liquid.liquid_network import LiquidNetModel
                
                # 根据LiquidNetModel实际API创建实例
                net = LiquidNetModel(
                    input_size=input_size,  # 使用input_size而非input_dim
                    hidden_size=hidden_size,
                    output_size=output_size,
                    learning_rate=0.001,
                    dropout_rate=0.3
                )
                
                # 使用LiquidNetModel的fit方法训练
                net.fit(
                    X_train_scaled, y_reg_train.values.reshape(-1, 1),
                    validation_data=(X_test_scaled, y_test_tensor.numpy()),
                    batch_size=64,
                    epochs=50,
                    patience=10,
                    verbose=True
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
                
                # 如果神经网络模型也存在
                if 'nn_regressor' in models:
                    net = models['nn_regressor']
                    net.eval()
                    with torch.no_grad():
                        nn_pred = net(X_test_tensor).numpy().flatten()
                else:
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
            return models
        else:
            self.logger.error(f"训练板块轮动模型 {model_name} 失败，未能成功训练任何子模型")
            return None

def predict_sector_rotation(model_dir, date, sector_list=None, forward_days=2, top_n=5):
    """
    使用训练好的板块轮动模型进行预测
    
    Args:
        model_dir: 模型目录
        date: 日期，格式为YYYYMMDD
        sector_list: 板块列表，如果为None则使用所有可用板块
        forward_days: 预测未来几天
        top_n: 返回表现最好的前N个板块
        
    Returns:
        dict: 预测结果
    """
    logger.info(f"使用板块轮动模型预测日期 {date} 未来 {forward_days} 天的板块轮动情况")
    
    # 创建模型训练器
    trainer = SectorModelTrainer(model_dir=model_dir)
    
    # 获取板块列表
    if sector_list is None:
        try:
            from stock_sectors_mapping import get_tng_sectors
            sector_list = get_tng_sectors()
        except Exception as e:
            logger.error(f"获取板块列表时出错: {e}")
            logger.error(traceback.format_exc())
            return None
    
    if not sector_list:
        logger.error("未能获取到任何板块数据")
        return None
    
    # 加载当前日期的板块特征
    current_features = trainer._load_multiple_sectors_features(sector_list, date)
    if current_features is None or current_features.empty:
        logger.warning(f"日期 {date} 无法加载板块特征")
        return None
    
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
    
    # 查找模型文件
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"模型目录 {model_dir} 不存在")
        return None
        
    # 尝试查找模型文件
    model_types = ['xgb_return', 'xgb_rank', 'xgb_clf', 'lgb_return', 'lnn_return']
    
    loaded_models = {}
    for model_type in model_types:
        if model_type.startswith('xgb'):
            model_file = model_path / f"{model_type}_model.json"
            if model_file.exists():
                try:
                    model = xgb.XGBRegressor() if 'return' in model_type or 'rank' in model_type else xgb.XGBClassifier()
                    model.load_model(model_file)
                    loaded_models[model_type] = model
                    logger.info(f"成功加载模型 {model_type}")
                except Exception as e:
                    logger.error(f"加载模型 {model_type} 出错: {e}")
        elif model_type.startswith('lgb'):
            model_file = model_path / f"{model_type}_model.txt"
            if model_file.exists():
                try:
                    model = lgb.Booster(model_file=str(model_file))
                    loaded_models[model_type] = model
                    logger.info(f"成功加载模型 {model_type}")
                except Exception as e:
                    logger.error(f"加载模型 {model_type} 出错: {e}")
        elif model_type.startswith('lnn'):
            model_dir = model_path / f"{model_type}_model"
            if model_dir.exists():
                try:
                    # 尝试导入液体神经网络
                    sys.path.append(str(get_project_root() / "to"))
                    from Liquid.nn.liquid.liquid_network import LiquidNetModel
                    model = LiquidNetModel.load(model_dir)
                    loaded_models[model_type] = model
                    logger.info(f"成功加载模型 {model_type}")
                except Exception as e:
                    logger.error(f"加载模型 {model_type} 出错: {e}")
    
    if not loaded_models:
        logger.error(f"未能加载任何模型")
        return None
    
    # 加载特征列名
    feature_file = model_path / "feature_columns.json"
    if not feature_file.exists():
        logger.error(f"特征列文件 {feature_file} 不存在")
        return None
        
    with open(feature_file, "r") as f:
        feature_cols = json.load(f)
    
    # 准备特征数据
    X = current_features[feature_cols].fillna(0)
    
    # 使用模型进行预测
    predictions = {}
    for model_type, model in loaded_models.items():
        try:
            if model_type.startswith('xgb'):
                preds = model.predict(X)
                predictions[model_type] = preds
            elif model_type.startswith('lgb'):
                preds = model.predict(X)
                predictions[model_type] = preds
            elif model_type.startswith('lnn'):
                preds = model.predict(X.values).flatten()
                predictions[model_type] = preds
        except Exception as e:
            logger.error(f"使用模型 {model_type} 预测出错: {e}")
            logger.error(traceback.format_exc())
    
    # 计算综合预测分数
    if not predictions:
        logger.error("没有任何模型预测成功")
        return None
    
    # 优先使用排名模型的结果
    if 'xgb_rank' in predictions:
        # 排名百分比越低越好
        current_features['predicted_rank'] = predictions['xgb_rank']
        rank_based_score = 1 - predictions['xgb_rank']  # 转换为分数，越高越好
    else:
        rank_based_score = None
    
    # 使用回归模型的结果
    if 'xgb_return' in predictions:
        current_features['predicted_return'] = predictions['xgb_return']
        return_score = predictions['xgb_return']
    elif 'lgb_return' in predictions:
        current_features['predicted_return'] = predictions['lgb_return']
        return_score = predictions['lgb_return']
    else:
        return_score = None
    
    # 使用分类模型的结果
    if 'xgb_clf' in predictions:
        current_features['predicted_probability'] = predictions['xgb_clf']
        clf_score = predictions['xgb_clf']
    else:
        clf_score = None
    
    # 计算综合分数
    # 将各个模型的分数归一化后组合
    scores = []
    
    if rank_based_score is not None:
        rank_min = rank_based_score.min()
        rank_max = rank_based_score.max()
        if rank_max > rank_min:
            norm_rank_score = (rank_based_score - rank_min) / (rank_max - rank_min)
            scores.append(norm_rank_score)
    
    if return_score is not None:
        return_min = return_score.min()
        return_max = return_score.max()
        if return_max > return_min:
            norm_return_score = (return_score - return_min) / (return_max - return_min)
            scores.append(norm_return_score)
    
    if clf_score is not None:
        scores.append(clf_score)  # 分类概率已经是[0,1]区间
    
    if scores:
        # 计算平均分数
        composite_score = np.mean(np.array(scores), axis=0)
        current_features['composite_score'] = composite_score
        
        # 按综合分数排序
        current_features['rank'] = current_features['composite_score'].rank(ascending=False)
        
        # 获取前N名板块
        top_sectors = current_features.sort_values('composite_score', ascending=False).head(top_n)
        
        result = {
            'date': date,
            'forward_days': forward_days,
            'top_sectors': []
        }
        
        for _, row in top_sectors.iterrows():
            sector_info = {
                'sector_code': row['sector_code'],
                'composite_score': float(row['composite_score']),
                'rank': int(row['rank'])
            }
            
            if 'predicted_return' in row:
                sector_info['predicted_return'] = float(row['predicted_return'])
            
            if 'predicted_probability' in row:
                sector_info['predicted_probability'] = float(row['predicted_probability'])
                
            result['top_sectors'].append(sector_info)
        
        logger.info(f"预测结果: 日期 {date} 未来 {forward_days} 天表现最好的前 {top_n} 个板块:")
        for sector in result['top_sectors']:
            logger.info(f"  {sector['sector_code']}: 综合分数 {sector['composite_score']:.4f}, 排名 {sector['rank']}")
            
        return result
    else:
        logger.error("无法计算综合分数")
        return None

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
    model_dir = args.model_dir if args.model_dir else get_model_path()
    if args.train_mode == 'rotation':
        model_dir = Path(model_dir) / args.model_name
    
    # 根据运行模式选择功能
    if args.mode == 'train':
        # 训练模式
        # 设置默认日期
        if not args.start_date:
            args.start_date = '20230501'
        if not args.end_date:
            args.end_date = '20230610'
        
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