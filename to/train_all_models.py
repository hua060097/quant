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
            cal_df = storage.get_trading_calendar('202330401', '20250317')
            
            if cal_df is not None and not cal_df.empty:
                if 'date' in cal_df.columns:
                    self.trading_days = cal_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    self.trading_days = cal_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
                    
                self.logger.info(f"成功加载交易日历，共 {len(self.trading_days)} 个交易日")
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
            # 构建文件路径 - 按板块名称划分目录
            # 首选加载优化后的主特征文件
            main_features_path = self.sector_features_dir / sector_code / f"{sector_code}_features.parquet"
            
            if main_features_path.exists():
                # 如果主特征文件存在，直接读取并筛选日期
                self.logger.debug(f"尝试从主特征文件加载板块 {sector_code} 的数据")
                df = pd.read_parquet(main_features_path)
                
                # 筛选指定日期
                if 'date' in df.columns:
                    date_df = df[df['date'] == date].copy()
                    if not date_df.empty:
                        self.logger.debug(f"从主特征文件成功加载板块 {sector_code} 日期 {date} 的数据")
                        return date_df
                elif 'date_str' in df.columns:
                    date_df = df[df['date_str'] == date].copy()
                    if not date_df.empty:
                        self.logger.debug(f"从主特征文件成功加载板块 {sector_code} 日期 {date} 的数据")
                        return date_df
            
            # 如果主特征文件不存在或未包含指定日期，尝试加载单日期文件
            single_date_path = self.sector_features_dir / sector_code / f"{sector_code}_{date}.parquet"
            
            if single_date_path.exists():
                self.logger.debug(f"尝试从单日期文件加载板块 {sector_code} 的数据")
                df = pd.read_parquet(single_date_path)
                if not df.empty:
                    self.logger.debug(f"从单日期文件成功加载板块 {sector_code} 日期 {date} 的数据")
                    return df
            
            # 尝试加载原始合并特征文件
            merged_file_path = self.sector_features_dir / sector_code / f"{sector_code}.parquet"
            
            if merged_file_path.exists():
                self.logger.debug(f"尝试从原始合并文件加载板块 {sector_code} 的数据")
                df = pd.read_parquet(merged_file_path)
                
                # 筛选指定日期
                if 'date' in df.columns:
                    date_df = df[df['date'] == date].copy()
                    if not date_df.empty:
                        self.logger.debug(f"从原始合并文件成功加载板块 {sector_code} 日期 {date} 的数据")
                        return date_df
                elif 'date_str' in df.columns:
                    date_df = df[df['date_str'] == date].copy()
                    if not date_df.empty:
                        self.logger.debug(f"从原始合并文件成功加载板块 {sector_code} 日期 {date} 的数据")
                        return date_df
            
            # 如果所有尝试都失败，则报告错误
            self.logger.warning(f"未找到板块 {sector_code} 日期 {date} 的特征数据")
            self.logger.warning(f"尝试路径: {main_features_path}, {single_date_path}, {merged_file_path}")
            return None
                
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 日期 {date} 的特征数据出错: {e}")
            self.logger.error(traceback.format_exc())
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
            self.logger.error(f"加载日期 {date} 的市场状态数据出错: {e}")
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

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练板块轮动模型")
    parser.add_argument("--start_date", type=str, help="开始日期，格式为YYYYMMDD")
    parser.add_argument("--end_date", type=str, help="结束日期，格式为YYYYMMDD")
    parser.add_argument("--lookback_short", type=int, default=20, help="短期回溯天数")
    parser.add_argument("--lookback_long", type=int, default=60, help="长期回溯天数")
    parser.add_argument("--forward_days", type=int, default=2, help="预测未来几天")
    parser.add_argument("--sector", type=str, help="指定板块代码（不指定则训练所有板块）")
    parser.add_argument("--max_workers", type=int, default=None, help="最大进程数")
    parser.add_argument("--use_lnn", action="store_true", help="是否使用液体神经网络")
    parser.add_argument("--output_dir", type=str, help="模型输出目录")
    
    args = parser.parse_args()
    
    # 设置默认日期
    if not args.start_date:
        args.start_date = '20230101'
    if not args.end_date:
        args.end_date = '20230331'
    
    # 创建训练器
    trainer = SectorModelTrainer(model_dir=args.output_dir, use_lnn=args.use_lnn)
    
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
    
    logger.info(f"将训练 {len(sector_list)} 个板块的模型: {sector_list}")
    
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

if __name__ == "__main__":
    main() 