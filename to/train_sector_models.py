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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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
import os
import time
import math
import inspect
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier

# 导入logger配置
from to.logger_config import configure_logger

# 将原来的logger配置替换为新的配置
logger = configure_logger(__name__, 'train_sector_models.log')


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

class FeatureProcessor:
    """
    特征处理器类 - 统一处理训练、验证、测试和预测阶段的特征
    保证特征类型一致性和完整性
    """
    
    def __init__(self):
        """初始化特征处理器"""
        # 特征元数据
        self.feature_metadata = {
            'categorical_features': [],        # 分类特征列表
            'categorical_mappings': {},        # 分类特征的类别映射
            'required_features': [],           # 必需特征列表
            'feature_defaults': {},            # 特征默认值
            'numeric_features': [],            # 数值特征列表
            'date_features': []                # 日期特征列表
        }
        self.logger = logger
        # 标记是否已初始化
        self.is_initialized = False
    
    def initialize_from_data(self, df, categorical_cols=None):
        """
        从数据中学习特征元数据
        
        Args:
            df: 训练数据DataFrame
            categorical_cols: 显式指定的分类特征列表
        """
        if self.is_initialized:
            return
            
        if df is None or df.empty:
            return
            
        # 记录所有列
        all_columns = df.columns.tolist()
        self.feature_metadata['required_features'] = all_columns
        
        # 识别和处理分类特征
        if categorical_cols is not None:
            self.feature_metadata['categorical_features'] = categorical_cols
        else:
            # 自动识别分类特征
            for col in all_columns:
                if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'object':
                    self.feature_metadata['categorical_features'].append(col)
        
        # 记录分类特征的可能值
        for col in self.feature_metadata['categorical_features']:
            if col in df.columns:
                # 对于分类特征，记录所有唯一值
                unique_values = df[col].dropna().unique().tolist()
                self.feature_metadata['categorical_mappings'][col] = unique_values
                
                # 特别处理market_group - 确保包含基本类别
                if col == 'market_group' and 'market_group' in df.columns:
                    basic_categories = ['bull', 'bear', 'neutral']
                    for cat in basic_categories:
                        if cat not in unique_values:
                            unique_values.append(cat)
                    self.feature_metadata['categorical_mappings'][col] = unique_values
        
        # 识别数值特征
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.feature_metadata['numeric_features'] = [
            col for col in numeric_cols if col not in self.feature_metadata['categorical_features']
        ]
        
        # 记录默认值
        for col in all_columns:
            if col in self.feature_metadata['numeric_features']:
                # 数值列用0填充
                self.feature_metadata['feature_defaults'][col] = 0.0
            elif col in self.feature_metadata['categorical_features']:
                # 分类列使用第一个类别或特殊值
                if col in self.feature_metadata['categorical_mappings'] and self.feature_metadata['categorical_mappings'][col]:
                    self.feature_metadata['feature_defaults'][col] = self.feature_metadata['categorical_mappings'][col][0]
                else:
                    self.feature_metadata['feature_defaults'][col] = "unknown"
        
        # 标记为已初始化
        self.is_initialized = True
        self.logger.info(f"特征处理器初始化完成，处理了 {len(all_columns)} 个特征列，其中分类特征 {len(self.feature_metadata['categorical_features'])} 个")
    
    def initialize_from_metadata(self, metadata):
        """
        从已有元数据初始化特征处理器
        
        Args:
            metadata: 特征元数据字典
        """
        self.feature_metadata = metadata
        self.is_initialized = True
        self.logger.info(f"特征处理器从元数据初始化完成，包含 {len(metadata.get('required_features', []))} 个特征")
    
    def process_features(self, df, is_training=False):
        """
        处理特征数据，确保类型一致性和完整性，但不进行启发式特征生成
        
        Args:
            df: 特征DataFrame
            is_training: 是否为训练阶段
            
        Returns:
            处理后的DataFrame
        """
        if df is None or df.empty:
            return df
            
        # 如果是训练阶段，初始化特征元数据
        if is_training and not self.is_initialized:
            self.initialize_from_data(df)
        
        # 创建副本以避免修改原始数据
        result_df = df.copy()
        
        # 检查缺失的必要基础特征 - 仅填充必要的基础类别和ID特征
        # 模型相关的特征(如lnn_output_feature)将通过特征流水线生成
        missing_features = [
            f for f in self.feature_metadata['required_features'] 
            if f not in result_df.columns
        ]
        
        if missing_features:
            self.logger.info(f"发现 {len(missing_features)} 个缺失特征，填充基础特征")
            
            # 仅填充基础特征，不进行启发式特征生成
            for feature in missing_features:
                if feature == 'market_group':
                    # 默认为bull市场状态
                    result_df['market_group'] = 'bull'
                elif feature in self.feature_metadata['feature_defaults']:
                    # 使用预定义的默认值填充基础特征
                    default_value = self.feature_metadata['feature_defaults'][feature]
                    result_df[feature] = default_value
                    self.logger.debug(f"填充缺失基础特征 {feature} = {default_value}")
                else:
                    # 对于没有预定义默认值的特征，使用0填充基础特征
                    result_df[feature] = 0.0
                    self.logger.warning(f"未找到特征 {feature} 的默认值，使用0填充")
                    
            # 记录高价值特征缺失情况，但不进行启发式生成，而是在特征流水线中处理
            special_features = ['lnn_output_feature', 'predicted_strength', 'market_prob_0', 'market_prob_1', 'market_prob_2']
            special_missing = [f for f in special_features if f in missing_features]
            if special_missing:
                self.logger.warning(f"以下高价值特征缺失，将在特征流水线中处理: {special_missing}")
        
        # 移除交互特征的创建，将由统一的特征流水线处理
        
        # 3. 处理分类特征，特别是market_group
        if 'market_group' in result_df.columns and 'market_group' in self.feature_metadata['categorical_features']:
            # 安全处理market_group特征，确保类型一致性
            self.logger.info("使用安全方式处理market_group特征")
            
            # 检查当前类型
            current_type = result_df['market_group'].dtype
            if isinstance(current_type, pd.CategoricalDtype):
                # 已经是分类类型，不需要处理
                self.logger.info(f"market_group已经是分类类型: {current_type}")
            else:
                # 转换为字符串
                result_df['market_group'] = result_df['market_group'].astype(str)
                
                # 替换特殊值0.0为默认类别bull
                if str(0.0) in result_df['market_group'].values or '0.0' in result_df['market_group'].values:
                    self.logger.info(f"market_group中发现值'0.0'，将替换为'bull'")
                    result_df['market_group'] = result_df['market_group'].replace({'0.0': 'bull', '0': 'bull', 0: 'bull'})
                
                # 获取有效的类别
                categories = self.feature_metadata['categorical_mappings'].get('market_group', ['bull', 'bear', 'neutral'])
                
                # 处理未知类别
                unknown_values = set(result_df['market_group'].dropna().unique()) - set(categories)
                if unknown_values:
                    self.logger.info(f"market_group中发现未知类别: {unknown_values}，将替换为'{categories[0]}'")
                    result_df['market_group'] = result_df['market_group'].apply(
                        lambda x: categories[0] if x in unknown_values else x
                    )
                
                # 转换为分类类型
                self.logger.info("将market_group转换为分类类型")
                try:
                    result_df['market_group'] = pd.Categorical(
                        result_df['market_group'], 
                        categories=categories
                    )
                    self.logger.info(f"成功转换特征 market_group 的数据类型: {current_type} -> {result_df['market_group'].dtype}")
                except Exception as e:
                    self.logger.warning(f"转换market_group为分类类型失败: {e}，将保持字符串类型")
                    result_df['market_group'] = result_df['market_group'].astype(str)
        
        # 4. 确保数值特征是数值类型
        for num_feature in self.feature_metadata['numeric_features']:
            if num_feature in result_df.columns:
                try:
                    result_df[num_feature] = pd.to_numeric(
                        result_df[num_feature], errors='coerce'
                    ).fillna(0.0)
                except Exception as e:
                    # 如果转换失败，使用默认值
                    self.logger.warning(f"转换特征 '{num_feature}' 为数值类型时出错: {e}，将使用默认值0")
                    result_df[num_feature] = 0.0
        
        return result_df


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
        
        # 初始化LiquidNetModel为None
        self.LiquidNetModel = None
        
        # 是否使用液体神经网络
        self.use_lnn = use_lnn
        if use_lnn:
            try:
                print(f"project_root: {get_project_root()}")
                # 尝试导入液体神经网络
                sys.path.append(str(get_project_root() / "to"))
                from Liquid.nn.liquid.liquid_network import LiquidNetwork, LiquidNetModel
                self.LiquidNetModel = LiquidNetModel
                self.logger.info("成功导入液体神经网络模型")
            except ImportError as e:
                self.logger.warning(f"导入液体神经网络失败: {e}，将使用传统模型")
                self.use_lnn = False
            except Exception as e:
                self.logger.warning(f"初始化液体神经网络时出错: {e}，将使用传统模型")
                self.use_lnn = False
        
        # 交易日历
        self.trading_days = []
        self._load_trading_calendar()
        
        # 添加特征缓存目录
        self.features_cache_dir = Path(get_data_path(), "features_cache")
        ensure_dir(self.features_cache_dir)
        
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
            print(f"发生错误: {e}")
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
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 在日期 {date} 的特征数据时出错: {e}")
            return None

    def _load_all_market_status(self):
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
            return market_status_data   
                
        except Exception as e:
            self.logger.error(f"加载市场状态数据时出错: {e}")
            self.logger.exception(e)
            return None
    
    def _add_dynamic_features(self, features_df, market_status):
        """
        基于市场环境动态添加或增强特征
        
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
        计算时间衰减权重 (已废弃，保留兼容性)
        
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
        
        # 保存模型组件 - 确保scaler被保存
        if 'scaler' in models:
            try:
                import joblib
                scaler_file = sector_dir / "scaler.joblib"
                joblib.dump(models['scaler'], scaler_file)
                self.logger.info(f"成功保存scaler到 {scaler_file}")
            except Exception as e:
                self.logger.error(f"保存scaler时出错: {e}")
                self.logger.error(traceback.format_exc())
        
        # 保存模型元数据
        metadata = {
            'feature_columns': feature_columns,
            'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'components': list(models.keys()),
            'seq_length': models.get('seq_length', 10),
            'categorical_features': models.get('categorical_features', []),
            'trained_cat_dtypes': models.get('trained_cat_dtypes', {}),
            'label_encoders': models.get('label_encoders', {})
        }
        
        # 保存市场模型和强度模型的特征
        if 'market_model_features' in models:
            metadata['market_model_features'] = models['market_model_features']
        if 'strength_model_features' in models:
            metadata['strength_model_features'] = models['strength_model_features']
        
        # 添加其他重要元数据
        if 'trained_features' in models:
            metadata['trained_features'] = models['trained_features']
        if 'feature_importance' in models:
            metadata['feature_importance'] = models['feature_importance']
        
        # 保存元数据
        metadata_file = sector_dir / "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 保存每个模型
        for model_name, model in models.items():
            # 跳过已处理的非模型组件
            if model_name in ['scaler', 'market_model_features', 'strength_model_features', 'seq_length']:
                continue
                
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

    def train_rotation_model(self, sector_list=None, start_date=None, end_date=None, model_name="sector_rotation_model", use_lnn=True):
        """
        训练板块轮动模型
        
        修改为滚动训练逻辑，使用一年(252个交易日)的训练窗口，每次滚动一天
        
        Args:
            sector_list: 板块代码列表
            start_date: 开始日期
            end_date: 结束日期
            model_name: 模型名称
            use_lnn: 是否使用LiquidNet网络
        
        Returns:
            训练好的模型包
        """
        try:
            # 0. 初始化
            if not model_name:
                model_name = "sector_rotation_model"
                
            # 导入必要的库
            import json
            import joblib
            
            # 设置训练参数
            target_cols = ['future_return_2d']  # 使用未来收益率作为目标
            ensure_dir(self.model_dir)  # 确保模型目录存在
            
            # 初始化summary_results变量，避免后续引用错误
            summary_results = {
                'training_dates': [],
                'model_metrics': {},
                'predictions': {}
            }
            
            # 1. 加载特征数据
            self.logger.info(f"加载板块特征数据: {start_date} 至 {end_date}...")
            features_df = self._batch_load_features(sector_list=sector_list, start_date=start_date, end_date=end_date)
            
            # 检查日期范围
            if features_df.empty:
                self.logger.error("无可用特征数据!")
                return None
            
            features_df['date'] = pd.to_datetime(features_df['date'])
            unique_dates = sorted(features_df['date'].unique())
            
            # 设置滚动训练的参数
            self.logger.info("设置滚动训练参数...")
            rolling_window_days = 252  # 用一年的数据作为训练窗口
            rolling_step = 1  # 每次滚动一天
            
            # 设置训练日期范围：从2025年1月的第一个交易日到数据中最后日期的倒数第三天
            # 开始日期：2025年1月的第一个交易日
            target_first_date = pd.to_datetime('2025-01-01')
            valid_start_dates = [d for d in unique_dates if d >= target_first_date]
            if not valid_start_dates:
                self.logger.error(f"数据中不包含2025年1月或之后的交易日!")
                return None
            
            # 结束日期：考虑到需要未来两天的收益进行评估，确保截止日期的科学性
            last_date = max(unique_dates)
            valid_unique_dates = sorted(unique_dates)
            if len(valid_unique_dates) <= 3:
                self.logger.error(f"数据中交易日数量不足，无法设置结束日期")
                return None
            
            # 查找每个日期的未来收益数据可用性
            future_return_available = {}
            for date in valid_unique_dates:
                # 检查该日期是否有future_return_2d数据
                date_df = features_df[features_df['date'] == date]
                if not date_df.empty and 'future_return_2d' in date_df.columns:
                    # 检查future_return_2d是否为NaN
                    future_return_available[date] = not date_df['future_return_2d'].isna().all()
                else:
                    future_return_available[date] = False
            
            # 找到最后一个有未来收益数据的日期作为终止日期
            available_dates = [date for date, has_data in future_return_available.items() if has_data]
            if not available_dates:
                self.logger.error("没有找到有未来收益数据的日期，无法进行训练")
                return None
            
            end_train_date = max(available_dates)
            self.logger.info(f"设置训练结束日期为: {end_train_date}，这是最后一个有完整未来收益数据的日期")
            
            # 确定最终的训练日期范围
            first_train_date = valid_start_dates[0]
            self.logger.info(f"训练日期范围: {first_train_date} 至 {end_train_date}, 数据开始日期: {min(unique_dates)}, 数据结束日期: {last_date}")
            
            # 处理market_risk_level列不是数值的问题
            if 'market_risk_level' in features_df.columns and features_df['market_risk_level'].dtype == 'object':
                self.logger.info("转换market_risk_level列...")
                risk_map = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
                features_df['market_risk_level'] = features_df['market_risk_level'].apply(lambda x: risk_map.get(str(x).lower(), 1) if isinstance(x, str) else x)
            
            # 分析可用数据的连续性
            unique_dates_list = sorted(unique_dates)
            date_gaps = []
            for i in range(1, len(unique_dates_list)):
                gap = (unique_dates_list[i] - unique_dates_list[i-1]).days
                if gap > 3:  # 记录超过3天的间隔
                    date_gaps.append((unique_dates_list[i-1], unique_dates_list[i], gap))

            if date_gaps:
                self.logger.warning(f"数据连续性分析: 发现 {len(date_gaps)} 处日期间隔大于3天:")
                for prev_date, next_date, gap in date_gaps[:5]:  # 只显示前5个间隔
                    self.logger.warning(f"  间隔: {prev_date} 至 {next_date} (相差 {gap} 天)")
                if len(date_gaps) > 5:
                    self.logger.warning(f"  ... 以及其他 {len(date_gaps) - 5} 处间隔")
            else:
                self.logger.info("数据连续性分析: 数据分布较为连续，无大间隔")
            
            # 5. 准备滚动训练
            rolling_results = {
                'models': {},
                'predictions': {},
                'metrics': {},
                'dates': []
            }
            
            # 使用未来2天收益作为标记top performer的依据
            if 'future_return_2d' not in features_df.columns:
                self.logger.error("特征数据中缺少future_return_2d列，无法进行滚动训练")
                return None
                
            # 滚动训练每一天的模型：从开始日期到结束日期
            train_dates = [d for d in unique_dates if d >= first_train_date and d <= end_train_date]
            
            # 确保训练日期之前有足够的历史数据
            valid_train_dates = []
            for train_date in train_dates:
                # 获取训练日期前252个交易日的最早日期
                train_date_obj = pd.to_datetime(train_date)
                earliest_required_date = train_date_obj - pd.Timedelta(days=252*1.5)  # 留更多余量
                
                # 检查是否有足够的历史数据
                history_dates = [d for d in unique_dates if d >= earliest_required_date and d < train_date_obj]
                if len(history_dates) >= 150:  # 至少需要150个交易日作为历史数据
                    valid_train_dates.append(train_date)
                else:
                    self.logger.warning(f"训练日期 {train_date} 之前没有足够的历史数据（仅有 {len(history_dates)} 天），跳过")
                    # 记录可用历史数据的详细信息
                    if history_dates:
                        self.logger.warning(f"  可用历史数据范围: {min(history_dates)} 至 {max(history_dates)} (共 {len(history_dates)} 天)")
            
            date_range = valid_train_dates  # 更新date_range变量，用于后续的预测和其他操作
            total_days = len(valid_train_dates)
            
            self.logger.info(f"将对 {total_days} 个交易日进行滚动训练")
            
            for idx, train_date in enumerate(valid_train_dates):
                self.logger.info(f"滚动训练进度: {idx+1}/{total_days}, 日期: {train_date}")
                
                # 训练该日期的模型
                model_package = self._train_rolling_day_model(
                    features_df=features_df,
                    train_date=train_date,
                    training_window_days=rolling_window_days,
                    model_name=f"{model_name}_rolling",
                    sector_list=sector_list
                )
                
                if model_package:
                    # 保存模型
                    rolling_results['models'][train_date.strftime('%Y%m%d')] = model_package
                    rolling_results['dates'].append(train_date.strftime('%Y%m%d'))
                    
                    # 使用模型对当天进行预测
                    train_date_str = train_date.strftime('%Y%m%d')
                    prediction_results = self._predict_with_rolling_model(
                        model_package=model_package,
                        features_df=features_df,
                        prediction_date=train_date
                    )
                    
                    # 修复：检查prediction_results是字典还是DataFrame
                    if prediction_results is not None:
                        # 如果是字典（预期的结构）
                        if isinstance(prediction_results, dict):
                            rolling_results['predictions'][train_date_str] = prediction_results
                            
                            # 计算当天预测的准确率
                            if 'metrics' in prediction_results:
                                rolling_results['metrics'][train_date_str] = prediction_results['metrics']
                        # 如果是DataFrame（兼容可能的直接返回DataFrame的情况）
                        elif isinstance(prediction_results, pd.DataFrame) and not prediction_results.empty:
                            rolling_results['predictions'][train_date_str] = {'predictions': prediction_results}
                        else:
                            self.logger.warning(f"日期 {train_date_str} 的预测结果格式不正确或为空")
            
            # 汇总滚动训练结果
            summary = self._summarize_rolling_results(rolling_results)
            
            # 初始化summary_results变量
            summary_results = {
                'training_dates': rolling_results.get('dates', []),
                'model_metrics': summary,
                'predictions': {}
            }
            
            # 处理predictions部分，确保预测结果也被保存
            for date_str, pred_data in rolling_results.get('predictions', {}).items():
                # 如果是完整的预测数据结构（包含predictions、metrics等）
                if isinstance(pred_data, dict) and 'predictions' in pred_data and isinstance(pred_data['predictions'], pd.DataFrame):
                    # 提取关键预测结果摘要
                    pred_df = pred_data['predictions']
                    top_n = min(10, len(pred_df))
                    
                    # 只保存排名前列的预测结果
                    if not pred_df.empty and 'sector_code' in pred_df.columns:
                        summary_results['predictions'][date_str] = {
                            'top_sectors': pred_df.head(top_n)['sector_code'].tolist(),
                            'prediction_scores': pred_df.head(top_n)['prediction'].tolist() if 'prediction' in pred_df.columns else [],
                            'model_date': date_str  # 使用相同日期作为模型日期
            }
            
            # 保存汇总结果
            rolling_summary_path = self.model_dir / f"{model_name}_rolling_summary.json"
            with open(rolling_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_results, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"滚动训练完成，汇总结果已保存到 {rolling_summary_path}")
            return rolling_results
                
        except Exception as e:
            print(f"发生错误: {e}")
            self.logger.error(f"训练板块轮动模型出错: {e}")
            traceback.print_exc()
            return None
       
           


    def _batch_load_features(self, sector_list, start_date, end_date):
        """批量加载多个板块多个日期的特征数据
        
        Args:
            sector_list: 板块列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 所有特征数据
        """
        # 添加缓存机制，避免重复加载和处理数据
        cache_dir = os.path.join(get_project_root(), "to", "data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件名，基于输入参数
        sector_str = "-".join(sorted(sector_list))
        cache_key = f"{sector_str[:100]}-{start_date}-{end_date}"
        import hashlib
        # cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_hash = "9953d2f3ca428eebeeaf18e6f986f03f"
        cache_file = os.path.join(cache_dir, f"batch_features_{cache_hash}.pkl")
        
        # 检查是否存在缓存
        if os.path.exists(cache_file):
            try:
                self.logger.info(f"发现缓存文件 {cache_file}，尝试加载...")
                with open(cache_file, 'rb') as f:
                    df_all = pickle.load(f)
                self.logger.info(f"成功从缓存加载特征数据，形状: {df_all.shape}")
                return df_all
            except Exception as e:
                self.logger.warning(f"加载缓存文件失败: {e}，将重新生成特征")
        
        # 确保有交易日历
        if self.trading_days is None:
            self._load_trading_calendar()
            
        # 过滤交易日
        trading_days_in_range = [day for day in self.trading_days if start_date <= day <= end_date]
        self.logger.info(f"在 {start_date} 至 {end_date} 期间共有 {len(trading_days_in_range)} 个交易日")
        
        # 先加载所有板块在整个日期范围的基础数据，以便缓存
        self.logger.info(f"开始加载 {len(sector_list)} 个板块的基础数据...")
        base_features = {}
        
        # 使用多线程并行加载基础数据
        with ThreadPoolExecutor(max_workers=min(10, len(sector_list))) as executor:
            futures = {
                executor.submit(self._load_base_features, sector, start_date, end_date): sector 
                for sector in sector_list
            }
            
            for future in as_completed(futures):
                sector = futures[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        base_features[sector] = result
                        self.logger.info(f"成功加载板块 {sector} 的基础数据，共 {len(result)} 条记录")
                    else:
                        self.logger.warning(f"板块 {sector} 没有找到基础数据")
                except Exception as e:
                    self.logger.error(f"加载板块 {sector} 基础数据时出错: {e}")
        
        # 如果没有任何数据，提前返回
        if not base_features:
            self.logger.warning("没有找到任何板块数据")
            return pd.DataFrame()
            
        self.logger.info(f"成功加载 {len(base_features)}/{len(sector_list)} 个板块的基础数据")
        
        # 提前加载市场环境数据
        self.logger.info("开始加载市场环境数据...")
        market_status_cache = self._load_all_market_status()
        self.logger.info(f"成功加载 {len(market_status_cache)} 个日期的市场环境数据")
        
        # 提前计算季节性特征 - 对每个板块
        self.logger.info("开始计算季节性特征...")
        seasonal_features = {}
        for sector, data in base_features.items():
            try:
                seasonal_features[sector] = self._compute_seasonal_patterns(sector, base_data=data)
                self.logger.debug(f"成功计算板块 {sector} 的季节性特征")
            except Exception as e:
                self.logger.error(f"计算板块 {sector} 季节性特征时出错: {e}")
                seasonal_features[sector] = {}
        
        # 准备所有任务 - 每个板块每个日期一个任务
        tasks = []
        for sector in base_features.keys():
            for date in trading_days_in_range:
                date_obj = pd.to_datetime(date)
                if date_obj in base_features[sector].index:
                    tasks.append((sector, date))
                    
        self.logger.info(f"共找到 {len(tasks)} 个特征构建任务")
        
        # 使用多线程处理每个板块-日期组合
        all_features = []
        futures = {}
        # 导入板块龙头股分析器
        from to.sector_leading_stock import SectorLeadingStocksAnalyzer        
        # 初始化分析器
        leading_analyzer = SectorLeadingStocksAnalyzer()
        
        with ThreadPoolExecutor(max_workers=min(20, len(tasks))) as executor:
            for sector, date in tasks:
                date_obj = pd.to_datetime(date)
                
                # 获取市场环境数据
                market_data = market_status_cache.get(date, None)
                
                # 获取板块季节特征
                seasonal_data = seasonal_features.get(sector, {})
                
                # 提交任务
                future = executor.submit(
                    self._process_features_with_layers,
                    sector_code=sector,
                    date=date_obj,
                    base_data=base_features[sector],
                    market_status=market_data,
                    seasonal_data=seasonal_data,
                    leading_analyzer=leading_analyzer
                )
                
                futures[future] = (sector, date)
            
            # 收集结果
            for i, future in enumerate(as_completed(futures)):
                sector_code, date = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # 不再需要再次计算未来收益率，因为已经在_process_features_with_layers中完成
                        all_features.append(result)
                    
                    # 进度日志
                    if (i+1) % 100 == 0 or (i+1) == len(tasks):
                        self.logger.info(f"已处理 {i+1}/{len(tasks)} 个特征构建任务")
                except Exception as e:
                    self.logger.error(f"处理板块 {sector_code} 日期 {date} 的特征时出错: {e}")
        
        self.logger.info(f"批量加载完成，成功构建 {len(all_features)} 条特征记录")
        
        if not all_features:
            self.logger.warning("没有生成任何有效特征")
            return pd.DataFrame()
        
        # 合并所有特征
        try:
            # 合并所有DataFrame
            df_all = pd.concat(all_features, ignore_index=True)
            
            # 添加日期字段(如果没有)
            if 'date' in df_all.columns and not pd.api.types.is_datetime64_any_dtype(df_all['date']):
                df_all['date'] = pd.to_datetime(df_all['date'])
                
            # 按日期排序
            if 'date' in df_all.columns:
                df_all = df_all.sort_values('date')
                
            self.logger.info(f"特征构建完成，共 {len(df_all)} 条记录，{df_all.shape[1]} 个特征")
            
            # 保存到缓存
            try:
                self.logger.info(f"保存特征数据到缓存 {cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump(df_all, f)
                self.logger.info(f"特征数据缓存保存成功")
                
                
                
            except Exception as e:
                self.logger.warning(f"保存特征数据到缓存失败: {e}")
            
            return df_all
            
        except Exception as e:
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"合并特征数据时出错: {e}")
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
                print(f"发生错误: {e}")
            except Exception as e:
                self.logger.error(f"读取 {main_features_path} 出错: {e}")
                return None
                
        except Exception as e:
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"加载板块 {sector_code} 基础数据时发生错误: {e}")
            return None

    def _process_features_with_layers(self, sector_code, date, base_data, market_status, seasonal_data, leading_analyzer=None):
        """使用三层特征体系处理单个板块单个日期的特征
        
        Args:
            sector_code: 板块代码
            date: 日期
            base_data: 基础层数据
            market_status: 市场环境层数据
            seasonal_data: 季节性特征数据
            leading_analyzer: 龙头股分析数据
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
            # 显式设置market_group特征，无论是否有市场状态数据
            if market_status:
                # 添加市场类型和周期指标
                features["market_type"] = market_status.get("market_type", "unknown")
                features["market_cycle"] = market_status.get("market_cycle", 0)
                features["market_trend_strength"] = market_status.get("trend_strength", 0)
                
                # 显式设置market_group特征，将market_type映射到bull/bear/neutral
                market_type = market_status.get("market_type", 0)
                if isinstance(market_type, (int, float)):
                    # 根据market_type值划分市场状态
                    if market_type in [11, 12, 13, 21, 22, 23]:  # 上涨趋势类型
                        features["market_group"] = "bull"
                    elif market_type in [41, 42, 43, 51, 52, 53]:  # 下跌趋势类型
                        features["market_group"] = "bear"
                    else:  # 震荡或不确定趋势类型
                        features["market_group"] = "neutral"
                else:
                    # 默认为neutral市场
                    features["market_group"] = "neutral"
            else:
                # 如果没有市场状态数据，使用默认值
                features["market_type"] = "unknown"
                features["market_cycle"] = 0
                features["market_trend_strength"] = 0
                features["market_group"] = "neutral"  # 使用默认的中性市场
            
            # 添加市场信号
            signals = market_status.get("signals", {})
            features["market_risk_level"] = signals.get("risk_level", "medium")
            # 将market_risk_level转换为数值类型
            risk_level_map = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
            if isinstance(features["market_risk_level"], str):
                features["market_risk_level"] = risk_level_map.get(features["market_risk_level"].lower(), 1)
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
            
            # 新增: 5. 板块龙头股特征层 - 添加龙头股相关特征
            # ------------------------------------------------------------
            try:
                
                # 尝试加载板块龙头股特征
                leading_features = leading_analyzer.load_leading_stocks_features(sector_code)
                
                # 如果找到特征数据，添加到特征字典中
                if leading_features is not None and not leading_features.empty:
                    # 找到最接近当前日期的数据
                    if 'date' in leading_features.columns:
                        # 确保日期格式一致
                        leading_features['date'] = pd.to_datetime(leading_features['date'])
                        date_obj = pd.to_datetime(date)
                        
                        # 找到最近的日期记录
                        closest_date_idx = (leading_features['date'] - date_obj).abs().idxmin()
                        leading_data = leading_features.loc[closest_date_idx]
                        
                        # 添加龙头股数量和比例特征
                        features['leading_stock_count'] = leading_data.get('leading_stock_count', 0)
                        features['leading_stock_ratio'] = leading_data.get('leading_stock_ratio', 0)
                        
                        # 添加涨停股票特征
                        features['limit_up_count'] = leading_data.get('limit_up_count', 0)
                        features['limit_up_ratio'] = leading_data.get('limit_up_ratio', 0)
                        features['multi_limit_count'] = leading_data.get('multi_limit_count', 0)
                        features['multi_limit_ratio'] = leading_data.get('multi_limit_ratio', 0)
                        features['broken_limit_count'] = leading_data.get('broken_limit_count', 0)
                        features['broken_limit_ratio'] = leading_data.get('broken_limit_ratio', 0)
                        
                        # 添加连板天数特征
                        features['max_consecutive_limit_days'] = leading_data.get('max_consecutive_limit_days', 0)
                        features['avg_consecutive_limit_days'] = leading_data.get('avg_consecutive_limit_days', 0)
                        
                        # 添加高弹性标的特征
                        features['high_elasticity_count'] = leading_data.get('high_elasticity_count', 0)
                        features['high_elasticity_ratio'] = leading_data.get('high_elasticity_ratio', 0)
                        
                        # 添加换手率特征
                        features['high_turnover_count'] = leading_data.get('high_turnover_count', 0)
                        features['high_turnover_ratio'] = leading_data.get('high_turnover_ratio', 0)
                        features['avg_turnover'] = leading_data.get('avg_turnover', 0)
                        
                        # 添加热度指标
                        features['heat_index'] = leading_data.get('heat_index', 0)
                        
                        # 添加市值分布特征
                        features['large_cap_ratio'] = leading_data.get('large_cap_ratio', 0)
                        features['mid_cap_ratio'] = leading_data.get('mid_cap_ratio', 0)
                        features['small_cap_ratio'] = leading_data.get('small_cap_ratio', 0)
            except Exception as e:
                print(f"发生错误: {e}")
            except Exception as e:
                self.logger.error(f"加载板块 {sector_code} 的龙头股特征时出错: {e}")
                # 继续处理其他特征，不中断
                
            # 6. 计算未来收益率 - 直接使用已加载的base_data
            # ------------------------------------------------------------
            try:
                # 确保交易日历已加载
                if self.trading_days is None:
                    self._load_trading_calendar()
                
                date_obj = pd.to_datetime(date)
                current_date_str = date_obj.strftime('%Y%m%d')
                
                # 确保当前日期在交易日历中
                if current_date_str in self.trading_days:
                    current_idx = self.trading_days.index(current_date_str)
                    current_close = features.get('close')
                    
                    if current_close and current_close > 0:
                        # 计算不同天数的未来收益率
                        for days in [1, 2, 3, 5, 10]:
                            # 检查是否有足够的未来交易日
                            if current_idx + days < len(self.trading_days):
                                future_date_str = self.trading_days[current_idx + days]
                                future_date = pd.to_datetime(future_date_str)
                                
                                # 检查未来日期是否在基础数据中
                                if future_date in base_data.index:
                                    # 获取未来收盘价
                                    future_close = base_data.loc[future_date, 'close']
                                    
                                    # 计算收益率
                                    if not pd.isna(future_close) and future_close > 0:
                                        features[f'future_return_{days}d'] = (future_close - current_close) / current_close
            except Exception as e:
                print(f"发生错误: {e}")
            except Exception as e:
                self.logger.error(f"计算板块 {sector_code} 日期 {date} 的未来收益率时出错: {e}")
                # 继续处理其他特征，不中断
            
            # 创建数据帧并返回
            return pd.DataFrame([features])
        except Exception as e:
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"处理板块 {sector_code} 日期 {date} 特征时错误: {str(e)}")
            traceback.print_exc()
            return None

    
    def _compute_seasonal_patterns(self, sector_code, base_data=None):
        """
        计算板块季节性模式
        
        Args:
            sector_code: 板块代码
            base_data: 预加载的基础数据，如果提供则不再重新加载
                
        Returns:
            dict: 季节性模式数据
        """
        self.logger.debug(f"开始计算板块 {sector_code} 的季节性模式")
        
        try:
            # 使用预加载的数据或加载历史数据
            features_df = None
            
            if base_data is not None and not base_data.empty:
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
            
            # 使用历史收益数据进行季节性计算（而不是未来收益）
            return_column = None
            for col in ['change_percent', 'pct_chg', 'return', 'daily_return']:
                if col in features_df.columns:
                    return_column = col
                    break
            
            if return_column is None:
                self.logger.warning(f"板块 {sector_code} 数据中没有找到收益率列，无法计算季节性模式")
                return {}
            
            # 月度表现 - 计算每个月的平均收益率
            monthly_returns = {}
            for month in range(1, 13):
                month_data = features_df[features_df['month'] == month]
                if len(month_data) > 0:
                    avg_return = month_data[return_column].mean()
                    monthly_returns[month] = float(avg_return)
            
            # 季度表现 - 计算每个季度的平均收益率
            quarterly_returns = {}
            for quarter in range(1, 5):
                quarter_data = features_df[features_df['quarter'] == quarter]
                if len(quarter_data) > 0:
                    avg_return = quarter_data[return_column].mean()
                    quarterly_returns[quarter] = float(avg_return)
            
            # 市场适应性 - 如果有市场类型数据，计算不同市场类型下的表现
            market_fit = {}
            if 'market_type' in features_df.columns:
                market_types = features_df['market_type'].unique()
                for market_type in market_types:
                    market_data = features_df[features_df['market_type'] == market_type]
                    if len(market_data) > 3:  # 确保有足够的样本
                        avg_return = market_data[return_column].mean()
                        market_fit[str(market_type)] = float(avg_return)
            
            # 整合结果
            seasonal_patterns = {
                'monthly': monthly_returns,
                'quarterly': quarterly_returns,
                'market_fit': market_fit
            }
            
            return seasonal_patterns
        except Exception as e:
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"计算板块 {sector_code} 季节性模式时出错: {e}")
            return {}

    def _calculate_future_returns(self, sector_code, date, days_list, all_data=None):
        """
        计算未来收益率
        
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
            
            # 优先使用预加载的全部数据计算未来收益率
            if all_data is not None and not all_data.empty:
                # 使用预加载的数据计算未来收益率
                # 确保数据索引是日期
                df = all_data.copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    else:
                        self.logger.warning(f"无法从预加载数据计算未来收益率: 缺少date列")
                        return {}
                
                # 确保当前日期存在于数据中
                if date not in df.index:
                    self.logger.warning(f"日期 {date} 不在预加载数据中")
                    return {}
                
                # 初始化结果字典
                future_returns = {}
                
                # 获取当前收盘价
                if 'close' not in df.columns:
                    self.logger.warning(f"预加载数据中缺少close列")
                    return {}
                
                # 获取当前收盘价
                try:
                    current_close = df.loc[date, 'close']
                    if pd.isna(current_close) or current_close <= 0:
                        self.logger.warning(f"日期 {date} 的收盘价无效: {current_close}")
                        return {}
                except Exception as e:
                    self.logger.warning(f"获取日期 {date} 收盘价时出错: {e}")
                    return {}
                
                # 计算指定天数的未来收益率
                for days in days_list:
                    # 使用交易日历获取未来日期
                    if self.trading_days is None:
                        self._load_trading_calendar()
                    
                    current_date_str = date.strftime('%Y%m%d')
                    if current_date_str not in self.trading_days:
                        continue
                    
                    current_idx = self.trading_days.index(current_date_str)
                    if current_idx + days >= len(self.trading_days):
                        continue
                    
                    future_date_str = self.trading_days[current_idx + days]
                    future_date = pd.to_datetime(future_date_str)
                    
                    # 检查未来日期是否在数据中
                    if future_date not in df.index:
                        continue
                    
                    # 获取未来收盘价并计算收益率
                    try:
                        future_close = df.loc[future_date, 'close']
                        if pd.isna(future_close) or future_close <= 0:
                            continue
                        
                        future_returns[days] = (future_close - current_close) / current_close
                    except Exception as e:
                        self.logger.debug(f"计算 {days} 天未来收益率时出错: {e}")
                        continue
                
                return future_returns
            else:
                # 如果没有预加载数据，则使用按需加载方式
                # 确保交易日历已加载
                if self.trading_days is None:
                    self._load_trading_calendar()
                
                # 获取当前日期对应的交易日索引
                current_date_str = date.strftime('%Y%m%d')
                if current_date_str not in self.trading_days:
                    self.logger.warning(f"日期 {current_date_str} 不在交易日历中")
                    return {}
                    
                current_idx = self.trading_days.index(current_date_str)
                
                # 初始化结果字典
                future_returns = {}
                
                # 计算指定天数的未来收益率
                for days in days_list:
                    if current_idx + days < len(self.trading_days):
                        future_date = self.trading_days[current_idx + days]
                        
                        # 加载当前日期和未来日期的收盘价
                        current_features = self._load_sector_features(sector_code, current_date_str)
                        future_features = self._load_sector_features(sector_code, future_date)
                        
                        if current_features is not None and future_features is not None:
                            # 获取收盘价
                            current_close = current_features.get('close', 0)
                            future_close = future_features.get('close', 0)
                            
                            # 计算收益率
                            if current_close > 0 and future_close > 0:
                                future_returns[days] = (future_close - current_close) / current_close
                
                return future_returns
        except Exception as e:
            self.logger.error(f"计算板块 {sector_code} 日期 {date} 的未来收益率时出错: {e}")
            print(f"发生错误: {e}")
            return {}


    def _evaluate_on_test_set(self, model_package, test_df, seq_features, date):
        """
        在测试集上评估模型
        
        Args:
            model_package: 模型包
            test_df: 测试集数据
            seq_features: 序列特征字典
            date: 测试日期
            
        Returns:
            评估指标
        """
        self.logger.info(f"评估日期 {date} 的模型表现")
        
        try:
            # 检查模型包的有效性
            if model_package is None:
                self.logger.error("模型包为None，无法评估")
                return None
                
            if not isinstance(model_package, dict):
                self.logger.error(f"模型包类型错误，应为dict，实际为{type(model_package)}")
                return None
            
            # 检查测试集数据
            if test_df is None or test_df.empty:
                self.logger.error("测试集为空，无法评估")
                return None
                
            # 获取特征处理器并应用
            feature_processor = None
            if 'feature_processor' in model_package:
                feature_processor = model_package['feature_processor']
                self.logger.info("使用模型包中的特征处理器")
            elif 'metadata' in model_package:
                # 从元数据创建特征处理器
                self.logger.info("从模型元数据创建特征处理器")
                feature_processor = FeatureProcessor()
                
                metadata = model_package['metadata']
                feature_metadata = {
                    'categorical_features': metadata.get('categorical_features', []),
                    'categorical_mappings': metadata.get('trained_cat_dtypes', {}),
                    'required_features': metadata.get('final_feature_columns', []),
                    'feature_defaults': {},
                    'numeric_features': [],
                    'date_features': []
                }
                
                feature_processor.initialize_from_metadata(feature_metadata)
            
            # 应用特征处理
            if feature_processor and feature_processor.is_initialized:
                self.logger.info("应用特征处理器处理测试数据...")
                test_df = feature_processor.process_features(test_df, is_training=False)
                
                # 特殊处理market_group
                if 'market_group' in test_df.columns and 'metadata' in model_package:
                    metadata = model_package['metadata']
                    if 'trained_cat_dtypes' in metadata and 'market_group' in metadata['trained_cat_dtypes']:
                        categories = metadata['trained_cat_dtypes']['market_group']
                        if categories:
                            try:
                                # 确保market_group是分类类型
                                if not isinstance(test_df['market_group'].dtype, pd.CategoricalDtype):
                                    # 转为字符串处理0.0问题
                                    test_df['market_group'] = test_df['market_group'].astype(str)
                                    
                                    # 处理0.0值
                                    if '0.0' in test_df['market_group'].values:
                                        test_df['market_group'] = test_df['market_group'].replace('0.0', 'bull')
                                        
                                    # 检查并替换未知类别
                                    unknown_values = [x for x in test_df['market_group'].unique() if x not in categories]
                                    if unknown_values:
                                        for val in unknown_values:
                                            test_df['market_group'] = test_df['market_group'].replace(val, categories[0])
                                            
                                    # 转为分类类型
                                    test_df['market_group'] = pd.Categorical(
                                        test_df['market_group'],
                                        categories=categories
                                    )
                            except Exception as e:
                                self.logger.warning(f"转换market_group类型时出错: {e}")
                                # 如果转换失败，保持为字符串类型
                                test_df['market_group'] = test_df['market_group'].astype(str)
            else:
                self.logger.warning("特征处理器不可用或未初始化，将使用原始数据")
                
            # 确保测试集有未来收益数据
            if 'future_return_2d' not in test_df.columns:
                self.logger.error("测试集中缺少future_return_2d收益数据，无法评估模型")
                return None
                
            # 保证数据质量和统计分析
            try:
                self.logger.info("验证未来收益率数据质量")
                # 用于分析的统计数据
                future_return_stats = {
                    'mean': test_df['future_return_2d'].mean(),
                    'std': test_df['future_return_2d'].std(),
                    'min': test_df['future_return_2d'].min(),
                    'max': test_df['future_return_2d'].max(),
                    'positive_ratio': (test_df['future_return_2d'] > 0).mean()
                }
                self.logger.info(f"未来收益率统计: 均值 {future_return_stats['mean']:.4f}, "
                               f"标准差 {future_return_stats['std']:.4f}, " 
                               f"正收益比例 {future_return_stats['positive_ratio']:.2%}")
            except Exception as e:
                self.logger.error(f"处理未来收益率数据时出错: {e}")
                self.logger.error(traceback.format_exc())
                return None
                
            # 检查序列特征
            if seq_features is None:
                self.logger.warning("序列特征为None，将尝试在没有序列特征的情况下评估")
            
            # 使用模型预测
            try:
                prediction = self._predict_with_sequence(model_package, test_df, seq_features, date)
            except Exception as e:
                self.logger.error(f"使用_predict_with_sequence进行预测时出错: {e}")
                self.logger.error(traceback.format_exc())
                return None
                
            if prediction is None:
                self.logger.error("预测结果为None，无法评估")
                return None
                
            # 获取预测的top sectors
            predicted_top_sectors = prediction.get('top_sectors', [])
            if not predicted_top_sectors:
                self.logger.error("预测结果中没有top_sectors字段或为空")
                return None
                
                            # 基于实际收益率确定表现最好的板块（前5个）
            actual_top_sectors_df = test_df.sort_values(by='future_return_2d', ascending=False).head(5)
            actual_top_sectors = actual_top_sectors_df['sector_code'].tolist()
            
            if not actual_top_sectors:
                self.logger.warning("测试集中无法确定实际表现最好的板块")
                return {
                    'mae': 0.0,
                    'rmse': 0.0,
                    'r2': 0.0,
                    'direction_accuracy': 0.0,
                    'predicted_top_sectors': predicted_top_sectors[:5],  # 默认取前5个
                    'actual_top_sectors': [],
                    'top5_avg_return': 0.0,
                    'top5_positive_ratio': 0.0,
                    'date': date.strftime('%Y%m%d') if isinstance(date, (datetime, pd.Timestamp)) else date
                }
            
            # 计算评估指标 - 针对回归模型
            metrics = {}
            
            # 预测与实际收益率的比较
            predicted_df = prediction.get('predictions', None)
            if predicted_df is not None and 'prediction' in predicted_df.columns and 'sector_code' in predicted_df.columns:
                # 合并预测值和实际值
                merged_df = pd.merge(
                    predicted_df[['sector_code', 'prediction']], 
                    test_df[['sector_code', 'future_return_2d']], 
                    on='sector_code', 
                    how='inner'
                )
                
                if not merged_df.empty:
                    # 计算回归指标
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    mae = mean_absolute_error(merged_df['future_return_2d'], merged_df['prediction'])
                    rmse = np.sqrt(mean_squared_error(merged_df['future_return_2d'], merged_df['prediction']))
                    r2 = r2_score(merged_df['future_return_2d'], merged_df['prediction'])
                    
                    # 计算方向准确率（正负号一致）
                    direction_matches = ((merged_df['future_return_2d'] > 0) == (merged_df['prediction'] > 0)).mean()
                    
                    # 计算预测的top5板块的实际平均收益率
                    top5_predicted_codes = predicted_df.head(5)['sector_code'].tolist()
                    top5_actual_returns = test_df[test_df['sector_code'].isin(top5_predicted_codes)]['future_return_2d']
                    top5_avg_return = top5_actual_returns.mean() if not top5_actual_returns.empty else 0
                    top5_positive_ratio = (top5_actual_returns > 0).mean() if not top5_actual_returns.empty else 0
                    
                    metrics['mae'] = mae
                    metrics['rmse'] = rmse
                    metrics['r2'] = r2
                    metrics['direction_accuracy'] = direction_matches
                    metrics['top5_avg_return'] = top5_avg_return
                    metrics['top5_positive_ratio'] = top5_positive_ratio
                else:
                    self.logger.warning("合并预测和实际值后的DataFrame为空")
                    metrics['mae'] = 0.0
                    metrics['rmse'] = 0.0
                    metrics['r2'] = 0.0
                    metrics['direction_accuracy'] = 0.0
                    metrics['top5_avg_return'] = 0.0
                    metrics['top5_positive_ratio'] = 0.0
            else:
                self.logger.warning("预测结果格式不正确，无法计算回归指标")
                metrics['mae'] = 0.0
                metrics['rmse'] = 0.0
                metrics['r2'] = 0.0
                metrics['direction_accuracy'] = 0.0
                metrics['top5_avg_return'] = 0.0
                metrics['top5_positive_ratio'] = 0.0
            
            # 找出预测和实际top 5中的共同板块
            common_sectors = [s for s in predicted_top_sectors[:5] if s in actual_top_sectors]
            
            metrics['common_sectors'] = common_sectors
            metrics['predicted_top_sectors'] = predicted_top_sectors[:5]
            metrics['actual_top_sectors'] = actual_top_sectors
            metrics['date'] = date.strftime('%Y%m%d') if isinstance(date, (datetime, pd.Timestamp)) else date
            
            # 添加板块名称，如果可能的话
            if 'sector_name' in test_df.columns:
                sector_name_map = {}
                for _, row in test_df.iterrows():
                    sector_name_map[row['sector_code']] = row['sector_name'] 
                
                # 定义比较的最大长度
                max_compare_len = min(5, len(predicted_top_sectors))
                # 计算正确预测的板块
                correct_predictions = [s for s in predicted_top_sectors[:max_compare_len] if s in actual_top_sectors]
                
                predicted_names = [sector_name_map.get(code, code) for code in predicted_top_sectors[:max_compare_len]]
                actual_names = [sector_name_map.get(code, code) for code in actual_top_sectors]
                correct_names = [sector_name_map.get(code, code) for code in correct_predictions]
                
                metrics['predicted_names'] = predicted_names
                metrics['actual_names'] = actual_names
                metrics['correct_names'] = correct_names
            
            # 计算评估指标
            accuracy = len(correct_predictions) / len(actual_top_sectors) if actual_top_sectors else 0
            precision = len(correct_predictions) / len(predicted_top_sectors[:max_compare_len]) if predicted_top_sectors else 0
            recall = len(correct_predictions) / len(actual_top_sectors) if actual_top_sectors else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 记录评估指标
            metrics['accuracy'] = accuracy
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            
            self.logger.info(f"日期 {date} 的模型准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
            self.logger.info(f"预测正确的板块: {len(correct_predictions)}/{len(actual_top_sectors)}")
            return metrics
            
        except Exception as e:
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"评估日期 {date} 的模型时出错: {e}")
            self.logger.error(traceback.format_exc())
            
            # 返回一个有效但表示失败的metrics
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'error': str(e),
                'correct_sectors': [],
                'predicted_top_sectors': [],
                'actual_top_sectors': [],
                'date': date.strftime('%Y%m%d') if isinstance(date, (datetime, pd.Timestamp)) else date
            }

    def _apply_feature_pipeline(self, features_df, market_status=None, mode='train'):
        """
        统一的特征流水线，确保训练和预测时特征生成逻辑一致
        
        Args:
            features_df: 输入特征DataFrame
            market_status: 市场状态数据
            mode: 'train' 或 'predict'，指示当前操作模式
            
        Returns:
            处理后的特征DataFrame
        """
        try:
            # 创建副本以避免修改原数据
            result_df = features_df.copy()
            
            # 2. 确保LNN特征存在
            # 注意：在训练模式中，我们暂不生成lnn_output_feature，因为LNN模型尚未训练
            if mode == 'predict' or 'lnn_output_feature' in result_df.columns:
                result_df = self._ensure_lnn_features(result_df)
            
            # 3. 特征标准化和归一化
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            # 排除目标变量和ID类列
            exclude_patterns = ['future_return', 'sector_code', 'date']
            norm_cols = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
            
            # 适用于训练和预测的特征工程
            # 3.1 处理极端值
            for col in norm_cols:
                if col in result_df.columns:
                    # 使用winsorize裁剪极端值至1%和99%分位数
                    try:
                        if result_df[col].nunique() > 10:  # 只处理有足够唯一值的列
                            q_low, q_high = result_df[col].quantile([0.01, 0.99])
                            result_df[col] = result_df[col].clip(q_low, q_high)
                    except Exception as e:
                        self.logger.warning(f"处理列 {col} 的极端值时出错: {e}")
            
            # 3.2 增加重要特征的交互项
            # 注意：这些交互项应在模型训练前确定，以确保训练和预测逻辑一致
            interaction_pairs = [
                ('momentum_5d', 'trend_direction'),
                ('volatility_5d', 'turnover_trend_5d'),
                ('predicted_strength', 'market_sentiment')
            ]
            
            for col1, col2 in interaction_pairs:
                if col1 in result_df.columns and col2 in result_df.columns:
                    interaction_name = f'{col1}_x_{col2}'
                    result_df[interaction_name] = result_df[col1] * result_df[col2]
                    self.logger.debug(f"创建交互特征: {interaction_name}")
            
            # 4. 根据板块代码添加季节性特征
            if 'sector_code' in result_df.columns and 'date' in result_df.columns:
                # 提取月份和星期几
                result_df['month'] = result_df['date'].dt.month
                result_df['weekday'] = result_df['date'].dt.weekday
                
                # 季节性指标：各板块在不同月份和星期的历史表现
                # 注：实际实现中，这应该基于历史数据预先计算
                
            # 返回处理后的特征
            self.logger.info(f"特征流水线完成 (模式: {mode})，特征数: {len(result_df.columns)}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"特征流水线处理出错: {e}")
            self.logger.error(traceback.format_exc())
            return features_df  # 出错时返回原始特征
    
    def _ensure_lnn_features(self, features_df, lnn_model=None):
        """
        确保lnn_output_feature特征存在，不存在则生成
        
        Args:
            features_df: 特征DataFrame
            lnn_model: 可选的LNN模型用于生成高质量特征
            
        Returns:
            包含lnn_output_feature特征的DataFrame
        """
        try:
            # 如果已经有lnn_output_feature，直接返回
            if 'lnn_output_feature' in features_df.columns:
                return features_df
                
            # 创建工作副本
            result_df = features_df.copy()
            
            # 方法1: 使用LNN模型生成特征 - 最高质量的特征来源
            if lnn_model is not None:
                try:
                    # 使用特征序列方法
                    self.logger.info("使用LNN模型生成lnn_output_feature特征")
                    # 根据模型类型进行处理
                    if hasattr(lnn_model, 'predict'):
                        # 准备用于预测的数据
                        lnn_features = [f for f in lnn_model.feature_names_ if f in result_df.columns]
                        lnn_features_ratio = len(lnn_features) / len(lnn_model.feature_names_) if hasattr(lnn_model, 'feature_names_') else 0
                        
                        # 检查特征覆盖率是否足够 - 提高到85%以确保更高质量
                        if lnn_features_ratio >= 0.85:
                            # 填充缺失特征
                            for feat in lnn_model.feature_names_:
                                if feat not in result_df.columns:
                                    result_df[feat] = 0.0
                            
                            # 使用模型预测
                            X_lnn = result_df[lnn_model.feature_names_]
                            result_df['lnn_output_feature'] = lnn_model.predict(X_lnn)
                            self.logger.info(f"成功使用LNN模型生成lnn_output_feature (特征覆盖率: {lnn_features_ratio:.2%})")
                            return result_df
                        else:
                            self.logger.warning(f"LNN模型特征覆盖率不足 ({lnn_features_ratio:.2%}), 寻找其他生成方法")
                except Exception as e:
                    self.logger.warning(f"使用LNN模型生成lnn_output_feature出错: {e}")
            
            # 方法2: 基于量价行为指标计算 (更优先于简单热度特征)
            self.logger.info("尝试使用量价行为指标生成lnn_output_feature")
            price_volume_features = [
                col for col in result_df.columns if any(
                    term in col.lower() for term in [
                        'price', 'volume', 'turnover', 'volatility', 'macd',
                        'rsi', 'kdj', 'bollinger', 'atr', 'obv', 'return'
                    ]
                ) and 'future_return' not in col.lower()
            ]
            
            if len(price_volume_features) >= 5:  # 确保有足够的价量指标
                # 标准化并加权平均
                # 使用加权分配: 收益率>价格>交易量
                price_vol_df = result_df[price_volume_features].copy()
                
                # 标准化每个特征
                for col in price_vol_df.columns:
                    if price_vol_df[col].std() > 0:
                        price_vol_df[col] = (price_vol_df[col] - price_vol_df[col].mean()) / price_vol_df[col].std()
                    else:
                        price_vol_df[col] = 0
                
                # 分配权重
                weights = {}
                for col in price_vol_df.columns:
                    if 'return' in col.lower():
                        weights[col] = 2.0  # 收益率指标权重高
                    elif 'price' in col.lower() or 'macd' in col.lower() or 'rsi' in col.lower():
                        weights[col] = 1.5  # 价格技术指标权重次之
                    else:
                        weights[col] = 1.0  # 其他指标权重正常
                
                # 计算加权平均
                weighted_sum = sum(price_vol_df[col] * weights[col] for col in price_vol_df.columns)
                weights_sum = sum(weights.values())
                
                result_df['lnn_output_feature'] = weighted_sum / weights_sum if weights_sum > 0 else 0
                # 确保值在合理范围内
                result_df['lnn_output_feature'] = result_df['lnn_output_feature'].clip(-3, 3)
                # 规范化到[0,1]范围
                result_df['lnn_output_feature'] = (result_df['lnn_output_feature'] + 3) / 6
                
                self.logger.info(f"成功基于 {len(price_volume_features)} 个量价指标生成lnn_output_feature")
                return result_df
            
            # 方法3: 基于热度和动量特征计算
            self.logger.info("使用热度和动量特征估算lnn_output_feature")
            heat_features = [col for col in result_df.columns if 'heat' in col.lower() or 'momentum' in col.lower()]
            if len(heat_features) >= 3:  # 确保有足够多的热度特征
                # 标准化热度特征后取加权平均值
                heat_df = result_df[heat_features].copy()
                
                # 标准化
                for col in heat_df.columns:
                    if heat_df[col].std() > 0:
                        heat_df[col] = (heat_df[col] - heat_df[col].mean()) / heat_df[col].std()
                    else:
                        heat_df[col] = 0
                
                # 赋予近期热度更高权重
                weights = {}
                for col in heat_df.columns:
                    if '_5d' in col or '_3d' in col:
                        weights[col] = 2.0  # 近期热度权重加倍
                    else:
                        weights[col] = 1.0
                
                # 计算加权平均
                weighted_sum = sum(heat_df[col] * weights[col] for col in heat_df.columns)
                weights_sum = sum(weights.values())
                
                result_df['lnn_output_feature'] = weighted_sum / weights_sum if weights_sum > 0 else 0
                # 规范化到[0,1]范围
                min_val, max_val = result_df['lnn_output_feature'].min(), result_df['lnn_output_feature'].max()
                if max_val > min_val:
                    result_df['lnn_output_feature'] = (result_df['lnn_output_feature'] - min_val) / (max_val - min_val)
                else:
                    result_df['lnn_output_feature'] = 0.5
                
                self.logger.info(f"基于 {len(heat_features)} 个热度特征生成lnn_output_feature")
                return result_df
                
            # 方法4: 使用趋势相关特征
            trend_features = [col for col in result_df.columns if 'trend' in col.lower() or 'direction' in col.lower()]
            if len(trend_features) >= 2:  # 确保有足够的趋势特征
                # 标准化趋势特征后取平均值
                normalized_trends = result_df[trend_features].copy()
                for col in normalized_trends.columns:
                    if normalized_trends[col].max() > normalized_trends[col].min():
                        normalized_trends[col] = (normalized_trends[col] - normalized_trends[col].min()) / (normalized_trends[col].max() - normalized_trends[col].min())
                    else:
                        normalized_trends[col] = 0.5
                
                result_df['lnn_output_feature'] = normalized_trends.mean(axis=1)
                self.logger.info(f"基于 {len(trend_features)} 个趋势特征生成lnn_output_feature")
                return result_df
            
            # 方法5: 如果上述方法都不适用，使用当前有的任何相关特征
            self.logger.warning("无专门的热度或趋势特征，尝试使用任何可能相关的数值特征")
            numeric_features = result_df.select_dtypes(include=['number']).columns.tolist()
            # 排除ID类列和未来收益类列
            exclude_patterns = ['code', 'id', 'future_', 'date', 'time', 'day', 'month', 'year']
            relevant_features = [
                col for col in numeric_features 
                if not any(pattern in col.lower() for pattern in exclude_patterns)
            ]
            
            if len(relevant_features) >= 5:
                # 使用主成分分析(PCA)的思想，取所有特征的第一主成分近似值
                features_df = result_df[relevant_features].copy()
                # 标准化
                for col in features_df.columns:
                    if features_df[col].std() > 0:
                        features_df[col] = (features_df[col] - features_df[col].mean()) / features_df[col].std()
                
                # 简单地取均值作为第一主成分的近似
                result_df['lnn_output_feature'] = features_df.mean(axis=1)
                # 规范化到[0,1]范围
                min_val, max_val = result_df['lnn_output_feature'].min(), result_df['lnn_output_feature'].max()
                if max_val > min_val:
                    result_df['lnn_output_feature'] = (result_df['lnn_output_feature'] - min_val) / (max_val - min_val)
                else:
                    result_df['lnn_output_feature'] = 0.5
                
                self.logger.info(f"基于 {len(relevant_features)} 个一般特征的均值生成lnn_output_feature")
                return result_df
                
            # 方法6: 如果以上方法都失败，使用默认值
            self.logger.warning("无可用特征计算lnn_output_feature，使用默认值0.5")
            result_df['lnn_output_feature'] = 0.5
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"确保lnn_output_feature特征存在时出错: {e}")
            self.logger.error(traceback.format_exc())
            # 返回原始DataFrame并添加默认lnn_output_feature
            features_df['lnn_output_feature'] = 0.5
            return features_df

    def _predict_with_sequence(self, model_package, test_df, seq_features, date):
        """
        使用序列特征进行预测
        
        Args:
            model_package: 模型包，包含模型、特征列等
            test_df: 测试集数据
            seq_features: 序列特征字典
            date: 预测日期
            
        Returns:
            预测结果字典
        """
        self.logger.info(f"使用序列特征预测日期 {date} 的板块表现")
        
        try:
            # 获取模型组件并进行健壮性检查
            if not isinstance(model_package, dict):
                self.logger.error(f"模型包类型错误，应为dict，实际为{type(model_package)}")
                return None
            
            market_model = model_package.get('market_classifier')
            strength_model = model_package.get('strength_predictor')
            final_model = model_package.get('model')  # 使用'model'键，对应_train_rolling_day_model返回的键名
            
            # 如果找不到final_model，尝试其他可能的键名
            if final_model is None:
                # 尝试其他可能的键名
                for key_name in ['final_model', 'final_ensemble', 'lgb_model']:
                    if key_name in model_package:
                        final_model = model_package[key_name]
                        self.logger.info(f"通过键名 '{key_name}' 找到final_model")
                        break
            
            lnn_model = model_package.get('lnn')
            market_submodels = model_package.get('market_submodels', {})
            scaler = model_package.get('scaler')
            market_features = model_package.get('market_model_features', [])
            strength_features = model_package.get('strength_model_features', [])
            feature_cols = model_package.get('feature_columns', [])
            seq_length = model_package.get('seq_length', 10)
            
            # 提取测试特征
            test_sectors = test_df['sector_code'].unique()
            
            # 特征列健壮性检查
            if not feature_cols:
                self.logger.warning("特征列列表为空，尝试使用所有数值列作为特征")
                numeric_cols = test_df.select_dtypes(include=['number']).columns.tolist()
                exclude_cols = ['sector_code', 'date', 'label_date']
                exclude_cols += [col for col in test_df.columns if col.startswith('future_return')]
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
                if not feature_cols:
                    self.logger.error("无法确定特征列，无法继续预测")
                    return None
            
            # 清理特征数据
            try:
                # 确保所有特征列都存在
                valid_feature_cols = [col for col in feature_cols if col in test_df.columns]
                if len(valid_feature_cols) < len(feature_cols):
                    self.logger.warning(f"有 {len(feature_cols) - len(valid_feature_cols)} 个特征列在测试数据中不存在，将使用 {len(valid_feature_cols)} 个有效列")
                    feature_cols = valid_feature_cols
                
                if not feature_cols:
                    self.logger.error("没有有效的特征列，无法继续预测")
                    return None
                
                X_test = test_df[feature_cols].fillna(0).values
                
                # 确保scaler可用
                if scaler is not None:
                    try:
                        from sklearn.exceptions import NotFittedError
                        from sklearn.utils.validation import check_is_fitted
                        
                        try:
                            check_is_fitted(scaler)
                            X_test_scaled = scaler.transform(X_test)
                        except NotFittedError:
                            self.logger.warning("Scaler未拟合，尝试使用原始值")
                            X_test_scaled = X_test
                        except Exception as e:
                            self.logger.warning(f"Scaler转换失败: {e}，将使用原始值")
                            X_test_scaled = X_test
                    except ImportError:
                        # 如果无法导入sklearn检查函数，假设scaler已拟合
                        try:
                            X_test_scaled = scaler.transform(X_test)
                        except Exception as e:
                            self.logger.warning(f"Scaler转换失败: {e}，将使用原始值")
                            X_test_scaled = X_test
                else:
                    self.logger.warning("Scaler不可用，使用原始值")
                    X_test_scaled = X_test
            except Exception as e:
                print(f"发生错误: {e}")
            except Exception as e:
                self.logger.error(f"准备特征数据时出错: {e}")
                self.logger.error(traceback.format_exc())
                return None
            
            # 市场模型预测
            market_pred = None
            if market_model is not None and market_features:
                try:
                    # 安全索引
                    valid_indices = []
                    for f in market_features:
                        if f in feature_cols:
                            valid_indices.append(feature_cols.index(f))
                    
                    if valid_indices:
                        X_market_test = X_test_scaled[:, valid_indices]
                        market_pred = market_model.predict_proba(X_market_test)
                    else:
                        self.logger.warning("没有有效的市场特征索引，跳过市场模型预测")
                        # 创建默认预测
                        market_pred_shape = model_package.get('market_pred_shape', (test_df.shape[0], 3))
                        market_pred = np.zeros(market_pred_shape)
                        market_pred[:, 1] = 1.0  # 中性类别
                except Exception as e:
                    self.logger.error(f"市场模型预测出错: {e}")
                    self.logger.error(traceback.format_exc())
                    # 创建默认预测
                    market_pred_shape = model_package.get('market_pred_shape', (test_df.shape[0], 3))
                    market_pred = np.zeros(market_pred_shape)
                    market_pred[:, 1] = 1.0  # 设置默认类别为中性
            else:
                # 创建默认预测
                market_pred_shape = model_package.get('market_pred_shape', (test_df.shape[0], 3))
                market_pred = np.zeros(market_pred_shape)
                market_pred[:, 1] = 1.0  # 设置默认类别为中性
            
            # 强度模型预测
            strength_pred = None
            try:
                # 提取强度特征
                valid_strength_indices = []
                for s in strength_features:
                    if s in feature_cols and 'market_prob_' not in s:
                        valid_strength_indices.append(feature_cols.index(s))
                
                if valid_strength_indices:
                    X_strength_test = X_test_scaled[:, valid_strength_indices]
                    
                    # 合并市场模型预测结果
                    if market_pred is not None:
                        X_strength_test_with_market = np.hstack([
                            X_strength_test,
                            market_pred
                        ])
                    else:
                        X_strength_test_with_market = X_strength_test
                    
                    # 强度模型预测
                    if strength_model is not None:
                        try:
                            if hasattr(strength_model, 'predict_proba'):
                                strength_pred = strength_model.predict_proba(X_strength_test_with_market)[:, 1]
                            else:
                                strength_pred = strength_model.predict(X_strength_test_with_market)
                        except Exception as e:
                            self.logger.error(f"强度模型预测出错: {e}")
                            self.logger.error(traceback.format_exc())
                            # 使用默认值
                            strength_pred = np.zeros(test_df.shape[0])
                    else:
                        # 使用默认值
                        strength_pred = np.zeros(test_df.shape[0])
                else:
                    self.logger.warning("没有有效的强度特征索引，使用默认强度预测值")
                    strength_pred = np.zeros(test_df.shape[0])
            except Exception as e:
                print(f"发生错误: {e}")
            except Exception as e:
                self.logger.error(f"准备强度模型特征时出错: {e}")
                self.logger.error(traceback.format_exc())
                strength_pred = np.zeros(test_df.shape[0])
            
            # 对每个板块添加LNN预测
            sector_final_scores = {}
            
            # 使用LNN进行预测（如果可用）
            if lnn_model is not None and seq_features:
                try:
                    # 为每个板块准备序列数据
                    lnn_predictions = {}
                    
                    for sector_code in test_sectors:
                        if sector_code in seq_features:
                            # 获取序列数据，处理可能的不同格式
                            sequence_data = None
                            try:
                                # 检查序列特征格式
                                if isinstance(seq_features[sector_code], dict):
                                    if 'features' in seq_features[sector_code]:
                                        # 新格式: {'features': array, 'labels': array, ...}
                                        sequence_data = seq_features[sector_code]['features']
                                        self.logger.info(f"使用新格式序列特征，板块 {sector_code}")
                                    else:
                                        self.logger.warning(f"序列特征字典缺少'features'键，板块 {sector_code}")
                                        continue
                                elif isinstance(seq_features[sector_code], np.ndarray) and hasattr(seq_features[sector_code], 'shape'):
                                    # 旧格式: 直接存储特征数组
                                    sequence_data = seq_features[sector_code]
                                    self.logger.info(f"使用旧格式序列特征，板块 {sector_code}")
                                elif hasattr(seq_features[sector_code], 'shape'):
                                    # 其他类型但有shape属性
                                    sequence_data = seq_features[sector_code]
                                    self.logger.info(f"使用shape兼容的序列特征，板块 {sector_code}, 类型: {type(seq_features[sector_code])}")
                                else:
                                    self.logger.warning(f"序列特征格式无效，无法提取shape，板块 {sector_code}, 类型: {type(seq_features[sector_code])}")
                                    continue
                                
                                # 确保序列长度足够
                                if not hasattr(sequence_data, 'shape') or len(sequence_data) < seq_length:
                                    self.logger.warning(f"板块 {sector_code} 的序列特征长度不足或无效: {getattr(sequence_data, 'shape', 'No shape')}")
                                    continue
                                
                                # 使用最近的seq_length天
                                # 安全地获取序列数据，避免越界
                                try:
                                    sequence = sequence_data[-seq_length:]
                                except Exception as e:
                                    self.logger.error(f"获取序列数据时出错: {e}")
                                    continue
                                
                                # 检查并确保序列数据是2D格式 [sequence_length, feature_dim]
                                if len(sequence.shape) != 2:
                                    self.logger.warning(f"序列数据维度不是2D，形状为 {sequence.shape}，尝试重塑")
                                    try:
                                        if len(sequence.shape) == 1:  # 一维数组
                                            # 重塑为[sequence_length, 1]
                                            sequence = sequence.reshape(sequence.shape[0], 1)
                                        elif len(sequence.shape) > 2:  # 高维数组
                                            # 如果是3D格式 [1, sequence_length, feature_dim]，取第一个样本
                                            if sequence.shape[0] == 1:
                                                sequence = sequence[0]
                                            else:
                                                self.logger.error(f"无法将维度为 {sequence.shape} 的序列数据重塑为2D")
                                                # 在循环中，可以使用continue跳过当前板块
                                                continue
                                        else:
                                            self.logger.error(f"无法处理形状为 {sequence.shape} 的序列数据")
                                            # 在循环中，可以使用continue跳过当前板块
                                            continue
                                    except Exception as e_reshape:
                                        self.logger.error(f"重塑序列数据时出错: {e_reshape}")
                                        # 在循环中，可以使用continue跳过当前板块
                                        continue
                                
                                # 标准化序列
                                if scaler is not None:
                                    try:
                                        # 重塑为2D进行缩放
                                        sequence_flat = sequence.reshape(-1, sequence.shape[-1])
                                        sequence_scaled_flat = scaler.transform(sequence_flat)
                                        sequence_scaled = sequence_scaled_flat.reshape(sequence.shape)
                                        
                                        # 转换为Torch张量并添加批次维度 [1, sequence_length, feature_dim]
                                        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)
                                    except Exception as e:
                                        self.logger.error(f"序列特征处理出错，板块 {sector_code}: {e}")
                                        self.logger.error(traceback.format_exc())
                                        continue
                                else:
                                    # 直接使用原始序列
                                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # 添加批次维度
                                
                                # 使用LNN模型预测
                                try:
                                    # 确保模型处于评估模式
                                    if hasattr(lnn_model, 'model'):
                                        lnn_model.model.eval()
                                    
                                    with torch.no_grad():
                                        # 修改这里：使用predict方法而不是直接调用，并处理可能的维度不匹配
                                        try:
                                            # 尝试获取预期的特征维度
                                            if hasattr(lnn_model, 'input_size'):
                                                expected_features = lnn_model.input_size
                                                actual_features = sequence_tensor.shape[-1] if len(sequence_tensor.shape) > 2 else sequence_tensor.shape[1]
                                                
                                                # 如果维度不匹配，尝试调整
                                                if expected_features != actual_features:
                                                    self.logger.warning(f"输入特征维度不匹配: 预期{expected_features}，实际{actual_features}")
                                                    # 此处可能需要调整张量维度
                                                    
                                            # 使用predict方法进行预测
                                            lnn_pred = lnn_model.predict(sequence_tensor.cpu().numpy())
                                            
                                            # 确保输出格式正确
                                            if isinstance(lnn_pred, np.ndarray):
                                                if lnn_pred.ndim > 1:
                                                    lnn_predictions[sector_code] = float(lnn_pred[0][0])
                                                else:
                                                    lnn_predictions[sector_code] = float(lnn_pred[0])
                                            else:
                                                lnn_pred_np = lnn_pred.cpu().numpy() if hasattr(lnn_pred, 'cpu') else np.array(lnn_pred)
                                                lnn_predictions[sector_code] = float(lnn_pred_np[0][0] if lnn_pred_np.ndim > 1 else lnn_pred_np[0])
                                        
                                        except Exception as e:
                                            print(f"发生错误: {e}")
                                        except Exception as e_predict:
                                            self.logger.error(f"使用predict方法预测失败: {e_predict}，尝试使用模型直接调用")
                                            # 如果predict方法失败，尝试使用模型的forward方法
                                            if hasattr(lnn_model, 'model'):
                                                outputs, _ = lnn_model.model(sequence_tensor)
                                                lnn_predictions[sector_code] = float(outputs.cpu().numpy()[0][0])
                                            else:
                                                # 最后的尝试，直接调用模型
                                                self.logger.warning("回退到直接调用模型")
                                                outputs, _ = lnn_model(sequence_tensor)
                                                lnn_predictions[sector_code] = float(outputs.cpu().numpy()[0][0])
                                except Exception as e:
                                    print(f"发生错误: {e}")
                                except Exception as e:
                                    self.logger.error(f"LNN模型预测出错，板块 {sector_code}: {e}")
                                    self.logger.error(traceback.format_exc())
                                    continue
                            except Exception as e:
                                print(f"发生错误: {e}")
                            except Exception as e:
                                self.logger.error(f"处理板块 {sector_code} 的序列特征时出错: {e}")
                                self.logger.error(traceback.format_exc())
                                continue
                    
                    # 合并LNN预测和强度模型预测
                    for i, row in test_df.iterrows():
                        sector_code = row['sector_code']
                        # 获取强度模型预测
                        if strength_pred is not None:
                            # 找到当前板块在strength_pred中的索引
                            idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)
                            # 安全地获取索引，避免越界
                            try:
                                if idx < len(strength_pred):
                                    strength_score = strength_pred[idx]
                                else:
                                    self.logger.warning(f"索引 {idx} 超出strength_pred范围 {len(strength_pred)}")
                                    strength_score = 0.0
                            except Exception as e:
                                self.logger.warning(f"获取强度预测时出错: {e}")
                                strength_score = 0.0
                        else:
                            strength_score = 0.0
                        
                        # 添加LNN预测（如果有）
                        lnn_score = lnn_predictions.get(sector_code, 0.0)
                        
                                                                    # 组合得分 - 使用自适应权重
                        combined_score = strength_score
                        if lnn_score != 0.0:  # 如果有LNN预测，不限制正值
                            # 获取市场状态
                            market_status = None
                            date_obj = pd.to_datetime(date) if not isinstance(date, pd.Timestamp) else date
                            try:
                                # 加载市场状态
                                market_status_path = os.path.join(get_project_root(), "to", "local_data", "market_status", "daily", f"market_status_{date_obj.strftime('%Y%m%d')}.json")
                                if os.path.exists(market_status_path):
                                    with open(market_status_path, 'r', encoding='utf-8') as f:
                                        market_status = json.load(f)
                            except Exception as e:
                                self.logger.warning(f"加载市场状态时出错: {e}")
                            
                            # 计算自适应权重
                            strength_weight, lnn_weight = self._calculate_adaptive_weights(model_package, market_status)
                            
                            # 使用计算的权重组合得分
                            combined_score = strength_weight * strength_score + lnn_weight * lnn_score
                        
                        # 应用校准器（如果有）
                        calibrated_score = combined_score
                        try:
                            # 获取校准器
                            calibrator = model_package.get('calibrator')
                            if calibrator is not None:
                                # 将预测值转换为适合校准器的格式
                                pred_to_calibrate = np.array([combined_score])
                                # 应用校准
                                calibrated_score = float(calibrator.transform(pred_to_calibrate)[0])
                                self.logger.debug(f"板块 {sector_code} 校准前: {combined_score:.4f}, 校准后: {calibrated_score:.4f}")
                        except Exception as e_cal:
                            self.logger.warning(f"应用校准器失败: {e_cal}，使用原始预测值")
                            calibrated_score = combined_score
                        
                        # 存储最终得分
                        sector_final_scores[sector_code] = {
                            'strength_score': float(strength_score),
                            'lnn_score': float(lnn_score),
                            'combined_score': float(combined_score),
                            'calibrated_score': float(calibrated_score),  # 添加校准后得分
                            'strength_weight': float(strength_weight if 'strength_weight' in locals() else 0.6),
                            'lnn_weight': float(lnn_weight if 'lnn_weight' in locals() else 0.4)
                        }
                except Exception as e:
                    print(f"发生错误: {e}")
                except Exception as e:
                    self.logger.error(f"LNN预测过程中出错: {e}")
                    self.logger.error(traceback.format_exc())
                    # 继续使用强度模型的结果
            else:
                # 仅使用强度模型结果
                self.logger.info("缺少LNN模型或序列特征，仅使用强度模型结果")
                for i, row in test_df.iterrows():
                    sector_code = row['sector_code']
                    if strength_pred is not None:
                        idx = test_df[test_df['sector_code'] == sector_code].index.get_loc(i)
                        # 安全地获取索引，避免越界
                        try:
                            if idx < len(strength_pred):
                                strength_score = strength_pred[idx]
                            else:
                                self.logger.warning(f"索引 {idx} 超出strength_pred范围 {len(strength_pred)}")
                                strength_score = 0.0
                        except Exception as e:
                            self.logger.warning(f"获取强度预测时出错: {e}")
                            strength_score = 0.0
                    else:
                        strength_score = 0.0
                    
                    # 应用校准器（如果有）
                    calibrated_score = strength_score
                    try:
                        # 获取校准器
                        calibrator = model_package.get('calibrator')
                        if calibrator is not None:
                            # 将预测值转换为适合校准器的格式
                            pred_to_calibrate = np.array([strength_score])
                            # 应用校准
                            calibrated_score = float(calibrator.transform(pred_to_calibrate)[0])
                            self.logger.debug(f"板块 {sector_code} 校准前: {strength_score:.4f}, 校准后: {calibrated_score:.4f}")
                    except Exception as e_cal:
                        self.logger.warning(f"应用校准器失败: {e_cal}，使用原始预测值")
                        calibrated_score = strength_score
                    
                    # 存储最终得分
                    sector_final_scores[sector_code] = {
                        'strength_score': float(strength_score),
                        'lnn_score': 0.0,
                        'combined_score': float(strength_score),
                        'calibrated_score': float(calibrated_score),  # 添加校准后得分
                        'strength_weight': 1.0,  # 只有强度模型时权重为1
                        'lnn_weight': 0.0  # LNN模型权重为0
                    }
            
            # 获取市场状态
            market_type = '32'  # 默认为中性行情
            market_trend_strength = 0
            if 'market_type' in test_df.columns:
                market_types = test_df['market_type'].unique()
                if len(market_types) > 0:
                    market_type = str(market_types[0])
            
            if 'trend_strength' in test_df.columns:
                trend_strengths = test_df['trend_strength'].unique()
                if len(trend_strengths) > 0:
                    market_trend_strength = float(trend_strengths[0])
            
            # 对板块排序 - 使用校准后的得分（如果可用）
            sorted_sectors = sorted(
                [(sector_code, data.get('calibrated_score', data['combined_score'])) for sector_code, data in sector_final_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # 获取预测结果
            result = {
                'date': date,
                'sector_scores': sector_final_scores,
                'sorted_sectors': sorted_sectors,
                'market_type': market_type,
                'market_trend_strength': market_trend_strength
            }
            
            return result
        except Exception as e:
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"预测时出错: {e}")
            self.logger.error(traceback.format_exc())
            return None


    def _train_rolling_day_model(self, features_df, train_date, training_window_days=252, model_name="rolling_sector_model", sector_list=None):
        """
        训练指定日期的滚动日模型
        
        Args:
            features_df: 包含所有板块特征的DataFrame
            train_date: 训练日期
            training_window_days: 训练窗口天数
            model_name: 模型名称
            sector_list: 板块列表
            
        Returns:
            训练好的模型包
        """
        try:
            # 初始化必要的变量，确保后续引用不会出错
            test_df = pd.DataFrame()  # 初始化为空DataFrame
            val_df = pd.DataFrame()  # 初始化为空DataFrame
            accuracy_val = 0.0  # 初始化验证集准确率
            metrics = {}  # 初始化指标字典
            market_submodels = {}  # 初始化市场子模型字典
            final_model_input_actual_cols = []  # 初始化最终模型特征列
            trained_cat_dtypes_map = {}  # 初始化分类特征数据类型映射
            label_encoder = {}  # 初始化标签编码器
            trained_label_encoders = {}  # 初始化标签编码器字典
            categorical_feature = []  # 初始化分类特征列表
            final_features = []  # 初始化最终特征列表
            y_meta_train = None  # 初始化元训练标签变量
            lnn_model = None  # 初始化lnn_model变量
            use_lnn = True
            
            # 初始化特征处理器
            feature_processor = FeatureProcessor()
            
            if sector_list is None:
                sector_list = features_df['sector_code'].unique()
            elif isinstance(sector_list, str):
                sector_list = [sector_list]
                
            train_date = pd.to_datetime(train_date)
            
            self.logger.info(f"开始为日期 {train_date} 训练模型...")
            
            # 1. 确定训练窗口
            date_obj = pd.to_datetime(train_date)
            # 训练窗口从训练日期前一年开始，到训练日期前一天结束
            train_start_date = date_obj - pd.Timedelta(days=training_window_days)
            train_end_date = date_obj  # 包含当天
            
            self.logger.info(f"准备训练窗口: {train_start_date} 至 {train_end_date} (共 {training_window_days} 天)")
            
            # 检查features_df中是否包含date列
            has_date_column = 'date' in features_df.columns
            
            # 创建训练数据的副本，同时确保date列存在
            if has_date_column:
                # 确保date是datetime类型
                if not pd.api.types.is_datetime64_dtype(features_df['date']):
                    features_df['date'] = pd.to_datetime(features_df['date'])
                
                # 使用时间窗口筛选，确保包含训练日期当天的数据
                train_df = features_df[(features_df['date'] >= train_start_date) & 
                                       (features_df['date'] <= train_end_date)].copy()
                
                self.logger.info(f"过滤后的训练数据: {train_start_date} 至 {train_end_date}, 共 {len(train_df)} 行")
            else:
                self.logger.error("features_df中没有date列，无法进行时间筛选")
                return None
            
            # 如果训练数据不足，返回None
            if len(train_df) < 50:
                self.logger.warning(f"训练日期 {train_date} 的有效数据不足 (仅有 {len(train_df)} 行)，跳过模型训练")
                return None
                
            # 确保有未来收益列用于预测
            future_return_col = 'future_return_2d'  # 使用2天后收益
            if future_return_col not in train_df.columns:
                self.logger.error(f"训练数据中缺少未来收益列 {future_return_col}")
                return None
                
            # 2. 创建收益率预测标签 - 直接使用future_return_2d作为目标
            if 'date' in train_df.columns:
                date_backup = train_df['date'].copy()
                self.logger.info(f"在创建收益率标签前，备份date列，示例值: {date_backup.head(1).values}")
            
            # 使用未来收益率作为预测目标
            train_df = self._create_return_prediction_labels(train_df, future_return_col)
            
            # 恢复date列（如果在处理过程中丢失）
            if 'date' not in train_df.columns and 'date_backup' in locals():
                train_df['date'] = date_backup
                self.logger.info("在创建收益率标签后，从备份中恢复了date列")
                
            # 初始化变量 - 将所有必要变量在这里初始化
            target_cols = ['future_return_2d']  # 直接使用未来收益率作为目标
            base_cols = [col for col in train_df.columns if col not in ['date', 'sector_code'] and not col.startswith('future_return_')]
            
            # 3. 特征工程 - 加回原版移除的特征工程代码
            self.logger.info("开始特征工程...")
            
            # 检查date列是否仍然存在
            if 'date' not in train_df.columns:
                if 'date_column' in train_df.columns:
                    self.logger.info("使用date_column恢复date列")
                    train_df['date'] = train_df['date_column']
                elif 'date_backup' in locals():
                    self.logger.info("使用备份的date_backup恢复date列")
                    train_df['date'] = date_backup
                else:
                    self.logger.error(f"train_df 中缺少 'date' 列。当前列: {train_df.columns.tolist()}")
                    return None
                    
            # 添加交互特征
            if 'heat_index' in train_df.columns and 'limit_up_count' in train_df.columns: 
                train_df['heat_limit_interaction'] = train_df['heat_index'] * train_df['limit_up_count']
            if 'leading_stock_count' in train_df.columns and 'high_turnover_ratio' in train_df.columns:
                train_df['leading_turnover_interaction'] = train_df['leading_stock_count'] * train_df['high_turnover_ratio']
            if 'multi_limit_ratio' in train_df.columns and 'broken_limit_ratio' in train_df.columns:
                train_df['limit_quality_score'] = train_df['multi_limit_ratio'] - train_df['broken_limit_ratio']
            
            # 4. 应用特征处理器
            self.logger.info("应用特征处理器处理训练数据...")
            categorical_cols = ['market_group'] if 'market_group' in train_df.columns else []
            # 初始化特征处理器并处理训练数据
            feature_processor.initialize_from_data(train_df, categorical_cols=categorical_cols)
            train_df = feature_processor.process_features(train_df, is_training=True)
            
            # 确保使用StandardScaler
            scaler = StandardScaler()
                
            # 5. ========== 数据集切分 (与原版一致) ==========
            # 确保 'date' 列存在于 train_df 中
            if 'date' not in train_df.columns:
                if 'date' in train_df.index.names:
                    self.logger.info("train_df 中 'date' 在索引中，执行 reset_index()")
                    train_df.reset_index(inplace=True)
                else:
                    self.logger.error(f"train_df 中缺少 'date' 列。当前列: {train_df.columns.tolist()}")
                    # 可以选择记录更多 train_df 的信息以帮助调试
                    # self.logger.info(f"train_df head:\n{train_df.head().to_string()}")
                    # self.logger.info(f"train_df shape: {train_df.shape}")
                    return None # 遇到严重错误，无法继续

            # 再次检查 'date' 列，如果 reset_index 后仍然不存在，说明问题更严重
            if 'date' not in train_df.columns:
                self.logger.error(f"train_df 中 'date' 列在 reset_index 后仍然缺失。")
                return None

            unique_dates = sorted(train_df['date'].unique())
            n_d = len(unique_dates)
            if n_d < 3: 
                self.logger.error(f"唯一日期 ({n_d}) 过少。"); 
                return None
            idx_70, idx_80 = int(n_d * 0.7), int(n_d * 0.8)
            idx_70 = max(1, idx_70); idx_80 = max(idx_70 + 1, idx_80) if idx_70 + 1 < n_d else n_d; idx_80 = min(idx_80, n_d) 
            if idx_70 >= idx_80 and idx_80 > 0 : idx_70 = idx_80 -1; idx_70 = max(0, idx_70)
            train_dates, val_dates = unique_dates[:idx_70], unique_dates[idx_70:idx_80]
            train_df_split = train_df[train_df['date'].isin(train_dates)].copy() if train_dates else pd.DataFrame()
            val_df = train_df[train_df['date'].isin(val_dates)].copy() if val_dates else pd.DataFrame()
            self.logger.info(f"拆分: 训 {len(train_df_split)}({len(train_dates)}天),验 {len(val_df)}({len(val_dates)}天)")
            if train_df_split.empty: 
                self.logger.error("训练集为空!"); 
                return None

            # 应用统一特征流水线处理训练集和验证集
            self.logger.info("应用特征流水线处理训练数据...")
            train_df_split = self._apply_feature_pipeline(train_df_split, None, mode='train')
            
            if not val_df.empty:
                self.logger.info("应用特征流水线处理验证数据...")
                val_df = self._apply_feature_pipeline(val_df, None, mode='validation')

            # 4. ========== 多层次模型架构 ==========
            self.logger.info("构建多层次模型...")
            if not train_df.empty:
                df_cols_source = train_df.columns.tolist()
                short_term_cols = [c for c in df_cols_source if 'momentum_5d' in c or 'momentum_10d' in c or 'change_percent' in c or 'amplitude' in c or 'volatility_5d' in c or 'volatility_10d' in c or '_ratio_5' in c or '_ratio_10' in c]
                mid_term_cols = [c for c in df_cols_source if 'momentum_20d' in c or 'ma20' in c or 'rsi14' in c or 'macd' in c or 'volatility_20d' in c or '_ratio_20' in c]
                leading_cols = [c for c in df_cols_source if 'leading' in c or 'limit' in c or 'heat' in c or 'turnover' in c or 'elasticity' in c or 'premium' in c or 'broken_limit' in c or 'multi_limit' in c or 'quality_score' in c or 'consecutive_limit_days' in c]
                market_cols = [c for c in df_cols_source if 'market_' in c or 'trend_' in c or 'cyb_' in c or 'signal_strength' in c]
                tech_cols = [c for c in df_cols_source if 'ma' in c or 'rsi' in c or 'macd' in c or 'volatility' in c or 'momentum' in c] 

            # --- 4.1 市场环境分类器 (与原版一致) ---
            market_env_feats_train = [c for c in market_cols if c in train_df_split.columns] 
            if not market_env_feats_train: 
                self.logger.error("市场分类器无可用训练特征!"); 
                return None
            X_market = train_df_split[market_env_feats_train].fillna(0)
            # 保存X_market_train以便后续在模型元数据中使用
            X_market_train = X_market
            
            if 'market_type' not in train_df_split.columns: 
                train_df_split['market_type'] = 'unknown_type_train_default' 
            y_market = train_df_split['market_type'].astype('category').cat.codes
            market_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            market_model.fit(X_market, y_market)
            
            # --- 4.2 市场环境细分模型 (与原版一致) ---
            market_groups_map_train = {'bull': [11, 12, 13, 21], 'bear': [43, 51, 52, 53], 'neutral': [22, 23, 31, 32, 33, 41, 42]}
            market_to_group_map_train = {str(m_val): grp_val for grp_val, ml_val in market_groups_map_train.items() for m_val in ml_val}
            train_df_split['market_group'] = train_df_split['market_type'].astype(str).map(lambda x_map_val: market_to_group_map_train.get(x_map_val, 'neutral'))
            
            # 初始化submodel_feature_lists_map
            submodel_feature_lists_map = {}
            market_submodels = {}
            
            for market_group_name_s, _ in [('bull', 'bull_market_model'), ('bear', 'bear_market_model'), ('neutral', 'neutral_market_model')]:
                market_train_sub_s_df = train_df_split[train_df_split['market_group'] == market_group_name_s]
                if len(market_train_sub_s_df) < 50: continue
                sub_model_features_cand_list_s = []
                if market_group_name_s == 'bull': sub_model_features_cand_list_s = list(set(leading_cols + short_term_cols[:15]))
                elif market_group_name_s == 'bear': sub_model_features_cand_list_s = list(set(market_cols + tech_cols)) 
                else: sub_model_features_cand_list_s = list(set(short_term_cols[:10] + mid_term_cols[:10] + leading_cols[:10] + market_cols[:5]))
                sub_model_features_train_actual_s = [c for c in sub_model_features_cand_list_s if c in market_train_sub_s_df.columns]
                if not sub_model_features_train_actual_s or not future_return_col or future_return_col not in market_train_sub_s_df.columns: continue
                X_market_sub_s = market_train_sub_s_df[sub_model_features_train_actual_s].fillna(0)
                y_market_return_sub_s = market_train_sub_s_df[future_return_col]
                sub_model_instance_s = xgb.XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
                if not X_market_sub_s.empty:
                    sub_model_instance_s.fit(X_market_sub_s, y_market_return_sub_s)
                    market_submodels[market_group_name_s] = {'model': sub_model_instance_s, 'features': sub_model_features_train_actual_s}
                    submodel_feature_lists_map[f'{market_group_name_s}_model_features'] = sub_model_features_train_actual_s

            # --- 4.3 板块相对强度预测模型 (与原版一致) ---
            if X_market.empty: 
                self.logger.error("X_market为空!"); 
                return None
            market_pred_train_proba = market_model.predict_proba(X_market)
            for i_p in range(market_pred_train_proba.shape[1]): 
                train_df_split[f'market_prob_{i_p}'] = market_pred_train_proba[:, i_p]
            strength_feat_cand_list = [c for c in train_df_split.columns if 'momentum' in c or 'relative_' in c or 'leading' in c or 'strength' in c or 'premium' in c or 'limit' in c or 'heat' in c or 'turnover' in c] + \
                                      [f'market_prob_{i_gen}' for i_gen in range(market_pred_train_proba.shape[1])]
            strength_feats_train_actual = list(set(c for c in strength_feat_cand_list if c in train_df_split.columns)) 
            if not strength_feats_train_actual: 
                self.logger.error("强度模型无可用训练特征!"); 
                return None
            X_strength = train_df_split[strength_feats_train_actual].fillna(0)
            # 保存X_strength_train以便后续在模型元数据中使用
            X_strength_train = X_strength
            
            # 直接使用future_return_2d作为回归目标，不再根据条件判断
            y_strength_target_actual = 'future_return_2d'
            if y_strength_target_actual not in train_df_split: 
                self.logger.error(f"强度模型目标列 {y_strength_target_actual} 缺失!"); 
                return None
            y_strength = train_df_split[y_strength_target_actual].astype(float)
            strength_model = xgb.XGBRegressor(random_state=42, verbosity=0)
            # 转换为numpy数组避免dtype问题
            X_strength_np = X_strength.to_numpy()
            strength_model.fit(X_strength_np, y_strength) 
            train_df_split['predicted_strength'] = strength_model.predict(X_strength_np)

            # --- 为验证集准备特征 (与原版一致) ---
            if not val_df.empty:
                val_X_market_df = pd.DataFrame(0, index=val_df.index, columns=market_env_feats_train)
                for col_val_m in market_env_feats_train: 
                    if col_val_m in val_df: val_X_market_df[col_val_m] = val_df[col_val_m]
                if not val_X_market_df.empty:
                    # 转换为numpy数组避免dtype问题
                    val_X_market_np = val_X_market_df.fillna(0).to_numpy()
                    market_pred_val_proba_v = market_model.predict_proba(val_X_market_np)
                    for i_v_p in range(market_pred_val_proba_v.shape[1]): val_df[f'market_prob_{i_v_p}'] = market_pred_val_proba_v[:, i_v_p]
                else: 
                    for i_v_p_fill in range(market_pred_train_proba.shape[1]): val_df[f'market_prob_{i_v_p_fill}'] = 0.0
                val_X_strength_df = pd.DataFrame(0, index=val_df.index, columns=strength_feats_train_actual)
                for col_val_s in strength_feats_train_actual:
                    if col_val_s in val_df: val_X_strength_df[col_val_s] = val_df[col_val_s]
                if not val_X_strength_df.empty:
                    # 转换为numpy数组避免dtype问题
                    val_X_strength_np = val_X_strength_df.fillna(0).to_numpy()
                    val_df['predicted_strength'] = strength_model.predict(val_X_strength_np)
                else: val_df['predicted_strength'] = 0.0
                if 'market_type' in val_df.columns: 
                    val_df['market_group'] = val_df['market_type'].astype(str).map(lambda x_val_mg: market_to_group_map_train.get(x_val_mg, 'neutral'))
                elif 'market_group' not in val_df.columns : val_df['market_group'] = 'neutral'

                # 对验证数据应用特征处理
                val_df = feature_processor.process_features(val_df, is_training=False)

            # --- 4.4 序列数据准备, 为LNN模型准备过去10天的序列输入 ---
            self.logger.info("准备序列数据为LNN模型...")
            
            # 准备用于序列模型的特征
            seq_length = 10  # 过去10天的数据作为序列输入
            
            # 从训练数据中创建序列特征
            sequence_train_data = {}
            try:
                sequence_train_data = self._create_sequence_features(train_df_split, sector_list=sector_list, seq_length=seq_length)
                if not sequence_train_data:
                    self.logger.warning("无法为训练数据创建序列特征，将跳过LNN模型训练")
            except Exception as e_seq_train:
                self.logger.error(f"创建训练序列特征时出错: {e_seq_train}")
                self.logger.error(traceback.format_exc())
                sequence_train_data = {}
            
            # 如果验证集不为空，也为验证集创建序列特征
            sequence_val_data = None
            if not val_df.empty:
                try:
                    sequence_val_data = self._create_sequence_features(val_df, sector_list=sector_list, seq_length=seq_length)
                    if not sequence_val_data:
                        self.logger.warning("无法为验证数据创建序列特征，将不使用验证数据")
                except Exception as e_seq_val:
                    self.logger.error(f"创建验证序列特征时出错: {e_seq_val}")
                    self.logger.error(traceback.format_exc())
                    sequence_val_data = None

            # --- 4.5 LNN 模型训练 ---
            final_feat_pool_cand_list = list(dict.fromkeys(col for col in (base_cols + ['predicted_strength', 'market_group'] + market_cols[:10] + leading_cols[:15] + short_term_cols[:10] + mid_term_cols[:5] + [c for c in train_df_split.columns if '_standardized' in c][:15]) if col in train_df_split.columns))
            if not final_feat_pool_cand_list: 
                self.logger.error("候选特征池为空!"); 
                return None

            # 初始化final_feat_pool为最终使用的特征池
            final_feat_pool = final_feat_pool_cand_list.copy()

            # 初始化X_final_train_from_pool_df，确保后续引用不会出错
            if not train_df_split[final_feat_pool_cand_list].empty: # 确保这些列的选择不会导致空DataFrame
                X_final_train_from_pool_df = train_df_split[final_feat_pool_cand_list].copy() 
            else: 
                X_final_train_from_pool_df = pd.DataFrame(columns=final_feat_pool_cand_list) # 创建带列的空DF

            if target_cols[0] not in train_df_split.columns: 
                self.logger.error(f"目标列 '{target_cols[0]}' 在 train_df_split 中不存在!"); 
                return None
            y_final_train = train_df_split[target_cols[0]] 
            
            X_scaled_lnn_input = np.array([]) # 重置
            
            # 明确检查LiquidNetModel是否可用，防止调用不存在的方法
            lnn_model_available = hasattr(self, 'LiquidNetModel') and self.LiquidNetModel is not None
            if not lnn_model_available:
                self.logger.warning("LiquidNetModel不可用，将跳过LNN模型训练")
                use_lnn = False
                lnn_model = None

            if use_lnn and lnn_model_available and sequence_train_data:
                self.logger.info(f"训练LNN序列模型...")
                try:
                    # _create_sequence_features返回的是一个字典，键是板块代码，值是对应的特征序列
                    # 需要将其整合为用于训练的特征矩阵和标签向量
                    all_sequences = []
                    all_labels = []
                    sequence_dates = []
                    sequence_sector_codes = []
                    
                    # 遍历所有板块的序列特征
                    for sector_code, sequence_data in sequence_train_data.items():
                        # 验证sequence_data的格式
                        if not isinstance(sequence_data, dict):
                            self.logger.warning(f"板块 {sector_code} 的序列数据不是字典格式，将跳过")
                            continue
                            
                        # 确保sequence_data包含'features'和'labels'键
                        if 'features' not in sequence_data or 'labels' not in sequence_data:
                            self.logger.warning(f"板块 {sector_code} 的序列数据缺少'features'或'labels'键，将跳过")
                            continue
                            
                        # 获取特征和标签
                        sequence_feature = sequence_data['features']
                        sequence_label = sequence_data['labels']
                        
                        # 验证特征数据
                        if not hasattr(sequence_feature, 'shape'):
                            self.logger.warning(f"板块 {sector_code} 的特征数据无shape属性，将跳过")
                            continue
                            
                        # 验证特征数据长度
                        if sequence_feature.shape[0] < seq_length:
                            self.logger.warning(f"板块 {sector_code} 的特征数据长度 {sequence_feature.shape[0]} 小于所需长度 {seq_length}，将跳过")
                            continue
                            
                        # 获取最近的seq_length天数据
                        try:
                            recent_sequence = sequence_feature[-seq_length:]
                            
                            # 验证数据是否为二维数组 (样本数, 特征数)
                            if len(recent_sequence.shape) != 2:
                                self.logger.warning(f"板块 {sector_code} 的特征数据维度 {len(recent_sequence.shape)} 不是2，将跳过")
                                continue
                                
                            # 获取该板块的最新标签
                            label = sequence_label[-1] if len(sequence_label) > 0 else 0
                            
                            # 添加到集合中
                            all_sequences.append(recent_sequence)
                            all_labels.append(label)
                            
                            # 记录序列对应的日期和板块代码
                            if 'dates' in sequence_data and len(sequence_data['dates']) > 0:
                                sequence_dates.append(sequence_data['dates'][-1])
                            sequence_sector_codes.append(sector_code)
                        except Exception as e:
                            self.logger.error(f"处理板块 {sector_code} 的序列数据时出错: {e}")
                            self.logger.error(traceback.format_exc())
                            continue
                    
                    if all_sequences:
                        # 将列表转换为numpy数组
                        sequence_features = np.array(all_sequences)
                        sequence_labels = np.array(all_labels)
                        
                        # 确保数据类型正确
                        sequence_features = sequence_features.astype(np.float32)
                        sequence_labels = sequence_labels.astype(np.float32)
                        
                        # 检查维度 - LNN模型需要3D输入 [samples, sequence_length, features]
                        if len(sequence_features.shape) == 3:
                            self.logger.info(f"序列特征维度正确: {sequence_features.shape}")
                        else:
                            # 如果维度不是3D，尝试重塑为3D
                            self.logger.warning(f"序列特征维度不是3D: {sequence_features.shape}，尝试重塑")
                            
                            # 假设数据是2D [samples*sequence_length, features]
                            # 尝试重塑为 [samples, sequence_length, features]
                            try:
                                num_samples = len(all_sequences)
                                if num_samples > 0:
                                    # 计算每个样本的特征数
                                    features_per_sample = sequence_features.shape[1] if len(sequence_features.shape) > 1 else 1
                                    
                                    # 重塑为3D
                                    sequence_features = sequence_features.reshape(num_samples, seq_length, features_per_sample)
                                    self.logger.info(f"重塑后的序列特征维度: {sequence_features.shape}")
                                else:
                                    self.logger.error("没有有效的序列样本，无法继续训练LNN模型")
                                    use_lnn = False
                                    lnn_model = None
                                    # 由于外层if检查了all_sequences，这里不需要continue，直接结束本轮if分支
                            except Exception as e:
                                self.logger.error(f"重塑序列特征维度时出错: {e}")
                                self.logger.error(traceback.format_exc())
                                use_lnn = False
                                lnn_model = None
                                # 由于在最外层try-except块中，这里不需要continue，直接结束本轮if分支
                                
                        
                        # 检查数据有效性
                        if not np.isfinite(sequence_features).all():
                            self.logger.warning("序列特征包含无穷或NaN值，将替换为0")
                            sequence_features = np.nan_to_num(sequence_features, nan=0.0, posinf=0.0, neginf=0.0)
                            
                        if not np.isfinite(sequence_labels).all():
                            self.logger.warning("标签包含无穷或NaN值，将替换为0")
                            sequence_labels = np.nan_to_num(sequence_labels, nan=0.0, posinf=0.0, neginf=0.0)
                            
                        # 检查序列形状
                        self.logger.info(f"序列特征形状: {sequence_features.shape}, 标签形状: {sequence_labels.shape}")
                        
                        # 在这里拟合scaler！！！
                        # 将3D序列数据重塑为2D以进行缩放
                        sequence_flat = sequence_features.reshape(-1, sequence_features.shape[-1])
                        self.logger.info(f"拟合StandardScaler，特征形状: {sequence_flat.shape}")
                        try:
                            # 检查数据是否包含NaN或无穷值
                            if not np.isfinite(sequence_flat).all():
                                self.logger.warning("序列特征包含NaN或无穷值，将替换为0后再进行缩放")
                                sequence_flat = np.nan_to_num(sequence_flat, nan=0.0, posinf=0.0, neginf=0.0)
                                
                            # 拟合缩放器
                            scaler.fit(sequence_flat)
                            
                            # 转换数据
                            sequence_flat_scaled = scaler.transform(sequence_flat)
                            sequence_features_scaled = sequence_flat_scaled.reshape(sequence_features.shape)
                            self.logger.info("StandardScaler拟合并转换成功")
                        except Exception as e_scaler:
                            self.logger.error(f"StandardScaler拟合或转换时出错: {e_scaler}")
                            self.logger.error(traceback.format_exc())
                            # 如果缩放失败，使用原始数据
                            sequence_features_scaled = sequence_features
                            self.logger.warning("由于Scaler错误，将使用未缩放的原始序列数据")
                            # 重置scaler，避免后续使用未正确拟合的scaler
                            scaler = None
                        
                        # 检查序列形状
                        self.logger.info(f"序列特征形状: {sequence_features.shape}, 标签形状: {sequence_labels.shape}")
                        
                        # 准备验证数据 (如果有)
                        validation_data = None  # 初始化为None
                        if sequence_val_data is not None and not val_df.empty:
                            try:
                                val_sequences = []
                                val_labels = []
                                
                                for sector_code, sequence_data in sequence_val_data.items():
                                    # 验证sequence_data的格式
                                    if not isinstance(sequence_data, dict):
                                        self.logger.warning(f"验证集中板块 {sector_code} 的序列数据不是字典格式，将跳过")
                                        continue
                                        
                                    # 确保sequence_data包含'features'和'labels'键
                                    if 'features' not in sequence_data or 'labels' not in sequence_data:
                                        self.logger.warning(f"验证集中板块 {sector_code} 的序列数据缺少'features'或'labels'键，将跳过")
                                        continue
                                        
                                    # 获取特征和标签
                                    sequence_feature = sequence_data['features']
                                    sequence_label = sequence_data['labels']
                                    
                                    # 验证特征数据
                                    if not hasattr(sequence_feature, 'shape'):
                                        self.logger.warning(f"验证集中板块 {sector_code} 的特征数据无shape属性，将跳过")
                                        continue
                                        
                                    # 验证特征数据长度
                                    if sequence_feature.shape[0] < seq_length:
                                        self.logger.warning(f"验证集中板块 {sector_code} 的特征数据长度 {sequence_feature.shape[0]} 小于所需长度 {seq_length}，将跳过")
                                        continue
                                        
                                    # 获取最近的seq_length天数据
                                    try:
                                        recent_sequence = sequence_feature[-seq_length:]
                                        
                                        # 验证数据是否为二维数组 (样本数, 特征数)
                                        if len(recent_sequence.shape) != 2:
                                            self.logger.warning(f"验证集中板块 {sector_code} 的特征数据维度 {len(recent_sequence.shape)} 不是2，将跳过")
                                            continue
                                            
                                        # 获取该板块的最新标签
                                        label = sequence_label[-1] if len(sequence_label) > 0 else 0
                                        
                                        # 添加到集合中
                                        val_sequences.append(recent_sequence)
                                        val_labels.append(label)
                                    except Exception as e:
                                        self.logger.error(f"处理验证集中板块 {sector_code} 的序列数据时出错: {e}")
                                        continue
                                
                                if val_sequences:
                                    # 转换为numpy数组
                                    val_sequence_features = np.array(val_sequences)
                                    val_sequence_labels = np.array(val_labels)
                                    
                                    # 确保数据类型正确
                                    val_sequence_features = val_sequence_features.astype(np.float32)
                                    val_sequence_labels = val_sequence_labels.astype(np.float32)
                                    
                                    # 检查维度 - LNN模型需要3D输入 [samples, sequence_length, features]
                                    if len(val_sequence_features.shape) == 3:
                                        self.logger.info(f"验证序列特征维度正确: {val_sequence_features.shape}")
                                    else:
                                        # 如果维度不是3D，尝试重塑为3D
                                        self.logger.warning(f"验证序列特征维度不是3D: {val_sequence_features.shape}，尝试重塑")
                                        
                                        # 尝试重塑为 [samples, sequence_length, features]
                                        try:
                                            num_samples = len(val_sequences)
                                            if num_samples > 0:
                                                # 计算每个样本的特征数
                                                features_per_sample = val_sequence_features.shape[1] if len(val_sequence_features.shape) > 1 else 1
                                                
                                                # 重塑为3D
                                                val_sequence_features = val_sequence_features.reshape(num_samples, seq_length, features_per_sample)
                                                self.logger.info(f"重塑后的验证序列特征维度: {val_sequence_features.shape}")
                                            else:
                                                self.logger.error("没有有效的验证序列样本，不使用验证数据")
                                                validation_data = None
                                                # 在这里不使用break或continue，而是在父级代码块结束后通过检查validation_data是否为None来决定是否继续
                                        except Exception as e:
                                            self.logger.error(f"重塑验证序列特征维度时出错: {e}")
                                            self.logger.error(traceback.format_exc())
                                            validation_data = None
                                            # 在这里不使用break或continue，而是在父级代码块结束后通过检查validation_data是否为None来决定是否继续
                                    
                                    # 检查数据有效性
                                    if not np.isfinite(val_sequence_features).all():
                                        self.logger.warning("验证序列特征包含无穷或NaN值，将替换为0")
                                        val_sequence_features = np.nan_to_num(val_sequence_features, nan=0.0, posinf=0.0, neginf=0.0)
                                        
                                    if not np.isfinite(val_sequence_labels).all():
                                        self.logger.warning("验证标签包含无穷或NaN值，将替换为0")
                                        val_sequence_labels = np.nan_to_num(val_sequence_labels, nan=0.0, posinf=0.0, neginf=0.0)
                                    
                                    # 确保特征维度与训练数据匹配
                                    if val_sequence_features.shape[2] != sequence_features.shape[2]:
                                        self.logger.error(f"验证特征维度 {val_sequence_features.shape[2]} 与训练特征维度 {sequence_features.shape[2]} 不匹配")
                                        validation_data = None
                                    else:
                                        # 使用已拟合的scaler缩放验证数据
                                        val_sequence_flat = val_sequence_features.reshape(-1, val_sequence_features.shape[-1])
                                        val_sequence_flat_scaled = scaler.transform(val_sequence_flat)
                                        val_sequence_features_scaled = val_sequence_flat_scaled.reshape(val_sequence_features.shape)
                                        
                                        # 设置验证数据
                                        validation_data = (val_sequence_features_scaled, val_sequence_labels)
                                        self.logger.info(f"验证数据准备完成，形状: {val_sequence_features_scaled.shape}")
                            except Exception as e:
                                print(f"发生错误: {e}")
                            except Exception as e_val:
                                self.logger.warning(f"准备验证数据时出错: {e_val}, 将不使用验证数据进行训练")
                                validation_data = None
                        
                        try:
                            # 获取特征维度
                            input_size = sequence_features.shape[2]  # 特征维度
                            output_size = 1  # 输出维度 (通常是1，代表回归预测值)
                            hidden_size = 64  # 隐藏层大小
                            num_layers = 2  # 层数
                            
                            # 检测CUDA是否可用
                            import torch
                            from torch import nn
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            
                            # 创建LNN模型实例 - 初始化为回归模型
                            lnn_model = self.LiquidNetModel(
                                input_size=input_size,
                                output_size=output_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                device=device
                            )
                            
                            # 修改为均方误差损失函数，适合回归任务
                            lnn_model.criterion = nn.MSELoss()
                            
                            self.logger.info(f"成功创建回归LNN模型，输入维度: {input_size}, 输出维度: {output_size}, 损失函数: MSELoss, 设备: {device}")
                        except Exception as e:
                            print(f"发生错误: {e}")
                        except Exception as e_lnn_init:
                            self.logger.error(f"创建LNN模型时出错: {e_lnn_init}")
                            self.logger.error(traceback.format_exc())
                            lnn_model = None
                            use_lnn = False
                        
                        # 初始化校准器变量
                        calibrator = None
                        
                        # 训练模型 - 使用缩放后的数据
                        fit_kwargs_lnn = {
                            'X': sequence_features_scaled,  # 使用缩放后的数据
                            'y': sequence_labels,  # 直接使用连续的收益率数值
                            'batch_size': 32, 
                            'epochs': 50  # 增加轮次以适应回归任务
                        }
                        
                        # 只有在验证数据可用时才添加相关参数
                        if validation_data is not None:
                            # 检查validation_data是否包含合适的格式
                            is_valid_validation_data = (
                                isinstance(validation_data, tuple) and 
                                len(validation_data) == 2 and
                                isinstance(validation_data[0], np.ndarray) and
                                isinstance(validation_data[1], np.ndarray) and
                                validation_data[0].size > 0 and
                                validation_data[1].size > 0
                            )
                            
                            if is_valid_validation_data:
                                self.logger.info(f"使用验证数据训练，验证数据形状: {validation_data[0].shape}")
                                fit_kwargs_lnn['validation_data'] = validation_data
                                fit_kwargs_lnn['early_stopping'] = True
                                fit_kwargs_lnn['patience'] = 15  # 增加早停耐心值
                            else: 
                                self.logger.warning("验证数据格式不正确，将不使用验证数据")
                                fit_kwargs_lnn['early_stopping'] = False
                        else: 
                            # 没有验证数据，则不使用早停
                            fit_kwargs_lnn['early_stopping'] = False
                            self.logger.info("没有验证数据可用，将不使用early stopping")
                        
                        # 训练模型并捕获任何可能的异常
                        if lnn_model is not None and hasattr(lnn_model, 'fit'):
                            try:
                                train_result = lnn_model.fit(**fit_kwargs_lnn)
                                self.logger.info("LNN回归模型训练完成")
                                
                                # 记录训练结果
                                if isinstance(train_result, dict) and 'loss' in train_result:
                                    history = train_result
                                    final_loss = history['loss'][-1] if history['loss'] else None
                                    self.logger.info(f"LNN模型最终训练损失: {final_loss}")
                                    if 'val_loss' in history:
                                        final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
                                        self.logger.info(f"LNN模型最终验证损失: {final_val_loss}")
                                
                                # 计算预测并评估模型
                                try:
                                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                                    
                                    # 使用训练好的模型预测训练集
                                    train_pred = lnn_model.predict(sequence_features_scaled)
                                    
                                    # 计算回归评估指标
                                    mae = mean_absolute_error(sequence_labels, train_pred)
                                    rmse = np.sqrt(mean_squared_error(sequence_labels, train_pred))
                                    r2 = r2_score(sequence_labels, train_pred)
                                    
                                    # 计算方向准确率 - 预测正负号与实际一致的比例
                                    direction_accuracy = np.mean((train_pred > 0) == (sequence_labels > 0))
                                    
                                    self.logger.info(f"LNN回归模型训练集评估 - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, 方向准确率: {direction_accuracy:.4f}")
                                    
                                    # 添加预测校准
                                    try:
                                        # 尝试导入校准器
                                        self.logger.info("正在创建回归预测校准器...")
                                        
                                        # 先尝试导入本地定义的校准器
                                        try:
                                            from to.regression_model_calibrator import ReturnCalibrator, create_default_calibrator
                                            calibrator = create_default_calibrator()
                                        except ImportError:
                                            self.logger.warning("无法导入本地校准器，尝试从Liquid库导入...")
                                            try:
                                                from to.Liquid.nn.liquid.calibration import ReturnCalibrator
                                                calibrator = ReturnCalibrator(method='ensemble')
                                            except ImportError:
                                                self.logger.error("无法导入任何校准器，将不使用校准")
                                                calibrator = None
                                        
                                        # 如果校准器创建成功，拟合并评估校准效果
                                        if calibrator is not None:
                                            # 拟合校准器
                                            calibrator.fit(train_pred, sequence_labels)
                                            self.logger.info("校准器拟合完成")
                                            
                                            # 应用校准器
                                            calibrated_pred = calibrator.transform(train_pred)
                                            
                                            # 评估校准效果
                                            cal_mae = mean_absolute_error(sequence_labels, calibrated_pred)
                                            cal_rmse = np.sqrt(mean_squared_error(sequence_labels, calibrated_pred))
                                            cal_r2 = r2_score(sequence_labels, calibrated_pred)
                                            cal_direction_accuracy = np.mean((calibrated_pred > 0) == (sequence_labels > 0))
                                            
                                            # 校准前后对比
                                            mae_change = (cal_mae - mae) / mae * 100
                                            rmse_change = (cal_rmse - rmse) / rmse * 100
                                            r2_change = (cal_r2 - r2) / r2 * 100 if r2 != 0 else float('inf')
                                            dir_acc_change = (cal_direction_accuracy - direction_accuracy) / direction_accuracy * 100
                                            
                                            self.logger.info(f"校准后评估 - MAE: {cal_mae:.4f} ({mae_change:.1f}%), RMSE: {cal_rmse:.4f} ({rmse_change:.1f}%), R²: {cal_r2:.4f} ({r2_change:.1f}%), 方向准确率: {cal_direction_accuracy:.4f} ({dir_acc_change:.1f}%)")
                                    except Exception as e_cal:
                                        self.logger.error(f"创建或应用校准器时出错: {e_cal}")
                                        self.logger.error(traceback.format_exc())
                                        calibrator = None
                                except Exception as e_eval:
                                    self.logger.error(f"评估LNN回归模型时出错: {e_eval}")
                                    self.logger.error(traceback.format_exc())
                            
                            except Exception as e_lnn_tr_final: 
                                self.logger.error(f"LNN训练出错: {e_lnn_tr_final}\n{traceback.format_exc()}")
                                lnn_model = None
                                use_lnn = False
                        else:
                            self.logger.error("LNN模型不可用或缺少fit方法，跳过LNN模型训练")
                            lnn_model = None
                            use_lnn = False
                except Exception as e:
                    print(f"发生错误: {e}")
                except Exception as e_lnn_tr_final: 
                    self.logger.error(f"LNN训练出错: {e_lnn_tr_final}\n{traceback.format_exc()}")
                    lnn_model = None
                    use_lnn = False
            else:
                self.logger.warning("不使用LNN模型或序列数据不可用")
                use_lnn = False
                lnn_model = None
            
            # --- 4.5 最终集成模型 (LightGBM Classifier) ---
            self.logger.info("准备最终集成模型输入...")
            if X_final_train_from_pool_df.empty:
                self.logger.error("X_final_train_from_pool_df为空!"); 
                return None
            
            # 确保temp_X_for_final_df包含date和sector_code列
            temp_X_for_final_df = X_final_train_from_pool_df.copy()
            
            # 检查是否存在date和sector_code列，如果不存在，需要从原始数据中获取
            if 'date' not in temp_X_for_final_df.columns or 'sector_code' not in temp_X_for_final_df.columns:
                self.logger.info("temp_X_for_final_df中缺少date或sector_code列，尝试从train_df_split中获取")
                
                # 检查train_df_split是否有这些列
                has_required_cols = all(col in train_df_split.columns for col in ['date', 'sector_code'])
                
                if has_required_cols:
                    # 为了避免混淆，创建一个新的DataFrame，只包含我们需要的列
                    id_cols_df = train_df_split[['date', 'sector_code']].copy()
                    
                    # 确保索引一致以进行合并
                    if len(id_cols_df) == len(temp_X_for_final_df):
                        # 如果行数一致，直接重置索引并合并
                        temp_X_for_final_df.reset_index(drop=True, inplace=True)
                        id_cols_df.reset_index(drop=True, inplace=True)
                        
                        # 合并列
                        temp_X_for_final_df['date'] = id_cols_df['date']
                        temp_X_for_final_df['sector_code'] = id_cols_df['sector_code']
                    else:
                        # 如果行数不一致，尝试更智能的合并方式
                        self.logger.warning(f"行数不匹配，temp_X_for_final_df: {len(temp_X_for_final_df)}, id_cols_df: {len(id_cols_df)}")
                        
                        # 尝试方法1：基于索引的映射
                        try:
                            # 创建一个索引列，用于稍后合并
                            id_cols_df['row_idx'] = range(len(id_cols_df))
                            temp_X_for_final_df['row_idx'] = range(len(temp_X_for_final_df))
                            
                            # 合并，确保保持原始顺序
                            temp_X_for_final_df = pd.merge(
                                temp_X_for_final_df, 
                                id_cols_df,
                                on='row_idx',
                                how='left'
                            )
                            
                            # 删除临时索引列
                            temp_X_for_final_df.drop('row_idx', axis=1, inplace=True)
                            
                            self.logger.info(f"成功使用索引映射添加date和sector_code列")
                        except Exception as e:
                            self.logger.warning(f"索引映射合并失败: {e}，尝试其他方法")
                            
                            # 尝试方法2：如果temp_X_for_final_df来自原始训练数据的部分，可尝试重构索引
                            try:
                                # 保留原始temp_X_for_final_df的索引作为列
                                temp_X_for_final_df['original_index'] = temp_X_for_final_df.index
                                
                                # 获取train_df_split的唯一标识符
                                if 'sector_code' in train_df_split.columns and 'date' in train_df_split.columns:
                                    # 创建两个数据集都有的特征，用于匹配
                                    common_cols = list(set(temp_X_for_final_df.columns) & set(train_df_split.columns))
                                    common_cols = [c for c in common_cols if c not in ['date', 'sector_code', 'original_index']]
                                    
                                    if common_cols:
                                        # 使用共同列进行匹配
                                        merged_df = pd.merge(
                                            temp_X_for_final_df,
                                            train_df_split[['date', 'sector_code'] + common_cols],
                                            on=common_cols,
                                            how='left'
                                        )
                                        
                                        # 如果合并成功并添加了所需列
                                        if 'date' in merged_df.columns and 'sector_code' in merged_df.columns:
                                            # 删除多余列
                                            temp_X_for_final_df = merged_df.drop(['original_index'], axis=1)
                                            self.logger.info(f"成功使用共同特征匹配添加date和sector_code列")
                                        else:
                                            raise ValueError("合并后数据集缺少必要列")
                                    else:
                                        raise ValueError("没有找到共同列用于匹配")
                                else:
                                    raise ValueError("train_df_split缺少必要的列")
                            except Exception as e:
                                print(f"发生错误: {e}")
                            except Exception as e2:
                                self.logger.warning(f"特征匹配合并失败: {e2}，使用默认填充")
                                
                                # 方法3：如果都失败了，使用默认填充
                                if 'date' not in temp_X_for_final_df.columns:
                                    # 使用训练日期填充
                                    temp_X_for_final_df['date'] = train_date
                                    self.logger.warning(f"使用训练日期 {train_date} 填充date列")
                                
                                if 'sector_code' not in temp_X_for_final_df.columns:
                                    # 如果有行业代码列表，则循环分配
                                    sector_codes = train_df_split['sector_code'].unique() if 'sector_code' in train_df_split.columns else ['default_sector']
                                    sector_mapping = {i: sector_codes[i % len(sector_codes)] for i in range(len(temp_X_for_final_df))}
                                    temp_X_for_final_df['sector_code'] = temp_X_for_final_df.index.map(lambda x: sector_mapping[x])
                                    self.logger.warning(f"使用循环分配方法填充sector_code列")
                    
                    self.logger.info(f"成功添加date和sector_code列到temp_X_for_final_df，形状: {temp_X_for_final_df.shape}")
                else:
                    self.logger.warning("无法从train_df_split中获取date或sector_code列，LNN特征合并可能会失败")
            
            lnn_out_feat_name = 'lnn_output_feature'
            temp_X_for_final_df[lnn_out_feat_name] = 0.0

            if lnn_model is not None and sequence_train_data is not None:
                self.logger.info("尝试为最终模型生成LNN特征...")
                try:
                    # 获取当前日期的所有板块LNN预测值，用于最终模型
                    lnn_outputs = {}
                    
                    for sector_code in temp_X_for_final_df['sector_code'].unique():
                        if sector_code in sequence_train_data:
                            # 提取序列特征并进行预测
                            seq_data = sequence_train_data[sector_code]
                            
                            # 检查序列特征格式
                            if isinstance(seq_data, dict) and 'features' in seq_data:
                                sequence = seq_data['features']
                            elif hasattr(seq_data, 'shape'):
                                sequence = seq_data
                            else:
                                self.logger.warning(f"板块 {sector_code} 的序列特征格式无效")
                                continue
                                
                            # 确保序列长度足够
                            if not hasattr(sequence, 'shape') or len(sequence) < seq_length:
                                self.logger.warning(f"板块 {sector_code} 的序列特征长度不足或无效")
                                continue
                                
                            # 使用最近的seq_length天
                            recent_sequence = sequence[-seq_length:]
                            
                            # 转换为3D张量 [1, seq_length, feature_dim]
                            # 检查维度
                            if len(recent_sequence.shape) < 2:
                                # 尝试重塑为2D
                                try:
                                    recent_sequence = recent_sequence.reshape(len(recent_sequence), 1)
                                    self.logger.debug(f"序列由1D重塑为2D，新形状: {recent_sequence.shape}")
                                except Exception as e:
                                    self.logger.error(f"测试数据: 重塑序列数据时出错: {e}")
                                    continue
                                    
                            # 应用scaler
                            if scaler is not None:
                                try:
                                    # 转换为2D进行缩放
                                    seq_flat = recent_sequence.reshape(-1, recent_sequence.shape[-1])
                                    seq_scaled = scaler.transform(seq_flat)
                                    seq_tensor = torch.FloatTensor(seq_scaled.reshape(recent_sequence.shape)).unsqueeze(0)
                                except Exception as e:
                                    self.logger.warning(f"缩放序列数据时出错: {e}")
                                    seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                            else:
                                seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                                
                            # 使用LNN模型预测
                            try:
                                with torch.no_grad():
                                    # 修改这里：使用predict方法而不是直接调用，并处理可能的维度不匹配
                                    try:
                                        # 尝试获取预期的特征维度
                                        if hasattr(lnn_model, 'input_size'):
                                            expected_features = lnn_model.input_size
                                            actual_features = seq_tensor.shape[-1] if len(seq_tensor.shape) > 2 else seq_tensor.shape[1]
                                            
                                            # 如果维度不匹配，尝试调整
                                            if expected_features != actual_features:
                                                self.logger.warning(f"输入特征维度不匹配: 预期{expected_features}，实际{actual_features}")
                                                # 此处可能需要调整张量维度
                                                
                                        # 使用predict方法进行预测
                                        lnn_pred = lnn_model.predict(seq_tensor.cpu().numpy())
                                        
                                        # 确保输出格式正确
                                        if isinstance(lnn_pred, np.ndarray):
                                            if lnn_pred.ndim > 1:
                                                lnn_outputs[sector_code] = float(lnn_pred[0][0])
                                            else:
                                                lnn_outputs[sector_code] = float(lnn_pred[0])
                                        else:
                                            lnn_pred_np = lnn_pred.cpu().numpy() if hasattr(lnn_pred, 'cpu') else np.array(lnn_pred)
                                            lnn_outputs[sector_code] = float(lnn_pred_np[0][0] if lnn_pred_np.ndim > 1 else lnn_pred_np[0])
                                    
                                        self.logger.debug(f"成功预测板块 {sector_code} 的LNN特征: {lnn_outputs[sector_code]}")
                                    
                                    except Exception as e:
                                        print(f"发生错误: {e}")
                                    except Exception as e_predict:
                                        self.logger.error(f"使用predict方法预测失败: {e_predict}，尝试使用模型直接调用")
                                        # 如果predict方法失败，尝试使用模型的forward方法
                                        if hasattr(lnn_model, 'model'):
                                            outputs, _ = lnn_model.model(seq_tensor)
                                            lnn_outputs[sector_code] = float(outputs.cpu().numpy()[0][0])
                                        else:
                                            # 最后的尝试，直接调用模型
                                            self.logger.warning("回退到直接调用模型")
                                            outputs, _ = lnn_model(seq_tensor)
                                            lnn_outputs[sector_code] = float(outputs.cpu().numpy()[0][0])
                                            self.logger.debug(f"使用模型直接调用成功，结果: {lnn_outputs[sector_code]}")
                            except Exception as e:
                                print(f"发生错误: {e}")
                            except Exception as e:
                                self.logger.error(f"LNN模型预测出错，板块 {sector_code}: {e}")
                                self.logger.error(traceback.format_exc())
                                continue
                    
                    # 将LNN预测结果映射到DataFrame中
                    def map_lnn_value(row):
                        sector = row['sector_code']
                        if sector in lnn_outputs:
                            return lnn_outputs[sector]
                        return 0.0
                        
                    # 使用apply添加LNN特征
                    temp_X_for_final_df[lnn_out_feat_name] = temp_X_for_final_df.apply(map_lnn_value, axis=1)
                    self.logger.info(f"成功为 {len(lnn_outputs)} 个板块添加LNN特征")
                    
                except Exception as e:
                    print(f"发生错误: {e}")
                except Exception as e:
                    self.logger.error(f"生成LNN特征时出错: {e}")
                    self.logger.error(traceback.format_exc())
                    # 使用0填充LNN特征列
                    temp_X_for_final_df[lnn_out_feat_name] = 0.0

            # final_model_input_actual_cols 已在开头初始化
            processed_X_final_parts = []
            final_model_candidate_cols = list(dict.fromkeys(final_feat_pool_cand_list + [lnn_out_feat_name]))
            # trained_cat_dtypes_map_save 已在开头初始化

            for col_meta_name in final_model_candidate_cols:
                if col_meta_name in temp_X_for_final_df.columns:
                    series_meta = temp_X_for_final_df[col_meta_name].copy() 
                    if isinstance(series_meta.dtype, pd.CategoricalDtype):
                        placeholder_cat = "MISSING_CAT_FINAL" 
                        if series_meta.isnull().any():
                            if not series_meta.isnull().all() and placeholder_cat not in series_meta.cat.categories:
                                series_meta = series_meta.cat.add_categories([placeholder_cat])
                            series_meta = series_meta.fillna(placeholder_cat)
                        # 确保分类特征是Series而不是Categorical对象
                        if not isinstance(series_meta, pd.Series):
                            series_meta = pd.Series(
                                series_meta,
                                index=temp_X_for_final_df.index,
                                name=col_meta_name
                            )
                        # 添加前再次检查类型，确保是Series
                        if isinstance(series_meta, pd.Series):
                            processed_X_final_parts.append(series_meta)
                            final_model_input_actual_cols.append(col_meta_name) 
                            trained_cat_dtypes_map[col_meta_name] = series_meta.cat.categories.tolist()
                        else:
                            self.logger.warning(f"跳过类型异常的特征: {col_meta_name}, 类型: {type(series_meta)}")
                    elif col_meta_name == 'market_group':
                        # 特殊处理market_group：将其转换为分类变量
                        self.logger.info(f"将market_group转换为分类特征")
                        market_group_categories = ['bull', 'bear', 'neutral']
                        # 确保使用Series包装分类数据
                        try:
                            series_meta = pd.Series(
                                pd.Categorical(series_meta, categories=market_group_categories),
                                index=series_meta.index,
                                name=col_meta_name
                            )
                            processed_X_final_parts.append(series_meta)
                            final_model_input_actual_cols.append(col_meta_name)
                            trained_cat_dtypes_map[col_meta_name] = market_group_categories
                        except Exception as e:
                            self.logger.warning(f"处理market_group特征时出错: {e}")
                    elif pd.api.types.is_numeric_dtype(series_meta):
                        # 对数值型特征处理缺失值
                        series_meta = series_meta.fillna(0)
                        processed_X_final_parts.append(series_meta)
                        final_model_input_actual_cols.append(col_meta_name)
                    elif pd.api.types.is_bool_dtype(series_meta):
                        # 布尔型特征也是LightGBM允许的
                        series_meta = series_meta.fillna(False)
                        processed_X_final_parts.append(series_meta)
                        final_model_input_actual_cols.append(col_meta_name)
                    else:
                        # 对于非数值、非布尔、非分类型特征（如字符串/对象类型），
                        # 我们跳过它们，因为LightGBM不支持
                        self.logger.warning(f"跳过不兼容的特征列: {col_meta_name}，类型: {series_meta.dtype}")
            
            # 记录特征处理结果
            self.logger.info(f"处理了 {len(processed_X_final_parts)} 个特征列，包括 {len([c for c in final_model_input_actual_cols if c in trained_cat_dtypes_map])} 个分类特征")
            
            if not processed_X_final_parts: 
                self.logger.error("最终模型无有效输入特征列!")
                # 记录更多调试信息
                self.logger.error(f"final_model_candidate_cols: {final_model_candidate_cols}")
                self.logger.error(f"temp_X_for_final_df列: {list(temp_X_for_final_df.columns)}")
                return None
                
            # 修复：确保所有元素都是Series或DataFrame
            valid_parts = []
            for part in processed_X_final_parts:
                if isinstance(part, pd.Series) or isinstance(part, pd.DataFrame):
                    valid_parts.append(part)
                elif hasattr(part, 'name') and hasattr(part, 'index'):
                    # 尝试将非Series/DataFrame对象转换为Series
                    try:
                        self.logger.warning(f"将类型为 {type(part)} 的对象转换为Series: {getattr(part, 'name', 'unknown')}")
                        series = pd.Series(part, index=temp_X_for_final_df.index, name=part.name)
                        valid_parts.append(series)
                    except Exception as e:
                        self.logger.error(f"转换为Series失败: {e}")
                else:
                    self.logger.warning(f"跳过类型为 {type(part)} 的无效对象")
            
            if not valid_parts:
                self.logger.error("处理后没有有效的特征列!")
                return None
                
            X_meta_train = pd.concat(valid_parts, axis=1)
            if not final_model_input_actual_cols: 
                self.logger.error("final_model_input_actual_cols 为空!"); 
                return None 
            X_meta_train = X_meta_train[final_model_input_actual_cols] 

            # 定义X_train_final和y_train_final，确保与之前的一致性
            X_train_final = X_meta_train
            y_train_final = y_final_train
            
            # 确保y_meta_train与y_train_final一致，避免未定义错误
            y_meta_train = y_train_final

            lgbm_categorical_features_fit = [c for c in X_meta_train.columns if isinstance(X_meta_train[c].dtype, pd.CategoricalDtype)]
            if not lgbm_categorical_features_fit: 
                lgbm_categorical_features_fit = 'auto'
            
            # 保存categorical_feature以便在模型元数据中使用
            categorical_feature = lgbm_categorical_features_fit

            # 使用回归模型代替分类模型，直接预测未来收益率
            final_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                objective='regression',  # 使用回归目标函数
                metric='mse',           # 使用均方误差评估
                importance_type='gain',
                random_state=42,
                verbose=-1
            )
            
            self.logger.info(f"最终回归模型使用 {X_train_final.shape[1]}特征预测未来收益率. LGBM分类特征: {categorical_feature}")
            if X_train_final.empty: 
                self.logger.error("X_train_final为空!"); 
                return None
            
            # 使用正确的变量进行模型训练
            final_model.fit(X_train_final, y_train_final, categorical_feature=categorical_feature)
            
            # 收集训练指标
            try:
                # 导入回归评估所需的库
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # 在训练集上的预测
                y_pred_train = final_model.predict(X_meta_train)
                
                # 计算回归指标
                mse_train = mean_squared_error(y_meta_train, y_pred_train)
                mae_train = mean_absolute_error(y_meta_train, y_pred_train)
                r2_train = r2_score(y_meta_train, y_pred_train)
                
                # 添加回归训练指标到metrics字典
                metrics = {
                    'train_mse': float(mse_train),
                    'train_mae': float(mae_train),
                    'train_r2': float(r2_train),
                    'train_date': train_date
                }
            except Exception as e_metrics_train:
                self.logger.warning(f"计算训练集指标时出错: {e_metrics_train}")
                metrics = {
                    'train_error': str(e_metrics_train),
                    'train_date': train_date
                }

            # 如果有验证数据，也计算验证集指标
            if not val_df.empty and 'future_return_2d' in val_df.columns:
                try:
                    # 初始化验证所需的变量
                    test_df = val_df.copy()  # 使用验证集作为测试数据
                    X_final = val_df.copy()  # 初始化特征DataFrame
                    
                    # 初始化model_metadata
                    model_metadata = {
                        'final_feature_columns': [col for col in X_train_final.columns if col not in ['date', 'sector_code', 'future_return_2d']]
                    }
                    
                    # 初始化final_feature_columns
                    final_feature_columns = model_metadata['final_feature_columns']
                    
                    # 初始化missing_cols
                    missing_cols = [col for col in final_feature_columns if col not in X_final.columns]
                    if 'lnn_output_feature' not in X_final.columns and 'lnn_output_feature' not in missing_cols:
                        missing_cols.append('lnn_output_feature')
                    
                    # 初始化分类特征相关变量
                    categorical_features = [c for c in X_train_final.columns if isinstance(X_train_final[c].dtype, pd.CategoricalDtype)]
                    trained_cat_dtypes = {col: X_train_final[col].dtype for col in categorical_features if col in X_train_final.columns}
                    label_encoders = {}  # 在实际应用中应从模型元数据中加载
                    
                    # 首先，尝试为验证集生成LNN特征
                    lnn_out_feat_name = 'lnn_output_feature'
                    
                    # 检查验证集是否包含必要的sector_code列
                    if 'sector_code' in val_df.columns and lnn_model is not None and sequence_train_data is not None:
                        self.logger.info("尝试为验证集生成真实的LNN特征...")
                        
                        # 创建一个字典来存储板块的LNN预测结果
                        val_lnn_outputs = {}
                        
                        # 创建测试LNN输出字典
                        test_lnn_outputs = {}
                        
                        # 遍历验证集中的每个板块
                        for sector_code in val_df['sector_code'].unique():
                            if sector_code in sequence_train_data:
                                try:
                                    # 提取序列特征并进行预测
                                    seq_data = sequence_train_data[sector_code]
                                    
                                    # 检查序列特征格式
                                    if isinstance(seq_data, dict) and 'features' in seq_data:
                                        sequence = seq_data['features']
                                    elif hasattr(seq_data, 'shape'):
                                        sequence = seq_data
                                    else:
                                        self.logger.warning(f"验证集: 板块 {sector_code} 的序列特征格式无效")
                                        continue
                                            
                                    # 确保序列长度足够
                                    if not hasattr(sequence, 'shape') or len(sequence) < seq_length:
                                        self.logger.warning(f"验证集: 板块 {sector_code} 的序列特征长度不足或无效")
                                        continue
                                            
                                    # 使用与训练集相同的逻辑处理序列数据
                                    recent_sequence = sequence[-seq_length:]
                                            
                                    # 确保序列至少是2D
                                    if len(recent_sequence.shape) < 2:
                                        try:
                                            recent_sequence = recent_sequence.reshape(len(recent_sequence), 1)
                                        except Exception as e:
                                            self.logger.error(f"测试数据: 重塑序列数据时出错: {e}")
                                            continue
                                            
                                    # 创建tensor变量
                                    seq_tensor = None
                                            
                                    # 应用scaler - 独立于if块，解决缩进问题
                                    if scaler is not None:
                                        try:
                                            # 确保维度正确后再应用scaler
                                            seq_flat = recent_sequence.reshape(-1, recent_sequence.shape[-1])
                                            seq_scaled = scaler.transform(seq_flat)
                                            seq_tensor = torch.FloatTensor(seq_scaled.reshape(recent_sequence.shape)).unsqueeze(0)
                                        except Exception as e:
                                            self.logger.warning(f"测试数据: 缩放序列数据时出错: {e}")
                                            # 失败则使用未缩放的数据
                                            seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                                    else:
                                        # 没有scaler时直接创建tensor
                                        seq_tensor = torch.FloatTensor(recent_sequence).unsqueeze(0)
                                            
                                    # 使用LNN模型预测 - 独立于if块，解决缩进问题
                                    try:
                                        with torch.no_grad():
                                            try:
                                                # 尝试使用predict方法
                                                lnn_pred = lnn_model.predict(seq_tensor.cpu().numpy())
                                                
                                                # 确保输出格式正确
                                                if isinstance(lnn_pred, np.ndarray):
                                                    if lnn_pred.ndim > 1:
                                                        test_lnn_outputs[sector_code] = float(lnn_pred[0][0])
                                                    else:
                                                        test_lnn_outputs[sector_code] = float(lnn_pred[0])
                                                else:
                                                    lnn_pred_np = lnn_pred.cpu().numpy() if hasattr(lnn_pred, 'cpu') else np.array(lnn_pred)
                                                    test_lnn_outputs[sector_code] = float(lnn_pred_np[0][0] if lnn_pred_np.ndim > 1 else lnn_pred_np[0])
                                            
                                            except Exception as e_predict:
                                                # 如果predict方法失败，尝试直接调用模型
                                                self.logger.error(f"测试数据: 使用predict方法预测失败: {e_predict}，尝试使用模型直接调用")
                                                
                                                # 尝试备用方法
                                                if hasattr(lnn_model, 'model'):
                                                    outputs, _ = lnn_model.model(seq_tensor)
                                                    test_lnn_outputs[sector_code] = float(outputs.cpu().numpy()[0][0])
                                                else:
                                                    outputs, _ = lnn_model(seq_tensor)
                                                    test_lnn_outputs[sector_code] = float(outputs.cpu().numpy()[0][0])
                                    except Exception as e:
                                        print(f"发生错误: {e}")
                                    except Exception as e:
                                        self.logger.error(f"测试数据: LNN模型预测出错，板块 {sector_code}: {e}")
                                        continue
                                except Exception as e:
                                    print(f"发生错误: {e}")
                                except Exception as e:
                                    self.logger.error(f"测试数据: 处理板块 {sector_code} 的LNN特征时出错: {e}")
                                    continue
                        
                        # 如果成功生成了LNN预测值，将其添加到测试数据
                        if test_lnn_outputs:
                            # 使用映射函数将LNN值添加到测试数据
                            def test_map_lnn_value(row):
                                sector = row['sector_code']
                                if sector in test_lnn_outputs:
                                    return test_lnn_outputs[sector]
                                return 0.0
                            
                            # 添加LNN特征列到测试数据
                            X_final['lnn_output_feature'] = test_df.apply(test_map_lnn_value, axis=1)
                            self.logger.info(f"成功为测试数据中的 {len(test_lnn_outputs)} 个板块添加真实LNN特征")
                            # 从missing_cols中移除lnn_output_feature
                            missing_cols.remove('lnn_output_feature')
                        else:
                            self.logger.warning("未能为测试数据生成任何LNN特征，将使用默认值0")
                            X_final['lnn_output_feature'] = 0.0
                    
                                        # 特殊处理market_group特征
                    if 'market_group' in missing_cols:
                        self.logger.info("为测试数据添加market_group特征...")
                        # 使用字符串值而不是分类对象，避免转换错误
                        X_final['market_group'] = 'neutral'  # 使用默认值'neutral'
                        # 从missing_cols中移除market_group
                        missing_cols.remove('market_group')
                    
                    # 特殊处理market_group特征
                    if 'market_group' in missing_cols:
                        self.logger.info("为测试数据添加market_group特征...")
                        # 使用字符串值而不是分类对象，避免转换错误
                        X_final['market_group'] = 'neutral'  # 使用默认值'neutral'
                        # 从missing_cols中移除market_group
                        missing_cols.remove('market_group')
                    
                                        # 特殊处理market_group特征
                    if 'market_group' in missing_cols:
                        self.logger.info("为测试数据添加market_group特征...")
                        # 使用字符串值而不是分类对象，避免转换错误
                        X_final['market_group'] = 'neutral'  # 使用默认值'neutral'
                        # 从missing_cols中移除market_group
                        missing_cols.remove('market_group')
                    
                    # 特殊处理market_group特征
                    if 'market_group' in missing_cols:
                        self.logger.info("为测试数据添加market_group特征...")
                        # 使用字符串值而不是分类对象，避免转换错误
                        X_final['market_group'] = 'neutral'  # 使用默认值'neutral'
                        # 从missing_cols中移除market_group
                        missing_cols.remove('market_group')
                    
                                        # 为其他缺失的列添加默认值0
                    for col in missing_cols:
                        X_final[col] = 0
                
                except Exception as e:
                    print(f"发生错误: {e}")
                # 确保只使用模型需要的特征列，且顺序一致
                try:
                    X_final = X_final[final_feature_columns]
                    self.logger.info(f"使用模型指定的 {len(final_feature_columns)} 个特征列")
                except KeyError as e:
                    self.logger.error(f"获取特征列时出错: {e}")
                    # 如果出错，尝试使用所有非标签列作为特征
                    non_feature_cols = ['date', 'sector_code', 'future_return_2d']
                    feature_cols = [col for col in X_final.columns if col not in non_feature_cols]
                    X_final = X_final[feature_cols]
                    self.logger.warning(f"使用所有可用特征列: {len(feature_cols)} 个")
            if 'final_feature_columns' not in model_metadata or not model_metadata['final_feature_columns']:
                # 如果没有特征列信息，使用所有非标签列
                non_feature_cols = ['date', 'sector_code', 'future_return_2d']
                non_feature_cols += [col for col in X_final.columns if col.startswith('future_return')]
                feature_cols = [col for col in X_final.columns if col not in non_feature_cols]
                X_final = X_final[feature_cols]
                self.logger.warning(f"模型元数据中缺少特征列信息，使用所有可用特征列: {len(feature_cols)} 个")
                
            # 确保分类特征的数据类型与训练时一致
            for cat_feat in categorical_features:
                if cat_feat in X_final.columns and cat_feat in trained_cat_dtypes:
                    try:
                        # 特殊处理market_group特征，避免类型转换错误
                        if cat_feat == 'market_group':
                            self.logger.info("使用安全方式处理market_group特征")
                            # 直接设置为固定字符串值，完全避开分类转换
                            X_final[cat_feat] = "neutral"  # 使用默认值作为字符串
                            # 跳过转为str的操作，避免转换错误
                            continue
                            
                        dtype = trained_cat_dtypes[cat_feat]
                        X_final[cat_feat] = X_final[cat_feat].astype(dtype)
                    except Exception as e:
                        self.logger.warning(f"无法转换分类特征 {cat_feat} 的数据类型: {str(e)}")
            
            # 应用标签编码器
            for col, encoder in label_encoders.items():
                if col in X_final.columns:
                    try:
                        # 确保处理未见过的值
                        unique_vals = X_final[col].unique()
                        for val in unique_vals:
                            if val not in encoder.classes_:
                                # 对未见过的值进行特殊处理，例如使用最常见的类别
                                self.logger.warning(f"列 {col} 中存在未见过的值: {val}，将替换为最常见类别")
                                X_final.loc[X_final[col] == val, col] = encoder.classes_[0]
                        
                        X_final[col] = encoder.transform(X_final[col])
                    except Exception as e:
                        self.logger.warning(f"标签编码失败 {col}: {str(e)}")
            
            # 7. 使用最终模型进行预测
            final_pred = None
            try:
                # 1. 检查特征列是否与训练时一致
                if final_feature_columns and len(final_feature_columns) > 0:
                    # 确保预测数据中包含所有需要的特征列，顺序与训练时一致
                    missing_cols = [col for col in final_feature_columns if col not in X_final.columns]
                    if missing_cols:
                        self.logger.warning(f"预测数据中缺少 {len(missing_cols)}/{len(final_feature_columns)} 特征列")
                        # 如果缺失的列太多，可能会影响预测质量
                        if len(missing_cols) > len(final_feature_columns) * 0.3:  # 超过30%特征缺失
                            self.logger.error(f"缺失特征过多，可能会严重影响预测质量")
                        
                        # 为缺失的列添加默认值0
                        for col in missing_cols:
                            X_final[col] = 0
                    
                    # 确保按照训练时的列顺序选择特征
                    X_final = X_final[final_feature_columns]
                
                # 2. 处理分类特征编码
                if isinstance(categorical_features, list) and categorical_features:
                    for cat_feat in categorical_features:
                        if cat_feat in X_final.columns:
                            # 尝试将分类特征转换为符合训练时的格式
                            if cat_feat in trained_cat_dtypes:
                                try:
                                    # 记录转换前的数据类型
                                    before_dtype = str(X_final[cat_feat].dtype)
                                    
                                    # 获取训练时的数据类型
                                    train_dtype = trained_cat_dtypes[cat_feat]
                                    
                                    # 转换类型
                                    X_final[cat_feat] = X_final[cat_feat].astype(train_dtype)
                                    self.logger.info(f"成功转换特征 {cat_feat} 的数据类型: {before_dtype} -> {train_dtype}")
                                except Exception as e:
                                    self.logger.warning(f"转换特征 {cat_feat} 的数据类型时出错: {e}")
                
                # 3. 确保数据类型一致，对所有列进行类型检查
                for col in X_final.columns:
                    # 检查是否为空值，如果是则填充为0
                    if X_final[col].isnull().any():
                        X_final[col] = X_final[col].fillna(0)
                    
                    # 确保是数值类型
                    if not pd.api.types.is_numeric_dtype(X_final[col]):
                        try:
                            X_final[col] = pd.to_numeric(X_final[col], errors='coerce').fillna(0)
                        except Exception as e:
                            self.logger.warning(f"将列 {col} 转换为数值类型时出错: {e}")
                
                # 4. 尝试不同的预测方法，增加健壮性
                # 尝试直接使用booster进行预测，设置predict_disable_shape_check=True
                if hasattr(final_model, 'booster_') and hasattr(final_model.booster_, 'predict'):
                    self.logger.info("使用LightGBM booster直接预测，设置predict_disable_shape_check=True")
                    try:
                        final_pred = final_model.booster_.predict(X_final.values, 
                                                            pred_leaf=False, 
                                                            pred_contrib=False,
                                                            predict_disable_shape_check=True)
                    except Exception as e1:
                        self.logger.warning(f"使用booster.predict方法失败: {e1}, 尝试其他方法")
                
                # 如果上面的方法失败，尝试使用predict_proba
                if final_pred is None and hasattr(final_model, 'predict_proba'):
                    self.logger.info("使用predict_proba方法预测")
                    try:
                        # 对于LightGBM模型，先修改_Booster的predict参数，再调用predict_proba
                        if hasattr(final_model, '_Booster'):
                            # 临时保存原方法
                            original_predict = final_model._Booster.predict
                            # 创建带有predict_disable_shape_check的包装方法
                            def wrapped_predict(*args, **kwargs):
                                kwargs['predict_disable_shape_check'] = True
                                return original_predict(*args, **kwargs)
                            # 替换方法
                            final_model._Booster.predict = wrapped_predict
                            
                        # 尝试预测
                        final_pred = final_model.predict_proba(X_final)
                        
                        # 恢复原方法
                        if hasattr(final_model, '_Booster'):
                            final_model._Booster.predict = original_predict
                            
                        # 对于二分类问题，取正类的概率
                        if hasattr(final_pred, 'shape') and final_pred.shape[1] > 1:
                            final_pred = final_pred[:, 1]
                    except Exception as e2:
                        self.logger.warning(f"使用predict_proba方法失败: {e2}, 尝试其他方法")
                
                # 如果前两种方法都失败，尝试使用标准predict方法
                if final_pred is None:
                    self.logger.info("使用predict方法预测")
                    try:
                        final_pred = final_model.predict(X_final)
                    except Exception as e3:
                        self.logger.error(f"所有预测方法都失败: {e3}")
                        # 使用默认值
                        final_pred = np.zeros(len(X_final))
            except Exception as e:
                self.logger.error(f"最终模型预测过程中发生错误: {str(e)}")
                self.logger.error(traceback.format_exc())
                # 使用默认值
                final_pred = np.zeros(len(X_final))
            
            # 8. 结果整合与评估
            result_df = pd.DataFrame()
            result_df['sector_code'] = test_df['sector_code']
            result_df['prediction'] = final_pred  # 收益率预测值
            
            # 如果有sector_name列，也包含它
            if 'sector_name' in test_df.columns:
                result_df['sector_name'] = test_df['sector_name']
            
            # 排序并添加排名（按预测收益率降序）
            result_df = result_df.sort_values(by='prediction', ascending=False)
            result_df['rank'] = range(1, len(result_df) + 1)
            
            # 计算收益率预测的评估指标
            metrics = {}
            future_return_col = 'future_return_2d'  # 实际收益率列
            
            # 如果测试数据中有未来收益，计算回归评估指标
            if future_return_col in test_df.columns:
                try:
                    # 添加实际收益率到结果
                    result_df['actual_return'] = test_df[future_return_col]
                    
                    # 计算回归指标：MSE、MAE、R²
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(result_df['actual_return'], result_df['prediction'])
                    mae = mean_absolute_error(result_df['actual_return'], result_df['prediction'])
                    r2 = r2_score(result_df['actual_return'], result_df['prediction'])
                    
                    # 方向准确率：预测与实际收益率正负方向一致的比例
                    direction_accuracy = np.mean((result_df['prediction'] > 0) == (result_df['actual_return'] > 0))
                    
                    # 计算前N名预测的实际平均收益率
                    top_n = min(5, len(result_df))
                    top_n_avg_return = result_df.head(top_n)['actual_return'].mean()
                    top_n_positive_ratio = (result_df.head(top_n)['actual_return'] > 0).mean()
                    
                    metrics['mse'] = mse
                    metrics['mae'] = mae
                    metrics['rmse'] = np.sqrt(mse)
                    metrics['r2'] = r2
                    metrics['direction_accuracy'] = direction_accuracy
                    metrics[f'top{top_n}_avg_return'] = top_n_avg_return
                    metrics[f'top{top_n}_positive_ratio'] = top_n_positive_ratio
                    
                    self.logger.info(f"回归评估指标: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")
                    self.logger.info(f"方向准确率: {direction_accuracy:.4f}, Top{top_n}平均收益: {top_n_avg_return:.4f}")
                except Exception as e:
                    self.logger.error(f"计算回归评估指标时出错: {e}")
                    self.logger.error(traceback.format_exc())
            
            # 为预测结果添加顶部板块信息
            # 选择预测排名前5的板块作为推荐
            top_n = min(5, len(result_df))
            top_sectors = result_df.head(top_n)['sector_code'].tolist()
            
            # 记录预测的顶部板块
            metrics['top_sectors'] = top_sectors
            # 使用date变量替代未定义的prediction_date
            metrics['date'] = date_obj.strftime('%Y%m%d') if isinstance(date_obj, (datetime, pd.Timestamp)) else date_obj
            
            # 添加板块名称，如果可能的话
            if 'sector_name' in test_df.columns:
                sector_name_map = {}
                for _, row in test_df.iterrows():
                    sector_name_map[row['sector_code']] = row['sector_name'] 
                
                top_sector_names = [sector_name_map.get(code, code) for code in top_sectors]
                metrics['top_sector_names'] = top_sector_names
                
                # 记录推荐板块及其预测收益率
                self.logger.info(f"日期 {date_obj} 的推荐板块:")
                for i, (code, name) in enumerate(zip(top_sectors, top_sector_names), 1):
                    pred_return = result_df[result_df['sector_code'] == code]['prediction'].values[0]
                    self.logger.info(f"  {i}. {name} ({code}): 预测收益率 {pred_return:.4f}")
            
            # 返回更丰富的结果信息
            # 创建预测结果字典
            prediction_results = {
                'predictions': result_df,
                'test_df': test_df,
                'metrics': metrics,
                'date': date_obj
            }
            
            # 创建完整的模型包
            model_package = {
                'market_classifier': market_model,
                'strength_predictor': strength_model,  # 现在是回归模型，不再是分类器
                'lgb_model': final_model,
                'final_model': final_model,
                'model': final_model, # 兼容旧的键名
                'lnn': lnn_model,
                'scaler': scaler, # 保存为LNN序列特征配套的scaler
                'calibrator': calibrator if 'calibrator' in locals() and calibrator is not None else None,  # 添加校准器
                'performance_history': {  # 添加性能历史记录字段
                    'strength_model_rmse': 0.05,  # 默认初始值
                    'strength_model_direction': 0.5,  # 默认初始值
                    'lnn_model_rmse': 0.05,  # 默认初始值
                    'lnn_model_direction': 0.5,  # 默认初始值
                },

                **market_submodels, # 包含其模型和特征 {'bull_model': model, 'bull_model_features': [...]}

                'metadata': {
                    # 每个模型实际使用的确切特征列表 - 用于严格特征对齐
                    'market_classifier_input_features': X_market_train.columns.tolist() if 'X_market_train' in locals() and hasattr(X_market_train, 'columns') else [],
                    'strength_predictor_input_features': X_strength_train.columns.tolist() if 'X_strength_train' in locals() and hasattr(X_strength_train, 'columns') else [], 
                    'final_model_input_features': X_final.columns.tolist() if 'X_final' in locals() and hasattr(X_final, 'columns') else [],
                    
                    # 保留原有字段以兼容性，但明确其用途
                    'market_features': market_env_feats_train if 'market_env_feats_train' in locals() else [],
                    'strength_model_actual_features': X_strength.columns.tolist() if 'X_strength' in locals() and hasattr(X_strength, 'columns') else [],
                    
                    # LNN 模型相关的元数据
                    'lnn_input_features': sequence_features.shape[2] if 'sequence_features' in locals() and hasattr(sequence_features, 'shape') and len(sequence_features.shape) > 2 else 0,
                    'lnn_base_feature_names': sequence_train_data.get(next(iter(sequence_train_data)), {}).get('feature_names', []) if sequence_train_data else [], 
                    'seq_length': seq_length if 'seq_length' in locals() else 0,
                    
                    # 最终模型实际使用的特征 - 与 final_model_input_features 保持一致性
                    'final_feature_columns': final_model_input_actual_cols if 'final_model_input_actual_cols' in locals() else [],
                    
                    # 分类特征相关信息
                    'categorical_features': categorical_feature if 'categorical_feature' in locals() else [], 
                    'trained_cat_dtypes': trained_cat_dtypes_map if 'trained_cat_dtypes_map' in locals() else {}, 
                    
                    # 训练时间和模型版本信息
                    'training_date': train_date.strftime('%Y%m%d'),
                    'model_version': '2.1.0', # 更新版本号表示回归模型
                    'feature_pipeline_version': '1.1.0', # 特征流水线版本
                    'model_type': 'regression',  # 标记模型类型为回归
                    'target_column': 'future_return_2d'  # 指定目标列
                },
                'feature_processor': feature_processor,  # 添加特征处理器
                'prediction_results': prediction_results if 'prediction_results' in locals() else {} # 训练日的预测结果（可选）
            }
            self.logger.info(f"成功创建模型包 ({train_date.strftime('%Y%m%d')})，包含关键组件和详细元数据。")
            return model_package
            
        except Exception as e:
            print(f"发生错误: {e}")
        except Exception as e:
            self.logger.error(f"预测过程中发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    
    def _predict_with_rolling_model(self, model_package, features_df, prediction_date):
        """使用滚动模型进行预测

        Args:
            model_package: 模型包
            features_df: 特征DataFrame
            prediction_date: 预测日期

        Returns:
            预测结果DataFrame 或包含预测结果和指标的字典
        """
        try:
            # 1. 准备数据
            if not isinstance(prediction_date, pd.Timestamp):
                date_obj = pd.to_datetime(prediction_date)
            else:
                date_obj = prediction_date

            # 过滤出预测日期的特征数据
            # 创建副本以避免 SettingWithCopyWarning
            test_df = features_df[features_df['date'] == date_obj].copy()

            if test_df.empty:
                self.logger.warning(f"日期 {prediction_date} 没有特征数据")
                return {'predictions': pd.DataFrame(), 'test_df': pd.DataFrame(), 'metrics': {}, 'date': date_obj, 'error': 'No feature data'}

            # 2. 获取模型组件
            market_model = model_package.get('market_classifier')
            strength_model = model_package.get('strength_predictor')

            final_model = None
            for key_name in ['final_ensemble', 'model', 'final_model', 'lgb_model']:
                if key_name in model_package:
                    final_model = model_package[key_name]
                    self.logger.info(f"通过键名 '{key_name}' 找到final_model")
                    break

            if final_model is None:
                self.logger.error("无法找到最终模型，请检查模型包键名")
                return {'predictions': pd.DataFrame(), 'test_df': test_df, 'metrics': {}, 'date': date_obj, 'error': 'Final model not found in package'}

            lnn_model = model_package.get('lnn')
            # market_submodels = {k: v for k, v in model_package.items() if '_submodel' in k} # 当前未使用

            model_metadata = model_package.get('metadata', {})
            if not model_metadata:
                self.logger.error("模型包中缺少元数据 'metadata'")
                return {'predictions': pd.DataFrame(), 'test_df': test_df, 'metrics': {}, 'date': date_obj, 'error': 'Metadata missing in model package'}

            # 获取特征处理器
            feature_processor = None
            if 'feature_processor' in model_package:
                feature_processor = model_package['feature_processor']
                self.logger.info("从模型包获取特征处理器")
            else:
                # 如果模型包没有特征处理器，从元数据创建一个
                self.logger.info("模型包中没有特征处理器，尝试从元数据创建")
                feature_processor = FeatureProcessor()
                
                # 构建特征元数据
                feature_metadata = {
                    'categorical_features': model_metadata.get('categorical_features', []),
                    'categorical_mappings': model_metadata.get('trained_cat_dtypes', {}),
                    'required_features': model_metadata.get('final_feature_columns', []),
                    'feature_defaults': {},
                    'numeric_features': [],
                    'date_features': []
                }
                
                # 尝试推断数值特征
                if 'final_feature_columns' in model_metadata:
                    numeric_features = [
                        col for col in model_metadata['final_feature_columns'] 
                        if col not in feature_metadata['categorical_features']
                    ]
                    feature_metadata['numeric_features'] = numeric_features
                
                feature_processor.initialize_from_metadata(feature_metadata)
            
            # 应用特征处理
            if feature_processor and feature_processor.is_initialized:
                self.logger.info("应用特征处理器处理测试数据...")
                test_df = feature_processor.process_features(test_df, is_training=False)
            else:
                self.logger.warning("特征处理器不可用或未初始化，将使用原始数据")

            # 严格使用训练时保存的各模型的输入特征列表
            # 获取市场模型特征列表 - 优先使用精确的输入特征列表
            market_features = model_metadata.get('market_classifier_input_features', [])
            if not market_features:
                # 兼容旧版本，使用'market_features'
                market_features = model_metadata.get('market_features', [])
                self.logger.warning("使用兼容性market_features，精确特征列表未找到")
                
            # 获取强度模型特征列表 - 优先使用精确的输入特征列表
            strength_model_actual_features = model_metadata.get('strength_predictor_input_features', [])
            if not strength_model_actual_features:
                # 兼容旧版本，使用'strength_model_actual_features'
                strength_model_actual_features = model_metadata.get('strength_model_actual_features', [])
                if not strength_model_actual_features:
                    self.logger.error("强度模型训练特征列表在元数据中完全缺失！")
                    return {'predictions': pd.DataFrame(), 'test_df': test_df, 'metrics': {}, 'date': date_obj, 'error': 'Strength model features missing'}

            # 获取最终模型特征列表 - 优先使用精确的输入特征列表
            final_feature_columns = model_metadata.get('final_model_input_features', [])
            if not final_feature_columns:
                # 兼容旧版本，使用'final_feature_columns'
                final_feature_columns = model_metadata.get('final_feature_columns', [])
                self.logger.warning("使用兼容性final_feature_columns，精确特征列表未找到")
            
            # 处理分类特征 - 应为列名列表
            categorical_features_names = model_metadata.get('categorical_features', [])
            if isinstance(categorical_features_names, str) and categorical_features_names == 'auto': # LightGBM 'auto' case
                categorical_features_names = [col for col, dtype_info in model_metadata.get('trained_cat_dtypes', {}).items() if isinstance(dtype_info, list)] # 推断
            elif not isinstance(categorical_features_names, list):
                self.logger.warning(f"元数据中的 'categorical_features' 格式不正确（应为列表），实际为: {type(categorical_features_names)}。将尝试从 trained_cat_dtypes 推断。")
                categorical_features_names = [col for col, dtype_info in model_metadata.get('trained_cat_dtypes', {}).items() if isinstance(dtype_info, list)]

            # 获取分类特征的类别映射
            trained_cat_dtypes = model_metadata.get('trained_cat_dtypes', {})
            
            # 记录特征数量信息，便于调试
            self.logger.info(f"特征列数量统计: 市场特征={len(market_features)}, 强度特征={len(strength_model_actual_features)}, 最终特征={len(final_feature_columns)}")
            # label_encoders = model_metadata.get('label_encoders', {}) # 当前未使用

            if 'date' in test_df.columns: # 备份date列
                date_backup = test_df['date'].copy()

            # 3. 测试数据预处理 (future_return_2d 标签)
            target_col = 'future_return_2d'
            has_return_data = target_col in test_df.columns
            if not has_return_data:
                self.logger.warning(f"测试数据中没有 '{target_col}'，无法评估预测质量。")
            else:
                # 记录收益率数据统计信息
                self.logger.info(f"收益率数据统计: 平均值={test_df[target_col].mean():.4f}, 标准差={test_df[target_col].std():.4f}")
                self.logger.info(f"正收益比例: {(test_df[target_col] > 0).mean():.2%}")


            if 'date' not in test_df.columns and 'date_backup' in locals() and date_backup is not None:
                test_df['date'] = date_backup # 恢复date列

            # 4. 应用特征流水线处理测试数据
            self.logger.info("应用特征流水线处理测试数据...")
            test_df = self._apply_feature_pipeline(test_df, None, mode='predict')
            
            # 5. 应用市场环境模型 (获取 market_prob_ 特征) - 使用正确的特征列表
            self.logger.info("应用市场环境分类器...")
            # 严格使用训练时保存的市场模型特征列表
            market_input_features = market_features
            if market_model is not None and market_input_features:
                try:
                    # 确保所有必需特征都存在
                    for f in market_input_features:
                        if f not in test_df.columns:
                            test_df[f] = 0.0
                            self.logger.debug(f"添加缺失市场特征: {f}")
                    
                    # 严格按照训练时的特征列表和顺序提取数据
                    market_X_pred = test_df[market_input_features].fillna(0.0).astype(float)
                    
                    # 记录诊断信息
                    self.logger.info(f"市场分类器输入特征统计: 特征数={len(market_input_features)}, 样本数={len(market_X_pred)}")
                    
                    # 预测市场状态概率
                    market_preds_proba = market_model.predict_proba(market_X_pred)
                    
                    # 添加预测结果到测试数据
                    for i in range(market_preds_proba.shape[1]):
                        test_df[f'market_prob_{i}'] = market_preds_proba[:, i]
                    self.logger.info(f"市场分类器预测成功: 市场类别数={market_preds_proba.shape[1]}")
                    
                except Exception as e:
                    self.logger.warning(f"市场分类器预测失败: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # 创建默认的 market_prob_ 列
                    num_market_classes = getattr(market_model, 'n_classes_', 3)
                    self.logger.info(f"使用默认市场概率: 市场类别数={num_market_classes}")
                    for i in range(num_market_classes):
                        test_df[f'market_prob_{i}'] = 1.0 / num_market_classes
            else:
                self.logger.warning("市场分类器不可用或特征列表为空，使用默认市场概率")
                # 检查下游模型是否依赖market_prob_特征
                market_prob_needed = any(f.startswith('market_prob_') for f in strength_model_actual_features + final_feature_columns)
                if market_prob_needed:
                    num_expected_probs = 3  # 默认三个市场类别
                    self.logger.info(f"为下游模型创建默认市场概率: 类别数={num_expected_probs}")
                    for i in range(num_expected_probs):
                        test_df[f'market_prob_{i}'] = 1.0 / num_expected_probs


            # 6. 应用强度预测模型 - 使用正确的特征列表
            self.logger.info(f"应用强度预测模型: 特征数={len(strength_model_actual_features)}")
            if strength_model is not None and strength_model_actual_features:
                try:
                    # 确保所有强度特征都存在
                    for f in strength_model_actual_features:
                        if f not in test_df.columns:
                            test_df[f] = 0.0
                            self.logger.debug(f"添加缺失强度特征: {f}")
                    
                    # 严格按照训练时的特征列表和顺序提取数据
                    strength_X = test_df[strength_model_actual_features].fillna(0.0).astype(float)
                    
                    # 记录诊断信息
                    self.logger.info(f"强度模型输入特征统计: 特征数={len(strength_model_actual_features)}, 样本数={len(strength_X)}")
                    
                    # 预测强度值
                    strength_pred = strength_model.predict(strength_X)
                    test_df['predicted_strength'] = strength_pred
                    self.logger.info(f"强度模型预测成功: 样本数={len(strength_pred)}, 平均强度={np.mean(strength_pred):.4f}")
                    
                except Exception as e:
                    self.logger.error(f"强度模型预测失败: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    test_df['predicted_strength'] = 0.5  # 默认中性强度值
                    self.logger.warning("使用默认强度值0.5替代")
            else:
                self.logger.warning("强度模型不可用或特征列表为空，使用默认强度值0.5")
                test_df['predicted_strength'] = 0.5  # 默认中性强度值


            # 6. 准备LNN模型的输入特征 (如果使用)
            if lnn_model is not None and model_package.get('seq_length', 0) > 0:
                self.logger.info("为LNN模型准备序列特征...")
                try:
                    seq_length = model_package.get('seq_length', 10)
                    # 注意：_create_sequence_features 需要原始的 features_df (包含历史数据)
                    # 而不是只有当天数据的 test_df
                    # 同时，它需要知道用于构建序列的基础特征列名
                    lnn_base_feature_names = model_metadata.get('lnn_base_feature_names') # 假设训练时保存了这个
                    if not lnn_base_feature_names:
                        # 如果没保存，尝试从 final_feature_columns 推断（排除 predicted_strength, lnn_output_feature, market_group 等）
                        # 这是一个不太可靠的后备方案
                        non_lnn_seq_cols = ['predicted_strength', 'lnn_output_feature', 'market_group'] + \
                                        [f'market_prob_{i}' for i in range(10)] # 假设最多10个market_prob
                        lnn_base_feature_names = [f for f in final_feature_columns if f not in non_lnn_seq_cols and f in features_df.columns]

                    if lnn_base_feature_names:
                        # _create_sequence_features 应该返回一个字典，键是sector_code
                        # 值是包含 'features' (numpy array) 的字典
                        all_sector_codes_today = test_df['sector_code'].unique()
                        # 传递原始的 features_df (包含历史) 和当日的板块列表
                        seq_data_dict = self._create_sequence_features(
                            features_df, # 传递包含历史数据的总features_df
                            sector_list=all_sector_codes_today,
                            end_date=date_obj, # 确保序列在预测日结束
                            seq_length=seq_length
                            # feature_cols=lnn_base_feature_names # 如果_create_sequence_features支持指定列
                        )

                        lnn_predictions_map = {}
                        if seq_data_dict:
                            scaler_for_lnn = model_package.get('scaler') # LNN训练时用的scaler
                            if scaler_for_lnn is None:
                                self.logger.warning("LNN的scaler未在模型包中找到，LNN预测可能不准确。")

                            for sector_code, data_packet in seq_data_dict.items():
                                if 'features' in data_packet and data_packet['features'].shape[0] == seq_length:
                                    sequence_np = data_packet['features'] # [seq_length, num_features]
                                    if scaler_for_lnn:
                                        try:
                                            # LNN的scaler通常在2D数据上fit，所以需要reshape
                                            sequence_np_flat = sequence_np.reshape(-1, sequence_np.shape[-1])
                                            sequence_np_scaled_flat = scaler_for_lnn.transform(sequence_np_flat)
                                            sequence_np_scaled = sequence_np_scaled_flat.reshape(sequence_np.shape)
                                        except Exception as e_scale:
                                            self.logger.warning(f"缩放板块 {sector_code} 的LNN序列特征失败: {e_scale}。将使用原始特征。")
                                            sequence_np_scaled = sequence_np
                                    else:
                                        sequence_np_scaled = sequence_np

                                    # 增加batch维度: [1, seq_length, num_features]
                                    lnn_input_tensor = torch.FloatTensor(sequence_np_scaled).unsqueeze(0).to(lnn_model.device)
                                    pred_val = lnn_model.predict(lnn_input_tensor.cpu().numpy()) # predict接受numpy
                                    lnn_predictions_map[sector_code] = pred_val[0] if pred_val.ndim > 0 else pred_val # 取单个预测值
                                else:
                                    self.logger.warning(f"板块 {sector_code} 的LNN序列数据不符合要求，形状: {data_packet.get('features', np.array([])).shape}")
                            test_df['lnn_output_feature'] = test_df['sector_code'].map(lnn_predictions_map).fillna(0.0)
                            self.logger.info(f"成功为 {len(lnn_predictions_map)} 个板块生成LNN特征。")
                        else:
                            self.logger.warning("未能为任何板块创建LNN序列特征。")
                            test_df['lnn_output_feature'] = 0.0
                    else:
                        self.logger.warning("LNN基础特征列名未定义，无法生成LNN特征。")
                        test_df['lnn_output_feature'] = 0.0
                except Exception as e:
                    self.logger.error(f"LNN模型准备过程中发生错误: {e}")
                    self.logger.error(traceback.format_exc())
                    # 如果模型包中没有LNN，或者不需要LNN特征，确保该列存在并为0（如果final_model期望它）
                    if 'lnn_output_feature' in final_feature_columns:
                        test_df['lnn_output_feature'] = 0.0


            # 7. 准备最终模型的输入特征 X_final
            if not final_feature_columns:
                self.logger.error("最终模型的特征列列表 (final_feature_columns) 为空！")
                # 使用test_df中所有数值列（排除特定列）作为后备
                excluded_for_final = ['date', 'sector_code', 'label_date'] + \
                                    [col for col in test_df.columns if 'future_return' in col]
                final_feature_columns = [col for col in test_df.select_dtypes(include=np.number).columns if col not in excluded_for_final]
                self.logger.warning(f"将使用推断的数值特征列 ({len(final_feature_columns)}) 进行最终预测。")
                if not final_feature_columns:
                    self.logger.error("推断最终特征列失败，无法继续预测。")
                    return {'predictions': pd.DataFrame(), 'test_df': test_df, 'metrics': {}, 'date': date_obj, 'error': 'Final features could not be determined'}

            # 在检查缺失特征前，确保关键特征已生成
            # 1. 确保market_group特征已生成 - 与_process_features_with_layers保持一致
            if 'market_group' not in test_df.columns and 'market_type' in test_df.columns:
                self.logger.info("主动生成market_group特征，基于market_type")
                # 创建临时副本避免PerformanceWarning
                test_df_copy = test_df.copy()
                
                # 基于market_type映射到market_group（与训练时逻辑保持一致）
                def map_market_group(market_type):
                    if isinstance(market_type, (int, float)):
                        if market_type in [11, 12, 13, 21, 22, 23]:  # 上涨趋势
                            return "bull"
                        elif market_type in [41, 42, 43, 51, 52, 53]:  # 下跌趋势
                            return "bear"
                        else:  # 震荡或不确定
                            return "neutral"
                    elif isinstance(market_type, str) and market_type.lower() in ["bull", "bear", "neutral"]:
                        return market_type.lower()
                    else:
                        return "neutral"  # 默认值
                
                test_df_copy['market_group'] = test_df_copy['market_type'].apply(map_market_group)
                test_df = test_df_copy
            
            # 2. 确保lnn_output_feature已生成 - 使用我们的辅助方法
            if 'lnn_output_feature' not in test_df.columns:
                self.logger.info("使用_ensure_lnn_features方法生成lnn特征")
                lnn_model = model_package.get('lnn')  # 获取LNN模型
                test_df = self._ensure_lnn_features(test_df, lnn_model)
                
            # 检查并处理剩余缺失的特征
            missing_final_fs = [f for f in final_feature_columns if f not in test_df.columns]
            if missing_final_fs:
                self.logger.warning(f"测试数据中仍有缺失特征: {missing_final_fs}，进行备用处理")
                # 创建临时副本避免PerformanceWarning
                test_df_copy = test_df.copy()
                
                # 处理可能仍然缺失的特征
                if 'market_group' in missing_final_fs:
                    # 从模型元数据获取分类值
                    if 'categorical_mappings' in model_metadata and 'market_group' in model_metadata['categorical_mappings']:
                        categories = model_metadata['categorical_mappings']['market_group']
                        test_df_copy['market_group'] = categories[0] if categories else 'bull'
                    else:
                        test_df_copy['market_group'] = 'neutral'  # 使用中性市场作为默认值
                    missing_final_fs.remove('market_group')
                
                if 'lnn_output_feature' in missing_final_fs:
                    # 再次调用我们的辅助方法来确保特征存在
                    test_df_copy = self._ensure_lnn_features(test_df_copy, lnn_model)
                    missing_final_fs.remove('lnn_output_feature')
                    
                test_df = test_df_copy

                # 处理其他缺失特征
                for f in missing_final_fs:
                    if '_momentum_' in f or '_strength_' in f:
                        # 动量特征，从其他相似特征中估算
                        similar_features = [col for col in test_df.columns if '_momentum_' in col or '_strength_' in col]
                        if similar_features:
                            test_df[f] = test_df[similar_features].mean(axis=1)
                        else:
                            test_df[f] = 0.0  # 默认中性值
                    elif '_ratio' in f or '_rate' in f:
                        # 比率类特征通常在0-1范围，使用中间值
                        test_df[f] = 0.5
                    else:
                        # 其他未知特征，使用0值
                        self.logger.warning(f"未找到特征 {f} 的生成方法，使用0值")
                        test_df[f] = 0.0

            X_final = test_df[final_feature_columns].fillna(0.0).copy() # 使用副本

            # 确保分类特征类型正确，特别是market_group
            categorical_features_names = model_metadata.get('categorical_features', [])
            trained_cat_dtypes = model_metadata.get('trained_cat_dtypes', {})
            
            # 特殊处理market_group
            if 'market_group' in X_final.columns and 'market_group' in trained_cat_dtypes:
                self.logger.info("使用LightGBM booster直接预测，设置predict_disable_shape_check=True")
                # 确保market_group是分类类型
                try:
                    # 先转为字符串类型处理0.0问题
                    if not isinstance(X_final['market_group'].dtype, pd.CategoricalDtype):
                        # 转换为字符串再处理
                        X_final['market_group'] = X_final['market_group'].astype(str)
                        
                        # 处理0.0值
                        if '0.0' in X_final['market_group'].values:
                            X_final['market_group'] = X_final['market_group'].replace('0.0', 'bull')
                            
                        # 获取训练时的类别
                        categories = trained_cat_dtypes['market_group']
                        if not categories:
                            categories = ['bull', 'bear', 'neutral']
                            
                        # 检查并替换未知类别
                        unknown_values = [x for x in X_final['market_group'].unique() if x not in categories]
                        if unknown_values:
                            for val in unknown_values:
                                X_final['market_group'] = X_final['market_group'].replace(val, categories[0])
                                
                        # 转为分类类型
                        X_final['market_group'] = pd.Categorical(
                            X_final['market_group'],
                            categories=categories
                        )
                except Exception as e:
                    self.logger.warning(f"转换market_group类型时出错: {e}")
                    # 如果转换失败，保持为字符串类型
                    X_final['market_group'] = X_final['market_group'].astype(str)

            # 8. 使用最终回归模型进行预测 - 直接预测未来收益率
            final_pred = None
            try:
                self.logger.info(f"传递给最终回归模型的特征形状: {X_final.shape}")
                
                # 4. 尝试不同的预测方法，增加健壮性
                # 尝试直接使用booster进行预测，设置predict_disable_shape_check=True
                if hasattr(final_model, 'booster_') and hasattr(final_model.booster_, 'predict'):
                    self.logger.info("使用LightGBM booster直接预测，设置predict_disable_shape_check=True")
                    try:
                        final_pred = final_model.booster_.predict(X_final.values, 
                                                         pred_leaf=False, 
                                                         pred_contrib=False,
                                                         predict_disable_shape_check=True)
                    except Exception as e1:
                        self.logger.warning(f"使用booster.predict方法失败: {e1}, 尝试其他方法")
                        final_pred = None
                
                # 如果上述方法失败，尝试使用wrapped_predict方法
                if final_pred is None and hasattr(final_model, 'wrapped_predict'):
                    self.logger.info("使用wrapped_predict方法预测")
                    try:
                        def wrapped_predict(*args, **kwargs):
                            try:
                                return final_model.predict(*args, **kwargs)
                            except Exception as e_inner:
                                self.logger.warning(f"predict方法内部错误: {e_inner}")
                                return np.zeros(args[0].shape[0])
                                
                        final_pred = wrapped_predict(X_final)
                    except Exception as e2:
                        self.logger.warning(f"使用wrapped_predict方法失败: {e2}, 尝试标准predict")
                        final_pred = None
                
                # 如果前两种方法都失败，尝试使用标准predict方法
                if final_pred is None:
                    self.logger.info("使用标准predict方法预测未来收益率")
                    final_pred = final_model.predict(X_final)

            except Exception as e:
                self.logger.error(f"最终模型预测过程中发生错误: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.logger.error(f"X_final dtypes:\n{X_final.dtypes.to_string()}")
                final_pred = np.zeros(len(X_final)) # 默认预测为0


            # 9. 结果整合与评估 - 回归预测结果
            result_df = pd.DataFrame({'sector_code': test_df['sector_code']})
            result_df['prediction'] = final_pred if final_pred is not None else 0.0 # 确保 prediction 列存在
            result_df['predicted_return'] = result_df['prediction']  # 复制一份，更明确地表示这是预测的收益率

            if 'sector_name' in test_df.columns:
                result_df['sector_name'] = test_df['sector_name']

            # 按预测收益率倒序排列，选出预测收益最高的板块
            result_df = result_df.sort_values(by='prediction', ascending=False).reset_index(drop=True)
            result_df['rank'] = result_df.index + 1

            # 10. 计算回归模型的评估指标
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics_output = {}
            future_return_col = 'future_return_2d'  # 实际收益率列名
            
            if future_return_col in test_df.columns:  # 确保未来收益率列存在
                try:
                    # 将实际收益率合并到result_df
                    actual_returns_map = test_df.set_index('sector_code')[future_return_col]
                    result_df['actual_return'] = result_df['sector_code'].map(actual_returns_map)
                    
                    # 过滤掉实际收益为NaN的情况
                    valid_results = result_df.dropna(subset=['actual_return'])
                    
                    if not valid_results.empty:
                        # 计算回归指标
                        y_true = valid_results['actual_return']
                        y_pred = valid_results['prediction']
                        
                        metrics_output['mse'] = float(mean_squared_error(y_true, y_pred))
                        metrics_output['rmse'] = float(np.sqrt(metrics_output['mse']))
                        metrics_output['mae'] = float(mean_absolute_error(y_true, y_pred))
                        metrics_output['r2'] = float(r2_score(y_true, y_pred))
                        
                        # 计算正确预测方向的比例
                        direction_correct = np.mean((y_pred > 0) == (y_true > 0))
                        metrics_output['direction_accuracy'] = float(direction_correct)
                        
                        # 计算预测前5名板块的平均实际收益
                        top5_actual_returns = valid_results.head(5)['actual_return'].values
                        metrics_output['top5_avg_return'] = float(np.mean(top5_actual_returns))
                        metrics_output['top5_positive_ratio'] = float(np.mean(top5_actual_returns > 0))
                        
                        # 保存预测的前5名板块代码和板块名称
                        metrics_output['top5_sectors'] = valid_results.head(5)['sector_code'].tolist()
                        if 'sector_name' in valid_results.columns:
                            metrics_output['top5_sector_names'] = valid_results.head(5)['sector_name'].tolist()
                        
                        # 输出日志
                        self.logger.info(f"回归预测评估 ({date_obj.strftime('%Y%m%d')}): "
                                        f"RMSE={metrics_output['rmse']:.4f}, "
                                        f"MAE={metrics_output['mae']:.4f}, "
                                        f"R²={metrics_output['r2']:.4f}, "
                                        f"方向准确率={metrics_output['direction_accuracy']:.2f}, "
                                        f"Top5平均收益率={metrics_output['top5_avg_return']:.4f}")
                    else:
                        self.logger.warning("无法计算评估指标，因为没有有效的实际收益率与预测对齐。")
                except Exception as e_metrics:
                    self.logger.error(f"计算回归评估指标时出错: {e_metrics}")
                    self.logger.error(traceback.format_exc())
            else:
                self.logger.info(f"数据中缺少未来收益率列{future_return_col}，跳过评估指标计算。")
                
            # 计算预测前5板块的收益情况
            try:
                # 获取预测前5的板块
                top5_predicted = result_df.head(5)
                
                # 如果有实际收益数据，计算这些板块的实际平均收益
                if 'actual_return' in top5_predicted.columns:
                    top5_avg_return = top5_predicted['actual_return'].mean()
                    top5_positive_ratio = (top5_predicted['actual_return'] > 0).mean()
                    
                    metrics_output['top5_avg_return'] = float(top5_avg_return)
                    metrics_output['top5_positive_ratio'] = float(top5_positive_ratio)
                    
                    self.logger.info(f"Top5预测板块平均收益率: {top5_avg_return:.4f}")
                    self.logger.info(f"Top5预测板块正收益比例: {top5_positive_ratio:.2%}")
            except Exception as e:
                self.logger.warning(f"计算Top5预测板块收益指标时出错: {e}")


            prediction_results = {
                'predictions': result_df,
                'test_df': test_df.reset_index(drop=True), # 确保 test_df 有干净的索引
                'metrics': metrics_output,
                'date': date_obj
            }
            return prediction_results

        except Exception as e:
            self.logger.error(f"预测过程中发生严重错误 ({prediction_date}): {str(e)}")
            self.logger.error(traceback.format_exc())
            # 返回一个包含错误信息的字典，而不是空的DataFrame，以便上层调用知道出了问题
            return {
                'predictions': pd.DataFrame(),
                'test_df': features_df[features_df['date'] == pd.to_datetime(prediction_date)].copy() if pd.to_datetime(prediction_date) in features_df['date'].values else pd.DataFrame(),
                'metrics': {},
                'date': pd.to_datetime(prediction_date),
                'error': str(e)
            }

    def _create_sequence_features(self, features_df, sector_list=None, end_date=None, seq_length=10):
            """
            为LNN模型创建序列特征
            
            Args:
                features_df: 特征DataFrame
                sector_list: 板块代码列表或单个板块代码
                end_date: 序列结束日期
                seq_length: 序列长度
                
            Returns:
                序列特征字典，键为板块代码，值为包含'features'和'labels'键的字典
            """
            try:
                # 处理输入参数
                if sector_list is None:
                    sector_list = features_df['sector_code'].unique()
                elif isinstance(sector_list, str):
                    sector_list = [sector_list]
                
                if end_date is None:
                    if 'date' in features_df.columns:
                        end_date = features_df['date'].max()
                    else:
                        self.logger.error("特征DataFrame中缺少'date'列且未提供end_date")
                        return {}
                    
                # 确保end_date是datetime类型
                end_date = pd.to_datetime(end_date)
                
                # 确保日期列是datetime类型
                if 'date' in features_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                        features_df['date'] = pd.to_datetime(features_df['date'])
                else:
                    self.logger.error("特征DataFrame中缺少'date'列")
                    return {}
                    
                # 检查是否有sector_code列
                if 'sector_code' not in features_df.columns:
                    self.logger.error("特征DataFrame中缺少'sector_code'列")
                    return {}
                
                # 排除标签和非特征列
                exclude_cols = ['sector_code', 'date', 'label_date', 'market_type']
                # 注意：不要排除market_group特征和future_return_2d，这些是模型所需的重要特征
                exclude_cols += [col for col in features_df.columns if col.startswith('future_return') and col != 'future_return_2d']
                exclude_cols += [col for col in features_df.columns if pd.api.types.is_datetime64_dtype(features_df[col])]
                
                # 处理所有数值特征，排除不需要的列
                numeric_cols = features_df.select_dtypes(include=['number']).columns.tolist()
                feature_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                # 检查是否有足够的特征
                if len(feature_cols) == 0:
                    self.logger.error("没有找到数值型特征列，无法创建序列特征")
                    # 处理极端情况：如果真的没有数值列，尝试使用所有非排除列作为特征
                    backup_cols = [col for col in features_df.columns if col not in exclude_cols]
                    if len(backup_cols) > 0:
                        self.logger.warning(f"尝试使用所有非排除列作为特征，共 {len(backup_cols)} 列")
                        feature_cols = backup_cols
                    else:
                        # 如果真的没有可用特征，创建一个单列的特征矩阵（全为0）
                        self.logger.warning("创建单列的零特征矩阵作为后备方案")
                        features_df['zero_feature'] = 0.0
                        feature_cols = ['zero_feature']
                
                self.logger.info(f"为序列特征选择了 {len(feature_cols)} 个列")
                
                # 检查是否已包含未来收益率作为预测目标
                has_return_target = 'future_return_2d' in features_df.columns
                if not has_return_target:
                    self.logger.warning("序列特征中缺少future_return_2d收益率数据，模型可能无法准确预测")
                    # 为保证兼容性，创建一个全为0的future_return_2d列
                    features_df['future_return_2d'] = 0.0
                    self.logger.info("已创建默认future_return_2d列（全部为0，仅用于保证兼容性）")
                else:
                    self.logger.info(f"使用future_return_2d列作为预测目标，平均收益率: {features_df['future_return_2d'].mean():.4f}")
                
                # 获取每个板块可用的日期以便记录
                all_dates_by_sector = {}
                for sector in sector_list:
                    sector_data = features_df[features_df['sector_code'] == sector]
                    if not sector_data.empty:
                        dates = sorted(sector_data['date'].unique())
                        all_dates_by_sector[sector] = dates
                        self.logger.info(f"板块 {sector} 有 {len(dates)} 个可用交易日")
                
                # 创建每个板块的序列特征
                sequence_features = {}
                
                for sector_code in sector_list:
                    # 获取该板块的特征
                    sector_df = features_df[features_df['sector_code'] == sector_code].copy()
                    if sector_df.empty:
                        self.logger.warning(f"板块 {sector_code} 没有数据，跳过")
                        continue
                        
                    # 按日期排序
                    sector_df = sector_df.sort_values('date')
                    
                    # 选择截止日期之前(含)的数据
                    sector_df = sector_df[sector_df['date'] <= end_date]
                    if sector_df.empty:
                        self.logger.warning(f"板块 {sector_code} 在截止日期 {end_date} 之前没有数据，跳过")
                        continue
                    
                    # 记录实际使用的日期范围
                    date_range_str = f"{sector_df['date'].min()} 至 {sector_df['date'].max()}"
                    self.logger.info(f"板块 {sector_code} 使用的日期范围: {date_range_str}, 共 {len(sector_df)} 天")

                    # 提取特征矩阵
                    try:
                        # 检查可用的特征列
                        valid_cols = [col for col in feature_cols if col in sector_df.columns]
                        
                        if not valid_cols:
                            self.logger.warning(f"板块 {sector_code} 没有任何有效特征列，跳过")
                            continue
                        
                        # 确保所有特征都是数值类型
                        # 先检查特征类型，处理可能的分类特征
                        for col in valid_cols.copy():
                            if isinstance(sector_df[col].dtype, pd.CategoricalDtype):
                                self.logger.warning(f"列 {col} 是分类类型，尝试转换为数值类型")
                                try:
                                    # 尝试将分类特征转换为数值（通过标签编码）
                                    # 创建临时Series以避免直接使用Categorical
                                    temp_series = pd.Series(sector_df[col], index=sector_df.index, name=col)
                                    # 将类别转换为整数编码
                                    sector_df[col] = pd.factorize(temp_series)[0]
                                except Exception as e:
                                    self.logger.warning(f"无法将分类特征 {col} 转换为数值: {e}")
                                    valid_cols.remove(col)
                        
                        # 使用强制转换和空值填充来处理非数值数据
                        X_df = sector_df[valid_cols].copy()
                        for col in valid_cols:
                            try:
                                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
                            except Exception as e:
                                self.logger.warning(f"将列 {col} 转换为数值时出错: {e}")
                                # 如果转换失败，创建全0列
                                X_df[col] = 0.0
                        
                        # 二次检查确保没有非数值数据
                        valid_cols = []
                        for col in X_df.columns:
                            if not pd.api.types.is_numeric_dtype(X_df[col]):
                                self.logger.warning(f"列 {col} 不是数值类型，将跳过该列")
                            else:
                                valid_cols.append(col)
                        
                        # 只使用有效的数值列
                        if not valid_cols:
                            self.logger.warning(f"板块 {sector_code} 没有有效的数值特征列，跳过")
                            continue
                        
                        X_df = X_df[valid_cols]
                        
                        # 记录最终实际使用的特征列
                        self.logger.info(f"板块 {sector_code} 的有效特征列: {len(valid_cols)} 个")
                        
                        # 转换为numpy数组
                        X = X_df.values
                        
                        # 验证数组不包含非数值数据
                        if not np.isfinite(X).all():
                            self.logger.warning(f"序列特征矩阵包含无穷或NaN值，将替换为0")
                            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # 提取收益率数据作为回归目标
                        y = None
                        if 'future_return_2d' in sector_df.columns:
                            y = sector_df['future_return_2d'].values
                        else:
                            # 如果没有收益率数据，创建一个零数组作为占位符
                            y = np.zeros(len(X))
                        
                        # 如果序列特征中存在NaN或无穷值，进行额外处理
                        if not np.isfinite(X).all():
                            self.logger.warning(f"替换NaN值后，序列特征仍然包含无穷或NaN值，进行彻底清理")
                            # 查找包含NaN或无穷值的列
                            bad_cols = np.where(~np.isfinite(X).all(axis=0))[0]
                            if len(bad_cols) > 0:
                                self.logger.info(f"发现 {len(bad_cols)} 个包含无效值的列，将只使用有效列")
                                # 只保留有效列
                                good_cols = np.where(np.isfinite(X).all(axis=0))[0]
                                if len(good_cols) > 0:
                                    X = X[:, good_cols]
                                    # 更新有效特征列名
                                    valid_cols = [valid_cols[i] for i in good_cols]
                                else:
                                    # 如果没有有效列，创建全零数组
                                    self.logger.warning("没有有效的特征列，使用全零数组")
                                    X = np.zeros((len(sector_df), 1))
                                    valid_cols = ["zero_feature"]
                        
                        # 校验数据形状
                        if X.shape[0] < seq_length:
                            # 如果数据太少，可能不适合LNN
                            data_percentage = X.shape[0] / seq_length * 100
                            self.logger.warning(f"板块 {sector_code} 的行数 {X.shape[0]} 小于所需序列长度 {seq_length} (仅 {data_percentage:.1f}%)，将使用零填充")
                            
                            # 如果数据太少，可能考虑跳过
                            if X.shape[0] < seq_length * 0.3:  # 如果少于30%，可能不适合使用
                                self.logger.warning(f"板块 {sector_code} 的有效数据量太少 (仅 {data_percentage:.1f}%)，跳过该板块")
                                continue
                            
                            # 零填充数据
                            padding_rows = seq_length - X.shape[0]
                            padding = np.zeros((padding_rows, X.shape[1]))
                            
                            # 在前面补零，保留最近的实际数据
                            X = np.vstack([padding, X])
                            
                            # 同样填充标签
                            y_padding = np.zeros(padding_rows)
                            y = np.concatenate([y_padding, y])
                            
                            self.logger.info(f"板块 {sector_code} 的序列特征已填充 {padding_rows} 行零值，新形状: {X.shape}")
                        else:
                            # 确保使用最新的seq_length天数据
                            X = X[-seq_length:]
                            y = y[-seq_length:]

                    except Exception as e:
                        self.logger.error(f"处理板块 {sector_code} 的序列特征时出错: {e}")
                        self.logger.error(traceback.format_exc())
                        continue
                    
                    # 记录序列创建结果
                    sequence_dates = sector_df['date'].tolist()
                    if len(sequence_dates) < seq_length:
                        # 如果日期不足，记录填充情况
                        padding_dates = ['zero_padding'] * (seq_length - len(sequence_dates))
                        sequence_dates = padding_dates + sequence_dates

                    # 统一存储为包含'features'和'labels'键的字典结构
                    sequence_features[sector_code] = {
                        'features': X,  # 原始特征矩阵
                        'labels': y,    # 标签数组
                        'dates': sequence_dates,  # 日期数组，用于跟踪时间信息
                        'feature_names': valid_cols,  # 特征列名，用于调试
                        'actual_rows': len(sector_df),  # 实际数据行数
                        'padded_rows': seq_length - len(sector_df) if len(sector_df) < seq_length else 0  # 填充的行数
                    }
                    
                self.logger.info(f"共创建了 {len(sequence_features)} 个板块的序列特征")
                return sequence_features
                
            except Exception as e:
                self.logger.error(f"创建序列特征时出错: {e}")
                self.logger.error(traceback.format_exc())
                return {}

    # 已删除_create_rolling_top_performer_labels方法，不再需要分类标签
    
    def _create_return_prediction_labels(self, df, future_return_col='future_return_2d'):
        """
        为回归模型创建标签，直接使用future_return_2d作为预测目标
        无需额外的标签转换，直接预测实际收益率
        
        Args:
            df: 包含特征和未来收益的DataFrame
            future_return_col: 未来收益列名
            
        Returns:
            处理后的DataFrame，保留原始future_return_2d作为预测目标
        """
        try:
            if future_return_col not in df.columns:
                self.logger.error(f"缺少未来收益列 {future_return_col}")
                return df
            
            # 确保未来收益列无缺失值
            if df[future_return_col].isna().any():
                self.logger.warning(f"发现future_return_col有缺失值，使用0填充")
                df[future_return_col] = df[future_return_col].fillna(0)
            
            # 添加一些收益统计特征，作为辅助指标（非必需）
            df['past_mean_return'] = df.groupby('sector_code')[future_return_col].transform(
                lambda x: x.shift(1).rolling(20, min_periods=5).mean().fillna(0)
            )
            
            df['past_volatility'] = df.groupby('sector_code')[future_return_col].transform(
                lambda x: x.shift(1).rolling(20, min_periods=5).std().fillna(0.001)
            )
            
            # 记录一些统计信息
            positive_count = (df[future_return_col] > 0).sum()
            total_count = len(df)
            positive_ratio = positive_count / total_count if total_count > 0 else 0
            
            self.logger.info(f"收益率标签统计: 正收益比例 {positive_ratio:.2%}, "
                           f"平均收益率 {df[future_return_col].mean():.4f}, "
                           f"最大值 {df[future_return_col].max():.4f}, "
                           f"最小值 {df[future_return_col].min():.4f}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"创建收益预测标签时出错: {e}")
            traceback.print_exc()
            return df

    def _summarize_rolling_results(self, rolling_results):
            """
            汇总滚动训练结果
            
            Args:
                rolling_results: 滚动训练结果字典
            
            Returns:
                汇总统计字典
            """
            try:
                if not rolling_results or 'metrics' not in rolling_results or not rolling_results['metrics']:
                    return {"error": "无可用结果数据"}
                
                # 提取所有日期的指标
                all_metrics = rolling_results['metrics']
                dates = sorted(all_metrics.keys())
                
                accuracies = []
                avg_returns = []
                positive_ratios = []
                
                for date in dates:
                    metrics = all_metrics.get(date, {})
                    if metrics:
                        # 处理直接来自_predict_with_rolling_model函数的metrics格式
                        if isinstance(metrics, dict) and 'metrics' in metrics:
                            metrics = metrics['metrics']
                            
                        accuracies.append(metrics.get('accuracy', 0))
                        
                        # 获取avg_return，如果不存在则尝试从predicted_top_sectors和actual_top_sectors计算
                        if 'avg_return' not in metrics:
                            # 这种情况下不计算avg_return，使用默认值0
                            avg_returns.append(0)
                        else:
                            avg_returns.append(metrics.get('avg_return', 0))
                        
                        # 获取positive_ratio，如果不存在则尝试从correct_sectors计算
                        if 'positive_ratio' not in metrics:
                            # 使用correct_sectors和predicted_top_sectors计算positive_ratio
                            correct = len(metrics.get('correct_sectors', []))
                            total = len(metrics.get('predicted_top_sectors', []))
                            pos_ratio = correct / total if total > 0 else 0
                            positive_ratios.append(pos_ratio)
                        else:
                            positive_ratios.append(metrics.get('positive_ratio', 0))
                
                # 收集回归指标
                mae_values = []
                rmse_values = []
                r2_values = []
                direction_accuracy_values = []
                top5_avg_returns = []
                top5_positive_ratios = []
                
                for date in dates:
                    metrics = all_metrics.get(date, {})
                    if isinstance(metrics, dict) and 'metrics' in metrics:
                        metrics = metrics['metrics']
                    
                    mae_values.append(metrics.get('mae', 0))
                    rmse_values.append(metrics.get('rmse', 0))
                    r2_values.append(metrics.get('r2', 0))
                    direction_accuracy_values.append(metrics.get('direction_accuracy', 0))
                    top5_avg_returns.append(metrics.get('top5_avg_return', 0))
                    top5_positive_ratios.append(metrics.get('top5_positive_ratio', 0))
                
                # 计算汇总统计 - 针对回归模型
                summary = {
                    "dates": {  # 日期相关信息
                        "first": dates[0] if dates else None,  # 第一个训练日期
                        "last": dates[-1] if dates else None,  # 最后一个训练日期
                        "count": len(dates)  # 总共交易日数量
                    },
                    "regression_metrics": {
                        "mae": {
                            "mean": float(np.mean(mae_values)) if mae_values else 0,
                            "std": float(np.std(mae_values)) if mae_values else 0,
                            "min": float(np.min(mae_values)) if mae_values else 0,
                            "max": float(np.max(mae_values)) if mae_values else 0
                        },
                        "rmse": {
                            "mean": float(np.mean(rmse_values)) if rmse_values else 0,
                            "std": float(np.std(rmse_values)) if rmse_values else 0,
                            "min": float(np.min(rmse_values)) if rmse_values else 0,
                            "max": float(np.max(rmse_values)) if rmse_values else 0
                        },
                        "r2": {
                            "mean": float(np.mean(r2_values)) if r2_values else 0,
                            "std": float(np.std(r2_values)) if r2_values else 0,
                            "min": float(np.min(r2_values)) if r2_values else 0,
                            "max": float(np.max(r2_values)) if r2_values else 0
                        },
                        "direction_accuracy": {
                            "mean": float(np.mean(direction_accuracy_values)) if direction_accuracy_values else 0,
                            "std": float(np.std(direction_accuracy_values)) if direction_accuracy_values else 0,
                            "min": float(np.min(direction_accuracy_values)) if direction_accuracy_values else 0,
                            "max": float(np.max(direction_accuracy_values)) if direction_accuracy_values else 0
                        }
                    },
                    "return": {  # 收益率统计（预测的top5板块平均收益率）
                        "mean": float(np.mean(top5_avg_returns)) if top5_avg_returns else 0,
                        "std": float(np.std(top5_avg_returns)) if top5_avg_returns else 0,
                        "min": float(np.min(top5_avg_returns)) if top5_avg_returns else 0,
                        "max": float(np.max(top5_avg_returns)) if top5_avg_returns else 0,
                        "cumulative": float(np.prod([1 + r for r in top5_avg_returns if not np.isnan(r)]) - 1) if top5_avg_returns else 0
                    },
                    "positive_ratio": {  # Top5板块中有正收益的比例
                        "mean": float(np.mean(top5_positive_ratios)) if top5_positive_ratios else 0,
                        "std": float(np.std(top5_positive_ratios)) if top5_positive_ratios else 0,
                        "min": float(np.min(top5_positive_ratios)) if top5_positive_ratios else 0,
                        "max": float(np.max(top5_positive_ratios)) if top5_positive_ratios else 0
                    }
                }
                
                # 添加月度绩效
                monthly_perf = {}
                for date, metrics in all_metrics.items():
                    if date and len(date) >= 6:
                        month = date[:6]  # YYYYMM
                        if month not in monthly_perf:
                            monthly_perf[month] = []
                        
                        # 处理直接来自_predict_with_rolling_model函数的metrics格式
                        if isinstance(metrics, dict) and 'metrics' in metrics:
                            metrics = metrics['metrics']
                        
                        # 尝试获取或计算avg_return
                        avg_return = 0
                        if 'avg_return' in metrics:
                            avg_return = metrics['avg_return']
                        
                        monthly_perf[month].append(avg_return)
                
                # 计算每月的绩效指标
                month_summary = {}
                for month, returns in monthly_perf.items():
                    month_summary[month] = {
                        "mean_return": float(np.mean(returns)),  # 月平均日收益率
                        "cumulative_return": float(np.prod([1 + r for r in returns]) - 1),  # 月累积收益率（连续复利）
                        "trading_days": len(returns)  # 月交易天数
                    }
                
                summary["monthly_performance"] = month_summary  # 月度绩效汇总
                
                # 添加所有日期的原始metrics数据，而不仅仅是前5个                
                debug_metrics = {}                
                # 保存所有日期的指标，确保汇总信息完整                
                for date in dates:                    
                    debug_metrics[date] = all_metrics[date]                
                summary["debug_sample_metrics"] = debug_metrics  # 所有日期的指标数据
                
                return summary
            
            except Exception as e:
                self.logger.error(f"汇总滚动训练结果时出错: {e}")
                self.logger.error(traceback.format_exc())
                return {"error": str(e)}



def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="板块轮动模型训练与预测")
    parser.add_argument('--mode', type=str, choices=['train'], 
                      help='操作模式：train (训练模型)')
    parser.add_argument('--start_date', type=str, help='开始日期，格式为YYYYMMDD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD')
    parser.add_argument('--date', type=str, help='预测日期，格式为YYYYMMDD')
    parser.add_argument('--top_n', type=int, default=5, help='选择前N个板块')
    parser.add_argument('--max_workers', type=int, help='并行训练的进程数')
    parser.add_argument('--use_lnn', action='store_true', help='是否使用液体神经网络模型')
    parser.add_argument('--sector_list', type=str, help='板块列表文件路径，用于板块轮动模型训练')
    parser.add_argument('--model_dir', type=str, help='模型保存目录')
    parser.add_argument('--model_version', type=str, default='latest', help='用于预测或回测的模型版本，默认使用最新版本')
    
    args = parser.parse_args()
    
    # 创建SectorModelTrainer实例
    trainer = SectorModelTrainer(model_dir=args.model_dir, use_lnn=args.use_lnn)
    
    # 加载板块列表
    sector_list = []
    try:
        # 优先使用参数指定的板块列表文件
        if args.sector_list and os.path.exists(args.sector_list):
            with open(args.sector_list, 'r', encoding='utf-8') as f:
                sector_list = [line.strip() for line in f if line.strip()]
            logger.info(f"从文件 {args.sector_list} 加载了 {len(sector_list)} 个板块")
        else:
            # 尝试从stock_sectors_mapping模块获取
            try:
                # 将当前工作目录添加到sys.path
                sys.path.append(os.getcwd())
                from stock_sectors_mapping import get_tng_sectors
                sector_list = get_tng_sectors()
                logger.info(f"从get_tng_sectors加载了 {len(sector_list)} 个板块")
            except ImportError:
                # 如果导入失败，尝试从to目录导入
                try:
                    sys.path.append(str(get_project_root()))
                    from to.stock_sectors_mapping import get_tng_sectors
                    sector_list = get_tng_sectors()
                    logger.info(f"从to.stock_sectors_mapping加载了 {len(sector_list)} 个板块")
                except ImportError:
                    logger.error("无法导入get_tng_sectors函数，请提供板块列表文件")
    except Exception as e:
        logger.error(f"获取板块列表时出错: {e}")
        logger.error(traceback.format_exc())
    
    
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
                use_lnn=args.use_lnn)
        print(f"板块轮动模型训练结果: {'成功' if models else '失败'}")
            
    else:
        print("请指定有效的运行模式: --mode train|predict|backtest")


if __name__ == "__main__":
    main() 