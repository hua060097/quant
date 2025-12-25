#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型类定义文件
包含神经网络和集成学习模型
"""

import logging
import numpy as np
import pandas as pd
import traceback
import pickle
import os
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.dummy import DummyClassifier

# 引入机器学习模型
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# 引入PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import copy

# 尝试导入HMM相关库
HAS_HMM = False
try:
    from hmmlearn import hmm
    import sklearn
    from sklearn.preprocessing import StandardScaler
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

# 设置日志
logger = logging.getLogger("ModelClasses")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class LiquidTimeCell(nn.Module):
    """液体时间常数循环单元，适用于时序数据处理"""
    def __init__(self, input_size, hidden_size):
        super(LiquidTimeCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 线性变换权重
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        # 初始化时间常数τ为可学习参数（控制状态更新速度）
        self.tau = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x, h):
        """前向传播函数
        参数:
            x: 输入张量 [batch_size, input_size]
            h: 隐藏状态 [batch_size, hidden_size]
        """
        # 连接输入和隐藏状态
        combined = torch.cat((x, h), dim=1)
        # 计算候选隐藏状态
        g = torch.tanh(self.W_g(combined))
        # 计算输出门
        o = torch.sigmoid(self.W_o(combined))
        # 使用时间常数τ更新隐藏状态（关键的LTC特性）
        h_new = h * (1 - 1/self.tau.exp()) + g * (1/self.tau.exp())
        # 应用输出门得到最终输出
        y = o * torch.tanh(h_new)
        
        return y, h_new

class LiquidNeuralNetwork(nn.Module):
    """基于液体时间常数的神经网络，专为金融时序数据设计"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 创建多层LTC
        self.cells = nn.ModuleList([
            LiquidTimeCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h=None):
        """前向传播
        参数:
            x: 输入序列 [batch_size, input_size]
            h: 初始隐藏状态列表 [num_layers, batch_size, hidden_size]
        """
        batch_size = x.shape[0]
        
        # 如果没有提供隐藏状态，初始化为零
        if h is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        
        # 通过每一层LTC
        new_h = []
        for i, cell in enumerate(self.cells):
            x, h_i = cell(x, h[i])
            new_h.append(h_i)
        
        # 输出层
        output = self.output_layer(x)
        return output, new_h

class MarketRegimeHMM:
    """隐马尔可夫模型用于识别市场状态"""
    
    def __init__(self, n_components=4, random_state=42):
        """
        初始化HMM模型
        
        Args:
            n_components: 隐藏状态数量(例如熊市、震荡熊市、震荡牛市、牛市)
            random_state: 随机种子
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
        if HAS_HMM:
            self.model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=1000,
                random_state=random_state
            )
            self.scaler = StandardScaler()
        else:
            logger.warning("hmmlearn库不可用，HMM功能将受限")
    
    def fit(self, data):
        """
        训练HMM模型
        
        Args:
            data: 市场数据，包括指数收益率、成交量等特征
            
        Returns:
            bool: 是否成功训练
        """
        if not HAS_HMM:
            logger.error("无法训练HMM模型，缺少必要的库")
            return False
            
        try:
            # 标准化数据
            scaled_data = self.scaler.fit_transform(data)
            # 训练HMM
            self.model.fit(scaled_data)
            self.is_fitted = True
            logger.info(f"HMM模型训练完成，隐藏状态数: {self.n_components}")
            return True
        except Exception as e:
            logger.error(f"HMM模型训练失败: {e}")
            return False
    
    def predict_states(self, data):
        """
        预测隐藏状态
        
        Args:
            data: 市场数据
            
        Returns:
            states: 预测的隐藏状态序列
        """
        if not HAS_HMM or not self.is_fitted:
            logger.error("无法预测状态，HMM模型未训练或库不可用")
            return None
        
        try:
            # 标准化数据
            scaled_data = self.scaler.transform(data)
            # 预测隐藏状态
            states = self.model.predict(scaled_data)
            return states
        except Exception as e:
            logger.error(f"预测隐藏状态失败: {e}")
            return None
    
    def save_model(self, filepath):
        """保存模型到文件"""
        if not HAS_HMM or not self.is_fitted:
            logger.error("无法保存未训练的HMM模型或库不可用")
            return False
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_fitted': self.is_fitted,
                    'n_components': self.n_components,
                    'random_state': self.random_state
                }, f)
            logger.info(f"HMM模型已保存到: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存HMM模型失败: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        """从文件加载模型"""
        if not HAS_HMM:
            logger.error("无法加载HMM模型，缺少必要的库")
            return None
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            hmm_model = cls(n_components=data['n_components'], random_state=data['random_state'])
            hmm_model.model = data['model']
            hmm_model.scaler = data['scaler']
            hmm_model.is_fitted = data['is_fitted']
            
            logger.info(f"已从文件加载HMM模型: {filepath}")
            return hmm_model
        except Exception as e:
            logger.error(f"加载HMM模型失败: {e}")
            return None

class EnsembleModel:
    """集成模型，结合XGBoost、LightGBM、CatBoost和LNN"""
    def __init__(self, input_size, hidden_size=64, output_size=2):
        """
        初始化集成模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LNN隐藏层大小
            output_size: 输出类别数 (通常为2表示二分类)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 各子模型
        self.models = {
            'xgboost': None,
            'lightgbm': None,
            'catboost': None,
            'lnn': None,
            'hmm': None
        }
        
        # 模型权重
        self.weights = {
            'xgboost': 0.3,
            'lightgbm': 0.3,
            'catboost': 0.2,
            'lnn': 0.2,
            'hmm': 0.0  # 默认不使用HMM进行直接预测
        }
        
        # 市场状态相关
        self.market_states = None
        self.state_models = {}  # 每个市场状态的特定模型
        
        logger.info(f"初始化EnsembleModel: 输入维度={input_size}, 输出维度={output_size}")
        logger.info(f"模型权重: XGBoost={self.weights['xgboost']}, LightGBM={self.weights['lightgbm']}, CatBoost={self.weights['catboost']}, LNN={self.weights['lnn']}")
        
        # 设置device属性，默认为CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 模型是否已训练的标志
        self.trained = False
        
        # 是否加载了权重
        self.loaded_weights = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None, market_features=None, accelerator=None, use_hmm=False):
        """
        训练集成模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征，如果为None则从训练集分割
            y_val: 验证标签，如果为None则从训练集分割
            market_features: 市场特征数据，用于训练HMM模型
            accelerator: 加速器对象
            use_hmm: 是否使用HMM
            
        Returns:
            self
        """
        logger.info(f"训练集成模型: 输入维度={self.input_size}, 输出维度={self.output_size}")
        
        # 确保输入是NumPy数组
        try:
            # 记录原始数据类型
            logger.info(f"原始X_train类型: {type(X_train)}，y_train类型: {type(y_train)}")
            
            # 首先检查并转换X_train
            if isinstance(X_train, pd.DataFrame):
                logger.info(f"将X_train从DataFrame转换为NumPy数组，形状: {X_train.shape}")
                X_train_np = X_train.values
            elif isinstance(X_train, list):
                logger.info("将X_train从列表转换为NumPy数组")
                X_train_np = np.array(X_train)
                # 确保转换后再设置dtype
                if X_train_np.size > 0:  # 确保数组不为空
                    X_train_np = X_train_np.astype(np.float64)
            elif isinstance(X_train, np.ndarray):
                logger.info("X_train已经是NumPy数组，确保为浮点型")
                X_train_np = X_train.astype(np.float64)
            else:
                # 尝试直接转换为NumPy数组
                logger.info(f"尝试将X_train(类型:{type(X_train)})转换为NumPy浮点数组")
                X_train_np = np.array(X_train)
                if X_train_np.size > 0:  # 确保数组不为空
                    X_train_np = X_train_np.astype(np.float64)
            
            # 处理y_train
            if isinstance(y_train, pd.Series):
                logger.info("将y_train从Series转换为NumPy数组")
                y_train_np = y_train.values
                if y_train_np.size > 0:  # 确保数组不为空
                    y_train_np = y_train_np.astype(np.float64)
            elif isinstance(y_train, list):
                logger.info("将y_train从列表转换为NumPy数组")
                y_train_np = np.array(y_train)
                if y_train_np.size > 0:  # 确保数组不为空
                    y_train_np = y_train_np.astype(np.float64)
            elif isinstance(y_train, np.ndarray):
                logger.info("y_train已经是NumPy数组，确保为浮点型")
                y_train_np = y_train.astype(np.float64)
            else:
                logger.info(f"尝试将y_train(类型:{type(y_train)})转换为NumPy浮点数组")
                y_train_np = np.array(y_train)
                if y_train_np.size > 0:  # 确保数组不为空
                    y_train_np = y_train_np.astype(np.float64)
                
            # 记录转换后的数据类型
            logger.info(f"转换后X_train_np类型: {type(X_train_np)}，形状: {X_train_np.shape}，数据类型: {X_train_np.dtype}")
            logger.info(f"转换后y_train_np类型: {type(y_train_np)}，形状: {y_train_np.shape if hasattr(y_train_np, 'shape') else 'N/A'}，数据类型: {y_train_np.dtype}")
            
        except Exception as e:
            logger.error(f"转换输入数据为NumPy数组失败: {str(e)}")
            raise ValueError(f"数据转换失败: {str(e)}")
        
        # 检查样本数量
        sample_count = len(X_train_np) if hasattr(X_train_np, '__len__') else 0
        if sample_count < 2:
            logger.warning(f"训练样本数量不足: {sample_count}")
            if sample_count == 0:
                raise ValueError("没有训练样本")
                
        # 设置验证数据
        if X_val is None or y_val is None:
            X_train_short, X_val, y_train_short, y_val = train_test_split(
                X_train_np, y_train_np, test_size=0.2, random_state=42
            )
        else:
            X_train_short, y_train_short = X_train_np, y_train_np
        
        # 确保X_train_short和y_train_short是NumPy数组并且是浮点型
        if not isinstance(X_train_short, np.ndarray):
            logger.info("将X_train_short转换为NumPy数组")
            X_train_short = np.array(X_train_short, dtype=np.float64)
        elif X_train_short.dtype != np.float64:
            logger.info(f"将X_train_short从{X_train_short.dtype}转换为float64")
            X_train_short = X_train_short.astype(np.float64)
            
        if not isinstance(y_train_short, np.ndarray):
            logger.info("将y_train_short转换为NumPy数组")
            y_train_short = np.array(y_train_short, dtype=np.float64)
        elif y_train_short.dtype != np.float64:
            logger.info(f"将y_train_short从{y_train_short.dtype}转换为float64")
            y_train_short = y_train_short.astype(np.float64)
        
        if use_hmm and HAS_HMM and market_features is not None:
            self._train_hmm(market_features)
        
        # 训练XGBoost
        logger.info("训练XGBoost模型...")
        try:
            # 记录数据信息
            logger.info(f"XGBoost训练数据: X_train_short形状={X_train_short.shape}, 类型={X_train_short.dtype}")
            logger.info(f"XGBoost训练标签: y_train_short形状={y_train_short.shape if hasattr(y_train_short, 'shape') else len(y_train_short)}, 类型={y_train_short.dtype if hasattr(y_train_short, 'dtype') else type(y_train_short)}")
            
            # 确保y_train是一维数组
            if isinstance(y_train_short, np.ndarray) and len(y_train_short.shape) > 1 and y_train_short.shape[1] > 1:
                logger.warning(f"y_train维度为{y_train_short.shape}，正在转换为一维数组...")
                y_train_short = np.squeeze(y_train_short)
            
            # 检查是否有NaN或Inf - 确保已经转换为浮点型数组
            has_nan_x = np.isnan(X_train_short).any()
            has_inf_x = np.isinf(X_train_short).any()
            if has_nan_x or has_inf_x:
                logger.warning("XGBoost训练数据包含NaN或Inf值，正在替换为0...")
                X_train_short = np.nan_to_num(X_train_short, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 检查y_train是否包含NaN - 确保已经转换为浮点型数组
            has_nan_y = np.isnan(y_train_short).any()
            if has_nan_y:
                logger.error("y_train包含NaN值")
                raise ValueError("标签包含NaN值")
            
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # 如果有加速器，尝试应用XGBoost加速
            if accelerator is not None:
                logger.info("使用加速器优化XGBoost训练...")
                try:
                    # 应用潜在的并行训练优化
                    if accelerator.has_cuda and hasattr(self.models['xgboost'], 'set_params'):
                        logger.info("XGBoost: 启用GPU加速")
                        self.models['xgboost'].set_params(tree_method='gpu_hist')
                    elif accelerator.enable_threading and hasattr(self.models['xgboost'], 'set_params'):
                        threads = accelerator.get_optimal_threads()
                        logger.info(f"XGBoost: 使用{threads}个线程")
                        self.models['xgboost'].set_params(n_jobs=threads)
                except Exception as e:
                    logger.warning(f"应用XGBoost加速失败: {e}")
            
            try:
                logger.info("开始训练XGBoost模型...")
                self.models['xgboost'].fit(X_train_short, y_train_short)
                logger.info("XGBoost模型训练完成")
            except Exception as e:
                logger.error(f"XGBoost训练失败: {e}")
                # 使用DummyClassifier作为备选
                logger.warning("使用DummyClassifier替代失败的XGBoost模型")
                self.models['xgboost'] = DummyClassifier().fit(X_train_short, y_train_short)
                
        except Exception as e:
            logger.error(f"XGBoost初始化或训练失败: {e}")
            logger.error(traceback.format_exc())
            self.models['xgboost'] = None  # 设置为None表示训练失败
        
        # 训练LightGBM
        logger.info("训练LightGBM模型...")
        try:
            # 记录数据信息
            logger.info(f"LightGBM训练数据: X_train_short形状={X_train_short.shape}, 类型={X_train_short.dtype}")
            logger.info(f"LightGBM训练标签: y_train_short形状={y_train_short.shape if hasattr(y_train_short, 'shape') else len(y_train_short)}, 类型={y_train_short.dtype if hasattr(y_train_short, 'dtype') else type(y_train_short)}")
            
            # 确保y_train是一维数组
            if isinstance(y_train_short, np.ndarray) and len(y_train_short.shape) > 1 and y_train_short.shape[1] > 1:
                logger.warning(f"y_train维度为{y_train_short.shape}，正在转换为一维数组...")
                y_train_short = np.squeeze(y_train_short)
                
            # 检查样本数是否足够
            if len(X_train_short) < 2:
                logger.warning("LightGBM训练样本数不足，可能导致模型表现不佳")
                
            # 检查是否有NaN或Inf - 确保已经转换为浮点型数组
            has_nan_x = np.isnan(X_train_short).any()
            has_inf_x = np.isinf(X_train_short).any()
            if has_nan_x or has_inf_x:
                logger.warning("LightGBM训练数据包含NaN或Inf值，正在替换为0...")
                X_train_short = np.nan_to_num(X_train_short, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 检查y_train是否包含NaN - 确保已经转换为浮点型数组
            has_nan_y = np.isnan(y_train_short).any()
            if has_nan_y:
                logger.error("y_train包含NaN值")
                raise ValueError("标签包含NaN值")
            
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                min_data_in_bin=1,   # 降低此参数以处理小数据集
                min_data_in_leaf=1,  # 降低此参数以处理小数据集
                min_child_samples=1, # 降低此参数以处理小数据集
                min_split_gain=0,    # 降低此参数使模型可以更容易分裂 
                verbose=-1           # 减少警告输出
            )
            
            # 如果有加速器，尝试应用LightGBM加速
            if accelerator is not None:
                logger.info("使用加速器优化LightGBM训练...")
                try:
                    if accelerator.has_cuda and hasattr(self.models['lightgbm'], 'set_params'):
                        logger.info("LightGBM: 启用GPU加速")
                        self.models['lightgbm'].set_params(device='gpu')
                    elif accelerator.enable_threading and hasattr(self.models['lightgbm'], 'set_params'):
                        threads = accelerator.get_optimal_threads()
                        logger.info(f"LightGBM: 使用{threads}个线程")
                        self.models['lightgbm'].set_params(n_jobs=threads)
                except Exception as e:
                    logger.warning(f"应用LightGBM加速失败: {e}")
            
            try:
                logger.info("开始训练LightGBM模型...")
                self.models['lightgbm'].fit(X_train_short, y_train_short)
                logger.info("LightGBM模型训练完成")
            except Exception as e:
                logger.error(f"LightGBM训练失败: {e}")
                # 使用DummyClassifier作为备选
                logger.warning("使用DummyClassifier替代失败的LightGBM模型")
                self.models['lightgbm'] = DummyClassifier().fit(X_train_short, y_train_short)
                
        except Exception as e:
            logger.error(f"LightGBM初始化或训练失败: {e}")
            logger.error(traceback.format_exc())
            self.models['lightgbm'] = None  # 设置为None表示训练失败
        
        # 训练CatBoost
        logger.info("训练CatBoost模型...")
        try:
            # 记录数据信息
            logger.info(f"CatBoost训练数据: X_train_short形状={X_train_short.shape}, 类型={X_train_short.dtype}")
            logger.info(f"CatBoost训练标签: y_train_short形状={y_train_short.shape if hasattr(y_train_short, 'shape') else len(y_train_short)}, 类型={y_train_short.dtype if hasattr(y_train_short, 'dtype') else type(y_train_short)}")
            
            # 确保y_train是一维数组
            if isinstance(y_train_short, np.ndarray) and len(y_train_short.shape) > 1 and y_train_short.shape[1] > 1:
                logger.warning(f"y_train维度为{y_train_short.shape}，正在转换为一维数组...")
                y_train_short = np.squeeze(y_train_short)
                
            # 检查样本数是否足够
            if len(X_train_short) < 2:
                logger.warning("CatBoost训练样本数不足，可能导致模型表现不佳")
                
            # 检查是否有NaN或Inf - 确保已经转换为浮点型数组
            has_nan_x = np.isnan(X_train_short).any()
            has_inf_x = np.isinf(X_train_short).any()
            if has_nan_x or has_inf_x:
                logger.warning("CatBoost训练数据包含NaN或Inf值，正在替换为0...")
                X_train_short = np.nan_to_num(X_train_short, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 检查y_train是否包含NaN - 确保已经转换为浮点型数组
            has_nan_y = np.isnan(y_train_short).any()
            if has_nan_y:
                logger.error("y_train包含NaN值")
                raise ValueError("标签包含NaN值")
            
            # 创建并配置CatBoost模型
            catboost_params = {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'random_seed': 42,
                'verbose': False
            }
            
            # 如果有加速器，尝试应用CatBoost加速
            if accelerator is not None:
                logger.info("使用加速器优化CatBoost训练...")
                try:
                    if accelerator.has_cuda:
                        logger.info("CatBoost: 启用GPU加速")
                        catboost_params['task_type'] = 'GPU'
                    elif accelerator.enable_threading:
                        threads = accelerator.get_optimal_threads()
                        logger.info(f"CatBoost: 使用{threads}个线程")
                        catboost_params['thread_count'] = threads
                except Exception as e:
                    logger.warning(f"应用CatBoost加速失败: {e}")
                    
            self.models['catboost'] = cb.CatBoostClassifier(**catboost_params)
            
            try:
                logger.info("开始训练CatBoost模型...")
                self.models['catboost'].fit(X_train_short, y_train_short, silent=True)
                logger.info("CatBoost模型训练完成")
            except Exception as e:
                logger.error(f"CatBoost训练失败: {e}")
                # 使用DummyClassifier作为备选
                logger.warning("使用DummyClassifier替代失败的CatBoost模型")
                self.models['catboost'] = DummyClassifier().fit(X_train_short, y_train_short)
                
        except Exception as e:
            logger.error(f"CatBoost初始化或训练失败: {e}")
            logger.error(traceback.format_exc())
            self.models['catboost'] = None  # 设置为None表示训练失败
        
        # 训练液体神经网络(LNN)
        if torch is not None:
            logger.info("训练Liquid Neural Network模型...")
            try:
                # 确保特征和标签是正确的格式
                X_tensor = torch.FloatTensor(X_train_short)
                y_tensor = torch.FloatTensor(y_train_short)
                
                # 创建数据集和数据加载器
                train_dataset = TensorDataset(X_tensor, y_tensor)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                # 验证集
                if X_val is not None and y_val is not None:
                    X_val_tensor = torch.FloatTensor(X_val)
                    y_val_tensor = torch.FloatTensor(y_val)
                    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                    val_loader = DataLoader(val_dataset, batch_size=32)
                else:
                    val_dataset = None
                    val_loader = None
                
                # 初始化模型
                if self.models['lnn'] is None:
                    self.models['lnn'] = LiquidNeuralNetwork(
                        input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        output_size=self.output_size
                    )
                
                # 如果有CUDA加速，将模型移至GPU
                device = torch.device('cuda:0' if torch.cuda.is_available() and accelerator is not None and accelerator.has_cuda else 'cpu')
                self.models['lnn'].to(device)
                
                # 定义损失函数和优化器
                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.models['lnn'].parameters(), lr=0.001)
                
                # 如果有加速器，检查是否启用混合精度训练
                scaler = None
                if accelerator is not None and hasattr(accelerator, 'enable_mixed_precision') and accelerator.enable_mixed_precision():
                    if hasattr(torch.cuda, 'amp'):
                        logger.info("启用自动混合精度训练")
                        scaler = torch.cuda.amp.GradScaler()
                
                # 训练模型
                num_epochs = 50
                self.models['lnn'].train()
                
                for epoch in range(num_epochs):
                    running_loss = 0.0
                    
                    # 使用tqdm显示进度条
                    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                        for inputs, targets in pbar:
                            # 将数据移动到设备
                            inputs, targets = inputs.to(device), targets.to(device)
                            
                            # 清零梯度
                            optimizer.zero_grad()
                            
                            # 前向传播
                            if scaler is not None:
                                # 使用自动混合精度
                                with torch.cuda.amp.autocast():
                                    outputs, _ = self.models['lnn'](inputs)
                                    loss = criterion(outputs, targets)
                                
                                # 反向传播和优化
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                # 标准训练
                                outputs, _ = self.models['lnn'](inputs)
                                loss = criterion(outputs, targets)
                                loss.backward()
                                optimizer.step()
                            
                            # 更新训练损失
                            running_loss += loss.item()
                        
                    # 每个epoch结束计算验证集上的损失
                    val_loss = 0.0
                    self.models['lnn'].eval()
                    
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs, _ = self.models['lnn'](inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    # 计算平均绝对误差
                    mae = 0.0
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs, _ = self.models['lnn'](inputs)
                            mae += torch.abs(outputs - targets).mean().item()
                    mae /= len(val_loader)
                    
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}, Val MAE: {mae:.6f}")
                    
                    # 恢复训练模式
                    self.models['lnn'].train()
                
                # 训练完成后切换到评估模式
                self.models['lnn'].eval()
                logger.info("完成LNN模型训练")
                
            except Exception as e:
                logger.error(f"LNN训练失败: {e}")
                logger.error(traceback.format_exc())
                self.models['lnn'] = None  # 设置为None表示训练失败
        
        return self
    
    def _train_hmm(self, market_features):
        """训练HMM模型识别市场状态"""
        if not HAS_HMM:
            logger.warning("HMM库不可用，无法训练市场状态模型")
            return
        
        try:
            # 创建并训练HMM
            hmm_model = MarketRegimeHMM(n_components=3)
            if hmm_model.fit(market_features):
                # 预测市场状态
                self.market_states = hmm_model.predict_states(market_features)
                self.models['hmm'] = hmm_model
                logger.info(f"HMM模型训练完成，识别了{len(np.unique(self.market_states))}种市场状态")
            else:
                logger.warning("HMM模型训练失败")
        except Exception as e:
            logger.error(f"训练HMM出错: {e}")
    
    def _train_lnn(self, X_train, y_train, X_val, y_val, accelerator=None):
        """训练液体神经网络"""
        try:
            # 转换为PyTorch张量
            # 先将DataFrame转换为NumPy数组
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            
            # 对于回归任务，将目标转换为FloatTensor而不是LongTensor
            X_train_tensor = torch.FloatTensor(X_train_np)
            y_train_tensor = torch.FloatTensor(y_train_np).view(-1, 1)  # 确保形状为 [batch_size, 1]
            X_val_tensor = torch.FloatTensor(X_val_np)
            y_val_tensor = torch.FloatTensor(y_val_np).view(-1, 1)  # 确保形状为 [batch_size, 1]
            
            # 检查是否可以使用CUDA
            device = 'cpu'
            if accelerator is not None and hasattr(accelerator, 'has_cuda') and accelerator.has_cuda:
                device = 'cuda'
                logger.info("LNN训练将使用CUDA加速")
                X_train_tensor = X_train_tensor.to(device)
                y_train_tensor = y_train_tensor.to(device)
                X_val_tensor = X_val_tensor.to(device)
                y_val_tensor = y_val_tensor.to(device)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            # 根据数据大小和是否有加速器决定批量大小
            batch_size = 32
            if accelerator is not None and hasattr(accelerator, 'recommended_batch_size'):
                batch_size = accelerator.recommended_batch_size(len(X_train))
                logger.info(f"使用加速器推荐的批量大小: {batch_size}")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # 如果有多核CPU且加速器可用，使用多进程数据加载
            if accelerator is not None:
                num_workers = 0
                if hasattr(accelerator, 'recommended_workers'):
                    num_workers = accelerator.recommended_workers()
                    logger.info(f"使用{num_workers}个工作线程加载训练数据")
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            # 创建模型
            # 注意：对于回归任务，output_size=1，而不是分类任务的2
            self.models['lnn'] = LiquidNeuralNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=1  # 回归任务只输出一个值
            )
            
            # 将模型移动到适当的设备
            self.models['lnn'].to(device)
            
            # 如果有加速器，尝试使用torch.compile优化模型
            if accelerator is not None and hasattr(torch, 'compile'):
                try:
                    logger.info("尝试使用torch.compile优化LNN模型")
                    self.models['lnn'] = torch.compile(self.models['lnn'])
                    logger.info("成功编译LNN模型")
                except Exception as e:
                    logger.warning(f"编译LNN模型失败: {e}")
            
            # 定义损失函数和优化器 - 对于回归任务使用MSELoss
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.models['lnn'].parameters(), lr=0.001)
            
            # 如果有加速器，检查是否启用混合精度训练
            scaler = None
            if accelerator is not None and hasattr(accelerator, 'enable_mixed_precision') and accelerator.enable_mixed_precision():
                if hasattr(torch.cuda, 'amp'):
                    logger.info("启用自动混合精度训练")
                    scaler = torch.cuda.amp.GradScaler()
            
            # 训练模型
            num_epochs = 50
            self.models['lnn'].train()
            
            for epoch in range(num_epochs):
                running_loss = 0.0
                
                # 使用tqdm显示进度条
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                    for inputs, targets in pbar:
                        # 将数据移动到设备
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # 清零梯度
                        optimizer.zero_grad()
                        
                        # 前向传播
                        if scaler is not None:
                            # 使用自动混合精度
                            with torch.cuda.amp.autocast():
                                outputs, _ = self.models['lnn'](inputs)
                                loss = criterion(outputs, targets)
                            
                            # 反向传播和优化
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # 标准训练
                            outputs, _ = self.models['lnn'](inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                        
                        # 更新训练损失
                        running_loss += loss.item()
                    
                # 每个epoch结束计算验证集上的损失
                val_loss = 0.0
                self.models['lnn'].eval()
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs, _ = self.models['lnn'](inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 计算平均绝对误差
                mae = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs, _ = self.models['lnn'](inputs)
                        mae += torch.abs(outputs - targets).mean().item()
                mae /= len(val_loader)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}, Val MAE: {mae:.6f}")
                
                # 恢复训练模式
                self.models['lnn'].train()
            
            # 训练完成后切换到评估模式
            self.models['lnn'].eval()
            logger.info("完成LNN模型训练")
            
        except Exception as e:
            logger.error(f"LNN训练失败: {e}")
            logger.error(traceback.format_exc())
            self.models['lnn'] = None  # 设置为None表示训练失败
    
    def _train_state_specific_models(self, X, y, states):
        """为每个市场状态训练专门的模型"""
        if states is None or len(states) != len(X):
            logger.error("市场状态数据与训练数据不匹配")
            return
        
        # 获取唯一的状态
        unique_states = np.unique(states)
        
        for state in unique_states:
            # 选择该状态下的样本
            state_mask = states == state
            if np.sum(state_mask) < 10:  # 如果样本太少，跳过
                logger.warning(f"状态{state}的样本数量过少({np.sum(state_mask)})，跳过训练")
                continue
                
            X_state = X[state_mask]
            y_state = y[state_mask]
            
            # 为该状态创建专门的模型
            self.state_models[state] = {}
            
            # XGBoost
            self.state_models[state]['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.state_models[state]['xgboost'].fit(X_state, y_state)
            
            # LightGBM
            self.state_models[state]['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.state_models[state]['lightgbm'].fit(X_state, y_state)
            
            logger.info(f"为状态{state}训练了专门的模型，样本数: {np.sum(state_mask)}")
    
    def predict(self, X, market_features=None):
        """
        预测样本的回归值
        
        Args:
            X: 特征矩阵，可以是DataFrame或numpy数组
            market_features: 用于预测市场状态的特征(如果有)
            
        Returns:
            预测的回归值（未来收益率）
        """
        try:
            # 转换输入数据为正确格式
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # 确保X是二维数组
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # 如果已训练了HMM模型并有市场特征，预测当前市场状态
            current_state = None
            if self.models['hmm'] is not None and market_features is not None:
                try:
                    current_state = self.models['hmm'].predict_state(market_features)
                    logger.debug(f"当前市场状态预测: {current_state}")
                except Exception as e:
                    logger.warning(f"预测市场状态失败: {e}")
            
            # 如果有状态特定模型且识别了当前状态，使用相应的模型
            if current_state is not None and current_state in self.state_models:
                logger.debug(f"使用状态{current_state}的专用模型")
                state_model = self.state_models[current_state]
                
                # 使用状态特定模型预测
                predictions = {}
                
                # 尝试所有模型预测
                self._try_all_models_predict(X, state_model, predictions)
                
                # 组合预测结果
                if not predictions:
                    logger.warning("状态特定模型预测失败，使用基本模型")
                else:
                    weights = self.weights.get(current_state, {'xgboost': 0.25, 'lightgbm': 0.25, 'catboost': 0.25, 'lnn': 0.25})
                    result = np.zeros(X.shape[0] if hasattr(X, 'shape') else len(X))
                    
                    for model_name, pred in predictions.items():
                        if pred is not None and len(pred) > 0:
                            result += weights.get(model_name, 0) * pred
                    
                    # 返回最终预测结果
                    return result
            
            # 使用基本模型预测
            logger.debug("使用基本模型预测")
            predictions = {}
            
            # 尝试所有基础模型预测
            self._try_all_models_predict(X, self.models, predictions)
            
            # 组合预测结果
            if not predictions:
                logger.warning("所有模型预测失败，返回默认值0.0")
                return np.array([0.0] * (X.shape[0] if hasattr(X, 'shape') else len(X)))
            
            # 计算加权平均
            result = np.zeros(X.shape[0] if hasattr(X, 'shape') else len(X))
            total_weight = 0.0
            for model_name, pred in predictions.items():
                if pred is not None and len(pred) > 0:
                    weight = self.weights.get(model_name, 0.25)
                    result += weight * pred
                    total_weight += weight
            
            # 如果总权重为0，返回默认值
            if total_weight == 0:
                logger.warning("总权重为0，返回默认值0.0")
                return np.array([0.0] * (X.shape[0] if hasattr(X, 'shape') else len(X)))
            
            # 标准化结果
            result /= total_weight
            
            # 返回最终预测结果（确保是numpy数组，不是字典）
            return result
        
        except Exception as e:
            logger.error(f"预测过程发生错误: {e}")
            logger.exception("预测错误详情:")
            # 返回默认预测值
            return np.array([0.0] * (X.shape[0] if hasattr(X, 'shape') else len(X)))
    
    def _try_all_models_predict(self, X, models, predictions):
        """尝试使用所有模型进行预测
        
        Args:
            X: 特征数据
            models: 模型字典
            predictions: 存储预测结果的字典
        """
        # XGBoost预测（回归模式）
        if 'xgboost' in models and models['xgboost'] is not None:
            try:
                # 对于回归任务，直接使用predict而非predict_proba
                xgb_pred = models['xgboost'].predict(X)
                predictions['xgboost'] = xgb_pred
            except Exception as e:
                logger.warning(f"XGBoost模型预测失败: {e}")
                
        # LightGBM预测（回归模式）
        if 'lightgbm' in models and models['lightgbm'] is not None:
            try:
                # 对于回归任务，直接使用predict而非predict_proba
                lgb_pred = models['lightgbm'].predict(X)
                predictions['lightgbm'] = lgb_pred
            except Exception as e:
                logger.warning(f"LightGBM模型预测失败: {e}")
                
        # CatBoost预测（回归模式）
        if 'catboost' in models and models['catboost'] is not None:
            try:
                # 对于回归任务，直接使用predict而非predict_proba
                cat_pred = models['catboost'].predict(X)
                predictions['catboost'] = cat_pred
            except Exception as e:
                logger.warning(f"CatBoost模型预测失败: {e}")
                
        # LNN预测（回归模式）
        if 'lnn' in models and models['lnn'] is not None:
            try:
                device = getattr(self, 'device', torch.device('cpu'))
                X_tensor = torch.FloatTensor(X).to(device)
                with torch.no_grad():
                    models['lnn'].eval()
                    outputs, _ = models['lnn'](X_tensor)
                    # 回归任务直接使用输出值，不需要softmax或sigmoid
                    lnn_pred = outputs.cpu().numpy().flatten()
                predictions['lnn'] = lnn_pred
            except Exception as e:
                logger.warning(f"LNN模型预测失败: {e}")
    
    def predict_proba(self, X, market_features=None):
        """
        为保持兼容性提供的方法，但对于回归任务不适用
        
        Args:
            X: 特征矩阵
            market_features: 用于预测市场状态的特征(如果有)
            
        Returns:
            预测值，包装成类似分类概率的格式 [样本数, 2]
            其中第二列为预测值，第一列为1-预测值（仅为保持API兼容）
        """
        try:
            # 获取预测结果（一维数组）
            pred_scores = self.predict(X, market_features)
            
            # 为保持API兼容性，将回归值转换成二维概率数组的格式
            # 但此转换对回归任务没有实际意义
            logger.warning("在回归模型上调用predict_proba，结果仅为保持API兼容")
            
            if isinstance(pred_scores, np.ndarray):
                n_samples = len(pred_scores)
                proba = np.zeros((n_samples, 2))
                
                # 归一化预测值到[0,1]区间，仅用于兼容性
                # 这只是一个近似转换，不应该用于实际决策
                normalized_scores = np.clip(pred_scores, -1, 1)
                normalized_scores = (normalized_scores + 1) / 2
                
                proba[:, 1] = normalized_scores  # 第二列为归一化后的预测值
                proba[:, 0] = 1 - normalized_scores  # 第一列为补值
                return proba
            else:
                raise ValueError(f"预测返回值类型错误: {type(pred_scores)}")
                
        except Exception as e:
            logger.error(f"predict_proba方法出错: {e}")
            logger.error(traceback.format_exc())
            # 如果出错，返回默认值（所有样本都预测为0）
            if isinstance(X, pd.DataFrame):
                n_samples = X.shape[0]
            else:
                n_samples = len(X)
            # 创建默认值数组
            default_proba = np.zeros((n_samples, 2))
            default_proba[:, 0] = 1.0  # 所有样本默认预测为0
            return default_proba
    
    def save(self, model_dir):
        """
        保存模型到指定目录
        
        Args:
            model_dir: 保存模型的目录
        """
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # 保存树模型
            for model_name in ['xgboost', 'lightgbm', 'catboost']:
                if self.models[model_name] is not None:
                    model_path = os.path.join(model_dir, f"{model_name}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.models[model_name], f)
            
            # 保存LNN模型
            if self.models['lnn'] is not None:
                lnn_path = os.path.join(model_dir, "lnn.pth")
                torch.save(self.models['lnn'].state_dict(), lnn_path)
            
            # 保存HMM模型
            if self.models['hmm'] is not None:
                hmm_path = os.path.join(model_dir, "hmm.pkl")
                self.models['hmm'].save_model(hmm_path)
            
            # 保存模型配置
            config_path = os.path.join(model_dir, "config.pkl")
            config = {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'weights': self.weights
            }
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
                
            # 保存状态特定模型
            if self.state_models:
                states_dir = os.path.join(model_dir, "state_models")
                os.makedirs(states_dir, exist_ok=True)
                
                for state, models in self.state_models.items():
                    state_dir = os.path.join(states_dir, f"state_{state}")
                    os.makedirs(state_dir, exist_ok=True)
                    
                    for model_name, model in models.items():
                        model_path = os.path.join(state_dir, f"{model_name}.pkl")
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
            
            logger.info(f"模型已保存到: {model_dir}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    @classmethod
    def load(cls, model_dir):
        """
        从指定目录加载模型
        
        Args:
            model_dir: 保存模型的目录
            
        Returns:
            EnsembleModel: 加载的模型
        """
        try:
            # 加载配置
            config_path = os.path.join(model_dir, "config.pkl")
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            # 创建模型实例
            model = cls(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                output_size=config['output_size']
            )
            
            # 设置权重
            model.weights = config['weights']
            
            # 加载树模型
            for model_name in ['xgboost', 'lightgbm', 'catboost']:
                model_path = os.path.join(model_dir, f"{model_name}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model.models[model_name] = pickle.load(f)
            
            # 加载LNN模型
            lnn_path = os.path.join(model_dir, "lnn.pth")
            if os.path.exists(lnn_path):
                model.models['lnn'] = LiquidNeuralNetwork(
                    input_size=config['input_size'],
                    hidden_size=config['hidden_size'],
                    output_size=config['output_size']
                )
                model.models['lnn'].load_state_dict(torch.load(lnn_path, weights_only=False))
                model.models['lnn'].eval()
            
            # 加载HMM模型
            hmm_path = os.path.join(model_dir, "hmm.pkl")
            if os.path.exists(hmm_path) and HAS_HMM:
                model.models['hmm'] = MarketRegimeHMM.load_model(hmm_path)
                
            # 加载状态特定模型
            states_dir = os.path.join(model_dir, "state_models")
            if os.path.exists(states_dir):
                model.state_models = {}
                
                # 列出所有状态目录
                state_dirs = [d for d in os.listdir(states_dir) if os.path.isdir(os.path.join(states_dir, d))]
                
                for state_dir in state_dirs:
                    state = int(state_dir.split('_')[1])
                    model.state_models[state] = {}
                    
                    state_path = os.path.join(states_dir, state_dir)
                    
                    for model_name in ['xgboost', 'lightgbm']:
                        model_path = os.path.join(state_path, f"{model_name}.pkl")
                        if os.path.exists(model_path):
                            with open(model_path, 'rb') as f:
                                model.state_models[state][model_name] = pickle.load(f)
            
            logger.info(f"成功从{model_dir}加载模型")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return None

    def is_trained(self):
        """检查模型是否已训练"""
        # 检查基本模型是否至少有一个已经训练
        has_trained_model = False
        if hasattr(self, 'models'):
            for model_name in ['xgboost', 'lightgbm', 'catboost', 'lnn']:
                if model_name in self.models and self.models[model_name] is not None:
                    has_trained_model = True
                    break
        return has_trained_model

# 测试代码
if __name__ == "__main__":
    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建测试数据
    import numpy as np
    np.random.seed(42)
    X_train = np.random.rand(1000, 20)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.rand(200, 20)
    
    # 创建并训练模型
    model = EnsembleModel(input_size=20, hidden_size=64, output_size=2)
    model.train(X_train, y_train)
    
    # 进行预测
    predictions = model.predict(X_test)
    print(f"预测形状: {predictions.shape}")
    
    # 保存模型
    model.save("temp_ensemble_model")
    
    # 加载模型
    loaded_model = EnsembleModel.load("temp_ensemble_model")
    
    # 使用加载的模型预测
    if loaded_model:
        loaded_predictions = loaded_model.predict(X_test)
        print(f"加载的模型预测形状: {loaded_predictions.shape}")
        print(f"原始和加载模型预测是否相近: {np.allclose(predictions, loaded_predictions, rtol=1e-5, atol=1e-5)}")
    
    # 清理临时文件
    import shutil
    if os.path.exists("temp_ensemble_model"):
        shutil.rmtree("temp_ensemble_model")
        print("已删除临时模型目录") 