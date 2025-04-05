#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
液体神经网络(Liquid Neural Network)模型实现

该模型特别适合处理时间序列数据，能更好地捕捉板块轮动和市场状态变化特征
基于MIT研究的LNN模型改进，专为金融时间序列优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class LiquidTimeCell(nn.Module):
    """
    液体时间单元(LTC) - LNN的基础构建块
    
    LTC使用连续时间动力学系统表示，能捕捉复杂的时间依赖关系
    τ参数控制状态更新速度，使模型能适应不同时间尺度的特征变化
    """
    def __init__(self, input_size, hidden_size, tau_init=1.0, 
                 nonlinearity='tanh', use_bias=True):
        super(LiquidTimeCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.use_bias = use_bias
        
        # 权重初始化
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        
        # 时间常数参数 τ (tau) - 控制状态更新速度
        # 较小的tau值导致状态更新更快，较大的tau值使状态更稳定
        self.tau = nn.Parameter(torch.ones(hidden_size) * tau_init)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        
        if self.use_bias:
            nn.init.zeros_(self.bias)
            
        # 确保tau是正值且初始值合理
        with torch.no_grad():
            self.tau.clamp_(min=0.1, max=10.0)
    
    def forward(self, input, hx=None, dt=1.0):
        """
        前向传播
        
        Args:
            input: 输入特征 [batch_size, input_size]
            hx: 上一个时间步的隐藏状态 [batch_size, hidden_size]
            dt: 时间步长，控制更新幅度
            
        Returns:
            新的隐藏状态
        """
        # 初始化隐藏状态(如果为None)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, 
                           device=input.device, dtype=input.dtype)
        
        # 计算输入映射
        i_t = F.linear(input, self.weight_ih, 
                      self.bias if self.use_bias else None)
        
        # 计算隐藏状态映射
        h_t = F.linear(hx, self.weight_hh, None)
        
        # 非线性激活函数选择
        if self.nonlinearity == 'tanh':
            activated = torch.tanh(i_t + h_t)
        elif self.nonlinearity == 'relu':
            activated = F.relu(i_t + h_t)
        else:
            raise ValueError(f"不支持的非线性激活函数: {self.nonlinearity}")
        
        # 使用时间常数τ控制状态更新 - 关键的LNN机制
        # 公式: h_new = h_old + (dt/tau) * (activated - h_old)
        tau_expanded = self.tau.expand_as(hx)  # 将tau扩展到与hx相同的形状
        h_new = hx + (dt / tau_expanded) * (activated - hx)
        
        return h_new

class LiquidNeuralNetwork(nn.Module):
    """
    液体神经网络模型 - 用于时间序列预测和分类
    
    特点:
    1. 多层LTC结构，捕捉不同抽象级别的时间特征
    2. 动态时间常数，自适应不同时间尺度的市场变化
    3. 专为金融时间序列特征优化的架构
    """
    def __init__(self, input_size, hidden_sizes, output_size, 
                 tau_init=1.0, dropout=0.2, batch_norm=True):
        super(LiquidNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.batch_norm = batch_norm
        
        # 构建多层LTC结构
        self.cells = nn.ModuleList()
        layer_in_size = input_size
        
        for hidden_size in hidden_sizes:
            cell = LiquidTimeCell(
                input_size=layer_in_size,
                hidden_size=hidden_size,
                tau_init=tau_init,
                nonlinearity='tanh'
            )
            self.cells.append(cell)
            layer_in_size = hidden_size
        
        # 批归一化层 - 提高训练稳定性
        if batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hs) for hs in hidden_sizes
            ])
        
        # Dropout层 - 防止过拟合
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in hidden_sizes
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x, hidden_states=None, dt=1.0):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_size]
            hidden_states: 各层的初始隐藏状态
            dt: 时间步长
            
        Returns:
            输出预测和新的隐藏状态
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态(如果为None)
        if hidden_states is None:
            hidden_states = [None] * len(self.cells)
        
        new_hidden_states = []
        current_input = x
        
        # 通过各层LTC
        for i, cell in enumerate(self.cells):
            h_new = cell(current_input, hidden_states[i], dt)
            new_hidden_states.append(h_new)
            
            # 应用批归一化(如果启用)
            if self.batch_norm:
                h_new = self.batch_norms[i](h_new)
            
            # 应用dropout
            h_new = self.dropouts[i](h_new)
            
            current_input = h_new
        
        # 输出层
        output = self.output_layer(current_input)
        
        return output, new_hidden_states
    
    def predict_proba(self, x):
        """
        预测概率分布
        
        Args:
            x: 输入特征 [batch_size, input_size]
            
        Returns:
            预测的概率分布
        """
        output, _ = self.forward(x)
        return F.softmax(output, dim=1).detach().cpu().numpy()
    
    def predict(self, x):
        """
        预测类别
        
        Args:
            x: 输入特征
            
        Returns:
            预测的类别
        """
        output, _ = self.forward(x)
        return torch.argmax(output, dim=1).detach().cpu().numpy()

class LNNClassifier(BaseEstimator, ClassifierMixin):
    """
    LNN分类器，提供与scikit-learn兼容的接口
    
    这个封装器使LNN模型可以无缝集成到现有的模型融合框架中
    支持与XGBoost、LightGBM等模型一起使用
    """
    def __init__(self, input_size=None, hidden_sizes=[64, 32], output_size=2, 
                 learning_rate=0.001, batch_size=64, epochs=100, 
                 dropout=0.2, tau_init=1.0, device=None,
                 early_stopping=True, patience=10, tensorboard=False,
                 class_weight=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.tau_init = tau_init
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.early_stopping = early_stopping
        self.patience = patience
        self.tensorboard = tensorboard
        self.class_weight = class_weight
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def _create_model(self, input_size=None):
        """创建LNN模型"""
        input_size = input_size or self.input_size
        
        if input_size is None:
            raise ValueError("input_size必须在初始化时或调用此方法时提供")
        
        model = LiquidNeuralNetwork(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size,
            tau_init=self.tau_init,
            dropout=self.dropout
        )
        return model.to(self.device)
    
    def _get_class_weights(self, y):
        """计算类别权重"""
        if self.class_weight == 'balanced':
            class_counts = np.bincount(y)
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(self.device)
            return class_weights
        return None
    
    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            sample_weight: 样本权重
            
        Returns:
            self
        """
        # 数据预处理
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            have_validation = True
        else:
            have_validation = False
        
        # 转换为张量
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        if have_validation:
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        # 样本权重处理
        if sample_weight is not None:
            sample_weight_tensor = torch.tensor(sample_weight, dtype=torch.float32).to(self.device)
        else:
            sample_weight_tensor = None
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        if have_validation:
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        
        # 初始化模型
        self.input_size = X.shape[1]
        self.model = self._create_model()
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        class_weights = self._get_class_weights(y)
        
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # 前向传播
                outputs, _ = self.model(batch_X)
                
                # 计算损失
                if sample_weight_tensor is not None:
                    batch_indices = torch.arange(len(batch_y))
                    batch_weights = sample_weight_tensor[batch_indices]
                    loss = criterion(outputs, batch_y) * batch_weights.mean()
                else:
                    loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / len(dataloader)
            train_acc = correct / total
            
            # 验证阶段
            if have_validation:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        outputs, _ = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_loss = val_loss / len(val_dataloader)
                val_acc = val_correct / val_total
                
                # 早停检查
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        logger.info(f"早停触发，在第{epoch+1}个epoch停止训练")
                        break
                
                # 记录历史
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                           f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                           f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # 记录训练历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 数据预处理
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            probas = F.softmax(outputs, dim=1).cpu().numpy()
        
        return probas
    
    def predict(self, X):
        """预测类别"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def save(self, path):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存PyTorch模型参数
        torch_path = f"{path}.pt"
        torch.save(self.model.state_dict(), torch_path)
        
        # 保存scaler和其他配置
        config = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout': self.dropout,
            'tau_init': self.tau_init,
            'scaler': self.scaler,
            'history': self.history
        }
        
        joblib.dump(config, f"{path}.joblib")
        logger.info(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型"""
        # 加载配置
        config = joblib.load(f"{path}.joblib")
        
        # 恢复参数
        self.input_size = config['input_size']
        self.hidden_sizes = config['hidden_sizes']
        self.output_size = config['output_size']
        self.dropout = config['dropout'] 
        self.tau_init = config['tau_init']
        self.scaler = config['scaler']
        self.history = config['history']
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载模型参数
        self.model.load_state_dict(torch.load(f"{path}.pt", 
                                            map_location=self.device))
        self.model.eval()
        
        self.is_fitted = True
        logger.info(f"模型已从 {path} 加载")
        return self 