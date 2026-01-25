#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版液体神经网络模型
针对回归任务进行优化，包括批归一化、Dropout等正则化方法
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
import pickle
import copy
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time

logger = logging.getLogger(__name__)

class LiquidTimeCell(nn.Module):
    """增强型液体时间常数单元，支持正则化"""
    def __init__(self, input_size, hidden_size, time_constant=0.1):
        super(LiquidTimeCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 线性变换权重
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 初始化时间常数τ为可学习参数
        self.tau = nn.Parameter(torch.ones(hidden_size) * time_constant)
        
        # 添加LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 参数初始化
        self.reset_parameters()
        
    def reset_parameters(self):
        """正交初始化以稳定训练"""
        nn.init.orthogonal_(self.W_g.weight)
        nn.init.orthogonal_(self.W_o.weight)
        nn.init.zeros_(self.W_g.bias)
        nn.init.zeros_(self.W_o.bias)
    
    def forward(self, x, h_prev=None):
        """前向传播函数
        参数:
            x: 输入张量 [batch_size, input_size]
            h_prev: 上一时刻隐藏状态 [batch_size, hidden_size]
        """
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            
        # 连接输入和隐藏状态
        combined = torch.cat((x, h_prev), dim=1)
        
        # 计算候选隐藏状态
        g = torch.tanh(self.W_g(combined))
        
        # 计算输出门
        o = torch.sigmoid(self.W_o(combined))
        
        # 使用时间常数τ更新隐藏状态（关键的LTC特性）
        h_new = h_prev * (1 - 1/self.tau.exp()) + g * (1/self.tau.exp())
        
        # 应用层归一化
        h_new = self.layer_norm(h_new)
        
        # 应用输出门得到最终输出
        y = o * torch.tanh(h_new)
        
        return y, h_new


class ImprovedLiquidNeuralNetwork(nn.Module):
    """增强版液体神经网络，针对回归任务优化"""
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, dropout_rate=0.2):
        super(ImprovedLiquidNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 特征归一化层
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # 创建改进的多层LTC
        self.cells = nn.ModuleList([
            LiquidTimeCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # 添加批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) 
            for _ in range(num_layers)
        ])
        
        # 添加Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出层 - 对于回归任务，不需要额外的激活函数
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x, h=None):
        """前向传播
        参数:
            x: 输入序列 [batch_size, seq_length, input_size] 或 [batch_size, input_size]
            h: 初始隐藏状态列表 [num_layers, batch_size, hidden_size]
        """
        # 处理输入维度
        original_shape = x.shape
        is_sequence = len(original_shape) > 2
        
        if is_sequence:
            batch_size, seq_length, _ = x.shape
            # 处理序列数据
            outputs = []
            
            # 如果没有提供隐藏状态，初始化为零
            if h is None:
                h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                     for _ in range(self.num_layers)]
            
            # 对序列中的每个时间步进行处理
            for t in range(seq_length):
                x_t = x[:, t, :]
                
                # 应用输入归一化
                try:
                    if x_t.size(0) > 1:  # BatchNorm需要至少两个样本
                        x_t = self.input_norm(x_t)
                except Exception as e:
                    # 如果归一化失败，继续使用原始输入
                    pass
                
                # 通过每一层LTC
                new_h = []
                for i, (cell, bn) in enumerate(zip(self.cells, self.batch_norms)):
                    x_t, h_i = cell(x_t, h[i])
                    
                    # 应用批归一化
                    try:
                        if x_t.size(0) > 1:  # BatchNorm需要至少两个样本
                            x_t = bn(x_t)
                    except Exception as e:
                        # 如果归一化失败，继续使用原始输出
                        pass
                    
                    new_h.append(h_i)
                    
                    # 除最后一层外应用Dropout
                    if i < self.num_layers - 1:
                        x_t = self.dropout(x_t)
                
                h = new_h
                outputs.append(x_t)
            
            # 仅使用序列的最后一个输出
            x = outputs[-1]
        else:
            # 处理非序列数据
            batch_size = x.shape[0]
            
            # 应用输入归一化
            try:
                if x.size(0) > 1:  # BatchNorm需要至少两个样本
                    x = self.input_norm(x)
            except Exception as e:
                # 如果归一化失败，继续使用原始输入
                pass
            
            # 如果没有提供隐藏状态，初始化为零
            if h is None:
                h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                     for _ in range(self.num_layers)]
            
            # 通过每一层LTC
            new_h = []
            for i, (cell, bn) in enumerate(zip(self.cells, self.batch_norms)):
                x, h_i = cell(x, h[i])
                
                # 应用批归一化
                try:
                    if x.size(0) > 1:  # BatchNorm需要至少两个样本
                        x = bn(x)
                except Exception as e:
                    # 如果归一化失败，继续使用原始输出
                    pass
                
                new_h.append(h_i)
                
                # 除最后一层外应用Dropout
                if i < self.num_layers - 1:
                    x = self.dropout(x)
            
            h = new_h
        
        # 输出层
        output = self.output_layer(x)
        return output, h


class ImprovedLiquidNetModel:    """增强版液体神经网络模型包装类，提供简易API"""    def __init__(self, input_size, output_size=1, hidden_size=64, num_layers=2,                  learning_rate=0.001, device="auto", dropout_rate=0.2,                 use_layer_norm=False, use_residual=False,                 lr_scheduler=None, lr_scheduler_params=None,                 clip_grad_norm=None, weight_decay=0.0):
        """
        初始化增强版液体神经网络模型
        
        Args:
            input_size: 输入特征维度
            output_size: 输出维度，默认为1（回归任务）
            hidden_size: 隐藏层大小
            num_layers: LTC层数
            learning_rate: 学习率
            device: 运行设备，"auto"将自动选择GPU/CPU
            dropout_rate: Dropout比率
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 创建模型
        self.model = ImprovedLiquidNeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 损失函数 - 对回归任务使用MSE
        self.criterion = nn.MSELoss()
        
        # 特征缩放器
        self.scaler = StandardScaler()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # 早期停止
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
        
        # 是否已训练标记
        self.is_fitted = False
    
    def _prepare_data(self, X_tensor, y_tensor=None, batch_size=32, shuffle=True, num_workers=0):
        """准备数据加载器"""
        if y_tensor is not None:
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        else:
            dataset = torch.utils.data.TensorDataset(X_tensor)
            
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,  # 使用多线程加载数据
            pin_memory=self.device.type == 'cuda'  # 仅在GPU上使用pin_memory
        )
        return loader
    
    def fit(self, X, y, batch_size=32, epochs=100, validation_data=None, 
            callbacks=None, early_stopping=True, patience=10, verbose=1,
            use_jit=False, num_workers=0):
        """
        训练模型
        
        Args:
            X: 输入特征，numpy数组或DataFrame
            y: 目标变量，numpy数组或Series
            batch_size: 批量大小
            epochs: 训练轮数
            validation_data: (X_val, y_val)元组
            callbacks: 回调函数列表
            early_stopping: 是否启用早停
            patience: 早停耐心值
            verbose: 日志详细程度
            use_jit: 是否使用JIT编译加速(CPU训练优化)
            num_workers: 数据加载线程数
            
        Returns:
            训练历史记录字典
        """
        # 转换为numpy数组
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # 调整目标维度
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # 处理3D数据的特征缩放 - 适用于序列数据
        if X.ndim == 3:
            # 获取原始形状
            orig_shape = X.shape
            n_samples, seq_len, n_features = orig_shape
            logger.info(f"处理3D序列数据: 原始形状={orig_shape}")
            
            # 重塑为2D以便StandardScaler处理 (样本*序列长度, 特征数)
            X_reshaped = X.reshape(-1, n_features)
            
            # 应用标准化
            X_scaled_2d = self.scaler.fit_transform(X_reshaped)
            
            # 重塑回3D
            X_scaled = X_scaled_2d.reshape(orig_shape)
            logger.info(f"标准化后形状={X_scaled.shape}")
        else:
            # 标准2D数据处理
            X_scaled = self.scaler.fit_transform(X)
            
        # 将日期列转换为时间戳
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Windows环境下JIT编译可能导致兼容性问题，因此禁用此功能
        if use_jit and self.device.type == 'cpu':
            logger.info("JIT优化已禁用，使用标准模式以确保兼容性")
            # 这里不执行JIT编译，避免与其他优化方式冲突
        
        # 准备数据加载器
        train_loader = self._prepare_data(X_tensor, y_tensor, batch_size, shuffle=True, num_workers=num_workers)
        
        # 准备验证数据
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
                
            # 调整目标维度
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
                
            # 特征缩放 - 使用训练集的缩放器 (处理3D数据)
            if X_val.ndim == 3:
                # 获取原始形状
                val_orig_shape = X_val.shape
                val_samples, val_seq_len, val_features = val_orig_shape
                logger.info(f"处理验证集3D序列数据: 原始形状={val_orig_shape}")
                
                # 重塑为2D
                X_val_reshaped = X_val.reshape(-1, val_features)
                
                # 应用标准化
                X_val_scaled_2d = self.scaler.transform(X_val_reshaped)
                
                # 重塑回3D
                X_val_scaled = X_val_scaled_2d.reshape(val_orig_shape)
                logger.info(f"验证集标准化后形状={X_val_scaled.shape}")
            else:
                # 标准2D数据处理
                X_val_scaled = self.scaler.transform(X_val)
                
            # 转换为Torch张量
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # 准备验证数据加载器
            val_loader = self._prepare_data(X_val_tensor, y_val_tensor, batch_size, shuffle=False, num_workers=num_workers)
        
        # 记录训练历史
        history = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': []
        }
        
        # 创建早期停止器，根据传入参数设置
        if early_stopping:
            # 使用传入的min_delta参数或默认值
            min_delta_value = getattr(locals().get('min_delta', 0.0001), 'real', 0.0001)
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta_value, restore_best_weights=True)
            logger.info(f"启用早停，耐心值={patience}, 最小改进阈值={min_delta_value}")
        
        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            # 添加时间统计
            batch_start_time = time.time()
            total_batches = len(train_loader)
            
            # 训练批次循环
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                # 正向传播
                self.optimizer.zero_grad()
                outputs, _ = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                # 记录损失和MAE
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - y_batch)).item()
                
                # 每处理10%的批次输出一次进度
                if verbose and (batch_idx + 1) % max(1, total_batches // 10) == 0:
                    elapsed = time.time() - batch_start_time
                    progress = (batch_idx + 1) / total_batches * 100
                    logger.info(f"轮次 {epoch+1} 批次进度: {batch_idx+1}/{total_batches} ({progress:.1f}%), "
                               f"用时: {elapsed:.2f}秒, 预计剩余: {elapsed/(batch_idx+1)*(total_batches-batch_idx-1):.2f}秒")
            
            # 计算平均损失和MAE
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            history['loss'].append(train_loss)
            history['mae'].append(train_mae)
            
            # 验证
            val_loss = 0.0
            val_mae = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        val_outputs, _ = self.model(X_val_batch)
                        val_batch_loss = self.criterion(val_outputs, y_val_batch)
                        val_loss += val_batch_loss.item()
                        val_mae += torch.mean(torch.abs(val_outputs - y_val_batch)).item()
                
                val_loss /= len(val_loader)
                val_mae /= len(val_loader)
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                
                # 早期停止
                if early_stopping:
                    # 传入模型以便在需要时保存最佳权重
                    should_stop = early_stopping(val_loss, self.model)
                    # 输出当前早停状态
                    if verbose:
                        logger.info(f"早停状态: 当前计数={early_stopping.counter}/{early_stopping.patience}, "
                                  f"最佳损失={early_stopping.best_loss:.6f}, 当前损失={val_loss:.6f}")
                    if should_stop:
                        logger.info(f"早期停止于第 {epoch+1} 轮")
                        break
            
            # 日志
            if verbose:
                # 更改为每一轮都输出日志，而不是只在1/10的轮次输出
                log_msg = f"轮次 {epoch+1}/{epochs}"
                log_msg += f" - loss: {train_loss:.4f}"
                log_msg += f" - mae: {train_mae:.4f}"
                if val_loader is not None:
                    log_msg += f" - val_loss: {val_loss:.4f}"
                    log_msg += f" - val_mae: {val_mae:.4f}"
                logger.info(log_msg)
                
                # 每5个batch输出一次进度日志
                if (epoch + 1) % 5 == 0:
                    logger.info(f"训练进度: {epoch+1}/{epochs} ({(epoch+1)/epochs*100:.1f}%)")
        
        self.is_fitted = True
        return history
    
    def predict(self, X):
        """
        使用模型进行预测
        
        Args:
            X: 输入特征，numpy数组或DataFrame
            
        Returns:
            预测结果数组
        """
        # 转换为numpy数组
        if hasattr(X, 'values'):
            X = X.values
            
        # 特征缩放 (处理3D数据)
        if X.ndim == 3:
            # 获取原始形状
            orig_shape = X.shape
            n_samples, seq_len, n_features = orig_shape
            
            # 重塑为2D
            X_reshaped = X.reshape(-1, n_features)
            
            # 应用标准化
            X_scaled_2d = self.scaler.transform(X_reshaped)
            
            # 重塑回3D
            X_scaled = X_scaled_2d.reshape(orig_shape)
            logger.info(f"预测时处理3D序列数据: 原始形状={orig_shape}")
        else:
            # 标准2D数据处理
            X_scaled = self.scaler.transform(X)
            
        # 转换为Torch张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 创建数据加载器以处理大量数据
        loader = self._prepare_data(X_tensor, batch_size=64, shuffle=False)
        
        # 预测模式
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X_batch in loader:
                # 数据加载器返回的是元组，需要取第一个元素
                X_batch = X_batch[0]
                outputs, _ = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
        
        # 合并批次预测结果
        predictions = np.vstack(predictions)
        
        # 如果是单列输出，转换为一维数组
        if predictions.shape[1] == 1:
            predictions = predictions.flatten()
            
        return predictions
    
    def save(self, filepath):
        """保存模型到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 准备保存数据
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'is_fitted': self.is_fitted
        }
        
        # 保存模型
        torch.save(save_dict, filepath)
        logger.info(f"模型已保存至 {filepath}")
    
    def load(self, filepath):
        """从文件加载模型"""
        # 检查文件存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件 {filepath} 不存在")
        
        # 加载保存的数据
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 更新模型属性
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.output_size = checkpoint['output_size']
        self.num_layers = checkpoint['num_layers']
        self.learning_rate = checkpoint['learning_rate']
        self.dropout_rate = checkpoint.get('dropout_rate', 0.2)  # 兼容旧版本
        self.is_fitted = checkpoint['is_fitted']
        self.scaler = checkpoint['scaler']
        
        # 重新创建模型
        self.model = ImprovedLiquidNeuralNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 重新创建优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 加载优化器参数
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 重新创建调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        logger.info(f"模型已从 {filepath} 加载")


class EarlyStopping:
    """提前停止类，用于防止过拟合"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        初始化早期停止器
        
        Args:
            patience: 容忍多少个epoch验证损失没有改善
            min_delta: 认为是改善的最小变化量
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model=None):
        """
        检查是否应该停止训练
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型（如果restore_best_weights=True）
            
        Returns:
            bool: 是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights and model is not None:
                self.best_model = copy.deepcopy(model.state_dict())
        # 判断是否有显著改善：当前损失必须比最佳损失小，且差值大于min_delta才算显著改善
        elif self.best_loss - val_loss > self.min_delta:
            # 有显著改善
            self.best_loss = val_loss
            # 更新最佳模型
            if self.restore_best_weights and model is not None:
                self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            # 没有显著改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # 如果需要恢复最佳权重
                if self.restore_best_weights and self.best_model is not None and model is not None:
                    model.load_state_dict(self.best_model)
        
        return self.early_stop 