#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
液体神经网络 (Liquid Neural Network) 实现

参考 MIT 的研究: https://arxiv.org/abs/2006.04439
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import traceback

logger = logging.getLogger(__name__)

class LiquidCell(nn.Module):
    """
    液体神经元细胞，可自适应调整时间常数和非线性激活函数
    """
    def __init__(self, input_size, hidden_size, time_constant=0.1):
        super(LiquidCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入权重
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        # 隐藏状态权重
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        # 偏置
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        
        # 时间常数（可学习）
        self.tau = nn.Parameter(torch.ones(hidden_size) * time_constant)
        
        # 非线性系数（可学习的非线性度）
        self.alpha = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.orthogonal_(self.W_hh)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, h_prev=None):
        """
        前向传播
        
        Args:
            x: 输入数据，形状 [batch_size, input_size]
            h_prev: 上一时刻隐藏状态，形状 [batch_size, hidden_size]
            
        Returns:
            h_next: 下一时刻隐藏状态
        """
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # 计算输入投影
        i_h = F.linear(x, self.W_ih, self.bias)
        
        # 计算隐藏状态投影
        h_h = F.linear(h_prev, self.W_hh)
        
        # 组合投影
        z = i_h + h_h
        logger.debug(f"液体细胞内组合投影z: min={z.min().item():.6f}, max={z.max().item():.6f}, mean={z.mean().item():.6f}")
        
        # 使用可学习的激活函数
        a = torch.tanh(z)
        f = self.alpha * a + self.beta
        logger.debug(f"液体细胞激活后f: min={f.min().item():.6f}, max={f.max().item():.6f}, mean={f.mean().item():.6f}")
        
        # 使用可学习的时间常数更新隐藏状态
        # Formula: h_next = (1 - dt/tau) * h_prev + (dt/tau) * f
        # 为简化计算，我们设定 dt = 1
        dt_tau = 1.0 / (1.0 + torch.exp(-self.tau))  # 使用sigmoid保证dt/tau在(0,1)范围内
        h_next = (1 - dt_tau.unsqueeze(0)) * h_prev + dt_tau.unsqueeze(0) * f
        
        return h_next

class LiquidNetwork(nn.Module):
    """
    液体神经网络，由多层液体神经元细胞组成
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LiquidNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 创建多层液体神经元
        self.cells = nn.ModuleList([
            LiquidCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size
            )
            for i in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # 使用较小的初始权重初始化输出层，避免输出过大
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)
        
        logger.info(f"LiquidNetwork初始化: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}")
        logger.info(f"输出层权重范围: min={self.output_layer.weight.min().item():.6f}, max={self.output_layer.weight.max().item():.6f}")
        
    def forward(self, x, h_prev=None):
        """
        前向传播
        
        Args:
            x: 输入数据，形状 [batch_size, input_size]
            h_prev: 上一时刻的隐藏状态列表，长度为num_layers
            
        Returns:
            output: 输出
            h_next: 下一时刻的隐藏状态列表
        """
        batch_size = x.size(0)
        logger.info(f"输入层数据统计: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
        
        # 初始化隐藏状态
        if h_prev is None:
            h_prev = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                     for _ in range(self.num_layers)]
        
        # 隐藏层前向传播
        h_next = []
        current_input = x
        
        for i, cell in enumerate(self.cells):
            h_i = cell(current_input, h_prev[i])
            logger.info(f"液体细胞{i}输出统计: min={h_i.min().item():.6f}, max={h_i.max().item():.6f}, mean={h_i.mean().item():.6f}")
            h_next.append(h_i)
            current_input = h_i
        
        # 输出层前记录最终隐藏层状态
        logger.info(f"输出层输入统计: min={h_next[-1].min().item():.6f}, max={h_next[-1].max().item():.6f}, mean={h_next[-1].mean().item():.6f}")
        
        # 对输出层的输入进行归一化处理，防止激活值过大导致极端输出
        # 应用简单的min-max归一化，保持原始分布形状但缩小范围
        normalized_input = h_next[-1]
        if normalized_input.numel() > 0 and (normalized_input.max() - normalized_input.min()) > 1e-6:
            normalized_range = 4.0  # 将值控制在[-2, 2]范围内
            input_min = normalized_input.min(dim=1, keepdim=True)[0]
            input_max = normalized_input.max(dim=1, keepdim=True)[0]
            normalized_input = (normalized_input - input_min) / (input_max - input_min) * normalized_range - normalized_range/2
            logger.info(f"归一化后输出层输入统计: min={normalized_input.min().item():.6f}, max={normalized_input.max().item():.6f}, mean={normalized_input.mean().item():.6f}")
        
        # 输出层
        output = self.output_layer(normalized_input)
        logger.info(f"输出层输出统计: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
        
        return output, h_next

class LiquidNetModel:
    """
    液体神经网络模型包装类，提供训练和预测接口
    """
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, 
                learning_rate=0.001, device="auto"):
        """
        初始化液体神经网络模型
        
        Args:
            input_size: 输入特征维度
            output_size: 输出维度
            hidden_size: 隐藏层大小
            num_layers: 隐藏层数量
            learning_rate: 学习率
            device: 使用的设备，"auto"表示自动选择
        """
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = LiquidNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers
        ).to(self.device)
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 设置损失函数
        if output_size == 1:  # 二分类或回归
            self.criterion = nn.BCEWithLogitsLoss() if output_size == 1 else nn.MSELoss()
        else:  # 多分类
            self.criterion = nn.CrossEntropyLoss()
            
        # 特征标准化
        self.scaler = StandardScaler()
        
        # 保存超参数
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def _prepare_data(self, X, y=None, batch_size=32, shuffle=True):
        """
        准备数据加载器
        
        Args:
            X: 特征数据，numpy数组
            y: 标签数据，numpy数组
            batch_size: 批大小
            shuffle: 是否打乱数据
            
        Returns:
            数据加载器
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            if self.output_size == 1:  # 二分类或回归
                y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
            else:  # 多分类
                y_tensor = torch.LongTensor(y).to(self.device)
                
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
            
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def fit(self, X, y, batch_size=32, epochs=100, validation_data=None, 
           callbacks=None, early_stopping=True, patience=10, verbose=1):
        """
        训练模型
        
        Args:
            X: 训练特征，numpy数组
            y: 训练标签，numpy数组
            batch_size: 批大小
            epochs: 训练轮数
            validation_data: 验证数据，形式为(X_val, y_val)
            callbacks: 回调函数列表
            early_stopping: 是否启用早停
            patience: 早停耐心值
            verbose: 日志详细程度
            
        Returns:
            训练历史记录
        """
        # 标准化特征
        X_train = self.scaler.fit_transform(X)
        
        # 准备数据加载器
        train_loader = self._prepare_data(X_train, y, batch_size, shuffle=True)
        
        # 准备验证数据
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = self.scaler.transform(X_val)
            val_loader = self._prepare_data(X_val, y_val, batch_size, shuffle=False)
        
        # 训练历史记录
        history = {
            'loss': [],
            'val_loss': [] if validation_data is not None else None,
            'accuracy': [],
            'val_accuracy': [] if validation_data is not None else None
        }
        
        # 早停相关
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_count = 0
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch in train_loader:
                # 获取数据
                if len(batch) == 2:
                    X_batch, y_batch = batch
                else:
                    X_batch = batch[0]
                    
                # 前向传播
                self.optimizer.zero_grad()
                outputs, _ = self.model(X_batch)
                
                # 计算损失
                if self.output_size == 1:  # 二分类或回归
                    loss = self.criterion(outputs, y_batch)
                    
                    # 计算准确率（仅对二分类）
                    predictions = torch.sigmoid(outputs) >= 0.5
                    y_true = y_batch >= 0.5
                    correct += (predictions == y_true).sum().item()
                else:  # 多分类
                    loss = self.criterion(outputs, y_batch)
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == y_batch).sum().item()
                
                total += y_batch.size(0)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 计算训练指标
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct / total if total > 0 else 0.0
            
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(train_accuracy)
            
            # 验证阶段
            if validation_data is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        X_batch, y_batch = batch
                        
                        # 前向传播
                        outputs, _ = self.model(X_batch)
                        
                        # 计算损失
                        if self.output_size == 1:  # 二分类或回归
                            loss = self.criterion(outputs, y_batch)
                            
                            # 计算准确率（仅对二分类）
                            predictions = torch.sigmoid(outputs) >= 0.5
                            y_true = y_batch >= 0.5
                            val_correct += (predictions == y_true).sum().item()
                        else:  # 多分类
                            loss = self.criterion(outputs, y_batch)
                            
                            # 计算准确率
                            _, predicted = torch.max(outputs.data, 1)
                            val_correct += (predicted == y_batch).sum().item()
                        
                        val_total += y_batch.size(0)
                        val_loss += loss.item()
                
                # 计算验证指标
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total if val_total > 0 else 0.0
                
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # 早停检查
                if early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_state = self.model.state_dict().copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    if no_improve_count >= patience:
                        logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                        break
            
            # 打印进度
            if verbose >= 1 and (epoch + 1) % (max(1, epochs // 10)) == 0:
                log_message = f"轮次 {epoch+1}/{epochs}"
                log_message += f" - loss: {avg_train_loss:.4f}"
                log_message += f" - accuracy: {train_accuracy:.4f}"
                
                if validation_data is not None:
                    log_message += f" - val_loss: {avg_val_loss:.4f}"
                    log_message += f" - val_accuracy: {val_accuracy:.4f}"
                
                logger.info(log_message)
        
        # 恢复最佳模型
        if early_stopping and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("已恢复最佳模型")
        
        return history
    
    def predict(self, X):
        """
        预测函数
        
        Args:
            X: 特征数据，numpy数组
            
        Returns:
            预测结果，numpy数组
        """
        # 添加日志记录原始数据
        logger.info(f"预测输入原始数据统计: 形状={X.shape}, min={np.min(X):.6f}, max={np.max(X):.6f}, mean={np.mean(X):.6f}, 非零值比例={np.count_nonzero(X)/X.size:.4f}")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        logger.info(f"标准化后特征统计: min={np.min(X_scaled):.6f}, max={np.max(X_scaled):.6f}, mean={np.mean(X_scaled):.6f}, std={np.std(X_scaled):.6f}")
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs, hidden_states = self.model(X_tensor)
            # 记录原始输出和隐藏状态统计信息
            logger.info(f"模型原始输出统计: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}, 非零值比例={torch.count_nonzero(outputs)/outputs.numel():.4f}")
            
            for i, h in enumerate(hidden_states):
                logger.info(f"隐藏层{i}统计: min={h.min().item():.6f}, max={h.max().item():.6f}, mean={h.mean().item():.6f}, 非零值比例={torch.count_nonzero(h)/h.numel():.4f}")
            
            # 1. 对输出进行裁剪，避免极端值
            clipped_outputs = torch.clamp(outputs, min=-10.0, max=10.0)
            logger.info(f"裁剪后输出统计: min={clipped_outputs.min().item():.6f}, max={clipped_outputs.max().item():.6f}, mean={clipped_outputs.mean().item():.6f}")
            
            if self.output_size == 1:  # 二分类或回归
                # 2. 实现分段Sigmoid函数，提高数值稳定性
                def stable_sigmoid(x):
                    # 拆分正负值分别处理，避免指数溢出
                    pos_mask = x >= 0
                    neg_mask = x < 0
                    
                    result = torch.zeros_like(x)
                    # 正值: 1/(1+exp(-x))
                    result[pos_mask] = 1.0 / (1.0 + torch.exp(-x[pos_mask]))
                    # 负值: exp(x)/(1+exp(x))
                    result[neg_mask] = torch.exp(x[neg_mask]) / (1.0 + torch.exp(x[neg_mask]))
                    return result
                
                # 使用稳定版Sigmoid
                predictions = stable_sigmoid(clipped_outputs).cpu().numpy().flatten()
                logger.info(f"使用稳定Sigmoid后的预测统计: min={np.min(predictions):.6f}, max={np.max(predictions):.6f}, mean={np.mean(predictions):.6f}")
            else:  # 多分类
                _, predictions = torch.max(clipped_outputs.data, 1)
                predictions = predictions.cpu().numpy()
        
        # 记录最终预测结果统计
        logger.info(f"最终预测结果统计: min={np.min(predictions):.6f}, max={np.max(predictions):.6f}, mean={np.mean(predictions):.6f}, 非零值数量={np.count_nonzero(predictions)}/{len(predictions)}")
        
        return predictions
    
    def predict_proba(self, X):
        """
        预测概率函数（仅对分类问题）
        
        Args:
            X: 特征数据，numpy数组
            
        Returns:
            预测概率，numpy数组
        """
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            
            if self.output_size == 1:  # 二分类
                proba = torch.sigmoid(outputs).cpu().numpy()
                # 二分类返回两列概率 [1-p, p]
                return np.hstack([1-proba, proba])
            else:  # 多分类
                proba = F.softmax(outputs, dim=1).cpu().numpy()
                return proba
        
    def save(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers
        }, filepath)
        
        logger.info(f"模型已保存到 {filepath}")
    
    def load(self, filepath):
        """
        Load model state from file.
        """
        try:
            logger.info(f"从 {filepath} 加载模型")
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 记录加载前的输出层权重统计
            if hasattr(self.model, 'output_layer'):
                logger.info(f"加载前输出层权重统计: min={self.model.output_layer.weight.min().item():.6f}, "
                           f"max={self.model.output_layer.weight.max().item():.6f}, "
                           f"mean={self.model.output_layer.weight.mean().item():.6f}")
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 暂存原始输出层权重（如果存在）
                if hasattr(self.model, 'output_layer') and 'model_state_dict' in checkpoint:
                    original_output_weights = None
                    output_layer_key = 'output_layer.weight'
                    if output_layer_key in checkpoint['model_state_dict']:
                        original_output_weights = checkpoint['model_state_dict'][output_layer_key].clone()
                        original_output_bias = checkpoint['model_state_dict']['output_layer.bias'].clone()
                        logger.info(f"保存原始模型输出层权重: shape={original_output_weights.shape}, "
                                  f"min={original_output_weights.min().item():.6f}, "
                                  f"max={original_output_weights.max().item():.6f}, "
                                  f"mean={original_output_weights.mean().item():.6f}, "
                                  f"std={original_output_weights.std().item():.6f}")
                
                # 加载模型状态
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'scaler_state' in checkpoint and self.scaler:
                    self.scaler.__dict__.update(checkpoint['scaler_state'])
                
                if 'optimizer_state_dict' in checkpoint and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                if 'scaler' in checkpoint and self.fp16_scaler:
                    self.fp16_scaler.load_state_dict(checkpoint['scaler'])
                
                # 特殊处理输出层权重
                if hasattr(self.model, 'output_layer') and original_output_weights is not None:
                    # 保持原始权重的分布特性，但缩小幅度
                    with torch.no_grad():
                        # 计算原始权重的均值和标准差
                        orig_mean = original_output_weights.mean().item()
                        orig_std = original_output_weights.std().item()
                        
                        # 计算权重间的相对差异
                        if orig_std > 0:
                            # 使用原始权重的分布，但将标准差缩小到0.01左右
                            target_std = 0.01
                            scale_factor = target_std / orig_std
                            
                            # 计算归一化后的权重
                            normalized_weights = (original_output_weights - orig_mean) * scale_factor
                            
                            # 为避免所有权重接近于0，增加一些随机扰动
                            noise = torch.randn_like(normalized_weights) * target_std * 0.1
                            final_weights = normalized_weights + noise
                            
                            # 更新输出层权重
                            self.model.output_layer.weight.copy_(final_weights)
                            
                            # 偏置也设置为小值但保持原有分布
                            bias_scale = 0.01 / (original_output_bias.std().item() + 1e-8)
                            final_bias = original_output_bias * bias_scale
                            self.model.output_layer.bias.copy_(final_bias)
                            
                            logger.info(f"保持分布的输出层权重设置完成: min={self.model.output_layer.weight.min().item():.6f}, "
                                      f"max={self.model.output_layer.weight.max().item():.6f}, "
                                      f"mean={self.model.output_layer.weight.mean().item():.6f}, "
                                      f"std={self.model.output_layer.weight.std().item():.6f}")
                        else:
                            # 如果原始权重标准差接近0，则使用随机初始化
                            logger.warning("原始权重标准差接近0，使用随机初始化输出层")
                            nn.init.normal_(self.model.output_layer.weight, mean=0.0, std=0.01)
                            nn.init.zeros_(self.model.output_layer.bias)
            else:
                self.model.load_state_dict(checkpoint)
                
                # 如果是直接加载的状态字典，也应处理输出层
                if hasattr(self.model, 'output_layer'):
                    # 记录加载后的权重统计
                    original_std = self.model.output_layer.weight.std().item()
                    
                    if original_std > 1.0:  # 如果标准差过大
                        # 缩小权重但保持分布
                        with torch.no_grad():
                            target_std = 0.01
                            scale_factor = target_std / original_std
                            normalized_weights = self.model.output_layer.weight * scale_factor
                            self.model.output_layer.weight.copy_(normalized_weights)
                            
                            # 缩小偏置
                            bias_scale = target_std / (self.model.output_layer.bias.std().item() + 1e-8)
                            self.model.output_layer.bias.mul_(bias_scale)
                            
                            logger.info(f"直接加载后调整输出层权重: std从{original_std:.6f}调整到{self.model.output_layer.weight.std().item():.6f}")
            
            logger.info(f"模型加载完成")
            return True
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            traceback.print_exc()
            return False 