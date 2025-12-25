#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
液体神经网络 (Liquid Neural Network) 实现 (修改后支持序列输入)

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
import time # 引入 time

# 假设 logger_config.py 可以正常导入并配置
try:
    from to.logger_config import configure_logger
    logger = configure_logger(__name__, 'liquid_network.log')
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("无法导入自定义 logger_config，使用 basicConfig。")


class LiquidCell(nn.Module):
    """
    液体神经元细胞 (与之前版本基本一致，处理单个时间步)
    """
    def __init__(self, input_size, hidden_size, time_constant=0.1):
        super(LiquidCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size) * time_constant)
        self.alpha = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.orthogonal_(self.W_hh)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        i_h = F.linear(x, self.W_ih, self.bias)
        h_h = F.linear(h_prev, self.W_hh)
        z = i_h + h_h
        a = torch.tanh(z)
        f = self.alpha * a + self.beta
        dt_tau = torch.sigmoid(self.tau) # 使用 sigmoid 保证 dt/tau 在 (0,1)
        h_next = (1 - dt_tau.unsqueeze(0)) * h_prev + dt_tau.unsqueeze(0) * f
        
        # 添加类型检查日志
        if not isinstance(h_next, torch.Tensor):
            logger.error(f"LNN CELL DEBUG: h_next IS NOT A TENSOR! Type: {type(h_next)}")
            
        return h_next

class LiquidNetwork(nn.Module):
    """
    液体神经网络 (修改后支持序列输入)
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LiquidNetwork, self).__init__()
        self.input_size = input_size # 现在是每个时间步的特征数
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            LiquidCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size
            )
            for i in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)
        
        logger.info(f"LiquidNetwork初始化(序列版): input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, num_layers={num_layers}")
        
    def forward(self, x_seq, h_prev=None):
        """
        前向传播 (处理序列输入)
        
        Args:
            x_seq: 输入序列数据，形状 [batch_size, sequence_length, input_size]
            h_prev: 上一序列（或初始）隐藏状态列表，长度为num_layers, 每个元素形状 [batch_size, hidden_size]
            
        Returns:
            final_output: 最后一个时间步的输出, 形状 [batch_size, output_size]
            h_final_list: 最后一个时间步的隐藏状态列表, 长度 num_layers, 每个元素形状 [batch_size, hidden_size]
        """
        batch_size, sequence_length, _ = x_seq.shape # 获取维度
        # logger.debug(f"LNN forward: 输入 x_seq 形状: {x_seq.shape}, device: {x_seq.device}")

        if h_prev is None:
            h_prev = [torch.zeros(batch_size, self.hidden_size, device=x_seq.device) 
                     for _ in range(self.num_layers)]
        
        outputs_last_layer_over_time = [] # 存储最后一个隐藏层在每个时间步的状态

        # 遍历时间步
        for t in range(sequence_length):
            current_step_input = x_seq[:, t, :] # 当前时间步输入: [batch_size, input_size]
            h_next_list_this_step = [] # 存储当前时间步计算出的下一隐藏状态
            
            input_to_cell_t = current_step_input
            for i, cell_module in enumerate(self.cells):
                # 输入是当前时间步的输入(或上一层的输出) 和 上一个时间步的隐藏状态 h_prev[i]
                h_i_next_t = cell_module(input_to_cell_t, h_prev[i]) 
                h_next_list_this_step.append(h_i_next_t)
                input_to_cell_t = h_i_next_t # 更新当前时间步的层间传递
            
            # 更新 h_prev 供下一个时间步使用
            h_prev = h_next_list_this_step 
            
            # 记录最后一个隐藏层在当前时间步的状态
            outputs_last_layer_over_time.append(h_prev[-1])

        # 最终模型输出通常基于最后一个时间步的状态
        last_step_hidden_state = outputs_last_layer_over_time[-1] # [batch_size, hidden_size]
        
        # 对最终隐藏状态进行归一化（可选，沿用之前的逻辑，但更健壮）
        normalized_final_hidden = last_step_hidden_state
        try: 
            temp_for_norm_final = last_step_hidden_state.clone() 
            if temp_for_norm_final.numel() > 0 and temp_for_norm_final.ndim == 2 \
               and temp_for_norm_final.shape[0] > 0 and temp_for_norm_final.shape[1] > 0:
                if not isinstance(temp_for_norm_final, torch.Tensor):
                     raise ValueError("temp_for_norm_final is not a PyTorch Tensor during normalization attempt.")
                
                min_val_f = temp_for_norm_final.min(dim=1, keepdim=True)[0]
                max_val_f = temp_for_norm_final.max(dim=1, keepdim=True)[0]
                range_val_f = max_val_f - min_val_f
                denominator_f = torch.where(range_val_f.abs() < 1e-7, torch.ones_like(range_val_f) * 1e-7, range_val_f)
                normalized_range = 4.0
                normalized_final_hidden = (temp_for_norm_final - min_val_f) / denominator_f * normalized_range - normalized_range / 2.0
                # logger.debug(f"LNN forward: 最终隐藏状态归一化后: min={normalized_final_hidden.min().item():.6f}, max={normalized_final_hidden.max().item():.6f}")
            elif temp_for_norm_final.numel() > 0: # 如果维度不符合预期
                 logger.warning(f"LNN forward: 最终隐藏状态形状 {temp_for_norm_final.shape} 不符合二维归一化预期，跳过归一化。")

        except TypeError as e_type_norm_final:
            logger.error(f"LNN内部最终隐藏状态归一化时捕获到TypeError: {e_type_norm_final}")
            logger.error(f"  发生时temp_for_norm_final类型: {type(last_hidden_state)}, 形状: {last_hidden_state.shape}")
            logger.error(traceback.format_exc())
        except Exception as e_other_norm_final:
            logger.error(f"LNN内部最终隐藏状态归一化时发生其他错误: {e_other_norm_final}. 使用原始隐藏状态。")

        final_output = self.output_layer(normalized_final_hidden) # 应用输出层
        # logger.debug(f"LNN forward: 最终输出形状: {final_output.shape}")
        
        # 返回最后一个时间步的输出和最后一个时间步的隐藏状态列表
        return final_output, h_prev 


class LiquidNetModel:
    """
    液体神经网络模型包装类 (修改后支持序列输入)
    """
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, 
                 learning_rate=0.001, device="auto"):
        if device == "auto": self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = torch.device(device)
        logger.info(f"使用设备: {self.device}")
        
        self.model = LiquidNetwork(input_size, hidden_size, output_size, num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss() if output_size == 1 else nn.CrossEntropyLoss() # 假设主要用于二分类或回归
        self.scaler = StandardScaler() # Scaler 现在需要在 fit 和 predict 中处理 3D 数据
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def _prepare_data(self, X_tensor, y_tensor=None, batch_size=32, shuffle=True):
        """准备数据加载器 (输入已是Tensor)"""
        if y_tensor is not None:
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def fit(self, X, y, batch_size=32, epochs=100, validation_data=None, 
           callbacks=None, early_stopping=True, patience=10, verbose=1):
        """
        训练模型 (处理序列输入 X)
        
        Args:
            X: 训练特征，numpy数组, 形状 [样本数, 时间步数, 特征数]
            y: 训练标签，numpy数组, 形状 [样本数] 或 [样本数, 1]
            ... (其他参数不变)
        """
        if X.ndim != 3:
            raise ValueError(f"输入特征 X 的维度应为 3 (样本数, 时间步数, 特征数)，但收到了 {X.ndim} 维")
        
        num_samples, sequence_length, num_features = X.shape
        if num_features != self.input_size:
             raise ValueError(f"输入特征 X 的第三维 ({num_features}) 与模型初始化的 input_size ({self.input_size}) 不符")

        logger.info(f"LNN fit: 输入 X 形状: {X.shape}")
        
        # MODIFICATION: 标准化处理 3D 数据
        # 1. Reshape to 2D: [样本数 * 时间步数, 特征数]
        X_reshaped = X.reshape(-1, num_features)
        # 2. Fit and transform the scaler on the 2D data
        logger.info("LNN fit: 正在进行特征标准化 (fit_transform)...")
        X_scaled_reshaped = self.scaler.fit_transform(X_reshaped)
        # 3. Reshape back to 3D: [样本数, 时间步数, 特征数]
        X_train_scaled = X_scaled_reshaped.reshape(num_samples, sequence_length, num_features)
        logger.info(f"LNN fit: 标准化后 X_train_scaled 形状: {X_train_scaled.shape}")

        # 准备训练数据加载器
        y_train_tensor = torch.FloatTensor(y.reshape(-1, 1) if self.output_size==1 else y).to(self.device) # 调整y形状
        train_loader = self._prepare_data(torch.FloatTensor(X_train_scaled).to(self.device), y_train_tensor, batch_size, shuffle=True)
        
        # 准备验证数据
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if X_val.ndim != 3 or X_val.shape[2] != num_features:
                 raise ValueError("验证数据 X_val 维度或特征数不正确")
            logger.info(f"LNN fit: 验证数据 X_val 形状: {X_val.shape}")
            # MODIFICATION: 标准化验证数据
            X_val_reshaped = X_val.reshape(-1, num_features)
            X_val_scaled_reshaped = self.scaler.transform(X_val_reshaped) # Use transform, not fit_transform
            X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape[0], sequence_length, num_features)
            logger.info(f"LNN fit: 标准化后 X_val_scaled 形状: {X_val_scaled.shape}")
            
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1) if self.output_size==1 else y_val).to(self.device)
            val_loader = self._prepare_data(torch.FloatTensor(X_val_scaled).to(self.device), y_val_tensor, batch_size, shuffle=False)
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        best_val_loss, no_improve_count = float('inf'), 0
        best_model_state = None
        
        # 训练循环
        for epoch in range(epochs):
            self.model.train()
            train_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in train_loader: # DataLoader 输出 Tensor
                self.optimizer.zero_grad()
                outputs, _ = self.model(X_batch) # 模型现在处理序列输入
                loss = self.criterion(outputs, y_batch)
                loss.backward(); self.optimizer.step()
                train_loss += loss.item()
                # 计算准确率 (根据任务调整)
                if self.output_size == 1: # 二分类
                    preds = torch.sigmoid(outputs) >= 0.5
                    correct += (preds == (y_batch >= 0.5)).sum().item()
                # else: 多分类逻辑...
                total += y_batch.size(0)
            avg_train_loss = train_loss / len(train_loader); train_accuracy = correct / total if total > 0 else 0.0
            history['loss'].append(avg_train_loss); history['accuracy'].append(train_accuracy)
            
            # 验证阶段
            avg_val_loss, val_accuracy = float('inf'), 0.0 # 初始化为默认值
            if val_loader is not None:
                self.model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for X_batch_val, y_batch_val in val_loader:
                        outputs_val, _ = self.model(X_batch_val)
                        loss_val = self.criterion(outputs_val, y_batch_val)
                        val_loss += loss_val.item()
                        if self.output_size == 1:
                             preds_val = torch.sigmoid(outputs_val) >= 0.5
                             val_correct += (preds_val == (y_batch_val >= 0.5)).sum().item()
                        val_total += y_batch_val.size(0)
                avg_val_loss = val_loss / len(val_loader); val_accuracy = val_correct / val_total if val_total > 0 else 0.0
                history['val_loss'].append(avg_val_loss); history['val_accuracy'].append(val_accuracy)

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
            
            if verbose >= 1 and (epoch + 1) % (max(1, epochs // 10)) == 0:
                 log_msg = f"轮次 {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - acc: {train_accuracy:.4f}"
                 if val_loader: log_msg += f" - val_loss: {avg_val_loss:.4f} - val_acc: {val_accuracy:.4f}"
                 logger.info(log_msg)
        
        if early_stopping and best_model_state is not None:
            self.model.load_state_dict(best_model_state); logger.info("已恢复最佳模型")
        return history
    
    def predict(self, X):
        """
        预测函数 (处理序列输入 X)
        
        Args:
            X: 特征数据，numpy数组, 形状 [样本数, 时间步数, 特征数]
            
        Returns:
            预测结果，numpy数组, 形状 [样本数] 或 [样本数, 输出维度]
        """
        if X.ndim != 3: raise ValueError(f"预测输入 X 的维度应为 3，但为 {X.ndim}")
        num_samples, sequence_length, num_features = X.shape
        if num_features != self.input_size: raise ValueError(f"预测输入 X 特征数 ({num_features}) 与模型 input_size ({self.input_size}) 不符")

        logger.info(f"LNN predict: 输入 X 形状: {X.shape}")
        
        # MODIFICATION: 标准化处理 3D 数据
        X_reshaped = X.reshape(-1, num_features)
        logger.info("LNN predict: 正在进行特征标准化 (transform)...")
        X_scaled_reshaped = self.scaler.transform(X_reshaped) # 使用 transform
        X_scaled = X_scaled_reshaped.reshape(num_samples, sequence_length, num_features)
        logger.info(f"LNN predict: 标准化后 X_scaled 形状: {X_scaled.shape}")

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        predictions = None # 初始化
        with torch.no_grad():
            try:
                outputs, hidden_states = self.model(X_tensor) # forward 现在返回最终输出和隐藏状态
                # outputs 形状应为 [样本数, 输出维度]
                logger.info(f"LNN predict: 模型原始输出 shape: {outputs.shape}")
                
                clipped_outputs = torch.clamp(outputs, min=-10.0, max=10.0)
                
                if self.output_size == 1: # 二分类或回归
                    # 稳定 Sigmoid
                    def stable_sigmoid(x_sig):
                        pos = x_sig >= 0; neg = x_sig < 0
                        res = torch.zeros_like(x_sig)
                        res[pos] = 1.0 / (1.0 + torch.exp(-x_sig[pos]))
                        res[neg] = torch.exp(x_sig[neg]) / (1.0 + torch.exp(x_sig[neg]))
                        return res
                    predictions = stable_sigmoid(clipped_outputs).cpu().numpy().flatten() # [样本数]
                else: # 多分类
                    _, predictions_tensor = torch.max(clipped_outputs.data, 1) # [样本数]
                    predictions = predictions_tensor.cpu().numpy()
                    
            except Exception as e_pred_runtime:
                 logger.error(f"LNN predict forward pass or output processing failed: {e_pred_runtime}\n{traceback.format_exc()}")
                 # 可以在这里返回一个默认值或重新抛出异常
                 predictions = np.zeros(num_samples) # 返回一个形状匹配的零数组作为失败指示

        if predictions is not None:
             logger.info(f"LNN predict: 最终预测结果 shape: {predictions.shape}, min={np.min(predictions):.6f}, max={np.max(predictions):.6f}")
        
        return predictions
    
    def predict_proba(self, X):
        """预测概率函数（处理序列输入 X，仅对分类问题）"""
        if X.ndim != 3: raise ValueError("预测概率输入 X 维度应为 3")
        num_samples, sequence_length, num_features = X.shape
        if num_features != self.input_size: raise ValueError("预测概率输入 X 特征数不符")

        X_reshaped = X.reshape(-1, num_features)
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(num_samples, sequence_length, num_features)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor) # outputs shape: [样本数, 输出维度]
            if self.output_size == 1: # 二分类
                proba = torch.sigmoid(outputs).cpu().numpy() # shape: [样本数, 1]
                return np.hstack([1 - proba, proba]) # shape: [样本数, 2]
            else: # 多分类
                proba = F.softmax(outputs, dim=1).cpu().numpy() # shape: [样本数, 输出维度]
                return proba
        
    def save(self, filepath):
        """保存模型状态（包括scaler）"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state': self.scaler.__dict__, # 保存 scaler 的状态
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers
        }, filepath)
        logger.info(f"模型已保存到 {filepath}")
    
    def load(self, filepath):
        """从文件加载模型状态（包括scaler）"""
        try:
            logger.info(f"从 {filepath} 加载LNN模型...")
            # map_location=self.device 确保加载到正确的设备
            # weights_only=False 因为我们要加载scaler和optimizer状态
            checkpoint = torch.load(filepath, map_location=self.device) 
            
            if 'model_state_dict' not in checkpoint:
                # 兼容可能只保存了 state_dict 的旧格式
                self.model.load_state_dict(checkpoint)
                logger.warning("只加载了模型 state_dict，optimizer和scaler状态未加载。")
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scaler_state' in checkpoint and hasattr(self, 'scaler'):
                    self.scaler.__dict__.update(checkpoint['scaler_state']) # 恢复 scaler 状态
                    logger.info("Scaler状态已加载。")
                else:
                     logger.warning("检查点中未找到scaler状态，或当前模型没有scaler属性。")
                # 可以在这里检查 input_size, output_size 等是否匹配
                if checkpoint.get('input_size') != self.input_size or \
                   checkpoint.get('output_size') != self.output_size or \
                   checkpoint.get('hidden_size') != self.hidden_size or \
                   checkpoint.get('num_layers') != self.num_layers:
                    logger.warning(f"加载的模型参数与当前实例参数不完全匹配！"
                                   f"加载的: in={checkpoint.get('input_size')}, hidden={checkpoint.get('hidden_size')}, out={checkpoint.get('output_size')}, layers={checkpoint.get('num_layers')}. "
                                   f"当前的: in={self.input_size}, hidden={self.hidden_size}, out={self.output_size}, layers={self.num_layers}")

            self.model.to(self.device) # 确保模型在正确的设备上
            logger.info(f"LNN模型加载完成。")
            return True
        except Exception as e:
            logger.error(f"加载LNN模型时出错: {e}\n{traceback.format_exc()}")
            return False