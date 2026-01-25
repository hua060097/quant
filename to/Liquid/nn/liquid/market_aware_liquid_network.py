#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场感知型液体神经网络模型
专门设计用于处理市场特征和板块特征的双路径融合架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import pickle
import copy
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import traceback
import gc
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
import sys
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 尝试导入序列特征处理模块
try:
    from to.sector_features.sequence_features import (
        prepare_multi_scale_sequence,
        extract_sequence_features,
        apply_temporal_weights
    )
except ImportError:
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
        from sector_features.sequence_features import (
            prepare_multi_scale_sequence,
            extract_sequence_features, 
            apply_temporal_weights
        )
    except ImportError:
        print("警告: 无法导入序列特征处理模块，多尺度特征功能将不可用")

logger = logging.getLogger(__name__)


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
        self.last_improvement = 0  # 上次改善的轮次
        print(f"早停设置: patience={patience}, min_delta={min_delta}, restore_best_weights={restore_best_weights}")
    
    def __call__(self, val_loss, model=None, epoch=None):
        """
        检查是否应该停止训练
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型（如果restore_best_weights=True）
            epoch: 当前epoch，用于日志记录
            
        Returns:
            bool: 是否应该停止训练
        """
        epoch_str = f"epoch {epoch}: " if epoch is not None else ""
        
        if self.best_loss is None:
            self.best_loss = val_loss
            print(f"{epoch_str}早停: 初始化最佳损失为 {val_loss:.6f}")
            if self.restore_best_weights and model is not None:
                self.best_model = copy.deepcopy(model.state_dict())
                print(f"{epoch_str}早停: 保存初始最佳模型")
            if epoch is not None:
                self.last_improvement = epoch
        # 判断是否有显著改善：当前损失必须比最佳损失小，且差值大于min_delta才算显著改善
        elif self.best_loss - val_loss > self.min_delta:
            # 有显著改善
            improvement = self.best_loss - val_loss
            self.best_loss = val_loss
            print(f"{epoch_str}早停: 损失改善 {improvement:.6f}，新的最佳损失: {val_loss:.6f}，重置计数器")
            # 更新最佳模型
            if self.restore_best_weights and model is not None:
                self.best_model = copy.deepcopy(model.state_dict())
                print(f"{epoch_str}早停: 更新最佳模型")
            self.counter = 0
            if epoch is not None:
                self.last_improvement = epoch
        else:
            self.counter += 1
            epochs_since_improvement = epoch - self.last_improvement if epoch is not None else self.counter
            print(f"{epoch_str}早停: 无显著改善，当前损失: {val_loss:.6f}，最佳损失: {self.best_loss:.6f}，"
                 f"计数器: {self.counter}/{self.patience}，"
                 f"上次改善: {epochs_since_improvement}轮前")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"{epoch_str}早停: 触发早停条件! {self.patience}轮未见改善")
        
        return self.early_stop

class RankingLoss(nn.Module):
    """排序损失函数，用于学习板块的相对排名"""
    
    def __init__(self, margin=0.1, method='pairwise'):
        """
        初始化排序损失
        
        Args:
            margin: 正负样本间的边界
            method: 排序学习方法 ('pairwise', 'listwise')
        """
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.method = method
        
    def forward(self, predictions, targets, target_direction=None):
        """
        计算排序损失
        
        Args:
            predictions: 预测值 [batch_size] 或 单个预测值
            targets: 真实值 [batch_size] 或 单个真实值
            target_direction: 可选的排序方向，用于pairwise模式下计算两个元素间的排序损失
            
        Returns:
            loss: 排序损失
        """
        # 处理单元素pairwise排序的特殊情况
        if target_direction is not None:
            # 这是两个元素间的pairwise损失计算
            # pred_i, pred_j, target_direction
            
            # 确保输入都是标量张量
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions, device=target_direction.device if isinstance(target_direction, torch.Tensor) else None)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, device=predictions.device)
            if not isinstance(target_direction, torch.Tensor):
                target_direction = torch.tensor(target_direction, device=predictions.device)
            
            # 处理不同维度的情况
            if predictions.dim() > 0:
                predictions = predictions.reshape(-1)[0]  # 取第一个元素
            if targets.dim() > 0:
                targets = targets.reshape(-1)[0]  # 取第一个元素
            if target_direction.dim() > 0:
                target_direction = target_direction.reshape(-1)[0]  # 取第一个元素
                
            # 计算边界损失
            loss = F.relu(self.margin - (predictions - targets) * target_direction)
            return loss

        if self.method == 'pairwise':
            # 创建所有可能的对子
            # 确保predictions是张量且维度正确
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions)
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)  # 将标量转换为一维张量
            
            n = predictions.size(0)
            if n <= 1:
                return torch.tensor(0.0, device=predictions.device)
                
            # 扩展为矩阵形式以便比较
            pred_i = predictions.unsqueeze(1).expand(n, n)
            pred_j = predictions.unsqueeze(0).expand(n, n)
            target_i = targets.unsqueeze(1).expand(n, n)
            target_j = targets.unsqueeze(0).expand(n, n)
            
            # 找出有效的对子（目标值有差异）
            valid_pairs = (target_i > target_j).float()
            if valid_pairs.sum() == 0:
                return torch.tensor(0.0, device=predictions.device)
                
            # 计算对子间的损失
            pairwise_loss = F.relu(self.margin - (pred_i - pred_j)) * valid_pairs
            return pairwise_loss.sum() / valid_pairs.sum()
            
        elif self.method == 'listwise':
            # 使用ListNet的顶部概率损失
            # 将预测和目标转换为概率分布
            pred_probs = F.softmax(predictions, dim=0)
            target_probs = F.softmax(targets, dim=0)
            # 计算交叉熵损失
            return F.kl_div(pred_probs.log(), target_probs, reduction='batchmean')
        else:
            raise ValueError(f"Unknown ranking method: {self.method}")

# 二阶段板块轮动模型实现
class SimpleGRUCell(nn.Module):
    """简化版GRU单元，比LiquidTimeCell更轻量化"""
    def __init__(self, input_size, hidden_size):
        super(SimpleGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 重置门和更新门
        self.rz_gate = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        # 候选隐藏状态
        self.n_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.rz_gate.weight)
        nn.init.zeros_(self.rz_gate.bias)
        nn.init.xavier_uniform_(self.n_gate.weight)
        nn.init.zeros_(self.n_gate.bias)
    
    def forward(self, x, h_prev=None):
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, input_size]
            h_prev: 上一时刻隐藏状态 [batch_size, hidden_size]
        
        Returns:
            output: 当前时刻输出 [batch_size, hidden_size]
            h_new: 更新后的隐藏状态 [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态（如果未提供）
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # 连接输入和隐藏状态
        combined = torch.cat((x, h_prev), dim=1)
        
        # 计算重置门和更新门
        rz = torch.sigmoid(self.rz_gate(combined))
        r, z = torch.chunk(rz, 2, dim=1)
        
        # 计算候选隐藏状态
        combined_r = torch.cat((x, r * h_prev), dim=1)
        n = torch.tanh(self.n_gate(combined_r))
        
        # 更新隐藏状态
        h_new = (1 - z) * h_prev + z * n
        
        return h_new, h_new

class SectorClassifierModel(nn.Module):
    """板块涨跌二分类模型
    预测各板块在未来几天内是否会上涨（二分类问题）
    """
    def __init__(self, sector_input_size, market_input_size, hidden_size=32, 
                 num_layers=1, dropout_rate=0.1, use_bn=True):
        super(SectorClassifierModel, self).__init__()
        
        # 配置参数
        self.sector_input_size = sector_input_size
        self.market_input_size = market_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        
        # 板块特征处理
        self.sector_gru = nn.GRU(
            input_size=sector_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 市场特征处理
        self.market_gru = nn.GRU(
            input_size=market_input_size if market_input_size > 0 else 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 批量归一化
        if use_bn:
            self.sector_norm = nn.BatchNorm1d(sector_input_size)
            self.market_norm = nn.BatchNorm1d(market_input_size if market_input_size > 0 else 1)
            self.fusion_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 输出层 - 二分类
        self.classifier = nn.Linear(hidden_size, 1)
    
    def forward(self, sector_features, market_features):
        """前向传播
        
        Args:
            sector_features: 板块特征 [batch_size, seq_len, sector_input_size]
            market_features: 市场特征 [batch_size, seq_len, market_input_size]
            
        Returns:
            upside_prob: 上涨概率 [batch_size, 1]
        """
        batch_size, seq_len, _ = sector_features.shape
        
        # 批量归一化 (需要调整维度)
        if self.use_bn:
            # 重塑以适应BatchNorm1d的输入要求
            sector_features_reshaped = sector_features.reshape(-1, self.sector_input_size)
            sector_features_normed = self.sector_norm(sector_features_reshaped)
            sector_features = sector_features_normed.reshape(batch_size, seq_len, -1)
            
            # 处理市场特征
            if market_features.shape[2] > 0:
                market_features_reshaped = market_features.reshape(-1, market_features.shape[2])
                market_features_normed = self.market_norm(market_features_reshaped)
                market_features = market_features_normed.reshape(batch_size, seq_len, -1)
        
        # 处理板块特征
        sector_out, _ = self.sector_gru(sector_features)
        sector_last = sector_out[:, -1, :]  # 取最后一个时间步
        
        # 处理市场特征
        market_out, _ = self.market_gru(market_features)
        market_last = market_out[:, -1, :]  # 取最后一个时间步
        
        # 融合特征
        combined = torch.cat([sector_last, market_last], dim=1)
        
        # 批量归一化
        if self.use_bn:
            combined = self.fusion_norm(combined)
        
        # 特征融合
        fused = self.fusion(combined)
        
        # 二分类输出
        logits = self.classifier(fused)
        
        # 转换为概率
        upside_prob = torch.sigmoid(logits)
        
        # 确保概率值严格在0-1范围内
        upside_prob = torch.clamp(upside_prob, 0.0, 1.0)
        
        return upside_prob

class SectorRankingModel(nn.Module):
    """板块排序模型
    基于板块特征和上涨概率预测，预测板块的相对收益排名
    """
    def __init__(self, sector_input_size, market_input_size, hidden_size=32, 
                 num_layers=1, dropout_rate=0.1, use_classifier_output=True,
                 use_bn=True, ranking_mode='pairwise'):
        super(SectorRankingModel, self).__init__()
        
        # 配置参数
        self.sector_input_size = sector_input_size
        self.market_input_size = market_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_classifier_output = use_classifier_output
        self.use_bn = use_bn
        self.ranking_mode = ranking_mode  # 'pairwise' 或 'pointwise'
        
        # 板块特征处理
        # 如果使用分类器输出，增加输入维度
        sector_dim = sector_input_size + (1 if use_classifier_output else 0)
        
        self.sector_gru = nn.GRU(
            input_size=sector_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 市场特征处理
        self.market_gru = nn.GRU(
            input_size=market_input_size if market_input_size > 0 else 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 批量归一化
        if use_bn:
            self.sector_norm = nn.BatchNorm1d(sector_dim)
            self.market_norm = nn.BatchNorm1d(market_input_size if market_input_size > 0 else 1)
            self.fusion_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 输出层 - 排序分数
        self.ranking = nn.Linear(hidden_size, 1)
    
    def forward(self, sector_features, market_features, classifier_output=None):
        """前向传播
        
        Args:
            sector_features: 板块特征 [batch_size, seq_len, sector_input_size]
            market_features: 市场特征 [batch_size, seq_len, market_input_size]
            classifier_output: 分类器输出的上涨概率 [batch_size, 1]
            
        Returns:
            ranking_score: 排名得分 [batch_size, 1]
        """
        batch_size, seq_len, _ = sector_features.shape
        
        # 如果有分类器输出，添加到板块特征
        if self.use_classifier_output and classifier_output is not None:
            # 扩展分类器输出维度以匹配序列长度
            expanded = classifier_output.unsqueeze(1).expand(-1, seq_len, -1)
            sector_features = torch.cat([sector_features, expanded], dim=2)
        
        # 批量归一化 (需要调整维度)
        if self.use_bn:
            # 重塑以适应BatchNorm1d的输入要求
            sector_features_reshaped = sector_features.reshape(-1, sector_features.shape[2])
            sector_features_normed = self.sector_norm(sector_features_reshaped)
            sector_features = sector_features_normed.reshape(batch_size, seq_len, -1)
            
            # 处理市场特征
            if market_features.shape[2] > 0:
                market_features_reshaped = market_features.reshape(-1, market_features.shape[2])
                market_features_normed = self.market_norm(market_features_reshaped)
                market_features = market_features_normed.reshape(batch_size, seq_len, -1)
        
        # 处理板块特征
        sector_out, _ = self.sector_gru(sector_features)
        
        # 使用注意力机制聚合序列信息
        sector_out = sector_out.transpose(0, 1)  # [seq_len, batch_size, hidden]
        attn_out, _ = self.attention(sector_out, sector_out, sector_out)
        attn_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, hidden]
        
        # 取加权后的最后一个时间步
        sector_last = attn_out[:, -1, :]
        
        # 处理市场特征
        market_out, _ = self.market_gru(market_features)
        market_last = market_out[:, -1, :]
        
        # 融合特征
        combined = torch.cat([sector_last, market_last], dim=1)
        
        # 批量归一化
        if self.use_bn:
            combined = self.fusion_norm(combined)
        
        # 特征融合
        fused = self.fusion(combined)
        
        # 排序得分输出
        ranking_score = self.ranking(fused)
        
        return ranking_score

class TwoStageSectorRotationModel(nn.Module):
    """
    二阶段板块轮动模型 - 专用于处理多尺度特征
    
    第一阶段：预测板块上涨概率
    第二阶段：预测板块排名得分
    最终综合两个阶段的结果
    """
    
    def __init__(self, sector_input_size, market_input_size, hidden_size=32,
                 num_layers=1, dropout_rate=0.1, use_bn=True, fusion_mode='dynamic',
                 device='cpu'):
        """
        初始化二阶段板块轮动模型
        
        Args:
            sector_input_size: 板块特征维度
            market_input_size: 市场特征维度
            hidden_size: 隐层大小
            num_layers: LSTM层数
            dropout_rate: Dropout比率
            use_bn: 是否使用BatchNorm
            fusion_mode: 特征融合模式 ('concat', 'attention', 'dynamic')
            device: 计算设备
        """
        super(TwoStageSectorRotationModel, self).__init__()
        
        self.sector_input_size = sector_input_size
        self.market_input_size = market_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fusion_mode = fusion_mode
        self.device = device
        
        # 短期、中期和长期编码器
        self.short_term_sector_encoder = nn.LSTM(
            input_size=sector_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.mid_term_sector_encoder = nn.LSTM(
            input_size=sector_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.long_term_sector_encoder = nn.LSTM(
            input_size=sector_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 短期、中期和长期市场编码器
        self.short_term_market_encoder = nn.LSTM(
            input_size=market_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.mid_term_market_encoder = nn.LSTM(
            input_size=market_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.long_term_market_encoder = nn.LSTM(
            input_size=market_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 多尺度融合层 - 注意力机制
        self.attention_heads = 4
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=self.attention_heads, 
            batch_first=True
        )
        
        # 时间尺度权重学习
        self.scale_importance = nn.Parameter(torch.ones(3) / 3)
        
        # 特征融合层
        if fusion_mode == 'concat':
            fusion_input_size = hidden_size * 2  # 板块特征和市场特征拼接
        elif fusion_mode == 'attention' or fusion_mode == 'dynamic':
            fusion_input_size = hidden_size
            
            # 板块-市场特征动态融合
            self.feature_attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        else:
            fusion_input_size = hidden_size  # 默认使用加和融合
        
        # 第一阶段：分类器 - 预测上涨概率
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 第二阶段：排名模型
        self.ranker = nn.Sequential(
            nn.Linear(fusion_input_size + 1, hidden_size),  # +1 是分类器输出
            nn.BatchNorm1d(hidden_size) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 输出融合层
        self.output_fusion = nn.Sequential(
            nn.Linear(2, 1)  # 融合分类和排名结果
        )
        
        # 初始化融合层权重，使其开始时平等考虑分类和排名结果
        self.output_fusion[0].weight.data = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        self.output_fusion[0].bias.data.zero_()
        
        # 损失函数
        self.classifier_loss_fn = nn.BCELoss()
        self.ranking_loss_fn = RankingLoss(margin=0.1, method='pairwise')
        self.regression_loss_fn = nn.MSELoss()
    
    def process_multi_scale_features(self, sector_features, market_features):
        """
        处理多尺度特征输入
        
        Args:
            sector_features: 字典形式的板块多尺度特征, 如{'short_term': tensor, 'mid_term': tensor, 'long_term': tensor}
            market_features: 字典形式的市场多尺度特征, 如{'short_term': tensor, 'mid_term': tensor, 'long_term': tensor}
            
        Returns:
            sector_encoding: 板块特征编码
            market_encoding: 市场特征编码
        """
        # 检查输入是否是字典格式（多尺度特征）
        if not isinstance(sector_features, dict) or not isinstance(market_features, dict):
            raise ValueError("多尺度特征输入必须是字典格式, 包含'short_term', 'mid_term', 'long_term'键")
        
        # 检查必要的键是否存在
        required_scales = ['short_term', 'mid_term', 'long_term']
        for scale in required_scales:
            if scale not in sector_features or scale not in market_features:
                raise ValueError(f"缺少必要的尺度'{scale}'")
        
        # 处理每个尺度的特征
        sector_encodings = {}
        market_encodings = {}
        
        # 1. 短期特征处理
        short_term_sector = sector_features['short_term']
        short_term_market = market_features['short_term']
        
        # 将张量转移到正确的设备上
        if isinstance(short_term_sector, torch.Tensor):
            short_term_sector = short_term_sector.to(self.device)
            short_term_market = short_term_market.to(self.device)
            
        # 确保张量类型是float
        short_term_sector = short_term_sector.float()
        short_term_market = short_term_market.float()
        
        # 使用LSTM编码板块和市场短期特征
        _, (short_sector_h_n, _) = self.short_term_sector_encoder(short_term_sector)
        _, (short_market_h_n, _) = self.short_term_market_encoder(short_term_market)
        
        # 取最后一层的hidden state作为编码
        sector_encodings['short_term'] = short_sector_h_n[-1]
        market_encodings['short_term'] = short_market_h_n[-1]
        
        # 2. 中期特征处理
        mid_term_sector = sector_features['mid_term']
        mid_term_market = market_features['mid_term']
        
        # 将张量转移到正确的设备上
        if isinstance(mid_term_sector, torch.Tensor):
            mid_term_sector = mid_term_sector.to(self.device)
            mid_term_market = mid_term_market.to(self.device)
            
        # 确保张量类型是float
        mid_term_sector = mid_term_sector.float()
        mid_term_market = mid_term_market.float()
        
        # 使用LSTM编码板块和市场中期特征
        _, (mid_sector_h_n, _) = self.mid_term_sector_encoder(mid_term_sector)
        _, (mid_market_h_n, _) = self.mid_term_market_encoder(mid_term_market)
        
        # 取最后一层的hidden state作为编码
        sector_encodings['mid_term'] = mid_sector_h_n[-1]
        market_encodings['mid_term'] = mid_market_h_n[-1]
        
        # 3. 长期特征处理
        long_term_sector = sector_features['long_term']
        long_term_market = market_features['long_term']
        
        # 将张量转移到正确的设备上
        if isinstance(long_term_sector, torch.Tensor):
            long_term_sector = long_term_sector.to(self.device)
            long_term_market = long_term_market.to(self.device)
            
        # 确保张量类型是float
        long_term_sector = long_term_sector.float()
        long_term_market = long_term_market.float()
        
        # 使用LSTM编码板块和市场长期特征
        _, (long_sector_h_n, _) = self.long_term_sector_encoder(long_term_sector)
        _, (long_market_h_n, _) = self.long_term_market_encoder(long_term_market)
        
        # 取最后一层的hidden state作为编码
        sector_encodings['long_term'] = long_sector_h_n[-1]
        market_encodings['long_term'] = long_market_h_n[-1]
        
        # 4. 多尺度特征融合 - 使用注意力机制整合不同尺度的信息
        
        # 为板块特征创建融合输入张量
        sector_stack = torch.stack([
            sector_encodings['short_term'],
            sector_encodings['mid_term'],
            sector_encodings['long_term']
        ], dim=1)  # 形状: [batch_size, 3, hidden_size]
        
        # 为市场特征创建融合输入张量
        market_stack = torch.stack([
            market_encodings['short_term'],
            market_encodings['mid_term'],
            market_encodings['long_term']
        ], dim=1)  # 形状: [batch_size, 3, hidden_size]
        
        # 使用多头注意力机制融合不同尺度的特征
        sector_attn, _ = self.scale_attention(sector_stack, sector_stack, sector_stack)
        market_attn, _ = self.scale_attention(market_stack, market_stack, market_stack)
        
        # 尺度重要性加权 - 学习不同尺度的重要性
        scale_weights = torch.softmax(self.scale_importance, dim=0)
        
        # 加权融合不同尺度的特征
        sector_encoding = torch.sum(sector_attn * scale_weights.view(1, 3, 1), dim=1)
        market_encoding = torch.sum(market_attn * scale_weights.view(1, 3, 1), dim=1)
        
        return sector_encoding, market_encoding
    
    def forward(self, sector_features, market_features, market_state=None, training=False):
        """
        前向传播，处理板块特征和市场特征
        
        Args:
            sector_features: 板块特征，多尺度格式
            market_features: 市场特征，多尺度格式
            market_state: 市场状态，可选
            training: 是否处于训练模式
            
        Returns:
            tuple: (最终得分, 上涨概率, 排名得分)
        """
        # 1. 处理多尺度特征
        sector_encoding, market_encoding = self.process_multi_scale_features(sector_features, market_features)
        
        # 2. 特征融合
        if self.fusion_mode == 'concat':
            # 简单拼接板块和市场特征
            fusion = torch.cat([sector_encoding, market_encoding], dim=-1)
        elif self.fusion_mode == 'attention' or self.fusion_mode == 'dynamic':
            # 使用注意力机制学习板块和市场特征的重要性
            concat_features = torch.cat([sector_encoding, market_encoding], dim=-1)
            attention_scores = torch.sigmoid(self.feature_attention(concat_features))
            
            # 动态融合
            fusion = attention_scores * sector_encoding + (1 - attention_scores) * market_encoding
        else:
            # 简单加和融合
            fusion = sector_encoding + market_encoding
        
        # 3. 第一阶段：分类，预测上涨概率
        upside_prob = self.classifier(fusion)
        
        # 4. 第二阶段：排名，预测相对排名得分
        # 将第一阶段的预测结果与特征融合
        ranking_input = torch.cat([fusion, upside_prob], dim=-1)
        ranking_score = self.ranker(ranking_input)
        
        # 5. 融合两阶段结果
        final_score = self.output_fusion(torch.cat([upside_prob, ranking_score], dim=-1))
        
        return final_score, upside_prob, ranking_score
    
    def compute_loss(self, outputs, targets):
        """计算损失
        
        Args:
            outputs: 模型输出 (final_scores, upside_probs, ranking_scores)
            targets: 目标值 [batch_size, 3]，包含三个目标
                    - targets[:, 0]: 收益率
                    - targets[:, 1]: 是否上涨(0/1)
                    - targets[:, 2]: 风险调整收益
                    
        Returns:
            dict: 损失字典，包含各损失组件和总损失
        """
        try:
            # 解包模型输出
            final_scores, upside_probs, ranking_scores = outputs
            
            # 确保upside_probs严格在0-1范围内
            upside_probs = torch.clamp(upside_probs, 0.0, 1.0)
            
            # 解包目标
            if targets.dim() < 2 or targets.shape[1] < 3:
                # 如果目标维度不足，扩展为兼容格式
                if targets.dim() < 2:
                    targets = targets.view(-1, 1)
                
                # 自动生成上涨标签
                upside_targets = (targets[:, 0] > 0).float().view(-1, 1)
                
                # 扩展为3列
                if targets.shape[1] == 1:
                    targets = torch.cat([
                        targets,  # 收益率
                        upside_targets,  # 上涨概率
                        targets   # 风险调整收益（暂用收益率代替）
                    ], dim=1)
            
            # 提取具体目标
            returns_targets = targets[:, 0].view(-1, 1)  # 收益率
            upside_targets = targets[:, 1].view(-1, 1)   # 上涨概率
            risk_adj_targets = targets[:, 2].view(-1, 1) # 风险调整收益
            
            # 确保upside_targets也在0-1范围内
            upside_targets = torch.clamp(upside_targets, 0.0, 1.0)
            
            # 计算分类损失（二分类交叉熵）
            classification_loss = self.classifier_loss_fn(upside_probs, upside_targets)
            
            # 计算排序损失
            ranking_loss = self.ranking_loss_fn(ranking_scores, returns_targets)
            
            # 计算配对排序损失（用于增强排序效果）
            batch_size = targets.shape[0]
            pairwise_loss = torch.tensor(0.0, device=returns_targets.device)
            
            # 如果批量大小大于1，计算配对损失
            if batch_size > 1:
                try:
                    # 矢量化实现，避免低效的双重循环
                    # 最多使用1000个配对，避免在大批量时计算过多配对
                    max_pairs = 1000
                    
                    # 构建所有可能的配对索引
                    i_indices = []
                    j_indices = []
                    valid_diffs = []
                    
                    # 使用所有配对
                    for i in range(batch_size-1):
                        for j in range(i+1, batch_size):
                            diff = abs(returns_targets[i].item() - returns_targets[j].item())
                            if diff > 0.01:  # 仅考虑收益率有明确差异的对
                                i_indices.append(i)
                                j_indices.append(j)
                                valid_diffs.append(returns_targets[i].item() - returns_targets[j].item())
                                # 限制最大配对数量，防止内存溢出
                                if len(i_indices) >= max_pairs:
                                    break
                        # 如果已经达到最大配对数量，跳出外层循环
                        if len(i_indices) >= max_pairs:
                            break
                    
                    # 如果有有效配对，计算损失
                    pair_count = len(i_indices)
                    if pair_count > 0:
                        # 批量获取分数和目标
                        i_scores = ranking_scores[i_indices]
                        j_scores = ranking_scores[j_indices]
                        # 创建target张量并确保维度匹配 - 关键修复
                        targets_tensor = torch.tensor(valid_diffs, device=returns_targets.device).sign()
                        
                        # 确保维度匹配 - pairwise_ranking_criterion要求i_scores和j_scores是一维张量
                        # 而targets_tensor也是一维张量
                        if i_scores.dim() > 1:
                            i_scores = i_scores.view(-1)
                        if j_scores.dim() > 1:
                            j_scores = j_scores.view(-1)
                        # 确保targets_tensor也是一维的
                        if targets_tensor.dim() > 1:
                            targets_tensor = targets_tensor.view(-1)
                        
                        # 批量计算损失
                        pairwise_loss = self.ranking_loss_fn(i_scores, j_scores, targets_tensor)
                except Exception as e:
                    # 如果配对损失计算失败，记录错误并返回零损失
                    print(f"配对损失计算出错，使用0值替代: {e}")
                    pairwise_loss = torch.tensor(0.0, device=returns_targets.device)
            
            # 计算总损失
            total_loss = (
                0.4 * classification_loss + 
                0.3 * ranking_loss + 
                0.3 * pairwise_loss
            )
            
            # 构建损失字典
            loss_dict = {
                'total_loss': total_loss,
                'classification_loss': classification_loss,
                'ranking_loss': ranking_loss,
                'pairwise_loss': pairwise_loss
            }
            
            return loss_dict
            
        except Exception as e:
            # 捕获所有异常，确保训练不会中断
            print(f"损失计算出错: {e}")
            import traceback
            print(traceback.format_exc())
            
            # 返回默认损失值
            device = next(self.parameters()).device
            default_loss = torch.tensor(1.0, device=device)
            return {
                'total_loss': default_loss,
                'classification_loss': default_loss * 0.4,
                'ranking_loss': default_loss * 0.3,
                'pairwise_loss': default_loss * 0.3
            }
    
    def save(self, filepath):
        """保存模型
        
        Args:
            filepath: 模型保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.state_dict(),
            'classifier_weight': self.classifier_weight,
            'ranker_weight': self.ranker_weight,
            'sector_input_size': self.sector_input_size,
            'market_input_size': self.market_input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'use_bn': self.use_bn,
            'fusion_mode': self.fusion_mode,
            'use_multi_scale': self.use_multi_scale,
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath, device='cpu'):
        """加载模型
        
        Args:
            filepath: 模型文件路径
            device: 计算设备
            
        Returns:
            加载的模型
        """
        # 加载保存的状态
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 创建模型
        model = cls(
            sector_input_size=checkpoint['sector_input_size'],
            market_input_size=checkpoint['market_input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout_rate=checkpoint['dropout_rate'],
            use_bn=checkpoint['use_bn'],
            fusion_mode=checkpoint['fusion_mode'],
            use_multi_scale=checkpoint['use_multi_scale'],
            device=device
        )
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复其他属性
        model.classifier_weight = checkpoint['classifier_weight'] 
        model.ranker_weight = checkpoint['ranker_weight']
        model.is_fitted = checkpoint['is_fitted']
        
        return model
    
    def fit(self, sector_features_data, market_features_data, y, batch_size=32, epochs=100, 
            validation_data=None, early_stopping=True, patience=10, min_delta=0.001, 
            verbose=1, learning_rate=0.001, weight_decay=0.0001, clip_grad_norm=1.0,
            use_amp=False, gradient_accumulation_steps=4, validate_every=2, prefetch_factor=2,
            val_batch_size=None, val_workers=None, resume_from_checkpoint=None, save_best_only=True,
            save_path=None, save_every=10):
        """训练模型
        
        Args:
            sector_features_data: 板块特征数据 [batch_size, seq_len, sector_input_size]
            market_features_data: 市场特征数据 [batch_size, seq_len, market_input_size]
            y: 目标值 [batch_size, output_dims]，如果是1D数组将自动扩展
            batch_size: 批次大小
            epochs: 训练轮数
            validation_data: 验证数据元组 (X_val_sector, X_val_market, y_val)
            early_stopping: 是否使用早停
            patience: 早停耐心值
            min_delta: 早停最小变化量
            verbose: 输出详细程度 (0=安静, 1=进度, 2=详细)
            learning_rate: 学习率
            weight_decay: 权重衰减
            clip_grad_norm: 梯度裁剪范数
            use_amp: 是否使用自动混合精度
            gradient_accumulation_steps: 梯度累积步数
            validate_every: 每多少个epoch验证一次
            prefetch_factor: 数据加载预取因子
            val_batch_size: 验证批次大小，默认等于训练批次大小
            val_workers: 验证数据加载器的工作线程数
            resume_from_checkpoint: 从检查点恢复训练，提供模型文件路径
            save_best_only: 是否只保存最佳模型
            save_path: 模型保存路径
            save_every: 每多少个epoch保存一次模型

        Returns:
            训练历史记录
        """
        import gc
        import traceback
        from torch.utils.data import DataLoader, TensorDataset
        from torch.cuda.amp import autocast, GradScaler
        
        # 设备检查
        device = next(self.parameters()).device
        
        # 初始化历史记录
        if not hasattr(self, 'history') or self.history is None:
            self.history = {
                'loss': [], 'val_loss': [],
                'classification_loss': [], 'val_classification_loss': [],
                'ranking_loss': [], 'val_ranking_loss': [],
                'pairwise_loss': [], 'val_pairwise_loss': []
            }
        
        # 当前epoch，用于从检查点恢复时
        current_epoch = 0
        best_loss = float('inf')
        
        # 从检查点恢复训练
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            print(f"恢复训练自检查点: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 新格式检查点
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if 'epoch' in checkpoint:
                    current_epoch = checkpoint['epoch']
                    print(f"从epoch {current_epoch} 继续训练")
                if 'best_loss' in checkpoint:
                    best_loss = checkpoint['best_loss']
                if 'train_history' in checkpoint:
                    self.history = checkpoint['train_history']
            else:
                # 旧格式，直接是状态字典
                self.load_state_dict(checkpoint, strict=False)
                
            print(f"模型状态已恢复，准备继续训练")
        
        # 准备数据
        self._prepare_fit_data(sector_features_data, market_features_data, y)
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(
            torch.tensor(sector_features_data, dtype=torch.float32),
            torch.tensor(market_features_data, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            num_workers=4
        )
        
        # 初始化验证数据加载器
        val_loader = None
        if validation_data is not None:
            X_val_sector, X_val_market, y_val = validation_data
            val_batch = val_batch_size if val_batch_size is not None else batch_size
            val_workers_count = val_workers if val_workers is not None else 2
            
            val_dataset = TensorDataset(
                torch.tensor(X_val_sector, dtype=torch.float32),
                torch.tensor(X_val_market, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                num_workers=val_workers_count
            )
        
        # 定义优化器和学习率调度器
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            verbose=True
        )
        
        # 初始化早停类
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None
        
        # 自动混合精度
        scaler = GradScaler() if use_amp else None
        
        # 训练循环
        total_batches = len(train_loader)
        
        # 确保保存目录存在
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 从当前_epoch开始训练
        for epoch in range(current_epoch, current_epoch + epochs):
            start_time = time.time()
            self.train()
            
            running_loss = 0.0
            running_class_loss = 0.0
            running_rank_loss = 0.0
            running_pairwise_loss = 0.0
            
            # 动态更新损失权重
            market_features_batch = torch.tensor(market_features_data[-1:], dtype=torch.float32).to(device)
            self.update_loss_weights(market_features_batch)
            
            # 动态更新超参数
            if hasattr(self, 'dynamic_hyperparams'):
                self.dynamic_hyperparams.update_params(market_features_batch, epoch)
                
            if verbose > 0:
                print(f"\nepoch {epoch+1}/{current_epoch+epochs}")
                print(f"total batches: {total_batches}")
                
            # 梯度累积计数器
            accum_steps = 0
            
            for batch_idx, (sector_batch, market_batch, targets) in enumerate(train_loader):
                batch_start = time.time()
                
                # 将数据移至设备
                sector_batch = sector_batch.to(device)
                market_batch = market_batch.to(device)
                targets = targets.to(device)
                
                # 前向传播（可选的混合精度）
                with autocast() if use_amp else nullcontext():
                    outputs = self(sector_batch, market_batch, training=True)
                    loss_dict = self.compute_loss(outputs, targets)
                    loss = loss_dict['total_loss'] / gradient_accumulation_steps
                
                # 反向传播（可选的混合精度）
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accum_steps += 1
                
                # 梯度累积
                if accum_steps == gradient_accumulation_steps or batch_idx == len(train_loader) - 1:
                    # 梯度裁剪
                    if clip_grad_norm > 0:
                        if use_amp:
                            scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                    
                    # 参数更新（可选的混合精度）
                    if use_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    accum_steps = 0
                
                # 记录损失
                running_loss += loss_dict['total_loss'].item() * gradient_accumulation_steps
                running_class_loss += loss_dict['classification_loss'].item()
                running_rank_loss += loss_dict['ranking_loss'].item()
                running_pairwise_loss += loss_dict['pairwise_loss'].item()
                
                # 输出进度
                if verbose > 0 and (batch_idx == 0 or (batch_idx + 1) % 100 == 0 or batch_idx == len(train_loader) - 1):
                    batch_time = time.time() - batch_start
                    avg_loss = running_loss / (batch_idx + 1)
                    print(f"{epoch+1} [{batch_idx+1}/{total_batches}]\t"
                          f"time {batch_time:.3f} ({batch_time:.3f})\t"
                          f"loss {loss_dict['total_loss'].item():.4f} ({avg_loss:.4f})")
            
            # 计算epoch平均损失
            epoch_loss = running_loss / len(train_loader)
            epoch_class_loss = running_class_loss / len(train_loader)
            epoch_rank_loss = running_rank_loss / len(train_loader)
            epoch_pairwise_loss = running_pairwise_loss / len(train_loader)
            
            # 记录训练损失
            self.history['loss'].append(epoch_loss)
            self.history['classification_loss'].append(epoch_class_loss)
            self.history['ranking_loss'].append(epoch_rank_loss)
            self.history['pairwise_loss'].append(epoch_pairwise_loss)
            
            epoch_time = time.time() - start_time
            
            # 验证
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                try:
                    self.eval()
                    val_start = time.time()
                    
                    # 使用累积损失代替收集所有输出
                    val_loss_sum = 0.0
                    val_class_loss_sum = 0.0
                    val_rank_loss_sum = 0.0
                    val_pairwise_loss_sum = 0.0
                    val_samples = 0
                    
                    with torch.no_grad():
                        for val_sector_batch, val_market_batch, val_targets in val_loader:
                            try:
                                # 移至设备
                                val_sector_batch = val_sector_batch.to(device)
                                val_market_batch = val_market_batch.to(device)
                                val_targets = val_targets.to(device)
                                
                                # 前向传播
                                val_outputs = self(val_sector_batch, val_market_batch, training=False)
                                
                                # 计算当前批次损失
                                batch_loss_dict = self.compute_loss(val_outputs, val_targets)
                                
                                # 累加损失
                                val_loss_sum += batch_loss_dict['total_loss'].item() * len(val_targets)
                                val_class_loss_sum += batch_loss_dict['classification_loss'].item() * len(val_targets)
                                val_rank_loss_sum += batch_loss_dict['ranking_loss'].item() * len(val_targets)
                                val_pairwise_loss_sum += batch_loss_dict['pairwise_loss'].item() * len(val_targets)
                                val_samples += len(val_targets)
                                
                            except Exception as e:
                                print(f"验证批次处理出错: {e}")
                                # 继续处理下一批次，不中断验证
                    
                    val_time = time.time() - val_start
                    
                    # 计算平均损失
                    if val_samples > 0:
                        try:
                            # 计算平均损失
                            avg_val_loss = val_loss_sum / val_samples
                            avg_val_class_loss = val_class_loss_sum / val_samples
                            avg_val_rank_loss = val_rank_loss_sum / val_samples
                            avg_val_pairwise_loss = val_pairwise_loss_sum / val_samples
                            
                            # 创建损失字典
                            val_losses = {
                                'total_loss': avg_val_loss,
                                'classification_loss': avg_val_class_loss,
                                'ranking_loss': avg_val_rank_loss,
                                'pairwise_loss': avg_val_pairwise_loss
                            }
                            
                            if verbose > 0:
                                print(f"验证完成，处理 {val_samples} 个样本，用时: {val_time:.2f} 秒")
                            
                            # 记录验证损失
                            self.history['val_loss'].append(val_losses['total_loss'])
                            self.history['val_classification_loss'].append(val_losses['classification_loss'])
                            self.history['val_ranking_loss'].append(val_losses['ranking_loss'])
                            self.history['val_pairwise_loss'].append(val_losses['pairwise_loss'])
                            
                            # 更新学习率
                            current_loss = val_losses['total_loss']
                            lr_scheduler.step(current_loss)
                                
                            # 检查早停
                            if early_stopping:
                                early_stop = early_stopper(current_loss, self, epoch+1)
                                
                                # 保存最佳模型
                                if current_loss < best_loss and save_path:
                                    best_loss = current_loss
                                    if save_best_only:
                                        self.save_model(
                                            save_path, 
                                            optimizer=self.optimizer,
                                            epoch=epoch+1,
                                            best_loss=best_loss
                                        )
                                        if verbose > 0:
                                            print(f"保存新的最佳模型，损失: {best_loss:.6f}")
                                
                                if early_stop:
                                    print(f"早停触发于epoch {epoch+1}")
                                    if early_stopper.restore_best_weights and early_stopper.best_model is not None:
                                        self.load_state_dict(early_stopper.best_model)
                                        print("已恢复最佳权重")
                                    break
                        
                            # 输出验证结果
                            if verbose > 0:
                                print(f"验证损失: {val_losses['total_loss']:.4f}, "
                                      f"分类损失: {val_losses['classification_loss']:.4f}, "
                                      f"排序损失: {val_losses['ranking_loss']:.4f}, "
                                      f"配对损失: {val_losses['pairwise_loss']:.4f}")
                            
                        except Exception as e:
                            print(f"验证阶段出错: {str(e)}")
                            print(traceback.format_exc())
                    else:
                        print(f"警告: 验证集结果为空")
                        
                except Exception as e:
                    print(f"验证过程出现错误: {str(e)}")
                    print(traceback.format_exc())
            
            # 定期保存模型检查点
            if save_path and not save_best_only and (epoch + 1) % save_every == 0:
                checkpoint_path = save_path.replace('.pt', f'_epoch{epoch+1}.pt')
                self.save_model(
                    checkpoint_path,
                    optimizer=self.optimizer,
                    epoch=epoch+1,
                    best_loss=best_loss
                )
                if verbose > 0:
                    print(f"保存epoch {epoch+1} 检查点到 {checkpoint_path}")
            
            # 输出epoch摘要
            if verbose > 0:
                print(f"epoch {epoch+1} 完成，用时 {epoch_time:.2f} 秒, "
                      f"训练损失: {epoch_loss:.4f}")
                
                # 清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 如果早停被触发，跳出训练循环
            if early_stopping and early_stopper.early_stop:
                break
        
        # 保存最终模型
        if save_path and not save_best_only:
            self.save_model(
                save_path,
                optimizer=self.optimizer,
                epoch=current_epoch+epochs,
                best_loss=best_loss
            )
            
        return self.history
        
    def _prepare_fit_data(self, sector_features_data, market_features_data, y):
        """准备训练数据

        Args:
            sector_features_data: 板块特征数据
            market_features_data: 市场特征数据
            y: 目标值
        """
        # 确保是numpy数组
        if isinstance(sector_features_data, torch.Tensor):
            sector_features_data = sector_features_data.cpu().numpy()
        if isinstance(market_features_data, torch.Tensor):
            market_features_data = market_features_data.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        # 检查数据是否包含无效值
        if np.isnan(sector_features_data).any() or np.isinf(sector_features_data).any():
            print("警告：板块特征数据包含NaN或Inf值，将替换为0")
            sector_features_data = np.nan_to_num(sector_features_data, nan=0.0, posinf=0.0, neginf=0.0)
            
        if np.isnan(market_features_data).any() or np.isinf(market_features_data).any():
            print("警告：市场特征数据包含NaN或Inf值，将替换为0")
            market_features_data = np.nan_to_num(market_features_data, nan=0.0, posinf=0.0, neginf=0.0)
            
        if np.isnan(y).any() or np.isinf(y).any():
            print("警告：目标值包含NaN或Inf值，将替换为0")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
        # 处理目标值维度
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # 自动扩展为多目标格式
        if y.shape[1] < 3:
            # 添加上涨概率特征和风险调整收益
            upside_prob = (y[:, 0] > 0).astype(float).reshape(-1, 1)
            
            if y.shape[1] == 1:
                # 扩展为 [收益率, 上涨概率, 风险调整收益]
                y = np.concatenate([y, upside_prob, y.copy()], axis=1)
            elif y.shape[1] == 2:
                # 已有2列，添加第3列
                y = np.concatenate([y, y[:, 0].reshape(-1, 1)], axis=1)
                
        # 确保二分类标签在0-1范围内
        y[:, 1] = np.clip(y[:, 1], 0.0, 1.0)
        
        return sector_features_data, market_features_data, y

    def _load_two_stage_model(self, model_name):
        """加载两阶段模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 包含模型和元数据的字典，如果加载失败返回None
        """
        start_time = time.time()
        
        try:
            logger.info(f"正在加载模型 {model_name}...")
            models_dir = os.path.join(self.base_path, 'models', 'sector_models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
                
            model_path = os.path.join(models_dir, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                logger.error(f"找不到模型文件：{model_path}")
                return None
                
            # 尝试兼容加载模型
            try:
                # 先尝试直接加载
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
                
                # 验证模型包是否包含必要的键
                required_keys = ['model', 'model_params', 'feature_columns', 'dates', 'model_type']
                if not all(key in model_package for key in required_keys):
                    missing_keys = [key for key in required_keys if key not in model_package]
                    logger.warning(f"模型包缺少必要的键: {missing_keys}")
                
                # 验证模型状态
                if 'model' in model_package and hasattr(model_package['model'], 'state_dict'):
                    logger.info(f"模型加载成功，用时 {time.time() - start_time:.2f} 秒")
                    return model_package
                else:
                    logger.error("模型格式不正确，需要重新创建模型实例")
                    
                    # 如果模型状态不可用，尝试只加载权重
                    # 此处可以添加更复杂的修复逻辑
                    
            except Exception as e:
                logger.error(f"加载模型时出错: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"加载模型 {model_name} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def load_model(self, model_path, map_location=None):
        """加载保存的模型

        Args:
            model_path: 模型保存路径
            map_location: 模型加载的设备映射，默认为None

        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False

            # 加载模型状态字典
            checkpoint = torch.load(model_path, map_location=map_location)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 新格式，包含更多元数据
                state_dict = checkpoint['model_state_dict']
                
                # 如果有优化器状态，也可以恢复
                if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        logger.warning(f"无法加载优化器状态: {str(e)}")
                
                # 恢复其他元数据
                if 'epoch' in checkpoint:
                    self.current_epoch = checkpoint['epoch']
                if 'best_loss' in checkpoint:
                    self.best_loss = checkpoint['best_loss']
                if 'train_history' in checkpoint:
                    self.history = checkpoint['train_history']
                    
            else:
                # 旧格式，直接是状态字典
                state_dict = checkpoint
            
            # 检查模型结构兼容性
            model_dict = self.state_dict()
            # 过滤不匹配的键
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            # 检查是否有缺失的键
            missing_keys = set(model_dict.keys()) - set(filtered_state_dict.keys())
            if missing_keys:
                logger.warning(f"模型加载：缺少键 {len(missing_keys)}/{len(model_dict)}")
                if len(missing_keys) < 10:
                    logger.warning(f"缺少的键: {missing_keys}")
            
            # 检查形状不匹配的键
            incompatible_keys = []
            for k, v in filtered_state_dict.items():
                if v.shape != model_dict[k].shape:
                    incompatible_keys.append(k)
                    logger.warning(f"形状不匹配: {k}, 加载的形状: {v.shape}, 当前模型形状: {model_dict[k].shape}")
            
            if incompatible_keys:
                for k in incompatible_keys:
                    filtered_state_dict.pop(k)
                logger.warning(f"由于形状不匹配，跳过 {len(incompatible_keys)} 个参数")
            
            # 加载筛选后的状态字典
            self.load_state_dict(filtered_state_dict, strict=False)
            
            logger.info(f"模型加载成功，加载了 {len(filtered_state_dict)}/{len(model_dict)} 个参数")
            return True
            
        except Exception as e:
            logger.error(f"加载模型出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def save_model(self, model_path, optimizer=None, epoch=None, best_loss=None):
        """保存模型及其元数据

        Args:
            model_path: 模型保存路径
            optimizer: 优化器，默认为None
            epoch: 当前轮数，默认为None
            best_loss: 最佳损失，默认为None

        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            
            # 准备保存数据
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'sector_input_size': self.sector_input_size,
                    'market_input_size': self.market_input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'dropout_rate': self.dropout_rate,
                    'use_bn': self.use_bn,
                    'fusion_mode': self.fusion_mode,
                    'use_multi_scale': self.use_multi_scale
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 添加额外数据
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if epoch is not None:
                checkpoint['epoch'] = epoch
            if best_loss is not None:
                checkpoint['best_loss'] = best_loss
            if hasattr(self, 'history') and self.history:
                checkpoint['train_history'] = self.history
                
            # 保存
            torch.save(checkpoint, model_path)
            logger.info(f"模型已保存到 {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def predict(self, sector_features, market_features, market_state=None, return_all=False):
        """
        预测板块表现得分
        
        Args:
            sector_features: 板块特征，多尺度字典格式
                        {'short_term': tensor, 'mid_term': tensor, 'long_term': tensor}
            market_features: 市场特征，字典格式或张量
            market_state: 可选的市场状态，用于动态调整输出
            return_all: 是否返回所有预测结果，默认仅返回最终得分
            
        Returns:
            根据return_all:
                - True: (final_scores, upside_probs, ranking_scores)
                - False: final_scores
        """
        # 设置为评估模式
        self.eval()
        
        with torch.no_grad():
            # 确保输入为多尺度特征字典格式
            if not isinstance(sector_features, dict) or not all(k in sector_features for k in ['short_term', 'mid_term', 'long_term']):
                raise ValueError("板块特征必须是包含短期、中期、长期尺度的字典格式")
                
            # 确保特征数据在正确的设备上
            device = self.classifier.weight.device
            
            # 处理多尺度特征输入，将所有特征移到正确的设备
            for scale in sector_features:
                if isinstance(sector_features[scale], torch.Tensor) and sector_features[scale].device != device:
                    sector_features[scale] = sector_features[scale].to(device)
            
            # 如果市场特征也是字典格式
            if isinstance(market_features, dict):
                for scale in market_features:
                    if isinstance(market_features[scale], torch.Tensor) and market_features[scale].device != device:
                        market_features[scale] = market_features[scale].to(device)
            elif isinstance(market_features, torch.Tensor) and market_features.device != device:
                market_features = market_features.to(device)
            
            # 执行前向传播
            outputs = self.forward(sector_features, market_features, market_state)
            
            # 检查并处理输出结果
            final_scores, upside_probs, ranking_scores = outputs
            
            # 处理NaN和Inf值
            if torch.isnan(final_scores).any() or torch.isinf(final_scores).any():
                logger.warning("警告: 最终得分包含NaN或Inf值，将替换为0")
                final_scores = torch.nan_to_num(final_scores, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if torch.isnan(upside_probs).any() or torch.isinf(upside_probs).any():
                logger.warning("警告: 上涨概率包含NaN或Inf值，将替换为0.5")
                upside_probs = torch.nan_to_num(upside_probs, nan=0.5, posinf=1.0, neginf=0.0)
                # 确保概率值在0-1范围内
                upside_probs = torch.clamp(upside_probs, 0.0, 1.0)
                
            if torch.isnan(ranking_scores).any() or torch.isinf(ranking_scores).any():
                logger.warning("警告: 排名得分包含NaN或Inf值，将替换为0")
                ranking_scores = torch.nan_to_num(ranking_scores, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 根据参数返回结果
            if return_all:
                return final_scores, upside_probs, ranking_scores
            else:
                # 只返回最终得分
                return final_scores

