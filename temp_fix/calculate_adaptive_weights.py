#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于计算自适应权重的函数，改进模型权重调整机制
"""

def _calculate_adaptive_weights(self, model_package, market_status=None):
    """
    计算自适应模型权重
    
    Args:
        model_package: 模型包，包含性能历史记录
        market_status: 市场状态数据
        
    Returns:
        tuple: (strength_weight, lnn_weight) 强度模型和LNN模型的权重
    """
    # 默认权重
    strength_weight, lnn_weight = 0.6, 0.4
    
    # 如果没有性能历史记录，使用默认权重
    performance_history = model_package.get('performance_history', {})
    if not performance_history:
        return strength_weight, lnn_weight
        
    try:
        # 基于历史性能调整权重
        strength_rmse = performance_history.get('strength_model_rmse', 0.05)
        lnn_rmse = performance_history.get('lnn_model_rmse', 0.05)
        
        # 预测误差越小，权重越大
        strength_rmse = max(strength_rmse, 0.0001)
        lnn_rmse = max(lnn_rmse, 0.0001)
        
        # 基于RMSE计算权重
        strength_weight = 1/strength_rmse / (1/strength_rmse + 1/lnn_rmse)
        lnn_weight = 1/lnn_rmse / (1/strength_rmse + 1/lnn_rmse)
        
        # 根据市场状态微调权重
        if market_status is not None:
            market_type = market_status.get("market_type", 0)
            market_trend_strength = market_status.get("trend_strength", 0)
            
            # 在强趋势市场中调整权重
            if isinstance(market_type, (int, float)):
                # 强趋势市场，增加LNN权重
                if market_type in [11, 12, 13, 21, 22, 23, 41, 42, 43, 51, 52, 53]:
                    weight_shift = min(0.1, market_trend_strength * 0.02)
                    strength_weight -= weight_shift
                    lnn_weight += weight_shift
                # 震荡市场，增加强度模型权重
                else:
                    weight_shift = min(0.1, (1 - market_trend_strength) * 0.02)
                    strength_weight += weight_shift
                    lnn_weight -= weight_shift
        
        # 考虑预测确信度 (如果可用)
        if 'strength_confidence' in performance_history and 'lnn_confidence' in performance_history:
            strength_conf = performance_history.get('strength_confidence', 0.5)
            lnn_conf = performance_history.get('lnn_confidence', 0.5)
            
            # 确信度越高，权重越大
            conf_factor = 0.2  # 确信度的影响因子
            strength_adj = strength_weight * (1 + (strength_conf - 0.5) * conf_factor)
            lnn_adj = lnn_weight * (1 + (lnn_conf - 0.5) * conf_factor)
            
            # 重新归一化
            total_adj = strength_adj + lnn_adj
            strength_weight = strength_adj / total_adj
            lnn_weight = lnn_adj / total_adj
        
        # 确保权重和为1
        total = strength_weight + lnn_weight
        strength_weight /= total
        lnn_weight /= total
        
        self.logger.debug(f"自适应权重: 强度={strength_weight:.4f}, LNN={lnn_weight:.4f}")
    except Exception as e:
        self.logger.warning(f"计算自适应权重时出错: {e}")
    
    return strength_weight, lnn_weight 