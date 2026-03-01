"""
阶梯式仓位控制 - 集成指南
Step Position Control - Integration Guide

本文档展示如何在现有的5分钟回测和实盘交易系统中集成阶梯式仓位控制
This document shows how to integrate step-based position control into existing 5-min backtest and live trading systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from step_position_control import StepPositionController, create_position_controller_from_data

# ========================================================================
# 场景1: 在5分钟回测中集成 (rqalpha_minute_datasource/strategy_backtest.py)
# ========================================================================

class IntegratedMinuteBacktestStrategy:
    """
    集成阶梯式仓位控制的5分钟回测策略
    """

    def __init__(self):
        # 初始化阶梯仓位控制器
        # Initialize step position controller
        self.position_controller = StepPositionController(
            top1_min=0.0,
            top1_max=1.0,
            step_thresholds=[
                (0.02, 1.0),  # Top 1: 预测强度前2% -> 满仓
                (0.01, 0.7),  # Top 2: 预测强度前1% -> 70%仓位
                (0.007, 0.4), # Top 3: 预测强度前0.7% -> 40%仓位
                (0.004, 0.3), # Top 4: 预测强度前0.4% -> 30%仓位
                (0.0, 0.2)    # Top 5: 其他 -> 20%仓位
            ]
        )

        self.initial_capital = 1000000.0

    def execute_buy_with_step_position(self,
                                      stock_code: str,
                                      top1_score: float,
                                      current_price: float,
                                      current_date: pd.Timestamp) -> Dict:
        """
        执行买入操作（使用阶梯式仓位控制）

        这是替换原有平滑函数逻辑的核心方法
        This is the core method replacing the original smooth function logic
        """

        # 旧逻辑（平滑函数）- 已废弃
        # Old logic (smooth function) - DEPRECATED
        # lead = top1_score - np.mean(top5_scores)
        # weight = some_smooth_function(lead)

        # 新逻辑（阶梯函数）- 当前使用
        # New logic (step function) - CURRENT
        position_weight = self.position_controller.get_position_weight(
            top1_score=top1_score,
            verbose=True
        )

        # 计算实际买入金额
        # Calculate actual buy amount
        capital_to_use = self.initial_capital * position_weight
        shares_to_buy = int(capital_to_use / current_price / 100) * 100  # 整手买入

        if shares_to_buy < 100:  # 至少一手
            return {
                'success': False,
                'reason': '计算出的股数不足1手（100股）'
            }

        actual_cost = shares_to_buy * current_price

        return {
            'success': True,
            'stock_code': stock_code,
            'buy_date': current_date,
            'buy_price': current_price,
            'shares': shares_to_buy,
            'cost': actual_cost,
            'position_weight': position_weight,
            'top1_score': top1_score,
            'capital_used': capital_to_use
        }


# ========================================================================
# 场景2: 在实盘交易中集成 (live_trading/trader.py)
# ========================================================================

class IntegratedLiveTrader:
    """
    集成阶梯式仓位控制的实盘交易器
    """

    def __init__(self, prediction_history: pd.DataFrame):
        # 从历史预测数据自动创建控制器
        # Auto-create controller from historical prediction data
        self.position_controller = create_position_controller_from_data(
            prediction_data=prediction_history,
            score_column='pred_top1',
            custom_thresholds=[
                (0.02, 1.0),
                (0.01, 0.7),
                (0.007, 0.4),
                (0.004, 0.3),
                (0.0, 0.2)
            ]
        )

        self.current_capital = 1000000.0

    def calculate_position_size(self,
                              stock_code: str,
                              top1_score: float,
                              current_price: float) -> Dict:
        """
        计算持仓大小（实盘交易专用）

        Args:
            stock_code: 股票代码
            top1_score: Top1预测得分
            current_price: 当前价格

        Returns:
            position_info: 包含股数、权重、资金等信息
        """
        position_weight = self.position_controller.get_position_weight(
            top1_score=top1_score,
            verbose=True
        )

        capital_to_use = self.current_capital * position_weight
        shares = int(capital_to_use / current_price / 100) * 100

        return {
            'stock_code': stock_code,
            'shares': shares,
            'weight': position_weight,
            'capital_required': shares * current_price,
            'top1_score': top1_score
        }


# ========================================================================
# 场景3: 与现有代码的对比示例
# ========================================================================

def compare_smooth_vs_step_position():
    """
    对比平滑函数 vs 阶梯函数的仓位控制效果
    """

    print("=" * 80)
    print("对比：平滑函数 vs 阶梯函数仓位控制")
    print("=" * 80)

    # 模拟预测得分
    test_scores = np.array([0.003, 0.005, 0.008, 0.015, 0.025, 0.04])

    # 方案1: 平滑函数（旧方法）
    # Smooth function (OLD METHOD)
    top5_scores_mean = 0.01  # 假设Top5平均分
    lead_smooth = test_scores - top5_scores_mean

    # 简单的线性映射作为平滑函数示例
    # Simple linear mapping as smooth function example
    weight_smooth = np.clip(lead_smooth * 50, 0.2, 1.0)  # 归一化并限制在[0.2, 1.0]

    # 方案2: 阶梯函数（新方法）
    # Step function (NEW METHOD)
    step_controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),
            (0.01, 0.7),
            (0.007, 0.4),
            (0.004, 0.3),
            (0.0, 0.2)
        ]
    )
    weight_step = np.array([step_controller.get_position_weight(s) for s in test_scores])

    # 对比结果
    # Comparison results
    comparison_df = pd.DataFrame({
        'top1_score': test_scores,
        'lead (smooth)': lead_smooth,
        'weight_smooth': weight_smooth,
        'weight_step': weight_step,
        'difference': weight_smooth - weight_step
    })

    print("\n仓位对比：")
    print(comparison_df.to_string(index=False))

    print("\n关键差异：")
    print("  1. 平滑函数：仓位随得分连续变化，对微小差异敏感")
    print("  2. 阶梯函数：仓位在阈值处跳跃，只在关键阈值变化")
    print("  3. 平滑函数优势：更精细的仓位控制")
    print("  4. 阶梯函数优势：更稳健，减少过度交易，降低交易成本")


# ========================================================================
# 场景4: 实际应用示例
# ========================================================================

def practical_example_usage():
    """
    实际应用示例：从预测到执行的完整流程
    """

    print("\n" + "=" * 80)
    print("实际应用示例：完整交易流程")
    print("=" * 80)

    # 步骤1: 获取预测数据
    # Step 1: Get prediction data
    prediction_result = {
        'date': pd.Timestamp('2026-02-14'),
        'stock_code': '000001.SZ',
        'pred_top1': 0.018,  # 预测得分1.8%
        'pred_top5': [0.018, 0.015, 0.012, 0.010, 0.008],
        'current_price': 15.50
    }

    # 步骤2: 初始化控制器
    # Step 2: Initialize controller
    controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),  # 1.8% < 2%，进入下一阶梯
            (0.01, 0.7),  # 1.8% >= 1%，匹配此阶梯
            (0.007, 0.4),
            (0.004, 0.3),
            (0.0, 0.2)
        ]
    )

    # 步骤3: 计算仓位权重
    # Step 3: Calculate position weight
    position_weight = controller.get_position_weight(
        top1_score=prediction_result['pred_top1'],
        verbose=True
    )

    # 步骤4: 计算买入股数
    # Step 4: Calculate buy shares
    capital = 1000000.0
    capital_to_use = capital * position_weight
    shares = int(capital_to_use / prediction_result['current_price'] / 100) * 100

    print(f"\n交易决策：")
    print(f"  股票代码: {prediction_result['stock_code']}")
    print(f"  预测得分: {prediction_result['pred_top1']:.4f} ({prediction_result['pred_top1']*100:.2f}%)")
    print(f"  当前价格: CNY {prediction_result['current_price']:.2f}")
    print(f"  仓位权重: {position_weight*100:.0f}%")
    print(f"  使用资金: CNY {capital_to_use:,.2f}")
    print(f"  买入股数: {shares} 股 ({shares//100} 手)")
    print(f"  预计成本: CNY {shares * prediction_result['current_price']:,.2f}")


# ========================================================================
# 运行示例
# ========================================================================

if __name__ == "__main__":
    # 运行所有示例
    # Run all examples
    compare_smooth_vs_step_position()
    practical_example_usage()

    print("\n" + "=" * 80)
    print("集成指南完成")
    print("=" * 80)
    print("\n下一步操作：")
    print("  1. 在 rqalpha_minute_datasource/strategy_backtest.py 中集成")
    print("  2. 在 live_trading/trader.py 中集成")
    print("  3. 运行回测验证效果")
    print("  4. 实盘验证")
