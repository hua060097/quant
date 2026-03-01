"""
阶梯式仓位控制模块
Step-based Position Control Module

用途：根据预测强度动态调整仓位权重
Use: Dynamically adjust position weights based on prediction strength
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StepPositionController:
    """
    阶梯式仓位控制器

    原理：根据预测值的强度（pred_strength）将仓位分为多个阶梯
    Principle: Divide positions into multiple levels based on prediction strength

    与平滑函数的区别：
    Difference from smooth functions:
    - 平滑函数：仓位随预测强度连续变化（如 lead = top1_score - mean(top5_scores)）
    - 阶梯函数：仓位在特定阈值点跳跃变化，每个阶梯有固定权重
    """

    def __init__(self,
                 top1_min: float = 0.0,
                 top1_max: float = 1.0,
                 # 阶梯阈值和权重配置
                 # Step thresholds and weights configuration
                 step_thresholds: List[Tuple[float, float]] = None):
        """
        初始化阶梯控制器

        Args:
            top1_min: 预测值最小值（用于归一化）
            top1_max: 预测值最大值（用于归一化）
            step_thresholds: 阶梯配置列表 [(threshold, weight), ...]
                            按threshold降序排列，第一个满足的阶梯生效
        """
        self.top1_min = top1_min
        self.top1_max = top1_max

        # 默认阶梯配置（基于testaa_insert_step_function.py中的模式）
        # Default step configuration (based on pattern in testaa_insert_step_function.py)
        if step_thresholds is None:
            self.step_thresholds = [
                (0.02, 1.0),   # Top 1: 预测强度前2% -> 满仓
                (0.01, 0.7),   # Top 2: 预测强度前1% -> 70%仓位
                (0.007, 0.4),  # Top 3: 预测强度前0.7% -> 40%仓位
                (0.004, 0.3),  # Top 4: 预测强度前0.4% -> 30%仓位
                (0.0, 0.2)     # Top 5: 其他 -> 20%仓位
            ]
        else:
            self.step_thresholds = step_thresholds

        logger.info(f"[INIT] 阶梯式仓位控制器初始化完成")
        logger.info(f"[INIT] 阶梯配置: {self.step_thresholds}")

    def calculate_pred_strength(self, top1_score: float) -> float:
        """
        计算预测强度（归一化到0-1范围）

        Args:
            top1_score: Top1预测得分

        Returns:
            pred_strength: 归一化后的预测强度 [0, 1]
        """
        if self.top1_max > self.top1_min:
            pred_strength = (top1_score - self.top1_min) / (self.top1_max - self.top1_min)
        else:
            pred_strength = 1.0

        # 确保在[0, 1]范围内
        pred_strength = max(0.0, min(1.0, pred_strength))
        return pred_strength

    def get_position_weight(self, top1_score: float, verbose: bool = False) -> float:
        """
        根据预测得分计算仓位权重（阶梯函数）

        Args:
            top1_score: Top1预测得分
            verbose: 是否输出详细日志

        Returns:
            weight: 仓位权重 [0, 1]
        """
        # 步骤1: 计算预测强度
        # Step 1: Calculate prediction strength
        pred_strength = self.calculate_pred_strength(top1_score)

        # 步骤2: 根据阶梯阈值确定权重
        # Step 2: Determine weight based on step thresholds
        weight = 0.0
        matched_level = None

        for threshold, step_weight in self.step_thresholds:
            if pred_strength >= threshold:
                weight = step_weight
                matched_level = f"strength>={threshold}"
                break

        if verbose:
            logger.info(f"  [POSITION] top1_score={top1_score:.6f} -> pred_strength={pred_strength:.6f} -> {matched_level} -> weight={weight:.2f}")

        return weight

    def calculate_portfolio_weights(self,
                                   prediction_data: pd.DataFrame,
                                   score_column: str = 'pred_top1') -> pd.Series:
        """
        为整个投资组合计算权重

        Args:
            prediction_data: 预测数据 DataFrame
            score_column: 预测得分列名

        Returns:
            weights: 每只股票的仓位权重 Series (index=stock_code或原index)
        """
        weights = prediction_data[score_column].apply(
            lambda score: self.get_position_weight(score)
        )
        return weights

    def update_top1_bounds(self, top1_min: float, top1_max: float):
        """
        更新Top1预测值的边界（用于动态归一化）

        Args:
            top1_min: 新的最小值
            top1_max: 新的最大值
        """
        self.top1_min = top1_min
        self.top1_max = top1_max
        logger.info(f"[UPDATE] 更新Top1边界: min={top1_min:.6f}, max={top1_max:.6f}")


def create_position_controller_from_data(prediction_data: pd.DataFrame,
                                       score_column: str = 'pred_top1',
                                       custom_thresholds: List[Tuple[float, float]] = None) -> StepPositionController:
    """
    根据历史数据自动创建仓位控制器

    Args:
        prediction_data: 历史预测数据
        score_column: 预测得分列名
        custom_thresholds: 自定义阶梯阈值

    Returns:
        controller: StepPositionController实例
    """
    # 计算Top1得分的边界
    top1_min = prediction_data[score_column].min()
    top1_max = prediction_data[score_column].max()

    logger.info(f"[AUTO] 自动计算Top1边界: min={top1_min:.6f}, max={top1_max:.6f}")

    controller = StepPositionController(
        top1_min=top1_min,
        top1_max=top1_max,
        step_thresholds=custom_thresholds
    )

    return controller


# ========================================================================
# 使用示例
# ========================================================================

if __name__ == "__main__":
    # 示例1: 基本使用
    print("=" * 80)
    print("示例1: 基本阶梯式仓位控制")
    print("=" * 80)

    controller = StepPositionController(
        top1_min=0.0,
        top1_max=1.0,
        step_thresholds=[
            (0.02, 1.0),  # 2%以上 -> 100%仓位
            (0.01, 0.7),  # 1-2% -> 70%仓位
            (0.007, 0.4), # 0.7-1% -> 40%仓位
            (0.004, 0.3), # 0.4-0.7% -> 30%仓位
            (0.0, 0.2)    # 0-0.4% -> 20%仓位
        ]
    )

    test_scores = [0.003, 0.005, 0.008, 0.015, 0.025]
    for score in test_scores:
        weight = controller.get_position_weight(score, verbose=True)

    print("\n" + "=" * 80)
    print("示例2: 从历史数据自动创建控制器")
    print("=" * 80)

    # 模拟历史预测数据
    np.random.seed(42)
    mock_data = pd.DataFrame({
        'pred_top1': np.random.beta(2, 5, 1000)  # 模拟偏态分布
    })

    auto_controller = create_position_controller_from_data(mock_data)

    print(f"\n自动创建的控制器配置:")
    print(f"  Top1范围: [{auto_controller.top1_min:.6f}, {auto_controller.top1_max:.6f}]")
    print(f"  阶梯数量: {len(auto_controller.step_thresholds)}")

    # 测试几个样本
    print(f"\n测试样本:")
    sample_scores = [0.1, 0.2, 0.3, 0.5, 0.7]
    for score in sample_scores:
        weight = auto_controller.get_position_weight(score, verbose=True)
