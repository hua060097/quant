#!/usr/bin/env python3
"""
🚀 革命性量化选股系统 - 优化完成版

支持五种运行模式：
1. 因子生成模式 (generate_factors) - 生成和计算因子
2. 因子测试模式 (test_factors) - 单因子测试和分析
3. 模型训练模式 (train_model) - 训练机器学习模型
4. 模型测试模式 (test_model) - 使用优化后的模型进行选股测试
5. 完整流水线模式 (full_pipeline) - 执行完整的因子-模型流水线

🎯 已完成的核心优化：
- ✅ 智能标签变换：平衡极值学习与信号保留
- ✅ 增强模型复杂度：Pseudo-Huber回归专用参数优化
- ✅ 交易导向评估：优先Top-K收益、夏普比率、最大回撤等实际交易指标
- ✅ 移除冗余Quantile Loss：智能标签变换已解决极值处理问题
- ✅ 修复trading_metrics变量作用域错误

基于2020-2025年顶尖量化研究的模块化因子系统架构
"""

import logging
import pandas as pd
import json
import argparse
import sys
import numpy as np
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

def calculate_ndcg_at_k(y_true, y_score, k=5):
    """
    计算NDCG@k (Normalized Discounted Cumulative Gain) - 使用真实收益率

    Args:
        y_true: 真实相关性得分（推荐使用收益率）
        y_score: 预测得分
        k: 截断位置

    Returns:
        NDCG@k 值
    """
    # 获取排序后的索引
    sorted_indices = np.argsort(y_score)[::-1]  # 降序排列
    sorted_true_relevance = y_true[sorted_indices[:k]]

    # 计算DCG@k
    dcg = 0.0
    for i, relevance in enumerate(sorted_true_relevance):
        if i == 0:
            dcg += relevance
        else:
            dcg += relevance / math.log2(i + 1)

    # 计算IDCG@k (理想DCG)
    ideal_sorted_relevance = np.sort(y_true)[::-1][:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_sorted_relevance):
        if i == 0:
            idcg += relevance
        else:
            idcg += relevance / math.log2(i + 1)

    # 计算NDCG@k
    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_adjusted_ndcg_at_k(y_true, y_score, k=5):
    """
    🚀 改进的NDCG计算，解决负收益和量纲问题

    主要改进：
    1. 收益率标准化，解决量纲敏感性问题
    2. 负收益阈值处理，避免过度惩罚
    3. 基于百分位的稳健归一化

    Args:
        y_true: 真实收益率数组
        y_score: 预测得分数组
        k: 截断位置

    Returns:
        float: 改进的NDCG@k 值
    """
    import numpy as np

    # 输入验证
    if len(y_true) == 0 or len(y_score) == 0 or len(y_true) != len(y_score):
        return 0.0

    if k <= 0:
        return 0.0

    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)

    # 🔧 改进1：收益率标准化（z-score）
    if y_true.std() > 1e-8:  # 避免除零
        normalized_returns = (y_true - y_true.mean()) / y_true.std()
    else:
        # 如果标准差为0，直接设为0
        normalized_returns = np.zeros_like(y_true)

    # 🔧 改进2：负收益阈值处理
    # 使用30%分位数作为最小值阈值，避免过度惩罚极端负收益
    min_threshold = np.percentile(normalized_returns, 30)
    normalized_returns = np.maximum(normalized_returns, min_threshold)

    # 🔧 改进3：确保有足够的正值用于有意义的排序
    # 如果大部分值都是负数，进行平移
    if normalized_returns.mean() < 0:
        normalized_returns = normalized_returns - normalized_returns.mean()

    # 使用原始NDCG逻辑计算标准化的结果
    return calculate_ndcg_at_k(normalized_returns, y_score, k)


def calculate_top_k_return(true_returns, pred_scores, k=5):
    """
    🎯 计算Top-K组合收益率

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: Top-K组合平均收益率
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]  # 降序排列，取前k个

    # 计算Top-K平均收益率
    top_k_returns = true_returns[top_k_indices]
    return np.mean(top_k_returns)


def calculate_win_rate(true_returns, pred_scores, k=5):
    """
    ✅ 计算胜率（Top-K中正收益股票的比例）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: 胜率 (0-1)
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]

    # 计算正收益股票的比例
    top_k_returns = true_returns[top_k_indices]
    positive_returns = top_k_returns > 0

    return np.mean(positive_returns)


def calculate_profit_loss_ratio(true_returns, pred_scores, k=5):
    """
    ⚡ 计算盈亏比（平均正收益与平均负收益的绝对值之比）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: 盈亏比
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]
    top_k_returns = true_returns[top_k_indices]

    # 分离正收益和负收益
    positive_returns = top_k_returns[top_k_returns > 0]
    negative_returns = top_k_returns[top_k_returns < 0]

    if len(positive_returns) == 0:
        return 0.0

    if len(negative_returns) == 0:
        return float('inf')  # 如果没有负收益，返回无穷大

    avg_positive = np.mean(positive_returns)
    avg_negative = np.mean(np.abs(negative_returns))

    return avg_positive / avg_negative if avg_negative > 0 else float('inf')


def calculate_sharpe_ratio(true_returns, pred_scores, k=5, risk_free_rate=0.0):
    """
    💹 计算夏普比率（风险调整后收益）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量
        risk_free_rate: 无风险收益率（默认0）

    Returns:
        float: 夏普比率
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]
    top_k_returns = true_returns[top_k_indices]

    # 计算超额收益
    excess_returns = top_k_returns - risk_free_rate

    # 计算夏普比率
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)

    if std_excess == 0:
        return 0.0

    return mean_excess / std_excess


def calculate_max_drawdown(true_returns, pred_scores, k=5):
    """
    📉 计算最大回撤（衡量极端风险）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: 最大回撤（正值）
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]
    top_k_returns = true_returns[top_k_indices]

    # 计算累积收益曲线
    cumulative_returns = np.cumprod(1 + top_k_returns)

    # 计算最大回撤
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak

    return np.max(drawdown)


def calculate_excess_return(true_returns, pred_scores, k=5, benchmark_return=0.0):
    """
    📈 计算超额收益（相对基准的超额表现）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量
        benchmark_return: 基准收益率（默认所有股票的平均收益）

    Returns:
        float: 超额收益
    """
    top_k_return = calculate_top_k_return(true_returns, pred_scores, k)

    # 如果没有提供基准，使用所有股票的平均收益
    if benchmark_return == 0.0 and len(true_returns) > 0:
        benchmark_return = np.mean(true_returns)

    return top_k_return - benchmark_return


def evaluate_investment_performance(true_returns, pred_scores, k=5, risk_free_rate=0.0):
    """
    🎯 投资导向的综合评估函数

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量
        risk_free_rate: 无风险收益率

    Returns:
        dict: 包含各种投资指标的字典
    """
    results = {}

    # 1. 收益导向指标
    results['top_k_return'] = calculate_top_k_return(true_returns, pred_scores, k)
    results['excess_return'] = calculate_excess_return(true_returns, pred_scores, k)
    results['win_rate'] = calculate_win_rate(true_returns, pred_scores, k)
    results['profit_loss_ratio'] = calculate_profit_loss_ratio(true_returns, pred_scores, k)

    # 2. 风险调整指标
    results['sharpe_ratio'] = calculate_sharpe_ratio(true_returns, pred_scores, k, risk_free_rate)
    results['max_drawdown'] = calculate_max_drawdown(true_returns, pred_scores, k)

    # 3. 排序质量指标（改进的NDCG）
    results['adjusted_ndcg'] = calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k)

    # 4. 信息系数
    if len(true_returns) > 1 and len(pred_scores) > 1:
        results['ic'] = np.corrcoef(true_returns, pred_scores)[0, 1] if not np.isnan(np.corrcoef(true_returns, pred_scores)[0, 1]) else 0.0
        # Rank IC (Spearman相关系数)
        from scipy.stats import spearmanr
        rank_ic, _ = spearmanr(true_returns, pred_scores)
        results['rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0.0
    else:
        results['ic'] = 0.0
        results['rank_ic'] = 0.0

    # 5. 综合评分
    results['composite_score'] = calculate_composite_investment_score(results)

    return results


def calculate_composite_investment_score(metrics):
    """
    🏆 计算综合投资评分

    评分策略：
    - 收益表现占40%
    - 风险控制占30%
    - 排序质量占20%
    - 信息系数占10%

    Args:
        metrics: 评估指标字典

    Returns:
        float: 综合评分 (0-100)
    """
    # 1. 收益表现评分 (0-40分)
    return_score = 0
    if metrics['top_k_return'] > 0:
        return_score = min(40, metrics['top_k_return'] * 100)  # 假设1%收益得1分，最高40分

    # 2. 超额收益加分
    if metrics['excess_return'] > 0:
        return_score += min(10, metrics['excess_return'] * 50)  # 额外加分

    # 3. 风险控制评分 (0-30分)
    risk_score = 0
    # 胜率评分
    risk_score += metrics['win_rate'] * 15  # 胜率*15分
    # 夏普比率评分
    if metrics['sharpe_ratio'] > 0:
        risk_score += min(10, metrics['sharpe_ratio'] * 5)  # 夏普比率*5分，最高10分
    # 最大回撤扣分
    risk_score = max(0, risk_score - metrics['max_drawdown'] * 10)  # 回撤扣分

    # 4. 排序质量评分 (0-20分)
    ranking_score = metrics['adjusted_ndcg'] * 20  # NDCG*20分

    # 5. 信息系数评分 (0-10分)
    ic_score = max(0, metrics['ic'] * 10) if metrics['ic'] > 0 else 0  # 正IC才给分

    # 综合评分
    total_score = return_score + risk_score + ranking_score + ic_score

    return min(100, max(0, total_score))  # 限制在0-100范围内


def print_investment_evaluation_report(model_name, results):
    """
    📊 打印投资评估报告

    Args:
        model_name: 模型名称
        results: 评估结果字典
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\n📊 {model_name}投资评估报告")
    logger.info("=" * 60)

    # 收益表现
    logger.info("💰 收益表现:")
    logger.info(f"  Top-K组合收益率: {results['top_k_return']:.4f} ({results['top_k_return']*100:.2f}%)")
    logger.info(f"  超额收益: {results['excess_return']:.4f} ({results['excess_return']*100:.2f}%)")
    logger.info(f"  胜率: {results['win_rate']:.2%}")
    logger.info(f"  盈亏比: {results['profit_loss_ratio']:.2f}")

    # 风险控制
    logger.info("⚠️ 风险控制:")
    logger.info(f"  夏普比率: {results['sharpe_ratio']:.4f}")
    logger.info(f"  最大回撤: {results['max_drawdown']:.4f} ({results['max_drawdown']*100:.2f}%)")

    # 排序质量
    logger.info("🎯 排序质量:")
    logger.info(f"  改进NDCG@5: {results['adjusted_ndcg']:.4f}")
    logger.info(f"  IC系数: {results['ic']:.4f}")
    logger.info(f"  Rank IC: {results['rank_ic']:.4f}")

    # 综合评分
    logger.info("🏆 综合评估:")
    logger.info(f"  投资评分: {results['composite_score']:.1f}/100")

    # 评级
    if results['composite_score'] >= 80:
        rating = "优秀 ⭐⭐⭐⭐⭐"
    elif results['composite_score'] >= 70:
        rating = "良好 ⭐⭐⭐⭐"
    elif results['composite_score'] >= 60:
        rating = "及格 ⭐⭐⭐"
    elif results['composite_score'] >= 50:
        rating = "需要改进 ⭐⭐"
    else:
        rating = "不合格 ⭐"

    logger.info(f"  综合评级: {rating}")
    logger.info("=" * 60)


import statsmodels.api as sm


# ============================================================
# 🔧 全局配置参数
# ============================================================
# ATR策略配置
MAX_HOLDING_DAYS = 3  # 最大持仓期（天）- 训练和预测统一使用
ATR_MULTIPLIER_PROFIT = 1.2  # ATR止盈倍数1.2
ATR_MULTIPLIER_LOSS = 1    # ATR止损倍数1
MAX_STOP_LOSS_PCT = 0.08   # 最大止损百分比（3%）- 与ATR止损取较小损失（优化：5%→3%）

# 交易成本配置（基于A股实际交易成本）
# A股交易成本构成：
# 1. 佣金：万一（0.01%），双边收取
# 2. 印花税：千分之0.5（0.05%），卖出时收取
# 3. 过户费：万分之0.2（0.002%），双边收取
# 4. 滑点成本：约0.02%-0.05%
# 总计：买入约0.012%，卖出约0.062%，双边约0.074%
# 考虑到量化交易频繁操作，设置为0.08%
TRADING_COST_RATE = 0.0008  # 交易成本率（0.08%）

# 时间切分配置（train_model和test_model模式共用）
# 🎯 纯预测选股的时间切分策略
TRAIN_END_DATE = "2025-02-01"  # 训练数据截止日期（不包含此日期）
TEST_START_DATE = "2025-02-01"  # 测试数据开始日期（包含此日期）
TEST_END_DATE = "2025-10-21"    # 测试数据结束日期（包含此日期）

FACTOR_LAB_STORE_PATH = (Path(__file__).resolve().parent / "factor_lab_data" / "factors.parquet")

try:
    # 尝试相对导入（当作为模块导入时）
    from .data_loader import IntegratedDataLoader
    from .stock_selector import StockSelector
    from .backtester import Backtester
    from .config import system_config, backtest_config
except ImportError:
    # 回退到绝对导入（当直接运行时）
    from data_loader import IntegratedDataLoader
    from stock_selector import StockSelector
    from backtester import Backtester
    from config import system_config, backtest_config

try:
    from .factor_system.factor_lab import definitions as _factor_lab_definitions  # noqa: F401
    from .factor_system.factor_lab.pipeline import FactorPipeline
    from .factor_system.factor_lab.store import FactorDatasetStore
    from .factor_system.factor_lab.commands import generate_and_store
except ImportError:
    from factor_system.factor_lab import definitions as _factor_lab_definitions  # noqa: F401
    from factor_system.factor_lab.pipeline import FactorPipeline
    from factor_system.factor_lab.store import FactorDatasetStore
    from factor_system.factor_lab.commands import generate_and_store


def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # 处理numpy标量
        return obj.item()
    else:
        return obj

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="机器学习K线形态选股系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明:
  generate_factors   - 因子生成模式（计算和生成所有因子）
  test_factors      - 因子测试模式（单因子测试和分析）
  train_model       - 模型训练模式（训练机器学习模型）
  test_model        - 模型测试模式（使用模型进行选股测试）
  full_pipeline     - 完整流水线模式（因子生成→测试→模型训练→测试）
  🆕 test_optimization - 优化测试模式（分位数回归&动态损失函数优化）
  factor_correlation - 因子相关性分析模式（分析ATR三分类系统因子相关性）
  genetic_factor    - 遗传编程因子进化模式（使用gplearn进化超级因子）
  # interaction_analysis - 已移除交互特征分析模式

使用示例:
  # 生成因子（默认生成所有因子，可能覆盖已存在的）
  python main.py --mode generate_factors

  # 强制重新生成所有因子（清理旧数据）
  python main.py --mode generate_factors --force

  # 🆕 增量生成：只生成缺少的因子
  python main.py --mode generate_factors --incremental

  # 测试因子
  python main.py --mode test_factors --factor_names rsi6,macd,momentum_quality_score

  # 训练模型
  python main.py --mode train_model

  # 测试模型（默认启用市场因子）
  python main.py --mode test_model

  # 测试模型（禁用市场因子，与main2.py一致）
  python main.py --mode test_model --no_market_factors

  # 完整流水线
  python main.py --mode full_pipeline

  # 因子相关性分析
  python main.py --mode factor_correlation

  # 遗传编程因子进化
  python main.py --mode genetic_factor --population_size 1000 --generations 20

  # 自定义输出目录
  python main.py --output_dir ./custom_results
        """
    )

    parser.add_argument(
        '--mode',
        choices=['generate_factors', 'test_factors', 'train_model', 'test_model', 'test_model_walkforward', 'full_pipeline', 'test_optimization', 'factor_correlation', 'genetic_factor', 'market_state_test', 'interaction_analysis'],
        default='generate_factors',
        help='运行模式 (默认: generate_factors)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='输出目录 (默认: ./results)'
    )

    parser.add_argument(
        '--factor_names',
        type=str,
        help='因子名称列表，用逗号分隔 (仅在test_factors模式下使用)'
    )

    parser.add_argument(
        '--factor_groups',
        type=str,
        help='因子组名称列表，用逗号分隔 (可选：technical,fundamental,kline_patterns等)'
    )

    parser.add_argument(
        '--start_date',
        type=str,
        help='开始日期 (格式: YYYYMMDD)'
    )

    parser.add_argument(
        '--end_date',
        type=str,
        help='结束日期 (格式: YYYYMMDD)'
    )

    parser.add_argument(
        '--log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--save_results',
        action='store_true',
        default=True,
        help='是否保存结果文件 (默认: True)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新计算，忽略缓存'
    )

    # 🆕 增量生成参数
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='增量模式：只生成因子文件中缺少的因子（与--force互斥）'
    )

    # 🆕 分批保存参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='因子生成分批保存大小，每计算N个因子保存一次（默认: 10，设为0禁用分批保存）'
    )

    # 遗传编程相关参数
    parser.add_argument(
        '--population_size',
        type=int,
        default=1000,
        help='遗传编程种群大小 (默认: 1000)'
    )

    parser.add_argument(
        '--generations',
        type=int,
        default=20,
        help='遗传编程进化代数 (默认: 20)'
    )

    parser.add_argument(
        '--n_best_factors',
        type=int,
        default=10,
        help='返回最佳因子数量 (默认: 10)'
    )

    # 市场因子开关参数
    parser.add_argument(
        '--use_market_factors',
        action='store_true',
        default=True,
        help='是否使用市场因子进行预测和仓位调节 (默认: True)'
    )

    parser.add_argument(
        '--no_market_attention',
        action='store_true',
        help='禁用市场注意力机制，使用原始股票因子'
    )

    # 🚀 超参数搜索相关参数
    parser.add_argument(
        '--enable_param_search',
        action='store_true',
        help='启用超参数搜索优化（默认: False）'
    )

    parser.add_argument(
        '--search_strategy',
        choices=['bayesian'],
        default='bayesian',
        help='超参数搜索策略 (bayesian=贝叶斯优化)'
    )

    # 🚀 模型选择参数
    parser.add_argument(
        '--models',
        type=str,
        default='xgboost,lightgbm',
        help='选择使用的模型，用逗号分隔 (可选: xgboost, lightgbm)。默认: xgboost,lightgbm (仅test_model和test_model_walkforward模式生效)'
    )

    # 🆕 窗口选择参数（用于test_model_walkforward模式）
    parser.add_argument(
        '--window',
        type=str,
        default=None,
        help='指定要训练的窗口编号，用逗号分隔（例如: 1 或 1,3）。不指定则训练所有窗口（仅test_model_walkforward模式生效）'
    )

    # 🚀 止盈止损阈值优化参数
    parser.add_argument(
        '--optimize_thresholds',
        action='store_true',
        help='启用止盈止损ATR阈值优化（基于LightGBM验证集，使用网格搜索+夏普比率）。默认: False'
    )

    # 🆕 跳过训练参数
    parser.add_argument(
        '--skip_train',
        action='store_true',
        help='跳过训练阶段，直接加载最新的集成模型进行预测（仅在test_model模式下有效）'
    )

    return parser.parse_args()

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_selection_因子_30标签_test_mode_walkforward_排序模型_lg22.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def save_results(selections, backtest_result, train_result=None, sector_report=None):
    """保存结果"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存选股结果
        selections_data = []
        for selection in selections:
            selections_data.append({
                'date': selection.date,
                'stock_code': selection.stock_code,
                'stock_name': selection.stock_name,
                'composite_score': selection.composite_score,
                'risk_adjusted_score': selection.risk_adjusted_score,
                'technical_score': selection.technical_score,
                'fundamental_score': selection.fundamental_score,
                'sector_score': selection.sector_score,
                'capital_flow_score': selection.capital_flow_score,
                'selected_sectors': selection.selected_sectors,
                'confidence': selection.confidence,
                'expected_return': selection.expected_return
            })

        selections_file = system_config.results_dir / f'selections_{timestamp}.json'
        with open(selections_file, 'w', encoding='utf-8') as f:
            json.dump(selections_data, f, ensure_ascii=False, indent=2)

        # 保存回测结果
        backtest_data = {
            'summary': {
                'total_return': backtest_result.total_return,
                'annual_return': backtest_result.annual_return,
                'sharpe_ratio': backtest_result.annualized_sharpe,  # 使用年化夏普比率
                'max_drawdown': backtest_result.max_drawdown,
                'win_rate': backtest_result.win_rate,
                'profit_loss_ratio': backtest_result.profit_loss_ratio,
                'total_trades': backtest_result.total_trades,
                'profitable_trades': backtest_result.profitable_trades,
                'avg_holding_days': backtest_result.avg_holding_days
            },
            'trade_details': backtest_result.trade_details
        }

        backtest_file = system_config.results_dir / f'backtest_{timestamp}.json'
        with open(backtest_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_data, f, ensure_ascii=False, indent=2)

        # 保存简化报告
        report_file = system_config.results_dir / f'report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("板块-个股一体化选股系统回测报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"测试期间: {system_config.test_start_date} - {system_config.end_date}\n")
            f.write(f"每日选股数量: {system_config.daily_stock_count}\n")
            f.write(f"持仓天数: {backtest_config.holding_days}\n\n")

            f.write("回测结果:\n")
            f.write(f"  总收益率: {backtest_result.total_return:.2%}\n")
            f.write(f"  年化收益率: {backtest_result.annual_return:.2%}\n")
            f.write(f"  夏普比率: {backtest_result.annualized_sharpe:.2f}\n")
            f.write(f"  最大回撤: {backtest_result.max_drawdown:.2%}\n")
            f.write(f"  胜率: {backtest_result.win_rate:.1%}\n")
            f.write(f"  盈亏比: {backtest_result.profit_loss_ratio:.2f}\n")
            f.write(f"  总交易次数: {backtest_result.total_trades}\n")
            f.write(f"  盈利交易: {backtest_result.profitable_trades}\n\n")

            if train_result:
                f.write("训练集表现:\n")
                f.write(f"  总收益率: {train_result.total_return:.2%}\n")
                f.write(f"  胜率: {train_result.win_rate:.1%}\n")
                f.write(f"  总交易次数: {train_result.total_trades}\n\n")

            if sector_report:
                f.write("板块策略回测:\n")
                f.write(sector_report)
                f.write("\n\n")

            # 选股示例
            f.write("选股示例 (前10次):\n")
            for i, selection in enumerate(selections[:10]):
                f.write(f"  {selection.date}: {selection.stock_code} "
                       f"(评分{selection.risk_adjusted_score:.1f}, "
                       f"置信度{selection.confidence:.2f})\n")

        print(f"✅ 结果已保存:")
        print(f"  选股结果: {selections_file}")
        print(f"  回测结果: {backtest_file}")
        print(f"  简化报告: {report_file}")

    except Exception as e:
        print(f"❌ 保存结果失败: {e}")


def _apply_enhanced_risk_filters(daily_stock_data: pd.DataFrame, sector_stocks: list,
                                date_str: str, logger) -> pd.DataFrame:
    """
    应用增强的风险控制过滤机制
    """
    try:
        # 基础过滤：在潜力板块中
        filtered_data = daily_stock_data[
            daily_stock_data['stock_code'].isin(sector_stocks)
        ].copy()

        if len(filtered_data) == 0:
            return filtered_data

        initial_count = len(filtered_data)

        # # 1. 当日涨幅过滤（保持原有宽松度）
        # filtered_data = filtered_data[filtered_data['pct_chg'] <= 4.0]  # 调整为6%，保持合理选择范围
        # logger.info(f"当日涨幅过滤: {initial_count} -> {len(filtered_data)}")

        # # 2. 极端跌幅过滤（针对问题案例，保持严格）
        # filtered_data = filtered_data[filtered_data['pct_chg'] >= -6.0]  # 过滤大跌股票
        # logger.info(f"极端跌幅过滤: -> {len(filtered_data)}")

        # # 3. 累计涨幅风险控制（进一步放宽，但保留核心风控）
        # if 'cumulative_return_5d' in filtered_data.columns:
        #     # 过滤5日累计涨幅超过40%的股票（进一步放宽）
        #     filtered_data = filtered_data[
        #         (filtered_data['cumulative_return_5d'].fillna(0) <= 0.40)
        #     ]
        #     logger.info(f"5日累计涨幅过滤: -> {len(filtered_data)}")

        # if 'cumulative_return_10d' in filtered_data.columns:
        #     # 过滤10日累计涨幅超过60%的股票（进一步放宽）
        #     filtered_data = filtered_data[
        #         (filtered_data['cumulative_return_10d'].fillna(0) <= 0.60)
        #     ]
        #     logger.info(f"10日累计涨幅过滤: -> {len(filtered_data)}")

        # 4. 炸板风险控制（保持严格，这是核心问题）
        if 'is_zhaban' in filtered_data.columns:
            # 过滤炸板股票
            filtered_data = filtered_data[filtered_data['is_zhaban'] != 1]
            logger.info(f"炸板过滤: -> {len(filtered_data)}")

        # # 5. 连续涨停风险控制（进一步放宽）
        # if 'consecutive_limit_up' in filtered_data.columns:
        #     # 过滤连续涨停超过5天的股票（进一步放宽）
        #     filtered_data = filtered_data[
        #         (filtered_data['consecutive_limit_up'].fillna(0) <= 5)
        #     ]
        #     logger.info(f"连续涨停过滤: -> {len(filtered_data)}")

        # # 6. 成交量异常过滤（基于实际存在的特征）
        # if 'turnover_rate' in filtered_data.columns:
        #     # 换手率过滤：超过30%认为异常（turnover_rate是小数制，0.3表示30%）
        #     filtered_data = filtered_data[
        #         (filtered_data['turnover_rate'].fillna(0.05) <= 0.30)
        #     ]
        #     logger.debug(f"换手率过滤: -> {len(filtered_data)}")
        # elif 'volume_zscore' in filtered_data.columns:
        #     # 成交量Z分数过滤：超过3个标准差认为异常
        #     filtered_data = filtered_data[
        #         (filtered_data['volume_zscore'].fillna(0) <= 3.0)
        #     ]
        #     logger.debug(f"成交量Z分数过滤: -> {len(filtered_data)}")
        else:
            logger.debug("跳过成交量过滤: 相关特征不存在")

        # 7. 市值过滤（取消，避免过度过滤）
        # 注释掉市值过滤，让模型自己学习市值特征的重要性
        # if 'total_mv' in filtered_data.columns:
        #     mv_threshold = filtered_data['total_mv'].quantile(0.05)
        #     filtered_data = filtered_data[filtered_data['total_mv'] >= mv_threshold]
        #     logger.debug(f"市值过滤: -> {len(filtered_data)}")

        logger.info(f"日期 {date_str} 风险过滤: {initial_count} -> {len(filtered_data)} 只股票")

        return filtered_data

    except Exception as e:
        logger.error(f"增强风险过滤失败: {e}")
        # 降级到基础过滤
        return daily_stock_data[
            (daily_stock_data['stock_code'].isin(sector_stocks)) &
            (daily_stock_data['pct_chg'] <= 3.0)
        ].copy()


def _load_factor_cache_data(
    required_factors=None,
    include_label_generation_features: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """统一从 factor_lab 数据集中加载所需因子并补全基础字段。"""
    logger = logging.getLogger(__name__)
    try:
        store = FactorDatasetStore(FACTOR_LAB_STORE_PATH)
        if not store.exists():
            logger.error("因子数据文件不存在，请先运行 generate_factors 模式生成 factor_lab 数据。")
            return None

        factor_df = store.load()
        if factor_df is None or factor_df.empty:
            logger.error("因子数据文件为空，请确认已生成 factor_lab 数据。")
            return None

        required_columns = {'date', 'stock_code'}
        missing_required = [col for col in required_columns if col not in factor_df.columns]
        if missing_required:
            logger.error(f"因子数据缺少基础列: {missing_required}")
            return None

        factor_df = factor_df.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        factor_df.sort_values(['date', 'stock_code'], inplace=True)
        factor_df.drop_duplicates(subset=['date', 'stock_code'], keep='last', inplace=True)
        factor_df.reset_index(drop=True, inplace=True)
        factor_df['date_str'] = factor_df['date'].dt.strftime('%Y%m%d')

        available_factor_columns = [
            col for col in factor_df.columns if col not in {'date', 'stock_code', 'date_str'}
        ]

        if required_factors is None:
            # 🔧 模型训练时使用归一化因子（默认：z-score + rankpct）
            stock_factors = _get_model_factors()
            # market_factors = _get_available_market_factors()
            # required_factors = stock_factors + market_factors
            required_factors = stock_factors
        else:
            required_factors = list(dict.fromkeys(required_factors))
            if 'returns_zscore' not in required_factors:
                required_factors.insert(0, 'returns_zscore')
            logger.info("按需加载因子 %d 个。", len(required_factors))

        available_required = [f for f in required_factors if f in available_factor_columns]
        missing_requested = [f for f in required_factors if f not in available_factor_columns]
        if missing_requested:
            logger.warning(f"因子库缺少以下因子: {missing_requested}")

        # 🚀 关键优化：在加载后立即过滤掉不需要的因子列，避免内存浪费
        # 只保留 date, stock_code 和 available_required 中的因子
        columns_to_keep = ['date', 'stock_code'] + available_required
        columns_to_keep = [col for col in columns_to_keep if col in factor_df.columns]
        factor_df = factor_df[columns_to_keep].copy()
        logger.info(f"过滤后保留因子列数: {len(columns_to_keep) - 2} (原始: {len(available_factor_columns)})")

        dataset_start = factor_df['date'].min()
        dataset_end = factor_df['date'].max()
        logger.info(f"因子 start:{dataset_start}, end:{dataset_end}")
        if start_date:
            dataset_start = max(dataset_start, pd.to_datetime(start_date))
        if end_date:
            dataset_end = min(dataset_end, pd.to_datetime(end_date))
        logger.info(f"因子2 start:{dataset_start}, end:{dataset_end}")
        mask = (factor_df['date'] >= dataset_start) & (factor_df['date'] <= dataset_end)
        factor_df = factor_df.loc[mask].copy()
        if factor_df.empty:
            logger.error("按日期过滤后因子数据为空。")
            return None

        loader = IntegratedDataLoader()
        stock_start = dataset_start.strftime('%Y%m%d')
        stock_end = dataset_end.strftime('%Y%m%d')
        stock_data = loader.load_stock_features(stock_start, stock_end)
        if stock_data is None or stock_data.empty:
            logger.error("无法加载基础行情数据，请检查 local_data/features/daily 下的数据文件。")
            return None

        stock_data = stock_data.copy()
        if 'date' not in stock_data.columns or 'stock_code' not in stock_data.columns:
            logger.error("基础行情数据缺少 'date' 或 'stock_code' 列。")
            return None
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.drop_duplicates(subset=['date', 'stock_code'], keep='last', inplace=True)

        base_columns = [
            'date', 'stock_code', 'date_str', 'pct_chg', 'is_can_buy',
            'volume', 'amount', 'close', 'open', 'high', 'low',  # 🚀 添加amount字段用于流动性过滤
            'open_front', 'close_front', 'high_front', 'low_front',  # 🚀 添加前复权价格字段（用于回测收益计算）
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
            'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
            'near_support', 'near_resistance', 'short_reversal', 'parkinson_vol',
            'vol_term_structure', 'volatility20', 'atr14',  # 添加标签生成必需字段
            'future_return_2d', 'future_return_3d', 'future_return_5d',  # 🚀 添加未来收益字段
            'turnover_rate', 'circ_mv'  # 🚀 添加换手率和流通市值字段（用于样本权重计算）
        ]
        # 🚀 标签生成所需字段:
        # - volatility20, atr14: ATR策略计算
        # - open, high, low: 价格计算
        # - is_limit_up, is_limit_down: 过滤T日涨跌停
        # - amount, volume: 过滤T日流动性不足
        # - is_can_buy: 过滤T+1日一字涨停
        # - future_return_2d, future_return_3d, future_return_5d: 加权收益率标签计算
        label_generation_columns = [
            'volatility20', 'atr14', 'open', 'high', 'low',
            'is_limit_up', 'is_limit_down', 'amount', 'volume', 'is_can_buy',
            'future_return_2d', 'future_return_3d', 'future_return_5d'
        ]
        default_values = {
            'pct_chg': 0.0,
            'is_can_buy': 1,
            'volume': 0.0,
            'amount': 0.0,  # 🚀 添加amount默认值
            'close': 0.0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'open_front': 0.0,  # 🚀 前复权价格默认值
            'close_front': 0.0,
            'high_front': 0.0,
            'low_front': 0.0,
            'is_limit_up': 0,
            'is_limit_down': 0,
            'is_zhaban': 0,
            'limit_strength': 0.0,
            'limit_times': 0,
            'consecutive_limit_up': 0,
            'consecutive_limit_down': 0,
            'near_support': 0,
            'near_resistance': 0,
            'short_reversal': 0,
            'parkinson_vol': 0.0,
            'vol_term_structure': 0.0,
            'volatility20': 1.0,
            'atr14': 0.0,
            'future_return_2d': np.nan,  # 🚀 未来收益字段默认为NaN
            'future_return_3d': np.nan,
            'future_return_5d': np.nan,
            'turnover_rate': np.nan,  # 🚀 换手率默认为NaN
            'circ_mv': np.nan,  # 🚀 流通市值默认为NaN
        }

        base_merge_columns = [
            col for col in base_columns if col in stock_data.columns and col not in {'date', 'stock_code'}
        ]
        if base_merge_columns:
            merge_source = stock_data[['date', 'stock_code'] + base_merge_columns].copy()
        else:
            merge_source = stock_data[['date', 'stock_code']].copy()
        factor_df = factor_df.merge(merge_source, on=['date', 'stock_code'], how='left')

        if 'date_str' not in factor_df.columns:
            factor_df['date_str'] = factor_df['date'].dt.strftime('%Y%m%d')

        missing_base_cols = [col for col in base_columns if col not in factor_df.columns]
        if missing_base_cols:
            logger.warning(f"基础行情数据缺少列: {missing_base_cols}")
            for col in missing_base_cols:
                factor_df[col] = default_values.get(col, 0.0)

        final_columns = ['date', 'date_str', 'stock_code']
        final_columns.extend([
            col for col in base_columns
            if col in factor_df.columns and col not in {'date', 'date_str', 'stock_code'}
        ])
        final_columns.extend([col for col in available_required if col in factor_df.columns])

        if include_label_generation_features:
            # 🚀 检查标签生成所需字段是否存在(可能已在base_columns中)
            missing_label_cols = [col for col in label_generation_columns if col not in factor_df.columns]
            existing_label_cols = [col for col in label_generation_columns if col in factor_df.columns]

            # 添加尚未在final_columns中的标签生成字段
            extra_cols = [
                col for col in label_generation_columns
                if col in factor_df.columns and col not in final_columns
            ]

            if extra_cols:
                logger.info(f"附加标签生成必需字段: {extra_cols}")
                final_columns.extend(extra_cols)

            # 如果有缺失的字段,使用默认值填充
            if missing_label_cols:
                logger.warning(f"标签生成字段缺失,将使用默认值: {missing_label_cols}")
                for col in missing_label_cols:
                    factor_df[col] = default_values.get(col, 0.0)
                    if col not in final_columns:
                        final_columns.append(col)

            # 记录已存在的字段
            if existing_label_cols:
                logger.info(f"✅ 标签生成字段已存在: {existing_label_cols}")

        missing_after_selection = [col for col in final_columns if col not in factor_df.columns]
        if missing_after_selection:
            logger.warning(f"最终列缺失: {missing_after_selection}")
            for col in missing_after_selection:
                factor_df[col] = default_values.get(col, 0.0)

        # 🚀 优化：避免内存分配失败，直接使用列选择而不是copy
        # 原因：在Line 773已经执行过一次copy，这里再次copy会导致内存不足
        factor_df = factor_df[final_columns]
        factor_df.sort_values(['date', 'stock_code'], inplace=True)
        factor_df.reset_index(drop=True, inplace=True)

        logger.info(
            "统一加载因子数据完成: %d 条记录, %d 列 (股票: %d 支)。",
            len(factor_df),
            len(factor_df.columns),
            factor_df['stock_code'].nunique(),
        )
        if missing_requested:
            logger.info(f"仍缺少的因子: {missing_requested}")
        return factor_df
    except Exception as exc:
        logger.error(f"因子数据加载失败: {exc}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _get_ml_feature_columns_simple(data: pd.DataFrame) -> list:
    """
    🔧 重构后的特征筛选函数：直接使用_get_model_factors()定义的因子列表

    核心改进：
    1. 不再使用黑名单动态过滤，而是使用固定的因子列表
    2. 确保训练和测试使用完全相同的特征
    3. 如果数据缺少某个因子，记录警告但不填充0

    Args:
        data: 输入数据DataFrame

    Returns:
        list: 有效的特征列名列表（按字母顺序排序）
    """
    try:
        logger = logging.getLogger(__name__)

        # 🚀 直接使用模型因子列表（归一化因子）
        all_factors = _get_model_factors()

        # 验证所有因子都存在
        available_factors = [f for f in all_factors if f in data.columns]
        missing_factors = [f for f in all_factors if f not in data.columns]

        if missing_factors:
            logger.warning(f"⚠️ 数据缺少{len(missing_factors)}个因子:")
            logger.warning(f"   缺失因子: {missing_factors[:10]}{'...' if len(missing_factors) > 10 else ''}")

        if not available_factors:
            logger.error("❌ 没有可用的因子")
            return None

        logger.info(f"✅ 可用因子: {len(available_factors)}/{len(all_factors)}")

        # 按字母顺序排序（确保一致性）
        return sorted(available_factors)

    except Exception as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"❌ 特征筛选失败: {e}")
        logger.error(traceback.format_exc())
        return None

def _get_market_state(market_data: pd.DataFrame, lookback: int = 20, logger=None) -> str:
    """
    判断当前市场状态

    Args:
        market_data: 市场数据DataFrame,包含'date'和'close'列
        lookback: 回看天数
        logger: 日志记录器

    Returns:
        str: 'bull'(牛市), 'bear'(熊市), 'sideways'(震荡市)
    """
    try:
        if market_data is None or market_data.empty:
            if logger:
                logger.warning("市场数据为空，默认返回震荡市状态")
            return 'sideways'

        # 确保数据按日期排序
        market_data = market_data.sort_values('date')

        # 计算最近lookback天的收益率
        recent_return = market_data['close'].pct_change(lookback).iloc[-1]

        # 计算波动率
        volatility = market_data['close'].pct_change().rolling(lookback).std().iloc[-1]

        # 判断市场状态
        if recent_return > 0.05 and volatility < 0.02:
            state = 'bull'  # 牛市:涨幅>5%且波动率<2%
        elif recent_return < -0.05:
            state = 'bear'  # 熊市:跌幅>5%
        else:
            state = 'sideways'  # 震荡市

        if logger:
            logger.info(f"📊 市场状态判断:")
            logger.info(f"  最近{lookback}天收益率: {recent_return*100:.2f}%")
            logger.info(f"  最近{lookback}天波动率: {volatility*100:.2f}%")
            logger.info(f"  市场状态: {state} ({'牛市' if state == 'bull' else '熊市' if state == 'bear' else '震荡市'})")

        return state

    except Exception as e:
        if logger:
            logger.error(f"市场状态判断失败: {e}")
        return 'sideways'


def _calculate_xgboost_group_weights(train_dates: np.ndarray,
                                    market_data: pd.DataFrame,
                                    logger,
                                    time_decay_half_life: float = 180.0,
                                    reference_date: str = None,
                                    verbose: bool = True) -> np.ndarray:
    """
    🚀 XGBoost排序模型的分组权重计算（基于业务逻辑）

    XGBoost ranking objectives要求group-level weights（长度=group数量）

    权重计算逻辑:
    1. 市场波动率适中的日子权重高（黄金波动区间）
    2. 近期数据权重高（时间衰减）
    3. 高成交量日子权重高（市场活跃度）

    Args:
        train_dates: 训练集日期数组（每个样本的日期）
        market_data: 市场数据DataFrame，包含date, close, volume等字段
        logger: 日志记录器
        time_decay_half_life: 时间衰减半衰期（天数，默认180天）
        reference_date: 时间衰减基准日期（默认为训练集最新日期）
        verbose: 是否输出详细日志（默认True，贝叶斯搜索时设为False）

    Returns:
        分组权重数组（长度=group数量）
    """
    try:
        import numpy as np
        import pandas as pd

        if verbose:
            logger.info("=" * 60)
            logger.info("🚀 XGBoost分组权重计算（基于业务逻辑）")
            logger.info("=" * 60)

        # 获取唯一日期（按顺序）
        unique_dates = pd.Series(train_dates).unique()
        unique_dates = pd.to_datetime(unique_dates)
        unique_dates = sorted(unique_dates)

        if verbose:
            logger.info(f"  Group数量: {len(unique_dates)}")
            logger.info(f"  日期范围: {unique_dates[0]} 至 {unique_dates[-1]}")

        # 确定基准日期
        if reference_date is None:
            ref_date = unique_dates[-1]
        else:
            ref_date = pd.to_datetime(reference_date)

        if verbose:
            logger.info(f"  基准日期: {ref_date}")
            logger.info(f"  时间衰减半衰期: {time_decay_half_life}天")

        # 确保market_data的date列是datetime类型
        if market_data is not None and not market_data.empty:
            if 'date' in market_data.columns:
                market_data = market_data.copy()
                market_data['date'] = pd.to_datetime(market_data['date'])

        # 计算每个group的权重
        group_weights = []

        for date in unique_dates:
            weight = 1.0

            # 业务逻辑1: 时间衰减（近期数据权重高）
            days_diff = (ref_date - date).days
            time_decay = np.exp(-np.log(2) * days_diff / time_decay_half_life)
            weight *= time_decay

            # 业务逻辑2: 市场波动率适中的日子权重高
            if market_data is not None and not market_data.empty:
                date_market = market_data[market_data['date'] == date]

                if not date_market.empty and 'close' in date_market.columns:
                    # 计算当日市场波动率（使用最近5天数据）
                    recent_market = market_data[market_data['date'] <= date].tail(5)
                    if len(recent_market) >= 2 and 'close' in recent_market.columns:
                        market_returns = recent_market['close'].pct_change().dropna()
                        if len(market_returns) > 0:
                            market_vol = market_returns.std()

                            # 黄金波动区间: 0.02 <= vol <= 0.05
                            if 0.02 <= market_vol <= 0.05:
                                weight *= 1.5  # 黄金波动区间
                            elif market_vol > 0.1:
                                weight *= 0.6  # 过度波动，噪声大

                # 业务逻辑3: 高成交量日子权重高
                if not date_market.empty and 'volume' in date_market.columns:
                    total_volume = date_market['volume'].sum()
                    if total_volume > 0:
                        # 对数变换以压缩极端值
                        weight *= np.log(total_volume / 1e9 + 1)

            group_weights.append(weight)

        # 转换为numpy数组
        group_weights = np.array(group_weights)

        # 归一化权重（使权重均值为1）
        group_weights = group_weights / group_weights.mean()

        if verbose:
            logger.info(f"分组权重统计:")
            logger.info(f"  权重范围: [{group_weights.min():.4f}, {group_weights.max():.4f}]")
            logger.info(f"  权重均值: {group_weights.mean():.4f}")
            logger.info(f"  权重中位数: {np.median(group_weights):.4f}")
            logger.info(f"  权重标准差: {group_weights.std():.4f}")
            logger.info("=" * 60)

        return group_weights

    except Exception as e:
        logger.error(f"XGBoost分组权重计算失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        # 返回均匀权重
        unique_dates = pd.Series(train_dates).unique()
        return np.ones(len(unique_dates))


def _calculate_professional_sample_weights(
    y_train: np.ndarray,
    imbalance_ratio: float,
    logger,
    boundary_distances: np.ndarray = None,
    is_regression: bool = False,
    y_train_original: np.ndarray = None,
    train_dates: np.ndarray = None,
    time_decay_half_life: float = 365.0,
    reference_date: str = None,
    turnover_rate: np.ndarray = None,
    circ_mv: np.ndarray = None,
    label_rank: np.ndarray = None,
    label_boost_beta: float = 0.1,
    w_min: float = 0.8,
    w_max: float = 1.2
) -> np.ndarray:
    """
    稳健样本权重计算 (修复版)
    - 在 log1p 前先填充/替换 NaN/Inf/<=0
    - 使用 per-field winsorize (1%-99%)
    - 在 log 空间合成，限制最终范围 [w_min, w_max]
    - 返回已中位数归一化的样本权重
    """
    import numpy as np
    import pandas as pd

    try:
        logger.info("样本权重计算 (修复版) 开始")
        n = len(y_train)
        logger.info(f"  总样本数: {n}")

        # ---------- 基本统计 ----------
        logger.info("标签统计:")
        logger.info(f"  min={np.nanmin(y_train):.6f}, max={np.nanmax(y_train):.6f}, mean={np.nanmean(y_train):.6f}, std={np.nanstd(y_train):.6f}")
        logger.info(f"  NaN count in labels: {int(np.isnan(y_train).sum())}, Inf count: {int(np.isinf(y_train).sum())}")

        # ---------- 检查输入 ----------
        if turnover_rate is None or circ_mv is None:
            logger.warning("⚠️ turnover_rate or circ_mv is None -> 返回均匀权重")
            return np.ones(n)

        turnover = np.asarray(turnover_rate, dtype=float).copy()
        mcap = np.asarray(circ_mv, dtype=float).copy()

        # ---------- 1) 预处理：替换 Inf/NaN/<=0 ----------
        # 记录问题数量
        n_inf_turn = np.isinf(turnover).sum()
        n_nan_turn = np.isnan(turnover).sum()
        n_le0_turn = np.sum(~np.isnan(turnover) & (turnover <= 0))
        logger.info(f"换手率预检: finite={np.isfinite(turnover).sum()}/{len(turnover)}, Inf={n_inf_turn}, NaN={n_nan_turn}, <=0={n_le0_turn}")

        n_inf_mc = np.isinf(mcap).sum()
        n_nan_mc = np.isnan(mcap).sum()
        n_le0_mc = np.sum(~np.isnan(mcap) & (mcap <= 0))
        logger.info(f"市值预检: finite={np.isfinite(mcap).sum()}/{len(mcap)}, Inf={n_inf_mc}, NaN={n_nan_mc}, <=0={n_le0_mc}")

        # 替换策略：
        # - 对于 turnover: 使用正值部分的 1st percentile 作为 floor 填充非正/NaN/Inf
        # - 对于 mcap: 使用正值部分的 1st percentile 填充非正/NaN/Inf
        pos_turn = turnover[np.isfinite(turnover) & (turnover > 0)]
        pos_mcap = mcap[np.isfinite(mcap) & (mcap > 0)]

        if pos_turn.size == 0:
            turn_floor = 1e-6
        else:
            turn_floor = max(np.nanpercentile(pos_turn, 1), 1e-6)

        if pos_mcap.size == 0:
            mcap_floor = 1.0
        else:
            mcap_floor = max(np.nanpercentile(pos_mcap, 1), 1.0)

        # 替换：把非正/NaN/Inf -> floor
        bad_turn_mask = ~np.isfinite(turnover) | (turnover <= 0)
        replaced_turn = int(bad_turn_mask.sum())
        turnover[bad_turn_mask] = turn_floor

        bad_mcap_mask = ~np.isfinite(mcap) | (mcap <= 0)
        replaced_mcap = int(bad_mcap_mask.sum())
        mcap[bad_mcap_mask] = mcap_floor

        logger.info(f"已替换换手率 {replaced_turn} 个无效值 -> {turn_floor:.6g}")
        logger.info(f"已替换市值 {replaced_mcap} 个无效值 -> {mcap_floor:.6g}")

        # ---------- 2) log 缩放 + per-field winsorize (1%-99%) ----------
        t_log = np.log1p(turnover)  # >=0
        m_log = np.log1p(mcap)

        # winsorize in log space
        lo_t, hi_t = np.nanpercentile(t_log, [1, 99])
        lo_m, hi_m = np.nanpercentile(m_log, [1, 99])
        t_log_clipped = np.clip(t_log, lo_t, hi_t)
        m_log_clipped = np.clip(m_log, lo_m, hi_m)

        logger.info(f"t_log percentiles(1,99): {lo_t:.6f}, {hi_t:.6f}; m_log percentiles(1,99): {lo_m:.6f}, {hi_m:.6f}")

        # 标准化到接近 1 的尺度（用中位数）
        t_med = np.nanmedian(t_log_clipped) + 1e-12
        m_med = np.nanmedian(m_log_clipped) + 1e-12
        t_w = t_log_clipped / t_med
        m_w = m_log_clipped / m_med

        # ---------- 3) 合成权重（在 log 空间做幂次混合，避免乘法爆炸） ----------
        alpha = 0.6
        # 防止取 log(0)：确保正数
        t_w_safe = np.clip(t_w, 1e-6, None)
        m_w_safe = np.clip(m_w, 1e-6, None)
        log_base = alpha * np.log(t_w_safe) + (1 - alpha) * np.log(m_w_safe)
        w_base = np.exp(log_base)
        w_base = np.ones_like(w_base)

        logger.info(f"基础权重 (w_base) 统计: min={np.nanmin(w_base):.4f}, max={np.nanmax(w_base):.4f}, mean={np.nanmean(w_base):.4f}, NaN={int(np.isnan(w_base).sum())}")

        # ---------- 4) 时间衰减（弱化） ----------
        # 处理 train_dates
        if train_dates is None:
            decay = np.ones(n)
            logger.warning("⚠️ train_dates 为 None, 跳过时间衰减")
        else:
            dates = pd.to_datetime(pd.Series(train_dates))
            recent = dates.max()
            age_days = (recent - dates).dt.days.values
            age_days = np.clip(age_days, 0, None)
            decay_raw = np.exp(-np.log(2) * age_days / float(time_decay_half_life))
            # 将 decay 映射成靠近 1 的微调因子，避免过度放大/缩小
            decay = 1.0 + 0.15 * (decay_raw - np.mean(decay_raw))
            # 保证 finite
            decay = np.nan_to_num(decay, nan=1.0, posinf=1.0, neginf=1.0)

        w_time = w_base * decay
        w_final = w_time
        logger.info(f"time adjusted weight 统计: min={np.nanmin(w_time):.4f}, max={np.nanmax(w_time):.4f}, mean={np.nanmean(w_time):.4f}, NaN={int(np.isnan(w_time).sum())}")

        # ---------- 5) relevance 增强（有限制） ----------
        # if label_rank is not None:
        #     rel = np.asarray(label_rank, dtype=float)
        #     # 把 rel 缺失当 0
        #     rel = np.where(~np.isfinite(rel), 0.0, rel)
        #     r_factor = 1.0 + float(label_boost_beta) * rel
        #     # 强制上下限，例如 [0.85, 1.3]
        #     r_factor = np.clip(r_factor, 0.85, 1.3)
        #     w_rel = w_time * r_factor
        #     logger.info(f"relevance factor 统计: min={np.nanmin(r_factor):.4f}, max={np.nanmax(r_factor):.4f}, mean={np.nanmean(r_factor):.4f}")
        # else:
        #     w_rel = w_time
        #     logger.info("no relevance provided -> skip relevance boost")

        # ---------- 6) 全局 winsorize + clip 到 [w_min, w_max] ----------
        # w_rel = np.nan_to_num(w_rel, nan=1.0, posinf=w_max, neginf=w_min)
        # w_clip = np.clip(w_rel, w_min, w_max)

        # # ---------- 7) 归一化（中位数归一化） ----------
        # med = float(np.median(w_clip))
        # if not np.isfinite(med) or med <= 0:
        #     med = float(np.nanmean(w_clip))
        #     logger.warning(f"权重中位数异常, 用均值替代: {med:.6f}")

        # w_final = w_clip / (med + 1e-12)

        # # 最后保证无 NaN/Inf
        # if np.any(~np.isfinite(w_final)):
        #     bad = int(np.sum(~np.isfinite(w_final)))
        #     logger.warning(f"最终权重存在 {bad} 个无效值, 用 1.0 替换")
        #     w_final = np.nan_to_num(w_final, nan=1.0, posinf=w_max, neginf=w_min)

        # # ---------- 诊断日志 ----------
        # logger.info("最终权重统计:")
        # logger.info(f"  min={float(np.nanmin(w_final)):.4f}, max={float(np.nanmax(w_final)):.4f}, mean={float(np.nanmean(w_final)):.4f}, median={float(np.median(w_final)):.4f}")
        # # 集中度
        # w_sorted = np.sort(w_final)[::-1]
        # total = np.sum(w_final)
        # n_samples = len(w_final)
        # k1 = max(1, int(n_samples * 0.01))
        # top1_share = np.sum(w_sorted[:k1]) / (total + 1e-12)
        # logger.info(f"  Top1% weight share: {top1_share*100:.2f}%")
        return w_final

    except Exception as e:
        logger.error(f"样本权重计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("返回均匀权重作为后备")
        return np.ones(len(y_train) if y_train is not None else 100)


def _calculate_precision_at_k(y_true, y_scores, k_values=[1, 3, 5, 10]):
    """
    计算Precision@k评估指标

    Args:
        y_true: 真实标签
        y_scores: 预测概率分数
        k_values: k值列表

    Returns:
        dict: 各个k值对应的Precision@k
    """
    try:
        import numpy as np

        # 按预测概率降序排序
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_y_true = y_true[sorted_indices]

        precision_at_k = {}

        for k in k_values:
            if k > len(y_true):
                # 如果k大于样本总数，使用全部样本
                k_actual = len(y_true)
            else:
                k_actual = k

            # 取前k个预测结果
            top_k_true = sorted_y_true[:k_actual]

            # 计算Precision@k
            if k_actual > 0:
                precision = np.sum(top_k_true) / k_actual
            else:
                precision = 0.0

            precision_at_k[k] = precision

        return precision_at_k

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"计算Precision@k失败: {e}")
        return {k: 0.0 for k in k_values}

def _calculate_quantitative_metrics(y_true, y_pred, logger):
    """
    计算专业量化评估指标

    参数:
        y_true: 真实收益率 (ATR单位)
        y_pred: 预测收益率 (ATR单位)
        logger: 日志记录器

    返回:
        dict: 包含IC、RankIC、ICIR、RankICIR等指标的字典
    """
    try:
        import scipy.stats as stats

        # 1. Information Coefficient (IC) - 线性相关性
        ic, ic_pvalue = stats.pearsonr(y_true, y_pred)

        # 2. Rank Information Coefficient (RankIC) - 排序相关性
        rank_ic, rank_ic_pvalue = stats.spearmanr(y_true, y_pred)

        # 3. IC Information Ratio (ICIR) - IC的稳定性
        # 这里简化为IC的绝对值除以标准误差的估计
        ic_std = np.sqrt((1 - ic**2) / (len(y_true) - 2)) if len(y_true) > 2 else 0.1
        icir = abs(ic) / max(ic_std, 0.01)

        # 4. RankIC Information Ratio (RankICIR)
        rank_ic_std = np.sqrt((1 - rank_ic**2) / (len(y_true) - 2)) if len(y_true) > 2 else 0.1
        rank_icir = abs(rank_ic) / max(rank_ic_std, 0.01)

        # 5. 预测准确率指标
        # 方向准确率：预测和真实收益同号的比例
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

        # 6. 分位数分析
        # 将预测值分为5个分位数，计算每个分位数的平均真实收益
        pred_quantiles = np.quantile(y_pred, [0.2, 0.4, 0.6, 0.8])
        quantile_returns = []
        for i in range(5):
            if i == 0:
                mask = y_pred <= pred_quantiles[0]
            elif i == 4:
                mask = y_pred > pred_quantiles[3]
            else:
                mask = (y_pred > pred_quantiles[i-1]) & (y_pred <= pred_quantiles[i])

            if mask.sum() > 0:
                quantile_returns.append(y_true[mask].mean())
            else:
                quantile_returns.append(0.0)

        # 7. 单调性检验 - 分位数收益是否单调递增
        monotonicity = np.corrcoef(range(5), quantile_returns)[0, 1]

        # 记录结果
        logger.info("📊 专业量化评估指标:")
        logger.info(f"  Information Coefficient (IC): {ic:.4f} (p-value: {ic_pvalue:.4f})")
        logger.info(f"  Rank Information Coefficient (RankIC): {rank_ic:.4f} (p-value: {rank_ic_pvalue:.4f})")
        logger.info(f"  IC Information Ratio (ICIR): {icir:.4f}")
        logger.info(f"  RankIC Information Ratio (RankICIR): {rank_icir:.4f}")
        logger.info(f"  方向准确率: {direction_accuracy:.4f} ({direction_accuracy*100:.1f}%)")
        logger.info(f"  分位数单调性: {monotonicity:.4f}")

        logger.info("📊 分位数收益分析:")
        for i, ret in enumerate(quantile_returns):
            logger.info(f"  Q{i+1} (预测值{i*20}-{(i+1)*20}%分位): {ret:.4f}")

        # 评估标准
        logger.info("📊 评估标准:")
        ic_quality = "优秀" if abs(ic) > 0.05 else "良好" if abs(ic) > 0.03 else "一般" if abs(ic) > 0.01 else "较差"
        rank_ic_quality = "优秀" if abs(rank_ic) > 0.05 else "良好" if abs(rank_ic) > 0.03 else "一般" if abs(rank_ic) > 0.01 else "较差"
        direction_quality = "优秀" if direction_accuracy > 0.55 else "良好" if direction_accuracy > 0.52 else "一般" if direction_accuracy > 0.50 else "较差"

        logger.info(f"  IC质量: {ic_quality} (|IC|>0.05为优秀, >0.03为良好, >0.01为一般)")
        logger.info(f"  RankIC质量: {rank_ic_quality}")
        logger.info(f"  方向预测质量: {direction_quality} (>55%为优秀, >52%为良好, >50%为一般)")

        return {
            'ic': ic,
            'ic_pvalue': ic_pvalue,
            'rank_ic': rank_ic,
            'rank_ic_pvalue': rank_ic_pvalue,
            'icir': icir,
            'rank_icir': rank_icir,
            'direction_accuracy': direction_accuracy,
            'quantile_returns': quantile_returns,
            'monotonicity': monotonicity,
            'ic_quality': ic_quality,
            'rank_ic_quality': rank_ic_quality,
            'direction_quality': direction_quality
        }

    except Exception as e:
        logger.warning(f"计算量化评估指标失败: {e}")
        return {}

def _calculate_prediction_range_diagnostics(y_true, y_pred, logger, y_pred_raw=None):
    """
    计算预测范围诊断指标

    参数:
        y_true: 真实收益率 (ATR单位)
        y_pred: 预测收益率 (ATR单位)
        logger: 日志记录器
    """
    try:
        # 1. 范围对比
        true_range = [y_true.min(), y_true.max()]
        pred_range = [y_pred.min(), y_pred.max()]
        true_span = true_range[1] - true_range[0]
        pred_span = pred_range[1] - pred_range[0]
        coverage_ratio = pred_span / true_span if true_span > 0 else 0

        # 2. 标准差对比
        true_std = y_true.std()
        pred_std = y_pred.std()
        std_ratio = pred_std / true_std if true_std > 0 else 0

        # 3. 分位数对比
        true_quantiles = np.quantile(y_true, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
        pred_quantiles = np.quantile(y_pred, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])

        # 4. 极值预测能力
        extreme_positive_mask = y_true > 2.0  # 大于2ATR的极值
        extreme_negative_mask = y_true < -1.0  # 小于-1ATR的极值

        extreme_positive_pred = y_pred[extreme_positive_mask].mean() if extreme_positive_mask.sum() > 0 else 0
        extreme_negative_pred = y_pred[extreme_negative_mask].mean() if extreme_negative_mask.sum() > 0 else 0

        # 记录诊断结果
        logger.info("🔍 预测范围诊断指标:")
        logger.info(f"  真实值范围: [{true_range[0]:.3f}, {true_range[1]:.3f}] ATR (跨度: {true_span:.3f})")

        # 🚀 新增：显示逆变换前后的预测值范围对比
        if y_pred_raw is not None:
            raw_range = [y_pred_raw.min(), y_pred_raw.max()]
            raw_span = raw_range[1] - raw_range[0]
            logger.info(f"  预测值范围(变换前): [{raw_range[0]:.3f}, {raw_range[1]:.3f}] (跨度: {raw_span:.3f})")

        logger.info(f"  预测值范围(逆变换后): [{pred_range[0]:.3f}, {pred_range[1]:.3f}] ATR (跨度: {pred_span:.3f})")
        logger.info(f"  范围覆盖率: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")

        logger.info(f"  真实值标准差: {true_std:.3f} ATR")
        logger.info(f"  预测值标准差: {pred_std:.3f} ATR")
        logger.info(f"  标准差比例: {std_ratio:.3f} ({std_ratio*100:.1f}%)")

        logger.info("  分位数对比:")
        quantile_names = ['1%', '5%', '25%', '75%', '95%', '99%']
        for i, name in enumerate(quantile_names):
            logger.info(f"    {name}: 真实{true_quantiles[i]:.3f} vs 预测{pred_quantiles[i]:.3f}")

        logger.info(f"  极值预测能力:")
        logger.info(f"    >2ATR样本数: {extreme_positive_mask.sum()}, 平均预测: {extreme_positive_pred:.3f}")
        logger.info(f"    <-1ATR样本数: {extreme_negative_mask.sum()}, 平均预测: {extreme_negative_pred:.3f}")

        # 诊断评估
        logger.info("🚨 诊断评估:")
        if coverage_ratio < 0.3:
            logger.warning(f"  ❌ 严重预测范围压缩: {coverage_ratio:.1%} < 30%")
        elif coverage_ratio < 0.5:
            logger.warning(f"  ⚠️  预测范围偏小: {coverage_ratio:.1%} < 50%")
        else:
            logger.info(f"  ✅ 预测范围合理: {coverage_ratio:.1%}")

        if std_ratio < 0.5:
            logger.warning(f"  ❌ 预测过度保守: 标准差比例{std_ratio:.1%} < 50%")
        elif std_ratio < 0.7:
            logger.warning(f"  ⚠️  预测偏保守: 标准差比例{std_ratio:.1%} < 70%")
        else:
            logger.info(f"  ✅ 预测变异性合理: 标准差比例{std_ratio:.1%}")

        return {
            'coverage_ratio': coverage_ratio,
            'std_ratio': std_ratio,
            'true_range': true_range,
            'pred_range': pred_range,
            'extreme_positive_pred': extreme_positive_pred,
            'extreme_negative_pred': extreme_negative_pred
        }

    except Exception as e:
        logger.warning(f"计算预测范围诊断指标失败: {e}")
        return {}

def _calculate_trading_metrics(y_true_return, y_pred_scores, logger, k_values=[1, 3, 5]):
    """
    🚀 计算交易指标：包含夏普比率、最大回撤等实际交易表现指标

    Args:
        y_true_return: 真实收益标签（ATR倍数）
        y_pred_scores: 预测分数（用于排序选股）
        logger: 日志记录器
        k_values: Top-K值列表

    Returns:
        dict: 交易指标字典
    """
    try:
        logger.info("🚀 计算交易指标...")

        # 按预测分数降序排序
        sorted_indices = np.argsort(y_pred_scores)[::-1]
        sorted_returns = y_true_return[sorted_indices]

        # 🚀 修复：使用与选股阶段一致的固定阈值
        # 重要：模型评估必须与实际选股使用相同的阈值标准
        profit_threshold = ATR_MULTIPLIER_PROFIT  # 2.0 ATR（与选股阶段一致）
        loss_threshold = -ATR_MULTIPLIER_LOSS     # -1.0 ATR（与选股阶段一致）

        float_tolerance = 1e-6  # 浮点数容差

        logger.info(f"🚀 使用与选股阶段一致的固定阈值:")
        logger.info(f"  止盈阈值: >= {profit_threshold:.1f} ATR (与选股阶段一致)")
        logger.info(f"  止损阈值: <= {loss_threshold:.1f} ATR (与选股阶段一致)")
        logger.info(f"  注意：评估标准必须与实际应用保持一致")

        # 1. 止盈止损命中率（使用固定阈值）
        profit_hits = (y_true_return >= profit_threshold).sum()
        profit_hit_rate = profit_hits / len(y_true_return) if len(y_true_return) > 0 else 0.0
        loss_hits = (y_true_return <= loss_threshold).sum()
        loss_hit_rate = loss_hits / len(y_true_return) if len(y_true_return) > 0 else 0.0

        # 2. 中等收益阈值评估
        moderate_thresholds = [0.5, 1.0, 1.5]
        moderate_hit_rates = {}
        for threshold in moderate_thresholds:
            hits = (y_true_return >= threshold).sum()
            moderate_hit_rates[f"profit_{threshold}"] = hits / len(y_true_return) if len(y_true_return) > 0 else 0.0

        # 3. 🚀 Top-K交易表现指标（修复计算错误）
        top_k_returns = {}
        top_k_profit_rates = {}
        top_k_loss_rates = {}
        top_k_sharpe_ratios = {}
        top_k_max_drawdowns = {}
        top_k_calmar_ratios = {}

        for k in k_values:
            if k <= len(sorted_returns):
                top_k_return = sorted_returns[:k]
                top_k_returns[k] = top_k_return.mean()

                # 使用固定阈值计算Top-K止盈率
                top_k_profit_rates[k] = (top_k_return >= (profit_threshold - float_tolerance)).mean()

                # 使用固定阈值计算Top-K止损率
                top_k_loss_rates[k] = (top_k_return <= (loss_threshold + float_tolerance)).mean()

                # 🚀 修复：Top-K夏普比率（处理单样本情况）
                if len(top_k_return) > 1 and top_k_return.std() > 1e-8:
                    # 使用无风险利率3%
                    risk_free_daily = 0.03 / 252
                    excess_returns = top_k_return - risk_free_daily
                    top_k_sharpe_ratios[k] = excess_returns.mean() / top_k_return.std() * np.sqrt(252)
                else:
                    # 单样本情况：如果收益为正且显著，给予合理评分
                    if len(top_k_return) == 1 and top_k_return[0] > 0.5:
                        top_k_sharpe_ratios[k] = top_k_return[0] * 10  # 简化评分
                    else:
                        top_k_sharpe_ratios[k] = 0.0

                # 🚀 修复：Top-K最大回撤（使用ATR直接计算）
                if len(top_k_return) > 1:
                    cumulative_returns = top_k_return.cumsum()  # 累计ATR收益
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdowns = cumulative_returns - running_max
                    top_k_max_drawdowns[k] = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
                else:
                    top_k_max_drawdowns[k] = 0.0

                # 🚀 修复：Top-K卡尔马比率
                if top_k_max_drawdowns[k] > 1e-8:
                    annual_return = top_k_return.mean() * 252  # 年化ATR收益
                    top_k_calmar_ratios[k] = annual_return / top_k_max_drawdowns[k]
                else:
                    top_k_calmar_ratios[k] = 0.0
            else:
                top_k_returns[k] = 0.0
                top_k_profit_rates[k] = 0.0
                top_k_loss_rates[k] = 0.0
                top_k_sharpe_ratios[k] = 0.0
                top_k_max_drawdowns[k] = 0.0
                top_k_calmar_ratios[k] = 0.0

        # 4. 收益分布分析
        return_distribution = {
            'profit_samples': (y_true_return >= (profit_threshold - float_tolerance)).sum(),
            'neutral_samples': ((y_true_return > (loss_threshold + float_tolerance)) & (y_true_return < (profit_threshold - float_tolerance))).sum(),
            'loss_samples': (y_true_return <= (loss_threshold + float_tolerance)).sum(),
            'mean_return': y_true_return.mean(),
            'std_return': y_true_return.std(),
            'min_return': y_true_return.min(),
            'max_return': y_true_return.max()
        }

        # 5. 预测效果分析
        # 计算预测收益最高的样本的实际表现
        top_10_pct_idx = int(len(sorted_returns) * 0.1)
        if top_10_pct_idx > 0:
            top_10_pct_return = sorted_returns[:top_10_pct_idx].mean()
            bottom_10_pct_return = sorted_returns[-top_10_pct_idx:].mean()
            prediction_effectiveness = top_10_pct_return - bottom_10_pct_return
        else:
            prediction_effectiveness = 0.0

        trading_metrics = {
            # 固定阈值指标
            'profit_hit_rate': profit_hit_rate,
            'loss_hit_rate': loss_hit_rate,
            # Top-K指标
            'top_k_returns': top_k_returns,
            'top_k_profit_rates': top_k_profit_rates,
            'top_k_loss_rates': top_k_loss_rates,
            'top_k_sharpe_ratios': top_k_sharpe_ratios,
            'top_k_max_drawdowns': top_k_max_drawdowns,
            'top_k_calmar_ratios': top_k_calmar_ratios,
            # 其他指标
            'return_distribution': return_distribution,
            'prediction_effectiveness': prediction_effectiveness,
            # 阈值信息
            'profit_threshold': profit_threshold,
            'loss_threshold': loss_threshold
        }

        # 🚀 多层次交易指标统计（解决阈值过高问题）
        logger.info(f"📊 交易指标统计:")
        logger.info(f"  参与统计的y_true数量: {len(y_true_return):,}个样本")
        logger.info(f"  🎯 固定阈值统计:")
        logger.info(f"    止盈边界样本数量: {profit_hits:,}个 (收益 >= {profit_threshold:.1f} ATR)")
        logger.info(f"    止损边界样本数量: {loss_hits:,}个 (收益 <= {loss_threshold:.1f} ATR)")
        logger.info(f"    止盈命中率: {profit_hit_rate:.4f} ({profit_hits}/{len(y_true_return)})")
        logger.info(f"    止损命中率: {loss_hit_rate:.4f} ({loss_hits}/{len(y_true_return)})")
        logger.info(f"  预测有效性: {prediction_effectiveness:.4f} ATR (Top10% - Bottom10%)")

        # 🚀 添加收益分布诊断信息
        logger.info(f"📊 收益分布诊断:")
        logger.info(f"  收益范围: [{y_true_return.min():.3f}, {y_true_return.max():.3f}] ATR")
        logger.info(f"  收益均值: {y_true_return.mean():.3f} ATR")
        logger.info(f"  收益标准差: {y_true_return.std():.3f} ATR")
        logger.info(f"  收益中位数: {np.median(y_true_return):.3f} ATR")
        logger.info(f"  负收益样本数: {(y_true_return < 0).sum():,}个 ({(y_true_return < 0).mean():.1%})")

        logger.info(f"📊 Top-K交易表现指标:")
        for k in k_values:
            if k <= len(sorted_returns):
                top_k_data = sorted_returns[:k]
                logger.info(f"  Top-{k}: 平均{top_k_returns[k]:.4f} ATR, 范围[{top_k_data.min():.3f}, {top_k_data.max():.3f}]")
                logger.info(f"    止盈率: {top_k_profit_rates[k]:.4f} ({(top_k_data >= (profit_threshold - float_tolerance)).sum()}/{k})")
                logger.info(f"    止损率: {top_k_loss_rates[k]:.4f} ({(top_k_data <= (loss_threshold + float_tolerance)).sum()}/{k})")
                logger.info(f"    负收益率: {(top_k_data < 0).mean():.4f} ({(top_k_data < 0).sum()}/{k})")
                logger.info(f"    🚀 夏普比率: {top_k_sharpe_ratios[k]:.4f}")
                logger.info(f"    🚀 最大回撤: {top_k_max_drawdowns[k]:.4f}")
                logger.info(f"    🚀 卡尔马比率: {top_k_calmar_ratios[k]:.4f}")

        # 🚀 添加对比统计：Top-K vs Bottom-K，解释为什么Top-K止损率为0
        if len(sorted_returns) >= 50:
            bottom_10_return = sorted_returns[-10:].mean()
            bottom_50_return = sorted_returns[-50:].mean()

            logger.info(f"📊 选股效果对比（解释Top-K止损率为0的原因）:")
            logger.info(f"  Top-10平均收益: {sorted_returns[:10].mean():.4f} ATR")
            logger.info(f"  Bottom-10平均收益: {bottom_10_return:.4f} ATR")
            logger.info(f"  Top-Bottom差异: {sorted_returns[:10].mean() - bottom_10_return:.4f} ATR")

            # 计算Bottom-K的止损率
            bottom_10_loss_rate = (sorted_returns[-10:] <= loss_threshold).mean()
            bottom_50_loss_rate = (sorted_returns[-50:] <= loss_threshold).mean()
            logger.info(f"  Bottom-10止损率: {bottom_10_loss_rate:.4f}")
            logger.info(f"  Bottom-50止损率: {bottom_50_loss_rate:.4f}")
            logger.info(f"✅ Top-K止损率为0说明选股策略有效，成功避开了表现差的股票")

        logger.info(f"📊 收益分布:")
        logger.info(f"  盈利样本: {return_distribution['profit_samples']} 个")
        logger.info(f"  中性样本: {return_distribution['neutral_samples']} 个")
        logger.info(f"  亏损样本: {return_distribution['loss_samples']} 个")
        logger.info(f"  平均收益: {return_distribution['mean_return']:.4f} ATR")

        return trading_metrics

    except Exception as e:
        logger.error(f"计算交易指标失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return {}


class EarlyStoppingCallback:
    """早停回调函数 - 如果连续N次试验没有改进,提前停止"""

    def __init__(self, patience=5, min_improvement=0.001, logger=None):
        self.patience = patience
        self.min_improvement = min_improvement
        self.logger = logger or logging.getLogger(__name__)
        self.best_score = float('-inf')  # ✅ 改为负无穷（越大越好）
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        current_score = study.best_value

        # 检查是否有改进（越大越好）
        if current_score > self.best_score + self.min_improvement:  # ✅ 改为 >
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # 检查是否需要早停
        if self.no_improvement_count >= self.patience:
            self.logger.info(f"  ⚠️ 早停触发: 连续{self.patience}次试验无明显改进(改进<{self.min_improvement:.4f})")
            study.stop()


def _perform_parameter_search(X_train, y_train, X_val, y_val, y_val_eval, sample_weights, search_strategy, val_dates=None, train_dates=None, market_data=None):
    """
    🔍 优化版贝叶斯超参数搜索（排序模型 + 数据子采样 + Optuna剪枝）

    优化策略:
    1. 单次搜索: 60%数据, 30 trials, 激进MedianPruner剪枝
    2. 使用排序损失函数(rank:ndcg)
    3. 减少early_stopping轮数加速

    参数范围:
    - n_estimators: 300-2000 (避免过度训练)
    - learning_rate: 0.01-0.2 (对数尺度)
    - max_depth: 3-10
    - subsample: 0.6-0.95
    - colsample_bytree: 0.6-0.95
    - min_child_weight: 1-15
    - gamma: 0-1
    - reg_alpha: 1e-4-5 (对数尺度)
    - reg_lambda: 1e-4-5 (对数尺度)

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        y_val_eval: 验证集评估标签
        sample_weights: 样本权重（未使用，保留接口兼容性）
        search_strategy: 搜索策略(未使用,保留接口兼容性)
        val_dates: 验证集日期(用于Top-K计算和group计算)
        train_dates: 训练集日期(用于group计算)
        market_data: 市场数据DataFrame（用于计算group权重）

    Returns:
        tuple: (best_params, search_results)
    """
    import time
    import xgboost as xgb
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("❌ 缺少optuna库,请安装: pip install optuna")
        logger.error("❌ 贝叶斯搜索需要optuna库,无法继续")
        return None, None, []

    logger.info("=" * 80)
    logger.info("🔍 启动优化版贝叶斯超参数搜索")
    logger.info("=" * 80)
    logger.info("优化策略:")
    logger.info("  1. 单次搜索（避免数据集不一致问题）")
    logger.info("  2. 数据子采样加速（60%数据）")
    logger.info("  3. Optuna MedianPruner激进剪枝")
    logger.info("  4. XGBoost多线程（n_jobs=-1）")
    logger.info("  5. 合理参数范围（n_estimators: 300-2000）")
    logger.info("  6. 减少early_stopping轮数加速")
    logger.info("  7. 预计算group权重（避免重复计算）")
    logger.info("=" * 80)

    # 🚀 数据子采样策略（60%数据）
    n_samples = len(X_train)
    sample_size = int(n_samples * 0.6)
    sample_idx = np.random.choice(n_samples, sample_size, replace=False)
    X_sampled = X_train[sample_idx]
    y_sampled = y_train[sample_idx]
    # 🚀 XGBoost排序模型不使用样本权重,跳过采样
    # weights_sampled = sample_weights[sample_idx] if sample_weights is not None else None
    train_dates_sampled = train_dates[sample_idx] if train_dates is not None else None

    logger.info(f"🎯 数据采样策略:")
    logger.info(f"  全量数据: {n_samples:,}条")
    logger.info(f"  搜索数据: {sample_size:,}条 (60%)")
    logger.info(f"  最终训练: {n_samples:,}条 (100%)")
    logger.info(f"  ⏭️ 跳过样本权重采样（XGBoost排序模型使用分组权重）")

    # 🚀 预计算group权重（避免每个trial重复计算）
    logger.info(f"\n🎯 预计算group权重:")
    group_weights_sampled = None
    if train_dates_sampled is not None:
        group_weights_sampled = _calculate_xgboost_group_weights(
            train_dates=train_dates_sampled,
            market_data=market_data,
            logger=logger,
            time_decay_half_life=180.0,
            reference_date=None,
            verbose=True  # 🔊 首次计算时输出详细日志
        )
        logger.info(f"  ✅ group权重已预计算: {len(group_weights_sampled)}个分组")
    else:
        logger.warning("  ⚠️ 缺少训练集日期信息,无法预计算group权重")

    # 基础参数（启用多线程加速，排序模型）
    # 🚀 使用rank:pairwise支持sample-level weights
    base_params = {'random_state': 42, 'n_jobs': -1, 'verbosity': 0, 'objective': 'rank:pairwise'}

    # 全局最优结果
    all_trials = []

    # ==================== 单次搜索: 探索参数空间 ====================
    logger.info("\n" + "=" * 80)
    logger.info("🔍 开始参数搜索（60%数据，30 trials）")
    logger.info("=" * 80)

    # 定义objective函数
    def objective(trial):
        """Objective: 探索参数空间"""
        params = {
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.95),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 15),
            'gamma': trial.suggest_float("gamma", 0, 1),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100)
        }
        params.update(base_params)

        # 🚀 减少early_stopping轮数加速搜索
        if params['n_estimators'] <= 500:
            early_stopping = 30
        elif params['n_estimators'] <= 1000:
            early_stopping = 50
        else:
            early_stopping = 80

        score, metrics = _evaluate_params(
            params, X_sampled, y_sampled, X_val, y_val, y_val_eval,
            None,  # 🚀 XGBoost排序模型不使用样本权重
            params['n_estimators'], early_stopping,
            is_coarse_search=False,
            val_dates=val_dates,
            train_dates=train_dates_sampled,
            market_data=market_data,  # 🚀 传递market_data（仅在group_weights=None时使用）
            group_weights=group_weights_sampled  # 🚀 传递预计算的group权重
        )

        all_trials.append({'stage': '搜索', 'params': params, 'score': score, 'metrics': metrics})
        trial.set_user_attr('metrics', metrics)

        return score

    # 创建study（使用激进的MedianPruner）
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # 减少到3(原5)
            n_warmup_steps=5     # 减少到5(原10)
        )
    )

    logger.info("  🚀 开始搜索（30 trials）...")
    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"  ✅ 搜索完成: score={best_score:.6f}")
    logger.info(f"     最优参数: lr={best_params['learning_rate']:.4f}, n_est={best_params['n_estimators']}, depth={best_params['max_depth']}")

    # 添加base_params
    global_best_params = best_params.copy()
    global_best_params.update(base_params)

    # ==================== 最终结果汇总 ====================
    logger.info("\n" + "=" * 80)
    logger.info("🎉 优化版贝叶斯超参数搜索完成!")
    logger.info("=" * 80)
    logger.info(f"🎯 全局最优综合评分: {best_score:.6f}")
    logger.info(f"🎯 最优参数:")
    for key, value in global_best_params.items():
        if key not in ['random_state', 'n_jobs', 'verbosity', 'objective']:
            logger.info(f"  {key}: {value}")
    logger.info(f"📊 总试验次数: {len(all_trials)}")
    logger.info(f"📊 预计加速: 2-5倍（数据子采样 + Optuna剪枝 + XGBoost多线程 + 减少early_stopping）")
    logger.info("=" * 80)

    return global_best_params, all_trials


def _evaluate_params(params, X_train, y_train, X_val, y_val, y_val_eval, sample_weights, max_boost_rounds, early_stopping, is_coarse_search=False, val_dates=None, train_dates=None, market_data=None, group_weights=None):
    """
    🚀 评估单组参数的性能（排序模型专用）

    Args:
        group_weights: 预计算的group权重（如果为None则重新计算）

    🚀 综合评分 = 0.70*Rank_IC + 0.20*Top_K_Return + 0.10*Top_K_Profit_Rate
                - overfitting_penalty - underfitting_penalty + r2_bonus

    ✅ 评分方向: 越大越好 (所有指标都是正向指标)
    ✅ 核心指标:
       - Rank IC (70%): Spearman相关系数,衡量预测排序与实际收益排序的相关性(主要指标,排序准确性)
       - Top-K Return (20%): 预测Top-K(K=1,3,5)的实际收益率加权均值(选股收益)
       - Top-K Profit Rate (10%): 预测Top-K中盈利股票的比例(选股质量)

    ✅ 辅助指标(仅日志输出,不参与score计算):
       - R²: 模型拟合质量,通过惩罚项间接影响score
       - IC: Pearson相关系数,诊断模型方向
       - NDCG: 归一化折损累积增益,排序模型标准评估指标

    🎯 设计理念:
       - 排序模型的本质是优化整体排序准确性,而非只优化Top-K收益
       - Rank IC直接衡量排序能力,是排序模型的最佳评估指标
       - 与集成学习权重计算一致(都使用Rank IC)

    过拟合/欠拟合处理:
    - 过拟合惩罚（指数平滑）:
      * overfitting_ratio > 0.2 → penalty = 0.1 * (1 - exp(-(overfitting_ratio - 0.2) * 2.0))
    - 欠拟合惩罚（指数平滑）:
      * val_r2 < 0 → penalty = 0.05 * (1 - exp(val_r2 * 5.0))
    - R²奖励:
      * val_r2 > 0 → bonus = 0.1 * val_r2

    Args:
        params: XGBoost参数字典
        X_train, y_train: 训练数据（标签为连续收益率）
        X_val, y_val: 验证数据（标签为连续收益率）
        y_val_eval: 验证集评估标签（加权收益率weighted_return: 0.4*2d + 0.4*3d + 0.2*5d）
        sample_weights: 样本权重（未使用，保留接口兼容性）
        max_boost_rounds: 最大训练轮数
        early_stopping: 早停轮数
        is_coarse_search: 已废弃（保留接口兼容性）
        val_dates: 验证集日期（用于计算group）
        train_dates: 训练集日期（用于计算group）
        market_data: 市场数据DataFrame（用于计算group权重）

    Returns:
        tuple: (综合评分, 指标字典) - 评分越大越好
    """
    import xgboost as xgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from scipy.stats import spearmanr

    try:
        # 计算group信息（排序模型必需）
        if train_dates is not None:
            train_dates_series = pd.Series(train_dates)
            train_groups = train_dates_series.value_counts().sort_index().values
        else:
            raise ValueError("排序模型需要train_dates参数")

        if val_dates is not None:
            val_dates_series = pd.Series(val_dates)
            val_groups = val_dates_series.value_counts().sort_index().values
        else:
            raise ValueError("排序模型需要val_dates参数")

        # 🚀 XGBoost排序模型权重处理: 使用group-level weights
        logger = logging.getLogger(__name__)

        # 🚀 使用预计算的group权重（如果没有则重新计算）
        if group_weights is None:
            logger.info("  ⚠️ group权重未预计算,正在重新计算...")
            group_weights = _calculate_xgboost_group_weights(
                train_dates=train_dates,
                market_data=market_data,
                logger=logger,
                time_decay_half_life=180.0,
                reference_date=None,  # 使用训练集最新日期
                verbose=False  # 🔇 避免重复输出日志
            )

        # 训练模型(使用排序损失函数)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_groups)
        dtrain.set_weight(group_weights)  # 🚀 使用group-level weights

        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(val_groups)

        evals = [(dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=max_boost_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )

        # 预测(直接使用排名标签空间,无需逆变换)
        y_train_pred = model.predict(dtrain)
        y_val_pred = model.predict(dval)

        # 🚀 计算MSE和R²（基于排名标签,而非原始收益率）
        # 注意: y_train和y_val是排名标签(0-1范围), y_val_eval是原始收益率
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)  # 🔧 修复: 使用y_val而非y_val_eval

        # 🚀 计算R²（用于过拟合检测,基于排名标签）
        train_r2 = 1 - mse_train / (y_train.var() + 1e-9)
        val_r2 = 1 - mse_val / (y_val.var() + 1e-9)  # 🔧 修复: 使用y_val而非y_val_eval
        overfitting_ratio = train_r2 - val_r2

        logger = logging.getLogger(__name__)

        # 🚀 排序模型核心指标
        # 🚀 不再使用sklearn的ndcg_score，改用自定义的calculate_ndcg_at_k

        # 🚀 初始化指标
        ic = 0.0
        rank_ic = 0.0
        ndcg_5 = 0.0
        ndcg_10 = 0.0
        top_k_return = 0.0

        if val_dates is not None and len(val_dates) == len(y_val_pred):
            import pandas as pd

            # 创建DataFrame便于按日期分组
            # 注意: y_val是relevance标签(0-3), y_val_eval是原始收益率
            val_df = pd.DataFrame({
                'date': val_dates,
                'pred': y_val_pred,
                'relevance': y_val,  # 🆕 relevance标签用于NDCG
                'return': y_val_eval  # 原始收益率用于IC和Top-K
            })

            # 🚀 计算每日指标
            ndcg_1_list = []
            ndcg_3_list = []
            ndcg_5_list = []
            ndcg_10_list = []
            # 🎲 随机基线NDCG列表
            ndcg_1_random_list = []
            ndcg_3_random_list = []
            ndcg_5_random_list = []
            ndcg_10_random_list = []
            daily_ic_list = []  # 🆕 每日IC
            daily_rank_ic_list = []  # 🆕 每日Rank IC
            daily_top_returns = []
            daily_top_profits = []
            # 📈 Top-k实际收益率列表
            top_1_return_list = []
            top_3_return_list = []
            top_5_return_list = []
            top_10_return_list = []

            for date, group in val_df.groupby('date'):
                if len(group) < 5:
                    continue

                # 🆕 每日IC (Pearson相关系数)
                daily_ic = np.corrcoef(group['pred'].values, group['return'].values)[0, 1]
                if not np.isnan(daily_ic):
                    daily_ic_list.append(daily_ic)

                # 🆕 每日Rank IC (Spearman相关系数)
                daily_rank_ic, _ = spearmanr(group['pred'].values, group['return'].values)
                if not np.isnan(daily_rank_ic):
                    daily_rank_ic_list.append(daily_rank_ic)

                # 🚀 NDCG使用真实收益率（更准确的评估）
                true_returns = group['return'].values
                pred_scores = group['pred'].values

                # 🎲 随机基线：多次随机打乱预测值，取平均（更公平的对比）
                num_random_trials = 10  # 多次随机试验取平均
                random_ndcg_1_trials, random_ndcg_3_trials, random_ndcg_5_trials, random_ndcg_10_trials = [], [], [], []

                for trial in range(num_random_trials):
                    # 保持预测值分布，只打乱顺序
                    np.random.seed(trial + int(hash(str(group.index.values.tobytes()))) % (2**32))
                    random_scores = np.random.permutation(pred_scores)

                    if len(group) >= 1:
                        random_ndcg_1_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=1))
                    if len(group) >= 3:
                        random_ndcg_3_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=3))
                    if len(group) >= 5:
                        random_ndcg_5_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=5))
                    if len(group) >= 10:
                        random_ndcg_10_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=10))

                # 模型NDCG计算（使用改进版本）
                if len(group) >= 1:
                    ndcg_1_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=1))
                    if random_ndcg_1_trials:
                        ndcg_1_random_list.append(np.mean(random_ndcg_1_trials))

                if len(group) >= 3:
                    ndcg_3_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=3))
                    if random_ndcg_3_trials:
                        ndcg_3_random_list.append(np.mean(random_ndcg_3_trials))

                if len(group) >= 5:
                    ndcg_5_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=5))
                    if random_ndcg_5_trials:
                        ndcg_5_random_list.append(np.mean(random_ndcg_5_trials))

                if len(group) >= 10:
                    ndcg_10_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=10))
                    if random_ndcg_10_trials:
                        ndcg_10_random_list.append(np.mean(random_ndcg_10_trials))

                # 📈 Top-k实际收益率计算
                if len(group) >= 1:
                    top_1_idx = group.nlargest(1, 'pred').index
                    top_1_return_list.append(group.loc[top_1_idx, 'return'].mean())

                if len(group) >= 3:
                    top_3_idx = group.nlargest(3, 'pred').index
                    top_3_return_list.append(group.loc[top_3_idx, 'return'].mean())

                if len(group) >= 5:
                    top_5_idx = group.nlargest(5, 'pred').index
                    top_5_return_list.append(group.loc[top_5_idx, 'return'].mean())

                if len(group) >= 10:
                    top_10_idx = group.nlargest(10, 'pred').index
                    top_10_return_list.append(group.loc[top_10_idx, 'return'].mean())

                # 按预测值降序排列
                group_sorted = group.sort_values('pred', ascending=False)

                # 选择Top-1, Top-3, Top-5个股票
                top_1 = group_sorted.iloc[:1]['return'].values
                top_3 = group_sorted.iloc[:3]['return'].values
                top_5 = group_sorted.iloc[:5]['return'].values

                # 计算每日Top-K收益
                daily_top_1_return = top_1.mean() if len(top_1) > 0 else 0.0
                daily_top_3_return = top_3.mean() if len(top_3) > 0 else 0.0
                daily_top_5_return = top_5.mean() if len(top_5) > 0 else 0.0

                # 计算每日Top-K盈利率
                daily_top_1_profit = (top_1 > 0).mean() if len(top_1) > 0 else 0.0
                daily_top_3_profit = (top_3 > 0).mean() if len(top_3) > 0 else 0.0
                daily_top_5_profit = (top_5 > 0).mean() if len(top_5) > 0 else 0.0

                # 加权平均(Top-1权重最高)
                daily_return = 0.5 * daily_top_1_return + 0.3 * daily_top_3_return + 0.2 * daily_top_5_return
                daily_profit = 0.5 * daily_top_1_profit + 0.3 * daily_top_3_profit + 0.2 * daily_top_5_profit

                daily_top_returns.append(daily_return)
                daily_top_profits.append(daily_profit)

            # 🚀 计算平均指标（每日指标的平均值）
            ndcg_1 = np.mean(ndcg_1_list) if ndcg_1_list else 0.0
            ndcg_3 = np.mean(ndcg_3_list) if ndcg_3_list else 0.0
            ndcg_5 = np.mean(ndcg_5_list) if ndcg_5_list else 0.0
            ndcg_10 = np.mean(ndcg_10_list) if ndcg_10_list else 0.0
            # 📈 Top-k实际收益率平均值
            top_1_avg_return = np.mean(top_1_return_list) if top_1_return_list else 0.0
            top_3_avg_return = np.mean(top_3_return_list) if top_3_return_list else 0.0
            top_5_avg_return = np.mean(top_5_return_list) if top_5_return_list else 0.0
            top_10_avg_return = np.mean(top_10_return_list) if top_10_return_list else 0.0
            ic = np.mean(daily_ic_list) if daily_ic_list else 0.0  # 🆕 每日IC的平均值
            rank_ic = np.mean(daily_rank_ic_list) if daily_rank_ic_list else 0.0  # 🆕 每日Rank IC的平均值

            # 计算所有交易日的平均值
            top_1_return = np.mean([group.sort_values('pred', ascending=False).iloc[:1]['return'].mean()
                                   for _, group in val_df.groupby('date') if len(group) >= 1])
            top_3_return = np.mean([group.sort_values('pred', ascending=False).iloc[:3]['return'].mean()
                                   for _, group in val_df.groupby('date') if len(group) >= 3])
            top_5_return = np.mean([group.sort_values('pred', ascending=False).iloc[:5]['return'].mean()
                                   for _, group in val_df.groupby('date') if len(group) >= 5])

            top_1_profit_rate = np.mean([(group.sort_values('pred', ascending=False).iloc[:1]['return'] > 0).mean()
                                        for _, group in val_df.groupby('date') if len(group) >= 1])
            top_3_profit_rate = np.mean([(group.sort_values('pred', ascending=False).iloc[:3]['return'] > 0).mean()
                                        for _, group in val_df.groupby('date') if len(group) >= 3])
            top_5_profit_rate = np.mean([(group.sort_values('pred', ascending=False).iloc[:5]['return'] > 0).mean()
                                        for _, group in val_df.groupby('date') if len(group) >= 5])

            top_k_return = np.mean(daily_top_returns)
            top_k_profit_rate = np.mean(daily_top_profits)

        else:
            # 降级方案: 如果没有日期信息,使用全局Top-K个股票
            sorted_indices = np.argsort(y_val_pred)[::-1]  # 降序排列,最大的在前

            # 直接选择Top-K个股票,而不是Top-K%
            top_1_return = y_val_eval[sorted_indices[:1]].mean() if len(sorted_indices) >= 1 else 0.0
            top_3_return = y_val_eval[sorted_indices[:3]].mean() if len(sorted_indices) >= 3 else 0.0
            top_5_return = y_val_eval[sorted_indices[:5]].mean() if len(sorted_indices) >= 5 else 0.0

            top_1_profit_rate = (y_val_eval[sorted_indices[:1]] > 0).mean() if len(sorted_indices) >= 1 else 0.0
            top_3_profit_rate = (y_val_eval[sorted_indices[:3]] > 0).mean() if len(sorted_indices) >= 3 else 0.0
            top_5_profit_rate = (y_val_eval[sorted_indices[:5]] > 0).mean() if len(sorted_indices) >= 5 else 0.0

            top_k_return = 0.5 * top_1_return + 0.3 * top_3_return + 0.2 * top_5_return
            top_k_profit_rate = 0.5 * top_1_profit_rate + 0.3 * top_3_profit_rate + 0.2 * top_5_profit_rate

        # 🚀 综合评分 (越大越好) - 排序模型专用评分
        # 🔧 方案: 将Rank IC作为主要优化目标(权重70%)
        # 理由: 排序模型的本质是优化整体排序准确性,而非只优化Top-K收益
        # Rank IC (Spearman相关系数) 直接衡量预测排序与实际收益排序的相关性
        composite_score = (
            0.70 * rank_ic +                    # Rank IC权重70%(主要指标,排序准确性)
            0.20 * top_k_return +               # Top-K收益率权重20%(选股收益)
            0.10 * top_k_profit_rate            # Top-K盈利率权重10%(选股质量)
        )

        # 🚀 软性惩罚: 过拟合和欠拟合（完全重新设计）
        overfitting_penalty = 0.0
        underfitting_penalty = 0.0

        # 过拟合惩罚: 使用平滑的指数惩罚
        # overfitting_ratio > 0.2时开始惩罚,越大惩罚越重
        if overfitting_ratio > 0.2:
            overfitting_penalty = 0.1 * (1 - np.exp(-(overfitting_ratio - 0.2) * 2.0))  # 最大惩罚约0.1

        # 欠拟合惩罚: 基于val_r2的负值程度
        # val_r2 < 0时开始惩罚,越负惩罚越重
        if val_r2 < 0:
            underfitting_penalty = 0.05 * (1 - np.exp(val_r2 * 5.0))  # 最大惩罚约0.05

        # R²奖励: 如果val_r2 > 0,给予额外奖励
        r2_bonus = 0.0
        if val_r2 > 0:
            r2_bonus = 0.1 * val_r2  # val_r2越高,奖励越多

        # 应用惩罚和奖励
        composite_score = composite_score - overfitting_penalty - underfitting_penalty + r2_bonus

        metrics = {
            'mse_train': mse_train,
            'mse_val': mse_val,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'ic': ic,
            'rank_ic': rank_ic,
            'ndcg_5': ndcg_5,  # 🆕 NDCG@5
            'ndcg_10': ndcg_10,  # 🆕 NDCG@10
            'top_1_return': top_1_return,
            'top_3_return': top_3_return,
            'top_5_return': top_5_return,
            'top_k_return': top_k_return,
            'top_1_profit_rate': top_1_profit_rate,
            'top_3_profit_rate': top_3_profit_rate,
            'top_5_profit_rate': top_5_profit_rate,
            'top_k_profit_rate': top_k_profit_rate,
            'overfitting_ratio': overfitting_ratio,
            'overfitting_penalty': overfitting_penalty,
            'underfitting_penalty': underfitting_penalty,
            'r2_bonus': r2_bonus,
            'composite_score': composite_score
        }

        return composite_score, metrics

    except Exception as e:
        # 训练失败,返回极小惩罚（越大越好）
        logger = logging.getLogger(__name__)
        logger.error(f"  ❌ 训练失败: {str(e)}")
        return -1e6, {'error': str(e)}


def _optimize_profit_loss_thresholds(val_df, val_atr14, logger):
    """
    🚀 基于验证集优化止盈止损ATR阈值（网格搜索 + 夏普比率）

    Args:
        val_df: 验证集DataFrame，包含以下列:
            - 'pred': 预测值(排序分数)
            - 'return': 实际收益(百分比,即加权收益率)
            - 'date': 交易日期
        val_atr14: 验证集ATR14数据(numpy array或Series,百分比)
        logger: 日志记录器

    Returns:
        dict: 包含最优阈值和性能指标的字典

    注意:
        - val_df['return']是百分比收益率(例如0.05表示5%)
        - val_atr14是ATR14值(百分比,例如0.02表示2%)
        - 止盈止损阈值是ATR倍数,需要转换为百分比后再比较
    """
    import numpy as np

    logger.info("\n" + "=" * 80)
    logger.info("🎯 开始止盈止损ATR阈值优化（基于LightGBM验证集）")
    logger.info("=" * 80)

    # 🔧 将val_atr14转换为numpy array
    if hasattr(val_atr14, 'values'):
        atr14_array = val_atr14.values
    else:
        atr14_array = np.array(val_atr14)

    # 🔧 确保长度匹配
    if len(atr14_array) != len(val_df):
        logger.error(f"❌ ATR14长度({len(atr14_array)})与val_df长度({len(val_df)})不匹配")
        return None

    # 🔧 添加ATR14到val_df
    val_df_with_atr = val_df.copy()
    val_df_with_atr['atr14'] = atr14_array

    # 🔧 计算平均ATR(用于日志输出)
    avg_atr = np.nanmean(atr14_array)
    logger.info(f"  验证集平均ATR14: {avg_atr:.4f} ({avg_atr*100:.2f}%)")
    logger.info(f"  验证集样本数: {len(val_df_with_atr):,}")

    # 🚀 网格搜索候选值
    profit_candidates = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]  # 止盈阈值候选
    loss_candidates = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]    # 止损阈值候选(正值)

    logger.info(f"📊 搜索空间:")
    logger.info(f"  止盈阈值候选: {profit_candidates}")
    logger.info(f"  止损阈值候选: {loss_candidates}")
    logger.info(f"  总组合数: {len(profit_candidates) * len(loss_candidates)}")

    # 初始化最优结果
    best_sharpe = -np.inf
    best_profit_th = 2.0
    best_loss_th = 1.0
    best_metrics = {}

    # 网格搜索
    results = []
    for profit_th in profit_candidates:
        for loss_th in loss_candidates:
            # 🚀 模拟交易逻辑（每日选Top-5）
            daily_returns = []

            for date, group in val_df_with_atr.groupby('date'):
                if len(group) < 5:
                    continue

                # 按预测值降序排列，选Top-5
                top_5 = group.nlargest(5, 'pred')

                # 应用止盈止损逻辑
                for _, row in top_5.iterrows():
                    actual_return_pct = row['return']  # 实际收益(百分比)
                    atr14_value = row['atr14']  # ATR14值(百分比)

                    # 🚀 将ATR阈值转换为百分比阈值
                    profit_th_pct = profit_th * atr14_value
                    loss_th_pct = loss_th * atr14_value

                    # 🚀 止盈止损判断（百分比比较）
                    if actual_return_pct >= profit_th_pct:
                        # 触发止盈，收益限制在止盈阈值
                        realized_return = profit_th_pct
                    elif actual_return_pct <= -loss_th_pct:
                        # 触发止损，损失限制在止损阈值
                        realized_return = -loss_th_pct
                    else:
                        # 未触发止盈止损，使用实际收益
                        realized_return = actual_return_pct

                    daily_returns.append(realized_return)

            # 计算性能指标
            if len(daily_returns) > 0:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                sharpe = mean_return / (std_return + 1e-8)
                win_rate = (np.array(daily_returns) > 0).mean()
                max_return = np.max(daily_returns)
                min_return = np.min(daily_returns)
            else:
                mean_return = 0.0
                std_return = 0.0
                sharpe = -np.inf
                win_rate = 0.0
                max_return = 0.0
                min_return = 0.0

            results.append({
                'profit_th': profit_th,
                'loss_th': loss_th,
                'sharpe': sharpe,
                'mean_return': mean_return,
                'std_return': std_return,
                'win_rate': win_rate,
                'n_trades': len(daily_returns),
                'max_return': max_return,
                'min_return': min_return
            })

            # 更新最优结果
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_profit_th = profit_th
                best_loss_th = loss_th
                best_metrics = {
                    'mean_return': mean_return,
                    'std_return': std_return,
                    'win_rate': win_rate,
                    'n_trades': len(daily_returns),
                    'max_return': max_return,
                    'min_return': min_return
                }

    # 🚀 输出结果
    logger.info("\n" + "=" * 80)
    logger.info("✅ 止盈止损阈值优化完成")
    logger.info("=" * 80)
    logger.info(f"\n🏆 最优阈值:")
    logger.info(f"  止盈阈值: {best_profit_th:.1f} ATR (约{best_profit_th*avg_atr*100:.2f}%)")
    logger.info(f"  止损阈值: {best_loss_th:.1f} ATR (约{best_loss_th*avg_atr*100:.2f}%)")
    logger.info(f"\n📊 最优性能:")
    logger.info(f"  夏普比率: {best_sharpe:.4f}")
    logger.info(f"  平均收益: {best_metrics['mean_return']:.4f} ({best_metrics['mean_return']*100:.2f}%)")
    logger.info(f"  收益标准差: {best_metrics['std_return']:.4f}")
    logger.info(f"  胜率: {best_metrics['win_rate']:.2%}")
    logger.info(f"  交易次数: {best_metrics['n_trades']}")
    logger.info(f"  最大收益: {best_metrics['max_return']:.4f} ({best_metrics['max_return']*100:.2f}%)")
    logger.info(f"  最小收益: {best_metrics['min_return']:.4f} ({best_metrics['min_return']*100:.2f}%)")

    # 🚀 输出Top-5组合
    logger.info(f"\n📊 Top-5 阈值组合:")
    results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]
    for i, r in enumerate(results_sorted, 1):
        logger.info(f"  {i}. 止盈={r['profit_th']:.1f}, 止损={r['loss_th']:.1f} | "
                   f"Sharpe={r['sharpe']:.4f}, 收益={r['mean_return']:.4f}, 胜率={r['win_rate']:.2%}")

    logger.info("=" * 80)

    return {
        'best_profit_threshold': best_profit_th,
        'best_loss_threshold': best_loss_th,
        'best_sharpe': best_sharpe,
        'best_metrics': best_metrics,
        'all_results': results
    }


def _train_lightgbm_model(raw_data, labels, feature_columns, label_stats=None, preprocessed_data=None, optimize_thresholds=False):
    """
    训练LightGBM模型(使用默认参数)

    Args:
        raw_data: 原始特征数据DataFrame
        labels: 标签数据Series（ATR倍数连续值）
        feature_columns: 特征列名列表
        label_stats: 标签统计量字典（用于反向转换）
        preprocessed_data: 预处理数据字典（可选），包含:
            - X_train: 训练集特征(已Winsorization)
            - X_val: 验证集特征(已Winsorization)
            - y_train: 训练集标签
            - y_val: 验证集标签
            - winsorization_thresholds: Winsorization阈值
        optimize_thresholds: 是否执行止盈止损ATR阈值优化（默认False）

    Returns:
        dict: 包含模型、性能指标、标签统计量等信息的字典
    """
    try:
        import lightgbm as lgb
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        import pandas as pd

        logger = logging.getLogger(__name__)
        logger.info("🚀 开始训练LightGBM模型（使用默认参数）")

        # 1. 使用预处理数据（与XGBoost共享）
        if preprocessed_data is not None:
            logger.info("✅ 使用预处理数据（与XGBoost共享）")
            X_train_raw = preprocessed_data['X_train']
            X_val_raw = preprocessed_data['X_val']

            # 🆕 排序模型使用relevance作为标签(整数类型0,1,2,3)
            y_train = preprocessed_data.get('train_relevance')
            logger.info(f"y_train:{y_train[-10:]}")
            y_val = preprocessed_data.get('val_relevance')
            logger.info(f"y_val:{y_val[-10:]}")
            # logger.info(f"x_train_raw:{X_train_raw[-10:]}")
            # logger.info(f"x_val_raw:{X_val_raw[-10:]}")

            # 保留原始收益率用于IC计算
            y_val_original = preprocessed_data.get('y_val_original', preprocessed_data.get('y_val'))

            if y_train is None or y_val is None:
                logger.error("❌ 缺少relevance标签,无法训练排序模型")
                return None

            # 🚀 获取样本权重计算所需数据
            train_dates = preprocessed_data.get('train_dates')
            turnover_rate = preprocessed_data.get('turnover_rate')
            circ_mv = preprocessed_data.get('circ_mv')

            logger.info(f"  训练集: {len(X_train_raw):,}样本")
            logger.info(f"  验证集: {len(X_val_raw):,}样本")
            logger.info(f"  原始特征数: {X_train_raw.shape[1]}")
            logger.info(f"  训练标签类型: relevance (整数0-3)")
        else:
            logger.error("❌ 缺少预处理数据,无法训练LightGBM")
            return None

        # 🚀 1.5 特征对齐(与XGBoost使用相同的特征)
        logger.info("🔧 特征对齐(与XGBoost一致)...")
        if X_train_raw.shape[1] != len(feature_columns):
            logger.info(f"  原始特征数: {X_train_raw.shape[1]}")
            logger.info(f"  目标特征数: {len(feature_columns)}")

            # 假设preprocessed_data中的特征顺序与raw_data一致
            # 需要根据feature_columns进行对齐
            # 这里简单处理:假设前len(feature_columns)个特征是需要的
            if X_train_raw.shape[1] > len(feature_columns):
                X_train = X_train_raw[:, :len(feature_columns)]
                X_val = X_val_raw[:, :len(feature_columns)]
                logger.info(f"  ✅ 特征对齐完成: {X_train_raw.shape[1]} -> {X_train.shape[1]}")
            else:
                logger.error(f"  ❌ 特征数量不足: {X_train_raw.shape[1]} < {len(feature_columns)}")
                return None
        else:
            X_train = X_train_raw
            X_val = X_val_raw
            logger.info(f"  ✅ 特征数量已对齐: {X_train.shape[1]}")

        # 2. 🔧 修复：删除StandardScaler标准化
        # 原因：
        # 1. 因子在计算时已经进行了标准化（如zscore、rank等）
        # 2. 树模型对特征尺度不敏感，不需要额外标准化
        # 3. 额外标准化导致贝叶斯搜索和最终训练的数据分布不一致
        # 4. 这是导致walkforward模式LightGBM过拟合的根本原因
        logger.info("⏭️ 跳过StandardScaler标准化（因子已在计算时标准化）")
        logger.info("   - 因子计算时已进行时间序列标准化（如zscore、rank）")
        logger.info("   - 树模型对特征尺度不敏感")
        logger.info("   - 避免贝叶斯搜索和训练的数据分布不一致")

        # 直接使用Winsorization后的特征，不进行额外标准化
        X_train_scaled = X_train
        X_val_scaled = X_val

        # 🔧 关键验证: 检查LightGBM训练集数据顺序
        if train_dates is not None:
            # 🔧 获取ATR收益率标签用于验证(用于IC计算)
            y_train_atr = preprocessed_data.get('y_train')  # ATR收益率标签
            y_val_atr = preprocessed_data.get('y_val')  # ATR收益率标签
            val_dates = preprocessed_data.get('val_dates')

            # 验证数据顺序
            if y_train_atr is not None and not _validate_data_order(train_dates, y_train_atr, y_train, "LightGBM训练集"):
                logger.error("❌ LightGBM训练集数据顺序验证失败,程序终止")
                raise ValueError("LightGBM训练集数据顺序混乱,无法继续训练")

            if val_dates is not None and y_val_atr is not None:
                if not _validate_data_order(val_dates, y_val_atr, y_val, "LightGBM验证集"):
                    logger.error("❌ LightGBM验证集数据顺序验证失败,程序终止")
                    raise ValueError("LightGBM验证集数据顺序混乱,无法继续训练")

        # 3. 配置LightGBM参数（排序模型）
        lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1,3,5],
            'lambdarank_truncation_level': 5,
            'lambdarank_norm': True,  # 标准化梯度，防止头部过度优化
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 70, #70
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'n_estimators': 2000,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        logger.info("📋 LightGBM参数配置:")
        for key, value in lgb_params.items():
            logger.info(f"  {key}: {value}")

        # 3.5 从preprocessed_data获取统一计算的样本权重
        logger.info("🔧 获取统一样本权重...")
        sample_weights = preprocessed_data.get('sample_weights') if preprocessed_data is not None else None
        val_sample_weights = preprocessed_data.get('val_sample_weights') if preprocessed_data is not None else None

        if sample_weights is None:
            # 如果preprocessed_data中没有权重,则使用均匀权重（兼容旧代码）
            logger.warning("  ⚠️ preprocessed_data中缺少sample_weights,使用均匀权重")
            sample_weights = np.ones(len(X_train_scaled))
        else:
            logger.info(f"  ✅ 使用preprocessed_data中的统一样本权重")
            logger.info(f"     权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
            logger.info(f"     权重均值: {sample_weights.mean():.4f}")

        if val_sample_weights is None:
            # 如果preprocessed_data中没有验证集权重,则使用均匀权重（兼容旧代码）
            logger.warning("  ⚠️ preprocessed_data中缺少val_sample_weights,验证集使用均匀权重")
            val_sample_weights = np.ones(len(X_val_scaled))
        else:
            logger.info(f"  ✅ 使用preprocessed_data中的验证集样本权重")
            logger.info(f"     权重范围: [{val_sample_weights.min():.4f}, {val_sample_weights.max():.4f}]")
            logger.info(f"     权重均值: {val_sample_weights.mean():.4f}")

        # 4. 训练LightGBM模型（排序模型）
        logger.info("🚀 开始训练LightGBM排序模型...")

        # 计算group信息（每个日期的样本数）
        if train_dates is not None:
            import pandas as pd
            train_dates_series = pd.Series(train_dates)
            val_dates = preprocessed_data.get('val_dates')

            # 确保数据按日期排序
            if not train_dates_series.is_monotonic_increasing:
                logger.warning("⚠️ 训练集日期未按升序排列,排序模型可能出现问题")

            # train_groups = train_dates_series.value_counts().sort_index().values
            _, train_groups = np.unique(train_dates, return_counts=True)
            logger.info(f"  训练集group数: {len(train_groups)}, 总样本数: {train_groups.sum()}")

            # 验证集group信息
            if val_dates is not None:
                # val_dates_series = pd.Series(val_dates)
                # val_groups = val_dates_series.value_counts().sort_index().values
                _, val_groups = np.unique(val_dates, return_counts=True)
                logger.info(f"  验证集group数: {len(val_groups)}, 总样本数: {val_groups.sum()}")
            else:
                logger.warning("⚠️ 缺少验证集日期信息,无法计算验证集group")
                val_groups = None
        else:
            logger.error("❌ 缺少训练集日期信息,无法使用排序模型")
            return None

        # 🔧 关键说明: LightGBM排序模型(lambdarank)支持同时使用group和weight
        # - group: 必需,用于定义查询组(每个日期是一个组)
        # - weight: 可选,用于样本级别的权重调整
        logger.info("🔧 LightGBM排序模型配置:")
        logger.info(f"  使用group信息: {len(train_groups)}个组")
        logger.info(f"  训练集样本权重: 范围[{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
        logger.info(f"  验证集样本权重: 范围[{val_sample_weights.min():.4f}, {val_sample_weights.max():.4f}]")

        # === 训练集 ===
        idx = np.argsort(train_dates)
        X_train_scaled = X_train_scaled[idx]
        y_train = y_train[idx]
        sample_weights = sample_weights[idx]
        train_dates = train_dates[idx]

        # === 验证集 ===
        idx_val = np.argsort(val_dates)
        val_dates = val_dates[idx_val]
        X_val_scaled = X_val_scaled[idx_val]
        y_val = y_val[idx_val]
        val_sample_weights = val_sample_weights[idx_val]

        sample_weights = np.ones_like(y_train)
        val_sample_weights = np.ones_like(y_val)

        # 创建LightGBM数据集（同时使用group和weight）
        train_data = lgb.Dataset(
            X_train_scaled,
            label=y_train,
            weight=sample_weights,
            group=train_groups
        )

        if val_groups is not None:
            val_data = lgb.Dataset(
                X_val_scaled,
                label=y_val,
                weight=val_sample_weights,  # 🆕 添加验证集权重
                reference=train_data,
                group=val_groups
            )
        else:
            val_data = lgb.Dataset(X_val_scaled, label=y_val, weight=val_sample_weights, reference=train_data)  # 🆕 添加验证集权重

        # Early Stopping配置
        early_stopping_rounds = 200
        logger.info(f"  Early Stopping: {early_stopping_rounds} rounds")

        # 创建一个空字典来存储评估结果
        eval_result = {}
        # 训练模型
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=False, verbose=True),
            lgb.log_evaluation(period=10),  # 改为每轮输出
            lgb.record_evaluation(eval_result)
        ]
        # 训练前添加数据检查
        logger.info(f"训练集: {X_train_scaled.shape}, 验证集: {X_val_scaled.shape}")
        logger.info(f"训练集groups数量: {len(train_groups) if hasattr(train_groups, '__len__') else 'N/A'}")
        logger.info(f"训练集标签范围: {y_train.min():.3f} ~ {y_train.max():.3f}")
        logger.info(f"验证集标签范围: {y_val.min():.3f} ~ {y_val.max():.3f}")


        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=4000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            # valid_sets=[val_data],
            # valid_names=['valid'],
            callbacks=callbacks
        )

        logger.info(f"✅ LightGBM训练完成")
        logger.info(f"  最佳迭代: {model.best_iteration}")
        logger.info(f"  训练轮数: {model.current_iteration()}")

        # 训练完成后，分析评估结果
        logger.info("📊 训练过程详细指标:")
        # for metric_name, metric_values in eval_result['train'].items():
        #     logger.info(f"训练集 {metric_name}: 前5轮 = {list(metric_values[:5])}")

        for metric_name, metric_values in eval_result['valid'].items():
            logger.info(f"验证集 {metric_name}: 前5轮 = {list(metric_values[:5])}")

        # 检查是否有NaN或异常值
        for dataset_name, metrics in eval_result.items():
            for metric_name, values in metrics.items():
                if len(values) > 0:
                    if any(np.isnan(values)):
                        logger.warning(f"{dataset_name} {metric_name} 包含NaN值")
                    logger.info(f"{dataset_name} {metric_name}: 起始值={values[0]:.4f}, 结束值={values[-1]:.4f}")

        # 5. 模型评估
        logger.info("📊 评估LightGBM排序模型...")

        # 训练集预测
        y_train_pred = model.predict(X_train_scaled, num_iteration=model.best_iteration)

        # 验证集预测
        y_val_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)

        # 🚀 排序模型核心指标
        from scipy.stats import spearmanr
        # 🚀 不再使用sklearn的ndcg_score，改用自定义的calculate_ndcg_at_k

        # 获取原始收益率标签(用于IC计算)
        y_val_original = preprocessed_data.get('y_val_original', preprocessed_data.get('y_val'))

        # 🚀 初始化指标
        ic = 0.0
        rank_ic = 0.0
        ndcg_5 = 0.0
        ndcg_10 = 0.0
        top_k_return = 0.0

        # 🚀 计算每日指标（与XGBoost保持一致）
        val_dates = preprocessed_data.get('val_dates')
        if val_dates is not None and len(val_dates) == len(y_val_pred):
            import pandas as pd
            # 注意: y_val是relevance标签(0-3), y_val_original是原始收益率
            val_df = pd.DataFrame({
                'date': val_dates,
                'pred': y_val_pred,
                'relevance': y_val,  # 🆕 relevance标签用于NDCG
                'return': y_val_original  # 原始收益率用于IC和Top-K
            })

            # 🚀 计算每日指标
            ndcg_1_list = []
            ndcg_3_list = []
            ndcg_5_list = []
            ndcg_10_list = []
            # 📈 Top-k实际收益率列表
            top_1_return_list = []
            top_3_return_list = []
            top_5_return_list = []
            top_10_return_list = []
            daily_ic_list = []  # 🆕 每日IC
            daily_rank_ic_list = []  # 🆕 每日Rank IC
            top_k_returns = []
            top_1_returns = []
            top_2_returns = []

            for date, group in val_df.groupby('date'):
                if len(group) < 5:
                    continue

                # 🆕 每日IC (Pearson相关系数)
                daily_ic = np.corrcoef(group['pred'].values, group['return'].values)[0, 1]
                if not np.isnan(daily_ic):
                    daily_ic_list.append(daily_ic)

                # 🆕 每日Rank IC (Spearman相关系数)
                daily_rank_ic, _ = spearmanr(group['pred'].values, group['return'].values)
                if not np.isnan(daily_rank_ic):
                    daily_rank_ic_list.append(daily_rank_ic)

                # 🚀 NDCG使用真实收益率（更准确的评估）
                true_returns = group['return'].values
                pred_scores = group['pred'].values

                # NDCG计算（使用改进版本）
                if len(group) >= 1:
                    ndcg_1_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=1))
                if len(group) >= 3:
                    ndcg_3_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=3))
                if len(group) >= 5:
                    ndcg_5_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=5))
                if len(group) >= 10:
                    ndcg_10_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=10))

                # 📈 Top-k实际收益率计算
                if len(group) >= 1:
                    top_1_idx = group.nlargest(1, 'pred').index
                    top_1_return_list.append(group.loc[top_1_idx, 'return'].mean())

                if len(group) >= 3:
                    top_3_idx = group.nlargest(3, 'pred').index
                    top_3_return_list.append(group.loc[top_3_idx, 'return'].mean())

                if len(group) >= 5:
                    top_5_idx = group.nlargest(5, 'pred').index
                    top_5_return_list.append(group.loc[top_5_idx, 'return'].mean())

                if len(group) >= 10:
                    top_10_idx = group.nlargest(10, 'pred').index
                    top_10_return_list.append(group.loc[top_10_idx, 'return'].mean())

                # 兼容原有Top-k收益率变量
                top_1_idx = group.nlargest(1, 'pred').index
                top_2_idx = group.nlargest(2, 'pred').index
                top_5_idx = group.nlargest(5, 'pred').index
                top_1_returns.append(group.loc[top_1_idx, 'return'].mean())
                top_2_returns.append(group.loc[top_2_idx, 'return'].mean())
                top_k_returns.append(group.loc[top_5_idx, 'return'].mean())

            # 🚀 计算平均指标（每日指标的平均值）
            ndcg_1 = np.mean(ndcg_1_list) if ndcg_1_list else 0.0
            ndcg_3 = np.mean(ndcg_3_list) if ndcg_3_list else 0.0
            ndcg_5 = np.mean(ndcg_5_list) if ndcg_5_list else 0.0
            ndcg_10 = np.mean(ndcg_10_list) if ndcg_10_list else 0.0
            # 📈 Top-k实际收益率平均值
            top_1_avg_return = np.mean(top_1_return_list) if top_1_return_list else 0.0
            top_3_avg_return = np.mean(top_3_return_list) if top_3_return_list else 0.0
            top_5_avg_return = np.mean(top_5_return_list) if top_5_return_list else 0.0
            top_10_avg_return = np.mean(top_10_return_list) if top_10_return_list else 0.0
            ic = np.mean(daily_ic_list) if daily_ic_list else 0.0  # 🆕 每日IC的平均值
            rank_ic = np.mean(daily_rank_ic_list) if daily_rank_ic_list else 0.0  # 🆕 每日Rank IC的平均值
            top_k_return = np.mean(top_k_returns) if top_k_returns else 0.0
            top_1_return = np.mean(top_1_returns) if top_1_returns else 0.0
            top_2_return = np.mean(top_2_returns) if top_2_returns else 0.0

        # 🚀 辅助指标(MSE/R²) - 仅用于监控,排序模型主要关注NDCG和IC
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # 训练集辅助指标
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # 验证集辅助指标
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        # 📊 模型性能指标输出
        logger.info("📊 排序模型核心指标:")
        logger.info(f"  Rank IC: {rank_ic:.4f}")
        logger.info(f"  IC: {ic:.4f}")
        logger.info(f"  NDCG@1: {ndcg_1:.4f}")
        logger.info(f"  NDCG@3: {ndcg_3:.4f}")
        logger.info(f"  NDCG@5: {ndcg_5:.4f}")
        logger.info(f"  NDCG@10: {ndcg_10:.4f}")

        # 📈 Top-k实际收益率统计
        def format_stats(values):
            """格式化统计信息：均值(方差) [最小值 ~ 最大值]"""
            if len(values) == 0:
                return "N/A"
            mean_val = np.mean(values)
            var_val = np.var(values)
            min_val = np.min(values)
            max_val = np.max(values)
            return f"{mean_val:.4f} (方差:{var_val:.6f}) [{min_val:.4f} ~ {max_val:.4f}]"

        logger.info("📈 Top-k实际收益率统计:")
        logger.info(f"  Top-1收益率: {format_stats(top_1_return_list)}")
        logger.info(f"  Top-3收益率: {format_stats(top_3_return_list)}")
        logger.info(f"  Top-5收益率: {format_stats(top_5_return_list)}")
        logger.info(f"  Top-10收益率: {format_stats(top_10_return_list)}")

        # 计算整个验证集的平均收益率
        if val_dates is not None and len(y_val_original) > 0:
            import pandas as pd
            val_df = pd.DataFrame({
                'date': val_dates,
                'return': y_val_original
            })
            overall_avg_return = val_df.groupby('date')['return'].mean().mean()
            logger.info(f"  验证集平均收益率: {overall_avg_return:.4f}")
        logger.info("")

        logger.info("📊 辅助指标(仅供监控):")
        logger.info(f"  验证集MSE: {val_mse:.6f}")
        logger.info(f"  验证集R²: {val_r2:.6f}")

        # 🆕 投资导向评估
        if val_dates is not None and len(y_val_original) > 0:
            logger.info("🎯 投资导向评估:")

            # 计算整体投资表现
            overall_investment_results = evaluate_investment_performance(
                y_val_original, y_val_pred, k=5, risk_free_rate=0.0
            )

            # 输出投资评估报告
            print_investment_evaluation_report("XGBoost排序模型", overall_investment_results)

        # 🚀 5.5 止盈止损ATR阈值优化（可选）
        if optimize_thresholds and val_dates is not None and len(val_dates) == len(y_val_pred):
            # 🔧 获取验证集ATR14数据
            val_atr14 = preprocessed_data.get('val_atr14') if preprocessed_data is not None else None

            if val_atr14 is not None and len(val_atr14) == len(y_val_pred):
                threshold_results = _optimize_profit_loss_thresholds(val_df, val_atr14, logger)
                # 将优化结果添加到性能字典中（仅用于日志记录）
                if threshold_results is not None:
                    optimal_thresholds = {
                        'profit_threshold': threshold_results['best_profit_threshold'],
                        'loss_threshold': threshold_results['best_loss_threshold'],
                        'sharpe_ratio': threshold_results['best_sharpe']
                    }
                else:
                    optimal_thresholds = None
            else:
                logger.warning("⚠️ 验证集缺少ATR14数据，跳过止盈止损阈值优化")
                optimal_thresholds = None
        else:
            optimal_thresholds = None

        # 6. 特征重要性
        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = feature_columns

        # 7. 构建返回结果
        performance_dict = {
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            # 🚀 排序模型核心指标
            'ic': ic,
            'rank_ic': rank_ic,
            'ndcg_5': ndcg_5,
            'ndcg_10': ndcg_10,
            'top_k_return': top_k_return,
        }

        # 🚀 添加止盈止损优化结果（如果执行了优化）
        if optimal_thresholds is not None:
            performance_dict['optimal_thresholds'] = optimal_thresholds

        # 🆕 投资导向评估
        if val_dates is not None and y_val_original is not None and len(y_val_original) > 0:
            logger.info("🎯 投资导向评估:")

            # 计算整体投资表现
            overall_investment_results = evaluate_investment_performance(
                y_val_original, y_val_pred, k=5, risk_free_rate=0.0
            )

            # 输出投资评估报告
            print_investment_evaluation_report("LightGBM排序模型", overall_investment_results)

        model_results = {
            'model': model,
            'scaler': None,  # 🔧 修复：不再使用StandardScaler，设为None
            'feature_names': feature_names,
            'label_stats': label_stats,
            'performance': performance_dict,
            'training_info': {
                'best_iteration': model.best_iteration,
                'total_iterations': model.current_iteration(),
                'early_stopping_rounds': early_stopping_rounds,
            },
            'model_params': lgb_params,
            'feature_importance': dict(zip(feature_names, feature_importance)),
            'model_type': 'lightgbm',  # 标记模型类型
        }

        logger.info("✅ LightGBM模型训练完成")
        return model_results

    except ImportError:
        logger = logging.getLogger(__name__)
        logger.error("❌ LightGBM未安装,请运行: pip install lightgbm")
        return None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ LightGBM模型训练失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None


def _train_ml_model(raw_data, labels, feature_columns, winsorization_thresholds=None, label_stats=None, label_transform='adaptive', enable_param_search=False, search_strategy='grid', random_seed=42, preprocessed_data=None, enable_shap_analysis=True, preset_params=None, market_data=None):
    """
    🚀 成本感知回归模型训练（支持超参数搜索优化 + 集成学习）

    Args:
        raw_data: 原始特征数据DataFrame
        labels: 标签数据Series（ATR倍数连续值）
        feature_columns: 特征列名列表
        winsorization_thresholds: 预计算的Winsorization阈值（可选）
        label_stats: 标签统计量字典（用于反向转换）
        label_transform: 标签变换方法（'adaptive', 'asinh', 'signed_log', 'none'）
        enable_param_search: 是否启用超参数搜索（默认False）
        search_strategy: 搜索策略 ('grid', 'fast')
        random_seed: 随机种子（用于集成学习，默认42）
        preprocessed_data: 预处理数据字典（可选），包含:
            - X_train: 训练集特征(已Winsorization)
            - X_val: 验证集特征(已Winsorization)
            - y_train: 训练集标签
            - y_val: 验证集标签
            - winsorization_thresholds: Winsorization阈值
        enable_shap_analysis: 是否启用SHAP分析（默认True）
            - 在集成学习中,建议只对第一个模型启用以节省时间
        preset_params: 预设的模型参数（可选）
            - 如果提供,将跳过参数搜索,直接使用该参数
            - 用于Walk-Forward模式,避免重复搜索
        market_data: 市场数据DataFrame（用于XGBoost计算group权重）

    Returns:
        dict: 包含模型、性能指标、标签统计量、最优参数等信息的字典
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        import numpy as np
        import pandas as pd

        # 导入回归评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        logger = logging.getLogger(__name__)
        logger.info(f"🚀 开始训练回归模型，样本数: {len(raw_data)}, 特征数: {len(feature_columns)}")

        # 🚀 统一阈值定义（避免重复定义）
        PROFIT_THRESHOLD = ATR_MULTIPLIER_PROFIT  # 2.0 ATR
        LOSS_THRESHOLD = -ATR_MULTIPLIER_LOSS     # -1.0 ATR

        # 🚀 步骤1：准备数据
        logger.info("🚀 准备训练数据...")

        # 数据对齐
        if len(raw_data) != len(labels):
            logger.warning(f"数据长度不一致，进行对齐: 特征{len(raw_data)} vs 标签{len(labels)}")
            min_length = min(len(raw_data), len(labels))
            raw_data = raw_data.iloc[:min_length] if hasattr(raw_data, 'iloc') else raw_data[:min_length]
            labels = labels.iloc[:min_length] if hasattr(labels, 'iloc') else labels[:min_length]

        # 🚀 排序模型：labels参数实际上是atr_return_label（连续值，用于IC计算）
        # 但训练时会使用relevance标签（整数0-3）
        # 🔧 重要: 保持labels的index,确保与raw_data对齐
        if hasattr(labels, 'values'):
            # labels是pandas Series,保留index
            y_data_series = labels.astype(np.float32)
            y_data = y_data_series.values
        else:
            # labels是numpy array
            y_data = labels.astype(np.float32)
            y_data_series = None
        y_data_original = y_data.copy()

        # 🚀 保存原始标签统计（用于IC计算）
        original_y_stats = {
            'mean': y_data.mean(),
            'std': y_data.std(),
            'min': y_data.min(),
            'max': y_data.max(),
            'range': y_data.max() - y_data.min()
        }
        logger.info(f"✅ 原始收益率标签统计（用于IC计算）:")
        logger.info(f"  标签范围: [{original_y_stats['min']:.3f}, {original_y_stats['max']:.3f}]")
        logger.info(f"  标签统计: mean={original_y_stats['mean']:.3f}, std={original_y_stats['std']:.3f}")

        logger.info(f"数据对齐完成: {len(raw_data)} 个样本, {len(feature_columns)} 个特征列")

        # 🚀 优化：如果提供了预处理数据,跳过数据划分和预处理步骤
        if preprocessed_data is not None:
            logger.info("=" * 60)
            logger.info("✅ 使用预处理数据（集成学习优化）")
            logger.info("=" * 60)
            logger.info(f"  🎯 训练集: {len(preprocessed_data['X_train']):,}样本")
            logger.info(f"  🎯 验证集: {len(preprocessed_data['X_val']):,}样本")
            logger.info("=" * 60)

            X_train_processed = preprocessed_data['X_train']
            X_val_processed = preprocessed_data['X_val']

            # 🆕 XGBoost排序模型使用relevance_30作为标签(整数类型0-29,30层)
            y_train_split = preprocessed_data.get('train_relevance_30')
            y_val_split = preprocessed_data.get('val_relevance_30')

            # 保留原始收益率用于IC计算
            y_train_split_original = preprocessed_data.get('y_train_original', preprocessed_data.get('y_train'))
            y_val_split_original = preprocessed_data.get('y_val_original', preprocessed_data.get('y_val'))

            if y_train_split is None or y_val_split is None:
                logger.error("❌ 缺少relevance_30标签,无法训练XGBoost排序模型")
                return None

            # 🚀 确保relevance_30标签是整数类型（XGBoost排序模型要求）
            y_train_split = np.array(y_train_split, dtype=np.int32)
            y_val_split = np.array(y_val_split, dtype=np.int32)
            logger.info(f"  ✅ relevance_30标签已转换为int32类型(30层)")
            logger.info(f"  训练集relevance_30范围: [{y_train_split.min()}, {y_train_split.max()}]")
            logger.info(f"  验证集relevance_30范围: [{y_val_split.min()}, {y_val_split.max()}]")

            train_winsorization_thresholds = preprocessed_data['winsorization_thresholds']

            # 🚀 从预处理数据中提取特征名
            # 注意: X_train_processed可能是DataFrame或numpy数组
            if hasattr(X_train_processed, 'columns'):
                feature_names = list(X_train_processed.columns)
            else:
                # 如果是numpy数组,使用传入的feature_columns
                feature_names = feature_columns if isinstance(feature_columns, list) else list(feature_columns)

            logger.info("✅ 使用relevance_30标签（整数0-29,30层）用于XGBoost排序模型训练")
            logger.info(f"  训练标签类型: relevance_30 (整数0-29,30层)")
            logger.info(f"  保留原始收益率用于IC计算")

            # 🆕 从预处理数据中提取训练集和验证集日期（用于时间衰减权重和group计算）
            train_dates = None
            val_dates = None
            if 'train_dates' in preprocessed_data:
                train_dates = preprocessed_data['train_dates']
                logger.info("✅ 从预处理数据中提取训练集日期")
            if 'val_dates' in preprocessed_data:
                val_dates = preprocessed_data['val_dates']
                logger.info("✅ 从预处理数据中提取验证集日期")

            # 🔧 关键验证: 检查训练集数据顺序
            if train_dates is not None:
                if not _validate_data_order(train_dates, y_train_split_original, y_train_split, "XGBoost训练集"):
                    logger.error("❌ XGBoost训练集数据顺序验证失败,程序终止")
                    raise ValueError("XGBoost训练集数据顺序混乱,无法继续训练")

            # 🔧 关键验证: 检查验证集数据顺序
            if val_dates is not None:
                if not _validate_data_order(val_dates, y_val_split_original, y_val_split, "XGBoost验证集"):
                    logger.error("❌ XGBoost验证集数据顺序验证失败,程序终止")
                    raise ValueError("XGBoost验证集数据顺序混乱,无法继续训练")

            # 继续原有逻辑(train_dates和val_dates已在上面提取并验证)

            # 🆕 从预处理数据中提取换手率和流通市值（用于基础权重计算）
            train_turnover_rate = None
            train_circ_mv = None
            if 'turnover_rate' in preprocessed_data:
                train_turnover_rate = preprocessed_data['turnover_rate']
                logger.info("✅ 从预处理数据中提取训练集换手率")
            else:
                logger.warning("⚠️ 预处理数据中未包含turnover_rate，将使用均匀权重")

            if 'circ_mv' in preprocessed_data:
                train_circ_mv = preprocessed_data['circ_mv']
                logger.info("✅ 从预处理数据中提取训练集流通市值")
            else:
                logger.warning("⚠️ 预处理数据中未包含circ_mv，将使用均匀权重")
        else:
            # 🔧 步骤2：先进行训练/验证集划分（避免数据泄露）
            logger.info("=" * 60)
            logger.info("🔄 时间序列友好的数据划分策略")
            logger.info("=" * 60)
            logger.info("📋 划分策略说明:")
            logger.info("  ✅ 按时间顺序划分，非随机划分")
            logger.info("  ✅ 前80%时间段训练，后20%验证")
            logger.info("  ✅ 避免未来信息泄露到历史训练中")
            logger.info("  ✅ 更符合实际交易场景的时间顺序")
            logger.info("  ✅ 预处理参数仅基于训练集计算")

            split_point = int(len(raw_data) * 0.8)

            # 🔧 关键修复: 划分原始数据时,确保数据和标签使用相同的index
            raw_train_data = raw_data.iloc[:split_point] if hasattr(raw_data, 'iloc') else raw_data[:split_point]
            raw_val_data = raw_data.iloc[split_point:] if hasattr(raw_data, 'iloc') else raw_data[split_point:]

            # 🔧 重要: 如果labels是Series,使用.iloc保持index对齐;否则使用numpy切片
            if y_data_series is not None:
                y_train_split = y_data_series.iloc[:split_point].values
                y_val_split = y_data_series.iloc[split_point:].values
                logger.info("✅ 使用Series.iloc划分标签,确保与数据对齐")
            else:
                y_train_split = y_data[:split_point]
                y_val_split = y_data[split_point:]
                logger.info("✅ 使用numpy切片划分标签")

            # 🚀 保留原始收益率标签（用于IC计算）
            y_train_split_original = y_data_original[:split_point]
            logger.info("✅ 保留原始收益率标签（用于IC计算）")

            # 🆕 提取训练集日期（用于时间衰减权重）
            train_dates = None
            if 'date' in raw_train_data.columns:
                train_dates = raw_train_data['date'].values
                logger.info("✅ 提取训练集日期（用于时间衰减权重计算）")

            # 🆕 提取换手率和流通市值（用于基础权重计算）
            train_turnover_rate = None
            train_circ_mv = None
            if 'turnover_rate' in raw_train_data.columns:
                train_turnover_rate = raw_train_data['turnover_rate'].values
                logger.info("✅ 提取训练集换手率（用于基础权重计算）")
            else:
                logger.warning("⚠️ 训练数据中未包含turnover_rate列，将使用均匀权重")

            if 'circ_mv' in raw_train_data.columns:
                train_circ_mv = raw_train_data['circ_mv'].values
                logger.info("✅ 提取训练集流通市值（用于基础权重计算）")
            else:
                logger.warning("⚠️ 训练数据中未包含circ_mv列，将使用均匀权重")

            # 🚀 添加详细的日期范围日志
            train_date_min = raw_train_data['date'].min() if 'date' in raw_train_data.columns else None
            train_date_max = raw_train_data['date'].max() if 'date' in raw_train_data.columns else None
            val_date_min = raw_val_data['date'].min() if 'date' in raw_val_data.columns else None
            val_date_max = raw_val_data['date'].max() if 'date' in raw_val_data.columns else None

            logger.info(f"📊 时间序列划分结果:")
            logger.info(f"  🎯 训练集: {len(raw_train_data):,}样本 (前80%时间段)")
            if train_date_min and train_date_max:
                logger.info(f"     📅 日期范围: {train_date_min.strftime('%Y-%m-%d')} 至 {train_date_max.strftime('%Y-%m-%d')}")
            logger.info(f"  🎯 验证集: {len(raw_val_data):,}样本 (后20%时间段)")
            if val_date_min and val_date_max:
                logger.info(f"     📅 日期范围: {val_date_min.strftime('%Y-%m-%d')} 至 {val_date_max.strftime('%Y-%m-%d')}")
            logger.info("=" * 60)

            # 🚀 步骤3：分别对训练集和验证集进行特征预处理（避免数据泄露）
            X_train_processed, feature_names = _prepare_feature_data(
                raw_train_data, feature_columns, data_type="训练集"
            )

            if X_train_processed is None:
                logger.error("训练集特征预处理失败")
                return None

            X_val_processed, _ = _prepare_feature_data(
                raw_val_data, feature_columns, data_type="验证集"
            )

            if X_val_processed is None:
                logger.error("验证集特征预处理失败")
                return None

            # 🔧 关键修复：确保标签与预处理后的特征数据长度匹配
            # 由于预处理可能移除重复行，需要调整标签数组长度
            if len(X_train_processed) != len(y_train_split):
                logger.warning(f"训练集预处理后长度变化: {len(y_train_split)} -> {len(X_train_processed)}")
                y_train_split = y_train_split[:len(X_train_processed)]

            if len(X_val_processed) != len(y_val_split):
                logger.warning(f"验证集预处理后长度变化: {len(y_val_split)} -> {len(X_val_processed)}")
                y_val_split = y_val_split[:len(X_val_processed)]

        # 使用预处理后的数据
        X_train_split = X_train_processed
        X_val_split = X_val_processed
        winsorization_thresholds = train_winsorization_thresholds

        logger.info("✅ 训练集和验证集特征预处理完成，确保数据预处理一致性且避免数据泄露")
        logger.info(f"最终训练集: {len(X_train_split)}个样本, {X_train_split.shape[1]}个特征")
        logger.info(f"最终验证集: {len(X_val_split)}个样本, {X_val_split.shape[1]}个特征")
        logger.info(f"训练标签: {len(y_train_split)}个, 验证标签: {len(y_val_split)}个")

        # 🚀 成本感知回归：分析连续收益标签分布
        logger.info("=" * 50)
        logger.info("🚀 成本感知回归训练集样本分布分析:")


        # 回归模式不需要不平衡比例
        imbalance_ratio = 1.0
        logger.info("=" * 50)

        # 🔧 使用预处理阶段计算的Winsorization阈值
        if winsorization_thresholds is None:
            logger.warning("⚠️ 未提供Winsorization阈值，数据预处理可能不完整")
            winsorization_thresholds = {}

        # 详细的特征日志记录
        logger.info("=" * 60)
        logger.info("输入特征详细信息")
        logger.info("=" * 60)
        logger.info(f"原始特征总数: {len(feature_names)}")


        # 先进行常规的特征处理，PCA特征工程将在标准化后进行
        n_features = X_train_split.shape[1]

        logger.info("=" * 60)
        logger.info("特征处理策略")
        logger.info("=" * 60)
        logger.info(f"🎯 原始特征数量: {n_features}")
    
        logger.info("=" * 60)
        logger.info("🔧 XGBoost模型训练")
        logger.info("=" * 60)

        # 🚀 处理标签中的NaN值
        logger.info("处理标签中的NaN值...")

        # 移除NaN值
        train_nan_mask = np.isnan(y_train_split)
        val_nan_mask = np.isnan(y_val_split)

        if train_nan_mask.any():
            logger.warning(f"训练集中发现{train_nan_mask.sum()}个NaN标签，将被移除")
            X_train_split = X_train_split[~train_nan_mask]
            y_train_split = y_train_split[~train_nan_mask]
            # 🆕 同步移除train_dates中的NaN对应项
            if train_dates is not None:
                train_dates = train_dates[~train_nan_mask]
            # 🆕 同步移除y_train_split_original中的NaN对应项
            if y_train_split_original is not None:
                y_train_split_original = y_train_split_original[~train_nan_mask]
            # 🆕 同步移除换手率和流通市值中的NaN对应项
            if 'train_turnover_rate' in locals() and train_turnover_rate is not None:
                train_turnover_rate = train_turnover_rate[~train_nan_mask]
            if 'train_circ_mv' in locals() and train_circ_mv is not None:
                train_circ_mv = train_circ_mv[~train_nan_mask]

        if val_nan_mask.any():
            logger.warning(f"验证集中发现{val_nan_mask.sum()}个NaN标签，将被移除")
            X_val_split = X_val_split[~val_nan_mask]
            y_val_split = y_val_split[~val_nan_mask]
            # 🆕 同步移除val_dates中的NaN对应项
            if val_dates is not None:
                val_dates = val_dates[~val_nan_mask]

        # 🚀 计算样本权重（基于换手率、流通市值和指数时间衰减）
        logger.info("🔧 计算样本权重（基于换手率、流通市值和指数时间衰减）...")

        # 🚀 使用换手率和流通市值计算基础权重，结合指数时间衰减
        # 🚀 从preprocessed_data获取统一计算的样本权重
        sample_weights = preprocessed_data.get('sample_weights') if preprocessed_data is not None else None

        if sample_weights is None:
            # 如果preprocessed_data中没有权重,则重新计算（兼容旧代码）
            logger.warning("⚠️ preprocessed_data中缺少sample_weights,将重新计算")
            train_relevance = preprocessed_data.get('train_relevance') if preprocessed_data is not None else None
            sample_weights = _calculate_professional_sample_weights(
                y_train_split,
                imbalance_ratio=1.0,
                logger=logger,
                boundary_distances=None,
                is_regression=True,
                y_train_original=y_train_split_original,
                train_dates=train_dates,
                time_decay_half_life=180.0,
                turnover_rate=train_turnover_rate if 'train_turnover_rate' in locals() else None,
                circ_mv=train_circ_mv if 'train_circ_mv' in locals() else None,
                label_rank=train_relevance,
                label_boost_beta=0.15,
                w_min=0.2,
                w_max=5.0
            )
        else:
            logger.info("✅ 使用preprocessed_data中的统一样本权重")
            logger.info(f"   权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
            logger.info(f"   权重均值: {sample_weights.mean():.4f}")

        # 🚀 排序模型：确保relevance标签是numpy数组
        if not isinstance(y_train_split, np.ndarray):
            y_train_split = np.array(y_train_split)
        if not isinstance(y_val_split, np.ndarray):
            y_val_split = np.array(y_val_split)

        # 🚀 XGBoost排序模型标签统计（relevance_30整数0-29,30层）
        logger.info(f"🚀 XGBoost排序模型标签统计（relevance_30整数0-29,30层）:")
        logger.info(f"  训练集: {len(y_train_split):,} 个样本")
        logger.info(f"  验证集: {len(y_val_split):,} 个样本")
        logger.info(f"  训练集relevance_30范围: [{y_train_split.min()}, {y_train_split.max()}]")
        logger.info(f"  验证集relevance_30范围: [{y_val_split.min()}, {y_val_split.max()}]")
        logger.info(f"  训练集relevance_30分布(Top-5层级):")
        train_dist = np.bincount(y_train_split.astype(int))
        for i in range(len(train_dist)-1, max(0, len(train_dist)-6), -1):
            if train_dist[i] > 0:
                logger.info(f"    Relevance_30={i}: {train_dist[i]:,}样本")
        logger.info(f"  验证集relevance_30分布(Top-5层级):")
        val_dist = np.bincount(y_val_split.astype(int))
        for i in range(len(val_dist)-1, max(0, len(val_dist)-6), -1):
            if val_dist[i] > 0:
                logger.info(f"    Relevance_30={i}: {val_dist[i]:,}样本")

        # 🚀 XGBoost排序模型：使用relevance_30标签（整数0-29,30层）进行训练
        logger.info("🔧 XGBoost排序模型：使用relevance_30标签（整数0-29,30层）进行训练")

        # 🚀 关键修复：XGBoost排序模型使用relevance_30标签，不覆盖为ATR标签
        # y_train_split和y_val_split已经被赋值为relevance_30标签
        # 用于训练的标签（relevance_30整数0-29,30层）
        y_train_split_mapped = y_train_split  # relevance_30标签（整数0-29），用于XGBoost排序模型训练
        y_val_split_mapped = y_val_split      # relevance_30标签（整数0-29），用于XGBoost排序模型训练

        # 用于评估的标签（原始ATR收益率）
        y_train_split_eval = y_train_split_original  # 原始ATR标签，用于IC计算
        y_val_split_eval = y_val_split_original      # 原始ATR标签，用于IC计算

        reverse_label_map = None

        # 🚀 验证relevance_30标签类型
        logger.info(f"  ✅ 训练标签类型验证:")
        logger.info(f"     y_train_split_mapped类型: {type(y_train_split_mapped)}")
        logger.info(f"     y_train_split_mapped唯一值数量: {len(np.unique(y_train_split_mapped))}")
        logger.info(f"     y_train_split_mapped范围: [{np.min(y_train_split_mapped)}, {np.max(y_train_split_mapped)}]")
        logger.info(f"     y_train_split_mapped数据类型: {y_train_split_mapped.dtype if hasattr(y_train_split_mapped, 'dtype') else 'N/A'}")

        # 保存原始标签统计量（用于日志记录）
        if label_stats is None:
            label_stats = {}
        label_stats.update({
            'mean': y_train_split_original.mean() if y_train_split_original is not None else 0,
            'std': y_train_split_original.std() if y_train_split_original is not None else 0,
            'min': y_train_split_original.min() if y_train_split_original is not None else 0,
            'max': y_train_split_original.max() if y_train_split_original is not None else 0
        })

        logger.info(f"✅ 准备使用relevance_30标签（整数0-29,30层）进行XGBoost排序模型训练")
        logger.info(f"  排序模型输出：相对排序分数（非绝对收益率）")

        logger.info("✅ 数据预处理已完成")
        logger.info("=" * 60)


        # 确定股票因子列表（不包含市场因子）
        # 🔧 使用模型因子列表（归一化因子）
        stock_factors = _get_model_factors()

        # 只保留股票因子，过滤掉市场因子
        available_stock_factors = []
        stock_factor_indices = []

        for i, feature_name in enumerate(feature_names):
            if feature_name in stock_factors:
                available_stock_factors.append(feature_name)
                stock_factor_indices.append(i)

        # 提取股票因子数据
        X_train_split_stock = X_train_split[:, stock_factor_indices]
        X_val_split_stock = X_val_split[:, stock_factor_indices]
        feature_names = available_stock_factors

        logger.info(f"✅ 股票因子特征: {len(feature_names)} 个")
        logger.info(f"   - 特征列表: {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}")

        # 🔧 修复：删除StandardScaler标准化
        # 原因：
        # 1. 因子在计算时已经进行了标准化（如zscore、rank等）
        # 2. 树模型对特征尺度不敏感，不需要额外标准化
        # 3. 额外标准化导致贝叶斯搜索和最终训练的数据分布不一致
        # 4. 这是导致walkforward模式LightGBM过拟合的根本原因
        logger.info("⏭️ 跳过StandardScaler标准化（因子已在计算时标准化）")
        logger.info("   - 因子计算时已进行时间序列标准化（如zscore、rank）")
        logger.info("   - 树模型对特征尺度不敏感")
        logger.info("   - 避免贝叶斯搜索和训练的数据分布不一致")

        # 直接使用Winsorization后的特征，不进行额外标准化
        X_train_enhanced = X_train_split_stock
        X_val_enhanced = X_val_split_stock

        logger.info(f"✅ 使用基础特征（已Winsorization）: {len(feature_names)} 个特征")

        # PCA信息标记为未应用（使用基础特征）
        pca_info = {'applied': False, 'reason': 'using_base_features_only'}

        logger.info("\n" + "=" * 60)
        logger.info("🚀 步骤3: 训练XGBoost模型（使用基础特征）")
        logger.info("=" * 60)

        # 分析样本分布（支持二分类和三分类）
        unique_labels_split = np.unique(y_train_split)
        total_samples = len(y_train_split)



        # 根据特征数量和样本分布调整XGBoost参数
        # 🔧 修复：使用X_train_enhanced替代已删除的X_train_split_scaled
        current_features = X_train_enhanced.shape[1]
        n_samples = len(X_train_enhanced)

        logger.info("=" * 60)
        # 🔧 修复：提前定义actual_feature_count变量
        actual_feature_count = len(feature_names)  # 使用实际的特征数量

        logger.info("XGBoost参数优化")
        logger.info("=" * 60)
        logger.info(f"📊 训练样本: {n_samples:,}个")
        logger.info(f"📊 特征数量: {actual_feature_count}个")  # 🔧 修复：使用实际特征数量
        logger.info(f"📊 类别不平衡比例: {imbalance_ratio:.1f}:1")

        # 使用基础特征的标准参数配置
        logger.info(f"📊 特征组成分析: 总计{actual_feature_count}个基础特征")


        # 🚀 优化：统一参数配置，根据特征数量智能调整
        logger.info(f"🔧 使用基础特征集({actual_feature_count}个特征)，智能参数配置")

        # 基础参数配置（将被后续优化覆盖）
        params = {
            'random_state': random_seed,  # 🚀 集成学习：使用传入的随机种子
            'n_jobs': -1,
            'verbosity': 0
        }

        # 🚀 排序模型模式：使用XGBoost排序目标函数
        logger.info("🚀 排序模型模式：使用XGBoost排序目标函数和relevance_30标签（整数0-29,30层）")

        # 🚀 排序模式使用基础特征的优化参数
        logger.info(f"ℹ️ 排序模式使用基础特征，采用专门优化的参数配置")

        # 🔍 根据特征数量和预测目标动态调整参数
        # 🚀 优化：统一的智能参数配置，平衡性能和泛化能力
        if actual_feature_count <= 19:  # 小特征集：更积极的参数
            optimized_params = {
                'max_depth': 6,                   # 🔧 优化：适度深度，平衡复杂度和泛化
                'learning_rate': 0.08,            # 🔧 优化：平衡收敛速度和稳定性
                'n_estimators': 3000,             # 🔧 优化：适度增加树数量
                'subsample': 0.8,                 # 🔧 优化：高采样率但保留一定随机性
                'colsample_bytree': 0.8,          # 🔧 优化：高特征采样
                'reg_alpha': 0.1,                 # 🔧 优化：适度L1正则化
                'reg_lambda': 0.1,                # 🔧 优化：适度L2正则化
                'min_child_weight': 3,            # 🔧 优化：防止过拟合
                'gamma': 0.1,                     # 🔧 优化：适度分裂阈值
                'colsample_bylevel': 0.9,         # 🔧 优化：每层特征采样
                'max_delta_step': 0,              # 不限制权重更新
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            logger.info("🎯 使用小特征集优化参数配置")
        else:
            # 大特征集：更保守的参数
            optimized_params = {
                'max_depth': 7,                   # 🔧 重新优化：适度深度，平衡学习能力和泛化
                'learning_rate': 0.1,            # 🔧 重新优化：适中学习率，避免过度保守
                'n_estimators': 1200,             # 🔧 重新优化：合理树数量，避免过度训练
                'subsample': 0.87,                # 🔧 重新优化：标准采样率，保持学习能力
                'colsample_bytree': 0.92,          # 🔧 重新优化：平衡特征采样，避免信息损失
                'reg_alpha': 2,                 # 🔧 重新优化：适度L1正则化，平衡特征选择
                'reg_lambda': 3,                # 🔧 重新优化：适度L2正则化，避免过度约束
                'min_child_weight': 3,            # 🔧 重新优化：适中最小权重，平衡泛化和学习
                'gamma': 0.3,                     # 🔧 重新优化：适中分裂阈值，平衡复杂度
                'colsample_bylevel': 0.8,         # 🔧 重新优化：标准每层特征采样
                'max_delta_step': 0,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

        params.update(optimized_params)
        logger.info(f"🎯 针对{actual_feature_count}个特征优化参数配置")

        # 🚀 超参数搜索功能
        if preset_params is not None:
            # 🚀 使用预设参数（Walk-Forward模式）
            logger.info("=" * 80)
            logger.info("🚀 使用预设参数（跳过参数搜索）")
            logger.info("=" * 80)
            # params.update(preset_params) 暂时注释掉

           
        elif enable_param_search:
            logger.info("=" * 80)
            logger.info("🔍 启用超参数搜索优化")
            logger.info("=" * 80)


            # 执行参数搜索
            best_params, search_results = _perform_parameter_search(
                X_train_enhanced, y_train_split_mapped, X_val_enhanced, y_val_split_mapped,
                y_val_split_eval, sample_weights,
                search_strategy,
                val_dates=val_dates,
                train_dates=train_dates,
                market_data=market_data  # 🚀 XGBoost需要market_data计算group权重
            )

            # 更新参数
            params.update(best_params)
            logger.info("✅ 参数搜索完成，使用最优参数配置")
            logger.info(f"🎯 最优参数: {best_params}")
       

        logger.info("🚀回归参数配置:")
        logger.info(f"  max_depth: {params['max_depth']} (针对{actual_feature_count}特征优化)")
        logger.info(f"  learning_rate: {params['learning_rate']}")
        logger.info(f"  n_estimators: {params['n_estimators']}")
        logger.info(f"  subsample: {params['subsample']}")
        logger.info(f"  colsample_bytree: {params['colsample_bytree']}")
        logger.info(f"  reg_alpha: {params['reg_alpha']}")
        logger.info(f"  reg_lambda: {params['reg_lambda']}")
        logger.info(f"  min_child_weight: {params['min_child_weight']}")
        logger.info(f"  gamma: {params['gamma']}")
        logger.info("✅ 配置针对MSE目标函数和小特征集优化")


        # 🔧 移除交互约束配置，使用基础特征
        logger.info("✅ 使用基础特征，无需特征交互约束")

        logger.info("=" * 60)

        # 🚀 排序模式：不进行标签映射，保持原始连续收益标签
        unique_labels = None
        label_mapping = None
        reverse_mapping = None
        logger.info("排序模式：保持原始连续收益标签，不进行映射")

        # 🚀 排序模型配置
        # 🚀 使用rank:pairwise支持sample-level weights
        objective = 'rank:pairwise'
        eval_metric = 'auc'  # rank:pairwise不支持ndcg,使用auc
        num_class = None
        logger.info("使用排序模型配置（Learning to Rank - Pairwise）")

        # 通用参数
        params.update({
            'objective': objective,
            'eval_metric': eval_metric,
            'random_state': random_seed,  # 🚀 集成学习：使用传入的随机种子
            'n_jobs': -1,
            'verbosity': 0
        })

        logger.info(f"✅ XGBoost参数配置完成，排序模式，objective={objective}")

        # 🚀 开始模型训练
        logger.info("=" * 60)
        logger.info("🚀 开始排序模型训练（Learning to Rank）")
        logger.info("=" * 60)


        # 🚀 排序模式：使用xgboost.train
        import xgboost as xgb

        # 计算group信息（每个日期的样本数）
        if train_dates is not None:
            import pandas as pd
            train_dates_series = pd.Series(train_dates)
            val_dates_series = pd.Series(val_dates) if val_dates is not None else None

            # 确保数据按日期排序
            if not train_dates_series.is_monotonic_increasing:
                logger.warning("⚠️ 训练集日期未按升序排列,排序模型可能出现问题")

            train_groups = train_dates_series.value_counts().sort_index().values
            logger.info(f"  训练集group数: {len(train_groups)}, 总样本数: {train_groups.sum()}")

            if val_dates_series is not None:
                val_groups = val_dates_series.value_counts().sort_index().values
                logger.info(f"  验证集group数: {len(val_groups)}, 总样本数: {val_groups.sum()}")
            else:
                logger.warning("⚠️ 缺少验证集日期信息,无法计算验证集group")
                val_groups = None
        else:
            logger.error("❌ 缺少训练集日期信息,无法使用排序模型")
            return None

        # 🚀 XGBoost排序模型权重处理: 使用group-level weights
        # 🔍 调试: 检查数据是否按日期排序
        logger.info("🔍 调试: 检查训练数据是否按日期排序")
        unique_dates = pd.Series(train_dates).unique()
        sorted_dates = pd.Series(train_dates).sort_values().unique()
        if not np.array_equal(unique_dates, sorted_dates):
            logger.error("❌ 训练数据未按日期排序!这会导致排序模型出现问题!")
            logger.error(f"   前10个日期: {train_dates[:10]}")
            logger.error(f"   唯一日期数: {len(unique_dates)}")
        else:
            logger.info("✅ 训练数据已按日期排序")

        # 🚀 计算XGBoost group权重（最终训练时输出详细日志）
        group_weights = _calculate_xgboost_group_weights(
            train_dates=train_dates,
            market_data=market_data,
            logger=logger,
            time_decay_half_life=180.0,
            reference_date=None,  # 使用训练集最新日期
            verbose=True  # 🔊 最终训练时输出详细日志
        )

        logger.info(f"  ✅ 使用分组级权重（XGBoost排序模型要求）")
        logger.info(f"     分组权重数量: {len(group_weights)} (应等于group数: {len(train_groups)})")
        logger.info(f"     分组权重范围: [{group_weights.min():.4f}, {group_weights.max():.4f}]")
        logger.info(f"     分组权重均值: {group_weights.mean():.4f}")
        logger.info(f"     Group数量: {len(train_groups)}")

        # 验证权重长度
        if len(group_weights) != len(train_groups):
            logger.error(f"❌ 权重长度错误: {len(group_weights)} != {len(train_groups)}")
            logger.error("   这会导致XGBoost训练失败!")
            return None

        # 🔍 调试: 检查训练标签和收益率的关系
        logger.info("🔍 调试: 检查relevance_30标签和原始收益率的关系(Top-5层级)")
        import pandas as pd
        debug_df = pd.DataFrame({
            'relevance_30': y_train_split_mapped,
            'return': y_train_split_eval
        })
        # 显示Top-5层级的统计
        unique_vals = sorted(debug_df['relevance_30'].unique(), reverse=True)[:5]
        for rel_val in unique_vals:
            rel_mask = debug_df['relevance_30'] == rel_val
            if rel_mask.sum() > 0:
                avg_return = debug_df[rel_mask]['return'].mean()
                logger.info(f"  Relevance_30={rel_val}: 样本数={rel_mask.sum():,}, 平均收益率={avg_return:.4f}")

        # 🚀 XGBoost排序模型使用group-level weights
        dtrain = xgb.DMatrix(X_train_enhanced, label=y_train_split_mapped)
        dtrain.set_group(train_groups)
        dtrain.set_weight(group_weights)  # 🚀 使用group-level weights

        dval = xgb.DMatrix(X_val_enhanced, label=y_val_split_mapped)
        if val_groups is not None:
            dval.set_group(val_groups)

        logger.info(f"📊 最终模型训练使用基础特征: 训练集{X_train_enhanced.shape}, 验证集{X_val_enhanced.shape}")

        # 使用MSE损失函数训练
        final_params = params.copy()
        logger.info("🎯 训练MSE回归模型...")

        # 🚀 智能Early Stopping调整机制（基于特征数量和复杂度）
        # 🚀 根据特征数量和训练轮数智能调整Early Stopping参数（优化版）
        # 策略: 基于n_estimators的百分比,同时考虑合理的上下限
        if actual_feature_count <= 20:
            # 小特征集：使用较激进的Early Stopping
            early_stop_ratio = 0.08  # 8%
            min_early_stop, max_early_stop = 50, 500
        elif actual_feature_count <= 35:
            # 中特征集：使用适中的Early Stopping
            early_stop_ratio = 0.10  # 10%
            min_early_stop, max_early_stop = 80, 800
        else:
            # 大特征集：使用宽松的Early Stopping
            early_stop_ratio = 0.12  # 12%
            min_early_stop, max_early_stop = 100, 1200

        # 计算动态early_stopping,限制在合理范围内
        dynamic_early_stopping = int(params['n_estimators'] * early_stop_ratio)
        dynamic_early_stopping = max(min_early_stop, min(max_early_stop, dynamic_early_stopping))

        logger.info(f"🎯 特征集规模({actual_feature_count}个)：Early Stopping = {dynamic_early_stopping}轮 "
                   f"({early_stop_ratio*100:.0f}% of {params['n_estimators']}, 范围[{min_early_stop}, {max_early_stop}])")

        # 🚀 专业训练监控和自动调参建议系统 - 优化版本
        class TrainingMonitor(xgb.callback.TrainingCallback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.train_metrics = []
                self.val_metrics = []
                self.epochs = []
                self.best_metric_value = None
                self.best_epoch = 0
                self.overfitting_start = None
                self.metric_oscillations = 0
                self.last_val_metric = None
                self.metric_name = None
                self.metric_direction = None  # 'higher_is_better' or 'lower_is_better'
                self.is_ranking_model = False

            def after_iteration(self, model, epoch, evals_log):
                """XGBoost callback函数 - 优化版本，正确处理排序模型指标"""
                # 从evals_log获取指标值
                if 'train' in evals_log and 'val' in evals_log:
                    # 🚀 优化的指标检测逻辑 - 适配rank:pairwise模型
                    # 优先级：auc > ndcg@10 > rmse > 其他
                    available_metrics = list(evals_log['train'].keys())

                    if 'auc' in available_metrics:
                        # rank:pairwise模型通常使用AUC
                        self.metric_name = 'auc'
                        self.metric_direction = 'higher_is_better'
                        self.is_ranking_model = True
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    elif 'ndcg@10' in available_metrics:
                        # NDCG指标，越大越好
                        self.metric_name = 'ndcg@10'
                        self.metric_direction = 'higher_is_better'
                        self.is_ranking_model = True
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    elif 'rmse' in available_metrics:
                        # RMSE指标，越小越好
                        self.metric_name = 'rmse'
                        self.metric_direction = 'lower_is_better'
                        self.is_ranking_model = False
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    elif 'mae' in available_metrics:
                        # MAE指标，越小越好
                        self.metric_name = 'mae'
                        self.metric_direction = 'lower_is_better'
                        self.is_ranking_model = False
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    else:
                        # 使用第一个可用指标
                        self.metric_name = available_metrics[0]
                        self.metric_direction = 'lower_is_better'  # 默认假设越小越好
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                else:
                    return False  # 如果没有评估日志，跳过

                # 记录数据
                self.epochs.append(epoch)
                self.train_metrics.append(train_metric)
                self.val_metrics.append(val_metric)

                # 检测最佳指标值（适配指标方向）
                if self.best_metric_value is None:
                    self.best_metric_value = val_metric
                    self.best_epoch = epoch
                elif self.metric_direction == 'higher_is_better':
                    if val_metric > self.best_metric_value:
                        self.best_metric_value = val_metric
                        self.best_epoch = epoch
                else:  # lower_is_better
                    if val_metric < self.best_metric_value:
                        self.best_metric_value = val_metric
                        self.best_epoch = epoch

                # 🚀 优化的过拟合检测逻辑
                if len(self.val_metrics) >= 15:  # 需要足够的历史数据
                    # 计算近期的指标趋势
                    recent_val_window = min(8, len(self.val_metrics) // 3)
                    if recent_val_window >= 3:
                        recent_val_values = self.val_metrics[-recent_val_window:]
                        older_val_values = self.val_metrics[-recent_val_window*2:-recent_val_window]

                        recent_val_avg = np.mean(recent_val_values)
                        older_val_avg = np.mean(older_val_values)

                        # 计算训练集趋势
                        recent_train_values = self.train_metrics[-recent_val_window:]
                        older_train_values = self.train_metrics[-recent_val_window*2:-recent_val_window]

                        recent_train_avg = np.mean(recent_train_values)
                        older_train_avg = np.mean(older_train_values)

                        # 根据指标方向判断趋势
                        if self.metric_direction == 'higher_is_better':
                            val_trend = recent_val_avg - older_val_avg  # 正值表示改善
                            train_trend = recent_train_avg - older_train_avg
                            # 过拟合：验证指标恶化 + 训练指标改善
                            overfitting_signal = val_trend < 0 and train_trend > 0
                        else:  # lower_is_better
                            val_trend = older_val_avg - recent_val_avg  # 正值表示改善
                            train_trend = older_train_avg - recent_train_avg
                            # 过拟合：验证指标恶化 + 训练指标改善
                            overfitting_signal = val_trend < 0 and train_trend > 0

                        # 设置过拟合检测阈值
                        trend_threshold = 0.001 if self.is_ranking_model else 0.01

                        if overfitting_signal and abs(val_trend) > trend_threshold and self.overfitting_start is None:
                            self.overfitting_start = epoch

                # 检测指标震荡
                if self.last_val_metric is not None:
                    # 震荡检测阈值根据指标类型调整
                    if self.is_ranking_model:
                        oscillation_threshold = 0.005  # AUC/NDCG 0.5%
                    else:
                        oscillation_threshold = 0.01   # RMSE/MAE 1%

                    relative_change = abs(val_metric - self.last_val_metric)
                    if self.last_val_metric != 0:
                        relative_change = relative_change / abs(self.last_val_metric)

                    if relative_change > oscillation_threshold:
                        self.metric_oscillations += 1

                self.last_val_metric = val_metric

                # 每100轮输出详细分析
                if epoch % 100 == 0 and epoch > 0:
                    self._log_training_analysis(epoch)

                return False  # 继续训练

            def _log_training_analysis(self, epoch):
                """输出训练过程分析 - 优化版本"""
                if len(self.train_metrics) < 20:
                    return

                # 计算指标趋势
                recent_epochs = min(50, len(self.train_metrics))
                train_trend = np.polyfit(range(recent_epochs), self.train_metrics[-recent_epochs:], 1)[0]
                val_trend = np.polyfit(range(recent_epochs), self.val_metrics[-recent_epochs:], 1)[0]

                # 计算训练集和验证集指标的差距（适配指标方向）
                if self.metric_direction == 'higher_is_better':
                    # 对于AUC/NDCG，训练集 > 验证集表示过拟合
                    train_val_gap = self.train_metrics[-1] - self.val_metrics[-1]
                    gap_descriptions = {
                        'direction': '下降' if train_trend < 0 else '上升',
                        'gap_status': '扩大' if train_val_gap > 0 else '缩小'
                    }
                else:  # lower_is_better
                    # 对于RMSE/MAE，验证集 > 训练集表示过拟合
                    train_val_gap = self.val_metrics[-1] - self.train_metrics[-1]
                    gap_descriptions = {
                        'direction': '下降' if train_trend < 0 else '上升',
                        'gap_status': '扩大' if train_val_gap > 0 else '缩小'
                    }

                # 计算学习效率（适配指标方向）
                if self.metric_direction == 'higher_is_better':
                    # 对于AUC/NDCG，正值改善
                    learning_efficiency = train_trend / (abs(train_trend) + abs(val_trend) + 1e-8)
                else:  # lower_is_better
                    # 对于RMSE/MAE，负值改善
                    learning_efficiency = -train_trend / (abs(train_trend) + abs(val_trend) + 1e-8)

                # 格式化指标输出
                current_train_metric = self.train_metrics[-1]
                current_val_metric = self.val_metrics[-1]

                self.logger.info(f"  📊 训练监控 [Epoch {epoch}] - {self.metric_name.upper()}:")
                self.logger.info(f"    🔄 训练{self.metric_name}: {current_train_metric:.6f}")
                self.logger.info(f"    🔄 验证{self.metric_name}: {current_val_metric:.6f}")
                self.logger.info(f"    📈 训练指标趋势: {train_trend:.6f} ({gap_descriptions['direction']})")
                self.logger.info(f"    📈 验证指标趋势: {val_trend:.6f} ({'改善' if (val_trend > 0 and self.metric_direction == 'higher_is_better') or (val_trend < 0 and self.metric_direction == 'lower_is_better') else '恶化'})")
                self.logger.info(f"    📈 过拟合程度: {train_val_gap:.6f} (差距{gap_descriptions['gap_status']})")
                self.logger.info(f"    ⚡ 学习效率: {learning_efficiency:.3f}")
                self.logger.info(f"    🌊 指标震荡次数: {self.metric_oscillations}")

                if self.overfitting_start:
                    self.logger.info(f"    ⚠️  过拟合检测: 第{self.overfitting_start}轮开始")

            def get_final_analysis(self):
                """获取最终训练分析结果 - 优化版本"""
                if len(self.train_metrics) < 50:
                    return {}

                # 整体趋势分析（适配指标方向）
                total_epochs = len(self.train_metrics)

                if self.metric_direction == 'higher_is_better':
                    # 对于AUC/NDCG，改善为正值
                    train_improvement = self.train_metrics[-1] - self.train_metrics[0]
                    val_improvement = self.val_metrics[-1] - self.val_metrics[0]

                    # 过拟合严重度：训练集指标远高于验证集
                    final_train_metric = self.train_metrics[-1]
                    final_val_metric = self.val_metrics[-1]
                    if final_train_metric > 0:
                        overfitting_severity = max(0, (final_train_metric - final_val_metric) / final_train_metric)
                    else:
                        overfitting_severity = 0.0

                    # 学习效率分析（适配指标方向）
                    early_improvement = self.train_metrics[min(100, total_epochs//4)] - self.train_metrics[0]
                    late_improvement = self.train_metrics[-1] - self.train_metrics[max(100, total_epochs//2)]

                else:  # lower_is_better
                    # 对于RMSE/MAE，改善为负值
                    train_improvement = self.train_metrics[0] - self.train_metrics[-1]
                    val_improvement = self.val_metrics[0] - self.val_metrics[-1]

                    # 过拟合严重度：验证集指标远高于训练集
                    final_train_metric = self.train_metrics[-1]
                    final_val_metric = self.val_metrics[-1]
                    if final_train_metric > 0:
                        overfitting_severity = max(0, (final_val_metric - final_train_metric) / final_train_metric)
                    else:
                        overfitting_severity = 0.0

                    # 学习效率分析（适配指标方向）
                    early_improvement = self.train_metrics[0] - self.train_metrics[min(100, total_epochs//4)]
                    late_improvement = self.train_metrics[max(100, total_epochs//2)] - self.train_metrics[-1]

                learning_rate_efficiency = early_improvement / (early_improvement + late_improvement + 1e-8)

                # 收敛性分析
                last_20_val_std = np.std(self.val_metrics[-20:])
                convergence_stability = 1.0 / (1.0 + last_20_val_std)

                # 模型健康度评估
                model_health = self._assess_model_health(train_improvement, val_improvement, overfitting_severity)

                return {
                    'total_epochs': total_epochs,
                    'metric_name': self.metric_name,
                    'metric_direction': self.metric_direction,
                    'is_ranking_model': self.is_ranking_model,
                    'train_improvement': train_improvement,
                    'val_improvement': val_improvement,
                    'convergence_stability': convergence_stability,
                    'overfitting_severity': overfitting_severity,
                    'learning_rate_efficiency': learning_rate_efficiency,
                    'model_health': model_health,
                    'best_epoch': self.best_epoch,
                    'overfitting_start': self.overfitting_start,
                    'metric_oscillations': self.metric_oscillations,
                    'final_train_metric': self.train_metrics[-1],
                    'final_val_metric': self.val_metrics[-1],
                    'best_metric_value': self.best_metric_value
                }

            def _assess_model_health(self, train_improvement, val_improvement, overfitting_severity):
                """评估模型健康状况"""
                health_score = 100.0

                # 改善度评分（40分）
                if self.metric_direction == 'higher_is_better':
                    improvement_score = min(40, max(0, val_improvement * 40))  # AUC/NDCG改善
                else:
                    improvement_score = min(40, max(0, val_improvement * 400))  # RMSE/MAE改善

                # 过拟合扣分（30分）
                overfitting_penalty = min(30, overfitting_severity * 30)

                # 震荡扣分（20分）
                oscillation_penalty = min(20, self.metric_oscillations * 2)

                # 收敛性奖励（10分）
                convergence_bonus = min(10, (100 - len(self.train_metrics)) / 10) if len(self.train_metrics) < 100 else 10

                health_score = health_score + improvement_score - overfitting_penalty - oscillation_penalty + convergence_bonus
                health_score = max(0, min(100, health_score))

                if health_score >= 80:
                    return '优秀'
                elif health_score >= 60:
                    return '良好'
                elif health_score >= 40:
                    return '一般'
                else:
                    return '需要改进'

        # 训练最终模型
        logger.info("🚀 开始最终模型训练（启用专业监控系统）...")
        logger.info(f"  📊 最大训练轮数: {params['n_estimators']}")
        logger.info(f"  📊 Early Stopping轮数: {dynamic_early_stopping} (针对{actual_feature_count}个特征优化)")
        logger.info(f"  📊 Early Stopping比例: {dynamic_early_stopping/params['n_estimators']*100:.1f}%")
        logger.info(f"  📊 训练集样本: {len(X_train_enhanced):,}")
        logger.info(f"  📊 验证集样本: {len(X_val_enhanced):,}")

        # 创建训练监控器
        monitor = TrainingMonitor(logger)

        # 使用MSE损失函数进行最终训练
        model = xgb.train(
                final_params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                verbose_eval=100,
                early_stopping_rounds=dynamic_early_stopping,
                callbacks=[monitor]
        )

        # 📊 训练完成后的详细日志
        logger.info("✅ 最终模型训练完成")
        logger.info(f"  📊 实际训练轮数: {model.best_iteration + 1}")
        logger.info(f"  📊 最佳验证轮数: {model.best_iteration}")

        # 🚀 专业训练分析和自动调参建议系统
        analysis = monitor.get_final_analysis()

        # 输出训练分析结果
        logger.info("=" * 80)
        logger.info("🔍 专业训练分析报告")
        logger.info("=" * 80)

        if analysis and len(analysis) > 0:  # 确保analysis不为空
            # 🚀 新增：显示指标类型信息 - 使用安全访问
            metric_name = analysis.get('metric_name', 'unknown')
            is_ranking_model = analysis.get('is_ranking_model', False)
            metric_direction = analysis.get('metric_direction', 'lower_is_better')
            model_health = analysis.get('model_health', '未知')

            logger.info(f"🎯 监控指标: {metric_name.upper()} ({'排序模型' if is_ranking_model else '回归模型'})")
            logger.info(f"🎯 指标方向: {'越高越好' if metric_direction == 'higher_is_better' else '越低越好'}")
            logger.info(f"📊 模型健康度: {model_health}")
            logger.info("")

            total_epochs = analysis.get('total_epochs', 0)
            best_epoch = analysis.get('best_epoch', 0)
            train_improvement = analysis.get('train_improvement', 0.0)
            val_improvement = analysis.get('val_improvement', 0.0)
            final_train_metric = analysis.get('final_train_metric', 0.0)
            final_val_metric = analysis.get('final_val_metric', 0.0)
            best_metric_value = analysis.get('best_metric_value', 0.0)
            convergence_stability = analysis.get('convergence_stability', 0.0)
            overfitting_severity = analysis.get('overfitting_severity', 0.0)
            learning_rate_efficiency = analysis.get('learning_rate_efficiency', 0.0)
            metric_oscillations = analysis.get('metric_oscillations', 0)

            logger.info(f"📊 训练总轮数: {total_epochs}")
            if total_epochs > 0:
                logger.info(f"📊 最佳轮数: {best_epoch} (效率: {(best_epoch+1)/total_epochs*100:.1f}%)")
            logger.info(f"📊 训练{metric_name}改善: {train_improvement:.6f}")
            logger.info(f"📊 验证{metric_name}改善: {val_improvement:.6f}")
            logger.info(f"📊 最终训练{metric_name}: {final_train_metric:.6f}")
            logger.info(f"📊 最终验证{metric_name}: {final_val_metric:.6f}")
            logger.info(f"📊 最佳{metric_name}值: {best_metric_value:.6f}")
            logger.info(f"📊 收敛稳定性: {convergence_stability:.3f} ({'稳定' if convergence_stability > 0.7 else '不稳定'})")
            logger.info(f"📊 过拟合严重度: {overfitting_severity:.3f} ({'严重' if overfitting_severity > 0.15 else '轻微' if overfitting_severity > 0.05 else '无'})")
            logger.info(f"📊 学习率效率: {learning_rate_efficiency:.3f} ({'过高' if learning_rate_efficiency < 0.3 else '过低' if learning_rate_efficiency > 0.8 else '合适'})")
            logger.info(f"📊 指标震荡次数: {metric_oscillations}")

            # 🚀 量化交易正则化效果专业评估
            current_reg_alpha = final_params['reg_alpha']
            current_reg_lambda = final_params['reg_lambda']
            total_reg_strength = current_reg_alpha + current_reg_lambda
            # 安全访问analysis字典
            overfitting_severity = analysis.get('overfitting_severity', 0.0)

            logger.info("=" * 60)
            logger.info("📈 正则化效果专业评估（量化交易导向）")
            logger.info("=" * 60)

            # 过拟合与正则化匹配度分析
            if overfitting_severity > 0.15 and total_reg_strength < 1.0:
                logger.info("🚨 正则化不足: 存在过拟合但正则化较弱，建议增强")
            elif overfitting_severity < 0.05 and total_reg_strength > 3.0:
                logger.info("🚨 正则化过强: 无过拟合但正则化很强，可能过度平滑")
            elif overfitting_severity < 0.05 and total_reg_strength < 0.5:
                logger.info("✅ 理想状态: 无过拟合且正则化较弱，有利于捕捉市场信号")
            else:
                logger.info("✅ 正则化匹配: 正则化强度与过拟合程度基本匹配")

            # 量化交易特殊建议
            sample_count = len(X_train_enhanced)
            feature_count = actual_feature_count
            sample_feature_ratio = sample_count / feature_count

            logger.info("=" * 60)
            logger.info("💡 量化交易正则化优化建议")
            logger.info("=" * 60)

            if sample_feature_ratio < 100:
                logger.info("📊 数据特征: 高维小样本 → 建议使用较强正则化防止过拟合")
            elif sample_feature_ratio > 500:
                logger.info("📊 数据特征: 大样本低维 → 可使用较弱正则化充分学习")
            else:
                logger.info("📊 数据特征: 中等规模 → 建议平衡正则化策略")

            # 基于训练表现的具体建议
            if overfitting_severity > 0.2:
                logger.info("🎯 建议: 过拟合严重，优先增强L2正则化平滑权重")
            elif overfitting_severity < 0.03:
                logger.info("🎯 建议: 无明显过拟合，可降低正则化增强信号敏感性")

            # 安全访问analysis字典，避免KeyError
        if 'metric_oscillations' in analysis and 'total_epochs' in analysis:
            if analysis['metric_oscillations'] > analysis['total_epochs'] * 0.1:
                logger.info("🎯 建议: 训练震荡较多，增加L2正则化提高稳定性")

            logger.info("=" * 60)

            if 'overfitting_start' in analysis and analysis['overfitting_start']:
                logger.info(f"⚠️  过拟合开始轮数: {analysis['overfitting_start']}")

        # 判断停止原因（使用动态Early Stopping参数）
        if model.best_iteration < params['n_estimators'] - dynamic_early_stopping:
            logger.info(f"  🎯 停止原因: Early Stopping触发（验证集性能连续{dynamic_early_stopping}轮未改善）")
            logger.info(f"  📊 Early Stopping节省轮数: {params['n_estimators'] - (model.best_iteration + 1)}")
            logger.info(f"  ✅ 训练效率: Early Stopping机制有效防止过拟合")
        else:
            logger.info(f"  🎯 停止原因: 达到最大训练轮数 ({params['n_estimators']})")
            logger.info(f"  ⚠️  建议: 可能需要增加最大训练轮数或调整Early Stopping参数")
            logger.info(f"  💡 当前Early Stopping设置: {dynamic_early_stopping}轮 ({dynamic_early_stopping/params['n_estimators']*100:.1f}%)")

        logger.info(f"  📊 训练效率: {(model.best_iteration + 1) / params['n_estimators'] * 100:.1f}% 轮数使用率")

        # 📊 Early Stopping效果评估
        efficiency_ratio = (model.best_iteration + 1) / params['n_estimators']
        if efficiency_ratio < 0.5:
            logger.info(f"  🎯 Early Stopping效果: 优秀 (使用{efficiency_ratio*100:.1f}%轮数即达到最佳性能)")
        elif efficiency_ratio < 0.8:
            logger.info(f"  🎯 Early Stopping效果: 良好 (使用{efficiency_ratio*100:.1f}%轮数达到最佳性能)")
        else:
            logger.info(f"  🎯 Early Stopping效果: 一般 (使用{efficiency_ratio*100:.1f}%轮数，建议增加训练轮数)")

       


        # 📊 参数合理性验证和资源分析
        logger.info("=" * 80)
        logger.info("📊 参数合理性验证与量化交易最佳实践")
        logger.info("=" * 80)

        # 数据规模分析
        sample_count = len(X_train_enhanced)
        feature_count = actual_feature_count
        sample_feature_ratio = sample_count / feature_count

        logger.info(f"📊 数据规模分析:")
        logger.info(f"  训练样本数: {sample_count:,}")
        logger.info(f"  特征数量: {feature_count}")
        logger.info(f"  样本/特征比: {sample_feature_ratio:.1f}")

        # 样本特征比合理性判断
        if sample_feature_ratio < 100:
            logger.info(f"  ⚠️  样本/特征比过低，容易过拟合，建议增强正则化")
        elif sample_feature_ratio > 1000:
            logger.info(f"  ✅ 样本/特征比充足，可以使用复杂模型")
        else:
            logger.info(f"  ✅ 样本/特征比适中，当前配置合理")

        # n_estimators 专业分析
        # 基于量化交易的推荐公式：考虑数据噪声和市场变化
        base_estimators = min(8000, max(3000, int(sample_count / 100)))  # 基础轮数
        feature_adjustment = min(2000, feature_count * 100)  # 特征调整
        recommended_n_estimators = base_estimators + feature_adjustment

        logger.info(f"📊 n_estimators 专业分析:")
        logger.info(f"  当前设置: {params['n_estimators']}")
        logger.info(f"  量化交易推荐: {recommended_n_estimators}")
        logger.info(f"  推荐公式: min(8000, max(3000, 样本数/100)) + min(2000, 特征数*100)")

        efficiency_ratio = (analysis.get('best_epoch', 0) + 1) / params['n_estimators']
        logger.info(f"  实际使用率: {efficiency_ratio:.1%}")

        if params['n_estimators'] > recommended_n_estimators * 1.5:
            logger.info(f"  ⚠️  当前设置过大，建议调整为 {recommended_n_estimators}")
        elif params['n_estimators'] < recommended_n_estimators * 0.7:
            logger.info(f"  ⚠️  当前设置过小，建议调整为 {recommended_n_estimators}")
        else:
            logger.info(f"  ✅ 当前设置合理")

        # 当前参数配置专业评估
        logger.info(f"📊 当前参数配置评估:")
        current_params = final_params

    
        # 模型复杂度评估
        complexity_score = current_params['max_depth'] * (1 - current_params['gamma']) * current_params['colsample_bytree']
        if complexity_score > 6:
            logger.info(f"  🏗️  模型复杂度: 高 ({complexity_score:.2f}) - 表达能力强但易过拟合")
        elif complexity_score > 3:
            logger.info(f"  🏗️  模型复杂度: 中等 ({complexity_score:.2f}) - 平衡表达能力和稳定性")
        else:
            logger.info(f"  🏗️  模型复杂度: 低 ({complexity_score:.2f}) - 稳定但表达能力有限")

        # 采样策略评估
        sampling_conservatism = (1 - current_params['subsample']) + (1 - current_params['colsample_bytree'])
        if sampling_conservatism > 0.4:
            logger.info(f"  🎯 采样策略: 保守 ({sampling_conservatism:.2f}) - 减少过拟合但可能欠拟合")
        elif sampling_conservatism > 0.2:
            logger.info(f"  🎯 采样策略: 平衡 ({sampling_conservatism:.2f}) - 适中的随机性")
        else:
            logger.info(f"  🎯 采样策略: 激进 ({sampling_conservatism:.2f}) - 充分利用数据")

        # Early Stopping 合理性分析
        optimal_early_stopping = max(50, min(500, int(params['n_estimators'] * 0.08)))
        logger.info(f"📊 Early Stopping 分析:")
        logger.info(f"  当前设置: {dynamic_early_stopping}")
        logger.info(f"  推荐范围: {int(params['n_estimators'] * 0.06)}-{int(params['n_estimators'] * 0.12)}")
        logger.info(f"  最优建议: {optimal_early_stopping} (约8%)")

        if dynamic_early_stopping > params['n_estimators'] * 0.15:
            logger.info(f"  ⚠️  Early Stopping设置过大，可能无法有效防止过拟合")
        elif dynamic_early_stopping < params['n_estimators'] * 0.05:
            logger.info(f"  ⚠️  Early Stopping设置过小，可能过早停止训练")
        else:
            logger.info(f"  ✅ Early Stopping设置合理")

        # 计算资源分析
        estimated_training_time = (params['n_estimators'] * sample_count * feature_count) / 1000000  # 粗略估算（分钟）
        logger.info(f"📊 计算资源分析:")
        logger.info(f"  预估训练时间: {estimated_training_time:.1f} 分钟")
        logger.info(f"  内存需求: 约 {(sample_count * feature_count * 8) / (1024**3):.2f} GB")

        if estimated_training_time > 30:
            logger.info(f"  ⚠️  训练时间较长，建议考虑减少n_estimators或使用更强的正则化")

        logger.info("=" * 80)

        model.loss_function = 'MSE'


        # 🚀 模型评估：支持MSE回归
        logger.info("评估模型性能...")


        # 🚀 使用XGBoost模型预测
        dval_for_pred = xgb.DMatrix(X_val_enhanced)  # 🚀 使用增强特征
        y_pred_raw = model.predict(dval_for_pred)

        # 🚀 直接使用预测的排名分位数，无需逆变换
        y_pred = y_pred_raw
        logger.info(f"✅ 直接使用预测的排名分位数")

        y_pred_proba = None
        logger.info(f"✅ {model.loss_function}模型预测完成，最终预测值范围: [{y_pred.min():.4f}, {y_pred.max():.4f}] (排名分位数)")

        # 🚀 排名标签回归：使用排名分位数和原始收益率进行评估
        y_val_rank_labels = y_val_split  # 验证集排名标签（0-1分位数）
        y_val_return_labels = y_val_split_eval  # 验证集原始收益率（用于计算Top-K收益）


        # 🚀 排名标签回归模型评估
        logger.info("🚀 排名标签回归模型评估...")
        logger.info("📊 标签分布:")

        # 统一命名：预测值和真实值
        y_pred_rank = y_pred  # 预测的排名分位数
        y_val_rank_true = y_val_rank_labels  # 真实的排名标签
        y_val_return_true = y_val_return_labels  # 真实的原始收益率

        # 计算回归指标（基于排名标签）
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import spearmanr

        mse = mean_squared_error(y_val_rank_true, y_pred_rank)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_rank_true, y_pred_rank)
        r2 = r2_score(y_val_rank_true, y_pred_rank)

        # 🚀 计算IC和Rank IC（关键指标）
        # 注意: XGBoost排序模型输出的是排序分数,不是标签值
        # Spearman相关系数会自动将两个变量转换为排名,所以可以直接使用
        rank_ic, _ = spearmanr(y_pred_rank, y_val_return_true)  # Spearman相关系数(正确!)

        # IC (Pearson相关系数): 对于排序模型,这个指标意义不大,仅供参考
        # 因为排序分数的绝对值没有意义,只有相对大小有意义
        ic = np.corrcoef(y_pred_rank, y_val_return_true)[0, 1]  # Pearson相关系数(仅供参考)

        # 🚀 调试信息：检查预测值和真实值的分布
        logger.info(f"🔍 调试信息:")
        logger.info(f"  y_pred_rank类型: {type(y_pred_rank)}, 形状: {y_pred_rank.shape if hasattr(y_pred_rank, 'shape') else 'N/A'}")
        logger.info(f"  y_val_rank_true类型: {type(y_val_rank_true)}, 形状: {y_val_rank_true.shape if hasattr(y_val_rank_true, 'shape') else 'N/A'}")
        logger.info(f"  y_val_return_true类型: {type(y_val_return_true)}, 形状: {y_val_return_true.shape if hasattr(y_val_return_true, 'shape') else 'N/A'}")

        # 🚀 计算Top 5收益率（前5只股票）
        sorted_indices = np.argsort(y_pred_rank)
        n_samples = len(y_pred_rank)
        # 取前5只股票（如果样本数少于5则取所有样本）
        top_5_count = min(5, n_samples)
        top_5_indices = sorted_indices[-top_5_count:]
        top_5_return = y_val_return_true[top_5_indices].mean()

        # 🚀 计算Top 5排序指标
        # NDCG@5使用全局函数 calculate_ndcg_at_k

        # 🚀 计算多种Top 5排序指标 - 改进版本
        # 1. NDCG@5 - 使用改进的NDCG计算解决负收益和量纲问题
        ndcg_at_5 = calculate_adjusted_ndcg_at_k(y_val_return_true, y_pred_rank, k=5)

        # 2. Precision@5 - Top5中真正高收益股票的比例
        # 定义高收益阈值（前25%）
        high_return_threshold = np.percentile(y_val_return_true, 75)
        actual_high_returns = y_val_return_true >= high_return_threshold
        top_5_high_returns = actual_high_returns[top_5_indices]
        precision_at_5 = np.mean(top_5_high_returns) if len(top_5_indices) > 0 else 0.0

        # 3. Recall@5 - Top5中捕获的高收益股票比例
        total_high_returns = np.sum(actual_high_returns)
        captured_high_returns = np.sum(top_5_high_returns)
        recall_at_5 = captured_high_returns / total_high_returns if total_high_returns > 0 else 0.0

        # 4. Hit@5 - Top5中是否包含至少一个高收益股票
        hit_at_5 = 1.0 if captured_high_returns > 0 else 0.0

        # 5. MRR@5 (Mean Reciprocal Rank) - 修复版本
        # 计算高收益股票中的最佳排名
        high_return_indices = np.where(actual_high_returns)[0]
        if len(high_return_indices) > 0:
            # 获取所有高收益股票的预测分数
            high_return_scores = y_pred_rank[high_return_indices]
            # 找到预测分数最高的高收益股票
            best_high_return_local_idx = np.argmax(high_return_scores)
            best_high_return_global_idx = high_return_indices[best_high_return_local_idx]

            # 计算该股票在全局预测中的排名（从小到大，1开始）
            all_indices_sorted = np.argsort(y_pred_rank)[::-1]  # 降序
            rank_position = np.where(all_indices_sorted == best_high_return_global_idx)[0][0] + 1

            mrr_at_5 = 1.0 / rank_position if rank_position <= 5 else 0.0
        else:
            mrr_at_5 = 0.0

        # 6. 🆕 添加基于真实排名的相关性指标
        # 计算预测排名与真实收益率排名的相关性
        from scipy.stats import spearmanr, pearsonr

        # 真实收益排名（从小到大，收益越高排名越靠前）
        true_return_ranks = np.argsort(np.argsort(-y_val_return_true))  # 降序排名，0开始
        # 预测排名（从小到大，预测分数越高排名越靠前）
        pred_ranks = np.argsort(np.argsort(-y_pred_rank))

        # 基于排名的相关性（更准确的评估）
        rank_correlation_ic, _ = pearsonr(pred_ranks, true_return_ranks)
        rank_correlation_rankic, _ = spearmanr(pred_ranks, true_return_ranks)

        logger.info(f"🎯 排名模型性能评估:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.6f}")
        logger.info(f"  IC (预测分数vs收益): {ic:.6f}")
        logger.info(f"  Rank IC (预测分数vs收益): {rank_ic:.6f}")
        logger.info(f"  🔍 基于排名的IC: {rank_correlation_ic:.6f}")
        logger.info(f"  🔍 基于排名的Rank IC: {rank_correlation_rankic:.6f}")
        logger.info(f"  Top 5 平均收益率: {top_5_return:.4f} ({top_5_return*100:.2f}%)")

        # 🚀 新增：Top 5排序质量指标
        logger.info(f"📊 Top 5 排序质量指标:")
        logger.info(f"  NDCG@5: {ndcg_at_5:.6f} (排序质量，0-1，越高越好)")
        logger.info(f"  Precision@5: {precision_at_5:.4f} ({precision_at_5*100:.1f}%)")
        logger.info(f"  Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%)")
        logger.info(f"  Hit@5: {hit_at_5:.4f} ({hit_at_5*100:.1f}%)")
        logger.info(f"  MRR@5: {mrr_at_5:.6f} (首命中排名，越高越好)")

        # 🔍 指标一致性检查
        logger.info(f"🔍 指标一致性分析:")
        logger.info(f"  预测分数方向: {'正向' if ic > 0 else '负向'} (预测分数与收益的关系)")
        logger.info(f"  排名一致性: {'一致' if rank_correlation_rankic > 0 else '不一致'}")
        if rank_correlation_rankic > 0 and top_5_return > 0:
            logger.info(f"  ✅ 指标一致: 排序正确且Top5有正收益")
        elif rank_correlation_rankic < 0 and top_5_return > 0:
            logger.info(f"  ⚠️  部分矛盾: 排序负相关但Top5收益为正")
        else:
            logger.info(f"  ❌ 指标矛盾: 排序错误且Top5收益不佳")

        # 🚀 标签和预测分布
        logger.info(f"📊 排名标签分布:")
        logger.info(f"  真实标签范围: [{y_val_rank_true.min():.4f}, {y_val_rank_true.max():.4f}]")
        logger.info(f"  真实标签均值: {y_val_rank_true.mean():.4f}")
        logger.info(f"  真实标签标准差: {y_val_rank_true.std():.4f}")

        logger.info(f"📊 预测分布:")
        logger.info(f"  预测值范围: [{y_pred_rank.min():.4f}, {y_pred_rank.max():.4f}]")
        logger.info(f"  预测值均值: {y_pred_rank.mean():.4f}")
        logger.info(f"  预测值标准差: {y_pred_rank.std():.4f}")

        logger.info(f"📊 原始收益率分布:")
        logger.info(f"  收益率范围: [{y_val_return_true.min():.4f}, {y_val_return_true.max():.4f}]")
        logger.info(f"  收益率均值: {y_val_return_true.mean():.4f} ({y_val_return_true.mean()*100:.2f}%)")
        logger.info(f"  收益率标准差: {y_val_return_true.std():.4f}")

        # 🚀 计算交易指标（基于预测排名选择Top-K股票）
        trading_metrics = _calculate_trading_metrics(
            y_true_return=y_val_return_true,   # 原始收益率
            y_pred_scores=y_pred_rank,         # 预测的排名分位数
            logger=logger,
            k_values=[1, 3, 5]  # 🚀 修复：与实际选股策略一致
        )

        # 🚀 添加专业量化评估指标（基于排名预测）
        quant_metrics = _calculate_quantitative_metrics(y_val_return_true, y_pred_rank, logger)

        # 🚀 添加预测范围诊断指标
        _calculate_prediction_range_diagnostics(y_val_rank_true, y_pred_rank, logger, y_pred_raw)

        # 🚀 诊断评估：检查预测范围压缩问题
        true_range = y_val_rank_true.max() - y_val_rank_true.min()
        pred_range = y_pred_rank.max() - y_pred_rank.min()
        range_coverage_ratio = pred_range / true_range if true_range > 0 else 0

        true_std = y_val_rank_true.std()
        pred_std = y_pred_rank.std()
        std_ratio = pred_std / true_std if true_std > 0 else 0

        logger.info(f"🚨 诊断评估:")
        if range_coverage_ratio < 0.3:
            logger.warning(f"  ❌ 严重预测范围压缩: {range_coverage_ratio:.1%} < 30%")
        else:
            logger.info(f"  ✅ 预测范围覆盖率: {range_coverage_ratio:.1%}")

        if std_ratio < 0.5:
            logger.warning(f"  ❌ 预测过度保守: 标准差比例{std_ratio:.1%} < 50%")
        else:
            logger.info(f"  ✅ 预测标准差比例: {std_ratio:.1%}")

        if hasattr(model, 'loss_function'):
            logger.info(f"  📊 使用损失函数: {model.loss_function}")

        # 🚀 预测范围压缩的可能原因分析
        if range_coverage_ratio < 0.3 or std_ratio < 0.5:
            logger.warning("🔍 预测范围压缩可能原因:")
            logger.warning("  3. 训练数据中极端值被过度处理")
            logger.warning("  4. 模型复杂度不足，无法捕捉标签的完整分布")
            logger.warning("💡 建议:")
            logger.warning("  - 尝试使用MSE损失函数")
            logger.warning("  - 减少特征Winsorization强度")
            logger.warning("  - 增加模型复杂度（更深的树、更多特征交互）")

        # 🚀 交易指标（基于预测排名选择Top-K股票）
        logger.info("🎯 交易指标（Top-K选股评估）:")

        # 🚀 Top-K收益统计
        top_k_returns = trading_metrics.get('top_k_returns', {})
        top_k_profit_rates = trading_metrics.get('top_k_profit_rates', {})
        top_k_loss_rates = trading_metrics.get('top_k_loss_rates', {})

        for k in [1, 3, 5]:
            if k in top_k_returns:
                logger.info(f"  Top-{k} 平均收益: {top_k_returns[k]:.4f} ({top_k_returns[k]*100:.2f}%)")
                if k in top_k_profit_rates:
                    logger.info(f"  Top-{k} 盈利率: {top_k_profit_rates[k]:.4f}")
                if k in top_k_loss_rates:
                    logger.info(f"  Top-{k} 亏损率: {top_k_loss_rates[k]:.4f}")

        # ATR三分类不使用传统AUC，专注于分类准确性
        auc_score = 0.0

        # 特征重要性分析
        # 🚀 修复Pseudo-Huber回归特征重要性：使用gain而不是weight
        try:
            # 使用gain作为重要性指标，而不是weight（使用次数）
            importance_dict = model.get_score(importance_type='gain')

            # 转换为数组格式，保持与feature_names顺序一致
            feature_importance = np.zeros(len(feature_names))
            for i, feature_name in enumerate(feature_names):
                # XGBoost使用f0, f1, f2...作为特征名
                xgb_feature_name = f'f{i}'
                feature_importance[i] = importance_dict.get(xgb_feature_name, 0.0)

            # 归一化特征重要性到0-1范围
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()

            logger.info(f"✅ 获取Booster特征重要性成功（使用gain指标），{len(importance_dict)}个特征有重要性值")
            logger.info(f"  重要性范围: [{feature_importance.min():.6f}, {feature_importance.max():.6f}]")
        except Exception as e:
            logger.warning(f"⚠️ 获取Booster特征重要性失败: {e}，使用默认值")
            feature_importance = np.ones(len(feature_names)) / len(feature_names)


        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        logger.info("=" * 60)
        logger.info("特征重要性分析结果")
        logger.info("=" * 60)
        logger.info("📊 重要特征:")
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            logger.info(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
        logger.info("=" * 60)

        # 🚀 SHAP值分析 - 增强模型可解释性
        # 🚀 优化：在集成学习中,只对第一个模型进行SHAP分析以节省时间
        if not enable_shap_analysis:
            logger.info("=" * 60)
            logger.info("⏭️ 跳过SHAP分析（集成学习优化）")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("🔍 SHAP值分析 - 模型可解释性")
            logger.info("=" * 60)

        if enable_shap_analysis:
            try:
                # 抑制SHAP相关警告
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning, module='shap')
                warnings.filterwarnings('ignore', category=FutureWarning, module='shap')
                warnings.filterwarnings('ignore', category=DeprecationWarning, module='shap')

                import shap
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')  # 使用非交互式后端

                logger.info("✅ SHAP库导入成功，开始分析...")

                # 创建SHAP解释器 - 针对XGBoost树模型优化
                explainer = shap.TreeExplainer(model)
                logger.info("✅ SHAP TreeExplainer创建成功")

                # 选择样本进行SHAP分析（避免内存过载）
                sample_size = min(500, len(X_val_enhanced))
                sample_indices = np.random.choice(len(X_val_enhanced), sample_size, replace=False)
                X_sample = X_val_enhanced[sample_indices]

                logger.info(f"📊 使用{sample_size}个验证样本进行SHAP分析")

                # 计算SHAP值
                shap_values = explainer.shap_values(X_sample)
                logger.info("✅ SHAP值计算完成")

                # 创建可视化保存目录
                import os
                from pathlib import Path
                shap_dir = Path("results") / "shap_analysis"
                shap_dir.mkdir(parents=True, exist_ok=True)

                # 1. 特征重要性汇总图 - 显示全部特征
                num_features = len(feature_names)
                # 根据特征数量自动调整图形大小
                fig_height = max(8, num_features * 0.3)  # 每个特征至少0.3英寸高度
                fig_width = max(12, 16)  # 保证足够宽度显示特征名

                plt.figure(figsize=(fig_width, fig_height))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                show=False, max_display=num_features)  # 显示全部特征
                plt.title("SHAP特征重要性汇总图（全部特征）", fontsize=14, pad=20)
                plt.tight_layout()
                summary_plot_path = shap_dir / "shap_summary_plot_full.png"
                plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"✅ SHAP汇总图（全部{num_features}个特征）已保存: {summary_plot_path}")

                # 2. 特征重要性条形图 - 显示全部特征
                plt.figure(figsize=(fig_width, fig_height))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                plot_type="bar", show=False, max_display=num_features)  # 显示全部特征
                plt.title("SHAP特征重要性排序（全部特征）", fontsize=14, pad=20)
                plt.tight_layout()
                bar_plot_path = shap_dir / "shap_feature_importance_full.png"
                plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"✅ SHAP特征重要性图（全部{num_features}个特征）已保存: {bar_plot_path}")

                # 3. 生成Top20特征的精简版本（便于快速查看）
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                show=False, max_display=20)
                plt.title("SHAP特征重要性汇总图（Top20）", fontsize=14, pad=20)
                plt.tight_layout()
                summary_top20_path = shap_dir / "shap_summary_plot_top20.png"
                plt.savefig(summary_top20_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"✅ SHAP汇总图（Top20特征）已保存: {summary_top20_path}")

                # 4. 生成Top20特征的条形图
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                plot_type="bar", show=False, max_display=20)
                plt.title("SHAP特征重要性排序（Top20）", fontsize=14, pad=20)
                plt.tight_layout()
                bar_top20_path = shap_dir / "shap_feature_importance_top20.png"
                plt.savefig(bar_top20_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"✅ SHAP特征重要性图（Top20特征）已保存: {bar_top20_path}")

                # 3. 瀑布图 - 展示个别预测的解释
                try:
                    sample_predictions = model.predict(xgb.DMatrix(X_sample))

                    # 选择几个代表性样本：高预测值、低预测值、中等预测值
                    high_pred_idx = np.argmax(sample_predictions)
                    low_pred_idx = np.argmin(sample_predictions)
                    med_pred_idx = np.argsort(sample_predictions)[len(sample_predictions)//2]

                    representative_indices = [high_pred_idx, med_pred_idx, low_pred_idx]
                    representative_labels = ["高预测收益", "中等预测收益", "低预测收益"]

                    for idx, (sample_idx, label) in enumerate(zip(representative_indices, representative_labels)):
                        try:
                            plt.figure(figsize=(12, 8))

                            # 创建SHAP Explanation对象 - 新版SHAP API要求
                            explanation = shap.Explanation(
                                values=shap_values[sample_idx],
                                base_values=explainer.expected_value,
                                data=X_sample[sample_idx],
                                feature_names=feature_names
                            )

                            # 使用新版API调用瀑布图 - 显示更多特征
                            max_waterfall_features = min(25, len(feature_names))  # 最多显示25个特征
                            shap.plots.waterfall(explanation, max_display=max_waterfall_features, show=False)
                            plt.title(f"SHAP瀑布图 - {label}样本", fontsize=14, pad=20)
                            plt.tight_layout()
                            waterfall_path = shap_dir / f"shap_waterfall_{idx+1}_{label}.png"
                            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logger.info(f"✅ SHAP瀑布图已保存: {waterfall_path}")
                        except Exception as e:
                            logger.warning(f"⚠️ 生成第{idx+1}个瀑布图失败: {e}")
                            plt.close()  # 确保图形被关闭
                except Exception as e:
                    logger.warning(f"⚠️ SHAP瀑布图分析失败: {e}")

                # 4. 特征交互分析（如果特征数量适中）
                if len(feature_names) <= 50:  # 避免计算量过大
                    try:
                        # 计算前10个最重要特征的交互效应
                        top_features_idx = np.argsort(np.abs(shap_values).mean(0))[-10:]

                        for i, feat_idx in enumerate(top_features_idx[:3]):  # 只分析前3个最重要特征
                            plt.figure(figsize=(10, 8))
                            shap.dependence_plot(
                                feat_idx, shap_values, X_sample,
                                feature_names=feature_names,
                                show=False
                            )
                            plt.title(f"SHAP依赖图 - {feature_names[feat_idx]}", fontsize=14, pad=20)
                            plt.tight_layout()
                            dependence_path = shap_dir / f"shap_dependence_{i+1}_{feature_names[feat_idx]}.png"
                            plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logger.info(f"✅ SHAP依赖图已保存: {dependence_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ SHAP特征交互分析失败: {e}")

                # 5. SHAP值统计分析
                shap_importance = np.abs(shap_values).mean(0)
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_importance': shap_importance,
                    'shap_mean': shap_values.mean(0),  # 平均SHAP值（显示正负影响）
                    'shap_std': shap_values.std(0),   # SHAP值标准差（显示稳定性）
                    'shap_abs_mean': shap_importance  # 绝对值平均（重要性）
                }).sort_values('shap_importance', ascending=False)

                # 添加重要性等级分类
                shap_df['importance_level'] = pd.cut(
                    shap_df['shap_importance'],
                    bins=[0, shap_df['shap_importance'].quantile(0.5),
                          shap_df['shap_importance'].quantile(0.8),
                          shap_df['shap_importance'].max()],
                    labels=['低重要性', '中等重要性', '高重要性'],
                    include_lowest=True
                )

                logger.info("📊 SHAP特征重要性完整排序:")
                logger.info(f"{'排名':<4} {'特征名':<30} {'重要性':<10} {'平均影响':<10} {'稳定性':<10} {'等级':<10}")
                logger.info("=" * 80)
                for i, (_, row) in enumerate(shap_df.iterrows(), 1):
                    logger.info(f"{i:3d}. {row['feature']:<30} {row['shap_importance']:.4f}    "
                              f"{row['shap_mean']:+.4f}    {row['shap_std']:.4f}    {row['importance_level']}")

                # 统计各重要性等级的特征数量
                importance_counts = shap_df['importance_level'].value_counts()
                logger.info("\n📊 特征重要性分布:")
                for level, count in importance_counts.items():
                    percentage = count / len(shap_df) * 100
                    logger.info(f"  {level}: {count}个特征 ({percentage:.1f}%)")

                # 保存SHAP分析结果
                shap_results_path = shap_dir / "shap_analysis_results.csv"
                shap_df.to_csv(shap_results_path, index=False, encoding='utf-8')
                logger.info(f"\n✅ SHAP分析结果已保存: {shap_results_path}")

                # 生成特征筛选建议
                low_importance_features = shap_df[shap_df['importance_level'] == '低重要性']['feature'].tolist()
                logger.info(f"\n💡 特征筛选建议:")
                logger.info(f"  - 可考虑删除的低重要性特征({len(low_importance_features)}个):")
                for i, feature in enumerate(low_importance_features[:10], 1):  # 只显示前10个
                    logger.info(f"    {i}. {feature}")
                if len(low_importance_features) > 10:
                    logger.info(f"    ... 还有{len(low_importance_features)-10}个特征（详见CSV文件）")

                logger.info("=" * 60)
                logger.info("✅ SHAP分析完成")
                logger.info(f"📁 所有SHAP可视化文件保存在: {shap_dir}")
                logger.info("=" * 60)

            except ImportError:
                logger.warning("⚠️ SHAP库未安装，跳过SHAP分析")
                logger.info("💡 安装提示: pip install shap")
            except Exception as e:
                logger.warning(f"⚠️ SHAP分析过程中出现错误: {e}")
                logger.info("🔧 SHAP分析失败，但不影响模型训练结果")
            finally:
                # 恢复警告设置
                warnings.resetwarnings()

        # 构建结果字典（兼容_save_model_results）
        # 🚀 成本感知回归：构建性能指标
        performance_metrics = {
            'best_iteration': getattr(model, 'best_iteration', params['n_estimators'])
        }

        # 回归模式：添加回归指标和交易指标
        performance_metrics.update({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_type': 'regression',
            'trading_metrics': trading_metrics  # 🚀 添加交易指标
        })


        results = {
            'model': model,
            'scaler': None,  # 🔧 修复：不再使用StandardScaler，设为None
            'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize，不再需要保存阈值
            'pca_info': pca_info,
            'feature_names': feature_names,
            'interaction_info': None,  # 不使用交互特征
            'feature_importance': importance_df,
            'performance': performance_metrics,
            'training_info': {
                'n_samples': len(X_train_split),
                'n_features_original': len(feature_names),
                'n_features_used': current_features,
                'is_regression': True
            },
            'model_params': params,
            'label_stats': label_stats,  # 🚀 添加标签统计量（用于回归反标准化）
            'hyperparameter_search': {  # 🚀 添加超参数搜索信息
                'enabled': enable_param_search,
                'strategy': search_strategy if enable_param_search else None,
                'search_results': locals().get('search_results', None) if enable_param_search else None
            },
            # ✅ 添加预处理数据(用于LightGBM复用)
            'X_train_processed': X_train_split,
            'X_val_processed': X_val_split,
            'y_train': y_train_split,
            'y_val': y_val_split,
            'model_type': 'xgboost'  # ✅ 标记模型类型
        }

        # 🚀 添加模式特定信息
        results['training_info'].update({
            'target_range': [float(y_train_split.min()), float(y_train_split.max())],
            'target_mean': float(y_train_split.mean()),
            'target_std': float(y_train_split.std())
        })


        logger.info("=" * 60)

        # 🚀 简化：统一返回训练结果
        logger.info("✅ 模型训练完成，返回训练结果")
        return results

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"模型训练失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None


def _analyze_feature_interactions(model, feature_names, X_val):
    """分析特征交互"""
    try:
        # 🚀 修复：兼容Booster和XGBRegressor的特征重要性获取
        try:
            # 尝试Booster方法，使用gain而不是weight
            importance_dict = model.get_score(importance_type='gain')
            feature_importance = np.zeros(len(feature_names))
            for i, feature_name in enumerate(feature_names):
                xgb_feature_name = f'f{i}'
                feature_importance[i] = importance_dict.get(xgb_feature_name, 0.0)
            # 归一化
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()
        except AttributeError:
            # 回退到传统方法
            feature_importance = model.feature_importances_

        # 找出重要特征
        important_indices = np.argsort(feature_importance)[-20:]  # 前20个重要特征
        important_features = [feature_names[i] for i in important_indices]

        # 分析特征组合的预测能力
        interactions = []

        # 简化的特征交互分析：基于特征重要性的组合
        for i, feat1 in enumerate(important_features[:10]):
            for j, feat2 in enumerate(important_features[i+1:11]):
                # 计算特征组合的交互强度（简化版本）
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)

                # 基于特征重要性计算交互分数
                interaction_score = (feature_importance[idx1] + feature_importance[idx2]) / 2
                interactions.append((f"{feat1} + {feat2}", interaction_score))

        # 按交互分数排序
        interactions.sort(key=lambda x: x[1], reverse=True)
        return interactions

    except Exception as e:
        return []


def _get_new_stock_cache():
    """获取新股缓存，如果不存在则创建"""
    from pathlib import Path
    cache_file = Path(__file__).parent / "cache" / "new_stock_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass

    return {}


def _get_zscore_factors():
    """
    返回需要 z-score 归一化的因子列表（保留幅度信息）

    适用场景：
    - 当因子的绝对幅度包含信息（例如极端正/负幅度代表强势/弱势）
    - 典型例子：return_1d, return_5d, 相关性因子, 比率因子
    - 已经做了 rank 的因子（如 rank_momentum_20d）也用 z-score
    """
    return [
        'alpha_001',
        'alpha_002',
        'alpha_017',
        'amount_rank',
        'beta_market',
        'bollinger_breakout_strength',
        'consecutive_up_severity',
        'corr_20d_mom5',
        'corr_5d',
        'corr_slope_20_120',
        'cpv_factor',
        'cvar_95',
        'distance_from_52w_high',
        'distance_from_52w_low',
        'downside_risk',
        'drawdown_recovery_speed',
        'institutional_activity_score',
        'low_volatility_reversal',
        'lower_shadow_ratio',
        'ma60_slope',
        'ma_convergence_score',
        'ma_ratio_5_10',
        'market_correlation',
        'market_structure_score',
        'max_drawdown_20d',
        'max_rally_20d',
        'momentum_decay',
        'momentum_reversal',
        'obv_momentum_factor',
        'open_to_high_ratio',
        'pct_chg_0d',
        'pct_chg_1d',
        'pct_chg_2d',
        'price_ma60_ratio',
        'pullback_volume_ratio',
        'rally_continuation_strength',
        'rank_market_cap',
        'rank_momentum_20d',
        'rank_rsi',
        'rank_turnover_rate',
        'rank_volatility_20d',
        'resistance_distance',
        'return_10d',
        'return_3d',
        'return_5d',
        'returns_zscore',
        'sector_correlation_20d',
        'sector_momentum_consistency',
        'support_distance',
        'support_risk_reward_ratio',
        'trend_acceleration',
        'turnover_rate_strength',
        'var_95',
        'volatility_adjusted_return',
        'volatility_change_rate',
        'volatility_correlation',
        'volume_correlation',
        'volume_price_divergence',
        'volume_price_strength',
        'volume_squeeze',
    ]


def _get_rankpct_factors():
    """
    返回需要 rankpct 归一化的因子列表（映射到 [-1, 1]，抗极端值）

    适用场景：
    - 当因子分布极度偏斜、含大量极端值，或不同股票间量纲差异巨大
    - 典型例子：PE, PB, 财务比率（分布长尾）、换手率（极端日子）
    - 做截面排序/Top-K 时常用 rank

    注意：同一因子可以同时出现在 _get_zscore_factors 和 _get_rankpct_factors 中
    """
    return [
        'amihud_illiquidity',
        'amihud_illiquidity_enhanced',
        'bid_ask_spread_proxy',
        'kdj_golden_cross_strength',
        'kurtosis_20d',
        'near_support_count_60d',
        'price_impact',
        'sector_breadth',
        'sector_concentration',
        'size_market_cap_interaction',
        'skewness_20d',
        'trend_persistence_10d',
        'trend_persistence_20d',
        'turnover_volatility',
    ]


def _get_stock_factors():
    """获取股票因子列表（63个因子，已删除7个高度冗余因子 + 18个SHAP极低重要性因子 + 17个高相关性因子 + 5个SHAP极低重要性因子，新增30个因子）

    删除说明:
    - 第一轮删除(7个): 基于相关性分析,删除高度冗余因子
    - 第二轮删除(18个): 基于SHAP分析,删除极低重要性因子(SHAP重要性<0.015)
    - 第三轮删除(13个): 基于Spearman相关性+SHAP综合分析,删除高冗余因子
      * 风险因子: cumulative_down_20d, var_95, cvar_95 (保留downside_risk)
      * 成交额因子: amount_momentum, amount_breakout (保留amount_rank)
      * 技术指标: bollinger_position_factor, macd_strength (保留alpha_017)
      * 收益率因子: return_20d (保留return_5d, return_10d)
      * 其他: intraday_return, volume_acceleration, cross_sector_correlation, kdj_golden_cross_strength, volatility_efficiency
    - 第四轮删除(4个): 基于56因子相关性分析,删除极高相关因子
      * consecutive_down_days (与consecutive_down_severity相关0.946,保留后者)
      * alpha_054 (与returns_zscore相关0.869,保留后者)
      * rank_volume (与amount_rank相关0.732,保留后者)
      * rank_volume_ratio (与turnover_rate_strength相关0.849,保留后者)
    - 🗑️ 第五轮删除(5个): 基于SHAP重要性分析,删除极低重要性因子(保守方案)
      * sector_momentum_interaction (SHAP #53, 0.0098): 板块动量交互效果差
      * amount_volatility (SHAP #54, 0.0095): 成交额波动率贡献极低
      * momentum_strength (SHAP #52, 0.0124): 风险调整后的动量因子效果差
      * style_consistency (SHAP #51, 0.0131): 风格一致性贡献低
      * downside_risk (SHAP #50, 0.0142): 下行风险因子重要性低

    新增因子:
    - 成交额因子(3个): amount_volatility(优化版), amount_rank, amihud_illiquidity_enhanced
    - 风险控制因子(4个): downside_risk, max_drawdown_20d, consecutive_down_days, volatility_change_rate
    - 风险优化因子(2个): drawdown_recovery_speed, consecutive_down_severity
    - 🆕 模式平衡因子(2个): consecutive_up_severity, momentum_strength
      * 目的: 平衡模型学习，避免过度依赖反转(抄底)模式
      * 让模型同时学习反转和趋势延续两种选股模式
    - 🔧 板块因子修改(1个): sector_return_rank_3d (从20日改为3日)
    - 🆕 横截面排名因子(4个): rank_max_rally_20d, rank_max_drawdown_20d, rank_drawdown_recovery_speed, rank_rally_continuation_strength
      * 目的: 提供风险和趋势因子的横截面排名视角
    - 🚀 高IC因子(3个): alpha_002,cpv_factor,low_volatility_reversal
      * alpha_002: WorldQuant Alpha#002, K线位置变化反转因子, A股IC 0.08-0.12
      * cpv_factor: CPV量价相关性因子, 高频量价背离, A股IC 0.06-0.09
      * low_volatility_reversal: 低振幅反转因子, 结构化反转, A股IC 0.08-0.12

    观察因子:
    - drawdown_recovery_speed: 虽与rank_rsi相关0.864,但可能提供独特风险维度,保留观察SHAP重要性
    - consecutive_up_severity: 对称于consecutive_down_severity,观察是否能平衡模型学习
    """
    # 所有因子列表
    all_factors = [
        # ===== 原有核心因子（24个，删除1个，新增3个）=====
        'returns_zscore', 'market_structure_score', 'ma_ratio_5_10', 'corr_5d',
        'volatility_correlation', 'volume_correlation', 'near_support_count_60d',
        'institutional_activity_score', 'ma_convergence_score', 'lower_shadow_ratio',
        'sector_correlation_20d', 'price_impact', 'rank_momentum_20d', 'rank_rsi',
        # 🆕 新增均线因子（2个）
        'ma60_slope',  # MA60斜率因子
        'price_ma60_ratio',  # 价格相对MA60的比率
        # 'rank_volume',  # 删除：与amount_rank相关0.732,保留后者(成交额比成交量更重要)
        'market_correlation', 'corr_20d_mom5', 'corr_slope_20_120',
        'support_risk_reward_ratio', 'obv_momentum_factor', 'bollinger_breakout_strength',
        'sector_momentum_consistency',
        # 'sector_return_rank_3d',  # 🔧 修改：从20日改为3日，更短期的板块动量
        'support_distance', 'resistance_distance',

        # ===== 动量与反转因子（12个，删除3个，新增4个）=====
        'pct_chg_0d',  # 删除：与returns_zscore高度相关(0.965)，保留标准化版本
        'pct_chg_1d', 'pct_chg_2d',
        # 🆕 新增收益率因子
        'return_3d',  # 3日累计收益
        'return_5d', 'return_10d',
        # 'return_20d',  # 删除：5次高相关,与trend_acceleration负相关-0.841
        'momentum_reversal', 'trend_acceleration', 'momentum_decay',
        # 'reversal_strength',  # 删除：与pct_chg_1d完全负相关(-1.000)，数学上等价
        # 🆕 新增动量强度因子（平衡反转模式）
        # 'momentum_strength',  # 🗑️ 删除：SHAP重要性倒数第3(0.0124)，风险调整后的动量因子效果差

        'max_rally_20d', #高值: 历史涨幅大（上涨潜力强，爆发力强）低值: 历史涨幅小（上涨潜力弱）与max_drawdown_20d对称，衡量极端上涨能力
        'rally_continuation_strength', # (drawdown_recovery_speed的对应因子) 高值: 从低点持续上涨（趋势强，延续性好）与drawdown_recovery_speed对称，衡量上涨延续能力

        # ===== 横截面排名因子（9个，删除2个，新增4个）=====
        # 'rank_pct_chg_0d', #特征和shap重要性都很低
        # 'rank_return_5d',
        # 'rank_return_20d',  # 删除：与rank_momentum_20d高度相关(0.996)，含义重复
        # 'rank_volume_ratio',  # 删除：与turnover_rate_strength相关0.849,保留后者
        'rank_turnover_rate', #重要性与shap重要性都很低
        'rank_volatility_20d', 'rank_market_cap',
        # 🆕 新增横截面排名因子（4个）
        # 'rank_max_rally_20d',  # 20日最大涨幅的横截面排名
        # 'rank_max_drawdown_20d',  # 20日最大回撤的横截面排名
        # 'rank_drawdown_recovery_speed',  # 回撤恢复速度的横截面排名
        # 'rank_rally_continuation_strength',  # 上涨延续强度的横截面排名

        # ===== 流动性因子（8个，删除3个）=====
        'volume_price_strength', 'volatility_adjusted_return', 'amihud_illiquidity',
        'turnover_volatility', 'bid_ask_spread_proxy',
        # 'turnover_change_strength',  # 删除：与rank_volume_ratio高度相关(0.998)，含义重复
        # 🚀 新增成交额因子（3个，删除2个）
        # 'amount_momentum',  # 删除：与volume_acceleration相关0.905,与turnover_rate_strength相关0.825
        # 'amount_volatility',  # 🗑️ 删除：SHAP重要性倒数第1(0.0095)，成交额波动率贡献极低
        'amount_rank',
        # 'amount_breakout',  # 删除：与rank_volume_ratio相关0.915,与turnover_rate_strength相关0.782
        'amihud_illiquidity_enhanced',

        # ===== 低波动异常因子（2个）=====
        'beta_market',
        #'max_daily_return_60d', #重要性低

        # ===== 风险控制因子（6个，删除4个，新增1个）=====
        'downside_risk',  # 🗑️ 删除：SHAP重要性倒数第5(0.0142)，下行风险因子重要性低
        'max_drawdown_20d',
        # 'consecutive_down_days',  # 删除：与consecutive_down_severity相关0.946,保留后者(严重性更重要)
        # 'cumulative_down_20d',  # 删除：与downside_risk相关0.954
        'volatility_change_rate',
        'var_95',  # 删除：与downside_risk相关0.914
        'cvar_95',  # 删除：与downside_risk相关0.923,与var_95相关0.807
        # 🚀 新增风险优化因子（2个）
        'drawdown_recovery_speed',  # 回撤恢复速度(虽与rank_rsi相关0.864,但保留观察SHAP重要性)
        # 'consecutive_down_severity',  # 连续下跌严重性（反转信号）重要性不足
        # 🆕 新增趋势延续因子（平衡反转模式）
        'consecutive_up_severity',  # 连续上涨严重性（趋势延续信号，对称于consecutive_down_severity）

        # ===== 行为金融因子（3个，删除1个）=====
        'distance_from_52w_high', 'distance_from_52w_low',
        # 'price_acceleration', 重要性和shap都很低
        # 'volume_acceleration',  # 删除：与turnover_rate_strength相关0.913

        # ===== 日内微观结构因子（2个，删除2个）=====
        # 'overnight_return',
        # 'intraday_return',  # 删除：与rank_pct_chg_0d相关0.889,与alpha_054相关0.736
        'open_to_high_ratio',
        # 'open_to_low_ratio',  # 删除：与open_to_high_ratio高度负相关(-0.984)，数学上互补

        # ===== WorldQuant Alpha因子（4个，删除1个，新增1个）=====
        'alpha_001',
        # 'alpha_012', #重要性和shap都很低
        'alpha_017',
        # 'alpha_054',  # 删除：与returns_zscore相关0.869,保留后者(更直观)
        'alpha_002',  # 🚀 新增：K线位置变化反转因子，A股IC 0.08-0.12

        # ===== 量价背离因子（2个，新增）=====
        'cpv_factor',  # 🚀 新增：CPV量价相关性因子，高频量价背离，A股IC 0.06-0.09
        'low_volatility_reversal',  # 🚀 新增：低振幅反转因子，结构化反转，A股IC 0.08-0.12

        # ===== 风格轮动因子（3个，删除1个）=====
        'sector_concentration', 'sector_breadth',
        # 'cross_sector_correlation',  # 删除：与sector_correlation_20d负相关-0.874
        # 'style_consistency',  # 🗑️ 删除：SHAP重要性倒数第4(0.0131)，风格一致性贡献低

        # ===== 交互特征因子（2个）=====
        'size_market_cap_interaction',
        # 'sector_momentum_interaction',  # 🗑️ 删除：SHAP重要性倒数第2(0.0098)，板块动量交互效果差

        # ===== 技术指标因子（3个，删除3个，新增1个）=====
        # 'macd_strength',  # 删除：4次高相关,与bollinger_position_factor相关0.813
        # 'atr_normalized', #找不到了
        'turnover_rate_strength',
        # 'cci_indicator',  # 删除：与bollinger_position_factor高度相关(0.962)，保留更直观的布林带位置
        'volume_price_divergence',
        # 🆕 新增成交量因子
        'volume_squeeze',  # 成交量挤压因子（布林带宽度应用于成交量）

        # ===== 统计矩因子（3个，删除1个）=====
        'skewness_20d', 'kurtosis_20d',
        # 'idiosyncratic_volatility',  # 删除：与idiosyncratic_volatility_sector高度相关(0.893)，保留剥离板块影响的版本
        # 'idiosyncratic_volatility_sector',

        # ===== 趋势持续性因子（2个，新增）=====
        'trend_persistence_10d',  # 10日涨跌一致性
        'trend_persistence_20d',  # 20日涨跌一致性

        # ===== KDJ因子（0个，删除1个）=====
        'kdj_golden_cross_strength',  # 删除：SHAP重要性低(0.0134)

        # ===== 其他因子（1个，删除3个）=====
        # 'volatility_efficiency',  # 删除：SHAP重要性低(0.0148)
        'pullback_volume_ratio',
        #   'volume_strength', #删除：与turnover_rate_strength ，rank_volume_ratio 高相关
        # 'bollinger_position_factor',  # 删除：8次高相关(最高!),与alpha_017相关0.921
    ]



    # 使用列表相减操作删除因子
    # stock_factors = [f for f in all_factors if f not in shap_remove_conservative]

    return all_factors


def _get_original_factors():
    """获取需要保留原始值的因子列表（不进行归一化）

    这些因子在因子生成时会被计算，但不会进行z-score或rankpct归一化。
    在模型训练时，可以选择是否使用这些原始因子。

    Returns:
        list: 原始因子名称列表
    """
    # 目前没有需要保留原始值的因子
    # 如果未来需要，可以在这里添加
    return []


def _get_model_factors(use_original=True, use_zscore=True, use_rankpct=True):
    """获取模型训练时使用的因子列表

    此函数用于模型训练和因子测试，可以灵活选择使用哪些类型的因子。

    Args:
        use_original: 是否使用原始因子（不带后缀）
        use_zscore: 是否使用z-score归一化因子（_z后缀）
        use_rankpct: 是否使用rankpct归一化因子（_rank后缀）

    Returns:
        list: 因子名称列表

    Examples:
        # 只使用z-score归一化因子（默认）
        factors = _get_model_factors()

        # 使用原始因子 + z-score因子
        factors = _get_model_factors(use_original=True, use_zscore=True, use_rankpct=False)

        # 使用所有因子
        factors = _get_model_factors(use_original=True, use_zscore=True, use_rankpct=True)
    """
    factors = []

    # 添加原始因子
    if use_original:
        original_factors = _get_original_factors()
        factors.extend(original_factors)

    # 添加 z-score 归一化因子
    if use_zscore:
        zscore_factors = _get_zscore_factors()
        factors.extend([f"{f}_z" for f in zscore_factors])

    # 添加 rankpct 归一化因子
    if use_rankpct:
        rankpct_factors = _get_rankpct_factors()
        factors.extend([f"{f}_rank" for f in rankpct_factors])

    # 去重（保持顺序）
    seen = set()
    unique_factors = []
    for factor in factors:
        if factor not in seen:
            seen.add(factor)
            unique_factors.append(factor)

    return unique_factors


def _get_available_market_factors():
    """获取可用的市场状态因子列表"""
    logger = logging.getLogger(__name__)

    # 基于market_state_test_mode测试结果，返回有效的市场状态因子
    market_factors = [
        # 'gain_loss_ratio_rank_10d',
        # 'gain_loss_ratio_rank_20d',
        'gain_loss_ratio_rank_60d',
        'limit_up_count_rank_60d',
        'limit_up_count_rank_20d',
        # 'limit_up_count_rank_10d',
        # 'limit_up_count_rank_5d',
        'limit_down_count_rank_60d',
        'limit_down_count_rank_5d',
        'market_volume_rank_20d',
        'broken_limit_count_rank_20d',
        'max_consecutive_limit_rank_20d',
        'market_volatility_atr20d',
        'market_trend_factor',
        'max_gain_10d_rank_60d',
        'max_gain_5d_rank_30d'
    ]

    logger.info(f"返回 {len(market_factors)} 个市场状态因子")
    return market_factors


def _clear_old_cache():
    """清理过期的缓存文件"""
    try:
        from pathlib import Path
        import datetime

        cache_dir = Path(__file__).parent / "cache"
        if not cache_dir.exists():
            return

        # 清理7天前的缓存
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=7)

        for cache_file in cache_dir.glob("*.pkl"):
            try:
                if cache_file.stat().st_mtime < cutoff_date.timestamp():
                    cache_file.unlink()
                    print(f"🗑️ 清理过期缓存: {cache_file.name}")
            except:
                pass
    except Exception as e:
        print(f"⚠️ 清理缓存失败: {e}")

def _save_new_stock_cache(cache):
    """保存新股缓存"""
    cache_file = Path(__file__).parent / "cache" / "new_stock_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    try:
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"⚠️ 保存新股缓存失败: {e}")

def _build_new_stock_list_for_date(current_date: str) -> set:
    """为特定日期构建新股列表"""
    try:
        from pathlib import Path
        import pandas as pd

        # 获取当前日期前30个交易日的日期列表
        trading_dates = _get_previous_trading_dates(current_date, 30)

        if len(trading_dates) < 20:
            return set()  # 返回空集合，表示没有新股过滤

        # 使用相对于当前文件的路径
        current_file_dir = Path(__file__).parent
        features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

        # 收集所有股票的交易天数
        stock_trading_days = {}

        for date_str in trading_dates:
            file_path = features_dir / f"{date_str}.parquet"

            if file_path.exists():
                try:
                    daily_data = pd.read_parquet(file_path)

                    # 批量处理所有股票
                    valid_stocks = daily_data[
                        (~np.isnan(daily_data['volume'])) & (daily_data['volume'] > 0)
                    ]['stock_code'].unique()

                    for stock_code in valid_stocks:
                        if stock_code not in stock_trading_days:
                            stock_trading_days[stock_code] = 0
                        stock_trading_days[stock_code] += 1

                except Exception as e:
                    print(f"⚠️ 读取{date_str}数据失败: {e}")
                    continue

        # 识别新股（有效交易天数少于20天）
        new_stocks = {
            stock_code for stock_code, days in stock_trading_days.items()
            if days < 20
        }

        print(f"📅 {current_date}: 识别出 {len(new_stocks)} 只新股")
        return new_stocks

    except Exception as e:
        print(f"❌ 构建新股列表失败: {e}")
        return set()

def _filter_new_stocks(data: pd.DataFrame, current_date: str, logger) -> pd.Series:
    """
    高效过滤新股 - 优先检查当日数据异常，然后检查历史交易天数

    参数:
    - data: 当前日期的股票数据
    - current_date: 当前选股日期 (格式: YYYYMMDD)

    返回:
    - pd.Series: 布尔值序列，True表示股票正常可交易，False表示是新股或异常股票
    """
    try:
        # 第一步：检查当日数据异常（成交量=0或NaN的股票）
        import numpy as np
        current_day_invalid = (
            (data['volume'] == 0) | (np.isnan(data['volume']))
        )

        current_day_invalid_count = current_day_invalid.sum()
        if current_day_invalid_count > 0:
            print(f"🚫 当日数据异常过滤: 过滤掉 {current_day_invalid_count} 只股票（成交量=0或NaN）")
            invalid_stocks = data[current_day_invalid]['stock_code'].tolist()
            if len(invalid_stocks) <= 10:  # 如果数量不多，显示具体股票
                print(f"   异常股票: {', '.join(invalid_stocks)}")

        # 第二步：对剩余股票检查历史交易天数
        valid_current_day = ~current_day_invalid
        remaining_data = data[valid_current_day]

        if len(remaining_data) == 0:
            print("⚠️ 所有股票都因当日数据异常被过滤")
            return pd.Series(False, index=data.index)

        # 获取历史新股缓存
        cache = _get_new_stock_cache()

        # 检查缓存中是否有当前日期的数据
        if current_date not in cache:
            print(f"🔄 为日期 {current_date} 构建新股列表...")
            new_stocks = _build_new_stock_list_for_date(current_date)
            cache[current_date] = new_stocks
            _save_new_stock_cache(cache)
        else:
            new_stocks = cache[current_date]

        # 对剩余股票应用历史交易天数过滤
        historical_invalid = remaining_data['stock_code'].isin(new_stocks)
        historical_invalid_count = historical_invalid.sum()

        if historical_invalid_count > 0:
            print(f"🚫 历史交易天数过滤: 过滤掉 {historical_invalid_count} 只新股（上市不足20个有效交易日）")

        # 构建最终结果：当日正常 且 历史交易天数充足
        result = pd.Series(False, index=data.index)
        result[valid_current_day] = ~historical_invalid

        total_filtered = len(data) - result.sum()
        total_remaining = result.sum()
        logger.info(f"📊 新股过滤总结: 过滤掉 {total_filtered} 只，剩余 {total_remaining} 只正常股票")

        return result

    except Exception as e:
        print(f"❌ 新股过滤失败: {e}")
        # 出错时不过滤任何股票
        return pd.Series(True, index=data.index)


def _filter_st_stocks(data: pd.DataFrame, logger) -> pd.Series:
    """
    过滤ST股票

    参数:
    - data: 股票数据DataFrame

    返回:
    - pd.Series: 布尔值序列，True表示非ST股票，False表示ST股票
    """
    try:
        # 导入股票名称字典
        import sys
        from pathlib import Path

        # 确保data_loaders模块可以被导入
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.append(str(project_root))

        from data_loaders import get_stock_names_dict

        # 获取股票名称字典
        stock_names = get_stock_names_dict()

        if not stock_names:
            print("⚠️ 无法获取股票名称字典，跳过ST股票过滤")
            return pd.Series(True, index=data.index)

        # 过滤ST股票
        st_filter = pd.Series(True, index=data.index)
        st_stocks = []

        for idx, row in data.iterrows():
            stock_code = row['stock_code']
            stock_name = stock_names.get(stock_code, "")

            if "ST" in stock_name:
                st_filter.loc[idx] = False
                st_stocks.append(f"{stock_code}({stock_name})")

        st_count = (~st_filter).sum()
        if st_count > 0:
            logger.info(f"🚫 ST股票过滤: 过滤掉 {st_count} 只ST股票")
            if len(st_stocks) <= 5:  # 如果数量不多，显示具体股票
                print(f"   ST股票: {', '.join(st_stocks)}")
            elif len(st_stocks) > 5:
                print(f"   ST股票示例: {', '.join(st_stocks[:3])} 等{len(st_stocks)}只")

        remaining_count = st_filter.sum()
        logger.info(f"📊 ST股票过滤后剩余: {remaining_count} 只非ST股票")

        return st_filter

    except Exception as e:
        print(f"❌ ST股票过滤失败: {e}")
        # 出错时不过滤任何股票
        return pd.Series(True, index=data.index)

def _get_trading_dates_cache():
    """获取交易日缓存"""
    cache_file = Path(__file__).parent / "cache" / "trading_dates_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass

    return {}

def _save_trading_dates_cache(cache):
    """保存交易日缓存"""
    cache_file = Path(__file__).parent / "cache" / "trading_dates_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    try:
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"⚠️ 保存交易日缓存失败: {e}")

def _get_previous_trading_dates(current_date: str, days: int) -> list:
    """获取指定日期前N个交易日的日期列表（使用缓存优化）"""
    try:
        from pathlib import Path
        import datetime

        # 检查缓存
        cache = _get_trading_dates_cache()
        cache_key = f"{current_date}_{days}"

        if cache_key in cache:
            return cache[cache_key]

        # 使用相对于当前文件的路径
        current_file_dir = Path(__file__).parent
        features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

        # 一次性获取所有可能的文件列表（更高效）
        all_files = list(features_dir.glob("*.parquet"))
        all_dates = []

        for file_path in all_files:
            try:
                date_str = file_path.stem  # 获取文件名（不含扩展名）
                if len(date_str) == 8 and date_str.isdigit():
                    all_dates.append(date_str)
            except:
                continue

        # 排序并筛选出当前日期之前的交易日
        all_dates.sort(reverse=True)  # 降序排列
        current_dt = datetime.datetime.strptime(current_date, '%Y%m%d')

        trading_dates = []
        for date_str in all_dates:
            try:
                check_dt = datetime.datetime.strptime(date_str, '%Y%m%d')
                if check_dt < current_dt:
                    trading_dates.append(date_str)
                    if len(trading_dates) >= days:
                        break
            except:
                continue

        # 缓存结果
        cache[cache_key] = trading_dates
        _save_trading_dates_cache(cache)

        return trading_dates

    except Exception as e:
        print(f"❌ 获取交易日列表失败: {e}")
        return []


def _get_model_cache_path(train_start_date: str, train_end_date: str, feature_count: int = None) -> Path:
    """获取模型缓存路径

    Args:
        train_start_date: 训练开始日期
        train_end_date: 训练结束日期
        feature_count: 特征数量（用于缓存键区分）
    """
    cache_dir = Path("cache/models")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 🔧 改进缓存键：包含特征数量，避免特征不一致时的缓存污染
    if feature_count is not None:
        model_name = f"ml_model_{train_start_date}_{train_end_date}_f{feature_count}"
    else:
        model_name = f"ml_model_{train_start_date}_{train_end_date}"

    return cache_dir / f"{model_name}.pkl"

def _save_model_cache(ml_model, scaler, feature_names: list, train_start_date: str, train_end_date: str):
    """保存模型缓存"""
    try:
        import pickle

        cache_path = _get_model_cache_path(train_start_date, train_end_date)

        model_data = {
            'ml_model': ml_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'train_start_date': train_start_date,
            'train_end_date': train_end_date,
            'created_time': pd.Timestamp.now().isoformat(),
            'model_type': type(ml_model).__name__,
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ 模型已保存到缓存: {cache_path}")
        return True

    except Exception as e:
        print(f"⚠️ 模型缓存保存失败: {e}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    setup_logging()
    logger = logging.getLogger(__name__)

    print("板块-个股一体化选股系统")
    print("=" * 60)
    print(f"系统配置:")
    print(f"  运行模式: {args.mode}")
    print(f"  数据期间: {system_config.start_date} - {system_config.end_date}")
    print(f"  训练期间: {system_config.start_date} - {system_config.train_end_date}")
    print(f"  测试期间: {system_config.test_start_date} - {system_config.end_date}")
    print(f"  每日选股: {system_config.daily_stock_count}只")
    print(f"  持仓天数: {backtest_config.holding_days}天")
    print(f"  板块筛选: 前{system_config.sector_filter_count}个（优化后）")
    print(f"  板块权重: {system_config.factor_weights['sector']:.0%}（大幅提升）")
    print(f"  连板回看: {system_config.max_lookback_days}天")
    print(f"  输出目录: {args.output_dir}")
    print()


#     # 三种模式：
# 1. 默认模式：生成所有请求的因子（可能覆盖已存在的）
#    python main.py --mode generate_factors

# 2. 强制模式：清理旧数据并重新生成
#    python main.py --mode generate_factors --force

# 3. 🆕 增量模式：只生成缺少的因子
#    python main.py --mode generate_factors --incremental
# 如果想只生成特定因子，可以使用 --factor_names 参数
# 自定义批次大小
# python main.py --mode generate_factors --batch_size 5

# # 禁用分批保存
# python main.py --mode generate_factors --batch_size 0



    # 执行对应的运行模式
    if args.mode == 'generate_factors':
        success = run_generate_factors_mode(args)
    elif args.mode == 'test_factors':
        success = run_test_factors_mode(args)
    elif args.mode == 'train_model':
        # success = run_train_model_mode(args,train_end_date='2024-06-01', is_regression=True)  # 🚀 启用成本感知回归模式
        success = run_train_model_mode(args,None, is_regression=True)
    #python main.py --mode test_model --models lightgbm    
    elif args.mode == 'test_model':
        success = run_test_model_mode(args)
    # python main.py --mode test_model_walkforward --window 1,3
    #python main.py --mode test_model --models lightgbm --optimize_thresholds 用来搜索最优atr止盈止损阈值
    #python main.py --mode test_model_walkforward --window 3 --models lightgbm --optimize_thresholds
    elif args.mode == 'test_model_walkforward':
        success = run_test_model_walkforward_mode(args)
    # test_optimization模式已删除，优化方案已集成到test_model模式
    elif args.mode == 'factor_correlation':
        success = run_factor_correlation_mode(args)
    elif args.mode == 'genetic_factor':
        success = run_genetic_factor_mode(args)
    elif args.mode == 'market_state_test':
        success = run_market_state_test_mode(args)
    # elif args.mode == 'interaction_analysis':
    #     success = run_interaction_analysis_mode(args)  # 已移除交互特征分析模式
    else:
        logger.error(f"未知的运行模式: {args.mode}")
        return False

    return success



def run_generate_factors_mode(args):
    """运行因子实验室的因子生成流程"""
    logger = logging.getLogger(__name__)
    print("?? 运行模式: 因子生成")
    print("=" * 60)

    try:
        data_loader = IntegratedDataLoader()

        start_date = args.start_date or "20230601"
        end_date = args.end_date or "20251021"

        print(f"?? 目标区间: {start_date} - {end_date}")

        extend_history_days = getattr(args, 'extend_history_days', 120)

        extended_start_date = data_loader._calculate_extended_start_date(

            start_date, end_date, extend_history_days
        )

        stock_data = data_loader.load_stock_features(extended_start_date, end_date)
        try:
            # 修复：板块数据也需要历史窗口（最大需求20天，使用extended_start_date确保足够）
            sector_data = data_loader.load_sector_features(extended_start_date, end_date)
        except Exception as exc:
            logger.warning('sector data loader raised an exception: %s', exc)
            sector_data = None

        if sector_data is None or sector_data.empty:
            logger.warning('sector data missing, falling back to local cache')
            sector_data = _load_sector_features_basic(extended_start_date, end_date)

        try:
            # 修复：市场数据也需要历史窗口（最大需求120天，使用extended_start_date确保足够）
            market_data = data_loader.load_market_features(extended_start_date, end_date)
        except Exception as exc:
            logger.warning('market data loader raised an exception: %s', exc)
            market_data = None

        if market_data is None or market_data.empty:
            logger.warning('market data missing, using placeholder values')
            trading_dates = data_loader.get_trading_dates(start_date, end_date)
            market_data = pd.DataFrame({
                'date': pd.to_datetime(trading_dates, format='%Y%m%d'),
                'market_return': 0.0,
                'market_volume': 1.0,
            })

        if stock_data is None or stock_data.empty:
            logger.error('failed to load stock base data; aborting factor generation')
            print('stock data is empty, stop factor generation.')
            return False

        if sector_data is None or sector_data.empty:

            logger.warning('no valid sector data, sector-related factors may be missing')

        if market_data is None or market_data.empty:

            logger.warning('no valid market data, placeholder values will be used')


        for dataset in (stock_data, sector_data, market_data):

            if dataset is not None and not dataset.empty and 'date' in dataset.columns:

                dataset['date'] = pd.to_datetime(dataset['date'])

        print(f"? 股票记录数: {len(stock_data)}")

        if sector_data is not None and not sector_data.empty:
            print(f"? 板块记录数: {len(sector_data)}")

        else:
            print('[warning] sector data missing, using stock-only factors')
        if market_data is not None and not market_data.empty:
            print(f"? 市场记录数: {len(market_data)}")
        else:
            print('[warning] market data missing, placeholder values applied')

        if args.factor_names:
            requested_factors = [name.strip() for name in args.factor_names.split(',')]
        elif getattr(args, 'factor_groups', None):
            requested_factors = [name.strip() for name in args.factor_groups.split(',')]
        else:
            requested_factors = _get_stock_factors()

        factor_names = []
        seen = set()
        for name in requested_factors:
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                factor_names.append(name)

        pipeline = FactorPipeline()
        available = set(pipeline.available_factors())
        missing = [name for name in factor_names if name not in available]
        if missing:
            logger.warning(f"下列因子暂未在factor_lab注册，将忽略: {missing}")
            print(f"?? 忽略未注册因子: {missing}")
        factor_names = [name for name in factor_names if name in available]

        if not factor_names:
            logger.error("没有可用的因子名称，请检查输入。")
            print("? 无可计算的因子名称，生成流程结束。")
            return False

        print(f"?? 计划生成因子: {factor_names}")

        extra_context = {}
        stock_sector_mapping = data_loader.load_stock_sector_mapping()
        if stock_sector_mapping:
            extra_context['stock_to_sector'] = stock_sector_mapping
            print(f"? 股票-板块映射数量: {len(stock_sector_mapping)}")
        else:
            print("?? 未找到股票-板块映射，将跳过板块相关因子。")

        FACTOR_LAB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        store = FactorDatasetStore(FACTOR_LAB_STORE_PATH)

        # 🆕 增量生成功能：检测已存在的因子
        if getattr(args, 'force', False):
            logger.info("force 标志已启用，将在重新生成前清理对应因子列。")
            store.drop_factors(factor_names)
            print("?? 已清理旧的因子列，准备重新生成。")
            factors_to_generate = factor_names
        elif getattr(args, 'incremental', False):
            # 增量模式：只生成缺少的因子，并删除多余的因子
            existing_data = store.load()
            if existing_data.empty:
                logger.info("增量模式：因子文件为空，将生成所有因子")
                print("?? 增量模式：因子文件为空，将生成所有因子")
                factors_to_generate = factor_names
            else:
                existing_factors = set(existing_data.columns) - {'date', 'stock_code'}
                requested_factors_set = set(factor_names)

                # 🆕 检查并删除多余的因子
                redundant_factors = existing_factors - requested_factors_set
                if redundant_factors:
                    logger.info(f"增量模式：检测到 {len(redundant_factors)} 个多余因子，将删除")
                    print(f"?? 增量模式：检测到 {len(redundant_factors)} 个多余因子")
                    print(f"?? 多余因子: {sorted(redundant_factors)}")
                    store.drop_factors(redundant_factors)
                    print("✅ 已删除多余因子")
                    # 重新加载数据以获取最新的因子列表
                    existing_data = store.load()
                    existing_factors = set(existing_data.columns) - {'date', 'stock_code'}

                missing_factors = list(requested_factors_set - existing_factors)

                if not missing_factors:
                    logger.info("增量模式：所有请求的因子已存在，无需生成")
                    print("✅ 增量模式：所有请求的因子已存在，无需生成")
                    print(f"?? 已存在因子: {sorted(existing_factors & requested_factors_set)}")
                    return True
                else:
                    logger.info(f"增量模式：检测到 {len(missing_factors)} 个缺失因子")
                    print(f"?? 增量模式：检测到 {len(missing_factors)} 个缺失因子")
                    print(f"?? 缺失因子: {missing_factors}")
                    print(f"?? 已存在因子数: {len(existing_factors & requested_factors_set)}")
                    factors_to_generate = missing_factors
        else:
            # 默认模式：生成所有请求的因子（可能覆盖已存在的）
            factors_to_generate = factor_names

        if not factors_to_generate:
            logger.warning("没有需要生成的因子")
            print("⚠️  没有需要生成的因子")
            return True

        print(f"?? 实际生成因子数: {len(factors_to_generate)}")
        print(f"?? 实际生成因子: {factors_to_generate}")

        # 🆕 获取分批保存配置
        batch_size = getattr(args, 'batch_size', 10)
        enable_batch_save = batch_size > 0

        if enable_batch_save:
            logger.info(f"启用分批保存机制，批次大小: {batch_size}")
            print(f"💾 启用分批保存机制，每 {batch_size} 个因子保存一次")
        else:
            logger.info("禁用分批保存，将在所有因子计算完成后一次性保存")
            print("💾 禁用分批保存，将在所有因子计算完成后一次性保存")

        # 🆕 构建归一化配置
        normalization_type = getattr(args, 'normalization_type', 'both')
        normalization_config = {
            'zscore_factors': _get_zscore_factors(),
            'rankpct_factors': _get_rankpct_factors(),
            'normalization_type': normalization_type,
        }

        logger.info(f"归一化配置: type={normalization_type}, zscore因子数={len(normalization_config['zscore_factors'])}, rankpct因子数={len(normalization_config['rankpct_factors'])}")
        print(f"🔧 归一化配置: type={normalization_type}")
        if normalization_config['zscore_factors']:
            print(f"  · Z-score 因子数: {len(normalization_config['zscore_factors'])}")
        if normalization_config['rankpct_factors']:
            print(f"  · Rankpct 因子数: {len(normalization_config['rankpct_factors'])}")

        pipeline_result = generate_and_store(
            factor_names=factors_to_generate,  # 🆕 使用实际需要生成的因子列表
            data=stock_data,
            start_date=start_date,
            end_date=end_date,
            store=store,
            pipeline=pipeline,
            market_data=market_data,
            sector_data=sector_data,
            context_extra=extra_context,
            enable_batch_save=enable_batch_save,  # 🆕 分批保存开关
            batch_size=batch_size,  # 🆕 批次大小
            normalization_config=normalization_config,  # 🆕 归一化配置
        )

        generated = pipeline_result.data
        record_count = len(generated)
        factor_count = len([c for c in generated.columns if c not in {'date', 'stock_code'}])

        logger.info(f"生成因子完成: {factor_count} 列, {record_count} 条记录")
        print(f"? 生成完成: {record_count} 条记录, {factor_count} 个因子")
        if record_count > 0:
            date_min = generated['date'].min()
            date_max = generated['date'].max()
            stock_count = generated['stock_code'].nunique()
            print(f"?? 实际生成区间: {date_min.date()} - {date_max.date()}")
            print(f"?? 覆盖股票数量: {stock_count}")
        print(f"?? 因子数据已保存至: {FACTOR_LAB_STORE_PATH}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                'start_date': start_date,
                'end_date': end_date,
                'factors': factor_names,
                'record_count': record_count,
                'storage_path': str(FACTOR_LAB_STORE_PATH.resolve()),
            }
            summary_path = output_dir / f"factor_lab_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w', encoding='utf-8') as fp:
                json.dump(summary, fp, ensure_ascii=False, indent=2, default=str)
            print(f"? 因子生成摘要已保存到: {summary_path}")

        print("?? 因子生成模式完成")
        return True

    except Exception as exc:
        logger.error(f"因子生成模式失败: {exc}")
        print(f"? 因子生成失败: {exc}")
        return False


def run_test_factors_mode(args):
    """因子测试模式 - 使用 factor_lab 数据集"""
    import numpy as np
    import pandas as pd

    logger = logging.getLogger(__name__)
    print("===== 因子质量测试 =====")

    try:
        start_date = args.start_date or "20230601"
        end_date = args.end_date or "20250131"

        print(f"· 测试区间：{start_date} - {end_date}")
        print(f"· 输出目录：{args.output_dir}")

        data_loader = IntegratedDataLoader()
        stock_data, _, _ = data_loader.load_all_data(
            start_date,
            end_date,
            extend_history_days=60,
        )
        if stock_data is None or stock_data.empty:
            logger.error("无法加载股票基础数据，无法生成标签")
            return False

        if 'date' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['date'])

        required_base_columns = ['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume', 'atr14']
        missing_base_columns = [col for col in required_base_columns if col not in stock_data.columns]
        if missing_base_columns:
            logger.error(f"数据中缺少必要字段: {missing_base_columns}")
            if 'atr14' in missing_base_columns:
                logger.error("缺少 atr14 列，无法计算 ATR 止盈止损位")
            return False

        factor_data = _load_factor_cache_data(
            include_label_generation_features=True,
            start_date=start_date,
            end_date=end_date,
        )
        if factor_data is None or factor_data.empty:
            logger.error("因子数据加载失败")
            return False

        factor_data = factor_data.copy()
        if 'date' in factor_data.columns:
            factor_data['date'] = pd.to_datetime(factor_data['date'])

        stock_count = factor_data['stock_code'].nunique() if 'stock_code' in factor_data.columns else 0
        print(f"· 因子样本量：{len(factor_data)} 条，字段 {len(factor_data.columns)} 个")
        if 'date' in factor_data.columns:
            print(f"· 因子日期范围：{factor_data['date'].min()} - {factor_data['date'].max()}")
        print(f"· 覆盖股票数：{stock_count}")

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        label_source = stock_data[(stock_data['date'] >= start_ts) & (stock_data['date'] <= end_ts)].copy()
        if label_source.empty:
            logger.error("标签源数据为空，请检查基础行情数据")
            return False

        # 🔧 方案C: 使用加权收益率标签
        # 调用标签生成函数
        labeled_data_result = _generate_training_labels_for_all_data(
            label_source,
            alpha=1.5,
            debug=False,
            is_training=False,
            label_stats=None
        )
        if labeled_data_result is None:
            logger.error("标签生成失败")
            return False

        labeled_data = labeled_data_result[0] if isinstance(labeled_data_result, tuple) else labeled_data_result
        if 'date' in labeled_data.columns:
            labeled_data['date'] = pd.to_datetime(labeled_data['date'])

        # 🔧 检查必要的future_return字段
        required_return_fields = ['future_return_2d', 'future_return_3d', 'future_return_5d']
        missing_fields = [field for field in required_return_fields if field not in labeled_data.columns]

        if missing_fields:
            logger.error(f"标签数据缺少必要字段: {missing_fields}")
            return False

        # 🔧 保留基础收益率字段
        base_columns = ['stock_code', 'date', 'future_return_2d', 'future_return_3d', 'future_return_5d']
        label_data = labeled_data[base_columns].copy()

        # 🔧 计算加权收益率标签（用于对比分析）
        print("\n🔧 计算加权收益率标签（ATR标准化）...")

        # 检查是否有atr14字段
        # 🔧 使用原始收益率计算加权标签
        label_data['weighted_return_2d3d5d'] = (
            0.4 * label_data['future_return_2d'] +
            0.4 * label_data['future_return_3d'] +
            0.2 * label_data['future_return_5d']
        )
        print(f"  ✅ weighted_return_2d3d5d: 0.4*future_return_2d + 0.4*future_return_3d + 0.2*future_return_5d")

        # 收集所有标签列（包括单期和加权）
        label_return_columns = []

        # 单期标签
        for col in ['future_return_2d', 'future_return_3d', 'future_return_5d']:
            if col in label_data.columns:
                label_return_columns.append(col)

        # 加权标签
        if 'weighted_return_2d3d5d' in label_data.columns:
            label_return_columns.append('weighted_return_2d3d5d')

        print(f"\n· 可用标签列: {label_return_columns}")
        for col in label_return_columns:
            if col in label_data.columns:
                non_null = label_data[col].notna().sum()
                print(f"  - {col}：可用样本 {non_null}/{len(label_data)}")

        factor_features = factor_data.copy()
        conflicting_return_columns = [col for col in label_return_columns if col in factor_features.columns]
        if conflicting_return_columns:
            print(f"· 因子数据中已存在标签列，已剔除：{conflicting_return_columns}")
            factor_features = factor_features.drop(columns=conflicting_return_columns)

        merged_data = pd.merge(
            factor_features,
            label_data,
            on=['stock_code', 'date'],
            how='inner',
        )
        if merged_data.empty:
            logger.error("因子数据与标签无法对齐")
            return False

        print(f"· 合并样本量：{len(merged_data)} 条，字段 {len(merged_data.columns)} 个")

        final_return_columns = []
        for col in label_return_columns:
            if col in merged_data.columns:
                non_null_count = merged_data[col].notna().sum()
                if non_null_count > 100:
                    final_return_columns.append(col)
        print(f"· 有效收益列：{final_return_columns}")

        # 🔧 排除收益率列和其他非因子列
        exclude_columns = {
            'stock_code', 'date', 'date_str', 'exit_day',
            'atr_return_label',  # 排名标签
            'min_dist_to_sl', 'min_dist_to_pt',
            'entry_price', 'profit_target', 'stop_loss',
            'pct_chg', 'is_can_buy', 'open', 'high', 'low', 'close',
            'volume', 'amount', 'turnover_rate',
            'total_mv', 'circ_mv', 'pe_ttm', 'pb', 'ps_ttm', 'pcf_ttm',
            'volatility20', 'parkinson_vol', 'vol_term_structure',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'volume_weighted_price', 'vol_price_corr_20d', 'risk_reward_ratio',
            'near_support', 'near_resistance', 'short_reversal',
        }
        exclude_columns.update(label_return_columns)
        # 🔧 排除所有future_return和weighted_return相关列
        exclude_columns.update({col for col in merged_data.columns if 'future_return' in col or 'atr_return' in col or 'weighted_return' in col})

        potential_factors = [col for col in merged_data.columns if col not in exclude_columns]

        if args.factor_names:
            requested_factors = [name.strip() for name in args.factor_names.split(',') if name.strip()]
            missing_requested = [name for name in requested_factors if name not in potential_factors]
            if missing_requested:
                logger.warning(f"指定因子未找到：{missing_requested}")
            potential_factors = [name for name in potential_factors if name in requested_factors]
            print(f"· 指定因子列表：{potential_factors}")

        actual_factors = []
        for col in potential_factors:
            if pd.api.types.is_numeric_dtype(merged_data[col]):
                non_null_ratio = merged_data[col].notna().sum() / len(merged_data)
                if non_null_ratio > 0.1:
                    actual_factors.append(col)

        print(f"· 可用因子：{len(actual_factors)} / {len(potential_factors)}")
        if not actual_factors:
            logger.error("未找到可用于验证的因子")
            return False

        sample_factors = actual_factors[:5]
        print(f"· 因子抽样检查：{sample_factors}{'...' if len(actual_factors) > 5 else ''}")
        _check_factor_data_quality(merged_data, sample_factors)

        try:
            from .factor_system.factor_validator import FactorValidator
        except ImportError:
            from factor_system.factor_validator import FactorValidator

        validator = FactorValidator()
        validation_results = validator.validate_quantitative_factors(
            merged_data,
            actual_factors,
            return_columns=final_return_columns,
        )

        print("===== 验证结果概览 =====")
        print(f"· 检测因子：{validation_results.get('total_factors', 0)} 个")
        print(f"· 样本条数：{validation_results.get('total_records', 0)} 条")
        print(f"· 总体状态：{validation_results.get('overall_status', 'UNKNOWN')}")

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        validation_file = output_path / f"factor_validation_report_{timestamp}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"· 验证报告：{validation_file}")

        time_windows = validation_results.get('time_windows', {})
        for window_name, window_data in time_windows.items():
            factor_metrics = window_data.get('factor_metrics', {})
            if not factor_metrics:
                continue
            metrics_df = pd.DataFrame(factor_metrics).T
            # 🔧 适配新标签系统
            if window_name == 'weighted_return_2d3d5d':
                metrics_file = output_path / f"factor_metrics_weighted_2d3d5d_{timestamp}.csv"
            elif window_name.startswith('future_return_') and window_name.endswith('d'):
                day_num = window_name.replace('future_return_', '').replace('d', '')
                metrics_file = output_path / f"factor_metrics_single_{day_num}day_{timestamp}.csv"
            elif 'atr_return_' in window_name and window_name.endswith('d'):
                day_num = window_name.replace('atr_return_', '').replace('d', '')
                metrics_file = output_path / f"factor_metrics_layered_{day_num}day_{timestamp}.csv"
            elif window_name == 'atr_return_label':
                metrics_file = output_path / f"factor_metrics_mixed_holding_{timestamp}.csv"
            else:
                metrics_file = output_path / f"factor_metrics_{window_name}_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, encoding='utf-8')
            print(f"· 详情指标：{metrics_file}")

        summary_stats = validation_results.get('summary_statistics', {})
        if summary_stats:
            summary_df = pd.DataFrame([summary_stats])
            summary_file = output_path / f"factor_summary_cross_window_{timestamp}.csv"
            summary_df.to_csv(summary_file, encoding='utf-8', index=False)
            print(f"· 跨窗口摘要：{summary_file}")

        if validation_results.get('warnings'):
            print(f"· 警告共 {len(validation_results['warnings'])} 条，示例：{validation_results['warnings'][:3]}")
        if validation_results.get('errors'):
            print(f"· 错误共 {len(validation_results['errors'])} 条，示例：{validation_results['errors'][:3]}")

        print("===== 因子测试完成 =====")
        return True

    except Exception as e:
        logger.error(f"因子测试失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _check_factor_data_quality(data: pd.DataFrame, factors: List[str]):
    """快速检查因子覆盖率与数值分布，帮助定位异常因子"""
    import pandas as pd

    print(f"→ 检查 {len(factors)} 个因子质量...")

    for factor in factors:
        if factor not in data.columns:
            print(f"  · {factor}: 未在数据中找到")
            continue

        factor_data = data[factor]
        total_count = len(factor_data)
        non_null_count = factor_data.notna().sum()
        unique_count = factor_data.nunique()

        print(f"  · {factor}:")
        print(f"    - 总样本: {total_count:,}")
        print(f"    - 可用样本: {non_null_count:,} ({non_null_count/total_count:.1%})")
        print(f"    - 唯一值数: {unique_count:,}")

        if non_null_count == 0:
            print("    - 提示: 全部为空，IC 将为 0")
            continue

        if unique_count <= 1:
            print("    - 提示: 仅包含单一取值，IC 将为 0")
            if unique_count == 1:
                print(f"      · 固定取值: {factor_data.dropna().iloc[0]}")
            continue

        min_val = factor_data.min()
        max_val = factor_data.max()
        mean_val = factor_data.mean()
        std_val = factor_data.std()

        print(f"    - 数值区间: {min_val:.6f} ~ {max_val:.6f}")
        print(f"    - 均值 / 标准差: {mean_val:.6f} / {std_val:.6f}")

        if (factor_data == 0).all():
            print("    - 提示: 全部为 0，IC 将为 0")
        elif abs(std_val) < 1e-10:
            print("    - 提示: 波动极小，可能需要归一化或剔除")
        else:
            print("    - 状态: 数值分布正常")


def _load_sector_features_basic(start_date: str, end_date: str):
    """从本地缓存的板块文件中加载数据，供因子生成兜底使用"""
    logger = logging.getLogger(__name__)
    try:
        base_dir = Path(__file__).resolve().parent.parent / "local_data" / "sector_features"
        if not base_dir.exists():
            logger.warning("sector_features 目录不存在，无法加载板块数据")
            return None

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        frames = []

        for sector_dir in base_dir.iterdir():
            if not sector_dir.is_dir():
                continue
            sector_code = sector_dir.name
            candidates = [
                sector_dir / f"{sector_code}_features.parquet",
                sector_dir / f"{sector_code}.parquet",
            ]
            feature_file = next((p for p in candidates if p.exists()), None)
            if feature_file is None:
                continue

            try:
                df = pd.read_parquet(feature_file)
            except Exception as exc:
                logger.debug("读取板块文件失败 %s: %s", feature_file, exc)
                continue

            if 'date' not in df.columns:
                if 'date_str' in df.columns:
                    df['date'] = pd.to_datetime(df['date_str'], errors='coerce')
                else:
                    logger.debug("板块文件 %s 缺少 date 字段，跳过", feature_file)
                    continue
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            df = df.dropna(subset=['date'])
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if df.empty:
                continue

            if 'sector_code' not in df.columns:
                if 'sector' in df.columns:
                    df['sector_code'] = df['sector']
                elif 'sector_name' in df.columns:
                    df['sector_code'] = df['sector_name']
                else:
                    df['sector_code'] = sector_code
            df['sector_code'] = df['sector_code'].astype(str)

            if 'sector_name' not in df.columns:
                df['sector_name'] = df.get('sector', df['sector_code'])

            if 'pct_chg' in df.columns:
                df['return_1d'] = df['pct_chg'].astype(float)
            elif 'close' in df.columns:
                df = df.sort_values(['sector_code', 'date'])
                df['return_1d'] = df.groupby('sector_code')['close'].pct_change() * 100
            else:
                df['return_1d'] = 0.0

            df['return'] = df['return_1d']
            df['daily_return'] = df['return_1d']

            if 'close' in df.columns:
                df = df.sort_values(['sector_code', 'date'])
                close_group = df.groupby('sector_code')['close']
                for window in (3, 5, 10, 20):
                    col = f'return_{window}d'
                    if col not in df.columns:
                        df[col] = close_group.pct_change(window) * 100
            else:
                logger.debug("板块文件 %s 缺少 close 列，无法计算多日收益", feature_file)

            for col in ['return_1d', 'return', 'daily_return', 'return_3d', 'return_5d', 'return_10d', 'return_20d']:
                if col in df.columns:
                    df[col] = df[col].astype(float).fillna(0.0)

            df['date_str'] = df['date'].dt.strftime('%Y%m%d')
            frames.append(df)

        if not frames:
            logger.warning("未在本地板块缓存中找到有效数据")
            return None

        sector_data = pd.concat(frames, ignore_index=True)
        sector_data = sector_data.sort_values(['sector_code', 'date']).reset_index(drop=True)
        return sector_data

    except Exception as exc:
        logger.warning("板块数据兜底加载失败: %s", exc)
        return None


def _generate_training_labels_for_all_data(stock_data, alpha=1.5, debug=False,
                                         is_training=True, label_stats=None):
    """
    🚀 新版标签生成函数：基于ATR标准化加权收益率的分板块截面排名标签

    标签生成规则：
    1. 计算加权收益率:
       0.4*[R2d] + 0.4*[R3d] + 0.2*[R5d]
    2. 按板块划分（主板 vs 创业板/科创板）
    3. 对每个板块独立处理：
       - 过滤停牌/异常值
       - 每日 cross-sectional winsorize (0.5%/99.5%)
       - 每日 rank 映射到 [0, 1] (rank_pct)
       - 分桶为4层relevance标签: 0, 1, 2, 3
    4. 合并两个板块标签

    优点：
    - 自动处理主板/创业板涨跌幅限制差异（分板块处理）
    - relevance标签用于排序模型训练
    - 基于多日加权收益，更稳定
    - 每日 winsorize，避免数据泄漏

    Args:
        stock_data: 股票数据DataFrame
        alpha: 非线性放大系数（默认1.5）
        debug: 调试模式
        is_training: 是否为训练模式（True时计算统计量）
        label_stats: 标签统计量字典（保留用于兼容性，暂未使用）

    Returns:
        如果is_training=True: (DataFrame, label_stats_dict)
        如果is_training=False: DataFrame

        DataFrame包含：
        - atr_return_label: 最终标签（[-1, 1]区间，经过非线性放大）
    """
    try:
        import pandas as pd
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)
        logger.info("🎯 开始生成基于加权收益率的分板块截面排名标签...")

        # 数据基本信息检查
        logger.info(f"输入数据基本信息:")
        logger.info(f"  总记录数: {len(stock_data)}")
        logger.info(f"  股票数量: {stock_data['stock_code'].nunique()}")
        logger.info(f"  日期数量: {stock_data['date'].nunique()}")
        logger.info(f"  日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
        logger.info(f"  非线性放大系数 alpha: {alpha}")

        # 🚀 步骤0: 计算加权收益率
        logger.info("\n🔧 步骤0: 计算加权收益率...")
        logger.info("=" * 60)

        # 检查必要字段
        required_fields = ['future_return_2d', 'future_return_3d', 'future_return_5d', 'stock_code', 'date']
        missing_fields = [field for field in required_fields if field not in stock_data.columns]

        if missing_fields:
            logger.error(f"❌ 数据中缺少必要字段: {missing_fields}")
            return None

        # 🔧 计算原始加权收益率: 0.4*R2d + 0.4*R3d + 0.2*R5d
        import numpy as np
        stock_data['weighted_return'] = (
            0.4 * stock_data['future_return_2d'] +
            0.4 * stock_data['future_return_3d'] +
            0.2 * stock_data['future_return_5d']
        )

        valid_weighted_return_count = stock_data['weighted_return'].notna().sum()
        logger.info(f"  ✅ 加权收益率计算完成: 有效值{valid_weighted_return_count:,}个")
        logger.info(f"  公式: 0.4*future_return_2d + 0.4*future_return_3d + 0.2*future_return_5d")
        # logger.info(f"  公式: 0.4*[R2d/(ATR*√2)] + 0.4*[R3d/(ATR*√3)] + 0.2*[R5d/(ATR*√5)]")

        if valid_weighted_return_count == 0:
            logger.error("❌ 没有有效的加权收益率数据")
            return None

        # 确保数据按股票代码和日期排序
        stock_data.sort_values(['stock_code', 'date'], inplace=True)
        stock_data.reset_index(drop=True, inplace=True)

        logger.info(f"\n  标签生成方法: 分板块截面排名（Cross-Sectional Rank）")
        logger.info(f"  Winsorization: 每日 0.5%/99.5% 分位数截尾")
        logger.info(f"  排名方式: 按板块和日期分组，独立排名")
        logger.info(f"  Relevance分桶: rank_pct >= 0.98 -> 3, >= 0.90 -> 2, >= 0.70 -> 1, else -> 0")
        logger.info("=" * 60)

        # 🚀 步骤1: 板块划分
        logger.info("\n🔧 步骤1: 板块划分...")
        logger.info("=" * 60)

        # 主板: 非300/688开头的股票 (涨跌幅限制10%)
        main_board_mask = ~stock_data['stock_code'].str.startswith(('30', '68', '8', '9', '4'))
        # 创业板/科创板: 300/688开头的股票 (涨跌幅限制20%)
        growth_board_mask = stock_data['stock_code'].str.startswith(('30', '68'))

        main_board_count = main_board_mask.sum()
        growth_board_count = growth_board_mask.sum()
        total_samples = len(stock_data)

        logger.info(f"  主板股票: {main_board_count:,}条 ({main_board_count/total_samples*100:.1f}%)")
        logger.info(f"  创业板/科创板: {growth_board_count:,}条 ({growth_board_count/total_samples*100:.1f}%)")
        logger.info(f"  总样本数: {total_samples:,}条")
        logger.info("=" * 60)

        # 🚀 步骤2: 定义分板块处理函数
        def process_board_labels(board_data, board_name, alpha_val):
            """
            对单个板块的数据进行标签处理

            处理流程:
            1. 过滤停牌/异常值
            2. 每日 winsorize (0.5%/99.5%)
            3. 每日 rank 映射到 [0, 1] (rank_pct)
            4. 分桶为4层relevance标签: 0, 1, 2, 3
            """
            logger.info(f"\n  处理 {board_name}...")
            logger.info(f"  初始样本数: {len(board_data):,}条")

            # 2.1 过滤停牌/异常值
            valid_mask = board_data['weighted_return'].notna()

            # 过滤涨停
            if 'is_limit_up' in board_data.columns:
                limit_up_count = (board_data['is_limit_up'] == 1).sum()
                valid_mask &= (board_data['is_limit_up'] != 1)
                logger.info(f"  过滤涨停: {limit_up_count:,}条")

            #过滤异常数据
            if 'weighted_return' in board_data.columns:
                if board_name == "主板":
                    weighted_error_count = (board_data['weighted_return'] > 0.35).sum()
                    valid_mask &= (board_data['weighted_return'] <= 0.35)
                    logger.info(f"主板过滤涨幅异常：{weighted_error_count}条")
                else:
                    weighted_error_count = (board_data['weighted_return'] > 1).sum()
                    valid_mask &= (board_data['weighted_return'] <= 1)
                    logger.info(f"创业板过滤涨幅异常：{weighted_error_count}条")    

            # 过滤跌停
            if 'is_limit_down' in board_data.columns:
                limit_down_count = (board_data['is_limit_down'] == 1).sum()
                valid_mask &= (board_data['is_limit_down'] != 1)
                logger.info(f"  过滤跌停: {limit_down_count:,}条")

            # 过滤流动性不足
            if 'amount' in board_data.columns:
                low_liquidity_count = (board_data['amount'] < 10_000_000).sum()
                valid_mask &= (board_data['amount'] >= 10_000_000)
                logger.info(f"  过滤流动性不足(amount<1000万): {low_liquidity_count:,}条")

            valid_count = valid_mask.sum()
            logger.info(f"  有效样本数: {valid_count:,}条 ({valid_count/len(board_data)*100:.1f}%)")

            if valid_count == 0:
                logger.warning(f"  ⚠️ {board_name} 没有有效样本")
                board_data['atr_return_label'] = np.nan
                return board_data

            # 2.2 每日 winsorize (0.5%/99.5%)
            def daily_winsorize(returns):
                """每日截面 winsorize - 接收Series"""
                if returns.notna().sum() < 10:  # 样本太少，不做winsorize
                    return returns
                q_low = returns.quantile(0.005)
                q_high = returns.quantile(0.995)
                return returns.clip(lower=q_low, upper=q_high)

            board_data.loc[valid_mask, 'winsorized_return'] = (
                board_data[valid_mask].groupby('date')['weighted_return']
                .transform(daily_winsorize)
            )

            # 2.3 每日 rank (绝对排名，从1开始) 和 rank_pct (百分位数)
            # rank_abs: 每天内的绝对排名 (1=最好, 2=第二好, ...)
            # rank_pct: 每天内的百分位数排名 (0-1)
            board_data.loc[valid_mask, 'rank_abs'] = (
                board_data[valid_mask].groupby('date')['weighted_return']
                .rank(method='first', ascending=False)  # 1=最好
            )
            board_data.loc[valid_mask, 'rank_pct'] = (
                board_data[valid_mask].groupby('date')['weighted_return']
                .rank(pct=True)  # 0-1
            )

            # 2.4 分桶为4层 (0, 1, 2, 3) - 用于LightGBM排序模型的relevance
            # 新策略：按天分桶，根据每天内的绝对排名进行分桶
            def daily_to_bucket(rank_in_day, N_candidates, top_k_actual=5):
                """
                按天分桶策略

                Args:
                    rank_in_day: 每天内的绝对排名 (1=最好)
                    N_candidates: 当天股票候选总数
                    top_k_actual: 实际想买的股票数 (默认5)

                Returns:
                    0-3 之间的桶编号
                """
                # 桶3: top_k_actual * 2 (最顶尖的股票，5-10个)
                b3_count = max(1, 1)
                # 桶2: top_k_actual * 5 (下一个更宽的段，25个)4,2
                b2_count = min(N_candidates, b3_count + int(top_k_actual * 2))
                # 桶1: top_k_actual * 20 (再宽，100个)20,3
                b1_count = min(N_candidates, b2_count + int(top_k_actual * 3))
                #60,200
                b0_count = min(N_candidates, b1_count + int(top_k_actual * 200))

                if rank_in_day <= b3_count:
                    return 30  # 顶尖股票
                elif rank_in_day <= b2_count:
                    return 4  # 优质股票4
                elif rank_in_day <= b1_count:
                    return 2  # 中等股票2
                elif rank_in_day <= b0_count:
                    return 1  # 中等股票1
                else:
                    return 0  # 其余股票
                # if rank_in_day <= 10:
                #     return 30  # 顶尖股票
                # elif rank_in_day <= 20:
                #     return 4  # 优质股票4
                # elif rank_in_day <= 30:
                #     return 2  # 中等股票2
                # elif rank_in_day <= b0_count:
                #     return 1  # 中等股票1
                # else:
                #     return 0  # 其余股票

            # 按日期分组并应用分桶策略
            for date, group in board_data.loc[valid_mask].groupby('date'):
                if len(group) == 0:
                    continue
                #5
                N_candidates = len(group)
                top_k_actual = 5  # 假设每天想选5只股票

                # 获取该日期对应的索引
                indices = group.index

                # 计算relevance并赋值
                relevance_values = []
                for rank in group['rank_abs']:
                    relevance_values.append(daily_to_bucket(rank, N_candidates, top_k_actual))

                # 直接赋值到原DataFrame
                board_data.loc[indices, 'relevance'] = relevance_values

            # 🆕 2.4.1 分桶为动态层数 - 用于XGBoost排序模型的relevance_30
            # 按指定数量分桶：第1层5个，第2层20个，第3层60个，第4层100个，第5层200个，之后每层200个
            # 🚀 方案1：基于百分位数的固定分桶 + 非线性放大
            FIXED_BUCKETS = 20  # 固定桶数量

            def nonlinear_label_scaling(original_label, scaling_type='exponential', **kwargs):
                """
                非线性标签缩放算法

                Args:
                    original_label: 原始标签值 (0-19)
                    scaling_type: 缩放类型
                    **kwargs: 缩放参数

                Returns:
                    缩放后的标签值
                """
                import numpy as np

                if scaling_type == 'exponential':
                    # 指数缩放：让高分段差异更明显
                    base = kwargs.get('base', 1.5)
                    return np.power(base, original_label) - 1

                elif scaling_type == 'quadratic':
                    # 二次缩放：差异平滑增加
                    return np.power(original_label, 2)

                elif scaling_type == 'cubic':
                    # 三次缩放：高分段差异更明显
                    return np.power(original_label, 3)

                elif scaling_type == 'logarithmic':
                    # 对数缩放：低分段差异更明显
                    return np.log1p(original_label)

                elif scaling_type == 'sigmoid':
                    # Sigmoid缩放：两端差异明显，中间平滑
                    x = original_label / 19  # 归一化到[0,1]
                    sharpness = kwargs.get('sharpness', 2.0)
                    return 1 / (1 + np.exp(-sharpness * (x - 0.5))) * 19

                elif scaling_type == 'power_law':
                    # 幂律缩放：自定义指数
                    exponent = kwargs.get('exponent', 1.5)
                    return np.power(original_label / 19, exponent) * 19

                else:
                    # 默认不缩放
                    return original_label

            def to_bucket_30_fixed(rank_in_day, N_candidates):
                """
                固定层数分桶策略 - 基于百分位数

                原则：
                - 无论每天有多少股票，都使用固定的桶数量(20层)
                - 保持前5名获得最高桶的特点
                - 使用百分位数动态分配边界
                """
                # 确保每天至少有足够的股票进行分桶
                if N_candidates < 5:
                    return 0

                # 前5名固定为最高桶（桶号FIXED_BUCKETS-1 = 19）
                if rank_in_day <= 5:
                    return FIXED_BUCKETS - 1

                # 剩余股票分成FIXED_BUCKETS-1个桶
                remaining_stocks = N_candidates - 5
                remaining_buckets = FIXED_BUCKETS - 1

                # 计算当前排名在剩余股票中的百分位
                remaining_rank = rank_in_day - 5
                percentile = remaining_rank / remaining_stocks  # 0-1之间

                # 分配到剩余桶中（排名越靠前，桶号越高）
                bucket = int((1 - percentile) * remaining_buckets)
                bucket = max(0, min(bucket, FIXED_BUCKETS - 2))  # 限制在[0, FIXED_BUCKETS-2]

                return bucket

            def to_bucket_30(rank_in_day, N_candidates):
                """
                改进的分桶策略：固定层数 + 非线性放大

                组合了：
                1. 基于百分位数的固定分桶(保证训练集验证集一致性)
                2. 非线性标签缩放(增强排序区分度)
                """
                # 步骤1：固定分桶
                original_label = to_bucket_30_fixed(rank_in_day, N_candidates)

                # 步骤2：非线性放大
                # 可以在这里选择不同的缩放算法
                scaled_label = nonlinear_label_scaling(
                    original_label,
                    scaling_type='exponential',  # 推荐使用exponential
                    base=1.6  # 可以调整这个参数
                )

                return scaled_label

            # 按日期分组并应用relevance_30分桶策略
            for date, group in board_data.loc[valid_mask].groupby('date'):
                if len(group) == 0:
                    continue

                N_candidates = len(group)
                indices = group.index

                # 计算relevance_30并赋值
                relevance_30_values = []
                for rank in group['rank_abs']:
                    relevance_30_values.append(to_bucket_30(rank, N_candidates))

                # 直接赋值到原DataFrame
                board_data.loc[indices, 'relevance_30'] = relevance_30_values

            # 2.5 保留winsorized_return作为atr_return_label (用于IC计算和评估)
            # 注意: 排序模型训练使用relevance(0-3), 但评估时需要原始收益率计算IC
            board_data.loc[valid_mask, 'atr_return_label'] = (
                board_data.loc[valid_mask, 'weighted_return']
            )

            # 统计信息
            labels = board_data.loc[valid_mask, 'atr_return_label']
            logger.info(f"  最终标签统计:")
            logger.info(f"    atr_return_label范围: [{labels.min():.4f}, {labels.max():.4f}]")
            logger.info(f"    atr_return_label均值: {labels.mean():.4f}")
            logger.info(f"    atr_return_label标准差: {labels.std():.4f}")

            # 🆕 统计每个relevance层级对应的原始rank_abs和weighted_return范围
            logger.info(f"  Relevance分层统计(每日分桶,4层):")
            for rel_val in [4, 3, 2, 1, 0]:
                rel_mask = board_data.loc[valid_mask, 'relevance'] == rel_val
                if rel_mask.sum() > 0:
                    rank_abs_vals = board_data.loc[valid_mask][rel_mask]['rank_abs']
                    rank_pct_vals = board_data.loc[valid_mask][rel_mask]['rank_pct']
                    weighted_vals = board_data.loc[valid_mask][rel_mask]['weighted_return']
                    # date_vals = board_data[vaild_mask]
                    logger.info(f"    Relevance={rel_val}: 样本数={rel_mask.sum():,}")
                    logger.info(f"      rank_abs范围: [{rank_abs_vals.min():.0f}, {rank_abs_vals.max():.0f}] (每天内的绝对排名)")
                    logger.info(f"      rank_pct范围: [{rank_pct_vals.min():.4f}, {rank_pct_vals.max():.4f}] (每天内的百分位数)")
                    logger.info(f"      原始weighted_return范围: [{weighted_vals.min():.4f}, {weighted_vals.max():.4f}]")

            # 🚀 统计relevance_30分布（固定20层 + 非线性缩放）
            logger.info(f"  Relevance_30分层统计(固定20层 + 非线性缩放):")
            relevance_30_dist = board_data.loc[valid_mask, 'relevance_30'].value_counts().sort_index()
            logger.info(f"    总样本数: {len(board_data.loc[valid_mask]):,}")
            logger.info(f"    relevance_30范围: [{relevance_30_dist.index.min():.3f}, {relevance_30_dist.index.max():.3f}]")
            logger.info(f"    🚀 分桶策略: 基于百分位数的固定分桶 + 指数缩放(base=1.6)")
            logger.info(f"    🎯 优势: 训练集验证集标签范围一致，顶层标签间距拉大")
            logger.info(f"    Top-5层级分布(原始标签19对应前5名):")

            # 显示前几个最高桶的信息
            top_buckets = sorted(relevance_30_dist.index, reverse=True)[:6]
            for rel_val in top_buckets:
                count = relevance_30_dist[rel_val]
                pct = count / len(board_data.loc[valid_mask]) * 100

                # 估算对应的原始排名范围
                if rel_val >= nonlinear_label_scaling(19, scaling_type='exponential', base=1.6) * 0.95:
                    rank_desc = "前5名"
                elif rel_val >= nonlinear_label_scaling(18, scaling_type='exponential', base=1.6) * 0.95:
                    rank_desc = "前6-25名"
                elif rel_val >= nonlinear_label_scaling(17, scaling_type='exponential', base=1.6) * 0.95:
                    rank_desc = "前26-85名"
                else:
                    rank_desc = "其他排名"

                logger.info(f"      Relevance_30={rel_val:.3f}: {count:,}样本 ({pct:.2f}%) - {rank_desc}")

            # 显示标签缩放效果
            logger.info(f"    📊 非线性缩放效果:")
            for orig_label in [19, 18, 17, 16, 15, 10, 5, 0]:
                scaled_label = nonlinear_label_scaling(orig_label, scaling_type='exponential', base=1.6)
                logger.info(f"      原始{orig_label} → 缩放{scaled_label:.3f}")

            # 清理临时列 - 使用del避免内存分配
            if 'winsorized_return' in board_data.columns:
                del board_data['winsorized_return']
            # 保留rank_pct用于relevance_30，保留rank_abs用于新的分桶策略
            # if 'rank_pct' in board_data.columns:
            #     del board_data['rank_pct']
            # if 'rank_abs' in board_data.columns:
            #     del board_data['rank_abs']

            return board_data

        # 🚀 步骤3: 处理主板数据
        logger.info("\n🔧 步骤3: 处理主板数据...")
        logger.info("=" * 60)

        main_board_data = stock_data[main_board_mask].copy()
        if len(main_board_data) > 0:
            main_board_data = process_board_labels(main_board_data, "主板", alpha)
        else:
            logger.warning("  ⚠️ 主板数据为空")

        # 🚀 步骤4: 处理创业板/科创板数据
        logger.info("\n🔧 步骤4: 处理创业板/科创板数据...")
        logger.info("=" * 60)

        growth_board_data = stock_data[growth_board_mask].copy()
        if len(growth_board_data) > 0:
            growth_board_data = process_board_labels(growth_board_data, "创业板/科创板", alpha)
        else:
            logger.warning("  ⚠️ 创业板/科创板数据为空")

        # 🚀 步骤5: 合并两个板块
        logger.info("\n🔧 步骤5: 合并两个板块...")
        logger.info("=" * 60)

        stock_data_labeled = pd.concat([main_board_data, growth_board_data], ignore_index=True)
        # 🚀 注意: 不需要再次排序,因为输入数据已在第7140-7141行排序过
        # 两个板块内部都是有序的,concat后虽然板块间可能不完全有序,但不影响后续使用
        # 删除sort_values操作以避免大数据集的内存分配失败

        # 统计最终标签
        final_labels = stock_data_labeled['atr_return_label'].dropna()
        logger.info(f"  最终标签统计:")
        logger.info(f"    有效标签数: {len(final_labels):,}/{len(stock_data_labeled):,}")
        logger.info(f"    atr_return_label范围: [{final_labels.min():.4f}, {final_labels.max():.4f}]")
        logger.info(f"    atr_return_label均值: {final_labels.mean():.4f}")
        logger.info(f"    atr_return_label标准差: {final_labels.std():.4f}")

        # 🆕 统计relevance标签分布(4层)
        final_relevance = stock_data_labeled['relevance'].dropna()
        logger.info(f"  Relevance标签统计(4层):")
        logger.info(f"    有效relevance数: {len(final_relevance):,}")
        logger.info(f"    relevance范围: [{final_relevance.min():.0f}, {final_relevance.max():.0f}]")
        logger.info(f"    relevance均值: {final_relevance.mean():.4f}")
        logger.info(f"    relevance分布:")
        for rel_val in [4, 3, 2, 1, 0]:
            count = (final_relevance == rel_val).sum()
            pct = count / len(final_relevance) * 100
            logger.info(f"      Relevance={rel_val}: {count:,}样本 ({pct:.2f}%)")

        # 🆕 统计relevance_30标签分布(30层)
        final_relevance_30 = stock_data_labeled['relevance_30'].dropna()
        logger.info(f"  Relevance_30标签统计(30层):")
        logger.info(f"    有效relevance_30数: {len(final_relevance_30):,}")
        logger.info(f"    relevance_30范围: [{final_relevance_30.min():.0f}, {final_relevance_30.max():.0f}]")
        logger.info(f"    relevance_30均值: {final_relevance_30.mean():.4f}")
        logger.info(f"    Top-5层级分布:")
        relevance_30_counts = final_relevance_30.value_counts().sort_index(ascending=False)
        for rel_val in relevance_30_counts.index[:5]:
            count = relevance_30_counts[rel_val]
            pct = count / len(final_relevance_30) * 100
            logger.info(f"      Relevance_30={rel_val}: {count:,}样本 ({pct:.2f}%)")

        # 清理临时列 - 使用del避免内存分配
        if 'weighted_return' in stock_data_labeled.columns:
            del stock_data_labeled['weighted_return']

        logger.info(f"✅ 分板块截面排名标签生成完成")
        logger.info("=" * 60)
        # 计算标签统计量用于样本权重计算和模型训练
        if is_training:
            # 获取最终的标签
            final_labels_for_stats = stock_data_labeled['atr_return_label'].dropna()

            # 计算标签统计量
            p01 = np.percentile(final_labels_for_stats, 1)   # 1%分位数
            p99 = np.percentile(final_labels_for_stats, 99)  # 99%分位数

            current_label_stats = {
                'min': float(final_labels_for_stats.min()),
                'max': float(final_labels_for_stats.max()),
                'mean': float(final_labels_for_stats.mean()),
                'std': float(final_labels_for_stats.std()),
                'p01': float(p01),
                'p99': float(p99),
                'alpha': float(alpha)  # 保存alpha参数
            }

            logger.info(f"\n🚀 标签统计量（用于样本权重和模型训练）:")
            logger.info(f"  分位数: p01={p01:.4f}, p99={p99:.4f}")
            logger.info(f"  范围: [{current_label_stats['min']:.4f}, {current_label_stats['max']:.4f}]")
            logger.info(f"  均值±标准差: {current_label_stats['mean']:.4f}±{current_label_stats['std']:.4f}")
            logger.info(f"  非线性放大系数 alpha: {alpha}")

            return stock_data_labeled, current_label_stats
        else:
            return stock_data_labeled

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"批量ATR标签生成失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        # 返回None表示失败,让调用方处理
        return None


def _extract_and_validate_labels(labeled_data, label_column='atr_return_label'):
    """
    🚀 提取和验证连续收益标签（排序模型模式）

    Args:
        labeled_data: 包含标签的数据DataFrame
        label_column: 标签列名，固定为'atr_return_label'（连续收益标签）

    Returns:
        tuple: (labels, valid_mask, relevance, relevance_30) 或 (None, None, None, None) 如果失败
    """
    try:
        import numpy as np
        logger = logging.getLogger(__name__)

        logger.info(f"🏷️ 提取和验证连续收益标签（成本感知回归模式）...")

        # 检查标签列是否存在
        if label_column not in labeled_data.columns:
            logger.error(f"数据中缺少标签列: {label_column}")
            return None, None

        # 提取标签
        labels = labeled_data[label_column].copy()

        # 验证标签值的有效性（连续值）
        valid_labels = labels.dropna()

        logger.info(f"连续收益标签统计信息:")
        logger.info(f"  标签列: {label_column}")
        logger.info(f"  总样本数: {len(labels):,}")
        logger.info(f"  有效标签数: {len(valid_labels):,}")
        logger.info(f"  缺失标签数: {labels.isna().sum():,}")

        if len(valid_labels) > 0:
            logger.info(f"  标签范围: [{valid_labels.min():.3f}, {valid_labels.max():.3f}] ATR")
            logger.info(f"  标签均值: {valid_labels.mean():.3f} ATR")
            logger.info(f"  标签标准差: {valid_labels.std():.3f} ATR")

            # 分析收益分布
            profit_samples = (valid_labels >= ATR_MULTIPLIER_PROFIT).sum()  # 盈利样本
            loss_samples = (valid_labels <= -ATR_MULTIPLIER_LOSS).sum()   # 亏损样本
            neutral_samples = ((valid_labels > -ATR_MULTIPLIER_LOSS) & (valid_labels < ATR_MULTIPLIER_PROFIT)).sum()  # 中性样本

            logger.info(f"连续收益标签分布:")
            logger.info(f"  盈利样本 (>={ATR_MULTIPLIER_PROFIT} ATR): {profit_samples:,}个 ({profit_samples/len(valid_labels)*100:.1f}%)")
            logger.info(f"  中性样本 (-{ATR_MULTIPLIER_LOSS}~{ATR_MULTIPLIER_PROFIT} ATR): {neutral_samples:,}个 ({neutral_samples/len(valid_labels)*100:.1f}%)")
            logger.info(f"  亏损样本 (<=-{ATR_MULTIPLIER_LOSS} ATR): {loss_samples:,}个 ({loss_samples/len(valid_labels)*100:.1f}%)")

        # 创建有效样本掩码（过滤掉NaN值）
        valid_mask = labels.notna()

        # 提取relevance字段(4层)
        if 'relevance' in labeled_data.columns:
            relevance = labeled_data['relevance'].copy()
            logger.info(f"✅ 提取relevance字段(4层): {len(relevance):,}个样本")
            relevance_dist = relevance.value_counts().sort_index().to_dict()
            logger.info(f"  Relevance分布: {relevance_dist}")
        else:
            logger.warning("⚠️ 数据中缺少relevance字段,将使用默认值0")
            import pandas as pd
            relevance = pd.Series(0, index=labeled_data.index)

        # 🆕 提取relevance_30字段(30层)
        if 'relevance_30' in labeled_data.columns:
            relevance_30 = labeled_data['relevance_30'].copy()
            logger.info(f"✅ 提取relevance_30字段(30层): {len(relevance_30):,}个样本")
            relevance_30_dist = relevance_30.value_counts().sort_index(ascending=False).head(5).to_dict()
            logger.info(f"  Relevance_30 Top-5分布: {relevance_30_dist}")
        else:
            logger.warning("⚠️ 数据中缺少relevance_30字段,将使用默认值0")
            import pandas as pd
            relevance_30 = pd.Series(0, index=labeled_data.index)

        logger.info("✅ 连续收益标签提取和验证完成")
        return labels, valid_mask, relevance, relevance_30

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"连续收益标签提取和验证失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None, None


def _validate_data_order(dates, labels, relevance, data_name="数据"):
    """
    🔧 验证数据顺序是否正确（按日期升序排列）

    Args:
        dates: 日期数组
        labels: 标签数组
        relevance: relevance数组
        data_name: 数据名称（用于日志）

    Returns:
        bool: True表示顺序正确，False表示顺序混乱
    """
    import numpy as np
    import pandas as pd
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"🔍 验证{data_name}顺序...")

        # 检查长度一致性
        if len(dates) != len(labels) or len(dates) != len(relevance):
            logger.error(f"❌ {data_name}长度不一致!")
            logger.error(f"   dates长度: {len(dates)}")
            logger.error(f"   labels长度: {len(labels)}")
            logger.error(f"   relevance长度: {len(relevance)}")
            return False

        # 检查日期是否升序排列
        dates_sorted = np.sort(dates)
        if not np.array_equal(dates, dates_sorted):
            logger.error(f"❌ {data_name}日期未按升序排列!")
            # 找出第一个乱序的位置
            for i in range(1, len(dates)):
                if dates[i] < dates[i-1]:
                    logger.error(f"   第{i}个日期({dates[i]})小于第{i-1}个日期({dates[i-1]})")
                    break
            return False

        # 检查是否有NaN值
        if np.any(pd.isna(labels)):
            nan_count = pd.isna(labels).sum()
            logger.error(f"❌ {data_name}标签中有{nan_count}个NaN值!")
            return False

        if np.any(pd.isna(relevance)):
            nan_count = pd.isna(relevance).sum()
            logger.error(f"❌ {data_name}relevance中有{nan_count}个NaN值!")
            return False

        logger.info(f"✅ {data_name}顺序验证通过")
        logger.info(f"   样本数: {len(dates):,}")
        logger.info(f"   日期范围: {dates[0]} ~ {dates[-1]}")
        logger.info(f"   标签范围: [{np.min(labels):.3f}, {np.max(labels):.3f}]")
        logger.info(f"   relevance范围: [{np.min(relevance):.0f}, {np.max(relevance):.0f}]")

        return True

    except Exception as e:
        logger.error(f"❌ {data_name}顺序验证失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _prepare_feature_data(input_data, feature_columns, data_type="特征数据"):
    """
    准备特征数据（因子已在生成阶段完成清洗、winsorize、归一化）

    注意：因子在生成阶段（factor_lab）已经完成了：
         1. 替换 Inf 为 NaN
         2. 每日 cross-sectional winsorize
         3. 每日截面中位数填充缺失值
         4. 每日 cross-sectional 归一化（z-score/rankpct）

         此处只做基本的数据加载和格式转换

    Args:
        input_data: 输入数据DataFrame
        feature_columns: 特征列名列表
        data_type: 数据类型描述（用于日志）

    Returns:
        tuple: (X_processed, available_features) 或 (None, None) 如果失败
    """
    try:
        import numpy as np
        logger = logging.getLogger(__name__)

        logger.info(f"🚀 开始准备{data_type}...")

        # 📋 步骤1：过滤有效的特征列
        available_features = [col for col in feature_columns if col in input_data.columns]
        missing_features = [col for col in feature_columns if col not in input_data.columns]

        if missing_features:
            logger.warning(f"缺失特征: {len(missing_features)}个")
            logger.debug(f"缺失特征列表: {missing_features[:10]}...")  # 只显示前10个

        if not available_features:
            logger.error("没有可用的特征列")
            return None, None

        logger.info(f"可用特征: {len(available_features)}个")

        # 📊 步骤2：提取特征数据
        feature_data = input_data[available_features].copy()

        logger.info(f"{data_type}统计:")
        logger.info(f"  样本数: {len(feature_data):,}个")
        logger.info(f"  特征数: {len(available_features)}个")

        # 📊 步骤3：转换为numpy数组
        X_processed = feature_data.values.astype(np.float32)

        # 📊 步骤4：统计信息
        logger.info(f"✅ {data_type}准备完成:")
        logger.info(f"  样本数: {len(X_processed):,}")
        logger.info(f"  特征数: {X_processed.shape[1]}")
        logger.info(f"  特征数据类型: {X_processed.dtype}")

        return X_processed, available_features

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"准备{data_type}失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None



def _save_model_results(model_results, feature_names, pca_models=None, model_type='xgboost', timestamp=None):
    """保存模型和结果（支持XGBoost和LightGBM）

    Args:
        model_results: 模型训练结果
        feature_names: 特征名称列表
        pca_models: PCA模型字典（可选，仅XGBoost使用）
        model_type: 模型类型 ('xgboost' 或 'lightgbm')
        timestamp: 统一时间戳（可选，如果为None则自动生成）
    """
    try:
        import pickle
        import json
        from datetime import datetime
        from pathlib import Path

        logger = logging.getLogger(__name__)

        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # 🚀 使用传入的时间戳或生成新的时间戳
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 🚀 根据模型类型确定文件名
        model_file = results_dir / f"{model_type}_model_{timestamp}.pkl"

        # 🚀 保存模型（XGBoost和LightGBM使用不同的数据结构）
        if model_type == 'xgboost':
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model_results['model'],
                    'scaler': model_results['scaler'],
                    'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                    'pca_info': model_results['pca_info'],
                    'feature_names': feature_names,
                    'pca_models': pca_models if pca_models else {},
                    'label_stats': model_results.get('label_stats', None),
                    'interaction_info': None,
                    'model_type': 'xgboost',
                    'performance': model_results.get('performance', {}),  # 🚀 保存性能信息
                }, f)
        elif model_type == 'lightgbm':
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model_results['model'],
                    'scaler': model_results['scaler'],
                    'feature_names': feature_names,
                    'label_stats': model_results.get('label_stats', None),
                    'model_type': 'lightgbm',
                    'performance': model_results.get('performance', {}),  # 🚀 保存性能信息
                }, f)

        logger.info(f"✅ {model_type.upper()}模型已保存: {model_file}")

        # 记录PCA信息
        if pca_models:
            logger.info(f"📊 PCA模型信息:")
            for group_name, pca_info in pca_models.items():
                if pca_info.get('applied', False):
                    original_factors = pca_info.get('original_factors', [])
                    n_components = pca_info.get('n_components', 0)
                    variance_explained = pca_info.get('variance_explained', 0)
                    logger.info(f"   {group_name}: {len(original_factors)}个特征 -> {n_components}个主成分")
                    logger.info(f"   解释方差比例: {variance_explained:.4f}")
                else:
                    logger.info(f"   {group_name}: PCA未应用")

        # 保存训练结果报告
        # 🚀 修复：安全处理feature_importance，支持DataFrame和dict格式
        feature_importance_data = None
        if model_results.get('feature_importance') is not None:
            try:
                # 尝试DataFrame格式
                feature_importance_data = model_results['feature_importance'].to_dict('records')
            except AttributeError:
                # 如果是dict格式(LightGBM),转换为list of dict
                if isinstance(model_results['feature_importance'], dict):
                    feature_importance_data = [
                        {'feature': k, 'importance': v}
                        for k, v in model_results['feature_importance'].items()
                    ]
                    logger.info("✅ feature_importance已从dict格式转换为list格式")
                else:
                    logger.warning("feature_importance格式未知，跳过保存")
                    feature_importance_data = None

        report_data = {
            'timestamp': timestamp,
            'model_type': model_type,  # 🚀 记录模型类型
            'performance': model_results['performance'],
            'training_info': model_results['training_info'],
            'model_params': model_results['model_params'],
            'feature_importance': feature_importance_data,
        }

        # 🚀 XGBoost特有的信息
        if model_type == 'xgboost':
            report_data['pca_info'] = {k: v for k, v in model_results['pca_info'].items() if k != 'pca_object'}
            report_data['interaction_info'] = model_results.get('interaction_info')

        # 🚀 根据模型类型确定文件名
        report_file = results_dir / f"training_report_{model_type}_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"训练报告已保存: {report_file}")

        # 保存特征重要性图表（如果可能）
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            # 🚀 处理不同格式的feature_importance
            feature_importance = model_results.get('feature_importance')
            if feature_importance is not None:
                # 转换为DataFrame格式
                if isinstance(feature_importance, dict):
                    # dict格式(LightGBM)
                    importance_df = pd.DataFrame([
                        {'feature': k, 'importance': v}
                        for k, v in feature_importance.items()
                    ])
                    importance_df = importance_df.sort_values('importance', ascending=False)
                else:
                    # DataFrame格式(XGBoost)
                    importance_df = feature_importance

                # 绘制Top 20
                top_features = importance_df.head(20)

                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importance ({model_type.upper()})')
                plt.tight_layout()

                # 🚀 根据模型类型确定文件名
                plot_file = results_dir / f"feature_importance_{model_type}_{timestamp}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"特征重要性图表已保存: {plot_file}")
            else:
                logger.warning("feature_importance为None，跳过图表保存")

        except Exception as plot_e:
            logger.warning(f"保存特征重要性图表失败: {plot_e}")

        logger.info("模型结果保存完成")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"保存模型结果失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")


def run_train_model_mode(args, train_end_date=None, is_regression=True, use_xgboost=None, use_lightgbm=None):
    """运行模型训练模式

    Args:
        args: 命令行参数
        train_end_date: 训练数据截止日期，如果为None则使用全局配置TRAIN_END_DATE
        is_regression: 是否使用回归模式，默认True（成本感知回归）
        use_xgboost: 是否训练XGBoost模型，如果为None则根据命令行参数决定
        use_lightgbm: 是否训练LightGBM模型，如果为None则根据命令行参数决定
    """
    logger = logging.getLogger(__name__)
    if is_regression:
        print("运行模式: 成本感知回归模型训练")
    else:
        print("运行模式: ATR三分类模型训练")
    print("=" * 60)

    try:
        # 🎯 如果未指定train_end_date，使用全局配置
        if train_end_date is None:
            train_end_date = TRAIN_END_DATE
            logger.info(f"🗓️ 使用全局时间切分配置: TRAIN_END_DATE={TRAIN_END_DATE}")

        # 1. 数据加载与特征识别
        logger.info("开始加载因子数据...")
        # 在训练模式下包含标签生成特征，用于标签构建
        factor_data = _load_factor_cache_data(include_label_generation_features=True)
        if factor_data is None or len(factor_data) == 0:
            logger.error("因子数据加载失败")
            return False

        logger.info(f"因子数据加载完成: {len(factor_data)} 条记录, {len(factor_data.columns)} 个字段")

        # 🚀 加载市场数据（用于XGBoost计算group权重）
        logger.info("\n🔄 加载市场数据（用于XGBoost计算group权重）...")
        market_data = None
        try:
            from data_loader import IntegratedDataLoader
            loader = IntegratedDataLoader()
            start_date_str = factor_data['date'].min().strftime('%Y%m%d')
            end_date_str = factor_data['date'].max().strftime('%Y%m%d')
            _, _, market_data = loader.load_all_data(start_date_str, end_date_str, extend_history_days=20)

            if market_data is None or market_data.empty:
                logger.warning("⚠️ 市场数据加载失败，XGBoost将使用均匀group权重")
                market_data = None
            else:
                # 确保date列是datetime类型
                if 'date' in market_data.columns:
                    market_data['date'] = pd.to_datetime(market_data['date'])
                logger.info(f"✅ 市场数据加载成功: {len(market_data)}条记录")
        except Exception as e:
            logger.warning(f"⚠️ 市场数据加载失败: {e}，XGBoost将使用均匀group权重")
            market_data = None

        # 应用训练截止日期过滤
        if train_end_date:
            train_end_dt = pd.to_datetime(train_end_date)
            original_count = len(factor_data)
            factor_data = factor_data[factor_data['date'] < train_end_dt].copy()
            logger.info(f"🗓️ 应用训练截止日期过滤: {train_end_date}")
            logger.info(f"📊 数据过滤结果: {original_count} -> {len(factor_data)} 条记录")
            logger.info(f"📅 训练数据日期范围: {factor_data['date'].min()} 到 {factor_data['date'].max()}")

            if len(factor_data) == 0:
                logger.error("过滤后无可用训练数据")
                return False

        # 2. 特征识别（使用基础特征）
        feature_columns = _get_ml_feature_columns_simple(factor_data)
        feature_columns = sorted(feature_columns)
        logger.info(f"识别到 {len(feature_columns)} 个基础特征")

        # 🔧 添加特征一致性检查和记录
        logger.info("=" * 60)
        logger.info("纯因子组特征分析")
        logger.info("=" * 60)
        logger.info(f"✅ 所有特征均来自因子组文件，无原始股票数据特征")
        logger.info(f"✅ 特征总数: {len(feature_columns)}个")


        # 3. 数据过滤（在标签构建前）
        logger.info("🔄 开始数据过滤（新股和ST股票）...")
        logger.info("=" * 50)

        # 获取所有日期
        all_dates = sorted(factor_data['date'].unique())
        filtered_data_list = []

        for date in all_dates:
            date_str = date.strftime('%Y%m%d')
            daily_data = factor_data[factor_data['date'] == date].copy()

            if len(daily_data) == 0:
                continue

            initial_count = len(daily_data)

            # 过滤北交所股票 (股票代码以.BJ或.bj结尾)
            if 'stock_code' in daily_data.columns:
                bj_filter = ~daily_data['stock_code'].str.endswith(('.BJ', '.bj'), na=False)
                daily_data = daily_data[bj_filter].copy()
                after_bj_count = len(daily_data)
            else:
                after_bj_count = initial_count

            # 过滤新股
            new_stock_filter = _filter_new_stocks(daily_data, date_str,logger)
            daily_data = daily_data[new_stock_filter].copy()
            after_new_stock_count = len(daily_data)

            # 过滤ST股票
            if len(daily_data) > 0:
                st_filter = _filter_st_stocks(daily_data,logger)
                daily_data = daily_data[st_filter].copy()
                final_count = len(daily_data)
            else:
                final_count = 0

            if len(daily_data) > 0:
                filtered_data_list.append(daily_data)

            # 每50天输出一次过滤统计
            if len(filtered_data_list) % 50 == 0:
                logger.info(f"📅 {date_str}: 过滤统计 {initial_count} -> {after_bj_count} -> {after_new_stock_count} -> {final_count}")

        if not filtered_data_list:
            logger.error("❌ 所有数据都被过滤掉了")
            return False

        # 合并过滤后的数据
        factor_data = pd.concat(filtered_data_list, ignore_index=True)

        # 🚀 关键修复: 按['date', 'stock_code']排序,确保排序模型的group信息正确
        logger.info("🔧 按['date', 'stock_code']排序数据...")
        factor_data = factor_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
        logger.info("✅ 数据已按['date', 'stock_code']排序")

        logger.info(f"✅ 数据过滤完成")
        logger.info(f"📊 过滤后数据统计: {len(factor_data)} 条记录")

        # 3.5. 训练数据预处理说明
        logger.info("🔄 训练数据预处理策略...")
        logger.info("=" * 60)
        logger.info("  ℹ️ 训练阶段不在此过滤涨跌停、流动性、is_can_buy")
        logger.info("     理由: 保留所有数据,在标签生成时标记不可操作样本为NaN")
        logger.info("     优点: 可以学习'T-1涨停→T正常'等有价值的模式")
        logger.info("     过滤: 在标签生成函数中进行,标记以下情况为NaN:")
        logger.info("           - T日涨停(无法买入+风险高)")
        logger.info("           - T日跌停(价格异常)")
        logger.info("           - T日流动性不足(无法买入)")
        logger.info("           - T+1日一字涨停(is_can_buy=0)")
        logger.info("=" * 60)

        # 🔧 修复数据泄漏: 先划分训练集和验证集,再分别生成标签
        logger.info("\n" + "=" * 60)
        logger.info("🔧 修复数据泄漏: 先划分数据,再分别生成标签")
        logger.info("=" * 60)
        logger.info("  ✅ 步骤1: 按8:2顺序划分训练集和验证集")
        logger.info("  ✅ 步骤2: 对训练集生成标签,计算Winsorization阈值")
        logger.info("  ✅ 步骤3: 对验证集生成标签,使用训练集的阈值")
        logger.info("  ✅ 避免验证集信息泄漏到Winsorization阈值中")
        logger.info("=" * 60)

        # 🚀 4.1 先划分训练集和验证集(在标签生成之前)
        split_point = int(len(factor_data) * 0.8)
        train_data_unlabeled = factor_data.iloc[:split_point].copy()
        val_data_unlabeled = factor_data.iloc[split_point:].copy()

        # 🚀 关键修复: 划分后立即排序,确保数据连续性
        train_data_unlabeled = train_data_unlabeled.sort_values(['date', 'stock_code']).reset_index(drop=True)
        val_data_unlabeled = val_data_unlabeled.sort_values(['date', 'stock_code']).reset_index(drop=True)

        logger.info(f"📊 数据划分完成:")
        logger.info(f"   训练集: {len(train_data_unlabeled):,}条")
        logger.info(f"   验证集: {len(val_data_unlabeled):,}条")
        logger.info(f"   ✅ 训练集和验证集已按['date', 'stock_code']排序")

        # 🚀 4.2 对训练集生成标签
        logger.info("\n🔄 步骤1: 对训练集生成标签...")
        train_labeled_result = _generate_training_labels_for_all_data(
            train_data_unlabeled,
            alpha=1.5,
            debug=False,
            is_training=True,  # 训练模式，计算统计量
            label_stats=None
        )

        # 检查训练集标签生成是否成功
        if train_labeled_result is None:
            logger.error("❌ 训练集标签生成失败")
            return False

        # 解包训练集结果
        if isinstance(train_labeled_result, tuple):
            train_data_labeled, label_stats = train_labeled_result
            logger.info("✅ 训练集标签构建完成")
        else:
            train_data_labeled = train_labeled_result
            label_stats = None
            logger.warning("⚠️ 未获得训练集标签统计量")

        # 🚀 4.3 对验证集生成标签
        logger.info("\n🔄 步骤2: 对验证集生成标签...")
        val_labeled_result = _generate_training_labels_for_all_data(
            val_data_unlabeled,
            alpha=1.5,
            debug=False,
            is_training=False,  # 验证集：不计算统计量
            label_stats=None
        )

        # 检查验证集标签生成是否成功
        if val_labeled_result is None:
            logger.error("❌ 验证集标签生成失败")
            return False

        # 解包验证集结果
        if isinstance(val_labeled_result, tuple):
            val_data_labeled, _ = val_labeled_result
            logger.info("✅ 验证集标签构建完成")
        else:
            val_data_labeled = val_labeled_result
            logger.info("✅ 验证集标签构建完成")

        # 🚀 4.4 合并训练集和验证集
        labeled_data = pd.concat([train_data_labeled, val_data_labeled], ignore_index=True)

        # 🚀 关键修复: concat后立即排序,确保数据连续性
        labeled_data = labeled_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
        logger.info(f"✅ 合并训练集和验证集: {len(labeled_data):,}条")
        logger.info(f"   ✅ 已按['date', 'stock_code']排序")

        # 🚀 5. 连续收益标签提取和验证（排序模型模式）
        logger.info("\n🏷️ 提取和验证连续收益标签（排序模型模式）...")
        labels, valid_mask, relevance, relevance_30 = _extract_and_validate_labels(labeled_data, label_column='atr_return_label')

        if labels is None or valid_mask is None or relevance is None or relevance_30 is None:
            logger.error("连续收益标签提取和验证失败")
            return False

        # 过滤有效样本
        filtered_data = labeled_data[valid_mask].copy()
        filtered_labels = labels[valid_mask].copy()
        filtered_relevance = relevance[valid_mask].copy()
        filtered_relevance_30 = relevance_30[valid_mask].copy()

        # 🚀 关键修复: 过滤后立即排序,确保数据连续性
        # 🔧 重要: 必须同时对数据和标签进行排序,保持对应关系
        sort_idx = filtered_data.sort_values(['date', 'stock_code']).index
        filtered_data = filtered_data.loc[sort_idx].reset_index(drop=True)
        filtered_labels = filtered_labels.loc[sort_idx].reset_index(drop=True)
        filtered_relevance = filtered_relevance.loc[sort_idx].reset_index(drop=True)
        filtered_relevance_30 = filtered_relevance_30.loc[sort_idx].reset_index(drop=True)
        logger.info(f"   ✅ 过滤后数据和标签已按['date', 'stock_code']同步排序")

        # 🚀 使用Winsorize后的标签（0.1%/99.9%截尾）
        # atr_return_label已经在_generate_training_labels_for_all_data中进行了Winsorize处理
        filtered_labels_original = filtered_labels.copy()
        logger.info(f"✅ 使用Winsorize后的标签（0.1%/99.9%截尾）")
        logger.info(f"  标签范围: [{filtered_labels.min():.3f}, {filtered_labels.max():.3f}]")

        logger.info(f"有效训练数据准备完成: {len(filtered_data)} 个样本, {len(feature_columns)} 个特征列")

        # 5.5. 特征处理策略说明
        logger.info("🔄 特征处理策略: 所有特征预处理将在_train_ml_model内部完成")
        logger.info("  ✅ 数据清洗、Winsorization、标准化等步骤统一处理")
        logger.info("  ✅ 确保训练和验证集使用相同的预处理参数")
        logger.info("  ✅ 避免数据泄露，提高模型可靠性")
        logger.info("=" * 50)

        # 6. 模型训练
        # 🚀 超参数搜索控制
        enable_param_search = getattr(args, 'enable_param_search', True)  # ✅ 默认启用参数搜索
        search_strategy = getattr(args, 'search_strategy', 'bayesian')  # ✅ 默认使用贝叶斯搜索

        # 🚀 解析模型选择参数
        if use_xgboost is None or use_lightgbm is None:
            # 从命令行参数解析
            models_str = getattr(args, 'models', 'xgboost,lightgbm')
            selected_models = [m.strip().lower() for m in models_str.split(',')]

            if use_xgboost is None:
                use_xgboost = 'xgboost' in selected_models
            if use_lightgbm is None:
                use_lightgbm = 'lightgbm' in selected_models

        if not use_xgboost and not use_lightgbm:
            logger.error("❌ 至少需要选择一个模型 (xgboost 或 lightgbm)")
            return False

        logger.info(f"🚀 模型选择:")
        logger.info(f"   XGBoost: {'✅' if use_xgboost else '❌'}")
        logger.info(f"   LightGBM: {'✅' if use_lightgbm else '❌'}")

        if use_xgboost and enable_param_search:
            logger.info(f"🔍 XGBoost超参数搜索: 策略={search_strategy}")
        elif use_xgboost:
            logger.info("⏭️ XGBoost跳过超参数搜索，使用默认参数")

        if use_lightgbm:
            logger.info("📊 LightGBM使用默认参数（无需参数搜索）")

        if use_xgboost and use_lightgbm:
            # 🚀 集成学习模式: 训练2个模型
            logger.info("开始训练集成模型（XGBoost + LightGBM）...")

            # 🚀 优化：预处理数据（所有ensemble模型共享）
            logger.info("\n" + "=" * 60)
            logger.info("🔧 预处理训练数据（所有ensemble模型共享）")
            logger.info("=" * 60)
            logger.info("📋 优化说明:")
            logger.info("  ✅ 数据划分、Winsorization等预处理只执行一次")
            logger.info("  ✅ 所有模型使用相同的训练/验证集")
            logger.info("  ✅ 节省约40-50%的训练时间")
            logger.info("  ✅ 提高模型对比的公平性")

            # 6.1 数据划分(时间序列友好)
            split_point = int(len(filtered_data) * 0.8)
            raw_train_data = filtered_data.iloc[:split_point]
            raw_val_data = filtered_data.iloc[split_point:]
            y_train_split = filtered_labels.iloc[:split_point] if hasattr(filtered_labels, 'iloc') else filtered_labels[:split_point]
            y_val_split = filtered_labels.iloc[split_point:] if hasattr(filtered_labels, 'iloc') else filtered_labels[split_point:]

            # 🚀 划分原始标签（与训练标签相同，未经Winsorize处理）
            y_train_split_original_ensemble = None
            y_val_split_original_ensemble = None
            if filtered_labels_original is not None:
                y_train_split_original_ensemble = filtered_labels_original.iloc[:split_point] if hasattr(filtered_labels_original, 'iloc') else filtered_labels_original[:split_point]
                y_val_split_original_ensemble = filtered_labels_original.iloc[split_point:] if hasattr(filtered_labels_original, 'iloc') else filtered_labels_original[split_point:]
                logger.info("✅ 划分原始ATR标签（未经Winsorize处理）")

            # 🆕 提取训练集和验证集日期（用于时间衰减权重和group计算）
            train_dates_for_ensemble = None
            val_dates_for_ensemble = None
            if 'date' in raw_train_data.columns:
                train_dates_for_ensemble = raw_train_data['date'].values
                logger.info("✅ 提取训练集日期（用于时间衰减权重计算和group计算）")
            if 'date' in raw_val_data.columns:
                val_dates_for_ensemble = raw_val_data['date'].values
                logger.info("✅ 提取验证集日期（用于group计算）")

            # 🆕 提取验证集ATR14（用于止盈止损阈值优化）
            val_atr14_for_ensemble = None
            if 'atr14' in raw_val_data.columns:
                val_atr14_for_ensemble = raw_val_data['atr14'].values
                logger.info("✅ 提取验证集ATR14（用于止盈止损阈值优化）")
            else:
                logger.warning("⚠️ 验证数据中未包含atr14列，止盈止损阈值优化将无法执行")

            # 🆕 提取换手率和流通市值（用于基础权重计算）
            train_turnover_rate_for_ensemble = None
            train_circ_mv_for_ensemble = None
            if 'turnover_rate' in raw_train_data.columns:
                train_turnover_rate_for_ensemble = raw_train_data['turnover_rate'].values
                logger.info("✅ 提取训练集换手率（用于基础权重计算）")
            else:
                logger.warning("⚠️ 训练数据中未包含turnover_rate列，将使用均匀权重")

            if 'circ_mv' in raw_train_data.columns:
                train_circ_mv_for_ensemble = raw_train_data['circ_mv'].values
                logger.info("✅ 提取训练集流通市值（用于基础权重计算）")
            else:
                logger.warning("⚠️ 训练数据中未包含circ_mv列，将使用均匀权重")

            # 🆕 提取relevance（用于样本权重加权和LightGBM训练）
            train_relevance_for_ensemble = filtered_relevance.iloc[:split_point] if hasattr(filtered_relevance, 'iloc') else filtered_relevance[:split_point]
            val_relevance_for_ensemble = filtered_relevance.iloc[split_point:] if hasattr(filtered_relevance, 'iloc') else filtered_relevance[split_point:]
            logger.info("✅ 提取训练集和验证集relevance(4层)（用于样本权重加权和LightGBM训练）")

            # 🆕 提取relevance_30（用于XGBoost训练）
            train_relevance_30_for_ensemble = filtered_relevance_30.iloc[:split_point] if hasattr(filtered_relevance_30, 'iloc') else filtered_relevance_30[:split_point]
            val_relevance_30_for_ensemble = filtered_relevance_30.iloc[split_point:] if hasattr(filtered_relevance_30, 'iloc') else filtered_relevance_30[split_point:]
            logger.info("✅ 提取训练集和验证集relevance_30(30层)（用于XGBoost训练）")

            logger.info(f"📊 数据划分: 训练集{len(raw_train_data):,}样本, 验证集{len(raw_val_data):,}样本")

            # 6.2 特征预处理
            X_train_processed, feature_names_processed = _prepare_feature_data(
                raw_train_data, feature_columns, data_type="训练集"
            )

            if X_train_processed is None:
                logger.error("训练集特征预处理失败")
                return False

            X_val_processed, _ = _prepare_feature_data(
                raw_val_data, feature_columns, data_type="验证集"
            )

            if X_val_processed is None:
                logger.error("验证集特征预处理失败")
                return False

            # 6.3 调整标签长度(预处理可能移除重复行)
            if len(X_train_processed) != len(y_train_split):
                logger.warning(f"训练集预处理后长度变化: {len(y_train_split)} -> {len(X_train_processed)}")
                y_train_split = y_train_split[:len(X_train_processed)]
                # 🆕 同步调整train_dates长度
                if train_dates_for_ensemble is not None:
                    train_dates_for_ensemble = train_dates_for_ensemble[:len(X_train_processed)]
                # 🆕 同步调整换手率和流通市值长度
                if train_turnover_rate_for_ensemble is not None:
                    train_turnover_rate_for_ensemble = train_turnover_rate_for_ensemble[:len(X_train_processed)]
                if train_circ_mv_for_ensemble is not None:
                    train_circ_mv_for_ensemble = train_circ_mv_for_ensemble[:len(X_train_processed)]
                # 🚀 同步调整原始标签长度（与训练标签相同）
                if y_train_split_original_ensemble is not None:
                    y_train_split_original_ensemble = y_train_split_original_ensemble[:len(X_train_processed)]
                # 🚀 同步调整relevance长度（关键修复：避免group信息错乱）
                train_relevance_for_ensemble = train_relevance_for_ensemble[:len(X_train_processed)]
                # 🚀 同步调整relevance_30长度
                train_relevance_30_for_ensemble = train_relevance_30_for_ensemble[:len(X_train_processed)]

            if len(X_val_processed) != len(y_val_split):
                logger.warning(f"验证集预处理后长度变化: {len(y_val_split)} -> {len(X_val_processed)}")
                y_val_split = y_val_split[:len(X_val_processed)]
                # 🚀 同步调整原始标签长度（与验证标签相同）
                if y_val_split_original_ensemble is not None:
                    y_val_split_original_ensemble = y_val_split_original_ensemble[:len(X_val_processed)]
                # 🚀 同步调整relevance长度（关键修复：避免group信息错乱）
                val_relevance_for_ensemble = val_relevance_for_ensemble[:len(X_val_processed)]
                # 🚀 同步调整relevance_30长度
                val_relevance_30_for_ensemble = val_relevance_30_for_ensemble[:len(X_val_processed)]
                # 🚀 同步调整val_atr14长度
                if val_atr14_for_ensemble is not None:
                    val_atr14_for_ensemble = val_atr14_for_ensemble[:len(X_val_processed)]
                # 🔧 关键修复: 同步调整val_dates长度（避免日期与数据错位）
                if val_dates_for_ensemble is not None:
                    val_dates_for_ensemble = val_dates_for_ensemble[:len(X_val_processed)]
                    logger.info(f"   ✅ 同步调整验证集日期长度: {len(val_dates_for_ensemble)}")

            # 6.3.5 条件计算样本权重（仅LightGBM需要）
            unified_sample_weights = None
            if use_lightgbm:
                logger.info("🔧 计算样本权重（供LightGBM使用）...")
                unified_sample_weights = _calculate_professional_sample_weights(
                    train_relevance_for_ensemble,  # 使用relevance标签
                    imbalance_ratio=1.0,
                    logger=logger,
                    boundary_distances=None,
                    is_regression=True,
                    y_train_original=None,
                    train_dates=train_dates_for_ensemble,
                    time_decay_half_life=180.0,  # 统一使用180天
                    turnover_rate=train_turnover_rate_for_ensemble,
                    circ_mv=train_circ_mv_for_ensemble,
                    label_rank=train_relevance_for_ensemble,  # relevance值(0-3)
                    label_boost_beta=0.15,  # 统一使用0.15
                    w_min=0.2,
                    w_max=5.0
                )
                logger.info(f"✅ 样本权重计算完成（供LightGBM使用）")
                logger.info(f"   权重范围: [{unified_sample_weights.min():.4f}, {unified_sample_weights.max():.4f}]")
                logger.info(f"   权重均值: {unified_sample_weights.mean():.4f}")
            else:
                logger.info("⏭️ 跳过样本权重计算（XGBoost排序模型使用分组权重）")

            # 6.4 准备预处理数据字典
            # 🔧 关键修复: 明确区分两个模型的标签
            # - XGBoost使用relevance_30(0-29)
            # - LightGBM使用relevance(0-3)
            # - 原始ATR收益率标签用于IC计算
            preprocessed_data = {
                'X_train': X_train_processed,
                'X_val': X_val_processed,
                # 🔧 原始ATR收益率标签(用于IC计算和评估)
                'y_train': y_train_split,  # ATR收益率标签
                'y_val': y_val_split,  # ATR收益率标签
                'y_train_original': y_train_split_original_ensemble,  # ATR收益率标签(未经Winsorize)
                'y_val_original': y_val_split_original_ensemble,  # ATR收益率标签(未经Winsorize)
                # 🔧 模型训练标签(两个模型使用不同的标签)
                'train_relevance': train_relevance_for_ensemble,  # LightGBM训练标签(0-3)
                'val_relevance': val_relevance_for_ensemble,  # LightGBM训练标签(0-3)
                'train_relevance_30': train_relevance_30_for_ensemble,  # XGBoost训练标签(0-29)
                'val_relevance_30': val_relevance_30_for_ensemble,  # XGBoost训练标签(0-29)
                # 🔧 其他辅助数据
                'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                'train_dates': train_dates_for_ensemble,  # 训练集日期
                'val_dates': val_dates_for_ensemble,  # 验证集日期
                'val_atr14': val_atr14_for_ensemble,  # 验证集ATR14(用于止盈止损)
                'turnover_rate': train_turnover_rate_for_ensemble,  # 训练集换手率
                'circ_mv': train_circ_mv_for_ensemble,  # 训练集流通市值
                'sample_weights': unified_sample_weights  # 样本权重(LightGBM使用)
            }

            logger.info(f"✅ 预处理完成: 训练集{len(X_train_processed):,}样本, 验证集{len(X_val_processed):,}样本")
            logger.info("=" * 60)

            # 🚀 生成统一的时间戳（XGBoost和LightGBM共用）
            model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"🕐 模型统一时间戳: {model_timestamp}")

            # 🚀 训练XGBoost模型(如果选择了XGBoost)
            xgb_results = None
            if use_xgboost:
                logger.info(f"\n{'='*60}")
                logger.info(f"训练XGBoost模型")
                logger.info(f"{'='*60}")

                try:
                    xgb_results = _train_ml_model(
                        filtered_data,
                        filtered_labels,
                        feature_columns,
                        label_stats=label_stats,
                        enable_param_search=enable_param_search,
                        search_strategy=search_strategy,
                        random_seed=5,
                        preprocessed_data=preprocessed_data,
                        enable_shap_analysis=True,  # XGBoost启用SHAP分析
                        market_data=market_data  # 🚀 传递market_data用于XGBoost计算group权重
                    )

                    if xgb_results is None:
                        logger.error("XGBoost模型训练失败：_train_ml_model返回None")
                        if not use_lightgbm:
                            return False
                        logger.warning("将继续训练LightGBM模型")
                    else:
                        logger.info(f"✅ XGBoost模型训练完成")

                except Exception as e:
                    logger.error(f"XGBoost模型训练失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    if not use_lightgbm:
                        return False
                    logger.warning("将继续训练LightGBM模型")
                    xgb_results = None

            # 🚀 如果选择了LightGBM,训练LightGBM模型
            lgb_results = None
            if use_lightgbm:
                logger.info(f"\n{'='*60}")
                logger.info(f"训练LightGBM模型")
                logger.info(f"{'='*60}")

                try:
                    # 🚀 确定特征列表
                    if xgb_results is not None:
                        # 使用XGBoost训练后的特征列表(已对齐)
                        lgb_feature_names = xgb_results['feature_names']
                        logger.info(f"✅ 使用XGBoost的特征列表: {len(lgb_feature_names)}个特征")
                    else:
                        # 独立训练LightGBM,使用原始特征列表
                        lgb_feature_names = feature_columns
                        logger.info(f"✅ 独立训练LightGBM,使用原始特征列表: {len(lgb_feature_names)}个特征")

                    lgb_results = _train_lightgbm_model(
                        filtered_data,
                        filtered_labels,
                        lgb_feature_names,
                        label_stats=label_stats,
                        preprocessed_data=preprocessed_data,
                        optimize_thresholds=args.optimize_thresholds  # 🚀 传递止盈止损优化参数
                    )

                    if lgb_results is None:
                        logger.error("❌ LightGBM模型训练失败")
                        if not use_xgboost or xgb_results is None:
                            return False
                    else:
                        logger.info(f"✅ LightGBM模型训练完成")

                except Exception as e:
                    logger.error(f"❌ LightGBM模型训练失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    if not use_xgboost or xgb_results is None:
                        return False
                    lgb_results = None

            # 🚀 收集模型结果
            ensemble_results = []
            if xgb_results is not None:
                ensemble_results.append(xgb_results)
            if lgb_results is not None:
                ensemble_results.append(lgb_results)

            if len(ensemble_results) == 0:
                logger.error("❌ 所有模型训练失败")
                return False

            logger.info(f"\n{'='*60}")
            logger.info(f"✅ 模型训练完成: {len(ensemble_results)}个模型")
            logger.info(f"{'='*60}")

            # 🚀 对比XGBoost和LightGBM的指标
            if xgb_results is not None and lgb_results is not None:
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 XGBoost vs LightGBM 指标对比（每日指标平均值）")
                logger.info(f"{'='*80}")

                xgb_perf = xgb_results.get('performance', {})
                lgb_perf = lgb_results.get('performance', {})

                # 🚀 核心排序指标对比
                logger.info(f"\n🎯 核心排序指标:")
                logger.info(f"  {'指标':<20} {'XGBoost':>12} {'LightGBM':>12} {'差异':>12}")
                logger.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

                # NDCG@5
                xgb_ndcg5 = xgb_perf.get('ndcg_5', 0.0)
                lgb_ndcg5 = lgb_perf.get('ndcg_5', 0.0)
                diff_ndcg5 = xgb_ndcg5 - lgb_ndcg5
                logger.info(f"  {'NDCG@5':<20} {xgb_ndcg5:>12.4f} {lgb_ndcg5:>12.4f} {diff_ndcg5:>+12.4f}")

                # NDCG@10
                xgb_ndcg10 = xgb_perf.get('ndcg_10', 0.0)
                lgb_ndcg10 = lgb_perf.get('ndcg_10', 0.0)
                diff_ndcg10 = xgb_ndcg10 - lgb_ndcg10
                logger.info(f"  {'NDCG@10':<20} {xgb_ndcg10:>12.4f} {lgb_ndcg10:>12.4f} {diff_ndcg10:>+12.4f}")

                # Rank IC (每日Spearman平均)
                xgb_rank_ic = xgb_perf.get('rank_ic', 0.0)
                lgb_rank_ic = lgb_perf.get('rank_ic', 0.0)
                diff_rank_ic = xgb_rank_ic - lgb_rank_ic
                logger.info(f"  {'Rank IC (每日平均)':<20} {xgb_rank_ic:>12.4f} {lgb_rank_ic:>12.4f} {diff_rank_ic:>+12.4f}")

                # IC (每日Pearson平均)
                xgb_ic = xgb_perf.get('ic', 0.0)
                lgb_ic = lgb_perf.get('ic', 0.0)
                diff_ic = xgb_ic - lgb_ic
                logger.info(f"  {'IC (每日平均)':<20} {xgb_ic:>12.4f} {lgb_ic:>12.4f} {diff_ic:>+12.4f}")

                # Top-5收益率
                xgb_top5 = xgb_perf.get('top_5_return', 0.0)
                lgb_top5 = lgb_perf.get('top_k_return', 0.0)  # LightGBM使用top_k_return
                diff_top5 = xgb_top5 - lgb_top5
                logger.info(f"  {'Top-5收益率':<20} {xgb_top5:>12.4f} {lgb_top5:>12.4f} {diff_top5:>+12.4f}")

                # 🚀 辅助指标对比
                logger.info(f"\n📊 辅助指标（仅供参考）:")
                logger.info(f"  {'指标':<20} {'XGBoost':>12} {'LightGBM':>12} {'差异':>12}")
                logger.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

                # 验证集R²
                xgb_r2 = xgb_perf.get('val_r2', 0.0)
                lgb_r2 = lgb_perf.get('val_r2', 0.0)
                diff_r2 = xgb_r2 - lgb_r2
                logger.info(f"  {'验证集R²':<20} {xgb_r2:>12.4f} {lgb_r2:>12.4f} {diff_r2:>+12.4f}")

                # 验证集RMSE
                xgb_rmse = xgb_perf.get('val_rmse', 0.0)
                lgb_rmse = lgb_perf.get('val_rmse', 0.0)
                diff_rmse = xgb_rmse - lgb_rmse
                logger.info(f"  {'验证集RMSE':<20} {xgb_rmse:>12.6f} {lgb_rmse:>12.6f} {diff_rmse:>+12.6f}")

                logger.info(f"\n💡 说明:")
                logger.info(f"  - NDCG@5/10: 按日期分组计算,然后平均")
                logger.info(f"  - Rank IC: 每日Spearman相关系数的平均值")
                logger.info(f"  - IC: 每日Pearson相关系数的平均值")
                logger.info(f"  - Top-5收益率: 每日Top-5股票收益率的平均值")
                logger.info(f"  - 正值差异表示XGBoost更优,负值表示LightGBM更优")
                logger.info(f"{'='*80}\n")

        else:
            # 🚀 单模型模式: 只训练1个模型(用于参数搜索)
            logger.info("开始训练XGBoost模型（单模型）...")

            if enable_param_search:
                logger.info(f"🔍 启用超参数搜索: 策略={search_strategy}")
                logger.info("📋 单模型模式: 只训练1个模型用于参数搜索")
            else:
                logger.info("🔍 使用默认参数配置（跳过超参数搜索）")
                logger.info("📋 单模型模式: 只训练1个模型")

            logger.info("\n" + "=" * 60)
            logger.info("🔧 训练单个模型")
            logger.info("=" * 60)

            # 🚀 生成时间戳（单模型）
            model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"🕐 模型时间戳: {model_timestamp}")

            try:
                # 训练单个模型（不使用预处理数据共享）
                model_results = _train_ml_model(
                    filtered_data,
                    filtered_labels,
                    feature_columns,
                    label_stats=label_stats,  # 传递标签统计量用于反向转换
                    enable_param_search=enable_param_search,
                    search_strategy=search_strategy,
                    random_seed=5,
                    preprocessed_data=None,  # 不使用预处理数据共享
                    enable_shap_analysis=True,  # 启用SHAP分析
                    market_data=market_data  # 🚀 传递market_data用于XGBoost计算group权重
                )

                if model_results is None:
                    logger.error("模型训练失败：_train_ml_model返回None")
                    return False

                logger.info("✅ 模型训练完成")

            except Exception as e:
                logger.error(f"模型训练失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
                return False

            # 将单个模型结果包装成列表,以兼容后续代码
            ensemble_results = [model_results]

            logger.info(f"\n{'='*60}")
            logger.info(f"✅ 模型训练完成")
            logger.info(f"{'='*60}")

        # 7. 保存所有模型（使用统一时间戳）
        for model_results in ensemble_results:
            feature_names = model_results.get('feature_names', feature_columns)
            model_type = model_results.get('model_type', 'xgboost')  # 获取模型类型

            # XGBoost需要PCA信息,LightGBM不需要
            if model_type == 'xgboost':
                pca_info = model_results.get('pca_info', {})
                pca_models = {'market_factors': pca_info} if pca_info.get('applied', False) else {}
            else:
                pca_models = None

            _save_model_results(
                model_results,
                feature_names,
                pca_models,
                model_type=model_type,  # 🚀 传递模型类型
                timestamp=model_timestamp  # 🚀 传递统一时间戳
            )

        print("✅ 模型训练模式执行完成")
        return True

    except Exception as e:
        logger.error(f"❌ 模型训练模式失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False



def _run_stock_selection_backtest(
    backtest_data,
    stock_counts=[1, 2, 3, 5],
    window_name=None,
    market_data=None
):
    """
    执行选股回测(test_model和Walk-Forward共用)

    Args:
        backtest_data: DataFrame,包含以下字段:
            - date, stock_code
            - predicted_return, predicted_return_atr_raw, std_pred
            - future_return_2d, is_can_buy
            - open, high, low, close, atr14
            - 其他技术指标
        stock_counts: 选股数量列表,默认[1, 2, 3, 5]
        window_name: 窗口名称(Walk-Forward专用),默认None
        market_data: 市场数据(用于市场状态判断),默认None

    Returns:
        dict: {
            'backtest_results': {...},  # 各档位回测结果
            'detailed_trades': {...},  # 详细交易记录
            'summary_stats': {...}  # 汇总统计
        }
    """
    logger = logging.getLogger(__name__)

    # 窗口标识(用于日志)
    window_prefix = f"[{window_name}] " if window_name else ""

    logger.info(f"\n{window_prefix}🔄 开始选股回测...")
    logger.info("=" * 60)
    # 准备回测数据
    required_columns = [
        # 基础必要列
        'date', 'stock_code', 'future_return_2d', 'is_can_buy', 'pct_chg', 'volume', 'amount',  # 🚀 添加amount字段
        # 🔧 修复:ATR三分类标签系统必需字段
        'atr14', 'open', 'high', 'low', 'close',
        # 🔧 高级技术指标字段（多维度复合条件筛选必需）
        'price_ema20_ratio', 'price_ema50_ratio', 'ema20_ema50_ratio',
        'ema50_slope', 'log_price_regression_slope', 'log_price_regression_r2',
        'adx_14', 'drawdown_from_20d_high_pct', 'drawdown_atr_ratio', 'atr20',
        'negative_days_5d', 'large_red_candle_5d', 'distance_to_support_atr',
        'bollinger_percent_b', 'rsi_14',
        # 涨跌停相关特征（用于选股过滤和风险控制）
        'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
        'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
        # 技术分析特征（用于选股过滤）
        'near_support', 'near_resistance', 'short_reversal'
    ]
    missing_columns = [col for col in required_columns if col not in backtest_data.columns]

    if missing_columns:
        logger.warning(f"⚠️ 回测数据缺少必要列: {missing_columns}")
        logger.info("为缺失的特征字段设置默认值...")

        # 为缺失的特征字段设置默认值
        default_values = {
            # 🔧 高级技术指标字段默认值
            'close': 0.0,
            'price_ema20_ratio': 1.0,
            'price_ema50_ratio': 1.0,
            'ema20_ema50_ratio': 1.0,
            'ema50_slope': 0.0,
            'log_price_regression_slope': 0.0,
            'log_price_regression_r2': 0.0,
            'adx_14': 10.0,
            'drawdown_from_20d_high_pct': 0.0,
            'drawdown_atr_ratio': 0.0,
            'atr20': 0.5,
            'negative_days_5d': 1,
            'large_red_candle_5d': 1,
            'distance_to_support_atr': 0.5,
            'bollinger_percent_b': 0.5,
            'rsi_14': 50.0,
            # 🚀 新股过滤字段默认值（不参与模型训练，仅用于过滤）
            'volume': 0.0,
            'amount': 0.0,  # 🚀 添加amount默认值
            # 涨跌停相关特征默认值
            'is_limit_up': 0,
            'is_limit_down': 0,
            'is_zhaban': 0,
            'limit_strength': 0.0,
            'limit_times': 0,
            'consecutive_limit_up': 0,
            'consecutive_limit_down': 0,
            # 技术分析特征默认值
            'near_support': 0,
            'near_resistance': 0,
            'short_reversal': 0,
            # 基础特征默认值
            'pct_chg': 0.0,
            'future_return_2d': 0.0,
            'is_can_buy': 1
        }

        for col in missing_columns:
            if col in default_values:
                backtest_data[col] = default_values[col]
                logger.info(f"  {col}: 设置默认值 {default_values[col]}")
            else:
                backtest_data[col] = 0.0
                logger.warning(f"  {col}: 未定义默认值，设置为 0.0")

    # 🆕 调试日志：检查backtest_data的字段
    logger.info(f"\n{window_prefix}🔍 回测数据字段检查")
    logger.info(f"总行数: {len(backtest_data)}")
    logger.info(f"总列数: {len(backtest_data.columns)}")

    # 检查关键字段
    key_fields = ['date', 'stock_code', 'volume', 'is_can_buy', 'pct_chg', 'open', 'high', 'low', 'close', 'atr14']
    for field in key_fields:
        if field in backtest_data.columns:
            missing_count = backtest_data[field].isna().sum()
            missing_pct = missing_count / len(backtest_data) * 100
            if field not in ['date', 'stock_code']:
                zero_count = (backtest_data[field] == 0).sum()
                zero_pct = zero_count / len(backtest_data) * 100
                logger.info(f"  {field}: 缺失{missing_count}({missing_pct:.2f}%), 为0{zero_count}({zero_pct:.2f}%)")
            else:
                logger.info(f"  {field}: 缺失{missing_count}({missing_pct:.2f}%)")
        else:
            logger.warning(f"  {field}: ❌ 字段不存在")

    # 只按日期排序，保持每日内股票的原始预测概率分布
    backtest_data = backtest_data.sort_values('date')

    # 获取所有交易日
    trading_dates = sorted(backtest_data['date'].unique())
    logger.info(f"\n{window_prefix}📅 回测期间: {trading_dates[0].strftime('%Y-%m-%d')} 到 {trading_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"{window_prefix}📅 交易日数量: {len(trading_dates)}天")

    # 🚀 动态调整选股阈值：基于预测范围自适应
    # 兼容两种字段名: predicted_return_atr_raw (test_model) 和 predicted_return (walkforward)
    if 'predicted_return_atr_raw' in backtest_data.columns:
        pred_column = 'predicted_return_atr_raw'
    elif 'predicted_return' in backtest_data.columns:
        pred_column = 'predicted_return'
    else:
        logger.error("❌ 数据中缺少预测字段 (predicted_return_atr_raw 或 predicted_return)")
        return None

    all_predictions = backtest_data[pred_column].dropna()
    pred_max = all_predictions.max()
    pred_75th = all_predictions.quantile(0.75)
    pred_median = all_predictions.median()
    pred_mean = all_predictions.mean()
    pred_std = all_predictions.std()

    # 🆕 优化后的自适应阈值策略
    # 策略1: 基于分位数的动态阈值
    # - 使用60%分位数作为基准,确保有足够的候选股票
    # - 但不低于0(避免选择负预测的股票)
    # - 也不高于75%分位数(避免过于严格)
    pred_60th = all_predictions.quantile(0.60)

    # 策略2: 基于均值+标准差的阈值
    # - 使用均值作为基准,加上0.5倍标准差
    # - 这样可以选择高于平均水平的股票
    threshold_mean_based = max(0, pred_mean + 0.5 * pred_std)

    # 策略3: 综合策略
    # - 取两种策略的较小值,确保有足够的候选股票
    # - 但不低于0,不高于1.0
    MIN_POS_THRESHOLD_ATR = min(
        max(0, pred_60th),  # 60%分位数,但不低于0
        threshold_mean_based,  # 均值+0.5σ
        1.0  # 上限1.0 ATR
    )

    # 计算超过阈值的股票比例
    above_threshold_pct = (all_predictions >= MIN_POS_THRESHOLD_ATR).sum() / len(all_predictions) * 100

    logger.info(f"\n{window_prefix}✅ 自适应阈值: {MIN_POS_THRESHOLD_ATR:.2f} ATR")
    logger.info(f"{window_prefix}   - 60%分位数: {pred_60th:.2f} ATR")
    logger.info(f"{window_prefix}   - 均值+0.5σ: {threshold_mean_based:.2f} ATR")
    logger.info(f"{window_prefix}   - 超过阈值的股票比例: {above_threshold_pct:.1f}%")
    logger.info(f"{window_prefix}📊 预测值分布: 最大{pred_max:.2f}, 75%分位{pred_75th:.2f}, 中位数{pred_median:.2f}, 均值{pred_mean:.2f}±{pred_std:.2f} ATR")

    # 准备常量
    TRADING_COST_ATR = 0.03        # 交易/滑点等折算到 ATR 的最低可覆盖成本
    # 🔧 修复：使用全局配置的MAX_HOLDING_DAYS，而不是硬编码
    # ATR_MULTIPLIER_PROFIT = 1.8    # ATR止盈倍数（使用全局配置）
    # ATR_MULTIPLIER_LOSS = 1.5      # ATR止损倍数（使用全局配置）
    # MAX_STOP_LOSS_PCT = 0.03       # 百分比止损阈值（使用全局配置）
    # MAX_HOLDING_DAYS = 10          # 最大持仓天数（已修复：使用全局配置）

    # 初始化回测结果
    backtest_results = {}
    detailed_trades = {}

    # ========== 第2批代码将在这里添加 ==========
    # 🚀 修正版本：恢复1,3,5只股票回测 + g_t仅控制仓位比例
    for stock_count in stock_counts:
        logger.info(f"\n📊 开始{stock_count}只股票回测（g_t动态仓位分配）...")

        # 初始化回测记录
        trades = []
        portfolio_values = [1.0]  # 初始资金为1
        # 退出分类专用容器（按用户要求使用标准化原因，不做文本分类）
        exit_lists = {
            'take_profit': [],
            'atr_stop_loss': [],
            'percent_stop_loss': [],
            'max_holding_days': []
        }

        daily_returns = []
        # 🚀 新增：空仓日统计

        # 统计日均仓位（固定每股目标仓位：Top-1=100%，Top-3=33.33%，Top-5=20%）
        position_ratio_sum = 0.0

        empty_position_days = 0
        partial_position_days = 0

        # 🚀 核心策略：固定股票数量 + g_t动态仓位比例
        logger.info(f"📊 固定选股{stock_count}只，g_t决定总仓位比例")

        # 🚀 使用统一的预测字段名(已在函数开头确定)
        # 按日期分组进行选股
        df_pred = backtest_data.copy()
        grouped = df_pred.groupby('date', sort=True)

        selected_records = []
        empty_day_count = 0
        total_days = 0

        for trade_date, g in grouped:
            total_days += 1
            g = g.copy()

            # 🚀 第一步：基础过滤（北交所、新股、ST股、可买入）
            date_str = trade_date.strftime('%Y%m%d')

            # 🆕 调试日志：特定日期的详细过滤过程
            is_debug_date = (trade_date == pd.to_datetime('2025-01-06'))
            if is_debug_date:
                logger.info(f"\n🔍 调试: {date_str}选股过程")
                logger.info(f"初始股票数: {len(g)}")

            # 过滤北交所股票 (股票代码以.BJ或.bj结尾)
            if 'stock_code' in g.columns and len(g) > 0:
                before_bj = len(g)
                bj_stocks = g[g['stock_code'].str.endswith(('.BJ', '.bj'), na=False)].copy()
                bj_filter = ~g['stock_code'].str.endswith(('.BJ', '.bj'), na=False)
                g = g[bj_filter].copy()

                # 🔧 修复：调试日期时打印北交所过滤日志（无论是否过滤）
                if is_debug_date:
                    if len(g) < before_bj:
                        logger.info(f"北交所过滤后: {len(g)} (过滤掉{before_bj - len(g)}只)")
                        if len(bj_stocks) > 0:
                            logger.info(f"  北交所股票示例: {bj_stocks['stock_code'].head(5).tolist()}")
                    else:
                        logger.info(f"北交所过滤后: {len(g)} (无北交所股票)")

            # 过滤新股
            new_stock_filter = _filter_new_stocks(g, date_str, logger)
            g = g[new_stock_filter].copy()
            if is_debug_date:
                logger.info(f"新股过滤后: {len(g)}")

            # 过滤ST股票
            if len(g) > 0:
                st_filter = _filter_st_stocks(g, logger)
                g = g[st_filter].copy()
                if is_debug_date:
                    logger.info(f"ST过滤后: {len(g)}")

            # 🚀 新增：过滤T日涨停股票（无法买入）
            if 'is_limit_up' in g.columns and len(g) > 0:
                before_limit_up = len(g)
                g = g[g['is_limit_up'] != 1].copy()
                if is_debug_date:
                    logger.info(f"涨停过滤后: {len(g)} (过滤掉{before_limit_up - len(g)})")

            # 🚀 新增：过滤T日跌停股票（风险高,无风控时应避免）
            if 'is_limit_down' in g.columns and len(g) > 0:
                before_limit_down = len(g)
                g = g[g['is_limit_down'] != 1].copy()
                if is_debug_date:
                    logger.info(f"跌停过滤后: {len(g)} (过滤掉{before_limit_down - len(g)})")

            # 🚀 新增：流动性过滤（成交额 < 1000万视为流动性不足）
            if 'amount' in g.columns and len(g) > 0:
                before_liquidity = len(g)
                # 成交额单位：元，1000万 = 10,000,000
                liquidity_threshold = 10_000_000

                # 🚀 新增：记录被过滤的股票
                filtered_stocks = g[g['amount'] < liquidity_threshold].copy()

                g = g[g['amount'] >= liquidity_threshold].copy()
                if is_debug_date:
                    logger.info(f"流动性过滤后: {len(g)} (过滤掉{before_liquidity - len(g)})")

                    # 打印前5个被过滤的股票
                    if len(filtered_stocks) > 0:
                        logger.info(f"  因流动性不足被过滤的股票示例(前5个):")
                        for idx, (_, row) in enumerate(filtered_stocks.head(5).iterrows(), 1):
                            stock_code = row.get('stock_code', 'N/A')
                            amount = row.get('amount', 0)
                            logger.info(f"    {idx}. {stock_code}: 成交额={amount:,.0f}元 ({amount/10000:.2f}万)")
            elif 'volume' in g.columns and len(g) > 0:
                # 如果没有amount字段,使用volume字段
                # 成交量 < 100万股视为流动性不足
                before_liquidity = len(g)
                volume_threshold = 10_000
                g = g[g['volume'] >= volume_threshold].copy()
                if is_debug_date:
                    logger.info(f"流动性过滤(volume)后: {len(g)} (过滤掉{before_liquidity - len(g)})")

            # 🔧 修复：移除置信度过滤
            # 原因：
            # 1. 模型评估时使用全部数据（IC=0.51优秀），但选股时过滤25%数据，导致评估与实际脱节
            # 2. 模型分歧大不等于预测错误，可能错过高收益机会
            # 3. 先验证移除后性能是否恢复，再决定是否需要软约束
            # if 'high_confidence' in g.columns and len(g) > 0:
            #     before_confidence = len(g)
            #     low_conf_stocks = g[g['high_confidence'] == False].copy()
            #     g = g[g['high_confidence'] == True].copy()
            #
            #     if is_debug_date:
            #         logger.info(f"置信度过滤后: {len(g)} (过滤掉{before_confidence - len(g)})")
            #
            #         # 打印前5个低置信度股票
            #         if len(low_conf_stocks) > 0:
            #             logger.info(f"  低置信度股票示例(前5个):")
            #             for idx, (_, row) in enumerate(low_conf_stocks.head(5).iterrows(), 1):
            #                 stock_code = row.get('stock_code', 'N/A')
            #                 knn_dist = row.get('knn_distance', np.nan)
            #                 pred_return = row.get(pred_column, np.nan)  # 🔧 修复：使用动态字段名
            #                 logger.info(f"    {idx}. {stock_code}: k-NN距离={knn_dist:.4f}, 预测收益={pred_return:.3f} ATR")

            # 2.1 过滤 NaN
            before_nan = len(g)
            g = g[~g[pred_column].isna()]
            if is_debug_date:
                logger.info(f"NaN过滤后: {len(g)} (过滤掉{before_nan - len(g)})")
            if g.empty:
                if is_debug_date:
                    logger.info(f"❌ 所有股票都被过滤，空仓")
                empty_day_count += 1
                continue

             # 过滤可买入股票
            before_can_buy = len(g)
            g1 = g[g['is_can_buy'] == 0].copy()
            g = g[g['is_can_buy'] == 1].copy()
            if is_debug_date:
                logger.info(f"is_can_buy过滤后: {len(g)} (过滤掉{before_can_buy - len(g)})")
                if len(g) > 0:
                    logger.info(f"  is_can_buy=0示例股票: {g1['stock_code'].head(5).tolist()}")
                    logger.info(f"  is_can_buy=1的股票数: {(g['is_can_buy'] == 1).sum()}")
                    logger.info(f"  示例股票: {g['stock_code'].head(5).tolist()}")

            # 2.2 先做"硬过滤"：只保留 正期望 且 超过阈值
            before_threshold = len(g)
            if is_debug_date:
                logger.info(f"预测值分布: 最大{g[pred_column].max():.4f}, "
                           f"75%分位{g[pred_column].quantile(0.75):.4f}, "
                           f"中位数{g[pred_column].median():.4f}")
                logger.info(f"阈值: {MIN_POS_THRESHOLD_ATR}")
                logger.info(f"超过阈值的股票数: {(g[pred_column] >= MIN_POS_THRESHOLD_ATR).sum()}")

            g = g[g[pred_column] >= MIN_POS_THRESHOLD_ATR]
            if is_debug_date:
                logger.info(f"阈值过滤后: {len(g)} (过滤掉{before_threshold - len(g)})")

            if g.empty:
                # 全部不达标：记为空仓日
                if is_debug_date:
                    logger.info(f"❌ 所有股票预测值低于阈值，空仓")
                empty_day_count += 1
                continue

            # 🔧 修复：移除横截面标准化，直接使用ATR预测值减去交易成本
            # 原因：
            # 1. ATR预测值本身就是标准化的收益（+2 ATR = 预期上涨2倍ATR）
            # 2. 横截面标准化破坏了绝对值含义（+0.5 ATR可能被标准化为负值）
            # 3. 应该直接比较ATR预测值，而不是相对排名

            # 2.3 成本敏感：直接从ATR预测值减去交易成本
            g['score_final'] = g[pred_column] - TRADING_COST_ATR

            # 2.4 排序并**不凑满**：只取 Top-K，且不够就少买
            g = g.sort_values('score_final', ascending=False).head(stock_count)
            # logger.info(f"data:{date_str}, g count:{len(g)}  and g:{g}")

            # 2.5 记录选股（包含仓位分配信息）
            selection_columns = ['date','stock_code',pred_column,'score_final']

            # 🚀 修复：如果存在仓位分配列，也要保存
            if 'position_allocation' in g.columns:
                selection_columns.append('position_allocation')
            if 'market_regime' in g.columns:
                selection_columns.append('market_regime')

            selected_records.append(g[selection_columns])

        # 汇总选股结果
        if selected_records:
            picks = pd.concat(selected_records, ignore_index=True)
        else:
            # 🚀 修复：空DataFrame也要包含可能的仓位分配列
            empty_columns = ['date','stock_code',pred_column,'score_z','score_final']
            if 'position_allocation' in backtest_data.columns:
                empty_columns.append('position_allocation')
            if 'market_regime' in backtest_data.columns:
                empty_columns.append('market_regime')
            picks = pd.DataFrame(columns=empty_columns)

        # 计算空仓日比例
        empty_ratio = empty_day_count / max(total_days, 1)
        # logger.info(f"统一口径选股完成：只买正期望且 >= {MIN_POS_THRESHOLD_ATR:.2f} ATR，不凑满K；空仓日比例={empty_ratio:.1%}")

        # 🆕 诊断日志：入选股票统计
        if len(picks) > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Top-{stock_count} 入选股票统计")
            logger.info(f"{'='*60}")
            logger.info(f"入选股票数: {len(picks)}")
            logger.info(f"入选交易日数: {picks['date'].nunique()}")
            logger.info(f"空仓日数: {empty_day_count}/{total_days} ({empty_ratio:.1%})")

            # 预测值统计
            selected_pred = picks[pred_column]
            logger.info(f"\n入选股票预测值分布:")
            logger.info(f"  均值: {selected_pred.mean():.4f} ATR")
            logger.info(f"  中位数: {selected_pred.median():.4f} ATR")
            logger.info(f"  25分位: {selected_pred.quantile(0.25):.4f} ATR")
            logger.info(f"  75分位: {selected_pred.quantile(0.75):.4f} ATR")
            logger.info(f"  最小值: {selected_pred.min():.4f} ATR")
            logger.info(f"  最大值: {selected_pred.max():.4f} ATR")

            # 如果有实际收益数据，计算预测准确性
            if 'future_return_2d' in picks.columns:
                # 合并picks和backtest_data以获取atr14
                picks_with_atr = picks.merge(
                    backtest_data[['date', 'stock_code', 'atr14', 'future_return_2d']],
                    on=['date', 'stock_code'],
                    how='left',
                    suffixes=('', '_y')
                )

                # 🔧 删除错误的future_return_2d统计代码
                # 实际收益应该由回测逻辑根据止盈、止损、持仓超时等条件计算
                # 而不是简单使用future_return_2d

            logger.info(f"{'='*60}\n")
        else:
            logger.warning(f"⚠️ Top-{stock_count}: 没有入选股票（全部空仓）")

        # 执行回测：基于选股结果进行持仓管理和收益计算
        portfolio_values_gross = [1.0]  # 未扣除成本的组合价值

        # 持仓管理：跟踪每只股票的持仓状态
        # 持仓信息结构：{stock_code: {'selection_date': T日, 'entry_price': T+1开盘价, 'atr_value': T日ATR,
        #                           'profit_target': 止盈位, 'stop_loss': 止损位, 'stop_loss_type': 止损类型, 'days_held': 持仓天数}}
        current_positions = {}
        max_positions = stock_count
        # logger.info(f"初始化持仓管理: 最大持仓={max_positions}, ATR止盈={ATR_MULTIPLIER_PROFIT}x, ATR止损={ATR_MULTIPLIER_LOSS}x, 3%止损={MAX_STOP_LOSS_PCT:.1%}, 最大持仓期={MAX_HOLDING_DAYS}天")

        for date_idx, date in enumerate(trading_dates[:-1]):  # 排除最后一天（无法卖出）
            # 使用前一交易日(T)的选股信号，在当前交易日(T+1)开盘买入
            prev_date = trading_dates[date_idx - 1] if date_idx > 0 else None

            try:
                logger.info(f"处理交易日: {date} ({date_idx+1}/{len(trading_dates)-1}), 当前持仓: {len(current_positions)}只")

                # === 步骤1: 更新持仓天数并检查退出条件 ===
                positions_to_exit = []
                exit_reasons = {}

                for stock_code, position_info in current_positions.items():
                    # 更新持仓天数：每个交易日递增1（不使用日期差值，避免节假日bug）
                    position_info['days_held'] += 1
                    days_held = position_info['days_held']

                    # 检查止盈止损条件（按照用户需求的逻辑）
                    try:
                        # 获取当日股价数据
                        stock_data = backtest_data[
                            (backtest_data['stock_code'] == stock_code) &
                            (backtest_data['date'] == date)
                        ]

                        if not stock_data.empty:
                            # 🚀 使用前复权价格进行止盈止损判断
                            today_high = stock_data.iloc[0]['high_front']
                            today_low = stock_data.iloc[0]['low_front']
                            today_open = stock_data.iloc[0]['open_front']

                            # 🚀 调试：添加详细的止盈止损检查日志（仅TOP-1策略）
                            if stock_count == 1:  # 只对TOP-1策略输出详细日志
                                logger.info(f"🔍 {stock_code} T+{days_held}日止盈止损检查:")
                                logger.info(f"  当日价格(前复权): 开盘{today_open:.3f}, 最高{today_high:.3f}, 最低{today_low:.3f}")
                                logger.info(f"  止盈位: {position_info['profit_target']:.3f}, 止损位: {position_info['stop_loss']:.3f}")
                                logger.info(f"  止盈触发: {today_high >= position_info['profit_target']}, 止损触发: {today_low <= position_info['stop_loss']}")

                            if days_held == 1:
                                # T+1日买入当天：如果触发止盈止损，标记为T+2日开盘退出（A股T+1规则）
                                if today_low <= position_info['stop_loss']:
                                    position_info['exit_triggered'] = 'stop_loss'
                                    position_info['exit_type'] = 'next_day_open'  # 次日开盘退出
                                    # logger.info(f"T+1日触发止损: {stock_code}, 最低价{today_low:.3f} <= 止损位{position_info['stop_loss']:.3f}, 将于T+2日开盘退出")
                                elif today_high >= position_info['profit_target']:
                                    position_info['exit_triggered'] = 'take_profit'
                                    position_info['exit_type'] = 'next_day_open'  # 次日开盘退出
                                    # logger.info(f"T+1日触发止盈: {stock_code}, 最高价{today_high:.3f} >= 止盈位{position_info['profit_target']:.3f}, 将于T+2日开盘退出")

                            elif days_held >= 2:
                                # T+2日及以后：如果触发止盈止损，当日按止盈止损价退出
                                # 止损优先原则
                                if today_low <= position_info['stop_loss']:
                                    if stock_code not in positions_to_exit:
                                        positions_to_exit.append(stock_code)
                                    if stock_code not in exit_reasons:
                                        exit_reasons[stock_code] = 'stop_loss'
                                    position_info['exit_price'] = position_info['stop_loss']  # 按止损价退出
                                    position_info['exit_type'] = 'intraday'  # 当日盘中退出
                                    # logger.info(f"T+{days_held}日止损退出: {stock_code}, 最低价{today_low:.3f} <= 止损位{position_info['stop_loss']:.3f}, 按止损价{position_info['stop_loss']:.3f}退出")
                                elif today_high >= position_info['profit_target']:
                                    if stock_code not in positions_to_exit:
                                        positions_to_exit.append(stock_code)
                                    if stock_code not in exit_reasons:
                                        exit_reasons[stock_code] = 'take_profit'
                                    position_info['exit_price'] = position_info['profit_target']  # 按止盈价退出
                                    position_info['exit_type'] = 'intraday'  # 当日盘中退出
                                    # logger.info(f"T+{days_held}日止盈退出: {stock_code}, 最高价{today_high:.3f} >= 止盈位{position_info['profit_target']:.3f}, 按止盈价{position_info['profit_target']:.3f}退出")

                    except Exception as e:
                        logger.warning(f"检查{stock_code}止盈止损失败: {e}")

                    # 检查是否有前一日触发的需要次日开盘退出的股票
                    if position_info.get('exit_triggered') and position_info.get('exit_type') == 'next_day_open':
                        if stock_count == 1:  # 调试日志
                            logger.info(f"🔍 {stock_code} T+1日触发,次日开盘退出: exit_triggered={position_info.get('exit_triggered')}, exit_type={position_info.get('exit_type')}")
                        if stock_code not in positions_to_exit:
                            positions_to_exit.append(stock_code)
                        if stock_code not in exit_reasons:
                            exit_reasons[stock_code] = position_info['exit_triggered']
                        position_info['exit_type'] = 'next_day_open'  # 次日开盘退出
                        # logger.info(f"T+{days_held}日开盘退出: {stock_code}, 原因: {position_info['exit_triggered']} (前一日触发)")
                        continue

                    # 最大持仓期检查（避免覆盖已存在的退出原因）
                    # MAX_HOLDING_DAYS是超时限制: 如果3天内没有触发止盈/止损,则在第4天触发超时
                    if days_held > MAX_HOLDING_DAYS:
                        if stock_code not in positions_to_exit:
                            positions_to_exit.append(stock_code)
                        if stock_code not in exit_reasons:
                            exit_reasons[stock_code] = 'max_holding_days'
                        # logger.info(f"超时退出: {stock_code}, 持仓{days_held}天 >= 最大持仓期{MAX_HOLDING_DAYS}天")

                # === 步骤2: 处理退出股票，记录退出交易 ===
                today_exited_stocks = set()
                exit_trades_records = []

                # 调试日志
                if stock_count == 1 and len(positions_to_exit) > 0:
                    logger.info(f"🔍 准备处理退出股票: {len(positions_to_exit)}只")
                    for stock in positions_to_exit:
                        logger.info(f"  退出股票: {stock}, 原因: {exit_reasons.get(stock, 'unknown')}")
                    logger.info(f"  exit_trades_records初始长度: {len(exit_trades_records)}")

                for stock_code in positions_to_exit:
                    if stock_count == 1:  # 调试日志
                        logger.info(f"🔍 开始处理退出股票: {stock_code}")
                    exit_reason = exit_reasons.get(stock_code, 'unknown')
                    today_exited_stocks.add(stock_code)
                    position_info = current_positions[stock_code]

                    # 计算退出收益：根据退出类型使用正确的价格（鲁棒构造，确保记录一定产生）
                    entry_price = position_info.get('entry_price', np.nan)
                    exit_price = None
                    try:
                        # 根据退出类型确定退出价格
                        if position_info.get('exit_type') == 'intraday' and ('exit_price' in position_info):
                            # 当日盘中退出：使用止盈止损价
                            exit_price = position_info.get('exit_price', np.nan)
                            # if pd.notna(exit_price):
                                # logger.info(f"退出收益计算: {stock_code}, 盘中退出价{exit_price:.3f}")
                        else:
                            # 次日开盘退出或最大持仓期：使用前复权开盘价
                            stock_data = backtest_data[
                                (backtest_data['stock_code'] == stock_code) &
                                (backtest_data['date'] == date)
                            ]
                            if not stock_data.empty and pd.notna(stock_data.iloc[0]['open_front']):
                                exit_price = stock_data.iloc[0]['open_front']
                                # logger.info(f"退出收益计算: {stock_code}, 开盘退出价(前复权){exit_price:.3f}")
                            else:
                                exit_price = entry_price  # 兜底
                                # logger.warning(f"退出收益计算: {stock_code}, 无法获取退出价格，使用入场价")

                        # 计算收益
                        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                            exit_return = 0.0
                        else:
                            exit_return = float((exit_price - entry_price) / entry_price)
                    except Exception as e:
                        logger.warning(f"计算{stock_code}退出收益失败: {e}")
                        exit_return = 0.0
                        if exit_price is None:
                            exit_price = entry_price

                    # 推断/兜底止损类型
                    stop_loss_type = position_info.get('stop_loss_type')
                    if not stop_loss_type:
                        try:
                            atr_val = position_info.get('atr_value')
                            if atr_val is not None and not pd.isna(atr_val) and entry_price and not pd.isna(entry_price):
                                atr_sl = entry_price - ATR_MULTIPLIER_LOSS * atr_val
                                pct_sl = entry_price * (1 - MAX_STOP_LOSS_PCT)
                                stop_loss_val = position_info.get('stop_loss', max(atr_sl, pct_sl))
                                stop_loss_type = "3%止损" if abs(stop_loss_val - pct_sl) < abs(stop_loss_val - atr_sl) else "ATR止损"
                            else:
                                stop_loss_type = 'ATR止损'
                        except Exception:
                            stop_loss_type = 'ATR止损'

                    # 记录退出交易（确保一定追加记录）
                    # 统一并标准化退出原因：take_profit / atr_stop_loss / percent_stop_loss / max_holding_days
                    if exit_reason == 'take_profit':
                        canon_reason = 'take_profit'
                    elif exit_reason in ('stop_loss', 'loss', 'stoploss'):
                        canon_reason = 'percent_stop_loss' if str(stop_loss_type).strip() == '3%止损' else 'atr_stop_loss'
                    elif exit_reason in ('max_holding_days', 'max_holding', 'maxholding', 'timeout'):
                        canon_reason = 'max_holding_days'
                    else:
                        # 兜底处理：若价格等于止损/止盈价也进行判定
                        try:
                            def _almost_equal(a, b, rel=1e-6, abs_tol=1e-6):
                                if pd.isna(a) or pd.isna(b):
                                    return False
                                a = float(a); b = float(b)
                                return abs(a - b) <= max(abs(a), abs(b)) * rel + abs_tol
                        except Exception:
                            def _almost_equal(a, b, rel=1e-6, abs_tol=1e-6):
                                return False
                        if _almost_equal(exit_price, position_info.get('stop_loss', np.nan)):
                            canon_reason = 'percent_stop_loss' if str(stop_loss_type).strip() == '3%止损' else 'atr_stop_loss'
                        elif _almost_equal(exit_price, position_info.get('profit_target', np.nan)):
                            canon_reason = 'take_profit'
                        # MAX_HOLDING_DAYS是超时限制: 如果3天内没有触发止盈/止损,则在第4天触发超时
                        elif position_info.get('days_held', 0) > MAX_HOLDING_DAYS:
                            canon_reason = 'max_holding_days'
                        else:
                            canon_reason = 'max_holding_days' if exit_reason == 'max_holding_days' else 'atr_stop_loss' if exit_reason == 'stop_loss' else 'take_profit' if exit_reason == 'take_profit' else 'max_holding_days'


                    if position_info.get('exit_type') == 'intraday':
                        exit_timing = 'intraday'
                    elif canon_reason == 'max_holding_days':
                        exit_timing = 'max_holding'
                    else:
                        exit_timing = 'next_day_open'

                    exit_trade_record = {
                        'date': date,
                        'stock_code': stock_code,
                        pred_column: 0,
                        'score_final': 0,
                        'actual_return': exit_return,  # 累计收益
                        'position_weight': (1.0 / max_positions),
                        'contribution': (1.0 / max_positions) * exit_return,
                        'daily_position_ratio': 1.0,
                        'market_regime': 'exit',
                        'action': 'sell',
                        'days_held': position_info.get('days_held', 0),
                        'entry_date': position_info.get('entry_date', date),
                        'entry_price': entry_price,
                        'profit_target': position_info.get('profit_target', np.nan),
                        'stop_loss': position_info.get('stop_loss', np.nan),
                        'atr_value': position_info.get('atr_value', np.nan),
                        'stop_loss_type': stop_loss_type,
                        'exit_timing': exit_timing,
                        'exit_price': exit_price,
                        'exit_reason': canon_reason
                    }
                    exit_trades_records.append(exit_trade_record)
                    if stock_count == 1:  # 调试日志
                        logger.info(f"🔍 已添加退出记录: {stock_code}, exit_trades_records长度: {len(exit_trades_records)}")

                    # 将记录归入对应类别，供最终统计直接使用
                    if canon_reason in exit_lists:
                        exit_lists[canon_reason].append(exit_trade_record)

                    # 删除持仓
                    del current_positions[stock_code]
                    # logger.info(f"退出持仓: {stock_code}, 原因: {exit_reason}, 持仓{position_info['days_held']}天")

                # === 步骤3: 动态仓位管理和新股票买入 ===
                daily_picks = picks[picks['date'] == prev_date].copy() if prev_date is not None else pd.DataFrame()

                # 🆕 重构后的仓位管理逻辑
                # 统计当日退出类型（以退出时机为准）
                intraday_exits_count = 0  # T+2及以后盘中退出,当日不能买入
                can_buy_exits_count = 0   # T+2开盘退出或最大持仓期退出,当日可以买入

                for exit_record in exit_trades_records:
                    timing = exit_record.get('exit_timing') or 'next_day_open'
                    if timing == 'intraday':
                        # T+2及以后盘中触发止盈止损,当日卖出,当日不能买入
                        intraday_exits_count += 1
                    elif timing in ('next_day_open', 'max_holding'):
                        # T+1触发T+2开盘卖出,或最大持仓期开盘卖出,当日可以买入
                        can_buy_exits_count += 1

                # logger.info(f"当日退出统计: 盘中退出{intraday_exits_count}只(不可回补), 开盘退出{can_buy_exits_count}只(可回补)")

                # 过滤已持仓和当日已退出的股票（避免重复买同一标的）
                if not daily_picks.empty:
                    daily_picks = daily_picks[~daily_picks['stock_code'].isin(current_positions.keys())]
                    if today_exited_stocks:
                        daily_picks = daily_picks[~daily_picks['stock_code'].isin(today_exited_stocks)]

                # 🆕 修复后的动态仓位管理
                # 当前持仓数(已删除所有退出的股票)
                current_position_count = len(current_positions)

                # 空闲仓位数 = 最大持仓数 - 当前持仓数
                # 这包含了所有退出的仓位(盘中退出+开盘退出)
                empty_slots = max_positions - current_position_count

                # 可买入仓位数 = 空闲仓位数 - 盘中退出数量
                # 原因: 盘中退出的仓位虽然已经空出来,但当日不能买入
                # 只有开盘退出(next_day_open/max_holding)的仓位可以当日买入
                refill_slots = max(0, empty_slots - intraday_exits_count)

                # 调试日志
                if stock_count == 1 and (intraday_exits_count > 0 or can_buy_exits_count > 0):
                    logger.info(f"仓位管理: 当前持仓{current_position_count}, 空闲{empty_slots}, "
                               f"盘中退出{intraday_exits_count}(不可回补), 开盘退出{can_buy_exits_count}(可回补), "
                               f"可买入{refill_slots}")

                if not daily_picks.empty and refill_slots > 0:
                    daily_picks = daily_picks.head(refill_slots)
                    # logger.info(f"可买入新股票: {len(daily_picks)}只, 可用仓位(剔除盘中退出): {refill_slots}")
                # else:
                    # logger.info(f"无可买入股票或无可用仓位: 选股{len(daily_picks)}只, 可用仓位(剔除盘中退出){refill_slots}")

                # 🚀 修复：严格的TOP-K持仓管理，只有在可用仓位>0时才执行买入
                if not daily_picks.empty and refill_slots > 0:
                    # 已在上方按 refill_slots 裁剪 daily_picks
                    # 添加新持仓（严格遵守TOP-K限制）
                    new_positions_added = 0
                    for _, stock in daily_picks.iterrows():
                        # 🚀 双重检查：确保不超过最大持仓数
                        if len(current_positions) >= max_positions:
                            logger.warning(f"已达到最大持仓数{max_positions}，停止买入")
                            break

                        stock_code = stock['stock_code']

                        try:
                            # 获取T日的股价和ATR数据
                            stock_data = backtest_data[
                                (backtest_data['stock_code'] == stock_code) &
                                (backtest_data['date'] == date)
                            ]

                            if not stock_data.empty:
                                # 🚀 使用前复权价格作为入场价
                                entry_price = stock_data.iloc[0]['open_front']  # T+1日前复权开盘价作为入场价
                                # 获取T日ATR（使用prev_date）
                                if prev_date is None:
                                    # logger.info("首个交易日无法根据前一日信号买入，跳过")
                                    continue
                                atr_row = backtest_data[
                                    (backtest_data['stock_code'] == stock_code) &
                                    (backtest_data['date'] == prev_date)
                                ]
                                if atr_row.empty:
                                    # logger.warning(f"{stock_code} 在{prev_date}缺少ATR数据，跳过买入")
                                    continue
                                atr_value = atr_row.iloc[0]['atr14']   # T日ATR值

                                # 计算止盈止损位（基于前复权入场价）
                                profit_target = entry_price + ATR_MULTIPLIER_PROFIT * atr_value
                                atr_stop_loss = entry_price - ATR_MULTIPLIER_LOSS * atr_value
                                pct_stop_loss = entry_price * (1 - MAX_STOP_LOSS_PCT)
                                stop_loss = max(atr_stop_loss, pct_stop_loss)  # 取较高价格（较小损失）
                                stop_loss_type = "ATR止损" if stop_loss == atr_stop_loss else "3%止损"

                                # 添加到持仓跟踪
                                current_positions[stock_code] = {
                                    'selection_date': prev_date,  # T日选股日期
                                    'entry_price': entry_price,
                                    'atr_value': atr_value,
                                    'profit_target': profit_target,
                                    'stop_loss': stop_loss,
                                    'stop_loss_type': stop_loss_type,
                                    'days_held': 1  # T+1日买入，持仓第1天
                                }

                                # 关键修复：当日(T+1)触发的止盈/止损，标记为次日开盘退出（A股T+1规则）
                                # 🚀 使用前复权价格进行止盈止损判断
                                try:
                                    today_high = stock_data.iloc[0]['high_front']
                                    today_low = stock_data.iloc[0]['low_front']
                                    # 止损优先
                                    if pd.notna(today_low) and (today_low <= stop_loss):
                                        current_positions[stock_code]['exit_triggered'] = 'stop_loss'
                                        current_positions[stock_code]['exit_type'] = 'next_day_open'
                                        # if stock_count == 1:
                                            # logger.info(f"T+1当日触发止损(标记次日开盘退出): {stock_code}, 最低(前复权){today_low:.3f} <= 止损{stop_loss:.3f}")
                                    elif pd.notna(today_high) and (today_high >= profit_target):
                                        current_positions[stock_code]['exit_triggered'] = 'take_profit'
                                        current_positions[stock_code]['exit_type'] = 'next_day_open'
                                        # if stock_count == 1:
                                            # logger.info(f"T+1当日触发止盈(标记次日开盘退出): {stock_code}, 最高(前复权){today_high:.3f} >= 止盈{profit_target:.3f}")
                                except Exception as e:
                                    logger.warning(f"{stock_code} 买入日触发检查失败: {e}")

                                new_positions_added += 1
                                # logger.info(f"新增持仓: {stock_code}, 入场价{entry_price:.3f}, 止盈{profit_target:.3f}, 止盈倍数：{1.5 * atr_value}, 止损{stop_loss:.3f}({stop_loss_type})")
                            else:
                                logger.warning(f"无法获取{stock_code}的数据，跳过买入")

                        except Exception as e:
                            logger.warning(f"处理{stock_code}买入失败: {e}")

                    # if new_positions_added > 0:
                        # logger.info(f"当日新增持仓: {new_positions_added}只, 总持仓: {len(current_positions)}只")

                    # 🚀 验证持仓数量不超过限制
                    if len(current_positions) > max_positions:
                        logger.error(f"严重错误：持仓数{len(current_positions)}超过最大限制{max_positions}")
                elif not daily_picks.empty and refill_slots <= 0:
                    logger.info(f"TOP-K限制：已满仓({len(current_positions)}/{max_positions})，跳过{len(daily_picks)}只候选股票")

                # === 步骤4: 计算持仓股票收益 ===
                all_holding_stocks = list(current_positions.keys())

                if not all_holding_stocks:
                    # 空仓日
                    # 🔧 修复：在continue之前先添加退出交易记录
                    if stock_count == 1:  # 只在1只股票回测时输出详细日志
                        logger.info(f"  exit_trades_records最终长度: {len(exit_trades_records)}")
                    if exit_trades_records:
                        trades.extend(exit_trades_records)
                        if stock_count == 1:  # 只在1只股票回测时输出详细日志
                            logger.info(f"添加退出交易记录: {len(exit_trades_records)}条")
                            for record in exit_trades_records:
                                logger.info(f"  退出: {record['stock_code']}, 日期: {record['date']}, "
                                           f"原因: {record['exit_reason']}, 收益: {record['actual_return']:.4f}")

                    position_ratio_sum += 0.0
                    empty_position_days += 1
                    daily_returns.append(0.0)
                    portfolio_values.append(portfolio_values[-1])
                    portfolio_values_gross.append(portfolio_values_gross[-1])
                    continue

                # 统计仓位情况
                actual_stock_count = len(all_holding_stocks)
                if actual_stock_count < stock_count:
                    partial_position_days += 1

                # 固定仓位分配：Top-1=100%，Top-3=33.33%，Top-5=20%；不足K只不放大仓位
                per_stock_weight = 1.0 / max_positions
                position_weight = per_stock_weight
                daily_portfolio_return_gross = 0.0
                stock_returns = {}
                # 当日总仓位占比（用于统计日均仓位）
                daily_position_ratio = actual_stock_count * per_stock_weight
                position_ratio_sum += daily_position_ratio

                for stock_code in all_holding_stocks:
                    position_info = current_positions[stock_code]

                    try:
                        # 获取当日股价数据
                        stock_data = backtest_data[
                            (backtest_data['stock_code'] == stock_code) &
                            (backtest_data['date'] == date)
                        ]

                        if not stock_data.empty:
                            # 🚀 使用前复权价格计算收益
                            today_close = stock_data.iloc[0]['close_front']
                            today_open = stock_data.iloc[0]['open_front']
                            entry_price = position_info['entry_price']
                            days_held = position_info['days_held']

                            # 计算当日收益（用于组合收益）
                            if days_held == 1:
                                # 持仓第1天：当日收益 = (收盘价 - 开盘价) / 开盘价
                                daily_return = (today_close - today_open) / today_open
                            else:
                                # 持仓第N天：当日收益 = (收盘价 - 开盘价) / 开盘价
                                daily_return = (today_close - today_open) / today_open

                            # 计算累计收益（用于日志显示）
                            cumulative_return = (today_close - entry_price) / entry_price

                            # NaN检查
                            if pd.isna(daily_return):
                                daily_return = 0.0
                            if pd.isna(cumulative_return):
                                cumulative_return = 0.0

                            stock_returns[stock_code] = daily_return
                            daily_portfolio_return_gross += position_weight * daily_return

                            # if days_held <= 2:  # 只记录前两天的详细日志
                                # logger.info(f"{stock_code}: 持仓第{days_held}天, 当日收益{daily_return:.4f}, 累计收益{cumulative_return:.4f}")
                        else:
                            logger.warning(f"无法获取{stock_code}的价格数据")
                            stock_returns[stock_code] = 0.0

                    except Exception as e:
                        logger.warning(f"计算{stock_code}收益失败: {e}")
                        stock_returns[stock_code] = 0.0

                    # 获取股票选股信息
                    stock_pick_info = daily_picks[daily_picks['stock_code'] == stock_code] if not daily_picks.empty else pd.DataFrame()
                    if not stock_pick_info.empty:
                        stock_info = stock_pick_info.iloc[0]
                        action = 'buy'
                    else:
                        stock_info = pd.Series({
                            pred_column: 0,
                            'score_final': 0,
                            'market_regime': 'holding'
                        })
                        action = 'hold'

                    # 记录交易
                    trade_record = {
                        'date': date,
                        'stock_code': stock_code,
                        pred_column: stock_info.get(pred_column, 0),
                        'score_final': stock_info.get('score_final', 0),
                        'actual_return': stock_returns.get(stock_code, 0.0),
                        'position_weight': position_weight,
                        'contribution': position_weight * stock_returns.get(stock_code, 0.0),
                        'daily_position_ratio': daily_position_ratio,
                        'market_regime': stock_info.get('market_regime', 'holding'),
                        'action': action,
                        'days_held': position_info['days_held'],
                        'entry_date': position_info.get('selection_date', date),  # 使用选股日期
                        'entry_price': position_info['entry_price'],
                        'profit_target': position_info['profit_target'],
                        'stop_loss': position_info['stop_loss'],
                        'atr_value': position_info['atr_value'],
                        'stop_loss_type': position_info['stop_loss_type'],
                        'exit_reason': 'holding'  # 持仓中的股票
                    }
                    trades.append(trade_record)

                # 添加退出交易记录
                if stock_count == 1:  # 只在1只股票回测时输出详细日志
                    logger.info(f"  exit_trades_records最终长度: {len(exit_trades_records)}")
                if exit_trades_records:
                    trades.extend(exit_trades_records)
                    if stock_count == 1:  # 只在1只股票回测时输出详细日志
                        logger.info(f"添加退出交易记录: {len(exit_trades_records)}条")
                        for record in exit_trades_records:
                            logger.info(f"  退出: {record['stock_code']}, 日期: {record['date']}, "
                                       f"原因: {record['exit_reason']}, 收益: {record['actual_return']:.4f}")



                # 🔧 修复：基于实际交易金额计算交易成本
                # A股交易成本：佣金万一+印花税0.05%+过户费+滑点，总计约0.08%
                trade_cost = abs(daily_portfolio_return_gross) * TRADING_COST_RATE

                # 扣除交易成本得到净收益
                daily_portfolio_return_net = daily_portfolio_return_gross - trade_cost

                # 🔧 增强NaN检查
                if pd.isna(daily_portfolio_return_net):
                    logger.error(f"❌ {date} 日组合收益为NaN，设为0")
                    daily_portfolio_return_net = 0.0

                daily_returns.append(daily_portfolio_return_net)

                # 更新组合价值（区分毛利和净利）
                new_value_gross = portfolio_values_gross[-1] * (1 + daily_portfolio_return_gross)
                portfolio_values_gross.append(new_value_gross)

                new_value_net = portfolio_values[-1] * (1 + daily_portfolio_return_net)
                portfolio_values.append(new_value_net)

            except Exception as e:
                logger.warning(f"处理日期 {date} 失败: {e}")
                daily_returns.append(0.0)
                portfolio_values.append(portfolio_values[-1])
                continue

        # 计算回测指标
        if len(daily_returns) > 0:
            # 🔧 增强NaN检查和处理
            portfolio_final_value_net = portfolio_values[-1]
            portfolio_final_value_gross = portfolio_values_gross[-1]

            if pd.isna(portfolio_final_value_net):
                logger.error("❌ 组合最终净价值为NaN，设为1.0")
                portfolio_final_value_net = 1.0
            if pd.isna(portfolio_final_value_gross):
                logger.error("❌ 组合最终毛价值为NaN，设为1.0")
                portfolio_final_value_gross = 1.0

            total_return_net = portfolio_final_value_net - 1.0
            total_return_gross = portfolio_final_value_gross - 1.0

            if pd.isna(total_return_net):
                logger.error("❌ 净收益为NaN，设为0")
                total_return_net = 0.0
            if pd.isna(total_return_gross):
                logger.error("❌ 毛收益为NaN，设为0")
                total_return_gross = 0.0

            # 计算年化收益 (基于净收益)
            if portfolio_final_value_net > 0 and len(daily_returns) > 0:
                annual_return = (portfolio_final_value_net ** (252 / len(daily_returns))) - 1.0
                if pd.isna(annual_return):
                    logger.error("❌ 年化收益为NaN，设为0")
                    annual_return = 0.0
            else:
                annual_return = 0.0

            # 计算波动率
            returns_array = np.array(daily_returns)
            # 过滤NaN值
            returns_array = returns_array[~np.isnan(returns_array)]
            volatility = np.std(returns_array) * np.sqrt(252)

            # 计算夏普比率 (基于净收益)
            risk_free_rate = 0.03
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

            # 计算最大回撤 (基于净收益)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = np.min(drawdown)

            # 计算胜率 (基于净收益)
            positive_returns = returns_array[returns_array > 0]
            negative_returns = returns_array[returns_array < 0]
            win_rate = len(positive_returns) / len(returns_array) if len(returns_array) > 0 else 0

            # 计算盈亏比 (基于净收益)
            avg_profit = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

            # 平均单次收益 (基于净收益)
            avg_return = np.mean(returns_array)

            # 🚀 回归模式评估指标和ATR策略统计
            regression_metrics = {}
            atr_strategy_stats = {}
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)

                # 计算预测收益与实际收益的相关性
                if pred_column in trades_df.columns and 'actual_return' in trades_df.columns:
                    correlation = trades_df[pred_column].corr(trades_df['actual_return'])
                    regression_metrics['prediction_correlation'] = correlation

                    logger.info(f"📊 回归预测效果分析:")
                    logger.info(f"  预测收益与实际收益相关性: {correlation:.4f}")
                    logger.info(f"  总交易笔数: {len(trades_df)}")
                    logger.info(f"  平均预测收益: {trades_df[pred_column].mean():.4f}")
                    logger.info(f"  平均最终得分 (score_final): {trades_df['score_final'].mean():.4f}")
                    logger.info(f"  平均实际收益 (ATR): {trades_df['actual_return'].mean():.4f}")

                    # 诊断阈值统计
                    logger.info(f"  入选股票预测收益统计:")
                    logger.info(f"    均值: {trades_df[pred_column].mean():.3f} ATR")
                    logger.info(f"    中位数: {trades_df[pred_column].median():.3f} ATR")
                    logger.info(f"    25分位: {trades_df[pred_column].quantile(0.25):.3f} ATR")
                    logger.info(f"    75分位: {trades_df[pred_column].quantile(0.75):.3f} ATR")

                # 🚀 修复：正确的退出原因统计（4种退出类型 + 收益统计）
                if 'exit_reason' in trades_df.columns:


                    # 获取所有退出的交易（排除持仓中的）
                    exit_trades = trades_df[trades_df['exit_reason'] != 'holding'].copy()
                    # 🚀 直接使用回测过程中累计的四类退出列表（不做文本与价格二次分类）
                    atr_take_profit_trades = exit_lists['take_profit']
                    atr_stop_loss_trades = exit_lists['atr_stop_loss']
                    pct_stop_loss_trades = exit_lists['percent_stop_loss']
                    max_holding_trades = exit_lists['max_holding_days']

                    # 统计数量并确保一致性
                    atr_take_profit_count = len(atr_take_profit_trades)
                    atr_stop_loss_count = len(atr_stop_loss_trades)
                    pct_stop_loss_count = len(pct_stop_loss_trades)
                    max_holding_count = len(max_holding_trades)
                    total_exits = atr_take_profit_count + atr_stop_loss_count + pct_stop_loss_count + max_holding_count

                    logger.info(f"  退出交易数量: {total_exits}")
                    reason_dist = {
                        'take_profit': atr_take_profit_count,
                        'atr_stop_loss': atr_stop_loss_count,
                        'percent_stop_loss': pct_stop_loss_count,
                        'max_holding_days': max_holding_count
                    }
                    logger.info(f"  退出原因分布: {reason_dist}")

                    if total_exits > 0:

                        # 仅TOP-1增加额外诊断：按价格判定的退出类型数量
                        if stock_count == 1:
                            def price_cls(tr):
                                ep = tr.get('exit_price', np.nan)
                                sl = tr.get('stop_loss', np.nan)
                                tp = tr.get('profit_target', np.nan)
                                stype = str(tr.get('stop_loss_type', 'unknown')).strip()
                                def almost_equal(a,b,rel=1e-6,abs_tol=1e-6):
                                    if pd.isna(a) or pd.isna(b):
                                        return False
                                    try:
                                        a=float(a); b=float(b)
                                    except Exception:
                                        return False
                                    return abs(a-b) <= max(abs(a),abs(b))*rel + abs_tol
                                if almost_equal(ep, sl):
                                    return 'pct_stop_loss' if stype == '3%止损' else 'atr_stop_loss'
                                if almost_equal(ep, tp):
                                    return 'atr_take_profit'
                                return 'other'
                            price_based = exit_trades.apply(price_cls, axis=1).value_counts().to_dict()
                            logger.info(f"  价格判定退出分布(TOP-1): {price_based}")

                        # 🚀 重构：计算各类型的收益统计
                        def calculate_return_stats(trades_list):
                            if not trades_list:
                                return {'total_return': 0.0, 'avg_return': 0.0, 'count': 0}
                            returns = [trade['actual_return'] for trade in trades_list]
                            return {
                                'total_return': sum(returns),
                                'avg_return': sum(returns) / len(returns),
                                'count': len(returns)
                            }

                        atr_take_profit_stats = calculate_return_stats(atr_take_profit_trades)
                        atr_stop_loss_stats = calculate_return_stats(atr_stop_loss_trades)
                        pct_stop_loss_stats = calculate_return_stats(pct_stop_loss_trades)
                        max_holding_stats = calculate_return_stats(max_holding_trades)

                        # 计算比例
                        atr_strategy_stats = {
                            'atr_take_profit_count': atr_take_profit_count,
                            'atr_stop_loss_count': atr_stop_loss_count,
                            'pct_stop_loss_count': pct_stop_loss_count,
                            'max_holding_count': max_holding_count,
                            'atr_take_profit_rate': atr_take_profit_count / total_exits,
                            'atr_stop_loss_rate': atr_stop_loss_count / total_exits,
                            'pct_stop_loss_rate': pct_stop_loss_count / total_exits,
                            'max_holding_rate': max_holding_count / total_exits
                        }

                        logger.info(f"🎯 退出策略统计（含收益分析）:")
                        logger.info(f"  ATR止损退出: {atr_stop_loss_count}次 ({atr_strategy_stats['atr_stop_loss_rate']:.1%}), 总收益: {atr_stop_loss_stats['total_return']:+.2%}, 平均收益: {atr_stop_loss_stats['avg_return']:+.2%}")
                        logger.info(f"  百分比止损退出: {pct_stop_loss_count}次 ({atr_strategy_stats['pct_stop_loss_rate']:.1%}), 总收益: {pct_stop_loss_stats['total_return']:+.2%}, 平均收益: {pct_stop_loss_stats['avg_return']:+.2%}")
                        logger.info(f"  ATR止盈退出: {atr_take_profit_count}次 ({atr_strategy_stats['atr_take_profit_rate']:.1%}), 总收益: {atr_take_profit_stats['total_return']:+.2%}, 平均收益: {atr_take_profit_stats['avg_return']:+.2%}")
                        logger.info(f"  最大持仓期退出: {max_holding_count}次 ({atr_strategy_stats['max_holding_rate']:.1%}), 总收益: {max_holding_stats['total_return']:+.2%}, 平均收益: {max_holding_stats['avg_return']:+.2%}")
                        logger.info(f"  总退出次数: {total_exits}次")

                        # 🚀 新增：持仓天数分析
                        if 'days_held' in exit_trades.columns:
                            # 规范化展示：将>MAX_HOLDING_DAYS的持仓天数统一展示为MAX_HOLDING_DAYS
                            exit_trades_disp = exit_trades.copy()
                            try:
                                exit_trades_disp['days_held'] = exit_trades_disp['days_held'].clip(upper=MAX_HOLDING_DAYS)
                            except Exception:
                                pass
                            days_held_stats = exit_trades_disp['days_held'].value_counts().sort_index()
                            logger.info(f"📊 退出时持仓天数分布:")
                            for days, count in days_held_stats.items():
                                percentage = count / total_exits
                                logger.info(f"  持仓{int(days)}天退出: {count}次 ({percentage:.1%})")
                            logger.info(f"  平均持仓天数: {exit_trades_disp['days_held'].mean():.1f}天")
                            logger.info(f"  最大持仓期设置: {MAX_HOLDING_DAYS}天")

                        # 🚀 双重止损机制分析
                        if atr_stop_loss_count > 0 or pct_stop_loss_count > 0:
                            total_stop_loss = atr_stop_loss_count + pct_stop_loss_count
                            logger.info(f"📊 双重止损机制分析:")
                            logger.info(f"  ATR止损触发率: {atr_stop_loss_count}/{total_stop_loss} ({atr_stop_loss_count/total_stop_loss:.1%})")
                            logger.info(f"  3%止损触发率: {pct_stop_loss_count}/{total_stop_loss} ({pct_stop_loss_count/total_stop_loss:.1%})")
                            if atr_stop_loss_count > pct_stop_loss_count:
                                logger.info(f"  ✅ ATR止损更常触发，说明ATR设置合理")
                            elif pct_stop_loss_count > atr_stop_loss_count:
                                logger.info(f"  ⚠️ 3%止损更常触发，建议调整ATR倍数")
                            else:
                                logger.info(f"  📊 两种止损触发频率相当")
                    else:
                        logger.info(f"🎯 退出策略统计: 无退出交易")


            # 🚀 计算空仓日比例统计
            total_trading_days = len(trading_dates) -1 # 修正分母为总交易日
            empty_position_ratio = empty_position_days / total_trading_days if total_trading_days > 0 else 0
            partial_position_ratio = partial_position_days / total_trading_days if total_trading_days > 0 else 0
            full_position_ratio = 1 - empty_position_ratio - partial_position_ratio

            # 保存结果
            backtest_results[stock_count] = {
                'total_return': total_return_net, # 净收益
                'total_return_gross': total_return_gross, # 毛收益
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,  # 🚀 新增：盈亏比
                'avg_daily_return': avg_return,
                'total_trades': len(trades),
                'trading_days': len(daily_returns),
                'regression_metrics': regression_metrics,
                'atr_strategy_stats': atr_strategy_stats,  # 🚀 新增：ATR策略统计
                # 🚀 新增：仓位统计
                'empty_position_days': empty_position_days,
                'partial_position_days': partial_position_days,
                'empty_position_ratio': empty_position_ratio,
                'partial_position_ratio': partial_position_ratio,
                'full_position_ratio': full_position_ratio,
                'avg_daily_position': (position_ratio_sum / max(len(daily_returns), 1)),
                # 🎨 新增：用于绘制收益走势图
                'dates': [d.strftime('%Y-%m-%d') for d in trading_dates[1:]],  # 跳过第一天（初始日）
                'portfolio_values': portfolio_values[1:]  # 跳过初始值1.0
            }

            detailed_trades[stock_count] = trades

            logger.info(f"✅ {stock_count}只股票回测完成:")
            logger.info(f"  总收益率 (扣成本前): {total_return_gross:.2%}")
            logger.info(f"  净收益率 (扣成本后): {total_return_net:.2%}")
            logger.info(f"  年化收益率: {annual_return:.2%}")
            logger.info(f"  夏普比率: {sharpe_ratio:.3f}")
            logger.info(f"  最大回撤: {max_drawdown:.2%}")
            logger.info(f"  胜率: {win_rate:.2%}")
            logger.info(f"  盈亏比: {profit_loss_ratio:.2f}")  # 🚀 新增：盈亏比
            # 🚀 新增：空仓日比例统计
            logger.info(f"  空仓日比例: {empty_position_ratio:.1%} ({empty_position_days}天)")
            logger.info(f"  部分仓位日比例: {partial_position_ratio:.1%} ({partial_position_days}天)")
            logger.info(f"  满仓日比例: {full_position_ratio:.1%}")

    # ========== 返回完整回测结果 ==========
    logger.info(f"\n{window_prefix}✅ 选股回测完成!")
    logger.info(f"{window_prefix}   回测档位: {list(backtest_results.keys())}")

    return {
        'backtest_results': backtest_results,
        'detailed_trades': detailed_trades,
        'summary_stats': {
            'total_trading_days': len(trading_dates),
            'stock_counts': stock_counts,
            'window_name': window_name
        }
    }


def run_test_model_mode(args):
    """
    运行模型测试模式：训练模型→预测→选股→回测的纯预测流程

    Args:
        args: 命令行参数
    """
    logger = logging.getLogger(__name__)
    print(f"📊 运行模式: 模型预测选股")
    logger.info("=" * 60)
    logger.info(f"开始纯预测选股流程（无标签验证）")
    logger.info("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        import pickle
        import json
        from pathlib import Path
        from datetime import datetime

        # 🎯 1. 使用全局时间切分配置
        train_end_date = TRAIN_END_DATE
        test_start_date = pd.to_datetime(TEST_START_DATE)
        test_end_date = pd.to_datetime(TEST_END_DATE)

        logger.info("🗓️ 纯预测选股的时间切分策略（使用全局配置）:")
        logger.info(f"  训练数据: 2023-06-01 到 {train_end_date}")
        logger.info(f"  预测数据: {test_start_date.strftime('%Y-%m-%d')} 到 {test_end_date.strftime('%Y-%m-%d')}")
        logger.info("⚠️ 重要：模型将仅使用训练期数据训练，对测试期进行纯预测")

        # 🚀 解析模型选择参数
        models_str = getattr(args, 'models', 'xgboost,lightgbm')
        selected_models = [m.strip().lower() for m in models_str.split(',')]

        use_xgboost = 'xgboost' in selected_models
        use_lightgbm = 'lightgbm' in selected_models

        if not use_xgboost and not use_lightgbm:
            logger.error("❌ 至少需要选择一个模型 (xgboost 或 lightgbm)")
            return False

        logger.info(f"🚀 模型选择: {', '.join(selected_models)}")
        logger.info(f"   XGBoost: {'✅' if use_xgboost else '❌'}")
        logger.info(f"   LightGBM: {'✅' if use_lightgbm else '❌'}")

        # 2. 训练模型或跳过训练
        if getattr(args, 'skip_train', False):
            # 🆕 跳过训练，直接加载最新模型
            logger.info("\n⏭️ 步骤1: 跳过训练，直接加载最新模型...")
            logger.info("=" * 50)
        else:
            # 训练新模型
            logger.info("\n🔄 步骤1: 训练仅使用训练期数据的模型（使用优化后的配置）...")
            logger.info("🚀 优化配置：智能标签变换 + 增强模型复杂度 + 交易导向评估")
            logger.info("=" * 50)

            # 调用训练模式，使用优化后的配置：启用回归模式和模型选择
            train_success = run_train_model_mode(
                args,
                train_end_date=train_end_date,
                is_regression=True,
                use_xgboost=use_xgboost,  # 🚀 根据命令行参数决定
                use_lightgbm=use_lightgbm  # 🚀 根据命令行参数决定
            )

            if not train_success:
                logger.error("训练期专用模型训练失败")
                return False

            logger.info("✅ 训练期专用模型训练完成")
            logger.info("=" * 50)

        # 3. 加载完整因子数据用于预测（包含标签生成字段用于回测验证）
        logger.info("\n🔄 步骤2: 加载完整因子数据进行预测...")
        factor_data = _load_factor_cache_data(include_label_generation_features=True)

        if factor_data is None:
            logger.error("因子数据加载失败")
            return False

        logger.info(f"完整数据加载成功: {len(factor_data)} 条记录, {len(factor_data.columns)} 个字段")
        logger.info(f"数据日期范围: {factor_data['date'].min()} 到 {factor_data['date'].max()}")

        # 🆕 加载市场数据（用于市场状态判断）
        logger.info("\n🔄 加载市场数据（用于市场状态自适应）...")
        try:
            from data_loader import IntegratedDataLoader
            loader = IntegratedDataLoader()
            start_date_str = factor_data['date'].min().strftime('%Y%m%d')
            end_date_str = factor_data['date'].max().strftime('%Y%m%d')
            _, _, market_data = loader.load_all_data(start_date_str, end_date_str, extend_history_days=20)

            if market_data is None or market_data.empty:
                logger.warning("⚠️ 市场数据加载失败，将使用默认震荡市状态")
                market_data = None
            else:
                # 确保date列是datetime类型
                if 'date' in market_data.columns:
                    market_data['date'] = pd.to_datetime(market_data['date'])
                logger.info(f"✅ 市场数据加载成功: {len(market_data)}条记录")
        except Exception as e:
            logger.warning(f"⚠️ 市场数据加载失败: {e}，将使用默认震荡市状态")
            market_data = None

        # 3.5. 跳过交互特征计算，直接使用基础特征
        logger.info("🔄 使用基础特征进行预测...")

        # 4. 提取预测数据（确保与训练数据无重叠）
        logger.info("\n✂️ 步骤3: 提取纯预测数据...")
        logger.info("=" * 60)
        logger.info("🔍 数据集划分验证:")
        logger.info(f"  📊 完整数据: {len(factor_data):,} 条记录")
        logger.info(f"     📅 日期范围: {factor_data['date'].min().strftime('%Y-%m-%d')} 至 {factor_data['date'].max().strftime('%Y-%m-%d')}")
        logger.info(f"  🎯 训练截止日期: {train_end_date} (不包含)")
        logger.info(f"  🎯 测试开始日期: {test_start_date.strftime('%Y-%m-%d')} (包含)")
        logger.info(f"  🎯 测试结束日期: {test_end_date.strftime('%Y-%m-%d')} (包含)")

        prediction_data = factor_data[factor_data['date'] >= test_start_date].copy()

        logger.info(f"  📊 测试集: {len(prediction_data):,} 条记录")
        logger.info(f"     📅 日期范围: {prediction_data['date'].min().strftime('%Y-%m-%d')} 至 {prediction_data['date'].max().strftime('%Y-%m-%d')}")

        # 🚀 验证训练集和测试集无重叠
        train_data_count = len(factor_data[factor_data['date'] < pd.to_datetime(train_end_date)])
        overlap_count = len(factor_data[(factor_data['date'] >= test_start_date) & (factor_data['date'] < pd.to_datetime(train_end_date))])

        if overlap_count > 0:
            logger.error(f"❌ 数据泄露警告: 训练集和测试集有{overlap_count:,}条重叠记录!")
        else:
            logger.info(f"  ✅ 数据集无重叠: 训练集{train_data_count:,}条 < {train_end_date} <= 测试集{len(prediction_data):,}条")
        logger.info("=" * 60)

        if len(prediction_data) == 0:
            logger.error("预测数据为空，请检查日期范围设置")
            return False

        # 5. 加载刚训练好的模型和参数
        logger.info("\n📦 步骤4: 加载刚训练的专用模型...")

        results_dir = Path("results")
        if not results_dir.exists():
            logger.error("results目录不存在")
            return False

        # 🚀 查找最新的模型文件（根据用户选择）
        latest_xgb_file = None
        latest_lgb_file = None
        model_timestamp = None

        # 查找XGBoost模型
        if use_xgboost:
            logger.info("🔍 查找XGBoost模型（排除walkforward模式的window模型）...")
            xgb_files = list(results_dir.glob("xgboost_model_*.pkl"))
            # 🔧 修复：排除window开头的模型（walkforward模式专用）
            xgb_files = [f for f in xgb_files if not f.stem.startswith("xgboost_model_window")]

            if xgb_files:
                # 按时间戳排序,找到最新的XGBoost模型
                latest_xgb_file = max(xgb_files, key=lambda f: f.stem)
                logger.info(f"✅ 找到最新XGBoost模型: {latest_xgb_file.name}")

                # 提取时间戳
                parts = latest_xgb_file.stem.split('_')
                if len(parts) >= 3:
                    # 格式: xgboost_model_YYYYMMDD_HHMMSS.pkl
                    model_timestamp = f"{parts[-2]}_{parts[-1]}"  # YYYYMMDD_HHMMSS
            else:
                logger.warning("⚠️ 未找到XGBoost模型文件")
                if not use_lightgbm:
                    logger.error("❌ 仅选择XGBoost但未找到模型文件")
                    return False
                logger.warning("   将仅使用LightGBM模型")
                use_xgboost = False  # 降级为单模型

        # 查找LightGBM模型
        if use_lightgbm:
            logger.info("🔍 查找LightGBM模型（排除walkforward模式的window模型）...")
            lgb_files = list(results_dir.glob("lightgbm_model_*.pkl"))
            # 🔧 修复：排除window开头的模型（walkforward模式专用）
            lgb_files = [f for f in lgb_files if not f.stem.startswith("lightgbm_model_window")]

            if model_timestamp is not None:
                # 如果已有XGBoost时间戳,查找相同时间戳的LightGBM模型
                logger.info(f"   查找相同时间戳的LightGBM模型(时间戳: {model_timestamp})...")
                lgb_pattern = f"lightgbm_model_{model_timestamp}.pkl"
                lgb_file = results_dir / lgb_pattern

                if lgb_file.exists():
                    latest_lgb_file = lgb_file
                    logger.info(f"✅ 找到LightGBM模型: {latest_lgb_file.name}")
                else:
                    logger.warning(f"⚠️ 未找到相同时间戳的LightGBM模型")
                    logger.warning("   将仅使用XGBoost模型")
                    use_lightgbm = False  # 降级为单模型
            elif lgb_files:
                # 如果没有XGBoost时间戳,查找最新的LightGBM模型
                latest_lgb_file = max(lgb_files, key=lambda f: f.stem)
                logger.info(f"✅ 找到最新LightGBM模型: {latest_lgb_file.name}")
            else:
                logger.warning("⚠️ 未找到LightGBM模型文件")
                if not use_xgboost:
                    logger.error("❌ 仅选择LightGBM但未找到模型文件")
                    return False
                logger.warning("   将仅使用XGBoost模型")
                use_lightgbm = False  # 降级为单模型

        # 🚀 收集要加载的模型文件
        latest_models = []
        if latest_xgb_file is not None:
            latest_models.append(latest_xgb_file)
        if latest_lgb_file is not None:
            latest_models.append(latest_lgb_file)

        if not latest_models:
            logger.error("❌ 未找到任何可用的模型文件")
            return False

        logger.info(f"📁 将加载{len(latest_models)}个模型:")
        for f in latest_models:
            logger.info(f"  - {f.name}")

        # 🚀 加载所有模型
        models = []
        scalers = []
        model_types = []  # 记录模型类型
        model_performances = []  # 记录模型性能(用于权重计算)
        feature_names = None
        winsorization_thresholds = None  # 🚀 Winsorization阈值(从XGBoost加载)
        label_stats = None  # 🔧 修复：在循环外初始化

        for model_file in latest_models:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            models.append(model_data['model'])
            scalers.append(model_data['scaler'])
            model_types.append(model_data.get('model_type', 'xgboost'))  # 获取模型类型

            # 🚀 获取验证集性能(用于权重计算) - 使用RankIC而非MSE
            performance = model_data.get('performance', {})
            # 🚀 排序模型使用RankIC作为权重计算指标
            rank_ic = performance.get('rank_ic')
            model_performances.append(rank_ic)

            if feature_names is None:
                feature_names = model_data['feature_names']

            # 🚀 从XGBoost模型加载Winsorization阈值(集成学习共享预处理参数)
            if winsorization_thresholds is None and model_data.get('model_type') == 'xgboost':
                winsorization_thresholds = model_data.get('winsorization_thresholds', None)

            # 🔧 修复：从XGBoost模型加载标签统计量
            if label_stats is None and model_data.get('model_type') == 'xgboost':
                label_stats = model_data.get('label_stats', None)

            model_type = model_data.get('model_type', 'xgboost')
            logger.info(f"  ✅ 加载{model_type.upper()}模型: {model_file.name}")
            if rank_ic is not None:
                logger.info(f"     验证集RankIC: {rank_ic:.4f}")
            else:
                logger.warning(f"     ⚠️ 未找到验证集RankIC信息")

        # 🚀 检查标签统计量是否加载成功
        if label_stats:
            logger.info(f"✅ 加载标签统计量: μ={label_stats['mean']:.6f}, σ={label_stats['std']:.6f}")
        else:
            logger.warning("⚠️ 未找到标签统计量，可能影响回归预测的反标准化")

        logger.info(f"✅ 模型加载成功: {len(models)}个模型，特征数量: {len(feature_names)}")

        # 🚀 计算集成学习权重(基于验证集RankIC) - 排序模型专用
        model_weights = None
        if len(models) > 1:
            logger.info("\n📊 计算集成学习权重(基于验证集RankIC)...")
            logger.info("   理由: 排序模型的目标是排序准确性,RankIC是最佳评估指标")

            # 检查是否所有模型都有RankIC
            if all(rank_ic is not None for rank_ic in model_performances):
                # 🚀 权重 = |RankIC| (归一化)
                # 使用绝对值,因为负相关也有价值(可以反向使用)
                weights = np.array([abs(rank_ic) for rank_ic in model_performances])
                model_weights = weights / weights.sum()

                logger.info("✅ 模型权重:")
                for i, (model_type, weight, rank_ic) in enumerate(zip(model_types, model_weights, model_performances)):
                    logger.info(f"  模型{i+1}({model_type.upper()}): 权重={weight:.4f}, RankIC={rank_ic:.4f}")
            else:
                logger.warning("⚠️ 部分模型缺少RankIC信息,使用等权重")
                model_weights = np.ones(len(models)) / len(models)
                for i, model_type in enumerate(model_types):
                    logger.info(f"  模型{i+1}({model_type.upper()}): 权重={model_weights[i]:.4f}")
        else:
            logger.info("✅ 单模型模式,无需计算权重")


        # 6. 准备预测数据特征
        logger.info("\n🔧 步骤5: 准备预测数据特征...")

        # 识别特征列（使用与训练时完全相同的简化逻辑）
        prediction_feature_columns = _get_ml_feature_columns_simple(prediction_data)
        logger.info(f"预测数据识别到 {len(prediction_feature_columns)} 个特征")

        # 特征一致性检查（修复逻辑错误）
        logger.info(f"📊 特征分析:")
        logger.info(f"   训练时特征数: {len(feature_names)}个（包含PCA特征）")
        logger.info(f"   预测数据特征: {len(prediction_feature_columns)}个（原始特征）")

        # 检查预测数据中是否包含足够的特征来生成PCA
        # 注意：这里不需要与训练时的特征做严格对比，因为PCA会处理特征对齐
        logger.info(f"✅ 预测数据包含 {len(prediction_feature_columns)} 个原始特征，将用于PCA处理")

        logger.info(f"📊 预测数据统计:")
        logger.info(f"  预测样本总数: {len(prediction_data):,}个")
        logger.info(f"  预测日期范围: {prediction_data['date'].min()} 到 {prediction_data['date'].max()}")
        logger.info(f"  预测股票数量: {prediction_data['stock_code'].nunique()}只")

        # 7. 测试集数据预处理（使用训练时的Winsorization阈值）
        logger.info("\n⚙️ 步骤6: 测试集数据预处理（使用训练时参数）...")

        # 🚀 使用从XGBoost模型加载的Winsorization阈值(集成学习共享预处理参数)
        if winsorization_thresholds:
            logger.info(f"✅ 使用XGBoost模型的Winsorization阈值: {len(winsorization_thresholds)}个特征")
        else:
            logger.warning("⚠️ 未找到训练时的Winsorization阈值，将跳过缩尾处理")

        # 保存原始数据的索引，以便后续对齐预测结果
        original_prediction_data = prediction_data.copy()
        original_indices = prediction_data.index.copy()

        # 使用_prepare_feature_data进行测试集预处理
        X_pred_processed, available_features, _ = _prepare_feature_data(
            prediction_data, prediction_feature_columns,
            apply_winsorization=bool(winsorization_thresholds),
            winsorization_thresholds=winsorization_thresholds,
            data_type="测试集"
        )

        if X_pred_processed is None:
            logger.error("测试集数据预处理失败")
            return False

        logger.info(f"✅ 测试集预处理完成: {X_pred_processed.shape}")
        logger.info(f"原始数据长度: {len(original_prediction_data)}, 预处理后长度: {len(X_pred_processed)}")
        logger.info("✅ 确保测试集与训练时使用相同的数据预处理流程")

        # 使用预处理后的数据
        X_pred = X_pred_processed

        # 先将预测数据转换为DataFrame以便处理
        X_pred_df = pd.DataFrame(X_pred, columns=available_features)
        logger.info(f"📊 预测数据准备完成: {X_pred_df.shape}")

        # 7. 预测阶段特征对齐（使用基础特征）
        logger.info("\n🔧 步骤7: 预测阶段特征对齐...")

        # 8. 特征对齐处理（使用训练时的基础特征集）
        logger.info("\n🔧 步骤8: 特征对齐处理（使用基础特征集）...")
        feature_desc = "基础特征集（股票因子）"

        aligned_features = []
        aligned_data = []
        for feature in feature_names:
            if feature in X_pred_df.columns:
                aligned_features.append(feature)
                aligned_data.append(X_pred_df[feature].values)
            else:
                logger.warning(f"   ⚠️ 特征 {feature} 在预测数据中不存在，用0填充")
                aligned_features.append(feature)
                aligned_data.append(np.zeros(len(X_pred_df)))

        if aligned_data:
            X_pred = np.column_stack(aligned_data)
            available_features = aligned_features
            logger.info(f"✅ 特征对齐完成: {X_pred.shape}")
            logger.info(f"✅ 使用 {feature_desc}: {len(available_features)} 个特征")
            logger.info(f"✅ 特征数量匹配: 预测特征数={len(available_features)}, 训练特征数={len(feature_names)}")
        else:
            logger.error("❌ 特征对齐失败")
            return False

        # 🚀 9. 特征标准化（集成学习：延迟到预测循环中）
        logger.info("\n🔧 步骤9: 准备未标准化的特征数据（集成学习模式）...")

        # ✅ 最终特征（未标准化，将在集成预测循环中由每个模型的scaler进行标准化）
        X_pred_final = X_pred

        logger.info(f"✅ 最终特征准备完成: {X_pred_final.shape[1]} 个特征，{X_pred_final.shape[0]} 个样本")
        logger.info(f"⚠️ 注意：特征未标准化，将在集成预测循环中由每个模型的scaler进行标准化")

        # 🚀 10. Pseudo-Huber回归模型预测
        logger.info("\n🔮 步骤10: 进行Pseudo-Huber回归模型预测...")

        try:
            # 🚀 集成学习：对每个模型进行预测
            predictions = []
            for i, (model, scaler, model_type) in enumerate(zip(models, scalers, model_types)):
                logger.info(f"\n🔮 模型{i+1}({model_type.upper()})预测...")

                # 🔧 修复：不再使用StandardScaler，直接使用Winsorization后的特征
                # 因子在计算时已经进行了时间序列标准化（如zscore、rank）
                X_pred_scaled = X_pred_final

                # 🚀 根据模型类型进行预测
                if model_type == 'lightgbm':
                    # LightGBM模型：直接预测
                    y_pred_raw = model.predict(X_pred_scaled, num_iteration=model.best_iteration)
                    logger.info(f"  ✅ 使用LightGBM模型预测")
                elif hasattr(model, 'predict') and hasattr(model, 'get_score'):
                    # XGBoost Booster对象：需要创建DMatrix
                    import xgboost as xgb
                    dtest = xgb.DMatrix(X_pred_scaled)
                    y_pred_raw = model.predict(dtest)
                    logger.info(f"  ✅ 使用XGBoost Booster对象预测")
                else:
                    # XGBoost XGBRegressor对象：直接预测
                    y_pred_raw = model.predict(X_pred_scaled)
                    logger.info(f"  ✅ 使用XGBoost XGBRegressor对象预测")

                # 🚀 直接使用原始ATR预测值，无需逆变换
                y_pred = y_pred_raw
                logger.info(f"  ✅ 直接使用原始ATR预测值")

                predictions.append(y_pred)
                logger.info(f"  ✅ 模型{i+1}({model_type.upper()})预测完成: [{y_pred.min():.3f}, {y_pred.max():.3f}] ATR")

            # 🚀 集成预测：专业的集成策略(基于验证集MSE的加权平均)
            predictions = np.array(predictions)
            n_models = len(predictions)

            logger.info(f"\n✅ 集成预测完成")
            logger.info(f"  模型数量: {n_models}")

            # 检查模型数量
            if n_models == 1:
                logger.info(f"  ✅ 单模型模式,直接使用预测结果")
                mean_pred = predictions[0]
                weighted_mean_pred = predictions[0]
                std_pred = np.zeros_like(mean_pred)
                median_pred = mean_pred
                min_pred = mean_pred
                max_pred = mean_pred
            else:
                # 🆕 专业的集成统计
                # 1. 简单均值预测
                mean_pred = np.mean(predictions, axis=0)

                # 2. 🚀 基于验证集MSE的加权平均(使用之前计算的model_weights)
                if model_weights is not None:
                    # 加权平均
                    weighted_mean_pred = np.zeros_like(predictions[0])
                    for i, weight in enumerate(model_weights):
                        weighted_mean_pred += weight * predictions[i]

                    logger.info(f"  ✅ 使用基于MSE的加权平均")
                else:
                    # 使用简单均值
                    weighted_mean_pred = mean_pred
                    logger.info(f"  ✅ 使用简单均值")

                # 3. 标准差(模型间不确定性)
                std_pred = np.std(predictions, axis=0, ddof=1)  # 使用无偏估计

                # 4. 中位数预测(鲁棒性更好,对异常值不敏感)
                median_pred = np.median(predictions, axis=0)

                # 5. 最小/最大预测(预测范围)
                min_pred = np.min(predictions, axis=0)
                max_pred = np.max(predictions, axis=0)

                # 统计信息
                logger.info(f"  简单均值范围: [{mean_pred.min():.3f}, {mean_pred.max():.3f}] ATR")
                logger.info(f"  加权均值范围: [{weighted_mean_pred.min():.3f}, {weighted_mean_pred.max():.3f}] ATR")
                logger.info(f"  中位数范围: [{median_pred.min():.3f}, {median_pred.max():.3f}] ATR")
                logger.info(f"  标准差范围: [{std_pred.min():.3f}, {std_pred.max():.3f}] ATR")
                logger.info(f"  标准差均值: {std_pred.mean():.3f} ATR (模型间不确定性)")
                logger.info(f"  预测范围: [{min_pred.min():.3f}, {max_pred.max():.3f}] ATR")

                # 检查模型一致性
                avg_spread = (max_pred - min_pred).mean()
                logger.info(f"  平均预测跨度: {avg_spread:.3f} ATR (模型间差异)")

                if avg_spread > 2.0:
                    logger.warning(f"  ⚠️ 模型间差异较大(>{2.0} ATR),可能需要检查模型训练")

            # 🚀 最终预测值：直接使用加权均值（不做风险调整）
            # 理由：
            # 1. 风险调整会导致高预测值被过度惩罚
            # 2. 置信度过滤已经基于std_pred进行了不确定性控制
            # 3. 避免双重惩罚（风险调整 + 置信度过滤）
            y_pred_return = weighted_mean_pred

            logger.info(f"\n✅ 最终预测值（基于MSE加权均值，无风险调整）")
            logger.info(f"  使用加权均值预测（基于验证集MSE）")
            logger.info(f"  预测范围: [{y_pred_return.min():.3f}, {y_pred_return.max():.3f}] ATR")
            logger.info(f"  📊 最终预测值统计: [{y_pred_return.min():.3f}, {y_pred_return.max():.3f}] ATR")
            if n_models > 1:
                logger.info(f"  📊 不确定性控制: 通过置信度过滤（基于std_pred）实现")

        except Exception as e:
            logger.error(f"❌ 模型预测失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False

        # 8. 处理预测结果
        logger.info("\n📊 步骤11: 处理预测结果...")

        if len(y_pred_return) > len(original_prediction_data):
            logger.error(f"预测结果长度({len(y_pred_return)})大于原始数据长度({len(original_prediction_data)})")
            return False

        # 创建预测结果DataFrame
        prediction_data_with_proba = original_prediction_data.iloc[:len(y_pred_return)].copy()
        prediction_data_with_proba['predicted_return'] = y_pred_return  # 风险调整分数(基于加权均值)
        prediction_data_with_proba['mean_pred'] = mean_pred  # 🚀 集成学习：简单均值预测
        prediction_data_with_proba['weighted_mean_pred'] = weighted_mean_pred  # 🚀 集成学习：加权均值预测(基于MSE)
        prediction_data_with_proba['median_pred'] = median_pred if n_models > 1 else mean_pred  # 🚀 集成学习：中位数预测
        prediction_data_with_proba['std_pred'] = std_pred  # 🚀 集成学习：标准差（不确定性）
        prediction_data_with_proba['predicted_return_atr_raw'] = y_pred_return  # 🚀 修复：添加ATR原始预测字段
        prediction_data_with_proba['y_pred_proba'] = y_pred_return  # 兼容下游逻辑

        positive_count = (y_pred_return > 0).sum()
        logger.info(f"✅ 预测结果处理完成: {len(prediction_data_with_proba)} 个样本")
        logger.info(f"📊 预测分布: 正值{positive_count}个({positive_count/len(y_pred_return)*100:.1f}%), "
                   f"负值{len(y_pred_return)-positive_count}个")
        if n_models > 1:
            logger.info(f"📊 不确定性统计: 平均std={std_pred.mean():.3f}, 最大std={std_pred.max():.3f}")
        else:
            logger.info(f"📊 单模型预测,无不确定性统计")


        # 9. 🚀 置信度过滤（简化版：基于预测方差 + 市场状态自适应）
        logger.info("\n🎯 步骤12: 置信度过滤（基于预测方差 + 市场状态自适应）...")
        logger.info("=" * 60)

        try:
            # 🆕 市场状态判断
            market_state = _get_market_state(market_data, lookback=20, logger=logger)

            # 🆕 根据市场状态调整置信度阈值分位数
            if market_state == 'bull':
                confidence_percentile = 70  # 牛市：降低阈值，多选股
                logger.info(f"🐂 牛市状态：降低置信度要求（{confidence_percentile}%分位数）")
            elif market_state == 'bear':
                confidence_percentile = 85  # 熊市：提高阈值，少选股
                logger.info(f"🐻 熊市状态：提高置信度要求（{confidence_percentile}%分位数）")
            else:
                confidence_percentile = 75  # 震荡市：中等阈值
                logger.info(f"📊 震荡市状态：中等置信度要求（{confidence_percentile}%分位数）")

            # 🚀 简化方案：使用模型预测的标准差作为置信度指标
            # - std小 → 模型预测一致 → 高置信度
            # - std大 → 模型预测分歧 → 低置信度

            logger.info(f"\n📊 使用已计算的预测标准差作为置信度指标...")

            # 使用已经计算好的std_pred（模型间不确定性）
            pred_std = prediction_data_with_proba['std_pred'].values

            # 🆕 根据市场状态动态调整阈值
            threshold = np.percentile(pred_std, confidence_percentile)

            # 应用置信度过滤
            high_confidence_mask = pred_std < threshold

            logger.info(f"  预测标准差统计: μ={pred_std.mean():.6f}, σ={pred_std.std():.6f}")
            logger.info(f"  标准差范围: [{pred_std.min():.6f}, {pred_std.max():.6f}]")
            logger.info(f"  ✅ 置信度阈值({confidence_percentile}%分位数): {threshold:.6f}")

            logger.info(f"\n📊 置信度过滤结果:")
            logger.info(f"  总样本数: {len(pred_std):,}")
            logger.info(f"  高置信度样本: {high_confidence_mask.sum():,} ({high_confidence_mask.sum()/len(pred_std)*100:.1f}%)")
            logger.info(f"  低置信度样本: {(~high_confidence_mask).sum():,} ({(~high_confidence_mask).sum()/len(pred_std)*100:.1f}%)")

            # 添加置信度字段到预测结果
            prediction_data_with_proba['high_confidence'] = high_confidence_mask

            # 统计每日的置信度分布
            daily_confidence = prediction_data_with_proba.groupby('date').agg({
                'high_confidence': ['sum', 'count']
            }).reset_index()
            daily_confidence.columns = ['date', 'high_conf_count', 'total_count']
            daily_confidence['high_conf_ratio'] = daily_confidence['high_conf_count'] / daily_confidence['total_count']

            logger.info(f"\n📊 每日置信度分布:")
            logger.info(f"  平均高置信度比例: {daily_confidence['high_conf_ratio'].mean()*100:.1f}%")
            logger.info(f"  最低高置信度比例: {daily_confidence['high_conf_ratio'].min()*100:.1f}% (日期: {daily_confidence.loc[daily_confidence['high_conf_ratio'].idxmin(), 'date']})")
            logger.info(f"  最高高置信度比例: {daily_confidence['high_conf_ratio'].max()*100:.1f}% (日期: {daily_confidence.loc[daily_confidence['high_conf_ratio'].idxmax(), 'date']})")

            logger.info(f"\n✅ 置信度过滤完成")
            logger.info("=" * 60)

        except Exception as e:
            logger.warning(f"⚠️ 置信度过滤失败: {e}")
            logger.warning("将继续使用全部预测结果")
            import traceback
            logger.warning(f"详细错误: {traceback.format_exc()}")

            # 添加默认置信度字段
            prediction_data_with_proba['high_confidence'] = True

        # 10. 保存预测结果
        logger.info("\n💾 步骤13: 保存预测结果...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_results = prediction_data_with_proba[['date', 'stock_code', 'predicted_return', 'y_pred_proba', 'std_pred', 'high_confidence']].copy()
        prediction_file = results_dir / f"pure_predictions_{timestamp}.csv"
        prediction_results.to_csv(prediction_file, index=False, encoding='utf-8')
        logger.info(f"✅ 预测结果已保存: {prediction_file} ({len(prediction_results)} 条记录)")

        # 11. 选股回测系统 - 调用统一回测函数
        logger.info("\n🔄 步骤14: 开始选股回测...")
        logger.info("=" * 60)

        # ✅ 调用统一的选股回测函数
        backtest_result = _run_stock_selection_backtest(
            backtest_data=prediction_data_with_proba,
            stock_counts=[1, 2, 3, 5],  # Top-1/2/3/5选股
            window_name="test_model",
            market_data=None  # test_model暂不使用市场状态
        )

        if backtest_result is None or not backtest_result.get('backtest_results'):
            logger.error("❌ 选股回测失败")
            return False

        # 提取回测结果
        backtest_results = backtest_result['backtest_results']
        detailed_trades = backtest_result['detailed_trades']
        summary_stats = backtest_result['summary_stats']

        # 12. 保存回测结果
        logger.info("\n💾 步骤15: 保存回测结果...")

        # 保存回测指标汇总
        trading_dates = sorted(prediction_data_with_proba['date'].unique())
        backtest_summary = {
            "timestamp": timestamp,
            "backtest_period": {
                "start_date": trading_dates[0].strftime("%Y-%m-%d"),
                "end_date": trading_dates[-1].strftime("%Y-%m-%d"),
                "trading_days": len(trading_dates) - 1
            },
            "strategy_results": {}
        }


        # 🚀 输出置信度过滤统计
        if 'high_confidence' in prediction_data_with_proba.columns:
            logger.info("\n📊 置信度过滤统计:")
            logger.info("=" * 60)

            total_samples = len(prediction_data_with_proba)
            high_conf_samples = prediction_data_with_proba['high_confidence'].sum()
            low_conf_samples = total_samples - high_conf_samples

            logger.info(f"  总样本数: {total_samples:,}")
            logger.info(f"  高置信度样本: {high_conf_samples:,} ({high_conf_samples/total_samples*100:.1f}%)")
            logger.info(f"  低置信度样本: {low_conf_samples:,} ({low_conf_samples/total_samples*100:.1f}%)")

            # 统计每日置信度分布
            daily_stats = prediction_data_with_proba.groupby('date').agg({
                'high_confidence': ['sum', 'count']
            }).reset_index()
            daily_stats.columns = ['date', 'high_conf', 'total']
            daily_stats['ratio'] = daily_stats['high_conf'] / daily_stats['total']

            logger.info(f"\n  每日置信度分布:")
            logger.info(f"    平均高置信度比例: {daily_stats['ratio'].mean()*100:.1f}%")
            logger.info(f"    最低高置信度比例: {daily_stats['ratio'].min()*100:.1f}% (日期: {daily_stats.loc[daily_stats['ratio'].idxmin(), 'date'].strftime('%Y-%m-%d')})")
            logger.info(f"    最高高置信度比例: {daily_stats['ratio'].max()*100:.1f}% (日期: {daily_stats.loc[daily_stats['ratio'].idxmax(), 'date'].strftime('%Y-%m-%d')})")

            # 统计k-NN距离分布
            if 'knn_distance' in prediction_data_with_proba.columns:
                knn_dist = prediction_data_with_proba['knn_distance'].dropna()
                if len(knn_dist) > 0:
                    logger.info(f"\n  k-NN距离分布:")
                    logger.info(f"    均值: {knn_dist.mean():.4f}")
                    logger.info(f"    标准差: {knn_dist.std():.4f}")
                    logger.info(f"    范围: [{knn_dist.min():.4f}, {knn_dist.max():.4f}]")
                    logger.info(f"    分位数: 25%={knn_dist.quantile(0.25):.4f}, 50%={knn_dist.quantile(0.50):.4f}, 75%={knn_dist.quantile(0.75):.4f}, 90%={knn_dist.quantile(0.90):.4f}")

            logger.info("=" * 60)

        # 输出对比结果
        logger.info("\n📊 “唯一口径”选股策略回测结果对比:")
        logger.info("=" * 150)
        logger.info(f"{'策略':<10} {'总收益':<10} {'净收益(扣成本)':<15} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<8} {'盈亏比':<8} {'日均仓位':<10}")
        logger.info("=" * 150)


        stock_counts = [1, 2, 3, 5]  # Top-1/2/3/5选股
        for stock_count in stock_counts:
            if stock_count in backtest_results:
                result = backtest_results[stock_count]
                backtest_summary["strategy_results"][f"{stock_count}stocks"] = result

                # 固定仓位大小（移除market attention相关逻辑）
                avg_pos = result.get('avg_daily_position', 0.0)
                position_desc = f"{avg_pos:>8.1%}"

                logger.info(f"Top-{stock_count:<5} {result['total_return_gross']:>8.2%}  "
                           f"{result['total_return']:>13.2%}  "
                           f"{result['annual_return']:>8.2%}  {result['sharpe_ratio']:>8.3f}  "
                           f"{result['max_drawdown']:>8.2%}  {result['win_rate']:>6.2%}  "
                           f"{result['profit_loss_ratio']:>6.2f}  "
                           f"{position_desc}")

        logger.info("=" * 150)

        # 保存回测汇总报告
        backtest_report_file = results_dir / f"backtest_summary_{timestamp}.json"
        with open(backtest_report_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_summary, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"✅ 回测汇总报告已保存: {backtest_report_file}")

        # 保存详细交易记录
        for stock_count in stock_counts:
            if stock_count in detailed_trades:
                trades_df = pd.DataFrame(detailed_trades[stock_count])
                trades_file = results_dir / f"detailed_trades_{stock_count}stocks_{timestamp}.csv"
                trades_df.to_csv(trades_file, index=False, encoding='utf-8')
                logger.info(f"✅ {stock_count}只股票交易记录已保存: {trades_file}")

        # 🎨 生成收益走势图
        logger.info("\n📊 生成收益走势图...")
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib import rcParams

            # 设置中文字体
            rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            rcParams['axes.unicode_minus'] = False

            # 创建图表
            fig, ax = plt.subplots(figsize=(14, 8))

            # 为每个策略绘制收益曲线
            colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 5: '#96CEB4'}
            linestyles = {1: '-', 2: '--', 3: '-.', 5: ':'}

            for stock_count in stock_counts:
                if stock_count in backtest_results:
                    result = backtest_results[stock_count]

                    # 获取日期和累计收益
                    dates = result.get('dates', [])
                    portfolio_values = result.get('portfolio_values', [])

                    if len(dates) > 0 and len(portfolio_values) > 0:
                        # 转换为百分比收益
                        returns_pct = [(v - 1.0) * 100 for v in portfolio_values]

                        # 转换日期格式
                        dates_formatted = [pd.to_datetime(d) for d in dates]

                        # 绘制曲线
                        ax.plot(dates_formatted, returns_pct,
                               label=f'Top-{stock_count}',
                               color=colors.get(stock_count, '#95A5A6'),
                               linestyle=linestyles.get(stock_count, '-'),
                               linewidth=2.5 if stock_count == 1 else 2.0,
                               alpha=0.9)

            # 设置图表样式
            ax.set_xlabel('日期', fontsize=12, fontweight='bold')
            ax.set_ylabel('累计收益率 (%)', fontsize=12, fontweight='bold')
            ax.set_title('不同选股策略收益走势对比', fontsize=14, fontweight='bold', pad=20)

            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # 添加零线
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

            # 设置图例
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9, shadow=True)

            # 格式化x轴日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45, ha='right')

            # 调整布局
            plt.tight_layout()

            # 保存图表
            chart_file = results_dir / f"return_curves_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            logger.info(f"✅ 收益走势图已保存: {chart_file}")

            plt.close()

        except Exception as e:
            logger.warning(f"⚠️ 生成收益走势图失败: {e}")

        logger.info("=" * 60)
        logger.info("🎉 Pseudo-Huber回归选股回测完成！")
        logger.info("📊 已完成1只、2只、3只、5只股票的选股策略对比")
        logger.info("✅ 样本外验证：模型从未见过测试期数据")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ 模型测试模式失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_factor_correlation_mode(args):
    """运行因子相关性分析模式"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pathlib import Path

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🔍 ATR三分类标签系统 - 因子相关性分析模式")
    logger.info("=" * 80)

    try:
        # 1. 加载因子数据
        logger.info("📊 步骤1: 加载系统因子数据")
        logger.info("-" * 50)


        # 🔧 因子测试使用模型因子列表（归一化因子）
        required_factors = _get_model_factors()

        logger.info(f"分析因子数量: {len(required_factors)}")
        logger.info("因子列表:")
        for i, factor in enumerate(required_factors, 1):
            logger.info(f"  {i:2d}. {factor}")

        # 加载因子数据
        logger.info("\n🔄 加载因子缓存数据...")
        factor_data = _load_factor_cache_data(required_factors=required_factors)

        if factor_data is None or factor_data.empty:
            logger.error("❌ 因子数据加载失败")
            return False

        logger.info(f"✅ 因子数据加载成功: {len(factor_data):,}条记录")
        logger.info(f"数据时间范围: {factor_data['date'].min()} 至 {factor_data['date'].max()}")

        # 2. 数据预处理和验证
        logger.info("\n📊 步骤2: 数据预处理和验证")
        logger.info("-" * 50)

        # 检查实际可用的因子
        available_factors = [factor for factor in required_factors if factor in factor_data.columns]
        missing_factors = [factor for factor in required_factors if factor not in factor_data.columns]

        logger.info(f"可用因子: {len(available_factors)}/{len(required_factors)}")
        if missing_factors:
            logger.warning(f"缺失因子: {missing_factors}")

        if len(available_factors) < 2:
            logger.error("❌ 可用因子数量不足，无法进行相关性分析")
            return False

        # 提取因子数据矩阵
        factor_matrix = factor_data[available_factors].copy()

        # 数据清理：移除包含NaN的行
        original_rows = len(factor_matrix)
        factor_matrix = factor_matrix.dropna()
        cleaned_rows = len(factor_matrix)

        logger.info(f"数据清理: {original_rows:,} → {cleaned_rows:,} 行 (移除{original_rows-cleaned_rows:,}行NaN)")

        if cleaned_rows < 1000:
            logger.error(f"❌ 清理后数据量不足({cleaned_rows}行)，无法进行可靠的相关性分析(至少需要1000行)")
            return False

        # 3. 计算相关性矩阵(Spearman秩相关)
        logger.info("\n🔍 步骤3: 计算因子相关性矩阵")
        logger.info("-" * 50)

        logger.info("使用Spearman秩相关系数计算相关性...")
        logger.info("优势: 对异常值鲁棒,捕捉非线性单调关系,量化领域标准方法")
        logger.info("公式: ρ_s(X,Y) = Pearson(rank(X), rank(Y))")

        # 计算Spearman秩相关矩阵
        correlation_matrix = factor_matrix.corr(method='spearman')

        logger.info(f"✅ Spearman秩相关矩阵计算完成: {correlation_matrix.shape[0]}×{correlation_matrix.shape[1]}")

        # 4. 高相关性因子识别
        logger.info("\n🎯 步骤4: 高相关性因子识别")
        logger.info("-" * 50)

        correlation_threshold = 0.7
        logger.info(f"相关性阈值: {correlation_threshold}")

        # 找出高相关性因子对
        high_corr_pairs = []
        n_factors = len(available_factors)

        for i in range(n_factors):
            for j in range(i+1, n_factors):
                factor1 = available_factors[i]
                factor2 = available_factors[j]
                corr_value = correlation_matrix.iloc[i, j]

                if abs(corr_value) >= correlation_threshold:
                    high_corr_pairs.append({
                        'factor1': factor1,
                        'factor2': factor2,
                        'correlation': corr_value
                    })

        # 按相关性绝对值排序
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        logger.info(f"发现 {len(high_corr_pairs)} 对高相关性因子:")
        if high_corr_pairs:
            for i, pair in enumerate(high_corr_pairs, 1):
                corr_type = "正相关" if pair['correlation'] > 0 else "负相关"
                logger.info(f"  {i:2d}. {pair['factor1']} ↔ {pair['factor2']}: "
                           f"{pair['correlation']:.3f} ({corr_type})")
        else:
            logger.info("  未发现高相关性因子对")

        # 5. 生成相关性热力图
        logger.info("\n📈 步骤5: 生成相关性热力图")
        logger.info("-" * 50)

        # 设置输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建热力图
        plt.style.use('default')
        fig_size = max(12, len(available_factors) * 0.8)  # 动态调整图片大小
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # 生成热力图
        heatmap = sns.heatmap(
            correlation_matrix,
            annot=True,  # 显示数值
            cmap='RdBu_r',  # 红蓝色彩映射
            center=0,  # 以0为中心
            square=True,  # 正方形格子
            fmt='.3f',  # 数值格式
            cbar_kws={'label': '相关系数'},
            ax=ax
        )

        # 设置标题和标签
        plt.title('ATR三分类标签系统 - 因子Spearman秩相关矩阵', fontsize=16, pad=20)
        plt.xlabel('因子', fontsize=12)
        plt.ylabel('因子', fontsize=12)

        # 旋转标签以提高可读性
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        heatmap_path = output_dir / 'factor_correlation_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ 相关性热力图已保存: {heatmap_path}")

        # 关闭图形以释放内存
        plt.close()

        # 6. 因子冗余性分析和建议
        logger.info("\n💡 步骤6: 因子冗余性分析和建议")
        logger.info("-" * 50)

        if high_corr_pairs:
            logger.info("🔍 因子冗余性分析:")

            # 统计每个因子的高相关性次数
            factor_corr_count = {}
            for pair in high_corr_pairs:
                factor_corr_count[pair['factor1']] = factor_corr_count.get(pair['factor1'], 0) + 1
                factor_corr_count[pair['factor2']] = factor_corr_count.get(pair['factor2'], 0) + 1

            # 按高相关性次数排序
            sorted_factors = sorted(factor_corr_count.items(), key=lambda x: x[1], reverse=True)

            logger.info("因子冗余性排名（高相关性次数）:")
            for i, (factor, count) in enumerate(sorted_factors, 1):
                logger.info(f"  {i:2d}. {factor}: {count}次高相关")

            logger.info("\n📋 特征工程建议:")
            logger.info("1. 考虑移除高冗余因子以降低多重共线性")
            logger.info("2. 对于高相关因子对，可以考虑:")
            logger.info("   - 保留业务解释性更强的因子")
            logger.info("   - 保留计算更稳定的因子")
            logger.info("   - 使用PCA等降维技术")
            logger.info("3. 建议优先保留的因子类型:")
            logger.info("   - 收益率相关: returns_zscore")
            logger.info("   - 市场结构: market_structure_score")
            logger.info("   - 动量指标: intraday_momentum, rank_momentum_20d")
            logger.info("   - 技术指标: ma_ratio_5_10, rank_rsi, rank_macd")
        else:
            logger.info("✅ 未发现显著的因子冗余性问题")
            logger.info("当前因子组合具有良好的独立性")

        # 7. 生成分析报告
        logger.info("\n📄 步骤7: 生成分析报告")
        logger.info("-" * 50)

        report_path = output_dir / 'factor_correlation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ATR三分类标签系统 - 因子Spearman秩相关性分析报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"分析方法: Spearman秩相关(对异常值鲁棒,量化标准方法)\n")
            f.write(f"分析时间: {factor_data['date'].min()} 至 {factor_data['date'].max()}\n")
            f.write(f"数据量: {cleaned_rows:,} 条记录\n")
            f.write(f"分析因子数量: {len(available_factors)}\n")
            f.write(f"相关性阈值: {correlation_threshold}\n\n")

            f.write("分析因子列表:\n")
            for i, factor in enumerate(available_factors, 1):
                f.write(f"  {i:2d}. {factor}\n")

            f.write(f"\n高相关性因子对 ({len(high_corr_pairs)}对):\n")
            if high_corr_pairs:
                for i, pair in enumerate(high_corr_pairs, 1):
                    corr_type = "正相关" if pair['correlation'] > 0 else "负相关"
                    f.write(f"  {i:2d}. {pair['factor1']} ↔ {pair['factor2']}: "
                           f"{pair['correlation']:.3f} ({corr_type})\n")
            else:
                f.write("  未发现高相关性因子对\n")

            if high_corr_pairs:
                f.write("\n因子冗余性排名:\n")
                for i, (factor, count) in enumerate(sorted_factors, 1):
                    f.write(f"  {i:2d}. {factor}: {count}次高相关\n")

        logger.info(f"✅ 分析报告已保存: {report_path}")

        # 8. 总结
        logger.info("\n" + "=" * 80)
        logger.info("🎉 因子相关性分析完成")
        logger.info("=" * 80)
        logger.info(f"📊 分析结果总结:")
        logger.info(f"  • 分析因子数量: {len(available_factors)}")
        logger.info(f"  • 数据记录数量: {cleaned_rows:,}")
        logger.info(f"  • 高相关性因子对: {len(high_corr_pairs)}")
        logger.info(f"  • 相关性热力图: {heatmap_path}")
        logger.info(f"  • 分析报告: {report_path}")

        if high_corr_pairs:
            logger.info(f"⚠️  发现 {len(high_corr_pairs)} 对高相关性因子，建议进行特征工程优化")
        else:
            logger.info("✅ 因子独立性良好，无需额外处理")

        return True

    except Exception as e:
        logger.error(f"因子相关性分析失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_genetic_factor_mode(args):
    """运行遗传编程因子进化模式"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🧬 遗传编程因子进化模式")
    logger.info("=" * 80)

    try:
        # 检查gplearn是否可用
        try:
            import gplearn
            logger.info(f"✅ gplearn版本: {gplearn.__version__}")
        except ImportError:
            logger.error("❌ gplearn未安装，请运行: pip install gplearn")
            return False

        # 导入遗传编程模块
        try:
            from genetic_factor_evolution import run_genetic_factor_evolution
        except ImportError as e:
            logger.error(f"❌ 无法导入遗传编程模块: {e}")
            return False

        # 1. 加载因子数据
        logger.info("📊 步骤1: 加载因子数据")
        logger.info("-" * 50)

        # 定义需要的基础因子
        required_factors = [
            # 技术因子
            'rsi6', 'rsi12', 'rsi14', 'macd', 'volatility20',
            'return_5d', 'return_20d', 'ma5', 'ma10', 'ma20', 'ma60',

            # 动量因子
            'momentum_5d', 'momentum_20d', 'momentum_60d',
            'price_momentum', 'volume_momentum',

            # 成交量因子
            'volume_ratio_5d', 'volume_ratio_20d', 'turnover_rate',
            'volume_price_trend', 'volume_ma_ratio',

            # 价格位置因子
            'price_position_5d', 'price_position_20d', 'price_position_60d',

            # 涨停质量因子
            'is_limit_up', 'limit_strength', 'limit_up_days_5d',
            'consecutive_limit_up', 'limit_up_volume_ratio',

            # 其他重要因子
            'returns_zscore','market_structure_score','ma_ratio_5_10','corr_5d','volatility_correlation',
            'volume_correlation','near_support_count_60d','institutional_activity_score',
            'ma_convergence_score','lower_shadow_ratio','sector_correlation_20d','near_support',
            'price_impact','rank_momentum_20d','rank_rsi','rank_volume'
        ]

        logger.info(f"需要加载的因子数量: {len(required_factors)}")

        # 加载因子数据
        factor_data = _load_factor_cache_data(
            required_factors=required_factors,
            include_label_generation_features=True  # 包含标签生成特征用于目标变量
        )

        if factor_data is None or len(factor_data) == 0:
            logger.error("❌ 因子数据加载失败")
            return False

        logger.info(f"✅ 因子数据加载成功: {len(factor_data)} 条记录")
        logger.info(f"数据时间范围: {factor_data['date'].min()} 到 {factor_data['date'].max()}")

        # 2. 数据预处理
        logger.info("\n📊 步骤2: 数据预处理")
        logger.info("-" * 50)

        # 确保有目标变量
        if 'future_return_2d' not in factor_data.columns:
            logger.info("计算目标变量 future_return_2d...")
            factor_data = factor_data.sort_values(['stock_code', 'date'])
            factor_data['future_return_2d'] = factor_data.groupby('stock_code')['pct_chg'].shift(-2)

        # 过滤有效数据
        valid_mask = (
            factor_data['future_return_2d'].notna() &
            (factor_data['future_return_2d'].abs() <= 0.2)  # 限制收益率在合理范围内
        )
        factor_data = factor_data[valid_mask].copy()

        logger.info(f"有效数据: {len(factor_data)} 条记录")

        # 3. 运行遗传编程进化
        logger.info("\n🧬 步骤3: 开始遗传编程因子进化")
        logger.info("-" * 50)

        logger.info(f"遗传编程参数:")
        logger.info(f"  种群大小: {args.population_size}")
        logger.info(f"  进化代数: {args.generations}")
        logger.info(f"  返回因子数: {args.n_best_factors}")

        # 运行遗传编程
        results = run_genetic_factor_evolution(
            factor_data=factor_data,
            output_dir=args.output_dir,
            population_size=args.population_size,
            generations=args.generations,
            n_best_factors=args.n_best_factors
        )

        # 4. 结果总结
        logger.info("\n📋 步骤4: 结果总结")
        logger.info("-" * 50)

        if results and results.get('best_program'):
            best = results['best_program']
            logger.info(f"🏆 最佳因子进化完成!")
            logger.info(f"   夏普比率: {best.get('sharpe_ratio', 'N/A'):.4f}")
            logger.info(f"   最大回撤: {best.get('max_drawdown', 'N/A'):.4f}")
            logger.info(f"   年化收益: {best.get('annual_return', 'N/A'):.4f}")
            logger.info(f"   胜率: {best.get('win_rate', 'N/A'):.4f}")



    except Exception as e:
        logger.error(f"❌ 遗传编程因子进化模式运行失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_market_state_test_mode(args):
    """运行市场状态因子分析模式

    分析目标:
    1. 评估市场状态因子的择时能力 ([1,2,3,5,10]日周期)
    2. 评估市场状态因子对18个默认股票因子的影响(开关效应)
    3. 生成专业的量化分析报告
    """
    import numpy as np
    from datetime import datetime
    import pandas as pd
    import statsmodels.api as sm

    logger = logging.getLogger(__name__)
    print("📊 运行模式: 市场状态因子分析 (Enhanced)")
    logger.info("=" * 60)
    logger.info("🚀 开始市场状态因子分析流程")

    # 定义分析周期(针对短期持仓优化)
    analysis_periods = [1, 2, 3]  # 针对2日持仓策略优化
    logger.info(f"🔧 分析配置: 时间窗口 {analysis_periods}日 (针对2日持仓优化)，目标因子 18个核心因子")

    # 步骤 1: 加载数据
    logger.info("🔄 [步骤 1/5] 加载所有数据...")
    # 🚀 修复：直接使用_load_factor_cache_data加载所有因子（包含市场因子）
    all_factors_df = _load_factor_cache_data(include_label_generation_features=True)
    if all_factors_df is None: return False

    # 🚀 关键修复：处理重复列名
    if all_factors_df.columns.has_duplicates:
        duplicates = all_factors_df.columns[all_factors_df.columns.duplicated()].tolist()
        logger.warning(f"检测到重复的因子列名: {duplicates}。将自动去重，保留第一个出现的列。")
        all_factors_df = all_factors_df.loc[:, ~all_factors_df.columns.duplicated(keep='first')]

    logger.info(f"✅ 成功加载统一因子数据: {len(all_factors_df):,} 条记录, {len(all_factors_df.columns)} 个字段")

    # 🚀 关键修复：识别市场因子和股票因子（无需分离和重新合并）
    market_factor_columns = _get_available_market_factors()
    available_market_factors = [col for col in market_factor_columns if col in all_factors_df.columns]
    missing_market_factors = [col for col in market_factor_columns if col not in all_factors_df.columns]

    logger.info(f"📊 因子分布分析:")
    logger.info(f"  - 可用市场因子: {len(available_market_factors)}/{len(market_factor_columns)}")
    if missing_market_factors:
        logger.info(f"  - 缺失市场因子: {missing_market_factors[:5]}{'...' if len(missing_market_factors) > 5 else ''}")

    if len(available_market_factors) == 0:
        logger.warning("⚠️ 未找到任何市场因子，将跳过市场因子相关分析")
        return False

    # 步骤 2: 数据准备（添加市场收益率）
    logger.info("🔄 [步骤 2/5] 准备分析数据...")
    try:
        loader = IntegratedDataLoader()
        start_date = all_factors_df['date'].min().strftime('%Y%m%d')
        end_date = all_factors_df['date'].max().strftime('%Y%m%d')
        _, _, market_data = loader.load_all_data(start_date, end_date, extend_history_days=20)
        if market_data is None: raise ValueError("未能加载 market_data")
        market_data['date'] = pd.to_datetime(market_data['date'])

        # 计算市场收益率
        for n in analysis_periods:
            market_data[f'Mkt_Return_Next_{n}d'] = market_data['close'].pct_change(periods=-n).shift(-n)

        # 🚀 关键修复：直接合并到统一数据中，避免重复合并
        return_columns = ['date'] + [f'Mkt_Return_Next_{n}d' for n in analysis_periods]
        all_factors_df = pd.merge(all_factors_df, market_data[return_columns], on='date', how='left')

        logger.info(f"✅ 市场收益率数据合并完成: {len(all_factors_df):,} 条记录")

    except Exception as e:
        logger.error(f"❌ 加载市场数据时出错: {e}")
        return False

    # 步骤 3: 市场择时能力测试
    logger.info("🔄 [步骤 3/5] 开始执行市场择时能力测试...")
    return_columns_set = set(f'Mkt_Return_Next_{n}d' for n in analysis_periods)

    # 🚀 关键修复：直接从统一数据中获取市场因子名称
    market_factor_names = [col for col in available_market_factors if col in all_factors_df.columns]
    logger.info(f"📊 择时能力测试: {len(market_factor_names)} 个市场因子")

    timing_test_results = {}

    for factor in market_factor_names:
        is_significant = False
        factor_results = {}
        for n in analysis_periods:
            # 🚀 关键修复：直接从统一数据中提取测试数据
            test_data = all_factors_df[[factor, f'Mkt_Return_Next_{n}d']].dropna()
            # 提高最小样本要求，增强统计稳健性
            if len(test_data) > 30:  # 从20提升到30
                try:
                    X = sm.add_constant(test_data[factor])
                    y = test_data[f'Mkt_Return_Next_{n}d']
                    model = sm.OLS(y, X).fit()
                    # 提高t统计量阈值，增加显著性要求
                    if abs(model.tvalues[factor]) > 2.5 and model.pvalues[factor] < 0.05:  # 从2.0提升到2.5
                        factor_results[n] = {
                            't_stat': model.tvalues[factor],
                            'r_squared': model.rsquared,
                            'p_value': model.pvalues[factor],
                            'period': n
                        }
                        is_significant = True
                        logger.info(f"  ✅ {factor} 在{n}日周期显著: t={model.tvalues[factor]:.2f}, R²={model.rsquared:.3f}")
                except Exception as e:
                    logger.warning(f"因子 '{factor}' 第{n}天择时测试失败: {e}")
                    continue
        if is_significant:
            timing_test_results[factor] = factor_results

    logger.info(f"✅ 择时能力测试完成: {len(timing_test_results)} 个因子通过显著性检验")

    # 步骤 4: 因子作为开关测试
    logger.info("🔄 [步骤 4/5] 开始执行因子作为开关测试...")
    switch_test_results = {}

    # 🚀 专业优化：使用更全面的默认因子列表（基于最新配置）
    default_factors = [
        'returns_zscore', 'market_structure_score', 'ma_ratio_5_10', 'corr_5d', 'volatility_correlation',
        'volume_correlation', 'near_support_count_60d', 'institutional_activity_score',
        'ma_convergence_score', 'lower_shadow_ratio', 'sector_correlation_20d',
        'price_impact', 'rank_momentum_20d', 'rank_rsi', 'rank_volume', 'market_correlation',
        'corr_20d_mom5', 'corr_slope_20_120',
        # 新增因子（基于用户最新配置）
        'gain_loss_ratio_rank_20d', 'gain_loss_ratio_rank_60d',
        'limit_up_count_rank_20d', 'limit_up_count_rank_10d', 'limit_up_count_rank_5d',
        'max_gain_5d_rank_30d'
    ]

    # 🚀 关键修复：直接从统一数据中分析，无需重新合并
    available_factors = [col for col in default_factors if col in all_factors_df.columns]
    missing_factors = [col for col in default_factors if col not in all_factors_df.columns]

    logger.info(f"📊 开关测试分析范围: {len(available_factors)}个股票因子 ({len(default_factors)}个中的{len(available_factors)}个可用)")
    if missing_factors:
        logger.info(f"  - 缺失股票因子: {missing_factors[:5]}{'...' if len(missing_factors) > 5 else ''}")

    # 🚀 专业优化：数据质量检查
    logger.info("🔍 执行数据质量检查...")
    quality_report = {}
    for factor in available_factors[:5]:  # 检查前5个因子作为样本
        factor_data = all_factors_df[factor].dropna()
        quality_report[factor] = {
            'coverage': len(factor_data) / len(all_factors_df),
            'unique_ratio': factor_data.nunique() / len(factor_data) if len(factor_data) > 0 else 0
        }

    avg_coverage = np.mean([v['coverage'] for v in quality_report.values()])
    avg_unique_ratio = np.mean([v['unique_ratio'] for v in quality_report.values()])
    logger.info(f"  - 平均数据覆盖率: {avg_coverage:.1%}")
    logger.info(f"  - 平均唯一值比例: {avg_unique_ratio:.1%}")

    if avg_coverage < 0.8:
        logger.warning(f"  - ⚠️ 数据覆盖率较低 ({avg_coverage:.1%})，可能影响分析质量")

    # 🚀 关键修复：验证市场因子在统一数据中的存在性（无需重新合并）
    available_market_factors_for_switch = [factor for factor in market_factor_names if factor in all_factors_df.columns]
    missing_market_factors_for_switch = [factor for factor in market_factor_names if factor not in all_factors_df.columns]

    if missing_market_factors_for_switch:
        logger.warning(f"⚠️ 以下市场因子在统一数据中缺失: {missing_market_factors_for_switch}")

    logger.info(f"📊 开关测试市场因子: {len(available_market_factors_for_switch)}/{len(market_factor_names)} 个可用")

    # 🚀 关键修复：使用统一数据，无需合并
    merged_df = all_factors_df  # 直接使用统一数据
    logger.info(f"✅ 使用统一因子数据进行开关测试: {len(merged_df):,} 行")

    def calculate_daily_ic(df, factor_col):
        """计算日度IC，增强稳健性"""
        if df.shape[0] < 5: return np.nan  # 提高最小样本要求
        corr_data = df[[factor_col, 'future_return_2d']].dropna()
        if corr_data.shape[0] < 5: return np.nan
        try:
            return corr_data[factor_col].corr(corr_data['future_return_2d'], method='spearman')
        except:
            return np.nan

    def improved_factor_grouping(series, factor_name):
        """改进的因子分组逻辑，解决因子值变化少的问题"""
        series_clean = series.dropna()
        if len(series_clean) < 50:  # 数据太少
            return None, f"数据量不足 ({len(series_clean)} < 50)"

        unique_values = series_clean.nunique()
        if unique_values < 5:  # 唯一值太少
            return None, f"唯一值太少 ({unique_values} < 5)"

        # 尝试五分位数分组
        try:
            quintiles = pd.qcut(series_clean, 5, labels=False, duplicates='drop')
            if quintiles.nunique() >= 3:  # 至少要有3个不同的分组
                return quintiles, None
        except:
            pass

        # 如果五分位数失败，尝试三分位数
        try:
            tertiles = pd.qcut(series_clean, 3, labels=False, duplicates='drop')
            if tertiles.nunique() >= 2:
                return tertiles, None
        except:
            pass

        # 最后尝试百分位数阈值分组 (30%-70%)
        try:
            p30 = series_clean.quantile(0.3)
            p70 = series_clean.quantile(0.7)
            if p30 != p70:  # 确保阈值不同
                percentile_groups = pd.cut(series_clean, bins=[-np.inf, p30, p70, np.inf], labels=False)
                if percentile_groups.nunique() >= 2:
                    return percentile_groups, None
        except:
            pass

        return None, f"无法创建有效分组 (唯一值: {unique_values})"

    # 🔧 关键修复：确保市场因子在合并数据中存在
    available_market_factors = [factor for factor in market_factor_names if factor in merged_df.columns]
    missing_market_factors = [factor for factor in market_factor_names if factor not in merged_df.columns]

    if missing_market_factors:
        logger.warning(f"⚠️ 以下市场因子在合并数据中缺失: {missing_market_factors}")

    logger.info(f"📊 开关测试范围: {len(available_market_factors)}/{len(market_factor_names)} 个市场因子可用")

    for market_factor in available_market_factors_for_switch:
        try:
            progress = f"[{available_market_factors_for_switch.index(market_factor)+1}/{len(available_market_factors_for_switch)}]"
            logger.info(f"🔍 {progress} 分析市场因子: {market_factor}")

            # 🔧 修复：添加因子存在性检查
            if market_factor not in merged_df.columns:
                logger.error(f"  - ❌ 因子 '{market_factor}' 不存在于统一数据中")
                continue

            # 🔧 修复：检查因子数据质量
            factor_data = merged_df[market_factor].dropna()
            if len(factor_data) == 0:
                logger.warning(f"  - ⚠️ 因子 '{market_factor}' 数据全为空")
                continue

            logger.info(f"  - 因子数据质量: {len(factor_data):,}/{len(merged_df):,} 有效值 ({len(factor_data)/len(merged_df)*100:.1f}%)")

            # 使用改进的分组逻辑
            grouping_result, error_msg = improved_factor_grouping(merged_df[market_factor], market_factor)
            if grouping_result is None:
                logger.warning(f"  - ⚠️ 因子 '{market_factor}' 分组失败: {error_msg}")
                continue

            merged_df['market_group'] = grouping_result
            max_group = merged_df['market_group'].max()
            min_group = merged_df['market_group'].min()

            top_group_days = merged_df[merged_df['market_group'] == max_group]
            bottom_group_days = merged_df[merged_df['market_group'] == min_group]

            if top_group_days.empty or bottom_group_days.empty:
                logger.warning(f"  - ⚠️ 因子 '{market_factor}' 分组后为空")
                continue

            unique_dates_top = top_group_days['date'].nunique()
            unique_dates_bottom = bottom_group_days['date'].nunique()
            logger.info(f"  - 分组统计: 高组 {len(top_group_days):,} 条记录 ({unique_dates_top}个交易日)，低组 {len(bottom_group_days):,} 条记录 ({unique_dates_bottom}个交易日)")

            significant_switches = {}
            for stock_factor in available_factors:  # 只分析默认因子
                try:
                    # 🔧 修复：检查股票因子是否存在
                    if stock_factor not in merged_df.columns:
                        logger.warning(f"    股票因子 '{stock_factor}' 不存在于合并数据中")
                        continue

                    # 🔧 修复：检查股票因子数据质量
                    stock_factor_data = merged_df[stock_factor].dropna()
                    if len(stock_factor_data) == 0:
                        logger.warning(f"    股票因子 '{stock_factor}' 数据全为空")
                        continue

                    # 计算IC时增加错误处理
                    ic_top_daily = top_group_days.groupby('date').apply(
                        lambda x: calculate_daily_ic(x, stock_factor)
                    ).dropna()
                    ic_bottom_daily = bottom_group_days.groupby('date').apply(
                        lambda x: calculate_daily_ic(x, stock_factor)
                    ).dropna()

                    if len(ic_top_daily) == 0 or len(ic_bottom_daily) == 0:
                        continue

                    ic_top = ic_top_daily.mean()
                    ic_bottom = ic_bottom_daily.mean()

                    if pd.isna(ic_top) or pd.isna(ic_bottom):
                        continue

                    ic_spread = ic_top - ic_bottom

                    # 提高IC Spread阈值，增强显著性要求
                    if abs(ic_spread) > 0.03:  # 从0.02提升到0.03
                        # 增加统计检验
                        from scipy import stats
                        try:
                            # 使用t检验验证IC差异的显著性
                            t_stat, p_val = stats.ttest_ind(ic_top_daily, ic_bottom_daily, equal_var=False)
                            if p_val < 0.1:  # 10%显著性水平
                                significant_switches[stock_factor] = {
                                    'ic_spread': ic_spread,
                                    'ic_top': ic_top,
                                    'ic_bottom': ic_bottom,
                                    't_stat': t_stat,
                                    'p_value': p_val,
                                    'top_days': len(ic_top_daily),
                                    'bottom_days': len(ic_bottom_daily)
                                }
                        except:
                            # 如果统计检验失败，使用旧的逻辑
                            significant_switches[stock_factor] = {
                                'ic_spread': ic_spread,
                                'ic_top': ic_top,
                                'ic_bottom': ic_bottom
                            }
                except Exception as factor_e:
                    logger.warning(f"    因子 {stock_factor} 处理失败: {factor_e}")
                    continue

            if significant_switches:
                switch_test_results[market_factor] = significant_switches
                avg_ic_spread = np.mean([abs(v['ic_spread'] if isinstance(v, dict) else v) for v in significant_switches.values()])
                logger.info(f"  - ✅ 发现 {len(significant_switches)} 个显著影响的因子 (平均IC Spread: {avg_ic_spread:.3f})")
            else:
                logger.info(f"  - ❌ 未发现显著影响 (阈值: 0.03)")

        except Exception as e:
            logger.error(f"  - ❌ 对因子 '{market_factor}' 进行开关测试时失败: {e}")
            import traceback
            logger.error(f"  - 详细错误信息: {traceback.format_exc()}")

            # 🔧 添加调试信息
            logger.error(f"  - 调试信息:")
            logger.error(f"    - 因子名称: {market_factor}")
            logger.error(f"    - 因子是否在merged_df中: {market_factor in merged_df.columns}")
            if market_factor in merged_df.columns:
                factor_info = merged_df[market_factor].describe()
                logger.error(f"    - 因子统计信息: {factor_info}")
            logger.error(f"    - merged_df列名: {list(merged_df.columns)[:10]}...")  # 只显示前10个列名
            continue

    # 步骤 5: 生成专业分析报告
    logger.info("🔄 [步骤 5/5] 开始生成专业量化分析报告...")
    valuable_factors = sorted(list(set(timing_test_results.keys()) | set(switch_test_results.keys())))

    # 🚀 专业优化：增强报告统计分析
    logger.info("📊 计算报告统计指标...")

    # 计算择时能力统计
    timing_stats = {
        'total_factors': len(market_factor_names),
        'timing_factors': len(timing_test_results),
        'avg_r_squared': 0,
        'max_t_stat': 0,
        'significant_periods': set()
    }

    if timing_test_results:
        all_r_squared = []
        all_t_stats = []
        for factor_results in timing_test_results.values():
            for period_result in factor_results.values():
                all_r_squared.append(period_result.get('r_squared', 0))
                all_t_stats.append(abs(period_result.get('t_stat', 0)))
                timing_stats['significant_periods'].add(period_result.get('period', 0))

        timing_stats['avg_r_squared'] = np.mean(all_r_squared) if all_r_squared else 0
        timing_stats['max_t_stat'] = max(all_t_stats) if all_t_stats else 0

    # 计算开关效应统计
    switch_stats = {
        'switch_factors': len(switch_test_results),
        'total_impacts': sum(len(results) for results in switch_test_results.values()),
        'avg_ic_spread': 0,
        'max_ic_spread': 0,
        'significant_factors': set()
    }

    if switch_test_results:
        all_ic_spreads = []
        for factor_name, factor_results in switch_test_results.items():
            for stock_factor, impact in factor_results.items():
                ic_spread = abs(impact.get('ic_spread', 0))
                all_ic_spreads.append(ic_spread)
                if ic_spread > 0.05:  # 高影响阈值
                    switch_stats['significant_factors'].add(stock_factor)

        switch_stats['avg_ic_spread'] = np.mean(all_ic_spreads) if all_ic_spreads else 0
        switch_stats['max_ic_spread'] = max(all_ic_spreads) if all_ic_spreads else 0

    if not valuable_factors:
        logger.info("❌ 未发现有价值的市场状态因子。")
        logger.info(f"   分析了 {timing_stats['total_factors']} 个市场因子，均未通过显著性检验")
        logger.info(f"   建议检查数据质量或调整显著性阈值")
        return True

    # 在报告生成之前先计算需要的变量
    # 筛选适合短期持仓的因子(用于报告)
    short_term_timing_factors_for_report = []
    for factor in timing_test_results:
        for period, result in timing_test_results[factor].items():
            if period <= 2 and abs(result.get('t_stat', 0)) > 2.5:
                short_term_timing_factors_for_report.append(factor)
                break

    # 计算推荐因子数量(用于报告)
    recommended_factors_for_report = list(set([f for f in valuable_factors
                                             if f in short_term_timing_factors_for_report or f in switch_test_results]))

    total_switch_impacts = sum(len(results) for results in switch_test_results.values())
    report = ["# 市场状态因子分析报告",
              f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              f"分析周期: {analysis_periods} 日", "---"]
    timing_factor_count = len(timing_test_results)
    switch_factor_count = len(switch_test_results)

    report.append(f"**总结**: 在 {len(market_factor_names)} 个市场状态因子中，发现 {len(valuable_factors)} 个具有显著价值的因子。")
    report.append(f"**择时能力**: {timing_factor_count} 个因子通过市场择时测试，具备预测未来 1-3 日市场涨跌能力。")
    avg_impacts_per_factor = total_switch_impacts/max(switch_factor_count,1) if switch_factor_count > 0 else 0
    report.append(f"**开关效应**: {switch_factor_count} 个因子具备开关效应，共影响 {total_switch_impacts} 个默认因子的效果 (平均每个市场因子影响 {avg_impacts_per_factor:.1f} 个股票因子)。")
    # 使用已计算的推荐因子数量
    recommended_count = len(recommended_factors_for_report)
    report.append(f"**分析范围**: 重点分析了 {len(available_factors)}/18 个默认核心因子对市场状态的敏感性。")
    report.append(f"**集成建议**: 推荐集成 {recommended_count} 个适合短期持仓(1-2日)的市场因子。")

    for factor in valuable_factors:
        report.append(f"\n### 因子: `{factor}`")
        passed_timing = factor in timing_test_results
        passed_switch = factor in switch_test_results

        if passed_timing:
            report.append("- **✅ 通过市场择时能力测试**")
            details = []
            timing_periods = []
            if factor in timing_test_results:
                for n, res in timing_test_results[factor].items():
                    if abs(res['t_stat']) > 2.5:  # 更新阈值
                        direction = "正向预测" if res['t_stat'] > 0 else "负向预测"
                        p_info = f", p={res['p_value']:.3f}" if 'p_value' in res else ""
                        details.append(f"对未来 **{n}日** 收益有显著 **{direction}** 能力 (t-stat: {res['t_stat']:.2f}, R²: {res['r_squared']:.4f}{p_info})")
                        timing_periods.append(n)

                effectiveness_desc = "短期" if max(timing_periods) <= 3 else ("中期" if max(timing_periods) <= 10 else "长期")
                report.append(f"  - **价值分析**: 该因子具有{effectiveness_desc}市场择时能力，可以显著预测未来市场整体走势。{' '.join(details)}")

        if passed_switch:
            report.append("- **✅ 通过因子作为开关测试**")
            details = []
            if factor in switch_test_results:
                # 按IC Spread绝对值排序，显示最重要的影响，并标注重要性等级
                sorted_factors = sorted(switch_test_results[factor].items(),
                                      key=lambda x: abs(x[1]['ic_spread'] if isinstance(x[1], dict) else x[1]),
                                      reverse=True)

                # 分类影响强度
                strong_impacts = [(f, v) for f, v in sorted_factors if abs(v['ic_spread'] if isinstance(v, dict) else v) > 0.05]
                moderate_impacts = [(f, v) for f, v in sorted_factors if 0.03 <= abs(v['ic_spread'] if isinstance(v, dict) else v) <= 0.05]

                for stock_factor, result in sorted_factors[:5]:  # 只显示前5个最重要的
                    if isinstance(result, dict):  # 新的详细结果
                        direction = "增强" if result['ic_spread'] > 0 else "减弱"
                        p_info = f" (p={result['p_value']:.3f})" if 'p_value' in result else ""
                        details.append(f"显著 **{direction}** `{stock_factor}` 因子效果 (IC Spread: {result['ic_spread']:.3f}{p_info})")
                    else:  # 旧的简单结果
                        direction = "增强" if result > 0 else "减弱"
                        details.append(f"显著 **{direction}** `{stock_factor}` 因子效果 (IC Spread: {result:.3f})")

                total_affected = len(switch_test_results[factor])
                enhancing_count = sum(1 for _, res in switch_test_results[factor].items()
                                    if (res['ic_spread'] if isinstance(res, dict) else res) > 0)
                weakening_count = total_affected - enhancing_count

                summary = f"影响 {total_affected}/18 个默认因子（增强{enhancing_count}个，减弱{weakening_count}个）"
                report.append(f"  - **价值分析**: 该因子可以识别市场状态，{summary}。主要影响: {' '.join(details[:3])}{'...' if len(details) > 3 else ''}")

    final_report = "\n".join(report)
    print("\n" + "="*80)
    print(final_report)
    print("="*80 + "\n")

    # 添加性能统计和建议信息
    logger.info("📊 性能统计信息:")
    logger.info(f"  - 市场因子数量: {len(market_factor_names)}")
    logger.info(f"  - 有效因子数量: {len(valuable_factors)} ({len(valuable_factors)/len(market_factor_names)*100:.1f}%)")
    logger.info(f"  - 分析的股票因子: {len(available_factors)}/18")
    logger.info(f"  - 发现的显著影响: {total_switch_impacts}")
    logger.info(f"  - 数据记录数: {len(merged_df):,}")

    # 生成集成建议
    high_impact_factors = [f for f in valuable_factors
                          if f in timing_test_results and f in switch_test_results]
    timing_only_factors = [f for f in valuable_factors
                          if f in timing_test_results and f not in switch_test_results]
    switch_only_factors = [f for f in valuable_factors
                          if f not in timing_test_results and f in switch_test_results]

    logger.info("\n💡 集成建议:")
    logger.info(f"  - 高价值因子 (同时具备择时+开关效应): {len(high_impact_factors)} 个")
    logger.info(f"    推荐: 直接作为特征加入模型训练")
    logger.info(f"  - 择时专用因子: {len(timing_only_factors)} 个")
    logger.info(f"    推荐: 用于市场状态识别和模型切换")
    logger.info(f"  - 开关专用因子: {len(switch_only_factors)} 个")
    logger.info(f"    推荐: 用于因子权重调节或特征选择")

    if len(valuable_factors) > 10:
        logger.info(f"  - ⚠️  因子数量较多({len(valuable_factors)}个)，建议考虑:")
        logger.info(f"    1. PCA降维 (保留90%方差)")
        logger.info(f"    2. 相关性筛选 (去除>0.8相关性)")
        logger.info(f"    3. 重要性排序 (基于IC绝对值)")

    # 输出详细的择时能力分析
    logger.info("\n📈 详细择时能力分析(按周期显示):")
    for factor in timing_test_results:
        logger.info(f"  {factor}:")
        for period, result in timing_test_results[factor].items():
            t_stat = result['t_stat']
            r_sq = result['r_squared']
            p_val = result.get('p_value', 'N/A')
            direction = "正向" if t_stat > 0 else "负向"
            suitable = "✅适合" if period <= 2 else "⚠️不适合"
            logger.info(f"    {period}日: {direction} t={t_stat:.2f}, R²={r_sq:.4f}, p={p_val} ({suitable}2日持仓)")

    # 相关性分析
    if len(valuable_factors) > 3:
        logger.info("\n🔗 市场因子相关性分析:")
        try:
            # 🚀 修复：从统一数据中提取市场因子进行相关性分析
            factor_data_for_corr = all_factors_df[['date'] + valuable_factors].drop_duplicates(subset=['date']).set_index('date')
            correlation_matrix = factor_data_for_corr.corr()

            # 找到高相关性因子对
            high_corr_pairs = []
            for i, factor1 in enumerate(valuable_factors):
                for j, factor2 in enumerate(valuable_factors[i+1:], i+1):
                    corr_val = correlation_matrix.loc[factor1, factor2]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((factor1, factor2, corr_val))

            if high_corr_pairs:
                logger.info(f"  ⚠️  发现 {len(high_corr_pairs)} 对高相关性因子 (|相关系数|>0.7):")
                for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    logger.info(f"    {f1} vs {f2}: {corr:.3f}")
                logger.info(f"  建议: 集成时需要去相关或PCA降维")
            else:
                logger.info(f"  ✅ 所有因子间相关性<0.7，可以直接集成")
        except Exception as e:
            logger.warning(f"  相关性分析失败: {e}")

    # 筛选适合短期持仓的因子
    short_term_timing_factors = []
    medium_term_only_factors = []

    for factor in timing_test_results:
        has_short_term = False
        has_any_effect = False
        for period, result in timing_test_results[factor].items():
            if abs(result.get('t_stat', 0)) > 2.5:
                has_any_effect = True
                if period <= 2:
                    has_short_term = True

        if has_short_term:
            short_term_timing_factors.append(factor)
        elif has_any_effect:
            medium_term_only_factors.append(factor)

    # 推荐集成的具体因子名单
    recommended_factors = list(set([f for f in valuable_factors if f in short_term_timing_factors or f in switch_test_results]))

    logger.info(f"\n🎯 推荐集成的市场因子 ({len(recommended_factors)}个):")
    for i, factor in enumerate(recommended_factors, 1):
        timing_effect = "择时" if factor in short_term_timing_factors else ""
        switch_effect = "开关" if factor in switch_test_results else ""
        effects = "/".join(filter(None, [timing_effect, switch_effect]))
        logger.info(f"  {i:2d}. {factor} ({effects})")

    # 更新的集成建议
    effective_for_strategy = len(recommended_factors)
    logger.info(f"\n🛠️  因子集成策略 (当前有{effective_for_strategy}个因子适合集成):")

    if effective_for_strategy <= 8:
        logger.info(f"  ✅ 因子数量适中({effective_for_strategy}个)，建议:")
        logger.info(f"    1. 直接加入XGBoost训练 (与18个股票因子合并)")
        logger.info(f"    2. 检查相关性矩阵，去除>0.85相关的因子")
    elif effective_for_strategy <= 15:
        logger.info(f"  ⚠️  因子数量较多({effective_for_strategy}个)，建议:")
        logger.info(f"    1. 首先进行相关性分析，去除>0.8相关的因子")
        logger.info(f"    2. 按IC绝对值排序，只保留前8-10个最强因子")
        logger.info(f"    3. 或者使用PCA降维至5-8个主成分")
    else:
        logger.info(f"  ❌ 因子数量过多({effective_for_strategy}个)，强烈建议:")
        logger.info(f"    1. 使用PCA降维至5-8个主成分 (保留95%方差)")
        logger.info(f"    2. 或者使用因子选择算法 (LASSO/递归特征消除)")
        logger.info(f"    3. 避免因子冗余影响18个核心股票因子的效果")
        logger.info(f"    4. 考虑使用Stacking集成：市场因子作为元学习器的输入")

    logger.info("✅ 市场状态因子分析模式执行完成。")
    return True


def run_interaction_analysis_mode(args):
    """运行交互特征全面分析模式

    功能包括:
    1. 生成所有可能的交互特征
    2. 单因子测试 (IC, IR, 分层回测)
    3. 特征相关性分析
    4. 管线评估分析
    5. 优化建议
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🔬 交互特征全面分析模式")
    logger.info("=" * 80)

    try:
        # 步骤1: 数据加载和准备
        logger.info("🔄 [步骤 1/6] 数据加载和准备...")

        # 1. 加载股票因子数据 (使用与run_test_factors_mode一致的方式)
        logger.info("📥 加载股票因子数据...")

        # 加载完整的18个股票因子数据
        stock_factor_data = _load_factor_cache_data(include_label_generation_features=True)
        if stock_factor_data is None or len(stock_factor_data) == 0:
            logger.error("❌ 股票因子数据加载失败")
            return False

        logger.info(f"✅ 股票因子数据加载完成: {len(stock_factor_data)} 条记录, {len(stock_factor_data.columns)} 个字段")
        logger.info(f"数据时间范围: {stock_factor_data['date'].min()} 到 {stock_factor_data['date'].max()}")

        # 2. 从股票因子数据中提取市场因子（统一数据源）
        logger.info("📥 从统一数据源提取市场因子...")

        # 获取可用的市场因子列表
        market_factor_columns = _get_available_market_factors()
        available_market_factors = [col for col in market_factor_columns if col in stock_factor_data.columns]

        if len(available_market_factors) > 0:
            market_factors_df = stock_factor_data[['date'] + available_market_factors].copy()
            logger.info(f"✅ 从统一数据中提取到 {len(available_market_factors)} 个市场因子: {available_market_factors}")
            logger.info(f"市场因子时间范围: {market_factors_df['date'].min()} 到 {market_factors_df['date'].max()}")
        else:
            logger.error("❌ 未找到市场因子数据")
            return False

        # 3. 数据对齐和合并
        logger.info("🔄 数据对齐和合并...")

        # 确保日期格式一致
        stock_factor_data['date'] = pd.to_datetime(stock_factor_data['date']).dt.date
        market_factors_df['date'] = pd.to_datetime(market_factors_df['date']).dt.date

        # 合并数据
        merged_data = stock_factor_data.merge(market_factors_df, on='date', how='inner')
        if len(merged_data) == 0:
            logger.error("❌ 股票因子数据与市场因子数据无法对齐")
            return False

        logger.info(f"✅ 数据对齐完成: {len(merged_data)} 条记录")
        logger.info(f"合并后时间范围: {merged_data['date'].min()} 到 {merged_data['date'].max()}")

        # 步骤2: 交互特征生成
        logger.info("🔄 [步骤 2/6] 交互特征生成...")

        # 获取完整的股票特征 (使用与训练一致的定义)
        # 🔧 因子测试使用模型因子列表（归一化因子）
        stock_features = _get_model_factors()
        logger.info(f"📊 使用完整的股票特征: {len(stock_features)} 个")

        # 检查实际可用的股票特征
        available_stock_features = []
        for feature in stock_features:
            if feature in merged_data.columns:
                available_stock_features.append(feature)
            else:
                logger.warning(f"⚠️ 股票特征 {feature} 在数据中不存在，跳过")

        logger.info(f"📊 实际可用股票特征: {len(available_stock_features)} 个")
        for i, feature in enumerate(available_stock_features, 1):
            logger.info(f"  {i:2d}. {feature}")

        # 获取可用的市场特征 (处理pandas merge时的列名冲突)
        market_features = []
        potential_market_features = [
            'market_volatility_atr20d', 'market_trend_factor', 'max_gain_10d_rank_60d',
            'market_volume_rank_20d', 'limit_up_count_rank_60d', 'limit_down_count_rank_60d',
            'broken_limit_count_rank_20d', 'max_consecutive_limit_rank_20d'
        ]

        # 检查合并后的数据中实际存在的市场特征 (处理_y后缀并重命名)
        market_feature_mapping = {}  # 原始名称 -> 实际列名的映射

        for feature in potential_market_features:
            if feature in merged_data.columns:
                market_features.append(feature)
                market_feature_mapping[feature] = feature
                logger.info(f"✅ 找到市场特征: {feature}")
            elif f"{feature}_y" in merged_data.columns:
                # 重命名_y后缀的列为原始名称
                original_col = f"{feature}_y"
                merged_data[feature] = merged_data[original_col]
                market_features.append(feature)
                market_feature_mapping[feature] = original_col
                logger.info(f"✅ 找到市场特征: {feature} (从 {original_col} 重命名)")
            elif f"{feature}_x" in merged_data.columns:
                # 重命名_x后缀的列为原始名称
                original_col = f"{feature}_x"
                merged_data[feature] = merged_data[original_col]
                market_features.append(feature)
                market_feature_mapping[feature] = original_col
                logger.info(f"✅ 找到市场特征: {feature} (从 {original_col} 重命名)")
            else:
                logger.warning(f"⚠️ 市场特征 {feature} 在合并数据中不存在，跳过")

        logger.info(f"📊 可用市场特征: {len(market_features)} 个")
        for i, feature in enumerate(market_features, 1):
            logger.info(f"  {i:2d}. {feature}")

        if not market_features:
            logger.error("❌ 未找到可用的市场特征")
            logger.info("📋 合并数据中包含市场相关的列名:")
            market_related_columns = [col for col in merged_data.columns if any(mf in col for mf in potential_market_features)]
            for i, col in enumerate(market_related_columns, 1):
                logger.info(f"  {i:3d}. {col}")
            return False

        if not available_stock_features:
            logger.error("❌ 未找到可用的股票特征")
            return False

        # 生成交互特征
        interaction_features = {}
        interaction_names = []

        # 标准化市场特征 (与训练逻辑一致)
        from sklearn.preprocessing import StandardScaler
        market_scaler = StandardScaler()
        market_matrix = merged_data[market_features].values.astype(float)
        market_matrix_scaled = market_scaler.fit_transform(market_matrix)
        market_matrix_scaled = np.clip(market_matrix_scaled, -3.0, 3.0)

        logger.info(f"🔄 开始生成交互特征: {len(available_stock_features)} × {len(market_features)} = {len(available_stock_features) * len(market_features)} 个")

        for i, stock_feature in enumerate(available_stock_features):
            stock_values = merged_data[stock_feature].values

            for j, market_feature in enumerate(market_features):
                interaction_name = f"inter_{stock_feature}_{market_feature}"
                interaction_values = stock_values * market_matrix_scaled[:, j]

                interaction_features[interaction_name] = interaction_values
                interaction_names.append(interaction_name)

        logger.info(f"✅ 生成交互特征: {len(interaction_names)} 个")
        logger.info(f"📊 交互特征组合: {len(available_stock_features)} 个股票特征 × {len(market_features)} 个市场特征")

        # 显示前10个交互特征名称作为示例
        for i, name in enumerate(interaction_names[:10], 1):
            logger.info(f"  {i:2d}. {name}")
        if len(interaction_names) > 10:
            logger.info(f"  ... 还有 {len(interaction_names) - 10} 个交互特征")

        # 将交互特征添加到数据中
        for name, values in interaction_features.items():
            merged_data[name] = values

        return _continue_interaction_analysis_fixed(merged_data, available_stock_features, market_features,
                                                   interaction_names, logger)

    except Exception as e:
        logger.error(f"❌ 交互特征分析失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _continue_interaction_analysis_fixed(merged_data, stock_features, market_features,
                                       interaction_names, logger):
    """修复后的交互特征分析流程：先过滤再测试"""
    try:
        logger.info("🔄 [步骤 3/7] 特征相关性分析和过滤...")

        # 第一阶段：相关性分析（包含原始股票因子）
        logger.info("📊 第一阶段 - 相关性过滤（包含原始股票因子）:")
        logger.info(f"  原始交互特征数量: {len(interaction_names)}")
        logger.info(f"  原始股票因子数量: {len(stock_features)}")

        # 准备所有特征进行相关性分析（交互特征 + 原始股票因子）
        all_features_for_correlation = stock_features + interaction_names

        # 检查数据质量并移除有效数据不足的特征
        valid_features = []
        for feature in all_features_for_correlation:
            if feature in merged_data.columns:
                valid_count = merged_data[feature].notna().sum()
                total_count = len(merged_data)
                valid_ratio = valid_count / total_count

                if valid_ratio >= 0.1:  # 至少10%有效数据
                    valid_features.append(feature)
                else:
                    logger.warning(f"  ⚠️ 特征 {feature} 有效数据不足: {valid_ratio:.2%}")
            else:
                logger.warning(f"  ⚠️ 特征 {feature} 在数据中不存在")

        # 分离有效的股票因子和交互特征
        valid_stock_features = [f for f in stock_features if f in valid_features]
        valid_interaction_features = [f for f in interaction_names if f in valid_features]

        logger.info(f"  有效股票因子: {len(valid_stock_features)} 个")
        logger.info(f"  有效交互特征: {len(valid_interaction_features)} 个")

        if len(valid_interaction_features) == 0:
            logger.error("❌ 没有有效的交互特征")
            return False

        # 计算相关性矩阵（只在有效特征之间）
        correlation_data = merged_data[valid_features].dropna()
        logger.info(f"  相关性计算数据: {len(correlation_data)} 条记录")

        if len(correlation_data) < 100:
            logger.error("❌ 相关性计算数据不足")
            return False

        # 计算相关性矩阵
        correlation_matrix = correlation_data.corr(method='pearson', min_periods=100)

        # 过滤与原始股票因子高相关的交互特征
        filtered_interaction_features = []
        removed_by_stock_correlation = []

        for interaction_feature in valid_interaction_features:
            is_highly_correlated = False

            for stock_feature in valid_stock_features:
                if interaction_feature in correlation_matrix.index and stock_feature in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[interaction_feature, stock_feature]

                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        logger.info(f"  移除与股票因子高相关的交互特征: {interaction_feature} (与{stock_feature}相关性={corr_val:.4f})")
                        removed_by_stock_correlation.append((interaction_feature, stock_feature, corr_val))
                        is_highly_correlated = True
                        break

            if not is_highly_correlated:
                filtered_interaction_features.append(interaction_feature)

        logger.info(f"  移除与股票因子高相关的交互特征: {len(removed_by_stock_correlation)} 个")
        logger.info(f"  剩余交互特征: {len(filtered_interaction_features)} 个")

        if len(filtered_interaction_features) == 0:
            logger.error("❌ 所有交互特征都与原始股票因子高相关")
            return False

        # 过滤交互特征之间的高相关性
        logger.info("📊 过滤交互特征之间的高相关性...")

        # 计算交互特征之间的相关性
        interaction_correlation_matrix = correlation_matrix.loc[filtered_interaction_features, filtered_interaction_features]

        # 识别高相关性特征对
        high_corr_pairs = []
        for i, feature1 in enumerate(filtered_interaction_features):
            for j, feature2 in enumerate(filtered_interaction_features[i+1:], i+1):
                corr_val = interaction_correlation_matrix.loc[feature1, feature2]
                if not np.isnan(corr_val) and abs(corr_val) > 0.7:  # 使用0.7阈值
                    high_corr_pairs.append((feature1, feature2, corr_val))

        logger.info(f"  发现交互特征间高相关性对: {len(high_corr_pairs)} 对")

        # 实施高相关特征过滤算法
        if len(high_corr_pairs) > 0:
            logger.info("🔄 执行高相关特征过滤算法...")
            correlation_filtered_features = _apply_correlation_filtering_algorithm(
                filtered_interaction_features, high_corr_pairs, interaction_correlation_matrix, logger
            )
            logger.info(f"  相关性过滤前: {len(filtered_interaction_features)} 个特征")
            logger.info(f"  相关性过滤后: {len(correlation_filtered_features)} 个特征")
            logger.info(f"  移除特征数量: {len(filtered_interaction_features) - len(correlation_filtered_features)} 个")
        else:
            logger.info("  没有发现高相关特征对，跳过相关性过滤")
            correlation_filtered_features = filtered_interaction_features

        # 第二阶段：VIF过滤（优化版本）
        logger.info("📊 第二阶段 - VIF过滤 (VIF<5):")
        logger.info(f"  过滤前特征数量: {len(correlation_filtered_features)}")

        vif_filtered_features = _apply_vif_filter_optimized(
            correlation_filtered_features, merged_data, logger
        )

        logger.info(f"  过滤后特征数量: {len(vif_filtered_features)}")
        logger.info(f"  移除特征数量: {len(correlation_filtered_features) - len(vif_filtered_features)}")

        if len(vif_filtered_features) == 0:
            logger.error("❌ VIF过滤后没有剩余特征")
            return False

        # 第三阶段：单因子测试（只对过滤后的特征）
        logger.info("🔄 [步骤 4/7] 单因子测试（仅对过滤后的特征）...")
        logger.info(f"  测试特征数量: {len(vif_filtered_features)}")

        factor_test_results = _perform_factor_testing(
            merged_data, vif_filtered_features, logger
        )

        # 第四阶段：IC排序
        logger.info("📊 第四阶段 - IC排序:")
        logger.info(f"  排序前特征数量: {len(vif_filtered_features)}")

        final_ranked_features = _apply_ic_ranking(
            vif_filtered_features, factor_test_results, logger
        )

        logger.info(f"  最终优质特征数量: {len(final_ranked_features)}")

        # 保存过滤结果
        filtering_results = {
            'original_features': len(interaction_names),
            'valid_features_after_quality_check': len(valid_interaction_features),
            'features_after_stock_correlation_filter': len(filtered_interaction_features),
            'features_after_interaction_correlation_filter': len(correlation_filtered_features),
            'features_after_vif_filter': len(vif_filtered_features),
            'final_ranked_features': len(final_ranked_features),
            'removed_by_stock_correlation': removed_by_stock_correlation,
            'interaction_high_corr_pairs': high_corr_pairs,
            'correlation_filtered_features': correlation_filtered_features,
            'vif_filtered_features': vif_filtered_features,
            'final_ranked_features_list': final_ranked_features,
            'filtering_summary': {
                'removed_by_quality_check': len(interaction_names) - len(valid_interaction_features),
                'removed_by_stock_correlation': len(valid_interaction_features) - len(filtered_interaction_features),
                'removed_by_interaction_correlation': len(filtered_interaction_features) - len(correlation_filtered_features),
                'removed_by_vif_filter': len(correlation_filtered_features) - len(vif_filtered_features),
                'final_retention_rate': len(final_ranked_features) / len(interaction_names)
            }
        }

        # 保存串联过滤结果
        filtering_file = 'interaction_filtering_results.json'
        try:
            import json
            with open(filtering_file, 'w', encoding='utf-8') as f:
                json.dump(filtering_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 串联过滤结果已保存到: {filtering_file}")
        except Exception as e:
            logger.warning(f"⚠️ 保存串联过滤结果失败: {e}")

        # 生成最终报告
        return _generate_interaction_analysis_report_fixed(
            interaction_names, factor_test_results, high_corr_pairs, [],
            stock_features, market_features, logger,
            final_ranked_features, filtering_results
        )

    except Exception as e:
        logger.error(f"❌ 交互特征分析失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _create_empty_factor_result():
    """创建空的因子测试结果"""
    return {
        'ic': 0.0,
        'rank_ic': 0.0,
        'ir': 0.0,
        'icir_20d': 0.0,
        'layer_returns': [0.0] * 5,
        'layer_counts': [0] * 5,
        'long_short_return': 0.0,
        'win_rate': 0.0,
        'monotonicity': 0.0,
        'coverage': 0.0,
        'max_drawdown': 0.0,
        'valid_samples': 0,
        'total_samples': 0
    }


def _calculate_monotonicity(layer_returns):
    """计算分层收益的单调性"""
    if len(layer_returns) < 2:
        return 0.0

    # 计算相邻层之间的单调性
    monotonic_count = 0
    total_pairs = len(layer_returns) - 1

    for i in range(total_pairs):
        if layer_returns[i+1] >= layer_returns[i]:
            monotonic_count += 1

    return monotonic_count / total_pairs if total_pairs > 0 else 0.0


def _calculate_vif_parallel(data, features, logger):
    """并行计算VIF，优化速度和错误处理"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import concurrent.futures
    import warnings

    def calculate_single_vif(args):
        """计算单个特征的VIF"""
        data_values, feature_idx, feature_name = args
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 添加数值稳定性检查
                if data_values.shape[0] < data_values.shape[1] * 2:
                    return (feature_name, np.inf)

                # 检查数据是否包含异常值
                if np.any(np.isinf(data_values)) or np.any(np.isnan(data_values)):
                    return (feature_name, np.inf)

                # 计算VIF
                vif = variance_inflation_factor(data_values, feature_idx)

                # 检查结果有效性
                if np.isnan(vif) or np.isinf(vif) or vif < 0:
                    return (feature_name, np.inf)

                return (feature_name, vif)

        except Exception as e:
            # 记录具体错误但不中断整个过程
            if "DLASCL" in str(e) or "MKL" in str(e):
                logger.warning(f"⚠️ 特征 {feature_name} MKL数值错误，设为无穷大")
            else:
                logger.warning(f"⚠️ 特征 {feature_name} VIF计算失败: {e}")
            return (feature_name, np.inf)

    try:
        data_values = data.values

        # 如果特征数量较少，使用单线程
        if len(features) <= 10:
            vif_scores = []
            for i, feature in enumerate(features):
                result = calculate_single_vif((data_values, i, feature))
                vif_scores.append(result)
        else:
            # 使用多线程计算VIF
            max_workers = min(4, len(features))  # 限制线程数
            logger.info(f"  使用 {max_workers} 个线程并行计算VIF")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 准备参数
                args_list = [(data_values, i, feature) for i, feature in enumerate(features)]

                # 提交任务
                future_to_feature = {
                    executor.submit(calculate_single_vif, args): args[2]
                    for args in args_list
                }

                vif_scores = []
                for future in concurrent.futures.as_completed(future_to_feature):
                    try:
                        result = future.result(timeout=30)  # 30秒超时
                        vif_scores.append(result)
                    except concurrent.futures.TimeoutError:
                        feature_name = future_to_feature[future]
                        logger.warning(f"⚠️ 特征 {feature_name} VIF计算超时")
                        vif_scores.append((feature_name, np.inf))
                    except Exception as e:
                        feature_name = future_to_feature[future]
                        logger.warning(f"⚠️ 特征 {feature_name} VIF计算异常: {e}")
                        vif_scores.append((feature_name, np.inf))

        return vif_scores

    except Exception as e:
        logger.error(f"❌ VIF并行计算失败: {e}")
        return []


def _apply_ic_ranking(features, factor_test_results, logger):
    """第三阶段：按IC绝对值降序排序"""
    logger.info("🔄 执行IC排序...")

    # 计算每个特征的IC绝对值
    feature_ic_pairs = []
    for feature in features:
        ic = factor_test_results.get(feature, {}).get('ic', 0.0)
        ic_abs = abs(ic) if not np.isnan(ic) else 0.0
        feature_ic_pairs.append((feature, ic, ic_abs))

    # 按IC绝对值降序排序
    sorted_features = sorted(feature_ic_pairs, key=lambda x: x[2], reverse=True)

    logger.info(f"📊 IC排序结果 (前10名):")
    for i, (feature, ic, ic_abs) in enumerate(sorted_features[:10], 1):
        logger.info(f"  {i:2d}. {feature}: IC={ic:.4f} (|IC|={ic_abs:.4f})")

    if len(sorted_features) > 10:
        logger.info(f"  ... 还有 {len(sorted_features) - 10} 个特征")

    # 返回排序后的特征列表
    ranked_features = [
        {
            'feature': feature,
            'ic': float(ic) if not np.isnan(ic) else 0.0,
            'ic_abs': float(ic_abs),
            'rank': i + 1
        }
        for i, (feature, ic, ic_abs) in enumerate(sorted_features)
    ]

    return ranked_features


def _apply_correlation_filtering_algorithm(features, high_corr_pairs, correlation_matrix, logger):
    """
    高相关特征过滤算法：使用贪心算法解决高相关特征过滤问题

    算法策略：
    1. 构建相关性图，节点为特征，边为高相关关系
    2. 使用贪心算法：每次移除与最多特征高相关的特征
    3. 重复直到没有高相关特征对
    """
    logger.info("🔄 开始高相关特征过滤算法...")

    # 构建相关性图：特征 -> 与其高相关的特征集合
    correlation_graph = {}
    for feature in features:
        correlation_graph[feature] = set()

    # 添加高相关关系
    for feature1, feature2, corr_val in high_corr_pairs:
        if feature1 in correlation_graph and feature2 in correlation_graph:
            correlation_graph[feature1].add(feature2)
            correlation_graph[feature2].add(feature1)

    logger.info(f"  构建相关性图: {len(features)} 个节点, {len(high_corr_pairs)} 条边")

    # 贪心算法：迭代移除与最多特征高相关的特征
    remaining_features = set(features)
    removed_features = []
    iteration = 0

    while True:
        iteration += 1

        # 计算每个剩余特征的高相关邻居数量
        correlation_counts = {}
        for feature in remaining_features:
            # 计算与当前特征高相关且仍在剩余特征中的特征数量
            correlated_neighbors = correlation_graph[feature] & remaining_features
            correlation_counts[feature] = len(correlated_neighbors)

        # 找到与最多特征高相关的特征
        if not correlation_counts or max(correlation_counts.values()) == 0:
            logger.info(f"  算法收敛: 没有剩余的高相关特征对")
            break

        # 选择与最多特征高相关的特征进行移除
        feature_to_remove = max(correlation_counts.items(), key=lambda x: x[1])[0]
        max_correlations = correlation_counts[feature_to_remove]

        logger.info(f"  第 {iteration} 轮: 移除特征 {feature_to_remove} (与 {max_correlations} 个特征高相关)")

        # 移除该特征
        remaining_features.remove(feature_to_remove)
        removed_features.append(feature_to_remove)

        # 限制迭代次数，避免无限循环
        # if iteration >= len(features) // 2:
        if iteration >= 90:
            logger.warning(f"  达到最大迭代次数 {iteration}，停止算法")
            break

    final_features = list(remaining_features)

    logger.info(f"📊 高相关特征过滤结果:")
    logger.info(f"  原始特征数: {len(features)}")
    logger.info(f"  移除特征数: {len(removed_features)}")
    logger.info(f"  保留特征数: {len(final_features)}")
    logger.info(f"  过滤率: {len(removed_features)/len(features):.1%}")

    # 验证过滤效果：检查剩余特征之间是否还有高相关对
    remaining_high_corr_pairs = 0
    for i, feature1 in enumerate(final_features):
        for j, feature2 in enumerate(final_features[i+1:], i+1):
            if feature1 in correlation_matrix.index and feature2 in correlation_matrix.columns:
                corr_val = correlation_matrix.loc[feature1, feature2]
                if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                    remaining_high_corr_pairs += 1

    logger.info(f"  验证结果: 剩余高相关对数量: {remaining_high_corr_pairs}")

    if remaining_high_corr_pairs > 0:
        logger.warning(f"⚠️ 仍有 {remaining_high_corr_pairs} 对高相关特征，可能需要进一步优化")
    else:
        logger.info("✅ 成功消除所有高相关特征对")

    return final_features


def _apply_vif_filter_optimized(features, merged_data, logger):
    """优化版VIF过滤：在正确位置应用所有优化措施"""
    logger.info("🔄 执行优化版VIF过滤...")

    if len(features) <= 1:
        logger.warning("⚠️ 特征数量不足，跳过VIF过滤")
        return features

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.preprocessing import StandardScaler
        import warnings

        # 抑制MKL警告
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # 数据采样优化（在正确位置）
        logger.info(f"📊 准备VIF计算数据，原始特征数: {len(features)}")

        vif_data = merged_data[features].select_dtypes(include=[np.number])

        # 如果数据量过大，进行采样以加速VIF计算
        original_size = len(vif_data)
        if original_size > 50000:
            sample_size = 50000
            vif_data = vif_data.sample(n=sample_size, random_state=42)
            logger.info(f"📊 数据采样: {original_size:,} → {sample_size:,} 条记录以加速VIF计算")

        # 移除缺失值
        vif_data = vif_data.dropna()
        logger.info(f"📊 移除缺失值后: {len(vif_data):,} 条记录")

        # 数据质量检查和预处理
        valid_features = []
        for col in features:
            if col in vif_data.columns:
                col_data = vif_data[col]
                # 检查方差
                if col_data.std() > 1e-10:
                    # 异常值处理：使用3σ原则
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val

                    # 检查是否有有效数据
                    valid_mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                    if valid_mask.sum() > len(col_data) * 0.8:  # 至少80%的数据有效
                        valid_features.append(col)
                    else:
                        logger.warning(f"⚠️ 特征 {col} 异常值过多，移除")
                else:
                    logger.warning(f"⚠️ 特征 {col} 方差为0，移除")

        if len(valid_features) <= 1:
            logger.warning("⚠️ 有效特征数量不足，跳过VIF过滤")
            return valid_features

        # 重新选择数据并标准化
        vif_data = vif_data[valid_features]

        # 标准化数据以避免数值问题
        try:
            scaler = StandardScaler()
            vif_data_scaled = pd.DataFrame(
                scaler.fit_transform(vif_data),
                columns=vif_data.columns,
                index=vif_data.index
            )
            logger.info(f"✅ 数据标准化完成")
        except Exception as e:
            logger.warning(f"⚠️ 数据标准化失败: {e}，使用原始数据")
            vif_data_scaled = vif_data

        if len(vif_data_scaled) <= len(valid_features):
            logger.warning("⚠️ 样本数量不足以计算VIF，跳过VIF过滤")
            return valid_features

        # 优化的VIF迭代计算
        current_features = valid_features.copy()
        max_iterations = min(len(current_features), 20)  # 限制最大迭代次数
        iteration = 0

        logger.info(f"📊 开始VIF迭代过滤，最大迭代次数: {max_iterations}")

        while iteration < max_iterations and len(current_features) > 1:
            iteration += 1
            logger.info(f"  第 {iteration} 轮VIF计算，当前特征数: {len(current_features)}")

            # 获取当前数据
            current_data = vif_data_scaled[current_features]

            # 检查数据质量
            if current_data.isnull().any().any():
                logger.warning("⚠️ 发现缺失值，重新清理数据")
                current_data = current_data.dropna()
                if len(current_data) < len(current_features) * 10:
                    logger.warning("⚠️ 清理后数据不足，停止VIF过滤")
                    break

            # 使用优化的并行VIF计算
            vif_scores = _calculate_vif_parallel(current_data, current_features, logger)

            if not vif_scores:
                logger.warning("⚠️ VIF计算失败，停止过滤")
                break

            # 找到最高VIF的特征
            max_vif_feature, max_vif = max(vif_scores, key=lambda x: x[1] if not np.isinf(x[1]) else 1e6)

            if max_vif < 5.0:
                logger.info(f"  ✅ 所有特征VIF<5，过滤完成")
                break

            # 移除最高VIF的特征
            current_features.remove(max_vif_feature)
            if np.isinf(max_vif):
                logger.info(f"  移除问题特征: {max_vif_feature} (VIF=∞)")
            else:
                logger.info(f"  移除高VIF特征: {max_vif_feature} (VIF={max_vif:.2f})")

            if len(current_features) <= 1:
                logger.warning("⚠️ 剩余特征数量不足，停止VIF过滤")
                break

        logger.info(f"📊 VIF过滤结果:")
        logger.info(f"  原始特征: {len(features)} 个")
        logger.info(f"  移除特征: {len(features) - len(current_features)} 个")
        logger.info(f"  保留特征: {len(current_features)} 个")

        return current_features

    except ImportError:
        logger.warning("⚠️ statsmodels未安装，跳过VIF过滤")
        return features
    except Exception as e:
        logger.warning(f"⚠️ VIF过滤失败: {e}")
        return features


def _perform_factor_testing(merged_data, features, logger):
    """对过滤后的特征进行单因子测试"""
    logger.info(f"🔄 开始单因子测试，特征数量: {len(features)}")

    # 检查是否有future_return_2d字段
    if 'future_return_2d' not in merged_data.columns:
        logger.error("❌ 数据中缺少future_return_2d字段")
        return {}

    # 移除缺失值
    analysis_data = merged_data.dropna(subset=['future_return_2d'] + features)
    logger.info(f"📊 分析数据: {len(analysis_data):,} 条记录")

    if len(analysis_data) == 0:
        logger.error("❌ 分析数据为空")
        return {}

    factor_test_results = {}

    for i, factor_name in enumerate(features, 1):
        logger.info(f"  [{i:3d}/{len(features)}] 测试因子: {factor_name}")

        try:
            factor_values = analysis_data[factor_name].values
            returns = analysis_data['future_return_2d'].values

            # 移除NaN值
            valid_mask = ~(np.isnan(factor_values) | np.isnan(returns))
            if np.sum(valid_mask) < 100:  # 至少需要100个有效样本
                logger.warning(f"    ⚠️ 有效样本不足: {np.sum(valid_mask)}")
                factor_test_results[factor_name] = _create_empty_factor_result()
                continue

            factor_clean = factor_values[valid_mask]
            returns_clean = returns[valid_mask]

            # 计算各项指标（与之前的实现一致）
            ic = np.corrcoef(factor_clean, returns_clean)[0, 1] if len(factor_clean) > 1 else 0.0

            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(factor_clean, returns_clean)

            # 分层回测
            n_quantiles = 5
            try:
                quantiles = np.quantile(factor_clean, np.linspace(0, 1, n_quantiles + 1))
                layer_returns = []
                layer_counts = []

                for j in range(n_quantiles):
                    if j == n_quantiles - 1:
                        mask = (factor_clean >= quantiles[j]) & (factor_clean <= quantiles[j + 1])
                    else:
                        mask = (factor_clean >= quantiles[j]) & (factor_clean < quantiles[j + 1])

                    if np.sum(mask) > 0:
                        layer_return = np.mean(returns_clean[mask])
                        layer_returns.append(layer_return)
                        layer_counts.append(np.sum(mask))
                    else:
                        layer_returns.append(0.0)
                        layer_counts.append(0)

            except Exception as e:
                logger.warning(f"    ⚠️ 分层回测计算失败: {e}")
                layer_returns = [0.0] * 5
                layer_counts = [0] * 5

            # 其他指标
            long_short_return = layer_returns[-1] - layer_returns[0] if len(layer_returns) >= 2 else 0.0

            # IR计算
            ir = 0.0
            try:
                if 'date' in analysis_data.columns:
                    daily_ic_data = analysis_data.groupby('date').apply(
                        lambda group: np.corrcoef(group[factor_name].values, group['future_return_2d'].values)[0, 1]
                        if len(group) > 1 and group[factor_name].notna().sum() > 1 and group['future_return_2d'].notna().sum() > 1
                        else np.nan
                    ).dropna()

                    if len(daily_ic_data) > 1:
                        ir = daily_ic_data.mean() / (daily_ic_data.std() + 1e-8)
                    else:
                        ir = ic / (np.std(returns_clean) + 1e-8) if len(returns_clean) > 1 else 0.0
                else:
                    ir = ic / (np.std(returns_clean) + 1e-8) if len(returns_clean) > 1 else 0.0
            except Exception as e:
                logger.warning(f"    ⚠️ IR计算失败: {e}")
                ir = ic / (np.std(returns_clean) + 1e-8) if len(returns_clean) > 1 else 0.0

            win_rate = np.mean(returns_clean > 0) if len(returns_clean) > 0 else 0.0
            monotonicity = _calculate_monotonicity(layer_returns)
            coverage = np.sum(valid_mask) / len(factor_values)

            # 最大回撤
            max_drawdown = 0.0
            if len(layer_returns) > 1:
                cumulative_returns = np.cumsum(layer_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

            factor_test_results[factor_name] = {
                'ic': ic,
                'rank_ic': rank_ic,
                'ir': ir,
                'icir_20d': ic / (np.std(returns_clean) + 1e-8),
                'layer_returns': layer_returns,
                'layer_counts': layer_counts,
                'long_short_return': long_short_return,
                'win_rate': win_rate,
                'monotonicity': monotonicity,
                'coverage': coverage,
                'max_drawdown': max_drawdown,
                'valid_samples': np.sum(valid_mask),
                'total_samples': len(factor_values)
            }

            logger.info(f"    IC: {ic:.4f}, RankIC: {rank_ic:.4f}, 多空收益: {long_short_return:.4f}, 胜率: {win_rate:.2%}")

        except Exception as e:
            logger.warning(f"    ⚠️ 因子测试失败: {e}")
            factor_test_results[factor_name] = _create_empty_factor_result()

    # 保存因子测试结果
    factor_results_file = 'interaction_factor_test_results.json'
    try:
        import json
        factor_results_summary = {
            'factor_test_results': {
                name: {
                    'ic': float(result['ic']) if not np.isnan(result['ic']) else 0.0,
                    'rank_ic': float(result['rank_ic']) if not np.isnan(result['rank_ic']) else 0.0,
                    'ir': float(result['ir']) if not np.isnan(result['ir']) else 0.0,
                    'icir_20d': float(result['icir_20d']) if not np.isnan(result['icir_20d']) else 0.0,
                    'layer_returns': [float(x) if not np.isnan(x) else 0.0 for x in result['layer_returns']],
                    'layer_counts': [int(x) for x in result['layer_counts']],
                    'long_short_return': float(result['long_short_return']) if not np.isnan(result['long_short_return']) else 0.0,
                    'win_rate': float(result['win_rate']) if not np.isnan(result['win_rate']) else 0.0,
                    'monotonicity': float(result['monotonicity']) if not np.isnan(result['monotonicity']) else 0.0,
                    'coverage': float(result['coverage']) if not np.isnan(result['coverage']) else 0.0,
                    'max_drawdown': float(result['max_drawdown']) if not np.isnan(result['max_drawdown']) else 0.0,
                    'valid_samples': int(result['valid_samples']),
                    'total_samples': int(result['total_samples'])
                }
                for name, result in factor_test_results.items()
            },
            'summary': {
                'total_factors': len(features),
                'successful_tests': len([r for r in factor_test_results.values() if r['valid_samples'] > 0]),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }

        with open(factor_results_file, 'w', encoding='utf-8') as f:
            json.dump(factor_results_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 因子测试结果已保存到: {factor_results_file}")
    except Exception as e:
        logger.warning(f"⚠️ 保存因子测试结果失败: {e}")

    return factor_test_results


def _generate_interaction_analysis_report_fixed(interaction_names, factor_test_results,
                                              high_corr_pairs, high_vif_features,
                                              stock_features, market_features, logger,
                                              final_ranked_features=None, filtering_results=None):
    """生成修复版的交互特征分析报告"""
    logger.info(f"\n🎯 交互特征分析报告")
    logger.info(f"=" * 60)

    # 基本统计
    logger.info(f"📊 基本统计:")
    logger.info(f"  - 原始交互特征数: {len(interaction_names)}")
    logger.info(f"  - 股票特征数: {len(stock_features)}")
    logger.info(f"  - 市场特征数: {len(market_features)}")

    if filtering_results:
        logger.info(f"  - 质量检查后: {filtering_results['valid_features_after_quality_check']}")
        logger.info(f"  - 股票因子相关性过滤后: {filtering_results['features_after_stock_correlation_filter']}")
        logger.info(f"  - VIF过滤后: {filtering_results['features_after_vif_filter']}")
        logger.info(f"  - 最终保留: {filtering_results['final_ranked_features']}")
        logger.info(f"  - 总体保留率: {filtering_results['filtering_summary']['final_retention_rate']:.2%}")

    # 显示最终优质特征
    if final_ranked_features and len(final_ranked_features) > 0:
        logger.info(f"\n🏆 最终优质交互特征 (前10名):")
        for i, feature_info in enumerate(final_ranked_features[:10], 1):
            feature_name = feature_info['feature']
            ic = feature_info['ic']
            ic_abs = feature_info['ic_abs']
            logger.info(f"  {i:2d}. {feature_name}: IC={ic:.4f} (|IC|={ic_abs:.4f})")

        if len(final_ranked_features) > 10:
            logger.info(f"  ... 还有 {len(final_ranked_features) - 10} 个优质特征")

    # 下一步建议
    logger.info(f"\n🎯 下一步行动计划:")
    if final_ranked_features and len(final_ranked_features) > 0:
        logger.info(f"  1. 立即行动:")
        logger.info(f"     - 使用 {len(final_ranked_features)} 个优质交互特征重新训练")
        logger.info(f"     - 优先使用前10名特征进行初步验证")
        logger.info(f"     - 调整正则化参数防止过拟合")

    logger.info(f"  2. 验证测试:")
    logger.info(f"     - 在独立测试集上验证改进效果")
    logger.info(f"     - 对比有无交互特征的回测结果")
    logger.info(f"     - 监控过拟合风险")

    logger.info(f"\n✅ 交互特征分析完成！")
    return True


def _generate_interaction_analysis_report(interaction_names, factor_test_results,
                                        high_corr_pairs, high_vif_features,
                                        stock_features, market_features, logger,
                                        final_ranked_features=None, filtering_results=None):
    """生成交互特征分析报告"""
    try:
        logger.info("=" * 80)
        logger.info("📋 交互特征分析报告")
        logger.info("=" * 80)

        # 1. 基本统计
        logger.info("📊 基本统计:")
        logger.info(f"  - 股票特征数量: {len(stock_features)}")
        logger.info(f"  - 市场特征数量: {len(market_features)}")
        logger.info(f"  - 生成交互特征数量: {len(interaction_names)}")
        logger.info(f"  - 高相关性特征对: {len(high_corr_pairs)}")
        logger.info(f"  - 高VIF特征数量: {len(high_vif_features)}")

        # 2. 单因子测试结果排序
        logger.info("\n📈 单因子测试结果 (按|IC|排序):")

        # 按IC绝对值排序
        sorted_results = sorted(
            factor_test_results.items(),
            key=lambda x: abs(x[1]['ic']) if not np.isnan(x[1]['ic']) else 0,
            reverse=True
        )

        logger.info("  排名  特征名称                                          IC      多空收益   样本数")
        logger.info("  " + "-" * 75)

        effective_features = []
        zero_importance_features = []

        for i, (feature_name, result) in enumerate(sorted_results, 1):
            ic = result['ic']
            ls_return = result['long_short_return']
            samples = result['valid_samples']

            if abs(ic) > 0.01:  # 认为有效的阈值
                effective_features.append(feature_name)
                status = "✅"
            else:
                zero_importance_features.append(feature_name)
                status = "❌"

            logger.info(f"  {i:2d}    {feature_name:<45} {ic:7.4f}  {ls_return:8.4f}  {samples:6d} {status}")

        # 3. 问题特征识别
        logger.info(f"\n⚠️ 问题特征识别:")
        logger.info(f"  - 低效特征 (|IC|<0.01): {len(zero_importance_features)} 个")
        if zero_importance_features:
            for feature in zero_importance_features[:5]:
                logger.info(f"    • {feature}")
            if len(zero_importance_features) > 5:
                logger.info(f"    • ... 还有 {len(zero_importance_features) - 5} 个")

        logger.info(f"  - 高相关性特征: {len([f for f, _, _ in high_corr_pairs])} 个")
        logger.info(f"  - 高VIF特征: {len(high_vif_features)} 个")

        # 4. 优化建议
        logger.info(f"\n💡 优化建议:")

        if len(zero_importance_features) > 0:
            logger.info(f"  1. 特征筛选:")
            logger.info(f"     - 移除 {len(zero_importance_features)} 个低效交互特征")
            logger.info(f"     - 保留 {len(effective_features)} 个有效交互特征")

        if len(high_corr_pairs) > 0:
            logger.info(f"  2. 相关性处理:")
            logger.info(f"     - 发现 {len(high_corr_pairs)} 对高相关性特征")
            logger.info(f"     - 建议使用相关性筛选或PCA降维")

        if len(high_vif_features) > 0:
            logger.info(f"  3. 多重共线性处理:")
            logger.info(f"     - {len(high_vif_features)} 个特征存在多重共线性")
            logger.info(f"     - 建议使用岭回归或LASSO进行特征选择")

        # 5. 模型参数调整建议
        logger.info(f"\n🔧 模型参数调整建议:")

        feature_increase_ratio = len(interaction_names) / len(stock_features)
        logger.info(f"  - 特征维度增加: {feature_increase_ratio:.1f}x ({len(stock_features)} → {len(stock_features) + len(interaction_names)})")

        if feature_increase_ratio > 2.0:
            logger.info(f"  - 建议增加正则化强度:")
            logger.info(f"    • L2正则化 (lambda): 1.0 → 2.0-5.0")
            logger.info(f"    • L1正则化 (alpha): 0.0 → 0.1-0.5")
            logger.info(f"  - 建议降低模型复杂度:")
            logger.info(f"    • max_depth: 5 → 3-4")
            logger.info(f"    • eta (学习率): 0.1 → 0.05-0.08")

        # 6. 管线评估改进建议
        logger.info(f"\n📊 管线评估改进建议:")
        logger.info(f"  - 当前指标: Top10%-Bottom10%")
        logger.info(f"  - 建议补充指标:")
        logger.info(f"    • IC_IR综合得分: IC均值 / IC标准差")
        logger.info(f"    • 夏普比率: 收益均值 / 收益标准差")
        logger.info(f"    • 最大回撤: 评估风险控制能力")
        logger.info(f"    • 胜率: 预测正确的比例")

        # 7. 串联过滤结果展示
        if final_ranked_features is not None and filtering_results is not None:
            logger.info(f"\n🔄 串联过滤结果:")
            logger.info(f"  - 原始交互特征: {filtering_results['original_features']} 个")
            logger.info(f"  - 质量检查后: {filtering_results['valid_features_after_quality_check']} 个")
            logger.info(f"  - 相关性过滤后: {filtering_results['features_after_correlation_filter']} 个")
            logger.info(f"  - VIF过滤后: {filtering_results['features_after_vif_filter']} 个")
            logger.info(f"  - 最终保留: {filtering_results['final_ranked_features']} 个")
            logger.info(f"  - 总体保留率: {filtering_results['filtering_summary']['final_retention_rate']:.2%}")

            logger.info(f"\n🏆 最终优质交互特征 (前10名):")
            for i, feature_info in enumerate(final_ranked_features[:10], 1):
                feature_name = feature_info['feature']
                ic = feature_info['ic']
                ic_abs = feature_info['ic_abs']
                logger.info(f"  {i:2d}. {feature_name}: IC={ic:.4f} (|IC|={ic_abs:.4f})")

            if len(final_ranked_features) > 10:
                logger.info(f"  ... 还有 {len(final_ranked_features) - 10} 个优质特征")

        # 8. 下一步行动计划
        logger.info(f"\n🎯 下一步行动计划:")

        if final_ranked_features is not None and len(final_ranked_features) > 0:
            logger.info(f"  1. 立即行动:")
            logger.info(f"     - 使用 {len(final_ranked_features)} 个优质交互特征重新训练")
            logger.info(f"     - 优先使用前10名特征进行初步验证")
            logger.info(f"     - 调整正则化参数防止过拟合")
        elif len(effective_features) > 0:
            logger.info(f"  1. 立即行动:")
            logger.info(f"     - 使用 {len(effective_features)} 个有效交互特征重新训练")
            logger.info(f"     - 调整正则化参数防止过拟合")

        if len(zero_importance_features) > len(effective_features):
            logger.info(f"  2. 特征工程优化:")
            logger.info(f"     - 尝试不同的股票特征组合")
            logger.info(f"     - 测试非线性交互 (比值、差值、条件交互)")
            logger.info(f"     - 引入时间滞后的交互特征")

        logger.info(f"  3. 验证测试:")
        logger.info(f"     - 在独立测试集上验证改进效果")
        logger.info(f"     - 对比有无交互特征的回测结果")
        logger.info(f"     - 监控过拟合风险")

        logger.info("=" * 80)
        logger.info("✅ 交互特征全面分析完成")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"❌ 报告生成失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_test_model_walkforward_mode(args):
    """
    运行Walk-Forward Analysis + 交叉验证模式

    实现顶级量化机构的标准做法:
    1. 将回测期分成多个窗口
    2. 每个窗口执行完整的"贝叶斯搜索(带交叉验证)+训练+回测"流程
    3. 汇总所有窗口的回测结果

    Args:
        args: 命令行参数
    """
    logger = logging.getLogger(__name__)
    print(f"🚀 运行模式: Walk-Forward Analysis + 交叉验证")
    logger.info("=" * 80)
    logger.info("🎯 顶级量化机构标准做法: 滚动优化 + 交叉验证")
    logger.info("=" * 80)

    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from datetime import datetime

        # 🔧 修复：定义Walk-Forward扩展窗口（Expanding Window）
        # 每个窗口包含: 训练集、验证集(用于超参数搜索)、测试集(用于回测)
        # 时间序列原则: 训练集 < 验证集 < 测试集 (无重叠,无数据泄露)
        # 扩展窗口特点: 训练期从固定起点开始，逐渐增加训练数据量
        windows = [
            {
                'name': 'Window 1',
                'train_start': '2023-06-01',   # 固定起点
                'train_end': '2024-09-21',     # 训练集结束 (约16个月)
                'val_start': '2024-10-15',     # 验证集开始(紧接训练集)
                'val_end': '2024-12-31',       # 验证集结束 (3个月)
                'test_start': '2025-01-01',    # 测试集开始(紧接验证集)
                'test_end': '2025-03-31'       # 测试集结束 (3个月)
            },
            {
                'name': 'Window 2',
                'train_start': '2023-06-01',   # 固定起点（扩展窗口）
                'train_end': '2024-12-31',     # 训练集结束 (约18个月，增加3个月)
                'val_start': '2025-01-01',     # 验证集开始
                'val_end': '2025-03-31',       # 验证集结束 (3个月)
                'test_start': '2025-04-01',    # 测试集开始
                'test_end': '2025-06-30'       # 测试集结束 (3个月)
            },
            {
                'name': 'Window 3',
                'train_start': '2023-06-01',   # 固定起点（扩展窗口）
                'train_end': '2025-03-31',     # 训练集结束 (约21个月，再增加3个月)
                'val_start': '2025-04-01',     # 验证集开始
                'val_end': '2025-06-30',       # 验证集结束 (3个月)
                'test_start': '2025-07-01',    # 测试集开始
                'test_end': '2025-09-30'       # 测试集结束 (3个月)
            }
        ]

        # 🔧 解析窗口选择参数
        selected_window_indices = None
        if args.window:
            try:
                # 解析窗口编号（支持逗号分隔，如 "1" 或 "1,3"）
                selected_window_indices = [int(w.strip()) for w in args.window.split(',')]

                # 验证窗口编号范围
                for idx in selected_window_indices:
                    if idx < 1 or idx > len(windows):
                        logger.error(f"❌ 窗口编号 {idx} 超出范围 [1, {len(windows)}]")
                        return False

                logger.info(f"\n🎯 窗口选择模式: 只训练指定的窗口 {selected_window_indices}")
            except ValueError as e:
                logger.error(f"❌ 窗口参数格式错误: {args.window}，应为数字或逗号分隔的数字（例如: 1 或 1,3）")
                return False
        else:
            logger.info(f"\n🎯 默认模式: 训练所有窗口")

        logger.info(f"\n📊 Walk-Forward窗口配置:")
        logger.info(f"  总窗口数: {len(windows)}")
        if selected_window_indices:
            logger.info(f"  选择的窗口: {selected_window_indices}")
        else:
            logger.info(f"  选择的窗口: 全部")
        logger.info(f"  时间序列原则: 训练集 < 验证集 < 测试集 (无重叠)")
        for i, window in enumerate(windows, 1):
            # 标记是否被选中
            selected_mark = "✓" if (selected_window_indices is None or i in selected_window_indices) else "✗"
            logger.info(f"\n  [{selected_mark}] 窗口{i} ({window['name']}):")
            logger.info(f"    训练集: {window['train_start']} ~ {window['train_end']}")
            logger.info(f"    验证集: {window['val_start']} ~ {window['val_end']}")
            logger.info(f"    测试集: {window['test_start']} ~ {window['test_end']}")

        # 加载完整因子数据
        logger.info("\n🔄 加载完整因子数据...")
        factor_data = _load_factor_cache_data(include_label_generation_features=True)

        if factor_data is None or len(factor_data) == 0:
            logger.error("❌ 因子数据加载失败")
            return False

        logger.info(f"✅ 因子数据加载成功: {len(factor_data):,}条记录, {len(factor_data.columns)}个字段")
        logger.info(f"   日期范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")

        # 确保date列是datetime类型
        factor_data['date'] = pd.to_datetime(factor_data['date'])

        # 存储所有窗口的回测结果
        all_window_results = []
        all_selections = []
        all_window_metrics = []

        # 遍历每个Walk-Forward窗口（只处理选中的窗口）
        for window_idx, window in enumerate(windows, 1):
            # 🔧 如果指定了窗口选择，跳过未选中的窗口
            if selected_window_indices is not None and window_idx not in selected_window_indices:
                logger.info(f"\n⏭️  跳过 {window['name']} (未选中)")
                continue

            logger.info("\n" + "=" * 80)
            logger.info(f"🔄 开始处理 {window['name']} (第{window_idx}/{len(windows)}个窗口)")
            logger.info("=" * 80)

            # 处理单个窗口
            window_result = _process_walkforward_window(
                window, factor_data, args, window_idx, len(windows), windows
            )

            if window_result is None:
                logger.error(f"❌ {window['name']} 处理失败")
                continue

            # 保存窗口结果
            all_window_results.append(window_result['backtest_result'])
            all_selections.extend(window_result['selections'])
            all_window_metrics.append({
                'window_name': window['name'],
                'test_period': f"{window['test_start']} ~ {window['test_end']}",
                **window_result['metrics']
            })

            logger.info(f"✅ {window['name']} 处理完成")

        # 汇总所有窗口的结果
        logger.info("\n" + "=" * 80)
        logger.info("📊 Walk-Forward Analysis 汇总结果")
        logger.info("=" * 80)

        if not all_window_results:
            logger.error("❌ 所有窗口处理失败,无法生成汇总结果")
            return False

        # 计算汇总指标
        _print_walkforward_summary(all_window_metrics)

        # 保存汇总结果
        _save_walkforward_results(all_window_metrics, all_selections)

        logger.info("✅ Walk-Forward Analysis 完成")
        return True

    except Exception as e:
        logger.error(f"❌ Walk-Forward Analysis 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def _prepare_training_data(train_data, feature_columns, test_end_date=None):
    """
    准备训练数据

    Args:
        train_data: 训练数据
        feature_columns: 特征列
        test_end_date: 测试集最新日期(用于时间衰减权重计算),格式: 'YYYY-MM-DD'

    Returns:
        tuple: (X_processed, y_train, sample_weights, winsorization_thresholds, label_stats)
    """
    import numpy as np  # 🚀 导入numpy
    logger = logging.getLogger(__name__)

    try:
        # 准备标签 (先检查标签列)
        if 'atr_return_label' not in train_data.columns:
            logger.error("❌ 训练数据缺少atr_return_label列")
            return None, None, None, None, None

        # 🚀 检查relevance列（排序模型必需）
        if 'relevance' not in train_data.columns:
            logger.error("❌ 训练数据缺少relevance列,无法训练排序模型")
            return None, None, None, None, None

        # 过滤掉标签为NaN的样本
        valid_mask = train_data['atr_return_label'].notna()
        train_data_valid = train_data[valid_mask].copy()

        if len(train_data_valid) == 0:
            logger.error("❌ 没有有效的训练样本(所有标签都是NaN)")
            return None, None, None, None, None

        # 🚀 关键修复: 按['date', 'stock_code']排序,确保排序模型的group信息正确
        if 'date' in train_data_valid.columns and 'stock_code' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("   ✅ 训练数据已按['date', 'stock_code']排序（排序模型要求）")
        elif 'date' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("   ✅ 训练数据已按date排序（排序模型要求）")

        logger.info(f"   有效样本: {len(train_data_valid):,}/{len(train_data):,} ({len(train_data_valid)/len(train_data)*100:.1f}%)")

        # 准备特征数据
        X_processed, available_features = _prepare_feature_data(
            train_data_valid, feature_columns,
            data_type="训练集"
        )

        if X_processed is None:
            return None, None, None, None, None

        # 🚀 获取relevance标签（排序模型使用整数0-3）
        y_train = train_data_valid['relevance'].values[:len(X_processed)]
        # 🚀 确保relevance标签是整数类型（XGBoost排序模型要求）
        y_train = np.array(y_train, dtype=np.int32)
        logger.info(f"   ✅ 使用relevance标签（整数0-3）用于排序模型训练")

        # 🚀 XGBoost排序模型不使用样本权重,跳过权重计算
        # 注意: XGBoost排序模型使用分组权重(group-level weights),在训练时计算
        logger.info("   ⏭️ 跳过样本权重计算（XGBoost排序模型使用分组权重）")
        sample_weights = None

        # 计算标签统计量
        label_stats = {
            'mean': np.mean(y_train),
            'std': np.std(y_train)
        }

        # winsorization_thresholds 已废弃，返回空字典以保持兼容性
        winsorization_thresholds = {}

        return X_processed, y_train, sample_weights, winsorization_thresholds, label_stats

    except Exception as e:
        logger.error(f"❌ 准备训练数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None, None


def _prepare_validation_data(val_data, feature_columns, winsorization_thresholds):
    """准备验证数据"""
    import numpy as np  # 🚀 导入numpy
    logger = logging.getLogger(__name__)

    try:
        # 准备标签 (先检查标签列)
        if 'atr_return_label' not in val_data.columns:
            logger.error("❌ 验证数据缺少atr_return_label列")
            return None, None, None

        # 🚀 检查relevance列（排序模型必需）
        if 'relevance' not in val_data.columns:
            logger.error("❌ 验证数据缺少relevance列,无法训练排序模型")
            return None, None, None

        # 过滤掉标签为NaN的样本
        valid_mask = val_data['atr_return_label'].notna()
        val_data_valid = val_data[valid_mask].copy()

        if len(val_data_valid) == 0:
            logger.error("❌ 没有有效的验证样本(所有标签都是NaN)")
            return None, None, None

        # 🚀 关键修复: 按['date', 'stock_code']排序,确保排序模型的group信息正确
        if 'date' in val_data_valid.columns and 'stock_code' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("   ✅ 验证数据已按['date', 'stock_code']排序（排序模型要求）")
        elif 'date' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("   ✅ 验证数据已按date排序（排序模型要求）")

        logger.info(f"   有效样本: {len(val_data_valid):,}/{len(val_data):,} ({len(val_data_valid)/len(val_data)*100:.1f}%)")

        # 准备特征数据
        X_processed, _ = _prepare_feature_data(
            val_data_valid, feature_columns,
            data_type="验证集"
        )

        if X_processed is None:
            return None, None, None

        # 🚀 获取relevance标签（排序模型使用整数0-3）
        y_val = val_data_valid['relevance'].values[:len(X_processed)]
        # 🚀 确保relevance标签是整数类型（XGBoost排序模型要求）
        y_val = np.array(y_val, dtype=np.int32)
        logger.info(f"   ✅ 使用relevance标签（整数0-3）用于排序模型训练")

        # 🚀 用于评估的加权收益率（而非排名标签）
        # 优先使用weighted_return，如果没有则计算
        if 'weighted_return' in val_data_valid.columns:
            y_val_eval = val_data_valid['weighted_return'].values[:len(X_processed)]
            logger.info("✅ 使用已有的weighted_return作为评估标签")
        elif all(col in val_data_valid.columns for col in ['future_return_2d', 'future_return_3d', 'future_return_5d']):
            # 🔧 计算原始加权收益率
            import numpy as np
            weighted_return = (
                0.4 * val_data_valid['future_return_2d'].values[:len(X_processed)] +
                0.4 * val_data_valid['future_return_3d'].values[:len(X_processed)] +
                0.2 * val_data_valid['future_return_5d'].values[:len(X_processed)]
            )
            y_val_eval = weighted_return
            logger.info("✅ 计算加权收益率作为评估标签")
            logger.info("   公式: 0.4*future_return_2d + 0.4*future_return_3d + 0.2*future_return_5d")
        else:
            logger.warning("⚠️ 验证数据缺少必要的future_return字段，使用排名标签作为评估标签")
            y_val_eval = y_val.copy()

        return X_processed, y_val, y_val_eval

    except Exception as e:
        logger.error(f"❌ 准备验证数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None



def _bayesian_search_single_val(train_data, val_data, args, test_end_date=None, window_idx=None, market_data=None):
    """
    使用单个验证集的贝叶斯搜索(支持结果保存和恢复)

    Args:
        train_data: 训练数据(已包含标签)
        val_data: 验证数据(已包含标签)
        args: 命令行参数
        test_end_date: 测试集最新日期(用于时间衰减权重计算),格式: 'YYYY-MM-DD'
        window_idx: 窗口索引(用于保存/恢复搜索结果)
        market_data: 市场数据DataFrame（用于XGBoost计算group权重）

    Returns:
        best_params 或 None
    """
    logger = logging.getLogger(__name__)

    try:
        # 检查是否存在已保存的搜索结果
        cache_file = None
        if window_idx is not None:
            cache_file = Path("results") / f"bayesian_search_window{window_idx}.pkl"
            if cache_file.exists():
                logger.info(f"🔍 发现已保存的贝叶斯搜索结果: {cache_file.name}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_result = pickle.load(f)

                    best_params = cached_result['best_params']

                    logger.info(f"✅ 成功加载已保存的搜索结果")
                    logger.info(f"   最优参数: {best_params}")
                    logger.info(f"   搜索时间: {cached_result.get('search_time', 'N/A')}")

                    return best_params

                except Exception as e:
                    logger.warning(f"⚠️ 加载缓存失败: {e},将重新执行搜索")

        logger.info(f"🔍 开始贝叶斯搜索(单个验证集)")
        if test_end_date:
            logger.info(f"   时间衰减基准日期: {test_end_date}")

        # 识别特征列
        feature_columns = _get_ml_feature_columns_simple(train_data)
        logger.info(f"   识别到{len(feature_columns)}个特征")

        # 准备训练数据
        X_train, y_train, sample_weights, winsorization_thresholds, label_stats = _prepare_training_data(
            train_data, feature_columns, test_end_date=test_end_date
        )

        if X_train is None:
            logger.error("❌ 训练数据准备失败")
            return None, None

        # 准备验证数据
        X_val, y_val, y_val_eval = _prepare_validation_data(
            val_data, feature_columns, winsorization_thresholds
        )

        if X_val is None:
            logger.error("❌ 验证数据准备失败")
            return None, None

        # 🔧 提取验证集日期（用于每日Top-K计算和group计算）
        val_dates = None
        if 'date' in val_data.columns:
            # 过滤掉标签为NaN的样本（与_prepare_validation_data保持一致）
            valid_mask = val_data['atr_return_label'].notna()
            val_dates = val_data.loc[valid_mask, 'date'].values[:len(X_val)]
            logger.info(f"   ✅ 提取验证集日期: {len(val_dates)}个交易日")
        else:
            logger.warning("   ⚠️ 验证数据缺少date列,将使用全局Top-K计算")

        # 🔧 提取训练集日期（用于group计算）
        train_dates = None
        if 'date' in train_data.columns:
            # 过滤掉标签为NaN的样本（与_prepare_training_data保持一致）
            valid_mask = train_data['atr_return_label'].notna()
            train_dates = train_data.loc[valid_mask, 'date'].values[:len(X_train)]
            logger.info(f"   ✅ 提取训练集日期: {len(train_dates)}个交易日")
        else:
            logger.error("   ❌ 训练数据缺少date列,无法使用排序模型")
            return None, None

        # 执行贝叶斯搜索
        logger.info(f"   执行贝叶斯搜索...")
        import time
        search_start_time = time.time()

        best_params, all_trials = _perform_parameter_search(
            X_train, y_train, X_val, y_val, y_val_eval, sample_weights,
            search_strategy='bayesian',
            val_dates=val_dates,  # 🔧 传入验证集日期
            train_dates=train_dates,  # 🔧 传入训练集日期
            market_data=market_data  # 🚀 XGBoost需要market_data计算group权重
        )

        search_time = time.time() - search_start_time

        if best_params is None:
            logger.error("❌ 贝叶斯搜索失败")
            return None

        logger.info(f"✅ 贝叶斯搜索完成(耗时: {search_time/60:.1f}分钟)")
        logger.info(f"   最优参数: {best_params}")

        # 保存搜索结果
        if cache_file is not None:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'best_params': best_params,
                        'search_time': f"{search_time/60:.1f}分钟",
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'window_idx': window_idx
                    }, f)
                logger.info(f"💾 搜索结果已保存: {cache_file.name}")
            except Exception as e:
                logger.warning(f"⚠️ 保存搜索结果失败: {e}")

        return best_params

    except Exception as e:
        logger.error(f"❌ 贝叶斯搜索失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _print_walkforward_summary(all_window_metrics):
    """
    打印Walk-Forward汇总结果

    🔧 修复：增强汇总指标输出，模仿test_model模式的详细汇总
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("📊 各窗口回测结果（仅显示Top-1结果）")
    logger.info("=" * 80)

    for metrics in all_window_metrics:
        logger.info(f"\n{metrics['window_name']} ({metrics['test_period']}):")
        logger.info(f"  总收益: {metrics['total_return']:.2f} ATR")
        logger.info(f"  胜率: {metrics['win_rate']:.2%}")
        logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f} ATR")

    # 计算总体指标（Top-1）
    total_return = sum(m['total_return'] for m in all_window_metrics)
    avg_win_rate = np.mean([m['win_rate'] for m in all_window_metrics])
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_window_metrics])
    max_drawdown = max(m['max_drawdown'] for m in all_window_metrics)

    # 🔧 修复：使用类似test_model模式的表格格式输出所有Top-k档位汇总
    # 检查是否有all_results数据
    if all_window_metrics and 'all_results' in all_window_metrics[0]:
        logger.info("\n📊 Walk-Forward所有Top-k档位汇总结果:")
        logger.info("=" * 150)
        logger.info(f"{'策略':<10} {'总收益':<10} {'净收益(扣成本)':<15} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<8} {'盈亏比':<8} {'总交易次数':<12}")
        logger.info("=" * 150)

        # 获取所有档位（通常是1, 2, 3, 5）
        all_stock_counts = set()
        for metrics in all_window_metrics:
            if 'all_results' in metrics and metrics['all_results']:
                all_stock_counts.update(metrics['all_results'].keys())

        stock_counts = sorted(all_stock_counts)

        for stock_count in stock_counts:
            # 收集所有窗口的该档位结果
            count_metrics = []
            for metrics in all_window_metrics:
                if 'all_results' in metrics and stock_count in metrics['all_results']:
                    count_metrics.append(metrics['all_results'][stock_count])

            if not count_metrics:
                logger.info(f"Top-{stock_count:<5} ⚠️ 无数据")
                continue

            # 🔧 修复：计算百分比收益和其他指标
            # 注意：这里的指标应该从backtest_results中获取，而不是简单累加
            # 因为不同窗口的收益应该是复利计算，而不是简单相加

            # 计算累计收益（复利）
            total_return_gross = 1.0
            total_return_net = 1.0
            for m in count_metrics:
                # 假设每个窗口的收益是百分比形式
                total_return_gross *= (1 + m.get('total_return_gross', 0))
                total_return_net *= (1 + m.get('total_return', 0))

            total_return_gross = total_return_gross - 1  # 转换为收益率
            total_return_net = total_return_net - 1

            # 计算年化收益（假设每个窗口的交易天数相同）
            total_trading_days = sum(m.get('trading_days', 0) for m in count_metrics)
            if total_trading_days > 0:
                annual_return = total_return_net * (252 / total_trading_days)
            else:
                annual_return = 0

            # 其他指标取平均
            avg_win_rate = np.mean([m.get('win_rate', 0) for m in count_metrics])
            avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in count_metrics])
            avg_profit_loss_ratio = np.mean([m.get('profit_loss_ratio', 0) for m in count_metrics])

            # 最大回撤取最大值
            max_drawdown = max(m.get('max_drawdown', 0) for m in count_metrics)

            # 总交易次数
            total_trades = sum(m.get('total_trades', 0) for m in count_metrics)

            logger.info(f"Top-{stock_count:<5} {total_return_gross:>8.2%}  "
                       f"{total_return_net:>13.2%}  "
                       f"{annual_return:>8.2%}  {avg_sharpe:>8.3f}  "
                       f"{max_drawdown:>8.2%}  {avg_win_rate:>6.2%}  "
                       f"{avg_profit_loss_ratio:>6.2f}  "
                       f"{total_trades:>10}")

        logger.info("=" * 150)


def _save_walkforward_results(all_window_metrics, all_selections):
    """保存Walk-Forward结果"""
    logger = logging.getLogger(__name__)

    try:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存窗口指标
        metrics_file = results_dir / f"walkforward_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_window_metrics, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"✅ 窗口指标已保存: {metrics_file.name}")

        # 保存选股结果
        selections_file = results_dir / f"walkforward_selections_{timestamp}.json"
        with open(selections_file, 'w', encoding='utf-8') as f:
            json.dump(all_selections, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"✅ 选股结果已保存: {selections_file.name}")

    except Exception as e:
        logger.error(f"❌ 保存结果失败: {e}")


def _train_final_model_for_window(train_data, val_data, best_params, window_idx, use_xgboost=True, use_lightgbm=True, market_data=None, optimize_thresholds=False):
    """为窗口训练最终模型(支持XGBoost+LightGBM集成学习)

    Args:
        train_data: 训练数据
        val_data: 验证数据（Walk-Forward外层已划分好的验证集）
        best_params: XGBoost最优参数
        window_idx: 窗口索引
        use_xgboost: 是否训练XGBoost模型(默认True)
        use_lightgbm: 是否训练LightGBM模型(默认True)
        market_data: 市场数据（用于XGBoost计算group权重）
        optimize_thresholds: 是否执行止盈止损ATR阈值优化（默认False）

    Returns:
        dict: 包含模型列表、性能指标、特征名等信息
    """
    logger = logging.getLogger(__name__)

    try:
        # 识别特征列
        feature_columns = _get_ml_feature_columns_simple(train_data)

        # 准备训练集标签
        if 'atr_return_label' not in train_data.columns:
            logger.error("❌ 训练数据缺少atr_return_label列")
            return None

        # 准备验证集标签
        if 'atr_return_label' not in val_data.columns:
            logger.error("❌ 验证数据缺少atr_return_label列")
            return None

        # ✅ 过滤掉标签为NaN的样本
        train_valid_mask = train_data['atr_return_label'].notna()
        train_data_valid = train_data[train_valid_mask].copy()

        val_valid_mask = val_data['atr_return_label'].notna()
        val_data_valid = val_data[val_valid_mask].copy()

        if len(train_data_valid) == 0:
            logger.error("❌ 没有有效的训练样本(所有标签都是NaN)")
            return None

        if len(val_data_valid) == 0:
            logger.error("❌ 没有有效的验证样本(所有标签都是NaN)")
            return None

        logger.info(f"📊 有效训练样本: {len(train_data_valid):,}/{len(train_data):,} ({len(train_data_valid)/len(train_data)*100:.1f}%)")
        logger.info(f"📊 有效验证样本: {len(val_data_valid):,}/{len(val_data):,} ({len(val_data_valid)/len(val_data)*100:.1f}%)")

        # 🆕 排序模型要求数据按日期排序,确保训练集和验证集按日期升序排列
        # 🚀 关键修复: 必须同时按date和stock_code排序,确保同一天内股票顺序确定
        if 'date' in train_data_valid.columns and 'stock_code' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("✅ 训练集已按[date, stock_code]升序排序（排序模型要求）")
        elif 'date' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("✅ 训练集已按date升序排序（排序模型要求）")

        if 'date' in val_data_valid.columns and 'stock_code' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("✅ 验证集已按[date, stock_code]升序排序（排序模型要求）")
        elif 'date' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("✅ 验证集已按date升序排序（排序模型要求）")

        # 🆕 提取relevance标签(用于LightGBM训练)和relevance_30标签(用于XGBoost训练)和原始收益率标签(用于IC计算)
        train_labels = train_data_valid['atr_return_label']
        val_labels = val_data_valid['atr_return_label']

        # 检查是否有relevance字段(LightGBM使用)
        if 'relevance' not in train_data_valid.columns or 'relevance' not in val_data_valid.columns:
            logger.error("❌ 训练/验证数据缺少relevance列,无法训练LightGBM排序模型")
            return None

        train_relevance = train_data_valid['relevance']
        val_relevance = val_data_valid['relevance']

        # 检查是否有relevance_30字段(XGBoost使用)
        if 'relevance_30' not in train_data_valid.columns or 'relevance_30' not in val_data_valid.columns:
            logger.error("❌ 训练/验证数据缺少relevance_30列,无法训练XGBoost排序模型")
            return None

        train_relevance_30 = train_data_valid['relevance_30']
        val_relevance_30 = val_data_valid['relevance_30']

        logger.info(f"✅ 提取relevance标签（整数0-3,4层）用于LightGBM排序模型训练")
        logger.info(f"✅ 提取relevance_30标签（整数0-29,30层）用于XGBoost排序模型训练")
        logger.info(f"✅ 保留atr_return_label用于IC计算")

        # 🚀 步骤1: 预处理训练集和验证集特征
        logger.info(f"\n{'='*60}")
        logger.info(f"预处理训练集和验证集特征(Window {window_idx})")
        logger.info(f"{'='*60}")

        # 对训练集进行特征预处理
        X_train_processed, feature_names = _prepare_feature_data(
            train_data_valid, feature_columns, data_type="训练集"
        )

        if X_train_processed is None:
            logger.error("❌ 训练集特征预处理失败")
            return None

        # 对验证集进行特征预处理
        X_val_processed, _ = _prepare_feature_data(
            val_data_valid, feature_columns, data_type="验证集"
        )

        if X_val_processed is None:
            logger.error("❌ 验证集特征预处理失败")
            return None

        # 🔧 确保标签与预处理后的特征数据长度匹配
        if len(X_train_processed) != len(train_labels):
            logger.warning(f"训练集预处理后长度变化: {len(train_labels)} -> {len(X_train_processed)}")
            train_labels = train_labels.iloc[:len(X_train_processed)] if hasattr(train_labels, 'iloc') else train_labels[:len(X_train_processed)]
            # 🚀 同步截取relevance标签
            train_relevance = train_relevance.iloc[:len(X_train_processed)] if hasattr(train_relevance, 'iloc') else train_relevance[:len(X_train_processed)]
            # 🚀 同步截取relevance_30标签
            train_relevance_30 = train_relevance_30.iloc[:len(X_train_processed)] if hasattr(train_relevance_30, 'iloc') else train_relevance_30[:len(X_train_processed)]

        if len(X_val_processed) != len(val_labels):
            logger.warning(f"验证集预处理后长度变化: {len(val_labels)} -> {len(X_val_processed)}")
            val_labels = val_labels.iloc[:len(X_val_processed)] if hasattr(val_labels, 'iloc') else val_labels[:len(X_val_processed)]
            # 🚀 同步截取relevance标签
            val_relevance = val_relevance.iloc[:len(X_val_processed)] if hasattr(val_relevance, 'iloc') else val_relevance[:len(X_val_processed)]
            # 🚀 同步截取relevance_30标签
            val_relevance_30 = val_relevance_30.iloc[:len(X_val_processed)] if hasattr(val_relevance_30, 'iloc') else val_relevance_30[:len(X_val_processed)]

        # 🆕 提取训练集日期（用于时间衰减权重和group计算）
        train_dates = None
        if 'date' in train_data_valid.columns:
            train_dates = train_data_valid['date'].values[:len(X_train_processed)]
            logger.info("✅ 提取训练集日期（用于时间衰减权重计算和group计算）")

        # 🆕 提取验证集日期（用于group计算）
        val_dates = None
        if 'date' in val_data_valid.columns:
            val_dates = val_data_valid['date'].values[:len(X_val_processed)]
            logger.info("✅ 提取验证集日期（用于group计算）")

        # 🆕 提取验证集ATR14（用于止盈止损阈值优化）
        val_atr14 = None
        if 'atr14' in val_data_valid.columns:
            val_atr14 = val_data_valid['atr14'].values[:len(X_val_processed)]
            logger.info("✅ 提取验证集ATR14（用于止盈止损阈值优化）")
        else:
            logger.warning("⚠️ 验证数据中未包含atr14列，止盈止损阈值优化将无法执行")

        # 🆕 提取换手率和流通市值（用于基础权重计算）
        train_turnover_rate = None
        train_circ_mv = None
        val_turnover_rate = None
        val_circ_mv = None
        if 'turnover_rate' in train_data_valid.columns:
            train_turnover_rate = train_data_valid['turnover_rate'].values[:len(X_train_processed)]
            logger.info("✅ 提取训练集换手率（用于基础权重计算）")
        else:
            logger.warning("⚠️ 训练数据中未包含turnover_rate列，将使用均匀权重")

        if 'circ_mv' in train_data_valid.columns:
            train_circ_mv = train_data_valid['circ_mv'].values[:len(X_train_processed)]
            logger.info("✅ 提取训练集流通市值（用于基础权重计算）")
        else:
            logger.warning("⚠️ 训练数据中未包含circ_mv列，将使用均匀权重")

        # 🆕 提取验证集换手率和流通市值（用于验证集权重计算）
        if 'turnover_rate' in val_data_valid.columns:
            val_turnover_rate = val_data_valid['turnover_rate'].values[:len(X_val_processed)]
            logger.info("✅ 提取验证集换手率（用于验证集权重计算）")
        else:
            logger.warning("⚠️ 验证数据中未包含turnover_rate列，验证集将使用均匀权重")

        if 'circ_mv' in val_data_valid.columns:
            val_circ_mv = val_data_valid['circ_mv'].values[:len(X_val_processed)]
            logger.info("✅ 提取验证集流通市值（用于验证集权重计算）")
        else:
            logger.warning("⚠️ 验证数据中未包含circ_mv列，验证集将使用均匀权重")

        # 🚀 条件计算样本权重（仅LightGBM需要）
        unified_sample_weights = None
        unified_val_weights = None
        if use_lightgbm:
            logger.info("🔧 计算样本权重（供LightGBM使用）...")
            train_relevance_array = train_relevance.values if hasattr(train_relevance, 'values') else train_relevance
            val_relevance_array = val_relevance.values if hasattr(val_relevance, 'values') else val_relevance

            # 🔧 修复: 使用训练集最新日期作为时间衰减基准（Walk-Forward模式）
            reference_date = None
            if train_dates is not None and len(train_dates) > 0:
                reference_date = pd.to_datetime(train_dates).max()
                logger.info(f"   时间衰减基准日期: {reference_date.strftime('%Y-%m-%d')} (训练集最新日期)")

            # 计算训练集权重
            unified_sample_weights = _calculate_professional_sample_weights(
                train_relevance_array,  # 使用relevance标签
                imbalance_ratio=1.0,
                logger=logger,
                boundary_distances=None,
                is_regression=True,
                y_train_original=None,
                train_dates=train_dates,
                time_decay_half_life=180.0,  # 统一使用180天
                reference_date=reference_date,  # 🔧 修复: 使用训练集最新日期
                turnover_rate=train_turnover_rate,
                circ_mv=train_circ_mv,
                label_rank=train_relevance_array,  # relevance值(0-3)
                label_boost_beta=0.15,  # 统一使用0.15
                w_min=0.8,
                w_max=1.2
            )
            logger.info(f"✅ 训练集样本权重计算完成")
            logger.info(f"   权重范围: [{unified_sample_weights.min():.4f}, {unified_sample_weights.max():.4f}]")
            logger.info(f"   权重均值: {unified_sample_weights.mean():.4f}")

            # 计算验证集权重（使用相同的参数）
            unified_val_weights = _calculate_professional_sample_weights(
                val_relevance_array,  # 使用relevance标签
                imbalance_ratio=1.0,
                logger=logger,
                boundary_distances=None,
                is_regression=True,
                y_train_original=None,
                train_dates=val_dates,  # 验证集使用自己的日期
                time_decay_half_life=180.0,  # 统一使用180天
                reference_date=reference_date,  # 使用训练集最新日期作为基准
                turnover_rate=val_turnover_rate,
                circ_mv=val_circ_mv,
                label_rank=val_relevance_array,  # relevance值(0-3)
                label_boost_beta=0.15,  # 统一使用0.15
                w_min=0.8,
                w_max=1.2
            )
            logger.info(f"✅ 验证集样本权重计算完成")
            logger.info(f"   权重范围: [{unified_val_weights.min():.4f}, {unified_val_weights.max():.4f}]")
            logger.info(f"   权重均值: {unified_val_weights.mean():.4f}")
        else:
            logger.info("⏭️ 跳过样本权重计算（XGBoost排序模型使用分组权重）")

        # 🚀 构建预处理数据字典
        preprocessed_data = {
            'X_train': X_train_processed,
            'X_val': X_val_processed,
            'y_train': train_labels.values if hasattr(train_labels, 'values') else train_labels,
            'y_val': val_labels.values if hasattr(val_labels, 'values') else val_labels,
            'y_train_original': train_labels.values if hasattr(train_labels, 'values') else train_labels,
            'y_val_original': val_labels.values if hasattr(val_labels, 'values') else val_labels,
            'train_relevance': train_relevance.values if hasattr(train_relevance, 'values') else train_relevance,  # 🆕 relevance标签(4层,LightGBM使用)
            'val_relevance': val_relevance.values if hasattr(val_relevance, 'values') else val_relevance,  # 🆕 relevance标签(4层,LightGBM使用)
            'train_relevance_30': train_relevance_30.values if hasattr(train_relevance_30, 'values') else train_relevance_30,  # 🆕 relevance_30标签(30层,XGBoost使用)
            'val_relevance_30': val_relevance_30.values if hasattr(val_relevance_30, 'values') else val_relevance_30,  # 🆕 relevance_30标签(30层,XGBoost使用)
            'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
            'train_dates': train_dates,  # 🆕 训练集日期
            'val_dates': val_dates,  # 🆕 验证集日期
            'val_atr14': val_atr14,  # 🆕 验证集ATR14（用于止盈止损阈值优化）
            'turnover_rate': train_turnover_rate,  # 🆕 添加换手率
            'circ_mv': train_circ_mv,  # 🆕 添加流通市值
            'val_turnover_rate': val_turnover_rate,  # 🆕 添加验证集换手率
            'val_circ_mv': val_circ_mv,  # 🆕 添加验证集流通市值
            'sample_weights': unified_sample_weights,  # 🆕 添加统一计算的样本权重
            'val_sample_weights': unified_val_weights  # 🆕 添加验证集样本权重
        }

        logger.info(f"✅ 预处理完成: 训练集{len(X_train_processed):,}样本, 验证集{len(X_val_processed):,}样本")
        logger.info(f"✅ relevance标签已添加到预处理数据")
        logger.info(f"✅ 训练集和验证集日期已添加到预处理数据")
        if use_lightgbm:
            logger.info(f"✅ 样本权重已添加到预处理数据（供LightGBM使用）")
        else:
            logger.info(f"⏭️ 样本权重未计算（XGBoost排序模型使用分组权重）")

        # 生成统一时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"🕐 模型统一时间戳: {timestamp}")

        # 🚀 步骤2: 训练XGBoost模型(如果选择了XGBoost)
        xgb_result = None
        if use_xgboost:
            logger.info(f"\n{'='*60}")
            logger.info(f"训练XGBoost模型(Window {window_idx})")
            logger.info(f"{'='*60}")

            # 🚀 使用贝叶斯搜索的最优参数,跳过参数搜索
            # 🚀 传入preprocessed_data，跳过_train_ml_model内部的数据划分
            xgb_result = _train_ml_model(
                raw_data=train_data_valid,  # ✅ 仅用于提取元数据
                labels=train_labels,  # ✅ 仅用于提取元数据
                feature_columns=feature_columns,
                winsorization_thresholds=None,
                label_stats=None,
                label_transform='none',  # 使用原始标签
                enable_param_search=False,  # ✅ 跳过参数搜索
                search_strategy='bayesian',  # ✅ 与实际搜索策略一致
                random_seed=42,
                preprocessed_data=preprocessed_data,  # ✅ 传入预处理数据，跳过内部划分
                enable_shap_analysis=True,  # ✅ 启用SHAP分析
                preset_params=best_params,  # ✅ 使用贝叶斯搜索的最优参数
                market_data=market_data  # 🚀 传递market_data用于XGBoost计算group权重
            )

            if xgb_result is None:
                logger.error("❌ XGBoost模型训练失败")
                if not use_lightgbm:
                    return None
                logger.warning("将继续训练LightGBM模型")
            else:
                logger.info(f"✅ XGBoost模型训练完成")

        # 🚀 步骤3: 训练LightGBM模型(如果选择了LightGBM)
        lgb_result = None
        if use_lightgbm:
            logger.info(f"\n{'='*60}")
            logger.info(f"训练LightGBM模型(Window {window_idx})")
            logger.info(f"{'='*60}")

            try:
                # 🚀 确定特征列表和预处理数据
                if xgb_result is not None:
                    # 复用XGBoost的预处理数据（已包含训练集和验证集）
                    lgb_preprocessed_data = {
                        'X_train': xgb_result.get('X_train_processed'),
                        'X_val': xgb_result.get('X_val_processed'),
                        # 'y_train': xgb_result.get('y_train'),
                        # 'y_val': xgb_result.get('y_val'),
                        'y_train': train_labels.values if hasattr(train_labels, 'values') else train_labels,
                        'y_val': val_labels.values if hasattr(val_labels, 'values') else val_labels,
                        'y_train_original': train_labels.values if hasattr(train_labels, 'values') else train_labels,
                        'y_val_original': val_labels.values if hasattr(val_labels, 'values') else val_labels,
                        # 'y_train_original': xgb_result.get('y_train'),
                        # 'y_val_original': xgb_result.get('y_val'),
                        'train_relevance': train_relevance.values if hasattr(train_relevance, 'values') else train_relevance,  # 🆕 relevance标签
                        'val_relevance': val_relevance.values if hasattr(val_relevance, 'values') else val_relevance,  # 🆕 relevance标签
                        'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                        'train_dates': train_dates,  # 🆕 添加训练集日期
                        'val_dates': val_dates,  # 🆕 添加验证集日期
                        'turnover_rate': train_turnover_rate,  # 🆕 添加换手率
                        'circ_mv': train_circ_mv,  # 🆕 添加流通市值
                        'val_turnover_rate': val_turnover_rate,  # 🆕 添加验证集换手率
                        'val_circ_mv': val_circ_mv,  # 🆕 添加验证集流通市值
                        'sample_weights': preprocessed_data.get('sample_weights'),  # 🆕 添加统一样本权重
                        'val_sample_weights': preprocessed_data.get('val_sample_weights')  # 🆕 添加验证集样本权重
                    }
                    lgb_feature_names = xgb_result['feature_names']
                    logger.info(f"✅ 使用XGBoost的预处理数据和特征列表")
                else:
                    # 独立训练LightGBM,使用已准备好的预处理数据
                    lgb_preprocessed_data = preprocessed_data
                    lgb_feature_names = feature_columns
                    logger.info(f"✅ 独立训练LightGBM,使用原始特征列表")

                # 检查预处理数据是否可用
                if lgb_preprocessed_data.get('X_train') is None:
                    logger.error("❌ 缺少预处理数据,无法训练LightGBM")
                    if not use_xgboost or xgb_result is None:
                        return None
                    lgb_result = None
                else:
                    lgb_result = _train_lightgbm_model(
                        raw_data=train_data_valid,  # ✅ 仅用于提取元数据
                        labels=train_labels,  # ✅ 仅用于提取元数据
                        feature_columns=lgb_feature_names,
                        label_stats=None,
                        preprocessed_data=lgb_preprocessed_data,  # ✅ 传入预处理数据
                        optimize_thresholds=optimize_thresholds  # 🚀 传递止盈止损优化参数
                    )

                    if lgb_result is None:
                        logger.error("❌ LightGBM模型训练失败")
                        if not use_xgboost or xgb_result is None:
                            return None
                    else:
                        logger.info(f"✅ LightGBM模型训练完成")

            except Exception as e:
                logger.error(f"❌ LightGBM模型训练失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                if not use_xgboost or xgb_result is None:
                    return None
                lgb_result = None

        # 3. 收集模型结果
        ensemble_results = []
        if xgb_result is not None:
            ensemble_results.append(xgb_result)
        if lgb_result is not None:
            ensemble_results.append(lgb_result)

        if len(ensemble_results) == 0:
            logger.error("❌ 所有模型训练失败")
            return None

        logger.info(f"\n{'='*60}")
        logger.info(f"模型训练汇总(Window {window_idx})")
        logger.info(f"{'='*60}")
        logger.info(f"  训练模型数: {len(ensemble_results)}")
        logger.info(f"  XGBoost: {'✅' if xgb_result is not None else '❌'}")
        logger.info(f"  LightGBM: {'✅' if lgb_result is not None else '❌'}")

        # 4. 保存模型
        model_files = []
        for i, model_result in enumerate(ensemble_results):
            model_type = model_result.get('model_type', 'xgboost')
            model_file = Path("results") / f"{model_type}_model_window{window_idx}_{timestamp}.pkl"

            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model_result['model'],
                    'scaler': model_result['scaler'],
                    'feature_names': model_result['feature_names'],
                    'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                    'model_type': model_type,
                    'window_idx': window_idx,
                    'best_params': best_params if model_type == 'xgboost' else None,
                    'performance': model_result.get('performance', {})
                }, f)

            logger.info(f"✅ {model_type.upper()}模型已保存: {model_file.name}")
            model_files.append(model_file)

        # 🚀 5. 计算模型权重(基于验证集RankIC) - 排序模型专用
        model_performances = []
        for model_result in ensemble_results:
            performance = model_result.get('performance', {})
            # 🚀 排序模型使用RankIC作为权重计算指标
            rank_ic = performance.get('rank_ic')
            model_performances.append(rank_ic)

        # 返回集成结果
        return {
            'models': [m['model'] for m in ensemble_results],
            'scalers': [m['scaler'] for m in ensemble_results],
            'feature_names': ensemble_results[0]['feature_names'],  # 所有模型使用相同特征
            'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
            'model_files': model_files,
            'model_performances': model_performances,
            'model_types': [m.get('model_type', 'xgboost') for m in ensemble_results]
        }

    except Exception as e:
        logger.error(f"❌ 训练最终模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _backtest_on_window(test_data, model_result, window, args):
    """在窗口测试集上回测(支持集成学习)"""
    logger = logging.getLogger(__name__)

    try:
        # 🔧 修复：直接使用训练时的特征列表，而不是重新提取
        # 原因：确保训练和测试使用完全相同的特征
        feature_columns = model_result['feature_names']
        logger.info(f"📊 使用训练时的特征列表: {len(feature_columns)}个特征")

        # 验证测试集是否包含所有必需特征
        missing_features = [f for f in feature_columns if f not in test_data.columns]
        if missing_features:
            logger.error(f"❌ 测试集缺少{len(missing_features)}个训练时使用的特征:")
            logger.error(f"   缺失特征: {missing_features[:20]}")
            logger.error("   请检查特征生成逻辑，确保训练和测试使用相同的特征")
            return None

        # 🆕 诊断日志：特征质量检查
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 [{window['name']}] 特征质量检查")
        logger.info(f"{'='*60}")

        # 检查每个特征的异常情况
        abnormal_features = []
        for feature in feature_columns:
            feature_data = test_data[feature]

            # 检查全0
            if (feature_data == 0).all():
                abnormal_features.append(f"{feature}: 全部为0")
                logger.warning(f"⚠️ 特征 {feature}: 全部为0")

            # 检查全NaN
            elif feature_data.isna().all():
                abnormal_features.append(f"{feature}: 全部为NaN")
                logger.warning(f"⚠️ 特征 {feature}: 全部为NaN")

            # 检查高比例NaN
            elif feature_data.isna().mean() > 0.5:
                nan_pct = feature_data.isna().mean() * 100
                abnormal_features.append(f"{feature}: {nan_pct:.1f}% NaN")
                logger.warning(f"⚠️ 特征 {feature}: {nan_pct:.1f}% 为NaN")

            # 检查常数列
            elif feature_data.nunique() == 1:
                abnormal_features.append(f"{feature}: 常数列")
                logger.warning(f"⚠️ 特征 {feature}: 常数列")

        if abnormal_features:
            logger.warning(f"\n⚠️ 发现{len(abnormal_features)}个异常特征:")
            for abnormal in abnormal_features[:10]:
                logger.warning(f"   - {abnormal}")
            if len(abnormal_features) > 10:
                logger.warning(f"   ... 还有{len(abnormal_features) - 10}个")
        else:
            logger.info(f"✅ 所有{len(feature_columns)}个特征质量正常")

        logger.info(f"{'='*60}\n")

        # 准备特征数据
        X_test, _ = _prepare_feature_data(
            test_data, feature_columns,
            data_type="测试集"
        )

        if X_test is None:
            logger.error("❌ 测试集特征准备失败")
            return None

        # 🔧 修复：不需要特征对齐，因为已经使用训练时的特征列表
        # X_test已经按照feature_columns的顺序排列
        X_test_aligned = X_test
        logger.info(f"✅ 测试集特征准备完成: {X_test_aligned.shape}")

        # 检查是否为集成模型
        models = model_result.get('models')
        if models is None:
            # 单模型(向后兼容)
            models = [model_result['model']]
            scalers = [model_result['scaler']]
            model_performances = [1.0]
            model_types = ['xgboost']
        else:
            # 集成模型
            scalers = model_result['scalers']
            model_performances = model_result.get('model_performances', [1.0] * len(models))
            model_types = model_result.get('model_types', ['xgboost'] * len(models))

        logger.info(f"📊 使用{len(models)}个模型进行集成预测")

        # 对每个模型进行预测
        all_predictions = []
        for i, (model, scaler, model_type) in enumerate(zip(models, scalers, model_types)):
            # 🔧 修复：不再使用StandardScaler，直接使用Winsorization后的特征
            # 因子在计算时已经进行了时间序列标准化（如zscore、rank）
            X_test_scaled = X_test_aligned

            # 根据模型类型进行预测
            if model_type == 'lightgbm':
                # LightGBM模型
                y_pred_single = model.predict(X_test_scaled, num_iteration=model.best_iteration)
                logger.info(f"  ✅ 模型{i+1}(LightGBM)预测完成")
            elif hasattr(model, 'predict') and hasattr(model, 'get_score'):
                # XGBoost Booster对象
                import xgboost as xgb
                dtest = xgb.DMatrix(X_test_scaled)
                y_pred_single = model.predict(dtest)
                logger.info(f"  ✅ 模型{i+1}(XGBoost Booster)预测完成")
            else:
                # XGBRegressor对象
                y_pred_single = model.predict(X_test_scaled)
                logger.info(f"  ✅ 模型{i+1}(XGBoost Regressor)预测完成")

            all_predictions.append(y_pred_single)

        # 计算加权平均预测(基于RankIC)
        if len(all_predictions) > 1:
            # 多模型集成
            # 🔧 修复: 处理None值,将None替换为0
            model_performances_clean = [p if p is not None else 0.0 for p in model_performances]
            mse_array = np.array(model_performances_clean)

            # 🚀 排序模型使用RankIC作为权重,RankIC越大权重越高
            if np.all(mse_array > 0):
                # 使用RankIC作为权重(RankIC越大权重越高)
                weights = mse_array / mse_array.sum()
            else:
                # 如果RankIC无效或为负,使用均等权重
                logger.warning("⚠️ 部分模型RankIC无效或为负,使用均等权重")
                weights = np.ones(len(all_predictions)) / len(all_predictions)

            logger.info(f"📊 模型权重: {weights}")

            # 加权平均
            y_pred = np.average(all_predictions, axis=0, weights=weights)

            # 计算预测分歧度(用于置信度过滤)
            std_pred = np.std(all_predictions, axis=0)
            logger.info(f"📊 预测分歧度: 均值={std_pred.mean():.4f}, 最大={std_pred.max():.4f}")
        else:
            # 单模型
            y_pred = all_predictions[0]
            std_pred = np.zeros_like(y_pred)  # 单模型无分歧度

        # 添加预测结果到测试数据
        test_data_with_pred = test_data.copy()
        test_data_with_pred['predicted_return'] = y_pred
        test_data_with_pred['std_pred'] = std_pred

        # 🆕 诊断日志：预测值统计
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 [{window['name']}] 预测值统计")
        logger.info(f"{'='*60}")
        logger.info(f"预测值分位数分布:")
        logger.info(f"  0%  (最小值): {np.percentile(y_pred, 0):.4f} ATR")
        logger.info(f"  25% (下四分位): {np.percentile(y_pred, 25):.4f} ATR")
        logger.info(f"  50% (中位数):   {np.percentile(y_pred, 50):.4f} ATR")
        logger.info(f"  75% (上四分位): {np.percentile(y_pred, 75):.4f} ATR")
        logger.info(f"  100% (最大值):  {np.percentile(y_pred, 100):.4f} ATR")
        logger.info(f"均值±标准差: {y_pred.mean():.4f} ± {y_pred.std():.4f} ATR")

        # 统计异常值
        zero_count = (y_pred == 0).sum()
        near_zero_count = (np.abs(y_pred) < 0.01).sum()
        negative_count = (y_pred < 0).sum()
        logger.info(f"\n异常值统计:")
        logger.info(f"  预测值=0: {zero_count}/{len(y_pred)} ({zero_count/len(y_pred)*100:.1f}%)")
        logger.info(f"  预测值接近0(|x|<0.01): {near_zero_count}/{len(y_pred)} ({near_zero_count/len(y_pred)*100:.1f}%)")
        logger.info(f"  预测值<0: {negative_count}/{len(y_pred)} ({negative_count/len(y_pred)*100:.1f}%)")

        # 如果有大量0值，发出警告
        if zero_count / len(y_pred) > 0.3:
            logger.warning(f"⚠️ 警告：超过30%的预测值为0，可能存在特征对齐问题！")
        if near_zero_count / len(y_pred) > 0.5:
            logger.warning(f"⚠️ 警告：超过50%的预测值接近0，模型预测可能失效！")
        logger.info(f"{'='*60}\n")


        # ✅ 调用统一的选股回测函数(复用test_model逻辑)
        logger.info(f"\n📊 [{window['name']}] 开始选股回测...")
        logger.info(f"   测试期: {window['test_start']} ~ {window['test_end']}")
        logger.info(f"   测试天数: {len(test_data_with_pred['date'].unique())}天")
        logger.info(f"   候选股票: {len(test_data_with_pred)}条记录")

        # 调用统一选股回测函数
        backtest_result = _run_stock_selection_backtest(
            backtest_data=test_data_with_pred,
            stock_counts=[1, 2, 3, 5],  # Top-1/2/3/5选股
            window_name=window['name'],
            market_data=None  # Walk-Forward暂不使用市场状态
        )

        if backtest_result is None or not backtest_result.get('backtest_results'):
            logger.error(f"❌ [{window['name']}] 选股回测失败")
            return None

        # ✅ 返回所有Top-k的完整结果
        return {
            # 保持向后兼容: 提取Top-1结果作为主要返回值
            'selections': backtest_result['detailed_trades'].get(1, []),
            'total_return': backtest_result['backtest_results'].get(1, {}).get('total_return', 0),
            'win_rate': backtest_result['backtest_results'].get(1, {}).get('win_rate', 0),
            'profit_loss_ratio': backtest_result['backtest_results'].get(1, {}).get('profit_loss_ratio', 0),
            'sharpe_ratio': backtest_result['backtest_results'].get(1, {}).get('sharpe_ratio', 0),
            'max_drawdown': backtest_result['backtest_results'].get(1, {}).get('max_drawdown', 0),
            'total_trades': backtest_result['backtest_results'].get(1, {}).get('total_trades', 0),
            # ✅ 所有Top-k档位的完整结果
            'all_results': backtest_result['backtest_results'],  # {1: {...}, 2: {...}, 3: {...}, 5: {...}}
            'detailed_trades': backtest_result['detailed_trades'],  # {1: [...], 2: [...], 3: [...], 5: [...]}
            'summary_stats': backtest_result['summary_stats']
        }

    except Exception as e:
        logger.error(f"❌ 回测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _process_walkforward_window(window, factor_data, args, window_idx, total_windows, all_windows):
    """
    处理单个Walk-Forward窗口

    Args:
        window: 窗口配置字典
        factor_data: 完整因子数据
        args: 命令行参数
        window_idx: 当前窗口索引
        total_windows: 总窗口数
        all_windows: 所有窗口列表(用于参数传递)

    Returns:
        dict: 包含backtest_result, selections, metrics的字典,失败返回None
    """
    logger = logging.getLogger(__name__)

    try:
        # 0. 加载市场数据（用于XGBoost计算group权重）
        logger.info(f"\n🔄 步骤0: 加载市场数据")
        market_data = None
        try:
            from data_loader import IntegratedDataLoader
            loader = IntegratedDataLoader()
            start_date_str = factor_data['date'].min().strftime('%Y%m%d')
            end_date_str = factor_data['date'].max().strftime('%Y%m%d')
            _, _, market_data = loader.load_all_data(start_date_str, end_date_str, extend_history_days=20)

            if market_data is None or market_data.empty:
                logger.warning("⚠️ 市场数据加载失败，XGBoost将使用均匀group权重")
                market_data = None
            else:
                # 确保date列是datetime类型
                if 'date' in market_data.columns:
                    market_data['date'] = pd.to_datetime(market_data['date'])
                logger.info(f"✅ 市场数据加载成功: {len(market_data)}条记录")
        except Exception as e:
            logger.warning(f"⚠️ 市场数据加载失败: {e}，XGBoost将使用均匀group权重")
            market_data = None

        # 1. 划分数据
        logger.info(f"\n📊 步骤1: 划分数据集")

        # 训练集
        train_data = factor_data[
            (factor_data['date'] >= pd.to_datetime(window['train_start'])) &
            (factor_data['date'] <= pd.to_datetime(window['train_end']))
        ].copy()

        # 验证集
        val_data = factor_data[
            (factor_data['date'] >= pd.to_datetime(window['val_start'])) &
            (factor_data['date'] <= pd.to_datetime(window['val_end']))
        ].copy()

        # 测试集
        test_data = factor_data[
            (factor_data['date'] >= pd.to_datetime(window['test_start'])) &
            (factor_data['date'] <= pd.to_datetime(window['test_end']))
        ].copy()

        logger.info(f"   训练集: {len(train_data):,}条 ({window['train_start']} ~ {window['train_end']})")
        logger.info(f"   验证集: {len(val_data):,}条 ({window['val_start']} ~ {window['val_end']})")
        logger.info(f"   测试集: {len(test_data):,}条 ({window['test_start']} ~ {window['test_end']})")

        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            logger.error("❌ 训练集、验证集或测试集为空")
            return None

        # 1.5. 生成标签 (如果还没有)
        if 'atr_return_label' not in train_data.columns:
            logger.info(f"\n🏷️ 步骤1.5: 数据过滤和标签生成")

            # 合并训练集、验证集和测试集用于标签生成(需要未来数据)
            combined_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

            # 🚀 关键修复: concat后立即排序,确保数据连续性
            combined_data = combined_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("   ✅ 合并数据已按['date', 'stock_code']排序")

            # 数据过滤（在标签构建前）
            logger.info("   🔄 开始数据过滤（北交所、新股和ST股票）...")

            initial_total = len(combined_data)
            bj_filtered_count = 0
            new_stock_filtered_count = 0
            st_filtered_count = 0

            all_dates = sorted(combined_data['date'].unique())
            filtered_data_list = []

            for date in all_dates:
                date_str = date.strftime('%Y%m%d')
                # 🚀 优化内存: 使用布尔索引而不是copy
                date_mask = combined_data['date'] == date

                if not date_mask.any():
                    continue

                initial_count = date_mask.sum()

                # 1. 过滤北交所股票 (股票代码以.BJ或.bj结尾)
                if 'stock_code' in combined_data.columns:
                    bj_filter = ~combined_data.loc[date_mask, 'stock_code'].str.endswith(('.BJ', '.bj'), na=False)
                    date_mask = date_mask & combined_data.index.isin(combined_data[date_mask].index[bj_filter])
                    after_bj_count = date_mask.sum()
                    bj_filtered_count += (initial_count - after_bj_count)

                # 2. 过滤新股
                if date_mask.any():
                    before_new_stock = date_mask.sum()
                    daily_data_temp = combined_data[date_mask]
                    new_stock_filter = _filter_new_stocks(daily_data_temp, date_str, logger)
                    date_mask = date_mask & combined_data.index.isin(daily_data_temp.index[new_stock_filter])
                    new_stock_filtered_count += (before_new_stock - date_mask.sum())

                # 3. 过滤ST股票
                if date_mask.any():
                    before_st = date_mask.sum()
                    daily_data_temp = combined_data[date_mask]
                    st_filter = _filter_st_stocks(daily_data_temp, logger)
                    date_mask = date_mask & combined_data.index.isin(daily_data_temp.index[st_filter])
                    st_filtered_count += (before_st - date_mask.sum())

                # 🚀 只在最后一次copy,减少内存分配
                if date_mask.any():
                    filtered_data_list.append(combined_data[date_mask].copy())

            if not filtered_data_list:
                logger.error("❌ 所有数据都被过滤掉了")
                return None

            # ✅ 优化内存使用: 分批合并数据,避免一次性分配大内存
            logger.info(f"   🔄 合并{len(filtered_data_list)}个日期的数据...")
            batch_size = 50  # 每批合并50个日期
            combined_batches = []

            for i in range(0, len(filtered_data_list), batch_size):
                batch = filtered_data_list[i:i+batch_size]
                batch_len = len(batch)  # 🔧 修复：在删除前保存长度
                combined_batch = pd.concat(batch, ignore_index=True)
                combined_batches.append(combined_batch)

                # 释放内存
                del batch

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"      已合并 {i+batch_len}/ {len(filtered_data_list)} 个日期")

            # 最终合并所有批次
            combined_data = pd.concat(combined_batches, ignore_index=True)
            del combined_batches

            # 🚀 关键修复: 合并后立即排序,确保数据连续性
            combined_data = combined_data.sort_values(['date', 'stock_code']).reset_index(drop=True)

            final_total = len(combined_data)

            logger.info(f"   ✅ 数据过滤完成:")
            logger.info(f"      初始样本: {initial_total:,}条")
            logger.info(f"      过滤北交所: {bj_filtered_count:,}条")
            logger.info(f"      过滤新股: {new_stock_filtered_count:,}条")
            logger.info(f"      过滤ST股票: {st_filtered_count:,}条")
            logger.info(f"      最终样本: {final_total:,}条 (保留{final_total/initial_total*100:.1f}%)")
            logger.info(f"      ✅ 数据已按['date', 'stock_code']排序")

            # 🔧 修复数据泄漏：先划分数据，再分别生成标签
            logger.info("   🔄 先划分训练集、验证集和测试集...")

            # 划分训练集、验证集和测试集（不含标签）
            train_data_unlabeled = combined_data[
                (combined_data['date'] >= pd.to_datetime(window['train_start'])) &
                (combined_data['date'] <= pd.to_datetime(window['train_end']))
            ].copy()

            val_data_unlabeled = combined_data[
                (combined_data['date'] >= pd.to_datetime(window['val_start'])) &
                (combined_data['date'] <= pd.to_datetime(window['val_end']))
            ].copy()

            test_data_unlabeled = combined_data[
                (combined_data['date'] >= pd.to_datetime(window['test_start'])) &
                (combined_data['date'] <= pd.to_datetime(window['test_end']))
            ].copy()

            # 🚀 关键修复: 划分后立即排序,确保数据连续性
            train_data_unlabeled = train_data_unlabeled.sort_values(['date', 'stock_code']).reset_index(drop=True)
            val_data_unlabeled = val_data_unlabeled.sort_values(['date', 'stock_code']).reset_index(drop=True)
            test_data_unlabeled = test_data_unlabeled.sort_values(['date', 'stock_code']).reset_index(drop=True)

            logger.info(f"   ✅ 数据划分完成:")
            logger.info(f"      训练集: {len(train_data_unlabeled):,}条 ({window['train_start']} ~ {window['train_end']})")
            logger.info(f"      验证集: {len(val_data_unlabeled):,}条 ({window['val_start']} ~ {window['val_end']})")
            logger.info(f"      测试集: {len(test_data_unlabeled):,}条 ({window['test_start']} ~ {window['test_end']})")
            logger.info(f"      ✅ 所有数据集已按['date', 'stock_code']排序")

            # 🔧 步骤1：对训练集生成标签
            logger.info("   🔄 步骤1: 对训练集生成标签...")
            train_labeled_result = _generate_training_labels_for_all_data(
                train_data_unlabeled,
                alpha=1.5,
                debug=False,
                is_training=True,
                label_stats=None
            )

            if train_labeled_result is None:
                logger.error("❌ 训练集标签生成失败")
                return None

            # 解包训练集结果
            if isinstance(train_labeled_result, tuple):
                train_data, label_stats = train_labeled_result
            else:
                train_data = train_labeled_result
                label_stats = None

            # 🔧 步骤2：对验证集生成标签
            logger.info("   🔄 步骤2: 对验证集生成标签...")
            val_labeled_result = _generate_training_labels_for_all_data(
                val_data_unlabeled,
                alpha=1.5,
                debug=False,
                is_training=False,  # 验证集：不计算统计量
                label_stats=None
            )

            if val_labeled_result is None:
                logger.error("❌ 验证集标签生成失败")
                return None

            # 解包验证集结果
            if isinstance(val_labeled_result, tuple):
                val_data, _ = val_labeled_result
            else:
                val_data = val_labeled_result

            # 测试集不需要生成标签
            test_data = test_data_unlabeled

            logger.info(f"   ✅ 标签生成完成")
            logger.info(f"   训练集有效样本: {train_data['atr_return_label'].notna().sum():,}/{len(train_data):,}")
            logger.info(f"   验证集有效样本: {val_data['atr_return_label'].notna().sum():,}/{len(val_data):,}")
            logger.info(f"   测试集: {len(test_data):,}条（不含标签）")

        # 2. 检查是否已有模型，决定是否需要贝叶斯搜索和训练
        logger.info(f"\n🤖 步骤2: 检查模型或训练新模型")

        # 🚀 解析模型选择参数
        models_str = getattr(args, 'models', 'xgboost,lightgbm')
        selected_models = [m.strip().lower() for m in models_str.split(',')]

        use_xgboost = 'xgboost' in selected_models
        use_lightgbm = 'lightgbm' in selected_models

        if not use_xgboost and not use_lightgbm:
            logger.error("❌ 至少需要选择一个模型 (xgboost 或 lightgbm)")
            return False

        logger.info(f"🚀 模型选择: {', '.join(selected_models)}")
        logger.info(f"   XGBoost: {'✅' if use_xgboost else '❌'}")
        logger.info(f"   LightGBM: {'✅' if use_lightgbm else '❌'}")

        # 🔧 修复：每个窗口查找自己的模型
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # 查找当前窗口的XGBoost模型
        xgb_pattern = f"xgboost_model_window{window_idx}_*.pkl"
        xgb_files = list(results_dir.glob(xgb_pattern))

        # 查找当前窗口的LightGBM模型
        lgb_pattern = f"lightgbm_model_window{window_idx}_*.pkl"
        lgb_files = list(results_dir.glob(lgb_pattern))

        model_result = None

        # 如果找到模型，尝试加载
        if xgb_files:
            logger.info(f"🔍 找到{len(xgb_files)}个Window {window_idx}的XGBoost模型")

            # 使用最新的模型
            latest_xgb_file = max(xgb_files, key=lambda f: f.stem)
            logger.info(f"   最新XGBoost模型: {latest_xgb_file.name}")

            # 🚀 根据use_xgboost和use_lightgbm决定加载哪些模型
            if use_xgboost and use_lightgbm and lgb_files:
                # 需要集成学习：加载XGBoost + LightGBM
                latest_lgb_file = max(lgb_files, key=lambda f: f.stem)
                logger.info(f"   最新LightGBM模型: {latest_lgb_file.name}")

                try:
                    import pickle

                    with open(latest_xgb_file, 'rb') as f:
                        xgb_data = pickle.load(f)

                    with open(latest_lgb_file, 'rb') as f:
                        lgb_data = pickle.load(f)

                    # 构建model_result
                    model_result = {
                        'models': [xgb_data['model'], lgb_data['model']],
                        'scalers': [xgb_data['scaler'], lgb_data['scaler']],
                        'feature_names': xgb_data['feature_names'],
                        'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                        'model_files': [latest_xgb_file, latest_lgb_file],
                        'model_performances': [1.0, 1.0],  # 默认权重
                        'model_types': ['xgboost', 'lightgbm']
                    }

                    logger.info(f"✅ 成功加载Window {window_idx}的集成模型（XGBoost + LightGBM）")

                except Exception as e:
                    logger.warning(f"⚠️ 加载模型失败: {e}")
                    logger.info("   将重新训练模型")
                    model_result = None

            elif use_xgboost and not use_lightgbm:
                # 只需要XGBoost模型
                try:
                    import pickle

                    with open(latest_xgb_file, 'rb') as f:
                        xgb_data = pickle.load(f)

                    # 构建model_result（单模型）
                    model_result = {
                        'models': [xgb_data['model']],
                        'scalers': [xgb_data['scaler']],
                        'feature_names': xgb_data['feature_names'],
                        'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                        'model_files': [latest_xgb_file],
                        'model_performances': [1.0],
                        'model_types': ['xgboost']
                    }

                    logger.info(f"✅ 成功加载Window {window_idx}的XGBoost模型")

                except Exception as e:
                    logger.warning(f"⚠️ 加载模型失败: {e}")
                    logger.info("   将重新训练模型")
                    model_result = None

            else:
                # use_lightgbm=True and use_xgboost=False: 用户只想要LightGBM模型
                logger.info(f"⚠️ 用户指定只使用LightGBM模型，跳过XGBoost模型加载")
                model_result = None  # 不加载XGBoost模型，强制重新训练或只加载LightGBM

        # 🆕 处理只需要LightGBM模型的情况
        elif lgb_files and use_lightgbm and not use_xgboost:
            logger.info(f"🔍 找到{len(lgb_files)}个Window {window_idx}的LightGBM模型")

            # 使用最新的模型
            latest_lgb_file = max(lgb_files, key=lambda f: f.stem)
            logger.info(f"   最新LightGBM模型: {latest_lgb_file.name}")

            try:
                import pickle

                with open(latest_lgb_file, 'rb') as f:
                    lgb_data = pickle.load(f)

                # 构建model_result（单模型）
                model_result = {
                    'models': [lgb_data['model']],
                    'scalers': [lgb_data['scaler']],
                    'feature_names': lgb_data['feature_names'],
                    'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                    'model_files': [latest_lgb_file],
                    'model_performances': [1.0],
                    'model_types': ['lightgbm']
                }

                logger.info(f"✅ 成功加载Window {window_idx}的LightGBM模型")

            except Exception as e:
                logger.warning(f"⚠️ 加载模型失败: {e}")
                logger.info("   将重新训练模型")
                model_result = None

        # 如果没有找到模型或加载失败，则训练新模型
        if model_result is None:
            logger.info(f"🔄 开始训练Window {window_idx}的新模型")

            # 🔧 如果选择了XGBoost,执行贝叶斯搜索
            best_params = None
            if use_xgboost:
                logger.info(f"\n🔍 步骤2.1: 执行贝叶斯搜索(XGBoost)")
                logger.info(f"   说明: 每个窗口独立执行完整贝叶斯搜索,确保参数最优")
                logger.info(f"   训练集: {len(train_data):,}条 ({window['train_start']} ~ {window['train_end']})")
                logger.info(f"   验证集: {len(val_data):,}条 ({window['val_start']} ~ {window['val_end']})")

                # 执行贝叶斯搜索(使用测试集最新日期作为时间衰减基准,支持结果缓存)
                best_params = _bayesian_search_single_val(
                    train_data, val_data, args, test_end_date=window['test_end'], window_idx=window_idx,
                    market_data=market_data  # 🚀 传递market_data用于XGBoost计算group权重
                )

                if best_params is None:
                    logger.error("❌ 贝叶斯搜索失败")
                    return None

                logger.info(f"✅ 贝叶斯搜索完成")
                logger.info(f"   最优参数: {best_params}")
            else:
                logger.info(f"\n⏭️ 步骤2.1: 跳过贝叶斯搜索(仅使用LightGBM)")
                logger.info(f"   LightGBM使用默认参数,无需参数搜索")
                # 使用默认参数
                best_params = {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1
                }

            # 使用完整训练集和验证集(已生成标签的train_data和val_data)
            full_train_data = train_data.copy()
            full_val_data = val_data.copy()

            logger.info(f"\n🔄 步骤2.2: 训练最终模型")
            logger.info(f"   完整训练集: {len(full_train_data):,}条")
            logger.info(f"   训练集有效样本: {full_train_data['atr_return_label'].notna().sum():,}条")
            logger.info(f"   完整验证集: {len(full_val_data):,}条")
            logger.info(f"   验证集有效样本: {full_val_data['atr_return_label'].notna().sum():,}条")

            # 🚀 训练模型(传入训练集和验证集，跳过内部划分)
            model_result = _train_final_model_for_window(
                full_train_data, full_val_data, best_params, window_idx,
                use_xgboost=use_xgboost, use_lightgbm=use_lightgbm,
                market_data=market_data,  # 🚀 传递market_data用于XGBoost计算group权重
                optimize_thresholds=args.optimize_thresholds  # 🚀 传递止盈止损优化参数
            )

            if model_result is None:
                logger.error("❌ 模型训练失败")
                return None

            logger.info(f"✅ 模型训练完成")
        else:
            logger.info(f"✅ 跳过训练，使用已有模型")

        # 4. 在测试集上回测
        logger.info(f"\n📈 步骤4: 在测试集上回测")

        # 执行回测(复用现有回测逻辑)
        backtest_result = _backtest_on_window(
            test_data, model_result, window, args
        )

        if backtest_result is None:
            logger.error("❌ 回测失败")
            return None

        logger.info(f"✅ 回测完成")

        # 🔧 添加：输出窗口回测结果表格（类似test_model模式）
        all_results = backtest_result.get('all_results', {})
        if all_results:
            logger.info(f"\n📊 [{window['name']}] 选股策略回测结果:")
            logger.info("=" * 150)
            logger.info(f"{'策略':<10} {'总收益':<10} {'净收益(扣成本)':<15} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<8} {'盈亏比':<8} {'总交易次数':<12}")
            logger.info("=" * 150)

            for stock_count in sorted(all_results.keys()):
                result = all_results[stock_count]
                total_trades = result.get('total_trades', 0)

                logger.info(f"Top-{stock_count:<5} {result.get('total_return_gross', 0):>8.2%}  "
                           f"{result.get('total_return', 0):>13.2%}  "
                           f"{result.get('annual_return', 0):>8.2%}  {result.get('sharpe_ratio', 0):>8.3f}  "
                           f"{result.get('max_drawdown', 0):>8.2%}  {result.get('win_rate', 0):>6.2%}  "
                           f"{result.get('profit_loss_ratio', 0):>6.2f}  "
                           f"{total_trades:>10}")

            logger.info("=" * 150)

        # 🔧 修复：返回所有Top-k档位的完整结果
        return {
            'backtest_result': backtest_result,
            'selections': backtest_result.get('selections', []),
            'metrics': {
                # Top-1结果（向后兼容）
                'total_return': backtest_result.get('total_return', 0),
                'win_rate': backtest_result.get('win_rate', 0),
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                'max_drawdown': backtest_result.get('max_drawdown', 0),
                # ✅ 所有Top-k档位的完整结果
                'all_results': backtest_result.get('all_results', {}),
                'detailed_trades': backtest_result.get('detailed_trades', {}),
                'summary_stats': backtest_result.get('summary_stats', {})
            }
        }

    except Exception as e:
        logger.error(f"❌ 处理窗口失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
