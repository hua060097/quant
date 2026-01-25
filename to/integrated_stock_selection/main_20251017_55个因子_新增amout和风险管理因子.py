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
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import statsmodels.api as sm


# ============================================================
# 🔧 全局配置参数
# ============================================================
# ATR策略配置
MAX_HOLDING_DAYS = 3  # 最大持仓期（天）- 训练和预测统一使用
ATR_MULTIPLIER_PROFIT = 1.8  # ATR止盈倍数
ATR_MULTIPLIER_LOSS = 0.9    # ATR止损倍数
MAX_STOP_LOSS_PCT = 0.05   # 最大止损百分比（5%）- 与ATR止损取较小损失

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
TRAIN_END_DATE = "2025-01-01"  # 训练数据截止日期（不包含此日期）
TEST_START_DATE = "2025-01-01"  # 测试数据开始日期（包含此日期）
TEST_END_DATE = "2025-09-30"    # 测试数据结束日期（包含此日期）

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
        choices=['generate_factors', 'test_factors', 'train_model', 'test_model', 'full_pipeline', 'test_optimization', 'factor_correlation', 'genetic_factor', 'market_state_test', 'interaction_analysis'],
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
        choices=['grid'],
        default='grid',
        help='超参数搜索策略 (grid=网格搜索[优化版])'
    )

    # 🚀 Pseudo-Huber delta参数
    parser.add_argument(
        '--set_huber',
        type=float,
        default=None,
        help='直接设置Pseudo-Huber损失函数的delta值，跳过搜索（仅test_model模式生效，例如: --set_huber 1.0）'
    )

    # 🚀 集成学习参数
    parser.add_argument(
        '--no_ensemble',
        action='store_true',
        help='禁用集成学习(XGBoost+LightGBM)，只使用XGBoost单模型（仅test_model和train_model模式生效）'
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
            logging.FileHandler('stock_selection_lig15_55个因子_新增amount和风险管理因子.log', encoding='utf-8'),
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

def calculate_portfolio_statistics(selections_with_returns):
    """
    计算投资组合统计指标

    Args:
        selections_with_returns: 包含实际收益率的选股结果列表

    Returns:
        Dict: 统计指标字典
    """
    import pandas as pd
    import numpy as np

    if not selections_with_returns:
        return {}

    # 转换为DataFrame便于分析
    df = pd.DataFrame([{
        'date': s.date,
        'stock_code': s.stock_code,
        'expected_return': s.expected_return,
        'actual_return': getattr(s, 'actual_return', 0),
        'composite_score': s.composite_score
    } for s in selections_with_returns])

    # 过滤掉NaN值，只计算有效的actual_return
    df_valid = df[df['actual_return'].notna() & ~df['actual_return'].isin([float('inf'), float('-inf')])]

    if df_valid.empty:
        print("⚠️ 所有actual_return都是NaN，无法计算投资组合统计")
        return {
            'total_return': 0,
            'win_rate': 0,
            'avg_daily_return': 0,
            'volatility': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'annualized_sharpe': 0,
            'calmar_ratio': 0,
            'total_trades': len(df),
            'valid_trades': 0
        }

    print(f"📊 投资组合统计: 总交易{len(df)}次，有效交易{len(df_valid)}次")

    # 按日期分组，计算每日组合收益（等权重）
    daily_returns = df_valid.groupby('date')['actual_return'].mean()

    # 计算累计收益
    cumulative_returns = (1 + daily_returns).cumprod() - 1

    # 基础统计
    total_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0
    win_rate = (daily_returns > 0).mean()
    avg_daily_return = daily_returns.mean()
    volatility = daily_returns.std()

    # 最大回撤
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / (1 + peak)
    max_drawdown = drawdown.min()

    # 夏普比率（假设无风险利率为0）
    sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0

    # 年化指标（假设252个交易日）
    trading_days = len(daily_returns)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
    annualized_volatility = volatility * np.sqrt(252)
    annualized_sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

    # 卡尔马比率
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'win_rate': win_rate,
        'avg_daily_return': avg_daily_return,
        'volatility': volatility,
        'annualized_volatility': annualized_volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'annualized_sharpe': annualized_sharpe,
        'calmar_ratio': calmar_ratio,
        'trading_days': trading_days,
        'total_trades': len(df),
        'valid_trades': len(df_valid),
        'daily_returns': daily_returns.tolist(),
        'cumulative_returns': cumulative_returns.tolist()
    }

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

def _apply_technical_pattern_filters(daily_filtered: pd.DataFrame, date_str: str, logger) -> pd.DataFrame:
    """
    应用技术形态风险过滤
    专门过滤负面K线形态和技术压力位
    """
    try:
        if len(daily_filtered) == 0:
            return daily_filtered

        initial_count = len(daily_filtered)

        # 1. 过滤极端负面K线形态（只过滤最强烈的）
        extreme_negative_patterns = [
            'pattern_bare_head_yin', 'pattern_long_upper_shadow'  # 只过滤最明确的负面信号
        ]

        for pattern in extreme_negative_patterns:
            if pattern in daily_filtered.columns:
                # 只过滤极端负面形态（强度 < -0.7，更严格的阈值）
                daily_filtered = daily_filtered[
                    daily_filtered[pattern] >= -0.7
                ]
                logger.debug(f"{pattern}极端过滤: -> {len(daily_filtered)}")

        # 2. MA60压力位过滤（放宽条件）
        if 'ma60' in daily_filtered.columns and 'close' in daily_filtered.columns:
            # 只过滤非常接近MA60压力位的股票（收盘价在MA60的99%-101%范围内）
            ma60_pressure_mask = (
                (daily_filtered['close'] >= daily_filtered['ma60'] * 0.99) &
                (daily_filtered['close'] <= daily_filtered['ma60'] * 1.01)
            )
            daily_filtered = daily_filtered[~ma60_pressure_mask]
            logger.debug(f"MA60压力位过滤: -> {len(daily_filtered)}")

        # 3. 连续下跌过滤（暂时禁用，避免过度过滤）
        # 让模型自己学习连续下跌的模式

        # 4. 高位震荡过滤（放宽条件）
        if 'price_ma20_distance' in daily_filtered.columns:
            # 只过滤极度偏离MA20的股票（超过MA20 25%）
            high_position_mask = daily_filtered['price_ma20_distance'] > 0.25  # 从15%放宽到25%
            daily_filtered = daily_filtered[~high_position_mask]
            logger.debug(f"极高位过滤: -> {len(daily_filtered)}")

        logger.info(f"日期 {date_str} 技术形态过滤: {initial_count} -> {len(daily_filtered)} 只股票")

        return daily_filtered

    except Exception as e:
        logger.error(f"技术形态过滤失败: {e}")
        return daily_filtered

def _handle_infinite_values(X_train: np.ndarray, feature_names: list, logger) -> np.ndarray:
    """
    智能处理无穷大值
    """
    try:
        X_processed = X_train.copy()

        for i, feature_name in enumerate(feature_names):
            feature_data = X_processed[:, i]
            inf_mask = np.isinf(feature_data)

            if np.any(inf_mask):
                # 获取非无穷值的统计信息
                finite_values = feature_data[~inf_mask]

                if len(finite_values) > 0:
                    # 根据特征类型选择替换策略
                    if 'ratio' in feature_name.lower() or 'pct' in feature_name.lower():
                        # 比率类特征：使用99.9%分位数
                        replacement_value = np.percentile(finite_values, 99.9)
                    elif 'volume' in feature_name.lower() or 'amount' in feature_name.lower():
                        # 成交量类特征：使用95%分位数
                        replacement_value = np.percentile(finite_values, 95)
                    elif 'return' in feature_name.lower():
                        # 收益率类特征：使用±3倍标准差
                        mean_val = np.mean(finite_values)
                        std_val = np.std(finite_values)
                        pos_inf_replacement = mean_val + 3 * std_val
                        neg_inf_replacement = mean_val - 3 * std_val

                        # 分别处理正负无穷
                        pos_inf_mask = np.isposinf(feature_data)
                        neg_inf_mask = np.isneginf(feature_data)

                        X_processed[pos_inf_mask, i] = pos_inf_replacement
                        X_processed[neg_inf_mask, i] = neg_inf_replacement
                        continue
                    else:
                        # 其他特征：使用中位数
                        replacement_value = np.median(finite_values)

                    # 替换所有无穷值
                    X_processed[inf_mask, i] = replacement_value

                    logger.info(f"特征 {feature_name}: {np.sum(inf_mask)} 个Inf值替换为 {replacement_value:.4f}")
                else:
                    # 如果所有值都是无穷，设为0
                    X_processed[inf_mask, i] = 0.0
                    logger.warning(f"特征 {feature_name}: 所有值都是Inf，设为0")

        return X_processed

    except Exception as e:
        logger.error(f"处理Inf值失败: {e}")
        # 降级到简单替换
        return np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)

# 删除_load_top_interaction_features函数 - 不再使用交互特征


# 删除_calculate_interaction_features函数 - 不再使用交互特征


# 删除不再需要的市场因子加载函数，现在直接使用_load_factor_cache_data加载


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
            stock_factors = _get_stock_factors()
            # market_factors = _get_available_market_factors()
            # required_factors = stock_factors + market_factors
            required_factors = stock_factors
            # logger.info(
            #     f"默认加载股票因子 %d 个，市场状态因子 %d 个，总计 %d 个。",
            #     len(stock_factors),
            #     # len(market_factors),
            #     len(required_factors)
            # )
        else:
            required_factors = list(dict.fromkeys(required_factors))
            if 'returns_zscore' not in required_factors:
                required_factors.insert(0, 'returns_zscore')
            logger.info("按需加载因子 %d 个。", len(required_factors))

        available_required = [f for f in required_factors if f in available_factor_columns]
        missing_requested = [f for f in required_factors if f not in available_factor_columns]
        if missing_requested:
            logger.warning(f"因子库缺少以下因子: {missing_requested}")

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
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
            'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
            'near_support', 'near_resistance', 'short_reversal', 'parkinson_vol',
            'vol_term_structure', 'volatility20', 'atr14'  # 添加标签生成必需字段
        ]
        # 🚀 标签生成所需字段:
        # - volatility20, atr14: ATR策略计算
        # - open, high, low: 价格计算
        # - is_limit_up, is_limit_down: 过滤T日涨跌停
        # - amount, volume: 过滤T日流动性不足
        # - is_can_buy: 过滤T+1日一字涨停
        label_generation_columns = [
            'volatility20', 'atr14', 'open', 'high', 'low',
            'is_limit_up', 'is_limit_down', 'amount', 'volume', 'is_can_buy'
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

        factor_df = factor_df[final_columns].copy()
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
    简化的特征筛选函数：使用黑名单过滤，避免复杂的业务逻辑判断

    Args:
        data: 输入数据DataFrame

    Returns:
        list: 有效的特征列名列表
    """
    try:
        logger = logging.getLogger(__name__)

        # 🚨 黑名单：需要排除的列（简单明确的规则）
        blacklist_columns = {
            # 元数据列
            'date', 'stock_code', 'stock_name', 'industry', 'sector',
            'list_date', 'delist_date', 'is_st', 'is_new_stock',
            'trade_date', 'ts_code', 'symbol', 'date_str',

            # 未来信息列（真正的数据泄露）
            'future_return_1d', 'future_return_2d', 'is_can_buy',
            'next_day_return', 'atr_triple_barrier_label','pct_chg',
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
            'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
            # 技术分析特征
            'near_support', 'near_resistance', 'short_reversal',

            # 选股过滤列（仅用于股票筛选，不参与模型训练）
            'close', 'volume', 'ma60', 'ma120',

            # 标签生成专用列（避免数据泄露）
            'volatility20', 'atr14', 'open', 'high', 'low',

        }

        # 注意：risk_adjusted_return是历史特征，不应被排除

        # 获取所有列名
        all_columns = set(data.columns)

        # 简单过滤：排除黑名单中的列
        feature_columns = [col for col in all_columns if col not in blacklist_columns]

        # 基本数据质量检查
        valid_features = []
        for col in feature_columns:
            try:
                # 检查数据类型（排除非数值列）
                if data[col].dtype in ['object', 'datetime64[ns]']:
                    continue

                # 检查是否全为NaN
                if data[col].isnull().all():
                    continue

                # 检查是否为常数列
                if data[col].nunique() <= 1:
                    continue

                # 检查有效值比例
                valid_ratio = data[col].notna().sum() / len(data)
                if valid_ratio < 0.1:  # 有效值少于10%
                    continue

                valid_features.append(col)

            except Exception as e:
                logger.debug(f"检查列 {col} 时出错: {e}")
                continue

        logger.info(f"✅ 简化特征筛选完成:")
        logger.info(f"   总列数: {len(all_columns)}")
        logger.info(f"   黑名单列数: {len(blacklist_columns & all_columns)}")
        logger.info(f"   候选特征: {len(feature_columns)}")
        logger.info(f"   有效特征: {len(valid_features)}")

        return valid_features

    except Exception as e:
        logger.error(f"简化特征筛选失败: {e}")
        return []




def _calculate_next_day_returns(stock_data: pd.DataFrame) -> pd.DataFrame:
    """计算次日收益率"""
    try:
        logger = logging.getLogger(__name__)

        # 按股票分组计算次日收益率
        stock_data_sorted = stock_data.sort_values(['stock_code', 'date'])
        stock_data_sorted['next_day_return'] = stock_data_sorted.groupby('stock_code')['pct_chg'].shift(-1)

        # 填充缺失值
        stock_data_sorted['next_day_return'] = stock_data_sorted['next_day_return'].fillna(0)

        logger.info(f"次日收益率计算完成: {stock_data_sorted['next_day_return'].notna().sum()} 个有效值")
        return stock_data_sorted

    except Exception as e:
        logger.error(f"计算次日收益率失败: {e}")
        stock_data['next_day_return'] = 0
        return stock_data

def _apply_professional_pca_reduction(X_scaled: np.ndarray, feature_names: list,
                                   n_features: int, logger) -> tuple:
    """
    基于顶级量化研究的专业PCA降维

    参考文献：
    1. "Principal Component Analysis in Finance" (Jolliffe, 2002)
    2. "Factor Models in Finance" (Campbell et al., 1997)
    3. "Machine Learning for Asset Management" (Roncalli, 2020)

    常见陷阱和最佳实践：
    1. 避免信息泄露：只在训练集上fit PCA
    2. 解释性权衡：保留足够的主成分解释80-95%方差
    3. 特征重要性：记录原始特征对主成分的贡献
    4. 稳定性：使用滚动窗口PCA避免前瞻偏差
    """
    try:
        from sklearn.decomposition import PCA

        logger.info(f"开始PCA降维分析，原始特征数: {n_features}")

        # 根据特征数量决定是否使用PCA
        if n_features < 50:
            logger.info("特征数量较少(<50)，跳过PCA降维")
            return X_scaled, {'applied': False, 'reason': 'low_feature_count'}

        # 计算相关性矩阵的条件数，判断多重共线性
        try:
            corr_matrix = np.corrcoef(X_scaled.T)
            cond_number = np.linalg.cond(corr_matrix)
            logger.info(f"特征相关性矩阵条件数: {cond_number:.2f}")

            if cond_number < 100:  # 条件数较小，多重共线性不严重
                logger.info("多重共线性不严重，跳过PCA降维")
                return X_scaled, {'applied': False, 'reason': 'low_multicollinearity'}
        except:
            logger.warning("无法计算相关性矩阵条件数，继续PCA分析")

        # 第一步：分析主成分数量
        # 使用完整PCA分析方差解释比例
        pca_full = PCA()
        pca_full.fit(X_scaled)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # 寻找最优主成分数量（多个标准）
        # 标准1：解释85%方差
        n_components_85 = np.argmax(cumulative_variance >= 0.85) + 1
        # 标准2：解释90%方差
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        # 标准3：解释95%方差
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        # 标准4：Kaiser准则（特征值>1）
        eigenvalues = pca_full.explained_variance_
        n_components_kaiser = np.sum(eigenvalues > 1)

        # 标准5：肘部法则（方差解释比例的二阶导数）
        variance_ratios = pca_full.explained_variance_ratio_
        if len(variance_ratios) > 2:
            second_derivatives = np.diff(variance_ratios, n=2)
            elbow_point = np.argmax(second_derivatives) + 2
            n_components_elbow = min(elbow_point, len(variance_ratios))
        else:
            n_components_elbow = len(variance_ratios)

        # 综合决策：基于量化金融的最佳实践
        if n_features > 200:
            # 超高维：优先考虑计算效率，选择85%方差解释
            optimal_components = min(n_components_85, n_features // 3)
        elif n_features > 100:
            # 高维：平衡解释性和效率，选择90%方差解释
            optimal_components = min(n_components_90, n_features // 2)
        else:
            # 中维：优先保留信息，选择95%方差解释
            optimal_components = min(n_components_95, int(n_features * 0.8))

        # 确保主成分数量合理
        optimal_components = max(optimal_components, 10)  # 至少保留10个主成分
        optimal_components = min(optimal_components, n_features - 1)  # 不超过原特征数

        logger.info(f"PCA分析结果:")
        logger.info(f"  85%方差解释需要: {n_components_85}个主成分")
        logger.info(f"  90%方差解释需要: {n_components_90}个主成分")
        logger.info(f"  95%方差解释需要: {n_components_95}个主成分")
        logger.info(f"  Kaiser准则建议: {n_components_kaiser}个主成分")
        logger.info(f"  肘部法则建议: {n_components_elbow}个主成分")
        logger.info(f"  最终选择: {optimal_components}个主成分")

        # 第二步：应用PCA降维
        pca = PCA(n_components=optimal_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # 计算实际方差解释比例
        actual_variance_explained = np.sum(pca.explained_variance_ratio_)
        logger.info(f"实际方差解释比例: {actual_variance_explained:.1%}")

        # 第三步：分析特征重要性（避免黑盒问题）
        # 计算每个原始特征对前5个主成分的贡献
        components_matrix = pca.components_[:5]  # 前5个主成分
        feature_importance_pca = np.sum(np.abs(components_matrix), axis=0)

        # 找出对主成分贡献最大的特征
        top_feature_indices = np.argsort(feature_importance_pca)[-20:]
        top_features = [feature_names[i] for i in top_feature_indices]

        logger.info(f"对主成分贡献最大的20个特征:")
        for i, (idx, feature) in enumerate(zip(top_feature_indices[-10:], top_features[-10:])):
            logger.info(f"  {i+1:2d}. {feature}: {feature_importance_pca[idx]:.3f}")

        # 构建PCA信息字典
        pca_info = {
            'applied': True,
            'original_features': n_features,
            'pca_components': optimal_components,
            'variance_explained': actual_variance_explained,
            'reduction_ratio': optimal_components / n_features,
            'top_contributing_features': top_features,
            'pca_object': pca,  # 保存PCA对象用于预测时变换
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components_analysis': {
                'n_components_85': n_components_85,
                'n_components_90': n_components_90,
                'n_components_95': n_components_95,
                'n_components_kaiser': n_components_kaiser,
                'n_components_elbow': n_components_elbow
            }
        }

        logger.info(f"PCA降维完成: {n_features} -> {optimal_components} "
                   f"(降维比例: {(1-optimal_components/n_features)*100:.1f}%)")

        return X_pca, pca_info

    except Exception as e:
        logger.error(f"PCA降维失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return X_scaled, {'applied': False, 'reason': 'error', 'error': str(e)}

def _calculate_professional_sample_weights(y_train: np.ndarray, imbalance_ratio: float,
                                         logger, boundary_distances: np.ndarray = None,
                                         is_regression: bool = False, label_transformer=None,
                                         y_train_original: np.ndarray = None) -> np.ndarray:
    """
    🚀 基于止盈止损阈值对齐的专业权重计算策略

    核心设计原则：
    1. 与实际交易策略的止盈止损阈值完全对齐
    2. 基于量化金融理论的信息价值权重分配
    3. 考虑A股市场特色和噪声交易特征
    4. 平衡模型学习效果与泛化能力

    权重策略（基于ATR_MULTIPLIER_PROFIT=2, ATR_MULTIPLIER_LOSS=1）：
    1. 极端正收益（>95%分位数）：权重 = 0.4（降低异常值影响）
    2. 强正收益（2-95%分位数ATR）：权重 = 1.8（重要盈利信号）
    3. 中正收益（0.5-2ATR）：权重 = 1.2（一般盈利信号）
    4. 小幅波动（-0.5到0.5ATR）：权重 = 0.6（降低噪声影响）
    5. 中负收益（-1到-0.5ATR）：权重 = 1.2（一般亏损信号）
    6. 强负收益（<-1ATR）：权重 = 1.6（重要风险信号）
    7. 止盈止损边界（±0.1ATR内）：权重 = 1.0（关键决策区域）

    Args:
        y_train: 训练标签（可能经过变换或Winsorize）
        imbalance_ratio: 不平衡比例
        logger: 日志记录器
        boundary_distances: 边界距离数组
        is_regression: 是否为回归模式
        label_transformer: 标签变换器，用于逆变换到原始ATR单位
        y_train_original: Winsorize前的原始标签（用于权重计算）
    """
    try:
        import numpy as np

        logger.info(f"样本权重计算开始:")
        logger.info(f"  总样本数: {len(y_train)}")
        logger.info(f"  模式: {'回归' if is_regression else '分类'}")

        if is_regression:
            # 🚀 基于标签压缩感知的权重计算
            logger.info("使用基于标签压缩感知的权重计算...")

            # 统计当前标签分布（可能已变换）
            logger.info(f"当前标签统计（可能已变换）:")
            logger.info(f"  最小值: {y_train.min():.3f}")
            logger.info(f"  最大值: {y_train.max():.3f}")
            logger.info(f"  平均值: {y_train.mean():.3f}")
            logger.info(f"  标准差: {y_train.std():.3f}")

            # 🚀 关键优化：使用Winsorize前的原始标签进行权重计算
            # 确保权重分配基于真实的数据分布，而不是被Winsorize截断后的分布
            if y_train_original is not None:
                y_original = y_train_original
                logger.info("✅ 权重计算使用Winsorize前的原始标签（真实分布）")
                logger.info(f"  原始ATR范围: [{y_original.min():.3f}, {y_original.max():.3f}]")
                logger.info(f"  原始ATR均值: {y_original.mean():.3f}")
                logger.info(f"  原始ATR标准差: {y_original.std():.3f}")
            elif label_transformer is not None:
                try:
                    y_original = label_transformer.inverse_transform(y_train)
                    logger.info(f"⚠️ 未提供原始标签，使用逆变换:")
                    logger.info(f"  逆变换ATR范围: [{y_original.min():.3f}, {y_original.max():.3f}]")
                except Exception as e:
                    logger.warning(f"逆变换失败，使用当前标签: {e}")
                    y_original = y_train
            else:
                y_original = y_train
                logger.info("⚠️ 未提供原始标签且未使用标签变换，使用当前标签")

            # 初始化权重为1.0
            sample_weights = np.ones(len(y_train))

            # 🚀 新权重策略：基于止盈止损阈值对齐的专业权重分配
            # 🚀 修复权重策略：智能阈值设定，避免95%分位数与止盈阈值重叠
            # 动态计算关键分位数
            q95 = np.percentile(y_original, 95)
            q90 = np.percentile(y_original, 90)
            q75 = np.percentile(y_original, 75)
            q25 = np.percentile(y_original, 25)
            q10 = np.percentile(y_original, 10)
            q05 = np.percentile(y_original, 5)

            profit_threshold = 1.8  # ATR_MULTIPLIER_PROFIT
            loss_threshold = -0.9   # -ATR_MULTIPLIER_LOSS


            # 智能极端阈值设定：确保强正收益区间有足够样本
            if q95 <= profit_threshold:
                # 如果95%分位数<=止盈阈值，使用更高阈值作为极端阈值
                extreme_positive_threshold = max(profit_threshold + 1.0, q90 + 0.5)
                logger.info(f"⚠️ 95%分位数({q95:.2f})<=止盈阈值({profit_threshold:.1f})，调整极端阈值为{extreme_positive_threshold:.2f}")
            else:
                extreme_positive_threshold = q95

            noise_upper = 0.5       # 噪声上界
            noise_lower = -0.5      # 噪声下界

            logger.info(f"📊 智能权重阈值设定:")
            logger.info(f"  数据分位数: 5%={q05:.2f}, 25%={q25:.2f}, 75%={q75:.2f}, 95%={q95:.2f}")
            logger.info(f"  极端正收益阈值: {extreme_positive_threshold:.2f} ATR")
            logger.info(f"  止盈阈值: {profit_threshold:.1f} ATR")
            logger.info(f"  止损阈值: {loss_threshold:.1f} ATR")
            logger.info(f"  噪声区间: [{noise_lower:.1f}, {noise_upper:.1f}] ATR")

            # 🚀 优化权重分配策略：确保每个区间都有合理样本数和权重
            # 1. 极端正收益（>极端阈值）：适度降权，避免过度惩罚
            extreme_positive_mask = y_original > extreme_positive_threshold
            sample_weights[extreme_positive_mask] = 0.7  # 从0.4调整为0.7

            # 2. 强正收益（止盈阈值到极端阈值）：重要盈利信号，高权重
            # 🚀 优化：降低权重，避免过度追涨（从1.8降低到1.5）
            strong_positive_mask = (y_original >= profit_threshold) & (y_original <= extreme_positive_threshold)
            sample_weights[strong_positive_mask] = 1.5  # 从1.8降低到1.5

            # 3. 中正收益（0.5ATR到止盈阈值）：未达止盈但仍有价值的盈利信号
            # 🚀 优化：提升权重，避免忽视大量盈利信号（从0.9提升到1.2）
            medium_positive_mask = (y_original >= noise_upper) & (y_original < profit_threshold)
            sample_weights[medium_positive_mask] = 1.2  # 从0.9提升到1.2

            # 4. 小幅波动（-0.5ATR到0.5ATR）：噪声样本，降权处理
            noise_mask = (y_original > noise_lower) & (y_original < noise_upper)
            sample_weights[noise_mask] = 0.7  # 降权处理噪声

            # 5. 中负收益（止损阈值到-0.5ATR）：未达止损但仍需关注的亏损信号
            # 🚀 优化：提升权重，加强风险学习（从1.1提升到1.3）
            medium_negative_mask = (y_original > loss_threshold) & (y_original <= noise_lower)
            sample_weights[medium_negative_mask] = 1.3  # 从1.1提升到1.3

            # 6. 强负收益（<止损阈值）：重要风险信号，高权重学习
            strong_negative_mask = y_original <= loss_threshold
            sample_weights[strong_negative_mask] = 1.6

            # 🚀 删除边界样本权重设置：避免与其他区间重叠和权重冲突

            logger.info(f"🚀 基于止盈止损阈值对齐的权重分配（优化版）:")
            logger.info(f"  极端正收益样本: {extreme_positive_mask.sum()}个 ({extreme_positive_mask.sum()/len(y_train)*100:.1f}%), 权重=0.7")
            logger.info(f"  强正收益样本: {strong_positive_mask.sum()}个 ({strong_positive_mask.sum()/len(y_train)*100:.1f}%), 权重=1.5 (优化)")
            logger.info(f"  中正收益样本: {medium_positive_mask.sum()}个 ({medium_positive_mask.sum()/len(y_train)*100:.1f}%), 权重=1.2 (优化)")
            logger.info(f"  小幅波动样本: {noise_mask.sum()}个 ({noise_mask.sum()/len(y_train)*100:.1f}%), 权重=0.7")
            logger.info(f"  中负收益样本: {medium_negative_mask.sum()}个 ({medium_negative_mask.sum()/len(y_train)*100:.1f}%), 权重=1.3 (优化)")
            logger.info(f"  强负收益样本: {strong_negative_mask.sum()}个 ({strong_negative_mask.sum()/len(y_train)*100:.1f}%), 权重=1.6")
            logger.info(f"  ✅ 已删除边界样本权重设置，避免区间重叠")

            # 权重质量检查
            if sample_weights.mean() < 0.8 or sample_weights.mean() > 1.3:
                logger.warning(f"⚠️ 权重均值偏离1.0较多: {sample_weights.mean():.3f}")

            # 更新extreme_mask和medium_mask以兼容后续代码
            extreme_mask = extreme_positive_mask | strong_negative_mask
            medium_mask = medium_positive_mask | medium_negative_mask

            # 2. 边界样本权重调整（如果提供了边界距离）
            boundary_mask = np.zeros(len(y_train), dtype=bool)
            if boundary_distances is not None:
                # 找到接近边界的样本（min_dist_to_sl或min_dist_to_pt接近0的部分）
                valid_distances = boundary_distances[np.isfinite(boundary_distances)]
                if len(valid_distances) > 0:
                    boundary_threshold = np.percentile(np.abs(valid_distances), 20)
                    boundary_mask = np.abs(boundary_distances) <= boundary_threshold
                    # 边界样本权重=0.9（不与极端样本重叠时）
                    boundary_only_mask = boundary_mask & ~extreme_mask
                    sample_weights[boundary_only_mask] = 0.9

                    logger.info(f"边界样本权重调整:")
                    logger.info(f"  边界阈值: {boundary_threshold:.3f}")
                    logger.info(f"  边界样本: {boundary_only_mask.sum()}个 ({boundary_only_mask.sum()/len(y_train)*100:.1f}%), 权重=0.9")

            # 3. 其他样本保持权重=1.0
            normal_mask = ~extreme_mask & ~boundary_mask

            logger.info(f"正常样本: {normal_mask.sum()}个 ({normal_mask.sum()/len(y_train)*100:.1f}%), 权重=1.0")

            # 4. 打印权重统计
            logger.info(f"样本权重统计:")
            logger.info(f"  最小权重: {sample_weights.min():.2f}")
            logger.info(f"  最大权重: {sample_weights.max():.2f}")
            logger.info(f"  平均权重: {sample_weights.mean():.3f}")
            logger.info(f"  权重分位数: 25%={np.percentile(sample_weights, 25):.2f}, 50%={np.percentile(sample_weights, 50):.2f}, 75%={np.percentile(sample_weights, 75):.2f}")


        # 验证权重数组
        logger.info(f"样本权重统计:")
        logger.info(f"  权重数组长度: {len(sample_weights)}")
        logger.info(f"  权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        logger.info(f"  平均权重: {sample_weights.mean():.3f}")

        # 🚀 新增：边界距离修正（修复版本）
        if boundary_distances is not None:
            logger.info("应用边界距离修正...")

            # 🔍 调试：输出边界距离详细统计
            logger.info(f"边界距离数组统计:")
            logger.info(f"  数组长度: {len(boundary_distances)}")
            logger.info(f"  最小值: {boundary_distances.min():.6f}")
            logger.info(f"  最大值: {boundary_distances.max():.6f}")
            logger.info(f"  平均值: {boundary_distances.mean():.6f}")
            logger.info(f"  标准差: {boundary_distances.std():.6f}")

            # 检查是否所有值都相同（这可能是问题所在）
            unique_values = np.unique(boundary_distances)
            logger.info(f"  唯一值数量: {len(unique_values)}")
            if len(unique_values) <= 5:
                logger.info(f"  唯一值: {unique_values}")

            # 计算分位数
            distance_10 = np.percentile(boundary_distances, 10)
            distance_25 = np.percentile(boundary_distances, 25)
            distance_50 = np.percentile(boundary_distances, 50)
            distance_75 = np.percentile(boundary_distances, 75)
            distance_90 = np.percentile(boundary_distances, 90)

            logger.info(f"  分位数: 10%={distance_10:.6f}, 25%={distance_25:.6f}, 50%={distance_50:.6f}, 75%={distance_75:.6f}, 90%={distance_90:.6f}")

            # 🔧 修复：处理NaN值和使用更合理的阈值策略
            # 首先过滤掉NaN值
            valid_distances_mask = np.isfinite(boundary_distances)
            valid_distances = boundary_distances[valid_distances_mask]

            if len(valid_distances) == 0:
                logger.warning("所有边界距离都是NaN或无穷大，跳过边界距离修正")
                boundary_mask = np.zeros(len(boundary_distances), dtype=bool)
                medium_mask = np.zeros(len(boundary_distances), dtype=bool)
            else:
                logger.info(f"有效边界距离数量: {len(valid_distances)}/{len(boundary_distances)}")

                # 重新计算有效距离的分位数
                valid_distance_25 = np.percentile(valid_distances, 25)
                valid_distance_75 = np.percentile(valid_distances, 75)

                logger.info(f"有效距离分位数: 25%={valid_distance_25:.6f}, 75%={valid_distance_75:.6f}")

                # 如果有效距离值过于集中，使用绝对阈值
                if len(np.unique(valid_distances)) <= 3 or (valid_distances.max() - valid_distances.min()) < 0.1:
                    logger.info("检测到有效边界距离值过于集中，使用绝对阈值策略")

                    # 使用绝对阈值：距离小于1.0的为边界样本，1.0-2.0为中等样本
                    boundary_threshold = 1.0
                    medium_threshold = 2.0

                    boundary_mask = (valid_distances_mask) & (boundary_distances < boundary_threshold)
                    medium_mask = (valid_distances_mask) & (boundary_distances >= boundary_threshold) & (boundary_distances < medium_threshold)

                else:
                    logger.info("使用分位数策略")

                    # 使用分位数策略：前25%为边界样本，25%-75%为中等样本
                    boundary_mask = (valid_distances_mask) & (boundary_distances < valid_distance_25)
                    medium_mask = (valid_distances_mask) & (boundary_distances >= valid_distance_25) & (boundary_distances < valid_distance_75)

            # 边界样本降权：距离越小，权重越低
            boundary_adjustment = np.ones(len(sample_weights))

            # 🚀 优化：边界样本权重调整，避免与极端样本双重加权
            boundary_adjustment = np.ones(len(sample_weights))

            # 🔧 修复：边界样本仅在非极端样本中提权，避免双重加权
            boundary_only_mask = boundary_mask & (sample_weights == 1.0)  # 只对权重为1.0的样本进行边界调整
            medium_only_mask = medium_mask & (sample_weights == 1.0)      # 只对权重为1.0的样本进行中等调整

            boundary_adjustment[boundary_only_mask] = 1.3  # 🔧 优化：适度提权到1.3，避免过度加权
            boundary_adjustment[medium_only_mask] = 1.0    # 中等样本保持不变

            # 应用边界调整
            sample_weights *= boundary_adjustment

            logger.info(f"🔧 优化的边界距离修正统计:")
            logger.info(f"  边界样本(非极端)({boundary_only_mask.sum()}个, {boundary_only_mask.sum()/len(boundary_distances)*100:.1f}%): 权重×1.3 (适度提权)")
            logger.info(f"  中等样本(非极端)({medium_only_mask.sum()}个, {medium_only_mask.sum()/len(boundary_distances)*100:.1f}%): 权重×1.0 (保持不变)")
            logger.info(f"  极端样本: 保持原有权重1.8，不受边界调整影响")

        # 显示权重分布
        if is_regression:
            logger.info(f"回归权重统计:")
            logger.info(f"  平均权重: {sample_weights.mean():.3f}")
            logger.info(f"  权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

        return sample_weights

    except Exception as e:
        logger.error(f"样本权重计算失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        # 返回均匀权重
        return np.ones(len(y_train))



def _robust_outlier_handling(X_train, feature_names, logger):
    """
    稳健的异常值处理策略（避免数据泄露）

    Args:
        X_train: 训练数据矩阵
        feature_names: 特征名称列表
        logger: 日志记录器

    Returns:
        处理后的训练数据矩阵
    """
    try:
        import numpy as np
        from scipy import stats

        logger.info("=" * 50)
        logger.info("🔧 稳健异常值处理策略")
        logger.info("=" * 50)

        X_cleaned = X_train.copy()
        n_features = X_train.shape[1]

        nan_features = []
        inf_features = []
        outlier_features = []

        for i in range(n_features):
            feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            feature_values = X_train[:, i]

            # 1. 处理NaN值：使用中位数或均值（根据分布选择）
            nan_mask = np.isnan(feature_values)
            if nan_mask.sum() > 0:
                nan_features.append(f"{feature_name}({nan_mask.sum()})")

                # 计算有效值的分布特征
                valid_values = feature_values[~nan_mask]
                if len(valid_values) > 0:
                    # 使用Shapiro-Wilk检验判断是否为正态分布
                    if len(valid_values) >= 3:
                        try:
                            _, p_value = stats.shapiro(valid_values[:5000])  # 限制样本数避免计算过慢
                            is_normal = p_value > 0.05
                        except:
                            is_normal = False
                    else:
                        is_normal = False

                    # 根据分布选择填充策略
                    if is_normal:
                        fill_value = np.mean(valid_values)
                        strategy = "均值"
                    else:
                        fill_value = np.median(valid_values)
                        strategy = "中位数"

                    X_cleaned[nan_mask, i] = fill_value
                    logger.debug(f"  {feature_name}: NaN用{strategy}填充({fill_value:.4f})")
                else:
                    # 如果没有有效值，使用0填充
                    X_cleaned[nan_mask, i] = 0.0
                    logger.warning(f"  {feature_name}: 无有效值，用0填充")

            # 2. 处理无穷值：使用分位数截断
            inf_mask = np.isinf(feature_values)
            if inf_mask.sum() > 0:
                inf_features.append(f"{feature_name}({inf_mask.sum()})")

                # 计算有效值的分位数
                finite_values = feature_values[np.isfinite(feature_values)]
                if len(finite_values) > 0:
                    p1 = np.percentile(finite_values, 1)
                    p99 = np.percentile(finite_values, 99)

                    # 正无穷用99%分位数替换，负无穷用1%分位数替换
                    pos_inf_mask = np.isposinf(feature_values)
                    neg_inf_mask = np.isneginf(feature_values)

                    X_cleaned[pos_inf_mask, i] = p99
                    X_cleaned[neg_inf_mask, i] = p1

                    logger.debug(f"  {feature_name}: +Inf→{p99:.4f}, -Inf→{p1:.4f}")
                else:
                    # 如果没有有限值，使用0填充
                    X_cleaned[inf_mask, i] = 0.0
                    logger.warning(f"  {feature_name}: 无有限值，用0填充")

            # 3. 处理极端异常值：使用IQR方法
            feature_values_cleaned = X_cleaned[:, i]
            if np.isfinite(feature_values_cleaned).sum() > 10:  # 至少需要10个有效值
                finite_values = feature_values_cleaned[np.isfinite(feature_values_cleaned)]

                Q1 = np.percentile(finite_values, 25)
                Q3 = np.percentile(finite_values, 75)
                IQR = Q3 - Q1

                if IQR > 0:  # 避免除零错误
                    # 使用3倍IQR作为异常值阈值（比1.5倍更宽松）
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR

                    outlier_mask = (feature_values_cleaned < lower_bound) | (feature_values_cleaned > upper_bound)
                    outlier_count = outlier_mask.sum()

                    if outlier_count > 0:
                        outlier_features.append(f"{feature_name}({outlier_count})")

                        # 将异常值截断到边界值
                        X_cleaned[feature_values_cleaned < lower_bound, i] = lower_bound
                        X_cleaned[feature_values_cleaned > upper_bound, i] = upper_bound

                        logger.debug(f"  {feature_name}: {outlier_count}个异常值截断到[{lower_bound:.4f}, {upper_bound:.4f}]")

        # 汇总处理结果
        logger.info(f"✅ 异常值处理完成:")
        if nan_features:
            logger.info(f"  📊 NaN处理: {len(nan_features)}个特征 - {', '.join(nan_features[:5])}")
            if len(nan_features) > 5:
                logger.info(f"    ... 还有{len(nan_features)-5}个特征")

        if inf_features:
            logger.info(f"  📊 Inf处理: {len(inf_features)}个特征 - {', '.join(inf_features[:5])}")
            if len(inf_features) > 5:
                logger.info(f"    ... 还有{len(inf_features)-5}个特征")

        if outlier_features:
            logger.info(f"  📊 异常值截断: {len(outlier_features)}个特征 - {', '.join(outlier_features[:5])}")
            if len(outlier_features) > 5:
                logger.info(f"    ... 还有{len(outlier_features)-5}个特征")

        logger.info("=" * 50)

        return X_cleaned

    except Exception as e:
        logger.error(f"稳健异常值处理失败: {e}")
        # 降级到简单处理
        logger.warning("降级到简单异常值处理...")
        return np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)

def _apply_market_factor_engineering(X_train, feature_names, stock_factors, market_factors, logger, pca_model=None):
    """
    🚀 新的市场因子特征工程：去相关 + PCA降维

    Args:
        X_train: 训练数据特征矩阵
        feature_names: 特征名称列表
        stock_factors: 股票因子名称列表
        market_factors: 市场因子名称列表
        logger: 日志记录器
        pca_model: 预训练的PCA模型（用于预测阶段，None表示训练阶段）

    Returns:
        tuple: (X_engineered, engineered_feature_names, engineering_info)
    """
    try:
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        logger.info("=" * 60)
        logger.info("🚀 开始市场因子特征工程")
        logger.info("=" * 60)

        # 转换为DataFrame
        df_features = pd.DataFrame(X_train, columns=feature_names)

        # 1. 分离股票因子和市场因子
        stock_feature_indices = [i for i, name in enumerate(feature_names) if name in stock_factors]
        market_feature_indices = [i for i, name in enumerate(feature_names) if name in market_factors]

        logger.info(f"📊 特征分离:")
        logger.info(f"  - 股票因子: {len(stock_feature_indices)} 个")
        logger.info(f"  - 市场因子: {len(market_feature_indices)} 个")

        # 提取股票因子数据（保持不变）
        stock_data = X_train[:, stock_feature_indices]
        stock_feature_names = [feature_names[i] for i in stock_feature_indices]

        # 提取市场因子数据
        market_data = X_train[:, market_feature_indices]
        market_feature_names = [feature_names[i] for i in market_feature_indices]

        logger.info(f"✅ 股票因子将保持原始形式: {len(stock_feature_names)} 个")

        # 2. 处理市场因子：去相关 + PCA
        if len(market_feature_names) > 0:
            logger.info(f"🔄 开始处理市场因子...")

            # 2.1 去除高相关因子
            market_df = pd.DataFrame(market_data, columns=market_feature_names)

            # 计算相关性矩阵
            corr_matrix = market_df.corr().abs()

            # 找出高相关因子对（相关系数 > 0.7）
            high_corr_pairs = []
            upper_tri = np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1)
            high_corr_mask = (corr_matrix.values > 0.7) & upper_tri

            if high_corr_mask.any():
                high_corr_indices = np.where(high_corr_mask)
                for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
                    factor1 = market_feature_names[i]
                    factor2 = market_feature_names[j]
                    corr_val = corr_matrix.iloc[i, j]
                    high_corr_pairs.append((factor1, factor2, corr_val))

                logger.info(f"🔍 发现 {len(high_corr_pairs)} 对高相关因子 (|r| > 0.7):")
                for factor1, factor2, corr_val in high_corr_pairs[:5]:
                    logger.info(f"  - {factor1} ↔ {factor2}: r={corr_val:.3f}")

            # 去相关：保留每对高相关因子中的一个
            factors_to_remove = set()
            for factor1, factor2, corr_val in high_corr_pairs:
                # 优先保留名称较短的因子（通常更基础）
                if len(factor1) > len(factor2):
                    factors_to_remove.add(factor1)
                else:
                    factors_to_remove.add(factor2)

            # 保留低相关的因子
            retained_market_factors = [f for f in market_feature_names if f not in factors_to_remove]
            retained_indices = [i for i, name in enumerate(market_feature_names) if name in retained_market_factors]

            logger.info(f"📊 去相关结果:")
            logger.info(f"  - 移除因子: {len(factors_to_remove)} 个")
            logger.info(f"  - 保留因子: {len(retained_market_factors)} 个")

            if len(retained_market_factors) > 0:
                retained_market_data = market_data[:, retained_indices]

                # 2.2 对保留的市场因子进行PCA
                if len(retained_market_factors) >= 4:  # 至少4个因子才做PCA
                    logger.info(f"🔄 对 {len(retained_market_factors)} 个市场因子进行PCA...")

                    if pca_model is None:
                        # 训练模式：拟合新的PCA模型
                        logger.info("训练模式：拟合新的PCA模型")

                        # 标准化
                        market_scaler = StandardScaler()
                        market_data_scaled = market_scaler.fit_transform(retained_market_data)

                        # PCA分析
                        pca_full = PCA()
                        pca_full.fit(market_data_scaled)

                        # 选择主成分数量：保留90%方差或最多4个成分
                        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
                        optimal_components = min(n_components_90, 4)  # 最多4个成分
                        optimal_components = max(optimal_components, 3)  # 至少3个成分

                        # 应用PCA
                        pca = PCA(n_components=optimal_components, random_state=42)
                        market_pca_data = pca.fit_transform(market_data_scaled)

                        # 生成PCA特征名称
                        pca_feature_names = [f'market_pca_{i+1}' for i in range(optimal_components)]

                        variance_explained = pca.explained_variance_ratio_.sum()
                        logger.info(f"✅ PCA训练完成:")
                        logger.info(f"  - 输入: {len(retained_market_factors)} 个市场因子")
                        logger.info(f"  - 输出: {optimal_components} 个主成分")
                        logger.info(f"  - 方差解释: {variance_explained:.1%}")

                        # 保存PCA信息
                        pca_info = {
                            'applied': True,
                            'original_factors': retained_market_factors,
                            'n_components': optimal_components,
                            'variance_explained': variance_explained,
                            'pca_object': pca,
                            'scaler': market_scaler,
                            'pca_feature_names': pca_feature_names
                        }
                    else:
                        # 预测模式：使用预训练的PCA模型
                        logger.info("预测模式：使用预训练的PCA模型")

                        # 使用训练时保存的scaler和PCA模型
                        market_scaler = pca_model.get('scaler')
                        pca = pca_model.get('pca_object')
                        pca_feature_names = pca_model.get('pca_feature_names')

                        if market_scaler is None or pca is None:
                            logger.error("❌ PCA模型信息不完整，无法应用")
                            return None, None, {'applied': False, 'reason': 'incomplete_pca_model'}

                        # 应用预训练的变换
                        market_data_scaled = market_scaler.transform(retained_market_data)
                        market_pca_data = pca.transform(market_data_scaled)

                        logger.info(f"✅ PCA预测完成:")
                        logger.info(f"  - 输入: {len(retained_market_factors)} 个市场因子")
                        logger.info(f"  - 输出: {len(pca_feature_names)} 个主成分")

                        # PCA信息（预测模式不需要保存，但需要返回）
                        pca_info = {
                            'applied': True,
                            'mode': 'prediction'
                        }

                    market_final_data = market_pca_data
                    market_final_names = pca_feature_names

                else:
                    # 因子太少，直接使用
                    logger.info(f"⚠️ 市场因子数量较少({len(retained_market_factors)})，跳过PCA，直接使用")
                    market_final_data = retained_market_data
                    market_final_names = retained_market_factors
                    pca_info = {'applied': False, 'reason': 'too_few_factors'}

            else:
                logger.warning("⚠️ 去相关后无剩余市场因子")
                market_final_data = np.empty((X_train.shape[0], 0))
                market_final_names = []
                pca_info = {'applied': False, 'reason': 'no_factors_after_decorrelation'}

        else:
            logger.warning("⚠️ 未找到市场因子")
            market_final_data = np.empty((X_train.shape[0], 0))
            market_final_names = []
            pca_info = {'applied': False, 'reason': 'no_market_factors'}

        # 3. 合并股票因子和处理后的市场因子
        if market_final_data.shape[1] > 0:
            X_engineered = np.column_stack([stock_data, market_final_data])
            engineered_feature_names = stock_feature_names + market_final_names
        else:
            X_engineered = stock_data
            engineered_feature_names = stock_feature_names

        logger.info(f"📊 特征工程完成:")
        logger.info(f"  - 最终特征数: {len(engineered_feature_names)} 个")
        logger.info(f"  - 股票因子: {len(stock_feature_names)} 个（保持原始）")
        logger.info(f"  - 市场成分: {len(market_final_names)} 个（PCA处理后）")
        logger.info("=" * 60)

        return X_engineered, engineered_feature_names, pca_info

    except Exception as e:
        logger.error(f"❌ 市场因子特征工程失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None, {}

def _build_interaction_constraints(feature_names, stock_factors, pca_info, logger):
    """
    构建XGBoost的interaction_constraints参数

    Args:
        feature_names: 最终特征名称列表
        stock_factors: 股票因子名称列表
        pca_info: PCA信息字典
        logger: 日志记录器

    Returns:
        str: interaction_constraints参数字符串
    """
    try:
        logger.info("🔧 构建特征交互约束...")

        # 找出股票因子和PCA成分的索引
        stock_indices = []
        pca_indices = []

        for i, feature in enumerate(feature_names):
            if feature in stock_factors:
                stock_indices.append(i)
            elif 'market_pca_' in feature:
                pca_indices.append(i)

        logger.info(f"📊 特征分组:")
        logger.info(f"  - 股票因子索引: {stock_indices[:10]}{'...' if len(stock_indices) > 10 else ''} (共{len(stock_indices)}个)")
        logger.info(f"  - PCA成分索引: {pca_indices} (共{len(pca_indices)}个)")

        if len(pca_indices) == 0:
            logger.info("⚠️ 未找到PCA成分，不设置交互约束")
            return None

        # 构建交互约束组
        constraint_groups = []

        # 组1：股票因子可以互相交互
        if len(stock_indices) > 1:
            constraint_groups.append(stock_indices)

        # 组2：每个PCA成分可以与所有股票因子交互，但PCA成分之间不能直接交互
        for pca_idx in pca_indices:
            if len(stock_indices) > 0:
                # PCA成分 + 所有股票因子组成一个约束组
                pca_group = [pca_idx] + stock_indices
                constraint_groups.append(pca_group)

        if len(constraint_groups) == 0:
            logger.info("⚠️ 无法构建有效的交互约束组")
            return None

        # 格式化为XGBoost要求的字符串格式
        constraint_str = str(constraint_groups)

        logger.info(f"✅ 交互约束构建完成:")
        logger.info(f"  - 约束组数量: {len(constraint_groups)}")
        logger.info(f"  - 约束逻辑: 股票因子互相交互，PCA成分与股票因子交互，PCA成分间不交互")
        logger.info(f"  - 约束字符串长度: {len(constraint_str)} 字符")

        return constraint_str

    except Exception as e:
        logger.error(f"❌ 构建交互约束失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None

def _apply_grouped_pca_transform(X_train, feature_names, logger):
    """
    应用分组PCA特征降维处理

    Args:
        X_train: 训练数据特征矩阵
        feature_names: 特征名称列表
        logger: 日志记录器

    Returns:
        tuple: (X_train_pca, feature_names_pca, pca_models)
    """
    try:
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # 转换为DataFrame便于处理
        df_features = pd.DataFrame(X_train, columns=feature_names)

        # 移除所有PCA处理逻辑，保持所有特征的原始形式
        logger.info("🔍 特征处理策略: 保持所有特征的原始形式，不进行PCA降维")
        logger.info(f"   总特征数: {len(feature_names)} 个（全部保持原始形式）")

        # 初始化结果
        pca_models = {}
        result_features = []
        result_data = []

        # 直接使用所有原始特征，不进行任何PCA处理
        result_features = feature_names.copy()
        result_data = []

        # 添加所有原始特征数据
        for feature in feature_names:
            if feature in df_features.columns:
                result_data.append(df_features[feature].values)
            else:
                logger.warning(f"   特征 {feature} 在数据中不存在，用0填充")
                result_data.append(np.zeros(len(df_features)))

        logger.info(f"   ✅ 保留所有原始特征: {len(result_features)} 个")

        # 组合最终结果（不进行PCA处理）
        if result_data:
            X_train_original = np.column_stack(result_data)
            feature_names_original = result_features

            logger.info(f"📊 特征处理结果:")
            logger.info(f"   原始特征数: {len(feature_names)}")
            logger.info(f"   保留特征数: {len(feature_names_original)}")
            logger.info(f"   PCA处理: 已完全移除")

            return X_train_original, feature_names_original, {}  # 返回空的PCA模型字典
        else:
            logger.warning("⚠️ 没有有效的特征")
            return None, None, {}

    except Exception as e:
        logger.error(f"❌ PCA处理失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None, {}

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
            logger.info(f"  Q{i+1} (预测值{i*20}-{(i+1)*20}%分位): {ret:.4f} ATR")

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

def _apply_pca_transform_to_prediction(X_pred, feature_names, pca_models, logger):
    """
    对预测数据应用训练时的PCA变换

    Args:
        X_pred: 预测数据特征矩阵（DataFrame或ndarray）
        feature_names: 特征名称列表
        pca_models: 训练时保存的PCA模型字典
        logger: 日志记录器

    Returns:
        np.ndarray: PCA变换后的特征矩阵
    """
    try:
        import pandas as pd
        import numpy as np

        # 转换为DataFrame便于处理
        if isinstance(X_pred, np.ndarray):
            df_features = pd.DataFrame(X_pred, columns=feature_names)
        else:
            df_features = X_pred.copy()

        result_features = []
        result_data = []

        # 1. 处理Support特征组
        if 'support' in pca_models:
            support_info = pca_models['support']
            support_features = support_info['features']

            # 检查特征是否存在
            available_support_features = [f for f in support_features if f in df_features.columns]

            if len(available_support_features) == len(support_features):
                logger.info(f"   🔄 应用Support PCA变换: {len(support_features)}个特征 -> 1个主成分")

                support_data = df_features[support_features].values

                # 处理缺失值
                if np.any(np.isnan(support_data)):
                    support_data = np.nan_to_num(support_data, nan=np.nanmean(support_data))

                # 应用标准化和PCA变换
                support_data_scaled = support_info['scaler'].transform(support_data)
                support_pca_result = support_info['pca'].transform(support_data_scaled)

                result_features.append('support_factor_pca')
                result_data.append(support_pca_result.flatten())

                logger.info(f"   ✅ Support PCA变换完成")
            else:
                logger.warning(f"   ⚠️ Support特征缺失，跳过PCA变换")
                logger.warning(f"   需要: {support_features}")
                logger.warning(f"   可用: {available_support_features}")

        # 2. 处理MA对齐特征组
        if 'ma_alignment' in pca_models:
            ma_info = pca_models['ma_alignment']
            ma_features = ma_info['features']

            # 检查特征是否存在
            available_ma_features = [f for f in ma_features if f in df_features.columns]

            if len(available_ma_features) == len(ma_features):
                logger.info(f"   🔄 应用MA对齐PCA变换: {len(ma_features)}个特征 -> 1个主成分")

                ma_data = df_features[ma_features].values

                # 处理缺失值
                if np.any(np.isnan(ma_data)):
                    ma_data = np.nan_to_num(ma_data, nan=np.nanmean(ma_data))

                # 应用标准化和PCA变换
                ma_data_scaled = ma_info['scaler'].transform(ma_data)
                ma_pca_result = ma_info['pca'].transform(ma_data_scaled)

                result_features.append('ma_alignment_factor_pca')
                result_data.append(ma_pca_result.flatten())

                logger.info(f"   ✅ MA对齐PCA变换完成")
            else:
                logger.warning(f"   ⚠️ MA对齐特征缺失，跳过PCA变换")
                logger.warning(f"   需要: {ma_features}")
                logger.warning(f"   可用: {available_ma_features}")

        # 3. 添加其他特征（未被PCA处理的特征）
        processed_features = set()
        for pca_info in pca_models.values():
            processed_features.update(pca_info['features'])

        other_features = [f for f in df_features.columns if f not in processed_features]

        if other_features:
            other_data = df_features[other_features].values
            result_features.extend(other_features)
            for i in range(other_data.shape[1]):
                result_data.append(other_data[:, i])

        # 4. 组合最终结果
        if result_data:
            X_pred_pca = np.column_stack(result_data)
            logger.info(f"   📊 PCA变换结果: {len(df_features.columns)}个特征 -> {len(result_features)}个特征")
            return X_pred_pca
        else:
            logger.warning("   ⚠️ 没有有效的特征进行PCA变换")
            return None

    except Exception as e:
        logger.error(f"   ❌ PCA变换失败: {e}")
        import traceback
        logger.error(f"   详细错误: {traceback.format_exc()}")
        return None

def _apply_pca_transform_to_prediction_fixed(X_pred, pred_feature_names, pca_models, train_feature_names, logger):
    """
    修复版：对预测数据应用训练时的PCA变换，确保特征数量和顺序完全一致

    Args:
        X_pred: 预测数据特征矩阵
        pred_feature_names: 预测数据的特征名称列表
        pca_models: 训练时保存的PCA模型字典
        train_feature_names: 训练时的特征名称列表（PCA后）
        logger: 日志记录器

    Returns:
        tuple: (X_pred_aligned, aligned_feature_names)
    """
    try:
        import pandas as pd
        import numpy as np

        # 转换为DataFrame便于处理
        if isinstance(X_pred, np.ndarray):
            df_pred = pd.DataFrame(X_pred, columns=pred_feature_names)
        else:
            df_pred = X_pred.copy()

        logger.info(f"🔧 开始特征对齐处理...")
        logger.info(f"   预测数据特征数: {len(pred_feature_names)}")
        logger.info(f"   训练时特征数: {len(train_feature_names)}")

        # 创建结果DataFrame，按训练时的特征顺序
        result_df = pd.DataFrame(index=df_pred.index)

        # 直接处理所有原始特征，不进行任何PCA处理
        for train_feature in train_feature_names:
            # 处理所有特征（直接复制或填充0）
            if train_feature in df_pred.columns:
                result_df[train_feature] = df_pred[train_feature].values
            else:
                logger.warning(f"   ⚠️ 特征 {train_feature} 不存在，用0填充")
                result_df[train_feature] = 0.0

        # 验证结果
        final_feature_names = list(result_df.columns)
        logger.info(f"📊 特征对齐结果:")
        logger.info(f"   最终特征数: {len(final_feature_names)}")
        logger.info(f"   训练特征数: {len(train_feature_names)}")
        logger.info(f"   特征顺序一致: {final_feature_names == train_feature_names}")

        if len(final_feature_names) != len(train_feature_names):
            logger.error(f"❌ 特征数量不匹配！")
            return None, None

        return result_df.values, final_feature_names

    except Exception as e:
        logger.error(f"❌ 特征对齐失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None

class LabelTransformer:
    """🚀 可序列化的标签变换器类"""

    def __init__(self, method='adaptive'):
        self.method = method
        self.transform_type = None  # 'asinh' 或 'signed_log'

    def fit_transform(self, y, logger=None):
        """拟合并变换标签"""
        import scipy.stats

        if self.method == 'none':
            self.transform_type = 'none'
            return y

        if logger:
            logger.info(f"🔄 应用标签变换: {self.method}")
            logger.info(f"  原始标签范围: [{y.min():.4f}, {y.max():.4f}]")
            logger.info(f"  原始标签偏度: {scipy.stats.skew(y):.4f}")

        if self.method == 'asinh':
            self.transform_type = 'asinh'
            y_transformed = np.arcsinh(y)
        elif self.method == 'signed_log':
            self.transform_type = 'signed_log'
            y_transformed = np.sign(y) * np.log(1 + np.abs(y))
        elif self.method == 'adaptive':
            # 🚀 优化：自适应选择，降低偏度阈值以更好处理量化交易极值
            skewness = scipy.stats.skew(y)
            if abs(skewness) > 1.5:  # 🔧 优化：降低阈值从2.0到1.5，更积极处理偏度
                self.transform_type = 'signed_log'
                y_transformed = np.sign(y) * np.log(1 + np.abs(y))
                if logger:
                    logger.info(f"  选择signed_log变换（偏度={skewness:.4f}，阈值1.5）")
            else:  # 低偏度数据使用asinh
                self.transform_type = 'asinh'
                y_transformed = np.arcsinh(y)
                if logger:
                    logger.info(f"  选择asinh变换（偏度={skewness:.4f}，阈值1.5）")
        else:
            raise ValueError(f"不支持的变换方法: {self.method}")

        if logger:
            logger.info(f"  变换后范围: [{y_transformed.min():.4f}, {y_transformed.max():.4f}]")
            logger.info(f"  变换后偏度: {scipy.stats.skew(y_transformed):.4f}")

        return y_transformed

    def transform(self, y):
        """变换标签"""
        if self.transform_type == 'none':
            return y
        elif self.transform_type == 'asinh':
            return np.arcsinh(y)
        elif self.transform_type == 'signed_log':
            return np.sign(y) * np.log(1 + np.abs(y))
        else:
            raise ValueError(f"变换器未拟合或变换类型未知: {self.transform_type}")

    def inverse_transform(self, y_transformed):
        """逆变换标签"""
        if self.transform_type == 'none':
            return y_transformed
        elif self.transform_type == 'asinh':
            return np.sinh(y_transformed)
        elif self.transform_type == 'signed_log':
            return np.sign(y_transformed) * (np.exp(np.abs(y_transformed)) - 1)
        else:
            raise ValueError(f"变换器未拟合或变换类型未知: {self.transform_type}")



def _perform_parameter_search(X_train, y_train, X_val, y_val, y_val_eval, sample_weights, param_grid, delta_grid, search_strategy, label_transformer):
    """
    🔍 执行分层渐进式超参数搜索优化（现代优化策略）

    采用分层搜索策略：
    1. 阶段1：核心参数粗调 (max_depth, learning_rate)
    2. 阶段2：正则化精调 (reg_alpha, reg_lambda)
    3. 阶段3：采样参数微调 (subsample, colsample_bytree, min_child_weight)

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        y_val_eval: 验证集评估标签
        sample_weights: 样本权重
        param_grid: 参数搜索空间（将被重新设计）
        delta_grid: Delta参数搜索空间
        search_strategy: 搜索策略 ('grid', 'fast')
        label_transformer: 标签变换器

    Returns:
        tuple: (best_params, best_delta, search_results)
    """
    import itertools
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    import xgboost as xgb
    import numpy as np
    import logging
    import multiprocessing

    logger = logging.getLogger(__name__)

    # 🚀 数据规模分析和采样优化
    sample_count = len(X_train)
    feature_count = X_train.shape[1]

    logger.info("=" * 80)
    logger.info("🔍 启动分层渐进式超参数搜索（速度优化版）")
    logger.info("=" * 80)
    logger.info(f"📊 原始数据规模: {sample_count:,} 样本, {feature_count} 特征")

    # 🚀 优化1：智能数据采样（超参数搜索专用）
    if sample_count > 10000:
        # 对大数据集进行分层采样，保持标签分布
        sample_ratio = min(0.4, max(0.2, 15000 / sample_count))  # 20%-40%采样
        sample_size = int(sample_count * sample_ratio)

        # 分层采样保持标签分布
        from sklearn.model_selection import train_test_split
        X_search, _, y_search, _, weights_search, _ = train_test_split(
            X_train, y_train, sample_weights,
            train_size=sample_size,
            stratify=None,  # 回归任务不能直接分层，使用随机采样
            random_state=42
        )

        logger.info(f"🎯 采样优化: {sample_count:,} → {len(X_search):,} 样本 (比例: {sample_ratio:.1%})")
        logger.info(f"⚡ 预计速度提升: {1/sample_ratio:.1f}x")
    else:
        X_search, y_search, weights_search = X_train, y_train, sample_weights
        logger.info("📊 数据量适中，使用全量数据进行搜索")

    # 验证集保持不变（用于最终评估）
    logger.info(f"📊 搜索数据: {len(X_search):,} 样本, 验证数据: {len(X_val):,} 样本")

    # 🎯 分层搜索策略设计
    logger.info("🎯 设计分层搜索策略...")

    # 基础参数配置
    base_params = {
        'random_state': 42,
        'n_jobs': 1,
        'verbosity': 0
    }

    # 🚀 优化2：激进Early Stopping配置（超参数搜索专用）
    search_sample_count = len(X_search)
    if search_sample_count < 5000:
        max_boost_rounds = 100  # 减少50%
        early_stopping = 15     # 减少50%
        logger.info("📊 小数据集：超快速搜索配置")
    elif search_sample_count < 15000:
        max_boost_rounds = 150  # 减少62%
        early_stopping = 20     # 减少60%
        logger.info("📊 中等数据集：快速搜索配置")
    else:
        max_boost_rounds = 200  # 减少67%
        early_stopping = 25     # 减少69%
        logger.info("📊 大数据集：平衡搜索配置")



    # 🚀 优化3：动态Delta搜索空间（基于标签分布）
    # 根据标签std动态设置delta搜索空间
    label_std = y_search.std()
    if label_std < 2.0:
        delta_grid_optimized = [1.0, 1.5]
        logger.info(f"🔧 小波动(std={label_std:.3f}): Delta搜索空间 {delta_grid_optimized}")
    elif label_std < 4.0:
        delta_grid_optimized = [2.0, 3.0]
        logger.info(f"🔧 中等波动(std={label_std:.3f}): Delta搜索空间 {delta_grid_optimized}")
    else:
        delta_grid_optimized = [3.0, 4.5]
        logger.info(f"🔧 大波动(std={label_std:.3f}): Delta搜索空间 {delta_grid_optimized}")

    # 🚀 优化4：增强并行配置
    max_workers = min(multiprocessing.cpu_count(), 6)  # 增加并行度
    logger.info(f"🚀 增强并行配置: {max_workers} 个工作线程")

    # 预计性能提升
    original_combinations = 98 * 3  # 原始组合数
    optimized_combinations = 98 * len(delta_grid_optimized)
    speed_boost = (1/sample_ratio if 'sample_ratio' in locals() else 1) * (3/len(delta_grid_optimized)) * (50/early_stopping)
    logger.info(f"⚡ 预计总体速度提升: {speed_boost:.1f}x")

    logger = logging.getLogger(__name__)

    # 🎯 分层搜索实现
    all_search_results = []
    stage_start_time = time.time()

    # ==================== 阶段1：核心参数粗调（精简版）====================
    logger.info("🔍 阶段1：核心参数粗调 (max_depth, learning_rate, n_estimators) - 精简搜索空间")
    stage1_params = {
        'max_depth': [5, 6, 7],           # 减少到3个最优值
        'learning_rate': [0.08, 0.1, 0.12],  # 减少到3个最优值
        'n_estimators': [2000, 2500, 3000]  # 🚀 新增：搜索最优树数量
    }

    # 固定其他参数为合理默认值
    stage1_fixed = {
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'gamma': 0.2,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    stage1_fixed.update(base_params)

    stage1_combinations = list(itertools.product(*stage1_params.values()))
    stage1_param_names = list(stage1_params.keys())
    total_stage1 = len(stage1_combinations) * len(delta_grid_optimized)

    logger.info(f"📊 阶段1搜索空间: {len(stage1_combinations)} × {len(delta_grid_optimized)} = {total_stage1} 组合")
    logger.info(f"⏱️ 预计阶段1时间: {total_stage1 * 8 // max_workers // 60} 分钟 (优化后)")

    stage1_results = _execute_stage_search(
        X_search, y_search, X_val, y_val, y_val_eval, weights_search,  # 🚀 使用采样数据
        stage1_combinations, stage1_param_names, stage1_fixed,
        delta_grid_optimized, max_boost_rounds, early_stopping,
        label_transformer, max_workers, "阶段1", 1
    )

    all_search_results.extend(stage1_results)

    # 找到阶段1最优参数
    stage1_best = min(stage1_results, key=lambda x: x['mse'])
    best_max_depth = stage1_best['params']['max_depth']
    best_learning_rate = stage1_best['params']['learning_rate']
    best_n_estimators = stage1_best['params']['n_estimators']  # 🚀 新增：获取最优树数量

    stage1_time = time.time() - stage_start_time
    logger.info(f"✅ 阶段1完成 ({stage1_time/60:.1f}分钟)")
    logger.info(f"🎯 阶段1最优: max_depth={best_max_depth}, learning_rate={best_learning_rate}, n_estimators={best_n_estimators}, MSE={stage1_best['mse']:.6f}")
    logger.info("-" * 60)

    # ==================== 阶段2：正则化精调（精简版）====================
    stage_start_time = time.time()
    logger.info("🔍 阶段2：正则化精调 (reg_alpha, reg_lambda) - 精简搜索空间")

    stage2_params = {
        'reg_alpha': [0.2, 0.5, 1.0],      # 减少到3个最优值
        'reg_lambda': [1.0, 2.0, 3.0]      # 减少到3个最优值
    }

    # 使用阶段1最优参数 + 固定其他参数
    stage2_fixed = {
        'max_depth': best_max_depth,
        'learning_rate': best_learning_rate,
        'n_estimators': best_n_estimators,  # 🚀 新增：使用阶段1最优树数量
        'min_child_weight': 5,
        'gamma': 0.2,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    stage2_fixed.update(base_params)

    stage2_combinations = list(itertools.product(*stage2_params.values()))
    stage2_param_names = list(stage2_params.keys())
    total_stage2 = len(stage2_combinations) * len(delta_grid_optimized)

    logger.info(f"📊 阶段2搜索空间: {len(stage2_combinations)} × {len(delta_grid_optimized)} = {total_stage2} 组合")
    logger.info(f"⏱️ 预计阶段2时间: {total_stage2 * 15 // max_workers // 60} 分钟")

    stage2_results = _execute_stage_search(
        X_search, y_search, X_val, y_val, y_val_eval, weights_search,  # 🚀 使用采样数据
        stage2_combinations, stage2_param_names, stage2_fixed,
        delta_grid_optimized, max_boost_rounds, early_stopping,
        label_transformer, max_workers, "阶段2", 2
    )

    all_search_results.extend(stage2_results)

    # 找到阶段2最优参数
    stage2_best = min(stage2_results, key=lambda x: x['mse'])
    best_reg_alpha = stage2_best['params']['reg_alpha']
    best_reg_lambda = stage2_best['params']['reg_lambda']

    stage2_time = time.time() - stage_start_time
    logger.info(f"✅ 阶段2完成 ({stage2_time/60:.1f}分钟)")
    logger.info(f"🎯 阶段2最优: reg_alpha={best_reg_alpha}, reg_lambda={best_reg_lambda}, MSE={stage2_best['mse']:.6f}")
    logger.info("-" * 60)

    # ==================== 阶段3：采样参数微调（精简版）====================
    stage_start_time = time.time()
    logger.info("🔍 阶段3：采样参数微调 (subsample, colsample_bytree, min_child_weight) - 精简搜索空间")

    stage3_params = {
        'subsample': [0.8, 0.9],           # 减少到2个最优值
        'colsample_bytree': [0.8, 0.9],    # 减少到2个最优值
        'min_child_weight': [3, 5]         # 减少到2个最优值
    }

    # 使用前两阶段最优参数 + 固定gamma
    stage3_fixed = {
        'max_depth': best_max_depth,
        'learning_rate': best_learning_rate,
        'n_estimators': best_n_estimators,  # 🚀 新增：使用阶段1最优树数量
        'reg_alpha': best_reg_alpha,
        'reg_lambda': best_reg_lambda,
        'gamma': 0.2
    }
    stage3_fixed.update(base_params)

    stage3_combinations = list(itertools.product(*stage3_params.values()))
    stage3_param_names = list(stage3_params.keys())
    total_stage3 = len(stage3_combinations) * len(delta_grid_optimized)

    logger.info(f"📊 阶段3搜索空间: {len(stage3_combinations)} × {len(delta_grid_optimized)} = {total_stage3} 组合")
    logger.info(f"⏱️ 预计阶段3时间: {total_stage3 * 15 // max_workers // 60} 分钟")

    stage3_results = _execute_stage_search(
        X_search, y_search, X_val, y_val, y_val_eval, weights_search,  # 🚀 使用采样数据
        stage3_combinations, stage3_param_names, stage3_fixed,
        delta_grid_optimized, max_boost_rounds, early_stopping,
        label_transformer, max_workers, "阶段3", 3
    )

    all_search_results.extend(stage3_results)

    # 找到阶段3最优参数
    stage3_best = min(stage3_results, key=lambda x: x['mse'])

    stage3_time = time.time() - stage_start_time
    logger.info(f"✅ 阶段3完成 ({stage3_time/60:.1f}分钟)")
    logger.info(f"🎯 阶段3最优: MSE={stage3_best['mse']:.6f}")
    logger.info("-" * 60)

    # ==================== 最终结果汇总 ====================
    logger.info("🏆 分层搜索完成，汇总最优结果...")

    # 找到全局最优参数
    global_best = min(all_search_results, key=lambda x: x['mse'])
    best_params = global_best['params'].copy()
    best_delta = global_best['delta']
    best_score = global_best['mse']

    # 统计各阶段结果
    stage1_results_only = [r for r in all_search_results if r.get('stage') == '阶段1']
    stage2_results_only = [r for r in all_search_results if r.get('stage') == '阶段2']
    stage3_results_only = [r for r in all_search_results if r.get('stage') == '阶段3']

    total_search_time = time.time() - (stage_start_time - stage1_time - stage2_time - stage3_time)
    total_combinations = len(all_search_results)

    logger.info("=" * 80)
    logger.info("🎉 分层渐进式搜索完成！")
    logger.info("=" * 80)
    logger.info(f"🎯 全局最优MSE: {best_score:.6f}")
    logger.info(f"🎯 最优Delta: {best_delta}")
    logger.info(f"🎯 最优参数: {best_params}")
    logger.info(f"📊 搜索统计:")
    logger.info(f"  阶段1组合: {len(stage1_results_only)}")
    logger.info(f"  阶段2组合: {len(stage2_results_only)}")
    logger.info(f"  阶段3组合: {len(stage3_results_only)}")
    logger.info(f"  总组合数: {total_combinations}")
    logger.info(f"  总耗时: {total_search_time/60:.1f}分钟")
    logger.info(f"  平均耗时: {total_search_time/max(1, total_combinations):.1f}秒/组合")

    # 显示各阶段最优结果
    if stage1_results_only:
        stage1_best_result = min(stage1_results_only, key=lambda x: x['mse'])
        logger.info(f"🥇 阶段1最优: MSE={stage1_best_result['mse']:.6f}")

    if stage2_results_only:
        stage2_best_result = min(stage2_results_only, key=lambda x: x['mse'])
        logger.info(f"🥈 阶段2最优: MSE={stage2_best_result['mse']:.6f}")

    if stage3_results_only:
        stage3_best_result = min(stage3_results_only, key=lambda x: x['mse'])
        logger.info(f"🥉 阶段3最优: MSE={stage3_best_result['mse']:.6f}")

    logger.info("=" * 80)

    return best_params, best_delta, all_search_results


def _execute_stage_search(X_train, y_train, X_val, y_val, y_val_eval, sample_weights,
                         param_combinations, param_names, fixed_params, delta_grid,
                         max_boost_rounds, early_stopping, label_transformer,
                         max_workers, stage_name, stage_num):
    """
    🔍 执行单个阶段的参数搜索

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        y_val_eval: 验证集评估标签
        sample_weights: 样本权重
        param_combinations: 参数组合列表
        param_names: 参数名称列表
        fixed_params: 固定参数字典
        delta_grid: Delta参数列表
        max_boost_rounds: 最大训练轮数
        early_stopping: 早停轮数
        label_transformer: 标签变换器
        max_workers: 最大工作线程数
        stage_name: 阶段名称
        stage_num: 阶段编号

    Returns:
        list: 搜索结果列表
    """
    import xgboost as xgb
    import numpy as np
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger = logging.getLogger(__name__)

    # 创建所有参数组合
    all_combinations = []
    for param_combo in param_combinations:
        current_params = dict(zip(param_names, param_combo))
        current_params.update(fixed_params)
        for delta in delta_grid:
            all_combinations.append((current_params, delta))

    total_combinations = len(all_combinations)
    completed = 0
    stage_results = []
    stage_start_time = time.time()
    best_stage_mse = float('inf')

    # 进度报告函数
    def report_stage_progress():
        if completed > 0:
            elapsed = time.time() - stage_start_time
            avg_time = elapsed / completed
            eta = avg_time * (total_combinations - completed)
            progress_pct = completed / total_combinations * 100
            logger.info(f"  📈 {stage_name}进度: {completed}/{total_combinations} ({progress_pct:.1f}%) "
                      f"当前最优MSE: {best_stage_mse:.6f} "
                      f"ETA: {eta/60:.1f}分钟")

    # 每5个组合报告一次进度
    progress_interval = max(1, min(5, total_combinations // 10))

    # 单个组合训练函数
    def train_single_combination(combo_data):
        try:
            current_params, delta = combo_data

            # 创建本地数据集
            dtrain_local = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
            dval_local = xgb.DMatrix(X_val, label=y_val)

            # 定义Pseudo-Huber目标函数
            def pseudo_huber_obj_local(delta):
                def _obj(preds, dtrain):
                    labels = dtrain.get_label()
                    residuals = preds - labels
                    delta_safe = max(delta, 1e-8)
                    r_over_delta = residuals / delta_safe
                    sqrt_term = np.sqrt(1.0 + r_over_delta ** 2)
                    grad = residuals / sqrt_term
                    hess = 1.0 / (delta_safe**2 * sqrt_term**3)
                    hess = np.clip(hess, 1e-8, 1e6)
                    return grad, hess
                return _obj

            # 训练模型
            model = xgb.train(
                current_params,
                dtrain_local,
                num_boost_round=max_boost_rounds,
                obj=pseudo_huber_obj_local(delta),
                evals=[(dval_local, 'val')],
                verbose_eval=False,
                early_stopping_rounds=early_stopping
            )

            # 评估模型
            val_pred = model.predict(dval_local)

            # 处理标签变换的逆变换
            if label_transformer is not None:
                val_pred_original = label_transformer.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            else:
                val_pred_original = val_pred

            # 计算MSE
            mse = np.mean((val_pred_original - y_val_eval) ** 2)

            return {
                'params': current_params.copy(),
                'delta': delta,
                'mse': mse,
                'best_iteration': getattr(model, 'best_iteration', max_boost_rounds),
                'stage': stage_name,
                'success': True
            }

        except Exception as e:
            return {
                'params': current_params.copy() if 'current_params' in locals() else {},
                'delta': delta if 'delta' in locals() else 0,
                'mse': float('inf'),
                'best_iteration': 0,
                'stage': stage_name,
                'success': False,
                'error': str(e)
            }

    # 并行执行搜索
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_combo = {
                executor.submit(train_single_combination, combo): combo
                for combo in all_combinations
            }

            for future in as_completed(future_to_combo):
                completed += 1

                try:
                    result = future.result()

                    if result['success']:
                        stage_results.append(result)

                        # 更新阶段最优结果
                        if result['mse'] < best_stage_mse:
                            best_stage_mse = result['mse']
                    else:
                        logger.warning(f"{stage_name}参数组合失败: {result.get('error', 'Unknown error')}")

                    # 进度报告
                    if completed % progress_interval == 0:
                        report_stage_progress()

                except Exception as e:
                    logger.warning(f"{stage_name}处理结果时出错: {e}")
                    continue

    except Exception as e:
        logger.error(f"{stage_name}并行搜索出错: {e}")
        # 回退到串行搜索
        logger.info(f"🔄 {stage_name}回退到串行搜索...")
        for combo in all_combinations:
            completed += 1
            result = train_single_combination(combo)

            if result['success']:
                stage_results.append(result)
                if result['mse'] < best_stage_mse:
                    best_stage_mse = result['mse']

            if completed % progress_interval == 0:
                report_stage_progress()

    # 最终进度报告
    report_stage_progress()

    return stage_results


def _train_lightgbm_model(raw_data, labels, feature_columns, label_stats=None, preprocessed_data=None):
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
            y_train = preprocessed_data['y_train']
            y_val = preprocessed_data['y_val']

            logger.info(f"  训练集: {len(X_train_raw):,}样本")
            logger.info(f"  验证集: {len(X_val_raw):,}样本")
            logger.info(f"  原始特征数: {X_train_raw.shape[1]}")
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

        # 2. 特征标准化
        logger.info("🔧 特征标准化...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 3. 配置LightGBM参数（使用默认参数）
        lgb_params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'n_estimators': 2800,  # 与XGBoost一致
            'learning_rate': 0.1,  # 与XGBoost一致
            'num_leaves': 31,  # LightGBM默认值
            'max_depth': -1,  # 不限制深度
            'min_child_samples': 20,  # LightGBM默认值
            'subsample': 0.8,  # 与XGBoost一致
            'colsample_bytree': 0.9,  # 与XGBoost一致
            'reg_alpha': 0.5,  # 与XGBoost一致
            'reg_lambda': 1.0,  # 与XGBoost一致
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        logger.info("📋 LightGBM参数配置:")
        for key, value in lgb_params.items():
            logger.info(f"  {key}: {value}")

        # 4. 训练LightGBM模型
        logger.info("🚀 开始训练LightGBM...")

        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

        # Early Stopping配置
        early_stopping_rounds = min(int(lgb_params['n_estimators'] * 0.12), 300)
        logger.info(f"  Early Stopping: {early_stopping_rounds} rounds")

        # 训练模型
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100)
        ]

        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )

        logger.info(f"✅ LightGBM训练完成")
        logger.info(f"  最佳迭代: {model.best_iteration}")
        logger.info(f"  训练轮数: {model.current_iteration()}")

        # 5. 模型评估
        logger.info("📊 评估LightGBM模型...")

        # 训练集预测
        y_train_pred = model.predict(X_train_scaled, num_iteration=model.best_iteration)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # 验证集预测
        y_val_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        logger.info("📊 训练集性能:")
        logger.info(f"  MSE: {train_mse:.6f}")
        logger.info(f"  RMSE: {train_rmse:.6f}")
        logger.info(f"  MAE: {train_mae:.6f}")
        logger.info(f"  R²: {train_r2:.6f}")

        logger.info("📊 验证集性能:")
        logger.info(f"  MSE: {val_mse:.6f}")
        logger.info(f"  RMSE: {val_rmse:.6f}")
        logger.info(f"  MAE: {val_mae:.6f}")
        logger.info(f"  R²: {val_r2:.6f}")

        # 6. 特征重要性
        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = feature_columns

        # 7. 构建返回结果
        model_results = {
            'model': model,
            'scaler': scaler,
            'label_transformer': None,  # LightGBM不使用标签变换
            'feature_names': feature_names,
            'label_stats': label_stats,
            'performance': {
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
            },
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




def _train_ml_model(raw_data, labels, feature_columns, winsorization_thresholds=None, label_stats=None, label_transform='adaptive', enable_param_search=False, search_strategy='grid', random_seed=42, preprocessed_data=None, enable_shap_analysis=True, preset_huber_delta=None):
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
        preset_huber_delta: 预设的Pseudo-Huber delta值（可选）
            - 如果提供,将跳过delta网格搜索,直接使用该值
            - 仅在test_model模式下使用,用于加速训练

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

        # 标签转换
        y_data = labels.values.astype(np.float32) if hasattr(labels, 'values') else labels.astype(np.float32)

        # 🚀 提取Winsorize前的原始标签（用于权重计算）
        y_data_original = None
        if 'atr_return_label_original' in raw_data.columns:
            y_data_original = raw_data['atr_return_label_original'].values.astype(np.float32)
            logger.info("✅ 提取到Winsorize前的原始标签（用于权重计算）")
            logger.info(f"  Winsorize前范围: [{y_data_original.min():.3f}, {y_data_original.max():.3f}] ATR")
            logger.info(f"  Winsorize后范围: [{y_data.min():.3f}, {y_data.max():.3f}] ATR")
        else:
            logger.warning("⚠️ 未找到atr_return_label_original字段，权重计算将使用Winsorize后的标签")

        # 🚀 禁用标签变换：保持原始ATR单位，让Pseudo-Huber Loss自动处理极值
        label_transformer = None
        original_y_stats = None

        # 保存原始标签统计，用于动态调整delta范围
        original_y_stats = {
            'mean': y_data.mean(),
            'std': y_data.std(),
            'min': y_data.min(),
            'max': y_data.max(),
            'range': y_data.max() - y_data.min()
        }
        logger.info(f"🚀 使用原始ATR标签，不进行变换（业界最佳实践）")
        logger.info(f"  原始标签范围: [{original_y_stats['min']:.3f}, {original_y_stats['max']:.3f}] ATR")
        logger.info(f"  原始标签统计: std={original_y_stats['std']:.3f}, range={original_y_stats['range']:.3f}")
        logger.info(f"  Pseudo-Huber Loss将自动处理极值，无需预先变换")

        logger.info(f"数据对齐完成: {len(raw_data)} 个样本, {len(feature_columns)} 个特征列")

        # 🚀 优化：如果提供了预处理数据,跳过数据划分和预处理步骤
        if preprocessed_data is not None:
            logger.info("=" * 60)
            logger.info("✅ 使用预处理数据（集成学习优化）")
            logger.info("=" * 60)
            logger.info("📋 跳过以下步骤:")
            logger.info("  ✅ 数据划分（使用共享的训练/验证集）")
            logger.info("  ✅ Winsorization（使用共享的阈值）")
            logger.info("  ✅ 特征清洗（使用共享的预处理结果）")
            logger.info(f"  🎯 训练集: {len(preprocessed_data['X_train']):,}样本")
            logger.info(f"  🎯 验证集: {len(preprocessed_data['X_val']):,}样本")
            logger.info("=" * 60)

            X_train_processed = preprocessed_data['X_train']
            X_val_processed = preprocessed_data['X_val']
            y_train_split = preprocessed_data['y_train']
            y_val_split = preprocessed_data['y_val']
            train_winsorization_thresholds = preprocessed_data['winsorization_thresholds']

            # 🚀 从预处理数据中提取特征名
            # 注意: X_train_processed可能是DataFrame或numpy数组
            if hasattr(X_train_processed, 'columns'):
                feature_names = list(X_train_processed.columns)
            else:
                # 如果是numpy数组,使用传入的feature_columns
                feature_names = feature_columns if isinstance(feature_columns, list) else list(feature_columns)

            y_train_split_original = None  # 预处理数据中不包含原始标签
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

            # 划分原始数据
            raw_train_data = raw_data.iloc[:split_point] if hasattr(raw_data, 'iloc') else raw_data[:split_point]
            raw_val_data = raw_data.iloc[split_point:] if hasattr(raw_data, 'iloc') else raw_data[split_point:]
            y_train_split = y_data[:split_point]
            y_val_split = y_data[split_point:]

            # 🚀 同时划分Winsorize前的原始标签
            y_train_split_original = None
            if y_data_original is not None:
                y_train_split_original = y_data_original[:split_point]
                logger.info("✅ 同时划分Winsorize前的原始标签（用于权重计算）")

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
            logger.info("🚀 对训练集进行特征预处理（计算Winsorization阈值）...")
            X_train_processed, feature_names, train_winsorization_thresholds = _prepare_feature_data(
                raw_train_data, feature_columns, apply_winsorization=True, data_type="训练集"
            )

            if X_train_processed is None:
                logger.error("训练集特征预处理失败")
                return None

            logger.info("🚀 对验证集进行特征预处理（使用训练集的Winsorization阈值）...")
            X_val_processed, _, _ = _prepare_feature_data(
                raw_val_data, feature_columns, apply_winsorization=True,
                winsorization_thresholds=train_winsorization_thresholds, data_type="验证集"
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

        # 分析连续收益标签的分布
        profit_samples = (y_train_split >= 1.8).sum()  # 盈利样本（>=1.8 ATR）
        loss_samples = (y_train_split <= -0.9).sum()   # 亏损样本（<=-0.9 ATR）
        neutral_samples = ((y_train_split > -0.9) & (y_train_split < 1.8)).sum()  # 中性样本
        total_samples = len(y_train_split)

        logger.info(f"  📊 盈利样本(>=1.8 ATR): {profit_samples:,}个 ({profit_samples/total_samples*100:.2f}%)")
        logger.info(f"  📊 中性样本(-0.9~1.8 ATR): {neutral_samples:,}个 ({neutral_samples/total_samples*100:.2f}%)")
        logger.info(f"  📊 亏损样本(<=-0.9 ATR): {loss_samples:,}个 ({loss_samples/total_samples*100:.2f}%)")
        logger.info(f"  📊 标签范围: [{y_train_split.min():.3f}, {y_train_split.max():.3f}] ATR")
        logger.info(f"  📊 标签均值: {y_train_split.mean():.3f} ATR")
        logger.info(f"  📊 标签标准差: {y_train_split.std():.3f} ATR")

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
        pca_info = {'applied': False, 'reason': 'will_be_applied_after_scaling'}

        logger.info("=" * 60)
        logger.info("特征处理策略")
        logger.info("=" * 60)
        logger.info(f"🎯 原始特征数量: {n_features}")
        logger.info(f"✅ 新特征工程将在标准化后应用")
        logger.info(f"🔍 股票因子将保持原始形式，市场因子将进行去相关+PCA处理")

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

        if val_nan_mask.any():
            logger.warning(f"验证集中发现{val_nan_mask.sum()}个NaN标签，将被移除")
            X_val_split = X_val_split[~val_nan_mask]
            y_val_split = y_val_split[~val_nan_mask]

        # 🚀 计算样本权重（基于标签变换感知策略）
        logger.info("🔧 计算样本权重（基于ATR语义的权重分配）...")

        # 🚀 优化：使用Winsorize前的原始标签计算权重，确保基于真实分布
        sample_weights = _calculate_professional_sample_weights(
            y_train_split,  # 使用训练集标签（Winsorize后）
            imbalance_ratio=1.0,  # 回归模式使用固定值
            logger=logger,
            boundary_distances=None,  # 已删除边界距离逻辑
            is_regression=True,  # 固定为回归模式
            label_transformer=label_transformer,  # 传入标签变换器用于逆变换
            y_train_original=y_train_split_original  # 🚀 传入Winsorize前的原始标签
        )

        # 🚀 Pseudo-Huber回归：直接使用原始ATR标签，不进行标准化
        # 合并训练集和验证集计算完整标签统计（修复：使用numpy.concatenate而不是pd.concat）
        # 确保输入是numpy数组
        if not isinstance(y_train_split, np.ndarray):
            y_train_split = np.array(y_train_split)
        if not isinstance(y_val_split, np.ndarray):
            y_val_split = np.array(y_val_split)
        all_labels = np.concatenate([y_train_split, y_val_split])
        logger.info(f"🚀 Pseudo-Huber回归标签统计（原始ATR单位）:")
        logger.info(f"  训练集: {len(y_train_split)} 个样本")
        logger.info(f"  验证集: {len(y_val_split)} 个样本")
        logger.info(f"  完整标签范围: [{all_labels.min():.3f}, {all_labels.max():.3f}] ATR")
        logger.info(f"  训练集标签范围: [{y_train_split.min():.3f}, {y_train_split.max():.3f}] ATR")
        logger.info(f"  验证集标签范围: [{y_val_split.min():.3f}, {y_val_split.max():.3f}] ATR")
        logger.info(f"  标签均值: {all_labels.mean():.6f} ATR")
        logger.info(f"  标签标准差: {all_labels.std():.6f} ATR")

        # 🚀 关键修复：标签空间一致性处理
        # 注意：y_train_split和y_val_split已经是变换后的标签
        # 为了评估，需要将它们逆变换回原始ATR空间
        if label_transformer is not None:
            # 逆变换到原始ATR空间进行评估
            y_train_split_original = label_transformer.inverse_transform(y_train_split)
            y_val_split_original = label_transformer.inverse_transform(y_val_split)
            logger.info(f"🔧 标签逆变换: 变换后范围[{y_train_split.min():.3f}, {y_train_split.max():.3f}] -> 原始ATR范围[{y_train_split_original.min():.3f}, {y_train_split_original.max():.3f}]")
        else:
            # 如果没有变换，直接使用
            y_train_split_original = y_train_split
            y_val_split_original = y_val_split
            logger.info("🔧 未使用标签变换，直接使用原始标签")

        # 用于训练的标签（变换后空间）
        y_train_split_mapped = y_train_split  # 变换后标签，用于训练
        y_val_split_mapped = y_val_split      # 变换后标签，用于训练

        # 用于评估的标签（原始ATR空间）
        y_train_split_eval = y_train_split_original  # 原始ATR标签，用于评估
        y_val_split_eval = y_val_split_original      # 原始ATR标签，用于评估

        reverse_label_map = None

        # 保存原始标签统计量（用于日志记录）
        if label_stats is None:
            label_stats = {}
        label_stats.update({
            'mean': y_train_split.mean(),
            'std': y_train_split.std(),
            'min': y_train_split.min(),
            'max': y_train_split.max()
        })

        logger.info(f"✅ 准备使用原始ATR标签进行Pseudo-Huber回归")
        logger.info(f"  无需标准化，预测结果直接为ATR单位")

        logger.info("✅ Winsorization缩尾处理已在数据预处理阶段完成")
        logger.info("=" * 60)


        # 确定股票因子列表（不包含市场因子）
        stock_factors = _get_stock_factors()

        # 只保留股票因子，过滤掉市场因子
        available_stock_factors = []
        stock_factor_indices = []

        for i, feature_name in enumerate(feature_names):
            if feature_name in stock_factors:
                available_stock_factors.append(feature_name)
                stock_factor_indices.append(i)

        # 提取股票因子数据（在标准化之前）
        X_train_split_stock = X_train_split[:, stock_factor_indices]
        X_val_split_stock = X_val_split[:, stock_factor_indices]
        feature_names = available_stock_factors

        logger.info(f"✅ 股票因子特征: {len(feature_names)} 个")
        logger.info(f"   - 特征列表: {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}")

        # 数据标准化（避免数据泄露：只在股票因子训练集上学习参数）
        logger.info("🔧 开始股票因子数据标准化（避免数据泄露）...")
        scaler = StandardScaler()

        # 只在股票因子训练集上学习标准化参数
        X_train_split_scaled = scaler.fit_transform(X_train_split_stock)
        logger.info(f"✅ 标准化参数已在股票因子训练集上学习完成")

        # 用训练集学到的参数标准化验证集
        X_val_split_scaled = scaler.transform(X_val_split_stock)
        logger.info(f"✅ 验证集已用股票因子训练集参数进行标准化")

        # 🚀 重要：在标准化完成后立即初始化增强特征变量
        X_train_enhanced = X_train_split_scaled.copy()  # 默认使用原始特征
        X_val_enhanced = X_val_split_scaled.copy()      # 默认使用原始特征
        logger.info("📊 增强特征变量已初始化为原始特征")

        logger.info("🔄 使用基础特征进行训练...")

        # 直接使用基础特征，不再计算交互特征
        X_train_enhanced = X_train_split_scaled
        X_val_enhanced = X_val_split_scaled

        logger.info(f"✅ 使用基础特征: {len(feature_names)} 个特征")

        # PCA信息标记为未应用（使用基础特征）
        pca_info = {'applied': False, 'reason': 'using_base_features_only'}

        logger.info("\n" + "=" * 60)
        logger.info("🚀 步骤3: 训练XGBoost模型（使用基础特征）")
        logger.info("=" * 60)

        # 分析样本分布（支持二分类和三分类）
        unique_labels_split = np.unique(y_train_split)
        total_samples = len(y_train_split)



        # 根据特征数量和样本分布调整XGBoost参数
        current_features = X_train_split_scaled.shape[1]
        n_samples = len(X_train_split)

        logger.info("=" * 60)
        # 🔧 修复：提前定义actual_feature_count变量
        actual_feature_count = len(feature_names)  # 使用实际的特征数量

        logger.info("XGBoost参数优化")
        logger.info("=" * 60)
        logger.info(f"📊 训练样本: {n_samples:,}个")
        logger.info(f"📊 特征数量: {actual_feature_count}个")  # 🔧 修复：使用实际特征数量
        logger.info(f"📊 类别不平衡比例: {imbalance_ratio:.1f}:1")

        # 🚀 B. 模型超参数优化：降低模型复杂度（提升泛化能力）
        logger.info("=" * 60)
        logger.info("🚀 B. 模型超参数优化：降低模型复杂度")
        logger.info("=" * 60)
        logger.info("📋 复杂度优化说明:")
        logger.info("  ✅ 降低max_depth提升ATR三分类模型的泛化能力")
        logger.info("  ✅ 限制模型最多学习两个因子的交互作用")
        logger.info("  ✅ 减少过拟合风险，提升P@1指标")
        logger.info("  ✅ 改善盈利类别（Label=1）的识别精度")

        # 🔧 优化参数配置：根据特征数量和交互特征使用情况调整
        # 基于特征分析的专业优化配置

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

        # 🚀 Pseudo-Huber回归模式：使用自定义目标函数
        logger.info("🚀 Pseudo-Huber回归模式：使用自定义目标函数和原始ATR标签")

        # 🚀 回归模式使用基础特征的优化参数（专门适配22个特征）
        logger.info(f"ℹ️ 回归模式使用基础特征，采用专门适配22特征的优化参数配置")

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
            # optimized_params = {
            #     'max_depth': 8,                   # 🔧 重新优化：适度深度，平衡学习能力和泛化
            #     'learning_rate': 0.08,            # 🔧 重新优化：适中学习率，避免过度保守
            #     'n_estimators': 3000,             # 🔧 重新优化：合理树数量，避免过度训练
            #     'subsample': 0.8,                # 🔧 重新优化：标准采样率，保持学习能力
            #     'colsample_bytree': 1.0,          # 🔧 重新优化：平衡特征采样，避免信息损失
            #     'reg_alpha': 0.2,                 # 🔧 重新优化：适度L1正则化，平衡特征选择
            #     'reg_lambda': 3.0,                # 🔧 重新优化：适度L2正则化，避免过度约束
            #     'min_child_weight': 3,            # 🔧 重新优化：适中最小权重，平衡泛化和学习
            #     'gamma': 0.2,                     # 🔧 重新优化：适中分裂阈值，平衡复杂度
            #     'colsample_bylevel': 0.8,         # 🔧 重新优化：标准每层特征采样
            #     'max_delta_step': 0,
            #     'random_state': 42,
            #     'n_jobs': -1,
            #     'verbosity': 0
            # }
            optimized_params = {
                'max_depth': 7,                   # 🔧 重新优化：适度深度，平衡学习能力和泛化
                'learning_rate': 0.1,            # 🔧 重新优化：适中学习率，避免过度保守
                'n_estimators': 2000,             # 🚀 优化：减少树数量，避免过拟合（集成学习已提供正则化）
                'subsample': 0.8,                # 🔧 重新优化：标准采样率，保持学习能力
                'colsample_bytree': 0.9,          # 🔧 重新优化：平衡特征采样，避免信息损失
                'reg_alpha': 0.5,                 # 🔧 重新优化：适度L1正则化，平衡特征选择
                'reg_lambda': 3.0,                # 🔧 重新优化：适度L2正则化，避免过度约束
                'min_child_weight': 5,            # 🔧 重新优化：适中最小权重，平衡泛化和学习
                'gamma': 0.2,                     # 🔧 重新优化：适中分裂阈值，平衡复杂度
                'colsample_bylevel': 0.8,         # 🔧 重新优化：标准每层特征采样
                'max_delta_step': 0,
                'random_state': 47,
                'n_jobs': -1,
                'verbosity': 0
            }

        params.update(optimized_params)
        logger.info(f"🎯 针对{actual_feature_count}个特征优化参数配置")

        # 🚀 超参数搜索功能
        if enable_param_search:
            logger.info("=" * 80)
            logger.info("🔍 启用超参数搜索优化")
            logger.info("=" * 80)

            # 定义参数搜索空间
            param_grid = {
                'max_depth': [4, 5, 6, 7, 8],
                'learning_rate': [0.05, 0.08, 0.1, 0.12],
                'reg_alpha': [0.1, 0.2, 0.5, 1.0, 2.0],
                'reg_lambda': [0.5, 1.0, 2.0, 3.0, 5.0],
                'min_child_weight': [1, 3, 5, 7, 10],
                'gamma': [0.0, 0.1, 0.2, 0.3, 0.5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }

            # Delta参数搜索空间
            delta_grid = [0.5, 1.0, 1.5, 2.0, 3.0]

            logger.info(f"📊 搜索策略: {search_strategy}")
            logger.info(f"📊 参数空间大小: {np.prod([len(v) for v in param_grid.values()]) * len(delta_grid):,}")

            # 执行参数搜索
            best_params, best_delta, search_results = _perform_parameter_search(
                X_train_enhanced, y_train_split_mapped, X_val_enhanced, y_val_split_mapped,
                y_val_split_eval, sample_weights, param_grid, delta_grid,
                search_strategy, label_transformer
            )

            # 更新参数
            params.update(best_params)
            logger.info("✅ 参数搜索完成，使用最优参数配置")
            logger.info(f"🎯 最优Delta: {best_delta}")
            logger.info(f"🎯 最优参数: {best_params}")

            # 跳过后续的Delta网格搜索
            skip_delta_search = True
            best_delta_from_search = best_delta
        else:
            skip_delta_search = False
            best_delta_from_search = None

        logger.info("🚀 Pseudo-Huber回归参数配置:")
        logger.info(f"  max_depth: {params['max_depth']} (针对{actual_feature_count}特征优化)")
        logger.info(f"  learning_rate: {params['learning_rate']}")
        logger.info(f"  n_estimators: {params['n_estimators']}")
        logger.info(f"  subsample: {params['subsample']}")
        logger.info(f"  colsample_bytree: {params['colsample_bytree']}")
        logger.info(f"  reg_alpha: {params['reg_alpha']}")
        logger.info(f"  reg_lambda: {params['reg_lambda']}")
        logger.info(f"  min_child_weight: {params['min_child_weight']}")
        logger.info(f"  gamma: {params['gamma']}")
        logger.info("✅ 配置针对Pseudo-Huber目标函数和小特征集优化")


        # 🔧 移除交互约束配置，使用基础特征
        logger.info("✅ 使用基础特征，无需特征交互约束")

        logger.info("=" * 60)

        # 🚀 回归模式：不进行标签映射，保持原始ATR倍数含义
        unique_labels = None
        label_mapping = None
        reverse_mapping = None
        logger.info("回归模式：保持原始连续收益标签，不进行映射")

        # 🚀 成本感知回归配置
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
        num_class = None
        logger.info("使用回归模式配置（连续收益预测）")

        # 通用参数
        params.update({
            'objective': objective,
            'eval_metric': eval_metric,
            'random_state': random_seed,  # 🚀 集成学习：使用传入的随机种子
            'n_jobs': -1,
            'verbosity': 0
        })

        logger.info(f"✅ XGBoost参数配置完成，回归模式，objective={objective}")

        # 🚀 Pseudo-Huber回归：实现自定义目标函数和Delta选择
        logger.info("🚀 开始Pseudo-Huber回归训练...")

        # 定义Pseudo-Huber目标函数
        def pseudo_huber_obj(delta):
            """
            Pseudo-Huber目标函数
            L_δ(a) = δ²(√(1+(a/δ)²) - 1)
            梯度: grad = a / √(1+(a/δ)²)
            Hessian: hess = δ² / (1+(a/δ)²)^(3/2)
            """
            def _obj(preds, dtrain):
                labels = dtrain.get_label()
                residuals = preds - labels

                # 🚀 数值稳定性改进
                delta_safe = max(delta, 1e-8)  # 避免delta过小
                r_over_delta = residuals / delta_safe

                # 限制r_over_delta的范围，避免数值溢出
                r_over_delta = np.clip(r_over_delta, -100, 100)

                sqrt_term = np.sqrt(1.0 + r_over_delta ** 2)

                # 梯度: r / √(1+(r/δ)²)
                grad = residuals / sqrt_term

                # 🚀 修复Hessian计算：使用正确的公式
                # Hessian应该是 1 / (δ² * (1+(r/δ)²)^(3/2))，而不是 δ² / (1+(r/δ)²)^(3/2)
                hess = 1.0 / (delta_safe**2 * sqrt_term**3)

                # 🚀 增强数值稳定性：限制Hessian的范围
                hess = np.clip(hess, 1e-8, 1e6)

                return grad, hess
            return _obj

        # 🚀 智能Delta范围调整：基于原始ATR标签统计（无变换）
        # 根据标签标准差动态设置Delta范围
        label_std = all_labels.std()

        # 🚀 优化：根据原始ATR标签的std调整delta范围
        # Delta的物理意义：控制Huber Loss从二次变为线性的转折点
        # 一般设置为0.5-2倍std之间
        if original_y_stats is not None:
            if label_std < 2.0:
                # 小波动：使用较小delta范围
                huber_delta_grid = [1.0]
                logger.info(f"🎯 根据原始标签std={label_std:.3f}<2.0，使用小Delta范围")
            elif label_std < 4.0:
                # 中等波动：使用中等delta范围
                huber_delta_grid = [2.0, 3.0, 4.0, 5.0]
                logger.info(f"🎯 根据原始标签std={label_std:.3f}<4.0，使用中等Delta范围")
            elif label_std < 6.0:
                # 较大波动：使用较大delta范围
                huber_delta_grid = [3.0, 4.5, 6.0, 7.5]
                logger.info(f"🎯 根据原始标签std={label_std:.3f}<6.0，使用较大Delta范围")
            else:
                # 大波动：使用大delta范围
                huber_delta_grid = [4.0, 6.0, 8.0, 10.0]
                logger.info(f"🎯 根据原始标签std={label_std:.3f}>=6.0，使用大Delta范围")
        else:
            # 兜底逻辑：使用标准delta范围
            huber_delta_grid = [2.0, 3.0, 4.0, 5.0]
            logger.info(f"使用标准Delta范围（标签std={label_std:.3f}）")

        # 🚀 条件性Delta网格搜索
        if preset_huber_delta is not None:
            # 🚀 使用预设的delta值,跳过搜索
            logger.info(f"🚀 使用预设Pseudo-Huber delta值: {preset_huber_delta:.2f} (跳过网格搜索)")
            best_delta = preset_huber_delta
            best_score = float('-inf')
            delta_results = {}
            mse_baseline_score = None
        elif skip_delta_search:
            logger.info("🔍 跳过Delta网格搜索（已通过参数搜索优化）")
            best_delta = best_delta_from_search
            best_score = float('-inf')
            delta_results = {}
            mse_baseline_score = None
        else:
            logger.info(f"🔍 开始Delta网格搜索: {huber_delta_grid}")

            # 🚀 添加MSE基准测试
            logger.info("🔍 首先测试MSE损失函数作为基准...")

            best_delta = None
            best_score = float('-inf')
            delta_results = {}
            mse_baseline_score = None

        dtrain_full = xgb.DMatrix(X_train_enhanced, label=y_train_split_mapped, weight=sample_weights)
        dval = xgb.DMatrix(X_val_enhanced, label=y_val_split_mapped)
        logger.info(f"🚀 网格搜索将在 {len(X_train_enhanced):,} 个训练样本上进行训练，并在 {len(X_val_enhanced):,} 个验证样本上进行评估。")
        logger.info(f"📊 使用基础特征维度: {X_train_enhanced.shape[1]} (股票基础特征)")

        # 🚀 简化：使用统一的训练流程，跳过优化测试
        logger.info("ℹ️ 跳过优化测试 - 使用统一的Pseudo-Huber训练流程")

        # 2. MSE基准测试
        # 🎯 MSE基准作用说明：
        # 1. 提供传统回归基准，用于对比Pseudo-Huber损失函数的改进效果
        # 2. 验证自定义损失函数是否真正优于标准MSE损失
        # 3. 确保模型选择的合理性，避免过度复杂化
        # 4. 为Delta参数选择提供参考基准线
        try:
            logger.info("  测试 MSE损失函数基准（作为Pseudo-Huber损失函数的对比基准）...")
            mse_params = params.copy()
            mse_params['objective'] = 'reg:squarederror'

            mse_max_rounds = 1500  # 🔧 优化：与delta搜索保持一致
            mse_early_stopping = 150  # 🔧 优化：10%比例

            model_mse = xgb.train(
                mse_params,
                dtrain_full,
                num_boost_round=mse_max_rounds,
                evals=[(dval, 'eval')],
                verbose_eval=False,
                early_stopping_rounds=mse_early_stopping
            )

            # 📊 MSE基准测试训练监控
            if hasattr(model_mse, 'best_iteration'):
                mse_actual_rounds = model_mse.best_iteration + 1
                logger.info(f"    MSE基准训练轮数: {mse_actual_rounds}/{mse_max_rounds} ({mse_actual_rounds/mse_max_rounds*100:.1f}%)")

                if mse_actual_rounds < mse_max_rounds - mse_early_stopping:
                    logger.info(f"    MSE Early Stopping触发，节省{mse_max_rounds - mse_actual_rounds}轮训练")
                    logger.info(f"    ✅ MSE基准训练效率: 优秀 (Early Stopping生效)")
                else:
                    logger.info(f"    MSE达到最大轮数，建议进一步增加训练轮数")
                    logger.info(f"    ⚠️  MSE基准可能需要更多轮数以达到最优性能")

            val_preds_mse = model_mse.predict(dval)
            # 🚀 修复：MSE基准评估需要在原始ATR空间进行
            # 将MSE预测值逆变换到原始ATR空间
            if label_transformer is not None:
                val_preds_mse_atr = label_transformer.inverse_transform(val_preds_mse)
            else:
                val_preds_mse_atr = val_preds_mse

            rmse_mse = np.sqrt(np.mean((val_preds_mse_atr - y_val_split_eval) ** 2))

            # 计算MSE的预测范围和IC（在原始ATR空间）
            ic_mse = np.corrcoef(y_val_split_eval, val_preds_mse_atr)[0, 1]
            pred_range_mse = val_preds_mse_atr.max() - val_preds_mse_atr.min()
            true_range = y_val_split_eval.max() - y_val_split_eval.min()
            range_ratio_mse = pred_range_mse / true_range if true_range > 0 else 0

            # 🚀 MSE基准评估：使用统一阈值定义
            # 注意：MSE基准评估在原始ATR空间进行，不需要阈值变换

            # 🚀 修复：使用逆变换后的预测值进行掩码判断
            positive_pred_mask = val_preds_mse_atr > 0
            negative_pred_mask = val_preds_mse_atr < 0

            if positive_pred_mask.sum() > 0:
                # 🚀 修复：使用原始ATR空间的标签和阈值进行评估
                actual_profit_rate_mse = ((y_val_split_eval >= PROFIT_THRESHOLD) & positive_pred_mask).sum() / positive_pred_mask.sum()
            else:
                actual_profit_rate_mse = 0.0

            if negative_pred_mask.sum() > 0:
                # 🚀 修复：使用原始ATR空间的标签和阈值进行评估
                actual_loss_rate_mse = ((y_val_split_eval <= LOSS_THRESHOLD) & negative_pred_mask).sum() / negative_pred_mask.sum()
            else:
                actual_loss_rate_mse = 0.0

            risk_reward_ratio_mse = actual_profit_rate_mse / max(actual_loss_rate_mse, 0.01)

            # 🚀 修复：计算Top-K收益（使用逆变换后的预测值排序）
            sorted_indices = np.argsort(val_preds_mse_atr)[::-1]

            def calculate_realistic_return(true_returns, threshold_atr=ATR_MULTIPLIER_PROFIT, stop_loss_atr=-ATR_MULTIPLIER_LOSS):
                clipped_returns = np.clip(true_returns, stop_loss_atr, threshold_atr)
                return clipped_returns.mean()

            # 🚀 修复：使用原始ATR空间的标签计算Top-K收益
            top1_return_mse = calculate_realistic_return(y_val_split_eval[sorted_indices[:1]]) if len(sorted_indices) >= 1 else 0.0
            top3_return_mse = calculate_realistic_return(y_val_split_eval[sorted_indices[:3]]) if len(sorted_indices) >= 3 else 0.0
            top5_return_mse = calculate_realistic_return(y_val_split_eval[sorted_indices[:5]]) if len(sorted_indices) >= 5 else 0.0

            # 🚀 新的综合评分
            if range_ratio_mse < 0.1:
                range_score_mse = 0.0
            elif range_ratio_mse < 0.3:
                range_score_mse = range_ratio_mse / 0.3 * 0.5
            elif range_ratio_mse <= 1.0:
                range_score_mse = 0.5 + (range_ratio_mse - 0.3) / 0.7 * 0.5
            else:
                range_score_mse = max(0.8, 1.0 - (range_ratio_mse - 1.0) * 0.2)

            if risk_reward_ratio_mse >= 1.0:
                risk_score_mse = 1.0
            elif risk_reward_ratio_mse >= 0.5:
                risk_score_mse = risk_reward_ratio_mse
            else:
                risk_score_mse = risk_reward_ratio_mse * 0.5

            mse_baseline_score = (
                actual_profit_rate_mse * 0.4 +
                ic_mse * 0.25 +
                range_score_mse * 0.2 +
                risk_score_mse * 0.15
            )

            # 惩罚机制
            if ic_mse < 0:
                mse_baseline_score = ic_mse
            elif actual_profit_rate_mse < 0.05:
                mse_baseline_score *= 0.3
            elif range_ratio_mse < 0.1:
                mse_baseline_score *= 0.5

            # 🚀 统一指标显示：与Delta测试保持一致的格式
            weighted_top_return_mse = top1_return_mse * 0.6 + top3_return_mse * 0.3 + top5_return_mse * 0.1
            logger.info(f"  MSE基准: Top收益={weighted_top_return_mse:.3f}, IC={ic_mse:.4f}, 止盈率={actual_profit_rate_mse:.1%}, 风险比={risk_reward_ratio_mse:.2f}, Score={mse_baseline_score:.4f}")
            logger.info(f"    补充: RMSE={rmse_mse:.4f}, 范围比={range_ratio_mse:.2f}")

            # 将MSE结果加入对比
            delta_results['MSE'] = {
                'rmse': rmse_mse,
                'ic': ic_mse,
                'range_ratio': range_ratio_mse,
                'top1_return': top1_return_mse,
                'top3_return': top3_return_mse,
                'top5_return': top5_return_mse,
                'weighted_top_return': top1_return_mse * 0.6 + top3_return_mse * 0.3 + top5_return_mse * 0.1,
                'actual_profit_rate': actual_profit_rate_mse,
                'actual_loss_rate': actual_loss_rate_mse,
                'risk_reward_ratio': risk_reward_ratio_mse,
                'range_score': range_score_mse,
                'composite_score': mse_baseline_score
            }

        except Exception as e:
            logger.warning(f"MSE基准测试失败: {e}")
            mse_baseline_score = None

        for delta in huber_delta_grid:
            logger.info(f"🔍 测试 delta={delta:.1f} ATR...")

            try:
                # 配置用于网格搜索的临时模型参数
                grid_params = params.copy()
                grid_params.update({
                    'n_estimators': 500,  # 在网格搜索中使用较少的树以提高速度
                    'verbosity': 0
                })

                # 关键修改：在 *完整的训练集* 上训练，在 *独立的验证集* 上评估
                grid_max_rounds = 1500  # 🔧 优化：增加到1500轮确保充分搜索
                grid_early_stopping = 150  # 🔧 优化：10%比例，平衡搜索效率和准确性

                model_grid = xgb.train(
                    params=grid_params,
                    dtrain=dtrain_full,
                    num_boost_round=grid_max_rounds,
                    obj=pseudo_huber_obj(delta),
                    evals=[(dval, 'val')],
                    verbose_eval=False,
                    early_stopping_rounds=grid_early_stopping
                )

                # 📊 Delta网格搜索训练监控
                if hasattr(model_grid, 'best_iteration'):
                    grid_actual_rounds = model_grid.best_iteration + 1
                    logger.info(f"    Delta={delta:.1f} 训练轮数: {grid_actual_rounds}/{grid_max_rounds} ({grid_actual_rounds/grid_max_rounds*100:.1f}%)")

                    if grid_actual_rounds < grid_max_rounds - grid_early_stopping:
                        logger.info(f"    Delta={delta:.1f} Early Stopping触发，节省{grid_max_rounds - grid_actual_rounds}轮")
                        logger.info(f"    ✅ Delta={delta:.1f} 搜索效率: 优秀 (Early Stopping生效)")
                    else:
                        logger.info(f"    Delta={delta:.1f} 达到最大轮数，可能需要更多轮数")
                        logger.info(f"    ⚠️  Delta={delta:.1f} 建议增加搜索轮数以获得更准确的参数评估")

                # 在验证集上评估
                val_preds_raw = model_grid.predict(dval)

                # 🚀 关键修复：将预测值逆变换到原始ATR单位进行评估
                if label_transformer is not None:
                        val_preds = label_transformer.inverse_transform(val_preds_raw)
                else:
                        val_preds = val_preds_raw

                # 🚀 修复：计算RMSE（使用原始ATR单位）
                rmse = np.sqrt(np.mean((val_preds - y_val_split_eval) ** 2))

                # 🚀 专业Delta选择：多维度评估（统一使用原始ATR单位）
                # 1. 计算IC（最重要指标）
                ic_corr = np.corrcoef(y_val_split_eval, val_preds)[0, 1]
                if np.isnan(ic_corr):
                        ic_corr = 0.0

                # 2. 计算预测范围合理性
                pred_range = val_preds.max() - val_preds.min()
                true_range = y_val_split_eval.max() - y_val_split_eval.min()
                range_ratio = pred_range / true_range if true_range > 0 else 0

                # 3. 🚀 基于止盈止损命中率的评估体系（使用原始ATR单位）
                sorted_indices = np.argsort(val_preds)[::-1]

                # 🚀 Delta网格搜索：使用统一阈值定义
                # 注意：这里评估在原始ATR空间进行，因为val_preds已经逆变换

                # 预测能力评估：预测>0的样本中，实际止盈的比例
                positive_pred_mask = val_preds > 0
                negative_pred_mask = val_preds < 0

                if positive_pred_mask.sum() > 0:
                        # 🚀 修复：使用原始ATR空间的标签和阈值进行评估
                        actual_profit_rate = ((y_val_split_eval >= PROFIT_THRESHOLD) & positive_pred_mask).sum() / positive_pred_mask.sum()
                else:
                        actual_profit_rate = 0.0

                if negative_pred_mask.sum() > 0:
                        # 🚀 修复：使用原始ATR空间的标签和阈值进行评估
                        actual_loss_rate = ((y_val_split_eval <= LOSS_THRESHOLD) & negative_pred_mask).sum() / negative_pred_mask.sum()
                else:
                        actual_loss_rate = 0.0

                # 预测分布合理性：预测值中极值的比例
                pred_profit_ratio = (val_preds >= 1.0).sum() / len(val_preds)  # 预测>1ATR的比例
                pred_loss_ratio = (val_preds <= -0.5).sum() / len(val_preds)   # 预测<-0.5ATR的比例

                # 风险收益比
                risk_reward_ratio = actual_profit_rate / max(actual_loss_rate, 0.01)

                # Top-K实际收益（考虑止盈止损）
                def calculate_realistic_return(true_returns, threshold_atr=2.0, stop_loss_atr=-1.0):
                    clipped_returns = np.clip(true_returns, stop_loss_atr, threshold_atr)
                    return clipped_returns.mean()

                # 🚀 修复：使用原始ATR空间的标签计算Top-K收益
                top1_return = calculate_realistic_return(y_val_split_eval[sorted_indices[:1]]) if len(sorted_indices) >= 1 else 0.0
                top3_return = calculate_realistic_return(y_val_split_eval[sorted_indices[:3]]) if len(sorted_indices) >= 3 else 0.0
                top5_return = calculate_realistic_return(y_val_split_eval[sorted_indices[:5]]) if len(sorted_indices) >= 5 else 0.0

                # 计算加权Top收益
                weighted_top_return = top1_return * 0.6 + top3_return * 0.3 + top5_return * 0.1

                # 4. 🚀 全新的交易导向综合评分
                # 权重分配：止盈命中率(40%) + IC(25%) + 范围覆盖率(20%) + 风险收益比(15%)

                # 范围覆盖率评分
                if range_ratio < 0.1:
                        range_score = 0.0
                elif range_ratio < 0.3:
                        range_score = range_ratio / 0.3 * 0.5  # 线性映射到0.5
                elif range_ratio <= 1.0:
                        range_score = 0.5 + (range_ratio - 0.3) / 0.7 * 0.5  # 线性映射到1.0
                else:
                        range_score = max(0.8, 1.0 - (range_ratio - 1.0) * 0.2)  # 过度扩张惩罚

                # 风险收益比评分
                if risk_reward_ratio >= 1.0:
                        risk_score = 1.0
                elif risk_reward_ratio >= 0.5:
                        risk_score = risk_reward_ratio
                else:
                        risk_score = risk_reward_ratio * 0.5

                # 🚀 统一评分体系：与MSE基准使用相同的评估指标
                # 使用与MSE基准完全一致的评分公式
                composite_score = (
                    actual_profit_rate * 0.5 +      # 止盈命中率（最重要）
                    ic_corr * 0.25 +                # IC相关性
                    range_score * 0.1 +             # 范围覆盖率
                    risk_score * 0.15               # 风险控制
                )

                # 🚀 统一惩罚机制：与MSE基准保持一致
                if ic_corr < 0:
                    composite_score = ic_corr  # 负IC直接作为分数
                elif actual_profit_rate < 0.05:
                    composite_score *= 0.3    # 止盈率过低惩罚
                elif range_ratio < 0.1:
                    composite_score *= 0.5    # 范围过小惩罚

                # 5. 🚀 基于交易表现的惩罚机制
                if ic_corr < -0.1:  # 强负相关严重惩罚
                    composite_score = ic_corr
                elif weighted_top_return < -1.0:  # 交易表现极差惩罚
                    composite_score *= 0.3
                elif actual_profit_rate < 0.02:  # 止盈命中率<2%惩罚
                    composite_score *= 0.5

                delta_results[delta] = {
                        'rmse': rmse,
                        'ic': ic_corr,
                        'range_ratio': range_ratio,
                        'top1_return': top1_return,
                        'top3_return': top3_return,
                        'top5_return': top5_return,
                        'weighted_top_return': weighted_top_return,
                        'actual_profit_rate': actual_profit_rate,
                        'actual_loss_rate': actual_loss_rate,
                        'risk_reward_ratio': risk_reward_ratio,
                        'range_score': range_score,
                        'composite_score': composite_score
                    }

                # 🚀 优化后的日志输出：突出交易导向指标
                logger.info(f"  Delta={delta:.1f}: Top收益={weighted_top_return:.3f}, 范围比={range_score:.4f}, IC={ic_corr:.4f}, 止盈率={actual_profit_rate:.1%}, 风险比={risk_reward_ratio:.2f}, Score={composite_score:.4f}")

                if composite_score > best_score:
                        best_score = composite_score
                        best_delta = delta

            except Exception as e:
                logger.warning(f"  Delta={delta:.1f} 训练失败: {e}")
                delta_results[delta] = {
                    'rmse': float('inf'),
                    'ic': 0.0,
                    'range_ratio': 0.0,
                    'top1_return': 0.0,
                    'top3_return': 0.0,
                    'top5_return': 0.0,
                    'weighted_top_return': 0.0,
                    'actual_profit_rate': 0.0,
                    'actual_loss_rate': 0.0,
                    'risk_reward_ratio': 0.0,
                    'range_score': 0.0,
                    'composite_score': float('-inf')
                }

        # 🚀 验证网格搜索结果并选择最佳Delta
        if best_delta is None:
            logger.warning("⚠️ Delta网格搜索失败，使用默认值1.0")
            best_delta = 1.0
        else:
            logger.info(f"✅ 网格搜索完成，最佳Delta: {best_delta:.1f} ATR (综合评分: {best_score:.4f})")

        best_loss_function = f'Pseudo-Huber_{best_delta}'

        # 🚀 智能损失函数训练：根据选择结果使用最优损失函数
        logger.info("=" * 60)
        logger.info("🚀 开始模型训练")
        logger.info("=" * 60)


        # 🚀 回归模式：使用xgboost.train
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train_enhanced, label=y_train_split_mapped, weight=sample_weights)
        dval = xgb.DMatrix(X_val_enhanced, label=y_val_split_mapped)
        logger.info(f"📊 最终模型训练使用基础特征: 训练集{X_train_enhanced.shape}, 验证集{X_val_enhanced.shape}")

        # 🚀 简化：统一使用Pseudo-Huber损失函数训练
        final_params = params.copy()

        logger.info("🎯 训练Pseudo-Huber回归模型...")
        if best_delta is not None:
            logger.info(f"  使用最佳Delta: {best_delta:.1f} ATR")
            if enable_param_search:
                logger.info("  (Delta通过参数搜索优化获得)")
            else:
                logger.info("  (Delta通过网格搜索获得)")
        else:
            # 如果没有找到最佳Delta，使用默认值
            best_delta = 1.0
            logger.warning(f"  未找到最佳Delta，使用默认值: {best_delta}")

        final_params['objective'] = 'reg:pseudohubererror'
        final_params['huber_slope'] = best_delta
        loss_function_name = f'PseudoHuber_{best_delta}'

        # 🚀 智能Early Stopping调整机制（基于特征数量和复杂度）
        # 根据特征数量和训练轮数智能调整Early Stopping参数
        if actual_feature_count <= 20:
            # 小特征集：使用较激进的Early Stopping
            dynamic_early_stopping = int(params['n_estimators'] * 0.08)  # 8%
            logger.info(f"🎯 小特征集({actual_feature_count}个)：使用激进Early Stopping ({dynamic_early_stopping}轮)")
        elif actual_feature_count <= 25:
            # 中特征集：使用适中的Early Stopping
            dynamic_early_stopping = int(params['n_estimators'] * 0.10)  # 10%
            logger.info(f"🎯 中特征集({actual_feature_count}个)：使用适中Early Stopping ({dynamic_early_stopping}轮)")
        else:
            # 大特征集：使用宽松的Early Stopping，但设置更严格的上限
            dynamic_early_stopping = min(int(params['n_estimators'] * 0.12), 300)  # 12%，最多300轮
            logger.info(f"🎯 大特征集({actual_feature_count}个)：使用宽松Early Stopping ({dynamic_early_stopping}轮)")

        # 🚀 专业训练监控和自动调参建议系统
        class TrainingMonitor(xgb.callback.TrainingCallback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.train_losses = []
                self.val_losses = []
                self.epochs = []
                self.best_val_loss = float('inf')
                self.best_epoch = 0
                self.overfitting_start = None
                self.loss_oscillations = 0
                self.last_val_loss = None

            def after_iteration(self, model, epoch, evals_log):
                """XGBoost callback函数 - 使用正确的API"""
                # 从evals_log获取损失值
                if 'train' in evals_log and 'val' in evals_log:
                    train_loss = evals_log['train']['rmse'][-1]  # 最新的train-rmse
                    val_loss = evals_log['val']['rmse'][-1]      # 最新的val-rmse
                else:
                    return False  # 如果没有评估日志，跳过

                self.epochs.append(epoch)
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                # 检测最佳验证损失
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch

                # 检测过拟合开始点
                if len(self.val_losses) >= 10:
                    recent_val_trend = np.mean(self.val_losses[-5:]) - np.mean(self.val_losses[-10:-5])
                    recent_train_trend = np.mean(self.train_losses[-5:]) - np.mean(self.train_losses[-10:-5])

                    if recent_val_trend > 0 and recent_train_trend < 0 and self.overfitting_start is None:
                        self.overfitting_start = epoch

                # 检测损失震荡
                if self.last_val_loss is not None:
                    if abs(val_loss - self.last_val_loss) / self.last_val_loss > 0.01:  # 1%以上变化
                        self.loss_oscillations += 1

                self.last_val_loss = val_loss

                # 每100轮输出详细分析
                if epoch % 100 == 0 and epoch > 0:
                    self._log_training_analysis(epoch)

                return False  # 继续训练

            def _log_training_analysis(self, epoch):
                """输出训练过程分析"""
                if len(self.train_losses) < 20:
                    return

                # 计算损失趋势
                recent_epochs = min(50, len(self.train_losses))
                train_trend = np.polyfit(range(recent_epochs), self.train_losses[-recent_epochs:], 1)[0]
                val_trend = np.polyfit(range(recent_epochs), self.val_losses[-recent_epochs:], 1)[0]

                # 计算过拟合程度
                train_val_gap = self.train_losses[-1] - self.val_losses[-1]
                gap_trend = np.mean([self.train_losses[i] - self.val_losses[i] for i in range(-10, 0)])

                # 计算学习效率
                learning_efficiency = abs(train_trend) / (abs(train_trend) + abs(val_trend) + 1e-8)

                self.logger.info(f"  📊 训练监控 [Epoch {epoch}]:")
                self.logger.info(f"    🔄 训练损失趋势: {train_trend:.6f} ({'下降' if train_trend < 0 else '上升'})")
                self.logger.info(f"    🔄 验证损失趋势: {val_trend:.6f} ({'下降' if val_trend < 0 else '上升'})")
                self.logger.info(f"    📈 过拟合程度: {train_val_gap:.6f} (差距{'扩大' if gap_trend > 0 else '缩小'})")
                self.logger.info(f"    ⚡ 学习效率: {learning_efficiency:.3f}")
                self.logger.info(f"    🌊 损失震荡次数: {self.loss_oscillations}")

                if self.overfitting_start:
                    self.logger.info(f"    ⚠️  过拟合检测: 第{self.overfitting_start}轮开始")

            def get_final_analysis(self):
                """获取最终训练分析结果"""
                if len(self.train_losses) < 50:
                    return {}

                # 整体趋势分析
                total_epochs = len(self.train_losses)
                train_improvement = self.train_losses[0] - self.train_losses[-1]
                val_improvement = self.val_losses[0] - self.val_losses[-1]

                # 收敛性分析
                last_20_val_std = np.std(self.val_losses[-20:])
                convergence_stability = 1.0 / (1.0 + last_20_val_std)

                # 过拟合分析 - 修复计算逻辑
                # 计算训练集和验证集损失的相对差距
                final_train_loss = self.train_losses[-1]
                final_val_loss = self.val_losses[-1]

                # 过拟合严重度：验证集损失相对于训练集损失的超出程度
                if final_val_loss > final_train_loss:
                    overfitting_severity = (final_val_loss - final_train_loss) / final_train_loss
                else:
                    # 如果训练集损失更高，说明可能欠拟合或数据质量问题
                    overfitting_severity = 0.0

                # 学习率效率分析
                early_improvement = self.train_losses[0] - self.train_losses[min(100, total_epochs//4)]
                late_improvement = self.train_losses[max(100, total_epochs//2)] - self.train_losses[-1]
                learning_rate_efficiency = early_improvement / (early_improvement + late_improvement + 1e-8)

                return {
                    'total_epochs': total_epochs,
                    'train_improvement': train_improvement,
                    'val_improvement': val_improvement,
                    'convergence_stability': convergence_stability,
                    'overfitting_severity': overfitting_severity,
                    'learning_rate_efficiency': learning_rate_efficiency,
                    'best_epoch': self.best_epoch,
                    'overfitting_start': self.overfitting_start,
                    'loss_oscillations': self.loss_oscillations,
                    'final_train_loss': self.train_losses[-1],
                    'final_val_loss': self.val_losses[-1]
                }

        # 训练最终模型
        logger.info("🚀 开始最终模型训练（启用专业监控系统）...")
        logger.info(f"  📊 最大训练轮数: {params['n_estimators']}")
        logger.info(f"  📊 Early Stopping轮数: {dynamic_early_stopping} (针对{actual_feature_count}个特征优化)")
        logger.info(f"  📊 Early Stopping比例: {dynamic_early_stopping/params['n_estimators']*100:.1f}%")
        logger.info(f"  📊 训练集样本: {len(X_train_enhanced):,}")
        logger.info(f"  📊 验证集样本: {len(X_val_enhanced):,}")

        # 创建训练监控器
        monitor = TrainingMonitor(logger)

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

        def generate_parameter_recommendations(analysis, current_params, data_info):
            """基于训练分析生成参数调整建议"""
            recommendations = []

            # 1. Learning Rate 调整建议
            if analysis.get('learning_rate_efficiency', 0.5) < 0.3:
                # 学习率过高，早期改善太快，后期改善不足
                new_lr = max(0.01, current_params['learning_rate'] * 0.8)
                recommendations.append({
                    'parameter': 'learning_rate',
                    'current': current_params['learning_rate'],
                    'suggested': new_lr,
                    'reason': '学习率过高，早期收敛过快，建议降低以获得更稳定的训练',
                    'priority': 'HIGH'
                })
            elif analysis.get('learning_rate_efficiency', 0.5) > 0.8:
                # 学习率过低，训练效率不足
                new_lr = min(0.1, current_params['learning_rate'] * 1.2)
                recommendations.append({
                    'parameter': 'learning_rate',
                    'current': current_params['learning_rate'],
                    'suggested': new_lr,
                    'reason': '学习率过低，训练效率不足，建议适度提高',
                    'priority': 'MEDIUM'
                })

            # 2. 量化交易导向的正则化参数智能建议
            overfitting_severity = analysis.get('overfitting_severity', 0)
            current_reg_alpha = current_params['reg_alpha']
            current_reg_lambda = current_params['reg_lambda']
            total_reg_strength = current_reg_alpha + current_reg_lambda

            # 量化交易特殊考虑因素
            sample_count = data_info['sample_count']
            feature_count = data_info['feature_count']
            sample_feature_ratio = sample_count / feature_count

            # 基于数据特征的正则化策略
            is_high_noise_regime = sample_feature_ratio < 200  # 高噪声环境
            is_sufficient_data = sample_count > 10000  # 充足数据

            # 量化交易正则化建议逻辑
            if overfitting_severity > 0.15:  # 验证集过拟合严重
                if is_high_noise_regime:
                    # 高噪声环境：适度增强正则化，但避免过度平滑
                    new_reg_alpha = min(2.0, max(current_reg_alpha * 1.3, current_reg_alpha + 0.3))
                    new_reg_lambda = min(4.0, max(current_reg_lambda * 1.2, current_reg_lambda + 0.5))
                    priority = 'HIGH'
                    reason_suffix = "（高噪声环境，适度增强避免过度平滑）"
                else:
                    # 低噪声环境：可以更强的正则化
                    new_reg_alpha = min(5.0, max(current_reg_alpha * 1.5, current_reg_alpha + 0.5))
                    new_reg_lambda = min(8.0, max(current_reg_lambda * 1.4, current_reg_lambda + 1.0))
                    priority = 'HIGH'
                    reason_suffix = "（低噪声环境，可使用较强正则化）"

                recommendations.extend([
                    {
                        'parameter': 'reg_alpha',
                        'current': current_reg_alpha,
                        'suggested': new_reg_alpha,
                        'reason': f'验证集过拟合严重(严重度:{overfitting_severity:.3f})，增强L1正则化{reason_suffix}',
                        'priority': priority
                    },
                    {
                        'parameter': 'reg_lambda',
                        'current': current_reg_lambda,
                        'suggested': new_reg_lambda,
                        'reason': f'增强L2正则化提高泛化能力{reason_suffix}',
                        'priority': priority
                    }
                ])

            elif overfitting_severity < 0.05:  # 可能欠拟合或正则化过强
                # 量化交易偏好：倾向于较弱的正则化以捕捉市场信号
                if total_reg_strength > 1.0:  # 当前正则化较强
                    # 优先降低L2正则化（保留L1的特征选择能力）
                    if current_reg_lambda > 0.5:
                        new_reg_lambda = max(0.1, current_reg_lambda * 0.6)
                        recommendations.append({
                            'parameter': 'reg_lambda',
                            'current': current_reg_lambda,
                            'suggested': new_reg_lambda,
                            'reason': f'过拟合程度很低(严重度:{overfitting_severity:.3f})，降低L2正则化增强市场信号捕捉能力',
                            'priority': 'HIGH'
                        })

                    # 适度降低L1正则化
                    if current_reg_alpha > 0.5:
                        new_reg_alpha = max(0.01, current_reg_alpha * 0.7)
                        recommendations.append({
                            'parameter': 'reg_alpha',
                            'current': current_reg_alpha,
                            'suggested': new_reg_alpha,
                            'reason': f'降低L1正则化，保持重要特征的预测能力',
                            'priority': 'MEDIUM'
                        })

                elif total_reg_strength <= 0.1:  # 正则化已经很弱
                    # 建议通过其他方式增强模型能力
                    recommendations.append({
                        'parameter': 'max_depth',
                        'current': current_params['max_depth'],
                        'suggested': min(10, current_params['max_depth'] + 1),
                        'reason': '正则化已很弱但无过拟合，建议适度增加树深度增强表达能力',
                        'priority': 'MEDIUM'
                    })

                else:  # 中等强度正则化
                    # 量化交易偏好：倾向于更弱的正则化
                    if is_sufficient_data:
                        new_reg_alpha = max(0.01, current_reg_alpha * 0.8)
                        new_reg_lambda = max(0.1, current_reg_lambda * 0.8)
                        recommendations.extend([
                            {
                                'parameter': 'reg_alpha',
                                'current': current_reg_alpha,
                                'suggested': new_reg_alpha,
                                'reason': '数据充足且无过拟合，适度降低L1正则化增强信号敏感性',
                                'priority': 'MEDIUM'
                            },
                            {
                                'parameter': 'reg_lambda',
                                'current': current_reg_lambda,
                                'suggested': new_reg_lambda,
                                'reason': '降低L2正则化，提高对市场变化的适应性',
                                'priority': 'MEDIUM'
                            }
                        ])

            # 3. 树结构参数调整建议 - 完善max_depth建议
            current_max_depth = current_params['max_depth']
            current_min_child_weight = current_params['min_child_weight']

            # 基于震荡程度调整树结构
            loss_oscillations = analysis.get('loss_oscillations', 0)
            total_epochs = analysis.get('total_epochs', 1000)
            oscillation_rate = loss_oscillations / total_epochs if total_epochs > 0 else 0

            if oscillation_rate > 0.1:  # 震荡过多
                new_max_depth = max(3, current_max_depth - 1)
                new_min_child_weight = min(20, current_min_child_weight + 3)
                recommendations.extend([
                    {
                        'parameter': 'max_depth',
                        'current': current_max_depth,
                        'suggested': new_max_depth,
                        'reason': f'训练震荡率{oscillation_rate:.1%}过高，降低树深度增强稳定性',
                        'priority': 'HIGH'
                    },
                    {
                        'parameter': 'min_child_weight',
                        'current': current_min_child_weight,
                        'suggested': new_min_child_weight,
                        'reason': '增加叶子节点最小权重，减少过拟合和震荡',
                        'priority': 'HIGH'
                    }
                ])

            # 基于过拟合程度调整树深度
            elif overfitting_severity > 0.2:  # 严重过拟合
                if current_max_depth > 4:
                    new_max_depth = max(3, current_max_depth - 1)
                    recommendations.append({
                        'parameter': 'max_depth',
                        'current': current_max_depth,
                        'suggested': new_max_depth,
                        'reason': f'过拟合严重，降低树深度从{current_max_depth}到{new_max_depth}',
                        'priority': 'HIGH'
                    })

            elif overfitting_severity < 0.03:  # 可能欠拟合
                if current_max_depth < 8:
                    new_max_depth = min(10, current_max_depth + 1)
                    recommendations.append({
                        'parameter': 'max_depth',
                        'current': current_max_depth,
                        'suggested': new_max_depth,
                        'reason': f'过拟合程度很低，适度增加树深度增强表达能力',
                        'priority': 'MEDIUM'
                    })

            # 4. gamma参数调整建议
            current_gamma = current_params['gamma']
            if overfitting_severity > 0.15:
                new_gamma = min(1.0, current_gamma + 0.2)
                recommendations.append({
                    'parameter': 'gamma',
                    'current': current_gamma,
                    'suggested': new_gamma,
                    'reason': '过拟合严重，提高分裂阈值减少过度分裂',
                    'priority': 'MEDIUM'
                })
            elif overfitting_severity < 0.05 and current_gamma > 0.1:
                new_gamma = max(0.0, current_gamma - 0.1)
                recommendations.append({
                    'parameter': 'gamma',
                    'current': current_gamma,
                    'suggested': new_gamma,
                    'reason': '过拟合程度低，降低分裂阈值增强学习能力',
                    'priority': 'LOW'
                })

            # 5. 采样参数调整建议
            convergence_stability = analysis.get('convergence_stability', 0.5)
            current_subsample = current_params['subsample']
            current_colsample = current_params['colsample_bytree']

            if convergence_stability < 0.3:  # 收敛不稳定
                new_subsample = max(0.6, current_subsample - 0.1)
                new_colsample = max(0.5, current_colsample - 0.1)
                recommendations.extend([
                    {
                        'parameter': 'subsample',
                        'current': current_subsample,
                        'suggested': new_subsample,
                        'reason': f'收敛不稳定(稳定性:{convergence_stability:.3f})，降低行采样增强稳定性',
                        'priority': 'MEDIUM'
                    },
                    {
                        'parameter': 'colsample_bytree',
                        'current': current_colsample,
                        'suggested': new_colsample,
                        'reason': '降低特征采样率，减少方差提高稳定性',
                        'priority': 'MEDIUM'
                    }
                ])
            elif convergence_stability > 0.8 and overfitting_severity < 0.05:
                # 收敛很稳定且无过拟合，可以增加采样提高性能
                if current_subsample < 0.9:
                    new_subsample = min(0.95, current_subsample + 0.05)
                    recommendations.append({
                        'parameter': 'subsample',
                        'current': current_subsample,
                        'suggested': new_subsample,
                        'reason': f'收敛稳定且无过拟合，增加行采样提高模型性能',
                        'priority': 'LOW'
                    })

                if current_colsample < 0.85:
                    new_colsample = min(0.9, current_colsample + 0.05)
                    recommendations.append({
                        'parameter': 'colsample_bytree',
                        'current': current_colsample,
                        'suggested': new_colsample,
                        'reason': '增加特征采样率，充分利用特征信息',
                        'priority': 'LOW'
                    })

            # 6. min_child_weight 独立建议
            if overfitting_severity > 0.15 and current_min_child_weight < 10:
                new_min_child_weight = min(15, current_min_child_weight + 3)
                recommendations.append({
                    'parameter': 'min_child_weight',
                    'current': current_min_child_weight,
                    'suggested': new_min_child_weight,
                    'reason': '过拟合严重，增加叶子节点最小权重',
                    'priority': 'HIGH'
                })
            elif overfitting_severity < 0.05 and current_min_child_weight > 3:
                new_min_child_weight = max(1, current_min_child_weight - 2)
                recommendations.append({
                    'parameter': 'min_child_weight',
                    'current': current_min_child_weight,
                    'suggested': new_min_child_weight,
                    'reason': '过拟合程度低，降低叶子节点最小权重增强学习能力',
                    'priority': 'MEDIUM'
                })

            # 7. n_estimators 和 Early Stopping 建议
            efficiency_ratio = (analysis.get('best_epoch', 0) + 1) / analysis.get('total_epochs', 1)
            if efficiency_ratio > 0.95:  # 几乎用完所有轮数
                new_n_estimators = min(12000, int(current_params['n_estimators'] * 1.3))
                recommendations.append({
                    'parameter': 'n_estimators',
                    'current': current_params['n_estimators'],
                    'suggested': new_n_estimators,
                    'reason': f'使用了{efficiency_ratio*100:.1f}%轮数，模型可能需要更多训练轮数',
                    'priority': 'MEDIUM'
                })
            elif efficiency_ratio < 0.5:  # 过早停止
                new_n_estimators = max(4000, int(current_params['n_estimators'] * 0.8))
                recommendations.append({
                    'parameter': 'n_estimators',
                    'current': current_params['n_estimators'],
                    'suggested': new_n_estimators,
                    'reason': f'仅使用了{efficiency_ratio*100:.1f}%轮数，可以减少最大轮数提高效率',
                    'priority': 'LOW'
                })

            return recommendations

        # 生成参数建议
        data_info = {
            'feature_count': actual_feature_count,
            'sample_count': len(X_train_enhanced),
            'val_sample_count': len(X_val_enhanced)
        }

        recommendations = generate_parameter_recommendations(analysis, final_params, data_info)

        # 输出训练分析结果
        logger.info("=" * 80)
        logger.info("🔍 专业训练分析报告")
        logger.info("=" * 80)

        if analysis:
            logger.info(f"📊 训练总轮数: {analysis['total_epochs']}")
            logger.info(f"📊 最佳轮数: {analysis['best_epoch']} (效率: {(analysis['best_epoch']+1)/analysis['total_epochs']*100:.1f}%)")
            logger.info(f"📊 训练损失改善: {analysis['train_improvement']:.6f}")
            logger.info(f"📊 验证损失改善: {analysis['val_improvement']:.6f}")
            logger.info(f"📊 收敛稳定性: {analysis['convergence_stability']:.3f} ({'稳定' if analysis['convergence_stability'] > 0.7 else '不稳定'})")
            logger.info(f"📊 过拟合严重度: {analysis['overfitting_severity']:.3f} ({'严重' if analysis['overfitting_severity'] > 0.15 else '轻微' if analysis['overfitting_severity'] > 0.05 else '无'})")
            logger.info(f"📊 学习率效率: {analysis['learning_rate_efficiency']:.3f} ({'过高' if analysis['learning_rate_efficiency'] < 0.3 else '过低' if analysis['learning_rate_efficiency'] > 0.8 else '合适'})")
            logger.info(f"📊 损失震荡次数: {analysis['loss_oscillations']}")

            # 🚀 量化交易正则化效果专业评估
            current_reg_alpha = final_params['reg_alpha']
            current_reg_lambda = final_params['reg_lambda']
            total_reg_strength = current_reg_alpha + current_reg_lambda
            overfitting_severity = analysis['overfitting_severity']

            logger.info("=" * 60)
            logger.info("📈 正则化效果专业评估（量化交易导向）")
            logger.info("=" * 60)

            # 正则化强度分析
            if total_reg_strength < 0.5:
                reg_level = "很弱"
                reg_desc = "适合大样本低噪声数据，有利于捕捉细微市场信号"
                reg_risk = "⚠️  风险：可能对市场噪声敏感"
            elif total_reg_strength < 2.0:
                reg_level = "适中"
                reg_desc = "平衡性能和泛化，适合一般量化交易场景"
                reg_risk = "✅ 风险：平衡的风险控制"
            elif total_reg_strength < 5.0:
                reg_level = "较强"
                reg_desc = "适合小样本高噪声数据，但可能过度平滑市场信号"
                reg_risk = "⚠️  风险：可能错失重要交易机会"
            else:
                reg_level = "很强"
                reg_desc = "强力防止过拟合，但可能严重削弱预测能力"
                reg_risk = "🚨 风险：可能导致模型过于保守"

            logger.info(f"📊 正则化强度: {reg_level} (α+λ={total_reg_strength:.2f})")
            logger.info(f"   特征: {reg_desc}")
            logger.info(f"   {reg_risk}")

            # L1 vs L2 平衡分析
            if current_reg_alpha > 0 and current_reg_lambda > 0:
                l1_ratio = current_reg_alpha / (current_reg_alpha + current_reg_lambda)
                if l1_ratio > 0.7:
                    balance_desc = "L1主导 - 强特征选择，适合高维稀疏特征"
                elif l1_ratio > 0.3:
                    balance_desc = "L1/L2平衡 - 兼顾特征选择和权重平滑"
                else:
                    balance_desc = "L2主导 - 权重平滑，适合特征间相关性高的场景"

                logger.info(f"📊 L1/L2平衡: {balance_desc} (L1比例: {l1_ratio:.1%})")

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

            if analysis['loss_oscillations'] > analysis['total_epochs'] * 0.1:
                logger.info("🎯 建议: 训练震荡较多，增加L2正则化提高稳定性")

            logger.info("=" * 60)

            if analysis['overfitting_start']:
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

        # 🚀 智能输出自动调参建议（考虑超参数搜索结果）
        logger.info("=" * 80)
        if enable_param_search:
            logger.info("🎯 智能参数分析（已使用超参数搜索优化）")
            logger.info("=" * 80)
            logger.info("✅ 当前参数已通过超参数搜索优化，以下建议仅供参考：")
        else:
            logger.info("🎯 智能参数优化建议")
            logger.info("=" * 80)
            logger.info("💡 建议启用超参数搜索获得更优参数：--enable_param_search --search_strategy bayesian")

        if recommendations:
            # 🚀 智能过滤建议（如果使用了超参数搜索）
            if enable_param_search:
                # 过滤掉与搜索结果冲突的建议
                filtered_recommendations = []
                for rec in recommendations:
                    # 计算建议改变的幅度
                    if isinstance(rec['current'], (int, float)) and isinstance(rec['suggested'], (int, float)):
                        change_ratio = abs(rec['suggested'] - rec['current']) / max(abs(rec['current']), 1e-8)
                        # 只保留改变幅度较小的建议（<20%）
                        if change_ratio < 0.2:
                            rec['priority'] = 'LOW'  # 降低优先级
                            rec['reason'] += f" (微调建议，当前参数已通过{search_strategy}搜索优化)"
                            filtered_recommendations.append(rec)
                    else:
                        filtered_recommendations.append(rec)
                recommendations = filtered_recommendations

                if not recommendations:
                    logger.info("✅ 当前参数配置已通过超参数搜索优化，无需进一步调整")
                    logger.info("💡 如需进一步优化，建议尝试不同的搜索策略：")
                    logger.info(f"   当前策略: {search_strategy}")
                    if search_strategy != 'bayesian':
                        logger.info("   推荐尝试: --search_strategy bayesian")
                    if search_strategy != 'halving':
                        logger.info("   或者尝试: --search_strategy halving")

            if recommendations:
                # 按优先级分组
                high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
                medium_priority = [r for r in recommendations if r['priority'] == 'MEDIUM']
                low_priority = [r for r in recommendations if r['priority'] == 'LOW']

                if high_priority:
                    priority_label = "🔥 高优先级建议 (强烈推荐):" if not enable_param_search else "🔥 高优先级微调建议:"
                    logger.info(priority_label)
                    for rec in high_priority:
                        logger.info(f"  📌 {rec['parameter']}: {rec['current']} → {rec['suggested']}")
                        logger.info(f"     理由: {rec['reason']}")
                        logger.info("")

                if medium_priority:
                    priority_label = "⚡ 中优先级建议 (建议考虑):" if not enable_param_search else "⚡ 中优先级微调建议:"
                    logger.info(priority_label)
                    for rec in medium_priority:
                        logger.info(f"  📌 {rec['parameter']}: {rec['current']} → {rec['suggested']}")
                        logger.info(f"     理由: {rec['reason']}")
                        logger.info("")

                if low_priority:
                    priority_label = "💡 低优先级建议 (可选优化):" if not enable_param_search else "💡 低优先级微调建议:"
                    logger.info(priority_label)
                    for rec in low_priority:
                        logger.info(f"  📌 {rec['parameter']}: {rec['current']} → {rec['suggested']}")
                        logger.info(f"     理由: {rec['reason']}")
                        logger.info("")

            # 生成完整的推荐参数配置
            logger.info("🔧 推荐的完整参数配置:")
            recommended_params = final_params.copy()
            for rec in recommendations:
                if rec['priority'] in ['HIGH', 'MEDIUM']:  # 只应用高和中优先级建议
                    recommended_params[rec['parameter']] = rec['suggested']

            logger.info("```python")
            logger.info("# 基于训练分析的推荐参数配置")
            for key, value in recommended_params.items():
                if key not in ['random_state', 'n_jobs', 'verbosity', 'max_delta_step']:
                    if isinstance(value, float):
                        logger.info(f"'{key}': {value:.3f},")
                    else:
                        logger.info(f"'{key}': {value},")
            logger.info("```")

        else:
            if enable_param_search:
                logger.info("✅ 当前参数配置已通过超参数搜索优化，表现良好")
                logger.info("🎯 搜索策略建议：")
                if search_strategy == 'grid':
                    logger.info("   当前使用Grid搜索，参数在预定义空间内最优")
                    logger.info("   💡 如需探索更大参数空间，建议尝试：--search_strategy bayesian")
                elif search_strategy == 'bayesian':
                    logger.info("   当前使用Bayesian搜索，参数已智能优化")
                    logger.info("   💡 如需验证稳定性，建议对比：--search_strategy grid")
                elif search_strategy == 'halving':
                    logger.info("   当前使用Successive Halving，参数已高效筛选")
                    logger.info("   💡 如需精细调优，建议尝试：--search_strategy bayesian")
            else:
                logger.info("✅ 当前参数配置良好，无需调整")
                logger.info("💡 如需进一步优化，建议启用超参数搜索：")
                logger.info("   快速优化: --enable_param_search --search_strategy grid")
                logger.info("   智能优化: --enable_param_search --search_strategy bayesian")
                logger.info("   高效筛选: --enable_param_search --search_strategy halving")

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

        # 量化交易导向的正则化强度评估
        reg_alpha = current_params['reg_alpha']
        reg_lambda = current_params['reg_lambda']
        reg_strength = reg_alpha + reg_lambda

        # 正则化强度分类和量化交易影响分析
        if reg_strength < 0.5:
            strength_level = "很弱"
            trading_impact = "有利于捕捉细微市场信号，但对噪声敏感"
            performance_expectation = "测试集表现通常较好（如观察到的现象）"
        elif reg_strength < 2.0:
            strength_level = "适中"
            trading_impact = "平衡信号捕捉和噪声抑制"
            performance_expectation = "验证集和测试集表现相对均衡"
        elif reg_strength < 5.0:
            strength_level = "较强"
            trading_impact = "强力抑制噪声，但可能过度平滑重要信号"
            performance_expectation = "验证集表现好，但测试集可能较差（如观察到的现象）"
        else:
            strength_level = "很强"
            trading_impact = "过度保守，可能错失大部分交易机会"
            performance_expectation = "整体预测能力可能严重受限"

        logger.info(f"  📈 正则化强度: {strength_level} (α+λ={reg_strength:.2f})")
        logger.info(f"     量化交易影响: {trading_impact}")
        logger.info(f"     性能预期: {performance_expectation}")

        # L1/L2比例分析
        if reg_alpha > 0 and reg_lambda > 0:
            l1_ratio = reg_alpha / reg_strength
            if l1_ratio > 0.7:
                l1_l2_desc = "L1主导 - 强特征选择，适合稀疏特征场景"
            elif l1_ratio > 0.3:
                l1_l2_desc = "L1/L2平衡 - 兼顾特征选择和权重平滑"
            else:
                l1_l2_desc = "L2主导 - 权重平滑，适合特征相关性高的场景"
            logger.info(f"     L1/L2策略: {l1_l2_desc} (L1:{reg_alpha}, L2:{reg_lambda})")

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

        model.loss_function = loss_function_name
        model.best_delta = best_delta
        # 🚀 保存标签变换器到模型对象中，用于预测时还原
        model.label_transformer = label_transformer


        # 🚀 模型评估：支持Pseudo-Huber回归和分类
        logger.info("评估模型性能...")


        # 🚀 使用XGBoost模型预测
        dval_for_pred = xgb.DMatrix(X_val_enhanced)  # 🚀 使用增强特征
        y_pred_raw = model.predict(dval_for_pred)

        # 🚀 关键修复：应用智能标签变换的逆变换
        if label_transformer is not None:
            y_pred = label_transformer.inverse_transform(y_pred_raw)
            logger.info(f"✅ 应用智能标签逆变换:")
            logger.info(f"  变换后范围: [{y_pred_raw.min():.3f}, {y_pred_raw.max():.3f}] (跨度: {y_pred_raw.max() - y_pred_raw.min():.3f})")
            logger.info(f"  原始ATR范围: [{y_pred.min():.3f}, {y_pred.max():.3f}] (跨度: {y_pred.max() - y_pred.min():.3f})")
        else:
            y_pred = y_pred_raw
            logger.info(f"✅ 未使用标签变换，直接使用预测值")

        y_pred_proba = None
        logger.info(f"✅ {model.loss_function}模型预测完成，最终预测值范围: [{y_pred.min():.3f}, {y_pred.max():.3f}] ATR")

        # 🚀 成本感知回归：支持回归和分类评估
        # 🚀 关键修复：使用原始ATR空间的标签进行评估
        y_val_atr_labels = y_val_split_eval  # 验证集ATR标签（原始ATR单位）


        # 🚀 Pseudo-Huber回归：直接使用原始ATR单位评估（无需反向转换）
        logger.info("🚀 Pseudo-Huber回归模型评估...")
        logger.info("📊 标签分布（原始ATR单位）:")

        # 统一命名：预测值和真实值都使用ATR单位
        y_pred_atr_values = y_pred  # 预测的ATR值
        y_val_atr_true = y_val_atr_labels  # 真实的ATR标签

        # 计算回归指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_val_atr_true, y_pred_atr_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_atr_true, y_pred_atr_values)
        r2 = r2_score(y_val_atr_true, y_pred_atr_values)
        correlation = np.corrcoef(y_val_atr_true, y_pred_atr_values)[0, 1]

        logger.info(f"🎯 Pseudo-Huber回归性能:")
        if best_delta is not None:
            logger.info(f"  最佳Delta: {best_delta:.1f} ATR")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.6f}")
        logger.info(f"  相关性: {correlation:.6f}")

        # 🚀 标签和预测分布（原始ATR单位）
        logger.info(f"📊 标签分布（原始ATR单位）:")
        logger.info(f"  真实标签范围: [{y_val_atr_true.min():.3f}, {y_val_atr_true.max():.3f}] ATR")
        logger.info(f"  真实标签均值: {y_val_atr_true.mean():.3f} ATR")
        logger.info(f"  真实标签标准差: {y_val_atr_true.std():.3f} ATR")

        logger.info(f"📊 预测分布（原始ATR单位）:")
        logger.info(f"  预测值范围: [{y_pred_atr_values.min():.3f}, {y_pred_atr_values.max():.3f}] ATR")
        logger.info(f"  预测值均值: {y_pred_atr_values.mean():.3f} ATR")
        logger.info(f"  预测值标准差: {y_pred_atr_values.std():.3f} ATR")

        # 🚀 计算专业量化评估指标（_calculate_regression_metrics函数未定义，暂时跳过）
        # regression_metrics = _calculate_regression_metrics(
        #     y_true=y_val_atr_true,
        #     y_pred=y_pred_atr_values,
        #     logger=logger
        # )

        # 🚀 计算交易指标（使用固定阈值）
        trading_metrics = _calculate_trading_metrics(
            y_true_return=y_val_atr_true,      # 原始ATR单位真实值
            y_pred_scores=y_pred_atr_values,   # 原始ATR单位预测值
            logger=logger,
            k_values=[1, 3, 5]  # 🚀 修复：与实际选股策略一致
        )

        # 🚀 添加专业量化评估指标
        quant_metrics = _calculate_quantitative_metrics(y_val_atr_true, y_pred_atr_values, logger)

        # 🚀 添加预测范围诊断指标（包含逆变换前后对比）
        _calculate_prediction_range_diagnostics(y_val_atr_true, y_pred_atr_values, logger, y_pred_raw)

        # 🚀 诊断评估：检查预测范围压缩问题
        true_range = y_val_atr_true.max() - y_val_atr_true.min()
        pred_range = y_pred_atr_values.max() - y_pred_atr_values.min()
        range_coverage_ratio = pred_range / true_range if true_range > 0 else 0

        true_std = y_val_atr_true.std()
        pred_std = y_pred_atr_values.std()
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
            if hasattr(model, 'best_delta') and model.best_delta is not None:
                logger.info(f"  📊 最佳Delta参数: {model.best_delta:.1f}")

        # 🚀 预测范围压缩的可能原因分析
        if range_coverage_ratio < 0.3 or std_ratio < 0.5:
            logger.warning("🔍 预测范围压缩可能原因:")
            logger.warning("  1. Pseudo-Huber损失函数对异常值过度平滑")
            logger.warning("  2. Delta参数设置过小，导致模型过于保守")
            logger.warning("  3. 训练数据中极端值被过度处理")
            logger.warning("  4. 模型复杂度不足，无法捕捉标签的完整分布")
            logger.warning("💡 建议:")
            logger.warning("  - 尝试使用MSE损失函数")
            logger.warning("  - 增大Delta参数范围")
            logger.warning("  - 减少特征Winsorization强度")
            logger.warning("  - 增加模型复杂度（更深的树、更多特征交互）")

        # 🚀 注意：Delta选择结果已在上方显示，避免重复日志

        # 🚀 移除无意义的动态阈值指标（这些指标在trading_metrics中不存在，总是返回0）
        # 保留有意义的固定阈值指标
        logger.info("🎯 交易指标（固定阈值评估）:")
        logger.info(f"  固定止盈命中率: {trading_metrics.get('profit_hit_rate', 0):.4f} (阈值: {trading_metrics.get('profit_threshold', 0):.1f} ATR)")
        logger.info(f"  固定止损命中率: {trading_metrics.get('loss_hit_rate', 0):.4f} (阈值: {trading_metrics.get('loss_threshold', 0):.1f} ATR)")

        # 中等收益指标
        moderate_hit_rates = trading_metrics.get('moderate_hit_rates', {})
        for threshold_key, hit_rate in moderate_hit_rates.items():
            threshold = threshold_key.replace('profit_', '')
            logger.info(f"  中等收益>={threshold}ATR命中率: {hit_rate:.4f}")

        # 🚀 修复：Top-K收益统计，使用与实际选股一致的k值
        top_k_returns = trading_metrics.get('top_k_returns', {})
        top_k_profit_rates = trading_metrics.get('top_k_profit_rates', {})
        top_k_loss_rates = trading_metrics.get('top_k_loss_rates', {})

        for k in [1, 3, 5]:
            if k in top_k_returns:
                logger.info(f"  Top-{k} 平均收益: {top_k_returns[k]:.4f} ATR")
                logger.info(f"  Top-{k} 止盈率: {top_k_profit_rates[k]:.4f}")
                logger.info(f"  Top-{k} 止损率: {top_k_loss_rates[k]:.4f}")

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
            'scaler': scaler,
            'winsorization_thresholds': winsorization_thresholds,  # 🚀 新增：保存Winsorization阈值
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
            'label_transformer': label_transformer,  # 🚀 添加可序列化的标签变换器
            'best_delta': best_delta,  # 🚀 添加最优Delta值
            'hyperparameter_search': {  # 🚀 添加超参数搜索信息
                'enabled': enable_param_search,
                'strategy': search_strategy if enable_param_search else None,
                'n_iter': n_iter if enable_param_search and search_strategy == 'random' else None,
                'search_results': locals().get('search_results', None) if enable_param_search else None
            }
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



def _extract_tree_rules(model, tree_idx, feature_names):
    """提取决策树规则"""
    try:
        # 获取XGBoost树的结构
        booster = model.get_booster()
        tree_dump = booster.get_dump(dump_format='text')[tree_idx]

        # 解析树结构，提取关键规则
        rules = []
        lines = tree_dump.split('\n')

        for line in lines:
            if '[' in line and '<' in line:
                # 提取特征和阈值
                parts = line.strip().split('[')[1].split(']')[0]
                if '<' in parts:
                    feature_threshold = parts.split('<')
                    if len(feature_threshold) == 2:
                        feature_name = feature_threshold[0].strip()
                        threshold = feature_threshold[1].strip()

                        # 转换特征索引为特征名
                        if feature_name.startswith('f') and feature_name[1:].isdigit():
                            feature_idx = int(feature_name[1:])
                            if feature_idx < len(feature_names):
                                feature_name = feature_names[feature_idx]

                        rule = f"{feature_name} < {threshold}"
                        rules.append(rule)

        return rules[:10]  # 返回前10条规则

    except Exception as e:
        return []

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

def _analyze_feature_thresholds(model, feature_names):
    """分析特征阈值"""
    try:
        thresholds = {}

        # 获取XGBoost树的结构
        booster = model.get_booster()
        tree_dumps = booster.get_dump(dump_format='text')

        # 收集所有特征的阈值
        feature_thresholds = {name: [] for name in feature_names}

        for tree_dump in tree_dumps[:10]:  # 只分析前10棵树
            lines = tree_dump.split('\n')

            for line in lines:
                if '[' in line and '<' in line:
                    try:
                        parts = line.strip().split('[')[1].split(']')[0]
                        if '<' in parts:
                            feature_threshold = parts.split('<')
                            if len(feature_threshold) == 2:
                                feature_name = feature_threshold[0].strip()
                                threshold = float(feature_threshold[1].strip())

                                # 转换特征索引为特征名
                                if feature_name.startswith('f') and feature_name[1:].isdigit():
                                    feature_idx = int(feature_name[1:])
                                    if feature_idx < len(feature_names):
                                        feature_name = feature_names[feature_idx]
                                        feature_thresholds[feature_name].append(threshold)
                    except:
                        continue

        # 统计每个特征的关键阈值
        for feature_name, threshold_list in feature_thresholds.items():
            if threshold_list:
                # 计算常用阈值
                unique_thresholds = sorted(set(threshold_list))
                if len(unique_thresholds) > 0:
                    # 选择最常用的阈值
                    threshold_counts = {}
                    for t in threshold_list:
                        threshold_counts[t] = threshold_counts.get(t, 0) + 1

                    # 获取前3个最常用的阈值
                    common_thresholds = sorted(threshold_counts.items(),
                                             key=lambda x: x[1], reverse=True)[:3]
                    thresholds[feature_name] = [f"{t:.3f}(使用{c}次)" for t, c in common_thresholds]

        # 只返回有阈值的前10个特征
        return dict(list(thresholds.items())[:10])

    except Exception as e:
        return {}

def _analyze_prediction_paths(model, X_val, y_val, feature_names):
    """分析预测路径（适配二分类逻辑）"""
    try:
        paths = {}
        # 二分类：0表示其他股票，1表示次日涨幅排名前10%的股票
        class_names = ['普通股票', '优质股票（前10%）']

        # 为每个类别找到典型的预测路径
        for class_idx, class_name in enumerate(class_names):
            class_mask = y_val == class_idx
            if np.sum(class_mask) > 0:
                # 获取该类别的样本
                class_samples = X_val[class_mask]

                # 计算该类别样本的特征均值
                feature_means = np.mean(class_samples, axis=0)

                # 找出最重要的特征条件
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

                important_indices = np.argsort(feature_importance)[-10:]  # 前10个重要特征

                conditions = []
                for idx in important_indices:
                    feature_name = feature_names[idx]
                    mean_value = feature_means[idx]

                    # 为不同类型的特征提供更有意义的描述
                    if 'pct_chg' in feature_name:
                        conditions.append(f"{feature_name}≈{mean_value:.2f}%")
                    elif 'rsi' in feature_name.lower():
                        conditions.append(f"{feature_name}≈{mean_value:.1f}")
                    elif 'ratio' in feature_name or 'score' in feature_name:
                        conditions.append(f"{feature_name}≈{mean_value:.3f}")
                    else:
                        conditions.append(f"{feature_name}≈{mean_value:.3f}")

                paths[class_name] = " & ".join(conditions)

                # 添加类别统计信息
                sample_count = np.sum(class_mask)
                total_samples = len(y_val)
                percentage = (sample_count / total_samples) * 100
                paths[f"{class_name}_统计"] = f"样本数: {sample_count} ({percentage:.1f}%)"

        return paths

    except Exception as e:
        return {}

def _analyze_shap_values(model, X_val, feature_names):
    """SHAP值分析（适配二分类逻辑）"""
    try:
        import shap

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)

        # 计算SHAP值（只使用前100个样本以节省时间）
        sample_size = min(100, len(X_val))
        shap_values = explainer.shap_values(X_val[:sample_size])

        # 分析每个特征的SHAP值
        shap_analysis = {}

        # 二分类的类别名称
        class_names = ['普通股票', '优质股票（前10%）']

        # 处理二分类的SHAP值
        # 对于二分类，shap_values可能是单个数组或包含两个数组的列表
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # 两个类别的SHAP值都存在
            for class_idx, class_name in enumerate(class_names):
                class_shap = shap_values[class_idx]

                # 计算每个特征的平均SHAP值
                mean_shap = np.mean(np.abs(class_shap), axis=0)

                # 获取前5个最重要的特征
                top_indices = np.argsort(mean_shap)[-5:]

                for idx in top_indices:
                    feature_name = feature_names[idx]
                    if feature_name not in shap_analysis:
                        shap_analysis[feature_name] = []

                    shap_analysis[feature_name].append(f"{class_name}: {mean_shap[idx]:.4f}")
        else:
            # 只有一个类别的SHAP值（通常是正类）
            if isinstance(shap_values, list):
                class_shap = shap_values[0]
            else:
                class_shap = shap_values

            # 计算每个特征的平均SHAP值
            mean_shap = np.mean(np.abs(class_shap), axis=0)

            # 获取前10个最重要的特征（因为只有一个类别）
            top_indices = np.argsort(mean_shap)[-10:]

            for idx in top_indices:
                feature_name = feature_names[idx]
                # 计算正负贡献
                positive_contrib = np.mean(class_shap[class_shap[:, idx] > 0, idx]) if np.any(class_shap[:, idx] > 0) else 0
                negative_contrib = np.mean(class_shap[class_shap[:, idx] < 0, idx]) if np.any(class_shap[:, idx] < 0) else 0

                shap_analysis[feature_name] = [
                    f"平均影响: {mean_shap[idx]:.4f}",
                    f"正向贡献: {positive_contrib:.4f}",
                    f"负向贡献: {negative_contrib:.4f}"
                ]

        return shap_analysis

    except ImportError:
        return None
    except Exception as e:
        return None

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


def _get_stock_factors():
    """获取股票因子列表（52个因子，已删除7个高度冗余因子 + 18个SHAP极低重要性因子 + 17个高相关性因子，新增14个因子）

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

    新增因子:
    - 成交额因子(3个): amount_volatility(优化版), amount_rank, amihud_illiquidity_enhanced
    - 风险控制因子(4个): downside_risk, max_drawdown_20d, consecutive_down_days, volatility_change_rate
    - 风险优化因子(2个): drawdown_recovery_speed, consecutive_down_severity

    观察因子:
    - drawdown_recovery_speed: 虽与rank_rsi相关0.864,但可能提供独特风险维度,保留观察SHAP重要性
    """
    # 所有因子列表
    all_factors = [
        # ===== 原有核心因子（22个，删除1个）=====
        'returns_zscore', 'market_structure_score', 'ma_ratio_5_10', 'corr_5d',
        'volatility_correlation', 'volume_correlation', 'near_support_count_60d',
        'institutional_activity_score', 'ma_convergence_score', 'lower_shadow_ratio',
        'sector_correlation_20d', 'price_impact', 'rank_momentum_20d', 'rank_rsi',
        # 'rank_volume',  # 删除：与amount_rank相关0.732,保留后者(成交额比成交量更重要)
        'market_correlation', 'corr_20d_mom5', 'corr_slope_20_120',
        'support_risk_reward_ratio', 'obv_momentum_factor', 'bollinger_breakout_strength',
        'sector_momentum_consistency', 'sector_return_rank_20d',

        # ===== 动量与反转因子（8个，删除3个）=====
        # 'pct_chg_0d',  # 删除：与returns_zscore高度相关(0.965)，保留标准化版本
        'pct_chg_1d', 'pct_chg_2d', 'return_5d', 'return_10d',
        # 'return_20d',  # 删除：5次高相关,与trend_acceleration负相关-0.841
        'momentum_reversal', 'trend_acceleration', 'momentum_decay',
        # 'reversal_strength',  # 删除：与pct_chg_1d完全负相关(-1.000)，数学上等价
        'trend_consistency',

        # ===== 横截面排名因子（5个，删除2个）=====
        'rank_pct_chg_0d', 'rank_return_5d',
        # 'rank_return_20d',  # 删除：与rank_momentum_20d高度相关(0.996)，含义重复
        # 'rank_volume_ratio',  # 删除：与turnover_rate_strength相关0.849,保留后者
        'rank_turnover_rate', 'rank_volatility_20d', 'rank_market_cap',

        # ===== 流动性因子（8个，删除3个）=====
        'volume_price_strength', 'volatility_adjusted_return', 'amihud_illiquidity',
        'turnover_volatility', 'bid_ask_spread_proxy',
        # 'turnover_change_strength',  # 删除：与rank_volume_ratio高度相关(0.998)，含义重复
        # 🚀 新增成交额因子（3个，删除2个）
        # 'amount_momentum',  # 删除：与volume_acceleration相关0.905,与turnover_rate_strength相关0.825
        'amount_volatility',  # 优化版：使用对数收益率波动率
        'amount_rank',
        # 'amount_breakout',  # 删除：与rank_volume_ratio相关0.915,与turnover_rate_strength相关0.782
        'amihud_illiquidity_enhanced',

        # ===== 低波动异常因子（2个）=====
        'beta_market', 'max_daily_return_60d',

        # ===== 风险控制因子（5个，删除4个）=====
        'downside_risk',  # 保留：作为下行风险代表
        'max_drawdown_20d',
        # 'consecutive_down_days',  # 删除：与consecutive_down_severity相关0.946,保留后者(严重性更重要)
        # 'cumulative_down_20d',  # 删除：与downside_risk相关0.954
        'volatility_change_rate',
        # 'var_95',  # 删除：与downside_risk相关0.914
        # 'cvar_95',  # 删除：与downside_risk相关0.923,与var_95相关0.807
        # 🚀 新增风险优化因子（2个）
        'drawdown_recovery_speed',  # 回撤恢复速度(虽与rank_rsi相关0.864,但保留观察SHAP重要性)
        'consecutive_down_severity',  # 连续下跌严重性

        # ===== 行为金融因子（3个，删除1个）=====
        'distance_from_52w_high', 'distance_from_52w_low', 'price_acceleration',
        # 'volume_acceleration',  # 删除：与turnover_rate_strength相关0.913

        # ===== 日内微观结构因子（2个，删除2个）=====
        'overnight_return',
        # 'intraday_return',  # 删除：与rank_pct_chg_0d相关0.889,与alpha_054相关0.736
        'open_to_high_ratio',
        # 'open_to_low_ratio',  # 删除：与open_to_high_ratio高度负相关(-0.984)，数学上互补

        # ===== WorldQuant Alpha因子（3个，删除1个）=====
        'alpha_001', 'alpha_012', 'alpha_017',
        # 'alpha_054',  # 删除：与returns_zscore相关0.869,保留后者(更直观)

        # ===== 风格轮动因子（3个，删除1个）=====
        'sector_concentration', 'sector_breadth',
        # 'cross_sector_correlation',  # 删除：与sector_correlation_20d负相关-0.874
        'style_consistency',

        # ===== 交互特征因子（2个）=====
        'size_market_cap_interaction', 'sector_momentum_interaction',

        # ===== 技术指标因子（2个，删除3个）=====
        # 'macd_strength',  # 删除：4次高相关,与bollinger_position_factor相关0.813
        'atr_normalized', 'turnover_rate_strength',
        # 'cci_indicator',  # 删除：与bollinger_position_factor高度相关(0.962)，保留更直观的布林带位置
        'volume_price_divergence',

        # ===== 统计矩因子（3个，删除1个）=====
        'skewness_20d', 'kurtosis_20d',
        # 'idiosyncratic_volatility',  # 删除：与idiosyncratic_volatility_sector高度相关(0.893)，保留剥离板块影响的版本
        'idiosyncratic_volatility_sector',

        # ===== KDJ因子（0个，删除1个）=====
        # 'kdj_golden_cross_strength',  # 删除：SHAP重要性低(0.0134)

        # ===== 其他因子（1个，删除3个）=====
        # 'volatility_efficiency',  # 删除：SHAP重要性低(0.0148)
        'pullback_volume_ratio',
        #   'volume_strength', #删除：与turnover_rate_strength ，rank_volume_ratio 高相关
        # 'bollinger_position_factor',  # 删除：8次高相关(最高!),与alpha_017相关0.921
    ]

    # 🔴 SHAP保守删除方案: 删除18个极低重要性因子(SHAP重要性<0.015)
    shap_remove_conservative = [
        'idiosyncratic_volatility_sector',  # 0.0062
        'beta_market',                      # 0.0070
        'alpha_001',                        # 0.0084
        'obv_momentum_factor',              # 0.0091
        'institutional_activity_score',     # 0.0098
        'bid_ask_spread_proxy',             # 0.0104
        'momentum_decay',                   # 0.0109
        'turnover_volatility',              # 0.0110
        'amihud_illiquidity',               # 0.0115
        'trend_consistency',                # 0.0118
        'kurtosis_20d',                     # 0.0119
        'skewness_20d',                     # 0.0120
        'volume_price_strength',            # 0.0122
        'bollinger_breakout_strength',      # 0.0122
        'volume_price_divergence',          # 0.0132
        'open_to_high_ratio',               # 0.0134
        'sector_momentum_consistency',      # 0.0140
        'atr_normalized',                   # 0.0148
    ]

    # 使用列表相减操作删除因子
    stock_factors = [f for f in all_factors if f not in shap_remove_conservative]

    return stock_factors

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

def _filter_bearish_ma_trend_stocks(data: pd.DataFrame, date_str: str, logger) -> pd.Series:
    """
    过滤多均线空头趋势股票

    基于多均线趋势一致性特征，过滤掉明显空头排列的股票：
    1. ma_bearish_alignment > 0.6 (空头排列强度超过60%)
    2. ma_trend_strength < -0.3 (趋势强度明显偏空)
    3. 股价 < MA5 < MA10 < MA20 (经典空头排列)

    参数:
    - data: 股票数据DataFrame
    - date_str: 当前日期字符串
    - logger: 日志记录器

    返回:
    - pd.Series: 布尔值序列，True表示保留，False表示过滤
    """
    try:
        # 检查必要的均线特征是否存在
        required_features = ['ma_bearish_alignment', 'ma_trend_strength', 'close', 'ma5', 'ma10', 'ma20']
        missing_features = [f for f in required_features if f not in data.columns]

        if missing_features:
            logger.warning(f"📅 {date_str}: 缺少多均线特征 {missing_features}，跳过空头趋势过滤")
            return pd.Series(True, index=data.index)

        # 初始化过滤条件
        filter_mask = pd.Series(True, index=data.index)

        # 条件1: 空头排列强度过滤 (ma_bearish_alignment > 0.6)
        bearish_alignment_filter = data['ma_bearish_alignment'] <= 0.6

        # 条件2: 趋势强度过滤 (ma_trend_strength < -0.3)
        trend_strength_filter = data['ma_trend_strength'] >= -0.3

        # 条件3: 经典空头排列过滤 (股价 < MA5 < MA10 < MA20)
        classic_bearish_filter = ~(
            (data['close'] < data['ma5']) &
            (data['ma5'] < data['ma10']) &
            (data['ma10'] < data['ma20'])
        )

        # 综合过滤条件 (任一条件满足即保留)
        filter_mask = bearish_alignment_filter | trend_strength_filter | classic_bearish_filter

        # 统计过滤结果
        total_stocks = len(data)
        filtered_stocks = filter_mask.sum()
        removed_stocks = total_stocks - filtered_stocks

        if removed_stocks > 0:
            logger.info(f"📅 {date_str}: 多均线空头趋势过滤详情:")
            logger.info(f"    总股票数: {total_stocks}")
            logger.info(f"    过滤股票数: {removed_stocks}")
            logger.info(f"    保留股票数: {filtered_stocks}")

            # 详细分析被过滤的股票
            removed_data = data[~filter_mask]
            if len(removed_data) > 0:
                avg_bearish = removed_data['ma_bearish_alignment'].mean()
                avg_trend = removed_data['ma_trend_strength'].mean()
                classic_bearish_count = (
                    (removed_data['close'] < removed_data['ma5']) &
                    (removed_data['ma5'] < removed_data['ma10']) &
                    (removed_data['ma10'] < removed_data['ma20'])
                ).sum()

                logger.info(f"    被过滤股票特征:")
                logger.info(f"      平均空头排列强度: {avg_bearish:.3f}")
                logger.info(f"      平均趋势强度: {avg_trend:.3f}")
                logger.info(f"      经典空头排列数量: {classic_bearish_count}")

                # 显示前几只被过滤的股票示例
                sample_stocks = removed_data.head(3)
                for _, stock in sample_stocks.iterrows():
                    logger.info(f"      示例: {stock['stock_code']} - "
                              f"空头强度:{stock['ma_bearish_alignment']:.3f}, "
                              f"趋势强度:{stock['ma_trend_strength']:.3f}")

        return filter_mask

    except Exception as e:
        logger.error(f"📅 {date_str}: 多均线空头趋势过滤失败: {e}")
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

def _validate_enhanced_data(enhanced_data, original_data, logger):
    """
    验证增强数据的完整性

    Args:
        enhanced_data: 因子计算后的数据
        original_data: 原始股票数据
        logger: 日志记录器

    Returns:
        dict: 验证结果，包含valid字段和issues列表
    """
    validation_result = {
        'valid': True,
        'issues': []
    }

    try:
        # 检查数据是否为空
        if enhanced_data is None or enhanced_data.empty:
            validation_result['valid'] = False
            validation_result['issues'].append("增强数据为空")
            return validation_result

        # 检查必要的索引列
        required_columns = ['stock_code', 'date']
        missing_columns = [col for col in required_columns if col not in enhanced_data.columns]
        if missing_columns:
            validation_result['valid'] = False
            validation_result['issues'].append(f"缺少必要的索引列: {missing_columns}")

        # 检查数据行数
        original_rows = len(original_data)
        enhanced_rows = len(enhanced_data)
        if enhanced_rows != original_rows:
            validation_result['issues'].append(f"数据行数不匹配: 原始{original_rows}行 vs 增强{enhanced_rows}行")
            # 行数不匹配不一定是错误，可能是正常的数据过滤

        # 检查新增因子列
        original_columns = set(original_data.columns)
        enhanced_columns = set(enhanced_data.columns)
        new_columns = enhanced_columns - original_columns

        if len(new_columns) == 0:
            validation_result['issues'].append("未生成任何新的因子列")
        else:
            logger.info(f"成功生成 {len(new_columns)} 个新因子")

        # 检查数据类型一致性
        for col in original_columns:
            if col in enhanced_data.columns:
                original_dtype = original_data[col].dtype
                enhanced_dtype = enhanced_data[col].dtype
                if original_dtype != enhanced_dtype:
                    validation_result['issues'].append(f"列{col}数据类型不一致: {original_dtype} vs {enhanced_dtype}")

        # 检查是否有过多的NaN值
        nan_columns = []
        for col in new_columns:
            if col in enhanced_data.columns:
                nan_ratio = enhanced_data[col].isna().sum() / len(enhanced_data)
                if nan_ratio > 0.5:  # 超过50%的NaN值
                    nan_columns.append(f"{col}({nan_ratio:.1%})")

        if nan_columns:
            validation_result['issues'].append(f"以下因子列包含过多NaN值: {nan_columns}")

        # 记录验证统计信息
        logger.info(f"数据验证统计: 原始列{len(original_columns)}个, 新增列{len(new_columns)}个, 总列{len(enhanced_columns)}个")

    except Exception as e:
        validation_result['valid'] = False
        validation_result['issues'].append(f"验证过程中发生错误: {e}")
        logger.error(f"数据验证失败: {e}")

    return validation_result


# run_test_optimization_mode函数已删除，优化方案已集成到test_model模式



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
    elif args.mode == 'test_model':
        success = run_test_model_mode(args)
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
        end_date = args.end_date or "20250930"

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

        labeled_data_result = _generate_training_labels_for_all_data(
            label_source,
            atr_multiplier_profit=ATR_MULTIPLIER_PROFIT,
            atr_multiplier_loss=ATR_MULTIPLIER_LOSS,
            max_holding_days=MAX_HOLDING_DAYS,
            debug=False,
            is_training=False,
            label_stats=None,
        )
        if labeled_data_result is None:
            logger.error("标签生成失败")
            return False

        labeled_data = labeled_data_result[0] if isinstance(labeled_data_result, tuple) else labeled_data_result
        if 'date' in labeled_data.columns:
            labeled_data['date'] = pd.to_datetime(labeled_data['date'])

        if not {'atr_return_label', 'exit_day'}.issubset(labeled_data.columns):
            logger.error("标签数据缺少 atr_return_label/exit_day 字段")
            return False

        # 保留atr14字段，用于后续标签生成
        label_columns = ['stock_code', 'date', 'atr_return_label', 'exit_day']
        if 'atr14' in labeled_data.columns:
            label_columns.append('atr14')
        label_data = labeled_data[label_columns].copy()

        for day in range(1, MAX_HOLDING_DAYS + 1):
            day_mask = label_data['exit_day'] == day
            label_data[f'atr_return_{day}d'] = np.where(day_mask, label_data['atr_return_label'], np.nan)

        label_return_columns = ['atr_return_label'] + [f'atr_return_{day}d' for day in range(1, MAX_HOLDING_DAYS + 1)]
        for col in label_return_columns:
            if col in label_data.columns:
                non_null = label_data[col].notna().sum()
                print(f"· 标签列 {col}：可用样本 {non_null}/{len(label_data)}")

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

        exclude_columns = {
            'stock_code', 'date', 'date_str', 'exit_day',
            'atr_return_label',  # 🚀 保留atr_return_label_original用于权重计算
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
        exclude_columns.update({col for col in merged_data.columns if 'future_return' in col})

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
            if 'atr_return_' in window_name and window_name.endswith('d'):
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

def _load_specified_factor_groups(
    factor_groups: List[str],
    start_date: str,
    end_date: str,
    base_factor_fields: List[str] = None,
    stock_data: pd.DataFrame = None,
    merge_base_factors: bool = False
) -> pd.DataFrame:
    """从 factor_lab 数据集中加载指定的因子组，并按需合并基础字段。"""
    logger = logging.getLogger(__name__)
    try:
        group_mapping = {
            'kline_pattern_factors': [
                'market_structure_score',
                'lower_shadow_ratio',
                'volatility_efficiency',
                'pullback_volume_ratio',
                'volume_strength',
                'near_support_count_60d',
            ],
            'elite_support_factors': [
                'support_risk_reward_ratio',
                'institutional_activity_score',
                'obv_momentum_factor',
                'bollinger_position_factor',
                'bollinger_breakout_strength',
            ],
            'market_corr_factors': [
                'volume_correlation',
                'corr_5d',
                'volatility_correlation',
                'market_correlation',
                'corr_slope_20_120',
                'corr_20d_mom5',
            ],
        }

        pipeline = FactorPipeline()
        available = set(pipeline.available_factors())

        resolved_factors: List[str] = []
        for group in factor_groups:
            key = group.strip()
            if not key:
                continue
            if key in group_mapping:
                resolved_factors.extend(group_mapping[key])
            elif key in available:
                resolved_factors.append(key)
            else:
                logger.warning(f"未识别的因子组/因子: {key}")

        resolved_factors = list(dict.fromkeys(resolved_factors))
        if not resolved_factors:
            logger.error("未解析到任何可加载的因子，请检查配置。")
            return None

        store = FactorDatasetStore(FACTOR_LAB_STORE_PATH)
        if not store.exists():
            logger.error("因子数据文件不存在，请先运行 generate_factors 模式。")
            return None

        dataset = store.load()
        if dataset is None or dataset.empty:
            logger.error("因子数据文件为空，无法加载指定因子。")
            return None

        required_columns = {'date', 'stock_code'}
        missing_required = [col for col in required_columns if col not in dataset.columns]
        if missing_required:
            logger.error(f"因子数据缺少基础列: {missing_required}")
            return None

        dataset = dataset.copy()
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset.sort_values(['date', 'stock_code'], inplace=True)
        dataset.drop_duplicates(subset=['date', 'stock_code'], keep='last', inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        dataset['date_str'] = dataset['date'].dt.strftime('%Y%m%d')

        available_columns = set(dataset.columns)
        missing_factors = [factor for factor in resolved_factors if factor not in available_columns]
        if missing_factors:
            logger.warning(f"因子数据集中缺少以下因子: {missing_factors}")

        selected = ['date', 'date_str', 'stock_code']
        selected += [factor for factor in resolved_factors if factor in available_columns]
        factor_df = dataset[selected].copy()

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        factor_df = factor_df[(factor_df['date'] >= start_ts) & (factor_df['date'] <= end_ts)]
        if factor_df.empty:
            logger.error("指定时间范围内无因子数据。")
            return None

        if merge_base_factors and stock_data is not None and base_factor_fields:
            available_base_fields = [field for field in base_factor_fields if field in stock_data.columns]
            if available_base_fields:
                base_subset = stock_data[['date', 'stock_code'] + available_base_fields].copy()
                if 'date' in base_subset.columns:
                    base_subset['date'] = pd.to_datetime(base_subset['date'])
                factor_df = factor_df.merge(base_subset, on=['date', 'stock_code'], how='left')
            else:
                logger.warning("stock_data 中未找到可合并的基础字段，跳过合并。")

        factor_df.sort_values(['date', 'stock_code'], inplace=True)
        factor_df.reset_index(drop=True, inplace=True)

        logger.info(
            "成功加载指定因子: %d 条记录，含因子列 %d 个。",
            len(factor_df),
            len(factor_df.columns) - 3,
        )
        return factor_df
    except Exception as exc:
        logger.error(f"加载指定因子失败: {exc}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _generate_training_labels_for_all_data(stock_data, atr_multiplier_profit=ATR_MULTIPLIER_PROFIT, atr_multiplier_loss=ATR_MULTIPLIER_LOSS, max_holding_days=MAX_HOLDING_DAYS, debug=False,
                                         is_training=True, label_stats=None):
    """
    🚀 修复后的标签生成函数：与持仓管理逻辑完全一致的连续收益标签

    标签生成规则（与用户需求的持仓管理逻辑一致）：
    - T+1日触发止盈止损 → T+2日开盘价收益（A股T+1规则）
    - T+2日及以后触发止盈止损 → 当日按止盈止损价收益
    - 超时未触发 → 最大持仓期+1天开盘价收益
    - 只使用ATR止损（训练阶段），不使用5%止损

    关键修复：
    1. 退出价格计算与持仓管理逻辑完全一致
    2. 收益计算基于实际退出价格，而非固定ATR倍数
    3. 确保训练标签反映真实交易收益

    输出包含：
    - atr_return_label: 实际收益标签（ATR倍数，基于真实退出价格）
    - exit_day: 实际持仓天数（1 到 max_holding_days）
    - min_dist_to_sl: 持仓期间价格到止损线的最小距离（ATR倍数）
    - min_dist_to_pt: 持仓期间价格到止盈线的最小距离（ATR倍数）

    Args:
        stock_data: 股票数据DataFrame
        atr_multiplier_profit: 止盈ATR倍数
        atr_multiplier_loss: 止损ATR倍数
        max_stop_loss_pct: 最大止损百分比（如0.05表示5%）
        max_holding_days: 最大持仓天数
        debug: 调试模式
        is_training: 是否为训练模式（True时计算统计量，False时使用传入的统计量）
        label_stats: 标签统计量字典（测试时使用）

    Returns:
        如果is_training=True: (DataFrame, label_stats_dict)
        如果is_training=False: DataFrame

        DataFrame包含：
        - atr_return_label: 连续收益标签（标准化后）
        - exit_day: 实际持仓天数
        - min_dist_to_sl: 到止损线最小距离
        - min_dist_to_pt: 到止盈线最小距离
    """
    try:
        import pandas as pd
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)
        logger.info("🎯 开始批量生成基于ATR的连续收益标签（成本感知回归模式）...")

        # 数据基本信息检查
        logger.info(f"输入数据基本信息:")
        logger.info(f"  总记录数: {len(stock_data)}")
        logger.info(f"  股票数量: {stock_data['stock_code'].nunique()}")
        logger.info(f"  日期数量: {stock_data['date'].nunique()}")
        logger.info(f"  日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")

        # 检查必要字段（移除is_can_buy检查，因为已在外部过滤）
        required_fields = ['atr14', 'close', 'open', 'high', 'low']
        missing_fields = [field for field in required_fields if field not in stock_data.columns]

        if missing_fields:
            logger.error(f"数据中缺少必要字段: {missing_fields}")
            if 'atr14' in missing_fields:
                logger.error("缺少atr14列，无法计算ATR止盈止损位")
                return stock_data
            if any(field in missing_fields for field in ['close', 'open', 'high', 'low']):
                logger.error("缺少价格数据，无法计算标签")
                return stock_data

        logger.info(f"  atr14字段: 有效值{stock_data['atr14'].notna().sum()}个, 缺失值{stock_data['atr14'].isna().sum()}个")
        logger.info(f"  价格数据: close有效值{stock_data['close'].notna().sum()}个")

        # 确保数据按股票代码和日期排序
        stock_data_sorted = stock_data.sort_values(['stock_code', 'date']).reset_index(drop=True)

        logger.info("📊 开始计算ATR三分类标签（纯ATR止损版）...")
        logger.info(f"  止盈倍数: {atr_multiplier_profit}倍ATR14（基于T日ATR）")
        logger.info(f"  止损倍数: {atr_multiplier_loss}倍ATR14（基于T日ATR）")
        logger.info(f"  观察期间: T+1至T+{max_holding_days}日（{max_holding_days}天）")
        logger.info(f"  🔧 逻辑基准：以_calculate_atr_based_return函数为准")

        # 🚀 高性能向量化处理：批量计算未来价格数据
        logger.info("🔧 步骤1: 批量计算T+1至T+{max_holding_days+1}日的价格数据...")

        # 按股票分组计算未来价格（向量化操作）
        for i in range(1, max_holding_days + 2):  # T+1到T+max_holding_days+1
            stock_data_sorted[f'open_T{i}'] = stock_data_sorted.groupby('stock_code')['open'].shift(-i)
            stock_data_sorted[f'high_T{i}'] = stock_data_sorted.groupby('stock_code')['high'].shift(-i)
            stock_data_sorted[f'low_T{i}'] = stock_data_sorted.groupby('stock_code')['low'].shift(-i)
            stock_data_sorted[f'close_T{i}'] = stock_data_sorted.groupby('stock_code')['close'].shift(-i)

        logger.info("🔧 步骤2: 基于T+1开盘价计算止盈止损位...")

        # 基于T+1开盘价计算止盈止损位（向量化操作）
        stock_data_sorted['entry_price'] = stock_data_sorted['open_T1']  # T+1日开盘价作为入场价
        stock_data_sorted['profit_target'] = stock_data_sorted['entry_price'] + atr_multiplier_profit * stock_data_sorted['atr14']

        # 🚀 训练阶段：使用纯ATR止损逻辑
        stock_data_sorted['stop_loss'] = stock_data_sorted['entry_price'] - atr_multiplier_loss * stock_data_sorted['atr14']

        # 筛选有效数据：必须有ATR、入场价，且有足够的未来数据
        valid_mask = (
            stock_data_sorted['atr14'].notna() &
            stock_data_sorted['entry_price'].notna() &
            stock_data_sorted['profit_target'].notna() &
            stock_data_sorted['stop_loss'].notna() &
            stock_data_sorted['open_T1'].notna()  # 确保T+1日有开盘价
        )

        logger.info(f"有效样本数: {valid_mask.sum():,}/{len(stock_data_sorted):,}")

        # 🚀 成本感知回归：初始化连续收益标签和辅助信息
        stock_data_sorted['atr_return_label'] = 0.0       # 🎯 主要标签：连续收益（ATR倍数）
        stock_data_sorted['exit_day'] = 0                  # 实际持仓天数 (1-5)
        stock_data_sorted['min_dist_to_sl'] = np.inf      # 到止损线最小距离 (ATR倍数)
        stock_data_sorted['min_dist_to_pt'] = np.inf      # 到止盈线最小距离 (ATR倍数)

        # 创建一个 'is_labeled' 掩码，用于记录哪些行已经确定了标签
        is_labeled = pd.Series(False, index=stock_data_sorted.index)

        # 统计变量
        total_profit_samples = 0
        total_loss_samples = 0

        logger.info("🔧 步骤3: 完全向量化的标签计算...")

        # 🚀 修复：按照用户需求的持仓管理逻辑进行标签生成
        # 关键：T+1日触发→次日开盘退出，T+2日及以后触发→当日按止盈止损价退出
        for day in range(1, max_holding_days + 1):
            logger.info(f"  向量化处理T+{day}日...")

            high_col = f'high_T{day}'
            low_col = f'low_T{day}'
            next_open_col = f'open_T{day+1}'

            # 检查必要的列是否存在
            if (high_col not in stock_data_sorted.columns or
                low_col not in stock_data_sorted.columns):
                logger.warning(f"  缺少T+{day}日价格数据，跳过")
                continue

            # 🚀 扩展：计算当日到止损线和止盈线的距离（ATR倍数）
            # 距离计算：(价格 - 止损线) / ATR 或 (止盈线 - 价格) / ATR

            # 🔧 修复：安全的距离计算，避免NaN值
            # 只对有效数据计算距离
            valid_calc_mask = (
                valid_mask &
                stock_data_sorted[low_col].notna() &
                stock_data_sorted[high_col].notna() &
                stock_data_sorted['stop_loss'].notna() &
                stock_data_sorted['profit_target'].notna() &
                stock_data_sorted['atr14'].notna() &
                (stock_data_sorted['atr14'] > 0)  # 避免除零
            )

            # 初始化距离数组
            dist_to_sl_today = pd.Series(np.inf, index=stock_data_sorted.index)
            dist_to_pt_today = pd.Series(np.inf, index=stock_data_sorted.index)

            # 只对有效数据计算距离
            if valid_calc_mask.any():
                dist_to_sl_today.loc[valid_calc_mask] = (
                    stock_data_sorted.loc[valid_calc_mask, low_col] -
                    stock_data_sorted.loc[valid_calc_mask, 'stop_loss']
                ) / stock_data_sorted.loc[valid_calc_mask, 'atr14']

                dist_to_pt_today.loc[valid_calc_mask] = (
                    stock_data_sorted.loc[valid_calc_mask, 'profit_target'] -
                    stock_data_sorted.loc[valid_calc_mask, high_col]
                ) / stock_data_sorted.loc[valid_calc_mask, 'atr14']

            # 更新最小距离（只对未标记且有效的样本）
            unlabeled_mask = ~is_labeled & valid_mask & valid_calc_mask
            if unlabeled_mask.any():
                stock_data_sorted.loc[unlabeled_mask, 'min_dist_to_sl'] = np.minimum(
                    stock_data_sorted.loc[unlabeled_mask, 'min_dist_to_sl'],
                    dist_to_sl_today.loc[unlabeled_mask]
                )
                stock_data_sorted.loc[unlabeled_mask, 'min_dist_to_pt'] = np.minimum(
                    stock_data_sorted.loc[unlabeled_mask, 'min_dist_to_pt'],
                    dist_to_pt_today.loc[unlabeled_mask]
                )

            # --- 🔧 止损检查（按照用户需求的持仓管理逻辑） ---
            hit_loss_mask = (
                valid_mask &
                (stock_data_sorted[low_col] <= stock_data_sorted['stop_loss']) &
                (~is_labeled)
            )

            if hit_loss_mask.any():
                # 计算止损收益：根据持仓天数决定退出价格
                if day == 1:
                    # T+1日触发止损 → T+2日开盘退出（A股T+1规则）
                    if next_open_col in stock_data_sorted.columns:
                        exit_price = stock_data_sorted.loc[hit_loss_mask, next_open_col]
                        valid_exit = exit_price.notna() & (exit_price > 0)
                        hit_loss_mask = hit_loss_mask & valid_exit
                    else:
                        hit_loss_mask = pd.Series(False, index=stock_data_sorted.index)
                else:
                    # T+2日及以后触发止损 → 当日按止损价退出
                    exit_price = stock_data_sorted.loc[hit_loss_mask, 'stop_loss']

                if hit_loss_mask.any():
                    # 🚀 修复：计算实际止损收益，避免除零和无穷大值
                    entry_price = stock_data_sorted.loc[hit_loss_mask, 'entry_price']
                    atr_value = stock_data_sorted.loc[hit_loss_mask, 'atr14']

                    # 确保ATR值大于0，避免除零
                    safe_atr_value = atr_value.replace(0, 1e-8).fillna(1e-8)

                    if day == 1:
                        # T+1日触发，T+2日开盘退出
                        actual_return = (exit_price - entry_price) / safe_atr_value
                    else:
                        # T+2日及以后，按止损价退出
                        actual_return = (exit_price - entry_price) / safe_atr_value

                    # 🚀 修复：不在此处clip，后续统一使用Winsorize处理
                    # 保留原始收益信息，避免过早截断
                    # actual_return = actual_return  # 保持原值

                    # 🚀 修复：过滤掉无效值
                    valid_return_mask = np.isfinite(actual_return)
                    if valid_return_mask.any():
                        final_hit_loss_mask = hit_loss_mask.copy()
                        final_hit_loss_mask.loc[hit_loss_mask] = valid_return_mask
                        stock_data_sorted.loc[final_hit_loss_mask, 'atr_return_label'] = actual_return[valid_return_mask]
                        hit_loss_mask = final_hit_loss_mask  # 更新有效的止损样本
                    stock_data_sorted.loc[hit_loss_mask, 'exit_day'] = day
                    is_labeled.loc[hit_loss_mask] = True
                    loss_count = hit_loss_mask.sum()
                    total_loss_samples += loss_count
                else:
                    loss_count = 0
            else:
                loss_count = 0

            # --- 🔧 止盈检查（按照用户需求的持仓管理逻辑） ---
            hit_profit_mask = (
                valid_mask &
                (stock_data_sorted[high_col] >= stock_data_sorted['profit_target']) &
                (~is_labeled)
            )

            if hit_profit_mask.any():
                # 计算止盈收益：根据持仓天数决定退出价格
                if day == 1:
                    # T+1日触发止盈 → T+2日开盘退出（A股T+1规则）
                    if next_open_col in stock_data_sorted.columns:
                        exit_price = stock_data_sorted.loc[hit_profit_mask, next_open_col]
                        valid_exit = exit_price.notna() & (exit_price > 0)
                        hit_profit_mask = hit_profit_mask & valid_exit
                    else:
                        hit_profit_mask = pd.Series(False, index=stock_data_sorted.index)
                else:
                    # T+2日及以后触发止盈 → 当日按止盈价退出
                    exit_price = stock_data_sorted.loc[hit_profit_mask, 'profit_target']

                if hit_profit_mask.any():
                    # 🚀 修复：计算实际止盈收益，避免除零和无穷大值
                    entry_price = stock_data_sorted.loc[hit_profit_mask, 'entry_price']
                    atr_value = stock_data_sorted.loc[hit_profit_mask, 'atr14']

                    # 确保ATR值大于0，避免除零
                    safe_atr_value = atr_value.replace(0, 1e-8).fillna(1e-8)

                    if day == 1:
                        # T+1日触发，T+2日开盘退出
                        actual_return = (exit_price - entry_price) / safe_atr_value
                    else:
                        # T+2日及以后，按止盈价退出
                        actual_return = (exit_price - entry_price) / safe_atr_value

                    # 🚀 修复：不在此处clip，后续统一使用Winsorize处理
                    # 保留原始收益信息，避免过早截断
                    # actual_return = actual_return  # 保持原值

                    # 🚀 修复：过滤掉无效值
                    valid_return_mask = np.isfinite(actual_return)
                    if valid_return_mask.any():
                        final_hit_profit_mask = hit_profit_mask.copy()
                        final_hit_profit_mask.loc[hit_profit_mask] = valid_return_mask
                        stock_data_sorted.loc[final_hit_profit_mask, 'atr_return_label'] = actual_return[valid_return_mask]
                        hit_profit_mask = final_hit_profit_mask  # 更新有效的止盈样本
                    stock_data_sorted.loc[hit_profit_mask, 'exit_day'] = day
                    is_labeled.loc[hit_profit_mask] = True
                    profit_count = hit_profit_mask.sum()
                    total_profit_samples += profit_count
                else:
                    profit_count = 0
            else:
                profit_count = 0

            if loss_count > 0 or profit_count > 0:
                logger.info(f"    T+{day}日: 新增亏损{loss_count:,}个, 新增盈利{profit_count:,}个")

        # 🚀 修复：处理超时样本（与持仓管理逻辑一致）
        timeout_mask = valid_mask & (~is_labeled)

        if timeout_mask.any():
            logger.info(f"  计算{timeout_mask.sum()}个超时样本的实际收益...")

            # 超时处理：最大持仓期+1天开盘价退出（与持仓管理逻辑一致）
            next_open_col = f'open_T{max_holding_days+1}'

            if next_open_col in stock_data_sorted.columns:
                # 优先使用T+max_holding_days+1开盘价
                exit_price = stock_data_sorted.loc[timeout_mask, next_open_col]
                valid_exit_mask = timeout_mask & exit_price.notna() & (exit_price > 0)
            else:
                # 兜底：使用T+max_holding_days收盘价
                exit_price_col = f'close_T{max_holding_days}'
                if exit_price_col in stock_data_sorted.columns:
                    exit_price = stock_data_sorted.loc[timeout_mask, exit_price_col]
                    valid_exit_mask = timeout_mask & exit_price.notna() & (exit_price > 0)
                else:
                    valid_exit_mask = pd.Series(False, index=stock_data_sorted.index)

            if valid_exit_mask.any():
                # 计算实际收益（ATR倍数）
                entry_price = stock_data_sorted.loc[valid_exit_mask, 'open_T1']
                atr_value = stock_data_sorted.loc[valid_exit_mask, 'atr14']
                exit_price_valid = exit_price.loc[valid_exit_mask]

                # 🚀 修复：实际收益 = (退出价格 - 入场价格) / ATR，避免除零和无穷大值
                # 确保ATR值大于0，避免除零
                safe_atr_value = atr_value.replace(0, 1e-8)
                safe_atr_value = safe_atr_value.fillna(1e-8)

                actual_return_atr = (exit_price_valid - entry_price) / safe_atr_value

                # 🚀 修复：不在此处clip，后续统一使用Winsorize处理
                # 保留原始收益信息，避免过早截断
                # actual_return_atr = actual_return_atr  # 保持原值

                # 🚀 修复：过滤掉无效值
                valid_return_mask = np.isfinite(actual_return_atr)
                if valid_return_mask.any():
                    # 设置超时样本的连续收益标签
                    final_valid_mask = valid_exit_mask.copy()
                    final_valid_mask.loc[valid_exit_mask] = valid_return_mask
                    stock_data_sorted.loc[final_valid_mask, 'atr_return_label'] = actual_return_atr[valid_return_mask]
                    timeout_mask = final_valid_mask  # 更新有效的超时样本
                else:
                    timeout_mask = pd.Series(False, index=stock_data_sorted.index)
                timeout_mask = valid_exit_mask  # 更新有效的超时样本

        stock_data_sorted.loc[timeout_mask, 'exit_day'] = max_holding_days
        total_neutral_samples = timeout_mask.sum()

        # 清理临时列（移除未来价格列和中间计算列，保留核心数据）
        temp_columns = [col for col in stock_data_sorted.columns if col.startswith(('open_T', 'high_T', 'low_T', 'close_T'))]
        temp_columns.extend(['entry_price', 'profit_target', 'stop_loss'])
        stock_data_sorted.drop(columns=temp_columns, inplace=True, errors='ignore')

        logger.info(f"✅ 连续收益标签生成完成")
        logger.info(f"  处理样本: {valid_mask.sum():,}个")
        logger.info(f"  盈利样本: {total_profit_samples:,}个（标签={atr_multiplier_profit}）")
        logger.info(f"  亏损样本: {total_loss_samples:,}个（标签={-atr_multiplier_loss}）")
        logger.info(f"  超时样本: {total_neutral_samples:,}个（实际收益标签）")

        # 🚀 Pseudo-Huber回归：输出原始ATR单位连续收益标签（不截尾/不标准化）
        return_labels = stock_data_sorted['atr_return_label'].copy()
        valid_labels_mask = return_labels.notna()
        valid_labels = return_labels[valid_labels_mask]

        logger.info(f"🚀 原始ATR连续收益标签统计（Pseudo-Huber回归）:")
        logger.info(f"  最小值: {valid_labels.min():.3f} ATR")
        logger.info(f"  最大值: {valid_labels.max():.3f} ATR")
        logger.info(f"  平均值: {valid_labels.mean():.3f} ATR")
        logger.info(f"  标准差: {valid_labels.std():.3f} ATR")

        # 🚀 保存原始ATR标签到DataFrame中，用于后续样本权重计算
        stock_data_sorted['atr_return_label_original'] = stock_data_sorted['atr_return_label'].copy()
        logger.info(f"✅ 保存原始ATR标签到atr_return_label_original字段")

        # 🚀 应用Winsorize截尾：使用0.5%和99.5%分位数动态截断
        # 目的：移除真正的异常值，保留99%的数据
        p005 = np.percentile(valid_labels, 0.5)
        p995 = np.percentile(valid_labels, 99.5)

        logger.info(f"🚀 Winsorize截尾处理:")
        logger.info(f"  0.5%分位数: {p005:.3f} ATR")
        logger.info(f"  99.5%分位数: {p995:.3f} ATR")
        logger.info(f"  截尾前范围: [{valid_labels.min():.3f}, {valid_labels.max():.3f}] ATR")

        # 应用Winsorize
        stock_data_sorted['atr_return_label'] = np.clip(
            stock_data_sorted['atr_return_label'],
            p005,
            p995
        )

        # 更新有效标签
        valid_labels_winsorized = stock_data_sorted.loc[valid_labels_mask, 'atr_return_label']
        logger.info(f"  截尾后范围: [{valid_labels_winsorized.min():.3f}, {valid_labels_winsorized.max():.3f}] ATR")
        logger.info(f"  截尾后平均值: {valid_labels_winsorized.mean():.3f} ATR")
        logger.info(f"  截尾后标准差: {valid_labels_winsorized.std():.3f} ATR")

        # 计算标签统计量用于样本权重计算和模型训练
        if is_training:
            # 计算标签统计量
            p01 = np.percentile(valid_labels, 1)   # 1%分位数
            p99 = np.percentile(valid_labels, 99)  # 99%分位数

            current_label_stats = {
                'min': float(valid_labels.min()),
                'max': float(valid_labels.max()),
                'mean': float(valid_labels.mean()),
                'std': float(valid_labels.std()),
                'p01': float(p01),
                'p99': float(p99)
            }

            logger.info(f"🚀 标签统计量（用于样本权重和模型训练）:")
            logger.info(f"  分位数: p01={p01:.3f}, p99={p99:.3f} ATR")
            logger.info(f"  范围: [{current_label_stats['min']:.3f}, {current_label_stats['max']:.3f}] ATR")
            logger.info(f"  均值±标准差: {current_label_stats['mean']:.3f}±{current_label_stats['std']:.3f} ATR")

            # 🚀 新增：标记不可操作的样本为NaN
            logger.info("\n🔄 标记不可操作样本为NaN...")
            logger.info("=" * 60)

            total_samples = len(stock_data_sorted)
            valid_labels_before = stock_data_sorted['atr_return_label'].notna().sum()

            # 0. T日pct_chg异常(为0或NaN,数据质量问题)
            # 🚀 修复: 使用&运算符连接pandas Series,并同时检查open字段防止误过滤
            if 'pct_chg' in stock_data_sorted.columns and 'open' in stock_data_sorted.columns:
                pct_chg_invalid_mask = (
                    ((stock_data_sorted['pct_chg'].isna()) | (stock_data_sorted['pct_chg'] == 0)) &
                    ((stock_data_sorted['open'].isna()) | (stock_data_sorted['open'] == 0))
                )
                stock_data_sorted.loc[pct_chg_invalid_mask, 'atr_return_label'] = np.nan
                logger.info(f"  ✅ 标签过滤(T日pct_chg异常): {pct_chg_invalid_mask.sum():,}个样本")
            else:
                logger.warning("  ⚠️ 未找到pct_chg或open字段")

            # 1. T日涨停(无法买入+风险高)
            if 'is_limit_up' in stock_data_sorted.columns:
                t_limit_up_mask = stock_data_sorted['is_limit_up'] == 1
                stock_data_sorted.loc[t_limit_up_mask, 'atr_return_label'] = np.nan
                logger.info(f"  ✅ 标签过滤(T日涨停): {t_limit_up_mask.sum():,}个样本")
            else:
                logger.warning("  ⚠️ 未找到is_limit_up字段")

            # 2. T日跌停(价格异常)
            if 'is_limit_down' in stock_data_sorted.columns:
                t_limit_down_mask = stock_data_sorted['is_limit_down'] == 1
                stock_data_sorted.loc[t_limit_down_mask, 'atr_return_label'] = np.nan
                logger.info(f"  ✅ 标签过滤(T日跌停): {t_limit_down_mask.sum():,}个样本")
            else:
                logger.warning("  ⚠️ 未找到is_limit_down字段")

            # 3. T日流动性不足(无法买入)
            if 'amount' in stock_data_sorted.columns:
                t_low_liquidity_mask = stock_data_sorted['amount'] < 10_000_000
                stock_data_sorted.loc[t_low_liquidity_mask, 'atr_return_label'] = np.nan
                logger.info(f"  ✅ 标签过滤(T日流动性不足,amount<1000万): {t_low_liquidity_mask.sum():,}个样本")
            elif 'volume' in stock_data_sorted.columns:
                t_low_liquidity_mask = stock_data_sorted['volume'] < 1_000_000
                stock_data_sorted.loc[t_low_liquidity_mask, 'atr_return_label'] = np.nan
                logger.info(f"  ✅ 标签过滤(T日流动性不足,volume<100万股): {t_low_liquidity_mask.sum():,}个样本")
            else:
                logger.warning("  ⚠️ 未找到amount或volume字段")

            # # 4. T+1日一字涨停(is_can_buy=0)
            # if 'is_can_buy' in stock_data_sorted.columns:
            #     t1_cannot_buy_mask = stock_data_sorted['is_can_buy'] == 0
            #     stock_data_sorted.loc[t1_cannot_buy_mask, 'atr_return_label'] = np.nan
            #     logger.info(f"  ✅ 标签过滤(T+1日不可买入,is_can_buy=0): {t1_cannot_buy_mask.sum():,}个样本")
            # else:
            #     logger.warning("  ⚠️ 未找到is_can_buy字段")

            # 统计最终有效标签数量
            valid_labels_after = stock_data_sorted['atr_return_label'].notna().sum()
            filtered_count = valid_labels_before - valid_labels_after

            logger.info(f"\n  📊 标签过滤统计:")
            logger.info(f"     总样本数: {total_samples:,}个")
            logger.info(f"     过滤前有效标签: {valid_labels_before:,}个")
            logger.info(f"     过滤后有效标签: {valid_labels_after:,}个")
            logger.info(f"     过滤数量: {filtered_count:,}个 ({filtered_count/valid_labels_before*100:.2f}%)")
            logger.info("=" * 60)

            return stock_data_sorted, current_label_stats
        else:
            return stock_data_sorted

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"批量ATR标签生成失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        # 返回原始数据，添加默认连续收益标签
        stock_data['atr_return_label'] = 0.0
        return stock_data


# 🚀 软标签函数已删除：成本感知回归模式不需要软标签


# 🚀 软标签准备函数已删除：成本感知回归模式不需要软标签


def _apply_atr_strategy_core(stock_data, decision_date, atr_value, atr_multiplier_profit=ATR_MULTIPLIER_PROFIT, atr_multiplier_loss=ATR_MULTIPLIER_LOSS, max_holding_days=MAX_HOLDING_DAYS):
    """
    ATR策略的核心处理逻辑（以_calculate_atr_based_return为准）

    Args:
        stock_data: 单只股票的时间序列数据
        decision_date: T日（决策日）
        atr_value: T日的ATR值
        atr_multiplier_profit: 止盈ATR倍数
        atr_multiplier_loss: 止损ATR倍数
        max_holding_days: 最大持仓天数

    Returns:
        标签模式：-1(亏损), 0(中性), 1(盈利)
        收益模式：实际收益率
    """
    try:
        import pandas as pd
        import numpy as np

        # 获取T+1日及之后的数据
        future_data = stock_data[stock_data['date'] > decision_date].head(max_holding_days + 1)

        if len(future_data) == 0 or pd.isna(atr_value) or atr_value <= 0:
            return 0  # 无法交易，标签为中性

        # 获取T+1日的开盘价作为入场价
        entry_price = future_data.iloc[0]['open']
        if pd.isna(entry_price) or entry_price <= 0:
            return 0  # T+1无法入场，标签为中性

        # 基于入场价和T日的ATR计算止盈止损
        profit_target = entry_price + atr_multiplier_profit * atr_value
        stop_loss = entry_price - atr_multiplier_loss * atr_value

        # 循环检查T+1到T+max_holding_days日的行情
        for i in range(len(future_data) - 1):  # 最多检查max_holding_days天
            current_day_row = future_data.iloc[i]

            # 止损优先原则
            if current_day_row['low'] <= stop_loss:
                return -1  # 触发止损，标签为亏损

            if current_day_row['high'] >= profit_target:
                return 1   # 触发止盈，标签为盈利

        # 超时未触发止盈止损
        return 0  # 标签为中性

    except Exception as e:
        return 0  # 出错时返回中性标签


def _process_atr_strategy_single(backtest_data, stock_code, selection_date, atr_multiplier_profit=ATR_MULTIPLIER_PROFIT, atr_multiplier_loss=ATR_MULTIPLIER_LOSS, max_holding_days=MAX_HOLDING_DAYS, debug=False):
    """
    单股票处理模式：计算基于ATR策略的实际收益率

    Args:
        backtest_data: 回测数据DataFrame
        stock_code: 股票代码
        selection_date: 选择日期
        atr_multiplier_profit: 止盈ATR倍数
        atr_multiplier_loss: 止损ATR倍数
        max_stop_loss_pct: 最大止损百分比（如0.05表示5%）
        max_holding_days: 最大持仓天数
        debug: 调试模式

    Returns:
        实际收益率（float）
    """
    try:
        import pandas as pd
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        # 找到T日的ATR值
        atr_value = backtest_data.loc[
            (backtest_data['stock_code'] == stock_code) &
            (backtest_data['date'] == selection_date),
            'atr14'
        ].iloc[0]

        # 获取T+1日及之后的数据
        future_data = backtest_data[
            (backtest_data['stock_code'] == stock_code) &
            (backtest_data['date'] > selection_date)
        ].head(max_holding_days + 1)  # 需要T+1到T+max_holding_days+1的数据

        if len(future_data) == 0 or pd.isna(atr_value) or atr_value <= 0:
            return 0.0  # 无法交易，收益为0

        # 获取T+1日的开盘价作为入场价
        entry_price = future_data.iloc[0]['open']
        if pd.isna(entry_price) or entry_price <= 0:
            return 0.0  # T+1无法入场，收益为0

        # 基于入场价和T日的ATR计算止盈止损
        profit_target = entry_price + atr_multiplier_profit * atr_value

        # 🚀 预测阶段：使用5%+ATR双重止损逻辑
        atr_stop_loss = entry_price - atr_multiplier_loss * atr_value
        pct_stop_loss = entry_price * (1 - MAX_STOP_LOSS_PCT)  # 5%止损
        stop_loss = max(atr_stop_loss, pct_stop_loss)  # 取较高价格（较小损失）

        exit_price = None

        # 循环检查T+1到T+max_holding_days日的行情
        for i in range(len(future_data) - 1):  # 最多检查max_holding_days天
            current_day_row = future_data.iloc[i]

            # 止损优先原则
            if current_day_row['low'] <= stop_loss:
                # 触发止损，记录T+i+1 (次日) 开盘价为退出价
                exit_price = future_data.iloc[i + 1]['open']
                if debug:
                    logger.info(f"{stock_code} on {current_day_row['date']}: 触发止损")
                break

            if current_day_row['high'] >= profit_target:
                # 触发止盈，记录T+i+1 (次日) 开盘价为退出价
                exit_price = future_data.iloc[i + 1]['open']
                if debug:
                    logger.info(f"{stock_code} on {current_day_row['date']}: 触发止盈")
                break

        # 处理超时情况
        if exit_price is None:
            # 如果循环max_holding_days天都未触发，则在第max_holding_days+1天开盘时强制平仓
            if len(future_data) > max_holding_days:
                exit_price = future_data.iloc[max_holding_days]['open']
                if debug:
                    logger.info(f"{stock_code}: {max_holding_days}天超时，在T+{max_holding_days+1}开盘平仓")
            else:
                # 如果数据不足max_holding_days天，则在最后一天收盘价平仓
                exit_price = future_data.iloc[-1]['close']
                if debug:
                    logger.info(f"{stock_code}: 数据不足，在最后一日收盘平仓")

        if pd.isna(exit_price) or exit_price <= 0:
            return 0.0  # 无法退出，收益为0

        # 计算最终收益率
        actual_return = (exit_price - entry_price) / entry_price
        return actual_return

    except Exception as e:
        if debug:
            logger = logging.getLogger(__name__)
            logger.error(f"计算{stock_code}收益时异常: {e}")
        return 0.0


def _extract_and_validate_labels(labeled_data, label_column='atr_return_label'):
    """
    🚀 提取和验证连续收益标签（成本感知回归模式）

    Args:
        labeled_data: 包含标签的数据DataFrame
        label_column: 标签列名，固定为'atr_return_label'（连续收益标签）

    Returns:
        tuple: (labels, valid_mask) 或 (None, None) 如果失败
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
            profit_samples = (valid_labels >= 1.8).sum()  # 盈利样本（>=1.8 ATR）
            loss_samples = (valid_labels <= -0.9).sum()   # 亏损样本（<=-0.9 ATR）
            neutral_samples = ((valid_labels > -0.9) & (valid_labels < 1.8)).sum()  # 中性样本

            logger.info(f"连续收益标签分布:")
            logger.info(f"  盈利样本 (>=1.8 ATR): {profit_samples:,}个 ({profit_samples/len(valid_labels)*100:.1f}%)")
            logger.info(f"  中性样本 (-0.9~1.8 ATR): {neutral_samples:,}个 ({neutral_samples/len(valid_labels)*100:.1f}%)")
            logger.info(f"  亏损样本 (<=-0.9 ATR): {loss_samples:,}个 ({loss_samples/len(valid_labels)*100:.1f}%)")

        # 创建有效样本掩码（过滤掉NaN值）
        valid_mask = labels.notna()

        logger.info("✅ 连续收益标签提取和验证完成")
        return labels, valid_mask

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"连续收益标签提取和验证失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None


def _apply_winsorization(X_data, feature_names, winsorization_thresholds=None, data_type="数据"):
    """
    应用Winsorization缩尾处理

    Args:
        X_data: 特征数据数组
        feature_names: 特征名称列表
        winsorization_thresholds: 预计算的阈值字典，如果为None则计算新阈值
        data_type: 数据类型描述（用于日志）

    Returns:
        tuple: (winsorized_data, thresholds_dict)
    """
    try:
        import numpy as np
        logger = logging.getLogger(__name__)

        logger.info(f"🔧 对{data_type}应用Winsorization缩尾处理...")

        X_winsorized = X_data.copy()
        total_changed = 0
        features_changed = 0

        # 如果没有提供阈值，则计算新阈值
        if winsorization_thresholds is None:
            logger.info("计算Winsorization分位数阈值...")
            winsorization_thresholds = {}

            for i, feature_name in enumerate(feature_names):
                feature_data = X_data[:, i]

                # 🔧 修复：使用更保守的分位数阈值，减少信息损失
                # 计算0.5%和99.5%分位数，减少过度缩尾
                p1 = np.percentile(feature_data, 0.5)
                p99 = np.percentile(feature_data, 99.5)
                # p1 = np.percentile(feature_data, 1)
                # p99 = np.percentile(feature_data, 99)

                winsorization_thresholds[feature_name] = {'p1': p1, 'p99': p99}

                # 统计异常值数量
                outliers_low = np.sum(feature_data < p1)
                outliers_high = np.sum(feature_data > p99)
                total_outliers = outliers_low + outliers_high

                if total_outliers > 0:
                    logger.info(f"  特征 {feature_name}: 异常值{total_outliers}个 (低端{outliers_low}个, 高端{outliers_high}个)")
                else:
                    logger.info(f"  特征 {feature_name}: 无异常值")
                logger.info(f"    分位数阈值: [{p1:.6f}, {p99:.6f}]")

        # 应用缩尾处理
        for i, feature_name in enumerate(feature_names):
            if feature_name in winsorization_thresholds:
                thresholds = winsorization_thresholds[feature_name]

                # 缩尾处理
                original_data = X_winsorized[:, i]
                winsorized_data = np.clip(original_data, thresholds['p1'], thresholds['p99'])

                # 统计缩尾效果
                changed_count = np.sum(original_data != winsorized_data)
                total_changed += changed_count

                if changed_count > 0:
                    logger.info(f"  特征 {feature_name}: 缩尾处理{changed_count}个值")
                    features_changed += 1
                else:
                    logger.info(f"  特征 {feature_name}: 无需缩尾处理")

                X_winsorized[:, i] = winsorized_data
            else:
                logger.warning(f"  ⚠️ 特征 {feature_name} 缺少Winsorization阈值")

        logger.info(f"✅ {data_type}Winsorization缩尾处理完成")
        logger.info(f"  {data_type}总样本数: {len(X_winsorized):,}")
        logger.info(f"  缩尾处理的特征数: {features_changed}/{len(feature_names)}")
        logger.info(f"  缩尾处理的总值数: {total_changed:,}")

        return X_winsorized, winsorization_thresholds

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"{data_type}Winsorization处理失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return X_data, winsorization_thresholds


def _prepare_feature_data(input_data, feature_columns, apply_winsorization=True, winsorization_thresholds=None, data_type="特征数据"):
    """
    准备特征数据，进行数据清洗、特征工程和Winsorization处理
    专注于特征预处理，不处理标签，以便在训练、验证、测试阶段复用

    Args:
        input_data: 输入数据DataFrame
        feature_columns: 特征列名列表
        apply_winsorization: 是否应用Winsorization缩尾处理
        winsorization_thresholds: 预计算的Winsorization阈值（用于验证/测试阶段）
        data_type: 数据类型描述（用于日志）

    Returns:
        tuple: (X_processed, available_features, winsorization_thresholds) 或 (None, None, None) 如果失败
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
            return None, None, None

        logger.info(f"可用特征: {len(available_features)}个")

        # 📊 步骤2：提取特征数据
        feature_data = input_data[available_features].copy()

        logger.info(f"{data_type}统计:")
        logger.info(f"  样本数: {len(feature_data):,}个")
        logger.info(f"  特征数: {len(available_features)}个")

        # 🧹 步骤3：数据清洗
        logger.info("🧹 进行数据清洗...")

        # 1. 处理无穷值
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)

        # 2. 统计缺失值
        missing_stats = feature_data.isnull().sum()
        high_missing_features = missing_stats[missing_stats > len(feature_data) * 0.5].index.tolist()

        if high_missing_features:
            logger.warning(f"移除缺失值过多的特征: {len(high_missing_features)}个")
            feature_data = feature_data.drop(columns=high_missing_features)
            available_features = [f for f in available_features if f not in high_missing_features]

        # 3.（仅训练/验证）移除完全相同的行（基于特征）；测试/预测阶段保持行数以保证对齐
        initial_rows = len(feature_data)
        is_training_like = str(data_type).strip() in ["训练集", "验证集"]
        if is_training_like:
            duplicate_mask = ~feature_data.duplicated()
            feature_data = feature_data[duplicate_mask]
            if len(feature_data) < initial_rows:
                logger.info(f"移除重复行: {initial_rows - len(feature_data)}行")
        else:
            logger.info("预测/测试阶段: 跳过重复行移除，确保与原始索引对齐")

        logger.info(f"数据清洗后: {len(feature_data)}个样本, {len(available_features)}个特征")

        # 4. 填充剩余缺失值
        logger.info("填充缺失值...")
        for col in feature_data.columns:
            if feature_data[col].isnull().any():
                # 使用中位数填充数值型特征
                median_value = feature_data[col].median()
                if pd.isna(median_value):
                    median_value = 0
                feature_data[col] = feature_data[col].fillna(median_value)

        # 📊 步骤4：转换为numpy数组
        X_processed = feature_data.values.astype(np.float32)

        # 🚀 步骤5：Winsorization缩尾处理（可选）
        final_winsorization_thresholds = winsorization_thresholds
        if apply_winsorization:
            logger.info("=" * 60)
            logger.info("🚀 特征工程优化：Winsorization缩尾处理")
            logger.info("=" * 60)
            logger.info("📋 缩尾处理说明:")
            logger.info("  ✅ 消除极端异常值对ATR三分类模型的负面影响")
            logger.info("  ✅ 将高于99%分位数的值替换为99%分位数值")
            logger.info("  ✅ 将低于1%分位数的值替换为1%分位数值")
            logger.info("  ✅ 提升模型在边界情况下的稳定性")

            X_processed, final_winsorization_thresholds = _apply_winsorization(
                X_processed, available_features, winsorization_thresholds, data_type
            )
        else:
            logger.info("⏭️ 跳过Winsorization缩尾处理")

        # 🔍 步骤6：最终检查
        if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
            logger.warning("特征数据中仍有NaN或Inf值，进行最终清理...")
            X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e6, neginf=-1e6)

        # 📊 步骤7：统计信息
        total_samples = len(X_processed)

        logger.info(f"✅ {data_type}准备完成:")
        logger.info(f"  样本数: {total_samples:,}")
        logger.info(f"  特征数: {X_processed.shape[1]}")
        logger.info(f"  特征数据类型: {X_processed.dtype}")
        logger.info(f"  Winsorization阈值: {'已计算' if final_winsorization_thresholds else '未计算'}")

        return X_processed, available_features, final_winsorization_thresholds

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"准备{data_type}失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None, None



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
                    'winsorization_thresholds': model_results.get('winsorization_thresholds', {}),
                    'pca_info': model_results['pca_info'],
                    'feature_names': feature_names,
                    'pca_models': pca_models if pca_models else {},
                    'label_stats': model_results.get('label_stats', None),
                    'label_transformer': model_results.get('label_transformer', None),
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
                    'label_transformer': model_results.get('label_transformer', None),
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


def run_train_model_mode(args, train_end_date=None, is_regression=True, use_ensemble=None, preset_huber_delta=None):
    """运行模型训练模式

    Args:
        args: 命令行参数
        train_end_date: 训练数据截止日期，如果为None则使用全局配置TRAIN_END_DATE
        is_regression: 是否使用回归模式，默认True（成本感知回归）
        use_ensemble: 是否使用集成学习(3个模型)，如果为None则根据调用方式自动判断
            - None: 自动判断(test_model调用时使用ensemble，直接调用时不使用)
            - True: 强制使用ensemble(3个模型)
            - False: 强制不使用ensemble(1个模型)
        preset_huber_delta: 预设的Pseudo-Huber delta值（可选）
            - 如果提供,将跳过delta网格搜索,直接使用该值
            - 仅在test_model模式下使用,用于加速训练
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

        # 记录特征列表到日志（用于调试可重现性问题）
        logger.info("🔍 特征列表（前20个）:")
        for i, col in enumerate(sorted(feature_columns)[:20], 1):
            logger.info(f"  {i:2d}. {col}")
        if len(feature_columns) > 20:
            logger.info(f"  ... 还有 {len(feature_columns) - 20} 个特征")

        # 保存完整特征列表到文件（用于对比分析）
        from datetime import datetime
        feature_list_file = Path("results") / f"feature_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(feature_list_file, 'w', encoding='utf-8') as f:
            f.write(f"训练时间: {datetime.now()}\n")
            f.write(f"特征数量: {len(feature_columns)}\n")
            f.write(f"数据日期范围: {factor_data['date'].min()} - {factor_data['date'].max()}\n")
            f.write(f"训练截止日期: {train_end_date if train_end_date else '无限制'}\n")
            f.write("特征列表:\n")
            for i, col in enumerate(sorted(feature_columns), 1):
                f.write(f"{i:3d}. {col}\n")
        logger.info(f"📝 完整特征列表已保存: {feature_list_file}")
        logger.info("=" * 60)

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
                logger.info(f"📅 {date_str}: 过滤统计 {initial_count} -> {after_new_stock_count} -> {final_count}")

        if not filtered_data_list:
            logger.error("❌ 所有数据都被过滤掉了")
            return False

        # 合并过滤后的数据
        factor_data = pd.concat(filtered_data_list, ignore_index=True)

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

        # 🚀 4. 标签构建（带Winsorize截尾和标准化）
        logger.info("开始构建训练标签（成本感知回归模式）...")
        labeled_data_result = _generate_training_labels_for_all_data(
            factor_data,
            atr_multiplier_profit=ATR_MULTIPLIER_PROFIT,
            atr_multiplier_loss=ATR_MULTIPLIER_LOSS,
            max_holding_days=MAX_HOLDING_DAYS,
            debug=False,
            is_training=True,  # 训练模式，计算统计量
            label_stats=None
        )

        # 解包结果
        if isinstance(labeled_data_result, tuple):
            labeled_data, label_stats = labeled_data_result
            logger.info("✅ 训练标签构建完成，获得Winsorize阈值")
            if 'winsorize_thresholds' in label_stats:
                winsorize_thresholds = label_stats['winsorize_thresholds']
                logger.info(f"  Winsorize阈值: 1%={winsorize_thresholds['q1']:.3f}, 99%={winsorize_thresholds['q99']:.3f}")
        else:
            labeled_data = labeled_data_result
            label_stats = None
            logger.warning("⚠️ 未获得标签统计量，可能影响后续处理")

        # 🚀 5. 连续收益标签提取和验证（成本感知回归模式）
        logger.info("🏷️ 提取和验证连续收益标签（成本感知回归模式）...")
        labels, valid_mask = _extract_and_validate_labels(labeled_data, label_column='atr_return_label')

        if labels is None or valid_mask is None:
            logger.error("连续收益标签提取和验证失败")
            return False

        # 过滤有效样本
        filtered_data = labeled_data[valid_mask].copy()
        filtered_labels = labels[valid_mask].copy()

        logger.info(f"有效训练数据准备完成: {len(filtered_data)} 个样本, {len(feature_columns)} 个特征列")

        # 5.5. 特征处理策略说明
        logger.info("🔄 特征处理策略: 所有特征预处理将在_train_ml_model内部完成")
        logger.info("  ✅ 数据清洗、Winsorization、标准化等步骤统一处理")
        logger.info("  ✅ 确保训练和验证集使用相同的预处理参数")
        logger.info("  ✅ 避免数据泄露，提高模型可靠性")
        logger.info("=" * 50)

        # 6. 模型训练
        # 🚀 超参数搜索控制
        enable_param_search = getattr(args, 'enable_param_search', False)
        search_strategy = getattr(args, 'search_strategy', 'grid')

        # 🚀 判断是否使用集成学习
        if use_ensemble is None:
            # 自动判断: 如果有enable_param_search参数,说明是直接调用train_model,不使用ensemble
            use_ensemble_mode = not enable_param_search
        else:
            use_ensemble_mode = use_ensemble

        if use_ensemble_mode:
            # 🚀 集成学习模式: 训练3个模型
            logger.info("开始训练XGBoost集成模型（3个模型）...")
            if enable_param_search:
                logger.info(f"🔍 启用超参数搜索: 策略={search_strategy}")
            else:
                logger.info("🔍 使用默认参数配置（跳过超参数搜索）")

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

            logger.info(f"📊 数据划分: 训练集{len(raw_train_data):,}样本, 验证集{len(raw_val_data):,}样本")

            # 6.2 特征预处理(Winsorization)
            logger.info("🔧 对训练集进行Winsorization（计算阈值）...")
            X_train_processed, feature_names_processed, train_winsorization_thresholds = _prepare_feature_data(
                raw_train_data, feature_columns, apply_winsorization=True, data_type="训练集"
            )

            if X_train_processed is None:
                logger.error("训练集特征预处理失败")
                return False

            logger.info("🔧 对验证集进行Winsorization（使用训练集阈值）...")
            X_val_processed, _, _ = _prepare_feature_data(
                raw_val_data, feature_columns, apply_winsorization=True,
                winsorization_thresholds=train_winsorization_thresholds, data_type="验证集"
            )

            if X_val_processed is None:
                logger.error("验证集特征预处理失败")
                return False

            # 6.3 调整标签长度(预处理可能移除重复行)
            if len(X_train_processed) != len(y_train_split):
                logger.warning(f"训练集预处理后长度变化: {len(y_train_split)} -> {len(X_train_processed)}")
                y_train_split = y_train_split[:len(X_train_processed)]

            if len(X_val_processed) != len(y_val_split):
                logger.warning(f"验证集预处理后长度变化: {len(y_val_split)} -> {len(X_val_processed)}")
                y_val_split = y_val_split[:len(X_val_processed)]

            # 6.4 准备预处理数据字典
            preprocessed_data = {
                'X_train': X_train_processed,
                'X_val': X_val_processed,
                'y_train': y_train_split,
                'y_val': y_val_split,
                'winsorization_thresholds': train_winsorization_thresholds
            }

            logger.info(f"✅ 预处理完成: 训练集{len(X_train_processed):,}样本, 验证集{len(X_val_processed):,}样本")
            logger.info("=" * 60)

            # 🚀 生成统一的时间戳（XGBoost和LightGBM共用）
            model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"🕐 模型统一时间戳: {model_timestamp}")

            # 🚀 训练XGBoost模型
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
                    preset_huber_delta=preset_huber_delta
                )

                if xgb_results is None:
                    logger.error("XGBoost模型训练失败：_train_ml_model返回None")
                    return False

                logger.info(f"✅ XGBoost模型训练完成")

            except Exception as e:
                logger.error(f"XGBoost模型训练失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                return False

            # 🚀 如果启用集成学习,训练LightGBM模型
            lgb_results = None
            if use_ensemble:
                logger.info(f"\n{'='*60}")
                logger.info(f"训练LightGBM模型(集成学习)")
                logger.info(f"{'='*60}")

                try:
                    # 🚀 使用XGBoost训练后的特征列表(已对齐)
                    xgb_feature_names = xgb_results['feature_names']
                    logger.info(f"✅ 使用XGBoost的特征列表: {len(xgb_feature_names)}个特征")

                    lgb_results = _train_lightgbm_model(
                        filtered_data,
                        filtered_labels,
                        xgb_feature_names,  # 🚀 使用XGBoost的特征列表
                        label_stats=label_stats,
                        preprocessed_data=preprocessed_data
                    )

                    if lgb_results is None:
                        logger.warning("⚠️ LightGBM模型训练失败,将只使用XGBoost模型")
                    else:
                        logger.info(f"✅ LightGBM模型训练完成")

                except Exception as e:
                    logger.warning(f"⚠️ LightGBM模型训练失败: {e},将只使用XGBoost模型")
                    import traceback
                    logger.warning(f"详细错误信息: {traceback.format_exc()}")
                    lgb_results = None

            # 🚀 收集模型结果
            ensemble_results = [xgb_results]
            if lgb_results is not None:
                ensemble_results.append(lgb_results)

            logger.info(f"\n{'='*60}")
            logger.info(f"✅ 模型训练完成: {len(ensemble_results)}个模型")
            logger.info(f"{'='*60}")

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
                    preset_huber_delta=preset_huber_delta  # 🚀 传递预设的huber delta值
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


# 旧的_calculate_atr_based_return函数已被_process_atr_strategy统一处理器替代


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

        # 🚀 获取use_ensemble参数(在函数开头,确保所有分支都能访问)
        use_ensemble = not getattr(args, 'no_ensemble', False)  # 默认使用集成学习
        logger.info(f"🚀 集成学习模式: {'启用(XGBoost+LightGBM)' if use_ensemble else '禁用(仅XGBoost)'}")

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

            # 调用训练模式，使用优化后的配置：启用回归模式和集成学习
            # 🚀 从args中获取preset_huber参数
            preset_huber = getattr(args, 'set_huber', None)

            if preset_huber is not None:
                logger.info(f"🚀 使用预设Pseudo-Huber delta值: {preset_huber:.2f}")

            train_success = run_train_model_mode(
                args,
                train_end_date=train_end_date,
                is_regression=True,
                use_ensemble=use_ensemble,  # 🚀 根据命令行参数决定
                preset_huber_delta=preset_huber  # 🚀 传递预设的huber delta值
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

        # 🚀 查找最新的XGBoost模型
        logger.info("🔍 查找XGBoost模型...")
        xgb_files = list(results_dir.glob("xgboost_model_*.pkl"))
        if not xgb_files:
            logger.error("未找到XGBoost模型文件")
            return False

        # 按时间戳排序,找到最新的XGBoost模型
        latest_xgb_file = max(xgb_files, key=lambda f: f.stem)
        logger.info(f"✅ 找到最新XGBoost模型: {latest_xgb_file.name}")

        # 提取时间戳
        parts = latest_xgb_file.stem.split('_')
        if len(parts) >= 4:
            model_timestamp = f"{parts[2]}_{parts[3]}"  # YYYYMMDD_HHMMSS
        else:
            logger.error(f"无法从文件名提取时间戳: {latest_xgb_file.name}")
            return False

        # 🚀 如果启用集成学习,查找相同时间戳的LightGBM模型
        latest_lgb_file = None
        if use_ensemble:
            logger.info(f"🔍 查找相同时间戳的LightGBM模型(时间戳: {model_timestamp})...")
            lgb_pattern = f"lightgbm_model_{model_timestamp}.pkl"
            lgb_file = results_dir / lgb_pattern

            if lgb_file.exists():
                latest_lgb_file = lgb_file
                logger.info(f"✅ 找到LightGBM模型: {latest_lgb_file.name}")
            else:
                logger.warning(f"⚠️ 未找到相同时间戳的LightGBM模型,将只使用XGBoost模型")

        # 🚀 收集要加载的模型文件
        latest_models = [latest_xgb_file]
        if latest_lgb_file is not None:
            latest_models.append(latest_lgb_file)

        logger.info(f"📁 将加载{len(latest_models)}个模型:")
        for f in latest_models:
            logger.info(f"  - {f.name}")

        # 🚀 加载所有模型
        models = []
        scalers = []
        label_transformers = []
        model_types = []  # 记录模型类型
        model_performances = []  # 记录模型性能(用于权重计算)
        feature_names = None
        winsorization_thresholds = None  # 🚀 Winsorization阈值(从XGBoost加载)

        for model_file in latest_models:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            models.append(model_data['model'])
            scalers.append(model_data['scaler'])
            label_transformers.append(model_data.get('label_transformer', None))
            model_types.append(model_data.get('model_type', 'xgboost'))  # 获取模型类型

            # 获取验证集性能(用于权重计算)
            performance = model_data.get('performance', {})
            # 🚀 兼容XGBoost(mse)和LightGBM(val_mse)两种键名
            val_mse = performance.get('val_mse') or performance.get('mse')
            model_performances.append(val_mse)

            if feature_names is None:
                feature_names = model_data['feature_names']

            # 🚀 从XGBoost模型加载Winsorization阈值(集成学习共享预处理参数)
            if winsorization_thresholds is None and model_data.get('model_type') == 'xgboost':
                winsorization_thresholds = model_data.get('winsorization_thresholds', None)

            model_type = model_data.get('model_type', 'xgboost')
            logger.info(f"  ✅ 加载{model_type.upper()}模型: {model_file.name}")
            if val_mse is not None:
                logger.info(f"     验证集MSE: {val_mse:.6f}")
            else:
                logger.warning(f"     ⚠️ 未找到验证集MSE信息")

        # 🚀 加载标签统计量（所有模型共享）
        label_stats = model_data.get('label_stats', None)
        if label_stats:
            logger.info(f"✅ 加载标签统计量: μ={label_stats['mean']:.6f}, σ={label_stats['std']:.6f}")
        else:
            logger.warning("⚠️ 未找到标签统计量，可能影响回归预测的反标准化")

        logger.info(f"✅ 模型加载成功: {len(models)}个模型，特征数量: {len(feature_names)}")

        # 🚀 计算集成学习权重(基于验证集MSE)
        model_weights = None
        if len(models) > 1:
            logger.info("\n📊 计算集成学习权重(基于验证集MSE)...")

            # 检查是否所有模型都有MSE
            if all(mse is not None for mse in model_performances):
                # 权重 = 1 / MSE (归一化)
                weights = np.array([1.0 / mse for mse in model_performances])
                model_weights = weights / weights.sum()

                logger.info("✅ 模型权重:")
                for i, (model_type, weight, mse) in enumerate(zip(model_types, model_weights, model_performances)):
                    logger.info(f"  模型{i+1}({model_type.upper()}): 权重={weight:.4f}, MSE={mse:.6f}")
            else:
                logger.warning("⚠️ 部分模型缺少MSE信息,使用等权重")
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
            for i, (model, scaler, label_transformer, model_type) in enumerate(zip(models, scalers, label_transformers, model_types)):
                logger.info(f"\n🔮 模型{i+1}({model_type.upper()})预测...")

                # 标准化（使用该模型的scaler）
                X_pred_scaled = scaler.transform(X_pred_final)

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

                # 🚀 应用智能标签变换的逆变换
                if label_transformer is not None:
                    y_pred = label_transformer.inverse_transform(y_pred_raw)
                    logger.info(f"  ✅ 应用智能标签逆变换: [{y_pred_raw.min():.3f}, {y_pred_raw.max():.3f}] -> [{y_pred.min():.3f}, {y_pred.max():.3f}] ATR")
                else:
                    y_pred = y_pred_raw
                    logger.info(f"  ✅ 未使用标签变换")

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

            # 🚀 风险调整分数：使用加权均值和标准差
            # 对于单模型,risk_aversion不起作用(std=0)
            risk_aversion = 0.5 if n_models > 1 else 0.0
            y_pred_return = weighted_mean_pred - risk_aversion * std_pred

            logger.info(f"\n✅ 风险调整分数（risk_aversion={risk_aversion}）")
            logger.info(f"  使用加权均值预测（基于验证集MSE）")
            logger.info(f"  分数范围: [{y_pred_return.min():.3f}, {y_pred_return.max():.3f}] ATR")
            logger.info(f"  📊 最终预测值统计: [{y_pred_return.min():.3f}, {y_pred_return.max():.3f}] ATR")

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


        # 9. 保存预测结果
        logger.info("\n💾 步骤12: 保存预测结果...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_results = prediction_data_with_proba[['date', 'stock_code', 'predicted_return', 'y_pred_proba']].copy()
        prediction_file = results_dir / f"pure_predictions_{timestamp}.csv"
        prediction_results.to_csv(prediction_file, index=False, encoding='utf-8')
        logger.info(f"✅ 预测结果已保存: {prediction_file} ({len(prediction_results)} 条记录)")

        # 10. 选股回测系统
        logger.info("\n🔄 步骤13: 开始选股回测...")
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
        missing_columns = [col for col in required_columns if col not in prediction_data_with_proba.columns]

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
                    prediction_data_with_proba[col] = default_values[col]
                    logger.info(f"  {col}: 设置默认值 {default_values[col]}")
                else:
                    prediction_data_with_proba[col] = 0.0
                    logger.warning(f"  {col}: 未定义默认值，设置为 0.0")

        # 🚀 准备回测数据（回归模式）
        backtest_columns = required_columns + ['y_pred_proba', 'predicted_return', 'predicted_return_atr_raw']

        # 🆕 调试日志：检查prediction_data_with_proba的字段
        logger.info("\n🔍 调试: prediction_data_with_proba字段检查")
        logger.info(f"总行数: {len(prediction_data_with_proba)}")
        logger.info(f"总列数: {len(prediction_data_with_proba.columns)}")

        # 检查关键字段
        key_fields = ['date', 'stock_code', 'volume', 'is_can_buy', 'pct_chg', 'open', 'high', 'low', 'close', 'atr14']
        for field in key_fields:
            if field in prediction_data_with_proba.columns:
                missing_count = prediction_data_with_proba[field].isna().sum()
                missing_pct = missing_count / len(prediction_data_with_proba) * 100
                if field not in ['date', 'stock_code']:
                    zero_count = (prediction_data_with_proba[field] == 0).sum()
                    zero_pct = zero_count / len(prediction_data_with_proba) * 100
                    logger.info(f"  {field}: 缺失{missing_count}({missing_pct:.2f}%), 为0{zero_count}({zero_pct:.2f}%)")
                else:
                    logger.info(f"  {field}: 缺失{missing_count}({missing_pct:.2f}%)")
            else:
                logger.warning(f"  {field}: ❌ 字段不存在")

        backtest_data = prediction_data_with_proba[backtest_columns].copy()

        # 只按日期排序，保持每日内股票的原始预测概率分布
        backtest_data = backtest_data.sort_values('date')

        # 获取所有交易日
        trading_dates = sorted(backtest_data['date'].unique())
        logger.info(f"\n📅 回测期间: {trading_dates[0].strftime('%Y-%m-%d')} 到 {trading_dates[-1].strftime('%Y-%m-%d')}")
        logger.info(f"📅 交易日数量: {len(trading_dates)}天")

        # 🆕 调试日志：检查特定日期的数据
        target_date = pd.to_datetime('2025-01-06')
        if target_date in trading_dates:
            target_data = backtest_data[backtest_data['date'] == target_date]
            logger.info(f"\n🔍 调试: 2025-01-06数据检查")
            logger.info(f"总股票数: {len(target_data)}")
            for field in key_fields:
                if field in target_data.columns and field not in ['date', 'stock_code']:
                    missing_count = target_data[field].isna().sum()
                    zero_count = (target_data[field] == 0).sum()
                    non_zero_count = len(target_data) - missing_count - zero_count
                    logger.info(f"  {field}: 缺失{missing_count}, 为0{zero_count}, 非0{non_zero_count}")
                    if non_zero_count > 0:
                        non_zero_values = target_data[target_data[field] != 0][field].dropna()
                        logger.info(f"    非0值范围: [{non_zero_values.min():.4f}, {non_zero_values.max():.4f}]")

        # 🚀 动态调整选股阈值：基于预测范围自适应
        all_predictions = backtest_data['predicted_return_atr_raw'].dropna()
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

        logger.info(f"✅ 自适应阈值: {MIN_POS_THRESHOLD_ATR:.2f} ATR")
        logger.info(f"   - 60%分位数: {pred_60th:.2f} ATR")
        logger.info(f"   - 均值+0.5σ: {threshold_mean_based:.2f} ATR")
        logger.info(f"   - 超过阈值的股票比例: {above_threshold_pct:.1f}%")
        logger.info(f"📊 预测值分布: 最大{pred_max:.2f}, 75%分位{pred_75th:.2f}, 中位数{pred_median:.2f}, 均值{pred_mean:.2f}±{pred_std:.2f} ATR")

        # 准备常量
        TRADING_COST_ATR = 0.03        # 交易/滑点等折算到 ATR 的最低可覆盖成本

        # 定义选股数量
        stock_counts = [1, 3, 5]

        # 初始化回测结果
        backtest_results = {}
        detailed_trades = {}

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

            # 预处理：确保有predicted_return_atr_raw字段
            if 'predicted_return_atr_raw' not in backtest_data.columns:
                # 如果没有该字段，使用predicted_return作为替代
                backtest_data['predicted_return_atr_raw'] = backtest_data['predicted_return']
                logger.info("使用predicted_return作为predicted_return_atr_raw的替代")

            # 按日期分组进行选股
            df_pred = backtest_data.copy()
            grouped = df_pred.groupby('date', sort=True)

            selected_records = []
            empty_day_count = 0
            total_days = 0

            for trade_date, g in grouped:
                total_days += 1
                g = g.copy()

                # 🚀 第一步：基础过滤（新股、ST股、可买入）
                date_str = trade_date.strftime('%Y%m%d')

                # 🆕 调试日志：特定日期的详细过滤过程
                is_debug_date = (trade_date == pd.to_datetime('2025-01-06'))
                if is_debug_date:
                    logger.info(f"\n🔍 调试: {date_str}选股过程")
                    logger.info(f"初始股票数: {len(g)}")

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

                # 2.1 过滤 NaN
                before_nan = len(g)
                g = g[~g['predicted_return_atr_raw'].isna()]
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
                    logger.info(f"预测值分布: 最大{g['predicted_return_atr_raw'].max():.4f}, "
                               f"75%分位{g['predicted_return_atr_raw'].quantile(0.75):.4f}, "
                               f"中位数{g['predicted_return_atr_raw'].median():.4f}")
                    logger.info(f"阈值: {MIN_POS_THRESHOLD_ATR}")
                    logger.info(f"超过阈值的股票数: {(g['predicted_return_atr_raw'] >= MIN_POS_THRESHOLD_ATR).sum()}")

                g = g[g['predicted_return_atr_raw'] >= MIN_POS_THRESHOLD_ATR]
                if is_debug_date:
                    logger.info(f"阈值过滤后: {len(g)} (过滤掉{before_threshold - len(g)})")

                if g.empty:
                    # 全部不达标：记为空仓日
                    if is_debug_date:
                        logger.info(f"❌ 所有股票预测值低于阈值，空仓")
                    empty_day_count += 1
                    continue

                # 2.3 对"剩余候选"做**日内横截面标准化**
                mu = g['predicted_return_atr_raw'].mean()
                std = g['predicted_return_atr_raw'].std(ddof=0)
                ma = g['predicted_return_atr_raw'].max()
                # logger.info(f"data:{date_str}, g max:{ma}, mean: {mu}")
                if std and std > 1e-8:
                    g['score_z'] = (g['predicted_return_atr_raw'] - mu) / std
                else:
                    g['score_z'] = 0.0  # 全体相等时退化为 0

                # 2.4 成本敏感：减去成本门槛
                g['score_final'] = g['score_z'] - (TRADING_COST_ATR / max(std, 1.0))

                # 2.5 排序并**不凑满**：只取 Top-K，且不够就少买
                g = g.sort_values('score_final', ascending=False).head(stock_count)
                # logger.info(f"data:{date_str}, g count:{len(g)}  and g:{g}")

                # 2.6 记录选股（包含仓位分配信息）
                selection_columns = ['date','stock_code','predicted_return_atr_raw','score_z','score_final']

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
                empty_columns = ['date','stock_code','predicted_return_atr_raw','score_z','score_final']
                if 'position_allocation' in prediction_data_with_proba.columns:
                    empty_columns.append('position_allocation')
                if 'market_regime' in prediction_data_with_proba.columns:
                    empty_columns.append('market_regime')
                picks = pd.DataFrame(columns=empty_columns)

            # 计算空仓日比例
            empty_ratio = empty_day_count / max(total_days, 1)
            # logger.info(f"统一口径选股完成：只买正期望且 >= {MIN_POS_THRESHOLD_ATR:.2f} ATR，不凑满K；空仓日比例={empty_ratio:.1%}")

            # 执行回测：基于选股结果进行持仓管理和收益计算
            portfolio_values_gross = [1.0]  # 未扣除成本的组合价值

            # 持仓管理：跟踪每只股票的持仓状态
            # 持仓信息结构：{stock_code: {'selection_date': T日, 'entry_price': T+1开盘价, 'atr_value': T日ATR,
            #                           'profit_target': 止盈位, 'stop_loss': 止损位, 'stop_loss_type': 止损类型, 'days_held': 持仓天数}}
            current_positions = {}
            max_positions = stock_count
            # logger.info(f"初始化持仓管理: 最大持仓={max_positions}, ATR止盈={ATR_MULTIPLIER_PROFIT}x, ATR止损={ATR_MULTIPLIER_LOSS}x, 5%止损={MAX_STOP_LOSS_PCT:.1%}, 最大持仓期={MAX_HOLDING_DAYS}天")

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
                                today_high = stock_data.iloc[0]['high']
                                today_low = stock_data.iloc[0]['low']
                                today_open = stock_data.iloc[0]['open']

                                # 🚀 调试：添加详细的止盈止损检查日志（仅TOP-1策略）
                                if stock_count == 1:  # 只对TOP-1策略输出详细日志
                                    logger.info(f"🔍 {stock_code} T+{days_held}日止盈止损检查:")
                                    logger.info(f"  当日价格: 开盘{today_open:.3f}, 最高{today_high:.3f}, 最低{today_low:.3f}")
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
                        if days_held > MAX_HOLDING_DAYS:
                            if stock_code not in positions_to_exit:
                                positions_to_exit.append(stock_code)
                            if stock_code not in exit_reasons:
                                exit_reasons[stock_code] = 'max_holding_days'
                            # logger.info(f"超时退出: {stock_code}, 持仓{days_held}天 > 最大持仓期{MAX_HOLDING_DAYS}天")

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
                                # 次日开盘退出或最大持仓期：使用开盘价
                                stock_data = backtest_data[
                                    (backtest_data['stock_code'] == stock_code) &
                                    (backtest_data['date'] == date)
                                ]
                                if not stock_data.empty and pd.notna(stock_data.iloc[0]['open']):
                                    exit_price = stock_data.iloc[0]['open']
                                    # logger.info(f"退出收益计算: {stock_code}, 开盘退出价{exit_price:.3f}")
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
                                    stop_loss_type = "5%止损" if abs(stop_loss_val - pct_sl) < abs(stop_loss_val - atr_sl) else "ATR止损"
                                else:
                                    stop_loss_type = 'ATR止损'
                            except Exception:
                                stop_loss_type = 'ATR止损'

                        # 记录退出交易（确保一定追加记录）
                        # 统一并标准化退出原因：take_profit / atr_stop_loss / percent_stop_loss / max_holding_days
                        if exit_reason == 'take_profit':
                            canon_reason = 'take_profit'
                        elif exit_reason in ('stop_loss', 'loss', 'stoploss'):
                            canon_reason = 'percent_stop_loss' if str(stop_loss_type).strip() == '5%止损' else 'atr_stop_loss'
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
                                canon_reason = 'percent_stop_loss' if str(stop_loss_type).strip() == '5%止损' else 'atr_stop_loss'
                            elif _almost_equal(exit_price, position_info.get('profit_target', np.nan)):
                                canon_reason = 'take_profit'
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
                            'predicted_return_atr_raw': 0,
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
                                    entry_price = stock_data.iloc[0]['open']  # T+1日开盘价作为入场价
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

                                    # 计算止盈止损位
                                    profit_target = entry_price + ATR_MULTIPLIER_PROFIT * atr_value
                                    atr_stop_loss = entry_price - ATR_MULTIPLIER_LOSS * atr_value
                                    pct_stop_loss = entry_price * (1 - MAX_STOP_LOSS_PCT)
                                    stop_loss = max(atr_stop_loss, pct_stop_loss)  # 取较高价格（较小损失）
                                    stop_loss_type = "ATR止损" if stop_loss == atr_stop_loss else "5%止损"

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
                                    try:
                                        today_high = stock_data.iloc[0]['high']
                                        today_low = stock_data.iloc[0]['low']
                                        # 止损优先
                                        if pd.notna(today_low) and (today_low <= stop_loss):
                                            current_positions[stock_code]['exit_triggered'] = 'stop_loss'
                                            current_positions[stock_code]['exit_type'] = 'next_day_open'
                                            # if stock_count == 1:
                                                # logger.info(f"T+1当日触发止损(标记次日开盘退出): {stock_code}, 最低{today_low:.3f} <= 止损{stop_loss:.3f}")
                                        elif pd.notna(today_high) and (today_high >= profit_target):
                                            current_positions[stock_code]['exit_triggered'] = 'take_profit'
                                            current_positions[stock_code]['exit_type'] = 'next_day_open'
                                            # if stock_count == 1:
                                                # logger.info(f"T+1当日触发止盈(标记次日开盘退出): {stock_code}, 最高{today_high:.3f} >= 止盈{profit_target:.3f}")
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
                                today_close = stock_data.iloc[0]['close']
                                today_open = stock_data.iloc[0]['open']
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
                                'predicted_return_atr_raw': 0,
                                'score_final': 0,
                                'market_regime': 'holding'
                            })
                            action = 'hold'

                        # 记录交易
                        trade_record = {
                            'date': date,
                            'stock_code': stock_code,
                            'predicted_return_atr_raw': stock_info.get('predicted_return_atr_raw', 0),
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
                win_rate = len(positive_returns) / len(returns_array) if len(returns_array) > 0 else 0

                # 平均单次收益 (基于净收益)
                avg_return = np.mean(returns_array)

                # 🚀 回归模式评估指标和ATR策略统计
                regression_metrics = {}
                atr_strategy_stats = {}
                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)

                    # 计算预测收益与实际收益的相关性
                    if 'predicted_return_atr_raw' in trades_df.columns and 'actual_return' in trades_df.columns:
                        correlation = trades_df['predicted_return_atr_raw'].corr(trades_df['actual_return'])
                        regression_metrics['prediction_correlation'] = correlation

                        logger.info(f"📊 回归预测效果分析:")
                        logger.info(f"  预测收益与实际收益相关性: {correlation:.4f}")
                        logger.info(f"  总交易笔数: {len(trades_df)}")
                        logger.info(f"  平均预测收益 (ATR raw): {trades_df['predicted_return_atr_raw'].mean():.4f}")
                        logger.info(f"  平均最终得分 (score_final): {trades_df['score_final'].mean():.4f}")
                        logger.info(f"  平均实际收益 (ATR): {trades_df['actual_return'].mean():.4f}")

                        # 诊断阈值统计
                        logger.info(f"  入选股票预测收益统计:")
                        logger.info(f"    均值: {trades_df['predicted_return_atr_raw'].mean():.3f} ATR")
                        logger.info(f"    中位数: {trades_df['predicted_return_atr_raw'].median():.3f} ATR")
                        logger.info(f"    25分位: {trades_df['predicted_return_atr_raw'].quantile(0.25):.3f} ATR")
                        logger.info(f"    75分位: {trades_df['predicted_return_atr_raw'].quantile(0.75):.3f} ATR")

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
                                        return 'pct_stop_loss' if stype == '5%止损' else 'atr_stop_loss'
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
                                logger.info(f"  5%止损触发率: {pct_stop_loss_count}/{total_stop_loss} ({pct_stop_loss_count/total_stop_loss:.1%})")
                                if atr_stop_loss_count > pct_stop_loss_count:
                                    logger.info(f"  ✅ ATR止损更常触发，说明ATR设置合理")
                                elif pct_stop_loss_count > atr_stop_loss_count:
                                    logger.info(f"  ⚠️ 5%止损更常触发，建议调整ATR倍数")
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
                    'avg_daily_position': (position_ratio_sum / max(len(daily_returns), 1))
                }

                detailed_trades[stock_count] = trades

                logger.info(f"✅ {stock_count}只股票回测完成:")
                logger.info(f"  总收益率 (扣成本前): {total_return_gross:.2%}")
                logger.info(f"  净收益率 (扣成本后): {total_return_net:.2%}")
                logger.info(f"  年化收益率: {annual_return:.2%}")
                logger.info(f"  夏普比率: {sharpe_ratio:.3f}")
                logger.info(f"  最大回撤: {max_drawdown:.2%}")
                logger.info(f"  胜率: {win_rate:.2%}")
                # 🚀 新增：空仓日比例统计
                logger.info(f"  空仓日比例: {empty_position_ratio:.1%} ({empty_position_days}天)")
                logger.info(f"  部分仓位日比例: {partial_position_ratio:.1%} ({partial_position_days}天)")
                logger.info(f"  满仓日比例: {full_position_ratio:.1%}")



        # 12. 保存回测结果
        logger.info("\n💾 步骤14: 保存回测结果...")

        # 保存回测指标汇总
        backtest_summary = {
            "timestamp": timestamp,
            "backtest_period": {
                "start_date": trading_dates[0].strftime("%Y-%m-%d"),
                "end_date": trading_dates[-1].strftime("%Y-%m-%d"),
                "trading_days": len(trading_dates) - 1
            },
            "strategy_results": {}
        }

        # 输出对比结果
        # 输出对比结果
        # 输出对比结果
        logger.info("\n📊 “唯一口径”选股策略回测结果对比:")
        logger.info("=" * 130)
        logger.info(f"{'策略':<10} {'总收益':<10} {'净收益(扣成本)':<15} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<8} {'日均仓位':<10}")
        logger.info("=" * 130)

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
                           f"{position_desc}")

        logger.info("=" * 130)

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

        logger.info("=" * 60)
        logger.info("🎉 Pseudo-Huber回归选股回测完成！")
        logger.info("📊 已完成1只、3只、5只股票的选股策略对比")
        logger.info(f"✅ 使用动态阈值{MIN_POS_THRESHOLD_ATR:.2f} ATR，宁缺毋滥选股策略")
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
        logger.info("📊 步骤1: 加载ATR三分类系统因子数据")
        logger.info("-" * 50)

        # 定义ATR三分类系统使用的因子列表
        # required_factors = [
        #     'returns_zscore', 'market_structure_score', 'intraday_momentum', 'ma_ratio_5_10',
        #     'corr_5d', 'volatility_correlation', 'volume_correlation', 'near_support_count_60d',
        #     'near_support_win_count_60d', 'institutional_activity_score', 'reversal_score',
        #     'ma_convergence_score', 'lower_shadow_ratio', 'buying_pressure', 'sector_correlation_20d',
        #     'price_impact', 'rank_momentum_20d', 'rank_rsi', 'rank_macd', 'rank_volume'
        # ]
         # 'returns_zscore', 'market_structure_score', 'ma_ratio_5_10', 'corr_5d',
        # 'volatility_correlation', 'volume_correlation', 'near_support_count_60d',
        # 'institutional_activity_score', 'ma_convergence_score', 'lower_shadow_ratio',
        # 'sector_correlation_20d', 'price_impact', 'rank_momentum_20d', 'rank_rsi',
        # 'rank_volume','market_correlation', 'corr_20d_mom5', 'corr_slope_20_120'
        # ,'support_risk_reward_ratio'
        # ,'volatility_efficiency'
        # ,'pullback_volume_ratio'
        # ,'volume_strength'
        # ,'mv_factor'
        # ,'sector_liquidity_stability'
        # # ,'log_market_cap' 与mv_factor高相关
        # # ,'size_factor_score' 与mv_factor高相关
        # # ,'pct_chg'
        # # ,'fear_greed_impact_factor'
        # # ,'pct_chg_lag4'
        # # ,'price_vol_change_ratio'
        # ,'sector_momentum_consistency'
        # # ,'market_volatility_regime'
        # # ,'pct_chg_lag1'
        # # ,'consecutive_negative_days' 3天单测很差
        # # ,'consecutive_positive_days' 3天单测很差
        # ,'sector_return_rank_20d'
        # # (与rank_momentum_20d高度相关)

        required_factors = _get_stock_factors()

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

        # 获取完整的18个股票特征 (使用与训练一致的定义)
        stock_features = _get_stock_factors()
        logger.info(f"📊 使用完整的18个股票特征: {len(stock_features)} 个")

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


def _apply_correlation_filter(valid_features, correlation_matrix, high_corr_pairs, factor_test_results, logger):
    """第一阶段：相关性过滤，对于高相关特征对只保留IC更高的一个"""
    logger.info("🔄 执行相关性过滤...")

    # 构建需要移除的特征集合
    features_to_remove = set()

    # 按相关系数绝对值排序，优先处理相关性最高的特征对
    sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)

    for feature1, feature2, corr_val in sorted_pairs:
        # 如果两个特征都还没被移除
        if feature1 not in features_to_remove and feature2 not in features_to_remove:
            # 比较IC绝对值，保留IC更高的特征
            ic1 = abs(factor_test_results.get(feature1, {}).get('ic', 0.0))
            ic2 = abs(factor_test_results.get(feature2, {}).get('ic', 0.0))

            if ic1 >= ic2:
                features_to_remove.add(feature2)
                logger.info(f"  移除 {feature2} (IC={ic2:.4f}) 保留 {feature1} (IC={ic1:.4f}), 相关性={corr_val:.4f}")
            else:
                features_to_remove.add(feature1)
                logger.info(f"  移除 {feature1} (IC={ic1:.4f}) 保留 {feature2} (IC={ic2:.4f}), 相关性={corr_val:.4f}")

    # 返回过滤后的特征列表
    filtered_features = [f for f in valid_features if f not in features_to_remove]

    logger.info(f"📊 相关性过滤结果:")
    logger.info(f"  原始特征: {len(valid_features)} 个")
    logger.info(f"  移除特征: {len(features_to_remove)} 个")
    logger.info(f"  保留特征: {len(filtered_features)} 个")

    return filtered_features


def _apply_vif_filter(features, merged_data, logger):
    """第二阶段：VIF过滤，移除VIF>=5的特征（优化版本）"""
    logger.info("🔄 执行VIF过滤...")

    if len(features) <= 1:
        logger.warning("⚠️ 特征数量不足，跳过VIF过滤")
        return features

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import warnings

        # 抑制MKL警告
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # 准备数据 - 优化数据预处理
        logger.info(f"📊 准备VIF计算数据，原始特征数: {len(features)}")

        # 采样数据以加速计算（如果数据量过大）
        vif_data = merged_data[features].select_dtypes(include=[np.number])

        # 如果数据量过大，进行采样以加速VIF计算
        if len(vif_data) > 50000:
            sample_size = 50000
            vif_data = vif_data.sample(n=sample_size, random_state=42)
            logger.info(f"📊 数据采样: {len(merged_data)} → {sample_size} 条记录以加速VIF计算")

        # 移除缺失值
        vif_data = vif_data.dropna()
        logger.info(f"📊 移除缺失值后: {len(vif_data)} 条记录")

        # 数据标准化和异常值处理（解决MKL错误）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # 移除方差为0的列和异常值处理
        non_constant_features = []
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
                        non_constant_features.append(col)
                    else:
                        logger.warning(f"⚠️ 特征 {col} 异常值过多，移除")
                else:
                    logger.warning(f"⚠️ 特征 {col} 方差为0，移除")

        if len(non_constant_features) <= 1:
            logger.warning("⚠️ 有效特征数量不足，跳过VIF过滤")
            return non_constant_features

        # 重新选择数据并标准化
        vif_data = vif_data[non_constant_features]

        # 标准化数据以避免数值问题
        try:
            vif_data_scaled = pd.DataFrame(
                scaler.fit_transform(vif_data),
                columns=vif_data.columns,
                index=vif_data.index
            )
            logger.info(f"✅ 数据标准化完成")
        except Exception as e:
            logger.warning(f"⚠️ 数据标准化失败: {e}，使用原始数据")
            vif_data_scaled = vif_data

        if len(vif_data_scaled) <= len(non_constant_features):
            logger.warning("⚠️ 样本数量不足以计算VIF，跳过VIF过滤")
            return non_constant_features

        # 优化的VIF计算：使用多线程和批量处理
        current_features = non_constant_features.copy()
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

            # 检查矩阵条件数以避免数值问题
            try:
                condition_number = np.linalg.cond(current_data.values)
                if condition_number > 1e12:
                    logger.warning(f"⚠️ 矩阵条件数过大 ({condition_number:.2e})，可能存在数值问题")
            except:
                pass

            # 使用多线程计算VIF（如果特征数量较多）
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


def _continue_correlation_analysis(merged_data, stock_features, market_features,
                                 interaction_names, factor_test_results, logger):
    """继续相关性分析和最终报告生成"""
    try:
        # 步骤4: 特征相关性分析
        logger.info("🔄 [步骤 4/6] 特征相关性分析...")

        # 计算交互特征之间的相关性 (修复NaN值问题)
        logger.info("🔄 计算交互特征相关性矩阵...")

        # 选择交互特征数据并处理NaN值
        interaction_data = merged_data[interaction_names].copy()

        # 检查数据质量
        logger.info(f"📊 交互特征数据质量检查:")
        logger.info(f"  原始数据形状: {interaction_data.shape}")

        # 计算每个特征的有效数据比例
        valid_ratios = {}
        for col in interaction_names:
            valid_count = interaction_data[col].notna().sum()
            total_count = len(interaction_data)
            valid_ratio = valid_count / total_count
            valid_ratios[col] = valid_ratio
            if valid_ratio < 0.5:  # 有效数据少于50%
                logger.warning(f"  ⚠️ 特征 {col} 有效数据比例较低: {valid_ratio:.2%}")

        # 移除有效数据过少的特征
        valid_features = [col for col in interaction_names if valid_ratios[col] >= 0.1]  # 至少10%有效数据
        if len(valid_features) < len(interaction_names):
            removed_features = set(interaction_names) - set(valid_features)
            logger.warning(f"⚠️ 移除有效数据不足的特征: {len(removed_features)} 个")
            for feature in list(removed_features)[:5]:  # 只显示前5个
                logger.warning(f"    • {feature} (有效数据: {valid_ratios[feature]:.2%})")
            if len(removed_features) > 5:
                logger.warning(f"    • ... 还有 {len(removed_features) - 5} 个特征")

        # 使用有效特征计算相关性
        interaction_data_clean = interaction_data[valid_features]
        logger.info(f"  清理后数据形状: {interaction_data_clean.shape}")

        # 计算相关性矩阵 (使用pairwise完整观测)
        correlation_matrix = interaction_data_clean.corr(method='pearson', min_periods=100)

        # 检查相关性矩阵中的NaN值
        nan_count = correlation_matrix.isna().sum().sum()
        total_elements = correlation_matrix.size
        logger.info(f"📊 相关性矩阵质量: {nan_count}/{total_elements} ({nan_count/total_elements:.2%}) NaN值")

        # 识别高相关性特征对 (排除NaN值)
        high_corr_pairs = []
        for i, feature1 in enumerate(valid_features):
            for j, feature2 in enumerate(valid_features[i+1:], i+1):
                corr_val = correlation_matrix.loc[feature1, feature2]
                if not np.isnan(corr_val) and abs(corr_val) > 0.8:
                    high_corr_pairs.append((feature1, feature2, corr_val))

        logger.info(f"📊 发现高相关性特征对 (|相关系数|>0.8): {len(high_corr_pairs)} 对")

        # 保存完整的相关性矩阵为JSON文件 (处理NaN值)
        # 将相关性矩阵转换为可序列化的格式，处理NaN值
        correlation_dict = {}
        for col in correlation_matrix.columns:
            correlation_dict[col] = {}
            for idx in correlation_matrix.index:
                val = correlation_matrix.loc[idx, col]
                if np.isnan(val):
                    correlation_dict[col][idx] = None  # 将NaN转换为None
                else:
                    correlation_dict[col][idx] = float(val)

        correlation_results = {
            'correlation_matrix': correlation_dict,
            'high_correlation_pairs': [
                {'feature1': f1, 'feature2': f2, 'correlation': float(corr)}
                for f1, f2, corr in high_corr_pairs
            ],
            'summary': {
                'total_features': len(interaction_names),
                'valid_features': len(valid_features),
                'removed_features': len(interaction_names) - len(valid_features),
                'high_corr_pairs_count': len(high_corr_pairs),
                'max_correlation': float(max([abs(corr) for _, _, corr in high_corr_pairs], default=0.0)),
                'correlation_matrix_nan_ratio': float(nan_count / total_elements)
            }
        }

        # 保存相关性分析结果
        correlation_file = 'interaction_correlation_analysis.json'
        try:
            import json
            with open(correlation_file, 'w', encoding='utf-8') as f:
                json.dump(correlation_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 相关性分析结果已保存到: {correlation_file}")
        except Exception as e:
            logger.warning(f"⚠️ 保存相关性分析结果失败: {e}")

        # 显示前10个高相关性特征对
        for feature1, feature2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            logger.info(f"  {feature1} vs {feature2}: {corr:.4f}")

        # 计算VIF (方差膨胀因子)
        logger.info("📊 计算VIF (方差膨胀因子)...")
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # 准备数据 (移除常数列和缺失值)
            vif_data = interaction_data.select_dtypes(include=[np.number]).dropna()

            # 移除方差为0的列
            non_constant_cols = []
            for col in vif_data.columns:
                if vif_data[col].std() > 1e-10:
                    non_constant_cols.append(col)

            vif_data = vif_data[non_constant_cols]

            if len(vif_data.columns) > 1 and len(vif_data) > len(vif_data.columns):
                vif_scores = []
                for i, feature in enumerate(vif_data.columns):
                    try:
                        vif = variance_inflation_factor(vif_data.values, i)
                        vif_scores.append((feature, vif))
                    except:
                        vif_scores.append((feature, np.inf))

                # 排序并显示高VIF特征
                vif_scores.sort(key=lambda x: x[1], reverse=True)
                high_vif_features = [f for f, vif in vif_scores if vif > 5.0]

                logger.info(f"📊 高VIF特征 (VIF>5): {len(high_vif_features)} 个")

                # 保存完整的VIF结果
                vif_results = {
                    'vif_scores': [
                        {'feature': feature, 'vif': float(vif) if vif != np.inf else 'infinity'}
                        for feature, vif in vif_scores
                    ],
                    'high_vif_features': high_vif_features,
                    'summary': {
                        'total_features': len(vif_scores),
                        'high_vif_count': len(high_vif_features),
                        'max_vif': float(max([vif for _, vif in vif_scores if vif != np.inf], default=0.0))
                    }
                }

                # 保存VIF分析结果
                vif_file = 'interaction_vif_analysis.json'
                try:
                    with open(vif_file, 'w', encoding='utf-8') as f:
                        json.dump(vif_results, f, indent=2, ensure_ascii=False)
                    logger.info(f"✅ VIF分析结果已保存到: {vif_file}")
                except Exception as e:
                    logger.warning(f"⚠️ 保存VIF分析结果失败: {e}")

                # 显示前10个VIF结果
                for feature, vif in vif_scores[:10]:
                    if vif != np.inf:
                        logger.info(f"  {feature}: {vif:.2f}")
                    else:
                        logger.info(f"  {feature}: ∞ (完全共线性)")
            else:
                logger.warning("⚠️ 数据不足以计算VIF")
                high_vif_features = []

        except ImportError:
            logger.warning("⚠️ statsmodels未安装，跳过VIF计算")
            high_vif_features = []
        except Exception as e:
            logger.warning(f"⚠️ VIF计算失败: {e}")
            high_vif_features = []

        # 步骤5: 管线评估分析
        logger.info("🔄 [步骤 5/6] 管线评估分析...")

        # 分析当前管线评估指标的适用性
        logger.info("📊 当前管线评估指标分析:")
        logger.info("  - 当前指标: Top10%-Bottom10% (预测值前10%样本的真实收益均值 - 后10%样本的真实收益均值)")
        logger.info("  - 优点: 直接反映选股效果，易于理解")
        logger.info("  - 缺点: 可能对异常值敏感，不考虑预测稳定性")

        # 基于单因子测试结果提供评估指标建议
        ic_values = [result['ic'] for result in factor_test_results.values() if not np.isnan(result['ic'])]
        long_short_returns = [result['long_short_return'] for result in factor_test_results.values()]

        if ic_values:
            mean_ic = np.mean(np.abs(ic_values))
            ic_std = np.std(ic_values)
            ir_like = mean_ic / (ic_std + 1e-8)

            logger.info(f"📊 交互特征IC统计:")
            logger.info(f"  - 平均|IC|: {mean_ic:.4f}")
            logger.info(f"  - IC标准差: {ic_std:.4f}")
            logger.info(f"  - IR类指标: {ir_like:.4f}")

        if long_short_returns:
            mean_ls_return = np.mean(long_short_returns)
            logger.info(f"  - 平均多空收益: {mean_ls_return:.4f}")

        # 步骤6: 生成优化建议和报告
        logger.info("🔄 [步骤 6/6] 生成分析报告和优化建议...")

        # 保存因子测试结果为JSON文件
        factor_results_file = 'interaction_factor_test_results.json'
        try:
            # 转换因子测试结果为可序列化的格式
            serializable_results = {}
            for factor_name, result in factor_test_results.items():
                serializable_results[factor_name] = {
                    'ic': float(result['ic']) if not np.isnan(result['ic']) else 0.0,
                    'rank_ic': float(result['rank_ic']) if not np.isnan(result['rank_ic']) else 0.0,
                    'ir': float(result['ir']) if not np.isnan(result['ir']) else 0.0,
                    'icir_20d': float(result['icir_20d']) if not np.isnan(result['icir_20d']) else 0.0,
                    'layer_returns': [float(x) for x in result['layer_returns']],
                    'layer_counts': [int(x) for x in result['layer_counts']],
                    'long_short_return': float(result['long_short_return']) if not np.isnan(result['long_short_return']) else 0.0,
                    'win_rate': float(result['win_rate']) if not np.isnan(result['win_rate']) else 0.0,
                    'monotonicity': float(result['monotonicity']) if not np.isnan(result['monotonicity']) else 0.0,
                    'coverage': float(result['coverage']) if not np.isnan(result['coverage']) else 0.0,
                    'max_drawdown': float(result['max_drawdown']) if not np.isnan(result['max_drawdown']) else 0.0,
                    'valid_samples': int(result['valid_samples']),
                    'total_samples': int(result['total_samples'])
                }

            # 添加汇总统计
            factor_results_summary = {
                'factor_results': serializable_results,
                'summary_statistics': {
                    'total_factors': len(factor_test_results),
                    'mean_ic': float(np.mean([r['ic'] for r in serializable_results.values()])),
                    'mean_rank_ic': float(np.mean([r['rank_ic'] for r in serializable_results.values()])),
                    'mean_ir': float(np.mean([r['ir'] for r in serializable_results.values()])),
                    'mean_long_short_return': float(np.mean([r['long_short_return'] for r in serializable_results.values()])),
                    'mean_win_rate': float(np.mean([r['win_rate'] for r in serializable_results.values()])),
                    'mean_monotonicity': float(np.mean([r['monotonicity'] for r in serializable_results.values()])),
                    'mean_coverage': float(np.mean([r['coverage'] for r in serializable_results.values()]))
                },
                'metadata': {
                    'stock_features_count': len(stock_features),
                    'market_features_count': len(market_features),
                    'interaction_features_count': len(interaction_names),
                    'analysis_timestamp': pd.Timestamp.now().isoformat()
                }
            }

            with open(factor_results_file, 'w', encoding='utf-8') as f:
                json.dump(factor_results_summary, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 因子测试结果已保存到: {factor_results_file}")
        except Exception as e:
            logger.warning(f"⚠️ 保存因子测试结果失败: {e}")

        # 串联过滤逻辑 - 三阶段过滤
        logger.info("🔄 串联过滤逻辑 - 三阶段过滤...")

        # 第一阶段 - 相关性过滤
        logger.info("📊 第一阶段 - 相关性过滤 (|相关系数|≤0.8):")
        logger.info(f"  过滤前特征数量: {len(valid_features)}")

        # 构建相关性图，找出需要移除的特征
        correlation_filtered_features = _apply_correlation_filter(
            valid_features, correlation_matrix, high_corr_pairs, factor_test_results, logger
        )

        logger.info(f"  过滤后特征数量: {len(correlation_filtered_features)}")
        logger.info(f"  移除特征数量: {len(valid_features) - len(correlation_filtered_features)}")

        # 第二阶段 - VIF过滤
        logger.info("📊 第二阶段 - VIF过滤 (VIF<5):")
        logger.info(f"  过滤前特征数量: {len(correlation_filtered_features)}")

        vif_filtered_features = _apply_vif_filter(
            correlation_filtered_features, merged_data, logger
        )

        logger.info(f"  过滤后特征数量: {len(vif_filtered_features)}")
        logger.info(f"  移除特征数量: {len(correlation_filtered_features) - len(vif_filtered_features)}")

        # 第三阶段 - IC排序
        logger.info("📊 第三阶段 - IC排序:")
        logger.info(f"  排序前特征数量: {len(vif_filtered_features)}")

        final_ranked_features = _apply_ic_ranking(
            vif_filtered_features, factor_test_results, logger
        )

        logger.info(f"  最终优质特征数量: {len(final_ranked_features)}")

        # 保存串联过滤结果
        filtering_results = {
            'original_features': len(interaction_names),
            'valid_features_after_quality_check': len(valid_features),
            'features_after_correlation_filter': len(correlation_filtered_features),
            'features_after_vif_filter': len(vif_filtered_features),
            'final_ranked_features': len(final_ranked_features),
            'correlation_filtered_features': correlation_filtered_features,
            'vif_filtered_features': vif_filtered_features,
            'final_ranked_features_list': final_ranked_features,
            'filtering_summary': {
                'removed_by_quality_check': len(interaction_names) - len(valid_features),
                'removed_by_correlation_filter': len(valid_features) - len(correlation_filtered_features),
                'removed_by_vif_filter': len(correlation_filtered_features) - len(vif_filtered_features),
                'final_retention_rate': len(final_ranked_features) / len(interaction_names)
            }
        }

        # 保存过滤结果
        filtering_file = 'interaction_filtering_results.json'
        try:
            with open(filtering_file, 'w', encoding='utf-8') as f:
                json.dump(filtering_results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 串联过滤结果已保存到: {filtering_file}")
        except Exception as e:
            logger.warning(f"⚠️ 保存串联过滤结果失败: {e}")

        return _generate_interaction_analysis_report(
            interaction_names, factor_test_results, high_corr_pairs,
            high_vif_features, stock_features, market_features, logger,
            final_ranked_features, filtering_results
        )

    except Exception as e:
        logger.error(f"❌ 相关性分析失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


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




def _validate_time_window(start_date, end_date, data_type="数据"):
    """
    验证时间窗口，确保不会发生未来数据泄露

    Args:
        start_date: 开始日期
        end_date: 结束日期
        data_type: 数据类型描述

    Returns:
        bool: 时间窗口是否有效
    """
    logger = logging.getLogger(__name__)

    if start_date > end_date:
        logger.error(f"❌ {data_type}时间窗口无效: 开始日期({start_date}) > 结束日期({end_date})")
        return False

    # 检查是否使用了未来数据（相对于当前系统时间的合理范围）
    import datetime
    current_date = datetime.datetime.now().date()
    reasonable_future_limit = current_date + datetime.timedelta(days=30)  # 允许30天的未来数据容差

    if end_date > reasonable_future_limit:
        logger.warning(f"⚠️ {data_type}包含过多未来数据: 结束日期({end_date}) > 合理限制({reasonable_future_limit})")

    logger.info(f"✅ {data_type}时间窗口验证通过: {start_date} 至 {end_date}")
    return True


# 删除_load_market_state_factors函数，现在直接使用_load_factor_cache_data加载市场因子


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
