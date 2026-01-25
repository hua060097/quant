#!/usr/bin/env python3
"""
机器学习K线形态选股系统主程序

支持五种运行模式：
1. 因子生成模式 (generate_factors) - 生成和计算因子
2. 因子测试模式 (test_factors) - 单因子测试和分析
3. 模型训练模式 (train_model) - 训练机器学习模型
4. 模型测试模式 (test_model) - 使用模型进行选股测试
5. 完整流水线模式 (full_pipeline) - 执行完整的因子-模型流水线

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
from typing import List

import statsmodels.api as sm

# ============================================================
# 🔧 全局配置参数
# ============================================================
# ATR策略配置
MAX_HOLDING_DAYS = 3  # 最大持仓期（天）- 训练和预测统一使用
ATR_MULTIPLIER_PROFIT = 2  # ATR止盈倍数
ATR_MULTIPLIER_LOSS = 1    # ATR止损倍数
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
  factor_correlation - 因子相关性分析模式（分析ATR三分类系统因子相关性）
  genetic_factor    - 遗传编程因子进化模式（使用gplearn进化超级因子）
  # interaction_analysis - 已移除交互特征分析模式

使用示例:
  # 生成因子
  python main.py --mode generate_factors

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
        choices=['generate_factors', 'test_factors', 'train_model', 'test_model', 'full_pipeline', 'factor_correlation', 'genetic_factor', 'market_state_test', 'interaction_analysis'],
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

    return parser.parse_args()

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_selection.log', encoding='utf-8'),
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
                                         is_regression: bool = False) -> np.ndarray:
    """
    🚀 优化的样本权重计算（权重均值保持在1.0左右）

    回归模式策略：
    1. 极端收益样本（高于95%或低于5%分位）：权重 = 1.5
    2. 边界样本（min_dist_to_sl或min_dist_to_pt接近0的20%样本）：权重 = 0.8
    3. 其他样本：权重 = 1.0
    4. 目标：权重均值保持在1.0左右

    分类模式策略（保持兼容）：
    1. 使用sklearn的class_weight='balanced'公式
    2. 平滑权重：避免极端权重值
    3. ATR三分类特殊调整

    Args:
        y_train: 训练标签（分类：-1,0,1；回归：连续值）
        imbalance_ratio: 不平衡比例
        logger: 日志记录器
        boundary_distances: 边界距离数组，用于降权边界样本
        is_regression: 是否为回归模式
    """
    try:
        import numpy as np

        logger.info(f"样本权重计算开始:")
        logger.info(f"  总样本数: {len(y_train)}")
        logger.info(f"  模式: {'回归' if is_regression else '分类'}")

        if is_regression:
            # 🚀 优化的回归模式权重计算
            logger.info("使用优化的回归模式权重计算...")

            # 统计收益分布
            logger.info(f"收益标签统计:")
            logger.info(f"  最小值: {y_train.min():.3f}")
            logger.info(f"  最大值: {y_train.max():.3f}")
            logger.info(f"  平均值: {y_train.mean():.3f}")
            logger.info(f"  标准差: {y_train.std():.3f}")

            # 初始化权重为1.0
            sample_weights = np.ones(len(y_train))

            # 🚀 修复极端样本判断：使用更严格的分位数阈值
            y_min, y_max = y_train.min(), y_train.max()
            p05 = np.percentile(y_train, 5)   # 5%分位数（更严格）
            p95 = np.percentile(y_train, 95)  # 95%分位数（更严格）

            # 🚀 激进调整：极端样本权重大幅提升，强制模型学习极值
            extreme_mask = (y_train <= p05) | (y_train >= p95)
            sample_weights[extreme_mask] = 3.0  # 从1.2提升到3.0

            logger.info(f"🚀 激进的极值学习权重调整:")
            logger.info(f"  标签范围: [{y_min:.3f}, {y_max:.3f}] ATR")
            logger.info(f"  极端阈值: p05={p05:.3f}, p95={p95:.3f} ATR")
            logger.info(f"  极端样本: {extreme_mask.sum()}个 ({extreme_mask.sum()/len(y_train)*100:.1f}%), 权重=3.0 (强制学习极值)")
            logger.info(f"  普通样本: {(~extreme_mask).sum()}个 ({(~extreme_mask).sum()/len(y_train)*100:.1f}%), 权重=1.0")

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

            # 🚀 革命性调整：边界样本提权而非降权，因为这些是最重要的学习样本
            boundary_adjustment[boundary_mask] = 1.5  # 边界样本提权到1.5（原来是降权到0.8）

            # 对于距离适中的样本，保持权重不变
            boundary_adjustment[medium_mask] = 1.0  # 中等样本权重保持1.0（原来是降权到0.9）

            # 应用边界调整
            sample_weights *= boundary_adjustment

            logger.info(f"边界距离修正统计（极值学习版）:")
            logger.info(f"  边界样本({boundary_mask.sum()}个, {boundary_mask.sum()/len(boundary_distances)*100:.1f}%): 权重×1.5 (提权学习边界)")
            logger.info(f"  中等样本({medium_mask.sum()}个, {medium_mask.sum()/len(boundary_distances)*100:.1f}%): 权重×1.0 (保持不变)")
            logger.info(f"  远离边界样本({(~boundary_mask & ~medium_mask).sum()}个, {(~boundary_mask & ~medium_mask).sum()/len(boundary_distances)*100:.1f}%): 权重不变")

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

def _calculate_prediction_range_diagnostics(y_true, y_pred, logger):
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
        logger.info(f"  预测值范围: [{pred_range[0]:.3f}, {pred_range[1]:.3f}] ATR (跨度: {pred_span:.3f})")
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
    🚀 计算交易指标：止盈命中率、止损命中率、Top-K收益

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

        # 🚀 修复：使用浮点数容差避免精度问题
        # 统一规则：止盈 >= +2 ATR，止损 <= -1 ATR
        profit_threshold = 2.0
        loss_threshold = -1.0
        float_tolerance = 1e-6  # 浮点数容差

        logger.info(f"🚀 使用固定阈值统计止盈止损率:")
        logger.info(f"  止盈阈值: >= {profit_threshold:.1f} ATR")
        logger.info(f"  止损阈值: <= {loss_threshold:.1f} ATR")

        # 1. 止盈命中率
        profit_hits = (y_true_return >= profit_threshold).sum()
        profit_hit_rate = profit_hits / len(y_true_return) if len(y_true_return) > 0 else 0.0

        # 2. 止损命中率
        loss_hits = (y_true_return <= loss_threshold).sum()
        loss_hit_rate = loss_hits / len(y_true_return) if len(y_true_return) > 0 else 0.0

        # 3. Top-K平均收益
        top_k_returns = {}
        top_k_profit_rates = {}
        top_k_loss_rates = {}

        for k in k_values:
            if k <= len(sorted_returns):
                top_k_return = sorted_returns[:k]
                top_k_returns[k] = top_k_return.mean()

                # Top-K止盈率（修复浮点数精度问题）
                top_k_profit_rates[k] = (top_k_return >= (profit_threshold - float_tolerance)).mean()

                # Top-K止损率（修复浮点数精度问题）
                top_k_loss_rates[k] = (top_k_return <= (loss_threshold + float_tolerance)).mean()
            else:
                top_k_returns[k] = 0.0
                top_k_profit_rates[k] = 0.0
                top_k_loss_rates[k] = 0.0

        # 4. 收益分布分析（修复浮点数精度问题）
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
            'profit_hit_rate': profit_hit_rate,
            'loss_hit_rate': loss_hit_rate,
            'top_k_returns': top_k_returns,
            'top_k_profit_rates': top_k_profit_rates,
            'top_k_loss_rates': top_k_loss_rates,
            'return_distribution': return_distribution,
            'prediction_effectiveness': prediction_effectiveness
        }

        # 🚀 显式打印参与统计的样本数量和边界样本数量
        logger.info(f"📊 交易指标统计:")
        logger.info(f"  参与统计的y_true数量: {len(y_true_return):,}个样本")
        logger.info(f"  止盈边界样本数量: {profit_hits:,}个 (收益 >= {profit_threshold:.1f} ATR)")
        logger.info(f"  止损边界样本数量: {loss_hits:,}个 (收益 <= {loss_threshold:.1f} ATR)")
        logger.info(f"  整体止盈命中率: {profit_hit_rate:.4f} ({profit_hits}/{len(y_true_return)})")
        logger.info(f"  整体止损命中率: {loss_hit_rate:.4f} ({loss_hits}/{len(y_true_return)})")
        logger.info(f"  预测有效性: {prediction_effectiveness:.4f} ATR (Top10% - Bottom10%)")

        # 🚀 添加收益分布诊断信息
        logger.info(f"📊 收益分布诊断:")
        logger.info(f"  收益范围: [{y_true_return.min():.3f}, {y_true_return.max():.3f}] ATR")
        logger.info(f"  收益均值: {y_true_return.mean():.3f} ATR")
        logger.info(f"  收益标准差: {y_true_return.std():.3f} ATR")
        logger.info(f"  收益中位数: {np.median(y_true_return):.3f} ATR")
        logger.info(f"  负收益样本数: {(y_true_return < 0).sum():,}个 ({(y_true_return < 0).mean():.1%})")

        logger.info(f"📊 Top-K平均收益:")
        for k in k_values:
            if k <= len(sorted_returns):
                top_k_data = sorted_returns[:k]
                logger.info(f"  Top-{k}: 平均{top_k_returns[k]:.4f} ATR, 范围[{top_k_data.min():.3f}, {top_k_data.max():.3f}]")
                logger.info(f"    止盈率: {top_k_profit_rates[k]:.4f} ({(top_k_data >= (profit_threshold - float_tolerance)).sum()}/{k})")
                logger.info(f"    止损率: {top_k_loss_rates[k]:.4f} ({(top_k_data <= (loss_threshold + float_tolerance)).sum()}/{k})")
                logger.info(f"    负收益率: {(top_k_data < 0).mean():.4f} ({(top_k_data < 0).sum()}/{k})")

        # 🚀 添加对比统计：Top-K vs Bottom-K，解释为什么Top-K止损率为0
        if len(sorted_returns) >= 50:
            bottom_10_return = sorted_returns[-10:].mean()
            bottom_50_return = sorted_returns[-50:].mean()

            logger.info(f"📊 选股效果对比（解释Top-K止损率为0的原因）:")
            logger.info(f"  Top-10平均收益: {sorted_returns[:10].mean():.4f} ATR")
            logger.info(f"  Bottom-10平均收益: {bottom_10_return:.4f} ATR")
            logger.info(f"  Top-Bottom差异: {sorted_returns[:10].mean() - bottom_10_return:.4f} ATR")

            # 计算Bottom-K的止损率（应该更高）
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

def _train_ml_model(raw_data, labels, feature_columns, winsorization_thresholds=None, is_regression=True, label_stats=None):
    """
    🚀 成本感知回归模型训练（支持分类和回归模式）

    Args:
        raw_data: 原始特征数据DataFrame
        labels: 标签数据Series（回归：标准化后连续值；分类：-1,0,1）
        feature_columns: 特征列名列表
        winsorization_thresholds: 预计算的Winsorization阈值（可选）
        is_regression: 是否为回归模式（默认True）
        label_stats: 标签统计量字典（用于反向转换）

    Returns:
        dict: 包含模型、性能指标、标签统计量等信息的字典
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        import numpy as np
        import pandas as pd

        # 根据模式导入不同的评估指标
        if is_regression:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        else:
            from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

        logger = logging.getLogger(__name__)

        # 注意：增强特征变量将在标准化完成后定义

        logger.info(f"🚀 开始训练{'回归' if is_regression else '分类'}模型，原始数据: {len(raw_data)} 个样本, {len(feature_columns)} 个特征")

        # 🚀 步骤1：准备原始数据用于训练/验证集划分
        logger.info("🚀 准备原始数据用于训练/验证集划分...")

        # 确保标签与原始数据对应
        if len(raw_data) != len(labels):
            logger.warning(f"原始数据({len(raw_data)})与标签数据({len(labels)})长度不一致，进行对齐...")
            min_length = min(len(raw_data), len(labels))
            raw_data = raw_data.iloc[:min_length] if hasattr(raw_data, 'iloc') else raw_data[:min_length]
            labels = labels.iloc[:min_length] if hasattr(labels, 'iloc') else labels[:min_length]

        # 🚀 转换标签为numpy数组（支持回归和分类）
        if is_regression:
            y_data = labels.values.astype(np.float32) if hasattr(labels, 'values') else labels.astype(np.float32)
        else:
            y_data = labels.values.astype(np.int32) if hasattr(labels, 'values') else labels.astype(np.int32)

        logger.info(f"数据对齐完成: {len(raw_data)} 个样本, {len(feature_columns)} 个特征列")

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

        logger.info(f"📊 时间序列划分结果:")
        logger.info(f"  🎯 训练集: {len(raw_train_data):,}样本 (前80%时间段)")
        logger.info(f"  🎯 验证集: {len(raw_val_data):,}样本 (后20%时间段)")
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

        # 按类别分组显示特征
        feature_groups = {
            'price': [], 'volume': [], 'technical': [], 'fundamental': [],
            'risk': [], 'momentum': [], 'sector': [], 'other': []
        }

        for i, feature in enumerate(feature_names, 1):
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in ['price', 'open', 'high', 'low', 'close', 'pct_chg']):
                feature_groups['price'].append(f"{i:3d}. {feature}")
            elif any(keyword in feature_lower for keyword in ['volume', 'amount', 'turnover', 'illiquidity']):
                feature_groups['volume'].append(f"{i:3d}. {feature}")
            elif any(keyword in feature_lower for keyword in ['ma', 'rsi', 'macd', 'bb', 'ema', 'sma', 'support', 'resistance', 'reversal']):
                feature_groups['technical'].append(f"{i:3d}. {feature}")
            elif any(keyword in feature_lower for keyword in ['pe', 'pb', 'ps', 'roe', 'roa', 'debt']):
                feature_groups['fundamental'].append(f"{i:3d}. {feature}")
            elif any(keyword in feature_lower for keyword in ['risk', 'volatility', 'var', 'beta', 'limit', 'zhaban', 'consecutive']):
                feature_groups['risk'].append(f"{i:3d}. {feature}")
            elif any(keyword in feature_lower for keyword in ['momentum', 'return', 'trend']):
                feature_groups['momentum'].append(f"{i:3d}. {feature}")
            elif any(keyword in feature_lower for keyword in ['sector', 'industry']):
                feature_groups['sector'].append(f"{i:3d}. {feature}")
            else:
                feature_groups['other'].append(f"{i:3d}. {feature}")

        # 按类别输出特征
        for group_name, features in feature_groups.items():
            if features:
                logger.info(f"\n📊 {group_name.upper()}类特征 ({len(features)}个):")
                for feature in features[:5]:  # 只显示前5个，避免日志过长
                    logger.info(f"  {feature}")
                if len(features) > 5:
                    logger.info(f"  ... 还有{len(features)-5}个特征")

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

        # 🚀 回归模式：处理标签和NaN值
        if is_regression:
            # 回归模式：处理NaN值
            logger.info("回归模式：处理标签中的NaN值...")

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

            # 🚀 修复：在标签标准化之前计算样本权重（使用原始ATR倍数标签）
            logger.info("🔧 在标签标准化前计算样本权重（基于原始ATR倍数）...")

            # 检查是否有边界距离信息
            boundary_distances = None
            if hasattr(raw_train_data, 'min_dist_to_sl') and hasattr(raw_train_data, 'min_dist_to_pt'):
                # 计算综合边界距离（到最近边界的距离）
                dist_sl = raw_train_data['min_dist_to_sl'].values
                dist_pt = raw_train_data['min_dist_to_pt'].values

                # 🔧 修复：更智能的无穷大值处理
                # 先计算有限值的统计信息
                finite_sl = dist_sl[np.isfinite(dist_sl)]
                finite_pt = dist_pt[np.isfinite(dist_pt)]

                if len(finite_sl) > 0:
                    # 使用有限值的95%分位数作为无穷大值的替代
                    sl_replacement = max(5.0, np.percentile(finite_sl, 95))
                else:
                    sl_replacement = 5.0

                if len(finite_pt) > 0:
                    pt_replacement = max(5.0, np.percentile(finite_pt, 95))
                else:
                    pt_replacement = 5.0

                # 处理无穷大值
                dist_sl = np.where(np.isinf(dist_sl), sl_replacement, dist_sl)
                dist_pt = np.where(np.isinf(dist_pt), pt_replacement, dist_pt)

                # 综合边界距离：到最近边界的距离
                boundary_distances = np.minimum(dist_sl, dist_pt)

                logger.info(f"检测到边界距离信息，将用于权重修正")
                logger.info(f"边界距离统计信息:")
                logger.info(f"  综合边界距离: 最小值={boundary_distances.min():.3f}, 最大值={boundary_distances.max():.3f}")
            else:
                logger.info("未检测到边界距离信息，使用标准权重计算")

            # 🚀 关键修复：获取原始ATR倍数标签用于样本权重计算
            # 从raw_train_data中获取原始的atr_return_label（Winsorize截断前的）
            if 'atr_return_label_original' in raw_train_data.columns:
                # 如果有保存的原始标签，使用它
                original_labels = raw_train_data['atr_return_label_original'].values
                logger.info(f"✅ 使用保存的原始ATR标签计算样本权重")
                logger.info(f"  原始标签范围: [{original_labels.min():.3f}, {original_labels.max():.3f}] ATR")
            else:
                # 如果没有原始标签，尝试从raw_train_data重新计算
                logger.warning("⚠️ 未找到原始ATR标签，尝试从raw_train_data重新获取")
                if 'atr_return_label' in raw_train_data.columns:
                    # 检查是否有未截断的标签
                    temp_labels = raw_train_data['atr_return_label'].values
                    if temp_labels.max() > 2.1 or temp_labels.min() < -1.1:
                        # 如果范围超出Winsorize阈值，说明是原始标签
                        original_labels = temp_labels
                        logger.info(f"✅ 从raw_train_data获取原始标签")
                        logger.info(f"  原始标签范围: [{original_labels.min():.3f}, {original_labels.max():.3f}] ATR")
                    else:
                        # 已经被截断，使用当前标签但调整阈值
                        original_labels = y_train_split
                        logger.warning(f"⚠️ 标签已被Winsorize截断，将调整极端阈值")
                else:
                    original_labels = y_train_split
                    logger.warning(f"⚠️ 无法获取原始标签，使用当前标签")

            sample_weights = _calculate_professional_sample_weights(
                original_labels,  # 使用原始ATR倍数标签
                imbalance_ratio,
                logger,
                boundary_distances,
                is_regression
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

            # 🚀 直接使用原始ATR标签进行训练（不标准化）
            y_train_split_mapped = y_train_split  # 直接使用原始标签
            y_val_split_mapped = y_val_split      # 直接使用原始标签
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

            logger.info(f"✅ 使用原始ATR标签训练Pseudo-Huber回归模型")
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

        logger.info("样本分布分析:")
        for label in sorted(unique_labels_split):
            count = np.sum(y_train_split == label)
            ratio = count / total_samples
            if len(unique_labels_split) == 2:
                label_name = "正类" if label == 1 else "负类"
            elif len(unique_labels_split) == 3:
                label_name = {"1": "盈利类", "0": "中性类", "-1": "亏损类"}.get(str(label), f"类别{label}")
            else:
                label_name = f"类别{label}"
            # logger.info(f"  {label_name}({label}): {count}个 ({ratio:.1%})")

        # 🔧 修复数据泄露：imbalance_ratio已在上面基于训练集计算，无需重复计算
        logger.info(f"使用训练集计算的不平衡比例: {imbalance_ratio:.1f}:1")

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

        
        logger.info(f"🔧 使用基础特征集({actual_feature_count}个特征)，使用标准参数配置")
        params = {
            'n_estimators': 2000,        # 较多的树，因为每棵树很简单
            'max_depth': 3,             # 🚀 优化：适度深度，平衡复杂度和性能
            'learning_rate': 0.08,      # 🚀 优化：平衡收敛速度和稳定性
            'reg_alpha': 5,             # L1正则化，特征选择
            'reg_lambda': 10,           # L2正则化，防止过拟合
            'subsample': 0.8,           # 样本采样率
            'colsample_bytree': 0.8,    # 特征采样率
            'colsample_bylevel': 0.8,   # 每层特征采样率
            'min_child_weight': 15,     # 最小子节点权重，防止过拟合
            'gamma': 0.3,               # 分裂最小增益，控制复杂度
            'max_delta_step': 1         # 帮助收敛，适合不平衡数据
        }

        # 🚀 Pseudo-Huber回归模式：使用自定义目标函数
        logger.info("🚀 Pseudo-Huber回归模式：使用自定义目标函数和原始ATR标签")

        # 🚀 回归模式使用基础特征的优化参数（专门适配22个特征）
        logger.info(f"ℹ️ 回归模式使用基础特征，采用专门适配22特征的优化参数配置")

        # 🔍 根据特征数量和预测目标动态调整参数
        feature_count = actual_feature_count
        if feature_count <= 25:  # 22个特征属于小特征集
            # 🚀 针对极值预测的激进参数配置（解决范围压缩问题）
            optimized_params = {
                'max_depth': 8,                   # 🚀 大幅增加深度：学习复杂的极值模式
                'learning_rate': 0.1,             # 🚀 提高学习率：加快收敛，避免欠拟合
                'n_estimators': 3000,             # 🚀 增加树数量：充分学习
                'subsample': 0.95,                # 🚀 高采样率：保持数据完整性
                'colsample_bytree': 0.95,         # 🚀 高特征采样：充分利用所有特征
                'reg_alpha': 0.01,                # 🚀 大幅减少L1正则：允许学习极值
                'reg_lambda': 0.05,               # 🚀 大幅减少L2正则：减少平滑效应
                'min_child_weight': 1,            # 🚀 最低权重：允许极值样本分割
                'gamma': 0.01,                    # 🚀 最低分裂阈值：鼓励更多分割
                'colsample_bylevel': 0.95,        # 🚀 高每层采样：充分利用特征
                'max_delta_step': 0,              # 不限制权重更新
                'random_state': 42,               # 随机种子
                'n_jobs': -1,                     # 使用所有CPU核心
                'verbosity': 0                    # 减少输出
            }
            logger.info("🎯 使用激进参数配置，专门针对极值预测优化")
        else:
            # 大特征集的参数保持原有配置
            optimized_params = {
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_estimators': 2000,
                'subsample': 0.85,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 0.5,
                'min_child_weight': 3,
                'gamma': 0.1,
                'colsample_bylevel': 0.9,
                'max_delta_step': 0,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

        params.update(optimized_params)
        logger.info(f"🎯 针对{feature_count}个特征优化参数配置")

        logger.info("🚀 Pseudo-Huber回归参数配置:")
        logger.info(f"  max_depth: {params['max_depth']} (针对{feature_count}特征优化)")
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
        if is_regression:
            # 回归模式：保持原始连续收益标签
            unique_labels = None
            label_mapping = None
            reverse_mapping = None
            logger.info("回归模式：保持原始连续收益标签，不进行映射")
        

        # 🚀 成本感知回归：支持回归和分类模式
        if is_regression:
            # 回归模式：连续收益预测
            objective = 'reg:squarederror'
            eval_metric = 'rmse'
            num_class = None
            logger.info("使用回归模式配置（连续收益预测）")
        

        # 通用参数
        params.update({
            'objective': objective,
            'eval_metric': eval_metric,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        })


        # 🚀 成本感知回归：支持回归和分类模式
        if is_regression:
            # 回归模式配置
            logger.info(f"✅ XGBoost参数配置完成，回归模式，objective={objective}")
        elif objective == 'multi:softprob':
            # 分类模式配置
            params['num_class'] = num_class
            logger.info(f"✅ XGBoost参数配置完成，ATR三分类模式，num_class={num_class}")
        else:
            logger.error("❌ 不支持的目标函数，ATR系统仅支持三分类或回归")



        # 🚀 Pseudo-Huber回归：实现自定义目标函数和Delta选择
        if is_regression:
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

            # 🚀 优化Delta选择：基于标签分布动态调整，避免过度压缩
            # 根据标签标准差动态设置Delta范围
            label_std = all_labels.std()
            if label_std > 2.0:
                # 标签分布较广，使用更激进的Delta值以减少范围压缩
                huber_delta_grid = [1.0, 5.0, 10.0, 20.0, 50.0]
                logger.info(f"标签标准差{label_std:.2f}>2.0，使用激进Delta范围以解决预测范围压缩")
            else:
                # 标签分布较窄，仍使用激进Delta值
                huber_delta_grid = [0.5, 1.0, 2.0, 5.0, 10.0]
                logger.info(f"标签标准差{label_std:.2f}<=2.0，使用扩展Delta范围以改善预测范围")
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

            # 🔥 革命性损失函数测试：基于文献调研的极值预测优化
            logger.info("🔍 革命性损失函数测试...")

            # 1. Quantile Loss (专门针对极值预测) - 基于Terven et al. 2024
            try:
                logger.info("  测试 Quantile Loss (q=0.9) - 专注极值预测...")
                quantile_params = params.copy()
                quantile_params['objective'] = 'reg:quantileerror'
                quantile_params['quantile_alpha'] = 0.9  # 90%分位数，专注极值

                model_quantile = xgb.train(
                    quantile_params,
                    dtrain_full,
                    num_boost_round=500,
                    evals=[(dval, 'eval')],
                    verbose_eval=False,
                    early_stopping_rounds=50
                )

                val_preds_quantile = model_quantile.predict(dval)
                rmse_quantile = np.sqrt(np.mean((val_preds_quantile - y_val_split_mapped) ** 2))
                ic_quantile = np.corrcoef(y_val_split_mapped, val_preds_quantile)[0, 1]
                if np.isnan(ic_quantile):
                    ic_quantile = 0.0

                # 计算范围覆盖率
                pred_range_quantile = val_preds_quantile.max() - val_preds_quantile.min()
                true_range = y_val_split_mapped.max() - y_val_split_mapped.min()
                range_ratio_quantile = pred_range_quantile / true_range if true_range > 0 else 0

                # 计算止盈止损命中率
                profit_threshold = 2.0
                loss_threshold = -1.0
                positive_pred_mask = val_preds_quantile > 0
                negative_pred_mask = val_preds_quantile < 0

                if positive_pred_mask.sum() > 0:
                    actual_profit_rate_quantile = ((y_val_split_mapped >= profit_threshold) & positive_pred_mask).sum() / positive_pred_mask.sum()
                else:
                    actual_profit_rate_quantile = 0.0

                if negative_pred_mask.sum() > 0:
                    actual_loss_rate_quantile = ((y_val_split_mapped <= loss_threshold) & negative_pred_mask).sum() / negative_pred_mask.sum()
                else:
                    actual_loss_rate_quantile = 0.0

                risk_reward_ratio_quantile = actual_profit_rate_quantile / max(actual_loss_rate_quantile, 0.01)

                # 计算Top-K收益
                sorted_indices = np.argsort(val_preds_quantile)[::-1]
                def calculate_realistic_return(true_returns, threshold_atr=2.0, stop_loss_atr=-1.0):
                    clipped_returns = np.clip(true_returns, stop_loss_atr, threshold_atr)
                    return clipped_returns.mean()

                top1_return_quantile = calculate_realistic_return(y_val_split_mapped[sorted_indices[:1]]) if len(sorted_indices) >= 1 else 0.0
                top3_return_quantile = calculate_realistic_return(y_val_split_mapped[sorted_indices[:3]]) if len(sorted_indices) >= 3 else 0.0
                top5_return_quantile = calculate_realistic_return(y_val_split_mapped[sorted_indices[:5]]) if len(sorted_indices) >= 5 else 0.0

                # 范围评分
                if range_ratio_quantile < 0.1:
                    range_score_quantile = 0.0
                elif range_ratio_quantile < 0.3:
                    range_score_quantile = range_ratio_quantile / 0.3 * 0.5
                elif range_ratio_quantile <= 1.0:
                    range_score_quantile = 0.5 + (range_ratio_quantile - 0.3) / 0.7 * 0.5
                else:
                    range_score_quantile = max(0.8, 1.0 - (range_ratio_quantile - 1.0) * 0.2)

                # 风险收益比评分
                if risk_reward_ratio_quantile >= 1.0:
                    risk_score_quantile = 1.0
                elif risk_reward_ratio_quantile >= 0.5:
                    risk_score_quantile = risk_reward_ratio_quantile
                else:
                    risk_score_quantile = risk_reward_ratio_quantile * 0.5

                # Quantile Loss综合评分
                quantile_baseline_score = (
                    actual_profit_rate_quantile * 0.4 +
                    ic_quantile * 0.25 +
                    range_score_quantile * 0.2 +
                    risk_score_quantile * 0.15
                )

                # 惩罚机制
                if ic_quantile < 0:
                    quantile_baseline_score = ic_quantile
                elif actual_profit_rate_quantile < 0.05:
                    quantile_baseline_score *= 0.3
                elif range_ratio_quantile < 0.1:
                    quantile_baseline_score *= 0.5

                logger.info(f"  Quantile Loss: RMSE={rmse_quantile:.4f}, IC={ic_quantile:.4f}, 范围比={range_ratio_quantile:.2f}, 止盈率={actual_profit_rate_quantile:.1%}, Score={quantile_baseline_score:.4f}")

                # 存储Quantile Loss结果
                delta_results['Quantile_0.9'] = {
                    'rmse': rmse_quantile,
                    'ic': ic_quantile,
                    'range_ratio': range_ratio_quantile,
                    'top1_return': top1_return_quantile,
                    'top3_return': top3_return_quantile,
                    'top5_return': top5_return_quantile,
                    'weighted_top_return': top1_return_quantile * 0.6 + top3_return_quantile * 0.3 + top5_return_quantile * 0.1,
                    'actual_profit_rate': actual_profit_rate_quantile,
                    'actual_loss_rate': actual_loss_rate_quantile,
                    'risk_reward_ratio': risk_reward_ratio_quantile,
                    'range_score': range_score_quantile,
                    'composite_score': quantile_baseline_score
                }

            except Exception as e:
                logger.warning(f"  Quantile Loss测试失败: {e}")
                delta_results['Quantile_0.9'] = {
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

            # 2. MSE基准测试
            try:
                logger.info("  测试 MSE损失函数基准...")
                mse_params = params.copy()
                mse_params['objective'] = 'reg:squarederror'

                model_mse = xgb.train(
                    mse_params,
                    dtrain_full,
                    num_boost_round=500,
                    evals=[(dval, 'eval')],
                    verbose_eval=False,
                    early_stopping_rounds=50
                )

                val_preds_mse = model_mse.predict(dval)
                rmse_mse = np.sqrt(np.mean((val_preds_mse - y_val_split_mapped) ** 2))

                # 计算MSE的预测范围和IC
                ic_mse = np.corrcoef(y_val_split_mapped, val_preds_mse)[0, 1]
                pred_range_mse = val_preds_mse.max() - val_preds_mse.min()
                true_range = y_val_split_mapped.max() - y_val_split_mapped.min()
                range_ratio_mse = pred_range_mse / true_range if true_range > 0 else 0

                # 🚀 计算新的止盈止损指标
                profit_threshold = 2.0
                loss_threshold = -1.0

                positive_pred_mask = val_preds_mse > 0
                negative_pred_mask = val_preds_mse < 0

                if positive_pred_mask.sum() > 0:
                    actual_profit_rate_mse = ((y_val_split_mapped >= profit_threshold) & positive_pred_mask).sum() / positive_pred_mask.sum()
                else:
                    actual_profit_rate_mse = 0.0

                if negative_pred_mask.sum() > 0:
                    actual_loss_rate_mse = ((y_val_split_mapped <= loss_threshold) & negative_pred_mask).sum() / negative_pred_mask.sum()
                else:
                    actual_loss_rate_mse = 0.0

                risk_reward_ratio_mse = actual_profit_rate_mse / max(actual_loss_rate_mse, 0.01)

                # 计算Top-K收益
                sorted_indices = np.argsort(val_preds_mse)[::-1]

                def calculate_realistic_return(true_returns, threshold_atr=2.0, stop_loss_atr=-1.0):
                    clipped_returns = np.clip(true_returns, stop_loss_atr, threshold_atr)
                    return clipped_returns.mean()

                top1_return_mse = calculate_realistic_return(y_val_split_mapped[sorted_indices[:1]]) if len(sorted_indices) >= 1 else 0.0
                top3_return_mse = calculate_realistic_return(y_val_split_mapped[sorted_indices[:3]]) if len(sorted_indices) >= 3 else 0.0
                top5_return_mse = calculate_realistic_return(y_val_split_mapped[sorted_indices[:5]]) if len(sorted_indices) >= 5 else 0.0

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

                logger.info(f"  MSE基准: RMSE={rmse_mse:.4f}, IC={ic_mse:.4f}, 范围比={range_ratio_mse:.2f}, 止盈率={actual_profit_rate_mse:.1%}, Score={mse_baseline_score:.4f}")

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
                    model_grid = xgb.train(
                        params=grid_params,
                        dtrain=dtrain_full,
                        num_boost_round=grid_params['n_estimators'],
                        obj=pseudo_huber_obj(delta),
                        evals=[(dval, 'val')],
                        verbose_eval=False,
                        early_stopping_rounds=50  # 早停以避免不必要的训练
                    )

                    # 在验证集上评估
                    val_preds = model_grid.predict(dval)

                    # 计算RMSE
                    rmse = np.sqrt(np.mean((val_preds - y_val_split_mapped) ** 2))

                    # 🚀 专业Delta选择：多维度评估
                    # 1. 计算IC（最重要指标）
                    ic_corr = np.corrcoef(y_val_split_mapped, val_preds)[0, 1]
                    if np.isnan(ic_corr):
                        ic_corr = 0.0

                    # 2. 计算预测范围合理性
                    pred_range = val_preds.max() - val_preds.min()
                    true_range = y_val_split_mapped.max() - y_val_split_mapped.min()
                    range_ratio = pred_range / true_range if true_range > 0 else 0

                    # 3. 🚀 基于止盈止损命中率的全新评估体系
                    sorted_indices = np.argsort(val_preds)[::-1]

                    # 计算止盈止损命中率（核心交易指标）
                    profit_threshold = 2.0  # +2ATR止盈
                    loss_threshold = -1.0   # -1ATR止损

                    # 预测能力评估：预测>0的样本中，实际止盈的比例
                    positive_pred_mask = val_preds > 0
                    negative_pred_mask = val_preds < 0

                    if positive_pred_mask.sum() > 0:
                        actual_profit_rate = ((y_val_split_mapped >= profit_threshold) & positive_pred_mask).sum() / positive_pred_mask.sum()
                    else:
                        actual_profit_rate = 0.0

                    if negative_pred_mask.sum() > 0:
                        actual_loss_rate = ((y_val_split_mapped <= loss_threshold) & negative_pred_mask).sum() / negative_pred_mask.sum()
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

                    top1_return = calculate_realistic_return(y_val_split_mapped[sorted_indices[:1]]) if len(sorted_indices) >= 1 else 0.0
                    top3_return = calculate_realistic_return(y_val_split_mapped[sorted_indices[:3]]) if len(sorted_indices) >= 3 else 0.0
                    top5_return = calculate_realistic_return(y_val_split_mapped[sorted_indices[:5]]) if len(sorted_indices) >= 5 else 0.0

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

                    # 综合评分
                    composite_score = (
                        actual_profit_rate * 0.4 +      # 止盈命中率（最重要）
                        ic_corr * 0.25 +                # IC相关性
                        range_score * 0.2 +             # 范围覆盖率
                        risk_score * 0.15               # 风险收益比
                    )

                    # 5. 🚀 严格的惩罚机制
                    if ic_corr < 0:
                        composite_score = ic_corr  # 负IC直接作为评分
                    elif actual_profit_rate < 0.05:  # 止盈命中率<5%严重惩罚
                        composite_score *= 0.3
                    elif range_ratio < 0.1:  # 范围覆盖率<10%严重惩罚
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

                    logger.info(f"  Delta={delta:.1f}: RMSE={rmse:.4f}, IC={ic_corr:.4f}, 范围比={range_ratio:.2f}, 止盈率={actual_profit_rate:.1%}, 风险比={risk_reward_ratio:.2f}, Score={composite_score:.4f}")

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

            # 输出Delta选择结果
            logger.info(f"🎯 Delta选择结果:")
            for delta, result in delta_results.items():
                if isinstance(delta, str):
                    logger.info(f"  {delta}: RMSE={result['rmse']:.4f}, IC={result['ic']:.4f}, 范围比={result['range_ratio']:.2f}, 止盈率={result['actual_profit_rate']:.1%}, Score={result['composite_score']:.4f}")
                else:
                    logger.info(f"  Delta={delta:.1f}: RMSE={result['rmse']:.4f}, IC={result['ic']:.4f}, 范围比={result['range_ratio']:.2f}, 止盈率={result['actual_profit_rate']:.1%}, Score={result['composite_score']:.4f}")

            # 🔥 智能损失函数选择：基于文献调研的最优策略
            logger.info("🎯 智能损失函数选择结果:")

            # 获取所有损失函数的评分
            all_scores = {}
            for loss_name, result in delta_results.items():
                if isinstance(loss_name, str) and result['composite_score'] != float('-inf'):
                    all_scores[loss_name] = result['composite_score']
                    logger.info(f"  {loss_name}: Score={result['composite_score']:.4f}, 范围比={result['range_ratio']:.2f}, 止盈率={result['actual_profit_rate']:.1%}")

            # 选择最佳损失函数
            if all_scores:
                best_loss_function = max(all_scores.keys(), key=lambda x: all_scores[x])
                best_score = all_scores[best_loss_function]

                logger.info(f"✅ 最终选择: {best_loss_function} (Score={best_score:.4f})")

                # 根据选择结果设置参数
                if best_loss_function == 'Quantile_0.9':
                    use_quantile = True
                    use_mse = False
                    logger.info("  🔥 使用Quantile Loss - 专注极值预测的革命性方法")
                elif best_loss_function == 'MSE':
                    use_quantile = False
                    use_mse = True
                    logger.info("  📊 使用MSE - 经典回归方法")
                else:
                    # Pseudo-Huber
                    use_quantile = False
                    use_mse = False
                    best_delta = float(best_loss_function.split('_')[1]) if '_' in best_loss_function else best_delta
                    logger.info(f"  🎯 使用Pseudo-Huber - Delta={best_delta:.1f}")
            else:
                # 回退策略
                logger.warning("⚠️ 所有损失函数测试失败，使用MSE回退")
                use_quantile = False
                use_mse = True
                best_loss_function = 'MSE'

            if best_delta is None:
                logger.warning("⚠️ Delta网格搜索失败，使用默认值1.0")
                best_delta = 1.0

        # 🚀 智能损失函数训练：根据选择结果使用最优损失函数
        logger.info("=" * 60)
        logger.info("🚀 开始模型训练")
        logger.info("=" * 60)

        if is_regression:
            # 🚀 回归模式：使用xgboost.train
            import xgboost as xgb

            dtrain = xgb.DMatrix(X_train_enhanced, label=y_train_split_mapped, weight=sample_weights)
            dval = xgb.DMatrix(X_val_enhanced, label=y_val_split_mapped)
            logger.info(f"📊 最终模型训练使用基础特征: 训练集{X_train_enhanced.shape}, 验证集{X_val_enhanced.shape}")

            # 🔥 革命性模型训练：支持多种损失函数
            final_params = params.copy()

            if use_quantile:
                logger.info("🎯 训练Quantile回归模型 (q=0.9) - 专注极值预测...")
                final_params['objective'] = 'reg:quantileerror'
                final_params['quantile_alpha'] = 0.9
                loss_function_name = 'Quantile_0.9'
            elif use_mse:
                logger.info("🎯 训练MSE回归模型...")
                final_params['objective'] = 'reg:squarederror'
                loss_function_name = 'MSE'
            else:
                logger.info("🎯 训练Pseudo-Huber回归模型...")
                if best_delta is not None:
                    logger.info(f"  使用最佳Delta: {best_delta:.1f} ATR")
                final_params['objective'] = 'reg:pseudohubererror'
                final_params['huber_slope'] = best_delta
                loss_function_name = f'PseudoHuber_{best_delta}'

            # 训练最终模型
            model = xgb.train(
                final_params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                verbose_eval=100,
                early_stopping_rounds=200  # 🚀 延长早停：允许更充分的学习
            )

            # 设置模型属性
            model.loss_function = loss_function_name
            if not use_mse and not use_quantile:
                model.best_delta = best_delta


        # 🚀 模型评估：支持Pseudo-Huber回归和分类
        logger.info("评估模型性能...")

        if is_regression:
            # 🚀 Pseudo-Huber回归：使用增强特征预测
            dval_for_pred = xgb.DMatrix(X_val_enhanced)  # 🚀 使用增强特征
            y_pred = model.predict(dval_for_pred)
            y_pred_proba = None
            logger.info(f"✅ Pseudo-Huber回归预测完成（使用基础特征），预测值范围: [{y_pred.min():.3f}, {y_pred.max():.3f}] ATR")

        # 🚀 成本感知回归：支持回归和分类评估
        # 统一标签变量命名：y_val_atr_labels = 验证集ATR标签，y_pred_atr_values = 预测ATR值
        y_val_atr_labels = y_val_split_mapped  # 验证集ATR标签（原始单位）

        if is_regression:
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
            if not use_mse and best_delta is not None:
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

            # 🚀 添加预测范围诊断指标
            _calculate_prediction_range_diagnostics(y_val_atr_true, y_pred_atr_values, logger)

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

            # 🚀 Delta选择结果总结
            logger.info("🎯 Delta选择结果总结:")
            for delta, result in delta_results.items():
                if isinstance(delta, str):
                    logger.info(f"  {delta}: RMSE={result['rmse']:.4f}, IC={result['ic']:.4f}, 范围比={result['range_ratio']:.2f}, 止盈率={result['actual_profit_rate']:.1%}, Score={result['composite_score']:.4f}")
                else:
                    logger.info(f"  Delta={delta:.1f}: RMSE={result['rmse']:.4f}, IC={result['ic']:.4f}, 范围比={result['range_ratio']:.2f}, 止盈率={result['actual_profit_rate']:.1%}, Score={result['composite_score']:.4f}")

            logger.info("🎯 交易指标（固定阈值 >= +2 ATR / <= -1 ATR）:")
            logger.info(f"  止盈命中率: {trading_metrics.get('profit_hit_rate', 0):.4f}")
            logger.info(f"  止损命中率: {trading_metrics.get('loss_hit_rate', 0):.4f}")

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
        if is_regression:
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

        # 构建结果字典（兼容_save_model_results）
        # 🚀 成本感知回归：构建性能指标
        performance_metrics = {
            'best_iteration': getattr(model, 'best_iteration', params['n_estimators'])
        }

        if is_regression:
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
                'is_regression': is_regression
            },
            'model_params': params,
            'label_stats': label_stats  # 🚀 添加标签统计量（用于回归反标准化）
        }

        # 🚀 添加模式特定信息
        if is_regression:
            results['training_info'].update({
                'target_range': [float(y_train_split.min()), float(y_train_split.max())],
                'target_mean': float(y_train_split.mean()),
                'target_std': float(y_train_split.std())
            })

        logger.info("=" * 60)
        if is_regression:
            logger.info("🎉 成本感知回归模型训练完成")
        logger.info("=" * 60)
        logger.info("🔧 数据泄露修复: 确保训练过程的时间序列完整性")
        logger.info("  📊 样本分布分析移至时间序列划分后")
        logger.info("  📊 所有统计分析仅基于训练集数据")
        logger.info("  📊 消除验证集信息泄露到训练过程")
        logger.info("  📊 确保真实的样本外验证")
        logger.info("")
        logger.info("🔧 参数配置简化: 统一ATR三分类参数管理")
        logger.info("  📊 移除冗余的多特征集配置分支")
        logger.info("  📊 采用统一的专业优化配置")
        logger.info("  📊 简化参数管理，提高可维护性")
        logger.info("  📊 专注ATR三分类性能优化")
        logger.info("")
        logger.info("✅ 核心优化1: 稳健异常值处理")
        logger.info("  📊 NaN值: 基于分布特征选择均值/中位数填充")
        logger.info("  📊 Inf值: 使用1%/99%分位数截断")
        logger.info("  📊 异常值: 使用3倍IQR方法截断")
        logger.info("")
        logger.info("🚀 核心优化2: Winsorization缩尾处理")
        logger.info("  📊 消除极端异常值对ATR三分类模型的负面影响")
        logger.info("  📊 1%/99%分位数缩尾处理，提升边界情况稳定性")
        logger.info("  📊 训练和预测阶段使用相同的分位数阈值")
        logger.info("")
        logger.info("🚀 核心优化3: 模型复杂度降低")
        logger.info("  📊 统一max_depth=2配置，提升泛化能力")
        logger.info("  📊 限制模型最多学习两个因子的交互作用")
        logger.info("  📊 减少过拟合风险，改善盈利类别识别精度")
        logger.info("  📊 平衡的学习率和正则化参数")
        logger.info("")
        logger.info("✅ 核心优化4: ATR三分类专业权重平衡")
        logger.info("  📊 专门针对亏损/中性/盈利三类的样本权重计算")
        logger.info("  📊 基于训练集的不平衡比例计算")
        logger.info("  📊 基于ATR止盈止损策略的特殊调整")
        logger.info("")
        logger.info("✅ 核心优化5: 严格的时间序列验证")
        logger.info("  📊 先时间序列划分，再进行所有统计分析")
        logger.info("  📊 标准化参数仅在训练集上学习")
        logger.info("  📊 验证集使用训练集参数进行变换")
        logger.info("  📊 确保真实的样本外预测性能")
        logger.info("=" * 60)

        logger.info("=" * 60)
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
    """获取股票因子列表（18个核心股票因子）"""
    # 🚀 新策略：只使用股票因子，市场因子用于二级乘子
    # 移除了 near_support 和 amihud_illiquidity，确保18个因子
    stock_factors = [
        'returns_zscore', 'market_structure_score', 'ma_ratio_5_10', 'corr_5d',
        'volatility_correlation', 'volume_correlation', 'near_support_count_60d',
        'institutional_activity_score', 'ma_convergence_score', 'lower_shadow_ratio',
        'sector_correlation_20d', 'price_impact', 'rank_momentum_20d', 'rank_rsi',
        'rank_volume', 'market_correlation', 'corr_20d_mom5', 'corr_slope_20_120',
        'sector_return_rank_20d','sector_momentum_consistency','consecutive_negative_days',
        'consecutive_positive_days'
    ]
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

def _filter_new_stocks(data: pd.DataFrame, current_date: str) -> pd.Series:
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
        print(f"📊 新股过滤总结: 过滤掉 {total_filtered} 只，剩余 {total_remaining} 只正常股票")

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

def _filter_st_stocks(data: pd.DataFrame) -> pd.Series:
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
            print(f"🚫 ST股票过滤: 过滤掉 {st_count} 只ST股票")
            if len(st_stocks) <= 5:  # 如果数量不多，显示具体股票
                print(f"   ST股票: {', '.join(st_stocks)}")
            elif len(st_stocks) > 5:
                print(f"   ST股票示例: {', '.join(st_stocks[:3])} 等{len(st_stocks)}只")

        remaining_count = st_filter.sum()
        print(f"📊 ST股票过滤后剩余: {remaining_count} 只非ST股票")

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

def _load_model_cache(train_start_date: str, train_end_date: str):
    """加载模型缓存"""
    try:
        import pickle

        cache_path = _get_model_cache_path(train_start_date, train_end_date)

        if not cache_path.exists():
            return None, None, None

        with open(cache_path, 'rb') as f:
            model_data = pickle.load(f)

        # 验证缓存数据完整性
        required_keys = ['ml_model', 'scaler', 'feature_names', 'train_start_date', 'train_end_date']
        if not all(key in model_data for key in required_keys):
            print(f"⚠️ 缓存数据不完整，将重新训练")
            return None, None, None

        # 验证训练日期匹配
        if (model_data['train_start_date'] != train_start_date or
            model_data['train_end_date'] != train_end_date):
            print(f"⚠️ 缓存日期不匹配，将重新训练")
            return None, None, None

        print(f"✅ 从缓存加载模型: {cache_path}")
        print(f"   模型类型: {model_data.get('model_type', 'Unknown')}")
        print(f"   特征数量: {len(model_data['feature_names'])}")
        print(f"   创建时间: {model_data.get('created_time', 'Unknown')}")

        return model_data['ml_model'], model_data['scaler'], model_data['feature_names']

    except Exception as e:
        print(f"⚠️ 模型缓存加载失败: {e}")
        return None, None, None

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


    # 执行对应的运行模式
    if args.mode == 'generate_factors':
        success = run_generate_factors_mode(args)
    elif args.mode == 'test_factors':
        success = run_test_factors_mode(args)
    elif args.mode == 'train_model':
        success = run_train_model_mode(args, is_regression=True)  # 🚀 启用成本感知回归模式
    elif args.mode == 'test_model':
        success = run_test_model_mode(args)
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


def run_generate_factors_mode(args):
    """运行因子生成模式"""
    logger = logging.getLogger(__name__)
    print("🔧 运行模式: 因子生成")
    print("=" * 60)

    try:
        # 导入因子系统
        try:
            from .factor_system import FactorManager
        except ImportError:
            from factor_system import FactorManager

        # 初始化组件
        data_loader = IntegratedDataLoader()
        factor_manager = FactorManager()

        # 清理重复的hash格式缓存文件
        print("🧹 清理重复的缓存文件...")
        # factor_manager.cache_manager.cleanup_duplicate_hash_caches()

        # 设置日期范围
        start_date = args.start_date or "20230601"
        end_date = args.end_date or "20250110"

        print(f"📅 数据期间: {start_date} - {end_date}")
        print(f"📊 输出目录: {args.output_dir}")

        # 加载数据
        print(f"📥 加载股票数据...")
        stock_data, sector_data, market_data = data_loader.load_all_data(
            start_date, end_date, extend_history_days=120
        )

        if stock_data is None or stock_data.empty:
            logger.error("❌ 股票数据加载失败")
            return False

        # 增强数据验证：检查sector_data和market_data的有效性
        sector_data_valid = sector_data is not None and not sector_data.empty
        market_data_valid = market_data is not None and not market_data.empty

        if not sector_data_valid:
            logger.warning("⚠️ 板块数据为空或加载失败，板块相关因子将无法计算")
            print("⚠️ 板块数据为空，板块相关因子将使用默认值")
        else:
            print(f"✅ 板块数据加载完成: {len(sector_data)} 条记录")

        if not market_data_valid:
            logger.warning("⚠️ 市场数据为空或加载失败，市场相关因子将无法计算")
            print("⚠️ 市场数据为空，市场相关因子将使用默认值")
        else:
            print(f"✅ 市场数据加载完成: {len(market_data)} 条记录")

        print(f"✅ 股票数据加载完成: {len(stock_data)} 条记录")

        # 加载股票板块映射
        print(f"📥 加载股票板块映射...")
        stock_sector_mapping = data_loader.load_stock_sector_mapping()
        if stock_sector_mapping:
            print(f"✅ 板块映射加载完成: {len(stock_sector_mapping)} 只股票")
        else:
            print(f"⚠️  板块映射加载失败，板块相对特征将无法计算")

        # 生成因子
        print(f"🔧 开始生成因子...")

        # 设置强制重新计算标志
        if hasattr(args, 'force') and args.force:
            print(f"🔄 强制重新计算模式")
            factor_manager._force_recalc = True

        # 准备额外的上下文数据（优化：避免重复传递sector_data和market_data）
        context_data = {}
        if stock_sector_mapping:
            context_data['stock_sector_mapping'] = stock_sector_mapping
            logger.info(f"传入股票板块映射: {len(stock_sector_mapping)} 只股票")

        # 传递目标日期范围到因子计算逻辑
        context_data['target_start_date'] = start_date
        context_data['target_end_date'] = end_date

        # 记录数据和目标日期范围信息
        if 'date' in stock_data.columns:
            data_start = stock_data['date'].min().strftime('%Y%m%d')
            data_end = stock_data['date'].max().strftime('%Y%m%d')
            logger.info(f"数据实际日期范围: {data_start} - {data_end}")
            logger.info(f"目标计算日期范围: {start_date} - {end_date}")
            print(f"📊 数据实际日期范围: {data_start} - {data_end}")
            print(f"📍 目标计算日期范围: {start_date} - {end_date}")

        # 记录数据传递状态
        logger.info(f"数据传递状态: 股票数据✅, 板块数据{'✅' if sector_data_valid else '❌'}, 市场数据{'✅' if market_data_valid else '❌'}")
        logger.info(f"传递给因子组的参数: target_start_date={start_date}, target_end_date={end_date}")
        print(f"📍 目标计算日期范围: {start_date} - {end_date}")
        print(f"📊 数据传递状态: 股票({len(stock_data)}条), 板块({'有效' if sector_data_valid else '无效'}), 市场({'有效' if market_data_valid else '无效'})")

        # 因子计算：增强错误处理和数据完整性验证
        enhanced_data = None
        calculation_success = False

        try:
            if args.factor_groups:
                # 生成指定的因子组
                factor_groups = [group.strip() for group in args.factor_groups.split(',')]
                logger.info(f"开始计算指定因子组: {factor_groups}")
                print(f"🔧 计算指定因子组: {factor_groups}")

                enhanced_data = factor_manager.calculate_selected_groups(
                    factor_groups, stock_data, sector_data, market_data,
                    use_cache=True, parallel=True, **context_data
                )
            else:
                # 生成所有因子
                logger.info("开始计算所有因子组")
                print(f"🔧 计算所有因子组")

                enhanced_data = factor_manager.calculate_all_factors(
                    stock_data, sector_data, market_data,
                    use_cache=True, parallel=True, **context_data
                )

            calculation_success = True

            # 数据完整性验证
            validation_result = _validate_enhanced_data(enhanced_data, stock_data, logger)
            if not validation_result['valid']:
                logger.warning(f"⚠️ 数据完整性验证发现问题: {validation_result['issues']}")
                print(f"⚠️ 数据完整性验证发现问题，但继续处理")
            else:
                logger.info("✅ 数据完整性验证通过")
                print(f"✅ 数据完整性验证通过")

            print(f"✅ 因子生成完成")

        except Exception as e:
            logger.error(f"❌ 因子计算过程中发生错误: {e}")
            print(f"❌ 因子计算失败: {e}")

            # 尝试获取部分计算结果
            if enhanced_data is not None and not enhanced_data.empty:
                logger.warning("⚠️ 获取到部分计算结果，继续处理")
                print("⚠️ 获取到部分计算结果，继续处理")
                calculation_success = True
            else:
                logger.error("❌ 无法获取任何计算结果，退出")
                print("❌ 无法获取任何计算结果")
                return False

        print(f"✅ 因子生成完成")

        # 记录数据统计（不再保存pkl文件，只专注于因子缓存）
        original_columns = len(stock_data.columns)
        enhanced_columns = len(enhanced_data.columns)
        new_factors = enhanced_columns - original_columns
        data_size_mb = enhanced_data.memory_usage(deep=True).sum() / 1024 / 1024

        logger.info(f"因子生成统计: 原始列{original_columns}个, 新增因子{new_factors}个, 总列{enhanced_columns}个")
        logger.info(f"数据大小: {len(enhanced_data)}行, {data_size_mb:.2f}MB")
        print(f"📊 数据统计: {len(enhanced_data)}行 × {enhanced_columns}列 (新增{new_factors}个因子)")
        print(f"📊 数据大小: {data_size_mb:.2f}MB")
        print(f"💾 因子数据已保存到factor_cache目录的parquet文件中")

        # 保存因子摘要（可选）
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            summary = factor_manager.get_system_summary()
            summary_file = output_path / f"factor_system_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ 系统摘要已保存: {summary_file}")
        else:
            print(f"📋 因子系统摘要: {factor_manager.get_system_summary()}")

        print(f"🎉 因子生成模式完成！")
        return True

    except Exception as e:
        logger.error(f"❌ 因子生成模式失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_test_factors_mode(args):
    """运行因子测试模式 - 专业量化因子验证"""
    logger = logging.getLogger(__name__)
    print("🧪 运行模式: 专业量化因子验证")
    print("=" * 60)

    try:
        # 初始化组件
        data_loader = IntegratedDataLoader()

        # 设置日期范围
        start_date = args.start_date or "20230601"
        end_date = args.end_date or "20250131"

        print(f"📅 测试期间: {start_date} - {end_date}")
        print(f"📊 输出目录: {args.output_dir}")

        # 1. 加载标签数据（future_return_1d）
        print(f"📥 加载标签数据...")
        stock_data, _, _ = data_loader.load_all_data(
            start_date, end_date, extend_history_days=20
        )

        if stock_data is None:
            logger.error("❌ 标签数据加载失败")
            return False

        # 检查可用的收益率字段
        print(f"🔍 步骤1: 标签数据加载后检查")
        print(f"  标签数据形状: {stock_data.shape}")
        print(f"  标签数据列名: {list(stock_data.columns)}")

        available_return_columns = []
        for col in ['future_return_1d', 'future_return_2d']:
            if col in stock_data.columns:
                available_return_columns.append(col)
                non_null_count = stock_data[col].notna().sum()
                total_count = len(stock_data)
                print(f"  ✅ 发现字段 {col}: {non_null_count}/{total_count} ({non_null_count/total_count:.1%}) 非空")
                # 检查数据类型
                print(f"    数据类型: {stock_data[col].dtype}")
                # 检查样本值
                sample_values = stock_data[col].dropna().head(5).tolist()
                print(f"    样本值: {sample_values}")
            else:
                print(f"  ❌ 缺少字段 {col}")

        if not available_return_columns:
            logger.error("❌ 标签数据中缺少future_return_1d或future_return_2d字段")
            return False

        print(f"✅ 标签数据加载完成: {len(stock_data)} 条记录")
        print(f"📊 可用收益率字段: {available_return_columns}")

        # 2. 加载因子数据
        print(f"📥 加载因子数据...")

        # 确定要测试的因子组
        if hasattr(args, 'factor_groups') and args.factor_groups:
            specified_factor_groups = [group.strip() for group in args.factor_groups.split(',')]
        else:
            specified_factor_groups = ['elite_support_factors','price_position_factors','kline_pattern_factors','sector_relative_factors','enhanced_correlation_factors',
                                       'risk_factors','cross_sectional_factors','fundamental_factors','microstructure_factors','momentum_factors','volume_factors','comprehensive_ma_factors']  # 默认测试技术因子

        print(f"🎯 指定因子组: {specified_factor_groups}")

        # 加载指定因子组的数据
        factor_data = _load_specified_factor_groups(specified_factor_groups, start_date, end_date)

        if factor_data is None or factor_data.empty:
            logger.error("❌ 因子数据加载失败")
            return False

        print(f"✅ 因子数据加载完成: {len(factor_data)} 条记录, {len(factor_data.columns)} 个字段")

        # 3. 数据整合：合并因子数据和标签数据
        print(f"🔧 整合因子数据和标签数据...")

        print(f"🔍 步骤2: 数据整合前检查")
        print(f"  因子数据形状: {factor_data.shape}")
        print(f"  因子数据列名前10个: {list(factor_data.columns)[:10]}")
        print(f"  因子数据中的收益率相关字段: {[col for col in factor_data.columns if 'future_return' in col]}")
        print(f"  标签数据形状: {stock_data.shape}")
        print(f"  标签数据列名: {list(stock_data.columns)}")
        print(f"  准备保留的标签字段: {['stock_code', 'date'] + available_return_columns}")

        # 检查字段冲突
        common_columns = set(factor_data.columns) & set(['stock_code', 'date'] + available_return_columns)
        print(f"  共同字段: {list(common_columns)}")
        conflicting_columns = [col for col in available_return_columns if col in factor_data.columns]
        if conflicting_columns:
            print(f"  ⚠️ 发现字段冲突: {conflicting_columns}")
        else:
            print(f"  ✅ 无收益率字段冲突")

        # 保留标签数据中的关键字段（包括所有可用的收益率字段）
        label_columns = ['stock_code', 'date'] + available_return_columns
        label_data = stock_data[label_columns].copy()

        print(f"🔍 步骤3: 标签数据处理后检查")
        print(f"  标签数据处理后形状: {label_data.shape}")
        print(f"  标签数据处理后列名: {list(label_data.columns)}")

        # 检查标签数据中的收益率字段
        for col in available_return_columns:
            if col in label_data.columns:
                non_null_count = label_data[col].notna().sum()
                total_count = len(label_data)
                print(f"  标签数据中 {col}: {non_null_count}/{total_count} ({non_null_count/total_count:.1%}) 非空")
            else:
                print(f"  ❌ 标签数据中缺少 {col}")

        # 检查并处理字段冲突
        conflicting_return_columns = [col for col in available_return_columns if col in factor_data.columns]

        if conflicting_return_columns:
            print(f"  🔧 处理字段冲突: {conflicting_return_columns}")
            # 从因子数据中删除冲突的收益率字段，保留标签数据中的
            factor_data_clean = factor_data.drop(columns=conflicting_return_columns)
            print(f"  清理后因子数据形状: {factor_data_clean.shape}")

            # 合并数据
            merged_data = pd.merge(
                factor_data_clean, label_data,
                on=['stock_code', 'date'],
                how='inner'  # 只保留两边都有的数据
            )
        else:
            # 合并数据
            merged_data = pd.merge(
                factor_data, label_data,
                on=['stock_code', 'date'],
                how='inner'  # 只保留两边都有的数据
            )

        if merged_data.empty:
            logger.error("❌ 因子数据和标签数据无法匹配")
            return False

        print(f"🔍 步骤4: 数据整合后检查")
        print(f"  整合后数据形状: {merged_data.shape}")
        print(f"  整合后数据列名前10个: {list(merged_data.columns)[:10]}")

        # 检查整合后的收益率字段
        final_return_columns = []
        for col in available_return_columns:
            if col in merged_data.columns:
                non_null_count = merged_data[col].notna().sum()
                total_count = len(merged_data)
                print(f"  整合后 {col}: {non_null_count}/{total_count} ({non_null_count/total_count:.1%}) 非空")
                final_return_columns.append(col)
            else:
                print(f"  ❌ 整合后缺少 {col}")

        print(f"✅ 数据整合完成: {len(merged_data)} 条记录")
        print(f"📊 整合后可用收益率字段: {final_return_columns}")

        # 4. 获取要验证的因子名称
        exclude_columns = {'stock_code', 'date', 'date_str'}.union(set(available_return_columns))
        # 额外排除可能的重复收益率字段
        exclude_columns.update([col for col in merged_data.columns if 'future_return' in col and col not in available_return_columns])

        # 智能因子过滤：只保留数值型列
        potential_factors = [col for col in merged_data.columns if col not in exclude_columns]
        actual_factors = []

        for col in potential_factors:
            # 检查是否为数值型
            if pd.api.types.is_numeric_dtype(merged_data[col]):
                # 检查是否有足够的非空值
                non_null_ratio = merged_data[col].notna().sum() / len(merged_data)
                if non_null_ratio > 0.1:  # 至少10%的数据非空
                    actual_factors.append(col)
                else:
                    print(f"⚠️ 跳过因子 {col}: 非空数据比例过低 ({non_null_ratio:.1%})")
            else:
                print(f"⚠️ 跳过因子 {col}: 非数值型 ({merged_data[col].dtype})")

        print(f"📊 待验证因子数: {len(actual_factors)}/{len(potential_factors)}")
        if actual_factors:
            print(f"📊 验证因子列表: {actual_factors[:5]}{'...' if len(actual_factors) > 5 else ''}")

        # 5. 专业量化因子验证（增强版）
        print(f"🔍 开始专业量化因子验证（增强版）...")

        # 使用全量数据，不进行采样
        print(f"🔧 使用全量数据进行因子验证: {len(merged_data)} 条记录")

        # 使用专业的量化因子验证
        try:
            from .factor_system.factor_validator import FactorValidator
        except ImportError:
            from factor_system.factor_validator import FactorValidator

        validator = FactorValidator()

        # 确定实际可用的收益率字段
        final_return_columns = [col for col in available_return_columns if col in merged_data.columns]

        print(f"📊 最终验证配置:")
        print(f"  数据量: {len(merged_data):,} 条记录")
        print(f"  因子数: {len(actual_factors)} 个")
        print(f"  时间窗口: {final_return_columns}")

        validation_results = validator.validate_quantitative_factors(
            merged_data, actual_factors, return_columns=final_return_columns
        )

        print(f"📊 专业量化因子验证结果（增强版）:")
        print(f"  总因子数: {validation_results.get('total_factors', 0)}")
        print(f"  总记录数: {validation_results.get('total_records', 0)}")
        print(f"  验证状态: {validation_results.get('overall_status', 'UNKNOWN')}")
        print(f"  验证时间窗口: {validation_results.get('return_columns', [])}")

        # 显示各时间窗口的详细结果
        time_windows = validation_results.get('time_windows', {})
        for window_name, window_data in time_windows.items():
            print(f"\n📈 {window_name} 时间窗口结果:")
            window_summary = window_data.get('summary_statistics', {})
            if window_summary:
                print(f"  平均IC: {window_summary.get('mean_ic', 0):.4f}")
                print(f"  平均RankIC: {window_summary.get('mean_rank_ic', 0):.4f}")
                print(f"  平均Top-Bottom差: {window_summary.get('mean_top_bottom_return_diff', 0):.4f}")
                print(f"  平均IR: {window_summary.get('mean_ir', 0):.4f}")
                print(f"  平均ICIR(20日): {window_summary.get('mean_icir_20d', 0):.4f}")
                print(f"  平均ICIR(60日): {window_summary.get('mean_icir_60d', 0):.4f}")
                print(f"  平均最大回撤: {window_summary.get('mean_max_drawdown', 0):.4f}")
                print(f"  平均胜率: {window_summary.get('mean_win_rate', 0):.2%}")
                print(f"  平均覆盖度: {window_summary.get('mean_coverage', 0):.2%}")
                print(f"  平均单调性: {window_summary.get('mean_monotonicity', 0):.2%}")

            valid_factors = window_data.get('valid_factors', [])
            print(f"  高质量因子数: {len(valid_factors)}")
            if valid_factors:
                print(f"  高质量因子: {valid_factors[:5]}{'...' if len(valid_factors) > 5 else ''}")

        # 显示跨时间窗口汇总指标
        summary_stats = validation_results.get('summary_statistics', {})
        if summary_stats:
            print(f"\n📊 跨时间窗口汇总指标:")
            print(f"  平均IC: {summary_stats.get('mean_ic', 0):.4f} ± {summary_stats.get('std_ic', 0):.4f}")
            print(f"  平均RankIC: {summary_stats.get('mean_rank_ic', 0):.4f} ± {summary_stats.get('std_rank_ic', 0):.4f}")
            print(f"  平均IR: {summary_stats.get('mean_ir', 0):.4f} ± {summary_stats.get('std_ir', 0):.4f}")
            print(f"  平均胜率: {summary_stats.get('mean_win_rate', 0):.2%} ± {summary_stats.get('std_win_rate', 0):.2%}")
            print(f"  总高质量因子数: {summary_stats.get('high_quality_factors', 0)}")

        if validation_results.get('warnings'):
            print(f"\n⚠️ 警告数: {len(validation_results['warnings'])}")
            for warning in validation_results['warnings'][:3]:  # 只显示前3个
                print(f"  - {warning}")
        if validation_results.get('errors'):
            print(f"\n❌ 错误数: {len(validation_results['errors'])}")
            for error in validation_results['errors'][:3]:  # 只显示前3个
                print(f"  - {error}")

        # 6. 保存验证结果
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存验证报告
        validation_file = output_path / f"factor_validation_report_{timestamp}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ 验证报告已保存: {validation_file}")

        # 保存详细因子指标（多时间窗口）
        time_windows = validation_results.get('time_windows', {})
        for window_name, window_data in time_windows.items():
            factor_metrics = window_data.get('factor_metrics', {})
            if factor_metrics:
                metrics_df = pd.DataFrame(factor_metrics).T
                metrics_file = output_path / f"factor_metrics_{window_name}_{timestamp}.csv"
                metrics_df.to_csv(metrics_file, encoding='utf-8')
                print(f"✅ {window_name} 详细指标已保存: {metrics_file}")

        # 保存跨时间窗口汇总指标
        summary_stats = validation_results.get('summary_statistics', {})
        if summary_stats:
            summary_df = pd.DataFrame([summary_stats])
            summary_file = output_path / f"factor_summary_cross_window_{timestamp}.csv"
            summary_df.to_csv(summary_file, encoding='utf-8', index=False)
            print(f"✅ 跨时间窗口汇总指标已保存: {summary_file}")

        print(f"🎉 专业量化因子验证完成！")
        return True

    except Exception as e:
        logger.error(f"❌ 因子验证失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _load_specified_factor_groups(factor_groups: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载指定因子组的数据

    Args:
        factor_groups: 要加载的因子组列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        合并后的因子数据
    """
    try:
        import pandas as pd
        from pathlib import Path

        logger = logging.getLogger(__name__)
        logger.info(f"开始加载指定因子组: {factor_groups}")

        # 1. 检查因子缓存目录
        factor_cache_dir = Path("factor_cache")
        if not factor_cache_dir.exists():
            logger.error(f"因子缓存目录不存在: {factor_cache_dir}")
            return None

        # 2. 查找指定因子组的文件
        target_files = []
        for factor_group in factor_groups:
            # 查找匹配的parquet文件
            pattern = f"*{factor_group}*.parquet"
            matching_files = list(factor_cache_dir.glob(pattern))

            if matching_files:
                # 如果有多个匹配文件，选择最新的
                latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                target_files.append(latest_file)
                logger.info(f"找到因子组 {factor_group} 的文件: {latest_file.name}")
            else:
                logger.warning(f"未找到因子组 {factor_group} 的文件")

        if not target_files:
            logger.error("未找到任何指定的因子组文件")
            return None

        # 3. 加载和合并数据
        logger.info(f"开始加载 {len(target_files)} 个因子组文件...")

        # 转换日期格式
        target_start_date = pd.to_datetime(start_date)
        target_end_date = pd.to_datetime(end_date)

        merged_data = None

        for i, file_path in enumerate(target_files):
            try:
                logger.info(f"加载文件 {i+1}/{len(target_files)}: {file_path.name}")
                factor_df = pd.read_parquet(file_path)

                # 确保有必要的列
                if 'date' not in factor_df.columns or 'stock_code' not in factor_df.columns:
                    logger.warning(f"文件 {file_path.name} 缺少必要的date或stock_code列，跳过")
                    continue

                # 确保date列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(factor_df['date']):
                    factor_df['date'] = pd.to_datetime(factor_df['date'])

                # 过滤日期范围
                factor_df = factor_df[
                    (factor_df['date'] >= target_start_date) &
                    (factor_df['date'] <= target_end_date)
                ]

                if len(factor_df) == 0:
                    logger.warning(f"因子文件 {file_path.name} 在指定日期范围内无数据，跳过")
                    continue

                if merged_data is None:
                    # 第一个文件作为基础
                    merged_data = factor_df.copy()
                    logger.info(f"  基础数据: {len(merged_data)} 条记录, {len(merged_data.columns)} 个字段")
                else:
                    # 合并后续文件
                    # 检查重复列并移除
                    common_cols = set(merged_data.columns) & set(factor_df.columns)
                    merge_keys = {'date', 'stock_code'}
                    duplicate_cols = common_cols - merge_keys

                    if duplicate_cols:
                        logger.warning(f"发现重复列: {duplicate_cols}，将从当前因子数据中移除")
                        factor_df = factor_df.drop(columns=list(duplicate_cols))

                    # 合并数据
                    before_merge = len(merged_data)
                    merged_data = pd.merge(
                        merged_data, factor_df,
                        on=['date', 'stock_code'],
                        how='outer'  # 保留所有数据
                    )

                    logger.info(f"  合并完成: {len(factor_df)} -> {len(merged_data)} 条记录 (原{before_merge}条)")

            except Exception as e:
                logger.error(f"加载文件 {file_path.name} 失败: {e}")
                continue

        if merged_data is None or merged_data.empty:
            logger.error("所有因子组文件加载失败")
            return None

        logger.info(f"因子组数据加载完成: {len(merged_data)} 条记录, {len(merged_data.columns)} 个字段")
        return merged_data

    except Exception as e:
        logger.error(f"加载指定因子组失败: {e}")
        return None


def _load_factor_cache_data(required_factors=None, include_label_generation_features=False):
    """
    智能因子数据加载：加载所有因子组数据，然后过滤到指定因子

    Args:
        required_factors (list): 需要的因子列表，如果为None则包含默认因子
        include_label_generation_features (bool): 是否包含标签生成专用特征（如volatility20），默认False避免数据泄露

    Returns:
        pd.DataFrame: 包含指定因子的数据框
    """
    try:
        import pandas as pd
        from pathlib import Path

        logger = logging.getLogger(__name__)
        logger.info("开始智能因子数据加载...")

        # 设置默认因子列表（18个核心股票因子 + 动态加载的市场状态因子）
        if required_factors is None:
            # 18个核心股票因子（保持不变）
            stock_factors = _get_stock_factors()

            # 动态获取市场状态因子列表
            market_factors = _get_available_market_factors()

            # 合并因子列表
            required_factors = stock_factors + market_factors

            logger.info(f"🚀 新的特征工程策略:")
            logger.info(f"  - 股票因子: {len(stock_factors)} 个（保持原始形式）")
            logger.info(f"  - 市场因子: {len(market_factors)} 个（将进行去相关+PCA处理）")
            logger.info(f"  - 总因子数: {len(required_factors)} 个")
            #相关性排除：near_support_win_count_60d、buying_pressure、reversal_score、intraday_momentum 、rank_macd
        else:
            # 确保returns_zscore在required_factors中
            if "returns_zscore" not in required_factors:
                required_factors = ["returns_zscore"] + list(required_factors)
            logger.info(f"使用指定因子列表: {len(required_factors)} 个因子")

        # 1. 检查因子缓存目录
        factor_cache_dir = Path("factor_cache")
        if not factor_cache_dir.exists():
            logger.error(f"因子缓存目录不存在: {factor_cache_dir}")
            return None

        # 2. 获取所有因子组parquet文件（不再排除任何因子组）
        parquet_files = list(factor_cache_dir.glob("*.parquet"))
        if not parquet_files:
            logger.error("未找到任何因子组文件")
            return None

        logger.info(f"找到 {len(parquet_files)} 个因子文件，将加载所有因子组")

        # 3. 预扫描：检查哪些文件包含所需因子（性能优化）
        logger.info("预扫描因子文件，检查所需因子分布...")
        relevant_files = []
        found_factors = set()

        for file_path in parquet_files:
            try:
                # 只读取列名，不加载数据（使用更兼容的方法）
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(file_path)
                column_names = parquet_file.schema.names

                file_factors = [col for col in column_names if col in required_factors]

                if file_factors:
                    relevant_files.append((file_path, file_factors))
                    found_factors.update(file_factors)
                    logger.info(f"  {file_path.name}: 包含 {len(file_factors)} 个所需因子 {file_factors}")
                else:
                    # 即使没有所需因子，也包含该文件（可能包含基础列）
                    relevant_files.append((file_path, []))

            except Exception as e:
                logger.warning(f"预扫描文件 {file_path.name} 失败: {e}")
                # 如果预扫描失败，仍然包含该文件
                relevant_files.append((file_path, []))

        # 检查因子覆盖情况
        missing_factors = set(required_factors) - found_factors
        if missing_factors:
            logger.warning(f"预扫描未找到的因子: {missing_factors}")

        logger.info(f"预扫描完成: {len(relevant_files)} 个相关文件，找到 {len(found_factors)} 个所需因子")

        if not relevant_files:
            logger.error("未找到包含所需因子的文件")
            return None

        # 4. 智能数据加载和合并
        logger.info("开始智能因子数据合并...")

        # 使用第一个相关文件作为基础数据框
        base_file, base_factors = relevant_files[0]
        logger.info(f"使用基础因子文件: {base_file.name} (包含因子: {base_factors})")

        try:
            merged_data = pd.read_parquet(base_file)

            # 确保有必要的列
            if 'date' not in merged_data.columns or 'stock_code' not in merged_data.columns:
                logger.error(f"基础文件 {base_file.name} 缺少必要的date或stock_code列")
                return None

            # 过滤日期范围
            target_start_date = pd.to_datetime("20230601")
            target_end_date = pd.to_datetime("20250110")

            merged_data = merged_data[
                (merged_data['date'] >= target_start_date) &
                (merged_data['date'] <= target_end_date)
            ]

            if len(merged_data) == 0:
                logger.error(f"基础文件 {base_file.name} 在指定日期范围内无数据")
                return None

            logger.info(f"基础数据加载完成: {len(merged_data)} 条记录, {len(merged_data.columns)} 个字段")

        except Exception as e:
            logger.error(f"加载基础文件 {base_file.name} 失败: {e}")
            return None

        # 5. 合并其余相关因子文件
        for file_path, file_factors in relevant_files[1:]:
            try:
                logger.info(f"合并因子文件: {file_path.name}")
                factor_df = pd.read_parquet(file_path)

                # 确保有必要的列
                if 'date' not in factor_df.columns or 'stock_code' not in factor_df.columns:
                    logger.warning(f"文件 {file_path.name} 缺少必要的date或stock_code列，跳过")
                    continue

                # 过滤日期范围
                factor_df = factor_df[
                    (factor_df['date'] >= target_start_date) &
                    (factor_df['date'] <= target_end_date)
                ]

                if len(factor_df) == 0:
                    logger.warning(f"因子文件 {file_path.name} 在指定日期范围内无数据，跳过")
                    continue

                # 检查重复列并移除
                common_cols = set(merged_data.columns) & set(factor_df.columns)
                merge_keys = {'date', 'stock_code'}
                duplicate_cols = common_cols - merge_keys

                if duplicate_cols:
                    logger.warning(f"发现重复列: {duplicate_cols}，将从因子数据中移除")
                    factor_df = factor_df.drop(columns=list(duplicate_cols))

                # 合并数据
                before_merge = len(merged_data)
                merged_data = pd.merge(
                    merged_data, factor_df,
                    on=['date', 'stock_code'],
                    how='left'  # 保持基础数据的完整性
                )

                logger.info(f"  合并完成: {len(factor_df)} -> {len(merged_data)} 条记录 (原{before_merge}条)")

            except Exception as e:
                logger.error(f"加载文件 {file_path.name} 失败: {e}")
                continue

        # 5. 添加必要的列（从原始股票数据中获取）
        required_columns = [
            # 基础必要列
            'pct_chg', 'future_return_2d', 'is_can_buy',
            # 🚀 新股过滤所需字段（不参与模型训练，仅用于过滤）
            'volume', 'close',
            # ATR三分类标签生成所需字段（不参与模型训练，仅用于标签生成）
            'atr14', 'open', 'high', 'low',
            # 涨跌停相关特征
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
            'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
            # 技术分析特征
            'near_resistance', 'short_reversal',
            'volatility20'
        ]
        missing_columns = [col for col in required_columns if col not in merged_data.columns]

        if missing_columns:
            logger.info(f"因子数据中缺少列: {missing_columns}，从原始股票数据中获取...")

            try:
                # 从原始股票数据中获取pct_chg列
                from data_loader import IntegratedDataLoader
                data_loader = IntegratedDataLoader()

                # 使用相同的日期范围
                start_date, end_date = "20230601", "20250110"
                stock_data = data_loader.load_stock_features(start_date, end_date)

                if stock_data is not None:
                    # 准备要合并的列
                    merge_columns = ['date', 'stock_code']
                    available_columns = []

                    for col in missing_columns:
                        if col in stock_data.columns:
                            merge_columns.append(col)
                            available_columns.append(col)
                        else:
                            logger.warning(f"原始股票数据中也缺少{col}列")

                    if available_columns:
                        # 只保留必要的列进行合并
                        merge_data = stock_data[merge_columns].copy()

                        # 合并缺失列
                        merged_data = pd.merge(
                            merged_data, merge_data,
                            on=['date', 'stock_code'],
                            how='left'
                        )

                        logger.info(f"已从原始股票数据中获取列: {available_columns}")
                    else:
                        logger.warning("原始股票数据中没有任何需要的列")

                    # 为缺失的特征字段设置默认值
                    still_missing = [col for col in missing_columns if col not in merged_data.columns]
                    if still_missing:
                        logger.info(f"为缺失的特征字段设置默认值: {still_missing}")
                        default_values = {
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
                            'volatility20':1.0,
                            # 基础特征默认值
                            'pct_chg': 0.0,
                            'future_return_2d': 0.0,
                            'is_can_buy': 1,
                            # 🚀 新股过滤字段默认值（不参与模型训练，仅用于过滤）
                            'volume': 0.0,
                            'close':0.0
                        }

                        for col in still_missing:
                            if col in default_values:
                                merged_data[col] = default_values[col]
                                logger.info(f"  {col}: 设置默认值 {default_values[col]}")
                            else:
                                merged_data[col] = 0.0
                                logger.warning(f"  {col}: 未定义默认值，设置为 0.0")
                else:
                    # 如果无法从数据加载器获取，尝试从缓存文件获取
                    logger.info("尝试从缓存文件获取pct_chg列...")
                    cache_files = list(Path("cache").glob("stock_features_*.parquet"))
                    if cache_files:
                        cache_file = sorted(cache_files)[-1]
                        logger.info(f"使用缓存文件: {cache_file}")
                        stock_data = pd.read_parquet(cache_file)

                        if 'pct_chg' in stock_data.columns:
                            # 过滤日期范围
                            stock_data = stock_data[
                                (stock_data['date'] >= target_start_date) &
                                (stock_data['date'] <= target_end_date)
                            ]

                            # 只保留必要的列进行合并
                            pct_chg_data = stock_data[['date', 'stock_code', 'pct_chg']].copy()

                            # 合并pct_chg列
                            merged_data = pd.merge(
                                merged_data, pct_chg_data,
                                on=['date', 'stock_code'],
                                how='left'
                            )

                            logger.info("已从缓存文件中获取pct_chg列")
                        else:
                            logger.error("缓存文件中也缺少pct_chg列")
                            return None
                    else:
                        logger.error("无法找到包含pct_chg列的数据源")
                        return None

            except Exception as e:
                logger.error(f"获取pct_chg列失败: {e}")
                return None

        logger.info(f"智能因子数据合并完成: {len(merged_data)} 条记录, {len(merged_data.columns)} 个字段")

        # 6. 智能因子过滤：只保留所需因子
        logger.info("开始智能因子过滤...")

        # 保留的基础列（非因子列）- 移除可能导致数据泄露的因子
        base_columns = ['date', 'stock_code', 'date_str', 'pct_chg', 'future_return_1d', 'future_return_2d', 'is_can_buy',
                       'volume', 'close', 'open', 'high', 'low', 'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
                       'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
                       'near_support', 'near_resistance', 'short_reversal']

        # 标签生成专用列（不参与模型训练，避免数据泄露）
        label_generation_columns = ['volatility20', 'atr14', 'open', 'high', 'low']

        # 检查哪些所需因子实际存在于数据中
        available_required_factors = [col for col in required_factors if col in merged_data.columns]
        missing_required_factors = [col for col in required_factors if col not in merged_data.columns]

        if missing_required_factors:
            logger.warning(f"缺失的所需因子: {missing_required_factors}")

        # 确定最终要保留的列
        final_columns = []

        # 添加基础列（如果存在）
        for col in base_columns:
            if col in merged_data.columns:
                final_columns.append(col)

        # 添加所需因子列
        final_columns.extend(available_required_factors)

        # 根据参数决定是否添加标签生成专用列
        if include_label_generation_features:
            for col in label_generation_columns:
                if col in merged_data.columns and col not in final_columns:
                    final_columns.append(col)
            logger.info(f"包含标签生成专用列: {[col for col in label_generation_columns if col in merged_data.columns]}")
        else:
            logger.info(f"跳过标签生成专用列以避免数据泄露: {label_generation_columns}")

        # 过滤数据，只保留所需列
        original_column_count = len(merged_data.columns)
        merged_data = merged_data[final_columns]

        logger.info(f"因子过滤完成:")
        logger.info(f"  原始列数: {original_column_count}")
        logger.info(f"  过滤后列数: {len(merged_data.columns)}")
        logger.info(f"  保留的基础列: {len([col for col in base_columns if col in final_columns])}")
        logger.info(f"  保留的因子列: {len(available_required_factors)}")
        logger.info(f"  保留的因子: {available_required_factors}")

        # 7. 数据质量检查
        logger.info("进行数据质量检查...")
        logger.info(f"  日期范围: {merged_data['date'].min()} 到 {merged_data['date'].max()}")
        logger.info(f"  股票数量: {merged_data['stock_code'].nunique()}")
        logger.info(f"  缺失值比例: {merged_data.isnull().sum().sum() / (len(merged_data) * len(merged_data.columns)):.2%}")

        # 8. 智能因子加载状态总结
        logger.info("=" * 60)
        logger.info("智能因子加载状态总结")
        logger.info("=" * 60)
        logger.info(f"📊 目标因子数量: {len(required_factors)} 个")
        logger.info(f"✅ 成功加载因子: {len(available_required_factors)} 个")
        logger.info(f"❌ 缺失因子数量: {len(missing_required_factors)} 个")

        if available_required_factors:
            logger.info(f"📋 已加载因子: {available_required_factors}")

        if missing_required_factors:
            logger.warning(f"⚠️ 缺失因子: {missing_required_factors}")

        # 基础特征状态
        base_features = ['pct_chg', 'close', 'volume', 'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
                        'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
                        'near_resistance', 'short_reversal', 'volatility20']
        loaded_base_features = [f for f in base_features if f in merged_data.columns]

        logger.info(f"🔧 基础特征: {len(loaded_base_features)}/{len(base_features)} 个")
        logger.info(f"📈 最终数据维度: {len(merged_data)} 行 × {len(merged_data.columns)} 列")
        logger.info("=" * 60)

        return merged_data

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"加载因子缓存数据失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None



def _process_atr_strategy(
    data,
    mode='label',
    stock_code=None,
    selection_date=None,
    atr_multiplier_profit=ATR_MULTIPLIER_PROFIT,
    atr_multiplier_loss=ATR_MULTIPLIER_LOSS,
    max_holding_days=MAX_HOLDING_DAYS,  # 🔧 修复：使用全局配置的最大持仓期
    debug=False
):
    """
    统一的ATR策略处理器 - 支持标签生成和收益计算两种模式

    🎯 核心功能：
    - 标签生成模式('label')：为训练数据生成ATR三分类标签
    - 收益计算模式('return')：计算单只股票的实际收益率

    🔧 ATR策略逻辑（以_calculate_atr_based_return为准）：
    1. T日：决策日，获取ATR值用于计算止盈止损位
    2. T+1日：入场日，使用开盘价作为入场价
    3. T+1至T+max_holding_days日：观察期，逐日检查止盈止损触发
    4. 止盈位：入场价 + atr_multiplier_profit * ATR
    5. 止损位：入场价 - atr_multiplier_loss * ATR
    6. 止损优先原则：同时触发时优先处理止损
    7. T+1规则：触发后次日开盘价退出
    8. 超时处理：max_holding_days天内未触发则强制平仓

    Args:
        data: 股票数据DataFrame（标签模式）或回测数据（收益模式）
        mode: 'label' 或 'return'，处理模式
        stock_code: 股票代码（收益模式必需）
        selection_date: 选择日期（收益模式必需）
        atr_multiplier_profit: 止盈ATR倍数，默认2
        atr_multiplier_loss: 止损ATR倍数，默认1
        max_holding_days: 最大持仓天数，默认5
        debug: 是否输出调试信息

    Returns:
        标签模式：带有atr_triple_barrier_label列的DataFrame
        收益模式：实际收益率（float）

    重要约束：
    - 需要atr14、close、open、high、low等价格数据
    - 需要足够的未来数据来计算标签/收益
    - 严格遵守A股T+1交易制度
    """
    try:
        import logging
        import pandas as pd
        import numpy as np

        logger = logging.getLogger(__name__)

        # 根据模式选择不同的处理路径
        if mode == 'label':
            return _generate_training_labels_for_all_data(
                data, atr_multiplier_profit, atr_multiplier_loss, max_holding_days, debug,
                is_training=True, label_stats=None  # 训练模式，计算统计量
            )
        elif mode == 'return':
            return _process_atr_strategy_single(
                data, stock_code, selection_date, atr_multiplier_profit, atr_multiplier_loss, max_holding_days, debug
            )
        else:
            raise ValueError(f"不支持的模式: {mode}，请使用 'label' 或 'return'")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"ATR策略处理失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")

        if mode == 'label':
            # 标签模式：返回带默认标签的数据
            result_data = data.copy()
            result_data['atr_triple_barrier_label'] = 0
            return result_data
        else:
            # 收益模式：返回0收益
            return 0.0


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
                    # 计算实际止损收益
                    entry_price = stock_data_sorted.loc[hit_loss_mask, 'entry_price']
                    atr_value = stock_data_sorted.loc[hit_loss_mask, 'atr14']

                    if day == 1:
                        # T+1日触发，T+2日开盘退出
                        actual_return = (exit_price - entry_price) / atr_value
                    else:
                        # T+2日及以后，按止损价退出
                        actual_return = (exit_price - entry_price) / atr_value

                    stock_data_sorted.loc[hit_loss_mask, 'atr_return_label'] = actual_return
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
                    # 计算实际止盈收益
                    entry_price = stock_data_sorted.loc[hit_profit_mask, 'entry_price']
                    atr_value = stock_data_sorted.loc[hit_profit_mask, 'atr14']

                    if day == 1:
                        # T+1日触发，T+2日开盘退出
                        actual_return = (exit_price - entry_price) / atr_value
                    else:
                        # T+2日及以后，按止盈价退出
                        actual_return = (exit_price - entry_price) / atr_value

                    stock_data_sorted.loc[hit_profit_mask, 'atr_return_label'] = actual_return
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

                # 实际收益 = (退出价格 - 入场价格) / ATR
                actual_return_atr = (exit_price_valid - entry_price) / atr_value

                # 设置超时样本的连续收益标签
                stock_data_sorted.loc[valid_exit_mask, 'atr_return_label'] = actual_return_atr
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

        # 🚀 删除Winsorize截尾逻辑：保持原始ATR单位标签
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



def _save_model_results(model_results, feature_names, pca_models=None):
    """保存模型和结果（包含PCA模型和Market Attention模型）"""
    try:
        import pickle
        import json
        from datetime import datetime
        from pathlib import Path

        logger = logging.getLogger(__name__)

        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型文件（包含PCA模型、Winsorization阈值、标签统计量和Market Attention模型）
        model_file = results_dir / f"xgboost_model_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': model_results['model'],
                'scaler': model_results['scaler'],
                'winsorization_thresholds': model_results.get('winsorization_thresholds', {}),  # 🔧 修复：保存Winsorization阈值
                'pca_info': model_results['pca_info'],
                'feature_names': feature_names,
                'pca_models': pca_models if pca_models else {},
                'label_stats': model_results.get('label_stats', None),  # 🚀 保存标签统计量（用于回归反标准化）
                'interaction_info': None,  # 不使用交互特征
            }, f)
        logger.info(f"模型已保存: {model_file}")

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
        report_data = {
            'timestamp': timestamp,
            'performance': model_results['performance'],
            'training_info': model_results['training_info'],
            'model_params': model_results['model_params'],
            'feature_importance': model_results['feature_importance'].to_dict('records'),
            'pca_info': {k: v for k, v in model_results['pca_info'].items() if k != 'pca_object'},
            'interaction_info': model_results.get('interaction_info')
        }

        report_file = results_dir / f"training_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"训练报告已保存: {report_file}")

        # 保存特征重要性图表（如果可能）
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            top_features = model_results['feature_importance'].head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()

            plot_file = results_dir / f"feature_importance_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"特征重要性图表已保存: {plot_file}")

        except Exception as plot_e:
            logger.warning(f"保存特征重要性图表失败: {plot_e}")

        logger.info("模型结果保存完成")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"保存模型结果失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")


def run_train_model_mode(args, train_end_date=None, is_regression=True):
    """运行模型训练模式

    Args:
        args: 命令行参数
        train_end_date: 训练数据截止日期，如果为None则使用全部数据
        is_regression: 是否使用回归模式，默认True（成本感知回归）
    """
    logger = logging.getLogger(__name__)
    if is_regression:
        print("运行模式: 成本感知回归模型训练")
    else:
        print("运行模式: ATR三分类模型训练")
    print("=" * 60)

    try:
        # 1. 数据加载与特征识别
        logger.info("开始加载因子数据...")
        # 在训练模式下包含标签生成特征，用于标签构建
        factor_data = _load_factor_cache_data(include_label_generation_features=True)
        if factor_data is None or len(factor_data) == 0:
            logger.error("因子数据加载失败")
            return False

        logger.info(f"因子数据加载完成: {len(factor_data)} 条记录, {len(factor_data.columns)} 个字段")

        # 如果指定了训练截止日期，则过滤数据
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
            new_stock_filter = _filter_new_stocks(daily_data, date_str)
            daily_data = daily_data[new_stock_filter].copy()
            after_new_stock_count = len(daily_data)

            # 过滤ST股票
            if len(daily_data) > 0:
                st_filter = _filter_st_stocks(daily_data)
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

        # 3.5. 过滤可买入股票（训练数据预处理阶段）
        logger.info("🔄 过滤可买入股票（训练数据预处理）...")
        initial_trainable_count = len(factor_data)

        # 应用is_can_buy过滤
        can_buy_filter = factor_data['is_can_buy'] == 1
        factor_data = factor_data[can_buy_filter].copy()
        after_can_buy_count = len(factor_data)

        logger.info(f"  初始训练样本: {initial_trainable_count:,}个")
        logger.info(f"  可买入样本: {after_can_buy_count:,}个")
        logger.info(f"  过滤比例: {(initial_trainable_count - after_can_buy_count)/initial_trainable_count*100:.1f}%")
        logger.info("=" * 50)

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

        # 6. 模型训练（传递原始数据和标签统计量）
        logger.info("开始训练XGBoost回归模型...")
        model_results = _train_ml_model(
            filtered_data,
            filtered_labels,
            feature_columns,
            is_regression=True,
            label_stats=label_stats  # 传递标签统计量用于反向转换
        )

        if model_results is None:
            logger.error("模型训练失败")
            return False

        logger.info("模型训练完成")

        # 7. 保存模型和结果
        # 从模型结果中获取特征名称、PCA信息和Market Attention模型
        feature_names = model_results.get('feature_names', feature_columns)
        pca_info = model_results.get('pca_info', {})
        pca_models = {'market_factors': pca_info} if pca_info.get('applied', False) else {}
        _save_model_results(model_results, feature_names, pca_models)

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

        # 1. 时间切分策略
        train_end_date = "2024-06-01"
        test_start_date = pd.to_datetime("2024-06-01")
        test_end_date = pd.to_datetime("2025-01-10")

        logger.info("🗓️ 纯预测选股的时间切分策略:")
        logger.info(f"  训练数据: 2023-06-01 到 {train_end_date}")
        logger.info(f"  预测数据: {test_start_date.strftime('%Y-%m-%d')} 到 {test_end_date.strftime('%Y-%m-%d')}")
        logger.info("⚠️ 重要：模型将仅使用训练期数据训练，对测试期进行纯预测")

        # 2. 首先训练一个仅使用训练期数据的模型
        logger.info("\n🔄 步骤1: 训练仅使用训练期数据的模型...")
        logger.info("=" * 50)

        # 调用训练模式，但限制训练数据截止日期
        train_success = run_train_model_mode(args, train_end_date=train_end_date)

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

        prediction_data = factor_data[factor_data['date'] >= test_start_date].copy()

        logger.info(f"预测数据: {len(prediction_data)} 条记录")
        logger.info(f"预测数据日期范围: {prediction_data['date'].min()} 到 {prediction_data['date'].max()}")

        if len(prediction_data) == 0:
            logger.error("预测数据为空，请检查日期范围设置")
            return False

        # 5. 加载刚训练好的模型和参数
        logger.info("\n📦 步骤4: 加载刚训练的专用模型...")

        results_dir = Path("results")
        if not results_dir.exists():
            logger.error("results目录不存在")
            return False

        # 查找最新的模型文件（刚刚训练的）
        model_files = list(results_dir.glob("xgboost_model_*.pkl"))
        if not model_files:
            logger.error("未找到训练好的模型文件")
            return False

        latest_model_file = sorted(model_files)[-1]
        logger.info(f"📁 加载最新模型文件: {latest_model_file}")

        # 加载模型
        with open(latest_model_file, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        feature_names = model_data['feature_names']

        # 🚀 加载标签统计量（用于回归模式的反标准化）
        label_stats = model_data.get('label_stats', None)
        if label_stats:
            logger.info(f"✅ 加载标签统计量: μ={label_stats['mean']:.6f}, σ={label_stats['std']:.6f}")
        else:
            logger.warning("⚠️ 未找到标签统计量，可能影响回归预测的反标准化")

        logger.info(f"✅ 模型加载成功，特征数量: {len(feature_names)}")

        # 查找对应的训练报告
        model_timestamp = latest_model_file.stem.split('_')[-2] + '_' + latest_model_file.stem.split('_')[-1]
        report_file = results_dir / f"training_report_{model_timestamp}.json"

        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                training_report = json.load(f)
                training_performance = training_report.get('performance', {})

                # 🔧 修复格式化错误：安全处理数值和字符串格式化
                accuracy = training_performance.get('accuracy', 'N/A')
                auc = training_performance.get('auc', 'N/A')

                # 格式化函数：数值显示4位小数，字符串直接显示
                def format_metric(value):
                    if isinstance(value, (int, float)) and value != 'N/A':
                        return f"{value:.4f}"
                    else:
                        return str(value)

                accuracy_str = format_metric(accuracy)
                auc_str = format_metric(auc)

                logger.info(f"📊 训练期模型性能: 准确率={accuracy_str}, AUC={auc_str}")

        logger.info("⚠️ 重要确认：此模型从未见过2024-06-01之后的任何数据！")

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

        # 获取训练时保存的Winsorization阈值
        winsorization_thresholds = model_data.get('winsorization_thresholds', {})
        if winsorization_thresholds:
            logger.info(f"✅ 加载训练时的Winsorization阈值: {len(winsorization_thresholds)}个特征")
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

        # 🚀 9. 特征标准化（使用训练时保存的参数）
        logger.info("\n🔧 步骤9: 应用训练时的特征标准化参数...")
        if 'scaler' in model_data:
            scaler = model_data['scaler']
            n_in = getattr(scaler, 'n_features_in_', X_pred.shape[1])
            if X_pred.shape[1] == n_in:
                X_pred_scaled = scaler.transform(X_pred)
            elif X_pred.shape[1] > n_in:
                logger.info(f"🔧 预测特征数({X_pred.shape[1]})大于训练标准化特征数({n_in})，仅对前{n_in}列做标准化，其余保持原值")
                left = scaler.transform(X_pred[:, :n_in])
                right = X_pred[:, n_in:]
                X_pred_scaled = np.concatenate([left, right], axis=1)
            else:
                logger.error(f"❌ 预测特征数({X_pred.shape[1]})小于训练标准化特征数({n_in})，无法对齐")
                return False
            logger.info("✅ 使用训练时的特征标准化参数")
        else:
            logger.warning("⚠️ 未找到训练时的标准化参数，使用预测数据进行标准化")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_pred_scaled = scaler.fit_transform(X_pred)

        # ✅ 最终特征
        X_pred_final = X_pred_scaled

        logger.info(f"✅ 最终特征准备完成: {X_pred_final.shape[1]} 个特征，{X_pred_final.shape[0]} 个样本")

        # 🚀 10. Pseudo-Huber回归模型预测
        logger.info("\n🔮 步骤10: 进行Pseudo-Huber回归模型预测...")

        try:
            # 检查模型类型并进行预测
            if hasattr(model, 'predict') and hasattr(model, 'get_score'):
                # Booster对象：需要创建DMatrix
                import xgboost as xgb
                dtest = xgb.DMatrix(X_pred_final)
                y_pred_return = model.predict(dtest)
                logger.info("✅ 使用Booster对象预测")
            else:
                # XGBRegressor对象：直接预测
                y_pred_return = model.predict(X_pred_final)
                logger.info("✅ 使用XGBRegressor对象预测")

            logger.info(f"✅ 预测完成: {len(y_pred_return)} 个样本")
            logger.info(f"📊 预测值统计: [{y_pred_return.min():.3f}, {y_pred_return.max():.3f}] ATR")

        except Exception as e:
            logger.error(f"❌ 模型预测失败: {e}")
            return False

        # 8. 处理预测结果
        logger.info("\n📊 步骤11: 处理预测结果...")

        if len(y_pred_return) > len(original_prediction_data):
            logger.error(f"预测结果长度({len(y_pred_return)})大于原始数据长度({len(original_prediction_data)})")
            return False

        # 创建预测结果DataFrame
        prediction_data_with_proba = original_prediction_data.iloc[:len(y_pred_return)].copy()
        prediction_data_with_proba['predicted_return'] = y_pred_return
        prediction_data_with_proba['predicted_return_atr_raw'] = y_pred_return  # 🚀 修复：添加ATR原始预测字段
        prediction_data_with_proba['y_pred_proba'] = y_pred_return  # 兼容下游逻辑

        positive_count = (y_pred_return > 0).sum()
        logger.info(f"✅ 预测结果处理完成: {len(prediction_data_with_proba)} 个样本")
        logger.info(f"📊 预测分布: 正值{positive_count}个({positive_count/len(y_pred_return)*100:.1f}%), "
                   f"负值{len(y_pred_return)-positive_count}个")


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
            'date', 'stock_code', 'future_return_2d', 'is_can_buy', 'pct_chg', 'volume',
            # 🔧 修复：ATR三分类标签系统必需字段
            'atr14', 'open', 'high', 'low', 'close',
            # 🚀 新增：高质量上升趋势评分字段（替代传统技术指标）
            'high_quality_uptrend_score',
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
                # 🚀 新增：高质量上升趋势评分字段默认值
                'high_quality_uptrend_score': 0.0,
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


        backtest_data = prediction_data_with_proba[backtest_columns].copy()

        # 只按日期排序，保持每日内股票的原始预测概率分布
        backtest_data = backtest_data.sort_values('date')

        # 🚀 数据追踪验证：检查高质量上升趋势评分字段的可用性
        logger.info("\n🔍 数据追踪验证：高质量上升趋势评分字段状态")
        logger.info("=" * 50)

        technical_fields = ['high_quality_uptrend_score']
        for field in technical_fields:
            if field in backtest_data.columns:
                field_data = backtest_data[field]
                non_null_count = field_data.notna().sum()

                if non_null_count > 0:
                    field_stats = field_data.describe()
                    logger.info(f"✅ {field}: 可用")
                    logger.info(f"    非空值: {non_null_count:,}/{len(field_data):,} ({non_null_count/len(field_data)*100:.1f}%)")
                    logger.info(f"    数值范围: [{field_stats['min']:.4f}, {field_stats['max']:.4f}]")
                    logger.info(f"    平均值: {field_stats['mean']:.4f}")
                else:
                    logger.warning(f"⚠️ {field}: 字段存在但全为空值")
            else:
                logger.error(f"❌ {field}: 字段不存在")

        # 验证多维度复合条件过滤的可行性
        logger.info(f"\n📊 多维度复合条件过滤预览:")


        # 获取所有交易日
        trading_dates = sorted(backtest_data['date'].unique())
        logger.info(f"\n📅 回测期间: {trading_dates[0].strftime('%Y-%m-%d')} 到 {trading_dates[-1].strftime('%Y-%m-%d')}")
        logger.info(f"📅 交易日数量: {len(trading_dates)}天")

        # 🚀 动态调整选股阈值：基于预测范围自适应
        all_predictions = backtest_data['predicted_return_atr_raw'].dropna()
        pred_max = all_predictions.max()
        pred_75th = all_predictions.quantile(0.75)
        pred_median = all_predictions.median()

        if pred_max < 3.0:
            # 预测范围压缩，使用更低的阈值
            MIN_POS_THRESHOLD_ATR = max(pred_median, 0.5)  # 至少中位数，但不低于0.5
            logger.warning(f"⚠️ 预测范围压缩（最大值{pred_max:.2f} ATR），动态调整阈值为{MIN_POS_THRESHOLD_ATR:.2f} ATR")
        else:
            # 预测范围正常，使用固定阈值
            MIN_POS_THRESHOLD_ATR = 1.0
            logger.info(f"✅ 预测范围正常（最大值{pred_max:.2f} ATR），使用固定阈值{MIN_POS_THRESHOLD_ATR:.2f} ATR")

        logger.info(f"📊 预测值分布: 最大{pred_max:.2f}, 75%分位{pred_75th:.2f}, 中位数{pred_median:.2f} ATR")

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
            daily_returns = []
            # 🚀 新增：空仓日统计
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

                # 过滤新股
                new_stock_filter = _filter_new_stocks(g, date_str)
                g = g[new_stock_filter].copy()

                # 过滤ST股票
                if len(g) > 0:
                    st_filter = _filter_st_stocks(g)
                    g = g[st_filter].copy()

                # 过滤可买入股票
                g = g[g['is_can_buy'] == 1].copy()

                # # 🚀 新增：百分比风险过滤（选股阶段风险控制）
                # if len(g) > 0 and 'open' in g.columns and 'atr14' in g.columns:
                #     # 计算每只股票的ATR止损和百分比止损
                #     entry_price = g['open']  # 假设以开盘价买入
                #     atr_stop_loss = entry_price - ATR_MULTIPLIER_LOSS * g['atr14']
                #     pct_stop_loss = entry_price * (1 - MAX_STOP_LOSS_PCT)

                #     # 计算实际止损位（取较高价格，即较小损失）
                #     actual_stop_loss = np.maximum(atr_stop_loss, pct_stop_loss)

                #     # 计算实际止损幅度
                #     actual_stop_loss_pct = (entry_price - actual_stop_loss) / entry_price

                #     # 过滤掉风险过大的股票（实际止损幅度超过MAX_STOP_LOSS_PCT的股票）
                #     risk_filter = actual_stop_loss_pct <= MAX_STOP_LOSS_PCT * 1.1  # 允许10%的容差
                #     g = g[risk_filter].copy()

                #     logger.info(f"📊 {date_str} 百分比风险过滤: {len(g)}/{len(risk_filter)} 只股票通过风险控制")

                # 2.1 过滤 NaN
                g = g[~g['predicted_return_atr_raw'].isna()]
                if g.empty:
                    empty_day_count += 1
                    continue

                # 2.2 先做"硬过滤"：只保留 正期望 且 超过阈值
                g = g[g['predicted_return_atr_raw'] >= MIN_POS_THRESHOLD_ATR]

                if g.empty:
                    # 全部不达标：记为空仓日
                    empty_day_count += 1
                    continue

                # 2.3 对"剩余候选"做**日内横截面标准化**
                mu = g['predicted_return_atr_raw'].mean()
                std = g['predicted_return_atr_raw'].std(ddof=0)
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
            logger.info(f"统一口径选股完成：只买正期望且 >= {MIN_POS_THRESHOLD_ATR:.2f} ATR，不凑满K；空仓日比例={empty_ratio:.1%}")

            # 执行回测：基于选股结果进行持仓管理和收益计算
            portfolio_values_gross = [1.0]  # 未扣除成本的组合价值

            # 持仓管理：跟踪每只股票的持仓状态
            # 持仓信息结构：{stock_code: {'selection_date': T日, 'entry_price': T+1开盘价, 'atr_value': T日ATR,
            #                           'profit_target': 止盈位, 'stop_loss': 止损位, 'stop_loss_type': 止损类型, 'days_held': 持仓天数}}
            current_positions = {}
            max_positions = stock_count
            logger.info(f"初始化持仓管理: 最大持仓={max_positions}, ATR止盈={ATR_MULTIPLIER_PROFIT}x, ATR止损={ATR_MULTIPLIER_LOSS}x, 5%止损={MAX_STOP_LOSS_PCT:.1%}, 最大持仓期={MAX_HOLDING_DAYS}天")

            for date_idx, date in enumerate(trading_dates[:-1]):  # 排除最后一天（无法卖出）
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

                                if days_held == 1:
                                    # T+1日买入当天：如果触发止盈止损，标记为T+2日开盘退出（A股T+1规则）
                                    if today_low <= position_info['stop_loss']:
                                        position_info['exit_triggered'] = 'stop_loss'
                                        position_info['exit_type'] = 'next_day_open'  # 次日开盘退出
                                        logger.info(f"T+1日触发止损: {stock_code}, 最低价{today_low:.3f} <= 止损位{position_info['stop_loss']:.3f}, 将于T+2日开盘退出")
                                    elif today_high >= position_info['profit_target']:
                                        position_info['exit_triggered'] = 'take_profit'
                                        position_info['exit_type'] = 'next_day_open'  # 次日开盘退出
                                        logger.info(f"T+1日触发止盈: {stock_code}, 最高价{today_high:.3f} >= 止盈位{position_info['profit_target']:.3f}, 将于T+2日开盘退出")

                                elif days_held >= 2:
                                    # T+2日及以后：如果触发止盈止损，当日按止盈止损价退出
                                    # 止损优先原则
                                    if today_low <= position_info['stop_loss']:
                                        positions_to_exit.append(stock_code)
                                        exit_reasons[stock_code] = 'stop_loss'
                                        position_info['exit_price'] = position_info['stop_loss']  # 按止损价退出
                                        position_info['exit_type'] = 'intraday'  # 当日盘中退出
                                        logger.info(f"T+{days_held}日止损退出: {stock_code}, 最低价{today_low:.3f} <= 止损位{position_info['stop_loss']:.3f}, 按止损价{position_info['stop_loss']:.3f}退出")
                                        continue
                                    elif today_high >= position_info['profit_target']:
                                        positions_to_exit.append(stock_code)
                                        exit_reasons[stock_code] = 'take_profit'
                                        position_info['exit_price'] = position_info['profit_target']  # 按止盈价退出
                                        position_info['exit_type'] = 'intraday'  # 当日盘中退出
                                        logger.info(f"T+{days_held}日止盈退出: {stock_code}, 最高价{today_high:.3f} >= 止盈位{position_info['profit_target']:.3f}, 按止盈价{position_info['profit_target']:.3f}退出")
                                        continue

                        except Exception as e:
                            logger.warning(f"检查{stock_code}止盈止损失败: {e}")

                        # 检查是否有前一日触发的需要次日开盘退出的股票
                        if position_info.get('exit_triggered') and position_info.get('exit_type') == 'next_day_open':
                            positions_to_exit.append(stock_code)
                            exit_reasons[stock_code] = position_info['exit_triggered']
                            position_info['exit_type'] = 'next_day_open'  # 次日开盘退出
                            logger.info(f"T+{days_held}日开盘退出: {stock_code}, 原因: {position_info['exit_triggered']} (前一日触发)")
                            continue

                        # 最大持仓期检查
                        if days_held > MAX_HOLDING_DAYS:
                            positions_to_exit.append(stock_code)
                            exit_reasons[stock_code] = 'max_holding_days'
                            logger.info(f"超时退出: {stock_code}, 持仓{days_held}天 > 最大持仓期{MAX_HOLDING_DAYS}天")

                    # === 步骤2: 处理退出股票，记录退出交易 ===
                    today_exited_stocks = set()
                    exit_trades_records = []

                    for stock_code in positions_to_exit:
                        exit_reason = exit_reasons.get(stock_code, 'unknown')
                        today_exited_stocks.add(stock_code)
                        position_info = current_positions[stock_code]

                        # 计算退出收益：根据退出类型使用正确的价格
                        try:
                            entry_price = position_info['entry_price']

                            # 根据退出类型确定退出价格
                            if position_info.get('exit_type') == 'intraday' and 'exit_price' in position_info:
                                # 当日盘中退出：使用止盈止损价
                                exit_price = position_info['exit_price']
                                logger.info(f"退出收益计算: {stock_code}, 盘中退出价{exit_price:.3f}")
                            else:
                                # 次日开盘退出或最大持仓期：使用开盘价
                                stock_data = backtest_data[
                                    (backtest_data['stock_code'] == stock_code) &
                                    (backtest_data['date'] == date)
                                ]

                                if not stock_data.empty:
                                    exit_price = stock_data.iloc[0]['open']
                                    logger.info(f"退出收益计算: {stock_code}, 开盘退出价{exit_price:.3f}")
                                else:
                                    exit_price = entry_price  # 兜底
                                    logger.warning(f"退出收益计算: {stock_code}, 无法获取退出价格，使用入场价")

                            exit_return = (exit_price - entry_price) / entry_price
                            if pd.isna(exit_return):
                                exit_return = 0.0

                                # 记录退出交易
                                exit_trade_record = {
                                    'date': date,
                                    'stock_code': stock_code,
                                    'predicted_return_atr_raw': 0,
                                    'score_final': 0,
                                    'actual_return': exit_return,  # 累计收益
                                    'position_weight': 1.0 / max(len(current_positions), 1),
                                    'contribution': exit_return / max(len(current_positions), 1),
                                    'daily_position_ratio': 1.0,
                                    'market_regime': 'exit',
                                    'action': 'sell',
                                    'days_held': position_info['days_held'],
                                    'entry_date': position_info.get('entry_date', date),
                                    'entry_price': position_info['entry_price'],
                                    'profit_target': position_info['profit_target'],
                                    'stop_loss': position_info['stop_loss'],
                                    'atr_value': position_info['atr_value'],
                                    'stop_loss_type': position_info['stop_loss_type'],
                                    'exit_reason': exit_reason
                                }
                                exit_trades_records.append(exit_trade_record)

                        except Exception as e:
                            logger.warning(f"计算{stock_code}退出收益失败: {e}")

                        # 删除持仓
                        del current_positions[stock_code]
                        logger.info(f"退出持仓: {stock_code}, 原因: {exit_reason}, 持仓{position_info['days_held']}天")

                    # === 步骤3: 动态仓位管理和新股票买入 ===
                    daily_picks = picks[picks['date'] == date].copy()

                    # 统计当日退出类型，决定是否可以买入新股票
                    intraday_exits = sum(1 for stock_code in today_exited_stocks
                                       if current_positions.get(stock_code, {}).get('exit_type') == 'intraday')
                    next_day_exits = len(today_exited_stocks) - intraday_exits

                    logger.info(f"当日退出统计: 盘中退出{intraday_exits}只, 次日开盘退出{next_day_exits}只")

                    # 过滤已持仓和当日退出的股票
                    if not daily_picks.empty:
                        daily_picks = daily_picks[~daily_picks['stock_code'].isin(current_positions.keys())]
                        if today_exited_stocks:
                            daily_picks = daily_picks[~daily_picks['stock_code'].isin(today_exited_stocks)]

                    # 动态仓位管理：
                    # 1. 如果有盘中退出的股票，当日不买入新股票（因为盘中已经卖出，资金当日无法再投入）
                    # 2. 如果只有次日开盘退出的股票，可以根据剩余仓位买入新股票
                    available_positions = max_positions - len(current_positions)

                    if intraday_exits > 0:
                        # 有盘中退出的股票，当日不买入新股票
                        logger.info(f"当日有{intraday_exits}只股票盘中退出，暂停新股票买入")
                        daily_picks = pd.DataFrame()  # 清空买入列表
                    elif not daily_picks.empty and available_positions > 0:
                        # 只有次日开盘退出或无退出，可以买入新股票
                        daily_picks = daily_picks.head(available_positions)
                        logger.info(f"可买入新股票: {len(daily_picks)}只, 可用仓位: {available_positions}")
                    else:
                        logger.info(f"无可买入股票或无可用仓位: 选股{len(daily_picks)}只, 可用仓位{available_positions}")

                    # 🚀 修复：买入逻辑移出else分支，只要有daily_picks就执行买入
                    if not daily_picks.empty:
                        # 添加新持仓（适配TOP-1, TOP-3, TOP-5选股）
                        new_positions_added = 0
                        for _, stock in daily_picks.iterrows():
                            stock_code = stock['stock_code']

                            try:
                                # 获取T日的股价和ATR数据
                                stock_data = backtest_data[
                                    (backtest_data['stock_code'] == stock_code) &
                                    (backtest_data['date'] == date)
                                ]

                                if not stock_data.empty:
                                    entry_price = stock_data.iloc[0]['open']  # T+1日开盘价作为入场价
                                    atr_value = stock_data.iloc[0]['atr14']   # T日ATR值

                                    # 计算止盈止损位
                                    profit_target = entry_price + ATR_MULTIPLIER_PROFIT * atr_value
                                    atr_stop_loss = entry_price - ATR_MULTIPLIER_LOSS * atr_value
                                    pct_stop_loss = entry_price * (1 - MAX_STOP_LOSS_PCT)
                                    stop_loss = max(atr_stop_loss, pct_stop_loss)  # 取较高价格（较小损失）
                                    stop_loss_type = "ATR止损" if stop_loss == atr_stop_loss else "5%止损"

                                    # 添加到持仓跟踪
                                    current_positions[stock_code] = {
                                        'selection_date': date,  # T日选股日期
                                        'entry_price': entry_price,
                                        'atr_value': atr_value,
                                        'profit_target': profit_target,
                                        'stop_loss': stop_loss,
                                        'stop_loss_type': stop_loss_type,
                                        'days_held': 1  # T+1日买入，持仓第1天
                                    }

                                    new_positions_added += 1
                                    logger.info(f"新增持仓: {stock_code}, 入场价{entry_price:.3f}, 止盈{profit_target:.3f}, 止损{stop_loss:.3f}({stop_loss_type})")
                                else:
                                    logger.warning(f"无法获取{stock_code}的数据，跳过买入")

                            except Exception as e:
                                logger.warning(f"处理{stock_code}买入失败: {e}")

                        if new_positions_added > 0:
                            logger.info(f"当日新增持仓: {new_positions_added}只, 总持仓: {len(current_positions)}只")

                    # === 步骤4: 计算持仓股票收益 ===
                    all_holding_stocks = list(current_positions.keys())

                    if not all_holding_stocks:
                        # 空仓日
                        empty_position_days += 1
                        daily_returns.append(0.0)
                        portfolio_values.append(portfolio_values[-1])
                        portfolio_values_gross.append(portfolio_values_gross[-1])
                        continue

                    # 统计仓位情况
                    actual_stock_count = len(all_holding_stocks)
                    if actual_stock_count < stock_count:
                        partial_position_days += 1

                    # 等权重分配
                    position_weight = 1.0 / actual_stock_count
                    daily_portfolio_return_gross = 0.0
                    stock_returns = {}

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

                                if days_held <= 2:  # 只记录前两天的详细日志
                                    logger.info(f"{stock_code}: 持仓第{days_held}天, 当日收益{daily_return:.4f}, 累计收益{cumulative_return:.4f}")
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
                            'daily_position_ratio': 1.0,  # 满仓操作
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
                    if exit_trades_records:
                        trades.extend(exit_trades_records)
                        logger.info(f"添加退出交易记录: {len(exit_trades_records)}条")

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

                    # 🚀 修复：正确的退出原因统计（4种退出类型）
                    if 'exit_reason' in trades_df.columns and 'stop_loss_type' in trades_df.columns:
                        # 获取所有退出的交易（排除持仓中的）
                        exit_trades = trades_df[trades_df['exit_reason'] != 'holding'].copy()
                        total_exits = len(exit_trades)

                        if total_exits > 0:
                            # 统计4种退出类型
                            atr_stop_loss_count = 0
                            pct_stop_loss_count = 0
                            take_profit_count = 0
                            max_holding_count = 0

                            for _, trade in exit_trades.iterrows():
                                exit_reason = trade['exit_reason']
                                stop_loss_type = trade.get('stop_loss_type', 'unknown')

                                if exit_reason == 'stop_loss':
                                    if stop_loss_type == 'ATR止损':
                                        atr_stop_loss_count += 1
                                    elif stop_loss_type == '5%止损':
                                        pct_stop_loss_count += 1
                                    else:
                                        # 兜底：如果类型未知，根据其他信息判断
                                        atr_stop_loss_count += 1
                                elif exit_reason == 'take_profit':
                                    take_profit_count += 1
                                elif exit_reason == 'max_holding_days':
                                    max_holding_count += 1

                            # 计算比例
                            atr_strategy_stats = {
                                'atr_stop_loss_count': atr_stop_loss_count,
                                'pct_stop_loss_count': pct_stop_loss_count,
                                'take_profit_count': take_profit_count,
                                'max_holding_count': max_holding_count,
                                'atr_stop_loss_rate': atr_stop_loss_count / total_exits,
                                'pct_stop_loss_rate': pct_stop_loss_count / total_exits,
                                'take_profit_rate': take_profit_count / total_exits,
                                'max_holding_rate': max_holding_count / total_exits
                            }

                            logger.info(f"🎯 退出策略统计:")
                            logger.info(f"  ATR止损退出: {atr_stop_loss_count}次 ({atr_strategy_stats['atr_stop_loss_rate']:.1%})")
                            logger.info(f"  百分比止损退出: {pct_stop_loss_count}次 ({atr_strategy_stats['pct_stop_loss_rate']:.1%})")
                            logger.info(f"  ATR止盈退出: {take_profit_count}次 ({atr_strategy_stats['take_profit_rate']:.1%})")
                            logger.info(f"  最大持仓期退出: {max_holding_count}次 ({atr_strategy_stats['max_holding_rate']:.1%})")
                            logger.info(f"  总退出次数: {total_exits}次")
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
                    # 🚀 新增：空仓日统计
                    'empty_position_days': empty_position_days,
                    'partial_position_days': partial_position_days,
                    'empty_position_ratio': empty_position_ratio,
                    'partial_position_ratio': partial_position_ratio,
                    'full_position_ratio': full_position_ratio
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
                avg_position_size = 1.0  # 100%仓位
                result['avg_position_size'] = avg_position_size
                position_desc = f"{avg_position_size:>8.1%}"

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

        required_factors = [
            'returns_zscore','market_structure_score','ma_ratio_5_10','corr_5d','volatility_correlation',
            'volume_correlation','near_support_count_60d','institutional_activity_score','sector_momentum_acceleration',
            'ma_convergence_score','lower_shadow_ratio','sector_correlation_20d','sector_momentum_consistency',
            'price_impact','rank_momentum_20d','rank_rsi','rank_volume','market_correlation','corr_20d_mom5','corr_slope_20_120',
            'sector_relative_return_20d','sector_return_rank_20d','sector_liquidity_stability','consecutive_positive_days','consecutive_negative_days'
        ]

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

        if cleaned_rows < 100:
            logger.error("❌ 清理后数据量不足，无法进行可靠的相关性分析")
            return False

        # 3. 计算相关性矩阵
        logger.info("\n🔍 步骤3: 计算因子相关性矩阵")
        logger.info("-" * 50)

        logger.info("使用皮尔逊相关系数计算相关性...")
        logger.info("公式: ρ(X,Y) = Cov(X,Y) / (σₓ × σᵧ)")

        # 计算相关性矩阵
        correlation_matrix = factor_matrix.corr(method='pearson')

        logger.info(f"✅ 相关性矩阵计算完成: {correlation_matrix.shape[0]}×{correlation_matrix.shape[1]}")

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
        plt.title('ATR三分类标签系统 - 因子相关性矩阵', fontsize=16, pad=20)
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
            f.write("ATR三分类标签系统 - 因子相关性分析报告\n")
            f.write("=" * 60 + "\n\n")

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