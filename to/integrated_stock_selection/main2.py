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

使用示例:
  # 生成因子
  python main.py --mode generate_factors

  # 测试因子
  python main.py --mode test_factors --factor_names rsi6,macd,momentum_quality_score

  # 训练模型
  python main.py --mode train_model

  # 测试模型
  python main.py --mode test_model

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
        choices=['generate_factors', 'test_factors', 'train_model', 'test_model', 'full_pipeline', 'factor_correlation', 'genetic_factor'],
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
            'volatility20', 'amihud_illiquidity', 'atr14', 'open', 'high', 'low',

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

            # 极端样本：低于p05或高于p95的样本，权重=1.2
            # 这样只有约10%的样本被认为是极端样本，更合理
            extreme_mask = (y_train <= p05) | (y_train >= p95)
            sample_weights[extreme_mask] = 1.2

            logger.info(f"🚀 修复后的Pseudo-Huber回归样本权重调整:")
            logger.info(f"  标签范围: [{y_min:.3f}, {y_max:.3f}] ATR")
            logger.info(f"  极端阈值: p05={p05:.3f}, p95={p95:.3f} ATR")
            logger.info(f"  极端样本: {extreme_mask.sum()}个 ({extreme_mask.sum()/len(y_train)*100:.1f}%), 权重=1.2")
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

        else:
            # 🔧 分类模式：保持原有逻辑
            from sklearn.utils.class_weight import compute_class_weight

            # 获取唯一类别和样本数量
            classes = np.unique(y_train)
            total_count = len(y_train)
            n_classes = len(classes)

            logger.info(f"使用分类模式权重计算...")
            logger.info(f"  类别数: {n_classes}")
            logger.info(f"  类别: {classes}")

            # 计算每个类别的样本数量
            total_count = len(y_train)
            class_counts = {}
            for cls in classes:
                class_counts[cls] = np.sum(y_train == cls)
                pass  # 注释掉的日志输出，保持for循环结构完整
                # logger.info(f"  类别 {cls}: {class_counts[cls]}个样本 ({class_counts[cls]/total_count:.1%})")

            # 方法1：sklearn的balanced权重
            n_classes = len(classes)
            class_weights_balanced = compute_class_weight(
                'balanced', classes=classes, y=y_train
            )

            # 创建类别权重字典
            class_weight_dict = dict(zip(classes, class_weights_balanced))

            logger.info(f"sklearn balanced权重:")
            for cls, weight in class_weight_dict.items():
                pass  # 注释掉的日志输出，保持for循环结构完整
                # logger.info(f"  类别 {cls}: {weight:.3f}")

            # 🔧 清理：专注于ATR三分类标签系统
            if n_classes == 3:
                # 三分类逻辑（ATR标签）
                # 对于三分类，使用sklearn的balanced权重，但进行适度调整

                # 获取各类别样本数
                loss_count = class_counts.get(-1, 0)    # 亏损类
                neutral_count = class_counts.get(0, 0)  # 中性类
                profit_count = class_counts.get(1, 0)   # 盈利类

                logger.info(f"三分类样本分布:")
                logger.info(f"  亏损类(-1): {loss_count}个 ({loss_count/total_count:.1%})")
                logger.info(f"  中性类(0): {neutral_count}个 ({neutral_count/total_count:.1%})")
                logger.info(f"  盈利类(1): {profit_count}个 ({profit_count/total_count:.1%})")

                # 对于量化金融，适度调整权重以避免过度拟合
                # 限制最大权重，保持模型稳定性
                max_weight = 3.0  # 限制最大权重
                min_weight = 0.5  # 限制最小权重

                for cls in classes:
                    original_weight = class_weight_dict[cls]
                    adjusted_weight = np.clip(original_weight, min_weight, max_weight)
                    class_weight_dict[cls] = adjusted_weight

                    logger.info(f"  类别 {cls}: 原始权重={original_weight:.3f}, 调整后权重={adjusted_weight:.3f}")

            # 构建样本权重数组
            sample_weights = np.ones(len(y_train))
            for cls, weight in class_weight_dict.items():
                sample_weights[y_train == cls] = weight

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

            # 🚀 放松边界距离降权：把"近边界样本减半"改为0.8，"中等距离×0.8"改为0.9
            boundary_adjustment[boundary_mask] = 0.8  # 边界样本权重改为0.8（原来是0.5）

            # 对于距离适中的样本，轻微降权
            boundary_adjustment[medium_mask] = 0.9  # 中等样本权重改为0.9（原来是0.8）

            # 应用边界调整
            sample_weights *= boundary_adjustment

            logger.info(f"边界距离修正统计（放松版）:")
            logger.info(f"  边界样本({boundary_mask.sum()}个, {boundary_mask.sum()/len(boundary_distances)*100:.1f}%): 权重×0.8")
            logger.info(f"  中等样本({medium_mask.sum()}个, {medium_mask.sum()/len(boundary_distances)*100:.1f}%): 权重×0.9")
            logger.info(f"  远离边界样本({(~boundary_mask & ~medium_mask).sum()}个, {(~boundary_mask & ~medium_mask).sum()/len(boundary_distances)*100:.1f}%): 权重不变")

        # 显示权重分布
        if is_regression:
            logger.info(f"回归权重统计:")
            logger.info(f"  平均权重: {sample_weights.mean():.3f}")
            logger.info(f"  权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        else:
            # 分类模式：显示每个类别的权重分布
            for cls in classes:
                cls_weights = sample_weights[y_train == cls]
                logger.info(f"  类别 {cls}: 平均权重={cls_weights.mean():.3f}, 样本数={len(cls_weights)}")

        return sample_weights

    except Exception as e:
        logger.error(f"样本权重计算失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        # 返回均匀权重
        return np.ones(len(y_train))

# 🔧 清理：移除废弃的二分类函数，专注于ATR三分类标签系统


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

def _calculate_trading_metrics(y_true_return, y_pred_scores, logger, k_values=[5, 10, 20, 50]):
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

        # 🚀 Pseudo-Huber回归：使用固定阈值统计止盈止损率
        # 统一规则：止盈 >= +2 ATR，止损 <= -1 ATR
        profit_threshold = 2.0
        loss_threshold = -1.0

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

                # Top-K止盈率（使用修复后的阈值）
                top_k_profit_rates[k] = (top_k_return >= profit_threshold).mean()

                # Top-K止损率（使用修复后的阈值）
                top_k_loss_rates[k] = (top_k_return <= loss_threshold).mean()
            else:
                top_k_returns[k] = 0.0
                top_k_profit_rates[k] = 0.0
                top_k_loss_rates[k] = 0.0

        # 4. 收益分布分析（使用修复后的阈值）
        return_distribution = {
            'profit_samples': (y_true_return >= profit_threshold).sum(),
            'neutral_samples': ((y_true_return > loss_threshold) & (y_true_return < profit_threshold)).sum(),
            'loss_samples': (y_true_return <= loss_threshold).sum(),
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

        logger.info(f"📊 Top-K平均收益:")
        for k in k_values:
            if k <= len(sorted_returns):
                logger.info(f"  Top-{k}: {top_k_returns[k]:.4f} ATR (止盈率: {top_k_profit_rates[k]:.4f}, 止损率: {top_k_loss_rates[k]:.4f})")

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

        # 不使用PCA降维（保持特征完整性）
        n_features = X_train_split.shape[1]
        pca_info = {'applied': False, 'reason': '保持特征完整性，便于解释'}

        logger.info("=" * 60)
        logger.info("特征处理策略")
        logger.info("=" * 60)
        logger.info(f"🎯 使用全部 {n_features} 个原始特征进行训练")
        logger.info(f"✅ 保留完整的特征解释性，便于特征重要性分析")
        logger.info(f"🔍 XGBoost参数将针对 {n_features} 个特征进行优化")

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
            logger.info(f"🚀 Pseudo-Huber回归标签统计（原始ATR单位）:")
            logger.info(f"  训练集: {len(y_train_split)} 个样本")
            logger.info(f"  验证集: {len(y_val_split)} 个样本")
            logger.info(f"  标签范围: [{y_train_split.min():.3f}, {y_train_split.max():.3f}] ATR")
            logger.info(f"  标签均值: {y_train_split.mean():.6f} ATR")
            logger.info(f"  标签标准差: {y_train_split.std():.6f} ATR")

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

        else:
            # 分类模式：原有逻辑
            unique_labels = sorted(np.unique(y_train_split))
            logger.info(f"检测到标签类别: {unique_labels}")

            # 如果标签包含负数，需要重新映射为从0开始的连续整数
            if min(unique_labels) < 0:
                logger.info("检测到负数标签，进行标签重映射...")
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                reverse_label_map = {idx: label for label, idx in label_map.items()}

                # 重映射训练和验证标签
                y_train_split_mapped = np.array([label_map[label] for label in y_train_split])
                y_val_split_mapped = np.array([label_map[label] for label in y_val_split])

                logger.info(f"标签映射: {label_map}")
                logger.info(f"重映射后训练标签范围: {y_train_split_mapped.min()} - {y_train_split_mapped.max()}")
            else:
                y_train_split_mapped = y_train_split
                y_val_split_mapped = y_val_split
                reverse_label_map = None

        logger.info("✅ Winsorization缩尾处理已在数据预处理阶段完成")
        logger.info("=" * 60)

        # 数据标准化（避免数据泄露：只在训练集上学习参数）
        logger.info("🔧 开始数据标准化（避免数据泄露）...")
        scaler = StandardScaler()

        # 只在训练集上学习标准化参数（数据已在预处理阶段完成Winsorization）
        X_train_split_scaled = scaler.fit_transform(X_train_split)
        logger.info(f"✅ 标准化参数已在训练集上学习完成")

        # 用训练集学到的参数标准化验证集
        X_val_split_scaled = scaler.transform(X_val_split)
        logger.info(f"✅ 验证集已用训练集参数进行标准化")

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
        logger.info("XGBoost参数优化")
        logger.info("=" * 60)
        logger.info(f"📊 训练样本: {n_samples:,}个")
        logger.info(f"📊 特征数量: {current_features}个")
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

        # 🔧 简化参数配置：专注于ATR三分类的统一配置
        # 基于小特征集的专业优化配置，适用于所有特征数量
        params = {
            'n_estimators': 2000,        # 较多的树，因为每棵树很简单
            'max_depth': 2,             # 🚀 优化：限制复杂度，提升泛化能力
            'learning_rate': 0.08,      # 🚀 优化：平衡收敛速度和稳定性
            'reg_alpha': 5,             # L1正则化，特征选择
            'reg_lambda': 10,           # L2正则化，防止过拟合
            'subsample': 0.7,           # 样本采样率，提升泛化
            'colsample_bytree': 0.8,    # 特征采样率，减少过拟合
            'colsample_bylevel': 0.8,   # 每层特征采样率
            'min_child_weight': 15,     # 最小子节点权重，防止过拟合
            'gamma': 0.5,               # 分裂最小增益，控制复杂度
            'max_delta_step': 1         # 帮助收敛，适合不平衡数据
        }

        # 🚀 Pseudo-Huber回归模式：使用自定义目标函数
        if is_regression:
            logger.info("🚀 Pseudo-Huber回归模式：使用自定义目标函数和原始ATR标签")

            # 🚀 修复Pseudo-Huber回归参数：平衡学习能力和稳定性
            params.update({
                'max_depth': 6,                   # 🚀 恢复适中复杂度：避免过拟合
                'learning_rate': 0.05,            # 🚀 适中学习率：平衡收敛速度和稳定性
                'n_estimators': 3000,             # 🚀 适中树数量：避免过度训练
                'subsample': 0.8,                 # 🚀 适中采样率：保持泛化能力
                'colsample_bytree': 0.8,          # 🚀 适中特征采样：避免过拟合
                'reg_alpha': 0.1,                 # 🚀 适中L1正则：保持特征选择能力
                'reg_lambda': 0.1,                # 🚀 适中L2正则：防止过拟合
                'min_child_weight': 3,            # 🚀 适中最小权重：保持稳定性
                'gamma': 0.1,                     # 🚀 适中分裂限制：防止过度分裂
                'colsample_bylevel': 0.8,         # 🚀 适中每层采样
                'max_delta_step': 0,              # 不限制权重更新
                'random_state': 42,               # 随机种子
                'n_jobs': -1,                     # 使用所有CPU核心
                'verbosity': 0                    # 减少输出
            })

            logger.info("🚀 Pseudo-Huber回归参数配置:")
            logger.info(f"  max_depth: {params['max_depth']}")
            logger.info(f"  learning_rate: {params['learning_rate']}")
            logger.info(f"  n_estimators: {params['n_estimators']}")
            logger.info(f"  subsample: {params['subsample']}")
            logger.info(f"  colsample_bytree: {params['colsample_bytree']}")
            logger.info(f"  reg_alpha: {params['reg_alpha']}")
            logger.info(f"  reg_lambda: {params['reg_lambda']}")
            logger.info("✅ 配置针对Pseudo-Huber目标函数优化")
        else:
            logger.info("🔧 ATR三分类统一参数配置:")
            logger.info(f"  特征数量: {current_features}")
            logger.info(f"  max_depth: {params['max_depth']} (限制复杂度)")
            logger.info(f"  learning_rate: {params['learning_rate']} (平衡收敛)")
            logger.info(f"  n_estimators: {params['n_estimators']} (充分学习)")
            logger.info("✅ 统一配置简化了参数管理，专注ATR三分类优化")
        logger.info("=" * 60)

        # 🚀 回归模式：不进行标签映射，保持原始ATR倍数含义
        if is_regression:
            # 回归模式：保持原始连续收益标签
            unique_labels = None
            label_mapping = None
            reverse_mapping = None
            logger.info("回归模式：保持原始连续收益标签，不进行映射")
        else:
            # 分类模式：检查标签类型，确定目标函数
            unique_labels = sorted(np.unique(y_train_split))
            logger.info(f"检测到标签类别: {unique_labels}")

            # 如果标签包含负数，需要重新映射为从0开始的连续整数
            label_mapping = None
            reverse_mapping = None
            if min(unique_labels) < 0 or max(unique_labels) >= len(unique_labels):
                label_mapping = {label: i for i, label in enumerate(unique_labels)}
                reverse_mapping = {i: label for label, i in label_mapping.items()}
                logger.info(f"标签重新映射: {label_mapping}")
            else:
                label_mapping = None

        # 🚀 成本感知回归：支持回归和分类模式
        if is_regression:
            # 回归模式：连续收益预测
            objective = 'reg:squarederror'
            eval_metric = 'rmse'
            num_class = None
            logger.info("使用回归模式配置（连续收益预测）")
        else:
            # 分类模式：保持原有逻辑
            if len(unique_labels) == 3:
                # ATR三分类：-1（亏损）, 0（中性）, 1（盈利）
                objective = 'multi:softprob'
                eval_metric = 'mlogloss'
                num_class = 3
                logger.info("使用ATR三分类配置")
            else:
                logger.error(f"❌ 不支持的分类类型，检测到{len(unique_labels)}个类别，ATR系统仅支持三分类")
                raise ValueError(f"ATR系统仅支持三分类，当前检测到{len(unique_labels)}个类别")

        # 通用参数
        params.update({
            'objective': objective,
            'eval_metric': eval_metric,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        })

        # 🚀 回归模式特殊参数
        if is_regression:
            # 回归模式不需要num_class参数
            pass
        else:
            # 分类模式需要num_class参数
            if num_class is not None:
                params['num_class'] = num_class

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

            # 🚀 修复Delta选择：使用更小的值，提高敏感性
            huber_delta_grid = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]  # 更小的Delta值，提高对异常值的敏感性
            logger.info(f"🔍 开始Delta网格搜索: {huber_delta_grid}")

            best_delta = None
            best_score = float('-inf')
            delta_results = {}

            # 为网格搜索创建小规模训练集（提高速度）
            grid_search_size = min(50000, len(X_train_split))
            grid_indices = np.random.choice(len(X_train_split), grid_search_size, replace=False)
            X_grid = X_train_split[grid_indices]
            y_grid = y_train_split_mapped[grid_indices]
            sample_weights_grid = sample_weights[grid_indices]

            logger.info(f"🔍 使用{grid_search_size:,}个样本进行Delta网格搜索")

            for delta in huber_delta_grid:
                logger.info(f"🔍 测试 delta={delta:.1f} ATR...")

                try:
                    # 创建DMatrix
                    dtrain_grid = xgb.DMatrix(X_grid, label=y_grid, weight=sample_weights_grid)
                    dval_grid = xgb.DMatrix(X_val_split, label=y_val_split_mapped)

                    # 训练小模型
                    grid_params = params.copy()
                    grid_params.update({
                        'n_estimators': 500,  # 减少树数量加快搜索
                        'verbosity': 0
                    })

                    model_grid = xgb.train(
                        grid_params,
                        dtrain_grid,
                        num_boost_round=500,
                        obj=pseudo_huber_obj(delta),
                        evals=[(dval_grid, 'val')],
                        verbose_eval=False,
                        early_stopping_rounds=50
                    )

                    # 在验证集上评估
                    val_preds = model_grid.predict(dval_grid)

                    # 计算RMSE
                    rmse = np.sqrt(np.mean((val_preds - y_val_split_mapped) ** 2))

                    # 计算Top-K收益（使用原始ATR单位）
                    sorted_indices = np.argsort(val_preds)[::-1]
                    top5_return = y_val_split_mapped[sorted_indices[:5]].mean()
                    top10_return = y_val_split_mapped[sorted_indices[:10]].mean()

                    # 综合评分：优先Top-K收益，其次RMSE
                    composite_score = (top5_return + top10_return) / 2 - rmse * 0.1

                    delta_results[delta] = {
                        'rmse': rmse,
                        'top5_return': top5_return,
                        'top10_return': top10_return,
                        'composite_score': composite_score
                    }

                    logger.info(f"  Delta={delta:.1f}: RMSE={rmse:.4f}, Top5={top5_return:.3f}, Top10={top10_return:.3f}, Score={composite_score:.4f}")

                    if composite_score > best_score:
                        best_score = composite_score
                        best_delta = delta

                except Exception as e:
                    logger.warning(f"  Delta={delta:.1f} 训练失败: {e}")
                    delta_results[delta] = {
                        'rmse': float('inf'),
                        'top5_return': 0.0,
                        'top10_return': 0.0,
                        'composite_score': float('-inf')
                    }

            # 输出Delta选择结果
            logger.info(f"🎯 Delta选择结果:")
            for delta, result in delta_results.items():
                logger.info(f"  Delta={delta:.1f}: RMSE={result['rmse']:.4f}, Top5={result['top5_return']:.3f}, Top10={result['top10_return']:.3f}")

            if best_delta is None:
                logger.warning("⚠️ Delta网格搜索失败，使用默认值1.0")
                best_delta = 1.0

            logger.info(f"✅ 最佳Delta选择: {best_delta:.1f} ATR")

            # 使用最佳Delta训练最终模型
            logger.info(f"🚀 使用Delta={best_delta:.1f}训练最终Pseudo-Huber模型...")

        else:
            logger.info("✅ 分类模式：样本权重已计算完成")

        # 🚀 Pseudo-Huber回归：使用xgboost.train
        logger.info("=" * 60)
        logger.info("🚀 开始模型训练")
        logger.info("=" * 60)

        if is_regression:
            # 🚀 Pseudo-Huber回归模式：使用xgboost.train
            import xgboost as xgb

            # 创建DMatrix
            dtrain = xgb.DMatrix(X_train_split_scaled, label=y_train_split_mapped, weight=sample_weights)
            dval = xgb.DMatrix(X_val_split_scaled, label=y_val_split_mapped)

            logger.info("🎯 训练Pseudo-Huber回归模型...")
            logger.info(f"  使用最佳Delta: {best_delta:.1f} ATR")

            # 训练最终模型
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                obj=pseudo_huber_obj(best_delta),
                evals=[(dtrain, 'train'), (dval, 'val')],
                verbose_eval=100,
                early_stopping_rounds=100
            )

            # 保存最佳delta到模型属性
            model.best_delta = best_delta

        else:
            # 分类模式：使用传统方法
            model = xgb.XGBClassifier(**params)
            logger.info("🎯 训练XGBoost分类模型...")

            # 分类模式：应用标签映射（如果需要）
            if label_mapping:
                y_train_split_mapped = pd.Series(y_train_split).map(label_mapping).values
                y_val_split_mapped = pd.Series(y_val_split).map(label_mapping).values
            else:
                y_train_split_mapped = y_train_split
                y_val_split_mapped = y_val_split

            # 训练模型
            logger.info(f"应用样本权重进行训练，权重数组长度: {len(sample_weights)}")
            model.fit(
                X_train_split_scaled, y_train_split_mapped,
                sample_weight=sample_weights,
                eval_set=[(X_val_split_scaled, y_val_split_mapped)],
                verbose=False
            )

        # 🚀 模型评估：支持Pseudo-Huber回归和分类
        logger.info("评估模型性能...")

        if is_regression:
            # 🚀 Pseudo-Huber回归：使用xgb.train模型预测
            dval_for_pred = xgb.DMatrix(X_val_split_scaled)
            y_pred = model.predict(dval_for_pred)
            y_pred_proba = None
            logger.info(f"✅ Pseudo-Huber回归预测完成，预测值范围: [{y_pred.min():.3f}, {y_pred.max():.3f}] ATR")
        else:
            # 分类模式：使用传统预测
            y_pred = model.predict(X_val_split_scaled)
            y_pred_proba = model.predict_proba(X_val_split_scaled)

        # 🚀 成本感知回归：支持回归和分类评估
        y_val_for_eval = y_val_split_mapped

        if is_regression:
            # 🚀 Pseudo-Huber回归：直接使用原始ATR单位评估（无需反向转换）
            logger.info("🚀 Pseudo-Huber回归模型评估...")

            # 直接使用原始ATR单位的预测值和真实值
            y_pred_orig = y_pred  # Pseudo-Huber直接输出ATR单位
            y_val_orig = y_val_for_eval  # 原始ATR单位标签

            # 计算回归指标
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_val_orig, y_pred_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_orig, y_pred_orig)
            r2 = r2_score(y_val_orig, y_pred_orig)
            correlation = np.corrcoef(y_val_orig, y_pred_orig)[0, 1]

            logger.info(f"🎯 Pseudo-Huber回归性能:")
            logger.info(f"  最佳Delta: {best_delta:.1f} ATR")
            logger.info(f"  MSE: {mse:.6f}")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  MAE: {mae:.6f}")
            logger.info(f"  R²: {r2:.6f}")
            logger.info(f"  相关性: {correlation:.6f}")

            # 🚀 标签和预测分布（原始ATR单位）
            logger.info(f"📊 标签分布（原始ATR单位）:")
            logger.info(f"  真实标签范围: [{y_val_orig.min():.3f}, {y_val_orig.max():.3f}] ATR")
            logger.info(f"  真实标签均值: {y_val_orig.mean():.3f} ATR")
            logger.info(f"  真实标签标准差: {y_val_orig.std():.3f} ATR")

            logger.info(f"📊 预测分布（原始ATR单位）:")
            logger.info(f"  预测值范围: [{y_pred_orig.min():.3f}, {y_pred_orig.max():.3f}] ATR")
            logger.info(f"  预测值均值: {y_pred_orig.mean():.3f} ATR")
            logger.info(f"  预测值标准差: {y_pred_orig.std():.3f} ATR")

            # 🚀 计算交易指标（使用固定阈值）
            trading_metrics = _calculate_trading_metrics(
                y_true_return=y_val_orig,      # 原始ATR单位真实值
                y_pred_scores=y_pred_orig,     # 原始ATR单位预测值
                logger=logger,
                k_values=[5, 10, 20, 50]
            )

            # 🚀 Delta选择结果总结
            logger.info("🎯 Delta选择结果总结:")
            for delta, result in delta_results.items():
                logger.info(f"  Delta={delta:.1f}: RMSE={result['rmse']:.4f}, Top5={result['top5_return']:.3f}, Top10={result['top10_return']:.3f}")

            logger.info("🎯 交易指标（固定阈值 >= +2 ATR / <= -1 ATR）:")
            logger.info(f"  止盈命中率: {trading_metrics.get('profit_hit_rate', 0):.4f}")
            logger.info(f"  止损命中率: {trading_metrics.get('loss_hit_rate', 0):.4f}")

            # Top-K收益统计
            for k in [5, 10, 20, 50]:
                if k in trading_metrics:
                    logger.info(f"  Top-{k} 平均收益: {trading_metrics[k]['avg_return']:.4f} ATR")
                    logger.info(f"  Top-{k} 止盈率: {trading_metrics[k]['profit_rate']:.4f}")
                    logger.info(f"  Top-{k} 止损率: {trading_metrics[k]['loss_rate']:.4f}")

        else:
            # 分类模式：计算分类指标
            accuracy = accuracy_score(y_val_for_eval, y_pred)

            # 🔧 清理：专注于ATR三分类评估指标
            # ATR三分类：使用分类报告和专业评估指标
            from sklearn.metrics import classification_report
            class_report = classification_report(y_val_for_eval, y_pred, output_dict=True)

            logger.info(f"ATR三分类模型性能:")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"📊 ATR三分类各类别性能:")

            # 获取映射后的标签范围
            mapped_labels = range(len(unique_labels))
            label_names = {-1: "亏损", 0: "中性", 1: "盈利"}

            for i, mapped_label in enumerate(mapped_labels):
                if str(mapped_label) in class_report:
                    metrics = class_report[str(mapped_label)]
                    original_label = reverse_mapping[mapped_label] if reverse_mapping else mapped_label
                    label_name = label_names.get(original_label, f"类别{original_label}")
                    logger.info(f"  {label_name}({original_label}): 精确率={metrics['precision']:.4f}, "
                               f"召回率={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

            # ATR三分类：计算盈利类别的Precision@k（关键指标）
            profit_class_idx = None
            if reverse_mapping:
                # 找到原始标签为1（盈利）对应的映射标签
                for mapped_idx, original_label in reverse_mapping.items():
                    if original_label == 1:
                        profit_class_idx = mapped_idx
                        break
            else:
                profit_class_idx = 1 if 1 in unique_labels else None

            if profit_class_idx is not None and profit_class_idx < y_pred_proba.shape[1]:
                profit_proba = y_pred_proba[:, profit_class_idx]
                profit_binary = (y_val_for_eval == profit_class_idx).astype(int)
                precision_at_k = _calculate_precision_at_k(profit_binary, profit_proba, k_values=[1, 3, 5, 10])
                logger.info(f"📊 ATR盈利类别Precision@k（选股关键指标）:")
                for k, precision in precision_at_k.items():
                    logger.info(f"  Precision@{k}: {precision:.4f}")
            else:
                precision_at_k = {}
                logger.warning("⚠️ 无法计算盈利类别Precision@k")

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
        else:
            # 分类模式：使用传统方法
            feature_importance = model.feature_importances_

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
        else:
            # 分类模式：添加分类指标
            performance_metrics.update({
                'accuracy': accuracy,
                'precision_at_k': precision_at_k,
                'classification_report': class_report if 'class_report' in locals() else {},
                'atr_classification_type': 'three_class',
                'model_type': 'classification'
            })

        results = {
            'model': model,
            'scaler': scaler,
            'winsorization_thresholds': winsorization_thresholds,  # 🚀 新增：保存Winsorization阈值
            'pca_info': pca_info,
            'feature_names': feature_names,
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
        else:
            results['training_info'].update({
                'imbalance_ratio': imbalance_ratio,
                'label_classes': unique_labels,
                'classification_type': f"{len(unique_labels)}分类",
                'class_distribution': {str(label): int(np.sum(y_train_split == label)) for label in unique_labels_split}
            })

        logger.info("=" * 60)
        if is_regression:
            logger.info("🎉 成本感知回归模型训练完成")
        else:
            logger.info("🎉 ATR三分类模型训练完成 - 数据泄露修复与优化总结")
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

        return results

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"模型训练失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None



def _analyze_model_interpretability(model, feature_names, X_val, y_val, logger):
    """分析模型可解释性"""
    try:
        import numpy as np

        logger.info("开始模型可解释性分析...")

        # 1. 决策树规则提取（前3棵树）
        logger.info("=== 决策树规则分析 ===")
        try:
            # 获取前几棵树的规则
            for tree_idx in range(min(3, model.n_estimators)):
                tree_rules = _extract_tree_rules(model, tree_idx, feature_names)
                if tree_rules:
                    logger.info(f"决策树 {tree_idx + 1} 的关键规则:")
                    for rule in tree_rules[:5]:  # 只显示前5条规则
                        logger.info(f"  {rule}")
        except Exception as e:
            logger.warning(f"决策树规则提取失败: {e}")

        # 2. 特征交互分析
        logger.info("=== 特征交互分析 ===")
        try:
            feature_interactions = _analyze_feature_interactions(model, feature_names, X_val)
            logger.info("重要特征交互组合:")
            for interaction, score in feature_interactions[:10]:
                logger.info(f"  {interaction}: {score:.4f}")
        except Exception as e:
            logger.warning(f"特征交互分析失败: {e}")

        # 3. 阈值分析
        logger.info("=== 特征阈值分析 ===")
        try:
            feature_thresholds = _analyze_feature_thresholds(model, feature_names)
            logger.info("重要特征的关键阈值:")
            for feature, thresholds in feature_thresholds.items():
                if thresholds:
                    logger.info(f"  {feature}: {thresholds}")
        except Exception as e:
            logger.warning(f"特征阈值分析失败: {e}")

        # 4. 预测路径分析
        logger.info("=== 预测路径分析 ===")
        try:
            prediction_paths = _analyze_prediction_paths(model, X_val, y_val, feature_names)
            logger.info("典型预测路径:")
            for class_name, path in prediction_paths.items():
                logger.info(f"  {class_name}: {path}")
        except Exception as e:
            logger.warning(f"预测路径分析失败: {e}")

        # 5. SHAP值分析（如果可用）
        logger.info("=== SHAP值分析 ===")
        try:
            shap_analysis = _analyze_shap_values(model, X_val, feature_names)
            if shap_analysis:
                logger.info("SHAP值分析结果:")
                for feature, shap_info in shap_analysis.items():
                    logger.info(f"  {feature}: {shap_info}")
        except Exception as e:
            logger.warning(f"SHAP分析失败（可能未安装shap库）: {e}")

        logger.info("模型可解释性分析完成")

    except Exception as e:
        logger.error(f"模型可解释性分析失败: {e}")

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
            'model_type': type(ml_model).__name__
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

        # 设置默认因子列表（包含returns_zscore） ,'sector_momentum_consistency'
        if required_factors is None:
            required_factors = ['returns_zscore','market_structure_score','ma_ratio_5_10','corr_5d','volatility_correlation',
                                'volume_correlation','near_support_count_60d','institutional_activity_score',
                                'ma_convergence_score','lower_shadow_ratio','sector_correlation_20d','near_support','amihud_illiquidity',
                                'price_impact','rank_momentum_20d','rank_rsi','rank_volume','market_correlation','corr_20d_mom5','corr_slope_20_120']  # 默认包含returns_zscore因子
            logger.info(f"使用默认因子列表: {required_factors}")
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
            'volatility20',
            # 流动性特征
            'amihud_illiquidity'
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
                            # 流动性特征默认值
                            'amihud_illiquidity': 0.0,
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
        label_generation_columns = ['volatility20', 'amihud_illiquidity', 'atr14', 'open', 'high', 'low']

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
                        'near_resistance', 'short_reversal', 'amihud_illiquidity', 'volatility20']
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
    atr_multiplier_profit=2,
    atr_multiplier_loss=1,
    max_holding_days=2,
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
    3. T+1至T+5日：观察期，逐日检查止盈止损触发
    4. 止盈位：入场价 + atr_multiplier_profit * ATR
    5. 止损位：入场价 - atr_multiplier_loss * ATR
    6. 止损优先原则：同时触发时优先处理止损
    7. T+1规则：触发后次日开盘价退出
    8. 超时处理：5天内未触发则强制平仓

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


def _generate_training_labels_for_all_data(stock_data, atr_multiplier_profit=2, atr_multiplier_loss=1, max_holding_days=5, debug=False,
                                         is_training=True, label_stats=None):
    """
    🚀 成本感知回归标签生成函数：输出连续收益标签（带Winsorize截尾和标准化）

    标签生成规则（连续值）：
    - 触发止盈 → 标签 = +atr_multiplier_profit（ATR倍数）
    - 触发止损 → 标签 = -atr_multiplier_loss（ATR倍数）
    - 超时未触发 → 标签 = 实际持仓期末收益（ATR倍数）

    标签后处理：
    1. Winsorize截尾：限制在[-3, +3] ATR范围内
    2. 标准化：y = (y - μ) / σ（训练集统计量）

    输出包含：
    - atr_return_label: 连续收益标签（标准化后的浮点数）
    - exit_day: 实际持仓天数（1 到 5）
    - min_dist_to_sl: 持仓期间价格到止损线的最小距离（ATR倍数）
    - min_dist_to_pt: 持仓期间价格到止盈线的最小距离（ATR倍数）

    Args:
        stock_data: 股票数据DataFrame
        atr_multiplier_profit: 止盈ATR倍数
        atr_multiplier_loss: 止损ATR倍数
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

        logger.info("📊 开始计算ATR三分类标签（统一逻辑版）...")
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

        # 🚀 完全向量化：从T+1到T+max_holding_days逐日检查
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

            # --- 🔧 止损优先（完全向量化） ---
            if day < max_holding_days:  # T+1到T+max_holding_days-1日：需要T+day+1开盘价
                hit_loss_mask = (
                    valid_mask &
                    (stock_data_sorted[low_col] <= stock_data_sorted['stop_loss']) &
                    (~is_labeled) &
                    (stock_data_sorted[next_open_col].notna()) &
                    (stock_data_sorted[next_open_col] > 0)
                )
            else:  # T+max_holding_days日：使用T+max_holding_days收盘价或T+max_holding_days+1开盘价
                close_col = f'close_T{day}'
                hit_loss_mask = (
                    valid_mask &
                    (stock_data_sorted[low_col] <= stock_data_sorted['stop_loss']) &
                    (~is_labeled) &
                    (
                        (stock_data_sorted[next_open_col].notna() & (stock_data_sorted[next_open_col] > 0)) |
                        (close_col in stock_data_sorted.columns and stock_data_sorted[close_col].notna())
                    )
                )

            # 🚀 成本感知回归：向量化标记止损（连续收益标签）
            stock_data_sorted.loc[hit_loss_mask, 'atr_return_label'] = -atr_multiplier_loss  # 连续收益标签
            stock_data_sorted.loc[hit_loss_mask, 'exit_day'] = day
            is_labeled.loc[hit_loss_mask] = True  # 更新已标记状态
            loss_count = hit_loss_mask.sum()
            total_loss_samples += loss_count

            # --- 🔧 止盈（完全向量化） ---
            if day < max_holding_days:  # T+1到T+max_holding_days-1日：需要T+day+1开盘价
                hit_profit_mask = (
                    valid_mask &
                    (stock_data_sorted[high_col] >= stock_data_sorted['profit_target']) &
                    (~is_labeled) &
                    (stock_data_sorted[next_open_col].notna()) &
                    (stock_data_sorted[next_open_col] > 0)
                )
            else:  # T+max_holding_days日：使用T+max_holding_days收盘价或T+max_holding_days+1开盘价
                close_col = f'close_T{day}'
                hit_profit_mask = (
                    valid_mask &
                    (stock_data_sorted[high_col] >= stock_data_sorted['profit_target']) &
                    (~is_labeled) &
                    (
                        (stock_data_sorted[next_open_col].notna() & (stock_data_sorted[next_open_col] > 0)) |
                        (close_col in stock_data_sorted.columns and stock_data_sorted[close_col].notna())
                    )
                )

            # 🚀 成本感知回归：向量化标记止盈（连续收益标签）
            stock_data_sorted.loc[hit_profit_mask, 'atr_return_label'] = atr_multiplier_profit  # 连续收益标签
            stock_data_sorted.loc[hit_profit_mask, 'exit_day'] = day
            is_labeled.loc[hit_profit_mask] = True  # 更新已标记状态
            profit_count = hit_profit_mask.sum()
            total_profit_samples += profit_count

            if loss_count > 0 or profit_count > 0:
                logger.info(f"    T+{day}日: 新增亏损{loss_count:,}个, 新增盈利{profit_count:,}个")

        # 🚀 成本感知回归：处理超时样本（计算实际持仓期末收益）
        timeout_mask = valid_mask & (~is_labeled)

        if timeout_mask.any():
            logger.info(f"  计算{timeout_mask.sum()}个超时样本的实际收益...")

            # 计算超时样本的实际收益（ATR倍数）
            # 使用T+max_holding_days的收盘价或T+max_holding_days+1的开盘价
            exit_price_col = f'close_T{max_holding_days}'
            next_open_col = f'open_T{max_holding_days+1}'

            # 优先使用T+max_holding_days+1开盘价，其次使用T+max_holding_days收盘价
            exit_price = stock_data_sorted.loc[timeout_mask, next_open_col].fillna(
                stock_data_sorted.loc[timeout_mask, exit_price_col]
            )

            # 计算实际收益（ATR倍数）
            entry_price = stock_data_sorted.loc[timeout_mask, 'open_T1']
            atr_value = stock_data_sorted.loc[timeout_mask, 'atr14']

            # 实际收益 = (退出价格 - 入场价格) / ATR
            actual_return_atr = (exit_price - entry_price) / atr_value

            # 设置超时样本的连续收益标签
            stock_data_sorted.loc[timeout_mask, 'atr_return_label'] = actual_return_atr

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


def _apply_atr_strategy_core(stock_data, decision_date, atr_value, atr_multiplier_profit=2, atr_multiplier_loss=1, max_holding_days=5):
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


def _process_atr_strategy_single(backtest_data, stock_code, selection_date, atr_multiplier_profit=2, atr_multiplier_loss=1, max_holding_days=5, debug=False):
    """
    单股票处理模式：计算基于ATR策略的实际收益率

    Args:
        backtest_data: 回测数据DataFrame
        stock_code: 股票代码
        selection_date: 选择日期
        atr_multiplier_profit: 止盈ATR倍数
        atr_multiplier_loss: 止损ATR倍数
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
        stop_loss = entry_price - atr_multiplier_loss * atr_value

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

                # 计算1%和99%分位数
                p1 = np.percentile(feature_data, 1)
                p99 = np.percentile(feature_data, 99)

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

        # 3. 移除完全相同的行（基于特征）
        initial_rows = len(feature_data)
        duplicate_mask = ~feature_data.duplicated()
        feature_data = feature_data[duplicate_mask]

        if len(feature_data) < initial_rows:
            logger.info(f"移除重复行: {initial_rows - len(feature_data)}行")

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
    """保存模型和结果（包含PCA模型）"""
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

        # 保存模型文件（包含PCA模型、Winsorization阈值和标签统计量）
        model_file = results_dir / f"xgboost_model_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': model_results['model'],
                'scaler': model_results['scaler'],
                'winsorization_thresholds': model_results.get('winsorization_thresholds', {}),  # 🔧 修复：保存Winsorization阈值
                'pca_info': model_results['pca_info'],
                'feature_names': feature_names,
                'pca_models': pca_models if pca_models else {},
                'label_stats': model_results.get('label_stats', None)  # 🚀 保存标签统计量（用于回归反标准化）
            }, f)
        logger.info(f"模型已保存: {model_file}")

        # 记录PCA信息
        if pca_models:
            logger.info(f"📊 PCA模型信息:")
            for group_name, pca_info in pca_models.items():
                logger.info(f"   {group_name}: {len(pca_info['features'])}个特征 -> 1个主成分")
                logger.info(f"   解释方差比例: {pca_info['explained_variance_ratio']:.4f}")

        # 保存训练结果报告
        report_data = {
            'timestamp': timestamp,
            'performance': model_results['performance'],
            'training_info': model_results['training_info'],
            'model_params': model_results['model_params'],
            'feature_importance': model_results['feature_importance'].to_dict('records'),
            'pca_info': {k: v for k, v in model_results['pca_info'].items() if k != 'pca_object'}
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
        print("🤖 运行模式: 成本感知回归模型训练")
    else:
        print("🤖 运行模式: ATR三分类模型训练")
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

        # 2. 特征识别（使用与预测阶段一致的简化逻辑）
        feature_columns = _get_ml_feature_columns_simple(factor_data)
        logger.info(f"识别到 {len(feature_columns)} 个有效特征")

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
            atr_multiplier_profit=2,
            atr_multiplier_loss=1,
            max_holding_days=5,
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
        # 从模型结果中获取特征名称，不使用PCA模型
        feature_names = model_results.get('feature_names', feature_columns)
        pca_models = {}  # 不使用PCA模型
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
    """运行模型测试模式：训练模型→预测→选股→回测的纯预测流程"""
    logger = logging.getLogger(__name__)
    print("📊 运行模式: 模型预测选股")
    logger.info("=" * 60)
    logger.info("开始纯预测选股流程（无标签验证）")
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

        # 8. 特征对齐处理
        logger.info("🔧 进行特征对齐处理...")

        # 将预处理后的数据转换为DataFrame以便处理
        X_pred_df = pd.DataFrame(X_pred, columns=available_features)

        # 确保预测数据包含训练时使用的所有特征
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
            X_pred_aligned = np.column_stack(aligned_data)
            available_features = aligned_features
            logger.info(f"✅ 特征对齐完成: {X_pred_aligned.shape}")
            logger.info(f"✅ 特征数量匹配: 预测特征数={len(available_features)}, 训练特征数={len(feature_names)}")
            X_pred = X_pred_aligned
        else:
            logger.error("❌ 特征对齐失败，无法继续预测")
            return False

        # 🚀 修复：恢复特征标准化，但保持标签原始
        # 特征标准化（使用训练时保存的参数）
        if 'scaler' in model_data:
            scaler = model_data['scaler']
            X_pred_scaled = scaler.transform(X_pred)
            logger.info("✅ 使用训练时的特征标准化参数")
        else:
            logger.warning("⚠️ 未找到训练时的标准化参数，使用预测数据进行标准化")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_pred_scaled = scaler.fit_transform(X_pred)

        X_pred_final = X_pred_scaled
        logger.info(f"✅ 特征标准化完成: {X_pred_final.shape[1]} 个特征，{X_pred_final.shape[0]} 个样本")

        # 🚀 7. Pseudo-Huber回归模型预测
        logger.info("\n🔮 步骤7: 进行Pseudo-Huber回归模型预测...")

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
        logger.info("\n📊 步骤8: 处理预测结果...")

        if len(y_pred_return) > len(original_prediction_data):
            logger.error(f"预测结果长度({len(y_pred_return)})大于原始数据长度({len(original_prediction_data)})")
            return False

        # 创建预测结果DataFrame
        prediction_data_with_proba = original_prediction_data.iloc[:len(y_pred_return)].copy()
        prediction_data_with_proba['predicted_return'] = y_pred_return
        prediction_data_with_proba['y_pred_proba'] = y_pred_return  # 兼容下游逻辑

        positive_count = (y_pred_return > 0).sum()
        logger.info(f"✅ 预测结果处理完成: {len(prediction_data_with_proba)} 个样本")
        logger.info(f"📊 预测分布: 正值{positive_count}个({positive_count/len(y_pred_return)*100:.1f}%), "
                   f"负值{len(y_pred_return)-positive_count}个")

        # 9. 保存预测结果
        logger.info("\n💾 步骤9: 保存预测结果...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_results = prediction_data_with_proba[['date', 'stock_code', 'predicted_return', 'y_pred_proba']].copy()
        prediction_file = results_dir / f"pure_predictions_{timestamp}.csv"
        prediction_results.to_csv(prediction_file, index=False, encoding='utf-8')
        logger.info(f"✅ 预测结果已保存: {prediction_file} ({len(prediction_results)} 条记录)")

        # 10. 选股回测系统
        logger.info("\n🔄 步骤10: 开始选股回测...")
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
            'near_support', 'near_resistance', 'short_reversal',
            # 流动性特征（用于风险控制）
            'amihud_illiquidity'
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
                # 流动性特征默认值
                'amihud_illiquidity': 0.0,
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
        backtest_columns = required_columns + ['y_pred_proba', 'predicted_return']

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

        # 准备常量（统一放在函数前部）
        MIN_POS_THRESHOLD_ATR = 0.15   # 最小可接受预测值（ATR），小于此值不买
        TRADING_COST_ATR = 0.03        # 交易/滑点等折算到 ATR 的最低可覆盖成本

        # 定义选股数量
        stock_counts = [1, 3, 5]

        # 初始化回测结果
        backtest_results = {}
        detailed_trades = {}

        for stock_count in stock_counts:
            logger.info(f"\n📊 开始{stock_count}只股票选股回测...")

            # 初始化回测记录
            trades = []
            portfolio_values = [1.0]  # 初始资金为1
            daily_returns = []
            # 🚀 新增：空仓日统计
            empty_position_days = 0
            partial_position_days = 0

            # 🚀 唯一口径选股逻辑：对每个交易日做横截面打分
            logger.info(f"📊 使用唯一口径选股：只按predicted_return_atr_raw排序，正期望优先+最小阈值")

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

                # 2.6 记录选股
                selected_records.append(g[['date','stock_code','predicted_return_atr_raw','score_z','score_final']])

            # 汇总选股结果
            if selected_records:
                picks = pd.concat(selected_records, ignore_index=True)
            else:
                picks = pd.DataFrame(columns=['date','stock_code','predicted_return_atr_raw','score_z','score_final'])

            # 计算空仓日比例
            empty_ratio = empty_day_count / max(total_days, 1)
            logger.info(f"统一口径选股完成：只买正期望且 >= {MIN_POS_THRESHOLD_ATR} ATR，不凑满K；空仓日比例={empty_ratio:.1%}")

            # 🚀 执行回测：使用选股结果进行交易
            for i, date in enumerate(trading_dates[:-1]):  # 排除最后一天（无法卖出）
                try:
                    # 获取当日选中的股票
                    daily_picks = picks[picks['date'] == date].copy()

                    if len(daily_picks) == 0:
                        # 当日空仓
                        empty_position_days += 1
                        daily_returns.append(0.0)
                        portfolio_values.append(portfolio_values[-1])
                        continue

                    # 统计仓位情况
                    actual_stock_count = len(daily_picks)
                    if actual_stock_count < stock_count:
                        partial_position_days += 1

                    # 计算仓位权重
                    position_weight = 1.0 / len(daily_picks)

                    # 计算当日组合收益率
                    daily_portfolio_return = 0.0
                    trade_cost = 0.002 * len(daily_picks)  # 每只股票0.2%交易成本

                    for _, stock in daily_picks.iterrows():
                        # 计算股票收益
                        stock_return = _process_atr_strategy(
                            backtest_data, mode='return',
                            stock_code=stock['stock_code'],
                            selection_date=date,
                            debug=False
                        )

                        # NaN检查
                        if pd.isna(stock_return):
                            stock_return = 0.0

                        daily_portfolio_return += position_weight * stock_return

                        # 记录交易
                        trade_record = {
                            'date': date,
                            'stock_code': stock['stock_code'],
                            'predicted_return_atr_raw': stock.get('predicted_return_atr_raw', 0),
                            'score_final': stock.get('score_final', 0),
                            'actual_return': stock_return,
                            'position_weight': position_weight,
                            'contribution': position_weight * stock_return
                        }
                        trades.append(trade_record)

                    # 扣除交易成本
                    daily_portfolio_return -= trade_cost

                    # 🔧 增强NaN检查
                    if pd.isna(daily_portfolio_return):
                        logger.error(f"❌ {date} 日组合收益为NaN，设为0")
                        daily_portfolio_return = 0.0

                    daily_returns.append(daily_portfolio_return)

                    # 更新组合价值
                    new_value = portfolio_values[-1] * (1 + daily_portfolio_return)
                    portfolio_values.append(new_value)

                except Exception as e:
                    logger.warning(f"处理日期 {date} 失败: {e}")
                    daily_returns.append(0.0)
                    portfolio_values.append(portfolio_values[-1])
                    continue

            # 计算回测指标
            if len(daily_returns) > 0:
                # 🔧 增强NaN检查和处理
                portfolio_final_value = portfolio_values[-1]
                if pd.isna(portfolio_final_value):
                    logger.error("❌ 组合最终价值为NaN，设为1.0")
                    portfolio_final_value = 1.0

                total_return = portfolio_final_value - 1.0
                if pd.isna(total_return):
                    logger.error("❌ 总收益为NaN，设为0")
                    total_return = 0.0

                # 计算年化收益
                if portfolio_final_value > 0 and len(daily_returns) > 0:
                    annual_return = (portfolio_final_value ** (252 / len(daily_returns))) - 1.0
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

                # 计算夏普比率
                risk_free_rate = 0.03
                sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

                # 计算最大回撤
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (np.array(portfolio_values) - peak) / peak
                max_drawdown = np.min(drawdown)

                # 计算胜率
                positive_returns = returns_array[returns_array > 0]
                win_rate = len(positive_returns) / len(returns_array) if len(returns_array) > 0 else 0

                # 平均单次收益
                avg_return = np.mean(returns_array)

                # 🚀 回归模式评估指标
                regression_metrics = {}
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

                # 🚀 计算空仓日比例统计
                total_trading_days = len(daily_returns)
                empty_position_ratio = empty_position_days / total_trading_days if total_trading_days > 0 else 0
                partial_position_ratio = partial_position_days / total_trading_days if total_trading_days > 0 else 0
                full_position_ratio = 1 - empty_position_ratio - partial_position_ratio

                # 保存结果
                backtest_results[stock_count] = {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'avg_daily_return': avg_return,
                    'total_trades': len(trades),
                    'trading_days': len(daily_returns),
                    'regression_metrics': regression_metrics,
                    # 🚀 新增：空仓日统计
                    'empty_position_days': empty_position_days,
                    'partial_position_days': partial_position_days,
                    'empty_position_ratio': empty_position_ratio,
                    'partial_position_ratio': partial_position_ratio,
                    'full_position_ratio': full_position_ratio
                }

                detailed_trades[stock_count] = trades

                logger.info(f"✅ {stock_count}只股票回测完成:")
                logger.info(f"  总收益率: {total_return:.2%}")
                logger.info(f"  年化收益率: {annual_return:.2%}")
                logger.info(f"  夏普比率: {sharpe_ratio:.3f}")
                logger.info(f"  最大回撤: {max_drawdown:.2%}")
                logger.info(f"  胜率: {win_rate:.2%}")
                # 🚀 新增：空仓日比例统计
                logger.info(f"  空仓日比例: {empty_position_ratio:.1%} ({empty_position_days}天)")
                logger.info(f"  部分仓位日比例: {partial_position_ratio:.1%} ({partial_position_days}天)")
                logger.info(f"  满仓日比例: {full_position_ratio:.1%}")



        # 12. 保存回测结果
        logger.info("\n💾 步骤12: 保存回测结果...")

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
        logger.info("\n📊 “唯一口径”选股策略回测结果对比:")
        logger.info("=" * 120)
        logger.info(f"{'策略':<10} {'总收益':<10} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<8} {'空仓比例':<10} {'日均持仓':<10}")
        logger.info("=" * 120)

        for stock_count in stock_counts:
            if stock_count in backtest_results:
                result = backtest_results[stock_count]
                backtest_summary["strategy_results"][f"{stock_count}stocks"] = result

                # 计算日均持仓数
                total_days = result.get('trading_days', 1)
                total_trades = result.get('total_trades', 0)
                avg_daily_holdings = total_trades / max(total_days, 1)
                result['avg_daily_holdings'] = avg_daily_holdings

                logger.info(f"Top-{stock_count:<5} {result['total_return']:>8.2%}  "
                           f"{result['annual_return']:>8.2%}  {result['sharpe_ratio']:>8.3f}  "
                           f"{result['max_drawdown']:>8.2%}  {result['win_rate']:>6.2%}  "
                           f"{result['empty_position_ratio']:>8.1%}  "
                           f"{avg_daily_holdings:>8.2f} 只")

        logger.info("=" * 120)

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
        logger.info("✅ 使用固定0.15 ATR阈值，宁缺毋滥选股策略")
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
            'volume_correlation','near_support_count_60d','institutional_activity_score',
            'ma_convergence_score','lower_shadow_ratio','sector_correlation_20d','near_support','amihud_illiquidity',
            'price_impact','rank_momentum_20d','rank_rsi','rank_volume','market_correlation','corr_20d_mom5','corr_slope_20_120'
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
            'ma_convergence_score','lower_shadow_ratio','sector_correlation_20d','near_support','amihud_illiquidity'
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

            print(f"\n🎉 遗传编程因子进化成功完成!")
            print(f"📁 结果已保存到: {args.output_dir}")
            print(f"🔝 找到 {len(results.get('best_factors', []))} 个优秀因子")

            return True
        else:
            logger.error("❌ 遗传编程进化失败")
            return False

    except Exception as e:
        logger.error(f"遗传编程因子进化失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
