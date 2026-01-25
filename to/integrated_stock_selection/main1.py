#!/usr/bin/env python3
"""
机器学习K线形态选股系统主程序

支持三种运行模式：
1. 特征计算模式 (calculate_features) - 计算缺失特征
2. 模型训练模式 (train_model) - 训练机器学习模型
3. 模型预测模式 (predict_stocks) - 使用模型进行选股预测
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

# 默认配置
class DefaultConfig:
    def __init__(self):
        self.factor_weights = {
            'technical': 0.25,
            'fundamental': 0.25,
            'sector': 0.20,
            'capital_flow': 0.30
        }
        self.min_market_cap = 50  # 最小市值50亿

system_config = DefaultConfig()
backtest_config = DefaultConfig()
from .data_loader import IntegratedDataLoader
from .stock_selector import StockSelector
from .backtester import Backtester

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
  calculate_features  - 计算缺失特征（特征工程）
  train_model        - 训练机器学习模型
  predict_stocks     - 使用模型进行选股预测

使用示例:
  # 计算缺失特征
  python main.py --mode calculate_features

  # 训练机器学习模型
  python main.py --mode train_model

  # 使用模型进行选股预测
  python main.py --mode predict_stocks

  # 自定义输出目录
  python main.py --output_dir ./custom_results
        """
    )

    parser.add_argument(
        '--mode',
        choices=['calculate_features', 'train_model', 'predict_stocks'],
        default='calculate_features',
        help='运行模式 (默认: calculate_features)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='输出目录 (默认: ./results)'
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

def _prepare_ml_training_data(stock_data: pd.DataFrame, sector_data: pd.DataFrame, data_loader) -> tuple:
    """准备机器学习训练数据"""
    try:
        from clean_ml_system.sector_potential_analyzer import SectorPotentialAnalyzer
        logger = logging.getLogger(__name__)

        # 初始化板块分析器
        sector_analyzer = SectorPotentialAnalyzer()

        # 🎯 关键修复：在for循环之前为全部数据生成标签
        logger.info("🎯 开始为全部股票数据生成训练标签...")
        stock_data_with_labels = _generate_training_labels_for_all_data(stock_data)
        logger.info("✅ 全部数据标签生成完成")

        # 准备特征和标签
        features_list = []
        labels_list = []
        feature_names = None

        # 按日期分组处理（使用已生成标签的数据）
        for date, daily_stock_data in stock_data_with_labels.groupby('date'):
            try:
                date_str = date.strftime('%Y%m%d')

                # 训练阶段：使用全市场数据，不进行板块过滤
                # 这样可以让模型学习到更多的负面形态特征

                # 只进行基础过滤：避免追高，但保留下跌股票用于学习
                # daily_filtered = daily_stock_data[
                #     (daily_stock_data['pct_chg'] <= 5.0)  & (daily_stock_data['pct_chg'] >= -5.0)
                # ].copy()
                daily_filtered = daily_stock_data
                # 过滤新股（训练阶段）
                if len(daily_filtered) > 0:
                    new_stock_filter = _filter_new_stocks(daily_filtered, date_str)
                    daily_filtered = daily_filtered[new_stock_filter]

                if len(daily_stock_data) > 0:
                    new_stock_filter = _filter_new_stocks(daily_stock_data, date_str)
                    daily_filtered = daily_stock_data[new_stock_filter]

                # # 过滤多均线空头趋势股票（训练阶段新增）
                # if len(daily_filtered) > 0:
                #     bearish_filter = _filter_bearish_ma_trend_stocks(daily_filtered, date_str, logger)
                #     daily_filtered = daily_filtered[bearish_filter]
                #     logger.info(f"📅 {date_str}: 训练阶段多均线空头趋势过滤后剩余 {len(daily_filtered)} 只候选股票")

                # 过滤ST股票（训练阶段）
                if len(daily_filtered) > 0:
                    st_stock_filter = _filter_st_stocks(daily_filtered)
                    daily_filtered = daily_filtered[st_stock_filter]

                # 移除技术形态过滤，让模型自主学习风险模式
                # daily_filtered = _apply_technical_pattern_filters(
                #     daily_filtered, date_str, logger
                # )

                if len(daily_filtered) == 0:
                    continue

                # 标签已经在循环前生成，无需重新生成
                # daily_filtered已经包含next_day_return_top10_pct标签

                # 准备特征
                feature_columns = _get_ml_feature_columns(daily_filtered)
                if feature_names is None:
                    feature_names = feature_columns
                    logger.info(f"📊 初始化特征列表: {len(feature_names)} 个特征")

                    # 检查数据中实际存在的特征
                    available_in_data = [f for f in feature_names if f in daily_filtered.columns]
                    missing_in_data = [f for f in feature_names if f not in daily_filtered.columns]

                    logger.info(f"📊 数据中可用特征: {len(available_in_data)}/{len(feature_names)}")
                    if missing_in_data:
                        logger.warning(f"📊 数据中缺失特征: {missing_in_data[:5]}{'...' if len(missing_in_data) > 5 else ''}")

                    if len(available_in_data) < len(feature_names) * 0.3:
                        logger.error(f"❌ 可用特征过少 ({len(available_in_data)}/{len(feature_names)})，无法继续")
                        return None, None, None

                # 检查特征完整性
                missing_features = [col for col in feature_names if col not in daily_filtered.columns]
                if missing_features:
                    logger.warning(f"日期 {date_str} 缺少特征 ({len(missing_features)}/{len(feature_names)}): {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")

                    # 过滤掉缺失的特征
                    available_feature_names = [col for col in feature_names if col in daily_filtered.columns]
                    missing_ratio = len(missing_features) / len(feature_names)

                    if missing_ratio > 0.2:  # 如果缺失超过20%的特征，跳过
                        logger.warning(f"日期 {date_str} 可用特征不足 ({len(available_feature_names)}/{len(feature_names)}, 缺失{missing_ratio:.1%})，跳过该日期")
                        continue
                    else:
                        logger.info(f"日期 {date_str} 使用可用特征继续处理 ({len(available_feature_names)}/{len(feature_names)})")
                        # 更新当前日期使用的特征列表
                        current_feature_names = available_feature_names
                else:
                    current_feature_names = feature_names

                # 提取特征（使用当前可用的特征）
                X_daily = daily_filtered[current_feature_names].fillna(0).values

                # 准备标签（使用次日涨幅市场排名特征进行二分类）
                if 'next_day_return_top10_pct' not in daily_filtered.columns:
                    continue

                # 二分类标签：1表示次日涨幅排名前10%，0表示其他
                y_daily = daily_filtered['next_day_return_top10_pct'].values.astype(int)

                # 添加到列表
                features_list.append(X_daily)
                labels_list.append(y_daily)

            except Exception as e:
                print(f"处理日期 {date} 数据失败: {e}")
                continue

            # 内存优化：每处理20个日期进行一次垃圾回收
            if len(features_list) % 20 == 0:
                import gc
                gc.collect()

        if not features_list:
            print("❌ 没有有效的训练数据")
            return None, None, None

        # 合并所有数据
        import numpy as np
        X_train = np.vstack(features_list)
        y_train = np.hstack(labels_list)

        # 检查是否有有效的训练数据
        if len(X_train) == 0:
            logger.error("❌ 没有有效的训练数据，请检查数据和特征")
            return None, None, None

        # 更新feature_names为实际使用的特征（取所有日期的交集）
        if len(X_train) > 0:
            # 使用第一个样本的特征数量来确定最终的特征名称
            actual_feature_count = X_train.shape[1]
            if actual_feature_count != len(feature_names):
                logger.warning(f"特征数量不匹配: 期望{len(feature_names)}, 实际{actual_feature_count}")
                # 使用可用特征的子集
                feature_names = feature_names[:actual_feature_count]

        logger.info(f"✅ 训练数据准备完成: {len(X_train)} 样本")
        # 二分类标签分布
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        label_distribution = dict(zip(unique_labels, label_counts))
        logger.info(f"   标签分布: {label_distribution} (0=其他, 1=Top10%)")

        # 详细特征统计
        logger.info("=== 特征统计信息 ===")
        logger.info(f"总特征数: {len(feature_names)}")

        # 按类别统计特征（使用正确的特征定义）
        single_day_patterns = [f for f in feature_names if f.startswith('pattern_') and f in [
            'pattern_bare_head_yang', 'pattern_bare_head_yin', 'pattern_hammer',
            'pattern_hanging_man', 'pattern_doji', 'pattern_morning_star',
            'pattern_evening_star', 'pattern_engulfing_bull', 'pattern_engulfing_bear',
            'pattern_piercing', 'pattern_dark_cloud'
        ]]

        ma_features = [f for f in feature_names if 'ma' in f and ('ma5' in f or 'ma10' in f or 'ma20' in f or 'ma30' in f or 'ma60' in f or 'ma120' in f or 'ma_' in f or 'price_ma' in f)]
        price_distance_features = [f for f in feature_names if 'price_low_distance' in f]

        # 技术指标特征（完整定义）
        technical_indicators = [
            'rsi6', 'rsi12', 'rsi14', 'ema12', 'ema26',
            'macd', 'macd_signal', 'macd_hist', 'rsi_macd_agreement',
            'bb_upper', 'bb_middle', 'bb_width', 'bb_position', 'atr14',
            'trend_strength', 'trend_consistency', 'bb_breakout_signal'
        ]
        technical_features_actual = [f for f in feature_names if f in technical_indicators]

        logger.info(f"单日K线形态特征: {len(single_day_patterns)} 个")
        if single_day_patterns:
            logger.info(f"  包含: {', '.join(single_day_patterns)}")

        logger.info(f"均线相关特征: {len(ma_features)} 个")
        if ma_features:
            logger.info(f"  包含: {', '.join(ma_features)}")

        logger.info(f"价格距离特征: {len(price_distance_features)} 个")
        if price_distance_features:
            logger.info(f"  包含: {', '.join(price_distance_features)}")

        logger.info(f"技术指标特征: {len(technical_features_actual)} 个")
        if technical_features_actual:
            logger.info(f"  包含: {', '.join(technical_features_actual)}")

        # 分析训练数据规模和质量
        logger.info("=== 训练数据规模分析 ===")
        logger.info(f"样本数量: {len(X_train)}")
        logger.info(f"特征数量: {len(feature_names)}")
        logger.info(f"样本/特征比: {len(X_train)/len(feature_names):.1f}")

        if len(X_train) < 1000:
            logger.warning(f"⚠️ 训练样本较少 ({len(X_train)})，可能影响模型性能")
            logger.warning("   建议：扩大训练时间范围或降低过滤条件")
        elif len(X_train) < 5000:
            logger.info(f"📊 训练样本适中 ({len(X_train)})，适合快速训练")
        else:
            logger.info(f"📊 训练样本充足 ({len(X_train)})，有利于模型学习")

        # 分析样本/特征比
        sample_feature_ratio = len(X_train) / len(feature_names)
        if sample_feature_ratio < 10:
            logger.warning(f"⚠️ 样本/特征比过低 ({sample_feature_ratio:.1f})，可能导致过拟合")
            logger.warning("   建议：减少特征数量或增加训练样本")
        elif sample_feature_ratio < 50:
            logger.info(f"📊 样本/特征比适中 ({sample_feature_ratio:.1f})")
        else:
            logger.info(f"📊 样本/特征比充足 ({sample_feature_ratio:.1f})")

        return X_train, y_train, feature_names

    except Exception as e:
        print(f"❌ 准备训练数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

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

def _get_ml_feature_columns(data: pd.DataFrame) -> list:
    """获取机器学习特征列（基于实际存在的特征，严格防止数据泄露）"""
    try:
        logger = logging.getLogger(__name__)

        # 严格排除未来信息特征（防止数据泄露）
        future_info_features = [
            'future_return_1d', 'future_return_2d', 'is_can_buy'
        ]

        
        # 价格衍生特征（删除绝对价格，使用相对特征）
        # ❌ 删除: 'open', 'high', 'low', 'close', 'volume', 'amount' - 绝对值特征
        price_features = [
            'pct_chg', 'turnover_rate',  # 保留相对特征
            'volume_weighted_price', 'vol_price_corr_20d', 'price_vol_change_ratio',
            'price_volume_divergence',
            # ✅ 新增高级价格位置特征
            'price_position_in_range_5d', 'price_position_in_range_20d', 'price_position_in_range_60d',
            # ✅ 新增价格相对强度特征
            'price_vs_ma5_ratio', 'price_vs_ma20_ratio', 'price_vs_ma60_ratio'
        ]

        # 收益率特征（仅历史，已验证return_5d和return_20d无数据泄露）
        return_features = [
            'return_5d', 'return_20d', 'risk_adjusted_return', 'returns_zscore',
            'cumulative_return_3d', 'cumulative_return_5d', 'cumulative_return_10d',
            # 新增短期反转特征
            'return_1d', 'return_3d'
        ]

        # 成交量特征（删除绝对成交量，使用相对特征）
        volume_features = [
            'volatility20', 'volume_zscore', 'parkinson_vol', 'vol_term_structure',
            'volatility_skew', 'volatility_adjusted_momentum', 'fd_amount',
            # ✅ 新增成交量相对特征
            'volume_ratio_5d', 'volume_ratio_20d', 'volume_percentile_60d',
            # ✅ 新增波动率标准化特征
            'volatility_zscore_20d', 'volatility_percentile_60d'
        ]

        # 技术指标特征（删除绝对均线值，保留相对指标）
        # ❌ 删除: 'ma5', 'ma10', 'ma20', 'ma30', 'ma60', 'ma120', 'ema12', 'ema26' - 绝对均线值
        technical_features = [
            'rsi6', 'rsi12', 'rsi14',  # RSI是相对指标，保留
            'macd', 'macd_signal', 'macd_hist', 'rsi_macd_agreement',  # MACD是相对指标，保留
            # 保留MA交叉强度特征（相对特征）
            'ma5_above_ma10_strength', 'ma5_below_ma10_strength',
            # ✅ 新增均线相对特征
            'ma_alignment_score', 'ma_convergence_score',
            'ma5_ma20_distance_pct', 'ma20_ma60_distance_pct'
        ]

        # 布林带特征（删除绝对布林带值，保留相对特征）
        # ❌ 删除: 'bb_upper', 'bb_middle', 'bb_lower' - 绝对布林带值
        bollinger_features = [
            'bb_width', 'bb_position', 'bb_breakout_signal'  # 保留相对特征
        ]

        # 涨跌停特征（基于实际存在的特征）
        limit_features = [
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_times', 'limit_strength',
            'time_to_limit_minutes', 'limit_quality_score', 'limit_up_days_5d',
            'limit_down_days_5d', 'limit_up_days_20d', 'consecutive_limit_up',
            'consecutive_limit_down', 'limit_up_volume_ratio', 'limit_up_turnover_ratio'
        ]

        # 基本面特征（修复：移除绝对市值，使用相对特征）
        # ❌ 删除: 'total_mv', 'circ_mv' - 绝对市值特征，不适合跨时间建模
        fundamental_features = [
            'float_ratio', 'mv_factor', 'capital_intensity',
            # ✅ 保留对数市值（相对特征）
            'log_market_cap',
            # ✅ 新增市值相对特征（基于专业量化研究）
            'market_cap_percentile',      # 市值百分位排名（0-1）
            'market_cap_zscore',          # 市值Z-score标准化
            'size_factor_score',          # 规模因子评分（小盘股溢价）
            'liquidity_adjusted_mv',      # 流动性调整市值
            'market_cap_momentum',        # 市值动量（相对变化）
            'sector_relative_mv'          # 板块内相对市值排名
        ]

        # 高级技术特征
        advanced_technical_features = [
            'atr14', 'trend_strength', 'trend_consistency', 'candlestick_strength'
        ]

        # K线市场微观结构特征（专业量化机构版本）
        # 替代传统主观形态识别，使用连续型数值特征量化市场微观结构
        microstructure_features = [
            # === 单日微观结构特征 ===
            'lower_shadow_ratio',      # 下影线比例：买方支撑强度
            'upper_shadow_ratio',      # 上影线比例：卖方压力强度
            'body_ratio',              # 实体比例：趋势信念强度
            'reversal_score',          # 日内反转强度
            'overnight_gap_pct',       # 隔夜跳空强度
            'intraday_momentum',       # 日内动量强度
            'price_position_ratio',    # 收盘价位置比例
            'volatility_efficiency',   # 波动率效率
            'buying_pressure',         # 买方压力指数
            'selling_pressure',        # 卖方压力指数
            'market_structure_score',  # 市场结构综合评分

            # === 基于DTW的多日形态相似度特征 ===
            'w_bottom_similarity',          # W底形态相似度
            'head_shoulders_similarity',    # 头肩顶形态相似度
            'cup_handle_similarity',        # 杯柄形态相似度
            'ascending_triangle_similarity', # 上升三角形相似度
            'flag_pattern_similarity',      # 旗形整理相似度
            'double_top_similarity',        # 双顶形态相似度

            # === 多日形态解构特征 ===
            # W底解构特征
            'w_bottom_vol_ratio',           # 波动率结构特征
            'w_bottom_vol_profile_score',   # 成交量剖面特征
            'w_bottom_rs_improvement',      # 相对强度变化特征

            # 头肩顶解构特征
            'hs_vol_progression',           # 成交量递进特征
            'hs_symmetry_score',            # 对称性评分
            'hs_neckline_strength',         # 颈线强度特征

            # 杯柄解构特征
            'cup_depth_ratio',              # 杯子深度比例
            'handle_vol_contraction',       # 手柄缩量特征
            'cup_rounding_score'            # 圆弧度评分
        ]

        # 支撑阻力特征
        support_resistance_features = [
            'support_level', 'resistance_level', 'risk_reward_ratio',
            'near_support', 'near_resistance'
        ]

        # 相对强度和动量特征
        momentum_features = [
            'relative_strength_20d', 'momentum_acceleration', 'short_reversal',
            'tech_momentum_signal'
        ]

        # 综合评分特征（移除重复指标，保留更专业的版本）
        composite_features = [
            'composite_trading_signal', 'technical_score', 'composite_score', 'board_quality'
            # 移除technical_consensus和multi_period_trend_agreement，使用feature_calculator.py的专业版本
            # 移除momentum_score，使用新增的ma_trend_strength
        ]

        # 其他量化特征
        quant_features = [
            'amihud_illiquidity', 'overbought_oversold', 'time_efficiency_score',
            'stability_score', 'continuity_factor', 'open_times', 'up_stat',
            # 新增专业量化特征
            'intraday_return', 'open_close_ratio', 'close_to_prev_close',
            'normalized_atr', 'turnover_rate_ratio_20d', 'turnover_rate_ratio_5d',
            'volume_zscore_daily'
        ]

        # feature_calculator.py新增的特征（120日版本）
        calculated_features = [
            'high_volatility', 'recent_limit_down', 'low_quality_limit',
            'limit_up_count_120d', 'has_limit_up_120d', 'limit_up_next_day_positive_rate_120d',
            'first_board_seal_rate_120d', 'consecutive_limit_rate_120d',
            'limit_up_premium_5pct_count_120d', 'limit_up_quality_score_120d'
        ]

        # 专业量化特征（阶段一新增）
        # 注意：next_day_return_top10_pct是训练标签，不能作为输入特征！
        professional_quant_features = [
            # 'next_day_return_top10_pct',  # 这是训练标签，不能作为特征！
            'support_bounce_signal',      # 技术止跌信号特征
            'panic_selling_exhaustion',   # 恐慌盘出尽特征
            'momentum_quality_score',     # 动量质量监控
            'stock_market_correlation',   # 个股市场相关性
            'return_ranking_2d_top10_pct',   # 2日涨幅排名
            'return_ranking_3d_top10_pct',   # 3日涨幅排名
            'return_ranking_5d_top10_pct',   # 5日涨幅排名
            'return_ranking_10d_top10_pct'   # 10日涨幅排名
        ]

        # 新增专业特征（替代重复指标）
        enhanced_features = [
            # 均线趋势一致性特征（专业版本）
            'ma_bullish_alignment', 'ma_bearish_alignment', 'ma_trend_strength',
            'ma_convergence_divergence', 'price_ma_position',

            # 支撑阻力位特征（专业版本）
            'resistance_proximity', 'support_proximity', 'resistance_strength',
            'support_strength', 'sr_zone_position',

            # 长上影+均线阻力位特征（2024年新增）
            'upper_shadow_ma_resistance', 'shadow_length_ratio', 'ma_resistance_confluence',
            'rejection_strength', 'volume_confirmation',

            # 高频量化特征（2024年新增）
            'price_momentum_acceleration', 'volume_price_divergence', 'intraday_volatility_regime',
            'order_flow_imbalance', 'market_impact_factor',

            # 保留专业版本的重复指标
            'technical_consensus', 'multi_period_trend_agreement'
        ]

        # 横截面排名特征
        cross_sectional_features = [
            # 基础特征的横截面排名
            'rank_return_20d', 'rank_volatility20', 'rank_turnover_rate', 'rank_total_mv',
            'rank_cumulative_return_5d', 'rank_cumulative_return_10d', 'rank_rsi14',
            'rank_return_1d', 'rank_return_3d',
            # 复合排名特征
            'rank_momentum_composite', 'rank_quality_composite', 'rank_risk_composite'
        ]

        # 多周期股票市场相关性特征
        correlation_features = [
            # 基础滚动指标
            'corr_5d', 'corr_10d', 'corr_20d', 'corr_60d', 'corr_120d',
            'beta_5d', 'beta_10d', 'beta_20d', 'beta_60d', 'beta_120d',
            # 期限结构特征
            'corr_slope_5_20', 'corr_slope_10_60', 'corr_slope_20_120',
            'beta_slope_5_20', 'beta_slope_10_60', 'beta_slope_20_120',
            # 动态变化特征
            'corr_10d_mom5', 'corr_20d_mom5', 'corr_60d_mom5',
            'beta_10d_mom5', 'beta_20d_mom5', 'beta_60d_mom5',
            # 相对位置特征
            'corr_20d_zscore250', 'corr_60d_zscore250',
            'beta_20d_zscore250', 'beta_60d_zscore250'
        ]

        # 均线距离特征
        ma_distance_features = [
            'ma_convergence', 'price_ma5_distance', 'price_ma10_distance',
            'price_ma20_distance', 'price_ma60_distance', 'ma_support_strength'
        ]

        # 股价低点距离特征
        price_distance_features = [
            'price_low_distance_3d', 'price_low_distance_5d', 'price_low_distance_10d',
            'price_low_distance_20d', 'price_low_distance_60d', 'price_low_distance_weighted'
        ]

        # 合并所有特征
        all_features = (
            price_features + return_features + volume_features + technical_features +
            bollinger_features + limit_features + fundamental_features +
            advanced_technical_features + microstructure_features + support_resistance_features +
            momentum_features + composite_features + quant_features + calculated_features +
            ma_distance_features + price_distance_features + enhanced_features +
            professional_quant_features + cross_sectional_features + correlation_features
        )

        # 严格检查数据泄露
        leaked_features = [col for col in all_features if col in future_info_features]
        if leaked_features:
            logger.error(f"检测到数据泄露特征: {leaked_features}")
            raise ValueError(f"数据泄露检测失败: {leaked_features}")

        # 只保留数据中存在的特征
        available_features = [col for col in all_features if col in data.columns]

        # 移除任何可能的未来信息特征
        safe_features = [col for col in available_features if col not in future_info_features]

        logger.info(f"特征选择完成: {len(safe_features)}/{len(all_features)} 个特征可用")
        logger.info(f"已排除未来信息特征: {len(future_info_features)} 个")

        # 记录特征分类统计
        feature_stats = {
            'price_features': len([f for f in price_features if f in safe_features]),
            'technical_features': len([f for f in technical_features if f in safe_features]),
            'microstructure_features': len([f for f in microstructure_features if f in safe_features]),
            'limit_features': len([f for f in limit_features if f in safe_features]),
            'calculated_features': len([f for f in calculated_features if f in safe_features]),
            'professional_quant_features': len([f for f in professional_quant_features if f in safe_features]),
            'cross_sectional_features': len([f for f in cross_sectional_features if f in safe_features]),
            'correlation_features': len([f for f in correlation_features if f in safe_features])
        }
        logger.info(f"特征分类统计: {feature_stats}")

        return safe_features

    except Exception as e:
        logger.error(f"获取特征列失败: {e}")
        return []



def _generate_training_labels_for_all_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    为全部股票数据生成训练标签（次日涨幅市场排名）

    Args:
        stock_data: 包含多天股票数据的DataFrame

    Returns:
        添加了标签的DataFrame
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("开始为全部数据生成训练标签（次日涨幅市场排名）...")

        # 数据基本信息检查
        logger.info(f"输入数据基本信息:")
        logger.info(f"  总记录数: {len(stock_data)}")
        logger.info(f"  股票数量: {stock_data['stock_code'].nunique()}")
        logger.info(f"  日期数量: {stock_data['date'].nunique()}")
        logger.info(f"  日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
        logger.info(f"  pct_chg字段: 有效值{stock_data['pct_chg'].notna().sum()}个, 缺失值{stock_data['pct_chg'].isna().sum()}个")

        # 确保数据按股票代码和日期排序
        stock_data_sorted = stock_data.sort_values(['stock_code', 'date']).reset_index(drop=True)

        # 计算次日收益率
        logger.info("计算次日收益率...")
        stock_data_sorted['next_day_return'] = stock_data_sorted.groupby('stock_code')['pct_chg'].shift(-1)

        # 检查次日收益率计算结果
        next_day_valid = stock_data_sorted['next_day_return'].notna().sum()
        next_day_missing = stock_data_sorted['next_day_return'].isna().sum()
        logger.info(f"次日收益率计算结果: 有效值{next_day_valid}个, 缺失值{next_day_missing}个")

        if next_day_valid == 0:
            logger.error("次日收益率计算失败：所有值都是NaN")
            stock_data['next_day_return_top10_pct'] = 0
            return stock_data

        # 计算风险调整后的次日收益率
        logger.info("计算风险调整后的次日收益率...")
        stock_data_sorted = _calculate_risk_adjusted_next_day_return(stock_data_sorted)

        # 初始化标签列
        stock_data_sorted['next_day_return_top10_pct'] = 0

        # 按日期计算每日排名
        logger.info("计算每日排名...")
        total_positive_labels = 0
        processed_dates = 0
        valid_dates = 0

        for date in stock_data_sorted['date'].unique():
            try:
                processed_dates += 1
                date_mask = stock_data_sorted['date'] == date
                date_data = stock_data_sorted[date_mask]

                if len(date_data) == 0:
                    continue

                # 使用风险调整后的次日收益率进行排序
                risk_adjusted_returns = date_data['risk_adjusted_next_day_return']
                valid_risk_adjusted = risk_adjusted_returns.dropna()

                if len(valid_risk_adjusted) == 0:
                    continue  # 当日没有有效的风险调整后收益率

                valid_dates += 1

                # 剔除极端样本：排除前1%的极端幸运儿
                extreme_threshold = valid_risk_adjusted.quantile(0.99)
                filtered_returns = valid_risk_adjusted[valid_risk_adjusted < extreme_threshold]

                if len(filtered_returns) < 10:  # 如果剔除后样本太少，使用原始数据
                    filtered_returns = valid_risk_adjusted

                # 计算90%分位数阈值（基于风险调整后收益率）
                threshold = filtered_returns.quantile(0.9)

                # 标记排名前10%的股票（排除极端样本）
                top10_mask = (
                    (risk_adjusted_returns >= threshold) &
                    (risk_adjusted_returns < extreme_threshold) &
                    (~risk_adjusted_returns.isna())
                )

                # 更新标签
                stock_data_sorted.loc[date_mask & top10_mask, 'next_day_return_top10_pct'] = 1

                # 统计当日正类数量
                daily_positive = top10_mask.sum()
                total_positive_labels += daily_positive

                if processed_dates % 50 == 0:  # 每50天输出一次进度
                    logger.info(f"处理进度: {processed_dates}天, 当前日期: {date.strftime('%Y-%m-%d')}, 当日Top10%: {daily_positive}只")

            except Exception as date_e:
                logger.error(f"处理日期 {date} 失败: {date_e}")
                continue

        # 将标签映射回原始数据
        result_data = stock_data.copy()
        label_mapping = stock_data_sorted[['stock_code', 'date', 'next_day_return_top10_pct']].copy()

        # 确保日期类型一致
        result_data['date'] = pd.to_datetime(result_data['date'])
        label_mapping['date'] = pd.to_datetime(label_mapping['date'])

        # 合并标签
        result_data = result_data.merge(
            label_mapping[['stock_code', 'date', 'next_day_return_top10_pct']],
            on=['stock_code', 'date'],
            how='left'
        )

        # 填充缺失值
        result_data['next_day_return_top10_pct'] = result_data['next_day_return_top10_pct'].fillna(0).astype(int)

        # 最终验证
        final_positive_count = result_data['next_day_return_top10_pct'].sum()
        total_samples = len(result_data)
        positive_ratio = final_positive_count / total_samples if total_samples > 0 else 0

        logger.info(f"训练标签生成完成:")
        logger.info(f"  处理日期数: {processed_dates}")
        logger.info(f"  有效日期数: {valid_dates}")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  正类数量: {final_positive_count}")
        logger.info(f"  正类比例: {positive_ratio:.1%}")

        if final_positive_count == 0:
            logger.error("标签生成失败：正类数量为0")

        return result_data

    except Exception as e:
        logger.error(f"生成训练标签失败: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        stock_data['next_day_return_top10_pct'] = 0
        return stock_data

def _calculate_risk_adjusted_next_day_return(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算风险调整后的次日收益率
    使用夏普比率思想：次日收益 / 近期波动率
    """
    try:
        logger = logging.getLogger(__name__)

        # 初始化风险调整后收益率
        stock_data['risk_adjusted_next_day_return'] = stock_data['next_day_return'].copy()

        # 按股票分组计算风险调整
        for stock_code in stock_data['stock_code'].unique():
            stock_mask = stock_data['stock_code'] == stock_code
            stock_data_subset = stock_data[stock_mask].copy().sort_values('date')

            if len(stock_data_subset) < 21:  # 至少需要21天数据计算20日波动率
                continue

            # 计算20日历史波动率（使用pct_chg）
            if 'pct_chg' in stock_data_subset.columns:
                rolling_std = stock_data_subset['pct_chg'].rolling(
                    window=20, min_periods=10
                ).std()

                # 避免除零，设置最小波动率
                rolling_std = np.maximum(rolling_std, 0.5)  # 最小0.5%的波动率

                # 计算风险调整后的次日收益率
                risk_adjusted_returns = (
                    stock_data_subset['next_day_return'] / rolling_std
                )

                # 限制在合理范围内
                risk_adjusted_returns = np.clip(risk_adjusted_returns, -10, 10)

                # 将结果赋值回原数据
                stock_data.loc[stock_mask, 'risk_adjusted_next_day_return'] = risk_adjusted_returns.fillna(
                    stock_data_subset['next_day_return']
                ).values

        logger.info("风险调整后次日收益率计算完成")
        return stock_data

    except Exception as e:
        logger.error(f"风险调整后次日收益率计算失败: {e}")
        # 如果失败，使用原始次日收益率
        stock_data['risk_adjusted_next_day_return'] = stock_data['next_day_return'].copy()
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
                                         logger) -> np.ndarray:
    """
    计算专业的样本权重

    基于以下策略：
    1. 基础权重：根据类别频率倒数
    2. 平滑权重：避免极端权重值
    3. 量化金融调整：考虑金融数据的特殊性
    """
    try:
        from sklearn.utils.class_weight import compute_class_weight

        # 方法1：sklearn的balanced权重
        classes = np.unique(y_train)
        class_weights_balanced = compute_class_weight(
            'balanced', classes=classes, y=y_train
        )

        # 方法2：自定义权重（更适合量化金融）
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        total_count = len(y_train)

        # 基础权重：频率倒数
        pos_weight_base = total_count / (2 * pos_count)
        neg_weight_base = total_count / (2 * neg_count)

        # 平滑权重：避免过度补偿
        # 使用对数平滑，减少极端权重
        pos_weight_smooth = 1 + np.log(imbalance_ratio) / 2
        neg_weight_smooth = 1.0

        # 综合权重：平衡效果和稳定性
        if imbalance_ratio > 20:  # 严重不平衡
            pos_weight_final = min(pos_weight_smooth, 5.0)  # 限制最大权重
            neg_weight_final = 1.0
        elif imbalance_ratio > 5:  # 中度不平衡
            pos_weight_final = min(pos_weight_base, 3.0)
            neg_weight_final = 1.0
        else:  # 轻度不平衡
            pos_weight_final = min(pos_weight_base, 2.0)
            neg_weight_final = 1.0

        # 构建样本权重数组
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 1] = pos_weight_final
        sample_weights[y_train == 0] = neg_weight_final

        logger.info(f"样本权重计算:")
        logger.info(f"  sklearn balanced权重: 正类={class_weights_balanced[1]:.2f}, 负类={class_weights_balanced[0]:.2f}")
        logger.info(f"  自定义权重: 正类={pos_weight_final:.2f}, 负类={neg_weight_final:.2f}")
        logger.info(f"  权重比例: {pos_weight_final/neg_weight_final:.2f}:1")

        return sample_weights

    except Exception as e:
        logger.error(f"样本权重计算失败: {e}")
        # 返回均匀权重
        return np.ones(len(y_train))

def _discretize_returns(returns: np.ndarray) -> np.ndarray:
    """
    将收益率离散化为2个类别（二分类）
    注意：这个函数已废弃，现在使用next_day_return_top10_pct作为训练标签
    """
    try:
        import numpy as np

        # 二分类：0=其他, 1=Top10%
        # 计算90%分位数作为阈值
        threshold = np.percentile(returns, 90)
        labels = (returns >= threshold).astype(int)

        return labels

    except Exception as e:
        print(f"离散化收益率失败: {e}")
        return np.zeros(len(returns), dtype=int)

def _generate_training_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    专门生成训练标签的函数（彻底修复版本）
    计算次日涨幅市场排名，用作训练目标
    注意：这个函数包含未来信息，只能用于生成训练标签！
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("开始生成训练标签（次日涨幅市场排名）...")

        # 检查缓存
        cache_file = Path("to/local_data/cache/training_labels_cache.pkl")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # 生成缓存键（基于数据的股票代码和日期）
        data_key = f"{len(data)}_{data['stock_code'].nunique()}_{data['date'].min()}_{data['date'].max()}"

        # 尝试加载缓存
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('key') == data_key:
                        cached_labels = cache_data['labels']
                        # 验证缓存数据质量
                        positive_count = np.sum(cached_labels == 1) if len(cached_labels) > 0 else 0
                        positive_ratio = positive_count / len(cached_labels) if len(cached_labels) > 0 else 0

                        # 只有缓存数据合理才使用
                        if (len(cached_labels) == len(data) and
                            positive_count > 0 and
                            0.05 <= positive_ratio <= 0.15):
                            logger.info("从缓存加载训练标签")
                            data['next_day_return_top10_pct'] = cached_labels
                            # 统计标签分布
                            label_counts = data['next_day_return_top10_pct'].value_counts()
                            total_samples = len(data)
                            logger.info(f"缓存标签加载完成:")
                            logger.info(f"  总样本数: {total_samples}")
                            logger.info(f"  正类(Top10%)数量: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total_samples:.1%})")
                            return data
                        else:
                            logger.warning(f"缓存数据质量异常，重新计算: 长度匹配={len(cached_labels)==len(data)}, 正类={positive_count}({positive_ratio:.1%})")
            except Exception as e:
                logger.warning(f"加载标签缓存失败: {e}")

        # 使用新的跨日期标签生成方法
        logger.info("使用跨日期标签生成方法...")

        # 数据基本信息检查
        logger.info(f"输入数据基本信息:")
        logger.info(f"  总记录数: {len(data)}")
        logger.info(f"  股票数量: {data['stock_code'].nunique()}")
        logger.info(f"  日期数量: {data['date'].nunique()}")
        logger.info(f"  日期范围: {data['date'].min()} 到 {data['date'].max()}")
        logger.info(f"  pct_chg字段: 有效值{data['pct_chg'].notna().sum()}个, 缺失值{data['pct_chg'].isna().sum()}个")
        logger.info(f"  pct_chg范围: {data['pct_chg'].min():.2f}% 到 {data['pct_chg'].max():.2f}%")

        # 获取当前日期
        current_dates = data['date'].unique()
        if len(current_dates) != 1:
            logger.warning(f"数据包含多个日期: {len(current_dates)}个，将分别处理")

        # 初始化标签列
        data['next_day_return_top10_pct'] = 0

        # 为每个日期计算次日收益率标签
        for current_date in current_dates:
            logger.info(f"处理日期: {current_date}")

            # 获取当前日期的数据
            current_data = data[data['date'] == current_date].copy()

            # 获取下一个交易日
            next_trading_date = _get_next_trading_date(current_date)
            if next_trading_date is None:
                logger.warning(f"无法获取{current_date}的下一个交易日，跳过标签生成")
                continue

            logger.info(f"下一个交易日: {next_trading_date}")

            # 加载下一个交易日的数据
            next_day_data = _load_next_day_data(next_trading_date, current_data['stock_code'].tolist())
            if next_day_data is None or next_day_data.empty:
                logger.warning(f"无法加载{next_trading_date}的数据，跳过标签生成")
                continue

            logger.info(f"成功加载下一交易日数据: {len(next_day_data)}条记录")

            # 计算次日收益率标签
            labels = _calculate_next_day_labels(current_data, next_day_data)

            # 更新标签
            current_indices = data[data['date'] == current_date].index
            data.loc[current_indices, 'next_day_return_top10_pct'] = labels

            logger.info(f"日期{current_date}标签生成完成: 正类{labels.sum()}个, 总数{len(labels)}个")

        # 检查最终结果
        total_positive = data['next_day_return_top10_pct'].sum()
        total_samples = len(data)
        positive_ratio = total_positive / total_samples if total_samples > 0 else 0

        logger.info(f"训练标签生成完成:")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  正类数量: {total_positive}")
        logger.info(f"  正类比例: {positive_ratio:.1%}")

        if total_positive == 0:
            logger.error("标签生成失败：正类数量为0")
            logger.error("可能原因：无法获取次日数据或次日数据质量问题")

        return data

    except Exception as e:
        logger.error(f"生成训练标签失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 设置默认值
        data['next_day_return_top10_pct'] = 0
        return data

def _calculate_professional_quant_metrics(y_true, y_pred, y_pred_proba, feature_names=None):
    """
    计算专业量化评价指标
    替换传统准确率为量化金融专业指标
    """
    try:
        import numpy as np
        from scipy.stats import spearmanr, pearsonr

        logger = logging.getLogger(__name__)

        # 确保输入是numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        # 如果是二分类，取正类概率
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]  # 正类(Top 10%)概率
        else:
            y_scores = y_pred_proba

        metrics = {}

        # 1. Precision@K: Top K预测信号的精确率
        for k in [50, 100, 200]:
            if len(y_scores) >= k:
                # 获取预测概率最高的K个样本的索引
                top_k_indices = np.argsort(y_scores)[-k:]
                # 计算这K个样本中真正为正类的比例
                precision_at_k = np.mean(y_true[top_k_indices])
                metrics[f'precision_at_{k}'] = precision_at_k
            else:
                metrics[f'precision_at_{k}'] = 0.0

        # 2. Recall: 真实Top 10%股票的召回率
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        metrics['recall'] = recall

        # 3. Information Coefficient (IC): 预测概率与真实标签的相关系数
        if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
            ic_pearson, _ = pearsonr(y_scores, y_true)
            metrics['ic_pearson'] = ic_pearson if not np.isnan(ic_pearson) else 0.0
        else:
            metrics['ic_pearson'] = 0.0

        # 4. RankIC: 基于排名的信息系数（更稳健）
        if len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
            rank_ic, _ = spearmanr(y_scores, y_true)
            metrics['rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0.0
        else:
            metrics['rank_ic'] = 0.0

        # 5. ICIR: IC信息比率（需要时间序列数据，这里用单次计算的近似）
        # 在实际应用中，ICIR = IC均值 / IC标准差，需要多期数据
        # 这里用IC的绝对值作为近似
        metrics['icir_approx'] = abs(metrics['ic_pearson'])

        # 6. 传统二分类指标（作为补充）
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # AUC-ROC
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        else:
            metrics['auc_roc'] = 0.5

        # 7. 量化交易特定指标
        # 预测为正类的样本中，真正为正类的比例（选股精确度）
        predicted_positives = np.sum(y_pred == 1)
        if predicted_positives > 0:
            stock_selection_precision = true_positives / predicted_positives
        else:
            stock_selection_precision = 0.0
        metrics['stock_selection_precision'] = stock_selection_precision

        # 预测概率的分布质量
        metrics['prediction_spread'] = np.std(y_scores)  # 预测概率的标准差
        metrics['prediction_range'] = np.max(y_scores) - np.min(y_scores)  # 预测概率的范围

        return metrics

    except Exception as e:
        logger.error(f"计算专业量化指标失败: {e}")
        return {
            'precision_at_50': 0.0, 'precision_at_100': 0.0, 'precision_at_200': 0.0,
            'recall': 0.0, 'ic_pearson': 0.0, 'rank_ic': 0.0, 'icir_approx': 0.0,
            'precision': 0.0, 'f1_score': 0.0, 'auc_roc': 0.5,
            'stock_selection_precision': 0.0, 'prediction_spread': 0.0, 'prediction_range': 0.0
        }

def _train_ml_model(X_train, y_train, feature_names):
    """训练机器学习模型（优化参数与日志）"""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        import xgboost as xgb
        import numpy as np

        logger = logging.getLogger(__name__)

        logger.info(f"开始训练模型，样本数: {len(X_train)}, 特征数: {len(feature_names)}")

        # 数据分布分析
        unique_labels, label_counts = np.unique(y_train, return_counts=True)
        label_distribution = dict(zip(unique_labels, label_counts))
        logger.info(f"标签分布: {label_distribution}")

        # 增强数据质量检查和处理
        nan_count = np.isnan(X_train).sum()
        inf_count = np.isinf(X_train).sum()

        # 检查极端异常值（基于实际特征计算逻辑）
        extreme_outliers = 0
        outlier_features = []
        for i, feature_name in enumerate(feature_names):
            feature_values = X_train[:, i]

            # 根据特征类型确定合理范围（2024年专业版本）
            if feature_name == 'pct_chg':
                # pct_chg是百分制，合理范围：-20%到20%（考虑ST股票）
                extreme_mask = (feature_values >= 30) | (feature_values <= -30)
                clip_min, clip_max = -30, 30
            elif feature_name in ['future_return_1d', 'future_return_2d']:
                # future_return是小数制，合理范围：-50%到50%
                extreme_mask = (feature_values > 0.6) | (feature_values < -0.6)
                clip_min, clip_max = -0.6, 0.6
            elif feature_name in ['return_5d', 'return_20d']:
                # return_Nd是小数制，代码中已限制-100%到1000%
                extreme_mask = (feature_values > 10.0) | (feature_values < -1.0)
                clip_min, clip_max = -1.0, 10.0
            elif 'rsi' in feature_name:
                # RSI指标范围：0到100
                extreme_mask = (feature_values > 100) | (feature_values < 0)
                clip_min, clip_max = 0, 100
            elif feature_name == 'bb_position':
                # 布林带位置：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif 'volatility' in feature_name:
                # 波动率是小数制，合理范围：0到100%
                extreme_mask = (feature_values > 1.0) | (feature_values < 0)
                clip_min, clip_max = 0, 1.0
            elif feature_name == 'macd':
                # MACD是价格差值，使用分位数方法
                p1, p99 = np.nanpercentile(feature_values, [1, 99])
                extreme_mask = (feature_values > p99 * 2) | (feature_values < p1 * 2)
                clip_min, clip_max = p1 * 2, p99 * 2
            elif 'turnover_rate' in feature_name:
                # 换手率是小数制，合理范围：0到0.5（50%）
                extreme_mask = (feature_values > 0.7) | (feature_values < 0)
                clip_min, clip_max = 0, 0.7
            # 新增特征的异常值处理
            elif feature_name in ['ma_bullish_alignment', 'ma_bearish_alignment', 'ma_convergence_divergence']:
                # 均线趋势特征：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name == 'ma_trend_strength':
                # 趋势强度：-1到1
                extreme_mask = (feature_values > 1) | (feature_values < -1)
                clip_min, clip_max = -1, 1
            elif feature_name == 'price_ma_position':
                # 价格相对均线位置：-1到1
                extreme_mask = (feature_values > 1) | (feature_values < -1)
                clip_min, clip_max = -1, 1
            elif feature_name in ['resistance_proximity', 'support_proximity', 'resistance_strength', 'support_strength']:
                # 支撑阻力位特征：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name == 'sr_zone_position':
                # 支撑阻力区间位置：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name in ['pattern_w_bottom', 'pattern_cup_handle'] and 'pattern_' in feature_name:
                # K线形态强度：0到1（新的强度评分）
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name == 'technical_consensus':
                # 技术共识：-1到1
                extreme_mask = (feature_values > 1) | (feature_values < -1)
                clip_min, clip_max = -1, 1
            elif feature_name == 'multi_period_trend_agreement':
                # 多周期趋势一致性：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            # 新增特征的异常值处理
            elif feature_name in ['upper_shadow_ma_resistance', 'shadow_length_ratio', 'ma_resistance_confluence',
                                'rejection_strength', 'volume_confirmation']:
                # 长上影阻力位特征：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name == 'price_momentum_acceleration':
                # 价格动量加速度：使用分位数方法
                p1, p99 = np.nanpercentile(feature_values, [1, 99])
                extreme_mask = (feature_values > p99 * 1.5) | (feature_values < p1 * 1.5)
                clip_min, clip_max = p1 * 1.5, p99 * 1.5
            elif feature_name in ['volume_price_divergence', 'intraday_volatility_regime']:
                # 量价背离和波动率状态：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name == 'order_flow_imbalance':
                # 订单流失衡：使用分位数方法
                p5, p95 = np.nanpercentile(feature_values, [5, 95])
                extreme_mask = (feature_values > p95 * 1.2) | (feature_values < p5 * 1.2)
                clip_min, clip_max = p5 * 1.2, p95 * 1.2
            elif feature_name == 'market_impact_factor':
                # 市场冲击因子：0到10
                extreme_mask = (feature_values > 10) | (feature_values < 0)
                clip_min, clip_max = 0, 10
            # 涨停质量特征的异常值处理（120日版本）
            elif feature_name == 'limit_up_count_120d':
                # 120日涨停次数：0到120
                extreme_mask = (feature_values > 120) | (feature_values < 0)
                clip_min, clip_max = 0, 120
            elif feature_name == 'has_limit_up_120d':
                # 是否有涨停：0或1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name in ['limit_up_next_day_positive_rate_120d', 'first_board_seal_rate_120d', 'consecutive_limit_rate_120d']:
                # 涨停相关比率：0到1
                extreme_mask = (feature_values > 1) | (feature_values < 0)
                clip_min, clip_max = 0, 1
            elif feature_name == 'limit_up_premium_5pct_count_120d':
                # 120日溢价次数：0到120
                extreme_mask = (feature_values > 120) | (feature_values < 0)
                clip_min, clip_max = 0, 120
            elif feature_name == 'limit_up_quality_score_120d':
                # 涨停质量评分：0到100
                extreme_mask = (feature_values > 100) | (feature_values < 0)
                clip_min, clip_max = 0, 100
            elif feature_name in ['total_mv', 'circ_mv'] or 'mv' in feature_name:
                # 市值特征：使用分位数方法，因为市值分布严重右偏
                # 市值不能为负，使用1%和99.5%分位数
                p1 = np.nanpercentile(feature_values, 1)
                p99_5 = np.nanpercentile(feature_values, 99.5)
                extreme_mask = (feature_values > p99_5) | (feature_values < p1)
                clip_min = max(0, p1)  # 确保市值不为负
                clip_max = p99_5
            elif feature_name in ['amount', 'volume'] or 'amount' in feature_name or 'volume' in feature_name:
                # 成交量/成交额特征：同样使用分位数方法
                p1 = np.nanpercentile(feature_values, 1)
                p99 = np.nanpercentile(feature_values, 99)
                extreme_mask = (feature_values > p99) | (feature_values < p1)
                clip_min = max(0, p1)  # 确保成交量/成交额不为负
                clip_max = p99
            elif 'log_' in feature_name or 'ratio' in feature_name:
                # 对数特征和比率特征：使用分位数方法
                p0_5 = np.nanpercentile(feature_values, 0.5)
                p99_5 = np.nanpercentile(feature_values, 99.5)
                extreme_mask = (feature_values > p99_5) | (feature_values < p0_5)
                clip_min, clip_max = p0_5, p99_5
            else:
                # 其他特征使用5σ原则（比3σ更宽松）
                mean_val = np.nanmean(feature_values)
                std_val = np.nanstd(feature_values)
                if std_val > 0:
                    extreme_mask = np.abs(feature_values - mean_val) > 5 * std_val
                    clip_min = mean_val - 5 * std_val
                    clip_max = mean_val + 5 * std_val
                else:
                    extreme_mask = np.zeros(len(feature_values), dtype=bool)
                    clip_min, clip_max = feature_values.min(), feature_values.max()

            extreme_count = extreme_mask.sum()
            if extreme_count > 0:
                extreme_outliers += extreme_count
                outlier_features.append(f"{feature_name}({extreme_count})")

                # 修复极端异常值
                X_train[:, i] = np.clip(feature_values, clip_min, clip_max)

        if extreme_outliers > 0:
            logger.warning(f"数据质量问题: NaN={nan_count}, Inf={inf_count}, 极端异常值={extreme_outliers}")
            logger.warning(f"包含极端异常值的特征: {', '.join(outlier_features[:10])}")
        else:
            logger.info(f"数据质量检查: NaN={nan_count}, Inf={inf_count}, 极端异常值={extreme_outliers}")

        # 处理Inf值
        if inf_count > 0:
            inf_features = []
            for i, feature_name in enumerate(feature_names):
                feature_inf_count = np.isinf(X_train[:, i]).sum()
                if feature_inf_count > 0:
                    inf_features.append(f"{feature_name}({feature_inf_count})")

            logger.warning(f"包含Inf值的特征: {', '.join(inf_features[:10])}")
            # 智能处理Inf值
            X_train = _handle_infinite_values(X_train, feature_names, logger)

        # 处理NaN值
        if nan_count > 0:
            X_train = np.nan_to_num(X_train, nan=0.0)

        # 数据预处理
        logger.info("开始数据标准化...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # PCA降维（基于顶级量化研究的最佳实践）
        X_scaled, pca_info = _apply_professional_pca_reduction(
            X_scaled, feature_names, n_features, logger
        )

        # 划分训练集和验证集
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # 分析样本分布，为XGBoost参数优化提供依据
        pos_samples = np.sum(y_train_split == 1)
        neg_samples = np.sum(y_train_split == 0)
        pos_ratio = pos_samples / len(y_train_split)
        imbalance_ratio = neg_samples / max(pos_samples, 1)  # 负样本/正样本比例

        logger.info(f"数据划分完成: 训练集{len(X_train_split)}样本, 验证集{len(X_val_split)}样本")
        logger.info(f"样本分布分析: 正类{pos_samples}个({pos_ratio:.1%}), 负类{neg_samples}个, 不平衡比例{imbalance_ratio:.1f}:1")

        # 根据特征数量、样本分布和数据质量调整XGBoost参数（2024年最佳实践）
        n_features = X_scaled.shape[1]  # 使用PCA后的特征数量
        n_samples = len(X_train_split)

        # 根据特征数量、样本分布和数据质量优化参数
        logger.info(f"根据{n_features}个特征、{n_samples}个样本、不平衡比例{imbalance_ratio:.1f}:1优化XGBoost参数")

        # 基于2024年最新研究的参数优化策略（适配PCA降维和样本不平衡）
        if n_features > 100:  # PCA后仍然是高维特征集
            # 更强的正则化，防止过拟合
            n_estimators = 800  # 增加树数量，依赖早停控制
            max_depth = 4  # 较浅的树，适合高维数据
            learning_rate = 0.03  # 更低的学习率，配合更多树
            reg_alpha = 0.1  # L1正则化
            reg_lambda = 0.1  # L2正则化
            subsample = 0.8  # 行采样
            colsample_bytree = 0.7  # 列采样
            colsample_bylevel = 0.8  # 每层列采样
            min_child_weight = max(3, int(imbalance_ratio / 2))  # 根据不平衡比例调整
            gamma = 0.1  # 分裂门槛
        elif n_features > 150:  # 超高维特征集（新增的专业特征）
            # 更强的正则化，防止过拟合
            n_estimators = 400  # 增加树数量，但用早停控制
            max_depth = 5  # 降低深度，减少过拟合
            learning_rate = 0.06  # 降低学习率，提高稳定性
            reg_alpha = 0.1  # 增强L1正则化
            reg_lambda = 0.1  # 增强L2正则化
            subsample = 0.75  # 更强的行采样
            colsample_bytree = 0.7  # 更强的列采样
            colsample_bylevel = 0.8  # 新增：每层列采样
            min_child_weight = 5  # 增加最小子节点权重
            gamma = 0.2  # 增加分裂门槛
        elif n_features > 120:  # 高维特征集
            n_estimators = 350
            max_depth = 6
            learning_rate = 0.07
            reg_alpha = 0.08
            reg_lambda = 0.08
            subsample = 0.8
            colsample_bytree = 0.75
            colsample_bylevel = 0.85
            min_child_weight = 4
            gamma = 0.15
        elif n_features > 50:  # PCA后中高维特征集
            n_estimators = 500
            max_depth = 6
            learning_rate = 0.05
            reg_alpha = 0.03
            reg_lambda = 0.03
            subsample = 0.85
            colsample_bytree = 0.8
            colsample_bylevel = 0.9
            min_child_weight = max(2, int(imbalance_ratio / 3))
            gamma = 0.05
        elif n_features > 20:  # PCA后中维特征集
            n_estimators = 400
            max_depth = 7
            learning_rate = 0.08
            reg_alpha = 0.02
            reg_lambda = 0.02
            subsample = 0.9
            colsample_bytree = 0.85
            colsample_bylevel = 0.95
            min_child_weight = max(1, int(imbalance_ratio / 5))
            gamma = 0.02
        else:  # PCA后低维特征集
            n_estimators = 300
            max_depth = 8
            learning_rate = 0.1
            reg_alpha = 0.01
            reg_lambda = 0.01
            subsample = 0.95
            colsample_bytree = 0.9
            colsample_bylevel = 1.0
            min_child_weight = max(1, int(imbalance_ratio / 8))
            gamma = 0.01

        # 根据样本数量进一步调整（2024年最佳实践）
        if n_samples > 100000:  # 大样本
            subsample = min(subsample, 0.7)  # 更强的采样
            learning_rate = max(learning_rate * 0.8, 0.05)  # 降低学习率
            n_estimators = min(n_estimators + 100, 500)  # 增加树数量
        elif n_samples > 50000:  # 中大样本
            subsample = min(subsample, 0.8)
            learning_rate = max(learning_rate * 0.9, 0.06)
        elif n_samples < 10000:  # 小样本
            subsample = min(subsample + 0.1, 1.0)  # 减少采样
            learning_rate = min(learning_rate * 1.2, 0.15)  # 提高学习率
            reg_alpha = max(reg_alpha * 0.5, 0.001)  # 减少正则化
            reg_lambda = max(reg_lambda * 0.5, 0.001)

        # 根据不平衡比例调整早停轮次和评估策略
        if imbalance_ratio > 20:
            early_stopping_rounds = 100  # 严重不平衡需要更多耐心
            eval_metric_list = ['logloss', 'auc', 'aucpr']  # 添加PR-AUC
        elif imbalance_ratio > 10:
            early_stopping_rounds = 80
            eval_metric_list = ['logloss', 'auc', 'aucpr']
        elif imbalance_ratio > 5:
            early_stopping_rounds = 60
            eval_metric_list = ['logloss', 'auc']
        else:
            early_stopping_rounds = 50
            eval_metric_list = ['logloss', 'auc']

        logger.info(f"优化XGBoost参数: n_estimators={n_estimators}, max_depth={max_depth}, "
                   f"learning_rate={learning_rate}, reg_alpha={reg_alpha}, reg_lambda={reg_lambda}")
        logger.info(f"采样参数: subsample={subsample}, colsample_bytree={colsample_bytree}, "
                   f"min_child_weight={min_child_weight}, gamma={gamma}")
        logger.info(f"不平衡处理: scale_pos_weight={min(imbalance_ratio, 10.0):.2f}, "
                   f"early_stopping_rounds={early_stopping_rounds}")

        # 使用2024年最佳实践的XGBoost分类器
        model = xgb.XGBClassifier(
            # 基础参数
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,

            # 采样参数（防止过拟合）
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,

            # 正则化参数
            reg_alpha=reg_alpha,  # L1正则化
            reg_lambda=reg_lambda,  # L2正则化
            min_child_weight=min_child_weight,  # 最小子节点权重
            gamma=gamma,  # 分裂门槛

            # 系统参数
            random_state=42,
            n_jobs=-1,
            verbosity=0,

            # 训练控制（动态调整）
            eval_metric='logloss',  # 二分类对数损失
            early_stopping_rounds=early_stopping_rounds,  # 根据不平衡比例动态调整

            # 二分类设置
            objective='binary:logistic',  # 二分类逻辑回归

            # 性能优化（2024年最佳实践）
            tree_method='hist',  # 直方图算法，适合高维数据
            max_bin=512,  # 增加分箱数量，提高精度
            grow_policy='depthwise',  # 深度优先生长，适合金融数据

            # 类别平衡（关键优化）
            scale_pos_weight=min(imbalance_ratio, 10.0),  # 限制最大权重为10，避免过度补偿

            # 内存优化
            max_delta_step=1,  # 限制每步更新，提高稳定性

            # 高级优化参数（2024年最佳实践）
            monotone_constraints=None,  # 可根据金融逻辑设置单调性约束
            interaction_constraints=None,  # 可设置特征交互约束
            validate_parameters=True,  # 参数验证
        )

        # 计算样本权重（进一步优化不平衡问题）
        sample_weights = _calculate_professional_sample_weights(
            y_train_split, imbalance_ratio, logger
        )

        # 训练模型（使用专业的训练策略）
        logger.info("开始训练XGBoost模型...")
        logger.info(f"使用样本权重: 正类权重={sample_weights[y_train_split==1][0]:.2f}, "
                   f"负类权重={sample_weights[y_train_split==0][0]:.2f}")

        model.fit(
            X_train_split, y_train_split,
            sample_weight=sample_weights,
            eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],
            eval_names=['train', 'valid'],
            eval_metric=eval_metric_list,  # 根据不平衡程度动态选择评估指标
            verbose=False
        )

        # 获取最佳迭代次数
        best_iteration = model.best_iteration
        logger.info(f"最佳迭代次数: {best_iteration}")

        # 输出PCA信息
        if pca_info['applied']:
            logger.info(f"PCA降维信息: {pca_info['original_features']} -> {pca_info['pca_components']} "
                       f"(方差解释: {pca_info['variance_explained']:.1%})")
        else:
            logger.info(f"未使用PCA降维: {pca_info['reason']}")

        # 验证模型 - 使用专业量化评价指标
        y_pred = model.predict(X_val_split)
        y_pred_proba = model.predict_proba(X_val_split)

        logger.info(f"模型训练完成")

        # 计算专业量化指标
        quant_metrics = _calculate_professional_quant_metrics(y_val_split, y_pred, y_pred_proba, feature_names)

        logger.info("=== 专业量化评价指标 ===")
        logger.info(f"Precision@50:  {quant_metrics['precision_at_50']:.3f}")
        logger.info(f"Precision@100: {quant_metrics['precision_at_100']:.3f}")
        logger.info(f"Precision@200: {quant_metrics['precision_at_200']:.3f}")
        logger.info(f"Recall (Top10%召回率): {quant_metrics['recall']:.3f}")
        logger.info(f"IC (信息系数): {quant_metrics['ic_pearson']:.3f}")
        logger.info(f"RankIC (排名信息系数): {quant_metrics['rank_ic']:.3f}")
        logger.info(f"ICIR (信息比率): {quant_metrics['icir_approx']:.3f}")
        logger.info(f"AUC-ROC: {quant_metrics['auc_roc']:.3f}")
        logger.info(f"选股精确度: {quant_metrics['stock_selection_precision']:.3f}")
        logger.info(f"预测概率分布质量: 标准差={quant_metrics['prediction_spread']:.3f}, 范围={quant_metrics['prediction_range']:.3f}")

        # 传统指标作为补充
        logger.info("=== 传统分类指标 (补充) ===")
        logger.info(f"Precision: {quant_metrics['precision']:.3f}")
        logger.info(f"F1-Score: {quant_metrics['f1_score']:.3f}")

        # 混淆矩阵
        conf_matrix = confusion_matrix(y_val_split, y_pred)
        logger.info(f"混淆矩阵:\n{conf_matrix}")

        # 特征重要性分析
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        logger.info("前20个重要特征:")
        for idx, row in importance_df.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # 分析特征重要性分布
        total_importance = importance_df['importance'].sum()
        top_10_importance = importance_df.head(10)['importance'].sum()
        top_20_importance = importance_df.head(20)['importance'].sum()

        logger.info(f"特征重要性分布:")
        logger.info(f"  前10个特征贡献: {top_10_importance/total_importance:.1%}")
        logger.info(f"  前20个特征贡献: {top_20_importance/total_importance:.1%}")
        logger.info(f"  有效特征数量(重要性>0.001): {len(importance_df[importance_df['importance'] > 0.001])}")

        # 按特征类型分析
        # 单日微观结构特征
        single_day_microstructure = importance_df[importance_df['feature'].str.contains(
            'shadow_ratio|body_ratio|reversal_score|gap_pct|momentum|pressure|structure_score', na=False
        )]

        # DTW形态相似度特征
        dtw_pattern_features = importance_df[importance_df['feature'].str.contains(
            'similarity$', na=False
        )]

        # 形态解构特征
        pattern_decomposition = importance_df[importance_df['feature'].str.contains(
            'vol_ratio|vol_profile|rs_improvement|vol_progression|symmetry|neckline|depth_ratio|contraction|rounding', na=False
        )]

        # 传统技术指标
        technical_features_analysis = importance_df[importance_df['feature'].str.contains('ma|rsi|macd|bb_', na=False)]
        volume_features_analysis = importance_df[importance_df['feature'].str.contains('volume|turnover', na=False)]

        if len(single_day_microstructure) > 0:
            logger.info(f"单日微观结构特征重要性: {single_day_microstructure['importance'].sum():.4f}")
        if len(dtw_pattern_features) > 0:
            logger.info(f"DTW形态相似度特征重要性: {dtw_pattern_features['importance'].sum():.4f}")
        if len(pattern_decomposition) > 0:
            logger.info(f"形态解构特征重要性: {pattern_decomposition['importance'].sum():.4f}")
        if len(technical_features_analysis) > 0:
            logger.info(f"技术指标特征重要性: {technical_features_analysis['importance'].sum():.4f}")
        if len(volume_features_analysis) > 0:
            logger.info(f"成交量特征重要性: {volume_features_analysis['importance'].sum():.4f}")

        # 预测概率分布分析（二分类）
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            class_names = ['其他股票', 'Top10%股票']
            for class_idx, class_name in enumerate(class_names):
                class_proba = y_pred_proba[:, class_idx]
                logger.info(f"{class_name}预测概率: 均值={class_proba.mean():.3f}, "
                           f"标准差={class_proba.std():.3f}")
        else:
            # 单列概率输出
            logger.info(f"Top10%预测概率: 均值={y_pred_proba.mean():.3f}, "
                       f"标准差={y_pred_proba.std():.3f}")

        # 模型可解释性分析
        _analyze_model_interpretability(model, feature_names, X_val_split, y_val_split, logger)

        return model, scaler

    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

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
        # 计算特征重要性
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
                        (daily_data['volume'] > 0) | (daily_data['amount'] > 0)
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
        # 第一步：检查当日数据异常（成交量=0且成交额=0的股票）
        current_day_invalid = (
            (data['volume'] == 0) & (data['amount'] == 0)
        )

        current_day_invalid_count = current_day_invalid.sum()
        if current_day_invalid_count > 0:
            print(f"🚫 当日数据异常过滤: 过滤掉 {current_day_invalid_count} 只股票（成交量=0且成交额=0）")
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

def _select_top_sectors_by_limit_up(sector_data, date_str, logger):
    """
    基于涨停个数筛选潜力板块

    Args:
        sector_data: 板块数据DataFrame
        date_str: 当前日期字符串 (YYYYMMDD格式)
        logger: 日志记录器

    Returns:
        list: 前5个潜力板块代码列表
    """
    logger.info(f"📊 {date_str}: 开始基于涨停个数筛选潜力板块")

    # 获取当天及前9天的日期（共10天）
    current_date = pd.Timestamp(date_str)
    date_range = []
    for i in range(10):
        check_date = current_date - pd.Timedelta(days=i)
        date_range.append(check_date)

    # 获取这10天内的板块数据
    historical_sector_data = sector_data[sector_data['date'].isin(date_range)]

    if historical_sector_data.empty:
        logger.warning(f"❌ {date_str}: 没有找到历史板块数据")
        return []

    # 计算每个板块在过去10天中单日最大涨停个数
    sector_max_limit_up = {}

    # 获取所有板块代码
    all_sectors = historical_sector_data['sector_code'].unique()

    for sector_code in all_sectors:
        sector_historical = historical_sector_data[
            historical_sector_data['sector_code'] == sector_code
        ]

        # 计算该板块在过去10天中每天的涨停个数，取最大值
        if 'limit_up_ratio' in sector_historical.columns:
            max_limit_up = sector_historical['limit_up_ratio'].max()
            sector_max_limit_up[sector_code] = max_limit_up if pd.notna(max_limit_up) else 0
        else:
            # 如果没有limit_up_ratio字段，使用其他涨停相关字段
            if 'limit_up_count' in sector_historical.columns:
                max_count = sector_historical['limit_up_count'].max()
                sector_max_limit_up[sector_code] = max_count if pd.notna(max_count) else 0
            else:
                sector_max_limit_up[sector_code] = 0

    # 按涨停个数排序，选择前5个板块
    sorted_sectors = sorted(sector_max_limit_up.items(), key=lambda x: x[1], reverse=True)
    top_sectors = [sector_code for sector_code, _ in sorted_sectors[:5]]

    # 输出筛选结果
    logger.info(f"🎯 {date_str}: 涨停个数排名前5的板块:")
    for i, (sector_code, count) in enumerate(sorted_sectors[:5], 1):
        logger.info(f"   {i}. {sector_code}: 最大涨停个数 {count:.1f}")

    return top_sectors

def _run_ml_stock_selection(data_loader, stock_selector, ml_model, scaler, feature_names,
                           start_date: str, end_date: str):
    """使用机器学习模型进行选股"""
    try:
        from clean_ml_system.sector_potential_analyzer import SectorPotentialAnalyzer
        import numpy as np
        logger = logging.getLogger(__name__)
        # 初始化组件
        sector_analyzer = SectorPotentialAnalyzer()
        all_selections = []

        # 加载测试期数据
        logger.info(f"开始加载测试期数据: {start_date} - {end_date}")
        try:
            stock_data, sector_data, market_data = data_loader.load_all_data(
                start_date, end_date, extend_history_days=80
            )
        except Exception as e:
            logger.error(f"❌ 数据加载过程中发生异常: {e}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return []

        if stock_data is None or stock_data.empty:
            logger.error("❌ 测试期数据加载失败")
            logger.error(f"测试日期范围: {start_date} - {end_date}")
            logger.error("请检查以下问题:")
            logger.error("1. 特征文件是否存在于 local_data/features/daily/ 目录")
            logger.error("2. 日期范围内是否有有效的交易日")
            logger.error("3. 特征文件格式是否正确")
            logger.error("4. 文件权限是否正确")
            return []

        # 计算增强特征
        logger.info(f"原始股票数据日期范围: {stock_data['date'].min()} - {stock_data['date'].max()}")
        logger.info(f"原始股票数据记录数: {len(stock_data)}")

        # 清理特征计算器的缓存，确保使用最新的数据范围
        # 注释掉清理缓存，启用增量计算以提升性能
        # logger.info("清理特征计算器缓存以确保数据完整性")
        # stock_selector.multi_factor_scorer.feature_calculator.clear_cache()

        enhanced_stock_data = stock_selector.multi_factor_scorer.feature_calculator.calculate_missing_features(stock_data)

        logger.info(f"增强后股票数据日期范围: {enhanced_stock_data['date'].min()} - {enhanced_stock_data['date'].max()}")
        logger.info(f"增强后股票数据记录数: {len(enhanced_stock_data)}")

        # 获取交易日
        trading_dates = data_loader.get_trading_dates(start_date, end_date)

        # 确保交易日期按时间顺序排列（从早到晚）
        trading_dates = sorted(trading_dates)

        print(f"开始测试期选股，共 {len(trading_dates)} 个交易日")
        print(f"交易日期范围: {trading_dates[0]} - {trading_dates[-1]}")

        # 检查前几个日期的数据可用性
        for i, date_str in enumerate(trading_dates[:10]):
            date_obj = pd.to_datetime(date_str)
            daily_count = len(enhanced_stock_data[enhanced_stock_data['date'] == date_obj])
            logger.info(f"日期 {date_str}: {daily_count} 条股票数据")
            if daily_count == 0:
                logger.warning(f"❌ 日期 {date_str} 没有股票数据，将被跳过")

        # 按日期进行选股
        for i, date_str in enumerate(trading_dates):
            try:
                if i % 20 == 0:  # 每20天显示一次进度
                    print(f"进度: {i+1}/{len(trading_dates)} ({date_str})")

                date_obj = pd.to_datetime(date_str)

                # 获取当日数据
                daily_stock_data = enhanced_stock_data[enhanced_stock_data['date'] == date_obj]
                daily_sector_data = sector_data[sector_data['date'] == date_obj]

                if daily_stock_data.empty or daily_sector_data.empty:
                    continue

                # 识别潜力板块
                # potential_results = sector_analyzer.analyze_daily_potential(daily_sector_data)
                # if not potential_results:
                #     logger.warning(f"❌ {date_str}: 没有发现潜力板块，跳过该日期")
                #     continue

                # 获取前5个潜力板块
                # top_sectors = [r.sector_code for r in potential_results[:5]]

                # 基于涨停个数筛选潜力板块
                # top_sectors = _select_top_sectors_by_limit_up(sector_data, date_str, logger)

                # if not top_sectors:
                #     logger.warning(f"❌ {date_str}: 没有筛选出潜力板块，跳过该日期")
                #     continue

                # # 获取潜力板块中的股票
                # sector_stocks = data_loader.get_stocks_in_sectors(top_sectors, date_str)

                # 使用最强板块数据获取潜力板块股票
                # try:
                #     # 导入ths_sector模块
                #     import sys
                #     from pathlib import Path
                #     ths_sector_path = Path(__file__).parent.parent / "ths_sector"
                #     if str(ths_sector_path) not in sys.path:
                #         sys.path.append(str(ths_sector_path))

                #     from ths_data_downloader import THSSectorDataDownloader

                #     # 创建下载器实例，指定正确的数据路径
                #     ths_downloader = THSSectorDataDownloader()
                #     # 修正数据路径为绝对路径
                #     ths_downloader.base_path = Path(__file__).parent.parent / "ths_sector" / "local_data" / "ths_sector"

                #     # 获取当日最强板块前6名
                #     top_hot_sectors = ths_downloader.get_top_hot_sectors(date_str, top_n=6)

                #     if not top_hot_sectors:
                #         logger.warning(f"❌ {date_str}: 没有获取到最强板块数据，跳过该日期")
                #         continue

                #     logger.info(f"🔥 {date_str}: 获取到前6强板块: {top_hot_sectors}")

                #     # 获取这些板块的所有股票
                #     sector_stocks = []
                #     for sector_code in top_hot_sectors:
                #         stocks = ths_downloader.get_stocks_by_sector_code(sector_code)
                #         sector_stocks.extend(stocks)

                #     # 去重
                #     sector_stocks = list(set(sector_stocks))
                #     logger.info(f"📈 {date_str}: 最强板块包含 {len(sector_stocks)} 只股票")

                # except Exception as e:
                #     logger.error(f"❌ {date_str}: 获取最强板块股票失败: {e}")
                #     continue
                # if not sector_stocks:
                #     logger.warning(f"❌ {date_str}: 潜力板块中没有股票，跳过该日期")
                #     continue
                # daily_filtered = daily_stock_data
                
                daily_filtered = daily_stock_data[(daily_stock_data['is_can_buy'] == 1)].copy()
                # 过滤当日涨幅超过3%的股票（让模型自己学习下跌风险）
                # daily_filtered = daily_stock_data[
                #     (daily_stock_data['stock_code'].isin(sector_stocks)) &
                #     (daily_stock_data['pct_chg'] <= 3.0) & (daily_stock_data['pct_chg'] >= -4.0)  # 只避免追高，不过滤下跌股票
                # ].copy()
                # daily_filtered = daily_stock_data[
                #     (daily_stock_data['pct_chg'] <= 3.0) & (daily_stock_data['pct_chg'] >= -4.0)  # 只避免追高，不过滤下跌股票
                # ].copy()

                # 过滤新股（测试阶段）
                if len(daily_filtered) > 0:
                    new_stock_filter = _filter_new_stocks(daily_filtered, date_str)
                    daily_filtered = daily_filtered[new_stock_filter]
                    logger.info(f"📅 {date_str}: 新股过滤后剩余 {len(daily_filtered)} 只候选股票")

                # 过滤多均线空头趋势股票（新增）
                # if len(daily_filtered) > 0:
                #     bearish_filter = _filter_bearish_ma_trend_stocks(daily_filtered, date_str, logger)
                #     daily_filtered = daily_filtered[bearish_filter]
                #     logger.info(f"📅 {date_str}: 多均线空头趋势过滤后剩余 {len(daily_filtered)} 只候选股票")

                # 过滤ST股票（测试阶段）
                if len(daily_filtered) > 0:
                    st_stock_filter = _filter_st_stocks(daily_filtered)
                    daily_filtered = daily_filtered[st_stock_filter]
                    logger.info(f"📅 {date_str}: ST股票过滤后剩余 {len(daily_filtered)} 只候选股票")

                if len(daily_filtered) == 0:
                    logger.warning(f"❌ {date_str}: 经过过滤后，没有候选股票")
                    continue

                # 准备特征
                missing_features = [col for col in feature_names if col not in daily_filtered.columns]
                if missing_features:
                    logger.warning(f"❌ {date_str}: 缺失关键特征: {missing_features[:10]}...")  # 只显示前10个
                    continue

                # 准备特征数据并处理异常值
                X_daily = daily_filtered[feature_names].fillna(0)

                # 处理无穷大值和过大数值
                # 将inf替换为NaN，然后用0填充
                X_daily = X_daily.replace([np.inf, -np.inf], np.nan).fillna(0)

                # 限制数值范围，避免过大的值
                X_daily = X_daily.clip(-1e10, 1e10)

                X_daily = X_daily.values

                # 标准化特征
                X_scaled = scaler.transform(X_daily)

                # 模型预测
                predictions = ml_model.predict(X_scaled)
                probabilities = ml_model.predict_proba(X_scaled)

                # 使用改进的选股逻辑
                from improved_selection_logic import ImprovedStockSelector

                # 初始化选股器
                selector = ImprovedStockSelector()

                # 计算动态阈值（基于当前市场环境）
                selector.dynamic_thresholds = {
                    'confidence': 0.65,  # 更高的置信度要求
                    'advantage': 0.20,   # 更高的优势要求
                    'market_regime': 'normal'
                }

                # 设置形态权重
                selector.pattern_weights = {
                    'bald_bearish_penalty': 0.15,  # 光头阴线惩罚
                    'consecutive_decline_penalty': 0.10,  # 连续下跌惩罚
                    'high_rsi_penalty': 0.12,  # 高RSI惩罚
                    'long_upper_shadow_penalty': 0.20,  # 长上影线惩罚
                    'below_all_ma_penalty': 0.18  # 股价在所有均线下方惩罚
                }

                # 应用增强选股逻辑
                candidate_stocks = selector.enhanced_stock_selection(
                    daily_filtered, predictions, probabilities, feature_names
                )



                selected_indices = []
                selected_stock_info = None

                # 选择评分最高的股票
                if candidate_stocks:
                    # 按最终评分排序
                    sorted_candidates = sorted(candidate_stocks, key=lambda x: x['final_score'], reverse=True)
                    best_stock = sorted_candidates[0]
                    selected_indices = [best_stock['index']]

                    # 保存选中股票的详细信息，用于后续创建StockSelection对象
                    selected_stock_info = {
                        'index': best_stock['index'],
                        'stock_code': best_stock['stock_code'],
                        'final_score': best_stock['final_score'],
                        'rise_prob': best_stock['rise_prob'],
                        'fall_prob': best_stock['fall_prob'],
                        'rise_advantage': best_stock['rise_advantage'],
                        'base_score': best_stock['base_score'],
                        'penalty_score': best_stock['penalty_score'],
                        'penalty_reasons': best_stock['penalty_reasons'],
                        'category': best_stock['category'],
                        'prediction': best_stock['prediction']
                    }

                    logger.info(f"✅ 选中{best_stock['category']}股票 {best_stock['stock_code']}: 评分{best_stock['final_score']:.3f}")

                # 如果没有符合条件的股票，跳过该日期
                if not selected_indices:
                    logger.info(f"❌ 日期 {date_str}: 没有符合质量要求的股票，跳过")
                    logger.info(f"  候选股票数: {len(candidate_stocks)}")

                    # 详细诊断信息：分析为什么没有候选股票
                    if len(predictions) > 0:
                        # 分析预测分布（二分类）
                        pred_counts = np.bincount(predictions, minlength=2)
                        total_stocks = len(predictions)
                        logger.info(f"  📊 预测分布: 其他股票{pred_counts[0]}({pred_counts[0]/total_stocks:.1%}), "
                              f"Top10%股票{pred_counts[1]}({pred_counts[1]/total_stocks:.1%})")

                        # 分析Top10%概率分布
                        top10_probs = probabilities[:, 1]  # Top10%概率
                        print(f"  📈 Top10%概率分布: 平均{top10_probs.mean():.3f}, 最大{top10_probs.max():.3f}, 最小{top10_probs.min():.3f}")

                        # 分析阈值过滤情况
                        high_prob_count = (top10_probs >= 0.65).sum()
                        print(f"  🎯 Top10%概率≥65%的股票数: {high_prob_count}")

                        if high_prob_count > 0:
                            # 进一步分析为什么这些股票没有被选中
                            print(f"  🔍 详细分析Top10%概率≥65%的股票:")
                            high_prob_indices = np.where(top10_probs >= 0.65)[0]
                            for idx in high_prob_indices[:5]:  # 只显示前5个
                                stock_code = daily_filtered.iloc[idx]['stock_code']
                                top10_prob = top10_probs[idx]
                                other_prob = probabilities[idx, 0]
                                advantage = top10_prob - other_prob
                                prediction = predictions[idx]
                                logger.info(f"    {stock_code}: Top10%概率{top10_prob:.3f}, 优势{advantage:.3f}, 预测类别{prediction}")

                        # 检查是否有形态惩罚导致的过滤
                        print(f"  ⚠️ 可能的过滤原因:")
                        print(f"    1. Top10%概率不足65%")
                        print(f"    2. Top10%优势不足20%")
                        print(f"    3. 预测类别不是Top10%(1)")
                        print(f"    4. 形态惩罚导致最终评分≤0")
                    continue

                # 创建选股结果
                for idx in selected_indices:
                    stock_row = daily_filtered.iloc[idx]

                    # 使用改进选股逻辑的结果
                    if selected_stock_info and selected_stock_info['index'] == idx:
                        # 使用改进逻辑的评分和置信度
                        composite_score = selected_stock_info['final_score']
                        confidence = selected_stock_info['rise_prob']  # 使用上涨概率作为置信度
                        expected_return = 0.05 if selected_stock_info['prediction'] == 1 else 0.02  # Top10%期望5%收益

                    else:
                        # 降级到原始逻辑（不应该发生）
                        prediction_prob = float(probabilities[idx, predictions[idx]])
                        prob_entropy = -np.sum(probabilities[idx] * np.log(probabilities[idx] + 1e-8))
                        certainty_score = 1.0 - (prob_entropy / np.log(2))  # 二分类熵
                        composite_score = 0.7 * prediction_prob + 0.3 * certainty_score

                        sorted_probs = np.sort(probabilities[idx])[::-1]
                        prob_gap = sorted_probs[0] - sorted_probs[1]
                        confidence = prediction_prob * (1 + prob_gap)
                        expected_return = 0.05 if predictions[idx] == 3 else 0.02

                    # 创建StockSelection对象
                    from stock_selector import StockSelection
                    selection = StockSelection(
                        date=date_str,
                        stock_code=stock_row['stock_code'],
                        stock_name=stock_row.get('stock_name', ''),
                        composite_score=float(composite_score),
                        confidence=float(min(confidence, 1.0)),  # 限制在1.0以内
                        expected_return=float(expected_return),
                        risk_adjusted_score=float(confidence),  # 使用置信度作为风险调整评分
                        technical_score=0.0,
                        fundamental_score=0.0,
                        sector_score=0.0,
                        capital_flow_score=0.0,
                        selected_sectors=top_hot_sectors if 'top_hot_sectors' in locals() else []
                    )

                    # 添加详细的预测信息和特征值（用于后续分析）
                    selection.prediction_details = {
                        'predicted_class': int(predictions[idx]),
                        'prediction_proba': {
                            'class_0_其他股票': float(probabilities[idx, 0]),
                            'class_1_Top10%股票': float(probabilities[idx, 1])
                        },
                        'features': {feature_name: float(stock_row[feature_name])
                                   for feature_name in feature_names if feature_name in stock_row}
                    }

                    if selected_stock_info:
                        selection.enhanced_selection_info = {
                            'final_score': selected_stock_info['final_score'],
                            'rise_prob': selected_stock_info['rise_prob'],
                            'fall_prob': selected_stock_info['fall_prob'],
                            'rise_advantage': selected_stock_info['rise_advantage'],
                            'base_score': selected_stock_info['base_score'],
                            'penalty_score': selected_stock_info['penalty_score'],
                            'penalty_reasons': selected_stock_info['penalty_reasons'],
                            'category': selected_stock_info['category']
                        }

                    all_selections.append(selection)

            except Exception as e:
                logger.error(f"处理日期 {date_str} 失败: {e}")
                import traceback
                logger.error(f"异常详情: {traceback.format_exc()}")
                logger.error(f"当前处理状态:")
                logger.error(f"  - 日期: {date_str}")
                logger.error(f"  - 当日股票数据: {len(daily_stock_data) if 'daily_stock_data' in locals() else '未获取'}")
                logger.error(f"  - 板块数据: {len(top_hot_sectors) if 'top_hot_sectors' in locals() else '未获取'}")
                continue

            # 内存优化：每处理10个日期进行一次垃圾回收
            if len(all_selections) % 10 == 0:
                import gc
                gc.collect()

        print(f"✅ 测试期选股完成，共选出 {len(all_selections)} 只股票")

        # 如果没有选出任何股票，提供详细的诊断信息
        if len(all_selections) == 0:
            logger.error("❌ 未选出任何股票，诊断信息:")
            logger.error(f"  - 处理的日期范围: {start_date} - {end_date}")
            logger.error(f"  - 股票数据总记录数: {len(stock_data) if stock_data is not None else 0}")
            logger.error(f"  - 板块数据总记录数: {len(sector_data) if sector_data is not None else 0}")
            logger.error(f"  - 市场数据总记录数: {len(market_data) if market_data is not None else 0}")

            # 检查特征数据
            if stock_data is not None and not stock_data.empty:
                logger.error(f"  - 股票数据特征数: {len(stock_data.columns)}")
                logger.error(f"  - 股票数据日期范围: {stock_data['date'].min()} - {stock_data['date'].max()}")
                logger.error(f"  - 唯一股票数: {stock_data['stock_code'].nunique()}")

            logger.error("  可能的原因:")
            logger.error("    1. 所有日期的股票都不满足选股条件")
            logger.error("    2. 模型预测结果都不符合阈值要求")
            logger.error("    3. 板块分析没有找到潜力板块")
            logger.error("    4. 新股过滤过于严格")
            logger.error("    5. 特征数据质量问题")

        return all_selections

    except Exception as e:
        logger.error(f"❌ 机器学习选股失败: {e}")
        import traceback
        logger.error(f"异常详情: {traceback.format_exc()}")
        return []


def _get_model_cache_path(train_start_date: str, train_end_date: str) -> Path:
    """获取模型缓存路径"""
    cache_dir = Path("cache/models")
    cache_dir.mkdir(parents=True, exist_ok=True)

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

def run_ml_pattern_selection_mode():
    """运行基于机器学习的K线形态选股模式"""
    logger = logging.getLogger(__name__)
    print("🤖 运行模式: 机器学习K线形态选股")
    print("=" * 60)

    # 清理过期缓存
    _clear_old_cache()

    try:
        # 导入必要的机器学习库
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import classification_report, accuracy_score
            import xgboost as xgb
        except ImportError as e:
            logger.error(f"缺少必要的机器学习库: {e}")
            print("❌ 请安装必要的机器学习库: pip install scikit-learn xgboost")
            return False, None, None

        # 1. 初始化组件
        data_loader = IntegratedDataLoader()
        stock_selector = StockSelector()

        # 2. 设置训练和测试期间
        train_start_date = "20230601"
        train_end_date = "20241230"  # 训练期结束时间
        test_start_date = "20250101"  # 测试期开始时间（无重叠）
        test_end_date = "20250725"  # 调整为有下一个交易日数据的最后日期

        print(f"📊 训练期: {train_start_date} - {train_end_date}")
        print(f"🧪 测试期: {test_start_date} - {test_end_date}")

        # 3. 检查模型缓存
        print(f"🔍 检查模型缓存...")
        cached_model, cached_scaler, cached_feature_names = _load_model_cache(train_start_date, train_end_date)

        if cached_model is not None:
            print(f"✅ 发现相同日期范围的训练模型，跳过训练过程")
            ml_model = cached_model
            scaler = cached_scaler
            feature_names = cached_feature_names
        else:
            print(f"📥 未发现缓存模型，开始训练过程...")

            # 4. 加载训练数据
            print(f"📥 加载训练数据...")
            train_stock_data, train_sector_data, train_market_data = data_loader.load_all_data(
                train_start_date, train_end_date, extend_history_days=80
            )

            if train_stock_data is None or train_stock_data.empty:
                print("❌ 训练数据加载失败")
                return False, None, None

            print(f"✅ 训练数据加载完成: {len(train_stock_data)} 条记录")

            # 5. 特征工程增强
            print(f"🔧 进行特征工程增强...")
            # 清理缓存确保使用最新特征
            # stock_selector.multi_factor_scorer.feature_calculator.clear_cache()
            enhanced_train_data = stock_selector.multi_factor_scorer.feature_calculator.calculate_missing_features(train_stock_data)

            # 6. 准备训练数据
            print(f"🎯 准备机器学习训练数据...")
            X_train, y_train, feature_names = _prepare_ml_training_data(
                enhanced_train_data, train_sector_data, data_loader
            )

            if X_train is None or len(X_train) == 0:
                logger.error("❌ 训练数据准备失败")
                return False, None, None

            logger.info(f"✅ 训练数据准备完成: {len(X_train)} 样本, {len(feature_names)} 特征")

            # 7. 训练机器学习模型
            print(f"🤖 训练机器学习模型...")
            ml_model, scaler = _train_ml_model(X_train, y_train, feature_names)

            if ml_model is None:
                print("❌ 模型训练失败")
                return False, None, None

            print(f"✅ 模型训练完成")

            # 8. 保存模型到缓存
            print(f"💾 保存模型到缓存...")
            _save_model_cache(ml_model, scaler, feature_names, train_start_date, train_end_date)

        # 9. 测试期选股
        print(f"🧪 开始测试期选股...")
        # 注意：无论模型是从缓存加载还是新训练的，都使用对应的feature_names
        test_selections = _run_ml_stock_selection(
            data_loader, stock_selector, ml_model, scaler, feature_names,
            test_start_date, test_end_date
        )

        if not test_selections:
            print("❌ 测试期选股失败")
            return False, None, None

        # 8. 计算实际收益率
        print(f"📈 计算实际收益率...")
        test_stock_data = data_loader.load_stock_features(test_start_date, test_end_date)

        if test_stock_data is not None and not test_stock_data.empty:
            # 为选股结果添加实际收益率
            selections_with_returns = []
            for selection in test_selections:
                stock_row = test_stock_data[
                    (test_stock_data['stock_code'] == selection.stock_code) &
                    (test_stock_data['date'] == pd.to_datetime(selection.date))
                ]

                if not stock_row.empty and 'future_return_1d' in stock_row.columns:
                    actual_return = stock_row['future_return_1d'].iloc[0]
                    selection.actual_return = actual_return
                else:
                    selection.actual_return = 0.0

                selections_with_returns.append(selection)
        else:
            selections_with_returns = test_selections

        # 9. 计算统计指标
        print(f"📊 计算投资组合统计指标...")
        statistics = calculate_portfolio_statistics(selections_with_returns)

        # 10. 显示结果
        print(f"\n✅ 机器学习K线形态选股完成")
        print(f"\n📊 投资组合统计分析:")
        print(f"  测试期间: {test_start_date} - {test_end_date}")
        print(f"  总交易次数: {statistics.get('total_trades', 0)} 次")
        print(f"  胜率: {statistics.get('win_rate', 0):.2%}")
        print(f"  总收益率: {statistics.get('total_return', 0):.2%}")
        print(f"  年化收益率: {statistics.get('annualized_return', 0):.2%}")
        print(f"  最大回撤: {statistics.get('max_drawdown', 0):.2%}")
        print(f"  夏普比率: {statistics.get('annualized_sharpe', 0):.3f}")
        print(f"  卡尔马比率: {statistics.get('calmar_ratio', 0):.3f}")

        return True, selections_with_returns, statistics

    except Exception as e:
        logger.error(f"❌ 机器学习K线形态选股失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


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


    if args.mode == 'ml_pattern_selection':
        success, ml_selections, ml_statistics = run_ml_pattern_selection_mode()
        if success and args.save_results:
            # 保存机器学习选股结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)

            # 保存选股结果
            selections_file = output_path / f"ml_pattern_selections_{timestamp}.json"
            selections_data = []
            for selection in ml_selections:
                selection_dict = {
                    'date': selection.date,
                    'stock_code': selection.stock_code,
                    'stock_name': selection.stock_name,
                    'composite_score': float(selection.composite_score),
                    'confidence': float(selection.confidence),
                    'expected_return': float(selection.expected_return),
                    'actual_return': float(getattr(selection, 'actual_return', 0.0)),
                }

                # 添加预测详情
                if hasattr(selection, 'prediction_details'):
                    selection_dict.update(selection.prediction_details)

                # 添加增强选股信息
                if hasattr(selection, 'enhanced_selection_info'):
                    selection_dict['enhanced_info'] = selection.enhanced_selection_info

                selections_data.append(selection_dict)

            # 转换numpy类型为Python原生类型
            selections_data_converted = convert_numpy_types(selections_data)

            with open(selections_file, 'w', encoding='utf-8') as f:
                json.dump(selections_data_converted, f, ensure_ascii=False, indent=2)

            # 保存详细分析报告
            report_file = output_path / f"ml_pattern_analysis_report_{timestamp}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("机器学习K线形态选股系统分析报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"测试期间: 20240602 - 20250122\n")
                f.write(f"选股策略: 基于机器学习的K线形态识别\n")
                f.write(f"模型类型: XGBoost分类器\n")
                f.write(f"特征工程: K线形态 + 技术指标 + 累计涨幅\n\n")

                f.write("核心指标:\n")
                f.write(f"  总交易次数: {ml_statistics.get('total_trades', 0)} 次\n")
                f.write(f"  胜率: {ml_statistics.get('win_rate', 0):.2%}\n")
                f.write(f"  总收益率: {ml_statistics.get('total_return', 0):.2%}\n")
                f.write(f"  年化收益率: {ml_statistics.get('annualized_return', 0):.2%}\n")
                f.write(f"  最大回撤: {ml_statistics.get('max_drawdown', 0):.2%}\n")
                f.write(f"  夏普比率: {ml_statistics.get('annualized_sharpe', 0):.3f}\n")
                f.write(f"  卡尔马比率: {ml_statistics.get('calmar_ratio', 0):.3f}\n\n")

                f.write("策略特色:\n")
                f.write("  ✅ 基于机器学习的特征组合学习\n")
                f.write("  ✅ K线形态自动识别\n")
                f.write("  ✅ 潜力板块动态筛选\n")
                f.write("  ✅ 避免追高的风险控制\n")
                f.write("  ✅ 多维度特征融合\n\n")

                # 策略评估
                win_rate = ml_statistics.get('win_rate', 0)
                sharpe = ml_statistics.get('annualized_sharpe', 0)
                max_dd = ml_statistics.get('max_drawdown', 0)

                f.write("策略评估:\n")
                if win_rate > 0.55 and sharpe > 1.0 and max_dd > -0.15:
                    f.write("  ✅ 机器学习策略表现优秀，建议实盘应用\n")
                elif win_rate > 0.50 and sharpe > 0.5:
                    f.write("  ⭐ 机器学习策略表现良好，可小仓位试验\n")
                else:
                    f.write("  ⚠️  机器学习策略需要进一步优化\n")

            logger.info(f"✅ 机器学习选股结果已保存: {selections_file}")
            logger.info(f"✅ 详细分析报告已保存: {report_file}")

        return success


    else:
        logger.error(f"未知的运行模式: {args.mode}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
