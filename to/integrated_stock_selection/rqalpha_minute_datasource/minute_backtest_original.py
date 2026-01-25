"""
5分钟级别的选股回测系统

基于现有_run_stock_selection_backtest的逻辑，实现5分钟级别的选股回测。
支持Top-K选股（K=1,2,3,5），保持原有的选股策略和交易逻辑。

作者: Claude Code Assistant
日期: 2025-11-22
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class MinuteLevelStockSelector:
    """
    5分钟级别选股回测系统

    基于现有的日线选股逻辑，适配5分钟级别回测
    """

    def __init__(self, minute_data_path: str, prediction_data: pd.DataFrame):
        """
        初始化5分钟选股回测系统

        Args:
            minute_data_path: 5分钟数据路径
            prediction_data: 预测数据（包含预测字段和筛选条件）
        """
        self.minute_data_path = minute_data_path
        self.prediction_data = prediction_data
        self.data_cache = {}  # 5分钟数据缓存

        logger.info("初始化5分钟级别选股回测系统")
        self._validate_prediction_data()
        self._load_minute_data()

    def _validate_prediction_data(self):
        """验证预测数据的必要字段"""
        required_columns = [
            'date', 'stock_code', 'predicted_return', 'future_return_2d', 'is_can_buy',
            'pct_chg', 'volume', 'amount', 'open', 'high', 'low', 'close',
            'atr14', 'price_ema20_ratio', 'price_ema50_ratio', 'ema20_ema50_ratio',
            'ema50_slope', 'log_price_regression_slope', 'log_price_regression_r2',
            'adx_14', 'drawdown_from_20d_high_pct', 'drawdown_atr_ratio', 'atr20',
            'negative_days_5d', 'large_red_candle_5d', 'distance_to_support_atr',
            'bollinger_percent_b', 'rsi_14', 'is_limit_up', 'is_limit_down',
            'is_zhaban', 'limit_strength', 'limit_times', 'consecutive_limit_up',
            'consecutive_limit_down', 'near_support', 'near_resistance', 'short_reversal'
        ]

        missing_columns = [col for col in required_columns if col not in self.prediction_data.columns]
        if missing_columns:
            logger.warning(f"预测数据缺少必要字段: {missing_columns}")
            # 设置默认值
            for col in missing_columns:
                if col in ['date', 'stock_code']:
                    continue
                elif col in ['predicted_return', 'future_return_2d', 'pct_chg']:
                    self.prediction_data[col] = 0.0
                elif col in ['is_can_buy', 'is_limit_up', 'is_limit_down', 'is_zhaban',
                           'near_support', 'near_resistance', 'short_reversal']:
                    self.prediction_data[col] = 1 if col == 'is_can_buy' else 0
                elif col in ['limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
                           'negative_days_5d', 'large_red_candle_5d']:
                    self.prediction_data[col] = 0
                else:
                    self.prediction_data[col] = 0.0

        logger.info(f"预测数据验证完成，共{len(self.prediction_data)}条记录")

    def _load_minute_data(self):
        """加载5分钟数据"""
        try:
            import glob
            import os

            # 获取所有月份目录
            month_dirs = glob.glob(os.path.join(self.minute_data_path, "*/"))
            logger.info(f"找到{len(month_dirs)}个月份的数据目录")

            # 按日期加载5分钟数据到缓存
            for month_dir in sorted(month_dirs):
                minute_file = os.path.join(month_dir, "minute_data.parquet")
                if os.path.exists(minute_file):
                    try:
                        # 使用pyarrow读取以避免numpy兼容性问题
                        import pyarrow.parquet as pq

                        # 按日期分组读取
                        table = pq.read_table(minute_file)
                        df = table.to_pandas()

                        if not df.empty:
                            # 按日期分组缓存
                            df['date_key'] = df['datetime'].dt.date
                            for date_key, date_group in df.groupby('date_key'):
                                date_str = date_key.strftime('%Y-%m-%d')
                                if date_str not in self.data_cache:
                                    self.data_cache[date_str] = []
                                self.data_cache[date_str].append(date_group)

                            logger.info(f"加载{month_dir}数据完成，覆盖{len(df['datetime'].dt.date.unique())}个交易日")

                    except Exception as e:
                        logger.error(f"加载{minute_file}失败: {e}")
                        continue

            # 合并每日的数据
            for date_str in self.data_cache:
                if isinstance(self.data_cache[date_str], list):
                    self.data_cache[date_str] = pd.concat(
                        self.data_cache[date_str], ignore_index=True
                    ).sort_values('datetime')

            logger.info(f"5分钟数据加载完成，共缓存{len(self.data_cache)}个交易日的数据")

        except Exception as e:
            logger.error(f"加载5分钟数据失败: {e}")
            self.data_cache = {}

    def _get_minute_data_for_date(self, date: datetime.date) -> pd.DataFrame:
        """获取指定日期的5分钟数据"""
        date_str = date.strftime('%Y-%m-%d')
        return self.data_cache.get(date_str, pd.DataFrame())

    def _apply_stock_filters(self, stock_data: pd.Series, minute_df: pd.DataFrame) -> bool:
        """
        应用股票过滤条件（复用原有逻辑）

        Args:
            stock_data: 股票预测数据
            minute_df: 当日5分钟数据

        Returns:
            bool: 是否通过过滤
        """
        try:
            stock_code = stock_data['stock_code']

            # 检查当日是否有5分钟数据
            stock_minute_data = minute_df[minute_df['stock_code'] == stock_code]
            if stock_minute_data.empty:
                return False

            # 1. 基础过滤条件
            if stock_data['is_can_buy'] != 1:
                return False

            # 北交所股票过滤
            if stock_code.startswith(('8', '4')):
                return False

            # ST股过滤
            if any(x in stock_code for x in ['ST', 'st', '*ST', '*st']):
                return False

            # 2. 价格过滤条件
            last_price = stock_data['close']
            if last_price <= 1.0 or last_price >= 10000:
                return False

            # 3. 成交量过滤条件
            avg_volume = stock_data['volume']
            if avg_volume < 1000000:  # 最低成交额100万
                return False

            # 4. 涨跌停状态过滤
            if stock_data['is_limit_up'] == 1 or stock_data['is_limit_down'] == 1:
                return False

            if stock_data['is_zhaban'] == 1:
                return False

            # # 5. 技术指标过滤（复用原有逻辑）
            # if stock_data['atr14'] < 0.02:  # ATR过低
            #     return False

            # if stock_data['rsi_14'] > 80:  # RSI过高
            #     return False

            # if stock_data['bollinger_percent_b'] > 0.95:  # 布林带上轨附近
            #     return False

            # # 6. 量比过滤
            # if 'volume_ratio' in stock_data and stock_data['volume_ratio'] < 0.8:
            #     return False

            return True

        except Exception as e:
            logger.error(f"股票过滤条件判断失败 {stock_data.get('stock_code', 'unknown')}: {e}")
            return False

    def _select_top_k_stocks(self, daily_stocks: pd.DataFrame, stock_count: int) -> pd.DataFrame:
        """
        选择Top-K股票

        Args:
            daily_stocks: 当日候选股票
            stock_count: 选股数量

        Returns:
            pd.DataFrame: 选中的股票
        """
        if daily_stocks.empty:
            return daily_stocks

        # 按预测值排序（降序）
        daily_stocks_sorted = daily_stocks.sort_values('predicted_return', ascending=False)

        # 取Top-K
        selected_stocks = daily_stocks_sorted.head(stock_count)

        logger.debug(f"当日共{len(daily_stocks)}只候选股票，选择Top-{stock_count}，实际选中{len(selected_stocks)}只")
        return selected_stocks

    def _calculate_minute_returns(self, selected_stocks: pd.DataFrame, minute_df: pd.DataFrame,
                                 holding_days: int = 2) -> List[Dict]:
        """
        计算基于5分钟数据的日线级别收益

        Args:
            selected_stocks: 选中的股票
            minute_df: 当日5分钟数据
            holding_days: 持仓天数

        Returns:
            List[Dict]: 交易记录
        """
        trades = []

        for _, stock in selected_stocks.iterrows():
            stock_code = stock['stock_code']
            predicted_return = stock['predicted_return']

            # 获取该股票的5分钟数据
            stock_minute_data = minute_df[minute_df['stock_code'] == stock_code].sort_values('datetime')
            if stock_minute_data.empty:
                continue

            # 找到买入时间点（T+1日开盘买入）
            # 使用预测日期+1天作为买入日
            buy_date = stock['date'] + timedelta(days=1)

            # 获取买入日的5分钟数据，假设在前复权数据中有
            buy_date_str = buy_date.strftime('%Y-%m-%d')
            buy_minute_df = self._get_minute_data_for_date(buy_date)

            if buy_minute_df.empty:
                logger.warning(f"未找到{buy_date_str}的5分钟数据用于买入，跳过股票{stock_code}")
                continue

            stock_buy_data = buy_minute_df[buy_minute_df['stock_code'] == stock_code].sort_values('datetime')
            if stock_buy_data.empty:
                continue

            # 在买入日开盘买入（第一个5分钟数据）
            buy_price = stock_buy_data.iloc[0]['close']
            buy_datetime = stock_buy_data.iloc[0]['datetime']

            # 计算卖出时间点（T+holding_days日收盘卖出）
            sell_date = buy_date + timedelta(days=holding_days)
            sell_date_str = sell_date.strftime('%Y-%m-%d')

            # 获取卖出日的5分钟数据
            sell_minute_df = self._get_minute_data_for_date(sell_date)

            if sell_minute_df.empty:
                logger.warning(f"未找到{sell_date_str}的5分钟数据用于卖出，跳过股票{stock_code}")
                continue

            stock_sell_data = sell_minute_df[sell_minute_df['stock_code'] == stock_code].sort_values('datetime')
            if stock_sell_data.empty:
                continue

            # 在卖出日收盘卖出（最后一个5分钟数据）
            sell_price = stock_sell_data.iloc[-1]['close']
            sell_datetime = stock_sell_data.iloc[-1]['datetime']

            # 计算收益率
            actual_return = (sell_price - buy_price) / buy_price

            # 记录交易
            trade = {
                'stock_code': stock_code,
                'buy_datetime': buy_datetime,
                'sell_datetime': sell_datetime,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'holding_days': (sell_datetime.date() - buy_datetime.date()).days,
                'profit_loss': (sell_price - buy_price),
                'profit_loss_pct': actual_return * 100
            }

            trades.append(trade)

        return trades

    def run_minute_level_backtest(self, stock_counts: List[int] = [1, 2, 3, 5],
                                 holding_days: int = 2) -> Dict:
        """
        运行5分钟级别选股回测

        Args:
            stock_counts: 选股数量列表
            holding_days: 持仓天数

        Returns:
            Dict: 回测结果
        """
        logger.info(f"开始5分钟级别选股回测，选股数量: {stock_counts}, 持仓时间: {holding_days}天")

        # 交易成本设置
        SLIPPAGE_RATE = 0.001  # 滑点0.1%
        COMMISSION_RATE = 0.0003  # 手续费0.03%
        TRADING_COST_TOTAL = SLIPPAGE_RATE + COMMISSION_RATE * 2  # 总交易成本

        # 初始化回测结果
        backtest_results = {}
        detailed_trades = {}

        # 获取所有交易日
        trading_dates = sorted(self.prediction_data['date'].unique())
        logger.info(f"回测期间: {trading_dates[0]} 到 {trading_dates[-1]}, 共{len(trading_dates)}个交易日")

        for stock_count in stock_counts:
            logger.info(f"\n开始Top-{stock_count}选股回测...")

            all_trades = []
            daily_returns = []
            portfolio_values = [1.0]  # 初始资金为1

            # 按日期进行选股和交易
        import time
        start_time = time.time()

        for date_idx, trade_date in enumerate(trading_dates):
            date_str = trade_date.strftime('%Y-%m-%d')

            # 添加进度日志
            if date_idx % 10 == 0 or date_idx == len(trading_dates) - 1:
                elapsed = time.time() - start_time
                logger.info(f"处理进度: {date_idx + 1}/{len(trading_dates)} ({(date_idx + 1)/len(trading_dates)*100:.1f}%), 已用时间: {elapsed:.1f}s")

            logger.debug(f"处理日期: {date_str}")

            # 获取当日预测数据
            filter_start = time.time()
            daily_predictions = self.prediction_data[self.prediction_data['date'] == trade_date]
            if daily_predictions.empty:
                continue
            filter_time = time.time() - filter_start

            # 获取当日5分钟数据
            data_start = time.time()
            minute_df = self._get_minute_data_for_date(trade_date)
            data_time = time.time() - data_start

            if minute_df.empty:
                logger.warning(f"未找到{date_str}的5分钟数据")
                continue

            logger.debug(f"{date_str} - 预测数据: {len(daily_predictions)}条, 5分钟数据: {len(minute_df)}条")

            # 应用过滤条件
            filter_start = time.time()
            filtered_stocks = []
            for _, stock in daily_predictions.iterrows():
                if self._apply_stock_filters(stock, minute_df):
                    filtered_stocks.append(stock)
            filter_time += time.time() - filter_start

            if not filtered_stocks:
                logger.debug(f"{date_str} 没有股票通过过滤条件")
                continue

            filtered_df = pd.DataFrame(filtered_stocks)

            # 选择Top-K股票
            select_start = time.time()
            selected_stocks = self._select_top_k_stocks(filtered_df, stock_count)
            select_time = time.time() - select_start

            if selected_stocks.empty:
                continue

            # 计算分钟级收益
            calc_start = time.time()
            daily_trades = self._calculate_minute_returns(
                selected_stocks, minute_df, holding_days
            )
            calc_time = time.time() - calc_start

            if daily_trades:
                logger.debug(f"{date_str} - 选股: {len(selected_stocks)}只, 交易: {len(daily_trades)}笔, "
                            f"耗时: 过滤{filter_time:.2f}s + 选择{select_time:.2f}s + 计算{calc_time:.2f}s")

                if daily_trades:
                    all_trades.extend(daily_trades)

                    # 计算当日收益（等权重分配）
                    daily_return = np.mean([trade['actual_return'] for trade in daily_trades])
                    daily_return_after_cost = daily_return - TRADING_COST_TOTAL
                    daily_returns.append(daily_return_after_cost)

                    # 更新净值
                    new_value = portfolio_values[-1] * (1 + daily_return_after_cost)
                    portfolio_values.append(new_value)

                    logger.debug(f"{date_str} 选中{len(selected_stocks)}只股票，当日收益: {daily_return_after_cost:.4f}")

            # 计算回测统计
            if all_trades:
                stats = self._calculate_backtest_stats(all_trades, portfolio_values, daily_returns)
                backtest_results[stock_count] = stats
                detailed_trades[stock_count] = all_trades

                logger.info(f"Top-{stock_count}回测完成: "
                          f"总收益: {stats['total_return']:.4f}, "
                          f"年化收益: {stats['annual_return']:.4f}, "
                          f"夏普比率: {stats['sharpe_ratio']:.4f}, "
                          f"最大回撤: {stats['max_drawdown']:.4f}")
            else:
                logger.warning(f"Top-{stock_count}没有产生任何交易")
                backtest_results[stock_count] = {
                    'total_trades': 0,
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_loss_ratio': 0.0
                }

        return {
            'backtest_results': backtest_results,
            'detailed_trades': detailed_trades,
            'summary_stats': {
                'trading_days': len(trading_dates),
                'holding_days': holding_days,
                'trading_cost': TRADING_COST_TOTAL
            }
        }

    def _calculate_backtest_stats(self, trades: List[Dict], portfolio_values: List[float],
                                daily_returns: List[float]) -> Dict:
        """
        计算回测统计指标

        Args:
            trades: 交易记录
            portfolio_values: 净值序列
            daily_returns: 日收益率序列

        Returns:
            Dict: 统计指标
        """
        if not trades:
            return {}

        # 基础统计
        total_trades = len(trades)
        profitable_trades = [t for t in trades if t['actual_return'] > 0]
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0

        # 收益统计
        total_return = portfolio_values[-1] - 1
        avg_return = np.mean([t['actual_return'] for t in trades])
        avg_profit = np.mean([t['actual_return'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['actual_return'] for t in trades if t['actual_return'] <= 0]) if total_trades > 0 else 0

        # 风险指标
        if daily_returns:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252 * 48) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # 最大回撤
        max_drawdown = 0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_loss_ratio': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'annual_return': total_return * 252 * 48 / len(trades) if trades else 0  # 年化收益
        }