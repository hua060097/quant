#!/usr/bin/env python3
"""
回测引擎
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import backtest_config
from stock_selector import StockSelection

@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annual_return: float
    sharpe_ratio: float  # 日度夏普比率（保留兼容性）
    annualized_sharpe: float  # 年化夏普比率（正确的）
    max_drawdown: float
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    profitable_trades: int
    avg_holding_days: float

    # 详细统计
    daily_returns: List[float]
    cumulative_returns: List[float]
    trade_details: List[Dict]

class Backtester:
    """回测引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.commission_rate = backtest_config.commission_rate
        self.slippage_rate = backtest_config.slippage_rate
        self.holding_days = backtest_config.holding_days
    
    def run_backtest(self, selections: List[StockSelection], 
                    stock_data: pd.DataFrame) -> BacktestResult:
        """运行回测"""
        try:
            self.logger.info(f"开始回测，选股记录: {len(selections)}条")
            
            if not selections:
                return self._create_empty_result()
            
            # 准备数据
            trades = self._prepare_trades(selections, stock_data)
            
            if not trades:
                self.logger.warning("没有有效的交易记录")
                return self._create_empty_result()
            
            # 计算交易结果
            trade_results = self._calculate_trade_results(trades, stock_data)
            
            # 计算回测指标
            backtest_result = self._calculate_backtest_metrics(trade_results)
            
            self.logger.info(f"回测完成: 总收益{backtest_result.total_return:.2%}, "
                           f"胜率{backtest_result.win_rate:.1%}")
            
            return backtest_result
            
        except Exception as e:
            self.logger.error(f"回测失败: {e}")
            return self._create_empty_result()
    
    def _prepare_trades(self, selections: List[StockSelection], 
                       stock_data: pd.DataFrame) -> List[Dict]:
        """准备交易数据"""
        try:
            trades = []
            
            for selection in selections:
                # 严格按照future_return_1d逻辑：T日选股，T+1开盘买入，T+2开盘卖出
                selection_date = pd.to_datetime(selection.date)  # T日

                # 查找T日的股票数据（用于获取future_return_1d）
                selection_data = stock_data[
                    (stock_data['stock_code'] == selection.stock_code) &
                    (stock_data['date'] == selection_date)
                ]

                if len(selection_data) == 0:
                    continue

                selection_row = selection_data.iloc[0]

                # 检查是否有future_return_1d数据
                if 'future_return_1d' not in selection_row or pd.isna(selection_row['future_return_1d']):
                    continue

                # 查找T+1日数据（买入日）
                buy_date = selection_date + timedelta(days=1)
                buy_data = stock_data[
                    (stock_data['stock_code'] == selection.stock_code) &
                    (stock_data['date'] >= buy_date)
                ]

                if len(buy_data) == 0:
                    continue

                buy_row = buy_data.iloc[0]
                actual_buy_date = buy_row['date']

                # 查找T+2日数据（卖出日）
                sell_date = actual_buy_date + timedelta(days=1)
                sell_data = stock_data[
                    (stock_data['stock_code'] == selection.stock_code) &
                    (stock_data['date'] >= sell_date)
                ]

                if len(sell_data) == 0:
                    continue

                sell_row = sell_data.iloc[0]
                actual_sell_date = sell_row['date']

                # 创建交易记录（严格按照future_return_1d）
                trade = {
                    'stock_code': selection.stock_code,
                    'selection_date': selection_date,  # T日选股
                    'buy_date': actual_buy_date,       # T+1日买入
                    'sell_date': actual_sell_date,     # T+2日卖出
                    'buy_price': buy_row.get('open', 0),   # T+1开盘价买入
                    'sell_price': sell_row.get('open', 0), # T+2开盘价卖出
                    'future_return_1d': selection_row['future_return_1d'],  # 真实的future_return_1d
                    'composite_score': selection.composite_score,
                    'confidence': selection.confidence,
                    'expected_return': selection.expected_return,
                    'holding_days': (actual_sell_date - actual_buy_date).days
                }
                
                trades.append(trade)
            
            self.logger.info(f"准备交易数据完成: {len(trades)}笔交易")
            return trades
            
        except Exception as e:
            self.logger.error(f"准备交易数据失败: {e}")
            return []
    
    def _calculate_trade_results(self, trades: List[Dict], 
                               stock_data: pd.DataFrame) -> List[Dict]:
        """计算交易结果"""
        try:
            trade_results = []
            
            for trade in trades:
                buy_price = trade['buy_price']
                sell_price = trade['sell_price']
                
                if buy_price <= 0 or sell_price <= 0:
                    continue
                
                # 计算收益率（严格按照future_return_1d）
                if buy_price <= 0 or sell_price <= 0:
                    continue

                # 实际收益率（T+1开盘到T+2开盘）
                gross_return = (sell_price - buy_price) / buy_price

                # 交易成本
                total_cost = self.commission_rate * 2 + self.slippage_rate * 2
                net_return = gross_return - total_cost

                # 验证与future_return_1d的一致性
                future_return_1d = trade.get('future_return_1d', 0)
                return_diff = abs(gross_return - future_return_1d) if future_return_1d != 0 else 0
                
                result = {
                    **trade,
                    'gross_return': gross_return,
                    'net_return': net_return,
                    'future_return_1d': future_return_1d,
                    'return_diff': return_diff,  # 与future_return_1d的差异
                    'is_profitable': net_return > 0,
                    'profit_amount': net_return * 10000,  # 假设1万元本金
                }
                
                trade_results.append(result)
            
            self.logger.info(f"交易结果计算完成: {len(trade_results)}笔有效交易")
            return trade_results
            
        except Exception as e:
            self.logger.error(f"计算交易结果失败: {e}")
            return []
    
    def _calculate_backtest_metrics(self, trade_results: List[Dict]) -> BacktestResult:
        """计算回测指标"""
        try:
            if not trade_results:
                return self._create_empty_result()
            
            # 基础统计
            returns = [trade['net_return'] for trade in trade_results]
            profitable_trades = [trade for trade in trade_results if trade['is_profitable']]
            losing_trades = [trade for trade in trade_results if not trade['is_profitable']]
            
            # 收益指标
            total_return = np.sum(returns)
            avg_return = np.mean(returns)
            
            # 计算年化收益率
            if trade_results:
                start_date = min(trade['buy_date'] for trade in trade_results)
                end_date = max(trade['sell_date'] for trade in trade_results)
                days = (end_date - start_date).days
                years = max(days / 365.25, 1/365.25)  # 至少1天
                annual_return = (1 + total_return) ** (1/years) - 1
            else:
                annual_return = 0
            
            # 风险指标
            return_std = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = (avg_return - backtest_config.risk_free_rate/252) / return_std if return_std > 0 else 0

            # 年化夏普比率（正确计算）
            annualized_return = (1 + avg_return) ** 252 - 1
            annualized_volatility = return_std * np.sqrt(252)
            annualized_sharpe = (annualized_return - backtest_config.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            # 最大回撤
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            # 胜率和盈亏比
            win_rate = len(profitable_trades) / len(trade_results)
            
            if profitable_trades and losing_trades:
                avg_profit = np.mean([trade['net_return'] for trade in profitable_trades])
                avg_loss = np.mean([abs(trade['net_return']) for trade in losing_trades])
                profit_loss_ratio = avg_profit / avg_loss
            else:
                profit_loss_ratio = 0
            
            # 持仓天数
            avg_holding_days = np.mean([trade['holding_days'] for trade in trade_results])
            
            # 每日收益率（简化计算）
            daily_returns = returns  # 简化处理
            cumulative_returns_list = cumulative_returns.tolist() if len(cumulative_returns) > 0 else []
            
            return BacktestResult(
                total_return=total_return,
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                annualized_sharpe=annualized_sharpe,  # 添加年化夏普比率
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_loss_ratio=profit_loss_ratio,
                total_trades=len(trade_results),
                profitable_trades=len(profitable_trades),
                avg_holding_days=avg_holding_days,
                daily_returns=daily_returns,
                cumulative_returns=cumulative_returns_list,
                trade_details=trade_results
            )
            
        except Exception as e:
            self.logger.error(f"计算回测指标失败: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> BacktestResult:
        """创建空的回测结果"""
        return BacktestResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            annualized_sharpe=0.0,  # 添加年化夏普比率
            max_drawdown=0.0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            total_trades=0,
            profitable_trades=0,
            avg_holding_days=0.0,
            daily_returns=[],
            cumulative_returns=[],
            trade_details=[]
        )
    
    def analyze_performance_by_score(self, trade_results: List[Dict]) -> Dict:
        """按评分分析表现"""
        try:
            if not trade_results:
                return {}
            
            # 按评分分组
            score_groups = {
                'high_score': [t for t in trade_results if t['composite_score'] >= 80],
                'medium_score': [t for t in trade_results if 60 <= t['composite_score'] < 80],
                'low_score': [t for t in trade_results if t['composite_score'] < 60]
            }
            
            analysis = {}
            
            for group_name, trades in score_groups.items():
                if trades:
                    returns = [t['net_return'] for t in trades]
                    win_rate = sum(1 for t in trades if t['is_profitable']) / len(trades)
                    avg_return = np.mean(returns)
                    
                    analysis[group_name] = {
                        'count': len(trades),
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'total_return': sum(returns)
                    }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"按评分分析表现失败: {e}")
            return {}
    
    def generate_performance_report(self, backtest_result: BacktestResult) -> str:
        """生成表现报告"""
        try:
            report = []
            report.append("=" * 60)
            report.append("回测表现报告")
            report.append("=" * 60)
            
            # 收益指标
            report.append(f"总收益率: {backtest_result.total_return:.2%}")
            report.append(f"年化收益率: {backtest_result.annual_return:.2%}")
            report.append(f"夏普比率: {backtest_result.sharpe_ratio:.2f}")
            report.append(f"最大回撤: {backtest_result.max_drawdown:.2%}")
            
            # 交易指标
            report.append(f"总交易次数: {backtest_result.total_trades}")
            report.append(f"盈利交易: {backtest_result.profitable_trades}")
            report.append(f"胜率: {backtest_result.win_rate:.1%}")
            report.append(f"盈亏比: {backtest_result.profit_loss_ratio:.2f}")
            report.append(f"平均持仓天数: {backtest_result.avg_holding_days:.1f}")
            
            # 按评分分析
            if backtest_result.trade_details:
                score_analysis = self.analyze_performance_by_score(backtest_result.trade_details)
                if score_analysis:
                    report.append("\n按评分分组表现:")
                    for group, stats in score_analysis.items():
                        report.append(f"  {group}: 交易{stats['count']}次, "
                                    f"胜率{stats['win_rate']:.1%}, "
                                    f"平均收益{stats['avg_return']:.2%}")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"生成表现报告失败: {e}")
            return "报告生成失败"
