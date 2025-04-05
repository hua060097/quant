#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测分析器模块
用于分析回测结果、生成报告和图表
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import seaborn as sns
from matplotlib.font_manager import FontProperties
import warnings
import random
from scipy import stats

# 配置日志
logger = logging.getLogger("backtest_analyzer")

# 尝试使用中文字体
try:
    # 尝试加载中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    CHINESE_FONT_AVAILABLE = True
except:
    logger.warning("无法加载中文字体，图表将使用默认字体")
    CHINESE_FONT_AVAILABLE = False

# 忽略matplotlib警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class BacktestAnalyzer:
    """回测结果分析器
    
    分析回测结果，生成各种报告和图表，对比基准表现
    """
    
    def __init__(self, backtest_result_file):
        """初始化分析器
        
        参数:
            backtest_result_file: 回测结果文件路径(.pkl)
        """
        self.result_file = backtest_result_file
        self.backtest_engine = None
        self.benchmark_data = {}
        self.random_portfolios = []
        self.reports_dir = 'to/results/reports'
        self.charts_dir = 'to/results/charts'
        
        # 确保输出目录存在
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # 加载回测引擎
        self._load_backtest_engine()
    
    def _load_backtest_engine(self):
        """加载回测引擎数据"""
        try:
            with open(self.result_file, 'rb') as f:
                self.backtest_engine = pickle.load(f)
            logger.info(f"成功加载回测结果: {self.result_file}")
            
            # 提取关键数据
            self.nav_df = self._extract_nav_data()
            self.trade_df = self._extract_trade_data()
            self.position_df = self._extract_position_data()
            
            return True
        except Exception as e:
            logger.error(f"加载回测结果失败: {e}")
            return False
    
    def _extract_nav_data(self):
        """提取净值数据"""
        try:
            # 提取每日净值
            dates = []
            navs = []
            
            for date, value in sorted(self.backtest_engine.daily_net_values.items()):
                dates.append(date)
                navs.append(value)
            
            nav_df = pd.DataFrame({
                'date': dates,
                'nav': navs
            })
            
            # 计算收益率和回撤
            nav_df['return'] = nav_df['nav'].pct_change()
            nav_df['cumret'] = (1 + nav_df['return']).cumprod() - 1
            nav_df['peak'] = nav_df['nav'].cummax()
            nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak']
            
            return nav_df
        except Exception as e:
            logger.error(f"提取净值数据出错: {e}")
            # 返回空的DataFrame
            return pd.DataFrame(columns=['date', 'nav', 'return', 'cumret', 'peak', 'drawdown'])
    
    def _extract_trade_data(self):
        """提取交易数据"""
        try:
            trade_history = self.backtest_engine.trade_history
            return pd.DataFrame(trade_history)
        except Exception as e:
            logger.error(f"提取交易数据出错: {e}")
            return pd.DataFrame()
    
    def _extract_position_data(self):
        """提取持仓数据"""
        try:
            position_history = []
            
            for date, positions in self.backtest_engine.position_history.items():
                for stock_code, details in positions.items():
                    position_history.append({
                        'date': date,
                        'stock_code': stock_code,
                        'volume': details['volume'],
                        'cost': details['cost'],
                        'price': details['price'],
                        'value': details['value'],
                        'weight': details['value'] / self.backtest_engine.daily_net_values.get(date, 1)
                    })
            
            return pd.DataFrame(position_history)
        except Exception as e:
            logger.error(f"提取持仓数据出错: {e}")
            return pd.DataFrame()
    
    def calculate_random_portfolio(self, num_trials=5):
        """计算随机投资组合表现
        
        通过随机选择股票和交易日期，模拟随机投资组合的表现
        
        参数:
            num_trials: 模拟次数
        """
        logger.info(f"开始计算 {num_trials} 个随机投资组合表现")
        
        try:
            # 从交易数据中提取股票池
            if not hasattr(self, 'trade_df') or self.trade_df.empty:
                logger.error("无法访问交易数据，随机投资组合计算失败")
                return False
            
            stock_pool = set(self.trade_df['stock_code'].unique())
            
            if len(stock_pool) < 10:
                # 股票池太小，扩展股票池
                extended_pool = set([f"60{i:04d}" for i in range(100, 400)])
                extended_pool.update([f"00{i:04d}" for i in range(100, 400)])
                stock_pool.update(extended_pool)
            
            # 获取回测时间段
            if not hasattr(self, 'nav_df') or self.nav_df.empty:
                logger.error("无法访问净值数据，随机投资组合计算失败")
                return False
                
            date_range = self.nav_df['date'].tolist()
            
            # 生成随机投资组合
            random_portfolios = []
            
            for i in range(num_trials):
                # 随机选择股票和持仓时间
                num_stocks = random.randint(5, 15)
                selected_stocks = random.sample(list(stock_pool), min(num_stocks, len(stock_pool)))
                
                # 模拟投资组合表现
                portfolio_nav = pd.DataFrame({'date': date_range, 'nav': 1.0})
                
                # 对每只股票分配权重
                weights = np.random.random(len(selected_stocks))
                weights = weights / weights.sum()
                
                # 根据随机产生的涨跌幅计算投资组合净值
                for i, date in enumerate(date_range[1:], 1):
                    daily_return = np.random.normal(0.0005, 0.02)  # 均值约0.05%，标准差2%
                    portfolio_nav.loc[i, 'nav'] = portfolio_nav.loc[i-1, 'nav'] * (1 + daily_return)
                
                portfolio_nav['return'] = portfolio_nav['nav'].pct_change()
                portfolio_nav['cumret'] = (1 + portfolio_nav['return']).cumprod() - 1
                
                # 计算性能指标
                returns = portfolio_nav['return'].dropna().values
                annual_return = np.mean(returns) * 252
                annual_volatility = np.std(returns) * np.sqrt(252)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                
                portfolio_data = {
                    'nav_df': portfolio_nav,
                    'annual_return': annual_return,
                    'annual_volatility': annual_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': portfolio_nav['nav'].pct_change().cummin().min()
                }
                
                random_portfolios.append(portfolio_data)
            
            # 保存结果
            self.random_portfolios = random_portfolios
            
            logger.info(f"成功计算 {len(random_portfolios)} 个随机投资组合")
            return True
            
        except Exception as e:
            logger.error(f"计算随机投资组合时出错: {e}")
            return False
    
    def load_benchmark_data(self, benchmarks):
        """加载基准数据
        
        参数:
            benchmarks: 基准代码列表，如 ['000001.SH', '399001.SZ']
        """
        logger.info(f"开始加载基准数据: {benchmarks}")
        
        try:
            from xtquant.xtdata import get_market_data
            
            # 获取回测时间段
            if not hasattr(self, 'nav_df') or self.nav_df.empty:
                logger.error("无法获取回测时间段，加载基准数据失败")
                return False
                
            start_date = min(self.nav_df['date'])
            end_date = max(self.nav_df['date'])
            
            # 格式化日期
            if isinstance(start_date, str):
                start_date_fmt = start_date
            else:
                start_date_fmt = start_date.strftime('%Y%m%d')
                
            if isinstance(end_date, str):
                end_date_fmt = end_date
            else:
                end_date_fmt = end_date.strftime('%Y%m%d')
            
            # 加载基准数据
            benchmark_data = {}
            
            for benchmark in benchmarks:
                # 获取基准行情数据
                try:
                    market_data = get_market_data(
                        field_list=['close'], 
                        stock_list=[benchmark], 
                        period='1d',
                        start_time=start_date_fmt,
                        end_time=end_date_fmt,
                        dividend_type='none'
                    )
                    
                    if market_data and 'close' in market_data:
                        close_prices = market_data['close'][benchmark]
                        dates = market_data['time']
                        
                        # 转换为DataFrame
                        benchmark_df = pd.DataFrame({
                            'date': dates,
                            'close': close_prices
                        })
                        
                        # 计算收益率
                        benchmark_df['return'] = benchmark_df['close'].pct_change()
                        benchmark_df['nav'] = (1 + benchmark_df['return']).cumprod()
                        benchmark_df.loc[0, 'nav'] = 1.0  # 设置初始净值为1
                        
                        # 计算累计收益率
                        benchmark_df['cumret'] = benchmark_df['nav'] - 1
                        
                        # 计算回撤
                        benchmark_df['peak'] = benchmark_df['nav'].cummax()
                        benchmark_df['drawdown'] = (benchmark_df['nav'] - benchmark_df['peak']) / benchmark_df['peak']
                        
                        # 保存
                        benchmark_data[benchmark] = benchmark_df
                        logger.info(f"成功加载基准数据: {benchmark}")
                    else:
                        logger.error(f"获取基准 {benchmark} 数据失败")
                except Exception as e:
                    logger.error(f"加载基准 {benchmark} 时出错: {e}")
            
            self.benchmark_data = benchmark_data
            
            return len(benchmark_data) > 0
            
        except Exception as e:
            logger.error(f"加载基准数据过程中出错: {e}")
            return False
    
    def plot_performance_comparison(self):
        """绘制表现对比图"""
        logger.info("开始绘制策略表现对比图")
        
        try:
            plt.figure(figsize=(12, 8))
            
            # 绘制策略净值曲线
            if hasattr(self, 'nav_df') and not self.nav_df.empty:
                plt.plot(
                    self.nav_df['date'], 
                    self.nav_df['nav'], 
                    label='策略净值', 
                    linewidth=2.5
                )
            
            # 绘制基准净值曲线
            for name, data in self.benchmark_data.items():
                if not data.empty:
                    plt.plot(
                        data['date'], 
                        data['nav'], 
                        label=f'基准: {name}',
                        linewidth=1.5,
                        alpha=0.8
                    )
            
            # 绘制随机投资组合
            if self.random_portfolios:
                # 只绘制第一个随机组合作为参考
                rand_portfolio = self.random_portfolios[0]
                plt.plot(
                    rand_portfolio['nav_df']['date'],
                    rand_portfolio['nav_df']['nav'],
                    label='随机组合',
                    linestyle='--',
                    alpha=0.7
                )
            
            # 设置图表
            plt.title('策略净值与基准对比', fontsize=14)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('净值', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # 保存图表
            output_path = os.path.join(self.charts_dir, 'performance_comparison.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            logger.info(f"表现对比图已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"绘制表现对比图时出错: {e}")
            return None
    
    def plot_drawdown(self):
        """绘制回撤图"""
        logger.info("开始绘制回撤分析图")
        
        try:
            plt.figure(figsize=(12, 6))
            
            # 绘制策略回撤
            if hasattr(self, 'nav_df') and not self.nav_df.empty:
                plt.fill_between(
                    self.nav_df['date'],
                    self.nav_df['drawdown'] * 100,
                    0,
                    alpha=0.6,
                    color='red',
                    label='策略回撤'
                )
            
            # 绘制基准回撤
            if self.benchmark_data:
                # 获取第一个基准
                benchmark_name = list(self.benchmark_data.keys())[0]
                benchmark = self.benchmark_data[benchmark_name]
                
                plt.fill_between(
                    benchmark['date'],
                    benchmark['drawdown'] * 100,
                    0,
                    alpha=0.4,
                    color='blue',
                    label=f'{benchmark_name} 回撤'
                )
            
            # 设置图表
            plt.title('策略回撤分析', fontsize=14)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('回撤 (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='lower left')
            
            # 保存图表
            output_path = os.path.join(self.charts_dir, 'drawdown_analysis.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            logger.info(f"回撤分析图已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"绘制回撤分析图时出错: {e}")
            return None
    
    def plot_monthly_returns(self):
        """绘制月度收益热力图"""
        logger.info("开始绘制月度收益热力图")
        
        try:
            # 确保净值数据可用
            if not hasattr(self, 'nav_df') or self.nav_df.empty:
                logger.error("无法访问净值数据，月度收益热力图生成失败")
                return None
            
            # 转换日期格式
            df = self.nav_df.copy()
            if isinstance(df['date'].iloc[0], str):
                df['date'] = pd.to_datetime(df['date'])
            
            # 计算月度收益
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # 分组计算月度收益
            monthly_returns = df.groupby(['year', 'month'])['return'].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            
            # 透视表
            heatmap_data = monthly_returns.pivot(
                index='year', 
                columns='month', 
                values='return'
            )
            
            # 绘制热力图
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                heatmap_data * 100,  # 转为百分比
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                linewidths=1,
                cbar_kws={'label': '收益率 (%)'}
            )
            
            # 设置图表
            plt.title('月度收益热力图 (%)', fontsize=14)
            plt.xlabel('月份', fontsize=12)
            plt.ylabel('年份', fontsize=12)
            
            # 保存图表
            output_path = os.path.join(self.charts_dir, 'monthly_returns.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            logger.info(f"月度收益热力图已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"绘制月度收益热力图时出错: {e}")
            return None
    
    def plot_trade_analysis(self):
        """绘制交易分析图"""
        logger.info("开始绘制交易分析图")
        
        try:
            # 确保交易数据可用
            if not hasattr(self, 'trade_df') or self.trade_df.empty:
                logger.error("无法访问交易数据，交易分析图生成失败")
                return None
            
            # 创建画布
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 买入和卖出次数分布
            trade_counts = self.trade_df['action'].value_counts()
            axes[0, 0].bar(trade_counts.index, trade_counts.values, color=['green', 'red'])
            axes[0, 0].set_title('交易次数分布', fontsize=12)
            axes[0, 0].set_ylabel('次数', fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 每笔交易盈亏分布
            # 计算每笔交易的盈亏
            buy_trades = self.trade_df[self.trade_df['action'] == 'buy']
            sell_trades = self.trade_df[self.trade_df['action'] == 'sell']
            
            trade_profits = []
            
            for _, buy in buy_trades.iterrows():
                stock_code = buy['stock_code']
                buy_date = buy['date']
                buy_amount = buy['amount']
                
                # 寻找对应的卖出记录
                matching_sells = sell_trades[
                    (sell_trades['stock_code'] == stock_code) & 
                    (sell_trades['date'] > buy_date)
                ]
                
                if not matching_sells.empty:
                    sell = matching_sells.iloc[0]
                    sell_amount = sell['amount']
                    profit = sell_amount - buy_amount
                    trade_profits.append(profit)
            
            if trade_profits:
                axes[0, 1].hist(trade_profits, bins=20, color='skyblue', edgecolor='black')
                axes[0, 1].axvline(0, color='red', linestyle='--')
                axes[0, 1].set_title('交易盈亏分布', fontsize=12)
                axes[0, 1].set_xlabel('盈亏', fontsize=10)
                axes[0, 1].set_ylabel('频次', fontsize=10)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 交易频率随时间变化
            if 'date' in self.trade_df.columns:
                # 确保日期格式正确
                if isinstance(self.trade_df['date'].iloc[0], str):
                    self.trade_df['date'] = pd.to_datetime(self.trade_df['date'])
                
                # 按周统计交易次数
                self.trade_df['week'] = self.trade_df['date'].dt.to_period('W')
                weekly_trades = self.trade_df.groupby('week').size()
                
                # 绘制周交易频率
                axes[1, 0].plot(range(len(weekly_trades)), weekly_trades.values, 'o-', color='purple')
                axes[1, 0].set_title('每周交易频率', fontsize=12)
                axes[1, 0].set_xlabel('周数', fontsize=10)
                axes[1, 0].set_ylabel('交易次数', fontsize=10)
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 按行业分类的交易次数
            if 'sector' in self.trade_df.columns:
                sector_counts = self.trade_df['sector'].value_counts().head(10)
                axes[1, 1].barh(sector_counts.index, sector_counts.values, color='orange')
                axes[1, 1].set_title('行业交易分布 (前10)', fontsize=12)
                axes[1, 1].set_xlabel('交易次数', fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                # 如果没有行业信息，显示股票交易次数
                stock_counts = self.trade_df['stock_code'].value_counts().head(10)
                axes[1, 1].barh(stock_counts.index, stock_counts.values, color='orange')
                axes[1, 1].set_title('股票交易次数 (前10)', fontsize=12)
                axes[1, 1].set_xlabel('交易次数', fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
            
            # 保存图表
            plt.tight_layout()
            output_path = os.path.join(self.charts_dir, 'trade_analysis.png')
            plt.savefig(output_path, dpi=150)
            plt.close()
            
            logger.info(f"交易分析图已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"绘制交易分析图时出错: {e}")
            return None
    
    def generate_performance_report(self):
        """生成绩效报告"""
        logger.info("开始生成绩效报告")
        
        try:
            # 计算关键绩效指标
            if not hasattr(self, 'nav_df') or self.nav_df.empty:
                logger.error("无法访问净值数据，绩效报告生成失败")
                return None
            
            # 初始化报告数据
            report_data = {}
            
            # 收益率相关指标
            returns = self.nav_df['return'].dropna().values
            
            # 累计收益率
            if len(self.nav_df) > 0:
                total_return = self.nav_df['nav'].iloc[-1] / self.nav_df['nav'].iloc[0] - 1
                report_data['总收益率'] = f"{total_return:.2%}"
            else:
                report_data['总收益率'] = "N/A"
            
            # 年化收益率
            if len(returns) > 0:
                annual_return = np.mean(returns) * 252  # 假设252个交易日
                report_data['年化收益率'] = f"{annual_return:.2%}"
            else:
                report_data['年化收益率'] = "N/A"
            
            # 年化波动率
            if len(returns) > 0:
                annual_volatility = np.std(returns) * np.sqrt(252)
                report_data['年化波动率'] = f"{annual_volatility:.2%}"
            else:
                report_data['年化波动率'] = "N/A"
            
            # 夏普比率
            if len(returns) > 0 and annual_volatility > 0:
                risk_free_rate = 0.03  # 假设无风险利率为3%
                sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                report_data['夏普比率'] = f"{sharpe_ratio:.2f}"
            else:
                report_data['夏普比率'] = "N/A"
            
            # 最大回撤
            if 'drawdown' in self.nav_df.columns:
                max_drawdown = self.nav_df['drawdown'].min()
                report_data['最大回撤'] = f"{max_drawdown:.2%}"
            else:
                report_data['最大回撤'] = "N/A"
            
            # 交易次数
            if hasattr(self, 'trade_df') and not self.trade_df.empty:
                buy_count = len(self.trade_df[self.trade_df['action'] == 'buy'])
                sell_count = len(self.trade_df[self.trade_df['action'] == 'sell'])
                report_data['买入次数'] = str(buy_count)
                report_data['卖出次数'] = str(sell_count)
                report_data['总交易次数'] = str(buy_count + sell_count)
            else:
                report_data['买入次数'] = "N/A"
                report_data['卖出次数'] = "N/A"
                report_data['总交易次数'] = "N/A"
            
            # 平均持仓周期
            # TODO: 计算平均持仓周期
            
            # 胜率
            if hasattr(self, 'trade_df') and not self.trade_df.empty:
                # 计算每笔交易盈亏
                buy_trades = self.trade_df[self.trade_df['action'] == 'buy']
                sell_trades = self.trade_df[self.trade_df['action'] == 'sell']
                
                win_count = 0
                lose_count = 0
                
                for _, buy in buy_trades.iterrows():
                    stock_code = buy['stock_code']
                    buy_date = buy['date']
                    buy_amount = buy['amount']
                    
                    matching_sells = sell_trades[
                        (sell_trades['stock_code'] == stock_code) & 
                        (sell_trades['date'] > buy_date)
                    ]
                    
                    if not matching_sells.empty:
                        sell = matching_sells.iloc[0]
                        sell_amount = sell['amount']
                        
                        if sell_amount > buy_amount:
                            win_count += 1
                        else:
                            lose_count += 1
                
                total_trades = win_count + lose_count
                if total_trades > 0:
                    win_rate = win_count / total_trades
                    report_data['胜率'] = f"{win_rate:.2%}"
                else:
                    report_data['胜率'] = "N/A"
            else:
                report_data['胜率'] = "N/A"
            
            # 生成报告文件
            report_lines = [
                "# 量化交易回测绩效报告",
                f"## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 策略绩效指标",
                ""
            ]
            
            for key, value in report_data.items():
                report_lines.append(f"- **{key}**: {value}")
            
            report_lines.extend([
                "",
                "## 基准对比",
                ""
            ])
            
            # 添加基准对比数据
            for name, data in self.benchmark_data.items():
                if not data.empty:
                    benchmark_return = data['nav'].iloc[-1] / data['nav'].iloc[0] - 1
                    report_lines.append(f"- **{name} 总收益率**: {benchmark_return:.2%}")
            
            # 写入文件
            report_path = os.path.join(self.reports_dir, 'performance_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"绩效报告已保存至: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"生成绩效报告时出错: {e}")
            return None
    
    def run_analysis(self):
        """运行所有分析功能并生成报告"""
        # 并行执行图表生成任务
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.plot_performance_comparison),
                executor.submit(self.plot_drawdown),
                executor.submit(self.plot_monthly_returns),
                executor.submit(self.plot_trade_analysis)
            ]
            
            # 等待所有图表生成完成
            for future in futures:
                future.result()
        
        # 生成绩效报告
        self.generate_performance_report()
        
        logger.info("回测分析完成")
        return True 