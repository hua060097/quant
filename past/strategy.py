import pandas as pd
import numpy as np
import talib
import tushare as ts
from miniqmt import *
import backtrader as bt  # 导入 backtrader 并使用别名 bt
from backtrader import Cerebro, TimeFrame

class MyStrategy:
    def __init__(self):
        # 初始化参数
        self.risk_ratio = 0.02  # 单笔风险比例
        self.position_size = 0.1  # 初始仓位比例
        self.stop_loss = 0.03  # 止损比例
        self.take_profit = 0.06  # 止盈比例
        
    def get_data(self, start_date, end_date):
        # 获取1分钟级别数据
        data = get_minute_data('000001.SH', start_date, end_date)
        return data
    
    def calculate_indicators(self, data):
        # 计算技术指标
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['close'])
        data['boll_upper'], data['boll_mid'], data['boll_lower'] = talib.BBANDS(data['close'])
        return data
    
    def get_market_sentiment(self):
        # 使用miniqmt获取大盘数据
        index_data = get_minute_data('000001.SH', '2024-01-01', '2025-01-01')
        
        # 计算市场情绪指标
        index_data['ma5'] = index_data['close'].rolling(5).mean()
        index_data['ma20'] = index_data['close'].rolling(20).mean()
        index_data['vol_ratio'] = index_data['volume'] / index_data['volume'].rolling(20).mean()
        
        # 计算市场情绪分数
        sentiment_score = 0
        if index_data['close'].iloc[-1] > index_data['ma20'].iloc[-1]:
            sentiment_score += 0.3
        if index_data['ma5'].iloc[-1] > index_data['ma20'].iloc[-1]:
            sentiment_score += 0.2
        if index_data['vol_ratio'].iloc[-1] > 1.5:
            sentiment_score += 0.1
            
        return sentiment_score

    def risk_management(self, data):
        # 动态仓位管理
        sentiment = self.get_market_sentiment()
        volatility = data['close'].pct_change().std()
        
        # 根据市场情绪和波动率调整仓位
        if sentiment > 0.5 and volatility < 0.02:
            self.position_size = 0.3
        elif sentiment > 0.3 and volatility < 0.03:
            self.position_size = 0.2
        else:
            self.position_size = 0.1
            
        return self.position_size

    def generate_signals(self, data):
        # 生成交易信号
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['close']
        signals['signal'] = 0
        
        # 获取当前市场情绪
        sentiment = self.get_market_sentiment()
        
        # 买入信号条件
        buy_condition = (
            (data['rsi'] < 30) & 
            (data['macdhist'] > 0) &
            (data['close'] > data['boll_lower']) &
            (sentiment > 0.4)
        )
        
        # 卖出信号条件
        sell_condition = (
            (data['rsi'] > 70) | 
            (data['macdhist'] < 0) |
            (data['close'] < data['boll_mid']) |
            (sentiment < 0.3)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals

    def execute_trades(self, signals):
        # 执行交易
        positions = pd.DataFrame(index=signals.index)
        positions['signal'] = signals['signal']
        positions['price'] = signals['price']
        positions['position'] = 0

        # 计算交易成本
        positions['cost'] = positions['price'] * self.position_size * self.risk_ratio    
    
    def backtest(self, data):
        # 回测框架
        cerebro = Cerebro()
         # 添加数据

        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # 添加策略
        cerebro.addstrategy(self)
        
        # 设置初始资金
        cerebro.broker.set_cash(1000000)
        
        # 设置交易手续费
        cerebro.broker.setcommission(commission=0.0003)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 运行回测
        results = cerebro.run()
        
        # 获取回测结果
        strat = results[0]
        sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
        drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
        trade_analysis = strat.analyzers.trades.get_analysis()
        
        # 打印回测结果
        print(f"夏普比率: {sharpe:.2f}")
        print(f"最大回撤: {drawdown:.2%}")
        print(f"总交易次数: {trade_analysis.total.closed}")
        print(f"胜率: {trade_analysis.won.pnl.total / trade_analysis.total.pnl.total:.2%}")
        
        # 绘制回测结果
        cerebro.plot()

if __name__ == "__main__":
    strategy = MyStrategy()
    data = strategy.get_data('2024-01-01', '2025-01-01')
    data = strategy.calculate_indicators(data)
    signals = strategy.generate_signals(data)
    strategy.backtest(data)