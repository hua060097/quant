#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测引擎补丁
将市场模拟器集成到现有回测引擎中
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/backtest_engine_patch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest_engine_patch")

# 尝试导入市场模拟器
try:
    from market_simulator import (
        MarketSimulator, OrderType, OrderDirection, OrderStatus,
        SlippageModel, VolumeBasedSlippageModel, FixedSlippageModel, RandomSlippageModel,
        CommissionModel, ChinaStockCommissionModel,
        PriceLimit, Order, Position, Account
    )
    HAS_MARKET_SIMULATOR = True
    logger.info("成功导入市场模拟器")
except ImportError:
    logger.error("无法导入市场模拟器，请确保market_simulator.py在正确的路径下")
    HAS_MARKET_SIMULATOR = False

# 尝试导入回测引擎
try:
    from backtest_signals import BacktestEngine
    HAS_BACKTEST_ENGINE = True
    logger.info("成功导入回测引擎")
except ImportError:
    logger.error("无法导入回测引擎，请确保backtest_signals.py在正确的路径下")
    HAS_BACKTEST_ENGINE = False

def patch_backtest_engine():
    """
    为回测引擎打补丁，增加市场模拟器功能
    """
    if not (HAS_MARKET_SIMULATOR and HAS_BACKTEST_ENGINE):
        logger.error("缺少必要的模块，无法为回测引擎打补丁")
        return False
    
    # 保存原始方法的引用
    original_init = BacktestEngine.__init__
    original_buy = BacktestEngine.buy
    original_sell = BacktestEngine.sell
    original_run_backtest = BacktestEngine.run_backtest
    
    # 增强后的初始化方法
    def enhanced_init(self, start_date, end_date, initial_capital=1000000, 
                     position_size=0.1, max_positions=10, 
                     use_enhanced_simulator=True, 
                     slippage_model=None, 
                     commission_model=None,
                     price_limit_pct=0.1,
                     t_plus_1=True):
        """
        增强的初始化方法，增加市场模拟器相关参数
        
        参数:
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            position_size: 仓位大小
            max_positions: 最大持仓数量
            use_enhanced_simulator: 是否使用增强的市场模拟器
            slippage_model: 滑点模型
            commission_model: 交易成本模型
            price_limit_pct: 涨跌停限制百分比
            t_plus_1: 是否启用T+1交易规则
        """
        # 调用原始初始化方法
        original_init(self, start_date, end_date, initial_capital, position_size, max_positions)
        
        # 增加市场模拟器相关属性
        self.use_enhanced_simulator = use_enhanced_simulator and HAS_MARKET_SIMULATOR
        
        if self.use_enhanced_simulator:
            # 创建市场模拟器
            self.market_simulator = MarketSimulator(
                slippage_model=slippage_model or VolumeBasedSlippageModel(),
                commission_model=commission_model or ChinaStockCommissionModel(),
                price_limit=PriceLimit(limit_pct=price_limit_pct),
                t_plus_1=t_plus_1
            )
            
            # 创建账户
            self.account = Account(initial_capital=initial_capital)
            
            logger.info(f"已启用增强的市场模拟器: {self.market_simulator}")
        else:
            self.market_simulator = None
            self.account = None
            logger.info("使用原始回测机制")
    
    # 增强后的买入方法
    def enhanced_buy(self, date, stock_code, price, reason=''):
        """
        增强的买入方法，使用市场模拟器处理交易
        
        参数:
            date: 交易日期
            stock_code: 股票代码
            price: 交易价格
            reason: 交易原因
        
        返回:
            是否成功买入
        """
        if not self.use_enhanced_simulator:
            # 使用原始买入方法
            return original_buy(self, date, stock_code, price, reason)
        
        # 检查交易日期
        if date not in self.trading_dates:
            logger.warning(f"日期 {date} 不在交易日历中，无法买入")
            return False
        
        # 获取当前持仓数量
        current_positions = len([p for p in self.account.positions.values() if p.volume > 0])
        
        # 检查是否超过最大持仓数量
        if current_positions >= self.max_positions:
            logger.info(f"已达到最大持仓数量 {self.max_positions}，无法买入 {stock_code}")
            return False
        
        # 获取股票当日数据
        market_data = self._get_stock_data(stock_code, date)
        if market_data is None or market_data.empty:
            logger.warning(f"无法获取股票 {stock_code} 在日期 {date} 的数据，无法买入")
            return False
        
        # 加载市场数据到模拟器
        self.market_simulator.load_market_data(date, [stock_code], {stock_code: market_data})
        
        # 计算买入数量
        available_cash = self.account.cash
        position_value = available_cash * self.position_size
        buy_price = market_data['close'].iloc[-1]  # 使用收盘价
        volume = int(position_value / buy_price / 100) * 100  # 买入的股数，取整百
        
        if volume <= 0:
            logger.warning(f"计算的买入数量为 {volume}，无法买入")
            return False
        
        # 创建买入订单
        order = self.market_simulator.place_order(
            self.account,
            stock_code,
            OrderDirection.BUY,
            volume,
            OrderType.MARKET,
            price=buy_price,
            reason=reason
        )
        
        # 检查订单状态
        if order.status == OrderStatus.FILLED:
            # 记录交易
            self.trade_history.append({
                'date': date,
                'stock_code': stock_code,
                'action': 'buy',
                'price': order.filled_price,
                'volume': order.filled_volume,
                'amount': order.filled_price * order.filled_volume,
                'commission': order.commission,
                'reason': reason
            })
            
            # 更新持仓信息
            self.positions[stock_code] = {
                'buy_date': date,
                'buy_price': order.filled_price,
                'volume': order.filled_volume,
                'reason': reason
            }
            
            logger.info(f"买入成功: {date}, {stock_code}, 价格: {order.filled_price:.2f}, "
                       f"数量: {order.filled_volume}, 金额: {order.filled_price * order.filled_volume:.2f}, "
                       f"手续费: {order.commission:.2f}")
            
            return True
        else:
            logger.warning(f"买入失败: {date}, {stock_code}, 原因: {order.rejected_reason}")
            return False
    
    # 增强后的卖出方法
    def enhanced_sell(self, date, stock_code, price, reason=''):
        """
        增强的卖出方法，使用市场模拟器处理交易
        
        参数:
            date: 交易日期
            stock_code: 股票代码
            price: 交易价格
            reason: 交易原因
        
        返回:
            是否成功卖出
        """
        if not self.use_enhanced_simulator:
            # 使用原始卖出方法
            return original_sell(self, date, stock_code, price, reason)
        
        # 检查交易日期
        if date not in self.trading_dates:
            logger.warning(f"日期 {date} 不在交易日历中，无法卖出")
            return False
        
        # 检查是否持有该股票
        if stock_code not in self.positions:
            logger.warning(f"未持有股票 {stock_code}，无法卖出")
            return False
        
        # 获取股票当日数据
        market_data = self._get_stock_data(stock_code, date)
        if market_data is None or market_data.empty:
            logger.warning(f"无法获取股票 {stock_code} 在日期 {date} 的数据，无法卖出")
            return False
        
        # 加载市场数据到模拟器
        self.market_simulator.load_market_data(date, [stock_code], {stock_code: market_data})
        
        # 获取持仓信息
        position = self.account.get_position(stock_code)
        if position.volume <= 0:
            logger.warning(f"持仓量为 {position.volume}，无法卖出")
            return False
        
        # 创建卖出订单
        sell_price = market_data['close'].iloc[-1]  # 使用收盘价
        order = self.market_simulator.place_order(
            self.account,
            stock_code,
            OrderDirection.SELL,
            position.volume,  # 全部卖出
            OrderType.MARKET,
            price=sell_price,
            reason=reason
        )
        
        # 检查订单状态
        if order.status == OrderStatus.FILLED:
            # 记录交易
            self.trade_history.append({
                'date': date,
                'stock_code': stock_code,
                'action': 'sell',
                'price': order.filled_price,
                'volume': order.filled_volume,
                'amount': order.filled_price * order.filled_volume,
                'commission': order.commission,
                'reason': reason
            })
            
            # 计算收益
            buy_info = self.positions[stock_code]
            buy_amount = buy_info['buy_price'] * buy_info['volume']
            sell_amount = order.filled_price * order.filled_volume
            profit = sell_amount - buy_amount - order.commission
            
            # 记录收益
            self.profits.append({
                'stock_code': stock_code,
                'buy_date': buy_info['buy_date'],
                'buy_price': buy_info['buy_price'],
                'sell_date': date,
                'sell_price': order.filled_price,
                'volume': order.filled_volume,
                'profit': profit,
                'profit_pct': profit / buy_amount,
                'hold_days': self._calculate_hold_days(buy_info['buy_date'], date)
            })
            
            # 删除持仓信息
            del self.positions[stock_code]
            
            logger.info(f"卖出成功: {date}, {stock_code}, 价格: {order.filled_price:.2f}, "
                       f"数量: {order.filled_volume}, 金额: {order.filled_price * order.filled_volume:.2f}, "
                       f"手续费: {order.commission:.2f}, 收益: {profit:.2f}")
            
            return True
        else:
            logger.warning(f"卖出失败: {date}, {stock_code}, 原因: {order.rejected_reason}")
            return False
    
    # 增强后的回测方法
    def enhanced_run_backtest(self):
        """
        增强的回测方法，使用市场模拟器处理每日更新
        
        返回:
            是否成功执行回测
        """
        if not self.use_enhanced_simulator:
            # 使用原始回测方法
            return original_run_backtest(self)
        
        logger.info(f"开始执行增强的回测: {self.start_date} 至 {self.end_date}")
        
        # 检查交易信号是否准备好
        if not self.trading_signals:
            logger.error("交易信号为空，无法执行回测")
            return False
        
        # 获取回测日期范围
        if self.start_date not in self.trading_dates or self.end_date not in self.trading_dates:
            logger.error("开始日期或结束日期不在交易日历中")
            return False
        
        start_idx = self.trading_dates.index(self.start_date)
        end_idx = self.trading_dates.index(self.end_date)
        
        # 每日净值记录
        self.daily_returns = []
        self.equity_curve = []
        
        # 计算初始净值
        initial_equity = self.account.total_value
        self.equity_curve.append({
            'date': self.trading_dates[start_idx],
            'equity': initial_equity,
            'return': 0.0,
            'positions': 0
        })
        
        # 记录回测开始时间
        start_time = datetime.now()
        logger.info(f"回测开始时间: {start_time}")
        
        # 遍历每个交易日
        for i in range(start_idx, end_idx + 1):
            current_date = self.trading_dates[i]
            
            logger.info(f"处理交易日: {current_date}")
            
            # 获取当日交易信号
            day_signals = self.trading_signals.get(current_date, {})
            
            # 先处理卖出信号
            for stock_code in list(self.positions.keys()):
                # 获取股票当日数据
                stock_data = self._get_stock_data(stock_code, current_date)
                if stock_data is None or stock_data.empty:
                    logger.warning(f"无法获取股票 {stock_code} 在日期 {current_date} 的数据，跳过处理")
                    continue
                
                # 检查是否有卖出信号
                sell_signal = day_signals.get(stock_code, {}).get('sell', False)
                if sell_signal:
                    # 执行卖出
                    self.sell(current_date, stock_code, stock_data['close'].iloc[-1], reason='卖出信号')
            
            # 再处理买入信号
            current_positions = len(self.positions)
            available_positions = self.max_positions - current_positions
            
            if available_positions > 0:
                # 获取买入信号
                buy_signals = [
                    (code, signal.get('score', 0), signal.get('reason', '买入信号'))
                    for code, signal in day_signals.items()
                    if signal.get('buy', False) and code not in self.positions
                ]
                
                # 按照评分排序
                buy_signals.sort(key=lambda x: x[1], reverse=True)
                
                # 买入信号
                for stock_code, score, reason in buy_signals[:available_positions]:
                    # 获取股票当日数据
                    stock_data = self._get_stock_data(stock_code, current_date)
                    if stock_data is None or stock_data.empty:
                        logger.warning(f"无法获取股票 {stock_code} 在日期 {current_date} 的数据，跳过买入")
                        continue
                    
                    # 执行买入
                    self.buy(current_date, stock_code, stock_data['close'].iloc[-1], reason=reason)
            
            # 更新所有股票当日收盘价
            for stock_code, position_info in self.positions.items():
                # 获取股票当日数据
                stock_data = self._get_stock_data(stock_code, current_date)
                if stock_data is None or stock_data.empty:
                    logger.warning(f"无法获取股票 {stock_code} 在日期 {current_date} 的数据，无法更新收盘价")
                    continue
                
                # 更新持仓的当前价格
                position = self.account.get_position(stock_code)
                position.update_price(stock_data['close'].iloc[-1])
            
            # 更新账户市值
            self.account.update_market_value()
            
            # 计算当日收益率
            daily_return, cumulative_return = self.account.calculate_returns(current_date)
            
            # 记录当日净值
            self.daily_returns.append(daily_return)
            self.equity_curve.append({
                'date': current_date,
                'equity': self.account.total_value,
                'return': daily_return,
                'positions': len(self.positions)
            })
            
            # 移动到下一个交易日
            self.market_simulator.next_day(self.account, self.trading_dates[i + 1] if i + 1 <= end_idx else None)
        
        # 计算回测结果
        self._calculate_performance_metrics()
        
        # 记录回测结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"回测结束时间: {end_time}, 耗时: {duration:.2f}秒")
        
        return True
    
    # 替换原始方法
    BacktestEngine.__init__ = enhanced_init
    BacktestEngine.buy = enhanced_buy
    BacktestEngine.sell = enhanced_sell
    BacktestEngine.run_backtest = enhanced_run_backtest
    
    # 添加属性以标识已打补丁
    BacktestEngine.PATCHED_WITH_MARKET_SIMULATOR = True
    
    logger.info("成功为回测引擎打补丁，增加了市场模拟器功能")
    return True

def create_enhanced_backtest_engine(
    start_date, end_date, initial_capital=1000000, 
    position_size=0.1, max_positions=10,
    slippage_model_type='volume', slippage_param=0.0001,
    commission_rate=0.00025, min_commission=5.0,
    price_limit_pct=0.1, t_plus_1=True
):
    """
    创建一个增强的回测引擎实例
    
    参数:
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        position_size: 仓位大小
        max_positions: 最大持仓数量
        slippage_model_type: 滑点模型类型，'fixed', 'volume' 或 'random'
        slippage_param: 滑点参数
        commission_rate: 佣金率
        min_commission: 最低佣金
        price_limit_pct: 涨跌停限制百分比
        t_plus_1: 是否启用T+1交易规则
        
    返回:
        增强的回测引擎实例
    """
    if not HAS_MARKET_SIMULATOR or not HAS_BACKTEST_ENGINE:
        logger.error("缺少必要的模块，无法创建增强的回测引擎")
        return None
    
    # 检查回测引擎是否已打补丁
    if not hasattr(BacktestEngine, 'PATCHED_WITH_MARKET_SIMULATOR'):
        patch_backtest_engine()
    
    # 创建滑点模型
    if slippage_model_type == 'fixed':
        slippage_model = FixedSlippageModel(slippage_pct=slippage_param)
    elif slippage_model_type == 'random':
        slippage_model = RandomSlippageModel(min_slippage=0, max_slippage=slippage_param)
    else:  # 默认使用基于成交量的滑点模型
        slippage_model = VolumeBasedSlippageModel(base_slippage=slippage_param)
    
    # 创建交易成本模型
    commission_model = ChinaStockCommissionModel(
        commission_rate=commission_rate,
        min_commission=min_commission
    )
    
    # 创建回测引擎
    engine = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        position_size=position_size,
        max_positions=max_positions,
        use_enhanced_simulator=True,
        slippage_model=slippage_model,
        commission_model=commission_model,
        price_limit_pct=price_limit_pct,
        t_plus_1=t_plus_1
    )
    
    return engine

if __name__ == "__main__":
    # 测试补丁
    result = patch_backtest_engine()
    print(f"补丁应用结果: {'成功' if result else '失败'}")
    
    # 创建测试实例
    if result:
        start_date = '20240101'
        end_date = '20240131'
        
        # 使用不同的滑点模型创建引擎
        engines = []
        
        # 固定滑点模型
        engines.append(create_enhanced_backtest_engine(
            start_date=start_date,
            end_date=end_date,
            slippage_model_type='fixed',
            slippage_param=0.0001
        ))
        
        # 基于成交量的滑点模型
        engines.append(create_enhanced_backtest_engine(
            start_date=start_date,
            end_date=end_date,
            slippage_model_type='volume',
            slippage_param=0.0001
        ))
        
        # 随机滑点模型
        engines.append(create_enhanced_backtest_engine(
            start_date=start_date,
            end_date=end_date,
            slippage_model_type='random',
            slippage_param=0.002
        ))
        
        print(f"创建了 {len(engines)} 个增强的回测引擎实例") 