#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场模拟器模块
用于模拟真实市场交易环境，包括滑点、交易成本、涨跌停限制等
"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Union, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta
import random

# 配置日志
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/market_simulator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("market_simulator")

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = 1  # 市价单
    LIMIT = 2   # 限价单
    STOP = 3    # 止损单

class OrderDirection(Enum):
    """订单方向枚举"""
    BUY = 1     # 买入
    SELL = 2    # 卖出

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = 1     # 待执行
    FILLED = 2      # 已成交
    PARTIAL = 3     # 部分成交
    CANCELLED = 4   # 已取消
    REJECTED = 5    # 已拒绝

class SlippageModel:
    """滑点模型基类"""
    
    def __init__(self):
        self.name = "基础滑点模型"
    
    def calculate_slippage(self, price: float, volume: float, 
                          direction: OrderDirection, 
                          avg_volume: float = None) -> float:
        """计算滑点"""
        return 0.0
    
    def get_execution_price(self, price: float, volume: float, 
                           direction: OrderDirection, 
                           avg_volume: float = None) -> float:
        """获取考虑滑点后的执行价格"""
        slippage = self.calculate_slippage(price, volume, direction, avg_volume)
        if direction == OrderDirection.BUY:
            return price * (1 + slippage)
        else:
            return price * (1 - slippage)

class FixedSlippageModel(SlippageModel):
    """固定比例滑点模型"""
    
    def __init__(self, slippage_pct: float = 0.0001):
        """
        初始化固定比例滑点模型
        
        参数:
            slippage_pct: 滑点百分比，默认0.01%
        """
        super().__init__()
        self.name = "固定比例滑点模型"
        self.slippage_pct = slippage_pct
    
    def calculate_slippage(self, price: float, volume: float, 
                          direction: OrderDirection, 
                          avg_volume: float = None) -> float:
        """计算固定比例滑点"""
        return self.slippage_pct

class VolumeBasedSlippageModel(SlippageModel):
    """基于成交量的滑点模型"""
    
    def __init__(self, base_slippage: float = 0.0001, volume_impact: float = 2.0):
        """
        初始化基于成交量的滑点模型
        
        参数:
            base_slippage: 基础滑点百分比，默认0.01%
            volume_impact: 成交量影响因子，默认2.0
        """
        super().__init__()
        self.name = "基于成交量的滑点模型"
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
    
    def calculate_slippage(self, price: float, volume: float, 
                          direction: OrderDirection, 
                          avg_volume: float = None) -> float:
        """
        计算基于成交量的滑点
        
        滑点 = 基础滑点 * (1 + 成交量影响因子 * 订单成交量/平均成交量)
        """
        if avg_volume is None or avg_volume == 0:
            return self.base_slippage
        
        volume_ratio = min(volume / avg_volume, 1.0)  # 限制比例不超过1
        return self.base_slippage * (1 + self.volume_impact * volume_ratio)

class RandomSlippageModel(SlippageModel):
    """随机滑点模型"""
    
    def __init__(self, min_slippage: float = 0.0, max_slippage: float = 0.002):
        """
        初始化随机滑点模型
        
        参数:
            min_slippage: 最小滑点百分比，默认0%
            max_slippage: 最大滑点百分比，默认0.2%
        """
        super().__init__()
        self.name = "随机滑点模型"
        self.min_slippage = min_slippage
        self.max_slippage = max_slippage
    
    def calculate_slippage(self, price: float, volume: float, 
                          direction: OrderDirection, 
                          avg_volume: float = None) -> float:
        """计算随机滑点"""
        return random.uniform(self.min_slippage, self.max_slippage)

class CommissionModel:
    """交易成本模型基类"""
    
    def __init__(self):
        self.name = "基础交易成本模型"
    
    def calculate_commission(self, price: float, volume: float, 
                            direction: OrderDirection) -> float:
        """计算交易成本"""
        return 0.0

class ChinaStockCommissionModel(CommissionModel):
    """中国A股交易成本模型"""
    
    def __init__(self, 
                commission_rate: float = 0.00025,      # 佣金率（双边）
                min_commission: float = 5.0,          # 最低佣金
                stamp_tax_rate: float = 0.001,        # 印花税率（仅卖出）
                transfer_fee_rate: float = 0.00002):  # 过户费率（沪市）
        """
        初始化中国A股交易成本模型
        
        参数:
            commission_rate: 佣金率，默认万分之2.5
            min_commission: 最低佣金，默认5元
            stamp_tax_rate: 印花税率，默认千分之1（仅卖出）
            transfer_fee_rate: 过户费率，默认十万分之2（沪市）
        """
        super().__init__()
        self.name = "中国A股交易成本模型"
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
    
    def calculate_commission(self, price: float, volume: float, 
                            direction: OrderDirection) -> float:
        """
        计算A股交易成本
        
        参数:
            price: 成交价格
            volume: 成交数量（股）
            direction: 交易方向
            
        返回:
            交易成本总额
        """
        # 计算总交易金额
        total_amount = price * volume
        
        # 计算佣金（双边）
        commission = max(total_amount * self.commission_rate, self.min_commission)
        
        # 计算印花税（仅卖出）
        stamp_tax = total_amount * self.stamp_tax_rate if direction == OrderDirection.SELL else 0
        
        # 计算过户费（仅沪市）
        # 这里简化处理，实际应根据股票代码判断是否沪市
        transfer_fee = total_amount * self.transfer_fee_rate
        
        # 计算总成本
        total_cost = commission + stamp_tax + transfer_fee
        
        return total_cost

class PriceLimit:
    """涨跌停限制"""
    
    def __init__(self, limit_pct: float = 0.1):
        """
        初始化涨跌停限制
        
        参数:
            limit_pct: 涨跌停限制百分比，默认10%
        """
        self.limit_pct = limit_pct
    
    def calculate_limits(self, prev_close: float) -> Tuple[float, float]:
        """
        计算涨跌停价格
        
        参数:
            prev_close: 昨日收盘价
            
        返回:
            (upper_limit, lower_limit): 涨停价和跌停价
        """
        # 四舍五入到分
        upper_limit = round(prev_close * (1 + self.limit_pct), 2)
        lower_limit = round(prev_close * (1 - self.limit_pct), 2)
        
        return upper_limit, lower_limit
    
    def is_limit_up(self, price: float, prev_close: float) -> bool:
        """判断是否涨停"""
        upper_limit, _ = self.calculate_limits(prev_close)
        return abs(price - upper_limit) < 0.005  # 考虑精度误差
    
    def is_limit_down(self, price: float, prev_close: float) -> bool:
        """判断是否跌停"""
        _, lower_limit = self.calculate_limits(prev_close)
        return abs(price - lower_limit) < 0.005  # 考虑精度误差

class Order:
    """订单类"""
    
    def __init__(self, 
                order_id: str,
                stock_code: str, 
                direction: OrderDirection,
                order_type: OrderType,
                price: float, 
                volume: int,
                timestamp: datetime,
                reason: str = ""):
        """
        初始化订单
        
        参数:
            order_id: 订单ID
            stock_code: 股票代码
            direction: 交易方向
            order_type: 订单类型
            price: 价格
            volume: 数量
            timestamp: 时间戳
            reason: 下单原因
        """
        self.order_id = order_id
        self.stock_code = stock_code
        self.direction = direction
        self.order_type = order_type
        self.price = price
        self.volume = volume
        self.timestamp = timestamp
        self.reason = reason
        self.status = OrderStatus.PENDING
        self.filled_volume = 0
        self.filled_price = 0.0
        self.filled_timestamp = None
        self.commission = 0.0
        self.rejected_reason = ""
    
    def fill(self, filled_price: float, filled_volume: int, 
             commission: float, timestamp: datetime):
        """成交"""
        self.filled_price = filled_price
        self.filled_volume = filled_volume
        self.commission = commission
        self.filled_timestamp = timestamp
        
        if filled_volume >= self.volume:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self):
        """取消"""
        self.status = OrderStatus.CANCELLED
    
    def reject(self, reason: str):
        """拒绝"""
        self.status = OrderStatus.REJECTED
        self.rejected_reason = reason
    
    def __str__(self):
        return (f"Order(id={self.order_id}, code={self.stock_code}, "
                f"direction={self.direction.name}, type={self.order_type.name}, "
                f"price={self.price}, volume={self.volume}, status={self.status.name})")

class Position:
    """持仓类"""
    
    def __init__(self, stock_code: str):
        """
        初始化持仓
        
        参数:
            stock_code: 股票代码
        """
        self.stock_code = stock_code
        self.volume = 0              # 持仓数量
        self.available_volume = 0    # 可用数量（考虑T+1）
        self.cost = 0.0              # 持仓成本
        self.market_value = 0.0      # 市值
        self.current_price = 0.0     # 当前价格
        self.profit_loss = 0.0       # 盈亏
        self.unrealized_pnl = 0.0    # 未实现盈亏
        self.realized_pnl = 0.0      # 已实现盈亏
        self.buy_orders = []         # 买入订单
        self.sell_orders = []        # 卖出订单
        self.today_bought = 0        # 今日买入（用于T+1）
    
    def update_price(self, price: float):
        """更新价格"""
        self.current_price = price
        self.market_value = self.volume * price
        self.unrealized_pnl = self.market_value - self.cost if self.volume > 0 else 0.0
    
    def buy(self, volume: int, price: float, commission: float):
        """买入"""
        old_cost = self.cost
        
        # 更新持仓
        self.volume += volume
        self.cost += price * volume + commission
        
        # T+1规则，今日买入不可卖出
        self.today_bought += volume
        
        # 实际可用数量不变，因为今日买入的部分要到下一交易日才能卖出
    
    def sell(self, volume: int, price: float, commission: float):
        """卖出"""
        if volume > self.available_volume:
            raise ValueError(f"可用数量不足: 要求{volume}，实际可用{self.available_volume}")
        
        # 计算已实现盈亏
        avg_cost = self.cost / self.volume if self.volume > 0 else 0
        realized_pnl = (price - avg_cost) * volume - commission
        
        # 更新持仓
        self.realized_pnl += realized_pnl
        self.available_volume -= volume
        self.volume -= volume
        
        # 更新持仓成本（按比例减少）
        if self.volume > 0:
            self.cost = self.cost * (1 - volume / (self.volume + volume))
        else:
            self.cost = 0
    
    def next_day(self):
        """处理下一交易日"""
        # T+1规则，昨日买入的今日可卖出
        self.available_volume += self.today_bought
        self.today_bought = 0
    
    def __str__(self):
        return (f"Position(code={self.stock_code}, volume={self.volume}, "
                f"available={self.available_volume}, cost={self.cost:.2f}, "
                f"current={self.current_price:.2f}, pnl={self.unrealized_pnl:.2f})")

class Account:
    """账户类"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        初始化账户
        
        参数:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital  # 现金
        self.market_value = 0.0      # 持仓市值
        self.total_value = initial_capital  # 总资产
        self.positions = {}          # 持仓 {stock_code: Position}
        self.orders = []             # 订单历史
        self.trade_history = []      # 交易历史
        self.daily_returns = []      # 每日收益率
        self.cumulative_returns = [] # 累计收益率
        self.commission_paid = 0.0   # 已支付手续费
    
    def get_position(self, stock_code: str) -> Position:
        """获取持仓，如果不存在则创建"""
        if stock_code not in self.positions:
            self.positions[stock_code] = Position(stock_code)
        return self.positions[stock_code]
    
    def update_market_value(self):
        """更新持仓市值和总资产"""
        self.market_value = sum(position.market_value for position in self.positions.values())
        self.total_value = self.cash + self.market_value
    
    def place_order(self, order: Order):
        """下单"""
        self.orders.append(order)
    
    def handle_filled_order(self, order: Order):
        """处理已成交订单"""
        position = self.get_position(order.stock_code)
        
        # 计算交易金额和佣金
        trade_amount = order.filled_price * order.filled_volume
        commission = order.commission
        
        # 根据交易方向更新持仓和现金
        if order.direction == OrderDirection.BUY:
            position.buy(order.filled_volume, order.filled_price, commission)
            self.cash -= (trade_amount + commission)
        else:  # SELL
            position.sell(order.filled_volume, order.filled_price, commission)
            self.cash += (trade_amount - commission)
        
        # 更新手续费统计
        self.commission_paid += commission
        
        # 记录交易历史
        self.trade_history.append({
            'date': order.filled_timestamp,
            'stock_code': order.stock_code,
            'action': 'buy' if order.direction == OrderDirection.BUY else 'sell',
            'price': order.filled_price,
            'volume': order.filled_volume,
            'amount': trade_amount,
            'commission': commission,
            'reason': order.reason
        })
        
        # 更新市值和总资产
        self.update_market_value()
    
    def next_day(self):
        """处理下一交易日"""
        # 更新所有持仓的T+1状态
        for position in self.positions.values():
            position.next_day()
    
    def calculate_returns(self, current_date: datetime):
        """计算收益率"""
        if not self.daily_returns:
            prev_value = self.initial_capital
        else:
            prev_value = self.total_value / (1 + self.daily_returns[-1])
        
        daily_return = (self.total_value / prev_value) - 1 if prev_value > 0 else 0
        cumulative_return = (self.total_value / self.initial_capital) - 1
        
        self.daily_returns.append(daily_return)
        self.cumulative_returns.append(cumulative_return)
        
        return daily_return, cumulative_return
    
    def __str__(self):
        return (f"Account(cash={self.cash:.2f}, market_value={self.market_value:.2f}, "
                f"total_value={self.total_value:.2f}, positions={len(self.positions)})")

class MarketSimulator:
    """市场模拟器"""
    
    def __init__(self, 
                slippage_model: SlippageModel = None,
                commission_model: CommissionModel = None,
                price_limit: PriceLimit = None,
                t_plus_1: bool = True):
        """
        初始化市场模拟器
        
        参数:
            slippage_model: 滑点模型
            commission_model: 交易成本模型
            price_limit: 涨跌停限制
            t_plus_1: 是否启用T+1交易规则
        """
        self.slippage_model = slippage_model or VolumeBasedSlippageModel()
        self.commission_model = commission_model or ChinaStockCommissionModel()
        self.price_limit = price_limit or PriceLimit()
        self.t_plus_1 = t_plus_1
        self.market_data = {}  # 市场数据缓存
        self.current_date = None
        
        logger.info(f"初始化市场模拟器: 滑点模型={self.slippage_model.name}, "
                    f"交易成本模型={self.commission_model.name}, "
                    f"T+1规则={'启用' if t_plus_1 else '禁用'}")
    
    def load_market_data(self, date: str, stocks: List[str], 
                         data: Dict[str, pd.DataFrame]):
        """
        加载市场数据
        
        参数:
            date: 日期
            stocks: 股票列表
            data: 市场数据 {stock_code: DataFrame}
        """
        self.current_date = date
        self.market_data = data
    
    def check_order_executable(self, order: Order, 
                              account: Account) -> Tuple[bool, str]:
        """
        检查订单是否可执行
        
        参数:
            order: 订单
            account: 账户
            
        返回:
            (可执行, 原因)
        """
        stock_code = order.stock_code
        
        # 检查市场数据是否可用
        if stock_code not in self.market_data:
            return False, f"无市场数据: {stock_code}"
        
        market_data = self.market_data[stock_code]
        if market_data.empty:
            return False, f"市场数据为空: {stock_code}"
        
        # 获取当前价格和成交量
        current_price = market_data['close'].iloc[-1]
        current_volume = market_data['volume'].iloc[-1]
        
        # 检查涨跌停
        if 'prev_close' in market_data.columns:
            prev_close = market_data['prev_close'].iloc[-1]
        else:
            prev_close = market_data['close'].iloc[-2] if len(market_data) > 1 else current_price
        
        upper_limit, lower_limit = self.price_limit.calculate_limits(prev_close)
        
        if order.direction == OrderDirection.BUY and current_price >= upper_limit:
            return False, f"涨停无法买入: {stock_code}"
        
        if order.direction == OrderDirection.SELL and current_price <= lower_limit:
            return False, f"跌停无法卖出: {stock_code}"
        
        # 检查T+1规则
        if self.t_plus_1 and order.direction == OrderDirection.SELL:
            position = account.get_position(stock_code)
            if order.volume > position.available_volume:
                return False, f"T+1限制，可用数量不足: 要求{order.volume}，实际可用{position.available_volume}"
        
        # 检查持仓和资金
        if order.direction == OrderDirection.SELL:
            position = account.get_position(stock_code)
            if order.volume > position.volume:
                return False, f"持仓不足: 要求{order.volume}，实际{position.volume}"
        else:  # BUY
            # 计算预估成交价格和成本
            execution_price = self.slippage_model.get_execution_price(
                current_price, order.volume, order.direction, current_volume)
            estimated_cost = execution_price * order.volume
            commission = self.commission_model.calculate_commission(
                execution_price, order.volume, order.direction)
            total_cost = estimated_cost + commission
            
            if total_cost > account.cash:
                return False, f"资金不足: 需要{total_cost:.2f}，实际{account.cash:.2f}"
        
        # 检查订单规模与流动性
        avg_volume = market_data['volume'].mean() if 'volume' in market_data.columns else current_volume
        if order.volume > avg_volume * 0.1:  # 订单量超过日均成交量的10%
            logger.warning(f"大单警告: {stock_code} 订单量{order.volume}股，"
                          f"日均成交量{avg_volume}股，比例{order.volume/avg_volume:.2%}")
            if order.volume > avg_volume * 0.3:  # 超过30%可能导致严重滑点
                return False, f"订单量过大，超过日均成交量的30%: {order.volume}>{avg_volume*0.3}"
        
        return True, "可执行"
    
    def execute_order(self, order: Order, account: Account) -> bool:
        """
        执行订单
        
        参数:
            order: 订单
            account: 账户
            
        返回:
            是否成功执行
        """
        stock_code = order.stock_code
        
        # 检查订单是否可执行
        executable, reason = self.check_order_executable(order, account)
        if not executable:
            order.reject(reason)
            logger.warning(f"订单拒绝: {order}, 原因: {reason}")
            return False
        
        # 获取市场数据
        market_data = self.market_data[stock_code]
        current_price = market_data['close'].iloc[-1]
        current_volume = market_data['volume'].iloc[-1]
        
        # 计算实际成交价格（考虑滑点）
        execution_price = self.slippage_model.get_execution_price(
            current_price, order.volume, order.direction, current_volume)
        
        # 计算交易成本
        commission = self.commission_model.calculate_commission(
            execution_price, order.volume, order.direction)
        
        # 完成订单
        order.fill(execution_price, order.volume, commission, datetime.now())
        
        # 更新账户
        account.handle_filled_order(order)
        
        logger.info(f"订单执行: {order}, 成交价: {execution_price:.4f}, 手续费: {commission:.2f}")
        
        return True
    
    def place_order(self, account: Account, stock_code: str, 
                   direction: OrderDirection, volume: int, 
                   order_type: OrderType = OrderType.MARKET, 
                   price: float = 0.0, reason: str = "") -> Order:
        """
        下单
        
        参数:
            account: 账户
            stock_code: 股票代码
            direction: 交易方向
            volume: 数量
            order_type: 订单类型，默认市价单
            price: 价格，市价单可为0
            reason: 下单原因
            
        返回:
            订单对象
        """
        # 生成订单ID
        order_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(account.orders)}"
        
        # 创建订单
        order = Order(
            order_id=order_id,
            stock_code=stock_code,
            direction=direction,
            order_type=order_type,
            price=price,
            volume=volume,
            timestamp=datetime.now(),
            reason=reason
        )
        
        # 下单
        account.place_order(order)
        
        # 执行订单
        self.execute_order(order, account)
        
        return order
    
    def next_day(self, account: Account, date: str = None):
        """
        处理下一交易日
        
        参数:
            account: 账户
            date: 新的交易日
        """
        # 更新当前日期
        if date:
            self.current_date = date
        
        # 处理T+1规则
        account.next_day()
        
        # 清除市场数据缓存
        self.market_data = {}
        
        logger.info(f"进入下一交易日: {self.current_date}")
    
    def __str__(self):
        return (f"MarketSimulator(date={self.current_date}, "
                f"滑点模型={self.slippage_model.name}, "
                f"交易成本模型={self.commission_model.name}, "
                f"T+1规则={'启用' if self.t_plus_1 else '禁用'})") 