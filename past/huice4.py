from datetime import datetime, timedelta
import xtquant.xtdata as xtdata
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor
import os
import random
from scipy import stats
from collections import defaultdict
import backtrader as bt
from joblib import Parallel, delayed, dump, load, Memory
import multiprocessing
from functools import partial
import tempfile
import gc
import psutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 创建临时目录用于缓存
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'quant_strategy_cache')
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"使用临时目录进行缓存: {TEMP_DIR}")

# 创建内存缓存对象
memory = Memory(TEMP_DIR, verbose=0)

# 监控内存使用情况的函数
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"当前内存使用: {mem_info.rss / (1024 * 1024):.2f} MB")

def calculate_factors(data):
    """计算技术因子（独立函数）"""
    factors = {}
    
    try:
        # 确保数据是正确的格式
        if 'close' not in data:
            logger.warning("数据中没有close列")
            return pd.DataFrame()
            
        # 获取close列数据
        stock_code = data['close'].index[0]  # 获取股票代码
        
        # 提取数据
        close = data['close'].loc[stock_code]
        volume = data['volume'].loc[stock_code] if 'volume' in data else None
        
        # 检查数据有效性
        if close.empty:
            logger.warning(f"股票 {stock_code} 的收盘价数据为空")
            return pd.DataFrame()
            
        # 价格动量因子
        factors['momentum_1d'] = close.pct_change()
        factors['momentum_5d'] = close.pct_change(5)
        factors['momentum_20d'] = close.pct_change(20)
        
        # 波动率因子
        factors['volatility'] = close.rolling(20).std()
        
        # 成交量因子
        if volume is not None and not volume.empty:
            # 避免除以零
            vol_mean = volume.rolling(5).mean()
            factors['volume_ratio'] = volume / vol_mean.replace(0, np.nan)
        
        # 趋势因子
        ma5 = talib.MA(close.values, timeperiod=5)
        ma20 = talib.MA(close.values, timeperiod=20)
        # 避免除以零
        ma20_nonzero = np.where(ma20 == 0, np.nan, ma20)
        factors['trend_strength'] = pd.Series((ma5 - ma20) / ma20_nonzero, index=close.index)
        
        # MACD因子
        macd, signal, hist = talib.MACD(close.values)
        factors['macd_hist'] = pd.Series(hist, index=close.index)
        
        # RSI因子
        factors['rsi'] = pd.Series(talib.RSI(close.values), index=close.index)
        
        # 布林带因子
        upper, middle, lower = talib.BBANDS(close.values)
        # 避免除以零
        band_width = upper - lower
        band_width_nonzero = np.where(band_width == 0, np.nan, band_width)
        factors['bb_position'] = pd.Series((close.values - lower) / band_width_nonzero, index=close.index)
        
        # 创建DataFrame
        df = pd.DataFrame(factors)
        
        # 替换无穷大值为NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
        
    except Exception as e:
        logger.warning(f"计算因子时出错: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

class StockData:
    """股票数据管理类"""
    def __init__(self):
        self.data_cache = {}
        
    def get_stock_data(self, stock_code, start_date, end_date, fields):
        """获取股票数据并缓存"""
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        try:
            data = xtdata.get_market_data(
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date,
                field_list=fields
            )
            
            # 检查数据有效性
            if data is None:
                logger.warning(f"获取股票 {stock_code} 数据返回None")
                return None
                
            # 检查是否包含所需字段
            if not all(field in data for field in fields):
                logger.warning(f"获取股票 {stock_code} 数据缺少所需字段")
                return None
                
            # 检查股票代码是否在索引中
            for field in fields:
                if stock_code not in data[field].index:
                    logger.warning(f"股票 {stock_code} 在 {field} 字段中不存在")
                    return None
            
            # 数据有效，缓存并返回
            self.data_cache[cache_key] = data
            return data
            
        except Exception as e:
            logger.warning(f"获取股票 {stock_code} 数据时出错: {str(e)}")
            return None

# 创建 backtrader 数据源
class XTDataFeed(bt.feeds.PandasData):
    """XT数据源适配器"""
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )
    
    def __init__(self, **kwargs):
        """初始化数据源，确保df属性可用"""
        super(XTDataFeed, self).__init__(**kwargs)
        # 保存DataFrame引用
        self.df = self.p.dataname
        self._idx = 0  # 初始化索引
        # 打印数据样本以便调试
        if not self.df.empty:
            logger.info(f"数据源 {kwargs.get('name', '未命名')} 初始化，数据范围: {self.df.index[0]} - {self.df.index[-1]}, 数据点数: {len(self.df)}")
            logger.info(f"数据样本:\n{self.df.head()}")
            logger.info(f"数据列: {list(self.df.columns)}")
            logger.info(f"数据类型: {self.df.dtypes}")
    
    def _load(self):
        """重写_load方法，确保日期索引正确处理"""
        if self._idx >= len(self.df):
            return False
            
        # 获取当前行
        row = self.df.iloc[self._idx]
        
        # 处理日期
        dt = self.df.index[self._idx]
        if isinstance(dt, str):
            try:
                dt = pd.to_datetime(dt)
            except:
                try:
                    dt = pd.to_datetime(dt, format='%Y%m%d')
                except:
                    dt = datetime.now()  # 最后的后备方案
        
        # 设置行数据
        self.lines.datetime[0] = bt.date2num(dt)
        
        # 获取所有行别名并正确设置数据
        for linename in self.getlinealiases():
            if linename == 'datetime':
                continue
                
            # 获取对应的列名
            colname = self.p.__dict__[linename]
            if colname is None:
                continue
                
            # 使用getattr获取line对象，而不是使用字符串索引
            line = getattr(self.lines, linename)
            
            # 设置数据 - 修正数据获取方式
            if colname in row.index:
                line[0] = row[colname]
            elif isinstance(colname, str) and colname in self.df.columns:
                line[0] = row[colname]
        
        self._idx += 1
        return True

class AdaptiveStopStrategy(bt.Strategy):
    """回测框架策略类"""
    params = (
        ('position_size', 0.2),
        ('stop_loss', 0.05),
        ('take_profit', 0.15),
        ('max_positions', 5),
        ('model', None),  # 添加模型参数
        ('scaler', None), # 添加标准化器参数
    )
    
    def __init__(self):
        self.orders = {}  # 跟踪订单
        self.positions_info = {}  # 跟踪持仓信息
        self.model = self.params.model  # 从参数获取模型
        self.scaler = self.params.scaler  # 从参数获取标准化器
        
        # 添加日志
        self.log("策略初始化")
        if self.model is not None and self.scaler is not None:
            self.log("模型已设置")
        
        # 确保在策略开始时就进行买入
        self.should_buy_on_start = True
        
        # 记录初始资金
        self.initial_cash = self.broker.getcash()
        
        # 记录每日持仓和资金状况
        self.daily_stats = {}
        
    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'[{dt.isoformat()}] {txt}')
        
    def start(self):
        """策略开始时调用"""
        self.initial_cash = self.broker.getcash()
        self.log(f"策略开始，初始资金: {self.initial_cash:.2f}")
        # 记录所有可用股票
        available_stocks = [d._name for d in self.datas]
        self.log(f"可用股票: {', '.join(available_stocks)}")
        # 记录初始资产
        self.log(f"初始总资产: {self.broker.getvalue():.2f}")
        
    def prenext(self):
        """数据不足时调用"""
        self.log(f"等待数据充足，当前持仓数: {len(self.positions)}, 可用资金: {self.broker.getcash():.2f}")
        
    def next(self):
        """每个交易日调用"""
        # 记录当前日期
        current_date = self.datas[0].datetime.date(0)
        
        # 计算实际持仓数量
        actual_positions_count = 0
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                actual_positions_count += 1
        
        # 记录当前状态
        self.log(f"当前持仓数: {actual_positions_count}, 可用资金: {self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}")
        
        # 打印每日持仓和资金摘要
        self.print_daily_summary()
        
        # 记录每个股票的价格和持仓情况
        holdings = []
        self.log("===== 当日所有股票价格 =====")
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                holdings.append(f"{data._name}({pos.size}股)")
                self.log(f"持有 {data._name}: {pos.size}股, 当前价格: {data.close[0]:.2f}, 持仓成本: {pos.price:.2f}, 持仓市值: {pos.size * data.close[0]:.2f}")
            else:
                self.log(f"股票 {data._name} 价格: {data.close[0]:.2f}, 可买入: {'是' if data._name not in self.positions_info and data._name not in self.orders else '否'}")
        self.log("===========================")
        
        # 记录每日统计
        self.daily_stats[current_date.isoformat()] = {
            'cash': self.broker.getcash(),
            'value': self.broker.getvalue(),
            'holdings': holdings
        }
        
        # 检查是否有未完成的订单
        pending_orders = len(self.orders)
        if pending_orders > 0:
            self.log(f"当前有 {pending_orders} 个未完成订单")
            for stock, order in self.orders.items():
                self.log(f"未完成订单: {stock}, 状态: {order.getstatusname()}")
        
        # 更新持仓信息和执行止盈止损
        self.manage_positions()
        
        # 选股并开仓
        if actual_positions_count < self.params.max_positions:
            # 确保在策略开始时就进行买入
            if self.should_buy_on_start:
                self.log("策略开始，主动买入股票")
                self.should_buy_on_start = False
                
                # 强制直接买入第一只股票
                self.log("尝试直接买入第一只股票...")
                data = self.datas[0]  # 直接使用第一只股票
                price = data.close[0]
                
                if price > 0:
                    # 使用固定数量
                    size = 100  # 买入100股
                    cost = size * price
                    
                    self.log(f"直接买入 {data._name}: 价格 {price:.2f}, 数量 {size}, 预计金额 {cost:.2f}, 当前可用资金 {self.broker.getcash():.2f}")
                    
                    # 记录买入前的资金
                    pre_buy_cash = self.broker.getcash()
                    
                    # 创建买入订单
                    order = self.buy(data=data, size=size, exectype=bt.Order.Market)
                    
                    if order:
                        self.log(f"直接买入订单创建成功: {data._name}, 订单ID: {order.ref}")
                        self.orders[data._name] = order
                        self.positions_info[data._name] = {
                            'max_return': 0,
                            'entry_price': price,
                        }
                        
                        # 检查资金是否变化
                        post_buy_cash = self.broker.getcash()
                        self.log(f"买入后资金变化: 买入前 {pre_buy_cash:.2f}, 买入后 {post_buy_cash:.2f}, 差额: {pre_buy_cash - post_buy_cash:.2f}")
                    else:
                        self.log(f"直接买入 {data._name} 订单创建失败")
                else:
                    self.log(f"股票 {data._name} 价格异常: {price}，无法买入")
                
                # 如果直接买入失败，尝试使用buy_all_stocks方法
                if actual_positions_count == 0 and len(self.orders) == 0:
                    self.log("直接买入失败，尝试使用buy_all_stocks方法")
                    self.buy_all_stocks()
            else:
                # 正常选股买入
                self.select_and_buy()
        else:
            self.log(f"当前实际持仓数 {actual_positions_count} 已达到最大持仓数 {self.params.max_positions}，不再买入")

    def print_daily_summary(self):
        """打印每日持仓和资金摘要"""
        current_date = self.datas[0].datetime.date(0)
        
        # 计算持仓总市值
        portfolio_value = 0.0
        positions_summary = []
        positions_count = 0
        
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                positions_count += 1
                market_value = pos.size * data.close[0]
                portfolio_value += market_value
                positions_summary.append({
                    'stock': data._name,
                    'size': pos.size,
                    'price': data.close[0],
                    'cost': pos.price,
                    'value': market_value,
                    'pnl': (data.close[0] - pos.price) * pos.size,
                    'return': (data.close[0] / pos.price - 1) * 100
                })
        
        # 打印摘要
        self.log(f"===== {current_date.isoformat()} 每日摘要 =====")
        self.log(f"可用资金: {self.broker.getcash():.2f}")
        self.log(f"持仓市值: {portfolio_value:.2f}")
        self.log(f"总资产: {self.broker.getvalue():.2f}")
        self.log(f"实际持仓数量: {positions_count}")
        
        if positions_summary:
            self.log("持仓明细:")
            for pos in positions_summary:
                self.log(f"  {pos['stock']}: {pos['size']}股, 现价: {pos['price']:.2f}, 成本: {pos['cost']:.2f}, 市值: {pos['value']:.2f}, 盈亏: {pos['pnl']:.2f}, 收益率: {pos['return']:.2f}%")
        else:
            self.log("当前无持仓")
            
        self.log("=============================")
        
    def stop(self):
        """策略结束时调用"""
        # 计算总收益
        final_value = self.broker.getvalue()
        total_return = (final_value / self.initial_cash - 1) * 100
        self.log(f"策略结束，最终资产: {final_value:.2f}, 总收益率: {total_return:.2f}%")
        
        # 记录每个股票的最终状态
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                self.log(f"最终持有 {data._name}: {pos.size}股, 最终价格: {data.close[0]:.2f}, 持仓市值: {pos.size * data.close[0]:.2f}")
            else:
                self.log(f"股票 {data._name} 最终价格: {data.close[0]:.2f}")
                
        # 打印每日资金变化摘要
        self.log("每日资金变化摘要:")
        for date, stats in sorted(self.daily_stats.items()):
            holdings_str = ", ".join(stats['holdings']) if stats['holdings'] else "无持仓"
            self.log(f"日期: {date}, 可用资金: {stats['cash']:.2f}, 总资产: {stats['value']:.2f}, 持仓: {holdings_str}")
            
    def buy_all_stocks(self):
        """在策略开始时买入所有可用股票"""
        # 计算实际持仓数量
        actual_positions_count = 0
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                actual_positions_count += 1
        
        # 计算每只股票可用资金
        available_stocks = [d for d in self.datas if self.getposition(d).size == 0 and d._name not in self.orders]
        self.log(f"可买入的股票数量: {len(available_stocks)}, 股票列表: {[d._name for d in available_stocks]}")
        
        if not available_stocks:
            self.log("没有可用股票可买入")
            return
            
        # 获取当前可用资金
        available_cash = self.broker.getcash()
        self.log(f"当前可用资金: {available_cash:.2f}")
        
        if available_cash <= 0:
            self.log(f"没有可用资金进行买入，当前可用资金: {available_cash:.2f}")
            return
            
        # 确保每只股票分配足够的资金，但不超过总资金的一定比例
        max_stocks_to_buy = min(len(available_stocks), self.params.max_positions - actual_positions_count)
        if max_stocks_to_buy <= 0:
            self.log(f"已达到最大持仓数 {self.params.max_positions}，不再买入")
            return
            
        cash_per_stock = available_cash / max_stocks_to_buy
        
        self.log(f"当前可用资金: {available_cash:.2f}, 计划买入 {max_stocks_to_buy} 只股票, 每只股票分配资金: {cash_per_stock:.2f}")
        
        # 记录买入前的资金状况
        pre_buy_cash = available_cash
        stocks_bought = 0
        
        # 检查数据是否有效
        for i, data in enumerate(available_stocks):
            self.log(f"检查股票 {data._name} (索引: {i}) 数据:")
            self.log(f"  - 数据长度: {len(data)}")
            self.log(f"  - 当前价格: {data.close[0] if len(data) > 0 else '无数据'}")
            self.log(f"  - 数据范围: {data.datetime.date(0) if len(data) > 0 else '无数据'} 至 {data.datetime.date(-1) if len(data) > 1 else '无数据'}")
            
            # 检查是否可以访问close数据
            try:
                price = data.close[0]
                self.log(f"  - 成功获取价格: {price}")
            except Exception as e:
                self.log(f"  - 获取价格失败: {str(e)}")
        
        # 强制买入第一只股票
        if available_stocks:
            try:
                data = available_stocks[0]
                price = data.close[0]
                self.log(f"尝试强制买入第一只股票 {data._name}, 价格: {price}")
                
                if price > 0:
                    # 使用固定数量
                    size = 100  # 买入100股
                    actual_cost = size * price
                    
                    if actual_cost <= available_cash:
                        self.log(f"强制买入 {data._name}: 价格 {price:.2f}, 数量 {size}, 预计金额 {actual_cost:.2f}")
                        
                        # 记录买入前的资金
                        pre_order_cash = self.broker.getcash()
                        
                        # 使用市价单
                        order = self.buy(data=data, size=size, exectype=bt.Order.Market)
                        
                        if order:
                            self.log(f"强制买入订单创建成功: {data._name}, 订单ID: {order.ref}, 状态: {order.getstatusname()}")
                            self.orders[data._name] = order
                            self.positions_info[data._name] = {
                                'max_return': 0,
                                'entry_price': price,
                            }
                            stocks_bought += 1
                            
                            # 检查资金是否变化
                            post_order_cash = self.broker.getcash()
                            self.log(f"订单创建后资金变化: 创建前 {pre_order_cash:.2f}, 创建后 {post_order_cash:.2f}, 差额: {pre_order_cash - post_order_cash:.2f}")
                            
                            # 检查持仓是否存在
                            pos = self.getposition(data)
                            if pos.size > 0:
                                self.log(f"持仓已创建: {data._name}, 数量: {pos.size}, 成本: {pos.price:.2f}")
                            else:
                                self.log(f"持仓未创建: {data._name}, 订单可能尚未执行")
                        else:
                            self.log(f"强制买入 {data._name} 订单创建失败")
                    else:
                        self.log(f"资金不足，无法强制买入: 需要 {actual_cost:.2f}, 可用 {available_cash:.2f}")
                else:
                    self.log(f"股票 {data._name} 价格异常: {price}，无法强制买入")
            except Exception as e:
                self.log(f"强制买入时发生错误: {str(e)}")
                import traceback
                self.log(f"错误详情: {traceback.format_exc()}")
        
        # 继续原有的买入逻辑
        for data in available_stocks[1:]:  # 跳过第一只股票，因为已经尝试买入
            # 如果已经达到最大持仓数，退出
            if stocks_bought + actual_positions_count >= self.params.max_positions:
                self.log(f"已达到最大持仓数 {self.params.max_positions}，停止买入")
                break
                
            # 检查是否已有未完成订单
            if data._name in self.orders:
                self.log(f"股票 {data._name} 已有未完成订单，跳过")
                continue
                
            # 获取最新的可用资金（可能已经被之前的订单减少）
            current_cash = self.broker.getcash()
            if current_cash <= 1000:  # 保留一些资金作为缓冲
                self.log(f"资金不足，无法继续买入，当前可用资金: {current_cash:.2f}")
                break
                
            # 计算可买入数量，确保不超过当前可用资金
            try:
                price = data.close[0]
                self.log(f"股票 {data._name} 当前价格: {price:.2f}")
                
                if price <= 0:
                    self.log(f"股票 {data._name} 价格异常: {price}，跳过")
                    continue
                    
                # 使用固定数量
                size = 100  # 买入100股
                actual_cost = size * price
                
                if actual_cost > current_cash:
                    self.log(f"资金不足，无法买入 {data._name}: 需要 {actual_cost:.2f}, 可用 {current_cash:.2f}")
                    continue
                
                self.log(f"股票 {data._name} 买入数量: {size} 股, 预计金额: {actual_cost:.2f}")
                
                # 记录买入前的资金
                pre_order_cash = self.broker.getcash()
                
                # 创建买入订单 - 使用市价单
                self.log(f"买入 {data._name}: 价格 {price:.2f}, 数量 {size}, 预计金额 {actual_cost:.2f}, 当前可用资金 {current_cash:.2f}")
                order = self.buy(data=data, size=size, exectype=bt.Order.Market)
                
                if order:
                    self.log(f"买入订单创建成功: {data._name}, 订单ID: {order.ref}, 订单状态: {order.getstatusname()}")
                    self.orders[data._name] = order
                    self.positions_info[data._name] = {
                        'max_return': 0,
                        'entry_price': price,
                    }
                    stocks_bought += 1
                    
                    # 检查资金是否变化
                    post_order_cash = self.broker.getcash()
                    self.log(f"订单创建后资金变化: 创建前 {pre_order_cash:.2f}, 创建后 {post_order_cash:.2f}, 差额: {pre_order_cash - post_order_cash:.2f}")
                else:
                    self.log(f"买入 {data._name} 订单创建失败，返回值为None")
            except Exception as e:
                self.log(f"买入 {data._name} 时发生错误: {str(e)}")
                import traceback
                self.log(f"错误详情: {traceback.format_exc()}")
        
        # 记录买入后的资金状况
        post_buy_cash = self.broker.getcash()
        self.log(f"买入操作完成，买入前资金: {pre_buy_cash:.2f}, 买入后资金: {post_buy_cash:.2f}, 使用资金: {pre_buy_cash - post_buy_cash:.2f}, 买入股票数: {stocks_bought}")
        
        # 检查订单状态
        if self.orders:
            self.log(f"当前未完成订单数: {len(self.orders)}")
            for stock_name, order in self.orders.items():
                self.log(f"订单详情 - 股票: {stock_name}, 订单ID: {order.ref}, 状态: {order.getstatusname()}")
        else:
            self.log("没有创建任何订单")
            
        # 检查持仓状态
        positions_count = 0
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                positions_count += 1
                self.log(f"持有 {data._name}: {pos.size}股, 成本: {pos.price:.2f}, 市值: {pos.size * data.close[0]:.2f}")
        
        if positions_count == 0:
            self.log("当前无持仓")
        else:
            self.log(f"当前实际持仓数量: {positions_count}")
        
    def notify_order(self, order):
        """订单状态变化通知"""
        # 获取订单相关的股票名称
        data_name = order.data._name
        
        # 记录订单详细信息
        self.log(f"订单状态变化 - 股票: {data_name}, 订单ID: {order.ref}, 状态: {order.getstatusname()}")
        self.log(f"订单详情 - 类型: {'买入' if order.isbuy() else '卖出'}, 数量: {order.created.size}, 创建价格: {order.created.price:.2f}")
        
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受，等待执行
            self.log(f"订单 {order.ref} 已{'提交' if order.status == order.Submitted else '接受'}，等待执行")
            return
            
        # 检查订单是否完成
        if order.status in [order.Completed]:
            self.log(f"订单 {order.ref} 已完成 - 执行价格: {order.executed.price:.2f}, 数量: {order.executed.size}, 成本: {order.executed.value:.2f}, 手续费: {order.executed.comm:.2f}")
            
            # 记录资金变化
            current_cash = self.broker.getcash()
            total_value = self.broker.getvalue()
            self.log(f"订单执行后资金状况 - 可用资金: {current_cash:.2f}, 总资产: {total_value:.2f}")
            
            # 检查持仓状态
            pos = self.getposition(order.data)
            if pos.size > 0:
                self.log(f"当前持仓 - 股票: {data_name}, 数量: {pos.size}, 成本: {pos.price:.2f}, 市值: {pos.size * order.data.close[0]:.2f}")
            else:
                self.log(f"无持仓 - 股票: {data_name}")
                
            # 如果是买入订单，更新持仓信息
            if order.isbuy():
                self.log(f"买入成功 - 股票: {data_name}, 价格: {order.executed.price:.2f}, 数量: {order.executed.size}")
                if data_name not in self.positions_info:
                    self.positions_info[data_name] = {
                        'max_return': 0,
                        'entry_price': order.executed.price,
                    }
            # 如果是卖出订单，清除持仓信息
            else:
                self.log(f"卖出成功 - 股票: {data_name}, 价格: {order.executed.price:.2f}, 数量: {order.executed.size}")
                if data_name in self.positions_info:
                    del self.positions_info[data_name]
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 订单被取消、保证金不足或被拒绝
            status_text = {
                order.Canceled: "取消",
                order.Margin: "保证金不足",
                order.Rejected: "拒绝"
            }.get(order.status, "未知状态")
            
            self.log(f"订单 {order.ref} 未完成 - 状态: {status_text}")
            
            # 记录拒绝原因
            if hasattr(order, 'reject_reason') and order.reject_reason:
                self.log(f"拒绝原因: {order.reject_reason}")
            
            # 检查是否有错误信息
            if hasattr(order, 'info') and order.info:
                self.log(f"订单信息: {order.info}")
                
            # 记录当前资金状况
            current_cash = self.broker.getcash()
            total_value = self.broker.getvalue()
            self.log(f"订单失败后资金状况 - 可用资金: {current_cash:.2f}, 总资产: {total_value:.2f}")
            
            # 如果是买入订单被拒绝，尝试记录可能的原因
            if order.isbuy():
                price = order.created.price if hasattr(order.created, 'price') else 0
                size = order.created.size if hasattr(order.created, 'size') else 0
                estimated_cost = price * size
                self.log(f"买入订单失败分析 - 预计成本: {estimated_cost:.2f}, 可用资金: {current_cash:.2f}, 差额: {current_cash - estimated_cost:.2f}")
                
                # 检查是否有足够的资金
                if estimated_cost > current_cash:
                    self.log(f"可能的拒绝原因: 资金不足 - 需要 {estimated_cost:.2f}, 可用 {current_cash:.2f}")
                
                # 检查价格是否有效
                if price <= 0:
                    self.log(f"可能的拒绝原因: 价格无效 - {price}")
                
                # 检查数量是否有效
                if size <= 0:
                    self.log(f"可能的拒绝原因: 数量无效 - {size}")
        
        # 清除已完成或失败的订单
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if data_name in self.orders and self.orders[data_name].ref == order.ref:
                self.log(f"从订单字典中移除订单 {order.ref}")
                del self.orders[data_name]
        
        # 计算并打印当前实际持仓数量
        positions_count = 0
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size > 0:
                positions_count += 1
        
        self.log(f"订单处理后 - 当前实际持仓数量: {positions_count}, 未完成订单数: {len(self.orders)}")
        
    def manage_positions(self):
        """管理现有持仓"""
        for data in self.datas:
            pos = self.getposition(data)
            if not pos.size:
                continue
                
            # 获取持仓信息
            if data._name not in self.positions_info:
                self.positions_info[data._name] = {
                    'max_return': 0,
                    'entry_price': pos.price,
                }
            
            pos_info = self.positions_info[data._name]
            current_price = data.close[0]
            entry_price = pos_info['entry_price']
            
            # 更新最大收益
            returns = (current_price - entry_price) / entry_price
            pos_info['max_return'] = max(pos_info['max_return'], returns)
            
            # 计算动态止损位
            stop_loss = self.calculate_adaptive_stop(pos_info, current_price)
            
            # 执行止损
            if returns < stop_loss:
                self.log(f"平仓 {data._name}: 当前收益率 {returns:.2%}, 止损位 {stop_loss:.2%}, 持仓数量: {pos.size}")
                self.close(data)
                if data._name in self.positions_info:
                    del self.positions_info[data._name]
                    
    def calculate_adaptive_stop(self, position_info, current_price):
        """计算自适应止损位"""
        entry_price = position_info['entry_price']
        max_return = position_info['max_return']
        
        # 根据最大收益设置不同的止损位
        if max_return > 0.15:  # 超过15%收益
            stop_loss = max_return * 0.7  # 保留70%收益
        elif max_return > 0.1:  # 10-15%收益
            stop_loss = max_return * 0.6  # 保留60%收益
        elif max_return > 0.05:  # 5-10%收益
            stop_loss = max_return * 0.5  # 保留50%收益
        else:
            stop_loss = -self.params.stop_loss
            
        return stop_loss
        
    def select_and_buy(self):
        """使用机器学习模型进行动态选股并买入"""
        # 获取当前可用资金
        available_cash = self.broker.getcash()
        if available_cash <= 0:
            self.log(f"没有可用资金进行买入，当前可用资金: {available_cash:.2f}")
            return
            
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0)
        self.log(f"当前日期: {current_date.isoformat()}, 开始选股...")
        
        # 获取可用股票（未持仓且无未完成订单）
        available_stocks = [d for d in self.datas if self.getposition(d).size == 0 and d._name not in self.orders]
        if not available_stocks:
            self.log("没有可用股票可买入")
            return
            
        self.log(f"可选股票数量: {len(available_stocks)}")
        
        # 记录买入前的资金状况
        pre_buy_cash = available_cash
        
        # 使用机器学习模型进行选股
        stock_scores = []
        
        for data in available_stocks:
            try:
                # 获取股票数据
                stock_code = data._name
                
                # 准备特征数据
                features = self.prepare_features(data)
                if features is None or len(features) == 0:
                    continue
                    
                # 使用模型预测
                if self.model is not None and self.scaler is not None:
                    # 标准化特征
                    scaled_features = self.scaler.transform([features])
                    
                    # 预测
                    prediction = self.model.predict_proba(scaled_features)[0]
                    
                    # 获取上涨概率
                    up_probability = prediction[1]  # 假设标签1表示上涨
                    
                    # 添加到候选列表
                    stock_scores.append((data, up_probability))
                    self.log(f"股票 {stock_code} 预测上涨概率: {up_probability:.4f}")
            except Exception as e:
                self.log(f"预测股票 {data._name} 时出错: {str(e)}")
                continue
        
        # 如果没有有效的预测结果，退出
        if not stock_scores:
            self.log("没有有效的预测结果，无法进行选股")
            return
            
        # 按预测概率排序，选择概率最高的股票
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        self.log(f"选股结果: {[(s[0]._name, s[1]) for s in stock_scores[:5]]}")
        
        # 选择预测概率最高的股票进行买入
        for data, score in stock_scores[:self.params.max_positions]:  # 最多买入max_positions只股票
            # 如果预测概率低于阈值，跳过
            if score < 0.6:  # 设置一个阈值，只买入预测上涨概率大于60%的股票
                self.log(f"股票 {data._name} 预测上涨概率 {score:.4f} 低于阈值 0.6，跳过")
                continue
                
            # 获取最新的可用资金（可能已经被之前的订单减少）
            current_cash = self.broker.getcash()
            if current_cash <= 0:
                self.log(f"资金已用完，无法继续买入，当前可用资金: {current_cash:.2f}")
                break
                
            # 计算仓位大小
            # 使用总资产的一定比例，但不超过当前可用资金
            value = self.broker.getvalue()
            position_value = value * self.params.position_size
            # 考虑手续费，略微减少可用资金
            adjusted_cash = current_cash * 0.999
            usable_cash = min(position_value, adjusted_cash)
            
            price = data.close[0]
            size = int(usable_cash / price)
            
            if size <= 0:
                self.log(f"股票 {data._name} 计算的买入数量为0，跳过")
                continue
                
            # 计算实际使用的资金
            actual_cost = size * price
            
            self.log(f"买入 {data._name}: 价格 {price:.2f}, 数量 {size}, 预计金额 {actual_cost:.2f}, 当前可用资金 {current_cash:.2f}, 预测上涨概率: {score:.4f}")
            order = self.buy(data=data, size=size)
            self.orders[data._name] = order
            self.positions_info[data._name] = {
                'max_return': 0,
                'entry_price': price,
            }
            
            # 只买入一只股票后退出，避免一次性使用过多资金
            break
            
        # 记录买入后的资金状况
        post_buy_cash = self.broker.getcash()
        if pre_buy_cash != post_buy_cash:
            self.log(f"买入操作完成，买入前资金: {pre_buy_cash:.2f}, 买入后资金: {post_buy_cash:.2f}, 使用资金: {pre_buy_cash - post_buy_cash:.2f}")

    def prepare_features(self, data):
        """为单个股票准备特征数据"""
        try:
            # 获取历史数据
            close = np.array([data.close[-i] for i in range(30, 0, -1)])
            high = np.array([data.high[-i] for i in range(30, 0, -1)])
            low = np.array([data.low[-i] for i in range(30, 0, -1)])
            volume = np.array([data.volume[-i] for i in range(30, 0, -1)])
            
            # 检查数据有效性
            if len(close) < 30 or np.isnan(close).any():
                self.log(f"股票 {data._name} 历史数据不足或包含NaN值")
                return None
                
            # 计算技术指标作为特征
            # 价格动量
            momentum_1d = close[-1] / close[-2] - 1
            momentum_5d = close[-1] / close[-6] - 1
            momentum_20d = close[-1] / close[-21] - 1
            
            # 波动率
            volatility = np.std(close[-20:]) / np.mean(close[-20:])
            
            # 成交量
            volume_ratio = volume[-1] / np.mean(volume[-5:])
            
            # 趋势
            ma5 = np.mean(close[-5:])
            ma20 = np.mean(close[-20:])
            trend_strength = (ma5 - ma20) / ma20
            
            # MACD
            ema12 = talib.EMA(close, timeperiod=12)
            ema26 = talib.EMA(close, timeperiod=26)
            macd = ema12[-1] - ema26[-1]
            
            # RSI
            rsi = talib.RSI(close, timeperiod=14)[-1]
            
            # 布林带
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            bb_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
            
            # 组合特征
            features = [
                momentum_1d, momentum_5d, momentum_20d,
                volatility, volume_ratio, trend_strength,
                macd, rsi, bb_position
            ]
            
            # 检查特征是否包含无穷大或NaN
            if np.isnan(features).any() or np.isinf(features).any():
                self.log(f"股票 {data._name} 特征包含无穷大或NaN值")
                # 替换无穷大和NaN
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
            return features
            
        except Exception as e:
            self.log(f"准备股票 {data._name} 特征时出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def run_backtest(self, start_date, end_date):
        """运行回测（优化版本）"""
        logger.info("开始回测...")
        
        try:
            # 1. 准备训练数据
            logger.info("准备训练数据...")
            # 获取训练用的股票池 - 扩大到全部A股
            all_stocks = xtdata.get_stock_list_in_sector('沪深A股')
            logger.info(f"总共选取 {len(all_stocks)} 只股票作为训练样本")
            
            # 设置训练数据时间范围为2023年全年
            training_start = '20230101'  # 2023年1月1日
            training_end = '20231231'    # 2023年12月31日
            logger.info(f"训练数据时间范围: {training_start} 至 {training_end}")
            
            # 并行处理训练数据
            X, y = self.parallel_prepare_data(all_stocks, training_start, training_end)
            
            if X.empty or y.empty:
                logger.error("未能获取有效的训练数据，无法继续回测")
                return
                
            logger.info(f"成功获取 {len(X)} 条训练数据")
            
            # 训练模型
            logger.info("训练机器学习模型...")
            if not self.train_model(X, y):
                logger.error("模型训练失败，无法继续回测")
                return
            
            # 2. 设置回测
            logger.info("创建Cerebro实例...")
            cerebro = bt.Cerebro()
            
            # 设置初始资金
            self.initial_cash = 1000000.0  # 设置为固定值，确保一致性
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=0.0003)  # 设置手续费为0.03%
            
            # 设置交易相关参数
            cerebro.broker.set_slippage_perc(0.001)  # 设置滑点为0.1%
            cerebro.broker.set_coc(False)  # 不使用收盘价交易
            cerebro.broker.set_checksubmit(True)  # 检查提交的订单
            
            # 打印初始设置
            logger.info(f"初始资金设置: {self.initial_cash:.2f}")
            logger.info(f"手续费设置: 0.03%")
            logger.info(f"滑点设置: 0.1%")
            
            # 添加策略，并传入模型和标准化器
            logger.info("添加策略...")
            cerebro.addstrategy(AdaptiveStopStrategy,
                               position_size=self.position_size,
                               stop_loss=self.stop_loss,
                               take_profit=self.take_profit,
                               max_positions=self.max_positions,
                               model=self.model,
                               scaler=self.scaler)
            
            # 添加数据
            logger.info("准备回测数据...")
            # 使用全部A股作为回测股票池
            backtest_stocks = all_stocks
            logger.info(f"选择 {len(backtest_stocks)} 只股票进行回测")
            
            # 使用实际的回测时间范围
            actual_start_date = datetime.strptime(start_date, '%Y%m%d')
            actual_end_date = datetime.strptime(end_date, '%Y%m%d')
            logger.info(f"回测时间范围: {actual_start_date.strftime('%Y-%m-%d')} 至 {actual_end_date.strftime('%Y-%m-%d')}")
            
            # 记录初始资金
            logger.info(f"初始资金: {self.initial_cash:.2f}")
            
            data_added = False
            for stock in backtest_stocks:
                try:
                    # 获取股票数据
                    logger.info(f"获取股票 {stock} 的历史数据...")
                    raw_data = xtdata.get_market_data(
                        stock_list=[stock],
                        period='1d',
                        start_time=start_date,
                        end_time=end_date,
                        field_list=['open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # 检查数据有效性
                    if raw_data is None:
                        logger.warning(f"股票 {stock} 数据为None，跳过")
                        continue
                        
                    logger.info(f"股票 {stock} 原始数据字段: {list(raw_data.keys())}")
                    
                    if 'close' not in raw_data:
                        logger.warning(f"股票 {stock} 数据中没有close字段，跳过")
                        continue
                    
                    logger.info(f"股票 {stock} close数据索引: {raw_data['close'].index}")
                    
                    if stock not in raw_data['close'].index:
                        logger.warning(f"股票 {stock} 不在close索引中，跳过")
                        continue
                    
                    # 提取数据
                    logger.info(f"提取股票 {stock} 的数据...")
                    df = pd.DataFrame({
                        'open': raw_data['open'].loc[stock],
                        'high': raw_data['high'].loc[stock],
                        'low': raw_data['low'].loc[stock],
                        'close': raw_data['close'].loc[stock],
                        'volume': raw_data['volume'].loc[stock]
                    })
                    
                    # 检查数据是否为空
                    if df.empty:
                        logger.warning(f"股票 {stock} 数据为空，跳过")
                        continue
                        
                    # 检查数据长度
                    if len(df) < 20:  # 至少需要20个交易日的数据
                        logger.warning(f"股票 {stock} 数据长度不足，跳过")
                        continue
                    
                    # 检查并处理NaN值
                    original_len = len(df)
                    df = df.dropna()
                    
                    if len(df) < original_len:
                        logger.info(f"删除了 {original_len - len(df)} 行包含NaN的数据")
                    
                    if df.empty:
                        logger.warning(f"股票 {stock} 清理后数据为空，跳过")
                        continue
                    
                    # 打印数据样本以便调试
                    logger.info(f"股票 {stock} 数据样本:\n{df.head()}")
                    logger.info(f"股票 {stock} 数据统计:\n{df.describe()}")
                    
                    # 添加到回测系统
                    logger.info(f"将股票 {stock} 添加到回测系统...")
                    data_feed = XTDataFeed(dataname=df, name=stock)
                    cerebro.adddata(data_feed)
                    data_added = True
                    logger.info(f"成功添加股票 {stock} 数据，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}, 数据点数: {len(df)}")
                    
                    # 检查数据是否可以正常访问
                    try:
                        # 尝试访问第一个数据点
                        first_close = data_feed.close[0]
                        logger.info(f"成功访问股票 {stock} 的第一个收盘价: {first_close}")
                    except Exception as e:
                        logger.warning(f"无法访问股票 {stock} 的数据: {str(e)}")
                        
                except Exception as e:
                    logger.warning(f"添加股票 {stock} 数据时出错: {str(e)}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    continue
            
            if not data_added:
                logger.error("没有有效的股票数据，无法进行回测")
                return
                
            # 添加分析器
            logger.info("添加分析器...")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # 打印初始投资组合价值
            logger.info(f"初始投资组合价值: {cerebro.broker.getvalue():.2f}")
            
            # 运行回测
            logger.info("执行回测...")
            results = cerebro.run(stdstats=True)  # 启用标准统计
            
            if not results:
                logger.error("回测结果为空")
                return
                
            # 获取策略实例
            strat = results[0]
            logger.info(f"回测完成，获取到策略实例: {strat}")
            
            # 打印最终投资组合价值
            logger.info(f"最终投资组合价值: {cerebro.broker.getvalue():.2f}")
            logger.info(f"总收益: {cerebro.broker.getvalue() - self.initial_cash:.2f}")
            logger.info(f"收益率: {(cerebro.broker.getvalue() / self.initial_cash - 1) * 100:.2f}%")
            
            # 生成详细报告
            self.generate_detailed_report(results[0])
            
            # 可选：绘制回测结果图表
            # cerebro.plot()
            
        except Exception as e:
            logger.error(f"回测过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def generate_detailed_report(self, result):
        """生成详细的回测报告"""
        try:
            # 获取分析器结果
            sharpe = result.analyzers.sharpe.get_analysis()
            drawdown = result.analyzers.drawdown.get_analysis()
            returns = result.analyzers.returns.get_analysis()
            trades = result.analyzers.trades.get_analysis()
            
            # 打印详细报告
            logger.info("\n====== 详细回测报告 ======")
            
            # 收益率
            if 'rtot' in returns:
                logger.info(f"总收益率: {returns['rtot']:.2%}")
            else:
                logger.info("总收益率: 无法计算")
                
            if 'rnorm' in returns:
                logger.info(f"年化收益率: {returns['rnorm']:.2%}")
            else:
                logger.info("年化收益率: 无法计算")
                
            # 夏普比率
            if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
                logger.info(f"夏普比率: {sharpe['sharperatio']:.2f}")
            else:
                logger.info("夏普比率: 无法计算")
                
            # 回撤
            if 'max' in drawdown and 'drawdown' in drawdown['max']:
                logger.info(f"最大回撤: {drawdown['max']['drawdown']:.2%}")
                logger.info(f"最大回撤期间: {drawdown['max']['len']} 天")
            else:
                logger.info("最大回撤: 无法计算")
            
            # 交易统计
            if trades and 'total' in trades and 'total' in trades['total']:
                total_trades = trades['total']['total']
                logger.info(f"总交易次数: {total_trades}")
                
                # 盈利交易
                if 'won' in trades and 'total' in trades['won']:
                    won_trades = trades['won']['total']
                    logger.info(f"盈利交易: {won_trades}")
                    
                    # 胜率
                    if total_trades > 0:
                        win_rate = won_trades / total_trades
                        logger.info(f"胜率: {win_rate:.2%}")
                    else:
                        logger.info("胜率: 无法计算 (无交易)")
                else:
                    logger.info("盈利交易: 0")
                    logger.info("胜率: 0.00%")
                
                # 亏损交易
                if 'lost' in trades and 'total' in trades['lost']:
                    logger.info(f"亏损交易: {trades['lost']['total']}")
                else:
                    logger.info("亏损交易: 0")
                
                # 平均持仓时间
                if 'len' in trades and 'average' in trades['len']:
                    logger.info(f"平均持仓时间: {trades['len']['average']:.1f} 天")
                else:
                    logger.info("平均持仓时间: 无法计算")
                
                # 最大盈利和亏损
                if 'won' in trades and 'pnl' in trades['won'] and 'max' in trades['won']['pnl']:
                    logger.info(f"最大单笔盈利: {trades['won']['pnl']['max']:.2f}")
                else:
                    logger.info("最大单笔盈利: 无法计算")
                    
                if 'lost' in trades and 'pnl' in trades['lost'] and 'max' in trades['lost']['pnl']:
                    logger.info(f"最大单笔亏损: {trades['lost']['pnl']['max']:.2f}")
                else:
                    logger.info("最大单笔亏损: 无法计算")
                
                # 收益回撤比
                if 'rtot' in returns and 'max' in drawdown and 'drawdown' in drawdown['max'] and drawdown['max']['drawdown'] != 0:
                    rr_ratio = returns['rtot'] / abs(drawdown['max']['drawdown'])
                    logger.info(f"收益回撤比: {rr_ratio:.2f}")
                else:
                    logger.info("收益回撤比: 无法计算")
            else:
                logger.warning("没有产生任何交易")
            
            logger.info("====================")
        except Exception as e:
            logger.error(f"生成报告时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

# 在类外部定义处理单个股票的函数
def _process_stock_for_training(strategy, code, start_date, end_date):
    """处理单个股票的训练数据（用于多进程）"""
    try:
        X, y = strategy.prepare_training_data(code, start_date, end_date)
        if not X.empty and not y.empty:
            return (code, X, y)
        return None
    except Exception as e:
        logger.warning(f"处理股票 {code} 时出错: {str(e)}")
        return None

class QuantStrategy:
    def __init__(self, initial_capital=1000000, position_size=0.2, stop_loss=0.05, 
                 take_profit=0.15, max_positions=5):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_positions = max_positions
        
        # 初始化数据管理器
        self.data_manager = StockData()
        
        # 机器学习模型
        self.model = RandomForestClassifier(
            n_estimators=200,  # 增加树的数量
            max_depth=5,       # 控制过拟合
            random_state=42,
            n_jobs=-1          # 并行训练
        )
        self.scaler = StandardScaler()
        
    def prepare_training_data(self, stock_code, start_date, end_date):
        """准备机器学习训练数据（优化版本）"""
        try:
            # 获取数据
            data = self.data_manager.get_stock_data(
                stock_code, start_date, end_date,
                ['open', 'high', 'low', 'close', 'volume']
            )
            
            if data is None:
                logger.warning(f"股票 {stock_code} 数据为None")
                return pd.DataFrame(), pd.Series()
                
            if 'close' not in data or stock_code not in data['close'].index:
                logger.warning(f"股票 {stock_code} 数据中没有close列或索引中不包含该股票")
                return pd.DataFrame(), pd.Series()
                
            # 计算因子
            X = calculate_factors(data)
            if X.empty:
                logger.warning(f"股票 {stock_code} 计算因子结果为空")
                return pd.DataFrame(), pd.Series()
            
            # 处理无穷大值和NaN值
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # 计算未来收益作为标签
            close = data['close'].loc[stock_code]
            future_returns = close.shift(-5).pct_change(5)
            y = (future_returns > future_returns.mean()).astype(int)
            
            # 删除含有NaN的行
            X_clean = X.dropna()
            y_clean = y.loc[X_clean.index].dropna()
            
            # 确保X和y的索引一致
            common_index = X_clean.index.intersection(y_clean.index)
            X_final = X_clean.loc[common_index]
            y_final = y_clean.loc[common_index]
            
            return X_final, y_final
            
        except Exception as e:
            logger.warning(f"处理股票 {stock_code} 数据时出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame(), pd.Series()
            
    def parallel_prepare_data(self, stock_codes, start_date, end_date):
        """并行处理训练数据（使用硬盘缓存和多进程）"""
        logger.info("开始处理训练数据...")
        log_memory_usage()
        
        # 创建临时文件用于存储中间结果
        temp_results_file = os.path.join(TEMP_DIR, f"batch_results_{start_date}_{end_date}.joblib")
        
        # 检查是否有缓存的结果
        if os.path.exists(temp_results_file):
            logger.info(f"发现缓存的训练数据，正在加载: {temp_results_file}")
            try:
                all_data = load(temp_results_file)
                logger.info(f"成功加载缓存数据，共 {len(all_data['X'])} 条记录")
                return all_data['X'], all_data['y']
            except Exception as e:
                logger.warning(f"加载缓存数据失败: {str(e)}，将重新处理数据")
        
        # 对股票池进行采样，如果数量太多
        if len(stock_codes) > 500:  # 限制处理的股票数量
            logger.info(f"股票池过大 ({len(stock_codes)} 只)，随机采样 500 只进行训练")
            random.seed(42)  # 设置随机种子确保可重复性
            stock_codes = random.sample(stock_codes, 500)
        
        # 分批处理以减少内存压力
        batch_size = 50  # 增加批次大小
        all_X_files = []
        all_y_files = []
        
        # 使用多进程并行处理
        num_workers = min(os.cpu_count(), 8)  # 限制最大进程数
        logger.info(f"使用 {num_workers} 个进程并行处理数据")
        
        # 处理所有股票
        for i in range(0, len(stock_codes), batch_size):
            batch = stock_codes[i:i+batch_size]
            logger.info(f"处理第 {i//batch_size + 1}/{(len(stock_codes)-1)//batch_size + 1} 批数据，共 {len(batch)} 只股票")
            
            # 由于多进程的限制，改为串行处理
            results = []
            for code in batch:
                result = _process_stock_for_training(self, code, start_date, end_date)
                results.append(result)
            
            # 过滤有效结果
            valid_results = [r for r in results if r is not None]
            logger.info(f"批次 {i//batch_size + 1} 中有 {len(valid_results)}/{len(batch)} 只股票返回有效数据")
            
            # 合并批次结果并保存到临时文件
            if valid_results:
                try:
                    # 提取X和y
                    batch_X_list = [r[1] for r in valid_results]
                    batch_y_list = [r[2] for r in valid_results]
                    
                    # 合并数据
                    batch_X = pd.concat(batch_X_list)
                    batch_y = pd.concat(batch_y_list)
                    
                    # 清理内存
                    del batch_X_list, batch_y_list, valid_results, results
                    gc.collect()
                    
                    # 保存到临时文件
                    batch_file_X = os.path.join(TEMP_DIR, f"batch_X_{i}.joblib")
                    batch_file_y = os.path.join(TEMP_DIR, f"batch_y_{i}.joblib")
                    
                    dump(batch_X, batch_file_X)
                    dump(batch_y, batch_file_y)
                    
                    all_X_files.append(batch_file_X)
                    all_y_files.append(batch_file_y)
                    
                    logger.info(f"批次 {i//batch_size + 1} 处理完成，保存了 {len(batch_X)} 条数据")
                    
                    # 清理内存
                    del batch_X, batch_y
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"处理批次 {i//batch_size + 1} 时出错: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            log_memory_usage()
        
        # 合并所有批次的结果
        logger.info("合并所有批次结果...")
        all_X = pd.DataFrame()
        all_y = pd.Series()
        
        for x_file, y_file in zip(all_X_files, all_y_files):
            try:
                # 加载批次数据
                batch_X = load(x_file)
                batch_y = load(y_file)
                
                # 合并数据
                all_X = pd.concat([all_X, batch_X]) if not all_X.empty else batch_X
                all_y = pd.concat([all_y, batch_y]) if not all_y.empty else batch_y
                
                # 清理内存
                del batch_X, batch_y
                gc.collect()
                
                # 删除临时文件
                os.remove(x_file)
                os.remove(y_file)
                
            except Exception as e:
                logger.error(f"合并批次数据时出错: {str(e)}")
        
        # 如果数据量太大，进行采样
        if len(all_X) > 1000000:
            logger.info(f"最终数据量过大 ({len(all_X)} 条)，进行随机采样")
            # 使用分层采样保持类别分布
            from sklearn.model_selection import train_test_split
            all_X, _, all_y, _ = train_test_split(
                all_X, all_y, train_size=1000000, random_state=42, stratify=all_y
            )
            logger.info(f"采样后数据量: {len(all_X)} 条")
        
        # 保存最终结果到缓存
        logger.info(f"数据处理完成，共获取 {len(all_X)} 条有效数据，正在保存到缓存")
        dump({'X': all_X, 'y': all_y}, temp_results_file)
        
        return all_X, all_y
        
    def train_model(self, X, y):
        """训练机器学习模型（使用硬盘空间优化）"""
        try:
            # 创建临时目录
            TEMP_DIR = "temp_model_data"
            os.makedirs(TEMP_DIR, exist_ok=True)
            
            # 记录内存使用情况
            def log_memory_usage():
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            logger.info("开始训练模型...")
            log_memory_usage()
            
            # 检查数据中是否存在无穷大值或NaN
            if np.isinf(X.values).any() or np.isnan(X.values).any():
                logger.warning("训练数据中存在无穷大值或NaN，进行清理")
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.dropna()
                y = y.loc[X.index]
                
            if X.empty or y.empty:
                logger.warning("清理后的训练数据为空，无法训练模型")
                return False
                
            # 如果数据量太大，进行采样
            if len(X) > 500000:
                logger.info(f"数据量过大 ({len(X)} 条)，进行随机采样")
                # 使用分层采样保持类别分布
                from sklearn.model_selection import train_test_split
                X_sampled, _, y_sampled, _ = train_test_split(
                    X, y, train_size=500000, random_state=42, stratify=y
                )
                X = X_sampled
                y = y_sampled
                logger.info(f"采样后数据量: {len(X)} 条")
                
            # 保存数据到临时文件，使用内存映射
            X_file = os.path.join(TEMP_DIR, "X_train.npy")
            y_file = os.path.join(TEMP_DIR, "y_train.npy")
            
            # 将数据保存为numpy数组
            X_np = X.values
            y_np = y.values
            
            # 保存到磁盘
            np.save(X_file, X_np)
            np.save(y_file, y_np)
            
            # 清理内存
            del X_np, X, y_np, y
            gc.collect()
            
            # 使用内存映射加载数据
            X_mmap = np.load(X_file, mmap_mode='r')
            y_mmap = np.load(y_file, mmap_mode='r')
            
            logger.info(f"数据已保存到磁盘并使用内存映射加载，训练数据大小: {X_mmap.shape}")
            log_memory_usage()
            
            # 标准化数据 - 使用批处理
            logger.info("开始标准化数据...")
            
            # 先拟合所有数据
            batch_size = 10000
            for i in range(0, len(X_mmap), batch_size):
                end = min(i + batch_size, len(X_mmap))
                if i == 0:  # 第一批用于拟合
                    self.scaler.fit(X_mmap[i:end])
                else:  # 后续批次用于部分拟合
                    self.scaler.partial_fit(X_mmap[i:end])
            
            # 创建临时文件存储标准化后的数据
            X_scaled_file = os.path.join(TEMP_DIR, "X_scaled.npy")
            X_scaled = np.memmap(X_scaled_file, dtype='float64', mode='w+', shape=X_mmap.shape)
            
            # 分批转换
            for i in range(0, len(X_mmap), batch_size):
                end = min(i + batch_size, len(X_mmap))
                X_scaled[i:end] = self.scaler.transform(X_mmap[i:end])
            
            logger.info("数据标准化完成")
            log_memory_usage()
            
            # 配置随机森林模型，使用外部内存
            logger.info("开始训练随机森林模型...")
            
            # 使用更高效的模型配置
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,  # 减少树的数量以加快训练
                max_depth=10,      # 增加深度以提高准确性
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',  # 使用特征的平方根数量
                random_state=42,
                n_jobs=-1,  # 使用所有可用的CPU核心
                verbose=1,
                class_weight='balanced'  # 处理类别不平衡
            )
            
            # 使用批处理训练模型
            logger.info("开始拟合模型...")
            self.model.fit(X_scaled, y_mmap)
            
            logger.info("模型训练完成")
            log_memory_usage()
            
            # 评估模型性能
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # 使用小批量进行预测以避免内存问题
            y_pred = np.zeros_like(y_mmap)
            for i in range(0, len(X_mmap), batch_size):
                end = min(i + batch_size, len(X_mmap))
                y_pred[i:end] = self.model.predict(X_scaled[i:end])
            
            # 计算性能指标
            accuracy = accuracy_score(y_mmap, y_pred)
            precision = precision_score(y_mmap, y_pred, average='binary')
            recall = recall_score(y_mmap, y_pred, average='binary')
            f1 = f1_score(y_mmap, y_pred, average='binary')
            
            logger.info(f"模型性能评估:")
            logger.info(f"准确率: {accuracy:.4f}")
            logger.info(f"精确率: {precision:.4f}")
            logger.info(f"召回率: {recall:.4f}")
            logger.info(f"F1分数: {f1:.4f}")
            
            # 特征重要性
            feature_importances = self.model.feature_importances_
            logger.info(f"特征重要性: {feature_importances}")
            
            # 清理临时文件
            try:
                os.remove(X_file)
                os.remove(y_file)
                os.remove(X_scaled_file)
                logger.info("临时文件已清理")
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def run_backtest(self, start_date, end_date):
        """运行回测（优化版本）"""
        logger.info("开始回测...")
        
        try:
            # 1. 准备训练数据
            logger.info("准备训练数据...")
            # 获取训练用的股票池 - 扩大到全部A股
            all_stocks = xtdata.get_stock_list_in_sector('沪深A股')
            logger.info(f"总共选取 {len(all_stocks)} 只股票作为训练样本")
            
            # 设置训练数据时间范围为2023年全年
            training_start = '20230101'  # 2023年1月1日
            training_end = '20231231'    # 2023年12月31日
            logger.info(f"训练数据时间范围: {training_start} 至 {training_end}")
            
            # 并行处理训练数据
            X, y = self.parallel_prepare_data(all_stocks, training_start, training_end)
            
            if X.empty or y.empty:
                logger.error("未能获取有效的训练数据，无法继续回测")
                return
                
            logger.info(f"成功获取 {len(X)} 条训练数据")
            
            # 训练模型
            logger.info("训练机器学习模型...")
            if not self.train_model(X, y):
                logger.error("模型训练失败，无法继续回测")
                return
            
            # 2. 设置回测
            logger.info("创建Cerebro实例...")
            cerebro = bt.Cerebro()
            
            # 设置初始资金
            self.initial_cash = 1000000.0  # 设置为固定值，确保一致性
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=0.0003)  # 设置手续费为0.03%
            
            # 设置交易相关参数
            cerebro.broker.set_slippage_perc(0.001)  # 设置滑点为0.1%
            cerebro.broker.set_coc(False)  # 不使用收盘价交易
            cerebro.broker.set_checksubmit(True)  # 检查提交的订单
            
            # 打印初始设置
            logger.info(f"初始资金设置: {self.initial_cash:.2f}")
            logger.info(f"手续费设置: 0.03%")
            logger.info(f"滑点设置: 0.1%")
            
            # 添加策略，并传入模型和标准化器
            logger.info("添加策略...")
            cerebro.addstrategy(AdaptiveStopStrategy,
                               position_size=self.position_size,
                               stop_loss=self.stop_loss,
                               take_profit=self.take_profit,
                               max_positions=self.max_positions,
                               model=self.model,
                               scaler=self.scaler)
            
            # 添加数据
            logger.info("准备回测数据...")
            # 使用全部A股作为回测股票池
            backtest_stocks = all_stocks
            logger.info(f"选择 {len(backtest_stocks)} 只股票进行回测")
            
            # 使用实际的回测时间范围
            actual_start_date = datetime.strptime(start_date, '%Y%m%d')
            actual_end_date = datetime.strptime(end_date, '%Y%m%d')
            logger.info(f"回测时间范围: {actual_start_date.strftime('%Y-%m-%d')} 至 {actual_end_date.strftime('%Y-%m-%d')}")
            
            # 记录初始资金
            logger.info(f"初始资金: {self.initial_cash:.2f}")
            
            data_added = False
            for stock in backtest_stocks:
                try:
                    # 获取股票数据
                    logger.info(f"获取股票 {stock} 的历史数据...")
                    raw_data = xtdata.get_market_data(
                        stock_list=[stock],
                        period='1d',
                        start_time=start_date,
                        end_time=end_date,
                        field_list=['open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # 检查数据有效性
                    if raw_data is None:
                        logger.warning(f"股票 {stock} 数据为None，跳过")
                        continue
                        
                    logger.info(f"股票 {stock} 原始数据字段: {list(raw_data.keys())}")
                    
                    if 'close' not in raw_data:
                        logger.warning(f"股票 {stock} 数据中没有close字段，跳过")
                        continue
                    
                    logger.info(f"股票 {stock} close数据索引: {raw_data['close'].index}")
                    
                    if stock not in raw_data['close'].index:
                        logger.warning(f"股票 {stock} 不在close索引中，跳过")
                        continue
                    
                    # 提取数据
                    logger.info(f"提取股票 {stock} 的数据...")
                    df = pd.DataFrame({
                        'open': raw_data['open'].loc[stock],
                        'high': raw_data['high'].loc[stock],
                        'low': raw_data['low'].loc[stock],
                        'close': raw_data['close'].loc[stock],
                        'volume': raw_data['volume'].loc[stock]
                    })
                    
                    # 检查数据是否为空
                    if df.empty:
                        logger.warning(f"股票 {stock} 数据为空，跳过")
                        continue
                        
                    # 检查数据长度
                    if len(df) < 20:  # 至少需要20个交易日的数据
                        logger.warning(f"股票 {stock} 数据长度不足，跳过")
                        continue
                    
                    # 检查并处理NaN值
                    original_len = len(df)
                    df = df.dropna()
                    
                    if len(df) < original_len:
                        logger.info(f"删除了 {original_len - len(df)} 行包含NaN的数据")
                    
                    if df.empty:
                        logger.warning(f"股票 {stock} 清理后数据为空，跳过")
                        continue
                    
                    # 打印数据样本以便调试
                    logger.info(f"股票 {stock} 数据样本:\n{df.head()}")
                    logger.info(f"股票 {stock} 数据统计:\n{df.describe()}")
                    
                    # 添加到回测系统
                    logger.info(f"将股票 {stock} 添加到回测系统...")
                    data_feed = XTDataFeed(dataname=df, name=stock)
                    cerebro.adddata(data_feed)
                    data_added = True
                    logger.info(f"成功添加股票 {stock} 数据，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}, 数据点数: {len(df)}")
                    
                    # 检查数据是否可以正常访问
                    try:
                        # 尝试访问第一个数据点
                        first_close = data_feed.close[0]
                        logger.info(f"成功访问股票 {stock} 的第一个收盘价: {first_close}")
                    except Exception as e:
                        logger.warning(f"无法访问股票 {stock} 的数据: {str(e)}")
                        
                except Exception as e:
                    logger.warning(f"添加股票 {stock} 数据时出错: {str(e)}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    continue
            
            if not data_added:
                logger.error("没有有效的股票数据，无法进行回测")
                return
                
            # 添加分析器
            logger.info("添加分析器...")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # 打印初始投资组合价值
            logger.info(f"初始投资组合价值: {cerebro.broker.getvalue():.2f}")
            
            # 运行回测
            logger.info("执行回测...")
            results = cerebro.run(stdstats=True)  # 启用标准统计
            
            if not results:
                logger.error("回测结果为空")
                return
                
            # 获取策略实例
            strat = results[0]
            logger.info(f"回测完成，获取到策略实例: {strat}")
            
            # 打印最终投资组合价值
            logger.info(f"最终投资组合价值: {cerebro.broker.getvalue():.2f}")
            logger.info(f"总收益: {cerebro.broker.getvalue() - self.initial_cash:.2f}")
            logger.info(f"收益率: {(cerebro.broker.getvalue() / self.initial_cash - 1) * 100:.2f}%")
            
            # 生成详细报告
            self.generate_detailed_report(results[0])
            
            # 可选：绘制回测结果图表
            # cerebro.plot()
            
        except Exception as e:
            logger.error(f"回测过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def generate_detailed_report(self, result):
        """生成详细的回测报告"""
        try:
            # 获取分析器结果
            sharpe = result.analyzers.sharpe.get_analysis()
            drawdown = result.analyzers.drawdown.get_analysis()
            returns = result.analyzers.returns.get_analysis()
            trades = result.analyzers.trades.get_analysis()
            
            # 打印详细报告
            logger.info("\n====== 详细回测报告 ======")
            
            # 收益率
            if 'rtot' in returns:
                logger.info(f"总收益率: {returns['rtot']:.2%}")
            else:
                logger.info("总收益率: 无法计算")
                
            if 'rnorm' in returns:
                logger.info(f"年化收益率: {returns['rnorm']:.2%}")
            else:
                logger.info("年化收益率: 无法计算")
                
            # 夏普比率
            if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
                logger.info(f"夏普比率: {sharpe['sharperatio']:.2f}")
            else:
                logger.info("夏普比率: 无法计算")
                
            # 回撤
            if 'max' in drawdown and 'drawdown' in drawdown['max']:
                logger.info(f"最大回撤: {drawdown['max']['drawdown']:.2%}")
                logger.info(f"最大回撤期间: {drawdown['max']['len']} 天")
            else:
                logger.info("最大回撤: 无法计算")
            
            # 交易统计
            if trades and 'total' in trades and 'total' in trades['total']:
                total_trades = trades['total']['total']
                logger.info(f"总交易次数: {total_trades}")
                
                # 盈利交易
                if 'won' in trades and 'total' in trades['won']:
                    won_trades = trades['won']['total']
                    logger.info(f"盈利交易: {won_trades}")
                    
                    # 胜率
                    if total_trades > 0:
                        win_rate = won_trades / total_trades
                        logger.info(f"胜率: {win_rate:.2%}")
                    else:
                        logger.info("胜率: 无法计算 (无交易)")
                else:
                    logger.info("盈利交易: 0")
                    logger.info("胜率: 0.00%")
                
                # 亏损交易
                if 'lost' in trades and 'total' in trades['lost']:
                    logger.info(f"亏损交易: {trades['lost']['total']}")
                else:
                    logger.info("亏损交易: 0")
                
                # 平均持仓时间
                if 'len' in trades and 'average' in trades['len']:
                    logger.info(f"平均持仓时间: {trades['len']['average']:.1f} 天")
                else:
                    logger.info("平均持仓时间: 无法计算")
                
                # 最大盈利和亏损
                if 'won' in trades and 'pnl' in trades['won'] and 'max' in trades['won']['pnl']:
                    logger.info(f"最大单笔盈利: {trades['won']['pnl']['max']:.2f}")
                else:
                    logger.info("最大单笔盈利: 无法计算")
                    
                if 'lost' in trades and 'pnl' in trades['lost'] and 'max' in trades['lost']['pnl']:
                    logger.info(f"最大单笔亏损: {trades['lost']['pnl']['max']:.2f}")
                else:
                    logger.info("最大单笔亏损: 无法计算")
                
                # 收益回撤比
                if 'rtot' in returns and 'max' in drawdown and 'drawdown' in drawdown['max'] and drawdown['max']['drawdown'] != 0:
                    rr_ratio = returns['rtot'] / abs(drawdown['max']['drawdown'])
                    logger.info(f"收益回撤比: {rr_ratio:.2f}")
                else:
                    logger.info("收益回撤比: 无法计算")
            else:
                logger.warning("没有产生任何交易")
            
            logger.info("====================")
        except Exception as e:
            logger.error(f"生成报告时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 设置回测参数
    start_date = '20240701'  # 修改为2024年7月1日
    end_date = '20241231'    # 修改为2024年12月31日
    
    # 创建策略实例
    strategy = QuantStrategy(
        initial_capital=1000000,
        position_size=0.2,
        stop_loss=0.05,
        take_profit=0.15,
        max_positions=5
    )
    
    # 运行回测
    try:
        strategy.run_backtest(start_date, end_date)
        logger.info("回测完成")
    except Exception as e:
        logger.error(f"回测过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
