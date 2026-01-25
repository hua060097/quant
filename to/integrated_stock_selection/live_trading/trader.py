"""
交易执行模块

基于迅投XtQuant API实现实盘交易
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# 自动添加xtquant路径到sys.path
def _setup_xtquant_path():
    """
    自动查找并添加xtquant到sys.path

    查找顺序：
    1. 检查当前目录及父目录下的xtquant文件夹
    2. 检查项目根目录的xtquant文件夹
    """
    # 如果已经能导入，则不需要处理
    try:
        import xtquant
        return
    except ImportError:
        pass

    # 获取当前文件的路径
    current_file = Path(__file__).resolve()

    # 可能的xtquant路径
    possible_paths = [
        # 当前目录的xtquant
        current_file.parent / 'xtquant',
        # 父目录的xtquant
        current_file.parent.parent / 'xtquant',
        # 父目录的父目录的xtquant (to/xtquant)
        current_file.parent.parent.parent / 'xtquant',
        # 系统级安装路径
        Path(sys.prefix) / 'Lib' / 'site-packages' / 'xtquant',
        Path(sys.prefix) / 'lib' / 'python3.12' / 'site-packages' / 'xtquant',
    ]

    for path in possible_paths:
        if path.exists() and path.is_dir():
            # 将父目录添加到sys.path（这样import xtquant才能找到）
            parent_dir = str(path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                logging.getLogger(__name__).info(f"添加xtquant路径: {parent_dir}")
            return

    # 如果都找不到，尝试在sys.path中查找
    for path in sys.path:
        xtquant_path = Path(path) / 'xtquant'
        if xtquant_path.exists():
            return

    # 还是找不到，记录警告但不阻止程序运行
    logging.getLogger(__name__).warning(
        "未找到xtquant目录，将在运行时动态导入"
    )

# 执行路径设置
_setup_xtquant_path()

from .config import XtQuantConfig, TradingConfig, MarketTimeConfig, PathConfig
from .order_manager import SmartOrderManager, LiveTradingCallback


class XtQuantTrader:
    """迅投交易器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.xt_trader = None
        self.account = None
        self.connected = False

        # 持仓记录
        self.positions = {}

        # 订单记录
        self.orders = {}

        # 订单管理器(连接后初始化)
        self.order_manager = None

        # 交易日历(由scheduler设置)
        self.trading_dates = []
    
    def connect(self) -> bool:
        """
        连接迅投API
        
        Returns:
            bool: 是否连接成功
        """
        try:
            # 检查配置
            if not XtQuantConfig.USERDATA_MINI_PATH:
                self.logger.error("未配置USERDATA_MINI_PATH,请在config.py中设置")
                return False
            
            if not XtQuantConfig.ACCOUNT_ID:
                self.logger.error("未配置ACCOUNT_ID,请在config.py中设置")
                return False
            
            # 导入迅投API
            try:
                from xtquant import xttrader
                from xtquant.xttype import StockAccount
            except ImportError:
                self.logger.error("未安装xtquant库,请先安装: pip install xtquant")
                return False

            # 生成唯一的session_id（确保多实例运行时不冲突）
            session_id = XtQuantConfig.generate_session_id()
            time_offset = session_id - XtQuantConfig.SESSION_ID_BASE

            # 创建交易对象
            self.xt_trader = xttrader.XtQuantTrader(
                XtQuantConfig.USERDATA_MINI_PATH,
                session_id  # ✅ 使用动态生成的session_id
            )
            self.logger.info(f"创建交易对象: session_id={session_id}")
            self.logger.info(f"  (基础ID={XtQuantConfig.SESSION_ID_BASE} + 偏移={time_offset})")

            # 创建账号对象
            self.account = StockAccount(XtQuantConfig.ACCOUNT_ID)
            self.logger.info(f"创建账号对象: {XtQuantConfig.ACCOUNT_ID}")

            # 初始化智能订单管理器
            self.order_manager = SmartOrderManager(self)

            # 注册交易回调（必须在start之前）
            self.callback = LiveTradingCallback(self.order_manager)
            self.xt_trader.register_callback(self.callback)
            self.logger.info("✅ 交易回调已注册")

            # 启动交易线程（必须在connect之前）
            self.xt_trader.start()
            self.logger.info("✅ 交易线程已启动")

            # 连接（返回0表示成功）
            connect_result = self.xt_trader.connect()
            if connect_result != 0:
                self.logger.error(f"连接失败,错误码: {connect_result}")
                self.logger.error("可能的原因:")
                self.logger.error("  1. MiniQMT客户端未启动")
                self.logger.error("  2. USERDATA_MINI_PATH路径配置错误")
                self.logger.error("  3. 账号ID配置错误")
                return False
            self.logger.info("✅ 连接成功")

            # 订阅账号（返回0表示成功）
            subscribe_result = self.xt_trader.subscribe(self.account)
            if subscribe_result != 0:
                self.logger.error(f"订阅账号失败,错误码: {subscribe_result}")
                return False
            self.logger.info("✅ 账号订阅成功")

            self.connected = True
            self.logger.info("✅ 迅投API连接完成")
            self.logger.info("✅ 智能订单管理器已初始化")

            return True
            
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.xt_trader and self.connected:
            # 停止交易线程（XtQuantTrader只有stop方法，没有disconnect方法）
            try:
                self.xt_trader.stop()
                self.connected = False
                self.logger.info("✅ 交易线程已停止，连接已断开")
            except Exception as e:
                self.logger.warning(f"停止交易线程时出错: {e}")
                self.connected = False
    
    def get_account_info(self) -> Optional[Dict]:
        """
        获取账户信息
        
        Returns:
            Dict: 账户信息
        """
        if not self.connected:
            self.logger.error("未连接")
            return None
        
        try:
            # 查询资金
            asset = self.xt_trader.query_stock_asset(self.account)
            
            if asset:
                info = {
                    'total_asset': asset.total_asset,      # 总资产
                    'cash': asset.cash,                    # 可用资金
                    'market_value': asset.market_value,    # 持仓市值
                    'frozen_cash': asset.frozen_cash       # 冻结资金
                }
                
                self.logger.info(f"账户信息: 总资产={info['total_asset']:.2f}, "
                               f"可用={info['cash']:.2f}, 市值={info['market_value']:.2f}")
                
                return info
            else:
                self.logger.error("查询账户信息失败")
                return None
                
        except Exception as e:
            self.logger.error(f"获取账户信息失败: {e}")
            return None
    
    def buy_stock(self, stock_code: str, amount: float) -> Optional[str]:
        """
        买入股票
        
        Args:
            stock_code: 股票代码(如600000.SH)
            amount: 买入金额(元)
        
        Returns:
            str: 订单ID,失败返回None
        """
        if not self.connected:
            self.logger.error("未连接")
            return None
        
        try:
            # 获取最新价
            price = self._get_latest_price(stock_code)
            if price is None:
                self.logger.error(f"获取{stock_code}最新价失败")
                return None
            
            # 计算股数(100股整数倍)
            volume = int(amount / price / 100) * 100
            
            if volume < 100:
                self.logger.warning(f"{stock_code}金额不足买入1手,跳过")
                return None
            
            # 下单
            order_id = self.xt_trader.order_stock(
                self.account,
                stock_code,
                TradingConfig.ORDER_TYPE_BUY,
                volume,
                TradingConfig.PRICE_TYPE_BUY,
                price
            )
            
            if order_id > 0:
                self.logger.info(f"✅ 买入委托成功: {stock_code}, {volume}股, 价格{price:.2f}")
                self.orders[order_id] = {
                    'stock_code': stock_code,
                    'type': 'buy',
                    'volume': volume,
                    'price': price,
                    'time': datetime.now()
                }
                return str(order_id)
            else:
                self.logger.error(f"买入委托失败: {stock_code}, 错误码{order_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"买入失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def sell_stock(self, stock_code: str, volume: int) -> Optional[str]:
        """
        卖出股票
        
        Args:
            stock_code: 股票代码
            volume: 卖出股数
        
        Returns:
            str: 订单ID,失败返回None
        """
        if not self.connected:
            self.logger.error("未连接")
            return None
        
        try:
            # 获取最新价
            price = self._get_latest_price(stock_code)
            if price is None:
                self.logger.error(f"获取{stock_code}最新价失败")
                return None
            
            # 下单
            order_id = self.xt_trader.order_stock(
                self.account,
                stock_code,
                TradingConfig.ORDER_TYPE_SELL,
                volume,
                TradingConfig.PRICE_TYPE_SELL,
                price
            )
            
            if order_id > 0:
                self.logger.info(f"✅ 卖出委托成功: {stock_code}, {volume}股, 价格{price:.2f}")
                self.orders[order_id] = {
                    'stock_code': stock_code,
                    'type': 'sell',
                    'volume': volume,
                    'price': price,
                    'time': datetime.now()
                }
                return str(order_id)
            else:
                self.logger.error(f"卖出委托失败: {stock_code}, 错误码{order_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"卖出失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _get_latest_price(self, stock_code: str) -> Optional[float]:
        """
        获取最新价（私有方法）

        Args:
            stock_code: 股票代码

        Returns:
            float: 最新价
        """
        try:
            from xtquant import xtdata

            # 获取最新行情
            quote = xtdata.get_full_tick([stock_code])

            if stock_code in quote:
                price = quote[stock_code]['lastPrice']
                return price
            else:
                return None

        except Exception as e:
            self.logger.error(f"获取最新价失败: {e}")
            return None

    def get_current_price(self, stock_code: str) -> Optional[float]:
        """
        获取最新价（公共方法）

        Args:
            stock_code: 股票代码

        Returns:
            float: 最新价
        """
        return self._get_latest_price(stock_code)

    def get_stock_atr(self, stock_code: str, period: int = 14) -> Optional[float]:
        """
        获取股票ATR值

        Args:
            stock_code: 股票代码
            period: ATR周期

        Returns:
            float: ATR值
        """
        try:
            from xtquant import xtdata
            import pandas as pd
            import numpy as np

            # 获取历史数据(需要period+1天)
            data = xtdata.get_market_data(
                stock_list=[stock_code],
                period='1d',
                count=period + 1,
                field_list=['high', 'low', 'close']
            )

            if stock_code not in data or len(data[stock_code]) < period:
                self.logger.warning(f"{stock_code}历史数据不足,无法计算ATR")
                return None

            df = data[stock_code]

            # 计算TR (True Range)
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # 计算ATR (简单移动平均)
            atr = tr.rolling(window=period).mean().iloc[-1]

            return float(atr)

        except Exception as e:
            self.logger.error(f"获取ATR失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return None

    def get_positions(self) -> Dict[str, Dict]:
        """
        获取持仓
        
        Returns:
            Dict: 持仓字典 {stock_code: {volume, cost_price, ...}}
        """
        if not self.connected:
            self.logger.error("未连接")
            return {}
        
        try:
            positions = self.xt_trader.query_stock_positions(self.account)
            
            result = {}
            if positions:
                for pos in positions:
                    result[pos.stock_code] = {
                        'volume': pos.volume,              # 持仓数量
                        'can_use_volume': pos.can_use_volume,  # 可用数量
                        'open_price': pos.open_price,      # 开仓价
                        'market_value': pos.market_value   # 市值
                    }
            
            return result
            
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            return {}

    def order_stock_async(self, account, stock_code: str, order_type: int,
                         order_volume: int, price_type: int, price: float,
                         strategy_name: str, order_remark: str) -> int:
        """
        异步下单

        Args:
            account: 账户对象
            stock_code: 股票代码
            order_type: 订单类型（买/卖）
            order_volume: 委托数量
            price_type: 价格类型
            price: 价格
            strategy_name: 策略名称
            order_remark: 订单备注

        Returns:
            int: 订单号，失败返回-1
        """
        if not self.connected:
            self.logger.error("未连接")
            return -1

        try:
            return self.xt_trader.order_stock(
                account,
                stock_code,
                order_type,
                order_volume,
                price_type,
                price,
                strategy_name,
                order_remark
            )
        except Exception as e:
            self.logger.error(f"异步下单失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return -1

    def cancel_order_stock(self, account, order_id: str) -> bool:
        """
        撤单

        Args:
            account: 账户对象
            order_id: 订单号

        Returns:
            bool: 是否成功
        """
        if not self.connected:
            self.logger.error("未连接")
            return False

        try:
            result = self.xt_trader.cancel_order_stock(account, int(order_id))
            return result == 0
        except Exception as e:
            self.logger.error(f"撤单失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False


    def query_stock_asset(self, account):
        """
        查询账户资金

        Args:
            account: 账户对象

        Returns:
            资金对象
        """
        if not self.connected:
            return None

        try:
            return self.xt_trader.query_stock_asset(account)
        except Exception as e:
            self.logger.error(f"查询资金失败: {e}")
            return None

    def query_stock_positions(self, account):
        """
        查询持仓

        Args:
            account: 账户对象

        Returns:
            持仓列表
        """
        if not self.connected:
            return None

        try:
            return self.xt_trader.query_stock_positions(account)
        except Exception as e:
            self.logger.error(f"查询持仓失败: {e}")
            return None

    def query_stock_orders(self, account):
        """
        查询订单

        Args:
            account: 账户对象

        Returns:
            订单列表
        """
        if not self.connected:
            return None

        try:
            return self.xt_trader.query_stock_orders(account)
        except Exception as e:
            self.logger.error(f"查询订单失败: {e}")
            return None


class TradingScheduler:
    """交易调度器"""

    def __init__(self, trader: XtQuantTrader):
        self.logger = logging.getLogger(__name__)
        self.trader = trader
        self.trading_dates = []
        self._load_trading_calendar()

    def _load_trading_calendar(self):
        """加载交易日历"""
        try:
            from xtquant import xtdata
            from datetime import datetime, timedelta

            # 1. 下载节假日数据（确保能获取未来交易日）
            # self.logger.info("下载节假日数据...")
            # try:
            #     xtdata.download_holiday_data()
            #     self.logger.info("✅ 节假日数据下载成功")
            # except Exception as e:
            #     self.logger.warning(f"节假日数据下载失败（可忽略）: {e}")

            # 2. 获取交易日历（包含未来日期）
            # 计算时间范围：过去90天到未来30天
            end_date = datetime.now() + timedelta(days=30)
            end_time_str = end_date.strftime('%Y%m%d')
            trading_dates = self._load_trading_calendar_from_tushare(end_date)

            # self.logger.info(f"获取交易日历 (结束时间: {end_time_str})...")
            # trading_dates = xtdata.get_trading_calendar(
            #     market='SH',
            #     end_time=end_time_str  # 指定未来时间，获取包含今天的交易日历
            # )

            self.logger.info(f"从xtdata获取到 {len(trading_dates) if trading_dates is not None else 0} 个原始交易日")

            # # 如果迅投API失败，尝试使用tushare
            # if trading_dates is None or len(trading_dates) == 0:
            #     self.logger.warning("迅投API获取交易日历失败，尝试使用tushare...")
            #     trading_dates = self._load_trading_calendar_from_tushare(end_date)

            # 转换为date对象（处理多种数据格式）
            self.trading_dates = []
            if trading_dates is None or len(trading_dates) == 0:
                self.logger.warning("未获取到交易日历数据")
                return

            for d in trading_dates:
                try:
                    # 如果是整数，判断长度
                    if isinstance(d, int):
                        date_str = str(d)
                        # 13位时间戳（毫秒级Unix时间戳）
                        if len(date_str) == 13:
                            timestamp = d / 1000  # 转换为秒
                            date_obj = datetime.fromtimestamp(timestamp).date()
                            self.trading_dates.append(date_obj)
                        # 8位数字（YYYYMMDD）
                        elif len(date_str) == 8:
                            date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                            self.trading_dates.append(date_obj)
                        else:
                            self.logger.warning(f"日期格式不正确（长度={len(date_str)}）: {d}")
                    # 如果是字符串，直接解析
                    elif isinstance(d, str):
                        # 尝试不同的格式
                        try:
                            date_obj = datetime.strptime(d, '%Y%m%d').date()
                        except ValueError:
                            # 如果包含时间戳，取前8位
                            date_obj = datetime.strptime(d[:8], '%Y%m%d').date()
                        self.trading_dates.append(date_obj)
                    # 如果是numpy.int64等类型
                    elif hasattr(d, 'item'):
                        date_int = d.item()
                        date_str = str(date_int)
                        # 13位时间戳（毫秒级Unix时间戳）
                        if len(date_str) == 13:
                            timestamp = date_int / 1000  # 转换为秒
                            date_obj = datetime.fromtimestamp(timestamp).date()
                            self.trading_dates.append(date_obj)
                        # 8位数字（YYYYMMDD）
                        elif len(date_str) == 8:
                            date_obj = datetime.strptime(date_str, '%Y%m%d').date()
                            self.trading_dates.append(date_obj)
                        else:
                            self.logger.warning(f"numpy日期格式不正确（长度={len(date_str)}）: {d}")
                    else:
                        self.logger.warning(f"未知的日期类型 {type(d)}: {d}")
                except Exception as e:
                    self.logger.warning(f"无法解析日期 {d} (类型={type(d).__name__}): {e}")
                    continue

            self.logger.info(f"加载交易日历: {len(self.trading_dates)}个交易日")

            if len(self.trading_dates) > 0:
                self.logger.info(f"  最早日期: {self.trading_dates[0]}")
                self.logger.info(f"  最晚日期: {self.trading_dates[-1]}")

            # 同步交易日历到 trader 对象（供 order_manager 使用）
            if self.trader:
                self.trader.trading_dates = self.trading_dates
                self.logger.info(f"✅ 交易日历已同步到 trader 对象")

        except Exception as e:
            self.logger.error(f"加载交易日历失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            self.trading_dates = []

    def _load_trading_calendar_from_tushare(self, end_date):
        """
        从tushare获取交易日历（备选方案）

        Args:
            end_date: 结束日期(datetime对象)

        Returns:
            list: 交易日期字符串列表 (YYYYMMDD格式)
        """
        try:
            import tushare as ts
            from datetime import datetime, timedelta

            # Tushare token
            TS_TOKEN = "022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000"

            self.logger.info("正在从tushare获取交易日历...")

            # 初始化tushare pro API
            pro = ts.pro_api(TS_TOKEN)

            # 计算开始日期（过去90天）
            start_date = datetime.now() - timedelta(days=90)
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')

            self.logger.info(f"  日期范围: {start_date_str} ~ {end_date_str}")

            # 获取交易日历（SSE=上交所）
            df = pro.trade_cal(
                exchange='SSE',
                start_date=start_date_str,
                end_date=end_date_str
            )

            if df is None or df.empty:
                self.logger.error("tushare返回空数据")
                return None

            # 筛选交易日（is_open=1）
            df_trade = df[df['is_open'] == 1]

            # 提取日期字符串
            trading_dates = df_trade['cal_date'].tolist()

            # 排序：tushare返回的是倒序，需要转为正序（从早到晚）
            trading_dates = sorted(trading_dates)

            self.logger.info(f"✅ 从tushare获取到 {len(trading_dates)} 个交易日")
            self.logger.info(f"  最早日期: {trading_dates[0] if trading_dates else 'N/A'}")
            self.logger.info(f"  最晚日期: {trading_dates[-1] if trading_dates else 'N/A'}")

            return trading_dates

        except ImportError:
            self.logger.error("未安装tushare库，请运行: pip install tushare")
            return None
        except Exception as e:
            self.logger.error(f"从tushare获取交易日历失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return None

    def wait_for_market_open(self, target_time: str = None):
        """
        等待开盘时间

        Args:
            target_time: 目标时间(HH:MM:SS),默认为早盘开盘时间
        """
        if target_time is None:
            target_time = MarketTimeConfig.MORNING_OPEN

        now = datetime.now()
        target = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {target_time}", '%Y-%m-%d %H:%M:%S')

        if now >= target:
            self.logger.info(f"已过目标时间{target_time},立即执行")
            return

        wait_seconds = (target - now).total_seconds()
        self.logger.info(f"等待开盘: 目标时间{target_time}, 还需等待{wait_seconds:.0f}秒")

        # 每分钟打印一次倒计时
        while datetime.now() < target:
            remaining = (target - datetime.now()).total_seconds()
            if remaining > 60:
                self.logger.info(f"距离开盘还有{remaining/60:.1f}分钟...")
                time.sleep(60)
            else:
                time.sleep(remaining)
                break

        self.logger.info("✅ 已到达目标时间")

    def is_trading_time(self) -> bool:
        """
        判断当前是否为交易时间

        Returns:
            bool: 是否为交易时间
        """
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')

        # 早盘
        if MarketTimeConfig.MORNING_OPEN <= current_time <= MarketTimeConfig.MORNING_CLOSE:
            return True

        # 午盘
        if MarketTimeConfig.AFTERNOON_OPEN <= current_time <= MarketTimeConfig.AFTERNOON_CLOSE:
            return True

        return False

    def execute_buy_orders(self, stocks: List[Dict], total_amount: float, num_stocks: int) -> Dict[str, str]:
        """
        执行买入订单(使用智能下单)

        Args:
            stocks: 股票列表
            total_amount: 总金额
            num_stocks: 实际买入数量

        Returns:
            Dict: {stock_code: order_id}
        """
        self.logger.info(f"开始执行买入: 总金额{total_amount}元, 股票数{num_stocks}")

        # 计算每只股票金额
        amount_per_stock = total_amount / num_stocks
        self.logger.info(f"每只股票金额: {amount_per_stock:.2f}元")

        # 获取开盘参考价(用于判断偏离)
        reference_prices = {}
        for stock in stocks[:num_stocks]:
            stock_code = stock['code']
            price = self.trader._get_latest_price(stock_code)
            if price:
                reference_prices[stock_code] = price

        # 执行买入(使用智能下单)
        order_ids = {}
        for i, stock in enumerate(stocks[:num_stocks]):
            stock_code = stock['code']
            self.logger.info(f"\n买入第{i+1}只: {stock_code}")

            # 获取参考价
            reference_price = reference_prices.get(stock_code)

            # 使用智能下单(带撤单改单)
            success = self.trader.order_manager.smart_buy_with_retry(
                stock_code,
                amount_per_stock,
                reference_price
            )

            if success:
                # 获取最后一个订单ID
                last_order_id = list(self.trader.order_manager.orders.keys())[-1] if self.trader.order_manager.orders else None
                if last_order_id:
                    order_ids[stock_code] = last_order_id
            else:
                self.logger.warning(f"{stock_code}买入失败")

        self.logger.info(f"\n买入完成: 成功{len(order_ids)}/{num_stocks}")
        return order_ids

    def execute_sell_orders(self, positions: Dict[str, Dict], min_prices: Dict[str, float] = None) -> Dict[str, str]:
        """
        执行卖出订单(使用智能下单)

        Args:
            positions: 持仓字典
            min_prices: {stock_code: min_price} 最低卖出价(可选)

        Returns:
            Dict: {stock_code: order_id}
        """
        self.logger.info(f"开始执行卖出: 持仓{len(positions)}只")

        if min_prices is None:
            min_prices = {}

        order_ids = {}
        for stock_code, pos in positions.items():
            # 只卖出可用数量
            volume = pos['can_use_volume']

            if volume < 100:
                self.logger.warning(f"{stock_code}可用数量不足1手,跳过")
                continue

            min_price = min_prices.get(stock_code)

            self.logger.info(f"\n卖出: {stock_code}, {volume}股" +
                           (f", 最低价{min_price:.2f}" if min_price else ""))

            # 使用智能下单(带撤单改单)
            success = self.trader.order_manager.smart_sell_with_retry(stock_code, volume, min_price)

            if success:
                # 获取最后一个订单ID
                last_order_id = list(self.trader.order_manager.orders.keys())[-1] if self.trader.order_manager.orders else None
                if last_order_id:
                    order_ids[stock_code] = last_order_id
            else:
                self.logger.warning(f"{stock_code}卖出失败")

        self.logger.info(f"\n卖出完成: 成功{len(order_ids)}/{len(positions)}")
        return order_ids

    def save_trade_record(self, date: str, action: str, stocks: List[str], order_ids: Dict[str, str],
                         entry_date: str = None, entry_price: float = None, volume: int = None,
                         exit_reason: str = None, profit_pct: float = None):
        """
        保存交易记录（增强版）

        Args:
            date: 日期
            action: 操作(buy/sell)
            stocks: 股票列表
            order_ids: 订单ID字典
            entry_date: 实际买入日期 (买入时记录)
            entry_price: 买入价格 (买入时记录)
            volume: 买入数量 (买入时记录)
            exit_reason: 卖出原因 (卖出时记录)
            profit_pct: 收益率 (卖出时记录)
        """
        try:
            # 读取现有记录
            if PathConfig.TRADE_RECORDS_FILE.exists():
                with open(PathConfig.TRADE_RECORDS_FILE, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            else:
                records = {}

            # 添加记录
            if date not in records:
                records[date] = []

            record = {
                'action': action,
                'stocks': stocks,
                'order_ids': order_ids,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # 买入记录的额外字段
            if action == 'buy' and entry_date and entry_price and volume:
                record['entry_date'] = entry_date
                record['entry_price'] = entry_price
                record['volume'] = volume

            # 卖出记录的额外字段
            if action == 'sell' and exit_reason:
                record['exit_reason'] = exit_reason
                if profit_pct is not None:
                    record['profit_pct'] = profit_pct

            records[date].append(record)

            # 保存
            with open(PathConfig.TRADE_RECORDS_FILE, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            self.logger.info(f"交易记录已保存: {PathConfig.TRADE_RECORDS_FILE}")

        except Exception as e:
            self.logger.error(f"保存交易记录失败: {e}")

    def calculate_holding_days(self, entry_date_str: str) -> int:
        """
        计算持仓天数（基于交易日历）

        Args:
            entry_date_str: 买入日期字符串 (YYYY-MM-DD)

        Returns:
            int: 持仓天数（0表示买入当天，1表示买入后第1个交易日，以此类推）
        """
        try:
            if not self.trading_dates:
                self.logger.warning("交易日历未加载，无法计算持仓天数")
                return 0

            from datetime import datetime
            entry_date = datetime.strptime(entry_date_str, '%Y-%m-%d').date()
            today = datetime.now().date()

            # 在交易日历中查找索引
            if entry_date not in self.trading_dates:
                self.logger.warning(f"买入日期{entry_date}不在交易日历中")
                # 使用自然日计算
                return (today - entry_date).days

            if today not in self.trading_dates:
                self.logger.warning(f"今天{today}不在交易日历中")
                # 使用自然日计算
                return (today - entry_date).days

            entry_index = self.trading_dates.index(entry_date)
            today_index = self.trading_dates.index(today)

            # 持仓天数 = 今天索引 - 买入索引
            holding_days = today_index - entry_index

            return holding_days

        except Exception as e:
            self.logger.error(f"计算持仓天数失败: {e}")
            return 0

    def load_latest_buy_record(self, stock_code: str) -> Optional[Dict]:
        """
        加载指定股票的最新买入记录

        优先从本地文件加载，找不到则从API查询当日成交记录并保存

        Args:
            stock_code: 股票代码

        Returns:
            Dict: 买入记录，包含entry_date, entry_price, volume等字段；找不到返回None
        """
        try:
            # 1. 尝试从本地文件加载
            if PathConfig.TRADE_RECORDS_FILE.exists():
                with open(PathConfig.TRADE_RECORDS_FILE, 'r', encoding='utf-8') as f:
                    records = json.load(f)

                # 按日期排序（从新到旧）
                sorted_dates = sorted(records.keys(), reverse=True)

                # 找到最新的买入记录
                for date in sorted_dates:
                    for record in records[date]:
                        if (record['action'] == 'buy' and
                            stock_code in record['stocks'] and
                            'entry_date' in record):

                            self.logger.info(f"✅ 从本地找到{stock_code}的买入记录: 日期={record['entry_date']}, "
                                           f"价格={record.get('entry_price', 'N/A')}, "
                                           f"数量={record.get('volume', 'N/A')}")
                            return record

            # 2. 本地找不到，尝试从API查询当日成交记录
            self.logger.warning(f"⚠️ 本地未找到{stock_code}的买入记录，尝试从API查询当日成交...")

            # 获取当日成交记录
            from xtquant import xtconstant
            trades = self.trader.xt_trader.query_stock_trades(self.trader.account)

            if not trades:
                self.logger.error(f"❌ 无法获取当日成交记录")
                return None

            # 筛选该股票的所有买入记录（支持多次买入）
            buy_trades = []
            for trade in trades:
                if (trade.stock_code == stock_code and
                    trade.order_type == xtconstant.STOCK_BUY):  # 23 = 买入
                    buy_trades.append(trade)

            if not buy_trades:
                self.logger.error(f"❌ 当日成交记录中也没有找到{stock_code}的买入记录")
                return None

            # 如果有多笔买入，汇总计算
            if len(buy_trades) > 1:
                self.logger.info(f"📊 发现{stock_code}有{len(buy_trades)}笔买入记录，进行汇总...")

                # 计算总数量、总金额、加权平均价格
                total_volume = 0
                total_amount = 0
                order_ids = {}

                for trade in buy_trades:
                    total_volume += trade.traded_volume
                    total_amount += trade.traded_amount
                    order_ids[stock_code] = trade.order_id
                    self.logger.info(f"  笔{len(order_ids)}: 价格={trade.traded_price:.2f}, "
                                   f"数量={trade.traded_volume}, "
                                   f"金额={trade.traded_amount:.2f}")

                # 计算加权平均价格
                weighted_avg_price = total_amount / total_volume if total_volume > 0 else 0

                self.logger.info(f"✅ 汇总结果: 总数量={total_volume}股, "
                               f"加权均价={weighted_avg_price:.2f}元, "
                               f"总金额={total_amount:.2f}元")

                # 构建汇总的买入记录
                today_str = datetime.now().strftime('%Y-%m-%d')
                buy_record = {
                    'action': 'buy',
                    'stocks': [stock_code],
                    'order_ids': order_ids,
                    'entry_date': today_str,
                    'entry_price': weighted_avg_price,
                    'volume': total_volume,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # 保存到本地
                self.save_trade_record(
                    date=datetime.now().strftime('%Y%m%d'),
                    action='buy',
                    stocks=[stock_code],
                    order_ids=order_ids,
                    entry_date=today_str,
                    entry_price=weighted_avg_price,
                    volume=total_volume
                )

                self.logger.info(f"✅ 从API获取并保存{stock_code}的汇总买入记录: "
                               f"日期={today_str}, "
                               f"价格={weighted_avg_price:.2f}, "
                               f"数量={total_volume}")

                return buy_record

            else:
                # 只有1笔买入，直接使用
                trade = buy_trades[0]
                today_str = datetime.now().strftime('%Y-%m-%d')
                buy_record = {
                    'action': 'buy',
                    'stocks': [stock_code],
                    'order_ids': {stock_code: trade.order_id},
                    'entry_date': today_str,
                    'entry_price': trade.traded_price,
                    'volume': trade.traded_volume,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # 保存到本地
                self.save_trade_record(
                    date=datetime.now().strftime('%Y%m%d'),
                    action='buy',
                    stocks=[stock_code],
                    order_ids={stock_code: trade.order_id},
                    entry_date=today_str,
                    entry_price=trade.traded_price,
                    volume=trade.traded_volume
                )

                self.logger.info(f"✅ 从API获取并保存{stock_code}的买入记录: "
                               f"日期={today_str}, "
                               f"价格={trade.traded_price:.2f}, "
                               f"数量={trade.traded_volume}")
                return buy_record

        except Exception as e:
            self.logger.error(f"加载买入记录失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def get_current_price(self, stock_code: str) -> Optional[float]:
        """
        获取最新价

        Args:
            stock_code: 股票代码

        Returns:
            float: 最新价
        """
        return self.trader.get_current_price(stock_code)

    def query_stock_orders(self, account):
        """
        查询订单

        Args:
            account: 账户对象

        Returns:
            订单列表
        """
        if not self.trader.connected:
            return []

        try:
            return self.trader.xt_trader.query_stock_orders(account)
        except Exception as e:
            self.logger.error(f"查询订单失败: {e}")
            return []

    def order_stock_async(self, account, stock_code: str, order_type: int,
                         order_volume: int, price_type: int, price: float,
                         strategy_name: str, order_remark: str):
        """
        异步下单

        Args:
            account: 账户对象
            stock_code: 股票代码
            order_type: 订单类型（买/卖）
            order_volume: 委托数量
            price_type: 价格类型
            price: 价格
            strategy_name: 策略名称
            order_remark: 订单备注

        Returns:
            int: 订单号
        """
        if not self.trader.connected:
            return -1

        try:
            return self.trader.xt_trader.order_stock(
                account,
                stock_code,
                order_type,
                order_volume,
                price_type,
                price,
                strategy_name,
                order_remark
            )
        except Exception as e:
            self.logger.error(f"异步下单失败: {e}")
            return -1

    def cancel_order_stock(self, account, order_id: str) -> bool:
        """
        撤单

        Args:
            account: 账户对象
            order_id: 订单号

        Returns:
            bool: 是否成功
        """
        if not self.trader.connected:
            return False

        try:
            result = self.trader.xt_trader.cancel_order_stock(account, int(order_id))
            return result == 0
        except Exception as e:
            self.logger.error(f"撤单失败: {e}")
            return False

