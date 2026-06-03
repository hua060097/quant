"""
智能订单管理器（全仓买入top1股票）

功能:
1. 全仓买入top1股票（不留缓冲）
2. 订单状态监控（通过回调）
3. 撤单重试机制
4. 全仓卖出
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, TYPE_CHECKING
import pandas as pd

# 类型检查导入（仅在类型检查时生效，避免运行时导入错误）
if TYPE_CHECKING:
    from xtquant.xttype import XtOrder, XtTrade, XtOrderError

try:
    from xtquant.xttrader import XtQuantTraderCallback
    from xtquant import xtconstant
except ImportError:
    # 如果迅投库未安装，定义占位符
    XtQuantTraderCallback = object

    class xtconstant:
        STOCK_BUY = 23#49
        STOCK_SELL = 24#50
        LATEST_PRICE = 5#40  # 最新价
        FIX_PRICE = 11#42  # 固定价格


class LiveTradingCallback(XtQuantTraderCallback):
    """
    实盘交易回调类

    处理订单状态更新、成交回报、订单错误等事件
    """

    def __init__(self, order_manager):
        self.order_manager = order_manager
        self.logger = logging.getLogger(__name__)

    def on_stock_order(self, order: 'XtOrder'):
        """
        订单状态更新回调

        Args:
            order: 订单对象
        """
        order_id = str(order.order_id)
        status = order.order_status
        stock_code = order.stock_code

        self.logger.info(f"[订单回调] {stock_code} 订单{order_id} 状态: {status}")

        # 更新订单追踪
        if order_id in self.order_manager.active_orders:
            self.order_manager.active_orders[order_id]['order_status'] = status
            self.order_manager.active_orders[order_id]['update_time'] = datetime.now().strftime('%H:%M:%S')
            self.order_manager.active_orders[order_id]['traded_volume'] = order.traded_volume

        # 状态处理
        if status in [53, 56]:  # 已成（53: 旧状态码, 56: 新状态码）
            self.logger.info(f"✅ 订单已成交: {order_id} (状态{status})")
            self.logger.info(f"  成交量: {order.traded_volume}")
            if order.traded_price > 0:
                self.logger.info(f"  成均价: {order.traded_price:.2f}")

            # 标记订单已完成
            if order_id in self.order_manager.active_orders:
                self.order_manager.active_orders[order_id]['completed'] = True
                # 记录成交价和数量（供 main.py 使用）
                self.order_manager.active_orders[order_id]['price'] = float(order.traded_price) if order.traded_price > 0 else 0
                self.order_manager.active_orders[order_id]['volume'] = int(order.traded_volume)

        elif status == 52:  # 部成
            self.logger.warning(f"⚠️ 订单部分成交: {order_id}")
            self.logger.warning(f"  已成交: {order.traded_volume}/{order.total_volume}")

            # 判断是否需要撤单重试
            if order.traded_volume == 0:
                # 部成但成交量为0，可能是价格问题
                self.logger.warning("  部成但成交为0，需要撤单重试")
                if order_id in self.order_manager.active_orders:
                    self.order_manager.active_orders[order_id]['need_cancel'] = True

        elif status == 57:  # 废单
            self.logger.error(f"❌ 订单被拒绝（废单）: {order_id}")
            self.logger.error(f"  可能原因: 价格不合法、数量不合法、账户问题等")

            # 标记需要重试
            if order_id in self.order_manager.active_orders:
                self.order_manager.active_orders[order_id]['need_retry'] = True

        elif status == 54:  # 已撤
            self.logger.info(f"🔄 订单已撤销: {order_id}")

    def on_stock_trade(self, trade: 'XtTrade'):
        """
        成交回报回调

        Args:
            trade: 成交对象
        """
        self.logger.info(f"[成交回报] 订单{trade.order_id}")
        self.logger.info(f"  成交号: {trade.traded_id}")
        self.logger.info(f"  价格: {trade.traded_price:.2f}")
        self.logger.info(f"  数量: {trade.traded_volume}")
        self.logger.info(f"  时间: {trade.traded_time}")

        # 保存成交记录
        self.order_manager._save_trade_record(trade)

    def on_order_error(self, order_error: 'XtOrderError'):
        """
        订单错误回调

        Args:
            order_error: 错误信息
        """
        self.logger.error(f"[订单错误] 订单{order_error.order_id}")
        self.logger.error(f"  错误代码: {order_error.error_id}")
        self.logger.error(f"  错误信息: {order_error.error_msg}")

        # 标记需要重试
        order_id = str(order_error.order_id)
        if order_id in self.order_manager.active_orders:
            self.order_manager.active_orders[order_id]['need_retry'] = True
            self.order_manager.active_orders[order_id]['error_msg'] = order_error.error_msg


class SmartOrderManager:
    """
    智能订单管理器（全仓买入版本）

    功能:
    1. 全仓买入top1股票（不留缓冲）
    2. 订单状态监控
    3. 撤单重试机制
    4. 全仓卖出
    """

    def __init__(self, trader):
        self.trader = trader
        self.logger = logging.getLogger(__name__)

        # 订单追踪: {order_id: order_info}
        self.active_orders = {}

        # 兼容属性：orders 指向 active_orders（供 main.py 使用）
        self.orders = self.active_orders

        # 重试配置
        self.MAX_RETRY_TIMES = 3  # 最大重试次数
        self.CANCEL_WAIT_SECONDS = 3  # 撤单等待时间（秒）
        self.ORDER_TIMEOUT_SECONDS = 10  # 订单超时时间（秒）

        # 价格缓存
        self.limit_price_cache = {}  # 涨停价缓存 {trade_date: {stock_code: up_limit}}
        self.ask_price_cache = {}  # 对手价缓存 {stock_code: ask_price}

    def smart_buy_full_position(self, stock_code: str,
                                buy_date: str,
                                target_volume: int = None) -> Optional[str]:
        """
        全仓买入top1股票

        流程:
        1. 等待到09:30（仅trade模式）
        2. 获取最新价（开盘价）
        3. 查询账户资金
        4. 计算买入数量（全仓或指定数量）
        5. 执行开盘风控检查（仅trade模式）
        6. 立即下单买入
        7. 监控订单状态
        8. 部成/超时 → 撤单重试
        9. 全部成交 → 返回订单号

        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            target_volume: 指定买入数量（可选，用于手动交易模式）

        Returns:
            str: 订单号（失败返回None）
        """
        self.logger.info("=" * 60)
        if target_volume:
            self.logger.info(f"开始指定数量买入: {stock_code}, {target_volume}股")
        else:
            self.logger.info(f"开始全仓买入: {stock_code}")
        self.logger.info("=" * 60)

        # 1. 等待到09:30（仅限trade模式，手动模式不等待）
        if target_volume is None:
            self._wait_until_market_open()

        # 2. 获取价格（区分买入当日和manual模式）
        if target_volume is None:
            # ===== trade模式（买入当日）：获取真实开盘价 =====
            self.logger.info(f"🎯 {stock_code}: 买入当日，获取真实开盘价进行风控")

            today_open = self._get_today_open_price_from_tick(stock_code)

            if today_open is None:
                self.logger.error(f"❌ {stock_code}: 无法获取开盘价，放弃买入")
                return None

            current_price = today_open
            self.logger.info(f"📊 {stock_code}: 今日开盘价 = {today_open:.2f}")
        else:
            # ===== manual模式：获取最新价 =====
            self.logger.info(f"📝 {stock_code}: manual模式，使用最新价")
            current_price = self.trader.get_current_price(stock_code)

            if not current_price:
                self.logger.error(f"无法获取{stock_code}的最新价")
                return None

        self.logger.info(f"当前价格: {current_price:.2f}元")

        # 3. 查询账户资金（仅trade模式需要）
        if target_volume is None:
            asset = self.trader.query_stock_asset(self.trader.account)

            if not asset:
                self.logger.error("查询账户资金失败")
                return None

            available_cash = asset.cash
            self.logger.info(f"可用资金: {available_cash:.2f}元")

            # 4. 计算全仓买入数量（不留缓冲）
            buy_volume = self._calculate_buy_volume_full(available_cash, current_price)
        else:
            # 手动模式：使用指定的数量
            buy_volume = target_volume

        if buy_volume < 100:
            self.logger.error(
                f"资金不足: 可买{buy_volume}股，不足1手（100股）"
            )
            return None

        estimated_amount = buy_volume * current_price

        self.logger.info(f"买入数量: {buy_volume}股")
        self.logger.info(f"预计金额: {estimated_amount:.2f}元")

        # 仅trade模式显示剩余资金
        if target_volume is None:
            remaining_cash = available_cash - estimated_amount
            self.logger.info(f"剩余资金: {remaining_cash:.2f}元")

        # 5. 执行开盘风控检查（仅trade模式）
        if target_volume is None:
            prev_close = self._get_prev_close(stock_code, buy_date)
            risk_check = self._check_opening_gap_risk(
                stock_code, buy_date, prev_close, current_price
            )

            if not risk_check['can_buy']:
                self.logger.warning(
                    f"未通过开盘风控 ({risk_check['reason']})，取消买入"
                )
                return None

            self.logger.info("✅ 通过风控检查，开始计算下单价格...")
        else:
            self.logger.info("✅ 手动模式，跳过风控检查，开始下单...")

        # 计算初始下单价格（基于对手价）
        if target_volume is None:
            # trade模式：使用对手价 * 1.002
            initial_order_price = self._calculate_order_price(stock_code, buy_date, prev_close, current_price)
            if not initial_order_price:
                self.logger.error(f"无法计算下单价格，放弃买入")
                return None
        else:
            # manual模式：使用最新价
            initial_order_price = current_price

        self.logger.info(f"初始下单价格: {initial_order_price:.2f}")

        # 智能重试循环（提价策略）
        for retry in range(1, 4):  # 最多3次
            # 计算当前重试的价格（每次提价0.1%）
            current_order_price = initial_order_price * (1.001 ** (retry - 1))

            self.logger.info(f"[第{retry}次下单] 价格: {current_order_price:.2f}, 数量: {buy_volume}股")

            # 下单
            order_id = self._place_buy_order(stock_code, buy_volume, current_order_price)

            if not order_id:
                self.logger.error("下单失败")
                continue

            # 等待成交（2秒）
            filled = self._wait_for_order_fill(order_id, timeout=2)

            if filled:
                self.logger.info(f"✅ 第{retry}次成交成功: 订单{order_id}")
                return order_id

            # 未成交
            if retry < 3:
                self.logger.warning(f"第{retry}次未成交，撤单重试...")
                self._cancel_order_with_retry(order_id)
            else:
                self.logger.error(f"❌ 3次重试后仍未成交，放弃买入")
                return None

        return None

    def smart_sell_full_position(self, stock_code: str,
                                 volume: int) -> Optional[str]:
        """
        全仓卖出

        Args:
            stock_code: 股票代码
            volume: 卖出数量

        Returns:
            str: 订单号（失败返回None）
        """
        self.logger.info("=" * 60)
        self.logger.info(f"开始全仓卖出: {stock_code}, {volume}股")
        self.logger.info("=" * 60)

        # 获取最新价
        current_price = self.trader.get_current_price(stock_code)

        if not current_price:
            self.logger.error(f"无法获取{stock_code}的最新价")
            return None

        self.logger.info(f"当前价格: {current_price:.2f}元")
        self.logger.info(f"卖出金额: {volume * current_price:.2f}元")

        # 下单卖出
        for retry in range(1, self.MAX_RETRY_TIMES + 1):
            self.logger.info(f"[卖出重试 {retry}/{self.MAX_RETRY_TIMES}]")

            order_id = self._place_sell_order(stock_code, volume)

            if not order_id:
                self.logger.error("下单失败")
                continue

            # 等待成交
            filled = self._wait_for_order_fill(order_id, timeout=self.ORDER_TIMEOUT_SECONDS)

            if filled:
                self.logger.info(f"✅ 全仓卖出成功: 订单{order_id}")
                return order_id

            # 未成交，撤单重试
            if retry < self.MAX_RETRY_TIMES:
                self.logger.warning("订单未成交，撤单重试...")
                self._cancel_order_with_retry(order_id)
            else:
                self.logger.error(f"❌ 卖出失败: 已重试{self.MAX_RETRY_TIMES}次")
                return None

        return None

    def _calculate_buy_volume_full(self, cash: float, price: float) -> int:
        """
        全仓买入计算（不留缓冲）

        Args:
            cash: 可用资金
            price: 参考价格

        Returns:
            int: 买入数量（100的整数倍）
        """
        volume = int(cash / price / 100) * 100
        return volume

    def _get_ask_price_from_xtdata(self, stock_code: str) -> Optional[float]:
        """
        从xtdata获取对手价（买一价 bidPrice[0]）

        Args:
            stock_code: 股票代码（如 '000001.SZ'）

        Returns:
            float: 对手价，失败返回None
        """
        # 检查缓存
        if stock_code in self.ask_price_cache:
            cached_price = self.ask_price_cache[stock_code]
            # 缓存有效期：5秒
            if (datetime.now() - cached_price['timestamp']).total_seconds() < 5:
                return cached_price['price']

        try:
            from xtquant import xtdata

            # 获取全推tick数据
            tick = xtdata.get_full_tick([stock_code])

            if stock_code not in tick:
                self.logger.error(f"无法获取 {stock_code} 的tick数据")
                return None

            stock_tick = tick[stock_code]

            # 取买一价作为对手价
            bid_price = stock_tick.get('bidPrice')

            if bid_price and len(bid_price) > 0:
                ask_price = bid_price[0]  # 买一价

                # 如果买一价为0（跌停），使用最新价
                if ask_price == 0 or ask_price is None:
                    ask_price = stock_tick.get('lastPrice', 0)

                if ask_price and ask_price > 0:
                    ask_price_float = float(ask_price)

                    # 缓存价格
                    self.ask_price_cache[stock_code] = {
                        'price': ask_price_float,
                        'timestamp': datetime.now()
                    }

                    self.logger.info(f"✅ {stock_code}: 对手价={ask_price_float:.2f}")
                    return ask_price_float

            self.logger.error(f"{stock_code}: 无法从tick数据获取有效对手价")
            return None

        except Exception as e:
            self.logger.error(f"获取对手价失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calculate_order_price(self, stock_code: str, buy_date: str,
                             prev_close: float, last_price: float) -> Optional[float]:
        """
        计算下单价格（基于对手价和涨停价）

        逻辑：
        1. 获取对手价（askPrice = bidPrice[0]）
        2. 计算涨停价（up_limit = prev_close * 1.05）
        3. 计算下单价 = min(askPrice * 1.002, up_limit)

        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            prev_close: 昨日收盘价
            last_price: 最新价（容错备用）

        Returns:
            float: 下单价格，失败返回None
        """
        # 1. 计算涨停价（昨日收盘价 * 1.05）
        up_limit = prev_close * 1.05

        self.logger.info(f"{stock_code}: 昨收={prev_close:.2f}, 涨停价={up_limit:.2f}")

        # 2. 获取对手价
        ask_price = self._get_ask_price_from_xtdata(stock_code)

        if ask_price is None:
            # 容错：如果无法获取对手价，使用最新价
            self.logger.warning(f"{stock_code}: 无法获取对手价，使用最新价 {last_price:.2f}")
            return min(last_price, up_limit)

        # 3. 计算下单价：min(对手价 * 1.002, 涨停价)
        order_price = min(ask_price * 1.002, up_limit)

        self.logger.info(f"{stock_code}: 对手价={ask_price:.2f}, 计算价={ask_price * 1.002:.2f}, 下单价={order_price:.2f}")

        return float(order_price)
    

    def _get_top_of_book(self, stock_code: str):
        """
        获取最简盘口：bid1 / ask1 / last / open
        普通用户可用：xtdata.get_full_tick 通常就有这些字段。
        """
        from xtquant import xtdata
        tick = xtdata.get_full_tick([stock_code])
        if stock_code not in tick or not tick[stock_code]:
            return None

        t = tick[stock_code]

        bid1 = None
        ask1 = None

        bid_arr = t.get("bidPrice")
        ask_arr = t.get("askPrice")

        if isinstance(bid_arr, (list, tuple)) and len(bid_arr) > 0:
            bid1 = float(bid_arr[0]) if bid_arr[0] else None
        if isinstance(ask_arr, (list, tuple)) and len(ask_arr) > 0:
            ask1 = float(ask_arr[0]) if ask_arr[0] else None

        lastp = t.get("lastPrice", None)
        openp = t.get("open", None)

        lastp = float(lastp) if lastp else None
        openp = float(openp) if openp else None

        # 容错：盘口为0/None时
        if bid1 is not None and bid1 <= 0:
            bid1 = None
        if ask1 is not None and ask1 <= 0:
            ask1 = None

        return {
            "bid1": bid1,
            "ask1": ask1,
            "last": lastp,
            "open": openp
        }


    def _place_buy_order(self, stock_code: str, volume: int,
                        price: float) -> Optional[str]:
        """
        下单买入（使用固定价格）

        Args:
            stock_code: 股票代码
            volume: 买入数量
            price: 买入价格（固定价格）

        Returns:
            str: 订单号（失败返回None）
        """
        try:
            # 使用固定价格下单（FIX_PRICE）而非最新价
            order_id = self.trader.order_stock_async(
                account=self.trader.account,
                stock_code=stock_code,
                order_type=xtconstant.STOCK_BUY,
                order_volume=volume,
                price_type=xtconstant.FIX_PRICE,  # 改为固定价格
                price=price,
                strategy_name='live_trading_5m',
                order_remark=stock_code
            )

            if order_id and order_id > 0:
                self.logger.info(f"下单成功: {stock_code} 买入{volume}股, 订单号={order_id}")

                # 添加到追踪
                self.active_orders[str(order_id)] = {
                    'order_id': str(order_id),
                    'stock_code': stock_code,
                    'order_type': 'buy',
                    'order_volume': volume,
                    'traded_volume': 0,
                    'order_status': 48,
                    'create_time': datetime.now().strftime('%H:%M:%S'),
                    'update_time': datetime.now().strftime('%H:%M:%S'),
                    'completed': False,
                    'need_cancel': False,
                    'need_retry': False,
                    # 兼容字段（供 main.py 使用）
                    'price': 0,  # 成交价，成交时更新
                    'volume': 0,  # 成交量，成交时更新
                }

                return str(order_id)
            else:
                self.logger.error(f"下单失败: {stock_code}, order_id={order_id}")
                return None

        except Exception as e:
            self.logger.error(f"下单异常: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _place_sell_order(self, stock_code: str, volume: int) -> Optional[str]:
        """
        下单卖出

        Args:
            stock_code: 股票代码
            volume: 卖出数量

        Returns:
            str: 订单号（失败返回None）
        """
        try:
            # 使用最新价下单
            order_id = self.trader.order_stock_async(
                account=self.trader.account,
                stock_code=stock_code,
                order_type=xtconstant.STOCK_SELL,
                order_volume=volume,
                price_type=xtconstant.LATEST_PRICE,
                price=0,  # 卖出时price参数无效
                strategy_name='live_trading_5m',
                order_remark=stock_code
            )

            if order_id and order_id > 0:
                self.logger.info(f"下单成功: {stock_code} 卖出{volume}股, 订单号={order_id}")

                # 添加到追踪
                self.active_orders[str(order_id)] = {
                    'order_id': str(order_id),
                    'stock_code': stock_code,
                    'order_type': 'sell',
                    'order_volume': volume,
                    'traded_volume': 0,
                    'order_status': 48,
                    'create_time': datetime.now().strftime('%H:%M:%S'),
                    'update_time': datetime.now().strftime('%H:%M:%S'),
                    'completed': False,
                    'need_cancel': False,
                    'need_retry': False,
                    # 兼容字段（供 main.py 使用）
                    'price': 0,  # 成交价，成交时更新
                    'volume': 0,  # 成交量，成交时更新
                }

                return str(order_id)
            else:
                self.logger.error(f"下单失败: {stock_code}, order_id={order_id}")
                return None

        except Exception as e:
            self.logger.error(f"下单异常: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _wait_for_order_fill(self, order_id: str,
                            timeout: int = 10) -> bool:
        """
        等待订单成交

        Args:
            order_id: 订单号
            timeout: 超时时间（秒）

        Returns:
            bool: 是否成交
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # 检查订单追踪
            if order_id in self.active_orders:
                order_info = self.active_orders[order_id]

                # 已完成
                if order_info['completed']:
                    return True

                # 需要撤单
                if order_info['need_cancel']:
                    return False

                # 需要重试
                if order_info['need_retry']:
                    return False

            # 查询订单状态
            order_status = self._query_order_status(order_id)

            if order_status in [53, 56]:  # 已成（53: 旧状态码, 56: 新状态码）
                return True
            elif order_status in [54, 57]:  # 已撤、废单
                return False
            elif order_status == 52:  # 部成
                # 检查是否部分成交了至少50%
                order_info = self._get_order_info(order_id)
                if order_info:
                    filled_ratio = order_info['traded_volume'] / order_info['order_volume']
                    if filled_ratio >= 0.5:
                        self.logger.info(f"部分成交{filled_ratio:.1%}，视为成功")
                        return True

            # 休眠0.5秒后重试
            time.sleep(0.5)

        # 超时
        self.logger.warning(f"等待订单成交超时: {order_id}")
        return False

    def _query_order_status(self, order_id: str) -> Optional[int]:
        """
        查询订单状态

        Args:
            order_id: 订单号

        Returns:
            int: 订单状态（48-57，失败返回None）
        """
        try:
            orders = self.trader.query_stock_orders(self.trader.account)

            for order in orders:
                if str(order.order_id) == order_id:
                    return order.order_status

            return None

        except Exception as e:
            self.logger.error(f"查询订单状态失败: {e}")
            return None

    def _get_order_info(self, order_id: str) -> Optional[Dict]:
        """
        获取订单详细信息

        Args:
            order_id: 订单号

        Returns:
            Dict: 订单信息
        """
        try:
            orders = self.trader.query_stock_orders(self.trader.account)

            for order in orders:
                if str(order.order_id) == order_id:
                    return {
                        'order_id': str(order.order_id),
                        'stock_code': order.stock_code,
                        'order_status': order.order_status,
                        'order_volume': order.total_volume,
                        'traded_volume': order.traded_volume,
                        'traded_price': order.traded_price,
                        'order_price': order.order_price,
                    }

            return None

        except Exception as e:
            self.logger.error(f"获取订单信息失败: {e}")
            return None

    def _cancel_order_with_retry(self, order_id: str,
                                max_retry: int = 3) -> bool:
        """
        撤单（带重试）

        Args:
            order_id: 订单号
            max_retry: 最大重试次数

        Returns:
            bool: 是否成功
        """
        for retry in range(1, max_retry + 1):
            self.logger.info(f"[撤单重试 {retry}/{max_retry}] {order_id}")

            # 先检查订单状态
            order_status = self._query_order_status(order_id)

            if order_status in [53, 56]:  # 已成（53: 旧状态码, 56: 新状态码）
                self.logger.info("订单已成交，无需撤单")
                return True
            elif order_status in [54, 57]:  # 已撤、废单
                self.logger.info("订单已撤销或废单，无需撤单")
                return True

            # 执行撤单
            result = self.trader.cancel_order_stock(self.trader.account, order_id)

            if result:
                self.logger.info(f"撤单请求成功: {order_id}")

                # 等待撤单生效
                time.sleep(self.CANCEL_WAIT_SECONDS)

                # 检查撤单结果
                order_status = self._query_order_status(order_id)
                if order_status in [54, 55, 57]:  # 已撤、部撤、废单
                    return True

            # 撤单失败，重试
            if retry < max_retry:
                time.sleep(1)

        self.logger.error(f"撤单失败: {order_id}")
        return False

    def _wait_until_market_open(self):
        """等待到09:30开盘"""
        target_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        now = datetime.now()

        if now < target_time:
            wait_seconds = (target_time - now).total_seconds()
            self.logger.info(f"等待到09:30开盘，还需等待 {wait_seconds:.0f} 秒")
            time.sleep(wait_seconds)
        else:
            self.logger.info("已过09:30，立即执行")

    def _get_prev_close(self, stock_code: str, current_date: str) -> Optional[float]:
        """
        获取前一日收盘价（使用交易日历）

        Args:
            stock_code: 股票代码
            current_date: 当前日期 (YYYY-MM-DD)

        Returns:
            float: 前一日收盘价
        """
        try:
            from xtquant import xtdata
            from datetime import datetime

            # 1. 从交易日历中找到当前日期的前一个交易日
            # 获取交易日历（复用trader的交易日历）
            if not hasattr(self.trader, 'trading_dates') or not self.trader.trading_dates:
                self.logger.error("交易日历未加载，无法获取前一日收盘价")
                return None

            trading_dates = self.trader.trading_dates
            current_dt = datetime.strptime(current_date, '%Y-%m-%d').date()

            # 找到当前日期在交易日历中的位置
            if current_dt in trading_dates:
                current_index = trading_dates.index(current_dt)
                # 前一个交易日
                if current_index > 0:
                    prev_trading_day = trading_dates[current_index - 1]
                else:
                    self.logger.warning(f"{current_date}: 没有上一个交易日")
                    return None
            else:
                # 如果当前日期不在交易日历中（比如今天），找最近的上一个交易日
                # 找到第一个小于当前日期的交易日
                prev_trading_day = None
                for td in reversed(trading_dates):
                    if td < current_dt:
                        prev_trading_day = td
                        break

                if prev_trading_day is None:
                    self.logger.warning(f"{current_date}: 无法找到上一个交易日")
                    return None

            # 2. 获取上一个交易日的收盘价（使用精确日期）
            prev_date_str = prev_trading_day.strftime('%Y%m%d')

            # 获取指定日期的日线数据（前复权）
            daily_data = xtdata.get_market_data_ex(
                stock_list=[stock_code],
                period='1d',
                start_time=prev_date_str,
                end_time=prev_date_str,
                count=1,
                dividend_type="front"  # 前复权
            )

            if stock_code in daily_data and len(daily_data[stock_code]) > 0:
                prev_close = daily_data[stock_code].iloc[0]['close']
                self.logger.info(f"  上一个交易日: {prev_trading_day}, 收盘价: {prev_close:.2f}")
                return float(prev_close)
            else:
                self.logger.error(f"{stock_code}: {prev_date_str} 无数据，无法获取前一日收盘价")
                return None

        except Exception as e:
            self.logger.error(f"获取前一日收盘价失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _get_today_open_price_from_tick(self, stock_code: str) -> Optional[float]:
        """
        从全推数据中获取今天的开盘价

        适用场景：买入当日（9:35启动）需要获取开盘价进行开盘跳空风控

        Args:
            stock_code: 股票代码，如 '000001.SZ'

        Returns:
            float: 开盘价，获取失败返回 None
        """
        from xtquant import xtdata

        try:
            # 1. 获取全推tick数据（实时快照）
            tick_data = xtdata.get_full_tick([stock_code])

            # 2. 解析返回数据
            # 返回格式：{stock_code: {字段: 值}}
            # 参考：https://dict.thinktrader.net/nativeApi/xtdata.html
            if stock_code not in tick_data:
                self.logger.error(f"❌ {stock_code}: 全推数据中没有该股票")
                return None

            # 获取tick数据（已经是字典格式，直接包含所有字段）
            latest_tick = tick_data[stock_code]

            if not latest_tick:
                self.logger.error(f"❌ {stock_code}: 全推数据为空")
                return None

            # 3. 提取开盘价
            # tick数据字段包括：time, lastPrice, open, high, low, lastClose, amount, volume等
            open_price = latest_tick.get('open')

            if open_price is None or open_price <= 0:
                self.logger.warning(f"⚠️ {stock_code}: 开盘价字段无效（{open_price}）")
                return None

            self.logger.info(f"✅ {stock_code}: 从全推数据获取到开盘价 {open_price:.2f}")
            return float(open_price)

        except Exception as e:
            self.logger.error(f"❌ {stock_code}: 获取开盘价失败 - {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _check_opening_gap_risk(self, stock_code: str, buy_date: str,
                               prev_close: Optional[float], buy_open: float) -> Dict:
        """
        开盘跳空风控检查

        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            prev_close: 前一日收盘价
            buy_open: 今日开盘价

        Returns:
            Dict: {'can_buy': bool, 'reason': str}
        """
        self.logger.info(f"🔎 {stock_code} 开盘风控检查")
        self.logger.info(f"  前收盘: {prev_close}, 今开盘: {buy_open}")

        # 无前日数据
        if prev_close is None:
            self.logger.info(f"  ✅ 无前日数据，不允许买入")
            return {'can_buy': False, 'reason': '无前日数据','skip_category': '开盘风控'}

        # 计算跳空幅度
        gap_ratio = (buy_open - prev_close) / prev_close
        self.logger.info(f"  跳空幅度: {gap_ratio:.4f} ({gap_ratio:.2%})")

        # 风控1: 跌空超过3%
        if gap_ratio < -0.03:
            self.logger.warning(f"  🚫 跌空超过3% ({gap_ratio:.2%})，禁止买入")
            return {
                'can_buy': False,
                'reason': f'跌空{-gap_ratio:.2%}',
                'skip_category': '开盘风控'
            }

        # 风控2: 高开超过5%
        elif gap_ratio > 0.05:
            self.logger.warning(f"  🚫 高开超过5% ({gap_ratio:.2%})，禁止买入")
            return {
                'can_buy': False,
                'reason': f'高开{gap_ratio:.2%}',
                'skip_category': '开盘风控'
            }

        self.logger.info(f"  ✅ 通过风控检查，跳空{gap_ratio:.2%}在范围内")
        return {'can_buy': True, 'reason': '通过风控', 'gap_ratio': gap_ratio}

    def _save_trade_record(self, trade: 'XtTrade'):
        """
        保存成交记录

        Args:
            trade: 成交对象
        """
        try:
            # 这里可以保存到数据库或文件
            self.logger.info(f"保存成交记录: {trade.order_id}")
        except Exception as e:
            self.logger.error(f"保存成交记录失败: {e}")
