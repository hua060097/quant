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


# ==============================================================================
# 配置项（写死在文件内，避免依赖 self.config）
# ==============================================================================
# 目标下单窗口（本地时间）
BUY_WINDOW_START = "09:30:00"
BUY_WINDOW_END   = "09:30:20"

# 当盘口快速下跌时，偏“被动”的比例（降低追价，利用下跌降低成本）
# 0.0=完全追价，1.0=完全被动。建议 0.55~0.75 区间。
FALL_PASSIVE_RATIO = 0.65

# 被动挂单时给一个很小的加价（避免挂得太低完全成交不到）
# 例如 0.0005 表示在 ask1 基础上 +5bp。
BUY_SMALL_PREMIUM = 0.0005



def _safe_float(x, default: float = float("nan")) -> float:
    """Best-effort float conversion for tick fields."""
    import numpy as np
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return default
            x = x[0]
        v = float(x)
        return v
    except Exception:
        return default


def _round_price(price: float, tick_size: float = 0.01) -> float:
    """Round price to A-share tick size (default 0.01)."""
    import numpy as np
    try:
        if price is None or not np.isfinite(price):
            return float("nan")
        price = max(float(price), tick_size)
        ticks = round(price / tick_size)
        return float(ticks * tick_size)
    except Exception:
        return float(price)

# 允许的最大追价（相对当日开盘价）
# 例：0.010 表示最多追到 open * 1.01
BUY_SLIPPAGE_CAP_PCT = 0.010

# 分批下单参数（资金 30~50 万：建议 3~5 片，兼顾成交概率与冲击成本）
BUY_SLICES = 4

# 每片之间等待秒数（给撮合/回报一个节奏，避免频繁撤改导致拒单/拥塞）
BUY_WAIT_SECONDS = 0.6

# 每片最多撤单重试次数
BUY_MAX_RETRIES_PER_SLICE = 3

# 快涨/快跌判定阈值（相对开盘价）
FAST_UP_TH = 0.006
FAST_DOWN_TH = -0.006

# 价格撮合微调（单位：元，默认 A 股 0.01）
DEFAULT_TICK_SIZE = 0.01


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
        """开盘 09:30:00~09:30:20 智能买入（最小改动版，含急涨/急跌逻辑 + 追价上限）。

        设计目标（贴合你的诉求）：
          - **急涨**：尽量在 BUY_SLIPPAGE_CAP_PCT 范围内“有限追价”抢到；超过 cap 则不追（宁可错过）。
          - **急跌**：窗口前段更被动（尽量靠近低位成交），窗口后段若仍未成交再逐步抬价保证成交。
          - 下单价格同时受：涨停价、open*cap、以及卖一/买一报价约束。

        说明：
          - MiniQMT 普通权限只有五档/一档，对“真正的盘口速度”无法完全拟合，本实现尽量稳健。
        """
        self.logger.info("=" * 60)
        if target_volume:
            self.logger.info(f"开始指定数量买入: {stock_code}, {target_volume}股")
        else:
            self.logger.info(f"开始全仓买入: {stock_code}")
        self.logger.info("=" * 60)

        # -------------------------
        # 1) trade 模式：等待到 09:30:00（手动模式不等待）
        # -------------------------
        if target_volume is None:
            self._wait_until_market_open()

        # -------------------------
        # 2) 获取 prev_close（用于涨停价/风控）
        # -------------------------
        prev_close = self._get_prev_close(stock_code, buy_date)
        if not prev_close or prev_close <= 0:
            self.logger.warning(f"[{stock_code}] prev_close 获取失败，后续将尽量用 tick 上限价兜底")

        # -------------------------
        # 3) 查询资金/计算目标买入数量（trade 模式）
        # -------------------------
        if target_volume is None:
            asset = self.trader.query_stock_asset(self.trader.account)
            if not asset:
                self.logger.error("查询账户资金失败")
                return None
            available_cash = float(asset.cash)
            self.logger.info(f"可用资金: {available_cash:.2f}元")
        else:
            available_cash = 0.0  # 手动模式用不上

        # -------------------------
        # 4) 在 09:30:00~09:30:20 里分片下单
        # -------------------------
        win_start = datetime.strptime(BUY_WINDOW_START, "%H:%M:%S").time()
        win_end = datetime.strptime(BUY_WINDOW_END, "%H:%M:%S").time()

        def _now_t() -> datetime.time:
            return datetime.now().time()

        # 若因延迟已经过了窗口，直接按当前价快速下单（但仍遵守 cap）
        if _now_t() > win_end:
            self.logger.warning(f"[{stock_code}] 当前已超过买入窗口 {BUY_WINDOW_END}，将直接按最新盘口尝试一次")

        # 目标成交数量（先用一次 tick 估算 open/cap 价来算手数）
        tick0 = self._get_tick(stock_code)
        if tick0 is None:
            self.logger.error(f"[{stock_code}] 无法获取 tick，放弃买入")
            return None

        open_px = _safe_float(tick0.get("open") or tick0.get("openPrice") or tick0.get("lastPrice"))
        last_px = _safe_float(tick0.get("lastPrice"))
        ask1 = _safe_float((tick0.get("askPrice") or [None])[0])
        bid1 = _safe_float((tick0.get("bidPrice") or [None])[0])

        if not open_px or open_px <= 0:
            open_px = last_px
        if not open_px or open_px <= 0:
            self.logger.error(f"[{stock_code}] open/last 均无效，放弃买入")
            return None

        # trade 模式：开盘风控（你原来的 _check_opening_gap_risk 仍保留）
        if target_volume is None:
            risk_check = self._check_opening_gap_risk(stock_code, buy_date, prev_close, open_px)
            if not risk_check.get("can_buy", True):
                self.logger.warning(f"未通过开盘风控 ({risk_check.get('reason')})，取消买入")
                return None

        # 涨停价优先从 tick 拿（更准确），其次用 prev_close * limit_pct 推断
        up_limit_tick = _safe_float(tick0.get("upLimitPrice") or tick0.get("upperLimit"))
        up_limit = up_limit_tick
        if not up_limit or up_limit <= 0:
            limit_pct = 0.20 if (stock_code.startswith("30") or stock_code.startswith("688")) else 0.10
            if prev_close and prev_close > 0:
                up_limit = prev_close * (1.0 + limit_pct)

        # 追价 cap（相对 open）
        cap_price = open_px * (1.0 + BUY_SLIPPAGE_CAP_PCT)
        if up_limit and up_limit > 0:
            cap_price = min(cap_price, up_limit)  # cap 不能越过涨停

        # 计算目标买入量
        if target_volume is None:
            ref_price_for_size = min(cap_price, max(ask1, last_px) if (ask1 and ask1 > 0) else open_px)
            buy_volume = self._calculate_buy_volume_full(available_cash, ref_price_for_size)
        else:
            buy_volume = int(target_volume)

        if buy_volume < 100:
            self.logger.error(f"资金/数量不足: buy_volume={buy_volume} (<100)")
            return None

        self.logger.info(f"[{stock_code}] open={open_px:.2f} last={last_px:.2f} ask1={ask1:.2f} bid1={bid1:.2f} cap={cap_price:.2f} up_limit={up_limit if up_limit else float('nan'):.2f}")
        self.logger.info(f"[{stock_code}] 计划买入数量: {buy_volume}股，分 {BUY_SLICES} 片，单片等待 {BUY_WAIT_SECONDS}s")

        remaining = int(buy_volume)
        last_order_id = None

        for s in range(1, BUY_SLICES + 1):
            if remaining < 100:
                break

            # 每片数量（最后一片吃剩余）
            if s < BUY_SLICES:
                slice_qty = max(100, (buy_volume // BUY_SLICES) // 100 * 100)
                slice_qty = min(slice_qty, remaining)
            else:
                slice_qty = remaining

            tick = self._get_tick(stock_code)
            if tick is None:
                self.logger.warning(f"[{stock_code}] 获取 tick 失败，跳过第{s}片")
                continue

            open_px2 = _safe_float(tick.get("open") or tick.get("openPrice") or open_px)
            last_px2 = _safe_float(tick.get("lastPrice") or last_px)
            ask1_2 = _safe_float((tick.get("askPrice") or [None])[0])
            bid1_2 = _safe_float((tick.get("bidPrice") or [None])[0])
            if not open_px2 or open_px2 <= 0:
                open_px2 = open_px

            cap_price2 = open_px2 * (1.0 + BUY_SLIPPAGE_CAP_PCT)
            if up_limit and up_limit > 0:
                cap_price2 = min(cap_price2, up_limit)

            # 窗口进度（用于急跌策略）
            try:
                now_dt = datetime.now()
                start_dt = now_dt.replace(hour=win_start.hour, minute=win_start.minute, second=win_start.second, microsecond=0)
                end_dt = now_dt.replace(hour=win_end.hour, minute=win_end.minute, second=win_end.second, microsecond=0)
                prog = 1.0
                if end_dt > start_dt:
                    prog = (now_dt - start_dt).total_seconds() / max(1e-6, (end_dt - start_dt).total_seconds())
                    prog = float(min(1.0, max(0.0, prog)))
            except Exception:
                prog = 1.0

            # 急涨/急跌判定（相对开盘）
            rel = (last_px2 - open_px2) / (open_px2 + 1e-12)
            is_fast_up = rel >= FAST_UP_TH
            is_fast_down = rel <= FAST_DOWN_TH

            # 计算本片下单价
            order_price = None
            if is_fast_up:
                # 急涨：用卖一(ask1)去抢，但严格不超过 cap
                if ask1_2 and ask1_2 > 0:
                    order_price = min(ask1_2, cap_price2)
                else:
                    order_price = min(last_px2, cap_price2)
            elif is_fast_down:
                # 急跌：前段更被动（靠近买一），后段逐步抬价保证成交
                if prog <= FALL_PASSIVE_RATIO:
                    # 更被动：贴近 bid1（若 bid1 不可用则用 last）
                    base = bid1_2 if (bid1_2 and bid1_2 > 0) else last_px2
                    order_price = min(base, cap_price2)
                else:
                    # 后段：往 ask1 靠，仍不超 cap
                    base = ask1_2 if (ask1_2 and ask1_2 > 0) else last_px2
                    order_price = min(base, cap_price2)
            else:
                # 常态：卖一为主（保证成交概率），但不超 cap
                base = ask1_2 if (ask1_2 and ask1_2 > 0) else last_px2
                order_price = min(base, cap_price2)

            if not order_price or order_price <= 0:
                self.logger.warning(f"[{stock_code}] 计算下单价失败，跳过第{s}片")
                continue

            # 极端保护：不允许超过 cap（再次保证）
            order_price = min(order_price, cap_price2)
            if up_limit and up_limit > 0:
                order_price = min(order_price, up_limit)

            self.logger.info(
                f"[{stock_code}] 第{s}/{BUY_SLICES}片 | qty={slice_qty} | rel={rel:.2%} | "
                f"fast_up={is_fast_up} fast_down={is_fast_down} | price={order_price:.2f} (cap={cap_price2:.2f})"
            )

            order_id = self._place_buy_order(stock_code, slice_qty, order_price)
            last_order_id = order_id
            if not order_id:
                self.logger.warning(f"[{stock_code}] 第{s}片下单失败")
                continue

            # 等待成交
            filled = self._wait_for_order_fill(order_id, timeout=BUY_WAIT_SECONDS)
            if filled:
                self.logger.info(f"[{stock_code}] 第{s}片成交完成: {order_id}")
                remaining -= slice_qty
                continue

            # 未成交：撤单并进入下一片（下一片会重新取 tick/重新定价）
            self.logger.warning(f"[{stock_code}] 第{s}片未在 {BUY_WAIT_SECONDS}s 内成交，撤单重试")
            self._cancel_order_with_retry(order_id)

            # 为避免频繁打单，稍作等待（但仍尽量保持在窗口内）
            time.sleep(0.2)

        if remaining <= 0:
            self.logger.info(f"✅ {stock_code} 买入完成，最后订单: {last_order_id}")
            return last_order_id

        self.logger.error(f"❌ {stock_code} 买入未完成，剩余: {remaining}股（可能因 cap 限制/急涨冲出上限）")
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
        """获取卖一价 askPrice[0]（买入对手价）。

        - 优先 askPrice[0]
        - 若 askPrice 不可用/为0：fallback 到 lastPrice
        - 再不行：fallback 到 bidPrice[0]

        备注：之前误用 bidPrice[0] 会导致开盘急涨时“永远买不到”（太被动），
             急跌时又会“买在高点”（因为 bid1 与真实可成交价格偏离）。
        """

        # 缓存（5秒）
        cached = self.ask_price_cache.get(stock_code)
        if cached is not None:
            if (datetime.now() - cached["timestamp"]).total_seconds() < 5:
                return cached["price"]

        tick = self._get_tick(stock_code)
        if not tick:
            return None

        ask_list = tick.get("askPrice")
        bid_list = tick.get("bidPrice")

        ask1 = None
        if isinstance(ask_list, (list, tuple)) and len(ask_list) > 0:
            ask1 = ask_list[0]

        if (ask1 is None) or (ask1 == 0):
            ask1 = tick.get("lastPrice", 0)

        if ((ask1 is None) or (ask1 == 0)) and isinstance(bid_list, (list, tuple)) and len(bid_list) > 0:
            ask1 = bid_list[0]

        if ask1 is None or ask1 == 0:
            return None

        ask_price_float = float(ask1)
        self.ask_price_cache[stock_code] = {"price": ask_price_float, "timestamp": datetime.now()}
        return ask_price_float

    def _get_tick(self, stock_code: str) -> Optional[dict]:
        """获取单只股票 full tick（普通用户可用，只有买卖一等基础档）。

        返回 tick 字典（不是 {code: tick} 的外层结构）。失败返回 None。
        """
        try:
            from xtquant import xtdata
            tick_data = xtdata.get_full_tick([stock_code])
            if not tick_data or stock_code not in tick_data:
                return None
            return tick_data[stock_code]
        except Exception as e:
            self.logger.warning(f"[BUY] 获取tick失败: {stock_code}, {e}")
            return None

    def _calculate_order_price(self, stock_code: str, buy_date: str,
                             prev_close: float, last_price: float) -> Optional[float]:
        """计算买入下单价格（核心约束：不超过 open*(1+BUY_SLIPPAGE_CAP_PCT)）。

        逻辑（开盘窗口专用）：
        1) 取卖一 askPrice[0] 作为“最快成交”的参考价
        2) 计算上限价：cap_price = open * (1 + BUY_SLIPPAGE_CAP_PCT)
        3) 计算涨停价：优先 tick.upLimitPrice，否则按板块 10%/20% 推断
        4) 下单价 = min(ask1*(1+SMALL_PREMIUM), cap_price, up_limit)

        急跌时会在 smart_buy_full_position 内把定价更偏被动（bid1）处理；
        这里保持“通用上限钳制”。

        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            prev_close: 昨日收盘价
            last_price: 最新价（容错备用）

        Returns:
            float: 下单价格，失败返回None
        """
        tick = self._get_tick(stock_code)
        open_price = None
        if tick:
            open_price = tick.get("open") or tick.get("openPrice")
        if open_price is None or open_price == 0:
            open_price = last_price

        # 涨停价：优先 tick 提供，否则按板块推断
        up_limit = None
        if tick:
            up_limit = tick.get("upLimitPrice") or tick.get("up_limit")

        if up_limit is None or up_limit == 0:
            code = str(stock_code)
            is_20 = code.startswith("30") or code.startswith("688")
            limit_pct = 0.20 if is_20 else 0.10
            up_limit = prev_close * (1.0 + limit_pct)

        cap_price = float(open_price) * (1.0 + BUY_SLIPPAGE_CAP_PCT)

        ask1 = self._get_ask_price_from_xtdata(stock_code)
        if ask1 is None:
            ask1 = last_price

        order_price = min(float(ask1) * (1.0 + BUY_SMALL_PREMIUM), cap_price, float(up_limit))
        return float(_round_price(order_price))

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
