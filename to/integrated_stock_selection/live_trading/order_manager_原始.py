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
import math
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, List, TYPE_CHECKING, Any
import pandas as pd

from .config import StrategyConfig

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


def _floor_price(price: float, tick_size: float = 0.01) -> float:
    """Floor price to A-share tick size (default 0.01)."""
    import numpy as np
    try:
        if price is None or not np.isfinite(price):
            return float("nan")
        price = max(float(price), tick_size)
        ticks = math.floor(price / tick_size + 1e-12)
        return float(max(tick_size, ticks * tick_size))
    except Exception:
        return float(price)

# 允许的最大追价（相对当日开盘价）
# 例：0.010 表示最多追到 open * 1.01
BUY_SLIPPAGE_CAP_PCT = 0.010
FAST_UP_BUY_SLIPPAGE_CAP_PCT = 0.030

# 分批下单参数（资金 30~50 万：建议 3~5 片，兼顾成交概率与冲击成本）
BUY_SLICES = 4
SMALL_ORDER_SINGLE_SLICE_MAX = 500
MEDIUM_ORDER_DOUBLE_SLICE_MAX = 1000

# 每片之间等待秒数（给撮合/回报一个节奏，避免频繁撤改导致拒单/拥塞）
BUY_WAIT_SECONDS = 0.6
FAST_UP_WAIT_SECONDS = 0.35
FAST_DOWN_WAIT_SECONDS = 1.0
FAST_DOWN_EXTRA_WAIT_SECONDS = 0.8
FAST_DOWN_PRE_ORDER_WAIT_SECONDS = 0.8
FAST_UP_CANCEL_WAIT_SECONDS = 0.8
FAST_DOWN_RECHECK_WAIT_SECONDS = 1.0
FAST_DOWN_MAX_OBSERVE_SECONDS = 900.0
FAST_DOWN_PASSIVE_RAMP_SECONDS = 300.0

# 快速下跌趋稳判定：当rel波动范围收窄到该值以内，且持续时间超过阈值，认为下跌已趋稳，可以开始被动买入
FAST_DOWN_STABILIZE_BAND = 0.005  # rel波动范围 ±0.5%
FAST_DOWN_STABILIZE_MIN_SAMPLES = 10  # 至少采集这么多轮才做趋稳判断
FAST_DOWN_STABILIZE_DURATION = 25.0  # 持续25秒以上才算趋稳

# 每片订单的真实成交等待超时（与切片节奏分离）
BUY_FILL_TIMEOUT_SECONDS = 2.5
FAST_UP_FILL_TIMEOUT_SECONDS = 1.5
FAST_DOWN_FILL_TIMEOUT_SECONDS = 2.5

# 等待未终态订单进入终态的补充时间
MID_OUTSTANDING_WAIT_SECONDS = 2.5
FINAL_OUTSTANDING_WAIT_SECONDS = 6.0
MAX_TICK_FAILURES_BEFORE_ABORT = 5

# 每片最多撤单重试次数
BUY_MAX_RETRIES_PER_SLICE = 3

# 快涨/快跌判定阈值（相对开盘价）
FAST_UP_TH = 0.008
FAST_UP_MICRO_TH = 0.0030
FAST_DOWN_TH = -0.006
FAST_DOWN_MICRO_TH = -0.0015
FAST_DOWN_MIN_STREAK = 1
FAST_UP_MIN_STREAK = 2

# 开盘微观结构状态机参数
OPEN_STICKY_INITIAL_WINDOW_SECONDS = 2.5
OPEN_STICKY_MAX_REL = 0.0020
WHIPSAW_LOOKBACK_BARS = 5
WHIPSAW_MIN_DIRECTION_FLIPS = 2
WHIPSAW_MIN_AMPLITUDE = 0.0080
WHIPSAW_PAUSE_SECONDS = 1.2
WHIPSAW_MAX_SLICE_RATIO = 0.25
OPEN_STICKY_MAX_SLICE_RATIO = 0.30
FAST_BREAKOUT_MAX_SLICE_RATIO = 0.50

# 深水反弹状态：相对开盘仍可能为负，但相对开盘后低点已明显修复。
DEEP_REBOUND_MIN_DRAWDOWN = -0.030
DEEP_REBOUND_FROM_LOW = 0.018
DEEP_REBOUND_MAX_REL_FROM_OPEN = 0.004
DEEP_REBOUND_MIN_UP_STREAK = 2
DEEP_REBOUND_MAX_SLICE_RATIO = 0.35
DEEP_REBOUND_BUY_SLIPPAGE_CAP_PCT = 0.015

# 入场二阶段保护：先允许成交一部分，观察开盘后走势；若转弱则停止补剩余仓位。
# 这不是选股过滤，而是防止“09:30 几秒内买满后当天一路下跌”的执行层保护。
ENTRY_STAGE_GUARD_ENABLED = False
ENTRY_STAGE_INITIAL_MAX_RATIO = 0.50
ENTRY_STAGE_CONFIRM_SECONDS = 300.0
ENTRY_STAGE_ABORT_REL_FROM_OPEN = -0.015
ENTRY_STAGE_ABORT_REL_FROM_AVG_FILL = -0.012
ENTRY_STAGE_MIN_RECHECK_SECONDS = 2.0

# 成交后执行闸门：避免试单成交后，在价格没有确认强度时几秒内买满。
# 这里处理的是实盘执行质量，不改变选股信号。
EXECUTION_GOVERNOR_ENABLED = True
EXECUTION_MIN_CONFIRM_SECONDS = 20.0
EXECUTION_RECHECK_SECONDS = 1.0
EXECUTION_MAX_WAIT_SECONDS = 1800.0
EXECUTION_WEAK_MAX_WAIT_SECONDS = 900.0
EXECUTION_OPEN_NO_ABORT_SECONDS = 900.0
EXECUTION_RECOVERY_REL_FROM_OPEN = 0.0015
EXECUTION_RECOVERY_REL_FROM_AVG_FILL = 0.0000
EXECUTION_SOFT_WEAK_REL_FROM_OPEN = -0.0025
EXECUTION_SOFT_WEAK_REL_FROM_AVG_FILL = -0.0015
EXECUTION_DEEP_WEAK_REL_FROM_OPEN = -0.0120
EXECUTION_DEEP_WEAK_REL_FROM_AVG_FILL = -0.0080
EXECUTION_PANIC_REL_FROM_OPEN = -0.0300
EXECUTION_PANIC_REL_FROM_AVG_FILL = -0.0250
EXECUTION_HARD_PANIC_REL_FROM_OPEN = -0.0550
EXECUTION_HARD_PANIC_REL_FROM_AVG_FILL = -0.0500
EXECUTION_FLAT_MAX_SLICE_RATIO = 0.25

# 价格撮合微调（单位：元，默认 A 股 0.01）
DEFAULT_TICK_SIZE = 0.01

TERMINAL_ORDER_STATUSES = {53, 54, 56, 57}
OPEN_ORDER_STATUSES = {48, 49, 50, 51, 52, 55}


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

        elif status in [52, 55]:  # 52=部成待撤, 55=部成
            order_volume = int(getattr(order, 'order_volume', 0) or 0)
            self.logger.warning(f"⚠️ 订单部分成交: {order_id} (状态{status})")
            self.logger.warning(f"  已成交: {order.traded_volume}/{order_volume}")
            if order_id in self.order_manager.active_orders:
                self.order_manager.active_orders[order_id]['price'] = float(order.traded_price) if order.traded_price > 0 else 0
                self.order_manager.active_orders[order_id]['volume'] = int(order.traded_volume)

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

        # 最近一次已提交的卖单信息（供监控侧做退出态同步）
        self.last_sell_submission = None

        # 兼容属性：orders 指向 active_orders（供 main.py 使用）
        self.orders = self.active_orders

        # 重试配置
        self.MAX_RETRY_TIMES = 3  # 最大重试次数
        self.CANCEL_WAIT_SECONDS = 3  # 撤单等待时间（秒）
        self.ORDER_TIMEOUT_SECONDS = 10  # 订单超时时间（秒）

        # 价格缓存
        self.limit_price_cache = {}  # 涨停价缓存 {trade_date: {stock_code: up_limit}}
        self.ask_price_cache = {}  # 对手价缓存 {stock_code: ask_price}

    def _normalize_buy_order_price(self, price: float,
                                   cap_price: Optional[float] = None,
                                   up_limit: Optional[float] = None,
                                   tick_size: float = DEFAULT_TICK_SIZE) -> float:
        """买单最终价格规整：先截断到 cap/up_limit，再按最小价差向下规整。"""
        normalized = float(price)
        if cap_price is not None:
            normalized = min(normalized, float(cap_price))
        if up_limit is not None and float(up_limit) > 0:
            normalized = min(normalized, float(up_limit))
        normalized = _floor_price(normalized, tick_size)
        if cap_price is not None:
            normalized = min(normalized, _floor_price(float(cap_price), tick_size))
        if up_limit is not None and float(up_limit) > 0:
            normalized = min(normalized, _floor_price(float(up_limit), tick_size))
        return float(normalized)

    def _get_slice_wait_seconds(self, is_fast_up: bool, is_fast_down: bool) -> float:
        if is_fast_up:
            return FAST_UP_WAIT_SECONDS
        if is_fast_down:
            return FAST_DOWN_WAIT_SECONDS
        return BUY_WAIT_SECONDS

    def _get_slice_fill_timeout(self, is_fast_up: bool, is_fast_down: bool) -> float:
        if is_fast_up:
            return FAST_UP_FILL_TIMEOUT_SECONDS
        if is_fast_down:
            return FAST_DOWN_FILL_TIMEOUT_SECONDS
        return BUY_FILL_TIMEOUT_SECONDS

    def _round_lot(self, volume: int) -> int:
        volume = int(volume or 0)
        if volume <= 0:
            return 0
        return max(100, (volume // 100) * 100)

    def _cap_slice_by_ratio(self, slice_qty: int, remaining: int, ratio: float) -> int:
        cap_qty = self._round_lot(int(remaining * float(ratio)))
        if cap_qty <= 0:
            cap_qty = min(remaining, 100)
        cap_qty = max(100, min(int(remaining), cap_qty))
        return max(100, min(int(slice_qty), cap_qty))

    def _detect_fast_up(self,
                        open_px: float,
                        last_px: float,
                        ask1: float,
                        bid1: float,
                        prev_quote: Optional[Dict[str, float]],
                        up_streak: int) -> tuple[bool, int]:
        """
        识别是否正在快速上涨（tick级别的持续上移）。

        核心改动：不再仅凭\"距离开盘价多远\"来判断，而是基于tick级别
        的持续上移信号。一旦tick不再上移，即使股价仍高于开盘价，也会退出
        FAST_BREAKOUT状态。

        两种进入方式：
        1. 持续上移：last/ask1/bid1 连续多tick上移 + 价格高于开盘微阈值
        2. 初始急涨：价格大幅高于开盘 + 当前tick有上移证据
        """
        rel = (last_px - open_px) / (open_px + 1e-12)

        if not prev_quote:
            if rel >= FAST_UP_TH:
                return True, max(up_streak, FAST_UP_MIN_STREAK)
            return False, 0

        prev_last = _safe_float(prev_quote.get('last'))
        prev_ask1 = _safe_float(prev_quote.get('ask1'))
        prev_bid1 = _safe_float(prev_quote.get('bid1'))

        up_signals = 0
        if pd.notna(prev_last) and last_px > prev_last + DEFAULT_TICK_SIZE / 2:
            up_signals += 1
        if pd.notna(prev_ask1) and ask1 and ask1 > 0 and ask1 >= prev_ask1 + DEFAULT_TICK_SIZE / 2:
            up_signals += 1
        if pd.notna(prev_bid1) and bid1 and bid1 > 0 and bid1 >= prev_bid1 + DEFAULT_TICK_SIZE / 2:
            up_signals += 1

        if up_signals >= 2:
            up_streak += 1
        else:
            up_streak = 0

        # 维持条件：tick持续上移 + 价格已高于开盘微阈值
        if up_streak >= FAST_UP_MIN_STREAK and rel >= FAST_UP_MICRO_TH:
            return True, up_streak

        # 进入条件：绝对涨幅较大 + 当前tick有上移证据
        if rel >= FAST_UP_TH and up_signals >= 1:
            return True, max(up_streak + 1, FAST_UP_MIN_STREAK)

        return False, up_streak

    def _count_direction_flips(self, rel_history: List[float]) -> int:
        signs: List[int] = []
        for rel in rel_history:
            if rel >= FAST_UP_MICRO_TH:
                signs.append(1)
            elif rel <= FAST_DOWN_MICRO_TH:
                signs.append(-1)
        if len(signs) < 2:
            return 0
        flips = 0
        prev_sign = signs[0]
        for sign in signs[1:]:
            if sign != prev_sign:
                flips += 1
                prev_sign = sign
        return flips

    def _classify_open_microstructure(self,
                                      open_px: float,
                                      last_px: float,
                                      ask1: float,
                                      bid1: float,
                                      elapsed_seconds: float,
                                      recent_quotes: List[Dict[str, float]],
                                      prev_quote: Optional[Dict[str, float]],
                                      up_streak: int,
                                      down_streak: int) -> Dict[str, Any]:
        rel = (last_px - open_px) / (open_px + 1e-12)
        spread_rel = 0.0
        if ask1 and ask1 > 0 and bid1 and bid1 > 0:
            spread_rel = max(0.0, (ask1 - bid1) / (open_px + 1e-12))

        is_fast_up, new_up_streak = self._detect_fast_up(
            open_px=open_px,
            last_px=last_px,
            ask1=ask1,
            bid1=bid1,
            prev_quote=prev_quote,
            up_streak=up_streak,
        )
        is_fast_down, new_down_streak = self._detect_fast_down(
            open_px=open_px,
            last_px=last_px,
            ask1=ask1,
            bid1=bid1,
            prev_quote=prev_quote,
            down_streak=down_streak,
        )

        rel_history = [float(q.get('rel', 0.0) or 0.0) for q in recent_quotes[-WHIPSAW_LOOKBACK_BARS:]]
        amplitude = (max(rel_history) - min(rel_history)) if rel_history else 0.0
        flips = self._count_direction_flips(rel_history)
        recent_up = sum(1 for item in rel_history[-3:] if item >= FAST_UP_MICRO_TH)
        recent_down = sum(1 for item in rel_history[-3:] if item <= FAST_DOWN_MICRO_TH)

        if elapsed_seconds <= OPEN_STICKY_INITIAL_WINDOW_SECONDS and len(recent_quotes) <= 2 and abs(rel) <= OPEN_STICKY_MAX_REL:
            state = 'OPEN_STICKY'
        elif flips >= WHIPSAW_MIN_DIRECTION_FLIPS and amplitude >= WHIPSAW_MIN_AMPLITUDE:
            state = 'WHIPSAW'
        elif is_fast_down:
            state = 'FAST_BREAKDOWN'
        elif is_fast_up:
            # 过滤下跌过程中的弱反弹，只有“最近几根也偏强”的突破才算 fast_up。
            if rel < 0.010 and recent_down >= 2 and recent_up <= 1:
                is_fast_up = False
                new_up_streak = 0
                state = 'NORMAL'
            else:
                state = 'FAST_BREAKOUT'
        else:
            state = 'NORMAL'

        return {
            'state': state,
            'rel': rel,
            'spread_rel': spread_rel,
            'is_fast_up': is_fast_up,
            'is_fast_down': is_fast_down,
            'up_streak': new_up_streak,
            'down_streak': new_down_streak,
            'whipsaw_flips': flips,
            'whipsaw_amplitude': amplitude,
        }

    def _resolve_effective_slices(self, buy_volume: int) -> int:
        """按目标股数决定拆单片数，避免小单过度切片。"""
        if buy_volume <= SMALL_ORDER_SINGLE_SLICE_MAX:
            return 1
        if buy_volume <= MEDIUM_ORDER_DOUBLE_SLICE_MAX:
            return 2
        return BUY_SLICES

    def _evaluate_entry_execution_governor(self,
                                           stock_code: str,
                                           buy_volume: int,
                                           filled_so_far: int,
                                           outstanding_so_far: int,
                                           avg_fill_price: float,
                                           open_px: float,
                                           last_px: float,
                                           elapsed_seconds: float,
                                           is_fast_up: bool,
                                           open_state: str,
                                           state: Dict[str, Any]) -> Dict[str, Any]:
        """成交后判断是否允许继续补剩余仓位。

        原则：
        - 未成交前不拦截，让第一笔试单去发现真实成交能力。
        - 已有成交后，如果价格没有站回开盘价/成交均价，不急着补仓。
        - 真正持续走强时放行；持续走弱或低于成本过久时放弃剩余。
        """
        if not EXECUTION_GOVERNOR_ENABLED:
            return {'action': 'proceed', 'reason': 'disabled'}

        if filled_so_far <= 0 or outstanding_so_far > 0:
            return {'action': 'proceed', 'reason': 'no_filled_or_has_outstanding'}

        if not open_px or open_px <= 0 or not last_px or last_px <= 0:
            return {'action': 'wait', 'sleep': EXECUTION_RECHECK_SECONDS, 'reason': 'invalid_price'}

        avg_fill = float(avg_fill_price or 0.0)
        if avg_fill <= 0:
            avg_fill = float(open_px)

        fill_ratio = filled_so_far / max(1, int(buy_volume))
        rel_open = (last_px - open_px) / (open_px + 1e-12)
        rel_avg = (last_px - avg_fill) / (avg_fill + 1e-12)
        now_ts = time.time()

        if 'first_fill_ts' not in state:
            state['first_fill_ts'] = now_ts
        if 'first_fill_elapsed' not in state:
            state['first_fill_elapsed'] = float(elapsed_seconds)

        since_first_fill = max(0.0, now_ts - float(state.get('first_fill_ts', now_ts)))
        is_deep_rebound = open_state == 'DEEP_REBOUND'
        is_breakout = bool(is_fast_up and rel_open >= FAST_UP_MICRO_TH and rel_avg >= EXECUTION_SOFT_WEAK_REL_FROM_AVG_FILL)
        is_recovered = (
            rel_open >= EXECUTION_RECOVERY_REL_FROM_OPEN
            and rel_avg >= EXECUTION_RECOVERY_REL_FROM_AVG_FILL
        )
        is_soft_weak = (
            rel_open <= EXECUTION_SOFT_WEAK_REL_FROM_OPEN
            or rel_avg <= EXECUTION_SOFT_WEAK_REL_FROM_AVG_FILL
        )
        is_deep_weak = (
            rel_open <= EXECUTION_DEEP_WEAK_REL_FROM_OPEN
            or rel_avg <= EXECUTION_DEEP_WEAK_REL_FROM_AVG_FILL
        )
        is_cost_panic = rel_avg <= EXECUTION_PANIC_REL_FROM_AVG_FILL
        is_open_panic = rel_open <= EXECUTION_PANIC_REL_FROM_OPEN
        is_panic_weak = (
            is_open_panic
            or (is_cost_panic and rel_open <= EXECUTION_DEEP_WEAK_REL_FROM_OPEN)
        )
        is_hard_panic_weak = (
            rel_open <= EXECUTION_HARD_PANIC_REL_FROM_OPEN
            or rel_avg <= EXECUTION_HARD_PANIC_REL_FROM_AVG_FILL
        )
        in_open_noise_window = elapsed_seconds < EXECUTION_OPEN_NO_ABORT_SECONDS

        if is_soft_weak or is_deep_weak:
            state.setdefault('weak_since_ts', now_ts)
        else:
            state.pop('weak_since_ts', None)
        weak_duration = max(0.0, now_ts - float(state.get('weak_since_ts', now_ts)))

        detail = (
            f"filled={filled_so_far}/{buy_volume}({fill_ratio:.0%}), "
            f"last={last_px:.2f}, open_rel={rel_open:.2%}, avg_rel={rel_avg:.2%}, "
            f"elapsed={elapsed_seconds:.1f}s, since_fill={since_first_fill:.1f}s, state={open_state}"
        )

        if is_breakout:
            return {
                'action': 'proceed',
                'reason': f"confirmed_strength, {detail}",
            }

        if is_deep_rebound and rel_avg >= EXECUTION_SOFT_WEAK_REL_FROM_AVG_FILL:
            return {
                'action': 'proceed_limited',
                'max_slice_ratio': DEEP_REBOUND_MAX_SLICE_RATIO,
                'reason': f"deep_rebound_repair_use_limited_slice, {detail}",
            }

        if is_panic_weak and in_open_noise_window and not is_hard_panic_weak:
            return {
                'action': 'wait',
                'sleep': EXECUTION_RECHECK_SECONDS,
                'reason': (
                    f"open_noise_panic_guard_wait_for_repair, weak_duration={weak_duration:.1f}s, "
                    f"{detail}"
                ),
            }

        if is_panic_weak:
            return {
                'action': 'abort',
                'reason': f"panic_weak_stop_remaining, {detail}",
            }

        if since_first_fill < EXECUTION_MIN_CONFIRM_SECONDS:
            return {
                'action': 'wait',
                'sleep': EXECUTION_RECHECK_SECONDS,
                'reason': f"wait_for_post_fill_confirmation, {detail}",
            }

        if is_recovered:
            return {
                'action': 'proceed',
                'reason': f"confirmed_recovery, {detail}",
            }

        if is_deep_weak:
            if (
                not in_open_noise_window
                and (weak_duration >= EXECUTION_WEAK_MAX_WAIT_SECONDS or since_first_fill >= EXECUTION_MAX_WAIT_SECONDS)
            ):
                return {
                    'action': 'abort',
                    'reason': f"deep_weak_not_repaired_after_long_observation, weak_duration={weak_duration:.1f}s, {detail}",
                }
            return {
                'action': 'wait',
                'sleep': EXECUTION_RECHECK_SECONDS,
                'reason': f"deep_weak_wait_for_repair, weak_duration={weak_duration:.1f}s, {detail}",
            }

        if is_soft_weak and not in_open_noise_window and weak_duration >= EXECUTION_WEAK_MAX_WAIT_SECONDS:
            return {
                'action': 'abort',
                'reason': f"weak_too_long_after_long_observation, weak_duration={weak_duration:.1f}s, {detail}",
            }

        if since_first_fill >= EXECUTION_MAX_WAIT_SECONDS:
            return {
                'action': 'abort',
                'reason': f"no_recovery_timeout_stop_remaining, {detail}",
            }

        if is_soft_weak or open_state in {'OPEN_STICKY', 'WHIPSAW'}:
            return {
                'action': 'wait',
                'sleep': EXECUTION_RECHECK_SECONDS,
                'reason': f"not_confirmed_or_soft_weak, {detail}",
            }

        return {
            'action': 'proceed_limited',
            'max_slice_ratio': EXECUTION_FLAT_MAX_SLICE_RATIO,
            'reason': f"flat_not_weak_use_small_slice, {detail}",
        }

    def smart_buy_full_position(self, stock_code: str,
                                buy_date: str,
                                target_volume: int = None,
                                wait_for_open: bool = None,
                                apply_opening_risk: bool = None,
                                target_rank: int = 1,
                                is_weak_market_env: bool = False) -> Optional[Dict[str, Any]]:
        """开盘后智能买入（含急涨/急跌/WHIPSAW状态机 + 追价上限）。

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
        if wait_for_open is None:
            wait_for_open = target_volume is None
        if apply_opening_risk is None:
            apply_opening_risk = target_volume is None

        if wait_for_open:
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
        # 4) 开盘后按状态机分片下单；若持续 fast_down，可超过原硬窗口继续观察
        # -------------------------
        win_start = datetime.strptime(BUY_WINDOW_START, "%H:%M:%S").time()

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
        if apply_opening_risk:
            risk_check = self._check_opening_gap_risk(
                stock_code,
                buy_date,
                prev_close,
                open_px,
                target_rank=target_rank,
                is_weak_market_env=is_weak_market_env,
            )
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

        effective_slices = self._resolve_effective_slices(buy_volume)
        self.logger.info(f"[{stock_code}] open={open_px:.2f} last={last_px:.2f} ask1={ask1:.2f} bid1={bid1:.2f} cap={cap_price:.2f} up_limit={up_limit if up_limit else float('nan'):.2f}")
        self.logger.info(f"[{stock_code}] 计划买入数量: {buy_volume}股，分 {effective_slices} 片，小单阈值={SMALL_ORDER_SINGLE_SLICE_MAX}股")

        remaining = int(buy_volume)
        last_order_id = None
        attempted_order_ids: List[str] = []
        aggressive_fast_up_mode = False
        prev_quote_snapshot: Optional[Dict[str, float]] = None
        recent_quotes: List[Dict[str, float]] = []
        fast_up_streak = 0
        fast_down_streak = 0
        session_low_rel = 0.0
        session_low_price = float(open_px)
        deep_rebound_streak = 0
        max_order_attempts = effective_slices + 2
        attempt_no = 0
        observe_rounds = 0
        fast_down_wait_logged = False
        fast_down_rel_tracker: List[tuple] = []  # (elapsed_seconds, rel) for stabilization detection
        tick_failures = 0
        entry_stage_guard_released = not ENTRY_STAGE_GUARD_ENABLED
        entry_stage_guard_aborted = False
        entry_stage_initial_target = max(100, int((buy_volume * ENTRY_STAGE_INITIAL_MAX_RATIO) // 100) * 100)
        entry_stage_initial_target = min(int(buy_volume), entry_stage_initial_target)
        execution_governor_state: Dict[str, Any] = {}

        while attempt_no < max_order_attempts:
            observe_rounds += 1
            exec_state = self._get_order_execution_totals(stock_code, attempted_order_ids)
            filled_so_far = int(exec_state['filled_volume'])
            outstanding_so_far = int(exec_state['outstanding_volume'])
            remaining = max(0, int(buy_volume) - filled_so_far - outstanding_so_far)

            if remaining < 100:
                if outstanding_so_far > 0:
                    self.logger.info(
                        f"[{stock_code}] 仍有未终态订单占用数量: filled={filled_so_far}, outstanding={outstanding_so_far}, "
                        f"remaining_to_order={remaining}，等待旧单最终回报"
                    )
                    self._wait_for_outstanding_orders(
                        stock_code,
                        attempted_order_ids,
                        timeout=MID_OUTSTANDING_WAIT_SECONDS,
                    )
                    exec_state = self._get_order_execution_totals(stock_code, attempted_order_ids)
                    filled_so_far = int(exec_state['filled_volume'])
                    outstanding_so_far = int(exec_state['outstanding_volume'])
                    remaining = max(0, int(buy_volume) - filled_so_far - outstanding_so_far)
                    if remaining < 100:
                        self.logger.info(
                            f"[{stock_code}] 已按真实成交/挂单更新剩余仓位: filled={filled_so_far}, "
                            f"outstanding={outstanding_so_far}, remaining_to_order={remaining}，停止继续下单"
                        )
                        break
                else:
                    self.logger.info(
                        f"[{stock_code}] 已按真实成交更新剩余仓位: filled={filled_so_far}, remaining={remaining}，停止继续下单"
                    )
                    break

            if ENTRY_STAGE_GUARD_ENABLED and not entry_stage_guard_released:
                if filled_so_far >= entry_stage_initial_target:
                    now_dt = datetime.now()
                    start_dt = now_dt.replace(hour=win_start.hour, minute=win_start.minute, second=win_start.second, microsecond=0)
                    elapsed_guard = max(0.0, (now_dt - start_dt).total_seconds())
                    if elapsed_guard < ENTRY_STAGE_CONFIRM_SECONDS:
                        wait_s = min(
                            ENTRY_STAGE_MIN_RECHECK_SECONDS,
                            max(0.1, ENTRY_STAGE_CONFIRM_SECONDS - elapsed_guard),
                        )
                        self.logger.info(
                            f"[{stock_code}] 入场二阶段保护：已成交 {filled_so_far}/{buy_volume} "
                            f"(首阶段目标={entry_stage_initial_target})，等待开盘后 "
                            f"{ENTRY_STAGE_CONFIRM_SECONDS:.0f}s 再决定是否补仓，sleep={wait_s:.1f}s"
                        )
                        time.sleep(wait_s)
                        continue

                    tick_guard = self._get_tick(stock_code)
                    if tick_guard is None:
                        self.logger.warning(
                            f"[{stock_code}] 入场二阶段保护：确认时获取 tick 失败，暂缓补仓"
                        )
                        time.sleep(ENTRY_STAGE_MIN_RECHECK_SECONDS)
                        continue

                    open_guard = _safe_float(tick_guard.get("open") or tick_guard.get("openPrice") or open_px)
                    last_guard = _safe_float(tick_guard.get("lastPrice") or tick_guard.get("last") or last_px)
                    exec_summary = self._summarize_buy_execution(
                        stock_code=stock_code,
                        requested_volume=int(buy_volume),
                        attempted_order_ids=attempted_order_ids,
                        fallback_buy_date=buy_date,
                        last_order_id=last_order_id,
                        open_px=open_px,
                    )
                    avg_fill = float(exec_summary.get("avg_price") or 0.0)
                    if not avg_fill or avg_fill <= 0:
                        avg_fill = float(open_guard or open_px)

                    rel_from_open = (last_guard - open_guard) / (open_guard + 1e-12) if open_guard and open_guard > 0 else 0.0
                    rel_from_avg = (last_guard - avg_fill) / (avg_fill + 1e-12) if avg_fill and avg_fill > 0 else 0.0
                    if (
                        rel_from_open <= ENTRY_STAGE_ABORT_REL_FROM_OPEN
                        or rel_from_avg <= ENTRY_STAGE_ABORT_REL_FROM_AVG_FILL
                    ):
                        self.logger.warning(
                            f"[{stock_code}] 入场二阶段保护触发：停止补剩余仓位 "
                            f"(filled={filled_so_far}/{buy_volume}, last={last_guard:.2f}, "
                            f"open_rel={rel_from_open:.2%}<= {ENTRY_STAGE_ABORT_REL_FROM_OPEN:.2%} "
                            f"或 avg_rel={rel_from_avg:.2%}<= {ENTRY_STAGE_ABORT_REL_FROM_AVG_FILL:.2%})"
                        )
                        entry_stage_guard_aborted = True
                        break

                    entry_stage_guard_released = True
                    self.logger.info(
                        f"[{stock_code}] 入场二阶段保护通过：允许补剩余仓位 "
                        f"(filled={filled_so_far}/{buy_volume}, last={last_guard:.2f}, "
                        f"open_rel={rel_from_open:.2%}, avg_rel={rel_from_avg:.2%})"
                    )

            # 每片数量（最后一片吃剩余）
            next_attempt_no = attempt_no + 1
            if aggressive_fast_up_mode:
                slice_qty = remaining
            elif next_attempt_no < effective_slices:
                slice_qty = max(100, (buy_volume // effective_slices) // 100 * 100)
                slice_qty = min(slice_qty, remaining)
            else:
                slice_qty = remaining
            if ENTRY_STAGE_GUARD_ENABLED and not entry_stage_guard_released:
                first_stage_remaining = max(0, entry_stage_initial_target - filled_so_far - outstanding_so_far)
                if first_stage_remaining < 100:
                    time.sleep(ENTRY_STAGE_MIN_RECHECK_SECONDS)
                    continue
                slice_qty = min(slice_qty, first_stage_remaining)

            tick = self._get_tick(stock_code)
            if tick is None:
                tick_failures += 1
                self.logger.warning(
                    f"[{stock_code}] 获取 tick 失败，暂停 {FAST_DOWN_RECHECK_WAIT_SECONDS:.2f}s 后重试 "
                    f"(failure={tick_failures}/{MAX_TICK_FAILURES_BEFORE_ABORT})"
                )
                if tick_failures >= MAX_TICK_FAILURES_BEFORE_ABORT:
                    self.logger.error(f"[{stock_code}] 连续获取 tick 失败，放弃当日买入")
                    return None
                time.sleep(FAST_DOWN_RECHECK_WAIT_SECONDS)
                continue
            tick_failures = 0

            open_px2 = _safe_float(tick.get("open") or tick.get("openPrice") or open_px)
            last_px2 = _safe_float(tick.get("lastPrice") or last_px)
            ask1_2 = _safe_float((tick.get("askPrice") or [None])[0])
            bid1_2 = _safe_float((tick.get("bidPrice") or [None])[0])
            if not open_px2 or open_px2 <= 0:
                open_px2 = open_px

            current_cap_pct = FAST_UP_BUY_SLIPPAGE_CAP_PCT if aggressive_fast_up_mode else BUY_SLIPPAGE_CAP_PCT
            cap_price2 = open_px2 * (1.0 + current_cap_pct)
            if up_limit and up_limit > 0:
                cap_price2 = min(cap_price2, up_limit)

            # 窗口进度（用于急跌策略）
            try:
                now_dt = datetime.now()
                start_dt = now_dt.replace(hour=win_start.hour, minute=win_start.minute, second=win_start.second, microsecond=0)
                prog = 1.0
                elapsed_seconds = max(0.0, (now_dt - start_dt).total_seconds())
                prog = float(min(1.0, max(0.0, elapsed_seconds / max(1e-6, FAST_DOWN_PASSIVE_RAMP_SECONDS))))
            except Exception:
                prog = 1.0
                elapsed_seconds = 999.0

            current_rel = (last_px2 - open_px2) / (open_px2 + 1e-12)
            preview_quotes = (recent_quotes + [{
                'rel': float(current_rel),
                'last': float(last_px2) if last_px2 and last_px2 > 0 else float("nan"),
                'ask1': float(ask1_2) if ask1_2 and ask1_2 > 0 else float("nan"),
                'bid1': float(bid1_2) if bid1_2 and bid1_2 > 0 else float("nan"),
            }])[-WHIPSAW_LOOKBACK_BARS:]
            micro = self._classify_open_microstructure(
                open_px=open_px2,
                last_px=last_px2,
                ask1=ask1_2,
                bid1=bid1_2,
                elapsed_seconds=elapsed_seconds,
                recent_quotes=preview_quotes,
                prev_quote=prev_quote_snapshot,
                up_streak=fast_up_streak,
                down_streak=fast_down_streak,
            )
            rel = float(micro['rel'])
            is_fast_up = bool(micro['is_fast_up'])
            is_fast_down = bool(micro['is_fast_down'])
            fast_up_streak = int(micro['up_streak'])
            fast_down_streak = int(micro['down_streak'])
            open_state = str(micro['state'])
            session_low_rel = min(float(session_low_rel), rel)
            if last_px2 and last_px2 > 0:
                if session_low_price <= 0 or float(last_px2) < float(session_low_price):
                    session_low_price = float(last_px2)
            rebound_from_low = (
                (float(last_px2) - float(session_low_price)) / (float(session_low_price) + 1e-12)
                if session_low_price and session_low_price > 0 and last_px2 and last_px2 > 0
                else 0.0
            )
            if (
                session_low_rel <= DEEP_REBOUND_MIN_DRAWDOWN
                and rebound_from_low >= DEEP_REBOUND_FROM_LOW
                and rel <= DEEP_REBOUND_MAX_REL_FROM_OPEN
                and fast_up_streak >= DEEP_REBOUND_MIN_UP_STREAK
                and not is_fast_down
            ):
                deep_rebound_streak += 1
                is_fast_up = False
                open_state = 'DEEP_REBOUND'
            else:
                deep_rebound_streak = 0
            prev_quote_snapshot = {
                'last': float(last_px2) if last_px2 and last_px2 > 0 else float("nan"),
                'ask1': float(ask1_2) if ask1_2 and ask1_2 > 0 else float("nan"),
                'bid1': float(bid1_2) if bid1_2 and bid1_2 > 0 else float("nan"),
            }
            recent_quotes = preview_quotes

            if is_fast_down and remaining >= 100:
                # 记录rel用于趋稳检测
                fast_down_rel_tracker.append((elapsed_seconds, rel))
                # 清理超过2倍趋稳时长的旧记录
                fast_down_rel_tracker = [
                    (e, r) for e, r in fast_down_rel_tracker
                    if elapsed_seconds - e <= FAST_DOWN_STABILIZE_DURATION * 2
                ]

                if elapsed_seconds <= FAST_DOWN_MAX_OBSERVE_SECONDS:
                    # ---- 趋稳检测：波动收窄 + 持续一定时间 → 退出观察，开始被动买入 ----
                    stabilized = False
                    if len(fast_down_rel_tracker) >= FAST_DOWN_STABILIZE_MIN_SAMPLES:
                        recent_rels = [r for _, r in fast_down_rel_tracker[-FAST_DOWN_STABILIZE_MIN_SAMPLES:]]
                        rel_range = max(recent_rels) - min(recent_rels)
                        rel_duration = fast_down_rel_tracker[-1][0] - fast_down_rel_tracker[-FAST_DOWN_STABILIZE_MIN_SAMPLES][0]
                        if rel_range <= FAST_DOWN_STABILIZE_BAND and rel_duration >= FAST_DOWN_STABILIZE_DURATION:
                            stabilized = True

                    if not stabilized:
                        if (not fast_down_wait_logged) or (observe_rounds % 3 == 0):
                            self.logger.info(
                                f"[{stock_code}] 检测到持续FAST_BREAKDOWN，暂不下单，继续观察 {FAST_DOWN_RECHECK_WAIT_SECONDS:.2f}s "
                                f"(elapsed={elapsed_seconds:.1f}s, rel={rel:.2%}, remaining={remaining}, observe_round={observe_rounds})"
                            )
                            fast_down_wait_logged = True
                        time.sleep(FAST_DOWN_RECHECK_WAIT_SECONDS)
                        continue
                    # 趋稳：记录日志后继续执行下单逻辑（仍保持is_fast_down=True走被动定价）
                    self.logger.info(
                        f"[{stock_code}] 下跌已趋稳(波动{rel_range:.2%}<{FAST_DOWN_STABILIZE_BAND:.2%})，"
                        f"切换到被动买入 (rel={rel:.2%}, remaining={remaining}, observe_round={observe_rounds})"
                    )
                    fast_down_rel_tracker.clear()
                else:
                    # 趋稳检查：观察窗口之后是否稳定过
                    if len(fast_down_rel_tracker) >= FAST_DOWN_STABILIZE_MIN_SAMPLES:
                        recent_rels = [r for _, r in fast_down_rel_tracker[-FAST_DOWN_STABILIZE_MIN_SAMPLES:]]
                        rel_range = max(recent_rels) - min(recent_rels)
                        if rel_range <= FAST_DOWN_STABILIZE_BAND:
                            self.logger.info(
                                f"[{stock_code}] 超{FAST_DOWN_MAX_OBSERVE_SECONDS:.0f}s但近期已趋稳(波动{rel_range:.2%})，"
                                f"切换到被动买入 (rel={rel:.2%}, remaining={remaining})"
                            )
                            fast_down_rel_tracker.clear()
                            # fall through to buy logic
                        else:
                            self.logger.warning(
                                f"[{stock_code}] 开盘后持续下跌已超过 {FAST_DOWN_MAX_OBSERVE_SECONDS:.0f}s，"
                                f"且未趋稳，放弃当日买入 (rel={rel:.2%}, remaining={remaining})"
                            )
                            if filled_so_far > 0:
                                break
                            return None
                    else:
                        self.logger.warning(
                            f"[{stock_code}] 开盘后持续下跌已超过 {FAST_DOWN_MAX_OBSERVE_SECONDS:.0f}s，放弃当日买入 "
                            f"(rel={rel:.2%}, remaining={remaining})"
                        )
                        if filled_so_far > 0:
                            break
                        return None
            fast_down_wait_logged = False

            if open_state == 'WHIPSAW':
                self.logger.info(
                    f"[{stock_code}] 检测到开盘WHIPSAW，先暂停 {WHIPSAW_PAUSE_SECONDS:.2f}s "
                    f"(flips={micro['whipsaw_flips']}, amp={micro['whipsaw_amplitude']:.2%}, rel={rel:.2%})"
                )
                time.sleep(WHIPSAW_PAUSE_SECONDS)
                tick = self._get_tick(stock_code)
                if tick is None:
                    self.logger.warning(f"[{stock_code}] WHIPSAW暂停后获取 tick 失败，跳过第{attempt_no}片")
                    continue
                open_px2 = _safe_float(tick.get("open") or tick.get("openPrice") or open_px2)
                last_px2 = _safe_float(tick.get("lastPrice") or last_px2)
                ask1_2 = _safe_float((tick.get("askPrice") or [None])[0])
                bid1_2 = _safe_float((tick.get("bidPrice") or [None])[0])
                if not open_px2 or open_px2 <= 0:
                    open_px2 = open_px
                current_rel = (last_px2 - open_px2) / (open_px2 + 1e-12)
                preview_quotes = (recent_quotes + [{
                    'rel': float(current_rel),
                    'last': float(last_px2) if last_px2 and last_px2 > 0 else float("nan"),
                    'ask1': float(ask1_2) if ask1_2 and ask1_2 > 0 else float("nan"),
                    'bid1': float(bid1_2) if bid1_2 and bid1_2 > 0 else float("nan"),
                }])[-WHIPSAW_LOOKBACK_BARS:]
                micro = self._classify_open_microstructure(
                    open_px=open_px2,
                    last_px=last_px2,
                    ask1=ask1_2,
                    bid1=bid1_2,
                    elapsed_seconds=elapsed_seconds + WHIPSAW_PAUSE_SECONDS,
                    recent_quotes=preview_quotes,
                    prev_quote=prev_quote_snapshot,
                    up_streak=fast_up_streak,
                    down_streak=fast_down_streak,
                )
                rel = float(micro['rel'])
                is_fast_up = bool(micro['is_fast_up'])
                is_fast_down = bool(micro['is_fast_down'])
                fast_up_streak = int(micro['up_streak'])
                fast_down_streak = int(micro['down_streak'])
                open_state = str(micro['state'])
                prev_quote_snapshot = {
                    'last': float(last_px2) if last_px2 and last_px2 > 0 else float("nan"),
                    'ask1': float(ask1_2) if ask1_2 and ask1_2 > 0 else float("nan"),
                    'bid1': float(bid1_2) if bid1_2 and bid1_2 > 0 else float("nan"),
                }
                recent_quotes = preview_quotes

                if is_fast_down and remaining >= 100:
                    if elapsed_seconds <= FAST_DOWN_MAX_OBSERVE_SECONDS:
                        self.logger.info(
                            f"[{stock_code}] WHIPSAW重判后仍为FAST_BREAKDOWN，继续观察 {FAST_DOWN_RECHECK_WAIT_SECONDS:.2f}s "
                            f"(elapsed={elapsed_seconds + WHIPSAW_PAUSE_SECONDS:.1f}s, rel={rel:.2%}, remaining={remaining})"
                        )
                        time.sleep(FAST_DOWN_RECHECK_WAIT_SECONDS)
                        continue
                    self.logger.warning(
                        f"[{stock_code}] WHIPSAW重判后仍持续下跌且超过观察上限，放弃当日买入 "
                        f"(rel={rel:.2%}, remaining={remaining})"
                    )
                    if filled_so_far > 0:
                        break
                    return None

            if EXECUTION_GOVERNOR_ENABLED and filled_so_far > 0:
                exec_summary_for_governor = self._summarize_buy_execution(
                    stock_code=stock_code,
                    requested_volume=int(buy_volume),
                    attempted_order_ids=attempted_order_ids,
                    fallback_buy_date=buy_date,
                    last_order_id=last_order_id,
                    open_px=open_px,
                )
                avg_fill_for_governor = float(exec_summary_for_governor.get("avg_price") or 0.0)
                governor_decision = self._evaluate_entry_execution_governor(
                    stock_code=stock_code,
                    buy_volume=int(buy_volume),
                    filled_so_far=int(filled_so_far),
                    outstanding_so_far=int(outstanding_so_far),
                    avg_fill_price=avg_fill_for_governor,
                    open_px=float(open_px2),
                    last_px=float(last_px2),
                    elapsed_seconds=float(elapsed_seconds),
                    is_fast_up=bool(is_fast_up),
                    open_state=str(open_state),
                    state=execution_governor_state,
                )
                governor_action = str(governor_decision.get('action') or 'proceed')
                if governor_action == 'abort':
                    self.logger.warning(
                        f"[{stock_code}] 执行闸门停止补剩余仓位: {governor_decision.get('reason')}"
                    )
                    break
                if governor_action == 'wait':
                    sleep_s = float(governor_decision.get('sleep') or EXECUTION_RECHECK_SECONDS)
                    self.logger.info(
                        f"[{stock_code}] 执行闸门暂停补仓 {sleep_s:.1f}s: {governor_decision.get('reason')}"
                    )
                    time.sleep(max(0.1, sleep_s))
                    continue
                if governor_action == 'proceed_limited':
                    max_slice_ratio = float(governor_decision.get('max_slice_ratio') or EXECUTION_FLAT_MAX_SLICE_RATIO)
                    adjusted_qty = self._cap_slice_by_ratio(
                        slice_qty=slice_qty,
                        remaining=remaining,
                        ratio=max_slice_ratio,
                    )
                    if adjusted_qty < slice_qty:
                        self.logger.info(
                            f"[{stock_code}] 执行闸门仅允许小额补仓: {slice_qty} -> {adjusted_qty}; "
                            f"{governor_decision.get('reason')}"
                        )
                        slice_qty = adjusted_qty

            attempt_no = next_attempt_no
            if is_fast_up and not aggressive_fast_up_mode:
                aggressive_fast_up_mode = True
                current_cap_pct = FAST_UP_BUY_SLIPPAGE_CAP_PCT
                cap_price2 = open_px2 * (1.0 + current_cap_pct)
                if up_limit and up_limit > 0:
                    cap_price2 = min(cap_price2, up_limit)
                if slice_qty < remaining:
                    slice_qty = self._cap_slice_by_ratio(slice_qty=remaining, remaining=remaining, ratio=FAST_BREAKOUT_MAX_SLICE_RATIO)
                self.logger.info(
                    f"[{stock_code}] 检测到 fast_up，切换为aggressive模式："
                    f"剩余仓位分段追单，cap_pct={current_cap_pct:.2%}, qty={slice_qty}"
                )

            if open_state == 'OPEN_STICKY':
                adjusted_qty = self._cap_slice_by_ratio(slice_qty=slice_qty, remaining=remaining, ratio=OPEN_STICKY_MAX_SLICE_RATIO)
                if adjusted_qty < slice_qty:
                    self.logger.info(
                        f"[{stock_code}] 开盘OPEN_STICKY，首段试单缩量: {slice_qty} -> {adjusted_qty} "
                        f"(elapsed={elapsed_seconds:.2f}s, rel={rel:.2%})"
                    )
                    slice_qty = adjusted_qty
            elif open_state == 'WHIPSAW':
                adjusted_qty = self._cap_slice_by_ratio(slice_qty=slice_qty, remaining=remaining, ratio=WHIPSAW_MAX_SLICE_RATIO)
                if adjusted_qty < slice_qty:
                    self.logger.info(
                        f"[{stock_code}] 开盘WHIPSAW，缩量试单: {slice_qty} -> {adjusted_qty} "
                        f"(flips={micro['whipsaw_flips']}, amp={micro['whipsaw_amplitude']:.2%})"
                    )
                    slice_qty = adjusted_qty
            elif open_state == 'DEEP_REBOUND':
                adjusted_qty = self._cap_slice_by_ratio(slice_qty=slice_qty, remaining=remaining, ratio=DEEP_REBOUND_MAX_SLICE_RATIO)
                if adjusted_qty < slice_qty:
                    self.logger.info(
                        f"[{stock_code}] 深水反弹DEEP_REBOUND，限制单片仓位: {slice_qty} -> {adjusted_qty} "
                        f"(low_rel={session_low_rel:.2%}, rebound_from_low={rebound_from_low:.2%}, "
                        f"streak={deep_rebound_streak})"
                    )
                    slice_qty = adjusted_qty

            # 计算本片下单价
            order_price = None
            if open_state == 'WHIPSAW':
                base = ask1_2 if (ask1_2 and ask1_2 > 0) else last_px2
                order_price = min(base, cap_price2)
            elif open_state == 'DEEP_REBOUND':
                deep_rebound_cap = open_px2 * (1.0 + DEEP_REBOUND_BUY_SLIPPAGE_CAP_PCT)
                if up_limit and up_limit > 0:
                    deep_rebound_cap = min(deep_rebound_cap, up_limit)
                base = ask1_2 if (ask1_2 and ask1_2 > 0) else last_px2
                order_price = min(base, deep_rebound_cap)
            elif is_fast_up:
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
                self.logger.warning(f"[{stock_code}] 计算下单价失败，跳过第{attempt_no}片")
                continue

            order_price_raw = float(order_price)
            order_price = self._normalize_buy_order_price(
                order_price_raw,
                cap_price=cap_price2,
                up_limit=up_limit,
                tick_size=DEFAULT_TICK_SIZE,
            )
            if not order_price or order_price <= 0:
                self.logger.warning(f"[{stock_code}] 规整后的下单价无效，跳过第{attempt_no}片")
                continue

            slice_wait_seconds = self._get_slice_wait_seconds(is_fast_up, is_fast_down)
            fill_timeout_seconds = self._get_slice_fill_timeout(is_fast_up, is_fast_down)

            self.logger.info(
                f"[{stock_code}] 第{attempt_no}/{effective_slices}片 | qty={slice_qty} | state={open_state} | rel={rel:.2%} | "
                f"fast_up={is_fast_up} fast_down={is_fast_down} | "
                f"low_rel={session_low_rel:.2%} rebound_low={rebound_from_low:.2%} | "
                f"price_raw={order_price_raw:.4f} -> price={order_price:.2f} "
                f"(cap={cap_price2:.2f}, pace_wait={slice_wait_seconds:.2f}s, fill_timeout={fill_timeout_seconds:.2f}s)"
            )

            order_id = self._place_buy_order(stock_code, slice_qty, order_price)
            last_order_id = order_id
            if not order_id:
                self.logger.warning(f"[{stock_code}] 第{attempt_no}片下单失败")
                continue
            attempted_order_ids.append(str(order_id))

            # 等待成交
            filled = self._wait_for_order_fill(
                order_id,
                timeout=fill_timeout_seconds,
                stock_code=stock_code,
                open_px=open_px2,
                initial_state=open_state,
            )
            if filled:
                order_info_after_fill = self._get_order_info(order_id)
                filled_volume = int(order_info_after_fill['traded_volume']) if order_info_after_fill else slice_qty
                filled_volume = max(0, min(slice_qty, filled_volume))
                self.logger.info(f"[{stock_code}] 第{attempt_no}片成交完成: {order_id}, 实际成交={filled_volume}股")
                continue

            if is_fast_down:
                tick_retry = self._get_tick(stock_code)
                if tick_retry is not None:
                    open_px_retry = _safe_float(tick_retry.get("open") or tick_retry.get("openPrice") or open_px2)
                    last_px_retry = _safe_float(tick_retry.get("lastPrice") or last_px2)
                    if open_px_retry and open_px_retry > 0 and last_px_retry and last_px_retry > 0:
                        rel_retry = (last_px_retry - open_px_retry) / (open_px_retry + 1e-12)
                        still_fast_down = rel_retry <= FAST_DOWN_TH
                        continuing_down = last_px_retry <= last_px2 + 1e-12
                        if still_fast_down and continuing_down:
                            self.logger.info(
                                f"[{stock_code}] 第{attempt_no}片仍在快速下跌，追加等待 {FAST_DOWN_EXTRA_WAIT_SECONDS:.2f}s "
                                f"(last {last_px2:.2f}->{last_px_retry:.2f}, rel {rel_retry:.2%})"
                            )
                            filled = self._wait_for_order_fill(
                                order_id,
                                timeout=FAST_DOWN_EXTRA_WAIT_SECONDS,
                                stock_code=stock_code,
                                open_px=open_px_retry,
                                initial_state='FAST_BREAKDOWN',
                            )
                            if filled:
                                order_info_after_fill = self._get_order_info(order_id)
                                filled_volume = int(order_info_after_fill['traded_volume']) if order_info_after_fill else slice_qty
                                filled_volume = max(0, min(slice_qty, filled_volume))
                                self.logger.info(f"[{stock_code}] 第{attempt_no}片在追加等待后成交: {order_id}, 实际成交={filled_volume}股")
                                continue

            # 未成交：撤单并进入下一片（下一片会重新取 tick/重新定价）
            self.logger.warning(f"[{stock_code}] 第{attempt_no}片未在等待窗口内成交，撤单重试")
            self._cancel_order_with_retry(
                order_id,
                max_retry=(1 if is_fast_up else BUY_MAX_RETRIES_PER_SLICE),
                cancel_wait_seconds=(FAST_UP_CANCEL_WAIT_SECONDS if is_fast_up else None),
            )

            # 为避免频繁打单，稍作等待（但仍尽量保持在窗口内）
            time.sleep(0.2)

        final_state = self._wait_for_outstanding_orders(
            stock_code,
            attempted_order_ids,
            timeout=FINAL_OUTSTANDING_WAIT_SECONDS,
        )
        if final_state['outstanding_volume'] > 0:
            self.logger.warning(
                f"[{stock_code}] 结束前仍有未终态订单: outstanding={final_state['outstanding_volume']}, "
                f"pending={final_state['pending_order_ids']}"
            )

        execution_result = self._summarize_buy_execution(
            stock_code=stock_code,
            requested_volume=int(buy_volume),
            attempted_order_ids=attempted_order_ids,
            fallback_buy_date=buy_date,
            last_order_id=last_order_id,
            open_px=open_px,
        )

        if execution_result['filled_volume'] <= 0:
            unfilled = max(0, int(execution_result['requested_volume']) - int(execution_result['filled_volume']))
            self.logger.error(f"❌ {stock_code} 买入未完成，剩余: {unfilled}股（可能因 cap 限制/急涨冲出上限）")
            return None

        self.logger.info(
            f"✅ {stock_code} 买入汇总: requested={execution_result['requested_volume']} "
            f"filled={execution_result['filled_volume']} avg_price={execution_result['avg_price']:.4f} "
            f"fully_filled={execution_result['fully_filled']}"
        )
        if execution_result['filled_volume'] < execution_result['requested_volume']:
            self.logger.warning(
                f"⚠️ {stock_code} 仅部分成交: filled={execution_result['filled_volume']}/"
                f"{execution_result['requested_volume']}"
            )
        return execution_result

    def _detect_fast_down(self,
                          open_px: float,
                          last_px: float,
                          ask1: float,
                          bid1: float,
                          prev_quote: Optional[Dict[str, float]],
                          down_streak: int) -> tuple[bool, int]:
        """
        识别是否正在快速下跌（tick级别的持续下移）。

        核心改动：不再仅凭\"距离开盘价多远\"来判断，而是基于tick级别的
        持续下移信号。一旦tick不再下移，即使股价仍低于开盘价，也会退出
        FAST_BREAKDOWN状态，切换到普通/被动买入。

        两种进入方式：
        1. 持续下移：last/ask1/bid1 连续多tick下移 + 价格低于开盘微阈值
        2. 初始急跌：价格大幅低于开盘 + 当前tick有下移证据
        """
        rel = (last_px - open_px) / (open_px + 1e-12)

        if not prev_quote:
            # 无tick历史：仅用绝对跌幅作为初始信号
            if rel <= FAST_DOWN_TH:
                return True, max(down_streak, FAST_DOWN_MIN_STREAK)
            return False, 0

        prev_last = _safe_float(prev_quote.get('last'))
        prev_ask1 = _safe_float(prev_quote.get('ask1'))
        prev_bid1 = _safe_float(prev_quote.get('bid1'))

        # 先计算tick级别的下移信号（对比当前tick vs 上一tick）
        down_signals = 0
        if pd.notna(prev_last) and last_px < prev_last - DEFAULT_TICK_SIZE / 2:
            down_signals += 1
        if pd.notna(prev_ask1) and ask1 and ask1 > 0 and ask1 <= prev_ask1 - DEFAULT_TICK_SIZE / 2:
            down_signals += 1
        if pd.notna(prev_bid1) and bid1 and bid1 > 0 and bid1 <= prev_bid1 - DEFAULT_TICK_SIZE / 2:
            down_signals += 1

        if down_signals >= 2:
            down_streak += 1
        else:
            # 关键修复：tick没有下移信号→趋势已停，重置streak
            down_streak = 0

        # 维持条件：tick持续下移 + 价格已低于开盘微阈值（确认方向）
        if down_streak >= FAST_DOWN_MIN_STREAK and rel <= FAST_DOWN_MICRO_TH:
            return True, down_streak

        # 进入条件：绝对跌幅较大 + 当前tick有下移证据（捕捉初跌）
        # 此条件只在down_streak=0时触发（即刚开始跌），不会锁定状态
        if rel <= FAST_DOWN_TH and down_signals >= 1:
            return True, max(down_streak + 1, FAST_DOWN_MIN_STREAK)

        return False, down_streak

    def _get_filled_volume_for_orders(self, stock_code: str, attempted_order_ids: List[str]) -> int:
        """汇总指定订单列表的真实成交量，避免超时后晚到成交导致重复下单。"""
        attempted_order_ids = [str(x) for x in attempted_order_ids if x]
        if not attempted_order_ids:
            return 0

        try:
            trades = self.trader.xt_trader.query_stock_trades(self.trader.account)
        except Exception as e:
            self.logger.warning(f"[{stock_code}] 查询实时成交量失败，将回退订单信息: {e}")
            trades = []

        total_volume = 0
        seen_trade_keys = set()

        if trades:
            for trade in trades:
                if trade.stock_code != stock_code:
                    continue
                order_id = str(getattr(trade, 'order_id', ''))
                if order_id not in attempted_order_ids:
                    continue
                trade_id = str(getattr(trade, 'traded_id', order_id))
                trade_key = (order_id, trade_id)
                if trade_key in seen_trade_keys:
                    continue
                seen_trade_keys.add(trade_key)
                total_volume += int(getattr(trade, 'traded_volume', 0) or 0)

        if total_volume > 0:
            return int(total_volume)

        # 若成交回报暂时不可用，则回退到订单对象中的成交量
        fallback_total = 0
        for order_id in attempted_order_ids:
            info = self._get_order_info(order_id)
            if not info:
                continue
            fallback_total += int(info.get('traded_volume', 0) or 0)
        return int(fallback_total)

    def _get_order_execution_totals(self, stock_code: str, attempted_order_ids: List[str]) -> Dict[str, Any]:
        """统计已尝试订单的已成交量与未终态占用量。"""
        attempted_order_ids = [str(x) for x in attempted_order_ids if x]
        if not attempted_order_ids:
            return {
                'filled_volume': 0,
                'outstanding_volume': 0,
                'pending_order_ids': [],
            }

        filled_volume = self._get_filled_volume_for_orders(stock_code, attempted_order_ids)
        outstanding_volume = 0
        pending_order_ids: List[str] = []

        try:
            orders = self.trader.query_stock_orders(self.trader.account)
        except Exception as e:
            self.logger.warning(f"[{stock_code}] 查询订单明细失败，未终态占用量退回active_orders: {e}")
            orders = []

        order_map = {str(order.order_id): order for order in orders if str(order.order_id) in attempted_order_ids}

        for order_id in attempted_order_ids:
            order = order_map.get(order_id)
            if order is not None:
                status = int(getattr(order, 'order_status', -1) or -1)
                order_volume = int(getattr(order, 'order_volume', 0) or 0)
                traded_volume = int(getattr(order, 'traded_volume', 0) or 0)
            else:
                active = self.active_orders.get(order_id, {})
                status = int(active.get('order_status', -1) or -1)
                order_volume = int(active.get('order_volume', 0) or 0)
                traded_volume = int(active.get('traded_volume', 0) or 0)

            if status in OPEN_ORDER_STATUSES:
                open_qty = max(0, order_volume - traded_volume)
                if open_qty > 0:
                    outstanding_volume += open_qty
                    pending_order_ids.append(order_id)

        return {
            'filled_volume': int(filled_volume),
            'outstanding_volume': int(outstanding_volume),
            'pending_order_ids': pending_order_ids,
        }

    def _wait_for_outstanding_orders(self, stock_code: str, attempted_order_ids: List[str], timeout: float = 1.2) -> Dict[str, Any]:
        """等待未终态订单进入终态，避免待撤期间继续重复下单。"""
        start = time.time()
        last_state = self._get_order_execution_totals(stock_code, attempted_order_ids)
        while time.time() - start < timeout:
            if last_state['outstanding_volume'] <= 0:
                return last_state
            time.sleep(0.2)
            last_state = self._get_order_execution_totals(stock_code, attempted_order_ids)
        return last_state

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
        self.last_sell_submission = None

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
            raw_price = float(price)
            price = self._normalize_buy_order_price(raw_price, tick_size=DEFAULT_TICK_SIZE)
            if price <= 0:
                self.logger.error(f"下单价格非法: {stock_code}, raw_price={raw_price}")
                return None
            if abs(price - raw_price) > 1e-9:
                self.logger.info(f"[{stock_code}] 下单价格最终规整: {raw_price:.6f} -> {price:.2f}")

            self.logger.info(f"[{stock_code}] 准备调用 order_stock_async: volume={volume}, price={price:.2f}")

            # 使用固定价格下单（FIX_PRICE）而非最新价。
            # 某些券商/接口在早盘高波动时可能卡住，因此这里加一层本地超时保护，
            # 避免主线程被同步调用永久阻塞。
            result_holder: Dict[str, Any] = {"order_id": None, "error": None}

            def _submit_order() -> None:
                try:
                    result_holder["order_id"] = self.trader.order_stock_async(
                        account=self.trader.account,
                        stock_code=stock_code,
                        order_type=xtconstant.STOCK_BUY,
                        order_volume=volume,
                        price_type=xtconstant.FIX_PRICE,
                        price=price,
                        strategy_name='live_trading_5m',
                        order_remark=stock_code,
                    )
                except Exception as e:
                    result_holder["error"] = e

            thread = threading.Thread(target=_submit_order, daemon=True)
            thread.start()
            thread.join(timeout=5.0)

            if thread.is_alive():
                self.logger.error(f"[{stock_code}] order_stock_async 超时未返回，放弃本次下单")
                return None

            if result_holder["error"] is not None:
                raise result_holder["error"]

            order_id = result_holder["order_id"]

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

    def _summarize_buy_execution(self, stock_code: str,
                                 requested_volume: int,
                                 attempted_order_ids: List[str],
                                 fallback_buy_date: str,
                                 last_order_id: Optional[str],
                                 open_px: float = 0.0) -> Dict[str, Any]:
        """汇总真实成交，避免把部分成交误当成全成。"""
        attempted_order_ids = [str(x) for x in attempted_order_ids if x]
        trade_rows: List[Dict[str, Any]] = []

        try:
            trades = self.trader.xt_trader.query_stock_trades(self.trader.account)
        except Exception as e:
            self.logger.warning(f"[{stock_code}] 查询成交回报失败，将回退到订单信息汇总: {e}")
            trades = []

        if trades:
            seen_trade_ids = set()
            for trade in trades:
                if trade.stock_code != stock_code:
                    continue
                order_id = str(trade.order_id)
                if attempted_order_ids and order_id not in attempted_order_ids:
                    continue
                trade_id = str(getattr(trade, 'traded_id', order_id))
                if trade_id in seen_trade_ids:
                    continue
                seen_trade_ids.add(trade_id)
                traded_volume = int(getattr(trade, 'traded_volume', 0) or 0)
                traded_price = float(getattr(trade, 'traded_price', 0) or 0.0)
                traded_amount = float(getattr(trade, 'traded_amount', traded_price * traded_volume) or 0.0)
                traded_time = getattr(trade, 'traded_time', None)
                trade_rows.append({
                    'order_id': order_id,
                    'trade_id': trade_id,
                    'volume': traded_volume,
                    'price': traded_price,
                    'amount': traded_amount if traded_amount > 0 else traded_price * traded_volume,
                    'trade_time': traded_time,
                })

        total_volume = sum(int(row['volume']) for row in trade_rows)
        total_amount = sum(float(row['amount']) for row in trade_rows)
        avg_price = (total_amount / total_volume) if total_volume > 0 else 0.0

        if total_volume <= 0 and attempted_order_ids:
            fallback_order_infos = [self._get_order_info(order_id) for order_id in attempted_order_ids]
            fallback_order_infos = [info for info in fallback_order_infos if info and int(info.get('traded_volume', 0) or 0) > 0]
            if fallback_order_infos:
                total_volume = sum(int(info['traded_volume']) for info in fallback_order_infos)
                total_amount = sum(float(info['traded_price']) * int(info['traded_volume']) for info in fallback_order_infos)
                avg_price = (total_amount / total_volume) if total_volume > 0 else 0.0

        first_fill_date = self._resolve_first_fill_date(trade_rows, fallback_buy_date)

        return {
            'success': total_volume >= 100,
            'fully_filled': total_volume >= int(requested_volume),
            'stock_code': stock_code,
            'requested_volume': int(requested_volume),
            'filled_volume': int(total_volume),
            'avg_price': float(avg_price),
            'open_price': float(open_px) if open_px else 0.0,
            'first_fill_date': first_fill_date,
            'last_order_id': str(last_order_id) if last_order_id else None,
            'order_ids': attempted_order_ids,
        }

    def _resolve_first_fill_date(self, trade_rows: List[Dict[str, Any]], fallback_buy_date: str) -> str:
        if not trade_rows:
            return fallback_buy_date

        first_trade_dt: Optional[datetime] = None
        for row in trade_rows:
            trade_time = row.get('trade_time')
            if trade_time in (None, '', 0):
                continue
            parsed = self._parse_trade_datetime(trade_time)
            if parsed is None:
                continue
            if first_trade_dt is None or parsed < first_trade_dt:
                first_trade_dt = parsed

        if first_trade_dt is not None:
            return first_trade_dt.strftime('%Y-%m-%d')
        return fallback_buy_date

    def _parse_trade_datetime(self, trade_time: Any) -> Optional[datetime]:
        try:
            text = str(trade_time).strip()
            if not text:
                return None
            if text.isdigit():
                if len(text) >= 14:
                    return datetime.strptime(text[:14], '%Y%m%d%H%M%S')
                if len(text) == 8:
                    return datetime.strptime(text, '%Y%m%d')
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'):
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    pass
            return None
        except Exception:
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
            self.logger.info(f"[{stock_code}] 准备调用 order_stock_async 卖出: volume={volume}")

            result_holder: Dict[str, Any] = {"order_id": None, "error": None}

            def _submit_order() -> None:
                try:
                    result_holder["order_id"] = self.trader.order_stock_async(
                        account=self.trader.account,
                        stock_code=stock_code,
                        order_type=xtconstant.STOCK_SELL,
                        order_volume=volume,
                        price_type=xtconstant.LATEST_PRICE,
                        price=0,
                        strategy_name='live_trading_5m',
                        order_remark=stock_code,
                    )
                except Exception as e:
                    result_holder["error"] = e

            thread = threading.Thread(target=_submit_order, daemon=True)
            thread.start()
            thread.join(timeout=5.0)

            if thread.is_alive():
                self.logger.error(f"[{stock_code}] 卖出 order_stock_async 超时未返回，放弃本次下单")
                return None

            if result_holder["error"] is not None:
                raise result_holder["error"]

            order_id = result_holder["order_id"]

            if order_id and order_id > 0:
                self.logger.info(f"下单成功: {stock_code} 卖出{volume}股, 订单号={order_id}")

                self.last_sell_submission = {
                    'stock_code': stock_code,
                    'order_id': str(order_id),
                    'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'volume': int(volume),
                }

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

    def _wait_for_order_fill(self,
                             order_id: str,
                             timeout: float = 10,
                             stock_code: Optional[str] = None,
                             open_px: Optional[float] = None,
                             initial_state: str = 'NORMAL') -> bool:
        """
        等待订单成交

        Args:
            order_id: 订单号
            timeout: 超时时间（秒）

        Returns:
            bool: 是否成交
        """
        start_time = time.time()
        initial_state = str(initial_state or 'NORMAL')

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
            elif order_status in [52, 55]:
                # 部成/部成待撤不是终态，继续等待真实终态回报，
                # 后续由 outstanding_qty 管理避免重复下单。
                pass

            if stock_code and open_px and open_px > 0:
                should_abort, abort_reason = self._should_abort_waiting_buy_order(
                    stock_code=stock_code,
                    open_px=float(open_px),
                    initial_state=initial_state,
                )
                if should_abort:
                    self.logger.warning(
                        f"等待成交期间盘口状态恶化，提前撤单: {order_id}, state={initial_state}, {abort_reason}"
                    )
                    if order_id in self.active_orders:
                        self.active_orders[order_id]['need_cancel'] = True
                    return False

            # 轮询频率要快于开盘前30秒内的订单状态变化
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                break
            time.sleep(min(0.2, max(0.05, remaining)))

        # 超时
        self.logger.warning(f"等待订单成交超时: {order_id}")
        return False

    def _should_abort_waiting_buy_order(self,
                                        stock_code: str,
                                        open_px: float,
                                        initial_state: str) -> tuple[bool, str]:
        """等待成交期间发现盘口明显转坏时，提前撤掉未成交买单。"""
        try:
            tick = self._get_tick(stock_code)
            if not tick:
                return False, ''
            last_px = _safe_float(tick.get("lastPrice"))
            ask1 = _safe_float((tick.get("askPrice") or [None])[0])
            bid1 = _safe_float((tick.get("bidPrice") or [None])[0])
            if not last_px or last_px <= 0:
                return False, ''

            rel = (last_px - open_px) / (open_px + 1e-12)
            spread_rel = 0.0
            if ask1 and ask1 > 0 and bid1 and bid1 > 0:
                spread_rel = max(0.0, (ask1 - bid1) / (open_px + 1e-12))

            if rel <= FAST_DOWN_TH:
                return True, f"rel={rel:.2%} <= FAST_DOWN_TH({FAST_DOWN_TH:.2%})"

            if initial_state in {'FAST_BREAKOUT', 'WHIPSAW'} and rel <= FAST_DOWN_MICRO_TH:
                return True, f"state={initial_state} reversed to rel={rel:.2%}"

            if initial_state == 'FAST_BREAKOUT' and rel <= 0 and spread_rel >= 0.003:
                return True, f"breakout failed: rel={rel:.2%}, spread={spread_rel:.2%}"

            return False, ''
        except Exception as e:
            self.logger.debug(f"等待成交期间盘口检查失败: {e}")
            return False, ''

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
                        'order_volume': int(getattr(order, 'order_volume', 0) or 0),
                        'traded_volume': order.traded_volume,
                        'traded_price': order.traded_price,
                        'order_price': float(getattr(order, 'price', 0) or 0.0),
                    }

            return None

        except Exception as e:
            self.logger.error(f"获取订单信息失败: {e}")
            return None

    def _cancel_order_with_retry(self, order_id: str,
                                max_retry: int = 3,
                                cancel_wait_seconds: Optional[float] = None) -> bool:
        """
        撤单（带重试）

        Args:
            order_id: 订单号
            max_retry: 最大重试次数

        Returns:
            bool: 是否成功
        """
        wait_seconds = float(cancel_wait_seconds) if cancel_wait_seconds is not None else float(self.CANCEL_WAIT_SECONDS)

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
                time.sleep(wait_seconds)

                # 检查撤单结果
                order_status = self._query_order_status(order_id)
                if order_status in [53, 54, 56, 57]:  # 终态：已成、已撤、废单
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
                                prev_close: Optional[float], buy_open: float,
                                target_rank: int = 1,
                                is_weak_market_env: bool = False) -> Dict:
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

        if int(target_rank) == 1:
            gap_down_threshold = StrategyConfig.GAP_DOWN_THRESHOLD_TOP1
        elif int(target_rank) == 2 and is_weak_market_env:
            gap_down_threshold = StrategyConfig.GAP_DOWN_THRESHOLD_WEAK_TOP2
        elif int(target_rank) == 2:
            gap_down_threshold = StrategyConfig.GAP_DOWN_THRESHOLD_STRONG_TOP2
        else:
            gap_down_threshold = StrategyConfig.GAP_DOWN_THRESHOLD_NON_TOP1
        gap_up_threshold = StrategyConfig.GAP_UP_THRESHOLD
        self.logger.info(
            f"  风控阈值: rank={int(target_rank)}, weak_env={is_weak_market_env}, "
            f"gap_down={gap_down_threshold:.2%}, gap_up={gap_up_threshold:.2%}"
        )

        # 风控1: 跌空超过阈值
        if gap_ratio < gap_down_threshold:
            self.logger.warning(
                f"  🚫 跌空超过阈值 {gap_down_threshold:.2%} ({gap_ratio:.2%})，禁止买入"
            )
            return {
                'can_buy': False,
                'reason': f'跌空{-gap_ratio:.2%}',
                'skip_category': '开盘风控'
            }

        # 风控2: 高开超过阈值
        elif gap_ratio > gap_up_threshold:
            self.logger.warning(
                f"  🚫 高开超过阈值 {gap_up_threshold:.2%} ({gap_ratio:.2%})，禁止买入"
            )
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
