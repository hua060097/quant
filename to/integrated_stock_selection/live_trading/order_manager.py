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
import os

# =============================================================================
# 交易参数（写死在代码内，不再依赖 self.config）
# =============================================================================
# 注意：你给的 Tushare token 建议放环境变量 TUSHARE_TOKEN；这里留作兜底
TUSHARE_TOKEN = "022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq"

# =========================
# Fast move detection
# =========================
# 两次取 tick 的采样间隔（秒）
# 普通 MiniQMT + 非L2 情况下，0.25~0.50s 是相对稳的区间
FAST_SAMPLE_INTERVAL_SEC = 0.35

# 买入窗口
# 原先 20 秒过短：遇到急跌往往来不及“等一等”，且普通 MiniQMT 回报/撤单存在延迟。
# 这里默认放宽到 3 分钟：09:30:00~09:33:00（急涨会自动加速，急跌会自动放慢）。
BUY_WINDOW_START = "09:30:00"
BUY_WINDOW_END = "09:33:00"

# 分片买入与等待（普通 MiniQMT 权限，回报/撤单有延迟；不宜过短）
BUY_SLICES = 4
BUY_WAIT_SECONDS = 1.0          # 单片等待成交时间
BUY_CANCEL_RETRY = 2            # 撤单重试次数
BUY_SLICE_GAP_SECONDS = 0.35    # 常态片间隔（急涨会缩短，急跌会拉长）

# “急涨/急跌”检测：用短间隔二次采样（更接近你对“加速度/短时趋势”的直觉）
FAST_SAMPLE_INTERVAL = 0.25     # 秒：两次取价间隔
FAST_UP_PER_SEC_TH = 0.010      # 每秒涨幅阈值：>1.0%/s 认为急涨
FAST_DOWN_PER_SEC_TH = -0.010   # 每秒跌幅阈值：<-1.0%/s 认为急跌

# 急跌时放慢节奏（让价格“走出来”），急涨时加速完成
FAST_DOWN_GAP_SECONDS = 1.20    # 急跌片间隔
FAST_UP_GAP_SECONDS = 0.15      # 急涨片间隔

# 追价上限：严格不超过 open*(1+cap)（同时也不超过涨停价）
BUY_SLIPPAGE_CAP_PCT = 0.010    # 1.0% 以内允许“有限追价”

# 急涨/急跌判定阈值（相对开盘价）
FAST_UP_TH = 0.006              # +0.6% 认为急涨
FAST_DOWN_TH = -0.006           # -0.6% 认为急跌

# 急跌时：前段更被动（靠近 bid1），后段逐步靠近 ask1 保证成交
FALL_PASSIVE_RATIO = 0.50

# 常态下为了提高成交概率，可在 ask1 基础上小追 1tick（但不超过 cap）
BUY_SMALL_PREMIUM_TICKS = 1

# 资金/费用保守预留（防止“可用资金不足”废单）
CASH_FEE_BUFFER = 0.0015        # 0.15%

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
        # --- Tushare: 获取涨停价（用于限价上限），做简单 cache，避免频繁请求 ---
        self._tushare_limit_cache: dict[tuple[str, str], float] = {}
        self._ts_pro = None


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


    @staticmethod
    def _safe_float(x):
        try:
            import numpy as np
            if x is None:
                return None
            if isinstance(x, (list, tuple)) and len(x) > 0:
                x = x[0]
            v = float(x)
            if not np.isfinite(v):
                return None
            return v
        except Exception:
            return None

    @staticmethod
    def _tick_size(price: float) -> float:
        # A股最小变动价位一般为 0.01
        return 0.01

    @classmethod
    def _round_price(cls, price: float) -> float:
        ts = cls._tick_size(price)
        return round(price / ts) * ts

    def _get_tick(self, stock_code: str) -> dict | None:
        """从 xtdata.get_full_tick 获取单个股票的 tick dict。"""
        try:
            from xtquant import xtdata
            tick_all = xtdata.get_full_tick([stock_code])
            if not tick_all or stock_code not in tick_all:
                return None
            return tick_all[stock_code]
        except Exception as e:
            self.logger.warning(f"[{stock_code}] 获取 tick 异常: {e}")
            return None

    def _get_up_limit_price(self, stock_code: str, trade_date: str, prev_close: float | None = None) -> float | None:
        """获取当日涨停价：优先 Tushare daily_limit；失败用 prev_close * (1+涨跌幅限制) 兜底。
        trade_date: 'YYYY-MM-DD' 或 'YYYYMMDD'
        """
        import os, math
        import pandas as pd

        ts_code = str(stock_code)
        td = str(trade_date).replace("-", "")
        key = (ts_code, td)
        if key in self._tushare_limit_cache:
            return self._tushare_limit_cache[key]

        up = None

        # 允许环境变量覆盖；否则使用模块常量
        token = os.getenv("TUSHARE_TOKEN", "").strip() or TUSHARE_TOKEN
        token = (token or "").strip()

        if token:
            try:
                import tushare as ts
                if self._ts_pro is None:
                    self._ts_pro = ts.pro_api(token)
                df = self._ts_pro.daily_limit(ts_code=ts_code, trade_date=td)
                if df is not None and len(df) > 0 and "up_limit" in df.columns:
                    up = float(df.iloc[0]["up_limit"])
                    if not math.isfinite(up):
                        up = None
            except Exception as e:
                self.logger.warning(f"[{stock_code}] Tushare daily_limit 获取失败({td}): {e}")

        if up is None and prev_close is not None and pd.notna(prev_close):
            code = ts_code.split(".")[0]
            is_20 = code.startswith("30") or code.startswith("688")
            limit_pct = 0.20 if is_20 else 0.10
            up = float(prev_close) * (1.0 + limit_pct)

        if up is not None:
            self._tushare_limit_cache[key] = up
        return up


    
    def smart_buy_full_position(self, stock_code: str) -> Optional[str]:
        """开盘 09:30:00~09:30:20 智能买入（含急涨/急跌逻辑 + 追价上限 + 涨停价约束）。

        目标：
        - 急涨：允许“小追”，但严格不超过 open*(1+BUY_SLIPPAGE_CAP_PCT) 且不越过涨停价；
        - 急跌：窗口前段更被动（尽量靠近 bid1），窗口后段再提高成交概率；
        - 分片下单 + 撤单重试，尽量减少开盘剧烈波动下的滑点。
        """
        self.logger.info("=" * 60)
        self.logger.info(f"开始全仓买入: {stock_code}")
        self.logger.info("=" * 60)

        # 等待到 09:30:00
        self._wait_until_market_open()

        buy_date = datetime.now().strftime("%Y-%m-%d")

        # 获取资金与前收
        asset = self.order_manager.trader.query_stock_asset(self.order_manager.trader.account)
        if not asset:
            self.logger.error("查询账户资金失败")
            return None
        available_cash = float(asset.cash)
        self.logger.info(f"可用资金: {available_cash:.2f}元")

        prev_close = self._get_prev_close_price(stock_code)
        if prev_close and prev_close > 0:
            self.logger.info(f"  前收盘: {prev_close:.2f}")
        else:
            self.logger.warning(f"[{stock_code}] prev_close 获取失败，后续涨停价仅能用板块涨跌幅估算兜底")

        # 取开盘 tick
        tick0 = self._get_tick(stock_code)
        if tick0 is None:
            self.logger.error(f"[{stock_code}] 无法获取开盘 tick")
            return None

        open_px = self._safe_float(tick0.get("open") or tick0.get("openPrice"))
        last_px = self._safe_float(tick0.get("lastPrice") or tick0.get("last") or tick0.get("close"))
        ask1 = self._safe_float((tick0.get("askPrice") or [None])[0])
        bid1 = self._safe_float((tick0.get("bidPrice") or [None])[0])

        if not open_px or open_px <= 0:
            open_px = last_px
        if not open_px or open_px <= 0:
            self.logger.error(f"[{stock_code}] open/last 均无效，放弃买入")
            return None

        # 开盘风控（保持你原有逻辑：基于 prev_close 与 open 的 gap）
        if prev_close and prev_close > 0:
            risk_check = self._check_opening_gap_risk(stock_code, buy_date, prev_close, open_px)
            if not risk_check.get("can_buy", True):
                self.logger.warning(f"[{stock_code}] 未通过开盘风控: {risk_check.get('reason')}，取消买入")
                return None

        # 涨停价（Tushare 优先；失败兜底）
        up_limit = self._get_up_limit_price(stock_code, buy_date, prev_close=prev_close)

        # 追价上限（open cap，同时不越过涨停）
        cap_price = float(open_px) * (1.0 + BUY_SLIPPAGE_CAP_PCT)
        if up_limit is not None and up_limit > 0:
            cap_price = min(cap_price, float(up_limit))

        self.logger.info(
            f"[{stock_code}] open={open_px:.2f} last={(last_px or open_px):.2f} "
            f"ask1={(ask1 or float('nan')):.2f} bid1={(bid1 or float('nan')):.2f} "
            f"cap={cap_price:.2f} up_limit={(up_limit if up_limit else float('nan')):.2f}"
        )

        # 估算可以买多少（以 cap 与盘口较大者做保守 sizing）
        ref_px = ask1 if (ask1 and ask1 > 0) else (last_px if (last_px and last_px > 0) else open_px)
        ref_px = min(ref_px, cap_price)
        buy_volume = self._calculate_buy_volume_full(available_cash, ref_px)
        if buy_volume < 100:
            self.logger.error(f"资金不足，无法买入（buy_volume={buy_volume}）")
            return None

        self.logger.info(f"[{stock_code}] 计划买入数量: {buy_volume}股，分 {BUY_SLICES} 片，单片等待 {BUY_WAIT_SECONDS}s")

        # 窗口设置
        win_start = datetime.strptime(BUY_WINDOW_START, "%H:%M:%S").time()
        win_end = datetime.strptime(BUY_WINDOW_END, "%H:%M:%S").time()

        remaining = int(buy_volume)
        last_order_id: Optional[str] = None

        for s in range(1, BUY_SLICES + 1):
            if remaining < 100:
                break

            now_t = datetime.now().time()
            in_window = (now_t >= win_start) and (now_t <= win_end)

            # 动态更新资金（避免“可用资金不足”废单）
            asset = self.order_manager.trader.query_stock_asset(self.order_manager.trader.account)
            cash_left = float(asset.cash) if asset else 0.0

            # 每片数量：先按“剩余/剩余片数”做均分（手数对齐），后续会根据急涨/急跌动态调整
            remaining_slices = max(1, BUY_SLICES - s + 1)
            base_qty = max(100, int(remaining // remaining_slices // 100 * 100))
            slice_qty = min(base_qty, remaining)

            tick = self._get_tick(stock_code)
            if tick is None:
                self.logger.warning(f"[{stock_code}] 获取 tick 失败，跳过第{s}片")
                continue

            open_px2 = self._safe_float(tick.get("open") or tick.get("openPrice")) or open_px
            last_px2 = self._safe_float(tick.get("lastPrice") or tick.get("last") or tick.get("close")) or last_px or open_px
            ask1_2 = self._safe_float((tick.get("askPrice") or [None])[0])
            bid1_2 = self._safe_float((tick.get("bidPrice") or [None])[0])

            cap_price2 = float(open_px2) * (1.0 + BUY_SLIPPAGE_CAP_PCT)
            if up_limit is not None and up_limit > 0:
                cap_price2 = min(cap_price2, float(up_limit))

            # 窗口进度（用于急跌策略）
            try:
                now_dt = datetime.now()
                start_dt = now_dt.replace(hour=win_start.hour, minute=win_start.minute, second=win_start.second, microsecond=0)
                end_dt = now_dt.replace(hour=win_end.hour, minute=win_end.minute, second=win_end.second, microsecond=0)
                prog = (now_dt - start_dt).total_seconds() / max(1e-6, (end_dt - start_dt).total_seconds()) if end_dt > start_dt else 1.0
                prog = float(min(1.0, max(0.0, prog)))
            except Exception:
                prog = 1.0

            # -------------------------
            # 急涨/急跌判定：用短间隔两次 tick 的“瞬时斜率”做主判据
            # last-open 只能描述“累计涨跌”，并不能刻画“此刻在加速/减速”。
            # 对普通 MiniQMT（无 L2）而言，短采样是最稳妥的替代方案。
            # -------------------------
            rel_open = (float(last_px2) - float(open_px2)) / (float(open_px2) + 1e-12)

            p0 = float(last_px2) if last_px2 and last_px2 > 0 else None
            rps = 0.0  # return per second
            try:
                if p0 is not None and FAST_SAMPLE_INTERVAL_SEC > 0:
                    time.sleep(float(FAST_SAMPLE_INTERVAL_SEC))
                    t2 = self._get_tick(stock_code)
                    if t2:
                        last_px3 = self._safe_float(t2.get("lastPrice") or t2.get("last") or last_px2)
                        if last_px3 and last_px3 > 0:
                            rps = (float(last_px3) - p0) / p0 / float(FAST_SAMPLE_INTERVAL_SEC)
                            # 用第二次采样更新盘口（后续定价用更“近”的 tick）
                            last_px2 = last_px3
                            ask1_2 = self._safe_float((t2.get("askPrice") or [ask1_2])[0])
                            bid1_2 = self._safe_float((t2.get("bidPrice") or [bid1_2])[0])
            except Exception:
                rps = 0.0

            is_fast_up = (rps >= FAST_UP_PER_SEC_TH) or (rel_open >= FAST_UP_TH and rps > 0)
            is_fast_down = (rps <= FAST_DOWN_PER_SEC_TH) or (rel_open <= FAST_DOWN_TH and rps < 0)

            # 动态调整本片下单数量：
            # - 急涨：前置更多仓位，避免“慢吞吞买不到”
            # - 急跌：降低单片数量，给“更低价成交”留余地
            if is_fast_up:
                slice_qty = int(min(remaining, max(100, (slice_qty * 1.5) // 100 * 100)))
            elif is_fast_down:
                # 越靠近窗口前段越慢买
                slow_factor = 0.6 if prog <= 0.6 else 0.8
                slice_qty = int(min(remaining, max(100, (slice_qty * slow_factor) // 100 * 100)))

            # 计算下单价
            if is_fast_up:
                base = ask1_2 if (ask1_2 and ask1_2 > 0) else last_px2
                order_price = min(base, cap_price2)
            elif is_fast_down:
                if prog <= FALL_PASSIVE_RATIO and in_window:
                    base = bid1_2 if (bid1_2 and bid1_2 > 0) else last_px2
                    order_price = min(base, cap_price2)
                else:
                    base = ask1_2 if (ask1_2 and ask1_2 > 0) else last_px2
                    order_price = min(base, cap_price2)
            else:
                base = ask1_2 if (ask1_2 and ask1_2 > 0) else last_px2
                order_price = min(base + BUY_SMALL_PREMIUM_TICKS * self._tick_size(base), cap_price2)

            order_price = float(self._round_price(order_price))
            order_price = min(order_price, cap_price2)
            if up_limit is not None and up_limit > 0:
                order_price = min(order_price, float(up_limit))

            # 资金上限校验（保守扣费）
            max_afford = int((cash_left / (order_price * (1.0 + CASH_FEE_BUFFER))) // 100 * 100) if cash_left > 0 else 0
            slice_qty = min(slice_qty, max_afford, remaining)
            slice_qty = int(slice_qty // 100 * 100)

            if slice_qty < 100:
                self.logger.warning(f"[{stock_code}] 资金不足以继续买入（cash_left={cash_left:.2f}），停止。剩余={remaining}")
                break

            self.logger.info(
                f"[{stock_code}] 第{s}/{BUY_SLICES}片 | qty={slice_qty} | rel={rel_open:.2%} rps={rps:.2%}/s | "
                f"fast_up={is_fast_up} fast_down={is_fast_down} | price={order_price:.2f} (cap={cap_price2:.2f})"
            )

            # 动态等待间隔：急涨更密，急跌更疏；常态走基准间隔。
            gap_sec = BUY_SLICE_GAP_SECONDS
            if is_fast_up:
                gap_sec = max(0.15, BUY_SLICE_GAP_SECONDS * 0.35)
            elif is_fast_down:
                gap_sec = BUY_SLICE_GAP_SECONDS * (2.5 if prog < 0.6 else 1.6)
            gap_sec = float(min(3.0, max(0.05, gap_sec)))

            order_id = self._place_buy_order(stock_code, slice_qty, order_price)
            last_order_id = order_id
            if not order_id:
                self.logger.warning(f"[{stock_code}] 第{s}片下单失败")
                time.sleep(gap_sec)
                continue

            if self._wait_for_order_fill(order_id, timeout=BUY_WAIT_SECONDS):
                self.logger.info(f"[{stock_code}] 第{s}片成交完成: {order_id}")
                remaining -= slice_qty
                time.sleep(gap_sec)
                continue

            self.logger.warning(f"[{stock_code}] 第{s}片未在 {BUY_WAIT_SECONDS}s 内成交，撤单重试")
            self._cancel_order_with_retry(order_id, max_retries=BUY_CANCEL_RETRY)
            time.sleep(gap_sec)

        if remaining <= 0:
            self.logger.info(f"✅ {stock_code} 买入完成，最后订单: {last_order_id}")
            return last_order_id

        self.logger.error(f"❌ {stock_code} 买入未完成，剩余: {remaining}股（可能因 cap 限制/急涨冲出上限/资金不足）")
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
