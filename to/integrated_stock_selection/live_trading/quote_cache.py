"""
xtdata whole-quote callback cache.

The quote callback is the only place that mutates realtime tick/bar state.
Trading and monitoring code read snapshots from this cache, so they no longer
depend on repeated get_full_tick polling.
"""

import logging
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional


class RealtimeQuoteCache:
    """Cache latest whole-quote ticks and aggregate realtime 5-minute bars."""

    _run_lock = threading.Lock()
    _run_started = False

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._subscribed_codes = set()
        self._latest_ticks: Dict[str, dict] = {}
        self._latest_update_dt: Dict[str, datetime] = {}
        self._bar_state: Dict[str, dict] = {}
        self._finalized_bars: Dict[str, Dict[str, dict]] = {}
        self._subscribe_error_logged = False

    def subscribe(self, stock_codes: Iterable[str]) -> bool:
        """Subscribe whole-quote callback for the requested stock codes."""
        codes = sorted({str(code).strip() for code in stock_codes if str(code).strip()})
        new_codes = [code for code in codes if code not in self._subscribed_codes]
        if not new_codes:
            self._start_xtdata_run_loop()
            return True

        try:
            from xtquant import xtdata

            xtdata.enable_hello = False
            seq = xtdata.subscribe_whole_quote(new_codes, callback=self._on_quote)
            with self._lock:
                self._subscribed_codes.update(new_codes)
            self.logger.info(f"✅ xtdata全推行情订阅: {new_codes}, seq={seq}")
            self._start_xtdata_run_loop()
            return True
        except Exception as e:
            if not self._subscribe_error_logged:
                self.logger.warning(f"xtdata全推行情订阅失败: {new_codes}, {e}")
                self._subscribe_error_logged = True
            return False

    def _start_xtdata_run_loop(self):
        """Start xtdata.run once in a daemon thread.

        xtdata.run is a blocking loop; running it in the foreground would stop
        the rest of the live-trading workflow.
        """
        with self._run_lock:
            if self.__class__._run_started:
                return

            def _runner():
                try:
                    from xtquant import xtdata

                    self.logger.info("xtdata.run行情回调循环启动")
                    xtdata.run()
                except Exception as e:
                    self.logger.error(f"xtdata.run行情回调循环退出: {e}")

            thread = threading.Thread(target=_runner, name="xtdata-whole-quote-run", daemon=True)
            thread.start()
            self.__class__._run_started = True

    def wait_first_tick(self, stock_code: str, timeout_seconds: float = 3.0) -> Optional[dict]:
        """Wait briefly for the first callback tick after subscription."""
        deadline = time.time() + max(float(timeout_seconds), 0.0)
        while time.time() <= deadline:
            tick = self.get_tick(stock_code, max_age_seconds=None)
            if tick:
                return tick
            time.sleep(0.1)
        return None

    def get_tick(self, stock_code: str, max_age_seconds: Optional[float] = 15.0) -> Optional[dict]:
        """Return a copy of latest tick if present and fresh enough."""
        code = str(stock_code).strip()
        with self._lock:
            tick = self._latest_ticks.get(code)
            updated_at = self._latest_update_dt.get(code)
            if not tick:
                return None
            if max_age_seconds is not None and updated_at is not None:
                age = (datetime.now() - updated_at).total_seconds()
                if age > max_age_seconds:
                    return None
            return deepcopy(tick)

    def get_current_price(self, stock_code: str, max_age_seconds: Optional[float] = 15.0) -> Optional[float]:
        tick = self.get_tick(stock_code, max_age_seconds=max_age_seconds)
        if not tick:
            return None
        price = tick.get("lastPrice")
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None
        return price if price > 0 else None

    def get_realtime_5m_bar(self, stock_code: str, close_time_str: str) -> Optional[dict]:
        """Return finalized 5-minute bar ending at close_time_str."""
        code = str(stock_code).strip()
        with self._lock:
            bar = self._finalized_bars.get(code, {}).get(close_time_str)
            if bar:
                return deepcopy(bar)
            state = self._bar_state.get(code)
            if state and state.get("time") == close_time_str:
                return deepcopy(state.get("bar"))
            return None

    def get_today_open(self, stock_code: str, max_age_seconds: Optional[float] = 60.0) -> Optional[float]:
        tick = self.get_tick(stock_code, max_age_seconds=max_age_seconds)
        if not tick:
            return None
        open_price = tick.get("open") or tick.get("openPrice")
        try:
            open_price = float(open_price)
        except (TypeError, ValueError):
            return None
        return open_price if open_price > 0 else None

    def _on_quote(self, datas):
        """xtdata whole-quote callback. All realtime state updates happen here."""
        if not isinstance(datas, dict):
            return

        now = datetime.now()
        with self._lock:
            for stock_code, tick in datas.items():
                if not isinstance(tick, dict):
                    continue

                code = str(stock_code).strip()
                self._latest_ticks[code] = dict(tick)
                self._latest_update_dt[code] = now

                price = self._safe_float(tick.get("lastPrice"))
                if price is None or price <= 0:
                    continue

                tick_dt = self._tick_datetime(tick, now)
                self._update_5m_bar(code, price, tick_dt)

    def _update_5m_bar(self, stock_code: str, price: float, tick_dt: datetime):
        close_dt = self._bar_close_datetime(tick_dt)
        close_time_str = close_dt.strftime("%Y%m%d %H:%M:00")

        state = self._bar_state.get(stock_code)
        if state and state["time"] != close_time_str:
            self._finalized_bars.setdefault(stock_code, {})[state["time"]] = dict(state["bar"])

        if not state or state["time"] != close_time_str:
            state = {
                "time": close_time_str,
                "bar": {
                    "time": close_time_str,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                },
            }
            self._bar_state[stock_code] = state
        else:
            bar = state["bar"]
            bar["high"] = max(float(bar["high"]), price)
            bar["low"] = min(float(bar["low"]), price)
            bar["close"] = price

        # At the exact boundary, the current interval has closed. Keep a copy
        # so monitor code can read the bar immediately after the boundary.
        if tick_dt.replace(second=0, microsecond=0) >= close_dt:
            self._finalized_bars.setdefault(stock_code, {})[close_time_str] = dict(state["bar"])

    @staticmethod
    def _bar_close_datetime(tick_dt: datetime) -> datetime:
        base = tick_dt.replace(second=0, microsecond=0)
        minute_mod = base.minute % 5
        if minute_mod == 0 and tick_dt.second == 0 and tick_dt.microsecond == 0:
            return base
        add_minutes = 5 - minute_mod if minute_mod else 5
        return base + timedelta(minutes=add_minutes)

    @staticmethod
    def _tick_datetime(tick: dict, fallback: datetime) -> datetime:
        tick_time = tick.get("time")
        if tick_time:
            try:
                text = str(tick_time).strip()
                if len(text) == 14 and text.isdigit():
                    return datetime.strptime(text, "%Y%m%d%H%M%S")
                ts = float(text)
                if ts > 10_000_000_000:
                    ts /= 1000.0
                return datetime.fromtimestamp(ts)
            except (TypeError, ValueError, OSError):
                pass
        return fallback

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
