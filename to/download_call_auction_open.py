"""
下载 Tushare 股票开盘集合竞价数据（stk_auction_o），按交易日保存为 parquet。

特性：
1. start_date / end_date 参数输入
2. 自动从同目录 download_daily.py 复用 token
3. 保存到 local_data/call_auction_open/
4. 按交易日一个 parquet
5. 支持增量更新：只影响请求区间内日期
6. 请求区间内默认覆盖旧文件（新数据覆盖旧数据）
7. 受控并发，避免把 Tushare 限频打爆
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import tushare as ts

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "local_data" / "call_auction_open"
LOG_DIR = BASE_DIR / "to" / "log"
TOKEN_SOURCE = BASE_DIR / "download_daily.py"
MAX_WORKERS = 3
RETRY_COUNT = 6


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("download_tushare_call_auction_open")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(LOG_DIR / "download_tushare_call_auction_open.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger()


def load_token() -> str:
    if not TOKEN_SOURCE.exists():
        raise FileNotFoundError(f"未找到 token 源文件: {TOKEN_SOURCE}")
    text = TOKEN_SOURCE.read_text(encoding="utf-8")
    patterns = [
        r"ts\.set_token\('([^']+)'\)",
        r'ts\.set_token\("([^"]+)"\)',
        r"set_token\('([^']+)'\)",
        r'set_token\("([^"]+)"\)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            token = m.group(1).strip()
            if token:
                return token
    raise ValueError("未能从 download_daily.py 中解析出 tushare token")


def normalize_date(date_str: str) -> str:
    s = str(date_str).replace("-", "").strip()
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"非法日期格式: {date_str}，应为 YYYYMMDD 或 YYYY-MM-DD")
    return s


def get_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = pro.trade_cal(exchange="", start_date=start_date, end_date=end_date, is_open="1")
    if cal is None or cal.empty:
        return []
    return sorted(cal["cal_date"].astype(str).unique().tolist())


def retry_fetch(func):
    def wrapper(*args, **kwargs):
        backoff = 1.0
        for i in range(RETRY_COUNT):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                if "每分钟最多访问" in msg or "frequency" in msg.lower():
                    sleep_s = 60 + random.random() * 5
                    logger.warning(f"触发 Tushare 分钟限频，休眠 {sleep_s:.1f}s 后重试，第 {i+1} 次")
                    time.sleep(sleep_s)
                    backoff = 1.0
                elif "每小时" in msg:
                    sleep_s = 3600
                    logger.warning(f"触发 Tushare 小时限频，休眠 {sleep_s}s 后重试，第 {i+1} 次")
                    time.sleep(sleep_s)
                else:
                    sleep_s = backoff + random.random()
                    logger.warning(f"请求异常: {e}，{sleep_s:.1f}s 后重试，第 {i+1} 次")
                    time.sleep(sleep_s)
                    backoff = min(backoff * 2, 30)
        raise RuntimeError("重试耗尽，接口仍失败")
    return wrapper


@retry_fetch
def fetch_trade_date(token: str, trade_date: str) -> pd.DataFrame:
    # 每个线程单独初始化 pro，避免共享 client 状态。
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.stk_auction_o(trade_date=trade_date)
    if df is None:
        return pd.DataFrame()
    return df.copy()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close", "open", "high", "low", "vol", "amount", "vwap"])
    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    numeric_cols = ["close", "open", "high", "low", "vol", "amount", "vwap"]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["trade_date", "ts_code"]).drop_duplicates(["trade_date", "ts_code"], keep="last")
    return out.reset_index(drop=True)


def save_day(df: pd.DataFrame, trade_date: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = OUTPUT_DIR / f"{trade_date}.parquet"
    tmp_path = OUTPUT_DIR / f"{trade_date}.tmp.parquet"
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(final_path)


def iter_dates_for_update(trade_dates: Iterable[str], overwrite: bool) -> list[str]:
    if overwrite:
        return list(trade_dates)
    todo = []
    for d in trade_dates:
        if not (OUTPUT_DIR / f"{d}.parquet").exists():
            todo.append(d)
    return todo


def download_range(start_date: str, end_date: str, overwrite: bool = True, max_workers: int = MAX_WORKERS) -> None:
    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    if start_date > end_date:
        raise ValueError("start_date 不能大于 end_date")

    token = load_token()
    ts.set_token(token)
    pro = ts.pro_api()
    trade_dates = get_trade_dates(pro, start_date, end_date)
    if not trade_dates:
        logger.warning("给定区间内没有开放交易日")
        return

    todo_dates = iter_dates_for_update(trade_dates, overwrite=overwrite)
    logger.info("=" * 80)
    logger.info("开始下载股票开盘集合竞价数据 stk_auction_o")
    logger.info(f"区间: {start_date} ~ {end_date}")
    logger.info(f"交易日总数: {len(trade_dates)}")
    logger.info(f"待处理交易日: {len(todo_dates)}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"覆盖旧文件: {overwrite}")
    logger.info(f"并发线程数: {max_workers}")
    logger.info("=" * 80)

    success_dates: list[str] = []
    failed_dates: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(fetch_trade_date, token, d): d for d in todo_dates}
        for future in as_completed(future_map):
            d = future_map[future]
            try:
                raw = future.result()
                out = transform(raw)
                save_day(out, d)
                success_dates.append(d)
                logger.info(f"完成 {d}: {len(out):,} 行")
            except Exception as e:
                failed_dates.append(d)
                logger.error(f"失败 {d}: {e}")

    logger.info("=" * 80)
    logger.info(f"下载完成，成功 {len(success_dates)} 天，失败 {len(failed_dates)} 天")
    if failed_dates:
        logger.warning(f"失败日期: {failed_dates}")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 Tushare 股票开盘集合竞价数据（stk_auction_o）")
    parser.add_argument("--start", required=True, help="开始日期，YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期，YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="并发线程数，默认 3")
    parser.add_argument("--skip-existing", action="store_true", help="若指定，则已存在日期不覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = max(1, min(int(args.max_workers), 4))
    # 这里人为压到 4 以内，避免在 Tushare 上无意义地堆线程。
    download_range(args.start, args.end, overwrite=not args.skip_existing, max_workers=workers)


if __name__ == "__main__":
    main()
