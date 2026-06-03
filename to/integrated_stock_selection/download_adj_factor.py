#!/usr/bin/env python3
"""
Download Tushare adj_factor data and save it under ../local_data/adj_factor.

The script downloads by trade_date, which is much cheaper than stock-by-stock
for building exclusion windows around ex-right/ex-dividend dates.

Default output:
    ../local_data/adj_factor/adj_factor_YYYYMM.parquet
    ../local_data/adj_factor/adj_factor_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import tushare as ts


TUSHARE_TOKEN = "gg99b01a244311f7742512463d9e1588954d174b86d695f8a9f1cc4b"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "local_data" / "adj_factor"
DEFAULT_START_DATE = "20230601"
DEFAULT_END_DATE = "20241031"


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait + random.random() * 0.03)
            self._last_call = time.time()


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("download_adj_factor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Download Tushare adj_factor by trade date.")
    parser.add_argument("--start_date", default=DEFAULT_START_DATE, help="YYYYMMDD")
    parser.add_argument("--end_date", default=DEFAULT_END_DATE, help="YYYYMMDD")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="保存目录")
    parser.add_argument("--workers", type=int, default=6, help="并发线程数，1万积分账号默认6")
    parser.add_argument("--api_interval", type=float, default=0.12, help="全局API最小间隔秒数")
    parser.add_argument("--retry", type=int, default=8, help="每个交易日最大重试次数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在月份文件")
    parser.add_argument("--calendar_file", default="", help="本地交易日历parquet，可选")
    return parser.parse_args()


def normalize_date_str(value) -> str:
    return pd.to_datetime(str(value)).strftime("%Y%m%d")


def load_local_calendar(calendar_file: Optional[str], start_date: str, end_date: str) -> Optional[List[str]]:
    candidates = []
    if calendar_file:
        candidates.append(Path(calendar_file))
    root = Path(__file__).resolve().parent.parent / "local_data" / "calendar"
    candidates.extend([
        root / "trading_calendar.parquet",
        Path(__file__).resolve().parent / "to" / "local_data" / "calendar" / "trading_calendar.parquet",
    ])

    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "date_str" in df.columns:
            dates = df["date_str"].astype(str).str.replace("-", "", regex=False).str[:8]
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        else:
            continue
        dates = sorted(d for d in dates.dropna().unique().tolist() if start_date <= d <= end_date)
        if dates:
            return dates
    return None


def load_trade_dates_from_tushare(pro, limiter: RateLimiter, start_date: str, end_date: str, logger: logging.Logger) -> List[str]:
    logger.info("未找到本地交易日历，使用Tushare trade_cal获取交易日")
    limiter.wait()
    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
    if cal is None or cal.empty:
        raise RuntimeError("Tushare trade_cal 返回空数据")
    return sorted(cal["cal_date"].astype(str).tolist())


def fetch_adj_factor_for_date(pro, limiter: RateLimiter, trade_date: str, retry: int, logger: logging.Logger) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, retry + 1):
        try:
            limiter.wait()
            df = pro.adj_factor(trade_date=trade_date)
            if df is None:
                df = pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
            if not df.empty:
                df = df[["ts_code", "trade_date", "adj_factor"]].copy()
                df["ts_code"] = df["ts_code"].astype(str)
                df["trade_date"] = df["trade_date"].astype(str)
                df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
            return df
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            sleep = min(45.0, 1.5 * attempt + random.random() * 2.0)
            if "频率" in msg or "limit" in msg.lower() or "每分钟" in msg:
                sleep = max(sleep, 20.0 + random.random() * 10.0)
            logger.warning("adj_factor %s 第%s/%s次失败: %s，%.1fs后重试", trade_date, attempt, retry, exc, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"adj_factor {trade_date} 重试失败: {last_exc}") from last_exc


def save_monthly(frames: Iterable[pd.DataFrame], output_dir: Path, overwrite: bool, logger: logging.Logger) -> pd.DataFrame:
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["ts_code", "trade_date", "adj_factor"])
    all_df = all_df.drop_duplicates(["ts_code", "trade_date"], keep="last")
    all_df = all_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    all_df["month"] = all_df["trade_date"].astype(str).str[:6]

    output_dir.mkdir(parents=True, exist_ok=True)
    for month, month_df in all_df.groupby("month", sort=True):
        out_file = output_dir / f"adj_factor_{month}.parquet"
        month_df = month_df.drop(columns=["month"])
        new_dates = set(month_df["trade_date"].astype(str).str[:8].unique().tolist())
        if out_file.exists():
            old_df = pd.read_parquet(out_file)
            old_dates = set(old_df["trade_date"].astype(str).str[:8].unique().tolist())
            combined = pd.concat([old_df, month_df], ignore_index=True)
            combined = combined.drop_duplicates(["ts_code", "trade_date"], keep="last")
            combined = combined.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
            combined_dates = set(combined["trade_date"].astype(str).str[:8].unique().tolist())
            lost_dates = sorted(old_dates - combined_dates)
            if lost_dates:
                raise RuntimeError(
                    f"保存 {month} 会丢失既有交易日: {lost_dates[:10]}，已终止，避免部分月份覆盖完整月份"
                )
            if overwrite:
                logger.warning(
                    "检测到 --overwrite，但按交易日安全合并保存 %s：旧日期=%s, 新日期=%s, 合并日期=%s；"
                    "不会删除本次未下载的同月既有日期",
                    month,
                    len(old_dates),
                    len(new_dates),
                    len(combined_dates),
                )
        else:
            combined = month_df
            combined_dates = set(combined["trade_date"].astype(str).str[:8].unique().tolist())
        daily_counts = combined.groupby(combined["trade_date"].astype(str).str[:8])["ts_code"].nunique()
        if len(daily_counts) > 0:
            median_count = daily_counts.median()
            low_days = daily_counts[daily_counts < median_count * 0.95]
            if len(low_days) > 0:
                raise RuntimeError(
                    f"保存 {month} 前发现单日行数异常偏低: {low_days.astype(int).to_dict()}，"
                    f"月内中位数={int(median_count)}，已终止"
                )
        combined.to_parquet(out_file, index=False, compression="snappy", engine="pyarrow")
        logger.info(
            "保存 %s: %s 行, 交易日=%s, 单日行数=%s~%s -> %s",
            month,
            f"{len(combined):,}",
            len(combined_dates),
            int(daily_counts.min()) if len(daily_counts) else 0,
            int(daily_counts.max()) if len(daily_counts) else 0,
            out_file,
        )
    return all_df.drop(columns=["month"])


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / f"download_adj_factor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    start_date = normalize_date_str(args.start_date)
    end_date = normalize_date_str(args.end_date)
    limiter = RateLimiter(args.api_interval)

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()

    trade_dates = load_local_calendar(args.calendar_file, start_date, end_date)
    if trade_dates is None:
        trade_dates = load_trade_dates_from_tushare(pro, limiter, start_date, end_date, logger)

    logger.info("准备下载adj_factor: %s ~ %s, 交易日=%s, workers=%s", start_date, end_date, len(trade_dates), args.workers)

    frames = []
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {
            executor.submit(fetch_adj_factor_for_date, pro, limiter, d, args.retry, logger): d
            for d in trade_dates
        }
        for idx, future in enumerate(as_completed(future_map), 1):
            date_str = future_map[future]
            try:
                df = future.result()
                frames.append(df)
                if idx % 20 == 0 or idx == len(future_map):
                    logger.info("下载进度 %s/%s, 最近日期=%s, rows=%s", idx, len(future_map), date_str, len(df))
            except Exception as exc:
                logger.exception("下载失败: %s | %s", date_str, exc)
                failed.append(date_str)

    if failed:
        raise RuntimeError(f"存在 {len(failed)} 个交易日adj_factor下载失败: {failed[:20]}")

    all_df = save_monthly(frames, output_dir, overwrite=args.overwrite, logger=logger)
    manifest = {
        "start_date": start_date,
        "end_date": end_date,
        "trade_dates": len(trade_dates),
        "rows": int(len(all_df)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": sorted(p.name for p in output_dir.glob("adj_factor_*.parquet")),
    }
    with open(output_dir / "adj_factor_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("全部完成: rows=%s, output=%s", f"{len(all_df):,}", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
