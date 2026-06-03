#!/usr/bin/env python3
"""
Fast train-set minute label cache generator.

This version does not use BacktestStrategy/build_minute_trade_cache. It is built
for full train-set batch generation:

- scans factors.parquet date/stock_code into a temporary SQLite universe store
  without keeping the full universe in memory
- processes target dates in small chunks
- loads only the needed daily OHLC monthly partitions
- loads minute parquet by month for the chunk holding window
- computes a stable training label rule directly
- marks samples invalid when adj_factor changes inside T..T+holding window

Stable training rule:
- signal date: T
- buy date: T+1
- entry skip: T+1 first 5m open vs T close < -3% or > +5%
- buy price: first 5m weighted price by default
  0.60*open + 0.15*high + 0.15*low + 0.10*close
- risk checks start from T+2:
  stop loss when close return <= -5%
  profit bearish exit when 3 consecutive bearish 5m bars appear on T+2/T+3
- otherwise exit at the last 5m close on holding end date
"""

from __future__ import annotations

import argparse
import gc
import logging
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


logger = logging.getLogger("generate_train_minute_label_cache")

DEFAULT_ADJ_FACTOR_DIR = Path(__file__).resolve().parent.parent / "local_data" / "adj_factor"
RESULT_COLUMNS = [
    "date", "stock_code", "predicted_return", "buy_executable", "skip_reason",
    "buy_price", "exit_price", "exit_date", "exit_reason", "minute_realized_return",
    "minute_min_return", "minute_max_drawdown", "stoploss_hit", "profit_bearish_hit",
    "holding_days_realized", "missing_minute_data", "adj_factor_changed",
    "adj_factor_missing", "adj_factor_window_count", "label_valid", "label_return",
    "buy_price_mode", "rule_signature",
]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="快速生成训练集分钟标签缓存")
    parser.add_argument("--source_parquet", default="factor_lab_data/factors.parquet", help="用于确定训练universe的parquet，只读取date/stock_code")
    parser.add_argument("--start_date", required=True, help="YYYYMMDD")
    parser.add_argument("--end_date", required=True, help="YYYYMMDD")
    parser.add_argument("--daily_data_root", default="../local_data/daily", help="已保留兼容参数；当前快速版优先使用5分钟T日最后close做prev_close")
    parser.add_argument("--minute_data_path", default="../local_data/minute", help="5分钟月度分区目录 YYYYMM/minute_data.parquet")
    parser.add_argument("--adj_factor_dir", default=str(DEFAULT_ADJ_FACTOR_DIR), help="download_adj_factor.py输出目录")
    parser.add_argument("--output_dir", default="results/minute_train_label_cache")
    parser.add_argument("--holding_days", type=int, default=3)
    parser.add_argument("--chunk_days", type=int, default=5, help="每个输出分片包含的目标交易日数，内存高就降到2~3")
    parser.add_argument("--universe_batch_size", type=int, default=500_000, help="扫描factors.parquet时的batch行数")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--buy_price_mode", choices=["open", "first_bar_weighted"], default="first_bar_weighted")
    parser.add_argument("--buy_open_weight", type=float, default=0.60)
    parser.add_argument("--buy_high_weight", type=float, default=0.15)
    parser.add_argument("--buy_low_weight", type=float, default=0.15)
    parser.add_argument("--buy_close_weight", type=float, default=0.10)
    parser.add_argument("--entry_gap_down", type=float, default=-0.03)
    parser.add_argument("--entry_gap_up", type=float, default=0.05)
    parser.add_argument("--stop_loss", type=float, default=-0.05)
    parser.add_argument("--bearish_count", type=int, default=3)
    parser.add_argument("--allow_missing_adj_factor", action="store_true")
    parser.add_argument("--keep_temp_universe", action="store_true")
    return parser.parse_args()


def ymd(value: str) -> pd.Timestamp:
    return pd.to_datetime(str(value), format="%Y%m%d")


def month_keys_between(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[str]:
    return [str(p).replace("-", "") for p in pd.period_range(start_ts.to_period("M"), end_ts.to_period("M"), freq="M")]


def load_trading_calendar() -> List[pd.Timestamp]:
    candidates = [
        Path(__file__).resolve().parent.parent / "local_data" / "calendar" / "trading_calendar.parquet",
        Path(__file__).resolve().parent / "to" / "local_data" / "calendar" / "trading_calendar.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "date_str" in df.columns:
            dates = pd.to_datetime(df["date_str"].astype(str))
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        else:
            continue
        return sorted(pd.Timestamp(d).normalize() for d in dates.dropna().unique())
    raise FileNotFoundError(f"未找到交易日历: {candidates}")


def next_trading_window(signal_day: pd.Timestamp, trading_days: List[pd.Timestamp], holding_days: int) -> Tuple[Optional[pd.Timestamp], List[pd.Timestamp]]:
    idx_map = {d: i for i, d in enumerate(trading_days)}
    idx = idx_map.get(pd.Timestamp(signal_day).normalize())
    if idx is None or idx + 1 >= len(trading_days):
        return None, []
    start = idx + 1
    end = min(len(trading_days), start + int(holding_days))
    return trading_days[start], trading_days[start:end]


def split_dates(dates: List[pd.Timestamp], chunk_days: int) -> List[List[pd.Timestamp]]:
    chunk_days = max(1, int(chunk_days))
    return [dates[i:i + chunk_days] for i in range(0, len(dates), chunk_days)]


def setup_universe_db(source_parquet: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp, batch_size: int, temp_dir: Path) -> sqlite3.Connection:
    db_path = temp_dir / "train_universe.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE universe(date TEXT NOT NULL, stock_code TEXT NOT NULL, PRIMARY KEY(date, stock_code))")

    pf = pq.ParquetFile(source_parquet)
    inserted = 0
    start_date = start_ts.normalize()
    end_date = end_ts.normalize()
    for batch in pf.iter_batches(columns=["date", "stock_code"], batch_size=int(batch_size)):
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            continue
        df = df.dropna(subset=["date", "stock_code"]).drop_duplicates(["date", "stock_code"])
        records = [(d.strftime("%Y-%m-%d"), str(s)) for d, s in zip(df["date"], df["stock_code"])]
        conn.executemany("INSERT OR IGNORE INTO universe(date, stock_code) VALUES (?, ?)", records)
        inserted += len(records)
        if inserted % 500_000 < len(records):
            conn.commit()
            logger.info("[UNIVERSE] 已写入临时universe: %s rows", f"{inserted:,}")
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_universe_date ON universe(date)")
    logger.info("[UNIVERSE] 构建完成: %s rows -> %s", f"{inserted:,}", db_path)
    return conn


def query_universe(conn: sqlite3.Connection, start_day: pd.Timestamp, end_day: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, stock_code FROM universe WHERE date >= ? AND date <= ? ORDER BY date, stock_code",
        conn,
        params=(start_day.strftime("%Y-%m-%d"), end_day.strftime("%Y-%m-%d")),
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_daily_ohlc(root: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for month in month_keys_between(start_ts, end_ts):
        path = root / month / "daily_data.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["date", "stock_code", "open", "high", "low", "close"]))
    if not frames:
        raise FileNotFoundError(f"未找到日线分区: {root}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= start_ts.normalize()) & (df["date"] <= end_ts.normalize())]
    return df.drop_duplicates(["date", "stock_code"], keep="last")


def load_minute_window(root: Path, start_day: pd.Timestamp, end_day: pd.Timestamp, stocks: Iterable[str]) -> pd.DataFrame:
    stock_set = set(str(s) for s in stocks)
    frames = []
    cols = ["stock_code", "datetime", "open", "high", "low", "close", "volume", "amount"]
    for month in month_keys_between(start_day, end_day):
        path = root / month / "minute_data.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=cols)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.normalize()
        df = df[
            (df["date"] >= start_day.normalize()) &
            (df["date"] <= end_day.normalize()) &
            (df["stock_code"].astype(str).isin(stock_set))
        ].copy()
        if not df.empty:
            frames.append(df)
        del df
    if not frames:
        return pd.DataFrame(columns=cols + ["date"])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "stock_code", "datetime"]).reset_index(drop=True)
    return out


def load_adj_factor(adj_dir: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[Tuple[str, pd.Timestamp], float]:
    frames = []
    for month in month_keys_between(start_ts, end_ts):
        path = adj_dir / f"adj_factor_{month}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["ts_code", "trade_date", "adj_factor"]))
    if not frames:
        raise FileNotFoundError(f"未找到adj_factor文件: {adj_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dt.normalize()
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    df = df.dropna(subset=["ts_code", "trade_date", "adj_factor"])
    df = df.drop_duplicates(["ts_code", "trade_date"], keep="last")
    lookup = {
        (str(r.ts_code), pd.Timestamp(r.trade_date)): float(r.adj_factor)
        for r in df.itertuples(index=False)
    }
    logger.info("[ADJ] loaded rows=%s", f"{len(lookup):,}")
    return lookup


def adj_window_flags(stock_code: str, signal_day: pd.Timestamp, trading_days: List[pd.Timestamp], holding_days: int, lookup: Dict[Tuple[str, pd.Timestamp], float], allow_missing: bool) -> Tuple[bool, bool, int]:
    # Include signal day because the T close -> T+1 open executable gap check
    # crosses an ex-right boundary when T+1 is an ex-dividend/ex-right date.
    _, win_days = next_trading_window(signal_day, trading_days, holding_days)
    check_days = [signal_day.normalize()] + [day.normalize() for day in win_days]
    vals = []
    missing = False
    for day in check_days:
        v = lookup.get((stock_code, day))
        if v is None or not np.isfinite(v):
            missing = True
        else:
            vals.append(float(v))
    changed = len(set(round(v, 8) for v in vals)) > 1
    if allow_missing:
        missing = False
    return changed, missing, len(vals)


def calc_buy_price(first_bar: pd.Series, args) -> float:
    open_price = float(first_bar["open"])
    if args.buy_price_mode == "open":
        return open_price
    weights = {
        "open": args.buy_open_weight,
        "high": args.buy_high_weight,
        "low": args.buy_low_weight,
        "close": args.buy_close_weight,
    }
    total = 0.0
    wsum = 0.0
    for col, w in weights.items():
        val = first_bar.get(col, np.nan)
        if pd.notna(val) and float(val) > 0:
            total += float(val) * float(w)
            wsum += float(w)
    return total / wsum if wsum > 0 else open_price


def compute_path_stats(path_df: pd.DataFrame, buy_price: float, exit_dt: pd.Timestamp) -> Tuple[float, float]:
    path = path_df[path_df["datetime"] <= exit_dt]
    if path.empty or buy_price <= 0:
        return 0.0, 0.0
    min_ret = float(((path["low"].astype(float) - buy_price) / buy_price).min())
    peak = buy_price
    worst_dd = 0.0
    for row in path.itertuples(index=False):
        high = float(row.high) if pd.notna(row.high) else np.nan
        low = float(row.low) if pd.notna(row.low) else np.nan
        if np.isfinite(high):
            peak = max(peak, high)
        if np.isfinite(low) and peak > 0:
            worst_dd = min(worst_dd, (low - peak) / peak)
    return min_ret, float(worst_dd)


def simulate_one(row, stock_minute: pd.DataFrame, trade_day: pd.Timestamp, holding_days_list: List[pd.Timestamp], args) -> Dict:
    signal_day = pd.Timestamp(row.date).normalize()
    stock_code = str(row.stock_code)
    result = {
        "date": signal_day, "stock_code": stock_code, "predicted_return": 0.0,
        "buy_executable": False, "skip_reason": "", "buy_price": np.nan,
        "exit_price": np.nan, "exit_date": pd.NaT, "exit_reason": "",
        "minute_realized_return": 0.0, "minute_min_return": 0.0, "minute_max_drawdown": 0.0,
        "stoploss_hit": False, "profit_bearish_hit": False, "holding_days_realized": 0,
        "missing_minute_data": False,
    }
    if trade_day is None or not holding_days_list:
        result.update(skip_reason="无T+1交易日", missing_minute_data=True)
        return result

    trade_minutes = stock_minute[stock_minute["date"] == trade_day]
    if trade_minutes.empty:
        result.update(skip_reason="T+1无5分钟数据", missing_minute_data=True)
        return result

    first_bar = trade_minutes.iloc[0]
    buy_open = float(first_bar["open"])
    prev_close = float(row.prev_close) if pd.notna(row.prev_close) else np.nan
    if not np.isfinite(prev_close) or prev_close <= 0:
        result.update(skip_reason="缺少T日5分钟收盘价", missing_minute_data=True)
        return result

    gap = (buy_open - prev_close) / prev_close
    if gap < args.entry_gap_down:
        result.update(skip_reason=f"开盘跳空{-gap:.2%}")
        return result
    if gap > args.entry_gap_up:
        result.update(skip_reason=f"开盘跳空{gap:.2%}")
        return result

    buy_price = calc_buy_price(first_bar, args)
    result["buy_price"] = buy_price
    result["buy_executable"] = True

    hold_set = set(holding_days_list)
    path = stock_minute[stock_minute["date"].isin(hold_set)].copy()
    if path.empty:
        result.update(skip_reason="持仓窗口无5分钟数据", missing_minute_data=True, buy_executable=False)
        return result

    exit_price = np.nan
    exit_dt = pd.NaT
    exit_reason = "到期"
    stoploss_hit = False
    bearish_hit = False
    bearish_run = 0
    risk_days = set(holding_days_list[1:])

    for bar in path.itertuples(index=False):
        current_day = pd.Timestamp(bar.date).normalize()
        close = float(bar.close)
        if current_day in risk_days:
            ret = (close - buy_price) / buy_price if buy_price > 0 else 0.0
            if ret <= args.stop_loss:
                exit_price = close
                exit_dt = pd.Timestamp(bar.datetime)
                exit_reason = f"风控: 累计收益低于{args.stop_loss:.2%}"
                stoploss_hit = True
                break
            is_bearish = close < float(bar.open)
            bearish_run = bearish_run + 1 if is_bearish else 0
            if bearish_run >= int(args.bearish_count):
                exit_price = close
                exit_dt = pd.Timestamp(bar.datetime)
                exit_reason = f"风控: 连续{args.bearish_count}根5分钟阴线"
                bearish_hit = True
                break

    if pd.isna(exit_dt):
        last_day = holding_days_list[-1]
        last_minutes = path[path["date"] == last_day]
        if last_minutes.empty:
            last_minutes = path
        last_bar = last_minutes.iloc[-1]
        exit_price = float(last_bar["close"])
        exit_dt = pd.Timestamp(last_bar["datetime"])

    min_ret, max_dd = compute_path_stats(path, buy_price, exit_dt)
    realized = (exit_price - buy_price) / buy_price if buy_price > 0 else 0.0
    result.update({
        "exit_price": float(exit_price),
        "exit_date": pd.Timestamp(exit_dt).normalize(),
        "exit_reason": exit_reason,
        "minute_realized_return": float(realized),
        "minute_min_return": float(min_ret),
        "minute_max_drawdown": float(max_dd),
        "stoploss_hit": bool(stoploss_hit),
        "profit_bearish_hit": bool(bearish_hit),
        "holding_days_realized": int(len([d for d in holding_days_list if d <= pd.Timestamp(exit_dt).normalize()])),
    })
    return result


def process_chunk(chunk_dates: List[pd.Timestamp], universe_conn: sqlite3.Connection, trading_days: List[pd.Timestamp], adj_lookup: Dict, args, rule_sig: str) -> pd.DataFrame:
    chunk_start = min(chunk_dates)
    chunk_end = max(chunk_dates)
    target_universe = query_universe(universe_conn, chunk_start, chunk_end)
    if target_universe.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    all_window_days = []
    trade_map = {}
    for d in chunk_dates:
        trade_day, win_days = next_trading_window(d, trading_days, args.holding_days)
        trade_map[d.normalize()] = (trade_day, win_days)
        all_window_days.extend(win_days)
    if not all_window_days:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    # 加载T日分钟数据用于同口径prev_close，避免日线复权价和分钟不复权价混用造成虚假跳空。
    minute_start = min(list(chunk_dates) + all_window_days)
    minute_end = max(all_window_days)

    stocks = target_universe["stock_code"].dropna().astype(str).unique().tolist()
    minute = load_minute_window(Path(args.minute_data_path), minute_start, minute_end, stocks)
    if minute.empty:
        target_rows = target_universe.copy()
        target_rows["prev_close"] = np.nan
    else:
        prev_close_df = (
            minute[minute["date"].isin(set(pd.Timestamp(d).normalize() for d in chunk_dates))]
            .sort_values(["date", "stock_code", "datetime"])
            .groupby(["date", "stock_code"], as_index=False)
            .tail(1)[["date", "stock_code", "close"]]
            .rename(columns={"close": "prev_close"})
        )
        target_rows = target_universe.merge(prev_close_df, on=["date", "stock_code"], how="left")
    logger.info(
        "[FAST-CACHE] chunk %s~%s universe=%s minute_rows=%s minute_window=%s~%s",
        chunk_start.date(), chunk_end.date(), f"{len(target_rows):,}", f"{len(minute):,}", minute_start.date(), minute_end.date()
    )

    minute_groups = {stock: df for stock, df in minute.groupby("stock_code", sort=False)}
    results = []
    for row in target_rows.itertuples(index=False):
        signal_day = pd.Timestamp(row.date).normalize()
        trade_day, win_days = trade_map.get(signal_day, (None, []))
        stock_df = minute_groups.get(str(row.stock_code), pd.DataFrame())
        rec = simulate_one(row, stock_df, trade_day, win_days, args)
        changed, missing_adj, adj_count = adj_window_flags(str(row.stock_code), signal_day, trading_days, args.holding_days, adj_lookup, args.allow_missing_adj_factor)
        label_valid = (
            bool(rec["buy_executable"])
            and not bool(rec["missing_minute_data"])
            and not bool(changed)
            and not bool(missing_adj)
        )
        rec.update({
            "adj_factor_changed": bool(changed),
            "adj_factor_missing": bool(missing_adj),
            "adj_factor_window_count": int(adj_count),
            "label_valid": bool(label_valid),
            "label_return": float(rec["minute_realized_return"]) if label_valid else np.nan,
            "buy_price_mode": args.buy_price_mode,
            "rule_signature": rule_sig,
        })
        results.append(rec)

    out = pd.DataFrame(results)
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[RESULT_COLUMNS]
    return out


def main() -> int:
    setup_logging()
    args = parse_args()
    source_path = Path(args.source_parquet)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = ymd(args.start_date)
    end_ts = ymd(args.end_date)
    trading_days = [d for d in load_trading_calendar() if start_ts <= d <= end_ts + pd.Timedelta(days=30)]
    target_days = [d for d in trading_days if start_ts <= d <= end_ts]
    chunks = split_dates(target_days, args.chunk_days)
    if not chunks:
        raise ValueError("目标区间没有交易日")

    # Load adj_factor slightly beyond end date for T+holding windows.
    adj_lookup = load_adj_factor(Path(args.adj_factor_dir), start_ts, end_ts + pd.Timedelta(days=30))
    buy_sig = "buyopen" if args.buy_price_mode == "open" else (
        f"buyw_o{args.buy_open_weight:.2f}_h{args.buy_high_weight:.2f}_"
        f"l{args.buy_low_weight:.2f}_c{args.buy_close_weight:.2f}"
    )
    rule_sig = (
        f"fast_hd{int(args.holding_days)}_gap{args.entry_gap_down:.3f}_{args.entry_gap_up:.3f}"
        f"_sl{args.stop_loss:.3f}_bear{int(args.bearish_count)}_{buy_sig}_adjfilter"
    )

    temp_obj = tempfile.TemporaryDirectory(prefix="train_universe_")
    temp_dir = Path(temp_obj.name)
    universe_conn = setup_universe_db(source_path, start_ts, end_ts, args.universe_batch_size, temp_dir)

    built = 0
    skipped = 0
    total_rows = 0
    total_chunks = len(chunks)
    logger.info(
        "[FAST-CACHE] start chunks=%s chunk_days=%s target=%s~%s output=%s",
        total_chunks, args.chunk_days, start_ts.date(), end_ts.date(), output_dir
    )
    try:
        for idx, chunk in enumerate(chunks, 1):
            chunk_start = min(chunk)
            chunk_end = max(chunk)
            cache_path = output_dir / (
                f"train_minute_label_cache_{chunk_start.strftime('%Y%m%d')}_"
                f"{chunk_end.strftime('%Y%m%d')}_{rule_sig}.parquet"
            )
            if cache_path.exists() and not args.overwrite:
                logger.info("[FAST-CACHE] exists, skip: %s", cache_path.name)
                skipped += 1
                continue
            logger.info(
                "[FAST-CACHE] processing chunk %s/%s: %s~%s",
                idx, total_chunks, chunk_start.date(), chunk_end.date()
            )
            cache_df = process_chunk(chunk, universe_conn, trading_days, adj_lookup, args, rule_sig)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_parquet(cache_path, index=False, compression="snappy", engine="pyarrow")
            built += 1
            total_rows += len(cache_df)
            gap_skip_count = int(cache_df["skip_reason"].astype(str).str.contains("开盘跳空", na=False).sum()) if len(cache_df) else 0
            missing_prev_close_count = int((cache_df["skip_reason"] == "缺少T日5分钟收盘价").sum()) if len(cache_df) else 0
            logger.info(
                "[FAST-CACHE] done %s/%s rows=%s valid=%s buy=%s gap_skip=%s missing_t_close=%s adj_changed=%s -> %s",
                idx,
                len(chunks),
                f"{len(cache_df):,}",
                f"{int(cache_df['label_valid'].sum()) if len(cache_df) else 0:,}",
                f"{int(cache_df['buy_executable'].sum()) if len(cache_df) else 0:,}",
                f"{gap_skip_count:,}",
                f"{missing_prev_close_count:,}",
                f"{int(cache_df['adj_factor_changed'].sum()) if len(cache_df) else 0:,}",
                cache_path.name
            )
            del cache_df
            gc.collect()
    finally:
        universe_conn.close()
        if args.keep_temp_universe:
            logger.info("[UNIVERSE] kept temp dir: %s", temp_dir)
        else:
            temp_obj.cleanup()

    logger.info("[FAST-CACHE] all done chunks=%s built=%s skipped=%s rows=%s output=%s", len(chunks), built, skipped, f"{total_rows:,}", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
