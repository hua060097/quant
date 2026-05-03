#!/usr/bin/env python3
"""
离线生成验证集分钟交易结果缓存。

设计目标：
1. 一次性为指定日期范围内的全量股票生成分钟交易结果缓存
2. 训练脚本只读取缓存，不在参数搜索阶段重跑分钟回测
3. 自动包含前后缓冲交易日，满足 MA5/T+2/T+3 风控所需上下文
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from rqalpha_minute_datasource.strategy_backtest_lead_fusion_rank_binary_v2 import (
    StrategyConfig,
    StrategyType,
    build_minute_trade_cache,
    get_minute_trade_cache_rule_signature,
)


logger = logging.getLogger(__name__)
FIXED_CHUNK_TRADING_DAYS = 10


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="生成验证集分钟交易缓存")
    parser.add_argument("--source_parquet", default="factor_lab_data/factors.parquet", help="用于确定目标股票 universe 和交易日的源 parquet")
    parser.add_argument("--daily_source_parquet", default=None, help="提供 open/high/low/close 的日频源 parquet；不传则自动在 cache/stock_features_*.parquet 中寻找")
    parser.add_argument("--start_date", required=True, help="目标缓存开始日期 YYYYMMDD")
    parser.add_argument("--end_date", required=True, help="目标缓存结束日期 YYYYMMDD")
    parser.add_argument("--context_before_days", type=int, default=8, help="向前额外保留的交易日数量")
    parser.add_argument("--context_after_days", type=int, default=5, help="向后额外保留的交易日数量")
    parser.add_argument("--holding_days", type=int, default=3, help="分钟回测持有天数")
    parser.add_argument("--minute_data_path", default="../local_data/minute/", help="分钟数据目录")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已有缓存")
    return parser.parse_args()


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def _load_trading_calendar() -> list[pd.Timestamp]:
    calendar_path = Path(__file__).parent / "to" / "local_data" / "calendar" / "trading_calendar.parquet"
    if not calendar_path.exists():
        raise FileNotFoundError(f"交易日历文件不存在: {calendar_path}")
    calendar_df = pd.read_parquet(calendar_path)
    if "date_str" in calendar_df.columns:
        return [pd.to_datetime(x) for x in calendar_df["date_str"].tolist()]
    if "date" in calendar_df.columns:
        return pd.to_datetime(calendar_df["date"]).tolist()
    raise ValueError(f"交易日历文件缺少 'date_str' 或 'date' 列，现有列: {list(calendar_df.columns)}")


def _find_stock_features_source(context_start: pd.Timestamp, context_end: pd.Timestamp) -> Path:
    cache_dir = Path(__file__).resolve().parent / "cache"
    candidates = sorted(cache_dir.glob("stock_features_*.parquet"))
    if not candidates:
        raise FileNotFoundError("未找到 cache/stock_features_*.parquet")

    valid_candidates = []
    for path in candidates:
        try:
            name = path.stem  # stock_features_YYYYMMDD_YYYYMMDD
            parts = name.split("_")
            if len(parts) < 4:
                continue
            start_str, end_str = parts[-2], parts[-1]
            file_start = pd.to_datetime(start_str)
            file_end = pd.to_datetime(end_str)
            if file_start <= context_start and file_end >= context_end:
                valid_candidates.append((file_start, file_end, path))
        except Exception:
            continue

    if not valid_candidates:
        raise FileNotFoundError(
            f"未找到覆盖 {context_start.date()} ~ {context_end.date()} 的 stock_features parquet"
        )

    # 选覆盖区间最小、最贴近的一份，降低读取体积
    valid_candidates.sort(key=lambda x: ((x[1] - x[0]).days, str(x[2])))
    return valid_candidates[0][2]


def _resolve_context_window(all_dates: list[pd.Timestamp], start_ts: pd.Timestamp, end_ts: pd.Timestamp, before_days: int, after_days: int):
    all_dates_sorted = sorted(pd.to_datetime(pd.Series(all_dates)).dropna().unique().tolist())
    if not all_dates_sorted:
        raise ValueError("源数据中没有可用交易日")

    target_dates = [d for d in all_dates_sorted if start_ts <= pd.Timestamp(d) <= end_ts]
    if not target_dates:
        raise ValueError(f"源数据中未找到目标区间 {start_ts.date()} ~ {end_ts.date()} 的交易日")

    first_idx = all_dates_sorted.index(target_dates[0])
    last_idx = all_dates_sorted.index(target_dates[-1])
    ctx_start_idx = max(0, first_idx - int(before_days))
    ctx_end_idx = min(len(all_dates_sorted) - 1, last_idx + int(after_days))
    context_dates = all_dates_sorted[ctx_start_idx:ctx_end_idx + 1]
    return [pd.Timestamp(d) for d in target_dates], [pd.Timestamp(d) for d in context_dates]


def _split_dates_into_chunks(target_dates: list[pd.Timestamp], chunk_size: int = FIXED_CHUNK_TRADING_DAYS) -> list[list[pd.Timestamp]]:
    return [target_dates[i:i + chunk_size] for i in range(0, len(target_dates), chunk_size)]


def main():
    _setup_logging()
    args = _parse_args()

    source_path = Path(args.source_parquet)
    if not source_path.exists():
        raise FileNotFoundError(f"未找到源 parquet: {source_path}")

    start_ts = pd.to_datetime(str(args.start_date))
    end_ts = pd.to_datetime(str(args.end_date))

    logger.info("[CACHE-BUILD] 读取交易日历文件...")
    calendar_dates = _load_trading_calendar()
    target_dates, context_dates = _resolve_context_window(
        all_dates=calendar_dates,
        start_ts=start_ts,
        end_ts=end_ts,
        before_days=args.context_before_days,
        after_days=args.context_after_days,
    )
    context_start = min(context_dates)
    context_end = max(context_dates)

    logger.info(
        f"[CACHE-BUILD] 目标区间: {target_dates[0].date()} ~ {target_dates[-1].date()}, "
        f"上下文区间: {context_start.date()} ~ {context_end.date()}"
    )

    logger.info(f"[CACHE-BUILD] 读取股票universe源: {source_path}")
    universe_df = pd.read_parquet(source_path, columns=["date", "stock_code"])
    universe_df = _normalize_dates(universe_df)

    cfg = StrategyConfig(
        strategy_type=StrategyType.TOP_1,
        stock_count=1,
        enable_dynamic_rank_switch=False,
        holding_days=int(args.holding_days),
        max_concurrent_positions=1,
        fixed_position_pct=1.0,
        position_schedule=[1.0],
    )
    rule_sig = get_minute_trade_cache_rule_signature(cfg)
    cache_dir = Path(__file__).resolve().parent / "results" / "minute_eval_cache"
    target_chunks = _split_dates_into_chunks(target_dates, FIXED_CHUNK_TRADING_DAYS)
    logger.info(f"[CACHE-BUILD] 将按固定 {FIXED_CHUNK_TRADING_DAYS} 个交易日分片，共 {len(target_chunks)} 个分片")

    daily_source_cache: dict[Path, pd.DataFrame] = {}
    total_rows = 0
    built_chunks = 0
    skipped_chunks = 0

    for chunk_idx, chunk_dates in enumerate(target_chunks, 1):
        chunk_start = min(chunk_dates)
        chunk_end = max(chunk_dates)
        _, chunk_context_dates = _resolve_context_window(
            all_dates=calendar_dates,
            start_ts=chunk_start,
            end_ts=chunk_end,
            before_days=args.context_before_days,
            after_days=args.context_after_days,
        )
        chunk_context_start = min(chunk_context_dates)
        chunk_context_end = max(chunk_context_dates)

        daily_source_path = Path(args.daily_source_parquet) if args.daily_source_parquet else _find_stock_features_source(
            chunk_context_start, chunk_context_end
        )
        logger.info(
            f"[CACHE-BUILD] 分片 {chunk_idx}/{len(target_chunks)}: "
            f"{chunk_start.date()} ~ {chunk_end.date()}, "
            f"context={chunk_context_start.date()} ~ {chunk_context_end.date()}, "
            f"日频源={daily_source_path.name}"
        )

        if daily_source_path not in daily_source_cache:
            logger.info(f"[CACHE-BUILD] 读取日频OHLC源必要列: {daily_source_path}")
            daily_source_df = pd.read_parquet(
                daily_source_path,
                columns=["date", "stock_code", "open", "high", "low", "close"]
            )
            daily_source_cache[daily_source_path] = _normalize_dates(daily_source_df)
        else:
            daily_source_df = daily_source_cache[daily_source_path]

        target_universe = universe_df[
            (universe_df["date"] >= chunk_start) & (universe_df["date"] <= chunk_end)
        ][["date", "stock_code"]].drop_duplicates().copy()

        context_rows = daily_source_df[
            (daily_source_df["date"] >= chunk_context_start) &
            (daily_source_df["date"] <= chunk_context_end)
        ].copy()
        target_rows = target_universe.merge(
            daily_source_df[["date", "stock_code", "open", "high", "low", "close"]],
            on=["date", "stock_code"],
            how="left"
        )

        cache_path = cache_dir / (
            f"val_minute_cache_{chunk_start.strftime('%Y%m%d')}_"
            f"{chunk_end.strftime('%Y%m%d')}_{rule_sig}.parquet"
        )

        if cache_path.exists() and not args.overwrite:
            logger.info(f"[CACHE-BUILD] 分片缓存已存在，跳过: {cache_path.name}")
            skipped_chunks += 1
            continue

        logger.info(
            f"[CACHE-BUILD] 分片数据量: context_rows={len(context_rows):,}, "
            f"target_rows={len(target_rows):,}, target_dates={target_rows['date'].nunique()}"
        )
        cache_df = build_minute_trade_cache(
            target_prediction_rows=target_rows,
            context_prediction_data=context_rows,
            minute_data_path=args.minute_data_path,
            trading_dates=[d.date() for d in chunk_context_dates],
            cache_path=cache_path,
            holding_days=int(args.holding_days),
            overwrite=bool(args.overwrite),
            log_prefix=f"[CACHE-BUILD-{chunk_idx}]",
        )
        built_chunks += 1
        total_rows += len(cache_df)
        logger.info(
            f"[CACHE-BUILD] 分片完成: rows={len(cache_df):,}, "
            f"buy_executable={cache_df['buy_executable'].mean():.1%}" if len(cache_df) > 0 and 'buy_executable' in cache_df.columns else
            f"[CACHE-BUILD] 分片完成: rows={len(cache_df):,}"
        )

    logger.info(
        f"[CACHE-BUILD] 全部完成: 分片总数={len(target_chunks)}, 新生成={built_chunks}, "
        f"已跳过={skipped_chunks}, 新增缓存行数={total_rows:,}"
    )


if __name__ == "__main__":
    main()
