from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_CACHE_DIR = Path("results/minute_train_label_cache")
DEFAULT_PATTERN = "train_minute_label_cache_*.parquet"
DEFAULT_POS_TOP_PCT = 0.005
DEFAULT_NEG_BOTTOM_PCT = 0.98
MIN_DAILY_RANK_SIZE = 50

_CACHE_FILE_RE = re.compile(r"train_minute_label_cache_(\d{8})_(\d{8})_.*\.parquet$")

DEFAULT_CACHE_COLUMNS = [
    "date",
    "stock_code",
    "label_return",
    "label_valid",
    "minute_realized_return",
    "minute_min_return",
    "minute_max_drawdown",
    "buy_executable",
    "skip_reason",
    "buy_price",
    "exit_price",
    "exit_date",
    "exit_reason",
    "missing_minute_data",
    "adj_factor_changed",
    "adj_factor_missing",
    "stoploss_hit",
    "profit_bearish_hit",
    "holding_days_realized",
]


def _to_ts(value) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        raise ValueError(f"invalid date value: {value!r}")
    return pd.Timestamp(ts).normalize()


def _cache_file_range(path: Path) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    match = _CACHE_FILE_RE.match(path.name)
    if not match:
        return None
    start = pd.to_datetime(match.group(1), format="%Y%m%d")
    end = pd.to_datetime(match.group(2), format="%Y%m%d")
    return pd.Timestamp(start), pd.Timestamp(end)


def find_minute_label_cache_files(
    start_date,
    end_date,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    pattern: str = DEFAULT_PATTERN,
) -> list[Path]:
    """Return cache parquet files whose filename date range overlaps the request."""
    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    matched: list[tuple[pd.Timestamp, Path]] = []
    for path in cache_path.glob(pattern):
        file_range = _cache_file_range(path)
        if file_range is None:
            continue
        file_start, file_end = file_range
        if file_end < start_ts or file_start > end_ts:
            continue
        matched.append((file_start, path))
    matched.sort(key=lambda item: (item[0], item[1].name))
    return [path for _, path in matched]


def _existing_columns(path: Path) -> Optional[set[str]]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return None


def load_minute_label_cache(
    start_date,
    end_date,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    pattern: str = DEFAULT_PATTERN,
    columns: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load minute label cache rows for the requested date range.

    The loader reads only cache files whose encoded filename range overlaps
    [start_date, end_date], then filters rows by date and de-duplicates
    (date, stock_code).
    """
    log = logger or logging.getLogger(__name__)
    start_ts = _to_ts(start_date)
    end_ts = _to_ts(end_date)
    requested_columns = list(columns) if columns is not None else list(DEFAULT_CACHE_COLUMNS)
    if "date" not in requested_columns:
        requested_columns.insert(0, "date")
    if "stock_code" not in requested_columns:
        requested_columns.insert(1, "stock_code")

    files = find_minute_label_cache_files(start_ts, end_ts, cache_dir=cache_dir, pattern=pattern)
    if not files:
        raise FileNotFoundError(
            f"no minute label cache files overlap {start_ts:%Y%m%d}~{end_ts:%Y%m%d} in {Path(cache_dir)}"
        )

    frames: list[pd.DataFrame] = []
    for source_order, path in enumerate(files):
        available = _existing_columns(path)
        use_columns = requested_columns
        if available is not None:
            use_columns = [col for col in requested_columns if col in available]
            missing_required = {"date", "stock_code"} - set(use_columns)
            if missing_required:
                raise ValueError(f"{path} missing required columns: {sorted(missing_required)}")

        frame = pd.read_parquet(path, columns=use_columns)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        mask = (frame["date"] >= start_ts) & (frame["date"] <= end_ts)
        frame = frame.loc[mask]
        if not frame.empty:
            frame["_cache_source_mtime"] = path.stat().st_mtime
            frame["_cache_source_order"] = source_order
            frames.append(frame)

    if not frames:
        raise ValueError(f"minute label cache files found, but no rows in {start_ts:%Y%m%d}~{end_ts:%Y%m%d}")

    cache_df = pd.concat(frames, ignore_index=True, copy=False)
    cache_df["stock_code"] = cache_df["stock_code"].astype(str)
    before = len(cache_df)
    cache_df = cache_df.sort_values(
        ["date", "stock_code", "_cache_source_mtime", "_cache_source_order"],
        kind="mergesort",
    ).drop_duplicates(
        ["date", "stock_code"], keep="last"
    )
    if before != len(cache_df):
        log.warning(
            "[MINUTE-LABEL] dropped duplicate cache rows: %s; kept newer cache file by mtime",
            before - len(cache_df),
        )
    cache_df = cache_df.drop(columns=["_cache_source_mtime", "_cache_source_order"], errors="ignore")

    log.info(
        "[MINUTE-LABEL] loaded cache: files=%s, rows=%s, dates=%s~%s",
        len(files),
        f"{len(cache_df):,}",
        cache_df["date"].min().strftime("%Y%m%d"),
        cache_df["date"].max().strftime("%Y%m%d"),
    )
    return cache_df.reset_index(drop=True)


def _build_daily_binary_labels(
    returns: pd.Series,
    dates: pd.Series,
    valid_mask: pd.Series,
    pos_top_pct: float,
    neg_bottom_pct: float,
) -> tuple[pd.Series, pd.Series]:
    rank_pct_desc = pd.Series(np.nan, index=returns.index, dtype="float64")
    label = pd.Series(np.nan, index=returns.index, dtype="float64")

    rank_source = pd.DataFrame(
        {
            "date": dates,
            "ret": returns,
            "valid": valid_mask.astype(bool),
        },
        index=returns.index,
    )
    for _, day_df in rank_source.loc[rank_source["valid"]].groupby("date", sort=False):
        day_ret = day_df["ret"].dropna()
        if len(day_ret) < MIN_DAILY_RANK_SIZE:
            continue
        rank_pct = day_ret.rank(method="first", ascending=False) / float(len(day_ret))
        rank_pct_desc.loc[rank_pct.index] = rank_pct

    label.loc[rank_pct_desc <= pos_top_pct] = 1.0
    label.loc[rank_pct_desc > (1.0 - neg_bottom_pct)] = 0.0
    return label, rank_pct_desc


def apply_minute_cache_labels(
    stock_data: pd.DataFrame,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    pos_top_pct: float = DEFAULT_POS_TOP_PCT,
    neg_bottom_pct: float = DEFAULT_NEG_BOTTOM_PCT,
    logger: Optional[logging.Logger] = None,
    require_label_valid: bool = True,
) -> pd.DataFrame:
    """
    Attach training labels from the generated minute cache.

    Output compatibility columns:
      atr_return_label, future_ret, relevance, label_rank_pct_desc,
      max_high_return, min_low_return.
    """
    log = logger or logging.getLogger(__name__)
    if stock_data.empty:
        return stock_data.copy()
    if "date" not in stock_data.columns or "stock_code" not in stock_data.columns:
        raise ValueError("stock_data must contain date and stock_code columns")

    data = stock_data.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["stock_code"] = data["stock_code"].astype(str)
    start_ts = data["date"].min()
    end_ts = data["date"].max()

    cache_df = load_minute_label_cache(start_ts, end_ts, cache_dir=cache_dir, logger=log)
    keep_cols = [col for col in DEFAULT_CACHE_COLUMNS if col in cache_df.columns]
    cache_df = cache_df[keep_cols]

    overlap_cols = [col for col in cache_df.columns if col not in {"date", "stock_code"} and col in data.columns]
    if overlap_cols:
        data = data.drop(columns=overlap_cols)

    data = data.merge(cache_df, on=["date", "stock_code"], how="left", sort=False, copy=False)
    return_col = "label_return" if "label_return" in data.columns else "minute_realized_return"
    if return_col not in data.columns:
        raise ValueError("minute cache must contain label_return or minute_realized_return")

    label_return = pd.to_numeric(data[return_col], errors="coerce")
    cache_valid = data[return_col].notna()
    if require_label_valid and "label_valid" in data.columns:
        cache_valid &= data["label_valid"].fillna(False).astype(bool)

    if "valid_cs_mask" in data.columns:
        cache_valid &= data["valid_cs_mask"].fillna(False).astype(bool)
    for flag_col, bad_value in [
        ("is_limit_up", 1),
        ("is_limit_down", 1),
        ("is_zhaban", 1),
        ("is_can_buy", 0),
    ]:
        if flag_col in data.columns:
            logger.info(f"is_limit_up 在数据中")
            cache_valid &= data[flag_col].fillna(0).ne(bad_value)
    if "amount" in data.columns:
        cache_valid &= pd.to_numeric(data["amount"], errors="coerce").fillna(0) >= 10_000_000

    label_return = label_return.where(cache_valid)

    data["atr_return_label"] = label_return
    data["future_ret"] = label_return
    data["max_high_return"] = pd.to_numeric(data.get("minute_realized_return", label_return), errors="coerce").where(cache_valid)
    data["min_low_return"] = pd.to_numeric(data.get("minute_min_return", np.nan), errors="coerce").where(cache_valid)
    if "minute_max_drawdown" in data.columns:
        data["minute_max_drawdown_label"] = pd.to_numeric(data["minute_max_drawdown"], errors="coerce").where(cache_valid)

    relevance, rank_pct_desc = _build_daily_binary_labels(
        returns=label_return,
        dates=data["date"],
        valid_mask=cache_valid,
        pos_top_pct=pos_top_pct,
        neg_bottom_pct=neg_bottom_pct,
    )
    data["relevance"] = relevance
    data["label_rank_pct_desc"] = rank_pct_desc

    pos_count = int((data["relevance"] == 1).sum())
    neg_count = int((data["relevance"] == 0).sum())
    valid_count = int(data["relevance"].notna().sum())
    cache_miss = int(data[return_col].isna().sum())
    log.info(
        "[MINUTE-LABEL] labels ready: rows=%s, cache_missing=%s, valid_train=%s, pos=%s, neg=%s, pos_ratio=%.2f%%",
        f"{len(data):,}",
        f"{cache_miss:,}",
        f"{valid_count:,}",
        f"{pos_count:,}",
        f"{neg_count:,}",
        (pos_count / valid_count * 100.0) if valid_count else 0.0,
    )
    return data
