"""
Import per-stock 5-minute CSV files into the local monthly minute parquet layout.

Source example:
    D:\\BaiduNetdiskDownload\\5min\\5min\\sh600000.csv

Target layout, matching advanced_quant_data_fixed.py:
    C:\\Users\\86151\\.trae-cn\\pythons\\path\\to\\local_data\\minute\\YYYYMM\\minute_data.parquet

Target columns:
    stock_code, datetime, open, high, low, close, volume, amount

The script writes monthly temporary SQLite stores first, then exports each month
to parquet. This keeps memory bounded by the CSV chunk size and avoids creating
millions of tiny parquet part files.
It is designed to be run manually by the user.
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_SOURCE_DIR = Path(r"D:\BaiduNetdiskDownload\5min\5min")
DEFAULT_TARGET_ROOT = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\minute")
DEFAULT_START_DATE = "20230601"
DEFAULT_END_DATE = "20241031"

CSV_COLUMNS = ["日期", "时间", "开盘", "最高", "最低", "收盘", "成交量", "成交额"]
TARGET_COLUMNS = ["stock_code", "datetime", "open", "high", "low", "close", "volume", "amount"]
SQL_COLUMNS = ["stock_code", "datetime", "open", "high", "low", "close", "volume", "amount"]


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("import_baidu_5min")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def stock_code_from_file(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    if len(stem) < 8:
        return None
    prefix = stem[:2]
    code = stem[2:]
    if not code.isdigit():
        return None
    if prefix == "sh":
        return f"{code}.SH"
    if prefix == "sz":
        return f"{code}.SZ"
    if prefix == "bj":
        return f"{code}.BJ"
    return None


def iter_source_files(source_dir: Path, include_bj: bool) -> Iterable[Path]:
    patterns = ["sh*.csv", "sz*.csv"]
    if include_bj:
        patterns.append("bj*.csv")
    for pattern in patterns:
        yield from sorted(source_dir.glob(pattern))


def parse_yyyymmdd(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%Y%m%d")


def transform_chunk(
    chunk: pd.DataFrame,
    stock_code: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    missing = [col for col in CSV_COLUMNS if col not in chunk.columns]
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")

    date_str = chunk["日期"].astype(str)
    date_key = date_str.str.replace("-", "", regex=False).str[:8]
    date_num = pd.to_numeric(date_key, errors="coerce")
    start_num = int(start_ts.strftime("%Y%m%d"))
    end_num = int(end_ts.strftime("%Y%m%d"))
    mask = date_num.between(start_num, end_num, inclusive="both")
    if not mask.any():
        return pd.DataFrame(columns=TARGET_COLUMNS)

    part = chunk.loc[mask, CSV_COLUMNS].copy()
    dt_text = part["日期"].astype(str) + " " + part["时间"].astype(str)
    part["datetime"] = pd.to_datetime(dt_text, errors="coerce")
    part = part.dropna(subset=["datetime"])
    if part.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    out = pd.DataFrame({
        "stock_code": stock_code,
        "datetime": part["datetime"].values,
        "open": pd.to_numeric(part["开盘"], errors="coerce"),
        "high": pd.to_numeric(part["最高"], errors="coerce"),
        "low": pd.to_numeric(part["最低"], errors="coerce"),
        "close": pd.to_numeric(part["收盘"], errors="coerce"),
        "volume": pd.to_numeric(part["成交量"], errors="coerce"),
        "amount": pd.to_numeric(part["成交额"], errors="coerce"),
    })

    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    out = out[out["open"] > 0]
    if out.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    out["stock_code"] = out["stock_code"].astype("object")
    out["datetime"] = pd.to_datetime(out["datetime"])
    for col in ["open", "high", "low", "close", "amount"]:
        out[col] = out[col].astype("float64")
    out["volume"] = out["volume"].fillna(0).round().astype("int64")
    return out[TARGET_COLUMNS]


class MonthlySqliteWriter:
    def __init__(self, temp_root: Path, commit_every: int = 500_000):
        self.temp_root = temp_root
        self.commit_every = int(commit_every)
        self.connections: Dict[str, sqlite3.Connection] = {}
        self.pending_rows: Dict[str, int] = {}

    def _connect(self, month: str) -> sqlite3.Connection:
        conn = self.connections.get(month)
        if conn is not None:
            return conn

        month_dir = self.temp_root / month
        month_dir.mkdir(parents=True, exist_ok=True)
        db_path = month_dir / "minute_import.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS minute_data (
                stock_code TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                amount REAL,
                PRIMARY KEY (stock_code, datetime)
            )
        """)
        self.connections[month] = conn
        self.pending_rows[month] = 0
        return conn

    def insert(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        df = df.copy()
        df["_month"] = df["datetime"].dt.strftime("%Y%m")
        written = 0
        for month, month_df in df.groupby("_month", sort=False):
            conn = self._connect(month)
            out = month_df.drop(columns=["_month"]).copy()
            out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            records = list(out[SQL_COLUMNS].itertuples(index=False, name=None))
            conn.executemany(
                """
                INSERT OR REPLACE INTO minute_data
                (stock_code, datetime, open, high, low, close, volume, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            row_count = len(records)
            self.pending_rows[month] = self.pending_rows.get(month, 0) + row_count
            if self.pending_rows[month] >= self.commit_every:
                conn.commit()
                self.pending_rows[month] = 0
            written += row_count
        return written

    def close(self):
        for conn in self.connections.values():
            conn.commit()
            conn.close()
        self.connections.clear()
        self.pending_rows.clear()


def write_temp_parts_by_month(df: pd.DataFrame, temp_root: Path, part_counters: Dict[str, int]) -> int:
    """Legacy parquet-part writer kept for reference; SQLite writer is used by default."""
    if df.empty:
        return 0

    df = df.copy()
    df["_month"] = df["datetime"].dt.strftime("%Y%m")
    written = 0
    for month, month_df in df.groupby("_month", sort=False):
        month_dir = temp_root / month
        month_dir.mkdir(parents=True, exist_ok=True)
        part_idx = part_counters.get(month, 0) + 1
        part_counters[month] = part_idx
        part_file = month_dir / f"part_{part_idx:06d}.parquet"
        month_df = month_df.drop(columns=["_month"])
        month_df.to_parquet(part_file, index=False, compression="snappy", engine="pyarrow")
        written += len(month_df)
    return written


def read_parquet_files(files: List[Path]) -> pd.DataFrame:
    frames = []
    for file in files:
        frames.append(pd.read_parquet(file))
    if not frames:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def finalize_month(
    month: str,
    temp_month_dir: Path,
    target_root: Path,
    overwrite_range: bool,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    logger: logging.Logger,
) -> int:
    sqlite_file = temp_month_dir / "minute_import.sqlite"
    part_files = sorted(temp_month_dir.glob("part_*.parquet"))
    if not sqlite_file.exists() and not part_files:
        return 0

    target_dir = target_root / month
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "minute_data.parquet"

    if sqlite_file.exists():
        conn = sqlite3.connect(str(sqlite_file))
        new_df = pd.read_sql_query(
            """
            SELECT stock_code, datetime, open, high, low, close, volume, amount
            FROM minute_data
            ORDER BY stock_code, datetime
            """,
            conn,
        )
        conn.close()
    else:
        new_df = read_parquet_files(part_files)
    new_df = new_df[TARGET_COLUMNS]
    new_df["datetime"] = pd.to_datetime(new_df["datetime"])

    month_start = pd.to_datetime(f"{month}01", format="%Y%m%d")
    month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    overwrite_full_month = overwrite_range and start_ts <= month_start and end_ts >= month_end

    if target_file.exists() and not overwrite_full_month:
        existing_df = pd.read_parquet(target_file)
        existing_df = existing_df[[col for col in TARGET_COLUMNS if col in existing_df.columns]].copy()
        missing_cols = [col for col in TARGET_COLUMNS if col not in existing_df.columns]
        if missing_cols:
            raise ValueError(f"目标文件 {target_file} 缺少字段: {missing_cols}")
        existing_df = existing_df[TARGET_COLUMNS]
        existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])

        if overwrite_range:
            in_range = existing_df["datetime"].between(start_ts, end_ts, inclusive="both")
            existing_df = existing_df.loc[~in_range].copy()

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        original_rows = len(existing_df)
    else:
        combined = new_df
        original_rows = 0 if not target_file.exists() else -1

    combined = combined.drop_duplicates(["stock_code", "datetime"], keep="last")
    combined = combined.sort_values(["stock_code", "datetime"]).reset_index(drop=True)
    combined = combined[TARGET_COLUMNS]

    tmp_file = target_file.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp_file, index=False, compression="snappy", engine="pyarrow")
    tmp_file.replace(target_file)

    logger.info(
        "完成分区 %s: 原有保留 %s 行, 新导入 %s 行, 最终 %s 行 -> %s",
        month,
        "整月覆盖" if original_rows < 0 else f"{original_rows:,}",
        f"{len(new_df):,}",
        f"{len(combined):,}",
        target_file,
    )
    return len(combined)


def update_metadata(target_root: Path, logger: logging.Logger) -> None:
    metadata_file = target_root.parent / "metadata" / "download_metadata.json"
    if not metadata_file.exists():
        logger.info("未找到metadata文件，跳过元数据更新: %s", metadata_file)
        return

    try:
        import json

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        months = sorted(
            p.name for p in target_root.iterdir()
            if p.is_dir() and (p / "minute_data.parquet").exists() and len(p.name) == 6 and p.name.isdigit()
        )
        metadata.setdefault("partitions", {})["minute"] = months
        metadata.setdefault("last_update", {})["minute"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("metadata已更新: %s", metadata_file)
    except Exception as exc:
        logger.warning("metadata更新失败，不影响parquet结果: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Baidu 5min CSV files into local monthly minute parquet files.")
    parser.add_argument("--source_dir", default=str(DEFAULT_SOURCE_DIR), help="单股票CSV目录")
    parser.add_argument("--target_root", default=str(DEFAULT_TARGET_ROOT), help="目标 local_data/minute 目录")
    parser.add_argument("--start_date", default=DEFAULT_START_DATE, help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", default=DEFAULT_END_DATE, help="结束日期 YYYYMMDD")
    parser.add_argument("--chunksize", type=int, default=500_000, help="CSV分块读取行数")
    parser.add_argument("--sqlite_commit_rows", type=int, default=500_000, help="SQLite每累计N行提交一次")
    parser.add_argument("--include_bj", action="store_true", help="同时导入北交所 bj*.csv，默认跳过")
    parser.add_argument("--overwrite_range", action="store_true", help="合并时删除目标文件中日期范围内旧记录，再写入本次导入记录")
    parser.add_argument("--keep_temp", action="store_true", help="保留临时分片目录，便于排查")
    parser.add_argument("--limit_files", type=int, default=0, help="调试用，只处理前N个CSV文件")
    parser.add_argument("--log_file", default="", help="日志文件路径，默认写到目标目录 import_baidu_5min_时间.log")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_root = Path(args.target_root)
    start_ts = parse_yyyymmdd(args.start_date)
    end_ts = parse_yyyymmdd(args.end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    target_root.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file) if args.log_file else target_root / f"import_baidu_5min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_root = target_root / "_baidu_5min_import_tmp" / run_id
    temp_root.mkdir(parents=True, exist_ok=True)

    files = list(iter_source_files(source_dir, include_bj=args.include_bj))
    if args.limit_files and args.limit_files > 0:
        files = files[:args.limit_files]

    logger.info("源目录: %s", source_dir)
    logger.info("目标目录: %s", target_root)
    logger.info("导入日期: %s ~ %s", args.start_date, args.end_date)
    logger.info("CSV文件数: %s", f"{len(files):,}")
    logger.info("临时目录: %s", temp_root)

    total_rows = 0
    failed_files: List[str] = []
    writer = MonthlySqliteWriter(temp_root, commit_every=args.sqlite_commit_rows)

    try:
        for idx, csv_file in enumerate(files, 1):
            stock_code = stock_code_from_file(csv_file)
            if stock_code is None:
                logger.warning("无法识别股票代码，跳过: %s", csv_file)
                continue

            if idx == 1 or idx % 100 == 0:
                logger.info("处理进度: %s/%s 当前: %s", f"{idx:,}", f"{len(files):,}", csv_file.name)

            try:
                file_rows = 0
                reader = pd.read_csv(
                    csv_file,
                    usecols=CSV_COLUMNS,
                    chunksize=args.chunksize,
                    encoding="utf-8",
                    low_memory=False,
                )
                for chunk in reader:
                    out = transform_chunk(chunk, stock_code, start_ts, end_ts)
                    written = writer.insert(out)
                    file_rows += written
                    total_rows += written
                    del chunk, out
                if file_rows == 0:
                    logger.debug("日期范围内无数据: %s", csv_file.name)
            except UnicodeDecodeError:
                try:
                    file_rows = 0
                    reader = pd.read_csv(
                        csv_file,
                        usecols=CSV_COLUMNS,
                        chunksize=args.chunksize,
                        encoding="gbk",
                        low_memory=False,
                    )
                    for chunk in reader:
                        out = transform_chunk(chunk, stock_code, start_ts, end_ts)
                        written = writer.insert(out)
                        file_rows += written
                        total_rows += written
                        del chunk, out
                except Exception as exc:
                    logger.exception("处理失败: %s | %s", csv_file, exc)
                    failed_files.append(str(csv_file))
            except Exception as exc:
                logger.exception("处理失败: %s | %s", csv_file, exc)
                failed_files.append(str(csv_file))

            if idx % 200 == 0:
                gc.collect()
    finally:
        writer.close()

    logger.info("CSV扫描完成，临时分片记录数: %s", f"{total_rows:,}")
    if failed_files:
        logger.error("存在失败文件 %s 个，终止合并。失败示例: %s", len(failed_files), failed_files[:10])
        return 2

    month_dirs = sorted([p for p in temp_root.iterdir() if p.is_dir()])
    if not month_dirs:
        logger.warning("没有生成任何临时分片，结束")
        return 0

    for month_dir in month_dirs:
        finalize_month(
            month=month_dir.name,
            temp_month_dir=month_dir,
            target_root=target_root,
            overwrite_range=args.overwrite_range,
            start_ts=start_ts,
            end_ts=end_ts,
            logger=logger,
        )
        gc.collect()

    update_metadata(target_root, logger)

    if not args.keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)
        logger.info("已清理临时目录: %s", temp_root)
    else:
        logger.info("按参数保留临时目录: %s", temp_root)

    logger.info("全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
