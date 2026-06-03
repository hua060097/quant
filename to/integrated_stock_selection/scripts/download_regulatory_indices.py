#!/usr/bin/env python3
"""
Download regulatory benchmark index daily data from Tushare.

The output is used by reg_severe_up_10d_ratio:
- 000002.SH: SSE A Share Index
- 399107.SZ: SZSE A Share Index
- 399102.SZ: ChiNext Composite Index
- 000688.SH: STAR 50 Index
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import tushare as ts


REGULATORY_INDEX_CODES = ["000002.SH", "399107.SZ", "399102.SZ", "000688.SH"]
API_RETRY_COUNT = 10
API_MIN_INTERVAL_SECONDS = 0.25

PROJECT_PARENT = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = PROJECT_PARENT / "local_data"
OUTPUT_DIR = LOCAL_DATA_DIR / "regulatory_indices"
LOG_DIR = LOCAL_DATA_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download_regulatory_indices.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("download_regulatory_indices")


def _normalize_yyyymmdd(value: str) -> str:
    return pd.to_datetime(str(value).replace("-", "")).strftime("%Y%m%d")


def _init_tushare(token: Optional[str]):
    tushare_token = token or os.environ.get("TUSHARE_TOKEN")
    if tushare_token:
        ts.set_token(tushare_token)
    return ts.pro_api()


def _retry_tushare_call(func, *args, **kwargs):
    last_error = None
    for attempt in range(1, API_RETRY_COUNT + 1):
        try:
            time.sleep(API_MIN_INTERVAL_SECONDS + random.random() * 0.05)
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            if any(key in msg for key in ("频率", "limit", "frequency", "每分钟", "每小时")):
                sleep_seconds = 45 + random.random() * 10
            else:
                sleep_seconds = min(2 ** attempt, 30) + random.random()
            logger.warning(
                "Tushare call failed, retrying in %.1fs (%s/%s): %s",
                sleep_seconds,
                attempt,
                API_RETRY_COUNT,
                exc,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Tushare call failed after {API_RETRY_COUNT} retries: {last_error}") from last_error


def _fetch_index_daily(pro, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = _retry_tushare_call(
        pro.index_daily,
        ts_code=index_code,
        start_date=start_date,
        end_date=end_date,
    )
    if df is None or df.empty:
        logger.warning("No data returned for %s", index_code)
        return pd.DataFrame()

    df = df.copy()
    if "trade_date" in df.columns:
        df = df.rename(columns={"trade_date": "date"})
    if "ts_code" in df.columns:
        df = df.rename(columns={"ts_code": "index_code"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["index_code"] = index_code
    df = df.dropna(subset=["date"])
    df = df.sort_values(["index_code", "date"]).reset_index(drop=True)
    return df


def _merge_existing(output_file: Path, new_data: pd.DataFrame) -> pd.DataFrame:
    if output_file.exists():
        old_data = pd.read_parquet(output_file)
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data.copy()

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["index_code"] = combined["index_code"].astype(str).str.upper()
    combined = combined.dropna(subset=["date", "index_code"])
    combined = combined.sort_values(["index_code", "date"]).drop_duplicates(
        ["index_code", "date"],
        keep="last",
    )
    return combined.reset_index(drop=True)


def download_regulatory_indices(
    start_date: str,
    end_date: str,
    codes: Iterable[str] = REGULATORY_INDEX_CODES,
    output_dir: Path = OUTPUT_DIR,
    token: Optional[str] = None,
) -> Path:
    start = _normalize_yyyymmdd(start_date)
    end = _normalize_yyyymmdd(end_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    pro = _init_tushare(token)

    frames: List[pd.DataFrame] = []
    for index_code in codes:
        code = str(index_code).strip().upper()
        logger.info("Downloading %s: %s ~ %s", code, start, end)
        df = _fetch_index_daily(pro, code, start, end)
        if not df.empty:
            frames.append(df)
            per_code_file = output_dir / f"{code.replace('.', '_')}.parquet"
            _merge_existing(per_code_file, df).to_parquet(per_code_file, index=False)
            logger.info("Saved %s rows to %s", len(df), per_code_file)

    if not frames:
        raise RuntimeError("No regulatory index data downloaded.")

    output_file = output_dir / "regulatory_indices.parquet"
    combined = _merge_existing(output_file, pd.concat(frames, ignore_index=True))
    combined.to_parquet(output_file, index=False)
    logger.info(
        "Saved combined regulatory indices: %s rows, %s indices -> %s",
        len(combined),
        combined["index_code"].nunique(),
        output_file,
    )
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Download regulatory benchmark index daily data.")
    parser.add_argument("--start", "--start_date", required=True, help="Start date, e.g. 20221201")
    parser.add_argument("--end", "--end_date", required=True, help="End date, e.g. 20260529")
    parser.add_argument(
        "--codes",
        nargs="*",
        default=REGULATORY_INDEX_CODES,
        help="Index codes to download.",
    )
    parser.add_argument("--token", default=None, help="Optional Tushare token. Defaults to TUSHARE_TOKEN env.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Output directory.")
    args = parser.parse_args()

    download_regulatory_indices(
        start_date=args.start,
        end_date=args.end,
        codes=args.codes,
        output_dir=Path(args.output_dir),
        token=args.token,
    )


if __name__ == "__main__":
    main()
