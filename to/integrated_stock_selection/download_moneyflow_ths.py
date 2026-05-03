#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DAILY_PATH = BASE_DIR / "download_daily.py"
OUTPUT_DIR = BASE_DIR / "local_data" / "moneyflow_ths"
FEATURE_DAILY_DIR = BASE_DIR / "local_data" / "features" / "daily"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tushare THS moneyflow data by trade date.")
    parser.add_argument("--start", required=True, help="Start date, format YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date, format YYYYMMDD")
    parser.add_argument("--max-workers", type=int, default=3, help="Max concurrent workers, default 3")
    parser.add_argument("--retry", type=int, default=4, help="Retry count per trade date, default 4")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing parquet files instead of overwrite")
    return parser.parse_args()


def validate_date_str(value: str) -> str:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date: {value}, expected YYYYMMDD") from exc
    return value


def extract_token_from_download_daily(path: Path) -> str:
    text = path.read_text(encoding="utf-8")

    assign_match = re.search(r"TUSHARE_TOKEN\s*=\s*(['\"])(.+?)\1", text)
    if assign_match:
        return assign_match.group(2)

    for line in text.splitlines():
        if "ts.set_token" in line:
            try:
                node = ast.parse(line.strip())
                call = node.body[0].value
                if getattr(call.func, "attr", "") == "set_token" and call.args:
                    arg = call.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        return arg.value
            except Exception:
                continue

    env_match = re.search(r"os\.getenv\((['\"])(.+?)\1", text)
    if env_match:
        env_name = env_match.group(2)
        token = os.getenv(env_name)
        if token:
            return token

    raise RuntimeError(f"Could not extract tushare token from {path}")


def load_trade_dates(start_date: str, end_date: str) -> list[str]:
    if FEATURE_DAILY_DIR.exists():
        dates = []
        for file in FEATURE_DAILY_DIR.glob("*.parquet"):
            m = re.fullmatch(r"(\d{8})\.parquet", file.name)
            if m:
                d = m.group(1)
                if start_date <= d <= end_date:
                    dates.append(d)
        if dates:
            return sorted(set(dates))
    raise RuntimeError(f"Trading date source not found or empty: {FEATURE_DAILY_DIR}")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def output_path(trade_date: str) -> Path:
    return OUTPUT_DIR / f"{trade_date}.parquet"


def download_one_day(
    pro,
    trade_date: str,
    retry: int,
    skip_existing: bool,
) -> tuple[str, str, int]:
    path = output_path(trade_date)
    if skip_existing and path.exists():
        return trade_date, "skipped", 0

    last_err = None
    for attempt in range(1, retry + 1):
        try:
            df = pro.moneyflow_ths(trade_date=trade_date)
            if df is None:
                df = pd.DataFrame()
            if not df.empty:
                df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
            tmp = path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            return trade_date, "ok", len(df)
        except Exception as exc:
            last_err = exc
            sleep_s = min(8, 1.2 * attempt) + random.uniform(0.2, 0.8)
            time.sleep(sleep_s)
    return trade_date, f"failed: {last_err}", -1


def main() -> None:
    args = parse_args()
    start_date = validate_date_str(args.start)
    end_date = validate_date_str(args.end)
    if start_date > end_date:
        raise ValueError("start date must be <= end date")

    ensure_output_dir()
    token = extract_token_from_download_daily(DOWNLOAD_DAILY_PATH)
    ts.set_token(token)
    pro = ts.pro_api()

    trade_dates = load_trade_dates(start_date, end_date)
    if args.skip_existing:
        trade_dates = [d for d in trade_dates if not output_path(d).exists()]

    print(f"[INFO] trade dates: {len(trade_dates)}")
    print(f"[INFO] output dir: {OUTPUT_DIR}")
    print(f"[INFO] max workers: {args.max_workers}")

    success = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    rows = 0

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(download_one_day, pro, d, args.retry, args.skip_existing): d
            for d in trade_dates
        }
        for future in as_completed(futures):
            trade_date, status, count = future.result()
            if status == "ok":
                success += 1
                rows += max(0, count)
                print(f"[OK] {trade_date}: rows={count}")
            elif status == "skipped":
                skipped += 1
                print(f"[SKIP] {trade_date}")
            else:
                failed.append((trade_date, status))
                print(f"[FAIL] {trade_date}: {status}")

    print(
        f"[DONE] success={success} skipped={skipped} failed={len(failed)} total_rows={rows}"
    )
    if failed:
        print("[FAILED_DATES]")
        for trade_date, status in failed:
            print(f"  {trade_date}: {status}")


if __name__ == "__main__":
    main()
