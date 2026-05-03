#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CALL_AUCTION_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\call_auction_open")
DAILY_FEATURE_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\features\daily")

JSON_PATHS = [
    RESULTS_DIR / "daily_top5_details_Window_3_20260417_145807.json",
    RESULTS_DIR / "daily_top5_details_Window_3_20260417_154536.json",
]


@dataclass(frozen=True)
class JsonConfig:
    path: Path
    label_name: str


CONFIGS = [
    JsonConfig(JSON_PATHS[0], "ret_50_50"),
    JsonConfig(JSON_PATHS[1], "ret_75_25"),
]


def round_price(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def get_limit_ratio(ts_code: str) -> float:
    code = ts_code.split(".")[0]
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("430", "440", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
                        "870", "871", "872", "873", "874", "875", "876", "877", "878", "879", "920", "921",
                        "922")):
        return 0.30
    return 0.10


def compute_limit_up_price(prev_close: float, ts_code: str) -> float:
    return round_price(prev_close * (1.0 + get_limit_ratio(ts_code)))


def load_json_rows(config: JsonConfig) -> pd.DataFrame:
    data = json.loads(config.path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for trade_date, payload in data.items():
        for rank in range(1, 6):
            key = f"top{rank}"
            info = payload.get(key)
            if not info:
                continue
            rows.append(
                {
                    "trade_date": str(trade_date),
                    "rank": rank,
                    "rank_name": key,
                    "stock_code": info["stock_code"],
                    "real_return": float(info["real_return"]),
                    "label_name": config.label_name,
                    "json_name": config.path.name,
                }
            )
    return pd.DataFrame(rows)


def load_call_auction_for_dates(trade_dates: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in sorted(set(trade_dates)):
        path = CALL_AUCTION_DIR / f"{trade_date}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["ts_code", "trade_date", "close", "high", "open", "low", "vwap"])
        df = df.rename(columns={"ts_code": "stock_code"})
        df["trade_date"] = df["trade_date"].astype(str)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_code", "trade_date", "close", "high", "open", "low", "vwap"])
    return pd.concat(frames, ignore_index=True)


def load_daily_features_for_dates(trade_dates: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in sorted(set(trade_dates)):
        path = DAILY_FEATURE_DIR / f"{trade_date}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["stock_code", "date_str", "close", "pct_chg"])
        df = df.rename(columns={"date_str": "trade_date", "close": "daily_close"})
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_code", "trade_date", "daily_close", "pct_chg"])
    return pd.concat(frames, ignore_index=True)


def build_analysis_frame(configs: list[JsonConfig]) -> pd.DataFrame:
    selected = pd.concat([load_json_rows(cfg) for cfg in configs], ignore_index=True)
    auction = load_call_auction_for_dates(selected["trade_date"].unique())
    daily = load_daily_features_for_dates(selected["trade_date"].unique())
    merged = selected.merge(auction, on=["trade_date", "stock_code"], how="left")
    merged = merged.merge(daily, on=["trade_date", "stock_code"], how="left")
    merged["prev_close"] = merged["daily_close"] / (1.0 + merged["pct_chg"] / 100.0)
    merged["limit_up_price"] = merged.apply(
        lambda row: compute_limit_up_price(row["prev_close"], row["stock_code"]) if pd.notna(row["prev_close"]) else pd.NA,
        axis=1,
    )
    merged["auction_high_hit_limit"] = (
        pd.notna(merged["high"])
        & pd.notna(merged["limit_up_price"])
        & (merged["high"] >= (merged["limit_up_price"] - 1e-9))
    )
    merged["ret_negative"] = merged["real_return"] < 0
    merged["ret_non_negative"] = ~merged["ret_negative"]
    return merged


def summarize(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hit = merged[merged["auction_high_hit_limit"]].copy()
    all_summary = []
    hit_summary = []

    for (label_name, rank_name), group in merged.groupby(["label_name", "rank_name"], sort=True):
        hit_group = group[group["auction_high_hit_limit"]]
        all_summary.append(
            {
                "label_name": label_name,
                "rank_name": rank_name,
                "sample_count": len(group),
                "hit_limit_count": int(group["auction_high_hit_limit"].sum()),
                "hit_limit_ratio": group["auction_high_hit_limit"].mean(),
            }
        )
        hit_summary.append(
            {
                "label_name": label_name,
                "rank_name": rank_name,
                "hit_sample_count": len(hit_group),
                "negative_count": int(hit_group["ret_negative"].sum()),
                "negative_prob": hit_group["ret_negative"].mean() if len(hit_group) else pd.NA,
                "non_negative_count": int(hit_group["ret_non_negative"].sum()),
                "avg_return": hit_group["real_return"].mean() if len(hit_group) else pd.NA,
                "median_return": hit_group["real_return"].median() if len(hit_group) else pd.NA,
                "min_return": hit_group["real_return"].min() if len(hit_group) else pd.NA,
                "max_return": hit_group["real_return"].max() if len(hit_group) else pd.NA,
            }
        )

    for label_name, group in merged.groupby("label_name", sort=True):
        hit_group = group[group["auction_high_hit_limit"]]
        all_summary.append(
            {
                "label_name": label_name,
                "rank_name": "all_top1_top5",
                "sample_count": len(group),
                "hit_limit_count": int(group["auction_high_hit_limit"].sum()),
                "hit_limit_ratio": group["auction_high_hit_limit"].mean(),
            }
        )
        hit_summary.append(
            {
                "label_name": label_name,
                "rank_name": "all_top1_top5",
                "hit_sample_count": len(hit_group),
                "negative_count": int(hit_group["ret_negative"].sum()),
                "negative_prob": hit_group["ret_negative"].mean() if len(hit_group) else pd.NA,
                "non_negative_count": int(hit_group["ret_non_negative"].sum()),
                "avg_return": hit_group["real_return"].mean() if len(hit_group) else pd.NA,
                "median_return": hit_group["real_return"].median() if len(hit_group) else pd.NA,
                "min_return": hit_group["real_return"].min() if len(hit_group) else pd.NA,
                "max_return": hit_group["real_return"].max() if len(hit_group) else pd.NA,
            }
        )

    return pd.DataFrame(all_summary), pd.DataFrame(hit_summary)


def main() -> None:
    merged = build_analysis_frame(CONFIGS)
    coverage_df, summary_df = summarize(merged)

    merged.to_csv(RESULTS_DIR / "call_auction_limit_hit_sample_detail_20260418.csv", index=False, encoding="utf-8-sig")
    coverage_df.to_csv(RESULTS_DIR / "call_auction_limit_hit_coverage_20260418.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(RESULTS_DIR / "call_auction_limit_hit_negative_prob_20260418.csv", index=False, encoding="utf-8-sig")

    print("=== Coverage ===")
    print(coverage_df.to_string(index=False))
    print("\n=== Negative Probability When Auction High Hit Limit-Up ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
