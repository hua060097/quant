from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "walkforward_metrics_20260502_002709.json"
MINUTE_ROOT = (ROOT / ".." / "local_data" / "minute").resolve()


def load_trades() -> pd.DataFrame:
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))[0]
    trades = pd.DataFrame(payload["detailed_trades"]["1"])
    trades["buy_datetime"] = pd.to_datetime(trades["buy_datetime"])
    trades["buy_date"] = trades["buy_datetime"].dt.date
    trades["sell_datetime"] = pd.to_datetime(trades["sell_datetime"])
    return trades


def build_required_pairs(trades: pd.DataFrame) -> Dict[str, set[Tuple[str, str]]]:
    required: Dict[str, set[Tuple[str, str]]] = {}
    unique_trade_days = sorted(pd.unique(trades["buy_date"]))
    all_trade_days = pd.DatetimeIndex(pd.to_datetime(unique_trade_days))
    for row in trades.itertuples(index=False):
        buy_ts = pd.Timestamp(row.buy_date)
        prev_days = all_trade_days[all_trade_days < buy_ts][-5:]
        month_key = buy_ts.strftime("%Y%m")
        required.setdefault(month_key, set()).add((row.stock_code, buy_ts.strftime("%Y-%m-%d")))
        for dt in prev_days:
            required.setdefault(dt.strftime("%Y%m"), set()).add((row.stock_code, dt.strftime("%Y-%m-%d")))
    return required


def load_daily_bars(required_pairs: Dict[str, set[Tuple[str, str]]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for month_key, pairs in sorted(required_pairs.items()):
        parquet_path = MINUTE_ROOT / month_key / "minute_data.parquet"
        if not parquet_path.exists():
            continue
        pair_df = pd.DataFrame(sorted(pairs), columns=["stock_code", "date_str"])
        stock_codes = pair_df["stock_code"].unique().tolist()
        pair_keys = set(map(tuple, pair_df[["stock_code", "date_str"]].itertuples(index=False, name=None)))
        dataset = ds.dataset(str(parquet_path), format="parquet")
        table = dataset.to_table(
            columns=["stock_code", "datetime", "open", "close"],
            filter=ds.field("stock_code").isin(stock_codes),
        )
        if table.num_rows == 0:
            continue
        df = table.to_pandas()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df = df[
            df[["stock_code", "date_str"]]
            .apply(lambda x: (x["stock_code"], x["date_str"]) in pair_keys, axis=1)
        ].copy()
        if df.empty:
            continue
        df = df.sort_values(["stock_code", "date", "datetime"])
        agg = (
            df.groupby(["stock_code", "date"], as_index=False)
            .agg(day_open=("open", "first"), day_close=("close", "last"))
        )
        frames.append(agg)
    if not frames:
        return pd.DataFrame(columns=["stock_code", "date", "day_open", "day_close"])
    daily = pd.concat(frames, ignore_index=True).drop_duplicates(["stock_code", "date"])
    return daily.sort_values(["stock_code", "date"]).reset_index(drop=True)


def compute_filter(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    by_stock: Dict[str, pd.DataFrame] = {
        code: grp.sort_values("date").reset_index(drop=True)
        for code, grp in daily.groupby("stock_code", sort=False)
    }
    rows = []
    for row in trades.itertuples(index=False):
        hist = by_stock.get(row.stock_code)
        if hist is None:
            continue
        hist = hist[hist["date"] <= row.buy_date].copy()
        if hist.empty:
            continue
        current = hist[hist["date"] == row.buy_date]
        if current.empty:
            continue
        prev = hist[hist["date"] < row.buy_date].tail(5)
        if len(prev) < 5:
            continue
        prev4 = prev.tail(4)["day_close"].tolist()
        prev5 = prev["day_close"].tolist()
        open_today = float(current.iloc[0]["day_open"])
        ma5_open = (sum(prev4) + open_today) / 5.0
        ma5_prev = sum(prev5) / 5.0
        slope = ma5_open - ma5_prev
        slope_pct = slope / ma5_prev if ma5_prev else None
        rows.append(
            {
                "stock_code": row.stock_code,
                "buy_datetime": row.buy_datetime,
                "buy_date": row.buy_date,
                "buy_price": row.buy_price,
                "sell_price": row.sell_price,
                "actual_return": row.actual_return,
                "profit_loss": row.profit_loss,
                "exit_reason": row.exit_reason,
                "open_today": open_today,
                "ma5_open_today": ma5_open,
                "ma5_prev_close": ma5_prev,
                "ma5_slope_abs": slope,
                "ma5_slope_pct": slope_pct,
                "skip_negative_slope": bool(slope < 0),
                "bigloss5": bool(row.actual_return <= -0.05),
                "bigloss7": bool(row.actual_return <= -0.07),
                "bigwin10": bool(row.actual_return >= 0.10),
            }
        )
    return pd.DataFrame(rows)


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    total = len(probe)
    skipped = probe[probe["skip_negative_slope"]].copy()
    kept = probe[~probe["skip_negative_slope"]].copy()
    summary = {
        "total_trades": total,
        "skipped_trades": len(skipped),
        "kept_trades": len(kept),
        "skip_ratio": len(skipped) / total if total else None,
        "skipped_bigloss5": int(skipped["bigloss5"].sum()),
        "skipped_bigloss7": int(skipped["bigloss7"].sum()),
        "skipped_bigwin10": int(skipped["bigwin10"].sum()),
        "kept_bigloss5": int(kept["bigloss5"].sum()),
        "kept_bigloss7": int(kept["bigloss7"].sum()),
        "kept_bigwin10": int(kept["bigwin10"].sum()),
        "avg_return_skipped": skipped["actual_return"].mean() if len(skipped) else None,
        "avg_return_kept": kept["actual_return"].mean() if len(kept) else None,
        "ret_sum_skipped": skipped["actual_return"].sum(),
        "ret_sum_kept": kept["actual_return"].sum(),
        "profit_sum_skipped": skipped["profit_loss"].sum(),
        "profit_sum_kept": kept["profit_loss"].sum(),
    }
    return pd.DataFrame([summary])


def main() -> None:
    trades = load_trades()
    required_pairs = build_required_pairs(trades)
    daily = load_daily_bars(required_pairs)
    probe = compute_filter(trades, daily)
    summary = summarize(probe)

    probe_path = RESULTS_DIR / "open_ma5_slope_entry_filter_probe.csv"
    summary_path = RESULTS_DIR / "open_ma5_slope_entry_filter_summary.csv"
    probe.to_csv(probe_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"\nprobe -> {probe_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
