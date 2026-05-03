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
    for row in trades.itertuples(index=False):
        buy_ts = pd.Timestamp(row.buy_date)
        month_key = buy_ts.strftime("%Y%m")
        required.setdefault(month_key, set()).add((row.stock_code, buy_ts.strftime("%Y-%m-%d")))
    return required


def load_buyday_minute_bars(required_pairs: Dict[str, set[Tuple[str, str]]]) -> pd.DataFrame:
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
            columns=["stock_code", "datetime", "open", "close", "high", "low"],
            filter=ds.field("stock_code").isin(stock_codes),
        )
        if table.num_rows == 0:
            continue
        df = table.to_pandas()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df = df[
            df[["stock_code", "date_str"]]
            .apply(lambda x: (x["stock_code"], x["date_str"]) in pair_keys, axis=1)
        ].copy()
        if df.empty:
            continue
        df["time_hm"] = df["datetime"].dt.strftime("%H:%M")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_code", "datetime", "open", "close", "high", "low", "date_str", "time_hm"])
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "datetime"]).reset_index(drop=True)


def derive_entry_prices(trades: pd.DataFrame, minute_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = {
        (code, date_str): grp.sort_values("datetime").reset_index(drop=True)
        for (code, date_str), grp in minute_df.groupby(["stock_code", "date_str"], sort=False)
    }

    for row in trades.itertuples(index=False):
        key = (row.stock_code, pd.Timestamp(row.buy_date).strftime("%Y-%m-%d"))
        grp = grouped.get(key)
        if grp is None or grp.empty:
            continue
        bar_0935 = grp[grp["time_hm"] == "09:35"]
        bar_0940 = grp[grp["time_hm"] == "09:40"]
        if bar_0935.empty:
            continue
        rows.append(
            {
                "stock_code": row.stock_code,
                "buy_datetime": row.buy_datetime,
                "buy_date": row.buy_date,
                "sell_datetime": row.sell_datetime,
                "sell_price": row.sell_price,
                "actual_return_base": row.actual_return,
                "profit_loss_base": row.profit_loss,
                "buy_price_0930_open": row.buy_price,
                "buy_price_0935_close": float(bar_0935.iloc[0]["close"]),
                "buy_price_0935_next_open": float(bar_0940.iloc[0]["open"]) if not bar_0940.empty else None,
                "exit_reason": row.exit_reason,
            }
        )
    return pd.DataFrame(rows)


def evaluate_variant(df: pd.DataFrame, variant_col: str) -> pd.DataFrame:
    probe = df.copy()
    probe = probe[probe[variant_col].notna()].copy()
    probe["alt_buy_price"] = probe[variant_col].astype(float)
    probe["alt_return"] = probe["sell_price"] / probe["alt_buy_price"] - 1.0
    probe["bigloss5"] = probe["alt_return"] <= -0.05
    probe["bigloss7"] = probe["alt_return"] <= -0.07
    probe["bigwin10"] = probe["alt_return"] >= 0.10
    probe["delta_vs_base"] = probe["alt_return"] - probe["actual_return_base"]
    summary = {
        "variant": variant_col,
        "trade_count": len(probe),
        "avg_return": probe["alt_return"].mean(),
        "ret_sum": probe["alt_return"].sum(),
        "base_ret_sum": probe["actual_return_base"].sum(),
        "ret_sum_delta_vs_base": probe["alt_return"].sum() - probe["actual_return_base"].sum(),
        "better_than_base_count": int((probe["delta_vs_base"] > 0).sum()),
        "worse_than_base_count": int((probe["delta_vs_base"] < 0).sum()),
        "avg_delta_vs_base": probe["delta_vs_base"].mean(),
        "bigloss5_count": int(probe["bigloss5"].sum()),
        "bigloss7_count": int(probe["bigloss7"].sum()),
        "bigwin10_count": int(probe["bigwin10"].sum()),
    }
    return probe, pd.DataFrame([summary])


def main() -> None:
    trades = load_trades()
    required_pairs = build_required_pairs(trades)
    minute_df = load_buyday_minute_bars(required_pairs)
    derived = derive_entry_prices(trades, minute_df)

    probes: List[pd.DataFrame] = []
    summaries: List[pd.DataFrame] = []
    for variant in ["buy_price_0935_close", "buy_price_0935_next_open"]:
        probe, summary = evaluate_variant(derived, variant)
        probe["variant"] = variant
        probes.append(probe)
        summaries.append(summary)

    probe_df = pd.concat(probes, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True)

    probe_path = RESULTS_DIR / "entry_0935_variants_probe.csv"
    summary_path = RESULTS_DIR / "entry_0935_variants_summary.csv"
    probe_df.to_csv(probe_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(summary_df.to_string(index=False))
    print(f"\nprobe -> {probe_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
