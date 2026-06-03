import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
METRICS = RESULTS / "walkforward_metrics_20260520_210454.json"
SHADOW_CSV = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.csv"
MINUTE_ROOT = BASE.parent / "local_data" / "minute"


def month_path(dt):
    return MINUTE_ROOT / dt.strftime("%Y%m") / "minute_data.parquet"


def load_stock_month(stock_code, dt):
    path = month_path(dt)
    if not path.exists():
        return pd.DataFrame()
    table = pq.read_table(
        path,
        filters=[("stock_code", "=", stock_code)],
        columns=["stock_code", "datetime", "open", "high", "low", "close", "volume", "amount"],
    )
    df = table.to_pandas()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def load_stock_around(stock_code, dt):
    frames = []
    for month_dt in [dt - pd.DateOffset(months=1), dt, dt + pd.DateOffset(months=1)]:
        df = load_stock_month(stock_code, pd.Timestamp(month_dt))
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["stock_code", "datetime"])
    return out.sort_values("datetime").reset_index(drop=True)


def load_actual_trades():
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = []
    for t in data[0]["all_results"]["1"]["trades"]:
        rows.append(
            {
                "source": "actual",
                "stock_code": t["stock_code"],
                "buy_datetime": pd.Timestamp(t["buy_datetime"]),
                "sell_datetime": pd.Timestamp(t["sell_datetime"]),
                "buy_price": float(t["buy_price"]),
                "sell_price": float(t["sell_price"]),
                "quantity": float(t["quantity"]),
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
                "exit_reason": t.get("exit_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def load_shadow_trades():
    if not SHADOW_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SHADOW_CSV)
    keep = [
        "stock_code",
        "buy_datetime",
        "sell_datetime",
        "buy_price",
        "sell_price",
        "quantity",
        "actual_return",
        "profit_loss",
        "exit_reason",
        "shadow_status",
        "entry_gap_ratio",
        "prev_close",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["source"] = "shadow"
    df["buy_datetime"] = pd.to_datetime(df["buy_datetime"])
    df["sell_datetime"] = pd.to_datetime(df["sell_datetime"])
    for c in ["buy_price", "sell_price", "quantity", "actual_return", "profit_loss", "entry_gap_ratio", "prev_close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_intraday_features(row):
    rec = row.to_dict()
    bars = load_stock_around(row["stock_code"], row["buy_datetime"])
    if bars.empty:
        rec["minute_available"] = False
        return rec

    buy_dt = row["buy_datetime"]
    day = bars[bars["datetime"].dt.date == buy_dt.date()].copy()
    prev = bars[bars["datetime"].dt.date < buy_dt.date()]
    if day.empty or prev.empty:
        rec["minute_available"] = False
        return rec

    first = day.iloc[0]
    prev_close = float(prev.iloc[-1]["close"])
    open_price = float(first["open"])
    close_935 = float(first["close"])
    high_935 = float(first["high"])
    low_935 = float(first["low"])

    rec["minute_available"] = True
    rec["prev_close_5m"] = prev_close
    rec["open_5m"] = open_price
    rec["close_0935"] = close_935
    rec["high_0935"] = high_935
    rec["low_0935"] = low_935
    rec["open_gap"] = open_price / prev_close - 1.0 if prev_close > 0 else np.nan
    rec["ret_0935_vs_open"] = close_935 / open_price - 1.0 if open_price > 0 else np.nan
    rec["high_0935_vs_open"] = high_935 / open_price - 1.0 if open_price > 0 else np.nan
    rec["low_0935_vs_open"] = low_935 / open_price - 1.0 if open_price > 0 else np.nan
    rec["buy_vs_open"] = row["buy_price"] / open_price - 1.0 if open_price > 0 else np.nan
    rec["final_return"] = float(row["actual_return"])
    rec["final_return_pct"] = float(row["actual_return"]) * 100.0
    rec["is_big_loss"] = bool(row["actual_return"] <= -0.05)
    rec["is_big_win"] = bool(row["actual_return"] >= 0.10)
    return rec


def summarize_group(df, group_cols):
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n=("stock_code", "size"),
        actual_n=("source", lambda s: int((s == "actual").sum())),
        shadow_n=("source", lambda s: int((s == "shadow").sum())),
        avg_final_ret=("final_return", "mean"),
        median_final_ret=("final_return", "median"),
        sum_profit_loss=("profit_loss", "sum"),
        big_loss_rate=("is_big_loss", "mean"),
        big_win_rate=("is_big_win", "mean"),
        avg_open_gap=("open_gap", "mean"),
        avg_0935_ret=("ret_0935_vs_open", "mean"),
    ).reset_index()
    for c in ["avg_final_ret", "median_final_ret", "big_loss_rate", "big_win_rate", "avg_open_gap", "avg_0935_ret"]:
        out[c] = out[c] * 100.0
    return out


def trigger_stats(df, name, mask):
    hit = df[mask].copy()
    miss = df[~mask].copy()
    def stats(part, prefix):
        if part.empty:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_avg_ret": np.nan,
                f"{prefix}_median_ret": np.nan,
                f"{prefix}_big_loss_rate": np.nan,
                f"{prefix}_big_win_rate": np.nan,
                f"{prefix}_sum_pnl": 0.0,
            }
        return {
            f"{prefix}_n": len(part),
            f"{prefix}_avg_ret": part["final_return"].mean() * 100,
            f"{prefix}_median_ret": part["final_return"].median() * 100,
            f"{prefix}_big_loss_rate": part["is_big_loss"].mean() * 100,
            f"{prefix}_big_win_rate": part["is_big_win"].mean() * 100,
            f"{prefix}_sum_pnl": part["profit_loss"].sum(),
        }
    rec = {"trigger": name}
    rec.update(stats(hit, "hit"))
    rec.update(stats(miss, "miss"))
    rec["hit_half_delta_pnl_if_halved"] = -0.5 * hit["profit_loss"].sum()
    rec["hit_big_loss_count"] = int(hit["is_big_loss"].sum())
    rec["hit_big_win_count"] = int(hit["is_big_win"].sum())
    return rec


def main():
    actual = load_actual_trades()
    shadow = load_shadow_trades()
    all_trades = pd.concat([actual, shadow], ignore_index=True, sort=False)
    all_trades = all_trades.sort_values(["buy_datetime", "source", "stock_code"]).reset_index(drop=True)

    rows = [add_intraday_features(row) for _, row in all_trades.iterrows()]
    out = pd.DataFrame(rows)
    out = out[out["minute_available"]].copy()

    out["open_gap_bin"] = pd.cut(
        out["open_gap"],
        bins=[-np.inf, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, np.inf],
        labels=["<=-5%", "-5~-3%", "-3~-1%", "-1~0%", "0~1%", "1~3%", "3~5%", ">5%"],
    )
    out["ret_0935_bin"] = pd.cut(
        out["ret_0935_vs_open"],
        bins=[-np.inf, -0.03, -0.015, -0.005, 0.0, 0.005, 0.015, 0.03, np.inf],
        labels=["<=-3%", "-3~-1.5%", "-1.5~-0.5%", "-0.5~0%", "0~0.5%", "0.5~1.5%", "1.5~3%", ">3%"],
    )

    out_path = RESULTS / "open_935_return_relationship_185_detail.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    summaries = []
    summaries.append(summarize_group(out, ["source"]))
    summaries[-1].insert(0, "summary_type", "source")
    summaries.append(summarize_group(out, ["open_gap_bin"]))
    summaries[-1].insert(0, "summary_type", "open_gap_bin")
    summaries.append(summarize_group(out, ["ret_0935_bin"]))
    summaries[-1].insert(0, "summary_type", "ret_0935_bin")
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    summary.to_csv(RESULTS / "open_935_return_relationship_185_summary.csv", index=False, encoding="utf-8-sig")

    cross = summarize_group(out, ["open_gap_bin", "ret_0935_bin"])
    cross = cross[cross["n"] >= 2].sort_values(["open_gap_bin", "ret_0935_bin"])
    cross.to_csv(RESULTS / "open_935_return_relationship_185_cross_bins.csv", index=False, encoding="utf-8-sig")

    trigger_defs = {
        "open_gap_lt_0": out["open_gap"] < 0,
        "open_gap_le_-1pct": out["open_gap"] <= -0.01,
        "open_gap_le_-3pct": out["open_gap"] <= -0.03,
        "ret0935_lt_0": out["ret_0935_vs_open"] < 0,
        "ret0935_le_-0.5pct": out["ret_0935_vs_open"] <= -0.005,
        "ret0935_le_-1pct": out["ret_0935_vs_open"] <= -0.01,
        "open_gap_lt_0_and_ret0935_lt_0": (out["open_gap"] < 0) & (out["ret_0935_vs_open"] < 0),
        "open_gap_le_-1pct_and_ret0935_lt_0": (out["open_gap"] <= -0.01) & (out["ret_0935_vs_open"] < 0),
        "open_gap_lt_0_and_ret0935_le_-0.5pct": (out["open_gap"] < 0) & (out["ret_0935_vs_open"] <= -0.005),
        "open_gap_le_-1pct_or_ret0935_le_-1pct": (out["open_gap"] <= -0.01) | (out["ret_0935_vs_open"] <= -0.01),
        "ret0935_gt_1.5pct": out["ret_0935_vs_open"] > 0.015,
        "ret0935_gt_3pct": out["ret_0935_vs_open"] > 0.03,
    }
    trig = pd.DataFrame([trigger_stats(out, k, v) for k, v in trigger_defs.items()])
    trig = trig.sort_values(["hit_avg_ret", "hit_big_loss_rate"], ascending=[True, False])
    trig.to_csv(RESULTS / "open_935_return_relationship_185_trigger_stats.csv", index=False, encoding="utf-8-sig")

    corr = out[["open_gap", "ret_0935_vs_open", "low_0935_vs_open", "high_0935_vs_open", "final_return"]].corr(
        method="spearman"
    )
    corr.to_csv(RESULTS / "open_935_return_relationship_185_spearman_corr.csv", encoding="utf-8-sig")

    print("rows", len(out), "actual", int((out["source"] == "actual").sum()), "shadow", int((out["source"] == "shadow").sum()))
    print("\n== spearman corr with final_return ==")
    print(corr["final_return"].sort_values().to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n== trigger stats top risky ==")
    cols = [
        "trigger",
        "hit_n",
        "hit_avg_ret",
        "hit_median_ret",
        "hit_big_loss_rate",
        "hit_big_win_rate",
        "hit_sum_pnl",
        "hit_half_delta_pnl_if_halved",
        "hit_big_loss_count",
        "hit_big_win_count",
    ]
    print(trig[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n== 0935 bin ==")
    print(summaries[2].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\noutputs:")
    for p in [
        out_path,
        RESULTS / "open_935_return_relationship_185_summary.csv",
        RESULTS / "open_935_return_relationship_185_cross_bins.csv",
        RESULTS / "open_935_return_relationship_185_trigger_stats.csv",
        RESULTS / "open_935_return_relationship_185_spearman_corr.csv",
    ]:
        print(p)


if __name__ == "__main__":
    main()
