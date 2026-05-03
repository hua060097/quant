from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "walkforward_metrics_20260502_134422.json"
FACTORS_PATH = ROOT / "factor_lab_data" / "factors.parquet"
MINUTE_ROOT = (ROOT / ".." / "local_data" / "minute").resolve()


def load_trades_with_vpe() -> pd.DataFrame:
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))[0]
    trades = pd.DataFrame(payload["detailed_trades"]["1"])
    trades["buy_datetime"] = pd.to_datetime(trades["buy_datetime"])
    trades["sell_datetime"] = pd.to_datetime(trades["sell_datetime"])
    trades["buy_date"] = trades["buy_datetime"].dt.normalize()

    factors = pd.read_parquet(
        FACTORS_PATH,
        columns=["date", "stock_code", "volume_price_exhaustion"],
    )
    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    calendar = pd.Index(sorted(factors["date"].dropna().unique()))
    cal_pos = {d: i for i, d in enumerate(calendar)}

    trades["pred_date"] = trades["buy_date"].map(
        lambda d: calendar[cal_pos[d] - 1] if d in cal_pos and cal_pos[d] > 0 else pd.NaT
    )
    trades["t2_date"] = trades["buy_date"].map(
        lambda d: calendar[cal_pos[d] + 1] if d in cal_pos and cal_pos[d] + 1 < len(calendar) else pd.NaT
    )

    merged = trades.merge(
        factors,
        left_on=["stock_code", "pred_date"],
        right_on=["stock_code", "date"],
        how="left",
    )
    return merged


def build_required_pairs(trades: pd.DataFrame) -> Dict[str, set[Tuple[str, str]]]:
    required: Dict[str, set[Tuple[str, str]]] = {}
    for row in trades.itertuples(index=False):
        for dt in (row.pred_date, row.buy_date, row.t2_date):
            if pd.isna(dt):
                continue
            ts = pd.Timestamp(dt)
            required.setdefault(ts.strftime("%Y%m"), set()).add((row.stock_code, ts.strftime("%Y-%m-%d")))
    return required


def load_minute_rows(required_pairs: Dict[str, set[Tuple[str, str]]]) -> pd.DataFrame:
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
            df[["stock_code", "date_str"]].apply(
                lambda x: (x["stock_code"], x["date_str"]) in pair_keys, axis=1
            )
        ].copy()
        if df.empty:
            continue
        df["time_hm"] = df["datetime"].dt.strftime("%H:%M")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_code", "datetime", "open", "close", "high", "low", "date_str", "time_hm"])
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "datetime"]).reset_index(drop=True)


def derive_features(trades: pd.DataFrame, minute_df: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        (code, date_str): grp.sort_values("datetime").reset_index(drop=True)
        for (code, date_str), grp in minute_df.groupby(["stock_code", "date_str"], sort=False)
    }
    rows: List[dict] = []
    for row in trades.itertuples(index=False):
        pred_key = (row.stock_code, pd.Timestamp(row.pred_date).strftime("%Y-%m-%d")) if pd.notna(row.pred_date) else None
        buy_key = (row.stock_code, pd.Timestamp(row.buy_date).strftime("%Y-%m-%d"))
        t2_key = (row.stock_code, pd.Timestamp(row.t2_date).strftime("%Y-%m-%d")) if pd.notna(row.t2_date) else None

        pred_df = grouped.get(pred_key) if pred_key else None
        buy_df = grouped.get(buy_key)
        t2_df = grouped.get(t2_key) if t2_key else None
        if pred_df is None or buy_df is None:
            continue

        prev_close = float(pred_df.iloc[-1]["close"])
        buy_open = float(buy_df.iloc[0]["open"])
        gap_ret = buy_open / prev_close - 1.0 if prev_close else None

        feat = {
            "stock_code": row.stock_code,
            "pred_date": row.pred_date,
            "buy_datetime": row.buy_datetime,
            "sell_datetime": row.sell_datetime,
            "actual_return": row.actual_return,
            "profit_loss": row.profit_loss,
            "exit_reason": row.exit_reason,
            "volume_price_exhaustion": row.volume_price_exhaustion,
            "gap_ret": gap_ret,
        }
        if t2_df is not None and not t2_df.empty:
            for time_hm, col_name in [("09:35", "t2_ret_0935"), ("10:00", "t2_ret_1000"), ("14:55", "t2_ret_close")]:
                bar = t2_df[t2_df["time_hm"] == time_hm]
                feat[col_name] = float(bar.iloc[0]["close"]) / buy_open - 1.0 if not bar.empty and buy_open else None
        else:
            feat["t2_ret_0935"] = None
            feat["t2_ret_1000"] = None
            feat["t2_ret_close"] = None
        rows.append(feat)
    return pd.DataFrame(rows)


def evaluate_variant(df: pd.DataFrame, name: str, mode: str, condition) -> dict:
    probe = df.copy()
    probe["trigger"] = condition(probe)
    probe["alt_return"] = probe["actual_return"]
    if mode == "skip":
        probe.loc[probe["trigger"], "alt_return"] = 0.0
    elif mode == "exit_0935":
        probe.loc[probe["trigger"], "alt_return"] = probe.loc[probe["trigger"], "t2_ret_0935"]
    elif mode == "exit_1000":
        probe.loc[probe["trigger"], "alt_return"] = probe.loc[probe["trigger"], "t2_ret_1000"]
    elif mode == "exit_close":
        probe.loc[probe["trigger"], "alt_return"] = probe.loc[probe["trigger"], "t2_ret_close"]
    else:
        raise ValueError(mode)

    probe = probe[probe["alt_return"].notna()].copy()
    return {
        "variant": name,
        "mode": mode,
        "trade_count": len(probe),
        "triggered": int(probe["trigger"].sum()),
        "base_ret_sum": float(probe["actual_return"].sum()),
        "alt_ret_sum": float(probe["alt_return"].sum()),
        "ret_sum_delta": float(probe["alt_return"].sum() - probe["actual_return"].sum()),
        "base_bigloss5": int((probe["actual_return"] <= -0.05).sum()),
        "alt_bigloss5": int((probe["alt_return"] <= -0.05).sum()),
        "base_bigloss7": int((probe["actual_return"] <= -0.07).sum()),
        "alt_bigloss7": int((probe["alt_return"] <= -0.07).sum()),
        "base_bigwin10": int((probe["actual_return"] >= 0.10).sum()),
        "alt_bigwin10": int((probe["alt_return"] >= 0.10).sum()),
        "trigger_bigloss5": int(((probe["actual_return"] <= -0.05) & probe["trigger"]).sum()),
        "trigger_bigwin10": int(((probe["actual_return"] >= 0.10) & probe["trigger"]).sum()),
        "avg_ret_triggered": float(probe.loc[probe["trigger"], "actual_return"].mean()) if int(probe["trigger"].sum()) else None,
    }


def run_for_threshold(features: pd.DataFrame, label: str, selector) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = features[selector(features)].copy()
    rows: List[dict] = []
    variants = [
        ("strict_gap_up_p2", "skip", lambda d: d["gap_ret"] >= 0.02),
        ("strict_gap_up_p3", "skip", lambda d: d["gap_ret"] >= 0.03),
        ("strict_t2_0935_m3", "exit_0935", lambda d: d["t2_ret_0935"] <= -0.03),
        ("strict_t2_1000_m1", "exit_1000", lambda d: d["t2_ret_1000"] <= -0.01),
        ("strict_t2_close_p1", "exit_close", lambda d: d["t2_ret_close"] <= 0.01),
    ]
    for name, mode, cond in variants:
        row = evaluate_variant(subset, f"{label}_{name}", mode, cond)
        row["basket"] = label
        rows.append(row)
    return pd.DataFrame(rows), subset


def main() -> None:
    trades = load_trades_with_vpe()
    required_pairs = build_required_pairs(trades)
    minute_df = load_minute_rows(required_pairs)
    features = derive_features(trades, minute_df)

    q70 = features["volume_price_exhaustion"].quantile(0.7)
    q80 = features["volume_price_exhaustion"].quantile(0.8)
    q85 = features["volume_price_exhaustion"].quantile(0.85)

    frames = []
    subsets: List[pd.DataFrame] = []
    for label, thr in [("vpe_q70", q70), ("vpe_q80", q80), ("vpe_q85", q85)]:
        summary, subset = run_for_threshold(features, label, lambda d, t=thr: d["volume_price_exhaustion"] >= t)
        summary["threshold"] = thr
        frames.append(summary)
        subset = subset.copy()
        subset["basket"] = label
        subsets.append(subset)

    summary = pd.concat(frames, ignore_index=True)
    detail = pd.concat(subsets, ignore_index=True)
    summary_path = RESULTS_DIR / "vpe_strict_mode_probe_summary.csv"
    detail_path = RESULTS_DIR / "vpe_strict_mode_probe_features.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nsummary -> {summary_path}")
    print(f"features -> {detail_path}")


if __name__ == "__main__":
    main()
