#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS = ROOT / "results" / "walkforward_metrics_20260430_142636.json"
DEFAULT_FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
DEFAULT_BARS = ROOT / "cache" / "stock_features_20230601_20260428.parquet"
DEFAULT_CALENDAR = ROOT.parent / "local_data" / "calendar" / "trading_calendar.parquet"
OUTPUT_CSV = ROOT / "results" / "first_breakdown_risk_prototype_scan.csv"
OUTPUT_TRADES_CSV = ROOT / "results" / "first_breakdown_risk_flagged_trades.csv"


def calc_trade_mdd(returns_pct: Iterable[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns_pct:
        equity *= 1.0 + float(r) / 100.0
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)
    return max_dd * 100.0


def load_trades(metrics_path: Path, calendar_path: Path) -> pd.DataFrame:
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rec = data[0]
    trades = rec["all_results"]["1"]["trades"]

    cal = pd.read_parquet(calendar_path)
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()
    trading_days = sorted(cal["date"].drop_duplicates().tolist())
    idx_map = {d: i for i, d in enumerate(trading_days)}

    rows = []
    for t in trades:
        buy_date = pd.to_datetime(t.get("buy_date") or t.get("buy_datetime")).normalize()
        sell_raw = t.get("sell_date") or t.get("sell_datetime")
        sell_date = pd.to_datetime(sell_raw).normalize() if sell_raw else pd.NaT
        pred_date = trading_days[idx_map[buy_date] - 1] if buy_date in idx_map and idx_map[buy_date] > 0 else pd.NaT
        ret_pct = float(t.get("net_return", t.get("actual_return", t.get("profit_loss_pct", 0.0)))) * 100.0
        rows.append(
            {
                "stock_code": t["stock_code"],
                "prediction_date": pred_date,
                "buy_date": buy_date,
                "sell_date": sell_date,
                "return_pct": ret_pct,
                "exit_reason": t.get("exit_reason", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("buy_date").reset_index(drop=True)


def load_prediction_day_features(factors_path: Path, bars_path: Path) -> pd.DataFrame:
    factor_cols = [
        "date",
        "stock_code",
        "max_rally_10d",
        "sector_momentum_accel_rank",
        "return_5d_sector_mean_pct",
    ]
    bar_cols = ["date", "stock_code", "open", "high", "low", "close", "pct_chg"]
    factors = pd.read_parquet(factors_path, columns=factor_cols)
    bars = pd.read_parquet(bars_path, columns=bar_cols)
    for df in (factors, bars):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    merged = bars.merge(factors, on=["date", "stock_code"], how="left")
    return merged


def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) 前期涨幅: 已强拉升的票更怕首阴后续杀
    rally = pd.to_numeric(out["max_rally_10d"], errors="coerce").fillna(0.0)
    out["risk_recent_surge"] = np.clip(rally, 0.0, 0.35) / 0.35

    # 2) 预测日大阴/转弱
    pct = pd.to_numeric(out["pct_chg"], errors="coerce").fillna(0.0) / 100.0
    out["risk_breakdown"] = np.clip((-pct) - 0.03, 0.0, 0.12) / 0.12

    # 3) 收盘贴近低点 = 尾盘承接弱
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    weak_close = 1.0 - (close - low) / ((high - low) + 1e-9)
    out["risk_weak_close"] = weak_close.clip(0.0, 1.0).fillna(0.0)

    # 4) 行业动量加速度弱
    sector_rank = pd.to_numeric(out["sector_momentum_accel_rank"], errors="coerce").fillna(0.5)
    out["risk_sector_weak"] = (1.0 - sector_rank).clip(0.0, 1.0)

    # 5) 行业近5日收益弱
    sector_ret5 = pd.to_numeric(out["return_5d_sector_mean_pct"], errors="coerce").fillna(0.5)
    out["risk_sector_ret5_weak"] = (1.0 - sector_ret5).clip(0.0, 1.0)

    out["risk_score"] = (
        0.30 * out["risk_recent_surge"]
        + 0.25 * out["risk_breakdown"]
        + 0.20 * out["risk_weak_close"]
        + 0.15 * out["risk_sector_weak"]
        + 0.10 * out["risk_sector_ret5_weak"]
    )
    return out


def evaluate_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    base = df.sort_values("buy_date").reset_index(drop=True).copy()
    baseline = {
        "strategy": "baseline",
        "threshold": np.nan,
        "flagged_trades": 0,
        "flagged_ratio": 0.0,
        "bigloss5_flagged": 0,
        "bigloss7_flagged": 0,
        "stoploss_flagged": 0,
        "remain_bigloss5": int((base["return_pct"] <= -5).sum()),
        "remain_bigloss7": int((base["return_pct"] <= -7).sum()),
        "remain_stoploss": int(base["exit_reason"].str.contains("累计收益低于", na=False).sum()),
        "mean_return_pct": float(base["return_pct"].mean()),
        "cum_return_pct": float((np.prod(1.0 + base["return_pct"] / 100.0) - 1.0) * 100.0),
        "trade_mdd_pct": float(calc_trade_mdd(base["return_pct"])),
    }
    rows = [baseline]

    valid = base["risk_score"].dropna()
    quantiles = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for q in quantiles:
        thr = float(valid.quantile(q))
        tmp = base.copy()
        tmp["flag"] = tmp["risk_score"] >= thr
        kept_returns = np.where(tmp["flag"], 0.0, tmp["return_pct"])
        rows.append(
            {
                "strategy": f"risk_score_ge_q{int(q*100)}_skip",
                "threshold": thr,
                "flagged_trades": int(tmp["flag"].sum()),
                "flagged_ratio": float(tmp["flag"].mean()),
                "bigloss5_flagged": int(((tmp["return_pct"] <= -5) & tmp["flag"]).sum()),
                "bigloss7_flagged": int(((tmp["return_pct"] <= -7) & tmp["flag"]).sum()),
                "stoploss_flagged": int((tmp["exit_reason"].str.contains("累计收益低于", na=False) & tmp["flag"]).sum()),
                "remain_bigloss5": int(((tmp["return_pct"] <= -5) & ~tmp["flag"]).sum()),
                "remain_bigloss7": int(((tmp["return_pct"] <= -7) & ~tmp["flag"]).sum()),
                "remain_stoploss": int((tmp["exit_reason"].str.contains("累计收益低于", na=False) & ~tmp["flag"]).sum()),
                "mean_return_pct": float(np.mean(kept_returns)),
                "cum_return_pct": float((np.prod(1.0 + kept_returns / 100.0) - 1.0) * 100.0),
                "trade_mdd_pct": float(calc_trade_mdd(kept_returns)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTORS)
    parser.add_argument("--bars", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    args = parser.parse_args()

    trades = load_trades(args.metrics, args.calendar)
    pred_features = load_prediction_day_features(args.factors, args.bars)
    merged = trades.merge(
        pred_features,
        left_on=["prediction_date", "stock_code"],
        right_on=["date", "stock_code"],
        how="left",
    )
    merged = build_risk_features(merged)

    scan = evaluate_thresholds(merged)
    scan.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    best_thr = float(scan.loc[scan["strategy"] == "risk_score_ge_q85_skip", "threshold"].iloc[0])
    flagged = merged[merged["risk_score"] >= best_thr].copy()
    flagged = flagged[
        [
            "stock_code",
            "prediction_date",
            "buy_date",
            "sell_date",
            "return_pct",
            "exit_reason",
            "risk_score",
            "risk_recent_surge",
            "risk_breakdown",
            "risk_weak_close",
            "risk_sector_weak",
            "risk_sector_ret5_weak",
            "pct_chg",
            "max_rally_10d",
            "sector_momentum_accel_rank",
            "return_5d_sector_mean_pct",
        ]
    ].sort_values(["return_pct", "buy_date"])
    flagged.to_csv(OUTPUT_TRADES_CSV, index=False, encoding="utf-8-sig")

    print("saved:", OUTPUT_CSV)
    print("saved:", OUTPUT_TRADES_CSV)
    print(scan.to_string(index=False))


if __name__ == "__main__":
    main()
