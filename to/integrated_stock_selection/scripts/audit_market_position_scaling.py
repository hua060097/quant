"""Audit market-factor position scaling on 185 actual trades without changing trade selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_correlation_rule_candidates import BASE, FACTORS, RESULTS, attach_previous_snapshot


METRICS = RESULTS / "walkforward_metrics_20260520_210454.json"
FACTORS_TO_TEST = [
    "market_trend_adaptive_beta",
    "market_60d_new_high_ratio",
    "ma5_strength_breadth_v3",
    "market_vols_adaptive_iv",
]


def load_trades() -> pd.DataFrame:
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    result = data[0]["all_results"]["1"]
    trades = pd.DataFrame(result["trades"])
    trades["buy_date"] = pd.to_datetime(trades["buy_datetime"].str[:10])
    trades["sell_datetime"] = pd.to_datetime(trades["sell_datetime"])
    trades["actual_return"] = pd.to_numeric(trades["actual_return"], errors="coerce")
    trades["profit_loss"] = pd.to_numeric(trades["profit_loss"], errors="coerce")
    return trades, float(result["initial_capital"])


def load_factors_and_historical_percentiles() -> pd.DataFrame:
    factors = pd.read_parquet(
        FACTORS,
        columns=["date", "stock_code"] + FACTORS_TO_TEST,
    )
    factors["date"] = pd.to_datetime(factors["date"])
    daily = factors.groupby("date", as_index=False)[FACTORS_TO_TEST].median(numeric_only=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    for factor in FACTORS_TO_TEST:
        values = pd.to_numeric(daily[factor], errors="coerce").to_numpy(dtype=float)
        pct = np.full(len(values), np.nan)
        for idx, value in enumerate(values):
            history = values[:idx + 1]
            history = history[np.isfinite(history)]
            if np.isfinite(value) and len(history) >= 60:
                pct[idx] = float(np.mean(history <= value))
        daily[f"{factor}_hist_pct"] = pct
    return daily


def attach_market_snapshot(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    dates = daily["date"].to_numpy(dtype="datetime64[ns]")
    rows = []
    value_cols = [col for col in daily.columns if col != "date"]
    for _, trade in trades.iterrows():
        row = trade.to_dict()
        pos = np.searchsorted(dates, np.datetime64(trade["buy_date"]), side="left") - 1
        if pos >= 0:
            snapshot = daily.iloc[int(pos)]
            row["factor_date"] = snapshot["date"]
            for col in value_cols:
                row[col] = snapshot[col]
        rows.append(row)
    return pd.DataFrame(rows)


def max_drawdown_from_closed_trades(df: pd.DataFrame, initial_capital: float, scale: pd.Series) -> float:
    ordered = df.assign(scaled_pnl=df["profit_loss"] * scale).sort_values(
        ["sell_datetime", "stock_code"]
    )
    equity = initial_capital + ordered["scaled_pnl"].cumsum().to_numpy(dtype=float)
    equity = np.insert(equity, 0, initial_capital)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.maximum(peak, 1e-12)
    return float(drawdown.max())


def evaluate(df: pd.DataFrame, initial_capital: float, scale: pd.Series, name: str) -> dict:
    scale = pd.Series(scale, index=df.index).fillna(1.0).clip(0.0, 1.0)
    pnl = df["profit_loss"] * scale
    losses_saved = df.loc[df["actual_return"] <= -.04, "profit_loss"] * (1 - scale[df["actual_return"] <= -.04])
    gains_lost = df.loc[df["actual_return"] >= .08, "profit_loss"] * (1 - scale[df["actual_return"] >= .08])
    return {
        "rule": name,
        "total_pnl": float(pnl.sum()),
        "closed_trade_max_drawdown": max_drawdown_from_closed_trades(df, initial_capital, scale),
        "mean_scale": float(scale.mean()),
        "scaled_trade_count": int((scale < 1).sum()),
        "half_or_less_count": int((scale <= .5).sum()),
        "big_loss_saved_pnl": float(-losses_saved.sum()),
        "big_gain_lost_pnl": float(gains_lost.sum()),
    }


def segment_metrics(df: pd.DataFrame, scale: pd.Series, name: str) -> list[dict]:
    unique_dates = np.array(sorted(df["buy_date"].unique()))
    rows = []
    for idx, dates in enumerate(np.array_split(unique_dates, 4), start=1):
        mask = df["buy_date"].isin(dates)
        base = float(df.loc[mask, "profit_loss"].sum())
        scaled = float((df.loc[mask, "profit_loss"] * scale.loc[mask]).sum())
        rows.append({
            "rule": name,
            "segment": f"S{idx}",
            "start": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
            "base_pnl": base,
            "scaled_pnl": scaled,
            "pnl_change": scaled - base,
            "mean_scale": float(scale.loc[mask].mean()),
        })
    return rows


def main() -> None:
    trades, initial_capital = load_trades()
    daily = load_factors_and_historical_percentiles()
    df = attach_market_snapshot(trades, daily)
    base_scale = pd.Series(1.0, index=df.index)
    baseline = evaluate(df, initial_capital, base_scale, "baseline")
    rows = [baseline]
    segment_rows = segment_metrics(df, base_scale, "baseline")

    risk_factors = [
        "market_trend_adaptive_beta",
        "market_60d_new_high_ratio",
        "ma5_strength_breadth_v3",
    ]
    candidates = []
    for factor in risk_factors:
        pct = pd.to_numeric(df[f"{factor}_hist_pct"], errors="coerce")
        for threshold in (.70, .80, .90, .95):
            for reduced_scale in (.75, .50, .25):
                scale = pd.Series(1.0, index=df.index)
                scale[pct >= threshold] = reduced_scale
                candidates.append((f"{factor}:pct>={threshold:.2f}->scale{reduced_scale:.2f}", scale))

    iv_pct = pd.to_numeric(df["market_vols_adaptive_iv_hist_pct"], errors="coerce")
    for threshold in (.05, .10, .20, .30):
        for reduced_scale in (.75, .50, .25):
            scale = pd.Series(1.0, index=df.index)
            scale[iv_pct <= threshold] = reduced_scale
            candidates.append((f"market_vols_adaptive_iv:pct<={threshold:.2f}->scale{reduced_scale:.2f}", scale))

    risk_pct = pd.DataFrame({
        factor: pd.to_numeric(df[f"{factor}_hist_pct"], errors="coerce")
        for factor in risk_factors
    })
    for threshold in (.70, .80, .90):
        high_count = (risk_pct >= threshold).sum(axis=1)
        for required_count in (2, 3):
            for reduced_scale in (.75, .50, .25):
                scale = pd.Series(1.0, index=df.index)
                scale[high_count >= required_count] = reduced_scale
                candidates.append((
                    f"risk_count>={required_count}@pct{threshold:.2f}->scale{reduced_scale:.2f}",
                    scale,
                ))

    # Smooth two-level variants: moderate heat -> 75%, extreme heat -> 50%.
    for factor in risk_factors:
        pct = pd.to_numeric(df[f"{factor}_hist_pct"], errors="coerce")
        scale = pd.Series(1.0, index=df.index)
        scale[pct >= .80] = .75
        scale[pct >= .90] = .50
        candidates.append((f"{factor}:tier80=.75,tier90=.50", scale))

    for name, scale in candidates:
        row = evaluate(df, initial_capital, scale, name)
        row["pnl_retention"] = row["total_pnl"] / baseline["total_pnl"]
        row["drawdown_reduction"] = baseline["closed_trade_max_drawdown"] - row["closed_trade_max_drawdown"]
        row["dd_reduction_ratio"] = row["drawdown_reduction"] / baseline["closed_trade_max_drawdown"]
        rows.append(row)
        segment_rows.extend(segment_metrics(df, scale, name))

    result = pd.DataFrame(rows)
    result.loc[result["rule"].eq("baseline"), "pnl_retention"] = 1.0
    result.loc[result["rule"].eq("baseline"), "drawdown_reduction"] = 0.0
    result.loc[result["rule"].eq("baseline"), "dd_reduction_ratio"] = 0.0
    segments = pd.DataFrame(segment_rows)
    stability = segments[~segments["rule"].eq("baseline")].groupby("rule").agg(
        positive_segment_count=("pnl_change", lambda x: int((x > 0).sum())),
        negative_segment_count=("pnl_change", lambda x: int((x < 0).sum())),
        segment_pnl_change_mean=("pnl_change", "mean"),
        segment_pnl_change_min=("pnl_change", "min"),
        segment_pnl_change_max=("pnl_change", "max"),
    ).reset_index()
    result = result.merge(stability, on="rule", how="left")
    result["efficiency"] = result["dd_reduction_ratio"] / np.maximum(1 - result["pnl_retention"], 1e-9)
    result.to_csv(RESULTS / "market_position_scaling_audit.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(RESULTS / "market_position_scaling_segments.csv", index=False, encoding="utf-8-sig")
    df.to_csv(RESULTS / "market_position_scaling_trade_snapshots.csv", index=False, encoding="utf-8-sig")

    feasible = result[
        (~result["rule"].eq("baseline"))
        & (result["pnl_retention"] >= .90)
        & (result["drawdown_reduction"] > 0)
    ].sort_values(["drawdown_reduction", "pnl_retention"], ascending=[False, False])
    print("BASELINE", baseline)
    print("\nFEASIBLE pnl retention >=90%")
    print(feasible.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
