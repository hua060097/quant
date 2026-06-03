import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
METRICS = RESULTS / "walkforward_metrics_20260520_210454.json"
MINUTE_ROOT = BASE.parent / "local_data" / "minute"
INITIAL_CAPITAL = 1_000_000.0


def load_trades():
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = []
    for t in data[0]["all_results"]["1"]["trades"]:
        rows.append(
            {
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
    return pd.DataFrame(rows).sort_values(["buy_datetime", "stock_code"]).reset_index(drop=True)


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


def same_day_after_buy(bars, buy_dt):
    if bars.empty:
        return bars
    return bars[(bars["datetime"].dt.date == buy_dt.date()) & (bars["datetime"] >= buy_dt)].copy()


def prev_day_close(bars, buy_dt):
    prev = bars[bars["datetime"].dt.date < buy_dt.date()]
    if prev.empty:
        return np.nan
    return float(prev.iloc[-1]["close"])


def price_at_or_after(day_bars, target_dt, column="close"):
    x = day_bars[day_bars["datetime"] >= target_dt]
    if x.empty:
        return np.nan, pd.NaT
    r = x.iloc[0]
    return float(r[column]), r["datetime"]


def first_pullback_price(day_bars, buy_price, max_minutes, pullback):
    end_dt = day_bars["datetime"].iloc[0] + pd.Timedelta(minutes=max_minutes)
    window = day_bars[day_bars["datetime"] <= end_dt]
    hit = window[window["low"] <= buy_price * (1 - pullback)]
    if hit.empty:
        return np.nan, pd.NaT
    r = hit.iloc[0]
    return min(float(r["close"]), buy_price * (1 - pullback)), r["datetime"]


def first_strength_confirm_price(day_bars, buy_price, open_price, max_gain_from_buy):
    """First 5-minute close above open while not already too extended from buy price."""
    if day_bars.empty or not np.isfinite(open_price):
        return np.nan, pd.NaT
    window = day_bars[day_bars["close"] > open_price]
    window = window[window["close"] <= buy_price * (1 + max_gain_from_buy)]
    if window.empty:
        return np.nan, pd.NaT
    r = window.iloc[0]
    return float(r["close"]), r["datetime"]


def strength_confirm_after(day_bars, buy_price, open_price, max_gain_from_buy, start_minutes):
    if day_bars.empty:
        return np.nan, pd.NaT
    start_dt = day_bars["datetime"].iloc[0] + pd.Timedelta(minutes=start_minutes)
    return first_strength_confirm_price(day_bars[day_bars["datetime"] >= start_dt], buy_price, open_price, max_gain_from_buy)


def tail_add_price_if_not_broken(day_bars, open_price, max_tail_drop_from_open, require_never_above_open=False):
    if day_bars.empty or not np.isfinite(open_price):
        return np.nan, pd.NaT, "missing"
    if require_never_above_open and float(day_bars["close"].max()) > open_price:
        return np.nan, pd.NaT, "stood_above_open"
    r = day_bars.iloc[-1]
    tail_close = float(r["close"])
    if tail_close >= open_price * (1 - max_tail_drop_from_open):
        return tail_close, r["datetime"], "tail_add"
    return np.nan, pd.NaT, "tail_drop_breached"


def ret_from_entry(entry_price, sell_price):
    if not np.isfinite(entry_price) or entry_price <= 0:
        return np.nan
    return sell_price / entry_price - 1.0


def max_drawdown_from_pnl(pnls, initial_capital=INITIAL_CAPITAL):
    s = pd.Series(pnls).fillna(0)
    equity = initial_capital + s.cumsum()
    peak = equity.cummax()
    dd = (equity / peak.replace(0, np.nan) - 1.0).fillna(0)
    return float(dd.min())


def analyze_trade(row):
    bars = load_stock_around(row["stock_code"], row["buy_datetime"])
    day = same_day_after_buy(bars, row["buy_datetime"])
    rec = row.to_dict()
    rec["minute_available"] = not day.empty
    if day.empty:
        return rec

    buy_price = row["buy_price"]
    sell_price = row["sell_price"]
    capital = row["quantity"] * buy_price
    prev_close = prev_day_close(bars, row["buy_datetime"])
    rec["prev_close_5m"] = prev_close
    rec["entry_gap_vs_prev_close"] = buy_price / prev_close - 1.0 if np.isfinite(prev_close) and prev_close > 0 else np.nan
    rec["day_bar_count_after_buy"] = len(day)
    rec["entry_open_5m"] = float(day.iloc[0]["open"])
    rec["day_close_ret"] = float(day.iloc[-1]["close"] / buy_price - 1.0)
    rec["min_low_ret_30m"] = float(day[day["datetime"] <= row["buy_datetime"] + pd.Timedelta(minutes=30)]["low"].min() / buy_price - 1.0)
    rec["max_high_ret_30m"] = float(day[day["datetime"] <= row["buy_datetime"] + pd.Timedelta(minutes=30)]["high"].max() / buy_price - 1.0)
    rec["first_15m_close_ret"] = np.nan
    p5, _ = price_at_or_after(day, row["buy_datetime"] + pd.Timedelta(minutes=5))
    rec["first_5m_close_ret"] = p5 / buy_price - 1.0 if np.isfinite(p5) else np.nan
    p15, _ = price_at_or_after(day, row["buy_datetime"] + pd.Timedelta(minutes=15))
    if np.isfinite(p15):
        rec["first_15m_close_ret"] = p15 / buy_price - 1.0
    p15_open_ref, _ = price_at_or_after(day, row["buy_datetime"] + pd.Timedelta(minutes=15))
    rec["first_15m_close_vs_open"] = p15_open_ref / rec["entry_open_5m"] - 1.0 if np.isfinite(p15_open_ref) and rec["entry_open_5m"] > 0 else np.nan
    first_15m = day[day["datetime"] <= row["buy_datetime"] + pd.Timedelta(minutes=15)]
    rec["min_low_ret_15m"] = float(first_15m["low"].min() / buy_price - 1.0) if not first_15m.empty else np.nan
    first_5m_high = float(day.iloc[0]["high"]) if len(day) else np.nan
    p10, _ = price_at_or_after(day, row["buy_datetime"] + pd.Timedelta(minutes=10))
    rec["first_5m_high_ret"] = first_5m_high / buy_price - 1.0 if np.isfinite(first_5m_high) else np.nan
    rec["first_10m_close_ret"] = p10 / buy_price - 1.0 if np.isfinite(p10) else np.nan
    no_fast_up_5m = not (np.isfinite(rec["first_5m_high_ret"]) and rec["first_5m_high_ret"] >= 0.008)
    no_fast_up_10m = not (np.isfinite(rec["first_10m_close_ret"]) and rec["first_10m_close_ret"] >= 0.003)
    is_entry_gap_down = np.isfinite(rec["entry_gap_vs_prev_close"]) and rec["entry_gap_vs_prev_close"] < 0
    weak_open_no_fast_up = bool(is_entry_gap_down and no_fast_up_5m and no_fast_up_10m)
    rec["weak_open_no_fast_up"] = weak_open_no_fast_up

    weak_open_masks = {
        "gap_down": bool(np.isfinite(rec["entry_gap_vs_prev_close"]) and rec["entry_gap_vs_prev_close"] < 0),
        "gap_down_0p5": bool(np.isfinite(rec["entry_gap_vs_prev_close"]) and rec["entry_gap_vs_prev_close"] <= -0.005),
        "gap_down_no_fastup": weak_open_no_fast_up,
        "first5_close_below_buy": bool(np.isfinite(rec["first_5m_close_ret"]) and rec["first_5m_close_ret"] < 0),
        "first15_close_below_buy": bool(np.isfinite(rec["first_15m_close_ret"]) and rec["first_15m_close_ret"] < 0),
        "first15_below_open": bool(np.isfinite(rec["first_15m_close_vs_open"]) and rec["first_15m_close_vs_open"] < 0),
        "first15_down_0p5": bool(np.isfinite(rec["first_15m_close_ret"]) and rec["first_15m_close_ret"] <= -0.005),
        "first15_down_1p0": bool(np.isfinite(rec["first_15m_close_ret"]) and rec["first_15m_close_ret"] <= -0.01),
        "low15_down_1p0": bool(np.isfinite(rec["min_low_ret_15m"]) and rec["min_low_ret_15m"] <= -0.01),
        "low30_down_1p5": bool(np.isfinite(rec["min_low_ret_30m"]) and rec["min_low_ret_30m"] <= -0.015),
        "low30_down_2p0": bool(np.isfinite(rec["min_low_ret_30m"]) and rec["min_low_ret_30m"] <= -0.02),
        "gap_down_or_first15_below_open": bool(
            (np.isfinite(rec["entry_gap_vs_prev_close"]) and rec["entry_gap_vs_prev_close"] < 0)
            or (np.isfinite(rec["first_15m_close_vs_open"]) and rec["first_15m_close_vs_open"] < 0)
        ),
        "gap_down_and_first15_below_open": bool(
            (np.isfinite(rec["entry_gap_vs_prev_close"]) and rec["entry_gap_vs_prev_close"] < 0)
            and (np.isfinite(rec["first_15m_close_vs_open"]) and rec["first_15m_close_vs_open"] < 0)
        ),
    }
    for weak_name, weak_value in weak_open_masks.items():
        rec[f"weak_open_{weak_name}"] = weak_value

    entry_prices = {"baseline": buy_price}
    entry_weights = {"baseline": 1.0}
    for minutes in [5, 10, 15, 30]:
        p, ts = price_at_or_after(day, row["buy_datetime"] + pd.Timedelta(minutes=minutes))
        entry_prices[f"delay_{minutes}m"] = p
        entry_weights[f"delay_{minutes}m"] = 1.0
        rec[f"delay_{minutes}m_entry_time"] = ts
        if np.isfinite(p):
            entry_prices[f"split50_now_delay_{minutes}m"] = 0.5 * buy_price + 0.5 * p
            entry_weights[f"split50_now_delay_{minutes}m"] = 1.0

    for pullback in [0.01, 0.015, 0.02]:
        p, ts = first_pullback_price(day, buy_price, 30, pullback)
        rec[f"pullback_{pullback:.3f}_30m_hit"] = pd.notna(ts)
        rec[f"pullback_{pullback:.3f}_30m_time"] = ts
        if np.isfinite(p):
            # Correct: 50% at open + 50% at pullback → full position
            entry_prices[f"FIX_pb{pullback:.3f}_open50_pb50"] = 0.5 * buy_price + 0.5 * p
            entry_weights[f"FIX_pb{pullback:.3f}_open50_pb50"] = 1.0
            # Legacy buggy versions (kept for comparison)
            entry_prices[f"split50_now_pb{pullback:.3f}_30m_else_half"] = 0.5 * buy_price + 0.5 * p
            entry_weights[f"split50_now_pb{pullback:.3f}_30m_else_half"] = 1.0
            entry_prices[f"pb{pullback:.3f}_30m_only_half"] = p
            entry_weights[f"pb{pullback:.3f}_30m_only_half"] = 0.5
        else:
            # Correct: no pullback → still 50% at open
            entry_prices[f"FIX_pb{pullback:.3f}_open50_pb50"] = buy_price
            entry_weights[f"FIX_pb{pullback:.3f}_open50_pb50"] = 0.5
            # Legacy
            entry_prices[f"split50_now_pb{pullback:.3f}_30m_else_half"] = buy_price
            entry_weights[f"split50_now_pb{pullback:.3f}_30m_else_half"] = 0.5
            entry_prices[f"pb{pullback:.3f}_30m_only_half"] = np.nan
            entry_weights[f"pb{pullback:.3f}_30m_only_half"] = 0.0

    # Conditional first-stage delay: if entry starts below previous close and fails to show fast-up,
    # delay the first 50%; otherwise use the normal 50-now + pullback-add framework.
    for delay_minutes in [5, 10, 15]:
        delayed_first, _ = price_at_or_after(day, row["buy_datetime"] + pd.Timedelta(minutes=delay_minutes))
        for pullback in [0.015, 0.02]:
            second, _ = first_pullback_price(day, buy_price, 30, pullback)
            if weak_open_no_fast_up and np.isfinite(delayed_first):
                if np.isfinite(second):
                    entry_prices[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = (
                        0.5 * delayed_first + 0.5 * second
                    )
                    entry_weights[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = 1.0
                else:
                    entry_prices[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = delayed_first
                    entry_weights[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = 0.5
            else:
                if np.isfinite(second):
                    entry_prices[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = (
                        0.5 * buy_price + 0.5 * second
                    )
                    entry_weights[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = 1.0
                else:
                    entry_prices[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = buy_price
                    entry_weights[f"cond_weak_delay{delay_minutes}m_50_pb{pullback:.3f}"] = 0.5

    # Stop-add version: always enter first 50%, but if the first 15m is weak, do not add the second half.
    weak_after_entry = bool(
        (np.isfinite(rec["first_15m_close_ret"]) and rec["first_15m_close_ret"] <= -0.015)
        or (np.isfinite(rec["min_low_ret_30m"]) and rec["min_low_ret_30m"] <= -0.02)
    )
    rec["weak_after_entry_stop_add"] = weak_after_entry
    for pullback in [0.015, 0.02]:
        second, _ = first_pullback_price(day, buy_price, 30, pullback)
        if weak_after_entry:
            entry_prices[f"split50_stop_add_if_weak_pb{pullback:.3f}"] = buy_price
            entry_weights[f"split50_stop_add_if_weak_pb{pullback:.3f}"] = 0.5
        elif np.isfinite(second):
            entry_prices[f"split50_stop_add_if_weak_pb{pullback:.3f}"] = 0.5 * buy_price + 0.5 * second
            entry_weights[f"split50_stop_add_if_weak_pb{pullback:.3f}"] = 1.0
        else:
            entry_prices[f"split50_stop_add_if_weak_pb{pullback:.3f}"] = buy_price
            entry_weights[f"split50_stop_add_if_weak_pb{pullback:.3f}"] = 0.5

    # User-proposed confirmation entry:
    # buy 50% first, then every 5-minute bar add the remaining 50% only if price
    # is above the open and not more than X% above the original buy price.
    # If no confirmation happens, add near close only when the tail drop from
    # open has not breached Y%; otherwise keep only the initial half.
    open_price = rec["entry_open_5m"]
    for confirm_cap in [0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
        confirm_price, confirm_ts = first_strength_confirm_price(day, buy_price, open_price, confirm_cap)
        for tail_stop in [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
            for strict_tail in [False, True]:
                suffix = "strict_tail" if strict_tail else "tail_any"
                name = f"split50_confirm{confirm_cap:.3f}_tailstop{tail_stop:.3f}_{suffix}"
                if np.isfinite(confirm_price):
                    entry_prices[name] = 0.5 * buy_price + 0.5 * confirm_price
                    entry_weights[name] = 1.0
                    rec[f"{name}_second_time"] = confirm_ts
                    rec[f"{name}_second_reason"] = "confirm"
                else:
                    tail_price, tail_ts, reason = tail_add_price_if_not_broken(
                        day, open_price, tail_stop, require_never_above_open=strict_tail
                    )
                    if np.isfinite(tail_price):
                        entry_prices[name] = 0.5 * buy_price + 0.5 * tail_price
                        entry_weights[name] = 1.0
                        rec[f"{name}_second_time"] = tail_ts
                        rec[f"{name}_second_reason"] = reason
                    else:
                        entry_prices[name] = buy_price
                        entry_weights[name] = 0.5
                        rec[f"{name}_second_time"] = tail_ts
                        rec[f"{name}_second_reason"] = reason

    # Conditional version: apply the confirmation/tail split only to weak-open
    # trades; keep the baseline full entry for all other trades.
    for weak_name, weak_value in weak_open_masks.items():
        for confirm_cap in [0.01, 0.015, 0.02, 0.025, 0.03]:
            confirm_price, confirm_ts = strength_confirm_after(day, buy_price, open_price, confirm_cap, start_minutes=5)
            for tail_stop in [0.02, 0.025, 0.03, 0.04, 0.05]:
                name = f"condweak_{weak_name}_confirm{confirm_cap:.3f}_tailstop{tail_stop:.3f}_strict"
                if not weak_value:
                    entry_prices[name] = buy_price
                    entry_weights[name] = 1.0
                    rec[f"{name}_second_reason"] = "not_weak_full"
                elif np.isfinite(confirm_price):
                    entry_prices[name] = 0.5 * buy_price + 0.5 * confirm_price
                    entry_weights[name] = 1.0
                    rec[f"{name}_second_time"] = confirm_ts
                    rec[f"{name}_second_reason"] = "confirm"
                else:
                    tail_price, tail_ts, reason = tail_add_price_if_not_broken(
                        day, open_price, tail_stop, require_never_above_open=True
                    )
                    if np.isfinite(tail_price):
                        entry_prices[name] = 0.5 * buy_price + 0.5 * tail_price
                        entry_weights[name] = 1.0
                        rec[f"{name}_second_time"] = tail_ts
                        rec[f"{name}_second_reason"] = reason
                    else:
                        entry_prices[name] = buy_price
                        entry_weights[name] = 0.5
                        rec[f"{name}_second_time"] = tail_ts
                        rec[f"{name}_second_reason"] = reason
    for name, entry in entry_prices.items():
        ret = ret_from_entry(entry, sell_price)
        weight = float(entry_weights.get(name, 1.0))
        rec[f"{name}_entry_price"] = entry
        rec[f"{name}_position_weight"] = weight
        rec[f"{name}_ret"] = ret
        rec[f"{name}_pnl"] = capital * weight * ret if np.isfinite(ret) else 0.0
        rec[f"{name}_delta_pnl"] = rec[f"{name}_pnl"] - row["profit_loss"] if np.isfinite(ret) else np.nan
    return rec


def summarize(out):
    rows = []
    strategies = sorted(
        c[:-4]
        for c in out.columns
        if c.endswith("_pnl") and not c.endswith("_delta_pnl") and c not in {"profit_loss"}
    )
    # Add baseline row
    rows.append({
        "strategy": "BASELINE (full entry)",
        "valid_trades": len(out),
        "total_pnl": float(out["profit_loss"].sum()),
        "delta_pnl": 0.0,
        "max_drawdown_pct": max_drawdown_from_pnl(out["profit_loss"]) * 100,
        "avg_ret": float(out["actual_return"].mean()),
        "avg_position_weight": 1.0,
        "half_or_less_count": 0,
        "loss_count": int((out["actual_return"] <= 0).sum()),
        "big_loss_count_le_-5pct": int((out["actual_return"] <= -0.05).sum()),
        "big_win_count_ge_10pct": int((out["actual_return"] >= 0.10).sum()),
    })
    for s in strategies:
        pnl_col = f"{s}_pnl"
        ret_col = f"{s}_ret"
        delta_col = f"{s}_delta_pnl"
        valid = out[ret_col].notna() if ret_col in out.columns else pd.Series(False, index=out.index)
        rows.append(
            {
                "strategy": s,
                "valid_trades": int(valid.sum()),
                "total_pnl": float(out.loc[valid, pnl_col].sum()),
                "delta_pnl": float(out.loc[valid, delta_col].sum()),
                "max_drawdown_pct": max_drawdown_from_pnl(out.loc[valid, pnl_col]) * 100,
                "avg_ret": float(out.loc[valid, ret_col].mean()),
                "avg_position_weight": float(out.loc[valid, f"{s}_position_weight"].mean())
                if f"{s}_position_weight" in out.columns
                else np.nan,
                "half_or_less_count": int((out.loc[valid, f"{s}_position_weight"] <= 0.5).sum())
                if f"{s}_position_weight" in out.columns
                else 0,
                "loss_count": int((out.loc[valid, ret_col] <= 0).sum()),
                "big_loss_count_le_-5pct": int((out.loc[valid, ret_col] <= -0.05).sum()),
                "big_win_count_ge_10pct": int((out.loc[valid, ret_col] >= 0.10).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_pnl", ascending=False)


def cohort_summary(out, cohort_mask, label):
    base = out[cohort_mask].copy()
    if base.empty:
        return pd.DataFrame()
    s = summarize(base)
    s.insert(0, "cohort", label)
    return s


def main():
    trades = load_trades()
    rows = [analyze_trade(row) for _, row in trades.iterrows()]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "intraday_entry_split_5min_trade_detail_185.csv", index=False, encoding="utf-8-sig")

    summary = summarize(out)
    cohorts = [
        cohort_summary(out, out["actual_return"] <= -0.05, "baseline_big_loss"),
        cohort_summary(out, out["actual_return"] > 0.10, "baseline_big_win"),
        cohort_summary(out, out["first_15m_close_ret"] <= -0.01, "first15m_down_1pct"),
        cohort_summary(out, out["min_low_ret_30m"] <= -0.02, "first30m_low_down_2pct"),
    ]
    cohort = pd.concat([x for x in cohorts if not x.empty], ignore_index=True)
    summary.to_csv(RESULTS / "intraday_entry_split_5min_strategy_summary_185.csv", index=False, encoding="utf-8-sig")
    cohort.to_csv(RESULTS / "intraday_entry_split_5min_cohort_summary_185.csv", index=False, encoding="utf-8-sig")

    path_stats = pd.DataFrame(
        [
            {
                "n": len(out),
                "minute_available": int(out["minute_available"].sum()),
                "baseline_big_loss": int((out["actual_return"] <= -0.05).sum()),
                "baseline_big_win": int((out["actual_return"] >= 0.10).sum()),
                "first15m_down_1pct": int((out["first_15m_close_ret"] <= -0.01).sum()),
                "first30m_low_down_2pct": int((out["min_low_ret_30m"] <= -0.02).sum()),
                "big_loss_first15m_down_1pct": int(((out["actual_return"] <= -0.05) & (out["first_15m_close_ret"] <= -0.01)).sum()),
                "big_win_first15m_down_1pct": int(((out["actual_return"] >= 0.10) & (out["first_15m_close_ret"] <= -0.01)).sum()),
                "big_loss_first30m_low_down_2pct": int(((out["actual_return"] <= -0.05) & (out["min_low_ret_30m"] <= -0.02)).sum()),
                "big_win_first30m_low_down_2pct": int(((out["actual_return"] >= 0.10) & (out["min_low_ret_30m"] <= -0.02)).sum()),
            }
        ]
    )
    path_stats.to_csv(RESULTS / "intraday_entry_path_stats_185.csv", index=False, encoding="utf-8-sig")

    print("== path stats ==")
    print(path_stats.to_string(index=False))
    print("\n== strategy summary top ==")
    print(summary.head(20).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n== corrected pullback strategies (FIX) ==")
    fix_pb = summary[summary["strategy"].str.startswith("FIX_pb")].copy()
    print(fix_pb.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n== user confirm-tail strategy top ==")
    user_confirm = summary[summary["strategy"].str.startswith("split50_confirm")].copy()
    print(user_confirm.head(20).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n== cohort summary top by cohort ==")
    print(cohort.sort_values(["cohort", "delta_pnl"], ascending=[True, False]).groupby("cohort").head(8).to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
