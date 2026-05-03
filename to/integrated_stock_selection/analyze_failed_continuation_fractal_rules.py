#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MINUTE_DIR = ROOT.parent / "local_data" / "minute"
BASELINE_METRICS = RESULTS_DIR / "walkforward_metrics_20260501_212757.json"  # 原因子93


@dataclass
class FractalRule:
    name: str
    peak_activation: float
    breakdown_type: str  # close_below_t1_low / low_below_t1_low
    weak_window: int
    weak_needed: int
    use_repair_failure: bool
    require_no_new_high: bool


RULES: List[FractalRule] = [
    FractalRule(
        name="peak4_break_t1low_repair_fail_3bars_2weak",
        peak_activation=0.04,
        breakdown_type="close_below_t1_low",
        weak_window=3,
        weak_needed=2,
        use_repair_failure=True,
        require_no_new_high=False,
    ),
    FractalRule(
        name="peak4_break_t1low_repair_fail_3bars_2weak_no_new_high",
        peak_activation=0.04,
        breakdown_type="close_below_t1_low",
        weak_window=3,
        weak_needed=2,
        use_repair_failure=True,
        require_no_new_high=True,
    ),
    FractalRule(
        name="peak4_last4_atleast3weak_no_new_high",
        peak_activation=0.04,
        breakdown_type="low_below_t1_low",
        weak_window=4,
        weak_needed=3,
        use_repair_failure=False,
        require_no_new_high=True,
    ),
    FractalRule(
        name="peak4_last4_atleast3weak",
        peak_activation=0.04,
        breakdown_type="low_below_t1_low",
        weak_window=4,
        weak_needed=3,
        use_repair_failure=False,
        require_no_new_high=False,
    ),
]


class MinuteLoader:
    def __init__(self, minute_dir: Path):
        self.minute_dir = minute_dir
        self.day_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    def get_day(self, stock_code: str, day: pd.Timestamp) -> pd.DataFrame:
        day_key = day.strftime("%Y-%m-%d")
        cache_key = (stock_code, day_key)
        if cache_key in self.day_cache:
            return self.day_cache[cache_key]
        month_file = self.minute_dir / day.strftime("%Y%m") / "minute_data.parquet"
        if not month_file.exists():
            self.day_cache[cache_key] = pd.DataFrame()
            return self.day_cache[cache_key]
        start_time = datetime.combine(day.date(), datetime.min.time())
        end_time = datetime.combine(day.date(), datetime.max.time())
        table = pq.read_table(
            month_file,
            filters=[
                ("stock_code", "=", stock_code),
                ("datetime", ">=", start_time),
                ("datetime", "<=", end_time),
            ],
            columns=["datetime", "stock_code", "open", "high", "low", "close", "volume"],
        )
        df = table.to_pandas()
        if not df.empty:
            df = df.sort_values("datetime").reset_index(drop=True)
        self.day_cache[cache_key] = df
        return df

    def find_next_trading_days(self, stock_code: str, anchor_day: pd.Timestamp, n: int) -> List[pd.Timestamp]:
        found: List[pd.Timestamp] = []
        cursor = anchor_day + pd.Timedelta(days=1)
        safety = 15
        while len(found) < n and safety > 0:
            df = self.get_day(stock_code, cursor)
            if not df.empty:
                found.append(cursor.normalize())
            cursor += pd.Timedelta(days=1)
            safety -= 1
        return found


def load_top1_trades(metrics_path: Path) -> pd.DataFrame:
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))[0]
    trades = pd.DataFrame(raw["detailed_trades"]["1"])
    trades["buy_datetime"] = pd.to_datetime(trades["buy_datetime"])
    trades["sell_datetime"] = pd.to_datetime(trades["sell_datetime"])
    trades["buy_date"] = trades["buy_datetime"].dt.normalize()
    trades["sell_date"] = trades["sell_datetime"].dt.normalize()
    return trades


def is_weak_bar(row) -> bool:
    op = float(row.open)
    cl = float(row.close)
    hi = float(row.high)
    lo = float(row.low)
    if cl < op:
        return True
    rng = max(hi - lo, 1e-9)
    close_loc = (cl - lo) / rng
    return close_loc <= 0.35


def simulate_fractal_rule(trade: pd.Series, rule: FractalRule, loader: MinuteLoader) -> Dict[str, object]:
    stock_code = trade["stock_code"]
    buy_dt = pd.Timestamp(trade["buy_datetime"])
    buy_date = pd.Timestamp(trade["buy_date"])
    sell_dt = pd.Timestamp(trade["sell_datetime"])
    buy_price = float(trade["buy_price"])

    buy_day_df = loader.get_day(stock_code, buy_date)
    if buy_day_df.empty:
        return {"triggered": False, "reason": "missing_buy_day"}
    t1_low = float(buy_day_df["low"].min())

    evaluation_days = [d for d in loader.find_next_trading_days(stock_code, buy_date, 3) if d <= sell_dt.normalize()]
    if not evaluation_days:
        return {"triggered": False, "reason": "no_eval_days", "t1_low": t1_low}

    peak_return = -999.0
    peak_frames: List[pd.DataFrame] = []
    buy_peak_df = buy_day_df[buy_day_df["datetime"] >= buy_dt].copy()
    if not buy_peak_df.empty:
        peak_frames.append(buy_peak_df)
    for eval_day in evaluation_days:
        day_df = loader.get_day(stock_code, eval_day)
        if not day_df.empty:
            peak_frames.append(day_df)
    if peak_frames:
        peak_all = pd.concat(peak_frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    else:
        return {"triggered": False, "reason": "no_peak_frames", "t1_low": t1_low}

    peak_rows = peak_all.to_dict("records")
    peak_idx = 0
    armed = False
    breakdown_started = False
    breakdown_reference_high: Optional[float] = None
    post_break_bars: List[dict] = []

    for eval_day in evaluation_days:
        day_df = loader.get_day(stock_code, eval_day)
        if day_df.empty:
            continue

        for row in day_df.itertuples(index=False):
            row_dt = pd.Timestamp(row.datetime)
            while peak_idx < len(peak_rows) and pd.Timestamp(peak_rows[peak_idx]["datetime"]) <= row_dt:
                peak_close = float(peak_rows[peak_idx]["close"])
                peak_return = max(peak_return, peak_close / buy_price - 1.0)
                peak_idx += 1
            if not armed and peak_return >= rule.peak_activation:
                armed = True
            if not armed:
                continue

            if rule.breakdown_type == "close_below_t1_low":
                broken = float(row.close) < t1_low
            elif rule.breakdown_type == "low_below_t1_low":
                broken = float(row.low) < t1_low
            else:
                raise ValueError(rule.breakdown_type)

            if not breakdown_started:
                if broken:
                    breakdown_started = True
                    breakdown_reference_high = float(row.high)
                    post_break_bars = [{
                        "dt": row_dt,
                        "weak": is_weak_bar(row),
                        "high": float(row.high),
                        "close": float(row.close),
                    }]
                continue

            post_break_bars.append({
                "dt": row_dt,
                "weak": is_weak_bar(row),
                "high": float(row.high),
                "close": float(row.close),
            })
            if len(post_break_bars) > rule.weak_window:
                post_break_bars.pop(0)

            if len(post_break_bars) < rule.weak_window:
                continue

            weak_count = sum(1 for x in post_break_bars if x["weak"])
            if weak_count < rule.weak_needed:
                continue

            if rule.use_repair_failure:
                assert breakdown_reference_high is not None
                repaired = any(x["high"] > breakdown_reference_high for x in post_break_bars)
                if repaired:
                    continue

            if rule.require_no_new_high:
                # after armed, if there is a fresh peak after breakdown, do not trigger
                current_peak_return = max((x["close"] / buy_price - 1.0) for x in post_break_bars)
                if current_peak_return >= peak_return - 1e-12:
                    continue

            exit_close = float(post_break_bars[-1]["close"])
            exit_return = exit_close / buy_price - 1.0
            return {
                "triggered": True,
                "trigger_dt": post_break_bars[-1]["dt"],
                "trigger_price": exit_close,
                "trigger_return": exit_return,
                "peak_return_at_trigger": peak_return,
                "t1_low": t1_low,
                "break_ref_high": breakdown_reference_high,
                "weak_count": weak_count,
            }

    return {"triggered": False, "t1_low": t1_low}


def main() -> None:
    baseline = load_top1_trades(BASELINE_METRICS)
    loader = MinuteLoader(MINUTE_DIR)

    per_trade_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    key_cases = {
        ("600722.SH", "2026-03-27"),
        ("688008.SH", "2026-01-26"),
        ("688472.SH", "2025-11-07"),
    }

    for rule in RULES:
        simulated = baseline.copy()
        simulated["candidate_triggered"] = False
        simulated["candidate_return"] = simulated["actual_return"].astype(float)
        simulated["candidate_profit_loss"] = simulated["profit_loss"].astype(float)
        simulated["candidate_reason"] = "baseline"

        for idx, trade in simulated.iterrows():
            res = simulate_fractal_rule(trade, rule, loader)
            per_trade_rows.append({
                "rule": rule.name,
                "stock_code": trade["stock_code"],
                "buy_date": pd.Timestamp(trade["buy_date"]).strftime("%Y-%m-%d"),
                "baseline_return": float(trade["actual_return"]),
                "baseline_exit_reason": trade["exit_reason"],
                "triggered": bool(res.get("triggered", False)),
                "trigger_dt": res.get("trigger_dt"),
                "trigger_return": res.get("trigger_return"),
                "peak_return_at_trigger": res.get("peak_return_at_trigger"),
                "weak_count": res.get("weak_count"),
                "t1_low": res.get("t1_low"),
                "break_ref_high": res.get("break_ref_high"),
            })

            if res.get("triggered", False):
                simulated.at[idx, "candidate_triggered"] = True
                simulated.at[idx, "candidate_return"] = float(res["trigger_return"])
                simulated.at[idx, "candidate_profit_loss"] = float(trade["quantity"]) * (
                    float(res["trigger_price"]) - float(trade["buy_price"])
                )
                simulated.at[idx, "candidate_reason"] = rule.name

        hard_stop_mask = simulated["exit_reason"].astype(str).str.contains("累计收益低于-5.00%")
        key_case_hit_mask = simulated.apply(
            lambda r: (r["stock_code"], pd.Timestamp(r["buy_date"]).strftime("%Y-%m-%d")) in key_cases,
            axis=1,
        )
        harmed_mask = (simulated["candidate_return"] - simulated["actual_return"]) < -0.02

        summary_rows.append({
            "rule": rule.name,
            "triggered_count": int(simulated["candidate_triggered"].sum()),
            "hard_stop_hits": int((simulated["candidate_triggered"] & hard_stop_mask).sum()),
            "key_case_hits": int((simulated["candidate_triggered"] & key_case_hit_mask).sum()),
            "harmed_trade_count_delta_lt_-2pct": int(harmed_mask.sum()),
            "ret_sum_base": float(simulated["actual_return"].sum()),
            "ret_sum_candidate": float(simulated["candidate_return"].sum()),
            "ret_sum_delta": float((simulated["candidate_return"] - simulated["actual_return"]).sum()),
            "profit_sum_base": float(simulated["profit_loss"].sum()),
            "profit_sum_candidate": float(simulated["candidate_profit_loss"].sum()),
            "profit_sum_delta": float((simulated["candidate_profit_loss"] - simulated["profit_loss"]).sum()),
            "hard_stop_count_base": int(hard_stop_mask.sum()),
            "hard_stop_count_after": int(
                (
                    simulated["candidate_reason"].astype(str).eq("baseline")
                    & simulated["exit_reason"].astype(str).str.contains("累计收益低于-5.00%")
                ).sum()
            ),
        })

    per_trade_df = pd.DataFrame(per_trade_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["hard_stop_hits", "profit_sum_delta", "ret_sum_delta"],
        ascending=[False, False, False],
    )

    per_trade_path = RESULTS_DIR / "failed_cont_fractal_rule_probe_per_trade.csv"
    summary_path = RESULTS_DIR / "failed_cont_fractal_rule_probe_summary.csv"
    per_trade_df.to_csv(per_trade_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("saved:", per_trade_path)
    print("saved:", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
