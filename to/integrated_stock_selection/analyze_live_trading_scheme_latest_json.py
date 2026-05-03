from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_trading.main import _build_select_scheme_choices


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
JSON_PATH = RESULTS_DIR / "daily_top5_details_Window_3_20260421_185521.json"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
DAILY_DIR = BASE_DIR.parent / "local_data" / "features" / "daily"

SCHEMES = [
    "mkt_exp_q30_reg10_raw_0p40_to_top2",
    "switch_reg_pct_mktstate_hi95_lo85",
    "switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto",
    "switch_reg_pct_mktstate_hi95_lo85_top2_same_threshold_veto",
    "switch_reg_pct_mktstate_hi95_lo85_top2_relaxed_threshold_veto",
]

FACTOR_COLS = [
    "date",
    "stock_code",
    "mkt_ma5_followthrough_v3",
    "market_60d_new_high_ratio",
    "severe_abnormal_pressure_10d",
    "cpv_factor",
    "high_to_close_drawdown",
]


def load_top5_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for d, payload in raw.items():
        row = {"date": pd.to_datetime(d)}
        stocks = []
        for rank in range(1, 6):
            item = payload.get(f"top{rank}", {}) or {}
            code = item.get("stock_code")
            pred = item.get("pred")
            real_ret = item.get("real_return")
            row[f"top{rank}_code"] = code
            row[f"top{rank}_pred"] = pred
            row[f"top{rank}_ret"] = real_ret
            if code:
                stocks.append(
                    {
                        "code": code,
                        "rank": rank,
                        "score": float(pred) if pred is not None else np.nan,
                    }
                )
        row["stocks_payload"] = stocks
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factor_df() -> pd.DataFrame:
    df = pd.read_parquet(FACTOR_PATH, columns=FACTOR_COLS).copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def calc_mdd(ret: pd.Series) -> float:
    traded = pd.to_numeric(ret, errors="coerce").dropna()
    if traded.empty:
        return np.nan
    eq = (1 + traded.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def dd_breach_count(ret: pd.Series, threshold: float = -0.15) -> int:
    traded = pd.to_numeric(ret, errors="coerce").dropna()
    if traded.empty:
        return 0
    eq = (1 + traded).cumprod()
    dd = eq / eq.cummax() - 1.0
    flag = dd <= threshold
    return int((flag & (~flag.shift(1, fill_value=False))).sum())


def build_buy_gap_map(df: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    date_candidates = sorted(df["date"].dt.strftime("%Y%m%d").unique().tolist())
    daily_files = sorted([p.stem for p in DAILY_DIR.glob("*.parquet") if p.stem.isdigit()])
    window_dates = [d for d in daily_files if date_candidates[0] <= d <= date_candidates[-1]]
    date_to_next = {window_dates[i]: window_dates[i + 1] for i in range(len(window_dates) - 1)}

    needed = {(d.strftime("%Y%m%d"), c) for d, c in zip(df["date"], df["top1_code"]) if pd.notna(c)}
    needed |= {(d.strftime("%Y%m%d"), c) for d, c in zip(df["date"], df["top2_code"]) if pd.notna(c)}

    next_dates = sorted({date_to_next[d] for d, _ in needed if d in date_to_next})
    open_map: dict[tuple[str, str], float] = {}
    for nd in next_dates:
        p = DAILY_DIR / f"{nd}.parquet"
        if not p.exists():
            continue
        day_df = pd.read_parquet(p, columns=["stock_code", "open"])
        for _, r in day_df.iterrows():
            open_map[(nd, r["stock_code"])] = float(r["open"]) if pd.notna(r["open"]) else np.nan

    close_map: dict[tuple[str, str], float] = {}
    for d in sorted({d for d, _ in needed}):
        p = DAILY_DIR / f"{d}.parquet"
        if not p.exists():
            continue
        day_df = pd.read_parquet(p, columns=["stock_code", "close"])
        for _, r in day_df.iterrows():
            close_map[(d, r["stock_code"])] = float(r["close"]) if pd.notna(r["close"]) else np.nan

    gap_map: dict[tuple[pd.Timestamp, str], float] = {}
    for d, code in needed:
        if d not in date_to_next:
            continue
        next_d = date_to_next[d]
        close_px = close_map.get((d, code))
        open_px = open_map.get((next_d, code))
        if close_px is None or open_px is None or not np.isfinite(close_px) or not np.isfinite(open_px) or close_px <= 0:
            continue
        gap_map[(pd.to_datetime(d), code)] = open_px / close_px - 1.0
    return gap_map


def summarize(ret: pd.Series) -> tuple[float, float, float, float]:
    traded = pd.to_numeric(ret, errors="coerce").dropna()
    if traded.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(traded.mean()),
        float((1 + traded).prod() - 1),
        calc_mdd(traded),
        float((traded > 0).mean()),
    )


def split_segments(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.sort_values("date").reset_index(drop=True)
    n = len(work)
    half = n // 2
    fifth = n // 5 if n >= 5 else n
    recent_n = min(13, n)
    return {
        "full": work,
        "first_half": work.iloc[:half],
        "second_half": work.iloc[half:],
        "seg5_last": work.iloc[-fifth:] if fifth > 0 else work.iloc[0:0],
        f"recent_{recent_n}d": work.iloc[-recent_n:],
    }


def scheme_pick_ret(row: pd.Series, info: dict) -> tuple[float | None, bool]:
    action = info.get("action", "buy")
    if action == "cash":
        return np.nan, True
    chosen_rank = int(info.get("chosen_rank", 0) or 0)
    if chosen_rank == 1:
        return row.get("top1_ret"), False
    if chosen_rank == 2:
        return row.get("top2_ret"), False
    return np.nan, True


def build_top2_reg_guard_variants(
    date: pd.Timestamp,
    factor_day: pd.DataFrame,
    selected_stocks: list[dict],
    base_choices: dict,
) -> dict[str, dict]:
    if len(selected_stocks) < 2:
        return {}

    base_info = base_choices.get("switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto")
    if not base_info:
        return {}

    top1 = selected_stocks[0]
    top2 = selected_stocks[1]
    top1_code = top1.get("code")
    top2_code = top2.get("code")

    state_value = base_info.get("state_signal_value")
    state_threshold = base_info.get("state_threshold")
    top1_reg_pct = base_info.get("top1_second_signal_value")
    top2_reg_pct = base_info.get("top2_second_signal_value")
    if state_value is None or state_threshold is None or top1_reg_pct is None:
        return {}

    weak_env = bool(state_value < state_threshold)
    top1_threshold = 0.85 if weak_env else 0.95
    top2_relaxed_threshold = 0.95 if weak_env else 0.98

    top_factor_panel = factor_day[["stock_code", "cpv_factor", "high_to_close_drawdown"]].copy()
    top_factor_panel["cpv_factor"] = pd.to_numeric(top_factor_panel["cpv_factor"], errors="coerce")
    top_factor_panel["high_to_close_drawdown"] = pd.to_numeric(top_factor_panel["high_to_close_drawdown"], errors="coerce")
    top_factor_panel = top_factor_panel.dropna(subset=["cpv_factor", "high_to_close_drawdown"])
    factor_map = top_factor_panel.set_index("stock_code")[["cpv_factor", "high_to_close_drawdown"]].to_dict("index")

    def _make_variant(name: str, top2_threshold: float) -> dict:
        if top1_reg_pct < top1_threshold:
            chosen_rank = 1
            selected_code = top1_code
        elif top2_reg_pct is not None and top2_reg_pct < top2_threshold:
            chosen_rank = 2
            selected_code = top2_code
        else:
            chosen_rank = 0
            selected_code = None

        if chosen_rank == 1:
            selected_factor = factor_map.get(top1_code)
        elif chosen_rank == 2:
            selected_factor = factor_map.get(top2_code)
        else:
            selected_factor = None

        selected_cpv = float(selected_factor["cpv_factor"]) if selected_factor is not None else None
        selected_h2c = float(selected_factor["high_to_close_drawdown"]) if selected_factor is not None else None
        veto_triggered = bool(
            selected_cpv is not None and selected_h2c is not None
            and selected_cpv >= 1.5
            and selected_h2c <= -0.015
        )

        if chosen_rank == 1:
            signal_value = top1_reg_pct
        elif chosen_rank == 2:
            signal_value = top2_reg_pct
        else:
            signal_value = None

        return {
            "name": name,
            "chosen_rank": 0 if veto_triggered else chosen_rank,
            "selected_code": None if veto_triggered else selected_code,
            "threshold": top1_threshold,
            "signal_value": signal_value,
            "chosen_by_top1": bool(chosen_rank == 1 and not veto_triggered),
            "action": "cash" if veto_triggered or chosen_rank == 0 else "buy",
            "state_factor_name": "mkt_ma5_followthrough_v3_expanding_q30_shift1",
            "state_threshold": state_threshold,
            "state_signal_value": state_value,
            "second_factor_name": "selected_severe_abnormal_pressure_10d_pct_rank",
            "second_threshold": top1_threshold,
            "top2_second_threshold": top2_threshold,
            "top1_second_signal_value": top1_reg_pct,
            "top2_second_signal_value": top2_reg_pct,
            "veto_factor_name": "selected_cpv_factor_and_high_to_close_drawdown",
            "veto_cpv_threshold": 1.5,
            "veto_h2c_threshold": -0.015,
            "veto_cpv_value": selected_cpv,
            "veto_h2c_value": selected_h2c,
            "veto_triggered": veto_triggered,
        }

    return {
        "switch_reg_pct_mktstate_hi95_lo85_top2_same_threshold_veto": _make_variant(
            "switch_reg_pct_mktstate_hi95_lo85_top2_same_threshold_veto",
            top1_threshold,
        ),
        "switch_reg_pct_mktstate_hi95_lo85_top2_relaxed_threshold_veto": _make_variant(
            "switch_reg_pct_mktstate_hi95_lo85_top2_relaxed_threshold_veto",
            top2_relaxed_threshold,
        ),
    }


def main() -> None:
    top5_df = load_top5_json(JSON_PATH)
    factor_df = load_factor_df()
    gap_map = build_buy_gap_map(top5_df)

    detail_rows: list[dict] = []
    for _, row in top5_df.iterrows():
        date = pd.to_datetime(row["date"])
        factor_day = factor_df[factor_df["date"] == date].copy()
        factor_day = factor_day.reset_index(drop=True)
        choices = _build_select_scheme_choices(date.strftime("%Y%m%d"), factor_day, row["stocks_payload"])
        choices.update(build_top2_reg_guard_variants(date, factor_day, row["stocks_payload"], choices))
        for scheme in SCHEMES:
            info = choices.get(scheme)
            if not info:
                continue
            selected_ret, cash_flag = scheme_pick_ret(row, info)
            selected_code = info.get("selected_code")
            buy_gap = gap_map.get((date, selected_code)) if selected_code else None
            buy_skip = bool(
                (not cash_flag)
                and buy_gap is not None
                and (buy_gap < -0.03 or buy_gap > 0.05)
            )
            detail_rows.append(
                {
                    "date": date,
                    "scheme": scheme,
                    "scheme_name": info.get("name"),
                    "action": info.get("action", "buy"),
                    "chosen_rank": info.get("chosen_rank", 0),
                    "selected_code": selected_code,
                    "selected_ret": pd.to_numeric(selected_ret, errors="coerce"),
                    "selected_ret_buy_filter": np.nan if buy_skip else pd.to_numeric(selected_ret, errors="coerce"),
                    "cash_flag": cash_flag,
                    "buy_gap": buy_gap,
                    "buy_skip": buy_skip,
                    "threshold": info.get("threshold"),
                    "signal_value": info.get("signal_value"),
                    "veto_triggered": info.get("veto_triggered", False),
                    "veto_cpv_value": info.get("veto_cpv_value"),
                    "veto_h2c_value": info.get("veto_h2c_value"),
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["scheme", "date"]).reset_index(drop=True)

    summary_rows: list[dict] = []
    for scheme, sub in detail_df.groupby("scheme", sort=False):
        for filter_mode, ret_col in {
            "no_buy_filter": "selected_ret",
            "buy_filter_open_gap": "selected_ret_buy_filter",
        }.items():
            working = sub.copy()
            working["ret_used"] = pd.to_numeric(working[ret_col], errors="coerce")
            segments = split_segments(working)
            full_dd_count = dd_breach_count(working["ret_used"])
            for seg_name, seg_df in segments.items():
                mean_ret, cum_ret, mdd, win_rate = summarize(seg_df["ret_used"])
                summary_rows.append(
                    {
                        "json_name": JSON_PATH.name,
                        "scheme": scheme,
                        "scheme_name": sub["scheme_name"].iloc[0],
                        "filter_mode": filter_mode,
                        "segment": seg_name,
                        "days": int(len(seg_df)),
                        "active_days": int(seg_df["ret_used"].notna().sum()),
                        "trade_rate": float(seg_df["ret_used"].notna().mean()),
                        "cash_days": int(seg_df["cash_flag"].sum()),
                        "buy_skip_days": int(seg_df["buy_skip"].sum()) if filter_mode == "buy_filter_open_gap" else 0,
                        "mean_ret": mean_ret,
                        "cum_ret": cum_ret,
                        "max_drawdown": mdd,
                        "win_rate": win_rate,
                        "full_dd_le_15_count": full_dd_count,
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["filter_mode", "segment", "cum_ret"], ascending=[True, True, False]
    ).reset_index(drop=True)

    score_rows = []
    for (scheme, filter_mode), sub in summary_df.groupby(["scheme", "filter_mode"], sort=False):
        row = {
            "scheme": scheme,
            "scheme_name": sub["scheme_name"].iloc[0],
            "filter_mode": filter_mode,
        }
        for _, r in sub.iterrows():
            row[f"cum__{r['segment']}"] = r["cum_ret"]
            row[f"mdd__{r['segment']}"] = r["max_drawdown"]
            row[f"trade__{r['segment']}"] = r["trade_rate"]
        row["score"] = (
            row.get("cum__second_half", 0) * 3.0
            + row.get("cum__seg5_last", 0) * 4.0
            + row.get("cum__recent_13d", 0) * 3.0
            + row.get("cum__full", 0) * 0.5
            + row.get("mdd__full", 0) * 0.5
        )
        score_rows.append(row)
    score_df = pd.DataFrame(score_rows).sort_values(["filter_mode", "score"], ascending=[True, False]).reset_index(drop=True)

    detail_path = RESULTS_DIR / "live_trading_scheme_compare_20260421_185521_detail.csv"
    summary_path = RESULTS_DIR / "live_trading_scheme_compare_20260421_185521_summary.csv"
    score_path = RESULTS_DIR / "live_trading_scheme_compare_20260421_185521_ranked.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")

    print(f"detail -> {detail_path}")
    print(f"summary -> {summary_path}")
    print(f"ranked -> {score_path}")
    print(score_df.to_string(index=False))


if __name__ == "__main__":
    main()
