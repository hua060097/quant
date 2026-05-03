from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_trading.main import _build_select_scheme_choices


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
JSON_PATH = RESULTS_DIR / "daily_top5_details_Window_3_20260417_154536.json"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
DAILY_DIR = BASE_DIR.parent / "local_data" / "features" / "daily"

BASE_SCHEMES = [
    "mkt_exp_q30_reg10_raw_0p40_to_top2",
    "switch_reg_pct_mktstate_hi95_lo85",
    "switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto",
]

SLOPE_FACTOR_COLS = [
    "date",
    "stock_code",
    "mkt_ma5_followthrough_v3",
    "market_60d_new_high_ratio",
    "severe_abnormal_pressure_10d",
    "cpv_factor",
    "high_to_close_drawdown",
    "log_price_slope_5d",
    "log_price_slope_10d",
    "log_price_slope_20d",
    "log_price_slope_10d_x_log_circ_mv",
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
    df = pd.read_parquet(FACTOR_PATH, columns=SLOPE_FACTOR_COLS).copy()
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

    needed = set()
    for rank in range(1, 6):
        needed |= {
            (d.strftime("%Y%m%d"), c)
            for d, c in zip(df["date"], df[f"top{rank}_code"])
            if pd.notna(c)
        }

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


def pick_rank_ret(row: pd.Series, chosen_rank: int | None, action: str = "buy") -> tuple[float | None, bool]:
    if action == "cash" or chosen_rank is None or chosen_rank <= 0:
        return np.nan, True
    return row.get(f"top{chosen_rank}_ret"), False


def should_switch_slope10d_ge6(top1_factors: dict) -> bool:
    v = pd.to_numeric(top1_factors.get("log_price_slope_10d"), errors="coerce")
    return bool(pd.notna(v) and v >= 6.0)


def should_switch_slope20d_band(top1_factors: dict) -> bool:
    v = pd.to_numeric(top1_factors.get("log_price_slope_20d"), errors="coerce")
    return bool(pd.notna(v) and 1.5 <= v < 2.0)


def should_switch_slope10d_mv_band(top1_factors: dict) -> bool:
    v = pd.to_numeric(top1_factors.get("log_price_slope_10d_x_log_circ_mv"), errors="coerce")
    return bool(pd.notna(v) and 40.0 <= v < 60.0)


def main() -> None:
    top5_df = load_top5_json(JSON_PATH)
    factor_df = load_factor_df()
    gap_map = build_buy_gap_map(top5_df)

    detail_rows: list[dict] = []
    for _, row in top5_df.iterrows():
        date = pd.to_datetime(row["date"])
        factor_day = factor_df[factor_df["date"] == date].copy().reset_index(drop=True)
        choices = _build_select_scheme_choices(date.strftime("%Y%m%d"), factor_day, row["stocks_payload"])
        factor_panel = factor_day.set_index("stock_code")
        top1_code = row.get("top1_code")
        top1_factors = factor_panel.loc[top1_code].to_dict() if pd.notna(top1_code) and top1_code in factor_panel.index else {}

        weak_env = False
        default_info = choices.get("switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto")
        if default_info:
            state_signal = default_info.get("state_signal_value")
            state_thr = default_info.get("state_threshold")
            if state_signal is not None and state_thr is not None:
                weak_env = bool(float(state_signal) < float(state_thr))

        for scheme in BASE_SCHEMES:
            info = choices.get(scheme)
            if not info:
                continue
            ret, cash_flag = pick_rank_ret(row, int(info.get("chosen_rank", 0) or 0), info.get("action", "buy"))
            selected_code = info.get("selected_code")
            buy_gap = gap_map.get((date, selected_code)) if selected_code else None
            buy_skip = bool((not cash_flag) and buy_gap is not None and (buy_gap < -0.03 or buy_gap > 0.05))
            detail_rows.append(
                {
                    "date": date,
                    "scheme": scheme,
                    "scheme_name": info.get("name"),
                    "chosen_rank": int(info.get("chosen_rank", 0) or 0),
                    "selected_code": selected_code,
                    "selected_ret": pd.to_numeric(ret, errors="coerce"),
                    "selected_ret_buy_filter": np.nan if buy_skip else pd.to_numeric(ret, errors="coerce"),
                    "cash_flag": cash_flag,
                    "buy_gap": buy_gap,
                    "buy_skip": buy_skip,
                    "weak_env": weak_env,
                }
            )

        pure_slope_schemes = [
            ("slope10d_ge6_top1_to_top2", should_switch_slope10d_ge6(top1_factors)),
            ("slope20d_1p5_2p0_top1_to_top2", should_switch_slope20d_band(top1_factors)),
            ("slope10d_mv_40_60_top1_to_top2", should_switch_slope10d_mv_band(top1_factors)),
        ]
        for scheme_name, do_switch in pure_slope_schemes:
            chosen_rank = 2 if do_switch else 1
            ret, cash_flag = pick_rank_ret(row, chosen_rank, "buy")
            selected_code = row.get(f"top{chosen_rank}_code")
            buy_gap = gap_map.get((date, selected_code)) if selected_code else None
            buy_skip = bool((not cash_flag) and buy_gap is not None and (buy_gap < -0.03 or buy_gap > 0.05))
            detail_rows.append(
                {
                    "date": date,
                    "scheme": scheme_name,
                    "scheme_name": scheme_name,
                    "chosen_rank": chosen_rank,
                    "selected_code": selected_code,
                    "selected_ret": pd.to_numeric(ret, errors="coerce"),
                    "selected_ret_buy_filter": np.nan if buy_skip else pd.to_numeric(ret, errors="coerce"),
                    "cash_flag": cash_flag,
                    "buy_gap": buy_gap,
                    "buy_skip": buy_skip,
                    "weak_env": weak_env,
                }
            )

        if default_info:
            default_rank = int(default_info.get("chosen_rank", 0) or 0)
            default_action = default_info.get("action", "buy")
            overlay_schemes = [
                ("weak_overlay_slope10d_ge6_on_default", should_switch_slope10d_ge6(top1_factors)),
                ("weak_overlay_slope20d_1p5_2p0_on_default", should_switch_slope20d_band(top1_factors)),
                ("weak_overlay_slope10d_mv_40_60_on_default", should_switch_slope10d_mv_band(top1_factors)),
            ]
            for scheme_name, slope_signal in overlay_schemes:
                if default_action == "cash":
                    chosen_rank = None
                    action = "cash"
                elif weak_env and default_rank == 1 and slope_signal:
                    chosen_rank = 2
                    action = "buy"
                else:
                    chosen_rank = default_rank
                    action = default_action
                ret, cash_flag = pick_rank_ret(row, chosen_rank, action)
                selected_code = row.get(f"top{chosen_rank}_code") if chosen_rank else None
                buy_gap = gap_map.get((date, selected_code)) if selected_code else None
                buy_skip = bool((not cash_flag) and buy_gap is not None and (buy_gap < -0.03 or buy_gap > 0.05))
                detail_rows.append(
                    {
                        "date": date,
                        "scheme": scheme_name,
                        "scheme_name": scheme_name,
                        "chosen_rank": int(chosen_rank or 0),
                        "selected_code": selected_code,
                        "selected_ret": pd.to_numeric(ret, errors="coerce"),
                        "selected_ret_buy_filter": np.nan if buy_skip else pd.to_numeric(ret, errors="coerce"),
                        "cash_flag": cash_flag,
                        "buy_gap": buy_gap,
                        "buy_skip": buy_skip,
                        "weak_env": weak_env,
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
            weak_trade_rate = float(working.loc[working["weak_env"], "ret_used"].notna().mean()) if (working["weak_env"]).any() else np.nan
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
                        "weak_env_trade_rate_full": weak_trade_rate if seg_name == "full" else np.nan,
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["filter_mode", "segment", "cum_ret"], ascending=[True, True, False]
    ).reset_index(drop=True)

    ranked_rows = []
    for (scheme, filter_mode), sub in summary_df.groupby(["scheme", "filter_mode"], sort=False):
        row = {
            "scheme": scheme,
            "scheme_name": sub["scheme_name"].iloc[0],
            "filter_mode": filter_mode,
        }
        seg_map = {r["segment"]: r for _, r in sub.iterrows()}
        for seg_name in ["full", "first_half", "second_half", "seg5_last", "recent_13d"]:
            if seg_name in seg_map:
                r = seg_map[seg_name]
                row[f"mean__{seg_name}"] = r["mean_ret"]
                row[f"cum__{seg_name}"] = r["cum_ret"]
                row[f"mdd__{seg_name}"] = r["max_drawdown"]
                row[f"trade__{seg_name}"] = r["trade_rate"]
        row["score"] = (
            1.00 * float(row.get("cum__second_half", np.nan) or 0.0)
            + 0.90 * float(row.get("cum__seg5_last", np.nan) or 0.0)
            + 0.80 * float(row.get("cum__recent_13d", np.nan) or 0.0)
            + 0.50 * float(row.get("cum__full", np.nan) or 0.0)
            + 0.30 * float(row.get("mean__full", np.nan) or 0.0)
            - 0.70 * abs(float(row.get("mdd__full", np.nan) or 0.0))
        )
        ranked_rows.append(row)

    ranked_df = pd.DataFrame(ranked_rows).sort_values(["filter_mode", "score"], ascending=[True, False]).reset_index(drop=True)

    detail_path = RESULTS_DIR / "slope_switch_vs_live_trading_detail_20260417.csv"
    summary_path = RESULTS_DIR / "slope_switch_vs_live_trading_summary_20260417.csv"
    ranked_path = RESULTS_DIR / "slope_switch_vs_live_trading_ranked_20260417.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    ranked_df.to_csv(ranked_path, index=False, encoding="utf-8-sig")

    print(detail_path)
    print(summary_path)
    print(ranked_path)
    print(ranked_df.to_string(index=False))


if __name__ == "__main__":
    main()
