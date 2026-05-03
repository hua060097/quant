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

BASE_SCHEME = "switch_reg_pct_mktstate_hi95_lo85"

FACTOR_COLS = [
    "date",
    "stock_code",
    "mkt_ma5_followthrough_v3",
    "market_60d_new_high_ratio",
    "severe_abnormal_pressure_10d",
    "cpv_factor",
    "high_to_close_drawdown",
    "buy_lg_amount_rate",
    "buy_sm_amount_rate",
    "net_d5_amount_to_turnover",
    "main_outflow_retail_takeover",
    "money_flow_pressure",
    "money_flow_pressure_pct",
    "abnormal_turnover",
    "abnormal_turnover_pct",
    "sector_turnover_change_5d_20d",
    "sector_turnover_change_5d_20d_pct",
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


def scheme_pick_ret(row: pd.Series, chosen_rank: int | None, action: str) -> tuple[float | None, bool]:
    if action == "cash" or chosen_rank is None:
        return np.nan, True
    if chosen_rank == 1:
        return row.get("top1_ret"), False
    if chosen_rank == 2:
        return row.get("top2_ret"), False
    if chosen_rank == 3:
        return row.get("top3_ret"), False
    return np.nan, True


def main() -> None:
    top5_df = load_top5_json(JSON_PATH)
    factor_df = load_factor_df()
    gap_map = build_buy_gap_map(top5_df)

    candidate_schemes = {
        "current_default_cpv1p5_h2c_m0p015": lambda f: pd.notna(f["cpv_factor"]) and pd.notna(f["high_to_close_drawdown"]) and f["cpv_factor"] >= 1.5 and f["high_to_close_drawdown"] <= -0.015,
        "lg_m1_h2c_m0p015": lambda f: pd.notna(f["buy_lg_amount_rate"]) and pd.notna(f["high_to_close_drawdown"]) and f["buy_lg_amount_rate"] <= -1 and f["high_to_close_drawdown"] <= -0.015,
        "lg_m2_h2c_m0p015": lambda f: pd.notna(f["buy_lg_amount_rate"]) and pd.notna(f["high_to_close_drawdown"]) and f["buy_lg_amount_rate"] <= -2 and f["high_to_close_drawdown"] <= -0.015,
        "sm_p2_h2c_m0p015": lambda f: pd.notna(f["buy_sm_amount_rate"]) and pd.notna(f["high_to_close_drawdown"]) and f["buy_sm_amount_rate"] >= 2 and f["high_to_close_drawdown"] <= -0.015,
        "outflow_takeover_p3_h2c_m0p015": lambda f: pd.notna(f["main_outflow_retail_takeover"]) and pd.notna(f["high_to_close_drawdown"]) and f["main_outflow_retail_takeover"] >= 3 and f["high_to_close_drawdown"] <= -0.015,
        "moneyflow_pressure_pct95_h2c_m0p015": lambda f: pd.notna(f["money_flow_pressure_pct"]) and pd.notna(f["high_to_close_drawdown"]) and f["money_flow_pressure_pct"] >= 0.95 and f["high_to_close_drawdown"] <= -0.015,
        "abn_turnover_pct95_h2c_m0p015": lambda f: pd.notna(f["abnormal_turnover_pct"]) and pd.notna(f["high_to_close_drawdown"]) and f["abnormal_turnover_pct"] >= 0.95 and f["high_to_close_drawdown"] <= -0.015,
        "sector_turnover_pct90_h2c_m0p015": lambda f: pd.notna(f["sector_turnover_change_5d_20d_pct"]) and pd.notna(f["high_to_close_drawdown"]) and f["sector_turnover_change_5d_20d_pct"] >= 0.90 and f["high_to_close_drawdown"] <= -0.015,
        "d5_to_turnover_le_m1_h2c_m0p015": lambda f: pd.notna(f["net_d5_amount_to_turnover"]) and pd.notna(f["high_to_close_drawdown"]) and f["net_d5_amount_to_turnover"] <= -1 and f["high_to_close_drawdown"] <= -0.015,
    }

    detail_rows: list[dict] = []
    for _, row in top5_df.iterrows():
        date = pd.to_datetime(row["date"])
        factor_day = factor_df[factor_df["date"] == date].copy().reset_index(drop=True)
        choices = _build_select_scheme_choices(date.strftime("%Y%m%d"), factor_day, row["stocks_payload"])
        base_info = choices.get(BASE_SCHEME)
        if not base_info:
            continue
        base_rank = int(base_info.get("chosen_rank", 0) or 0)
        base_code = base_info.get("selected_code")
        base_panel = factor_day.set_index("stock_code")
        base_factors = base_panel.loc[base_code].to_dict() if base_code in base_panel.index else {}

        for scheme, veto_fn in candidate_schemes.items():
            veto = bool(veto_fn(base_factors)) if base_code else True
            action = "cash" if veto else "buy"
            ret, cash_flag = scheme_pick_ret(row, base_rank if not veto else None, action)
            buy_gap = gap_map.get((date, base_code)) if base_code else None
            buy_skip = bool(
                (not cash_flag)
                and buy_gap is not None
                and (buy_gap < -0.03 or buy_gap > 0.05)
            )
            detail_rows.append(
                {
                    "date": date,
                    "scheme": scheme,
                    "selected_rank": base_rank if not veto else 0,
                    "selected_code": base_code if not veto else None,
                    "ret": pd.to_numeric(ret, errors="coerce"),
                    "ret_buy_filter": np.nan if buy_skip else pd.to_numeric(ret, errors="coerce"),
                    "cash_flag": cash_flag,
                    "buy_gap": buy_gap,
                    "buy_skip": buy_skip,
                    "veto_triggered": veto,
                    "cpv_factor": base_factors.get("cpv_factor"),
                    "high_to_close_drawdown": base_factors.get("high_to_close_drawdown"),
                    "buy_lg_amount_rate": base_factors.get("buy_lg_amount_rate"),
                    "buy_sm_amount_rate": base_factors.get("buy_sm_amount_rate"),
                    "main_outflow_retail_takeover": base_factors.get("main_outflow_retail_takeover"),
                    "money_flow_pressure_pct": base_factors.get("money_flow_pressure_pct"),
                    "abnormal_turnover_pct": base_factors.get("abnormal_turnover_pct"),
                    "sector_turnover_change_5d_20d_pct": base_factors.get("sector_turnover_change_5d_20d_pct"),
                    "net_d5_amount_to_turnover": base_factors.get("net_d5_amount_to_turnover"),
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["scheme", "date"]).reset_index(drop=True)

    summary_rows: list[dict] = []
    for scheme, sub in detail_df.groupby("scheme", sort=False):
        for filter_mode, ret_col in {
            "no_buy_filter": "ret",
            "buy_filter_open_gap": "ret_buy_filter",
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

    ranked_rows = []
    for (scheme, filter_mode), sub in summary_df.groupby(["scheme", "filter_mode"], sort=False):
        row = {"scheme": scheme, "filter_mode": filter_mode}
        for _, r in sub.iterrows():
            row[f"cum__{r['segment']}"] = r["cum_ret"]
            row[f"mdd__{r['segment']}"] = r["max_drawdown"]
            row[f"trade__{r['segment']}"] = r["trade_rate"]
        row["score"] = (
            2.0 * (row.get("cum__full") or 0.0)
            + 1.5 * (row.get("cum__second_half") or 0.0)
            + 1.5 * (row.get("cum__recent_13d") or 0.0)
            + 1.5 * (row.get("cum__seg5_last") or 0.0)
            + 0.5 * (row.get("mdd__full") or 0.0)
        )
        ranked_rows.append(row)
    ranked_df = pd.DataFrame(ranked_rows).sort_values(["filter_mode", "score"], ascending=[True, False])

    detail_path = RESULTS_DIR / "h2c_moneyflow_turnover_veto_latest_detail_20260422.csv"
    summary_path = RESULTS_DIR / "h2c_moneyflow_turnover_veto_latest_summary_20260422.csv"
    ranked_path = RESULTS_DIR / "h2c_moneyflow_turnover_veto_latest_ranked_20260422.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    ranked_df.to_csv(ranked_path, index=False, encoding="utf-8-sig")

    focus = summary_df[
        (summary_df["filter_mode"] == "buy_filter_open_gap")
        & (summary_df["segment"].isin(["full", "second_half", "recent_13d", "seg5_last"]))
    ].copy()
    print(focus.pivot(index="scheme", columns="segment", values="cum_ret").sort_values("full", ascending=False))
    print("\nMDD:")
    print(
        summary_df[
            (summary_df["filter_mode"] == "buy_filter_open_gap")
            & (summary_df["segment"] == "full")
        ][["scheme", "cum_ret", "max_drawdown", "trade_rate", "win_rate", "full_dd_le_15_count"]]
        .sort_values("cum_ret", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
