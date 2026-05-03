from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DAILY_DIR = BASE_DIR.parent / "local_data" / "features" / "daily"
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"

JSON_PATHS = [
    RESULTS_DIR / "daily_top5_details_Window_3_20260417_145807.json",
    RESULTS_DIR / "daily_top5_details_Window_3_20260417_154536.json",
]

NEEDED_FACTOR_COLS = [
    "date",
    "stock_code",
    "mkt_ma5_followthrough_v3",
    "severe_abnormal_pressure_10d",
    "mkt_dispersion_shock_v2",
    "mkt_regime_trend_pressure_v2",
    "market_trend_strength",
    "abnormal_turnover",
    "ppo_strength",
]


def expanding_threshold(series: pd.Series, q: float, min_hist: int = 20) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = np.full(len(vals), np.nan)
    hist: list[float] = []
    for i, v in enumerate(vals):
        if len(hist) >= min_hist:
            out[i] = float(pd.Series(hist).quantile(q))
        if pd.notna(v):
            hist.append(float(v))
    return pd.Series(out, index=series.index)


def calc_mdd(ret: pd.Series) -> float:
    if ret.dropna().empty:
        return np.nan
    eq = (1 + ret.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def load_top5_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for d, payload in raw.items():
        row = {"date": pd.to_datetime(d)}
        for rank in range(1, 6):
            key = f"top{rank}"
            item = payload.get(key, {})
            row[f"top{rank}_code"] = item.get("stock_code")
            row[f"top{rank}_ret"] = item.get("real_return")
            row[f"top{rank}_pred"] = item.get("pred")
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def load_factor_frame() -> pd.DataFrame:
    factor_df = pd.read_parquet(FACTOR_PATH, columns=NEEDED_FACTOR_COLS).copy()
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    return factor_df


def attach_factor(df: pd.DataFrame, factor_df: pd.DataFrame, rank: int, factor_name: str) -> pd.Series:
    code_col = f"top{rank}_code"
    tmp = (
        df[["date", code_col]]
        .rename(columns={code_col: "stock_code"})
        .merge(
            factor_df[["date", "stock_code", factor_name]],
            on=["date", "stock_code"],
            how="left",
        )
    )
    return pd.to_numeric(tmp[factor_name], errors="coerce")


def load_buy_filter_map(df: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
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


def summarize(ret: pd.Series, active_mask: pd.Series, buy_skip_mask: pd.Series, scheme: str) -> dict:
    traded = ret[active_mask].dropna()
    return {
        "scheme": scheme,
        "days": int(len(ret)),
        "active_days": int(active_mask.sum()),
        "pass_rate": float(active_mask.mean()) if len(active_mask) else np.nan,
        "buy_skip_days": int(buy_skip_mask.sum()),
        "buy_skip_rate": float(buy_skip_mask.mean()) if len(buy_skip_mask) else np.nan,
        "win_rate": float((traded > 0).mean()) if len(traded) else np.nan,
        "mean_ret": float(traded.mean()) if len(traded) else np.nan,
        "cum_ret": float((1 + traded).prod() - 1) if len(traded) else np.nan,
        "max_drawdown": calc_mdd(traded),
    }


def build_scheme_frames(df: pd.DataFrame) -> list[tuple[str, pd.Series, pd.Series]]:
    work = df.copy()
    mkt_thr_q30 = expanding_threshold(work["mkt_ma5_followthrough_v3"], 0.30)
    reg_top1 = pd.to_numeric(work["top1__severe_abnormal_pressure_10d"], errors="coerce")
    choose_top1 = (pd.to_numeric(work["mkt_ma5_followthrough_v3"], errors="coerce") >= mkt_thr_q30) & mkt_thr_q30.notna()

    top1_ret = pd.to_numeric(work["top1_ret"], errors="coerce")
    top2_ret = pd.to_numeric(work["top2_ret"], errors="coerce")

    current_rank = pd.Series(np.where(choose_top1 & (reg_top1 >= 0.40), 2, np.where(choose_top1, 1, 2)), index=work.index)
    current_ret = pd.Series(np.where(choose_top1 & (reg_top1 >= 0.40), top2_ret, np.where(choose_top1, top1_ret, top2_ret)), index=work.index)

    ma5_q60 = expanding_threshold(work["mkt_ma5_followthrough_v3"], 0.60)
    shock_q70 = expanding_threshold(work["mkt_dispersion_shock_v2"], 0.70)
    skip_ma5_shock = (
        (pd.to_numeric(work["mkt_ma5_followthrough_v3"], errors="coerce") <= ma5_q60) &
        (pd.to_numeric(work["mkt_dispersion_shock_v2"], errors="coerce") <= shock_q70) &
        ma5_q60.notna() & shock_q70.notna()
    )

    abn_diff = pd.to_numeric(work["top1__abnormal_turnover"], errors="coerce") - pd.to_numeric(work["top2__abnormal_turnover"], errors="coerce")
    ppo_diff = pd.to_numeric(work["top1__ppo_strength"], errors="coerce") - pd.to_numeric(work["top2__ppo_strength"], errors="coerce")
    abn_q30 = expanding_threshold(abn_diff, 0.30)
    ppo_q40 = expanding_threshold(ppo_diff, 0.40)
    diff_flip = (abn_diff >= abn_q30) & (ppo_diff >= ppo_q40) & abn_q30.notna() & ppo_q40.notna()

    trend_q50 = expanding_threshold(work["market_trend_strength"], 0.50)
    pressure_q50 = expanding_threshold(work["mkt_regime_trend_pressure_v2"], 0.50)
    skip_trend_pressure = (
        (pd.to_numeric(work["market_trend_strength"], errors="coerce") <= trend_q50) &
        (pd.to_numeric(work["mkt_regime_trend_pressure_v2"], errors="coerce") <= pressure_q50) &
        trend_q50.notna() & pressure_q50.notna()
    )

    return [
        ("baseline_top1", pd.Series(1, index=work.index), top1_ret),
        ("baseline_top2", pd.Series(2, index=work.index), top2_ret),
        ("mkt_exp_q30", pd.Series(np.where(choose_top1, 1, 2), index=work.index), pd.Series(np.where(choose_top1, top1_ret, top2_ret), index=work.index)),
        ("current_switch_mkt_exp_q30_reg10_raw_0p40", current_rank, current_ret),
        (
            "diff_switch_plus_current_switch",
            pd.Series(np.where(diff_flip, 2, current_rank), index=work.index),
            pd.Series(np.where(diff_flip, top2_ret, current_ret), index=work.index),
        ),
        (
            "riskoff_skip_plus_current_switch",
            pd.Series(np.where(skip_ma5_shock, np.nan, current_rank), index=work.index),
            pd.Series(np.where(skip_ma5_shock, np.nan, current_ret), index=work.index),
        ),
        (
            "alt_skip_trend_pressure_plus_current_switch",
            pd.Series(np.where(skip_trend_pressure, np.nan, current_rank), index=work.index),
            pd.Series(np.where(skip_trend_pressure, np.nan, current_ret), index=work.index),
        ),
    ]


def compare_on_json(json_path: Path, factor_df: pd.DataFrame) -> pd.DataFrame:
    df = load_top5_json(json_path)

    market_cols = [
        "mkt_ma5_followthrough_v3",
        "mkt_dispersion_shock_v2",
        "mkt_regime_trend_pressure_v2",
        "market_trend_strength",
    ]
    market_map = factor_df.groupby("date")[market_cols].first().reset_index()
    df = df.merge(market_map, on="date", how="left")

    for rank in [1, 2]:
        for factor_name in ["severe_abnormal_pressure_10d", "abnormal_turnover", "ppo_strength"]:
            df[f"top{rank}__{factor_name}"] = attach_factor(df, factor_df, rank, factor_name)

    gap_map = load_buy_filter_map(df)
    rows: list[dict] = []

    for scheme_name, selected_rank, selected_ret in build_scheme_frames(df):
        no_filter_active = selected_ret.notna()
        no_filter_skip = pd.Series(False, index=df.index)
        out = summarize(selected_ret, no_filter_active, no_filter_skip, scheme_name)
        out["json_name"] = json_path.name
        out["label_version"] = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"
        out["filter_mode"] = "no_buy_filter"
        rows.append(out)

        chosen_code = pd.Series(np.where(selected_rank == 1, df["top1_code"], np.where(selected_rank == 2, df["top2_code"], None)), index=df.index)
        buy_skip = []
        filtered_ret = []
        for dt, code, ret in zip(df["date"], chosen_code, selected_ret):
            if pd.isna(ret) or code is None or pd.isna(code):
                buy_skip.append(False)
                filtered_ret.append(np.nan)
                continue
            gap = gap_map.get((pd.Timestamp(dt), str(code)))
            should_skip = gap is not None and (gap < -0.03 or gap > 0.05)
            buy_skip.append(bool(should_skip))
            filtered_ret.append(np.nan if should_skip else ret)

        buy_skip_mask = pd.Series(buy_skip, index=df.index)
        filtered_ret_s = pd.Series(filtered_ret, index=df.index)
        active_mask = filtered_ret_s.notna()
        out2 = summarize(filtered_ret_s, active_mask, buy_skip_mask, scheme_name)
        out2["json_name"] = json_path.name
        out2["label_version"] = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"
        out2["filter_mode"] = "buy_filter_open_gap"
        rows.append(out2)

    return pd.DataFrame(rows)


def main() -> None:
    factor_df = load_factor_frame()
    summary = pd.concat([compare_on_json(p, factor_df) for p in JSON_PATHS], ignore_index=True)
    summary = summary.sort_values(["json_name", "filter_mode", "cum_ret"], ascending=[True, True, False]).reset_index(drop=True)
    out_path = RESULTS_DIR / "latest_json_scheme_compare_20260417.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
