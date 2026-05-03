from __future__ import annotations

import json
from dataclasses import dataclass
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

MARKET_FACTORS = [
    "mkt_ma5_followthrough_v3",
    "market_trend_strength",
    "mkt_regime_trend_pressure_v2",
    "mkt_dispersion_shock_v2",
    "mkt_sector_dispersion",
    "sector_hotspot_concentration_v3",
    "sector_leadership_persistence_v3",
    "strong_stock_volume_confirmation_v3",
    "mkt_emotion_stability_v2",
    "emotion_quality_balance_v3",
]

STOCK_FACTORS = [
    "severe_abnormal_pressure_10d",
    "abnormal_turnover",
    "ppo_strength",
    "volume_structure_alpha",
    "sector_turnover_change_5d_20d",
    "return_10d_sector_mean",
    "sector_leader_capture_rank",
]

FACTOR_COLS = ["date", "stock_code"] + MARKET_FACTORS + STOCK_FACTORS


@dataclass(frozen=True)
class RuleDef:
    name: str
    family: str
    mode: str  # skip / switch
    desc: str


def load_top5_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for d, payload in raw.items():
        row = {"date": pd.to_datetime(d)}
        for rank in range(1, 6):
            item = payload.get(f"top{rank}", {})
            row[f"top{rank}_code"] = item.get("stock_code")
            row[f"top{rank}_ret"] = item.get("real_return")
            row[f"top{rank}_pred"] = item.get("pred")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factor_df() -> pd.DataFrame:
    factor_df = pd.read_parquet(FACTOR_PATH, columns=FACTOR_COLS).copy()
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    return factor_df


def calc_mdd(ret: pd.Series) -> float:
    traded = ret.dropna()
    if traded.empty:
        return np.nan
    eq = (1 + traded.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


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
        day_df = pd.read_parquet(DAILY_DIR / f"{nd}.parquet", columns=["stock_code", "open"])
        for _, r in day_df.iterrows():
            open_map[(nd, r["stock_code"])] = float(r["open"]) if pd.notna(r["open"]) else np.nan

    close_map: dict[tuple[str, str], float] = {}
    for d in sorted({d for d, _ in needed}):
        day_df = pd.read_parquet(DAILY_DIR / f"{d}.parquet", columns=["stock_code", "close"])
        for _, r in day_df.iterrows():
            close_map[(d, r["stock_code"])] = float(r["close"]) if pd.notna(r["close"]) else np.nan

    out: dict[tuple[pd.Timestamp, str], float] = {}
    for d, code in needed:
        if d not in date_to_next:
            continue
        next_d = date_to_next[d]
        c = close_map.get((d, code))
        o = open_map.get((next_d, code))
        if c is None or o is None or not np.isfinite(c) or not np.isfinite(o) or c <= 0:
            continue
        out[(pd.to_datetime(d), code)] = o / c - 1.0
    return out


def summarize_scheme(name: str, ret: pd.Series, active_mask: pd.Series, skip_mask: pd.Series, json_name: str, label_version: str, filter_mode: str, family: str, desc: str) -> dict:
    traded = ret[active_mask].dropna()
    return {
        "scheme": name,
        "family": family,
        "desc": desc,
        "json_name": json_name,
        "label_version": label_version,
        "filter_mode": filter_mode,
        "days": int(len(ret)),
        "active_days": int(active_mask.sum()),
        "pass_rate": float(active_mask.mean()) if len(active_mask) else np.nan,
        "skip_days": int(skip_mask.sum()),
        "skip_rate": float(skip_mask.mean()) if len(skip_mask) else np.nan,
        "win_rate": float((traded > 0).mean()) if len(traded) else np.nan,
        "mean_ret": float(traded.mean()) if len(traded) else np.nan,
        "cum_ret": float((1 + traded).prod() - 1) if len(traded) else np.nan,
        "max_drawdown": calc_mdd(traded),
    }


def attach_pct(df: pd.DataFrame, factor_df: pd.DataFrame, rank: int, factor: str) -> pd.Series:
    pct_col = f"{factor}__pct"
    if pct_col not in factor_df.columns:
        raise KeyError(pct_col)
    code_col = f"top{rank}_code"
    tmp = (
        df[["date", code_col]]
        .rename(columns={code_col: "stock_code"})
        .merge(factor_df[["date", "stock_code", pct_col]], on=["date", "stock_code"], how="left")
    )
    return pd.to_numeric(tmp[pct_col], errors="coerce")


def attach_raw(df: pd.DataFrame, factor_df: pd.DataFrame, rank: int, factor: str) -> pd.Series:
    code_col = f"top{rank}_code"
    tmp = (
        df[["date", code_col]]
        .rename(columns={code_col: "stock_code"})
        .merge(factor_df[["date", "stock_code", factor]], on=["date", "stock_code"], how="left")
    )
    return pd.to_numeric(tmp[factor], errors="coerce")


def prepare_factor_df(raw: pd.DataFrame) -> pd.DataFrame:
    factor_df = raw.copy()
    for factor in STOCK_FACTORS:
        factor_df[f"{factor}__pct"] = factor_df.groupby("date")[factor].rank(pct=True, method="average")
    return factor_df


def build_base_df(json_path: Path, factor_df: pd.DataFrame) -> pd.DataFrame:
    df = load_top5_json(json_path)
    market_map = factor_df.groupby("date")[MARKET_FACTORS].first().reset_index()
    df = df.merge(market_map, on="date", how="left")
    for factor in STOCK_FACTORS:
        for rank in [1, 2]:
            df[f"top{rank}__{factor}"] = attach_raw(df, factor_df, rank, factor)
            df[f"top{rank}__{factor}__pct"] = attach_pct(df, factor_df, rank, factor)
    return df


def current_switch(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    thr = expanding_threshold(df["mkt_ma5_followthrough_v3"], 0.30)
    choose_top1 = (pd.to_numeric(df["mkt_ma5_followthrough_v3"], errors="coerce") >= thr) & thr.notna()
    reg = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
    ranks = pd.Series(np.where(choose_top1 & (reg >= 0.40), 2, np.where(choose_top1, 1, 2)), index=df.index)
    rets = pd.Series(np.where(choose_top1 & (reg >= 0.40), df["top2_ret"], np.where(choose_top1, df["top1_ret"], df["top2_ret"])), index=df.index)
    return ranks, pd.to_numeric(rets, errors="coerce")


def market_skip_rules(df: pd.DataFrame) -> dict[RuleDef, pd.Series]:
    out: dict[RuleDef, pd.Series] = {}
    low_factor_qs = {
        "mkt_ma5_followthrough_v3": [0.5, 0.6, 0.7],
        "market_trend_strength": [0.5, 0.6],
        "mkt_regime_trend_pressure_v2": [0.4, 0.5, 0.6],
        "mkt_dispersion_shock_v2": [0.6, 0.7],
        "sector_leadership_persistence_v3": [0.4, 0.5],
        "strong_stock_volume_confirmation_v3": [0.4, 0.5],
        "mkt_emotion_stability_v2": [0.4, 0.5],
        "emotion_quality_balance_v3": [0.4, 0.5],
    }
    high_factor_qs = {
        "mkt_sector_dispersion": [0.6, 0.7],
        "sector_hotspot_concentration_v3": [0.7, 0.8],
    }

    for factor, qs in low_factor_qs.items():
        s = pd.to_numeric(df[factor], errors="coerce")
        for q in qs:
            thr = expanding_threshold(s, q)
            mask = (s <= thr) & thr.notna()
            out[RuleDef(f"skip_{factor}_le_expq{int(q*100)}", "market_skip_single", "skip", f"{factor} <= expanding q{int(q*100)}")] = mask

    for factor, qs in high_factor_qs.items():
        s = pd.to_numeric(df[factor], errors="coerce")
        for q in qs:
            thr = expanding_threshold(s, q)
            mask = (s >= thr) & thr.notna()
            out[RuleDef(f"skip_{factor}_ge_expq{int(q*100)}", "market_skip_single", "skip", f"{factor} >= expanding q{int(q*100)}")] = mask

    pair_defs = [
        ("mkt_ma5_followthrough_v3", 0.6, "<=", "mkt_dispersion_shock_v2", 0.7, "<="),
        ("market_trend_strength", 0.5, "<=", "mkt_regime_trend_pressure_v2", 0.5, "<="),
        ("market_trend_strength", 0.5, "<=", "mkt_sector_dispersion", 0.6, ">="),
        ("sector_leadership_persistence_v3", 0.4, "<=", "strong_stock_volume_confirmation_v3", 0.4, "<="),
        ("mkt_emotion_stability_v2", 0.4, "<=", "mkt_sector_dispersion", 0.6, ">="),
    ]
    for f1, q1, op1, f2, q2, op2 in pair_defs:
        s1 = pd.to_numeric(df[f1], errors="coerce")
        s2 = pd.to_numeric(df[f2], errors="coerce")
        t1 = expanding_threshold(s1, q1)
        t2 = expanding_threshold(s2, q2)
        m1 = (s1 <= t1) if op1 == "<=" else (s1 >= t1)
        m2 = (s2 <= t2) if op2 == "<=" else (s2 >= t2)
        mask = m1 & m2 & t1.notna() & t2.notna()
        name = f"skip_{f1}_{'le' if op1=='<=' else 'ge'}q{int(q1*100)}__{f2}_{'le' if op2=='<=' else 'ge'}q{int(q2*100)}"
        desc = f"{f1} {op1} exp q{int(q1*100)} AND {f2} {op2} exp q{int(q2*100)}"
        out[RuleDef(name, "market_skip_pair", "skip", desc)] = mask

    return out


def stock_switch_rules(df: pd.DataFrame) -> dict[RuleDef, pd.Series]:
    out: dict[RuleDef, pd.Series] = {}

    top1_pct_high = {
        "abnormal_turnover": [0.85, 0.90, 0.95],
        "ppo_strength": [0.80, 0.85, 0.90],
        "volume_structure_alpha": [0.80, 0.85, 0.90],
        "sector_turnover_change_5d_20d": [0.75, 0.80, 0.85],
        "severe_abnormal_pressure_10d": [0.85, 0.90, 0.95],
    }
    top1_pct_low = {
        "return_10d_sector_mean": [0.15, 0.20, 0.30],
        "sector_leader_capture_rank": [0.10, 0.15, 0.20],
    }

    for factor, thrs in top1_pct_high.items():
        s = pd.to_numeric(df[f"top1__{factor}__pct"], errors="coerce")
        for thr in thrs:
            mask = s >= thr
            out[RuleDef(f"switch_top1_{factor}_pct_ge_{int(thr*100)}", "stock_switch_top1_pct", "switch", f"top1 {factor} cross-sec pct >= {thr:.2f}")] = mask

    for factor, thrs in top1_pct_low.items():
        s = pd.to_numeric(df[f"top1__{factor}__pct"], errors="coerce")
        for thr in thrs:
            mask = s <= thr
            out[RuleDef(f"switch_top1_{factor}_pct_le_{int(thr*100)}", "stock_switch_top1_pct", "switch", f"top1 {factor} cross-sec pct <= {thr:.2f}")] = mask

    reg_raw = pd.to_numeric(df["top1__severe_abnormal_pressure_10d"], errors="coerce")
    for thr in [0.30, 0.40, 0.50]:
        out[RuleDef(f"switch_top1_reg10_raw_ge_{str(thr).replace('.', 'p')}", "stock_switch_top1_raw", "switch", f"top1 severe_abnormal_pressure_10d >= {thr:.2f}")] = reg_raw >= thr

    diff_pairs = [
        ("abnormal_turnover", "ppo_strength", 0.30, 0.40),
        ("abnormal_turnover", "sector_turnover_change_5d_20d", 0.30, 0.60),
        ("ppo_strength", "sector_turnover_change_5d_20d", 0.40, 0.60),
        ("abnormal_turnover", "volume_structure_alpha", 0.30, 0.40),
    ]
    for f1, f2, q1, q2 in diff_pairs:
        d1 = pd.to_numeric(df[f"top1__{f1}__pct"], errors="coerce") - pd.to_numeric(df[f"top2__{f1}__pct"], errors="coerce")
        d2 = pd.to_numeric(df[f"top1__{f2}__pct"], errors="coerce") - pd.to_numeric(df[f"top2__{f2}__pct"], errors="coerce")
        t1 = expanding_threshold(d1, q1)
        t2 = expanding_threshold(d2, q2)
        mask = (d1 >= t1) & (d2 >= t2) & t1.notna() & t2.notna()
        name = f"switch_diff_{f1}_q{int(q1*100)}__{f2}_q{int(q2*100)}"
        desc = f"(top1-top2){f1} pct-diff >= exp q{int(q1*100)} AND {f2} pct-diff >= exp q{int(q2*100)}"
        out[RuleDef(name, "stock_switch_diff_pair", "switch", desc)] = mask

    return out


def apply_scheme(df: pd.DataFrame, base_rank: pd.Series, base_ret: pd.Series, rule: RuleDef, mask: pd.Series) -> tuple[pd.Series, pd.Series]:
    if rule.mode == "skip":
        rank = pd.Series(np.where(mask, np.nan, base_rank), index=df.index)
        ret = pd.Series(np.where(mask, np.nan, base_ret), index=df.index)
        return rank, pd.to_numeric(ret, errors="coerce")

    rank = pd.Series(np.where(mask, 2, base_rank), index=df.index)
    ret = pd.Series(np.where(mask, df["top2_ret"], base_ret), index=df.index)
    return rank, pd.to_numeric(ret, errors="coerce")


def apply_buy_filter(df: pd.DataFrame, rank: pd.Series, ret: pd.Series, gap_map: dict[tuple[pd.Timestamp, str], float]) -> tuple[pd.Series, pd.Series]:
    chosen_code = pd.Series(np.where(rank == 1, df["top1_code"], np.where(rank == 2, df["top2_code"], None)), index=df.index)
    buy_skip = []
    filtered_ret = []
    for dt, code, r in zip(df["date"], chosen_code, ret):
        if pd.isna(r) or code is None or pd.isna(code):
            buy_skip.append(False)
            filtered_ret.append(np.nan)
            continue
        gap = gap_map.get((pd.Timestamp(dt), str(code)))
        should_skip = gap is not None and (gap < -0.03 or gap > 0.05)
        buy_skip.append(bool(should_skip))
        filtered_ret.append(np.nan if should_skip else r)
    return pd.Series(filtered_ret, index=df.index), pd.Series(buy_skip, index=df.index)


def run_for_json(json_path: Path, factor_df: pd.DataFrame) -> pd.DataFrame:
    df = build_base_df(json_path, factor_df)
    gap_map = build_buy_gap_map(df)
    base_rank, base_ret = current_switch(df)
    label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"

    rows: list[dict] = []
    baseline_defs = [
        RuleDef("baseline_top1", "baseline", "switch", "always top1"),
        RuleDef("baseline_top2", "baseline", "switch", "always top2"),
        RuleDef("current_switch_mkt_exp_q30_reg10_raw_0p40", "baseline", "switch", "current default switch"),
    ]
    baseline_payload = [
        (pd.Series(1, index=df.index), pd.to_numeric(df["top1_ret"], errors="coerce")),
        (pd.Series(2, index=df.index), pd.to_numeric(df["top2_ret"], errors="coerce")),
        (base_rank, base_ret),
    ]

    for rule, (rank, ret) in zip(baseline_defs, baseline_payload):
        rows.append(summarize_scheme(rule.name, ret, ret.notna(), pd.Series(False, index=df.index), json_path.name, label_version, "no_buy_filter", rule.family, rule.desc))
        filt_ret, buy_skip = apply_buy_filter(df, rank, ret, gap_map)
        rows.append(summarize_scheme(rule.name, filt_ret, filt_ret.notna(), buy_skip, json_path.name, label_version, "buy_filter_open_gap", rule.family, rule.desc))

    market_rules = market_skip_rules(df)
    stock_rules = stock_switch_rules(df)

    for rule_map in [market_rules, stock_rules]:
        for rule, mask in rule_map.items():
            rank, ret = apply_scheme(df, base_rank, base_ret, rule, mask)
            rows.append(summarize_scheme(rule.name, ret, ret.notna(), mask if rule.mode == "skip" else pd.Series(False, index=df.index), json_path.name, label_version, "no_buy_filter", rule.family, rule.desc))
            filt_ret, buy_skip = apply_buy_filter(df, rank, ret, gap_map)
            rows.append(summarize_scheme(rule.name, filt_ret, filt_ret.notna(), buy_skip if rule.mode == "switch" else (buy_skip | mask), json_path.name, label_version, "buy_filter_open_gap", rule.family, rule.desc))

    combo_rows: list[dict] = []
    for skip_rule, skip_mask in market_rules.items():
        for switch_rule, switch_mask in stock_rules.items():
            combined_name = f"{skip_rule.name}__{switch_rule.name}"
            combined_desc = f"{skip_rule.desc}; else {switch_rule.desc} -> top2"
            rank, ret = apply_scheme(df, base_rank, base_ret, switch_rule, switch_mask)
            rank = pd.Series(np.where(skip_mask, np.nan, rank), index=df.index)
            ret = pd.Series(np.where(skip_mask, np.nan, ret), index=df.index)
            combo_rows.append(summarize_scheme(combined_name, ret, ret.notna(), skip_mask, json_path.name, label_version, "no_buy_filter", "combo_skip_switch", combined_desc))
            filt_ret, buy_skip = apply_buy_filter(df, rank, ret, gap_map)
            combo_rows.append(summarize_scheme(combined_name, filt_ret, filt_ret.notna(), buy_skip | skip_mask, json_path.name, label_version, "buy_filter_open_gap", "combo_skip_switch", combined_desc))

    return pd.concat([pd.DataFrame(rows), pd.DataFrame(combo_rows)], ignore_index=True)


def rank_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["scheme", "family", "desc", "filter_mode"], as_index=False)
        .agg(
            avg_mean_ret=("mean_ret", "mean"),
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mdd=("max_drawdown", "mean"),
            worst_mdd=("max_drawdown", "min"),
            avg_win_rate=("win_rate", "mean"),
            avg_pass_rate=("pass_rate", "mean"),
        )
    )
    return agg.sort_values(["filter_mode", "avg_cum_ret"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    factor_df = prepare_factor_df(load_factor_df())
    per_json = pd.concat([run_for_json(p, factor_df) for p in JSON_PATHS], ignore_index=True)
    agg = rank_aggregate(per_json)

    per_json_path = RESULTS_DIR / "meaningful_rule_search_per_json_20260417.csv"
    agg_path = RESULTS_DIR / "meaningful_rule_search_aggregate_20260417.csv"
    per_json.to_csv(per_json_path, index=False, encoding="utf-8-sig")
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    for mode in ["no_buy_filter", "buy_filter_open_gap"]:
        print(f"\n=== {mode} top 20 ===")
        sub = agg[agg["filter_mode"] == mode].head(20)
        print(sub[["scheme", "family", "avg_cum_ret", "worst_cum_ret", "avg_mean_ret", "avg_mdd", "worst_mdd", "avg_pass_rate"]].to_string(index=False))

    print(f"\nSaved: {per_json_path}")
    print(f"Saved: {agg_path}")


if __name__ == "__main__":
    main()
