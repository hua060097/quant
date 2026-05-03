#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_raw_pct_hybrid_and_failures as rawmod


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SW2_MAP_PATH = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\sw_sector2\sw_sector.json")
DAILY_FEATURE_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data\features\daily")
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"


FOCUS_FEATURES = [
    "cpv_factor",
    "shadow_dominance",
    "money_flow_pressure",
    "intraday_reversal",
    "open_to_high_ratio",
    "amihud_illiquidity",
    "abnormal_turnover",
    "ppo_strength",
    "volume_structure_alpha",
    "return_10d_sector_mean",
]


def load_sw2_mapping() -> dict[str, str]:
    return json.loads(SW2_MAP_PATH.read_text(encoding="utf-8"))


def load_daily_pct_for_dates(dates: list[str], sw2_map: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in sorted(set(dates)):
        path = DAILY_FEATURE_DIR / f"{trade_date}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["stock_code", "date_str", "pct_chg"])
        df = df.rename(columns={"date_str": "trade_date"})
        df["sw_sector"] = df["stock_code"].map(sw2_map)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stock_code", "trade_date", "pct_chg", "sw_sector"])
    return pd.concat(frames, ignore_index=True)


def build_sector_leader_failure_indicator(loss_df: pd.DataFrame) -> pd.DataFrame:
    sw2_map = load_sw2_mapping()
    dates = pd.to_datetime(loss_df["date"]).dt.strftime("%Y%m%d").tolist()
    unique_dates = sorted(set(dates))

    all_dates_needed = set(unique_dates)
    date_index = pd.Index(unique_dates)
    prev_map = {}
    for i in range(1, len(date_index)):
        prev_map[date_index[i]] = date_index[i - 1]
        all_dates_needed.add(date_index[i - 1])

    daily = load_daily_pct_for_dates(list(all_dates_needed), sw2_map)
    if daily.empty:
        return pd.DataFrame(columns=["date", "stock_code", "sector_top3_prevup_nextdown_count", "sector_top3_prevup_nextdown_ge2"])

    daily["trade_date"] = daily["trade_date"].astype(str)
    daily["pct_chg"] = pd.to_numeric(daily["pct_chg"], errors="coerce")

    # Top3 strongest stocks within each SW2 sector on t day, ranked by t-day pct_chg.
    daily = daily.dropna(subset=["sw_sector", "pct_chg"]).copy()
    daily["rank_in_sector_today"] = daily.groupby(["trade_date", "sw_sector"])["pct_chg"].rank(method="first", ascending=False)
    top3 = daily[daily["rank_in_sector_today"] <= 3].copy()

    prev_pct = daily[["trade_date", "stock_code", "pct_chg"]].rename(columns={"trade_date": "prev_trade_date", "pct_chg": "prev_pct_chg"})
    top3["prev_trade_date"] = top3["trade_date"].map(prev_map)
    top3 = top3.merge(prev_pct, on=["prev_trade_date", "stock_code"], how="left")

    # Leader-failure definition: previous day rallied strongly, current day drops hard.
    top3["leader_failed"] = (top3["prev_pct_chg"] >= 7.0) & (top3["pct_chg"] <= -3.0)
    sector_count = (
        top3.groupby(["trade_date", "sw_sector"], as_index=False)["leader_failed"]
        .sum()
        .rename(columns={"leader_failed": "sector_top3_prevup_nextdown_count"})
    )
    sector_count["sector_top3_prevup_nextdown_ge2"] = sector_count["sector_top3_prevup_nextdown_count"] >= 2

    selected = loss_df[["date", "selected_code"]].drop_duplicates().copy()
    selected["top1_sector"] = selected["selected_code"].map(sw2_map)
    selected["trade_date"] = pd.to_datetime(selected["date"]).dt.strftime("%Y%m%d")
    out = selected.merge(
        sector_count,
        left_on=["trade_date", "top1_sector"],
        right_on=["trade_date", "sw_sector"],
        how="left",
    )
    out["sector_top3_prevup_nextdown_count"] = out["sector_top3_prevup_nextdown_count"].fillna(0).astype(int)
    out["sector_top3_prevup_nextdown_ge2"] = out["sector_top3_prevup_nextdown_ge2"].fillna(False)
    return out[["date", "selected_code", "sector_top3_prevup_nextdown_count", "sector_top3_prevup_nextdown_ge2"]]


def attach_selected_features(loss_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["date", "stock_code"] + FOCUS_FEATURES
    factor_df = pd.read_parquet(FACTOR_PATH, columns=cols).copy()
    factor_df = factor_df.rename(columns={"stock_code": "selected_code"})
    return loss_df.merge(factor_df, on=["date", "selected_code"], how="left")


def summarize_feature_relations(loss_df: pd.DataFrame) -> pd.DataFrame:
    work = loss_df.copy()
    work["bad3"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.03
    work["bad5"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.05
    rows: list[dict] = []
    for bad_col in ["bad3", "bad5"]:
        bad_mask = work[bad_col].fillna(False)
        good_mask = (~bad_mask) & work["filtered_ret"].notna()
        for col in FOCUS_FEATURES:
            s = pd.to_numeric(work[col], errors="coerce")
            bad = s[bad_mask].dropna()
            good = s[good_mask].dropna()
            if len(bad) < 5 or len(good) < 5:
                continue
            pooled = np.sqrt((bad.var(ddof=1) + good.var(ddof=1)) / 2.0)
            effect = (bad.mean() - good.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan
            rows.append({
                "loss_group": bad_col,
                "feature": col,
                "bad_mean": float(bad.mean()),
                "good_mean": float(good.mean()),
                "mean_diff": float(bad.mean() - good.mean()),
                "effect_z": float(effect) if np.isfinite(effect) else np.nan,
                "bad_count": int(len(bad)),
                "good_count": int(len(good)),
            })
    return pd.DataFrame(rows).sort_values(["loss_group", "effect_z"], ascending=[True, False])


def summarize_sector_failure(loss_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    work = loss_df.merge(sector_df, left_on=["date", "selected_code"], right_on=["date", "selected_code"], how="left")
    work["bad3"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.03
    work["bad5"] = pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.05

    rows: list[dict] = []
    for bad_col in ["bad3", "bad5"]:
        bad = work[work[bad_col].fillna(False)]
        good = work[(~work[bad_col].fillna(False)) & work["filtered_ret"].notna()]
        rows.append({
            "loss_group": bad_col,
            "bad_count": int(len(bad)),
            "good_count": int(len(good)),
            "bad_sector_fail_ge2_ratio": float(bad["sector_top3_prevup_nextdown_ge2"].mean()) if len(bad) else np.nan,
            "good_sector_fail_ge2_ratio": float(good["sector_top3_prevup_nextdown_ge2"].mean()) if len(good) else np.nan,
            "bad_sector_fail_count_mean": float(bad["sector_top3_prevup_nextdown_count"].mean()) if len(bad) else np.nan,
            "good_sector_fail_count_mean": float(good["sector_top3_prevup_nextdown_count"].mean()) if len(good) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    _, _, loss_df, _ = rawmod.compare_hybrids()
    loss_df = loss_df[~loss_df["buy_skip"]].copy()
    loss_df["selected_code"] = np.where(
        pd.to_numeric(loss_df["selected_rank"], errors="coerce") == 1,
        loss_df["top1_code"],
        loss_df["top2_code"],
    )
    loss_df = attach_selected_features(loss_df)

    feature_rel = summarize_feature_relations(loss_df)
    sector_fail_df = build_sector_leader_failure_indicator(loss_df)
    sector_summary = summarize_sector_failure(loss_df, sector_fail_df)

    merged_detail = loss_df.merge(sector_fail_df, left_on=["date", "selected_code"], right_on=["date", "selected_code"], how="left")

    feature_rel.to_csv(RESULTS_DIR / "selected_big_loss_focus_feature_relation_20260418.csv", index=False, encoding="utf-8-sig")
    sector_summary.to_csv(RESULTS_DIR / "selected_big_loss_sector_leader_failure_relation_20260418.csv", index=False, encoding="utf-8-sig")
    merged_detail.to_csv(RESULTS_DIR / "selected_big_loss_with_sector_failure_detail_20260418.csv", index=False, encoding="utf-8-sig")

    print("=== Focus Feature Relation ===")
    print(feature_rel.to_string(index=False))
    print("\n=== Sector Leader Failure Relation ===")
    print(sector_summary.to_string(index=False))


if __name__ == "__main__":
    main()
