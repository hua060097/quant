from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
DAILY_DIR = ROOT / ".." / "local_data" / "features" / "daily"
MKT = "mkt_ma5_followthrough_v3"
NH = "market_60d_new_high_ratio"
SECTOR = "sector_ma5_followthrough_v3"
REG = "severe_abnormal_pressure_10d"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")


def summarize(series: pd.Series) -> dict[str, float]:
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return {"days": 0, "mean": np.nan, "cum": np.nan, "mdd": np.nan, "win": np.nan}
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "days": int(len(s)),
        "mean": float(s.mean()),
        "cum": float(eq.iloc[-1] - 1.0),
        "mdd": float(dd.min()),
        "win": float((s > 0).mean()),
    }


def build_expanding_threshold(series: pd.Series, q: float, fallback: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def build_rolling_threshold(series: pd.Series, q: float, window: int, fallback: float) -> pd.Series:
    vals = []
    min_obs = min(window, 20)
    for i in range(len(series)):
        hist = series.iloc[max(0, i - window):i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= min_obs else fallback)
    return pd.Series(vals, index=series.index, dtype="float64")


def load_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in sorted(raw.items()):
        row = {"date": pd.to_datetime(ds)}
        for k in range(1, 6):
            item = daily.get(f"top{k}", {})
            row[f"top{k}_code"] = item.get("stock_code")
            row[f"top{k}_ret"] = float(item.get("real_return", np.nan))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factors(path: Path) -> pd.DataFrame:
    cols = ["date", "stock_code", MKT, NH, SECTOR, REG]
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_factors(top_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    out = top_df.copy()
    daily_mkt = factor_df.groupby("date", as_index=False)[[MKT, NH]].first()
    out = out.merge(daily_mkt, on="date", how="left")
    for rank in [1, 2]:
        part = factor_df[["date", "stock_code", SECTOR, REG]].rename(
            columns={
                "stock_code": f"top{rank}_code",
                SECTOR: f"top{rank}_sector_ma5",
                REG: f"top{rank}_reg",
            }
        )
        out = out.merge(part, on=["date", f"top{rank}_code"], how="left")

    rank_df = factor_df[["date", "stock_code", REG]].copy()
    rank_df["reg_pct_rank"] = rank_df.groupby("date")[REG].rank(pct=True, method="average")
    for rank in [1, 2]:
        part = rank_df[["date", "stock_code", "reg_pct_rank"]].rename(
            columns={"stock_code": f"top{rank}_code", "reg_pct_rank": f"top{rank}_reg_pct"}
        )
        out = out.merge(part, on=["date", f"top{rank}_code"], how="left")
    return out


def load_buy_filter_map(dates: pd.Series, codes: pd.Series) -> dict[tuple[pd.Timestamp, str], float]:
    use_dates = sorted(pd.to_datetime(dates).dt.strftime("%Y%m%d").unique().tolist())
    file_map = {p.stem: p for p in DAILY_DIR.glob("*.parquet")}
    date_to_next = {use_dates[i]: use_dates[i + 1] for i in range(len(use_dates) - 1)}
    close_maps: dict[str, dict[str, float]] = {}
    open_maps: dict[str, dict[str, float]] = {}
    needed_dates = sorted(set(use_dates) | set(date_to_next.values()))
    needed_codes = set(codes.dropna().astype(str).tolist())

    for ds in needed_dates:
        p = file_map.get(ds)
        if p is None:
            continue
        tmp = pd.read_parquet(p, columns=["stock_code", "open", "close"])
        tmp = tmp[tmp["stock_code"].isin(needed_codes)]
        close_maps[ds] = dict(zip(tmp["stock_code"], pd.to_numeric(tmp["close"], errors="coerce")))
        open_maps[ds] = dict(zip(tmp["stock_code"], pd.to_numeric(tmp["open"], errors="coerce")))

    gap_map = {}
    for ds in use_dates[:-1]:
        next_ds = date_to_next.get(ds)
        if next_ds is None:
            continue
        close_dict = close_maps.get(ds, {})
        open_dict = open_maps.get(next_ds, {})
        for code in needed_codes:
            c = close_dict.get(code)
            o = open_dict.get(code)
            if c is None or o is None or pd.isna(c) or pd.isna(o) or abs(c) < 1e-9:
                continue
            gap_map[(pd.Timestamp(ds), code)] = float((o - c) / c)
    return gap_map


def apply_buy_filter(df: pd.DataFrame, scheme: str, selected_rank: pd.Series, selected_ret: pd.Series, gap_map: dict[tuple[pd.Timestamp, str], float]) -> pd.DataFrame:
    out = pd.DataFrame({"date": df["date"], "scheme": scheme, "selected_rank": selected_rank, "raw_ret": selected_ret})
    out["selected_code"] = np.where(out["selected_rank"] == 1, df["top1_code"], np.where(out["selected_rank"] == 2, df["top2_code"], None))
    gaps = []
    skipped = []
    final_ret = []
    for _, row in out.iterrows():
        code = row["selected_code"]
        rank = row["selected_rank"]
        if pd.isna(rank) or code is None:
            gaps.append(np.nan)
            skipped.append(False)
            final_ret.append(np.nan)
            continue
        gap = gap_map.get((row["date"], code), np.nan)
        bad = pd.notna(gap) and (gap < -0.03 or gap > 0.05)
        gaps.append(gap)
        skipped.append(bool(bad))
        final_ret.append(np.nan if bad else row["raw_ret"])
    out["gap"] = gaps
    out["skipped"] = skipped
    out["final_ret"] = final_ret
    return out


def add_partition_rows(rows: list[dict], trade_df: pd.DataFrame) -> None:
    parts = {
        "full": trade_df,
        "first_half": trade_df.iloc[: len(trade_df) // 2].copy(),
        "second_half": trade_df.iloc[len(trade_df) // 2 :].copy(),
        "added_10d": trade_df[(trade_df["date"] >= ADDED_START) & (trade_df["date"] <= ADDED_END)].copy(),
    }
    for name, part in parts.items():
        st = summarize(part["final_ret"])
        rows.append(
            {
                "scheme": str(part["scheme"].iloc[0]) if len(part) else "",
                "partition": name,
                **st,
                "trade_days": int(part["final_ret"].notna().sum()),
                "skip_days": int(part["skipped"].sum()),
                "skip_rate": float(part["skipped"].mean()) if len(part) else np.nan,
            }
        )


def main() -> None:
    top_df = load_json(JSON_PATH)
    factor_df = load_factors(FACTOR_PATH)
    df = attach_factors(top_df, factor_df)

    train_daily = factor_df.groupby("date", as_index=False)[[MKT, NH]].first()
    train_daily = train_daily[(train_daily["date"] >= TRAIN_START) & (train_daily["date"] <= TRAIN_END)]
    mkt_q40 = float(train_daily[MKT].quantile(0.4))
    nh_q70 = float(train_daily[NH].quantile(0.7))
    mkt_q30_fallback = float(train_daily[MKT].quantile(0.3))
    df["mkt_q30_thr"] = build_expanding_threshold(df[MKT], 0.3, mkt_q30_fallback)

    sector_fallback = float(df["top1_sector_ma5"].dropna().quantile(0.5))
    df["sector_roll10_q50_thr"] = build_rolling_threshold(df["top1_sector_ma5"], 0.5, 10, sector_fallback)

    choose_mkt_top1 = df[MKT] >= df["mkt_q30_thr"]
    top1_high98 = df["top1_reg_pct"] >= 0.98
    top1_low10 = df["top1_reg_pct"] <= 0.10
    top2_high98 = df["top2_reg_pct"] >= 0.98

    scheme_defs: list[tuple[str, pd.Series, pd.Series]] = []
    scheme_defs.append(("mkt_q40", pd.Series(np.where(df[MKT] >= mkt_q40, 1, 2), index=df.index), pd.Series(np.where(df[MKT] >= mkt_q40, df["top1_ret"], df["top2_ret"]), index=df.index)))
    scheme_defs.append(("mkt_exp_q30", pd.Series(np.where(choose_mkt_top1, 1, 2), index=df.index), pd.Series(np.where(choose_mkt_top1, df["top1_ret"], df["top2_ret"]), index=df.index)))
    scheme_defs.append(("mkt_q40_plus_new_high_ratio_q70", pd.Series(np.where((df[MKT] >= mkt_q40) & (df[NH] >= nh_q70), 1, 2), index=df.index), pd.Series(np.where((df[MKT] >= mkt_q40) & (df[NH] >= nh_q70), df["top1_ret"], df["top2_ret"]), index=df.index)))
    scheme_defs.append(("sector_ma5_followthrough_v3_roll10_q50", pd.Series(np.where(df["top1_sector_ma5"] >= df["sector_roll10_q50_thr"], 1, 2), index=df.index), pd.Series(np.where(df["top1_sector_ma5"] >= df["sector_roll10_q50_thr"], df["top1_ret"], df["top2_ret"]), index=df.index)))

    scheme_defs.append(("mkt_exp_q30_top1_hi98_to_top2", pd.Series(np.where(choose_mkt_top1 & top1_high98, 2, np.where(choose_mkt_top1, 1, 2)), index=df.index), pd.Series(np.where(choose_mkt_top1 & top1_high98, df["top2_ret"], np.where(choose_mkt_top1, df["top1_ret"], df["top2_ret"])), index=df.index)))

    scheme_defs.append(("mkt_exp_q30_top1_hi98_to_cash", pd.Series(np.where(choose_mkt_top1 & top1_high98, np.nan, np.where(choose_mkt_top1, 1, 2)), index=df.index), pd.Series(np.where(choose_mkt_top1 & top1_high98, np.nan, np.where(choose_mkt_top1, df["top1_ret"], df["top2_ret"])), index=df.index)))

    scheme_defs.append(("mkt_exp_q30_top1_hi98_to_top2_else_top2_lo10_to_top1", pd.Series(np.where(choose_mkt_top1 & top1_high98, 2, np.where(choose_mkt_top1, 1, np.where(top1_low10, 1, 2))), index=df.index), pd.Series(np.where(choose_mkt_top1 & top1_high98, df["top2_ret"], np.where(choose_mkt_top1, df["top1_ret"], np.where(top1_low10, df["top1_ret"], df["top2_ret"]))), index=df.index)))

    scheme_defs.append(("mkt_exp_q30_top1_hi98_and_top2_not_hi98_to_top2", pd.Series(np.where(choose_mkt_top1 & top1_high98 & ~top2_high98, 2, np.where(choose_mkt_top1, 1, 2)), index=df.index), pd.Series(np.where(choose_mkt_top1 & top1_high98 & ~top2_high98, df["top2_ret"], np.where(choose_mkt_top1, df["top1_ret"], df["top2_ret"])), index=df.index)))

    scheme_defs.append(("mkt_exp_q30_selected_stock_hi98_to_cash", pd.Series(np.where(choose_mkt_top1, np.where(top1_high98, np.nan, 1), np.where(top2_high98, np.nan, 2)), index=df.index), pd.Series(np.where(choose_mkt_top1, np.where(top1_high98, np.nan, df["top1_ret"]), np.where(top2_high98, np.nan, df["top2_ret"])), index=df.index)))

    scheme_defs.append(("mkt_exp_q30_selected_stock_hi98_to_fallback_top1", pd.Series(np.where(choose_mkt_top1, np.where(top1_high98, 2, 1), np.where(top2_high98, 1, 2)), index=df.index), pd.Series(np.where(choose_mkt_top1, np.where(top1_high98, df["top2_ret"], df["top1_ret"]), np.where(top2_high98, df["top1_ret"], df["top2_ret"])), index=df.index)))

    all_codes = pd.concat([df["top1_code"], df["top2_code"]], ignore_index=True)
    gap_map = load_buy_filter_map(df["date"], all_codes)

    summary_rows = []
    partition_rows = []
    detail_frames = []
    for name, ranks, rets in scheme_defs:
        trade_df = apply_buy_filter(df, name, ranks, rets, gap_map)
        detail_frames.append(trade_df)
        add_partition_rows(partition_rows, trade_df)
        st = summarize(trade_df["final_ret"])
        summary_rows.append(
            {
                "scheme": name,
                **st,
                "trade_days": int(trade_df["final_ret"].notna().sum()),
                "skip_days": int(trade_df["skipped"].sum()),
                "skip_rate": float(trade_df["skipped"].mean()),
                "cash_days": int(trade_df["selected_rank"].isna().sum()),
                "top1_selected_days": int((trade_df["selected_rank"] == 1).sum()),
                "top2_selected_days": int((trade_df["selected_rank"] == 2).sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    partition_df = pd.DataFrame(partition_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    out_dir = ROOT / "results"
    stem = JSON_PATH.stem
    summary_path = out_dir / f"{stem}_pct98_global_compare_summary.csv"
    partition_path = out_dir / f"{stem}_pct98_global_compare_partition.csv"
    detail_path = out_dir / f"{stem}_pct98_global_compare_detail.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(partition_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(summary_df.to_string(index=False))
    print()
    print(partition_df.to_string(index=False))
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {partition_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
