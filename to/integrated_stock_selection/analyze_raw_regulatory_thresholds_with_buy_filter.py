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
REG10 = "severe_abnormal_pressure_10d"
ABN3 = "abnormal_deviation_3d_ratio"
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


def load_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in sorted(raw.items()):
        rows.append(
            {
                "date": pd.to_datetime(ds),
                "top1_code": daily.get("top1", {}).get("stock_code"),
                "top2_code": daily.get("top2", {}).get("stock_code"),
                "top1_ret": float(daily.get("top1", {}).get("real_return", np.nan)),
                "top2_ret": float(daily.get("top2", {}).get("real_return", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_factors(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "stock_code", MKT, REG10, ABN3])
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach(top_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    daily_mkt = factor_df.groupby("date", as_index=False)[[MKT]].first()
    out = top_df.merge(daily_mkt, on="date", how="left")
    for factor in [REG10, ABN3]:
        part = factor_df[["date", "stock_code", factor]].rename(columns={"stock_code": "top1_code", factor: f"top1_{factor}"})
        out = out.merge(part, on=["date", "top1_code"], how="left")
    return out


def load_buy_filter_map(dates: pd.Series, codes: pd.Series) -> dict[tuple[pd.Timestamp, str], float]:
    use_dates = sorted(pd.to_datetime(dates).dt.strftime("%Y%m%d").unique().tolist())
    file_map = {p.stem: p for p in DAILY_DIR.glob("*.parquet")}
    date_to_next = {use_dates[i]: use_dates[i + 1] for i in range(len(use_dates) - 1)}
    needed_codes = set(codes.dropna().astype(str))
    close_maps: dict[str, dict[str, float]] = {}
    open_maps: dict[str, dict[str, float]] = {}
    for ds in sorted(set(use_dates) | set(date_to_next.values())):
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
        for code in needed_codes:
            c = close_maps.get(ds, {}).get(code)
            o = open_maps.get(next_ds, {}).get(code)
            if c is None or o is None or pd.isna(c) or pd.isna(o) or abs(c) < 1e-9:
                continue
            gap_map[(pd.Timestamp(ds), code)] = float((o - c) / c)
    return gap_map


def apply_buy_filter(df: pd.DataFrame, scheme: str, selected_rank: pd.Series, selected_ret: pd.Series, gap_map: dict[tuple[pd.Timestamp, str], float]) -> pd.DataFrame:
    out = pd.DataFrame({"date": df["date"], "scheme": scheme, "selected_rank": selected_rank, "raw_ret": selected_ret})
    out["selected_code"] = np.where(out["selected_rank"] == 1, df["top1_code"], np.where(out["selected_rank"] == 2, df["top2_code"], None))
    final_ret = []
    skipped = []
    for _, row in out.iterrows():
        rank = row["selected_rank"]
        code = row["selected_code"]
        if pd.isna(rank) or code is None:
            final_ret.append(np.nan)
            skipped.append(False)
            continue
        gap = gap_map.get((row["date"], code), np.nan)
        bad = pd.notna(gap) and (gap < -0.03 or gap > 0.05)
        final_ret.append(np.nan if bad else row["raw_ret"])
        skipped.append(bool(bad))
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
        rows.append({"scheme": str(part["scheme"].iloc[0]), "partition": name, **st, "trade_days": int(part["final_ret"].notna().sum()), "skip_days": int(part["skipped"].sum()), "skip_rate": float(part["skipped"].mean())})


def main() -> None:
    top_df = load_json(JSON_PATH)
    factor_df = load_factors(FACTOR_PATH)
    df = attach(top_df, factor_df)
    train_daily = factor_df.groupby("date", as_index=False)[[MKT]].first()
    train_daily = train_daily[(train_daily["date"] >= TRAIN_START) & (train_daily["date"] <= TRAIN_END)]
    mkt_q30 = float(train_daily[MKT].quantile(0.3))
    df["mkt_q30_thr"] = build_expanding_threshold(df[MKT], 0.3, mkt_q30)
    choose_top1 = df[MKT] >= df["mkt_q30_thr"]

    scheme_defs: list[tuple[str, pd.Series, pd.Series]] = [
        ("mkt_exp_q30", pd.Series(np.where(choose_top1, 1, 2), index=df.index), pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index))
    ]

    for thr in [0.3, 0.35, 0.4, 0.45, 0.5]:
        high = df[f"top1_{REG10}"] >= thr
        name = f"mkt_exp_q30_reg10_raw_{thr:.2f}_to_top2"
        ranks = pd.Series(np.where(choose_top1 & high, 2, np.where(choose_top1, 1, 2)), index=df.index)
        rets = pd.Series(np.where(choose_top1 & high, df["top2_ret"], np.where(choose_top1, df["top1_ret"], df["top2_ret"])), index=df.index)
        scheme_defs.append((name, ranks, rets))

    for thr in [0.6, 0.7, 0.8, 0.9, 1.0]:
        high = df[f"top1_{ABN3}"] >= thr
        name = f"mkt_exp_q30_abn3_raw_{thr:.2f}_to_top2"
        ranks = pd.Series(np.where(choose_top1 & high, 2, np.where(choose_top1, 1, 2)), index=df.index)
        rets = pd.Series(np.where(choose_top1 & high, df["top2_ret"], np.where(choose_top1, df["top1_ret"], df["top2_ret"])), index=df.index)
        scheme_defs.append((name, ranks, rets))

    gap_map = load_buy_filter_map(df["date"], pd.concat([df["top1_code"], df["top2_code"]], ignore_index=True))
    summary_rows = []
    partition_rows = []
    for name, ranks, rets in scheme_defs:
        trade_df = apply_buy_filter(df, name, ranks, rets, gap_map)
        st = summarize(trade_df["final_ret"])
        summary_rows.append({"scheme": name, **st, "trade_days": int(trade_df["final_ret"].notna().sum()), "skip_days": int(trade_df["skipped"].sum()), "skip_rate": float(trade_df["skipped"].mean())})
        add_partition_rows(partition_rows, trade_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)
    partition_df = pd.DataFrame(partition_rows)
    out_dir = ROOT / "results"
    stem = JSON_PATH.stem
    summary_path = out_dir / f"{stem}_raw_regulatory_thresholds_with_buy_filter_summary.csv"
    partition_path = out_dir / f"{stem}_raw_regulatory_thresholds_with_buy_filter_partition.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(partition_path, index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    print()
    print(partition_df.to_string(index=False))
    print()
    print(f"Saved: {summary_path}")
    print(f"Saved: {partition_path}")


if __name__ == "__main__":
    main()
