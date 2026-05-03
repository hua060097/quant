from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
FACTOR_PATH = ROOT / "factor_lab_data" / "factors.parquet"
MKT = "mkt_ma5_followthrough_v3"
REG = "severe_abnormal_pressure_10d"
TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")
ADDED_START = pd.Timestamp("2026-03-24")
ADDED_END = pd.Timestamp("2026-04-07")
RAW_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
PCT_THRESHOLDS = [0.90, 0.95, 0.98, 0.99]
MKT_STRONG_QS = [0.4, 0.5, 0.6]


def summarize(s: pd.Series) -> dict[str, float]:
    s = pd.Series(s).dropna().astype(float)
    if s.empty:
        return {"days": 0, "mean": float("nan"), "cum": float("nan"), "mdd": float("nan"), "win": float("nan")}
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


def load_top_json(path: Path) -> pd.DataFrame:
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


def load_factor_panel(path: Path) -> pd.DataFrame:
    cols = ["date", "stock_code", MKT, REG]
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_factors(top_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    daily_mkt = factor_df.groupby("date", as_index=False)[[MKT]].first()
    top1_factor = factor_df[["date", "stock_code", REG]].rename(columns={"stock_code": "top1_code", REG: "top1_reg"})
    out = top_df.merge(daily_mkt, on="date", how="left")
    out = out.merge(top1_factor, on=["date", "top1_code"], how="left")

    rank_df = factor_df[["date", "stock_code", REG]].copy()
    rank_df["reg_pct_rank"] = rank_df.groupby("date")[REG].rank(pct=True, method="average")
    top1_rank = rank_df[["date", "stock_code", "reg_pct_rank"]].rename(columns={"stock_code": "top1_code", "reg_pct_rank": "top1_reg_pct"})
    out = out.merge(top1_rank, on=["date", "top1_code"], how="left")
    return out.sort_values("date").reset_index(drop=True)


def add_partitions(rows: list[dict], df: pd.DataFrame, ret_col: str, scheme: str) -> None:
    parts = {
        "full": df,
        "first_half": df.iloc[: len(df) // 2].copy(),
        "second_half": df.iloc[len(df) // 2 :].copy(),
        "added_10d": df[(df["date"] >= ADDED_START) & (df["date"] <= ADDED_END)].copy(),
    }
    for name, part in parts.items():
        rows.append({"scheme": scheme, "partition": name, **summarize(part[ret_col])})


def build_combo_returns(df: pd.DataFrame, high_risk: pd.Series, mode: str) -> pd.Series:
    choose_mkt_top1 = df["mkt_choose_top1"]
    if mode == "highrisk_top2":
        return pd.Series(
            np.where(choose_mkt_top1, np.where(high_risk, df["top2_ret"], df["top1_ret"]), df["top2_ret"]),
            index=df.index,
            dtype="float64",
        )
    return pd.Series(
        np.where(choose_mkt_top1, np.where(high_risk, 0.0, df["top1_ret"]), df["top2_ret"]),
        index=df.index,
        dtype="float64",
    )


def main() -> None:
    top_df = load_top_json(JSON_PATH)
    factor_df = load_factor_panel(FACTOR_PATH)
    df = attach_factors(top_df, factor_df)

    train_daily_mkt = (
        factor_df.groupby("date", as_index=False)[[MKT]]
        .first()
        .loc[lambda x: (x["date"] >= TRAIN_START) & (x["date"] <= TRAIN_END), MKT]
        .dropna()
    )
    mkt_q30 = float(train_daily_mkt.quantile(0.3))
    df["mkt_thr_q30"] = build_expanding_threshold(df[MKT], 0.3, mkt_q30)
    df["mkt_choose_top1"] = df[MKT] >= df["mkt_thr_q30"]
    df["baseline_ret"] = np.where(df["mkt_choose_top1"], df["top1_ret"], df["top2_ret"])

    rows = []
    partition_rows = []

    # baseline
    rows.append({"scheme": "mkt_exp_q30_baseline", **summarize(df["baseline_ret"])})

    # raw threshold rules
    for thr in RAW_THRESHOLDS:
        high_risk = df["top1_reg"] >= thr
        for mode in ["highrisk_top2", "highrisk_cash"]:
            scheme = f"raw_{thr:.1f}_{mode}"
            ret = build_combo_returns(df, high_risk, mode)
            rows.append(
                {
                    "scheme": scheme,
                    "family": "raw",
                    "threshold": thr,
                    "mode": mode,
                    **summarize(ret),
                    "high_risk_days": int((df["mkt_choose_top1"] & high_risk).sum()),
                    "top1_days": int((df["mkt_choose_top1"] & ~high_risk).sum()),
                    "top2_days": int((~df["mkt_choose_top1"]).sum() + ((mode == "highrisk_top2") * (df["mkt_choose_top1"] & high_risk).sum())),
                    "cash_days": int(((mode == "highrisk_cash") * (df["mkt_choose_top1"] & high_risk).sum())),
                }
            )

    # cross-sectional pct rules
    for thr in PCT_THRESHOLDS:
        high_risk = df["top1_reg_pct"] >= thr
        for mode in ["highrisk_top2", "highrisk_cash"]:
            scheme = f"pct_{thr:.2f}_{mode}"
            ret = build_combo_returns(df, high_risk, mode)
            rows.append(
                {
                    "scheme": scheme,
                    "family": "pct_rank",
                    "threshold": thr,
                    "mode": mode,
                    **summarize(ret),
                    "high_risk_days": int((df["mkt_choose_top1"] & high_risk).sum()),
                    "top1_days": int((df["mkt_choose_top1"] & ~high_risk).sum()),
                    "top2_days": int((~df["mkt_choose_top1"]).sum() + ((mode == "highrisk_top2") * (df["mkt_choose_top1"] & high_risk).sum())),
                    "cash_days": int(((mode == "highrisk_cash") * (df["mkt_choose_top1"] & high_risk).sum())),
                }
            )

    # regime-specific raw thresholds based on mkt strength
    for mq in MKT_STRONG_QS:
        fallback = float(train_daily_mkt.quantile(mq))
        strong_thr = build_expanding_threshold(df[MKT], mq, fallback)
        market_strong = df[MKT] >= strong_thr
        for weak_thr in [0.6, 0.8, 1.0]:
            for strong_thr_val in [0.6, 0.8, 1.0]:
                high_risk = np.where(market_strong, df["top1_reg"] >= strong_thr_val, df["top1_reg"] >= weak_thr)
                high_risk = pd.Series(high_risk, index=df.index)
                scheme = f"regime_mq{int(mq*100)}_weak{weak_thr:.1f}_strong{strong_thr_val:.1f}_top2"
                ret = build_combo_returns(df, high_risk, "highrisk_top2")
                rows.append(
                    {
                        "scheme": scheme,
                        "family": "regime_raw",
                        "threshold": np.nan,
                        "mode": "highrisk_top2",
                        "mkt_regime_q": mq,
                        "weak_thr": weak_thr,
                        "strong_thr": strong_thr_val,
                        **summarize(ret),
                        "high_risk_days": int((df["mkt_choose_top1"] & high_risk).sum()),
                        "top1_days": int((df["mkt_choose_top1"] & ~high_risk).sum()),
                        "top2_days": int((~df["mkt_choose_top1"]).sum() + (df["mkt_choose_top1"] & high_risk).sum()),
                        "cash_days": 0,
                    }
                )

    result_df = pd.DataFrame(rows).sort_values(["cum", "mean"], ascending=False).reset_index(drop=True)

    # materialize best by family + baseline
    chosen = [result_df[result_df["scheme"] == "mkt_exp_q30_baseline"].iloc[0]]
    for family in ["raw", "pct_rank", "regime_raw"]:
        fam = result_df[result_df["family"] == family]
        if not fam.empty:
            chosen.append(fam.iloc[0])

    for row in chosen:
        scheme = row["scheme"]
        if scheme == "mkt_exp_q30_baseline":
            df[scheme] = df["baseline_ret"]
        elif row.get("family") == "raw":
            high_risk = df["top1_reg"] >= float(row["threshold"])
            df[scheme] = build_combo_returns(df, high_risk, str(row["mode"]))
        elif row.get("family") == "pct_rank":
            high_risk = df["top1_reg_pct"] >= float(row["threshold"])
            df[scheme] = build_combo_returns(df, high_risk, str(row["mode"]))
        else:
            mq = float(row["mkt_regime_q"])
            fallback = float(train_daily_mkt.quantile(mq))
            strong_thr = build_expanding_threshold(df[MKT], mq, fallback)
            market_strong = df[MKT] >= strong_thr
            high_risk = np.where(market_strong, df["top1_reg"] >= float(row["strong_thr"]), df["top1_reg"] >= float(row["weak_thr"]))
            df[scheme] = build_combo_returns(df, pd.Series(high_risk, index=df.index), "highrisk_top2")
        add_partitions(partition_rows, df, scheme, scheme)

    partition_df = pd.DataFrame(partition_rows)

    out_dir = ROOT / "results"
    stem = JSON_PATH.stem
    result_path = out_dir / f"{stem}_regulatory_realistic_thresholds_scan.csv"
    partition_path = out_dir / f"{stem}_regulatory_realistic_thresholds_partition.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    partition_df.to_csv(partition_path, index=False, encoding="utf-8-sig")

    print("Top rules")
    print(result_df.head(30).to_string(index=False))
    print()
    print("Best by family partitions")
    print(partition_df.to_string(index=False))
    print()
    print(f"Saved: {result_path}")
    print(f"Saved: {partition_path}")


if __name__ == "__main__":
    main()
