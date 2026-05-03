from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analyze_meaningful_rule_search as am
import analyze_raw_pct_hybrid_and_failures as rawmod


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MONEYFLOW_DIR = BASE_DIR.parent / "local_data" / "moneyflow_ths"

MONEYFLOW_COLS = [
    "trade_date",
    "ts_code",
    "pct_change",
    "net_amount",
    "net_d5_amount",
    "buy_lg_amount",
    "buy_lg_amount_rate",
    "buy_md_amount_rate",
    "buy_sm_amount_rate",
]

SCHEMES = [
    "current_switch_mkt_exp_q30_reg10_raw_0p40",
    "switch_reg_pct_mktstate_hi95_lo85",
]


def summarize(series: pd.Series) -> tuple[float, float, float, float]:
    traded = pd.to_numeric(series, errors="coerce").dropna()
    if traded.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(traded.mean()),
        float((1 + traded).prod() - 1),
        am.calc_mdd(traded),
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


def build_pct_scheme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pct_thr = rawmod.build_pct_state_threshold(df, 0.95, 0.85)
    top1_pct = pd.to_numeric(df["top1__severe_abnormal_pressure_10d__pct"], errors="coerce")
    choose_top1 = top1_pct < pct_thr
    rank = pd.Series(np.where(choose_top1, 1, 2), index=df.index)
    ret = pd.Series(np.where(choose_top1, df["top1_ret"], df["top2_ret"]), index=df.index)
    return rank, pd.to_numeric(ret, errors="coerce")


def build_base_scheme_map(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    return {
        "current_switch_mkt_exp_q30_reg10_raw_0p40": am.current_switch(df),
        "switch_reg_pct_mktstate_hi95_lo85": build_pct_scheme(df),
    }


def load_moneyflow() -> pd.DataFrame:
    paths = sorted(MONEYFLOW_DIR.glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No moneyflow parquet files found in {MONEYFLOW_DIR}")
    frames = []
    for path in paths:
        df = pd.read_parquet(path, columns=MONEYFLOW_COLS).copy()
        df = df.rename(columns={"trade_date": "date", "ts_code": "selected_code"})
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for col in out.columns:
        if col not in {"date", "selected_code"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["net_amount__pct"] = out.groupby("date")["net_amount"].rank(pct=True, method="average")
    out["net_d5_amount__pct"] = out.groupby("date")["net_d5_amount"].rank(pct=True, method="average")
    out["buy_lg_amount__pct"] = out.groupby("date")["buy_lg_amount"].rank(pct=True, method="average")
    return out


def attach_selected_moneyflow(
    df: pd.DataFrame,
    rank: pd.Series,
    ret: pd.Series,
    gap_map: dict[tuple[pd.Timestamp, str], float],
    moneyflow_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_code = pd.Series(
        np.where(rank == 1, df["top1_code"], np.where(rank == 2, df["top2_code"], None)),
        index=df.index,
    )
    out = df.copy()
    out["selected_rank"] = rank
    out["selected_code"] = selected_code
    out["selected_ret"] = pd.to_numeric(ret, errors="coerce")
    out["filtered_ret"], out["buy_skip"] = am.apply_buy_filter(df, rank, ret, gap_map)
    out = out.merge(moneyflow_df, on=["date", "selected_code"], how="left")
    return out


def build_veto_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    rules: dict[str, pd.Series] = {}

    lg_rate = pd.to_numeric(df["buy_lg_amount_rate"], errors="coerce")
    net_amt = pd.to_numeric(df["net_amount"], errors="coerce")
    net_d5 = pd.to_numeric(df["net_d5_amount"], errors="coerce")
    sm_rate = pd.to_numeric(df["buy_sm_amount_rate"], errors="coerce")
    d5_pct = pd.to_numeric(df["net_d5_amount__pct"], errors="coerce")
    day_pct = pd.to_numeric(df["pct_change"], errors="coerce")

    rules["lg_rate_le_m1"] = lg_rate <= -1.0
    rules["lg_rate_le_m2"] = lg_rate <= -2.0
    rules["lg_rate_le_m3"] = lg_rate <= -3.0
    rules["net_amount_le_0"] = net_amt <= 0
    rules["net_d5_le_0"] = net_d5 <= 0
    rules["net_d5_pct_le_30"] = d5_pct <= 0.30
    rules["net_d5_pct_le_20"] = d5_pct <= 0.20

    # 金融含义更明确的组合：大单流出 + 当日/5日净流出 + 散户接盘/当日上涨
    rules["lg_m2_and_net0"] = (lg_rate <= -2.0) & (net_amt <= 0)
    rules["lg_m2_and_d5pct30"] = (lg_rate <= -2.0) & (d5_pct <= 0.30)
    rules["lg_m1_and_d5pct20"] = (lg_rate <= -1.0) & (d5_pct <= 0.20)
    rules["lg_m2_and_sm_p2"] = (lg_rate <= -2.0) & (sm_rate >= 2.0)
    rules["lg_m2_and_up"] = (lg_rate <= -2.0) & (day_pct >= 2.0)
    rules["net0_and_up"] = (net_amt <= 0) & (day_pct >= 2.0)
    rules["lg_m2_and_net0_and_up"] = (lg_rate <= -2.0) & (net_amt <= 0) & (day_pct >= 2.0)
    return rules


def analyze_bad_relations(df: pd.DataFrame, scheme: str, label_version: str, json_name: str) -> list[dict]:
    work = df[~df["buy_skip"]].copy()
    rows = []
    metrics = [
        "pct_change",
        "net_amount",
        "net_d5_amount",
        "buy_lg_amount",
        "buy_lg_amount_rate",
        "buy_md_amount_rate",
        "buy_sm_amount_rate",
        "net_amount__pct",
        "net_d5_amount__pct",
        "buy_lg_amount__pct",
    ]
    for bad_name, bad_mask in {
        "bad3": pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.03,
        "bad5": pd.to_numeric(work["filtered_ret"], errors="coerce") <= -0.05,
    }.items():
        bad = work[bad_mask]
        good = work[~bad_mask & work["filtered_ret"].notna()]
        for col in metrics:
            s_bad = pd.to_numeric(bad[col], errors="coerce").dropna()
            s_good = pd.to_numeric(good[col], errors="coerce").dropna()
            if len(s_bad) < 3 or len(s_good) < 10:
                continue
            diff = float(s_bad.mean() - s_good.mean())
            pooled = float(np.sqrt((s_bad.var(ddof=1) + s_good.var(ddof=1)) / 2)) if len(s_bad) > 1 and len(s_good) > 1 else np.nan
            effect = diff / pooled if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
            rows.append(
                {
                    "scheme": scheme,
                    "json_name": json_name,
                    "label_version": label_version,
                    "bad_type": bad_name,
                    "feature": col,
                    "bad_n": len(s_bad),
                    "good_n": len(s_good),
                    "bad_mean": float(s_bad.mean()),
                    "good_mean": float(s_good.mean()),
                    "mean_diff": diff,
                    "effect_size": effect,
                }
            )
    return rows


def evaluate_veto(df: pd.DataFrame, veto_name: str, veto_mask: pd.Series) -> list[dict]:
    work = df.copy()
    work["veto"] = veto_mask.fillna(False)
    work["veto_ret"] = pd.to_numeric(work["filtered_ret"], errors="coerce").mask(work["veto"])
    rows = []
    for seg_name, seg_df in split_segments(work).items():
        mean_ret, cum_ret, mdd, win_rate = summarize(seg_df["veto_ret"])
        rows.append(
            {
                "segment": seg_name,
                "active_days": int(seg_df["veto_ret"].notna().sum()),
                "trade_rate": float(seg_df["veto_ret"].notna().mean()),
                "mean_ret": mean_ret,
                "cum_ret": cum_ret,
                "max_drawdown": mdd,
                "win_rate": win_rate,
            }
        )
    return rows


def main() -> None:
    factor_df = am.prepare_factor_df(am.load_factor_df())
    moneyflow_df = load_moneyflow()

    relation_rows: list[dict] = []
    eval_rows: list[dict] = []

    for json_path in am.JSON_PATHS:
        label_version = "0.5_0.5" if "145807" in json_path.name else "0.75_0.25"
        df = am.build_base_df(json_path, factor_df)
        gap_map = am.build_buy_gap_map(df)
        scheme_map = build_base_scheme_map(df)

        for scheme_name in SCHEMES:
            rank, ret = scheme_map[scheme_name]
            work = attach_selected_moneyflow(df, rank, ret, gap_map, moneyflow_df)

            relation_rows.extend(analyze_bad_relations(work, scheme_name, label_version, json_path.name))

            # baseline
            for row in evaluate_veto(work, "baseline", pd.Series(False, index=work.index)):
                row.update(
                    {
                        "scheme": scheme_name,
                        "veto": "baseline",
                        "json_name": json_path.name,
                        "label_version": label_version,
                    }
                )
                eval_rows.append(row)

            rules = build_veto_rules(work)
            for veto_name, veto_mask in rules.items():
                veto_rate = float(veto_mask.fillna(False).mean())
                if veto_rate < 0.01 or veto_rate > 0.35:
                    continue
                for row in evaluate_veto(work, veto_name, veto_mask):
                    row.update(
                        {
                            "scheme": scheme_name,
                            "veto": veto_name,
                            "json_name": json_path.name,
                            "label_version": label_version,
                            "veto_rate": veto_rate,
                        }
                    )
                    eval_rows.append(row)

    relation_df = pd.DataFrame(relation_rows).sort_values(
        ["scheme", "bad_type", "effect_size"], ascending=[True, True, False]
    )
    relation_path = RESULTS_DIR / "moneyflow_bad_sample_relation_20260419.csv"
    relation_df.to_csv(relation_path, index=False, encoding="utf-8-sig")

    detailed = pd.DataFrame(eval_rows)
    detailed_path = RESULTS_DIR / "moneyflow_veto_scheme_compare_20260419.csv"
    detailed.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    agg = (
        detailed.groupby(["scheme", "veto", "segment"], as_index=False)
        .agg(
            avg_cum_ret=("cum_ret", "mean"),
            worst_cum_ret=("cum_ret", "min"),
            avg_mean_ret=("mean_ret", "mean"),
            avg_mdd=("max_drawdown", "mean"),
            avg_trade_rate=("trade_rate", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_veto_rate=("veto_rate", "mean"),
        )
        .sort_values(["scheme", "segment", "avg_cum_ret"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    agg_path = RESULTS_DIR / "moneyflow_veto_scheme_compare_aggregate_20260419.csv"
    agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    score_rows = []
    for (scheme, veto), sub in agg.groupby(["scheme", "veto"], sort=False):
        row = {"scheme": scheme, "veto": veto}
        for _, r in sub.iterrows():
            row[f"cum__{r['segment']}"] = r["avg_cum_ret"]
            row[f"mdd__{r['segment']}"] = r["avg_mdd"]
        row["score"] = (
            row.get("cum__second_half", 0) * 3.0
            + row.get("cum__seg5_last", 0) * 4.0
            + row.get("cum__recent_13d", 0) * 3.0
            + row.get("cum__full", 0) * 0.5
            + row.get("mdd__full", 0) * 0.5
        )
        score_rows.append(row)
    score_df = pd.DataFrame(score_rows).sort_values(["scheme", "score"], ascending=[True, False]).reset_index(drop=True)
    score_path = RESULTS_DIR / "moneyflow_veto_scheme_ranked_20260419.csv"
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")

    print(f"[OK] relation -> {relation_path}")
    print(f"[OK] detailed -> {detailed_path}")
    print(f"[OK] aggregate -> {agg_path}")
    print(f"[OK] ranked -> {score_path}")


if __name__ == "__main__":
    main()
