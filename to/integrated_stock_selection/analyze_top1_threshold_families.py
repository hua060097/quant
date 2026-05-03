import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = BASE_DIR / "results" / "daily_top5_details_Window_3_20260413_010230.json"
FACTORS_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
OUT_DIR = BASE_DIR / "results"


CANDIDATE_FACTORS = [
    "market_trend_strength",
    "market_trend_adaptive_beta",
    "coskewness",
    "max_drawdown_20d",
    "mkt_sector_dispersion",
    "sector_vols_divergence_alpha",
    "return_10d_sector_mean",
    "sector_limit_overflow_alpha",
    "sector_turnover_change_5d_20d",
    "volume_structure_alpha",
    "sector_leader_capture_rank",
    "abnormal_turnover",
    "ppo_strength",
    "price_ma5_ratio",
    "intra_sector_momentum_rank",
]


def load_top5_json(json_path: Path) -> pd.DataFrame:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for date_str, rec in raw.items():
        row = {"date": pd.to_datetime(date_str)}
        for topk in ["top1", "top2", "top3", "top4", "top5"]:
            row[f"{topk}_code"] = rec[topk]["stock_code"]
            row[f"{topk}_pred"] = rec[topk]["pred"]
            row[f"{topk}_ret"] = rec[topk]["real_return"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["top1_positive"] = (df["top1_ret"] > 0).astype(int)
    df["spread12"] = df["top1_ret"] - df["top2_ret"]
    return df


def load_factor_snapshot() -> pd.DataFrame:
    schema = set(pq.ParquetFile(FACTORS_PATH).schema.names)
    cols = [c for c in CANDIDATE_FACTORS if c in schema]
    return pd.read_parquet(FACTORS_PATH, columns=["date", "stock_code"] + cols)


def attach_top_factors(top_df: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    fac = fac.copy()
    fac["date"] = pd.to_datetime(fac["date"])
    work = top_df.copy()
    factor_cols = [c for c in fac.columns if c not in ("date", "stock_code")]
    for side in ["top1", "top2"]:
        tmp = fac.rename(
            columns={
                "stock_code": f"{side}_code",
                **{c: f"{side}__{c}" for c in factor_cols},
            }
        )
        work = work.merge(tmp, on=["date", f"{side}_code"], how="left")
    return work


def add_cross_sectional_percentiles(work: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [c for c in fac.columns if c not in ("date", "stock_code")]
    pct = fac[["date", "stock_code"]].copy()
    for col in factor_cols:
        pct[col] = fac.groupby("date")[col].rank(pct=True, method="average")
    out = work.copy()
    for side in ["top1", "top2"]:
        tmp = pct.rename(
            columns={
                "stock_code": f"{side}_code",
                **{c: f"{side}__{c}__cs_pct" for c in factor_cols},
            }
        )
        out = out.merge(tmp, on=["date", f"{side}_code"], how="left")
    return out


def expanding_threshold(series: pd.Series, q: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = np.full(len(values), np.nan, dtype=float)
    history = []
    for i, val in enumerate(values):
        if history:
            out[i] = float(pd.Series(history).quantile(q))
        if pd.notna(val):
            history.append(float(val))
    return pd.Series(out, index=series.index)


def calc_max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize_strategy(returns: pd.Series, active_mask: pd.Series | None = None) -> dict:
    ret = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    active = active_mask if active_mask is not None else pd.Series(True, index=ret.index)
    return {
        "days": int(len(ret)),
        "active_days": int(active.sum()),
        "pass_rate": float(active.mean()) if len(active) else np.nan,
        "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
        "mean_ret": float(ret.mean()) if len(ret) else np.nan,
        "median_ret": float(ret.median()) if len(ret) else np.nan,
        "cum_ret": float((1.0 + ret).prod() - 1.0) if len(ret) else np.nan,
        "max_drawdown": calc_max_drawdown(ret),
    }


def scan_single_factors(work: pd.DataFrame):
    gate_rows = []
    switch_rows = []
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    factor_cols = [c.replace("top1__", "") for c in work.columns if c.startswith("top1__") and "__cs_pct" not in c]
    for factor in factor_cols:
        top1_series = pd.to_numeric(work[f"top1__{factor}"], errors="coerce")
        top2_series = pd.to_numeric(work[f"top2__{factor}"], errors="coerce")
        cs_series = pd.to_numeric(work.get(f"top1__{factor}__cs_pct"), errors="coerce")

        valid = top1_series.notna() & work["top1_ret"].notna() & work["top2_ret"].notna()
        if int(valid.sum()) < 40:
            continue

        # family 1: full-sample top1 history quantiles
        hist_values = top1_series[valid]
        for q in quantiles:
            thr = float(hist_values.quantile(q))
            for direction in ["high", "low"]:
                cond = top1_series >= thr if direction == "high" else top1_series <= thr
                active = valid & cond
                if int(active.sum()) >= 18:
                    gate_rows.append({
                        "family": "hist_top1_quantile",
                        "factor": factor,
                        "q": q,
                        "thr": thr,
                        "direction": direction,
                        **summarize_strategy(work["top1_ret"].where(active, 0.0), active),
                    })
                switch_rows.extend(_single_switch_rules(work, valid, cond, factor, "hist_top1_quantile", q, thr, direction))

        # family 2: expanding shift1
        for q in quantiles:
            thr_series = expanding_threshold(top1_series, q)
            for direction in ["high", "low"]:
                cond = top1_series >= thr_series if direction == "high" else top1_series <= thr_series
                active = valid & cond & thr_series.notna()
                if int(active.sum()) >= 18:
                    gate_rows.append({
                        "family": "expanding_top1_quantile_shift1",
                        "factor": factor,
                        "q": q,
                        "thr": np.nan,
                        "direction": direction,
                        **summarize_strategy(work["top1_ret"].where(active, 0.0), active),
                    })
                switch_rows.extend(_single_switch_rules(work, valid & thr_series.notna(), cond, factor, "expanding_top1_quantile_shift1", q, np.nan, direction))

        # family 3: daily cross-sectional percentile
        if cs_series.notna().sum() >= 40:
            for q in quantiles:
                thr = q
                for direction in ["high", "low"]:
                    cond = cs_series >= thr if direction == "high" else cs_series <= thr
                    active = valid & cond
                    if int(active.sum()) >= 18:
                        gate_rows.append({
                            "family": "daily_cs_percentile",
                            "factor": factor,
                            "q": q,
                            "thr": thr,
                            "direction": direction,
                            **summarize_strategy(work["top1_ret"].where(active, 0.0), active),
                        })
                    switch_rows.extend(_single_switch_rules(work, valid, cond, factor, "daily_cs_percentile", q, thr, direction))

        # family 4: fixed raw thresholds from in-sample decile values
        fixed_grid = hist_values.quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        for thr in fixed_grid:
            for direction in ["high", "low"]:
                cond = top1_series >= thr if direction == "high" else top1_series <= thr
                active = valid & cond
                if int(active.sum()) >= 18:
                    gate_rows.append({
                        "family": "fixed_raw_threshold",
                        "factor": factor,
                        "q": np.nan,
                        "thr": float(thr),
                        "direction": direction,
                        **summarize_strategy(work["top1_ret"].where(active, 0.0), active),
                    })
                switch_rows.extend(_single_switch_rules(work, valid, cond, factor, "fixed_raw_threshold", np.nan, float(thr), direction))

    return pd.DataFrame(gate_rows), pd.DataFrame(switch_rows)


def _single_switch_rules(work, valid_mask, cond, factor, family, q, thr, direction):
    rows = []
    choose_top1 = pd.Series(np.where(cond & valid_mask, work["top1_ret"], work["top2_ret"]), index=work.index)
    rows.append({
        "family": family,
        "factor": factor,
        "q": q,
        "thr": thr,
        "direction": direction,
        "rule": "cond_top1_else_top2",
        **summarize_strategy(choose_top1[valid_mask]),
    })
    choose_top2 = pd.Series(np.where(cond & valid_mask, work["top2_ret"], work["top1_ret"]), index=work.index)
    rows.append({
        "family": family,
        "factor": factor,
        "q": q,
        "thr": thr,
        "direction": direction,
        "rule": "cond_top2_else_top1",
        **summarize_strategy(choose_top2[valid_mask]),
    })
    return rows


def scan_combo_rules(work: pd.DataFrame, gate_df: pd.DataFrame, switch_df: pd.DataFrame):
    combo_gate_rows = []
    combo_switch_rows = []

    gate_short = gate_df.sort_values(["win_rate", "mean_ret"], ascending=False)["factor"].drop_duplicates().head(8).tolist()
    switch_short = switch_df.sort_values(["mean_ret", "win_rate"], ascending=False)["factor"].drop_duplicates().head(8).tolist()

    for f1, f2 in combinations(gate_short, 2):
        s1 = pd.to_numeric(work[f"top1__{f1}"], errors="coerce")
        s2 = pd.to_numeric(work[f"top1__{f2}"], errors="coerce")
        valid = s1.notna() & s2.notna() & work["top1_ret"].notna()
        if int(valid.sum()) < 30:
            continue
        for q1, thr1 in s1[valid].quantile([0.3, 0.5, 0.7]).items():
            for d1 in ["high", "low"]:
                c1 = s1 >= thr1 if d1 == "high" else s1 <= thr1
                for q2, thr2 in s2[valid].quantile([0.3, 0.5, 0.7]).items():
                    for d2 in ["high", "low"]:
                        c2 = s2 >= thr2 if d2 == "high" else s2 <= thr2
                        active = valid & c1 & c2
                        if int(active.sum()) < 15:
                            continue
                        combo_gate_rows.append({
                            "f1": f1, "q1": q1, "thr1": float(thr1), "d1": d1,
                            "f2": f2, "q2": q2, "thr2": float(thr2), "d2": d2,
                            **summarize_strategy(work["top1_ret"].where(active, 0.0), active),
                        })

    for f1, f2 in combinations(switch_short, 2):
        d1 = pd.to_numeric(work[f"top1__{f1}"], errors="coerce") - pd.to_numeric(work[f"top2__{f1}"], errors="coerce")
        d2 = pd.to_numeric(work[f"top1__{f2}"], errors="coerce") - pd.to_numeric(work[f"top2__{f2}"], errors="coerce")
        valid = d1.notna() & d2.notna() & work["top1_ret"].notna() & work["top2_ret"].notna()
        if int(valid.sum()) < 30:
            continue
        for q1, thr1 in d1[valid].quantile([0.3, 0.5, 0.7]).items():
            for dir1 in ["high", "low"]:
                c1 = d1 >= thr1 if dir1 == "high" else d1 <= thr1
                for q2, thr2 in d2[valid].quantile([0.3, 0.5, 0.7]).items():
                    for dir2 in ["high", "low"]:
                        c2 = d2 >= thr2 if dir2 == "high" else d2 <= thr2
                        cond = valid & c1 & c2
                        choose_top1 = pd.Series(np.where(cond, work["top1_ret"], work["top2_ret"]), index=work.index)
                        combo_switch_rows.append({
                            "f1": f1, "q1": q1, "thr1": float(thr1), "d1": dir1,
                            "f2": f2, "q2": q2, "thr2": float(thr2), "d2": dir2,
                            "rule": "cond_top1_else_top2",
                            "trigger_days": int(cond.sum()),
                            **summarize_strategy(choose_top1[valid]),
                        })
                        choose_top2 = pd.Series(np.where(cond, work["top2_ret"], work["top1_ret"]), index=work.index)
                        combo_switch_rows.append({
                            "f1": f1, "q1": q1, "thr1": float(thr1), "d1": dir1,
                            "f2": f2, "q2": q2, "thr2": float(thr2), "d2": dir2,
                            "rule": "cond_top2_else_top1",
                            "trigger_days": int(cond.sum()),
                            **summarize_strategy(choose_top2[valid]),
                        })

    return pd.DataFrame(combo_gate_rows), pd.DataFrame(combo_switch_rows)


def build_summary(gate_df, switch_df, combo_gate_df, combo_switch_df, work):
    base_rows = [
        {"scheme": "baseline_top1", **summarize_strategy(work["top1_ret"])},
        {"scheme": "baseline_top2", **summarize_strategy(work["top2_ret"])},
    ]

    picks = []
    if len(gate_df):
        best_gate = gate_df.sort_values(["win_rate", "mean_ret", "cum_ret"], ascending=False).iloc[0].to_dict()
        best_gate["scheme"] = "best_single_gate"
        picks.append(best_gate)
        best_gate_cov = gate_df.sort_values(["mean_ret", "win_rate", "cum_ret"], ascending=False).iloc[0].to_dict()
        best_gate_cov["scheme"] = "best_single_gate_return"
        picks.append(best_gate_cov)
    if len(switch_df):
        best_switch = switch_df.sort_values(["mean_ret", "win_rate", "cum_ret"], ascending=False).iloc[0].to_dict()
        best_switch["scheme"] = "best_single_switch"
        picks.append(best_switch)
    if len(combo_gate_df):
        best_combo_gate = combo_gate_df.sort_values(["win_rate", "mean_ret", "cum_ret"], ascending=False).iloc[0].to_dict()
        best_combo_gate["scheme"] = "best_combo_gate"
        picks.append(best_combo_gate)
    if len(combo_switch_df):
        best_combo_switch = combo_switch_df.sort_values(["mean_ret", "win_rate", "cum_ret"], ascending=False).iloc[0].to_dict()
        best_combo_switch["scheme"] = "best_combo_switch"
        picks.append(best_combo_switch)

    return pd.DataFrame(base_rows + picks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=str(DEFAULT_JSON_PATH))
    args = parser.parse_args()

    json_path = Path(args.json_path)
    top_df = load_top5_json(json_path)
    fac = load_factor_snapshot()
    work = attach_top_factors(top_df, fac)
    work = add_cross_sectional_percentiles(work, fac)

    gate_df, switch_df = scan_single_factors(work)
    combo_gate_df, combo_switch_df = scan_combo_rules(work, gate_df, switch_df)
    summary_df = build_summary(gate_df, switch_df, combo_gate_df, combo_switch_df, work)

    OUT_DIR.mkdir(exist_ok=True)
    out_tag = json_path.stem.replace("daily_top5_details_", "")
    gate_df.sort_values(["win_rate", "mean_ret", "cum_ret"], ascending=False).to_csv(
        OUT_DIR / f"top1_gate_threshold_family_scan_{out_tag}.csv", index=False, encoding="utf-8-sig"
    )
    switch_df.sort_values(["mean_ret", "win_rate", "cum_ret"], ascending=False).to_csv(
        OUT_DIR / f"top1_top2_switch_threshold_family_scan_{out_tag}.csv", index=False, encoding="utf-8-sig"
    )
    combo_gate_df.sort_values(["win_rate", "mean_ret", "cum_ret"], ascending=False).to_csv(
        OUT_DIR / f"top1_gate_combo_family_scan_{out_tag}.csv", index=False, encoding="utf-8-sig"
    )
    combo_switch_df.sort_values(["mean_ret", "win_rate", "cum_ret"], ascending=False).to_csv(
        OUT_DIR / f"top1_top2_switch_combo_family_scan_{out_tag}.csv", index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        OUT_DIR / f"top1_threshold_family_best_summary_{out_tag}.csv", index=False, encoding="utf-8-sig"
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
