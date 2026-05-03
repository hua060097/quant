import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "results" / "daily_top5_details_Window_3_20260428_123434.json"
FACTORS_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
OUT_PATH = BASE_DIR / "results" / "top1_position_control_scan_Window_3_20260428_123434.csv"
SUMMARY_PATH = BASE_DIR / "results" / "top1_position_control_best_summary_Window_3_20260428_123434.csv"

MARKET_FACTORS = [
    "market_trend_strength",
    "market_trend_strength_exp",
    "market_60d_new_high_ratio",
    "market_60d_new_high_ratio_exp",
    "mkt_regime_trend_pressure_v2",
    "mkt_regime_trend_pressure_v2_exp",
    "mkt_ma5_followthrough_v3",
    "mkt_ma5_followthrough_v3_exp",
    "market_trend_adaptive_beta",
]

WEIGHT_SCHEMES = [
    ("w0_50_100", (0.0, 0.5, 1.0)),
    ("w20_60_100", (0.2, 0.6, 1.0)),
    ("w30_60_100", (0.3, 0.6, 1.0)),
    ("w40_70_100", (0.4, 0.7, 1.0)),
    ("w50_75_100", (0.5, 0.75, 1.0)),
]


def load_top5_json() -> pd.DataFrame:
    raw = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for date_str, rec in raw.items():
        rows.append({
            "date": pd.to_datetime(date_str),
            "top1_code": rec["top1"]["stock_code"],
            "top1_ret": rec["top1"]["real_return"],
            "top2_ret": rec["top2"]["real_return"],
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def load_factor_data() -> pd.DataFrame:
    schema = set(pq.ParquetFile(FACTORS_PATH).schema.names)
    cols = [c for c in MARKET_FACTORS + ["price_ma5_ratio"] if c in schema]
    fac = pd.read_parquet(FACTORS_PATH, columns=["date", "stock_code"] + cols)
    fac["date"] = pd.to_datetime(fac["date"])
    return fac


def build_work_df(top_df: pd.DataFrame, fac: pd.DataFrame) -> pd.DataFrame:
    work = top_df.copy()
    top1_fac = fac.rename(columns={"stock_code": "top1_code", "price_ma5_ratio": "top1_price_ma5_ratio"})
    market_cols = [c for c in fac.columns if c not in ("date", "stock_code", "price_ma5_ratio")]
    market_snapshot = fac.groupby("date", as_index=False)[market_cols].first()
    work = work.merge(top1_fac[["date", "top1_code", "top1_price_ma5_ratio"]], on=["date", "top1_code"], how="left")
    work = work.merge(market_snapshot, on="date", how="left")
    # add top1 price_ma5 cs percentile for gate
    pct = fac[["date", "stock_code", "price_ma5_ratio"]].copy()
    pct["top1_price_ma5_cs_pct"] = pct.groupby("date")["price_ma5_ratio"].rank(pct=True, method="average")
    pct = pct.rename(columns={"stock_code": "top1_code"})
    work = work.merge(pct[["date", "top1_code", "top1_price_ma5_cs_pct"]], on=["date", "top1_code"], how="left")
    return work.sort_values("date").reset_index(drop=True)


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
    ret = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if len(ret) == 0:
        return np.nan
    equity = (1.0 + ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize(returns: pd.Series) -> Dict[str, float]:
    ret = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    return {
        "days": int(len(ret)),
        "mean": float(ret.mean()),
        "cum": float((1.0 + ret).prod() - 1.0),
        "mdd": calc_max_drawdown(ret),
        "win": float((ret > 0).mean()),
        "q10": float(ret.quantile(0.10)),
    }


def add_segment_metrics(row: Dict, returns: pd.Series) -> Dict:
    seg5 = returns.iloc[-33:] if len(returns) >= 33 else returns.iloc[0:0]
    recent13 = returns.iloc[-13:] if len(returns) >= 13 else returns.iloc[0:0]
    row.update({
        "seg5_mean": float(seg5.mean()) if len(seg5) else np.nan,
        "seg5_cum": float((1.0 + seg5).prod() - 1.0) if len(seg5) else np.nan,
        "seg5_mdd": calc_max_drawdown(seg5) if len(seg5) else np.nan,
        "recent13_mean": float(recent13.mean()) if len(recent13) else np.nan,
        "recent13_cum": float((1.0 + recent13).prod() - 1.0) if len(recent13) else np.nan,
        "recent13_mdd": calc_max_drawdown(recent13) if len(recent13) else np.nan,
    })
    return row


def score_rule(cum: float, mdd: float, seg5_cum: float, recent13_cum: float) -> float:
    dd = abs(float(mdd))
    return (
        1.00 * float(cum)
        - 0.45 * dd
        + 0.35 * float(seg5_cum)
        + 0.20 * float(recent13_cum)
    )


def scan(work: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    base_ret = pd.to_numeric(work["top1_ret"], errors="coerce").fillna(0.0)
    gate_mask = pd.to_numeric(work["top1_price_ma5_cs_pct"], errors="coerce") >= 0.4
    gate_ret = base_ret.where(gate_mask, 0.0)

    for rule_name, ret in [
        ("baseline_top1", base_ret),
        ("gate_price_ma5_cs_pct_ge_0.4", gate_ret),
    ]:
        row = {"rule": rule_name, **summarize(ret)}
        row = add_segment_metrics(row, ret)
        row["score"] = score_rule(row["cum"], row["mdd"], row["seg5_cum"], row["recent13_cum"])
        rows.append(row)

    for factor in MARKET_FACTORS:
        series = pd.to_numeric(work[factor], errors="coerce")
        if series.notna().sum() < 80:
            continue
        for low_q in [0.2, 0.3, 0.4]:
            for high_q in [0.6, 0.7, 0.8]:
                if high_q <= low_q:
                    continue
                low_thr = expanding_threshold(series, low_q)
                high_thr = expanding_threshold(series, high_q)
                valid = low_thr.notna() & high_thr.notna() & series.notna()
                if int(valid.sum()) < 60:
                    continue
                for direction in ["high_better", "low_better"]:
                    if direction == "high_better":
                        low_mask = series <= low_thr
                        high_mask = series >= high_thr
                    else:
                        low_mask = series >= high_thr
                        high_mask = series <= low_thr
                    mid_mask = valid & ~(low_mask | high_mask)
                    for scheme_name, (w_low, w_mid, w_high) in WEIGHT_SCHEMES:
                        weights = pd.Series(0.0, index=work.index)
                        weights.loc[valid] = w_mid
                        weights.loc[low_mask & valid] = w_low
                        weights.loc[high_mask & valid] = w_high
                        ret = base_ret * weights
                        row = {
                            "rule": f"posctl__{factor}__{direction}__q{int(low_q*100)}_{int(high_q*100)}__{scheme_name}",
                            "factor": factor,
                            "direction": direction,
                            "low_q": low_q,
                            "high_q": high_q,
                            "weights": scheme_name,
                            "low_weight": w_low,
                            "mid_weight": w_mid,
                            "high_weight": w_high,
                            **summarize(ret),
                            "avg_position": float(weights.mean()),
                            "low_days": int((low_mask & valid).sum()),
                            "mid_days": int(mid_mask.sum()),
                            "high_days": int((high_mask & valid).sum()),
                        }
                        row = add_segment_metrics(row, ret)
                        row["score"] = score_rule(row["cum"], row["mdd"], row["seg5_cum"], row["recent13_cum"])
                        rows.append(row)

                        gated_ret = base_ret.where(gate_mask, 0.0) * weights
                        row2 = {
                            "rule": f"gate_posctl__{factor}__{direction}__q{int(low_q*100)}_{int(high_q*100)}__{scheme_name}",
                            "factor": factor,
                            "direction": direction,
                            "low_q": low_q,
                            "high_q": high_q,
                            "weights": scheme_name,
                            "low_weight": w_low,
                            "mid_weight": w_mid,
                            "high_weight": w_high,
                            **summarize(gated_ret),
                            "avg_position": float((weights * gate_mask.astype(float)).mean()),
                            "low_days": int((low_mask & valid).sum()),
                            "mid_days": int(mid_mask.sum()),
                            "high_days": int((high_mask & valid).sum()),
                        }
                        row2 = add_segment_metrics(row2, gated_ret)
                        row2["score"] = score_rule(row2["cum"], row2["mdd"], row2["seg5_cum"], row2["recent13_cum"])
                        rows.append(row2)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def main():
    top_df = load_top5_json()
    fac = load_factor_data()
    work = build_work_df(top_df, fac)
    out = scan(work)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    summary = []
    baseline = out[out["rule"] == "baseline_top1"].iloc[0].to_dict()
    gate = out[out["rule"] == "gate_price_ma5_cs_pct_ge_0.4"].iloc[0].to_dict()
    best_pos = out[out["rule"].str.startswith("posctl__")].iloc[0].to_dict()
    best_gate_pos = out[out["rule"].str.startswith("gate_posctl__")].iloc[0].to_dict()
    summary.extend([baseline, gate, best_pos, best_gate_pos])
    pd.DataFrame(summary).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_PATH}")
    print(pd.DataFrame(summary)[["rule", "cum", "mdd", "mean", "seg5_cum", "recent13_cum", "avg_position", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
