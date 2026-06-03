import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
METRICS_185 = RESULTS / "walkforward_metrics_20260520_210454.json"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"


FOCUS_FACTORS = [
    "ppo_strength",
    "mkt_sector_dispersion",
    "main_outflow_retail_takeover",
    "pct_chg_1d",
    "buy_lg_amount_rate",
    "relative_strength_20d",
    "high_volatility_pullback_alpha",
    "price_ma60_ratio",
    "abnormal_turnover_pct",
]


def load_trades():
    data = json.loads(METRICS_185.read_text(encoding="utf-8"))
    result = data[0]["all_results"]["1"]
    rows = []
    for t in result["trades"]:
        rows.append(
            {
                "stock_code": t["stock_code"],
                "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                "sell_datetime": pd.Timestamp(t["sell_datetime"]),
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
            }
        )
    trades = pd.DataFrame(rows).sort_values(["sell_datetime", "stock_code"]).reset_index(drop=True)
    return trades, float(result["initial_capital"])


def mark_drawdown_state(trades, initial_capital):
    equity = initial_capital
    peak = initial_capital
    peak_idx = -1
    max_dd = 0.0
    max_dd_peak_idx = -1
    max_dd_trough_idx = -1
    states = []
    for i, row in trades.iterrows():
        pre_equity = equity
        pre_peak = peak
        in_dd_at_entry_close = pre_equity < pre_peak
        equity += row["profit_loss"]
        if equity > peak:
            peak = equity
            peak_idx = i
        post_dd = (peak - equity) / peak if peak > 0 else 0.0
        if post_dd > max_dd:
            max_dd = post_dd
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i
        states.append(
            {
                "pre_equity": pre_equity,
                "pre_peak": pre_peak,
                "post_equity": equity,
                "post_peak": peak,
                "drawdown_after_close": post_dd,
                "dd_segment_trade": bool(in_dd_at_entry_close or equity < pre_peak),
                "dd_loss_trade": bool((in_dd_at_entry_close or equity < pre_peak) and row["profit_loss"] < 0),
            }
        )
    out = pd.concat([trades, pd.DataFrame(states)], axis=1)
    out["dd_max_window_trade"] = False
    if max_dd_peak_idx >= 0 and max_dd_trough_idx >= max_dd_peak_idx:
        out.loc[max_dd_peak_idx:max_dd_trough_idx, "dd_max_window_trade"] = True
    return out


def load_factor_frame():
    df = pd.read_parquet(FACTOR_FILE)
    df["date"] = pd.to_datetime(df["date"])
    skip = {"date", "stock_code", "stock_name", "name", "sw_l1", "sw_l2", "sw_l3", "industry"}
    numeric_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return df[["date", "stock_code"] + numeric_cols].sort_values(["stock_code", "date"]), numeric_cols


def attach_prev_factor_snapshot(events, factors, numeric_cols):
    grouped = {
        stock: (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
        for stock, g in factors.groupby("stock_code", sort=False)
    }
    rows = []
    for _, event in events.iterrows():
        rec = event.to_dict()
        item = grouped.get(event["stock_code"])
        if item is not None:
            dates, g = item
            pos = np.searchsorted(dates, np.datetime64(event["buy_date"]), side="left") - 1
            if pos >= 0:
                r = g.iloc[int(pos)]
                rec["factor_date"] = r["date"]
                for c in numeric_cols:
                    rec[c] = r[c]
        rows.append(rec)
    return pd.DataFrame(rows)


def separation_table(df, factors, dd_col):
    dd_mask = df[dd_col].fillna(False).astype(bool)
    dd = df[dd_mask].copy()
    normal = df[~dd_mask].copy()
    rows = []
    for f in factors:
        if f not in df.columns:
            rows.append({"factor": f, "status": "missing"})
            continue
        a = pd.to_numeric(dd[f], errors="coerce").dropna()
        b = pd.to_numeric(normal[f], errors="coerce").dropna()
        pooled = np.nanstd(np.r_[a, b])
        sep = abs((a.median() - b.median()) / pooled) if pooled > 0 and len(a) and len(b) else np.nan
        rows.append(
            {
                "factor": f,
                "dd_median": a.median(),
                "normal_median": b.median(),
                "dd_p20": a.quantile(0.2),
                "dd_p80": a.quantile(0.8),
                "normal_p20": b.quantile(0.2),
                "normal_p80": b.quantile(0.8),
                "separation": sep,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def eval_rule(df, rule):
    hit = pd.Series(True, index=df.index)
    for f, op, th in rule:
        s = pd.to_numeric(df[f], errors="coerce")
        if op == ">":
            hit &= s > th
        elif op == ">=":
            hit &= s >= th
        elif op == "<":
            hit &= s < th
        elif op == "<=":
            hit &= s <= th
        else:
            raise ValueError(op)
    dd = df["dd_segment_trade"]
    return {
        "hits": int(hit.sum()),
        "dd_hits": int((hit & dd).sum()),
        "normal_hits": int((hit & ~dd).sum()),
        "dd_coverage": float((hit & dd).sum() / dd.sum()) if dd.sum() else np.nan,
        "normal_false_rate": float((hit & ~dd).sum() / (~dd).sum()) if (~dd).sum() else np.nan,
        "hit_avg_return": float(df.loc[hit, "actual_return"].mean()) if hit.any() else np.nan,
        "hit_pnl": float(df.loc[hit, "profit_loss"].sum()) if hit.any() else 0.0,
    }


def main():
    trades, initial = load_trades()
    marked = mark_drawdown_state(trades, initial)
    factors, numeric_cols = load_factor_frame()
    df = attach_prev_factor_snapshot(marked, factors, numeric_cols)

    print("== DD state summary ==")
    dd_cols = ["dd_max_window_trade", "dd_loss_trade", "dd_segment_trade"]
    for dd_col in dd_cols:
        mask = df[dd_col].fillna(False).astype(bool)
        print(
            "{}: trades={} dd={} normal={} dd_avg={:.2%} normal_avg={:.2%} dd_pnl={:.0f} normal_pnl={:.0f}".format(
                dd_col,
                len(df),
                int(mask.sum()),
                int((~mask).sum()),
                df.loc[mask, "actual_return"].mean(),
                df.loc[~mask, "actual_return"].mean(),
                df.loc[mask, "profit_loss"].sum(),
                df.loc[~mask, "profit_loss"].sum(),
            )
        )

    dd_col = "dd_max_window_trade"
    sep = separation_table(df, FOCUS_FACTORS, dd_col).sort_values("separation", ascending=False)
    out_sep = RESULTS / "dd_regime_185_focus_factor_validation.csv"
    sep.to_csv(out_sep, index=False, encoding="utf-8-sig")
    print(f"\n== Focus factors ({dd_col}) ==")
    print(sep.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Full-sample quantiles reproduce the style of the Deepseek result, but are optimistic.
    candidates = {}
    if "mkt_sector_dispersion" in df.columns and "buy_lg_amount_rate" in df.columns:
        candidates["sector_disp>P60 & buy_lg>P50"] = [
            ("mkt_sector_dispersion", ">", pd.to_numeric(df["mkt_sector_dispersion"], errors="coerce").quantile(0.60)),
            ("buy_lg_amount_rate", ">", pd.to_numeric(df["buy_lg_amount_rate"], errors="coerce").quantile(0.50)),
        ]
    if "mkt_sector_dispersion" in df.columns and "main_outflow_retail_takeover" in df.columns:
        candidates["sector_disp>P60 & main_outflow<P30"] = [
            ("mkt_sector_dispersion", ">", pd.to_numeric(df["mkt_sector_dispersion"], errors="coerce").quantile(0.60)),
            (
                "main_outflow_retail_takeover",
                "<",
                pd.to_numeric(df["main_outflow_retail_takeover"], errors="coerce").quantile(0.30),
            ),
        ]
        candidates["sector_disp>P60 & main_outflow<P20"] = [
            ("mkt_sector_dispersion", ">", pd.to_numeric(df["mkt_sector_dispersion"], errors="coerce").quantile(0.60)),
            (
                "main_outflow_retail_takeover",
                "<",
                pd.to_numeric(df["main_outflow_retail_takeover"], errors="coerce").quantile(0.20),
            ),
        ]

    rows = []
    for name, rule in candidates.items():
        row = {"rule": name, "expr": " & ".join(f"{f}{op}{th:.4g}" for f, op, th in rule)}
        # eval_rule uses dd_segment_trade by default, so inline the selected DD mask here.
        hit = pd.Series(True, index=df.index)
        for f, op, th in rule:
            s = pd.to_numeric(df[f], errors="coerce")
            hit &= s > th if op == ">" else s < th
        dd = df[dd_col].fillna(False).astype(bool)
        row.update(
            {
                "hits": int(hit.sum()),
                "dd_hits": int((hit & dd).sum()),
                "normal_hits": int((hit & ~dd).sum()),
                "dd_coverage": float((hit & dd).sum() / dd.sum()) if dd.sum() else np.nan,
                "normal_false_rate": float((hit & ~dd).sum() / (~dd).sum()) if (~dd).sum() else np.nan,
                "hit_avg_return": float(df.loc[hit, "actual_return"].mean()) if hit.any() else np.nan,
                "hit_pnl": float(df.loc[hit, "profit_loss"].sum()) if hit.any() else 0.0,
            }
        )
        rows.append(row)
    out_rules = pd.DataFrame(rows).sort_values(["normal_false_rate", "dd_coverage"], ascending=[True, False])
    out_rules.to_csv(RESULTS / "dd_regime_185_rule_validation.csv", index=False, encoding="utf-8-sig")
    print("\n== Candidate rules using full-sample quantiles ==")
    print(out_rules.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
