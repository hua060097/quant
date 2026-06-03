import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
METRICS_185 = RESULTS / "walkforward_metrics_20260520_210454.json"
SHADOW_185 = RESULTS / "walkforward_shadow_skipped_opportunities_20260520_210454.csv"
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

SIGNIFICANT_MIN_TRADES = 4
SIGNIFICANT_MIN_DD = 0.05


def load_actual_trades():
    data = json.loads(METRICS_185.read_text(encoding="utf-8"))
    result = data[0]["all_results"]["1"]
    rows = []
    for t in result["trades"]:
        rows.append(
            {
                "event_type": "actual",
                "stock_code": t["stock_code"],
                "buy_datetime": pd.Timestamp(t["buy_datetime"]),
                "sell_datetime": pd.Timestamp(t["sell_datetime"]),
                "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
                "shadow_status": "",
            }
        )
    return pd.DataFrame(rows), float(result["initial_capital"])


def load_shadow_trades():
    if not SHADOW_185.exists():
        return pd.DataFrame()
    df = pd.read_csv(SHADOW_185)
    if df.empty:
        return df
    out = pd.DataFrame(
        {
            "event_type": "shadow",
            "stock_code": df["stock_code"].astype(str),
            "buy_datetime": pd.to_datetime(df["buy_datetime"]),
            "sell_datetime": pd.to_datetime(df["sell_datetime"]),
            "buy_date": pd.to_datetime(df["buy_datetime"].astype(str).str[:10]),
            "actual_return": pd.to_numeric(df["actual_return"], errors="coerce"),
            "profit_loss": pd.to_numeric(df["profit_loss"], errors="coerce"),
            "shadow_status": df.get("shadow_status", ""),
        }
    )
    return out


def mark_continuous_dd_segments(events, initial_capital, segment_prefix):
    events = events.sort_values(["sell_datetime", "event_type", "stock_code"]).reset_index(drop=True).copy()
    equity = initial_capital
    peak = initial_capital
    in_dd = False
    current_segment_id = None
    segment_start_idx = None
    segment_start_peak = None
    next_segment_no = 1
    rows = []
    segments = []

    for idx, event in events.iterrows():
        pre_equity = equity
        pre_peak = peak
        equity = equity + float(event["profit_loss"])

        starts_dd = (not in_dd) and equity < peak
        if starts_dd:
            current_segment_id = f"{segment_prefix}_{next_segment_no:03d}"
            next_segment_no += 1
            segment_start_idx = idx
            segment_start_peak = peak
            in_dd = True

        assigned_segment = current_segment_id if in_dd else ""
        drawdown_after = (peak - equity) / peak if peak > 0 else 0.0
        if in_dd:
            drawdown_after = (segment_start_peak - equity) / segment_start_peak if segment_start_peak else drawdown_after

        rows.append(
            {
                "pre_equity": pre_equity,
                "pre_peak": pre_peak,
                "post_equity": equity,
                "post_peak": max(peak, equity),
                "continuous_dd_segment_id": assigned_segment,
                "continuous_dd_trade": bool(assigned_segment),
                "drawdown_after_close": drawdown_after,
            }
        )

        recovers = in_dd and equity >= segment_start_peak
        if recovers:
            segment_events = events.loc[segment_start_idx:idx]
            segment_states = pd.DataFrame(rows).loc[segment_start_idx:idx]
            segments.append(
                {
                    "segment_id": current_segment_id,
                    "start_sell_datetime": segment_events["sell_datetime"].iloc[0],
                    "end_sell_datetime": segment_events["sell_datetime"].iloc[-1],
                    "trade_count": int(len(segment_events)),
                    "actual_count": int(segment_events["event_type"].eq("actual").sum()),
                    "shadow_count": int(segment_events["event_type"].eq("shadow").sum()),
                    "profit_loss": float(segment_events["profit_loss"].sum()),
                    "avg_return": float(segment_events["actual_return"].mean()),
                    "max_drawdown": float(segment_states["drawdown_after_close"].max()),
                    "recovered": True,
                }
            )
            in_dd = False
            current_segment_id = None
            segment_start_idx = None
            segment_start_peak = None

        if equity > peak:
            peak = equity

    if in_dd and current_segment_id:
        segment_events = events.loc[segment_start_idx:]
        segment_states = pd.DataFrame(rows).loc[segment_start_idx:]
        segments.append(
            {
                "segment_id": current_segment_id,
                "start_sell_datetime": segment_events["sell_datetime"].iloc[0],
                "end_sell_datetime": segment_events["sell_datetime"].iloc[-1],
                "trade_count": int(len(segment_events)),
                "actual_count": int(segment_events["event_type"].eq("actual").sum()),
                "shadow_count": int(segment_events["event_type"].eq("shadow").sum()),
                "profit_loss": float(segment_events["profit_loss"].sum()),
                "avg_return": float(segment_events["actual_return"].mean()),
                "max_drawdown": float(segment_states["drawdown_after_close"].max()),
                "recovered": False,
            }
        )

    marked = pd.concat([events, pd.DataFrame(rows)], axis=1)
    return marked, pd.DataFrame(segments)


def assign_shadow_to_actual_segments(actual_marked, actual_segments, shadows):
    shadows = shadows.copy()
    shadows["actual_dd_segment_id"] = ""
    if shadows.empty or actual_segments.empty:
        return shadows
    for _, seg in actual_segments.iterrows():
        mask = shadows["sell_datetime"].between(seg["start_sell_datetime"], seg["end_sell_datetime"], inclusive="both")
        shadows.loc[mask, "actual_dd_segment_id"] = seg["segment_id"]
    return shadows


def enrich_actual_segments_with_shadow(actual_segments, shadow_assigned):
    if actual_segments.empty:
        return actual_segments
    out = actual_segments.copy()
    if shadow_assigned.empty:
        out["assigned_shadow_count"] = 0
        out["assigned_shadow_pnl"] = 0.0
        return out
    grouped = (
        shadow_assigned[shadow_assigned["actual_dd_segment_id"].astype(str).ne("")]
        .groupby("actual_dd_segment_id")
        .agg(assigned_shadow_count=("stock_code", "size"), assigned_shadow_pnl=("profit_loss", "sum"))
    )
    out = out.merge(grouped, left_on="segment_id", right_index=True, how="left")
    out["assigned_shadow_count"] = out["assigned_shadow_count"].fillna(0).astype(int)
    out["assigned_shadow_pnl"] = out["assigned_shadow_pnl"].fillna(0.0).astype(float)
    out["segment_plus_assigned_shadow_pnl"] = out["profit_loss"] + out["assigned_shadow_pnl"]
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


def factor_separation(events, dd_col):
    dd = events[events[dd_col].astype(bool)]
    normal = events[~events[dd_col].astype(bool)]
    rows = []
    for factor in FOCUS_FACTORS:
        if factor not in events.columns:
            rows.append({"factor": factor, "status": "missing"})
            continue
        a = pd.to_numeric(dd[factor], errors="coerce").dropna()
        b = pd.to_numeric(normal[factor], errors="coerce").dropna()
        pooled = np.nanstd(np.r_[a, b])
        rows.append(
            {
                "factor": factor,
                "dd_median": float(a.median()) if len(a) else np.nan,
                "normal_median": float(b.median()) if len(b) else np.nan,
                "dd_mean": float(a.mean()) if len(a) else np.nan,
                "normal_mean": float(b.mean()) if len(b) else np.nan,
                "separation": float(abs((a.median() - b.median()) / pooled)) if pooled > 0 and len(a) and len(b) else np.nan,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows).sort_values("separation", ascending=False)


def eval_candidate_rules(events, dd_col):
    rows = []
    rule_specs = [
        ("sector_disp>P60 & buy_lg>P50", "mkt_sector_dispersion", ">", 0.60, "buy_lg_amount_rate", ">", 0.50),
        (
            "sector_disp>P60 & main_outflow<P30",
            "mkt_sector_dispersion",
            ">",
            0.60,
            "main_outflow_retail_takeover",
            "<",
            0.30,
        ),
        (
            "sector_disp>P60 & main_outflow<P20",
            "mkt_sector_dispersion",
            ">",
            0.60,
            "main_outflow_retail_takeover",
            "<",
            0.20,
        ),
    ]
    dd = events[dd_col].astype(bool)
    for name, f1, op1, q1, f2, op2, q2 in rule_specs:
        if f1 not in events.columns or f2 not in events.columns:
            continue
        th1 = pd.to_numeric(events[f1], errors="coerce").quantile(q1)
        th2 = pd.to_numeric(events[f2], errors="coerce").quantile(q2)
        s1 = pd.to_numeric(events[f1], errors="coerce")
        s2 = pd.to_numeric(events[f2], errors="coerce")
        hit = (s1 > th1 if op1 == ">" else s1 < th1) & (s2 > th2 if op2 == ">" else s2 < th2)
        rows.append(
            {
                "rule": name,
                "expr": f"{f1}{op1}{th1:.4g} & {f2}{op2}{th2:.4g}",
                "hits": int(hit.sum()),
                "dd_hits": int((hit & dd).sum()),
                "normal_hits": int((hit & ~dd).sum()),
                "dd_coverage": float((hit & dd).sum() / dd.sum()) if dd.sum() else np.nan,
                "normal_false_rate": float((hit & ~dd).sum() / (~dd).sum()) if (~dd).sum() else np.nan,
                "hit_pnl": float(events.loc[hit, "profit_loss"].sum()),
                "hit_avg_return": float(events.loc[hit, "actual_return"].mean()) if hit.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["normal_false_rate", "dd_coverage"], ascending=[True, False])


def add_significant_dd_labels(events, segments, prefix):
    events = events.copy()
    if segments.empty:
        events[f"{prefix}_significant_dd_trade"] = False
        return events, segments
    sig = segments[
        (segments["trade_count"] >= SIGNIFICANT_MIN_TRADES)
        & (segments["max_drawdown"] >= SIGNIFICANT_MIN_DD)
    ].copy()
    sig_ids = set(sig["segment_id"].astype(str))
    events[f"{prefix}_significant_dd_trade"] = events["continuous_dd_segment_id"].astype(str).isin(sig_ids)
    return events, sig


def build_pre_dd_events(events, significant_segments, label, trade_counts=(1, 2, 3), calendar_days=(1, 2, 3)):
    rows = []
    events = events.sort_values(["sell_datetime", "event_type", "stock_code"]).reset_index(drop=True)
    for _, seg in significant_segments.iterrows():
        start = pd.Timestamp(seg["start_sell_datetime"])
        before = events[events["sell_datetime"] < start].copy()
        if before.empty:
            continue
        for n in trade_counts:
            sample = before.tail(n).copy()
            sample["pre_dd_window"] = f"{label}_pre{n}_trades"
            sample["target_segment_id"] = seg["segment_id"]
            rows.append(sample)
        for d in calendar_days:
            sample = before[before["sell_datetime"] >= start - pd.Timedelta(days=d)].copy()
            if sample.empty:
                continue
            sample["pre_dd_window"] = f"{label}_pre{d}_calendar_days"
            sample["target_segment_id"] = seg["segment_id"]
            rows.append(sample)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


def pre_dd_rule_eval(events, pre_events, window_col="pre_dd_window"):
    if pre_events.empty:
        return pd.DataFrame()
    normal = events[~events["continuous_dd_trade"].astype(bool)].copy()
    combined_rows = []
    rule_specs = [
        ("sector_disp>P60 & buy_lg>P50", "mkt_sector_dispersion", ">", 0.60, "buy_lg_amount_rate", ">", 0.50),
        (
            "sector_disp>P60 & main_outflow<P30",
            "mkt_sector_dispersion",
            ">",
            0.60,
            "main_outflow_retail_takeover",
            "<",
            0.30,
        ),
        (
            "sector_disp>P60 & main_outflow<P20",
            "mkt_sector_dispersion",
            ">",
            0.60,
            "main_outflow_retail_takeover",
            "<",
            0.20,
        ),
    ]
    for name, f1, op1, q1, f2, op2, q2 in rule_specs:
        if f1 not in events.columns or f2 not in events.columns:
            continue
        th1 = pd.to_numeric(events[f1], errors="coerce").quantile(q1)
        th2 = pd.to_numeric(events[f2], errors="coerce").quantile(q2)
        for window, sample in pre_events.groupby(window_col):
            s1 = pd.to_numeric(sample[f1], errors="coerce")
            s2 = pd.to_numeric(sample[f2], errors="coerce")
            hit = (s1 > th1 if op1 == ">" else s1 < th1) & (s2 > th2 if op2 == ">" else s2 < th2)
            segment_hits = sample.loc[hit, "target_segment_id"].astype(str).nunique()
            total_segments = sample["target_segment_id"].astype(str).nunique()
            n1 = pd.to_numeric(normal[f1], errors="coerce")
            n2 = pd.to_numeric(normal[f2], errors="coerce")
            normal_hit = (n1 > th1 if op1 == ">" else n1 < th1) & (n2 > th2 if op2 == ">" else n2 < th2)
            combined_rows.append(
                {
                    "rule": name,
                    "window": window,
                    "expr": f"{f1}{op1}{th1:.4g} & {f2}{op2}{th2:.4g}",
                    "pre_events": int(len(sample)),
                    "pre_event_hits": int(hit.sum()),
                    "target_segments": int(total_segments),
                    "target_segment_hits": int(segment_hits),
                    "target_segment_coverage": float(segment_hits / total_segments) if total_segments else np.nan,
                    "normal_hits": int(normal_hit.sum()),
                    "normal_events": int(len(normal)),
                    "normal_false_rate": float(normal_hit.sum() / len(normal)) if len(normal) else np.nan,
                }
            )
    return pd.DataFrame(combined_rows).sort_values(
        ["target_segment_coverage", "normal_false_rate", "pre_event_hits"],
        ascending=[False, True, False],
    )


def segment_level_factor_summary(events, significant_segments):
    if significant_segments.empty:
        return pd.DataFrame()
    rows = []
    for _, seg in significant_segments.iterrows():
        sample = events[events["continuous_dd_segment_id"].astype(str).eq(str(seg["segment_id"]))]
        row = {
            "segment_id": seg["segment_id"],
            "trade_count": seg["trade_count"],
            "actual_count": seg.get("actual_count", np.nan),
            "shadow_count": seg.get("shadow_count", np.nan),
            "max_drawdown": seg["max_drawdown"],
            "profit_loss": seg["profit_loss"],
            "start_sell_datetime": seg["start_sell_datetime"],
            "end_sell_datetime": seg["end_sell_datetime"],
        }
        for factor in FOCUS_FACTORS:
            if factor in sample.columns:
                row[f"{factor}_median"] = pd.to_numeric(sample[factor], errors="coerce").median()
                row[f"{factor}_mean"] = pd.to_numeric(sample[factor], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("max_drawdown", ascending=False)


def print_summary(name, events, segments):
    dd = events["continuous_dd_trade"].astype(bool)
    print(f"\n== {name} ==")
    print(
        "events={} actual={} shadow={} segments={} dd_events={} normal_events={} dd_pnl={:.0f} normal_pnl={:.0f}".format(
            len(events),
            int(events["event_type"].eq("actual").sum()),
            int(events["event_type"].eq("shadow").sum()),
            len(segments),
            int(dd.sum()),
            int((~dd).sum()),
            events.loc[dd, "profit_loss"].sum(),
            events.loc[~dd, "profit_loss"].sum(),
        )
    )
    if not segments.empty:
        print(segments.sort_values("max_drawdown", ascending=False).head(10).to_string(index=False))


def main():
    actual, initial = load_actual_trades()
    shadow = load_shadow_trades()

    actual_marked, actual_segments = mark_continuous_dd_segments(actual, initial, "actual_dd")
    shadow_assigned = assign_shadow_to_actual_segments(actual_marked, actual_segments, shadow)
    actual_segments_enriched = enrich_actual_segments_with_shadow(actual_segments, shadow_assigned)
    actual_with_shadow_tags = pd.concat(
        [
            actual_marked.assign(actual_dd_segment_id=actual_marked["continuous_dd_segment_id"]),
            shadow_assigned.assign(
                continuous_dd_segment_id=shadow_assigned["actual_dd_segment_id"],
                continuous_dd_trade=shadow_assigned["actual_dd_segment_id"].astype(str).ne(""),
            ),
        ],
        ignore_index=True,
    ).sort_values(["sell_datetime", "event_type", "stock_code"])

    opportunity = pd.concat([actual, shadow], ignore_index=True)
    opportunity_marked, opportunity_segments = mark_continuous_dd_segments(opportunity, initial, "opportunity_dd")

    factors, numeric_cols = load_factor_frame()
    actual_with_shadow_factors = attach_prev_factor_snapshot(actual_with_shadow_tags, factors, numeric_cols)
    opportunity_factors = attach_prev_factor_snapshot(opportunity_marked, factors, numeric_cols)
    actual_with_shadow_factors, actual_significant_segments = add_significant_dd_labels(
        actual_with_shadow_factors, actual_segments_enriched, "actual"
    )
    opportunity_factors, opportunity_significant_segments = add_significant_dd_labels(
        opportunity_factors, opportunity_segments, "opportunity"
    )

    actual_pre = build_pre_dd_events(actual_with_shadow_factors, actual_significant_segments, "actual")
    opportunity_pre = build_pre_dd_events(opportunity_factors, opportunity_significant_segments, "opportunity")
    actual_pre_eval = pre_dd_rule_eval(actual_with_shadow_factors, actual_pre)
    opportunity_pre_eval = pre_dd_rule_eval(opportunity_factors, opportunity_pre)
    actual_segment_factor_summary = segment_level_factor_summary(actual_with_shadow_factors, actual_significant_segments)
    opportunity_segment_factor_summary = segment_level_factor_summary(opportunity_factors, opportunity_significant_segments)

    actual_segments_enriched.to_csv(RESULTS / "dd_segments_185_actual_equity.csv", index=False, encoding="utf-8-sig")
    opportunity_segments.to_csv(RESULTS / "dd_segments_185_opportunity_stream.csv", index=False, encoding="utf-8-sig")
    actual_with_shadow_factors.to_csv(RESULTS / "dd_events_185_actual_segments_with_shadow.csv", index=False, encoding="utf-8-sig")
    opportunity_factors.to_csv(RESULTS / "dd_events_185_opportunity_stream.csv", index=False, encoding="utf-8-sig")
    actual_significant_segments.to_csv(RESULTS / "dd_segments_185_actual_significant.csv", index=False, encoding="utf-8-sig")
    opportunity_significant_segments.to_csv(RESULTS / "dd_segments_185_opportunity_significant.csv", index=False, encoding="utf-8-sig")
    actual_pre.to_csv(RESULTS / "dd_pre_events_185_actual_significant.csv", index=False, encoding="utf-8-sig")
    opportunity_pre.to_csv(RESULTS / "dd_pre_events_185_opportunity_significant.csv", index=False, encoding="utf-8-sig")
    actual_pre_eval.to_csv(RESULTS / "dd_pre_rule_eval_185_actual_significant.csv", index=False, encoding="utf-8-sig")
    opportunity_pre_eval.to_csv(RESULTS / "dd_pre_rule_eval_185_opportunity_significant.csv", index=False, encoding="utf-8-sig")
    actual_segment_factor_summary.to_csv(
        RESULTS / "dd_segment_factor_summary_185_actual_significant.csv", index=False, encoding="utf-8-sig"
    )
    opportunity_segment_factor_summary.to_csv(
        RESULTS / "dd_segment_factor_summary_185_opportunity_significant.csv", index=False, encoding="utf-8-sig"
    )

    actual_sep = factor_separation(actual_with_shadow_factors, "continuous_dd_trade")
    opportunity_sep = factor_separation(opportunity_factors, "continuous_dd_trade")
    actual_rules = eval_candidate_rules(actual_with_shadow_factors, "continuous_dd_trade")
    opportunity_rules = eval_candidate_rules(opportunity_factors, "continuous_dd_trade")

    actual_sep.to_csv(RESULTS / "dd_regime_185_actual_with_shadow_factor_sep.csv", index=False, encoding="utf-8-sig")
    opportunity_sep.to_csv(RESULTS / "dd_regime_185_opportunity_factor_sep.csv", index=False, encoding="utf-8-sig")
    actual_rules.to_csv(RESULTS / "dd_regime_185_actual_with_shadow_rule_eval.csv", index=False, encoding="utf-8-sig")
    opportunity_rules.to_csv(RESULTS / "dd_regime_185_opportunity_rule_eval.csv", index=False, encoding="utf-8-sig")

    print_summary("Actual equity DD segments + shadow assignment", actual_with_shadow_factors, actual_segments_enriched)
    print(f"\n-- significant actual DD segments: trades>={SIGNIFICANT_MIN_TRADES}, max_dd>={SIGNIFICANT_MIN_DD:.0%} --")
    print(actual_significant_segments.to_string(index=False))
    print("\n-- factor separation: actual DD segments with shadows assigned --")
    print(actual_sep.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n-- candidate rules: actual DD segments with shadows assigned --")
    print(actual_rules.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n-- pre-DD candidate rules: actual significant segments --")
    print(actual_pre_eval.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print_summary("Opportunity stream DD segments", opportunity_factors, opportunity_segments)
    print(f"\n-- significant opportunity DD segments: trades>={SIGNIFICANT_MIN_TRADES}, max_dd>={SIGNIFICANT_MIN_DD:.0%} --")
    print(opportunity_significant_segments.to_string(index=False))
    print("\n-- factor separation: opportunity stream --")
    print(opportunity_sep.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n-- candidate rules: opportunity stream --")
    print(opportunity_rules.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n-- pre-DD candidate rules: opportunity significant segments --")
    print(opportunity_pre_eval.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
