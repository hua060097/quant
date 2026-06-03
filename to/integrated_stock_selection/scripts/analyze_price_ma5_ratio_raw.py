"""price_ma5_ratio_raw 作为 three_rule_v3 候选规则的全面审计。

price_ma5_ratio_raw = (close - ma5) / ma5 * 100
含义：价格相对5日均线的乖离率百分比。正值=高于MA5（短期超买），负值=低于MA5。

分析：单因子 + 横截面分位 + 与其他因子组合
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
METRICS_FILE = RESULTS / "walkforward_metrics_20260601_024558.json"
SELECTIONS_FILE = RESULTS / "walkforward_selections_20260601_024558.json"
SHADOW_FILE = RESULTS / "walkforward_shadow_skipped_opportunities_20260601_024558.json"

RULE_COLS = [
    "runup_ma5_exhaustion", "cpv_factor", "prev_day_blowoff_risk",
    "sector_momentum_accel_rank", "sector_leadership_persistence_v3",
    "return_20d_intra_z", "market_liquidity_change", "ulcer_index",
    "momentum_peak_decay_3d", "intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2", "market_advance_decline_ratio",
    "limit_down_3d", "top10_10d_strength_level_v3", "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d", "kdj_j_9",
    "score_z_in_day", "price_ma5_ratio_raw",
]


def load_json_records(path):
    obj = json.load(open(path, encoding="utf-8"))
    records = []
    def walk(x):
        if isinstance(x, list):
            for i in x: walk(i)
        elif isinstance(x, dict):
            keys = set(x)
            if "stock_code" in keys and ("actual_return" in keys or "profit_loss_pct" in keys
                    or "buy_datetime" in keys or "date" in keys or "trade_date" in keys):
                records.append(x)
            else:
                for v in x.values(): walk(v)
    walk(obj)
    return pd.DataFrame(records)


def normalize_events(df, source):
    df = df.copy()
    if "date" not in df.columns:
        for c in ["trade_date", "buy_date", "selection_date", "buy_datetime",
                   "shadow_attempt_buy_date", "shadow_prediction_date"]:
            if c in df.columns: df["date"] = df[c]; break
    if "stock_code" not in df.columns:
        for c in ["code", "ts_code", "selected_stock"]:
            if c in df.columns: df["stock_code"] = df[c]; break
    ret_candidates = ["actual_return", "return", "trade_return", "future_return",
                      "future_return_5d", "final_return", "realized_return"]
    if "actual_return" not in df.columns:
        for c in ret_candidates:
            if c in df.columns: df["actual_return"] = df[c]; break
    keep = [c for c in df.columns if c in {"date", "stock_code", "actual_return"}]
    df = df[keep].copy()
    df["source"] = source
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str)
    df["actual_return"] = pd.to_numeric(df.get("actual_return"), errors="coerce")
    return df.dropna(subset=["date", "stock_code", "actual_return"])


def current_v3_masks(df):
    out = pd.DataFrame(index=df.index)
    out["r1"] = (df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)
    out["r2"] = (
        ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
        | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
        | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15))
    )
    out["r3"] = (df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40)
    out["r4"] = (df["market_liquidity_change"] < 0.94) & (
        (df["ulcer_index"] < -0.80) | (df["momentum_peak_decay_3d"] > 1.0)
        | (df["intra_sector_momentum_rank"] < 0.67)
    )
    out["r5"] = (df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38)
    out["r6"] = df["limit_down_3d"] == 1
    out["r7"] = (df["top10_10d_strength_level_v3"] >= 0.70) & (df["buy_sm_amount_rate"] <= -1.25)
    out["r8"] = (df["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80) & (df["kdj_j_9"] <= 0.75)
    out["r9"] = df["score_z_in_day"] <= 1.50
    out["current_v3"] = out.any(axis=1)
    return out.fillna(False)


def bucket_counts(df):
    r = df["actual_return"]
    return {"n": int(len(df)), "mean": float(r.mean()) if len(df) else np.nan,
            "big_loss<=-4%": int((r <= -0.04).sum()),
            "small_loss": int(((r > -0.04) & (r < 0)).sum()),
            "small_gain3-8%": int(((r >= 0.03) & (r < 0.08)).sum()),
            "big_gain>=8%": int((r >= 0.08).sum()),
            "big_gain>=10%": int((r >= 0.10).sum())}


def eval_mask(df, mask, label=""):
    mask = mask.fillna(False)
    cur = df["current_v3"].fillna(False)
    inc = mask & ~cur
    inc_rows = df[inc]
    bl = inc_rows["actual_return"] <= -0.04
    bg = inc_rows["actual_return"] >= 0.08
    sg = (inc_rows["actual_return"] >= 0.03) & (inc_rows["actual_return"] < 0.08)
    return {"label": label,
            "standalone": bucket_counts(df[mask]),
            "incremental": bucket_counts(inc_rows),
            "overlap_current": int((mask & cur).sum()),
            "inc_actual_big_loss": int(((inc_rows["source"] == "actual") & bl).sum()),
            "inc_actual_big_gain": int(((inc_rows["source"] == "actual") & bg).sum()),
            "inc_shadow_big_loss": int(((inc_rows["source"] == "shadow") & bl).sum()),
            "inc_shadow_big_gain": int(((inc_rows["source"] == "shadow") & bg).sum()),
            "inc_small_gain_rows": inc_rows.loc[sg, [
                "source","date","stock_code","actual_return","price_ma5_ratio_raw",
                "price_ma5_ratio_raw_pct"
            ]].to_dict("records"),
            "inc_bl_rows": inc_rows.loc[bl, [
                "source","date","stock_code","actual_return","price_ma5_ratio_raw",
                "price_ma5_ratio_raw_pct","market_advance_decline_ratio"
            ]].sort_values("actual_return").to_dict("records"),
            "inc_bg_rows": inc_rows.loc[bg, [
                "source","date","stock_code","actual_return","price_ma5_ratio_raw",
                "price_ma5_ratio_raw_pct","market_advance_decline_ratio"
            ]].sort_values("actual_return", ascending=False).to_dict("records")}


def main():
    print("Loading data...")
    selections = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(
        ["source","date","stock_code","actual_return"])

    schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
    factor_cols = ["date","stock_code"] + [c for c in RULE_COLS if c in schema_cols]
    factors = pd.read_parquet(FACTORS, columns=factor_cols)
    for col in RULE_COLS:
        if col not in factors.columns: factors[col] = np.nan
    factors["date"] = pd.to_datetime(factors["date"])

    # Compute daily cross-sectional percentile
    raw_col = "price_ma5_ratio_raw"
    pct_col = raw_col + "_pct"
    print(f"Computing daily cross-sectional percentiles for {raw_col}...")
    factors[pct_col] = factors.groupby("date")[raw_col].transform(lambda x: x.rank(pct=True))
    print(f"  pct stats: {factors[pct_col].describe().to_dict()}")

    trading_days = np.array(sorted(factors["date"].dropna().unique()))
    prev_dates = []
    for d in events["date"]:
        trade_date = pd.Timestamp(d).normalize()
        pos = np.searchsorted(trading_days, np.datetime64(trade_date), side="left") - 1
        prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)
    events["factor_date"] = prev_dates
    snap = factors.rename(columns={"date": "factor_date"})
    events = events.merge(snap, on=["factor_date","stock_code"], how="left")
    masks = current_v3_masks(events)
    events = pd.concat([events, masks], axis=1)
    events["bucket"] = pd.cut(events["actual_return"],
        bins=[-np.inf, -0.04, 0, 0.03, 0.08, np.inf],
        labels=["big_loss","small_loss","flat_small_gain","small_gain","big_gain"])

    raw = events[raw_col]
    pct = events[pct_col]
    adr = events["market_advance_decline_ratio"]

    print(f"\n{'='*80}")
    print(f"DATA: {len(events)} events, missing raw={int(raw.isna().sum())}, missing pct={int(pct.isna().sum())}")
    print(f"\n{raw_col} distribution on events:")
    print(f"  {raw.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_string()}")
    print(f"\n{pct_col} distribution on events:")
    print(f"  {pct.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_string()}")

    print(f"\n{'='*80}")
    print(f"FACTOR BY BUCKET (combined)")
    for col in [raw_col, pct_col]:
        print(f"\n  {col}:")
        print(events.groupby(["source","bucket"])[col].describe(
            percentiles=[.5,.75,.9]).to_string())

    # Unfiltered analysis
    uf = events[~events["current_v3"]]
    uf_bl = uf[uf["actual_return"] <= -0.04]
    uf_bg = uf[uf["actual_return"] >= 0.08]
    dc = ["source","date","stock_code","actual_return",raw_col,pct_col,"market_advance_decline_ratio"]

    print(f"\n{'='*80}")
    print("UNFILTERED BIG LOSSES")
    for _, r in uf_bl.sort_values("actual_return").iterrows():
        print(f"  {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f} "
              f"raw={r[raw_col]:+.2f} pct={r[pct_col]:.4f} adr={r['market_advance_decline_ratio']:.4f}")
    print(f"\nUNFILTERED BIG GAINS")
    for _, r in uf_bg.sort_values("actual_return", ascending=False).iterrows():
        print(f"  {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f} "
              f"raw={r[raw_col]:+.2f} pct={r[pct_col]:.4f} adr={r['market_advance_decline_ratio']:.4f}")

    # ============================================================
    # SINGLE THRESHOLD SCAN (raw and pct)
    # ============================================================
    def scan_single(values, name, thresholds):
        print(f"\n{'='*80}")
        print(f"SINGLE THRESHOLD SCAN: {name}")
        print(f"Score = inc_big_loss - 2.5 * inc_big_gain8")
        for t in thresholds:
            mask = values >= t
            s = eval_mask(events, mask, f"{name}>={t}")
            inc = s["incremental"]
            score = inc["big_loss<=-4%"] - 2.5 * inc["big_gain>=8%"]
            print(f"  {name}>={t}: standalone_n={s['standalone']['n']:3d} inc_n={inc['n']:2d} "
                  f"inc_big_loss={inc['big_loss<=-4%']} inc_big_gain8={inc['big_gain>=8%']} "
                  f"inc_small_gain={inc['small_gain3-8%']} inc_mean={inc['mean']:+.4f} "
                  f"overlap={s['overlap_current']} score={score:.1f}")

    # Raw thresholds - scan meaningful values
    raw_ps = raw.dropna().quantile([.5,.75,.8,.85,.9,.95,.98,.99])
    raw_thresholds = sorted(set(
        [-5.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0] +
        [round(float(q),1) for q in raw_ps]
    ))
    scan_single(raw, raw_col, raw_thresholds)

    pct_thresholds = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.92,0.95,0.97,0.98,0.99]
    scan_single(pct, f"{raw_col}_pct", pct_thresholds)

    # Detail for best raw threshold
    for t in [3.0, 4.0, 5.0, 6.0]:
        mask = raw >= t
        s = eval_mask(events, mask, f"{raw_col}>={t}")
        inc = s["incremental"]
        if inc["n"] == 0: continue
        print(f"\n  Detail: {raw_col}>={t}")
        print(f"    big_losses: {json.dumps(s['inc_bl_rows'], ensure_ascii=False, default=str)}")
        if s['inc_bg_rows']:
            print(f"    BIG_GAINS_KILLED: {json.dumps(s['inc_bg_rows'], ensure_ascii=False, default=str)}")
        if s['inc_small_gain_rows']:
            print(f"    SMALL_GAINS_KILLED: {json.dumps(s['inc_small_gain_rows'], ensure_ascii=False, default=str)}")

    # ============================================================
    # TWO-FACTOR SCAN
    # ============================================================
    print(f"\n{'='*80}")
    print("TWO-FACTOR COMBO SCAN")
    print("Score = inc_big_loss - 3.0 * inc_big_gain8 - 0.5 * inc_small_gain")

    candidates = {
        "market_advance_decline_ratio": [("<=",0.30),("<=",0.35),("<=",0.38),("<=",0.45),("<=",0.50),
                                          (">=",1.5),(">=",2.0),(">=",3.0)],
        "top10_10d_strength_level_v3": [(">=",0.55),(">=",0.60),(">=",0.65),(">=",0.70),(">=",0.75)],
        "market_liquidity_change": [("<",0.90),("<",0.94),("<",0.98),("<",1.00)],
        "mkt_dispersion_shock_v2": [("<",0.90),("<",0.98),("<",1.05)],
        "runup_ma5_exhaustion": [(">",0.6),(">",0.7),(">",0.8)],
        "cpv_factor": [("<=",-0.8),("<=",-0.6),("<=",-0.4),(">=",0.4)],
        "prev_day_blowoff_risk": [(">=",1.0),(">=",2.0),(">=",3.0),(">=",5.0)],
        "sector_momentum_accel_rank": [("<=",0.15),("<=",0.30),("<=",0.50)],
        "ulcer_index": [("<",-1.2),("<",-0.8),("<",-0.4)],
        "momentum_peak_decay_3d": [(">",0.5),(">",1.0),(">",2.0)],
        "intra_sector_momentum_rank": [("<",0.30),("<",0.50),("<",0.67)],
        "buy_sm_amount_rate": [("<=",-1.5),("<=",-1.25),("<=",-1.0)],
    }

    def compare(s, op, v):
        s = s.astype(float)
        if op == "<=": return s <= v
        if op == "<": return s < v
        if op == ">=": return s >= v
        if op == ">": return s > v
        raise ValueError(op)

    def scan_twof(values, name, thresholds):
        rows = []
        for vt in thresholds:
            base = values >= vt
            for factor, tests in candidates.items():
                if factor not in events.columns: continue
                for op, val in tests:
                    mask = base & compare(events[factor], op, val)
                    s = eval_mask(events, mask, f"{name}>={vt}&{factor}{op}{val:g}")
                    inc = s["incremental"]
                    if inc["n"] == 0: continue
                    score = inc["big_loss<=-4%"] - 3.0 * inc["big_gain>=8%"] - 0.5 * inc["small_gain3-8%"]
                    rows.append({
                        "rule": s["label"], "inc_n": inc["n"], "inc_mean": inc["mean"],
                        "inc_big_loss": inc["big_loss<=-4%"], "inc_small_loss": inc["small_loss"],
                        "inc_small_gain": inc["small_gain3-8%"], "inc_big_gain8": inc["big_gain>=8%"],
                        "inc_big_gain10": inc["big_gain>=10%"],
                        "inc_actual_bl": s["inc_actual_big_loss"],
                        "inc_actual_bg": s["inc_actual_big_gain"],
                        "inc_shadow_bl": s["inc_shadow_big_loss"],
                        "inc_shadow_bg": s["inc_shadow_big_gain"],
                        "overlap": s["overlap_current"], "score": score,
                        "bl_rows": s["inc_bl_rows"], "bg_rows": s["inc_bg_rows"],
                        "sg_rows": s["inc_small_gain_rows"],
                    })
        return pd.DataFrame(rows).sort_values(["score","inc_big_loss","inc_big_gain8","inc_n"],
                                               ascending=[False,False,True,True])

    sc = ["rule","inc_n","inc_mean","inc_big_loss","inc_small_loss","inc_small_gain",
          "inc_big_gain8","inc_big_gain10","inc_actual_bl","inc_actual_bg",
          "inc_shadow_bl","inc_shadow_bg","overlap","score"]

    # Raw two-factor
    raw_2f_t = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]
    twof_raw = scan_twof(raw, raw_col, raw_2f_t)
    print(f"\n--- TWO-FACTOR (raw) TOP 20 ---")
    print(twof_raw.head(20)[sc].to_string(index=False))

    zbg_raw = twof_raw[(twof_raw["inc_big_gain8"]==0) & (twof_raw["inc_big_loss"]>0)]
    print(f"\n--- TWO-FACTOR (raw) ZERO BIG-GAIN ---")
    if not zbg_raw.empty:
        print(zbg_raw.head(20)[sc].to_string(index=False))
        for _, row in zbg_raw.head(5).iterrows():
            print(f"\n  Rule: {row['rule']} (inc_n={row['inc_n']}, inc_big_loss={row['inc_big_loss']}, score={row['score']:.1f})")
            print(f"    big_losses: {json.dumps(row['bl_rows'], ensure_ascii=False, default=str)}")
            if row['bg_rows']: print(f"    BIG_GAINS: {json.dumps(row['bg_rows'], ensure_ascii=False, default=str)}")
            if row['sg_rows']: print(f"    SMALL_GAINS: {json.dumps(row['sg_rows'], ensure_ascii=False, default=str)}")

    # Pct two-factor
    pct_2f_t = [0.75,0.80,0.85,0.90,0.95,0.97,0.98,0.99]
    twof_pct = scan_twof(pct, f"{raw_col}_pct", pct_2f_t)
    print(f"\n--- TWO-FACTOR (pct) TOP 20 ---")
    print(twof_pct.head(20)[sc].to_string(index=False))

    zbg_pct = twof_pct[(twof_pct["inc_big_gain8"]==0) & (twof_pct["inc_big_loss"]>0)]
    print(f"\n--- TWO-FACTOR (pct) ZERO BIG-GAIN ---")
    if not zbg_pct.empty:
        print(zbg_pct.head(20)[sc].to_string(index=False))
        for _, row in zbg_pct.head(5).iterrows():
            print(f"\n  Rule: {row['rule']} (inc_n={row['inc_n']}, inc_big_loss={row['inc_big_loss']}, score={row['score']:.1f})")
            print(f"    big_losses: {json.dumps(row['bl_rows'], ensure_ascii=False, default=str)}")
            if row['bg_rows']: print(f"    BIG_GAINS: {json.dumps(row['bg_rows'], ensure_ascii=False, default=str)}")
            if row['sg_rows']: print(f"    SMALL_GAINS: {json.dumps(row['sg_rows'], ensure_ascii=False, default=str)}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
