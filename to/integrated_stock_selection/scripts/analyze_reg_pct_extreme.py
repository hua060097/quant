"""reg_severe_up_10d_ratio 横截面极端分位精确分析。
使用 0.98, 0.99, 0.992, 0.995, 0.998+ 等极端阈值。
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
    "score_z_in_day", "reg_severe_up_10d_ratio",
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
    return {
        "n": int(len(df)), "mean": float(r.mean()) if len(df) else np.nan,
        "big_loss<=-4%": int((r <= -0.04).sum()),
        "small_loss": int(((r > -0.04) & (r < 0)).sum()),
        "small_gain3-8%": int(((r >= 0.03) & (r < 0.08)).sum()),
        "big_gain>=8%": int((r >= 0.08).sum()),
        "big_gain>=10%": int((r >= 0.10).sum()),
    }


def eval_mask(df, mask, label=""):
    mask = mask.fillna(False)
    cur = df["current_v3"].fillna(False)
    inc = mask & ~cur
    inc_rows = df[inc]
    bl = inc_rows["actual_return"] <= -0.04
    bg = inc_rows["actual_return"] >= 0.08
    sg = (inc_rows["actual_return"] >= 0.03) & (inc_rows["actual_return"] < 0.08)
    return {
        "label": label,
        "standalone": bucket_counts(df[mask]),
        "incremental": bucket_counts(inc_rows),
        "overlap_current": int((mask & cur).sum()),
        "inc_actual_big_loss": int(((inc_rows["source"] == "actual") & bl).sum()),
        "inc_actual_big_gain": int(((inc_rows["source"] == "actual") & bg).sum()),
        "inc_shadow_big_loss": int(((inc_rows["source"] == "shadow") & bl).sum()),
        "inc_shadow_big_gain": int(((inc_rows["source"] == "shadow") & bg).sum()),
        "inc_small_gain_rows": inc_rows.loc[sg, [
            "source", "date", "stock_code", "actual_return",
            "reg_severe_up_10d_ratio", "reg_severe_up_10d_ratio_pct",
            "market_advance_decline_ratio"
        ]].to_dict("records"),
        "inc_big_gain_rows": inc_rows.loc[bg, [
            "source", "date", "stock_code", "actual_return",
            "reg_severe_up_10d_ratio", "reg_severe_up_10d_ratio_pct",
            "market_advance_decline_ratio"
        ]].sort_values("actual_return", ascending=False).to_dict("records"),
        "inc_big_loss_rows": inc_rows.loc[bl, [
            "source", "date", "stock_code", "actual_return",
            "reg_severe_up_10d_ratio", "reg_severe_up_10d_ratio_pct",
            "market_advance_decline_ratio"
        ]].sort_values("actual_return").to_dict("records"),
    }


def main():
    print("Loading data...")
    selections = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    events = pd.concat([selections, shadow], ignore_index=True).drop_duplicates(
        ["source", "date", "stock_code", "actual_return"])

    schema_cols = set(pq.ParquetFile(FACTORS).schema.names)
    factor_cols = ["date", "stock_code"] + [c for c in RULE_COLS if c in schema_cols]
    factors = pd.read_parquet(FACTORS, columns=factor_cols)
    for col in RULE_COLS:
        if col not in factors.columns: factors[col] = np.nan
    factors["date"] = pd.to_datetime(factors["date"])

    reg_col = "reg_severe_up_10d_ratio"
    pct_col = "reg_severe_up_10d_ratio_pct"
    factors[pct_col] = factors.groupby("date")[reg_col].transform(lambda x: x.rank(pct=True))

    trading_days = np.array(sorted(factors["date"].dropna().unique()))
    prev_dates = []
    for d in events["date"]:
        trade_date = pd.Timestamp(d).normalize()
        pos = np.searchsorted(trading_days, np.datetime64(trade_date), side="left") - 1
        prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)
    events["factor_date"] = prev_dates
    snap = factors.rename(columns={"date": "factor_date"})
    events = events.merge(snap, on=["factor_date", "stock_code"], how="left")
    masks = current_v3_masks(events)
    events = pd.concat([events, masks], axis=1)

    reg_pct = events[pct_col]
    reg_raw = events[reg_col]
    adr = events["market_advance_decline_ratio"]

    # ============================================================
    # 1. 极高 reg_pct 单阈值扫描 (0.98-0.999)
    # ============================================================
    print("=" * 80)
    print("EXTREME reg_pct SINGLE THRESHOLD SCAN")
    print(f"Events median reg_pct: {reg_pct.median():.4f}")
    print(f"Events p75: {reg_pct.quantile(0.75):.4f}, p90: {reg_pct.quantile(0.90):.4f}, p95: {reg_pct.quantile(0.95):.4f}")
    print()

    for t in [0.980, 0.985, 0.988, 0.990, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.9985, 0.999]:
        mask = reg_pct >= t
        s = eval_mask(events, mask, f"reg_pct>={t:.4f}")
        inc = s["incremental"]
        score = inc["big_loss<=-4%"] - 2.5 * inc["big_gain>=8%"]
        print(f"  reg_pct>={t:.4f}: standalone_n={s['standalone']['n']:3d} inc_n={inc['n']:2d} "
              f"inc_big_loss={inc['big_loss<=-4%']} inc_big_gain8={inc['big_gain>=8%']} "
              f"inc_small_gain={inc['small_gain3-8%']} inc_mean={inc['mean']:.4f} "
              f"overlap={s['overlap_current']} score={score:.1f}")

        # Detail for best thresholds
        if t in [0.990, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998]:
            print(f"    big_losses: {json.dumps(s['inc_big_loss_rows'], ensure_ascii=False, default=str)}")
            if s['inc_big_gain_rows']:
                print(f"    BIG_GAINS_KILLED: {json.dumps(s['inc_big_gain_rows'], ensure_ascii=False, default=str)}")
            if s['inc_small_gain_rows']:
                print(f"    SMALL_GAINS_KILLED: {json.dumps(s['inc_small_gain_rows'], ensure_ascii=False, default=str)}")

    # ============================================================
    # 2. reg_pct + adr 组合 (极端pct)
    # ============================================================
    print("\n" + "=" * 80)
    print("EXTREME reg_pct + ADR TWO-FACTOR SCAN")
    print()

    for pt in [0.980, 0.985, 0.988, 0.990, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.9985, 0.999]:
        for adr_t in [0.30, 0.35, 0.38, 0.45, 0.50]:
            mask = (reg_pct >= pt) & (adr <= adr_t)
            s = eval_mask(events, mask, f"reg_pct>={pt:.4f}&adr<={adr_t:.2f}")
            inc = s["incremental"]
            if inc["n"] == 0:
                continue
            score = inc["big_loss<=-4%"] - 3.0 * inc["big_gain>=8%"] - 0.5 * inc["small_gain3-8%"]
            bg_kill = inc["big_gain>=8%"]
            sg_kill = inc["small_gain3-8%"]
            bl = inc["big_loss<=-4%"]
            print(f"  reg_pct>={pt:.4f} & adr<={adr_t:.2f}: "
                  f"inc_n={inc['n']:2d} inc_big_loss={bl} inc_big_gain8={bg_kill} "
                  f"inc_small_gain={sg_kill} inc_mean={inc['mean']:.4f} "
                  f"overlap={s['overlap_current']} score={score:.1f}")
            if bl > 0 and bg_kill == 0 and sg_kill == 0:
                print(f"    *** ZERO FALSE KILL *** big_losses: {json.dumps(s['inc_big_loss_rows'], ensure_ascii=False, default=str)}")
            if bg_kill > 0:
                print(f"    BIG_GAINS: {json.dumps(s['inc_big_gain_rows'], ensure_ascii=False, default=str)}")
            if sg_kill > 0:
                print(f"    SMALL_GAINS: {json.dumps(s['inc_small_gain_rows'], ensure_ascii=False, default=str)}")

    # ============================================================
    # 3. reg_pct + adr + 第三个因子 (极端pct)
    # ============================================================
    print("\n" + "=" * 80)
    print("EXTREME reg_pct + ADR + THIRD FACTOR (three-factor)")
    print()

    third_factors = {
        "top10_10d_strength_level_v3": [(">=", 0.60), (">=", 0.65), (">=", 0.70)],
        "buy_sm_amount_rate": [("<=", -1.0), ("<=", -1.25)],
        "market_liquidity_change": [("<", 0.94), ("<", 0.98)],
        "mkt_dispersion_shock_v2": [("<", 0.90), ("<", 0.98)],
        "runup_ma5_exhaustion": [(">", 0.7), (">", 0.8)],
    }

    def compare(s, op, v):
        s = s.astype(float)
        if op == "<=": return s <= v
        if op == "<": return s < v
        if op == ">=": return s >= v
        if op == ">": return s > v
        raise ValueError(op)

    for pt in [0.990, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]:
        for adr_t in [0.35, 0.38, 0.45]:
            base_mask = (reg_pct >= pt) & (adr <= adr_t)
            for fname, tests in third_factors.items():
                if fname not in events.columns: continue
                for op, val in tests:
                    mask = base_mask & compare(events[fname], op, val)
                    s = eval_mask(events, mask, f"reg_pct>={pt:.4f}&adr<={adr_t:.2f}&{fname}{op}{val:g}")
                    inc = s["incremental"]
                    if inc["n"] == 0: continue
                    bl = inc["big_loss<=-4%"]
                    bg = inc["big_gain>=8%"]
                    sg = inc["small_gain3-8%"]
                    score = bl - 3.0 * bg - 0.5 * sg
                    if bl > 0 and bg == 0 and sg == 0:
                        print(f"  *** ZERO FALSE KILL *** reg_pct>={pt:.4f}&adr<={adr_t:.2f}&{fname}{op}{val:g}: "
                              f"inc_n={inc['n']} inc_big_loss={bl} overlap={s['overlap_current']} score={score:.1f}")
                        print(f"    {json.dumps(s['inc_big_loss_rows'], ensure_ascii=False, default=str)}")

    # ============================================================
    # 4. 直接列出 reg_pct>=0.99 且未过滤的所有事件
    # ============================================================
    print("\n" + "=" * 80)
    print("ALL UNFILTERED EVENTS WITH reg_pct >= 0.99")
    uf_high = events[(~events["current_v3"]) & (reg_pct >= 0.99)]
    cols = ["source", "date", "stock_code", "actual_return", reg_col, pct_col, "market_advance_decline_ratio"]
    uf_high = uf_high.sort_values("actual_return")
    for _, r in uf_high.iterrows():
        tag = ""
        if r["actual_return"] <= -0.04: tag = " ← BIG LOSS"
        elif r["actual_return"] >= 0.08: tag = " ← BIG GAIN"
        elif r["actual_return"] >= 0.03: tag = " ← SMALL GAIN"
        print(f"  {r['source']:8s} {r['date']} {r['stock_code']:12s} ret={r['actual_return']:+.4f} "
              f"reg_raw={r[reg_col]:.4f} reg_pct={r[pct_col]:.4f} adr={r['market_advance_decline_ratio']:.4f}{tag}")

    # Also show the distribution tail
    print(f"\n  Total unfiltered with reg_pct>=0.99: {len(uf_high)}")
    print(f"  Big losses: {int((uf_high['actual_return']<=-0.04).sum())}")
    print(f"  Big gains: {int((uf_high['actual_return']>=0.08).sum())}")
    print(f"  Small gains (3-8%): {int(((uf_high['actual_return']>=0.03)&(uf_high['actual_return']<0.08)).sum())}")

    # Compare with raw threshold
    print("\n" + "=" * 80)
    print("COMPARISON: reg_raw >= 0.18 vs extreme reg_pct")
    print()
    for rule_label, mask in [
        ("reg_raw>=0.18 & adr<=0.45", (reg_raw >= 0.18) & (adr <= 0.45)),
    ]:
        s = eval_mask(events, mask, rule_label)
        inc = s["incremental"]
        print(f"  {rule_label}:")
        print(f"    inc_n={inc['n']}, inc_big_loss={inc['big_loss<=-4%']}, "
              f"inc_big_gain8={inc['big_gain>=8%']}, inc_small_gain={inc['small_gain3-8%']}, "
              f"inc_mean={inc['mean']:.4f}, overlap={s['overlap_current']}")
        print(f"    big_losses: {json.dumps(s['inc_big_loss_rows'], ensure_ascii=False, default=str)}")

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
