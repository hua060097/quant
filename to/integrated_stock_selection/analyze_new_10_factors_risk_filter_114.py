import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LOG_PATH = Path("stock_selection_test_mode_walkforward_分类模型_原因子114.log")
FACTOR_PATH = Path("factor_lab_data/factors.parquet")

NEW_FACTORS = [
    "market_advance_decline_ratio",
    "market_advance_decline_ratio_percentile_20d",
    "stock_volume_ratio_20d",
    "high_position_volume_surge",
    "upper_shadow_pressure_5d",
    "lower_shadow_absorption_5d",
    "shadow_reversal_volume_pressure",
    "prior_high_distance_20d",
    "prior_high_proximity_20d",
    "prior_high_volume_contest_20d",
]

CURRENT_RULE_FACTORS = [
    "close_below_ma5_after_runup",
    "turnover_z_20",
    "runup_ma5_exhaustion",
    "top10_10d_strength_level_v3",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
]


def parse_trades(log_path: Path) -> pd.DataFrame:
    buy_re = re.compile(
        r"买入成功:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*([\d.]+),\s*数量:\s*(\d+),\s*时间:\s*(\d{4}-\d{2}-\d{2})\s*([\d:]+)"
    )
    sell_re = re.compile(
        r"卖出:\s*(\d+\.(?:SZ|SH|BJ)),\s*价格:\s*([\d.]+),\s*收益:\s*([-\d.]+)\s*\(([-\d.]+)%\),\s*原因:\s*(.+)"
    )
    pending = {}
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = buy_re.search(line)
            if m:
                stock = m.group(1)
                pending.setdefault(stock, []).append(
                    {
                        "stock_code": stock,
                        "buy_price": float(m.group(2)),
                        "buy_qty": int(m.group(3)),
                        "buy_date": pd.Timestamp(m.group(4)).normalize(),
                        "buy_time": m.group(5),
                    }
                )
                continue
            m = sell_re.search(line)
            if m:
                stock = m.group(1)
                if stock not in pending or not pending[stock]:
                    continue
                buy = pending[stock].pop(0)
                rows.append(
                    {
                        **buy,
                        "sell_price": float(m.group(2)),
                        "return_abs": float(m.group(3)),
                        "return_pct": float(m.group(4)) / 100.0,
                        "sell_reason": m.group(5).strip(),
                    }
                )
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades
    trades["group"] = np.select(
        [
            trades["return_pct"] <= -0.05,
            trades["return_pct"] >= 0.15,
            trades["return_pct"] < 0,
            trades["return_pct"] >= 0,
        ],
        ["big_loss", "big_gain", "small_loss", "small_gain"],
        default="other",
    )
    return trades


def load_factor_slice(cols: list[str], min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    schema = pq.read_schema(FACTOR_PATH)
    available = [c for c in cols if c in schema.names]
    base_cols = ["date", "stock_code"]
    df = pd.read_parquet(FACTOR_PATH, columns=base_cols + available)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Need one prior trading day before first buy date.
    df = df[(df["date"] <= max_date) & (df["date"] >= min_date - pd.Timedelta(days=10))].copy()
    return df, available, [c for c in cols if c not in schema.names]


def attach_prediction_date(trades: pd.DataFrame, factor_dates: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(factor_dates).dt.normalize().unique()).sort_values().reset_index(drop=True)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    out = trades.copy()
    pred_dates = []
    for buy_date in out["buy_date"]:
        idx = np.searchsorted(date_values, np.datetime64(buy_date), side="left") - 1
        pred_dates.append(pd.Timestamp(date_values[idx]) if idx >= 0 else pd.NaT)
    out["pred_date"] = pred_dates
    return out


def summarize_groups(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for f in factors:
        valid = df[f].dropna()
        if valid.empty:
            continue
        row = {
            "factor": f,
            "n": int(df[f].notna().sum()),
            "nan_rate": float(df[f].isna().mean()),
            "corr": float(df[[f, "return_pct"]].dropna().corr().iloc[0, 1])
            if len(df[[f, "return_pct"]].dropna()) >= 3
            else np.nan,
        }
        for g in ["big_loss", "big_gain", "small_loss", "small_gain"]:
            vals = df.loc[df["group"] == g, f].dropna()
            row[f"{g}_n"] = int(len(vals))
            row[f"{g}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{g}_median"] = float(vals.median()) if len(vals) else np.nan
        row["loss_gain_diff"] = row["big_loss_mean"] - row["big_gain_mean"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("loss_gain_diff", ascending=False)


def eval_mask(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    mask = mask.fillna(False).astype(bool)
    blocked = df[mask]
    kept = df[~mask]
    big_loss_total = int((df["group"] == "big_loss").sum())
    big_gain_total = int((df["group"] == "big_gain").sum())
    return {
        "rule": name,
        "blocked": int(mask.sum()),
        "blocked_rate": float(mask.mean()),
        "blocked_big_loss": int((blocked["group"] == "big_loss").sum()),
        "blocked_big_gain": int((blocked["group"] == "big_gain").sum()),
        "blocked_small_loss": int((blocked["group"] == "small_loss").sum()),
        "blocked_small_gain": int((blocked["group"] == "small_gain").sum()),
        "big_loss_capture": int((blocked["group"] == "big_loss").sum()) / max(big_loss_total, 1),
        "big_gain_damage": int((blocked["group"] == "big_gain").sum()) / max(big_gain_total, 1),
        "kept_return_sum": float(kept["return_pct"].sum()),
        "blocked_return_sum": float(blocked["return_pct"].sum()),
    }


def threshold_search(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    big_loss_total = int((df["group"] == "big_loss").sum())
    for f in factors:
        s = df[f]
        valid = s.dropna()
        if len(valid) < 20:
            continue
        quantiles = sorted(set([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]))
        for q in quantiles:
            th = float(valid.quantile(q))
            for direction in [">=", "<="]:
                mask = s >= th if direction == ">=" else s <= th
                r = eval_mask(df, mask, f"{f} {direction} {th:.6g} (q={q:.2f})")
                if r["blocked"] == 0:
                    continue
                r.update({"factor": f, "direction": direction, "threshold": th, "q": q})
                # Prefer rules that catch loss, avoid big gain, and do not block too much.
                r["score"] = (
                    r["blocked_big_loss"] * 3.0
                    - r["blocked_big_gain"] * 2.5
                    - r["blocked_small_gain"] * 0.45
                    - max(0, r["blocked_rate"] - 0.35) * 10
                )
                if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.55:
                    rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def combo_search(df: pd.DataFrame, single_rules: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if single_rules.empty:
        return single_rules
    rows = []
    specs = single_rules.head(top_n)[["factor", "direction", "threshold", "rule"]].to_dict("records")
    for a, b in itertools.combinations(specs, 2):
        ma = df[a["factor"]] >= a["threshold"] if a["direction"] == ">=" else df[a["factor"]] <= a["threshold"]
        mb = df[b["factor"]] >= b["threshold"] if b["direction"] == ">=" else df[b["factor"]] <= b["threshold"]
        for op, mask in [("AND", ma & mb), ("OR", ma | mb)]:
            r = eval_mask(df, mask, f"({a['rule']}) {op} ({b['rule']})")
            if r["blocked_big_loss"] <= 0 or r["blocked_rate"] > 0.6:
                continue
            r["score"] = (
                r["blocked_big_loss"] * 3.0
                - r["blocked_big_gain"] * 2.5
                - r["blocked_small_gain"] * 0.45
                - max(0, r["blocked_rate"] - 0.35) * 10
            )
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def current_rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {}
    if {"close_below_ma5_after_runup", "turnover_z_20"}.issubset(df.columns):
        masks["114_default_close_below_ma5_and_turnover_z"] = (
            (df["close_below_ma5_after_runup"] == 1) & (df["turnover_z_20"] > 1.0)
        )
    if {"runup_ma5_exhaustion", "top10_10d_strength_level_v3"}.issubset(df.columns):
        masks["current_runup_rule"] = (
            (df["runup_ma5_exhaustion"] > 0.8) & (df["top10_10d_strength_level_v3"] >= 0.5)
        )
    if {"prev_day_blowoff_risk", "sector_momentum_accel_rank"}.issubset(df.columns):
        masks["current_blowoff_rule"] = (
            (df["prev_day_blowoff_risk"] >= 4.15) & (df["sector_momentum_accel_rank"] <= 0.41)
        )
    return masks


def main():
    trades = parse_trades(LOG_PATH)
    print(f"trades={len(trades)}")
    print(trades["group"].value_counts().to_string())

    needed = NEW_FACTORS + CURRENT_RULE_FACTORS
    factors, available, missing = load_factor_slice(needed, trades["buy_date"].min(), trades["buy_date"].max())
    print(f"\navailable={available}")
    print(f"missing={missing}")

    trades = attach_prediction_date(trades, factors["date"])
    merged = trades.merge(
        factors,
        left_on=["pred_date", "stock_code"],
        right_on=["date", "stock_code"],
        how="left",
    )

    usable_new = [f for f in NEW_FACTORS if f in merged.columns and merged[f].notna().any()]
    print(f"\nusable_new={usable_new}")
    print("\nGroup distribution by new factors:")
    summary = summarize_groups(merged, usable_new)
    print(summary[[
        "factor", "corr", "big_loss_n", "big_gain_n", "big_loss_mean", "big_gain_mean",
        "small_loss_mean", "small_gain_mean", "loss_gain_diff", "nan_rate"
    ]].to_string(index=False))

    print("\nCurrent/default rule effects:")
    base_masks = current_rule_masks(merged)
    base_rows = [eval_mask(merged, m, n) for n, m in base_masks.items()]
    if base_rows:
        print(pd.DataFrame(base_rows).to_string(index=False))

    print("\nBest single new-factor rules:")
    single = threshold_search(merged, usable_new)
    if single.empty:
        print("no single rule")
    else:
        cols = [
            "rule", "blocked", "blocked_big_loss", "blocked_big_gain",
            "blocked_small_loss", "blocked_small_gain", "big_loss_capture",
            "big_gain_damage", "blocked_return_sum", "kept_return_sum", "score"
        ]
        print(single[cols].head(30).to_string(index=False))

    print("\nBest two-factor combinations among new factors:")
    combos = combo_search(merged, single)
    if combos.empty:
        print("no combo")
    else:
        cols = [
            "rule", "blocked", "blocked_big_loss", "blocked_big_gain",
            "blocked_small_loss", "blocked_small_gain", "big_loss_capture",
            "big_gain_damage", "blocked_return_sum", "kept_return_sum", "score"
        ]
        print(combos[cols].head(30).to_string(index=False))

    print("\nComplement with default/current rules:")
    rows = []
    if not single.empty:
        for base_name, base_mask in base_masks.items():
            for spec in single.head(20)[["factor", "direction", "threshold", "rule"]].to_dict("records"):
                m = merged[spec["factor"]] >= spec["threshold"] if spec["direction"] == ">=" else merged[spec["factor"]] <= spec["threshold"]
                rows.append(eval_mask(merged, base_mask | m, f"{base_name} OR {spec['rule']}"))
                rows.append(eval_mask(merged, base_mask & m, f"{base_name} AND {spec['rule']}"))
    comp = pd.DataFrame(rows)
    if comp.empty:
        print("no complement")
    else:
        comp["score"] = (
            comp["blocked_big_loss"] * 3.0
            - comp["blocked_big_gain"] * 2.5
            - comp["blocked_small_gain"] * 0.45
            - np.maximum(0, comp["blocked_rate"] - 0.35) * 10
        )
        comp = comp[(comp["blocked_big_loss"] > 0) & (comp["blocked_rate"] <= 0.6)]
        cols = [
            "rule", "blocked", "blocked_big_loss", "blocked_big_gain",
            "blocked_small_loss", "blocked_small_gain", "big_loss_capture",
            "big_gain_damage", "blocked_return_sum", "kept_return_sum", "score"
        ]
        print(comp.sort_values(["score", "blocked_big_loss"], ascending=[False, False])[cols].head(30).to_string(index=False))

    print("\nBig loss/new factor detail:")
    detail_cols = ["stock_code", "buy_date", "pred_date", "return_pct", "sell_reason"] + usable_new
    print(merged.loc[merged["group"] == "big_loss", detail_cols].sort_values("return_pct").to_string(index=False))


if __name__ == "__main__":
    main()
