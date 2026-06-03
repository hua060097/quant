import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS_PATH = Path("results/walkforward_metrics_20260524_203930.json")
AFTER_DAILY_PATH = Path("results/candidate_new_meta_factors_after_three_rule_v3_daily.csv")
BASE_DAILY_PATH = Path("results/candidate_new_meta_factors_base_tradable_no_rule_daily.csv")
OUT_PREFIX = Path("results/score_z_in_day_rule_audit")


def bucket_counts(s: pd.Series) -> dict:
    return {
        "n": int(len(s)),
        "big_loss_<=-4": int((s <= -0.04).sum()),
        "small_loss_-4~0": int(((s > -0.04) & (s < 0)).sum()),
        "flat_0~3": int(((s >= 0) & (s < 0.03)).sum()),
        "small_gain_3~8": int(((s >= 0.03) & (s < 0.08)).sum()),
        "big_gain_>=8": int((s >= 0.08).sum()),
        "big_gain_strict_>=10": int((s >= 0.10).sum()),
        "mean_return": float(s.mean()) if len(s) else np.nan,
        "median_return": float(s.median()) if len(s) else np.nan,
    }


def summarize_hit(df: pd.DataFrame, hit: pd.Series, value_col: str = "actual_return") -> dict:
    hit = hit.fillna(False)
    sub = df.loc[hit]
    keep = df.loc[~hit]
    out = {
        "hit": bucket_counts(sub[value_col]),
        "keep": bucket_counts(keep[value_col]),
        "hit_codes": ";".join(
            f"{r.stock_code}@{r.signal_date}:{r[value_col]:+.2%}"
            for _, r in sub.sort_values(value_col).iterrows()
        ),
    }
    out["hit"]["loss_minus_big_gain"] = out["hit"]["big_loss_<=-4"] - out["hit"]["big_gain_>=8"]
    out["hit"]["loss_to_big_gain_ratio"] = out["hit"]["big_loss_<=-4"] / (
        out["hit"]["big_gain_>=8"] + 1
    )
    return out


def signal_date_from_buy(buy_dates: pd.Series, valid_signal_dates: list[pd.Timestamp]) -> pd.Series:
    signal_dates = pd.Series(pd.to_datetime(valid_signal_dates)).sort_values().drop_duplicates()
    buy = pd.to_datetime(buy_dates).dt.normalize()
    idx = np.searchsorted(signal_dates.values, buy.values, side="left") - 1
    out = []
    for i in idx:
        out.append(signal_dates.iloc[i].strftime("%Y-%m-%d") if i >= 0 else None)
    return pd.Series(out, index=buy_dates.index)


def load_metrics_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))[0]["all_results"]["1"]
    trades = pd.DataFrame(metrics["trades"])
    shadow = pd.DataFrame(metrics["shadow_skipped_opportunity_trades"])

    for df in [trades, shadow]:
        if df.empty:
            continue
        df["buy_date"] = pd.to_datetime(df["buy_datetime"]).dt.strftime("%Y-%m-%d")
        df["actual_return"] = df["actual_return"].astype(float)
        df["predicted_return"] = df["predicted_return"].astype(float)
    return trades, shadow


def attach_score_z(
    df: pd.DataFrame,
    after_daily: pd.DataFrame,
    base_daily: pd.DataFrame,
    prefer_shadow_prediction_date: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    all_signal_dates = sorted(set(base_daily["date"]) | set(after_daily["date"]))
    if prefer_shadow_prediction_date and "shadow_prediction_date" in out.columns:
        out["signal_date"] = pd.to_datetime(out["shadow_prediction_date"]).dt.strftime("%Y-%m-%d")
    else:
        out["signal_date"] = signal_date_from_buy(out["buy_datetime"], all_signal_dates)

    after_cols = ["date", "top1_code", "score_z_in_day", "top1_return", "top1_pred"]
    base_cols = ["date", "top1_code", "score_z_in_day", "top1_return", "top1_pred"]
    after_key = after_daily[after_cols].rename(
        columns={
            "date": "signal_date",
            "top1_code": "stock_code",
            "score_z_in_day": "score_z_after",
            "top1_return": "label_return_after",
            "top1_pred": "top1_pred_after",
        }
    )
    base_key = base_daily[base_cols].rename(
        columns={
            "date": "signal_date",
            "top1_code": "stock_code",
            "score_z_in_day": "score_z_base",
            "top1_return": "label_return_base",
            "top1_pred": "top1_pred_base",
        }
    )
    out = out.merge(after_key, on=["signal_date", "stock_code"], how="left")
    out = out.merge(base_key, on=["signal_date", "stock_code"], how="left")
    out["score_z_in_day"] = out["score_z_after"].combine_first(out["score_z_base"])
    out["score_source"] = np.where(out["score_z_after"].notna(), "after_current_v3", "base_pre_v3")
    return out


def threshold_grid(df: pd.DataFrame, label: str, value_col: str = "actual_return") -> pd.DataFrame:
    rows = []
    valid = df[df["score_z_in_day"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    abs_thresholds = [1.40, 1.50, 1.60, 1.70, 1.75, 1.80, 1.85, 1.90, 2.00, 2.10, 2.20]
    q_thresholds = []
    for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        q_thresholds.append((f"q{q:.2f}", float(valid["score_z_in_day"].quantile(q))))

    for tag, thr in [(f"abs_{t:.2f}", t) for t in abs_thresholds] + q_thresholds:
        hit = valid["score_z_in_day"] <= thr
        rec = summarize_hit(valid, hit, value_col=value_col)
        rows.append(
            {
                "sample": label,
                "threshold_tag": tag,
                "threshold": thr,
                "hit_n": rec["hit"]["n"],
                "hit_big_loss": rec["hit"]["big_loss_<=-4"],
                "hit_small_loss": rec["hit"]["small_loss_-4~0"],
                "hit_flat_0_3": rec["hit"]["flat_0~3"],
                "hit_small_gain_3_8": rec["hit"]["small_gain_3~8"],
                "hit_big_gain": rec["hit"]["big_gain_>=8"],
                "hit_big_gain_strict": rec["hit"]["big_gain_strict_>=10"],
                "hit_mean": rec["hit"]["mean_return"],
                "keep_n": rec["keep"]["n"],
                "keep_big_loss": rec["keep"]["big_loss_<=-4"],
                "keep_big_gain": rec["keep"]["big_gain_>=8"],
                "keep_mean": rec["keep"]["mean_return"],
                "loss_minus_big_gain": rec["hit"]["loss_minus_big_gain"],
                "loss_to_big_gain_ratio": rec["hit"]["loss_to_big_gain_ratio"],
                "hit_codes": rec["hit_codes"],
            }
        )
    return pd.DataFrame(rows)


def monotonic_bins(df: pd.DataFrame, label: str, value_col: str = "actual_return", bins: int = 5) -> pd.DataFrame:
    valid = df[df["score_z_in_day"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["bin"] = pd.qcut(valid["score_z_in_day"], q=min(bins, valid["score_z_in_day"].nunique()), duplicates="drop")
    rows = []
    for b, g in valid.groupby("bin", observed=True):
        rec = bucket_counts(g[value_col])
        rec.update(
            {
                "sample": label,
                "bin": str(b),
                "score_min": float(g["score_z_in_day"].min()),
                "score_max": float(g["score_z_in_day"].max()),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def daily_threshold_grid(daily: pd.DataFrame, label: str) -> pd.DataFrame:
    work = daily.rename(columns={"top1_return": "actual_return", "top1_code": "stock_code"}).copy()
    work["signal_date"] = work["date"]
    return threshold_grid(work, label)


def daily_monotonic(daily: pd.DataFrame, label: str) -> pd.DataFrame:
    work = daily.rename(columns={"top1_return": "actual_return", "top1_code": "stock_code"}).copy()
    return monotonic_bins(work, label)


def main() -> None:
    after_daily = pd.read_csv(AFTER_DAILY_PATH)
    base_daily = pd.read_csv(BASE_DAILY_PATH)
    trades, shadow = load_metrics_frames()

    trades_a = attach_score_z(trades, after_daily, base_daily)
    shadow_a = attach_score_z(shadow, after_daily, base_daily, prefer_shadow_prediction_date=True)

    shadow_pass = shadow_a[~shadow_a.get("exhaustion_blocked", False).astype(bool)].copy()
    shadow_blocked = shadow_a[shadow_a.get("exhaustion_blocked", False).astype(bool)].copy()
    combined_pass = pd.concat([trades_a, shadow_pass], ignore_index=True)

    frames = {
        "daily_after_current_v3": after_daily.rename(columns={"top1_return": "actual_return", "top1_code": "stock_code"}),
        "actual_trades": trades_a,
        "shadow_current_pass": shadow_pass,
        "actual_plus_shadow_current_pass": combined_pass,
        "shadow_current_blocked_overlap": shadow_blocked,
    }

    grids = []
    bins = []
    summaries = {}
    for name, df in frames.items():
        if df.empty:
            continue
        if "signal_date" not in df.columns:
            df = df.copy()
            df["signal_date"] = df.get("date", None)
        grids.append(threshold_grid(df, name))
        bins.append(monotonic_bins(df, name))
        valid = df[df["score_z_in_day"].notna()]
        summaries[name] = {
            "n": int(len(df)),
            "score_attached_n": int(len(valid)),
            "base": bucket_counts(valid["actual_return"]) if len(valid) else {},
            "score_quantiles": {
                str(q): float(valid["score_z_in_day"].quantile(q))
                for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50]
            }
            if len(valid)
            else {},
        }

    grid_df = pd.concat(grids, ignore_index=True)
    bin_df = pd.concat(bins, ignore_index=True)
    grid_df.to_csv(f"{OUT_PREFIX}_threshold_grid.csv", index=False, encoding="utf-8-sig")
    bin_df.to_csv(f"{OUT_PREFIX}_monotonic_bins.csv", index=False, encoding="utf-8-sig")
    trades_a.to_csv(f"{OUT_PREFIX}_actual_trades_attached.csv", index=False, encoding="utf-8-sig")
    shadow_a.to_csv(f"{OUT_PREFIX}_shadow_attached.csv", index=False, encoding="utf-8-sig")

    interesting = grid_df[
        grid_df["sample"].isin(["daily_after_current_v3", "actual_trades", "shadow_current_pass", "actual_plus_shadow_current_pass"])
    ].copy()
    interesting = interesting[
        interesting["threshold_tag"].isin(["abs_1.50", "abs_1.60", "abs_1.70", "abs_1.80", "abs_1.85", "abs_1.90", "q0.10", "q0.15", "q0.20"])
    ]
    interesting.to_csv(f"{OUT_PREFIX}_interesting_thresholds.csv", index=False, encoding="utf-8-sig")

    out = {
        "metrics_path": str(METRICS_PATH),
        "summaries": summaries,
        "interesting_thresholds": interesting.drop(columns=["hit_codes"], errors="ignore").to_dict("records"),
    }
    Path(f"{OUT_PREFIX}_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
