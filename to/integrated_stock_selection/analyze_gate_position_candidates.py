from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FACTOR_PATH = BASE_DIR / "factor_lab_data" / "factors.parquet"
JSON_PATH = BASE_DIR / "results" / "daily_top5_details_Window_3_20260325_230521.json"

TRAIN_START = pd.Timestamp("2023-06-01")
TRAIN_END = pd.Timestamp("2025-03-01")

STATE_FACTORS = [
    "mkt_ma5_followthrough_v3",
    "ma5_strength_breadth_v3",
    "strong_stock_volume_confirmation_v3",
    "top10_10d_strength_level_v3",
    "top10_10d_today_return_level_v3",
    "top10_10d_recent3_strength_v3",
    "top10_5d_recent3_strength_v3",
    "top10_3d_recent3_strength_v3",
    "emotion_quality_balance_v3",
    "high_to_close_drawdown",
]

QUANTILES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_EXPANDING_HISTORY = 20
CALIBRATION_RATIO = 0.6


def max_drawdown(returns: pd.Series) -> float:
    x = pd.Series(returns).dropna().astype(float)
    if x.empty:
        return float("nan")
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def summarize(returns: pd.Series) -> dict[str, float]:
    x = pd.Series(returns).dropna().astype(float)
    if x.empty:
        return {
            "days": 0,
            "mean": np.nan,
            "cum": np.nan,
            "win": np.nan,
            "mdd": np.nan,
        }
    return {
        "days": int(len(x)),
        "mean": float(x.mean()),
        "cum": float((1.0 + x).prod() - 1.0),
        "win": float((x > 0).mean()),
        "mdd": max_drawdown(x),
    }


def load_top5_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in raw.items():
        row = {"date": pd.to_datetime(ds)}
        for k in range(1, 6):
            info = daily[f"top{k}"]
            row[f"pred{k}"] = float(info["pred"])
            row[f"ret{k}"] = float(info["real_return"])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["pred_gap_12"] = df["pred1"] - df["pred2"]
    df["pred_gap_1_vs_25"] = df["pred1"] - df[["pred2", "pred3", "pred4", "pred5"]].mean(axis=1)
    return df


def load_daily_factors() -> pd.DataFrame:
    cols = ["date"] + STATE_FACTORS
    df = pd.read_parquet(FACTOR_PATH, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date", as_index=False).first().sort_values("date").reset_index(drop=True)


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    top = load_top5_json(JSON_PATH)
    factors = load_daily_factors()
    df = top.merge(factors, on="date", how="left").sort_values("date").reset_index(drop=True)

    train_daily = factors[(factors["date"] >= TRAIN_START) & (factors["date"] <= TRAIN_END)].copy()
    q40 = float(train_daily["mkt_ma5_followthrough_v3"].quantile(0.4))
    df["q40_target_rank"] = np.where(df["mkt_ma5_followthrough_v3"] >= q40, 1, 2)
    df["ret_q40"] = np.where(df["q40_target_rank"] == 1, df["ret1"], df["ret2"])
    return df, train_daily


def threshold_series_fixed(df: pd.DataFrame, train_daily: pd.DataFrame, signal: str, q: float) -> pd.Series:
    thr = float(train_daily[signal].quantile(q))
    return pd.Series(thr, index=df.index, dtype=float)


def threshold_series_expanding(df: pd.DataFrame, signal: str, q: float) -> pd.Series:
    s = df[signal].astype(float)
    thr = s.expanding(min_periods=MIN_EXPANDING_HISTORY).quantile(q).shift(1)
    return thr


def apply_rule(
    base_returns: pd.Series,
    signal_values: pd.Series,
    thresholds: pd.Series,
    orientation: str,
    mode: str,
) -> pd.Series:
    favorable = pd.Series(True, index=base_returns.index)
    valid = thresholds.notna() & signal_values.notna()
    if orientation == "high":
        favorable.loc[valid] = signal_values.loc[valid] >= thresholds.loc[valid]
    else:
        favorable.loc[valid] = signal_values.loc[valid] <= thresholds.loc[valid]

    if mode == "gate":
        return base_returns.where(favorable, 0.0)
    if mode == "half":
        return base_returns.where(favorable, base_returns * 0.5)
    raise ValueError(mode)


def split_blocks(df: pd.DataFrame, n_blocks: int = 4) -> list[pd.Index]:
    blocks = np.array_split(np.arange(len(df)), n_blocks)
    return [pd.Index(block) for block in blocks if len(block) > 0]


def stability_score(strategy_returns: pd.Series, baseline_returns: pd.Series, blocks: list[pd.Index]) -> tuple[int, int]:
    wins = 0
    valid = 0
    for block in blocks:
        s = summarize(strategy_returns.iloc[block])
        b = summarize(baseline_returns.iloc[block])
        if np.isnan(s["mean"]) or np.isnan(b["mean"]):
            continue
        valid += 1
        if s["mean"] > b["mean"]:
            wins += 1
    return wins, valid


def evaluate_candidates(df: pd.DataFrame, train_daily: pd.DataFrame) -> pd.DataFrame:
    calibration_end = int(len(df) * CALIBRATION_RATIO)
    blocks = split_blocks(df, 4)

    bases = {
        "top1": df["ret1"],
        "q40": df["ret_q40"],
    }

    signal_methods: list[tuple[str, str]] = []
    for signal in STATE_FACTORS:
        signal_methods.append((signal, "fixed_train"))
        signal_methods.append((signal, "expanding"))
    for signal in ["pred_gap_12", "pred_gap_1_vs_25"]:
        signal_methods.append((signal, "expanding"))

    rows: list[dict[str, object]] = []
    for base_name, base_returns in bases.items():
        baseline_full = summarize(base_returns)
        baseline_cal = summarize(base_returns.iloc[:calibration_end])
        baseline_val = summarize(base_returns.iloc[calibration_end:])

        for signal, method in signal_methods:
            if signal not in df.columns:
                continue
            signal_values = df[signal].astype(float)
            if signal_values.notna().sum() == 0:
                continue

            best_per_mode: dict[str, dict[str, object] | None] = {"gate": None, "half": None}
            for q in QUANTILES:
                if method == "fixed_train":
                    thresholds = threshold_series_fixed(df, train_daily, signal, q)
                else:
                    thresholds = threshold_series_expanding(df, signal, q)

                for orientation in ["high", "low"]:
                    for mode in ["gate", "half"]:
                        strategy_returns = apply_rule(base_returns, signal_values, thresholds, orientation, mode)
                        cal_stats = summarize(strategy_returns.iloc[:calibration_end])
                        val_stats = summarize(strategy_returns.iloc[calibration_end:])
                        full_stats = summarize(strategy_returns)
                        wins, valid = stability_score(strategy_returns, base_returns, blocks)
                        record = {
                            "base": base_name,
                            "signal": signal,
                            "method": method,
                            "mode": mode,
                            "q": q,
                            "orientation": orientation,
                            "cal_mean": cal_stats["mean"],
                            "cal_cum": cal_stats["cum"],
                            "cal_mdd": cal_stats["mdd"],
                            "val_mean": val_stats["mean"],
                            "val_cum": val_stats["cum"],
                            "val_mdd": val_stats["mdd"],
                            "full_mean": full_stats["mean"],
                            "full_cum": full_stats["cum"],
                            "full_win": full_stats["win"],
                            "full_mdd": full_stats["mdd"],
                            "block_wins": wins,
                            "block_total": valid,
                            "baseline_full_mean": baseline_full["mean"],
                            "baseline_full_cum": baseline_full["cum"],
                            "baseline_full_mdd": baseline_full["mdd"],
                            "baseline_cal_cum": baseline_cal["cum"],
                            "baseline_val_cum": baseline_val["cum"],
                        }

                        current = best_per_mode[mode]
                        if current is None:
                            best_per_mode[mode] = record
                        else:
                            better = False
                            if record["cal_cum"] > current["cal_cum"] + 1e-12:
                                better = True
                            elif abs(record["cal_cum"] - current["cal_cum"]) <= 1e-12:
                                if record["cal_mdd"] > current["cal_mdd"]:
                                    better = True
                            if better:
                                best_per_mode[mode] = record

            for mode in ["gate", "half"]:
                if best_per_mode[mode] is not None:
                    rows.append(best_per_mode[mode])

    return pd.DataFrame(rows)


def print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("EMPTY")
        return
    out = df.copy()
    cols = [
        "base",
        "mode",
        "signal",
        "method",
        "q",
        "orientation",
        "cal_cum",
        "val_cum",
        "val_mean",
        "val_mdd",
        "full_cum",
        "full_mean",
        "full_mdd",
        "block_wins",
        "block_total",
        "baseline_full_cum",
        "baseline_full_mean",
        "baseline_full_mdd",
    ]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    df, train_daily = build_dataset()
    print(
        f"sample_days={len(df)} range={df['date'].min().date()} ~ {df['date'].max().date()} "
        f"calibration={df['date'].iloc[0].date()} ~ {df['date'].iloc[int(len(df) * CALIBRATION_RATIO) - 1].date()} "
        f"validation={df['date'].iloc[int(len(df) * CALIBRATION_RATIO)].date()} ~ {df['date'].iloc[-1].date()}"
    )
    print(
        "BASE top1 "
        f"mean={summarize(df['ret1'])['mean']:.3%} cum={summarize(df['ret1'])['cum']:.4f} "
        f"win={summarize(df['ret1'])['win']:.1%} mdd={summarize(df['ret1'])['mdd']:.3%}"
    )
    print(
        "BASE q40  "
        f"mean={summarize(df['ret_q40'])['mean']:.3%} cum={summarize(df['ret_q40'])['cum']:.4f} "
        f"win={summarize(df['ret_q40'])['win']:.1%} mdd={summarize(df['ret_q40'])['mdd']:.3%}"
    )

    results = evaluate_candidates(df, train_daily)
    results.to_csv(BASE_DIR / "gate_position_candidates_20260325_230521.csv", index=False, encoding="utf-8-sig")

    for base in ["top1", "q40"]:
        base_df = results[results["base"] == base].copy()
        gate = base_df[base_df["mode"] == "gate"].sort_values(["val_cum", "full_cum"], ascending=False).head(10)
        half = base_df[base_df["mode"] == "half"].sort_values(["val_cum", "full_cum"], ascending=False).head(10)
        print_section(f"{base.upper()} GATE TOP 10", gate)
        print_section(f"{base.upper()} HALF/FULL TOP 10", half)


if __name__ == "__main__":
    main()
