from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = ROOT / "results" / "daily_top5_details_Window_3_20260413_171803.json"
Q_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def summarize(returns: pd.Series) -> dict[str, float]:
    s = pd.Series(returns).dropna().astype(float)
    if s.empty:
        return {
            "days": 0,
            "mean": float("nan"),
            "cum": float("nan"),
            "mdd": float("nan"),
            "win": float("nan"),
        }
    eq = (1.0 + s).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "days": int(len(s)),
        "mean": float(s.mean()),
        "cum": float(eq.iloc[-1] - 1.0),
        "mdd": float(dd.min()),
        "win": float((s > 0).mean()),
    }


def _sector_hhi(sectors: list[str]) -> float:
    clean = [s for s in sectors if s]
    if not clean:
        return float("nan")
    counts = Counter(clean)
    n = sum(counts.values())
    return float(sum((v / n) ** 2 for v in counts.values()))


def load_json_panel(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for ds, daily in sorted(raw.items()):
        top = {f"top{i}": daily.get(f"top{i}", {}) for i in range(1, 6)}
        preds = [top[f"top{i}"].get("pred", np.nan) for i in range(1, 6)]
        rets = [top[f"top{i}"].get("real_return", np.nan) for i in range(1, 6)]
        sectors = [top[f"top{i}"].get("sw_sector") for i in range(1, 6)]
        sector_counts = Counter([s for s in sectors if s])
        top1_sector = sectors[0]
        same_as_top1 = sum(1 for s in sectors if s == top1_sector) if top1_sector else 0
        rows.append(
            {
                "date": pd.to_datetime(ds),
                "top1_ret": float(rets[0]),
                "top2_ret": float(rets[1]),
                "top1_pos": int(float(rets[0]) > 0) if pd.notna(rets[0]) else np.nan,
                "top1_sector": top1_sector,
                "top2_sector": sectors[1],
                "top1_top2_same_sector": int(top1_sector == sectors[1]) if top1_sector and sectors[1] else 0,
                "top5_sector_count": int(len(sector_counts)),
                "top5_sector_hhi": _sector_hhi(sectors),
                "top1_sector_share_in_top5": float(same_as_top1 / 5.0),
                "top5_pred_mean": float(np.mean(preds)) if np.isfinite(preds).all() else np.nan,
                "pred_gap_12": float(preds[0] - preds[1]) if np.isfinite(preds[:2]).all() else np.nan,
                "pred_gap_1_mean25": float(preds[0] - np.mean(preds[1:5])) if np.isfinite(preds).all() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_expanding_threshold(series: pd.Series, q: float) -> pd.Series:
    vals = []
    for i in range(len(series)):
        hist = series.iloc[:i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= 20 else np.nan)
    return pd.Series(vals, index=series.index, dtype="float64")


def build_rolling_threshold(series: pd.Series, q: float, window: int) -> pd.Series:
    vals = []
    min_obs = min(window, 20)
    for i in range(len(series)):
        hist = series.iloc[max(0, i - window):i].dropna()
        vals.append(float(hist.quantile(q)) if len(hist) >= min_obs else np.nan)
    return pd.Series(vals, index=series.index, dtype="float64")


def scan_continuous_factor(df: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    switch_rows: list[dict[str, float | str]] = []
    gate_rows: list[dict[str, float | str]] = []
    series = df[factor].astype("float64")
    for q in Q_GRID:
        threshold_defs = [
            (f"expanding_q{int(q * 100)}_shift1", build_expanding_threshold(series, q)),
            (f"rolling10_q{int(q * 100)}_shift1", build_rolling_threshold(series, q, 10)),
            (f"rolling20_q{int(q * 100)}_shift1", build_rolling_threshold(series, q, 20)),
        ]
        for mode, thr in threshold_defs:
            valid = series.notna() & thr.notna() & df["top1_ret"].notna() & df["top2_ret"].notna()
            if not valid.any():
                continue
            high = series >= thr
            ret_high_top1 = np.where(high, df["top1_ret"], df["top2_ret"])
            ret_high_top2 = np.where(high, df["top2_ret"], df["top1_ret"])
            top1_cash_high = np.where(high, df["top1_ret"], 0.0)
            high_df = df.loc[valid & high]
            low_df = df.loc[valid & ~high]
            switch_rows.append(
                {"factor": factor, "mode": mode, "direction": "high_top1_low_top2", **summarize(pd.Series(ret_high_top1).where(valid))}
            )
            switch_rows.append(
                {"factor": factor, "mode": mode, "direction": "high_top2_low_top1", **summarize(pd.Series(ret_high_top2).where(valid))}
            )
            gate_rows.append(
                {
                    "factor": factor,
                    "mode": mode,
                    "direction": "high_top1_else_cash",
                    **summarize(pd.Series(top1_cash_high).where(valid)),
                    "high_days": int(len(high_df)),
                    "low_days": int(len(low_df)),
                    "pos_delta": (
                        float(high_df["top1_pos"].mean() - low_df["top1_pos"].mean())
                        if len(high_df) and len(low_df)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(switch_rows), pd.DataFrame(gate_rows)


def analyze_binary_flag(df: pd.DataFrame, factor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag = df[factor].fillna(0).astype(int) == 1
    valid = df["top1_ret"].notna() & df["top2_ret"].notna()
    switch_rows = []
    gate_rows = []
    for direction, selected in [
        ("flag_top1_else_top2", np.where(flag, df["top1_ret"], df["top2_ret"])),
        ("flag_top2_else_top1", np.where(flag, df["top2_ret"], df["top1_ret"])),
    ]:
        switch_rows.append({"factor": factor, "mode": "binary", "direction": direction, **summarize(pd.Series(selected).where(valid))})
    for direction, selected in [
        ("flag_top1_else_cash", np.where(flag, df["top1_ret"], 0.0)),
        ("nonflag_top1_else_cash", np.where(flag, 0.0, df["top1_ret"])),
    ]:
        gate_rows.append({"factor": factor, "mode": "binary", "direction": direction, **summarize(pd.Series(selected).where(valid))})
    return pd.DataFrame(switch_rows), pd.DataFrame(gate_rows)


def build_daily_detail(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["top1_better_than_top2"] = (out["top1_ret"] > out["top2_ret"]).astype(int)
    out["top1_minus_top2_ret"] = out["top1_ret"] - out["top2_ret"]
    return out[
        [
            "date",
            "top1_ret",
            "top2_ret",
            "top1_minus_top2_ret",
            "top1_better_than_top2",
            "top1_sector",
            "top2_sector",
            "top1_top2_same_sector",
            "top5_sector_count",
            "top5_sector_hhi",
            "top1_sector_share_in_top5",
            "pred_gap_12",
            "pred_gap_1_mean25",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze top5 industry structure from daily_top5_details json.")
    parser.add_argument("--json", default=str(DEFAULT_JSON), help="daily_top5_details json 路径")
    args = parser.parse_args()

    json_path = Path(args.json)
    df = load_json_panel(json_path)

    switch_parts = []
    gate_parts = []

    for factor in ["top5_sector_hhi", "top1_sector_share_in_top5"]:
        s1, g1 = scan_continuous_factor(df, factor)
        switch_parts.append(s1)
        gate_parts.append(g1)

    for factor in ["top1_top2_same_sector"]:
        s2, g2 = analyze_binary_flag(df, factor)
        switch_parts.append(s2)
        gate_parts.append(g2)

    switch_df = pd.concat(switch_parts, ignore_index=True)
    gate_df = pd.concat(gate_parts, ignore_index=True)
    detail_df = build_daily_detail(df)

    out_dir = ROOT / "results"
    stem = json_path.stem
    switch_path = out_dir / f"{stem}_industry_switch_scan.csv"
    gate_path = out_dir / f"{stem}_industry_top1_gate_scan.csv"
    detail_path = out_dir / f"{stem}_industry_daily_detail.csv"
    switch_df.to_csv(switch_path, index=False, encoding="utf-8-sig")
    gate_df.to_csv(gate_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("Overall baseline:")
    print(f"range={df['date'].min().date()} ~ {df['date'].max().date()} days={len(df)}")
    for col in ["top1_ret", "top2_ret"]:
        st = summarize(df[col])
        print(f"{col}: mean={st['mean']:.4%} cum={st['cum']:.6f} mdd={st['mdd']:.4%} win={st['win']:.4%}")
    print()

    print("Same-sector vs cross-sector:")
    same = df[df["top1_top2_same_sector"] == 1]
    diff = df[df["top1_top2_same_sector"] == 0]
    for label, part in [("same_sector", same), ("cross_sector", diff)]:
        if part.empty:
            continue
        print(
            f"{label}: days={len(part)} top1_mean={part['top1_ret'].mean():.4%} "
            f"top2_mean={part['top2_ret'].mean():.4%} top1_better={(part['top1_ret'] > part['top2_ret']).mean():.4%}"
        )
    print()

    print("Best industry switch rules:")
    print(
        switch_df.sort_values(["cum", "mean"], ascending=False)
        .head(20)[["factor", "mode", "direction", "mean", "cum", "mdd", "win"]]
        .to_string(index=False)
    )
    print()
    print("Best industry top1 gate rules:")
    print(
        gate_df.sort_values(["cum", "mean"], ascending=False)
        .head(20)[["factor", "mode", "direction", "mean", "cum", "mdd", "win"]]
        .to_string(index=False)
    )
    print()
    print(f"Saved: {switch_path}")
    print(f"Saved: {gate_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
