# -*- coding: utf-8 -*-
"""4B.1：TURNOVER 物理生境双专家（Active/Coiled）CV。

每日按 TURNOVER 截面分位 P65 切两个互斥生境：
  Active  : rank_pct(TURNOVER) >= pct
  Coiled  : rank_pct(TURNOVER) <  pct
两个专家保持同一 L0 标签，各自训练 H23@cp，测试日各取本生境同行业去重 Rank1；
Coiled 的 Rank1 必须与 Active Rank1 不同申万二级行业。
输出生活区/潜伏区单笔与等权双状元组合的日频均值。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage15_v5_pipeline import (  # noqa: E402
    EMBARGO_DAYS, HP_GRID, build_forward_folds, compute_sample_weights,
    load_dataset_and_labels, setup_logger, train_predict_fold,
)


def _rank_pct(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    return s.groupby(pd.to_datetime(dates), sort=False).rank(pct=True).to_numpy(dtype=float)


def _pick_top1(df: pd.DataFrame, pred_col: str, exclude_sector: str | None = None):
    sub = df[np.isfinite(df[pred_col].to_numpy(dtype=float))]
    if exclude_sector is not None:
        sub = sub[sub["sector"].astype(str) != str(exclude_sector)]
    if not len(sub):
        return None
    sub = sub.sort_values(pred_col, ascending=False, kind="stable")
    sub = sub.drop_duplicates(subset=["sector"], keep="first")
    return sub.iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_start", default="2021-06-01")
    ap.add_argument("--cv_end", default="2025-12-31")
    ap.add_argument("--mask_mode", choices=["old", "fwd", "new"], default="fwd")
    ap.add_argument("--pids", default="H23")
    ap.add_argument("--cp", type=int, default=170)
    ap.add_argument("--seeds", default="42,2024,777")
    ap.add_argument("--num_threads", type=int, default=32)
    ap.add_argument("--train_months", type=int, default=21)
    ap.add_argument("--test_months", type=int, default=3)
    ap.add_argument("--turnover_pct", type=float, default=0.65)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out / "run.log")
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    hp = next(h for h in HP_GRID if h["pid"] == str(args.pids))

    load_args = argparse.Namespace(
        features_parquet=args.features_parquet, data_start=args.data_start,
        label_v51=False, enable_buffer_pos=False, mask_mode=args.mask_mode)
    bundle = load_dataset_and_labels(load_args, logger)
    X, y, y_raw = bundle["X"], bundle["y"], bundle["y_raw"]
    core_pos, buf_pos = bundle["core_pos"], bundle["buf_pos"]
    dates, codes, sectors = bundle["dates"], bundle["codes"], bundle["sectors"]
    train_mask = bundle["train_mask"]
    test_mask = bundle["test_mask"]
    trade_mask = bundle["trade_mask"]
    dep = np.array([str(s) for s in sectors], dtype=object)
    feat_cols = list(bundle["feature_cols"])
    if "TURNOVER" not in feat_cols:
        raise SystemExit("[HABITAT] TURNOVER 不在特征列，无法切生境")
    turn = X[:, feat_cols.index("TURNOVER")].astype(float)
    logger.info(f"[HABITAT] features={len(feat_cols)} rows={len(X):,} mask={args.mask_mode} "
                f"P{args.turnover_pct*100:.0f} cp={args.cp}")

    uniq_days = np.unique(dates[trade_mask])
    folds = build_forward_folds(uniq_days, train_months=args.train_months,
                                test_months=args.test_months, embargo_days=EMBARGO_DAYS,
                                cv_end=args.cv_end, logger=logger)
    daily_rows = []
    for fo in folds:
        tr_m = ((dates >= np.datetime64(fo["train_start"]))
                & (dates <= np.datetime64(fo["train_end"])) & train_mask)
        te_m = ((dates >= np.datetime64(fo["test_start"]))
                & (dates <= np.datetime64(fo["test_end"])) & test_mask)
        if tr_m.sum() < args.num_threads or te_m.sum() < 100:
            logger.warning(f"[HABITAT] Fold{fo['fold_id']} 样本不足，跳过")
            continue
        tr_rank = _rank_pct(turn[tr_m], dates[tr_m])
        high_tr = tr_rank >= args.turnover_pct
        low_tr = ~high_tr
        w_tr = compute_sample_weights(core_pos[tr_m], buf_pos[tr_m], y_raw[tr_m], dates[tr_m])
        preds = {}
        for side, mask_tr in (("active", high_tr), ("coiled", low_tr)):
            if mask_tr.sum() < 1000:
                preds[side] = None
                continue
            res = train_predict_fold(
                X[tr_m][mask_tr], y[tr_m][mask_tr], w_tr[mask_tr],
                X[te_m], dates[te_m], hp, [args.cp], seeds, ["lgb"], args.num_threads)
            preds[side] = res["lgb"][args.cp]
            logger.info(f"[HABITAT] Fold{fo['fold_id']} {side} train={int(mask_tr.sum()):,} "
                        f"test={int(te_m.sum()):,}")
        if preds.get("active") is None or preds.get("coiled") is None:
            logger.warning(f"[HABITAT] Fold{fo['fold_id']} 专家训练样本不足，跳过")
            continue
        te_rank = _rank_pct(turn[te_m], dates[te_m])
        df = pd.DataFrame({
            "date": pd.to_datetime(dates[te_m]), "code": codes[te_m],
            "sector": dep[te_m], "ret": np.asarray(y_raw[te_m], dtype=float),
            "predA": preds["active"], "predB": preds["coiled"],
            "turn_pct": te_rank,
        })
        for d, g in df.groupby("date", sort=True):
            a = _pick_top1(g[g["turn_pct"] >= args.turnover_pct], "predA")
            if a is None:
                continue
            b = _pick_top1(g[g["turn_pct"] < args.turnover_pct], "predB",
                           exclude_sector=str(a["sector"]))
            ra = float(a["ret"]); rb = float(b["ret"]) if b is not None else np.nan
            daily_rows.append({
                "fold_id": int(fo["fold_id"]), "date": str(pd.Timestamp(d).date()),
                "active_code": str(a["code"]), "active_sector": str(a["sector"]),
                "active_ret": ra,
                "coiled_code": str(b["code"]) if b is not None else "",
                "coiled_sector": str(b["sector"]) if b is not None else "",
                "coiled_ret": rb,
                "twin_ret": float(np.nanmean([ra, rb])) if np.isfinite(rb) else np.nan,
            })
    if not daily_rows:
        logger.error("[HABITAT] 无有效样本")
        return 1
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(out / "habitat_split_daily.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_days": int(len(daily)),
        "active_mean": float(daily["active_ret"].mean()),
        "active_win": float((daily["active_ret"] > 0).mean()),
        "coiled_mean": float(daily["coiled_ret"].mean()),
        "coiled_win": float((daily["coiled_ret"] > 0).mean()),
        "coiled_n": int(daily["coiled_ret"].notna().sum()),
        "twin_mean": float(daily["twin_ret"].mean()),
        "twin_win": float((daily["twin_ret"] > 0).mean()),
        "twin_n": int(daily["twin_ret"].notna().sum()),
        "turnover_pct": float(args.turnover_pct),
        "cp": int(args.cp), "pid": str(args.pids), "mask_mode": args.mask_mode,
    }
    pd.DataFrame([summary]).to_csv(out / "habitat_split_summary.csv",
                                   index=False, encoding="utf-8-sig")
    logger.info(f"[HABITAT] {summary}")
    logger.info("\n" + daily.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())