# -*- coding: utf-8 -*-
"""4B.2：残差盲区猎手（Residual Monster Boost）CV。

每折：
1) 主模型 H23@cp 在全训练集上训练，得到训练集 in-sample 排序；按行业去重后
   前 10 名记为“已捕获”。
2) 权重调整：已捕获正样本 × captured_w（默认 0.15）；漏掉的 Rank>10 正样本
   × missed_w（默认 1.25）；其余保持原权重。
3) 用调整权重训练第二模型；测试日 Model1 Rank1 + Model2 不同行业 Rank1 组合。
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
    ap.add_argument("--captured_w", type=float, default=0.15)
    ap.add_argument("--missed_w", type=float, default=1.25)
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
    logger.info(f"[RESIDUAL] rows={len(X):,} mask={args.mask_mode} cp={args.cp} "
                f"captured_w={args.captured_w} missed_w={args.missed_w}")

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
        if tr_m.sum() < 50000 or te_m.sum() < 100:
            logger.warning(f"[RESIDUAL] Fold{fo['fold_id']} 样本不足，跳过")
            continue
        w_tr = compute_sample_weights(core_pos[tr_m], buf_pos[tr_m], y_raw[tr_m], dates[tr_m])
        r1 = train_predict_fold(
            X[tr_m], y[tr_m], w_tr, X[te_m], dates[te_m], hp, [args.cp],
            seeds, ["lgb"], args.num_threads, want_train=True)
        p1 = r1["lgb"][args.cp]
        p1_tr = r1["train"][args.cp]

        tr = pd.DataFrame({
            "date": pd.to_datetime(dates[tr_m]), "code": codes[tr_m],
            "sector": np.asarray(sectors[tr_m], dtype=object),
            "y": np.asarray(y[tr_m], dtype=float), "pred": p1_tr,
            "ret": np.asarray(y_raw[tr_m], dtype=float),
        })
        tr = tr.sort_values(["date", "pred"], ascending=[True, False], kind="stable")
        tr = tr.drop_duplicates(subset=["date", "sector"], keep="first").copy()
        tr["rank"] = tr.groupby("date", sort=False).cumcount() + 1
        captured = (tr["y"].to_numpy(dtype=float) > 0.5) & (tr["rank"].to_numpy() <= 10)
        missed = (tr["y"].to_numpy(dtype=float) > 0.5) & (tr["rank"].to_numpy() > 10)
        w2 = w_tr.copy()
        w2[tr.index.to_numpy()[captured]] *= float(args.captured_w)
        w2[tr.index.to_numpy()[missed]] *= float(args.missed_w)
        logger.info(f"[RESIDUAL] Fold{fo['fold_id']} train={int(tr_m.sum()):,} "
                    f"captured_pos={int(captured.sum())} missed_pos={int(missed.sum())}")
        r2 = train_predict_fold(
            X[tr_m], y[tr_m], w2, X[te_m], dates[te_m], hp, [args.cp],
            seeds, ["lgb"], args.num_threads)
        p2 = r2["lgb"][args.cp]

        te = pd.DataFrame({
            "date": pd.to_datetime(dates[te_m]), "code": codes[te_m],
            "sector": np.asarray(sectors[te_m], dtype=object),
            "ret": np.asarray(y_raw[te_m], dtype=float),
            "pred1": p1, "pred2": p2,
        })
        for d, g in te.groupby("date", sort=True):
            m1 = _pick_top1(g, "pred1")
            if m1 is None:
                continue
            m2 = _pick_top1(g, "pred2", exclude_sector=str(m1["sector"]))
            r1v = float(m1["ret"]); r2v = float(m2["ret"]) if m2 is not None else np.nan
            daily_rows.append({
                "fold_id": int(fo["fold_id"]), "date": str(pd.Timestamp(d).date()),
                "m1_code": str(m1["code"]), "m1_sector": str(m1["sector"]),
                "m1_ret": r1v,
                "m2_code": str(m2["code"]) if m2 is not None else "",
                "m2_sector": str(m2["sector"]) if m2 is not None else "",
                "m2_ret": r2v,
                "twin_ret": float(np.nanmean([r1v, r2v])) if np.isfinite(r2v) else np.nan,
            })
    if not daily_rows:
        logger.error("[RESIDUAL] 无有效样本")
        return 1
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(out / "residual_boost_daily.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_days": int(len(daily)),
        "m1_mean": float(daily["m1_ret"].mean()),
        "m1_win": float((daily["m1_ret"] > 0).mean()),
        "m2_mean": float(daily["m2_ret"].mean()),
        "m2_win": float((daily["m2_ret"] > 0).mean()),
        "m2_n": int(daily["m2_ret"].notna().sum()),
        "twin_mean": float(daily["twin_ret"].mean()),
        "twin_win": float((daily["twin_ret"] > 0).mean()),
        "twin_n": int(daily["twin_ret"].notna().sum()),
        "captured_w": float(args.captured_w), "missed_w": float(args.missed_w),
        "cp": int(args.cp), "pid": str(args.pids), "mask_mode": args.mask_mode,
    }
    pd.DataFrame([summary]).to_csv(out / "residual_boost_summary.csv",
                                   index=False, encoding="utf-8-sig")
    logger.info(f"[RESIDUAL] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())