#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风控否决规则训练（左尾排雷器，V5.1 预注册方法论）。

定位：主模型（stage15）负责"找弹性"，本脚本负责"在冠军模型挑出的头部候选里
一票否决大跌股"。两者解耦——风控的净增量可独立度量。

输入
----
- `cv_top5_details.csv`：stage15 `--mode cv` 冠军超参的 11 折样本外 Top5 明细
  （列含 date/rank/stock_code/return(=r3)/score_z_in_day/fold_id/param/cp）。
- 特征矩阵 `lgb_features.parquet`（join 风险特征，缺列自动跳过）。
- 行情缓存 `cache/stock_features_*.parquet`（派生 乖离/上影/量比；可缺省）。
- 日内降频缓存 `local_data/minute/daily_features_cache.parquet`（可选）。

输出
----
- `risk_veto_rules.json`：Top N 条稳健规则（depth-2 阈值组合）+ 历史/walk-forward
  统计；`apply_veto(rules, df)` 供回测/实盘级联否决（Rank1 被否 → 顺延 Rank2 →
  … → 空仓）复用。
- `risk_veto_report.md`：逐折明细。

预注册方法论（防过拟合纪律，禁改）
----------------------------------
1. **特征白名单**：仅 5 类具"派发/衰竭/失速"物理含义的特征（RISK_FEATURE_POOL），
   禁全特征搜索（Top 池样本少，全搜=多重比较自由度爆炸，必出巧合噪音）。
2. **Walk-forward**：第 k 折只用 1..k-1 折样本挖规则（严格过去时）。
3. **规则形式**：`[Fa>qa ∧ Fb<qb]` / `[Fa<qa ∧ Fb>qb]`；阈值取训练前缀样本的
   全局分位 {0.10, 0.20, 0.80, 0.90}；支持度（命中样本数）≥ `--min-support`。
4. **验收**：全历史 拦截大亏(y=r3≤-5%)数/误杀大涨(y=r3≥+7%)数 ≥ `--ratio-min`(3.5)
   且 >`--fold-pos-frac`(70%) 的历史折上"级联净收益增量"为正。
5. 输出按 (拦截精确率=命中大亏/否决行数, 拦截大亏数, 比例) 排序去重（同一特征
   对保留最优）取 Top `--max-rules`；精确率优先防止"大网"规则否决半个股票池。
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
RESULTS = _HERE / "results" / "merge_v2"

CRASH_TH = -0.05      # 大亏标签：r3 <= -5%
BOOM_TH = 0.07        # 大涨标签：r3 >= +7%
QS = (0.10, 0.20, 0.80, 0.90)

#: 风控特征白名单（5 类物理含义；存在性过滤，缺列自动跳过并报告）
RISK_FEATURE_POOL = (
    # ① 日内高位派发/回撤（intraday 降频）
    "id_hi_shadow_dump_s3", "id_hi_shadow_dump",
    "id_hi_amt_share20_s3", "id_hi_amt_share20",
    "id_dd_recovery", "id_updown_vol_ratio_s3",
    # ② 主力资金流背离（factors.parquet 存量，若入矩阵）
    "flow_price_divergence_10d", "flow_momentum_decay_5d_is",
    "flow_main_net_rate_is",
    # ③ 短期涨幅透支/波动/派发（矩阵 + 行情派生）
    "IVOL20", "TURNOVER", "bias5", "bias10", "ret5", "up_shadow", "amt_ratio",
    "mi_post_high_drawdown",
    # ④ 板块失速/身位
    "GS_sec_nh60_ratio_level", "GS_sec_xs_disp_level",
    "is_ret3_rank", "is_leader_gap5", "is_turn_rank",
    # ⑤ 当日横截面置信度（cv 明细自带）
    "score_z_in_day",
)


def find_price_cache() -> Path | None:
    cands = sorted(glob.glob(str(_HERE / "cache" / "stock_features_*.parquet")))
    best, best_end = None, ""
    for c in cands:
        m = re.search(r"_(\d{8})_(\d{8})\.parquet$", Path(c).name)
        if m and m.group(1) != m.group(2) and m.group(2) > best_end:
            best, best_end = Path(c), m.group(2)
    return best or (Path(cands[-1]) if cands else None)


def find_intraday_cache() -> Path | None:
    for cand in (_HERE.parent / "local_data" / "minute",
                 _HERE / "local_data" / "minute"):
        p = cand / "daily_features_cache.parquet"
        if p.exists():
            return p
    return None


def derive_price_features(price_path: Path, dates_lo, dates_hi) -> pd.DataFrame:
    """行情派生：bias5/bias10/ret5/up_shadow/amt_ratio（全回溯窗，PIT 安全）。"""
    cols = ["stock_code", "date", "high", "low", "close", "amount"]
    import pyarrow.parquet as pq
    have = set(pq.read_schema(price_path).names)
    cols = [c for c in cols if c in have]
    d = pd.read_parquet(price_path, columns=cols)
    d["date"] = pd.to_datetime(d["date"])
    lo = pd.Timestamp(dates_lo) - pd.Timedelta(days=45)
    d = d[(d["date"] >= lo) & (d["date"] <= pd.Timestamp(dates_hi))]
    d["stock_code"] = d["stock_code"].astype(str)
    d = d.sort_values(["stock_code", "date"], kind="stable")
    g = d.groupby("stock_code", sort=False)
    close = d["close"]
    ma5 = g["close"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    ma10 = g["close"].rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    amt_ma20 = g["amount"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    d["bias5"] = close / ma5.where(ma5 > 0) - 1.0
    d["bias10"] = close / ma10.where(ma10 > 0) - 1.0
    d["ret5"] = close / g["close"].shift(5) - 1.0
    rng = (d["high"] - d["low"]).where(lambda s: s > 0)
    d["up_shadow"] = (d["high"] - close) / rng
    d["amt_ratio"] = d["amount"] / amt_ma20.where(amt_ma20 > 0)
    return d[["stock_code", "date", "bias5", "bias10", "ret5",
              "up_shadow", "amt_ratio"]]


def load_candidates(args, logger) -> tuple[pd.DataFrame, list[str], list[str]]:
    d = pd.read_csv(args.cv_details, encoding="utf-8-sig")
    d = d[(d["param"] == args.pid) & (d["cp"] == args.cp)].copy()
    if not len(d):
        raise SystemExit(f"cv-details 中无 param={args.pid} cp={args.cp} 的样本")
    d["date"] = pd.to_datetime(d["date"])
    d["stock_code"] = d["stock_code"].astype(str)
    d = d.sort_values(["date", "rank"], kind="stable").reset_index(drop=True)
    logger.info(f"[风控] 候选样本 {len(d)} 行 | {d['date'].min().date()}~{d['date'].max().date()} "
                f"| 折数 {d['fold_id'].nunique()}")

    # 特征矩阵 join
    if args.features and Path(args.features).exists():
        fdf = pd.read_parquet(args.features)
        fdf["date"] = pd.to_datetime(fdf["date"])
        fdf["stock_code"] = fdf["stock_code"].astype(str)
        keep = [c for c in fdf.columns if c not in ("stock_code", "date")]
        d = d.merge(fdf[["stock_code", "date"] + keep], on=["stock_code", "date"],
                    how="left")
    else:
        logger.warning("[风控] 特征矩阵缺失，仅用 cv 明细+行情派生特征")

    # 行情派生
    price_path = Path(args.price_cache) if args.price_cache else find_price_cache()
    if price_path and price_path.exists():
        try:
            pder = derive_price_features(price_path, d["date"].min(), d["date"].max())
            d = d.merge(pder, on=["stock_code", "date"], how="left")
        except Exception as e:
            logger.warning(f"[风控] 行情派生失败 {str(e)[:120]}")
    else:
        logger.warning("[风控] 行情缓存缺失，bias/ret5/up_shadow/amt_ratio 不可用")

    # 日内降频缓存 join
    ip = Path(args.intraday_cache) if args.intraday_cache else find_intraday_cache()
    if ip and ip.exists():
        try:
            idf = pd.read_parquet(ip)
            idf["date"] = pd.to_datetime(idf["date"])
            idf["stock_code"] = idf["stock_code"].astype(str)
            icols = [c for c in idf.columns if c in RISK_FEATURE_POOL
                     and c not in ("score_z_in_day",)]
            if icols:
                d = d.merge(idf[["stock_code", "date"] + icols],
                            on=["stock_code", "date"], how="left")
        except Exception as e:
            logger.warning(f"[风控] 日内缓存 join 失败 {str(e)[:120]}")
    else:
        logger.info("[风控] 日内降频缓存不存在（id_* 特征跳过）")

    pool = [c for c in RISK_FEATURE_POOL if c in d.columns]
    missing = [c for c in RISK_FEATURE_POOL if c not in d.columns]
    # 有效样本：return 有限（r3 已由 cv 明细保证）
    d = d[np.isfinite(d["return"].to_numpy(dtype=float))].reset_index(drop=True)
    logger.info(f"[风控] 特征池 {len(pool)}/{len(RISK_FEATURE_POOL)} 可用：{pool}")
    if missing:
        logger.info(f"[风控] 缺失（跳过）：{missing}")
    return d, pool, missing


def mine_rules(train: pd.DataFrame, pool: list[str],
               min_support: int = 20, ratio_min: float = 3.5) -> list[dict]:
    """在训练前缀上挖 depth-2 阈值规则（阈值=训练样本全局分位）。"""
    qv = {}
    for f in pool:
        v = pd.to_numeric(train[f], errors="coerce")
        if np.isfinite(v).sum() < 100:
            continue
        qv[f] = {q: float(v.quantile(q)) for q in QS}
    feats = sorted(qv.keys())
    ret = train["return"].to_numpy(dtype=float)
    crash = ret <= CRASH_TH
    boom = ret >= BOOM_TH
    rules = []
    for fa, fb in itertools.combinations(feats, 2):
        va = pd.to_numeric(train[fa], errors="coerce").to_numpy(dtype=float)
        vb = pd.to_numeric(train[fb], errors="coerce").to_numpy(dtype=float)
        for qa in QS:
            for qb in QS:
                for da, db in ((1, -1), (-1, 1)):
                    ma = va > qv[fa][qa] if da > 0 else va < qv[fa][qa]
                    mb = vb > qv[fb][qb] if db > 0 else vb < qv[fb][qb]
                    veto = np.nan_to_num(ma, nan=False).astype(bool) \
                        & np.nan_to_num(mb, nan=False).astype(bool)
                    sup = int(veto.sum())
                    if sup < min_support:
                        continue
                    c = int((veto & crash).sum())
                    b = int((veto & boom).sum())
                    ratio = c / max(b, 1)
                    if c < 5 or ratio < ratio_min:
                        continue
                    rules.append({
                        "fa": fa, "fb": fb, "qa": qa, "qb": qb,
                        "dir_a": da, "dir_b": db,
                        "th_a": qv[fa][qa], "th_b": qv[fb][qb],
                        "train": {"support": sup, "crash_caught": c,
                                  "boom_killed": b, "ratio": round(ratio, 3),
                                  "precision": round(c / sup, 4)},
                    })
    # 同一特征对保留最优（按拦截精确率=命中大亏/否决行数；防止"大网"规则
    # 靠多拦随机大亏刷 crash_caught 而实际否决半个股票池）
    best: dict = {}
    for r in rules:
        k = (r["fa"], r["fb"])
        s = (r["train"]["precision"], r["train"]["crash_caught"],
             r["train"]["ratio"])
        if k not in best or s > (best[k]["train"]["precision"],
                                 best[k]["train"]["crash_caught"],
                                 best[k]["train"]["ratio"]):
            best[k] = r
    out = sorted(best.values(),
                 key=lambda r: (-r["train"]["precision"],
                                -r["train"]["crash_caught"],
                                -r["train"]["ratio"]))
    return out


def apply_veto_mask(df: pd.DataFrame, rule: dict) -> np.ndarray:
    """单规则触发掩码（NaN 比较=False，即缺值不否决）。
    兼容两种规则形态：内部挖掘 dict（fa/fb）与 risk_veto_rules.json（features）。"""
    fa = rule["fa"] if "fa" in rule else rule["features"][0]
    fb = rule["fb"] if "fb" in rule else rule["features"][1]
    va = pd.to_numeric(df[fa], errors="coerce").to_numpy(dtype=float)
    vb = pd.to_numeric(df[fb], errors="coerce").to_numpy(dtype=float)
    ma = va > rule["th_a"] if rule["dir_a"] > 0 else va < rule["th_a"]
    mb = vb > rule["th_b"] if rule["dir_b"] > 0 else vb < rule["th_b"]
    return np.nan_to_num(ma, nan=False).astype(bool) \
        & np.nan_to_num(mb, nan=False).astype(bool)


def apply_veto(rules: list[dict], df: pd.DataFrame,
               skip_missing: bool = False) -> np.ndarray:
    """任一规则触发 → True（供回测/实盘级联否决消费）。
    严格模式（默认）：规则引用的特征列缺失时报错（防静默漏特征导致风控失效）；
    skip_missing=True 时跳过引用了缺失列的规则。"""
    missing = sorted({f for r in rules
                      for f in ((r["fa"], r["fb"]) if "fa" in r else r["features"])
                      if f not in df.columns})
    if missing and not skip_missing:
        raise ValueError(
            f"apply_veto: 帧缺少规则特征列 {missing}；请先 join 对应特征"
            "（lgb_features / 行情派生 / 日内降频缓存）")
    m = np.zeros(len(df), dtype=bool)
    for r in rules:
        fa = r["fa"] if "fa" in r else r["features"][0]
        fb = r["fb"] if "fb" in r else r["features"][1]
        if skip_missing and (fa not in df.columns or fb not in df.columns):
            continue
        m |= apply_veto_mask(df, r)
    return m


def cascade_by_fold(df: pd.DataFrame, veto: np.ndarray) -> pd.DataFrame:
    """按 (date, rank) 级联：每日取第一个未否决候选的 return；全否决=0。
    返回 {date, fold_id, base_ret(rank1), veto_ret} 逐日表。"""
    g = df.assign(_veto=veto).sort_values(["date", "rank"], kind="stable")
    base = g.groupby("date", sort=True)["return"].first()
    keep = g[~g["_veto"]].groupby("date", sort=True)["return"].first()
    fold = g.groupby("date", sort=True)["fold_id"].first()
    out = pd.DataFrame({"base_ret": base, "veto_ret": keep, "fold_id": fold})
    out["veto_ret"] = out["veto_ret"].fillna(0.0)
    return out.reset_index()


def evaluate_rule(df: pd.DataFrame, rule: dict) -> dict:
    """全历史统计 + 逐折净增量。"""
    veto = apply_veto_mask(df, rule)
    ret = df["return"].to_numpy(dtype=float)
    c = int((veto & (ret <= CRASH_TH)).sum())
    b = int((veto & (ret >= BOOM_TH)).sum())
    casc = cascade_by_fold(df, veto)
    casc["delta"] = casc["veto_ret"] - casc["base_ret"]
    folds = casc.groupby("fold_id")["delta"].agg(["mean", "count"])
    pos = int((folds["mean"] > 0).sum())
    return {
        "support": int(veto.sum()), "crash_caught": c, "boom_killed": b,
        "ratio": round(c / max(b, 1), 3),
        "folds_total": int(len(folds)), "folds_positive": pos,
        "folds_positive_frac": round(pos / max(len(folds), 1), 3),
        "mean_delta": round(float(casc["delta"].mean()), 6),
        "fold_deltas": {str(int(k)): round(float(v), 6)
                        for k, v in folds["mean"].items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="风控否决规则训练（walk-forward depth-2）")
    ap.add_argument("--cv-details", required=True,
                    help="stage15 CV 输出 cv_top5_details.csv")
    ap.add_argument("--features",
                    default=str(RESULTS / "lgb_ready" / "lgb_features.parquet"))
    ap.add_argument("--price-cache", default="", help="行情缓存（缺省自动定位 cache/）")
    ap.add_argument("--intraday-cache", default="",
                    help="日内降频缓存（缺省自动定位 local_data/minute/）")
    ap.add_argument("--pid", default="H15", help="冠军超参 pid")
    ap.add_argument("--cp", type=int, default=400, help="冠军 checkpoint")
    ap.add_argument("--max-rules", type=int, default=5)
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--ratio-min", type=float, default=3.5)
    ap.add_argument("--fold-pos-frac", type=float, default=0.70)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    import logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (_HERE / "results" / f"risk_veto_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("risk_veto")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(out_dir / "run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)

    t0 = time.time()
    df, pool, missing = load_candidates(args, logger)
    if len(pool) < 4:
        logger.error(f"可用风控特征过少（{len(pool)}<4），无法成对挖规则")
        return 1
    fold_ids = sorted(int(x) for x in df["fold_id"].unique())
    if len(fold_ids) < 2:
        logger.error(f"仅 {len(fold_ids)} 个折（fold_id={fold_ids}），walk-forward 需 ≥2 折；"
                     "请传入 CV 模式产出的 cv_top5_details.csv（11 折），"
                     "而非 OOS 明细（oos2026_top5_details.csv 为单折 99）")
        return 1
    logger.info(f"[风控] walk-forward 折序: {fold_ids}")

    # walk-forward：逐前缀挖规则（仅记录过程），最终规则取全前缀（fold 1..K-1）
    prefix_rules = []
    for k_i in range(1, len(fold_ids)):
        train = df[df["fold_id"].isin(fold_ids[:k_i])]
        rules = mine_rules(train, pool, args.min_support, args.ratio_min)
        prefix_rules.append({"k": k_i, "n_train": len(train), "n_rules": len(rules)})
        logger.info(f"[风控] 前缀 fold1..{fold_ids[k_i - 1]} 训练 {len(train)} 行 "
                    f"-> 候选规则 {len(rules)} 条")
    final_rules = mine_rules(df[df["fold_id"].isin(fold_ids[:-1])], pool,
                             args.min_support, args.ratio_min)

    # 验收：全历史 ratio + 逐折正占比（严格按预注册阈值）
    accepted = []
    for r in final_rules:
        ev = evaluate_rule(df, r)
        r["history"] = ev
        if (ev["ratio"] >= args.ratio_min
                and ev["folds_positive_frac"] > args.fold_pos_frac):
            accepted.append(r)
    accepted = accepted[:max(0, args.max_rules)]
    logger.info(f"[风控] 最终候选 {len(final_rules)} 条 -> 验收通过 {len(accepted)} 条")

    rules_out = []
    for i, r in enumerate(accepted, 1):
        da = ">" if r["dir_a"] > 0 else "<"
        db = ">" if r["dir_b"] > 0 else "<"
        expr = (f"{r['fa']} {da} {r['th_a']:.6g} AND {r['fb']} {db} {r['th_b']:.6g}")
        rules_out.append({
            "id": i, "expr": expr, "features": [r["fa"], r["fb"]],
            "dir_a": r["dir_a"], "dir_b": r["dir_b"],
            "th_a": r["th_a"], "th_b": r["th_b"],
            "train": r["train"], "history": r["history"],
        })
        logger.info(f"[规则{i}] {expr} | 拦亏={r['history']['crash_caught']} "
                    f"误杀={r['history']['boom_killed']} "
                    f"ratio={r['history']['ratio']} "
                    f"折正={r['history']['folds_positive']}/{r['history']['folds_total']} "
                    f"Δ均值={r['history']['mean_delta']}")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "walk-forward depth-2 veto (预注册: 白名单特征/分位阈值/ratio>=%.1f/"
                  "折正占比>%.0f%%)" % (args.ratio_min, args.fold_pos_frac * 100),
        "pid": args.pid, "cp": int(args.cp),
        "n_candidates": int(len(df)),
        "feature_pool": pool, "features_missing": missing,
        "walk_forward_prefix": prefix_rules,
        "rules": rules_out,
        "note": "级联消费: Rank1 被否 -> 顺延 Rank2 -> ... -> 空仓 (apply_veto)",
    }
    (out_dir / "risk_veto_rules.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    rep = [f"# 风控否决规则报告（{ts}）", "",
           f"- 冠军: {args.pid}@cp{args.cp} | 候选样本 {len(df)} | 特征池 {len(pool)}",
           f"- 验收标准: ratio≥{args.ratio_min} 且 折正占比>{args.fold_pos_frac:.0%}",
           f"- 通过规则: {len(rules_out)} 条", ""]
    for r in rules_out:
        rep.append(f"## 规则{r['id']}: `{r['expr']}`")
        rep.append(f"- 全历史: 拦亏 {r['history']['crash_caught']} / "
                   f"误杀大涨 {r['history']['boom_killed']} / ratio {r['history']['ratio']}")
        rep.append(f"- 折正: {r['history']['folds_positive']}/"
                   f"{r['history']['folds_total']} | Δ均值 {r['history']['mean_delta']}")
        rep.append(f"- 逐折 Δ: {r['history']['fold_deltas']}")
        rep.append("")
    (out_dir / "risk_veto_report.md").write_text("\n".join(rep), encoding="utf-8")
    logger.info(f"[风控] 完成 {time.time()-t0:.0f}s -> {out_dir}/risk_veto_rules.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
