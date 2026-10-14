# -*- coding: utf-8 -*-
"""v2 评估引擎：实现 MERGE_V2_PLAN §4 指标与 §8 门（H2 记录、POOL_S3 删除判据）。

口径：
- 评估样本：tradable==1（labels_prod.parquet），每日截面有效样本 <200 的日期剔除。
- RankIC：逐日截面 Spearman(factor, R3)，tp 口径 R3。
- t_NW：RankIC 序列的 Newey–West t，lag=3（与标签 3 日重叠一致）。
- 置换 null：在日内打乱因子值，重算 RankIC 均值；向量化实现。
- TopK/Enrich/AUC/Resid/Mono 只作诊断（D4），不参与删除。
- 删除只由 POOL_S3（§6.2）决定；H2-A 的 p 值只用于 FDR 报告（§9.2）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prodlabel import R3_THR  # noqa: F401  （标签阈值与生产一致）

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
MIN_N = 200
NW_LAG = 3


# ---------- 基础统计 ----------

def _day_blocks(df: pd.DataFrame):
    """按 date 排序，返回 (codes, starts) 供向量化逐日计算。"""
    d = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    code, starts = pd.factorize(d["date"])
    return d, np.asarray(code), np.asarray(starts)


def _within_rank(v: np.ndarray, day: np.ndarray, n_days: int) -> np.ndarray:
    """日内排名（平均秩，1-based），NaN 保持 NaN。"""
    r = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    order = np.lexsort((np.where(ok, v, 0.0), day))
    vs = np.where(ok, v, 0.0)[order]
    ds = day[order]
    n = len(v)
    # 日内位置（0-based）
    db = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1]])
    pos = np.arange(n) - np.repeat(db, np.diff(np.r_[db, n]))
    # 并列组内取平均
    change = np.ones(n, bool)
    change[1:] = (ds[1:] != ds[:-1]) | (vs[1:] != vs[:-1])
    grp = np.cumsum(change) - 1
    ng = int(grp.max()) + 1
    sum_pos = np.bincount(grp, weights=pos, minlength=ng)
    cnt = np.bincount(grp, minlength=ng)
    avg = sum_pos / np.where(cnt > 0, cnt, 1) + 1.0
    r[order] = np.where(ok[order], avg[grp], np.nan)
    return r


def _daily_corr(xr: np.ndarray, yr: np.ndarray, day: np.ndarray, n_days: int) -> np.ndarray:
    """向量化逐日 Pearson（输入为日内排名即得 Spearman）。返回每 code 的 IC。"""
    ok = np.isfinite(xr) & np.isfinite(yr)
    d = day[ok]
    x = xr[ok]
    y = yr[ok]
    n = np.bincount(d, minlength=n_days).astype(float)
    sx = np.bincount(d, weights=x, minlength=n_days)
    sy = np.bincount(d, weights=y, minlength=n_days)
    sxx = np.bincount(d, weights=x * x, minlength=n_days)
    syy = np.bincount(d, weights=y * y, minlength=n_days)
    sxy = np.bincount(d, weights=x * y, minlength=n_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / np.where(n > 0, n, 1)
        vx = sxx - sx * sx / np.where(n > 0, n, 1)
        vy = syy - sy * sy / np.where(n > 0, n, 1)
        r = cov / np.sqrt(vx * vy)
    r[n < MIN_N] = np.nan
    r[~np.isfinite(r)] = np.nan
    return r


def _prepared(df: pd.DataFrame, factor: str, label: str = "r3"):
    """返回 (day codes, uniq dates, x ranks, y ranks, n_days)。"""
    d, day, _ = _day_blocks(df)
    n_days = int(day.max()) + 1
    x = _within_rank(d[factor].to_numpy(dtype=float), day, n_days)
    y = _within_rank(d[label].to_numpy(dtype=float), day, n_days)
    return day, d["date"].to_numpy(), x, y, n_days


def rankic_series(df: pd.DataFrame, factor: str, label: str = "r3") -> pd.Series:
    """逐日 Spearman IC，返回以 date 为索引的 Series。"""
    day, dates, x, y, n_days = _prepared(df, factor, label)
    r = _daily_corr(x, y, day, n_days)
    uniq = np.unique(dates)
    return pd.Series({pd.Timestamp(uniq[i]).date().isoformat(): float(r[i]) for i in range(len(uniq))})


def nw_t(x: np.ndarray, lag: int = NW_LAG) -> float:
    """Newey–West t（Bartlett 核）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    mu = x.mean()
    e = x - mu
    w = e ** 2
    s = w.sum()
    for L in range(1, lag + 1):
        cov_lag = float((e[L:] * e[:-L]).sum())
        s += 2.0 * (1.0 - L / (lag + 1)) * cov_lag
    var = s / n / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var))


def perm_null_mean(df: pd.DataFrame, factor: str, label: str = "r3",
                   n_perm: int = 100, seed: int = 0) -> tuple:
    """日内打乱因子的 RankIC 均值零分布。返回 (p_value, null_mean, null_std)。
    p 为双侧：P(|null| >= |obs|)。打乱的是日内排名向量（等价于打乱值后重排）。"""
    day, _, x, y, n_days = _prepared(df, factor, label)
    obs_arr = _daily_corr(x, y, day, n_days)
    obs = float(np.nanmean(obs_arr))
    # 每日块位置（day 已按 date 排序，故块连续）
    bounds = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    blocks = [slice(bounds[i], bounds[i + 1] if i + 1 < len(bounds) else len(day))
              for i in range(len(bounds))]
    xr = x.copy()
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        pv = xr.copy()
        for b in blocks:
            seg = pv[b]
            fseg = seg[np.isfinite(seg)]
            if len(fseg):
                pv[b][np.isfinite(seg)] = fseg[rng.permutation(len(fseg))]
        nulls[i] = float(np.nanmean(_daily_corr(pv, y, day, n_days)))
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0 or not np.isfinite(obs):
        return float("nan"), float("nan"), float("nan")
    # 有限置换校正（Codex 审计）：p=(1+k)/(B+1)，B 次全不超越时 p=1/(B+1) 而非 0
    k = int((np.abs(nulls) >= abs(obs)).sum())
    p = float((1 + k) / (len(nulls) + 1))
    return p, float(nulls.mean()), float(nulls.std())


# ---------- 诊断指标 ----------

def topk_lift(df: pd.DataFrame, factor: str, ks=(5, 20, 50)) -> dict:
    """Enrich@K = P(y=1|TopK)/P(y=1)，逐日计算后取均值。仅诊断。"""
    d = df.sort_values(["date", factor], ascending=[True, False])
    res = {}
    for k in ks:
        top = d.groupby("date", sort=False).head(k)
        hit = top.groupby("date")["y_prod"].mean()
        base = d.groupby("date")["y_prod"].mean()
        lift = (hit / base.replace(0, np.nan)).dropna()
        res[k] = {"mean": float(lift.mean()) if len(lift) else float("nan"),
                  "n_days": int(len(lift))}
    return res


def auc_day(df: pd.DataFrame, factor: str) -> float:
    """逐日 AUC(factor, y_prod) 的均值。Mann-Whitney 秩公式。仅诊断。"""
    day, _, x, _, n_days = _prepared(df, factor, "y_prod")
    d, _, _ = _day_blocks(df)
    yb = (d["y_prod"].to_numpy() == 1).astype(float)
    yb[~np.isfinite(x)] = np.nan
    ok = np.isfinite(yb)
    dd = day[ok]
    xx = x[ok]
    yy = yb[ok]
    n1 = np.bincount(dd, weights=yy, minlength=n_days)
    n = np.bincount(dd, minlength=n_days).astype(float)
    r1 = np.bincount(dd, weights=xx * yy, minlength=n_days)
    n0 = n - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc[(n1 < 2) | (n0 < MIN_N)] = np.nan
    auc = auc[np.isfinite(auc)]
    return float(auc.mean()) if len(auc) else float("nan")


# ---------- H2 记录与 POOL_S3 ----------

def h2_record(df: pd.DataFrame, factor: str, n_perm: int = 100, seed: int = 0) -> dict:
    """H2：计算子检验 A 与诊断列 B–E，并记录到台账。不决定删除。"""
    ic = rankic_series(df, factor, "r3")
    v = ic.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    t = nw_t(v)
    # B–E 诊断列（只作标注）
    diag = {}
    m = df.copy()
    m["ret1_rank"] = m.groupby("date")["r3"].rank(pct=True)
    hot5 = m[m["ret1_rank"] >= 0.8]
    hot2 = m[m["ret1_rank"] >= 0.9]
    # mkt_up/dn 用 R3 的等权日均值符号近似（诊断列，不进删除）
    daym = m.groupby("date")["r3"].mean()
    up = set(daym[daym > 0].index)
    dn = set(daym[daym <= 0].index)
    for tag, sub in (("hot5", hot5), ("hot2", hot2),
                     ("mkt_up", m[m["date"].isin(up)]), ("mkt_dn", m[m["date"].isin(dn)])):
        iv = rankic_series(sub, factor, "r3").to_numpy(dtype=float)
        iv = iv[np.isfinite(iv)]
        diag[tag] = float(iv.mean()) if len(iv) else float("nan")
        diag[tag + "_t"] = nw_t(iv)
    p_null, null_mean, null_std = perm_null_mean(df, factor, "r3", n_perm=n_perm, seed=seed)
    return {"factor": factor, "n_days": int(len(v)),
            "rankic_mean": float(v.mean()) if len(v) else float("nan"),
            "t_nw": float(t),
            "null_p": float(p_null), "null_mean": float(null_mean),
            **{("diag_" + k): v2 for k, v2 in diag.items()}}


def pool_s3_delete(rec: dict) -> tuple:
    """POOL_S3 删除判据（唯一）：三者同时满足才删除。返回 (delete: bool, reason)。"""
    a = abs(rec.get("rankic_mean", float("nan"))) < 0.015
    b = abs(rec.get("t_nw", float("nan"))) < 2
    c = not (rec.get("null_p", 0.0) < 0.05)
    if a and b and c:
        return True, "POOL_S3: rankic弱且t弱且null不显著"
    return False, ""


def bh_fdr(pvals: dict, q: float = 0.10) -> dict:
    """BH-FDR。输入 {候选: p}，返回 {候选: (显著, q值)}。"""
    names = list(pvals.keys())
    ps = np.array([pvals[n] for n in names], dtype=float)
    order = np.argsort(ps, kind="mergesort")
    m = len(ps)
    qs = np.full(m, np.nan)
    prev = 1.0
    out = {}
    for rank, idx in enumerate(order[::-1], start=1):
        i = m - rank
        j = order[i]
        if np.isfinite(ps[j]):
            prev = min(prev, ps[j] * m / (i + 1))
            qs[j] = prev
    for j, n in enumerate(names):
        out[n] = (bool(np.isfinite(qs[j]) and qs[j] <= q), float(qs[j]) if np.isfinite(qs[j]) else float("nan"))
    return out


def sha12(obj) -> str:
    import hashlib as _h
    return _h.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
