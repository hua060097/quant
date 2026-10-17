# -*- coding: utf-8 -*-
"""评估加速层：日内排名的更快实现 + 排名复用 + 只算 sxy 的置换零分布。

**为什么要这个模块**（2026-10-10 profile 结论）：
`perm_null_mean` 单因子在 4.99M 行面板上耗 12~15s，其中
  - `_within_rank` 双键 lexsort 占 **48%**（x、y 各一次，且 `h2_record` 里
    会因 `rankic_series` + `perm_null_mean` 各调一次 `_prepared` 而**重复计算**）；
  - `_daily_corr` 21 次占 26%；
  - `rng.permutation` 22300 次占 14%。
三个加速点：
  1. **排名**：改用 `scipy.stats.rankdata` 逐块计算（块已连续，无需双键全排序），
     实测 **2.4×**（3.16s→1.32s），结果与 lexsort 实现逐值一致。
  2. **复用**：y（标签 r3）的日内排名对**所有因子相同**，每个面板只算一次；
     x 每次因子只算一次。旧实现每因子算 2 次 x + 2 次 y。
  3. **置换**：原实现只置换有限值（NaN 位置固定），故每天的
     n、Σx、Σx²、Σy、Σy² 都是置换不变量，**只有 Σxy 变**。
     故每次置换只需 1 次 bincount 而非 7 次。同 seed 下与原实现**逐位一致**。

用法：
    from fastranks import PanelRanker
    pr = PanelRanker(df)                  # 一次
    for f in factors:
        xr = pr.x_rank(f)                 # 每因子一次
        ic = pr.daily_ic(xr)
        p, mu, sd = pr.perm_null(xr, n_perm=50)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import eval_v2 as E

MIN_N = E.MIN_N
NW_LAG = E.NW_LAG


def mad_zscore_soft(values: np.ndarray, day, soft_cap: float = 5.0) -> np.ndarray:
    """MAD 鲁棒 Z-score + 超 cap 对数软压缩（逐日截面，严格单调，无需硬缩尾）。

    z = (x - median) / (1.4826*MAD)；MAD<=1e-12 时回退 1.2533*mean|x-med|，
    再不行回退 std，仍退化则 NaN。|z|<=cap 原值保留（Top 极值梯度完整）；
    |z|>cap 用 sign*(cap + ln(1+|z|-cap)) 软压缩（保序，坏点不撑爆直方图）。
    day 可为任意可 factorize 的日期键（内部编码，与行序无关）。
    """
    v = np.asarray(values, dtype=float)
    codes, _ = pd.factorize(np.asarray(day), sort=True)
    s = pd.Series(v)
    g = s.groupby(codes, sort=False)
    med = g.transform('median')
    mad = (s - med).abs().groupby(codes, sort=False).transform('median')
    meanabs = (s - med).abs().groupby(codes, sort=False).transform('mean')
    std = g.transform('std')
    tol = 1e-12
    scale = np.where(np.asarray(mad) > tol, 1.4826 * np.asarray(mad),
                     np.where(np.asarray(meanabs) > tol,
                              1.2533 * np.asarray(meanabs),
                              np.asarray(std, dtype=float)))
    with np.errstate(invalid='ignore', divide='ignore'):
        z = (v - np.asarray(med, dtype=float)) / np.where(scale > tol, scale, np.nan)
    az = np.abs(z)
    cap = float(soft_cap)
    with np.errstate(invalid='ignore', divide='ignore'):
        _soft = np.sign(z) * (cap + np.log1p(np.maximum(az - cap, 0.0)))
    out = np.where(az <= cap, z, _soft)
    out[~np.isfinite(np.asarray(v))] = np.nan
    return out


def centered_u(values: np.ndarray, day: np.ndarray, nd: int, d: int) -> np.ndarray:
    """有向居中百分位秩 u = d*(RankPct-0.5) ∈ [-0.5, 0.5]（F1v3 全族统一定义）。

    全仓唯一定义点（收敛 f1_batch._centered_u、gate_g2g3_sector._centered_u
    及 f1_sector switch 分支手抄版；三者曾因分抄出现双重乘 d 分叉）。
    约定：对**原始值**排名再乘一次 d；缺失保持 NaN。
    """
    rk = blockwise_rank(np.where(np.isfinite(values), values, np.nan), day)
    dcnt = np.bincount(day, minlength=nd).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = rk / np.maximum(dcnt[day], 1)
    u = int(d) * (pct - 0.5)
    u[~np.isfinite(np.asarray(values, dtype=float))] = np.nan
    return u


def blockwise_rank(v: np.ndarray, day: np.ndarray) -> np.ndarray:
    """日内平均秩（NaN 保持 NaN）。

    要求 `day` 块连续（同日行相邻；单调非降即满足）。若传入 stock-major
    等未排序 `day`（如 `evaluator.load_panel()` 行序经 `pos_me` 切片后的
    `_day_codes`），自动稳定排序后计算再映射回原行序——与已排序输入逐值一致。
    不加此守卫时 660 万行会被切成 660 万个长度 1 碎块：每次调用触发上千万次
    单元素 `rankdata`（~80 秒/对）且碎块秩恒为 1.0 使下游相关恒为 NaN
    （`habitat_seats` early-break 永不触发 + 正交检查静默放行）。
    与 `eval_v2._within_rank` 的 lexsort 实现逐值一致，实测快 2.4×；
    已排序输入仅多一道向量化单调性检查，零行为变化。
    """
    v = np.asarray(v, dtype=float)
    day = np.asarray(day)
    r = np.full(len(v), np.nan)
    if len(v) == 0:
        return r
    if len(day) > 1 and bool(np.any(day[1:] < day[:-1])):
        order = np.argsort(day, kind="stable")
        r_sorted = blockwise_rank(v[order], day[order])
        r[order] = r_sorted
        return r
    b = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    e = np.r_[b[1:], len(v)]
    for i in range(len(b)):
        lo, hi = int(b[i]), int(e[i])
        seg = v[lo:hi]
        ok = np.isfinite(seg)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        r[lo + idx] = rankdata(seg[idx], method="average")
    return r


class PanelRanker:
    """面板级缓存：day 结构 + y 日内排名（跨因子复用）。"""

    def __init__(self, df: pd.DataFrame, label: str = "r3", verbose: bool = False):
        d, day, _ = E._day_blocks(df)
        self.day = day
        self.dates = d["date"].to_numpy()
        self.n_days = int(day.max()) + 1
        self._df = d
        self.uniq = np.unique(self.dates)
        self.y = blockwise_rank(d[label].to_numpy(dtype=float), day)
        if verbose:
            print(f"PanelRanker: {len(d):,} 行, {self.n_days} 天, y 排名已缓存")

    def x_rank(self, factor: str) -> np.ndarray:
        return blockwise_rank(self._df[factor].to_numpy(dtype=float), self.day)

    def values(self, factor: str) -> np.ndarray:
        return self._df[factor].to_numpy(dtype=float)

    def daily_ic(self, xr: np.ndarray) -> np.ndarray:
        return E._daily_corr(xr, self.y, self.day, self.n_days)

    def ic_series(self, xr: np.ndarray) -> pd.Series:
        r = self.daily_ic(xr)
        u = self.uniq
        return pd.Series({pd.Timestamp(u[i]).date().isoformat(): float(r[i])
                          for i in range(len(u))})

    def perm_null(self, xr: np.ndarray, n_perm: int = 50,
                  seed: int = 0) -> tuple[float, float, float]:
        """日内打乱 xr 的有限值（NaN 位置固定）→ RankIC 均值的零分布。

        与 `eval_v2.perm_null_mean` 同 seed 下**逐位一致**：RNG 消费顺序
        （按块序 permutation）与 p 值公式 (1+k)/(B+1) 均不变。
        """
        y = self.y
        day = self.day
        n_days = self.n_days
        obs = float(np.nanmean(E._daily_corr(xr, y, day, n_days)))

        ok = np.isfinite(xr) & np.isfinite(y)
        d = day[ok]
        xv = xr[ok]
        yv = y[ok]
        n = np.bincount(d, minlength=n_days).astype(float)
        sx = np.bincount(d, weights=xv, minlength=n_days)
        sy = np.bincount(d, weights=yv, minlength=n_days)
        sxx = np.bincount(d, weights=xv * xv, minlength=n_days)
        syy = np.bincount(d, weights=yv * yv, minlength=n_days)
        np.maximum(n, 1, out=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
        b = np.flatnonzero(np.r_[True, d[1:] != d[:-1]]) if len(d) else np.array([], int)
        e = np.r_[b[1:], len(d)]
        segs = [slice(int(b[i]), int(e[i])) for i in range(len(b))]

        rng = np.random.default_rng(seed)
        nulls = np.empty(n_perm)
        for i in range(n_perm):
            xp = xv.copy()
            for s in segs:
                m = s.stop - s.start
                if m > 1:
                    xp[s] = xp[s][rng.permutation(m)]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (np.bincount(d, weights=xp * yv, minlength=n_days)
                     - sx * sy / n) / denom
            r[n < MIN_N] = np.nan
            r[~np.isfinite(r)] = np.nan
            nulls[i] = float(np.nanmean(r))
        nulls = nulls[np.isfinite(nulls)]
        if len(nulls) == 0 or not np.isfinite(obs):
            return float("nan"), float("nan"), float("nan")
        k = int((np.abs(nulls) >= abs(obs)).sum())
        return float((1 + k) / (len(nulls) + 1)), float(nulls.mean()), float(nulls.std())

    def h2_core(self, factor: str, n_perm: int = 50, seed: int = 0) -> dict:
        """H2 核心列（不含 diag B–E）：rankic_mean / t_nw / null_p。"""
        xr = self.x_rank(factor)
        ic = self.daily_ic(xr)
        v = ic[np.isfinite(ic)]
        p, mu, sd = self.perm_null(xr, n_perm=n_perm, seed=seed)
        return {"factor": factor, "n_days": int(len(v)),
                "rankic_mean": float(v.mean()) if len(v) else float("nan"),
                "t_nw": float(E.nw_t(v)),
                "null_p": p, "null_mean": mu, "null_std": sd}
