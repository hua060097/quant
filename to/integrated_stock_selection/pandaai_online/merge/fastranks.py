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


def blockwise_rank(v: np.ndarray, day: np.ndarray) -> np.ndarray:
    """日内平均秩（NaN 保持 NaN）。要求 `day` 已排序（块连续）。

    与 `eval_v2._within_rank` 的 lexsort 实现逐值一致，实测快 2.4×。
    """
    v = np.asarray(v, dtype=float)
    r = np.full(len(v), np.nan)
    if len(v) == 0:
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
