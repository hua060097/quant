# -*- coding: utf-8 -*-
"""H3 交互检验：F1 条件增益判定（MERGE_V2_PLAN §8.4 v2.8.7 冻结版）。

流程（单候选）：
  F1 = d·x·max(s·g,0)，d = base 方向（base20.json），s = 预注册门控侧（假设登记册）。
  F1 与基因子严格同一样本（内连接有限值行）。
判定：
  主指标 ΔIC（F1−base 日配对差）：块自助 95% 区间下界 > 0，且 6 时间块中 ≥4 块为正。
  块 placebo：循环块置换门控（块长 20 日）200 次，p=(1+k)/201 < 0.05。
  level-leak：时间序列相关 + 分桶对照（只诊断触发条件，不直接判泄漏）。
  FDR：在 F1 检验族上 BH（q≤0.10）；F3 到达后重算。
通过 H3 只代表获得继续评估资格，不等于交易增益已确认。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_v2 as E

BLOCK_BOOT = 5
N_BLOCKS = 6
PLACEBO_BLOCK = 20
N_PERM = 200
SEED = 0


def build_f1(x: np.ndarray, g: np.ndarray, d: int, s: int) -> np.ndarray:
    """F1 = d·x·max(s·g,0)。非有限输入 → NaN。"""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    w = np.maximum(s * g, 0.0)
    out = d * x * w
    out[~(np.isfinite(x) & np.isfinite(g))] = np.nan
    return out


def delta_ic(dates: np.ndarray, ic_f1: np.ndarray, ic_base: np.ndarray) -> dict:
    """ΔIC 主指标：mean(IC_F1 有限日) − mean(IC_base 有限日) + 块自助 CI + 6 块一致性。

    弃权语义（v2.8.8 审计修正）：F1 在门控 off 日弃权（NaN），不参与均值。
    不得用"共同有限日的逐日配对差"，因为 on-day 上 F1 与 base 排名完全相同，
    配对差恒为 0——此前版本有此 bug，已修正。
    输入为按 date 对齐的日序列。
    """
    f = np.asarray(ic_f1, dtype=float)
    b = np.asarray(ic_base, dtype=float)
    dd = np.asarray(dates)
    mf = np.isfinite(f)
    mb = np.isfinite(b)
    out = {"n_days_f1": int(mf.sum()), "n_days_base": int(mb.sum()),
           "mean": float(f[mf].mean() - b[mb].mean()) if mf.any() and mb.any() else float("nan")}

    def _stat(take):
        take = np.asarray(take)
        ff = f[take]
        bb = b[take]
        ff = ff[np.isfinite(ff)]
        bb = bb[np.isfinite(bb)]
        if len(ff) == 0 or len(bb) == 0:
            return float("nan")
        return float(ff.mean() - bb.mean())

    n = len(dd)
    if n < 60 or not mf.any():
        out.update({"ci_lo": float("nan"), "ci_hi": float("nan"),
                    "nblocks_pos": 0, "blocks": []})
        return out
    # CI：循环块自助（块长 5 日），每次重采样后重算两侧均值
    rng = np.random.default_rng(SEED)
    nb = max(1, n // BLOCK_BOOT)
    means = np.empty(2000)
    for i in range(2000):
        starts = rng.integers(0, n, size=nb)
        take = np.concatenate([(starts + j) % n for j in range(BLOCK_BOOT)])[:n]
        means[i] = _stat(take)
    means = means[np.isfinite(means)]
    out["ci_lo"] = float(np.quantile(means, 0.025)) if len(means) else float("nan")
    out["ci_hi"] = float(np.quantile(means, 0.975)) if len(means) else float("nan")
    # 6 块一致性：按全部日历日等分，每块内用同一统计量
    bounds = np.array_split(np.arange(n), N_BLOCKS)
    bmeans = [_stat(b_) for b_ in bounds]
    out["blocks"] = bmeans
    out["nblocks_pos"] = int(sum(1 for v in bmeans if np.isfinite(v) and v > 0))
    out["block_dates"] = [str(pd.Timestamp(dd[b_[0]]).date()) if len(b_) else None for b_ in bounds]
    return out


def slope_test(ic_base: np.ndarray, g_daily: np.ndarray) -> dict:
    """条件斜率（仅解释）：RankIC_base = a + b·g + e，NW 标准误。"""
    m = np.isfinite(ic_base) & np.isfinite(g_daily)
    y, x = np.asarray(ic_base)[m], np.asarray(g_daily)[m]
    if len(y) < 60:
        return {"b": float("nan"), "t": float("nan"), "n": int(len(y))}
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n = len(y)
    L = 3
    meat = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1)
        G = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        meat = meat + w * (G + G.T)
    try:
        bread = np.linalg.inv(X.T @ X)
        V = bread @ meat @ bread
        se = float(np.sqrt(max(V[1, 1], 0)))
        t = float(beta[1] / se) if se > 0 else float("nan")
    except np.linalg.LinAlgError:
        t = float("nan")
    return {"b": float(beta[1]), "t": t, "n": int(n)}


def block_placebo(daily_ic_f1_fn, gate_daily: np.ndarray, base_daily_ic: np.ndarray,
                  n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """循环块置换门控（块长 20 日）200 次。daily_ic_f1_fn(g_perm) -> 日 IC 序列。
    返回 p=(1+k)/201、零分布分位与真实门控的位置百分位。"""
    g = np.asarray(gate_daily, dtype=float)
    n = len(g)
    # 缺失位置固定：只在有限值位置之间置换，NaN 位置不动（v2.8.8 审计补丁）。
    # 否则置换会改变哪些日子缺失，零分布失真。
    fin = np.isfinite(g)

    def _delta(f1_ic):
        f1_ic = np.asarray(f1_ic, dtype=float)
        b = np.asarray(base_daily_ic, dtype=float)
        ff = f1_ic[np.isfinite(f1_ic)]
        bb = b[np.isfinite(b)]
        if len(ff) == 0 or len(bb) == 0:
            return float("nan")
        # 弃权语义：分开取均值再相减（不得用 nanmean(F1-base)，见 v2.8.8 审计）。
        return float(ff.mean() - bb.mean())

    obs = _delta(daily_ic_f1_fn(g))
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    gf = g[fin]
    nf = len(gf)
    nb = max(1, nf // PLACEBO_BLOCK)
    pieces = np.array_split(gf.copy(), nb)
    for i in range(n_perm):
        # 块置换块顺序（保留块内自相关，破坏门控与收益的对齐）
        order = rng.permutation(nb)
        gpf = np.concatenate([pieces[j] for j in order])[:nf]
        gp = np.full(n, np.nan)
        gp[fin] = gpf
        nulls[i] = _delta(daily_ic_f1_fn(gp))
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0 or not np.isfinite(obs):
        return {"p": float("nan"), "obs": obs, "null_n": 0, "pct": float("nan")}
    k = int((np.abs(nulls) >= abs(obs)).sum())
    p = float((1 + k) / (len(nulls) + 1))
    return {"p": p, "obs": obs, "null_n": int(len(nulls)),
            "null_q50": float(np.median(nulls)), "null_q95": float(np.quantile(nulls, 0.95)),
            "pct": float((np.abs(nulls) <= abs(obs)).mean())}


def level_leak(dates: np.ndarray, spread_f1: np.ndarray, spread_base: np.ndarray,
               g_abs: np.ndarray) -> dict:
    """level-leak 诊断（时间序列相关 + 分桶对照）。
    触发风险的条件（需同时满足）：|corr(spread_F1,|g|)|>0.8 且 ΔIC 不显著
    且基因子对照 corr(spread_base,|g|)<0.5。否则只记为幅度依赖，不判泄漏。"""
    def _sp(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30:
            return float("nan")
        ra = pd.Series(a[m]).rank().to_numpy()
        rb = pd.Series(b[m]).rank().to_numpy()
        ra -= ra.mean()
        rb -= rb.mean()
        den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / den) if den > 0 else float("nan")
    c_f1 = _sp(np.asarray(spread_f1, dtype=float), np.asarray(g_abs, dtype=float))
    c_base = _sp(np.asarray(spread_base, dtype=float), np.asarray(g_abs, dtype=float))
    return {"corr_f1_abs_g": c_f1, "corr_base_abs_g": c_base,
            "risk": bool(abs(c_f1) > 0.8) if np.isfinite(c_f1) else False}


def daily_spread(dates: np.ndarray, score_rank_pct: np.ndarray, ret: np.ndarray,
                 day: np.ndarray, n_days: int, q: float = 0.2) -> np.ndarray:
    """每日 spread：Top 分位 R3 均值减 Bottom 分位均值。score_rank_pct 为日内百分位排名。"""
    out = np.full(n_days, np.nan)
    ok = np.isfinite(score_rank_pct) & np.isfinite(ret)
    d, s, r = day[ok], score_rank_pct[ok], ret[ok]
    hi = s >= (1 - q)
    lo = s <= q
    mean_hi = np.bincount(d[hi], weights=r[hi], minlength=n_days) / np.maximum(
        np.bincount(d[hi], minlength=n_days), 1)
    mean_lo = np.bincount(d[lo], weights=r[lo], minlength=n_days) / np.maximum(
        np.bincount(d[lo], minlength=n_days), 1)
    cnt_hi = np.bincount(d[hi], minlength=n_days)
    cnt_lo = np.bincount(d[lo], minlength=n_days)
    out[:] = mean_hi - mean_lo
    out[(cnt_hi < 20) | (cnt_lo < 20)] = np.nan
    return out
