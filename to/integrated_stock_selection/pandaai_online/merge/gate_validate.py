# -*- coding: utf-8 -*-
"""门控 G0 资格（MERGE_V2_PLAN §8.2 的资格层，被搜索线复用）。

2026-10-12 拆分：本模块只保留 G0.1（二值形式资格）与 G0.2（连续形式资格）
及其内部实现（`_segments`/`_half_life`）。此前随手工门控（M1–M4/S1–S3'）存在的
旧 CLI（`main()` 迭代旧 parquet）、G2/G3 的 `NotImplementedError` 空壳、
`coverage_diag`/`g4_independence`/`g1_describe` 已随旧门控定义一并删除：

- G2/G3 的现行实现 = `gate_g2g3.py`（市场线由 `run_gate_pipeline.py` 调用）
- G4 的现行实现 = `gate_g4_clusters.py`（series|cross 双口径）
- G0 + 方向中性度的搜索线预筛 = `gate_search_screen.py`（本模块的 G0 供其复用）
所有阈值来自 §8.2 冻结规则，本模块不得自行调整。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE_VERSION = "v2.8.18"   # 覆盖率口径修订（评估窗分母）

G0_SEG_MED = 20
G0_FLIP_MAX = 8
G0_HALFLIFE = 20
G0_MIN_SEGS = 5
G0_MIN_ONDAYS = 60
G0_MIN_DAYS = 126
G0_COV_MIN = 0.8
G0_STD_MIN = 0.2
G0_MED_MAX = 0.8


def _segments(mask: np.ndarray):
    """返回每段 (start, length)。mask 为 bool 数组。"""
    x = np.asarray(mask, dtype=bool)
    chg = np.flatnonzero(np.diff(np.r_[False, x, False]))
    starts = chg[::2]
    lens = np.diff(chg)[::2]
    return starts, lens


def _half_life(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 30:
        return float("nan")
    v = v - v.mean()
    denom = float((v ** 2).sum())
    if denom <= 0:
        return 0.0
    for lag in range(1, n):
        ac = float((v[lag:] * v[:-lag]).sum()) / denom
        if ac < 0.5:
            return float(lag)
    return float(n)


def g0_binary(dates: pd.Series, p: pd.Series) -> dict:
    """G0.1 二值形式资格。返回 dict（含 status）。"""
    m = np.isfinite(p.to_numpy(dtype=float))
    days = pd.to_datetime(dates).dt.date.to_numpy()[m]
    pv = p.to_numpy(dtype=float)[m]
    n_days = len(days)
    if n_days < G0_MIN_DAYS:
        return {"form": "binary", "g0_status": "insufficient",
                "reason": f"可评估日 {n_days} < {G0_MIN_DAYS}", "n_days": int(n_days)}
    on = pv >= 0.7
    _, lens = _segments(on)
    n_seg = len(lens)
    n_on = int(on.sum())
    years = (days.max() - days.min()).days / 365.25 if len(days) else 0
    flips = int(2 * n_seg)
    flips_py = flips / years if years > 0 else float("nan")
    med = float(np.median(lens)) if n_seg else 0.0
    hl = _half_life(pv)
    if n_seg < G0_MIN_SEGS or n_on < G0_MIN_ONDAYS:
        return {"form": "binary", "g0_status": "insufficient",
                "reason": f"on 段 {n_seg} 个（需≥{G0_MIN_SEGS}），on 日 {n_on}（需≥{G0_MIN_ONDAYS}）",
                "n_days": int(n_days), "n_seg": int(n_seg), "n_on": int(n_on),
                "seg_med": med, "flips_py": float(flips_py), "halflife": float(hl)}
    ok = (med >= G0_SEG_MED) and (flips_py <= G0_FLIP_MAX) and (hl >= G0_HALFLIFE)
    return {"form": "binary", "g0_status": "pass" if ok else "fail",
            "reason": "" if ok else (
                f"段长中位数 {med:.1f}（需≥{G0_SEG_MED}）/年翻转 {flips_py:.1f}"
                f"（需≤{G0_FLIP_MAX}）/半衰期 {hl:.1f}（需≥{G0_HALFLIFE}）"),
            "n_days": int(n_days), "n_seg": int(n_seg), "n_on": int(n_on),
            "seg_med": med, "flips_py": float(flips_py), "halflife": float(hl)}


def g0_continuous(dates: pd.Series, g: pd.Series,
                  eval_start: str | None = None) -> dict:
    """G0.2 连续形式资格。返回 dict（含 status）。

    `eval_start`（v2.8.18 修订）：覆盖率的分母口径。
    - None：旧口径，分母为**全部历史行数**（含评估窗之前的 warm-up）。
    - 日期串：分母为**该日起的日期数**，即评估窗口径。

    旧口径的缺陷（2026-10-10 取证）：需要长预热的门控以全文件天数为分母会被
    误判 fail（预热期计入分母），属度量口径问题，不是门控质量问题。
    按 §8.2.4，本修订作为**新规则版本**（v2.8.18）生效：不回溯修改已判定的
    门控状态。
    """
    v = g.to_numpy(dtype=float)
    d = pd.to_datetime(dates)
    m = np.isfinite(v)
    if eval_start is not None:
        em = (d >= pd.Timestamp(eval_start)).to_numpy()
        m_eval = m & em
        denom = int(em.sum())
        n_days = int(m_eval.sum())
        cov = float(n_days / denom) if denom else float("nan")
        vv = v[m_eval]
    else:
        n_days = int(m.sum())
        cov = float(n_days / len(v)) if len(v) else float("nan")
        vv = v[m]
    if n_days < G0_MIN_DAYS:
        return {"form": "continuous", "g0_status": "insufficient",
                "reason": f"可评估日 {n_days} < {G0_MIN_DAYS}", "n_days": n_days,
                "eval_start": eval_start}
    sd = float(vv.std())
    med = float(np.median(vv))
    ok = (cov >= G0_COV_MIN) and (sd >= G0_STD_MIN) and (abs(med) <= G0_MED_MAX)
    return {"form": "continuous", "g0_status": "pass" if ok else "fail",
            "reason": "" if ok else (
                f"覆盖率 {cov:.2f}（需≥{G0_COV_MIN}）/std {sd:.3f}"
                f"（需≥{G0_STD_MIN}）/|中位数| {abs(med):.3f}（需≤{G0_MED_MAX}）"),
            "n_days": n_days, "cov": float(cov), "std": float(sd), "median": float(med),
            "eval_start": eval_start}
