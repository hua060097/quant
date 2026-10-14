"""去重与轴管理：公式哈希（grammar）+ 逐日截面 Spearman。

判据：|ρ|>0.7 同轴；0.4~0.7 部分重叠；<0.4 独立。

[CORR-CALIBRATION 2026-10-02] 相关性口径由「每月最后 1 个交易日」改为「每月全部交易日」。
原口径每月仅 1 个截面（~5000 只样本），逐月聚合后 ρ 的估计噪声远大于信号，
既可能漏检真实高相关对，也可能把偶然同序的因子判为同轴。改用月内全部交易日后，
每月约 21 个截面 × ~5000 只，ρ 估计的月内标准误下降约 √21 ≈ 4.6 倍，
且与 definitions 层 `tier2_info_audit` / `task1_equiv_scan` 的口径统一。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import windows
from grammar import Node, expression_hash  # noqa: F401


def formula_hash(node: Node) -> str:
    return expression_hash(node)


def monthly_spearman(d: pd.DataFrame, start: str = '2023-06-01',
                     end: str = '2026-07-10', min_obs: int = 50,
                     data_cut: str | None = windows.DEFAULT_DATA_CUT) -> float:
    """d 需含 date/a/b；对月内【全部交易日】逐日算 Spearman，再跨日取均值。

    名称沿用 monthly_spearman 以保持调用方兼容，语义已由"月末单日"变为"月内全日"。

    数据截止日（2026-10-11）：`end` 的默认值 `'2026-07-10'` 是历史遗留（**含 2026 数据**）。
    `data_cut` 默认取 `windows.DEFAULT_DATA_CUT`，实际用 `min(end, data_cut)`，
    故默认不再读到 2026；`data_cut=windows.CUT_OFF`（'off'）显式恢复遗留行为（历史重放用）。
    """
    end = windows.cut_bound(end, data_cut)
    d = d[(d['date'] >= pd.Timestamp(start)) & (d['date'] <= pd.Timestamp(end))]
    d = d.dropna(subset=['a', 'b'])
    if d.empty:
        return float('nan')
    vals = []
    for _, g in d.groupby('date', sort=True):
        a = g['a'].to_numpy(dtype=float)
        b = g['b'].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_obs:
            continue
        ar = pd.Series(a[ok]).rank().to_numpy()
        br = pd.Series(b[ok]).rank().to_numpy()
        if ar.std() == 0 or br.std() == 0:
            continue
        vals.append(float(np.corrcoef(ar, br)[0, 1]))
    return float(np.mean(vals)) if vals else float('nan')


def axis_label(rho: float) -> str:
    if not np.isfinite(rho):
        return 'unknown'
    if abs(rho) > 0.7:
        return 'same-axis'
    if abs(rho) > 0.4:
        return 'partial-overlap'
    return 'independent'


def max_abs_corr(rhos: dict) -> tuple:
    if not rhos:
        return float('nan'), None
    name = max(rhos, key=lambda k: abs(rhos[k]) if np.isfinite(rhos[k]) else -1.0)
    rho = rhos[name]
    if not np.isfinite(rho):
        return float('nan'), None
    return abs(rho), name
