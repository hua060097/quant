"""vwap 非线性诊断（2026-09-26）：月末截面口径的互信息 / 尾部分离度。

动机：榜2 的 LGB 复合分全是单调线性统计（|RankIC|×|RankICIR|×|mono|），
结构上抓不到非单调（U 形）因子；MI/JSD 补这块盲区。MI 无符号，与管线
`|·|` 哲学一致，方向仍单独存放。

口径：与正交去重同基——只用扫描阶段已求值（且已缓存）的 pos_me 月末截面
因子值 + 同位置标签，不新增任何因子求值。纯 numpy 直方图估计，确定性，
无外部依赖（刻意不用 sklearn 的 kNN 版：慢且要引依赖）。

- mi_monthly：逐截面分位分箱 MI（类 IC 口径），返回均值/Std/归一化 NMI
  [CORR-CALIBRATION 2026-10-02] 分组键语义由「月」改为「交易日」，每个交易日一个截面。
- jsd_tails：全窗 pooled 顶/底分位标签分布的 Jensen-Shannon 散度 [0, ln2)，
  直接对应"能不能选出头尾"（LGB 每天选 topN 的用法）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MI_X_BINS = 10
MI_Y_BINS = 10
MI_MIN_ROWS = 50
JSD_Q = 0.2
JSD_BINS = 20
JSD_MIN_ROWS = 200


def _quantile_edges(v: np.ndarray, k: int) -> np.ndarray | None:
    e = np.unique(np.quantile(v, np.linspace(0.0, 1.0, k + 1)))
    return e if len(e) >= 2 else None


def _binned_mi(x: np.ndarray, y: np.ndarray,
               xb: int = MI_X_BINS, yb: int = MI_Y_BINS) -> tuple[float, float]:
    """单截面分位分箱 MI（nats）与 NMI=MI/sqrt(Hx*Hy)。"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, dtype=float)[ok], np.asarray(y, dtype=float)[ok]
    if len(x) < MI_MIN_ROWS:
        return float('nan'), float('nan')
    ex = _quantile_edges(x, xb)
    ey = _quantile_edges(y, yb)
    if ex is None or ey is None:
        return 0.0, 0.0
    j, _, _ = np.histogram2d(x, y, bins=[ex, ey])
    tot = j.sum()
    if tot <= 0:
        return 0.0, 0.0
    p = j / tot
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(p > 0, p * (np.log(p) - np.log(px) - np.log(py)), 0.0)
        hx = float(-np.nansum(np.where(px > 0, px * np.log(px), 0.0)))
        hy = float(-np.nansum(np.where(py > 0, py * np.log(py), 0.0)))
    mi = float(np.nansum(term))
    nmi = float(mi / np.sqrt(hx * hy)) if hx > 0 and hy > 0 else 0.0
    return mi, nmi


def mi_monthly(f_me: np.ndarray, y_me: np.ndarray, month_group: np.ndarray,
               xb: int = MI_X_BINS, yb: int = MI_Y_BINS) -> dict:
    """逐月截面 MI → {'mi', 'mi_std', 'nmi', 'n_months'}（均值口径，类 IC）。"""
    mis, nmis = [], []
    f_me = np.asarray(f_me, dtype=float)
    y_me = np.asarray(y_me, dtype=float)
    for g in np.unique(month_group):
        idx = np.flatnonzero(month_group == g)
        mi, nmi = _binned_mi(f_me[idx], y_me[idx], xb, yb)
        if np.isfinite(mi):
            mis.append(mi)
            nmis.append(nmi)
    if not mis:
        return {'mi': float('nan'), 'mi_std': float('nan'),
                'nmi': float('nan'), 'n_months': 0}
    return {'mi': float(np.mean(mis)),
            'mi_std': float(np.std(mis, ddof=1)) if len(mis) > 1 else 0.0,
            'nmi': float(np.mean(nmis)), 'n_months': len(mis)}


def _kl(p: np.ndarray, m: np.ndarray) -> float:
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_m = np.where(m > 0, m, 1.0)
        return float(np.sum(np.where(p > 0, p * (np.log(p) - np.log(safe_m)), 0.0)))


def jsd_tails(f: np.ndarray, y: np.ndarray,
              q: float = JSD_Q, bins: int = JSD_BINS) -> float:
    """顶/底分位（默认上下 20%）标签 pooled 分布的 JSD（nats，[0, ln2)）。"""
    f = np.asarray(f, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]
    if len(f) < JSD_MIN_ROWS:
        return float('nan')
    lo = np.quantile(f, q)
    hi = np.quantile(f, 1.0 - q)
    yt, yb = y[f >= hi], y[f <= lo]
    if len(yt) < MI_MIN_ROWS or len(yb) < MI_MIN_ROWS:
        return float('nan')
    edges = np.unique(np.quantile(np.concatenate([yt, yb]),
                                  np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    pt, _ = np.histogram(yt, bins=edges)
    pb, _ = np.histogram(yb, bins=edges)
    if pt.sum() <= 0 or pb.sum() <= 0:
        return 0.0
    pt = pt / pt.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pt + pb)
    return 0.5 * _kl(pt, m) + 0.5 * _kl(pb, m)


def _binned_cond_mi(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    bins: int = 5) -> float:
    """单截面条件互信息 I(Y;X|Z)（nats），链式法则经分位分箱直方图估计。

    I(Y;X|Z) = I(Y;(X,Z)) - I(Y;Z)：联合 (X,Z) 取 bins×bins 格，Z 取 bins 格。
    """
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < MI_MIN_ROWS:
        return float('nan')
    ex = _quantile_edges(x, bins)
    ey = _quantile_edges(y, bins)
    ez = _quantile_edges(z, bins)
    if ex is None or ey is None or ez is None:
        return 0.0

    def _mi2(a, ea, b, eb):
        j, _, _ = np.histogram2d(a, b, bins=[ea, eb])
        tot = j.sum()
        if tot <= 0:
            return 0.0
        p = j / tot
        px = p.sum(axis=1, keepdims=True)
        py = p.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(p > 0, p * (np.log(p) - np.log(px) - np.log(py)), 0.0)
        return float(np.nansum(term))

    # 联合 (X,Z) 的格点索引
    ix = np.clip(np.digitize(x, ex[1:-1]), 0, len(ex) - 2)
    iz = np.clip(np.digitize(z, ez[1:-1]), 0, len(ez) - 2)
    joint = ix * (len(ez) - 1) + iz
    n_jcell = (len(ex) - 1) * (len(ez) - 1)
    ej = np.arange(n_jcell + 1) - 0.5
    mi_joint = _mi2(joint.astype(float), ej, y, ey)
    mi_z = _mi2(z, ez, y, ey)
    return max(0.0, mi_joint - mi_z)


def _binned_joint_mi(x: np.ndarray, z: np.ndarray, y: np.ndarray,
                     bins: int = 5) -> float:
    """单截面联合互信息 I((X,Z);Y)（nats）：JMI 排序项的构件。"""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < MI_MIN_ROWS:
        return float('nan')
    ex = _quantile_edges(x, bins)
    ey = _quantile_edges(y, bins)
    ez = _quantile_edges(z, bins)
    if ex is None or ey is None or ez is None:
        return 0.0
    ix = np.clip(np.digitize(x, ex[1:-1]), 0, len(ex) - 2)
    iz = np.clip(np.digitize(z, ez[1:-1]), 0, len(ez) - 2)
    joint = (ix * (len(ez) - 1) + iz).astype(float)
    ej = np.arange((len(ex) - 1) * (len(ez) - 1) + 1) - 0.5
    j, _, _ = np.histogram2d(joint, y, bins=[ej, ey])
    tot = j.sum()
    if tot <= 0:
        return 0.0
    p = j / tot
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(p > 0, p * (np.log(p) - np.log(px) - np.log(py)), 0.0)
    return max(0.0, float(np.nansum(term)))


def cond_mi_monthly(f_me: np.ndarray, y_me: np.ndarray, z_me: np.ndarray,
                    month_group: np.ndarray, bins: int = 5) -> dict:
    """逐月截面 I(Y;X|Z) → {'cmi', 'cmi_std', 'n_months'}。"""
    out = []
    f_me = np.asarray(f_me, dtype=float)
    y_me = np.asarray(y_me, dtype=float)
    z_me = np.asarray(z_me, dtype=float)
    for g in np.unique(month_group):
        idx = np.flatnonzero(month_group == g)
        v = _binned_cond_mi(f_me[idx], y_me[idx], z_me[idx], bins)
        if np.isfinite(v):
            out.append(v)
    if not out:
        return {'cmi': float('nan'), 'cmi_std': float('nan'), 'n_months': 0}
    return {'cmi': float(np.mean(out)),
            'cmi_std': float(np.std(out, ddof=1)) if len(out) > 1 else 0.0,
            'n_months': len(out)}


def joint_mi_monthly(f_me: np.ndarray, g_me: np.ndarray, y_me: np.ndarray,
                     month_group: np.ndarray, bins: int = 5) -> dict:
    """逐月截面 I((X,Z);Y) → {'jmi', 'jmi_std', 'n_months'}。"""
    out = []
    f_me = np.asarray(f_me, dtype=float)
    g_me = np.asarray(g_me, dtype=float)
    y_me = np.asarray(y_me, dtype=float)
    for g in np.unique(month_group):
        idx = np.flatnonzero(month_group == g)
        v = _binned_joint_mi(f_me[idx], g_me[idx], y_me[idx], bins)
        if np.isfinite(v):
            out.append(v)
    if not out:
        return {'jmi': float('nan'), 'jmi_std': float('nan'), 'n_months': 0}
    return {'jmi': float(np.mean(out)),
            'jmi_std': float(np.std(out, ddof=1)) if len(out) > 1 else 0.0,
            'n_months': len(out)}


def day_index(row_dates) -> tuple[np.ndarray, np.ndarray]:
    """行日期 → (截面位置, 对齐到该位置的交易日标签)；与 build_top 的 pos_me 同语义。

    [CORR-CALIBRATION 2026-10-02] 原 `month_end_index` 只取每月最后一个交易日，
    与 build_top / decorrelate_top / model_suitability 口径不一致（那几处已改为月内全日）。
    现统一返回月内全部交易日，每个交易日一个截面。

    函数名由 month_end_index 改为 day_index；旧名保留为别名以防外部引用。
    """
    rd = pd.Series(np.asarray(row_dates))
    days = rd.dt.strftime('%Y%m%d').to_numpy()
    uniq = np.unique(days)
    if len(uniq) > 1:
        pos = np.flatnonzero(np.isin(days, uniq).to_numpy())
    else:
        # 单一日期：无法构造多截面，退化为单截面以保持可用
        pos = np.arange(len(rd))
    return pos, days[pos]


# 旧名保留（语义已变为逐日截面）
month_end_index = day_index


def nonlinear_for_ctxs(ctxs: dict, labels: list, node) -> dict:
    """对给定 ctx 字典逐标签算逐截面 MI/JSD（供 run_search full 阶段与 rescore 复用）。

    要求 ctx 有 .values(node)/.labels/.row_dates（DataContext 接口）；
    复用 Evaluator 节点缓存（调用前一般已 screen 过），只新增直方图统计。
    """
    out = {}
    for lab in labels:
        try:
            ctx = ctxs[lab]
            vals = np.asarray(ctx.values(node), dtype=float)
            y = np.asarray(ctx.labels, dtype=float)
            pos, groups = day_index(ctx.row_dates)
            d = mi_monthly(vals[pos], y[pos], groups)
            d['jsd'] = jsd_tails(vals[pos], y[pos])
            out[lab] = d
        except Exception:
            out[lab] = {'mi': float('nan'), 'mi_std': float('nan'),
                        'nmi': float('nan'), 'n_months': 0,
                        'jsd': float('nan')}
    return out
