"""条件 RankIC 诊断（2026-09-24）：热点子集（主管线 COND-IC 同口径）+ 市场状态族。

移植自 main_alpha_*_妖股_copy.py / multi_horizon_factor_test 的 cond_spec 设计
（掩码 + 子集内 label 秩预计算共享，因子侧子集内重排名），键名保持一致：
  cond_{tag}_rank_ic_{mean,std,ir,tstat,pvalue,positive_ratio} + cond_{tag}_n_mean/_ndays

掩码族（对齐主管线口径，RET1 与 pct_chg/100 单调一致 → 截面分位相同）：
  hot5_q{70,80,90}：RET1 的 5 日累计（rolling5.sum）当日截面分位 ≥ q/100
                    且 换手 2 日均值在过去 60 日分位 ≥ 0.6
  hot5_cold       ：5 日累计分位 < 0.8 的补集（可排但不在主档）
  hot2_*          ：同上，2 日累计
  mkt_up/mkt_dn   ：市场等权日收益 >0 / <0（全市场同日同掩码；新增维度）

用途：全市场 RankIC 高 ≠ 栖息地内有用——hot* 直接度量"实际交易票内的区分力"，
mkt_* 度量市况条件稳定性。均为诊断，不进 rank_score。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist

import _paths  # noqa: F401
from evaluator import ensure_local_fields

MIN_STOCKS = 30
RET_PCTS = (70, 80, 90)
TURN_MIN = 0.6
TURN_LOOKBACK = 60
TURN_MIN_PERIODS = 20


def _day_slices(dcodes: np.ndarray, n_day: np.ndarray) -> np.ndarray:
    """date-order 下逐日分段的起止（cumsum offsets）。"""
    return np.concatenate([[0], np.cumsum(n_day)])


def _rank_pct_per_day(values: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """date-order 连续分段内逐日 rank pct（average ties；NaN 输入→NaN 排名）。

    与主管线 _group_rank_pct_sorted 同语义（scipy rankdata 逐日切片）。
    """
    out = np.full(len(values), np.nan)
    for k in range(len(seg) - 1):
        a, b = seg[k], seg[k + 1]
        if b <= a:
            continue
        v = values[a:b]
        fin = np.isfinite(v)
        if fin.sum() < 2:
            continue
        from scipy.stats import rankdata
        r = rankdata(v[fin], method="average") / float(fin.sum())
        out[np.flatnonzero(fin) + a] = r
    return out


def build_cond_prep(frame: pd.DataFrame) -> dict:
    """全面板条件掩码预计算（全因子共享，一次构建）。

    返回 {'tags': {tag: bool数组(面板行序)}，'note': str}。panel 行序 (stock_code, date)。
    """
    ensure_local_fields(frame)
    codes = frame['stock_code']
    dates = frame['date']
    ret = frame.groupby(codes, sort=False)['CLOSE'].pct_change(fill_method=None)
    turn = frame['TURNOVER']

    g = ret.groupby(codes, sort=False)
    cond_ret5 = g.transform(lambda s: s.rolling(5, min_periods=5).sum())
    cond_ret2 = g.transform(lambda s: s.rolling(2, min_periods=2).sum())
    turn2 = turn.groupby(codes, sort=False).transform(
        lambda s: s.rolling(2, min_periods=2).mean())
    turn_hist = turn2.groupby(codes, sort=False).transform(
        lambda s: s.rolling(TURN_LOOKBACK, min_periods=TURN_MIN_PERIODS).rank(pct=True))

    mkt = frame['MKT_RET']

    # 逐日截面分位（date-order 排序一次，逐日 rankdata）
    order = np.argsort(dates.to_numpy(), kind="stable")
    codes_sorted, uniq_dates = pd.factorize(dates.to_numpy()[order], sort=True)
    dcodes = np.zeros(len(frame), dtype=np.int64)
    dcodes[order] = codes_sorted
    n_day = np.bincount(dcodes, minlength=len(uniq_dates))
    seg = _day_slices(dcodes, n_day)
    r5 = _rank_pct_per_day(cond_ret5.to_numpy(dtype=float)[order], seg)[np.argsort(order)]
    r2 = _rank_pct_per_day(cond_ret2.to_numpy(dtype=float)[order], seg)[np.argsort(order)]

    tags: dict[str, np.ndarray] = {}
    fin5 = np.isfinite(r5) & np.isfinite(turn_hist.to_numpy(dtype=float))
    fin2 = np.isfinite(r2) & np.isfinite(turn_hist.to_numpy(dtype=float))
    for q in RET_PCTS:
        tags[f'hot5_q{q}'] = fin5 & (r5 >= q / 100.0) & (turn_hist.to_numpy() >= TURN_MIN)
        tags[f'hot2_q{q}'] = fin2 & (r2 >= q / 100.0) & (turn_hist.to_numpy() >= TURN_MIN)
    tags['hot5_cold'] = np.isfinite(r5) & (r5 < 0.8)
    tags['hot2_cold'] = np.isfinite(r2) & (r2 < 0.8)
    mkt_f = mkt.to_numpy(dtype=float)
    tags['mkt_up'] = np.isfinite(mkt_f) & (mkt_f > 0)
    tags['mkt_dn'] = np.isfinite(mkt_f) & (mkt_f < 0)
    return {'tags': tags, 'ret_pcts': RET_PCTS, 'turn_min': TURN_MIN}


def _cond_ctx_layout(ctx, prep: dict) -> dict:
    """把面板掩码映射到 ctx 行并压缩为 day-contiguous 索引（每 ctx 一次，全因子共享）。"""
    cache = getattr(ctx, '_cond_layout_cache', None)
    if cache is not None:
        return cache
    di = ctx.di
    order_di = np.argsort(di, kind="stable")  # ctx 行按截面日稳定排序
    d_sorted = di[order_di]
    n_day = np.bincount(d_sorted)
    seg = _day_slices(d_sorted, n_day)
    D = len(n_day)
    out = {'order_di': order_di, 'seg': seg, 'D': D, 'n_day': n_day, 'tags': {}}
    for tag, mask_full in prep['tags'].items():
        m_ctx = mask_full[ctx._pos]
        rows = np.flatnonzero(m_ctx[order_di])          # day-order 下的行位置
        out['tags'][tag] = {'rows': rows, 'dcodes': d_sorted[rows]}
    ctx._cond_layout_cache = out
    return out


def _stat_block(ic_by_day: np.ndarray, prefix: str) -> dict:
    ic = ic_by_day[np.isfinite(ic_by_day)]
    out: dict = {}
    if len(ic) < 3:
        return {f'{prefix}_{k}': np.nan for k in
                ('mean', 'std', 'ir', 'tstat', 'pvalue', 'positive_ratio')}
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    out[f'{prefix}_mean'] = mean
    out[f'{prefix}_std'] = std
    out[f'{prefix}_ir'] = mean / std if std > 0 else np.nan
    t = mean * np.sqrt(len(ic)) / std if std > 0 else np.nan
    out[f'{prefix}_tstat'] = t
    out[f'{prefix}_pvalue'] = float(2.0 * _t_dist.sf(abs(t), len(ic) - 1)) if np.isfinite(t) else np.nan
    out[f'{prefix}_positive_ratio'] = float((ic > 0).mean())
    return out


def cond_ic_metrics(ctx, source, prep: dict, name: str = 'factor',
                    min_stocks: int = MIN_STOCKS,
                    tags: tuple | list | None = None) -> dict:
    """单因子在全部条件族上的子集内逐日 RankIC（与主管线 cond_* 键名一致）。

    source: AST 节点或因子值数组（面板行序；数组须与 ctx 面板同长，如
    f_full[ctx._pos]）——refs 面板因子（无公式）用数组入口。
    tags: 只算指定条件族（默认 None = 全算 10 个，保持历史行为）。
    habitat_seats 只需 ('hot5_q80',)，habitat_pool 只需 ('mkt_up','mkt_dn')，
    省去无关族的逐日重排名。布局（_cond_ctx_layout）仍全量构建并走 ctx 缓存，
    与 tags 无关；返回字典仅含请求族的键。"""
    layout = _cond_ctx_layout(ctx, prep)
    order_di = layout['order_di']
    seg = layout['seg']
    D = layout['D']
    values = source if isinstance(source, np.ndarray) else ctx.values(source)
    # ctx.labels 同步到 day-order
    lab_sorted = ctx.labels[order_di]
    out: dict = {}
    _want = set(tags) if tags is not None else None
    for tag, info in layout['tags'].items():
        if _want is not None and tag not in _want:
            continue
        rows = info['rows']
        dc = info['dcodes']
        key = f'cond_{tag}'
        if len(rows) < min_stocks * 3:
            out.update({f'{key}_rank_ic_{k}': np.nan for k in
                        ('mean', 'std', 'ir', 'tstat', 'pvalue', 'positive_ratio')})
            out[f'{key}_n_mean'] = np.nan
            out[f'{key}_ndays'] = 0
            continue
        v = values[order_di][rows]
        lab = lab_sorted[rows]
        fin = np.isfinite(v) & np.isfinite(lab)
        v_f, lab_f, dc_f = v[fin], lab[fin], dc[fin]
        n_day_c = np.bincount(dc_f, minlength=D)
        cnn = n_day_c
        cok = cnn >= min_stocks
        if not cok.any():
            out.update({f'{key}_rank_ic_{k}': np.nan for k in
                        ('mean', 'std', 'ir', 'tstat', 'pvalue', 'positive_ratio')})
            out[f'{key}_n_mean'] = np.nan
            out[f'{key}_ndays'] = 0
            continue
        # 子集内逐日重排名（先按日分段：dc_f 非降 → reduceat 分段）
        seg_c = np.concatenate([[0], np.cumsum(n_day_c)])
        fr = _rank_pct_per_day(v_f, seg_c)
        lr = _rank_pct_per_day(lab_f, seg_c)
        ok_rows = cok[dc_f] & np.isfinite(fr) & np.isfinite(lr)
        sf = np.bincount(dc_f[ok_rows], weights=fr[ok_rows], minlength=D)
        sl = np.bincount(dc_f[ok_rows], weights=lr[ok_rows], minlength=D)
        sff = np.bincount(dc_f[ok_rows], weights=fr[ok_rows] * fr[ok_rows], minlength=D)
        sll = np.bincount(dc_f[ok_rows], weights=lr[ok_rows] * lr[ok_rows], minlength=D)
        sfl = np.bincount(dc_f[ok_rows], weights=fr[ok_rows] * lr[ok_rows], minlength=D)
        nn = np.maximum(cnn, 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            cov = sfl - sf * sl / nn
            den = np.sqrt(np.maximum(sff - sf * sf / nn, 0.0)
                          * np.maximum(sll - sl * sl / nn, 0.0))
            ic = np.where(cok & (den > 0), cov / np.maximum(den, 1e-12), np.nan)
        block = _stat_block(ic, f'{key}_rank_ic')
        out.update(block)
        out[f'{key}_n_mean'] = float(cnn[cok].mean())
        out[f'{key}_ndays'] = int(cok.sum())
    return out


def summarize(res: dict, tags=('hot5_q80', 'hot2_q80', 'hot5_cold', 'mkt_up', 'mkt_dn')) -> str:
    """控制台紧凑摘要：主 tag 的 mean RankIC。"""
    parts = []
    for t in tags:
        v = res.get(f'cond_{t}_rank_ic_mean')
        parts.append(f'{t}={"-" if v is None or not np.isfinite(v) else "%+.4f" % v}')
    return ' '.join(parts)
