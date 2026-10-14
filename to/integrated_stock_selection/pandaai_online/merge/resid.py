"""风险轴残差 IC（秩空间顺序正交化）：识别"伪装的风格暴露"。

轴顺序：MARKET_CAP → TURNOVER → VOL20。顺序正交化而非多元回归，避免市值×换手
（实测 ρ≈-0.81）共线导致的系数不稳。

context 诊断链（2026-09-24 层1，仅诊断不进 rank_score）：在风险轴之前先剥离
SW2 行业均值（行业内去均值）+ BETA60 秩正交化——回答"行业/市场 beta 解释掉之后
因子还剩多少"。产出 drop_ind = 1 − ctx_sia/raw_sia。

实现：每个调仓截面内，把因子 rank 依次对各轴 rank 回归取残差（含去均值），
再对残差跑 screen_one（与全量口径一致），得到残差 SiA/RankIC/换手/净超额。
用法（0 算力）：
  python resid.py --factor amihud120        # 快速样本（800股）诊断
  python resid.py --factor amihud120 --full # 全量面板
  python resid.py --calibrate               # 全部夹具 + 池成员对照表
  python resid.py --calibrate --context     # 附带行业/beta 中性化诊断（层1）
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import ensure_local_fields, load_panel
from field_catalog import sw2_mapping
from fitness import DataContext, build_fast_context, tier1
from fixtures import FIXTURES
from grammar import Node, field, ts, virtual_node

AXIS_ORDER = ('MARKET_CAP', 'TURNOVER', 'VOL20')


def axis_values(ctx: DataContext) -> dict:
    cache = getattr(ctx, '_axis_cache', None)
    if cache is not None:
        return cache
    ev = ctx.evaluator
    mkt = ev.values(field('MARKET_CAP'))[ctx._pos]
    turn = ev.values(field('TURNOVER'))[ctx._pos]
    vol20 = ev.values(ts('STDDEV', virtual_node('RET1'), 20, mp=20))[ctx._pos]
    cache = {'MARKET_CAP': mkt, 'TURNOVER': turn, 'VOL20': vol20}
    ctx._axis_cache = cache
    return cache


def context_axis_values(ctx: DataContext) -> dict:
    """层1 上下文轴（惰性）：BETA60 值 + 逐股行业标签。"""
    cache = getattr(ctx, '_ctx_axis_cache', None)
    if cache is not None:
        return cache
    ensure_local_fields(ctx.evaluator.frame)
    beta = ctx.evaluator.values(field('BETA60'))[ctx._pos]
    ind = pd.Series(ctx.stock_codes.astype(str)).map(sw2_mapping()) \
        .fillna('__NA__').to_numpy()
    cache = {'BETA60': beta, 'IND': ind}
    ctx._ctx_axis_cache = cache
    return cache


def _rank_pct(v: np.ndarray) -> np.ndarray:
    n = len(v)
    ranks = np.empty(n, dtype=float)
    order = np.argsort(v, kind='mergesort')
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return ranks / n


def _date_groups(ctx: DataContext) -> list:
    cache = getattr(ctx, '_di_groups', None)
    if cache is None:
        order = np.argsort(ctx.di, kind='mergesort')
        cuts = np.flatnonzero(np.diff(ctx.di[order], prepend=-1))
        cache = np.split(order, cuts[1:])
        ctx._di_groups = cache
    return cache


def residual_values(ctx: DataContext, values: np.ndarray, min_obs: int = 50,
                    with_context: bool = False) -> np.ndarray:
    """逐截面顺序正交化；返回秩空间残差（未覆盖=NaN）。

    with_context=True（层1）：链首先做 SW2 行业内均值剥离（行业成员≥5 才剥）+
    BETA60 秩正交化，再走 MARKET_CAP→TURNOVER→VOL20 风险轴。
    """
    resid = np.full(len(values), np.nan)
    axes = axis_values(ctx)
    ctx_axes = context_axis_values(ctx) if with_context else None
    for idx in _date_groups(ctx):
        if len(idx) < min_obs:
            continue
        f = values[idx]
        ok = np.isfinite(f)
        for axis in AXIS_ORDER:
            ok &= np.isfinite(axes[axis][idx])
        if with_context:
            ok &= np.isfinite(ctx_axes['BETA60'][idx])
        if int(ok.sum()) < min_obs:
            continue
        sel = idx[ok]
        r = _rank_pct(f[ok])
        r = r - r.mean()
        if with_context:
            inds = ctx_axes['IND'][sel]
            for s in np.unique(inds):
                m = inds == s
                if int(m.sum()) >= 5:  # 行业成员太少不剥离，避免噪声
                    r[m] = r[m] - r[m].mean()
        for axis in AXIS_ORDER:
            ra = _rank_pct(axes[axis][sel])
            ra = ra - ra.mean()
            den = float(np.dot(ra, ra))
            if den <= 0:
                continue
            beta = float(np.dot(r, ra)) / den
            r = r - beta * ra
        if with_context:
            rb = _rank_pct(ctx_axes['BETA60'][sel])
            rb = rb - rb.mean()
            den = float(np.dot(rb, rb))
            if den > 0:
                beta = float(np.dot(r, rb)) / den
                r = r - beta * rb
        resid[sel] = r
    return resid


def resid_metrics(ctx: DataContext, source, name: str = 'factor',
                  context: bool = False) -> dict:
    values = source if isinstance(source, np.ndarray) else ctx.values(source)
    raw = tier1(ctx.screen(values, name=f'{name}:raw'))
    rd = tier1(ctx.screen(residual_values(ctx, values), name=f'{name}:resid'))
    drop = None
    if raw['sia'] and np.isfinite(raw['sia']) and raw['sia'] > 0:
        drop = float(1.0 - (rd['sia'] / raw['sia']))
    out = {'raw_sia': raw['sia'], 'resid_sia': rd['sia'],
           'raw_rank_ic': raw['rank_ic'], 'resid_rank_ic': rd['rank_ic'],
           'resid_ic': rd['ic'], 'resid_turnover': rd['turnover'],
           'resid_net': rd['net'], 'drop': drop}
    if context:
        # 层1（2026-09-24）：行业+beta 剥离后的残差诊断；仅报告不进 rank_score
        rc = tier1(ctx.screen(residual_values(ctx, values, with_context=True),
                              name=f'{name}:resid_ctx'))
        out['resid_ctx_sia'] = rc['sia']
        out['resid_ctx_rank_ic'] = rc['rank_ic']
        if raw['sia'] and np.isfinite(raw['sia']) and raw['sia'] > 0:
            out['drop_ind'] = float(1.0 - (rc['sia'] / raw['sia']))
        else:
            out['drop_ind'] = None
    return out


def _fmt(x, spec='%.4f'):
    if x is None:
        return '-'
    try:
        if not np.isfinite(x):
            return '-'
    except TypeError:
        return '-'
    return spec % x


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='风险轴残差 IC 诊断（0 算力）')
    ap.add_argument('--factor', default='')
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--context', action='store_true',
                    help='附带行业/beta 中性化诊断（层1：drop_ind 列）')
    ap.add_argument('--stocks', type=int, default=800)
    args = ap.parse_args()
    frame = load_panel()
    if args.full:
        ctx = DataContext(frame=frame)
    else:
        ctx = build_fast_context(frame, n_stocks=args.stocks)
    targets = {}
    if args.factor:
        if args.factor in FIXTURES:
            targets[args.factor] = FIXTURES[args.factor]()
        else:
            raise SystemExit(f'unknown fixture {args.factor!r}')
    if args.calibrate or not targets:
        targets = {name: builder() for name, builder in sorted(FIXTURES.items())}
        targets['mktcap(field)'] = field('MARKET_CAP')
    print(f'ctx rows={ctx.n_rows:,} sections={len(ctx.ud)} label={ctx.label_name}')
    if args.context:
        print(f'{"factor":16s} {"raw_sia":>8s} {"resid_sia":>9s} {"drop":>7s} '
              f'{"ctx_sia":>8s} {"drop_ind":>8s} {"raw_RIC":>8s} {"resid_RIC":>9s} {"resid_turn":>10s}')
        for name, node in targets.items():
            m = resid_metrics(ctx, node, name=name, context=True)
            print(f'{name:16s} {_fmt(m["raw_sia"]):>8s} {_fmt(m["resid_sia"]):>9s} '
                  f'{_fmt(m["drop"]):>7s} {_fmt(m.get("resid_ctx_sia")):>8s} '
                  f'{_fmt(m.get("drop_ind")):>8s} {_fmt(m["raw_rank_ic"]):>8s} '
                  f'{_fmt(m["resid_rank_ic"]):>9s} {_fmt(m["resid_turnover"]):>10s}')
        return
    print(f'{"factor":16s} {"raw_sia":>8s} {"resid_sia":>9s} {"drop":>7s} '
          f'{"raw_RIC":>8s} {"resid_RIC":>9s} {"resid_turn":>10s}')
    for name, node in targets.items():
        m = resid_metrics(ctx, node, name=name)
        print(f'{name:16s} {_fmt(m["raw_sia"]):>8s} {_fmt(m["resid_sia"]):>9s} '
              f'{_fmt(m["drop"]):>7s} {_fmt(m["raw_rank_ic"]):>8s} '
              f'{_fmt(m["resid_rank_ic"]):>9s} {_fmt(m["resid_turnover"]):>10s}')


if __name__ == '__main__':
    main()
