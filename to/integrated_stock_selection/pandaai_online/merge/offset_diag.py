"""采样稳健性诊断（2026-09-24）：cycle=3 截面锚点 offset 0/1/2 三组各评一遍。

背景：主口径每 3 个交易日取一个截面（与平台 A 分口径、实盘调仓节奏对齐，且避免
重叠截面 ICIR 虚高 ~√3）。本诊断回答"因子是否在挑调仓时点"：
  - 三组 RankIC 同号且接近 → 结论稳健，起点多样性无影响；
  - 分歧大/换号 → 因子对调仓起点敏感（危险信号，可能是周期内时间结构驱动）。

方向纪律：用 offset=0（主口径）的方向冻结其余两组，与 holdout 冻结同一规则。
offset=0 的读数直接复用主口径结果（同一 DataContext），不重算。

用法（0 算力）：
  python offset_diag.py --factor slope60                # 夹具，快评面板（800股）
  python offset_diag.py --factor slope60 --full         # 全量面板
  python offset_diag.py --formula "TS_MEAN(SUB(RET1,IND_RET),60)" --label vwap
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import Evaluator, load_panel
from fitness import END, START, DataContext, build_fast_context, tier1
from fixtures import FIXTURES
from parser import ParseError, parse_formula

FAST_MONTHS = 24


def build_offset_ctxs(frame, evaluator: Evaluator, cycle: int = 3,
                      labels=('c2c',), offsets=(1, 2),
                      start: str = START, end: str = END) -> dict:
    """错位截面 DataContext 缓存（标签序列构建是主要成本，整批只建一次）。"""
    out = {}
    for off in offsets:
        for lab in labels:
            out[(off, lab)] = DataContext(cycle=cycle, start=start, end=end,
                                          frame=frame, label=lab,
                                          evaluator=evaluator, offset=off)
    return out


def offset_scan(node, ctxs: dict, main_rank_ics: dict, directions,
                labels=('c2c',), offsets=(0, 1, 2), name: str = 'factor') -> dict:
    """三组错位截面各评 RankIC（方向冻结）；offset=0 复用主口径读数。

    directions: 每标签自己的 offset-0 方向（dict），或单一方向（int，全部标签共用）。
    次标签（如 vwap）用自己的自动方向——跨标签方向语义不同，不能共用主标签方向。
    返回 {lab: {'rank_ics': {off: ric}, 'mean','min','max','spread','sign_consistent'}}
    """
    out: dict = {}
    for lab in labels:
        d_dir = int(directions[lab]) if isinstance(directions, dict) else int(directions)
        v0 = main_rank_ics.get(lab)
        # 方向对齐：主口径 rank_ic 是原始符号（direction=0 → 负），offset 1/2 已经
        # fixed_direction 翻正——offset 0 也按各自方向翻正后才可比。
        # rank_ics 键用字符串（JSON round-trip 后 int 键变 str，report 侧读取需一致）
        if v0 is not None and np.isfinite(v0):
            v0 = float(v0) * (1.0 if d_dir == 1 else -1.0)
        out[lab] = {'rank_ics': {'0': v0}, 'dir': d_dir}
    for off in offsets:
        if off == 0:
            continue
        for lab in labels:
            ctx = ctxs.get((off, lab))
            if ctx is None:
                continue
            d = directions[lab] if isinstance(directions, dict) else directions
            m = ctx.screen(node, name=f'{name}:off{off}', fixed_direction=int(d))
            # ctx.screen 返回原始 screen 字典（键 rank_ic_mean）；rank_ic 是 tier1 映射键
            out[lab]['rank_ics'][str(off)] = tier1(m).get('rank_ic')
    for lab in labels:
        ric = [out[lab]['rank_ics'].get(str(o)) for o in offsets]
        vals = [float(x) for x in ric if x is not None and np.isfinite(x)]
        out[lab]['mean'] = float(np.mean(vals)) if vals else None
        out[lab]['min'] = float(np.min(vals)) if vals else None
        out[lab]['max'] = float(np.max(vals)) if vals else None
        out[lab]['spread'] = (float(max(vals) - min(vals))
                              if len(vals) == len(offsets) else None)
        out[lab]['sign_consistent'] = bool(vals) and all(v > 0 for v in vals)
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
    ap = argparse.ArgumentParser(description='采样稳健性诊断（offset 0/1/2）')
    ap.add_argument('--factor', default='', help='fixtures.py 夹具名')
    ap.add_argument('--formula', default='', help='公式文本（与 --factor 二选一）')
    ap.add_argument('--label', action='append', default=None,
                    help='标签，可重复（默认 c2c；第一个为主标签定方向）')
    ap.add_argument('--full', action='store_true', help='全量面板（默认快评 800 股）')
    ap.add_argument('--stocks', type=int, default=800)
    args = ap.parse_args()
    if not args.factor and not args.formula:
        raise SystemExit('需要 --factor 或 --formula')
    if args.formula:
        try:
            node = parse_formula(args.formula)
        except ParseError as exc:
            raise SystemExit(f'parse failed: {exc}')
    else:
        if args.factor not in FIXTURES:
            raise SystemExit(f'unknown fixture {args.factor!r}')
        node = FIXTURES[args.factor]()

    labels = list(args.label or ['c2c'])
    frame = load_panel()
    if args.full:
        ev = Evaluator(frame)
        start = START
        ctx0 = {lab: DataContext(frame=frame, label=lab, evaluator=ev)
                for lab in labels}
    else:
        # 快评面板：同一股票抽样/窗口（同 seed），多标签共享同一 Evaluator（因子值只算一次）
        main_fast = build_fast_context(frame, n_stocks=args.stocks, label=labels[0])
        ev = main_fast.evaluator
        start = (pd.Timestamp(END) - pd.DateOffset(months=FAST_MONTHS)).strftime('%Y%m%d')
        ctx0 = {labels[0]: main_fast}
        for lab in labels[1:]:
            ctx0[lab] = build_fast_context(frame, n_stocks=args.stocks, label=lab,
                                           evaluator=ev)

    octxs = build_offset_ctxs(ctx0[labels[0]].evaluator.frame, ev,
                              cycle=ctx0[labels[0]].cycle, labels=labels,
                              start=start, end=END)
    screens = {lab: ctx0[lab].screen(node, name=f'off0:{lab}') for lab in labels}
    directions = {lab: int(screens[lab].get('direction', 1)) for lab in labels}
    main_ric = {lab: tier1(screens[lab]).get('rank_ic') for lab in labels}
    res = offset_scan(node, octxs, main_ric, directions, labels=labels, name='offdiag')
    print(f'labels={labels} directions={directions} '
          f'ctx rows={ctx0[labels[0]].n_rows:,} sections={len(ctx0[labels[0]].ud)}')
    for lab in labels:
        d = res[lab]
        ric = d['rank_ics']
        print(f'  [{lab}] RankIC(0/1/2) = {_fmt(ric.get(0), "%+.4f")} / '
              f'{_fmt(ric.get(1), "%+.4f")} / {_fmt(ric.get(2), "%+.4f")}  '
              f'spread={_fmt(d["spread"])} min={_fmt(d["min"], "%+.4f")} '
              f'sign_ok={d["sign_consistent"]}')


if __name__ == '__main__':
    main()
