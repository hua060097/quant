"""15 人池补充变体：去掉 al2 / 6629 / 03f0 / da42462f 的组合矩阵（复用 pool_matrix15 面板）。"""
from __future__ import annotations

import copy
import sys

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import pool_legacy as pool_mod
from evaluator import load_panel, Evaluator
from parser import parse_formula
from pool_matrix15 import FORMULA_MEMBERS, LEGACY


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    state15 = {'cycle': 3, 'label': 'c2c_3d',
               'members': list(LEGACY) + [{'name': k, 'panel': 'inline',
                                           'dir': int(v[1]), 'sia': float(v[2])}
                                          for k, v in FORMULA_MEMBERS.items()],
               'reserve': []}
    extra = pool_mod.build_extra_panel(LEGACY)
    frame = load_panel()
    ev = Evaluator(frame)
    for name, (f, _d, _s) in FORMULA_MEMBERS.items():
        v = np.asarray(ev.values(parse_formula(f)), dtype=float)
        d = frame[['stock_code', 'date']].copy()
        d[name] = v
        extra = extra.merge(d, on=['stock_code', 'date'], how='outer')
    members = [e['name'] for e in state15['members']]

    def without(*names: str) -> list:
        return [m for m in members if m not in names]

    sets = {
        '现池15人': members,
        '去al2': without('al2_m9_voldevrank'),
        '去6629+03f0 (13人)': without('6629e15d73b6', '03f0d225461b'),
        '去al2+6629+03f0 (12人)': without('al2_m9_voldevrank', '6629e15d73b6',
                                          '03f0d225461b'),
        '去al2+6629+03f0+da42 (11人)': without('al2_m9_voldevrank', '6629e15d73b6',
                                               '03f0d225461b', 'da42462f269d'),
        '去al2+6629+03f0+6be09 (11人)': without('al2_m9_voldevrank', '6629e15d73b6',
                                                '03f0d225461b', '6be09e6d76db'),
        '去al2+6629+03f0+voldev5 (11人)': without('al2_m9_voldevrank', '6629e15d73b6',
                                                  '03f0d225461b', 'al1_m2_voldev5'),
        '去6629+03f0+da42 (12人)': without('6629e15d73b6', '03f0d225461b',
                                           'da42462f269d'),
    }
    table = pool_mod.comb_table(extra, sets, state15)
    for name, row in table.items():
        print('%-30s n=%2d rawA=%.4f NA=%.4f NC=%.4f Rex=%+.1f%% Turn=%.1f Comb=%.4f' % (
            name, row['n_members'], row['rawA'], row['NA'], row['NC'],
            (row['Rex'] or 0) * 100, (row['Turnover'] or 0) * 100, row['Comb']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
