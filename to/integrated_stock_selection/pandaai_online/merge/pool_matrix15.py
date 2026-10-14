"""当前 15 人池（平台实际成员）的精确 ΔComb 归因与删除矩阵。

成员（2026-09-20 平台实况）：
  面板成员: skew120 / distance_from_52w_high(=dist250_high) / amihud120 /
            vr120_s20 / comp6 / mktcap
  公式成员: da42462f269d / d18ab547d73c / e69118f1c233 / 6629e15d73b6 /
            6be09e6d76db / 03f0d225461b / al1_m2_voldev5 / al1_m1_retamt5 / al2_m9_voldevrank

只读计算；输出 LOO ΔComb 与删除变体矩阵。
用法：python pool_matrix15.py
"""
from __future__ import annotations

import copy
import sys

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import pool_legacy as pool_mod
from evaluator import load_panel, Evaluator
from parser import parse_formula

FORMULA_MEMBERS = {
    'da42462f269d': ('(TS_MEAN(TURNOVER,5))/(TS_MEAN(TURNOVER,250))', 0, 0.01363),
    'd18ab547d73c': ('EMA(((VOLUME)/(TS_MEAN(VOLUME,250)))-(1),10)', 0, 0.01310),
    'e69118f1c233': ('EMA(((VOLUME)/(TS_MEAN(VOLUME,250)))-(1),20)', 0, 0.01110),
    '6629e15d73b6': ('(TS_MEAN(TURNOVER,5))/(TS_MEAN(TURNOVER,120))', 0, 0.01053),
    '6be09e6d76db': ('STDDEV(((VOLUME)/(TS_MEAN(VOLUME,250)))-(1),10)', 0, 0.01439),
    '03f0d225461b': ('(STDDEV(TURNOVER,5))/(TS_MEAN(TURNOVER,120))', 0, 0.01170),
    'al1_m2_voldev5': ('EMA(((VOLUME)/(TS_MEAN(VOLUME,250)))-(1),5)', 0, 0.01525),
    'al1_m1_retamt5': ('EMA(TS_MEAN(((CLOSE)/(DELAY(CLOSE,1)))-(AMOUNT),5),5)', 1, 0.01354),
    'al2_m9_voldevrank': ('(((RANK(0-(((VOLUME)/(TS_MEAN(VOLUME,250)))-(1))))+(RANK(0-(CORR(((CLOSE)/(DELAY(CLOSE,1)))-(1),VOLUME,40)))))+(RANK(0-(STDDEV(((CLOSE)/(DELAY(CLOSE,1)))-(1),40)))))+(RANK(0-(TS_MEAN(((CLOSE)/(DELAY(CLOSE,1)))-(1),40))))',
                          1, 0.01153),
}
LEGACY = [
    {'name': 'skew120', 'panel': 'qs_factory_panel.parquet', 'dir': 0, 'sia': 0.0041},
    {'name': 'distance_from_52w_high', 'panel': 'factors', 'dir': 0, 'sia': 0.0012},
    {'name': 'amihud120', 'panel': 'qs_factory_panel.parquet', 'dir': 1, 'sia': 0.0019},
    {'name': 'vr120_s20', 'panel': 'qs_r3_panel.parquet', 'dir': 0, 'sia': 0.0116},
    {'name': 'comp6', 'panel': 'qs_r3_panel.parquet', 'dir': 1, 'sia': 0.0063},
    {'name': 'mktcap', 'panel': 'cache_log_total_mv', 'dir': 0, 'sia': 0.0043},
]


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
        print(f'value ok: {name}', flush=True)

    members = [e['name'] for e in state15['members']]
    print(f'\n===== LOO（15 人现池, base Comb 见下） =====', flush=True)
    loo = pool_mod.leave_one_out(extra, state15)
    print(f'full Comb={loo["full_comb"]:.4f}', flush=True)
    for r in sorted(loo['rows'], key=lambda x: x['delta']):
        print(f'  去掉 {r["dropped"]:24s} Comb={r["Comb"]:.4f} ΔComb={-r["delta"]:+.4f} '
              f'NC={r["NC"]:.4f} Rex={100 * (r["Rex"] or 0):+.1f}% Turn={r["Turnover"]:.2f}',
              flush=True)

    def without(*names: str) -> list:
        return [m for m in members if m not in names]

    sets = {
        '现池15人': members,
        '去6629e15d': without('6629e15d73b6'),
        '去03f0d225': without('03f0d225461b'),
        '去两者': without('6629e15d73b6', '03f0d225461b'),
        '去6629+03f0+dist52w': without('6629e15d73b6', '03f0d225461b',
                                       'distance_from_52w_high'),
        '再去da42462f': without('6629e15d73b6', '03f0d225461b', 'da42462f269d',
                                'distance_from_52w_high'),
    }
    print('\n===== 删除变体矩阵 =====', flush=True)
    table = pool_mod.comb_table(extra, sets, state15)
    for name, row in table.items():
        print('%-26s n=%2d rawA=%.4f NA=%.4f NC=%.4f Rex=%+.1f%% Turn=%.1f Comb=%.4f' % (
            name, row['n_members'], row['rawA'], row['NA'], row['NC'],
            (row['Rex'] or 0) * 100, (row['Turnover'] or 0) * 100, row['Comb']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
