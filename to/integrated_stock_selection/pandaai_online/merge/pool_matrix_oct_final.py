"""终选组合矩阵：只测探针后仍有效的候选（M8 门控因子已被平台证伪剔除）。

变体：
  base15                        现状 15 人（对照）
  del5                          删 5（10 人）
  del5+2189+SC                  推荐提交集（12 人）
  del5+2189                     单加 M9（11 人）
  del5+2189+b33f                双 M9（12 人）
  del5+2189+b33f+SC             双 M9+保险位（13 人，备选）

用法：python pool_matrix_oct_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pool_legacy as pool_mod
from pool_matrix_oct import R_BAND, b_proxy, build_extra, load_state, print_table

OUT = Path(__file__).resolve().parent / 'out'
DEL5 = ['al2_m9_voldevrank', '6629e15d73b6', '03f0d225461b', 'da42462f269d',
        'amihud120']


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    state = load_state()
    members15 = [e['name'] for e in state['members']]
    del5 = [m for m in members15 if m not in DEL5]
    assert len(del5) == 10, del5

    print('building extra panel...', flush=True)
    extra = build_extra(state)

    sets = {
        'base15': members15,
        'del5': del5,
        'del5+2189+SC': del5 + ['MG-M9-2189e707', 'SC-amt120'],
        'del5+2189': del5 + ['MG-M9-2189e707'],
        'del5+2189+b33f': del5 + ['MG-M9-2189e707', 'MG-M9-b33f0267'],
        'del5+2189+b33f+SC': del5 + ['MG-M9-2189e707', 'MG-M9-b33f0267',
                                     'SC-amt120'],
    }
    for r in R_BAND:
        b = b_proxy(state, r)
        table = pool_mod.comb_table(extra, sets, state, b_override=b)
        base_pts = 40000.0 * (0.20 * table['base15']['NA']
                              + 0.35 * table['base15']['NB'] + 0.45 * 1.0)
        print_table(f'终选组合 R={r}', table, base_pts, nb_on=True)
        payload = {k: dict(row) for k, row in table.items()}
        (OUT / f'pool_matrix_oct_final_R{r}.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=float),
            encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
