"""2026-10 窗口池修改变体矩阵（比赛规则代理，0 平台调用）。

与 pool_matrix15.py 的差别：
  1. 成员/候选全部来自 pool_state.json（2026-10-03 版：15 members + 6 reserve，
     inline 成员带 formula 字段），不再硬编码。
  2. comb_proxy 通过 b_override 追加 0.35×NB 项（B 代理 = holdout_sia（已知）否则
     sia×R，R∈{0.40,0.5625,0.62,0.85}；0.5625=平台隐含 rawB/rawA）。
  3. 双口径并列：Comb_L = 0.2NA+0.35NB+0.45NC_local；Comb_P = 0.2NA+0.35NB+0.45×1.0
     （平台 NC 已饱和的判断，见计划 Context）+ pts_P = 40000×Comb_P。
  4. 变体：base15（回归锚）/del4/del5/单加 6 候选/组合/贪心/终选集 LOO。

平台积分敏感度：+0.001 rawA=+100 分/月；+0.001 rawB=+233 分/月（NB 权重 0.35/0.06）。

用法：python pool_matrix_oct.py            # 全矩阵（~40 次池回测，约 1 小时）
      python pool_matrix_oct.py --fast     # 跳过贪心与 LOO（约 15 分钟）
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import pool_legacy as pool_mod
from evaluator import Evaluator, load_panel
from parser import parse_formula

OUT = Path(__file__).resolve().parent / 'out'
DEL4 = ['al2_m9_voldevrank', '6629e15d73b6', '03f0d225461b', 'da42462f269d']
DEL5 = DEL4 + ['amihud120']
R_BAND = (0.40, 0.5625, 0.62, 0.85)


def load_state() -> dict:
    return pool_mod.load_pool_state()


def build_extra(state: dict) -> pd.DataFrame:
    """非 inline 成员走 build_extra_panel，inline 成员（含 reserve）经 evaluator 求值合并。"""
    entries = pool_mod.all_entries(state)
    legacy = [e for e in entries if e.get('panel') != 'inline']
    extra = pool_mod.build_extra_panel(legacy)
    frame = load_panel()
    ev = Evaluator(frame)
    for entry in entries:
        if entry.get('panel') != 'inline':
            continue
        name, formula = entry['name'], entry['formula']
        v = np.asarray(ev.values(parse_formula(formula)), dtype=float)
        d = frame[['stock_code', 'date']].copy()
        d[name] = v
        extra = extra.merge(d, on=['stock_code', 'date'], how='outer')
        print(f'value ok: {name}', flush=True)
    return extra


def b_proxy(state: dict, ratio: float) -> dict:
    out = {}
    for e in pool_mod.all_entries(state):
        if 'holdout_sia' in e and e['holdout_sia'] is not None:
            out[e['name']] = float(e['holdout_sia'])
        else:
            out[e['name']] = float(e['sia']) * ratio
    return out


def state_with(state: dict, members: list[str]) -> dict:
    entries = {e['name']: e for e in pool_mod.all_entries(state)}
    return {**state, 'members': [entries[n] for n in members]}


def print_table(title: str, table: dict, base_pts: float | None, nb_on: bool) -> None:
    print(f'\n===== {title} =====', flush=True)
    head = ('%-30s %2s %7s %7s %7s %7s %7s %7s %8s %8s %8s %9s'
            if nb_on else
            '%-30s %2s %7s %7s %7s %8s %8s %8s %9s')
    print(head % (('variant', 'n', 'rawA', 'NA', 'rawB', 'NB', 'NC_L', 'Rex%',
                   'Turn%', 'Comb_L', 'Comb_P', 'dpts_P') if nb_on else
                 ('variant', 'n', 'rawA', 'NA', 'NC_L', 'Rex%', 'Turn%',
                  'Comb_L', 'dpts_L')), flush=True)
    for name, row in table.items():
        rex = f'{100 * (row["Rex"] or 0):+.1f}'
        turn = f'{100 * (row["Turnover"] or 0):.1f}'
        if nb_on:
            pts_p = 40000.0 * (0.20 * row['NA'] + 0.35 * row['NB'] + 0.45 * 1.0)
            comb_p = 0.20 * row['NA'] + 0.35 * row['NB'] + 0.45 * 1.0
            dpts = '' if base_pts is None else f'{pts_p - base_pts:+.0f}'
            print(('%-30s %2d %7.4f %7.4f %7.4f %7.4f %7.4f %7s %8s %8.4f %8.4f %9s'
                   % (name, row['n_members'], row['rawA'], row['NA'],
                      row.get('rawB', float('nan')), row.get('NB', float('nan')),
                      row['NC'], rex, turn, row['Comb'], comb_p, dpts)), flush=True)
        else:
            pts_l = 40000.0 * row['Comb']
            dpts = '' if base_pts is None else f'{pts_l - base_pts:+.0f}'
            print(('%-30s %2d %7.4f %7.4f %7.4f %7s %8s %8.4f %9s'
                   % (name, row['n_members'], row['rawA'], row['NA'], row['NC'],
                      rex, turn, row['Comb'], dpts)), flush=True)


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='跳过贪心与 LOO')
    args = ap.parse_args()

    state = load_state()
    entries = {e['name']: e for e in pool_mod.all_entries(state)}
    all_names = list(entries)
    members15 = [e['name'] for e in state['members']]
    reserve = [e['name'] for e in state['reserve']]

    print('building extra panel (inline formulas evaluated)...', flush=True)
    extra = build_extra(state)
    print(f'extra panel: {extra.shape}', flush=True)

    def without(names: list[str], *drop: str) -> list[str]:
        return [m for m in names if m not in drop]

    # ---- 锚点校验：NB 关（b_override=None）时 base15/del4 必须复现 pool15b.log ----
    sets_off = {
        'base15': members15,
        'del4': without(members15, *DEL4),
        'del5': without(members15, *DEL5),
    }
    table = pool_mod.comb_table(extra, sets_off, state)
    base_pts_off = 40000.0 * table['base15']['Comb']
    print_table('锚点（NB 关，须复现 pool15b.log：base15 NC=0.3201 Comb=0.1681；'
                'del4 NC=0.4280 Comb=0.2146）', table, base_pts_off, nb_on=False)
    anchor = {'base15': table['base15'], 'del4': table['del4']}
    (OUT / 'pool_matrix_oct_anchor.json').write_text(
        json.dumps(anchor, ensure_ascii=False, indent=1), encoding='utf-8')

    # ---- 主矩阵：NB 开，R 敏感性带 ----
    variants: dict[str, list[str]] = {
        'base15': members15,
        'del4': without(members15, *DEL4),
        'del5': without(members15, *DEL5),
    }
    del5 = variants['del5']
    for cand in reserve:
        variants[f'del5+{cand}'] = del5 + [cand]
    variants['del5+bdf8c3ae+SC-amt120'] = del5 + ['MG-M8-bdf8c3ae', 'SC-amt120']
    variants['del4+bdf8c3ae'] = without(members15, *DEL4) + ['MG-M8-bdf8c3ae']
    variants['del5+3adds'] = del5 + ['MG-M8-bdf8c3ae', 'MG-M8-6f143582', 'SC-amt120']

    for r in R_BAND:
        b = b_proxy(state, r)
        sub = {k: v for k, v in variants.items()}
        table = pool_mod.comb_table(extra, sub, state, b_override=b)
        base_pts = 40000.0 * (0.20 * table['base15']['NA']
                              + 0.35 * table['base15']['NB'] + 0.45 * 1.0)
        print_table(f'NB 开 R={r}（B 代理 = holdout 否则 sia×{r}）', table,
                    base_pts, nb_on=True)
        payload = {k: {kk: vv for kk, vv in row.items()} for k, row in table.items()}
        (OUT / f'pool_matrix_oct_R{r}.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=float),
            encoding='utf-8')

    if args.fast:
        print('\n[--fast] 跳过贪心与 LOO', flush=True)
        return 0

    # ---- 贪心（del5 起点，全部 6 候选，max_add=3）----
    for r in R_BAND:
        b = b_proxy(state, r)
        state_del5 = state_with(state, del5)
        cands = [entries[n] for n in reserve]
        g = pool_mod.greedy_forward(extra, state_del5, cands, max_add=3,
                                    min_gain=0.001, b_override=b)
        print(f'\n===== 贪心 R={r}：base Comb_L={g["base_comb"]:.4f} -> '
              f'{g["final_comb"]:.4f} =====', flush=True)
        for s in g['steps']:
            print('  + %-22s dComb_L=%+.4f NC=%s Rex=%s Turn=%s'
                  % (s['candidate'], s['gain'],
                     f"{s['NC']:.4f}" if s['NC'] is not None else '-',
                     f"{100 * s['Rex']:+.1f}%" if s['Rex'] is not None else '-',
                     f"{100 * s['Turnover']:.1f}%" if s['Turnover'] is not None else '-'),
                  flush=True)
        (OUT / f'pool_matrix_oct_greedy_R{r}.json').write_text(
            json.dumps(g, ensure_ascii=False, indent=1, default=float),
            encoding='utf-8')

    # ---- 终选集 LOO（del5 + bdf8c3ae + SC-amt120）----
    final_set = del5 + ['MG-M8-bdf8c3ae', 'SC-amt120']
    state_final = state_with(state, final_set)
    for r in (0.5625, 0.62):
        b = b_proxy(state, r)
        loo = pool_mod.leave_one_out(extra, state_final, b_override=b)
        print(f'\n===== 终选集 LOO R={r}（full Comb_L={loo["full_comb"]:.4f}） =====',
              flush=True)
        for row in sorted(loo['rows'], key=lambda x: x['delta']):
            print('  去 %-24s dComb_L=%+.4f NC=%.4f Rex=%s Turn=%s'
                  % (row['dropped'], -row['delta'], row['NC'],
                     f"{100 * (row['Rex'] or 0):+.1f}%", f"{100 * (row['Turnover'] or 0):.1f}%"),
                  flush=True)
        (OUT / f'pool_matrix_oct_loo_R{r}.json').write_text(
            json.dumps(loo, ensure_ascii=False, indent=1, default=float),
            encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
