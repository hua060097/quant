"""B0 验收（本地 0 算力）：机制优先 + 两级评估 + 提案通道 + 结构指纹。

E1 结构指纹：canonical 级重复同指纹；窗口/常数/字段不同则不同指纹（2026-09-16 修订）
E2 两级等价性：9 夹具 fast(600股/24月) vs full 的 SiA/换手对比，秩相关 ≥0.5
E3 机制搜索集成：mechanism 撒种+精英精化+fast→full+提案直通，台账含 fast/local 事件
E4 机制配额：按历史最好 A 分加权、配额和=n
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import load_panel
from fixtures import FIXTURES
from fitness import DataContext, HOLDOUT_START, build_fast_context, tier1
from grammar import Node, const, field, structural_fingerprint, ts, virtual_node
from ledger import Ledger


def e1_fingerprint() -> tuple[bool, list[str]]:
    ret = virtual_node('RET1')
    turn = field('TURNOVER')
    cases = [
        # 2026-09-16 修订：窗口值与常数进入指纹，只杀 canonical 级重复
        ('same-window-same-const', ts('TS_MEAN', ret, 20), ts('TS_MEAN', ret, 20), True),
        ('diff-window (was clone)', ts('TS_MEAN', ret, 20), ts('TS_MEAN', ret, 40), False),
        ('diff-window-large', ts('TS_MEAN', ret, 20), ts('TS_MEAN', ret, 120), False),
        ('diff-field', ts('TS_MEAN', ret, 20), ts('TS_MEAN', turn, 20), False),
        ('diff-const', Node('MUL', (ret, const(0.5))), Node('MUL', (ret, const(2.0))), False),
        ('wrapped', ts('EMA', ts('TS_MEAN', ret, 20), 5), ts('TS_MEAN', ret, 20), False),
    ]
    log = []
    all_ok = True
    for name, a, b, same in cases:
        got = structural_fingerprint(a) == structural_fingerprint(b)
        ok = got == same
        all_ok &= ok
        log.append(f'  E1 {name:22s} same={got} expected={same} '
                   f'{"PASS" if ok else "FAIL"}')
    return all_ok, log


def e2_two_stage(frame: pd.DataFrame) -> tuple[bool, list[str]]:
    t0 = time.time()
    fast = build_fast_context(frame, n_stocks=600, months=24)
    full = DataContext(frame=frame)
    rows = []
    for name, builder in sorted(FIXTURES.items()):
        node = builder()
        f = tier1(fast.screen(node, name=name))
        g = tier1(full.screen(node, name=name))
        rows.append((name, f, g))
    fast_sia = np.array([r[1]['sia'] for r in rows], dtype=float)
    full_sia = np.array([r[2]['sia'] for r in rows], dtype=float)
    okv = np.isfinite(fast_sia) & np.isfinite(full_sia)
    if okv.sum() >= 3:
        rho = float(pd.Series(fast_sia[okv]).corr(pd.Series(full_sia[okv]), method='spearman'))
    else:
        rho = float('nan')
    passed = bool(okv.sum() >= 5 and np.isfinite(rho) and rho >= 0.5)
    log = [f'  E2 fast rows={fast.n_rows:,} ({time.time() - t0:.0f}s) '
           f'fast-vs-full SiA rank corr={rho:.3f} {"PASS" if passed else "FAIL"}']
    for name, f, g in rows:
        log.append(f'    {name:14s} fast_sia={f["sia"]:.4f} full_sia={g["sia"]:.4f} '
                   f'fast_turn={((f["turnover"] or -1) * 100):5.1f} '
                   f'full_turn={((g["turnover"] or -1) * 100):5.1f}')
    return passed, log


def e3_integration() -> tuple[bool, list[str]]:
    import run_search as rs

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proposals = td / 'proposals.jsonl'
        proposals.write_text(
            json.dumps({'hypothesis': '平均绝对收益（集成测试）',
                        'formula': 'TS_MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1),20)',
                        'direction': 0, 'source': 'test'}, ensure_ascii=False) + '\n',
            encoding='utf-8')
        args = SimpleNamespace(
            ledger=str(td / 'ledger.jsonl'), strategy='mechanism', n=8, seed=42,
            label=None, allow_documented=False, top=4, cycle=3, cut=HOLDOUT_START,
            no_exclude=True, proposals=[str(proposals)], from_ledger=False,
            fast=True, fast_stocks=400, fast_months=18, fast_keep_sia=0.0,
            fast_top_frac=0.0, full_cap=3, refine_elites=1, refine_children=1,
            mutation_rate=0.35, oos_sia=0.0, stall=0, pool_corr=False,
            resid=False, resid_weight=0.0,
            greedy_top=0, greedy_add=1, greedy_min_gain=0.001, report=False)
        old_out = rs.OUT_DIR
        rs.OUT_DIR = td
        try:
            payload = rs.SearchRun(args).run()
        finally:
            rs.OUT_DIR = old_out
        events = Ledger(args.ledger).stats()['events']
        sources = [r['source'] for r in payload['shortlist']]
        checks = {
            'fast_tested>=8': payload['stats']['fast'] >= 8,
            'full>=3': payload['stats']['full'] >= 3,
            'proposal_in_shortlist': 'proposal' in sources,
            'events_fast_local': events.get('fast', 0) >= 8 and events.get('local', 0) >= 3,
            'shortlist_written': len(payload['shortlist']) >= 1,
        }
        ok = all(checks.values())
        return ok, [f'  E3 {k}={v}' for k, v in checks.items()] + \
            [f"  E3 sources={sources} fast={payload['stats']} {'PASS' if ok else 'FAIL'}"]


def e4_plan() -> tuple[bool, list[str]]:
    from grammar import axis_families
    from run_search import _axis_plan
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'ledger.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for _ in range(5):
                f.write(json.dumps({'schema_version': 3, 'event': 'local',
                                    'candidate_id': 'x', 'expression_hash': 'h',
                                    'formula': 'f', 'direction': 1,
                                    'family': 'M1_return', 'hypothesis': '',
                                    'label': 'c2c', 'created_at': '', 'data': {}},
                                   ensure_ascii=False) + '\n')
        quotas, counts = _axis_plan(Ledger(path), 40)
        expected_axes = set(axis_families(allow_documented=False))
        ok = (sum(quotas.values()) == 40 and set(quotas) == expected_axes
              and counts.get('trend_position') == 5
              and all(v >= quotas['trend_position'] for k, v in quotas.items()
                      if k != 'trend_position'))
        return ok, [f"  E4 quotas={quotas} trend_position_tested="
                    f"{counts.get('trend_position')} sum={sum(quotas.values())} "
                    f'{"PASS" if ok else "FAIL"}']


def run_all() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    t0 = time.time()
    print('=' * 78)
    print('B0 验收：机制优先 + 两级评估 + 提案通道 + 结构指纹')
    print('=' * 78)
    frame = load_panel()
    print(f'panel rows={len(frame):,} {time.time() - t0:.0f}s\n', flush=True)
    checks = [('E1', e1_fingerprint), ('E2', lambda: e2_two_stage(frame)),
              ('E3', e3_integration), ('E4', e4_plan)]
    results = []
    for label, fn in checks:
        print(f'[{label}] ...', flush=True)
        ok, log = fn()
        results.append((label, ok))
        print('\n'.join(log), flush=True)
        print(f'[{label}] {"PASS" if ok else "FAIL"}\n', flush=True)
    summary = ' | '.join(f'{k}:{"PASS" if v else "FAIL"}' for k, v in results)
    print('=' * 78)
    print(f'SUMMARY {summary} ({time.time() - t0:.0f}s)')
    print('=' * 78)
    return 0 if all(v for _, v in results) else 1


if __name__ == '__main__':
    sys.exit(run_all())
