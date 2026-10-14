"""B3 验收（2026-09-16 目标重定义后）：官方 A 口径 + 换手软惩罚 + 无门槛 + vwap 名单。

T1 官方口径：turnover_month 定义范围、c_proxy 公式、turnover_penalty、rank_score 对照
T2 无门槛搜索：vwap 不再必过、双名单与 jsonl 产出、字段齐全（含 vwap_top 排序）
T3 方向导出：executor 写出 `name ~ formula ~ direction` 且 validate_file 通过
T4 指纹覆盖：新指纹不塌缩（唯一点占比高），窗口/常数可区分
T5 rescore：合成台账行重排逻辑 + 端到端 rescre（--full-top 0）落盘与排序
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import _paths  # noqa: F401
import run_search as rs
from executor import validate_file, write_candidates
from fitness import HOLDOUT_START, c_proxy_of, rank_score, stability_of, tier1, turnover_penalty
from grammar import const, field, structural_fingerprint, ts, virtual_node
from ledger import Ledger
from parser import parse_formula


def t1_official_metrics() -> tuple[bool, list[str]]:
    frame = rs.load_panel()
    ctx = rs.DataContext(cycle=3, frame=frame, label='c2c')
    checks, log = {}, []

    # 1) turnover_month 是"当月各调仓日换手之和"的月均值：cycle3 每月 5~8 个调仓日
    m = ctx.screen(ts('TS_MEAN', virtual_node('RET1'), 20), name='t1:ret20')
    t, tm = m.get('turnover'), m.get('turnover_month')
    ok_tm = (t is not None and tm is not None and np.isfinite(t) and np.isfinite(tm)
             and 3.0 * t <= tm <= 9.0 * t)
    checks['turnover_month_range'] = bool(ok_tm)
    log.append(f'  T1 turnover={t:.4f} turnover_month={tm:.4f} ratio={(tm / t if t else 0):.2f}')

    # 2) c_proxy 公式逐值一致
    tier = {'net': 0.10, 'turnover_month': 0.9, 'sharpe': 1.5, 'mdd': 0.2}
    expect = max(0.10, 0.0) / max(0.9, 0.3) * 1.5 * (1 - 1.2 * 0.2)
    checks['c_proxy_formula'] = abs(c_proxy_of(tier) - expect) < 1e-12

    # 3) 换手软惩罚：≤ref 不罚；>ref 线性下降；缺失时用 7×单次换手近似
    p_lo = turnover_penalty({'turnover_month': 0.2}, 0.3)
    p_hi = turnover_penalty({'turnover_month': 0.9}, 0.3)
    p_fb = turnover_penalty({'turnover': 0.10}, 0.3)
    checks['penalty'] = (abs(p_lo - 1.0) < 1e-12 and p_hi < p_lo
                         and abs(p_fb - 0.3 / 0.7) < 1e-12)

    # 4) rank_score：同一 tier 下加惩罚 ≤ 无惩罚；T_month ≤ ref 时相等
    node = parse_formula('TS_MEAN(TURNOVER,20)')
    low = {'sia': 0.004, 'turnover_month': 0.2}
    high = {'sia': 0.004, 'turnover_month': 1.5}
    s_low_pen = rank_score(low, node, turnover_ref=0.3)
    s_low_no = rank_score(low, node, turnover_ref=None)
    s_high_pen = rank_score(high, node, turnover_ref=0.3)
    checks['rank_score_penalty'] = (abs(s_low_pen - s_low_no) < 1e-12
                                    and s_high_pen < s_low_pen)

    # 5) 稳健性 tiebreak：分年 IC 按方向对齐后取最小
    st1 = stability_of({'yearly_ic': {'2024': 0.05, '2025': -0.01}, 'ic_mean': 0.03})
    st2 = stability_of({'yearly_ic': {'2024': 0.05, '2025': -0.01}, 'ic_mean': -0.03})
    checks['stability'] = (abs(st1 - (-0.01)) < 1e-12 and abs(st2 - (-0.05)) < 1e-12)

    ok = all(checks.values())
    return ok, log + [f'  T1 {k}={v}' for k, v in checks.items()] + \
        [f'  T1 official metrics {"PASS" if ok else "FAIL"}']


def t2_no_gate_search() -> tuple[bool, list[str]]:
    panel = rs.load_panel()
    del panel
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        args = SimpleNamespace(
            ledger=str(td / 'ledger.jsonl'), strategy='mechanism', n=6, seed=17,
            label=['c2c', 'vwap'], allow_documented=True, top=5, cycle=3,
            cut=HOLDOUT_START, no_exclude=True, proposals=None, from_ledger=False,
            fast=True, fast_stocks=300, fast_months=12, fast_keep_sia=0.0,
            fast_top_frac=0.0, full_cap=5, refine_elites=1, refine_children=1,
            mutation_rate=0.35, oos_sia=0.0, stall=0, pool_corr=False,
            resid=False, resid_weight=0.0,
            greedy_top=0, greedy_add=1, greedy_min_gain=0.001, report=False)
        old_out = rs.OUT_DIR
        rs.OUT_DIR = td
        try:
            payload = rs.SearchRun(args).run()
        finally:
            rs.OUT_DIR = old_out
        rows = payload.get('shortlist', [])
        vtop = payload.get('vwap_top', [])
        keys = []
        for r in vtop:
            t = (r.get('labels') or {}).get('vwap') or {}
            rk, ir, w = t.get('rank_ic'), t.get('rank_ic_ir'), t.get('rank_ic_win')
            if all(v is not None and v == v for v in (rk, ir, w)):
                keys.append(abs(rk) * abs(ir) * abs(w))
            else:
                keys.append(t.get('sia') or 0.0)
        files = list(td.glob('shortlist_*.json')) + list(td.glob('shortlist_*.jsonl'))
        checks = {
            'gate_off': payload.get('dual_gate') is None,
            'no_dual_rejects': payload.get('dual_rejected') == [],
            'rows_have_new_fields': all(
                all(k in r for k in ('score_nopen', 'stability', 'turnover_month', 'c_proxy'))
                for r in rows),
            'vwap_top_sorted': all(keys[i] >= keys[i + 1] - 1e-12 for i in range(len(keys) - 1)),
            'vwap_rank_fields': all(
                ((r.get('labels') or {}).get('vwap') or {}).get('rank_ic_ir') is not None
                for r in vtop[:3]) if vtop else True,
            'vwap_top_bounded': len(vtop) <= 20,
            'json_and_jsonl': len(files) >= 2,
        }
        ok = all(checks.values())
        return ok, [f'  T2 {k}={v}' for k, v in checks.items()] + \
            [f'  T2 no-gate search {"PASS" if ok else "FAIL"}']


def t3_direction_export() -> tuple[bool, list[str]]:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'candidates.txt'
        rows = [
            {'name': 'B3-dir0', 'formula': 'TS_MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1),20)',
             'direction': 0},
            {'name': 'B3-dir1', 'formula': 'TS_MEAN(ABS(CLOSE/DELAY(CLOSE,1)-1),20)',
             'direction': 1},
        ]
        write_candidates(rows, out, tag='b3', notes=('selftest',))
        ok_file, msg = validate_file(out)
        lines = [ln for ln in out.read_text(encoding='utf-8').splitlines()
                 if ln and not ln.lstrip().startswith('#')]
        dirs = [ln.split('~')[-1].strip() for ln in lines]
        checks = {'validate': ok_file, 'two_rows': len(lines) == 2,
                  'directions': dirs == ['0', '1']}
        ok = all(checks.values())
        return ok, [f'  T3 {k}={v}' for k, v in checks.items()] + \
            [f'  T3 direction export {msg if not ok_file else "PASS" if ok else "FAIL"}']


def t4_fingerprint_coverage() -> tuple[bool, list[str]]:
    import hashlib
    import random

    import grammar as g

    def old_fp(node) -> str:
        fields, ops, buckets = [], [], []
        for _, n in g.walk(node):
            if n.op == 'FIELD':
                fields.append(str(n.name))
            elif n.op != 'CONST':
                ops.append(n.op)
            if n.window:
                buckets.append('S' if n.window <= 20 else 'M' if n.window <= 60 else 'L')
        payload = '|'.join([','.join(sorted(fields)), ','.join(ops),
                            ','.join(sorted(buckets))])
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    rng = random.Random(23)
    new_fps, old_fps = set(), set()
    n = 120
    for _ in range(n):
        _, node = g.random_tree(rng, allow_documented=True)
        new_fps.add(structural_fingerprint(node))
        old_fps.add(old_fp(node))
    a = ts('TS_MEAN', field('CLOSE'), 20)
    b = ts('TS_MEAN', field('CLOSE'), 40)
    c = ts('TS_MEAN', field('CLOSE'), 20)
    checks = {'new_much_richer': len(new_fps) >= len(old_fps) + 8,
              'new_ratio>=0.80': len(new_fps) / n >= 0.80,
              'window_distinct': structural_fingerprint(a) != structural_fingerprint(b),
              'canonical_same': structural_fingerprint(a) == structural_fingerprint(c)}
    ok = all(checks.values())
    return ok, [f'  T4 unique new={len(new_fps)} old={len(old_fps)} n={n}'] + \
        [f'  T4 {k}={v}' for k, v in checks.items()] + \
        [f'  T4 fingerprint coverage {"PASS" if ok else "FAIL"}']


def t5_rescore() -> tuple[bool, list[str]]:
    rows = {
        'h1': {'formula': 'TS_MEAN(TURNOVER,20)', 'family': 'M2_liquidity', 'direction': 0,
               'labels': {'c2c': {'sia': 0.004, 'turnover': 0.03, 'turnover_month': 0.2,
                                  'net': 0.05, 'sharpe': 1.2, 'mdd': 0.1,
                                  'yearly_ic': {'2024': 0.03, '2025': 0.01}, 'ic_mean': 0.02},
                          'vwap': {'sia': 0.0010}}},
        'h2': {'formula': 'TS_MEAN(TURNOVER,20)', 'family': 'M2_liquidity', 'direction': 0,
               'labels': {'c2c': {'sia': 0.008, 'turnover': 0.20, 'turnover_month': 1.4,
                                  'net': -0.02, 'sharpe': 0.5, 'mdd': 0.3,
                                  'yearly_ic': {'2024': 0.06, '2025': -0.03}, 'ic_mean': 0.03},
                          'vwap': {'sia': 0.0030}}},
    }
    scored = rs._rescore_rows(rows, 0.3)
    by = {r['hash']: r for r in scored}
    checks = {}
    checks['direction_zero_preserved'] = (by['h1']['direction'] == 0
                                          and by['h2']['direction'] == 0)
    checks['metrics_present'] = all(
        all(k in by[h] for k in ('rank_ic', 'ic', 'ic_ir', 'win', 'sharpe', 'mdd'))
        for h in ('h1', 'h2'))
    checks['penalty_applied'] = (abs(by['h1']['penalty'] - 1.0) < 1e-12
                                 and by['h2']['penalty'] < 1.0)
    checks['score_uses_penalty'] = by['h1']['score'] == by['h1']['score_nopen']
    checks['vwap_fields'] = (by['h2']['vwap_sia'] == 0.0030
                             and by['h2']['vwap_ratio'] is not None)
    # 端到端：真实台账 + --full-top 0，输出到临时目录
    with tempfile.TemporaryDirectory() as td:
        old_out = rs.OUT_DIR
        rs.OUT_DIR = Path(td)
        try:
            ns = SimpleNamespace(ledger=str(Ledger().path), top=5, vwap_top=5,
                                 turnover_ref=0.3, no_turnover_penalty=False,
                                 full_top=0, label=['c2c', 'vwap'], cycle=3)
            rs.cmd_rescore(ns)
        finally:
            rs.OUT_DIR = old_out
        out = list(Path(td).glob('rescore_*.json'))
        jsonl = list(Path(td).glob('rescore_*.jsonl'))
        if out:
            payload = json.loads(out[0].read_text(encoding='utf-8'))
            top = payload.get('c2c_top', [])
            checks['rescore_files'] = bool(jsonl) and bool(payload.get('vwap_top'))
            checks['rescore_sorted'] = all(
                top[i]['score'] >= top[i + 1]['score'] for i in range(len(top) - 1))
            checks['rescore_metrics'] = all(
                all(k in r for k in ('rank_ic', 'ic', 'ic_ir', 'win', 'sharpe', 'mdd',
                                     'vwap_rank_sia', 'rank_sia'))
                for r in top)
            checks['rescore_direction_sane'] = all(r.get('direction') in (0, 1) for r in top)
        else:
            checks['rescore_files'] = False
            checks['rescore_sorted'] = False
            checks['rescore_metrics'] = False
            checks['rescore_direction_sane'] = False
    ok = all(checks.values())
    return ok, [f'  T5 {k}={v}' for k, v in checks.items()] + \
        [f'  T5 rescore {"PASS" if ok else "FAIL"}']


def run_all() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    t0 = time.time()
    print('=' * 78)
    print('B3 验收：官方A口径 + 换手软惩罚 + 无门槛(vwap仅诊断) + 双名单 + rescore')
    print('=' * 78)
    checks = [('T1', t1_official_metrics), ('T2', t2_no_gate_search),
              ('T3', t3_direction_export), ('T4', t4_fingerprint_coverage),
              ('T5', t5_rescore)]
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
