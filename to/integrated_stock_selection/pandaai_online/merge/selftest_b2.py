"""B2 验收（本地 0 算力）：标签系统 + 多标签 + 残差 IC + 机制/log + grammar v2。

L1 vwap 标签与 inline 参考实现逐值一致（公式原义，含 buy=open(T+1) 与 1e-12 保护）
L2 多标签：一次评估两标签；台账 hashes_for 的"公式×标签"分离策略正确
R1 残差不变式：MARKET_CAP 过轴后残差≈0；amihud120 掉幅>0.2 且残差 SiA 下降
T1 机制优化：M2 同时产出 raw/log 换手变体与 log 市值形式；轴配额计划正确
T2 grammar v2：M9/M10/M11 生成与校验、文档族开关、AMOUNT 安全尺度、深度上限
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import _paths  # noqa: F401
from evaluator import load_panel
from fitness import DataContext, HOLDOUT_START, build_fast_context
from fixtures import FIXTURES
from grammar import field, random_tree, walk
from ledger import Ledger
from resid import resid_metrics


def l1_vwap_label(frame) -> tuple[bool, list[str]]:
    ctx = DataContext(frame=frame, label='vwap')
    codes = frame['stock_code']
    tp = (frame['OPEN'] + frame['HIGH'] + frame['LOW'] + frame['CLOSE']) / 4.0
    buy = frame['OPEN'].groupby(codes, sort=False).shift(-1)
    sell = (0.7 * tp.groupby(codes, sort=False).shift(-2)
            + 0.3 * tp.groupby(codes, sort=False).shift(-3))
    ref = ((sell - buy) / (buy + 1e-12)).to_numpy(dtype=float)[ctx._pos]
    ok = np.isfinite(ctx.labels) & np.isfinite(ref)
    diff = np.abs(ctx.labels[ok] - ref[ok])
    max_abs = float(diff.max()) if diff.size else float('nan')
    n_finite = int(np.isfinite(ref).sum())
    passed = (max_abs <= 1e-12 and n_finite == len(ctx.labels)
              and ctx.label_name == 'vwap_3d' and ctx.n_rows > 1_000_000)
    return passed, [f'  L1 vwap label rows={ctx.n_rows:,} max|Δ|={max_abs:.3e} '
                    f'finite_ref={n_finite:,} name={ctx.label_name} '
                    f'{"PASS" if passed else "FAIL"}']


def l2_multi_label() -> tuple[bool, list[str]]:
    import run_search as rs

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        args = SimpleNamespace(
            ledger=str(td / 'ledger.jsonl'), strategy='random',
            label=['c2c', 'vwap'], allow_documented=False, n=6, seed=13, top=4,
            cycle=3, cut=HOLDOUT_START,
            no_exclude=True, proposals=None, from_ledger=False,
            dual_gate=True, dual_min_sia=0.0, dual_min_ratio=0.0,
            fast=True, fast_stocks=400, fast_months=18, fast_keep_sia=0.0,
            fast_top_frac=0.0, full_cap=2, refine_elites=1, refine_children=0,
            mutation_rate=0.35, oos_sia=0.0, stall=0, pool_corr=False,
            resid=True, resid_weight=0.0, greedy_top=0, greedy_add=1,
            greedy_min_gain=0.001, report=False)
        old_out = rs.OUT_DIR
        rs.OUT_DIR = td
        try:
            payload = rs.SearchRun(args).run()
        finally:
            rs.OUT_DIR = old_out
        ledger = Ledger(args.ledger)
        tested = ledger.tested_by_label()
        rows = payload['shortlist']
        h_c2c = ledger.hashes_for(['c2c'])
        h_vwap = ledger.hashes_for(['vwap'])
        h_both = ledger.hashes_for(['c2c', 'vwap'])
        checks = {
            'payload.labels': payload.get('labels') == ['c2c', 'vwap'],
            'rows_have_both_tiers': bool(rows) and all(
                set(r.get('labels', {})) == {'c2c', 'vwap'} for r in rows),
            'resid_present': all(r.get('resid') is not None for r in rows),
            'dual_fields': all('pass_dual' in r and 'dual_ratio' in r for r in rows),
            'tested_both': tested.get('c2c', 0) > 0 and tested.get('vwap', 0) > 0,
            'exclude_c2c_nonempty': len(h_c2c) > 0,
            'exclude_both_subset': h_both.issubset(h_c2c) and h_both.issubset(h_vwap),
        }
        ok = all(checks.values())
        return ok, [f'  L2 {k}={v}' for k, v in checks.items()] + \
            [f"  L2 tested_by_label={tested} n_full={payload['n_full']} "
             f"shortlist={len(rows)} {'PASS' if ok else 'FAIL'}"]


def r1_residual(frame) -> tuple[bool, list[str]]:
    ctx = build_fast_context(frame, n_stocks=600)
    m_size = resid_metrics(ctx, field('MARKET_CAP'), name='mktcap')
    m_amihud = resid_metrics(ctx, FIXTURES['amihud120'](), name='amihud120')
    rric = m_size['resid_rank_ic']
    # MARKET_CAP 对自身轴正交化后残差为恒零：resid screen 无效(字段None)或 IC≈0，均属预期
    ok_size = ((rric is None or abs(rric) < 0.01)
               and (m_size['drop'] is None or m_size['drop'] > 0.8))
    drop_a = m_amihud['drop']
    ok_amihud = (drop_a is not None and drop_a > 0.2
                 and (m_amihud['resid_sia'] or 0) < (m_amihud['raw_sia'] or 0))
    ok = bool(ok_size and ok_amihud)
    rric_text = 'None(残差恒零→无效)' if rric is None else f'{rric:+.5f}'
    return ok, [
        f"  R1 MARKET_CAP raw_sia={m_size['raw_sia']:.4f} "
        f"resid_ric={rric_text} drop={m_size['drop']} "
        f"{'PASS' if ok_size else 'FAIL'}",
        f"  R1 amihud120 raw_sia={m_amihud['raw_sia']:.4f} "
        f"resid_sia={m_amihud['resid_sia']:.4f} drop={m_amihud['drop']:.3f} "
        f"{'PASS' if ok_amihud else 'FAIL'}",
    ]


def t1_mechanism() -> tuple[bool, list[str]]:
    from grammar import axis_families
    from run_search import _axis_plan
    rng = random.Random(7)
    raw_turn = log_turn = size_mktcap = 0
    for _ in range(400):
        _, node = random_tree(rng, family='M2_liquidity')
        for _, n in walk(node):
            if n.op == 'TS_MEAN' and n.args:
                child = n.args[0]
                if child.op == 'LOG' and child.args and child.args[0].op == 'FIELD' \
                        and child.args[0].name == 'TURNOVER':
                    log_turn += 1
                if child.op == 'FIELD' and child.name == 'TURNOVER':
                    raw_turn += 1
            if n.op == 'LOG' and n.args and n.args[0].op == 'FIELD' \
                    and n.args[0].name == 'MARKET_CAP':
                size_mktcap += 1
    ok_variants = raw_turn >= 20 and log_turn >= 20 and size_mktcap >= 20
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
    axes_default = set(axis_families(allow_documented=False))
    ok_plan = (sum(quotas.values()) == 40 and set(quotas) == axes_default
               and counts.get('trend_position') == 5
               and all(q >= quotas['trend_position'] for a, q in quotas.items()
                       if a != 'trend_position'))
    ok = bool(ok_variants and ok_plan)
    return ok, [
        f'  T1 M2 variants raw_turn={raw_turn} log_turn={log_turn} '
        f'log_mktcap={size_mktcap} {"PASS" if ok_variants else "FAIL"}',
        f'  T1 axis plan sum={sum(quotas.values())} axes={len(quotas)} '
        f'trend_position_tested={counts.get("trend_position")} '
        f'{"PASS" if ok_plan else "FAIL"}',
    ]


def t2_grammar_v2() -> tuple[bool, list[str]]:
    """grammar v2：M9 复合 / M10 路径 / M11 状态门 / M12 文档族 / M13 技术指标 /
    AMOUNT 安全尺度 / 深度 / render-parse 往返。"""
    import grammar as g
    rng = random.Random(11)
    # 1) 新族可生成且只用 VERIFIED 算子
    samples = {}
    for fam in ('M9_composite', 'M10_shape', 'M11_regime'):
        ok = True
        for _ in range(60):
            f, node = g.random_tree(rng, family=fam)
            v, _ = g.validate(node)
            ok &= bool(f == fam and v)
        samples[fam] = ok
    # 2) 复合结构：>=2 个 RANK + 至少 1 个 ADD
    _, comp = g.random_tree(rng, family='M9_composite')
    n_rank = sum(1 for _, n in walk(comp) if n.op == 'RANK')
    n_add = sum(1 for _, n in walk(comp) if n.op == 'ADD')
    ok_comp = n_rank >= 2 and n_add >= 1
    # 3) 文档族默认拒绝、allow_documented 后通过
    doc_default_ok = True
    doc_allow_ok = True
    for fam in ('M12_orderstats', 'M13_tech'):
        for _ in range(30):
            f, node = g.random_tree(rng, family=fam, allow_documented=True)
            v0, _ = g.validate(node)
            v1, _ = g.validate(node, allow_documented=True)
            doc_default_ok &= (not v0)
            doc_allow_ok &= bool(v1)
    # 4) 文档轴默认不入计划、开关后入计划
    from grammar import axis_families
    ok_doc_axis = ('order_stats' not in axis_families(False)) and \
        ('tech' not in axis_families(False)) and \
        ('order_stats' in axis_families(True)) and ('tech' in axis_families(True))
    # 5) AMOUNT 安全尺度：M2/M9 中凡以 AMOUNT 为分母的式子必须 ×1e9
    ok_scale = True
    for _ in range(120):
        _, node = g.random_tree(rng, family='M2_liquidity')
        _, comp2 = g.random_tree(rng, family='M9_composite')
        for tree in (node, comp2):
            for _, n in walk(tree):
                if n.op == 'DIV' and n.args and n.args[1].op == 'FIELD' \
                        and n.args[1].name == 'AMOUNT':
                    has_scale = any(
                        m.op == 'MUL' and any(
                            c.op == 'CONST' and c.value == 1000000000.0
                            for c in m.args)
                        for _, m in walk(tree))
                    ok_scale &= has_scale
    # 6) 技术指标 render→parse 往返一致（公式即平台提交文本）
    from parser import parse_formula
    ok_round = True
    for _ in range(40):
        _, node = g.random_tree(rng, family='M13_tech', allow_documented=True)
        text = g.render(node)
        try:
            back = parse_formula(text)
            ok_round &= (g.render(back) == text)
        except Exception:
            ok_round = False
    # 深度上限 16（comp6 类复合实测 ~15）
    ok_depth = all(g.depth(g.random_tree(rng)[1]) <= g.MAX_DEPTH for _ in range(120))
    checks = {
        'M9': samples['M9_composite'], 'M10': samples['M10_shape'],
        'M11': samples['M11_regime'], 'composite_shape': ok_comp,
        'doc_default_reject': doc_default_ok, 'doc_allow_pass': doc_allow_ok,
        'doc_axis_gate': ok_doc_axis, 'amount_scale': ok_scale,
        'tech_roundtrip': ok_round, 'depth<=16': ok_depth,
    }
    ok = all(checks.values())
    return ok, [f'  T2 {k}={v}' for k, v in checks.items()] + \
        [f'  T2 grammar v2 {"PASS" if ok else "FAIL"}']


def run_all() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    t0 = time.time()
    print('=' * 78)
    print('B2 验收：标签系统 + 多标签 + 残差 IC + 机制/log + grammar v2')
    print('=' * 78)
    frame = load_panel()
    print(f'panel rows={len(frame):,} {time.time() - t0:.0f}s\n', flush=True)
    checks = [('L1', lambda: l1_vwap_label(frame)), ('L2', l2_multi_label),
              ('R1', lambda: r1_residual(frame)), ('T1', t1_mechanism),
              ('T2', t2_grammar_v2)]
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
