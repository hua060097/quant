"""B1 验收（本地 0 算力）：SLOPE 全窗向量化（cumsum）等价性与提速。

V1a 全量面板 slope60（LOG(CLOSE),60）：向量化 vs rolling.apply 逐值最大差
V1b 子面板 5 个 M5 斜率模板：同上
V2  全量面板耗时对比：要求向量化 ≥8x 且 ≤15s
V3  端到端 sanity：slope60 经 DataContext 全量筛选，SiA/方向与既有全量读数一致
"""
from __future__ import annotations

import random
import sys
import time

import numpy as np

import _paths  # noqa: F401
from evaluator import Evaluator, _ols_slope_window, load_panel
from fitness import DataContext, build_fast_context, tier1
from fixtures import FIXTURES
from grammar import random_tree, walk

TOL = 1e-9


def _compare(vec: np.ndarray, app: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(vec) & np.isfinite(app)
    if not ok.any():
        return float('inf'), 0
    diff = np.abs(vec[ok] - app[ok])
    return float(diff.max()), int(ok.sum())


def v1_full(frame) -> tuple[bool, list[str]]:
    ev = Evaluator(frame)
    node = FIXTURES['slope60']()
    child = node.args[0]
    s = ev.evaluate(child)
    t0 = time.time()
    app = ev._apply(s, 60, 60, _ols_slope_window).to_numpy(dtype=float)
    t_app = time.time() - t0
    t0 = time.time()
    vec = ev.slope_vectorized(s, 60).to_numpy(dtype=float)
    t_vec = time.time() - t0
    worst, n = _compare(vec, app)
    ok = worst <= TOL and n > 1_000_000
    speed = t_app / t_vec if t_vec > 0 else float('inf')
    return ok, [f'  V1a full slope60: max|Δ|={worst:.3e} n={n:,} '
                f'apply={t_app:.1f}s vec={t_vec:.1f}s speedup={speed:.1f}x '
                f'{"PASS" if ok else "FAIL"}']


def v1_subset(frame) -> tuple[bool, list[str]]:
    fast = build_fast_context(frame, n_stocks=600, months=24)
    ev = fast.evaluator
    rng = random.Random(3)
    nodes = []
    while len(nodes) < 5:
        _, node = random_tree(rng, family='M5_trend')
        if any(n.op == 'SLOPE' for _, n in walk(node)):
            nodes.append(node)
    log = []
    all_ok = True
    for i, node in enumerate(nodes):
        slope_nodes = [(p, n) for p, n in walk(node) if n.op == 'SLOPE']
        for p, sn in slope_nodes:
            child = sn.args[0]
            s = ev.evaluate(child)
            vec = ev.slope_vectorized(s, int(sn.window)).to_numpy(dtype=float)
            app = ev._apply(s, int(sn.window), int(sn.mp), _ols_slope_window).to_numpy(dtype=float)
            worst, n = _compare(vec, app)
            ok = worst <= TOL
            all_ok &= ok
            log.append(f'  V1b node{i} window={sn.window} max|Δ|={worst:.3e} n={n:,} '
                       f'{"PASS" if ok else "FAIL"}')
    return all_ok, log


def v2_speed(frame) -> tuple[bool, list[str]]:
    ev = Evaluator(frame)
    s = ev.evaluate(FIXTURES['slope60']().args[0])
    t0 = time.time()
    ev._apply(s, 60, 60, _ols_slope_window)
    t_app = time.time() - t0
    t0 = time.time()
    ev.slope_vectorized(s, 60)
    t_vec = time.time() - t0
    speed = t_app / t_vec if t_vec > 0 else float('inf')
    ok = t_vec <= 15.0 and speed >= 8.0
    return ok, [f'  V2 apply={t_app:.1f}s vec={t_vec:.1f}s speedup={speed:.1f}x '
                f'{"PASS" if ok else "FAIL"}']


def v3_screen(frame) -> tuple[bool, list[str]]:
    ctx = DataContext(frame=frame)
    m = ctx.screen(FIXTURES['slope60'](), name='slope60')
    t = tier1(m)
    # 校准值 0.0037 为 aligned_v1 口径（2021-09~2026-08 全样本+z2）实测；数据刷新后需重校
    ok = (abs(t['sia'] - 0.0037) <= 1e-3 and int(m.get('direction', -1)) == 0)
    return ok, [f"  V3 slope60 screen sia={t['sia']:.4f} (期望≈0.0037) "
                f"dir={m.get('direction')} turn={t['turnover']:.3f} "
                f"{'PASS' if ok else 'FAIL'}"]


def run_all() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    t0 = time.time()
    print('=' * 78)
    print('B1 验收：SLOPE 全窗向量化（cumsum）等价 + 提速')
    print('=' * 78)
    frame = load_panel()
    print(f'panel rows={len(frame):,} {time.time() - t0:.0f}s\n', flush=True)
    checks = [('V1a', lambda: v1_full(frame)), ('V1b', lambda: v1_subset(frame)),
              ('V2', lambda: v2_speed(frame)), ('V3', lambda: v3_screen(frame))]
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
