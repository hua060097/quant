#!/usr/bin/env python3
"""Paired Rank1 CV comparison: A2 frozen 84 vs v2-inc feature set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _rows(tasks_dir: Path, cp: int) -> pd.DataFrame:
    frames = []
    for f in sorted(tasks_dir.glob('*_rows.parquet')):
        d = pd.read_parquet(f)
        d = d[(d['cp'] == cp) & (d['rank'] == 1) & (d['universe'] == 'dedup')]
        if not d.empty:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    return d.groupby(['date', 'fold_id'], as_index=False).agg(
        ret=('return', 'mean'))


def _boot(x: np.ndarray, n: int = 1000, seed: int = 42) -> list[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    for i in range(n):
        v[i] = x[rng.integers(0, len(x), len(x))].mean()
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def _summary_score(path: Path, cp: int) -> dict:
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    d = d[d['cp'] == cp]
    return d.iloc[0].to_dict() if len(d) else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--a2-tasks-dir', default='results/stage15_cv_A2_old84_yvalid_20261021/cv_tasks')
    ap.add_argument('--v2-tasks-dir', default='results/stage15_cv_v2inc85_oldmask_20261022/cv_tasks')
    ap.add_argument('--a2-summary', default='results/stage15_cv_A2_old84_yvalid_20261021/cv_summary.csv')
    ap.add_argument('--v2-summary', default='results/stage15_cv_v2inc85_oldmask_20261022/cv_summary.csv')
    ap.add_argument('--cps', default='170,200,260')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--out-dir', default='results/rank_cliff_diag_20261022')
    args = ap.parse_args()

    out = {}
    for cp in [int(x) for x in args.cps.split(',') if x.strip()]:
        a = _rows(Path(args.a2_tasks_dir), cp)
        b = _rows(Path(args.v2_tasks_dir), cp)
        if a.empty or b.empty:
            out[cp] = {'error': 'missing rows', 'a2_n': len(a), 'v2_n': len(b)}
            continue
        m = a.merge(b, on=['date', 'fold_id'], suffixes=('_a2', '_v2'))
        if m.empty:
            out[cp] = {'error': 'no paired dates'}
            continue
        delta = (m['ret_v2'] - m['ret_a2']).to_numpy(dtype=float)
        fold = m.groupby('fold_id').agg(a2=('ret_a2', 'mean'),
                                         v2=('ret_v2', 'mean'))
        fold['delta'] = fold['v2'] - fold['a2']
        fold_sorted = fold.sort_values('delta', ascending=False)
        out[cp] = {
            'n_paired_days': int(len(m)),
            'a2_mean': float(m['ret_a2'].mean()),
            'v2_mean': float(m['ret_v2'].mean()),
            'delta_mean': float(delta.mean()),
            'delta_ci95': _boot(delta, args.bootstrap),
            'fold_win_rate': float((fold['delta'] > 0).mean()),
            'fold_win_count': int((fold['delta'] > 0).sum()),
            'fold_n': int(len(fold)),
            'top3_fold_delta_mean': float(fold_sorted['delta'].head(3).mean()),
            'top3_fold_delta': {str(int(k)): float(v) for k, v in fold_sorted['delta'].head(3).items()},
            'fold_delta': {str(int(k)): float(v) for k, v in fold['delta'].items()},
            'a2_summary': _summary_score(Path(args.a2_summary), cp),
            'v2_summary': _summary_score(Path(args.v2_summary), cp),
        }
        a_ps = out[cp]['a2_summary'].get('plateau_score')
        v_ps = out[cp]['v2_summary'].get('plateau_score')
        out[cp]['plateau_not_worse'] = bool(
            v_ps is not None and a_ps is not None and float(v_ps) >= float(a_ps))
        out[cp]['adopt'] = bool(
            out[cp]['v2_mean'] > out[cp]['a2_mean'] and out[cp]['plateau_not_worse'])

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / 'v2inc_vs_A2.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# v2-inc vs A2 Rank1 CV', '']
    for cp, r in out.items():
        lines += [f'## cp={cp}', '',
                  f"- A2 mean={r.get('a2_mean', float('nan')):.4%}; "
                  f"v2 mean={r.get('v2_mean', float('nan')):.4%}; "
                  f"delta={r.get('delta_mean', float('nan')):.4%} "
                  f"CI={r.get('delta_ci95')}",
                  f"- fold wins={r.get('fold_win_count')}/{r.get('fold_n')}; "
                  f"top3 mean={r.get('top3_fold_delta_mean', float('nan')):.4%}; "
                  f"plateau ok={r.get('plateau_not_worse')}; adopt={r.get('adopt')}", '']
    (out_path / 'v2inc_vs_A2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
