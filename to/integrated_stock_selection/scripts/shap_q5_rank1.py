#!/usr/bin/env python3
"""SHAP (LightGBM pred_contrib) attribution for Q5 Rank1 samples."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def _load_daily(tasks_dir: Path, cp: int) -> pd.DataFrame:
    frames = []
    for f in sorted(tasks_dir.glob('*_rows.parquet')):
        d = pd.read_parquet(f)
        d = d[(d['cp'] == cp) & (d['rank'] <= 2) & (d['universe'] == 'raw')]
        if not d.empty:
            frames.append(d)
    if not frames:
        raise SystemExit('no rows')
    d = pd.concat(frames, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    return d


def _q5_dates(tasks_dir: Path, cp: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _load_daily(tasks_dir, cp)
    r1 = d[d['rank'] == 1].groupby('date').agg(
        r1=('return', 'mean'), p1=('pred', 'mean'),
        fold=('fold_id', 'first'), stock=('stock_code', 'first')).reset_index()
    r2 = d[d['rank'] == 2].groupby('date').agg(p2=('pred', 'mean')).reset_index()
    m = r1.merge(r2, on='date', how='inner').sort_values('date').reset_index(drop=True)
    m['gap12'] = m['p1'] - m['p2']
    vals = m['gap12'].to_numpy(dtype=float)
    q = np.full(len(m), -1, dtype=int)
    for i in range(60, len(m)):
        h = vals[i - 60:i]
        h = h[np.isfinite(h)]
        if len(h) < 20 or not np.isfinite(vals[i]):
            continue
        qs = np.quantile(h, [0.2, 0.4, 0.6, 0.8])
        q[i] = int(np.digitize(vals[i], qs))
    m['q'] = q
    return m, d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--features-parquet', required=True)
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--rows-dir', default='results/stage15_cv_A2_old84_yvalid_20261021/cv_tasks')
    ap.add_argument('--cp', type=int, default=170)
    ap.add_argument('--out-dir', default='results/shap_q5_20261022')
    args = ap.parse_args()

    q5, raw = _q5_dates(Path(args.rows_dir), args.cp)
    r1 = raw[raw['rank'] == 1].copy()
    r1['date'] = pd.to_datetime(r1['date'])
    r1 = r1.drop_duplicates(subset=['date'], keep='first')
    q5_dates = set(q5.loc[q5['q'] == 4, 'date'])
    r1['is_q5'] = r1['date'].isin(q5_dates)

    df = pd.read_parquet(args.features_parquet)
    df['stock_code'] = df['stock_code'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    keys = ['stock_code', 'date']
    feat_cols = [c for c in df.columns if c not in keys]
    merged = r1[['date', 'stock_code', 'fold_id', 'is_q5']].merge(
        df[keys + feat_cols], on=keys, how='left')
    merged = merged.dropna(subset=feat_cols, how='all')
    print(f'features={len(feat_cols)} q5_dates={len(q5_dates)} rows={len(merged)}')

    contribs = {}
    for fold, g in merged.groupby('fold_id', sort=True):
        pat = str(Path(args.models_dir) / f'lgb_H23_Fold{int(fold):02d}_seed*_cp{args.cp}.txt')
        models = sorted(glob.glob(pat))
        if not models:
            print(f'warning: no models for fold {fold}: {pat}')
            continue
        g = g.reset_index(drop=True)
        acc = None
        for mp in models:
            bst = lgb.Booster(model_file=mp)
            if len(feat_cols) != bst.num_feature():
                raise SystemExit(f'feature count mismatch: matrix={len(feat_cols)} '
                                 f'model={bst.num_feature()} ({mp})')
            X = g[feat_cols].to_numpy(dtype=float)
            c = bst.predict(X, num_iteration=args.cp, pred_contrib=True)
            c = np.asarray(c, dtype=float)[:, :-1]
            acc = c if acc is None else acc + c
        if acc is None:
            continue
        acc /= len(models)
        for j, col in enumerate(feat_cols):
            contribs.setdefault(col, {'q5': [], 'other': []})
            contribs[col]['q5'].append(acc[g['is_q5'].to_numpy(), j])
            contribs[col]['other'].append(acc[~g['is_q5'].to_numpy(), j])

    rows = []
    for col, v in contribs.items():
        a = np.concatenate(v['q5']) if v['q5'] else np.array([])
        b = np.concatenate(v['other']) if v['other'] else np.array([])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(a) or not len(b):
            continue
        rows.append({'feature': col,
                     'q5_pos_mean': float(np.clip(a, 0, None).mean()),
                     'other_pos_mean': float(np.clip(b, 0, None).mean()),
                     'q5_pos_sum': float(np.clip(a, 0, None).sum()),
                     'other_pos_sum': float(np.clip(b, 0, None).sum()),
                     'q5_mean': float(a.mean()), 'other_mean': float(b.mean()),
                     'diff_pos': float(np.clip(a, 0, None).mean()
                                       - np.clip(b, 0, None).mean())})
    rows.sort(key=lambda x: -x['diff_pos'])
    top = rows[:20]
    gate = [r for r in rows if '__GM_newhigh_ratio_chg__gain' in r['feature']]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'shap_q5_rank1.json').write_text(json.dumps(top, ensure_ascii=False, indent=2),
                                            encoding='utf-8')
    (out / 'shap_q5_all.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    (out / 'shap_q5_gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    lines = ['# Q5 Rank1 SHAP (positive contribution, Q5 - Q1-Q4)', '',
             '| feature | Q5 pos mean | Q1-Q4 pos mean | diff |',
             '|---|---:|---:|---:|']
    for r in top:
        lines.append(f"| {r['feature']} | {r['q5_pos_mean']:.6f} | "
                     f"{r['other_pos_mean']:.6f} | {r['diff_pos']:+.6f} |")
    lines += ['', '## __GM_newhigh_ratio_chg__gain gate columns', '',
              '| feature | Q5 pos mean | Q1-Q4 pos mean | diff |',
              '|---|---:|---:|---:|']
    for r in sorted(gate, key=lambda x: -x['q5_pos_mean']):
        lines.append(f"| {r['feature']} | {r['q5_pos_mean']:.6f} | "
                     f"{r['other_pos_mean']:.6f} | {r['diff_pos']:+.6f} |")
    (out / 'shap_q5_rank1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
