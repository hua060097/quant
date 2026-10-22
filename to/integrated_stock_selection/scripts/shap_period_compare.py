#!/usr/bin/env python3
"""Compare Rank1 SHAP contributions for 2026 Q2 vs Q3 (Fold11 proxy model)."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--features', required=True)
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--details', required=True)
    ap.add_argument('--model-fold', type=int, default=11)
    ap.add_argument('--cp', type=int, default=170)
    ap.add_argument('--out-dir', default='results/shap_q2q3_20261023')
    args = ap.parse_args()

    det = pd.read_csv(args.details)
    det = det[(det.param == 'H23') & (det.cp == args.cp) & (det['rank'] == 1)
              & (det.universe == 'dedup')].copy()
    det['date'] = pd.to_datetime(det.date).dt.normalize()
    det['stock_code'] = det.stock_code.astype(str)
    det['period'] = np.where(det.date.dt.quarter == 2, 'Q2',
                             np.where(det.date.dt.quarter == 3, 'Q3', ''))
    det = det[det.period != '']

    df = pd.read_parquet(args.features)
    df['stock_code'] = df.stock_code.astype(str)
    df['date'] = pd.to_datetime(df.date).dt.normalize()
    feat_cols = [c for c in df.columns if c not in ('stock_code', 'date')]
    m = det.merge(df, on=['stock_code', 'date'], how='left')
    m = m.dropna(subset=feat_cols, how='all')
    if m.empty:
        raise SystemExit('no matched feature rows')

    models = sorted(glob.glob(str(Path(args.models_dir) /
                                  f'lgb_H23_Fold{args.model_fold:02d}_seed*_cp{args.cp}.txt')))
    if not models:
        raise SystemExit('no Fold11 models')
    acc = None
    for mp in models:
        bst = lgb.Booster(model_file=mp)
        if bst.num_feature() != len(feat_cols):
            raise SystemExit(f'feature mismatch {bst.num_feature()} vs {len(feat_cols)}')
        c = bst.predict(m[feat_cols].to_numpy(dtype=float),
                        num_iteration=args.cp, pred_contrib=True)
        c = np.asarray(c, dtype=float)[:, :-1]
        acc = c if acc is None else acc + c
    acc /= len(models)
    rows = []
    for j, col in enumerate(feat_cols):
        a = acc[(m.period == 'Q2').to_numpy(), j]
        b = acc[(m.period == 'Q3').to_numpy(), j]
        rows.append({
            'feature': col,
            'q2_pos_mean': float(np.clip(a, 0, None).mean()),
            'q3_pos_mean': float(np.clip(b, 0, None).mean()),
            'q2_mean': float(a.mean()), 'q3_mean': float(b.mean()),
            'diff_pos_q3_q2': float(np.clip(b, 0, None).mean()
                                    - np.clip(a, 0, None).mean()),
            'q2_raw_mean': float(m.loc[m.period == 'Q2', col].mean()),
            'q3_raw_mean': float(m.loc[m.period == 'Q3', col].mean()),
        })
    rows.sort(key=lambda x: -x['diff_pos_q3_q2'])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'shap_q2q3.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    lines = ['# 2026 Q2 vs Q3 Rank1 SHAP (Fold11 proxy)', '',
             f'- models={len(models)}, q2_rows={int((m.period=="Q2").sum())}, '
             f'q3_rows={int((m.period=="Q3").sum())}',
             '| feature | Q2 pos | Q3 pos | diff | Q2 raw | Q3 raw |',
             '|---|---:|---:|---:|---:|---:|']
    for r in rows[:30]:
        lines.append(f"| {r['feature']} | {r['q2_pos_mean']:.6f} | "
                     f"{r['q3_pos_mean']:.6f} | {r['diff_pos_q3_q2']:+.6f} | "
                     f"{r['q2_raw_mean']:.4g} | {r['q3_raw_mean']:.4g} |")
    (out / 'shap_q2q3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
