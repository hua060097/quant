#!/usr/bin/env python3
"""Top30 SHAP rerank: remove/cap stock-habitat contributions and re-pick Rank1."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def _load_models(models_dir: Path, fold: int, cp: int) -> list[str]:
    return sorted(glob.glob(str(models_dir / f'lgb_H23_Fold{fold:02d}_seed*_cp{cp}.txt')))


def _contrib(models: list[str], feat_cols: list[str], g: pd.DataFrame,
             cp: int) -> tuple[np.ndarray, np.ndarray]:
    acc = None
    base = None
    for mp in models:
        bst = lgb.Booster(model_file=mp)
        if bst.num_feature() != len(feat_cols):
            raise SystemExit(f'feature mismatch model={bst.num_feature()} matrix={len(feat_cols)}')
        c = np.asarray(bst.predict(g[feat_cols].to_numpy(dtype=float),
                                   num_iteration=cp, pred_contrib=True), dtype=float)
        base = c[:, -1] if base is None else base + c[:, -1]
        cc = c[:, :-1]
        acc = cc if acc is None else acc + cc
    return acc / len(models), base / len(models)


def _compare(rows: pd.DataFrame, contrib: np.ndarray, feat_cols: list[str],
             base: np.ndarray) -> list[dict]:
    df = rows.copy()
    df['return'] = pd.to_numeric(df['return'], errors='coerce')
    df['pred'] = pd.to_numeric(df['pred'], errors='coerce')
    if '191b2f2c49b2' not in feat_cols or '5beb4eba2fb5' not in feat_cols:
        raise SystemExit('missing target stock-habitat feature')
    i191 = feat_cols.index('191b2f2c49b2')
    i5b = feat_cols.index('5beb4eba2fb5')
    shap191 = np.clip(contrib[:, i191], 0, None)
    shap5b = np.clip(contrib[:, i5b], 0, None)
    pos = np.clip(contrib, 0, None)
    excess = (pos - np.minimum(pos, 0.10)).sum(axis=1)
    df['s_base'] = df['pred']
    df['s1'] = df['pred'] - shap191
    df['s2'] = df['pred'] - shap191 - shap5b
    df['s3'] = df['pred'] - excess
    out = []
    for date, g in df.groupby('date', sort=True):
        base_row = g.loc[g['s_base'].idxmax()]
        rec = {'date': date, 'base_ret': base_row['return'],
               'base_p7': float(base_row['return'] >= 0.07)}
        for s in ('s1', 's2', 's3'):
            r = g.loc[g[s].idxmax()]
            rec[s + '_ret'] = r['return']
            rec[s + '_p7'] = float(r['return'] >= 0.07)
            rec[s + '_stock'] = r['stock_code']
        rec['base_stock'] = base_row['stock_code']
        out.append(rec)
    return out


def _summary(day_rows: list[dict], label: str) -> list[dict]:
    d = pd.DataFrame(day_rows)
    if d.empty:
        return []
    out = []
    for s, name in (('base', 'original'), ('s1', 'minus_191'),
                    ('s2', 'minus_191_5beb'), ('s3', 'cap_0.10')):
        ret = d[s + '_ret'].to_numpy(dtype=float)
        out.append({'period': label, 'scheme': name, 'days': int(len(ret)),
                    'precision7': float(np.nanmean(ret >= 0.07)),
                    'mean_r3': float(np.nanmean(ret))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cv-tasks', default='results/stage15_cv_A2_old84_yvalid_20261021/cv_tasks')
    ap.add_argument('--cv-features', default='results/merge_v2/lgb_ready/lgb_features.parquet')
    ap.add_argument('--oos-details', default='results/stage15_oos_A2_raw_oldmask_20261021/oos2026_top5_details.csv')
    ap.add_argument('--oos-features', default='results/merge_v2/lgb_ready/lgb_features_plus2026.parquet')
    ap.add_argument('--models-dir', default='results/a2_models')
    ap.add_argument('--cp', type=int, default=170)
    ap.add_argument('--out-dir', default='results/top30_shap_rerank_20261023')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv_rows = []
    for f in sorted(Path(args.cv_tasks).glob('*_rows.parquet')):
        d = pd.read_parquet(f)
        d = d[(d.cp == args.cp) & (d['rank'] <= 30) & (d.universe == 'dedup')]
        if not d.empty:
            cv_rows.append(d)
    cv = pd.concat(cv_rows, ignore_index=True)
    cv['date'] = pd.to_datetime(cv.date).dt.normalize()
    cv['stock_code'] = cv.stock_code.astype(str)

    cvf = pd.read_parquet(args.cv_features)
    cvf['date'] = pd.to_datetime(cvf.date).dt.normalize()
    cvf['stock_code'] = cvf.stock_code.astype(str)
    feat_cols = [c for c in cvf.columns if c not in ('stock_code', 'date')]
    cvm = cv.merge(cvf, on=['stock_code', 'date'], how='inner')
    all_days = []
    fold_rows = []
    for fold, gfold in cvm.groupby('fold_id', sort=True):
        models = _load_models(Path(args.models_dir), int(fold), args.cp)
        if not models:
            raise SystemExit(f'no models for fold {fold}')
        gfold = gfold.reset_index(drop=True)
        contrib, base = _contrib(models, feat_cols, gfold, args.cp)
        days = _compare(gfold, contrib, feat_cols, base)
        all_days.extend(days)
        fold_rows.extend(_summary(days, f'CV_Fold{int(fold):02d}'))
        print(f'fold {fold} rows={len(gfold)} days={len(days)}')
    cv_summary = _summary(all_days, 'CV_all')
    del cvf, cvm

    oos = pd.read_csv(args.oos_details)
    oos = oos[(oos.param == 'H23') & (oos.cp == args.cp) & (oos['rank'] <= 30)
              & (oos.universe == 'dedup')].copy()
    oos['date'] = pd.to_datetime(oos.date).dt.normalize()
    oos['stock_code'] = oos.stock_code.astype(str)
    oof = pd.read_parquet(args.oos_features)
    oof['date'] = pd.to_datetime(oof.date).dt.normalize()
    oof['stock_code'] = oof.stock_code.astype(str)
    oos_cols = [c for c in feat_cols if c in oof.columns]
    if len(oos_cols) != len(feat_cols):
        raise SystemExit(f'oos feature mismatch: need {len(feat_cols)} got {len(oos_cols)}')
    oos = oos.merge(oof[['stock_code', 'date'] + oos_cols],
                    on=['stock_code', 'date'], how='inner')
    models = _load_models(Path(args.models_dir), 11, args.cp)
    contrib, base = _contrib(models, oos_cols, oos, args.cp)
    days = _compare(oos, contrib, oos_cols, base)
    dd = pd.DataFrame(days)
    dd['quarter'] = pd.to_datetime(dd.date).dt.quarter
    oos_summary = []
    for q in (1, 2, 3):
        oos_summary.extend(_summary(dd[dd.quarter == q].to_dict('records'), f'2026Q{q}'))
    oos_summary.extend(_summary(days, '2026_all'))

    report = {'cv': cv_summary, 'cv_folds': fold_rows, 'oos': oos_summary,
              'n_features': len(feat_cols)}
    (out / 'top30_shap_rerank.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Top30 SHAP 削峰重排', '',
             '| period | scheme | days | Precision@7% | mean r3 |',
             '|---|---|---:|---:|---:|']
    for r in cv_summary + oos_summary:
        lines.append(f"| {r['period']} | {r['scheme']} | {r['days']} | "
                     f"{r['precision7']:.2%} | {r['mean_r3']:.4%} |")
    lines += ['', '## CV 11 folds', '',
              '| period | scheme | days | Precision@7% | mean r3 |',
              '|---|---|---:|---:|---:|']
    for r in fold_rows:
        lines.append(f"| {r['period']} | {r['scheme']} | {r['days']} | "
                     f"{r['precision7']:.2%} | {r['mean_r3']:.4%} |")
    (out / 'top30_shap_rerank.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
