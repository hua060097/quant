#!/usr/bin/env python3
"""Full-market y=1 rate and Rank1 precision by CV fold / 2026 quarter."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage15_v5_pipeline import load_dataset_and_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='results/merge_v2/lgb_ready_v2/lgb_features_plus2026.parquet')
    ap.add_argument('--cv-details', default='results/stage15_cv_A2_old84_yvalid_20261021/cv_top5_details.csv')
    ap.add_argument('--oos-details', default='results/stage15_oos_A2_raw_oldmask_20261021/oos2026_top5_details.csv')
    ap.add_argument('--out', default='results/full_market_label_audit_20261023')
    args = ap.parse_args()

    log = logging.getLogger('label_audit')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(logging.StreamHandler())
    ns = SimpleNamespace(
        features_parquet=str((ROOT / args.features).resolve()),
        data_start='2021-06-01', cv_end='2026-09-18',
        label_v51=False, enable_buffer_pos=False, mask_mode='old')
    bundle = load_dataset_and_labels(ns, log)
    lab = pd.DataFrame({
        'date': pd.to_datetime(bundle['dates']).normalize(),
        'stock_code': bundle['codes'].astype(str),
        'y': bundle['y'].astype(float),
        'valid': bundle['valid_mask'].astype(bool),
    })

    cv = pd.read_csv(args.cv_details)
    cv = cv[(cv.param == 'H23') & (cv.cp == 170) & (cv['rank'] == 1)
            & (cv.universe == 'dedup')].copy()
    cv['date'] = pd.to_datetime(cv.date).dt.normalize()
    cv['stock_code'] = cv.stock_code.astype(str)
    cv = cv.merge(lab, on=['date', 'stock_code'], how='left')
    oos = pd.read_csv(args.oos_details)
    oos = oos[(oos.param == 'H23') & (oos.cp == 170) & (oos['rank'] == 1)
              & (oos.universe == 'dedup')].copy()
    oos['date'] = pd.to_datetime(oos.date).dt.normalize()
    oos['stock_code'] = oos.stock_code.astype(str)
    oos = oos.merge(lab, on=['date', 'stock_code'], how='left')

    rows = []
    for fold, g in cv.groupby('fold_id', sort=True):
        start, end = g.date.min(), g.date.max()
        full = lab[(lab.date >= start) & (lab.date <= end) & lab.valid]
        rows.append({'period': f'CV_Fold{int(fold):02d}', 'start': start.date(),
                     'end': end.date(), 'full_y1_rate': float(full.y.mean()),
                     'precision1': float(g.loc[g.valid, 'y'].mean()),
                     'rank1_daily_mean': float(g['return'].mean()),
                     'rank1_n': int(g.valid.sum())})
    for q in (1, 2, 3):
        g = oos[oos.date.dt.quarter == q]
        if g.empty:
            continue
        start, end = g.date.min(), g.date.max()
        full = lab[(lab.date >= start) & (lab.date <= end) & lab.valid]
        rows.append({'period': f'2026Q{q}', 'start': start.date(), 'end': end.date(),
                     'full_y1_rate': float(full.y.mean()),
                     'precision1': float(g.loc[g.valid, 'y'].mean()),
                     'rank1_daily_mean': float(g['return'].mean()),
                     'rank1_n': int(g.valid.sum())})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix('.csv'), index=False, encoding='utf-8-sig')
    out.with_suffix('.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    print(df.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
