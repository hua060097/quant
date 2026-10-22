#!/usr/bin/env python3
"""Quarterly attribution using existing A2 CV/OOS details and minute pkls."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rqalpha_minute_datasource'))


def _nav_mdd(path: Path, start, end) -> tuple[float, float, int]:
    r = pickle.load(open(path, 'rb'))['Top-1']
    dates = list(r['trading_dates'])
    vals = np.asarray(r['daily_portfolio_values'], dtype=float)
    if len(vals) > len(dates):
        vals = vals[1:1 + len(dates)]
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    idx = [i for i, d in enumerate(dates) if s <= pd.Timestamp(d) <= e]
    if len(idx) < 2:
        return float('nan'), float('nan'), len(idx)
    v = vals[idx]
    nav = v / v[0]
    return float(nav[-1] - 1), float(np.min(nav / np.maximum.accumulate(nav) - 1)), len(idx)


def main() -> int:
    root = Path('.')
    cv = pd.read_csv(root / 'results/from_server_20261021/A2_cv_top5_details.csv')
    cv = cv[(cv.param == 'H23') & (cv.cp == 170) & (cv['rank'] <= 30)
            & (cv.universe == 'dedup')].copy()
    cv['date'] = pd.to_datetime(cv.date)
    cv['y1'] = (cv['return'] >= 0.07).astype(float)
    oos = pd.read_csv(root / 'results/from_server_20261021/oos2026_A2_raw_oldmask_top5_details.csv')
    oos = oos[(oos.param == 'H23') & (oos.cp == 170) & (oos['rank'] <= 5)
              & (oos.universe == 'dedup')].copy()
    oos['date'] = pd.to_datetime(oos.date)
    oos['y1'] = (oos['return'] >= 0.07).astype(float)

    cv_pkl = root / 'results/exec_overlay_cv_20261022_Exec_Confirm_50.pkl'
    oos_pkl = root / 'results/exec_overlay_oos_20261022/OOS_Exec_Confirm_50.pkl'
    rows = []
    for fold, g in cv.groupby('fold_id', sort=True):
        r1 = g[g['rank'] == 1]
        start, end = g.date.min(), g.date.max()
        tot, mdd, nd = _nav_mdd(cv_pkl, start, end)
        rows.append({'period': f'CV_Fold{int(fold):02d}', 'start': start.date(),
                     'end': end.date(), 'top30_y1_rate': g.y1.mean(),
                     'precision1': r1.y1.mean(), 'r1_daily_mean': r1['return'].mean(),
                     'minute_total': tot, 'minute_mdd': mdd, 'minute_days': nd})
    for label, mask in (('2026Q1', oos.date.dt.quarter == 1),
                        ('2026Q2', oos.date.dt.quarter == 2),
                        ('2026Q3', oos.date.dt.quarter == 3)):
        g = oos[mask]
        if g.empty:
            continue
        r1 = g[g['rank'] == 1]
        start, end = g.date.min(), g.date.max()
        tot, mdd, nd = _nav_mdd(oos_pkl, start, end)
        rows.append({'period': label, 'start': start.date(), 'end': end.date(),
                     'top30_y1_rate': g.y1.mean(), 'precision1': r1.y1.mean(),
                     'r1_daily_mean': r1['return'].mean(), 'minute_total': tot,
                     'minute_mdd': mdd, 'minute_days': nd})
    out = pd.DataFrame(rows)
    out.to_csv(root / 'results/step4_quarter_attribution.csv', index=False,
               encoding='utf-8-sig')
    print(out.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
