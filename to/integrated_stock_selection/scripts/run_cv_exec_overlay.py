#!/usr/bin/env python3
"""Exec_Confirm_50 vs A_skip/A_half using the real minute engine."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage15_v5_pipeline import (  # noqa: E402
    OOS_MINUTE_OVERRIDES, find_best_price_cache, run_minute_backtest_oos,
)

EXEC = {
    'entry_risk_down_threshold': -0.03,
    'entry_risk_up_threshold': 0.08,
    'enable_first_bar_down_veto': True,
    'first_bar_down_threshold': -0.02,
}


def _q5_dates(df: pd.DataFrame, param: str, cp: int) -> set[pd.Timestamp]:
    d = df[(df['param'].astype(str) == str(param)) & (df['cp'].astype(int) == int(cp))
           & (df['rank'].astype(int) <= 2) & (df['universe'].astype(str) == 'raw')].copy()
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    r1 = d[d['rank'] == 1].groupby('date').agg(p1=('pred', 'mean')).reset_index()
    r2 = d[d['rank'] == 2].groupby('date').agg(p2=('pred', 'mean')).reset_index()
    m = r1.merge(r2, on='date', how='inner').sort_values('date').reset_index(drop=True)
    g = (m['p1'] - m['p2']).to_numpy(dtype=float)
    q = np.full(len(m), -1, dtype=int)
    for i in range(60, len(m)):
        h = g[i - 60:i]
        h = h[np.isfinite(h)]
        if len(h) < 20 or not np.isfinite(g[i]):
            continue
        qs = np.quantile(h, [0.2, 0.4, 0.6, 0.8])
        q[i] = int(np.digitize(g[i], qs))
    return set(m.loc[q == 4, 'date'])


def _pred(df: pd.DataFrame, param: str, cp: int) -> pd.DataFrame:
    d = df[(df['param'].astype(str) == str(param)) & (df['cp'].astype(int) == int(cp))
           & (df['rank'].astype(int) == 1) & (df['universe'].astype(str) == 'dedup')].copy()
    if 'scheme' in d.columns:
        d = d[d['scheme'].astype(str) == 'lgb']
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    d['stock_code'] = d['stock_code'].astype(str)
    return d[['date', 'stock_code', 'pred']].rename(
        columns={'pred': 'predicted_return'}).drop_duplicates('date').reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--details', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--param', default='H23')
    ap.add_argument('--cp', type=int, default=170)
    ap.add_argument('--test-start', required=True)
    ap.add_argument('--test-end', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--price-path', default='')
    ap.add_argument('--configs', default='',
                    help='comma subset: Exec_Confirm_50,A_skip,A_half')
    ap.add_argument('--q5-dates', default='',
                    help='CSV with date column; use when details has no raw rows')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('exec_overlay')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(out / 'run.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    raw = pd.read_csv(args.details)
    if args.q5_dates:
        q5 = set(pd.to_datetime(pd.read_csv(args.q5_dates)['date']).dt.normalize())
    else:
        q5 = _q5_dates(raw, args.param, args.cp)
    pred = _pred(raw, args.param, args.cp)
    pred = pred[(pred['date'] >= pd.Timestamp(args.test_start))
                & (pred['date'] <= pd.Timestamp(args.test_end))].copy()
    if pred.empty:
        raise SystemExit('empty prediction window')
    price_path = Path(args.price_path) if args.price_path else find_best_price_cache(args.test_start)
    logger.info(f'[OVERLAY] {args.tag} pred={len(pred)} q5={sum(d in q5 for d in pred.date)} '
                f'price={price_path}')

    base = pred.copy()
    a_skip = pred.copy()
    a_skip['regime_skip_buy'] = a_skip['date'].isin(q5)
    a_half = pred.copy()
    a_half['regime_position_scale'] = np.where(a_half['date'].isin(q5), 0.5, 1.0)
    rows = []
    _only = {x.strip() for x in args.configs.split(',') if x.strip()}
    for name, p in (('Exec_Confirm_50', base), ('A_skip', a_skip), ('A_half', a_half)):
        if _only and name not in _only:
            continue
        od = out / f'{args.tag}_{name}'
        od.mkdir(parents=True, exist_ok=True)
        res = run_minute_backtest_oos(
            pred_df=p.copy(), price_path=price_path,
            test_start=args.test_start, test_end=args.test_end,
            out_dir=od, logger=logger, strategy_overrides=dict(OOS_MINUTE_OVERRIDES, **EXEC),
            out_name=f'minute_bt_{name}.pkl', position_schedule=[0.5, 0.5],
            max_concurrent_positions=2)
        if not res and (od / f'minute_bt_{name}.pkl').exists():
            res = pd.read_pickle(od / f'minute_bt_{name}.pkl')
        perf = (res or {}).get('Top-1') or {}
        mdd = float(perf.get('max_drawdown', np.nan))
        ann = float(perf.get('annual_return', np.nan)) * 100.0
        rows.append({
            'tag': args.tag, 'config': name,
            'total_return_pct': float(perf.get('total_return', np.nan)),
            'annual_return_pct': ann,
            'max_drawdown_pct': mdd,
            'sharpe': float(perf.get('sharpe_ratio', np.nan)),
            'calmar': ann / abs(mdd) if np.isfinite(ann) and abs(mdd) > 1e-9 else np.nan,
            'trades': int(perf.get('total_trades', 0) or 0),
            'skip_days': int(a_skip['regime_skip_buy'].sum()) if name != 'Exec_Confirm_50' else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / f'{args.tag}_summary.csv', index=False, encoding='utf-8-sig')
    print(df.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
