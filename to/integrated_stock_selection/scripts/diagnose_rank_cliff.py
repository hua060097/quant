#!/usr/bin/env python3
"""Zero-training Rank-1 cliff diagnostics from existing CV/OOS prediction rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_ci(x: np.ndarray, n_boot: int, seed: int = 42) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, len(x), len(x))].mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _max2_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, g in df.groupby('fold_id', sort=False):
        for date, d in g.groupby('date', sort=False):
            d = d.sort_values(['rank', 'pred'], ascending=[True, False])
            kept = d.groupby('sector', sort=False, group_keys=False).head(2).copy()
            kept = kept.sort_values(['pred', 'rank'], ascending=[False, True]).head(5)
            kept['rank'] = np.arange(1, len(kept) + 1)
            kept['universe'] = 'max2_per_sector'
            kept['fold_id'] = fold_id
            rows.append(kept)
    if not rows:
        return df.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def _daily_mean(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    d = df[(df['universe'] == universe) & (df['rank'] <= 5)].copy()
    if d.empty:
        return d
    g = d.groupby(['date', 'rank'], sort=True).agg(
        ret=('return', 'mean'),
        win=('return', lambda x: float((x > 0).mean())),
        pred=('pred', 'mean'),
        score_z=('score_z_in_day', 'mean'),
        n=('return', 'size'),
    ).reset_index()
    return g


def _diag_a(df: pd.DataFrame, label: str, n_boot: int) -> dict:
    out = {'label': label, 'universes': {}, 'rank1_minus_rank2': {}}
    for uni in ('raw', 'dedup', 'max2_per_sector'):
        d = _daily_mean(df, uni)
        if d.empty:
            continue
        rec = {'n_dates': int(d['date'].nunique()), 'ranks': {}}
        for rank in range(1, 6):
            s = d[d['rank'] == rank].sort_values('date')
            if s.empty:
                continue
            lo, hi = _bootstrap_ci(s['ret'].to_numpy(), n_boot)
            rec['ranks'][str(rank)] = {
                'n': int(len(s)),
                'mean_ret': float(s['ret'].mean()),
                'win': float(s['win'].mean()),
                'ci95': [lo, hi],
            }
        out['universes'][uni] = rec
    raw = _daily_mean(df, 'raw')
    ded = _daily_mean(df, 'dedup')
    m2 = _daily_mean(df, 'max2_per_sector')
    for uni, d in (('raw', raw), ('dedup', ded), ('max2_per_sector', m2)):
        if d.empty:
            continue
        a = d[d['rank'] == 1][['date', 'ret']].rename(columns={'ret': 'r1'})
        b = d[d['rank'] == 2][['date', 'ret']].rename(columns={'ret': 'r2'})
        m = a.merge(b, on='date', how='inner')
        if not m.empty:
            diff = (m['r1'] - m['r2']).to_numpy()
            lo, hi = _bootstrap_ci(diff, n_boot)
            out['rank1_minus_rank2'][uni] = {
                'n': int(len(diff)), 'mean': float(diff.mean()),
                'ci95': [lo, hi],
            }
    return out


def _bucket_table(df: pd.DataFrame, n_boot: int) -> dict:
    r1 = df[(df['rank'] == 1) & (df['universe'] == 'raw')].copy()
    r2 = df[(df['rank'] == 2) & (df['universe'] == 'raw')].copy()
    if r1.empty or r2.empty:
        return {}
    a = r1.groupby('date').agg(r1_ret=('return', 'mean'),
                               p1=('pred', 'mean'),
                               z1=('score_z_in_day', 'mean')).reset_index()
    b = r2.groupby('date').agg(p2=('pred', 'mean')).reset_index()
    m = a.merge(b, on='date', how='inner')
    if m.empty:
        return {}
    m['gap12'] = m['p1'] - m['p2']
    out = {}
    for col, label in (('gap12', 'gap12'), ('z1', 'score_z_rank1')):
        x = m[col].to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 10:
            continue
        qs = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
        m['_bucket'] = np.digitize(x, qs)
        rec = []
        for k in range(5):
            s = m[m['_bucket'] == k]
            if s.empty:
                continue
            lo, hi = _bootstrap_ci(s['r1_ret'].to_numpy(), n_boot)
            rec.append({'bucket': k + 1, 'n': int(len(s)),
                        'x_min': float(s[col].min()), 'x_max': float(s[col].max()),
                        'r1_mean': float(s['r1_ret'].mean()),
                        'r1_win': float((s['r1_ret'] > 0).mean()),
                        'r1_ci95': [lo, hi]})
        out[label] = rec
    return out


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int,
                    seed: int = 43) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        vals[i] = a[rng.integers(0, len(a), len(a))].mean() \
            - b[rng.integers(0, len(b), len(b))].mean()
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _gap_daily(df: pd.DataFrame) -> pd.DataFrame:
    r1 = df[(df['rank'] == 1) & (df['universe'] == 'raw')].copy()
    r2 = df[(df['rank'] == 2) & (df['universe'] == 'raw')].copy()
    if r1.empty or r2.empty:
        return pd.DataFrame()
    a = r1.groupby('date').agg(r1_ret=('return', 'mean'),
                               p1=('pred', 'mean'),
                               z1=('score_z_in_day', 'mean')).reset_index()
    b = r2.groupby('date').agg(p2=('pred', 'mean')).reset_index()
    m = a.merge(b, on='date', how='inner')
    if not m.empty:
        m['gap12'] = m['p1'] - m['p2']
    return m


def _gap_q5_stats(df: pd.DataFrame, n_boot: int) -> dict:
    m = _gap_daily(df)
    if m.empty:
        return {}
    x = m['gap12'].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(m['r1_ret'].to_numpy(dtype=float))
    if ok.sum() < 10:
        return {}
    qs = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
    m['_bucket'] = np.digitize(x, qs)
    q34 = m[m['_bucket'].isin((2, 3))]
    q5 = m[m['_bucket'] == 4]
    out = {
        'n_q34_dates': int(len(q34)),
        'n_q5_dates': int(len(q5)),
        'q34_mean': float(q34['r1_ret'].mean()),
        'q5_mean': float(q5['r1_ret'].mean()),
        'q34_minus_q5': float(q34['r1_ret'].mean() - q5['r1_ret'].mean()),
        'q5_dates': [str(pd.Timestamp(d).date()) for d in sorted(q5['date'])],
    }
    out['ci95'] = list(_bootstrap_diff(q34['r1_ret'].to_numpy(),
                                       q5['r1_ret'].to_numpy(), n_boot))
    r1 = df[(df['rank'] == 1) & (df['universe'] == 'raw')].copy()
    r1 = r1.merge(m[['date', '_bucket']], on='date', how='inner')
    folds = []
    for fold, g in r1.groupby('fold_id', sort=True):
        g5 = g[g['_bucket'] == 4]
        if g5.empty:
            folds.append({'fold': int(fold), 'n_q5': 0,
                          'q5_mean': None, 'fold_r1_mean': float(g['return'].mean()),
                          'q5_below_fold': None})
            continue
        folds.append({'fold': int(fold), 'n_q5': int(len(g5)),
                      'q5_mean': float(g5['return'].mean()),
                      'fold_r1_mean': float(g['return'].mean()),
                      'q5_below_fold': bool(g5['return'].mean() < g['return'].mean())})
    out['fold_stability'] = folds
    out['q5_below_fold_count'] = sum(1 for x in folds if x['q5_below_fold'])
    return out


def _rolling_quantile_stats(df: pd.DataFrame, n_boot: int,
                            window: int = 60) -> dict:
    m = _gap_daily(df)
    if m.empty:
        return {}
    m = m.sort_values('date').reset_index(drop=True)
    out = {}
    for col, label in (('gap12', 'gap12_roll_q'), ('p1', 'pred_r1_roll_q')):
        vals = m[col].to_numpy(dtype=float)
        buckets = np.full(len(vals), -1, dtype=int)
        for i in range(window, len(vals)):
            hist = vals[i - window:i]
            hist = hist[np.isfinite(hist)]
            if len(hist) < 20 or not np.isfinite(vals[i]):
                continue
            qs = np.quantile(hist, [0.2, 0.4, 0.6, 0.8])
            buckets[i] = int(np.digitize(vals[i], qs))
        rec = []
        for k in range(5):
            s = m[buckets == k]
            if s.empty:
                continue
            lo, hi = _bootstrap_ci(s['r1_ret'].to_numpy(), n_boot)
            rec.append({'bucket': k + 1, 'n': int(len(s)),
                        'r1_mean': float(s['r1_ret'].mean()),
                        'r1_win': float((s['r1_ret'] > 0).mean()),
                        'ci95': [lo, hi]})
        out[label] = rec
    return out


def _q5_feature_stats(df: pd.DataFrame, n_boot: int,
                      features_dir: Path) -> dict:
    if not features_dir.exists():
        return {'skipped': f'missing features dir: {features_dir}'}
    m = _gap_daily(df)
    if m.empty:
        return {}
    x = m['gap12'].to_numpy(dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return {}
    qs = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
    m['_bucket'] = np.digitize(x, qs)
    r1 = df[(df['rank'] == 1) & (df['universe'] == 'raw')].copy()
    r1 = r1.merge(m[['date', '_bucket']], on='date', how='inner')
    all_dates = sorted(pd.to_datetime(df['date'].unique()))
    frames = []
    for d in all_dates:
        fp = features_dir / f'{pd.Timestamp(d):%Y%m%d}.parquet'
        if not fp.exists():
            continue
        try:
            frames.append(pd.read_parquet(
                fp, columns=['stock_code', 'date', 'close', 'pct_chg',
                             'turnover_rate']))
        except Exception:
            continue
    if not frames:
        return {'skipped': f'no daily features under {features_dir}'}
    feat = pd.concat(frames, ignore_index=True)
    feat['stock_code'] = feat['stock_code'].astype(str)
    feat['date'] = pd.to_datetime(feat['date'])
    idx = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    prev5 = {d: all_dates[max(0, idx[d] - 5)] for d in all_dates}
    r1['stock_code'] = r1['stock_code'].astype(str)
    r1['date'] = pd.to_datetime(r1['date'])
    r1 = r1.merge(feat, on=['stock_code', 'date'], how='left')
    r1['d5'] = r1['date'].map(prev5)
    close5 = feat[['stock_code', 'date', 'close']].rename(
        columns={'date': 'd5', 'close': 'close5'})
    r1 = r1.merge(close5, on=['stock_code', 'd5'], how='left')
    r1['ret5'] = r1['close'] / r1['close5'] - 1.0
    out = {'n_q5_rows': int((r1['_bucket'] == 4).sum()),
           'n_q1_q4_rows': int((r1['_bucket'] < 4).sum()), 'metrics': {}}
    for col in ('pct_chg', 'turnover_rate', 'ret5'):
        a = r1.loc[r1['_bucket'] < 4, col].to_numpy(dtype=float)
        b = r1.loc[r1['_bucket'] == 4, col].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 or len(b) == 0:
            continue
        out['metrics'][col] = {
            'q1_q4_mean': float(a.mean()), 'q5_mean': float(b.mean()),
            'diff': float(b.mean() - a.mean()),
            'ci95': list(_bootstrap_diff(b, a, n_boot)),
        }
    return out


def _load_ret5(df: pd.DataFrame, features_dir: Path):
    if not features_dir.exists():
        return None
    r1 = df[(df['rank'] == 1) & (df['universe'] == 'dedup')].copy()
    if r1.empty:
        r1 = df[(df['rank'] == 1) & (df['universe'] == 'raw')].copy()
    if r1.empty:
        return None
    dates = sorted(pd.to_datetime(df['date'].unique()))
    frames = []
    for d in dates:
        fp = features_dir / f'{pd.Timestamp(d):%Y%m%d}.parquet'
        if not fp.exists():
            continue
        try:
            frames.append(pd.read_parquet(
                fp, columns=['stock_code', 'date', 'close', 'turnover_rate']))
        except Exception:
            continue
    if not frames:
        return None
    feat = pd.concat(frames, ignore_index=True)
    feat['stock_code'] = feat['stock_code'].astype(str)
    feat['date'] = pd.to_datetime(feat['date'])
    turn_p30 = feat.groupby('date', sort=False)['turnover_rate'].quantile(0.30)
    turn_p30 = turn_p30.rename('turn_p30').reset_index()
    feat = feat.merge(turn_p30, on='date', how='left')
    idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    prev5 = {d: (dates[idx[d] - 5] if idx[d] >= 5 else pd.NaT) for d in dates}
    r1['stock_code'] = r1['stock_code'].astype(str)
    r1['date'] = pd.to_datetime(r1['date'])
    r1 = r1.merge(feat, on=['stock_code', 'date'], how='left')
    r1['d5'] = r1['date'].map(prev5)
    close5 = feat[['stock_code', 'date', 'close']].rename(
        columns={'date': 'd5', 'close': 'close5'})
    r1 = r1.merge(close5, on=['stock_code', 'd5'], how='left')
    r1['ret5'] = r1['close'] / r1['close5'] - 1.0
    r1['turn_low'] = r1['turnover_rate'] <= r1['turn_p30']
    return r1.groupby(['date', 'fold_id'], sort=False).agg(
        ret5=('ret5', 'mean'), turnover=('turnover_rate', 'mean'),
        turn_low=('turn_low', 'mean')).reset_index()


def _perf(x: pd.Series) -> dict:
    r = pd.to_numeric(x, errors='coerce').to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {}
    nav = np.cumprod(1.0 + r)
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    return {'n': int(len(r)), 'mean_daily': float(r.mean()),
            'win': float((r > 0).mean()), 'nav_end': float(nav[-1]),
            'total_return': float(nav[-1] - 1.0), 'mdd': mdd}


def _perf_eq3(r1: pd.Series, weight: np.ndarray,
              cost: float = 0.0016) -> dict:
    r = pd.to_numeric(r1, errors='coerce').to_numpy(dtype=float)
    w = np.asarray(weight, dtype=float)
    ok = np.isfinite(r) & np.isfinite(w)
    r, w = r[ok], w[ok]
    if len(r) == 0:
        return {}
    ret_day = w * r - w * cost
    eq_ret = ret_day / 3.0
    nav = np.cumprod(1.0 + eq_ret)
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    sd = float(np.std(eq_ret, ddof=1)) if len(eq_ret) > 1 else float('nan')
    sharpe = float(np.mean(eq_ret) / sd * np.sqrt(252.0)) \
        if np.isfinite(sd) and sd > 0 else float('nan')
    return {'n': int(len(r)), 'mean_daily': float(np.mean(eq_ret)),
            'win': float((eq_ret > 0).mean()), 'nav_end': float(nav[-1]),
            'total_return': float(nav[-1] - 1.0), 'mdd': mdd,
            'sharpe': sharpe}


def _rank2_collapse(df: pd.DataFrame) -> dict:
    m = _gap_daily(df)
    if m.empty or 'p2' not in m.columns:
        return {}
    x = m['gap12'].to_numpy(dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return {}
    qs = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
    m['gap_bucket'] = np.digitize(x, qs)
    p2 = m['p2'].to_numpy(dtype=float)
    roll_q = np.full(len(m), np.nan, dtype=float)
    for i in range(60, len(m)):
        hist = p2[i - 60:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 20 or not np.isfinite(p2[i]):
            continue
        qq = np.quantile(hist, [0.2, 0.4, 0.6, 0.8])
        roll_q[i] = float(np.digitize(p2[i], qq) + 1)
    m['p2_roll_q'] = roll_q
    q5 = m[m['gap_bucket'] == 4]
    other = m[m['gap_bucket'] < 4]
    return {
        'n_q5': int(len(q5)),
        'q5_p2_mean': float(q5['p2'].mean()),
        'q1q4_p2_mean': float(other['p2'].mean()),
        'q5_p2_roll_q_mean': float(q5['p2_roll_q'].mean()),
        'q1q4_p2_roll_q_mean': float(other['p2_roll_q'].mean()),
    }


def _overlay_strategies(df: pd.DataFrame, features_dir: Path,
                        periods: bool = False) -> dict:
    m = _gap_daily(df)
    if m.empty:
        return {}
    r1 = df[(df['rank'] == 1) & (df['universe'] == 'dedup')].copy()
    if r1.empty:
        r1 = df[(df['rank'] == 1) & (df['universe'] == 'raw')].copy()
    top = df[(df['rank'] <= 3) & (df['universe'] == 'dedup')].copy()
    if top.empty:
        top = df[(df['rank'] <= 3) & (df['universe'] == 'raw')].copy()
    fold_map = r1.groupby('date', sort=False)['fold_id'].first()
    fold_unique = r1.groupby('date', sort=False)['fold_id'].nunique().le(1).all()
    r1 = r1.groupby('date', sort=False).agg(r1=('return', 'mean')).reset_index()
    top = top.groupby('date', sort=False).agg(top3=('return', 'mean')).reset_index()
    panel = m[['date', 'gap12']].merge(r1, on='date', how='inner')
    panel = panel.merge(top, on='date', how='inner')
    if panel.empty:
        return {}
    panel['fold_id'] = panel['date'].map(fold_map)
    ret5 = _load_ret5(df, features_dir)
    if ret5 is not None:
        ret5 = ret5.groupby('date', sort=False).agg(
            ret5=('ret5', 'mean')).reset_index()
        panel = panel.merge(ret5, on='date', how='left')
    else:
        panel['ret5'] = np.nan
    panel = panel.sort_values('date').reset_index(drop=True)
    gap = panel['gap12'].to_numpy(dtype=float)
    q = np.full(len(panel), -1, dtype=int)
    for i in range(60, len(panel)):
        hist = gap[i - 60:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < 20 or not np.isfinite(gap[i]):
            continue
        qs = np.quantile(hist, [0.2, 0.4, 0.6, 0.8])
        q[i] = int(np.digitize(gap[i], qs))
    panel['q'] = q
    high = panel['q'].to_numpy() == 4
    weak = panel['ret5'].to_numpy(dtype=float) <= 0
    turn_low = panel.get('turn_low', pd.Series(np.nan, index=panel.index))
    turn_low = turn_low.to_numpy(dtype=float) >= 0.5
    scale_a = np.where(high, 0.5, 1.0)
    scale_a_skip = np.where(high, 0.0, 1.0)
    scale_b = np.where(high & weak, 0.5, 1.0)
    scale_b_skip = np.where(high & weak, 0.0, 1.0)
    scale_c_half = np.where(weak, 0.5, 1.0)
    scale_c_skip = np.where(weak, 0.0, 1.0)
    scale_d_half = np.where(turn_low, 0.5, 1.0)
    scale_d_skip = np.where(turn_low, 0.0, 1.0)
    out = {}
    for name, sc in (('baseline', np.ones(len(panel))),
                     ('A_half', scale_a), ('A_skip', scale_a_skip),
                     ('B_half', scale_b), ('B_skip', scale_b_skip),
                     ('C_ret5_half', scale_c_half), ('C_ret5_skip', scale_c_skip),
                     ('D_turnlow_half', scale_d_half), ('D_turnlow_skip', scale_d_skip)):
        out[name + '_rank1'] = _perf(panel['r1'] * sc)
        out[name + '_3slot'] = _perf(panel['top3'] + (sc - 1.0) * panel['r1'] / 3.0)
        out[name + '_eq3'] = _perf_eq3(panel['r1'], sc)
    out['n_dates'] = int(len(panel))
    out['n_signal_A'] = int(high.sum())
    out['n_signal_B'] = int((high & weak).sum())
    out['n_signal_C'] = int(weak.sum())
    out['n_signal_D'] = int(turn_low.sum())
    out['by_fold'] = {}
    if fold_unique:
        for fold, g in panel.groupby('fold_id', sort=False):
            hi = g['q'].to_numpy() == 4
            wk = g['ret5'].to_numpy(dtype=float) <= 0
            sca = np.where(hi, 0.5, 1.0)
            scb = np.where(hi & wk, 0.5, 1.0)
            scs = np.where(hi & wk, 0.0, 1.0)
            out['by_fold'][str(int(fold))] = {
                'baseline_rank1': _perf(g['r1']),
                'A_half_rank1': _perf(g['r1'] * sca),
                'B_half_rank1': _perf(g['r1'] * scb),
                'B_skip_rank1': _perf(g['r1'] * scs),
                'baseline_3slot': _perf(g['top3']),
                'A_half_3slot': _perf(g['top3'] + (sca - 1.0) * g['r1'] / 3.0),
                'B_half_3slot': _perf(g['top3'] + (scb - 1.0) * g['r1'] / 3.0),
                'B_skip_3slot': _perf(g['top3'] + (scs - 1.0) * g['r1'] / 3.0),
            }
    if periods:
        months = pd.to_datetime(panel['date']).dt.month
        for label, mask in (('H1', months <= 6), ('Q3', months >= 7)):
            g = panel[mask]
            if g.empty:
                continue
            hi = g['q'].to_numpy() == 4
            wk = g['ret5'].to_numpy(dtype=float) <= 0
            sca = np.where(hi, 0.5, 1.0)
            scb = np.where(hi & wk, 0.5, 1.0)
            scs = np.where(hi & wk, 0.0, 1.0)
            gtl = g.get('turn_low', pd.Series(np.nan, index=g.index))
            gtl = gtl.to_numpy(dtype=float) >= 0.5
            sca_skip = np.where(hi, 0.0, 1.0)
            scc = np.where(wk, 0.5, 1.0)
            scd = np.where(gtl, 0.5, 1.0)
            out['period_' + label] = {
                'baseline_rank1': _perf(g['r1']),
                'A_half_rank1': _perf(g['r1'] * sca),
                'A_skip_rank1': _perf(g['r1'] * sca_skip),
                'B_half_rank1': _perf(g['r1'] * scb),
                'B_skip_rank1': _perf(g['r1'] * scs),
                'C_ret5_half_rank1': _perf(g['r1'] * scc),
                'D_turnlow_half_rank1': _perf(g['r1'] * scd),
                'baseline_3slot': _perf(g['top3']),
                'A_half_3slot': _perf(g['top3'] + (sca - 1.0) * g['r1'] / 3.0),
                'B_half_3slot': _perf(g['top3'] + (scb - 1.0) * g['r1'] / 3.0),
                'B_skip_3slot': _perf(g['top3'] + (scs - 1.0) * g['r1'] / 3.0),
                'baseline_eq3': _perf_eq3(g['r1'], np.ones(len(g))),
                'A_half_eq3': _perf_eq3(g['r1'], sca),
                'A_skip_eq3': _perf_eq3(g['r1'], sca_skip),
                'C_ret5_half_eq3': _perf_eq3(g['r1'], scc),
                'D_turnlow_half_eq3': _perf_eq3(g['r1'], scd),
            }
    return out


def _load_cv(tasks_dir: Path, cp: int) -> pd.DataFrame:
    files = sorted(tasks_dir.glob('*_rows.parquet'))
    if not files:
        raise SystemExit(f'no rows.parquet under {tasks_dir}')
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        if 'cp' in d.columns:
            d = d[d['cp'] == cp]
        d = d[d['rank'] <= 30]
        if not d.empty:
            frames.append(d)
    if not frames:
        raise SystemExit(f'no cp={cp} rows')
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cv-tasks-dir', default='results/stage15_cv_A2_old84_yvalid_20261021/cv_tasks')
    ap.add_argument('--oos-details', default='results/stage15_oos_A2_raw_oldmask_20261021/oos2026_top5_details.csv')
    ap.add_argument('--cp', type=int, default=170)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--features-dir', default='../local_data/features/daily',
                    help='local daily features for Q5 feature profile; set empty to skip')
    ap.add_argument('--out-dir', default='results/rank_cliff_diag_20261022')
    args = ap.parse_args()

    cv = _load_cv(Path(args.cv_tasks_dir), args.cp)
    cv['date'] = pd.to_datetime(cv['date'])
    cv_raw = cv[cv['universe'] == 'raw'].copy()
    cv_max2 = _max2_rows(cv_raw)
    cv_all = pd.concat([cv, cv_max2], ignore_index=True)

    oos = pd.read_csv(args.oos_details)
    oos = oos[oos['cp'] == args.cp].copy()
    oos['date'] = pd.to_datetime(oos['date'])
    oos = oos[oos['rank'] <= 5].copy()
    oos_raw = oos[oos['universe'] == 'raw'].copy()
    oos_max2 = _max2_rows(oos_raw)
    oos_all = pd.concat([oos, oos_max2], ignore_index=True)

    cv_buckets = _bucket_table(cv_all, args.bootstrap)
    oos_buckets = _bucket_table(oos_all, args.bootstrap)
    cv_gap = _gap_q5_stats(cv_all, args.bootstrap)
    oos_gap = _gap_q5_stats(oos_all, args.bootstrap)
    cv_rolling = _rolling_quantile_stats(cv_all, args.bootstrap)
    oos_rolling = _rolling_quantile_stats(oos_all, args.bootstrap)
    cv_feature = _q5_feature_stats(cv_all, args.bootstrap,
                                   Path(args.features_dir)) \
        if args.features_dir else {'skipped': 'disabled'}
    cv_overlay = _overlay_strategies(cv_all, Path(args.features_dir),
                                     periods=False) if args.features_dir else {}
    oos_overlay = _overlay_strategies(oos_all, Path(args.features_dir),
                                      periods=True) if args.features_dir else {}
    cv_rank2 = _rank2_collapse(cv_all)
    oos_rank2 = _rank2_collapse(oos_all)
    report = {
        'cp': args.cp,
        'bootstrap': args.bootstrap,
        'cv': {'rows': int(len(cv_all)), 'dates': int(cv_all['date'].nunique()),
               'diagnostics': _diag_a(cv_all, 'CV_H23_11fold_3seed', args.bootstrap),
               'buckets': cv_buckets, 'gap_q5': cv_gap,
               'rolling': cv_rolling, 'q5_features': cv_feature,
               'overlay': cv_overlay, 'rank2_collapse': cv_rank2},
        'oos': {'rows': int(len(oos_all)), 'dates': int(oos_all['date'].nunique()),
                'diagnostics': _diag_a(oos_all, 'OOS_2026', args.bootstrap),
                'buckets': oos_buckets, 'gap_q5': oos_gap,
                'rolling': oos_rolling, 'overlay': oos_overlay,
                'rank2_collapse': oos_rank2},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Rank-1 Cliff Diagnostics', '',
             f'- cp={args.cp}; bootstrap={args.bootstrap}',
             f"- CV rows={len(cv_all):,}, dates={cv_all['date'].nunique()}; OOS rows={len(oos_all):,}, dates={oos_all['date'].nunique()}",
             '', '## A. 去重/native/max2 下的 Rank 1-5', '']
    for sec in ('cv', 'oos'):
        d = report[sec]['diagnostics']
        lines += [f"### {sec.upper()}", '',
                  '| universe | rank | n | mean_ret | win | CI95 |',
                  '|---|---:|---:|---:|---:|---|']
        for uni, rec in d.get('universes', {}).items():
            for rank, v in rec.get('ranks', {}).items():
                lines.append(f"| {uni} | {rank} | {v['n']} | {v['mean_ret']:.4%} | "
                             f"{v['win']:.2%} | [{v['ci95'][0]:.4%}, {v['ci95'][1]:.4%}] |")
        lines.append('')
        for uni, v in d.get('rank1_minus_rank2', {}).items():
            lines.append(f"- {uni}: Rank1-Rank2 = {v['mean']:.4%} "
                         f"(95% CI [{v['ci95'][0]:.4%}, {v['ci95'][1]:.4%}], n={v['n']})")
        lines.append('')
        lines += ['#### B. gap12 / score_z buckets', '',
                  '| bucket | n | x range | Rank1 mean | Rank1 win | CI95 |',
                  '|---:|---:|---|---:|---:|---|']
        for col, rec in report[sec]['buckets'].items():
            for v in rec:
                lines.append(f"| {col} Q{v['bucket']} | {v['n']} | "
                             f"[{v['x_min']:.4g}, {v['x_max']:.4g}] | "
                             f"{v['r1_mean']:.4%} | {v['r1_win']:.2%} | "
                             f"[{v['r1_ci95'][0]:.4%}, {v['r1_ci95'][1]:.4%}] |")
        g = report[sec].get('gap_q5') or {}
        if g:
            lines += ['#### C. gap12 Q3+Q4 vs Q5', '',
                      f"- Q3+Q4 mean={g['q34_mean']:.4%}; Q5 mean={g['q5_mean']:.4%}; "
                      f"diff={g['q34_minus_q5']:.4%} "
                      f"(95% CI [{g['ci95'][0]:.4%}, {g['ci95'][1]:.4%}], "
                      f"n_q5={g['n_q5_dates']})",
                      f"- Q5 below fold Rank1 mean: **{g['q5_below_fold_count']} folds**"]
            for f in g.get('fold_stability', []):
                if f.get('n_q5'):
                    lines.append(f"  - fold {f['fold']}: Q5 {f['q5_mean']:.4%} "
                                 f"vs fold {f['fold_r1_mean']:.4%} "
                                 f"=> below={f['q5_below_fold']}")
            lines.append('')
        r = report[sec].get('rolling') or {}
        if r:
            lines += ['#### D. 60-day causal rolling quantiles', '',
                      '| signal | bucket | n | Rank1 mean | Rank1 win | CI95 |',
                      '|---|---:|---:|---:|---:|---|']
            for col, rec in r.items():
                for v in rec:
                    lines.append(f"| {col} Q{v['bucket']} | {v['n']} | "
                                 f"{v['r1_mean']:.4%} | {v['r1_win']:.2%} | "
                                 f"[{v['ci95'][0]:.4%}, {v['ci95'][1]:.4%}] |")
            lines.append('')
        if sec == 'cv':
            qf = report[sec].get('q5_features') or {}
            if qf.get('metrics'):
                lines += ['#### E. Q5 Rank1 stock feature profile (T-day, local features)', '',
                          f"- n_q5_rows={qf['n_q5_rows']}, n_q1_q4_rows={qf['n_q1_q4_rows']}",
                          '| feature | Q1-Q4 mean | Q5 mean | Q5-Q1Q4 | 95% CI |',
                          '|---|---:|---:|---:|---|']
                for col, v in qf['metrics'].items():
                    if col == 'ret5':
                        fmt = lambda x: f'{x:.2%}'
                    else:
                        fmt = lambda x: f'{x:.2f}%'
                    lines.append(f"| {col} | {fmt(v['q1_q4_mean'])} | "
                                 f"{fmt(v['q5_mean'])} | {fmt(v['diff'])} | "
                                 f"[{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}] |")
                lines.append('')
            elif qf.get('skipped'):
                lines += [f"#### E. Q5 feature profile skipped: {qf['skipped']}", '']
    lines += ['## F. 后处理策略对拍（日频；3槽=dedup Rank1-3 等权代理）', '']
    for sec in ('cv', 'oos'):
        o = report[sec].get('overlay') or {}
        if not o:
            continue
        lines += [f"### {sec.upper()}", '',
                  f"- signal A={o.get('n_signal_A', 0)} days; "
                  f"signal B={o.get('n_signal_B', 0)} days; n={o.get('n_dates', 0)}",
                  '| strategy | rank1 mean | rank1 total | rank1 MDD | 3slot mean | 3slot total | 3slot MDD |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name in ('baseline', 'A_half', 'B_half', 'B_skip'):
            r1 = o.get(name + '_rank1') or {}
            s3 = o.get(name + '_3slot') or {}
            if not r1 or not s3:
                continue
            lines.append(f"| {name} | {r1['mean_daily']:.4%} | {r1['total_return']:.2%} | "
                         f"{r1['mdd']:.2%} | {s3['mean_daily']:.4%} | "
                         f"{s3['total_return']:.2%} | {s3['mdd']:.2%} |")
        lines.append('')
        if sec == 'cv' and o.get('by_fold'):
            lines += ['| fold | base r1 | A r1 | B r1 | Bskip r1 | base 3slot | A 3slot | B 3slot | Bskip 3slot |',
                      '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for fold, g in o['by_fold'].items():
                vals = []
                for key in ('baseline_rank1', 'A_half_rank1', 'B_half_rank1',
                            'B_skip_rank1', 'baseline_3slot', 'A_half_3slot',
                            'B_half_3slot', 'B_skip_3slot'):
                    vals.append((g.get(key) or {}).get('mean_daily', float('nan')))
                lines.append('| ' + fold + ' | ' + ' | '.join(f'{v:.4%}' for v in vals) + ' |')
            lines.append('')
        for period in ('H1', 'Q3'):
            g = o.get('period_' + period) or {}
            if not g:
                continue
            lines += [f"#### {sec.upper()} {period}", '',
                      '| strategy | rank1 mean | rank1 total | rank1 MDD | 3slot mean | 3slot total | 3slot MDD |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for name in ('baseline', 'A_half', 'B_half', 'B_skip'):
                r1 = g.get(name + '_rank1') or {}
                s3 = g.get(name + '_3slot') or {}
                if not r1:
                    continue
                lines.append(f"| {name} | {r1['mean_daily']:.4%} | {r1['total_return']:.2%} | "
                             f"{r1['mdd']:.2%} | "
                             f"{(s3.get('mean_daily', float('nan'))):.4%} | "
                             f"{(s3.get('total_return', float('nan'))):.2%} | "
                             f"{(s3.get('mdd', float('nan'))):.2%} |")
            lines.append('')
    lines += ['## G. 修正口径验证（Stage15 Rank1-only 3槽；成本 0.16%/次）', '']
    for sec in ('cv', 'oos'):
        rc = report[sec].get('rank2_collapse') or {}
        o = report[sec].get('overlay') or {}
        if rc:
            lines += [f"### {sec.upper()} Rank2 坍塌检查", '',
                      f"- Q5 日 Rank2 pred 均值={rc['q5_p2_mean']:.6f} vs "
                      f"Q1-Q4={rc['q1q4_p2_mean']:.6f}",
                      f"- Q5 日 Rank2 60日滚动分位均值={rc['q5_p2_roll_q_mean']:.3f} vs "
                      f"Q1-Q4={rc['q1q4_p2_roll_q_mean']:.3f}",
                      f"- n_q5={rc['n_q5']}", '']
        if o:
            lines += [f"### {sec.upper()} 3槽净值", '',
                      '| strategy | n | mean/day | total | MDD | Sharpe |',
                      '|---|---:|---:|---:|---:|---:|']
            for name in ('baseline', 'A_half', 'A_skip',
                         'C_ret5_half', 'C_ret5_skip',
                         'D_turnlow_half', 'D_turnlow_skip'):
                v = o.get(name + '_eq3') or {}
                if not v:
                    continue
                lines.append(f"| {name} | {v['n']} | {v['mean_daily']:.4%} | "
                             f"{v['total_return']:.2%} | {v['mdd']:.2%} | "
                             f"{v['sharpe']:+.2f} |")
            lines.append('')
            for period in ('H1', 'Q3'):
                g = o.get('period_' + period) or {}
                if not g:
                    continue
                lines += [f"#### {sec.upper()} {period} 3槽净值", '',
                          '| strategy | total | MDD | Sharpe |',
                          '|---|---:|---:|---:|']
                for name in ('baseline', 'A_half', 'A_skip'):
                    v = g.get(name + '_eq3') or {}
                    if v:
                        lines.append(f"| {name} | {v['total_return']:.2%} | "
                                     f"{v['mdd']:.2%} | {v['sharpe']:+.2f} |")
                lines.append('')
    (out / 'diagnostics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
