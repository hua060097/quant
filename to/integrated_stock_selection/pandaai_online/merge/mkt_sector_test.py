# -*- coding: utf-8 -*-
"""市场/行业因子测试（merge 引擎补充，2026-09-23）。

背景：merge 引擎的 grammar 只覆盖个股公式（OHLCV 截面算子），缺 SW2 行业/市场级
因子的表达与评估。本模块补齐：
  1) SW2 行业映射加载（config/sw2_industry_encoding.pkl）；
  2) 从 factors.parquet 按名字读因子列，映射到 aligned_v1 评估面板行；
  3) 按因子层级自动选择评估口径（用户任务#3"市场和行业因子需要你来定指标"）：
     - 个股层  : 走现有 aligned 标准 screen（RankIC/ICIR(Pearson)/换手/净超额）；此处不重复，
                 提示使用 refs.py / run_search.py。本模块只打印一句指引。
     - 市场层  : 时序口径。f_t=当日因子值(广播)，y_t=当日全市场等权标签。
                 ts_spearman/ts_pearson/t/胜率 + 高/低三分位状态标签均值差 + 近63/126日。
     - 行业层  : 截面口径（行业折叠）。逐期按行业排名与行业均值标签的 RankIC 序列，
                 mean/std/IR/胜率 + 高/低三分位行业标签差。
用法：
  python mkt_sector_test.py --factors market_trend_strength,return_3d_sector_mean --label c2c
  python mkt_sector_test.py --factors mkt_limit_balance,mkt_sector_rotation_speed_20d --label vwap
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import load_panel
from fitness import START, END, DataContext

HERE = Path(__file__).resolve().parent
STORE = HERE.parent.parent / 'factor_lab_data' / 'factors.parquet'
SW_PKL = HERE.parent.parent / 'config' / 'sw2_industry_encoding.pkl'


def load_sw2() -> dict:
    with open(SW_PKL, 'rb') as f:
        obj = pickle.load(f)
    mp = obj.get('mappings', {}).get('stock_to_industry', {})
    if not mp:
        raise RuntimeError(f'SW2 映射为空: {SW_PKL}')
    return {str(k): str(v) for k, v in mp.items()}


def load_factor_on_ctx(ctx: DataContext, name: str) -> np.ndarray:
    """从 store 读取因子列并映射到 ctx 行（缺行=NaN）。"""
    cols = pd.read_parquet(STORE, columns=['stock_code', 'date', name])
    cols['date'] = pd.to_datetime(cols['date'])
    cols = cols[(cols['date'] >= pd.Timestamp(START)) & (cols['date'] <= pd.Timestamp(END))]
    cols['stock_code'] = cols['stock_code'].astype(str)
    ser = cols.set_index(['stock_code', 'date'])[name]
    idx = pd.MultiIndex.from_arrays([ctx.stock_codes.astype(str), pd.to_datetime(ctx.row_dates)])
    return pd.to_numeric(ser.reindex(idx), errors='coerce').to_numpy(dtype=float)


def detect_level(values: np.ndarray, ctx: DataContext, sw2: dict) -> str:
    """按"日内/行业内常值性"判层级。"""
    df = pd.DataFrame({
        'd': ctx.di, 'c': ctx.stock_codes,
        's': pd.Series(ctx.stock_codes).map(sw2).fillna('__NA__').to_numpy(),
        'v': values,
    })
    ok = df['v'].notna()
    if ok.sum() < 1000:
        return 'unknown'
    d_daily = df[ok].groupby('d')['v'].nunique()
    frac_const_day = float((d_daily <= 1).mean())
    if frac_const_day >= 0.9:
        return 'market'
    d_sec = df[ok].groupby(['d', 's'])['v'].nunique()
    frac_const_sec = float((d_sec <= 1).mean())
    if frac_const_sec >= 0.9:
        return 'sector'
    return 'stock'


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    if xr.std() <= 0 or yr.std() <= 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    if x[m].std() <= 0 or y[m].std() <= 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def market_metrics(values: np.ndarray, ctx: DataContext) -> dict:
    df = pd.DataFrame({'d': ctx.di, 'v': values, 'y': ctx.labels}).dropna()
    if df.empty:
        return {}
    g = df.groupby('d').agg(v=('v', 'median'), y=('y', 'mean'))
    v, y = g['v'].to_numpy(), g['y'].to_numpy()
    rho = _spearman(v, y)
    pear = _pearson(v, y)
    n = int(np.isfinite(v).sum())
    t = float(pear * np.sqrt(max(n - 2, 1)) / max(np.sqrt(max(1 - pear ** 2, 1e-12)), 1e-12)) if np.isfinite(pear) else np.nan
    q_hi = np.nanquantile(v, 2 / 3); q_lo = np.nanquantile(v, 1 / 3)
    hi = float(np.nanmean(y[v >= q_hi])) if (v >= q_hi).any() else np.nan
    lo = float(np.nanmean(y[v <= q_lo])) if (v <= q_lo).any() else np.nan
    tail = g.tail(126)
    rho126 = _spearman(tail['v'].to_numpy(), tail['y'].to_numpy())
    tail63 = g.tail(63)
    rho63 = _spearman(tail63['v'].to_numpy(), tail63['y'].to_numpy())
    return {'level': 'market', 'n_dates': n, 'ts_spearman': rho, 'ts_pearson': pear, 'ts_t': t,
            'ts_pos_ratio': float(np.nanmean(np.sign(v) == np.sign(y))) if n else np.nan,
            'state_hi_mean_y': hi, 'state_lo_mean_y': lo, 'state_spread': hi - lo,
            'ts_spearman_63': rho63, 'ts_spearman_126': rho126}


def sector_metrics(values: np.ndarray, ctx: DataContext, sw2: dict) -> dict:
    sec = pd.Series(ctx.stock_codes).map(sw2).to_numpy()
    df = pd.DataFrame({'d': ctx.di, 's': sec, 'v': values, 'y': ctx.labels}).dropna()
    df = df[df['s'].astype(str) != 'nan']
    if df.empty:
        return {}
    g = df.groupby(['d', 's']).agg(v=('v', 'median'), y=('y', 'mean')).reset_index()
    rhos, spreads = [], []
    for d, sub in g.groupby('d'):
        if len(sub) < 10:
            continue
        r = _spearman(sub['v'].to_numpy(), sub['y'].to_numpy())
        if np.isfinite(r):
            rhos.append(r)
        v = sub['v'].to_numpy(); y = sub['y'].to_numpy()
        if np.isfinite(v).sum() >= 10:
            q_hi = np.nanquantile(v, 2 / 3); q_lo = np.nanquantile(v, 1 / 3)
            if (v >= q_hi).any() and (v <= q_lo).any():
                spreads.append(float(np.nanmean(y[v >= q_hi]) - np.nanmean(y[v <= q_lo])))
    rhos = np.array(rhos, dtype=float)
    if len(rhos) == 0:
        return {}
    return {'level': 'sector', 'n_dates': int(len(rhos)),
            'cs_spearman': float(rhos.mean()), 'cs_std': float(rhos.std(ddof=1)) if len(rhos) > 1 else np.nan,
            'cs_ir': float(rhos.mean() / (rhos.std(ddof=1) + 1e-12)) if len(rhos) > 1 else np.nan,
            'cs_pos_ratio': float((rhos > 0).mean()),
            'state_spread': float(np.mean(spreads)) if spreads else np.nan,
            'cs_spearman_63': float(rhos[-63:].mean()) if len(rhos) >= 10 else np.nan,
            'cs_spearman_126': float(rhos[-126:].mean()) if len(rhos) >= 10 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description='市场/行业因子测试（aligned 面板，SW2 分层）')
    ap.add_argument('--factors', required=True, help='逗号分隔的 store 因子名')
    ap.add_argument('--label', default='c2c', choices=['c2c', 'vwap'])
    ap.add_argument('--cycle', type=int, default=3)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    names = [x.strip() for x in args.factors.split(',') if x.strip()]
    print(f'[MKT-SEC] label={args.label}{args.cycle}d  因子 {len(names)} 个')
    ctx = DataContext(cycle=args.cycle, label=args.label)
    print(f'[MKT-SEC] ctx rows={ctx.n_rows:,} dates={len(ctx.ud)}')
    sw2 = load_sw2()
    print(f'[MKT-SEC] SW2 映射: {len(sw2)} 只股票')

    rows = []
    for nm in names:
        try:
            vals = load_factor_on_ctx(ctx, nm)
        except Exception as e:
            print(f'  [ERR] {nm}: 读取失败 {e}')
            continue
        lvl = detect_level(vals, ctx, sw2)
        if lvl == 'market':
            m = market_metrics(vals, ctx)
        elif lvl == 'sector':
            m = sector_metrics(vals, ctx, sw2)
        elif lvl == 'stock':
            print(f'  [SKIP] {nm}: 个股层 → 用 refs.py/run_search.py 走 aligned screen（本模块只管市场/行业）')
            continue
        else:
            print(f'  [SKIP] {nm}: 层级无法判定(覆盖不足)')
            continue
        m['factor'] = nm
        rows.append(m)
        if lvl == 'market':
            print(f'  [{nm}] market tsρ={m["ts_spearman"]:+.3f} t={m["ts_t"]:+.2f} '
                  f'状态差={m["state_spread"]:+.4f} ρ63={m["ts_spearman_63"]:+.3f} ρ126={m["ts_spearman_126"]:+.3f}')
        else:
            print(f'  [{nm}] sector csρ={m["cs_spearman"]:+.4f} IR={m["cs_ir"]:+.3f} '
                  f'胜率={m["cs_pos_ratio"]:.1%} 状态差={m["state_spread"]:+.4f} ρ63={m["cs_spearman_63"]:+.4f}')

    if rows:
        out = pd.DataFrame(rows)
        if args.out:
            p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(p, index=False, encoding='utf-8-sig')
            print(f'[MKT-SEC] 已保存: {p}')
        print('\n[汇总]')
        cols = [c for c in ['factor', 'level', 'ts_spearman', 'ts_t', 'cs_spearman', 'cs_ir',
                            'cs_pos_ratio', 'state_spread', 'ts_spearman_63', 'ts_spearman_126',
                            'cs_spearman_63', 'cs_spearman_126'] if c in out.columns]
        print(out[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
