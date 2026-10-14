# -*- coding: utf-8 -*-
"""生产标签（部署口径 multihorizon_35）与 TP 代理收益，供 v2 评估统一使用。

与 main_alpha_二分类_原因子_with_test_metrics_妖股_copy_原始_20260720_regression_rankgauss_binary_labelround.py
的 _generate_training_labels_for_all_data 同口径：
  TP(t)    = (open + high + low + close) / 4
  buy(T)   = open(T+1)
  sell3(T) = 0.7*TP(T+2) + 0.3*TP(T+3)            R3 = sell3/buy - 1
  sell5(T) = 0.7*TP(T+4) + 0.3*TP(T+5)            R5 = sell5/buy - 1
  DD3(T)   = (min(low(T+2), low(T+3)) - buy) / buy   （A股T+1不可卖，不纳入T+1低点）
  y_prod   = R3 >= 0.07 & R5 >= R3 & DD3 > -0.05     部署口径（CS_LABEL_FAMILY=multihorizon_35）
  y_r3dd   = R3 >= 0.07 & DD3 > -0.05                对照：去掉持续性条件
  y_fade   = y_r3dd & fade3 <= 0.03                  候选：窗内回落约束
  fade3(T) = (max(high(T+1..T+3)) - sell3) / max_high
可交易掩码（信号日 T）：涨停/跌停/炸板/不可买/成交额<1e7（与 daily_atr_label.py 一致）；
另加 T+1 涨停（开盘不可买入）、T..T+3 任一 ST、标签为 NaN 均视为不可交易。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R3_THR = 0.07
DD_THR = -0.05
FADE_THR = 0.03
AMOUNT_MIN = 1e7
REQ_COLS = ['stock_code', 'date', 'open', 'high', 'low', 'close', 'amount',
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'is_can_buy']


def add_production_labels(df: pd.DataFrame) -> pd.DataFrame:
    """返回按 (stock_code, date) 排序的新 DataFrame，附加标签与可交易掩码列。

    输入至少含 REQ_COLS；可选 is_st（0/1，缺失视为 0）。
    新增列：tp, buy, r3, r5, dd3, fade3, st_win, lu1, tradable, y_prod, y_r3dd, y_fade。
    """
    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise KeyError(f'缺少必需列: {miss}')
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values(['stock_code', 'date'], kind='mergesort').reset_index(drop=True)
    if 'is_st' in out.columns:
        st = out['is_st'].fillna(0).astype(np.int8)
    else:
        st = pd.Series(np.zeros(len(out), dtype=np.int8), index=out.index)
    out['tp'] = (out['open'] + out['high'] + out['low'] + out['close']) / 4.0
    g = out.groupby('stock_code', sort=False)
    gtp = out.groupby('stock_code', sort=False)['tp']
    buy = g['open'].shift(-1)
    sell3 = 0.7 * gtp.shift(-2) + 0.3 * gtp.shift(-3)
    sell5 = 0.7 * gtp.shift(-4) + 0.3 * gtp.shift(-5)
    den = buy + 1e-12
    r3 = (sell3 - buy) / den
    r5 = (sell5 - buy) / den
    minlow = np.fmin(g['low'].shift(-2), g['low'].shift(-3))
    dd3 = (minlow - buy) / den
    maxhigh = np.fmax(np.fmax(g['high'].shift(-1), g['high'].shift(-2)), g['high'].shift(-3))
    fade3 = (maxhigh - sell3) / (maxhigh + 1e-12)
    sg = st.groupby(out['stock_code'], sort=False)
    st_win = (st.astype(bool)
              | sg.shift(-1).fillna(0).astype(bool)
              | sg.shift(-2).fillna(0).astype(bool)
              | sg.shift(-3).fillna(0).astype(bool))
    lu1 = g['is_limit_up'].shift(-1)
    tradable = ((out['is_limit_up'] != 1) & (out['is_limit_down'] != 1) & (out['is_zhaban'] != 1)
                & (out['is_can_buy'] == 1) & (out['amount'] >= AMOUNT_MIN) & (lu1 != 1)
                & (~st_win) & r3.notna() & r5.notna() & dd3.notna())
    y_r3dd = (r3 >= R3_THR) & (dd3 > DD_THR)
    out['buy'] = buy
    out['r3'] = r3
    out['r5'] = r5
    out['dd3'] = dd3
    out['fade3'] = fade3
    out['st_win'] = st_win.astype(np.int8)
    out['lu1'] = lu1
    out['tradable'] = tradable.astype(np.int8)
    out['y_prod'] = ((r3 >= R3_THR) & (r5 >= r3) & (dd3 > DD_THR)).astype(np.int8)
    out['y_r3dd'] = y_r3dd.astype(np.int8)
    out['y_fade'] = (y_r3dd & (fade3 <= FADE_THR)).astype(np.int8)
    return out


def label_stats(df: pd.DataFrame) -> dict:
    """tradable==1 样本上的正类基础率、零正类日占比、头部5%日占正类比例、逐年基础率。"""
    t = df[df['tradable'] == 1]
    years = t['date'].dt.year
    res = {}
    for col in ['y_prod', 'y_r3dd', 'y_fade']:
        per_day = t.groupby('date')[col].agg(['sum', 'count'])
        s = per_day['sum'].astype(float)
        k = max(1, int(round(0.05 * len(s))))
        res[col] = {
            'rate': float(s.sum() / max(1.0, float(per_day['count'].sum()))),
            'zero_day_frac': float((s == 0).mean()) if len(s) else float('nan'),
            'top5pct_day_share': float(s.sort_values(ascending=False).iloc[:k].sum() / max(1.0, float(s.sum()))),
            'yearly': {int(y): round(float(v), 4) for y, v in t.groupby(years)[col].mean().items()},
        }
    res['n_tradable'] = int(len(t))
    res['n_days'] = int(t['date'].nunique())
    return res


def check_vs_atr(df: pd.DataFrame) -> dict:
    """与 daily_atr_label.build_atr_label 的 R3 逐行比对（两边都有限值的行）。"""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from daily_atr_label import build_atr_label
    base = df.copy()
    base['date'] = pd.to_datetime(base['date'])
    ours = add_production_labels(base)[['stock_code', 'date', 'r3']]
    theirs = build_atr_label(base.copy())[['stock_code', 'date', 'atr_return_label']]
    m = ours.merge(theirs, on=['stock_code', 'date'], how='inner')
    ok = m['r3'].notna() & m['atr_return_label'].notna()
    if not ok.any():
        return {'n': 0, 'max_abs_diff': float('nan')}
    d = (m.loc[ok, 'r3'].astype(float) - m.loc[ok, 'atr_return_label'].astype(float)).abs()
    return {'n': int(ok.sum()), 'max_abs_diff': float(d.max())}
