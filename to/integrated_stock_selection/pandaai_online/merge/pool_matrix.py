"""池归因与替换矩阵（2026-09-20，为 10 月 1-3 日换池决策准备）。

只读计算：不改 pool_state.json。评估在 pool_sim 现窗口（2023-03-01~2026-07-10，
官方池宇宙=剔 ST/停牌/一字板）下进行；候选列由 merge evaluator 现算并入 extra 面板。

产出（stdout）：
1. leave_one_out：7 成员各自对池 Comb 的边际贡献；
2. 替换/加入矩阵：P7 / +mktcap / -highpos120 / +al1 候选 的 Comb 对比；
3. 候选 × 池成员 逐日截面 Spearman（月内全部交易日；dedup 判据 |ρ|>0.7 同轴）。

用法：python pool_matrix.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import pool_legacy as pool_mod
from evaluator import load_panel, Evaluator
from parser import parse_formula

HERE = Path(__file__).resolve().parent

CANDIDATES = {
    'al1_m2_voldev5': ('EMA(((VOLUME)/(TS_MEAN(VOLUME,250)))-(1),5)', 0, 0.01525),
    'al1_m1_retamt5': ('EMA(TS_MEAN(((CLOSE)/(DELAY(CLOSE,1)))-(AMOUNT),5),5)', 1, 0.01354),
    'al1_brk60cv20': ('(-(CLOSE / DELAY(TS_MIN(LOW, 60), 1) - 1) * VOLUME / TS_MEAN(VOLUME, 20))',
                      1, 0.01361),
}


def month_end_spearman(extra: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """候选 × 池成员 相关矩阵：对月内【全部交易日】逐日算截面 Spearman，再跨日取均值。

    [CORR-CALIBRATION 2026-10-02] 原为「每月最后 1 个交易日」，每月仅 1 个截面，
    ρ 估计噪声过大。改用月内全日，与 dedup.py / tier2_info_audit 口径统一。
    函数名沿用 month_end_spearman 以保持调用方兼容。
    """
    d = extra.dropna(subset=['date']).copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.drop_duplicates(subset=['stock_code', 'date'])
    acc = np.zeros((len(cols), len(cols)))
    cnt = 0
    for _, g in d.groupby('date', sort=True):
        block = g[cols].to_numpy(dtype=float)
        ok = np.isfinite(block).all(axis=1)
        block = block[ok]
        if block.shape[0] < 50:
            continue
        ranks = np.column_stack([pd.Series(block[:, j]).rank().to_numpy()
                                 for j in range(len(cols))])
        ranks -= ranks.mean(axis=0)
        sd = ranks.std(axis=0)
        bad = ~(sd > 0)
        if bad.any():
            continue
        ranks = ranks / sd
        corr = ranks.T @ ranks / block.shape[0]
        np.clip(corr, -1.0, 1.0, out=corr)
        acc += corr
        cnt += 1
    if cnt == 0:
        return pd.DataFrame(np.full((len(cols), len(cols)), np.nan),
                            index=cols, columns=cols)
    rho = acc / cnt
    return pd.DataFrame(rho, index=cols, columns=cols)


def main() -> int:
    if hasattr(__import__('sys').stdout, 'reconfigure'):
        __import__('sys').stdout.reconfigure(encoding='utf-8', errors='replace')
    state = pool_mod.load_pool_state()
    entries = pool_mod.all_entries(state)
    extra = pool_mod.build_extra_panel(entries)
    print(f'extra panel: {extra.shape}', flush=True)

    frame = load_panel()
    ev = Evaluator(frame)
    for name, (f, dirn, sia) in CANDIDATES.items():
        v = np.asarray(ev.values(parse_formula(f)), dtype=float)
        d = frame[['stock_code', 'date']].copy()
        d[name] = v
        extra = extra.merge(d, on=['stock_code', 'date'], how='outer')
        print(f'candidate {name}: rows {int(np.isfinite(d[name]).sum()):,}', flush=True)

    state2 = copy.deepcopy(state)
    for name, (f, dirn, sia) in CANDIDATES.items():
        state2['members'].append({'name': name, 'panel': 'inline', 'dir': int(dirn),
                                  'sia': float(sia)})

    print('\n===== 1) leave_one_out（现池 7 成员边际） =====', flush=True)
    loo = pool_mod.leave_one_out(extra, state)
    print(f'full Comb={loo["full_comb"]:.4f}')
    for r in sorted(loo['rows'], key=lambda x: x['delta']):
        print(f'  去掉 {r["dropped"]:24s} Comb={r["Comb"]:.4f} ΔComb={-r["delta"]:+.4f} '
              f'NC={r["NC"]:.4f} Rex={100 * (r["Rex"] or 0):+.1f}% Turn={r["Turnover"]:.2f}')

    members = [e['name'] for e in state['members']]
    base6 = [x for x in members if x != 'highpos120']
    # 2026-09-24：显式定义替换集（旧版按键名子串 'skew'/'slope60' 事后删成员，且
    # 'P7-hp-slope60+mktcap+v' 名实不符——名字声明去掉 slope60 但内容保留）
    sets = {
        'P7（现池）': members,
        'P7+mktcap': members + ['mktcap'],
        'P7-highpos120+mktcap': base6 + ['mktcap'],
        'P7-hp+mktcap+voldev5': base6 + ['mktcap', 'al1_m2_voldev5'],
        'P7-hp+mktcap+retamt5': base6 + ['mktcap', 'al1_m1_retamt5'],
        'P7-hp+mktcap+vol+ret': base6 + ['mktcap', 'al1_m2_voldev5', 'al1_m1_retamt5'],
        'P7-hp+mktcap+all3': base6 + ['mktcap'] + list(CANDIDATES),
        'P7-hp-skew120+mktcap+v+r': [x for x in base6 if x != 'skew120']
                                    + ['mktcap', 'al1_m2_voldev5', 'al1_m1_retamt5'],
        'P7-hp-slope60+mktcap+v+r': [x for x in base6 if x != 'slope60']
                                    + ['mktcap', 'al1_m2_voldev5', 'al1_m1_retamt5'],
        'P7-hp-slope60+mktcap+v': [x for x in base6 if x != 'slope60']
                                  + ['mktcap', 'al1_m2_voldev5'],
    }
    print('\n===== 2) 替换/加入矩阵 =====', flush=True)
    table = pool_mod.comb_table(extra, sets, state2)
    for name, row in table.items():
        print('%-30s n=%d rawA=%.4f NA=%.4f NC=%.4f Rex=%+.1f%% Turn=%.1f Comb=%.4f' % (
            name, row['n_members'], row['rawA'], row['NA'], row['NC'],
            (row['Rex'] or 0) * 100, (row['Turnover'] or 0) * 100, row['Comb']), flush=True)

    cols = list(CANDIDATES) + [e['name'] for e in entries]
    cols = [c for c in cols if c in extra.columns]
    print('\n===== 3) 候选 × 池成员 逐日截面 Spearman（月内全日）=====', flush=True)
    rho = month_end_spearman(extra, cols)
    print(rho.round(3).to_string(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
