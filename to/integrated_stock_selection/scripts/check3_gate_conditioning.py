#!/usr/bin/env python3
"""Check 3: CV-only gate conditioning + truncated specialist candidates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/check3_gate_conditioning_20261023'
OUT.mkdir(parents=True, exist_ok=True)
GATE = ROOT / 'results/merge_v2/gates_search_audit_20261023/gate_search_market.parquet'
DAILY_IC = ROOT / 'results/monthly_turnover_rankic_tail_20261023/daily_turnover_rankic.csv'
TOP = ROOT / 'results/from_server_20261020/cv_top5_details.csv'
POOL = ROOT / 'results/merge_v2/from_server_20261018/specialist_pool.json'
CUT = pd.Timestamp('2025-12-31')
B = 1000
BLOCK = 20
RNG = np.random.default_rng(20261023)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan
    a = pd.Series(x[m]).rank().to_numpy(dtype=float)
    b = pd.Series(y[m]).rank().to_numpy(dtype=float)
    a -= a.mean(); b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def boot_p(x: np.ndarray, y: np.ndarray, rho: float) -> float:
    n = len(x)
    nb = int(np.ceil(n / BLOCK))
    hits = 0
    for _ in range(B):
        starts = RNG.integers(0, max(1, n - BLOCK + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in starts])[:n]
        r = spearman(x[idx], y[idx])
        if np.isfinite(r) and abs(r) >= abs(rho):
            hits += 1
    return (1.0 + hits) / (B + 1.0)


def bh_q(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p); order = np.argsort(p); q = np.full(n, np.nan)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        j = order[n - rank]
        if np.isfinite(p[j]):
            prev = min(prev, p[j] * n / (n - rank + 1))
            q[j] = prev
    return q


def load_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_parquet(GATE)
    g['date'] = pd.to_datetime(g.date)
    gp = g.pivot_table(index='date', columns='gate_id', values='p').sort_index()
    gp = gp.loc[:CUT]
    gv = 2.0 * gp - 1.0
    d = pd.read_csv(DAILY_IC)
    d['date'] = pd.to_datetime(d.date)
    d = d[d.date <= CUT].set_index('date').sort_index()
    d['past20_ic'] = d['rank_ic'].rolling(20, min_periods=20).mean()
    d['future3_ic'] = (d['rank_ic'].shift(-1) + d['rank_ic'].shift(-2)
                       + d['rank_ic'].shift(-3)) / 3.0
    # keep T such that T+1..T+3 are still in CV
    idx = d.index
    valid_t = pd.Series(idx, index=idx).shift(-3).notna()
    d = d[valid_t]
    top = pd.read_csv(TOP, usecols=['param', 'cp', 'scheme', 'rank', 'date', 'return'])
    top = top[(top['param'] == 'H23') & (top.cp == 170) & (top.scheme == 'lgb')
              & (top['rank'] == 1)].copy()
    top['date'] = pd.to_datetime(top.date)
    top = top[top.date <= CUT].set_index('date')['return'].rename('a2_rank1')
    return gv, d[['past20_ic', 'future3_ic']].join(top, how='left')


def main() -> int:
    gv, target = load_signals()
    idx = target.index.intersection(gv.index)
    target = target.reindex(idx).copy()
    signals = gv.reindex(idx).copy()
    signals['past20_turnover_ic'] = target['past20_ic']
    rows = []
    for sig in signals.columns:
        x = signals[sig].to_numpy(dtype=float)
        yf = target['future3_ic'].to_numpy(dtype=float)
        ya = target['a2_rank1'].to_numpy(dtype=float)
        for tname, y in (('future3_turnover_ic', yf), ('a2_rank1_return', ya)):
            rho = spearman(x, y)
            rows.append({'signal': sig, 'target': tname,
                         'n': int((np.isfinite(x) & np.isfinite(y)).sum()),
                         'rho': rho,
                         'p_boot': boot_p(x, y, rho) if np.isfinite(rho) else np.nan})
    res = pd.DataFrame(rows)
    for tname, z in res.groupby('target'):
        q = bh_q(z.p_boot.to_numpy())
        res.loc[z.index, 'q_bh'] = q
        res.loc[z.index, 'sig_bh'] = q <= 0.10
    res.to_csv(OUT / 'conditional_correlations.csv', index=False,
               encoding='utf-8-sig')

    # Top truncated nh60/sec candidates: no local specialist factor panel -> no F1 rerun.
    pool = json.loads(POOL.read_text(encoding='utf-8'))
    rows = []
    for x in pool['factors']:
        if not x.get('selected'):
            continue
        score = abs(float(x.get('t_slope', 0))) * abs(
            float(x.get('hi_ic', 0)) - float(x.get('lo_ic', 0)))
        rows.append({'factor_id': x['factor_id'], 'gate': x['gate'],
                     'form': x.get('suggested_form'), 'score': score,
                     't_slope': x.get('t_slope'), 'hi_ic': x.get('hi_ic'),
                     'lo_ic': x.get('lo_ic'), 'target_class': x.get('target_class')})
    p = pd.DataFrame(rows).sort_values('score', ascending=False)
    p['selected_global_top20'] = False
    p.loc[p.head(20).index, 'selected_global_top20'] = True
    p.to_csv(OUT / 'specialist_pool_scores.csv', index=False, encoding='utf-8-sig')
    for gid in ('GM_nh60_prem_chg', 'GM_sec_top5_persist_level',
                'GM_sec_top5_persist_ratio'):
        p[p.gate == gid].to_csv(OUT / f'truncated_{gid}.csv', index=False,
                                encoding='utf-8-sig')

    lines = [
        '# Check 3: CV-only gate conditioning / truncated specialists', '',
        '## 1. T 日门控 vs future 3D TURNOVER RankIC / A2 Rank1', '',
        '- 目标：未来 T+1..T+3 日 turnover 截面 RankIC 均值；A2 H23@170 Rank1 的 r3 收益。',
        '- 信号：48 个市场门控 g=2p-1，加 past20 日 turnover RankIC；block bootstrap 1000 次（20 日块），BH-FDR q<=0.10。',
        '- CV 仅 <=2025-12-31；A2 Rank1 数据从 2023-03-31 起。', '',
        '### 与 future3_turnover_ic 显著（q<=0.10）', '',
        '| signal | rho | p_boot | q_bh | n |',
        '|---|---:|---:|---:|---:|',
    ]
    z = res[(res.target == 'future3_turnover_ic') & res.sig_bh].sort_values('rho')
    for _, r in z.iterrows():
        lines.append(f"| {r.signal} | {r.rho:+.3f} | {r.p_boot:.4f} | {r.q_bh:.4f} | {int(r.n)} |")
    lines += ['', '### 与 A2 Rank1 显著（q<=0.10）', '',
              '| signal | rho | p_boot | q_bh | n |', '|---|---:|---:|---:|---:|']
    z = res[(res.target == 'a2_rank1_return') & res.sig_bh].sort_values('rho')
    for _, r in z.iterrows():
        lines.append(f"| {r.signal} | {r.rho:+.3f} | {r.p_boot:.4f} | {r.q_bh:.4f} | {int(r.n)} |")
    best = res[res.target == 'future3_turnover_ic'].sort_values('rho').head(5)
    lines += ['', '### future3 IC 最负 5 个', '',
              '| signal | rho | q_bh |', '|---|---:|---:|']
    for _, r in best.iterrows():
        lines.append(f"| {r.signal} | {r.rho:+.3f} | {r.q_bh:.4f} |")
    lines += [
        '', '## 2. 被 Top20 截断的 nh60/sec 候选', '',
        '- `specialist_pool` 中有 15 个 `GM_nh60_prem_chg`、79 个 `GM_sec_top5_persist_level`、34 个 ratio 的 selected 因子；',
        '  但 `select_specialists --max-market-cands 20` 只保留全局 score Top20，因此它们 0 个进入 gain cands。',
        '- 本地没有 `specialist_factors.parquet`，也没有这些门控的 F1 `delta_mean/CI/nblocks/placebo` 记录，',
        '  无法在不重算因子值的情况下诚实地判定 H3；**结论是未验证，而不是通过/不通过**。',
        '',
        '| gate | selected | max score | 最高候选是否在全局 Top20 |',
        '|---|---:|---:|---:|',
    ]
    for gid in ('GM_nh60_prem_chg', 'GM_sec_top5_persist_level',
                'GM_sec_top5_persist_ratio'):
        z = p[p.gate == gid]
        lines.append(f"| {gid} | {len(z)} | {z.score.max() if len(z) else np.nan:.4f} | "
                     f"{'是' if len(z) and z.iloc[0].selected_global_top20 else '否'} |")
    lines += ['', '- Top20 全局截断分位：约 0.1680；nh60 最高 0.0768，sec_top5 level/ratio 最高 .1297/.1530。',
              '- 由于这些候选没有 F1 记录，不能把“未进 cands”误写成“F1 不通过”。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
