#!/usr/bin/env python3
"""Check 2: pipeline_all G0/G2/G3 and monthly p for July-extreme gates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/check2_gate_pipeline_20261023'
OUT.mkdir(parents=True, exist_ok=True)
GATE = ROOT / 'results/merge_v2/gates_search_audit_20261023/gate_search_market.parquet'
PIPE = ROOT / 'results/merge_v2/gates_search/pipeline_all.csv'
BANK2 = ROOT / 'results/merge_v2/from_server_20261018/gates_search/certified_gate_bank.json'

GATES = [
    'GM_newhigh_ratio_level', 'GM_nh60_prem_chg', 'GM_sec_disp_ratio',
    'GM_vol_med_ratio', 'GM_amp_med_ratio', 'GM_xs_disp_ratio',
    'GM_xs_iqr_ratio', 'GM_sec_disp_level', 'GM_xs_disp_level',
    'GM_amp_med_level', 'GM_xs_iqr_level', 'GM_amt_hhi_level',
    'GM_vol_med_level', 'GM_newhigh_ratio_chg', 'GM_sec_top5_persist_level',
]


def main() -> int:
    g = pd.read_parquet(GATE)
    g['date'] = pd.to_datetime(g.date)
    w = g.pivot_table(index='date', columns='gate_id', values='p').sort_index()
    m = w.resample('ME').mean()
    cols = ['2024-01-31', '2024-02-29'] + [
        str(pd.Timestamp(f'2026-{i:02d}-01') + pd.offsets.MonthEnd(0))
        for i in range(1, 10)]
    tab = m.reindex(columns=GATES).loc[cols].T
    tab.columns = ['2024-01', '2024-02'] + [f'2026-{i:02d}' for i in range(1, 10)]
    tab.to_csv(OUT / 'monthly_p_15gates.csv', encoding='utf-8-sig')

    p = pd.read_csv(PIPE)
    pm = p.set_index('gate_id')
    bank2 = json.loads(BANK2.read_text(encoding='utf-8'))
    b2 = {e['gate_id']: e for e in bank2['entries'] if e['level'] == 'market'}
    rows = []
    for gate in GATES:
        if gate not in pm.index:
            b = b2.get(gate, {})
            rows.append({'gate': gate, 'in_pipeline_all': False,
                         'g0_cont': None, 'neutral': None, 'dir_corr': None,
                         'median_b': b.get('median_b'), 'same_frac': b.get('same_frac'),
                         'folds_same': None, 'ci_lo': None, 'g2_pass': None,
                         'g3_pass': b.get('g3_pass'), 'g3_p': None,
                         'score': b.get('score'), 'in_bank_v2': gate in b2,
                         'elimination': 'not in local 40-gate pipeline_all; '
                                        'server bank-v2 entry available'})
            continue
        r = pm.loc[gate]
        if gate not in b2:
            if bool(r.g2_pass) and not bool(r.g3_pass):
                elim = 'G2 pass; G3 fail; not bank-v2'
            elif bool(r.g3_pass):
                elim = 'G2 fail; G3 pass; not bank-v2 (G4/max-market cap)'
            else:
                elim = 'G2 fail; G3 fail'
        else:
            elim = 'passed to bank-v2'
        rows.append({'gate': gate, 'in_pipeline_all': True,
                     'g0_cont': r.get('g0_cont'), 'neutral': r.get('neutral'),
                     'dir_corr': r.get('dir_corr'), 'median_b': r.get('median_b'),
                     'same_frac': r.get('same_frac'), 'folds_same': r.get('folds_same'),
                     'ci_lo': r.get('ci_lo'), 'g2_pass': bool(r.g2_pass),
                     'g3_pass': bool(r.g3_pass), 'g3_p': r.get('g3_p'),
                     'score': r.get('score'), 'in_bank_v2': gate in b2,
                     'elimination': elim})
    pipe = pd.DataFrame(rows)
    pipe.to_csv(OUT / 'pipeline_metrics_15gates.csv', index=False,
                encoding='utf-8-sig')

    h1 = m.loc['2026-01-31':'2026-06-30', GATES].mean()
    jul = m.loc['2026-07-31', GATES]
    aug_sep = m.loc['2026-08-31':'2026-09-30', GATES].mean()
    summary = pd.DataFrame({'gate': GATES, 'h1_mean': h1.reindex(GATES).values,
                            'july': jul.reindex(GATES).values,
                            'aug_sep_mean': aug_sep.reindex(GATES).values})
    summary.to_csv(OUT / 'h1_vs_july.csv', index=False, encoding='utf-8-sig')

    lines = [
        '# Check 2: 13 个 July 极端门控 + 2 个对照门控', '',
        '## 1. pipeline_all G0/G2/G3 / 淘汰原因', '',
        '| gate | in pipeline | g0 | neutral | dir_corr | median_b | same_frac | folds | ci_lo | G2 | G3 | g3_p | score | bank-v2 | 淘汰原因 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for _, r in pipe.iterrows():
        def f(v, n=4):
            try:
                return f'{float(v):.{n}f}'
            except Exception:
                return '—'
        lines.append(
            f"| {r.gate} | {int(r.in_pipeline_all)} | {r.g0_cont or '—'} | "
            f"{r.neutral if pd.notna(r.neutral) else '—'} | {f(r.dir_corr)} | "
            f"{f(r.median_b, 5)} | {f(r.same_frac, 3)} | "
            f"{'' if pd.isna(r.folds_same) else int(r.folds_same)} | "
            f"{f(r.ci_lo, 5)} | {r.g2_pass if pd.notna(r.g2_pass) else '—'} | "
            f"{r.g3_pass if pd.notna(r.g3_pass) else '—'} | {f(r.g3_p)} | "
            f"{f(r.score, 3)} | {int(r.in_bank_v2)} | {r.elimination} |")
    lines += ['', '## 2. 月均 p：2024-01 / 2024-02 / 2026-01..09', '',
              '| gate | 2024-01 | 2024-02 | ' + ' | '.join(f'2026-{i:02d}' for i in range(1, 10)) + ' | H1 mean | July | Aug-Sep |',
              '|---|---:|---:|' + '---:|' * 9 + '---:|---:|---:|']
    for gate in GATES:
        row = tab.loc[gate]
        s = summary[summary.gate == gate].iloc[0]
        lines.append(f"| {gate} | {row['2024-01']:.3f} | {row['2024-02']:.3f} | " +
                     ' | '.join(f'{row[f"2026-{i:02d}"]:.3f}' for i in range(1, 10)) +
                     f" | {s.h1_mean:.3f} | {s.july:.3f} | {s.aug_sep_mean:.3f} |")
    lines += [
        '', '## 3. 读数', '',
        '- **2026 H1 误报**：高波动/振幅/离散度/集中度这一组在 2026-01~06 已经长期高位，'
        '不是 7 月才出现；7 月只是继续/加剧。若把它们当作“7 月退潮信号”，H1 会大量误报。',
        '- `GM_newhigh_ratio_level` 不同：H1 均值明显高于 7 月，7 月才是低分位；'
        '`GM_nh60_prem_chg` 也在 7 月最低（H1 内有 5 月 .826 的高位）。',
        '- **2024-01 vs 2026-07**：2024-01 并没有普遍极端；高波动/离散度门控大多在 .4~.9。'
        '**2024-02** 才出现与 2026-07 类似的高位簇（vol/amp/xs_iqr/amt_hhi 近 1）。',
        '- 因此这组高波动门控更像“波动/扩散状态”，会在 2024-02 和 2026-07 同时亮起；'
        '它们不是 7 月特有的退潮证据。真正与 7 月绑定的，是 `GM_newhigh_ratio_level` 与 '
        '`GM_nh60_prem_chg` 的低位。',
        '- G2/G3：11 个高波动/离散度门控除 `GM_xs_disp_ratio` 外均 G2 fail 且 G3 fail；'
        '`GM_xs_disp_ratio` G2 fail 但 G3 pass。`GM_newhigh_ratio_level` G2 fail、G3 pass，'
        '但 bank-v2 未收录（G4/10 名额截断）。',
        '- `GM_nh60_prem_chg` / `GM_sec_top5_persist_level` 不在本地 40-gate pipeline_all；'
        'server bank-v2 中 `GM_nh60_prem_chg` score .8845、`GM_sec_top5_persist_level` score .8012。',
    ]
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
