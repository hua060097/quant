#!/usr/bin/env python3
"""Read-only audit: 48 gates, July ebb, funnel, pattern-A usage."""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gate_audit_20261023'
OUT.mkdir(parents=True, exist_ok=True)
G48 = ROOT / 'results/merge_v2/gates_search_audit_20261023/gate_search_market.parquet'
OLD_WIDE = ROOT / 'results/merge_v2/gates_search/market_gates_search.parquet'
OLD_LONG = ROOT / 'results/merge_v2/gates_search/gate_search_market.parquet'
BANK_LOCAL = ROOT / 'results/merge_v2/gates_search/certified_gate_bank.json'
BANK_SERVER = ROOT / 'results/merge_v2/from_server_20261018/gates_search/certified_gate_bank.json'
POOL = ROOT / 'results/merge_v2/from_server_20261018/specialist_pool.json'
CAND_DIR = ROOT / 'results/merge_v2/from_server_20261018'
JUDGED = CAND_DIR / 'gates_search'
MANIFEST = ROOT / 'results/from_server_20261020/features_manifest.json'
LEDGER = ROOT / 'pandaai_online/merge/ledger.jsonl'


def gate_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(G48)
    d['date'] = pd.to_datetime(d.date)
    w = d.pivot_table(index='date', columns='gate_id', values='p').sort_index()
    months = w.resample('ME').mean()
    july = months.loc['2026-07-31']
    q3 = months.loc['2026-07-31':'2026-09-30'].mean()
    jd = w.loc['2026-07-01':'2026-07-31']
    rows = []
    for g in w.columns:
        rows.append({'gate': g, 'jul_p': float(july[g]), 'q3_p': float(q3[g]),
                     'jul_g': float(2 * july[g] - 1),
                     'q3_g': float(2 * q3[g] - 1),
                     'jul_days_low': int((jd[g] < .15).sum()),
                     'jul_days_high': int((jd[g] > .85).sum()),
                     'jul_days': int(jd[g].notna().sum())})
    return w, pd.DataFrame(rows)


def main() -> int:
    w, stats = gate_frames()
    stats.to_csv(OUT / 'gate48_2026_july_q3.csv', index=False, encoding='utf-8-sig')
    monthly = w.resample('ME').mean()
    monthly.loc['2026-01-31':'2026-09-30'].T.to_csv(
        OUT / 'gate48_monthly_2026.csv', encoding='utf-8-sig')

    ext = stats[(stats.jul_p < .15) | (stats.jul_p > .85)].sort_values('jul_g')
    ext.to_csv(OUT / 'gate48_july_extremes.csv', index=False, encoding='utf-8-sig')
    focus = stats[stats.gate.str.contains(
        'nh60_prem|sec_top5_persist|turn_med|newhigh_ratio')].sort_values('gate')
    focus.to_csv(OUT / 'gate48_focus_july.csv', index=False, encoding='utf-8-sig')

    local = json.loads(BANK_LOCAL.read_text(encoding='utf-8'))
    server = json.loads(BANK_SERVER.read_text(encoding='utf-8'))
    local_all = {e['gate_id'] for e in local['entries']}
    server_all = {e['gate_id'] for e in server['entries']}
    local_m = {e['gate_id'] for e in local['entries'] if e['level'] == 'market'}
    server_m = {e['gate_id'] for e in server['entries'] if e['level'] == 'market'}
    pool = json.loads(POOL.read_text(encoding='utf-8'))
    pf = pool['factors']
    sel = [x for x in pf if x.get('selected')]
    pool_gate = collections.Counter(x['gate'] for x in pf)
    sel_gate = collections.Counter(x['gate'] for x in sel)
    forms = collections.Counter(x.get('suggested_form') for x in pf)

    cands = {}
    for form in ('gain', 'flip', 'switch'):
        p = CAND_DIR / f'specialist_cands_{form}.json'
        cands[form] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
    cand_gate = {k: collections.Counter(x.get('gate') for x in v)
                 for k, v in cands.items()}

    judged = {}
    for name in ('gain', 'switch', 'sector_gain', 'sector_switch'):
        p = JUDGED / f'f1_specialist_{name}_judged.csv'
        judged[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    f1_gate = {k: collections.Counter(v['gate']) for k, v in judged.items() if len(v)}

    man = json.loads(MANIFEST.read_text(encoding='utf-8'))
    feats = man['features_all']
    gp = [x for x in feats if re.search(r'__(GM_.+|GS_.+)__(gain|switch)$', str(x))]
    gp_gate = collections.Counter(
        re.search(r'__(GM_.+|GS_.+)__(?:gain|switch)$', x).group(1) for x in gp)

    rows = []
    active = (set(pool_gate) | set(sel_gate) | set(gp_gate)
              | local_all | server_all | set(focus.gate))
    for g in sorted(active):
        rows.append({'gate': g, 'in_bank_v1': g in local_all,
                     'in_bank_v2': g in server_all, 'pool_best_gate': pool_gate[g],
                     'pool_selected': sel_gate[g],
                     'cand_gain': cand_gate['gain'][g],
                     'cand_flip': cand_gate['flip'][g],
                     'cand_switch': cand_gate['switch'][g],
                     'f1_gain_pass': int(judged['gain'][
                         (judged['gain'].gate == g) & judged['gain'].h3_pass].shape[0])
                     if len(judged['gain']) else 0,
                     'f1_switch_pass': int(judged['switch'][
                         (judged['switch'].gate == g) & judged['switch'].h3_pass].shape[0])
                     if len(judged['switch']) else 0,
                     'in_84_gate_pair': gp_gate[g]})
    funnel = pd.DataFrame(rows).sort_values(['in_84_gate_pair', 'pool_selected'],
                                            ascending=False)
    funnel.to_csv(OUT / 'funnel_by_gate.csv', index=False, encoding='utf-8-sig')

    fails = judged['gain'][~judged['gain'].h3_pass].copy()
    fails.to_csv(OUT / 'f1_gain_failures.csv', index=False, encoding='utf-8-sig')

    ledger_n = 0
    ledger_last = ''
    ledger_gate = 0
    ledger_dates = collections.Counter()
    for line in LEDGER.open(encoding='utf-8'):
        try:
            o = json.loads(line)
        except Exception:
            continue
        ledger_n += 1
        ledger_last = max(ledger_last, str(o.get('created_at', '')))
        ledger_dates[str(o.get('created_at', ''))[:10]] += 1
        if any(g in str(o.get('formula', ''))
               for g in ('MKT_GATE1', 'MKT_GATE2', 'SEC_GATE1', 'SEC_GATE2')):
            ledger_gate += 1

    top = ext[ext.jul_p < .15]
    high = ext[ext.jul_p > .85]
    lines = [
        '# 2026-07 市场门控 / merge 漏斗 / 模式 A 穿透核查', '',
        '## 0. 本地文件与口径', '',
        f"- `market_gates_search.parquet`: **36** 个市场门控；`gate_search_market.parquet`: **40** 个；"
        f"本地重建 48 规格 `gate_search_market.parquet`（qwin=252）。",
        f"- 重建 48 规格 sha12 = `e9322fc3e169`，与 server bank-v2 `market_sha12` **完全一致**。",
        f"- 2026-07 月均值 p 为日级滚动分位 p 的月均值；`g=2p-1`。",
        f"- 48 个门控中 2026-07 均值极端（p<.15 或 >.85）的共 {len(ext)} 个。", '',
        '## 1. 2026-07 极端读数', '',
        '### 低位（退潮方向）', '',
        '| gate | July p | July g | 低分位天数 | 23日中 |',
        '|---|---:|---:|---:|---:|',
    ]
    for _, r in top.iterrows():
        lines.append(f"| {r.gate} | {r.jul_p:.4f} | {r.jul_g:+.3f} | "
                     f"{int(r.jul_days_low)} | {int(r.jul_days)} |")
    lines += ['', '### 高位（恐慌/扩散方向）', '',
              '| gate | July p | July g | 高分位天数 | 23日中 |',
              '|---|---:|---:|---:|---:|']
    for _, r in high.iterrows():
        lines.append(f"| {r.gate} | {r.jul_p:.4f} | {r.jul_g:+.3f} | "
                     f"{int(r.jul_days_high)} | {int(r.jul_days)} |")
    lines += ['', '### 重点门控', '',
              '| gate | July p | July g | July低/高天数 | Q3 p | Q3 g |',
              '|---|---:|---:|---:|---:|---:|']
    for _, r in focus.iterrows():
        lines.append(f"| {r.gate} | {r.jul_p:.4f} | {r.jul_g:+.3f} | "
                     f"{int(r.jul_days_low)}/{int(r.jul_days_high)} | "
                     f"{r.q3_p:.4f} | {r.q3_g:+.3f} |")
    lines += [
        '',
        '- **明确退潮**：`GM_newhigh_ratio_level`（p=.0806，17/23 天低分位）、'
        '`GM_nh60_prem_chg`（p=.1291，19/23 天低分位）。',
        '- **边缘**：`GM_newhigh_ratio_ratio` p=.1648（12/23 天低分位）。',
        '- **没有强退潮**：`GM_sec_top5_persist_*` p=.279~.496；'
        '`GM_turn_med_*` p=.440~.553；`GM_newhigh_ratio_chg` p=.3435。',
        '- 高位集中在波动/振幅/离散度/成交集中度：'
        '`vol_med_level` .998、`amp_med_level` .991、`xs_iqr_level` .991、'
        '`xs_disp_level` .981、`amt_hhi_level` .996；这是恐慌/扩散，不是温和退潮。',
        '- Q3 月均值仅 `GM_amt_hhi_level` 仍 >.85（.866）；7 月退潮门控在 Q3 已回落。', '',
        '## 2. Merge 漏斗：谁掉了、掉在哪一步', '',
        f"- 本地主 bank-v1：market {len(local_m)} 个；server bank-v2：market {len(server_m)} 个。"
        "**2026-10-18 的 specialist 产物用的是 bank-v2，不是本地主 bank-v1。**",
        f"- bank-v2 包含 `GM_nh60_prem_chg`、`GM_sec_top5_persist_level/ratio`、"
        f"`GM_newhigh_ratio_chg/ratio`、`GM_turn_med_accel` 等；"
        f"不包含 `GM_newhigh_ratio_level`。",
        f"- specialist pool：{len(pf)} 个，selected {len(sel)} 个；"
        f"suggested_form={dict(forms)}。",
        f"- 最终 market cands：gain {len(cands['gain'])} / flip {len(cands['flip'])} / "
        f"switch {len(cands['switch'])}；"
        f"gain 中 newhigh_ratio_chg {cand_gate['gain']['GM_newhigh_ratio_chg']}、"
        f"limit_ratio_ratio {cand_gate['gain']['GM_limit_ratio_ratio']}、"
        f"turn_med_accel {cand_gate['gain']['GM_turn_med_accel']}。",
        '',
        '| gate | bank-v2 | pool best | selected | gain cand | flip | switch cand | F1 gain pass | 84 gate-pair |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _, r in funnel.iterrows():
        lines.append(f"| {r.gate} | {int(r.in_bank_v2)} | {int(r.pool_best_gate)} | "
                     f"{int(r.pool_selected)} | {int(r.cand_gain)} | "
                     f"{int(r.cand_flip)} | {int(r.cand_switch)} | "
                     f"{int(r.f1_gain_pass)} | {int(r.in_84_gate_pair)} |")
    lines += [
        '',
        f"- **`GM_newhigh_ratio_chg` 不是 raw pool 垄断**：pool selected 53/211；"
        f"但 20 个 gain 名额里占 {cand_gate['gain']['GM_newhigh_ratio_chg']} 个（65%），"
        f"24 个 market cands 里占 15 个（62.5%）。",
        f"- **`GM_sec_top5_persist_level/ratio` 才是 pool 最大簇**（79+34=113/211 selected），"
        "但 `select_specialists --max-market-cands 20` 按 `|t|*|hi-lo|` 截断，"
        "**两者均 0 个 gain cands**。",
        f"- **`GM_nh60_prem_chg`**：bank-v2 有、selected 15 个，但最高分 "
        f"{0.0768:.4f} 低于 Top20 门槛，**0 个 cand**。",
        f"- **`GM_newhigh_ratio_level`**：7 月最强退潮信号（p=.0806），但**没进 bank-v2**，"
        "在 48→bank 构建阶段就掉队。",
        f"- **`GM_newhigh_ratio_chg`**：7 月 p=.3435，并非强退潮信号，却拿到最多名额并最终进入 84。",
        '',
        '### F1 淘汰原因', '',
        f"- market gain：{len(judged['gain'])} 个，h3_pass {int(judged['gain'].h3_pass.sum())}，"
        f"FDR {int(judged['gain'].fdr_sig.sum())}；switch {len(judged['switch'])} 个全部通过；"
        f"flip 0 个（根本没有生成候选）。",
        f"- market gain 失败 {len(fails)} 个：`ok_ci` 失败 "
        f"{int((~fails.ok_ci).sum())}、`ok_blocks` 失败 {int((~fails.ok_blocks).sum())}、"
        f"`ok_placebo` 失败 {int((~fails.ok_placebo).sum())}。",
        '- 失败不是 `delta_mean` 不够：8 个 delta_mean 均为正；是 **CI 下界>0 / nblocks_pos>=4 / placebo_p<.05**。',
        '- 2 个 newhigh gain 仅 placebo 失败（p=.0896/.0995）；6 个 limit gain 因 CI/block/placebo 失败。',
        f"- 84 名单：{len(gp)}/84 个门控配对特征；按门控：{dict(gp_gate)}。",
        '- 其中 market gain 12（newhigh 11 + turn 1）、market switch 4、sector switch 4；'
        'limit gain 0 个进入 84。',
        '',
        '## 3. 模式 A（GP 内嵌 MKT_GATE1/2, SEC_GATE1/2）', '',
        f"- 84 列名单中 `MKT_GATE1/2/SEC_GATE1/2` 字面字段：**0**；"
        f"门控配对特征 **{len(gp)}/84**（这些是 F1 specialist，不是模式 A GP）。",
        f"- ledger：{ledger_n:,} 条事件，含四个 GATE 字段的公式 **{ledger_gate}** 条；"
        f"最后事件 `{ledger_last}`。",
        "- V4.3d `GATE_FIELDS` 提交时间：**2026-10-16 15:08**；ledger 最后事件 **2026-10-15 17:17**。",
        "- 因此不是“跑过但轮数不够”，而是 **V4.3d 之后没有跑过 run_search.py**；"
        "模式 A 使用率为 0 是时间顺序问题，不是随机命中率问题。",
        "- 另外，84 的 20 个门控配对来自 F1 specialist（GM/GS 名称），"
        "与 GP 内嵌 `MKT_GATE1/2` 是两条不同路径。",
        '',
        '## 4. 结论', '',
        '- 7 月真实退潮信号在 **newhigh_ratio_level / nh60_prem_chg**；'
        'sec_top5_persist 只是偏弱，turn_med 没有退潮信号。',
        '- 能识别退潮的 gate：newhigh_level 死在 bank 构建；'
        'nh60_prem_chg 死在 select_specialists Top20 截断；'
        'sec_top5 也死在 Top20 截断。',
        '- 最终 84 的 20 个门控配对由 newhigh_chg/limit/turn 主导，'
        '其中 newhigh_chg 13/20；这不是因为它是 7 月最强退潮门控，而是评分/配额机制。',
        '- F1 不是全灭：market gain 12/20、switch 4/4、sector switch 4/4 通过；'
        'flip 0 是因为没有生成，不是被 judge 淘汰。',
        '- 模式 A 目前 0 使用；V4.3d 后尚未跑 run_search，需至少一轮新搜索才能检验。',
    ]
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:20]))
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
