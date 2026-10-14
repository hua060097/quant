"""会话报告：短名单 JSON + 台账 → Markdown（Experiment Monitor/Analytic Panel 的本地版）。

用法：
  python report.py out/shortlist_xxx.json [--out report.md]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import _paths  # noqa: F401
from field_catalog import formula_uses_local
from ledger import Ledger


def _cell(value, fmt: str = '') -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        if value != value:
            return '-'
        return format(value, fmt) if fmt else f'{value:.4f}'
    return str(value)


def build_report(shortlist_path: Path | str, out_path: Path | str | None = None,
                 ledger_path: Path | str | None = None) -> Path:
    payload = json.loads(Path(shortlist_path).read_text(encoding='utf-8'))
    entries = payload.get('shortlist', [])
    labels = payload.get('labels') or ['vwap']
    primary = payload.get('primary_label') or labels[0]
    profile = payload.get('profile') or ('local-vwap' if primary == 'vwap'
                                         else 'competition-c2c')
    # 运气门槛（T3，2026-09-25）：n_tested 次试验下纯噪声最好成绩的期望 |RankIC|
    # z ≈ sqrt(2 ln n)；单截面 se ≈ 1/√平均截面股票数；均值 se = se/√截面数
    import math as _math
    n_tested = int((payload.get('n_tested_by_label') or {}).get(
        primary, payload.get('n_tested') or 0))
    luck_threshold = None
    if n_tested > 50 and payload.get('ctx_rows') and payload.get('ctx_sections'):
        _z = _math.sqrt(2.0 * _math.log(n_tested))
        _avg_n = float(payload['ctx_rows']) / max(1, int(payload['ctx_sections']))
        _se_mean = (1.0 / _math.sqrt(max(_avg_n, 2.0))) / _math.sqrt(int(payload['ctx_sections']))
        luck_threshold = _z * _se_mean
    lines = [
        f"# 因子搜索会话报告 {payload.get('generated_at', '')}",
        '',
        f"- 短名单文件：`{Path(shortlist_path).name}`，策略 {payload.get('strategy', '-')}，"
        f"标签 {labels}，种子 {payload.get('seed')}，"
        f"周期 {payload.get('cycle')}，切分 {payload.get('cut')}",
        f"- profile `{profile}`，主标签 `{primary}`，primary metric `{payload.get('primary_search_metric', primary)}`；label semantics: "
        f"{(payload.get('label_semantics') or {}).get(primary, primary)}",
        f"- 生成 {payload.get('n_generated')}，快速评估 {payload.get('n_fast', '-')}，"
        f"全量评估 {payload.get('n_full', '-')}，有效 {payload.get('n_pool_ok')}，"
        f"换手软惩罚 T_ref={payload.get('turnover_ref')}，"
        f"OOS-SiA 触发阈值 {payload.get('oos_sia_threshold')}",
    ]
    if payload.get('stats'):
        lines.append(f"- 快速层统计：{payload['stats']}")
    dual = payload.get('dual_gate')
    if dual:
        lines.append(f"- 显式双标签门：primary={primary}，secondary={dual['secondary']} SiA≥{dual['min_sia']} 且 "
                     f"比值≥{dual['min_ratio']}；全量拒绝 {payload.get('n_dual_rejected', 0)} 个")
    else:
        lines.append('- 附加标签只作独立诊断/榜单，不对主 profile 设必过门；'
                     'C2C 平台兼容性只在 export/upload 时检查。')
    if ledger_path:
        stats = Ledger(ledger_path).stats()
        lines.append(f"- 台账：{stats['events']}，全局假设数 {stats['tested']}，"
                     f"多重检验阈值 p<{stats['p_threshold']:.4g}")
    if luck_threshold is not None:
        lines.append(f"- **运气门槛（多重检验紧缩）**：n_tested={n_tested}，纯噪声最好成绩期望 "
                     f"|RankIC|≈{luck_threshold:.4f}；显著列 ✓=|RankIC|≥门槛，"
                     f"⚠运气区=与运气不可区分（建议结合 holdout/栖息地读数人工复核）")
    decay_gate = payload.get('max_decay_gate')
    if decay_gate is not None:
        lines.append(f"- decay 门槛：后半段/前半段 RankIC 比 ≥{decay_gate} 才入短名单"
                     f"（<1 为衰减趋势；本列为诊断参考）")
    cols = ['候选', '来源', 'dir', '本地/C2C可上传', '综合分', '无惩罚分',
            'train target', 'holdout target', 'raw SiA', 'RankIC', 'ICIR(P)', 'IC胜率', 'RankICIR', 'decay', '显著',
            'dom', '换手', 'T_month', '净超额', 'C_proxy', '稳健']
    cols += ['公式']
    lines += ['', f'## 主短名单（{primary}，score 为搜索排序指标而非统计资格）', '',
              '| ' + ' | '.join(cols) + ' |',
              '| ' + ' | '.join('---' for _ in cols) + ' |']
    for r in entries:
        holdout = r.get('holdout') or {}
        train = r.get('train') or {}
        main_t = (r.get('labels') or {}).get(primary) or {}
        dec = ((r.get('decay') or {}).get(primary) or {}).get('decay_ratio')
        ric_val = main_t.get('rank_ic')
        sig = ''
        if luck_threshold is not None and ric_val is not None:
            sig = '✓' if abs(float(ric_val)) >= luck_threshold else '⚠运气区'
        _fd = r.get('field_dom') or {}
        dom_txt = (f"{_fd['max']:.2f}@{_fd['field']}"
                   if _fd.get('max') is not None else '-')
        capability = ('local-only' if r.get('local_only') else
                      ('C2C✓' if r.get('competition_uploadable') else 'C2C?'))
        cells = [str(r.get('name', '-')), str(r.get('source', '-')),
                 str(r.get('direction', '-')),
                 capability,
                 '%.4f' % float(r.get('score') or 0.0),
                 '%.4f' % float(r.get('score_nopen') or 0.0),
                 _cell(r.get('train_search_value', train.get('sia')), '.4f'),
                 _cell(r.get('holdout_search_value', holdout.get('sia')), '.4f'),
                 _cell(r.get('sia'), '.4f'),
                 _cell(ric_val, '+.4f'),
                 _cell(main_t.get('ic_ir'), '+.3f'),
                 _cell(main_t.get('win'), '.1%'),
                 _cell(main_t.get('rank_ic_ir'), '+.3f'),
                 _cell(dec, '.2f'),
                 sig,
                 dom_txt,
                 _cell(r.get('turnover'), '.1%'),
                 _cell(r.get('turnover_month'), '.2f'),
                 _cell(r.get('net'), '+.2%'),
                 _cell(r.get('c_proxy'), '.3f'),
                 _cell(r.get('stability'), '.3f')]
        cells.append(f"`{r.get('formula', '')}`")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## 机制解释', '']
    for r in entries:
        lines.append(f"- **{r.get('name')}**：{r.get('explain', '-')}")

    vwap_rows = payload.get('vwap_top') or []
    ctx_top = payload.get('ctx_top') or []
    if vwap_rows:
        vwap_role = '主榜' if primary == 'vwap' else '附加诊断榜'
        lines += ['', f'## 本地 VWAP 榜（{vwap_role}；formula-only label）', '',
                  '| 候选 | 后端能力 | VWAP RankSiA | VWAP SiA | VWAP RankIC | VWAP RankICIR | '
                  'VWAP IC | VWAP mono | turnover | C2C SiA | 公式 |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in vwap_rows:
            vt = (r.get('labels') or {}).get('vwap') or {}
            ct = (r.get('labels') or {}).get('c2c') or {}
            rank_sia = None
            if all(vt.get(k) is not None for k in ('rank_ic', 'rank_ic_ir', 'rank_ic_win')) \
                    and all(vt.get(k) == vt.get(k) for k in ('rank_ic', 'rank_ic_ir', 'rank_ic_win')):
                rank_sia = abs(vt['rank_ic']) * abs(vt['rank_ic_ir']) * abs(vt['rank_ic_win'])
            capability = ('local-only' if r.get('local_only') else
                          ('C2C✓' if r.get('competition_uploadable') else 'C2C?'))
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |' % (
                r.get('name', '-'), capability, _cell(rank_sia, '.4f'),
                _cell(vt.get('sia'), '.4f'), _cell(vt.get('rank_ic'), '.4f'),
                _cell(vt.get('rank_ic_ir'), '.3f'), _cell(vt.get('ic'), '.4f'),
                _cell(vt.get('mono'), '.3f'), _cell(vt.get('turnover'), '.1%'),
                _cell(ct.get('sia'), '.4f'), r.get('formula', '')))

    c2c_top = payload.get('c2c_top') or []
    competition_top = payload.get('competition_top') or []
    if c2c_top:
        lines += ['', '## C2C 竞赛扩展（本地搜索结果；上传子集经过能力过滤）', '',
                  f'- 本地 C2C 评估候选 {len(c2c_top)}；当前可上传子集 {len(competition_top)}。',
                  '- 仅上传子集需满足竞赛字段/算子合同；local-only 候选继续保留在本地结果。', '',
                  '| 候选 | C2C SiA | C2C可上传 | 公式 |', '| --- | --- | --- | --- |']
        for r in c2c_top:
            ct = (r.get('labels') or {}).get('c2c') or {}
            lines.append('| %s | %s | %s | `%s` |' % (
                r.get('name', '-'), _cell(ct.get('sia'), '.4f'),
                'yes' if r.get('competition_uploadable') else 'no', r.get('formula', '')))

    # 机制解释是公式级属性，与目标标签/profile 无关。
    if vwap_rows:
        lines += ['', '## VWAP 榜机制解释', '']
        for r in vwap_rows:
            mark = '（本地上下文）' if formula_uses_local(r.get('formula', '')) else ''
            lines.append(f"- **{r.get('name')}**{mark}：{r.get('explain', '-')}")

    # 采样稳健性（2026-09-24）：cycle=3 锚点 offset 0/1/2 三组错位截面 RankIC 一致性
    od_rows = [r for r in entries if r.get('offset_diag')]
    if od_rows:
        lines += ['', '## 采样稳健性（RankIC @ offset 0/1/2，方向按各标签 offset-0 冻结）', '',
                  '三组同号且接近 = 结论对调仓起点稳健；分歧大 = 因子在挑调仓时点（危险信号）。'
                  '⚠ = 方向冻结后三组未全正（含 Pearson/Spearman 方向冲突的情形）。', '',
                  '| 候选 | 标签 | RankIC@0 | RankIC@1 | RankIC@2 | spread | 三组同号 |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for r in od_rows:
            od = r['offset_diag']
            for lab in (payload.get('labels') or ['vwap']):
                d = od.get(lab) or {}
                if not d:
                    continue
                rk = d.get('rank_ics') or {}

                def _g(o):
                    v = rk.get(o, rk.get(str(o)))
                    return _cell(v, '+.4f')

                lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
                    r.get('name', '-'), lab, _g(0), _g(1), _g(2),
                    _cell(d.get('spread'), '.4f'),
                    '✓' if d.get('sign_consistent') else '⚠'))

    # 条件 RankIC（2026-09-24）：热点子集（栖息地）+ 市场状态，与主管线 COND-IC 同口径
    ci_rows = [r for r in entries if r.get('cond_ic')]
    if ci_rows:
        lines += ['', '## 条件 RankIC（热点子集栖息地 + 市场状态；主管线 COND-IC 同口径）', '',
                  'hot* = 实际交易票内（涨幅分位×换手分位）的子集 RankIC——全市场 IC 高但 hot IC 低'
                  ' = 栖息地稀释；mkt_* = 市场涨/跌日 RankIC。', '',
                  '| 候选 | 标签 | hot5_q80 | hot2_q80 | hot5_q90 | hot5_cold | mkt_up | mkt_dn |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in ci_rows:
            ci = r['cond_ic']
            for lab in (payload.get('labels') or ['vwap']):
                d = ci.get(lab) or {}
                if not d:
                    continue

                def _g(tag):
                    return _cell(d.get(f'cond_{tag}_rank_ic_mean'), '+.4f')

                lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                    r.get('name', '-'), lab, _g('hot5_q80'), _g('hot2_q80'),
                    _g('hot5_q90'), _g('hot5_cold'), _g('mkt_up'), _g('mkt_dn')))

    greedy = payload.get('greedy')
    if greedy:
        lines += ['', '## 贪婪 ΔComb 选择（池阶段工具）', '',
                  f"- 现池基准 Comb = {greedy.get('base_comb'):.4f}；"
                  f"选择后 = {greedy.get('final_comb'):.4f}；"
                  f"建议加入：{greedy.get('selected')}"]
        for step in greedy.get('steps', []):
            lines.append(f"  - {step['candidate']}：ΔComb +{step['gain']:.4f}，"
                         f"NC {step['NC']:.4f}，换手 {step['Turnover']:.2%}")
    rej = payload.get('dual_rejected') or []
    if rej:
        secondary = (payload.get('dual_gate') or {}).get('secondary')
        lines += ['', f'## 双标签未过清单（{primary} × {secondary}）', '']
        for r in rej[:12]:
            lines.append(f"- {r['name']}：primary metric {r['sia']:.4f} → "
                         f"secondary {float(r.get('dual_sia') or 0.0):.4f}"
                         f"（比值 {float(r.get('dual_ratio') or 0.0):.2f}）")
    tips = ['- 本报告全部为本地零算力结果；平台探针需显式批准后按 export 命令执行。',
            '- 主 profile 与 label semantics 见报告头；附加标签不自动成为必过筛选门。',
            '- local-only/行业上下文因子可进入本地 VWAP 搜索；C2C 兼容性只影响竞赛导出/上传。',
            '- 当前 merge `vwap` 标签为公式型，未应用 `daily_atr_label` 的可交易性掩码；不要与 masked label 结果混比。']
    lines += ['', '## 决策提示', ''] + tips + ['']
    out = Path(out_path) if out_path else Path(shortlist_path).with_suffix('.report.md')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def main() -> None:
    import argparse

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='短名单 JSON → Markdown 报告')
    ap.add_argument('shortlist')
    ap.add_argument('--out', default='')
    ap.add_argument('--ledger', default=str(Ledger().path))
    args = ap.parse_args()
    print(build_report(args.shortlist, args.out or None, ledger_path=args.ledger))


if __name__ == '__main__':
    main()
