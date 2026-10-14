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
    labels = payload.get('labels') or ['c2c']
    # 运气门槛（T3，2026-09-25）：n_tested 次试验下纯噪声最好成绩的期望 |RankIC|
    # z ≈ sqrt(2 ln n)；单截面 se ≈ 1/√平均截面股票数；均值 se = se/√截面数
    import math as _math
    n_tested = int(payload.get('n_tested') or 0)
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
        f"- 生成 {payload.get('n_generated')}，快速评估 {payload.get('n_fast', '-')}，"
        f"全量评估 {payload.get('n_full', '-')}，有效 {payload.get('n_pool_ok')}，"
        f"换手软惩罚 T_ref={payload.get('turnover_ref')}，"
        f"OOS-SiA 触发阈值 {payload.get('oos_sia_threshold')}",
    ]
    if payload.get('stats'):
        lines.append(f"- 快速层统计：{payload['stats']}")
    dual = payload.get('dual_gate')
    if dual:
        lines.append(f"- （历史口径）双主必过：{dual['secondary']} SiA≥{dual['min_sia']} 且 "
                     f"比值≥{dual['min_ratio']}；全量拒绝 {payload.get('n_dual_rejected', 0)} 个")
    else:
        lines.append('- vwap 仅诊断（2026-09-16 起不再必过）：主分按 c2c；'
                     'vwap 名单见下方/`vwap_top`')
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
    cols = ['候选', '来源', 'dir', '综合分', '无惩罚分', 'train SiA', 'holdout SiA',
            'RankIC', 'ICIR(P)', 'IC胜率', 'RankICIR', 'decay', '显著', 'dom', '换手',
            'T_month', '净超额', 'C_proxy', '稳健', 'vwap SiA']
    cols += ['公式']
    lines += ['', '## 短名单（c2c 主分 = SiA × 换手软惩罚 × 复杂度）', '',
              '| ' + ' | '.join(cols) + ' |',
              '| ' + ' | '.join('---' for _ in cols) + ' |']
    for r in entries:
        holdout = r.get('holdout') or {}
        ct = (r.get('labels') or {}).get('c2c') or {}
        vwap_t = (r.get('labels') or {}).get('vwap') or {}
        dec = ((r.get('decay') or {}).get('c2c') or {}).get('decay_ratio')
        ric_val = ct.get('rank_ic')
        sig = ''
        if luck_threshold is not None and ric_val is not None:
            sig = '✓' if abs(float(ric_val)) >= luck_threshold else '⚠运气区'
        _fd = r.get('field_dom') or {}
        dom_txt = (f"{_fd['max']:.2f}@{_fd['field']}"
                   if _fd.get('max') is not None else '-')
        cells = [str(r.get('name', '-')), str(r.get('source', '-')),
                 str(r.get('direction', '-')),
                 '%.4f' % float(r.get('score') or 0.0),
                 '%.4f' % float(r.get('score_nopen') or 0.0),
                 '%.4f' % float(r.get('sia') or 0.0),
                 _cell(holdout.get('sia'), '.4f'),
                 _cell(ric_val, '+.4f'),
                 _cell(ct.get('ic_ir'), '+.3f'),
                 _cell(ct.get('win'), '.1%'),
                 _cell(ct.get('rank_ic_ir'), '+.3f'),
                 _cell(dec, '.2f'),
                 sig,
                 dom_txt,
                 _cell(r.get('turnover'), '.1%'),
                 _cell(r.get('turnover_month'), '.2f'),
                 _cell(r.get('net'), '+.2%'),
                 _cell(r.get('c_proxy'), '.3f'),
                 _cell(r.get('stability'), '.3f'),
                 _cell(vwap_t.get('sia'), '.4f')]
        cells.append(f"`{r.get('formula', '')}`")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## 机制解释', '']
    for r in entries:
        lines.append(f"- **{r.get('name')}**：{r.get('explain', '-')}")

    vwap_rows = payload.get('vwap_top') or []
    ctx_top = payload.get('ctx_top') or []
    if vwap_rows:
        lines += ['', '## vwap top（非比赛口径，供 LightGBM 项目查询）', '',
                  '| 候选 | 本地 | vwap rank_sia | vwap SiA | vwap/c2c | vwap RankIC | vwap RankICIR | '
                  'vwap IC | vwap mono | vwap 换手 | c2c SiA | 公式 |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in vwap_rows:
            vt = (r.get('labels') or {}).get('vwap') or {}
            sia = r.get('sia') or 0.0
            vs = vt.get('sia')
            ratio = (vs / sia) if (vs is not None and sia > 0) else None
            rank_sia = None
            if all(vt.get(k) is not None for k in ('rank_ic', 'rank_ic_ir', 'rank_ic_win')) \
                    and all(vt.get(k) == vt.get(k) for k in ('rank_ic', 'rank_ic_ir', 'rank_ic_win')):
                rank_sia = (abs(vt['rank_ic']) * abs(vt['rank_ic_ir']) * abs(vt['rank_ic_win']))
            local_mark = 'M14*' if formula_uses_local(r.get('formula', '')) else ''
            lines.append('| %s | %s | %s | %.4f | %s | %s | %s | %s | %s | %s | %.4f | `%s` |' % (
                r.get('name', '-'), local_mark, _cell(rank_sia, '.4f'), vs or 0.0,
                f'{ratio:.2f}' if ratio is not None else '-',
                _cell(vt.get('rank_ic'), '.4f'), _cell(vt.get('rank_ic_ir'), '.3f'),
                _cell(vt.get('ic'), '.4f'), _cell(vt.get('mono'), '.3f'),
                _cell(vt.get('turnover'), '.1%'),
                sia, r.get('formula', '')))

    # 机制解释（2026-09-24 补：explain 是公式级属性，c2c/vwap 同公式同解释；
    # vwap 榜此前只有表格无解释，补一节）
    if vwap_rows:
        lines += ['', '## vwap 榜机制解释（解释的是因子机制本身，两口径同一公式）', '']
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
            for lab in (payload.get('labels') or ['c2c']):
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
            for lab in (payload.get('labels') or ['c2c']):
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
        lines += ['', '## vwap 未过清单（已不拦截，仅诊断）', '']
        for r in rej[:12]:
            lines.append(f"- {r['name']}：c2c SiA {r['sia']:.4f} → "
                         f"vwap {float(r.get('dual_sia') or 0.0):.4f}"
                         f"（比值 {float(r.get('dual_ratio') or 0.0):.2f}）")
    tips = ['- 本报告全部为本地零算力结果；平台探针需显式批准后按 export 命令执行。',
            '- 单因子阶段口径：官方 A 代理（|RankIC|×|ICIR|×胜率(|IC|>0.02)）+ 换手软惩罚；'
            'B/C 为前瞻项、提交前不可观测。',
            '- vwap 名单仅供自用（LightGBM），不参与比赛选种。']
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
