"""提案入口：外部/LLM/人工公式 → parse → validate → lint → 小样本语义冒烟 → 去重 → 台账。

对应 Alpha-GPT 的 Thought Decompiler（Algorithm 1）本地化：LLM/人工产出不再直接进搜索，
必须先过验证闭环；退回结构化错误信息，供提案方修正重试（retry 由调用方驱动）。

输入 JSONL（每行）：
  {"hypothesis": "...", "formula": "RANK(...)", "direction": 0|1（可选）,
   "tags": ["liquidity"], "source": "llm|human|paper", "family": "..."}

用法：
  python pandaai_online/merge/intake.py proposals.jsonl [--limit 20] [--no-smoke] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import Evaluator, load_panel
from grammar import expand, expression_hash, render, validate, walk
from ledger import Ledger
from lint import has_errors, lint
from parser import ParseError, parse_formula

SAMPLE_STOCKS = 300
SAMPLE_DAYS = 900
SPARSE_UNIQUE = 100
# 外部公式含展开的 RET1/LOGC 虚拟链，深度预算放宽；搜索种群仍用 MAX_DEPTH=6。
INTAKE_MAX_DEPTH = 12

# 假设↔公式一致性机检（2026-09-24，DeepSeek 条件认同的 lint 级版本；LLM 语义打分属 P2）：
# 假设关键词暗示的字段/算子未出现在公式中 → warning（不拒绝，提请提案方自查）。
_HYP_FIELD_HINTS = [
    (('换手', '流动性', '成交额', '量能', 'liquidity', 'turnover'),
     {'TURNOVER', 'VOLUME', 'AMOUNT'}),
    (('市值', '规模', '小盘', '大盘', 'size'), {'MARKET_CAP'}),
    (('量价', '背离', '量能配合'), {'VOLUME', 'CLOSE', 'TURNOVER', 'AMOUNT'}),
]
_HYP_OP_HINTS = [
    (('偏度', '尾部', '崩盘', 'skew'), {'TS_SKEW'}),
    (('峰度', 'kurt'), {'TS_KURT'}),
    (('波动', 'volatility', '低波'), {'STDDEV'}),
]


def hypothesis_field_check(hypothesis: str, node) -> list[dict]:
    """假设关键词 ↔ 公式字段/算子一致性机检；返回 warning 级 issues。"""
    issues: list[dict] = []
    text = str(hypothesis or '')
    if not text.strip() or node is None:
        return issues
    ex = expand(node)
    fields = {n.name for _, n in walk(ex) if n.op == 'FIELD'}
    ops = {n.op for _, n in walk(ex)}
    low = text.lower()
    for keys, need in _HYP_FIELD_HINTS:
        if any(k in low for k in keys) and not (fields & need):
            issues.append({'level': 'warning', 'code': 'hypothesis-field',
                           'message': f'假设提到 {"|".join(keys[:2])} 但公式未含相关字段'
                                      f'（{"/".join(sorted(need))}）；请确认假设与公式一致'})
    for keys, need in _HYP_OP_HINTS:
        if any(k in low for k in keys) and not (ops & need):
            issues.append({'level': 'warning', 'code': 'hypothesis-field',
                           'message': f'假设提到 {"|".join(keys[:2])} 但公式未含相应算子'
                                      f'（{"/".join(sorted(need))}）；请确认假设与公式一致'})
    return issues


def smoke_frame(n_stocks: int = SAMPLE_STOCKS, days: int = SAMPLE_DAYS,
                seed: int = 0) -> pd.DataFrame:
    frame = load_panel()
    codes = sorted(frame['stock_code'].unique().tolist())
    rng = random.Random(seed)
    pick = set(rng.sample(codes, min(n_stocks, len(codes))))
    sub = frame[frame['stock_code'].isin(pick)].copy()
    last = sub['date'].max()
    sub = sub[sub['date'] >= last - pd.Timedelta(days=days)]
    return sub.sort_values(['stock_code', 'date']).reset_index(drop=True)


def sparse_unique_median(values: np.ndarray, dates: np.ndarray, n_dates: int = 21) -> float:
    """末 n_dates 个截面的独立值中位数（稀疏并列检查；F-B04/B08 教训的共享实现）。

    intake 冒烟与 run_search 生成期检查共用：阈值 SPARSE_UNIQUE、口径一致。
    """
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    v = values[finite]
    d = np.asarray(dates)[finite]
    counts = [len(np.unique(v[d == x])) for x in np.unique(d)[-n_dates:]]
    return float(np.median(counts)) if counts else 0.0


def _smoke_stats(ev: Evaluator, node) -> dict:
    values = ev.values(node)
    finite = np.isfinite(values)
    coverage = float(finite.mean()) if len(values) else 0.0
    if not finite.any():
        return {'coverage': coverage, 'nonzero': 0.0, 'unique_median': 0.0, 'std': 0.0}
    v = values[finite]
    return {'coverage': coverage,
            'nonzero': float(np.mean(v != 0)),
            'unique_median': sparse_unique_median(values, ev.frame['date'].to_numpy()),
            'std': float(np.std(v))}


def check_candidate(formula: str, direction: int | None = None,
                    smoke: pd.DataFrame | None = None,
                    allow_documented: bool = True, hypothesis: str = '') -> dict:
    out = {'formula': formula, 'direction': direction, 'status': 'ok', 'issues': []}
    try:
        node = parse_formula(formula)
    except ParseError as exc:
        out.update({'status': 'reject', 'node': None, 'canonical': formula,
                    'issues': [{'level': 'error', 'code': 'parse', 'message': str(exc)}]})
        return out
    ok, msg = validate(node, allow_documented=allow_documented, strict_const=False,
                       max_depth=INTAKE_MAX_DEPTH)
    issues = list(lint(node))
    issues.extend(hypothesis_field_check(hypothesis, node))
    if not ok:
        issues.insert(0, {'level': 'error', 'code': 'validate', 'message': msg})
    out.update({'node': node, 'canonical': render(node),
                'expression_hash': expression_hash(node)})
    if direction not in (None, 0, 1):
        issues.append({'level': 'error', 'code': 'direction',
                       'message': f'direction 必须为 0/1，收到 {direction!r}'})
    if not has_errors(issues) and smoke is not None:
        try:
            stats = _smoke_stats(Evaluator(smoke), node)
            out['stats'] = stats
            if stats['coverage'] < 0.3:
                issues.append({'level': 'error', 'code': 'coverage',
                               'message': f"覆盖率过低 {stats['coverage']:.2f}"})
            elif stats['coverage'] < 0.6:
                issues.append({'level': 'warning', 'code': 'coverage',
                               'message': f"覆盖率偏低 {stats['coverage']:.2f}"})
            if stats['std'] == 0:
                issues.append({'level': 'error', 'code': 'constant',
                               'message': '因子在冒烟样本上为常数'})
            if 0 < stats['unique_median'] < SPARSE_UNIQUE:
                issues.append({'level': 'warning', 'code': 'sparse-tie',
                               'message': f"截面独立值中位数 {stats['unique_median']:.0f}"
                                          f" < {SPARSE_UNIQUE}，稀疏并列换手风险"
                                          '（参考 F-B04/B08：本地1-2% vs 平台88%）'})
        except Exception as exc:
            issues.append({'level': 'error', 'code': 'runtime',
                           'message': f'冒烟求值失败：{str(exc)[:200]}'})
    if has_errors(issues):
        out['status'] = 'reject'
    elif any(i['level'] == 'warning' for i in issues):
        out['status'] = 'warn'
    out['issues'] = issues
    return out


def intake_file(path: Path | str, ledger: Ledger | None = None,
                smoke: pd.DataFrame | None = None, limit: int | None = None) -> dict:
    ledger = ledger or Ledger()
    known = ledger.hashes()
    lines = [x for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    rows = [json.loads(x) for x in lines]
    if limit:
        rows = rows[:limit]
    counts = {'ok': 0, 'warn': 0, 'reject': 0, 'duplicate': 0}
    results = []
    for row in rows:
        formula = str(row.get('formula', '')).strip()
        hypothesis = str(row.get('hypothesis', ''))
        direction = row.get('direction')
        check = check_candidate(formula, direction=direction, smoke=smoke,
                                hypothesis=hypothesis)
        expr_hash = check.get('expression_hash')
        if expr_hash and expr_hash in known:
            check['status'] = 'duplicate'
            check['issues'].append({'level': 'info', 'code': 'duplicate',
                                    'message': '台账中已存在同公式候选'})
        elif expr_hash:
            known.add(expr_hash)
        counts[check['status']] = counts.get(check['status'], 0) + 1
        check['hypothesis'] = hypothesis
        if check['status'] in ('ok', 'warn') and check.get('node') is not None:
            parsed_direction = int(direction) if direction in (0, 1) else 1
            ledger.log('proposed', check['node'], parsed_direction,
                       family=str(row.get('family', 'intake')), hypothesis=hypothesis,
                       data={'source': row.get('source', ''),
                             'tags': row.get('tags', []),
                             'issues': check['issues'],
                             'canonical': check['canonical'],
                             'direction_pending': direction not in (0, 1)})
        else:
            ledger.log_raw('rejected' if check['status'] == 'reject' else 'duplicate',
                           formula,
                           direction=direction if direction in (0, 1) else None,
                           family=str(row.get('family', 'intake')), hypothesis=hypothesis,
                           data={'issues': check['issues'], 'source': row.get('source', '')})
        results.append(check)
    return {'path': str(path), 'counts': counts, 'results': results}


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='提案入口：验证 + 冒烟 + 去重 + 台账')
    ap.add_argument('file', type=Path)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ledger', default=str(Ledger().path))
    ap.add_argument('--no-smoke', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    smoke = None if args.no_smoke else smoke_frame()
    report = intake_file(args.file, Ledger(args.ledger), smoke=smoke,
                         limit=args.limit or None)
    print(f"intake {report['path']}: {report['counts']}")
    for item in report['results']:
        mark = {'ok': 'OK  ', 'warn': 'WARN', 'reject': 'FAIL', 'duplicate': 'DUP '}[item['status']]
        print(f"  {mark} {item['canonical'][:96]}")
        for issue in item['issues']:
            print(f"        [{issue['level']}] {issue['code']}: {issue['message']}")
    if args.out:
        payload = {**report, 'results': [{k: v for k, v in r.items() if k != 'node'}
                                         for r in report['results']]}
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                             default=str), encoding='utf-8')
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
