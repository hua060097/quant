# -*- coding: utf-8 -*-
"""个股线快评精度基准（MERGE_V2_PLAN §20.14；Codex 设计）。

**要回答的问题**：快评（抽样 2500 只 + 26 个月尾窗）会不会**系统性淘汰**本该进全量的好候选？

快评同时改变了两件事，单看一个相关系数无法分辨原因，故做**四组对照**：

| 组 | 截面 | 时间窗 | 作用 |
|---|---|---|---|
| A | 抽样 | 24 个月尾窗 | = 现行快评（复现真实筛选行为） |
| B | 全量 | 24 个月尾窗 | 单独隔离**截面抽样**效应 |
| C | 抽样 | 完整窗 | 单独隔离**窗口缩短**效应 |
| D | 全量 | 完整窗 | 全量评估基准 |

**五类指标**（对每个对照组 vs D）：
  1. SiA 符号一致率      2. 排序相关性（Spearman）   3. Top-K 命中率
  4. **错误淘汰率**（最重要）   5. SiA 偏差与误差分布

并**单独**做 `--full-cap` **截断损失**分析（影子评估）：
把"快评前 cap 名"与"全量真实前 cap 名"比对，统计重合度与**真实优秀候选被截断的比例**，
并按家族分组，避免总体指标掩盖某一类被系统性淘汰。

**候选集必须包含"会被快评淘汰的"候选**（`--band low` 段），否则错误淘汰率恒为 0——
这正是"不能只挑成功候选来验证快评"（Codex）的落实。

**纪律**：本模块**只测量、不改任何生产参数**。基准完成前不得动快评阈值 / `--full-cap` / 筛选流程。

用法：
    python bench_fast_precision.py --n 40 --seed 11
    python bench_fast_precision.py --n 8 --seed 11 --out /tmp/bench   # 小样试跑
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _paths  # noqa: F401
import windows as W
from evaluator import Evaluator, load_panel
from fitness import (DataContext, START, build_fast_frame, pick_stocks,
                     rank_score, sia_of, tier1)
from grammar import generate_population, render
from ledger import Ledger
from parser import parse_formula

RESULTS = HERE.parent.parent / 'results' / 'merge_v2'
OUT_DIR = RESULTS / 'bench_fast'

#: 快评的产线参数（与 run_search 默认一致；基准必须复现真实行为）
FAST_KEEP_SIA = 0.0012
FAST_TOP_FRAC = 0.5
FULL_CAP = 60


# ---------------------------------------------------------------- 候选集
def rows_from_ledger(ledger: Ledger, band: str, n: int, rng: random.Random) -> list[dict]:
    """从台账取某 SiA 段的公式。band ∈ {high, mid, low, full}。

    分段依据快评规则：`high` ≥4e-3（会触发 holdout）、`mid` ∈ [1.2e-3, 4e-3)
    （过 fast-keep-sia）、`low` ∈ (0, 1.2e-3)（**会被快评淘汰**）。
    `full` = 任何 local（全量评估过）事件，不分段。
    """
    lo, hi = {'high': (0.004, float('inf')), 'mid': (0.0012, 0.004),
              'low': (0.0, 0.0012), 'full': (0.0, float('inf'))}[band]
    events = ('local',) if band == 'full' else ('fast', 'local')
    pool: dict[str, dict] = {}
    for rec in ledger.read():
        if rec.get('event') not in events:
            continue
        if rec.get('label') != 'c2c' or not rec.get('formula'):
            continue
        t = (rec.get('data') or {}).get('tier1') or {}
        s = t.get('sia')
        if s is None or not (lo <= float(s) < hi):
            continue
        pool[rec['expression_hash']] = {
            'formula': rec['formula'], 'family': rec.get('family') or 'x',
            'src_sia': float(s), 'source': 'ledger:%s' % band}
    keys = sorted(pool)
    rng.shuffle(keys)
    return [pool[k] for k in keys[:n]]


def rows_from_generator(n: int, seed: int) -> list[dict]:
    """全新随机候选（无历史）。"""
    out = []
    for fam, node in generate_population(max(1, n), seed=seed):
        out.append({'formula': render(node), 'family': fam, 'src_sia': None,
                    'source': 'generated'})
    return out[:n]


def build_candidates(args, ledger: Ledger) -> list[dict]:
    """**分层配额**构造候选集：每个 SiA 段都保留名额（洗牌后再截断会随机丢掉整段）。

    关键：`low` 段是"会被快评淘汰"的候选——**错误淘汰率只有在它被纳入时才非零**。
    这也是 Codex「不能只挑成功候选来验证快评」的落实。
    """
    rng = random.Random(args.seed)
    bands = ('low', 'mid', 'high', 'full')
    per = max(1, args.n // (len(bands) + 1))  # 留一份给 generated
    rows: list[dict] = []
    seen: set[str] = set()
    got_by_band: dict[str, int] = {}
    for b in bands:
        got = 0
        for r in rows_from_ledger(ledger, b, per * 6, rng):
            if r['formula'] in seen:
                continue
            seen.add(r['formula'])
            rows.append(r)
            got += 1
            if got >= per:
                break
        got_by_band[b] = got
    room = max(1, args.n - len(rows))
    for r in rows_from_generator(room, args.seed):
        if r['formula'] in seen:
            continue
        seen.add(r['formula'])
        rows.append(r)
        got_by_band['generated'] = got_by_band.get('generated', 0) + 1
        if len(rows) >= args.n:
            break
    print('    分层配额（每段 %d）：%s' % (per, got_by_band))
    rng.shuffle(rows)
    return rows[:args.n]


# ---------------------------------------------------------------- 评估
def evaluate_all(ctxs: dict, rows: list[dict], args) -> pd.DataFrame:
    out = []
    n = len(rows)
    for i, r in enumerate(rows, 1):
        try:
            node = parse_formula(r['formula'])
        except Exception as exc:
            print('  [skip] parse fail: %s (%s)' % (r['formula'][:50], exc))
            continue
        rec = dict(r)
        ok = True
        for arm, ctx in ctxs.items():
            try:
                m = ctx.screen(node, name=arm)
                t = tier1(m)
            except Exception as exc:
                print('  [skip %s] %s: %s' % (arm, r['formula'][:40], str(exc)[:80]))
                ok = False
                break
            sia = float(t.get('sia') or 0.0)
            rec['%s_sia' % arm] = sia
            rec['%s_score' % arm] = rank_score(
                {'sia': sia, 'turnover_month': t.get('turnover_month'),
                 'turnover': t.get('turnover')}, node, turnover_ref=0.3)
            rec['%s_rank_ic' % arm] = t.get('rank_ic')
        if not ok:
            continue
        out.append(rec)
        if i % 5 == 0 or i == n:
            print('    %d/%d 完成（最后: %s）' % (i, n, r['formula'][:44]), flush=True)
    return pd.DataFrame(out)


def _spearman(a, b) -> float:
    if len(a) < 3:
        return float('nan')
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def fast_kept_mask(df: pd.DataFrame, arm: str, cap: int) -> np.ndarray:
    """**忠实复现** `run_search` 的快评晋级规则（`run_search.py:825-838`）。

    产线逻辑三步：
      ① `keep = {sia ≥ fast_keep_sia}`（`dual_ok` 默认恒真，故略）
      ② `keep ∪= {按 fast_score 排序的前 int(n × fast_top_frac) 名}`
      ③ **对 keep 集按 fast_score 排序后取前 cap**（`survivors[:full_cap]`）

    ⚠ 第 ③ 步是"截断 keep 集"，**不是**"keep 与全局前 cap 求交"——
    若 keep 中有部分成员落在全局 cap 名之外，产线仍会取到它们（只要 keep 集内部
    排序在前 cap）。旧实现用了求交，会**高估**错误淘汰（2026-10-11 修正）。
    另：产线在截断前按 `expression_hash` 去重；本基准候选公式唯一，去重为空操作。
    """
    sia = df['%s_sia' % arm].to_numpy(dtype=float)
    score = df['%s_score' % arm].to_numpy(dtype=float)
    cand = sia >= FAST_KEEP_SIA
    k_top = int(len(df) * FAST_TOP_FRAC)
    if k_top > 0:
        cand[np.argsort(-score)[:k_top]] = True
    order = [i for i in np.argsort(-score) if cand[i]]
    out = np.zeros(len(df), dtype=bool)
    out[order[:cap]] = True
    return out


def diag_k(n: int, cap: int) -> int:
    """诊断用 Top-K：`n > cap` 时等于生产名额 cap；否则退化，须缩小 K 并显式标注。

    为什么必须缩小：若 `n ≤ cap`，则"D 的前 cap 名"就是全部候选 → topk_hit_rate 恒为 1、
    截断分析无意义。此时用 `max(3, n//3)` 作诊断 K，**只作组间对照**，
    **不代表生产的 `--full-cap` 截断损失**（后者需 `n > cap` 的更大规模运行）。
    """
    return int(min(cap, max(3, n // 3)))


def compare(df: pd.DataFrame, arm: str, cap: int) -> dict:
    d_sia = df['D_sia'].to_numpy(dtype=float)
    a_sia = df['%s_sia' % arm].to_numpy(dtype=float)
    d_sc = df['D_score'].to_numpy(dtype=float)
    a_sc = df['%s_score' % arm].to_numpy(dtype=float)
    k = diag_k(len(df), cap)
    a_top = set(np.argsort(-a_sc)[:k].tolist())
    a_kept = fast_kept_mask(df, arm, cap)
    order_d = np.argsort(-d_sc)
    # 用户口径的 headline：**遗漏的优秀因子数**（D 前 K 中未晋级的个数）
    rec = {}
    for kk in (10, 20, 60):
        if len(df) < kk:
            rec['missed_at_%d' % kk] = None
            rec['recall_at_%d' % kk] = None
            continue
        d_t = set(order_d[:kk].tolist())
        miss = [j for j in d_t if not a_kept[j]]
        rec['missed_at_%d' % kk] = len(miss)
        rec['recall_at_%d' % kk] = (kk - len(miss)) / kk
    d_top = set(order_d[:k].tolist())
    false_elim = [j for j in d_top if not a_kept[j]]
    err = a_sia - d_sia
    out = {
        'arm': arm, 'n': len(df), 'k_diag': k, 'n_gt_cap': len(df) > cap,
        'sign_consistency': float(np.mean(np.sign(a_sia) == np.sign(d_sia))),
        'spearman_score': _spearman(a_sc, d_sc),
        'spearman_sia': _spearman(a_sia, d_sia),
        'topk_hit_rate': (len(d_top & a_top) / k) if k else float('nan'),
        'false_elim_rate': (len(false_elim) / k) if k else float('nan'),
        'n_false_elim': len(false_elim),
        'bias_mean_sia': float(np.mean(err)),
        'err_median_abs': float(np.median(np.abs(err))),
        'err_max_abs': float(np.max(np.abs(err))),
        'kept_n': int(a_kept.sum()),
    }
    out.update(rec)
    return out


def truncation_groups(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """按家族看 D 前 K 的截断情况（避免总体指标掩盖某类被系统性淘汰）。

    K = `diag_k(n, cap)`；当 `n ≤ cap` 时这只是组间对照，非生产截断损失。
    """
    k = diag_k(len(df), cap)
    d_sc = df['D_score'].to_numpy(dtype=float)
    df = df.copy()
    df['_d_rank'] = pd.Series(-d_sc).rank(method='first').astype(int)
    df['_in_d_top'] = df['_d_rank'] <= k
    df['_kept_by_A'] = fast_kept_mask(df, 'A', cap)
    g = df.groupby('family')
    rows = []
    for fam, sub in g:
        dtop = sub[sub['_in_d_top']]
        rows.append({
            'family': fam, 'n': len(sub),
            'n_in_D_top': len(dtop),
            'n_D_top_eliminated': int((~dtop['_kept_by_A']).sum()) if len(dtop) else 0,
            'elim_rate_within_D_top': (float((~dtop['_kept_by_A']).mean())
                                       if len(dtop) else float('nan')),
        })
    return pd.DataFrame(rows).sort_values('n_D_top_eliminated', ascending=False)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--n', type=int, default=40, help='候选数（默认 40）')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--data-cut', default='auto',
                    help='数据截止日（默认 %s）' % W.DEFAULT_DATA_CUT)
    ap.add_argument('--label', default='c2c')
    ap.add_argument('--cycle', type=int, default=3)
    ap.add_argument('--fast-stocks', type=int, default=2500)
    ap.add_argument('--fast-months', type=int, default=26)
    ap.add_argument('--fast-seed', type=int, default=11)
    ap.add_argument('--cap', type=int, default=FULL_CAP, help='全量名额（默认 60）')
    ap.add_argument('--out', default=str(OUT_DIR))
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    cut = W.normalize_data_cut(args.data_cut)
    bad = W.check_data_cut(cut)
    if bad:
        raise SystemExit('data_cut 违规：' + '；'.join(bad))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 78)
    print('个股线快评精度基准（PLAN §20.14）')
    print('=' * 78)
    print('data_cut=%s  label=%s  cycle=%d  n=%d  cap=%d' %
          (cut, args.label, args.cycle, args.n, args.cap))
    print()

    t0 = time.time()
    print('--- 加载面板 ---')
    panel = load_panel()
    print('  %d 行, %s ~ %s, %d 只股票 (%.0fs)'
          % (len(panel), panel['date'].min().date(), panel['date'].max().date(),
             panel['stock_code'].nunique(), time.time() - t0))
    if pd.Timestamp(cut) < panel['date'].max():
        panel = panel[panel['date'] <= pd.Timestamp(cut)].copy()
        print('  已裁到 data_cut: %d 行, %s ~ %s'
              % (len(panel), panel['date'].min().date(), panel['date'].max().date()))

    # --- 4 组对照上下文 ---
    print('\n--- 构造四组对照上下文 ---')
    fast_frame = build_fast_frame(panel, n_stocks=args.fast_stocks,
                                  months=args.fast_months, seed=args.fast_seed,
                                  end=cut)
    picked = pick_stocks(panel, n_stocks=args.fast_stocks, seed=args.fast_seed)
    sampled_full = panel[panel['stock_code'].isin(set(picked))].copy()
    fast_start = (pd.Timestamp(cut) - pd.DateOffset(months=args.fast_months)
                  ).strftime('%Y%m%d')
    ev_full = Evaluator(panel)
    ev_samp = Evaluator(sampled_full)
    ctxs = {
        'A': DataContext(cycle=args.cycle, start=fast_start, end=cut,
                         frame=fast_frame, label=args.label,
                         evaluator=Evaluator(fast_frame)),
        'B': DataContext(cycle=args.cycle, start=fast_start, end=cut,
                         frame=panel, label=args.label, evaluator=ev_full),
        'C': DataContext(cycle=args.cycle, start=START, end=cut,
                         frame=sampled_full, label=args.label, evaluator=ev_samp),
        'D': DataContext(cycle=args.cycle, start=START, end=cut,
                         frame=panel, label=args.label, evaluator=ev_full),
    }
    for k, c in ctxs.items():
        print('  %s: rows=%-9d sections=%-5d stocks=%-5d 窗 %s ~ %s'
              % (k, c.n_rows, len(c.ud), len(set(c.stock_codes)),
                 pd.Timestamp(c.ud[0]).date(), pd.Timestamp(c.ud[-1]).date()))

    # --- 候选集 ---
    print('\n--- 构造候选集（分段，必须含"会被快评淘汰"的 low 段）---')
    ledger = Ledger(data_cut=W.LEGACY_DATA_CUT)
    rows = build_candidates(args, ledger)
    from collections import Counter
    print('  n=%d  来源分布: %s' % (len(rows), dict(Counter(r['source'] for r in rows))))

    # --- 评估 ---
    print('\n--- 四组对照评估（D 最贵，最后跑）---')
    df = evaluate_all(ctxs, rows, args)
    if df.empty:
        raise SystemExit('无有效候选，基准未完成')

    # --- 指标 ---
    print('\n--- 指标（各组 vs D）---')
    metrics = pd.DataFrame([compare(df, a, args.cap) for a in ('A', 'B', 'C')])
    pd.set_option('display.width', 240)
    print(metrics.to_string(index=False, float_format=lambda x: '%.4f' % x))

    print('\n--- 头行指标：**遗漏的优秀因子**（用户口径；D = 全量真实排名）---')
    for _, r in metrics.iterrows():
        parts = []
        for kk in (10, 20, 60):
            m = r.get('missed_at_%d' % kk)
            parts.append('top%-2d 漏 %s / %d' % (kk, '-' if m is None else int(m), kk))
        print('  %s：%s' % (r['arm'], '   '.join(parts)))

    kd = diag_k(len(df), args.cap)
    groups = truncation_groups(df, args.cap)
    print('\n--- 截断损失（按家族；A 快评 vs D 真实前 %d）---' % kd)
    if len(df) <= args.cap:
        print('  ⚠ n=%d ≤ cap=%d：诊断 K 已缩为 %d，**只作组间对照**；'
              '生产 --full-cap=%d 的截断损失需 n > %d 的更大规模运行。'
              % (len(df), args.cap, kd, args.cap, args.cap))
    print(groups.to_string(index=False, float_format=lambda x: '%.4f' % x))

    # --- 落盘 ---
    stag = '%s_n%d_seed%d' % (cut.replace('-', ''), len(df), args.seed)
    p_csv = out_dir / ('bench_fast_%s.csv' % stag)
    p_m = out_dir / ('bench_fast_%s_metrics.csv' % stag)
    p_g = out_dir / ('bench_fast_%s_groups.csv' % stag)
    p_md = out_dir / ('bench_fast_%s.md' % stag)
    df.to_csv(p_csv, index=False)
    metrics.to_csv(p_m, index=False)
    groups.to_csv(p_g, index=False)
    _write_md(p_md, args, cut, ctxs, df, metrics, groups)

    print('\n产物：')
    for p in (p_csv, p_m, p_g, p_md):
        print('  ', p)
    print('总耗时 %.0fs' % (time.time() - t0))
    print()
    print('注：本基准**只测量**。基准完成前不得修改快评阈值 / --full-cap / 筛选流程（PLAN §20.14）。')
    return 0


def _write_md(path: Path, args, cut: str, ctxs: dict, df: pd.DataFrame,
              metrics: pd.DataFrame, groups: pd.DataFrame) -> None:
    lines = []
    lines.append('# 个股线快评精度基准（PLAN §20.14）')
    lines.append('')
    lines.append('> **只测量、不改生产参数。** 数据截止日 `%s`，label `%s`，cycle %d，'
                 'cap %d，候选 %d。' % (cut, args.label, args.cycle, args.cap, len(df)))
    lines.append('')
    lines.append('## 四组设计')
    lines.append('')
    lines.append('| 组 | 截面 | 时间窗 | 作用 |')
    lines.append('|---|---|---|---|')
    lines.append('| A | 抽样 %d | %d 月尾窗 | = 现行快评 |' % (args.fast_stocks, args.fast_months))
    lines.append('| B | 全量 | %d 月尾窗 | 隔离**截面抽样**效应 |' % args.fast_months)
    lines.append('| C | 抽样 %d | 完整窗 | 隔离**窗口缩短**效应 |' % args.fast_stocks)
    lines.append('| D | 全量 | 完整窗 | 全量评估基准 |')
    lines.append('')
    lines.append('上下文规模：')
    lines.append('')
    lines.append('| 组 | 行数 | 截面数 | 股票数 | 窗口 |')
    lines.append('|---|---|---|---|---|')
    for k in ('A', 'B', 'C', 'D'):
        c = ctxs[k]
        lines.append('| %s | %d | %d | %d | %s ~ %s |' % (
            k, c.n_rows, len(c.ud), len(set(c.stock_codes)),
            pd.Timestamp(c.ud[0]).date(), pd.Timestamp(c.ud[-1]).date()))
    lines.append('')
    lines.append('## 指标（各组 vs D）')
    lines.append('')
    lines.append(metrics.to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('**读法**：')
    lines.append('')
    lines.append('- `sign_consistency`：SiA 符号一致率。快评把 D 中正向的判成负向即为符号事故。')
    lines.append('- `false_elim_rate`：**最重要**——D 前 cap 名中，被该组快评规则淘汰的比例（假阴性）。')
    lines.append('  快评只是门槛；被它淘汰的候选，后续全量再准也无法挽回。')
    lines.append('- `bias_mean_sia`：SiA 偏差均值。正=快评系统性高估。')
    lines.append('- `topk_hit_rate` 用排序分（`rank_score`）口径；`spearman_*` 给出单调关系。')
    lines.append('')
    lines.append('## 截断损失（按家族）')
    lines.append('')
    kd = diag_k(len(df), args.cap)
    if len(df) <= args.cap:
        lines.append('> ⚠ **`n=%d ≤ cap=%d`：诊断 K 已缩为 `%d`，只作组间对照。** '
                     '生产 `--full-cap=%d` 的截断损失需 `n > %d` 的更大规模运行。'
                     % (len(df), args.cap, kd, args.cap, args.cap))
        lines.append('')
    lines.append(groups.to_markdown(index=False, floatfmt='.4f'))
    lines.append('')
    lines.append('## 纪律')
    lines.append('')
    lines.append('- 本基准**不修改**任何生产参数；结论须结合**错误淘汰率**与**计算收益**共同判断。')
    lines.append('- 样本量有限（n=%d），只作诊断；不得据单一指标（尤其单一相关系数）下结论。' % len(df))
    lines.append('- 门控线的抽样否决（ΔIC 符号一致率 1/3）**不可直接搬用**到个股线。')
    lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
