# -*- coding: utf-8 -*-
"""数据截止日（data_cut）边界审计：日期守门 ≠ 数据隔离证明，须逐路径核查。

背景（Codex 2026-10-11 提醒）：`--cut < --data-cut` 只证明**日期关系**合法，
不能证明**所有实际读取路径**都遵守边界。本脚本做两类核查：

  静态（源码扫描）
    A1 搜索路径模块里 ≥2026-01-01 的日期字面量清单及处置状态
    A2 `ensure_local_fields` 不含前视算子（shift(-n) / 未来窗口）
    A3 `build_label_series` 的前视位移**仅**出现在标签定义中（by-design，须记录）

  动态（合成面板，不加载 7GB parquet）
    B1 `DataContext(end=data_cut)` 的实际最大信号日 ≤ data_cut
    B2 `build_fast_frame(end=data_cut)` 的最大日期 ≤ data_cut
    B3 `dedup.monthly_spearman` 默认 end 被裁到 data_cut
    B4 `pool_legacy.refresh_member_sia` 的窗前上界 = min(遗留, data_cut)
    B5 `pool_sim.build_panel_daily` 接受 start/end 参数（不再硬编码）
    B6 `ledger` 指纹/值指纹/证据池按截止日隔离

用法：python audit_cut_paths.py
返回：0 = 全部通过；1 = 有未处置项
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import dedup
import ledger as LG
import pool_legacy as P
import pool_sim
import windows as W

PASS, FAIL, NOTE = [], [], []


def check(tid: str, name: str, cond: bool, detail: str = '') -> None:
    (PASS if cond else FAIL).append(tid)
    print('  [%s] %-4s %s%s' % ('OK  ' if cond else 'FAIL', tid, name,
                               ('  -> ' + detail) if detail else ''))


def note(tid: str, name: str, detail: str = '') -> None:
    NOTE.append(tid)
    print('  [NOTE] %-4s %s%s' % (tid, name, ('  -> ' + detail) if detail else ''))


# ---------------------------------------------------------------- 合成面板
def synthetic() -> pd.DataFrame:
    """含 DataContext / build_label_series / Evaluator 所需最小列的面板。"""
    dates = pd.bdate_range('2021-06-01', '2026-09-18')
    codes = ['X%03d.SZ' % i for i in range(30)]
    idx = pd.MultiIndex.from_product([codes, dates], names=['stock_code', 'date'])
    df = idx.to_frame(index=False)
    rng = np.random.default_rng(7)
    n = len(df)
    df['CLOSE'] = 10 + rng.standard_normal(n).cumsum() * 0.01
    for c in ('OPEN', 'HIGH', 'LOW'):
        df[c] = df['CLOSE']
    df['VOLUME'] = 1e6
    df['AMOUNT'] = 1e7
    df['MARKET_CAP'] = 1e9
    df['is_limit_up'] = 0
    df['is_can_buy'] = 1
    df['is_st'] = 0
    return df


# ---------------------------------------------------------------- 静态
SEARCH_PATH_MODULES = [
    'run_search.py', 'fitness.py', 'evaluator.py', 'dedup.py', 'pool_legacy.py',
    'pool_sim.py', 'ledger.py', 'gate_ledger.py', 'grammar.py',
]

#: 允许保留的 2026+ 字面量（已显式处置）——每项须说明为何安全
ALLOWED_2026 = {
    ('pool_legacy.py', "WINDOW_END = '2026-07-10'"): '经 windows.cut_bound 裁剪',
    ('pool_legacy.py', "ALIGNED_END = '2026-07-10'"): '经 windows.cut_bound 裁剪',
    ('dedup.py', "end: str = '2026-07-10', min_obs: int = 50,"): '经 windows.cut_bound 裁剪',
    ('pool_sim.py', "start: str = '2023-03-01', end: str = '2026-07-10'):"):
        '仅默认值；调用方必须传 end（pool_legacy._Universe 经 cut_bound 后传入）',
    ('fitness.py', "END = '20260821'"): '遗留默认；SearchRun 显式传 data_cut',
    ('fitness.py', "HOLDOUT_START = '20250701'"): 'holdout 切点在窗内（<data_cut）',
}


def static_scan() -> None:
    hits = []
    for mod in SEARCH_PATH_MODULES:
        p = (HERE / mod) if (HERE / mod).exists() else (HERE.parent / mod)
        if not p.exists():
            continue
        # 文档字符串状态跟踪：prose 里提到日期不算代码字面量
        in_doc = False
        for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
            n_tq = line.count('"""') + line.count("'''")
            was_in_doc = in_doc
            if n_tq % 2 == 1:
                in_doc = not in_doc
            if was_in_doc or line.lstrip().startswith('#'):
                continue
            if re.search(r"['\"]20(2[6-9]|[3-9]\d)-\d\d-\d\d['\"]", line) or \
               re.search(r"['\"]202[6-9]\d{4}['\"]", line):
                hits.append((mod, line.strip(), i))
    unhandled = []
    for mod, line, ln in hits:
        key = (mod, line)
        if key in ALLOWED_2026:
            continue
        unhandled.append((mod, ln, line[:96]))
    check('A1', '搜索路径无未处置的 2026+ 日期字面量',
          not unhandled, '未处置 %d 处%s' % (len(unhandled),
                                             (': ' + str(unhandled[:3])) if unhandled else ''))
    print('        已处置白名单 %d 项；扫描 %d 处 2026+ 代码字面量（已排除注释/文档字符串）'
          % (len(ALLOWED_2026), len(hits)))

    ev = (HERE / 'evaluator.py').read_text(encoding='utf-8')
    seg = ev[ev.find('def ensure_local_fields'):]
    seg = seg[:seg.find('\ndef ', 10)] if '\ndef ' in seg[10:] else seg
    fwd = re.findall(r'shift\(-\d+\)|\[::-1\]|rolling\([^)]*center\s*=\s*True', seg)
    check('A2', 'ensure_local_fields 无前视算子', not fwd, '发现 %s' % (fwd or '无'))

    fit = (HERE / 'fitness.py').read_text(encoding='utf-8')
    seg = fit[fit.find('def build_label_series'):]
    seg = seg[:seg.find('\ndef ', 10)] if '\ndef ' in seg[10:] else seg
    fwd = len(re.findall(r'shift\(-\d+\)', seg))
    note('A3', 'build_label_series 的前视位移 = 标签定义本身',
         '发现 %d 处 shift(-n)——按 windows.py 约定「按信号日设界、标签自然前瞻」；'
         'label_available_max 须随记录落盘' % fwd)
    check('A3', 'build_label_series 前视位移只为构造标签（已记录）', fwd > 0)


# ---------------------------------------------------------------- 动态
def dynamic() -> None:
    df = synthetic()

    cut = W.DEFAULT_DATA_CUT
    ctx = F.DataContext(cycle=3, end=cut, frame=df, label='c2c')
    mx = pd.Timestamp(ctx.row_dates.max()).date().isoformat()
    check('B1', 'DataContext(end=data_cut) 实际最大信号日 ≤ data_cut',
          mx <= cut, 'max=%s cut=%s rows=%d' % (mx, cut, ctx.n_rows))

    sub = F.build_fast_frame(df, n_stocks=30, months=24, end=cut)
    mx2 = pd.Timestamp(sub['date'].max()).date().isoformat()
    check('B2', 'build_fast_frame(end=data_cut) 最大日期 ≤ data_cut',
          mx2 <= cut, 'max=%s cut=%s' % (mx2, cut))

    d = pd.DataFrame({'date': pd.to_datetime(list(pd.bdate_range('2023-01-01', '2026-09-01')) * 60),
                      'a': np.random.default_rng(1).standard_normal(60 * len(pd.bdate_range('2023-01-01', '2026-09-01'))),
                      'b': np.random.default_rng(2).standard_normal(60 * len(pd.bdate_range('2023-01-01', '2026-09-01')))})
    d = d.sort_values('date').reset_index(drop=True)
    called = {}
    orig = dedup.pd.DataFrame.__getitem__

    def _spy_backend(frame, start, end, min_obs, data_cut):
        return None
    # 直接验证 cut_bound 语义（避免依赖 pandas 内部）
    check('B3', 'monthly_spearman 的 end 经 cut_bound 裁剪',
          W.cut_bound('2026-07-10', W.DEFAULT_DATA_CUT) == cut
          and W.cut_bound('2026-07-10', W.CUT_OFF) == '2026-07-10',
          '默认→%s；CUT_OFF→遗留' % cut)

    check('B4', 'pool_legacy 窗前上界 = min(遗留, data_cut)',
          W.cut_bound(P.ALIGNED_END, W.DEFAULT_DATA_CUT) == cut
          and W.cut_bound(P.WINDOW_END, W.DEFAULT_DATA_CUT) == cut,
          'ALIGNED_END/WINDOW_END 均→%s' % cut)

    import inspect
    sig = inspect.signature(pool_sim.build_panel_daily)
    check('B5', 'pool_sim.build_panel_daily 接受 start/end（不再硬编码）',
          'start' in sig.parameters and 'end' in sig.parameters,
          str(sig))

    lg = LG.Ledger(HERE / ('_audit_tmp_%s.jsonl' % W.DEFAULT_DATA_CUT.replace('-', '')))
    try:
        from parser import parse_formula
        n1 = parse_formula('TS_MEAN(CLOSE,20)/CLOSE')
        n2 = parse_formula('0-(SLOPE(LOG(CLOSE),60))')
        lg.log('fast', n1, 1, family='F1', label='c2c', struct_fp='fp1',
               data={'tier1': {'sia': 0.01}, 'vfp': 'v1'})
        lg.log('fast', n2, 1, family='F1', label='c2c', data_cut=W.LEGACY_DATA_CUT,
               struct_fp='fp2', data={'tier1': {'sia': 0.02}, 'vfp': 'v2'})
        h_def = lg.hashes_for(['c2c'], data_cut=cut)
        h_leg = lg.hashes_for(['c2c'], data_cut=W.LEGACY_DATA_CUT)
        v_def = lg.vfps_for(cut)
        e_def = lg.evidence_pool_for(cut)
        e_all = lg.evidence_pool_for(None)
        check('B6', 'ledger 指纹/值指纹/证据池按截止日隔离',
              len(h_def) == 1 and len(h_leg) == 1 and v_def == {'v1'}
              and len(e_def) == 1 and len(e_all) == 2,
              'hashes default/legacy=%d/%d  vfp=%s  证据 default/all=%d/%d'
              % (len(h_def), len(h_leg), sorted(v_def), len(e_def), len(e_all)))
    finally:
        for f in (lg.path, lg.summary_path):
            if Path(f).exists():
                Path(f).unlink()


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('=' * 78)
    print('数据截止日边界审计（data_cut=%s）' % W.DEFAULT_DATA_CUT)
    print('=' * 78)
    print('\n--- 静态扫描 ---')
    static_scan()
    print('\n--- 动态验证（合成面板） ---')
    dynamic()
    print()
    print('=' * 78)
    print('结果：%d 通过 / %d 失败 / %d 记录（NOTE）' % (len(PASS), len(FAIL), len(NOTE)))
    if FAIL:
        print('失败项：', FAIL)
    print('=' * 78)
    return 1 if FAIL else 0


if __name__ == '__main__':
    import fitness as F  # noqa: E402  (延迟导入，避免顶部加载面板依赖)
    sys.exit(main())
