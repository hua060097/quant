# -*- coding: utf-8 -*-
"""T-CUT：数据截止日（data_cut）指纹隔离的回归测试。

验证「候选身份 = 公式 × 数据截止日」的核心不变量：
  T1 新台账默认写入默认截止日（选择窗上界）
  T2 显式截止日参数生效且被规范
  T3 隔离：默认截止日下不返回遗留截止日的已测公式（**用户指令的核心**）
  T4 遗留截止日下仍可查询到自己的记录（标记而非删除）
  T5 data_cut=None 保持旧的全量语义（向后兼容）
  T6 摘要按截止日分别计数（cuts / tested_pairs）
  T7 值指纹（vfp）按截止日隔离
  T8 结构性排除集（fps_covered）随截止日隔离
  T9 越界截止日被拒绝（check_data_cut）
  T10 生产台账自检：迁移后遗留标记齐全

用法：python selftest_cut.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import _paths  # noqa: F401
import windows as W
from grammar import structural_fingerprint
from ledger import DEFAULT_DATA_CUT, Ledger
from parser import parse_formula as parse

PASS, FAIL = [], []


def check(tid: str, name: str, cond: bool, detail: str = '') -> None:
    (PASS if cond else FAIL).append(tid)
    print('  [%s] %-4s %s%s' % ('OK  ' if cond else 'FAIL', tid, name,
                               ('  -> ' + detail) if detail else ''))


def _tmp_ledger() -> Ledger:
    d = Path(tempfile.mkdtemp(prefix='cutsel_'))
    return Ledger(d / 'ledger.jsonl')


NODES = [
    'TS_MEAN(CLOSE,20)/CLOSE',
    '((CLOSE)/(TS_MAX(HIGH,40)))-(1)',
    '0-(SLOPE(LOG(CLOSE),60))',
]


def main() -> int:
    print('=== T-CUT：数据截止日指纹隔离 ===')
    lg = _tmp_ledger()

    # ---- T1 默认写入
    n0 = parse(NODES[0])
    n1 = parse(NODES[1])
    n2 = parse(NODES[2])
    r0 = lg.log('fast', n0, 1, family='F1', label='c2c',
                struct_fp=structural_fingerprint(n0),
                data={'tier1': {'sia': 0.01}, 'vfp': 'aaaaaaaaaaaa'})
    r1 = lg.log('fast', n1, 1, family='F1', label='c2c',
                struct_fp=structural_fingerprint(n1),
                data={'tier1': {'sia': 0.02}, 'vfp': 'bbbbbbbbbbbb'})
    check('T1', 'log() 默认写入 DEFAULT_DATA_CUT',
          r0.get('data_cut') == DEFAULT_DATA_CUT == '2025-12-31',
          'data_cut=%r' % r0.get('data_cut'))

    # ---- T2 显式截止日
    r2 = lg.log('fast', n2, 1, family='F2', label='c2c', data_cut='2025/06/30',
                struct_fp=structural_fingerprint(n2),
                data={'tier1': {'sia': 0.03}, 'vfp': 'cccccccccccc'})
    check('T2', 'log() 显式截止日被规范化',
          r2.get('data_cut') == '2025-06-30', 'data_cut=%r' % r2.get('data_cut'))

    # ---- T3/T4/T5 隔离
    h_def = lg.hashes_for(['c2c'], data_cut='2025-12-31')
    h_old = lg.hashes_for(['c2c'], data_cut='2025-06-30')
    h_all = lg.hashes_for(['c2c'], data_cut=None)
    check('T3', '默认截止日不含其他截止日的公式',
          h_def == {r0['expression_hash'], r1['expression_hash']}
          and r2['expression_hash'] not in h_def,
          'n=%d' % len(h_def))
    check('T4', '遗留截止日可查到自己的记录',
          h_old == {r2['expression_hash']}, 'n=%d' % len(h_old))
    check('T5', 'data_cut=None 恢复全量语义',
          h_all == {r0['expression_hash'], r1['expression_hash'],
                    r2['expression_hash']}, 'n=%d' % len(h_all))

    # ---- T6 摘要按截止日计数
    cuts = lg.cut_stats()
    tested_def = lg.count_tested('2025-12-31')
    tested_jun = lg.count_tested('2025-06-30')
    check('T6', 'cuts/tested_pairs 按截止日分别计数',
          set(cuts) == {'2025-12-31', '2025-06-30'} and tested_def == 2
          and tested_jun == 1 and lg.count_tested() == 3,
          'cuts=%s def=%d jun=%d all=%d' % (sorted(cuts), tested_def, tested_jun,
                                            lg.count_tested()))

    # ---- T7 值指纹隔离
    v_def = lg.vfps_for('2025-12-31')
    v_jun = lg.vfps_for('2025-06-30')
    check('T7', 'vfp 按截止日隔离',
          v_def == {'aaaaaaaaaaaa', 'bbbbbbbbbbbb'} and v_jun == {'cccccccccccc'},
          'def=%d jun=%d' % (len(v_def), len(v_jun)))

    # ---- T8 结构指纹随截止日隔离
    fp_def = lg.fps_covered(h_def)
    fp_jun = lg.fps_covered(h_old)
    check('T8', 'fps_covered 随传入排除集隔离',
          fp_def and fp_jun and not (fp_def & fp_jun),
          'def=%d jun=%d 交集=%d' % (len(fp_def), len(fp_jun), len(fp_def & fp_jun)))

    # ---- T9 越界拒绝（边界语义：上界=衰减监控起点前；下界=面板首日 2021-06-01）
    bad_hi = W.check_data_cut('2026-01-05')      # 触及监控段 → 拒绝
    bad_lo = W.check_data_cut('2020-01-01')      # 早于面板首日 → 拒绝
    ok_mid = W.check_data_cut('2025-12-31')      # 选择窗上界 → 合规
    ok_early = W.check_data_cut('2023-01-01')    # 更早的合法截止日 → 合规（2026-10-11 放宽）
    check('T9', '越界截止日被拒绝、合法截止日（含更早的）合规',
          bool(bad_hi) and bool(bad_lo) and not ok_mid and not ok_early,
          'hi=%d lo=%d ok_mid=%d ok_early=%d' % (len(bad_hi), len(bad_lo),
                                                 len(ok_mid), len(ok_early)))

    # ---- T11 数据版本「只记录、不过滤」+ 显式重测杠杆（2026-10-11 用户裁定）
    # 裁定：cache 是日频重生成的，指纹若以内容哈希为作用键 ⇒ 每个交易日全部失效。
    # 故：**指纹的作用键只有 cut_date**；data_version 只写事件供溯源。
    # 换代时的显式杠杆 = `retest_before`（只重测最后事件早于该时刻的公式）。
    lg2 = _tmp_ledger()
    na = parse('TS_MEAN(CLOSE,10)/CLOSE')
    nb = parse('0-(TS_MEAN(CLOSE,40)/CLOSE)')
    nc_ = parse('TS_MAX(HIGH,20)/CLOSE')
    r_a = lg2.log('fast', na, 1, family='F', label='c2c', data_cut='2025-12-31',
                  data_version='panel_aaa',
                  data={'tier1': {'sia': 0.01}, 'vfp': 'v1'})
    r_b = lg2.log('fast', nb, 1, family='F', label='c2c', data_cut='2025-12-31',
                  data_version='panel_bbb',
                  data={'tier1': {'sia': 0.02}, 'vfp': 'v2'})
    lg2.log('fast', nc_, 1, family='F', label='c2c', data_cut='2025-12-31',
            data={'tier1': {'sia': 0.03}, 'vfp': 'v3'})          # 无版本（旧记录）
    # (a) 生产口径（不传 data_version）：**三个版本都要在** —— 指纹不随版本变化
    h_prod = lg2.hashes_for(['c2c'], data_cut='2025-12-31')
    v_prod = lg2.vfps_for('2025-12-31')
    e_prod = lg2.evidence_pool_for('2025-12-31')
    check('T11a', '生产口径不按数据版本过滤（指纹只依赖 cut_date）',
          len(h_prod) == 3 and v_prod == {'v1', 'v2', 'v3'} and len(e_prod) == 3,
          'hashes=%d vfp=%s 证据=%d' % (len(h_prod), sorted(v_prod), len(e_prod)))
    # (b) 审计口径：显式传版本仍可筛（供"这批数字出自哪个数据世代"的追溯）
    h_a = lg2.hashes_for(['c2c'], data_cut='2025-12-31', data_version='panel_aaa')
    check('T11b', '审计口径可按数据版本追溯（非生产路径）',
          h_a == {r_a['expression_hash']}, 'ver_a=%d' % len(h_a))
    # (c) 显式重测杠杆：retest_before = 未来时刻 ⇒ 全部重新纳入（排除集为空）
    h_future = lg2.hashes_for(['c2c'], data_cut='2025-12-31',
                              retest_before='2999-01-01')
    check('T11c', 'retest_before 未来时刻 ⇒ 全部重新纳入（显式换代杠杆）',
          len(h_future) == 0, 'n=%d' % len(h_future))

    # ---- T10 生产台账自检（迁移后遗留标记齐全）
    prod = Ledger()
    if prod.path.exists():
        recs = prod.read()
        no_cut = sum(1 for r in recs if not r.get('data_cut'))
        legacy = sum(1 for r in recs if r.get('data_cut_legacy'))
        st = prod.summary()
        n_prod_def = prod.count_tested(DEFAULT_DATA_CUT)
        check('T10', '生产台账已标记遗留截止日',
              no_cut == 0 and legacy == len(recs) and n_prod_def == 0,
              'n=%d 缺cut=%d legacy=%d 默认cut已测=%d'
              % (len(recs), no_cut, legacy, n_prod_def))
    else:
        check('T10', '生产台账已标记遗留截止日', True, '台账不存在，跳过')

    print()
    print('结果：%d/%d 通过' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        print('失败项：', FAIL)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
