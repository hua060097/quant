# -*- coding: utf-8 -*-
"""方向语义验收：用**合成数据**验证 G2/G3/F1 的判定方向。

背景（Codex 2026-10-12）：真实数据上"通过率翻转"**不能**证明修复正确——
必须用已知答案的合成数据，覆盖 正效应 / 负效应 / 零效应 × 门控方向(pos/neg) 全组合，
确认每个判定函数在每种情形下给出正确结论。

设计（效应强度可控，零效应由大样本构造）：
  合成 IC 矩阵 ic[T,F]（F 个因子共享同一个日级效应 → 组级统计量稳定）
  合成门控 g[T]（约一半日子为 +，一半为 −）
  情形：
    POS  : g>0 的日子 IC 上抬 +E   → 期望 G2 过、G3 过
    NEG  : g>0 的日子 IC 下压 −E   → 期望 G2 **过**（§8.2：95% 区间不含 0 = 双侧）、
                                     G3 **不过**（§8.2：单侧「超过 95% 分位」）；
    NULL : IC 与 g 无关             → 期望 G2 不过、G3 不过
  F1 方向：对同一门控，s=+1（on = g>0）应看到该效应；s=−1（on = g<0）应看到**相反**号。

用法：python selftest_gate_direction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_g2g3 as G23
from f1_batch import run_one_market

PASS, FAIL = [], []
T, F = 750, 12          # 750 个交易日 × 12 个因子（够 g2_ci_bootstrap 的块数）
EFFECT = 0.02           # 日 IC 的均值位移（量级 ~ 真实 ΔIC）
NOISE = 0.06            # 日 IC 的噪声（IC 量级 ~0.05）


def check(tid: str, name: str, cond: bool, detail: str = '') -> None:
    (PASS if cond else FAIL).append(tid)
    print('  [%s] %-4s %s%s' % ('OK  ' if cond else 'FAIL', tid, name,
                               ('  -> ' + detail) if detail else ''))


def synth(kind: str, seed: int = 7):
    """返回 (ic[T,F], g[T])。g 约半数日子为正。"""
    rng = np.random.default_rng(seed)
    g = np.where(np.arange(T) % 2 == 0, 0.6, -0.6)        # 干净交替，避免与噪声混淆
    ic = rng.standard_normal((T, F)) * NOISE
    if kind == 'POS':
        ic[g > 0] += EFFECT
    elif kind == 'NEG':
        ic[g > 0] -= EFFECT
    elif kind == 'NULL':
        pass
    else:
        raise ValueError(kind)
    return ic, g


def g2_verdict(ic, g):
    st = G23.g2_stat(ic, g)
    lo, hi = G23.g2_ci_bootstrap(ic, g, n_boot=200)
    ok = (st["folds_same"] == G23.G2_FOLDS) and G23.ci_excludes_zero(lo, hi) \
        and st["same_frac"] >= G23.G2_SAME_SIGN_MIN
    return bool(ok), st, lo, hi


def g3_verdict(ic, g):
    return G23.g3_placebo(ic, g, n_perm=100)


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print('=' * 78)
    print('方向语义验收（合成数据：POS / NEG / NULL）')
    print('=' * 78)

    res = {}
    for kind in ('POS', 'NEG', 'NULL'):
        ic, g = synth(kind)
        ok2, st, lo, hi = g2_verdict(ic, g)
        g3 = g3_verdict(ic, g)
        res[kind] = (ok2, g3['pass'], st, lo, hi, g3)
        print('\n[%s] 中位斜率=%+.5f 同号=%.2f 4折=%d/4 CI=[%+.5f,%+.5f]  '
              'G2=%s  | G3 obs=%+.5f q05=%+.5f q95=%+.5f -> %s'
              % (kind, st['median_b'], st['same_frac'], st['folds_same'], lo, hi,
                 ok2, g3['obs'], g3['q05'], g3['q95'], g3['pass']))

    print()
    print('--- 断言 ---')
    check('D1', 'POS 效应 → G2 过', res['POS'][0],
          'med=%+.5f CI=[%+.5f,%+.5f]' % (res['POS'][2]['median_b'], res['POS'][3], res['POS'][4]))
    check('D2', 'POS 效应 → G3 过（正向超 q95）', res['POS'][1],
          'obs=%+.5f q95=%+.5f p2=%.4f' % (res['POS'][5]['obs'], res['POS'][5]['q95'],
                                           res['POS'][5]['p_two_sided']))
    check('D3', 'NEG 效应 → G2 **过**（规格原文：95% 区间不含 0，双侧）', res['NEG'][0],
          'med=%+.5f CI=[%+.5f,%+.5f]' % (res['NEG'][2]['median_b'], res['NEG'][3], res['NEG'][4]))
    # G3 = **单侧**（§8.2 原文「真实值**超过**零分布 95% 分位」）。
    # 2026-10-12 规则回退：本测试曾断言 NEG → G3 过（双侧口径）；
    # 但原规格明确为单侧，双侧属研究规则变更，已回退。
    # 故此处断言 NEG **不过** G3，并同时断言「双侧诊断下会过」——
    # 用以证明「单侧是规格选择，不是实现缺陷」。
    check('D4', 'NEG 效应 → G3 **不过**（单侧规格）；但双侧诊断下会过',
          (not res['NEG'][1]) and bool(res['NEG'][5].get('pass_two_sided_diag')),
          'obs=%+.5f q95=%+.5f p1=%.4f | 双侧诊断=%s'
          % (res['NEG'][5]['obs'], res['NEG'][5]['q95'],
             res['NEG'][5]['p_one_sided'],
             res['NEG'][5].get('pass_two_sided_diag')))
    check('D5', 'NULL 效应 → G2 不过', not res['NULL'][0],
          'med=%+.5f CI=[%+.5f,%+.5f]' % (res['NULL'][2]['median_b'], res['NULL'][3], res['NULL'][4]))
    check('D6', 'NULL 效应 → G3 不过', not res['NULL'][1],
          'obs=%+.5f p2=%.4f' % (res['NULL'][5]['obs'], res['NULL'][5]['p_two_sided']))

    # ---------- F1 方向语义（s=+1 / s=-1）----------
    print()
    print('--- F1 方向语义（run_one_market，m 未参与 delta 计算故传 None）---')
    uniq = np.arange(T)
    for kind in ('POS', 'NEG'):
        ic, g = synth(kind)
        ic_z = ic.mean(axis=1)              # 组级口径：逐日跨因子均值 IC
        r_pos = run_one_market(None, 'x', 1, g, 1, uniq, ic_z)     # on = g>0
        r_neg = run_one_market(None, 'x', 1, g, -1, uniq, ic_z)    # on = g<0
        d_pos = float(r_pos['delta']['mean'])
        d_neg = float(r_neg['delta']['mean'])
        p_pos = float(r_pos['placebo']['p'])
        p_neg = float(r_neg['placebo']['p'])
        print('  [%s] s=+1 delta=%+.5f p=%.4f | s=-1 delta=%+.5f p=%.4f'
              % (kind, d_pos, p_pos, d_neg, p_neg))
        if kind == 'POS':
            check('D7', 'POS 效应：s=+1 得正 ΔIC、s=-1 得负 ΔIC（方向语义正确）',
                  d_pos > 0 > d_neg, '%+.5f / %+.5f' % (d_pos, d_neg))
        else:
            check('D8', 'NEG 效应：**s=-1 得正 ΔIC**（这正是被硬编码 s=1 埋没的方向）',
                  d_neg > 0 > d_pos, 's=-1 %+.5f / s=+1 %+.5f' % (d_neg, d_pos))

    print()
    print('=' * 78)
    print('结果：%d/%d 通过' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        print('失败项：', FAIL)
    print('=' * 78)
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
