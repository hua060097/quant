# -*- coding: utf-8 -*-
"""F1v3三算子权重合成自测（纯numpy，无行情数据依赖）。

覆盖：gain裁剪边界/NaN传递、flip符号翻转、switch混合边界、
旧build_f1冻结语义不变。
用法：python selftest_interaction_forms.py
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
import h3_eval as H  # noqa: E402

PASS, FAIL = [], []


def check(tid: str, name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(tid)
    print("  [%s] %-4s %s%s" % ("OK  " if cond else "FAIL", tid, name,
                                ("  -> " + detail) if detail else ""))


def main() -> int:
    print("=== F1v3 interaction forms ===")
    g = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, np.nan])
    w = H.build_gain_weight(g, 1)
    check("G1", "gain s=+1 clip下界/上界",
          np.isclose(w[0], 0.5) and np.isclose(w[4], 1.5) and np.isclose(w[2], 1.0)
          and np.isnan(w[5]),
          "w=%s" % np.array2string(w, precision=2))
    w = H.build_gain_weight(g, -1)
    check("G2", "gain s=-1 反向",
          np.isclose(w[0], 1.5) and np.isclose(w[4], 0.5),
          "w=%s" % np.array2string(w, precision=2))
    check("G3", "gain恒正（无零塌缩）", bool(np.all(w[:5] > 0)))

    f = H.build_flip_weight(np.array([-1.0, 0.5, np.nan]), -1)
    check("F1", "flip s=-1 翻转符号",
          np.isclose(f[0], 1.0) and np.isclose(f[1], -0.5) and np.isnan(f[2]),
          "w=%s" % np.array2string(f, precision=2))
    fz = H.build_flip_weight(np.array([0.09, -0.09, 0.10, -0.11]), 1)
    check("F2", "flip中性死区|g|<0.10置NaN",
          np.isnan(fz[0]) and np.isnan(fz[1])
          and np.isclose(fz[2], 0.10) and np.isclose(fz[3], -0.11),
          "w=%s" % np.array2string(fz, precision=2))

    b = H.build_switch_blend(np.array([1.0, 0.0, 0.5, np.nan]),
                             np.array([2.0, 2.0, 2.0, 2.0]),
                             np.array([0.0, 0.0, 0.0, 0.0]))
    check("S1", "switch端点/中点混合（等方差归一）",
          np.isclose(b[0], 2.0) and np.isclose(b[1], 0.0)
          and np.isclose(b[2], 1.0 / np.sqrt(0.5)) and np.isnan(b[3]),
          "F=%s" % np.array2string(b, precision=2))
    _st = H.static_consensus(np.array([2.0, 2.0]), np.array([0.0, 0.0]))
    check("S2", "静态共识同归一（/√0.5）",
          np.isclose(_st[0], 1.0 / np.sqrt(0.5)) and np.isclose(_st[1], 1.0 / np.sqrt(0.5)),
          "S=%s" % np.array2string(_st, precision=3))

    f1 = H.build_f1(np.array([0.3, -0.2]), np.array([0.5, -0.5]), 1, 1)
    check("B1", "旧build_f1冻结语义不变",
          np.isclose(f1[0], 0.15) and np.isclose(f1[1], 0.0),
          "F1=%s" % np.array2string(f1, precision=3))

    print(f"interaction forms: {len(PASS)}/{len(PASS) + len(FAIL)} PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
