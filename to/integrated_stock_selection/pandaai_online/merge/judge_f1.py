# -*- coding: utf-8 -*-
"""对 F1 批次结果套用**计划 §8.4 的冻结判定口径**。

判定（PLAN §8.4，逐字对照）：
  主指标 ΔIC = mean(IC_F1 − IC_base)（逐日配对差）要求：
    (a) 块自助法（块长 5 日）95% 区间**下界 > 0**
    (b) 评估窗等分 6 个连续块，**至少 4 块 ΔIC > 0**
  块 placebo：200 次，p = (1+k)/201，要求 p < 0.05
  多重检验：BH-FDR（q ≤ 0.10）
  ⇒ h3_pass = (a) and (b) and (c)

**修正记录（2026-10-10）**：此前 glt 实现用 `nblocks >= 5`，与 §8.4 的
"至少 4 块"不符。该错误由"从 M4 旧数据反推"引入——旧批次里 18 个通过的
nblocks 均为 5~6、2 个失败者 nblocks=4，但**那 2 个失败的主因是 p=0.164**，
块数并非判别因素，故旧数据 100/100 吻合掩盖了差异。按 §8.4 修正后
**M3 从"0 通过"变为"4 个通过名义判据（但 0 个过 FDR）"**。

用法：python judge_f1.py <batch.csv> [more.csv ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  （f1_family 在 merge 目录内）

R = Path(r"D:\.trae-cn\pythons\path\to\integrated_stock_selection"
         r"\results\merge_v2")


def bh_q(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    q = np.full(n, np.nan)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        i = n - rank
        j = order[i]
        if np.isfinite(p[j]):
            prev = min(prev, p[j] * n / (i + 1))
            q[j] = prev
    return q


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- 检验族硬约束（2026-10-12，Codex 要求）----
    # §8.4 规定检验族 = F1 ≤100 + F3 ≤10（同一族；F1-only 为过渡态），
    # 且「运行后不得修改…检验族」。但本函数原先对**输入文件的全部行**做 BH 校正，
    # 族规模完全由文件决定 ⇒ 把族从 720 扩到 1440 后 FDR 结论会随之改变而无人报警。
    # 现：**默认必须提供 --family <manifest>** 并逐一对应校验；不匹配即失败。
    argv = list(sys.argv[1:])
    family_manifest = None
    allow_reason = None
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--family":
            family_manifest = argv[i + 1]; i += 2; continue
        if argv[i] == "--allow-unregistered-family":
            allow_reason = argv[i + 1]; i += 2; continue
        files.append(argv[i]); i += 1

    if family_manifest is None and allow_reason is None:
        print("拒绝执行：未声明检验族。\n"
              "  §8.4 要求检验族预注册（F1 ≤100 + F3 ≤10，同一族），"
              "而 BH 校正的分母必须来自预注册清单，不能由输入文件决定。\n"
              "  用法：python judge_f1.py --family <manifest.json> <input.csv> ...\n"
              "  诊断/探索用途请显式声明：--allow-unregistered-family \"<理由>\""
              "（输出将标记 family_status=unregistered，FDR 不得作为正式结论）")
        return 3

    fam_rep = None
    if family_manifest is not None:
        import f1_family as FF
        man = json.loads(Path(family_manifest).read_text(encoding="utf-8"))
        # 用**并集**校验（多文件时应逐文件校验；此处按文件逐个校验）
        print(f"检验族清单：{family_manifest}  family_id={man.get('family_id')} "
              f"n_expected={man.get('n_expected')} hash={man.get('members_hash')}")
    else:
        man = None
        print(f"⚠ 未声明检验族（理由：{allow_reason}）：FDR 结果**不得**作为正式结论。")

    rc = 0
    for fn in files:
        p = Path(fn)
        if not p.is_file():
            print(f"缺文件 {fn}")
            continue
        d = pd.read_csv(p)

        if man is not None:
            import f1_family as FF
            fam_rep = FF.validate(man, FF.load_input_members(fn))
            d["family_id"] = man.get("family_id")
            d["family_status"] = "registered" if fam_rep["ok"] else "mismatch"
            d["family_members_hash"] = fam_rep["members_hash_input"]
            print(f"\n--- 族校验 {p.name} ---")
            print("  " + json.dumps({k: fam_rep[k] for k in
                  ("ok", "n_expected", "n_input", "n_duplicates", "n_missing",
                   "n_extra", "s_mismatch_bases", "over_spec_cap")},
                  ensure_ascii=False))
            if not fam_rep["ok"]:
                print("  ✗ 输入成员与预注册族**不匹配** —— 拒绝判定（不得静默扩大/缩小检验族）")
                print(f"    缺 {fam_rep['n_missing']} 例: {fam_rep['missing_sample']}")
                print(f"    多 {fam_rep['n_extra']} 例: {fam_rep['extra_sample']}")
                rc = 2
                continue
        else:
            d["family_id"] = None
            d["family_status"] = "unregistered"
            d["family_members_hash"] = None

        d["fdr_q"] = bh_q(d["placebo_p"].to_numpy(float))
        d["fdr_sig"] = d["fdr_q"] <= 0.10
        # 计划 §8.4 的三项要求
        d["ok_ci"] = d["delta_ci_lo"] > 0
        d["ok_blocks"] = d["nblocks_pos"] >= 4
        d["ok_placebo"] = d["placebo_p"] < 0.05
        d["h3_pass"] = d["ok_ci"] & d["ok_blocks"] & d["ok_placebo"]
        form = d["form"].iloc[0] if "form" in d.columns else "?"
        gate = d["gate"].iloc[0] if "gate" in d.columns else "?"
        ds = d["delta_mean"].to_numpy(float)
        print(f"\n=== {p.name}  gate={gate} form={form} ===")
        print(f"  n={len(d)}  delta中位 {np.median(ds):+.5f}  "
              f"正号 {int((ds > 0).sum())}/{len(d)}")
        print(f"  (a) CI下界>0      : {int(d['ok_ci'].sum())}/{len(d)}")
        print(f"  (b) nblocks>=4    : {int(d['ok_blocks'].sum())}/{len(d)}")
        print(f"  (c) placebo_p<0.05: {int(d['ok_placebo'].sum())}/{len(d)}")
        print(f"  ⇒ h3_pass（三项同时）: {int(d['h3_pass'].sum())}/{len(d)}")
        print(f"  联合 FDR 通过（q<=0.10）: {int(d['fdr_sig'].sum())}/{len(d)}"
              + ("" if man is not None else "   ⚠ 族未预注册，非正式结论"))
        w = d[d["h3_pass"]]
        if len(w):
            print("  通过明细:")
            for _, r in w.sort_values("delta_mean", ascending=False).iterrows():
                print(f"    {r['name']:<44s} delta={r['delta_mean']:+.5f} "
                      f"ci_lo={r['delta_ci_lo']:+.5f} p={r['placebo_p']:.4f} "
                      f"q={r['fdr_q']:.4f} blk={int(r['nblocks_pos'])}")
        out = p.with_name(p.stem + "_judged.csv")
        d.to_csv(out, index=False, encoding="utf-8")
        print(f"  → {out.name}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
