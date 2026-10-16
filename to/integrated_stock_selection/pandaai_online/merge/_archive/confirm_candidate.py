# -*- coding: utf-8 -*-
"""P1：GM_limit_ratio_ratio 正式确认（在冻结判据下补齐 F1）。

按 Codex 的六条要求执行：
  1. 先查台账，确认已完成哪些步骤（**已完成**：screened / g2g3，F1 事件=0）
  2. 标记为「搜索幸存者，待正式确认」，记录选择偏差来源
  3. 冻结判据后补齐 F1/ΔIC（**不因唯一幸存者改阈值**）
  4. 明确多重检验校正覆盖的候选范围
  5. 同一批数据上的旧 F1 结果只作探索性依据（代码重跑不产生独立证据）
  6. 记录完整结果与口径

产出：
  - provenance 事件（含选择偏差标注）写入 gate_ledger
  - f1 事件（含口径 + "同批数据"标注）写入 gate_ledger
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_ledger as GL  # noqa: E402
import gates as G  # noqa: E402
import boundary as B  # noqa: E402
import pandas as pd  # noqa: E402
import judge_f1 as JF  # noqa: E402
import windows as W  # noqa: E402

ROOT = HERE.parent.parent
RESULTS = ROOT / "results" / "merge_v2"
SPACE = RESULTS / "gates_search"
CUT = W.FACTOR_SCREEN_MAX

TARGET = {"level": "market", "base": "limit_ratio", "op": "ratio"}
GATE_ID = "GM_limit_ratio_ratio"

#: F1 口径（f1_batch 的 N_PERM=200；h3_eval.delta_ic 的块自助 2000 次）
F1_CALIBER = {"window_end": CUT, "form": "t", "factor_set": "base20",
              "label": "r3", "data_version": B.factor_data_version(),
              "n_perm": 200, "n_boot": 2000}

PROVENANCE = (
    "**搜索幸存者，待正式确认。** 该候选从本批市场门控候选中经预设资格筛选保留"
    "（54 规格 → 36 eligible → 40 全跑 → G2/G3 + 方向中性度 → 短名单 1 个）。"
    "方向分层结果显示其低/中/高三个市场方向层内 ΔIC 均为正，具进一步验证价值。"
    "但该结果属于**搜索后的候选选择，存在选择偏差**，不得把当前观测效应直接视为"
    "无偏的样本外效果。后续按冻结口径完成剩余检验，且**不得因其为本批唯一幸存者"
    "而调整判定阈值**。最终状态以完整检验记录与独立确认结果为准。"
)

SCOPE = (
    "**多重检验校正覆盖范围**：门控资格层 G2/G3 的检验族 = 本批 40 个市场门控候选"
    "（caliber 5144d212554b）；交互候选层 F1 的检验族 = 该门控 × base20 = 20 个候选。"
    "两层为**层级式检验**：G2/G3 前置筛选缩小了 F1 的检验范围，但**不自动保证"
    "F1 的 FDR 结论仍有效**（前置筛选本身消耗了多重性预算）。"
    "FDR 族**不与个股线合并**（来源与假设族不同）。"
)

SAME_DATA_NOTE = (
    "**同一批数据的探索性重跑**：2026-10-10 曾在台账建立前、于**同一窗口/同一形式/"
    "同一因子集**上跑过一次 F1（`results/merge_v2/gates_search/f1_market_search_t.csv`，720 候选）。"
    "本次为在台账口径下的正式记录，**不构成独立的新检验**——代码重跑不使旧数据"
    "变为独立证据。"
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lg = GL.GateLedger()

    # ---- 1) 记录 provenance（含选择偏差标注）----
    lg.log("provenance", TARGET, verdict="survivor_pending",
           data={"gate_id": GATE_ID, "note": PROVENANCE, "scope": SCOPE})
    print("[1] provenance 已记录（搜索幸存者，待正式确认）")

    # ---- 2) 生成候选并跑 F1（冻结口径）----
    b20 = json.loads((RESULTS / "base20.json").read_text(encoding="utf-8"))["base20"]
    cands = [{"name": f"{x['factor']}__{GATE_ID}__pos", "base": x["factor"],
              "gate": GATE_ID, "gate_type": "market", "s": 1,
              "d": int(x["direction"])} for x in b20]
    cp = SPACE / "cands_confirm_target.json"
    cp.write_text(json.dumps(cands, ensure_ascii=False), encoding="utf-8")
    out = SPACE / "f1_confirm_target.csv"
    if out.exists():
        out.unlink()
    gp = SPACE / "market_gates_search.parquet"
    print(f"[2] 运行 F1（{len(cands)} 个候选，form=t，label_max={CUT}）…")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "f1_batch.py", "--cands", str(cp),
         "--out", str(out), "--tag", "confirm", "--form", "t",
         "--gates-parquet", str(gp)],
        cwd=str(HERE), capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    print(f"    rc={r.returncode}  {time.time() - t0:.0f}s")
    if r.returncode != 0:
        print((r.stderr or "")[-800:])
        return 1

    # ---- 3) 判定（§8.4 三项 + FDR）----
    d = pd.read_csv(out)
    d["fdr_q"] = JF.bh_q(d["placebo_p"].to_numpy(float))
    d["fdr_sig"] = d["fdr_q"] <= 0.10
    d["ok_ci"] = d["delta_ci_lo"] > 0
    d["ok_blocks"] = d["nblocks_pos"] >= 4
    d["ok_placebo"] = d["placebo_p"] < 0.05
    d["h3_pass"] = d["ok_ci"] & d["ok_blocks"] & d["ok_placebo"]
    d.to_csv(SPACE / "f1_confirm_target_judged.csv", index=False,
             encoding="utf-8")

    n_pass = int(d["h3_pass"].sum())
    n_fdr = int((d["h3_pass"] & d["fdr_sig"]).sum())
    print(f"\n[3] 判定：h3_pass {n_pass}/{len(d)}；其中过 FDR(q≤0.10) {n_fdr}/{len(d)}")
    print(f"    delta 中位 {d['delta_mean'].median():+.5f}  "
          f"正号 {int((d['delta_mean'] > 0).sum())}/{len(d)}")
    print(f"    (a)CI下界>0 {int(d['ok_ci'].sum())}  "
          f"(b)≥4块 {int(d['ok_blocks'].sum())}  "
          f"(c)p<0.05 {int(d['ok_placebo'].sum())}")

    # ---- 4) 回写台账（含口径 + 同批数据标注）----
    lg.log("f1", TARGET, caliber=F1_CALIBER,
           verdict=("confirmed_by_f1" if (n_pass and n_fdr) else
                    "f1_partial" if n_pass else "f1_none"),
           data={"n_candidates": len(d), "h3_pass": n_pass, "fdr_pass": n_fdr,
                 "delta_median": float(d["delta_mean"].median()),
                 "n_pos": int((d["delta_mean"] > 0).sum()),
                 "note": SAME_DATA_NOTE,
                 "test_family": "gate_interaction_F1",
                 "scope": "该门控 × base20 = 20 个候选"})
    print("[4] f1 事件已回写台账（含口径与「同批数据」标注）")
    print(f"\n台账统计 → {json.dumps(lg.stats(), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
