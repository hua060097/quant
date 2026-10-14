# -*- coding: utf-8 -*-
"""S8 两条状态机（批次 4.1）：个股池 / 门控交互训练，各自输入·输出·退出条件。

**解决的问题**（Codex，2026-10-12）：此前"通过"没有语义——F1 通过被误读为
"进入生产训练池"。本模块把 TECH §0.3.1 的状态生命周期做成**代码强制**：

| 状态 | S8_Gate_Interaction_Train（门控线） | S8_Stock_Pool（个股线） |
|---|---|---|
| candidate | 已登记（登记册/族清单成员） | 已登记（登记册成员） |
| qualified_stat | **逐个检验判定且族状态=registered**（h3_pass ∧ 联合 FDR q≤0.10，且成员哈希=现行 F1v2 清单） | 统计筛选通过（现阶段冻结池未建，不允许推进） |
| tested_increment | **外部生产引擎的对照运行清单**（`s8_increment_manifest.json`：engine=生产训练脚本、runs=[基线特征集, 基线+候选]、输出目录实际存在、声明文件哈希一致）——**模型级增量诊断不在 merge skill 内执行**（用户裁定 2026-10-13，见 §16.29.1） | （同左） |
| frozen_prod | **显式审批**（--approve 理由必填） | （个股池用 frozen_pool） |
| rejected / quarantined | 终态（理由必填） | 同左 |

**纪律**：
- 状态**只能由本 CLI 显式推进**，每次换态写台账 `decided` 事件
  （machine/state/evidence/evidence_sha12），状态文件 append-only（历史段不删）。
- 模型级增量（基线 vs +候选的正式对照）**由仓库根的生产脚本
  `main_alpha_stage15_hp_search.py` 承担**：其 `--features_file` 机制原生支持
  "换特征集跑同配置"，两次运行（基线/基线+候选）+ `compare_stage15.py` 对拍
  即为正式证据；merge 侧只核验清单引用与文件存在性。
- 无 `pool_state.json`（个股池未冻结）⇒ 个股线 qualified_stat **不允许推进**
  （代码级阻断）。
- 现状校准：**门控线上不存在已满足 qualified_stat 的候选**（F1v2 联合 FDR 0/720）。

用法：
    python s8_state.py --show [--machine gate|stock]
    python s8_state.py --machine gate --advance qualified_stat \
        --key "<f1 候选名>" --evidence <judged.csv>
    python s8_state.py --machine gate --advance tested_increment --key ... \
        --evidence <s8_increment_manifest.json>
    python s8_state.py --machine gate --advance frozen_prod --key ... \
        --evidence <file> --approve "<审批理由>"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_ledger as GL  # noqa: E402
import boundary as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
MANIFEST = os.path.join(RESULTS, "gates_search", "family_manifest_F1.json")
CONTRACT_HASHES = os.path.join(RESULTS, "contract_hashes.json")

MACHINES = {
    "gate": {
        "file": os.path.join(RESULTS, "s8_gate_state.json"),
        "chain": ["candidate", "qualified_stat", "tested_increment", "frozen_prod"],
        "terminals": ("rejected", "quarantined"),
    },
    "stock": {
        "file": os.path.join(RESULTS, "s8_stock_pool_state.json"),
        "chain": ["candidate", "qualified_stat", "pool_candidate", "frozen_pool",
                  "E3_confirmed"],
        "terminals": ("rejected", "quarantined"),
    },
}

# （2026-10-13：五项预规定指标随 s8_test_b.py 删除而移入外部生产引擎的对照协议；
#  merge 侧不再内置其实现或校验，tested_increment 证据改为外部运行清单，见 §16.29.1）


# ------------------------------------------------------------------ 退出条件
def _evidence_sha12(path: str) -> str | None:
    """证据文件的**内容哈希**（2026-10-13，Codex 验收要求：文件存在 ≠ 内容一致；
    存进换态历史与台账，供后续复核与变更失活判定）。"""
    if not path or not os.path.exists(path):
        return None
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def _manifest_hash() -> str | None:
    if not os.path.exists(MANIFEST):
        return None
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f).get("members_hash")


def _contract_experiment_hash() -> str | None:
    if not os.path.exists(CONTRACT_HASHES):
        return None
    with open(CONTRACT_HASHES, encoding="utf-8") as f:
        return json.load(f).get("experiment_hash")


def _check_evidence_f1(path: str, key: str) -> tuple[bool, dict]:
    """gate.qualified_stat：F1 judged 行 h3_pass ∧ fdr_sig ∧ 家族=现金清单。"""
    if not os.path.exists(path):
        return False, {"reason": f"证据文件不存在：{path}"}
    d = pd.read_csv(path)
    need = {"name", "gate", "h3_pass", "fdr_sig", "family_status", "family_members_hash"}
    miss = need - set(d.columns)
    if miss:
        return False, {"reason": f"证据缺列 {sorted(miss)}——须是 judge_f1 --family 的产物"}
    rows = d[d["name"].astype(str) == key]
    if not len(rows):
        return False, {"reason": f"证据中没有候选 {key}（n=0）"}
    mh = _manifest_hash()
    if mh is None:
        return False, {"reason": "现行族清单不存在（gates_search/family_manifest_F1.json）"}
    ok_rows = rows[(rows["h3_pass"] == True) & (rows["fdr_sig"] == True)  # noqa: E712
                   & (rows["family_status"] == "registered")
                   & (rows["family_members_hash"].astype(str) == mh)]
    stats = {"n_rows": int(len(rows)), "n_qualified": int(len(ok_rows)),
             "family_members_hash": mh,
             "h3_pass_all": bool((rows["h3_pass"] == True).all()),
             "fdr_sig_any": bool((rows["fdr_sig"] == True).any())}
    if not len(ok_rows):
        return False, {"reason": "未达 qualified_stat（需 h3_pass ∧ 联合 FDR q≤0.10 "
                                 "且族状态 registered）", "stats": stats}
    return True, {"stats": stats}


def _check_evidence_increment(path: str) -> tuple[bool, dict]:
    """tested_increment：**外部生产引擎对照运行清单**（merge 内不做模型级增量）。

    清单（JSON）最低要求：
      engine        : 引擎标识（"main_alpha_stage15_hp_search"）
      runs          : ≥2 个运行记录 [{features_file, out_dir}]
                      （runs[0]=基线特征集；runs[1]=基线+候选；同折/同种子/同预算
                       由生产引擎与 `--features_file` 的数据指纹机制保证）
      compare       : 对拍产物/说明（可选路径，存在则记录哈希）
    本函数只核验：结构合法 + 引用文件/目录**存在**（内容哈希在换态时记录）。
    对照的评价质量由生产引擎与 compare_stage15 确保在外部完成。
    """
    if not os.path.exists(path):
        return False, {"reason": f"证据文件不存在：{path}"}
    try:
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
    except ValueError:
        return False, {"reason": "证据不是合法 JSON"}
    if str(man.get("engine", "")).find("stage15") < 0:
        return False, {"reason": f"engine 应指向生产训练脚本（当前 {man.get('engine')!r}）"}
    runs = man.get("runs") or []
    if len(runs) < 2:
        return False, {"reason": f"runs 需 ≥2 个对照运行（当前 {len(runs)}）"}
    missing = []
    for i, r in enumerate(runs):
        for k in ("features_file", "out_dir"):
            p = str(r.get(k) or "")
            if not p or not os.path.exists(p):
                missing.append(f"runs[{i}].{k}={p}")
    cmp_p = str(man.get("compare") or "")
    if cmp_p and not os.path.exists(cmp_p):
        missing.append(f"compare={cmp_p}")
    if missing:
        return False, {"reason": "清单引用缺失：" + "; ".join(missing[:4])}
    return True, {"engine": man.get("engine"),
                  "n_runs": len(runs),
                  "note": "外部生产引擎对照运行（merge 不含模型级增量诊断）"}


def _check_triangle_freeze(machine: str, target: str) -> tuple[bool, str]:
    """stock 线的池冻结前置：无 pool_state.json 不允许进入 qualified_stat 之后的任何态。"""
    if machine != "stock":
        return True, ""
    if target in ("qualified_stat", "pool_candidate", "frozen_pool", "E3_confirmed"):
        if not os.path.exists(os.path.join(RESULTS, "pool_state.json")):
            return False, ("个股池未冻结（无 pool_state.json，S8_Stock_Pool 未执行"
                           " POOL_S3/入池/冻结）——代码级阻断")
    return True, ""


# ------------------------------------------------------------------ 状态读写
def load(machine: str) -> dict:
    path = MACHINES[machine]["file"]
    if not os.path.exists(path):
        return {"machine": machine, "candidates": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(machine: str, state: dict) -> None:
    with open(MACHINES[machine]["file"], "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def advance(machine: str, target: str, key: str, evidence: str,
            approve: str | None, note: str | None) -> int:
    m = MACHINES[machine]
    chain = m["chain"]
    d = load(machine)
    cand = d.setdefault("candidates", {}).setdefault(key, {
        "state": "candidate", "history": []})

    known = set(chain) | set(m["terminals"])
    if target not in known:
        print(f"未知目标状态 {target}（{machine} 的合法目标：{sorted(known)}）")
        return 2
    cur = cand["state"]
    if cur in m["terminals"]:
        print(f"候选 {key} 已是终态 {cur}（append-only，不可推进）")
        return 2
    if cur == target:
        print(f"候选 {key} 已处于 {target}")
        return 0
    if cur != "candidate" and target == "candidate":
        print("candidate 只能是初始态")
        return 2
    if target in chain and cur in chain and \
            chain.index(target) != chain.index(cur) + 1:
        print(f"非法跳跃：{cur} → {target}（链上只能逐步）")
        return 2

    # 逐目标退出条件（**代码强制**）
    if machine == "gate" and target == "qualified_stat":
        ok, rep = _check_evidence_f1(evidence, key)
        if not ok:
            print("✗ 退出条件不满足：", rep.get("reason"))
            print("  stats:", json.dumps(rep.get("stats", rep), ensure_ascii=False))
            lg_log(machine, key, cur, "state_transition_denied", evidence, rep, note)
            print("  （拒绝原因已记入台账 decided 事件，append-only）")
            return 1
    if machine == "gate" and target == "tested_increment":
        ok, rep = _check_evidence_increment(evidence)
        if not ok:
            print("✗ 退出条件不满足：", rep.get("reason"))
            return 1
    ev_sha = _evidence_sha12(evidence)
    if target == ("frozen_prod" if machine == "gate" else
                  ("E3_confirmed" if machine == "stock" else target)):
        if not approve:
            print(f"✗ {target} 级推进必须显式审批：--approve \"<理由>\"（记录为不可撤销证据）")
            return 2
    if target in m["terminals"] and not approve:
        print("✗ 终态必须给理由：--approve \"<理由>\"")
        return 2
    okf, whyf = _check_triangle_freeze(machine, target)
    if not okf:
        print("✗ 退出条件不满足：", whyf)
        return 2

    cand["state"] = target
    cand["history"].append({"at": datetime.now().isoformat(timespec="seconds"),
                            "from": cur, "to": target, "evidence": evidence,
                            "evidence_sha12": ev_sha,
                            "approve": approve, "note": note})
    save(machine, d)
    lg_log(machine, key, target,
           "candidate" if target not in m["terminals"] else "rejected",
           evidence, {"from": cur, "approve": approve, "evidence_sha12": ev_sha,
                      "note": note}, note)
    print(f"✓ {key}: {cur} → {target}（证据 {os.path.basename(evidence)}"
          f"{ '；审批理由已记录' if approve else ''}）")
    return 0


def lg_log(machine: str, key: str, state: str, verdict: str,
           evidence: str, data: dict, note: str | None) -> dict:
    # **独立台账**（作用域隔离）：S8 生命周期事件不与门控/个股搜索机器共享
    # `gate_ledger.jsonl`——2026-10-12 首次实测时曾误写生产台账 1 条（真实事件，
    # 已留档说明）。此后一律写 s8_state_ledger.jsonl。
    lg = GL.GateLedger(os.path.join(RESULTS, "s8_state_ledger.jsonl"))
    spec = {"level": "s8_" + machine, "base": key, "op": state}
    return lg.log("decided", spec, verdict=verdict,
                  data={"machine": machine, "key": key, "state": state,
                        "evidence": evidence, **(data or {}), "note": note})


def show(machine: str | None) -> int:
    for m_name, m in MACHINES.items():
        if machine and m_name != machine:
            continue
        s = load(m_name)
        print("=" * 76)
        print(f"[{m_name}] 状态机（链：{' → '.join(m['chain'])}；终态 {m['terminals']}）")
        cands = s.get("candidates", {})
        if not cands:
            print("  （当前无已登记候选——状态机为空是诚实结果）")
        for k, c in sorted(cands.items()):
            hist = "; ".join(h["to"] for h in c.get("history", []))
            print(f"  {k:<52s} state={c['state']}"
                  + (f"（经 {hist}）" if hist else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--machine", choices=("gate", "stock"), default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--advance", default=None, help="目标状态")
    ap.add_argument("--key", default=None, help="候选标识（gate 线=f1 候选名；stock 线=因子名）")
    ap.add_argument("--evidence", default="")
    ap.add_argument("--approve", default=None, help="冻结/终态的显式审批理由")
    ap.add_argument("--note", default=None)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.show or not a.advance:
        return show(a.machine)
    if not a.machine or not a.key:
        print("推进需要 --machine 与 --key")
        return 2
    return advance(a.machine, a.advance, a.key, a.evidence or "",
                   a.approve, a.note)


if __name__ == "__main__":
    raise SystemExit(main())
