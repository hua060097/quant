# -*- coding: utf-8 -*-
"""F1 检验族清单（manifest）：构建 / 校验 / 成员哈希。

**背景（2026-10-12，Codex 要求）**：§8.4 规定「检验族 = 本轮全部 F1 候选
（≤100：5 门控 × 20 基因子）**+ F3（≤10，需预先登记配对）**」，
且「运行后不得修改定义、阈值、块长、置换次数或**检验族**」。
但 `judge_f1.bh_q` 此前对**输入文件的全部行**做 BH 校正 ——
**族规模完全由文件决定，没有任何校验**。实测后果：本轮把族从 720 扩到 1440 后
FDR 结论随之改变（0/720 → 0/1440），而代码不会报警。这是**机制性缺陷**。

**F1 与 F3 是同一检验族还是两个？** 以规格原文为准（不自行选择）：
§8.4 写作「检验族 = 本轮全部 F1 候选（≤100：…）**+ F3（≤10，需预先登记配对）**」，
且 §8.4「F1-only 过渡规则（v2.8.7）」写明「F3 尚未构造，本轮 FDR 族**暂时只含 F1**，
结论标注『F1-only』。**F3 到达后重算联合 FDR**，不得沿用 F1-only 的调整后 p 值」。
⇒ **同一族**；F1-only 是**过渡态**，不是独立族。

**本模块提供机器可执行的族校验**：
  - 清单由 `--from-registry` 从 `hypothesis_registry.csv` 的 F1 族生成（预注册来源）
  - 校验输入 CSV 的成员 `(gate, base, s)` 与清单**逐一对应**：
    多、少、方向不符、重复 —— 全部失败（不得静默扩大或缩小检验族）
  - 输出成员清单哈希，供结果文件留痕

用法：
    python f1_family.py --from-registry                    # 生成 F1 族清单
    python f1_family.py --validate <manifest> <input.csv>  # 校验
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
DEFAULT_MANIFEST = os.path.join(RESULTS, "gates_search", "family_manifest_F1.json")

#: §8.4 规格的族上限（F1 ≤100 + F3 ≤10）。**版本化**，不永久硬编码为唯一保障。
#: v1 = 旧手工门控世代（5 门控 × 20 = 100，已删除）；v2 = 搜索世代。
SPEC_F1_MAX = 100
SPEC_F3_MAX = 10
#: v2 世代（搜索族）规格上限 = 冻结空间 eligible 门控 × base 因子数（§16.27）。
#: 2026-10-12 冻结时点 = 36 × 20 = 720。族规模变化必须**换新世代/新清单**，
#: 不得沿用旧哈希（validate 逐一对应校验兜底）。
SPEC_SEARCH_MAX = 720
MANIFEST_VERSION = "2026-10-12"


def _sig(expected_sign: str) -> int:
    """登记册的 `expected_sign` → `s`。'+' → +1；'-' → -1；其余（如 F2 的「同基因子」）不适用。"""
    t = str(expected_sign).strip()
    if t in ("+", "1", "+1"):
        return 1
    if t in ("-", "-1"):
        return -1
    raise ValueError(f"无法把 expected_sign={expected_sign!r} 解析为 s（±1）")


def members_hash(members: list[tuple]) -> str:
    """成员清单哈希：对 `(gate, base, s)` 排序后规范化再 sha1（前 12 位）。"""
    canon = "\n".join("%s\x00%s\x00%d" % (g, b, int(s))
                      for g, b, s in sorted(members))
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def build_from_registry(registry: str, family: str = "F1", gen: int = 2,
                        spec_cap: int | None = None) -> dict:
    """从假设登记册生成族清单（**预注册来源**，不是从结果文件反推）。

    `gen`：世代号。v1 = 旧手工门控世代（§8.4 原文 ≤100，产物已删除）；
    v2 = 搜索世代（冻结空间 eligible × base，§16.27 预注册，s=+1）。
    `spec_cap`：显式覆盖（默认按世代取 SPEC_F1_MAX / SPEC_SEARCH_MAX）。
    """
    import csv
    rows = []
    with open(registry, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("family") or "").strip() != family:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"登记册 {registry} 里没有 family={family} 的条目")
    members = []
    for r in rows:
        members.append((r["gate"], r["primitives"], _sig(r["expected_sign"])))
    dup = len(members) - len(set(members))
    mh = members_hash(members)
    gates = sorted({m[0] for m in members})
    bases = sorted({m[1] for m in members})
    sset = sorted({m[2] for m in members})
    cap = int(spec_cap if spec_cap is not None else
              (SPEC_F1_MAX if gen == 1 else SPEC_SEARCH_MAX))
    return {
        "family_id": f"{family}v{gen}",
        "family_version": MANIFEST_VERSION,
        "spec_ref": ("PLAN §8.4（检验族 = F1 ≤100 + F3 ≤10，同一族；F1-only 为过渡态）"
                     if gen == 1 else
                     "PLAN §16.27（搜索世代：冻结空间 eligible 门控 × base20，s=+1；"
                     "F3 未构造，F1-only 过渡态——F3 到达后重算联合 FDR）"),
        "registry": os.path.basename(registry),
        "n_expected": len(members),
        "n_duplicates": dup,
        "gates": gates,
        "n_bases": len(bases),
        "s_values": sset,
        "spec_cap": cap,
        "members_hash": mh,
        "members": [{"gate": g, "base": b, "s": int(s)} for g, b, s in members],
        "note": ("s 由登记册 expected_sign 预注册；双向（s=-1）不在本族内。"
                 if gen == 1 else
                 "v2 搜索世代：s=+1 预注册（--both-directions 双向不在本族内，属探索）。"),
    }


def build_from_cands(cands_path: str, family: str = "F1v3",
                     form: str = "gain") -> dict:
    """从cands JSON直接构建新族清单（F1v3_gain/flip/switch，不碰登记册与旧族）。

    成员键仍为 (gate, base, s)（switch的base取"off__def"复合键），故
    `validate`/`judge_f1`逐一对应逻辑零改动；form记入manifest元信息，
    不同form不得合并FDR（由调用方分开跑judge）。
    """
    cands = json.load(open(cands_path, encoding="utf-8"))
    members = []
    for c in cands:
        base = c.get("base") or (
            f"{c.get('base_off')}__{c.get('base_def')}" if c.get("base_off") else None)
        if not c.get("gate") or not base:
            raise SystemExit(f"cands条目缺gate/base: {c.get('name')}")
        members.append((str(c["gate"]), str(base), int(c.get("s", 1))))
    dup = len(members) - len(set(members))
    mh = members_hash(members)
    return {
        "family_id": f"{family}_{form}",
        "family_version": MANIFEST_VERSION,
        "spec_ref": ("PLAN §21（F1v3新族：bank门控 × 偏科/机制代表因子；"
                      "form=%s，与F1v2平行对照，不得合并FDR）" % form),
        "registry": os.path.basename(cands_path),
        "n_expected": len(members),
        "n_duplicates": dup,
        "gates": sorted({m[0] for m in members}),
        "n_bases": len({m[1] for m in members}),
        "s_values": sorted({m[2] for m in members}),
        "spec_cap": len(members),
        "form": form,
        "members_hash": mh,
        "members": [{"gate": g, "base": b, "s": int(s)} for g, b, s in members],
        "note": "成员来自cands JSON原文；switch复合base键保持逐一对应可校验。",
    }


def load_input_members(path: str) -> list[tuple]:
    """读 F1 结果/候选 CSV，抽出成员 `(gate, base, s)`。"""
    import pandas as pd
    d = pd.read_csv(path)
    need = {"gate", "base", "s"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(f"{path} 缺列 {sorted(missing)}；无法做族校验")
    return [(str(r["gate"]), str(r["base"]), int(r["s"])) for _, r in d.iterrows()]


def validate(manifest: dict, members: list[tuple]) -> dict:
    """逐一对应校验。返回含 ok 与四类差异的报告（不抛错，由调用方决定如何失败）。"""
    exp = {(m["gate"], m["base"], int(m["s"])) for m in manifest["members"]}
    got = set(members)
    dup = len(members) - len(got)
    missing = sorted(exp - got)          # 清单有、输入无
    extra = sorted(got - exp)            # 输入有、清单无（含方向不符）
    s_mismatch = sorted({(g, b) for (g, b, s) in extra
                         if any((g, b, s2) in exp for s2 in (1, -1))})
    n_exp = int(manifest["n_expected"])
    ok = (not missing and not extra and dup == 0 and len(members) == n_exp)
    return {
        "ok": bool(ok),
        "family_id": manifest.get("family_id"),
        "members_hash_expected": manifest.get("members_hash"),
        "members_hash_input": members_hash(members),
        "n_expected": n_exp,
        "n_input": len(members),
        "n_duplicates": dup,
        "n_missing": len(missing),
        "n_extra": len(extra),
        "missing_sample": missing[:8],
        "extra_sample": extra[:8],
        "s_mismatch_bases": s_mismatch[:8],
        "over_spec_cap": bool(n_exp > int(manifest.get("spec_cap", SPEC_F1_MAX))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-registry", action="store_true",
                    help="从 hypothesis_registry.csv 生成 F1 族清单")
    ap.add_argument("--registry", default=os.path.join(RESULTS, "hypothesis_registry.csv"))
    ap.add_argument("--family", default="F1")
    ap.add_argument("--gen", type=int, default=2,
                    help="世代号（1=旧手工门控 §8.4 原文; 2=搜索世代 §16.27）")
    ap.add_argument("--spec-cap", type=int, default=None,
                    help="族上限覆盖（默认 v1=100 / v2=720）")
    ap.add_argument("--out", default=DEFAULT_MANIFEST)
    ap.add_argument("--from-cands", default=None,
                    help="从cands JSON构建F1v3新族清单（配合--family/--form）")
    ap.add_argument("--form", default="gain",
                    help="新族组合形式注记（gain/flip/switch，不改变成员键）")
    ap.add_argument("--validate", nargs=2, metavar=("MANIFEST", "INPUT_CSV"))
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.from_registry:
        m = build_from_registry(a.registry, family=a.family, gen=a.gen,
                                spec_cap=a.spec_cap)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print("族清单 → %s" % a.out)
        print("  family_id=%s  n_expected=%d  门控=%s  s=%s  hash=%s  spec_cap=%d"
              % (m["family_id"], m["n_expected"], m["gates"], m["s_values"],
                 m["members_hash"], m["spec_cap"]))
        if m["n_duplicates"]:
            print("  ⚠ 登记册内有 %d 条重复成员" % m["n_duplicates"])
        if m["n_expected"] > m["spec_cap"]:
            print("  ⚠ 超出该世代规格上限 %d（须换新世代/新清单，不得沿用旧哈希）"
                  % m["spec_cap"])
        return 0

    if a.from_cands:
        m = build_from_cands(a.from_cands, family=a.family, form=a.form)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        print("新族清单 → %s" % a.out)
        print("  family_id=%s  n_expected=%d  hash=%s form=%s"
              % (m["family_id"], m["n_expected"], m["members_hash"], m["form"]))
        if m["n_duplicates"]:
            print("  ⚠ cands内有 %d 条重复成员" % m["n_duplicates"])
        return 0

    if a.validate:
        man, inp = a.validate
        m = json.loads(open(man, encoding="utf-8").read())
        rep = validate(m, load_input_members(inp))
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0 if rep["ok"] else 2

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
