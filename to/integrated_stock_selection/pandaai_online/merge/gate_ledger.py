# -*- coding: utf-8 -*-
"""门控指纹台账（MERGE_V2_PLAN §17.4/§17.8）。

**为什么需要**：进化搜索线有完整的三级指纹（`expression_hash` / `struct_fp` / `vfp`）
+ 跨 run 排除集（`ledger.hashes_for`）+ sidecar 增量摘要；门控线只有 `gate_id` 命名，
导致**跨批次重复评估**（每次重跑 54 个门控）。

**三级指纹**（§17.4）：
  - `candidate_fp`：候选身份 = 规格元组哈希
  - `family_fp`   ：近克隆家族 = (层级, 基础量) 哈希
  - `value_fp`    ：值等价（复用 `evaluator.value_fingerprint`，对 p 序列）

**候选身份不随测试窗口改变**；窗口/标签/口径属于 `caliber`（评估记录）。
台账据此判定每个候选的测试状态：

  completed  该 caliber 下已完成，可复用
  untested   未测，需执行
  stale      口径已变（窗口/形式/因子集/数据版本），需补测

**设计对齐 `ledger.py`**：append-only JSONL + sidecar 摘要（记 byte_offset，增量消费）。

用法：
    python gate_ledger.py --stats
    python gate_ledger.py --pending --caliber '{"window_end":"2025-12-31","form":"t"}'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluator import value_fingerprint  # noqa: E402

DEFAULT_PATH = os.path.join(HERE, "gate_ledger.jsonl")
SCHEMA = 1
SUMMARY_SCHEMA = 1

#: 构成候选身份的规格字段（不含 caliber）
SPEC_FIELDS = ("level", "base", "op", "window", "qwin", "agg", "form")

#: 构成 caliber 的字段（影响测试有效性；变了就要补测）
#: 2026-10-10 修正：`n_perm`/`n_boot` 必须计入——实测 G3 判定在 n_perm 100→60 时
#: **翻转**（True→False），即蒙特卡洛分辨率改变会改变结论；若不记入口径，
#: 台账会把两次不同分辨率的运行当成同一 caliber 而误判 `completed`。
CALIBER_FIELDS = ("window_end", "form", "factor_set", "label", "data_version",
                  "n_perm", "n_boot", "panel_sha")


def canonical_spec(spec: dict) -> dict:
    """只保留 SPEC_FIELDS，缺省补 None，保证哈希稳定。"""
    return {k: spec.get(k) for k in SPEC_FIELDS}


def candidate_fp(spec: dict) -> str:
    """候选身份指纹：规格元组。"""
    payload = json.dumps(canonical_spec(spec), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def family_fp(spec: dict) -> str:
    """近克隆家族：同一(层级,基础量)为一族（§17.4，Codex 建议按基础量）。"""
    payload = f"{spec.get('level')}|{spec.get('base')}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def gate_value_fp(p_values) -> str:
    """门控 p 序列的**边际分布**指纹（复用 `evaluator.value_fingerprint`）。

    ⚠ **不适用于门控线的等价判定**（2026-10-12 实测，见
    `results/merge_v2/gate_space_redundancy_audit_2026-10-12.md` §3.2）：

    `evaluator.value_fingerprint` 只取归一化后的 9 个分位点，是**边际分布**指纹，
    为**截面因子值**设计（对截面因子，仿射等价 ⇔ 同一排序 ⇔ 同一因子）。
    门控 p 是**时序**对象："何时开启"由时序对齐决定，**边际相同 ≠ 同一门控**。

    实测：行业门控 p = 逐日跨行业 `rank(pct=True)` → 边际恒为 [0,1] 均匀
    （std 恰 = 1/√12 = 0.2887，仅 131 个取值）→ **14 个门控只产出 2 个指纹**。
    若用作拒绝判据，会把 12/14 误判为重复。故本函数**仅保留作诊断记录**，
    等价判定一律用 `gate_order_fp`。
    """
    try:
        return value_fingerprint(np.asarray(p_values, dtype=float))
    except Exception:
        return ""


def gate_order_fp(p_values, min_n: int = 500) -> str:
    """门控序列的**排序指纹**（2026-10-12）：单调等价 ⇔ 同一门控。

    ## 语义
    先按 4 位有效数字量化（吸收浮点噪声），再取**秩排列**并哈希。
    两条序列秩排列相同 ⇔ 互为单调变换 ⇔ 门控开启/关闭时点相同 ⇔ 同一门控。

    ## 与 `gate_value_fp` 的区别
    - `gate_value_fp` 测**边际分布**（对 rank 归一化的行业门控退化为常数）→ 不可用
    - `gate_order_fp` 测**时序对齐**（正是"何时开启"这一研究关心的对象）→ 可用

    ## 不做镜像规范化
    `value_fingerprint` 会在 mean<0 时取负（等价 `x` 与 `−x`）。门控**不能**这样：
    `on = p ≥ 阈值`，故 `p` 与 `1−p` 的激活方向相反，**不是**同一门控。

    ## 不做量化（2026-10-12 修正）
    初版按"4 位有效数字"量化后再取秩，**破坏了秩不变性**：绝对舍入网格随量级变化
    （`x` 舍到 0.1、`3x+5` 舍到 1.0）→ 并列模式不同 → 秩不同 → 单调等价对**漏判**。
    合成夹具 T1 实测 FAIL。故**直接对原值取秩**：
    单调变换的秩在实数意义下严格相同（`3x+5`、`exp(x)` 与原序列同秩）。
    浮点噪声理论上可翻转极近值 → 表现为**漏判**（false negative），
    对"拒绝型"判据是保守方向（不会误拒真实候选），可接受。

    行数 <500 返回 ''（与 `value_fingerprint` 一致的样本量下限）。
    """
    v = np.asarray(p_values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < min_n:
        return ""
    ranks = np.empty(len(v), dtype=np.int64)
    ranks[np.argsort(v, kind="stable")] = np.arange(len(v), dtype=np.int64)
    return hashlib.sha1(ranks.tobytes()).hexdigest()[:12]


def structure_check(values, name: str = "", min_n: int = 500,
                    min_uniq: int = 10) -> tuple[bool, dict]:
    """输入结构自检（2026-10-12）：诊断/判据工具的**强制前置**质量控制。

    防复现已发生的"程序正常输出、结果无意义"错误（
    `results/merge_v2/gate_space_redundancy_audit_2026-10-12.md` 附录 A）：
      1) 把 `(date, sector)` 按日期均值压平 → std=0 → 假"同指纹"/假"13 簇"
         （已发生 2 次：G4 v1、L2a 初测）
      2) 用共线因子作效应基 → 假高相关（须在调用方显式声明分析口径）
      3) 未声明正则化的白化 → 结论随参数翻转（同上，属分析口径）

    本函数只查 1)（**对象结构**）；2)/3) 属分析口径，须由调用方显式声明，
    不能靠本函数兜底。

    返回 `(ok, report)`；`ok=False` 时调用方**必须拒绝输出结论**。
    """
    v = np.asarray(values, dtype=float)
    fin = np.isfinite(v)
    n = int(fin.sum())
    ok = True
    why: list[str] = []
    if n < min_n:
        ok = False
        why.append(f"非NaN仅 {n} < {min_n}")
    uq = int(len(np.unique(v[fin]))) if n else 0
    if uq < min_uniq:
        ok = False
        why.append(f"唯一值仅 {uq} < {min_uniq}（疑被压平/退化）")
    sd = float(np.std(v[fin])) if n else 0.0
    if not (sd > 0):
        ok = False
        why.append(f"std={sd:.6g}（常数序列）")
    return ok, {"name": name, "n": n, "uniq": uq, "std": sd,
                "ok": bool(ok), "why": why}


def caliber_key(caliber: dict) -> str:
    """评估口径指纹：变了即需补测。"""
    c = {k: (caliber or {}).get(k) for k in CALIBER_FIELDS}
    payload = json.dumps(c, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class GateLedger:
    def __init__(self, path: str | os.PathLike = DEFAULT_PATH):
        self.path = str(path)
        self.summary_path = self.path.replace(".jsonl", "_summary.json")
        self._summary: dict | None = None

    # ---------------------------------------------------------------- 写入
    def log(self, event: str, spec: dict, data: dict | None = None,
            caliber: dict | None = None, verdict: str | None = None) -> dict:
        """append-only 事件。event ∈ {seen, screened, g2g3, f1, decided}。"""
        cal = {k: (caliber or {}).get(k) for k in CALIBER_FIELDS}
        rec = {
            "schema": SCHEMA,
            "event": event,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "candidate_fp": candidate_fp(spec),
            "family_fp": family_fp(spec),
            "spec": canonical_spec(spec),
            "caliber": cal,
            "caliber_key": caliber_key(cal),
        }
        if data:
            rec["data"] = data
        if verdict:
            rec["verdict"] = verdict
        # 目录不存在则创建（2026-10-13：临时台账路径的目录缺失会让批处理
        # 半途崩溃，且曾连带导致产物被局部版本覆盖——见 PROGRESS 第二十轮）
        _d = os.path.dirname(self.path)
        if _d:
            os.makedirs(_d, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    # ------------------------------------------------------------- sidecar
    def _empty_summary(self) -> dict:
        return {
            "summary_schema": SUMMARY_SCHEMA,
            "byte_offset": 0,
            "n_events": 0,
            "events": {},
            "candidates": {},   # fp -> {spec, family_fp, value_fp, calibers:{k:{...}}}
            "families": {},     # family_fp -> [candidate_fp]
            "value_fps": {},    # value_fp -> [candidate_fp]
            "calibers": {},     # caliber_key -> {n, label}
        }

    def _ingest(self, s: dict, recs: list[dict]) -> None:
        for rec in recs:
            ev = rec.get("event")
            s["events"][ev] = s["events"].get(ev, 0) + 1
            s["n_events"] += 1
            fp = rec.get("candidate_fp")
            if not fp:
                continue
            cand = s["candidates"].setdefault(fp, {
                "spec": rec.get("spec"), "family_fp": rec.get("family_fp"),
                "value_fp": "", "first_seen": rec.get("created_at"),
                "calibers": {}, "verdict": None,
            })
            if rec.get("spec"):
                cand["spec"] = rec["spec"]
            fam = rec.get("family_fp")
            if fam:
                cand["family_fp"] = fam
                bucket = s["families"].setdefault(fam, [])
                if fp not in bucket:
                    bucket.append(fp)
            d = rec.get("data") or {}
            vfp = d.get("value_fp")
            if vfp:
                cand["value_fp"] = vfp
                vb = s["value_fps"].setdefault(vfp, [])
                if fp not in vb:
                    vb.append(fp)
            ck = rec.get("caliber_key")
            if ck:
                cal = cand["calibers"].setdefault(ck, {
                    "caliber": rec.get("caliber"), "events": [],
                    "verdict": None, "last_at": rec.get("created_at")})
                if ev not in cal["events"]:
                    cal["events"].append(ev)
                cal["last_at"] = rec.get("created_at")
                if rec.get("verdict"):
                    cal["verdict"] = rec["verdict"]
                    cand["verdict"] = rec["verdict"]
                s["calibers"][ck] = {"n": s["calibers"].get(ck, {}).get("n", 0) + 1,
                                     "label": rec.get("caliber")}

    def _read_from(self, offset: int):
        if not os.path.exists(self.path) or os.path.getsize(self.path) <= offset:
            return [], offset
        with open(self.path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        parts = chunk.split(b"\n")
        recs = []
        for raw in parts[:-1]:
            line = raw.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line.decode("utf-8")))
            except Exception:
                continue
        consumed = sum(len(p) + 1 for p in parts[:-1])
        return recs, offset + consumed

    def _save(self, s: dict) -> None:
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, default=str)
        os.replace(tmp, self.summary_path)

    def summary(self) -> dict:
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        if self._summary is not None and self._summary["byte_offset"] == size:
            return self._summary
        s = self._empty_summary()
        if os.path.exists(self.summary_path):
            try:
                with open(self.summary_path, encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("summary_schema") == SUMMARY_SCHEMA \
                        and prev.get("byte_offset", 0) <= size:
                    s = prev
            except Exception:
                s = self._empty_summary()
        recs, off = self._read_from(s["byte_offset"])
        if recs:
            self._ingest(s, recs)
            s["byte_offset"] = off
        self._summary = s
        if recs:
            self._save(s)
        return s

    # ------------------------------------------------------- 状态与查询
    def state(self, spec: dict, caliber: dict) -> str:
        """返回 completed / untested / stale。"""
        s = self.summary()
        fp = candidate_fp(spec)
        cand = s["candidates"].get(fp)
        if cand is None:
            return "untested"
        ck = caliber_key(caliber)
        if ck in cand["calibers"]:
            return "completed"
        return "stale"

    def pending(self, specs: list[dict], caliber: dict) -> list[dict]:
        """筛选出需要执行的候选（untested 或 stale），并附上状态。"""
        out = []
        for spec in specs:
            st = self.state(spec, caliber)
            if st != "completed":
                out.append({**spec, "_state": st})
        return out

    def tested_pairs(self, prefix: str = "gate") -> int:
        """多重检验分母：唯一 (candidate_fp, caliber_key) 对中跑过 f1 的。"""
        s = self.summary()
        n = 0
        for fp, cand in s["candidates"].items():
            for ck, cal in cand["calibers"].items():
                if "f1" in cal.get("events", []):
                    n += 1
        return n

    def stats(self) -> dict:
        s = self.summary()
        fam_multi = sum(1 for v in s["families"].values() if len(v) > 1)
        val_dup = sum(len(v) - 1 for v in s["value_fps"].values() if len(v) > 1)
        return {
            "n_events": s["n_events"],
            "events": dict(s["events"]),
            "n_candidates": len(s["candidates"]),
            "n_families": len(s["families"]),
            "families_with_multi": fam_multi,
            "value_fp_duplicates": val_dup,
            "n_calibers": len(s["calibers"]),
            "tested_f1_pairs": self.tested_pairs(),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="门控指纹台账")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--pending", action="store_true",
                    help="列出给定 caliber 下待执行的候选（需配合 --specs）")
    ap.add_argument("--specs", default="", help="规格 JSON 文件（列表）")
    ap.add_argument("--caliber", default="{}", help="caliber JSON")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    lg = GateLedger(a.path)
    if a.stats or (not a.pending):
        st = lg.stats()
        print(json.dumps(st, ensure_ascii=False, indent=1))
    if a.pending:
        if not a.specs or not os.path.isfile(a.specs):
            print("需要 --specs <json>")
            return 1
        specs = json.loads(open(a.specs, encoding="utf-8").read())
        cal = json.loads(a.caliber)
        pend = lg.pending(specs, cal)
        print(f"caliber={cal}")
        print(f"待执行 {len(pend)}/{len(specs)}")
        from collections import Counter
        print("状态分布:", dict(Counter(p["_state"] for p in pend)))
        for p in pend[:20]:
            print(f"  [{p['_state']}] {p.get('level')}/{p.get('base')}/"
                  f"{p.get('op')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
