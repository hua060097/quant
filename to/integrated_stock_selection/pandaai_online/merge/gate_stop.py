# -*- coding: utf-8 -*-
"""门控搜索停止规则（MERGE_V2_PLAN §17.7）——**与进化搜索保持一致的简单口径**。

## 进化搜索的停止机制（已核实 `run_search.py`）

| 参数 | 语义 | 默认 |
|---|---|---|
| `--n N` | 撒种候选数上限——**硬预算**，生成完即停 | 40 |
| `--full-cap C` | 进入昂贵全评的候选数上限——硬预算 | args |
| `--stall K` | 连续 K 个候选无改进即早停（`_note_score`）——**可选** | 0（关） |

即：**主要靠"指定轮数/预算，跑完即停"；早停是可选的辅助。**

## 本模块的口径（同一套语义 + 一条方向级暂停）

1. **硬预算**：`n_seeds`（一批生成数）、`full_cap`（进昂贵层数）——与 `--n`/`--full-cap` 同构。
2. **可选早停**：`stall`——与 `--stall` 同构，默认关。
3. **方向级暂停**：连续 `keeper_rate_batches` 批 keeper 率 < `keeper_rate_min`
   → **暂停该方向并诊断**（不自动永久终止；批次过小不按比例判定）。

## keeper 定义（Codex 要求：不得混用三条线）

| 线 | keeper 定义 |
|---|---|
| 门控线 | 通过 G2/G3 **且** 方向中性（= 有资格生成交互候选） |
| 个股线 | 快评达保留标准、可进下一轮（`fast_keep_sia` 等） |
| 正式入池 | 完成正式统计检验与确认——**不是 keeper** |

用法：
    python gate_stop.py --status
    python gate_stop.py --record --n 40 --keeper 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "gate_stop_state.json")


@dataclass
class StopConfig:
    n_seeds: int = 40            # 一批生成的候选数上限（对应 --n）
    full_cap: int = 12           # 进入昂贵层的上限（对应 --full-cap）
    stall: int = 0               # 连续无改进早停，0=关（对应 --stall）
    keeper_rate_min: float = 0.10
    keeper_rate_batches: int = 2
    min_batch_for_rate: int = 20  # 批次小于此不按比例判定（比例波动太大）


@dataclass
class StopState:
    cfg: StopConfig = field(default_factory=StopConfig)
    best: float = -1.0
    stall_count: int = 0
    batches: list = field(default_factory=list)
    seed_budget_used: int = 0
    full_budget_used: int = 0

    # ------------------------------------------------------------ 早停（同 --stall）
    def note(self, score: float) -> bool:
        """报告一个候选的分数。返回 True 表示触发早停。"""
        if score > self.best:
            self.best, self.stall_count = score, 0
            return False
        self.stall_count += 1
        return bool(self.cfg.stall and self.stall_count >= self.cfg.stall)

    # ------------------------------------------------------------ 硬预算（同 --n）
    def budget_left(self, seed_used: int | None = None,
                    full_used: int | None = None) -> dict:
        su = self.seed_budget_used if seed_used is None else seed_used
        fu = self.full_budget_used if full_used is None else full_used
        return {"n_seeds_left": max(self.cfg.n_seeds - su, 0),
                "full_cap_left": max(self.cfg.full_cap - fu, 0)}

    # -------------------------------------------------- 批次记录 + 方向级暂停
    def record_batch(self, n: int, n_keeper: int) -> tuple[bool, str]:
        """记录一批。返回 (pause_direction, reason)。

        注意：这里的"暂停"是**方向级**的，不是永久终止——
        低 keeper 率也可能是生成器缺乏多样性或预筛过严，须人工诊断。
        """
        rate = (n_keeper / n) if n else float("nan")
        self.batches.append({"n": int(n), "n_keeper": int(n_keeper),
                             "keeper_rate": rate})
        self.seed_budget_used += int(n)
        self.full_budget_used += int(n_keeper)
        k = self.cfg.keeper_rate_batches
        if len(self.batches) < k:
            return False, ""
        recent = self.batches[-k:]
        if any(b["n"] < self.cfg.min_batch_for_rate for b in recent):
            return False, "批次过小，不按比例判定"
        rates = [b["keeper_rate"] for b in recent]
        if all(r < self.cfg.keeper_rate_min for r in rates):
            return True, (f"连续 {k} 批 keeper 率 "
                          f"{['%.3f' % r for r in rates]} < "
                          f"{self.cfg.keeper_rate_min} → 暂停该方向并诊断")
        return False, ""

    def pause_status(self) -> tuple[bool, str]:
        """只读检查：当前批次历史是否会触发方向级暂停（不改状态）。"""
        k = self.cfg.keeper_rate_batches
        if len(self.batches) < k:
            return False, f"批次不足 {k}"
        recent = self.batches[-k:]
        if any(b["n"] < self.cfg.min_batch_for_rate for b in recent):
            return False, "批次过小，不按比例判定"
        rates = [b["keeper_rate"] for b in recent]
        if all(r < self.cfg.keeper_rate_min for r in rates):
            return True, (f"连续 {k} 批 keeper 率 "
                          f"{['%.3f' % r for r in rates]} < "
                          f"{self.cfg.keeper_rate_min}")
        return False, "正常"

    # ------------------------------------------------------------ 持久化
    def save(self, path: str = DEFAULT_PATH) -> None:
        d = asdict(self)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "StopState":
        if not os.path.isfile(path):
            return cls()
        try:
            d = json.loads(open(path, encoding="utf-8").read())
            cfg = StopConfig(**d.pop("cfg", {}))
            return cls(cfg=cfg, **d)
        except Exception:
            return cls()

    def report(self) -> str:
        lines = ["停止规则状态（与进化搜索同构：硬预算 + 可选早停）",
                 f"  预算: n_seeds {self.seed_budget_used}/{self.cfg.n_seeds}  "
                 f"full_cap {self.full_budget_used}/{self.cfg.full_cap}",
                 f"  早停: stall={self.stall_count}/"
                 f"{self.cfg.stall or '关'}  best={self.best:.4f}",
                 f"  批次: {len(self.batches)}"]
        for i, b in enumerate(self.batches, 1):
            lines.append(f"    #{i}  n={b['n']:<4d} keeper={b['n_keeper']:<3d} "
                         f"rate={b['keeper_rate']:.3f}")
        pause, why = self.pause_status()
        lines.append(f"  方向级暂停：{'是' if pause else '否'} — {why}")
        return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="门控搜索停止规则状态")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="本批候选数")
    ap.add_argument("--keeper", type=int, default=0, help="本批 keeper 数")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.reset:
        if os.path.isfile(a.path):
            os.remove(a.path)
        print("已重置")
        return 0
    st = StopState.load(a.path)
    if a.record:
        pause, why = st.record_batch(a.n, a.keeper)
        st.save(a.path)
        print(f"记录一批：n={a.n} keeper={a.keeper}")
        print(f"  → {'暂停方向' if pause else '继续'}：{why or '正常'}")
    print(st.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
