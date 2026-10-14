# -*- coding: utf-8 -*-
"""给"已搜索过的候选"打上搜索窗口标签（用户 2026-10-10 要求）。

动机：此前所有 P0/P1 搜索一律读到 `label_max=2024-12-26`，但该上界只散落在
各脚本默认值与 `base20.json` 里，**没有记录下界**，导致无法回答"某个因子的
证据来自哪段数据"，也无法安全地让新搜索使用更大数据范围。

本脚本：
1. 从 `boundary_log.jsonl` 反查每批搜索真实用过的窗口（不硬编码）；
2. 产出 `search_registry.csv`（每个搜索产物一行：窗口、证据等级、数据源）；
3. 给各候选级产物**追加** `search_start`/`search_end`/`label_available_max`/
   `evidence` 四列（纯追加，不改动任何既有列与取值）；
4. 原文件先备份到 `_untagged_backup/`，可一键还原。

用法：
    python tag_search_windows.py            # 打标 + 备份
    python tag_search_windows.py --verify   # 只校验，不写
    python tag_search_windows.py --restore  # 从备份还原
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import windows as W  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "results" / "merge_v2"
LOG = RESULTS / "boundary_log.jsonl"
BACKUP = RESULTS / "_untagged_backup"

# 候选级产物：文件 → (键列, 说明)
CANDIDATE_ARTIFACTS: dict[str, tuple[str, str]] = {
    "roles.csv": ("name", "CURATED 174 角色标注"),
    "h2_stock_all.csv": ("factor", "124 个 STOCK 因子的 H2 记录"),
    "h2_joint.csv": ("factor", "124 STOCK + 32 SECTOR + 40 F2 的 H2 联合记录"),
    "h2_stock.csv": ("factor", "STOCK 子集 H2"),
    "f1_joint_judged.csv": ("name", "100 个 F1 交互候选的 H3 联合判定"),
    "f1_market_judged.csv": ("name", "市场线 F1 判定"),
    "hypothesis_registry.csv": ("id", "140 条 HYPOTHESIS 登记"),
}

# 产物 → 生成它的 boundary label 前缀（用于反查真实窗口）
LABEL_PREFIX: dict[str, tuple[str, ...]] = {
    "roles.csv": ("audit-dir", "h3calib"),
    "h2_stock_all.csv": ("h2-batch", "h2-w", "h2-ex", "h2p-"),
    "h2_joint.csv": ("h2-batch", "h2-w", "h2-ex", "h2p-"),
    "h2_stock.csv": ("h2-batch", "h2-w"),
    "f1_joint_judged.csv": ("f1-", "f1s-", "m4diag"),
    "f1_market_judged.csv": ("f1-", "f1-spread"),
    "hypothesis_registry.csv": ("intake",),
}

# 自测/调试读取**不是搜索证据**：它们故意读到文件最大日期（2026-09-18）。
# 若把它们计入，会把搜索窗口错标成全量（这是首次打标出现的错误）。
NON_SEARCH_PREFIXES = ("selftest", "dbg", "debug", "ix-selftest", "pool-selftest",
                       "perturb", "calib2", "audit-dir-neg")

TAG_COLS = ["search_start", "search_end", "label_available_max", "evidence"]


def read_log() -> list[dict]:
    if not LOG.is_file():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def observed_windows() -> dict[str, dict]:
    """从 boundary_log 反查：label 前缀 → 实际用过的窗口集合。"""
    out: dict[str, dict] = {}
    for r in read_log():
        lb = r.get("label") or ""
        op = r.get("op")
        cap = r.get("data_max") or r.get("label_max")
        mx = r.get("signal_max_date")
        if not lb or not cap:
            continue
        rec = out.setdefault(lb, {"caps": set(), "actual_max": set(), "ops": set(),
                                  "label_caps": set()})
        rec["caps"].add(cap)
        if op == "read_labels":
            rec["label_caps"].add(cap)
        if mx:
            rec["actual_max"].add(mx)
        rec["ops"].add(op)
    return out


def _match(label: str, prefixes: tuple[str, ...]) -> bool:
    if label.startswith(NON_SEARCH_PREFIXES):
        return False
    return any(label.startswith(p) for p in prefixes)


def resolve_window(artifact: str, obs: dict[str, dict]) -> dict:
    """确定某产物的真实搜索窗口（由 boundary_log 证据推出，不硬编码）。

    有效窗口 = 因子读取上界 ∩ 标签读取上界，取**两者较小值**。例如
    h2-batch 的因子读到 2026-09-18，但标签只读到 2024-12-26，故有效窗口
    止于 2024-12-26（`max` 是错的，会把窗口错标成全量）。
    """
    prefixes = LABEL_PREFIX.get(artifact, ())
    caps: set[str] = set()
    label_caps: set[str] = set()
    for lb, rec in obs.items():
        if not _match(lb, prefixes):
            continue
        caps |= rec["caps"]
        label_caps |= rec["label_caps"]
    if caps:
        end = min(caps)
        return {"search_start": W.FACTORS_FIRST_SIGNAL_DATE,
                "search_end": end,
                "label_available_max": min(label_caps) if label_caps else end,
                "evidence": "E1",
                "source": "boundary_log"}
    return {"search_start": W.LEGACY_SEARCH_START,
            "search_end": W.LEGACY_SEARCH_END,
            "label_available_max": W.LEGACY_SEARCH_END,
            "evidence": "E1",
            "source": "legacy_fallback"}


def build_registry(obs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for art, (key, desc) in CANDIDATE_ARTIFACTS.items():
        p = RESULTS / art
        if not p.is_file():
            continue
        w = resolve_window(art, obs)
        n = len(pd.read_csv(p, usecols=[key]))
        rows.append({
            "artifact": art, "key_column": key, "n_candidates": n,
            "description": desc,
            "search_start": w["search_start"], "search_end": w["search_end"],
            "label_available_max": w["label_available_max"],
            "evidence": w["evidence"], "window_source": w["source"],
            "covers_prod_window": w["search_end"] >= W.SIGNAL_MAX_DATE,
        })
    rows.append({
        "artifact": "(参考) 生产版窗口", "key_column": "-", "n_candidates": 0,
        "description": "计划 §11.2 规定的新搜索应使用的窗口",
        "search_start": W.FACTORS_FIRST_SIGNAL_DATE,
        "search_end": W.SIGNAL_MAX_DATE,
        "label_available_max": W.LABEL_MAX_DATE,
        "evidence": "E1", "window_source": "MERGE_V2_PLAN §10.4/§11.2",
        "covers_prod_window": True,
    })
    for v in W.VERSION_WINDOWS:
        if v["version"] == "P":
            continue
        rows.append({
            "artifact": f"(参考) 框架版本 {v['version']}", "key_column": "-",
            "n_candidates": 0,
            "description": f"外层窗 {v['outer_start']}~{v['outer_end']}",
            "search_start": W.FACTORS_FIRST_SIGNAL_DATE,
            "search_end": v["cutoff"], "label_available_max": W.LABEL_MAX_DATE,
            "evidence": "E2", "window_source": "MERGE_V2_PLAN §3.4",
            "covers_prod_window": v["cutoff"] >= W.SIGNAL_MAX_DATE,
        })
    return pd.DataFrame(rows)


def tag_artifacts(obs: dict[str, dict], restore: bool = False,
                  verify: bool = False) -> int:
    changed = 0
    BACKUP.mkdir(parents=True, exist_ok=True)
    for art in CANDIDATE_ARTIFACTS:
        p = RESULTS / art
        if not p.is_file():
            continue
        bak = BACKUP / art
        if restore:
            if bak.is_file():
                shutil.copy2(bak, p)
                print(f"  还原 {art}")
                changed += 1
            continue
        src = bak if bak.is_file() else p
        if not bak.is_file():
            shutil.copy2(p, bak)
        df = pd.read_csv(src)
        already = all(c in df.columns for c in TAG_COLS)
        if already and verify:
            continue
        if already and not verify:
            df = df.drop(columns=TAG_COLS)
        w = resolve_window(art, obs)
        for c, v in (("search_start", w["search_start"]),
                     ("search_end", w["search_end"]),
                     ("label_available_max", w["label_available_max"]),
                     ("evidence", w["evidence"])):
            df[c] = v
        if not (verify or already):
            df.to_csv(p, index=False, encoding="utf-8")
        print(f"  打标 {art}: {w['search_start']}~{w['search_end']} "
              f"({len(df)} 行, source={w['source']})")
        changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="给已搜索候选打搜索窗口标签")
    ap.add_argument("--verify", action="store_true", help="只校验不写")
    ap.add_argument("--restore", action="store_true", help="从备份还原")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    obs = observed_windows()
    print(f"boundary_log 反查到 {len(obs)} 个 label 的读取记录\n")

    print("=== 打标 ===")
    n = tag_artifacts(obs, restore=args.restore, verify=args.verify)

    if not args.restore:
        reg = build_registry(obs)
        reg_path = RESULTS / "search_registry.csv"
        if not args.verify:
            reg.to_csv(reg_path, index=False, encoding="utf-8")
        print(f"\n=== 搜索窗口登记表 → {reg_path.name} ===")
        print(reg[["artifact", "n_candidates", "search_start", "search_end",
                   "evidence", "window_source", "covers_prod_window"]]
              .to_string(index=False))
        stale = reg[(~reg["covers_prod_window"])
                    & (~reg["artifact"].str.startswith("(参考)"))]
        if len(stale):
            print(f"\n注意：{len(stale)} 个候选产物未覆盖生产窗口 "
                  f"(≤ {W.SIGNAL_MAX_DATE})，其结论仅对旧窗口成立。")
    print(f"\n{'校验' if args.verify else '还原' if args.restore else '打标'}完成，"
          f"处理 {n} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
