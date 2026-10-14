# -*- coding: utf-8 -*-
"""全流程冒烟测试（小样，**仅验流程 / 速度 / 逻辑 / 缺陷，不作任何统计结论**）。

上服务器大规模搜索前的验证闸门。逐阶段调用真实 CLI，记录耗时与不变量：

  S0 生成 → S1 指纹+台账对账 → S2 预筛链 → S4 排序+停止 → S5 G2/G3
  → S7 短名单+台账回写 → 增量对账（二次运行应 0 pending）

⚠ **小样设置不产生可用统计结论**（样本量、置换次数、分辨率均被压缩）。
   目的是把流程跑通、暴露 bug、优化速度与逻辑。

用法：
    python smoke_pipeline.py              # 市场 + 行业 + 增量对账
    python smoke_pipeline.py --keep       # 保留台账（默认每次清空重建，保证可复现）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PY = sys.executable
LOGS = HERE / "out" / "smoke"
LOGS.mkdir(parents=True, exist_ok=True)

#: 冒烟规模（**故意压小**）
SMOKE = {
    "market_gates": 4,      # 市场门控数
    "market_n_boot": 40,
    "market_n_perm": 20,
    "sector_factors": 5,    # 行业门控的因子数
    "sector_min_factors": 3,
    "sector_gates": ["GS_sec_xs_disp_level", "GS_sec_amp_med_level"],
    "sector_n_perm": 6,
    "sector_g3_factors": 4,
    "sector_n_boot": 20,
}

results: list[tuple[str, bool, float, str]] = []


def run(name: str, args: list[str], timeout: int = 3600) -> tuple[bool, float, str]:
    t0 = time.time()
    r = subprocess.run([PY, "-X", "utf8", *args], cwd=str(HERE),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    dt = time.time() - t0
    ok = r.returncode == 0
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
    if not ok:
        tail = (r.stderr or tail)[-400:]
    (LOGS / f"{name}.log").write_text(
        f"$ {' '.join(args)}\nrc={r.returncode}\n--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr}\n", encoding="utf-8")
    results.append((name, ok, dt, tail))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name:<26s} {dt:6.1f}s  {tail.splitlines()[-1][:80] if tail else ''}",
          flush=True)
    return ok, dt, tail


def files(*paths) -> bool:
    return all(p.is_file() for p in paths)


def main() -> int:
    ap = argparse.ArgumentParser(description="全流程冒烟测试（小样）")
    ap.add_argument("--keep", action="store_true", help="保留冒烟台账（默认清空重建）")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    R = HERE.parent.parent / "results" / "merge_v2"
    SP = R / "gates_search"
    # 2026-10-12 修复：冒烟**不得**写生产产物。
    # 原实现用 `HERE/gate_ledger.jsonl`（= `gate_ledger.DEFAULT_PATH`，与生产同一文件）
    # 且把 `pipeline_*.csv` / `g2g3_sector.csv` 直接写进生产 `SP`（并用 --limit-gates
    # 截断）→ 每次冒烟都覆盖生产台账与生产短名单（实测：pipeline_all.csv 40→4 行、
    # pipeline_shortlist.csv 3→0 行）。现改为**独立冒烟目录**，生产产物零接触。
    SD = LOGS / "space"
    SD.mkdir(parents=True, exist_ok=True)
    LEDGER = SD / "gate_ledger.jsonl"
    STOP = SD / "gate_stop_state.json"
    # 流水线需从 space-dir 读预生成的门控面板 → 复制一份到冒烟目录
    import shutil
    for f in ("gate_search_market.parquet", "gate_search_sector.parquet",
              "gate_search_space.csv"):
        src = SP / f
        if src.is_file():
            shutil.copy2(src, SD / f)

    print("=" * 78)
    print("全流程冒烟测试（小样：仅验流程/速度/逻辑/缺陷，不作统计结论）")
    print("=" * 78)

    if not a.keep:
        for p in (LEDGER, SD / "gate_ledger_summary.json", STOP):
            if p.is_file():
                p.unlink()
        print(f"已清空冒烟台账与停止状态（独立目录 {SD}，生产产物零接触）\n")

    t_all = time.time()

    print("--- 阶段 0：单元/回归自测 ---")
    for m in ("selftest_gates.py", "selftest_stop.py"):
        run(m, [m])
    run("selftest_stop_状态", ["gate_stop.py", "--status", "--path", str(STOP)])

    print("\n--- 阶段 1：搜索空间生成（S0） ---")
    # --out-dir SD：不得覆盖生产面板（原实现写生产 SP）
    run("gen_space", ["gate_search_space.py", "--out-dir", str(SD)])
    run("gen_space_list", ["gate_search_space.py", "--list"])
    ok = files(SD / "gate_search_market.parquet", SD / "gate_search_sector.parquet")
    results.append(("gen_space_产物", ok, 0.0,
                    "market/sector parquet 均存在" if ok else "缺产物"))
    print(f"  [{'OK  ' if ok else 'FAIL'}] gen_space_产物")

    print("\n--- 阶段 2：预筛（S2：G0 + 方向中性度） ---")
    run("screen", ["gate_search_screen.py", "--space-dir", str(SD),
                   "--out", str(SD / "screen.csv")])

    print("\n--- 阶段 3：市场流水线（S0→S7） ---")
    # 先删产物：否则"产物存在"检查会被上一次的旧文件骗过（自查发现的假通过）
    for f in ("pipeline_shortlist.csv", "pipeline_all.csv"):
        p = SD / f
        if p.is_file():
            p.unlink()
    run("pipeline_market", ["run_gate_pipeline.py", "--all", "--no-f1",
                            "--space-dir", str(SD),
                            "--ledger", str(LEDGER), "--stop-state", str(STOP),
                            "--limit-gates", str(SMOKE["market_gates"]),
                            "--n-boot", str(SMOKE["market_n_boot"]),
                            "--n-perm", str(SMOKE["market_n_perm"])])
    ok = files(SD / "pipeline_shortlist.csv", SD / "pipeline_all.csv")
    results.append(("pipeline_market_产物", ok, 0.0,
                    "shortlist/all 均存在" if ok else "缺产物"))
    print(f"  [{'OK  ' if ok else 'FAIL'}] pipeline_market_产物")

    print("\n--- 阶段 4：行业流水线（专用后端） ---")
    p = SD / "g2g3_sector.csv"
    if p.is_file():
        p.unlink()
    run("pipeline_sector", ["gate_g2g3_sector.py",
                            "--out", str(SD / "g2g3_sector.csv"),
                            "--ledger", str(LEDGER),
                            "--limit-factors", str(SMOKE["sector_factors"]),
                            "--gates", *SMOKE["sector_gates"],
                            "--n-perm", str(SMOKE["sector_n_perm"]),
                            "--g3-factors", str(SMOKE["sector_g3_factors"]),
                            "--n-boot", str(SMOKE["sector_n_boot"]),
                            "--min-factors", str(SMOKE["sector_min_factors"])])
    ok = files(SD / "g2g3_sector.csv")
    n_row = 0
    if ok:
        import pandas as _pd
        n_row = len(_pd.read_csv(SD / "g2g3_sector.csv"))
    results.append(("pipeline_sector_产物", ok and n_row > 0, 0.0,
                    f"g2g3_sector.csv {n_row} 行" if ok else "缺"))
    print(f"  [{'OK  ' if ok else 'FAIL'}] pipeline_sector_产物  "
          f"g2g3_sector.csv {n_row} 行")

    print("\n--- 阶段 5：增量对账（同口径二次运行应 0 pending） ---")
    r = subprocess.run([PY, "-X", "utf8", "run_gate_pipeline.py", "--no-f1",
                        "--space-dir", str(SD),
                        "--ledger", str(LEDGER), "--stop-state", str(STOP),
                        "--limit-gates", str(SMOKE["market_gates"]),
                        "--n-boot", str(SMOKE["market_n_boot"]),
                        "--n-perm", str(SMOKE["market_n_perm"])],
                       cwd=str(HERE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    ok = "待执行 0/" in (r.stdout or "")
    line = next((l.strip() for l in (r.stdout or "").splitlines()
                 if "待执行" in l), "")
    results.append(("增量对账（二次 0 pending）", ok, 0.0, line))
    print(f"  [{'OK  ' if ok else 'FAIL'}] 增量对账（二次 0 pending）  {line}")

    print("\n--- 阶段 6：台账与停止状态 ---")
    r = subprocess.run([PY, "-X", "utf8", "gate_ledger.py", "--stats",
                        "--path", str(LEDGER)],
                       cwd=str(HERE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    try:
        st = json.loads(r.stdout)
        ok = st["n_candidates"] > 0 and st["n_events"] > 0
        detail = (f"候选 {st['n_candidates']} 事件 {st['n_events']} "
                  f"家族 {st['n_families']} F1对 {st['tested_f1_pairs']}")
    except Exception:
        ok, detail = False, (r.stdout or r.stderr)[:120]
    results.append(("台账统计", ok, 0.0, detail))
    print(f"  [{'OK  ' if ok else 'FAIL'}] 台账统计  {detail}")
    run("stop_status", ["gate_stop.py", "--status", "--path", str(STOP)])

    print("\n" + "=" * 78)
    n_ok = sum(1 for _, ok, _, _ in results if ok)
    print(f"冒烟结果：{n_ok}/{len(results)} 通过；总耗时 {time.time() - t_all:.0f}s")
    print("=" * 78)
    slow = sorted([r for r in results if r[2] > 5], key=lambda x: -x[2])
    if slow:
        print("耗时 >5s 的阶段（提速候选）：")
        for n, _, dt, _ in slow:
            print(f"  {n:<28s} {dt:6.1f}s")
    bad = [n for n, ok, _, _ in results if not ok]
    if bad:
        print(f"失败：{bad}")
    print(f"日志 → {LOGS}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
