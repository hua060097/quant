# -*- coding: utf-8 -*-
"""新因子晋级回写生产库（V4.3c，显式离线操作，严禁自动调用）。

将在 merge 中新入选的 mined/custom/interact 特征列并入
factor_lab_data/factors.parquet，并同步 roles.csv + 作废 store_vfps 缓存。

用法：
    python promote_to_store.py --values new_cols.parquet \\
        --roles new_roles.json [--backup-dir backups/] [--apply]

--values：列 = stock_code/date + 新特征列（与面板同行序无关，按键合并）。
--roles：[{name, group, role, note}]，role 限 STOCK/STOCK_REL/CROSS_SEC。
默认 dry-run（只校验打印）；--apply 才真正重写（先备份）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
FACTORS = os.path.join(ROOT, "factor_lab_data", "factors.parquet")
ROLES = os.path.join(RESULTS, "roles.csv")

ALLOWED_ROLES = ("STOCK", "STOCK_REL", "CROSS_SEC")


def main() -> int:
    ap = argparse.ArgumentParser(description="新因子晋级回写生产库（显式离线）")
    ap.add_argument("--values", required=True, help="新特征列 parquet")
    ap.add_argument("--roles", required=True, help="新列 roles 定义 JSON")
    ap.add_argument("--backup-dir", default=os.path.join(ROOT, "factor_lab_data", "backups"))
    ap.add_argument("--apply", action="store_true", help="真正重写（默认 dry-run）")
    a = ap.parse_args()

    new = pd.read_parquet(a.values)
    new["date"] = pd.to_datetime(new["date"])
    roles_new = json.load(open(a.roles, encoding="utf-8"))
    assert isinstance(roles_new, list) and roles_new, "roles 非空列表"
    for r in roles_new:
        assert r.get("role") in ALLOWED_ROLES, f"role 非法 {r}"
        assert r["name"] in new.columns, f"values 缺列 {r['name']}"

    base = pd.read_parquet(FACTORS)
    base["date"] = pd.to_datetime(base["date"])
    overlap = [c for c in new.columns if c in base.columns and c not in ("stock_code", "date")]
    if overlap:
        raise SystemExit(f"列名冲突（已在库中，用孪生规则而非回写）: {overlap}")
    mg = base.merge(new, on=["stock_code", "date"], how="left")
    assert len(mg) == len(base), "合并行数膨胀（键重复）"
    newcols = [r["name"] for r in roles_new]
    print(f"base {base.shape} + new {len(newcols)} cols; "
          f"覆盖率: {float(np.isfinite(mg[newcols].to_numpy()).mean()):.3f}", flush=True)

    if not a.apply:
        print("dry-run：未写盘（加 --apply 执行）", flush=True)
        return 0
    os.makedirs(a.backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk = os.path.join(a.backup_dir, f"factors.parquet.{ts}.bak")
    shutil.copy2(FACTORS, bk)
    print(f"备份 → {bk}", flush=True)
    rbk = os.path.join(a.backup_dir, f"roles.csv.{ts}.bak")
    shutil.copy2(ROLES, rbk)
    roles = pd.read_csv(ROLES)
    add = pd.DataFrame([{"name": r["name"], "group": r.get("group", "promoted"),
                          "role": r["role"], "nunique_med": np.nan, "coverage": np.nan,
                          "note": r.get("note", "promoted from merge"),
                          "search_start": "", "search_end": "",
                          "label_available_max": "", "evidence": "promote"}
                         for r in roles_new])
    roles = pd.concat([roles, add], ignore_index=True)
    tmp_factors = f"{FACTORS}.tmp.{os.getpid()}"
    tmp_roles = f"{ROLES}.tmp.{os.getpid()}"
    try:
        mg.to_parquet(tmp_factors, index=False)
        roles.to_csv(tmp_roles, index=False)
        os.replace(tmp_factors, FACTORS)
        os.replace(tmp_roles, ROLES)
    finally:
        for tmp in (tmp_factors, tmp_roles):
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
    # 作废 store_vfps 缓存（sha 已变，下次自动重算）
    import glob as _g
    for p in _g.glob(os.path.join(RESULTS, "store_vfps_*.json")):
        os.remove(p)
        print(f"作废缓存 {os.path.basename(p)}", flush=True)
    print(f"回写完成：+{len(newcols)} 列", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
