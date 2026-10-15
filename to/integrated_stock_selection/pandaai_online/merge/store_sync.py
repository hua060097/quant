# -*- coding: utf-8 -*-
"""生产库 store 指纹同步（V4.3c）：factors.parquet 存量因子防重复挖矿。

`run_search` 快评用 fast 面板值指纹（`seen_vfp`）跳过已见等价式。
本模块在**完全相同的 fast 面板构造参数**下预计算存量列的 `value_fingerprint`，
调用方并入 `seen_vfp` 后，GP 撞上已入库因子（含历史 20 个 top）的仿射等价体
即 `vfp_dup` 跳过。结果缓存 `store_vfps_<data_cut>.json`（含 factors sha，
文件换代自动失效重算）。
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")

STORE_ROLES = ("STOCK", "STOCK_REL", "CROSS_SEC")


def store_columns() -> list[str]:
    """可入源的存量个股列（roles 口径 ∩ 文件实际列；排除 MARKET/SECTOR 广播类）。"""
    roles = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
    want = [str(x) for x in roles[roles["role"].isin(STORE_ROLES)]["name"].tolist()]
    import pyarrow.parquet as _pq

    fp = os.path.join(ROOT, "factor_lab_data", "factors.parquet")
    have = set(map(str, _pq.read_schema(fp).names))
    return [c for c in want if c in have]


def factors_sha12() -> str:
    h = hashlib.sha1()
    with open(os.path.join(ROOT, "factor_lab_data", "factors.parquet"), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def sync_store_vfps(frame, n_stocks: int, months: int, stratify,
                    end: str, data_cut: str) -> set[str]:
    """与 run_search 快评**同构**的 fast 面板上算存量列 vfp；返回指纹集。

    同构 = 同一 `build_fast_frame(frame, n_stocks, months, stratify, end)` 调用
    （seed 默认 11，两边都不传）。缓存命中（sha+参数一致）直接读盘。
    """
    from evaluator import value_fingerprint
    from fitness import build_fast_frame

    outp = os.path.join(RESULTS, f"store_vfps_{data_cut}.json")
    sha = factors_sha12()
    params = {"sha": sha, "n_stocks": int(n_stocks), "months": int(months),
              "stratify": stratify, "end": str(end)}
    if os.path.exists(outp):
        try:
            old = json.load(open(outp, encoding="utf-8"))
            if all(old.get("params", {}).get(k) == v for k, v in params.items()):
                return set(old.get("vfps", {}).values())
        except Exception:
            pass
    cols = store_columns()
    if not cols:
        return set()
    import boundary as B

    fac = B.read_factors(cols, data_max=end, label="store-vfps")
    fac["date"] = pd.to_datetime(fac["date"])
    fast = build_fast_frame(frame, n_stocks=int(n_stocks), months=int(months),
                            stratify=stratify, end=end)
    key = fac[["stock_code", "date"]].copy()
    key["_k"] = key["stock_code"].astype(str) + "|" + key["date"].astype(str)
    fmap = {k: i for i, k in enumerate(key["_k"].to_numpy())}
    fkk = (fast["stock_code"].astype(str) + "|"
           + pd.to_datetime(fast["date"]).astype(str)).to_numpy()
    pos = np.array([fmap.get(k, -1) for k in fkk], dtype=np.int64)
    ok = pos >= 0
    vfps = {}
    for c in cols:
        try:
            v = fac[c].to_numpy(dtype=float)[pos[ok]]
            fp = value_fingerprint(v)
            if fp:
                vfps[c] = fp
        except Exception:
            continue
    json.dump({"params": params, "n_cols": len(vfps), "vfps": vfps},
              open(outp, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"store_vfps: {len(vfps)}/{len(cols)} cols -> {outp}", flush=True)
    return set(vfps.values())
