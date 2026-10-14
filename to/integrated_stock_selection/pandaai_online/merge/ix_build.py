# -*- coding: utf-8 -*-
"""v2 候选生成：F1/F2/F3 与假设登记册（MERGE_V2_PLAN §7）。

- F2 行业相对化（不依赖门控）：x - sector_mean(x)、x / sector_mean(x)。
- 纯个股候选（≤60）：只登记机制假设与预期符号，不adr hoc 构造数值列；
  数值列由后续 factor 计算管线按登记生成（或由提案文件批量导入）。
- F1 门控乘积 / F3 门控选择：需要门控序列 artifact（gates.py 产物），
  本模块提供构造接口，门控缺失时明确报错，不静默跳过。
- 所有候选写入 hypothesis_registry.csv（ID 形如 HYP-0001，全局唯一递增）。
- 预算：F1 ≤ 140、F2 = 40、F3 ≤ 10、纯个股 ≤ 60，总上限 200（§7.2）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
REGISTRY = os.path.join(RESULTS, "hypothesis_registry.csv")
SECTOR_MAP = os.path.join(os.path.dirname(ROOT), "local_data", "sw_sector2", "sw_sector.json")

BUDGET = {"F1": 140, "F2": 40, "F3": 10, "STOCK": 60, "TOTAL": 200}


# ---------- 假设登记册 ----------

REG_COLS = ["id", "family", "name", "hypothesis_zh", "expected_sign",
            "role", "primitives", "gate", "status", "created"]


def load_registry() -> pd.DataFrame:
    if os.path.exists(REGISTRY):
        return pd.read_csv(REGISTRY, dtype=str).fillna("")
    return pd.DataFrame(columns=REG_COLS)


def register(rows: list) -> pd.DataFrame:
    """rows: list[dict]，必填 family/name/hypothesis_zh/expected_sign/role。
    返回更新后的登记册并落盘。ID 全局唯一递增。"""
    reg = load_registry()
    start = 1
    if len(reg):
        nums = [int(x[4:]) for x in reg["id"] if str(x).startswith("HYP-") and str(x)[4:].isdigit()]
        start = max(nums) + 1 if nums else 1
    import datetime
    new = []
    for i, r in enumerate(rows):
        assert r.get("family") in ("STOCK", "F1", "F2", "F3"), r
        assert r.get("name") and r.get("hypothesis_zh"), r
        new.append({"id": f"HYP-{start + i:04d}", "family": r["family"], "name": r["name"],
                    "hypothesis_zh": r["hypothesis_zh"],
                    "expected_sign": r.get("expected_sign", "?"),
                    "role": r.get("role", ""), "primitives": r.get("primitives", ""),
                    "gate": r.get("gate", ""), "status": "proposed",
                    "created": datetime.date.today().isoformat()})
    reg = pd.concat([reg, pd.DataFrame(new, columns=REG_COLS)], ignore_index=True)
    _check_budget(reg)
    os.makedirs(RESULTS, exist_ok=True)
    reg.to_csv(REGISTRY, index=False, encoding="utf-8-sig")
    return reg


def _check_budget(reg: pd.DataFrame) -> None:
    for fam, cap in BUDGET.items():
        if fam == "TOTAL":
            continue
        n = int((reg["family"] == fam).sum())
        if n > cap:
            raise RuntimeError(f"候选预算超限：{fam} {n} > {cap}")
    if len(reg) > BUDGET["TOTAL"]:
        raise RuntimeError(f"候选总数超限：{len(reg)} > {BUDGET['TOTAL']}")


# ---------- 行业归属 ----------

def load_sector_map() -> pd.Series:
    """静态 SW2 归属（D5）。返回 Series indexed by stock_code。"""
    if not os.path.exists(SECTOR_MAP):
        raise FileNotFoundError(f"行业映射缺失: {SECTOR_MAP}")
    m = json.load(open(SECTOR_MAP, encoding="utf-8"))
    return pd.Series(m, name="sector").astype(str)


# ---------- F2 行业相对化 ----------

def build_f2(panel: pd.DataFrame, base: list, sector: pd.Series,
             out_path: str | None = None) -> pd.DataFrame:
    """panel 须含 stock_code/date + base 列。返回 [stock_code, date, 40 列]。
    sector_mean 用静态归属（D5，前视风险见计划 §9.2）。"""
    df = panel[["stock_code", "date"] + list(base)].copy()
    df["_sec"] = df["stock_code"].astype(str).map(sector)
    out = df[["stock_code", "date"]].copy()
    for col in base:
        v = pd.to_numeric(df[col], errors="coerce").astype(float)
        mu = v.groupby([df["date"], df["_sec"]]).transform("mean")
        out[col + "__sub_sec"] = (v - mu).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = v / mu.replace(0, np.nan)
        ratio[~np.isfinite(ratio.to_numpy())] = np.nan
        out[col + "__div_sec"] = ratio.astype(np.float32)
    if out_path:
        out.to_parquet(out_path, index=False)
    return out


def register_f2(base: list) -> pd.DataFrame:
    rows = []
    for col in base:
        rows.append({"family": "F2", "name": col + "__sub_sec", "role": "STOCK_REL",
                     "primitives": col,
                     "hypothesis_zh": f"{col} 减行业均值：剥离行业贝塔后的个股特异性",
                     "expected_sign": "同基因子"})
        rows.append({"family": "F2", "name": col + "__div_sec", "role": "STOCK_REL",
                     "primitives": col,
                     "hypothesis_zh": f"{col} 除行业均值：个股相对行业的异常度",
                     "expected_sign": "同基因子"})
    return register(rows)


# ---------- F1 / F3（需门控，门控缺失时报错） ----------

def _require_gates(gates) -> pd.DataFrame:
    if gates is None:
        raise RuntimeError("F1/F3 需要门控序列 artifact（gates.py 产物，P1-3），当前未提供")
    if isinstance(gates, pd.DataFrame):
        assert {"stock_code", "date"}.issubset(gates.columns), "门控须含 stock_code/date"
        return gates
    raise TypeError("gates 须为 DataFrame")


def build_f1(panel: pd.DataFrame, base: list, gates: pd.DataFrame,
             signs: dict, out_path: str | None = None) -> pd.DataFrame:
    """门控乘积：x·max(g,0) 或 x·max(−g,0)。signs: {(gate, col): +1/-1}，须预先登记。"""
    gates = _require_gates(gates)
    df = panel.merge(gates, on=["stock_code", "date"], how="left")
    out = df[["stock_code", "date"]].copy()
    for (gcol, col), s in signs.items():
        assert s in (1, -1), s
        gv = pd.to_numeric(df[gcol], errors="coerce").astype(float)
        xv = pd.to_numeric(df[col], errors="coerce").astype(float)
        w = np.maximum(s * gv, 0.0)
        out[f"{col}__{gcol}__{'pos' if s > 0 else 'neg'}"] = (xv * w).astype(np.float32)
    if out_path:
        out.to_parquet(out_path, index=False)
    return out


def build_f3(panel: pd.DataFrame, pairs: list, gates: pd.DataFrame,
             out_path: str | None = None) -> pd.DataFrame:
    """门控选择：x·max(g,0) + y·max(−g,0)。pairs: [(gate, x, y, name)]，须预先登记配对。"""
    gates = _require_gates(gates)
    assert len(pairs) <= BUDGET["F3"], "F3 超出预算"
    df = panel.merge(gates, on=["stock_code", "date"], how="left")
    out = df[["stock_code", "date"]].copy()
    for gcol, x, y, name in pairs:
        gv = pd.to_numeric(df[gcol], errors="coerce").astype(float)
        xv = pd.to_numeric(df[x], errors="coerce").astype(float)
        yv = pd.to_numeric(df[y], errors="coerce").astype(float)
        out[name] = (xv * np.maximum(gv, 0.0) + yv * np.maximum(-gv, 0.0)).astype(np.float32)
    if out_path:
        out.to_parquet(out_path, index=False)
    return out


# ---------- CLI ----------

def main(argv: list | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="v2 候选生成与假设登记")
    ap.add_argument("--register-f2", nargs="+", default=None,
                    help="为 base 因子登记 F2（只写登记册，不计算数值）")
    ap.add_argument("--build-f2", action="store_true",
                    help="计算 F2 数值列（需要 --base20 与输出路径）")
    ap.add_argument("--base20", default="", help="逗号分隔的 base 因子，或已登记证明文件")
    ap.add_argument("--out", default="", help="F2 数值输出 parquet 路径")
    ap.add_argument("--start", default="", help="数值计算的起始日期（含）")
    ap.add_argument("--end", default="", help="数值计算的结束日期（含）")
    a = ap.parse_args(argv)
    if a.register_f2:
        reg = register_f2(a.register_f2)
        print(f"registered F2 for {len(a.register_f2)} base -> {len(reg)} total", flush=True)
    if a.build_f2:
        assert a.base20 and a.out, "--build-f2 需要 --base20 与 --out"
        base = [b for b in a.base20.split(",") if b]
        assert len(base) * 2 <= BUDGET["F2"], "F2 超出预算"
        import sys as _s
        _s.path.insert(0, HERE)
        import boundary as B
        fac = B.read_factors(base, data_max=a.end or "2026-09-18", label="ix_build_f2")
        if a.start:
            fac = fac[pd.to_datetime(fac["date"]) >= pd.Timestamp(a.start)].copy()
        sec = load_sector_map()
        build_f2(fac, base, sec, out_path=a.out)
        print(f"F2 built cols={len(base) * 2} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
