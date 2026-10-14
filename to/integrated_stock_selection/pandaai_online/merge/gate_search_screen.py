# -*- coding: utf-8 -*-
"""门控搜索：G0 粗筛 + 方向中性度检查（两级评估的第一级，便宜）。

**为什么加方向中性度**：本轮取证发现 M4（宽度）R²=79% 被市场方向解释，正交化后
条件效应消失。因此搜索空间里的市场级候选**必须**先过"方向中性"这一关，否则
会重复 M4 的错误——产出的"有效门控"只是市场涨跌的代理。

**B5（2026-10-12）：与 `run_gate_pipeline.py` 共享同一套 S2 层实现**——
此前本模块只剩 G0+中性度（旧 S2），管线另有 legality/dust/order_fp/structure 四层
⇒ 两条候选生成路径（prepare_f1 读本模块的 screen.csv vs 管线短名单）口径分裂。
改为：本模块 import 管线的 `spec_legality`/`dust_check`/`sparsity_check` 与
`gate_ledger` 的 `gate_order_fp`/`structure_check`，对市场与行业候选**同样施加**，
并把结果并入 `eligible`——一套实现两个入口，`screen.csv` 与管线判定同口径。

**两级评估（借鉴 grammar 的快面板粗筛思路）**：
  第一级（本模块）：G0 资格 + 方向中性度 + 共享 S2 层 → 从 54 个里筛出存活者
  第二级（后续）：仅对存活者跑 F1（贵：全量面板 + 200 次 placebo）

用法：
    python gate_search_screen.py --corr-max 0.6
    python gate_search_screen.py --corr-max 0.6 --out results/merge_v2/gates_search/screen.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_validate as GV  # noqa: E402
import gate_ledger as GL  # noqa: E402
import gate_search_space as GSS  # noqa: E402
import gates as G  # noqa: E402
import run_gate_pipeline as RP  # noqa: E402  共享 S2 层实现（单一事实源）
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")
CUT = W.FACTOR_SCREEN_MAX
EVAL_START = W.FACTORS_FIRST_SIGNAL_DATE


def _specs_by_id() -> dict[str, dict]:
    return {s["gate_id"]: s for s in GSS.generate_space()}


def _actual_qwin(space_dir: str) -> int:
    """B4：spec_legality 的分位窗读**实际生成值**（与管线一致）。"""
    reg_path = os.path.join(space_dir, "gate_search_registry.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path, encoding="utf-8") as f:
                return int(json.load(f).get("qwin") or 252)
        except (ValueError, OSError):
            pass
    print("  ⚠ 未找到 gate_search_registry.json（spec_legality 退回默认 qwin=252）",
          flush=True)
    return 252


def _series_layers(d: pd.DataFrame, spec: dict | None, qwin: int) -> dict:
    """共享 S2 层：③合法性 / ④尘埃 / ⑨稀疏 / ⑧结构自检（与 run_gate_pipeline 同一实现）。"""
    ok_leg, why_leg = (True, "")
    if spec is not None:
        ok_leg, why_leg = RP.spec_legality(spec, qwin)
    ok_dust, rho_t = RP.dust_check(d["p"].to_numpy())
    sp = RP.sparsity_check(d["p"].to_numpy())
    ok_st, st_rep = GL.structure_check(d["p"].to_numpy(), name=str(spec.get("gate_id") if spec else "n/a"))
    return {"legal": ok_leg, "legal_reason": why_leg,
            "dust": ok_dust, "rho_time": rho_t,
            "sp_on_frac": sp["on_frac"], "sp_warn": sp["warn"],
            "structure_ok": ok_st,
            "structure_why": "; ".join(st_rep["why"])}


def _order_fp(p: np.ndarray) -> str:
    """⑥ 排序等价指纹（与管线同一判据；对 rank 归一化序列不再退化为边际分布）。"""
    return GL.gate_order_fp(p)


def market_direction() -> pd.DataFrame:
    """市场方向代理：20/60 日累计等权收益 + 其 252 日分位。"""
    mkt = G.load_market_daily()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = mkt.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": mkt["date"]})
    for w in (20, 60):
        out[f"cum{w}"] = mkt["r_m"].rolling(w, min_periods=w).sum()
        out[f"cum{w}_p"] = G._roll_quantile(out[f"cum{w}"], G.QWIN,
                                            G.QMIN["market"]).to_numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="门控搜索 G0 + 方向中性度粗筛")
    ap.add_argument("--space-dir", default=SPACE)
    ap.add_argument("--corr-max", type=float, default=0.6,
                    help="|corr(p, 市场方向分位)| 上限；超限即判为方向代理")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    md = market_direction()
    md = md[(md["date"] >= EVAL_START) & (md["date"] <= CUT)].set_index("date")
    specs = _specs_by_id()
    qwin = _actual_qwin(a.space_dir)
    seen_ofp: dict[str, str] = {}

    rows = []
    # ---- 市场级 ----
    gm = pd.read_parquet(os.path.join(a.space_dir, "gate_search_market.parquet"))
    gm["date"] = pd.to_datetime(gm["date"])
    gm = gm[(gm["date"] >= EVAL_START) & (gm["date"] <= CUT)]
    print(f"市场候选 {gm['gate_id'].nunique()} 个，{len(gm):,} 行")
    for gid, sub in gm.groupby("gate_id", sort=True):
        sub = sub.sort_values("date")
        d = pd.DataFrame({"date": sub["date"].to_numpy(),
                          "p": sub["p"].to_numpy(dtype=float)})
        d = d.merge(md.reset_index(), on="date", how="left")
        g0b = GV.g0_binary(d["date"], d["p"])
        g0c = GV.g0_continuous(d["date"], 2 * d["p"] - 1, eval_start=EVAL_START)
        corr20 = float(d["p"].corr(d["cum20_p"], method="spearman"))
        corr60 = float(d["p"].corr(d["cum60_p"], method="spearman"))
        cmax = np.nanmax(np.abs([corr20, corr60]))
        lay = _series_layers(d, specs.get(gid), qwin)
        ofp = _order_fp(d["p"].to_numpy())
        order_dup = bool(ofp and ofp in seen_ofp)      # ⑥ 与管线同一判据
        if ofp and not order_dup:
            seen_ofp[ofp] = gid
        base_ok = (g0c["g0_status"] == "pass") and (cmax < a.corr_max)
        rows.append({
            "gate_id": gid, "level": "market", "g0_binary": g0b["g0_status"],
            "g0_cont": g0c["g0_status"], "cov": g0c.get("cov"),
            "std": g0c.get("std"), "med": g0c.get("median"),
            "seg_med": g0b.get("seg_med"), "flips_py": g0b.get("flips_py"),
            "corr_dir20": corr20, "corr_dir60": corr60, "corr_dir_max": cmax,
            "direction_neutral": bool(cmax < a.corr_max),
            "legal": lay["legal"], "legal_reason": lay["legal_reason"],
            "dust": lay["dust"], "rho_time": lay["rho_time"],
            "sp_on_frac": lay["sp_on_frac"], "sp_warn": lay["sp_warn"],
            "structure_ok": lay["structure_ok"], "structure_why": lay["structure_why"],
            "order_fp": ofp, "order_dup": order_dup,
            "eligible": bool(base_ok and lay["legal"] and lay["dust"]
                             and lay["structure_ok"] and not order_dup),
        })
    # ---- 行业级 ----
    gs = pd.read_parquet(os.path.join(a.space_dir, "gate_search_sector.parquet"))
    gs["date"] = pd.to_datetime(gs["date"])
    gs = gs[(gs["date"] >= EVAL_START) & (gs["date"] <= CUT)]
    print(f"行业候选 {gs['gate_id'].nunique()} 个，{len(gs):,} 行")
    for gid, sub in gs.groupby("gate_id", sort=True):
        per_ok, per_ins, per_fail = 0, 0, 0
        stds, covs = [], []
        for sec, t in sub.groupby("sector", sort=False):
            r = GV.g0_continuous(t["date"], 2 * t["p"] - 1, eval_start=EVAL_START)
            if r["g0_status"] == "pass":
                per_ok += 1
            elif r["g0_status"] == "insufficient":
                per_ins += 1
            else:
                per_fail += 1
            if r.get("std") is not None:
                stds.append(r["std"])
            if r.get("cov") is not None:
                covs.append(r["cov"])
        n = per_ok + per_ins + per_fail
        agg = ("pass" if per_ok / max(n, 1) >= 0.5
               else ("insufficient" if per_ins > per_fail else "fail"))
        # 共享层：dust/structure 用逐行业估计的近似（秩相关对压平方向敏感——
        # 行业门控 p 为跨行业截面 rank，此处用展平序列并依赖 structure_check 拒绝退化）
        d_sec = pd.DataFrame({"date": sub["date"].to_numpy(),
                              "p": sub["p"].to_numpy(dtype=float)})
        lay = _series_layers(d_sec, specs.get(gid), qwin)
        ofp = _order_fp(d_sec["p"].to_numpy())
        order_dup = bool(ofp and ofp in seen_ofp)
        if ofp and not order_dup:
            seen_ofp[ofp] = gid
        rows.append({
            "gate_id": gid, "level": "sector", "g0_binary": "",
            "g0_cont": agg, "cov": float(np.nanmedian(covs)) if covs else np.nan,
            "std": float(np.nanmedian(stds)) if stds else np.nan,
            "med": np.nan, "seg_med": np.nan, "flips_py": np.nan,
            "corr_dir20": np.nan, "corr_dir60": np.nan, "corr_dir_max": np.nan,
            "direction_neutral": True,           # 行业级本就截面化，不适用该检验
            "legal": lay["legal"], "legal_reason": lay["legal_reason"],
            "dust": lay["dust"], "rho_time": lay["rho_time"],
            "sp_on_frac": lay["sp_on_frac"], "sp_warn": lay["sp_warn"],
            "structure_ok": lay["structure_ok"], "structure_why": lay["structure_why"],
            "order_fp": ofp, "order_dup": order_dup,
            "eligible": bool(agg == "pass" and lay["legal"] and lay["dust"]
                             and lay["structure_ok"] and not order_dup),
            "sector_pass": per_ok, "sector_n": n,
        })

    df = pd.DataFrame(rows)
    out = a.out or os.path.join(a.space_dir, "screen.csv")
    df = df.sort_values(["eligible", "corr_dir_max"],
                        ascending=[False, True]).reset_index(drop=True)
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n=== G0 + 方向中性度 + 共享 S2 层（corr 上限 {a.corr_max}）===")
    print(f"  G0 连续形式通过: {int((df['g0_cont'] == 'pass').sum())}/{len(df)}")
    m = df[df["level"] == "market"]
    print(f"  市场级方向中性: {int(m['direction_neutral'].sum())}/{len(m)}")
    print(f"  共享层拒绝：合法性 {int((~df['legal']).sum())} / 尘埃 "
          f"{int((~df['dust']).sum())} / "
          f"结构 {int((~df['structure_ok']).sum())} / 排序等价重复 {int(df['order_dup'].sum())}"
          "（与 run_gate_pipeline 同一实现，B5）")
    print(f"  **最终 eligible（G0 过 且 方向中性 且 共享层全过）: {int(df['eligible'].sum())}/{len(df)}**")
    print(f"\n通过清单：")
    w = df[df["eligible"]]
    if not len(w):
        print("  （无）")
    for _, r in w.iterrows():
        extra = (f"  corr_dir={r['corr_dir_max']:.3f}" if r["level"] == "market"
                 else f"  行业通过 {int(r['sector_pass'])}/{int(r['sector_n'])}")
        print(f"  {r['gate_id']:<30s} std={r['std']:.3f} cov={r['cov']:.2f}{extra}")
    print(f"\n被方向代理判掉的（corr 超限）：")
    bad = df[(df["level"] == "market") & (~df["direction_neutral"])
             & (df["g0_cont"] == "pass")]
    for _, r in bad.head(10).iterrows():
        print(f"  {r['gate_id']:<30s} corr_dir={r['corr_dir_max']:.3f}  ← 疑似 M4 式代理")
    print(f"\n输出 → {out}  用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
