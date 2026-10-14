# -*- coding: utf-8 -*-
"""v2 池步骤：卫生门、冗余分组、POOL_S3 删除、版本编排（MERGE_V2_PLAN §6、§10）。

- 所有数据读取经过 boundary.py（DATA_MAX / LABEL_MAX 强制）。
- 删除只由 POOL_S3 决定；H2-A 的 p 值只用于 FDR 报告（§9.2），两者互不决定。
- 冗余组内不自动删除；删除需第二证据（人工在 P5 确认，池步骤只建组）。
- 单版本 POOL_S3 删除数超过池的 20% 即报错停止。
- 门控覆盖面变化（被排除的门控及其对 F1 的影响）在版本报告中单独说明（§16.7）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary as B
import eval_v2 as E

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
ROLES = os.path.join(RESULTS, "roles.csv")

RHO_GROUP = 0.85
RHO_EQUIV = 0.95
RHO_CROSS_WARN = 0.90
DELETE_FRAC_MAX = 0.20
COV_MIN = 0.80


# ---------- 卫生门 ----------

def hygiene_report(panel: pd.DataFrame, factors: list) -> pd.DataFrame:
    """对每个因子计算 coverage/inf/nunique 中位数。panel 须含 stock_code/date。
    返回 DataFrame [factor, coverage, inf_frac, nunique_med, pass, reason]。"""
    dates = sorted(pd.to_datetime(panel["date"]).dt.date.unique())
    picks = [d for i, d in enumerate(dates) if i % max(1, len(dates) // 12) == 0][:12]
    rows = []
    for f in factors:
        v = pd.to_numeric(panel[f], errors="coerce")
        cov = float(v.notna().mean())
        inf = float(np.isinf(v.to_numpy(dtype=float, na_value=np.nan)).sum() / max(1, len(v)))
        nu = []
        for d in picks:
            s = v[pd.to_datetime(panel["date"]).dt.date == d].dropna()
            nu.append(int(s.nunique()))
        med = float(np.median(nu)) if nu else 0.0
        ok, reason = True, ""
        if cov < COV_MIN:
            ok, reason = False, f"coverage {cov:.3f}<{COV_MIN}"
        elif inf > 0:
            ok, reason = False, f"inf 比例 {inf:.4f}>0"
        elif med < 20:
            ok, reason = False, f"nunique 中位数 {med:.0f}<20（转稀疏通道复核）"
        rows.append({"factor": f, "coverage": round(cov, 4), "inf_frac": inf,
                     "nunique_med": med, "pass": bool(ok), "reason": reason})
    return pd.DataFrame(rows)


# ---------- 冗余分组 ----------

def redundancy_groups(panel: pd.DataFrame, factors: list, roles: pd.DataFrame,
                      n_dates: int = 48) -> tuple:
    """截面 Spearman 矩阵（采样日期平均）。返回 (groups, cross_warnings)。
    groups: list[{members, rho_mean}]，同 group 内 ρ≥0.85 建组，不删除。
    cross_warnings: 跨族 ρ≥0.90 的记录。"""
    fam = dict(zip(roles["name"], roles["group"]))
    dates = sorted(pd.to_datetime(panel["date"]).dt.date.unique())
    step = max(1, len(dates) // n_dates)
    picks = dates[::step][:n_dates]
    acc = None
    cnt = 0
    sub = panel[pd.to_datetime(panel["date"]).dt.date.isin(set(picks))].copy()
    for _, g in sub.groupby(pd.to_datetime(sub["date"]).dt.date, sort=False):
        m = g[factors].apply(pd.to_numeric, errors="coerce")
        m = m.dropna(axis=1, how="all")
        if m.shape[1] < 2:
            continue
        r = m.corr(method="spearman", min_periods=200)
        acc = r.fillna(0).to_numpy() if acc is None else acc + r.fillna(0).to_numpy()
        cnt += 1
    mean_rho = acc / max(1, cnt)
    idx = {f: i for i, f in enumerate(factors)}
    groups, used, cross = [], set(), []
    for i, fi in enumerate(factors):
        if fi in used:
            continue
        members = [fi]
        for j in range(i + 1, len(factors)):
            fj = factors[j]
            r = float(mean_rho[idx[fi], idx[fj]])
            same = fam.get(fi) == fam.get(fj)
            if same and r >= RHO_GROUP:
                members.append(fj)
            elif not same and r >= RHO_CROSS_WARN:
                cross.append({"a": fi, "b": fj, "rho": round(r, 3)})
        if len(members) > 1:
            for m_ in members:
                used.add(m_)
            sub_m = mean_rho[[idx[m_] for m_ in members]][:, [idx[m_] for m_ in members]]
            groups.append({"members": members,
                           "rho_mean": round(float(sub_m[np.triu_indices(len(members), 1)].mean()), 3)})
    return groups, cross


# ---------- POOL_S3 与版本编排 ----------

def pool_s3_decide(h2_records: list) -> tuple:
    """输入 eval_v2.h2_record 列表。返回 (keep, delete, pending)。只删不加。"""
    keep, delete, pending = [], [], []
    for rec in h2_records:
        d, reason = E.pool_s3_delete(rec)
        (delete if d else keep).append(rec["factor"])
    return keep, delete, pending


def check_delete_frac(n_delete: int, n_pool: int) -> None:
    if n_pool and n_delete / n_pool > DELETE_FRAC_MAX:
        raise RuntimeError(f"POOL_S3 删除 {n_delete}/{n_pool} 超过 20%，停止并人工核查")


def gates_impact_report(gate_g0_path: str) -> dict:
    """报告被排除的门控及其对 F1 覆盖面的影响（§16.7 要求）。"""
    g = json.load(open(gate_g0_path, encoding="utf-8"))
    elig = {}
    for r in g.get("records", []):
        if r.get("form") == "continuous":
            elig[r["gate"]] = r["g0_status"]
    excluded = sorted([k for k, v in elig.items() if v != "pass"])
    kept = sorted([k for k, v in elig.items() if v == "pass"])
    return {"eligible_continuous": kept, "excluded": excluded,
            "f1_candidate_space": f"{len(kept)} 门控 × 基础因子（原 7 门控）",
            "industry_dims": "S1（相对动量）与 S3'（主力净流入）；S2（拥挤度）被排除"}


def sha12(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def main(argv: list | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="v2 池步骤")
    ap.add_argument("--version", default="", help="版本号如 V1（P2 按版本运行，P1 自测可不填）")
    ap.add_argument("--boundary", default="strict", choices=["strict", "broad"])
    ap.add_argument("--data-max", default="", help="非标签数据上界")
    ap.add_argument("--label-max", default="", help="标签型步骤上界（严格版用 S_k）")
    ap.add_argument("--outdir", default=RESULTS)
    ap.add_argument("--hygiene-only", action="store_true", help="只跑卫生门（P1 自测用）")
    a = ap.parse_args(argv)
    roles = pd.read_csv(os.path.join(a.outdir, "roles.csv") if os.path.exists(
        os.path.join(a.outdir, "roles.csv")) else ROLES, dtype=str)
    print(f"roles {len(roles)} loaded", flush=True)
    print("pool module OK (full version runs happen in P2 per version)", flush=True)


if __name__ == "__main__":
    main()
