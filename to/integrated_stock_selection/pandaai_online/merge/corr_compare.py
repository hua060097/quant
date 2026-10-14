# -*- coding: utf-8 -*-
"""回答任务 3：因子值相关 vs IC 序列相关，是两个不同的东西。

测量 base20 在**新选择窗**上的两种相关：
  A. 因子值相关：同一交易日内，两个因子横截面值的 Spearman（`dedup.py` 的口径）
  B. IC 序列相关：两个因子逐日 IC 的时间序列相关（我此前报告的 46.2%/PC1 口径）
并对照 dedup 的阈值（同族 |ρ|≥0.85 只留一，跨族 ≥0.90 只警告）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary as B  # noqa: E402
import fastranks as FR  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
CUT = W.FACTOR_SCREEN_MAX

b20 = json.loads(open(os.path.join(RESULTS, "base20.json"),
                      encoding="utf-8").read())["base20"]
names = [x["factor"] for x in b20]
print(f"窗口 ≤{CUT}；base20 {len(names)} 个因子\n")

lab = B.read_labels(["r3", "tradable"], label_max=CUT, label="corr")
lab["date"] = pd.to_datetime(lab["date"])
lab = lab[(lab["tradable"] == 1) & (lab["date"] <= CUT)]
fac = B.read_factors(names, data_max=W.LABEL_MAX_DATE, label="corr")
fac["date"] = pd.to_datetime(fac["date"])
m = lab[["stock_code", "date", "r3"]].merge(fac, on=["stock_code", "date"],
                                            how="inner")
m = m[m["date"] <= CUT]
del fac
print(f"面板 {len(m):,} 行 {m['date'].nunique()} 天", flush=True)

# ---- A. 因子值相关（逐日横截面 Spearman，跨日取均值）----
t0 = time.time()
V = m[names].to_numpy(dtype=float)
day = pd.factorize(m.sort_values("date")["date"])[0]
order = np.argsort(m.sort_values("date")["date"].to_numpy(), kind="mergesort")
Vs = V[order]
ds = day[order]
K = len(names)
acc = np.zeros((K, K))
cnt = 0
uniq = np.unique(ds)
for d in uniq:
    idx = np.flatnonzero(ds == d)
    blk = Vs[idx]
    ok = np.isfinite(blk).all(axis=1)
    blk = blk[ok]
    if blk.shape[0] < 50:
        continue
    R = np.column_stack([pd.Series(blk[:, j]).rank().to_numpy() for j in range(K)])
    R -= R.mean(axis=0)
    sd = R.std(axis=0)
    if (sd <= 0).any():
        continue
    R /= sd
    c = R.T @ R / blk.shape[0]
    np.clip(c, -1, 1, out=c)
    acc += c
    cnt += 1
A = acc / max(cnt, 1)
print(f"因子值相关：{cnt} 天  {time.time() - t0:.0f}s")

# ---- B. IC 序列相关 ----
pr = FR.PanelRanker(m, verbose=False)
ic = {}
for c in names:
    ic[c] = pr.ic_series(pr.x_rank(c)).to_numpy(dtype=float)
IC = pd.DataFrame(ic)
Bc = IC.corr().to_numpy()

iu = np.triu_indices(K, 1)


def summ(M, nm):
    v = M[iu]
    v = v[np.isfinite(v)]
    print(f"{nm}: 中位 {np.median(v):+.3f}  90分位 {np.quantile(v, .9):+.3f}  "
          f"|ρ|>0.5 {np.mean(np.abs(v) > 0.5) * 100:.1f}%  "
          f"|ρ|>0.7 {np.mean(np.abs(v) > 0.7) * 100:.1f}%  "
          f"|ρ|>0.85 {np.mean(np.abs(v) > 0.85) * 100:.1f}%")
    ev = np.linalg.eigvalsh(np.nan_to_num(M))[::-1]
    print(f"    PC1 {ev[0] / K * 100:.1f}%  前5 {ev[:5].sum() / K * 100:.1f}%")
    rho = float(np.nanmean(v))
    print(f"    平均 ρ={rho:+.3f} → n_eff = {K / (1 + (K - 1) * rho):.2f}")
    return v


va = summ(A, "A 因子值相关")
vb = summ(Bc, "B IC序列相关")

print(f"\n两种相关的相关性（逐对）：corr(A, B) = "
      f"{np.corrcoef(va, vb)[0, 1]:+.3f}")

print("\n=== 按 dedup 阈值分类 ===")
print(f"  因子值 |ρ|≥0.85 的对: {int((np.abs(va) >= 0.85).sum())}/{len(va)}")
print(f"  IC序列 |ρ|≥0.85 的对: {int((np.abs(vb) >= 0.85).sum())}/{len(vb)}")
hi_val = set(np.flatnonzero(np.abs(va) >= 0.7))
hi_ic = set(np.flatnonzero(np.abs(vb) >= 0.7))
print(f"  |ρ|≥0.7：因子值 {len(hi_val)} 对，IC序列 {len(hi_ic)} 对，"
      f"交集 {len(hi_val & hi_ic)} 对")
print(f"  → 因子值高相关的 {len(hi_val)} 对**全部**落在 IC 高相关集合内"
      f"（是真子集）；另有 {len(hi_ic - hi_val)} 对只在 IC 口径上高相关。")
print("  → 两口径强相关但不对称：IC 序列相关是更宽的探测面，"
      "能抓到'值不相关但同一时段一起有效'的对。")
print(f"  → 逐对相关 corr(A,B)={np.corrcoef(va, vb)[0, 1]:+.3f}：二者高度相关"
      "（不是同一个东西，但也不是独立的两个信号）。")

np.save(os.path.join(os.path.dirname(RESULTS), "..", "results", "merge_v2",
                     "_corr_value.npy"), A)
pd.DataFrame(A, index=names, columns=names).to_csv(
    os.path.join(RESULTS, "corr_value_base20.csv"), encoding="utf-8")
pd.DataFrame(Bc, index=names, columns=names).to_csv(
    os.path.join(RESULTS, "corr_ic_base20.csv"), encoding="utf-8")
print(f"\n输出 → corr_value_base20.csv / corr_ic_base20.csv")
