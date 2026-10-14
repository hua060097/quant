# -*- coding: utf-8 -*-
"""门控搜索结果判定：按门控聚合 h3_pass / FDR，找出能产出因子的门控。

用法：python judge_gate_search.py <batch.csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_f1 import bh_q  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = Path(sys.argv[1])
    d = pd.read_csv(p)
    d["fdr_q"] = bh_q(d["placebo_p"].to_numpy(float))
    d["fdr_sig"] = d["fdr_q"] <= 0.10
    # 计划 §8.4：(a) CI下界>0 且 (b) nblocks>=4 且 (c) placebo p<0.05
    d["h3_pass"] = ((d["delta_ci_lo"] > 0) & (d["nblocks_pos"] >= 4)
                    & (d["placebo_p"] < 0.05))
    form = d["form"].iloc[0]
    print(f"=== {p.name}  form={form}  总候选 {len(d)} ===")
    print(f"  全族 FDR（m={len(d)}）通过: {int(d['fdr_sig'].sum())}/{len(d)}")
    print(f"  h3_pass: {int(d['h3_pass'].sum())}/{len(d)}")

    agg = d.groupby("gate").agg(
        n=("name", "size"),
        delta_med=("delta_mean", "median"),
        n_pos=("delta_mean", lambda s: int((s > 0).sum())),
        p_sig=("placebo_p", lambda s: int((s < 0.05).sum())),
        h3=("h3_pass", "sum"),
        fdr=("fdr_sig", "sum"),
        p_min=("placebo_p", "min"),
    ).reset_index()
    agg = agg.sort_values(["h3", "delta_med"], ascending=False).reset_index(drop=True)
    print(f"\n{'gate':<30s}{'delta中位':>10s}{'正号':>8s}{'p<.05':>7s}"
          f"{'h3_pass':>9s}{'FDR':>6s}{'p_min':>8s}")
    for _, r in agg.iterrows():
        print(f"{r['gate']:<30s}{r['delta_med']:>10.5f}"
              f"{int(r['n_pos']):>5d}/{int(r['n']):<2d}{int(r['p_sig']):>7d}"
              f"{int(r['h3']):>9d}{int(r['fdr']):>6d}{r['p_min']:>8.4f}")

    good = agg[agg["h3"] > 0]
    print(f"\n=== 能产出可入池因子的门控：{len(good)}/{len(agg)} ===")
    if not len(good):
        print("  （无）")
    else:
        for _, r in good.iterrows():
            print(f"  {r['gate']:<30s} h3_pass={int(r['h3'])}/{int(r['n'])} "
                  f"FDR={int(r['fdr'])}  delta中位 {r['delta_med']:+.5f}")
    out = p.with_name(p.stem + "_judged.csv")
    d.to_csv(out, index=False, encoding="utf-8")
    agg.to_csv(p.with_name(p.stem + "_bygate.csv"), index=False, encoding="utf-8")
    print(f"\n输出 → {out.name} / {p.stem}_bygate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
