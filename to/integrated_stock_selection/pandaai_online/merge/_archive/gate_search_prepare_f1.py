# -*- coding: utf-8 -*-
"""把搜索空间的市场级候选转成 f1_batch 可读的门控 parquet，并生成候选 JSON。

输出 schema 与 `gates/market_gates.parquet` 对齐：
    date, {gate}_p, {gate}_g, {gate}_t

用法：
    python gate_search_prepare_f1.py --screen .../screen.csv --out-dir .../gates_search
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")


def main() -> int:
    ap = argparse.ArgumentParser(description="搜索门控 → f1_batch 输入")
    ap.add_argument("--screen", default=os.path.join(RESULTS, "gates_search", "screen.csv"))
    ap.add_argument("--out-dir", default=os.path.join(RESULTS, "gates_search"))
    ap.add_argument("--both-directions", action="store_true", default=False,
                    help="额外生成 `__neg`（s=-1）方向。**默认关闭** —— §8.4 要求 s 由假设"
                         "登记册预注册（本轮全为 +1）；双向属**事后扩检验族**，"
                         "仅供探索，**不得进入原预注册族的正式判定**")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sc = pd.read_csv(a.screen)
    el = sc[(sc["level"] == "market") & (sc["eligible"])]["gate_id"].tolist()
    print(f"eligible 市场门控 {len(el)} 个")

    gm = pd.read_parquet(os.path.join(a.out_dir, "gate_search_market.parquet"))
    gm["date"] = pd.to_datetime(gm["date"])
    gm = gm[gm["gate_id"].isin(el)]
    piv = gm.pivot_table(index="date", columns="gate_id", values="p", aggfunc="mean")
    piv = piv.sort_index()
    out = pd.DataFrame({"date": piv.index})
    for gid in el:
        p = piv[gid].to_numpy(dtype=float)
        out[f"{gid}_p"] = p
        out[f"{gid}_g"] = 2 * p - 1
        out[f"{gid}_t"] = G.thresh_form(p)
    dst = os.path.join(a.out_dir, "market_gates_search.parquet")
    out.to_parquet(dst, index=False)
    print(f"→ {dst}  {out.shape}  列样例 "
          f"{[c for c in out.columns[:7]]}")

    b20 = json.loads(open(os.path.join(RESULTS, "base20.json"),
                          encoding="utf-8").read())["base20"]
    cands = []
    # **默认只出 `s=+1`（`__pos`）——这是 §8.4 的预注册口径**：
    #   「s = 预注册的门控侧（+1/-1），每个（门控，基因子）对在假设登记册中预先写定，
    #     不得事后挑选」；且「本轮 100 个 F1 的门控侧符号统一预注册为 +1…
    #     反向效应将表现为负 ΔIC，不隐藏」（登记册 HYP-0041~0140 实测 expected_sign 全 +）。
    #
    # **2026-10-12 规则回退记录**：本函数曾在同轮内改为默认出双向（`__neg`, s=-1），
    # 理由是「负向门控从未在正确方向被测过」。但那是**事后挑选方向 + 事后扩检验族**，
    # 违反 §8.4 的两条冻结条款（「不得事后挑选」「运行后不得修改检验族」）。
    # 已回退：双向改为**显式 opt-in**（`--both-directions`），且**不得进入原预注册族的正式判定**
    # （它属于一个新的、需独立预注册的检验族）。
    for g in el:
        for x in b20:
            cands.append({"name": f"{x['factor']}__{g}__pos", "base": x["factor"],
                          "gate": g, "gate_type": "market", "s": 1,
                          "d": int(x["direction"])})
            if a.both_directions:
                cands.append({"name": f"{x['factor']}__{g}__neg", "base": x["factor"],
                              "gate": g, "gate_type": "market", "s": -1,
                              "d": int(x["direction"])})
    cp = os.path.join(a.out_dir, "cands_market_search.json")
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)
    print(f"→ {cp}  {len(cands)} 个候选（{len(el)} 门控 × {len(b20)} 因子"
          f"{' × 2 方向（**非预注册，仅供探索**）' if a.both_directions else '，s=+1 预注册口径'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
