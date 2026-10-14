# -*- coding: utf-8 -*-
"""生成 F1 候选定义 JSON（供 f1_batch.py / f1_sector.py 使用）。

候选格式（见 f1_batch.py 头）：[{name, base, gate, gate_type, s, d}]
- base：base20 因子名；d：其方向（base20.json）
- gate：门控名；gate_type：market/sector（由门控前缀决定）
- s：门控方向，沿用既有 F1 一律 +1

用法：
    python make_f1_cands.py --gates M3 --out cands_M3.json
    python make_f1_cands.py --gates S1 S2 S3 --out cands_sector.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 F1 候选定义 JSON")
    ap.add_argument("--gates", nargs="+", required=True)
    ap.add_argument("--s", type=int, default=1, help="门控方向（默认 +1）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b20 = json.loads(open(os.path.join(RESULTS, "base20.json"),
                          encoding="utf-8").read())["base20"]
    cands = []
    for g in a.gates:
        gt = "market" if g.startswith("M") else "sector"
        for x in b20:
            cands.append({"name": f"{x['factor']}__{g}__pos",
                          "base": x["factor"], "gate": g, "gate_type": gt,
                          "s": int(a.s), "d": int(x["direction"])})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)
    mk = sum(1 for c in cands if c["gate_type"] == "market")
    sc = len(cands) - mk
    print(f"门控 {a.gates} → {len(cands)} 个候选（market {mk} / sector {sc}）"
          f" → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
