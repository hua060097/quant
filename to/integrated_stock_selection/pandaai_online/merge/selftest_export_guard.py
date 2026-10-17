# -*- coding: utf-8 -*-
"""export矩阵永久守卫：不得含c2c-only因子（c2c榜为审计轨，vwap榜为入模轨）。

检查：
 1. features_manifest.json 的 features_all 中，凡出现在 flat 且为 c2c-only
    （仅在c2c榜、不在vwap榜）的因子 → 违规；
 2. roster（vwap_top20.json base20）须全部在vwap榜。
数据缺失（本地无服务器产物）时 SKIP 退出（rc=0），不误报。
用法: python selftest_export_guard.py [--flat ...] [--manifest ...]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
OUT = os.path.join(HERE, "out")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", default="")
    ap.add_argument("--manifest",
                    default=os.path.join(RESULTS, "lgb_ready",
                                         "features_manifest.json"))
    a = ap.parse_args()
    flat_path = a.flat or ""
    if not flat_path:
        cands = sorted(glob.glob(os.path.join(OUT, "ALIGNED_TOP2_*_flat.csv")),
                       key=lambda p: os.path.getmtime(p))
        flat_path = cands[-1] if cands else ""
    if not flat_path or not os.path.exists(a.manifest):
        print(f"SKIP: 产物缺失（flat={flat_path or '无'} "
              f"manifest={a.manifest if os.path.exists(a.manifest) else '无'}）"
              f"——在服务器上运行以获得真实守卫检查")
        return 0
    import pandas as pd
    flat = pd.read_csv(flat_path, encoding="utf-8-sig")
    vwap_ids = set(flat[flat["label"] == "vwap"]["factor_id"].astype(str))
    c2c_ids = set(flat[flat["label"] == "c2c"]["factor_id"].astype(str))
    c2c_only = c2c_ids - vwap_ids
    mani = json.load(open(a.manifest, encoding="utf-8"))
    feats = list(mani.get("features_all", []))
    bad = sorted(set(feats) & c2c_only)
    print(f"flat={os.path.basename(flat_path)} vwap={len(vwap_ids)} "
          f"c2c-only={len(c2c_only)} features={len(feats)}")
    roster_path = os.path.join(RESULTS, "vwap_top20.json")
    if os.path.exists(roster_path):
        roster = {str(x["factor"]) for x in
                  json.load(open(roster_path, encoding="utf-8")).get("base20", [])}
        bad_roster = sorted(roster - vwap_ids)
        if bad_roster:
            print(f"FAIL roster含非vwap榜因子: {bad_roster[:8]}")
            return 1
        print(f"roster守卫: {len(roster)} 全在vwap榜 PASS")
    if bad:
        print(f"FAIL export矩阵含c2c-only因子: {bad[:8]}")
        return 1
    print(f"export守卫: {len(feats)} 特征无c2c-only PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
