# -*- coding: utf-8 -*-
"""离线已认证门控字典构建器（V4.2 低频层，不进日常流水线）。

定位：`gate_search_space.py`（枚举54规格）+ `run_gate_pipeline.py`（市场G0~G4/G2/G3）
+ `gate_g2g3_sector.py`（行业G2/G3）跑完后，本脚本**一次性**把短名单固化为静态字典。
日常因子搜索/F1只读加载本字典，零重复筛选开销。

输入（全部只读）：
  - 市场短名单：gates_search/pipeline_shortlist.csv（g2_pass & neutral 行）
  - 市场G4聚类：gates_search/g4_clusters_market_pipeline.csv（若缺则不做簇去重，如实记录）
  - 行业G2/G3：gates_search/g2g3_sector.csv（g2_pass 行）
  - 行业面板：gates_search/gate_search_sector.parquet（为 direct states 算截面相关）
  - 注册表：gates_search/gate_search_registry.json（面板sha/qwin）

输出：
  - gates_search/certified_gate_bank.json（版本化清单：gate_id/level/base/op/s/统计量/来源）
  - gates_search/direct_sector_states.parquet + .json（2~3个正交纯行业状态，Owner豁免D8入模）

方向s锁定规则（只记录，不改任何冻结结论）：
  - 市场：s = +1 if median_b > 0 else -1（G2组级斜率符号；旧cands硬编码s=+1，差异逐条记录）
  - 行业：s = +1 if delta_median > 0 else -1
日常消费（f1_batch/f1_sector --gate-bank，run_gate_pipeline --gate-bank）用本字典的s覆盖，
默认不传则保持历史行为（s=+1）不变。

用法：
    python build_gate_bank.py --max-market 8 --max-sector 5 --max-sector-direct 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")
BANK_VERSION = "bank-v2"


def _sha12(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def _sign(x) -> int:
    try:
        return 1 if float(x) > 0 else -1
    except Exception:
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space-dir", default=SPACE)
    ap.add_argument("--market-shortlist", default=None)
    ap.add_argument("--market-all", default=None)
    ap.add_argument("--sector-g2g3", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--data-cut", default=W.DEFAULT_DATA_CUT)
    ap.add_argument("--max-market", type=int, default=8)
    ap.add_argument("--max-sector", type=int, default=5)
    ap.add_argument("--max-sector-direct", type=int, default=2,
                    help="直入模型的正交纯行业状态数（默认2，上限3，需Owner豁免记录）")
    ap.add_argument("--g4-thr-note", default="series>=0.8",
                    help="G4簇去重口径备注（复用管线已算簇文件，不重算）")
    a = ap.parse_args()

    space = a.space_dir
    mshort = a.market_shortlist or os.path.join(space, "pipeline_shortlist.csv")
    mall = a.market_all or os.path.join(space, "pipeline_all.csv")
    secg = a.sector_g2g3 or os.path.join(space, "g2g3_sector.csv")
    regp = a.registry or os.path.join(space, "gate_search_registry.json")
    for p in (mshort, secg):
        if not os.path.exists(p):
            raise SystemExit(f"缺少输入 {p}（先跑 run_gate_pipeline.py 与 gate_g2g3_sector.py）")

    mdf = pd.read_csv(mshort)
    mdf = mdf[(mdf.get("g2_pass") == True) & (mdf.get("neutral") == True)].copy()
    if "score" in mdf.columns:
        mdf = mdf.sort_values("score", ascending=False)
    # G4簇去重：复用管线已落盘簇文件（诊断口径series/0.8），每簇留score最高1个
    g4path = os.path.join(space, "g4_clusters_market_pipeline.csv")
    g4_note = "missing"
    if os.path.exists(g4path):
        try:
            keep = set()
            cl = pd.read_csv(g4path)
            score_of = {str(r["gate_id"]): float(r.get("score", 0) or 0)
                        for _, r in mdf.iterrows()}
            for _, r in cl.iterrows():
                members = [x for x in str(r["members"]).split(";") if x in score_of]
                if members:
                    keep.add(max(members, key=lambda x: score_of[x]))
            # 单成员簇的门控也在members列中，已被keep覆盖；短名单中不在任何簇的保留
            clustered = {x for _, r in cl.iterrows() for x in str(r["members"]).split(";")}
            for _, r in mdf.iterrows():
                if str(r["gate_id"]) not in clustered:
                    keep.add(str(r["gate_id"]))
            mdf = mdf[mdf["gate_id"].astype(str).isin(keep)].copy()
            g4_note = f"reused {os.path.basename(g4path)} ({len(cl)} clusters)"
        except Exception as e:
            g4_note = f"cluster-reuse-failed: {str(e)[:100]}"
    mdf = mdf.head(max(0, a.max_market))

    # 断路4修复：direct 不再强制 g2_pass（Class3 背景坐标 ≠ Class4 调节器）。
    # 候选 = 全部已测 sector 门控，按 (g2_pass 优先, |delta| 降序) 排；
    # bank 条目取前 max_sector，direct 在全序上贪心 |rho|<0.7 取满 max_direct。
    # （V4.3a 起 14/14 全测，无需 screen.csv 未测量回填；未测门控诚实排除。）
    sdf_all = pd.read_csv(secg)
    sdf_all["_absd"] = sdf_all["delta_median"].abs()
    sdf_all = sdf_all.sort_values(["g2_pass", "g3_pass", "_absd"],
                                  ascending=[False, False, False])
    # entries（Class4 认证交互）：g2_pass 且（G3 未测 或 g3_pass）。
    # G3 默认跳过时 g3_tested=False → 只要求 g2；跑过 G3 则必须双过。
    # direct（Class3 背景坐标）用全序另行贪心，不在此过滤。
    def _sector_ok(_r) -> bool:
        if _r.get("g2_pass") != True:
            return False
        if "g3_tested" in sdf_all.columns and bool(_r.get("g3_tested")):
            return bool(_r.get("g3_pass"))
        return True
    _passed = sdf_all[sdf_all.apply(_sector_ok, axis=1)].copy()
    if len(_passed):
        sdf = _passed.head(max(0, a.max_sector)).copy()
    else:
        sdf = sdf_all.head(1).copy()
        print("  warning: sector 无合格 entries，取首行兜底", flush=True)

    reg = {}
    if os.path.exists(regp):
        try:
            reg = json.loads(open(regp, encoding="utf-8").read())
        except Exception:
            reg = {}

    entries = []
    for _, r in mdf.iterrows():
        # V4.3c：优先用分簇有效中位数 median_b_eff（--by-group 胜出簇符号），
        # 否则回退全池 median_b；s 来源如实记录。
        mb_eff = r.get("median_b_eff", None)
        mb_eff = float(mb_eff) if mb_eff is not None and np.isfinite(float(mb_eff)) else None
        mb = float(r.get("median_b", 0) or 0)
        mb_use = mb_eff if mb_eff is not None else mb
        _grp = str(r.get("slope_group", "") or "")
        entries.append({"gate_id": str(r["gate_id"]), "level": "market",
                        "base": str(r.get("base", "")), "op": str(r.get("op", "")),
                        "s": _sign(mb_use),
                        "s_source": f"G2 group[{_grp}] median_b_eff sign"
                        if _grp and mb_eff is not None else "G2 median_b sign",
                        "median_b": mb, "median_b_eff": mb_eff,
                        "slope_group": _grp,
                        "same_frac": float(r.get("same_frac", 0) or 0),
                        "score": float(r.get("score", 0) or 0),
                        "g3_pass": bool(r.get("g3_pass", False))})
    for _, r in sdf.iterrows():
        dm = float(r.get("delta_median", 0) or 0)
        # 分簇拯救时用簇中位数定 s（全池 ~0 会锁错方向）。
        _me = r.get("median_b_eff", None)
        _me = float(_me) if _me is not None and np.isfinite(float(_me)) else None
        _dm_use = _me if _me is not None else dm
        _gsrc = (f"sector group[{r.get('slope_group', '')}] median_b_eff sign"
                 if _me is not None else "sector delta_median sign")
        # V4.3a：GS_sec_amp_med_accel → base=sec_amp_med, op=accel
        #（旧 split("_")[1] 恒得 "sec" 且 op 硬编码 level）。
        _tail = str(r["gate_id"])[3:] if str(r["gate_id"]).startswith("GS_") else str(r["gate_id"])
        _base, _, _op = _tail.rpartition("_")
        if not _base:
            _base, _op = _tail, "level"
        entries.append({"gate_id": str(r["gate_id"]), "level": "sector",
                        "base": _base, "op": _op,
                        "s": _sign(_dm_use), "s_source": _gsrc,
                        "delta_median": dm, "median_b_eff": _me,
                        "slope_group": str(r.get("slope_group", "") or ""),
                        "same_frac": float(r.get("same_frac", 0) or 0),
                        "score": None,
                        "g3_pass": bool(r.get("g3_pass", False))})

    # 纯行业直入 states：在**全序**（g2_pass优先，|delta|降序）上贪心取
    # 两两|rho|<0.7者，不限于 bank 条目（Class3 背景坐标不设 G2 硬门）。
    direct = []
    direct_note = "skipped"
    try:
        _ordf = sdf_all.copy()
        sec_ids = list(_ordf["gate_id"].astype(str))
        if sec_ids and a.max_sector_direct > 0:
            secdir = os.path.join(space, "gate_search_sector.parquet")
            sg = pd.read_parquet(secdir, columns=["date", "sector", "gate_id", "p"])
            sg["date"] = pd.to_datetime(sg["date"])
            sg = sg[sg["gate_id"].isin(sec_ids)].copy()
            pivots = {}
            _union = None
            _wide = {}
            for gid, sub in sg.groupby("gate_id"):
                w = sub.pivot_table(index=["date", "sector"], values="p", aggfunc="mean")
                _wide[gid] = w["p"]
                _union = w.index if _union is None else _union.union(w.index)
            for gid, s in _wide.items():
                pivots[gid] = s.reindex(_union).to_numpy(dtype=float)
            order = sec_ids

            def _rho(a_, b_):
                m = np.isfinite(a_) & np.isfinite(b_)
                if m.sum() < 30:
                    return float("nan")
                ra = pd.Series(a_[m]).rank().to_numpy()
                rb = pd.Series(b_[m]).rank().to_numpy()
                ra -= ra.mean()
                rb -= rb.mean()
                den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
                return float((ra * rb).sum() / den) if den > 0 else float("nan")

            for gid in order:
                if len(direct) >= min(a.max_sector_direct, 3):
                    break
                ok = True
                for prev in direct:
                    rho = _rho(pivots[gid], pivots[prev])
                    if np.isfinite(rho) and abs(rho) >= 0.7:
                        ok = False
                        break
                if ok:
                    direct.append(gid)
            # 落盘直入 states（宽表：index (date, sector)，列即 gate_id；
            # 长表直接 merge 会导致一对多笛卡尔膨胀，且 gate_id 字符串列会混入特征。
            # p 中心化为 2p-1（与消费侧口径一致）。
            _w = sg[sg["gate_id"].isin(direct)].pivot_table(
                index=["date", "sector"], columns="gate_id", values="p",
                aggfunc="mean")
            dsg = (2.0 * _w - 1.0).reset_index()
            dsg.to_parquet(os.path.join(space, "direct_sector_states.parquet"), index=False)
            with open(os.path.join(space, "direct_sector_states.json"), "w",
                       encoding="utf-8") as f:
                json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"),
                           "gates": direct,
                           "owner_exemption": "D8 clarification: max 2-3 sector SxT states "
                                              "allowed (Owner 2026-10-15); market 1xT stays 0"},
                          f, ensure_ascii=False, indent=1)
            _npass = int((_ordf["g2_pass"] == True).sum()) \
                if "g2_pass" in _ordf.columns else 0
            direct_note = (f"selected {direct} (|rho|<0.7 greedy, "
                           f"order=g2pass-first+|delta|, {_npass}/{len(_ordf)} pass)")
    except Exception as e:
        direct_note = f"failed: {str(e)[:150]}"

    bank = {"version": BANK_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_cut": W.normalize_data_cut(a.data_cut),
            "panel": {"market_sha12": reg.get("market_sha12"),
                      "sector_sha12": reg.get("sector_sha12"),
                      "qwin": reg.get("qwin"),
                      "market_path": os.path.join(space, "gate_search_market.parquet"),
                      "sector_path": os.path.join(space, "gate_search_sector.parquet")},
            "provenance": {"market_shortlist": mshort, "market_all": mall,
                           "sector_g2g3": secg, "registry": regp,
                           "g4": g4_note},
            "n_market": int((pd.DataFrame(entries)["level"] == "market").sum()) if entries else 0,
            "n_sector": int((pd.DataFrame(entries)["level"] == "sector").sum()) if entries else 0,
            "s_locking_note": "legacy cands used s=+1; bank locks s from G2 sign; "
                              "daily F1 must use bank s via --gate-bank",
            "direct_sector_states": direct,
            "direct_note": direct_note,
            "entries": entries}
    outp = os.path.join(space, "certified_gate_bank.json")
    # 版本归档：覆写前若存在旧版 bank，原样归档（禁静默覆盖）。
    if os.path.exists(outp):
        try:
            _old = json.load(open(outp, encoding="utf-8"))
            _ov = str(_old.get("version", "unknown"))
            if _ov != BANK_VERSION:
                _arc = os.path.join(space, f"certified_gate_bank_{_ov}.json")
                if not os.path.exists(_arc):
                    with open(_arc, "w", encoding="utf-8") as _f:
                        json.dump(_old, _f, ensure_ascii=False, indent=1)
                    print(f"  旧版归档 → {_arc}")
        except Exception as _e:
            print(f"  旧版归档跳过（{str(_e)[:80]}）")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=1)
    print(f"bank -> {outp} (market {bank['n_market']} / sector {bank['n_sector']})")
    print(f"  G4: {g4_note}")
    print(f"  direct states: {direct_note}")
    for e in entries:
        print(f"  {e['gate_id']:<28s} {e['level']:<6s} s={e['s']:+d} "
              f"{e.get('median_b', e.get('delta_median')):+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
