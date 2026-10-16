# -*- coding: utf-8 -*-
"""合格交互候选实例化落盘（V4.2）：judged CSV → approved_interactions.parquet。

只实例化 h3_pass & fdr_sig 的配对（F1v3 gain/flip/switch；g/t 旧族不进此表）。
特征值与 F1 计算同定义（u 有向居中秩 + H 三权重 helper），原值落盘，
下游严禁再做单日截面标准化（见 export_lgb_dataset skip_daily_cs_norm_cols）。

flip 死区 fill 规则（Owner/Gemini 终审）：
  仅当 u 有限、g 有限、但权重为 NaN（即死区 |g|<0.10）时填 0.0；
  u 本身缺失（停牌/未上市）保持 NaN；g 缺失日保持 NaN。
gain/switch 不填（NaN 保持）。

用法：
    python materialize_interactions.py --judged f1_specialist_gain.csv \\
        --cands specialist_cands_gain.json --base-panel specialist
    python materialize_interactions.py --judged ... --dry-run   # 只打印不写盘
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
import windows as W  # noqa: E402
import h3_eval as H  # noqa: E402
from f1_batch import _centered_u  # noqa: E402  与F1同一定義，防口径分叉

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _load_base_panel(base_panel: str, need_cols: list[str]):
    """返回 (m_base: stock_code/date/因子列, d_map)。m 未合标签（落盘无需标签）。"""
    if base_panel == "base20":
        import boundary as B
        fac = B.read_factors(need_cols, data_max=W.LABEL_MAX_DATE, label="materialize")
        fac["date"] = pd.to_datetime(fac["date"])
        b20 = {b["factor"]: b["direction"] for b in
               json.load(open(os.path.join(RESULTS, "base20.json"),
                              encoding="utf-8"))["base20"]}
        return fac, b20
    _vp = os.path.join(RESULTS, "vwap_top20_factors.parquet"
                       if base_panel == "vwap_top20" else "specialist_factors.parquet")
    _vj = os.path.join(RESULTS, "vwap_top20.json"
                       if base_panel == "vwap_top20" else "specialist_pool.json")
    if not os.path.exists(_vp):
        raise SystemExit(f"--base-panel {base_panel} 需要 {_vp}")
    fac = pd.read_parquet(_vp)
    fac["date"] = pd.to_datetime(fac["date"])
    jm = json.load(open(_vj, encoding="utf-8"))
    items = jm.get("base20", []) if base_panel == "vwap_top20" else jm.get("factors", [])
    key = "factor" if base_panel == "vwap_top20" else "factor_id"
    dmap = {b[key]: int(b["direction"]) for b in items}
    missing = [c for c in need_cols if c not in fac.columns]
    if missing:
        raise SystemExit(f"{base_panel}面板缺列 {len(missing)} 个: {missing[:5]}")
    return fac, dmap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judged", required=True, help="judge_f1 输出的 judged CSV")
    ap.add_argument("--cands", required=True, help="同批 F1 的 cands JSON（switch 取腿方向）")
    ap.add_argument("--base-panel", choices=("base20", "vwap_top20", "specialist"),
                    required=True)
    ap.add_argument("--gates-market", default=os.path.join(SPACE, "market_gates_search.parquet"))
    ap.add_argument("--gates-sector", default=os.path.join(SPACE, "gate_search_sector.parquet"))
    ap.add_argument("--label-max", default=W.FACTOR_SCREEN_MAX)
    ap.add_argument("--out", default=os.path.join(RESULTS, "approved_interactions.parquet"))
    ap.add_argument("--dry-run", action="store_true", help="只打印将落盘的列与行数，不写盘")
    a = ap.parse_args()

    dj = pd.read_csv(a.judged, encoding="utf-8-sig")
    ok = _as_bool(dj["h3_pass"]) & _as_bool(dj["fdr_sig"])
    passed = dj[ok].copy()
    print(f"judged {len(dj)} 行 → 通过 {len(passed)} 行", flush=True)
    if not len(passed):
        # V4.3a：空通过时若目标已存在（前序算子产物），不得清空覆盖。
        if os.path.exists(a.out):
            print(f"无通过配对且 {a.out} 已存在：保留原文件不覆盖", flush=True)
            return 0
        print("无通过配对，落盘为空表（分支畅通证明见合成自测）", flush=True)
        if not a.dry_run:
            pd.DataFrame(columns=["stock_code", "date"]).to_parquet(a.out, index=False)
        return 0

    cands = {c["name"]: c for c in json.load(open(a.cands, encoding="utf-8"))}
    need = set()
    for _, r in passed.iterrows():
        c = cands.get(r["name"], {})
        need.add(r["base"] if r["form"] in ("g", "t", "gain", "flip")
                 else c.get("base_off", r["base"]))
        if r["form"] == "switch":
            need.add(c.get("base_def", r["base"]))
    need = sorted(x for x in need if x)
    m, dmap = _load_base_panel(a.base_panel, need)
    m = m[m["date"] <= pd.Timestamp(a.label_max)].copy()
    # blockwise_rank 要求 day 块连续：必须先按 date 稳定排序（否则碎块数≈行数，直接卡死）
    m = m.sort_values("date", kind="stable").reset_index(drop=True)
    dates = pd.to_datetime(m["date"])
    udays = np.sort(dates.unique())
    day = np.searchsorted(udays, dates).astype(np.int64)
    nd = len(udays)

    # 方向以 cands 为准（switch 腿方向只存在于 cands），dmap 仅作回退
    for _cn, _cc in cands.items():
        if _cc.get("base"):
            dmap.setdefault(_cc["base"], int(_cc.get("d", 1)))
        if _cc.get("base_off"):
            dmap[_cc["base_off"]] = int(_cc.get("d_off", 1))
        if _cc.get("base_def"):
            dmap[_cc["base_def"]] = int(_cc.get("d_def", 1))
    u_of: dict[str, np.ndarray] = {}
    for col in need:
        d = int(dmap.get(col, 1))
        xv = pd.to_numeric(m[col], errors="coerce").to_numpy(dtype=float)
        u_of[col] = _centered_u(xv, day, nd, d)

    mgates = pd.read_parquet(a.gates_market)
    mgates["date"] = pd.to_datetime(mgates["date"])
    sgates = pd.read_parquet(a.gates_sector)
    sgates["date"] = pd.to_datetime(sgates["date"])
    import gates as G  # noqa: E402  (ix_build 已归档；同文件同值已验证)
    secmap = G.load_sector_map()
    msec = m["stock_code"].astype(str).map(secmap)

    out = pd.DataFrame({"stock_code": m["stock_code"].to_numpy(),
                        "date": pd.to_datetime(m["date"]).dt.date.astype(str).to_numpy()})
    n_filled = 0
    for _, r in passed.iterrows():
        form, gate = str(r["form"]), str(r["gate"])
        s = int(r["s"])
        col = f"{r['base']}__{gate}__{form}"
        if str(r.get("gate_type", "market")) == "market":
            col_g = gate + "_g"
            if col_g not in mgates.columns:
                print(f"  跳过 {r['name']}：市场面板缺列 {col_g}", flush=True)
                continue
            gmap = dict(zip(mgates["date"].dt.date.astype(str),
                            mgates[col_g].to_numpy()))
            gt = np.array([gmap.get(str(d), np.nan)
                           for d in pd.to_datetime(m["date"]).dt.date.astype(str)],
                          dtype=float)
            if form == "gain":
                out[col] = u_of[r["base"]] * H.build_gain_weight(gt, s)
            elif form == "flip":
                u = u_of[r["base"]]
                w = H.build_flip_weight(gt, s)
                f = u * w
                fill = np.isfinite(u) & np.isfinite(gt) & ~np.isfinite(f)
                n_filled += int(fill.sum())
                f = np.where(fill, 0.0, f)
                out[col] = f
            elif form == "switch":
                c = cands.get(r["name"], {})
                p = np.clip((gt + 1.0) / 2.0, 0.0, 1.0)
                p[~np.isfinite(gt)] = np.nan
                out[col] = H.build_switch_blend(
                    p, u_of[c.get("base_off", r["base"])],
                    u_of[c.get("base_def", r["base"])])
            else:
                print(f"  跳过 {r['name']}：旧族 {form} 不进此表", flush=True)
                continue
        else:
            sub = sgates[sgates["gate_id"] == gate]
            prow = sub.pivot_table(index="date", columns="sector", values="p",
                                   aggfunc="mean")
            # V4.3a：日期对齐——sgates 从 2020-01 起而 udays 从面板起始日起，
            # 不 reindex 会使 Gv[di] 取错 1.5 年（与 f1_sector:176 同口径）。
            prow = prow.reindex(pd.to_datetime(udays))
            if form == "gain":
                G = 2.0 * prow - 1.0
            elif form in ("flip", "switch"):
                G = 2.0 * prow - 1.0
                P = prow
            else:
                print(f"  跳过 {r['name']}：旧族 {form} 不进此表", flush=True)
                continue
            gcols, gset = list(G.columns), set(G.columns)
            ci = np.array([gcols.index(x) if x in gset else -1 for x in msec])
            di = np.searchsorted(udays, dates.to_numpy()).astype(np.int64)
            Gv = G.to_numpy(dtype=float)
            okrow = (ci >= 0)
            gv = np.full(len(m), np.nan)
            gv[okrow] = Gv[di[okrow], ci[okrow]]
            if form in ("gain", "flip"):
                u = u_of[r["base"]]
                w = H.build_gain_weight(gv, s) if form == "gain" \
                    else H.build_flip_weight(gv, s)
                f = u * w
                if form == "flip":
                    fill = np.isfinite(u) & np.isfinite(gv) & ~np.isfinite(f)
                    n_filled += int(fill.sum())
                    f = np.where(fill, 0.0, f)
                out[col] = f
            else:
                c = cands.get(r["name"], {})
                Pv = P.to_numpy(dtype=float)
                pv = np.full(len(m), np.nan)
                pv[okrow] = Pv[di[okrow], ci[okrow]]
                pv = np.clip(pv, 0.0, 1.0)
                out[col] = H.build_switch_blend(
                    pv, u_of[c.get("base_off", r["base"])],
                    u_of[c.get("base_def", r["base"])])
    out = out.drop(columns=[c for c in out.columns
                            if c not in ("stock_code", "date")
                            and out[c].isna().all()])
    print(f"特征列 {len(out.columns) - 2} 个，行 {len(out)}，flip死区填0 {n_filled} 格",
          flush=True)
    if a.dry_run:
        print(f"dry-run：{list(out.columns)}", flush=True)
        return 0
    # V4.3a：三式分批落盘合并——目标已存在且非空时按键合并新列，不覆盖前序算子产物。
    if os.path.exists(a.out):
        try:
            _old = pd.read_parquet(a.out)
            if len(_old) and set(("stock_code", "date")).issubset(_old.columns):
                _old["date"] = _old["date"].astype(str)
                _new = out.copy()
                _new["date"] = _new["date"].astype(str)
                _merged = _old.merge(_new, on=["stock_code", "date"], how="outer",
                                     suffixes=("", "_new"))
                _dup = [c for c in _merged.columns if c.endswith("_new")]
                for c in _dup:
                    _base = c[:-4]
                    _merged[_base] = _merged[_base].combine_first(_merged[c])
                    _merged = _merged.drop(columns=[c])
                out = _merged
                print(f"合并既有 {a.out}：{_old.shape} + 新 {len(_new)} 行", flush=True)
        except Exception as e:
            print(f"  既有文件合并失败（{str(e)[:100]}），直接覆盖", flush=True)
    out.to_parquet(a.out, index=False)
    print(f"落盘 → {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
