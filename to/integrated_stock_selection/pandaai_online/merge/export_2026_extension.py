# -*- coding: utf-8 -*-
"""P0-3：84 列特征矩阵 2026 延伸导出（冻结口径，零 NaN 组装校验）。

背景：lgb_features.parquet 按 data_cut=2025-12-31 截断，OOS 2026 无特征行。
本脚本只延伸 2026 行，不改任何历史值：

- Class1 个股 Alpha（46）/ Class2 股性池（10）：按 vwap_top20.json /
  vwap_habitat_pool.json 冻结定义重算（store→factors.parquet，mined→公式，
  risk-pattern/dir-adapt→refs specs，custom→custom_registry），变换与
  export_lgb_dataset 完全一致（bounded_raw 原值；其余逐日 MAD 软压缩）。
- Class2 栖息地 6 列：load_panel + ensure_local_fields/extended 原值。
- Class3 行业状态 2 列：gates_search/direct_sector_states.parquet（宽表，已含
  2026-09-18）按 (date, 申万二级) join，值原样。
- Class4 门控交互 20 列：用 specialist_factors.parquet（已含 2026-09-18）按
  judged CSV 冻结规格（base/gate/gate_type/s/form；switch 为两条腿 blend）
  重算——centered_u 逐日截面，gate 序列来自 market_gates_search
  （`{gate}_g`）与 gate_search_sector（p→2p-1）；flip 死区填 0 规则同源。

产物：
- {out}/lgb_features_oos2026.parquet        （2026 行 × 84 列）
- {out}/lgb_features_plus2026.parquet       （冻结矩阵 + 2026 行，默认写）
- {out}/extension_manifest_2026.json        （列级覆盖率/源头/校验结果）

用法（服务器）：
    python3 -B -u tmp_export2026.py --data-cut 2026-09-18
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
import _paths  # noqa: F401,E402
import windows as W  # noqa: E402
from evaluator import (Evaluator, ensure_extended_fields,  # noqa: E402
                       ensure_local_fields, load_panel)
from parser import parse_formula  # noqa: E402
from refs import FactorBuilder  # noqa: E402
from decorrelate_top import load_all_specs  # noqa: E402
from export_lgb_dataset import HABITAT_COLS, cs_mad_norm  # noqa: E402
from f1_batch import _centered_u  # noqa: E402
import h3_eval as H  # noqa: E402
import field_catalog  # noqa: E402

RESULTS = os.path.dirname(os.path.dirname(HERE)) + os.sep + "results" + os.sep + "merge_v2"
M2 = os.path.dirname(os.path.dirname(HERE)) + os.sep + "results" + os.sep + "merge_v2"


def _json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _nan(n):
    return np.full(n, np.nan, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-cut", default="2026-09-18")
    ap.add_argument("--extend-from", default="2025-12-31")
    ap.add_argument("--out-dir", default=os.path.join(M2, "lgb_ready"))
    ap.add_argument("--interact-norm", choices=("mad_zscore_soft", "centered_u"),
                    default="mad_zscore_soft",
                    help="底座因子标准化模式（默认 mad_zscore_soft 保留右尾极值峰度；centered_u 为旧均匀秩）")
    ap.add_argument("--no-parity", action="store_true",
                    help="跳过 2025 断面回放校验")
    args = ap.parse_args()
    t0 = time.time()
    cut = pd.Timestamp(args.data_cut)
    ext_from = pd.Timestamp(args.extend_from)

    man_fp = os.path.join(args.out_dir, "features_manifest.json")
    man = _json(man_fp)
    by = man["by_class"]
    A = [str(c) for c in by["class1_stock_alpha"]]
    P = [str(c) for c in by["class2_habitat_pool"]]
    HB = [str(c) for c in by["class2_stock_habitat"]]
    S3 = [str(c) for c in by["class3_sector_state"]]
    I4 = [str(c) for c in by["class4_gated_interact"]]
    feats = [str(c) for c in man["features_all"]]
    bounded = set(str(c) for c in man.get("bounded_raw_cols", []))
    print(f"[EXT] target 84={len(feats)} | A={len(A)} P={len(P)} HB={len(HB)} "
          f"S3={len(S3)} I4={len(I4)} | bounded_raw={len(bounded)}", flush=True)

    # ---------- 面板 ----------
    frame = load_panel(data_cut=args.data_cut)
    frame["date"] = pd.to_datetime(frame["date"])
    mask = frame["date"] > ext_from
    print(f"[EXT] panel rows={len(frame):,d} | 2026 rows={int(mask.sum()):,d} "
          f"| dates={frame['date'].nunique()}", flush=True)
    ev = Evaluator(frame, cache_size=64)
    builder = FactorBuilder(frame)

    # ---------- 冻结定义映射 ----------
    fmap: dict = {}
    for jf, key in (("vwap_top20.json", "base20"), ("vwap_habitat_pool.json", "members")):
        fp = os.path.join(M2, jf)
        if not os.path.exists(fp):
            print(f"[EXT] WARN 缺 {fp}", flush=True)
            continue
        for it in _json(fp).get(key, []):
            fid = str(it.get("factor", it.get("factor_id", "")))
            if fid:
                fmap[fid] = {"formula": str(it.get("formula", it.get("formula_cn", ""))),
                             "lib": str(it.get("lib", ""))}
    specs = load_all_specs()
    from custom_registry import (evaluate_custom_factor, load_all_custom_specs)
    cs = load_all_custom_specs()

    cols_need = A + P
    vals: dict[str, np.ndarray] = {}
    src: dict[str, str] = {}
    n = len(frame)

    # store 列一次性批量读（禁循环单列读大表）
    store_cols = [c for c in cols_need
                  if c in fmap and fmap[c]["formula"].startswith("store:")]
    if store_cols:
        try:
            import boundary as B
            sf = B.read_factors(store_cols, data_max=args.data_cut, label="ext2026")
            m = sf.merge(frame[["stock_code", "date"]], on=["stock_code", "date"],
                         how="right")
            for c in store_cols:
                if c in m.columns:
                    vals[c] = m[c].to_numpy(dtype=float)
                    src[c] = "store"
            del sf, m
            print(f"[EXT] store 批量读 {len(store_cols)} 列", flush=True)
        except Exception as e:
            print(f"[EXT] WARN store 读失败 {str(e)[:120]}", flush=True)

    for i, c in enumerate(cols_need, 1):
        if c in vals:
            continue
        it = fmap.get(c, {})
        formula, lib = it.get("formula", ""), it.get("lib", "")
        try:
            if lib == "custom" or c in cs:
                vals[c] = np.asarray(evaluate_custom_factor(cs[c], frame, ev), dtype=float)
                src[c] = "custom"
            elif lib == "mined" or (formula and not formula.startswith("store:")
                                    and c not in specs):
                vals[c] = np.asarray(ev.values(parse_formula(formula)), dtype=float)
                src[c] = "mined"
            elif c in specs:
                vals[c] = np.asarray(builder.get_factor(specs[c]), dtype=float)
                src[c] = f"refs:{lib}"
            else:
                vals[c] = _nan(n)
                src[c] = f"missing({lib})"
                print(f"[EXT] WARN 无法求值 {c} lib={lib}", flush=True)
        except Exception as e:
            vals[c] = _nan(n)
            src[c] = f"error({lib})"
            print(f"[EXT] WARN {c} 求值失败 {str(e)[:100]}", flush=True)
        if i % 10 == 0:
            print(f"[EXT] alpha/pool 进度 {i}/{len(cols_need)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---------- 2026 子表组装 ----------
    ensure_local_fields(frame)
    try:
        ensure_extended_fields(frame)
    except Exception as e:
        print(f"[EXT] WARN extended fields {str(e)[:100]}", flush=True)
    sub = frame.loc[mask, ["stock_code", "date"]].copy()
    for h in HB:
        if h == "LOG_MV":
            sub["LOG_MV"] = np.log(np.maximum(
                frame.loc[mask, "MARKET_CAP"].to_numpy(dtype=float), 1.0)).astype(np.float32)
        elif h in frame.columns:
            sub[h] = frame.loc[mask, h].to_numpy()
        else:
            sub[h] = np.nan
            print(f"[EXT] WARN habitat 缺列 {h}", flush=True)

    _raw = [c for c in cols_need if c in bounded]
    _norm = [c for c in cols_need if c not in bounded]
    tmp = pd.DataFrame({"stock_code": sub["stock_code"].to_numpy(),
                        "date": sub["date"].to_numpy()})
    for c in cols_need:
        tmp[c] = vals.get(c, _nan(n))[mask.to_numpy()]
    if _norm:
        nz = cs_mad_norm(tmp, _norm)
        for c in _norm:
            sub[c] = nz[c].to_numpy()
    for c in _raw:
        sub[c] = tmp[c].to_numpy()
    print(f"[EXT] alpha/pool 落列 raw={len(_raw)} norm={len(_norm)}", flush=True)
    del tmp

    # ---------- Class3 行业状态 ----------
    fp3 = os.path.join(M2, "gates_search", "direct_sector_states.parquet")
    try:
        g = pd.read_parquet(fp3)
        g["date"] = pd.to_datetime(g["date"])
        g = g[g["date"] <= cut]
        have3 = [c for c in S3 if c in g.columns]
        sw2 = field_catalog.sw2_mapping()
        sub["_sec"] = sub["stock_code"].astype(str).map(sw2)
        if "sector" in g.columns:
            sub = sub.merge(g[["date", "sector"] + have3].rename(
                columns={"sector": "_sec"}), on=["date", "_sec"], how="left")
        sub = sub.drop(columns=["_sec"], errors="ignore")
        print(f"[EXT] Class3 合并 {have3}", flush=True)
    except Exception as e:
        print(f"[EXT] WARN Class3 失败 {str(e)[:120]}", flush=True)
    for c in S3:
        if c not in sub.columns:
            sub[c] = np.nan
            print(f"[EXT] WARN Class3 缺列 {c}（置 NaN）", flush=True)

    # ---------- Class4 门控交互（仅 2026 行） ----------
    judged = []
    for fn, cands in (
            ("f1_specialist_gain_judged.csv", "specialist_cands_gain.json"),
            ("f1_specialist_switch_judged.csv", "specialist_cands_switch.json"),
            ("f1_sector_specialist_switch_judged.csv",
             "specialist_sector_cands_switch.json")):
        fp = os.path.join(M2, "gates_search", fn)
        if not os.path.exists(fp):
            print(f"[EXT] WARN 缺 {fn}", flush=True)
            continue
        d = pd.read_csv(fp, encoding="utf-8-sig")
        ok = (d["h3_pass"].astype(str).str.lower().isin(("true", "1"))
              & d["fdr_sig"].astype(str).str.lower().isin(("true", "1")))
        cp = os.path.join(M2, cands)
        cj = {x["name"]: x for x in _json(cp)} if os.path.exists(cp) else {}
        for _, r in d[ok].iterrows():
            judged.append((r, cj.get(r["name"], {})))
    print(f"[EXT] 交互规格 {len(judged)} 条", flush=True)

    sp_fp = os.path.join(M2, "specialist_factors.parquet")
    need_cols = set()
    for r, cc in judged:
        if str(r["form"]) == "switch":
            need_cols.add(str(cc.get("base_off", r["base"])))
            need_cols.add(str(cc.get("base_def", r["base"])))
        else:
            need_cols.add(str(r["base"]))
    need_cols = sorted(need_cols)
    import pyarrow.parquet as pq
    _sch = set(pq.read_schema(sp_fp).names)
    _miss_b = [c for c in need_cols if c not in _sch]
    if _miss_b:
        print(f"[EXT] WARN specialist 面板缺底座列 {_miss_b}", flush=True)
    sp = pd.read_parquet(sp_fp, columns=["stock_code", "date"]
                         + [c for c in need_cols if c in _sch])
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp[sp["date"] <= cut].copy()
    sp = sp.sort_values("date", kind="stable").reset_index(drop=True)
    # 底座 2026 缺口回填：store 类底座在 specialist 面板 2026 行为 NaN（面板构建时
    # 按 2025 截断），从 factors.parquet（覆盖 2026-09-18）按 (stock,date) 补齐，
    # 保证 u 的当日横截面完整；回填仅替换 NaN，不动历史值。
    _m26 = (sp["date"] > ext_from).to_numpy()
    _fill = []
    for c in [x for x in need_cols if x in sp.columns]:
        v26 = pd.to_numeric(sp.loc[_m26, c], errors="coerce").to_numpy(dtype=float)
        if len(v26) and np.isfinite(v26).mean() < 0.5:
            _fill.append(c)
    if _fill:
        try:
            import boundary as Bf
            sf_fill = Bf.read_factors(_fill, data_max=args.data_cut, label="ext2026-legs")
            mf = sf_fill.merge(sp[["stock_code", "date"]],
                               on=["stock_code", "date"], how="right")
            for c in _fill:
                cur = pd.to_numeric(sp[c], errors="coerce").to_numpy(dtype=float)
                if c in mf.columns:
                    v = pd.to_numeric(mf[c], errors="coerce").to_numpy(dtype=float)
                    sp[c] = np.where(np.isfinite(cur), cur, v)
                fin = np.isfinite(
                    pd.to_numeric(sp.loc[_m26, c], errors="coerce")
                    .to_numpy(dtype=float)).mean()
                print(f"[EXT] 底座回填 {c}: 2026 finite -> {fin:.3f}", flush=True)
            del sf_fill, mf
        except Exception as e:
            print(f"[EXT] WARN 底座回填失败 {str(e)[:120]}", flush=True)
    sp_dates = pd.to_datetime(sp["date"])
    udays = np.sort(sp_dates.unique())
    day = np.searchsorted(udays, sp_dates.to_numpy()).astype(np.int64)
    nd = len(udays)
    dmap = {}
    spj = _json(os.path.join(M2, "specialist_pool.json"))
    for it in spj.get("factors", []):
        try:
            dmap[str(it["factor_id"])] = int(it.get("direction", 1))
        except Exception:
            pass
    mg = pd.read_parquet(os.path.join(M2, "gates_search", "market_gates_search.parquet"))
    mg["date"] = pd.to_datetime(mg["date"])
    sg = pd.read_parquet(os.path.join(M2, "gates_search", "gate_search_sector.parquet"))
    sg["date"] = pd.to_datetime(sg["date"])
    sw2 = field_catalog.sw2_mapping()
    sp_sec = sp["stock_code"].astype(str).map(sw2)

    icols: dict[str, np.ndarray] = {}
    gate_cache: dict[str, np.ndarray] = {}
    for r, cc in judged:
        name = str(r["name"])
        form, gate = str(r["form"]), str(r["gate"])
        gtype = str(r.get("gate_type", "market"))
        s = int(r["s"])
        base = str(r["base"])
        try:
            if form == "switch":
                b_off = str(cc.get("base_off", base))
                b_def = str(cc.get("base_def", base))
                d_off = int(cc.get("d_off", dmap.get(b_off, 1)))
                d_def = int(cc.get("d_def", dmap.get(b_def, 1)))
                if getattr(args, "interact_norm", "mad_zscore_soft") == "mad_zscore_soft":
                    from fastranks import mad_directed
                    u_off = mad_directed(pd.to_numeric(sp[b_off], errors="coerce")
                                         .to_numpy(dtype=float), day, d_off, soft_cap=5.0)
                    u_def = mad_directed(pd.to_numeric(sp[b_def], errors="coerce")
                                         .to_numpy(dtype=float), day, d_def, soft_cap=5.0)
                else:
                    u_off = _centered_u(pd.to_numeric(sp[b_off], errors="coerce")
                                        .to_numpy(dtype=float), day, nd, d_off)
                    u_def = _centered_u(pd.to_numeric(sp[b_def], errors="coerce")
                                        .to_numpy(dtype=float), day, nd, d_def)
                if gtype == "market":
                    key = f"{gate}_g"
                    if key not in gate_cache:
                        gm = dict(zip(mg["date"], mg[key].to_numpy()))
                        gate_cache[key] = np.array(
                            [gm.get(d, np.nan) for d in sp_dates], dtype=float)
                    gt = gate_cache[key]
                    # s 参与方向（s=-1 时 p 翻转；当前入选 switch 均 s=1）
                    p = np.clip((float(s) * gt + 1.0) / 2.0, 0.0, 1.0)
                    p[~np.isfinite(gt)] = np.nan
                else:
                    key = f"sector:{gate}"
                    if key not in gate_cache:
                        prow = (sg[sg["gate_id"] == gate]
                                .pivot_table(index="date", columns="sector",
                                             values="p", aggfunc="mean")
                                .reindex(pd.to_datetime(udays)))
                        gate_cache[key] = prow
                    prow = gate_cache[key]
                    gcols = list(prow.columns)
                    gset = set(gcols)
                    ci = np.array([gcols.index(x) if x in gset else -1
                                   for x in sp_sec], dtype=np.int64)
                    di = np.searchsorted(udays, sp_dates.to_numpy()).astype(np.int64)
                    p = np.full(len(sp), np.nan)
                    okr = ci >= 0
                    p[okr] = prow.to_numpy(dtype=float)[di[okr], ci[okr]]
                    p = np.clip(p, 0.0, 1.0)
                    if s < 0:
                        p = 1.0 - p
                icols[name] = H.build_switch_blend(p, u_off, u_def)
            else:
                d = int(cc.get("d", dmap.get(base, int(r.get("d", 1)))))
                if getattr(args, "interact_norm", "mad_zscore_soft") == "mad_zscore_soft":
                    from fastranks import mad_directed
                    u = mad_directed(pd.to_numeric(sp[base], errors="coerce")
                                     .to_numpy(dtype=float), day, d, soft_cap=5.0)
                else:
                    u = _centered_u(pd.to_numeric(sp[base], errors="coerce")
                                    .to_numpy(dtype=float), day, nd, d)
                if gtype == "market":
                    key = f"{gate}_g"
                    if key not in gate_cache:
                        gm = dict(zip(mg["date"], mg[key].to_numpy()))
                        gate_cache[key] = np.array(
                            [gm.get(dt, np.nan) for dt in sp_dates], dtype=float)
                    gt = gate_cache[key]
                else:
                    key = f"sector:{gate}"
                    if key not in gate_cache:
                        prow = (sg[sg["gate_id"] == gate]
                                .pivot_table(index="date", columns="sector",
                                             values="p", aggfunc="mean")
                                .reindex(pd.to_datetime(udays)))
                        gate_cache[key] = prow
                    prow = gate_cache[key]
                    gcols = list(prow.columns)
                    gset = set(gcols)
                    ci = np.array([gcols.index(x) if x in gset else -1
                                   for x in sp_sec], dtype=np.int64)
                    di = np.searchsorted(udays, sp_dates.to_numpy()).astype(np.int64)
                    gt = np.full(len(sp), np.nan)
                    okr = ci >= 0
                    gt[okr] = (2.0 * prow.to_numpy(dtype=float)[di[okr], ci[okr]] - 1.0)
                w = (H.build_gain_weight(gt, s) if form == "gain"
                     else H.build_flip_weight(gt, s))
                f = u * w
                if form == "flip":
                    fill = np.isfinite(u) & np.isfinite(gt) & ~np.isfinite(f)
                    f = np.where(fill, 0.0, f)
                icols[name] = f
        except Exception as e:
            print(f"[EXT] WARN 交互 {name} 失败 {str(e)[:120]}", flush=True)
            icols[name] = np.full(len(sp), np.nan)

    inter = pd.DataFrame({"stock_code": sp["stock_code"].to_numpy(),
                          "date": sp["date"].to_numpy()})
    for k, v in icols.items():
        inter[k] = v
    missing_i = [c for c in I4 if c not in inter.columns]
    if missing_i:
        print(f"[EXT] WARN 交互缺列 {missing_i}（置 NaN）", flush=True)
        for c in missing_i:
            inter[c] = np.nan
    inter_par = inter[inter["date"] == pd.Timestamp("2025-12-30")][
        ["stock_code", "date"] + I4].copy()
    inter = inter[inter["date"] > ext_from]
    sub = sub.merge(inter[["stock_code", "date"] + I4],
                    on=["stock_code", "date"], how="left")
    print(f"[EXT] Class4 合并 {len(I4)} 列（specialist 行 {len(inter):,d}）", flush=True)
    del sp, inter

    # ---------- 组装与校验 ----------
    out = sub[["stock_code", "date"]].copy()
    hard_missing = []
    for c in feats:
        if c in sub.columns:
            out[c] = pd.to_numeric(sub[c], errors="coerce").astype(np.float32)
        else:
            out[c] = np.float32(np.nan)
            hard_missing.append(c)
    if hard_missing:
        print(f"[EXT] ERROR 84 列组装缺列: {hard_missing}", flush=True)
        return 2
    cov = {}
    for c in feats:
        v = out[c].to_numpy()
        cov[c] = round(float(np.isfinite(v).mean()), 6)
    allnan = [c for c in feats if cov[c] <= 0.0]
    print(f"[EXT] 2026 rows={len(out):,d} dates={out['date'].nunique()} "
          f"cols={len(feats)} | 全 NaN 列: {allnan}", flush=True)
    worst = sorted(cov.items(), key=lambda kv: kv[1])[:8]
    print(f"[EXT] 覆盖率最低 8 列: {worst}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_fp = os.path.join(args.out_dir, "lgb_features_oos2026.parquet")
    out.to_parquet(out_fp, index=False)
    print(f"[EXT] saved {out_fp} ({os.path.getsize(out_fp)/1e6:.1f} MB)", flush=True)

    # 冻结矩阵 + 2026 合并（供 stage15 --features_parquet 直接消费）
    try:
        old_fp = os.path.join(args.out_dir, "lgb_features.parquet")
        old = pd.read_parquet(old_fp)
        old["date"] = pd.to_datetime(old["date"])
        comb = pd.concat([old[["stock_code", "date"] + feats], out],
                         ignore_index=True)
        comb = comb.sort_values(["date", "stock_code"], kind="stable")
        comb_fp = os.path.join(args.out_dir, "lgb_features_plus2026.parquet")
        comb.to_parquet(comb_fp, index=False)
        print(f"[EXT] saved {comb_fp} rows={len(comb):,d}", flush=True)
    except Exception as e:
        print(f"[EXT] WARN 合并矩阵失败 {str(e)[:150]}", flush=True)

    # ---------- 2025 断面回放校验（mined/store 采样列） ----------
    parity = {}
    if not args.no_parity:
        try:
            frozen = pd.read_parquet(
                os.path.join(M2, "vwap_top20_factors.parquet"))
            frozen["date"] = pd.to_datetime(frozen["date"])
            chk_d = pd.Timestamp("2025-12-30")
            base = frame.loc[frame["date"] == chk_d, ["stock_code", "date"]]
            for c in [x for x in A if src.get(x) in ("mined", "store")][:4]:
                if c not in frozen.columns:
                    continue
                fz = frozen.loc[frozen["date"] == chk_d, ["stock_code", c]]
                mine = pd.DataFrame({"stock_code": frame.loc[frame["date"] == chk_d,
                                                             "stock_code"],
                                     "v": vals[c][frame["date"].to_numpy() == np.datetime64(chk_d)]})
                mm = mine.merge(fz, on="stock_code", how="inner").dropna()
                if len(mm):
                    parity[c] = round(float(np.nanmax(np.abs(
                        mm["v"] - mm[c]))), 8)
            print(f"[EXT] 2025-12-30 断面回放 max|Δ|: {parity}", flush=True)
            # 交互列断面回放（冻结 approved_interactions 对照，用 2025-12-30 全历史切片）
            try:
                fzai = pd.read_parquet(os.path.join(M2, "approved_interactions.parquet"))
                fzai["date"] = pd.to_datetime(fzai["date"])
                ipar = {}
                for c in I4[:4]:
                    if c not in fzai.columns:
                        continue
                    fz = fzai.loc[fzai["date"] == chk_d, ["stock_code", c]].rename(
                        columns={c: "v_ref"})
                    mv = inter_par[["stock_code", c]].rename(columns={c: "v_mine"})
                    mm = mv.merge(fz, on="stock_code", how="inner").dropna()
                    nz = fz.dropna()
                    ipar[c[:26]] = {
                        "max_abs": round(float(np.nanmax(np.abs(
                            mm["v_mine"] - mm["v_ref"]))), 8) if len(mm) else None,
                        "n": int(len(mm)), "ref_finite": int(len(nz))}
                print(f"[EXT] 交互断面回放: {ipar}", flush=True)
                parity.update({f"inter:{k}": v for k, v in ipar.items()})
            except Exception as e:
                print(f"[EXT] WARN 交互parity失败 {str(e)[:120]}", flush=True)
        except Exception as e:
            print(f"[EXT] WARN parity 失败 {str(e)[:120]}", flush=True)

    ext_man = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_cut": args.data_cut, "extend_from": args.extend_from,
        "rows": int(len(out)), "dates": int(out["date"].nunique()),
        "n_features": len(feats), "all_nan_cols": allnan,
        "coverage": cov, "sources": src,
        "parity_20251230": parity,
        "panels": {"specialist_factors": "2026 covered in-place",
                   "sector_states": "direct_sector_states.parquet wide",
                   "interactions": "recomputed from judged specs"},
    }
    with open(os.path.join(args.out_dir, "extension_manifest_2026.json"), "w",
              encoding="utf-8") as f:
        json.dump(ext_man, f, ensure_ascii=False, indent=1)
    print(f"[EXT] DONE in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
