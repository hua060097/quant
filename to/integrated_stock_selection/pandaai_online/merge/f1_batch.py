# -*- coding: utf-8 -*-
"""F1 批量： market 门控用掩码捷径（已证明等价），sector 门控用全重算。
用法: python f1_batch.py --cands cand_w0.json --out f1_w0.csv --tag w0
cands JSON: [{name, base, gate, gate_type(market/sector), s, d}]
输出每行: H3 全部原始统计量（§8.4），失败原因记入 fail_reason。
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
import boundary as B
import eval_v2 as E
import h3_eval as H
import windows as W

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
# 搜索窗口改为单一事实源（windows.py）。原为硬编码 "2024-12-26"，即 P0/P1
# 时期的旧窗口；现默认生产版口径 signal≤2026-09-11 / labels≤2026-09-18。
# 复现旧结果请显式传 --label-max 2024-12-26。
LABEL_MAX = W.FACTOR_SCREEN_MAX
N_PERM = 200


# 门控文件路径（默认 = 搜索空间产出的市场候选池；2026-10-12 起旧手工门控
# `gates/market_gates.parquet` 已删除——门控定义由搜索生成）
GATES_PARQUET = os.path.join(RESULTS, "gates_search", "market_gates_search.parquet")


def load_gates():
    mg = pd.read_parquet(GATES_PARQUET)
    mg["date"] = pd.to_datetime(mg["date"])
    return mg


def _centered_u(values: np.ndarray, day: np.ndarray, nd: int, d: int) -> np.ndarray:
    """有向居中百分位秩（F1v3 全族统一定义，见 fastranks.centered_u 本体；
    此别名保留供 materialize 等外部导入，防口径分叉）。"""
    import fastranks as FR
    return FR.centered_u(values, day, nd, d)


def build_market_bundle(m, base_cols_all: list[str], d_map: dict) -> dict:
    """批内共享预计算（提速）：day/yrk/u_cols/Ubar 只算一次。

    前提：m 已按 date 稳定排序（main 保证），则 _prepared 的 day 码与
    uniq 日期顺序严格对齐，门控日序列可直接用整数索引广播。
    """
    import fastranks as FR  # noqa: F401
    day, _, _, yrk, nd = E._prepared(m, base_cols_all[0], "r3")
    u_cols: dict[str, np.ndarray] = {}
    for bcol in base_cols_all:
        xv = pd.to_numeric(m[bcol], errors="coerce").to_numpy(dtype=float)
        u_cols[bcol] = _centered_u(xv, day, nd, int(d_map.get(bcol, 1)))
    Ubar = np.nanmean(np.column_stack([u_cols[b] for b in base_cols_all]), axis=1)
    return {"day": day, "nd": nd, "yrk": yrk, "u_cols": u_cols, "Ubar": Ubar}


def run_one_market_gain(m, base_cols_all: list[str], d_map: dict, k_col: str,
                        gate_series: np.ndarray, s: int, uniq_dates: np.ndarray,
                        prep: dict | None = None) -> dict:
    """F1v3_gain市场分支：底仓倾斜增量（精确重算，无掩码捷径）。

    Ubar = K个u的等权均值；S_static = Ubar+0.25*u_k；S_gated = Ubar+0.25*u_k*w_t，
    w_t = clip(1+0.5*s*g_t, 0.2, 1.8)。ΔIC为两套多因子分数的日Spearman差，
    placebo为门控日序列块置换后全重算（与sector线同成本量级，仅供短名单规模批次）。
    prep 为 build_market_bundle 预计算（main 逐批一次）；缺省则现算（兼容旧调用）。
    """
    import fastranks as FR
    if prep is None:
        prep = build_market_bundle(m, base_cols_all, d_map)
    day, nd, yrk = prep["day"], prep["nd"], prep["yrk"]
    u_cols, Ubar = prep["u_cols"], prep["Ubar"]
    g = np.asarray(gate_series, dtype=float)
    gt = g[day] if len(g) == int(nd) else np.full(len(day), np.nan)
    uk = u_cols[k_col]
    w = np.clip(1.0 + 0.5 * float(s) * gt, 0.2, 1.8)
    w[~np.isfinite(gt)] = np.nan
    Fk = uk * w

    def _ic_of(scores: np.ndarray) -> np.ndarray:
        fr = FR.blockwise_rank(np.where(np.isfinite(scores), scores, np.nan), day)
        return E._daily_corr(fr, yrk, day, nd)

    S_static = Ubar + 0.25 * uk
    S_gated = Ubar + 0.25 * Fk
    ic_static = _ic_of(S_static)

    def _gated_ic(gp: np.ndarray) -> np.ndarray:
        gtr = np.asarray(gp, dtype=float)[day]
        wr = np.clip(1.0 + 0.5 * float(s) * gtr, 0.2, 1.8)
        wr[~np.isfinite(gtr)] = np.nan
        return _ic_of(Ubar + 0.25 * uk * wr)

    ic_gated = _gated_ic(g)
    res = {"delta": H.delta_ic(np.asarray(uniq_dates), ic_gated, ic_static)}
    res["slope"] = H.slope_test(ic_static, g)
    res["placebo"] = H.block_placebo(_gated_ic, g, ic_static, n_perm=N_PERM, seed=0)
    res["_ic_static"] = ic_static
    res["_ic_gated"] = ic_gated
    return res


def run_one_market(m, base_col, d, gate_series, s, uniq_dates, ic_z):
    """掩码捷径：on-day 上 F1 与 d·x 秩相同（已证明），只置换掩码。"""
    g = np.asarray(gate_series, dtype=float)
    on = np.isfinite(g) & (s * g > 0)
    f1_ic = np.where(on, ic_z, np.nan)

    def fn(gp):
        gp = np.asarray(gp, dtype=float)
        o2 = np.isfinite(gp) & (s * gp > 0)
        return np.where(o2, ic_z, np.nan)

    res = {"delta": H.delta_ic(np.asarray(uniq_dates), f1_ic, ic_z)}
    res["slope"] = H.slope_test(ic_z, g)
    res["placebo"] = H.block_placebo(fn, g, ic_z, n_perm=N_PERM, seed=0)
    # spread：F1 在 on-day 的 spread = base spread；off-day NaN
    res["_f1_ic"] = f1_ic
    return res


def run_one_market_flip(m, base_cols_all: list[str], d_map: dict, k_col: str,
                        gate_series: np.ndarray, s: int, uniq_dates: np.ndarray,
                        prep: dict | None = None) -> dict:
    """F1v3_flip市场分支：方向翻转调制。

    F = u*w_t，w_t = s*g_t（死区 NaN）。同日全市场同乘标量，故
    IC_t(F) ≡ sign(w_t)·IC_t(u)（w_t>0 保序、<0 倒序、NaN 缺席）——
    数学恒等，无需 500 万行重排名（含 200 次置换内，提速约 5000 倍）。
    专救无条件IC≈0但条件斜率极大的符号翻转型偏科因子。
    """
    if prep is None:
        prep = build_market_bundle(m, base_cols_all, d_map)
    uk = prep["u_cols"][k_col]
    g = np.asarray(gate_series, dtype=float)
    day, yrk = prep["day"], prep["yrk"]

    def _ic_of_u() -> np.ndarray:
        import fastranks as FR
        fr = FR.blockwise_rank(np.where(np.isfinite(uk), uk, np.nan), day)
        return E._daily_corr(fr, yrk, day, prep["nd"])

    ic_base = _ic_of_u()

    def _gated_ic(gp: np.ndarray) -> np.ndarray:
        return np.sign(H.build_flip_weight(np.asarray(gp, dtype=float), s)) * ic_base

    ic_gated = _gated_ic(g)
    res = {"delta": H.delta_ic(np.asarray(uniq_dates), ic_gated, ic_base)}
    res["slope"] = H.slope_test(ic_base, g)
    res["placebo"] = H.block_placebo(_gated_ic, g, ic_base, n_perm=N_PERM, seed=0)
    res["_ic_static"] = ic_base
    res["_ic_gated"] = ic_gated
    return res


def run_one_market_switch(m, base_cols_all: list[str], d_map: dict,
                          off_col: str, def_col: str,
                          gate_series: np.ndarray, uniq_dates: np.ndarray,
                          prep: dict | None = None) -> dict:
    """F1v3_switch市场分支：双因子软切换（精确重算）。

    F = p*u_off + (1-p)*u_def，p∈[0,1]为门控分位；主判定为单因子ΔIC
    （F vs 静态等权共识 0.5*u_off+0.5*u_def，与行业线统一，V4.2收敛）；
    底仓倾斜（Ubar+0.25*blend vs Ubar+0.25*共识）仅作辅助列，不进placebo/FDR。
    prep 为 build_market_bundle 预计算（main 逐批一次）；缺省则现算。
    """
    import fastranks as FR
    if prep is None:
        prep = build_market_bundle(m, base_cols_all, d_map)
    day, nd, yrk = prep["day"], prep["nd"], prep["yrk"]
    u_cols, Ubar = prep["u_cols"], prep["Ubar"]
    xoff = pd.to_numeric(m[off_col], errors="coerce").to_numpy(dtype=float)
    xdef = pd.to_numeric(m[def_col], errors="coerce").to_numpy(dtype=float)
    u_off = _centered_u(xoff, day, nd, int(d_map.get(off_col, 1)))
    u_def = _centered_u(xdef, day, nd, int(d_map.get(def_col, 1)))
    g = np.asarray(gate_series, dtype=float)
    p = np.clip((g + 1.0) / 2.0, 0.0, 1.0)
    p[~np.isfinite(g)] = np.nan
    pt = p[day] if len(p) == int(nd) else np.full(len(day), np.nan)
    blend = H.build_switch_blend(pt, u_off, u_def)
    static_blend = 0.5 * u_off + 0.5 * u_def

    def _ic_of(scores: np.ndarray) -> np.ndarray:
        fr = FR.blockwise_rank(np.where(np.isfinite(scores), scores, np.nan), day)
        return E._daily_corr(fr, yrk, day, nd)

    # 主判定：单因子静态共识对照（与行业线统一）
    ic_static_single = _ic_of(static_blend)

    def _gated_ic(gp: np.ndarray) -> np.ndarray:
        _gp = np.asarray(gp, dtype=float)
        pp = np.clip((_gp + 1.0) / 2.0, 0.0, 1.0)
        pp[~np.isfinite(_gp)] = np.nan
        return _ic_of(H.build_switch_blend(pp[day], u_off, u_def))

    ic_gated = _gated_ic(g)
    res = {"delta": H.delta_ic(np.asarray(uniq_dates), ic_gated, ic_static_single)}
    res["slope"] = H.slope_test(ic_static_single, g)
    res["placebo"] = H.block_placebo(_gated_ic, g, ic_static_single,
                                     n_perm=N_PERM, seed=0)
    # 辅助列：底仓倾斜（诊断用，不进placebo/FDR）
    _ic_tilt_static = _ic_of(Ubar + 0.25 * static_blend)
    _ic_tilt_gated = _ic_of(Ubar + 0.25 * blend)
    _f = np.asarray(_ic_tilt_gated, dtype=float)[np.isfinite(_ic_tilt_gated)]
    _b = np.asarray(_ic_tilt_static, dtype=float)[np.isfinite(_ic_tilt_static)]
    res["tilt_delta"] = float(_f.mean() - _b.mean()) if len(_f) and len(_b) else float("nan")
    res["_ic_static"] = ic_static_single
    res["_ic_gated"] = ic_gated
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="f1")
    ap.add_argument("--label-max", default=LABEL_MAX,
                    help=f"信号日上界（默认选择窗 {W.FACTOR_SCREEN_MAX}；"
                         f"复现 P0/P1 旧结果用 {W.LEGACY_SEARCH_END}）")
    ap.add_argument("--form", default="g", choices=("g", "t", "gain", "flip", "switch"),
                    help="门控组合形式：g=连续(2p-1，on 为 g>0)；"
                         "t=THRESH(p>=0.7 为 on，中间弃权)；"
                         "gain=中心化增益新族F1v3_gain（u*(1+0.5*s*g)+底仓倾斜）；"
                         "flip=方向翻转（u*s*g，救符号翻转偏科因子）；"
                         "switch=双因子软切换（p*u_off+(1-p)*u_def，需cands带base_off/base_def）。"
                         "新三式与g/t平行对照，不得合并FDR。见 PLAN §21")
    ap.add_argument("--base-panel", choices=("base20", "vwap_top20", "specialist"),
                    default="base20",
                    help="基因子面板（默认base20保证历史可复现；"
                         "vwap_top20读build_top动态正交面板+方向；"
                         "specialist读select_specialists偏科池面板+方向）")
    ap.add_argument("--gate-bank", default=None,
                    help="已认证门控字典certified_gate_bank.json（只读消费）："
                         "cands门控过滤到bank清单且s用bank锁定值覆盖；"
                         "不传则保持历史行为（s按cands自带值）")
    ap.add_argument("--gates-parquet", default=None,
                    help="市场门控 parquet 路径（默认 gates_search/market_gates_search.parquet）；"
                         "指向其它候选池时用于批量评估新门控")
    ap.add_argument("--no-f1-cache", dest="no_f1_cache", action="store_true",
                    default=False,
                    help="关闭配对指纹缓存，全部重算（默认启用缓存断点续算）")
    a = ap.parse_args()
    label_max = a.label_max
    if a.gates_parquet:
        global GATES_PARQUET
        GATES_PARQUET = a.gates_parquet
        print(f"[{a.tag}] gates parquet = {a.gates_parquet}", flush=True)
    win = {"search_start": W.FACTORS_FIRST_SIGNAL_DATE, "search_end": label_max,
           "label_available_max": W.LABEL_MAX_DATE, "evidence": "E1"}
    print(f"[{a.tag}] window {win['search_start']}~{win['search_end']} "
          f"(labels≤{win['label_available_max']})", flush=True)
    cands = json.load(open(a.cands, encoding="utf-8"))
    if a.gate_bank:
        bank = json.load(open(a.gate_bank, encoding="utf-8"))
        bmap = {e["gate_id"]: e for e in bank.get("entries", [])
                if e.get("level") == "market"}
        n0 = len(cands)
        cands = [c for c in cands if c.get("gate") in bmap]
        n_over = 0
        # V4.3a：specialist 面板时只做白名单过滤，不覆盖偏科因子个体锁定的 s_pair
        #（其 s 由自身有向斜率符号锁定；组级 bank s 会配反一半）。
        _keep_s = (a.base_panel == "specialist")
        for c in cands:
            if _keep_s:
                continue
            if int(c.get("s", 1)) != int(bmap[c["gate"]]["s"]):
                n_over += 1
            c["s"] = int(bmap[c["gate"]]["s"])
        print(f"[{a.tag}] gate-bank {a.gate_bank}: {n0}->{len(cands)} cands, "
              f"s overridden {n_over} (bank {bank.get('version')}"
              f"{', keep-s-pair' if _keep_s else ''})", flush=True)
    print(f"[{a.tag}] candidates={len(cands)}", flush=True)
    if not cands:
        # 空桶早退：某交互桶（如 flip）选出 0 候选时写空表正常退出，
        # 不进面板/置换流程（防 base_cols[0] IndexError 中断整条 SOP）。
        pd.DataFrame(columns=["name", "base", "gate", "gate_type", "s", "d",
                              "form", "delta_mean", "placebo_p", "fail_reason",
                              "pair_fp", "cached"]).to_csv(
            a.out, mode="w", header=True, index=False, encoding="utf-8-sig")
        print(f"[{a.tag}] 空候选：写出空表 -> {a.out}", flush=True)
        return 0
    # 本批只跑 market 门控（掩码捷径，已证明等价）。
    # sector 门控需全重算（逐置换重排），独立第二批实现。
    sc = [c for c in cands if c["gate_type"] != "market"]
    assert not sc, f"本批只接受 market 候选，sector 候选 {len(sc)} 个请走第二批"
    mk = cands
    # labels
    lab = B.read_labels(["r3", "y_prod", "tradable"], label_max=label_max, label=f"f1-{a.tag}")
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab[(lab["tradable"] == 1) & (lab["date"] <= label_max)].copy()
    # gates
    mg = load_gates()
    # base 面板（去重列）：默认base20历史链路；vwap_top20/specialist读动态面板
    # switch复合键（off__def）不是面板列，只取真实列
    base_cols = sorted({c["base"] for c in cands if not (c.get("base_off") and c.get("base_def"))}
                       | {c["base_off"] for c in cands if c.get("base_off")}
                       | {c["base_def"] for c in cands if c.get("base_def")})
    if a.base_panel in ("vwap_top20", "specialist"):
        _vp = os.path.join(RESULTS, "vwap_top20_factors.parquet"
                           if a.base_panel == "vwap_top20" else "specialist_factors.parquet")
        _vj = os.path.join(RESULTS, "vwap_top20.json"
                           if a.base_panel == "vwap_top20" else "specialist_pool.json")
        if not os.path.exists(_vp):
            raise SystemExit(f"--base-panel {a.base_panel} 需要 {_vp}")
        fac = pd.read_parquet(_vp)
        fac["date"] = pd.to_datetime(fac["date"])
        _jm = json.load(open(_vj, encoding="utf-8"))
        _items = _jm.get("base20", []) if a.base_panel == "vwap_top20" \
            else _jm.get("factors", [])
        _key = "factor" if a.base_panel == "vwap_top20" else "factor_id"
        b20 = {b[_key]: int(b["direction"]) for b in _items}
        missing = [c for c in base_cols if c not in fac.columns]
        if missing:
            raise SystemExit(f"{a.base_panel}面板缺列 {len(missing)} 个: {missing[:5]}")
        m = fac.merge(lab[["stock_code", "date", "r3", "y_prod"]], on=["stock_code", "date"], how="inner")
        m = m[m["date"] <= label_max].copy()
        del fac
    else:
        fac = B.read_factors(base_cols, data_max=W.LABEL_MAX_DATE, label=f"f1-{a.tag}")
        fac["date"] = pd.to_datetime(fac["date"])
        m = fac.merge(lab[["stock_code", "date", "r3", "y_prod"]], on=["stock_code", "date"], how="inner")
        m = m[m["date"] <= label_max].copy()
        del fac
        b20 = {b["factor"]: b["direction"] for b in
               json.load(open(os.path.join(RESULTS, "base20.json"), encoding="utf-8"))["base20"]}
    # V4.3a：m 必须按 date 稳定排序（fac stock-major；_prepared 只排内部副本；
    # gain/flip/switch 用 m 行序的 xv/row_dates 配 date-major 的 day/yrk——同源错位）。
    # 旧 g/t 经 _prepared/ic_cache 与日期键映射天然免疫。
    m = m.sort_values("date", kind="stable").reset_index(drop=True)
    print(f"[{a.tag}] panel rows={len(m)} base_panel={a.base_panel}", flush=True)
    # 预计算每 base 的 IC_z（d·x）日序列
    day, dates_u, _, _, nd = E._prepared(m, base_cols[0], "r3")
    _ud = np.unique(pd.to_datetime(m["date"]).to_numpy())
    assert (pd.to_datetime(m["date"]).to_numpy() == _ud[day]).all(), "m行序与日码错位"
    uniq = np.unique(m["date"].to_numpy())
    ic_cache = {}
    for bcol in base_cols:
        d = b20.get(bcol, 1)
        _, _, xr, yr, _ = E._prepared(m.assign(_z=pd.to_numeric(m[bcol], errors="coerce") * d), "_z", "r3")
        ic_cache[bcol] = E._daily_corr(xr, yr, day, nd)
    print(f"[{a.tag}] base IC cached", flush=True)
    # 三式共享预计算（提速）：day/yrk/u_cols/Ubar 全批一次，逐候选复用。
    _prep = build_market_bundle(m, base_cols, b20) \
        if a.form in ("gain", "flip", "switch") else None
    if _prep is not None:
        print(f"[{a.tag}] market bundle ready ({len(base_cols)} bases)", flush=True)

    started_empty = not os.path.exists(a.out)
    done_n = 0
    import f1_cache as FC
    _cache = {} if a.no_f1_cache else FC.load_cache()
    _dver = B.factor_data_version()
    print(f"[{a.tag}] f1_pair_cache: {len(_cache)} hit-able "
          f"({'off' if a.no_f1_cache else 'on'})", flush=True)
    _recs, _n_hit, _n_new = [], 0, 0
    for i, c in enumerate(mk):
        t1 = time.time()
        _base = (f"{c.get('base_off')}__{c.get('base_def')}"
                 if a.form == "switch" and c.get("base_off") else c["base"])
        rec = {"name": c["name"], "base": _base, "gate": c["gate"],
               "gate_type": c["gate_type"], "s": c["s"], "d": c["d"],
               "form": a.form,
               **({"tilt_delta": ""} if a.form == "switch" else {}),
               "search_start": win["search_start"], "search_end": win["search_end"],
               "label_available_max": win["label_available_max"],
               "evidence": win["evidence"]}
        _fp = FC.pair_fp(_base, c["gate"], c["s"], c["d"], a.form,
                         label_max, _dver)
        rec["pair_fp"] = _fp
        _hit = None if a.no_f1_cache else FC.cache_hit(_fp, _cache)
        if _hit is not None:
            rec.update({k: _hit.get(k) for k in FC.STAT_KEYS if _hit.get(k) is not None})
            rec["fail_reason"] = ""
            rec["cached"] = True
            _recs.append(rec)
            _n_hit += 1
            print(f"[{a.tag} {i + 1}/{len(mk)}] {c['name']} CACHE-HIT "
                  f"d={rec.get('delta_mean')} p={rec.get('placebo_p')}", flush=True)
            continue
        rec["cached"] = False
        try:
            _gcol = c["gate"] + "_" + (a.form if a.form in ("g", "t") else "g")
            gmap = dict(zip(pd.to_datetime(mg["date"]).dt.date.astype(str),
                            mg[_gcol].to_numpy()))
            g = np.array([gmap.get(str(pd.Timestamp(dd).date()), np.nan) for dd in uniq])
            if a.form in ("gain", "flip", "switch"):
                if c["gate_type"] != "market":
                    raise SystemExit(f"f1_batch {a.form}分支只接受market候选"
                                     f"（sector请走f1_sector --form {a.form}）")
                if a.form == "gain":
                    r = run_one_market_gain(m, base_cols, b20, c["base"], g, c["s"],
                                            uniq, prep=_prep)
                elif a.form == "flip":
                    r = run_one_market_flip(m, base_cols, b20, c["base"], g, c["s"],
                                            uniq, prep=_prep)
                else:
                    if not c.get("base_off") or not c.get("base_def"):
                        raise SystemExit("switch候选须带 base_off/base_def（见select_specialists输出）")
                    r = run_one_market_switch(m, base_cols, b20, c["base_off"],
                                              c["base_def"], g, uniq, prep=_prep)
                    rec["base"] = f"{c['base_off']}__{c['base_def']}"
            else:
                r = run_one_market(m, c["base"], c["d"], g, c["s"], uniq, ic_cache[c["base"]])
            r.pop("_f1_ic", None)
            dl = r["delta"]
            pl = r["placebo"]
            sl = r["slope"]
            rec.update({
                "delta_mean": dl.get("mean"), "delta_ci_lo": dl.get("ci_lo"),
                "delta_ci_hi": dl.get("ci_hi"), "nblocks_pos": dl.get("nblocks_pos"),
                "blocks": json.dumps(dl.get("blocks"), ensure_ascii=False),
                "slope_b": sl.get("b"), "slope_t": sl.get("t"),
                "placebo_p": pl.get("p"), "placebo_obs": pl.get("obs"),
                "placebo_q95": pl.get("null_q95"), "placebo_pct": pl.get("pct"),
                "n_days_f1": dl.get("n_days_f1"), "n_days_base": dl.get("n_days_base"),
                "fail_reason": "",
            })
            if a.form == "switch" and "tilt_delta" in r:
                # switch辅助列：底仓倾斜诊断（与主判定同文件不同列，不进FDR）
                rec["tilt_delta"] = r.get("tilt_delta")
            if not a.no_f1_cache:
                FC.save_result(_fp, {k: rec.get(k) for k in FC.STAT_KEYS
                                     + ("fail_reason",)})
                _cache[_fp] = {k: rec.get(k) for k in FC.STAT_KEYS + ("fail_reason",)}
            _n_new += 1
        except Exception as e:
            import traceback
            rec.update({"fail_reason": str(e)[:300]})
            print(f"[{a.tag}] {c['name']} ERROR {str(e)[:120]}", flush=True)
            if not a.no_f1_cache:
                FC.save_result(_fp, {"fail_reason": rec["fail_reason"]})
        _recs.append(rec)
        done_n += 1
        print(f"[{a.tag} {done_n}/{len(mk)}] {c['name']} "
              f"d={rec.get('delta_mean')} p={rec.get('placebo_p')} {time.time()-t1:.0f}s", flush=True)
    # 全量去重覆盖写（按 name keep-last）：多次运行幂等，消灭追加重复行。
    _out_df = pd.DataFrame(_recs)
    if len(_out_df):
        _out_df = _out_df.drop_duplicates(subset=["name"], keep="last")
    _out_df.to_csv(a.out, mode="w", header=True, index=False, encoding="utf-8-sig")
    print(f"[{a.tag}] DONE hit={_n_hit} new={_n_new} rows={len(_out_df)} -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
