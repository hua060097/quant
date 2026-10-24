# -*- coding: utf-8 -*-
"""F1 sector 批：行业门控全重算（逐置换重排）。
用法: python f1_sector.py --cands cand_s.json --out f1_s.csv --tag s0
候选: [{name, base, gate(gate_id), s, d}]
门控面板：搜索空间长表（date/sector/gate_id/raw/p）——
  旧手工门控宽表（gates/sector_gates.parquet，S1_p/S2_p/S3_p）已于 2026-10-12
  随手工门控定义一并删除；`--form` 映射（对齐 f1_batch）：
  `g` = 2p−1（连续）；`t` = `gates.thresh_form(p)`（THRESH：p≥0.7 on / ≤0.3 off / 中间 NaN）。
placebo: 对 sector-gate 面板做日期块置换（全行业同置换，保留截面结构），
  每次重算 F1 分数与日 IC。200 次。
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
import fastranks as FR
import gates as G
import h3_eval as H
import windows as W

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
# 搜索窗口改为单一事实源（windows.py）；原为硬编码 "2024-12-26"（P0/P1 旧窗口）。
LABEL_MAX = W.FACTOR_SCREEN_MAX
N_PERM = 200
PLACEBO_BLOCK = 20
SEED = 0


#: 并行工作进程共享（fork 继承，免 pickling 大面板）
_F1S_CTX = None


def _par_jobs(n) -> int:
    """并行度归一化：非 posix 平台退化串行（spawn 下大面板 pickling 太贵）."""
    import os
    try:
        nj = max(1, int(n or 1))
    except Exception:
        return 1
    if nj > 1 and os.name != "posix":
        print("  [warn] 非 posix 平台，--jobs 退化为串行", flush=True)
        return 1
    return nj


def _run_f1_sector_one(c: dict, idx: int, rec: dict, ctx: dict) -> dict:
    """单个候选完整计算（串行/并行共用；不碰缓存文件，由父进程统一落缓存）。
    返回填好的 rec（含 fail_reason；异常路径与串行一致）。"""
    m, sg, uniq = ctx["m"], ctx["sg"], ctx["uniq"]
    day, nd, n_dates = ctx["day"], ctx["nd"], ctx["n_dates"]
    sec_ids, sec_uniq, yrk = ctx["sec_ids"], ctx["sec_uniq"], ctx["yrk"]
    date_idx, Ubar = ctx["date_idx"], ctx["Ubar"]
    form, tag = ctx["form"], ctx["tag"]
    try:
        # 该门控的矩阵（搜索长表：p → 按 form 映射，与 f1_batch 的列语义对齐）
        gp = sg[sg["gate_id"] == c["gate"]].pivot_table(
            index="date", columns="sector", values="p", aggfunc="mean")
        if form == "t":
            arr = G.thresh_form(gp.to_numpy(dtype=float).ravel())
            gp = pd.DataFrame(arr.reshape(gp.shape),
                              index=gp.index, columns=gp.columns)
        else:
            gp = 2.0 * gp - 1.0
        gp = gp.reindex(pd.to_datetime(uniq))
        Gm = gp.to_numpy(dtype=float)  # [n_dates, n_sectors_all]
        # switch用原始分位矩阵P（与G同行列对齐，不随form映射改变）
        Praw = sg[sg["gate_id"] == c["gate"]].pivot_table(
            index="date", columns="sector", values="p", aggfunc="mean")
        Praw = Praw.reindex(pd.to_datetime(uniq))
        Pmat = Praw.reindex(columns=gp.columns).to_numpy(dtype=float)
        # sector 列对齐到 sec_uniq
        gp_cols = list(gp.columns)
        gp_set = set(gp_cols)
        col_idx = np.array([gp_cols.index(s) if s in gp_set else -1 for s in sec_uniq])
        sec_row = sec_ids  # 每行的 sector 位置
        _is_switch = (form == "switch")
        if _is_switch:
            # switch复合base不是面板列；掩码改用off/def有效性（u在下文对齐）
            z = np.full(len(m), np.nan)
            for _cc in (c.get("base_off"), c.get("base_def")):
                _xv = pd.to_numeric(m[_cc], errors="coerce").to_numpy(dtype=float)
                z = np.where(np.isfinite(_xv), 1.0, z)
        else:
            xv = pd.to_numeric(m[c["base"]], errors="coerce").to_numpy(dtype=float)
            z = c["d"] * xv
        rv = pd.to_numeric(m["r3"], errors="coerce").to_numpy(dtype=float)
        _is_gain = (form == "gain")
        _is_flip = (form == "flip")
        _is_switch = (form == "switch")
        if _is_switch and (not c.get("base_off") or not c.get("base_def")):
            raise SystemExit("switch候选须带 base_off/base_def（见select_specialists输出）")
        _inorm = ctx.get("interact_norm", "mad_zscore_soft")
        if _is_gain or _is_flip:
            if _inorm == "mad_zscore_soft":
                u_row = FR.mad_directed(xv, day, int(c["d"]), soft_cap=5.0)
            else:
                from f1_batch import _centered_u as _cu
                u_row = _cu(xv, day, nd, int(c["d"]))
        elif _is_switch:
            # switch legs方向：cands自带d_off/d_def（缺省1）；统一调公共定义。
            u_off_row, u_def_row = None, None
            for _cc, _dd, _slot in ((c["base_off"], int(c.get("d_off", 1)), 0),
                                    (c["base_def"], int(c.get("d_def", 1)), 1)):
                _xv = pd.to_numeric(m[_cc], errors="coerce").to_numpy(dtype=float)
                if _inorm == "mad_zscore_soft":
                    _u = FR.mad_directed(_xv, day, _dd, soft_cap=5.0)
                else:
                    _u = FR.centered_u(_xv, day, nd, _dd)
                if _slot == 0:
                    u_off_row = _u
                else:
                    u_def_row = _u
            u_row = None
        else:
            u_row = None

        def f1_scores(Gm_, Pm=None):
            okrow = (sec_row >= 0) & np.isfinite(z)
            ci = col_idx[sec_row[okrow]]
            valid_col = ci >= 0
            vv = np.full(okrow.sum(), np.nan)
            rr, cc = date_idx[okrow][valid_col], ci[valid_col]
            vv[valid_col] = Gm_[rr, cc]
            f1 = np.full(len(m), np.nan)
            if _is_gain:
                ww = H.build_gain_weight(vv, c["s"])
                f1[okrow] = u_row[okrow] * ww
            elif _is_flip:
                ww = H.build_flip_weight(vv, c["s"])
                f1[okrow] = u_row[okrow] * ww
            elif _is_switch:
                Puse = Pm if Pm is not None else Pmat
                pv = np.full(okrow.sum(), np.nan)
                pv[valid_col] = Puse[rr, cc]
                pv = np.clip(pv, 0.0, 1.0)
                f1[okrow] = H.build_switch_blend(
                    pv, u_off_row[okrow], u_def_row[okrow])
            else:
                ww = np.maximum(c["s"] * vv, 0.0)
                f1[okrow] = z[okrow] * ww
            return f1

        def _tilt_ic(F1k: np.ndarray) -> np.ndarray:
            # 底仓倾斜（与市场 gain 同构）：S=Ubar+0.25·F vs Ubar+0.25·u。
            _fr = FR.blockwise_rank(
                np.where(np.isfinite(Ubar + 0.25 * F1k),
                         Ubar + 0.25 * F1k, np.nan), day)
            return E._daily_corr(_fr, yrk, day, nd)

        def f1_ic_of(Gm_, Pm=None):
            if _is_gain:
                return _tilt_ic(f1_scores(Gm_, Pm))
            fr = FR.blockwise_rank(f1_scores(Gm_, Pm), day)   # 2.4× vs _within_rank
            return E._daily_corr(fr, yrk, day, nd)

        f1_ic = f1_ic_of(Gm)
        if _is_gain:
            _s_static = Ubar + 0.25 * u_row
            br = FR.blockwise_rank(
                np.where(np.isfinite(_s_static), _s_static, np.nan), day)
        elif _is_flip:
            br = FR.blockwise_rank(np.where(np.isfinite(u_row), u_row, np.nan), day)
        elif _is_switch:
            _static = H.static_consensus(u_off_row, u_def_row)
            br = FR.blockwise_rank(np.where(np.isfinite(_static), _static, np.nan), day)
        else:
            br = FR.blockwise_rank(np.where(np.isfinite(z), z, np.nan), day)
        base_ic = E._daily_corr(br, yrk, day, nd)
        if _is_switch:
            rec["base"] = f"{c['base_off']}__{c['base_def']}"
        dl = H.delta_ic(np.asarray(uniq), f1_ic, base_ic)
        # slope 用日均门控（诊断）
        gmean = np.nanmean(Gm, axis=1)
        sl = H.slope_test(base_ic, gmean)
        # placebo：日期块置换（全行业同置换）
        rng = np.random.default_rng(SEED)
        nb = max(1, n_dates // PLACEBO_BLOCK)
        pieces = np.array_split(np.arange(n_dates), nb)
        obs = float(np.nanmean(f1_ic) - np.nanmean(base_ic)) \
            if np.isfinite(f1_ic).any() and np.isfinite(base_ic).any() else float("nan")

        def _d2(f1s):
            f1s = np.asarray(f1s, dtype=float)
            ff = f1s[np.isfinite(f1s)]
            bb = base_ic[np.isfinite(base_ic)]
            if len(ff) == 0 or len(bb) == 0:
                return float("nan")
            return float(ff.mean() - bb.mean())

        nulls = np.empty(N_PERM)
        for pi in range(N_PERM):
            order = rng.permutation(nb)
            pidx = np.concatenate([pieces[j] for j in order])[:n_dates]
            if _is_switch:
                nulls[pi] = _d2(f1_ic_of(Gm[pidx], Pmat[pidx]))
            else:
                nulls[pi] = _d2(f1_ic_of(Gm[pidx]))
        nulls = nulls[np.isfinite(nulls)]
        # 净增益诊断：相对随机置换门控的增量（扣除零分布基线；判定仍用 H3 原口径）。
        mu_null = float(np.mean(nulls)) if len(nulls) else float("nan")
        if len(nulls) == 0 or not np.isfinite(obs):
            p = float("nan")
        else:
            k = int((np.abs(nulls) >= abs(obs)).sum())
            p = float((1 + k) / (len(nulls) + 1))
        # spread / level-leak（诊断）：F1 分数的日内百分位排名 → spread
        f1s = f1_scores(Gm)
        fr_all = FR.blockwise_rank(f1s, day)
        dcnt = np.bincount(day, minlength=nd).astype(float)
        rkp = fr_all / np.maximum(dcnt[day], 1)
        sp_f1 = H.daily_spread(np.arange(nd), rkp, rv, day, nd, q=0.2)
        ga = np.nanmean(np.abs(Gm), axis=1)
        mm = np.isfinite(sp_f1) & np.isfinite(ga)
        if mm.sum() < 30:
            scorr = float("nan")
        else:
            ra = pd.Series(sp_f1[mm]).rank().to_numpy()
            rb = pd.Series(ga[mm]).rank().to_numpy()
            ra -= ra.mean(); rb -= rb.mean()
            den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
            scorr = float((ra * rb).sum() / den) if den > 0 else float("nan")
        rec.update({
            "delta_mean": dl.get("mean"), "delta_ci_lo": dl.get("ci_lo"),
            "delta_ci_hi": dl.get("ci_hi"), "nblocks_pos": dl.get("nblocks_pos"),
            "blocks": json.dumps(dl.get("blocks"), ensure_ascii=False),
            "slope_b": sl.get("b"), "slope_t": sl.get("t"),
            "placebo_p": p, "placebo_obs": obs,
            "placebo_q95": float(np.quantile(np.abs(nulls), 0.95)) if len(nulls) else float("nan"),
            "delta_net": (float(dl.get("mean") - mu_null)
                          if np.isfinite(dl.get("mean", float("nan")))
                          and np.isfinite(mu_null) else float("nan")),
            "spread_abs_g_corr": scorr,
            "leak_risk": bool(abs(scorr) > 0.8) if np.isfinite(scorr) else False,
            "n_days_f1": dl.get("n_days_f1"), "n_days_base": dl.get("n_days_base"),
            "fail_reason": "",
        })
    except Exception as e:
        rec.update({"fail_reason": str(e)[:300]})
        print(f"[{tag}] {c['name']} ERROR {str(e)[:150]}", flush=True)
    return rec


def _f1_sector_worker(args) -> tuple:
    """工作进程：(序号, 候选, rec骨架) → (序号, rec)。共享经 fork 继承。"""
    idx, c, rec = args
    return idx, _run_f1_sector_one(c, idx, rec, _F1S_CTX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="f1s")
    ap.add_argument("--label-max", default=LABEL_MAX,
                    help=f"信号日上界（默认选择窗 {W.FACTOR_SCREEN_MAX}；"
                         f"复现 P0/P1 旧结果用 {W.LEGACY_SEARCH_END}）")
    ap.add_argument("--form", default="g", choices=("g", "t", "gain", "flip", "switch"),
                    help="门控组合形式：g=连续(2p-1)；t=THRESH(p>=0.7 为 on)；"
                         "gain=中心化增益F1v3；flip=方向翻转F1v3；"
                         "switch=双因子软切换F1v3（需cands带base_off/base_def）。"
                         "新三式与g/t平行对照，不得合并FDR。见 PLAN §21")
    ap.add_argument("--gates-parquet", default=os.path.join(
        RESULTS, "gates_search", "gate_search_sector.parquet"),
        help="行业门控面板（搜索长表：date/sector/gate_id/raw/p）；"
             "旧手工门控宽表已于 2026-10-12 删除")
    ap.add_argument("--no-f1-cache", dest="no_f1_cache", action="store_true",
                    default=False,
                    help="关闭配对指纹缓存，全部重算（默认启用缓存断点续算）")
    ap.add_argument("--jobs", type=int, default=1,
                    help="未命中缓存候选的并行进程数（12 核建议 8；"
                         "结果与串行逐值一致，非 posix 平台自动退化串行）")
    ap.add_argument("--base-panel", choices=("base20", "vwap_top20", "specialist"),
                    default="base20",
                    help="基因子面板（默认base20保证历史可复现；"
                         "vwap_top20读build_top动态正交面板；"
                         "specialist读select_specialists偏科池面板）")
    ap.add_argument("--gate-bank", default=None,
                    help="已认证门控字典certified_gate_bank.json（只读消费）："
                         "cands门控过滤到bank清单且s用bank锁定值覆盖；"
                         "不传则保持历史行为")
    ap.add_argument("--sample-stocks", type=int, default=0,
                    help=">0 时按固定种子抽样 N 只股票（**仅粗筛用**，"
                         "不得用于正式判定；见 gate_search_report §4.1）")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--interact-norm", choices=("mad_zscore_soft", "centered_u"),
                    default="mad_zscore_soft",
                    help="底座因子标准化模式（默认 mad_zscore_soft 保留右尾极值峰度；centered_u 为旧均匀秩）")
    a = ap.parse_args()
    label_max = a.label_max
    win = {"search_start": W.FACTORS_FIRST_SIGNAL_DATE, "search_end": label_max,
           "label_available_max": W.LABEL_MAX_DATE, "evidence": "E1"}
    print(f"[{a.tag}] window {win['search_start']}~{win['search_end']} "
          f"(labels≤{win['label_available_max']})", flush=True)
    cands = json.load(open(a.cands, encoding="utf-8"))
    if a.gate_bank:
        bank = json.load(open(a.gate_bank, encoding="utf-8"))
        bmap = {e["gate_id"]: e for e in bank.get("entries", [])
                if e.get("level") == "sector"}
        n0 = len(cands)
        cands = [c for c in cands if c.get("gate") in bmap]
        n_over = 0
        # V4.3a：specialist 面板时只做白名单过滤，不覆盖个体锁定的 s_pair。
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
        # 空桶早退（与 f1_batch 同口径）：写空表正常退出，防 IndexError 中断 SOP。
        pd.DataFrame(columns=["name", "base", "gate", "gate_type", "s", "d",
                              "form", "delta_mean", "placebo_p", "fail_reason",
                              "pair_fp", "cached"]).to_csv(
            a.out, mode="w", header=True, index=False, encoding="utf-8-sig")
        print(f"[{a.tag}] 空候选：写出空表 -> {a.out}", flush=True)
        return 0
    # labels
    lab = B.read_labels(["r3", "y_prod", "tradable"], label_max=label_max, label=f"f1s-{a.tag}")
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab[(lab["tradable"] == 1) & (lab["date"] <= label_max)].copy()
    # gates（搜索长表）
    import gates as G
    sg = pd.read_parquet(a.gates_parquet)
    if "gate_id" not in sg.columns:
        raise SystemExit(
            f"{a.gates_parquet} 不是搜索空间行业面板（缺 gate_id 列）；"
            "旧手工门控宽表已于 2026-10-12 删除，门控定义由搜索生成")
    sg["gate_id"] = sg["gate_id"].astype(str)
    sg["date"] = pd.to_datetime(sg["date"])
    gate_ids = {str(c["gate"]) for c in cands}
    missing = sorted(gate_ids - set(sg["gate_id"].unique()))
    if missing:
        raise SystemExit(f"门控面板缺少 {len(missing)} 个候选门控: {missing[:8]}")
    print(f"[{a.tag}] gates panel {sg.shape} from {a.gates_parquet}", flush=True)
    # sector map（ix_build 已归档；gates.load_sector_map 同文件同值已验证）
    import gates as G
    secmap = G.load_sector_map()
    # base 面板：默认base20历史链路；vwap_top20/specialist读动态面板
    # switch复合键（off__def）不是面板列，只取真实列
    base_cols = sorted({c["base"] for c in cands if not (c.get("base_off") and c.get("base_def"))}
                       | {c["base_off"] for c in cands if c.get("base_off")}
                       | {c["base_def"] for c in cands if c.get("base_def")})
    if a.base_panel in ("vwap_top20", "specialist"):
        _vp = os.path.join(RESULTS, "vwap_top20_factors.parquet"
                           if a.base_panel == "vwap_top20" else "specialist_factors.parquet")
        if not os.path.exists(_vp):
            raise SystemExit(f"--base-panel {a.base_panel} 需要 {_vp}")
        fac = pd.read_parquet(_vp)
        fac["date"] = pd.to_datetime(fac["date"])
        missing = [c for c in base_cols if c not in fac.columns]
        if missing:
            raise SystemExit(f"{a.base_panel}面板缺列 {len(missing)} 个: {missing[:5]}")
        m = fac.merge(lab[["stock_code", "date", "r3"]], on=["stock_code", "date"], how="inner")
        m = m[m["date"] <= label_max].copy()
        del fac
    else:
        fac = B.read_factors(base_cols, data_max=W.LABEL_MAX_DATE, label=f"f1s-{a.tag}")
        fac["date"] = pd.to_datetime(fac["date"])
        m = fac.merge(lab[["stock_code", "date", "r3"]], on=["stock_code", "date"], how="inner")
        m = m[m["date"] <= label_max].copy()
        del fac
    m["_sec"] = m["stock_code"].astype(str).map(secmap)
    # V4.3a：m 必须按 date 稳定排序（fac 是 stock-major 行序；_prepared 只排
    # 内部副本，day/yrk 是 date-major；否则 _sec/xv 与 day/yrk 全错位——
    # 与 gate_g2g3_sector 第④点同源 bug）。
    m = m.sort_values("date", kind="stable").reset_index(drop=True)
    # 抽样截面（两级评估的粗筛级；**仅用于粗筛，不得用于正式判定**——
    # 见 gate_search_report §4.1 与 Codex 对抽样精度的要求）
    if getattr(a, "sample_stocks", 0):
        rng = np.random.default_rng(getattr(a, "sample_seed", 0))
        codes = np.array(sorted(m["stock_code"].astype(str).unique()))
        pick = set(codes[rng.choice(len(codes), size=min(a.sample_stocks, len(codes)),
                                    replace=False)].tolist())
        m = m[m["stock_code"].astype(str).isin(pick)].copy()
    print(f"[{a.tag}] panel rows={len(m)} unmapped={int(m['_sec'].isna().mean()*len(m))}", flush=True)
    # 日码与 sector 码
    day, dates_u, _, _, nd = E._prepared(m, base_cols[0], "r3")
    uniq = np.unique(m["date"].to_numpy())
    assert (pd.to_datetime(m["date"]).to_numpy() == uniq[day]).all(), "m行序与日码错位"
    n_dates = len(uniq)
    sec_ids, sec_uniq = pd.factorize(m["_sec"].astype(str))
    n_sec = len(sec_uniq)
    # R3 日内排名（全候选共享）
    _, _, _, yrk, _ = E._prepared(m, base_cols[0], "r3")
    # gain 底仓倾斜底座（V4.3x 修复）：固定参考底仓（vwap_top20 roster 在
    # specialist 面板中的有向居中秩均值），不再按 batch base_cols 自求平均——
    # 同质偏科桶自求平均会把交互信号吸进底仓（方差主导致边际 tilt 归零），
    # 且使 pair 缓存带批次依赖。roster 缺失/可用列不足时回退 batch 口径+告警。
    from f1_batch import build_reference_ubar
    _Ubar, _ubar_info = build_reference_ubar(m, day, nd, interact_norm=a.interact_norm)
    if _Ubar is None:
        print(f"  [warn] 参考底仓不可用（{_ubar_info}），回退 batch Ubar"
              f"（批次相关！）", flush=True)
        _dmap: dict[str, int] = {}
        for _cc in cands:
            for _bb in ([_cc.get("base")] if not _cc.get("base_off")
                        else [_cc.get("base_off"), _cc.get("base_def")]):
                if _bb and _bb not in _dmap:
                    _k = "d_off" if _bb == _cc.get("base_off") else (
                        "d_def" if _bb == _cc.get("base_def") else "d")
                    try:
                        _dmap[_bb] = int(_cc.get(_k, 1))
                    except Exception:
                        _dmap[_bb] = 1
        _u_all: dict[str, np.ndarray] = {}
        for _bc in base_cols:
            _xv = pd.to_numeric(m[_bc], errors="coerce").to_numpy(dtype=float)
            if getattr(a, "interact_norm", "mad_zscore_soft") == "mad_zscore_soft":
                _u_all[_bc] = FR.mad_directed(_xv, day, int(_dmap.get(_bc, 1)), soft_cap=5.0)
            else:
                _u_all[_bc] = FR.centered_u(_xv, day, nd, int(_dmap.get(_bc, 1)))
        _Ubar = np.nanmean(np.column_stack(list(_u_all.values())), axis=1) \
            if _u_all else np.full(len(m), np.nan)
    # gate 面板矩阵 G[date_idx, sec_idx]，按 uniq 日期顺序
    date_idx = day  # _prepared 按 date 排序，day 即日期码（0..nd-1），与 uniq 顺序一致
    assert nd == n_dates
    import f1_cache as FC
    _cache = {} if a.no_f1_cache else FC.load_cache()
    _dver = B.factor_data_version()
    _ubar_fp = (_ubar_info or {}).get("hash", "")
    print(f"[{a.tag}] f1_pair_cache: {len(_cache)} hit-able "
          f"({'off' if a.no_f1_cache else 'on'}) ubar={_ubar_fp}", flush=True)
    _recs, _n_hit, _n_new = [], 0, 0
    # 缓存命中判定（纯字典运算）串行；未命中才进计算（串行/并行）
    _miss = []
    for i, c in enumerate(cands):
        _base = (f"{c.get('base_off')}__{c.get('base_def')}"
                 if a.form == "switch" and c.get("base_off") else c["base"])
        rec = {"name": c["name"], "base": _base, "gate": c["gate"],
               "gate_type": "sector", "s": c["s"], "d": c["d"], "form": a.form,
               "search_start": win["search_start"], "search_end": win["search_end"],
               "label_available_max": win["label_available_max"],
               "evidence": win["evidence"]}
        _fp = FC.pair_fp(_base, c["gate"], c["s"], c["d"], a.form,
                         label_max, _dver, _ubar_fp)
        rec["pair_fp"] = _fp
        _hit = None if a.no_f1_cache else FC.cache_hit(_fp, _cache)
        if _hit is not None:
            rec.update({k: _hit.get(k) for k in FC.STAT_KEYS + ("spread_abs_g_corr",
                                                               "leak_risk")
                        if _hit.get(k) is not None})
            rec["fail_reason"] = ""
            rec["cached"] = True
            _recs.append((i, rec))
            _n_hit += 1
            print(f"[{a.tag} {i + 1}/{len(cands)}] {c['name']} CACHE-HIT "
                  f"d={rec.get('delta_mean')} p={rec.get('placebo_p')}", flush=True)
            continue
        rec["cached"] = False
        _miss.append((i, c, rec, _fp))
    _n_total = len(cands)

    def _post(_idx, _rec, _fp, _t1):
        if not a.no_f1_cache:
            _save = {k: _rec.get(k) for k in
                     FC.STAT_KEYS + ("fail_reason", "spread_abs_g_corr", "leak_risk")}
            FC.save_result(_fp, _save)
            _cache[_fp] = _save
        _recs.append((_idx, _rec))
        print(f"[{a.tag} {_idx + 1}/{_n_total}] {_rec.get('name')} "
              f"d={_rec.get('delta_mean')} p={_rec.get('placebo_p')} {time.time()-_t1:.0f}s",
              flush=True)
        return 1

    _nj = _par_jobs(a.jobs)
    if _nj <= 1 or len(_miss) <= 1:
        for i, c, rec, _fp in _miss:
            t1 = time.time()
            _ctx1 = {"m": m, "sg": sg, "uniq": uniq, "day": day, "nd": nd,
                     "n_dates": n_dates, "sec_ids": sec_ids, "sec_uniq": sec_uniq,
                     "yrk": yrk, "date_idx": date_idx, "Ubar": _Ubar,
                     "form": a.form, "tag": a.tag, "interact_norm": a.interact_norm}
            _run_f1_sector_one(c, i, rec, _ctx1)
            _n_new += _post(i, rec, _fp, t1)
    else:
        import concurrent.futures as _cf
        global _F1S_CTX
        _F1S_CTX = {"m": m, "sg": sg, "uniq": uniq, "day": day, "nd": nd,
                    "n_dates": n_dates, "sec_ids": sec_ids, "sec_uniq": sec_uniq,
                    "yrk": yrk, "date_idx": date_idx, "Ubar": _Ubar,
                    "form": a.form, "tag": a.tag, "interact_norm": a.interact_norm}
        _nj = min(_nj, len(_miss))
        print(f"[{a.tag}] 并行计算：{len(_miss)} 候选 × {_nj} 进程 …", flush=True)
        _t1_all = {i: time.time() for i, _, _, _ in _miss}
        with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
            _order = {i: k for k, (i, _, _, _) in enumerate(_miss)}
            for idx, _rec in _ex.map(_f1_sector_worker,
                                     [(i, c, rec) for i, c, rec, _ in _miss]):
                _k = _order[idx]
                _, _, _, _fp = _miss[_k]
                _n_new += _post(idx, _rec, _fp, _t1_all[idx])
    _recs.sort(key=lambda x: x[0])
    _recs = [r for _, r in _recs]
    # 全量去重覆盖写（按 name keep-last）：多次运行幂等，消灭追加重复行。
    _out_df = pd.DataFrame(_recs)
    if len(_out_df):
        _out_df = _out_df.drop_duplicates(subset=["name"], keep="last")
    _out_df.to_csv(a.out, mode="w", header=True, index=False, encoding="utf-8-sig")
    print(f"[{a.tag}] DONE hit={_n_hit} new={_n_new} rows={len(_out_df)} -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
