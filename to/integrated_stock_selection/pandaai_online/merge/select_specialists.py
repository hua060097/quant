# -*- coding: utf-8 -*-
"""状态偏科因子选择器（V4.2 日常层）：无条件平庸但对认证门控条件斜率显著的因子。

输入：
  - build_top 榜单 flat（ALIGNED_TOP2_*_flat.csv）：vwap LGB 排序取 rank #lo~#hi
  - certified_gate_bank.json：市场门控清单 + G2 锁定方向 s
过程（market门控，精确计算）：
  对每个候选算逐日 RankIC（vwap标签，选择窗），对每个bank门控做
  IC_t ~ g_t 回归（h3_eval.slope_test，NW标准误），取 |t| 最大者；
  无条件 |rank_ic|<0.05 且 |t|>=阈值 → specialist；
  高低门控分组IC异号 → 建议 flip，否则 gain。
输出：
  - specialist_pool.json（因子方向/门控/t值/建议形式 + switch对）
  - specialist_factors.parquet（stock_code/date/因子值列，供F1 --base-panel specialist）
  - specialist_cands_gain/flip/switch.json（分form cands文件，各自跑对应--form）

V1范围：只做market门控精确t_slope；sector仍走既有sector F1路径
（vwap_top20/base20面板），sector偏科t_slope列为TODO（见refactor_doc §六）。
switch对为V1启发式：offense取trend_position轴LGB最高，defense取
volatility/distribution/liquidity_size轴LGB最高，各至多3个 × bank前2门控，
上限--max-switch对（默认12）。

用法：
    python select_specialists.py --rank-lo 21 --rank-hi 150 --t-min 2.5 --max-factors 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import windows as W  # noqa: E402
import h3_eval as H  # noqa: E402
from evaluator import Evaluator, load_panel  # noqa: E402
from fitness import DataContext  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")
OUT_DIR = HERE

OFFENSE_AXES = ("trend_position",)
DEFENSE_AXES = ("volatility", "distribution", "liquidity_size")


def _dsign(v) -> int:
    """flat direction 0/1 → F1/落盘 ±1（0=反向；refs_cond_t1.py:98 同口径）。

    V4.3a：flat 沿用 fitness/executor 的 0/1 编码（1=正向，0=反向）；
    直接透传 0 会使 _centered_u(..., d=0) 全零。vwap_top20.json 侧
    build_top.py:821 已做同映射，此处补齐 cands/pool 侧。
    """
    try:
        iv = int(v)
    except Exception:
        return 1
    return -1 if iv in (0, -1) else 1


def _latest_flat() -> str:
    cands = sorted(Path(OUT_DIR).glob("ALIGNED_TOP2_*_flat.csv"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("out/下无 ALIGNED_TOP2_*_flat.csv（先跑 build_top.py）")
    return str(cands[-1])


def _daily_rankic(values: np.ndarray, labels: np.ndarray,
                  dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐日截面Spearman IC（vwap标签，周期采样对齐后的行）。

    向量化实现（提速约 100 倍）：日码一次 factorize，blockwise_rank 一次排名，
    bincount 一次相关——替代逐日全表扫描 + 5 万次 scipy.spearmanr。
    语义与旧实现逐值一致（含 30 样本下限；tied-rank 用 average 法，
    与 spearmanr 默认一致）。
    """
    import fastranks as FR
    v = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    order = np.argsort(np.asarray(dates), kind="stable")
    vs, ys = v[order], y[order]
    dcodes, uniq = pd.factorize(np.asarray(dates)[order], sort=True)
    dcodes = np.asarray(dcodes)
    nd = int(dcodes.max()) + 1 if len(dcodes) else 0
    # pairwise 子集内排名（与旧 spearmanr(v[ok], y[ok]) 逐值一致；各自排名会错位）。
    m = np.isfinite(vs) & np.isfinite(ys)
    fr = FR.blockwise_rank(np.where(m, vs, np.nan), dcodes)
    yr = FR.blockwise_rank(np.where(m, ys, np.nan), dcodes)
    ok = np.isfinite(fr) & np.isfinite(yr)
    d = dcodes[ok]
    x, yy = fr[ok], yr[ok]
    n = np.bincount(d, minlength=nd).astype(float)
    sx = np.bincount(d, weights=x, minlength=nd)
    sy = np.bincount(d, weights=yy, minlength=nd)
    sxx = np.bincount(d, weights=x * x, minlength=nd)
    syy = np.bincount(d, weights=yy * yy, minlength=nd)
    sxy = np.bincount(d, weights=x * yy, minlength=nd)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / np.where(n > 0, n, 1)
        vx = sxx - sx * sx / np.where(n > 0, n, 1)
        vy = syy - sy * sy / np.where(n > 0, n, 1)
        r = cov / np.sqrt(vx * vy)
    r[n < 30] = np.nan
    r[~np.isfinite(r)] = np.nan
    return np.asarray(uniq), r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flat", default="")
    ap.add_argument("--bank", default=os.path.join(SPACE, "certified_gate_bank.json"))
    ap.add_argument("--label", default="vwap")
    ap.add_argument("--rank-lo", type=int, default=1,
                    help="V4.3d双模：起点放宽至#1，允许主力强因子参与gain降回撤")
    ap.add_argument("--rank-hi", type=int, default=400,
                    help="断路3修复：flip 型正负抵消因子 LGB≈0 排在 #200 之后，"
                         "150 会系统漏杀；默认扫到 #400（flat 全覆盖）")
    ap.add_argument("--t-min", type=float, default=2.5)
    ap.add_argument("--uncond-max", type=float, default=0.05)
    ap.add_argument("--max-factors", type=int, default=150)
    ap.add_argument("--max-switch", type=int, default=12)
    ap.add_argument("--data-cut", default=W.DEFAULT_DATA_CUT)
    ap.add_argument("--resume", default="",
                    help="已算出的specialist_pool.json，跳过其中已有factor_id")
    ap.add_argument("--no-export-panel", action="store_true",
                    help="只输出pool/cands JSON，不写因子值面板")
    a = ap.parse_args()

    t0 = time.time()
    flat = a.flat or _latest_flat()
    df = pd.read_csv(flat, encoding="utf-8-sig")
    vdf = df[df["label"] == a.label].sort_values("LGB", ascending=False).reset_index(drop=True)
    vdf["_rank"] = np.arange(1, len(vdf) + 1)
    window = vdf[(vdf["_rank"] >= a.rank_lo) & (vdf["_rank"] <= a.rank_hi)].copy()
    print(f"flat {os.path.basename(flat)} vwap={len(vdf)} window[#{a.rank_lo}~#{a.rank_hi}]={len(window)}",
          flush=True)

    bank = json.load(open(a.bank, encoding="utf-8"))
    gates = [e for e in bank.get("entries", []) if e.get("level") == "market"]
    if not gates:
        raise SystemExit(f"{a.bank} 无market门控")
    print(f"bank {bank.get('version')} market gates={len(gates)}", flush=True)
    gpanel = pd.read_parquet(os.path.join(SPACE, "gate_search_market.parquet"),
                             columns=["date", "gate_id", "p"])
    gpanel["date"] = pd.to_datetime(gpanel["date"])

    done_ids: set[str] = set()
    if a.resume and os.path.exists(a.resume):
        try:
            _old = json.load(open(a.resume, encoding="utf-8"))
            done_ids = {f["factor_id"] for f in _old.get("factors", [])}
            print(f"resume: skip {len(done_ids)} computed", flush=True)
        except Exception:
            pass

    frame = load_panel()
    ev = Evaluator(frame)
    from refs import FactorBuilder
    builder = FactorBuilder(frame)
    from decorrelate_top import factor_full_values, load_all_specs
    specs_by_id = load_all_specs()
    ctx = DataContext(cycle=3, start=W.FACTORS_FIRST_SIGNAL_DATE, end=a.data_cut,
                      frame=frame, label=a.label, evaluator=ev)
    pos = ctx._pos
    row_dates = pd.to_datetime(ctx.row_dates)
    labels = ctx.labels.astype(float)
    print(f"ctx rows={ctx.n_rows:,} days={len(np.unique(ctx.row_dates))}", flush=True)

    pool, panel_vals = [], {}
    todo = [r for _, r in window.iterrows() if str(r["factor_id"]) not in done_ids]
    todo = todo[:max(0, a.max_factors)]
    # store 存量列一次性批量映射（单次读表；四源 helper 统一消费）。
    store_map: dict = {}
    try:
        _need_store = sorted({str(r["factor_id"]) for r in todo if str(r["lib"]) == "store"})
        if _need_store:
            import boundary as _B
            _sf = _B.read_factors(_need_store, data_max=a.data_cut, label="specialist-store")
            _m = _sf.merge(frame[["stock_code", "date"]],
                           on=["stock_code", "date"], how="right")
            for _c in _need_store:
                _vv = _m[_c].to_numpy(dtype=float) if _c in _m.columns else None
                if _vv is not None and len(_vv) == len(frame):
                    store_map[_c] = _vv
            print(f"store 批量映射 {len(store_map)}/{len(_need_store)} 列", flush=True)
    except Exception as _e:
        print(f"warning: store 批量映射失败 {str(_e)[:100]}", flush=True)
    for i, r in enumerate(todo, 1):
        fid, lib = str(r["factor_id"]), str(r["lib"])
        try:
            full = factor_full_values(lib, fid, str(r.get("formula_cn", "")), frame,
                                      ev, builder, specs_by_id, store_map)
            if full is None:
                continue
        except Exception:
            continue
        vals = np.asarray(full, dtype=float)[pos]
        panel_vals[fid] = np.asarray(full, dtype=float)
        ud, ic = _daily_rankic(vals, labels, ctx.row_dates)
        udates = pd.to_datetime(ud)
        best = None
        for g in gates:
            sub = gpanel[gpanel["gate_id"] == g["gate_id"]].set_index("date")["p"]
            gg = (2.0 * sub - 1.0).reindex(udates).to_numpy(dtype=float)
            m = np.isfinite(ic) & np.isfinite(gg)
            if m.sum() < 60:
                continue
            st = H.slope_test(ic[m], gg[m])
            t = st.get("t")
            if t is None or not np.isfinite(t):
                continue
            if best is None or abs(float(t)) > abs(best["t"]):
                hi = gg[m] >= np.nanmedian(gg[m])
                lo = ~hi
                _hi_ic = ic[m][hi]
                _lo_ic = ic[m][lo]
                best = {"gate": g["gate_id"], "s": int(g["s"]), "t": float(t),
                        "b": float(st.get("b", float("nan"))),
                        "hi_ic": float(np.nanmean(_hi_ic)),
                        "lo_ic": float(np.nanmean(_lo_ic)),
                        "hi_win": float(np.nanmean(_hi_ic > 0)) if len(_hi_ic) else float("nan"),
                        "lo_win": float(np.nanmean(_lo_ic > 0)) if len(_lo_ic) else float("nan")}
        if best is None:
            continue
        uncond = r.get("rank_ic")
        try:
            uncond = float(uncond) if uncond is not None and np.isfinite(float(uncond)) else 0.0
        except Exception:
            uncond = 0.0
        suggested = "flip" if (np.isfinite(best["hi_ic"]) and np.isfinite(best["lo_ic"])
                               and best["hi_ic"] * best["lo_ic"] < 0) else "gain"
        # V4.3a：个体 s_pair——按自身有向斜率符号锁定，而非照抄 bank 组级 s。
        # best["b"] 为原始 IC 对中心化门控的斜率；有向斜率 = d_sign*b；
        # ≥0 → 高门控放大有效 → s=+1，否则 s=-1（配反则 F1 ΔIC 必负被拒）。
        _ds = _dsign(r.get("direction"))
        _b = best["b"]
        _s_pair = int(best["s"]) if not np.isfinite(_b) \
            else (1 if (_ds * _b) >= 0 else -1)
        # V4.3d 双模硬门槛（禁纯弱噪声混入）：
        # A类【强但有回撤的主力】rank<=60 & |uncond|>=0.030 &（高低差>=0.025 或 icir<0.50）
        # B类【真半场王者】|t|>=t_min & max(|hi|,|lo|)>=0.035 & 半场胜率>=0.58
        try:
            _icir = float(r.get("ic_ir"))
        except Exception:
            _icir = float("nan")
        _au = abs(uncond)
        _hdiff = abs(best["hi_ic"] - best["lo_ic"]) \
            if np.isfinite(best["hi_ic"]) and np.isfinite(best["lo_ic"]) else float("nan")
        _hmax = max(abs(best["hi_ic"]), abs(best["lo_ic"])) \
            if np.isfinite(best["hi_ic"]) and np.isfinite(best["lo_ic"]) else float("nan")
        _wmax = max(best["hi_win"], best["lo_win"]) \
            if np.isfinite(best["hi_win"]) and np.isfinite(best["lo_win"]) else float("nan")
        _cls_a = (int(r["_rank"]) <= 60 and _au >= 0.030
                  and (np.isfinite(_hdiff) and _hdiff >= 0.025
                       or (np.isfinite(_icir) and _icir < 0.50)))
        _cls_b = (abs(best["t"]) >= a.t_min
                  and np.isfinite(_hmax) and _hmax >= 0.035
                  and np.isfinite(_wmax) and _wmax >= 0.58)
        _cls = "A" if _cls_a else ("B" if _cls_b else "")
        if _cls == "A":
            suggested = "gain"
        rec = {"factor_id": fid, "lib": lib, "family": str(r.get("family", "")),
               "formula": str(r.get("formula_cn", "")),
               "direction": _ds,
               "rank": int(r["_rank"]), "LGB": float(r.get("LGB", 0) or 0),
               "uncond_rank_ic": uncond, "gate": best["gate"], "s": _s_pair,
               "s_bank": int(best["s"]),
               "t_slope": best["t"], "slope_b": best["b"],
               "hi_ic": best["hi_ic"], "lo_ic": best["lo_ic"],
               "hi_win": best["hi_win"], "lo_win": best["lo_win"],
               "suggested_form": suggested, "target_class": _cls,
               "selected": bool(_cls != "")}
        pool.append(rec)
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {fid} t={best['t']:+.2f}@{best['gate']} "
                  f"sel={rec['selected']} {time.time()-t0:.0f}s", flush=True)

    sel = [p for p in pool if p["selected"]]
    _na = sum(1 for p in sel if p.get("target_class") == "A")
    _nb = sum(1 for p in sel if p.get("target_class") == "B")
    print(f"specialists: {len(sel)}/{len(pool)} (A主力={_na}, B半场={_nb})",
          flush=True)

    # switch对：offense/defense代表 × bank前2门控（V1启发式，上限max-switch）
    from grammar import AXIS_OF
    by_axis: dict[str, list] = {}
    for _, r in vdf.iterrows():
        ax = AXIS_OF.get(str(r.get("family", "")), "other")
        by_axis.setdefault(ax, []).append(r)
    off_cands, def_cands = [], []
    for ax in OFFENSE_AXES:
        off_cands += list(by_axis.get(ax, []))[:3]
    for ax in DEFENSE_AXES:
        def_cands += list(by_axis.get(ax, []))[:3]
    switch_pairs = []
    for g in gates[:2]:
        for ro in off_cands:
            for rd in def_cands:
                if str(ro["factor_id"]) == str(rd["factor_id"]):
                    continue
                switch_pairs.append({"off": str(ro["factor_id"]), "def": str(rd["factor_id"]),
                                     "gate": g["gate_id"], "s": int(g["s"])})
                if len(switch_pairs) >= a.max_switch:
                    break
            if len(switch_pairs) >= a.max_switch:
                break
        if len(switch_pairs) >= a.max_switch:
            break
    print(f"switch pairs: {len(switch_pairs)}", flush=True)

    # 面板导出：specialists + switch legs 全量值
    need_ids = {p["factor_id"] for p in sel}
    for sp in switch_pairs:
        need_ids.add(sp["off"])
        need_ids.add(sp["def"])
    if not a.no_export_panel:
        cols = {"stock_code": frame["stock_code"].to_numpy(),
                "date": pd.to_datetime(frame["date"]).to_numpy()}
        # 复用已算值；缺的现算
        id2row = {}
        for _, r in vdf.iterrows():
            id2row[str(r["factor_id"])] = r
        for fid in sorted(need_ids):
            if fid in panel_vals:
                cols[fid] = np.asarray(panel_vals[fid], dtype=float)
                continue
            r = id2row.get(fid)
            if r is None:
                continue
            try:
                full = factor_full_values(str(r["lib"]), fid, str(r.get("formula_cn", "")),
                                          frame, ev, builder, specs_by_id, store_map)
                if full is not None:
                    cols[fid] = np.asarray(full, dtype=float)
            except Exception:
                continue
        pdf = pd.DataFrame(cols)
        pdf.to_parquet(os.path.join(RESULTS, "specialist_factors.parquet"), index=False)
        print(f"specialist_factors.parquet: {pdf.shape}", flush=True)

    # pool清单与分form cands
    dirmap = {}
    for _, r in vdf.iterrows():
        dirmap[str(r["factor_id"])] = _dsign(r.get("direction"))
    out_pool = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                "bank": a.bank, "flat": flat, "label": a.label,
                "rank_window": [a.rank_lo, a.rank_hi], "t_min": a.t_min,
                "uncond_max": a.uncond_max, "data_cut": a.data_cut,
                "n_scanned": len(pool), "n_selected": len(sel),
                "factors": pool, "switch_pairs": switch_pairs,
                "sector_note": "market + sector cands both generated "
                               "(specialist_sector_cands_*.json for f1_sector)"}
    with open(os.path.join(RESULTS, "specialist_pool.json"), "w", encoding="utf-8") as f:
        json.dump(out_pool, f, ensure_ascii=False, indent=1)

    def _cand(p, form):
        return {"name": f"{p['factor_id']}__{p['gate']}__{form}",
                "base": p["factor_id"], "gate": p["gate"], "gate_type": "market",
                "s": int(p["s"]), "d": int(dirmap.get(p["factor_id"], 1)),
                "form": form}

    for form in ("gain", "flip"):
        sub = [p for p in sel if p["suggested_form"] == form]
        with open(os.path.join(RESULTS, f"specialist_cands_{form}.json"), "w",
                   encoding="utf-8") as f:
            json.dump([_cand(p, form) for p in sub], f, ensure_ascii=False, indent=1)
        print(f"specialist_cands_{form}.json: {len(sub)}", flush=True)
    sw = []
    for sp in switch_pairs:
        sw.append({"name": f"{sp['off']}__{sp['def']}__{sp['gate']}__switch",
                   "base": f"{sp['off']}__{sp['def']}",
                   "base_off": sp["off"], "base_def": sp["def"],
                   "gate": sp["gate"], "gate_type": "market", "s": int(sp["s"]),
                   "d": 1, "d_off": int(dirmap.get(sp["off"], 1)),
                   "d_def": int(dirmap.get(sp["def"], 1)), "form": "switch"})
    with open(os.path.join(RESULTS, "specialist_cands_switch.json"), "w",
               encoding="utf-8") as f:
        json.dump(sw, f, ensure_ascii=False, indent=1)
    print(f"specialist_cands_switch.json: {len(sw)}", flush=True)
    # F1v3族清单直连（members键与f1输出rec一致，可直接judge）
    import f1_family as FF
    for form, path in (("gain", "specialist_cands_gain.json"),
                       ("flip", "specialist_cands_flip.json"),
                       ("switch", "specialist_cands_switch.json")):
        cp = os.path.join(RESULTS, path)
        man = FF.build_from_cands(cp, family="F1v3", form=form)
        mp = os.path.join(SPACE, f"family_manifest_F1v3_{form}.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        print(f"family F1v3_{form}: n={man['n_expected']} hash={man['members_hash']} -> {mp}",
              flush=True)
    # 断路3修复：行业门控偏科候选（bank sector门控 × 偏科因子/switch对）。
    # s 取 bank 行业锁定符号（组级预注册）；d 取因子自身方向。供 f1_sector 开箱消费。
    sec_gates = [e for e in bank.get("entries", []) if e.get("level") == "sector"]
    if sec_gates:
        def _scand(p, form, sg):
            return {"name": f"{p['factor_id']}__{sg['gate_id']}__{form}",
                    "base": p["factor_id"], "gate": sg["gate_id"],
                    "gate_type": "sector", "s": int(sg["s"]),
                    "d": int(dirmap.get(p["factor_id"], 1)), "form": form}
        for form in ("gain", "flip"):
            sub = [p for p in sel if p["suggested_form"] == form]
            sec_cands = [_scand(p, form, sg) for p in sub for sg in sec_gates]
            with open(os.path.join(RESULTS, f"specialist_sector_cands_{form}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(sec_cands, f, ensure_ascii=False, indent=1)
            print(f"specialist_sector_cands_{form}.json: {len(sec_cands)}", flush=True)
        ssw = []
        for sg in sec_gates:
            for sp in switch_pairs:
                ssw.append({"name": f"{sp['off']}__{sp['def']}__{sg['gate_id']}__switch",
                            "base": f"{sp['off']}__{sp['def']}",
                            "base_off": sp["off"], "base_def": sp["def"],
                            "gate": sg["gate_id"], "gate_type": "sector",
                            "s": int(sg["s"]), "d": 1,
                            "d_off": int(dirmap.get(sp["off"], 1)),
                            "d_def": int(dirmap.get(sp["def"], 1)), "form": "switch"})
        with open(os.path.join(RESULTS, "specialist_sector_cands_switch.json"),
                  "w", encoding="utf-8") as f:
            json.dump(ssw, f, ensure_ascii=False, indent=1)
        print(f"specialist_sector_cands_switch.json: {len(ssw)}", flush=True)
        for form, path in (("gain", "specialist_sector_cands_gain.json"),
                           ("flip", "specialist_sector_cands_flip.json"),
                           ("switch", "specialist_sector_cands_switch.json")):
            cp = os.path.join(RESULTS, path)
            man = FF.build_from_cands(cp, family="F1v3_sector", form=form)
            mp = os.path.join(SPACE, f"family_manifest_F1v3_sector_{form}.json")
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(man, f, ensure_ascii=False, indent=1)
            print(f"family F1v3_sector_{form}: n={man['n_expected']} "
                  f"hash={man['members_hash']} -> {mp}", flush=True)
    else:
        print("bank 无sector门控：跳过 sector cands 生成", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
