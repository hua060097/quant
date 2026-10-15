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


def _latest_flat() -> str:
    cands = sorted(Path(OUT_DIR).glob("ALIGNED_TOP2_*_flat.csv"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("out/下无 ALIGNED_TOP2_*_flat.csv（先跑 build_top.py）")
    return str(cands[-1])


def _daily_rankic(values: np.ndarray, labels: np.ndarray,
                  dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐日截面Spearman IC（vwap标签，周期采样对齐后的行）。"""
    from scipy.stats import spearmanr
    ud = np.unique(dates)
    ic = np.full(len(ud), np.nan)
    for i, dd in enumerate(ud):
        m = (dates == dd)
        v, y = values[m], labels[m]
        ok = np.isfinite(v) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        try:
            r = spearmanr(v[ok], y[ok])
            r = r[0] if isinstance(r, tuple) else getattr(r, "statistic", float("nan"))
        except Exception:
            r = float("nan")
        ic[i] = float(r) if np.isfinite(r) else np.nan
    return ud, ic


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flat", default="")
    ap.add_argument("--bank", default=os.path.join(SPACE, "certified_gate_bank.json"))
    ap.add_argument("--label", default="vwap")
    ap.add_argument("--rank-lo", type=int, default=21)
    ap.add_argument("--rank-hi", type=int, default=150)
    ap.add_argument("--t-min", type=float, default=2.5)
    ap.add_argument("--uncond-max", type=float, default=0.05)
    ap.add_argument("--max-factors", type=int, default=60)
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
    from decorrelate_top import load_all_specs
    specs_by_id = load_all_specs()
    from custom_registry import evaluate_custom_factor as _ecf
    from custom_registry import load_all_custom_specs as _lcs
    ctx = DataContext(cycle=3, start=W.FACTORS_FIRST_SIGNAL_DATE, end=a.data_cut,
                      frame=frame, label=a.label, evaluator=ev)
    pos = ctx._pos
    row_dates = pd.to_datetime(ctx.row_dates)
    labels = ctx.labels.astype(float)
    print(f"ctx rows={ctx.n_rows:,} days={len(np.unique(ctx.row_dates))}", flush=True)

    from parser import parse_formula
    pool, panel_vals = [], {}
    todo = [r for _, r in window.iterrows() if str(r["factor_id"]) not in done_ids]
    todo = todo[:max(0, a.max_factors)]
    for i, r in enumerate(todo, 1):
        fid, lib = str(r["factor_id"]), str(r["lib"])
        try:
            if lib == "mined":
                node = parse_formula(str(r["formula_cn"]))
                full = np.asarray(ev.values(node), dtype=float)
            elif lib == "custom":
                spec = _lcs().get(fid)
                if spec is None:
                    continue
                full = np.asarray(_ecf(spec, frame, ev), dtype=float)
            else:
                spec = specs_by_id.get(fid)
                if spec is None:
                    continue
                full = builder.get_factor(spec).to_numpy(dtype=float)
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
                best = {"gate": g["gate_id"], "s": int(g["s"]), "t": float(t),
                        "b": float(st.get("b", float("nan"))),
                        "hi_ic": float(np.nanmean(ic[m][hi])),
                        "lo_ic": float(np.nanmean(ic[m][lo]))}
        if best is None:
            continue
        uncond = r.get("rank_ic")
        try:
            uncond = float(uncond) if uncond is not None and np.isfinite(float(uncond)) else 0.0
        except Exception:
            uncond = 0.0
        suggested = "flip" if (np.isfinite(best["hi_ic"]) and np.isfinite(best["lo_ic"])
                               and best["hi_ic"] * best["lo_ic"] < 0) else "gain"
        rec = {"factor_id": fid, "lib": lib, "family": str(r.get("family", "")),
               "formula": str(r.get("formula_cn", "")),
               "direction": int(r["direction"]) if pd.notna(r.get("direction")) else 1,
               "rank": int(r["_rank"]), "LGB": float(r.get("LGB", 0) or 0),
               "uncond_rank_ic": uncond, "gate": best["gate"], "s": best["s"],
               "t_slope": best["t"], "slope_b": best["b"],
               "hi_ic": best["hi_ic"], "lo_ic": best["lo_ic"],
               "suggested_form": suggested,
               "selected": bool(abs(best["t"]) >= a.t_min and abs(uncond) < a.uncond_max)}
        pool.append(rec)
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {fid} t={best['t']:+.2f}@{best['gate']} "
                  f"sel={rec['selected']} {time.time()-t0:.0f}s", flush=True)

    sel = [p for p in pool if p["selected"]]
    print(f"specialists: {len(sel)}/{len(pool)} (|t|>={a.t_min}, |uncond|<{a.uncond_max})",
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
                lib = str(r["lib"])
                if lib == "mined":
                    full = np.asarray(ev.values(parse_formula(str(r["formula_cn"]))), dtype=float)
                elif lib == "custom":
                    spec = _lcs().get(fid)
                    full = np.asarray(_ecf(spec, frame, ev), dtype=float) if spec else None
                else:
                    spec = specs_by_id.get(fid)
                    full = builder.get_factor(spec).to_numpy(dtype=float) if spec else None
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
        try:
            dirmap[str(r["factor_id"])] = int(r["direction"]) if pd.notna(r.get("direction")) else 1
        except Exception:
            dirmap[str(r["factor_id"])] = 1
    out_pool = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                "bank": a.bank, "flat": flat, "label": a.label,
                "rank_window": [a.rank_lo, a.rank_hi], "t_min": a.t_min,
                "uncond_max": a.uncond_max, "data_cut": a.data_cut,
                "n_scanned": len(pool), "n_selected": len(sel),
                "factors": pool, "switch_pairs": switch_pairs,
                "sector_note": "market-gate specialists only in V1; "
                               "sector path unchanged (vwap_top20/base20)"}
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
    print(f"done {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
