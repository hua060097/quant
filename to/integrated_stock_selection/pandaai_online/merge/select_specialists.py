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

#: roles.csv group → switch配对轴（store 因子无模板 family，按存量分组映射；
#: context 系分组不属于进攻/防御，不映射，保守归 other）。
_STORE_GROUP_AXIS = {
    "momentum_and_position": "trend_position",
    "directional_alpha": "trend_position",
    "volatility_and_liquidity": "volatility",
    "risk_and_exhaustion": "volatility",
    "tail_protect": "volatility",
    "capital_flow_risk": "liquidity_size",
}
#: proposal 公式算子启发式（run_search 提案 family 固定为 'proposal'，
#: 按公式根节点/算子特征映射；命中多类时防御优先，无命中归 other）。
_PROPOSAL_DEF_OPS = ("STDDEV", "VAR", "SKEW", "AMIHUD")
_PROPOSAL_OFF_OPS = ("SLOPE", "RET1", "HIGH")


def _resolve_axis(r, roles_group_map: dict | None = None) -> str:
    """switch 配对用机制轴归属（纯查表 + 字符串启发式，无外部调用）。

    优先级：AXIS_OF（M 模板 family）> `ref:<axis>` 去前缀（本身即轴名）
    > `custom_<axis>` 去前缀 > store 按 roles.csv group
    > proposal 按公式算子启发式 > other。
    此前 `AXIS_OF.get(family, "other")` 把 store / ref:* / proposal /
    custom_* 全判成 other，使它们永远进不了 OFFENSE/DEFENSE 候选。
    """
    from grammar import AXIS_OF
    fam = str(r.get("family", "") or "")
    if fam in AXIS_OF:
        return AXIS_OF[fam]
    if fam.startswith("ref:"):
        # flat.csv 的 refs 行经 build_top.norm_refs 覆写为 'ref:'+lib，
        # lib 取值见 refs.py LIB_REPOS/--libs（directional/risk-pattern/
        # volume-stat/dir-adapt），与 refs.py 注册时的 ref:<axis> 轴名是两套
        # 命名——兼容字典优先，轴名原值直通。
        _r_ax = fam[4:]
        return {"directional": "trend_position", "dir-adapt": "trend_position",
                "risk-pattern": "volatility",
                "volume-stat": "liquidity_size"}.get(_r_ax, _r_ax or "other")
    if fam.startswith("custom_"):
        return fam[7:] or "other"
    if fam == "store":
        return _STORE_GROUP_AXIS.get(
            (roles_group_map or {}).get(str(r.get("factor_id", ""))), "other")
    if fam == "proposal":
        f = str(r.get("formula", "") or r.get("formula_cn", "") or "").upper()
        if any(k in f for k in _PROPOSAL_DEF_OPS):
            return "volatility"
        if any(k in f for k in _PROPOSAL_OFF_OPS):
            return "trend_position"
        return "other"
    return "other"


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


#: 并行工作进程共享（fork 继承，免 pickling 大面板/store 映射）
_SP_CTX = None


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


def _sector_tsec(vals_signed: np.ndarray, labels: np.ndarray,
                 row_dates: np.ndarray, sec: dict, gid: str) -> float:
    """单因子对单行业门控的横截面 halves 响应 t（sector 预筛用）。

    g>=0 行业半场 vs g<0 行业半场 的逐日 RankIC 配对差 → nw_t。
    绝不用日均门控 gmean 做斜率：横截面门控（S×T，靠同日内行业间差异）
    求跨行业日均会把信号抹平归零，误杀真赢家。
    vals_signed 须已乘因子方向（signed basis）；返回 signed t（排序/过滤用 |t|）。
    sec 缺失/门控缺失/公共日不足 → NaN（调用方跳过）。
    """
    try:
        import pandas as _pd
        from eval_v2 import nw_t
        _G = (sec.get("panels") or {}).get(gid)
        if _G is None:
            return float("nan")
        _day = np.asarray((sec.get("row_day") or {}).get(gid, []), dtype=int)
        _col = np.asarray((sec.get("row_col") or {}).get(gid, []), dtype=int)
        _n = len(vals_signed)
        if len(_day) != _n or len(_col) != _n:
            return float("nan")
        _ok = (_day >= 0) & (_col >= 0)
        if _ok.sum() < 60:
            return float("nan")
        _g = np.full(_n, np.nan)
        _g[_ok] = np.asarray(_G["G"], dtype=float)[_day[_ok], _col[_ok]]
        _v = np.asarray(vals_signed, dtype=float)
        _y = np.asarray(labels, dtype=float)
        _d = np.asarray(row_dates)
        _fin = np.isfinite(_g) & np.isfinite(_v) & np.isfinite(_y)
        _hi_m = _fin & (_g >= 0)
        _lo_m = _fin & (_g < 0)
        if _hi_m.sum() < 30 or _lo_m.sum() < 30:
            return float("nan")
        _ud_hi, _ic_hi = _daily_rankic(_v[_hi_m], _y[_hi_m], _d[_hi_m])
        _ud_lo, _ic_lo = _daily_rankic(_v[_lo_m], _y[_lo_m], _d[_lo_m])
        _s_hi = _pd.Series(_ic_hi, index=_pd.to_datetime(_ud_hi))
        _s_lo = _pd.Series(_ic_lo, index=_pd.to_datetime(_ud_lo))
        _common = _s_hi.index.intersection(_s_lo.index)
        if len(_common) < 10:
            return float("nan")
        _diff = (_s_hi.reindex(_common) - _s_lo.reindex(_common)).to_numpy(
            dtype=float)
        _diff = _diff[np.isfinite(_diff)]
        if len(_diff) < 10:
            return float("nan")
        _t = nw_t(_diff)
        return float(_t) if _t is not None and np.isfinite(_t) else float("nan")
    except Exception:
        return float("nan")


def _run_specialist_one(r, ctx: dict):
    """单个因子偏科评估（串行/并行共用）：返回 pool 记录（未选中也返回，含 selected 标记；
    完全失败返回 None）。"""
    import pandas as _pd
    frame, ev = ctx["frame"], ctx["ev"]
    builder, specs = ctx["builder"], ctx["specs"]
    store_map, gpanel, gates = ctx["store_map"], ctx["gpanel"], ctx["gates"]
    pos, labels, row_dates = ctx["pos"], ctx["labels"], ctx["row_dates"]
    t_min = ctx["t_min"]
    from decorrelate_top import factor_full_values
    fid, lib = str(r["factor_id"]), str(r["lib"])
    try:
        full = factor_full_values(lib, fid, str(r.get("formula_cn", "")), frame,
                                  ev, builder, specs, store_map)
        if full is None:
            return None
    except Exception:
        return None
    vals = np.asarray(full, dtype=float)[pos]
    ud, ic = _daily_rankic(vals, labels, row_dates)
    udates = _pd.to_datetime(ud)
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
            # 半场胜率 = 与该半场自身 IC 均值同号的天数占比（防 flip 负半场
            # 误杀：flip 高低半场异号，任一固定符号阈值必杀一半；_ds=±1 均自洽）。
            # hi_ic/lo_ic 保持原始 measured 口径展示（F1 不消费，仅审计）。
            _hi_mean = float(np.nanmean(_hi_ic)) if len(_hi_ic) else float("nan")
            _lo_mean = float(np.nanmean(_lo_ic)) if len(_lo_ic) else float("nan")
            _hi_win = (float(np.nanmean(np.sign(_hi_ic) == np.sign(_hi_mean)))
                       if len(_hi_ic) and np.isfinite(_hi_mean) else float("nan"))
            _lo_win = (float(np.nanmean(np.sign(_lo_ic) == np.sign(_lo_mean)))
                       if len(_lo_ic) and np.isfinite(_lo_mean) else float("nan"))
            best = {"gate": g["gate_id"], "s": int(g["s"]), "t": float(t),
                    "b": float(st.get("b", float("nan"))),
                    "hi_ic": float(np.nanmean(_hi_ic)),
                    "lo_ic": float(np.nanmean(_lo_ic)),
                    "hi_win": _hi_win, "lo_win": _lo_win}
    if best is None:
        return None
    uncond = r.get("rank_ic")
    try:
        uncond = float(uncond) if uncond is not None and np.isfinite(float(uncond)) else 0.0
    except Exception:
        uncond = 0.0
    suggested = "flip" if (np.isfinite(best["hi_ic"]) and np.isfinite(best["lo_ic"])
                           and best["hi_ic"] * best["lo_ic"] < 0) else "gain"
    # V4.3a：个体 s_pair——按自身有向斜率符号锁定，而非照抄 bank 组级 s。
    _ds = _dsign(r.get("direction"))
    _b = best["b"]
    _s_pair = int(best["s"]) if not np.isfinite(_b) \
        else (1 if (_ds * _b) >= 0 else -1)
    # V4.3d 双模硬门槛（禁纯弱噪声混入）：
    # A类【强但有回撤的主力】rank<=60 & |uncond|>=0.030 & |t|>=2.0 & 高低差>=0.025
    #   （旧 or _icir<0.50 分支在 flat 未年化 ic_ir 口径下 100% 恒真，已删除）
    # B类【真半场王者】|t|>=t_min & 主半场（|ic|较大者）|ic|>=0.035 且同向胜率>=0.58
    _au = abs(uncond)
    _hdiff = abs(best["hi_ic"] - best["lo_ic"]) \
        if np.isfinite(best["hi_ic"]) and np.isfinite(best["lo_ic"]) else float("nan")
    _ahi = abs(best["hi_ic"]) if np.isfinite(best["hi_ic"]) else -1.0
    _alo = abs(best["lo_ic"]) if np.isfinite(best["lo_ic"]) else -1.0
    _cls_a = (int(r["_rank"]) <= 60 and _au >= 0.030
              and np.isfinite(best["t"]) and abs(best["t"]) >= 2.0
              and np.isfinite(_hdiff) and _hdiff >= 0.025)
    _cls_b = (abs(best["t"]) >= t_min
              and ((_ahi >= _alo and _ahi >= 0.035
                    and np.isfinite(best["hi_win"]) and best["hi_win"] >= 0.58)
                   or (_alo > _ahi and _alo >= 0.035
                       and np.isfinite(best["lo_win"]) and best["lo_win"] >= 0.58)))
    _cls = "A" if _cls_a else ("B" if _cls_b else "")
    # A 类主力统一走 gain（降回撤）；高低异号的 flip 即使达 A 也不覆盖。
    if _cls == "A" and best["hi_ic"] * best["lo_ic"] >= 0:
        suggested = "gain"
    # 行业 halves 响应 t_sec（Bug4 预筛用；signed basis，与 traded 方向一致）。
    _t_sec: dict = {}
    try:
        _secb = ctx.get("sec")
        if _secb is not None:
            _sv = _ds * np.asarray(vals, dtype=float)
            for _sg in _secb.get("gates", []):
                _gid = str(_sg.get("gate_id", ""))
                if not _gid:
                    continue
                _t_sec[_gid] = _sector_tsec(_sv, labels, row_dates, _secb, _gid)
    except Exception:
        pass
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
           "t_sec": {k: (float(v) if v is not None else float("nan"))
                     for k, v in _t_sec.items()},
           "selected": bool(_cls != "")}
    return rec


def _specialist_worker(args) -> tuple:
    """工作进程：(序号, 行) → (序号, pool记录或None)。共享经 fork 继承。"""
    idx, r = args
    try:
        rec = _run_specialist_one(r, _SP_CTX)
    except Exception:
        return idx, None
    return idx, rec


def _market_cand_score(p) -> float:
    """市场桶截断评分 |t_slope|*|hi-lo|（方向无关；缺失按 0 计，优先截掉）。"""
    try:
        _t = p.get("t_slope")
        _t = abs(float(_t)) if _t is not None and np.isfinite(float(_t)) else 0.0
    except Exception:
        _t = 0.0
    try:
        _hi, _lo = p.get("hi_ic"), p.get("lo_ic")
        _hd = abs(float(_hi) - float(_lo)) \
            if _hi is not None and _lo is not None \
            and np.isfinite(float(_hi)) and np.isfinite(float(_lo)) else 0.0
    except Exception:
        _hd = 0.0
    return _t * _hd


def _quota_select(sub: list, gates: list[str], total_cap: int,
                  per_min: int = 2, per_max: int = 5) -> list:
    """Per-gate quota: reserve min seats, never exceed max, then fill by score."""
    cap = max(0, int(total_cap))
    pmin = max(0, int(per_min))
    pmax = max(pmin, int(per_max))
    by: dict[str, list] = {}
    for p in sub:
        by.setdefault(str(p.get("gate", "")), []).append(p)
    for rows in by.values():
        rows.sort(key=_market_cand_score, reverse=True)
    chosen, used = [], set()
    per_gate: dict[str, int] = {}
    for gid in gates:
        for p in by.get(str(gid), [])[:pmin]:
            if len(chosen) >= cap:
                break
            fid = str(p.get("factor_id"))
            if fid in used:
                continue
            chosen.append(p)
            used.add(fid)
            per_gate[str(gid)] = per_gate.get(str(gid), 0) + 1
    if len(chosen) < cap:
        pool = []
        for gid in gates:
            pool.extend(by.get(str(gid), []))
        pool.sort(key=lambda p: -_market_cand_score(p))
        for p in pool:
            if len(chosen) >= cap:
                break
            gid = str(p.get("gate", ""))
            fid = str(p.get("factor_id"))
            if fid in used or per_gate.get(gid, 0) >= pmax:
                continue
            chosen.append(p)
            used.add(fid)
            per_gate[gid] = per_gate.get(gid, 0) + 1
    return chosen


def _select_sector_cands(sub: list, sec_gates: list, cap: int, scand_fn) -> list:
    """行业桶装配：按 |t_sec| 降序 + |t|>=2.0 过滤 + Top 上限（分门控）。
    绝不用日均门控 gmean。缺 t_sec 的门控无候选。返回 cand 记录列表。"""
    import numpy as _np
    sec_cands = []
    _cap_s = max(0, int(cap))
    for sg in sec_gates or []:
        _gid = str(sg.get("gate_id", ""))
        if not _gid:
            continue
        _ranked = sorted(
            [p for p in sub
             if _np.isfinite((p.get("t_sec") or {}).get(_gid, float("nan")))
             and abs((p.get("t_sec") or {})[_gid]) >= 2.0],
            key=lambda p: -abs(p["t_sec"][_gid]))
        if len(_ranked) > _cap_s:
            _ranked = _ranked[:_cap_s]
        sec_cands += [scand_fn(p, sg) for p in _ranked]
    return sec_cands


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
    ap.add_argument("--max-factors", type=int, default=400,
                    help="扫描上限（默认 400 覆盖 #1~#400 全窗，释放 #151~#400 "
                         "的 flip 富集区；旧 150 会把无条件 LGB≈0 的 flip 全截掉）")
    ap.add_argument("--max-switch", type=int, default=12)
    ap.add_argument("--max-market-cands", type=int, default=30,
                    help="gain/flip total cap (default 30; switch is extra)")
    ap.add_argument("--per-gate-min", type=int, default=2,
                    help="minimum candidates per certified bank gate")
    ap.add_argument("--per-gate-max", type=int, default=5,
                    help="maximum candidates per certified bank gate")
    ap.add_argument("--max-sector-cands", type=int, default=15,
                    help="单行业门控单 form 容量上限：按 |t_sec| 降序截断 "
                         "（使 sector 检验族 M<=15*门控数）")
    ap.add_argument("--data-cut", default=W.DEFAULT_DATA_CUT)
    ap.add_argument("--resume", default="",
                    help="已算出的specialist_pool.json，跳过其中已有factor_id")
    ap.add_argument("--no-export-panel", action="store_true",
                    help="只输出pool/cands JSON，不写因子值面板")
    ap.add_argument("--jobs", type=int, default=1,
                    help="因子评估并行进程数（12 核建议 8；"
                         "结果与串行逐值一致，非 posix 平台自动退化串行）")
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

    # 行业预筛静态 bundle（Bug4）：sector 门控横截面 halves 响应 t_sec 用。
    # 缺面板/secmap 时为 None → t_sec 全空，sector cands 为空并大声告警
    # （不清灰旧行为，避免静默重演 FDR 分母膨胀）。
    sec_gates = [e for e in bank.get("entries", []) if e.get("level") == "sector"]
    print(f"bank sector gates={len(sec_gates)}", flush=True)
    _sec_static = None
    if sec_gates:
        try:
            import gates as _G
            _secmap = _G.load_sector_map()
            _sg = pd.read_parquet(os.path.join(SPACE, "gate_search_sector.parquet"),
                                  columns=["date", "sector", "gate_id", "p"])
            _sg["date"] = pd.to_datetime(_sg["date"])
            _gates = {}
            for _e in sec_gates:
                _gid = str(_e["gate_id"])
                _piv = _sg[_sg["gate_id"] == _gid].pivot_table(
                    index="date", columns="sector", values="p", aggfunc="mean")
                _piv = _piv.sort_index()
                _gates[_gid] = {
                    "dates": _piv.index.to_numpy(),
                    "cols": [str(c) for c in _piv.columns],
                    "G": (2.0 * _piv.to_numpy(dtype=float) - 1.0)}
            _sec_static = {"gates": [{"gate_id": str(e["gate_id"])}
                                     for e in sec_gates],
                           "secmap": {str(k): str(v) for k, v in _secmap.items()},
                           "panels": _gates}
            print(f"sector 预筛面板 gates={len(_gates)}", flush=True)
        except Exception as _e:
            print(f"warning: sector 预筛面板加载失败 {str(_e)[:120]}，"
                  f"sector cands 将为空", flush=True)
            _sec_static = None

    done_ids: set[str] = set()
    if a.resume and os.path.exists(a.resume):
        try:
            _old = json.load(open(a.resume, encoding="utf-8"))
            done_ids = {f["factor_id"] for f in _old.get("factors", [])}
            print(f"resume: skip {len(done_ids)} computed", flush=True)
        except Exception:
            pass

    frame = load_panel()
    ev = Evaluator(frame, cache_size=32)
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

    # ctx 行级行业索引（sector 预筛用；与 ctx 同行序，一次算好）。
    # row_day/row_col 按门控分别对齐（各门控日期覆盖可能不同，-1 表缺失）。
    _sec_bundle = None
    if _sec_static is not None:
        try:
            _row_sec_name = pd.Series(
                np.asarray(ctx.stock_codes).astype(str)).map(
                    _sec_static["secmap"]).fillna("__NA__").to_numpy()
            _row_dates64 = pd.to_datetime(np.asarray(ctx.row_dates)).to_numpy()
            _sec_bundle = {
                "gates": _sec_static["gates"],
                "panels": _sec_static["panels"],
                "row_day": {
                    gid: pd.DatetimeIndex(gp["dates"]).get_indexer(_row_dates64)
                    for gid, gp in _sec_static["panels"].items()},
                "row_col": {
                    gid: np.array([{c: j for j, c in enumerate(gp["cols"])}.get(
                        s, -1) for s in _row_sec_name])
                    for gid, gp in _sec_static["panels"].items()}}
        except Exception as _e:
            print(f"warning: sector 行索引失败 {str(_e)[:120]}", flush=True)
            _sec_bundle = None

    pool = []
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
    # 单因子评估走 _run_specialist_one（串行/并行共用）；面板值由导出兜底现算，
    # 不再全量缓存（原 panel_vals 150×53MB≈8GB，是内存峰值主因之一）。
    _sp_ctx = {"frame": frame, "ev": ev, "builder": builder,
               "specs": specs_by_id, "store_map": store_map,
               "gpanel": gpanel, "gates": gates, "pos": pos,
               "labels": labels, "row_dates": ctx.row_dates,
               "t_min": a.t_min, "sec": _sec_bundle}
    _nj = _par_jobs(a.jobs)
    if _nj <= 1 or len(todo) <= 1:
        for i, r in enumerate(todo, 1):
            try:
                rec = _run_specialist_one(r, _sp_ctx)
            except Exception:
                continue
            if rec is None:
                continue
            pool.append(rec)
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {rec['factor_id']} "
                      f"t={rec['t_slope']:+.2f}@{rec['gate']} "
                      f"sel={rec['selected']} {time.time()-t0:.0f}s", flush=True)
    else:
        import concurrent.futures as _cf
        global _SP_CTX
        _SP_CTX = _sp_ctx
        _nj = min(_nj, len(todo))
        print(f"  并行评估：{len(todo)} 因子 × {_nj} 进程 …", flush=True)
        _items = list(enumerate(todo, 1))
        _done, _buf = 0, []
        with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
            for _idx, _rec in _ex.map(_specialist_worker, _items):
                if _rec is not None:
                    _buf.append((_idx, _rec))
                _done += 1
                if _done % 10 == 0 or _done == len(todo):
                    print(f"  [{_done}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
        _buf.sort(key=lambda x: x[0])
        pool = [r for _, r in _buf]

    sel = [p for p in pool if p["selected"]]
    _na = sum(1 for p in sel if p.get("target_class") == "A")
    _nb = sum(1 for p in sel if p.get("target_class") == "B")
    print(f"specialists: {len(sel)}/{len(pool)} (A主力={_na}, B半场={_nb})",
          flush=True)

    # switch对：offense/defense代表 × bank前2门控（V1启发式，上限max-switch）
    # roles.csv 因子名→group 映射（store 配对用；缺文件/缺列时退化为空映射，
    # store 全归 other，即旧行为）。
    _roles_group_map: dict = {}
    try:
        _roles_csv = os.path.join(RESULTS, "roles.csv")
        if os.path.exists(_roles_csv):
            _rdf = pd.read_csv(_roles_csv, usecols=["name", "group"])
            _roles_group_map = {
                str(_rn): str(_rg) for _rn, _rg in
                zip(_rdf["name"].tolist(), _rdf["group"].tolist())}
    except Exception:
        _roles_group_map = {}
    # switch互补配对（V4.3x 修复）：旧口径按无条件LGB取轴代表——全天候强因子
    # 静态等权已享分散化红利（信噪比×√2），动态切换白白丢红利却无状态收益，
    # 故0/12恒灭。新口径（gemini裁决）：off=进攻轴∩对**同一门控**有向斜率
    # 显著为正（signed_t≥+1.5），def=防御轴∩显著为负（signed_t≤-1.5）；
    # gate取各因子自身best gate（有斜率统计的唯门口径），按|signed_t|降序
    # 配对，上限max-switch。选出几个算几个，不凑数。
    switch_pairs = []
    _slope_thr = 1.5

    def _signed_t(p):
        try:
            _t = float(p.get("t_slope", float("nan")))
        except Exception:
            return float("nan")
        return p.get("direction", 1) * _t if np.isfinite(_t) else float("nan")

    _off_pool = [p for p in sel
                 if _resolve_axis(pd.Series({"family": p.get("family"),
                                             "factor_id": p.get("factor_id"),
                                             "formula": p.get("formula"),
                                             "formula_cn": p.get("formula")}),
                                  _roles_group_map) in OFFENSE_AXES
                 and np.isfinite(_signed_t(p)) and _signed_t(p) >= _slope_thr]
    _def_pool = [p for p in sel
                 if _resolve_axis(pd.Series({"family": p.get("family"),
                                             "factor_id": p.get("factor_id"),
                                             "formula": p.get("formula"),
                                             "formula_cn": p.get("formula")}),
                                  _roles_group_map) in DEFENSE_AXES
                 and np.isfinite(_signed_t(p)) and _signed_t(p) <= -_slope_thr]
    _by_gate_off: dict[str, list] = {}
    _by_gate_def: dict[str, list] = {}
    for p in _off_pool:
        _by_gate_off.setdefault(str(p["gate"]), []).append(p)
    for p in _def_pool:
        _by_gate_def.setdefault(str(p["gate"]), []).append(p)
    for _gid in sorted(set(_by_gate_off) & set(_by_gate_def)):
        _os = sorted(_by_gate_off[_gid], key=lambda p: -abs(_signed_t(p)))
        _ds_ = sorted(_by_gate_def[_gid], key=lambda p: -abs(_signed_t(p)))
        _used_off, _used_def = set(), set()
        for ro in _os:
            if len(switch_pairs) >= a.max_switch:
                break
            if str(ro["factor_id"]) in _used_off:
                continue
            for rd in _ds_:
                if (str(rd["factor_id"]) in _used_def
                        or str(ro["factor_id"]) == str(rd["factor_id"])):
                    continue
                switch_pairs.append({
                    "off": str(ro["factor_id"]), "def": str(rd["factor_id"]),
                    "gate": _gid, "s": int(ro.get("s_bank", 1)),
                    "off_t": round(_signed_t(ro), 3),
                    "def_t": round(_signed_t(rd), 3)})
                _used_off.add(str(ro["factor_id"]))
                _used_def.add(str(rd["factor_id"]))
                break
    print(f"switch pairs: {len(switch_pairs)} "
          f"(off_pool={len(_off_pool)} def_pool={len(_def_pool)}, "
          f"互补口径 |signed_t|>={_slope_thr})", flush=True)

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
        sub = _quota_select(
            sub, [str(g["gate_id"]) for g in gates],
            total_cap=int(a.max_market_cands),
            per_min=int(getattr(a, "per_gate_min", 2)),
            per_max=int(getattr(a, "per_gate_max", 5)))
        with open(os.path.join(RESULTS, f"specialist_cands_{form}.json"), "w",
                   encoding="utf-8") as f:
            json.dump([_cand(p, form) for p in sub], f, ensure_ascii=False, indent=1)
        _counts = {str(g["gate_id"]): 0 for g in gates}
        for p in sub:
            _counts[str(p["gate"])] = _counts.get(str(p["gate"]), 0) + 1
        print(f"specialist_cands_{form}.json: {len(sub)} "
              f"per-gate={ {k: v for k, v in _counts.items() if v} }", flush=True)
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
            # Bug4 行业预筛：按 |t_sec| 降序 + |t|>=2.0 过滤 + Top 上限。
            # 绝不用日均门控 gmean（横截面信号会被抹平）。缺 t_sec 时该门控
            # 无候选（大声告警，不清灰旧行为）。
            _before = len(sub)
            sec_cands = _select_sector_cands(
                sub, sec_gates, a.max_sector_cands,
                lambda p, sg, _f=form: _scand(p, _f, sg))
            if not sec_cands and _before:
                print(f"warning: specialist_sector_cands_{form}.json 为空 "
                      f"（{len(sub)} 候选中无 |t_sec|>=2.0 者）", flush=True)
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
