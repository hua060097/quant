# -*- coding: utf-8 -*-
"""行业侧 G2/G3 后端（§8.2 + §17.2：**保留专用后端，不机械复用市场实现**）。

## 为什么不能复用市场实现（Codex 第 3 条）

| 维度 | 市场门控 | 行业门控 |
|---|---|---|
| 门控取值 | 每日一个标量（全市场共享） | 每 (股票, 日) 一个值（该股所属行业的当日值） |
| 条件效应 | 日级掩码 `IC_t·1[g_t>0]`（掩码捷径） | **逐股掩码**，必须重建 F1 分数再排名 |
| 有效样本（§8.3） | 交易日数 | **行业数（131）**，须同时报告行业内/行业间 |
| 成本 | 秒级 | ~1.5 s/(门控×因子)（须重建 F1） |

市场版的门控是标量 → 掩码不改变日内秩 → 可用捷径；
行业版的门控**因股而异** → 必须按 gain 式 `F1 = u·clip(1+0.5·s·g)`
（`u` 有向居中秩，`g = 2p-1` 中心化门控；与 `f1_sector --form gain` 同构，
V4.3a 起废除旧 `d·x·max(s·g,0)` 截断式）重建后排名。

## 本实现的统计量（无偏对称差口径）

- **主口径 `delta_sym`**：`Δ = mean(IC(u·w(+g)) − IC(u·w(−g)))`——正反对称，
  乘性方差稀释偏置在零假设下严格相消（期望为 0）；符号直接锁定 bank s。
  旧 `IC(F1)−IC(base)` 口径因恒正偏置（实测 23 门控全挤在 −0.001）已废除。
- **分簇**：与市场 `--by-group` 对齐（7 真机制簇；t2/ungrouped 不成簇）；
  全池优先，仅全池未过时分簇拯救。

G2 判定：4 折同号 + 块自助（块长 5，时间块）95% 区间不含 0 + 同号因子占比 ≥60%。
G3：**块置换门控矩阵**（全行业同置换，与 `f1_sector` 的 placebo 一致）→ 零分布 →
V4.3a gain 口径新族预注册**双侧**（`|Δ| > q95(|null|)`，方向由 bank s 锁定；
旧 `form="t"` 族的单侧冻结规格保留为诊断列，不改旧行）。

**G3 的成本控制（如实声明）**：置换需为每个因子重建 F1，成本 = n_perm × n_factors。
全量 200×123 不可行（≈10 小时），故 G3 用**受限因子集 + 较少置换**（默认 20×40），
仅用于构造零分布标度，**不改变 G2 的因子样本**。

用法：
    python gate_g2g3_sector.py --limit-factors 12 --n-perm 20   # 冒烟
    python gate_g2g3_sector.py                                   # 全量
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
import boundary as B  # noqa: E402
import eval_v2 as E  # noqa: E402
import fastranks as FR  # noqa: E402
import gate_ledger as GL  # noqa: E402
import gate_g2g3 as GG  # noqa: E402  复用双侧判定 ci_excludes_zero（避免口径分叉）
import gates as G  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")
CUT = W.FACTOR_SCREEN_MAX

G2_FOLDS = 4
G2_SAME_SIGN_MIN = 0.60
G2_BOOT_BLOCK = 5
G2_BOOT_N = 200
PLACEBO_BLOCK = 20


def load_panel(names: list[str]):
    """面板 = 因子列 + r3 + 行业映射（供逐股掩码用）。

    V4.3a：返回前按 date 稳定排序。`factors.parquet` 是 stock-major 行序，
    而 `PanelRanker.day` 是 date-major 码——不对齐会导致门控归属与日内分组
    全错位（bank-v1 行业脏数据的第二病因，仅次于截断）。
    """
    lab = B.read_labels(["r3", "tradable"], label_max=CUT, label="g2sec")
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab[(lab["tradable"] == 1) & (lab["date"] <= CUT)]
    fac = B.read_factors(names, data_max=W.LABEL_MAX_DATE, label="g2sec")
    fac["date"] = pd.to_datetime(fac["date"])
    m = fac.merge(lab[["stock_code", "date", "r3"]], on=["stock_code", "date"],
                  how="inner")
    m = m[m["date"] <= CUT].copy()
    sm = G.load_sector_map()
    m["_sec"] = m["stock_code"].astype(str).map(sm)
    m = m.sort_values("date", kind="stable").reset_index(drop=True)
    return m


def _centered_u(xv: np.ndarray, day: np.ndarray, nd: int, d: float) -> np.ndarray:
    """有向居中百分位秩（统一定义见 fastranks.centered_u；此别名保留供内部调用）。
    输入必须为**原始值**（函数内排名后再乘 d）；传入 d*xv 会在 d=-1 时负负得正。
    """
    return FR.centered_u(xv, day, nd, int(d))


def build_sector_gate_matrix(sg: pd.DataFrame, gid: str, uniq: np.ndarray):
    """返回 (G [n_dates, n_sectors], sector_col_index)。"""
    sub = sg[sg["gate_id"] == gid]
    piv = sub.pivot_table(index="date", columns="sector", values="p", aggfunc="mean")
    piv = piv.reindex(pd.to_datetime(uniq))
    cols = list(piv.columns)
    return piv.to_numpy(dtype=float), {c: i for i, c in enumerate(cols)}


def prep_row_index(day: np.ndarray, sec_ids: np.ndarray,
                   colmap: dict) -> tuple[np.ndarray, np.ndarray]:
    """预计算**逐行**的 (date_idx, sector_col_idx)。

    **性能关键（实测）**：原实现把
    `np.array([colmap.get(x, -1) for x in sec_ids])` 放在 `f1_ic_series` 内部，
    每次调用都要对 500 万行做 Python 级字典查找（≈8 秒/次）。
    在 5 因子×2 门控 + 6×4×2 次 G3 置换的冒烟档下累计约 464 秒（占 427s 总耗时的绝大部分）。
    本函数把它 **只算一次**。
    """
    uniq_sorted = np.sort(np.unique(day))
    date_idx = np.searchsorted(uniq_sorted, day)
    ci = np.array([colmap.get(x, -1) for x in sec_ids], dtype=np.int64)
    return date_idx, ci


def f1_ic_series_from_idx(u, day, nd, yrk, date_idx, ci, G, s=1):
    """按逐股掩码重建 F1 → 日内排名 → 逐日 IC（索引已预计算）。

    V4.3a（gain 口径新族）：``F1 = u * clip(1 + 0.5*s*g)``，其中
    ``u`` 为调用方预计算的有向居中秩（``_centered_u``），``g = 2*vv - 1``
    为中心化门控状态。相对旧实现的三处修正：
    1. 门控中心化：面板 ``p ∈ (0,1]``，旧式直接 ``1+0.5*vv`` 使权重恒 ≥1
       （只有放大没有衰减）；现 ``g ∈ (-1,1]`` 对称调制；
    2. 无塌缩：旧 ``max(s*vv, 0)`` 清零致 ΔIC 坍缩 -0.020，已废除；
    3. 无 Level-Leak：输入为日内居中秩而非原始值，截面均值已消除。
    """
    u = np.asarray(u, dtype=float)
    ok = (ci >= 0) & np.isfinite(u)
    inb = ci < G.shape[1]
    sel = ok & inb
    vv = np.full(len(u), np.nan)
    vv[sel] = G[date_idx[sel], ci[sel]]
    g = 2.0 * vv - 1.0
    ww = np.clip(1.0 + 0.5 * float(s) * g, 0.2, 1.8)
    ww[~np.isfinite(g)] = np.nan
    f1 = np.where(np.isfinite(u) & np.isfinite(ww), u * ww, np.nan)
    fr = FR.blockwise_rank(f1, day)
    return E._daily_corr(fr, yrk, day, nd)


def f1_ic_series(u, day, nd, yrk, sec_ids, G, colmap, s=1):
    """兼容旧签名（内部预计算索引；单次调用场景用）。"""
    date_idx, ci = prep_row_index(day, sec_ids, colmap)
    return f1_ic_series_from_idx(u, day, nd, yrk, date_idx, ci, G, s)


def _g2_worker(args) -> dict:
    """G2 工作进程：因子子集 × 全部行业门控 → 对称差日序列。

    自包含（独立 load_panel/PanelRanker/门控索引；CUT 取模块全局，与主进程一致），
    返回 {gate: [(factor, dsym_mean, dsym_daily_list, d), ...]}
    （daily 转 list 便于 pickle）。
    """
    facs, gates, chunk = args
    import pandas as _pd
    out: dict = {g: [] for g in gates}
    _sg = _pd.read_parquet(os.path.join(SPACE, "gate_search_sector.parquet"))
    _sg["date"] = _pd.to_datetime(_sg["date"])
    _sg = _sg[_sg["date"] <= CUT].copy()
    _day = _nd = None
    _uniq = _sec = None
    _gmats = {}
    _chunks = [facs[i:i + chunk] for i in range(0, len(facs), chunk)]
    for _ck in _chunks:
        _m = load_panel(_ck)
        _pr = FR.PanelRanker(_m, verbose=False)
        if _day is None:
            _day, _nd = _pr.day, _pr.n_days
            _uniq = np.unique(_m["date"].to_numpy())
            _sec = _m["_sec"].astype(str).to_numpy()
            for _g in gates:
                _Gm, _cm = build_sector_gate_matrix(_sg, _g, _uniq)
                _di, _ci = prep_row_index(_day, _sec, _cm)
                _gmats[_g] = (_Gm, _di, _ci)
        for _c in _ck:
            _ic_raw = _pr.daily_ic(_pr.x_rank(_c))
            _d = 1.0 if float(np.nanmean(_ic_raw)) >= 0 else -1.0
            _xv = pd.to_numeric(_m[_c], errors="coerce").to_numpy(dtype=float)
            _u = _centered_u(_xv, _day, _nd, _d)
            for _g in gates:
                _Gm, _di, _ci = _gmats[_g]
                _icp = f1_ic_series_from_idx(_u, _day, _nd, _pr.y, _di, _ci, _Gm, 1)
                _icm = f1_ic_series_from_idx(_u, _day, _nd, _pr.y, _di, _ci, _Gm, -1)
                _m1, _m2 = np.isfinite(_icp), np.isfinite(_icm)
                _ds = np.full(len(_icp), np.nan)
                _ds[_m1 & _m2] = _icp[_m1 & _m2] - _icm[_m1 & _m2]
                out[_g].append((_c, float(np.nanmean(_ds))
                                if np.isfinite(_ds).any() else float("nan"),
                                _ds.tolist(), float(_d)))
        del _m, _pr
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="行业侧 G2/G3（专用后端）")
    ap.add_argument("--limit-factors", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="G2 并行工作进程数（按因子分组；12 核服务器建议 10；"
                         "默认 1 保持原行为）")
    ap.add_argument("--gates", nargs="*", default=None)
    ap.add_argument("--g3-gates", nargs="*", default=None,
                    help="只对这些门控跑 G3（G2 仍跑全部）。成本 ≈ n_gates × n_perm × "
                         "g3_factors × 1.38**秒**（实测）；不给则全部跑")
    ap.add_argument("--no-g3", action="store_true", default=True,
                    help="默认跳过 G3（行业 G3 与下游 f1_sector 200 次置换重复，省数小时；"
                         "G2 双侧已足够强）。加 --run-g3 则跑 G3。")
    ap.add_argument("--run-g3", dest="no_g3", action="store_false",
                    help="显式跑 G3（与 --g3-gates 联用限定门控）")
    ap.add_argument("--chunk", type=int, default=15,
                    help="因子分块（控制内存；每块独立建 PanelRanker 缓存 y 排名）")
    ap.add_argument("--n-boot", type=int, default=G2_BOOT_N)
    ap.add_argument("--n-perm", type=int, default=40, help="G3 置换次数（受限）")
    ap.add_argument("--g3-factors", type=int, default=20,
                    help="G3 使用的因子数（成本 = n_perm × n_factors）")
    ap.add_argument("--min-factors", type=int, default=10,
                    help="组级统计所需的最少因子数（<此值跳过；"
                         "**冒烟可调小**，全量应为 10）")
    ap.add_argument("--out", default=os.path.join(SPACE, "g2g3_sector.csv"))
    # 2026-10-12 新增：与市场线 `run_gate_pipeline.py --ledger` 对齐。
    # 原实现无条件写 `GL.DEFAULT_PATH` → 冒烟测试会污染生产台账（已实际发生）。
    ap.add_argument("--ledger", default=GL.DEFAULT_PATH,
                    help="台账路径（默认生产台账；冒烟/回归须指向独立路径）")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    roles = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
    names = list(roles.query("role=='STOCK'")["name"])
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(os.path.join(ROOT, "factor_lab_data",
                                           "factors.parquet")).schema_arrow.names)
    names = [c for c in names if c in have]
    if a.limit_factors:
        names = names[:a.limit_factors]

    sc = pd.read_csv(os.path.join(SPACE, "screen.csv"))
    el = sc[(sc["level"] == "sector") & (sc["eligible"])]["gate_id"].tolist()
    if a.gates:
        el = [g for g in el if g in a.gates]
    print(f"行业门控 {len(el)} 个；因子 {len(names)} 个（分块 {a.chunk}）；"
          f"G3 用 {min(a.g3_factors, len(names))} 因子 × {a.n_perm} 置换")

    sg = pd.read_parquet(os.path.join(SPACE, "gate_search_sector.parquet"))
    sg["date"] = pd.to_datetime(sg["date"])
    sg = sg[sg["date"] <= CUT]

    # ---------------------------------------------------------------- 分块 G2
    # 无偏对称差口径（消除乘性方差稀释偏置）：Δsym,daily = IC(u·w(+g)) − IC(u·w(−g))，
    # 零假设下期望严格为 0（双侧方差稀释项相消）；符号直接锁定 bank s。
    # 逐块：载入 → PanelRanker（y 排名缓存）→ 每 (因子×门控) 正负两套 F1 IC。
    per_gate_ds = {g: [] for g in el}          # gate -> [per-factor Δsym 均值]
    per_gate_syms = {g: [] for g in el}        # gate -> [per-factor Δsym 日序列]
    per_gate_factors = {g: [] for g in el}     # gate -> [因子名]（与上面两表对齐）
    dirmap_all = {}
    if int(a.jobs) > 1:
        # 并行 G2：按因子分组，每组独立进程（各自 load_panel/PanelRanker）。
        import concurrent.futures as _cf
        _nj = max(1, int(a.jobs))
        _groups = [names[i::max(1, _nj)] for i in range(max(1, _nj))]
        _groups = [gg for gg in _groups if gg]
        print(f"  并行 G2：{len(_groups)} 组 × {len(el)} 门控 …", flush=True)
        with _cf.ProcessPoolExecutor(max_workers=len(_groups)) as _ex:
            _futs = [_ex.submit(_g2_worker, (gg, el, a.chunk)) for gg in _groups]
            for _fu in _cf.as_completed(_futs):
                _res = _fu.result()
                for _g in el:
                    for _c, _dm, _dl, _dd in _res.get(_g, []):
                        per_gate_factors[_g].append(_c)
                        per_gate_ds[_g].append(_dm)
                        per_gate_syms[_g].append(np.asarray(_dl, dtype=float))
                        dirmap_all[_c] = _dd
        print(f"  并行 G2 完成 {time.time() - t0:.0f}s", flush=True)
    else:
        day = nd = None
        uniq = sec_ids = None
        chunks = [names[i:i + a.chunk] for i in range(0, len(names), a.chunk)]
        for ci, chunk in enumerate(chunks, 1):
            m = load_panel(chunk)
            pr = FR.PanelRanker(m, verbose=(ci == 1))
            if day is None:
                day, nd = pr.day, pr.n_days
                uniq = np.unique(m["date"].to_numpy())
                sec_ids = m["_sec"].astype(str).to_numpy()
                # 对齐断言：m 已按 date 排序，PanelRanker 内排序应为恒等，
                # 否则门控归属/日内分组全错位（bank-v1 行业脏数据主因之一）。
                _chk = pd.to_datetime(m["date"]).to_numpy()
                assert (_chk == pr.uniq[pr.day]).all(), "m行序与PanelRanker日码错位"
                # 预计算每个门控的 (date_idx, sector_col_idx)，避免逐 (因子×门控) 重算
                gmats = {}
                for g in el:
                    Gmat, colmap = build_sector_gate_matrix(sg, g, uniq)
                    di, ci = prep_row_index(day, sec_ids, colmap)
                    gmats[g] = (Gmat, di, ci)
                print(f"  索引预计算完成 {time.time() - t0:.0f}s", flush=True)
            for c in chunk:
                _ic_raw = pr.daily_ic(pr.x_rank(c))         # 方向 = sign(全样本 IC 均值)
                d = 1.0 if float(np.nanmean(_ic_raw)) >= 0 else -1.0
                dirmap_all[c] = d
                xv = pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)
                # 对原始值排名再乘 d（与 f1_batch._centered_u 逐字同义；勿传 d*xv，
                # 否则 d=-1 时负负得正方向全反——f1_sector 同 bug 已并修）。
                u = _centered_u(xv, day, nd, d)
                for g in el:
                    Gmat, di, ci = gmats[g]
                    ic_p = f1_ic_series_from_idx(u, day, nd, pr.y, di, ci, Gmat, 1)
                    ic_m = f1_ic_series_from_idx(u, day, nd, pr.y, di, ci, Gmat, -1)
                    m1, m2 = np.isfinite(ic_p), np.isfinite(ic_m)
                    dsym = np.full(len(ic_p), np.nan)
                    dsym[m1 & m2] = ic_p[m1 & m2] - ic_m[m1 & m2]
                    per_gate_syms[g].append(dsym)
                    per_gate_factors[g].append(c)
                    per_gate_ds[g].append(float(np.nanmean(dsym))
                                          if np.isfinite(dsym).any() else np.nan)
            del m, pr
            print(f"  块 {ci}/{len(chunks)}（{len(chunk)} 因子） "
                  f"{time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- 组级统计
    # 无偏对称差 + 机制分簇（与市场 --by-group 对齐）：全池优先，仅全池未过
    # （多机制对冲）时用 7 真机制簇拯救；t2_promoted/ungrouped 不成簇。
    def _stat_on(idxs):
        _ds = np.array([per_gate_ds[g][_j] for _j in idxs])
        _fin = np.isfinite(_ds)
        if int(_fin.sum()) < a.min_factors:
            return None
        _syms = [per_gate_syms[g][_j] for _j in np.flatnonzero(_fin)]
        _med = float(np.median(_ds[_fin]))
        _sgn = np.sign(_med) if _med != 0 else 1.0
        _same = float((np.sign(_ds[_fin]) == _sgn).mean())
        _edges = np.linspace(0, T, G2_FOLDS + 1).astype(int)
        _folds = 0
        for _i in range(G2_FOLDS):
            _sl = slice(_edges[_i], _edges[_i + 1])
            _fd = []
            for _s in _syms:
                _a1 = _s[_sl]
                if np.isfinite(_a1).any():
                    _fd.append(np.nanmean(_a1))
            _fd = np.asarray(_fd)
            _fd = _fd[np.isfinite(_fd)]
            if len(_fd) >= 5 and np.sign(np.median(_fd)) == _sgn:
                _folds += 1
        _rng = np.random.default_rng(0)
        _nb = max(1, T // G2_BOOT_BLOCK)
        _boots = np.empty(a.n_boot)
        for _bi in range(a.n_boot):
            _pick = _rng.integers(0, _nb, size=_nb)
            _idx = np.concatenate([np.arange(_j * G2_BOOT_BLOCK,
                                             min((_j + 1) * G2_BOOT_BLOCK, T))
                                   for _j in _pick])
            _per = []
            for _s in _syms:
                _a1 = _s[_idx]
                if np.isfinite(_a1).sum() > 20:
                    _per.append(np.nanmean(_a1))
            _boots[_bi] = np.nanmedian(np.asarray(_per))
        _boots = _boots[np.isfinite(_boots)]
        _lo = float(np.quantile(_boots, 0.025)) if len(_boots) else np.nan
        _hi = float(np.quantile(_boots, 0.975)) if len(_boots) else np.nan
        return {"med": _med, "sgn": _sgn, "same": _same, "folds": _folds,
                "lo": _lo, "hi": _hi, "n": int(_fin.sum())}

    _gmap = {}
    try:
        _roles = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
        _gmap = dict(zip(_roles["name"].astype(str), _roles["group"].astype(str)))
    except Exception:
        pass
    rows = []
    print(f"\n{'gate':<26s}{'Δsym中位':>11s}{'同号':>7s}{'4折':>6s}"
          f"{'CI下界':>11s}{'簇':>16s}{'G3':>6s}")
    T = len(per_gate_syms[el[0]][0])
    for g in el:
        _all = list(range(len(per_gate_syms[g])))
        _st = _stat_on(_all)
        if _st is None:
            print(f"{g:<26s}  跳过（有效因子不足 min_factors={a.min_factors}）",
                  flush=True)
            continue
        _ok_full = (_st["folds"] == G2_FOLDS) \
            and GG.ci_excludes_zero(_st["lo"], _st["hi"]) \
            and _st["same"] >= G2_SAME_SIGN_MIN
        _grp, _geff = "full_pool", _st["med"]
        _gsame, _gfolds, _glo, _ghi = _st["same"], _st["folds"], _st["lo"], _st["hi"]
        _gn, _gok = _st["n"], _ok_full
        if not _ok_full:
            _by_grp: dict[str, list[int]] = {}
            for _j in _all:
                _gg = _gmap.get(str(per_gate_factors[g][_j]), "ungrouped")
                if _gg in ("t2_promoted", "ungrouped"):
                    continue
                _by_grp.setdefault(_gg, []).append(_j)
            _best = None
            for _gnm, _cols in _by_grp.items():
                if len(_cols) < 5:
                    continue
                _s = _stat_on(_cols)
                if _s is None:
                    continue
                _ok = (_s["folds"] == G2_FOLDS) \
                    and GG.ci_excludes_zero(_s["lo"], _s["hi"]) \
                    and _s["same"] >= 0.65
                if _ok and (_best is None or abs(_s["med"]) > abs(_best[1])):
                    _best = (_gnm, _s["med"], _s["same"], _s["folds"],
                             _s["lo"], _s["hi"], _s["n"])
            if _best is not None:
                _grp, _geff = _best[0], _best[1]
                _gsame, _gfolds, _glo, _ghi, _gn = _best[2], _best[3], \
                    _best[4], _best[5], _best[6]
                _gok = True
        rows.append({"gate_id": g, "n_factors": _gn,
                     "delta_median": _st["med"], "same_frac": _st["same"],
                     "folds_same": _st["folds"], "ci_lo": _st["lo"], "ci_hi": _st["hi"],
                     "g2_pass": bool(_gok),
                     "slope_group": _grp, "median_b_eff": _geff,
                     "slope_group_same": _gsame, "slope_group_folds": _gfolds,
                     "slope_group_pass": bool(_gok and _grp != "full_pool"),
                     "g3_pass": False, "g3_p": np.nan, "g3_tested": False})
        print(f"{g:<26s}{_st['med']:>+11.5f}{_st['same']:>7.2f}"
              f"{_st['folds']:>4d}/{G2_FOLDS}{_st['lo']:>11.6f}{_grp:>16s}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    # ---------------------------------------------------------------- G2 先落盘
    # 2026-10-12 修复：G2 是 14 门控 × 124 因子 ≈ 50 分钟的成果，而 G3 更贵
    #（实测 ≈ n_perm × g3_factors × 1.38 s **每个门控**）。原实现在 G3 之后才写盘，
    # 一次 G3 中途超时（实测跑了 4h 被杀）会把 G2 全部丢失。故 G2 结束即落盘，
    # 且 G3 每完成一个门控就覆盖写一次。
    pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8")
    print(f"\n[G2 已落盘] {a.out}（{len(rows)} 个门控；G3 未跑前 g3_pass=False）",
          flush=True)
    if a.no_g3:
        print("\n--no-g3：G2 完成即退出（不跑 G3、不回写台账）。"
              "G3 结果请沿用上一次同口径运行的产物。")
        print(f"用时 {time.time() - t0:.0f}s")
        return 0

    # ---------------------------------------------------------------- G3（受限）
    # 仅幸存者跑 G3（显式 --g3-gates 可覆盖指定）：G2 已是强过滤，全量跑 G3
    # 是算力浪费；且对称差零假设必须走置换（单因子恒正偏置已证）。
    if a.g3_gates:
        g3_el = set(a.g3_gates)
    else:
        g3_el = {r["gate_id"] for r in rows if r.get("g2_pass")}
    g3_rows = [r for r in rows if r["gate_id"] in g3_el]
    for r in rows:
        r["g3_tested"] = bool(r["gate_id"] in g3_el)
    print(f"\nG3（受限因子集 {min(a.g3_factors, len(names))} × {a.n_perm} 置换 × "
          f"{len(g3_rows)} 幸存门控）…", flush=True)
    g3f = names[:min(a.g3_factors, len(names))]
    m3 = load_panel(g3f)
    pr3 = FR.PanelRanker(m3, verbose=False)
    day3, nd3 = pr3.day, pr3.n_days
    uniq3 = np.unique(m3["date"].to_numpy())
    sec3 = m3["_sec"].astype(str).to_numpy()
    _chk3 = pd.to_datetime(m3["date"]).to_numpy()
    assert (_chk3 == pr3.uniq[pr3.day]).all(), "m3行序与PanelRanker日码错位"
    u3 = {}
    for c in g3f:
        d = dirmap_all[c]
        xv = pd.to_numeric(m3[c], errors="coerce").to_numpy(dtype=float)
        u3[c] = _centered_u(xv, day3, nd3, d)
    for r in g3_rows:
        Gmat, colmap = build_sector_gate_matrix(sg, r["gate_id"], uniq3)
        di3, ci3 = prep_row_index(day3, sec3, colmap)
        ndates = Gmat.shape[0]
        nbb = max(1, ndates // PLACEBO_BLOCK)
        pieces = np.array_split(np.arange(ndates), nbb)
        rng2 = np.random.default_rng(0)
        nulls = np.empty(a.n_perm)
        for pi in range(a.n_perm):
            order = rng2.permutation(nbb)
            pidx = np.concatenate([pieces[j] for j in order])[:ndates]
            per = []
            for c in g3f:
                # 对称差零分布：同一置换下正负两套 F1 重建后相减（偏置相消）。
                ic_p = f1_ic_series_from_idx(u3[c], day3, nd3, pr3.y, di3, ci3,
                                             Gmat[pidx], 1)
                ic_m = f1_ic_series_from_idx(u3[c], day3, nd3, pr3.y, di3, ci3,
                                             Gmat[pidx], -1)
                m1, m2 = np.isfinite(ic_p), np.isfinite(ic_m)
                if m1.any() and m2.any():
                    per.append(ic_p[m1].mean() - ic_m[m2].mean())
            nulls[pi] = np.nanmedian(np.asarray(per))
        nulls = nulls[np.isfinite(nulls)]
        if len(nulls):
            # **双侧主判据**（gain 对称差新族预注册）：零假设下期望严格为 0，
            # 方向由对称差符号锁定（bank s）。
            q05 = float(np.quantile(nulls, 0.05))
            q95 = float(np.quantile(nulls, 0.95))
            q95abs = float(np.quantile(np.abs(nulls), 0.95))
            k1 = int((nulls >= r["delta_median"]).sum())
            k2 = int((np.abs(nulls) >= abs(r["delta_median"])).sum())
            r["g3_pass"] = bool(abs(r["delta_median"]) > q95abs)
            r["g3_p"] = float((1 + k2) / (len(nulls) + 1))
            r["g3_lo95"] = q05
            r["g3_hi95"] = q95
            r["g3_q95abs"] = q95abs
            r["g3_p_single_diag"] = float((1 + k1) / (len(nulls) + 1))
            r["g3_pass_single_diag"] = bool(r["delta_median"] > q95)
            r["dir"] = "positive" if r["delta_median"] > 0 else "negative"
        print(f"  {r['gate_id']:<26s} G3={'True' if r['g3_pass'] else 'False'} "
              f"p={r['g3_p']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        # G3 增量落盘（超时/中断时保住已完成的门控）
        pd.DataFrame(rows).to_csv(a.out, index=False, encoding="utf-8")
    del m3, pr3

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["g2_pass", "g3_pass", "delta_median"],
                            ascending=[False, False, False])
    else:
        df = pd.DataFrame(columns=["gate_id", "n_factors", "delta_median",
                                   "same_frac", "folds_same", "ci_lo",
                                   "g2_pass", "g3_pass", "g3_p"])
        print("\n⚠ 无门控进入组级统计（有效因子数不足 min_factors）")
    df.to_csv(a.out, index=False, encoding="utf-8")
    print(f"\n=== 汇总（{len(df)} 个行业门控）===")
    print(f"  G2 通过: {int(df['g2_pass'].sum())} | "
          f"G3 通过: {int(df['g3_pass'].sum())}")
    for _, r in df[df["g2_pass"]].iterrows():
        print(f"    {r['gate_id']:<26s} ΔIC中位 {r['delta_median']:+.5f} "
              f"同号 {r['same_frac']:.2f} 4折 {int(r['folds_same'])}/4")

    # 台账回写（**共享协议**：与市场线同一 gate_ledger，但记 level=sector）
    lg = GL.GateLedger(a.ledger)
    # 对称差口径（无偏）：delta = IC(u·w(+g)) − IC(u·w(−g))，零假设期望严格为 0；
    # g3_rule 记 "two-sided"（新族预注册；旧族单侧冻结规格不受影响）。
    cal = {"window_end": CUT, "form": "gain-sym", "delta_def": "symmetric",
           "g3_rule": "two-sided",
           "factor_set": f"stock{len(names)}",
           "label": "r3", "data_version": B.factor_data_version(),
           "n_perm": a.n_perm, "n_boot": a.n_boot}
    g3n = min(a.g3_factors, len(names))
    for _, r in df.iterrows():
        lg.log("g2g3_sector",
               {"level": "sector", "base": r["gate_id"], "op": None},
               caliber=cal,
               verdict=("candidate" if (r["g2_pass"] and r["g3_pass"]) else
                        "weak" if r["g2_pass"] else "reject"),
               data={"delta_median": r["delta_median"],
                     "same_frac": r["same_frac"],
                     "folds_same": int(r["folds_same"]), "ci_lo": r["ci_lo"],
                     "g2_pass": bool(r["g2_pass"]), "g3_pass": bool(r["g3_pass"]),
                     "g3_p": r["g3_p"], "n_factors": int(r["n_factors"]),
                     "g3_note": f"G3 用受限因子集 {g3n} × {a.n_perm} 置换 × "
                                f"{len(g3_rows)}/{len(rows)} 门控（成本约束）",
                     "backend": "sector 专用（逐股掩码 + F1 重建）"})
    print(f"\n输出 → {a.out}")
    print(f"台账 → {a.ledger}"
          f"（新增 {len(df)} 条 g2g3_sector 事件）")
    print(f"用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
