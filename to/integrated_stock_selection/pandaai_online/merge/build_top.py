"""两个榜单（aligned_v1 口径）：

榜1 = 官方A（c2c 标签）top20 —— 上传因子大赛决策用
      （主排序 A_official=|RankIC|×|ICIR(Pearson)|×IC胜率；附 C 代理复核列）
榜2 = LGB 复合分（vwap 标签）top20 —— 自用 LightGBM（分类选每日可交易 topN, n<=5，
      2~3 日持仓）：
      LGB = |RankIC|×|RankICIR|×rank胜率 × |mono| × T_ref/max(单次调仓换手,T_ref)

候选池（全部）：
  - 路径1/2：merge 台账挖矿候选（按旧证据 sia 每标签前 N 并集预筛 → aligned_v1 全窗重估）
  - 路径3：refs.py 的外部因子筛选结果（refs_screen_*_flat.csv，aligned_v1 直评）

用法：
  python build_top.py --top 200
产物：out/ALIGNED_TOP2_<ts>.md + .json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import windows as W
from cache_utils import find_stock_features
from decorrelate_top import factor_full_values, load_all_specs, month_end_spearman_matrix
from evaluator import Evaluator, load_panel
from fitness import DataContext, build_label_series, sia_of, tier1
from grammar import expression_hash
from ledger import Ledger
from nonlinear import jsd_tails, mi_monthly
from parser import parse_formula
from refs import FactorBuilder, lgb_score
from run_search import _ledger_candidates

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'out'
CORR_THRESHOLD = 0.7

#: 并行工作进程共享（fork 继承，免 pickling 大面板）
_MP_FRAME = None
_MP_END = None


def _par_init(lock) -> None:
    """进程池初始化：ledger 跨进程写锁（frame/end 经 fork 继承）."""
    import ledger as _LG
    _LG.set_write_lock(lock)


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


def _auto_jobs_build_top() -> int:
    """--jobs auto：按核数+可用内存自动定并行进程数。
    fork 共享面板，每 worker 工作集约 3GB（值计算/打分）；
    取 min(核数-2, 0.7×可用内存/3GB)，下限 1。"""
    import os
    cores = os.cpu_count() or 4
    mem_gb = None
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    mem_gb = int(line.split()[1]) / 1048576.0
                    break
    except Exception:
        pass
    if mem_gb is None:
        try:
            import psutil
            mem_gb = psutil.virtual_memory().available / 1073741824.0
        except Exception:
            mem_gb = 8.0
    cpu_cap = max(1, cores - 2)
    mem_cap = max(1, int(0.7 * mem_gb / 3.0))
    nj = max(1, min(cpu_cap, mem_cap))
    print(f'[AUTO] 核数={cores} 可用内存={mem_gb:.1f}GB -> jobs={nj} '
          f'(cpu限={cpu_cap} 内存限={mem_cap})', flush=True)
    return nj


#: 非线性诊断(MI/JSD)并行共享（fork 继承 cols/y/日分组，免 pickling 大数组）
_NL_CTX = None


def _nl_worker(j: int):
    """非线性诊断工作进程：单候选 MI/JSD（纯函数，串并逐值一致）。"""
    c = _NL_CTX
    try:
        _d = mi_monthly(c["cols"][j], c["yv"], c["day_group"])
        _d["jsd"] = jsd_tails(c["cols"][j], c["yv"])
        return j, _d
    except Exception:
        return j, None


def _nl_backfill(cols: list, idxs: list, labels: list, yv,
                 day_group, jobs: int = 1) -> dict:
    """MI/JSD 回填（串并同一 worker；_ex.map 保序 → nl_map 插入顺序与串行一致）。

    cols: 已求值候选数组列表；idxs: 对应的 df 行号；labels: list(df.index)。
    """
    nl_map: dict = {}
    _nj = _par_jobs(jobs)
    if _nj > 1 and len(cols) > 1:
        global _NL_CTX
        _NL_CTX = {"cols": cols, "yv": yv, "day_group": day_group}
        import concurrent.futures as _cf
        _nw = min(_nj, len(cols))
        print(f'非线性诊断并行：{len(cols)} 个候选 × {_nw} 进程 …', flush=True)
        try:
            with _cf.ProcessPoolExecutor(max_workers=_nw) as _ex:
                for _j, _d in _ex.map(_nl_worker, range(len(cols))):
                    if _d is not None:
                        nl_map[labels[idxs[_j]]] = _d
        finally:
            _NL_CTX = None
    else:
        for _j, _v in enumerate(cols):
            try:
                _d = mi_monthly(_v, yv, day_group)
                _d['jsd'] = jsd_tails(_v, yv)
                nl_map[labels[idxs[_j]]] = _d
            except Exception:
                continue
    return nl_map


#: 条件 RankIC 并行共享（fork 继承 frame/ctx/prep，免 pickling 大对象）
_COND_CTX = None


def _cond_worker(task):
    """条件 RankIC 工作进程：单候选（串并同一份求值代码，逐值一致）。

    task = (lib, fid, formula_raw, formula_str, family)：formula_raw 原样回填
    输出字段（与串行 r['formula_cn'] 恒等），formula_str 供求值解析。
    """
    lib, fid, formula_raw, formula_str, family = task
    c = _COND_CTX
    try:
        f_full = factor_full_values(lib, fid, formula_str, c["frame"], c["ev"],
                                    c["builder"], c["specs"], c["store_full"])
        if f_full is None:
            return None
        per = {}
        for lab, ctx in c["ctxs"].items():
            per[lab] = c["cond_ic_metrics"](ctx, f_full[ctx._pos], c["cond_prep"],
                                            name=fid)
        return {'name': f"{fid} ({family})", 'formula': formula_raw,
                'lib': lib, 'per': per}
    except Exception:
        return None


#: dom 诊断并行共享（fork 继承 val_store/frame/字段值，免 pickling）
_DOM_CTX = None


def _quick_dom_core(v: np.ndarray, fields_vals: dict):
    """单候选单字段支配度（与串行 _quick_dom 逐字同口径）。"""
    from scipy.stats import spearmanr as _sp
    out = {}
    for f, fv in fields_vals.items():
        ok = np.isfinite(v) & np.isfinite(fv)
        if ok.sum() < 30:
            out[f] = np.nan
            continue
        rho, _ = _sp(v[ok], fv[ok])
        out[f] = abs(rho)
    mf = max(out, key=lambda k: out[k] if np.isfinite(out[k]) else -1.0)
    if not np.isfinite(out[mf]):
        return None
    return mf, out[mf]


def _dom_worker(task):
    """dom 诊断工作进程：单候选（串并同一份求值代码，逐值一致）。"""
    _key, lib, fid, fcn = task
    c = _DOM_CTX
    try:
        _v = c["val_store"].get(_key)
        if _v is None:
            if lib == 'store' and c["store_vals"] and fid in c["store_vals"]:
                _v = c["store_vals"][fid]
            else:
                _full = factor_full_values(lib, fid, fcn, c["frame"], c["ev"],
                                           c["builder"], c["specs"], None)
                if _full is None:
                    return _key, None
                _v = np.asarray(_full, dtype=float)[c["pos_me"]]
        _dd = _quick_dom_core(np.asarray(_v, dtype=float), c["fields_vals"])
        return _key, _dd
    except Exception:
        return _key, None


def _ref_spec_key(spec: dict) -> str:
    raw = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)
    return 'ref_' + str(spec.get('factor_id'))[:24] + '_' \
        + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]


def _value_cache_dir(frame: pd.DataFrame, pos_me: np.ndarray) -> Path:
    """因子值缓存目录（2026-09-26）：面板指纹 = 源 parquet 身份/大小/mtime + 面板行数 + 月末行数。

    面板（load_panel 读取的 stock_features_*.parquet）一旦更新——新增交易日、复权价变化、
    股票池变化——指纹就变化，旧目录不再命中，自动重算，避免用旧数据算出的因子值排榜。
    """
    src = find_stock_features()
    st = src.stat()
    fp = hashlib.sha1(f'{src.name}|{st.st_size}|{st.st_mtime_ns}|{len(frame)}|'
                      f'{len(pos_me)}'.encode()).hexdigest()[:16]
    return OUT_DIR / 'value_cache' / fp


def _cache_load(cache_dir: Path | None, key: str, n: int) -> np.ndarray | None:
    if cache_dir is None:
        return None
    p = cache_dir / f'{key}.npy'
    if not p.exists():
        return None
    try:
        v = np.load(p)
    except Exception:
        return None
    if v.ndim != 1 or v.shape[0] != n:
        return None
    return v


def _cache_save(cache_dir: Path | None, key: str, v: np.ndarray) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f'{key}.npy'
        tmp = p.with_name(p.name + '.tmp')
        with tmp.open('wb') as f:
            np.save(f, np.asarray(v, dtype=np.float32))
        tmp.replace(p)
    except Exception:
        pass


#: twin 扫描值计算工作进程共享（fork 继承，免 pickling 大面板）
_SCAN_CTX = None


def _scan_value_key(r, specs_by_id):
    """单行分支判定 + 缓存键（纯查表/解析，不求值；与循环体内联逻辑逐字一致）。
    返回 (branch, key)：branch ∈ mined/custom/store/refs；
    解析失败或查表缺失返回 (None, None)（调用方跳过，与原 continue 一致）。"""
    try:
        if r['lib'] == 'mined':
            node = parse_formula(str(r['formula_cn']))
            return "mined", expression_hash(node)
        if r['lib'] == 'custom':
            from custom_registry import load_all_custom_specs as _lcs
            _spec = _lcs().get(str(r['factor_id']))
            if _spec is None:
                return None, None
            return "custom", 'custom_' + _spec.spec_hash
        if r['lib'] == 'store':
            return "store", 'store_' + str(r['factor_id'])
        spec = specs_by_id.get(str(r['factor_id']))
        if spec is None:
            return None, None
        return "refs", _ref_spec_key(spec)
    except Exception:
        return None, None


def _scan_value_worker(args) -> tuple:
    """扫描值工作进程：(序号, branch, lib, fid, 公式) →
    (序号, float32数组或None)。串行回退直接调本函数，串并同一份求值代码。
    store 分支不走 worker（值已在内存 store_vals，主进程直填）。"""
    idx, branch, lib, fid, formula = args
    _c = _SCAN_CTX
    try:
        if branch == "mined":
            node = parse_formula(str(formula))
            v = np.asarray(_c["ev"].values(node), dtype=float)[_c["pos_me"]]
        elif branch == "custom":
            from custom_registry import evaluate_custom_factor as _ecf
            from custom_registry import load_all_custom_specs as _lcs
            _spec = _lcs().get(str(fid))
            if _spec is None:
                return idx, None
            v = np.asarray(_ecf(_spec, _c["frame"], _c["ev"]),
                           dtype=float)[_c["pos_me"]]
        else:
            spec = _c["specs"].get(str(fid))
            if spec is None:
                return idx, None
            v = _c["builder"].get_factor(spec).to_numpy(
                dtype=float)[_c["pos_me"]]
        return idx, np.asarray(v, dtype=np.float32)
    except Exception:
        return idx, None


def vwap_orthogonal_indices(df_vwap: pd.DataFrame, frame: pd.DataFrame, ev: Evaluator,
                            builder: FactorBuilder, pos_me: np.ndarray,
                            day_group: np.ndarray, specs_by_id: dict,
                            max_scan: int = 150, keep: int = 20,
                            threshold: float = CORR_THRESHOLD,
                            cache_dir: Path | None = None,
                            val_store: dict | None = None,
                            stats: dict | None = None,
                            label: str = '',
                            y_me: np.ndarray | None = None,
                            store_vals: dict | None = None,
                            twin_thr: float = 0.95,
                            jobs: int = 1) -> tuple[list[int], dict, dict]:
    """榜2 排序层轻量正交（2026-09-24）：按 LGB 降序扫描前 max_scan 个候选，
    逐日截面 |Spearman|>threshold 的只留 LGB 最优者（与 decorrelate_top 同判据，
    但在榜单生成时即生效，decorrelate_top 仍可独立复核）。

    2026-09-26 加速：候选在 pos_me（逐日截面）的因子值经 cache_dir 落盘复用
    （key=表达式哈希，面板指纹目录），val_store 回传本次已算值供 dom 复用。

    2026-09-26 非线性诊断：label=='vwap' 且 y_me 给定时，对全部成功求值的
    扫描候选算月度 MI / 尾部 JSD（c2c 不动）。

    返回 (保留的 df 行号顺序, {df行号: 与已选者的最大|ρ|},
    {df索引标签: {'mi','mi_std','nmi','n_months','jsd'}})；值不可得的候选跳过。

    jobs：值计算阶段并行进程数（缓存命中/缺失判定串行，未命中经进程池并行；
    合并后按行序还原，twin 判定与串行逐值一致；MI/JSD 与 twin 判定保持串行）。
    """
    cols, idxs = [], []
    # Phase A（串行）：分支判定 + 缓存命中直填；store 走内存直填；
    # 未命中排队。行序在 Phase B 合并后还原，twin 优先级顺序不变。
    _done: list = []  # (i_row, key, r, v)
    _miss: list = []  # (i_row, key, r, branch)
    for i_row, (_, r) in enumerate(df_vwap.head(max_scan).iterrows()):
        key = None
        try:
            branch, key = _scan_value_key(r, specs_by_id)
            if branch is None:
                continue
            if branch == "store":
                # V4.3c store 源：值来自预对齐的存量面板（不走 specs/builder）。
                try:
                    if store_vals is None or str(r['factor_id']) not in store_vals:
                        continue
                    v = np.asarray(store_vals[str(r['factor_id'])],
                                   dtype=float).astype(np.float32)
                    if len(v) != len(pos_me):
                        continue
                except Exception:
                    continue
                _done.append((i_row, key, r, v))
                continue
            v = _cache_load(cache_dir, key, len(pos_me))
            if v is None:
                _miss.append((i_row, key, r, branch))
                continue
            if stats is not None:
                stats['hit'] = stats.get('hit', 0) + 1
            _done.append((i_row, key, r, v))
        except Exception:
            continue
    # Phase B：未命中求值（串行回退直接调 worker，串并同一份代码；
    # map 有序，合并后按 i_row 还原行序，twin 优先级与原循环一致）。
    _nj = _par_jobs(jobs)
    _got: list = []  # (i_row, key, r, v-or-None)
    global _SCAN_CTX
    if _miss:
        if _nj > 1 and len(_miss) > 1:
            import concurrent.futures as _cf
            _SCAN_CTX = {"frame": frame, "ev": ev, "builder": builder,
                         "specs": specs_by_id, "pos_me": pos_me}
            _nj = min(_nj, len(_miss))
            print(f'正交扫描并行求值：{len(_miss)} 个 × {_nj} 进程 …', flush=True)
            with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
                for _k, _vv in _ex.map(
                        _scan_value_worker,
                        [(k, _br, str(_rr['lib']), str(_rr['factor_id']),
                          str(_rr.get('formula_cn', '')))
                         for k, (_ii, _kk, _rr, _br) in enumerate(_miss)]):
                    _ii, _kk, _rr, _br = _miss[_k]
                    _got.append((_ii, _kk, _rr, _vv))
        else:
            _SCAN_CTX = {"frame": frame, "ev": ev, "builder": builder,
                         "specs": specs_by_id, "pos_me": pos_me}
            for _k, (_ii, _kk, _rr, _br) in enumerate(_miss):
                _, _vv = _scan_value_worker(
                    (_k, _br, str(_rr['lib']), str(_rr['factor_id']),
                     str(_rr.get('formula_cn', ''))))
                _got.append((_ii, _kk, _rr, _vv))
        for (_ii, _kk, _rr, _vv) in _got:
            if _vv is None:
                continue
            _cache_save(cache_dir, _kk, _vv)
            if stats is not None:
                stats['miss'] = stats.get('miss', 0) + 1
            _done.append((_ii, _kk, _rr, _vv))
    _done.sort(key=lambda x: x[0])
    for (_i_row, _key, _rr, _vv) in _done:
        cols.append(_vv.astype(np.float32))
        idxs.append(_i_row)
        if val_store is not None and _key is not None:
            val_store[(_rr['lib'], str(_rr['factor_id']))] = _vv
    # 非线性诊断（2026-09-26）：只对 vwap 扫描候选算 MI/JSD（c2c 不动）；
    # 用已求值（cols），不新增因子求值。2026-10-19 并行化（_nl_backfill）。
    nl_map: dict = {}
    if label == 'vwap' and y_me is not None:
        nl_map = _nl_backfill(cols, idxs, list(df_vwap.index),
                              np.asarray(y_me, dtype=float), day_group, jobs)
    if not cols:
        return [], {}, nl_map
    corr = month_end_spearman_matrix(np.column_stack(cols), day_group)
    if corr is None:
        return idxs[:keep], {}, nl_map
    # 孪生公平单遍（串味修复后的公平版）：按 LGB 降序单遍扫描，谁分高谁先审；
    # 非 store 候选若与任一已扫描 store 孪生（|ρ|>=twin_thr，即已入库列的
    # 数学孪生体），由 store 列顶替入池（能进才进）或直接丢弃——低分 store
    # 永不抢占高分非 store 席位；<0.95 同机制变体按 LGB 与阈值公平竞争。
    _is_store = []
    for _j in range(len(idxs)):
        try:
            _is_store.append(str(df_vwap.iloc[idxs[_j]]['lib']) == 'store')
        except Exception:
            _is_store.append(False)
    _store_js = [x for x in range(len(idxs)) if _is_store[x]]
    kept, rho_map, twin_of = [], {}, {}

    def _mx_vs(js, ks):
        if not ks:
            return float('nan')
        _p = np.abs(corr[js, ks])
        _f = _p[np.isfinite(_p)]
        return float(_f.max()) if _f.size else float('nan')

    for j in range(len(idxs)):
        if len(kept) >= keep:
            break
        if j in kept:
            # 前面被孪生顶替提前入池的 store，再次走到时跳过（防自相关 1.0）。
            continue
        if _is_store[j]:
            if not kept:
                kept.append(j)
                continue
            mx = _mx_vs(j, kept)
            rho_map[idxs[j]] = mx
            if np.isfinite(mx) and mx <= threshold:
                kept.append(j)
            continue
        # 非 store：先查孪生（对全部已扫描 store，不限于已选；含 kept 为空的首名）。
        _sj, _smx = -1, float('nan')
        for _s in _store_js:
            _c = corr[j, _s]
            if np.isfinite(_c) and abs(float(_c)) > (_smx if np.isfinite(_smx) else -1.0):
                _smx, _sj = abs(float(_c)), _s
        if _sj >= 0 and np.isfinite(_smx) and _smx >= twin_thr:
            twin_of[idxs[j]] = {'twin': idxs[_sj], 'rho': _smx}
            if _sj in kept:
                rho_map[idxs[j]] = _smx
                continue
            _ms = _mx_vs(_sj, kept) if kept else float('nan')
            if (not kept) or (np.isfinite(_ms) and _ms <= threshold):
                kept.append(_sj)
                rho_map[idxs[_sj]] = _ms
            else:
                rho_map[idxs[j]] = _smx
            continue
        if not kept:
            kept.append(j)
            continue
        mx = _mx_vs(j, kept)
        rho_map[idxs[j]] = mx
        if np.isfinite(mx) and mx <= threshold:
            kept.append(j)
    return [idxs[j] for j in kept], rho_map, nl_map


def load_refs_rows() -> pd.DataFrame:
    files = sorted(OUT_DIR.glob('refs_screen_*_flat.csv'))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['lib', 'factor_id', 'label'], keep='last')
    print(f'refs rows={len(df):,} from {len(files)} flat files', flush=True)
    return df


def mined_prefilter(top: int, habitat_top: int = 50,
                    data_cut: str | None = None) -> tuple[list, set]:
    ledger = Ledger()
    # 串味修复：只读本 cut 事件；旧 cut（2026-08-21）full-window 免算指标
    # 不得混入本 cut 榜单（须重评，不走 bucket 捷径）。
    rows = _ledger_candidates(ledger, data_cut=data_cut)
    print(f'ledger candidates: {len(rows)} (cut={data_cut})', flush=True)
    picks: list[str] = []
    for lab in ('c2c', 'vwap'):
        cand = []
        for h, b in rows.items():
            m = b['labels'].get(lab)
            if not m or m.get('sia') is None or not np.isfinite(m['sia']):
                continue
            cand.append((float(m['sia']), h))
        cand.sort(key=lambda x: -x[0])
        picks.extend(h for _, h in cand[:top])
    # 断路1修复：habitat 豁免因子（线性 sia≈0 但 habitat_score 高）并集捞入，
    # 否则 Class2 在 build_top 入口即被 sia 排序杀光。配额 habitat_top（默认50）。
    try:
        hcand = []
        for h, b in rows.items():
            _hs = [(b['labels'].get(lab) or {}).get('habitat_score') for lab in ('c2c', 'vwap')]
            _hs = [float(x) for x in _hs if x is not None and np.isfinite(x)]
            if _hs:
                hcand.append((max(_hs), h))
        hcand.sort(key=lambda x: -x[0])
        picks.extend(h for _, h in hcand[:max(0, habitat_top)])
        print(f'habitat union: +{min(max(0, habitat_top), len(hcand))} (pool {len(hcand)})',
              flush=True)
    except Exception as exc:
        print(f'warning: habitat union 跳过 {str(exc)[:100]}', flush=True)
    uniq = list(dict.fromkeys(picks))
    # habitat 并集哈希集合（供下游 hab 扩面优先排序，避免 400 名截断漏杀）
    _hab_h = set()
    try:
        _hc = []
        for h, b in rows.items():
            _hs = [(b['labels'].get(lab) or {}).get('habitat_score')
                   for lab in ('c2c', 'vwap')]
            _hs = [float(x) for x in _hs if x is not None and np.isfinite(x)]
            if _hs:
                _hc.append((max(_hs), h))
        _hc.sort(key=lambda x: -x[0])
        _hab_h = {h for _, h in _hc[:max(0, habitat_top)]}
    except Exception:
        pass
    out = []
    for h in uniq:
        b = rows[h]
        m_c = b['labels'].get('c2c') or {}
        m_v = b['labels'].get('vwap') or {}
        sia_c = m_c.get('sia')
        sia_v = m_v.get('sia')
        best = max([s for s in (sia_c, sia_v) if s is not None and np.isfinite(s)],
                   default=0.0)
        out.append({'hash': h, 'formula': b['formula'], 'family': b.get('family') or 'x',
                    'old_sia': best, 'habitat_pick': bool(h in _hab_h)})
    out.sort(key=lambda r: -r['old_sia'])
    # 加速（2026-09-26）：携带 bucket 全量数据，让 eval_mined 能直接复用台账指标
    for r in out:
        r['bucket'] = rows[r['hash']]
    print(f'mined pick union={len(out)}', flush=True)
    return out, _hab_h


def eval_custom(frame: pd.DataFrame | None = None,
                ev: Evaluator | None = None,
                end: str | None = None) -> pd.DataFrame:
    """V4.1三源合一之custom源：评估`custom_registry`全部已注册自定义因子。

    与`eval_mined`同口径（双标签DataContext全窗重估，LGB按`refs.lgb_score T_ref=0.3`），
    结果记台账`local`事件`source='custom'`，供正交中枢统一去重。失败/覆盖不足记valid=False。
    """
    from custom_registry import evaluate_custom_factor, load_all_custom_specs
    ledger = Ledger()
    if frame is None:
        frame = load_panel()
    if ev is None:
        ev = Evaluator(frame, cache_size=32)
    specs = load_all_custom_specs()
    print(f'custom specs={len(specs)}', flush=True)
    ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev, end=(end or W.DEFAULT_DATA_CUT))
            for lab in ('c2c', 'vwap')}
    rows = []
    for fid, spec in sorted(specs.items()):
        try:
            f_full = np.asarray(evaluate_custom_factor(spec, frame, ev), dtype=float)
        except Exception:
            continue
        rec = {'lib': 'custom', 'factor_id': fid, 'name_cn': spec.description or fid,
               'base': None, 'transform': None, 'window': None,
               'formula_cn': spec.formula_str, 'uploaded': '', 'family': spec.family,
               'labels': {}}
        ok_any = False
        for lab, ctx in ctxs.items():
            v = f_full[ctx._pos]
            cov = float(np.isfinite(v).mean()) if len(v) else 0.0
            if cov <= 0.1:
                rec['labels'][lab] = {'valid': False, 'coverage': cov}
                continue
            m = ctx.screen(v, name=fid)
            t = tier1(m)
            t['coverage'] = cov
            a_off = float(sia_of(m))
            lgb = lgb_score(m)
            t['direction'] = int(m.get('direction', 1))
            rec['labels'][lab] = {'valid': True, 'coverage': cov, 'A_official': a_off,
                                  'LGB': lgb, **t}
            try:
                from parser import parse_formula as _pf
                node = _pf(spec.formula_str) if spec.kind == 'formula' else None
            except Exception:
                node = None
            if node is not None:
                ledger.log('local', node, int(m.get('direction', 1)), family=spec.family,
                           label=lab, data={'metrics': m, 'tier1': t, 'source': 'custom',
                                            'custom_id': fid, 'spec_hash': spec.spec_hash})
            ok_any = True
        if ok_any:
            rows.append(rec)
    flat = []
    for rec in rows:
        for lab in ('c2c', 'vwap'):
            r = rec['labels'].get(lab) or {}
            if not r.get('valid'):
                continue
            flat.append({'lib': rec['lib'], 'factor_id': rec['factor_id'],
                         'name_cn': rec['name_cn'], 'base': rec['base'],
                         'transform': rec['transform'], 'window': rec['window'],
                         'formula_cn': rec['formula_cn'], 'uploaded': rec['uploaded'],
                         'family': rec.get('family', ''),
                         'habitat_pick': bool(rec.get('habitat_pick', False)),
                         'label': lab, **{k: r.get(k) for k in
                                          ('coverage', 'A_official', 'LGB', 'direction')},
                         **{k: r.get(k) for k in
                            ('sia', 'rank_ic', 'ic', 'ic_ir', 'win', 'rank_ic_ir',
                             'rank_ic_win', 'turnover', 'turnover_month', 'net', 'long',
                             'sharpe', 'mdd', 'mono', 'n_valid_periods')}})
    return pd.DataFrame(flat)


def _store_aligned(frame: pd.DataFrame, cut: str | None = None):
    """存量列对齐到 frame 行：返回 (m, pos)，m 含 store 列，pos[i] 为 frame 第 i 行
    在 m 中的行号（-1=缺失）。与 mkt_sector_test 同映射模式。
    cut 为 data_cut 上界（默认 frame 最大日；传 cut 以遵守铁律）。"""
    import boundary as B
    from store_sync import store_columns
    cols = store_columns()
    if not cols:
        return None, None
    _dmax = str(cut) if cut else str(pd.to_datetime(frame["date"]).max().date())
    fac = B.read_factors(cols, data_max=_dmax, label="store-eval")
    fac["date"] = pd.to_datetime(fac["date"])
    m = fac.merge(frame[["stock_code", "date"]].drop_duplicates(),
                  on=["stock_code", "date"], how="inner")
    del fac
    key = m[["stock_code", "date"]].copy()
    key["_k"] = key["stock_code"].astype(str) + "|" + key["date"].astype(str)
    fmap = {k: i for i, k in enumerate(key["_k"].to_numpy())}
    fkk = (frame["stock_code"].astype(str) + "|"
           + pd.to_datetime(frame["date"]).astype(str)).to_numpy()
    pos = np.array([fmap.get(k, -1) for k in fkk], dtype=np.int64)
    return m, pos


#: eval_store 工作进程共享（fork 继承；store 列评估不写台账，无需写锁）
_ES_CTX = None


def _eval_store_one(fid: str, f_full: np.ndarray, frame_len: int,
                    pos: np.ndarray, ctxs: dict):
    """单 store 列双标签评估（串行/并行共用；纯函数，无台账写）。
    返回 rec（ok_any 为假时返回 None，与原循环的 append 条件一致）。
    异常直接抛出（与原串行循环一致：坏列即中断，便于暴露数据问题）。"""
    from fitness import sia_of, tier1
    from refs import lgb_score
    rec = {'lib': 'store', 'factor_id': fid, 'name_cn': fid,
           'base': None, 'transform': None, 'window': None,
           'formula_cn': f'store:{fid}', 'uploaded': '', 'family': 'store',
           'labels': {}}
    ok_any = False
    for lab, ctx in ctxs.items():
        v = np.full(frame_len, np.nan)
        v[pos >= 0] = f_full[pos[pos >= 0]]
        v = v[ctx._pos]
        cov = float(np.isfinite(v).mean()) if len(v) else 0.0
        if cov <= 0.1:
            rec['labels'][lab] = {'valid': False, 'coverage': cov}
            continue
        sc = ctx.screen(v, name=fid)
        t = tier1(sc)
        t['coverage'] = cov
        a_off = float(sia_of(sc))
        lgb = lgb_score(sc)
        t['direction'] = int(sc.get('direction', 1))
        rec['labels'][lab] = {'valid': True, 'coverage': cov, 'A_official': a_off,
                              'LGB': lgb, **t}
        ok_any = True
    return rec if ok_any else None


def _eval_store_worker(fid: str):
    """store 列工作进程：fid → rec-or-None。共享经 fork 继承。"""
    _ctx = _ES_CTX
    f_full = _ctx["m"][fid].to_numpy(dtype=float)
    return _eval_store_one(fid, f_full, _ctx["frame_len"], _ctx["pos"],
                           _ctx["ctxs"])


def eval_store(frame: pd.DataFrame | None = None,
                 ev: Evaluator | None = None,
                 end: str | None = None,
                 _aligned=None, jobs: int = 1) -> pd.DataFrame:
    """V4.3c 第四源 store：评估 factors.parquet 存量个股列.

    口径与 eval_mined/eval_custom 完全一致（双标签 DataContext 全窗重估，
    LGB 按 refs.lgb_score T_ref=0.3），lib='store'。存量列即 ground truth，
    不写台账（防生产列污染搜索台账）；防重复挖矿由 store_vfps 快评同步承担，
    防重复入选由正交扫描孪生优先（store 优先）承担。
    范围：roles.csv role in (STOCK, STOCK_REL, CROSS_SEC) ∩ 文件实际列。
    _aligned：main 已调过 _store_aligned 时传入 (m, pos)，避免重复读表。
    jobs：列级并行进程数（各列独立、无台账写；结果与串行逐值一致）。
    """
    if frame is None:
        frame = load_panel()
    if ev is None:
        ev = Evaluator(frame, cache_size=32)
    m, pos = _aligned if _aligned is not None else _store_aligned(frame)
    if m is None:
        return pd.DataFrame()
    cols = [c for c in m.columns if c not in ("stock_code", "date")]
    print(f'store columns={len(cols)}', flush=True)
    ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev, end=(end or W.DEFAULT_DATA_CUT))
            for lab in ('c2c', 'vwap')}
    rows = []
    _nj = _par_jobs(jobs)
    if _nj <= 1 or len(cols) <= 1:
        for fid in cols:
            if fid not in m.columns:
                continue
            f_full = m[fid].to_numpy(dtype=float)
            rec = _eval_store_one(fid, f_full, len(frame), pos, ctxs)
            if rec is not None:
                rows.append(rec)
    else:
        import concurrent.futures as _cf
        global _ES_CTX
        _ES_CTX = {"m": m, "pos": pos, "ctxs": ctxs, "frame_len": len(frame)}
        _nj = min(_nj, len(cols))
        print(f'store 并行评估：{len(cols)} 列 × {_nj} 进程 …', flush=True)
        with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
            for _rec in _ex.map(_eval_store_worker, cols):
                if _rec is not None:
                    rows.append(_rec)
    flat = []
    for rec in rows:
        for lab in ('c2c', 'vwap'):
            r = rec['labels'].get(lab) or {}
            if not r.get('valid'):
                continue
            flat.append({'lib': rec['lib'], 'factor_id': rec['factor_id'],
                         'name_cn': rec['name_cn'], 'base': rec['base'],
                         'transform': rec['transform'], 'window': rec['window'],
                         'formula_cn': rec['formula_cn'], 'uploaded': rec['uploaded'],
                         'family': rec.get('family', ''),
                         'habitat_pick': bool(rec.get('habitat_pick', False)),
                         'label': lab, **{k: r.get(k) for k in
                                          ('coverage', 'A_official', 'LGB', 'direction')},
                         **{k: r.get(k) for k in
                            ('sia', 'rank_ic', 'ic', 'ic_ir', 'win', 'rank_ic_ir',
                             'rank_ic_win', 'turnover', 'turnover_month', 'net', 'long',
                             'sharpe', 'mdd', 'mono', 'n_valid_periods')}})
    return pd.DataFrame(flat)


def _eval_mined_one(p: dict, ctxs: dict, ev: Evaluator, ledger: Ledger):
    """单个 pick 慢路径求值（串行/并行共用；ledger 在并行时经进程锁直写）。
    返回 (rec_or_None, n_slow_inc)。"""
    try:
        node = parse_formula(p['formula'])
    except Exception:
        return None, 0
    try:
        f_full = ev.values(node)
    except Exception:
        return None, 0
    name = f"{p['family']}:{p['hash'][:10]}"
    rec = {'lib': 'mined', 'factor_id': p['hash'][:12], 'name_cn': p['formula'],
           'base': None, 'transform': None, 'window': None,
           'formula_cn': p['formula'], 'uploaded': '', 'family': p['family'],
           'labels': {}}
    ok_any = False
    n_inc = 0
    for lab, ctx in ctxs.items():
        v = np.asarray(f_full, dtype=float)[ctx._pos]
        cov = float(np.isfinite(v).mean()) if len(v) else 0.0
        if cov <= 0.1:
            rec['labels'][lab] = {'valid': False, 'coverage': cov}
            continue
        m = ctx.screen(v, name=name)
        t = tier1(m)
        t['coverage'] = cov
        a_off = float(sia_of(m))
        lgb = lgb_score(m)
        t['direction'] = int(m.get('direction', 1))
        rec['labels'][lab] = {'valid': True, 'coverage': cov, 'A_official': a_off,
                              'LGB': lgb, **t}
        ledger.log('local', node, int(m.get('direction', 1)), family=p['family'],
                   label=lab, data={'metrics': m, 'tier1': t,
                                    'source': 'aligned-top'})
        ok_any = True
        n_inc += 1
    if ok_any:
        rec['habitat_pick'] = bool(p.get('habitat_pick', False))
        return rec, n_inc
    return None, 0


def _eval_mined_worker(chunk) -> tuple:
    """慢路径工作进程：picks 子集 → ([(原序号, rec)], n_slow)。
    frame/end 经 fork 继承；自建小缓存 Evaluator（省 26GB）+ ctx；ledger 直写经锁。"""
    from evaluator import Evaluator
    from fitness import DataContext
    from ledger import Ledger
    frame, end = _MP_FRAME, _MP_END
    ev = Evaluator(frame, cache_size=32)
    ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev, end=end)
            for lab in ('c2c', 'vwap')}
    ledger = Ledger()
    out, ns = [], 0
    for idx, p in chunk:
        try:
            rec, inc = _eval_mined_one(p, ctxs, ev, ledger)
        except Exception:
            continue
        if rec is not None:
            out.append((idx, rec))
            ns += inc
    return out, ns


def eval_mined(picks: list[dict], frame: pd.DataFrame | None = None,
                ev: Evaluator | None = None,
                builder: FactorBuilder | None = None,
                end: str | None = None, jobs: int = 1) -> pd.DataFrame:
    ledger = Ledger()
    if frame is None:
        frame = load_panel()
    print(f'panel rows={len(frame):,}', flush=True)
    if ev is None:
        ev = Evaluator(frame, cache_size=32)
    if builder is None:
        builder = FactorBuilder(frame)
    ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev, end=(end or W.DEFAULT_DATA_CUT))
            for lab in ('c2c', 'vwap')}
    # 快路径串行（纯字典运算无 ledger 写）→ 慢候选收集
    tmp: list = []
    slow: list = []
    t0 = time.time()
    n_fast = 0
    for i, p in enumerate(picks, 1):
        # 加速（2026-09-26）：台账已有 full-window 双标签指标的直接复用，不重评
        b = p.get('bucket') or {}
        if b.get('window') == 'full' and all(
                (b.get('labels', {}).get(lab) or {}).get('sia') is not None
                for lab in ('c2c', 'vwap')):
            rec = {'lib': 'mined', 'factor_id': p['hash'][:12], 'name_cn': p['formula'],
                   'base': None, 'transform': None, 'window': None,
                   'formula_cn': p['formula'], 'uploaded': '', 'family': p['family'],
                   'labels': {}}
            for lab in ('c2c', 'vwap'):
                m = dict(b['labels'][lab])
                a_off = abs(m.get('rank_ic') or 0) * abs(m.get('ic_ir') or 0) * (m.get('win') or 0)
                turn = m.get('turnover')
                pen = 0.3 / max(float(turn), 0.3) if turn is not None and np.isfinite(turn) else 1.0
                mono = abs(float(m.get('mono') or 0)) if m.get('mono') is not None else 1.0
                lgb = abs(m.get('rank_ic') or 0) * abs(m.get('rank_ic_ir') or 0) \
                    * (m.get('rank_ic_win') or 0) * mono * pen
                m['direction'] = int(m['direction']) if m.get('direction') is not None \
                    else int(b.get('direction', 1))
                rec['labels'][lab] = {'valid': True, 'coverage': np.nan,
                                     'A_official': a_off, 'LGB': lgb, **m}
            rec['habitat_pick'] = bool(p.get('habitat_pick', False))
            tmp.append((i, rec))
            n_fast += 1
            continue
        slow.append((i, p))
    print(f'  fast-path: {n_fast} 复用，{len(slow)} 需现算 {time.time() - t0:.0f}s', flush=True)
    n_slow = 0
    nj = _par_jobs(jobs)
    if nj <= 1 or len(slow) <= 1:
        for k, (i, p) in enumerate(slow, 1):
            rec, inc = _eval_mined_one(p, ctxs, ev, ledger)
            if rec is not None:
                tmp.append((i, rec))
                n_slow += inc
            if k % 10 == 0 or k == len(slow):
                print(f'  [{k}/{len(slow)}] slow={n_slow} {time.time() - t0:.0f}s', flush=True)
    else:
        import concurrent.futures as _cf
        import multiprocessing as _mp
        global _MP_FRAME, _MP_END
        _MP_FRAME, _MP_END = frame, (end or W.DEFAULT_DATA_CUT)
        _nj = min(nj, len(slow))
        _chunks = [slow[i::_nj] for i in range(_nj)]
        _lock = _mp.Lock()
        _done = 0
        with _cf.ProcessPoolExecutor(max_workers=_nj, initializer=_par_init,
                                      initargs=(_lock,)) as _ex:
            for _out, _ns in _ex.map(_eval_mined_worker, _chunks):
                tmp.extend(_out)
                n_slow += _ns
                _done += 1
                print(f'  [chunk {_done}/{len(_chunks)}] slow={n_slow} {time.time() - t0:.0f}s',
                      flush=True)
    tmp.sort(key=lambda x: x[0])
    rows = [r for _, r in tmp]
    flat = []
    for rec in rows:
        for lab in ('c2c', 'vwap'):
            r = rec['labels'].get(lab) or {}
            if not r.get('valid'):
                continue
            flat.append({'lib': rec['lib'], 'factor_id': rec['factor_id'],
                         'name_cn': rec['name_cn'], 'base': rec['base'],
                         'transform': rec['transform'], 'window': rec['window'],
                         'formula_cn': rec['formula_cn'], 'uploaded': rec['uploaded'],
                         'family': rec.get('family', ''),
                         'habitat_pick': bool(rec.get('habitat_pick', False)),
                         'label': lab, **{k: r.get(k) for k in
                                          ('coverage', 'A_official', 'LGB', 'direction')},
                         **{k: r.get(k) for k in
                            ('sia', 'rank_ic', 'ic', 'ic_ir', 'win', 'rank_ic_ir',
                             'rank_ic_win', 'turnover', 'turnover_month', 'net', 'long',
                             'sharpe', 'mdd', 'mono', 'n_valid_periods')}})
    return pd.DataFrame(flat)


def _xsec_spearman(a: np.ndarray, b: np.ndarray, day_codes: np.ndarray,
                   nd: int, min_n: int = 30) -> float:
    """窗内逐日截面 Spearman 均值（与正交扫描 month_end_spearman_matrix 同口径；
    禁止跨日期股票混合 corrcoef）。"""
    import fastranks as FR
    import eval_v2 as E
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float('nan')
    ra = FR.blockwise_rank(np.where(m, a, np.nan), day_codes)
    rb = FR.blockwise_rank(np.where(m, b, np.nan), day_codes)
    ic = E._daily_corr(ra, rb, day_codes, nd)
    ic = ic[np.isfinite(ic)]
    return float(ic.mean()) if len(ic) else float('nan')


#: hab 扩面工作进程共享（fork 继承，免 pickling 大面板）
_HABX_CTX = None
#: 扩面 worker 进程内 custom 注册表懒加载（进程级单例，免每任务重复读表）
_LCSX_CACHE = None


def _habx_key(lib: str, fid: str, formula_cn: str, specs: dict | None):
    """扩面值缓存键（与 twin 扫描同键，保证跨阶段命中一致）。
    mined→表达式哈希；custom→'custom_'+spec_hash；refs→_ref_spec_key；
    store 不走缓存（值已在内存 store_vals）；解析/查表失败返回 None（跳过）。"""
    try:
        if lib == "mined":
            return expression_hash(parse_formula(str(formula_cn)))
        if lib == "custom":
            from custom_registry import load_all_custom_specs as _lcs
            _sp = _lcs().get(str(fid))
            return ("custom_" + _sp.spec_hash) if _sp is not None else None
        if lib == "store":
            return None
        _spx = (specs or {}).get(str(fid))
        return _ref_spec_key(_spx) if _spx is not None else None
    except Exception:
        return None


def _habx_worker(args) -> tuple:
    """扩面工作进程：(序号, hab键, lib, fid, 公式, 缓存键) →
    (序号, hab键, 缓存键, float32数组或None)。串行回退直接调本函数，
    故串并同一份求值代码，零漂移。"""
    idx, key, lib, fid, formula_cn, ckey = args
    _c = _HABX_CTX
    try:
        if lib == "mined":
            _vv = np.asarray(_c["ev"].values(parse_formula(str(formula_cn))),
                             dtype=float)[_c["pos_me"]]
        elif lib == "custom":
            global _LCSX_CACHE
            from custom_registry import evaluate_custom_factor as _ecf
            from custom_registry import load_all_custom_specs as _lcs
            if _LCSX_CACHE is None:
                _LCSX_CACHE = _lcs()
            _sp = _LCSX_CACHE.get(str(fid))
            if _sp is None:
                return idx, key, ckey, None
            _vv = np.asarray(_ecf(_sp, _c["frame"], _c["ev"]),
                             dtype=float)[_c["pos_me"]]
        else:
            _spx = _c["specs"].get(str(fid))
            if _spx is None:
                return idx, key, ckey, None
            _vv = _c["builder"].get_factor(_spx).to_numpy(
                dtype=float)[_c["pos_me"]]
        return idx, key, ckey, np.asarray(_vv, dtype=np.float32)
    except Exception:
        return idx, key, ckey, None


def _habitat_expand(vwap_df, val_store: dict, store_vals: dict, ev,
                    builder, frame, pos_me: np.ndarray, cache_dir,
                    specs: dict, jobs: int = 1):
    """habitat 估值扩面（串行/并行共用装配）。
    val_store 只含 LGB 前 scan 名；全表扫描，优先级：habitat_pick（台账豁免）
    > store/机制家族 > 其余；上限 150。
    返回 (hab_vals, extra_keys)：hab_vals 初始为 dict(val_store) 的拷贝；
    extra_keys 为扩面候选键集（seats 漏斗优先排序用）。
    求值经 _habx_worker（串行回退直接调，串并同一份代码）；命中 value_cache
    的直接复用（与 twin 扫描同键）。整体失败时 warning 降级（hab_vals 退化
    为 dict(val_store)），与旧行为一致。"""
    hab_vals: dict = dict(val_store)
    _extra_all: list = []
    try:
        # 扩面优先族：store/机制家族 + custom（用户策划的自定义因子——日内降频/
        # 板块身位等本就是为强势股栖息地设计的，不优先则会被 other 挤过 [:150]
        # 截断、永不进栖息地席位/股性池评估；仅影响候选排序，入选仍须过
        # hot5_q80 |t|>=2.5 或 jsd/反差门槛）。
        _HAB_FAMS = ('context', 'ctx', 'vol', 'M7', 'M14', 'M16', 'M17',
                     'shape', 'skew', 'M3', 'store', 'custom')
        _extra0, _extra1, _extra2 = [], [], []
        for _pos in range(len(vwap_df)):
            _r = vwap_df.iloc[_pos]
            _key = (_r['lib'], str(_r['factor_id']))
            if _key in hab_vals:
                continue
            _fam = str(_r.get('family', ''))
            if bool(_r.get('habitat_pick', False)):
                _extra0.append((_pos, _r))
            elif _r['lib'] == 'store' or any(k in _fam for k in _HAB_FAMS):
                _extra1.append((_pos, _r))
            else:
                _extra2.append((_pos, _r))
        _extra_all = (_extra0 + _extra1 + _extra2)[:150]
        print(f'habitat 扩面候选：pick={len(_extra0)} fam={len(_extra1)} '
              f'other={len(_extra2)} -> 算{len(_extra_all)} 个', flush=True)
        # 预检（串行，纯查表+读缓存）：store 走内存直填；命中 value_cache 的
        # 直接复用（与 twin 扫描同键）；未命中才进求值（串行/并行）。
        _hx_tasks: list = []
        for _pos, _r in _extra_all:
            _key = (_r['lib'], str(_r['factor_id']))
            try:
                if _r['lib'] == 'store':
                    _vv = store_vals.get(str(_r['factor_id']))
                    if _vv is None:
                        continue
                    hab_vals[_key] = np.asarray(_vv, dtype=np.float32)
                    continue
                _ckey = _habx_key(str(_r['lib']), str(_r['factor_id']),
                                  str(_r.get('formula_cn', '')), specs)
                if _ckey is None:
                    continue
                _vhit = _cache_load(cache_dir, _ckey, len(pos_me))
                if _vhit is not None:
                    hab_vals[_key] = _vhit
                    continue
                _hx_tasks.append((_key, str(_r['lib']), str(_r['factor_id']),
                                  str(_r.get('formula_cn', '')), _ckey))
            except Exception:
                continue
        _hx_nj = _par_jobs(jobs)
        _hx_results: list = []
        global _HABX_CTX
        if _hx_tasks and _hx_nj > 1 and len(_hx_tasks) > 1:
            import concurrent.futures as _cf
            _HABX_CTX = {"frame": frame, "ev": ev, "builder": builder,
                         "specs": specs, "pos_me": pos_me}
            _hx_nj = min(_hx_nj, len(_hx_tasks))
            print(f'habitat 扩面并行求值：{len(_hx_tasks)} 个 × {_hx_nj} 进程 …',
                  flush=True)
            with _cf.ProcessPoolExecutor(max_workers=_hx_nj) as _ex:
                for _, _key, _ckey, _vv in _ex.map(
                        _habx_worker,
                        [(k, _key, _lib, _fid, _f, _ck)
                         for k, (_key, _lib, _fid, _f, _ck) in
                         enumerate(_hx_tasks)]):
                    _hx_results.append((_key, _ckey, _vv))
        else:
            _HABX_CTX = {"frame": frame, "ev": ev, "builder": builder,
                         "specs": specs, "pos_me": pos_me}
            for _i, (_key, _lib, _fid, _f, _ckey) in enumerate(_hx_tasks):
                _, _key2, _ckey2, _vv = _habx_worker(
                    (_i, _key, _lib, _fid, _f, _ckey))
                _hx_results.append((_key2, _ckey2, _vv))
        for _key, _ckey, _vv in _hx_results:
            if _vv is None:
                continue
            _cache_save(cache_dir, _ckey, _vv)
            hab_vals[_key] = _vv
        print(f'habitat 估值扩面：+{len(hab_vals) - len(val_store)} 个', flush=True)
    except Exception as exc:
        print(f'warning: habitat 扩面失败 {str(exc)[:120]}', flush=True)
    _extra_keys = set()
    try:
        for _xp, _xr in _extra_all:
            _extra_keys.add((_xr['lib'], str(_xr['factor_id'])))
    except Exception:
        _extra_keys = set()
    return hab_vals, _extra_keys


def _window_shim_ctx(day_group: np.ndarray, y_me: np.ndarray):
    """选择窗 shim ctx：供 cond_ic_metrics 在 pos_me 切片上计算窗内条件 IC。

    与全窗 cond 口径同函数、窗内数据（与 LGB 排序同窗，对齐选择语义）。
    """
    from types import SimpleNamespace
    di, _ = pd.factorize(day_group, sort=True)
    return SimpleNamespace(di=np.asarray(di), _pos=np.arange(len(di)),
                           labels=np.asarray(y_me, dtype=float))


#: seats/pool 打分工作进程共享（fork 继承，免 pickling 大面板字典）
_HABSC_CTX = None


def _hab_score_worker(args) -> tuple:
    """打分工作进程：(序号, key, 因子名, kind) → (序号, 载荷)。
    kind='seat' 返回 (t, ndays)；kind='pool' 返回 (mu, md)（可含 None）。
    共享经 fork 继承；异常返回失败载荷（父进程过滤逻辑与串行一致）。
    父进程侧的阈值/过滤/排序代码串并共用，逐字同一份。"""
    idx, key, fname, kind = args
    _ctx = _HABSC_CTX
    try:
        from cond_ic import cond_ic_metrics
        _arr = np.asarray(_ctx["vals"][key], dtype=float)
        if kind == "seat":
            per = cond_ic_metrics(_ctx["shim"], _arr, _ctx["prep"],
                                  name=fname, tags=("hot5_q80",))
            t = per.get("cond_hot5_q80_rank_ic_tstat")
            nd = per.get("cond_hot5_q80_ndays", 0)
            t = float(t) if t is not None and np.isfinite(t) else float("nan")
            return idx, (t, nd)
        per = cond_ic_metrics(_ctx["shim"], _arr, _ctx["prep"],
                              name=fname, tags=("mkt_up", "mkt_dn"))
        return idx, (per.get("cond_mkt_up_rank_ic_mean"),
                     per.get("cond_mkt_dn_rank_ic_mean"))
    except Exception:
        if kind == "seat":
            return idx, (float("nan"), 0)
        return idx, (None, None)


def _new_pair_cache(v: np.ndarray) -> list:
    """候选/池成员秩缓存单元 [有限掩码, 掩码内有限数, 秩或None]（惰性填秩）。"""
    A = np.isfinite(v)
    return [A, int(A.sum()), None]


def _pair_rho(a: np.ndarray, b: np.ndarray, day: np.ndarray, nd: int,
              cache_a: list | None = None, cache_b: list | None = None,
              min_n: int = 30) -> float:
    """单对逐日截面 Spearman 均值；与 _xsec_spearman 逐位一致。

    P3 加速（2026-10-19，位级等价）：pair 掩码 m=A&B；若 A⊆B（na==|m|，
    预热 NaN 场景恒成立）则 pair 掩码即 A → 复用该列在自身掩码下的秩缓存，
    免去每对重复 blockwise_rank（1.5~5s → ~0.1-0.3s）；不嵌套时回退逐对现算。
    """
    import eval_v2 as E
    import fastranks as FR
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    A = cache_a[0] if cache_a is not None else np.isfinite(a)
    B = cache_b[0] if cache_b is not None else np.isfinite(b)
    m = A & B
    nm = int(m.sum())
    if nm < min_n:
        return float('nan')
    if cache_a is not None and cache_a[1] == nm:
        if cache_a[2] is None:
            cache_a[2] = FR.blockwise_rank(np.where(A, a, np.nan), day) \
                if cache_a[1] else None
        ra = cache_a[2]
    else:
        ra = FR.blockwise_rank(np.where(m, a, np.nan), day)
    if cache_b is not None and cache_b[1] == nm:
        if cache_b[2] is None:
            cache_b[2] = FR.blockwise_rank(np.where(B, b, np.nan), day) \
                if cache_b[1] else None
        rb = cache_b[2]
    else:
        rb = FR.blockwise_rank(np.where(m, b, np.nan), day)
    ic = E._daily_corr(ra, rb, day, nd)
    ic = ic[np.isfinite(ic)]
    return float(ic.mean()) if len(ic) else float('nan')


def _pair_vs_pool(v_s: np.ndarray, pool_s: list, day_s: np.ndarray,
                  nd: int, threshold: float,
                  cache: list | None = None,
                  pool_caches: list | None = None) -> bool:
    """候选（已按 day_s 重排）与池的正交否决（返回 True=拒绝）。
    串行 early-break：首个 |ρ|>threshold 即 return True。大多数落选候选
    在前 1~3 对即淘汰，远快于全量 map 后 any（80 对必满）。
    刻意不用线程/进程池：early-break 语义下全量并行会放大算量 20~40 倍。
    P3（2026-10-19）：pair 计算走 _pair_rho 秩缓存（掩码嵌套时位级等价复用），
    cache/pool_caches 由调用方持有（_new_pair_cache 构造）。
    输入须行序一致（day_s 单调 + 数组同序重排），否则结果无意义。"""
    for _j, _pv in enumerate(pool_s):
        _cb = pool_caches[_j] if pool_caches is not None else None
        _c = _pair_rho(v_s, _pv, day_s, nd, cache_a=cache, cache_b=_cb)
        if np.isfinite(_c) and abs(float(_c)) > threshold:
            return True
    return False


def habitat_seats(vwap_df: pd.DataFrame, vwap_order: list[int] | None,
                  val_store: dict, pos_me: np.ndarray, day_group: np.ndarray,
                  y_vwap_me: np.ndarray, prep_me: dict,
                  day_codes: np.ndarray, nd_me: int,
                  n_seats: int = 10, t_min: float = 2.5,
                  threshold: float = 0.7,
                  extra_keys: set | None = None, jobs: int = 1) -> tuple[list[int], list[dict]]:
    """Class1 栖息地尖兵席位（V4.3c）：扫描已估值但未入选者中 hot5_q80 |t|>=2.5 者，
    按 t 降序、与池内+席位间逐一 |ρ|<=threshold 正交保留，上限 n_seats。

    新选择族版本（预注册阈值 2.5/0.7/配额 10）；快评通道 C 豁免禁令不动
    （仅终选席位，不碰快评）。返回 (df 行号, 明细)。
    漏斗（V4.3x 修复）：habitat_pick > 本轮扩面 > LGB，上限 250——旧
    sorted(-LGB)[:60] 把低无条件 LGB 的栖息地扩面候选全挡在门外。
    加速（零语义变更）：打分经进程池并行（tags 只算 hot5_q80，过滤逐字同
    串行）；行序在内存中一次重排（day_s 单调 + 数组同序副本，pos_me 与
    磁盘缓存不动），pairwise 经线程池并行（判定与串行 early-break 恒等）。
    """
    from cond_ic import cond_ic_metrics
    # vwap_order 为 head 枚举 positional 行号（与 iloc 对齐，非 index 标签）。
    kept = set(vwap_order or [])
    cands = []
    for _pos in range(len(vwap_df)):
        if _pos in kept:
            continue
        r = vwap_df.iloc[_pos]
        _key = (r['lib'], str(r['factor_id']))
        if _key not in val_store:
            continue
        cands.append((_pos, r))
    _exk = extra_keys or set()

    def _ck(x):
        _p2, _r2 = x
        _k2 = (_r2['lib'], str(_r2['factor_id']))
        _pr = (0 if bool(_r2.get('habitat_pick', False))
               else (1 if _k2 in _exk else 2))
        return (_pr, -float(_r2.get('LGB', 0) or 0))

    cands = sorted(cands, key=_ck)[:250]
    shim = _window_shim_ctx(day_group, y_vwap_me)
    scored = []
    _nj = _par_jobs(jobs)
    _use_par = _nj > 1 and len(cands) > 1
    _seat_cache: dict = {}
    if _use_par:
        import concurrent.futures as _cf
        global _HABSC_CTX
        _HABSC_CTX = {"shim": shim, "prep": prep_me, "vals": val_store}
        _nj = min(_nj, len(cands))
        with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
            for _idx, _pay in _ex.map(
                    _hab_score_worker,
                    [(k, (rr['lib'], str(rr['factor_id'])), str(rr['factor_id']),
                      "seat") for k, (_pp, rr) in enumerate(cands)]):
                _seat_cache[_idx] = _pay
    for _k, (_pos, r) in enumerate(cands):
        try:
            if _use_par:
                t, nd = _seat_cache.get(_k, (float("nan"), 0))
            else:
                per = cond_ic_metrics(shim, val_store[(r['lib'], str(r['factor_id']))],
                                      prep_me, name=str(r['factor_id']),
                                      tags=("hot5_q80",))
                t = per.get('cond_hot5_q80_rank_ic_tstat')
                nd = per.get('cond_hot5_q80_ndays', 0)
                t = float(t) if t is not None and np.isfinite(t) else float('nan')
            if np.isfinite(t) and abs(t) >= t_min and nd >= 10:
                scored.append((_pos, r, t))
        except Exception:
            continue
    scored.sort(key=lambda x: -abs(x[2]))
    # 行序一次重排（内存副本；pos_me 与磁盘 value_cache 不动）：day_s 单调后
    # blockwise_rank 守卫直通，省掉 pairwise 近万次 660 万行 argsort。
    # 重排对（值数组, day）一致进行，_daily_corr 行序无关，故结果与碎裂行序
    # 逐值一致（fp 求和顺序差异除外，~1e-12，不影响阈值判定）。
    _order_me = np.argsort(np.asarray(day_codes), kind="stable")
    _day_s = np.asarray(day_codes)[_order_me]
    pool_s = []
    pool_caches = []
    for _kp in (vwap_order or []):
        try:
            _r = vwap_df.iloc[_kp]
            _vv = val_store.get((_r['lib'], str(_r['factor_id'])))
            if _vv is not None:
                _vv_s = np.asarray(_vv, dtype=float)[_order_me]
                pool_s.append(_vv_s)
                pool_caches.append(_new_pair_cache(_vv_s))
        except Exception:
            continue
    seats, detail = [], []
    for _pos, r, t in scored:
        if len(seats) >= n_seats:
            break
        _v_s = np.asarray(val_store[(r['lib'], str(r['factor_id']))],
                          dtype=float)[_order_me]
        _cache_s = _new_pair_cache(_v_s)
        if _pair_vs_pool(_v_s, pool_s, _day_s, nd_me, threshold,
                         cache=_cache_s, pool_caches=pool_caches):
            continue
        seats.append(_pos)
        pool_s.append(_v_s)
        pool_caches.append(_cache_s)
        detail.append({'factor_id': str(r['factor_id']), 'lib': str(r['lib']),
                       'hot5_q80_t': round(float(t), 3)})
    return seats, detail


def habitat_pool(vwap_df: pd.DataFrame, vwap_order: list[int] | None,
                 val_store: dict, pos_me: np.ndarray, day_group: np.ndarray,
                 y_vwap_me: np.ndarray, prep_me: dict, vwap_nl: dict,
                 day_codes: np.ndarray, nd_me: int,
                 top_n: int = 10, threshold: float = 0.7,
                 jobs: int = 1) -> list[dict]:
    """Class2 股性榜单（V4.3c）：已估值扫描候选按 habitat_score 排名，
    与 Class1 池内 + 榜内逐一 |ρ|<=threshold 正交取 TopN。

    habitat_score = jsd_tails + 0.5*|mkt_up_ic - mkt_dn_ic|（精确 cond 版；
    快评用 pooled 代理版）。返回 [{factor_id, lib, score, jsd, contrast}]。
    加速（零语义变更）：cond 经进程池并行（tags 只算 mkt_up/mkt_dn）；
    行序在内存中一次重排（与 seats 同模式，pos_me 与磁盘缓存不动），
    pairwise 经线程池并行（判定与串行 early-break 恒等）。
    """
    from cond_ic import cond_ic_metrics
    from nonlinear import jsd_tails
    shim = _window_shim_ctx(day_group, y_vwap_me)
    _nj = _par_jobs(jobs)
    # 打分键全集（串并同序；并行时先批量算 cond，再走同一串行组装循环）
    _keys: list = []
    _seen = set()
    for _pos in range(len(vwap_df)):
        r = vwap_df.iloc[_pos]
        _key = (r['lib'], str(r['factor_id']))
        if _key in _seen or _key not in val_store:
            continue
        _seen.add(_key)
        _keys.append((_pos, _key))
    _use_par = _nj > 1 and len(_keys) > 1
    _cond_cache: dict = {}
    if _use_par:
        import concurrent.futures as _cf
        global _HABSC_CTX
        _HABSC_CTX = {"shim": shim, "prep": prep_me, "vals": val_store}
        _nj = min(_nj, len(_keys))
        with _cf.ProcessPoolExecutor(max_workers=_nj) as _ex:
            for _idx, _pay in _ex.map(
                    _hab_score_worker,
                    [(k, _key, str(_key[1]), "pool") for k, (_pp, _key) in
                     enumerate(_keys)]):
                _cond_cache[_idx] = _pay
    _order_me = np.argsort(np.asarray(day_codes), kind="stable")
    _day_s = np.asarray(day_codes)[_order_me]
    pool_s = []
    pool_caches = []
    for _kp in (vwap_order or []):
        try:
            _r = vwap_df.iloc[_kp]
            _vv = val_store.get((_r['lib'], str(_r['factor_id'])))
            if _vv is not None:
                _vv_s = np.asarray(_vv, dtype=float)[_order_me]
                pool_s.append(_vv_s)
                pool_caches.append(_new_pair_cache(_vv_s))
        except Exception:
            continue
    scored = []
    for _k, (_pos, _key) in enumerate(_keys):
        r = vwap_df.iloc[_pos]
        try:
            _v = np.asarray(val_store[_key], dtype=float)
            _nl = vwap_nl.get(vwap_df.index[_pos]) or {}
            _jsd = _nl.get('jsd')
            if _jsd is None or not np.isfinite(_jsd):
                _jsd = float(jsd_tails(_v, np.asarray(y_vwap_me, dtype=float)))
            if _use_par:
                _mu, _md = _cond_cache.get(_k, (None, None))
            else:
                per = cond_ic_metrics(shim, _v, prep_me, name=str(r['factor_id']),
                                      tags=("mkt_up", "mkt_dn"))
                _mu = per.get('cond_mkt_up_rank_ic_mean')
                _md = per.get('cond_mkt_dn_rank_ic_mean')
            _con = (abs(float(_mu) - float(_md))
                    if _mu is not None and _md is not None
                    and np.isfinite(_mu) and np.isfinite(_md) else float('nan'))
            _j = float(_jsd) if np.isfinite(_jsd) else 0.0
            _sc = float(_j + 0.5 * _con) if np.isfinite(_con) else float('nan')
            if np.isfinite(_sc):
                scored.append((_key, r, _sc, float(_j), float(_con)))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[2])
    out = []
    for _key, r, _sc, _j, _con in scored:
        if len(out) >= top_n:
            break
        _v_s = np.asarray(val_store[_key], dtype=float)[_order_me]
        _cache_s = _new_pair_cache(_v_s)
        if _pair_vs_pool(_v_s, pool_s, _day_s, nd_me, threshold,
                         cache=_cache_s, pool_caches=pool_caches):
            continue
        pool_s.append(_v_s)
        pool_caches.append(_cache_s)
        out.append({'factor_id': str(r['factor_id']), 'lib': str(r['lib']),
                    'family': str(r.get('family', '')),
                    'formula_cn': str(r.get('formula_cn', '')),
                    'habitat_score': round(float(_sc), 4),
                    'habitat_jsd': round(float(_j), 4),
                    'habitat_contrast': round(float(_con), 4)})
    return out


def norm_refs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['uploaded'] = df['uploaded'].fillna('')
    df['sia'] = df['A_official']
    df['rank_ic'] = df['rank_ic_mean']
    df['ic'] = df['ic_mean']
    df['win'] = df['ic_win_rate']
    df['rank_ic_win'] = df['rank_ic_win_rate']
    df['net'] = df['net_excess_ann']
    df['long'] = df['long_excess_ann']
    df['sharpe'] = df['long_sharpe']
    df['mdd'] = df['long_max_drawdown']
    df['mono'] = df['monotonicity']
    # mono 修复（2026-09-26）：flat 里的 LGB 是筛选时旧口径（负 mono 归零），
    # 用当前 lgb_score（|mono|）从已存指标重算，保证 refs 与 mined 排序口径一致。
    df['LGB'] = [lgb_score({'rank_ic_mean': r, 'rank_ic_ir': ir, 'rank_ic_win_rate': w,
                            'turnover': t, 'monotonicity': mo})
                 for r, ir, w, t, mo in zip(df['rank_ic_mean'], df['rank_ic_ir'],
                                           df['rank_ic_win_rate'], df['turnover'],
                                           df['monotonicity'])]
    df['family'] = 'ref:' + df['lib'].astype(str)
    df['formula_cn'] = df['formula_cn'].astype(str)
    return df


def fmt(x, spec='%.4f') -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return '-'
        return spec % v
    except (TypeError, ValueError):
        return '-'


def pct(x) -> str:
    try:
        v = float(x)
        return f'{v * 100:.1f}%' if np.isfinite(v) else '-'
    except (TypeError, ValueError):
        return '-'


def cond_ic_table(df: pd.DataFrame, frame: pd.DataFrame, ev: Evaluator,
                  builder: FactorBuilder, specs_by_id: dict,
                  vwap_order: list[int] | None, vwap_df: pd.DataFrame,
                  prep: dict | None = None, end: str | None = None,
                  jobs: int = 1) -> list[dict]:
    """双榜入选者的条件 RankIC（2026-09-25）：栖息地（hot5/hot2）+ 市场状态族，
    与主管线 COND-IC 同口径。c2c 榜 top20 + 榜2 正交入选者并集，逐候选×逐标签。"""
    from cond_ic import build_cond_prep, cond_ic_metrics
    from decorrelate_top import factor_full_values
    cond_prep = prep if prep is not None else build_cond_prep(frame)
    _ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev, end=(end or W.DEFAULT_DATA_CUT))
             for lab in ('c2c', 'vwap')}
    c2c_sel = df[(df['label'] == 'c2c')].sort_values('A_official', ascending=False).head(20)
    vwap_sel = (vwap_df.iloc[vwap_order] if vwap_order else vwap_df.head(20))
    # store 存量列一次性批量映射（四源 helper 统一消费）。
    _store_full: dict = {}
    try:
        _need = sorted({str(r['factor_id']) for _, r in
                        list(c2c_sel.iterrows()) + list(vwap_sel.iterrows())
                        if str(r['lib']) == 'store'})
        if _need:
            import boundary as _B
            _sf = _B.read_factors(_need, data_max=(end or W.DEFAULT_DATA_CUT),
                                  label='cond-store')
            _m = _sf.merge(frame[['stock_code', 'date']],
                           on=['stock_code', 'date'], how='right')
            for _c in _need:
                _vv = _m[_c].to_numpy(dtype=float) if _c in _m.columns else None
                if _vv is not None and len(_vv) == len(frame):
                    _store_full[_c] = _vv
            del _sf, _m
    except Exception:
        pass
    seen = set()
    _tasks = []
    for _sel in (c2c_sel, vwap_sel):
        for _, r in _sel.iterrows():
            key = (r['lib'], str(r['factor_id']))
            if key in seen:
                continue
            seen.add(key)
            try:
                _fr_raw = r['formula_cn']
                _fam = r['family']
            except Exception:
                continue
            _tasks.append((str(r['lib']), str(r['factor_id']),
                           _fr_raw, str(r.get('formula_cn', '')), _fam))
    # 2026-10-19 并行化：逐候选独立（串并同一 worker，_ex.map 保序 → out 顺序恒等）
    global _COND_CTX
    _COND_CTX = {"frame": frame, "ev": ev, "builder": builder,
                 "specs": specs_by_id, "store_full": _store_full,
                 "ctxs": _ctxs, "cond_prep": cond_prep,
                 "cond_ic_metrics": cond_ic_metrics}
    out = []
    _nj = _par_jobs(jobs)
    if _nj > 1 and len(_tasks) > 1:
        import concurrent.futures as _cf
        _nw = min(_nj, len(_tasks))
        print(f'条件 RankIC 并行：{len(_tasks)} 个候选 × {_nw} 进程 …', flush=True)
        try:
            with _cf.ProcessPoolExecutor(max_workers=_nw) as _ex:
                for _res in _ex.map(_cond_worker, _tasks):
                    if _res is not None:
                        out.append(_res)
        finally:
            _COND_CTX = None
    else:
        for _task in _tasks:
            _res = _cond_worker(_task)
            if _res is not None:
                out.append(_res)
        _COND_CTX = None
    return out


def build_md(df: pd.DataFrame, out_json: dict, vwap_order: list[int] | None = None,
             vwap_rho: dict | None = None, cond_table: list[dict] | None = None,
             c2c_order: list[int] | None = None, dom_info: dict | None = None) -> str:
    lines = ['# 对齐口径双榜单（aligned_v1）',
             '',
             f'生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  '
             f'候选池：挖矿 {out_json["n_mined_eval"]} + 外部参考 {out_json["n_refs"]} '
             f'= {out_json["n_total"]} 条(因子×标签)',
             '',
             '**口径**：窗口 2021-09-01~2026-08-21（≈平台"入池日前五年"）、全样本（含 ST/一字板/停牌）、'
             '每 3 交易日调仓、因子截面 zscore+clip±2σ。'
             '校准参照 qs2-R741：平台 IC -0.0328/ICIR -0.2749/RankIC -0.0687，'
             '本地同口径 IC -0.0353/ICIR -0.2914/RankIC -0.0782（IC/ICIR 差 ≤8%，RankIC 恒 +14% 样本偏差）。',
             '',
             '**榜1 排序** = 官方A：`|RankIC|×|ICIR(Pearson)|×IC胜率`（c2c_3d，上传决策主排序；'
             '净超额/夏普/回撤为单因子C代理复核列，非官方计分）。**榜2 排序** = LGB 复合分：'
             '`|RankIC|×|RankICIR|×rank胜率 × |mono| × T_ref/max(单次调仓换手,T_ref)`，T_ref=0.3'
             '（vwap_3d，自用 LightGBM；换手按每次调仓 2~3 日持仓口径）。',
             '']

    c2c_all = df[(df['label'] == 'c2c')].sort_values('A_official', ascending=False)
    c2c_orth = bool(c2c_order)
    c2c = c2c_all.iloc[c2c_order] if c2c_orth else c2c_all.head(20)
    lines += ['## 榜1：官方A 口径 top20（c2c_3d 标签，全池）', '']
    if c2c_orth:
        lines += [f'> 榜1 已做正交去重：按 A_official 降序扫描，与已选者逐日截面 '
                  f'|ρ|>{CORR_THRESHOLD} 的高相关候选只留 A_official 最优者'
                  f'（数学孪生/窗口微变被合并）。', '']
    lines += ['| 排名 | 因子 | 来源 | 方向 | RankIC | ICIR(P) | IC胜率 | 换手/次 | 净超额(年) | 夏普 | 回撤 | A官方 | 已上传 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for j, (_, r) in enumerate(c2c.iterrows(), 1):
        _di = dom_info.get((r['lib'], str(r['factor_id'])), {})
        _dom = f" [dom={_di['max']:.2f}@{_di['field']}]" if _di else ''
        nm = f"{r['factor_id']} ({r['family']}){_dom} {str(r['formula_cn'])[:70]}"
        lines.append('| %d | %s | %s | %s | %+.4f | %+.3f | %s | %s | %s | %s | %s | %s | %s |' % (
            j, nm, r['lib'], int(r['direction']) if r['direction'] == r['direction'] else '-',
            float(r['rank_ic']), float(r['ic_ir']), pct(r['win']), pct(r['turnover']),
            pct(r['net']), fmt(r['sharpe']), pct(r['mdd']),
            '%.5f' % float(r['A_official']), r['uploaded'] or ''))
    lines.append('')

    vwap_df = df[(df['label'] == 'vwap')].sort_values('LGB', ascending=False)
    orthogonal = bool(vwap_order)
    vwap = vwap_df.iloc[vwap_order] if orthogonal else vwap_df.head(20)
    lines += ['## 榜2：LGB 复合分 top20（vwap_3d 标签，全池，自用 LightGBM）', '']
    if orthogonal:
        n_skip = int(sum(1 for m in (vwap_rho or {}).values() if np.isfinite(m)
                         and m > CORR_THRESHOLD))
        lines += [f'> 榜2 已做排序层正交：按 LGB 降序扫描，与已选者逐日截面 |ρ|>{CORR_THRESHOLD} '
                  f'的高相关候选只留 LGB 最优者（扫描内高相关剔除 {n_skip} 个；'
                  'decorrelate_top.py 仍可独立复核）。', '']
    lines += ['| 排名 | 因子 | 来源 | 方向 | RankIC | RankICIR | rank胜率 | 单调性 | 换手/次 | 覆盖度 | LGB | MI | JSD | (参考)A官方 | 已上传 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for j, (_, r) in enumerate(vwap.iterrows(), 1):
        nm = f"{r['factor_id']} ({r['family']}) {str(r['formula_cn'])[:70]}"
        lines.append('| %d | %s | %s | %s | %+.4f | %+.3f | %s | %s | %s | %s | %.5f | %s | %s | %s | %s |' % (
            j, nm, r['lib'], int(r['direction']) if r['direction'] == r['direction'] else '-',
            float(r['rank_ic']), float(r['rank_ic_ir']), pct(r['rank_ic_win']),
            fmt(r['mono'], '%+.2f'), pct(r['turnover']), pct(r['coverage']),
            float(r['LGB']), fmt(r.get('mi'), '%.4f'), fmt(r.get('jsd'), '%.3f'),
            fmt(r.get('A_official'), '%.5f'), r['uploaded'] or ''))
    lines.append('')
    if cond_table:
        lines += ['', '## 条件 RankIC（双榜入选者；栖息地/市场状态，主管线 COND-IC 同口径）', '',
                  'hot* = 实际交易票内子集 RankIC（栖息地）；mkt_* = 市场涨/跌日。'
                  '全市场 IC 高但 hot 低 = 栖息地稀释——喂 LGB 前重点看此列。', '',
                  '| 候选 | 标签 | hot5_q80 | hot2_q80 | hot5_cold | mkt_up | mkt_dn |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for row in cond_table:
            for lab in ('c2c', 'vwap'):
                d = (row.get('per') or {}).get(lab) or {}
                if not d:
                    continue

                def _g(tag):
                    return fmt(d.get(f'cond_{tag}_rank_ic_mean'), '%+.4f')

                lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
                    row['name'], lab, _g('hot5_q80'), _g('hot2_q80'),
                    _g('hot5_cold'), _g('mkt_up'), _g('mkt_dn')))
        lines.append('')
    lines.append('> 注：单因子指标无法度量多因子互补性；终审建议用现有 LightGBM 做留一/置换'
                 '重要性（ΔAUC 或 Δprecision@5）。refs 因子的平台公式翻译在 upload 时按需补。')
    return '\n'.join(lines)


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='aligned_v1 双榜单')
    ap.add_argument('--top', type=int, default=200,
                    help='挖矿候选按旧证据每标签前 N 并集预筛')
    ap.add_argument('--from-flat', default='',
                    help='跳过重估，从已有 ALIGNED_TOP2_*_flat.csv 重建 MD')
    ap.add_argument('--scan', type=int, default=150,
                    help='榜1/榜2 正交扫描深度（按指标降序扫描前 N 个候选）')
    ap.add_argument('--keep-pool', type=int, default=20,
                    help='V4.1榜2正交通道保留数（默认20保持历史行为；SOP示例80）')
    ap.add_argument('--keep-base', type=int, default=20,
                    help='V4.1门控靶标面板机制均衡抽取数（默认20；每family最多4个）')
    ap.add_argument('--include-custom', action='store_true', default=False,
                    help='纳入custom_registry自定义因子与正交去重（默认关闭以保持历史可比）')
    ap.add_argument('--include-store', action='store_true', default=True,
                    help='纳入factors.parquet存量个股列为第四源store（V4.3c默认开）')
    ap.add_argument('--no-store', dest='include_store', action='store_false',
                    help='关闭store第四源（历史可比对照）')
    ap.add_argument('--habitat-seats', type=int, default=10,
                    help='Class1栖息地尖兵席位（hot5_q80 |t|>=2.5正交保留数，默认10）')
    ap.add_argument('--habitat-pool', type=int, default=10,
                    help='Class2股性榜单TopN（按habitat_score正交，默认10）')
    ap.add_argument('--export-panel', action='store_true', default=False,
                    help='导出vwap_top20.json与因子值面板（默认关闭；只需JSON时可只看榜单）')
    ap.add_argument('--data-cut', default=W.DEFAULT_DATA_CUT,
                    help='数据截止日铁律（默认 2025-12-31）：load_panel/DataContext/'
                         'pos_me/B.read_factors 全链路钳制，严禁触及 2026 数据（PLAN §20.6.2）')
    ap.add_argument('--jobs', default='1',
                    help='并行进程数：整数或 auto（按核数+可用内存自动定，'
                         '用于 eval_mined/评估/扩面/席位池打分等；'
                         '结果与串行逐值一致，非 posix 平台自动退化串行）')
    args = ap.parse_args()
    _jobs_raw = str(args.jobs).strip().lower()
    if _jobs_raw in ('auto', '0', '-1'):
        args.jobs = _auto_jobs_build_top()
    else:
        args.jobs = max(1, int(args.jobs or 1))

    t0 = time.time()
    if args.from_flat:
        df = pd.read_csv(args.from_flat)
        df['uploaded'] = df['uploaded'].fillna('')
        dl: dict = {}
        for rec in Ledger().read():
            d = rec.get('data') or {}
            if rec.get('event') == 'local' and d.get('source') == 'aligned-top' \
                    and rec.get('label') == 'c2c' and rec.get('direction') is not None:
                dl[str(rec.get('expression_hash'))[:12]] = int(rec['direction'])
        mask_m = df['lib'] == 'mined'
        df.loc[mask_m, 'direction'] = df.loc[mask_m, 'factor_id'].map(dl)
        df['direction'] = pd.to_numeric(df['direction'], errors='coerce')
        n_mined = int(df.loc[df['lib'] == 'mined', 'factor_id'].nunique())
        n_refs = int(df.loc[df['lib'] != 'mined', 'factor_id'].nunique())
        payload = {'generated_at': datetime.now().isoformat(timespec='seconds'),
                   'from_flat': args.from_flat, 'n_mined_eval': n_mined,
                   'n_refs': n_refs, 'n_total': len(df)}
        md = build_md(df, payload)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = OUT_DIR / f'ALIGNED_TOP2_{ts}.md'
        md_path.write_text(md, encoding='utf-8')
        print(f'saved {md_path}  {time.time() - t0:.0f}s', flush=True)
        return 0

    # V4.3c data-cut 铁律：面板/评估/pos_me/因子读取全部钳制在 cut 内。
    frame = load_panel(data_cut=args.data_cut)
    ev = Evaluator(frame, cache_size=32)
    builder = FactorBuilder(frame)
    refs_df = norm_refs(load_refs_rows())
    _picks, _hab_h = mined_prefilter(args.top, data_cut=args.data_cut)
    mined_df = eval_mined(_picks, frame=frame, ev=ev,
                          builder=builder, end=args.data_cut, jobs=args.jobs)
    custom_df = eval_custom(frame=frame, ev=ev, end=args.data_cut) \
        if args.include_custom else pd.DataFrame()
    # V4.3c：_store_aligned 全局只调一次（eval_store 与 store_vals 复用，
    # 禁重复读 7GB 表 + 重复造 627 万键字典）。
    _store_aligned_once = None
    if args.include_store:
        try:
            _store_aligned_once = _store_aligned(frame, cut=args.data_cut)
            if _store_aligned_once[0] is None:
                _store_aligned_once = None
        except Exception as exc:
            print(f'warning: store 对齐失败（{str(exc)[:120]}），跳过store源',
                  flush=True)
    store_df = eval_store(frame=frame, ev=ev, end=args.data_cut,
                          _aligned=_store_aligned_once, jobs=args.jobs) \
        if _store_aligned_once is not None else pd.DataFrame()
    cols = ['lib', 'factor_id', 'name_cn', 'base', 'transform', 'window', 'formula_cn',
            'uploaded', 'label', 'coverage', 'A_official', 'LGB', 'sia', 'rank_ic', 'ic',
            'ic_ir', 'win', 'rank_ic_ir', 'rank_ic_win', 'turnover', 'turnover_month',
            'net', 'long', 'sharpe', 'mdd', 'mono', 'direction', 'n_valid_periods', 'family',
            'mi', 'mi_std', 'nmi', 'jsd', 'habitat_pick']
    # 非线性诊断列（2026-09-26）：正交扫描后回填 vwap 行，其余 NaN
    for _f in (refs_df, mined_df, custom_df, store_df):
        for _k in ('mi', 'mi_std', 'nmi', 'jsd'):
            _f[_k] = np.nan
    cols_present = [c for c in cols if c in refs_df.columns]
    refs_df = refs_df.reindex(columns=cols_present)
    _mined = mined_df.reindex(columns=cols_present)
    _custom = custom_df.reindex(columns=cols_present) if len(custom_df) else custom_df
    _store = store_df.reindex(columns=cols_present) if len(store_df) else store_df
    # FutureWarning 抑制（pandas concat dtype 变更提示，不影响结果）
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        df = pd.concat([refs_df, _mined] + ([_custom] if len(_custom) else [])
                       + ([_store] if len(_store) else []),
                       ignore_index=True)
    for c in ('A_official', 'LGB', 'rank_ic', 'rank_ic_ir', 'ic_ir', 'win',
              'rank_ic_win', 'turnover', 'net', 'sharpe', 'mdd', 'mono', 'coverage',
              'mi', 'mi_std', 'nmi', 'jsd'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    n_mined = int(df.loc[df['lib'] == 'mined', 'factor_id'].nunique()) if len(df) else 0
    n_refs = int(df.loc[df['lib'] == 'refs', 'factor_id'].nunique()) if len(df) else 0
    n_custom = int(df.loc[df['lib'] == 'custom', 'factor_id'].nunique()) if len(df) else 0
    n_store = int(df.loc[df['lib'] == 'store', 'factor_id'].nunique()) if len(df) else 0
    payload = {'generated_at': datetime.now().isoformat(timespec='seconds'),
               'top': args.top, 'n_mined_eval': n_mined, 'n_refs': n_refs,
               'n_custom': n_custom, 'n_store': n_store,
               'keep_pool': args.keep_pool, 'keep_base': args.keep_base,
               'n_total': len(df), 'vwap_orthogonal': False}

    # 榜1/榜2 排序层正交（2026-09-24/25）：共用逐日截面基础设施
    vwap_order: list[int] | None = None
    vwap_rho: dict | None = None
    c2c_order: list[int] | None = None
    vwap_df = df[(df['label'] == 'vwap')].sort_values('LGB', ascending=False)
    c2c_df = df[(df['label'] == 'c2c')].sort_values('A_official', ascending=False)

    from fitness import START, END as FIT_END
    _fit_end = min(str(FIT_END), str(args.data_cut))
    mask_d = ((frame['date'] >= pd.Timestamp(START))
              & (frame['date'] <= pd.Timestamp(_fit_end))).to_numpy()
    # [CORR-CALIBRATION 2026-10-02] 相关性截面由「每月最后 1 个交易日」改为「月内全部交易日」。
    # 与 decorrelate_top / dedup / pool_matrix 同步；pos_me 形参名沿用旧名以保持兼容。
    pos_me = np.flatnonzero(mask_d)
    day_group = frame['date'].to_numpy()[pos_me]
    # 非线性诊断用 vwap 标签（逐日位置一次算好；c2c 不用）
    y_vwap_me = build_label_series(frame, 'vwap', 3).to_numpy(dtype=float)[pos_me]
    _specs = load_all_specs()
    # 因子值缓存（2026-09-26）：正交扫描在 pos_me（逐日截面）的因子值按
    # 表达式hash 落盘；val_store 在本次运行内给 dom 复用；stats 统计命中。
    cache_dir = _value_cache_dir(frame, pos_me)
    val_store: dict = {}
    cache_stats: dict = {}
    print(f'value cache dir: {cache_dir}', flush=True)
    # V4.3c store 值（pos_me 切片，供正交扫描孪生判定；存量即真值，不进 value_cache）。
    # 复用上游已对齐的 (_sm, _spos)，不再二次读表。
    store_vals: dict = {}
    if _store_aligned_once is not None:
        try:
            _sm, _spos = _store_aligned_once
            for _c in [c for c in _sm.columns if c not in ("stock_code", "date")]:
                _vv = np.full(len(frame), np.nan)
                _vv[_spos >= 0] = _sm[_c].to_numpy(dtype=float)[_spos[_spos >= 0]]
                store_vals[_c] = _vv[pos_me].astype(np.float32)
            print(f'store values: {len(store_vals)} cols on pos_me', flush=True)
            del _store_aligned_once
        except Exception as exc:
            print(f'warning: store 值映射失败（{str(exc)[:120]}），孪生判定退化', flush=True)
            store_vals = {}

    vwap_nl: dict = {}
    if len(vwap_df) > 20:
        vwap_order, vwap_rho, vwap_nl = vwap_orthogonal_indices(
            vwap_df, frame, ev, builder, pos_me, day_group, specs_by_id=_specs,
            max_scan=args.scan, keep=max(args.keep_pool, args.keep_base),
            cache_dir=cache_dir, val_store=val_store,
            stats=cache_stats, label='vwap', y_me=y_vwap_me,
            store_vals=store_vals or None, jobs=args.jobs)
        payload['vwap_orthogonal'] = bool(vwap_order)
        payload['vwap_orth_scan'] = int(min(args.scan, len(vwap_df)))
        print(f'榜2 正交：候选 {len(vwap_df)} → 保留 {len(vwap_order)}  '
              f'cache hit={cache_stats.get("hit", 0)} miss={cache_stats.get("miss", 0)}',
              flush=True)

    # 榜1 正交（2026-09-25）：c2c 榜也做正交去重，防数学孪生/高相关变体重复入选
    c2c_df = df[(df['label'] == 'c2c')].sort_values('A_official', ascending=False)
    if len(c2c_df) > 20:
        c2c_order, c2c_rho, _ = vwap_orthogonal_indices(
            c2c_df, frame, ev, builder, pos_me, day_group, specs_by_id=_specs,
            max_scan=args.scan, cache_dir=cache_dir, val_store=val_store,
            stats=cache_stats, label='c2c', store_vals=store_vals or None,
            jobs=args.jobs)
        payload['c2c_orthogonal'] = bool(c2c_order)
        payload['c2c_orth_scan'] = int(min(args.scan, len(c2c_df)))
        print(f'榜1 正交：候选 {len(c2c_df)} → 保留 {len(c2c_order)}  '
              f'cache hit={cache_stats.get("hit", 0)} miss={cache_stats.get("miss", 0)}',
              flush=True)
    payload['value_cache'] = {'hit': cache_stats.get('hit', 0),
                              'miss': cache_stats.get('miss', 0)}
    # 非线性诊断回填（2026-09-26）：vwap 扫描候选的 MI/JSD 进 df（c2c 行保持 NaN）
    if vwap_nl:
        for _lix, _d in vwap_nl.items():
            for _k in ('mi', 'mi_std', 'nmi', 'jsd'):
                df.loc[_lix, _k] = _d.get(_k)
        _n_nl = len(vwap_nl)
        print(f'非线性诊断：vwap 扫描 {_n_nl} 个候选中位 MI 回填完成', flush=True)

    # 条件 RankIC（2026-09-25）：双榜入选者的栖息地/市况条件 IC 汇总
    cond_table = None
    try:
        cond_table = cond_ic_table(df, frame, ev, builder, _specs,
                                   vwap_order, vwap_df, jobs=args.jobs)
        payload['cond_table_n'] = len(cond_table)
        print(f'条件 RankIC：双榜入选者 {len(cond_table)} 个完成', flush=True)
    except Exception as exc:
        cond_table = None
        print(f'warning: 条件 RankIC 汇总失败 {exc}', flush=True)

    # V4.3c 栖息地席位 + Class2 股性榜单（窗内 cond 口径，与 LGB 选择同窗）。
    seat_order: list[int] = []
    seat_detail: list[dict] = []
    hab_pool: list[dict] = []
    _day_codes, _ = pd.factorize(day_group, sort=True)
    _day_codes = np.asarray(_day_codes)
    _nd_me = int(_day_codes.max()) + 1 if len(_day_codes) else 0
    # habitat 估值扩面——val_store 只含 LGB 前 scan 名；全表扫描，
    # 优先级：habitat_pick（台账豁免）> store/机制家族 > 其余；上限 150。
    hab_vals, _extra_keys = _habitat_expand(
        vwap_df, val_store, store_vals, ev, builder, frame, pos_me,
        cache_dir, _specs, jobs=args.jobs)
    try:
        from cond_ic import build_cond_prep
        _prep_me_full = build_cond_prep(frame)
        _prep_me = {'tags': {k: np.asarray(v)[pos_me]
                             for k, v in _prep_me_full['tags'].items()}}
        if args.habitat_seats > 0 and len(val_store):
            seat_order, seat_detail = habitat_seats(
                vwap_df, vwap_order, hab_vals, pos_me, day_group,
                y_vwap_me, _prep_me, _day_codes, _nd_me,
                n_seats=args.habitat_seats, extra_keys=_extra_keys,
                jobs=args.jobs)
            print(f'栖息地席位：{len(seat_order)} 个 '
                  f'{[d["factor_id"] for d in seat_detail]}', flush=True)
        if args.habitat_pool > 0 and len(val_store):
            hab_pool = habitat_pool(
                vwap_df, vwap_order, hab_vals, pos_me, day_group,
                y_vwap_me, _prep_me, vwap_nl, _day_codes, _nd_me,
                top_n=args.habitat_pool, jobs=args.jobs)
            print(f'Class2股性榜单：{len(hab_pool)} 个', flush=True)
    except Exception as exc:
        print(f'warning: 栖息地席位/股性榜单失败 {str(exc)[:150]}', flush=True)
    payload['habitat_seats'] = seat_detail
    payload['habitat_pool_n'] = len(hab_pool)

    # dom 诊断（2026-09-25）：双榜入选者的单字段支配度（快速 Spearman @ pos_me）
    # 2026-10-19 并行化：逐候选独立（串并同一 worker，map 保序 → _dom_info 顺序恒等）
    _DOM_FIELDS = ('CLOSE', 'VOLUME', 'AMOUNT', 'MARKET_CAP', 'TURNOVER')
    _dom_info: dict = {}
    if all(f in frame.columns for f in _DOM_FIELDS):
        _dom_tasks = []
        _dom_seen = set()
        for _tag, _sel_df, _order in (('c2c', c2c_df, c2c_order),
                                      ('vwap', vwap_df, vwap_order)):
            if _order is None:
                continue
            for _idx in _order:
                _r = _sel_df.iloc[_idx]
                _key = (_r['lib'], str(_r['factor_id']))
                if _key in _dom_seen:
                    continue
                _dom_seen.add(_key)
                try:
                    _fcn = str(_r.get('formula_cn', ''))
                except Exception:
                    continue
                _dom_tasks.append((_key, str(_r['lib']),
                                   str(_r['factor_id']), _fcn))
        global _DOM_CTX
        _DOM_CTX = {"val_store": val_store, "store_vals": store_vals or {},
                    "frame": frame, "ev": ev, "builder": builder,
                    "specs": _specs, "pos_me": pos_me,
                    "fields_vals": {f: pd.to_numeric(frame[f].iloc[pos_me],
                                                     errors='coerce')
                                    .to_numpy(dtype=float)
                                    for f in _DOM_FIELDS}}
        _nj_dom = _par_jobs(args.jobs)
        try:
            if _nj_dom > 1 and len(_dom_tasks) > 1:
                import concurrent.futures as _cf
                _nw = min(_nj_dom, len(_dom_tasks))
                print(f'dom 诊断并行：{len(_dom_tasks)} 个候选 × {_nw} 进程 …',
                      flush=True)
                with _cf.ProcessPoolExecutor(max_workers=_nw) as _ex:
                    for _key, _dd in _ex.map(_dom_worker, _dom_tasks):
                        if _dd is not None:
                            _dom_info[_key] = {'max': round(_dd[1], 2),
                                               'field': _dd[0]}
            else:
                for _t in _dom_tasks:
                    _key, _dd = _dom_worker(_t)
                    if _dd is not None:
                        _dom_info[_key] = {'max': round(_dd[1], 2),
                                           'field': _dd[0]}
        finally:
            _DOM_CTX = None
    payload['dom_info'] = {f'{k[0]}_{k[1]}': v for k, v in _dom_info.items()}

    md = build_md(df, payload, vwap_order=vwap_order, vwap_rho=vwap_rho,
                  cond_table=cond_table, c2c_order=c2c_order, dom_info=_dom_info)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = OUT_DIR / f'ALIGNED_TOP2_{ts}.md'
    md_path.write_text(md, encoding='utf-8')
    df.to_csv(OUT_DIR / f'ALIGNED_TOP2_{ts}_flat.csv', index=False, encoding='utf-8-sig')
    # V4.1动态靶标面板：vwap正交顺序按LGB降序、每family最多4个均衡抽取keep_base个
    # 断路2修复：因子值面板必须落 vwap_order[:keep_pool] 全部正交主力 + 席位
    #（旧代码只落 family 均衡的 keep_base=20 列，60 个主力凭空蒸发）。
    # json 同步扩为面板全集（F1 方向表；门控基准仍取前 keep_base）。
    try:
        _vsel = vwap_df.iloc[vwap_order] if vwap_order else vwap_df.head(args.keep_base)
        _picked, _fam_cnt = [], {}
        for _, _r in _vsel.iterrows():
            _f = str(_r.get('family', 'x'))
            if _fam_cnt.get(_f, 0) >= 4:
                continue
            _picked.append(_r)
            _fam_cnt[_f] = _fam_cnt.get(_f, 0) + 1
            if len(_picked) >= args.keep_base:
                break
        _pool_rows = list(vwap_df.iloc[(vwap_order or [])[:max(0, args.keep_pool)]].iterrows())
        _pool_rows = [r for _, r in _pool_rows]
        _seen_panel = set()
        _panel_rows = []
        for _r in _picked + _pool_rows + [vwap_df.iloc[_sp] for _sp in seat_order]:
            _fid = str(_r.get('factor_id'))
            if _fid in _seen_panel:
                continue
            _seen_panel.add(_fid)
            _panel_rows.append(_r)
        _top20 = []
        for _r in _panel_rows:
            _d = _r.get('direction')
            try:
                _d = int(_d)
            except Exception:
                _d = 1
            _top20.append({'factor': str(_r.get('factor_id')),
                           'direction': 1 if _d == 1 else -1,
                           'family': str(_r.get('family', '')),
                           'formula': str(_r.get('formula_cn', '')),
                           'lib': str(_r.get('lib', ''))})
        # V4.3c 席位标记（_panel_rows 已含席位；此处只打标记供 export 强制收录）。
        _seat_ids = set()
        try:
            for _sp in seat_order:
                _seat_ids.add(str(vwap_df.iloc[_sp].get('factor_id')))
        except Exception:
            pass
        for _e in _top20:
            if _e['factor'] in _seat_ids:
                _e['habitat_seat'] = True
        _base_dir = HERE.parent.parent / 'results' / 'merge_v2'
        _base_dir.mkdir(parents=True, exist_ok=True)
        (_base_dir / 'vwap_top20.json').write_text(
            json.dumps({'generated_at': payload.get('generated_at'), 'n': len(_top20),
                        'base20': _top20}, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'vwap_top20.json: {len(_top20)} factors', flush=True)
        (_base_dir / 'habitat_seats.json').write_text(
            json.dumps({'generated_at': payload.get('generated_at'),
                        'seats': sorted(_seat_ids)}, ensure_ascii=False, indent=1),
            encoding='utf-8')
        if args.export_panel and _top20:
            _panel_cols = {'stock_code': frame['stock_code'].to_numpy(),
                           'date': pd.to_datetime(frame['date']).to_numpy()}
            # V4.3c：store 列一次性批量读取（禁循环内单列重复读 7GB 表）。
            # _panel_rows = picked20 + order[:keep_pool] + seats（断路2：全量落盘）。
            _panel_store = sorted({str(_r.get('factor_id')) for _r in _panel_rows
                                   if _r.get('lib') == 'store'})
            _panel_smap: dict = {}
            if _panel_store:
                try:
                    import boundary as _B
                    _sf = _B.read_factors(_panel_store, data_max=args.data_cut,
                                          label='top20-panel-store')
                    _m = _sf.merge(frame[['stock_code', 'date']],
                                   on=['stock_code', 'date'], how='right')
                    for _c in _panel_store:
                        _vv = _m[_c].to_numpy(dtype=float) if _c in _m.columns else None
                        if _vv is not None and len(_vv) == len(frame):
                            _panel_smap[_c] = _vv
                    del _sf, _m
                except Exception as _e:
                    print(f'  warning: panel store 批量读失败 {str(_e)[:100]}',
                          flush=True)
            for _r in _panel_rows:
                _fid = str(_r.get('factor_id'))
                try:
                    if _r.get('lib') == 'mined':
                        _vals = np.asarray(ev.values(parse_formula(str(_r['formula_cn']))),
                                           dtype=float)
                    elif _r.get('lib') == 'custom':
                        from custom_registry import evaluate_custom_factor as _ecf
                        from custom_registry import load_all_custom_specs as _lcs
                        _spec = _lcs().get(_fid)
                        _vals = np.asarray(_ecf(_spec, frame, ev), dtype=float) \
                            if _spec is not None else np.full(len(frame), np.nan)
                    elif _r.get('lib') == 'store':
                        _vals = _panel_smap.get(_fid, np.full(len(frame), np.nan))
                    else:
                        _spec = _specs.get(_fid)
                        _vals = builder.get_factor(_spec).to_numpy(dtype=float) \
                            if _spec is not None else np.full(len(frame), np.nan)
                except Exception:
                    _vals = np.full(len(frame), np.nan)
                _panel_cols[_fid] = np.asarray(_vals, dtype=float)
            _pdf = pd.DataFrame(_panel_cols)
            _pdf.to_parquet(_base_dir / 'vwap_top20_factors.parquet', index=False)
            print(f'vwap_top20_factors.parquet: {_pdf.shape}', flush=True)
    except Exception as _e:
        print(f'warning: vwap_top20 export failed {str(_e)[:200]}', flush=True)
    # V4.3c Class2 股性榜单落盘（全窗值，供 export Class2）。
    try:
        if hab_pool:
            _hp_cols = {'stock_code': frame['stock_code'].to_numpy(),
                        'date': pd.to_datetime(frame['date']).to_numpy()}
            _store_need = [h['factor_id'] for h in hab_pool if h['lib'] == 'store']
            _store_map = {}
            if _store_need:
                import boundary as _B2
                _sf2 = _B2.read_factors(_store_need, data_max=args.data_cut,
                                        label='habitat-pool-store')
                _m2 = _sf2.merge(frame[['stock_code', 'date']],
                                 on=['stock_code', 'date'], how='right')
                for _c in _store_need:
                    _vv2 = _m2[_c].to_numpy(dtype=float) if _c in _m2.columns else None
                    if _vv2 is not None and len(_vv2) == len(frame):
                        _store_map[_c] = _vv2
            for h in hab_pool:
                _fid = h['factor_id']
                try:
                    if h['lib'] == 'mined':
                        _hv = np.asarray(ev.values(parse_formula(h['formula_cn'])),
                                         dtype=float)
                    elif h['lib'] == 'custom':
                        from custom_registry import evaluate_custom_factor as _ecf2
                        from custom_registry import load_all_custom_specs as _lcs2
                        _sp2 = _lcs2().get(_fid)
                        _hv = np.asarray(_ecf2(_sp2, frame, ev), dtype=float) \
                            if _sp2 is not None else np.full(len(frame), np.nan)
                    elif h['lib'] == 'store' and _fid in _store_map:
                        _hv = _store_map[_fid]
                    else:
                        _sp3 = _specs.get(_fid)
                        _hv = builder.get_factor(_sp3).to_numpy(dtype=float) \
                            if _sp3 is not None else np.full(len(frame), np.nan)
                except Exception:
                    _hv = np.full(len(frame), np.nan)
                _hp_cols[_fid] = np.asarray(_hv, dtype=float)
            pd.DataFrame(_hp_cols).to_parquet(
                _base_dir / 'vwap_habitat_pool.parquet', index=False)
            (_base_dir / 'vwap_habitat_pool.json').write_text(
                json.dumps({'generated_at': payload.get('generated_at'),
                            'members': hab_pool}, ensure_ascii=False, indent=1),
                encoding='utf-8')
            print(f'vwap_habitat_pool.parquet: {len(hab_pool)} factors', flush=True)
    except Exception as _e:
        print(f'warning: habitat pool export failed {str(_e)[:200]}', flush=True)
    (OUT_DIR / f'ALIGNED_TOP2_{ts}.json').write_text(
        json.dumps(payload, ensure_ascii=False) + '\n' + df.to_json(orient='records', force_ascii=False),
        encoding='utf-8')
    print(f'saved {md_path}  total {time.time() - t0:.0f}s', flush=True)
    print(md[:2000], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
