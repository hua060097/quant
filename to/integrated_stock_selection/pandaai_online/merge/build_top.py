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
from cache_utils import find_stock_features
from decorrelate_top import load_all_specs, month_end_spearman_matrix
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


def vwap_orthogonal_indices(df_vwap: pd.DataFrame, frame: pd.DataFrame, ev: Evaluator,
                            builder: FactorBuilder, pos_me: np.ndarray,
                            day_group: np.ndarray, specs_by_id: dict,
                            max_scan: int = 150, keep: int = 20,
                            threshold: float = CORR_THRESHOLD,
                            cache_dir: Path | None = None,
                            val_store: dict | None = None,
                            stats: dict | None = None,
                            label: str = '',
                            y_me: np.ndarray | None = None) -> tuple[list[int], dict, dict]:
    """榜2 排序层轻量正交（2026-09-24）：按 LGB 降序扫描前 max_scan 个候选，
    逐日截面 |Spearman|>threshold 的只留 LGB 最优者（与 decorrelate_top 同判据，
    但在榜单生成时即生效，decorrelate_top 仍可独立复核）。

    2026-09-26 加速：候选在 pos_me（逐日截面）的因子值经 cache_dir 落盘复用
    （key=表达式哈希，面板指纹目录），val_store 回传本次已算值供 dom 复用。

    2026-09-26 非线性诊断：label=='vwap' 且 y_me 给定时，对全部成功求值的
    扫描候选算月度 MI / 尾部 JSD（c2c 不动）。

    返回 (保留的 df 行号顺序, {df行号: 与已选者的最大|ρ|},
    {df索引标签: {'mi','mi_std','nmi','n_months','jsd'}})；值不可得的候选跳过。
    """
    cols, idxs = [], []
    for i_row, (_, r) in enumerate(df_vwap.head(max_scan).iterrows()):
        key = None
        try:
            if r['lib'] == 'mined':
                node = parse_formula(str(r['formula_cn']))
                key = expression_hash(node)
                v = _cache_load(cache_dir, key, len(pos_me))
                if v is None:
                    v = np.asarray(ev.values(node), dtype=float)[pos_me].astype(np.float32)
                    _cache_save(cache_dir, key, v)
                    if stats is not None:
                        stats['miss'] = stats.get('miss', 0) + 1
                elif stats is not None:
                    stats['hit'] = stats.get('hit', 0) + 1
            elif r['lib'] == 'custom':
                try:
                    from custom_registry import evaluate_custom_factor as _ecf
                    from custom_registry import load_all_custom_specs as _lcs
                    _spec = _lcs().get(str(r['factor_id']))
                    if _spec is None:
                        continue
                    key = 'custom_' + _spec.spec_hash
                    v = _cache_load(cache_dir, key, len(pos_me))
                    if v is None:
                        v = np.asarray(_ecf(_spec, frame, ev), dtype=float)[pos_me] \
                            .astype(np.float32)
                        _cache_save(cache_dir, key, v)
                        if stats is not None:
                            stats['miss'] = stats.get('miss', 0) + 1
                    elif stats is not None:
                        stats['hit'] = stats.get('hit', 0) + 1
                except Exception:
                    continue
            else:
                spec = specs_by_id.get(str(r['factor_id']))
                if spec is None:
                    continue
                key = _ref_spec_key(spec)
                v = _cache_load(cache_dir, key, len(pos_me))
                if v is None:
                    v = builder.get_factor(spec).to_numpy(dtype=float)[pos_me] \
                        .astype(np.float32)
                    _cache_save(cache_dir, key, v)
                    if stats is not None:
                        stats['miss'] = stats.get('miss', 0) + 1
                elif stats is not None:
                    stats['hit'] = stats.get('hit', 0) + 1
        except Exception:
            continue
        cols.append(v.astype(np.float32))
        idxs.append(i_row)
        if val_store is not None and key is not None:
            val_store[(r['lib'], str(r['factor_id']))] = v
    # 非线性诊断（2026-09-26）：只对 vwap 扫描候选算 MI/JSD（c2c 不动）；
    # 用已求值（cols），不新增因子求值。
    nl_map: dict = {}
    if label == 'vwap' and y_me is not None:
        _yv = np.asarray(y_me, dtype=float)
        _labels = list(df_vwap.index)
        for _j, _v in enumerate(cols):
            try:
                _d = mi_monthly(_v, _yv, day_group)
                _d['jsd'] = jsd_tails(_v, _yv)
                nl_map[_labels[idxs[_j]]] = _d
            except Exception:
                continue
    if not cols:
        return [], {}, nl_map
    corr = month_end_spearman_matrix(np.column_stack(cols), day_group)
    if corr is None:
        return idxs[:keep], {}, nl_map
    kept, rho_map = [], {}
    for j in range(len(idxs)):
        if not kept:
            kept.append(j)
            continue
        pair = np.abs(corr[j, kept])
        finite = pair[np.isfinite(pair)]
        mx = float(finite.max()) if finite.size else float('nan')
        rho_map[idxs[j]] = mx
        if finite.size and mx <= threshold:
            kept.append(j)
        if len(kept) >= keep:
            break
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


def mined_prefilter(top: int) -> list[dict]:
    ledger = Ledger()
    rows = _ledger_candidates(ledger)
    print(f'ledger candidates: {len(rows)}', flush=True)
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
    uniq = list(dict.fromkeys(picks))
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
                    'old_sia': best})
    out.sort(key=lambda r: -r['old_sia'])
    # 加速（2026-09-26）：携带 bucket 全量数据，让 eval_mined 能直接复用台账指标
    for r in out:
        r['bucket'] = rows[r['hash']]
    print(f'mined pick union={len(out)}', flush=True)
    return out


def eval_custom(frame: pd.DataFrame | None = None,
                ev: Evaluator | None = None) -> pd.DataFrame:
    """V4.1三源合一之custom源：评估`custom_registry`全部已注册自定义因子。

    与`eval_mined`同口径（双标签DataContext全窗重估，LGB按`refs.lgb_score T_ref=0.3`），
    结果记台账`local`事件`source='custom'`，供正交中枢统一去重。失败/覆盖不足记valid=False。
    """
    from custom_registry import evaluate_custom_factor, load_all_custom_specs
    ledger = Ledger()
    if frame is None:
        frame = load_panel()
    if ev is None:
        ev = Evaluator(frame)
    specs = load_all_custom_specs()
    print(f'custom specs={len(specs)}', flush=True)
    ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev)
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
                         'label': lab, **{k: r.get(k) for k in
                                          ('coverage', 'A_official', 'LGB', 'direction')},
                         **{k: r.get(k) for k in
                            ('sia', 'rank_ic', 'ic', 'ic_ir', 'win', 'rank_ic_ir',
                             'rank_ic_win', 'turnover', 'turnover_month', 'net', 'long',
                             'sharpe', 'mdd', 'mono', 'n_valid_periods')}})
    return pd.DataFrame(flat)


def eval_mined(picks: list[dict], frame: pd.DataFrame | None = None,
               ev: Evaluator | None = None,
               builder: FactorBuilder | None = None) -> pd.DataFrame:
    ledger = Ledger()
    if frame is None:
        frame = load_panel()
    print(f'panel rows={len(frame):,}', flush=True)
    if ev is None:
        ev = Evaluator(frame)
    if builder is None:
        builder = FactorBuilder(frame)
    ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev)
            for lab in ('c2c', 'vwap')}
    rows = []
    t0 = time.time()
    n_fast = n_slow = 0
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
            rows.append(rec)
            n_fast += 1
            continue
        try:
            node = parse_formula(p['formula'])
        except Exception:
            continue
        try:
            f_full = ev.values(node)
        except Exception:
            continue
        name = f"{p['family']}:{p['hash'][:10]}"
        rec = {'lib': 'mined', 'factor_id': p['hash'][:12], 'name_cn': p['formula'],
               'base': None, 'transform': None, 'window': None,
               'formula_cn': p['formula'], 'uploaded': '', 'family': p['family'],
               'labels': {}}
        ok_any = False
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
            n_slow += 1
        if ok_any:
            rows.append(rec)
        if i % 10 == 0 or i == len(picks):
            print(f'  [{i}/{len(picks)}] fast={n_fast} slow={n_slow} {time.time() - t0:.0f}s', flush=True)
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
                         'label': lab, **{k: r.get(k) for k in
                                          ('coverage', 'A_official', 'LGB', 'direction')},
                         **{k: r.get(k) for k in
                            ('sia', 'rank_ic', 'ic', 'ic_ir', 'win', 'rank_ic_ir',
                             'rank_ic_win', 'turnover', 'turnover_month', 'net', 'long',
                             'sharpe', 'mdd', 'mono', 'n_valid_periods')}})
    return pd.DataFrame(flat)


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
                  vwap_order: list[int] | None, vwap_df: pd.DataFrame) -> list[dict]:
    """双榜入选者的条件 RankIC（2026-09-25）：栖息地（hot5/hot2）+ 市场状态族，
    与主管线 COND-IC 同口径。c2c 榜 top20 + 榜2 正交入选者并集，逐候选×逐标签。"""
    from cond_ic import build_cond_prep, cond_ic_metrics
    cond_prep = build_cond_prep(frame)
    _ctxs = {lab: DataContext(frame=frame, label=lab, evaluator=ev)
             for lab in ('c2c', 'vwap')}
    c2c_sel = df[(df['label'] == 'c2c')].sort_values('A_official', ascending=False).head(20)
    vwap_sel = (vwap_df.iloc[vwap_order] if vwap_order else vwap_df.head(20))
    seen = set()
    out = []
    for _sel in (c2c_sel, vwap_sel):
        for _, r in _sel.iterrows():
            key = (r['lib'], str(r['factor_id']))
            if key in seen:
                continue
            seen.add(key)
            try:
                if r['lib'] == 'mined':
                    node = parse_formula(str(r['formula_cn']))
                    f_full = np.asarray(ev.values(node), dtype=float)
                else:
                    spec = specs_by_id.get(str(r['factor_id']))
                    if spec is None:
                        continue
                    f_full = builder.get_factor(spec).to_numpy(dtype=float)
                per = {}
                for lab, ctx in _ctxs.items():
                    per[lab] = cond_ic_metrics(ctx, f_full[ctx._pos], cond_prep,
                                               name=str(r['factor_id']))
                out.append({'name': f"{r['factor_id']} ({r['family']})",
                            'formula': r['formula_cn'], 'lib': r['lib'], 'per': per})
            except Exception:
                continue
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
    ap.add_argument('--export-panel', action='store_true', default=False,
                    help='导出vwap_top20.json与因子值面板（默认关闭；只需JSON时可只看榜单）')
    args = ap.parse_args()

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

    frame = load_panel()
    ev = Evaluator(frame)
    builder = FactorBuilder(frame)
    refs_df = norm_refs(load_refs_rows())
    mined_df = eval_mined(mined_prefilter(args.top), frame=frame, ev=ev, builder=builder)
    custom_df = eval_custom(frame=frame, ev=ev) if args.include_custom else pd.DataFrame()
    cols = ['lib', 'factor_id', 'name_cn', 'base', 'transform', 'window', 'formula_cn',
            'uploaded', 'label', 'coverage', 'A_official', 'LGB', 'sia', 'rank_ic', 'ic',
            'ic_ir', 'win', 'rank_ic_ir', 'rank_ic_win', 'turnover', 'turnover_month',
            'net', 'long', 'sharpe', 'mdd', 'mono', 'direction', 'n_valid_periods', 'family',
            'mi', 'mi_std', 'nmi', 'jsd']
    # 非线性诊断列（2026-09-26）：正交扫描后回填 vwap 行，其余 NaN
    for _f in (refs_df, mined_df, custom_df):
        for _k in ('mi', 'mi_std', 'nmi', 'jsd'):
            _f[_k] = np.nan
    cols_present = [c for c in cols if c in refs_df.columns]
    refs_df = refs_df.reindex(columns=cols_present)
    _mined = mined_df.reindex(columns=cols_present)
    _custom = custom_df.reindex(columns=cols_present) if len(custom_df) else custom_df
    # FutureWarning 抑制（pandas concat dtype 变更提示，不影响结果）
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        df = pd.concat([refs_df, _mined] + ([_custom] if len(_custom) else []),
                       ignore_index=True)
    for c in ('A_official', 'LGB', 'rank_ic', 'rank_ic_ir', 'ic_ir', 'win',
              'rank_ic_win', 'turnover', 'net', 'sharpe', 'mdd', 'mono', 'coverage',
              'mi', 'mi_std', 'nmi', 'jsd'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    n_mined = int(df.loc[df['lib'] == 'mined', 'factor_id'].nunique()) if len(df) else 0
    n_refs = int(df.loc[df['lib'] == 'refs', 'factor_id'].nunique()) if len(df) else 0
    n_custom = int(df.loc[df['lib'] == 'custom', 'factor_id'].nunique()) if len(df) else 0
    payload = {'generated_at': datetime.now().isoformat(timespec='seconds'),
               'top': args.top, 'n_mined_eval': n_mined, 'n_refs': n_refs,
               'n_custom': n_custom, 'keep_pool': args.keep_pool, 'keep_base': args.keep_base,
               'n_total': len(df), 'vwap_orthogonal': False}

    # 榜1/榜2 排序层正交（2026-09-24/25）：共用逐日截面基础设施
    vwap_order: list[int] | None = None
    vwap_rho: dict | None = None
    c2c_order: list[int] | None = None
    vwap_df = df[(df['label'] == 'vwap')].sort_values('LGB', ascending=False)
    c2c_df = df[(df['label'] == 'c2c')].sort_values('A_official', ascending=False)

    from fitness import START, END as FIT_END
    mask_d = ((frame['date'] >= pd.Timestamp(START))
              & (frame['date'] <= pd.Timestamp(FIT_END))).to_numpy()
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

    vwap_nl: dict = {}
    if len(vwap_df) > 20:
        vwap_order, vwap_rho, vwap_nl = vwap_orthogonal_indices(
            vwap_df, frame, ev, builder, pos_me, day_group, specs_by_id=_specs,
            max_scan=args.scan, keep=max(args.keep_pool, args.keep_base),
            cache_dir=cache_dir, val_store=val_store,
            stats=cache_stats, label='vwap', y_me=y_vwap_me)
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
            stats=cache_stats, label='c2c')
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
                                   vwap_order, vwap_df)
        payload['cond_table_n'] = len(cond_table)
        print(f'条件 RankIC：双榜入选者 {len(cond_table)} 个完成', flush=True)
    except Exception as exc:
        cond_table = None
        print(f'warning: 条件 RankIC 汇总失败 {exc}', flush=True)

    # dom 诊断（2026-09-25）：双榜入选者的单字段支配度（快速 Spearman @ pos_me）
    from scipy.stats import spearmanr as _sp

    def _quick_dom(values_me, frame):
        # 2026-09-26：传入已是 pos_me 切片的因子值（来自 val_store 或现算），
        # frame/pos_me 取自闭包。
        v = np.asarray(values_me, dtype=float)
        out = {}
        for f in ('CLOSE', 'VOLUME', 'AMOUNT', 'MARKET_CAP', 'TURNOVER'):
            fv = pd.to_numeric(frame[f].iloc[pos_me], errors='coerce').to_numpy(dtype=float)
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

    _dom_info: dict = {}
    for _tag, _sel_df, _order in (('c2c', c2c_df, c2c_order), ('vwap', vwap_df, vwap_order)):
        if _order is None:
            continue
        for _idx in _order:
            _r = _sel_df.iloc[_idx]
            _key = (_r['lib'], str(_r['factor_id']))
            if _key in _dom_info:
                continue
            try:
                _v = val_store.get(_key)
                if _v is None:
                    if _r['lib'] == 'mined':
                        _node = parse_formula(str(_r['formula_cn']))
                        _v = np.asarray(ev.values(_node), dtype=float)[pos_me]
                    else:
                        _spec = _specs.get(str(_r['factor_id']))
                        if _spec is None:
                            continue
                        _v = builder.get_factor(_spec).to_numpy(dtype=float)[pos_me]
                _dd = _quick_dom(_v, frame)
                if _dd is None:
                    continue
                _mf, _mv = _dd
                _dom_info[_key] = {'max': round(_mv, 2), 'field': _mf}
            except Exception:
                pass
    payload['dom_info'] = {f'{k[0]}_{k[1]}': v for k, v in _dom_info.items()}

    md = build_md(df, payload, vwap_order=vwap_order, vwap_rho=vwap_rho,
                  cond_table=cond_table, c2c_order=c2c_order, dom_info=_dom_info)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = OUT_DIR / f'ALIGNED_TOP2_{ts}.md'
    md_path.write_text(md, encoding='utf-8')
    df.to_csv(OUT_DIR / f'ALIGNED_TOP2_{ts}_flat.csv', index=False, encoding='utf-8-sig')
    # V4.1动态靶标面板：vwap正交顺序按LGB降序、每family最多4个均衡抽取keep_base个
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
        _top20 = []
        for _r in _picked:
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
        _base_dir = HERE.parent.parent / 'results' / 'merge_v2'
        _base_dir.mkdir(parents=True, exist_ok=True)
        (_base_dir / 'vwap_top20.json').write_text(
            json.dumps({'generated_at': payload.get('generated_at'), 'n': len(_top20),
                        'base20': _top20}, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'vwap_top20.json: {len(_top20)} factors', flush=True)
        if args.export_panel and _top20:
            _panel_cols = {'stock_code': frame['stock_code'].to_numpy(),
                           'date': pd.to_datetime(frame['date']).to_numpy()}
            for _r in _picked:
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
    (OUT_DIR / f'ALIGNED_TOP2_{ts}.json').write_text(
        json.dumps(payload, ensure_ascii=False) + '\n' + df.to_json(orient='records', force_ascii=False),
        encoding='utf-8')
    print(f'saved {md_path}  total {time.time() - t0:.0f}s', flush=True)
    print(md[:2000], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
