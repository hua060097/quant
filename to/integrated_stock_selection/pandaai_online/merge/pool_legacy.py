"""平台池 legacy API 恢复（B1，2026-10-10）。

背景：`pool.py` 在 v2 工作中被 v2 实现覆盖写入，legacy 平台池 API 丢失，导致
`run_search` / `selftest_phase_a` / `selftest_supplements` / `pool_matrix*.py`
共 8 个文件不可用。本文件按调用方实测用法 + 存留工件（`pool_sim.py`、
`pool_matrix*.py` docstring、`knowledge.py` 机制条目、平台积分敏感度）
重建 legacy 行为；v2 实现保留在 `pool_v2.py`。

官方池机制（2026-09 完整规则文档逐条核对，见 pool_sim.py 文件头）：
- 合成：截面 1%-99% Winsorize → Z-score → 等权平均，持前 10% 等权；方向 0 取负。
- 宇宙：is_can_buy & 非 ST & 非一字涨停；调仓日 = 交易日历每 cycle 个取 1。
- A：SiA = |RankIC|×|ICIR|×胜率（方向胜率）；rawA = 成员平均；
  NA = min(max(rawA/0.08,0),1) × Decay，并受新人 cap 约束
  （DECAY_TABLE：池龄<6 月 → 1.00/cap 0.70；6-12 月 → 0.90；>12 月 → 0.85）。
- B：样本外，本地无真值；NB = min(max(rawB/0.06,0),1)，rawB 由 b_override 给定
  （平台积分敏感度：+0.001 rawB = 40000×0.35×(0.001/0.06) = +233 分/月）。
- C：本地代理走 pool_sim.pool_backtest 的官方月度口径，NC = min(max(rawC/0.6,0),1)。
- 官方三段式：**Comb = 0.20·NA + 0.35·NB + 0.45·NC**
  （+0.001 rawA = 40000×0.20×(0.001/0.08) = +100 分/月，与平台敏感度自洽）。

与 legacy 原实现的已知不可复原项（诚实声明，非静默近似）：
1. `refresh_member_sia` 的 aligned_v1 窗原为内部常量，此处显式化为
   ALIGNED_START/ALIGNED_END（与 dedup.monthly_spearman 默认窗一致）。
2. selftest G4/S6 的期望值（0.1840/0.1864/0.1953/0.1805）是 **2026-09 的 7~15 人
   历史池**在当时的 pool_state.json 下标定的；pool_state.json 已于 2026-10-03
   换池（amihud120 等被移除，成员 12 人），故这些数字在当前 state 下**必然不可复现**。
   这是数据漂移，不是实现缺陷。判定时须区分二者。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import _paths  # noqa: F401  注入 merge 与 pandaai_online 两条 sys.path（pool_sim 在上一级）

import pool_sim
import windows
from cache_utils import find_stock_features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SD = HERE.parent / 'screen_5d'
FLD = ROOT / 'factor_lab_data'

POOL_STATE = HERE / 'pool_state.json'

# pool_sim.build_panel_daily 的官方宇宙窗（A/C 口径唯一 construction 路径）
WINDOW_START = '2023-03-01'
WINDOW_END = '2026-07-10'
# aligned_v1 同窗化窗（refresh_member_sia 用；与 dedup 默认窗一致）
ALIGNED_START = '2023-06-01'
ALIGNED_END = '2026-07-10'

W_A, W_B, W_C = 0.20, 0.35, 0.45
A_SCALE, A_CLAMP = 0.08, 1.0
B_SCALE = 0.06
KEYS = ('stock_code', 'date')


# ------------------------------------------------------------------ 状态与条目
def load_pool_state(path: Path | str | None = None) -> dict:
    """读 pool_state.json（原实现无参；path 为可选覆盖，测试用）。"""
    p = Path(path) if path is not None else POOL_STATE
    return json.loads(p.read_text(encoding='utf-8'))


def all_entries(state: dict) -> list[dict]:
    """members + reserve 的扁平条目列表，每项至少含 name/dir/sia/panel。"""
    out: list[dict] = []
    for bucket in ('members', 'reserve'):
        for e in state.get(bucket, []) or []:
            item = dict(e)
            item.setdefault('bucket', bucket)
            out.append(item)
    return out


def _entry_map(state: dict) -> dict[str, dict]:
    return {e['name']: e for e in all_entries(state)}


# ------------------------------------------------------------------ 面板装配
def _resolve_panel(panel: str) -> Path | None:
    if not panel or panel == 'inline':
        return None
    if panel.endswith('.parquet'):
        for cand in (SD / panel, FLD / panel):
            if cand.is_file():
                return cand
        return None
    if panel == 'factors':
        p = FLD / 'factors.parquet'
        return p if p.is_file() else None
    return None


def _mktcap_panel(names: list[str]) -> pd.DataFrame | None:
    """cache_log_total_mv：本地无落盘面板，由 cache 的 total_mv 取对数现算。

    口径见 local_mktcap.py:30（log(clip(total_mv,>=1))），dir=0 即小市值。
    """
    if 'mktcap' not in names:
        return None
    sf = pd.read_parquet(find_stock_features(), columns=['stock_code', 'date', 'total_mv'])
    sf['date'] = pd.to_datetime(sf['date'])
    out = sf[['stock_code', 'date']].copy()
    out['mktcap'] = np.log(pd.to_numeric(sf['total_mv'], errors='coerce')
                           .clip(lower=1.0)).to_numpy(dtype=float)
    return out


def build_extra_panel(entries: list[dict]) -> pd.DataFrame:
    """把非 inline 条目的因子列并成一张 (stock_code, date, <name>...) 面板。

    inline 条目不在此求值（由调用方经 evaluator 求值后 merge，见 pool_matrix_oct
    的 build_extra / run_search 的 cand_df 路径）。
    """
    frames: list[pd.DataFrame] = []
    by_panel: dict[str, list[str]] = {}
    for e in entries:
        if e.get('panel') == 'inline':
            continue
        by_panel.setdefault(e.get('panel') or '', []).append(e['name'])

    for panel, names in by_panel.items():
        if panel == 'cache_log_total_mv':
            sub = _mktcap_panel(names)
            if sub is not None:
                frames.append(sub)
            continue
        path = _resolve_panel(panel)
        if path is None:
            continue
        have = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in dict.fromkeys(names) if c in have]
        if not cols:
            continue
        sub = pd.read_parquet(path, columns=list(KEYS) + cols)
        sub['date'] = pd.to_datetime(sub['date'])
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=list(KEYS))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=KEYS, how='outer')
    out['date'] = pd.to_datetime(out['date'])
    return out.drop_duplicates(subset=list(KEYS)).reset_index(drop=True)


# ------------------------------------------------------------------ A / B 代理
def _decay(age_months: float) -> tuple[float, float]:
    return next((d, c) for lim, d, c in pool_sim.DECAY_TABLE if age_months <= lim)


def _na_from_sia(sias: list[float], age_months: float) -> dict:
    rawA = float(np.mean(sias)) if sias else 0.0
    decay, cap = _decay(age_months)
    na = min(max(rawA / A_SCALE, 0.0), A_CLAMP) * decay
    na = min(na, cap)
    return {'rawA': rawA, 'NA': float(na), 'decay': decay, 'cap': cap}


def _nb_from_b(bs: list[float]) -> dict:
    rawB = float(np.mean(bs)) if bs else 0.0
    return {'rawB': rawB, 'NB': float(min(max(rawB / B_SCALE, 0.0), 1.0))}


def _sia_of(entry: dict, sia_override: dict | None) -> float:
    if sia_override and entry['name'] in sia_override:
        v = sia_override[entry['name']]
        if v is not None and np.isfinite(v):
            return float(v)
    v = entry.get('sia')
    return float(v) if v is not None and np.isfinite(float(v)) else 0.0


# ------------------------------------------------------------------ 宇宙与回测
class _Universe:
    """官方宇宙日频面板（build_panel_daily）+ 池回测，按 (extra 列集, cycle) 缓存。"""

    def __init__(self, extra: pd.DataFrame, cycle: int, label: str,
                 data_cut: str | None = windows.DEFAULT_DATA_CUT):
        self.cycle = int(cycle)
        self.label = label
        # 池回测窗按数据截止日裁剪（2026-10-11）：WINDOW_END='2026-07-10' 是历史遗留、
        # 含 2026 数据；默认裁到 DEFAULT_DATA_CUT，CUT_OFF（'off'）恢复遗留行为。
        self.start = WINDOW_START
        self.end = windows.cut_bound(WINDOW_END, data_cut)
        xp = extra.copy()
        xp['date'] = pd.to_datetime(xp['date'])
        self.m, self.reb = pool_sim.build_panel_daily(
            [], self.cycle, self.label, extra_panel=xp,
            start=self.start, end=self.end)

    def comb(self, names: list[str], directions: dict, sias: list[float],
             age_months: float, bs: list[float]) -> dict:
        present = [n for n in names if n in self.m.columns]
        missing = [n for n in names if n not in self.m.columns]
        if not present:
            return {'n_members': 0, 'NC': float('nan'), 'Comb': float('nan'),
                    'Rex': None, 'Turnover': None, 'missing': names,
                    'rawA': 0.0, 'NA': 0.0, 'rawB': 0.0, 'NB': 0.0,
                    'members': [], 'directions': {}}
        bt = pool_sim.pool_backtest(self.m, present, directions, self.reb,
                                    self.label, self.cycle)
        na = _na_from_sia(sias, age_months)
        nb = _nb_from_b(bs)
        nc = float(bt.get('NC_mean', float('nan')))
        comb = W_A * na['NA'] + W_B * nb['NB'] + W_C * nc
        return {'n_members': len(present), 'NC': nc, 'Comb': float(comb),
                'Rex': bt.get('Rex_ann_mean'), 'Turnover': bt.get('Turnover_month_mean'),
                'missing': missing, 'members': present,
                'directions': {n: int(directions.get(n, 1)) for n in present},
                **na, **nb,
                'n_months_scored': bt.get('n_months_scored')}


def _plan(state: dict, names: list[str], b_override: dict | None,
          sia_override: dict | None) -> tuple[dict, list[float], list[float]]:
    """把名单解析成 (directions, sias, bs)，并剔除 extra 里不存在的条目由上层处理。"""
    emap = _entry_map(state)
    dirs: dict[str, int] = {}
    sias: list[float] = []
    for n in names:
        e = emap.get(n, {'name': n, 'dir': 1, 'sia': 0.0})
        dirs[n] = int(e.get('dir', 1))
        sias.append(_sia_of(e, sia_override))
    bs = []
    if b_override:
        for n in names:
            v = b_override.get(n)
            bs.append(float(v) if v is not None and np.isfinite(float(v)) else 0.0)
    return dirs, sias, bs


# ------------------------------------------------------------------ 公开 API
def comb_proxy(sia: dict[str, float], members: list[str], NC: float,
               age_months: float = 3, b_override: dict | None = None) -> dict:
    """纯标量 Comb 代理（无需面板；diag_pool_amihud 的用法）。

    sia: {member_name: SiA}；NC: 上游 pool_backtest 的 NC_mean。
    """
    sias = [float(sia.get(n, 0.0) or 0.0) for n in members]
    na = _na_from_sia(sias, age_months)
    bs = ([float(b_override.get(n, 0.0) or 0.0) for n in members]
          if b_override else [])
    nb = _nb_from_b(bs)
    nc = float(NC) if NC is not None and np.isfinite(float(NC)) else float('nan')
    return {'Comb': float(W_A * na['NA'] + W_B * nb['NB'] + W_C * nc),
            'NC': nc, 'members': list(members), **na, **nb}


def comb_table(extra: pd.DataFrame, sets: dict[str, list[str]], state: dict,
               b_override: dict | None = None,
               sia_override: dict | None = None,
               data_cut: str | None = windows.DEFAULT_DATA_CUT) -> dict[str, dict]:
    """多变体 Comb 表。row 键：n_members/rawA/NA/rawB/NB/NC/Rex/Turnover/Comb。

    `data_cut`：池回测窗上界（默认裁到 DEFAULT_DATA_CUT，见 `_Universe`）。
    """
    cycle = int(state.get('cycle', 3))
    label = state.get('label', 'c2c_3d')
    age = float(state.get('age_months', 3))
    uni = _Universe(extra, cycle, label, data_cut=data_cut)
    table: dict[str, dict] = {}
    for set_name, names in sets.items():
        avail = [n for n in dict.fromkeys(names) if n in uni.m.columns]
        dirs, sias, bs = _plan(state, avail, b_override, sia_override)
        row = uni.comb(avail, dirs, sias, age, bs)
        row['requested'] = list(dict.fromkeys(names))
        table[set_name] = row
    return table


def marginal_sets(state: dict) -> dict[str, list[str]]:
    """G4 边际集：现池 / +mktcap / -highpos120+mktcap / -amihud120+mktcap。

    名单按当前 state 派生；被点名删除的成员若已不在池中，则该变体退化为现池
    （并在 comb_table 的 missing 字段留痕），键名保持稳定以兼容 G4_EXPECTED。
    """
    members = [e['name'] for e in state.get('members', []) or []]

    def without(drop: str) -> list[str]:
        return [m for m in members if m != drop]

    return {
        'P7（现池）': members,
        'P7+mktcap': members + ['mktcap'],
        'P7-highpos120+mktcap': without('highpos120') + ['mktcap'],
        'P7-amihud120+mktcap': without('amihud120') + ['mktcap'],
    }


def leave_one_out(extra: pd.DataFrame, state: dict,
                  b_override: dict | None = None,
                  sia_override: dict | None = None,
                  data_cut: str | None = windows.DEFAULT_DATA_CUT) -> dict:
    """逐成员剔除的 ΔComb 归因。`data_cut` 见 `_Universe`。"""
    cycle = int(state.get('cycle', 3))
    label = state.get('label', 'c2c_3d')
    age = float(state.get('age_months', 3))
    uni = _Universe(extra, cycle, label, data_cut=data_cut)
    members = [e['name'] for e in state.get('members', []) or []]
    avail = [n for n in members if n in uni.m.columns]

    dirs, sias, bs = _plan(state, avail, b_override, sia_override)
    full = uni.comb(avail, dirs, sias, age, bs)

    rows = []
    for i, name in enumerate(avail):
        sub = avail[:i] + avail[i + 1:]
        d2, s2, b2 = _plan(state, sub, b_override, sia_override)
        r = uni.comb(sub, d2, s2, age, b2)
        rows.append({'dropped': name, 'Comb': r['Comb'],
                     'delta': float(r['Comb'] - full['Comb']),
                     'NC': r['NC'], 'Rex': r['Rex'], 'Turnover': r['Turnover']})
    return {'full_comb': full['Comb'], 'full': full, 'rows': rows,
            'missing': full.get('missing', [])}


def greedy_forward(extra: pd.DataFrame, state: dict, cands: list[dict],
                   max_add: int = 3, min_gain: float = 0.001,
                   sia_override: dict | None = None,
                   b_override: dict | None = None,
                   data_cut: str | None = windows.DEFAULT_DATA_CUT) -> dict:
    """只加不换的贪婪 ΔComb（原实现语义；replace/remove 未接线，见 skill_review P1）。

    `data_cut`：池回测窗上界（默认裁到 DEFAULT_DATA_CUT，见 `_Universe`）。
    """
    cycle = int(state.get('cycle', 3))
    label = state.get('label', 'c2c_3d')
    age = float(state.get('age_months', 3))
    uni = _Universe(extra, cycle, label, data_cut=data_cut)

    pool = [e['name'] for e in state.get('members', []) or []]
    reserve_names = {e['name'] for e in state.get('reserve', []) or []}

    def evaluate(names: list[str]) -> dict:
        avail = [n for n in dict.fromkeys(names) if n in uni.m.columns]
        d, s, b = _plan(state, avail, b_override, sia_override)
        return uni.comb(avail, d, s, age, b)

    base = evaluate(pool)
    cur = float(base['Comb'])
    steps: list[dict] = []
    selected: list[str] = []

    cands = [c for c in cands if c.get('name') not in pool]
    for _ in range(int(max_add)):
        best = None
        for c in cands:
            nm = c['name']
            if nm in pool or nm in selected or nm not in uni.m.columns:
                continue
            r = evaluate(pool + [nm])
            gain = float(r['Comb'] - cur)
            if not np.isfinite(gain):
                continue
            if best is None or gain > best[1]:
                best = (nm, gain, r)
        if best is None or best[1] < float(min_gain):
            break
        nm, gain, r = best
        pool.append(nm)
        selected.append(nm)
        cur = float(r['Comb'])
        steps.append({'candidate': nm, 'gain': gain, 'Comb': r['Comb'],
                      'NC': r['NC'], 'Rex': r['Rex'], 'Turnover': r['Turnover'],
                      'n_members': r['n_members']})

    return {'base_comb': base['Comb'], 'final_comb': cur, 'selected': selected,
            'steps': steps, 'final_members': pool,
            'promoted_reserve': [n for n in selected if n in reserve_names]}


def refresh_member_sia(state: dict, cycle: int | None = None,
                       extra: pd.DataFrame | None = None,
                       data_cut: str | None = windows.DEFAULT_DATA_CUT) -> dict:
    """aligned_v1 窗同窗化：重估成员 SiA（RankIC×|ICIR|×方向胜率）。

    评估网格取**各成员自身的日期栅格**并按 cycle 抽稀，而非强制套用 cache 交易日历
    的第 0::cycle 格：qs_factory_panel / qs_r3_panel / new_slow_panel 只落调仓日，
    且其栅格与 cache 日历**零重叠**（实测 skew120/vr120_s20/comp6 在
    2023-06-01~2026-07-10 各 251 个日期，与 cache 派生栅格交集为 0）。
    若强制套 cache 栅格，这三个成员会被误判为 stale。

    返回 {name: sia, ..., 'stale': [无面板/无法求值的成员]}；inline 成员需调用方
    先经 evaluator 求值并并入 extra（与 build_extra_panel 的约定一致）。
    """
    cycle = int(cycle if cycle is not None else state.get('cycle', 3))
    # 窗上界按数据截止日裁剪（2026-10-11）：ALIGNED_END='2026-07-10' 是历史遗留、含 2026。
    aligned_end = windows.cut_bound(ALIGNED_END, data_cut)
    entries = all_entries(state)
    if extra is None:
        extra = build_extra_panel([e for e in entries if e.get('panel') != 'inline'])

    sf = pd.read_parquet(find_stock_features(),
                         columns=['stock_code', 'date', 'close', 'is_limit_up',
                                  'is_can_buy', 'is_st', 'open', 'high'])
    sf['date'] = pd.to_datetime(sf['date'])
    sf = sf.sort_values(['stock_code', 'date'])
    sf['_y'] = sf.groupby('stock_code', sort=False)['close'].shift(-cycle) / sf['close'] - 1.0
    sf = sf[(sf['date'] >= ALIGNED_START) & (sf['date'] <= aligned_end)]
    one_bar = (sf['is_limit_up'] == 1) & (sf['open'] == sf['high'])
    sf = sf[(sf['is_can_buy'] == 1) & (sf['is_st'] == 0) & (~one_bar)]
    lab = sf[['stock_code', 'date', '_y']].dropna(subset=['_y'])

    out: dict = {}
    stale: list[str] = []
    for e in entries:
        nm = e['name']
        if nm not in extra.columns:
            stale.append(nm)
            continue
        d = extra[[*KEYS, nm]].copy()
        d['date'] = pd.to_datetime(d['date'])
        d = d[np.isfinite(pd.to_numeric(d[nm], errors='coerce'))]
        if d.empty:
            stale.append(nm)
            continue
        # 成员自身栅格 ∩ 标签可用 ∩ aligned 窗，按 cycle 抽稀
        grid = sorted(set(d['date']) & set(lab['date']))
        lo, hi = pd.Timestamp(ALIGNED_START), pd.Timestamp(aligned_end)
        grid = [x for x in grid if lo <= x <= hi]
        if len(grid) > cycle:
            grid = grid[::cycle]
        if len(grid) < 2:
            stale.append(nm)
            continue
        m = d[d['date'].isin(set(grid))].merge(lab, on=KEYS, how='inner')
        m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=[nm, '_y'])
        ics = []
        for _, g in m.groupby('date', sort=True):
            if len(g) < 50:
                continue
            a = g[nm].rank().to_numpy(dtype=float)
            b = g['_y'].rank().to_numpy(dtype=float)
            if a.std() == 0 or b.std() == 0:
                continue
            ics.append(float(np.corrcoef(a, b)[0, 1]))
        if not ics:
            stale.append(nm)
            continue
        arr = np.asarray(ics, dtype=float)
        ic = float(arr.mean())
        # ICIR 取 **非年化** mean/std：与 screen_one 落盘的 ic_ir 同量级
        # （screen 读数如 skew120 RIC=-0.0392 / ic_ir=-0.2112 反推 std≈0.186）。
        # 若用 mean/std*sqrt(n) 会整体放大约 √251≈15.8 倍，与 pool_state 存量
        # sia（0.0041~0.0116）严重不符，故此处不年化。残余口径差见 B1 报告。
        ir = float(arr.mean() / arr.std(ddof=1)) if len(arr) > 1 \
            and arr.std(ddof=1) > 0 else 0.0
        dirn = int(e.get('dir', 1))
        win = float(np.mean(np.sign(arr) == (1 if dirn == 1 else -1)))
        out[nm] = float(abs(ic) * abs(ir) * win)
    out['stale'] = stale
    return out


def _state_with(state: dict, members: list[str]) -> dict:
    emap = _entry_map(state)
    return {**state, 'members': [copy.deepcopy(emap[n]) for n in members]}
