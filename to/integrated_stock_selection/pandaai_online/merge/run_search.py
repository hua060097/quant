"""合并引擎入口：搜索/导出/提案/报告/台账/夹具。

搜索默认策略（mechanism-first，2026-09-15 批准；2026-09-24 配额改 UCB）：
  1) 按机制轴 UCB 配额撒种（近 90 天归一 reward + 探索项；冠军轴下限；未测轴乐观初始化）
  2) 一级快速评估（股票抽样 + 近N月窗，结构指纹近克隆预杀——指纹已跨 run 落盘）
  3) 每轴保留精英，精英内变异精化
  4) 幸存者进入全量窗评估（多标签一次评估：因子值只算一次）
  5) 贪婪 ΔComb + 残差 IC 诊断 + 短名单 JSON + Markdown 报告
默认 profile=local-vwap；可用 --profile competition-c2c 运行竞赛扩展。
附加标签用 --label c2c/--label vwap；附加标签只作测量，不自动变成主标签或必过门。

本地 VWAP 为默认 profile；C2C 是可选竞赛适配 profile。平台表达能力只约束 export/upload，
本地 evaluator、grammar、筛选与 vwap 榜保留 local-only/行业上下文候选。

子命令：
  search   见上
  export   从短名单生成 batch.py 的 candidates.txt（只写文件，不发平台请求）
  intake   外部/LLM 公式提案：parse/validate/lint/冒烟/去重 → 台账
  propose  LLM 提案（无API key时生成 inbox 模板）
  report   由短名单 JSON 生成 Markdown 会话报告
  ledger   台账统计
  fixtures 打印回归夹具公式
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
import windows as W
import pool_legacy as pool_mod
from dedup import max_abs_corr, monthly_spearman
from cache_utils import find_stock_features
from evaluator import (Evaluator, ensure_extended_fields, ensure_local_fields,
                         load_panel, value_fingerprint)
from executor import competition_status, estimate_credits, validate_file, write_candidates
from field_catalog import formula_uses_local
from explain import explain
from fitness import (
    DataContext, END, HOLDOUT_START, LABELS, build_fast_frame, c_proxy_of,
    node_count, rank_score, stability_of, tier1, train_holdout,
)
from grammar import (
    _composite_legs, AXIS_OF, axis_families, composite_from_legs, crossover,
    expression_hash, generate_population, mutate, mutate_with_type,
    mutate_window_neighbor, random_tree, render, structural_fingerprint, validate,
)
from intake import INTAKE_MAX_DEPTH, SPARSE_UNIQUE, sparse_unique_median
from ledger import Ledger
from lint import dust_errors, has_errors, lint
from nonlinear import nonlinear_for_ctxs
from parser import ParseError, parse_formula
from resid import resid_metrics

OUT_DIR = Path(__file__).resolve().parent / 'out'


def node_name(family: str, node) -> str:
    return f"MG-{family.split('_')[0]}-{expression_hash(node)[:8]}"


def resolve_search_profile(args) -> tuple[str, str, list[str]]:
    """返回 (profile, primary_label, labels)，C2C 上传能力不参与本地搜索 profile 解析。

    显式 --profile 优先决定 primary；旧式 --label 列表仍兼容（首标签为 primary）；
    完全省略时默认本地 vwap。
    """
    requested = list(getattr(args, 'label', None) or [])
    profile = getattr(args, 'profile', None)
    profile_to_label = {'local-vwap': 'vwap', 'competition-c2c': 'c2c'}
    if profile is None:
        primary = requested[0] if requested else 'vwap'
        profile = 'competition-c2c' if primary == 'c2c' else 'local-vwap'
    else:
        if profile not in profile_to_label:
            raise ValueError(f'unknown search profile {profile!r}')
        primary = profile_to_label[profile]
    labels = list(dict.fromkeys([primary] + requested))
    return profile, primary, labels


def valid_rows_for_label(rows: list[dict], label: str) -> list[dict]:
    """Per-label valid candidates; independent of C2C export capability."""
    return [r for r in rows
            if bool(((r.get('labels') or {}).get(label) or {}).get('valid'))]


def label_search_value(tier: dict, label: str) -> float:
    """Profile-specific search reward; keep the raw C2C SiA intact for competition reports."""
    if label == 'vwap':
        values = [tier.get('rank_ic'), tier.get('rank_ic_ir'), tier.get('rank_ic_win')]
        if all(v is not None and np.isfinite(v) for v in values):
            return float(abs(values[0]) * abs(values[1]) * abs(values[2]))
    return float(tier.get('sia') or 0.0)


def label_search_metric(label: str) -> str:
    return 'rank_sia' if label == 'vwap' else 'sia'


def member_rhos(ctx: DataContext, values, member_panel: pd.DataFrame,
                entries: list[dict],
                data_cut: str | None = W.DEFAULT_DATA_CUT) -> dict:
    base = pd.DataFrame({'stock_code': ctx.stock_codes, 'date': ctx.row_dates,
                         'a': values})
    m = base.merge(member_panel, on=['stock_code', 'date'], how='left')
    out = {}
    for entry in entries:
        col = entry['name']
        if col in m.columns:
            sub = m[['date', 'a', col]].rename(columns={col: 'b'})
            out[col] = monthly_spearman(sub, data_cut=data_cut)
    return out


def _axis_plan(ledger: Ledger, n: int, allow_documented: bool = False,
               ucb_c: float = 0.3, recent_days: int = 90,
               reward_label: str = 'blend',
               data_cut: str | None = None) -> tuple[dict, dict]:
    """机制轴撒种配额：UCB1 式分配（2026-09-24 修订；2026-09-25 reward 双标签化）。

    旧版 w=0.5+0.5×(best-ever/global-best)+未测+0.25 的缺陷：冠军轴永久锁 w=1.0、
    未测轴 w=0.75 反而低于冠军 → 探索被系统性压制；best 还混用 fast/full 两种 SiA 口径。

    reward：search CLI 默认 auto（按 primary label 隔离）；blend 仅在显式选择时混合。
    本地 VWAP 搜索不读取 C2C 竞赛 reward，防止平台目标改变本地机制配额。

    其余不变：
      base = max(近 recent_days 天均值, 0.5×全期均值)；w = base + ucb_c×sqrt(ln(n_total)/n_axis)
      未测轴乐观初始化；证伪轴（n≥20 且全期均值<0.02）w×0.3；数据源 events_min。
    """
    import math
    fams_by_axis = axis_families(allow_documented=allow_documented)
    counts = {axis: 0 for axis in fams_by_axis}
    best = {axis: 0.0 for axis in fams_by_axis}
    axis_rows: dict[str, list] = {a: [] for a in fams_by_axis}
    cutoff = ((datetime.now() - timedelta(days=recent_days)).isoformat(timespec='seconds')
              if recent_days and recent_days > 0 else '')
    _cut = (W.normalize_data_cut(data_cut) if data_cut is not None else None)
    for row in ledger.summary()['events_min']:
        # 行格式：5 元组（旧）或 6 元组（含数据截止日，2026-10-11 起）
        created, family, lab, ev, sia = row[:5]
        if _cut is not None and (row[5] if len(row) > 5 else None) != _cut:
            continue
        axis = AXIS_OF.get(family)
        if axis is None and family.startswith('ref:'):
            axis = family[4:]  # refs 路径事件（2026-09-24 起写入 tier1+语义轴 family）
        if axis not in counts:
            continue
        counts[axis] += 1
        if sia is not None and float(sia) > best[axis]:
            best[axis] = float(sia)
        axis_rows[axis].append((created, lab, ev, sia))
    n_total = max(1, sum(counts.values()))

    _C2C = (None, 'c2c')
    _VWAP = ('vwap',)

    def _group_smax(labs) -> dict:
        smax = {'fast': 0.0, 'local': 0.0}
        for rows in axis_rows.values():
            for _c, lab, ev, sia in rows:
                if (lab in labs) and sia is not None and float(sia) > smax.get(ev, 0.0):
                    smax[ev] = float(sia)
        return smax

    def _reward(rows: list, labs, recent_only: bool, smax: dict) -> float | None:
        pool = [r for r in rows if (r[1] in labs) and (not cutoff or r[0] >= cutoff)] \
            if recent_only and cutoff else [r for r in rows if r[1] in labs]
        vals = [float(sia) / smax[ev] for _c, _l, ev, sia in pool
                if sia is not None and smax.get(ev, 0.0) > 0]
        return sum(vals) / len(vals) if vals else None

    smax_c2c = _group_smax(_C2C)
    smax_vwap = _group_smax(_VWAP)
    all_norms = [float(sia) / smax_c2c[ev]
                 for rows in axis_rows.values() for _c, lab, ev, sia in rows
                 if (lab in _C2C) and sia is not None and smax_c2c.get(ev, 0.0) > 0]
    all_norms += [float(sia) / smax_vwap[ev]
                  for rows in axis_rows.values() for _c, lab, ev, sia in rows
                  if (lab in _VWAP) and sia is not None and smax_vwap.get(ev, 0.0) > 0]
    global_mean = sum(all_norms) / len(all_norms) if all_norms else 0.0

    def _axis_base(rows: list) -> float | None:
        if reward_label == 'c2c':
            return max(_reward(rows, _C2C, True, smax_c2c) or 0.0,
                       0.5 * (_reward(rows, _C2C, False, smax_c2c) or 0.0))
        if reward_label == 'vwap':
            return max(_reward(rows, _VWAP, True, smax_vwap) or 0.0,
                       0.5 * (_reward(rows, _VWAP, False, smax_vwap) or 0.0))
        rc_r = _reward(rows, _C2C, True, smax_c2c)
        rc_a = _reward(rows, _C2C, False, smax_c2c)
        rv_r = _reward(rows, _VWAP, True, smax_vwap)
        rv_a = _reward(rows, _VWAP, False, smax_vwap)
        bc = max(rc_r or 0.0, 0.5 * (rc_a or 0.0)) if (rc_r is not None or rc_a is not None) else None
        bv = max(rv_r or 0.0, 0.5 * (rv_a or 0.0)) if (rv_r is not None or rv_a is not None) else None
        if bc is None and bv is None:
            return None
        if bc is None:
            return bv
        if bv is None:
            return bc
        return 0.5 * bc + 0.5 * bv

    weights = {}
    for a in fams_by_axis:
        rows = axis_rows[a]
        n_axis = counts[a]
        if n_axis == 0:
            w = global_mean + ucb_c * math.log(n_total) ** 0.5
        else:
            base = _axis_base(rows)
            base = base if base is not None else 0.0
            w = base + ucb_c * (math.log(n_total) / n_axis) ** 0.5
            if n_axis >= 20 and (_reward(rows, _C2C, False, smax_c2c) or 0.0) < 0.02 \
                    and (_reward(rows, _VWAP, False, smax_vwap) or 0.0) < 0.02:
                w *= 0.3
        weights[a] = w
    axes = sorted(fams_by_axis)
    if n <= len(axes):
        chosen = sorted(axes, key=lambda a: -weights[a])[:max(1, n)]
        return {a: 1 for a in chosen}, counts
    total = sum(weights.values())
    raw = {a: n * weights[a] / total for a in axes}
    quotas = {a: max(1, int(round(raw[a]))) for a in axes}
    diff = n - sum(quotas.values())
    order = sorted(axes, key=lambda a: raw[a] - quotas[a], reverse=True)
    i = 0
    while diff != 0 and i < 1000:
        a = order[i % len(order)]
        if diff > 0:
            quotas[a] += 1
            diff -= 1
        elif quotas[a] > 1:
            quotas[a] -= 1
            diff += 1
        i += 1
    return quotas, counts


def _load_proposal_rows(path: Path | str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _node_from_row(row: dict):
    text = str(row.get('formula', '')).strip()
    try:
        node = parse_formula(text)
    except ParseError as exc:
        print(f'  proposal skip (parse): {text[:60]} :: {exc}')
        return None
    ok, msg = validate(node, allow_documented=True, strict_const=False,
                       max_depth=INTAKE_MAX_DEPTH)
    if not ok:
        print(f'  proposal skip (validate): {text[:60]} :: {msg}')
        return None
    return node


def _pending_from_ledger(ledger: Ledger, label: str) -> list[tuple]:
    out = []
    for _, view in ledger.latest().items():
        events = view['events']
        if 'proposed' not in events or 'local' in events or 'fast' in events:
            continue
        rec = events['proposed']
        if rec.get('label') not in (None, label):
            continue
        node = _node_from_row({'formula': rec.get('formula', '')})
        if node is not None:
            out.append(('proposal', node, rec.get('hypothesis', '')))
    return out


def _sample_weighted_one(rng: random.Random, pool: list, weight_idx: int = 0) -> int:
    """按权重无放回抽一个，返回下标（权重 = pool[i][weight_idx]，SiA 加权腿库用）。"""
    weights = [max(float(p[weight_idx]), 1e-6) for p in pool]
    r = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= r:
            return i
    return len(pool) - 1


def habitat_fast_score(vals: np.ndarray, labels: np.ndarray,
                        mkt: np.ndarray) -> dict:
    """Class2 栖息地快评代理分（V4.3c）：jsd_tails + 0.5*|牛熊市况 IC 反差|。

    用 pooled Spearman 近似市况反差（快评代理；全量/排榜期用 cond_ic 精确版）。
    返回 {'habitat_score','habitat_jsd','habitat_contrast'}（全 NaN 表弃权）。
    """
    from nonlinear import jsd_tails as _jsd
    v = np.asarray(vals, dtype=float)
    y = np.asarray(labels, dtype=float)
    m = np.asarray(mkt, dtype=float)
    out = {'habitat_score': float('nan'), 'habitat_jsd': float('nan'),
           'habitat_contrast': float('nan')}
    try:
        fin = np.isfinite(v) & np.isfinite(y)
        if fin.sum() < 500:
            return out
        out['habitat_jsd'] = float(_jsd(v[fin], y[fin]))
        up = fin & np.isfinite(m) & (m > 0)
        dn = fin & np.isfinite(m) & (m < 0)
        if up.sum() < 200 or dn.sum() < 200:
            return out
        su = pd.Series(v[up]).corr(pd.Series(y[up]), method='spearman')
        sd = pd.Series(v[dn]).corr(pd.Series(y[dn]), method='spearman')
        if np.isfinite(su) and np.isfinite(sd):
            out['habitat_contrast'] = float(abs(su - sd))
            j = out['habitat_jsd']
            out['habitat_score'] = float((0.0 if not np.isfinite(j) else j)
                                         + 0.5 * out['habitat_contrast'])
    except Exception:
        pass
    return out


class SearchRun:
    def __init__(self, args):
        self.args = args
        self.ledger = _ledger_with_cut(args)
        # 数据截止日（2026-10-11）：候选身份 = 公式 × 数据截止日，且**同时是评估窗上界**
        # （指纹记的截止日必须与实际用到的数据一致，否则记录说谎）。
        # 默认 = 选择窗上界（windows.DEFAULT_DATA_CUT = 2025-12-31）。
        self.data_cut = self.ledger.data_cut
        # 数据版本（2026-10-11）：**只作溯源记录，不参与指纹过滤**。
        # 用户裁定：「指纹还是完全依赖 cut_date，不要随 cache md5 变化」——因为 cache 是
        # 日频重生成的，若以内容哈希为作用键，则每个交易日全部指纹失效，「已测过」机制失去意义。
        # 故这里记的是**声明式**版本（`windows.declared_data_version`，只在显式刷新时变），
        # 写进事件供审计回答「这个数字出自哪个数据世代」；`hashes_for` 等**不传**该键。
        self.data_version = W.declared_data_version("panel")
        self.ledger.data_version = self.data_version
        self.use_legacy_evidence = bool(getattr(args, 'use_legacy_evidence', False))
        self._pool_reported = False
        self.profile, self.primary, self.labels = resolve_search_profile(args)
        for lab in self.labels:
            if lab not in LABELS:
                raise SystemExit(f'unknown label {lab!r}; available: {LABELS}')
        self.secondary = next((lab for lab in self.labels if lab != self.primary), None)
        self.reward_label = str(getattr(args, 'reward_label', 'auto') or 'auto')
        if self.reward_label == 'auto':
            self.reward_label = self.primary
        if self.reward_label not in ('c2c', 'vwap', 'blend'):
            raise SystemExit(f'unknown reward label {self.reward_label!r}')
        self.dual_gate = bool(getattr(args, 'dual_gate', False) and self.secondary)
        self.dual_min_sia = float(getattr(args, 'dual_min_sia', 0.0012))
        self.dual_min_ratio = float(getattr(args, 'dual_min_ratio', 0.25))
        if getattr(args, 'no_turnover_penalty', False):
            self.turnover_ref = None
        elif getattr(args, 'turnover_ref', None) is not None:
            self.turnover_ref = float(args.turnover_ref)
        else:
            # 0.3 是 C2C 大赛 C 口径的官方 floor，不应静默成为本地 VWAP 排序门。
            self.turnover_ref = 0.3 if self.primary == 'c2c' else None
        if getattr(args, 'oos_sia', None) is None:
            # 旧 0.004 holdout 触发阈值来自 C2C 使用习惯；VWAP profile 默认不据此
            # 触发/排序内部 holdout，用户可显式提供自己的阈值。
            args.oos_sia = 0.004 if self.primary == 'c2c' else 0.0
        fast_keep = getattr(args, 'fast_keep_sia', None)
        self.fast_keep_sia = (0.0012 if self.primary == 'c2c' else 0.0) \
            if fast_keep is None else float(fast_keep)
        args.fast_keep_sia = self.fast_keep_sia
        if getattr(args, 'platform_prior', False) and self.primary != 'c2c':
            raise SystemExit('--platform-prior 仅适用于 competition-c2c profile；VWAP 本地排序不使用平台校准先验')
        local_context = getattr(args, 'local_context', None)
        self.local_context = True if local_context is None else bool(local_context)
        args.local_context = self.local_context
        args.allow_documented = bool(getattr(args, 'allow_documented', False))
        frame = load_panel()
        # 层2（2026-09-24）+ 扩展（V4.3 Class2）：--local-context 时在全量面板上预计算
        # MKT_RET/IND_RET/BETA60/DISP + VWAP/IND_EXCESS/BETA20/IVOL20/AMIHUD20，
        # fast 抽样面板由同一 frame 切片继承（市场/行业收益不因抽样而失真）。
        # 默认本地搜索开放上下文字段；--no-local-context 仅用于兼容/对照，不是 C2C 门禁。
        if self.local_context:
            ensure_extended_fields(frame)
            if not args.allow_documented:
                args.allow_documented = True
                print('note: local context enabled: 本地 evaluator 可计算的上下文/文档模板进入搜索；'
                      'C2C 上传能力仅在 export/upload 校验')
        # 评估窗上界 = 数据截止日（2026-10-11）：指纹记的截止日必须与实际用到的
        # 数据一致，否则记录说谎。全量/快评两个 context 与 fast 面板都用该上界。
        _cut_ts = pd.Timestamp(self.data_cut)
        _hol = pd.Timestamp(getattr(args, 'cut', HOLDOUT_START))
        if _hol >= _cut_ts:
            raise SystemExit(
                f'holdout 切点 {_hol.date()} 不早于数据截止日 {_cut_ts.date()}：'
                f'保留窗会为空。请调小 --cut 或调大 --data-cut。')
        # 内存上限：Evaluator LRU 默认 512 个全面板序列（6.6M 行×8B≈53MB/个，
        # 全满约 27GB）。搜索期同节点复用集中在单候选评估内，32 个足够；
        # 纯 memoization 上限，结果逐值不变。
        self.evaluator = Evaluator(frame, cache_size=32)
        self.ctxs = {lab: DataContext(cycle=args.cycle, frame=frame, label=lab,
                                      end=self.data_cut, evaluator=self.evaluator)
                     for lab in self.labels}
        self.ctx = self.ctxs[self.primary]
        if args.fast:
            self.fast_frame = build_fast_frame(frame, n_stocks=args.fast_stocks,
                                               months=args.fast_months,
                                               stratify=(getattr(args, 'fast_stratify', '') or None),
                                               end=self.data_cut)
            self.fast_evaluator = Evaluator(self.fast_frame, cache_size=32)
            end_ts = pd.Timestamp(self.data_cut)
            start = (end_ts - pd.DateOffset(months=args.fast_months)).strftime('%Y%m%d')
            self.fast_ctxs = {lab: DataContext(cycle=args.cycle, start=start,
                                               end=self.data_cut,
                                               frame=self.fast_frame, label=lab,
                                               evaluator=self.fast_evaluator)
                              for lab in self.labels}
        else:
            self.fast_frame = None
            self.fast_ctxs = None
        self.state = pool_mod.load_pool_state()
        self.entries = pool_mod.all_entries(self.state)
        self.member_panel = None
        if args.pool_corr:
            self.member_panel = pool_mod.build_extra_panel(self.entries)
        # 排除集按数据截止日隔离（self.data_cut 已在 __init__ 早段设定）
        self.seen_hash = (set() if args.no_exclude
                          else self.ledger.hashes_for(
                              self.labels, data_cut=self.data_cut,
                              retest_before=getattr(args, 'retest_before', None)))
        # 跨 run 结构指纹（2026-09-24）：fp 组内任一 hash 已在排除集 → 近克隆家族跳过；
        # 此前 seen_fp 只在本次 run 内存生效，跨 run 近克隆会重新全量评估
        self.seen_fp = set() if args.no_exclude \
            else self.ledger.fps_covered(self.seen_hash)
        # 值分布指纹（2026-09-25）：数学/仿射/镜像等价的因子同指纹——
        # 跨 run 装载（fast 面板确定性 → vfp 稳定）；2026-10-11 起按数据截止日隔离
        self.seen_vfp = set() if args.no_exclude \
            else self.ledger.vfps_for(self.data_cut)
        # V4.3c store 预去重：factors.parquet 存量列在**同构 fast 面板**上的 vfp
        # 并入排除集（防重复挖矿；与台账 vfp 同为分布指纹，行序无关可比）。
        if args.fast and not args.no_exclude and not getattr(args, 'no_store_sync', False):
            try:
                from store_sync import sync_store_vfps as _ssv
                _sv = _ssv(frame, n_stocks=args.fast_stocks,
                            months=args.fast_months,
                            stratify=(getattr(args, 'fast_stratify', '') or None),
                            end=self.data_cut, data_cut=self.data_cut)
                self.seen_vfp |= _sv
                print(f'store_vfps: 并入 {len(_sv)} 个存量指纹', flush=True)
            except Exception as exc:
                print(f'warning: store_vfps 同步失败（{str(exc)[:120]}），继续不同步跑',
                      flush=True)
        self.full_hashes = set()
        self.best = -1.0
        self.stall = 0
        self.platform_prior = None
        if getattr(args, 'platform_prior', False):
            cal_path = OUT_DIR / 'calibration.json'
            if cal_path.exists():
                try:
                    self.platform_prior = json.loads(
                        cal_path.read_text(encoding='utf-8'))
                    self._progress(f"platform prior: b={self.platform_prior.get('b')} "
                                   f"n={self.platform_prior.get('n')} "
                                   f"R2={self.platform_prior.get('r2')}")
                except Exception as exc:
                    print(f'warning: calibration.json 读取失败 {exc}')
            else:
                print('warning: --platform-prior 但 out/calibration.json 不存在'
                      '（先跑 platform-ingest + calibrate）')
        self.rng = random.Random(args.seed)
        # fail-sparse-tie 生成期拦截（2026-09-24）：截面独立值中位数<100 的候选不再进全量
        # （此前只拦外部提案；自家 grammar 的 M8/M11 离散门候选同样有平台换手 88% 风险）
        self.sparse_gate = not bool(getattr(args, 'no_sparse_tie_gate', False))
        self.stats = {'fp_skip': 0, 'dup': 0, 'fast': 0, 'full': 0,
                      'sparse_tie': 0, 'sparse_gate': 0, 'lint_reject': 0,
                      'dust_reject': 0, 'redundant_legs': 0, 'vfp_dup': 0}
        self.progress_path = (Path(args.progress_file)
                              if getattr(args, 'progress_file', None) else None)
        if self.progress_path:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            self.progress_path.write_text('', encoding='utf-8')

    def _progress(self, msg: str) -> None:
        line = f'[{datetime.now():%H:%M:%S}] {msg}'
        print(line, flush=True)
        if self.progress_path:
            with self.progress_path.open('a', encoding='utf-8') as f:
                f.write(line + '\n')

    def _leg_redundancy(self, ctx, legs_values: list) -> tuple[float, float]:
        """复合因子腿间截面秩相关（max, mean）——腿值已对齐 ctx 行。

        逐腿按截面 rank（rankdata），逐对闭式 Pearson of ranks → 逐日序列 → 取
        max/mean |ρ|。腿数 ≤6 → ≤15 对，开销可忽略。"""
        from scipy.stats import rankdata
        order = np.argsort(ctx.di, kind="stable")
        d = ctx.di[order]
        D = int(d.max()) + 1 if len(d) else 0
        n_day = np.bincount(d, minlength=D)
        seg = np.concatenate([[0], np.cumsum(n_day)])
        ranks = []
        for v in legs_values:
            vv = np.asarray(v, dtype=float)[order]
            r = np.full(len(vv), np.nan)
            for k in range(D):
                a, b = int(seg[k]), int(seg[k + 1])
                ok = np.isfinite(vv[a:b])
                if ok.sum() < 30:
                    continue
                r[np.flatnonzero(ok) + a] = rankdata(vv[a:b][ok]) / float(ok.sum())
            ranks.append(r)
        abss = []
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                ri, rj = ranks[i], ranks[j]
                ok = np.isfinite(ri) & np.isfinite(rj)
                if ok.sum() < 30:
                    continue
                dok, ri_ok, rj_ok = d[ok], ri[ok], rj[ok]
                nn = np.maximum(np.bincount(dok, minlength=D), 1)
                s_i = np.bincount(dok, weights=ri_ok, minlength=D)
                s_j = np.bincount(dok, weights=rj_ok, minlength=D)
                s_ii = np.bincount(dok, weights=ri_ok * ri_ok, minlength=D)
                s_jj = np.bincount(dok, weights=rj_ok * rj_ok, minlength=D)
                s_ij = np.bincount(dok, weights=ri_ok * rj_ok, minlength=D)
                with np.errstate(invalid='ignore', divide='ignore'):
                    cov = s_ij - s_i * s_j / nn
                    den = np.sqrt(np.maximum(s_ii - s_i * s_i / nn, 0.0)
                                  * np.maximum(s_jj - s_j * s_j / nn, 0.0))
                    corr = np.where((den > 0) & (nn >= 30), cov / np.maximum(den, 1e-12),
                                    np.nan)
                fin = corr[np.isfinite(corr)]
                if len(fin):
                    abss.append(float(np.abs(fin).mean()))
        if not abss:
            return float('nan'), float('nan')
        return float(max(abss)), float(sum(abss) / len(abss))

    def _note_score(self, score: float) -> bool:
        if score > self.best + 1e-6:
            self.best, self.stall = score, 0
            return False
        self.stall += 1
        return bool(self.args.stall and self.stall >= self.args.stall)

    def _screens(self, ctxs: dict, node, name: str) -> dict:
        return {lab: ctxs[lab].screen(node, name=name) for lab in self.labels}

    def _dual(self, tiers: dict) -> dict:
        """次标签诊断；profile 的 primary 独立决定主 shortlist。

        默认 dual_gate=False：附加标签照算并入库，但不参与 primary 筛选。若显式 --dual-gate，
        则将 primary 与第一个附加标签作用户指定的双标签筛选。
        """
        sec = tiers.get(self.secondary) or {}
        pri = tiers.get(self.primary) or {}
        sec_sia = label_search_value(sec, self.secondary) if self.secondary else 0.0
        pri_sia = label_search_value(pri, self.primary)
        ratio = (sec_sia / pri_sia) if pri_sia > 0 else 0.0
        ok = bool(self.dual_gate and sec_sia >= self.dual_min_sia
                  and ratio >= self.dual_min_ratio)
        return {'pass_dual': ok, 'dual_sia': sec_sia, 'dual_ratio': ratio}

    def _nonlinear(self, node) -> dict:
        """搜索期非线性诊断（2026-09-26）：full 阶段对各标签算月度 MI / 尾部 JSD。

        与 build_top 同口径（逐日截面）：复用 Evaluator 节点缓存（screens 已求值），
        只新增直方图统计（~0.3s/候选/标签）。run_fast 阶段不算（保持快评速度）。
        """
        return nonlinear_for_ctxs(self.ctxs, self.labels, node)

    def evaluate_fast(self, family: str, node, source: str = 'seed'):
        h = expression_hash(node)
        if h in self.seen_hash:
            self.stats['dup'] += 1
            return None
        fp = structural_fingerprint(node)
        if fp in self.seen_fp:
            self.stats['fp_skip'] += 1
            return None
        self.seen_hash.add(h)
        self.seen_fp.add(fp)
        name = node_name(family, node)
        # 语义 lint（2026-09-25）：量纲错误等 error 级问题在快评前拦截——此前只有
        # intake 跑 lint，grammar 变异（子树替换/换字段）能造出 CLOSE−AMOUNT 这类
        # 量纲错误的退化表达式（数值上 ≈ −AMOUNT，"碰巧能用"但字面无金融含义）
        _lint = lint(node)
        if has_errors(_lint):
            self.stats['lint_reject'] = self.stats.get('lint_reject', 0) + 1
            return None
        ctxs = self.fast_ctxs or self.ctxs
        _pri = ctxs[self.primary]
        # 数值尘埃（2026-09-25）：ADD/SUB 两侧实测数量级差 ≥100× → 小项为装饰
        _de = dust_errors(node, _pri.evaluator.magnitude)
        if _de:
            self.stats['dust_reject'] = self.stats.get('dust_reject', 0) + 1
            return None
        # 腿间冗余（2026-09-25）：复合结构逐对腿截面秩相关过高 → 冗余堆叠，拒绝
        _legs = _composite_legs(node)
        if _legs and len(_legs) >= 2:
            try:
                _leg_vals = [_pri.values(leg) for leg in _legs]
                _maxp, _meanp = self._leg_redundancy(_pri, _leg_vals)
                if np.isfinite(_meanp) and (_meanp > 0.7 or _maxp > 0.9):
                    self.stats['redundant_legs'] = self.stats.get('redundant_legs', 0) + 1
                    return None
            except Exception:
                pass
        # 值分布指纹（2026-09-25）：数学/仿射/镜像等价 → 同指纹，dup 预杀
        vfp = value_fingerprint(_pri.values(node))
        if vfp:
            if vfp in self.seen_vfp:
                self.stats['vfp_dup'] = self.stats.get('vfp_dup', 0) + 1
                return None
            self.seen_vfp.add(vfp)
        screens = self._screens(ctxs, node, name)
        tiers = {lab: tier1(m) for lab, m in screens.items()}
        dual = self._dual(tiers)
        stability = stability_of(screens[self.primary])
        pri = ctxs[self.primary]
        sparse_median = sparse_unique_median(pri.values(node), pri.row_dates)
        sparse_hit = 0 < sparse_median < SPARSE_UNIQUE
        if sparse_hit:
            self.stats['sparse_tie'] += 1
        primary_search_value = label_search_value(tiers[self.primary], self.primary)
        score_tier = {**tiers[self.primary], 'sia': primary_search_value}
        score = rank_score(score_tier, node,
                           turnover_ref=self.turnover_ref)
        score_nopen = rank_score(score_tier, node, turnover_ref=None)
        # V4.3c Class2 栖息地快评通道：线性 SiA 外的第二晋级依据（尾部发散+市况反差）。
        hab = {'habitat_score': float('nan'), 'habitat_jsd': float('nan'),
               'habitat_contrast': float('nan')}
        if getattr(self.args, 'habitat_track', False):
            try:
                _hv = np.asarray(_pri.values(node), dtype=float)
                _pos = _pri._pos
                _mk = np.asarray(_pri.evaluator.frame['MKT_RET'].to_numpy(dtype=float)[_pos],
                                 dtype=float)
                hab = habitat_fast_score(_hv[_pos], _pri.labels, _mk)
            except Exception:
                pass
        for lab in self.labels:
            search_value = label_search_value(tiers[lab], lab)
            self.ledger.log('fast', node, int(screens[lab].get('direction', 1)),
                            family=family, label=lab, struct_fp=fp,
                            data_cut=self.data_cut,
                            data={'fast': screens[lab], 'tier1': tiers[lab],
                                  'search_value': search_value,
                                  'search_metric': label_search_metric(lab),
                                  'source': source, 'dual': dual,
                                  'stability': stability, 'vfp': vfp,
                                  'sparse_tie_median': sparse_median,
                                  'sparse_hit': sparse_hit})
        self.stats['fast'] += 1
        if sparse_hit and self.sparse_gate:
            self.stats['sparse_gate'] += 1
            return None
        return {'name': name, 'family': family, 'formula': render(node),
                'expression_hash': h, 'node': node, 'source': source, 'vfp': vfp,
                'fast': tiers[self.primary], 'fast_labels': tiers,
                'fast_search_value': primary_search_value,
                'fast_score': score, 'fast_score_nopen': score_nopen,
                'stability': stability, 'sparse_tie_median': sparse_median,
                'sparse_hit': sparse_hit, **dual, **hab}

    def evaluate_full(self, rec: dict) -> dict:
        args = self.args
        node = rec['node']
        h = expression_hash(node)
        if h in self.full_hashes:
            return None
        self.full_hashes.add(h)
        name = rec['name']
        screens = self._screens(self.ctxs, node, name)
        tiers = {lab: tier1(m) for lab, m in screens.items()}
        nl = self._nonlinear(node)
        for lab, t in tiers.items():
            d = nl.get(lab) or {}
            t['mi'] = d.get('mi')
            t['mi_std'] = d.get('mi_std')
            t['nmi'] = d.get('nmi')
            t['jsd'] = d.get('jsd')
        primary = tiers[self.primary]
        primary_search_value = label_search_value(primary, self.primary)
        rhos, rho_max, rho_member = {}, None, None
        if self.member_panel is not None:
            values = self.ctx.values(node)
            rhos = member_rhos(self.ctx, values, self.member_panel, self.entries,
                               data_cut=self.data_cut)
            rho_max, rho_member = max_abs_corr(rhos)
        direction = int(screens[self.primary].get('direction', 1))
        train = holdout = None
        if primary['valid'] and args.oos_sia and primary_search_value >= args.oos_sia:
            split = train_holdout(self.ctx, node, name=name, cut=args.cut)
            train = tier1(split['train'])
            holdout = tier1(split['holdout'])
            direction = int(split['direction'])
        rm = None
        if args.resid and primary['valid']:
            try:
                rm = resid_metrics(self.ctx, node, name=name)
            except Exception as exc:
                rm = {'error': str(exc)[:160]}
        train_search_value = label_search_value(train, self.primary) if train else None
        holdout_search_value = label_search_value(holdout, self.primary) if holdout else None
        base = train_search_value if train_search_value is not None else primary_search_value
        oos = holdout_search_value
        drop = (rm or {}).get('drop')
        dual = self._dual(tiers)
        stability = stability_of(screens[self.primary])
        prior = None
        if self.platform_prior is not None and self.platform_prior.get('b') is not None:
            tm = primary.get('turnover_month')
            if tm is not None and np.isfinite(tm) and float(tm) > 0:
                prior = float(np.clip(float(tm) ** float(self.platform_prior['b']),
                                      0.5, 2.0))
        score = rank_score({'sia': base, 'turnover_month': primary.get('turnover_month'),
                            'turnover': primary.get('turnover')},
                           node, rho_max, oos, drop, args.resid_weight,
                           turnover_ref=self.turnover_ref, platform_prior=prior)
        score_nopen = rank_score({'sia': base, 'turnover_month': primary.get('turnover_month'),
                                   'turnover': primary.get('turnover')},
                                  node, rho_max, oos, drop, args.resid_weight,
                                  turnover_ref=None)
        for lab in self.labels:
            search_value = label_search_value(tiers[lab], lab)
            extra = ({'train': train, 'holdout': holdout, 'pool_corr': rhos,
                      'score': score, 'score_nopen': score_nopen, 'stability': stability,
                      'search_value': search_value,
                      'search_metric': label_search_metric(lab),
                      'resid': rm, 'dual': dual,
                      'habitat_score': rec.get('habitat_score'),
                      'habitat_jsd': rec.get('habitat_jsd'),
                      'habitat_contrast': rec.get('habitat_contrast'),
                      'source': rec.get('source', 'template')}
                      if lab == self.primary else
                      {'search_value': search_value,
                       'search_metric': label_search_metric(lab),
                       'source': rec.get('source', 'template')})
            self.ledger.log('local', node, int(screens[lab].get('direction', 1)),
                            family=rec['family'], label=lab,
                            struct_fp=structural_fingerprint(node),
                            data_cut=self.data_cut,
                            data={'metrics': screens[lab], 'tier1': tiers[lab],
                                  **extra})
        self.stats['full'] += 1
        return {'name': name, 'family': rec['family'], 'formula': render(node),
                'expression_hash': h, 'direction': direction, 'node': node,
                'label_directions': {lab: int(screens[lab].get('direction', 1))
                                     for lab in self.labels},
                 'source': rec.get('source', 'template'), 'valid': primary['valid'],
                 'sia': primary['sia'], 'train': train, 'holdout': holdout,
                 'search_metric': label_search_metric(self.primary),
                 'search_value': primary_search_value,
                 'train_search_value': train_search_value,
                 'holdout_search_value': holdout_search_value,
                'turnover': primary['turnover'],
                'turnover_month': primary.get('turnover_month'),
                'c_proxy': primary.get('c_proxy'),
                'stability': stability,
                'net': primary['net'],
                'mono': primary['mono'], 'labels': tiers,
                'rho_max': rho_max, 'rho_member': rho_member,
                'nodes': node_count(node), 'score': score, 'score_nopen': score_nopen,
                **dual,
                'resid': rm, 'fast_score': rec.get('fast_score'),
                'explain': explain(node, direction)}

    def evaluate_full_custom(self, rec: dict) -> dict | None:
        """Python型自定义因子全量评估（无AST，不进grammar指纹；用log_raw留痕）。

        rec含custom_values（与frame对齐）。按各label DataContext.screen求值，
        主label记search_value/score（turnover惩罚按profile），计入FDR分母
        （tested_pairs经log_raw的local事件进入）。
        """
        args = self.args
        values = np.asarray(rec.get('custom_values'), dtype=float)
        if len(values) != len(self.evaluator.frame):
            return None
        name = rec.get('name', 'custom:unnamed')
        tiers = {}
        screens = {}
        for lab in self.labels:
            ctx = self.ctxs[lab]
            m = ctx.screen(values, name=name)
            screens[lab] = m
            tiers[lab] = tier1(m)
        primary = tiers[self.primary]
        if not primary.get('valid'):
            return None
        primary_search_value = label_search_value(primary, self.primary)
        score = rank_score({'sia': primary_search_value,
                            'turnover_month': primary.get('turnover_month'),
                            'turnover': primary.get('turnover')},
                           __import__('grammar').field('CLOSE'),
                           turnover_ref=self.turnover_ref)
        for lab in self.labels:
            self.ledger.log_raw('local', rec.get('formula', f"py:{rec.get('custom_id', '?')}"),
                                direction=int(screens[lab].get('direction', 1)),
                                family=rec.get('family', 'custom'),
                                label=lab, data_cut=self.data_cut,
                                data={'metrics': screens[lab], 'tier1': tiers[lab],
                                      'search_value': label_search_value(tiers[lab], lab),
                                      'search_metric': label_search_metric(lab),
                                      'source': 'custom',
                                      'custom_id': rec.get('custom_id'),
                                      'score': score if lab == self.primary else None})
        self.stats['full'] += 1
        return {'name': name, 'family': rec.get('family', 'custom'),
                'formula': rec.get('formula', ''), 'direction': int(screens[self.primary].get('direction', 1)),
                'node': None, 'source': 'custom', 'valid': True,
                'sia': primary['sia'], 'labels': tiers,
                'search_value': primary_search_value, 'score': score,
                'turnover': primary['turnover'], 'net': primary['net']}

    def _leg_library(self, max_legs: int = 40, max_turn: float = 0.20) -> list:
        """腿库：台账中当前 primary label 已评估、单次换手≤20% 的慢腿，按对应 SiA 排序。

        2026-09-24 起从台账 sidecar 的 evidence_pool 读取（免 17MB 全量 JSON 扫）；
        过滤语义与旧实现逐条一致：fast/local、目标标签、换手过滤先于首见 hash 去重、
        node_count≤12。SiA 加权抽样与多样性去重在 _seeds 消费端做。
        2026-10-11：证据池按**数据截止日**隔离（旧 winner 系遗留窗口选出，不得复用）。
        """
        out, seen = [], set()
        for entry in self._evidence_pool():
            h, turn = entry['hash'], entry.get('turnover')
            if turn is None or float(turn) > max_turn or h in seen:
                continue
            seen.add(h)
            try:
                node = parse_formula(entry['formula'])
            except Exception:
                continue
            if node_count(node) > 12:
                continue
            out.append((float(entry['sia']), node, int(entry.get('direction', 1))))
        out.sort(key=lambda x: -x[0])
        return out[:max_legs]

    def _evidence_pool(self) -> list:
        """按数据截止日取证据池（腿库/精英回流的原料）。

        默认只取当前截止日的证据：历史 winner 是用**遗留截止日**（含 2026 数据）
        选出来的，直接复用会把旧的选择偏差带进新搜索。`--use-legacy-evidence`
        恢复全量（历史重放用）。首次调用会打印被排除的条数，避免"池子变空"被误读。
        """
        if self.use_legacy_evidence:
            return self.ledger.evidence_pool_for(None, label=self.primary)
        pool = self.ledger.evidence_pool_for(self.data_cut, label=self.primary)
        if not self._pool_reported:
            self._pool_reported = True
            total = len(self.ledger.summary().get('evidence_pool') or [])
            if total != len(pool):
                self._progress(
                    f'evidence pool: {len(pool)}/{total} 条属于 '
                    f'(label={self.primary}, data_cut={self.data_cut}, data_version={self.data_version})'
                    f'（其余为其他口径，已排除；--use-legacy-evidence 可全量）')
        return pool

    def _axis_winners(self, per_axis: int = 6) -> dict[str, list]:
        """各轴历史 winner（sidecar evidence_pool 按 SiA 降序 top，供精英回流撒种）。"""
        fams_by_axis = axis_families(
            allow_documented=getattr(self.args, 'allow_documented', False))
        best: dict[str, list] = {}
        for entry in self._evidence_pool():
            axis = AXIS_OF.get(entry['family'])
            if axis in fams_by_axis:
                best.setdefault(axis, []).append(entry)
        out = {}
        for axis, rows in best.items():
            rows.sort(key=lambda r: -r['sia'])
            out[axis] = rows[:per_axis]
        return out

    def _seeds(self) -> list[tuple]:
        args = self.args
        if args.strategy == 'random':
            return [(fam, node) for fam, node in
                    generate_population(args.n, args.seed, exclude=self.seen_hash,
                                        allow_documented=args.allow_documented)]
        quotas, counts = _axis_plan(
            self.ledger, args.n, allow_documented=args.allow_documented,
            ucb_c=float(getattr(args, 'ucb_c', 0.3)),
            recent_days=int(getattr(args, 'axis_recent_days', 90)),
            reward_label=self.reward_label,
            data_cut=self.data_cut)
        fams_by_axis = axis_families(allow_documented=args.allow_documented)
        print('axis plan: ' + ' '.join(
            f'{axis}:{q}(tested {counts[axis]})' for axis, q in sorted(quotas.items())),
            flush=True)
        legs = self._leg_library() if getattr(args, 'leg_library', True) else []
        if legs:
            print(f'leg library: {len(legs)} legs (slow, tested)', flush=True)
        elite_frac = (0.0 if getattr(args, 'no_elite_seed', False)
                      else float(getattr(args, 'elite_seed_frac', 0.25)))
        winners = self._axis_winners() if elite_frac > 0 else {}
        n_elite = 0
        seeds, local_fp = [], set()
        for axis, quota in sorted(quotas.items()):
            fams = fams_by_axis[axis]
            if axis == 'context' and not getattr(args, 'local_context', False):
                continue  # 显式 --no-local-context 用于对照/兼容，不是 C2C 搜索门
            axis_winners = winners.get(axis) or []
            for _ in range(quota):
                picked = None
                for _try in range(10):
                    node, fam = None, None
                    if (axis_winners and self.rng.random() < elite_frac):
                        # 精英回流（2026-09-24）：历史 winner 单次变异 → 近邻探索；
                        # grammar.generate_population 的 elites 参数由此路径等效替代
                        row = self.rng.choice(axis_winners)
                        try:
                            wnode = parse_formula(row['formula'])
                        except Exception:
                            wnode = None
                        if wnode is not None:
                            cand = mutate(wnode, self.rng, args.mutation_rate,
                                            allow_documented=args.allow_documented)
                            if (validate(cand, allow_documented=args.allow_documented)[0]
                                    and expression_hash(cand) not in self.seen_hash):
                                node, fam = cand, row['family']
                                n_elite += 1
                    if node is None:
                        if (axis == 'composite' and len(legs) >= 2
                                and self.rng.random() < 0.5):
                            k = self.rng.choice((2, 3, 4))
                            # SiA 加权无放回抽样 + 结构指纹多样性（同族近亲腿不同抽）
                            pool = list(legs)
                            target = min(k, len(pool))
                            chosen, leg_fps = [], set()
                            while len(chosen) < target and pool:
                                pick = _sample_weighted_one(self.rng, pool)
                                sia_i, node_i, d_i = pool.pop(pick)
                                fp_i = structural_fingerprint(node_i)
                                if fp_i in leg_fps:
                                    continue
                                leg_fps.add(fp_i)
                                chosen.append((node_i, d_i))
                            node = (composite_from_legs(chosen, self.rng)
                                    if len(chosen) >= 2 else None)
                            fam = 'M9_composite'
                        else:
                            fam = self.rng.choice(fams)
                            _, node = random_tree(self.rng, family=fam,
                                                  allow_documented=args.allow_documented)
                    if node is None:
                        continue
                    fp = structural_fingerprint(node)
                    if fp not in local_fp:
                        local_fp.add(fp)
                        picked = (fam, node)
                        break
                if picked is not None:
                    seeds.append(picked)
        if getattr(args, 'local_context', False):
            # 本地上下文探索保底配额：不让早期历史 reward 把 context 机制轴挤出搜索。
            n_ctx = max(2, args.n // 8)
            added = 0
            for _ in range(n_ctx * 6):
                if added >= n_ctx:
                    break
                fam_c, node_c = random_tree(self.rng, family='M14_context',
                                            allow_documented=True)
                fp_c = structural_fingerprint(node_c)
                if fp_c in local_fp or expression_hash(node_c) in self.seen_hash:
                    continue
                local_fp.add(fp_c)
                seeds.append((fam_c, node_c))
                added += 1
            if added:
                print(f'context seeding: +{added} M14_context（本地上下文候选）',
                      flush=True)
        if n_elite:
            print(f'elite seeding: {n_elite}/{len(seeds)} seeds from ledger winners '
                  f'(frac={elite_frac})', flush=True)
        return seeds

    def run(self) -> dict:
        args = self.args
        t0 = time.time()
        proposal_nodes = []
        n_profile_mismatch = 0
        for path in (args.proposals or []):
            for row in _load_proposal_rows(path):
                source_profile = row.get('profile')
                if source_profile and source_profile != self.profile:
                    n_profile_mismatch += 1
                    continue
                node = _node_from_row(row)
                if node is not None:
                    proposal_nodes.append(('proposal', node, str(row.get('hypothesis', ''))))
        if args.from_ledger:
            proposal_nodes += _pending_from_ledger(self.ledger, self.primary)
        if getattr(args, 'include_custom', False):
            try:
                from custom_registry import evaluate_custom_factor as _ecf
                from custom_registry import load_all_custom_specs as _lcs
                _specs = _lcs()
                _tested = self.ledger.hashes_for(self.labels, data_cut=self.data_cut)
                _n_custom = 0
                for _fid, _spec in sorted(_specs.items()):
                    try:
                        if _spec.kind == 'formula':
                            from parser import parse_formula as _pf
                            from grammar import expression_hash as _eh
                            _node = _pf(_spec.formula_str)
                            if _eh(_node) in _tested:
                                continue
                            proposal_nodes.append((_spec.family, _node,
                                                   _spec.description or _fid))
                        else:
                            _vals = _ecf(_spec, self.evaluator.frame, self.evaluator)
                            _rec = {'name': f"custom:{_fid}", 'family': _spec.family,
                                    'formula': _spec.formula_str, 'node': None,
                                    'source': 'custom', 'custom_id': _fid,
                                    'custom_values': np.asarray(_vals, dtype=float)}
                            # Python型无AST，直接走全量评估通道（evaluate_full_custom）
                            _rec = self.evaluate_full_custom(_rec)
                            if _rec:
                                _n_custom += 1
                    except Exception:
                        continue
                self._progress(f'custom: loaded {len(_specs)} specs, '
                               f'formula-type queued, python-type evaluated {_n_custom}')
            except Exception as _e:
                self._progress(f'warning: custom registry failed {str(_e)[:120]}')
        self._progress(f'start seed={args.seed} strategy={args.strategy} '
                       f'labels={self.labels} allow_documented={args.allow_documented} '
                       f'n={args.n} full_cap={args.full_cap} proposals={len(proposal_nodes)} '
                       f'fast={"on" if args.fast else "off"}')
        if n_profile_mismatch:
            self._progress(f'proposal scope: skipped {n_profile_mismatch} explicitly tagged for another profile')

        fast_rows = []
        seed_total = args.n
        for i, (fam, node) in enumerate(self._seeds(), 1):
            rec = self.evaluate_fast(fam, node, 'seed')
            if rec:
                fast_rows.append(rec)
                self._progress(
                    f'[seed {i}/{seed_total}] {rec["name"]} fscore={rec["fast_score"]:.4f} '
                    f'{label_search_metric(self.primary)}={rec["fast_search_value"]:.4f} '
                    f'dual={rec.get("pass_dual")} '
                    f'elapsed={time.time() - t0:.0f}s')
                if self._note_score(rec['fast_score']):
                    self._progress(f'stall stop at seeds ({self.stall})')
                    break
        mut_stats: dict = {}
        if args.strategy == 'mechanism' and fast_rows and args.refine_children > 0:
            # T6（2026-09-25）：编辑分支成败统计（mini-AlphaMemo）——持久化于
            # 按目标标签隔离；C2C 历史文件沿用旧路径，VWAP 不消费 C2C 奖励。
            mut_path = (OUT_DIR / 'mutation_stats.json' if self.primary == 'c2c'
                        else OUT_DIR / f'mutation_stats_{self.primary}.json')
            if mut_path.exists():
                try:
                    mut_stats = json.loads(mut_path.read_text(encoding='utf-8'))
                except Exception:
                    mut_stats = {}
            mut_prefer = {b: float(np.clip(1.0 + 50.0 * (s.get('sum_gain', 0.0)
                                                        / max(1, s.get('n', 1))), 0.2, 3.0))
                          for b, s in mut_stats.items()}
            elites: dict = {}
            for rec in fast_rows:
                elites.setdefault(rec['family'], []).append(rec)
            stopped = False
            for fam, recs in sorted(elites.items()):
                recs.sort(key=lambda r: -r['fast_score'])
                for elite in recs[:args.refine_elites]:
                    for _ci in range(args.refine_children):
                        branch = 'mutate'
                        if (getattr(args, 'crossover', True) and len(recs) >= 2
                                and self.rng.random() < getattr(args, 'crossover_rate', 0.3)):
                            other = self.rng.choice([r for r in recs if r is not elite])
                            child = crossover(elite['node'], other['node'], self.rng,
                                              allow_documented=args.allow_documented)
                            branch = 'crossover'
                        elif _ci == 0:
                            # T5：窗口邻域精调优先（围绕精英现值的局部收敛）
                            child = mutate_window_neighbor(elite['node'], self.rng)
                            branch = 'win_neighbor'
                        else:
                            child, branch = mutate_with_type(
                                elite['node'], self.rng, args.mutation_rate,
                                prefer=mut_prefer,
                                allow_documented=args.allow_documented)
                        ok, _ = validate(child, allow_documented=args.allow_documented)
                        if not ok:
                            continue
                        rec = self.evaluate_fast(fam, child, 'refine')
                        if rec:
                            _st = mut_stats.setdefault(branch, {'n': 0, 'sum_gain': 0.0})
                            _st['n'] += 1
                            _st['sum_gain'] += float(rec['fast_score']) - float(elite['fast_score'])
                        if rec:
                            fast_rows.append(rec)
                            self._progress(
                                f'[refine] {rec["name"]} fscore={rec["fast_score"]:.4f} '
                                f'{label_search_metric(self.primary)}={rec["fast_search_value"]:.4f} '
                                f'dual={rec.get("pass_dual")} '
                                f'elapsed={time.time() - t0:.0f}s')
                            if self._note_score(rec['fast_score']):
                                self._progress(f'stall stop at refine ({self.stall})')
                                stopped = True
                                break
                    if stopped:
                        break
                if stopped:
                    break

        dual_ok = (lambda r: bool(r.get('pass_dual', True))) if self.dual_gate \
            else (lambda r: True)
        n_dual_fast_rej = sum(1 for r in fast_rows if not dual_ok(r))
        if self.dual_gate and n_dual_fast_rej:
            self._progress(f'dual gate: fast rejected {n_dual_fast_rej} candidates '
                           f'(secondary={self.secondary} min_sia={self.dual_min_sia} '
                           f'min_ratio={self.dual_min_ratio})')
        if args.fast:
            keep = [r for r in fast_rows
                    if r.get('fast_search_value', 0) >= self.fast_keep_sia and dual_ok(r)]
            if args.fast_top_frac > 0 and fast_rows:
                ranked_fast = sorted([r for r in fast_rows if dual_ok(r)],
                                     key=lambda r: -r['fast_score'])
                keep += ranked_fast[:int(len(fast_rows) * args.fast_top_frac)]
            seen, survivors = set(), []
            for rec in sorted(keep, key=lambda r: -r['fast_score']):
                if rec['expression_hash'] in seen:
                    continue
                seen.add(rec['expression_hash'])
                survivors.append(rec)
            survivors = survivors[:args.full_cap]
            # V4.3c 栖息地通道：habitat_score 前列额外晋级（防线性 IC 误杀）；
            # 名额 = full_cap*habitat_frac（默认 15%），与线性幸存者去重后追加。
            n_hab = 0
            if getattr(args, 'habitat_track', False):
                _hq = max(1, int(round(args.full_cap
                                       * float(getattr(args, 'habitat_frac', 0.15)))))
                _cand = [r for r in fast_rows
                         if dual_ok(r) and r['expression_hash'] not in seen
                         and r.get('habitat_score') is not None
                         and np.isfinite(r['habitat_score'])]
                for rec in sorted(_cand, key=lambda r: -r['habitat_score'])[:_hq]:
                    seen.add(rec['expression_hash'])
                    survivors.append(rec)
                    n_hab += 1
            self.stats['habitat_extra'] = self.stats.get('habitat_extra', 0) + n_hab
        else:
            survivors = [r for r in fast_rows if dual_ok(r)][:args.full_cap]
        self._progress(f'fast tested={self.stats["fast"]} fp_skip={self.stats["fp_skip"]} '
                       f'dup={self.stats["dup"]} vfp_dup={self.stats["vfp_dup"]} '
                       f'lint={self.stats["lint_reject"]} dust={self.stats["dust_reject"]} '
                       f'red_legs={self.stats["redundant_legs"]} sparse={self.stats["sparse_gate"]} '
                       f'habitat_extra={self.stats.get("habitat_extra", 0)} '
                       f'-> full {len(survivors)}+{len(proposal_nodes)} '
                       f'elapsed={time.time() - t0:.0f}s')

        full_rows = []
        full_total = len(proposal_nodes) + len(survivors)
        full_done = 0
        for _fam, node, _hyp in proposal_nodes:
            rec = {'name': node_name('proposal', node), 'family': 'proposal',
                   'formula': render(node), 'expression_hash': expression_hash(node),
                   'node': node, 'source': 'proposal'}
            row = self.evaluate_full(rec)
            full_done += 1
            if row:
                full_rows.append(row)
                self._progress(f'[full {full_done}/{full_total}] {row["name"]} '
                               f'sia={row["sia"]:.4f} turn={row["turnover"] * 100:.1f} '
                               f'net={row["net"]:+.2f} dual={row.get("pass_dual")} '
                               f'elapsed={time.time() - t0:.0f}s')
        for rec in survivors:
            row = self.evaluate_full(rec)
            full_done += 1
            if row:
                full_rows.append(row)
                self._progress(f'[full {full_done}/{full_total}] {row["name"]} '
                               f'sia={row["sia"]:.4f} turn={row["turnover"] * 100:.1f} '
                               f'net={row["net"]:+.2f} dual={row.get("pass_dual")} '
                               f'elapsed={time.time() - t0:.0f}s')

        for r in full_rows:
            # 本地能力与竞赛上传能力分开记录。C2C field/operator 限制只由 export/upload 消费。
            r['local_only'] = formula_uses_local(r['formula'])
            r['competition_uploadable'], r['competition_reject_reason'] = \
                competition_status(r['formula'])
        valid = [r for r in full_rows if r['valid']]
        n_local_only = sum(1 for r in valid if r['local_only'])
        if self.dual_gate:
            dual_pass = [r for r in valid if r.get('pass_dual', False)]
            dual_rejected = [r for r in valid if not r.get('pass_dual', False)]
        else:
            dual_pass, dual_rejected = valid, []
        # T2/T3（2026-09-25）：衰减诊断 + 可选硬门槛（在截断前应用，避免被淘汰者占名额）
        if dual_pass:
            from fitness import decay_metrics as _dm, field_dominance as _fd
            for r in dual_pass[:max(args.top * 3, 30)]:
                try:
                    _vals = self.ctx.values(r['node'])
                    _dec = {self.primary: _dm(self.ctx, _vals)}
                    r['field_dom'] = _fd(self.ctx, _vals)
                    if self.secondary:
                        _sec = self.ctxs[self.secondary]
                        _dec[self.secondary] = _dm(_sec, _sec.values(r['node']))
                    r['decay'] = _dec
                except Exception as exc:
                    r['decay'] = {'error': str(exc)[:120]}
            if getattr(args, 'max_decay', None) is not None:
                _md = float(args.max_decay)
                _before = len(dual_pass)
                dual_pass = [r for r in dual_pass
                             if not r.get('decay') or (r['decay'].get(self.primary) or {}).get('decay_ratio') is None
                             or r['decay'][self.primary]['decay_ratio'] >= _md]
                if len(dual_pass) < _before:
                    self._progress(f'max_decay 门槛（decay_ratio≥{_md}）：淘汰 '
                                   f'{_before - len(dual_pass)} 个后半段衰减候选')
            if getattr(args, 'min_rank_ic', 0):
                _mri = float(args.min_rank_ic)
                _before = len(dual_pass)
                dual_pass = [r for r in dual_pass
                             if abs((r['labels'].get(self.primary) or {}).get('rank_ic') or 0) >= _mri]
                if len(dual_pass) < _before:
                    self._progress(f'min_rank_ic 门槛（|RankIC|≥{_mri}）：'
                                   f'淘汰 {_before - len(dual_pass)} 个')
        dual_pass.sort(key=lambda r: (-r['score'],
                                      -(r['stability'] if r.get('stability') == r.get('stability')
                                        else -9.0)))
        # 族内多样性约束（2026-09-25）：每 family 最多 N 席，防单族垄断短名单
        _fam_cap = max(1, int(getattr(args, 'family_cap', 3)))
        final = []
        _fam_cnt: dict = {}
        for r in dual_pass:
            f = r['family']
            if _fam_cnt.get(f, 0) >= _fam_cap:
                continue
            final.append(r)
            _fam_cnt[f] = _fam_cnt.get(f, 0) + 1
            if len(final) >= args.top:
                break
        if len(final) < args.top:
            _selected = set(id(r) for r in final)
            for r in dual_pass:
                if id(r) not in _selected:
                    final.append(r)
                    _selected.add(id(r))
                    if len(final) >= args.top:
                        break
        # 层1（2026-09-24）：终选成员的行业/beta 中性化诊断（仅报告，不进 rank_score）
        if args.resid and final:
            from resid import resid_metrics as _rm
            for r in final:
                try:
                    rm2 = _rm(self.ctx, r['node'], name=r['name'], context=True)
                    r['resid_ind'] = {'drop_ind': rm2.get('drop_ind'),
                                      'resid_ctx_sia': rm2.get('resid_ctx_sia'),
                                      'resid_ctx_rank_ic': rm2.get('resid_ctx_rank_ic')}
                except Exception as exc:
                    r['resid_ind'] = {'error': str(exc)[:120]}
        # 采样稳健性（2026-09-24）：cycle=3 锚点 offset 0/1/2 三组错位截面 RankIC
        # 一致性（方向按主口径冻结）——检验"因子是否在挑调仓时点"
        if final and not getattr(args, 'no_offset_diag', False):
            try:
                from offset_diag import build_offset_ctxs, offset_scan as _oscan
                octxs = build_offset_ctxs(self.evaluator.frame, self.evaluator,
                                          cycle=args.cycle, labels=self.labels)
                for r in final:
                    main_ric = {lab: (r['labels'].get(lab) or {}).get('rank_ic')
                                for lab in self.labels}
                    dirs = r.get('label_directions') or {lab: int(r['direction'])
                                                         for lab in self.labels}
                    r['offset_diag'] = _oscan(r['node'], octxs, main_ric, dirs,
                                              labels=self.labels, name=r['name'])
                n_sign = sum(1 for r in final if all(
                    (r.get('offset_diag') or {}).get(lab, {}).get('sign_consistent')
                    for lab in self.labels))
                self._progress(f'offset 0/1/2 采样稳健性：{n_sign}/{len(final)} 个候选'
                               f'三组 RankIC 同号（详见下方稳健性表）')
            except Exception as exc:
                self._progress(f'warning: offset 诊断失败 {exc}')
        # 条件 RankIC（2026-09-24）：热点子集（主管线 COND-IC 同口径）+ 市场状态族；
        # 栖息地内区分力诊断，不进 rank_score
        if final and not getattr(args, 'no_cond_ic', False):
            try:
                from cond_ic import build_cond_prep, cond_ic_metrics
                self._progress('条件 RankIC：构建条件掩码（hot5/hot2/mkt）...')
                cond_prep = build_cond_prep(self.evaluator.frame)
                for r in final:
                    per = {}
                    for lab in self.labels:
                        per[lab] = cond_ic_metrics(self.ctxs[lab], r['node'],
                                                   cond_prep, name=r['name'])
                    r['cond_ic'] = per
                self._progress(f'条件 RankIC：{len(final)} 个候选 × {len(self.labels)} '
                               f'标签 × 10 掩码族完成（详见下方条件表）')
            except Exception as exc:
                self._progress(f'warning: 条件 RankIC 诊断失败 {exc}')
        # T8（2026-09-25）：假设回填——栖息地增强/稀释结论按族聚合自动入 KB
        if final and any(r.get('cond_ic') for r in final):
            try:
                from knowledge import append as _kb_append, load as _kb_load
                _fam_ratio: dict = {}
                for r in final:
                    _d = (r.get('cond_ic') or {}).get(self.primary) or {}
                    _hot = _d.get('cond_hot5_q80_rank_ic_mean')
                    _full = ((r['labels'].get(self.primary) or {}).get('rank_ic'))
                    if _hot is None or _full is None or not np.isfinite(_hot) \
                            or abs(float(_full)) < 1e-9:
                        continue
                    _fam_ratio.setdefault(r['family'], []).append(
                        float(_hot) / float(_full))
                _existing = {e.get('id') for e in _kb_load()}
                _new = []
                for _fam, _ratios in sorted(_fam_ratio.items()):
                    if len(_ratios) < 2:
                        continue
                    _mean_r = float(np.mean(_ratios))
                    _kind = ('栖息地增强' if _mean_r > 1.3
                             else ('栖息地稀释' if _mean_r < 0.7 else None))
                    if _kind is None:
                        continue
                    _eid = f'auto-habitat-{self.profile}-{_fam}'
                    if _eid in _existing:
                        continue
                    _kb_append({
                        'id': _eid, 'status': 'rule', 'family': _fam,
                        'profile': self.profile, 'label': self.primary,
                        'data_cut': self.data_cut,
                        'mechanism': f'自动回填：{_fam} 在热点子集(hot5_q80)的 RankIC 为'
                                     f'全市场的 {_mean_r:.2f}×（{_kind}），样本 n={len(_ratios)}',
                        'evidence': 'search cond_ic 诊断跨候选聚合；全市场 IC 高≠栖息地内有用',
                        'tags': ['auto', 'habitat', 'cond-ic'],
                        'source': f'knowledge auto-habitat {datetime.now().strftime("%Y-%m-%d")}',
                    })
                    _existing.add(_eid)
                    _new.append(_eid)
                if _new:
                    self._progress(f'KB 假设回填：+{len(_new)} 条栖息地结论'
                                   f'（{", ".join(_new)}）')
            except Exception as exc:
                self._progress(f'warning: KB 栖息地回填失败 {exc}')
        def _rank_sia(r: dict, label: str) -> float:
            t = r['labels'][label]
            rk, ir, w = t.get('rank_ic'), t.get('rank_ic_ir'), t.get('rank_ic_win')
            if all(v is not None and np.isfinite(v) for v in (rk, ir, w)):
                return abs(float(rk)) * abs(float(ir)) * abs(float(w))
            return float(t.get('sia') or 0.0)

        vwap_rows = valid_rows_for_label(full_rows, 'vwap')
        vwap_rows.sort(key=lambda r: -_rank_sia(r, 'vwap'))
        vwap_n = int(getattr(args, 'vwap_top', 20))
        vwap_top = [{k: v for k, v in r.items() if k != 'node'}
                    for r in vwap_rows[:vwap_n]]
        c2c_rows = valid_rows_for_label(full_rows, 'c2c')
        c2c_rows.sort(key=lambda r: -float((r['labels'].get('c2c') or {}).get('sia') or 0.0))
        c2c_n = int(getattr(args, 'c2c_top', 20))
        c2c_top = [{k: v for k, v in r.items() if k != 'node'}
                    for r in c2c_rows[:c2c_n]]
        competition_top = [{k: v for k, v in r.items() if k != 'node'}
                           for r in c2c_rows if r.get('competition_uploadable')][:c2c_n]

        greedy = None
        if args.greedy_top > 0 and final:
            if self.member_panel is None:
                self.member_panel = pool_mod.build_extra_panel(self.entries)
            cands = final[:args.greedy_top]
            cand_df = pd.DataFrame({'stock_code': self.ctx.stock_codes,
                                    'date': self.ctx.row_dates})
            for r in cands:
                cand_df[r['name']] = self.ctx.values(r['node'])
            extra = self.member_panel.merge(cand_df, on=['stock_code', 'date'], how='outer')
            cand_meta = [{'name': r['name'], 'dir': r['direction'], 'sia': r['sia']}
                         for r in cands]
            reserve = [{'name': e['name'], 'dir': e['dir'], 'sia': e['sia']}
                       for e in self.state.get('reserve', [])]
            sia_override = None
            if getattr(args, 'refresh_sia', False):
                # 同窗化（2026-09-24）：成员 SiA 在 aligned_v1 窗重估后再与 NC 相加
                sia_override = pool_mod.refresh_member_sia(
            self.state, cycle=args.cycle, data_cut=self.data_cut)
                stale = sia_override.pop('stale', [])
                self._progress(f'refreshed member sia (aligned_v1/c2c); '
                               f'stale(no panel)={stale or "[]"}')
            self._progress(f'greedy ΔComb over {len(cand_meta) + len(reserve)} candidates ...')
            greedy = pool_mod.greedy_forward(extra, self.state, cand_meta + reserve,
                                             max_add=args.greedy_add,
                                             min_gain=args.greedy_min_gain,
                                             sia_override=sia_override,
                                             data_cut=self.data_cut)
            self._progress(f"  base={greedy['base_comb']:.4f} -> "
                           f"final={greedy['final_comb']:.4f} "
                           f"selected={greedy['selected']}")
            if greedy.get('promoted_reserve'):
                self._progress(f"  注意：reserve 候选 {greedy['promoted_reserve']} 已入选，"
                               f"应从 pool_state.json 的 reserve 挪入 members（本程序不自动改写）")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f'shortlist_{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}.json'
        clean = [{k: v for k, v in r.items() if k != 'node'} for r in final]
        n_tested_by_label = {
            lab: self.ledger.count_tested(self.data_cut, label=lab)
            for lab in self.labels
        }
        dual_rej = [{'name': r['name'], 'sia': r.get('search_value', r['sia']),
                     'dual_sia': r.get('dual_sia'),
                     'dual_ratio': r.get('dual_ratio'), 'score': r['score']}
                    for r in sorted(dual_rejected, key=lambda r: -r['score'])]
        payload = {'generated_at': datetime.now().isoformat(timespec='seconds'),
                   'seed': args.seed, 'cycle': args.cycle, 'cut': args.cut,
                   'strategy': args.strategy, 'profile': self.profile,
                   'primary_label': self.primary,
                   'primary_search_metric': label_search_metric(self.primary),
                   'labels': self.labels, 'reward_label': self.reward_label,
                   'label_semantics': ({'vwap': 'vwap_formula_only'}
                                       if 'vwap' in self.labels else {}),
                   'local_context': self.local_context,
                   'turnover_ref': self.turnover_ref,
                   'dual_gate': ({'secondary': self.secondary,
                                  'min_sia': self.dual_min_sia,
                                  'min_ratio': self.dual_min_ratio}
                                 if self.dual_gate else None),
                   'n_generated': len(fast_rows), 'n_fast': self.stats['fast'],
                   'n_full': self.stats['full'], 'n_pool_ok': len(dual_pass),
                   'n_dual_rejected': len(dual_rejected),
                   'n_dual_rejected_fast': n_dual_fast_rej,
                   'n_local_only': n_local_only,
                   'oos_sia_threshold': args.oos_sia,
                   'fast_keep_sia': self.fast_keep_sia,
                   'data_cut': self.data_cut,
                   'data_version': self.data_version,
                   'n_tested': n_tested_by_label.get(self.primary, 0),
                   'n_tested_cut': self.ledger.count_tested(self.data_cut),
                   'n_tested_by_label': n_tested_by_label,
                   'ctx_rows': int(self.ctx.n_rows), 'ctx_sections': len(self.ctx.ud),
                   'max_decay_gate': getattr(args, 'max_decay', None),
                   'min_rank_ic_gate': getattr(args, 'min_rank_ic', None),
                   'resid_weight': args.resid_weight, 'stats': self.stats,
                    'greedy': greedy, 'dual_rejected': dual_rej,
                    'shortlist': clean, 'vwap_top': vwap_top,
                    'c2c_top': c2c_top, 'competition_top': competition_top}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                            encoding='utf-8')
        jsonl_path = out_path.with_suffix('.jsonl')
        with jsonl_path.open('w', encoding='utf-8') as f:
            for r in vwap_rows:
                row = {k: v for k, v in r.items() if k != 'node'}
                row['labels'] = {lab: t for lab, t in r['labels'].items()}
                f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        self._progress(f'shortlist -> {out_path}')
        self._progress(f'jsonl     -> {jsonl_path}（本地多标签榜单；C2C 上传能力另由 adapter 过滤）')
        if mut_stats:
            try:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                mut_path = (OUT_DIR / 'mutation_stats.json' if self.primary == 'c2c'
                            else OUT_DIR / f'mutation_stats_{self.primary}.json')
                mut_path.write_text(
                    json.dumps(mut_stats, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass
        if dual_rejected:
            self._progress(f'注：{len(dual_rejected)} 个候选未过显式双标签门 '
                           f'({self.primary} × {self.secondary})；该门默认关闭')
        if n_local_only:
            self._progress(f'注：{n_local_only} 个本地 shortlist 候选含 LOCAL_ONLY 字段/算子；'
                           '保留在本地搜索结果，C2C adapter 会单独过滤不可上传项')
        for r in final:
            oos = r['holdout']['sia'] if r['holdout'] else float('nan')
            drop = (r.get('resid') or {}).get('drop')
            ind_drop = (r.get('resid_ind') or {}).get('drop_ind')
            local_vwap_t = r['labels'].get('vwap')
            local_vwap_sia = (local_vwap_t or {}).get('sia')
            print('%-22s %-8s dir=%d score=%.4f nopen=%.4f %s_sia=%.4f oos=%s '
                  'resid_drop=%s ind_drop=%s decay=%s dom=%s turn=%5.1f%% net=%+.2f vwap=%s c2c=%s' % (
                      r['name'], r['source'], r['direction'], r['score'],
                      r['score_nopen'] or 0.0, self.primary, r['sia'],
                      ('%.4f' % oos) if oos == oos else '-',
                      ('%.2f' % drop) if drop is not None and drop == drop else '-',
                      ('%.2f' % ind_drop) if ind_drop is not None and ind_drop == ind_drop else '-',
                      ('%.2f' % (r.get('decay') or {}).get(self.primary, {}).get('decay_ratio')
                       or float('nan')) if (r.get('decay') or {}).get(self.primary, {}).get('decay_ratio') is not None else '-',
                      ('%.2f@%s' % ((r.get('field_dom') or {}).get('max') or 0,
                                    (r.get('field_dom') or {}).get('field', '?'))
                       ) if (r.get('field_dom') or {}).get('max') is not None else '-',
                      (r['turnover'] or -1) * 100, (r['net'] or 0.0) * 100,
                      ('%.4f' % local_vwap_sia) if local_vwap_sia is not None else '-',
                      ('%.4f' % (r['labels'].get('c2c') or {}).get('sia'))
                      if (r['labels'].get('c2c') or {}).get('sia') is not None else '-'), flush=True)
        if vwap_rows:
            print('\n-- local VWAP top%d（formula-only label; 本地 profile） --' % vwap_n, flush=True)
            for r in vwap_rows[:vwap_n]:
                vt = r['labels']['vwap']
                c2c_sia = float((r['labels'].get('c2c') or {}).get('sia') or 0.0)
                ratio = (float(vt['sia'] or 0.0) / c2c_sia) if c2c_sia else float('nan')
                print('%-22s %-13s vwap=%.4f ratio=%.2f RankIC=%s IC=%s ICIR=%s mono=%s turn=%5.1f%%' % (
                    r['name'], r['family'], vt['sia'] or 0.0, ratio,
                    ('%.4f' % vt['rank_ic']) if vt.get('rank_ic') is not None else '-',
                    ('%.4f' % vt['ic']) if vt.get('ic') is not None else '-',
                    ('%.3f' % vt['ic_ir']) if vt.get('ic_ir') is not None else '-',
                       ('%.3f' % vt['mono']) if vt.get('mono') is not None else '-',
                    (r['turnover'] or -1) * 100), flush=True)
        if final and any(r.get('offset_diag') for r in final):
            print('\n-- 采样稳健性：RankIC(offset 0/1/2，方向按各标签 offset-0 冻结；'
                  '⚠=冻结后未全正，含 Pearson/Spearman 方向冲突) --',
                  flush=True)
            for r in final:
                od = r.get('offset_diag') or {}
                parts = []
                for lab in self.labels:
                    d = od.get(lab) or {}
                    rk = d.get('rank_ics') or {}

                    def _g(o):
                        v = rk.get(o, rk.get(str(o)))
                        return ('%+.4f' % v) if v is not None and np.isfinite(v) else '-'

                    parts.append('%s %s/%s/%s%s' % (
                        lab, _g(0), _g(1), _g(2),
                        '' if d.get('sign_consistent') else '⚠'))
                print('%-22s %s' % (r['name'], ' | '.join(parts)), flush=True)
        if final and any(r.get('cond_ic') for r in final):
            from cond_ic import summarize as _ci_sum
            print('\n-- 条件 RankIC（hot5_q80 / hot2_q80 / mkt_up；栖息地内区分力，非比赛口径） --',
                  flush=True)
            for r in final:
                ci = r.get('cond_ic') or {}
                parts = []
                for lab in self.labels:
                    d = ci.get(lab) or {}
                    parts.append(f'{lab}[{_ci_sum(d)}]' if d else f'{lab}[-]')
                print('%-22s %s' % (r['name'], ' | '.join(parts)), flush=True)
        if args.report:
            from report import build_report
            md = build_report(out_path, ledger_path=args.ledger)
            print(f'report  -> {md}', flush=True)
        return payload


def _ledger_with_cut(args) -> Ledger:
    """按 `--data-cut` 打开台账：写入事件自动带上该截止日（默认 = 选择窗上界）。

    同时记录**声明式数据版本**（`windows.declared_data_version`）——**只作溯源，不作过滤**。
    见 `SearchRun.__init__` 的说明：指纹只用 `cut_date`，否则日频更新会让指纹每日失效。
    """
    lg = Ledger(args.ledger)
    lg.data_cut = W.normalize_data_cut(getattr(args, 'data_cut', None))
    bad = W.check_data_cut(lg.data_cut)
    if bad and not getattr(args, 'allow_2026_cut', False):
        raise SystemExit('data_cut 违规：' + '；'.join(bad)
                         + '（如确需历史重放，显式加 --allow-2026-cut）')
    try:
        lg.data_version = W.declared_data_version("panel")
    except Exception:
        lg.data_version = None
    return lg


def cmd_search(args) -> None:
    t0 = time.time()
    run = SearchRun(args)
    print(f'profile={run.profile} primary={run.primary} labels={run.labels} | '
          f'ctx rows={run.ctx.n_rows:,} sections={len(run.ctx.ud)} | '
          f'fast rows={run.fast_frame is not None and len(run.fast_frame) or 0:,} | '
          f'ledger tested[{run.primary}]={run.ledger.count_tested(run.data_cut, label=run.primary)} | '
          f'{time.time() - t0:.0f}s', flush=True)
    run.run()
    if run.primary == 'c2c':
        try:
            from knowledge import auto_reflow
            added = auto_reflow(ledger=run.ledger, label=run.primary,
                                data_cut=run.data_cut, profile=run.profile)
            if added:
                print(f'knowledge auto-reflow: +{len(added)} failed entries '
                      f'({", ".join(a["id"] for a in added)})')
        except Exception as exc:
            print(f'warning: knowledge auto-reflow failed {exc}')
    else:
        print('knowledge auto-reflow: skipped for local VWAP (profile-specific failure cutoff '
              'is not yet calibrated)')


def _payload_primary(payload: dict, which: str | None = None) -> str:
    if payload.get('primary_label'):
        return str(payload['primary_label'])
    labels = payload.get('labels') or []
    if labels:
        return str(labels[0])
    if which == 'c2c' and payload.get('c2c_top'):
        return 'c2c'
    if which == 'vwap' and payload.get('vwap_top'):
        return 'vwap'
    return 'vwap'


def cmd_export(args) -> None:
    payload = json.loads(Path(args.shortlist).read_text(encoding='utf-8'))
    primary = _payload_primary(payload, args.list)
    rows = _upload_rows(payload, args.list)
    if not rows:
        raise SystemExit(f'no {args.list} candidates in this run; run the matching label/profile first')
    target_label = args.list if args.list in ('c2c', 'vwap') else primary
    if args.min_holdout_sia > 0 and target_label != primary:
        raise SystemExit('--min-holdout-sia applies only to the run primary label; '
                         'rerun with that profile primary or omit this filter')
    if args.min_holdout_sia > 0:
        rows = [r for r in rows
                if r.get('holdout_search_value',
                        (r.get('holdout') or {}).get('sia', 0)) >= args.min_holdout_sia]
    if args.max_abs_pool_corr is not None:
        rows = [r for r in rows
                if not (r.get('rho_max') == r.get('rho_max'))
                or r.get('rho_max', 0) <= args.max_abs_pool_corr]
    rows, rejected = prepare_competition_rows(rows, target_label, primary)
    if rejected:
        print(f'export 适配器：{len(rejected)} 个候选不符合 C2C 字段/算子/方向合同；'
              '仅从本次竞赛批次排除，本地搜索结果保留')
        for r in rejected[:8]:
            print(f"  {r['name']}: {r['reason']}")
    rows = rows[:args.top]
    if not rows:
        raise SystemExit('no candidates left after filters')
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / f'candidates_{args.tag}.txt')
    result = write_candidates(rows, out_path, tag=args.tag,
                              notes=(f"profile={payload.get('profile', 'legacy')} "
                                     f"selection_label={target_label}",
                                     *(args.note or ())))
    ok, msg = validate_file(out_path)
    if not ok:
        raise SystemExit(f'exported file failed batch.py validation: {msg}')
    est = estimate_credits(len(rows), args.per_run)
    print(f"wrote {result['txt']} ({result['n']} candidates; selection label={target_label})\n"
          f"meta {result['meta']}\n{msg}")
    print(f"credit estimate: probe≈{est['probe_credits']} + full≈{est['full_estimate']} "
          f"= {est['total_estimate']} (per_run={args.per_run})")
    print('next (after explicit approval + NO_PROXY): '
          f'python "$HOME/.claude/skills/pandaai-factor-online/scripts/batch.py" '
          f'"{result["txt"]}" --start {args.start} --end {args.end} --cycle {args.cycle} '
          f'--prefix "{args.tag}-" --max-runs {len(rows)} --hypotheses 0')


BATCH_SCRIPT_DEFAULT = (Path.home() / '.claude' / 'skills' / 'pandaai-factor-online'
                        / 'scripts' / 'batch.py')
CREDITS_PER_RUN = 2.2  # 2026-09-16 实测：12 run = 26 credit


def _upload_rows(payload: dict, which: str) -> list[dict]:
    primary = _payload_primary(payload, which)
    if which == 'vwap':
        rows = payload.get('vwap_top') or (
            payload.get('shortlist') if primary == 'vwap' else [])
    elif which == 'shortlist':
        rows = payload.get('shortlist') or []
    elif which == 'c2c':
        rows = payload.get('competition_top') or payload.get('c2c_top') or (
            payload.get('shortlist') if primary == 'c2c' else [])
    else:
        raise ValueError(f'unknown candidate list {which!r}')
    return [r for r in rows if r.get('formula') and r.get('name')]


def cmd_upload(args) -> None:
    """上传候选：默认 --create-only（0 credit）；--run 才执行平台分析（显式批准）。"""
    import os
    import subprocess
    payload = json.loads(Path(args.input).read_text(encoding='utf-8'))
    primary = _payload_primary(payload, args.list)
    rows = _upload_rows(payload, args.list)
    if not rows:
        raise SystemExit(f'no {args.list} candidates in this run; run the matching label/profile first')
    target_label = args.list if args.list in ('c2c', 'vwap') else primary
    rows, rejected = prepare_competition_rows(rows, target_label, primary)
    if rejected:
        print(f'upload 适配器：{len(rejected)} 个候选不符合 C2C 字段/算子/方向合同；'
              '仅从本次竞赛批次排除，本地候选仍保留')
        for r in rejected[:8]:
            print(f"  {r['name']}: {r['reason']}")
    if args.min_sia > 0:
        rows = [r for r in rows if float(r.get('sia') or 0) >= args.min_sia]
    if args.min_net is not None:
        rows = [r for r in rows if (r.get('net') is not None
                                    and float(r['net']) >= args.min_net)]
    if args.max_turnover is not None:
        rows = [r for r in rows if (r.get('turnover') is not None
                                    and float(r['turnover']) <= args.max_turnover)]
    # 远程平台结果是 C2C；候选排序仍依据明确选择的来源标签，不让 profile score
    # 或 C2C 平台能力影响本地 VWAP 排名。
    rows.sort(key=lambda r: -(r.get('sia') or 0.0))
    rows = rows[:args.top]
    if not rows:
        raise SystemExit('no candidates left after filters')
    for r in rows:
        if int(r.get('direction', -1)) not in (0, 1):
            raise SystemExit(f"candidate {r.get('name')} missing direction")
    tag = args.tag or Path(args.input).stem
    prefix = args.prefix if args.prefix is not None else f'{tag}-'
    out_path = Path(args.out) if args.out else (OUT_DIR / f'candidates_{tag}.txt')
    info = write_candidates(rows, out_path, tag=tag,
                            notes=(f'upload --list {args.list}; '
                                   f'selection_label={target_label}; '
                                   f'{"create-only (free)" if not args.run else "RUN (paid)"}',
                                   f'window {args.start}-{args.end} cycle{args.cycle}'))
    ok, msg = validate_file(out_path)
    if not ok:
        raise SystemExit(f'exported file failed batch.py validation: {msg}')
    batch = Path(args.batch_script)
    cmd = [sys.executable, str(batch), str(out_path), '--start', args.start,
           '--end', args.end, '--cycle', str(args.cycle),
           '--group-number', str(args.group_number),
           '--round-trip', str(args.round_trip), '--prefix', prefix,
           '--max-runs', str(len(rows)), '--hypotheses', str(args.hypotheses)]
    if not args.run:
        cmd.append('--create-only')
    print(f"wrote {info['txt']} ({info['n']} candidates)  [{args.list}]")
    for r in rows:
        print('  %-24s dir=%d sia=%.4f ~ %s' % (
            r['name'], int(r['direction']), float(r.get('sia') or 0), r['formula']))
    if args.run:
        est = len(rows) * CREDITS_PER_RUN
        print(f'\n!! --run 显式批准：将执行平台分析，{len(rows)} 个 run，'
              f'预计消耗 ≈ {est:.0f} credit（约 {CREDITS_PER_RUN}/个）')
        print('   （余额：pandaai-cli --json balance；若无意花费请去掉 --run 重跑本命令）')
    else:
        print('\n模式：create-only（只创建因子对象，0 credit；未执行分析）')
    print('命令：' + ' '.join(cmd))
    if args.print_only:
        return
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    no_proxy = 'localhost,127.0.0.1,pandaaiquant.com,www.pandaaiquant.com'
    env['NO_PROXY'] = (env.get('NO_PROXY', '') + ',' + no_proxy).strip(',')
    proc = subprocess.run(cmd, env=env)
    raise SystemExit(proc.returncode)


def cmd_intake(args) -> None:
    from intake import intake_file, smoke_frame
    smoke = None if args.no_smoke else smoke_frame()
    report = intake_file(args.file, _ledger_with_cut(args), smoke=smoke,
                         limit=args.limit or None,
                         profile=getattr(args, 'profile', 'local-vwap'))
    print(f"intake {report['path']}: {report['counts']}")
    for item in report['results']:
        mark = {'ok': 'OK  ', 'warn': 'WARN', 'reject': 'FAIL',
                'duplicate': 'DUP '}[item['status']]
        print(f"  {mark} {item['canonical'][:100]}")
        for issue in item['issues']:
            print(f"        [{issue['level']}] {issue['code']}: {issue['message']}")
    if args.out:
        payload = {**report, 'results': [{k: v for k, v in r.items() if k != 'node'}
                                         for r in report['results']]}
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                             default=str), encoding='utf-8')
        print(f'saved {args.out}')


def cmd_propose(args) -> None:
    from llm_client import available, propose, write_inbox_template
    profile = getattr(args, 'profile', 'local-vwap')
    if not available():
        path = args.inbox or str(OUT_DIR / 'proposals_inbox.jsonl')
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_inbox_template(path, args.hypothesis, args.k, profile=profile)
        print('未检测到 OPENAI_API_KEY/OPENAI_BASE_URL：已生成 inbox 模板 ' + path)
        print(f'填写后运行：python pandaai_online/merge/intake.py "{path}"')
        return
    items = propose(args.hypothesis, args.k, profile=profile)
    if not items:
        raise SystemExit('LLM 未返回可解析的候选')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(args.out or OUT_DIR / 'proposals.jsonl')
    with path.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps({'hypothesis': args.hypothesis, **item},
                               ensure_ascii=False) + '\n')
    print(f'LLM 返回 {len(items)} 个候选 -> {path}')
    from intake import intake_file, smoke_frame
    report = intake_file(path, _ledger_with_cut(args), smoke=smoke_frame(), profile=profile)
    print(f"intake: {report['counts']}")
    for item in report['results']:
        mark = {'ok': 'OK  ', 'warn': 'WARN', 'reject': 'FAIL',
                'duplicate': 'DUP '}[item['status']]
        print(f"  {mark} {item['canonical'][:100]}")


def _ledger_candidates(ledger: Ledger, data_cut: str | None = None) -> dict:
    """台账 → {expression_hash: {formula, family, direction, labels: {lab: 指标}}}。

    data_cut 非空时硬过滤：只聚合该截止日的事件（缺字段旧记录按 LEGACY 计，
    与 hashes_for/vfps_for 同口径）。不传则全量（decay 衰减监控等显式跨期诊断用）。
    """
    out: dict = {}
    _cut = W.normalize_data_cut(data_cut) if data_cut else None
    for rec in ledger.read():
        if rec.get('event') not in ('fast', 'local'):
            continue
        if _cut is not None and str(rec.get('data_cut') or W.LEGACY_DATA_CUT) != _cut:
            continue
        lab = rec.get('label')
        if lab not in LABELS:
            continue
        d = rec.get('data') or {}
        t = d.get('tier1') or {}
        if t.get('sia') is None:
            continue
        h = rec.get('expression_hash')
        if not h or not rec.get('formula'):
            continue
        bucket = out.setdefault(h, {'formula': rec['formula'],
                                    'family': rec.get('family') or 'x',
                                    'direction': int(rec.get('direction', 1)),
                                    'window': 'fast',
                                    'labels': {}})
        if rec.get('event') == 'local':
            bucket['window'] = 'full'
        if rec.get('event') == 'local' and rec.get('label') == 'c2c':
            bucket['direction'] = int(rec.get('direction', 1))
        raw = d.get('fast') or d.get('metrics') or {}
        bucket['labels'][lab] = {
            'sia': float(t['sia']), 'turnover': t.get('turnover'),
            'coverage': t.get('coverage'),
            'n_valid_periods': raw.get('n_valid_periods'),
            'direction': int(rec.get('direction', 1)),
            'mi': t.get('mi'), 'mi_std': t.get('mi_std'),
            'nmi': t.get('nmi'), 'jsd': t.get('jsd'),
            'habitat_score': d.get('habitat_score'),
            'turnover_month': t.get('turnover_month'), 'net': t.get('net'),
            'sharpe': t.get('sharpe'), 'mdd': t.get('mdd'),
            'rank_ic': t.get('rank_ic'), 'ic': t.get('ic'),
            'ic_ir': t.get('ic_ir'), 'win': t.get('win'),
            'rank_ic_ir': t.get('rank_ic_ir'), 'rank_ic_win': t.get('rank_ic_win'),
            'cost': t.get('cost'), 'long': t.get('long'), 'mono': t.get('mono'),
            'stability': d.get('stability'),
            'yearly_ic': raw.get('yearly_ic') or {},
            'ic_mean': raw.get('ic_mean'),
        }
    return out


def _rescore_rows(rows: dict, turnover_ref: float | None) -> list[dict]:
    """按新目标重排：score = A × 换手软惩罚 × 复杂度因子；稳健性作第二键。"""
    out = []
    for h, b in rows.items():
        rec = b['labels'].get('c2c')
        if not rec:
            continue
        tm = rec.get('turnover_month')
        t = rec.get('turnover')
        if tm is None or not np.isfinite(tm):
            tm = 7.0 * float(t) if t is not None and np.isfinite(t) else None
        nodes, parsed = 1, True
        try:
            node = parse_formula(b['formula'])
            nodes = node_count(node)
        except Exception:
            parsed = False
            node = None
        pen = 1.0
        if turnover_ref is not None and tm is not None:
            pen = float(turnover_ref) / max(float(tm), float(turnover_ref))
        comp = 1.0 / (1.0 + 0.02 * max(0, nodes - 8))
        stability = rec.get('stability')
        if stability is None or not np.isfinite(stability):
            stability = stability_of({'yearly_ic': rec.get('yearly_ic') or {},
                                      'ic_mean': rec.get('ic_mean')})
        v = b['labels'].get('vwap') or {}
        vwap_sia = v.get('sia')
        ratio = (vwap_sia / rec['sia']) if (vwap_sia is not None and rec['sia'] > 0) else None

        def _rank_sia(x: dict | None):
            if not x:
                return None
            r, ir, w = x.get('rank_ic'), x.get('rank_ic_ir'), x.get('rank_ic_win')
            if r is None or ir is None or w is None:
                return None
            if not (np.isfinite(r) and np.isfinite(ir) and np.isfinite(w)):
                return None
            return abs(float(r)) * abs(float(ir)) * abs(float(w))

        out.append({
            'hash': h, 'name': (node_name(b['family'], node) if node is not None
                                else f"MG-{b['family'].split('_')[0]}-{h[:8]}"),
            'family': b['family'], 'direction': b['direction'],
            'label_directions': {lab: int(m['direction']) for lab, m in b['labels'].items()
                                 if m.get('direction') in (0, 1)},
            'formula': b['formula'], 'parsed': parsed,
            'window': b.get('window', 'fast'),
            'sia': rec['sia'], 'turnover': t, 'turnover_month': tm,
            'net': rec.get('net'), 'sharpe': rec.get('sharpe'), 'mdd': rec.get('mdd'),
            'rank_ic': rec.get('rank_ic'), 'ic': rec.get('ic'),
            'ic_ir': rec.get('ic_ir'), 'win': rec.get('win'),
            'cost': rec.get('cost'), 'long': rec.get('long'), 'mono': rec.get('mono'),
            'c_proxy': c_proxy_of({'net': rec.get('net'), 'turnover_month': tm,
                                   'sharpe': rec.get('sharpe'), 'mdd': rec.get('mdd')}),
            'stability': stability, 'penalty': pen,
            'score': rec['sia'] * pen * comp,
            'score_nopen': rec['sia'] * comp,
            'vwap_sia': vwap_sia, 'vwap_ratio': ratio,
            'rank_sia': _rank_sia(rec), 'vwap_rank_sia': _rank_sia(v),
            'vwap_rank_ic': v.get('rank_ic'), 'vwap_ic': v.get('ic'),
            'vwap_ic_ir': v.get('ic_ir'), 'vwap_win': v.get('win'),
            'vwap_rank_ic_ir': v.get('rank_ic_ir'), 'vwap_rank_win': v.get('rank_ic_win'),
            'vwap_turnover': v.get('turnover'), 'vwap_net': v.get('net'),
            'vwap_mono': v.get('mono'), 'vwap_sharpe': v.get('sharpe'),
            'vwap_mdd': v.get('mdd'), 'vwap_long': v.get('long'),
            'vwap_cost': v.get('cost'),
        })
    return out


def cmd_rescore(args) -> None:
    if not hasattr(args, 'vwap_sort'):
        args.vwap_sort = 'sia'
    if not hasattr(args, 'vwap_top'):
        args.vwap_top = 20
    ledger = _ledger_with_cut(args)
    rows = _ledger_candidates(ledger, data_cut=ledger.data_cut)
    print(f'ledger candidates: {len(rows)}（含 fast/local 证据，cut={ledger.data_cut}）',
          flush=True)
    ref = None if args.no_turnover_penalty else args.turnover_ref
    if args.full_top > 0:
        labels = list(args.label or ['c2c', 'vwap'])
        frame = load_panel()
        ev = Evaluator(frame)
        ctxs = {lab: DataContext(cycle=args.cycle, frame=frame, label=lab,
                                 end=ledger.data_cut,
                                 evaluator=ev) for lab in labels}
        pre = _rescore_rows(rows, ref)
        pre.sort(key=lambda r: -r['score'])

        def _vkey_pre(r: dict) -> float:
            v = {'sia': r.get('vwap_sia'), 'ic': r.get('vwap_ic'),
                 'icir': r.get('vwap_ic_ir'), 'rank_ic': r.get('vwap_rank_ic'),
                 'rank_sia': r.get('vwap_rank_sia'),
                 'mono': r.get('vwap_mono')}.get(args.vwap_sort)
            if v is None or not np.isfinite(v):
                v = r.get('vwap_sia')
            return abs(float(v)) if v is not None and np.isfinite(v) else -1.0

        vpre = sorted([r for r in pre if r['vwap_sia'] is not None],
                      key=lambda r: -_vkey_pre(r))
        pick = list(dict.fromkeys([r['hash'] for r in pre[:args.full_top]]
                                  + [r['hash'] for r in vpre[:args.vwap_top]]))
        print(f'full-window re-eval: {len(pick)} candidates', flush=True)
        for i, h in enumerate(pick, 1):
            b = rows[h]
            try:
                node = parse_formula(b['formula'])
            except Exception:
                continue
            name = node_name(b['family'], node)
            screens = {lab: ctxs[lab].screen(node, name=name) for lab in labels}
            tiers = {lab: tier1(m) for lab, m in screens.items()}
            _nl = nonlinear_for_ctxs(ctxs, labels, node)
            for _lab, _t in tiers.items():
                _d = _nl.get(_lab) or {}
                _t['mi'] = _d.get('mi')
                _t['mi_std'] = _d.get('mi_std')
                _t['nmi'] = _d.get('nmi')
                _t['jsd'] = _d.get('jsd')
            for lab in labels:
                t = tiers[lab]
                b['labels'][lab] = {**b['labels'].get(lab, {}),
                                    **{k: t.get(k) for k in
                                       ('sia', 'turnover', 'turnover_month', 'net',
                                        'sharpe', 'mdd', 'rank_ic', 'ic', 'ic_ir',
                                        'win', 'cost', 'long', 'mono',
                                        'rank_ic_ir', 'rank_ic_win')}}
                ledger.log('local', node, int(screens[lab].get('direction', 1)),
                           family=b['family'], label=lab,
                           data={'metrics': screens[lab], 'tier1': tiers[lab],
                                 'source': 'rescore'})
            b['window'] = 'full'
            b['direction'] = int(screens[labels[0]].get('direction', 1))
            print(f'  [{i}/{len(pick)}] {name} dir={b["direction"]} '
                  f'sia={tiers[labels[0]]["sia"]:.4f} '
                  f'turn={(tiers[labels[0]]["turnover"] or 0) * 100:.1f}%', flush=True)
    scored = _rescore_rows(rows, ref)
    scored.sort(key=lambda r: (-r['score'],
                               -(r['stability'] if r['stability'] == r['stability'] else -9.0)))
    c2c_top = scored[:args.top]

    def _vkey(r: dict) -> float:
        v = {'sia': r.get('vwap_sia'), 'ic': r.get('vwap_ic'),
             'icir': r.get('vwap_ic_ir'), 'rank_ic': r.get('vwap_rank_ic'),
             'rank_sia': r.get('vwap_rank_sia'),
             'mono': r.get('vwap_mono')}.get(args.vwap_sort)
        if v is None or not np.isfinite(v):
            v = r.get('vwap_sia')
        return abs(float(v)) if v is not None and np.isfinite(v) else -1.0

    vwap_rows = sorted([r for r in scored if r['vwap_sia'] is not None],
                       key=lambda r: (-_vkey(r), -r['score']))[:args.vwap_top]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUT_DIR / f'rescore_{ts}.json'
    payload = {'generated_at': datetime.now().isoformat(timespec='seconds'),
               'turnover_ref': ref, 'n_candidates': len(scored),
               'c2c_top': c2c_top, 'vwap_top': vwap_rows}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                        encoding='utf-8')
    jsonl_path = out_path.with_suffix('.jsonl')
    with jsonl_path.open('w', encoding='utf-8') as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    md_path = out_path.with_suffix('.report.md')

    def _f(v, fmt='.4f'):
        return format(float(v), fmt) if v is not None and np.isfinite(v) else '-'

    lines = [f'# rescore 重排报告 {payload["generated_at"]}',
             '',
             f'- 候选数 {len(scored)}；换手软惩罚 T_ref={ref}',
             f'- 分数 = SiA × T_ref/max(T_month,T_ref) × 复杂度因子；稳健性=分年 IC 对齐最小值',
             f'- 指标口径：c2c_3d（比赛口径；RankIC=Spearman、IC=Pearson、ICIR=IC 均值/样本std、'
             f'胜率=方向单尾 |IC|>0.02 占比，与官方 A 分公式一致）',
             '', '## c2c 入选（A 主分，比赛口径）', '',
             '| 候选 | 窗口 | 家族 | dir | RankIC | IC | ICIR | 胜率 | SiA | 换手 | T_month | '
             '净超额 | 夏普 | 回撤 | C_proxy | 稳健 | 分数 | vwap SiA(比值) |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | '
             '--- | --- | --- | --- | --- |']
    for r in c2c_top:
        lines.append('| %s | %s | %s | %d | %s | %s | %s | %s | %.4f | %.1f%% | %s | %s | %s | %s | %s | %s | %.4f | %s |' % (
            r['name'], r['window'], r['family'], r['direction'],
            _f(r.get('rank_ic')), _f(r.get('ic')), _f(r.get('ic_ir')), _f(r.get('win'), '.3f'),
            r['sia'], (r['turnover'] or 0) * 100,
            f"{r['turnover_month']:.2f}" if r['turnover_month'] is not None else '-',
            f"{r['net']:+.2%}" if r['net'] is not None and np.isfinite(r['net']) else '-',
            _f(r.get('sharpe'), '.2f'), _f(r.get('mdd'), '.1%'),
            _f(r['c_proxy'], '.3f'), _f(r['stability'], '.3f'), r['score'],
            f"{r['vwap_sia']:.4f}({r['vwap_ratio']:.2f})" if r['vwap_sia'] is not None else '-'))
    lines += ['', '## vwap top（非比赛口径，供 LightGBM 项目；排序=%s）' % args.vwap_sort, '',
              '| 候选 | 窗口 | vwap rank_sia | vwap SiA | ratio | vwap RankIC | vwap RankICIR | '
              'vwap IC | vwap mono | vwap 换手 | c2c SiA | 公式 |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in vwap_rows:
        lines.append('| %s | %s | %s | %.4f | %s | %s | %s | %s | %s | %s | %.4f | `%s` |' % (
            r['name'], r['window'], _f(r.get('vwap_rank_sia')),
            r['vwap_sia'],
            f"{r['vwap_ratio']:.2f}" if r['vwap_ratio'] is not None else '-',
            _f(r.get('vwap_rank_ic')), _f(r.get('vwap_rank_ic_ir')),
            _f(r.get('vwap_ic')), _f(r.get('vwap_mono'), '.3f'),
            f"{(r['vwap_turnover'] or 0) * 100:.1f}%" if r.get('vwap_turnover') is not None else '-',
            r['sia'], r['formula']))
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nrescore -> {out_path}')
    print(f'jsonl   -> {jsonl_path}')
    print(f'report  -> {md_path}')
    print('\n== c2c 入选 top%d ==' % len(c2c_top))
    for r in c2c_top:
        print('%-22s %-6s sia=%.4f RIC=%s ICIR=%s turn=%5.1f%% T=%s net=%s score=%.4f vwap=%s' % (
            r['name'], r['window'], r['sia'],
            _f(r.get('rank_ic')), _f(r.get('ic_ir')),
            (r['turnover'] or 0) * 100,
            f"{r['turnover_month']:.2f}" if r['turnover_month'] is not None else '-',
            f"{r['net']:+.1%}" if r['net'] is not None and np.isfinite(r['net']) else '-',
            r['score'],
            f"{r['vwap_sia']:.4f}" if r['vwap_sia'] is not None else '-'))
    print('\n== vwap top%d（排序=%s） ==' % (len(vwap_rows), args.vwap_sort))
    for r in vwap_rows:
        print('%-22s %-6s rank_sia=%s vwap=%.4f ratio=%s RankIC=%s RankICIR=%s mono=%s turn=%5.1f%%' % (
            r['name'], r['window'], _f(r.get('vwap_rank_sia')),
            r['vwap_sia'],
            f"{r['vwap_ratio']:.2f}" if r['vwap_ratio'] is not None else '-',
            _f(r.get('vwap_rank_ic')), _f(r.get('vwap_rank_ic_ir')),
            _f(r.get('vwap_mono'), '.3f'), (r['vwap_turnover'] or 0) * 100))


def _parse_candidates_txt(path: Path) -> dict[str, tuple[str, int]]:
    """candidates.txt（`name ~ formula ~ direction`）→ {name: (formula, direction)}。"""
    out: dict[str, tuple[str, int]] = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '~' not in line:
            continue
        parts = [p.strip() for p in line.split('~')]
        if len(parts) != 3:
            continue
        out[parts[0]] = (parts[1], int(parts[2]) if parts[2] in ('0', '1') else 1)
    return out


def cmd_platform_ingest(args) -> None:
    """batch.py 产物（report.csv + meta.json + candidates.txt）→ platform 事件回填台账。

    消除断链（2026-09-24）：ledger.py 早已声明 platform 事件但全仓 0 写入者，
    平台 IC vs 本地 IC 校准只能人工读 CSV。label 置 None（不污染公式×标签排除集）；
    platform_sia 代理 = |RankIC×ICIR|（report.csv 无 IC 胜率列，缺的因子由系数吸收）。
    """
    import csv as _csv
    report = Path(args.report)
    if not report.exists():
        raise SystemExit(f'report not found: {report}')
    meta_path = (Path(args.meta) if args.meta else
                 report.with_name(report.name.replace('.report.csv', '.meta.json')))
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    meta_by_name = {c.get('name'): c for c in meta.get('candidates', [])}
    txt_path = (Path(args.txt) if args.txt else
                (meta_path.parent / meta.get('file', '') if meta.get('file') else None))
    txt_map = _parse_candidates_txt(txt_path) if txt_path and txt_path.exists() else {}
    if not txt_map:
        print(f'warning: candidates txt 未找到/为空（{txt_path}），仅 meta 含 hash 的候选可回填')
    ledger = _ledger_with_cut(args)
    rows = list(_csv.DictReader(report.read_text(encoding='utf-8-sig').splitlines()))

    def _f(key):
        def get(r):
            v = r.get(key)
            try:
                return float(v) if v not in (None, '') else None
            except (TypeError, ValueError):
                return None
        return get

    n_ing = n_skip = 0
    for r in rows:
        if str(r.get('status', '')) != 'completed':
            n_skip += 1
            continue
        name = r.get('name', '')
        mc = meta_by_name.get(name) or {}
        formula = txt_map.get(name, (None, None))[0]
        expr_hash = mc.get('expression_hash')
        if not expr_hash and formula:
            try:
                expr_hash = expression_hash(parse_formula(formula))
            except Exception:
                expr_hash = None
        if not expr_hash or not formula:
            print(f'  skip {name}: 无 expression_hash/formula 映射')
            n_skip += 1
            continue
        turnover = _f('turnover_pct')(r)
        data = {'source': 'platform-run', 'factor_id': r.get('factor_id'),
                'run_id': r.get('run_id'), 'tag': meta.get('tag') or meta_path.stem,
                'rank_ic': _f('rank_ic')(r), 'ic_mean': _f('ic_mean')(r),
                'ic_ir': _f('ic_ir')(r), 'ic_p_value': _f('ic_p_value')(r),
                'monotonicity': _f('monotonicity')(r),
                'long_excess': _f('long_excess_pct')(r),
                'turnover': turnover / 100.0 if turnover is not None else None,
                'annual_cost': _f('annual_cost_pct')(r),
                'net_excess': _f('net_excess_pct')(r),
                'long_sharpe': _f('long_sharpe')(r),
                'long_mdd': _f('long_max_drawdown_pct')(r),
                'long_monthly_win': _f('long_monthly_win_rate_pct')(r),
                'sia_proxy': (lambda ric, iir: abs(ric * iir)
                              if ric is not None and iir is not None else None)(
                    _f('rank_ic')(r), _f('ic_ir')(r)),
                'local_sia': mc.get('sia'), 'local_turnover': mc.get('turnover'),
                'error': r.get('error') or ''}
        ledger.log_raw('platform', formula,
                       direction=int(mc.get('direction') or 1),
                       family=mc.get('family') or 'platform', data=data, label=None)
        n_ing += 1
    print(f'platform ingest: {n_ing} events, skipped {n_skip} -> {ledger.path}')


def cmd_calibrate(args) -> None:
    """platform↔local 配对拟合：platform_sia ≈ a × local_sia × T_local^b（log-log OLS）。

    b<0 表示本地换手越高、平台实现率越低（amihud 教训的定量版）。
    产出 calibration.json；rank_score 仅在 search --platform-prior 时消费（opt-in）。
    """
    ledger = _ledger_with_cut(args)
    plat: dict = {}
    local: dict = {}
    for rec in ledger.read():
        if rec.get('event') == 'platform':
            d = rec.get('data') or {}
            if d.get('sia_proxy') is not None:
                plat[rec['expression_hash']] = d  # append-only：后写覆盖，保留最新 run
        elif rec.get('event') in ('fast', 'local') and rec.get('label') in (None, 'c2c'):
            t = (rec.get('data') or {}).get('tier1') or {}
            if t.get('sia') is not None:
                local[rec['expression_hash']] = (
                    float(t['sia']), t.get('turnover_month') or t.get('turnover'))
    pairs = []
    for h, d in plat.items():
        if h not in local:
            continue
        ls, t = local[h]
        ps = float(d['sia_proxy'])
        if ls <= 0 or ps <= 0 or t is None or not np.isfinite(t) or t <= 0:
            continue
        pairs.append((ls, ps, float(t), d.get('tag') or ''))
    if len(pairs) < args.min_n:
        raise SystemExit(f'配对样本不足：{len(pairs)} < {args.min_n}（先跑 platform-ingest）')
    y = np.array([np.log(p[1]) - np.log(p[0]) for p in pairs])
    x = np.array([np.log(p[2]) for p in pairs])
    b, log_a = np.polyfit(x, y, 1)
    pred = b * x + log_a
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    out = {'fitted_at': datetime.now().isoformat(timespec='seconds'),
           'a': float(np.exp(log_a)), 'b': float(b), 'n': len(pairs), 'r2': float(r2),
           'model': 'platform_sia_proxy = a * local_sia * T_local^b（log-log OLS）',
           'note': 'sia_proxy=|RankIC×ICIR|（平台 report 无 IC 胜率列）；'
                   'rank_score 消费时仅用 T^b 相对项并 clip [0.5,2.0]',
           'pairs': [{'local_sia': p[0], 'platform_sia_proxy': p[1],
                      'local_turnover': p[2], 'tag': p[3]} for p in pairs]}
    out_path = Path(args.out) if args.out else OUT_DIR / 'calibration.json'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'calibration: a={out["a"]:.4g} b={out["b"]:.3f} n={out["n"]} R2={r2:.3f}')
    print(f'saved {out_path}')


def cmd_decay_check(args) -> None:
    """衰减监控（T7，2026-09-25）：台账 winners 在当前面板重评 + 衰减指标表。

    对台账 local c2c 证据、公式可解析的候选按旧 SiA 取前 N，全窗重评并计算
    decay_ratio（后半段/前半段 RankIC 比）、roll_min（滚动 60 期最低）、max_neg_run
    （最长连负段）。decay_ratio < 1 为衰减趋势；结合新旧 SiA 对照可发现失效因子。
    注意：池成员（factors.parquet 面板列、无公式）不在此范围——用 refresh_member_sia。
    """
    ledger = _ledger_with_cut(args)
    # 衰减监控显式跨期：不过滤 data_cut（目的即新旧对照），但全部现窗重评，
    # 不复用任何旧指标，故无串味。
    rows = _ledger_candidates(ledger)
    frame = load_panel()
    ev = Evaluator(frame)
    ctx = DataContext(frame=frame, label='c2c', end=ledger.data_cut, evaluator=ev)
    from fitness import decay_metrics as _dm
    scored = []
    for h, b in sorted(rows.items(), key=lambda kv: -(kv[1]['labels'].get('c2c') or {}).get('sia') or 0):
        if len(scored) >= args.top:
            break
        old = b['labels'].get('c2c') or {}
        if old.get('sia') is None:
            continue
        try:
            node = parse_formula(b['formula'])
        except Exception:
            continue
        try:
            m = ctx.screen(node, name=f'decay:{h[:8]}')
            t = tier1(m)
            dec = _dm(ctx, ctx.values(node))
        except Exception as exc:
            print(f'  skip {h[:10]}: {str(exc)[:80]}')
            continue
        if not t.get('valid'):
            continue
        scored.append({
            'hash': h, 'formula': b['formula'], 'family': b.get('family') or '',
            'old_sia': float(old['sia']), 'new_sia': float(t['sia'] or 0.0),
            'sia_ratio': (float(t['sia']) / float(old['sia']))
            if float(old['sia']) > 0 else np.nan,
            'rank_ic': t.get('rank_ic'), 'turnover': t.get('turnover'),
            'decay_ratio': dec.get('decay_ratio'), 'roll_min': dec.get('roll_min'),
            'max_neg_run': dec.get('max_neg_run'),
        })
    scored.sort(key=lambda r: (r['decay_ratio'] if r['decay_ratio'] is not None else 9))
    out_rows = []
    for r in scored:
        flag = ''
        if args.max_decay is not None and r['decay_ratio'] is not None \
                and r['decay_ratio'] < args.max_decay:
            flag = ' ⚠衰减'
        if args.min_sia_ratio is not None and np.isfinite(r['sia_ratio']) \
                and r['sia_ratio'] < args.min_sia_ratio:
            flag += ' ⚠退步'
        out_rows.append({**r, 'flag': flag.strip()})
        print('%-22s %-8s old=%.4f new=%.4f ratio=%.2f decay=%s roll_min=%s neg_run=%s%s' % (
            r['hash'][:10], r['family'][:8], r['old_sia'], r['new_sia'],
            r['sia_ratio'] if np.isfinite(r['sia_ratio']) else float('nan'),
            ('%.2f' % r['decay_ratio']) if r['decay_ratio'] is not None else '-',
            ('%.4f' % r['roll_min']) if r['roll_min'] is not None else '-',
            r['max_neg_run'] if r['max_neg_run'] is not None else '-',
            flag), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'decay_check_{datetime.now():%Y%m%d_%H%M%S}.csv'
    pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'decay-check: {len(out_rows)} candidates -> {out_path}')


def cmd_report(args) -> None:
    from report import build_report
    md = build_report(args.shortlist, args.out or None, ledger_path=args.ledger)
    print(f'report -> {md}')


def cmd_ledger(args) -> None:
    stats = _ledger_with_cut(args).stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_fixtures(_args) -> None:
    from fixtures import FIXTURES
    for name, builder in sorted(FIXTURES.items()):
        print(f'{name:14s} {render(builder())}')


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ledger', default=str(Ledger().path))
    ap.add_argument('--data-cut', default='auto',
                    help='数据截止日（候选身份 = 公式 × 数据截止日）。auto/默认 = '
                         '%s（选择窗上界）；可传 ISO 日期或 legacy（=%s，仅历史重放）'
                         % (W.DEFAULT_DATA_CUT, W.LEGACY_DATA_CUT))
    ap.add_argument('--allow-2026-cut', action='store_true', default=False,
                    help='显式允许触及衰减监控段的截止日（仅历史重放；默认拒绝）')
    ap.add_argument('--use-legacy-evidence', action='store_true', default=False,
                    help='腿库/精英回流不按数据截止日隔离（复用遗留窗口选出的 winner；'
                         '默认隔离，因为旧 winner 带旧选择偏差）')
    ap.add_argument('--retest-before', default=None,
                    help='**显式**重测杠杆（ISO 日期/时间，如 2026-10-12）：'
                         '只把最后事件早于该时刻的公式重新纳入候选。'
                         '用于数据发生**实质变化**时（批量复权重算、数据源更换、修数据）；'
                         '**日常日频更新不要用**——指纹的作用键只有 --data-cut，'
                         '不会随 cache 每日重生成而失效。')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('search')
    sp.add_argument('--strategy', choices=('mechanism', 'random'), default='mechanism')
    sp.add_argument('--profile', choices=('local-vwap', 'competition-c2c'), default=None,
                    help='主搜索目标。省略时默认 local-vwap；competition-c2c 为可选大赛扩展。')
    sp.add_argument('--allow-documented', action='store_true',
                    help='额外允许本地已实现的 DOCUMENTED 算子族；平台可用性另由 export/upload 校验')
    sp.add_argument('--label', action='append', default=None,
                    help='附加测量标签，可重复。无 --profile 时首个标签兼容旧行为作为主标签；'
                         '显式 --profile 时 profile 标签为主，其余只作诊断')
    sp.add_argument('--dual-gate', action='store_true', default=False,
                    help='显式启用 primary 与第一个附加标签的双标签必过门（默认关闭）')
    sp.add_argument('--no-dual-gate', dest='dual_gate', action='store_false')
    sp.add_argument('--dual-min-sia', type=float, default=0.0012,
                    help='（仅 --dual-gate 时）附加标签 search metric 最低值')
    sp.add_argument('--dual-min-ratio', type=float, default=0.25,
                    help='（仅 --dual-gate 时）附加标签/主标签 search metric 比值最低值')
    sp.add_argument('--vwap-top', type=int, default=20,
                    help='本地 VWAP 榜长度（按 vwap RankIC×RankICIR×胜率排序）')
    sp.add_argument('--c2c-top', type=int, default=20,
                    help='C2C 竞赛扩展榜长度（按 C2C SiA 排序；上传能力另行筛选）')
    sp.add_argument('--turnover-ref', type=float, default=None,
                    help='换手软惩罚参考 T_ref；profile 默认 local-vwap 不施加，competition-c2c=0.3。'
                         '显式数值可覆盖 profile 默认')
    sp.add_argument('--no-turnover-penalty', action='store_true',
                    help='关闭换手软惩罚（对照用，同时仍记录 score_nopen）')
    sp.add_argument('--ucb-c', type=float, default=0.3,
                    help='机制轴 UCB 探索系数（0=退化为纯均值贪心）')
    sp.add_argument('--reward-label', choices=('auto', 'c2c', 'vwap', 'blend'), default='auto',
                    help='机制轴 UCB reward 标签（auto=profile 主标签；blend 为显式混合诊断）')
    sp.add_argument('--axis-recent-days', type=int, default=90,
                    help='轴 reward 滑动窗口天数（0=全期）')
    sp.add_argument('--no-elite-seed', dest='no_elite_seed', action='store_true',
                    default=False, help='关闭精英回流（不从台账 winner 变异撒种）')
    sp.add_argument('--elite-seed-frac', type=float, default=0.25,
                    help='每次撒种尝试用历史 winner 变异的概率（默认 0.25）')
    sp.add_argument('--no-leg-library', dest='leg_library', action='store_false',
                    default=True, help='关闭腿库重组（composite 轴恢复纯随机腿）')
    sp.add_argument('--no-crossover', dest='crossover', action='store_false',
                    default=True, help='关闭精英交叉（refine 只用单点变异）')
    sp.add_argument('--crossover-rate', type=float, default=0.30)
    sp.add_argument('--n', type=int, default=40, help='撒种候选数（按机制轴分配）')
    sp.add_argument('--seed', type=int, default=1)
    sp.add_argument('--top', type=int, default=12)
    sp.add_argument('--cycle', type=int, default=3)
    sp.add_argument('--cut', default=HOLDOUT_START)
    sp.add_argument('--no-exclude', action='store_true')
    sp.add_argument('--proposals', action='append', default=None,
                    help='提案 JSONL（intake 格式），直接走全量评估，可重复')
    sp.add_argument('--from-ledger', action='store_true',
                    help='把台账中 proposed 但未评估的候选纳入全量评估')
    sp.add_argument('--include-custom', action='store_true', default=False,
                    help='纳入custom_registry自定义因子（公式型走全量，Python型走custom通道并计入检验分母）')
    sp.add_argument('--fast', action='store_true', default=True)
    sp.add_argument('--no-fast', dest='fast', action='store_false')
    sp.add_argument('--fast-stocks', type=int, default=2500)
    sp.add_argument('--fast-months', type=int, default=26)
    sp.add_argument('--fast-stratify', choices=('', 'size'), default='',
                    help='快评抽样方式（T4）：size=按市值五分位分层（修小票稀释），'
                         '默认空=纯随机（保持历史校准可比）')
    sp.add_argument('--fast-keep-sia', type=float, default=None,
                    help='快评 SiA 保留门；profile 默认 local-vwap=0（不以 C2C SiA 阈值淘汰），'
                         'competition-c2c=0.0012；全评仍受 --full-cap 预算限制')
    sp.add_argument('--fast-top-frac', type=float, default=0.5)
    sp.add_argument('--full-cap', type=int, default=60)
    sp.add_argument('--refine-elites', type=int, default=2)
    sp.add_argument('--refine-children', type=int, default=2)
    sp.add_argument('--mutation-rate', type=float, default=0.35)
    sp.add_argument('--oos-sia', type=float, default=None,
                    help='train SiA 达此阈值才计算 holdout（0=关闭；profile 默认 local-vwap=0，'
                         'competition-c2c=0.004）')
    sp.add_argument('--stall', type=int, default=0, help='连续无改进多少个后早停（0=关）')
    sp.add_argument('--no-pool-corr', dest='pool_corr', action='store_false',
                    help='跳过池相关计算（更快）')
    sp.add_argument('--resid', action='store_true', default=True,
                    help='计算风险轴残差 IC 诊断（默认开）')
    sp.add_argument('--no-resid', dest='resid', action='store_false')
    sp.add_argument('--resid-weight', type=float, default=0.0,
                     help='残差掉幅惩罚权重（默认0=仅诊断；0.5=掉幅100%%时降权一半）')
    sp.add_argument('--no-sparse-tie-gate', dest='no_sparse_tie_gate',
                    action='store_true', default=False,
                    help='关闭生成期稀疏并列拦截（默认开：截面独立值中位数<100 的候选不进全量）')
    sp.add_argument('--local-context', dest='local_context', action='store_true', default=None,
                    help='启用 MKT_RET/IND_RET/BETA60/DISP 本地字段与上下文模板（默认开启）')
    sp.add_argument('--no-local-context', dest='local_context', action='store_false',
                    help='关闭本地上下文字段/模板（仅做对照；不代表 C2C 是搜索能力边界）')
    sp.add_argument('--greedy-top', type=int, default=0,
                    help='对前N名做贪婪ΔComb选择（0=关；每个trial约数十秒）')
    sp.add_argument('--greedy-add', type=int, default=2)
    sp.add_argument('--greedy-min-gain', type=float, default=0.001)
    sp.add_argument('--refresh-sia', action='store_true', default=False,
                    help='ΔComb 前在 aligned_v1 窗重估池成员 SiA（同窗化后再与 NC 相加）')
    sp.add_argument('--platform-prior', action='store_true', default=False,
                    help='用 out/calibration.json 的 T^b 平台实现率先验乘入 rank_score'
                         '（opt-in；需先跑 platform-ingest + calibrate）')
    sp.add_argument('--no-offset-diag', dest='no_offset_diag', action='store_true',
                    default=False,
                    help='关闭终选成员的采样稳健性诊断（offset 0/1/2 错位截面 RankIC 一致性）')
    sp.add_argument('--no-cond-ic', dest='no_cond_ic', action='store_true',
                    default=False,
                    help='关闭终选成员的条件 RankIC 诊断（热点子集 hot5/hot2 + 市场状态族，'
                         '与主管线 COND-IC 同口径）')
    sp.add_argument('--habitat-track', dest='habitat_track', action='store_true',
                    default=True,
                    help='Class2 栖息地快评通道（默认开）：habitat_score 前列预留晋级名额，'
                         '防线性 IC 误杀股性/风险切分因子')
    sp.add_argument('--no-habitat-track', dest='habitat_track', action='store_false',
                    help='关闭 Class2 栖息地快评通道')
    sp.add_argument('--habitat-frac', type=float, default=0.15,
                    help='栖息地通道晋级名额占 full-cap 比例（默认 0.15）')
    sp.add_argument('--no-store-sync', dest='no_store_sync', action='store_true',
                    default=False,
                    help='跳过 factors.parquet 存量指纹同步（默认同步防重复挖矿）')
    sp.add_argument('--max-decay', type=float, default=None,
                    help='可选硬门槛：后半段/前半段 RankIC 比值 ≥ 该值才入短名单'
                         '（如 0.6；默认关=仅诊断输出 decay 列）')
    sp.add_argument('--min-rank-ic', type=float, default=0,
                    help='可选硬门槛：主标签 |RankIC| ≥ 该值才入短名单（默认 0=关；'
                         '运气紧缩参考值约 0.003-0.005）')
    sp.add_argument('--family-cap', type=int, default=3,
                    help='主 profile 短名单每 family 最多 N 席（族内多样性约束，防单族垄断；'
                         '默认 3；不足时由其他族递补）')
    sp.add_argument('--report', action='store_true', default=True)
    sp.add_argument('--no-report', dest='report', action='store_false')
    sp.add_argument('--progress-file', default=None,
                    help='进度日志文件（逐候选一行；stdout 同时打印）')
    sp.set_defaults(func=cmd_search)

    ep = sub.add_parser('export')
    ep.add_argument('shortlist')
    ep.add_argument('--list', choices=('c2c', 'vwap', 'shortlist'), default='c2c',
                    help='本地榜单来源；导出前再按 C2C 平台能力过滤（默认竞赛榜）')
    ep.add_argument('--top', type=int, default=8)
    ep.add_argument('--tag', default='mg1')
    ep.add_argument('--out', default='')
    ep.add_argument('--cycle', type=int, default=3)
    ep.add_argument('--start', default='20210901')
    ep.add_argument('--end', default=None,
                    help='平台回测窗上界（默认跟随 --data-cut，即 %s；'
                         '显式传 20260821 可复现旧平台读数）' % W.DEFAULT_DATA_CUT)
    ep.add_argument('--per-run', type=int, default=2)
    ep.add_argument('--min-holdout-sia', type=float, default=0.0)
    ep.add_argument('--max-abs-pool-corr', type=float, default=None)
    ep.add_argument('--note', action='append')
    ep.set_defaults(func=cmd_export)

    up = sub.add_parser('upload')
    up.add_argument('input', help='shortlist_*.json 或 rescore_*.json')
    up.add_argument('--list', choices=('c2c', 'vwap', 'shortlist'), default='c2c',
                    help='取哪个名单（rescore: c2c/vwap；search: shortlist/vwap）')
    up.add_argument('--top', type=int, default=8, help='取前 N 个（默认 8，一批 ≤6 建议）')
    up.add_argument('--tag', default=None, help='文件名/前缀标签（默认取输入文件 stem）')
    up.add_argument('--prefix', default=None, help='平台因子名前缀（默认 tag-）')
    up.add_argument('--out', default='')
    up.add_argument('--min-sia', type=float, default=0.0)
    up.add_argument('--min-net', type=float, default=None)
    up.add_argument('--max-turnover', type=float, default=None)
    up.add_argument('--start', default='20210901')
    up.add_argument('--end', default=None,
                    help='平台回测窗上界（默认跟随 --data-cut，即 %s）' % W.DEFAULT_DATA_CUT)
    up.add_argument('--cycle', type=int, default=3)
    up.add_argument('--group-number', type=int, default=10)
    up.add_argument('--round-trip', type=float, default=0.003)
    up.add_argument('--hypotheses', type=int, default=0)
    up.add_argument('--batch-script', default=str(BATCH_SCRIPT_DEFAULT))
    up.add_argument('--run', action='store_true',
                    help='显式批准：执行平台分析（消耗 credit，约 2.2/个）；默认 create-only 免费')
    up.add_argument('--print-only', action='store_true', help='只写文件与打印命令，不调 batch.py')
    up.set_defaults(func=cmd_upload)

    ip = sub.add_parser('intake')
    ip.add_argument('file')
    ip.add_argument('--limit', type=int, default=0)
    ip.add_argument('--no-smoke', action='store_true')
    ip.add_argument('--profile', choices=('local-vwap', 'competition-c2c'),
                    default='local-vwap')
    ip.add_argument('--out', default='')
    ip.set_defaults(func=cmd_intake)

    pp = sub.add_parser('propose')
    pp.add_argument('hypothesis')
    pp.add_argument('--k', type=int, default=8)
    pp.add_argument('--profile', choices=('local-vwap', 'competition-c2c'),
                    default='local-vwap')
    pp.add_argument('--inbox', default='')
    pp.add_argument('--out', default='')
    pp.set_defaults(func=cmd_propose)

    rp = sub.add_parser('report')
    rp.add_argument('shortlist')
    rp.add_argument('--out', default='')
    rp.set_defaults(func=cmd_report)

    rs = sub.add_parser('rescore')
    rs.add_argument('--top', type=int, default=20, help='c2c 入选名单长度')
    rs.add_argument('--vwap-top', type=int, default=20, help='vwap 名单长度')
    rs.add_argument('--vwap-sort', choices=('rank_sia', 'sia', 'ic', 'icir', 'rank_ic', 'mono'),
                    default='rank_sia',
                    help='vwap 名单排序键：rank_sia=|RankIC|×|RankICIR|×rank胜率（默认，适合树模型/topN）；'
                         'sia=官方同形；mono=单调性')
    rs.add_argument('--turnover-ref', type=float, default=0.3)
    rs.add_argument('--no-turnover-penalty', action='store_true')
    rs.add_argument('--full-top', type=int, default=20,
                    help='对 top N（含 vwap top N）做全窗复评（0=只用台账已有指标）')
    rs.add_argument('--label', action='append', default=None)
    rs.add_argument('--cycle', type=int, default=3)
    rs.set_defaults(func=cmd_rescore)

    lp = sub.add_parser('ledger')
    lp.set_defaults(func=cmd_ledger)

    pi = sub.add_parser('platform-ingest',
                        help='batch.py 产物 → platform 事件回填台账（消除平台反馈断链）')
    pi.add_argument('report', help='candidates_<tag>.report.csv')
    pi.add_argument('--meta', default='', help='默认取 report 同名 .meta.json')
    pi.add_argument('--txt', default='', help='默认取 meta.file 指向的 candidates txt')
    pi.set_defaults(func=cmd_platform_ingest)

    cal = sub.add_parser('calibrate', help='platform↔local 配对拟合 a×local_sia×T^b')
    cal.add_argument('--min-n', type=int, default=5)
    cal.add_argument('--out', default='')
    cal.set_defaults(func=cmd_calibrate)

    dc = sub.add_parser('decay-check',
                        help='衰减监控：台账 winners 全窗重评 + decay_ratio/roll_min/连负段')
    dc.add_argument('--top', type=int, default=50, help='按旧 SiA 取前 N 个候选')
    dc.add_argument('--max-decay', type=float, default=None,
                    help='标记阈值：decay_ratio 低于该值打 ⚠衰减（如 0.6）')
    dc.add_argument('--min-sia-ratio', type=float, default=None,
                    help='标记阈值：新/旧 SiA 比低于该值打 ⚠退步（如 0.5）')
    dc.set_defaults(func=cmd_decay_check)

    fp = sub.add_parser('fixtures')
    fp.set_defaults(func=cmd_fixtures)

    args = ap.parse_args()
    # 平台回测窗上界跟随数据截止日（2026-10-11）：旧默认 '20260821' 会让
    # batch.py 在**含 2026 的窗**上算平台指标，进而污染 platform 事件与
    # calibration.json。默认改为 data_cut；显式传值可复现旧读数。
    if getattr(args, 'end', None) in (None, '') and hasattr(args, 'data_cut'):
        args.end = W.normalize_data_cut(args.data_cut).replace('-', '')
    args.func(args)


if __name__ == '__main__':
    main()
