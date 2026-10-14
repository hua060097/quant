"""本地筛选上下文：复用 factor_screen_cycle5.screen_one，产出平台同口径指标。

- 口径 aligned_v1（2026-09-20 经 qs_platform_align3 用 qs2-R741 官方读数校准）：
  窗口 2021-09-01~2026-08-21（≈平台"入池日前五年"，硬编码常量，平台滚动时需手动更新）、
  全样本宇宙（不剔 ST/一字板/停牌，与平台分析一致）、每 cycle 交易日调仓、
  因子值逐截面 zscore 后 clip ±2σ（screen_one winsor='z2'）。
  校准结果：本地 IC/ICIR 与平台差 ≤8%（RankIC 恒 ≈+14% 系统偏差，来自样本差异）。
- 标签：c2c_Nd（平台口径）；vwap（本地生产口径，自用 LightGBM，非官方）。
- 方向冻结：训练窗确定方向后，holdout 用同一窗口评估（翻转后按 direction=1 报告）。
- 单因子阶段：不做换手硬门槛（官方 A/B 不含换手/净）；排序含换手软惩罚
  turnover_ref/max(Turnover_month, ref)（官方 C 口径的 T_month，默认 ref=0.3=官方 floor）。
- 官方 A（sia_of）= |RankIC|×|ICIR(Pearson)|×IC胜率；净收益/夏普/回撤为 C 代理诊断，
  不参与 A 计分（competition_rules.md 0811 口径）。
- vwap 非必过：仅作诊断/自用 LightGBM 输出。
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import Evaluator, load_panel
from factor_screen_cycle5 import ONE_WAY_COST, screen_one
from grammar import Node, render, walk

START = '20210901'
END = '20260821'
HOLDOUT_START = '20250701'
WINSOR = 'z2'
DEFAULT_TURNOVER_REF = 0.3
LABELS = ('c2c', 'vwap')


def build_label_series(frame: pd.DataFrame, kind: str, cycle: int) -> pd.Series:
    """标签注册表。

    - c2c : close(T+N)/close(T)-1（平台口径，已校准）
    - vwap: 本地生产口径（daily_atr_label.py 原义）：
            buy(T)=open(T+1)；sell(T)=0.7*TP(T+2)+0.3*TP(T+3)，TP=(O+H+L+C)/4；
            label=(sell-buy)/(buy+1e-12)。固定为 T+2/T+3，仅支持 cycle=3。
            注意：本地生产还叠加 ST窗口/涨跌停/amount<1e7/复权掩码，本实现按用户提供的
            公式原义只含公式部分；如需生产掩码需另行启用（未实现）。
    """
    codes = frame['stock_code']
    if kind == 'c2c':
        close = frame['CLOSE']
        return close.groupby(codes, sort=False).shift(-cycle) / close - 1.0
    if kind == 'vwap':
        if int(cycle) != 3:
            raise ValueError('vwap 标签固定为 T+1 买入、T+2/T+3 卖出，仅支持 cycle=3')
        tp = (frame['OPEN'] + frame['HIGH'] + frame['LOW'] + frame['CLOSE']) / 4.0
        buy = frame['OPEN'].groupby(codes, sort=False).shift(-1)
        sell_t2 = tp.groupby(codes, sort=False).shift(-2)
        sell_t3 = tp.groupby(codes, sort=False).shift(-3)
        sell = 0.7 * sell_t2 + 0.3 * sell_t3
        # 2026-09-24 修复：buy 极小值/非正时置 NaN（旧实现 buy+1e-12 会把脏数据
        # 放大成巨大有限标签污染截面；np.isfinite 只挡 inf 不挡 1e6 量级假标签）
        return pd.Series(
            np.where(buy > 1e-6, (sell - buy) / buy, np.nan), index=buy.index)
    raise KeyError(f'unknown label {kind!r}')


class DataContext:
    """一次加载，多个因子共用；rows(node) 返回与 screen 行对齐的因子值。

    offset（2026-09-24）：截面锚点错位（0=主口径）。0≤offset<cycle；用于采样稳健性
    诊断（错位三组各评一遍看 RankIC 是否一致），主口径与平台对齐不变。
    """

    def __init__(self, cycle: int = 3, start: str = START, end: str = END,
                 universe: bool = False, frame: pd.DataFrame | None = None,
                 label: str = 'c2c', evaluator: Evaluator | None = None,
                 offset: int = 0):
        self.cycle = int(cycle)
        self.offset = int(offset)
        if not 0 <= self.offset < self.cycle:
            raise ValueError(f'offset must be in [0, {self.cycle}), got {offset}')
        self.label_kind = str(label)
        frame = load_panel() if frame is None else frame
        self.evaluator = evaluator if evaluator is not None else Evaluator(frame)
        codes = frame['stock_code']
        lab_series = build_label_series(frame, self.label_kind, self.cycle)
        label = lab_series
        dates = frame['date']
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if universe:
            one_bar = (frame['is_limit_up'] == 1) & (frame['OPEN'] == frame['HIGH'])
            mask &= (frame['is_can_buy'] == 1) & (frame['is_st'] == 0) & (~one_bar)
        cal = np.array(sorted(dates[mask].unique()))
        mask &= dates.isin(set(cal[self.offset::self.cycle]))
        lab = label.to_numpy(dtype=float)
        mask &= np.isfinite(lab)
        pos = np.flatnonzero(mask.to_numpy())
        self._pos = pos
        self.stock_codes = frame['stock_code'].to_numpy()[pos]
        self.row_dates = frame['date'].to_numpy()[pos]
        self.labels = lab[pos]
        self.ud = np.array(sorted(set(self.row_dates)))
        self.di = np.searchsorted(self.ud, self.row_dates).astype(np.int64)
        self.dmonths = np.array([pd.Timestamp(x).strftime('%Y-%m') for x in self.ud])
        self.dyears = np.array([pd.Timestamp(x).year for x in self.ud])
        self.ppy = 252.0 / self.cycle
        self.label_name = f'{self.label_kind}_{self.cycle}d'
        self.n_rows = len(pos)

    def values(self, node: Node) -> np.ndarray:
        return self.evaluator.values(node)[self._pos]

    def split_mask(self, cut: str = HOLDOUT_START) -> tuple[np.ndarray, np.ndarray]:
        cut_ts = np.datetime64(pd.Timestamp(cut))
        holdout = self.row_dates >= cut_ts
        return ~holdout, holdout

    def screen(self, source, name: str = 'factor', mask: np.ndarray | None = None,
               fixed_direction: int | None = None) -> dict:
        values = source if isinstance(source, np.ndarray) else self.values(source)
        if mask is None:
            v, y, c = values, self.labels, self.stock_codes
            di, ud, dm, dy = self.di, self.ud, self.dmonths, self.dyears
        else:
            v, y, c = values[mask], self.labels[mask], self.stock_codes[mask]
            rd = self.row_dates[mask]
            ud = np.array(sorted(set(rd)))
            di = np.searchsorted(ud, rd).astype(np.int64)
            dm = np.array([pd.Timestamp(x).strftime('%Y-%m') for x in ud])
            dy = np.array([pd.Timestamp(x).year for x in ud])
        if fixed_direction == 0:
            v = -v
        r = screen_one(np.asarray(v, dtype=float), np.asarray(y, dtype=float), c, di, len(ud),
                       dm, dy, name, self.label_name, self.ppy, winsor=WINSOR)
        r['factor'] = name
        if fixed_direction is not None:
            r['direction'] = int(fixed_direction)
        return r


def pick_stocks(frame: pd.DataFrame, n_stocks: int = 2500, seed: int = 11,
                stratify: str | None = None) -> list[str]:
    """快评抽样选股（确定性）：返回被抽中的 stock_code 列表（已排序）。

    stratify='size' 按股票总市值中位数五分位分层（修小票稀释）；默认 None=纯随机。
    抽股逻辑与 `build_fast_frame` 完全一致——后者调用本函数，故抽出的股票集合**逐只相同**。
    `bench_fast_precision` 用它在**完整窗**上构造同一批抽样股票，以分离"抽样"与"窗口缩短"。
    """
    codes = sorted(frame['stock_code'].unique().tolist())
    rng = random.Random(seed)
    if stratify == 'size':
        med = frame.groupby('stock_code', sort=False)['MARKET_CAP'].median().reindex(codes)
        quint = pd.qcut(med.rank(method='first'), 5, labels=False)
        layer_counts = quint.value_counts().to_dict()
        alloc = {k: max(1, int(round(n_stocks * v / max(1, len(codes)))))
                 for k, v in layer_counts.items()}
        diff = n_stocks - sum(alloc.values())
        order_layers = sorted(alloc, key=lambda k: -layer_counts[k])
        i = 0
        while diff != 0 and order_layers:
            k = order_layers[i % len(order_layers)]
            if diff > 0:
                alloc[k] += 1
                diff -= 1
            elif alloc[k] > 1:
                alloc[k] -= 1
                diff += 1
            i += 1
        pick: set = set()
        for k in sorted(alloc):
            layer_codes = sorted(quint[quint == k].index.tolist())
            pick.update(rng.sample(layer_codes, min(alloc[k], len(layer_codes))))
    else:
        pick = set(rng.sample(codes, min(n_stocks, len(codes))))
    return sorted(pick)


def build_fast_frame(frame: pd.DataFrame, n_stocks: int = 2500, months: int = 26,
                     seed: int = 11, warmup_months: int = 15,
                     stratify: str | None = None,
                     end: str = END) -> pd.DataFrame:
    """一级评估用的抽样面板：股票抽样 + 近 N 月（含垫底期）。

    stratify='size'（T4，2026-09-25）：按股票总市值中位数五分位分层抽样——纯随机抽样会
    稀释小票，使小市值因子的快评读数失真。默认 None=纯随机（保持历史校准可比）。

    end（2026-10-11）：评估窗上界 = **数据截止日**（默认模块 END，由调用方传入
    `search --data-cut` 的值）。窗口随截止日移动，故值指纹等派生量亦随之变化。
    """
    pick = set(pick_stocks(frame, n_stocks=n_stocks, seed=seed, stratify=stratify))
    sub = frame[frame['stock_code'].isin(pick)].copy()
    end_ts = pd.Timestamp(end)
    warm = (end_ts - pd.DateOffset(months=months + warmup_months)).strftime('%Y%m%d')
    sub = sub[(sub['date'] >= pd.Timestamp(warm)) & (sub['date'] <= end_ts)]
    return sub.reset_index(drop=True)


def build_fast_context(frame: pd.DataFrame, n_stocks: int = 2500, months: int = 26,
                       cycle: int = 3, seed: int = 11, warmup_months: int = 15,
                       label: str = 'c2c', evaluator: Evaluator | None = None,
                       stratify: str | None = None,
                       end: str = END) -> DataContext:
    """两级评估的一级：股票抽样 + 近 N 月评估窗（保留垫底期），口径与全量一致。

    主要提速来自股票抽样（行数 ~1/3），窗口缩短只影响屏幕段数。方向/门槛仍由全量窗复核。
    end（2026-10-11）：评估窗上界 = 数据截止日（见 `build_fast_frame`）。
    """
    sub = build_fast_frame(frame, n_stocks=n_stocks, months=months, seed=seed,
                           warmup_months=warmup_months, stratify=stratify, end=end)
    end_ts = pd.Timestamp(end)
    start = (end_ts - pd.DateOffset(months=months)).strftime('%Y%m%d')
    return DataContext(cycle=cycle, start=start, end=end, frame=sub, label=label,
                       evaluator=evaluator)


def sia_of(metrics: dict) -> float:
    return abs(metrics.get('rank_ic_mean') or 0.0) * abs(metrics.get('ic_ir') or 0.0) \
        * (metrics.get('ic_win_rate') or 0.0)


def tier1(metrics: dict) -> dict:
    turn = metrics.get('turnover')
    turn_m = metrics.get('turnover_month')
    net = metrics.get('net_excess_ann')
    sharpe = metrics.get('long_sharpe')
    mdd = metrics.get('long_max_drawdown')
    tier = {
        'valid': bool(metrics.get('valid')),
        'sia': sia_of(metrics),
        'rank_ic': metrics.get('rank_ic_mean'),
        'ic': metrics.get('ic_mean'),
        'ic_ir': metrics.get('ic_ir'),
        'win': metrics.get('ic_win_rate'),
        'rank_ic_ir': metrics.get('rank_ic_ir'),
        'rank_ic_win': metrics.get('rank_ic_win_rate'),
        'turnover': turn,
        'turnover_month': turn_m,
        'cost': metrics.get('annual_cost'),
        'net': net,
        'long': metrics.get('long_excess_ann'),
        'sharpe': sharpe,
        'mdd': mdd,
        'mono': metrics.get('monotonicity'),
    }
    tier['c_proxy'] = c_proxy_of(tier)
    return tier


def c_proxy_of(tier: dict) -> float:
    """官方 C 公式的单因子代理（同形，声明为本地代理）：

    rawC = max(net,0)/max(Turnover_month,0.3) × SR × (1-1.2×MaxDD)
    其中 SR/MaxDD 用本因子多头组合的周期序列代理（非日频账本）。
    """
    net = tier.get('net')
    tm = tier.get('turnover_month')
    sr = tier.get('sharpe')
    mdd = tier.get('mdd')
    if any(v is None or not np.isfinite(v) for v in (net, tm, sr, mdd)):
        return float('nan')
    return float(max(float(net), 0.0) / max(float(tm), 0.3) * float(sr)
                 * (1.0 - 1.2 * float(mdd)))


def turnover_penalty(tier: dict, ref: float = DEFAULT_TURNOVER_REF) -> float:
    """换手软惩罚：T_month 越大分越低；T_month ≤ ref 时不罚。

    T_month 缺省时用 7×单次换手近似（cycle=3 每月约 7 个调仓日）。
    """
    tm = tier.get('turnover_month')
    if tm is None or not np.isfinite(tm):
        t = tier.get('turnover')
        if t is None or not np.isfinite(t):
            return 1.0
        tm = 7.0 * float(t)
    return float(ref) / max(float(tm), float(ref))


def stability_of(metrics: dict) -> float:
    """稳健性 tiebreak：分年 IC 按方向对齐后的最小值（越接近正越好）。"""
    years = metrics.get('yearly_ic') or {}
    ic = metrics.get('ic_mean')
    if not years or ic is None or not np.isfinite(ic):
        return float('nan')
    sgn = 1.0 if ic > 0 else -1.0
    vals = [sgn * float(v) for v in years.values() if v is not None and np.isfinite(v)]
    return float(min(vals)) if vals else float('nan')


def node_count(node: Node) -> int:
    return sum(1 for _ in walk(node))


def complexity_factor(node: Node) -> float:
    """复杂度正则（Alpha-GPT GP 防过拟合）：节点数超8后按2%/节点衰减，下限不变负。"""
    return 1.0 / (1.0 + 0.02 * max(0, node_count(node) - 8))


def rank_score(tier: dict, node: Node, pool_rho: float | None = None,
               oos_sia: float | None = None, resid_drop: float | None = None,
               resid_weight: float = 0.0,
               turnover_ref: float | None = DEFAULT_TURNOVER_REF,
               platform_prior: float | None = None) -> float:
    """综合排序分 = SiA(A代理) × 复杂度因子 × 换手软惩罚 × 池相关惩罚 × 残差掉幅惩罚 × 平台先验。

    - oos_sia 由训练窗方向冻结后在保留窗计算；未提供则不惩罚。
    - pool_rho 为与池内成员的最大|Spearman|，>0.4 后线性降权，1.0 时降到 0.5 倍。
    - resid_drop 为风险轴正交化后 SiA 的相对掉幅（0~1）；resid_weight 默认 0（仅诊断）。
    - turnover_ref 为官方 C 口径的换手软惩罚参考（T_month≤ref 不罚，默认 0.3=官方 floor）；
      None 时不施加换手惩罚（用于对照）。
    - platform_prior 为平台实现率相对乘子（calibrate 的 T^b 项，clip [0.5,2.0]）；
      None 时不施加（默认，opt-in 由 search --platform-prior 传入）。
    """
    base = float(tier.get('sia') or 0.0)
    if oos_sia is not None:
        base = min(base, max(0.0, float(oos_sia)))
    score = base * complexity_factor(node)
    if turnover_ref is not None:
        score *= turnover_penalty(tier, turnover_ref)
    if pool_rho is not None and np.isfinite(pool_rho) and abs(pool_rho) > 0.4:
        score *= 1.0 - 0.5 * min(1.0, (abs(pool_rho) - 0.4) / 0.6)
    if resid_drop is not None and np.isfinite(resid_drop) and resid_weight > 0:
        score *= max(0.0, 1.0 - float(resid_weight) * min(1.0, max(0.0, float(resid_drop))))
    if platform_prior is not None and np.isfinite(platform_prior):
        score *= float(platform_prior)
    return float(score)


def decay_metrics(ctx: DataContext, values: np.ndarray, roll: int = 60) -> dict:
    """样本内衰减诊断（2026-09-25）：从逐截面 RankIC 序列派生，不进 rank_score。

    - decay_ratio：后半段均值/前半段均值（方向对齐后；<1=衰减趋势，≈1=平稳，
      >1=增强）。区分"趋势衰减"与"周期波动"要看 roll_min 与分段比是否同向。
    - roll_min：滚动 roll 期对齐 IC 的最小值（最差时期有多差）
    - max_neg_run：方向对齐后 IC 连续为负的最长段（期数）
    """
    from scipy.stats import rankdata
    order = np.argsort(ctx.di, kind="stable")
    v = np.asarray(values, dtype=float)[order]
    y = np.asarray(ctx.labels, dtype=float)[order]
    d = ctx.di[order]
    n_day = np.bincount(d)
    D = len(n_day)
    seg = np.concatenate([[0], np.cumsum(n_day)])
    ic = np.full(D, np.nan)
    for k in range(D):
        a, b = int(seg[k]), int(seg[k + 1])
        vv, yy = v[a:b], y[a:b]
        ok = np.isfinite(vv) & np.isfinite(yy)
        if ok.sum() < 30:
            continue
        rv = rankdata(vv[ok])
        ry = rankdata(yy[ok])
        rv = rv - rv.mean()
        ry = ry - ry.mean()
        den = float(np.sqrt((rv * rv).sum() * (ry * ry).sum()))
        ic[k] = float((rv * ry).sum() / den) if den > 0 else np.nan
    fin = np.isfinite(ic)
    empty = {'ic_first': None, 'ic_second': None, 'decay_ratio': None,
             'roll_min': None, 'max_neg_run': None, 'n_sections': int(fin.sum())}
    if fin.sum() < 20:
        return empty
    sgn = 1.0 if np.nanmean(ic) >= 0 else -1.0
    ical = np.where(fin, ic * sgn, np.nan)
    half = max(1, D // 2)
    first = float(np.nanmean(ical[:half]))
    second = float(np.nanmean(ical[half:]))
    decay_ratio = (second / first) if (np.isfinite(first) and first > 0) else None
    roll_min = None
    if fin.sum() >= max(10, roll // 2):
        s = pd.Series(ical).rolling(roll, min_periods=max(5, roll // 2)).min()
        roll_min = float(s.min())
    neg = (ical < 0) & fin
    run = mx = 0
    for x in neg:
        run = run + 1 if x else 0
        mx = max(mx, run)
    return {'ic_first': first if np.isfinite(first) else None,
            'ic_second': second if np.isfinite(second) else None,
            'decay_ratio': (float(decay_ratio)
                            if decay_ratio is not None and np.isfinite(decay_ratio) else None),
            'roll_min': roll_min, 'max_neg_run': int(mx), 'n_sections': int(fin.sum())}


_FIELD_DOM_FIELDS = ('CLOSE', 'VOLUME', 'AMOUNT', 'MARKET_CAP', 'TURNOVER', 'RET1')


def field_dominance(ctx: DataContext, values: np.ndarray, min_stocks: int = 30) -> dict:
    """单字段支配诊断（2026-09-25）：因子与 6 个原始字段（CLOSE/VOLUME/AMOUNT/
    MARKET_CAP/TURNOVER/RET1）的逐截面 Spearman 秩相关均值，取 max |ρ|。

    max > 0.9 = 单字段马甲（复杂结构只是换装，机制 = 该字段本身）；
    0.6-0.9 = 字段主导（部分新意）；< 0.6 = 真复合。仅诊断，不进 rank_score。
    字段秩按 ctx 缓存（全候选共享）；Amihud 类量纲混算因子落中间档，不误杀。"""
    from scipy.stats import rankdata
    cache = getattr(ctx, '_fdom_cache', None)
    if cache is None:
        order = np.argsort(ctx.di, kind="stable")
        d = ctx.di[order]
        D = int(d.max()) + 1 if len(d) else 0
        n_day = np.bincount(d, minlength=D)
        seg = np.concatenate([[0], np.cumsum(n_day)])
        franks = {}
        for f in _FIELD_DOM_FIELDS:
            fv = ctx.values(Node('FIELD', name=f))[order]
            r = np.full(len(fv), np.nan)
            for k in range(D):
                a, b = int(seg[k]), int(seg[k + 1])
                ok = np.isfinite(fv[a:b])
                if ok.sum() < min_stocks:
                    continue
                r[np.flatnonzero(ok) + a] = rankdata(fv[a:b][ok]) / float(ok.sum())
            franks[f] = r
        cache = {'order': order, 'd': d, 'D': D, 'n_day': n_day,
                 'seg': seg, 'franks': franks, 'min_stocks': min_stocks}
        ctx._fdom_cache = cache
    order = cache['order']
    d = cache['d']
    D = cache['D']
    seg = cache['seg']
    v_sorted = np.asarray(values, dtype=float)[order]
    r = np.full(len(v_sorted), np.nan)
    for k in range(D):
        a, b = int(seg[k]), int(seg[k + 1])
        ok = np.isfinite(v_sorted[a:b])
        if ok.sum() < min_stocks:
            continue
        r[np.flatnonzero(ok) + a] = rankdata(v_sorted[a:b][ok]) / float(ok.sum())
    out = {}
    for f, fr in cache['franks'].items():
        ok = np.isfinite(r) & np.isfinite(fr)
        dok = d[ok]
        if len(dok) < 10:
            out[f] = np.nan
            continue
        nn = np.maximum(np.bincount(dok, minlength=D), 1)
        s_f = np.bincount(dok, weights=r[ok], minlength=D)
        s_g = np.bincount(dok, weights=fr[ok], minlength=D)
        s_ff = np.bincount(dok, weights=r[ok] * r[ok], minlength=D)
        s_gg = np.bincount(dok, weights=fr[ok] * fr[ok], minlength=D)
        s_fg = np.bincount(dok, weights=r[ok] * fr[ok], minlength=D)
        with np.errstate(invalid='ignore', divide='ignore'):
            cov = s_fg - s_f * s_g / nn
            den = np.sqrt(np.maximum(s_ff - s_f * s_f / nn, 0.0)
                          * np.maximum(s_gg - s_g * s_g / nn, 0.0))
            corr = np.where((den > 0) & (nn >= min_stocks), cov / np.maximum(den, 1e-12),
                            np.nan)
        fin = corr[np.isfinite(corr)]
        out[f] = float(np.abs(fin).mean()) if len(fin) else np.nan
    max_field = max(out, key=lambda k: (out[k] if np.isfinite(out[k]) else -1))
    return {'per_field': out, 'max': float(out[max_field]), 'field': max_field,
            'n_sections': int(np.isfinite(r).sum())}


def train_holdout(ctx: DataContext, node: Node, name: str = 'factor',
                  cut: str = HOLDOUT_START) -> dict:
    train_mask, holdout_mask = ctx.split_mask(cut)
    train = ctx.screen(node, name=name, mask=train_mask)
    direction = int(train.get('direction', 1))
    holdout = ctx.screen(node, name=name, mask=holdout_mask, fixed_direction=direction)
    return {'name': name, 'formula': render(node), 'direction': direction,
            'train': train, 'holdout': holdout}


def main() -> None:
    import time

    ap = argparse.ArgumentParser(description='本地筛选冒烟：训练/保留双窗指标')
    ap.add_argument('--factor', default='slope60', help='fixtures.py 中的回归夹具名')
    ap.add_argument('--cut', default=HOLDOUT_START)
    args = ap.parse_args()
    from fixtures import FIXTURES

    t0 = time.time()
    ctx = DataContext()
    print(f'ctx rows={ctx.n_rows:,} sections={len(ctx.ud)} '
          f'{pd.Timestamp(ctx.ud[0]).date()}~{pd.Timestamp(ctx.ud[-1]).date()} {time.time() - t0:.0f}s')
    node = FIXTURES[args.factor]()
    out = train_holdout(ctx, node, name=args.factor, cut=args.cut)
    for key in ('train', 'holdout'):
        r = out[key]
        tier = tier1(r)
        print(f"{key:8s} dir={out['direction']} RIC={tier['rank_ic']} IR={tier['ic_ir']} "
              f"win={tier['win']} turn={tier['turnover']} net={tier['net']} sia={tier['sia']:.4f}")


if __name__ == '__main__':
    main()
