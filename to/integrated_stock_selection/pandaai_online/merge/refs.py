"""第3条因子发现路径：外部已有因子直接筛选（clean-room，aligned_v1 口径）。

与路径1（grammar 机制搜索）、路径2（论文/LLM 提案）并列；结果写同一台账，
rescore/报告天然覆盖三路径。

来源库：
  - directional / risk-pattern / volume-stat：quantskills refs/*/factor_index.json
  - dir-adapt：directional 12 base 长窗适配（60/120/250 × raw/smooth/confirm_volume/
    stability/vol_scaled，qs_dir_adapt 同规格）

口径：与 fitness.DataContext 完全一致（aligned_v1：窗口 2021-09-01~2026-08-21、
全样本、每 3 交易日调仓、因子截面 zscore+clip±2σ），指标复用
factor_screen_cycle5.screen_one。平台公式翻译（upload 用）暂不在此做，按需补。

评分：
- A_official（官方A，c2c 主排序）= |RankIC|×|ICIR(Pearson)|×IC胜率（fitness.sia_of）
- LGB（自用 LightGBM 复合分，vwap 主排序）= |RankIC|×|RankICIR|×rank胜率
  × |mono| × T_ref/max(turnover,T_ref)；turnover=每次调仓的 top 替换率
  （2~3 日持仓口径，非日频），T_ref=0.3；coverage 仅列示不剔除。

台账：每因子×每标签写 local 事件（source='ref-screen'），多重检验分母一并计入。

用法：
  python refs.py                                        # 4 库全量
  python refs.py --libs directional dir-adapt           # 指定库
  python refs.py --libs risk-pattern --top 200          # 按旧证据(screen_5d A_proxy)
                                                        # 每标签前 N 并集预筛
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import load_panel
from fitness import DataContext, sia_of, tier1
from ledger import Ledger

HERE = Path(__file__).resolve().parent
PON = HERE.parent
REFS = PON / 'refs'
SD = PON / 'screen_5d'
OUT_DIR = HERE / 'out'

LIB_REPOS = {
    'directional': REFS / 'skill-quant-factor-directional-alpha' / 'factor_index.json',
    'risk-pattern': REFS / 'skill-quant-factor-risk-pattern-alpha' / 'factor_index.json',
    'volume-stat': REFS / 'skill-quant-factor-volume-stat-alpha' / 'factor_index.json',
}
OLD_FLAT = {
    'directional': 'qs_dir296_dual_flat.csv',
    'risk-pattern': 'qs_risk-pattern_dual_flat.csv',
    'volume-stat': 'qs_volume-stat_dual_flat.csv',
    'dir-adapt': 'qs_dir_adapt_dual_flat.csv',
}
ADAPT_BASES = ['return', 'skip_return', 'reversal', 'sma_gap', 'ema_gap', 'dual_ema_gap',
               'sma_slope', 'range_position', 'breakout', 'breakdown', 'trend_strength',
               'efficiency']
ADAPT_WINDOWS = (60, 120, 250)
ADAPT_TRANSFORMS = ('raw', 'smooth', 'confirm_volume', 'stability', 'vol_scaled')
LGB_TREF = 0.3

# base → 机制轴（grammar.AXIS_OF 的轴名；2026-09-24 起 ref 事件以 ref:<axis> 为 family，
# 使 _axis_plan/UCB 配额能看到这 34% 台账事件；lib/factor_id 等出处保留在 data）
REF_BASE_AXIS = {
    'return': 'trend_position', 'skip_return': 'trend_position', 'reversal': 'trend_position',
    'sma_gap': 'trend_position', 'ema_gap': 'trend_position', 'dual_ema_gap': 'trend_position',
    'sma_slope': 'trend_position', 'range_position': 'trend_position',
    'breakout': 'trend_position', 'breakdown': 'trend_position',
    'trend_strength': 'trend_position', 'gap_sum': 'trend_position',
    'intraday': 'trend_position', 'drawdown': 'trend_position',
    'efficiency': 'shape', 'upper_wick': 'shape', 'lower_wick': 'shape',
    'rsi': 'tech', 'rsi_reversal': 'tech', 'stoch': 'tech',
    'atr_ratio': 'volatility', 'range_ratio': 'volatility',
    'realized_vol': 'volatility', 'downside_vol': 'volatility',
    'volume_ratio': 'liquidity_size', 'volume_z': 'liquidity_size',
    'dollar_volume': 'liquidity_size',
    'price_volume_corr': 'pv_relation', 'obv_slope': 'pv_relation',
    'ts_rank_close': 'order_stats', 'ts_rank_volume': 'order_stats',
    'ret_skew': 'distribution', 'ret_kurt': 'distribution',
}


def ref_axis(spec: dict) -> str:
    base = str(spec.get('base') or '')
    return REF_BASE_AXIS.get(base, 'liquidity_size')
UPLOADED_JSONS = ('qs_dir296_upload.json', 'qs_dir296_upload_c2c.json',
                  'qs_dir296_upload_c2c_b.json', 'qs_dir296_upload_vwap.json',
                  'qs2_upload_qs2-c2c.json', 'qs2_upload_qs2-vwap.json')


def minp(window: int, floor: int = 5) -> int:
    inner = min(floor, window // 2) if window > 2 else 2
    return min(window, max(2, inner))


def sl_minp(window: int) -> int:
    return min(30, max(5, window // 2))


def _slope_apply(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    den = float((x ** 2).sum())

    def calc(v: np.ndarray) -> float:
        if np.isnan(v).any():
            return np.nan
        return float(np.dot(x, v - v.mean()) / den)

    return series.rolling(window, min_periods=window).apply(calc, raw=True)


class FactorBuilder:
    """qs_quantskills_all 的 32 base × 8 transform 本地算子库（pandas 向量化）。"""

    def __init__(self, frame: pd.DataFrame):
        self.sf = frame
        self.g = frame.groupby('stock_code', sort=False)
        self.gc = frame['close'].astype(float)
        self.go = frame['open'].astype(float)
        self.gh = frame['high'].astype(float)
        self.gl = frame['low'].astype(float)
        self.gv = frame['volume'].astype(float)
        frame['_ret'] = self.g['close'].transform(lambda x: x.pct_change()).to_numpy()
        frame['_vpct'] = self.g['volume'].transform(lambda x: x.pct_change()).to_numpy()
        self.base_cache: dict = {}

    def get_base(self, base: str, w: int, p: dict) -> pd.Series:
        key = (base, w)
        if key in self.base_cache:
            return self.base_cache[key]
        sf, g = self.sf, self.g
        gc, go, gh, gl, gv = self.gc, self.go, self.gh, self.gl, self.gv
        mp = minp(w)
        fast = int(p.get('fast', max(3, w // 3)))
        slow = int(p.get('slow', w))
        if base == 'return':
            raw = gc / g['close'].shift(w) - 1.0
        elif base == 'skip_return':
            sk = int(p.get('skip', 5))
            raw = g['close'].shift(sk) / g['close'].shift(sk + w) - 1.0
        elif base == 'reversal':
            raw = -(gc / g['close'].shift(w) - 1.0)
        elif base == 'sma_gap':
            raw = gc / g['close'].transform(
                lambda x: x.rolling(w, min_periods=mp).mean()) - 1.0
        elif base == 'ema_gap':
            raw = gc / g['close'].transform(
                lambda x: x.ewm(span=w, adjust=False, min_periods=minp(w, 3)).mean()) - 1.0
        elif base == 'dual_ema_gap':
            ef = g['close'].transform(
                lambda x: x.ewm(span=fast, adjust=False, min_periods=minp(fast, 3)).mean())
            es = g['close'].transform(
                lambda x: x.ewm(span=slow, adjust=False, min_periods=minp(slow, 3)).mean())
            raw = (ef - es) / gc.replace(0, np.nan)
        elif base == 'sma_slope':
            ma = g['close'].transform(lambda x: x.rolling(w, min_periods=mp).mean())
            raw = _slope_apply(ma, sl_minp(w)) / gc.replace(0, np.nan)
        elif base == 'range_position':
            lo = g['low'].transform(lambda x: x.rolling(w, min_periods=mp).min())
            hi = g['high'].transform(lambda x: x.rolling(w, min_periods=mp).max())
            raw = (gc - lo) / (hi - lo).replace(0, np.nan) - 0.5
        elif base == 'breakout':
            hh = g['high'].transform(lambda x: x.shift(1).rolling(w, min_periods=mp).max())
            raw = gc / hh.replace(0, np.nan) - 1.0
        elif base == 'breakdown':
            ll = g['low'].transform(lambda x: x.shift(1).rolling(w, min_periods=mp).min())
            raw = -(gc / ll.replace(0, np.nan) - 1.0)
        elif base == 'trend_strength':
            vol = g['_ret'].transform(lambda x: x.rolling(w, min_periods=mp).std(ddof=0))
            raw = (gc / g['close'].shift(w) - 1.0) / vol.replace(0, np.nan)
        elif base == 'efficiency':
            path = g['close'].transform(
                lambda x: x.diff().abs().rolling(w, min_periods=mp).sum())
            raw = (gc - g['close'].shift(w)).abs() / path.replace(0, np.nan)
        elif base == 'rsi':
            d = g['close'].transform(lambda x: x.diff())
            ag = d.groupby(sf['stock_code'], sort=False).transform(
                lambda x: x.clip(lower=0).ewm(alpha=1.0 / w, adjust=False,
                                              min_periods=mp).mean())
            al = d.groupby(sf['stock_code'], sort=False).transform(
                lambda x: (-x.clip(upper=0)).ewm(alpha=1.0 / w, adjust=False,
                                                 min_periods=mp).mean())
            rs = ag / al.replace(0, np.nan)
            raw = (100 - 100 / (1 + rs)) / 100 - 0.5
        elif base == 'rsi_reversal':
            d = g['close'].transform(lambda x: x.diff())
            ag = d.groupby(sf['stock_code'], sort=False).transform(
                lambda x: x.clip(lower=0).ewm(alpha=1.0 / w, adjust=False,
                                              min_periods=mp).mean())
            al = d.groupby(sf['stock_code'], sort=False).transform(
                lambda x: (-x.clip(upper=0)).ewm(alpha=1.0 / w, adjust=False,
                                                 min_periods=mp).mean())
            rs = ag / al.replace(0, np.nan)
            raw = 0.5 - (100 - 100 / (1 + rs)) / 100
        elif base == 'stoch':
            lo = g['low'].transform(lambda x: x.rolling(w, min_periods=mp).min())
            hi = g['high'].transform(lambda x: x.rolling(w, min_periods=mp).max())
            raw = (gc - lo) / (hi - lo).replace(0, np.nan) - 0.5
        elif base == 'atr_ratio':
            prev = g['close'].shift(1)
            tr = pd.concat([(gh - gl), (gh - prev).abs(), (gl - prev).abs()],
                           axis=1).max(axis=1)
            tr.name = '_tr_x'
            sf['_tr_x'] = tr.to_numpy()
            raw = g['_tr_x'].transform(
                lambda x: x.rolling(w, min_periods=mp).mean()) / gc.replace(0, np.nan)
        elif base == 'range_ratio':
            rr = ((gh - gl) / gc.replace(0, np.nan))
            rr.name = '_rr_x'
            sf['_rr_x'] = rr.to_numpy()
            raw = g['_rr_x'].transform(lambda x: x.rolling(w, min_periods=mp).mean())
        elif base == 'realized_vol':
            raw = g['_ret'].transform(lambda x: x.rolling(w, min_periods=mp).std(ddof=0))
        elif base == 'downside_vol':
            neg = sf['_ret'].where(sf['_ret'] < 0, 0.0)
            neg.name = '_neg_x'
            sf['_neg_x'] = neg.to_numpy()
            raw = g['_neg_x'].transform(lambda x: x.rolling(w, min_periods=mp).std(ddof=0))
        elif base == 'gap_sum':
            gp = (go / g['close'].shift(1).replace(0, np.nan) - 1.0)
            gp.name = '_gp_x'
            sf['_gp_x'] = gp.to_numpy()
            raw = g['_gp_x'].transform(lambda x: x.rolling(w, min_periods=mp).sum())
        elif base == 'intraday':
            itd = (gc / go.replace(0, np.nan) - 1.0)
            itd.name = '_itd_x'
            sf['_itd_x'] = itd.to_numpy()
            raw = g['_itd_x'].transform(lambda x: x.rolling(w, min_periods=mp).mean())
        elif base == 'upper_wick':
            top = pd.concat([go, gc], axis=1).max(axis=1)
            uw = (gh - top) / (gh - gl).replace(0, np.nan)
            uw.name = '_uw_x'
            sf['_uw_x'] = uw.to_numpy()
            raw = g['_uw_x'].transform(lambda x: x.rolling(w, min_periods=mp).mean())
        elif base == 'lower_wick':
            bot = pd.concat([go, gc], axis=1).min(axis=1)
            lw = (bot - gl) / (gh - gl).replace(0, np.nan)
            lw.name = '_lw_x'
            sf['_lw_x'] = lw.to_numpy()
            raw = g['_lw_x'].transform(lambda x: x.rolling(w, min_periods=mp).mean())
        elif base == 'drawdown':
            raw = gc / g['close'].transform(
                lambda x: x.rolling(w, min_periods=mp).max()) - 1.0
        elif base == 'volume_ratio':
            raw = gv / g['volume'].transform(
                lambda x: x.rolling(w, min_periods=mp).mean()) - 1.0
        elif base == 'volume_z':
            mu = g['volume'].transform(lambda x: x.rolling(w, min_periods=mp).mean())
            sd = g['volume'].transform(lambda x: x.rolling(w, min_periods=mp).std(ddof=0))
            raw = (gv - mu) / sd.replace(0, np.nan)
        elif base == 'dollar_volume':
            dv = gc * gv
            dv.name = '_dv_x'
            sf['_dv_x'] = dv.to_numpy()
            raw = sf['_dv_x'] / g['_dv_x'].transform(
                lambda x: x.rolling(w, min_periods=mp).mean()) - 1.0
        elif base == 'price_volume_corr':
            x, y, code = sf['_ret'], sf['_vpct'], sf['stock_code']
            mx = x.groupby(code, sort=False).transform(
                lambda s: s.rolling(w, min_periods=mp).mean())
            my = y.groupby(code, sort=False).transform(
                lambda s: s.rolling(w, min_periods=mp).mean())
            mxy = (x * y).groupby(code, sort=False).transform(
                lambda s: s.rolling(w, min_periods=mp).mean())
            sx = x.groupby(code, sort=False).transform(
                lambda s: s.rolling(w, min_periods=mp).std(ddof=0))
            sy = y.groupby(code, sort=False).transform(
                lambda s: s.rolling(w, min_periods=mp).std(ddof=0))
            cov = mxy - mx * my
            raw = cov / (sx * sy).replace(0, np.nan)
        elif base == 'obv_slope':
            sgn = np.sign(g['close'].diff()).fillna(0) * gv
            sgn.name = '_sgn_x'
            sf['_sgn_x'] = sgn.to_numpy()
            obv = g['_sgn_x'].transform(lambda x: x.cumsum())
            obv.name = '_obv_x'
            sf['_obv_x'] = obv.to_numpy()
            raw = _slope_apply(sf['_obv_x'], min(w, 60)) / g['volume'].transform(
                lambda x: x.rolling(w, min_periods=mp).mean()).replace(0, np.nan)
        elif base == 'ts_rank_close':
            raw = g['close'].transform(
                lambda x: x.rolling(w, min_periods=mp).rank(method='max', pct=True)) - 0.5
        elif base == 'ts_rank_volume':
            raw = g['volume'].transform(
                lambda x: x.rolling(w, min_periods=mp).rank(method='max', pct=True)) - 0.5
        elif base == 'ret_skew':
            raw = g['_ret'].transform(lambda x: x.rolling(w, min_periods=minp(w, 10)).skew())
        elif base == 'ret_kurt':
            raw = g['_ret'].transform(lambda x: x.rolling(w, min_periods=minp(w, 10)).kurt())
        else:
            raise ValueError(base)
        raw = pd.to_numeric(raw, errors='coerce')
        raw.index = sf.index
        self.base_cache[key] = raw
        return raw

    def get_factor(self, spec: dict) -> pd.Series:
        p = spec.get('params') or {}
        w = int(p['window'])
        raw = self.get_base(spec['base'], w, p)
        tr = spec['transform']
        sf, g, gv = self.sf, self.g, self.gv
        gb = raw.groupby(sf['stock_code'], sort=False)
        norm = int(p.get('norm', max(20, w)))
        if tr == 'raw':
            out = raw
        elif tr == 'z':
            mu = gb.transform(lambda x: x.rolling(norm, min_periods=minp(norm)).mean())
            sd = gb.transform(lambda x: x.rolling(norm, min_periods=minp(norm)).std(ddof=0))
            out = (raw - mu) / sd.replace(0, np.nan)
        elif tr == 'delta':
            lag = int(p.get('lag', max(1, w // 4)))
            out = raw - gb.shift(lag)
        elif tr == 'smooth':
            span = int(p.get('smooth', max(3, w // 3)))
            out = gb.transform(
                lambda x: x.ewm(span=span, adjust=False, min_periods=minp(span, 3)).mean())
        elif tr == 'rank':
            out = gb.transform(
                lambda x: x.rolling(norm, min_periods=minp(norm)).rank(method='max', pct=True)) - 0.5
        elif tr == 'vol_scaled':
            vsd = g['_ret'].transform(lambda x: x.rolling(norm, min_periods=minp(norm)).std(ddof=0))
            out = raw / vsd.replace(0, np.nan)
        elif tr == 'stability':
            ssd = gb.transform(lambda x: x.rolling(norm, min_periods=minp(norm)).std(ddof=0))
            out = raw / ssd.replace(0, np.nan)
        elif tr == 'confirm_volume':
            vma = g['volume'].transform(lambda x: x.rolling(norm, min_periods=minp(norm)).mean())
            out = raw * (gv / vma.replace(0, np.nan))
        elif tr == 'compress':
            out = np.tanh(raw)
        else:
            raise ValueError(tr)
        return pd.to_numeric(out, errors='coerce').replace([np.inf, -np.inf], np.nan)


def dir_adapt_specs() -> list[dict]:
    out = []
    for base in ADAPT_BASES:
        for w in ADAPT_WINDOWS:
            for tr in ADAPT_TRANSFORMS:
                out.append({'factor_id': f'{base}_{w}_{tr}',
                            'formula_cn': f'{tr}({base}, {w})',
                            'name_cn': f'{tr}({base}, {w})',
                            'base': base, 'transform': tr,
                            'params': {'window': w, 'norm': 20, 'smooth': 3, 'skip': 5,
                                       'fast': 3, 'slow': w}})
    return out


def load_lib_specs(lib: str, top: int | None) -> list[dict]:
    if lib == 'dir-adapt':
        specs = dir_adapt_specs()
    else:
        specs = json.loads(LIB_REPOS[lib].read_text(encoding='utf-8'))
    if not top or top <= 0:
        return list(specs)
    flat_path = SD / OLD_FLAT[lib]
    if not flat_path.exists():
        print(f'[{lib}] 旧证据缺失 {flat_path.name}，全量评估', flush=True)
        return list(specs)
    df = pd.read_csv(flat_path)
    keep: set = set()
    for lab in ('c2c_3d', 'vwap_3d'):
        sub = df[df['label'] == lab].sort_values('A_proxy', ascending=False)
        keep.update(sub.head(int(top))['factor_id'].astype(str))
    picked = [s for s in specs if str(s['factor_id']) in keep]
    print(f'[{lib}] 旧证据预筛 top{top}/标签 → {len(picked)}/{len(specs)}', flush=True)
    return picked


def uploaded_map() -> dict:
    out: dict = {}
    for name in UPLOADED_JSONS:
        p = SD / name
        if not p.exists():
            continue
        try:
            j = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        for it in (j.get('shortlist') or []):
            nm = str(it.get('name') or '')
            for pref in ('qsdir-', 'qs2-'):
                if nm.startswith(pref):
                    out.setdefault(str(nm[len(pref):]), nm)
    return out


def lgb_score(m: dict, t_ref: float = LGB_TREF) -> float:
    """自用 LightGBM 复合分（vwap）：rank 版 SiA × |单调性| × 单次调仓换手软惩罚。

    turnover = 每次调仓 top 组替换率（screen_one 口径，2~3 日持仓），
    非日频年化；T_ref=0.3。

    2026-09-26 修复：mono 从 max(0,mono) 改为 abs(mono)——mono=−1 表示完美
    反转单调（信号方向由 dir 标记），不应归零惩罚。|mono| 越高 = 分位收益
    排列越单调 = 信号质量越好。
    """
    turn = m.get('turnover')
    pen = 1.0 if turn is None or not np.isfinite(turn) \
        else float(t_ref) / max(float(turn), float(t_ref))
    mono = m.get('monotonicity')
    mono = abs(float(mono)) if mono is not None and np.isfinite(mono) else 1.0
    ric = abs(m.get('rank_ic_mean') or 0.0)
    rir = abs(m.get('rank_ic_ir') or 0.0)
    rw = m.get('rank_ic_win_rate') or 0.0
    return float(ric * rir * rw * abs(mono) * pen)


METRIC_KEYS = ('rank_ic_mean', 'rank_ic_ir', 'rank_ic_win_rate', 'ic_mean', 'ic_ir',
               'ic_win_rate', 'turnover', 'turnover_month', 'net_excess_ann',
               'long_excess_ann', 'long_sharpe', 'long_max_drawdown', 'monotonicity',
               'n_valid_periods', 'direction')


def main(argv=None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='第3路径：外部已有因子筛选（aligned_v1）')
    ap.add_argument('--libs', nargs='+', default=['directional', 'risk-pattern',
                                                  'volume-stat', 'dir-adapt'],
                    choices=list(LIB_REPOS) + ['dir-adapt'])
    ap.add_argument('--top', type=int, default=0,
                    help='按旧证据每标签前 N 并集预筛；0=全量')
    ap.add_argument('--t-ref', type=float, default=LGB_TREF)
    args = ap.parse_args(argv)

    t0 = time.time()
    frame = load_panel()
    print(f'panel rows={len(frame):,} {time.time() - t0:.0f}s', flush=True)
    ctxs = {lab: DataContext(frame=frame, label=lab) for lab in ('c2c', 'vwap')}
    for lab, ctx in ctxs.items():
        print(f'[{lab}] rows={ctx.n_rows:,} sections={len(ctx.ud)} '
              f'{str(ctx.ud[0])[:10]}~{str(ctx.ud[-1])[:10]}', flush=True)
    builder = FactorBuilder(frame)
    ledger = Ledger()
    uploaded = uploaded_map()

    rows = []
    for lib in args.libs:
        specs = load_lib_specs(lib, args.top)
        print(f'[{lib}] specs={len(specs)}', flush=True)
        for i, spec in enumerate(specs, 1):
            fid = str(spec['factor_id'])
            t1 = time.time()
            rec = {'lib': lib, 'factor_id': fid,
                   'name_cn': spec.get('name_cn') or spec.get('formula_cn'),
                   'base': spec.get('base'), 'transform': spec.get('transform'),
                   'window': (spec.get('params') or {}).get('window'),
                   'formula_cn': spec.get('formula_cn'),
                   'uploaded': uploaded.get(fid, ''), 'labels': {}}
            try:
                f_full = builder.get_factor(spec).to_numpy(dtype=float)
            except Exception as exc:  # noqa: BLE001
                rec['error'] = repr(exc)
                rows.append(rec)
                continue
            for lab, ctx in ctxs.items():
                v = f_full[ctx._pos]
                cov = float(np.isfinite(v).mean()) if len(v) else 0.0
                if cov <= 0.1:
                    rec['labels'][lab] = {'valid': False, 'coverage': cov}
                    continue
                m = ctx.screen(v, name=f'{lib}:{fid}')
                a_off = float(sia_of(m))
                lgb = lgb_score(m, args.t_ref)
                slim = {k: (None if m.get(k) is None or (isinstance(m.get(k), float)
                                                         and not np.isfinite(m.get(k)))
                            else (float(m[k]) if isinstance(m.get(k), (int, float, np.floating))
                                  else m.get(k))) for k in METRIC_KEYS}
                rec['labels'][lab] = {'valid': True, 'coverage': cov,
                                      'A_official': a_off, 'LGB': lgb, **slim}
                ledger.log_raw(
                    'local', str(spec.get('formula_cn') or fid),
                    direction=int(m.get('direction', 1)),
                    # family 编码语义轴：_axis_plan/UCB/evidence_pool 由此识别 ref 证据
                    family=f"ref:{ref_axis(spec)}",
                    data={'source': 'ref-screen', 'lib': lib, 'factor_id': fid,
                          'base': spec.get('base'), 'transform': spec.get('transform'),
                          'window': (spec.get('params') or {}).get('window'),
                          'sia': a_off, 'lgb': lgb, 'coverage': cov,
                          'tier1': tier1(m),
                          'metrics': {k: (float(m[k]) if isinstance(m[k], (int, float, np.floating))
                                          else m[k]) for k in METRIC_KEYS
                                      if m.get(k) is not None}},
                    label=lab)
            rows.append(rec)
            c = rec['labels'].get('c2c', {})
            v = rec['labels'].get('vwap', {})
            print(f'[{lib} {i}/{len(specs)}] {fid:20s} '
                  f'c2c A={c.get("A_official", 0) or 0:.5f} | vwap LGB={v.get("LGB", 0) or 0:.5f} '
                  f'{time.time() - t1:.1f}s', flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
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
                         'label': lab, **{k: r.get(k) for k in
                                          ('coverage', 'A_official', 'LGB')},
                         **{k: r.get(k) for k in METRIC_KEYS}})
    df = pd.DataFrame(flat)
    csv_path = OUT_DIR / f'refs_screen_{ts}_flat.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    json_path = OUT_DIR / f'refs_screen_{ts}.json'
    json_path.write_text(json.dumps({'generated_at': datetime.now().isoformat(timespec='seconds'),
                                     'libs': args.libs, 'top': args.top, 't_ref': args.t_ref,
                                     'rows': rows}, ensure_ascii=False, default=str),
                         encoding='utf-8')
    print(f'saved {csv_path} / {json_path}  total {time.time() - t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
