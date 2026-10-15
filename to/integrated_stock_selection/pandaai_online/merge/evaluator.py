"""本地向量化求值器：平台公式语法 → 逐股时序/截面序列（Phase A 合并引擎）。

- 数据源：cache/stock_features_*.parquet（自动发现），按 (stock_code, date) 排序。
- 语义与历史脚本（new_slow_factors/qs_factory/qs_r3）逐算子对齐；min_periods 显式存于 AST。
- AMOUNT 保持原始元、不做缩放（本地公式=平台公式逐字一致）；RET1/LOGC 由虚拟序列展开。
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from cache_utils import find_stock_features
from field_catalog import (EXTENDED_ONLY_FIELDS, LOCAL_COLUMN, LOCAL_ONLY_FIELDS,
                             PLATFORM_FIELDS, sw2_mapping)
from grammar import Node, virtual_node

_RAW_COLUMNS = ('stock_code', 'date', 'open', 'high', 'low', 'close', 'volume',
                'amount', 'turnover_rate', 'total_mv', 'is_limit_up', 'is_can_buy', 'is_st')


def value_fingerprint(values, n_q: int = 9) -> str:
    """值分布指纹（2026-09-25）：数学等价/仿射等价的因子 → 同指纹。

    - 归一：(v − median) / IQR —— 去装饰常数平移与倍乘（x 与 2x、x−1 同指纹；
      对 rank-IC 搜索而言仿射等价 = 同一排序 = 同一因子，去重语义正确）
    - 镜像规范化：mean<0 时取负（x 与 −x 同指纹；方向由搜索自动检测）
    - 量化：'.3e'（4 位有效数字）吸收浮点噪声；哈希取 sha1 前 12 位
    行数 <500 返回 ''（样本不足不做值域判定）。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 500:
        return ''
    med = float(np.median(v))
    iqr = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = float(np.std(v))
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = 1.0
    z = (v - med) / iqr
    if float(np.mean(z)) < 0:
        z = -z
    qs = np.quantile(z, np.linspace(0.01, 0.99, n_q))
    payload = ','.join(format(float(q), '.3e') for q in qs)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_panel(data_cut: str | None = None) -> pd.DataFrame:
    """读取缓存并映射为平台字段名；含原始交易状态列供宇宙过滤。

    AMOUNT 保持原始元（不做隐藏缩放）：本地公式=平台公式逐字一致。
    AMOUNT 做分母时必须显式 ×1000000000（平台对 ~1e-11 数值处理异常，2026-09-16 探针实证）。

    V4.1：`data_cut=None`保持历史行为（全量读取，不破坏现有调用方）；
    显式传入时用PyArrow filters在I/O层裁剪`date <= data_cut`，并做内存二次裁剪保底。
    """
    fp = str(find_stock_features())
    if data_cut is None:
        raw = pd.read_parquet(fp, columns=list(_RAW_COLUMNS))
    else:
        bound = pd.Timestamp(data_cut)
        try:
            raw = pd.read_parquet(fp, columns=list(_RAW_COLUMNS),
                                  filters=[("date", "<=", bound)])
        except Exception:
            raw = pd.read_parquet(fp, columns=list(_RAW_COLUMNS))
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw[raw["date"] <= bound].copy()
    raw['date'] = pd.to_datetime(raw['date'])
    raw['stock_code'] = raw['stock_code'].astype(str)
    for name, col in LOCAL_COLUMN.items():
        series = pd.to_numeric(raw[col], errors='coerce').astype(float)
        raw[name] = series
    # 2026-09-24：inf→NaN（与 pool_sim.load_panel 对齐；DIV 无零保护产生的 inf
    # 此前只靠 screen_one 截面内 isfinite 兜底，这里在源头统一）
    cols = list(LOCAL_COLUMN)
    raw[cols] = raw[cols].replace([np.inf, -np.inf], np.nan)
    raw = raw.sort_values(['stock_code', 'date']).reset_index(drop=True)
    return raw


def _ols_slope_window(v: np.ndarray) -> float:
    if not np.isfinite(v).all():
        return np.nan
    t = np.arange(len(v), dtype=float)
    tc = t - t.mean()
    return float(np.dot(tc, v - v.mean()) / np.dot(tc, tc))


def _ols_intercept_window(v: np.ndarray) -> float:
    if not np.isfinite(v).all():
        return np.nan
    t = np.arange(len(v), dtype=float)
    tc = t - t.mean()
    slope = float(np.dot(tc, v - v.mean()) / np.dot(tc, tc))
    return float(v.mean() - slope * t.mean())


def _ols_forcast_window(v: np.ndarray) -> float:
    inter = _ols_intercept_window(v)
    if not np.isfinite(inter):
        return np.nan
    t = np.arange(len(v), dtype=float)
    tc = t - t.mean()
    slope = float(np.dot(tc, v - v.mean()) / np.dot(tc, tc))
    return float(inter + slope * len(v))


def _ts_rank_window(v: np.ndarray) -> float:
    if not np.isfinite(v).all():
        return np.nan
    return float((v <= v[-1]).mean())


def _argmax_window(v: np.ndarray) -> float:
    if not np.isfinite(v).all():
        return np.nan
    return float(np.argmax(v))


def _argmin_window(v: np.ndarray) -> float:
    if not np.isfinite(v).all():
        return np.nan
    return float(np.argmin(v))


def _mad_window(v: np.ndarray) -> float:
    if not np.isfinite(v).all():
        return np.nan
    return float(np.mean(np.abs(v - v.mean())))


def ensure_local_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """惰性计算本地上下文列（2026-09-24 层2；2026-09-25 加 DISP）：幂等。

    - MKT_RET: 全市场等权日收益（同日广播；停牌股 NaN 不计入均值）
    - IND_RET: 个股所属 SW2 行业的等权日收益（含自身；行业成员即 SW2 映射）
    - BETA60 : 个股收益对市场收益的 60 日滚动 beta = Cov(r_i,r_m)/Var(r_m)
      （全窗 min_periods=60，与 SLOPE 同回归语义）
    - DISP   : 行业内分化（2026-09-25）= 同日同行业 RET1 的截面标准差，广播到
      成员股——高分散 = 选股富矿环境（CS dispersion 文献）；单成员行业 → NaN
    直接在传入 frame 上补列（SearchRun 的全量/fast 面板共享同一对象时自动继承）。
    """
    if all(c in frame.columns for c in LOCAL_ONLY_FIELDS):
        return frame
    codes = frame['stock_code']
    ret = frame.groupby(codes, sort=False)['CLOSE'].pct_change(fill_method=None)
    frame['MKT_RET'] = ret.groupby(frame['date'], sort=False).transform('mean')
    ind = codes.map(sw2_mapping()).fillna('__NA__')
    frame['IND_RET'] = ret.groupby([frame['date'], ind], sort=False).transform('mean')
    frame['DISP'] = ret.groupby([frame['date'], ind], sort=False).transform('std')

    def _rmean(s: pd.Series) -> pd.Series:
        return s.groupby(codes, sort=False).rolling(60, min_periods=60) \
            .mean().reset_index(level=0, drop=True)

    m = frame['MKT_RET']
    er, em = _rmean(ret), _rmean(m)
    erm = _rmean(ret * m)
    em2 = _rmean(m * m)
    cov = erm - er * em
    var = em2 - em * em
    frame['BETA60'] = cov / var.where(var > 0)
    return frame


def ensure_extended_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """V4.1可选扩展列（不碰`ensure_local_fields`原4字段，避免历史指纹变动）。

    新增（全部缺失才补，逐列幂等；滚动一律`groupby(stock_code).rolling`前向窗口，
    禁`shift(-n)`/`center=True`，满足`audit_cut_paths A2`）：
    - VWAP: (O+H+L+C)/4 日内均价代理
    - IND_EXCESS: RET1 - IND_RET（个股相对行业残量）
    - BETA20: 20日 Cov(RET1,MKT_RET)/Var(MKT_RET)，min_periods=20
    - IVOL20: 20日特质波动 sqrt(max(Var_r - beta^2*Var_m, 0))
    - AMIHUD20: log1p(20日 mean(|RET1|/AMOUNT*1e9))，min_periods=15
    """
    ensure_local_fields(frame)
    codes = frame['stock_code']
    if 'VWAP' not in frame.columns:
        frame['VWAP'] = (frame['OPEN'] + frame['HIGH'] + frame['LOW'] + frame['CLOSE']) / 4.0
    ret = frame.groupby(codes, sort=False)['CLOSE'].pct_change(fill_method=None)
    if 'IND_EXCESS' not in frame.columns:
        if 'IND_RET' not in frame.columns:
            ensure_local_fields(frame)
        frame['IND_EXCESS'] = ret - frame['IND_RET']

    def _rmean_w(s: pd.Series, win: int) -> pd.Series:
        return s.groupby(codes, sort=False).rolling(win, min_periods=win) \
            .mean().reset_index(level=0, drop=True)

    if 'BETA20' not in frame.columns:
        m = frame['MKT_RET']
        er, em = _rmean_w(ret, 20), _rmean_w(m, 20)
        cov = _rmean_w(ret * m, 20) - er * em
        var_m = _rmean_w(m * m, 20) - em * em
        frame['BETA20'] = cov / var_m.where(var_m > 1e-12)
    if 'IVOL20' not in frame.columns:
        m = frame['MKT_RET']
        beta = frame['BETA20']
        er = _rmean_w(ret, 20)
        em = _rmean_w(m, 20)
        var_r = _rmean_w(ret * ret, 20) - er * er
        var_m = _rmean_w(m * m, 20) - em * em
        resid_var = (var_r - beta * beta * var_m).clip(lower=0.0)
        frame['IVOL20'] = np.sqrt(resid_var)
    if 'AMIHUD20' not in frame.columns:
        ill = (ret.abs() / frame['AMOUNT'].where(frame['AMOUNT'] > 0)) * 1e9
        ill_mean20 = ill.groupby(codes, sort=False).rolling(20, min_periods=15) \
            .mean().reset_index(level=0, drop=True)
        frame['AMIHUD20'] = np.log1p(ill_mean20.clip(lower=0.0))
    return frame


class Evaluator:
    """在固定面板上求值；子表达式按 AST 节点缓存（LRU）。"""

    def __init__(self, frame: pd.DataFrame, cache_size: int = 512):
        self.frame = frame
        self.codes = frame['stock_code']
        self.dates = frame['date']
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._mag: dict = {}

    def _memo(self, node: Node, compute: Callable[[], pd.Series]) -> pd.Series:
        hit = self._cache.get(node)
        if hit is not None:
            self._cache.move_to_end(node)
            return hit
        out = compute()
        if not isinstance(out, pd.Series):
            out = pd.Series(np.asarray(out, dtype=float), index=self.frame.index)
        self._cache[node] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def evaluate(self, node: Node) -> pd.Series:
        return self._memo(node, lambda: self._compute(node))

    def values(self, node: Node) -> np.ndarray:
        return self.evaluate(node).to_numpy(dtype=float)

    def magnitude(self, node: Node) -> float:
        """子表达式典型数量级（|值| 中位数，1/37 抽样，memo）——数值尘埃检查用。"""
        cached = self._mag.get(node)
        if cached is None:
            v = self.evaluate(node)
            s = v.iloc[::37].abs()
            s = s[np.isfinite(s)]
            cached = float(s.median()) if len(s) else 0.0
            self._mag[node] = cached
        return cached

    def _roll(self, s: pd.Series, w: int, mp: int,
              fn: Callable[[pd.core.window.rolling.Rolling], pd.Series]) -> pd.Series:
        return s.groupby(self.codes, sort=False).transform(
            lambda x, fn_=fn: fn_(x.rolling(w, min_periods=mp)))

    def _apply(self, s: pd.Series, w: int, mp: int,
               f: Callable[[np.ndarray], float]) -> pd.Series:
        return s.groupby(self.codes, sort=False).transform(
            lambda x, f_=f: x.rolling(w, min_periods=mp).apply(f_, raw=True))

    def _shift(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(self.codes, sort=False).shift(n)

    def slope_vectorized(self, s: pd.Series, w: int) -> pd.Series:
        """全窗 OLS 斜率的 cumsum 向量化（B1）：与 rolling.apply 等价到浮点重排误差。

        公式（窗口 u=s..i，长度 w）：
          slope = [Σu·y − (Σu)(Σy)/w] / [Σu² − (Σu)²/w]，要求窗口内 w 个值全有限。
        """
        codes = self.codes
        y = s.to_numpy(dtype=float)
        finite = np.isfinite(y)
        y0 = np.where(finite, y, 0.0)
        pos = s.groupby(codes, sort=False).cumcount().to_numpy(dtype=float)
        ycum = pd.Series(y0, index=s.index).groupby(codes, sort=False).cumsum()
        uycum = pd.Series(pos * y0, index=s.index).groupby(codes, sort=False).cumsum()
        u2cum = pd.Series(pos * pos, index=s.index).groupby(codes, sort=False).cumsum()
        ucum = pd.Series(pos, index=s.index).groupby(codes, sort=False).cumsum()
        fcum = pd.Series(finite.astype(float), index=s.index).groupby(
            codes, sort=False).cumsum()
        gb_y, gb_f = ycum.groupby(codes, sort=False), fcum.groupby(codes, sort=False)
        sy = ycum - gb_y.shift(w).fillna(0.0)
        suy = uycum - uycum.groupby(codes, sort=False).shift(w).fillna(0.0)
        su2 = u2cum - u2cum.groupby(codes, sort=False).shift(w).fillna(0.0)
        su = ucum - ucum.groupby(codes, sort=False).shift(w).fillna(0.0)
        cnt = fcum - gb_f.shift(w).fillna(0.0)
        denom = su2 - su * su / w
        num = suy - su * sy / w
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = (num / denom).to_numpy(dtype=float)
        ok = (cnt.to_numpy(dtype=float) == float(w)) & (pos >= w - 1) & (denom.to_numpy(dtype=float) > 0)
        return pd.Series(np.where(ok, slope, np.nan), index=s.index)

    def _compute(self, node: Node) -> pd.Series:
        op = node.op
        if op == 'FIELD':
            if node.name in PLATFORM_FIELDS:
                return self.frame[node.name]
            if node.name in LOCAL_ONLY_FIELDS:
                # 本地上下文字段（层2）：首次访问时惰性计算并缓存到 frame
                if node.name not in self.frame.columns:
                    ensure_local_fields(self.frame)
                return self.frame[node.name]
            if node.name in EXTENDED_ONLY_FIELDS:
                # V4.3 Class2 扩展字段：惰性走 ensure_extended_fields
                #（超集，内调 ensure_local；原 4 字段值不受影响）
                if node.name not in self.frame.columns:
                    ensure_extended_fields(self.frame)
                return self.frame[node.name]
            return self.evaluate(virtual_node(node.name))
        if op == 'CONST':
            return pd.Series(float(node.value), index=self.frame.index)
        if op in ('TS_REGRESS_RESID', 'TS_REGRESS_BETA'):
            # T9（2026-09-25）：滚动 y~x 回归（本地专用算子，禁上平台）。
            # beta = Cov(y,x)/Var(x)；resid = (y−ȳ) − β·(x−x̄)（各用滚动均值）。
            # 全窗 min_periods=w（回归语义）；与 BETA60 同一数学结构、任意 x 泛化。
            w = int(node.window or 0)
            y = self.evaluate(node.args[0])
            x = self.evaluate(node.args[1])

            def _rmean(s: pd.Series) -> pd.Series:
                return s.groupby(self.codes, sort=False) \
                    .rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)

            my, mx = _rmean(y), _rmean(x)
            mxy = _rmean(y * x)
            mxx = _rmean(x * x)
            var = mxx - mx * mx
            beta = (mxy - my * mx) / var.where(var.abs() > 1e-12)
            if op == 'TS_REGRESS_BETA':
                return beta
            alpha = my - beta * mx
            return y - alpha - beta * x
        args = [self.evaluate(a) for a in node.args]
        w, mp = int(node.window or 0), int(node.mp or 0)
        if op == 'NEG':
            return -args[0]
        if op == 'ABS':
            return args[0].abs()
        if op == 'LOG':
            return pd.Series(np.log(np.maximum(args[0].to_numpy(dtype=float), 1e-6)),
                             index=self.frame.index)
        if op == 'LOGABS':
            return pd.Series(np.log(np.maximum(np.abs(args[0].to_numpy(dtype=float)), 1e-6)),
                             index=self.frame.index)
        if op == 'SIGN':
            return pd.Series(np.sign(args[0].to_numpy(dtype=float)), index=self.frame.index)
        if op == 'SIGNEDPOWER':
            v = args[0].to_numpy(dtype=float)
            with np.errstate(invalid='ignore'):
                return pd.Series(np.sign(v) * np.abs(v) ** float(node.value),
                                 index=self.frame.index)
        if op == 'POWER':
            with np.errstate(invalid='ignore'):
                return pd.Series(args[0].to_numpy(dtype=float) ** float(node.value),
                                 index=self.frame.index)
        if op == 'AS_FLOAT':
            return args[0]
        if op == 'ADD':
            return args[0] + args[1]
        if op == 'SUB':
            return args[0] - args[1]
        if op == 'MUL':
            return args[0] * args[1]
        if op == 'DIV':
            with np.errstate(divide='ignore', invalid='ignore'):
                return args[0] / args[1]
        if op in ('GT', 'GE', 'LT', 'LE'):
            a, b = args[0].to_numpy(dtype=float), args[1].to_numpy(dtype=float)
            fn = {'GT': np.greater, 'GE': np.greater_equal,
                  'LT': np.less, 'LE': np.less_equal}[op]
            return pd.Series(np.where(fn(a, b), 1.0, 0.0), index=self.frame.index)
        if op == 'AND':
            return pd.Series(np.minimum(args[0].to_numpy(float), args[1].to_numpy(float)),
                             index=self.frame.index)
        if op == 'OR':
            return pd.Series(np.maximum(args[0].to_numpy(float), args[1].to_numpy(float)),
                             index=self.frame.index)
        if op == 'MAX':
            return pd.Series(np.maximum(args[0].to_numpy(float), args[1].to_numpy(float)),
                             index=self.frame.index)
        if op == 'MIN':
            return pd.Series(np.minimum(args[0].to_numpy(float), args[1].to_numpy(float)),
                             index=self.frame.index)
        if op == 'MEAN':
            return (args[0] + args[1]) / 2.0
        if op == 'IF':
            cond = args[0].to_numpy(dtype=float) > 0.5
            return pd.Series(np.where(cond, args[1].to_numpy(dtype=float),
                                      args[2].to_numpy(dtype=float)), index=self.frame.index)
        if op in ('RANK', 'SCALE', 'ZSCORE'):
            s = args[0]
            g = s.groupby(self.dates, sort=False)
            if op == 'RANK':
                return g.rank(pct=True)
            if op == 'ZSCORE':
                return g.transform(lambda x: (x - x.mean()) / x.std())
            return g.transform(lambda x: 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0)
        if op in ('REF', 'DELAY'):
            return self._shift(args[0], w)
        if op in ('DIFF', 'DELTA'):
            return args[0] - self._shift(args[0], w)
        if op in ('RETURNS', 'ROC', 'PCT_CHANGE'):
            return args[0] / self._shift(args[0], w) - 1.0
        if op in ('TS_MEAN', 'MA'):
            return self._roll(args[0], w, mp, lambda r: r.mean())
        if op in ('STD', 'STDDEV'):
            return self._roll(args[0], w, mp, lambda r: r.std())
        if op == 'VAR':
            return self._roll(args[0], w, mp, lambda r: r.var())
        if op == 'SUM':
            return self._roll(args[0], w, mp, lambda r: r.sum())
        if op in ('TS_MAX', 'HHV'):
            return self._roll(args[0], w, mp, lambda r: r.max())
        if op in ('TS_MIN', 'LLV'):
            return self._roll(args[0], w, mp, lambda r: r.min())
        if op == 'TS_MEDIAN':
            return self._roll(args[0], w, mp, lambda r: r.median())
        if op == 'TS_SKEW':
            return self._roll(args[0], w, mp, lambda r: r.skew())
        if op == 'TS_KURT':
            return self._roll(args[0], w, mp, lambda r: r.kurt())
        if op in ('TS_MAD', 'AVEDEV'):
            return self._apply(args[0], w, mp, _mad_window)
        if op in ('TS_MIDDLE',):
            return (self._roll(args[0], w, mp, lambda r: r.max())
                    + self._roll(args[0], w, mp, lambda r: r.min())) / 2.0
        if op == 'TS_RANK':
            return self._apply(args[0], w, mp, _ts_rank_window)
        if op == 'TS_ARGMAX':
            return self._apply(args[0], w, mp, _argmax_window)
        if op == 'TS_ARGMIN':
            return self._apply(args[0], w, mp, _argmin_window)
        if op == 'TS_ZSCORE':
            mean = self._roll(args[0], w, mp, lambda r: r.mean())
            std = self._roll(args[0], w, mp, lambda r: r.std())
            return (args[0] - mean) / std
        if op == 'EMA':
            return args[0].groupby(self.codes, sort=False).transform(
                lambda x: x.ewm(span=w, adjust=False, min_periods=mp).mean())
        if op == 'WMA':
            weights = np.arange(1, w + 1, dtype=float)
            weights /= weights.sum()

            def _wma(v, weights_=weights):
                if not np.isfinite(v).all():
                    return np.nan
                return float(np.dot(v, weights_))

            return self._apply(args[0], w, mp, _wma)
        if op == 'DECAYLINEAR':
            weights = 0.9 ** np.arange(w - 1, -1, -1, dtype=float)
            weights /= weights.sum()

            def _decay(v, weights_=weights):
                if not np.isfinite(v).all():
                    return np.nan
                return float(np.dot(v, weights_))

            return self._apply(args[0], w, mp, _decay)
        if op == 'SLOPE':
            if w >= 2 and mp == w:
                return self.slope_vectorized(args[0], w)
            return self._apply(args[0], w, mp, _ols_slope_window)
        if op == 'INTERCEPT':
            return self._apply(args[0], w, mp, _ols_intercept_window)
        if op == 'FORCAST':
            return self._apply(args[0], w, mp, _ols_forcast_window)
        if op == 'ANGLE':
            slope = self._apply(args[0], w, mp, _ols_slope_window)
            return pd.Series(np.degrees(np.arctan(slope.to_numpy(dtype=float))),
                             index=self.frame.index)
        if op == 'SHARPE':
            mean = self._roll(args[0], w, mp, lambda r: r.mean())
            std = self._roll(args[0], w, mp, lambda r: r.std())
            return mean / std
        if op == 'COUNT':
            truth = (args[0].to_numpy(dtype=float) > 0).astype(float)
            return self._roll(pd.Series(truth, index=self.frame.index), w, mp,
                              lambda r: r.sum())
        if op == 'EVERY':
            truth = (args[0].to_numpy(dtype=float) > 0).astype(float)
            return self._roll(pd.Series(truth, index=self.frame.index), w, mp,
                              lambda r: r.min())
        if op == 'EXIST':
            truth = (args[0].to_numpy(dtype=float) > 0).astype(float)
            return self._roll(pd.Series(truth, index=self.frame.index), w, mp,
                              lambda r: r.max())
        if op == 'PRODUCT':
            return self._roll(args[0], w, mp, lambda r: r.apply(np.prod, raw=True))
        if op in ('CORR', 'COV'):
            mean_a = self._roll(args[0], w, mp, lambda r: r.mean())
            mean_b = self._roll(args[1], w, mp, lambda r: r.mean())
            mean_ab = self._roll(args[0] * args[1], w, mp, lambda r: r.mean())
            cov = mean_ab - mean_a * mean_b
            if op == 'COV':
                return cov
            var_a = self._roll(args[0] * args[0], w, mp, lambda r: r.mean()) - mean_a * mean_a
            var_b = self._roll(args[1] * args[1], w, mp, lambda r: r.mean()) - mean_b * mean_b
            denom = np.sqrt(np.where((var_a > 0) & (var_b > 0),
                                     var_a.to_numpy(dtype=float) * var_b.to_numpy(dtype=float),
                                     np.nan))
            return pd.Series(cov.to_numpy(dtype=float) / denom, index=self.frame.index)
        if op == 'TS_REGRESSION':
            raise NotImplementedError(op)
        return self._compute_tech(node, args)

    def _ema(self, s: pd.Series, n: int, mp: int = 1) -> pd.Series:
        return s.groupby(self.codes, sort=False).transform(
            lambda x: x.ewm(span=n, adjust=False, min_periods=mp).mean())

    def _sma_cn(self, s: pd.Series, n: int, m: int) -> pd.Series:
        """中国式SMA：X*M/N + prev*(1-M/N)（通达信语义）。"""
        alpha = float(m) / float(n)
        return s.groupby(self.codes, sort=False).transform(
            lambda x: x.ewm(alpha=alpha, adjust=False, min_periods=1).mean())

    def _compute_tech(self, node: Node, args: list) -> pd.Series:
        from field_catalog import PARAM_OPS
        op = node.op
        w = int(node.window or 0)
        mp = int(node.mp or 0)
        params = tuple(node.params) if node.params else \
            tuple(PARAM_OPS.get(op, {}).get('params', ()))
        s = args[0] if args else None
        if op == 'RSI':
            delta = s - self._shift(s, 1)
            up = delta.clip(lower=0.0)
            down = (-delta).clip(lower=0.0)
            au = self._sma_cn(up, w, 1)
            ad = self._sma_cn(down, w, 1)
            return 100.0 * au / (au + ad).replace(0, np.nan)
        if op == 'WR':
            hh = self._roll(s, w, mp, lambda r: r.max())
            ll = self._roll(s, w, mp, lambda r: r.min())
            return (hh - s) / (hh - ll).replace(0, np.nan) * 100.0
        if op == 'PSY':
            up = (s > self._shift(s, 1)).astype(float)
            return self._roll(up, w, mp, lambda r: r.mean()) * 100.0
        if op == 'BIAS':
            ma = self._roll(s, w, mp, lambda r: r.mean())
            return (s - ma) / ma.replace(0, np.nan) * 100.0
        if op == 'CCI':
            ma = self._roll(s, w, mp, lambda r: r.mean())
            dev = (s - ma).abs()
            mad = self._roll(dev, w, mp, lambda r: r.mean())
            return (s - ma) / (0.015 * mad).replace(0, np.nan)
        if op == 'BOLL_MID':
            return self._roll(s, w, mp, lambda r: r.mean())
        if op in ('BOLL_UPPER', 'BOLL_LOWER', 'BOLL_WIDTH'):
            p = float(params[0]) if params else 2.0
            ma = self._roll(s, w, mp, lambda r: r.mean())
            sd = self._roll(s, w, mp, lambda r: r.std())
            if op == 'BOLL_UPPER':
                return ma + p * sd
            if op == 'BOLL_LOWER':
                return ma - p * sd
            return 2.0 * p * sd
        if op == 'DEMA':
            e1 = self._ema(s, w, mp)
            e2 = self._ema(e1, w, mp)
            return 2.0 * e1 - e2
        if op == 'TEMA':
            e1 = self._ema(s, w, mp)
            e2 = self._ema(e1, w, mp)
            e3 = self._ema(e2, w, mp)
            return 3.0 * e1 - 3.0 * e2 + e3
        if op == 'ADV':
            return self._roll(s, w, mp, lambda r: r.mean())
        if op in ('MACD_DIF', 'MACD_DEA', 'MACD'):
            short, long_, m = (int(params[0]), int(params[1]), int(params[2])) \
                if len(params) >= 3 else (12, 26, 9)
            dif = self._ema(s, short) - self._ema(s, long_)
            dea = self._ema(dif, m)
            if op == 'MACD_DIF':
                return dif
            if op == 'MACD_DEA':
                return dea
            return (dif - dea) * 2.0
        if op in ('KDJ_K', 'KDJ_D', 'KDJ_J'):
            close, high, low = args[0], args[1], args[2]
            n, m1, m2 = (int(params[0]), int(params[1]), int(params[2])) \
                if len(params) >= 3 else (9, 3, 3)
            ll = self._roll(low, n, n, lambda r: r.min())
            hh = self._roll(high, n, n, lambda r: r.max())
            rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100.0
            k = self._sma_cn(rsv, m1, 1)
            if op == 'KDJ_K':
                return k
            d = self._sma_cn(k, m2, 1)
            if op == 'KDJ_D':
                return d
            return 3.0 * k - 2.0 * d
        if op in ('MASS', 'MASSMA'):
            high, low = args[0], args[1]
            n1, n2, m = (int(params[0]), int(params[1]), int(params[2])) \
                if len(params) >= 3 else (9, 25, 6)
            rng_ = high - low
            e1 = self._ema(rng_, n1, n1)
            e2 = self._ema(e1, n1, n1)
            ratio = e1 / e2.replace(0, np.nan)
            mass = self._roll(ratio, n2, n2, lambda r: r.sum())
            return self._roll(mass, m, m, lambda r: r.mean())
        if op == 'VR':
            close, vol = args[0], args[1]
            n = int(params[0]) if params else 26
            delta = close - self._shift(close, 1)
            up = vol.where(delta > 0, 0.0)
            flat = vol.where(delta == 0, 0.0)
            down = vol.where(delta < 0, 0.0)
            us = self._roll(up, n, n, lambda r: r.sum())
            fs = self._roll(flat, n, n, lambda r: r.sum())
            ds = self._roll(down, n, n, lambda r: r.sum())
            return (us + 0.5 * fs) / (ds + 0.5 * fs).replace(0, np.nan) * 100.0
        if op == 'OBV':
            close, vol = args[0], args[1]
            delta = close - self._shift(close, 1)
            signed = vol.where(delta > 0, 0.0) - vol.where(delta < 0, 0.0)
            return signed.groupby(self.codes, sort=False).cumsum()
        raise NotImplementedError(f'op {op} not implemented locally')


def main() -> None:
    import argparse
    import time

    from grammar import render
    from grammar import ts as gts

    ap = argparse.ArgumentParser(description='求值器冒烟：对指定表达式求值并打印覆盖/耗时')
    ap.add_argument('expr', nargs='?', default='SLOPE(LOG(CLOSE),60)')
    args = ap.parse_args()
    t0 = time.time()
    frame = load_panel()
    ev = Evaluator(frame)
    node = gts('SLOPE', Node('LOG', (Node('FIELD', name='CLOSE'),)), 60)
    vals = ev.values(node)
    print(f'render={render(node)} rows={len(frame):,} finite={np.isfinite(vals).mean():.3f} '
          f'{time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
