"""平台字段/算子目录与本地映射（Phase A 合并引擎）。

- 字段：平台公式模式 8 个基础字段 ↔ 本地 stock_features 列，逐字一致（AMOUNT 单位=元）。
- 虚拟序列：RET1/LOGC，展开后仍是平台合法公式（提交历史有同形公式）。
- AMOUNT 做分母的公式必须在表达式内显式 ×1000000000：平台对 ~1e-11 数值处理异常
  （2026-09-16 尺度探针实证：换手 30.68%→4.93%、净 −0.50%→+13.86%）。
- 算子分级：
  VERIFIED   已出现在已提交候选公式（有平台 run 记录）；
  DOCUMENTED 官方手册有但未平台提交（默认不进第一轮种群，可用 --allow-documented 启用）。
"""
from __future__ import annotations

from pathlib import Path

PLATFORM_FIELDS = ('OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOLUME', 'AMOUNT', 'TURNOVER', 'MARKET_CAP')

# 本地专用字段（2026-09-24 层2）：市场/行业上下文列，本地可算、平台无对应字段。
# 含这些字段的公式**禁止导出/上传**（export/upload 链路硬闸门），仅用于 vwap 自用线；
# grammar.validate 仅在 allow_documented=True 时放行（与 DOCUMENTED 算子同一通道）。
LOCAL_ONLY_FIELDS = ('MKT_RET', 'IND_RET', 'BETA60', 'DISP')

# 本地专用算子（T9，2026-09-25）：滚动 y~x 回归——平台无对应算子，含这些算子的公式
# 禁止上传（formula_uses_local 同步拦截算子名）；随 allow_documented 通道放行 validate。
LOCAL_ONLY_OPS = ('TS_REGRESS_RESID', 'TS_REGRESS_BETA')

LOCAL_COLUMN = {
    'OPEN': 'open', 'CLOSE': 'close', 'HIGH': 'high', 'LOW': 'low',
    'VOLUME': 'volume', 'AMOUNT': 'amount', 'TURNOVER': 'turnover_rate',
    'MARKET_CAP': 'total_mv',
}

LOCAL_SCALE = {}  # AMOUNT 不做本地缩放：本地=平台=元；
# AMOUNT 做分母的公式必须在表达式内显式 ×1e9（平台对 ~1e-11 数值处理异常，2026-09-16 探针实证）

VIRTUAL_NAMES = ('RET1', 'LOGC', 'GAP1', 'BODY')

WINDOWS = (3, 5, 6, 10, 12, 14, 20, 24, 26, 40, 60, 120, 250)
SHIFT_WINDOWS = (1, 2) + WINDOWS
MAX_DEPTH = 16  # comp6 类复合实测深度 ~15；复杂度由节点数因子惩罚，此处只做上限保护
CONST_WHITELIST = (0.0, 0.3, 0.5, 0.8, 1.0, 2.0, 1000000000.0)
# 2026-09-24：0.3/0.8 供条件门分位阈值变异（RANK(X)>0.5 一档太窄，门限微调需候选值）
EXPONENTS = (0.5, 2.0)

# 官方技术指标（公式模式手册确认可用；待平台冒烟后升 VERIFIED）
TECH_TS_OPS = {'RSI', 'WR', 'PSY', 'BIAS', 'CCI', 'BOLL_WIDTH', 'DEMA', 'TEMA', 'ADV'}
# 多参技术指标：(表达式个数, 默认数值参数, 是否带窗口参数)
PARAM_OPS = {
    'BOLL_MID': {'expr': 1, 'params': (2.0,), 'window': True},
    'BOLL_UPPER': {'expr': 1, 'params': (2.0,), 'window': True},
    'BOLL_LOWER': {'expr': 1, 'params': (2.0,), 'window': True},
    'MACD_DIF': {'expr': 1, 'params': (12.0, 26.0, 9.0), 'window': False},
    'MACD_DEA': {'expr': 1, 'params': (12.0, 26.0, 9.0), 'window': False},
    'MACD': {'expr': 1, 'params': (12.0, 26.0, 9.0), 'window': False},
    'KDJ_K': {'expr': 3, 'params': (9.0, 3.0, 3.0), 'window': False},
    'KDJ_D': {'expr': 3, 'params': (9.0, 3.0, 3.0), 'window': False},
    'KDJ_J': {'expr': 3, 'params': (9.0, 3.0, 3.0), 'window': False},
    'MASS': {'expr': 2, 'params': (9.0, 25.0, 6.0), 'window': False},
    'MASSMA': {'expr': 2, 'params': (9.0, 25.0, 6.0), 'window': False},
    'VR': {'expr': 2, 'params': (26.0,), 'window': False},
    'OBV': {'expr': 2, 'params': (), 'window': False},
}

TS_OPS = {
    'REF', 'DELAY', 'DIFF', 'DELTA', 'TS_MEAN', 'MA', 'SUM', 'PRODUCT',
    'STD', 'STDDEV', 'VAR', 'TS_MAX', 'TS_MIN', 'TS_MIDDLE', 'TS_MAD',
    'TS_RANK', 'TS_ARGMAX', 'TS_ARGMIN', 'TS_ZSCORE', 'TS_SKEW', 'TS_KURT',
    'TS_MEDIAN', 'AVEDEV', 'EMA', 'WMA', 'DECAYLINEAR', 'SLOPE', 'ANGLE',
    'INTERCEPT', 'FORCAST', 'RETURNS', 'ROC', 'PCT_CHANGE', 'COUNT',
    'EVERY', 'EXIST', 'HHV', 'LLV',
} | TECH_TS_OPS | set(LOCAL_ONLY_OPS)
CS_OPS = {'RANK', 'SCALE', 'ZSCORE'}

MP_FULL = {
    'PRODUCT', 'TS_MIDDLE', 'TS_MAD', 'TS_RANK', 'TS_ARGMAX', 'TS_ARGMIN',
    'TS_SKEW', 'TS_KURT', 'TS_MEDIAN', 'AVEDEV', 'WMA', 'DECAYLINEAR',
    'SLOPE', 'ANGLE', 'INTERCEPT', 'FORCAST', 'CORR', 'COV', 'COUNT',
    'EVERY', 'EXIST',
} | TECH_TS_OPS
MP_075 = {
    'TS_MEAN', 'MA', 'SUM', 'STD', 'STDDEV', 'VAR', 'TS_MAX', 'TS_MIN',
    'HHV', 'LLV', 'TS_ZSCORE',
}
MP_NONE = {'REF', 'DELAY', 'DIFF', 'DELTA', 'RETURNS', 'ROC', 'PCT_CHANGE'}
MP_EMA = {'EMA'}

ARITY = {
    'FIELD': 0, 'CONST': 0, 'NEG': 1, 'ABS': 1, 'LOG': 1, 'LOGABS': 1,
    'SIGN': 1, 'AS_FLOAT': 1, 'SIGNEDPOWER': 1, 'POWER': 1,
    'RANK': 1, 'SCALE': 1, 'ZSCORE': 1,
    'ADD': 2, 'SUB': 2, 'MUL': 2, 'DIV': 2, 'GT': 2, 'GE': 2, 'LT': 2,
    'LE': 2, 'AND': 2, 'OR': 2, 'MAX': 2, 'MIN': 2, 'MEAN': 2,
    'IF': 3, 'CORR': 2, 'COV': 2,
}
for _op in TS_OPS:
    ARITY[_op] = 1
for _op in LOCAL_ONLY_OPS:  # 滚回归 y~x：双表达式 + 窗口
    ARITY[_op] = 2
for _op, _spec in PARAM_OPS.items():
    ARITY[_op] = int(_spec['expr'])

VERIFIED_OPS = {
    'TS_MEAN', 'DELAY', 'LOG', 'MAX', 'TS_MIN', 'RANK', 'TS_MAX', 'STDDEV',
    'SLOPE', 'ABS', 'IF', 'CORR', 'TS_SKEW', 'COUNT', 'EMA', 'MIN',
    'TS_ZSCORE',
} | {'ADD', 'SUB', 'MUL', 'DIV', 'GT', 'GE', 'LT', 'LE', 'NEG'}

DOCUMENTED_OPS = {
    'LOGABS', 'SIGN', 'AS_FLOAT', 'SIGNEDPOWER', 'POWER', 'SCALE', 'AND', 'OR',
    'MEAN', 'REF', 'DIFF', 'DELTA', 'MA', 'SUM', 'PRODUCT', 'STD', 'VAR',
    'TS_MIDDLE', 'TS_MAD', 'TS_RANK', 'TS_ARGMAX', 'TS_ARGMIN', 'TS_KURT',
    'TS_MEDIAN', 'AVEDEV', 'WMA', 'DECAYLINEAR', 'ANGLE', 'INTERCEPT',
    'FORCAST', 'RETURNS', 'ROC', 'PCT_CHANGE', 'EVERY', 'EXIST', 'HHV', 'LLV',
    'CORRELATION', 'COVARIANCE',
} | TECH_TS_OPS | set(PARAM_OPS)

IMPLEMENTED_OPS = set(ARITY)


def default_mp(op: str, window: int) -> int | None:
    """本地 min_periods 约定：全窗算子=w；均值族=0.75w；EMA=span；移位族=None；
    滚回归（T9）=w（回归语义要求全窗）。"""
    if op in LOCAL_ONLY_OPS:
        return window
    if op in MP_NONE:
        return None
    if op in MP_FULL:
        return window
    if op in MP_EMA:
        return window
    if op in MP_075:
        return max(2, int(round(window * 0.75)))
    return None


def local_scale(field: str) -> float:
    return LOCAL_SCALE.get(field, 1.0)


_SW2_CACHE: dict | None = None


def sw2_mapping() -> dict[str, str]:
    """SW2 股票→行业映射（config/sw2_industry_encoding.pkl），进程内缓存。"""
    global _SW2_CACHE
    if _SW2_CACHE is None:
        import pickle
        p = (Path(__file__).resolve().parent.parent.parent
             / 'config' / 'sw2_industry_encoding.pkl')
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        mp = obj.get('mappings', {}).get('stock_to_industry', {})
        if not mp:
            raise RuntimeError(f'SW2 映射为空: {p}')
        _SW2_CACHE = {str(k): str(v) for k, v in mp.items()}
    return _SW2_CACHE


def formula_uses_local(text: str) -> bool:
    """公式是否引用本地专用字段或本地专用算子（命中即禁止上平台）。"""
    import re
    t = str(text)
    if any(re.search(rf'\b{f}\b', t) for f in LOCAL_ONLY_FIELDS):
        return True
    return any(re.search(rf'\b{op}\s*\(', t) for op in LOCAL_ONLY_OPS)
