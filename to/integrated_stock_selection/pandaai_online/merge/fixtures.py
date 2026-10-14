"""Phase A 回归夹具：与历史脚本逐字对齐的表达式 + 期望来源。

对齐脚本：
- qs_factory.py         : amihud120 / skew120
- new_slow_factors.py   : slope60 / highpos120 / amihud60 / vol_pct60 / efficiency60
- qs_r3.py              : vr120_s20 / zturn20

说明：夹具允许白名单外常数（1e-9/60/1e9），仅用于回放校验，不进入搜索种群。
"""
from __future__ import annotations

import _paths  # noqa: F401
from grammar import Node, const, field, ts, virtual_node


def _ret1() -> Node:
    return virtual_node('RET1')


def _min1(x: Node) -> Node:
    return Node('SUB', (x, const(1.0)))


def _scaled_illiquidity(ret_abs: Node) -> Node:
    """平台安全形式的非流动性项：ABS(ret)/AMOUNT*1e9（平台对 ~1e-11 数值处理异常）。"""
    return Node('MUL', (Node('DIV', (ret_abs, field('AMOUNT'))), const(1000000000.0)))


def amihud120() -> Node:
    ill = _scaled_illiquidity(Node('ABS', (_ret1(),)))
    return Node('LOG', (Node('ADD', (const(1.0), ts('TS_MEAN', ill, 120, mp=120))),))


def skew120() -> Node:
    return ts('TS_SKEW', _ret1(), 120, mp=120)


def slope60() -> Node:
    return ts('SLOPE', virtual_node('LOGC'), 60, mp=60)


def highpos120() -> Node:
    return _min1(Node('DIV', (field('CLOSE'), ts('TS_MAX', field('HIGH'), 120, mp=90))))


def amihud60() -> Node:
    ill = _scaled_illiquidity(Node('ABS', (_ret1(),)))
    mean_ill = Node('DIV', (ts('SUM', ill, 60, mp=40), const(60)))
    return Node('LOG', (Node('ADD', (const(1.0), mean_ill)),))


def vol_pct60() -> Node:
    std20 = ts('STDDEV', _ret1(), 20, mp=20)
    return ts('TS_RANK', std20, 60, mp=60)


def efficiency60() -> Node:
    num = Node('ABS', (_min1(Node('DIV', (field('CLOSE'), ts('DELAY', field('CLOSE'), 60, mp=None)))),))
    den = ts('SUM', Node('ABS', (_ret1(),)), 60, mp=60)
    return Node('DIV', (num, Node('MAX', (den, const(1e-9)))))


def vr120_s20() -> Node:
    vr = _min1(Node('DIV', (field('VOLUME'), ts('TS_MEAN', field('VOLUME'), 120, mp=90))))
    return ts('EMA', vr, 20, mp=20)


def zturn20() -> Node:
    return ts('TS_MEAN', field('TURNOVER'), 20, mp=15)


FIXTURES = {
    'amihud120': amihud120,
    'skew120': skew120,
    'slope60': slope60,
    'highpos120': highpos120,
    'amihud60': amihud60,
    'vol_pct60': vol_pct60,
    'efficiency60': efficiency60,
    'vr120_s20': vr120_s20,
    'zturn20': zturn20,
}

PANEL_SOURCE = {
    'amihud120': ('qs_factory_panel.parquet', 'qs_factory_c2c3.json'),
    'skew120': ('qs_factory_panel.parquet', 'qs_factory_c2c3.json'),
    'slope60': ('new_slow_panel.parquet', 'new_slow_c2c3.json'),
    'highpos120': ('new_slow_panel.parquet', 'new_slow_c2c3.json'),
    'amihud60': ('new_slow_panel.parquet', 'new_slow_c2c3.json'),
    'vol_pct60': ('new_slow_panel.parquet', 'new_slow_c2c3.json'),
    'efficiency60': ('new_slow_panel.parquet', 'new_slow_c2c3.json'),
    'vr120_s20': ('qs_r3_panel.parquet', 'qs_r3_c2c3.json'),
    'zturn20': ('qs_r3_panel.parquet', 'qs_r3_c2c3.json'),
}
