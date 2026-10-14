"""AST → 中文机制解释（Thought Decompiler 的"反解释"能力，本地确定性实现）。

用于台账、短名单报告、轴命名与人工审阅；不依赖 LLM。
"""
from __future__ import annotations

import _paths  # noqa: F401
from grammar import Node

_FIELD = {
    'OPEN': '开盘价', 'CLOSE': '收盘价', 'HIGH': '最高价', 'LOW': '最低价',
    'VOLUME': '成交量', 'AMOUNT': '成交额', 'TURNOVER': '换手率',
    'MARKET_CAP': '总市值', 'RET1': '日收益(收盘/前收-1)', 'LOGC': '对数收盘价',
}

_STAT = {
    'TS_MEAN': '均值', 'MA': '均值', 'STDDEV': '标准差', 'STD': '标准差',
    'VAR': '方差', 'TS_MAX': '最大值', 'HHV': '最大值', 'TS_MIN': '最小值',
    'LLV': '最小值', 'TS_MEDIAN': '中位数', 'TS_SKEW': '偏度', 'TS_KURT': '峰度',
    'TS_MAD': '平均绝对偏差', 'AVEDEV': '平均绝对偏差', 'TS_MIDDLE': '(最大+最小)/2',
    'TS_RANK': '时序分位', 'TS_ZSCORE': '时序z值', 'SUM': '求和', 'PRODUCT': '连乘',
    'SLOPE': '线性回归斜率', 'ANGLE': '回归角度', 'INTERCEPT': '回归截距',
    'FORCAST': '回归预测值', 'WMA': '线性加权均值', 'DECAYLINEAR': '衰减加权均值',
    'EMA': '指数均值', 'COUNT': '为真的次数', 'EVERY': '是否全真', 'EXIST': '是否出现过',
    'TS_ARGMAX': '最大值位置', 'TS_ARGMIN': '最小值位置',
}

_CMP = {'GT': '大于', 'GE': '不小于', 'LT': '小于', 'LE': '不大于'}


def _txt(node: Node) -> str:
    op = node.op
    if op == 'FIELD':
        return _FIELD.get(node.name, str(node.name))
    if op == 'CONST':
        return format(float(node.value), 'g')
    args = [_txt(a) for a in node.args]
    if op == 'NEG':
        return f"负的({args[0]})"
    if op == 'ABS':
        return f"{args[0]}的绝对值"
    if op == 'LOG':
        return f"{args[0]}的对数"
    if op == 'LOGABS':
        return f"{args[0]}绝对值的对数"
    if op == 'SIGN':
        return f"{args[0]}的符号"
    if op in ('SIGNEDPOWER', 'POWER'):
        return f"{args[0]}的{format(float(node.value), 'g')}次幂"
    if op == 'ADD':
        return f"{args[0]}加{args[1]}"
    if op == 'SUB':
        return f"{args[0]}减{args[1]}"
    if op == 'MUL':
        return f"{args[0]}乘{args[1]}"
    if op == 'DIV':
        return f"{args[0]}除以{args[1]}"
    if op in _CMP:
        return f"{args[0]}{_CMP[op]}{args[1]}"
    if op == 'AND':
        return f"{args[0]}且{args[1]}"
    if op == 'OR':
        return f"{args[0]}或{args[1]}"
    if op == 'IF':
        return f"当{args[0]}时取{args[1]}，否则取{args[2]}"
    if op in ('MAX', 'MIN', 'MEAN'):
        word = {'MAX': '较大值', 'MIN': '较小值', 'MEAN': '均值'}[op]
        return f"{args[0]}与{args[1]}的{word}"
    if op in ('RANK', 'ZSCORE', 'SCALE'):
        word = {'RANK': '截面排名分位', 'ZSCORE': '截面z值', 'SCALE': '截面minmax缩放'}[op]
        return f"{args[0]}的{word}"
    if op in ('REF', 'DELAY'):
        return f"{args[0]}在{int(node.window)}期前的值"
    if op in ('DIFF', 'DELTA'):
        return f"{args[0]}相对{int(node.window)}期前的变化量"
    if op in ('RETURNS', 'ROC', 'PCT_CHANGE'):
        return f"{args[0]}相对{int(node.window)}期前的涨跌幅"
    if op in ('CORR',):
        return f"{args[0]}与{args[1]}过去{int(node.window)}期的相关系数"
    if op == 'COV':
        return f"{args[0]}与{args[1]}过去{int(node.window)}期的协方差"
    if op in ('COUNT', 'EVERY', 'EXIST'):
        return f"过去{int(node.window)}期{args[0]}{_STAT[op]}"
    if op in _STAT:
        return f"过去{int(node.window)}期{args[0]}的{_STAT[op]}"
    if op == 'AS_FLOAT':
        return args[0]
    return f"{op}({', '.join(args)})"


def explain(node: Node, direction: int | None = None, max_len: int = 300) -> str:
    text = _txt(node)
    if direction is not None:
        side = '高' if int(direction) == 1 else '低'
        text = f'{text}；方向 {int(direction)}：做多因子值{side}的一端'
    if len(text) > max_len:
        text = text[:max_len - 1] + '…'
    return text


if __name__ == '__main__':
    import sys

    from fixtures import FIXTURES
    from grammar import render

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    for name, builder in sorted(FIXTURES.items()):
        node = builder()
        print(f'{name:14s} {render(node)}')
        print(f'{"":14s} -> {explain(node)}')
