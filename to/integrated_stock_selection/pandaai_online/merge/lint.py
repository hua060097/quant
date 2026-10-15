"""语义规则库：单位一致性/数学域/稀疏并列/常见陷阱（含未验证算子提示）。

对应 Alpha-GPT 3.2.1 的 "rule base"（数学规则、单位一致性、金融域规则），
以及我们自己的事故教训（稀疏并列、MEAN 陷阱、20日z毒药等）。

接口：lint(node) -> list[{'level': 'error'|'warning'|'info', 'code': str, 'message': str}]
"""
from __future__ import annotations

import _paths  # noqa: F401
from field_catalog import VERIFIED_OPS, LOCAL_ONLY_OPS
from grammar import Node, walk, TS_OPS

_DIM = {
    'OPEN': 'price', 'CLOSE': 'price', 'HIGH': 'price', 'LOW': 'price',
    'VOLUME': 'volume', 'AMOUNT': 'price*volume', 'TURNOVER': 'ratio',
    'MARKET_CAP': 'mcap',
    # 本地专用字段（层2）：市场/行业收益为比率，beta 无量纲
    'MKT_RET': 'ratio', 'IND_RET': 'ratio', 'BETA60': 'scalar',
}
_VIRTUAL_DIM = {'RET1': 'ratio', 'LOGC': 'logprice'}
_SCALAR = 'scalar'


def _norm(dim: str) -> str:
    if dim in ('', _SCALAR):
        return _SCALAR
    parts = dim.split('*')
    return '*'.join(sorted(p for p in parts if p))


def _mul(a: str, b: str) -> str:
    if a == _SCALAR:
        return b
    if b == _SCALAR:
        return a
    return _norm(a + '*' + b)


def _div(a: str, b: str) -> str:
    if b == _SCALAR:
        return a
    num = [p for p in a.split('*') if p] if a != _SCALAR else []
    den = [p for p in b.split('*') if p] if b != _SCALAR else []
    for d in den:
        if d in num:
            num.remove(d)
        else:
            num.append(f'1/{d}')
    return _norm('*'.join(sorted(num))) if num else _SCALAR


def _infer(node: Node, issues: list) -> str:
    op = node.op
    if op == 'FIELD':
        return _DIM.get(node.name, _VIRTUAL_DIM.get(node.name, _SCALAR))
    if op == 'CONST':
        return _SCALAR
    args = [_infer(a, issues) for a in node.args]
    if op in ('NEG', 'ABS'):
        return args[0]
    if op in ('LOG', 'LOGABS'):
        if args[0] in ('ratio', 'scalar', 'logprice') or args[0].startswith('1/'):
            issues.append({'level': 'warning', 'code': 'log-domain',
                           'message': f'{op} 的输入可能非正（比率/差分序列），平台侧可能产生 NaN'})
        return 'log'
    if op in ('SIGN', 'AS_FLOAT'):
        return _SCALAR
    if op in ('SIGNEDPOWER', 'POWER'):
        return 'ratio'
    if op in ('ADD', 'SUB'):
        a, b = args
        if a != b and _SCALAR not in (a, b):
            issues.append({'level': 'error', 'code': 'unit-mismatch',
                           'message': f'{op} 两侧单位不一致：{a} vs {b}（如 成交量+价格 这类错误）'})
        # 尺度混算（2026-09-25）：量纲相同/豁免但尺度差数量级的加减——
        # (CLOSE/DELAY)−AMOUNT ≈ −AMOUNT，小项沦为数值尘埃，字面含义失效
        _sa, _sb = _scale_infer(node.args[0]), _scale_infer(node.args[1])
        if (_sa is not None and _sb is not None and _sa != _sb
                and 'MIXED' not in (_sa, _sb)):
            issues.append({'level': 'error', 'code': 'scale-mix',
                           'message': f'{op} 两侧尺度混算（{_sa} vs {_sb}）：大数吞掉小项，'
                                      f'表达式退化为大数一侧——请改成比值/排名形式'})
        # 截面常数（2026-09-25）：broadcast 序列（MKT_RET 类）相加/相减 → 仍为
        # 全截面同值（RANK 后恒 0.5），无截面区分度
        _csa, _csb = _cs_infer(node.args[0]), _cs_infer(node.args[1])
        if _csa == 'broadcast' and _csb == 'broadcast':
            issues.append({'level': 'error', 'code': 'cs-constant',
                           'message': f'{op} 两侧均为全截面同值（broadcast 序列，'
                                      f'如 MKT_RET）：结果无截面区分度'})
        # 截面常数（2026-09-25）：broadcast 序列（MKT_RET 类）加/减 → 仍全截面同值
        _csa, _csb = _cs_infer(node.args[0]), _cs_infer(node.args[1])
        if _csa == 'broadcast' and _csb == 'broadcast':
            issues.append({'level': 'error', 'code': 'cs-constant',
                           'message': f'{op} 两侧均为全截面同值（broadcast，如 MKT_RET）：'
                                      f'结果无截面区分度'})
        return a if a != _SCALAR else b
    # 截面算子作用于 broadcast 序列（MKT_RET 类）→ 全截面同值（RANK 恒 0.5，退化）
    if op in ('RANK', 'ZSCORE', 'SCALE') and _cs_infer(node) == 'broadcast':
        issues.append({'level': 'error', 'code': 'cs-constant',
                       'message': f'{op} 的输入为全截面同值（broadcast 序列，如 MKT_RET）：'
                                  f'截面排名无区分度——如需市场状态请改用时序比较'
                                  f'（MKT_RET vs 其自身均值）'})
    if op == 'MUL':
        return _mul(args[0], args[1])
    if op == 'DIV':
        if args[1] in ('volume', 'amount', 'ratio', 'mcap'):
            issues.append({'level': 'warning', 'code': 'div-zero-risk',
                           'message': f'分母 {args[1]} 可能为 0（停牌/无量），平台侧换手或异常值风险'})
        return _div(args[0], args[1])
    if op in ('GT', 'GE', 'LT', 'LE'):
        if args[0] != args[1] and _SCALAR not in args:
            issues.append({'level': 'warning', 'code': 'compare-unit',
                           'message': f'比较两侧单位不同：{args[0]} vs {args[1]}'})
        return 'bool'
    if op in ('AND', 'OR'):
        return 'bool'
    if op == 'IF':
        issues.append({'level': 'warning', 'code': 'tie-risk',
                       'message': 'IF 门控可能产生阶梯/并列值，平台并列打破会放大换手（参考 F-B04/B08 事故）'})
        return args[1] if args[1] != _SCALAR else args[2]
    if op in ('RANK', 'SCALE', 'ZSCORE'):
        return 'rank' if op == 'RANK' else _SCALAR
    if op in ('COUNT', 'EVERY', 'EXIST'):
        issues.append({'level': 'warning', 'code': 'tie-risk',
                       'message': f'{op} 输出为离散事件计数，稀疏并列会导致平台换手虚高'})
        return 'count'
    if op in ('CORR',):
        return _SCALAR
    if op in ('COV',):
        return _mul(args[0], args[1])
    if op == 'TS_REGRESS_BETA':
        return _SCALAR
    if op == 'TS_REGRESS_RESID':
        return args[0]  # 残差与 y 同量纲
    if op == 'MEAN' and len(node.args) == 2 and node.args[1].op == 'CONST':
        issues.append({'level': 'warning', 'code': 'mean-trap',
                       'message': 'MEAN(X,N) 是逐元素均值不是滚动均值，滚动请用 TS_MEAN/MA'})
    return args[0] if args else _SCALAR


def _amount_div_unscaled(node: Node) -> bool:
    """存在 DIV 且除数子树含 AMOUNT 字段（含嵌套，如 TS_MEAN(AMOUNT,20)）。"""
    if node.op == 'DIV':
        for _, n in walk(node.args[1]):
            if n.op == 'FIELD' and n.name == 'AMOUNT':
                return True
    return any(_amount_div_unscaled(a) for a in node.args)


def _has_const(node: Node, value: float) -> bool:
    if node.op == 'CONST' and float(node.value) == value:
        return True
    return any(_has_const(a, value) for a in node.args)


def _scale_infer(node: Node) -> str | None:
    """尺度推断（2026-09-25）：'bounded'（有界，如比率/秩/换手/常数）、'price'、
    'large'（AMOUNT/VOLUME/MARKET_CAP 等大数）——量纲相同但尺度差数个数量级的
    ADD/SUB 是退化表达式（如 (CLOSE/DELAY(CLOSE,1))−AMOUNT ≈ −AMOUNT）。

    规则：CONST/秩类/比率→bounded；OHLC→price；AMOUNT/VOLUME/MCAP→large；
    DIV 同类→bounded、分母 bounded→分子类、分子 large→large；SUB/ADD 异类→MIXED；
    MUL 仅同类有界保持有界；TS 窗口算子透传（CORR 有界）；LOG 压缩后有界。"""
    op = node.op
    if op == 'CONST':
        return 'bounded'
    if op == 'FIELD':
        return {'CLOSE': 'price', 'OPEN': 'price', 'HIGH': 'price', 'LOW': 'price',
                'AMOUNT': 'large', 'VOLUME': 'large', 'MARKET_CAP': 'large',
                'TURNOVER': 'bounded', 'MKT_RET': 'bounded', 'IND_RET': 'bounded',
                'BETA60': 'bounded', 'BETA20': 'bounded', 'IND_EXCESS': 'bounded',
                'IVOL20': 'bounded', 'VWAP': 'bounded', 'AMIHUD20': 'bounded',
                'DISP': 'bounded'}.get(node.name)
    cs = [_scale_infer(a) for a in node.args]
    if op in ('NEG', 'ABS'):
        return cs[0] if cs else None
    if op == 'LOG' or op == 'LOGABS':
        return 'bounded'
    if op in ('SIGN', 'AS_FLOAT', 'RANK', 'ZSCORE', 'SCALE', 'COUNT', 'EVERY',
              'EXIST', 'TS_RANK', 'TS_ARGMAX', 'TS_ARGMIN', 'CORR'):
        return 'bounded'
    if op in ('DIV',):
        if cs[0] is not None and cs[0] == cs[1]:
            return 'bounded'
        if cs[1] == 'bounded':
            return cs[0]
        if cs[0] == 'large' and cs[1] in ('price', 'bounded'):
            return 'large'
        return None
    if op in ('SUB', 'ADD'):
        if cs[0] is not None and cs[1] is not None and cs[0] != cs[1]:
            return 'MIXED'
        return cs[0] or cs[1]
    if op == 'MUL':
        if cs[0] == 'bounded' and cs[1] == 'bounded':
            return 'bounded'
        return None
    if op in TS_OPS or op in ('COV',):
        return cs[0] if len(set(cs)) == 1 else None
    if op == 'IF':
        return cs[1] if (len(cs) > 2 and cs[1] == cs[2]) else None
    return None


def _cs_infer(node: Node, depth: int = 0) -> str | None:
    """截面变异推断（2026-09-25）：'broadcast'（同日全市场同值，如 MKT_RET）/
    'varying'（逐股不同）/ None。

    broadcast 序列经截面算子（RANK/ZSCORE/SCALE）→ 全截面常数（无区分度，退化）；
    经时序算子透传（仍 broadcast）；与逐股序列的 CORR/回归/算术 → varying。"""
    if depth > 30:
        return None
    op = node.op
    if op == 'CONST':
        return 'broadcast'
    if op == 'FIELD':
        return 'broadcast' if node.name == 'MKT_RET' else 'varying'
    cs = [_cs_infer(a, depth + 1) for a in node.args]
    if op in ('NEG', 'ABS'):
        return cs[0] if cs else None
    if op == 'LOG' or op == 'LOGABS':
        return cs[0] if cs else None
    if op in ('RANK', 'ZSCORE', 'SCALE'):
        # 截面算子：broadcast 输入 → 全截面同值（退化标志）
        return 'broadcast' if (cs and cs[0] == 'broadcast') else 'varying'
    if op in ('CORR', 'TS_REGRESS_RESID', 'TS_REGRESS_BETA'):
        # 与逐股序列的相关/回归 → 逐股不同；两个 broadcast → 常数
        return 'varying' if 'varying' in cs else (
            'broadcast' if cs and all(c == 'broadcast' for c in cs) else None)
    if op in TS_OPS or op in ('COV',):
        return cs[0] if cs and len(set(cs)) == 1 else None
    if op in ('ADD', 'SUB', 'MUL', 'DIV', 'MAX', 'MIN', 'MEAN'):
        if cs and all(c == 'broadcast' for c in cs):
            return 'broadcast'
        if 'varying' in cs:
            return 'varying'
        return None
    if op == 'IF':
        if len(cs) == 3:
            if 'varying' in (cs[1], cs[2]):
                return 'varying'
            if cs[1] == 'broadcast' and cs[2] == 'broadcast':
                return 'broadcast'
            return None
        return None
    return None


def lint(node: Node) -> list:
    issues: list = []
    _infer(node, issues)
    # AMOUNT 尺度政策（2026-09-16 平台探针实证）代码化：AMOUNT 作分母必须显式 ×1e9，
    # 平台对 ~1e-11 数值处理异常（amihud120 原版换手 30.68%、×1e9 版 4.93%）。
    if _amount_div_unscaled(node) and not _has_const(node, 1000000000.0):
        issues.append({'level': 'warning', 'code': 'amount-scale',
                       'message': 'AMOUNT 作分母但表达式无 ×1000000000：平台对 ~1e-11 数值'
                                  '处理异常（2026-09-16 探针实证，换手 30.68%→4.93%）'})
    for _, n in walk(node):
        if n.op not in VERIFIED_OPS and n.op not in ('FIELD', 'CONST'):
            issues.append({'level': 'info', 'code': 'unverified-op',
                           'message': f'{n.op} 在已提交批次中未出现过，首次上平台建议单独冒烟'})
    seen = set()
    unique = []
    for item in issues:
        key = (item['level'], item['code'], item['message'])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def has_errors(issues: list) -> bool:
    return any(i['level'] == 'error' for i in issues)


def type_errors(node: Node) -> list:
    """组合期类型检查（2026-09-25）：unit-mismatch + scale-mix + cs-constant。

    供 grammar 变异/交叉在"出生前"重掷退化子树（lint 兜底之外的预防层）——
    与强类型 GP / grammar-masked 生成（AlphaGen/Hubble/WorldQuant FAST 的静态类型）
    同一思想：让非法表达式不可表示，而不是生出来再杀。"""
    issues: list = []
    _infer(node, issues)
    return [i for i in issues if i.get('code') in ('unit-mismatch', 'scale-mix', 'cs-constant')]


def dust_errors(node: Node, magnitude_fn, ratio: float = 100.0) -> list:
    """数值尘埃检查（2026-09-25）：ADD/SUB 两侧实测数量级差 ≥ratio 倍 → 小项为
    数值尘埃（静态尺度推断的数值补充层，覆盖 DIV 尘埃长尾，如 CLOSE/市值 ≈ 1e-9）。

    magnitude_fn: 节点 → 典型 |值|（Evaluator.magnitude，memo 化）。仅检查 error 级；
    常数平移（如 TURNOVER−1，比值 ~20）不误杀。"""
    issues: list = []
    for _p, n in walk(node):
        if n.op in ('ADD', 'SUB') and len(n.args) == 2:
            try:
                ma = abs(float(magnitude_fn(n.args[0])))
                mb = abs(float(magnitude_fn(n.args[1])))
            except Exception:
                continue
            if ma > 0 and mb > 0:
                r = max(ma, mb) / min(ma, mb)
                if r >= ratio:
                    issues.append({'level': 'error', 'code': 'dust-dominance',
                                   'message': f'{n.op} 两侧实测数量级差 {r:.1e}×：'
                                              f'小项为数值尘埃（该项贡献 <1%）'})
    return issues
