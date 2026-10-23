"""因子表达式树：渲染/哈希/校验/机制模板/变异交叉（Phase A 合并引擎）。

- 平台语法渲染：虚拟序列 RET1/LOGC 展开为平台合法子公式。
- expression_hash 以“展开后的平台公式”为准：同公式不同本地 min_periods 视为同一因子。
- 机制模板 M1~M8 只使用 VERIFIED 算子；DOCUMENTED 算子需 allow_documented=True。
"""
from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass, replace
from typing import Iterator

import _paths  # noqa: F401
from field_catalog import (
    ARITY, CS_OPS, CONST_WHITELIST, EXPONENTS, EXTENDED_ONLY_FIELDS, GATE_FIELDS,
    LOCAL_ONLY_FIELDS, MAX_DEPTH, MP_NONE, PARAM_OPS, PLATFORM_FIELDS,
    SHIFT_WINDOWS, TS_OPS, VERIFIED_OPS, WINDOWS, default_mp,
)


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple = ()
    window: int | None = None
    mp: int | None = None
    name: str | None = None
    value: float | None = None
    params: tuple = ()


def field(name: str) -> Node:
    return Node('FIELD', name=name)


def const(v: float) -> Node:
    return Node('CONST', value=float(v))


def ts(op: str, child: Node, window: int, mp: int | None = None) -> Node:
    return Node(op, (child,), window=window,
                mp=default_mp(op, window) if mp is None else mp)


def cs(op: str, child: Node) -> Node:
    return Node(op, (child,))


def virtual_node(name: str) -> Node:
    if name == 'RET1':
        return Node('SUB', (Node('DIV', (field('CLOSE'), ts('DELAY', field('CLOSE'), 1, mp=None))),
                            const(1.0)))
    if name == 'LOGC':
        return Node('LOG', (field('CLOSE'),))
    if name == 'GAP1':
        # T9（2026-09-25）：隔夜缺口 = OPEN/昨收 − 1（平台合法算子，可上传）
        return Node('SUB', (Node('DIV', (field('OPEN'),
                                         ts('DELAY', field('CLOSE'), 1, mp=None))),
                            const(1.0)))
    if name == 'BODY':
        # T9：日内实体 = (CLOSE − OPEN)/OPEN（平台合法算子，可上传）
        return Node('DIV', (Node('SUB', (field('CLOSE'), field('OPEN'))), field('OPEN')))
    raise KeyError(name)


def expand(node: Node) -> Node:
    if node.op == 'FIELD' and node.name in ('RET1', 'LOGC'):
        return expand(virtual_node(node.name))
    if not node.args:
        return node
    return replace(node, args=tuple(expand(a) for a in node.args))


def _fmt_number(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return format(float(v), '.12g')


_BINARY_SYMBOL = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/'}
_COMPARE_SYMBOL = {'GT': '>', 'GE': '>=', 'LT': '<', 'LE': '<='}


def _render(node: Node) -> str:
    op = node.op
    if op == 'FIELD':
        return str(node.name)
    if op == 'CONST':
        return _fmt_number(float(node.value))
    if op == 'NEG':
        return f"0-({_render(node.args[0])})"
    if op in ('SIGNEDPOWER', 'POWER'):
        return f"{op}({_render(node.args[0])},{_fmt_number(float(node.value))})"
    if op in PARAM_OPS:
        spec = PARAM_OPS[op]
        parts = [_render(x) for x in node.args]
        if spec['window']:
            parts.append(str(int(node.window)))
        params = tuple(node.params) if node.params else tuple(spec['params'])
        parts.extend(_fmt_number(float(p)) for p in params)
        return f"{op}({','.join(parts)})"
    if op in _BINARY_SYMBOL:
        a, b = (_render(x) for x in node.args)
        return f"({a}){_BINARY_SYMBOL[op]}({b})"
    if op in _COMPARE_SYMBOL:
        return f"({_render(node.args[0])}){_COMPARE_SYMBOL[op]}({_render(node.args[1])})"
    if op in ('AND', 'OR'):
        sym = '&&' if op == 'AND' else '||'
        return f"({_render(node.args[0])}){sym}({_render(node.args[1])})"
    if op == 'IF':
        c, a, b = (_render(x) for x in node.args)
        return f"IF({c},{a},{b})"
    if op in ('MAX', 'MIN', 'MEAN'):
        return f"{op}({_render(node.args[0])},{_render(node.args[1])})"
    if op in TS_OPS or op in ('CORR', 'COV'):
        parts = [_render(x) for x in node.args]
        parts.append(str(int(node.window)))
        return f"{op}({','.join(parts)})"
    if op in CS_OPS or ARITY.get(op) == 1:
        return f"{op}({_render(node.args[0])})"
    raise ValueError(f'unrenderable op {op}')


def render(node: Node) -> str:
    return _render(expand(node))


def canonical(node: Node) -> str:
    return ''.join(render(node).lower().split())


def expression_hash(node: Node) -> str:
    return hashlib.sha256(canonical(node).encode()).hexdigest()


def _window_bucket(window: int) -> str:
    if window <= 20:
        return 'S'
    if window <= 60:
        return 'M'
    return 'L'


def structural_fingerprint(node: Node) -> str:
    """近重复指纹：字段多重集 + 带窗口算子序列 + 常数集（忽略 mp）。

    2026-09-16 修订：不再按窗口粗桶（S/M/L），窗口值与常数进入指纹，
    避免 TS_MEAN(X,120) 与 TS_MEAN(X,250) 这类"同结构不同参数"被误杀为克隆
    （旧版导致 69% 生成在评估前被跳过、覆盖塌缩）。
    """
    fields, ops, consts = [], [], []
    for _, n in walk(node):
        if n.op == 'FIELD':
            fields.append(str(n.name))
        elif n.op == 'CONST':
            consts.append(format(float(n.value), '.6g'))
        else:
            ops.append(f'{n.op}@{int(n.window)}' if n.window else n.op)
    payload = '|'.join([','.join(sorted(fields)), ','.join(ops), ','.join(sorted(consts))])
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def candidate_id(node: Node, direction: int) -> str:
    return hashlib.sha256(f"{canonical(node)}|{int(direction)}".encode()).hexdigest()[:12]


def depth(node: Node) -> int:
    if not node.args:
        return 1
    return 1 + max(depth(a) for a in node.args)


def walk(node: Node, path: tuple = ()) -> Iterator[tuple]:
    yield path, node
    for i, a in enumerate(node.args):
        yield from walk(a, path + (i,))


def get_at(node: Node, path: tuple) -> Node:
    for i in path:
        node = node.args[i]
    return node


def replace_at(node: Node, path: tuple, new: Node) -> Node:
    if not path:
        return new
    i = path[0]
    args = list(node.args)
    args[i] = replace_at(args[i], path[1:], new)
    return replace(node, args=tuple(args))


_CHILD_CATEGORY = {
    'IF': ('bool', 'num', 'num'),
    'GT': ('num', 'num'), 'GE': ('num', 'num'), 'LT': ('num', 'num'), 'LE': ('num', 'num'),
    'AND': ('bool', 'bool'), 'OR': ('bool', 'bool'),
}


def positions(node: Node, path: tuple = ()) -> Iterator[tuple]:
    cats = _CHILD_CATEGORY.get(node.op)
    for i, a in enumerate(node.args):
        cat = cats[i] if cats else 'num'
        yield path + (i,), a, cat
        yield from positions(a, path + (i,))


def validate(node: Node, allow_documented: bool = False,
             strict_const: bool = True, max_depth: int = MAX_DEPTH) -> tuple[bool, str]:
    allowed = set(VERIFIED_OPS)
    if depth(node) > max_depth:
        return False, f'depth {depth(node)} > {max_depth}'
    for _, n in walk(expand(node)):
        if n.op == 'FIELD':
            if n.name in PLATFORM_FIELDS:
                continue
            # 本地专用字段：本地能力 profile 可搜索/求值；含本地字段的公式只在 C2C
            # export/upload adapter 按 formula_uses_local 硬拦截，不影响 local shortlist。
            if n.name in LOCAL_ONLY_FIELDS and allow_documented:
                continue
            if n.name in EXTENDED_ONLY_FIELDS and allow_documented:
                continue
            if n.name in GATE_FIELDS and allow_documented:
                continue
            return False, f'unknown field {n.name}'
        if n.op == 'CONST':
            if n.value is None or (strict_const and float(n.value) not in CONST_WHITELIST):
                return False, f'const {n.value} not in whitelist'
            continue
        if n.op not in ARITY:
            return False, f'unknown op {n.op}'
        if len(n.args) != ARITY[n.op]:
            return False, f'{n.op} expects {ARITY[n.op]} args, got {len(n.args)}'
        if n.op in PARAM_OPS:
            spec = PARAM_OPS[n.op]
            if spec['window']:
                if not isinstance(n.window, int) or n.window not in WINDOWS:
                    return False, f'bad window {n.window} on {n.op}'
            elif n.window is not None:
                return False, f'window on non-window op {n.op}'
            if n.params and len(n.params) != len(spec['params']):
                return False, f'{n.op} expects {len(spec["params"])} params'
        elif n.op in TS_OPS or n.op in ('CORR', 'COV'):
            if not isinstance(n.window, int):
                return False, f'missing window on {n.op}'
            pool = SHIFT_WINDOWS if n.op in MP_NONE else WINDOWS
            if n.window not in pool:
                return False, f'bad window {n.window} on {n.op}'
        elif n.window is not None:
            return False, f'window on non-TS op {n.op}'
        if n.op in MP_NONE and n.mp is not None:
            return False, f'{n.op} must not carry mp'
        if not allow_documented and n.op not in allowed:
            return False, f'documented-only op {n.op} (allow_documented=False)'
    return True, ''


def _win(rng: random.Random, pool=WINDOWS) -> int:
    return rng.choice(pool)


def random_condition(rng: random.Random) -> Node:
    w = _win(rng, (5, 10, 20, 60))
    left = field('CLOSE') if rng.random() < 0.5 else virtual_node('RET1')
    right = ts('DELAY', field('CLOSE'), w, mp=None)
    # 2026-09-24：GE/LT/LE 进条件生成（此前只产 GT；四个比较算子都是 VERIFIED）
    cmp_op = rng.choice(('GT', 'GE', 'LT', 'LE'))
    if rng.random() < 0.5:
        return Node(cmp_op, (ts('TS_MEAN', left, w), right))
    return Node(cmp_op, (left, right))


def random_num(rng: random.Random) -> Node:
    if rng.random() < 0.5:
        return field(rng.choice(('CLOSE', 'VOLUME', 'TURNOVER', 'AMOUNT')))
    return ts('TS_MEAN', field(rng.choice(('CLOSE', 'VOLUME', 'TURNOVER', 'AMOUNT'))),
              _win(rng, (5, 10, 20, 60)))


def m1_return(rng: random.Random) -> Node:
    ret = virtual_node('RET1')
    w = _win(rng, (5, 10, 20, 40, 60, 120))
    span = rng.choice((3, 5, 10, 20))
    form = rng.randrange(6)
    if form == 0:
        return ts('TS_MEAN', ret, w)
    if form == 1:
        return Node('NEG', (ts('TS_MEAN', ret, w),))
    if form == 2:
        return Node('DIV', (ts('TS_MEAN', ret, w), ts('STDDEV', ret, w)))
    if form == 3:
        return ts('EMA', ts('TS_MEAN', ret, w), span)
    if form == 4:
        pos = Node('IF', (Node('GT', (ret, const(0))), ret, const(0)))
        return ts('TS_MEAN', pos, w)
    return Node('NEG', (ts('EMA', ts('TS_MEAN', ret, w), span),))


def m2_liquidity(rng: random.Random) -> Node:
    ret = virtual_node('RET1')
    w = _win(rng, (20, 40, 60, 120, 250))
    form = rng.randrange(6)

    def turnover_level() -> Node:
        # log 政策：A/B 实测（800股/24月）换手水平 log 版 SiA 反而下降，故 log 仅保留
        # 35% 探索配额；市值 log 实测 SiA 翻倍，size 形式一律 LOG。
        base = field('TURNOVER')
        if rng.random() < 0.35:
            base = Node('LOG', (base,))
        return ts('TS_MEAN', base, w)

    if form == 0:
        return turnover_level()
    if form == 1:
        return Node('NEG', (turnover_level(),))
    if form == 2:
        vr = Node('SUB', (Node('DIV', (field('VOLUME'), ts('TS_MEAN', field('VOLUME'), w))),
                          const(1.0)))
        return ts('EMA', vr, rng.choice((5, 10, 20)))
    if form == 3:
        ill = Node('MUL', (Node('DIV', (Node('ABS', (ret,)), field('AMOUNT'))),
                           const(1000000000.0)))
        return Node('LOG', (Node('ADD', (const(1.0), ts('TS_MEAN', ill, w))),))
    if form == 4:
        small = _win(rng, (5, 10, 20))
        big = w if w > small else 60
        return Node('DIV', (ts('TS_MEAN', field('TURNOVER'), small),
                            ts('TS_MEAN', field('TURNOVER'), big)))
    size = Node('LOG', (field('MARKET_CAP'),))
    return Node('NEG', (size,)) if rng.random() < 0.5 else size


def m3_distribution(rng: random.Random) -> Node:
    ret = virtual_node('RET1')
    w = _win(rng, (60, 120, 250))
    form = rng.randrange(3)
    if form == 0:
        return ts('TS_SKEW', ret, w)
    if form == 1:
        return Node('NEG', (ts('TS_SKEW', ret, w),))
    return ts('TS_ZSCORE', ret, w)


def m4_location(rng: random.Random) -> Node:
    w = _win(rng, (20, 40, 60, 120, 250))
    form = rng.randrange(5)
    if form == 0:
        return Node('SUB', (Node('DIV', (field('CLOSE'), ts('TS_MAX', field('HIGH'), w))),
                            const(1.0)))
    if form == 1:
        return Node('SUB', (Node('DIV', (field('CLOSE'), ts('TS_MIN', field('LOW'), w))),
                            const(1.0)))
    if form == 2:
        span = Node('SUB', (ts('TS_MAX', field('HIGH'), w), ts('TS_MIN', field('LOW'), w)))
        return Node('DIV', (Node('SUB', (field('CLOSE'), ts('TS_MIN', field('LOW'), w))), span))
    if form == 3:
        inside = Node('SUB', (Node('DIV', (field('CLOSE'), ts('TS_MAX', field('HIGH'), w))),
                              const(1.0)))
        return Node('NEG', (inside,))
    return ts('TS_MEAN', Node('DIV', (field('CLOSE'), ts('TS_MAX', field('HIGH'), w))), w)


def m5_trend(rng: random.Random) -> Node:
    logc = virtual_node('LOGC')
    w = _win(rng, (20, 40, 60, 120, 250))
    form = rng.randrange(4)
    if form == 0:
        return ts('SLOPE', logc, w)
    if form == 1:
        return Node('NEG', (ts('SLOPE', logc, w),))
    if form == 2:
        return Node('DIV', (ts('SLOPE', logc, w), ts('STDDEV', virtual_node('RET1'), w)))
    return ts('EMA', ts('SLOPE', logc, w), rng.choice((3, 5, 10, 20)))


def m6_price_volume(rng: random.Random) -> Node:
    w = _win(rng, (20, 40, 60, 120))
    ret = virtual_node('RET1')
    pairs = [(field('CLOSE'), field('VOLUME')), (ret, field('VOLUME')),
             (field('CLOSE'), field('TURNOVER')), (ret, field('TURNOVER'))]
    a, b = rng.choice(pairs)
    node = Node('CORR', (a, b), window=w, mp=default_mp('CORR', w))
    return Node('NEG', (node,)) if rng.random() < 0.4 else node


def m7_volatility(rng: random.Random) -> Node:
    ret = virtual_node('RET1')
    w = _win(rng, (20, 40, 60, 120))
    form = rng.randrange(4)
    if form == 0:
        return ts('STDDEV', ret, w)
    if form == 1:
        return Node('NEG', (ts('STDDEV', ret, w),))
    if form == 2:
        return Node('DIV', (ts('STDDEV', ret, w), ts('TS_MEAN', Node('ABS', (ret,)), w)))
    return ts('STDDEV', Node('DIV', (field('CLOSE'), ts('TS_MEAN', field('CLOSE'), w))), w)


def m8_gate(rng: random.Random) -> Node:
    ret = virtual_node('RET1')
    w = _win(rng, (5, 10, 20, 40, 60))
    w2 = _win(rng, (10, 20, 40, 60))
    form = rng.randrange(4)
    if form == 0:
        cond = Node('GT', (field('CLOSE'), ts('DELAY', field('CLOSE'), w, mp=None)))
        return ts('TS_MEAN', Node('IF', (cond, ret, const(0))), w2)
    if form == 1:
        cond = Node('GT', (field('TURNOVER'), ts('TS_MEAN', field('TURNOVER'), w)))
        return ts('TS_MEAN', Node('IF', (cond, ret, const(0))), w2)
    if form == 2:
        return ts('TS_MEAN', Node('IF', (Node('GT', (ret, const(0))), ret, const(0))), w)
    cond = Node('GT', (cs('RANK', field('TURNOVER')), const(0.5)))
    return ts('TS_MEAN', Node('IF', (cond, ret, const(0))), w2)


# ---------------------------------------------------------------- grammar v2
def _safe_illiquidity(ret: Node, w: int) -> Node:
    """平台安全非流动性：LOG(1+TS_MEAN(ABS(ret)/AMOUNT*1e9,w))。"""
    ill = Node('MUL', (Node('DIV', (Node('ABS', (ret,)), field('AMOUNT'))),
                       const(1000000000.0)))
    return Node('LOG', (Node('ADD', (const(1.0), ts('TS_MEAN', ill, w))),))


def _composite_leg(rng: random.Random) -> Node:
    """复合腿银行（全部 VERIFIED 算子；AMOUNT 腿用平台安全尺度）。
    T10（2026-09-25）：加入 GAP1/BODY 腿（M15 机制进复合；平台合法可上传）。"""
    ret = virtual_node('RET1')
    logc = virtual_node('LOGC')
    w = _win(rng, (40, 60, 120, 250))
    span = rng.choice((3, 5, 10, 20))
    builders = [
        lambda: ts('TS_MEAN', ret, w),
        lambda: ts('TS_SKEW', ret, max(w, 60)),
        lambda: Node('SUB', (Node('DIV', (field('CLOSE'), ts('TS_MAX', field('HIGH'), w))),
                             const(1.0))),
        lambda: ts('SLOPE', logc, w),
        lambda: ts('TS_MEAN', field('TURNOVER'), w),
        lambda: Node('SUB', (Node('DIV', (field('VOLUME'), ts('TS_MEAN', field('VOLUME'), w))),
                             const(1.0))),
        lambda: _safe_illiquidity(ret, w),
        lambda: ts('STDDEV', ret, w),
        lambda: ts('EMA', ts('TS_MEAN', ret, w), span),
        lambda: Node('CORR', (ret, field('VOLUME')), window=w, mp=default_mp('CORR', w)),
        lambda: virtual_node('GAP1'),
        lambda: virtual_node('BODY'),
        lambda: ts('TS_MEAN', virtual_node('BODY'), w),
    ]
    leg = rng.choice(builders)()
    return Node('NEG', (leg,)) if rng.random() < 0.5 else leg


def m9_composite(rng: random.Random) -> Node:
    """自动 RANK 和复合（comp6 结构）：2~4 条腿，各取 RANK 后等权相加。"""
    k = rng.choice((2, 3, 4))
    legs = []
    for _ in range(k):
        legs.append(Node('RANK', (_composite_leg(rng),)))
    acc = legs[0]
    for leg in legs[1:]:
        acc = Node('ADD', (acc, leg))
    return acc


def composite_from_legs(legs: list[tuple[Node, int]], rng: random.Random) -> Node | None:
    """腿库交叉重组（comp6 式）：把已评估的强腿按各自方向组成 RANK 和复合。

    legs: [(node, direction)]，direction=1 表示高值优（直接 RANK），0 表示低值优（RANK(0-x)）。
    返回 None 表示构造/校验失败，由调用方退回随机模板。
    """
    if len(legs) < 2:
        return None
    parts = []
    for node, d in legs:
        x = node if int(d) == 1 else Node('NEG', (node,))
        parts.append(Node('RANK', (x,)))
    rng.shuffle(parts)
    acc = parts[0]
    for leg in parts[1:]:
        acc = Node('ADD', (acc, leg))
    if depth(acc) > MAX_DEPTH:
        return None
    ok, _ = validate(acc, allow_documented=True)
    return acc if ok else None


def m10_shape(rng: random.Random) -> Node:
    """路径/形状特征：效率比、区间位置、均线距离、让利回吐、上下行波动结构。"""
    ret = virtual_node('RET1')
    w = _win(rng, (20, 40, 60, 120, 250))
    form = rng.randrange(5)
    if form == 0:  # 效率比
        move = Node('ABS', (Node('SUB', (Node('DIV', (field('CLOSE'),
                                                     ts('DELAY', field('CLOSE'), w, mp=None))),
                                       const(1.0))),))
        return Node('DIV', (move, ts('TS_MEAN', Node('ABS', (ret,)), w)))
    if form == 1:  # 区间位置
        span = Node('SUB', (ts('TS_MAX', field('HIGH'), w), ts('TS_MIN', field('LOW'), w)))
        return Node('DIV', (Node('SUB', (field('CLOSE'), ts('TS_MIN', field('LOW'), w))), span))
    if form == 2:  # 双均线距离
        w2 = _win(rng, (5, 10, 20))
        big = w if w > w2 else 60
        return Node('SUB', (Node('DIV', (ts('TS_MEAN', field('CLOSE'), w2),
                                         ts('TS_MEAN', field('CLOSE'), big))), const(1.0)))
    if form == 3:  # 让利回吐（runup giveback）
        runup = Node('SUB', (Node('DIV', (field('CLOSE'), ts('TS_MIN', field('LOW'), w))),
                             const(1.0)))
        return Node('NEG', (ts('EMA', runup, rng.choice((3, 5, 10, 20))),))
    # 上下行波动结构
    down = ts('STDDEV', Node('IF', (Node('LT', (ret, const(0))), ret, const(0))), w)
    allv = ts('STDDEV', ret, w)
    return Node('DIV', (down, allv))


def m11_regime(rng: random.Random) -> Node:
    """状态门（regime conditional）：波动/相关/换手 z 状态下的收益条件均值与不对称。"""
    ret = virtual_node('RET1')
    w = _win(rng, (20, 40, 60))
    w2 = _win(rng, (10, 20, 40, 60))
    form = rng.randrange(5)
    if form == 0:  # 高波状态才持有
        vol = ts('STDDEV', ret, w)
        cond = Node('GT', (vol, ts('TS_MEAN', vol, w2)))
        return ts('TS_MEAN', Node('IF', (cond, ret, const(0))), w2)
    if form == 1:  # 量价正相关状态才持有
        corr = Node('CORR', (field('CLOSE'), field('VOLUME')), window=w,
                    mp=default_mp('CORR', w))
        cond = Node('GT', (corr, const(0)))
        return ts('TS_MEAN', Node('IF', (cond, ret, const(0))), w2)
    if form == 2:  # 换手 z 高位才持有
        cond = Node('GT', (ts('TS_ZSCORE', field('TURNOVER'), w), const(0.5)))
        return ts('TS_MEAN', Node('IF', (cond, ret, const(0))), w2)
    if form == 3:  # 上行/全波动占比
        up = ts('TS_MEAN', Node('IF', (Node('GT', (ret, const(0))), ret, const(0))), w)
        return Node('DIV', (up, ts('TS_MEAN', Node('ABS', (ret,)), w)))
    # 低波状态下的振幅
    vol = ts('STDDEV', ret, w)
    cond = Node('LT', (vol, ts('TS_MEAN', vol, w2)))
    return ts('TS_MEAN', Node('IF', (cond, Node('ABS', (ret,)), const(0))), w2)


def m12_orderstats(rng: random.Random) -> Node:
    """DOCUMENTED 算子族：TS_RANK/TS_ARGMAX/TS_MEDIAN/TS_KURT/保号幂（需平台冒烟）。"""
    ret = virtual_node('RET1')
    w = _win(rng, (20, 40, 60, 120))
    form = rng.randrange(6)
    if form == 0:
        return ts('TS_RANK', field('CLOSE'), w)
    if form == 1:
        return ts('TS_RANK', ret, w)
    if form == 2:
        return ts('TS_ARGMAX', field('HIGH'), w)
    if form == 3:
        med = ts('TS_MEDIAN', field('CLOSE'), w)
        return Node('SUB', (Node('DIV', (field('CLOSE'), med)), const(1.0)))
    if form == 4:
        return ts('TS_KURT', ret, w)
    return Node('SIGNEDPOWER', (ret,), value=rng.choice((0.5, 2.0)))


def m13_tech(rng: random.Random) -> Node:
    """官方技术指标族（DOCUMENTED，需 --allow-documented + 平台冒烟）。"""
    w = _win(rng, (6, 12, 14, 20, 24, 26, 40, 60))
    close, high, low = field('CLOSE'), field('HIGH'), field('LOW')
    form = rng.randrange(14)
    if form == 0:
        node = ts('RSI', close, w)
    elif form == 1:
        node = ts('WR', close, w)
    elif form == 2:
        node = ts('PSY', close, w)
    elif form == 3:
        node = ts('BIAS', close, w)
    elif form == 4:
        node = ts('CCI', close, w)
    elif form == 5:
        node = ts('BOLL_WIDTH', close, w)
    elif form == 6:
        span = Node('SUB', (ts('BOLL_UPPER', close, w), ts('BOLL_LOWER', close, w)))
        node = Node('DIV', (Node('SUB', (close, ts('BOLL_LOWER', close, w))), span))
    elif form == 7:
        node = Node('KDJ_K', (close, high, low))
    elif form == 8:
        node = Node('KDJ_J', (close, high, low))
    elif form == 9:
        node = Node('MACD_DIF', (close,))
    elif form == 10:
        node = Node('MASS', (high, low))
    elif form == 11:
        node = Node('VR', (close, field('VOLUME')))
    elif form == 12:
        obv = Node('OBV', (close, field('VOLUME')))
        node = ts('SLOPE', obv, _win(rng, (20, 40, 60)))
    else:
        dema = ts('DEMA', close, w)
        node = Node('SUB', (Node('DIV', (dema, close)), const(1.0)))
    return Node('NEG', (node,)) if rng.random() < 0.4 else node


def m14_context(rng: random.Random) -> Node:
    """市场/行业上下文族（2026-09-24 层2，本地专用）：MKT_RET/IND_RET/BETA60 交互。

    全部 VERIFIED 算子 + 三个本地字段；含本地字段的公式禁止 C2C 上传，但保留本地搜索/评估。
    随 local-context 能力启用生成。
    V4.3 Class2：form 6~9 新增滚动回归 Beta/残差与扩展字段腿（任意窗口 Beta、
    行业 Beta、市场残差、相对行业残量信息比），搜索引擎可名正言顺搜出 Beta20 类特征。
    """
    ret = virtual_node('RET1')
    mkt = field('MKT_RET')
    ind = field('IND_RET')
    beta = field('BETA60')
    w = _win(rng, (20, 40, 60, 120))
    w2 = _win(rng, (5, 10, 20, 40))
    wb = _win(rng, (10, 20, 40, 60))
    form = rng.randrange(12)
    if form == 0:
        # 市况门：市场均量下行期做反转、上行期做动量（市况状态切换）
        cond = Node('LT', (ts('TS_MEAN', mkt, w), const(0)))
        node = ts('TS_MEAN', Node('IF', (cond, Node('NEG', (ret,)), ret)), w2)
    elif form == 1:
        # 行业相对强度（alpha 流）：个股收益 − 行业收益 的平滑
        rel = Node('SUB', (ret, ind))
        node = ts('TS_MEAN', rel, w)
        if rng.random() < 0.5:
            node = ts('EMA', node, w2)
    elif form == 2:
        # 行业相对强度的风险调整（信息比率形）
        rel = Node('SUB', (ret, ind))
        node = Node('DIV', (ts('TS_MEAN', rel, w), ts('STDDEV', rel, w)))
    elif form == 3:
        # 个股-市场滚动相关（beta 相关类，逐股滚动实现）
        node = Node('CORR', (ret, mkt), window=w, mp=default_mp('CORR', w))
    elif form == 4:
        # 行业门：行业均量下行期做反转
        cond = Node('LT', (ts('TS_MEAN', ind, w), const(0)))
        node = ts('TS_MEAN', Node('IF', (cond, Node('NEG', (ret,)), ret)), w2)
    elif form == 6:
        # 市场 Beta（任意窗口）：个股对市场的弹性，股性切分核心
        node = Node('TS_REGRESS_BETA', (ret, mkt), window=wb,
                    mp=default_mp('TS_REGRESS_BETA', wb))
    elif form == 7:
        # 行业 Beta：个股对行业的弹性（跟风股 vs 独立妖股）
        node = Node('TS_REGRESS_BETA', (ret, ind), window=wb,
                    mp=default_mp('TS_REGRESS_BETA', wb))
    elif form == 8:
        # 市场残差平滑：剥离市场成分后的特质收益
        node = ts('TS_MEAN', Node('TS_REGRESS_RESID', (ret, mkt), window=wb,
                                 mp=default_mp('TS_REGRESS_RESID', wb)), w2)
    elif form == 10:
        # V4.3d 模式A：市场门控翻转（牛市做动量、熊市做反转的原生进化形）
        _gate = 'MKT_GATE1' if rng.random() < 0.5 else 'MKT_GATE2'
        node = Node('MUL', (field(_gate),
                            Node('RANK', (ts('TS_MEAN', ret, w2),))))
    elif form == 11:
        # V4.3d 模式A：行业门控条件加权（高状态行业放大个股腿）
        _gate = 'SEC_GATE1' if rng.random() < 0.5 else 'SEC_GATE2'
        node = Node('MUL', (field(_gate),
                            Node('RANK', (ts('TS_MEAN', ret, w2),))))
    else:
        # 相对行业残量信息比：IND_EXCESS 的均值/波动（V4.3 扩展字段腿）
        iex = field('IND_EXCESS')
        node = Node('DIV', (ts('TS_MEAN', iex, w), ts('STDDEV', iex, w)))
    if form == 5:
        # alpha 分解：RET − BETA60×MKT_RET（剥离市场成分后的残差收益平滑）
        node = ts('TS_MEAN', Node('SUB', (ret, Node('MUL', (beta, mkt)))), w)
    return Node('NEG', (node,)) if rng.random() < 0.35 else node


def m15_gap(rng: random.Random) -> Node:
    """隔夜缺口/日内实体族（T9，2026-09-25）：GAP1=OPEN/昨收−1，BODY=(C−O)/O。

    A 股短周期文献强因子（隔夜收益、日内动量）；全部平台合法算子（OPEN/DELAY/CLOSE），
    可上传比赛线。此前 grammar 对 OPEN 0 覆盖（模板缺失），本族补齐该轴。
    """
    gap = virtual_node('GAP1')
    body = virtual_node('BODY')
    ret = virtual_node('RET1')
    w = _win(rng, (5, 10, 20, 40, 60))
    w2 = _win(rng, (3, 5, 10))
    form = rng.randrange(6)
    if form == 0:
        node = ts('TS_MEAN', gap, w)                     # 隔夜缺口动量
    elif form == 1:
        node = ts('TS_MEAN', body, w)                    # 日内实体方向
    elif form == 2:
        node = ts('EMA', ts('TS_MEAN', gap, w), w2)      # 平滑缺口
    elif form == 3:
        node = Node('DIV', (ts('TS_MEAN', body, w), ts('STDDEV', body, w)))  # 实体信息比率
    elif form == 4:
        node = Node('CORR', (ret, gap), window=w, mp=default_mp('CORR', w))  # 缺口-动量相关
    else:
        node = ts('TS_MEAN', Node('SUB', (gap, ts('TS_MEAN', gap, w))), w)   # 缺口偏离均值
    return Node('NEG', (node,)) if rng.random() < 0.4 else node


_M16_LEG_BUILDERS = [
    # 跨机制腿池（tag, builder）：组合时强制两腿异族——机制 A×B 的系统化配对
    ('M1_mom', lambda rng: virtual_node('RET1')),
    ('M1_smooth', lambda rng: ts('TS_MEAN', virtual_node('RET1'), _win(rng, (20, 60, 120)))),
    ('M2_turn', lambda rng: ts('TS_MEAN', field('TURNOVER'), _win(rng, (10, 20, 60)))),
    ('M2_volratio', lambda rng: Node('SUB', (Node('DIV', (
        field('VOLUME'), ts('TS_MEAN', field('VOLUME'), 250, mp=250))), const(1.0)))),
    ('M3_skew', lambda rng: ts('TS_SKEW', virtual_node('RET1'), 250)),
    ('M5_slope', lambda rng: ts('SLOPE', virtual_node('LOGC'), _win(rng, (40, 120)))),
    ('M7_vol', lambda rng: ts('STDDEV', virtual_node('RET1'), _win(rng, (20, 60)))),
    ('M15_gap', lambda rng: virtual_node('GAP1')),
    ('M15_body', lambda rng: virtual_node('BODY')),
]


def _m16_two_legs(rng: random.Random) -> tuple[Node, Node]:
    """抽两条**异族**腿（tag 不同）——跨机制组合的意义就在 A×B 配对。"""
    for _ in range(10):
        ta, tb = rng.sample(_M16_LEG_BUILDERS, 2)
        if ta[0] != tb[0]:
            return ta[1](rng), tb[1](rng)
    return _M16_LEG_BUILDERS[0][1](rng), _M16_LEG_BUILDERS[1][1](rng)


def m16_interaction(rng: random.Random) -> Node:
    """跨机制交互族（T10，2026-09-25）：两个不同机制的腿按 条件门/秩乘/秩差/比率 组合。

    此前跨机制组合只有 M9 的 RANK 等权相加（腿库抽签撞运气）；本族系统化 A×B：
    M1 动量 × M2 流动性、M15 缺口 × M1 动量、M2 × M15 等。全部平台合法算子，
    可上传；映射 interaction 轴。"""
    a, b = _m16_two_legs(rng)
    w = _win(rng, (10, 20, 40))
    form = rng.randrange(8)
    if form == 0:
        # 条件门：状态 A 高于自身均值时持有 B（跨机制状态切换）
        cond = Node('GT', (a, ts('TS_MEAN', a, w)))
        node = ts('TS_MEAN', Node('IF', (cond, b, const(0))), w)
    elif form == 1:
        # 秩乘：两机制的截面排名相乘（交集强化，双高才高）
        node = Node('MUL', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 2:
        # 秩差：A 机制强于 B 机制（跨机制相对强弱）
        node = Node('SUB', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 3:
        # 比率：A 均值 / B 均值（跨机制弹性）
        node = Node('DIV', (ts('TS_MEAN', a, w), ts('TS_MEAN', b, w)))
    elif form == 4:
        # 秩最大：两机制中较强的那个（非对称选优——谁强用谁）
        node = Node('MAX', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 5:
        # 秩最小：两机制中较弱的那个（双低信号——木桶效应）
        node = Node('MIN', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 6:
        # 双条件门：两机制状态同时对齐才持有动量（更严格的过滤）
        # AND 用 MUL 实现（GT 输出 0/1 → MUL = 逻辑 AND，平台合法 VERIFIED 算子）
        cond_a = Node('GT', (a, ts('TS_MEAN', a, w)))
        cond_b = Node('GT', (b, ts('TS_MEAN', b, w)))
        node = ts('TS_MEAN', Node('IF', (Node('MUL', (cond_a, cond_b)),
                                          virtual_node('RET1'), const(0))), w)
    else:
        # 双偏离差：两机制各自偏离自身均值的差（共振偏离度）
        node = ts('TS_MEAN', Node('SUB',
                                  (Node('SUB', (a, ts('TS_MEAN', a, w))),
                                   Node('SUB', (b, ts('TS_MEAN', b, w))))), w)
    return Node('NEG', (node,)) if rng.random() < 0.4 else node


def _beta_leg(rng: random.Random, ret: Node, against: str) -> Node:
    """滚动回归 Beta 腿（V4.3 Class2）：窗口与 mp 一致（回归全窗语义）。"""
    w = _win(rng, (10, 20, 40, 60))
    return Node('TS_REGRESS_BETA', (ret, field(against)), window=w,
                mp=default_mp('TS_REGRESS_BETA', w))


def m16_ctx_interaction(rng: random.Random) -> Node:
    """跨机制交互·上下文版（T10，2026-09-25，本地专用）：行业状态/上下文腿
    与 M1/M2/M15 逐股腿的交互。

    腿分三层（2026-09-25 加 IND_RET/DISP 行业状态腿）：
      逐股上下文: RET1−IND_RET（残量动量）、RET1−BETA60×MKT_RET（alpha）
      行业状态:   IND_RET（行业收益——TS 算子下即行业动量/波动）、DISP（行业内分化）
      广播:       MKT_RET（同日全市场同值——截面秩恒 0.5，仅 form0 状态门可用）
    含本地字段 → C2C adapter 禁止上传（formula_uses_local），本地 profile 保留评估；随 local-context 启用。"""
    ret = virtual_node('RET1')
    ctx_legs = [
        lambda rng: Node('SUB', (ret, field('IND_RET'))),
        lambda rng: Node('SUB', (ret, Node('MUL', (field('BETA60'), field('MKT_RET'))))),
        # V4.3 Class2：任意窗口市场/行业 Beta 腿 + 特质波动腿（股性切分）
        lambda rng: _beta_leg(rng, ret, 'MKT_RET'),
        lambda rng: _beta_leg(rng, ret, 'IND_RET'),
        lambda rng: field('IVOL20'),
        lambda rng: field('BETA20'),
    ]
    state_legs = [
        lambda rng: field('IND_RET'),
        lambda rng: field('DISP'),
    ]
    bcast_legs = [lambda rng: field('MKT_RET')]
    base_legs = [
        lambda rng: ts('TS_MEAN', ret, _win(rng, (20, 60))),
        lambda rng: ts('TS_MEAN', field('TURNOVER'), _win(rng, (10, 20))),
        lambda rng: virtual_node('GAP1'),
        lambda rng: virtual_node('BODY'),
    ]
    form = rng.randrange(10)
    if form in (1, 2, 4, 5):
        # 秩算术/秩选优：行业状态腿（跨行业变异）合法；broadcast 腿截面秩恒 0.5 → 排除
        a = rng.choice(ctx_legs + state_legs)(rng)
    elif form in (0, 3, 6):
        # 状态门/比率/双条件：条件可为行业状态/广播/逐股上下文
        a = rng.choice(ctx_legs + state_legs + bcast_legs)(rng)
    else:
        # 双偏离差：仅逐股上下文腿（broadcast 偏离为常数，混入后无截面变异）
        a = rng.choice(ctx_legs)(rng)
    b = rng.choice(base_legs)(rng)
    w = _win(rng, (10, 20, 40))
    if form == 8:
        # V4.3d 模式A：市场门控翻转（原生进化形 flip）
        _gate = 'MKT_GATE1' if rng.random() < 0.5 else 'MKT_GATE2'
        node = Node('MUL', (field(_gate), Node('RANK', (b,))))
    elif form == 9:
        # V4.3d 模式A：行业门控条件加权（原生进化形 sector-gain）
        _gate = 'SEC_GATE1' if rng.random() < 0.5 else 'SEC_GATE2'
        node = Node('MUL', (field(_gate), Node('RANK', (b,))))
    elif form == 0:
        cond = Node('GT', (a, ts('TS_MEAN', a, w)))
        node = ts('TS_MEAN', Node('IF', (cond, b, const(0))), w)
    elif form == 1:
        node = Node('MUL', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 2:
        node = Node('SUB', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 3:
        node = Node('DIV', (ts('TS_MEAN', a, w), ts('TS_MEAN', b, w)))
    elif form == 4:
        node = Node('MAX', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 5:
        node = Node('MIN', (Node('RANK', (a,)), Node('RANK', (b,))))
    elif form == 6:
        cond_a = Node('GT', (a, ts('TS_MEAN', a, w)))
        cond_b = Node('GT', (b, ts('TS_MEAN', b, w)))
        node = ts('TS_MEAN', Node('IF', (Node('AND', (cond_a, cond_b)),
                                          virtual_node('RET1'), const(0))), w)
    else:
        node = ts('TS_MEAN', Node('SUB',
                                  (Node('SUB', (a, ts('TS_MEAN', a, w))),
                                   Node('SUB', (b, ts('TS_MEAN', b, w))))), w)
    return Node('NEG', (node,)) if rng.random() < 0.4 else node


def m17_context_factor(rng: random.Random) -> Node:
    """纯上下文因子族（T11，2026-09-25，本地专用）：IND_RET/DISP 的 TS 变换——
    行业动量/行业波动/行业内分化/期限结构，广播到成员股。

    跨截面有变异（131 个行业→131 个不同值），可被截面 IC 框架评估；
    但有效截面 N = 行业数（131）而非股票数——ICIR 显著性需自行折扣。
    含本地字段 → C2C adapter 禁止上传；本地 profile 保留评估；随 local-context 启用。"""
    ind_ret = field('IND_RET')
    disp = field('DISP')
    mkt = field('MKT_RET')
    w = _win(rng, (5, 10, 20, 40, 60))
    w_long = _win(rng, (60, 120))
    form = rng.randrange(6)
    if form == 0:
        node = ts('TS_MEAN', ind_ret, w)                                # 行业动量
    elif form == 1:
        node = ts('STDDEV', ind_ret, w)                                 # 行业波动
    elif form == 2:
        node = ts('TS_MEAN', disp, w)                                   # 行业内分化水平
    elif form == 3:
        # 分散度状态：当前分化 / 历史分化均值（>1 = 分化加剧）
        node = Node('DIV', (disp, ts('TS_MEAN', disp, 60)))
    elif form == 4:
        # 行业动量期限结构：短均值 − 长均值（>0 = 短期强势）
        node = Node('SUB', (ts('TS_MEAN', ind_ret, 5), ts('TS_MEAN', ind_ret, w_long)))
    else:
        # 行业动量市场相对：行业动量 − 市场动量（行业轮动的方向性）
        node = ts('TS_MEAN', Node('SUB', (ind_ret, mkt)), w)
    return Node('NEG', (node,)) if rng.random() < 0.4 else node


DOC_TEMPLATES = {
    'M12_orderstats': m12_orderstats,
    'M13_tech': m13_tech,
    'M14_context': m14_context,
    'M16_ctx_interaction': m16_ctx_interaction,
    'M17_context_factor': m17_context_factor,
}


TEMPLATES = {
    'M1_return': m1_return,
    'M2_liquidity': m2_liquidity,
    'M3_distribution': m3_distribution,
    'M4_location': m4_location,
    'M5_trend': m5_trend,
    'M6_pvcorr': m6_price_volume,
    'M7_volatility': m7_volatility,
    'M8_gate': m8_gate,
    'M9_composite': m9_composite,
    'M10_shape': m10_shape,
    'M11_regime': m11_regime,
    'M15_gap': m15_gap,
    'M16_interaction': m16_interaction,
}

ALL_TEMPLATES = {**TEMPLATES, **DOC_TEMPLATES}

# 机制轴（8 轴 + 文档轴 + context 本地轴）：配额按轴饱和/已测次数分配，而不是按模板族
AXIS_OF = {
    'M1_return': 'trend_position',
    'M4_location': 'trend_position',
    'M5_trend': 'trend_position',
    'M10_shape': 'shape',
    'M2_liquidity': 'liquidity_size',
    'M3_distribution': 'distribution',
    'M6_pvcorr': 'pv_relation',
    'M7_volatility': 'volatility',
    'M8_gate': 'conditional',
    'M11_regime': 'conditional',
    'M9_composite': 'composite',
    'M12_orderstats': 'order_stats',
    'M13_tech': 'tech',
    'M14_context': 'context',
    'M15_gap': 'trend_position',
    'M16_interaction': 'interaction',
    'M16_ctx_interaction': 'interaction',
    'M17_context_factor': 'context',
}
AXES = tuple(sorted(set(AXIS_OF.values())))


def axis_families(allow_documented: bool = False) -> dict:
    pool = ALL_TEMPLATES if allow_documented else TEMPLATES
    out: dict = {}
    for fam, axis in AXIS_OF.items():
        if fam in pool:
            out.setdefault(axis, []).append(fam)
    return {axis: sorted(fams) for axis, fams in out.items()}


def random_tree(rng: random.Random, family: str | None = None,
                allow_documented: bool = False) -> tuple[str, Node]:
    pool = ALL_TEMPLATES if allow_documented else TEMPLATES
    for _ in range(20):
        fam = family or rng.choice(sorted(pool))
        node = pool[fam](rng)
        ok, _ = validate(node, allow_documented=allow_documented)
        if ok:
            return fam, node
        family = None
    return 'M1_return', ts('TS_MEAN', virtual_node('RET1'), 20)


def _gen_subtree(rng: random.Random, cat: str) -> Node:
    if cat == 'bool':
        return random_condition(rng)
    if rng.random() < 0.55:
        _, node = random_tree(rng)
        return node
    return random_num(rng)


def _composite_legs(node: Node) -> list[Node] | None:
    """M9 复合的腿列表（根为左折叠 ADD 链、叶为 RANK(...)）；非复合返回 None。"""
    leaves = []
    cur = node
    while cur.op == 'ADD':
        leaves.append(cur.args[1])
        cur = cur.args[0]
    leaves.append(cur)
    leaves.reverse()
    if len(leaves) >= 2 and all(l.op == 'RANK' and l.args for l in leaves):
        return leaves
    return None


def _rebuild_adds(legs: list[Node]) -> Node:
    acc = legs[0]
    for leg in legs[1:]:
        acc = Node('ADD', (acc, leg))
    return acc


_THRESHOLD_VALUES = (0.3, 0.5, 0.8)


_MUT_BRANCHES = ('window', 'op', 'subtree', 'exp', 'neg', 'thresh', 'legs', 'field')
_MUT_ROLL = {'window': 0.10, 'op': 0.30, 'subtree': 0.50, 'exp': 0.70,
             'neg': 0.80, 'thresh': 0.87, 'legs': 0.92, 'field': 0.98}


def _mutate_once(node: Node, rng: random.Random, rate: float = 0.35,
                 prefer: dict | None = None,
                 allow_documented: bool = False) -> tuple[Node, str]:
    """mutate 的带编辑类型版本（T6，2026-09-25；mini-AlphaMemo）。

    prefer: {branch: weight}——按权重指派要尝试的编辑分支（分支内守卫不满足时自然
    落到后续分支）；调用方用历史"分支→增益"统计构造 prefer，实现变异自适应。
    组合期类型约束（2026-09-25）：结果若含 unit-mismatch/scale-mix（如 CLOSE−AMOUNT
    的大数吞小项）则重掷（≤2 次），退化表达式出生前拦截——强类型 GP 思想。
    返回 (node, branch)：branch 为本次指派的分支（attempt-level 归因）。
    """
    if prefer:
        w = [max(float(prefer.get(b, 1.0)), 0.0) for b in _MUT_BRANCHES]
        if sum(w) <= 0:
            w = [1.0] * len(_MUT_BRANCHES)
        r = rng.random() * sum(w)
        acc = 0.0
        branch = _MUT_BRANCHES[-1]
        for b, wi in zip(_MUT_BRANCHES, w):
            acc += wi
            if acc >= r:
                branch = b
                break
        roll = _MUT_ROLL[branch]
    else:
        roll = rng.random()
        branch = ('window' if roll < 0.26 else 'op' if roll < 0.44 else
                  'subtree' if roll < 0.66 else 'exp' if roll < 0.76 else
                  'neg' if roll < 0.84 else 'thresh' if roll < 0.90 else
                  'legs' if roll < 0.95 else 'field')
    ts_spots = [(p, n) for p, n in walk(node) if n.op in TS_OPS or n.op in ('CORR', 'COV')]
    if roll < 0.26 and ts_spots:
        p, target = rng.choice(ts_spots)
        new_w = _win(rng)
        cand = replace_at(node, p, replace(target, window=new_w, mp=default_mp(target.op, new_w)))
        ok, _ = validate(cand, allow_documented=allow_documented)
        return (cand if ok else node), branch
    if roll < 0.44 and ts_spots:
        p, target = rng.choice(ts_spots)
        options = [op for op in sorted(TS_OPS)
                   if op in VERIFIED_OPS and op != target.op and ARITY[op] == ARITY[target.op]]
        if options:
            new_op = rng.choice(options)
            cand = replace_at(node, p, replace(target, op=new_op,
                                               mp=default_mp(new_op, target.window or 20)))
            ok, _ = validate(cand, allow_documented=allow_documented)
            return (cand if ok else node), branch
    if roll < 0.66:
        # 子树替换：局部重掷（≤5 次）直到类型干净——ADD/SUB 兄弟位的尺度约束
        # 意味着随机子树有较高坏抽率，重掷让其他腿保留（整只候选不陪葬）
        from lint import type_errors as _te
        spots = list(positions(node))
        if spots:
            for _try in range(5):
                p, _, cat = rng.choice(spots)
                cand = replace_at(node, p, _gen_subtree(rng, cat))
                ok, _ = validate(cand, allow_documented=allow_documented)
                if ok and not _te(cand):
                    return cand, branch
            return node, branch
    if roll < 0.76:
        consts = [p for p, n in walk(node) if n.op in ('SIGNEDPOWER', 'POWER')]
        if consts:
            p = rng.choice(consts)
            cand = replace_at(node, p, replace(get_at(node, p),
                                               value=rng.choice(EXPONENTS)))
            ok, _ = validate(cand, allow_documented=allow_documented)
            return (cand if ok else node), branch
    if roll < 0.84:
        # NEG 翻转（2026-09-24 新增）：拆已有 NEG，或在数值位包一层（方向探索不再依赖重生成）
        negs = [p for p, n in walk(node) if n.op == 'NEG']
        if negs and rng.random() < 0.5:
            p = rng.choice(negs)
            cand = replace_at(node, p, get_at(node, p).args[0])
        else:
            spots = [p for p, _n, cat in positions(node) if cat == 'num']
            if spots:
                p = rng.choice(spots)
                cand = replace_at(node, p, Node('NEG', (get_at(node, p),)))
            else:
                cand = Node('NEG', (node,))
        ok, _ = validate(cand, allow_documented=allow_documented)
        return (cand if ok else node), branch
    if roll < 0.90:
        # 门限微调（2026-09-24 新增）：0.3/0.5/0.8 分位阈值互换（配 0.3/0.8 进常数白名单）
        spots = [(p, n) for p, n in walk(node)
                 if n.op == 'CONST' and float(n.value) in _THRESHOLD_VALUES]
        if spots:
            p, target = rng.choice(spots)
            new_v = rng.choice([t for t in _THRESHOLD_VALUES
                                if t != float(target.value)])
            cand = replace_at(node, p, replace(target, value=new_v))
            ok, _ = validate(cand, allow_documented=allow_documented)
            return (cand if ok else node), branch
    if roll < 0.95:
        # 腿数±1（2026-09-24 新增）：M9 复合加/减一条腿（加腿来自腿银行）
        legs = _composite_legs(node)
        if legs:
            if len(legs) < 4 and rng.random() < 0.5:
                legs = legs + [Node('RANK', (_composite_leg(rng),))]
            elif len(legs) > 2:
                legs.pop(rng.randrange(len(legs)))
            cand = _rebuild_adds(legs)
            ok, _ = validate(cand, allow_documented=allow_documented)
            return (cand if ok else node), branch
    fields = [p for p, n in walk(node) if n.op == 'FIELD']
    if not fields:
        return node, branch
    p = rng.choice(fields)
    # V4.3 Class2：本地 profile 下换字段池含本地/扩展字段（validate 按同旗放行；
    # 默认关闭，行为不变；C2C 上传仍由 formula_uses_local 硬拦）。
    _pool = list(PLATFORM_FIELDS)
    if allow_documented:
        _pool += [f for f in list(LOCAL_ONLY_FIELDS) + list(EXTENDED_ONLY_FIELDS)]
    options = [f for f in _pool if f != get_at(node, p).name]
    cand = replace_at(node, p, field(rng.choice(options)))
    ok, _ = validate(cand, allow_documented=allow_documented)
    return (cand if ok else node), branch


def mutate_with_type(node: Node, rng: random.Random, rate: float = 0.35,
                 prefer: dict | None = None, _depth: int = 0,
                 allow_documented: bool = False) -> tuple[Node, str]:
    """带重掷的包装：结果含 unit-mismatch/scale-mix 时原地重摇（≤2 次），
    退化表达式出生前拦截（强类型 GP / grammar masking 思想）。"""
    from lint import type_errors  # 惰性导入避免循环（lint 依赖 grammar）
    res, branch = _mutate_once(node, rng, rate, prefer,
                               allow_documented=allow_documented)
    if _depth < 3 and res is not node and type_errors(res):
        return mutate_with_type(node, rng, rate, prefer=prefer, _depth=_depth + 1,
                                allow_documented=allow_documented)
    return res, branch


def mutate(node: Node, rng: random.Random, rate: float = 0.35,
           allow_documented: bool = False) -> Node:
    return mutate_with_type(node, rng, rate,
                            allow_documented=allow_documented)[0]


def mutate_window_neighbor(node: Node, rng: random.Random) -> Node:
    """窗口邻域精调（T5，2026-09-25）：随机取一个 TS 位点，窗口移到白名单相邻档（±1）。

    与随机改窗（mutate 的 window 分支，跳任意档）不同：围绕现值局部搜索，供精英
    精化阶段收敛用。无 TS 位点/无邻域时原样返回。"""
    ts_spots = [(p, n) for p, n in walk(node) if n.op in TS_OPS or n.op in ('CORR', 'COV')]
    if not ts_spots:
        return node
    p, target = rng.choice(ts_spots)
    w = int(target.window or 0)
    pool = SHIFT_WINDOWS if target.op in MP_NONE else WINDOWS
    if w not in pool:
        return node
    i = pool.index(w)
    choices = [pool[j] for j in (i - 1, i + 1) if 0 <= j < len(pool)]
    if not choices:
        return node
    new_w = rng.choice(choices)
    cand = replace_at(node, p, replace(target, window=new_w,
                                       mp=default_mp(target.op, new_w)))
    ok, _ = validate(cand)
    return cand if ok else node


def crossover(a: Node, b: Node, rng: random.Random,
              allow_documented: bool = False) -> Node:
    from lint import type_errors  # 惰性导入避免循环
    pa = [(p, c) for p, _, c in positions(a)]
    pb = [(q, d) for q, _, d in positions(b)]
    matches = [(p, q) for p, c in pa for q, d in pb if c == d]
    if not matches:
        return a
    rng.shuffle(matches)
    for p, q in matches[:3]:  # 组合期类型约束：前 3 个兼容位点中取第一个类型干净者
        cand = replace_at(a, p, get_at(b, q))
        ok, _ = validate(cand, allow_documented=allow_documented)
        if ok and not type_errors(cand):
            return cand
    return a


def generate_population(n: int, seed: int, exclude: set[str] | None = None,
                        elites: list[tuple[str, Node]] | None = None,
                        mutation_rate: float = 0.35, max_factor: int = 40,
                        allow_documented: bool = False
                        ) -> list[tuple[str, Node]]:
    rng = random.Random(seed)
    seen = set(exclude or set())
    elites = elites or []
    out: list[tuple[str, Node]] = []
    tries = 0
    limit = max(n, 1) * max_factor
    while len(out) < n and tries < limit:
        tries += 1
        if elites and rng.random() < 0.55:
            family, parent = rng.choice(elites)
            child = mutate(parent, rng, mutation_rate)
            if rng.random() < 0.5:
                _, other = rng.choice(elites)
                child = crossover(child, other, rng,
                                  allow_documented=allow_documented)
        else:
            family, child = random_tree(rng, allow_documented=allow_documented)
        ok, _ = validate(child, allow_documented=allow_documented)
        if not ok:
            continue
        h = expression_hash(child)
        if h in seen:
            continue
        seen.add(h)
        out.append((family, child))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='因子表达式树工具（渲染/校验/抽样）')
    ap.add_argument('--sample', type=int, default=0, help='随机抽样 N 个模板表达式并打印')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--allow-documented', action='store_true',
                    help='允许 DOCUMENTED 算子族（M12，首次上平台需冒烟）')
    args = ap.parse_args()
    if args.sample:
        for family, node in generate_population(args.sample, args.seed,
                                                allow_documented=args.allow_documented):
            print(f'{family:16s} {render(node)}')


if __name__ == '__main__':
    main()
