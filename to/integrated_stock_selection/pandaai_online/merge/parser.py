"""平台公式字符串 → AST（提案入口/LLM 输出反解析）。

对应 Alpha-GPT Thought Decompiler 的 parse 环节：外部自然语言/LLM/人工产出的公式
必须先变成结构化 AST 才能校验、去重、求值。

支持：
- 8 基础字段 + 虚拟序列 RET1/LOGC
- 算术 + - * /、比较 > >= < <=、逻辑 && ||、一元负号
- 算子目录函数调用（TS/CS/二参/三参；窗口为数字字面量）
- 数字常数（含科学计数法，如 1e-09）
- 大小写不敏感（内部统一大写）

不做白名单/单位校验（那是 lint/validate 的职责），只做语法、算子存在性、元数。
"""
from __future__ import annotations

import re

import _paths  # noqa: F401
from field_catalog import (
    ARITY, CS_OPS, EXTENDED_ONLY_FIELDS, GATE_FIELDS, LOCAL_ONLY_FIELDS, MP_NONE,
    PARAM_OPS, PLATFORM_FIELDS, SHIFT_WINDOWS, TS_OPS, VIRTUAL_NAMES, WINDOWS,
    default_mp,
)
from grammar import Node

_ALIAS = {'CORRELATION': 'CORR', 'COVARIANCE': 'COV'}


class ParseError(ValueError):
    def __init__(self, message: str, pos: int, text: str):
        self.pos = pos
        super().__init__(f'{message} @{pos}: ...{text[max(0, pos - 25):pos + 25]}...')


_TOKEN = re.compile(r"""
    (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>>=|<=|&&|\|\||[+\-*/(),<>])
  | (?P<space>\s+)
""", re.X)

_CMP = {'>': 'GT', '>=': 'GE', '<': 'LT', '<=': 'LE'}


def _tokenize(text: str) -> list[tuple]:
    tokens = []
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() != pos:
            raise ParseError(f'unexpected character {text[pos]!r}', pos, text)
        pos = m.end()
        if m.lastgroup == 'space':
            continue
        tokens.append((m.lastgroup, m.group(), m.start()))
    if pos != len(text):
        raise ParseError(f'unexpected character {text[pos]!r}', pos, text)
    return tokens


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def _peek(self) -> tuple:
        if self.i < len(self.tokens):
            return self.tokens[self.i]
        return None, None, len(self.text)

    def _next(self) -> tuple:
        tok = self._peek()
        self.i += 1
        return tok

    def _expect(self, value: str) -> tuple:
        kind, tok, pos = self._peek()
        if tok != value:
            raise ParseError(f'expected {value!r}', pos, self.text)
        return self._next()

    def parse(self) -> Node:
        node = self.parse_or()
        kind, tok, pos = self._peek()
        if tok is not None:
            raise ParseError(f'unexpected token {tok!r}', pos, self.text)
        return node

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self._peek()[1] == '||':
            self._next()
            left = Node('OR', (left, self.parse_and()))
        return left

    def parse_and(self) -> Node:
        left = self.parse_cmp()
        while self._peek()[1] == '&&':
            self._next()
            left = Node('AND', (left, self.parse_cmp()))
        return left

    def parse_cmp(self) -> Node:
        left = self.parse_add()
        kind, tok, pos = self._peek()
        if tok in _CMP:
            self._next()
            return Node(_CMP[tok], (left, self.parse_add()))
        return left

    def parse_add(self) -> Node:
        left = self.parse_mul()
        while self._peek()[1] in ('+', '-'):
            sym = self._next()[1]
            right = self.parse_mul()
            if sym == '+':
                left = Node('ADD', (left, right))
            elif left.op == 'CONST' and float(left.value) == 0.0:
                left = Node('NEG', (right,))  # 0-(x) 归一化为 NEG，保证 render 往返稳定
            else:
                left = Node('SUB', (left, right))
        return left

    def parse_mul(self) -> Node:
        left = self.parse_unary()
        while self._peek()[1] in ('*', '/'):
            op = '*' if self._next()[1] == '*' else '/'
            right = self.parse_unary()
            left = Node('MUL' if op == '*' else 'DIV', (left, right))
        return left

    def parse_unary(self) -> Node:
        if self._peek()[1] == '-':
            self._next()
            return Node('NEG', (self.parse_unary(),))
        return self.parse_primary()

    def parse_primary(self) -> Node:
        kind, tok, pos = self._next()
        if kind == 'number':
            return Node('CONST', value=float(tok))
        if kind == 'ident':
            if self._peek()[1] == '(':
                return self._parse_call(tok, pos)
            if tok in PLATFORM_FIELDS or tok in VIRTUAL_NAMES \
                    or tok in LOCAL_ONLY_FIELDS or tok in EXTENDED_ONLY_FIELDS \
                    or tok in GATE_FIELDS:
                # LOCAL/EXTENDED（层2/扩展）：本地 parser 可解析（render→parse 往返/榜单重评用）；
                # 上传闸门由 formula_uses_local 在 export/upload 链路独立拦截
                return Node('FIELD', name=tok)
            raise ParseError(f'unknown field {tok!r}', pos, self.text)
        if tok == '(':
            node = self.parse_or()
            self._expect(')')
            return node
        raise ParseError(f'unexpected token {tok!r}', pos, self.text)

    def _parse_call(self, name: str, pos: int) -> Node:
        name = _ALIAS.get(name, name)
        self._expect('(')
        raw_args: list[Node] = []
        if self._peek()[1] != ')':
            while True:
                raw_args.append(self.parse_or())
                if self._peek()[1] == ',':
                    self._next()
                    continue
                break
        self._expect(')')
        return self._build_call(name, raw_args, pos)

    def _build_call(self, name: str, raw_args: list[Node], pos: int) -> Node:
        if name in PARAM_OPS:
            spec = PARAM_OPS[name]
            n_expr = int(spec['expr'])
            defaults = tuple(spec['params'])
            if len(raw_args) < n_expr:
                raise ParseError(f'{name} expects {n_expr} expression(s)', pos, self.text)
            rest = raw_args[n_expr:]
            window = None
            if spec['window']:
                if not rest or rest[0].op != 'CONST' or \
                        float(rest[0].value) != int(rest[0].value):
                    raise ParseError(f'{name} window must be an integer literal', pos, self.text)
                window = int(rest[0].value)
                if window not in WINDOWS:
                    raise ParseError(f'{name} window {window} not in {WINDOWS}', pos, self.text)
                rest = rest[1:]
            if rest:
                if len(rest) != len(defaults) or any(a.op != 'CONST' for a in rest):
                    raise ParseError(
                        f'{name} expects {len(defaults)} numeric param(s)', pos, self.text)
                params = tuple(float(a.value) for a in rest)
            else:
                params = defaults
            return Node(name, tuple(raw_args[:n_expr]), window=window, params=params)
        if name in TS_OPS or name in ('CORR', 'COV'):
            n_expr = ARITY[name]
            if len(raw_args) != n_expr + 1:
                raise ParseError(f'{name} expects {n_expr} expression(s) + window', pos, self.text)
            window_node = raw_args[-1]
            if window_node.op != 'CONST' or float(window_node.value) != int(window_node.value):
                raise ParseError(f'{name} window must be an integer literal', pos, self.text)
            window = int(window_node.value)
            pool = SHIFT_WINDOWS if name in MP_NONE else WINDOWS
            if window not in pool:
                raise ParseError(f'{name} window {window} not in {pool}', pos, self.text)
            return Node(name, tuple(raw_args[:-1]), window=window,
                        mp=default_mp(name, window))
        if name in ('SIGNEDPOWER', 'POWER'):
            if len(raw_args) != 2 or raw_args[1].op != 'CONST':
                raise ParseError(f'{name}(X, p) expects a numeric literal p', pos, self.text)
            return Node(name, (raw_args[0],), value=float(raw_args[1].value))
        if name in CS_OPS or ARITY.get(name) == 1:
            if len(raw_args) != 1:
                raise ParseError(f'{name} expects 1 argument', pos, self.text)
            return Node(name, (raw_args[0],))
        if name in ('MAX', 'MIN', 'MEAN'):
            if len(raw_args) != 2:
                raise ParseError(f'{name}(A,B) expects 2 arguments', pos, self.text)
            return Node(name, tuple(raw_args))
        if name == 'IF':
            if len(raw_args) != 3:
                raise ParseError('IF(cond,a,b) expects 3 arguments', pos, self.text)
            return Node('IF', tuple(raw_args))
        raise ParseError(f'unknown function {name!r}', pos, self.text)


def parse_formula(text: str, upper: bool = True) -> Node:
    """解析平台公式。upper=True 时先统一大写（数字的科学计数法不受影响）。"""
    if not text or not text.strip():
        raise ParseError('empty formula', 0, text or '')
    return _Parser(text.upper() if upper else text).parse()


def main() -> None:
    import argparse
    import sys

    from grammar import expression_hash, render, validate

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='平台公式 → AST 解析器')
    ap.add_argument('formula', nargs='+')
    ap.add_argument('--no-validate', action='store_true')
    args = ap.parse_args()
    for text in args.formula:
        try:
            node = parse_formula(text)
        except ParseError as exc:
            print(f'PARSE-FAIL {text!r}: {exc}')
            raise SystemExit(1)
        ok, msg = (True, '')
        if not args.no_validate:
            ok, msg = validate(node, strict_const=False)
        print(f'{"OK " if ok else "INVALID"} {render(node)}')
        if not ok:
            print(f'    {msg}')
        print(f'    hash={expression_hash(node)}')


if __name__ == '__main__':
    main()
