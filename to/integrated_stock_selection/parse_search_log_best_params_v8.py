#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parse_search_log_best_params_v8.py

Key fix (vs v6/v7):
- Many backtest metric lines are emitted by module loggers (e.g. rqalpha_minute_datasource.strategy_backtest_lead_z)
  and DO NOT carry the [windowX-search-iterY-segZ] prefix.
- The __main__ line with [windowX-search-iterY-segZ] appears AFTER the backtest block, e.g.:
    ... Top-1 summary ...
    ... 风控统计 ...
    __main__ - INFO - [window3-search-iter2-seg2] [OK] 5分钟级别回测完成

So v8 buffers the latest completed "Top-1 策略回测完成" block (including 风控统计) and assigns it to the next
[windowX-search-iterY-segZ] completion line.

Scoring:
- seg_base = total_return - k * max_drawdown
- seg_score = seg_base - lambda_risk * (sell_risk / max(trades,1)) - mu_skip * (buy_skip / max(trades+buy_skip,1))

Iter score across seg1/2/3:
- iter_score = 0.6 * median(seg_scores) + 0.4 * min(seg_scores)

Usage:
  python parse_search_log_best_params_v8.py "xxx.log" --k 1.0 --lambda-risk 20 --mu-skip 10 --topn 10 --require3
"""

from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

# Context line emitted by __main__
RE_CTX = re.compile(r"\[(?P<ctx>window\d+-search-iter(?P<iter>\d+)-seg(?P<seg>\d+))\]")
RE_CTX_OK = re.compile(r"\[(?P<ctx>window\d+-search-iter(?P<iter>\d+)-seg(?P<seg>\d+))\].*(?:回测完成|回测成功完成|5分钟级别回测)")

RE_PARAMS = re.compile(r"\[PARAMS\]\s*current_params:\s*(\{.*\})\s*$")

# Top-1 block lines
RE_TOP1_START = re.compile(r"Top-1\s*策略回测完成")
RE_TOTAL_RETURN = re.compile(r"总收益:\s*([\-]?\d+(?:\.\d+)?)\s*%")
RE_MAX_DD = re.compile(r"最大回撤:\s*([\-]?\d+(?:\.\d+)?)\s*%")
RE_SHARPE = re.compile(r"夏普比率:\s*([\-]?\d+(?:\.\d+)?)")
RE_TRADES = re.compile(r"交易次数:\s*(\d+)")
RE_WINRATE = re.compile(r"胜率:\s*([\-]?\d+(?:\.\d+)?)\s*%")

# Risk stats (your log style)
# e.g. "卖出风控 (持仓，止损阈值: -5.00%): 24次 / 24笔交易"
RE_SELL_RISK = re.compile(r"卖出风控.*?:\s*(\d+)\s*次(?:\s*/\s*(\d+)\s*笔交易)?")
# Optional: other debug lines, count occurrences
RE_RISK_DEBUG = re.compile(r"\[风控统计调试\]\s*(.+?):\s*(\d+)\s*次")
# Buy skip / cannot buy
RE_BUY_SKIP = re.compile(r"(?:被跳过买入|skipped_buy_attempts|买入失败|无法买入)\s*[:=：].*?(\d+)")
RE_BUY_RISK_NONE = re.compile(r"买入风控.*?无触发")

def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        try:
            import ast
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

@dataclass
class BlockMetrics:
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe: Optional[float] = None
    trades: Optional[int] = None
    win_rate: Optional[float] = None
    sell_risk: Optional[int] = None
    buy_skip: Optional[int] = None
    risk_debug: Dict[str, int] = None

    def __post_init__(self):
        if self.risk_debug is None:
            self.risk_debug = {}

@dataclass
class SegmentMetrics:
    window: str
    iter: int
    seg: int
    params: Optional[dict] = None

    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe: Optional[float] = None
    trades: Optional[int] = None
    win_rate: Optional[float] = None
    buy_skip: int = 0
    sell_risk: int = 0
    risk_debug: Dict[str, int] = None

    base_score: Optional[float] = None
    seg_score: Optional[float] = None
    risk_sell_ratio: Optional[float] = None
    buy_skip_ratio: Optional[float] = None

    def __post_init__(self):
        if self.risk_debug is None:
            self.risk_debug = {}

    def compute_scores(self, k: float, lambda_risk: float, mu_skip: float) -> None:
        if self.total_return is None or self.max_drawdown is None:
            return
        base = float(self.total_return) - float(k) * float(self.max_drawdown)
        self.base_score = base

        t = int(self.trades or 0)
        bs = int(self.buy_skip or 0)
        sr = int(self.sell_risk or 0)

        self.risk_sell_ratio = sr / max(t, 1)
        self.buy_skip_ratio = bs / max(t + bs, 1)
        self.seg_score = base - float(lambda_risk) * self.risk_sell_ratio - float(mu_skip) * self.buy_skip_ratio

@dataclass
class IterResult:
    window: str
    iter: int
    params: Optional[dict]
    segs: Dict[int, SegmentMetrics]
    iter_score: Optional[float] = None

    def compute_iter_score(self) -> None:
        scores = [m.seg_score for m in self.segs.values() if m.seg_score is not None]
        if not scores:
            self.iter_score = None
            return
        scores = sorted(scores)
        n = len(scores)
        med = scores[n//2] if n % 2 == 1 else 0.5*(scores[n//2-1] + scores[n//2])
        self.iter_score = 0.6 * med + 0.4 * min(scores)

def _extract_ctx(line: str) -> Optional[Tuple[str,int,int]]:
    m = RE_CTX.search(line)
    if not m:
        return None
    window = m.group("ctx").split("-search-")[0]
    return window, int(m.group("iter")), int(m.group("seg"))

def parse_log(path: Path, k: float, lambda_risk: float, mu_skip: float, require3: bool) -> Dict[Tuple[str,int], IterResult]:
    iters: Dict[Tuple[str,int], IterResult] = {}
    last_params: Dict[Tuple[str,int], dict] = {}

    # Buffer the most recent completed Top1 block (since it has no [window..] prefix)
    in_top1_block = False
    current_block = BlockMetrics()
    pending_block: Optional[BlockMetrics] = None

    def _ensure_iter(window: str, it: int) -> IterResult:
        key = (window, it)
        if key not in iters:
            iters[key] = IterResult(window=window, iter=it, params=last_params.get(key), segs={})
        return iters[key]

    def _ensure_seg(window: str, it: int, seg: int) -> SegmentMetrics:
        ir = _ensure_iter(window, it)
        if seg not in ir.segs:
            ir.segs[seg] = SegmentMetrics(window=window, iter=it, seg=seg, params=ir.params)
        return ir.segs[seg]

    def _block_is_usable(b: BlockMetrics) -> bool:
        return (b.total_return is not None) and (b.max_drawdown is not None)

    def _finalize_current_block():
        nonlocal in_top1_block, current_block, pending_block
        # Only keep if it has core metrics
        if _block_is_usable(current_block):
            pending_block = current_block
        in_top1_block = False
        current_block = BlockMetrics()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Params lines usually have [window..] earlier; bind if available
            mp = RE_PARAMS.search(line)
            if mp:
                params = _safe_json_loads(mp.group(1))
                if params:
                    # try bind by the latest ctx in this line (sometimes the params line has ctx elsewhere)
                    ctx = _extract_ctx(line)
                    if ctx is not None:
                        w,it,seg = ctx
                        key = (w,it)
                        last_params[key] = params
                        ir = _ensure_iter(w,it)
                        ir.params = ir.params or params
                        # also attach to existing seg
                        if seg in ir.segs:
                            ir.segs[seg].params = ir.segs[seg].params or params
                    else:
                        # no ctx: can't bind; skip (rare)
                        pass

            # Detect Top-1 block start (module logger line)
            if RE_TOP1_START.search(line):
                in_top1_block = True
                current_block = BlockMetrics()
                # do not continue; allow same line to also contain metrics (rare)

            # If we are inside a top1 block, extract block metrics
            if in_top1_block:
                m = RE_TOTAL_RETURN.search(line)
                if m: current_block.total_return = float(m.group(1))
                m = RE_MAX_DD.search(line)
                if m: current_block.max_drawdown = float(m.group(1))
                m = RE_SHARPE.search(line)
                if m: current_block.sharpe = float(m.group(1))
                m = RE_TRADES.search(line)
                if m: current_block.trades = int(m.group(1))
                m = RE_WINRATE.search(line)
                if m: current_block.win_rate = float(m.group(1))

                m = RE_SELL_RISK.search(line)
                if m:
                    current_block.sell_risk = int(m.group(1))
                    # if trades absent but " / xx笔交易" exists, fill it
                    if current_block.trades is None and m.group(2):
                        try:
                            current_block.trades = int(m.group(2))
                        except Exception:
                            pass

                m = RE_RISK_DEBUG.search(line)
                if m:
                    name = m.group(1).strip()
                    cnt = int(m.group(2))
                    current_block.risk_debug[name] = cnt

                m = RE_BUY_SKIP.search(line)
                if m:
                    try:
                        current_block.buy_skip = int(m.group(1))
                    except Exception:
                        pass

                # Heuristic end of block: when we hit "Strategy-Based回测完成" OR a new __main__ ctx ok line
                if "Strategy-Based回测完成" in line:
                    _finalize_current_block()
                    continue

            # If we see the __main__ completion line with ctx, bind pending_block to it
            m_ok = RE_CTX_OK.search(line)
            if m_ok:
                ctx = _extract_ctx(line)
                if ctx is not None and pending_block is not None:
                    w,it,seg = ctx
                    ir = _ensure_iter(w,it)
                    # attach params if seen earlier
                    if ir.params is None and (w,it) in last_params:
                        ir.params = last_params[(w,it)]
                    sm = _ensure_seg(w,it,seg)

                    # copy pending block into segment
                    sm.total_return = pending_block.total_return
                    sm.max_drawdown = pending_block.max_drawdown
                    sm.sharpe = pending_block.sharpe
                    sm.trades = pending_block.trades
                    sm.win_rate = pending_block.win_rate
                    sm.sell_risk = int(pending_block.sell_risk or 0)
                    sm.buy_skip = int(pending_block.buy_skip or 0)
                    sm.risk_debug = dict(pending_block.risk_debug or {})

                    # clear pending once assigned
                    pending_block = None

            # If we exit a top1 block but didn't see Strategy-Based line, we still may want to finalize when we hit a new ctx ok line
            if in_top1_block and RE_CTX_OK.search(line):
                _finalize_current_block()

    # compute scores & iter scores
    for ir in iters.values():
        for sm in ir.segs.values():
            sm.compute_scores(k=k, lambda_risk=lambda_risk, mu_skip=mu_skip)
        ir.compute_iter_score()

    if require3:
        for ir in iters.values():
            need = [1,2,3]
            if any((sid not in ir.segs) or (ir.segs[sid].seg_score is None) for sid in need):
                ir.iter_score = None

    return iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=str)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--lambda-risk", type=float, default=20.0, dest="lambda_risk")
    ap.add_argument("--mu-skip", type=float, default=10.0, dest="mu_skip")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--require3", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    path = Path(args.logfile)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    iters = parse_log(path, k=args.k, lambda_risk=args.lambda_risk, mu_skip=args.mu_skip, require3=args.require3)
    scored = [ir for ir in iters.values() if ir.iter_score is not None]
    scored.sort(key=lambda x: x.iter_score, reverse=True)

    print(f"Parsed segments: {sum(len(ir.segs) for ir in iters.values())}; scored iters: {len(scored)}")
    print("="*90)

    for rank, ir in enumerate(scored[:args.topn], start=1):
        print(f"RANK #{rank}: window={ir.window} iter={ir.iter}  iter_score={ir.iter_score:.6f}  (k={args.k}, lambda_risk={args.lambda_risk}, mu_skip={args.mu_skip})")
        for sid in sorted(ir.segs.keys()):
            sm = ir.segs[sid]
            if sm.seg_score is None:
                continue
            print(
                f"  seg{sid}: ret={sm.total_return:.2f}% dd={sm.max_drawdown:.2f}% "
                f"base={sm.base_score:.6f} score={sm.seg_score:.6f} "
                f"sharpe={sm.sharpe if sm.sharpe is not None else 'NA'} "
                f"trades={sm.trades if sm.trades is not None else 'NA'} "
                f"buy_skip={sm.buy_skip} sell_risk={sm.sell_risk} "
                f"risk_ratio={sm.risk_sell_ratio:.3f} skip_ratio={sm.buy_skip_ratio:.3f}"
            )
            # show main risk debug counts if present
            if sm.risk_debug:
                top_dbg = sorted(sm.risk_debug.items(), key=lambda kv: kv[1], reverse=True)[:5]
                dbg_str = ", ".join([f"{k}:{v}" for k,v in top_dbg])
                print(f"       risk_debug_top: {dbg_str}")
        if ir.params is not None:
            print(f"  params: {json.dumps(ir.params, ensure_ascii=False)}")
        print("-"*90)

    if args.out:
        payload: Dict[str, Any] = {
            "k": args.k, "lambda_risk": args.lambda_risk, "mu_skip": args.mu_skip,
            "iters": [
                {
                    "window": ir.window, "iter": ir.iter, "iter_score": ir.iter_score,
                    "params": ir.params,
                    "segs": {sid: asdict(sm) for sid, sm in ir.segs.items()},
                }
                for ir in scored
            ]
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote json to: {args.out}")

if __name__ == "__main__":
    main()
