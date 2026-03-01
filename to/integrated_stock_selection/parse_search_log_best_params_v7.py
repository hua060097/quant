#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v7: Parse walkforward parameter-search logs and rank params by a robust score.

Fix vs v6:
- Extract buy-skip count from multiple common log styles:
  1) "被跳过买入=6" (often printed in summary lines)
  2) "skipped_buy_attempts存在: True ... 长度: 6" (debug prints)
  3) "skipped_buy_attempts: ... 长度: 6"
- Keep v6 penalties:
    seg_score = (total_return - k*max_drawdown) - lambda_risk*risk_sell_ratio - mu_skip*buy_skip_ratio
  but note: if sell_risk cannot be found in log, risk_sell_ratio=0.

Usage:
  python parse_search_log_best_params_v7.py "xxx.log" --k 1.0 --lambda-risk 20 --mu-skip 10 --topn 10 --require3
"""

from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

RE_CTX = re.compile(r"\[(?P<ctx>window\d+-search-iter(?P<iter>\d+)-seg(?P<seg>\d+))\]")
RE_PARAMS = re.compile(r"\[PARAMS\]\s*current_params:\s*(\{.*\})\s*$")

RE_TOTAL_RETURN = re.compile(r"总收益:\s*([\-]?\d+(?:\.\d+)?)\s*%")
RE_MAX_DD = re.compile(r"最大回撤:\s*([\-]?\d+(?:\.\d+)?)\s*%")
RE_SHARPE = re.compile(r"夏普比率:\s*([\-]?\d+(?:\.\d+)?)")
RE_TRADES = re.compile(r"交易次数:\s*(\d+)")
RE_WINRATE = re.compile(r"胜率:\s*([\-]?\d+(?:\.\d+)?)\s*%")

# sell_risk (may or may not exist)
RE_SELL_RISK = re.compile(r"(?:卖出风控|风控卖出|sell_risk)\s*[:=：]?\s*(\d+)")

# buy_skip styles
RE_BUY_SKIP_1 = re.compile(r"(?:被跳过买入|buy_skip)\s*[:=：]\s*(\d+)")
RE_BUY_SKIP_2 = re.compile(r"skipped_buy_attempts.*?长度[:：]\s*(\d+)")
RE_BUY_SKIP_3 = re.compile(r"被跳过买入\s*=\s*(\d+)")  # in summary inline

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
    buy_skip: Optional[int] = None
    sell_risk: Optional[int] = None

    base_score: Optional[float] = None
    seg_score: Optional[float] = None
    risk_sell_ratio: Optional[float] = None
    buy_skip_ratio: Optional[float] = None

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

def _extract_context(line: str) -> Optional[Tuple[str,int,int]]:
    m = RE_CTX.search(line)
    if not m:
        return None
    window = m.group("ctx").split("-search-")[0]
    return window, int(m.group("iter")), int(m.group("seg"))

def parse_log(path: Path, k: float, lambda_risk: float, mu_skip: float, require3: bool) -> Tuple[Dict[Tuple[str,int], IterResult], List[dict]]:
    iters: Dict[Tuple[str,int], IterResult] = {}
    finals: List[dict] = []
    current_ctx: Optional[Tuple[str,int,int]] = None
    last_params: Dict[Tuple[str,int], dict] = {}

    def _get_seg(window: str, it: int, seg: int) -> SegmentMetrics:
        key = (window, it)
        if key not in iters:
            iters[key] = IterResult(window=window, iter=it, params=last_params.get(key), segs={})
        if seg not in iters[key].segs:
            iters[key].segs[seg] = SegmentMetrics(window=window, iter=it, seg=seg, params=iters[key].params)
        return iters[key].segs[seg]

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            ctx = _extract_context(line)
            if ctx is not None:
                current_ctx = ctx

            mp = RE_PARAMS.search(line)
            if mp and current_ctx is not None:
                params = _safe_json_loads(mp.group(1))
                if params:
                    w,it,seg = current_ctx
                    key = (w,it)
                    last_params[key] = params
                    if key not in iters:
                        iters[key] = IterResult(window=w, iter=it, params=params, segs={})
                    else:
                        iters[key].params = iters[key].params or params
                    sm = iters[key].segs.get(seg)
                    if sm is not None:
                        sm.params = sm.params or params

            # metrics extraction
            if current_ctx is not None:
                w,it,seg = current_ctx
                sm = _get_seg(w,it,seg)

                m = RE_TOTAL_RETURN.search(line)
                if m: sm.total_return = float(m.group(1))
                m = RE_MAX_DD.search(line)
                if m: sm.max_drawdown = float(m.group(1))
                m = RE_SHARPE.search(line)
                if m: sm.sharpe = float(m.group(1))
                m = RE_TRADES.search(line)
                if m: sm.trades = int(m.group(1))
                m = RE_WINRATE.search(line)
                if m: sm.win_rate = float(m.group(1))
                m = RE_SELL_RISK.search(line)
                if m: sm.sell_risk = int(m.group(1))

                # buy_skip multi-style
                m = RE_BUY_SKIP_1.search(line) or RE_BUY_SKIP_2.search(line) or RE_BUY_SKIP_3.search(line)
                if m:
                    try:
                        sm.buy_skip = int(m.group(1))
                    except Exception:
                        pass

    # compute scores
    for ir in iters.values():
        for sm in ir.segs.values():
            sm.compute_scores(k=k, lambda_risk=lambda_risk, mu_skip=mu_skip)
        ir.compute_iter_score()

    if require3:
        for ir in iters.values():
            need = [1,2,3]
            if any((sid not in ir.segs) or (ir.segs[sid].seg_score is None) for sid in need):
                ir.iter_score = None

    return iters, finals

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

    iters, finals = parse_log(path, k=args.k, lambda_risk=args.lambda_risk, mu_skip=args.mu_skip, require3=args.require3)
    scored = [ir for ir in iters.values() if ir.iter_score is not None]
    scored.sort(key=lambda x: x.iter_score, reverse=True)

    print(f"Parsed segments: {sum(len(ir.segs) for ir in iters.values())}; scored iters: {len(scored)}")
    print(f"Found final Top-1 summaries (non-search): {len(finals)}")
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
                f"buy_skip={sm.buy_skip if sm.buy_skip is not None else 0} "
                f"sell_risk={sm.sell_risk if sm.sell_risk is not None else 0} "
                f"risk_ratio={sm.risk_sell_ratio:.3f} skip_ratio={sm.buy_skip_ratio:.3f}"
            )
        if ir.params is not None:
            print(f"  params: {json.dumps(ir.params, ensure_ascii=False)}")
        print("-"*90)

    if args.out:
        payload: Dict[str, Any] = {
            "k": args.k,
            "lambda_risk": args.lambda_risk,
            "mu_skip": args.mu_skip,
            "iters": [
                {
                    "window": ir.window,
                    "iter": ir.iter,
                    "iter_score": ir.iter_score,
                    "params": ir.params,
                    "segs": {sid: asdict(sm) for sid, sm in ir.segs.items()},
                }
                for ir in scored
            ],
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote json to: {args.out}")

if __name__ == "__main__":
    main()
