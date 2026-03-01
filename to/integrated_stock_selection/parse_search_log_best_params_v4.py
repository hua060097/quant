#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse walkforward search log and recover best params.

Supports logs where each backtest segment prints a Top-1 summary like:
  🎯 Top-1 策略回测完成:
     总收益: 76.18%
     年化收益: 2092.18%
     最大回撤: 22.92%
     夏普比率: 5.125

And parameters like:
  [PARAMS] current_params:{...}

Header line:
  [window3-search-iter68-seg3] ...

Scoring:
  seg_score = total_return - k * max_drawdown
  iter_score = 0.6 * median(seg_scores) + 0.4 * min(seg_scores)

Usage (Windows):
  python parse_search_log_best_params_v4.py your.log --k 1.0 --topn 10

Notes:
- Reads file as bytes and decodes with utf-8 (fallback gb18030) line-by-line.
- Does NOT require grep/findstr.
"""
import argparse
import ast
import re
import statistics
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple

HEADER_RE = re.compile(r"\[window(?P<win>\d+)-search-iter(?P<iter>\d+)-seg(?P<seg>\d+)\]")
PARAM_RE  = re.compile(r"\[PARAMS\]\s*current_params\s*:\s*(\{.*\})")
# Allow occasional "current_params:{...}" without [PARAMS]
PARAM_RE2 = re.compile(r"current_params\s*:\s*(\{.*\})")

TOTAL_RE  = re.compile(r"总收益\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*%")
ANN_RE    = re.compile(r"年化收益\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*%")
DD_RE     = re.compile(r"最大回撤\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*%")
SHARPE_RE = re.compile(r"夏普比率\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*$")

TOP1_MARK = re.compile(r"Top-?1|TOP-?1|Top\s*1|策略回测完成")

def decode_line(bline: bytes) -> str:
    # fast path utf-8
    try:
        return bline.decode("utf-8").rstrip("\r\n")
    except Exception:
        return bline.decode("gb18030", errors="ignore").rstrip("\r\n")

def safe_literal_eval_dict(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def seg_score(total_return: Optional[float], max_drawdown: Optional[float], k: float) -> Optional[float]:
    if total_return is None or max_drawdown is None:
        return None
    return float(total_return) - float(k) * float(max_drawdown)

def iter_score(seg_scores, require3: bool=True) -> Optional[float]:
    # seg_scores: dict {1: score, 2: score, 3: score}
    if require3 and not all(s in seg_scores for s in (1,2,3)):
        return None
    vals = [seg_scores.get(s) for s in (1,2,3)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return 0.6 * statistics.median(vals) + 0.4 * min(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to log file")
    ap.add_argument("--k", type=float, default=1.0, help="penalty coefficient for drawdown")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--require3", action="store_true", help="require seg1..3 all present (recommended)")
    args = ap.parse_args()

    # store per iter
    iters: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"params": None, "metrics": defaultdict(dict), "seg_score": {}})

    current_iter: Optional[int] = None
    current_seg: Optional[int] = None

    # We read streaming to handle huge logs.
    with open(args.log, "rb") as f:
        for bline in f:
            line = decode_line(bline)

            mh = HEADER_RE.search(line)
            if mh:
                current_iter = int(mh.group("iter"))
                current_seg = int(mh.group("seg"))
                # ensure keys exist
                _ = iters[current_iter]
                continue

            if current_iter is None or current_seg is None:
                continue

            # params
            mp = PARAM_RE.search(line) or PARAM_RE2.search(line)
            if mp:
                d = safe_literal_eval_dict(mp.group(1))
                if d is not None:
                    iters[current_iter]["params"] = d
                continue

            # metrics: keep latest within segment
            m = TOTAL_RE.search(line)
            if m:
                iters[current_iter]["metrics"][current_seg]["total_return"] = float(m.group(1))
                continue
            m = ANN_RE.search(line)
            if m:
                iters[current_iter]["metrics"][current_seg]["annual_return"] = float(m.group(1))
                continue
            m = DD_RE.search(line)
            if m:
                iters[current_iter]["metrics"][current_seg]["max_drawdown"] = float(m.group(1))
                continue
            m = SHARPE_RE.search(line)
            if m:
                iters[current_iter]["metrics"][current_seg]["sharpe_ratio"] = float(m.group(1))
                continue

    # compute scores
    scored = []
    for it, obj in iters.items():
        seg_scores = {}
        for seg in (1,2,3):
            met = obj["metrics"].get(seg, {})
            s = seg_score(met.get("total_return"), met.get("max_drawdown"), args.k)
            if s is not None:
                seg_scores[seg] = s
        obj["seg_score"] = seg_scores
        sc = iter_score(seg_scores, require3=args.require3)
        if sc is None:
            continue
        scored.append((sc, it, seg_scores, obj["params"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"Parsed iters: {len(iters)}; scored iters: {len(scored)} (k={args.k})")
    for rank, (sc, it, segs, params) in enumerate(scored[:args.topn], start=1):
        print("="*80)
        print(f"Rank {rank}: iter={it} iter_score={sc:.6f} seg_scores={segs}")
        print(f"params={params}")

    if scored:
        best = scored[0]
        print("\nBEST PARAMS:")
        print(best[3])

if __name__ == "__main__":
    main()
