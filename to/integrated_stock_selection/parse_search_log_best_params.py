#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse search log and compute robust score:
score = 0.6 * median(segment_scores) + 0.4 * min(segment_scores)

It extracts:
- iter and seg from lines like:
  ✅ [window3-search-iter68-seg3] ... current_params:{...}
- segment metrics from block:
  🎯 Top-1 策略回测完成:
     总收益: 76.18%
     年化收益: ...
     最大回撤: ...
     夏普比率: ...

Usage:
  python parse_search_log_best_params.py path/to/log.log --metric total_return --topn 5

By default uses total_return as segment score.
"""
import re, ast, argparse, statistics
from collections import defaultdict

def parse_log(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    iters = defaultdict(lambda: {"params": None, "segs": {}})

    p_head = re.compile(r"\[window(?P<win>\d+)-search-iter(?P<iter>\d+)-seg(?P<seg>\d+)\].*?current_params:(?P<params>\{.*\})")
    p_total = re.compile(r"总收益:\s*([\-]?\d+(?:\.\d+)?)%")
    p_ann = re.compile(r"年化收益:\s*([\-]?\d+(?:\.\d+)?)%")
    p_dd = re.compile(r"最大回撤:\s*([\-]?\d+(?:\.\d+)?)%")
    p_sharpe = re.compile(r"夏普比率:\s*([\-]?\d+(?:\.\d+)?)\s*$")

    i = 0
    while i < len(lines):
        line = lines[i]
        m = p_head.search(line)
        if m:
            it = int(m.group("iter"))
            seg = int(m.group("seg"))
            params_raw = m.group("params")
            try:
                params = ast.literal_eval(params_raw)
            except Exception:
                params = None
            if params is not None:
                iters[it]["params"] = params

            total = ann = dd = sharpe = None
            j_end = min(i + 150, len(lines))
            for j in range(i, j_end):
                if total is None:
                    mt = p_total.search(lines[j])
                    if mt: total = float(mt.group(1))
                if ann is None:
                    ma = p_ann.search(lines[j])
                    if ma: ann = float(ma.group(1))
                if dd is None:
                    md = p_dd.search(lines[j])
                    if md: dd = float(md.group(1))
                if sharpe is None:
                    ms = p_sharpe.search(lines[j])
                    if ms: sharpe = float(ms.group(1))
                if total is not None and dd is not None and sharpe is not None:
                    break

            iters[it]["segs"][seg] = {
                "total_return": total,
                "annual_return": ann,
                "max_drawdown": dd,
                "sharpe": sharpe,
            }
            i = j
        i += 1
    return iters

def robust_score(seg_scores):
    seg_scores = [x for x in seg_scores if x is not None]
    if len(seg_scores) == 0:
        return float("-inf")
    med = statistics.median(seg_scores)
    mn = min(seg_scores)
    return 0.6 * med + 0.4 * mn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to log file")
    ap.add_argument("--metric", default="total_return",
                    choices=["total_return","annual_return","sharpe","neg_max_drawdown"],
                    help="segment score metric; neg_max_drawdown uses -max_drawdown")
    ap.add_argument("--topn", type=int, default=5)
    args = ap.parse_args()

    iters = parse_log(args.log)

    scored = []
    for it, obj in iters.items():
        segs = obj["segs"]
        if len(segs) < 3:
            continue
        scores = []
        for s in (1, 2, 3):
            m = segs.get(s, {})
            if args.metric == "neg_max_drawdown":
                v = m.get("max_drawdown")
                v = None if v is None else -float(v)
            else:
                v = m.get(args.metric)
                v = None if v is None else float(v)
            scores.append(v)
        sc = robust_score(scores)
        scored.append((sc, it, scores, obj["params"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"Parsed iters: {len(iters)}; scored iters: {len(scored)}")
    for rank, (sc, it, scores, params) in enumerate(scored[:args.topn], start=1):
        print("="*60)
        print(f"Rank {rank}: iter={it} score={sc:.6f} seg_scores={scores}")
        print("params=", params)

    if scored:
        sc, it, scores, params = scored[0]
        print("\nBEST:")
        print(f"iter={it} score={sc:.6f} seg_scores={scores}")
        print(params)

if __name__ == "__main__":
    main()
