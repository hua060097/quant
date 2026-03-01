#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust log parser for Windows + mixed encodings.

It extracts iter/seg blocks from lines containing:
  [window3-search-iter68-seg3]

Then looks forward (within a window) for:
- current_params:{...}  (may be on same line or next lines)
- Top-1 backtest summary metrics:
    总收益: xx.xx%
    年化收益: xx.xx%
    最大回撤: xx.xx%
    夏普比率: x.xxx

If Chinese text is garbled in console, this script still parses the file bytes directly.
It tries utf-8 first, then gbk.

Score:
  score = 0.6 * median(segment_scores) + 0.4 * min(segment_scores)

Usage (Windows):
  python parse_search_log_best_params_v2.py your.log --metric total_return --topn 10

Options:
  --metric total_return | annual_return | sharpe | neg_max_drawdown
  --require3 require seg1..3 all present (default true)
"""
import argparse, ast, re, statistics
from collections import defaultdict

HEADER_RE = re.compile(r"\[window(?P<win>\d+)-search-iter(?P<iter>\d+)-seg(?P<seg>\d+)\]")
PARAM_RE = re.compile(r"(?:current_params|\\[PARAMS\\].*?current_params)\\s*:\\s*(\\{.*\\})")
# Metric lines (Chinese). If encoding is wrong, these may not match; we also have a fallback percent scanner.
TOTAL_RE = re.compile(r"总收益\\s*[:：]\\s*([\\-]?\\d+(?:\\.\\d+)?)%")
ANN_RE   = re.compile(r"年化收益\\s*[:：]\\s*([\\-]?\\d+(?:\\.\\d+)?)%")
DD_RE    = re.compile(r"最大回撤\\s*[:：]\\s*([\\-]?\\d+(?:\\.\\d+)?)%")
SHARPE_RE= re.compile(r"夏普比率\\s*[:：]\\s*([\\-]?\\d+(?:\\.\\d+)?)\\s*$")

PCT_RE = re.compile(r"([\\-]?\\d+(?:\\.\\d+)?)%")  # fallback: first few %s after 'Top-1' block

def _decode_bytes(b: bytes) -> str:
    # try utf-8 then gbk
    try:
        s = b.decode("utf-8")
        # heuristic: if too many U+FFFD replacements, treat as failed
        if s.count("\ufffd") > 10:
            raise UnicodeDecodeError("utf-8", b, 0, 1, "too many replacement chars")
        return s
    except Exception:
        try:
            return b.decode("gbk", errors="ignore")
        except Exception:
            return b.decode("utf-8", errors="ignore")

def read_lines(path: str):
    with open(path, "rb") as f:
        data = f.read()
    text = _decode_bytes(data)
    # keep line breaks
    return text.splitlines()

def robust_score(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("-inf")
    med = statistics.median(vals)
    mn = min(vals)
    return 0.6 * med + 0.4 * mn

def parse_log(path: str, lookahead: int = 200):
    lines = read_lines(path)
    iters = defaultdict(lambda: {"params": None, "segs": {}})

    i = 0
    while i < len(lines):
        line = lines[i]
        mh = HEADER_RE.search(line)
        if not mh:
            i += 1
            continue

        it = int(mh.group("iter"))
        seg = int(mh.group("seg"))

        # ---- find params (same line or subsequent lines)
        params = None
        mp = PARAM_RE.search(line)
        if mp:
            raw = mp.group(1)
            try:
                params = ast.literal_eval(raw)
            except Exception:
                params = None
        if params is None:
            for j in range(i+1, min(i+30, len(lines))):
                mp = PARAM_RE.search(lines[j])
                if mp:
                    raw = mp.group(1)
                    try:
                        params = ast.literal_eval(raw)
                    except Exception:
                        params = None
                    break
        if params is not None:
            iters[it]["params"] = params

        # ---- find metrics block in lookahead window
        total = ann = dd = sharpe = None
        # first try strict Chinese labels
        for j in range(i, min(i+lookahead, len(lines))):
            lj = lines[j]
            if total is None:
                m = TOTAL_RE.search(lj)
                if m: total = float(m.group(1))
            if ann is None:
                m = ANN_RE.search(lj)
                if m: ann = float(m.group(1))
            if dd is None:
                m = DD_RE.search(lj)
                if m: dd = float(m.group(1))
            if sharpe is None:
                m = SHARPE_RE.search(lj)
                if m: sharpe = float(m.group(1))
            if total is not None and dd is not None and sharpe is not None:
                break

        # fallback: if Chinese labels not found, try to infer from nearby % numbers after "Top-1" marker
        if total is None:
            # find a marker line
            marker_idx = None
            for j in range(i, min(i+lookahead, len(lines))):
                if "Top-1" in lines[j] or "Top1" in lines[j] or "TOP1" in lines[j]:
                    marker_idx = j
                    break
            if marker_idx is not None:
                pct_vals = []
                for j in range(marker_idx, min(marker_idx+40, len(lines))):
                    pct_vals += [float(x) for x in PCT_RE.findall(lines[j])[:3]]
                    if len(pct_vals) >= 3:
                        break
                # heuristic: first % is total_return, second is annual_return, third is max_drawdown
                if pct_vals:
                    total = pct_vals[0]
                    if len(pct_vals) > 1: ann = pct_vals[1]
                    if len(pct_vals) > 2: dd = pct_vals[2]

        iters[it]["segs"][seg] = {
            "total_return": total,
            "annual_return": ann,
            "max_drawdown": dd,
            "sharpe": sharpe,
        }

        i += 1

    return iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to log file")
    ap.add_argument("--metric", default="total_return",
                    choices=["total_return","annual_return","sharpe","neg_max_drawdown"])
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--lookahead", type=int, default=220)
    ap.add_argument("--require3", action="store_true", help="require seg1..3 all present")
    args = ap.parse_args()

    iters = parse_log(args.log, lookahead=args.lookahead)

    scored = []
    for it, obj in iters.items():
        segs = obj["segs"]
        if args.require3 and not all(k in segs for k in (1,2,3)):
            continue
        # collect available segs 1..3
        vals = []
        for s in (1,2,3):
            m = segs.get(s, {})
            if args.metric == "neg_max_drawdown":
                v = m.get("max_drawdown")
                v = None if v is None else -float(v)
            else:
                v = m.get(args.metric)
                v = None if v is None else float(v)
            vals.append(v)

        sc = robust_score(vals)
        if sc == float("-inf"):
            continue
        scored.append((sc, it, vals, obj["params"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"Parsed iters: {len(iters)}; scored iters: {len(scored)}")
    for rank, (sc, it, vals, params) in enumerate(scored[:args.topn], start=1):
        print("="*70)
        print(f"Rank {rank}: iter={it} score={sc:.6f} seg_scores={vals}")
        print("params=", params)

    if scored:
        sc, it, vals, params = scored[0]
        print("\nBEST:")
        print(f"iter={it} score={sc:.6f} seg_scores={vals}")
        print(params)

if __name__ == "__main__":
    main()
