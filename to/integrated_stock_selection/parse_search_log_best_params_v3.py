#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3: Robust parser + diagnostics.
- Parses [windowX-search-iterY-segZ] headers.
- Finds current_params dict (same line or within next 50 lines).
- Finds Top-1 summary metrics within lookahead window.
- If metrics not found, dumps diagnostic snippets to a debug file.

This is designed for Windows logs with mixed encodings and line breaks.
"""
import argparse, ast, re, statistics
from collections import defaultdict

HEADER_RE = re.compile(r"\[window(?P<win>\d+)-search-iter(?P<iter>\d+)-seg(?P<seg>\d+)\]")
PARAM_RE  = re.compile(r"(?:current_params|\[PARAMS\].*?current_params)\s*:\s*(\{.*\})")

# Chinese metric labels (robust separators)
TOTAL_RE  = re.compile(r"总收益\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*%")
ANN_RE    = re.compile(r"年化收益\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*%")
DD_RE     = re.compile(r"最大回撤\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*%")
SHARPE_RE = re.compile(r"夏普比率\s*[:：]\s*([\-]?\d+(?:\.\d+)?)\s*$")

# Fallback markers/regex
TOP1_MARK_RE = re.compile(r"(Top-?1|TOP-?1|Top_?1|Top\s*1)")
PCT_RE = re.compile(r"([\-]?\d+(?:\.\d+)?)\s*%")
NUM_RE = re.compile(r"([\-]?\d+(?:\.\d+)?)")

def _decode_bytes(b: bytes) -> str:
    # try utf-8 then gbk; accept whichever yields fewer replacement chars
    best = None
    best_bad = 10**9
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            s = b.decode(enc, errors="replace")
        except Exception:
            continue
        bad = s.count("\ufffd")
        if bad < best_bad:
            best = s
            best_bad = bad
    if best is None:
        return b.decode("utf-8", errors="ignore")
    return best

def read_lines(path: str):
    data = open(path, "rb").read()
    text = _decode_bytes(data)
    return text.splitlines()

def robust_score(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("-inf")
    return 0.6 * statistics.median(vals) + 0.4 * min(vals)

def parse_log(path: str, lookahead: int, params_lookahead: int, debug_dump: int, debug_file: str):
    lines = read_lines(path)
    iters = defaultdict(lambda: {"params": None, "segs": {}, "fail_snips": []})
    dumped = 0

    for i, line in enumerate(lines):
        mh = HEADER_RE.search(line)
        if not mh:
            continue
        it = int(mh.group("iter"))
        seg = int(mh.group("seg"))

        # params
        params = None
        mp = PARAM_RE.search(line)
        if mp:
            try:
                params = ast.literal_eval(mp.group(1))
            except Exception:
                params = None
        if params is None:
            for j in range(i+1, min(i+1+params_lookahead, len(lines))):
                mp = PARAM_RE.search(lines[j])
                if mp:
                    try:
                        params = ast.literal_eval(mp.group(1))
                    except Exception:
                        params = None
                    break
        if params is not None:
            iters[it]["params"] = params

        # metrics (strict)
        total = ann = dd = sharpe = None
        end = min(i+1+lookahead, len(lines))
        for j in range(i, end):
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

        # fallback 1: find Top1 marker then parse % values order
        if total is None:
            marker = None
            for j in range(i, end):
                if TOP1_MARK_RE.search(lines[j]) or "策略回测完成" in lines[j]:
                    marker = j
                    break
            if marker is not None:
                pct_vals = []
                for j in range(marker, min(marker+60, len(lines))):
                    pct_vals += [float(x) for x in PCT_RE.findall(lines[j])[:5]]
                    if len(pct_vals) >= 3:
                        break
                if pct_vals:
                    total = pct_vals[0]
                    if len(pct_vals) > 1: ann = pct_vals[1]
                    if len(pct_vals) > 2: dd = pct_vals[2]
                # Sharpe fallback: find first standalone number on lines containing '夏普' or 'sharpe'
                if sharpe is None:
                    for j in range(marker, min(marker+80, len(lines))):
                        if ("夏普" in lines[j]) or ("Sharpe" in lines[j]) or ("sharpe" in lines[j]):
                            nm = NUM_RE.search(lines[j])
                            if nm:
                                try:
                                    sharpe = float(nm.group(1))
                                except Exception:
                                    pass
                                break

        iters[it]["segs"][seg] = {
            "total_return": total,
            "annual_return": ann,
            "max_drawdown": dd,
            "sharpe": sharpe,
        }

        # diagnostics dump if total still None
        if total is None and dumped < debug_dump:
            snip = "\n".join(lines[i:min(i+40, len(lines))])
            iters[it]["fail_snips"].append((seg, snip))
            dumped += 1

    # write debug file
    if debug_dump > 0:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("### DEBUG SNIPPETS (first failures)\n\n")
            wrote = 0
            for it, obj in sorted(iters.items()):
                for seg, snip in obj.get("fail_snips", []):
                    f.write("="*80 + "\n")
                    f.write(f"iter={it} seg={seg}\n")
                    f.write(snip + "\n\n")
                    wrote += 1
                    if wrote >= debug_dump:
                        break
                if wrote >= debug_dump:
                    break

    return iters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--metric", default="total_return",
                    choices=["total_return","annual_return","sharpe","neg_max_drawdown"])
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--lookahead", type=int, default=400)
    ap.add_argument("--params_lookahead", type=int, default=60)
    ap.add_argument("--require3", action="store_true")
    ap.add_argument("--debug_dump", type=int, default=20, help="dump first N failures to debug file")
    ap.add_argument("--debug_file", default="parse_debug_snippets.txt")
    args = ap.parse_args()

    iters = parse_log(args.log, args.lookahead, args.params_lookahead, args.debug_dump, args.debug_file)

    scored = []
    for it, obj in iters.items():
        segs = obj["segs"]
        if args.require3 and not all(k in segs for k in (1,2,3)):
            continue
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
    if scored:
        for rank, (sc, it, vals, params) in enumerate(scored[:args.topn], start=1):
            print("="*70)
            print(f"Rank {rank}: iter={it} score={sc:.6f} seg_scores={vals}")
            print("params=", params)
        sc, it, vals, params = scored[0]
        print("\nBEST:")
        print(f"iter={it} score={sc:.6f} seg_scores={vals}")
        print(params)
    else:
        print(f"[WARN] No scored iters. See debug file: {args.debug_file}")

if __name__ == "__main__":
    main()
