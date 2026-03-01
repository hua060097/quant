#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse walkforward parameter-search logs and rank candidates by a robust score.

What it extracts (best-effort):
- per (window, iter, seg) current_params
- Top-1 backtest summary: total_return, max_drawdown, sharpe_ratio, total_trades, win_rate
- risk stats: buy_skip_total, sell_risk_total, skipped_buy_attempts (if printed)
- optional "final test" Top-1 summary (outside search-iter segments)

Scoring:
- segment_score = total_return - k * max_drawdown   (both in percentage points)
- iter_score (default) = 0.6 * median(segment_scores) + 0.4 * min(segment_scores)

Usage:
  python parse_search_log_best_params_v5.py <logfile> --k 1.0 --topn 10
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


SEG_TAG_RE = re.compile(r"\[(?P<wname>window\d+)-search-iter(?P<iter>\d+)-seg(?P<seg>\d+)\]")
PARAMS_RE = re.compile(r"\[PARAMS\]\s*current_params:(\{.*\})\s*$")
TOP1_HEAD_RE = re.compile(r"Top-1\s*策略回测完成|🎯\s*Top-1\s*策略回测完成")
# Examples:
#    总收益: 108.29%
#    最大回撤: 16.05%
#    夏普比率: 5.960
RET_RE = re.compile(r"总收益:\s*([-\d\.]+)%")
DD_RE = re.compile(r"最大回撤:\s*([-\d\.]+)%")
SHARPE_RE = re.compile(r"夏普比率:\s*([-\d\.]+)")
TRADES_RE = re.compile(r"交易次数:\s*(\d+)")
WIN_RE = re.compile(r"胜率:\s*([-\d\.]+)%")

# Risk stats (your logs show these)
BUY_SKIP_RE = re.compile(r"买入风控 \(开盘\):\s*(\d+)次")
SELL_RISK_RE = re.compile(r"卖出风控 .*?:\s*(\d+)次\s*/\s*(\d+)笔交易")
# sometimes: "🛡️ 总计风控保护: 6次买入被跳过"
BUY_SKIP2_RE = re.compile(r"总计风控保护:\s*(\d+)次买入被跳过")

# Some logs print skipped_buy_attempts length
SKIPPED_BUY_ATTEMPTS_RE = re.compile(r"skipped_buy_attempts.*长度:\s*(\d+)")


def _decode_bytes(b: bytes) -> str:
    # prefer utf-8, then gbk/gb18030; choose with least replacement chars
    best = None
    best_bad = 10**18
    for enc in ("utf-8", "gbk", "gb18030"):
        s = b.decode(enc, errors="replace")
        bad = s.count("\ufffd")
        if bad < best_bad:
            best = s
            best_bad = bad
            if bad == 0:
                break
    return best or b.decode("utf-8", errors="replace")


def _safe_parse_params(s: str) -> Dict[str, Any]:
    s = s.strip()
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # fallback: try json
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": s}


@dataclass
class SegmentResult:
    window: str
    iter: int
    seg: int
    params: Optional[Dict[str, Any]] = None

    total_return: Optional[float] = None   # percentage points
    max_drawdown: Optional[float] = None   # percentage points
    sharpe: Optional[float] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None       # percentage points

    buy_skip: Optional[int] = None
    sell_risk_hits: Optional[int] = None
    sell_risk_total_trades: Optional[int] = None
    skipped_buy_attempts: Optional[int] = None

    def is_complete(self) -> bool:
        return (self.total_return is not None) and (self.max_drawdown is not None)

    def segment_score(self, k: float) -> Optional[float]:
        if self.total_return is None or self.max_drawdown is None:
            return None
        return float(self.total_return) - float(k) * float(self.max_drawdown)


def parse_log(path: Path) -> Tuple[Dict[Tuple[str,int,int], SegmentResult], List[Dict[str, Any]]]:
    b = path.read_bytes()
    s = _decode_bytes(b)
    lines = s.splitlines()

    seg_results: Dict[Tuple[str,int,int], SegmentResult] = {}
    final_summaries: List[Dict[str, Any]] = []

    # state machine: current segment tag (window,iter,seg) if in search
    cur_key: Optional[Tuple[str,int,int]] = None
    waiting_top1 = False
    # Top1 block maybe belongs to current seg or final
    top1_target_key: Optional[Tuple[str,int,int]] = None
    top1_tmp: Dict[str, Any] = {}

    def _ensure_seg(key: Tuple[str,int,int]) -> SegmentResult:
        if key not in seg_results:
            seg_results[key] = SegmentResult(window=key[0], iter=key[1], seg=key[2])
        return seg_results[key]

    def _flush_top1():
        nonlocal waiting_top1, top1_target_key, top1_tmp
        if not top1_tmp:
            waiting_top1 = False
            top1_target_key = None
            return
        if top1_target_key is None:
            final_summaries.append(dict(top1_tmp))
        else:
            sr = _ensure_seg(top1_target_key)
            for k,v in top1_tmp.items():
                if hasattr(sr, k):
                    setattr(sr, k, v)
        waiting_top1 = False
        top1_target_key = None
        top1_tmp = {}

    for ln in lines:
        # segment tag
        m = SEG_TAG_RE.search(ln)
        if m:
            # entering a segment context; flush any previous top1
            _flush_top1()
            cur_key = (m.group("wname"), int(m.group("iter")), int(m.group("seg")))
            _ensure_seg(cur_key)

        # reset segment context when leaving search-eval blocks
        if "参数搜索完成" in ln or "[SEARCH STATS]" in ln or re.search(r"\[Window\s*\d+\]", ln):
            cur_key = None
            continue

        # params line
        m = PARAMS_RE.search(ln)
        if m:
            if cur_key is not None:
                _ensure_seg(cur_key).params = _safe_parse_params(m.group(1))
            continue

        # top1 head
        if TOP1_HEAD_RE.search(ln):
            _flush_top1()
            waiting_top1 = True
            top1_target_key = cur_key  # if inside segment; else None
            top1_tmp = {}
            continue

        if waiting_top1:
            # extract metrics
            m = RET_RE.search(ln)
            if m:
                top1_tmp["total_return"] = float(m.group(1))
                continue
            m = DD_RE.search(ln)
            if m:
                top1_tmp["max_drawdown"] = float(m.group(1))
                continue
            m = SHARPE_RE.search(ln)
            if m:
                top1_tmp["sharpe"] = float(m.group(1))
                continue
            m = TRADES_RE.search(ln)
            if m:
                top1_tmp["total_trades"] = int(m.group(1))
                continue
            m = WIN_RE.search(ln)
            if m:
                top1_tmp["win_rate"] = float(m.group(1))
                continue
            # risk stats inside top1 block
            m = BUY_SKIP_RE.search(ln)
            if m:
                top1_tmp["buy_skip"] = int(m.group(1))
                continue
            m = BUY_SKIP2_RE.search(ln)
            if m:
                top1_tmp["buy_skip"] = int(m.group(1))
                continue
            m = SELL_RISK_RE.search(ln)
            if m:
                top1_tmp["sell_risk_hits"] = int(m.group(1))
                top1_tmp["sell_risk_total_trades"] = int(m.group(2))
                continue
            m = SKIPPED_BUY_ATTEMPTS_RE.search(ln)
            if m:
                top1_tmp["skipped_buy_attempts"] = int(m.group(1))
                continue

            # heuristic end of top1 block: we saw "Strategy-Based回测完成" or a new segment tag line
            if "Strategy-Based回测完成" in ln or "✅ Strategy-Based回测完成" in ln:
                _flush_top1()
                continue

        else:
            # outside Top1 block, still may have skipped_buy_attempts debug lines tied to seg
            m = SKIPPED_BUY_ATTEMPTS_RE.search(ln)
            if m and cur_key is not None:
                _ensure_seg(cur_key).skipped_buy_attempts = int(m.group(1))
                continue

    _flush_top1()
    return seg_results, final_summaries


def score_iters(seg_results: Dict[Tuple[str,int,int], SegmentResult], k: float) -> List[Dict[str, Any]]:
    # group by (window, iter)
    grouped: Dict[Tuple[str,int], List[SegmentResult]] = {}
    for (w, it, sg), sr in seg_results.items():
        grouped.setdefault((w,it), []).append(sr)

    ranked: List[Dict[str, Any]] = []
    for (w,it), segs in grouped.items():
        segs_sorted = sorted(segs, key=lambda x: x.seg)
        seg_scores = [s.segment_score(k) for s in segs_sorted if s.segment_score(k) is not None]
        if len(seg_scores) == 0:
            continue
        agg = 0.6 * median(seg_scores) + 0.4 * min(seg_scores)
        ranked.append({
            "window": w,
            "iter": it,
            "iter_score": agg,
            "k": k,
            "segment_scores": seg_scores,
            "segments": [asdict(s) for s in segs_sorted],
            "params": (segs_sorted[0].params if segs_sorted and segs_sorted[0].params else None),
        })
    ranked.sort(key=lambda d: d["iter_score"], reverse=True)
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=str)
    ap.add_argument("--k", type=float, default=1.0, help="score = total_return - k*max_drawdown (percentage points)")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--out", type=str, default="", help="optional path to write json results")
    args = ap.parse_args()

    path = Path(args.logfile)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    seg_results, final_summaries = parse_log(path)
    ranked = score_iters(seg_results, k=args.k)

    print(f"Parsed segments: {len(seg_results)}; scored iters: {len(ranked)}")
    if len(final_summaries) > 0:
        print(f"Found final Top-1 summaries (non-search): {len(final_summaries)}")
        # show last one briefly
        last = final_summaries[-1]
        if "total_return" in last and "max_drawdown" in last:
            s = float(last["total_return"]) - args.k * float(last["max_drawdown"])
            print(f"Last final summary: total_return={last.get('total_return')}%, max_dd={last.get('max_drawdown')}%, score={s:.4f}")

    for i, row in enumerate(ranked[: max(0, args.topn)]):
        print("=" * 90)
        print(f"RANK #{i+1}: window={row['window']} iter={row['iter']}  iter_score={row['iter_score']:.6f}  (k={args.k})")
        if row.get("params") is not None:
            print(f"params: {row['params']}")
        segs = row["segments"]
        for seg in segs:
            tr = seg.get("total_return")
            dd = seg.get("max_drawdown")
            sh = seg.get("sharpe")
            td = seg.get("total_trades")
            bs = seg.get("buy_skip")
            srh = seg.get("sell_risk_hits")
            skp = seg.get("skipped_buy_attempts")
            sc = None
            if tr is not None and dd is not None:
                sc = float(tr) - args.k * float(dd)
            print(f"  seg{seg['seg']}: ret={tr}% dd={dd}% score={sc} sharpe={sh} trades={td} buy_skip={bs} sell_risk={srh} skipped_buy_attempts={skp}")

    if args.out:
        outp = Path(args.out)
        payload = {
            "k": args.k,
            "ranked": ranked,
            "final_summaries": final_summaries,
        }
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote: {outp}")

if __name__ == "__main__":
    main()
