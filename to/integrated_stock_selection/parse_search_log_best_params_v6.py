#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse walkforward parameter-search logs and rank params by a robust score.

Supports:
- Per-segment metrics extracted from "Top-1 策略回测完成" blocks:
    total_return (%), max_drawdown (%), sharpe_ratio, total_trades,
    skipped_buy_attempts (len), and sell_risk count (if printed).
- Segment score:
    base = total_return - k * max_drawdown
    seg_score = base - lambda_risk * risk_sell_ratio - mu_skip * buy_skip_ratio

Where:
    risk_sell_ratio = sell_risk / max(trades, 1)
    buy_skip_ratio  = buy_skip / max(trades + buy_skip, 1)

- Iter score across 3 val segments:
    iter_score = 0.6 * median(seg_scores) + 0.4 * min(seg_scores)

Usage (Windows):
    python parse_search_log_best_params_v6.py "xxx.log" --k 1.0 --lambda-risk 20 --mu-skip 10 --topn 10

Notes:
- The parser is streaming and does not assume metrics appear close to headers.
- It binds metrics to the current [windowX-search-iterY-segZ] context.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

# -------------------------
# Regex patterns (tolerant)
# -------------------------
RE_CTX = re.compile(r"\[(?P<ctx>window\d+-search-iter(?P<iter>\d+)-seg(?P<seg>\d+))\]")
RE_PARAMS = re.compile(r"\[PARAMS\]\s*current_params:\s*(\{.*\})\s*$")

# Summary lines: allow spaces, fullwidth, etc.
RE_TOTAL_RETURN = re.compile(r"总收益:\s*([\-]?\d+(?:\.\d+)?)\s*%")
RE_MAX_DD = re.compile(r"最大回撤:\s*([\-]?\d+(?:\.\d+)?)\s*%")
RE_SHARPE = re.compile(r"夏普比率:\s*([\-]?\d+(?:\.\d+)?)")
RE_TRADES = re.compile(r"交易次数:\s*(\d+)")
RE_WINRATE = re.compile(r"胜率:\s*([\-]?\d+(?:\.\d+)?)\s*%")

# Optional: some logs print risk sells explicitly
RE_SELL_RISK = re.compile(r"(?:卖出风控|风控卖出|risk[_\s]?sell)\s*[:：]?\s*(\d+)")
# Optional: some logs print skipped buy attempts explicitly
RE_SKIPPED_BUY = re.compile(r"(?:被跳过买入|skipped_buy_attempts)\s*[:：]?\s*(\d+)")


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        # Some logs use single quotes; fall back to ast.literal_eval
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


@dataclass
class SegmentMetrics:
    window: str
    iter: int
    seg: int
    params: Optional[dict] = None

    total_return: Optional[float] = None  # percent
    max_drawdown: Optional[float] = None  # percent
    sharpe: Optional[float] = None
    trades: Optional[int] = None
    win_rate: Optional[float] = None  # percent
    buy_skip: Optional[int] = None
    sell_risk: Optional[int] = None

    # computed
    base_score: Optional[float] = None
    seg_score: Optional[float] = None
    risk_sell_ratio: Optional[float] = None
    buy_skip_ratio: Optional[float] = None

    def is_complete_minimal(self) -> bool:
        return (self.total_return is not None) and (self.max_drawdown is not None)

    def compute_scores(self, k: float, lambda_risk: float, mu_skip: float) -> None:
        if self.total_return is None or self.max_drawdown is None:
            return
        base = float(self.total_return) - float(k) * float(self.max_drawdown)
        self.base_score = base

        t = int(self.trades or 0)
        bs = int(self.buy_skip or 0)
        sr = int(self.sell_risk or 0)

        # ratios
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
        if len(scores) == 0:
            self.iter_score = None
            return
        scores_sorted = sorted(scores)
        # median
        n = len(scores_sorted)
        if n % 2 == 1:
            med = scores_sorted[n // 2]
        else:
            med = 0.5 * (scores_sorted[n // 2 - 1] + scores_sorted[n // 2])
        self.iter_score = 0.6 * med + 0.4 * min(scores_sorted)


def _extract_context(line: str) -> Optional[Tuple[str, int, int]]:
    m = RE_CTX.search(line)
    if not m:
        return None
    window = m.group("ctx").split("-search-")[0]
    it = int(m.group("iter"))
    seg = int(m.group("seg"))
    return window, it, seg


def parse_log(
    log_path: Path,
    k: float,
    lambda_risk: float,
    mu_skip: float,
    require3: bool = True,
    verbose_debug: bool = False
) -> Tuple[Dict[Tuple[str, int], IterResult], List[dict]]:
    """
    Returns:
      iters: {(window, iter): IterResult}
      finals: list of final non-search summaries (if found)
    """
    iters: Dict[Tuple[str, int], IterResult] = {}
    finals: List[dict] = []

    current_ctx: Optional[Tuple[str, int, int]] = None  # window, iter, seg
    current_nonsearch = False  # for final summary blocks

    # For robust detection of "final" summary, treat lines without [windowX-search-iterY-segZ] but containing "Top-1 策略回测完成" as final.
    in_top1_block = False
    top1_block_has_ctx = False

    # For segments: capture metrics until we have minimal complete set; bind to ctx if exists.
    # Also capture params when seen; sometimes params line appears after block.
    last_params_by_iter: Dict[Tuple[str, int], dict] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            ctx = _extract_context(line)
            if ctx is not None:
                current_ctx = ctx
                current_nonsearch = False

            # Params
            mp = RE_PARAMS.search(line)
            if mp:
                params = _safe_json_loads(mp.group(1))
                if params and current_ctx is not None:
                    w, it, sg = current_ctx
                    key = (w, it)
                    last_params_by_iter[key] = params
                    if key not in iters:
                        iters[key] = IterResult(window=w, iter=it, params=params, segs={})
                    else:
                        iters[key].params = iters[key].params or params
                    # also attach to segment if already exists
                    segm = iters[key].segs.get(sg)
                    if segm is not None:
                        segm.params = segm.params or params

            # Detect Top1 summary block start
            if "Top-1" in line and ("策略回测完成" in line or "策略回测完成" in line):
                in_top1_block = True
                top1_block_has_ctx = (current_ctx is not None and RE_CTX.search(line) is not None)
                if ctx is None and RE_CTX.search(line) is None:
                    current_nonsearch = True

            # Parse metrics lines (works regardless of in_top1_block; some logs print metrics without an explicit "Top-1 策略回测完成" line)
            m_ret = RE_TOTAL_RETURN.search(line)
            m_dd = RE_MAX_DD.search(line)
            m_sh = RE_SHARPE.search(line)
            m_tr = RE_TRADES.search(line)
            m_wr = RE_WINRATE.search(line)
            m_sr = RE_SELL_RISK.search(line)
            m_bs = RE_SKIPPED_BUY.search(line)

            if any([m_ret, m_dd, m_sh, m_tr, m_wr, m_sr, m_bs]):
                if current_ctx is not None:
                    w, it, sg = current_ctx
                    key = (w, it)
                    if key not in iters:
                        iters[key] = IterResult(window=w, iter=it, params=last_params_by_iter.get(key), segs={})
                    if sg not in iters[key].segs:
                        iters[key].segs[sg] = SegmentMetrics(window=w, iter=it, seg=sg, params=iters[key].params)
                    sm = iters[key].segs[sg]

                    if m_ret: sm.total_return = float(m_ret.group(1))
                    if m_dd: sm.max_drawdown = float(m_dd.group(1))
                    if m_sh: sm.sharpe = float(m_sh.group(1))
                    if m_tr: sm.trades = int(m_tr.group(1))
                    if m_wr: sm.win_rate = float(m_wr.group(1))
                    if m_sr: sm.sell_risk = int(m_sr.group(1))
                    if m_bs: sm.buy_skip = int(m_bs.group(1))
                else:
                    # no ctx -> treat as final summary candidate if it looks like a backtest summary
                    # We'll accumulate into last finals entry or start new when ret+dd appear.
                    if m_ret or m_dd:
                        if not finals or finals[-1].get("_closed", False):
                            finals.append({"_closed": False})
                        fs = finals[-1]
                        if m_ret: fs["total_return"] = float(m_ret.group(1))
                        if m_dd: fs["max_drawdown"] = float(m_dd.group(1))
                        if m_sh: fs["sharpe_ratio"] = float(m_sh.group(1))
                        if m_tr: fs["total_trades"] = int(m_tr.group(1))
                        if m_bs: fs["buy_skip"] = int(m_bs.group(1))
                        if m_sr: fs["sell_risk"] = int(m_sr.group(1))

            # Heuristic: close "final" summary when a separator line appears
            if current_ctx is None and finals and (line.startswith("====") or line.startswith("---")):
                finals[-1]["_closed"] = True

    # compute scores
    for it_res in iters.values():
        for sm in it_res.segs.values():
            sm.compute_scores(k=k, lambda_risk=lambda_risk, mu_skip=mu_skip)
        it_res.compute_iter_score()

    # filter require3
    if require3:
        for key in list(iters.keys()):
            segs = iters[key].segs
            # Expect seg ids 1..3 (val segments)
            have = [s for s in (1,2,3) if s in segs and segs[s].seg_score is not None]
            if len(have) < 3:
                # drop incomplete
                iters[key].iter_score = None

    # clean finals
    for fs in finals:
        fs.pop("_closed", None)
        if "total_return" in fs and "max_drawdown" in fs:
            fs["score"] = fs["total_return"] - k * fs["max_drawdown"]

    finals = [fs for fs in finals if ("total_return" in fs and "max_drawdown" in fs)]

    return iters, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=str, help="Path to log file")
    ap.add_argument("--k", type=float, default=1.0, help="drawdown penalty coefficient")
    ap.add_argument("--lambda-risk", type=float, default=20.0, dest="lambda_risk",
                    help="penalty weight for risk sell ratio (sell_risk/trades)")
    ap.add_argument("--mu-skip", type=float, default=10.0, dest="mu_skip",
                    help="penalty weight for buy-skip ratio (buy_skip/(trades+buy_skip))")
    ap.add_argument("--topn", type=int, default=10, help="print top N")
    ap.add_argument("--require3", action="store_true", help="require seg1/2/3 all present for scoring")
    ap.add_argument("--out", type=str, default="", help="optional json output path")
    args = ap.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        raise SystemExit(f"File not found: {log_path}")

    iters, finals = parse_log(
        log_path, k=args.k, lambda_risk=args.lambda_risk, mu_skip=args.mu_skip,
        require3=args.require3
    )

    scored = [ir for ir in iters.values() if ir.iter_score is not None]
    scored.sort(key=lambda x: x.iter_score, reverse=True)

    print(f"Parsed segments: {sum(len(ir.segs) for ir in iters.values())}; scored iters: {len(scored)}")
    print(f"Found final Top-1 summaries (non-search): {len(finals)}")
    if finals:
        last = finals[-1]
        print(f"Last final summary: total_return={last.get('total_return'):.2f}%, max_dd={last.get('max_drawdown'):.2f}%, score={last.get('score'):.4f}")

    print("=" * 90)
    for rank, ir in enumerate(scored[: args.topn], start=1):
        print(f"RANK #{rank}: window={ir.window} iter={ir.iter}  iter_score={ir.iter_score:.6f}  (k={args.k}, lambda_risk={args.lambda_risk}, mu_skip={args.mu_skip})")
        for seg_id in sorted(ir.segs.keys()):
            sm = ir.segs[seg_id]
            if sm.seg_score is None:
                continue
            print(
                f"  seg{seg_id}: ret={sm.total_return:.2f}% dd={sm.max_drawdown:.2f}% "
                f"base={sm.base_score:.6f} score={sm.seg_score:.6f} "
                f"sharpe={sm.sharpe if sm.sharpe is not None else 'NA'} "
                f"trades={sm.trades if sm.trades is not None else 'NA'} "
                f"buy_skip={sm.buy_skip if sm.buy_skip is not None else 0} "
                f"sell_risk={sm.sell_risk if sm.sell_risk is not None else 0} "
                f"risk_ratio={sm.risk_sell_ratio:.3f} skip_ratio={sm.buy_skip_ratio:.3f}"
            )
        # params
        if ir.params is not None:
            print(f"  params: {json.dumps(ir.params, ensure_ascii=False)}")
        print("-" * 90)

    if args.out:
        out_path = Path(args.out)
        payload: Dict[str, Any] = {
            "k": args.k,
            "lambda_risk": args.lambda_risk,
            "mu_skip": args.mu_skip,
            "finals": finals,
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
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote json to: {out_path}")


if __name__ == "__main__":
    main()
