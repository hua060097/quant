"""Rank1 尾部专项（vwap 口径）：对每个候选因子，模拟"只用该因子选 top1"的场景。

核心指标（为 LightGBM top-N 模型设计）：
- T1_mean      = Rank1 未来收益均值（模型实际收益代理）
- T1_crash     = Rank1 踩雷率（future_ret<=-8%）——与模型 top5 踩雷率 14.3% 同口径
- T1_q10       = Rank1 收益的 10% 分位（尾部风险）
- T1_mdd       = Rank1 序列的逐日 NAV 最大回撤
全部在 vwap_3d 标签下计算（与 LightGBM 一致）。
同时输出 c2c 同口径作平台参照。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from evaluator import load_panel
from fitness import DataContext, sia_of
from ledger import Ledger

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
LGB_TREF = 0.3
F24 = ("2025-03-06", "2025-09-04")
F56 = ("2025-09-05", "2026-01-12")


def _mp075(w: int) -> int:
    return max(2, int(round(w * 0.75)))


class Prim:
    def __init__(self, frame: pd.DataFrame):
        self.f = frame
        self.codes = frame["stock_code"]
        self.close = frame["close"].astype(float)
        self.open_ = frame["open"].astype(float)
        self.high = frame["high"].astype(float)
        self.low = frame["low"].astype(float)
        self.vol = frame["volume"].astype(float)
        self.turn = frame["turnover_rate"].astype(float)
        self.amount = frame["amount"].astype(float)
        self.dates = frame["date"]
        self.ret1 = self.close.groupby(self.codes, sort=False).pct_change()
        self._c = {}

    def ts_mean(self, s, w):
        return s.groupby(self.codes, sort=False).transform(
            lambda x: x.rolling(w, min_periods=_mp075(w)).mean())

    def ema(self, s, w):
        return s.groupby(self.codes, sort=False).transform(
            lambda x: x.ewm(span=w, adjust=False, min_periods=w).mean())

    def shift(self, s, n):
        return s.groupby(self.codes, sort=False).shift(n)

    def cum(self, w):
        return self.close / self.shift(self.close, w) - 1.0

    def corr_rolling(self, a, b, w):
        ma = self.ts_mean(a, w); mb = self.ts_mean(b, w)
        mab = self.ts_mean(a * b, w)
        va = self.ts_mean(a * a, w) - ma * ma
        vb = self.ts_mean(b * b, w) - mb * mb
        cov = mab - ma * mb
        den = np.sqrt((va.clip(lower=0) * vb.clip(lower=0)).to_numpy(dtype=float))
        return pd.Series(np.where(den > 0, cov.to_numpy(dtype=float) / den, np.nan), index=self.f.index)

    def cross_mean(self, s):
        return s.groupby(self.dates).transform("mean")


# 只保留 voldev_250 门控系列（上轮 top 表现）+ 少量对照
PAYLOADS = {
    "voldev_250": lambda P: P.vol / P.ts_mean(P.vol, 250).replace(0, np.nan) - 1.0,
    "ma_ret5": lambda P: P.ts_mean(P.ret1, 5),
    "ma_ret20": lambda P: P.ts_mean(P.ret1, 20),
    "low20_gap": lambda P: P.close / P.ts_mean(P.low, 20).replace(0, np.nan) - 1.0,
}
GATES = {
    "cum10>0.10": lambda P: P.cum(10) > 0.10,
    "cum5>0.10": lambda P: P.cum(5) > 0.10,
    "cum5>0.05": lambda P: P.cum(5) > 0.05,
    "ma_ret5>0": lambda P: P.ts_mean(P.ret1, 5) > 0,
    "cum5<0": lambda P: P.cum(5) < 0,
}


def rank1_tail(ctx, v_full: np.ndarray, direction: int) -> dict:
    """模拟"只用该因子选 top1"，收集 Rank1 的日收益序列与踩雷率。"""
    d_ = 1 if direction == 1 else -1
    key = np.asarray(v_full, dtype=float) * d_
    dates = ctx.row_dates
    ud = ctx.ud
    di = ctx.di
    labels = ctx.labels
    rank1_rets = []
    n_crash = 0
    for j in range(len(ud)):
        mask = di == j
        if mask.sum() < 100:
            continue
        kk = np.argsort(key[mask], kind="mergesort")
        rank1_rets.append(labels[mask][kk[-1]])
    pool = np.array(rank1_rets, dtype=float)
    pool = pool[np.isfinite(pool)]
    nav = np.cumprod(1.0 + pool) if len(pool) else np.array([1.0])
    mdd = float(np.max(np.maximum.accumulate(nav) / nav - 1.0)) if len(pool) else np.nan
    return {
        "T1_mean": float(pool.mean()) if len(pool) else np.nan,
        "T1_crash": float((pool <= -0.08).mean()) if len(pool) else np.nan,
        "T1_q10": float(np.percentile(pool, 10)) if len(pool) else np.nan,
        "T1_win": float((pool > 0).mean()) if len(pool) else np.nan,
        "T1_mdd": mdd,
        "T1_n": int(len(pool)),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    frame = load_panel()
    prim = Prim(frame)
    ctxs = {lab: DataContext(frame=frame, label=lab) for lab in ("vwap", "c2c")}
    print(f"panel loaded {time.time()-t0:.0f}s", flush=True)

    specs = []
    for gname, gfn in GATES.items():
        for pname, pfn in PAYLOADS.items():
            specs.append({"spec_id": f"cond_{gname.replace('>','_gt_').replace('<','_lt_')}__{pname}",
                          "gate": gname, "gfn": gfn, "payload": pname, "pfn": pfn})
    # 对照：无条件版（门关的裸 payload）
    for pname, pfn in PAYLOADS.items():
        specs.append({"spec_id": f"raw__{pname}", "gate": None, "gfn": None,
                      "payload": pname, "pfn": pfn, "raw": True})
    print(f"specs={len(specs)}", flush=True)

    rows = []
    for i, sp in enumerate(specs, 1):
        _t1 = time.time()
        try:
            payload = sp["pfn"](prim).replace([np.inf, -np.inf], np.nan)
            if sp.get("raw"):
                v_full = payload
            else:
                gate = sp["gfn"](prim)
                v_full = payload.where(gate, 0.0)
            v_full = pd.to_numeric(v_full, errors="coerce")
        except Exception as exc:
            rows.append({"spec_id": sp["spec_id"], "error": repr(exc)})
            continue
        rec = {"spec_id": sp["spec_id"], "gate": sp.get("gate") or "raw", "payload": sp["payload"]}
        for lab, ctx in ctxs.items():
            v = pd.to_numeric(v_full, errors="coerce").to_numpy(dtype=float)[ctx._pos]
            cov = float(np.isfinite(v).mean()) if len(v) else 0.0
            if cov <= 0.1:
                rec[lab] = {"valid": False, "coverage": cov}
                continue
            m = ctx.screen(v, name=sp["spec_id"])
            a_off = float(sia_of(m))
            d1 = int(m.get("direction", 1))
            # Rank1 尾部（vwap / c2c）
            vt = np.asarray(v, dtype=float)
            key = vt * (1 if d1 == 1 else -1)
            dates = ctx.row_dates
            ud = ctx.ud
            di = ctx.di
            labels = ctx.labels
            # 全窗口 Rank1
            r1_list = []
            for j in range(len(ud)):
                mask = di == j
                if mask.sum() < 100:
                    continue
                kk = np.argsort(key[mask], kind="mergesort")
                r1_list.append(labels[mask][kk[-1]])
            pool = np.array(r1_list, dtype=float)
            pool = pool[np.isfinite(pool)]
            nav = np.cumprod(1.0 + pool) if len(pool) else np.array([1.0])
            mdd = float(np.max(np.maximum.accumulate(nav) / nav - 1.0)) if len(pool) else np.nan
            t1 = {"T1_mean": float(pool.mean()) if len(pool) else np.nan,
                  "T1_crash": float((pool <= -0.08).mean()) if len(pool) else np.nan,
                  "T1_q10": float(np.percentile(pool, 10)) if len(pool) else np.nan,
                  "T1_win": float((pool > 0).mean()) if len(pool) else np.nan,
                  "T1_mdd": mdd, "T1_n": len(pool)}
            # 窗口拆分
            for wname, (lo, hi) in (("F24", F24), ("F56", F56)):
                sel = []
                for j in range(len(ud)):
                    if not (pd.Timestamp(lo) <= pd.Timestamp(ud[j]) <= pd.Timestamp(hi)):
                        continue
                    mask = di == j
                    if mask.sum() < 100:
                        continue
                    kk = np.argsort(key[mask], kind="mergesort")
                    sel.append(labels[mask][kk[-1]])
                wp = np.array(sel, dtype=float)
                wp = wp[np.isfinite(wp)]
                t1[f"{wname}_mean"] = float(wp.mean()) if len(wp) else np.nan
                t1[f"{wname}_crash"] = float((wp <= -0.08).mean()) if len(wp) else np.nan
                t1[f"{wname}_n"] = len(wp)
            rec[lab] = {"valid": True, "coverage": cov, "A_official": a_off,
                        "rank_ic": float(m.get("rank_ic_mean") or np.nan),
                        "ic_ir": float(m.get("ic_ir") or np.nan),
                        "rank_ic_ir": float(m.get("rank_ic_ir") or np.nan),
                        "ic_win": float(m.get("ic_win_rate") or np.nan),
                        "mono": float(m.get("monotonicity") or np.nan),
                        "turnover": float(m.get("turnover") or np.nan),
                        "direction": d1,
                        "LGB": float(abs(m.get("rank_ic_mean") or 0) * abs(m.get("rank_ic_ir") or 0)
                                     * (m.get("rank_ic_win_rate") or 0) * max(0, float(m.get("monotonicity") or 0))
                                     * min(1.0, LGB_TREF / max(float(m.get("turnover") or 0.3), LGB_TREF))),
                        **t1}
        rows.append(rec)
        vw = rec.get("vwap") or {}
        print(f"[{i}/{len(specs)}] {sp['spec_id'][:46]:46s} vwap A={vw.get('A_official', 0) or 0:.5f} "
              f"T1_mean={vw.get('T1_mean', 0) or 0:+.4f} T1_crash={vw.get('T1_crash', 0) or 0:.1%} "
              f"F24_mean={vw.get('F24_mean', 0) or 0:+.4f} {time.time()-_t1:.1f}s", flush=True)

    flat = []
    for rec in rows:
        for lab in ("vwap", "c2c"):
            r = rec.get(lab) or {}
            if not r.get("valid"):
                continue
            flat.append({"spec_id": rec["spec_id"], "gate": rec["gate"],
                         "payload": rec["payload"], "label": lab,
                         **{k: v for k, v in r.items() if k not in ("valid",)}})
    df = pd.DataFrame(flat)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"refs_cond_t1_{ts}_flat.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"saved {csv_path}  total {time.time()-t0:.0f}s", flush=True)

    # 打印 vwap 排名表
    v = df[(df["label"] == "vwap") & (df["T1_crash"].notna())].copy()
    print("\n===== vwap 口径：T1_crash（Rank1 踩雷率）升序 =====")
    print(v.sort_values("T1_crash")[
        ["spec_id", "A_official", "T1_mean", "T1_crash", "T1_q10", "T1_win", "T1_mdd", "F24_mean", "F24_crash"]
    ].head(14).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n===== vwap 口径：A_official 降序 =====")
    print(v.sort_values("A_official", ascending=False)[
        ["spec_id", "A_official", "T1_mean", "T1_crash", "T1_q10", "T1_win", "T1_mdd", "F24_mean"]
    ].head(14).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
