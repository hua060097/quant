"""定向条件/形态/市场交互因子筛选（第3路径扩展，aligned_v1 口径）。

背景：F2/F4 归因显示大亏 Rank1 是"追高结构"（cum5 中位 +15% vs 正收益 +8.7%），
且"近端涨幅门条件类/蜡烛形态/市场残差交互"三个机制轴在 M1~M13 模板中为缺口。
本脚本按用户指定做定向小网格（非全量盲搜）。

三族：
- cond   条件类：IF(近端涨幅门, payload, 0)，平台公式可表达（IF/DELAY/MAX/MIN 均 VERIFIED）
         gate: cum5>thr / cum10>thr / ma_ret5>thr；极性 hi(过热给分)/lo(低位给分)
- shape  形态类：上影/下影/实体/跳空/日内幅度（rolling w 均值 + ema3）
- market 市场交互/残差：个股收益-市场收益残差的累计（本地截面聚合口径，
         不可上传平台，仅自用 LightGBM 特征）

评估：fitness.DataContext（aligned_v1：2021-09-01~2026-08-21、全样本、3日调仓、z2 截尾），
c2c+vwap 双标签；官方 A（sia_of）+ LGB 复合分 + **F2/F4 尾部专项**
（F2/F4 窗口内按因子方向取 top10%：平均收益 / 大亏率<=-8% 占比 / 天数）。

用法：python refs_cond.py
产物：out/refs_cond_<ts>_flat.csv + .json + top MD
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
SD = HERE.parent / "screen_5d"
LGB_TREF = 0.3

F24 = ("2025-03-06", "2025-09-04")   # Fold2+Fold4 test 窗口
F56 = ("2025-09-05", "2026-01-12")


def _mp075(w: int) -> int:
    return max(2, int(round(w * 0.75)))


class Prim:
    """面板基本量与派生原语（merge 语义：MP_075/EMA=span/截面聚合本地版）。"""

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

    def ts_mean(self, s: pd.Series, w: int) -> pd.Series:
        return s.groupby(self.codes, sort=False).transform(
            lambda x: x.rolling(w, min_periods=_mp075(w)).mean())

    def ts_std(self, s: pd.Series, w: int) -> pd.Series:
        return s.groupby(self.codes, sort=False).transform(
            lambda x: x.rolling(w, min_periods=_mp075(w)).std())

    def ema(self, s: pd.Series, w: int) -> pd.Series:
        return s.groupby(self.codes, sort=False).transform(
            lambda x: x.ewm(span=w, adjust=False, min_periods=w).mean())

    def shift(self, s: pd.Series, n: int) -> pd.Series:
        return s.groupby(self.codes, sort=False).shift(n)

    def cum(self, w: int) -> pd.Series:
        return self.close / self.shift(self.close, w) - 1.0

    def corr_rolling(self, a: pd.Series, b: pd.Series, w: int) -> pd.Series:
        ma = self.ts_mean(a, w)
        mb = self.ts_mean(b, w)
        mab = self.ts_mean(a * b, w)
        va = self.ts_mean(a * a, w) - ma * ma
        vb = self.ts_mean(b * b, w) - mb * mb
        cov = mab - ma * mb
        den = np.sqrt((va.clip(lower=0) * vb.clip(lower=0)).to_numpy(dtype=float))
        return pd.Series(np.where(den > 0, cov.to_numpy(dtype=float) / den, np.nan),
                         index=self.f.index)

    def cross_mean(self, s: pd.Series) -> pd.Series:
        """截面日均值（本地聚合口径，不可上传）。"""
        return s.groupby(self.dates).transform("mean")


def build_prims() -> tuple[Prim, pd.DataFrame]:
    frame = load_panel()
    prim = Prim(frame)
    return prim, frame


PAYLOADS = {
    "ma_ret5": lambda P: P.ts_mean(P.ret1, 5),
    "ma_ret20": lambda P: P.ts_mean(P.ret1, 20),
    "turnover_ratio_5_250": lambda P: (P.ts_mean(P.turn, 5)
                                       / P.ts_mean(P.turn, 250).replace(0, np.nan)),
    "voldev_250": lambda P: P.vol / P.ts_mean(P.vol, 250).replace(0, np.nan) - 1.0,
    "corr_rv_40": lambda P: P.corr_rolling(P.ret1, P.vol, 40),
    "skew_ret_120": lambda P: P.ret1.groupby(P.codes, sort=False).transform(
        lambda x: x.rolling(120, min_periods=120).skew()),
    "low20_gap": lambda P: P.close / P.ts_mean(P.low, 20).replace(0, np.nan) - 1.0,
    "upper_wick_10": lambda P: ((P.high - pd.concat([P.open_, P.close], axis=1).max(axis=1))
                                / (P.high - P.low).replace(0, np.nan)
                                ).groupby(P.codes, sort=False).transform(
        lambda x: x.rolling(10, min_periods=8).mean()),
}
GATES = {
    "cum5>0.05": lambda P: P.cum(5) > 0.05,
    "cum5>0.10": lambda P: P.cum(5) > 0.10,
    "cum10>0.10": lambda P: P.cum(10) > 0.10,
    "ma_ret5>0": lambda P: P.ts_mean(P.ret1, 5) > 0,
    "ma_ret5>0.05": lambda P: P.ts_mean(P.ret1, 5) > 0.05,
    "cum5<0": lambda P: P.cum(5) < 0,
}
SHAPES = {
    "upper_wick": lambda P: ((P.high - pd.concat([P.open_, P.close], axis=1).max(axis=1))
                             / (P.high - P.low).replace(0, np.nan)),
    "lower_wick": lambda P: ((pd.concat([P.open_, P.close], axis=1).min(axis=1) - P.low)
                             / (P.high - P.low).replace(0, np.nan)),
    "body_ratio": lambda P: (P.close - P.open_) / P.close.replace(0, np.nan),
    "gap": lambda P: P.open_ / P.shift(P.close, 1).replace(0, np.nan) - 1.0,
    "intraday": lambda P: P.close / P.open_.replace(0, np.nan) - 1.0,
}
MARKET = {
    "resid_cum5": lambda P: P.cum(5) - P.cross_mean(P.cum(5)),
    "resid_ret10": lambda P: (P.close / P.shift(P.close, 10) - 1.0)
                             - P.cross_mean(P.close / P.shift(P.close, 10) - 1.0),
    "resid_turnover_dev": lambda P: (P.turn / P.ts_mean(P.turn, 5).replace(0, np.nan) - 1.0)
                                     - P.cross_mean(P.turn / P.ts_mean(P.turn, 5).replace(0, np.nan) - 1.0),
    "mkt_vol_state": lambda P: P.cross_mean(P.ret1.abs()),
}


def make_specs() -> list[dict]:
    specs = []
    for gname, gfn in GATES.items():
        for pname, pfn in PAYLOADS.items():
            for pol in ("hi", "lo"):
                specs.append({"family": "cond", "gate": gname, "payload": pname,
                              "pol": pol, "gfn": gfn, "pfn": pfn,
                              "spec_id": f"cond_{gname.replace('>','_gt_').replace('<','_lt_')}__{pol}__{pname}"})
    for sname, sfn in SHAPES.items():
        for w in (5, 10):
            for tf in ("raw", "ema3"):
                specs.append({"family": "shape", "gate": sname, "window": w, "tf": tf,
                              "spec_id": f"shape_{sname}_{w}_{tf}"})
    for mname, mfn in MARKET.items():
        for tf in ("raw", "ema3"):
            specs.append({"family": "market", "gate": mname, "tf": tf,
                          "spec_id": f"market_{mname}_{tf}"})
    return specs


def compute_value(prim: Prim, spec: dict) -> pd.Series:
    out = None
    if spec["family"] == "cond":
        gate = spec["gfn"](prim)
        payload = spec["pfn"](prim).replace([np.inf, -np.inf], np.nan)
        if spec["pol"] == "hi":
            cond = gate
        else:
            cond = ~gate
        v = payload.where(cond, 0.0)
        out = v
    elif spec["family"] == "shape":
        base = SHAPES[spec["gate"]](prim)
        base = base.replace([np.inf, -np.inf], np.nan)
        if spec["tf"] == "ema3":
            out = prim.ema(base, 3)
        else:
            out = base
    else:
        out = MARKET[spec["gate"]](prim).replace([np.inf, -np.inf], np.nan)
        if spec["tf"] == "ema3":
            out = prim.ema(out, 3)
    return pd.to_numeric(out, errors="coerce")


def lgb_score(m: dict, t_ref: float = LGB_TREF) -> float:
    turn = m.get("turnover")
    pen = 1.0 if turn is None or not np.isfinite(turn) \
        else float(t_ref) / max(float(turn), float(t_ref))
    mono = m.get("monotonicity")
    mono = max(0.0, float(mono)) if mono is not None and np.isfinite(mono) else 1.0
    return float(abs(m.get("rank_ic_mean") or 0) * abs(m.get("rank_ic_ir") or 0)
                 * (m.get("rank_ic_win_rate") or 0) * mono * pen)


def tail_utility(values: np.ndarray, dates: pd.DatetimeIndex, lo: str, hi: str,
                 direction: int, future: np.ndarray) -> dict:
    """窗口内逐调仓日 top10% 按方向选取，池化收益与大亏率。"""
    lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
    sel = []
    for d in np.unique(values_dates := _dates_in):
        pass
    return {}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    frame = load_panel()
    print(f"panel rows={len(frame):,} {time.time()-t0:.0f}s", flush=True)
    prim = Prim(frame)
    ctxs = {lab: DataContext(frame=frame, label=lab) for lab in ("c2c", "vwap")}
    for lab, ctx in ctxs.items():
        print(f"[{lab}] rows={ctx.n_rows:,} sections={len(ctx.ud)} "
              f"{str(ctx.ud[0])[:10]}~{str(ctx.ud[-1])[:10]}", flush=True)
    ledger = Ledger()

    # 折窗口调仓日索引（ctx.ud 为全部调仓日）
    def win_idx(ctx, lo, hi):
        return [i for i, d in enumerate(ctx.ud)
                if pd.Timestamp(lo) <= pd.Timestamp(d) <= pd.Timestamp(hi)]

    rows = []
    specs = make_specs()
    print(f"specs={len(specs)}", flush=True)
    cache_payloads: dict = {}
    for i, sp in enumerate(specs, 1):
        t1 = time.time()
        try:
            v_full = compute_value(prim, sp)
        except Exception as exc:  # noqa: BLE001
            rows.append({"spec_id": sp["spec_id"], "family": sp["family"], "error": repr(exc)})
            continue
        rec = {"spec_id": sp["spec_id"], "family": sp["family"], "gate": sp.get("gate"),
               "payload": sp.get("payload"), "pol": sp.get("pol"),
               "window": sp.get("window"), "tf": sp.get("tf")}
        for lab, ctx in ctxs.items():
            v = pd.to_numeric(v_full, errors="coerce").to_numpy(dtype=float)[ctx._pos]
            cov = float(np.isfinite(v).mean()) if len(v) else 0.0
            if cov <= 0.1:
                rec[lab] = {"valid": False, "coverage": cov}
                continue
            m = ctx.screen(v, name=sp["spec_id"])
            a_off = float(sia_of(m))
            # 尾部专项：窗口内逐调仓日 top10%（按方向）
            def tail(lo, hi):
                idxs = win_idx(ctx, lo, hi)
                if len(idxs) < 10:
                    return {"mean": np.nan, "大亏率": np.nan, "n": 0}
                vals = v
                rets = ctx.labels
                d_ = int(m.get("direction", 1))
                picks = []
                for j in idxs:
                    mask = ctx.di == j
                    if mask.sum() < 100:
                        continue
                    key = np.asarray(vals[mask], dtype=float) * (1 if d_ == 1 else -1)
                    kk = np.argsort(key, kind="mergesort")
                    n10 = max(int(mask.sum() * 0.1), 1)
                    picks.append(rets[mask][kk[-n10:]])
                pool = np.concatenate(picks) if picks else np.array([])
                pool = pool[np.isfinite(pool)]
                return {"mean": float(pool.mean()) if len(pool) else np.nan,
                        "大亏率": float((pool <= -0.08).mean()) if len(pool) else np.nan,
                        "n": int(len(pool))}
            t24 = tail(F24[0], F24[1])
            t56 = tail(F56[0], F56[1])
            rec[lab] = {"valid": True, "coverage": cov, "A_official": a_off,
                        "LGB": float(abs(m.get("rank_ic_mean") or 0) * abs(m.get("rank_ic_ir") or 0)
                                     * (m.get("rank_ic_win_rate") or 0)),
                        "rank_ic": float(m.get("rank_ic_mean") or np.nan),
                        "ic_ir": float(m.get("ic_ir") or np.nan),
                        "rank_ic_ir": float(m.get("rank_ic_ir") or np.nan),
                        "ic_win": float(m.get("ic_win_rate") or np.nan),
                        "mono": float(m.get("monotonicity") or np.nan),
                        "turnover": float(m.get("turnover") or np.nan),
                        "direction": int(m.get("direction", 1)),
                        "F24_top10_mean": t24["mean"], "F24_top10_大亏率": t24["大亏率"],
                        "F24_top10_n": t24["n"],
                        "F56_top10_mean": t56["mean"], "F56_top10_大亏率": t56["大亏率"]}
            if lab == "c2c":
                ledger.log_raw("local", f"{sp['spec_id']}|{sp['family']}", direction=int(m.get("direction", 1)),
                               family="ref:cond-shape-market",
                               data={"source": "ref-screen", "spec": sp["spec_id"],
                                     "sia": a_off, "lgb": rec[lab]["LGB"]},
                               label=lab)
        rows.append(rec)
        c2 = rec.get("c2c") or {}
        vw = rec.get("vwap") or {}
        print(f"[{i}/{len(specs)}] {sp['spec_id'][:46]:46s} "
              f"c2c A={c2.get('A_official', 0) or 0:.5f} vwap LGB={vw.get('LGB', 0) or 0:.5f} "
              f"F24大亏率={c2.get('F24_top10_大亏率') if c2.get('F24_top10_大亏率') is not None else 'nan'} "
              f"{time.time()-t1:.1f}s", flush=True)

    flat = []
    for rec in rows:
        for lab in ("c2c", "vwap"):
            r = rec.get(lab) or {}
            if not r.get("valid"):
                continue
            flat.append({"spec_id": rec["spec_id"], "family": rec["family"],
                         "gate": rec.get("gate"), "payload": rec.get("payload"),
                         "pol": rec.get("pol"), "window": rec.get("window"), "tf": rec.get("tf"),
                         "label": lab, **{k: v for k, v in r.items() if k not in ("valid",)}})
    df = pd.DataFrame(flat)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"refs_cond_{ts}_flat.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    (OUT_DIR / f"refs_cond_{ts}.json").write_text(
        json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                    "n_specs": len(specs)}, ensure_ascii=False) + "\n" + df.to_json(orient="records", force_ascii=False),
        encoding="utf-8")
    print(f"saved {csv_path}  total {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
