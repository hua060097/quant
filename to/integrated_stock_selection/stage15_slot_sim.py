# -*- coding: utf-8 -*-
"""日频槽位组合模拟器（stage15 真实净值口径 v1，2026-09-25）

目的：替代两处失真口径——
  - CV/deploy 的 rank_daily cum/mdd：每日满仓逐日复利（2~3日重叠持仓被重复计本）
  - 分钟引擎：路径依赖强（状态机/止损/门控），不适合当批量相对排序
本模拟按"每日入场 + N 槽轮换 + 持仓 3 日"复现可实现的组合净值：
  信号 t 日 → 次日 open(t+1) 买入（付 one-way 成本）→
  t+1 收盘 mark、t+2 收盘 mark → t+3 以 0.7*close(t+2)+0.3*close(t+3) 卖出（付成本）。
  槽位 N（默认3=与持仓天数匹配；2=现行引擎口径）→ 无空槽的入场被跳过（计数）。
  每槽资金独立复利；空闲槽保持现金；全程按槽等权 1/N（不随行情调仓）。
兼容输入：deploy 的 stock_details_top5.csv（date/rank/stock_code）与
          quick 的 hp_stock_details_top5.csv（含 fold_id/checkpoint/param，逐组独立模拟）。

用法：
  python stage15_slot_sim.py --details results\\deploy_...\\stock_details_top5.csv --slots 3 --rank 1
  python stage15_slot_sim.py --details results\\stage15hp_quick_xxx\\hp_stock_details_top5.csv --slots 2,3
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ONE_WAY_COST = 0.003


def _find_prices() -> Path:
    cands = sorted(glob.glob(str(ROOT / "cache" / "stock_features_*.parquet")))
    if not cands:
        raise FileNotFoundError("cache/stock_features_*.parquet 不存在")
    cands.sort(key=lambda p: Path(p).stem.replace("stock_features_", ""))
    return Path(cands[0])


def load_prices(cache: Path) -> tuple[dict, np.ndarray]:
    px = pd.read_parquet(cache, columns=["stock_code", "date", "open", "close"])
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px["stock_code"] = px["stock_code"].astype(str)
    px = px.dropna(subset=["open", "close"])
    px = px[px["open"] > 0]
    px = px[px["close"] > 0]
    px = px.sort_values(["stock_code", "date"])
    cal = np.array(sorted(px["date"].unique()))
    cal_idx = {d: i for i, d in enumerate(cal)}
    # code -> {date: (open, close)}
    by_code: dict[str, dict] = {}
    for code, g in px.groupby("stock_code", sort=False):
        by_code[code] = {d: (float(o), float(c))
                         for d, o, c in zip(g["date"], g["open"], g["close"])}
    return by_code, cal


def simulate_group(picks: pd.DataFrame, by_code: dict, cal: np.ndarray, cal_idx: dict,
                   slots: int, rank: int, cost: float, start=None, end=None) -> dict:
    sub = picks[picks["rank"] == rank].copy()
    if sub.empty:
        return {}
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    if start is not None:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end is not None:
        sub = sub[sub["date"] <= pd.Timestamp(end)]
    sub = sub.sort_values("date")
    if sub.empty:
        return {}
    n_cal = len(cal)
    events = []
    skipped = 0
    for _, row in sub.iterrows():
        t = row["date"]
        code = str(row["stock_code"])
        ti = cal_idx.get(t)
        if ti is None or ti + 3 >= n_cal:
            skipped += 1
            continue
        series = by_code.get(code)
        if series is None:
            skipped += 1
            continue
        d1, d2, d3 = cal[ti + 1], cal[ti + 2], cal[ti + 3]
        p1, p2, p3 = series.get(d1), series.get(d2), series.get(d3)
        if p1 is None or p2 is None or p3 is None:
            skipped += 1
            continue
        o1, c1 = p1
        _, c2 = p2
        _, c3 = p3
        blended = 0.7 * c2 + 0.3 * c3
        gross = (1 - cost) * (blended / o1) * (1 - cost) - 1.0
        events.append(dict(entry_idx=ti + 1, exit_idx=ti + 3, code=code, gross=gross,
                           marks=[(ti + 1, c1 / o1), (ti + 2, c2 / o1), (ti + 3, blended / o1)]))
    if not events:
        return {}
    events.sort(key=lambda e: (e["entry_idx"], e["code"]))
    # 槽分配：取 free_from 最小且 <= entry_idx 的槽
    slot_free = [-1] * slots
    slot_trades = [[] for _ in range(slots)]
    for e in events:
        best = None
        for si in range(slots):
            if slot_free[si] <= e["entry_idx"]:
                if best is None or slot_free[si] < slot_free[best]:
                    best = si
        if best is None:
            skipped += 1
            continue
        slot_trades[best].append(e)
        slot_free[best] = e["exit_idx"] + 1
    # 逐槽构建净值（独立复利，空闲现金），逐日求和
    nav = np.zeros(n_cal, dtype=float)
    all_idx = []
    for si in range(slots):
        eq = 1.0 / slots
        arr = np.full(n_cal, np.nan, dtype=float)
        arr[0] = eq
        for e in slot_trades[si]:
            base = eq
            a, b = e["entry_idx"], e["exit_idx"]
            for ii, mm in e["marks"]:
                arr[ii] = base * (1 - cost) * mm
            eq = base * (1 - cost) * e["marks"][-1][1] * (1 - cost)
            arr[b] = eq
            all_idx.extend([a, b])
        s = pd.Series(arr).ffill().bfill().to_numpy()
        nav += s
    if not all_idx:
        return {}
    lo, hi = min(all_idx), max(all_idx)
    nav_s = pd.Series(nav[lo:hi + 1])
    ret = nav_s.pct_change().dropna()
    n = len(ret)
    if n < 5:
        return {}
    ann = float(nav_s.iloc[-1]) ** (252.0 / n) - 1.0 if nav_s.iloc[-1] > 0 else np.nan
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else np.nan
    dd = float((nav_s / nav_s.cummax() - 1).min())
    pnls = [e["gross"] for sl in slot_trades for e in sl]
    return dict(n_days=n, nav_end=float(nav_s.iloc[-1]), ann=float(ann), sharpe=sharpe,
                maxdd=dd, trades=len(pnls), skipped=skipped,
                pos_win=float(np.mean([g > 0 for g in pnls])) if pnls else np.nan,
                avg_pnl=float(np.mean(pnls)) if pnls else np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description="日频槽位组合模拟（真实净值口径）")
    ap.add_argument("--details", required=True)
    ap.add_argument("--slots", default="3", help="槽位数，逗号分隔（如 2,3）")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--cost", type=float, default=ONE_WAY_COST)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    details = pd.read_csv(args.details, encoding="utf-8-sig")
    need = {"rank", "stock_code", "date"}
    if not need.issubset(details.columns):
        raise SystemExit(f"details 需含 {need} 列，实际: {list(details.columns)}")
    cache = _find_prices()
    print(f"[SIM] 价格缓存: {cache.name}", flush=True)
    by_code, cal = load_prices(cache)
    cal_idx = {d: i for i, d in enumerate(cal)}
    print(f"[SIM] 股票={len(by_code):,} 交易日={len(cal)} {pd.Timestamp(cal[0]).date()}~{pd.Timestamp(cal[-1]).date()}", flush=True)

    keys = [c for c in ("param", "checkpoint", "fold_id") if c in details.columns]
    rows = []
    for slots in [int(x) for x in str(args.slots).split(",") if x.strip()]:
        if keys:
            iterator = details.groupby(keys, sort=False)
        else:
            iterator = [((), details)]
        for key, g in iterator:
            r = simulate_group(g, by_code, cal, cal_idx, slots, args.rank, args.cost,
                               start=args.start, end=args.end)
            if not r:
                continue
            if keys:
                r.update(dict(zip(keys, key if isinstance(key, tuple) else (key,))))
            r["slots"] = slots
            r["rank"] = args.rank
            rows.append(r)
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("无模拟结果")
    cols = keys + ["slots", "rank", "n_days", "nav_end", "ann", "sharpe", "maxdd",
                   "trades", "skipped", "pos_win", "avg_pnl"]
    res = res[[c for c in cols if c in res.columns]]
    pd.set_option("display.width", 240)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.out:
        res.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[SIM] 已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
