# -*- coding: utf-8 -*-
"""H2 重跑（新选择窗 ≤2025-12-31），基于 fastranks 加速层。

**为什么必须重跑**：旧 P0/P1 的 H2 用 label_max=2024-12-26（869 交易日），
新选择窗到 2025-12-31（1115 交易日，多 246 天）。样本量不同 → t_nw 系统性
偏大、null_p 与 RankIC_mean 均不可比；若不重跑，新搜到的因子会因"多看一年"
而系统性占优。

**加速**（见 `fastranks.py` 文件头 profile 结论）：单因子 36.9s → 4.5s（**8.2×**），
且与原实现**逐位一致**（同 seed 同 RNG 消费顺序 + 只算 sxy 的置换）。
再叠加多进程 → 124 因子约 1~2 分钟。

用法：
    python h2_rerun.py                    # 全量（多进程）
    python h2_rerun.py --workers 6 --chunk 8
    python h2_rerun.py --limit 8 --no-parallel --verify   # 对拍原实现
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary as B  # noqa: E402
import eval_v2 as E  # noqa: E402
import fastranks as FR  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
CUT = W.FACTOR_SCREEN_MAX
_NP = 50


def build_panel(chunk: list[str]) -> tuple[pd.DataFrame, FR.PanelRanker]:
    lab = B.read_labels(["r3", "tradable"], label_max=CUT, label="h2_rerun")
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab[(lab["tradable"] == 1) & (lab["date"] <= CUT)]
    fac = B.read_factors(chunk, data_max=W.LABEL_MAX_DATE, label="h2_rerun")
    fac["date"] = pd.to_datetime(fac["date"])
    m = lab[["stock_code", "date", "r3"]].merge(fac, on=["stock_code", "date"],
                                                how="inner")
    m = m[m["date"] <= CUT]
    del fac
    return m, FR.PanelRanker(m)


def work(chunk: list[str]) -> list[dict]:
    m, pr = build_panel(chunk)
    out = []
    for c in chunk:
        if c not in m.columns:
            continue
        t0 = time.time()
        try:
            r = pr.h2_core(c, n_perm=_NP, seed=0)
        except Exception as e:
            r = {"factor": c, "error": str(e)[:200]}
        r["secs"] = round(time.time() - t0, 2)
        out.append(r)
    del m, pr
    return out


def main() -> int:
    global _NP
    ap = argparse.ArgumentParser(description="H2 重跑（新选择窗，fastranks 加速）")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--verify", action="store_true", help="对拍 eval_v2.h2_record")
    ap.add_argument("--out", default=os.path.join(RESULTS, "h2_stock_2025.csv"))
    a = ap.parse_args()
    _NP = a.n_perm
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    roles = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
    names = list(roles.query("role=='STOCK'")["name"])
    if a.limit:
        names = names[:a.limit]

    if a.verify:
        chunk = names[:min(4, len(names))]
        m, pr = build_panel(chunk)
        print(f"panel {len(m):,} 行 {m['date'].nunique()} 天")
        for c in chunk:
            t0 = time.time()
            fast = pr.h2_core(c, n_perm=_NP, seed=0)
            t1 = time.time()
            slow = E.h2_record(m, c, n_perm=_NP, seed=0)
            t2 = time.time()
            keys = ["n_days", "rankic_mean", "t_nw", "null_p", "null_mean"]
            ok = all((np.isnan(fast[k]) and np.isnan(slow[k]))
                     or abs(fast[k] - slow[k]) < 1e-12 for k in keys
                     if k in slow and k in fast)
            print(f"  {c:<28s} fast {t1 - t0:5.2f}s | slow {t2 - t1:5.2f}s | "
                  f"提速 {(t2 - t1) / max(t1 - t0, 1e-9):4.1f}x | 逐位一致={ok}")
            if not ok:
                for k in keys:
                    if k in slow and k in fast:
                        print(f"      {k}: fast={fast[k]} slow={slow[k]}")
        return 0

    chunks = [names[i:i + a.chunk] for i in range(0, len(names), a.chunk)]
    print(f"窗口 ≤{CUT}；因子 {len(names)}；chunk {a.chunk} → {len(chunks)} 任务；"
          f"进程 {a.workers if not a.no_parallel else 1}；n_perm={_NP}")

    t0 = time.time()
    rows: list[dict] = []
    if a.no_parallel:
        for i, ch in enumerate(chunks, 1):
            rows.extend(work(ch))
            print(f"  [{i}/{len(chunks)}] {len(rows)} 因子 {time.time() - t0:.0f}s",
                  flush=True)
    else:
        with Pool(a.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(work, chunks), 1):
                rows.extend(res)
                print(f"  [{i}/{len(chunks)}] {len(rows)} 因子 {time.time() - t0:.0f}s",
                      flush=True)

    df = pd.DataFrame(rows)
    df["window_start"] = W.FACTORS_FIRST_SIGNAL_DATE
    df["window_end"] = CUT
    df["label_available_max"] = W.LABEL_MAX_DATE
    df["evidence"] = "E1"
    cols = ["factor", "n_days", "rankic_mean", "t_nw", "null_p", "null_mean",
            "window_start", "window_end", "label_available_max", "evidence"]
    df = df[[c for c in cols if c in df.columns] +
            [c for c in df.columns if c not in cols]].sort_values("factor")
    df.to_csv(a.out, index=False, encoding="utf-8")
    print(f"\n输出 → {a.out}  {df.shape}  用时 {time.time() - t0:.0f}s")

    old = os.path.join(RESULTS, "h2_stock_all.csv")
    if os.path.isfile(old):
        o = pd.read_csv(old).set_index("factor")
        n = df.set_index("factor")
        c = [x for x in n.index if x in o.index]
        d_ic = n.loc[c, "rankic_mean"] - o.loc[c, "rankic_mean"]
        d_t = n.loc[c, "t_nw"].abs() - o.loc[c, "t_nw"].abs()
        flip = int(((n.loc[c, "rankic_mean"] > 0) != (o.loc[c, "rankic_mean"] > 0)).sum())
        print(f"\n与旧窗（≤{W.LEGACY_SEARCH_END}，{int(o['n_days'].median())} 天）对比：")
        print(f"  n_days {int(o['n_days'].median())} → {int(n['n_days'].median())}")
        print(f"  |Δrankic_mean| 中位 {d_ic.abs().median():.5f} "
              f"90分位 {d_ic.abs().quantile(.9):.5f}")
        print(f"  |t_nw| 增长中位 {d_t.median():+.3f}")
        print(f"  符号翻转 {flip}/{len(c)}")
        npo = (o.loc[c, "null_p"] < 0.05).sum()
        npn = (n.loc[c, "null_p"] < 0.05).sum()
        print(f"  null_p<0.05 个数 {int(npo)} → {int(npn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
