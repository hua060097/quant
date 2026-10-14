# -*- coding: utf-8 -*-
"""门控层回归测试（2026-10-12 改写：夹具全部合成，不依赖任何真实门控产物）。

背景：旧手工门控 M1–M4/S1–S3' 已删除（定义由搜索生成），原 T1/T2/T3 读
`gates/{market,sector}_gates.parquet`。现改为对**仍存在的共享基础设施与
搜索空间 OPS** 做性质测试，并保留原测试意图：

  T1 共享基础设施性质：`thresh_form` 边界 / NaN 透传；`_roll_quantile` 输出域与确定性。
  T2 G0 覆盖率口径（v2.8.18 修订意图，合成数据复现）：
     带预热缺口的序列在「全历史分母」下 fail、「评估窗分母」下 pass；
     且无预热缺口的序列不因口径变化而由 pass 转 fail。
  T3 门控因果性：搜索空间的 4 个操作化 OPS + `_roll_quantile` 只用 ≤T 数据——
     截断后重算，T 之前的值不得改变。
  T4 F1 掩码捷径等价性：`h3_eval.build_f1` 的连续形式与 f1_batch 的掩码捷径
     在 on-day 上逐日 IC 相同。

用法：python selftest_gates.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_validate as GV  # noqa: E402
import gate_search_space as GSS  # noqa: E402
import gates as G  # noqa: E402
import h3_eval as H  # noqa: E402

log: list[str] = []
ok_all = True


def chk(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all &= bool(cond)
    log.append(f'  [{"PASS" if cond else "FAIL"}] {name:<46s} {detail}')


def t1_shared_infra() -> None:
    """thresh_form 边界 / NaN；_roll_quantile 输出域与确定性。"""
    p = np.array([0.9, 0.7, 0.70001, 0.5, 0.30001, 0.3, 0.1, np.nan])
    g = G.thresh_form(p)
    chk("T1 thresh_form 上边界(>=THRESH_HI) → +1",
        g[0] == 1.0 and g[1] == 1.0 and g[2] == 1.0)
    chk("T1 thresh_form 下边界(<=THRESH_LO) → −1", g[5] == -1.0 and g[6] == -1.0)
    chk("T1 thresh_form 中间带 → NaN（含恰高于下界 1e-5）",
        np.isnan(g[3]) and np.isnan(g[4]))
    chk("T1 thresh_form NaN 透传", np.isnan(g[7]))
    s = pd.Series(np.arange(400, dtype=float), index=pd.date_range("2024-01-01", periods=400))
    q = G._roll_quantile(s, window=63, min_periods=63)
    fin = q.dropna()
    chk("T1 _roll_quantile 输出域 ⊂ (0,1]",
        bool((fin > 0).all() and (fin <= 1).all()), f"n={len(fin)}")
    q2 = G._roll_quantile(s, window=63, min_periods=63)
    chk("T1 _roll_quantile 确定性（两次逐值相同）",
        bool(np.array_equal(q.to_numpy(), q2.to_numpy(), equal_nan=True)))
    trunc = G._roll_quantile(s.iloc[:200], window=63, min_periods=63)
    mx = float(np.nanmax(np.abs(q.iloc[:200].to_numpy() - trunc.to_numpy())))
    chk("T1 _roll_quantile 因果（截断不改 ≤cut 值）", np.isfinite(mx) and mx == 0.0,
        f"max|Δ|={mx:.1e}")


def t2_g0_coverage() -> None:
    """G0.2 覆盖率口径：合成预热缺口序列复现 v2.8.18 的两类结果。

    构造要点（复现旧 M3 失败的形态）：预热缺口**大部分落在评估窗之前**——
    全历史分母下 cov=(N−w)/N=0.731<0.8 → fail；评估窗从预热结束之后开始 →
    cov≈1.0 → pass。门控值用有界正弦（std≈0.35≥0.2、|median|≈0≤0.8），
    避免随机游走序列随机触犯 std/median 阈值。
    """
    rng = np.random.default_rng(7)
    n, warm = 1300, 350
    idx = pd.Series(pd.date_range("2021-01-01", periods=n, freq="B"))
    t = np.arange(n) / 300.0
    pa = 0.5 + 0.25 * np.sin(2 * np.pi * t) + 0.02 * rng.normal(size=n)
    pa[:warm] = np.nan
    # eval_start 必须在预热结束之后（≈ idx[warm] = 2021-01-01 + 350 交易日）
    eval_start = str(pd.Timestamp(idx.iloc[warm]).date())
    ga = pd.Series(2.0 * pa - 1.0)
    full = GV.g0_continuous(idx, ga, eval_start=None)
    evalw = GV.g0_continuous(idx, ga, eval_start=eval_start)
    chk("T2 预热序列：全历史分母 → fail（复现旧口径缺陷）",
        full["g0_status"] == "fail", f"cov={full.get('cov')}")
    chk("T2 预热序列：评估窗分母 → pass（v2.8.18 修复）",
        evalw["g0_status"] == "pass", f"cov={evalw.get('cov')}")
    chk("T2 两种口径覆盖率显著不同",
        abs(evalw["cov"] - full["cov"]) > 0.1,
        f"{full.get('cov')} vs {evalw.get('cov')}")
    # 无预热缺口 → 口径变化不得由 pass 转 fail
    pb = 0.5 + 0.25 * np.sin(2 * np.pi * t) + 0.02 * rng.normal(size=n)
    gb = pd.Series(2.0 * pb - 1.0)
    r_none = GV.g0_continuous(idx, gb, eval_start=None)
    r_eval = GV.g0_continuous(idx, gb, eval_start=eval_start)
    chk("T2 无预热序列：口径变化不改判定方向",
        (r_none["g0_status"] == r_eval["g0_status"]),
        f"{r_none['g0_status']} vs {r_eval['g0_status']}")


def t3_causality() -> None:
    """搜索空间 OPS + _roll_quantile：截断 T 之后的数据，T 之前的值不得改变。"""
    rng = np.random.default_rng(3)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n)
    src = pd.Series(np.abs(0.5 + 0.01 * np.cumsum(rng.normal(size=n))), idx)
    worst = {}
    for op, _zh, fn in GSS.OPS:
        full = fn(src)
        cut = 350
        part = fn(src.iloc[:cut])
        d = np.nanmax(np.abs(full.iloc[:cut].to_numpy() - part.to_numpy()))
        worst[op] = float(d) if np.isfinite(d) else 0.0
    mx = max(worst.values())
    chk("T3 OPS 因果（4 个操作化截断重算逐值一致）", mx < 1e-12,
        " ".join(f"{k}={v:.1e}" for k, v in worst.items()))
    # 无 NaN 注入时 chg/accel 首段 NaN 数与窗口一致（窗口完整性）
    chg = [fn for op, _z, fn in GSS.OPS if op == "chg"][0]
    out = chg(src)
    chk("T3 OPS chg 前 20 日为 NaN（20 日差分窗口）",
        bool(np.isnan(out.iloc[:20]).all()) and
        bool(np.isfinite(out.iloc[20:]).all()),
        f"head_nan={int(np.isnan(out.iloc[:20]).sum())}")


def t4_mask_equivalence() -> None:
    """F1 连续形式 vs 掩码捷径：on-day 上逐日 IC 必须相同。"""
    rng = np.random.default_rng(0)
    n_stock, n_day = 400, 40
    x = rng.normal(size=(n_day, n_stock))
    y = rng.normal(size=(n_day, n_stock))
    g = rng.normal(size=n_day)
    d, s = 1, 1
    f1 = np.array([H.build_f1(x[i], g[i], d, s) for i in range(n_day)])
    ic_cont = []
    for i in range(n_day):
        v = f1[i]
        ok = np.isfinite(v) & np.isfinite(y[i])
        if ok.sum() < 50 or np.std(v[ok]) == 0:
            ic_cont.append(np.nan)
            continue
        ic_cont.append(float(np.corrcoef(
            pd.Series(v[ok]).rank(), pd.Series(y[i][ok]).rank())[0, 1]))
    ic_mask = []
    for i in range(n_day):
        on = np.isfinite(g[i]) and (s * g[i] > 0)
        if not on:
            ic_mask.append(np.nan)
            continue
        v = d * x[i]
        ok = np.isfinite(v) & np.isfinite(y[i])
        ic_mask.append(float(np.corrcoef(
            pd.Series(v[ok]).rank(), pd.Series(y[i][ok]).rank())[0, 1]))
    a, b = np.array(ic_cont), np.array(ic_mask)
    f = np.isfinite(a) & np.isfinite(b)
    mx = float(np.max(np.abs(a[f] - b[f]))) if f.any() else float("nan")
    chk("T4 掩码捷径与连续形式逐日 IC 相同", np.isfinite(mx) and mx < 1e-12,
        f"n={int(f.sum())} max|Δ|={mx:.2e}")
    chk("T4 off-day 两者同为 NaN",
        bool(np.all(np.isnan(a[~np.isfinite(b)]) & np.isnan(b[~np.isfinite(a)]))),
        f"off 天数={int((~np.isfinite(b)).sum())}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 78)
    print("门控层回归测试（合成夹具，不依赖真实门控产物）")
    print("=" * 78)
    for name, fn in (("T1 共享基础设施性质", t1_shared_infra),
                     ("T2 G0 覆盖率口径", t2_g0_coverage),
                     ("T3 门控因果性", t3_causality),
                     ("T4 掩码等价性", t4_mask_equivalence)):
        print(f"[{name}] ...", flush=True)
        try:
            fn()
        except Exception as e:
            chk(f"{name} 未抛异常", False, f"{type(e).__name__}: {str(e)[:120]}")
    print("\n".join(log))
    print(f'\nSUMMARY {"PASS" if ok_all else "FAIL"}')
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
