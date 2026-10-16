# -*- coding: utf-8 -*-
"""因子时间稳定性诊断（2026-10-10）。**纯诊断，不改动任何冻结规则。**

动机与依据（见 results/merge_v2/stability_diag_report.md）：
用户提出「因子若在某段时间完全失效则放弃」。方向正确，但朴素形式在统计上
不可行——用本项目 124 个 STOCK 因子的实测数据反推：

    日频 RankIC std ≈ 0.175；因子 |RankIC| 中位 0.0245
    126 日块的 MDE(双侧5%,功效50%) ≈ 0.0306  >  中位 |RankIC| 0.0245
    ⇒ 一半的因子在单个半年块内无法与零区分，任何"按块显著性/符号"的删除
      规则必然大规模误杀。
    H0 下"连续 2 块反向"的概率本身就有 25%，"连续 3 块"12.5%。

因此本模块改用两个统计功效显著更高的诊断量，**均不做删除，只出报告**：

1. **IC 集中度 conc_top10**：Σ(top 10% |日IC| 的 IC) / Σ(全部 |日IC|)。
   |IC| 完全均匀时恰为 0.10；越高说明预测力越依赖少数几天。这是连续量，不受
   块长限制，能在 869 日样本上稳定估计——直接针对"靠某一段行情吃老本"。
   注意该值受日 IC 分布偏度影响，**判读必须对照零分布的百分位**，不可套用
   理论值 0.10。

2. **衰减趋势检验**：对方向校正后的日 IC 序列做时间趋势回归（NW t），
   以及"最近 252 日 vs 前 252 日"的均值差（NW t）。用满全部样本，功效远高于
   分块符号检验。负向显著才提示衰退。

零假设标定：对日 IC 序列做**块内置换**（block=5，覆盖 R3 的 3 日标签重叠），
保留边际分布与短期自相关，破坏时间结构。给出每个诊断量的经验零分布与触发率，
供后续决定阈值——**阈值必须由零分布反推，不得事后手调**。

窗口：只用设计窗 2021-06-01 ~ 2024-12-26（`windows.LEGACY_SEARCH_END`）。
2025-01-02 ~ 2026-09-11 的 415 个交易日**保持封存**，留作 P2 留出实验，
本模块一律不读（`--window` 可显式指定，但默认封存）。

用法：
    python stability_diag.py                  # STOCK 124，设计窗
    python stability_diag.py --block 252      # 块长
    python stability_diag.py --window prod    # 显式改窗（会破坏留出，请慎用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary as B  # noqa: E402
import eval_v2 as E  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
CHUNK = 16          # 一次载入的因子列数（4.1M 行 × 16 列 ≈ 0.5GB float64）
MIN_BLOCK_DAYS = 120


# ------------------------------------------------------------------ 诊断量
def ic_concentration(ic: np.ndarray, q: float = 0.10) -> float:
    """Σ(top q |日IC| 的 IC)/Σ|日IC|。

    警告：该量只依赖日 IC 的**多重集**，对时间顺序不变 → 块内置换零分布退化
    （实测变动幅度 0.00e+00），必须用**截面内置换**做零分布。且它主要反映日 IC
    分布的偏度（正均值右偏因子会得到负值），不宜直接解读为"时间集中度"。
    保留它仅供与 `top10_contrib` 对照。
    """
    v = ic[np.isfinite(ic)]
    if len(v) < 20:
        return float("nan")
    denom = np.abs(v).sum()
    if denom <= 0:
        return float("nan")
    k = max(int(round(len(v) * q)), 1)
    top = v[np.argsort(-np.abs(v), kind="mergesort")[:k]]
    return float(top.sum() / denom)


def top10_contrib(ic: np.ndarray, q: float = 0.10) -> float:
    """【已弃用，见 jackknife_top10】保留仅为与旧结果对照。

    分母是"总 IC 和"，而该量在 H0 下本身就是接近 0 的噪声量（std ≈
    sd×√n），故比值在零假设下方差爆炸：解析零分布分位 -1.43/+1.96，
    真实置换抽样仅 -0.007/+0.039，两者差两个数量级。不可用于任何校准。
    """
    v = ic[np.isfinite(ic)]
    if len(v) < 20:
        return float("nan")
    total = v.sum()
    if abs(total) < 1e-12:
        return float("nan")
    k = max(int(round(len(v) * q)), 1)
    top = v[np.argsort(-np.abs(v), kind="mergesort")[:k]]
    return float(top.sum() / total)


def jackknife_top10(ic: np.ndarray, q: float = 0.10) -> float:
    """丢弃 |IC| 最大的 q 天后，剩余日的 IC 均值 ÷ 原均值。

    ≈1.0 → 该因子不依赖少数几天（稳健）；≈0 → 预测力几乎全在那几天。
    分母是因子自身的 ic_mean（对真实因子由 869 天充分确定），不像
    `top10_contrib` 那样在 H0 下退化，故可用于真实因子间的横向比较。
    """
    v = ic[np.isfinite(ic)]
    if len(v) < 40:
        return float("nan")
    mu = v.mean()
    if abs(mu) < 1e-9:
        return float("nan")
    k = max(int(round(len(v) * q)), 1)
    keep = np.argsort(-np.abs(v), kind="mergesort")[k:]
    return float(v[keep].mean() / mu)


def ic_acf(x: np.ndarray, max_lag: int = 10) -> dict:
    """日 IC 序列的自相关 + 有效样本量。

    n_eff = n / (1 + 2·Σ_{k=1..L}(1−k/n)·ρ_k)（Bartlett 型）。
    n_eff/n 越接近 1，块均 IC 的抽样误差越小、时间稳定性越可测。
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 50:
        return {}
    v = v - v.mean()
    denom = float((v * v).sum())
    if denom <= 0:
        return {}
    acf = [float((v[k:] * v[:-k]).sum() / denom) for k in range(1, max_lag + 1)]
    s = 1.0 + 2.0 * sum((1.0 - (k + 1) / n) * a for k, a in enumerate(acf))
    s = max(s, 1.0)
    return {"acf1": acf[0], "acf5": acf[4] if len(acf) >= 5 else float("nan"),
            "acf10": acf[9] if len(acf) >= 10 else float("nan"),
            "n_eff_ratio": float(1.0 / s), "n_eff": float(n / s)}


def nw_slope_t(y: np.ndarray, lag: int = E.NW_LAG) -> tuple[float, float]:
    """对 y ~ t 做 OLS，返回 (slope, NW t)。用满样本，功效高于分块检验。"""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    y = y[ok]
    n = len(y)
    if n < 30:
        return float("nan"), float("nan")
    t = np.arange(n, dtype=float)
    tm, ym = t.mean(), y.mean()
    sxx = ((t - tm) ** 2).sum()
    if sxx <= 0:
        return float("nan"), float("nan")
    slope = float(((t - tm) * (y - ym)).sum() / sxx)
    e = y - (ym + slope * (t - tm))
    s = (e ** 2).sum()
    for L in range(1, lag + 1):
        s += 2.0 * (1.0 - L / (lag + 1)) * float((e[L:] * e[:-L]).sum())
    var = s / n / n
    if var <= 0:
        return slope, float("nan")
    return slope, float(slope / np.sqrt(var))


def nw_mean_diff_t(y: np.ndarray, w: int, lag: int = E.NW_LAG) -> tuple[float, float]:
    """最近 w 日均值 − 前 w 日均值，及该差的 NW t。"""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2 * w + 10:
        return float("nan"), float("nan")
    a, b = float(y[:w].mean()), float(y[-w:].mean())
    delta = b - a
    # 围绕两段均值的残差 → NW 修正后求 delta 的标准误
    e = y - (a + b) / 2.0
    s = float((e ** 2).sum())
    for L in range(1, lag + 1):
        s += 2.0 * (1.0 - L / (lag + 1)) * float((e[L:] * e[:-L]).sum())
    se = np.sqrt(s / n / n) if s > 0 else np.nan
    t = delta / se if se and np.isfinite(se) else float("nan")
    return float(delta), float(t)


def block_profile(ic: np.ndarray, blk: int) -> dict:
    """按固定块长切块，返回块均序列与摘要。"""
    v = ic[np.isfinite(ic)]
    n = len(v)
    nb = n // blk
    if nb < 2:
        return {"n_blocks": 0, "means": [], "worst": float("nan"),
                "pos_frac": float("nan"), "max_consec_neg": 0,
                "coverage": float("nan"), "first": float("nan"),
                "last": float("nan")}
    means = [float(v[i * blk:(i + 1) * blk].mean()) for i in range(nb)]
    sign = np.sign(means)
    run = best = 0
    for s in sign:
        run = run + 1 if s < 0 else 0
        best = max(best, run)
    return {"n_blocks": nb, "means": means,
            "worst": float(min(means)), "best": float(max(means)),
            "pos_frac": float((sign > 0).mean()),
            "max_consec_neg": int(best),
            "coverage": float(nb * blk / max(n, 1)),
            "first": float(means[0]), "last": float(means[-1])}


def sign_consistency(adj: np.ndarray, n_blocks: int = 5) -> dict:
    """子样本 IC 符号一致性（split-half sign consistency）。

    把方向校正后的日 IC 序列切成 n_blocks 段等长连续块，各算块均 IC，输出：
    - same_sign_blocks：与全样本均值同号的块数（**不要求每块显著**）
    - min_over_median：块均 IC 的最小值 / 中位数（>0 说明无反向块）
    - blk_std / blk_mean：块间离散度相对水平
    这一类统计量是**顺序相关**的，故块内置换零分布对它是有效的
    （与 conc_top10 相反，后者对顺序不变，时间置换无效）。
    """
    v = np.asarray(adj, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < n_blocks * 60:
        return {}
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    means = np.array([v[edges[i]:edges[i + 1]].mean()
                      for i in range(n_blocks) if edges[i + 1] > edges[i]])
    nb = len(means)
    if nb < 3:
        return {}
    overall = float(v.mean())
    s = 1.0 if overall >= 0 else -1.0
    same = int((np.sign(means) == s).sum())
    med = float(np.median(means))
    # 分母护栏：|中位数| 过小时该比值会爆炸（第三次踩到同一类分母不稳定问题）
    ratio = float(np.min(means) / med) if abs(med) > 1e-6 else float("nan")
    return {"sc_blocks": nb, "sc_same_sign": same,
            "sc_same_frac": same / nb, "sc_min_over_median": ratio,
            "sc_blk_std": float(np.std(means, ddof=1)),
            "sc_blk_std_rel": float(np.std(means, ddof=1) / abs(overall))
            if abs(overall) > 1e-9 else float("nan"),
            "sc_mean": float(np.mean(means))}


def all_metrics(ic_raw: np.ndarray, blk: int) -> dict:
    """对一个因子的方向校正日 IC 序列计算全部诊断量。"""
    v = ic_raw[np.isfinite(ic_raw)]
    if len(v) < 50:
        return {}
    ic_mean = float(v.mean())
    adj = ic_raw * (1.0 if ic_mean >= 0 else -1.0)   # 方向校正：衰减 = 正向趋零
    m = {
        "n_days": int(len(v)),
        "ic_mean": ic_mean,
        "t_nw": float(E.nw_t(v)),
        "sd_daily": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        "conc_top10": ic_concentration(adj),
        "jack10": jackknife_top10(adj, 0.10),
        "jack05": jackknife_top10(adj, 0.05),
    }
    sl, st = nw_slope_t(adj)
    m["slope_per_day"], m["slope_t"] = sl, st
    m.update(ic_acf(adj))
    for nb in (5, 8):
        m.update({f"nb{nb}_{k}": v for k, v in sign_consistency(adj, nb).items()})
    dlt, dt = nw_mean_diff_t(adj, w=blk)
    m[f"delta_{blk}"] = dlt
    m[f"delta_{blk}_t"] = dt
    m[f"ic_last{blk}"] = float(adj[np.isfinite(adj)][-blk:].mean())
    bp = block_profile(adj, blk)
    m.update({"n_blocks": bp["n_blocks"], "worst_blk": bp["worst"],
              "best_blk": bp["best"], "pos_blk_frac": bp["pos_frac"],
              "max_consec_neg_blk": bp["max_consec_neg"],
              "blk_coverage": bp["coverage"]})
    return m


def null_ic(ic_raw: np.ndarray, perm_block: int,
            rng: np.random.Generator) -> np.ndarray:
    """块内置换后的日 IC 序列（保留边际分布与短期自相关，破坏时间结构）。

    注意：**对顺序无关的统计量（如 conc_top10）此变换不变**，故其时间置换
    零分布是退化的，不能用作该量的校准——那类量需要截面内置换零分布。
    """
    v = ic_raw[np.isfinite(ic_raw)]
    segs = [v[i:i + perm_block] for i in range(0, len(v), perm_block)]
    order = rng.permutation(len(segs))
    return np.concatenate([segs[i] for i in order])


def null_distribution(ic_raw: np.ndarray, blk: int, n_perm: int,
                      perm_block: int, rng: np.random.Generator) -> pd.DataFrame:
    """块内置换零分布：保留边际与短期自相关，破坏时间结构。"""
    v = ic_raw[np.isfinite(ic_raw)]
    if len(v) < 50:
        return pd.DataFrame()
    rows = []
    for _ in range(n_perm):
        m = all_metrics(null_ic(v, perm_block, rng), blk)
        if m:
            rows.append(m)
    return pd.DataFrame(rows)


def analytic_ic_null(n_dates: int, n_per_day: float, n_perm: int,
                     rng: np.random.Generator) -> np.ndarray:
    """H0（因子与标签无关）下日 RankIC 序列的解析零分布。

    两侧独立随机排名的 Spearman ρ：均值 0、方差 ≈ 1/(n−3)。故按天独立抽
    N(0, 1/√(n_t−3)) 即可，代价可忽略（真实截面置换的百万次行操作 → 这里
    是 n_dates×n_perm 次抽样）。

    用途：校准**对时间顺序不变**的量（conc_top10 / top10_contrib）——时间
    置换对它们完全无效（实测变动 0.00e+00），而逐因子真实置换又慢 30 倍以上。
    解析近似的准确性由 `crosssec_null` 的少量真实置换抽样验证。
    """
    sd = 1.0 / np.sqrt(max(n_per_day - 3.0, 1.0))
    return rng.normal(0.0, sd, size=(n_perm, n_dates))


def crosssec_null(panel: pd.DataFrame, factor: str, n_perm: int,
                  rng: np.random.Generator) -> np.ndarray:
    """截面内置换的日 IC 序列（真正的"该因子与标签无关"零分布）。

    日内打乱因子排名 → 重算逐日 RankIC。用于校准**对时间顺序不变**的量
    （conc_top10 / top10_contrib）——时间置换对它们无效（见 `ic_concentration`
    的警告）。返回长度 n_perm 的 conc/contrib 数组。
    """
    day, dates, x, y, n_days = E._prepared(panel, factor, "r3")
    bounds = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    blocks = [slice(bounds[k], bounds[k + 1] if k + 1 < len(bounds) else len(day))
              for k in range(len(bounds))]
    out_c, out_t = [], []
    for _ in range(n_perm):
        xr = x.copy()
        for b in blocks:
            seg = xr[b]
            f = seg[np.isfinite(seg)]
            if len(f):
                xr[b][np.isfinite(seg)] = f[rng.permutation(len(f))]
        r = E._daily_corr(xr, y, day, n_days)
        r = r[np.isfinite(r)]
        if len(r) < 50:
            continue
        adj = r * (1.0 if r.mean() >= 0 else -1.0)
        out_c.append(ic_concentration(adj))
        out_t.append(top10_contrib(adj))
    return np.asarray(out_c), np.asarray(out_t)


# ------------------------------------------------------------------ 面板
def load_roles(role: str = "STOCK") -> list[str]:
    d = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
    return list(d.loc[d["role"] == role, "name"])


def load_labels(label_max: str) -> pd.DataFrame:
    lab = B.read_labels(["r3", "tradable"], label_max=label_max, label="stab_diag")
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab[(lab["tradable"] == 1) & (lab["date"] <= label_max)]
    return lab[["stock_code", "date", "r3"]]


def ic_matrix(names: list[str], label_max: str, chunk: int = CHUNK,
              progress: bool = True, cache: bool = True) -> pd.DataFrame:
    """分块读取因子并计算逐日 IC，返回 date × factor 的 IC 矩阵。

    结果按窗口缓存到 `stability_ic_<start>_<end>.parquet`，避免指标迭代时
    反复读取 627 万行 × 124 列的 factors.parquet。
    """
    tag = f"{W.FACTORS_FIRST_SIGNAL_DATE}_{label_max}"
    cp = os.path.join(RESULTS, f"stability_ic_{tag}.parquet")
    if cache and os.path.isfile(cp):
        mat = pd.read_parquet(cp)
        mat.index = pd.to_datetime(mat.index)
        keep = [c for c in names if c in mat.columns]
        missing = [c for c in names if c not in mat.columns]
        if not missing:
            if progress:
                print(f"  复用 IC 缓存 {os.path.basename(cp)} {mat.shape}")
            return mat[keep]
        names = [c for c in names if c in mat.columns] + missing
        base = mat
    else:
        base = None

    todo = names if base is None else [c for c in names if c not in base.columns]
    lab = load_labels(label_max)
    keys = lab[["stock_code", "date"]]
    out: dict[str, pd.Series] = {}
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        fac = B.read_factors(part, data_max=W.LABEL_MAX_DATE, label="stab_diag")
        fac["date"] = pd.to_datetime(fac["date"])
        m = keys.merge(fac, on=["stock_code", "date"], how="inner")
        del fac
        m = m.merge(lab, on=["stock_code", "date"], how="inner")
        m = m[m["date"] <= label_max]
        for c in part:
            if c not in m.columns:
                continue
            s = E.rankic_series(m, c, "r3")
            if len(s):
                out[c] = s
        del m
        if progress:
            print(f"  IC 进度 {min(i + chunk, len(todo))}/{len(todo)}", flush=True)
    if not out and base is not None:
        return base
    new = pd.DataFrame(out).sort_index()
    new.index = pd.to_datetime(new.index)
    mat = new if base is None else base.join(new, how="outer")
    mat = mat[sorted(c for c in mat.columns if c in set(names))]
    if cache:
        mat.to_parquet(cp)
        if progress:
            print(f"  IC 缓存 → {os.path.basename(cp)} {mat.shape}")
    return mat


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="因子时间稳定性诊断（纯诊断）")
    ap.add_argument("--block", type=int, default=252, help="块长（交易日）")
    ap.add_argument("--perm", type=int, default=1000, help="每因子置换次数")
    ap.add_argument("--perm-block", type=int, default=5,
                    help="置换块长（须覆盖 R3 的 3 日标签重叠）")
    ap.add_argument("--role", default="STOCK")
    ap.add_argument("--window", default="legacy", choices=("legacy", "prod"),
                    help="legacy=设计窗(默认，封存留出)；prod=生产窗(会破坏留出)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个因子")
    ap.add_argument("--xsec-n", type=int, default=30,
                    help="做截面内置换零分布的因子数（对顺序无关量必需）")
    ap.add_argument("--xsec-perm", type=int, default=1000,
                    help="解析零分布的模拟次数")
    ap.add_argument("--xsec-verify", type=int, default=40,
                    help="真实截面置换抽样次数（仅用于验证解析近似）")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    label_max = (W.LEGACY_SEARCH_END if args.window == "legacy"
                 else W.SIGNAL_MAX_DATE)
    print(f"窗口：{W.FACTORS_FIRST_SIGNAL_DATE} ~ {label_max} "
          f"(标签可得上界 {W.LABEL_MAX_DATE})  块长={args.block}  "
          f"置换={args.perm}×block{args.perm_block}")

    names = load_roles(args.role)
    if args.limit:
        names = names[:args.limit]
    print(f"因子（role={args.role}）：{len(names)}")

    mat = ic_matrix(names, label_max)
    if mat.empty:
        print("无 IC 序列，终止")
        return 1
    print(f"IC 矩阵 {mat.shape}  {mat.index.min().date()} ~ {mat.index.max().date()}")

    # 与既有 H2 记录对账，确认本模块面板构造与 H2 一致
    h2_path = os.path.join(RESULTS, "h2_stock_all.csv")
    audit = ""
    if os.path.isfile(h2_path):
        h2 = pd.read_csv(h2_path).set_index("factor")
        common = [c for c in mat.columns if c in h2.index]
        mine = mat[common].mean()
        d = (mine - h2.loc[common, "rankic_mean"]).abs()
        audit = (f"对账 H2：{len(common)} 个因子，|Δrankic_mean| 中位 "
                 f"{float(d.median()):.2e}，最大 {float(d.max()):.2e}")
        print(audit)

    rng = np.random.default_rng(args.seed)
    recs, nulls = [], []
    for i, c in enumerate(mat.columns, 1):
        m = all_metrics(mat[c].to_numpy(dtype=float), args.block)
        if not m:
            continue
        m["factor"] = c
        nd = null_distribution(mat[c].to_numpy(dtype=float), args.block,
                               args.perm, args.perm_block, rng)
        if len(nd):
            nd["factor"] = c
            nulls.append(nd)
        recs.append(m)
        if i % 20 == 0 or i == len(mat.columns):
            print(f"  诊断进度 {i}/{len(mat.columns)}", flush=True)

    res = pd.DataFrame(recs)
    res["window_start"] = W.FACTORS_FIRST_SIGNAL_DATE
    res["window_end"] = label_max
    if nulls:
        nul = pd.concat(nulls, ignore_index=True)
        # 经验 p 值：真实值在零分布中的双侧位置
        for col in ("conc_top10", "slope_t", f"delta_{args.block}_t",
                    "max_consec_neg_blk", "pos_blk_frac"):
            if col in res.columns and col in nul.columns:
                pooled = nul[col].to_numpy(dtype=float)
                pooled = pooled[np.isfinite(pooled)]
                obs = res[col].to_numpy(dtype=float)
                p = np.array([(1.0 + np.sum(np.abs(pooled - np.median(pooled))
                                             >= abs(v - np.median(pooled))))
                              / (len(pooled) + 1) if np.isfinite(v) else np.nan
                              for v in obs])
                res[f"p_{col}"] = p
        nul.to_csv(os.path.join(RESULTS, "stability_diag_null.csv"),
                   index=False, encoding="utf-8")
    res.to_csv(os.path.join(RESULTS, "stability_diag.csv"),
               index=False, encoding="utf-8")

    # ---- 校准核查：时间置换零分布的触发率 vs 实际触发率 ----
    if nulls:
        nul_all = pd.concat(nulls, ignore_index=True)
        print("\n===== 校准核查（时间置换零分布）=====")
        print(f"{'统计量':<16s}{'规则':>10s}{'实际':>9s}{'零分布':>9s}   判定")
        for col, thr, lab in ((f"slope_t", -2, "t < -2"),
                              (f"delta_{args.block}_t", -2, "t < -2"),
                              ("max_consec_neg_blk", 2, ">=2块")):
            if col not in res.columns:
                continue
            if col == "max_consec_neg_blk":
                obs = float((res[col] >= thr).mean() * 100)
                nulr = float((nul_all[col] >= thr).mean() * 100)
            else:
                obs = float((res[col] < thr).mean() * 100)
                nulr = float((nul_all[col] < thr).mean() * 100)
            verdict = "校准" if abs(obs - nulr) < 3 else "偏离(该统计量无鉴别力)"
            print(f"{col:<16s}{lab:>10s}{obs:8.1f}%{nulr:8.1f}%   {verdict}")
        print("注：delta_252_t 用相邻两段窗口，块内置换破坏连续性 → 零分布畸宽；"
              "实测双侧触发率≈名义值的 10 倍，属无鉴别力统计量，不可用作判据。")

    # ---- 预测力集中度：jackknife（因子间横向比较，不需 H0 校准）----
    print("\n===== 预测力集中度（jackknife）=====")
    print("  丢弃 |IC| 最大的若干天后，剩余 IC 均值 / 原均值；越接近 1 越不依赖少数几天")
    for col, labq in (("jack10", "10%"), ("jack05", "5%")):
        if col not in res:
            continue
        qv = np.nanquantile(res[col], [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"  丢弃 {labq:>3s} 天 分位(5/25/50/75/95%): "
              + " ".join(f"{x:+.3f}" for x in qv))
        print(f"          <0.5（严重依赖少数几天）: {int(np.nansum(res[col] < 0.5))}"
              f"/{len(res)}；<0（去掉后反向）: {int(np.nansum(res[col] < 0))}/{len(res)}")

    # ---- 因子间日 IC 相关结构 ----
    cs = mat.corr(min_periods=200)
    iu = np.triu_indices_from(cs.values, k=1)
    pair = cs.values[iu]
    pair = pair[np.isfinite(pair)]
    print("\n===== 因子间日 IC 相关结构 =====")
    print(f"  两两日 IC 相关: 中位 {np.median(pair):+.3f}  "
          f"90分位 {np.quantile(pair, .9):+.3f}  "
          f"|rho|>0.3 比例 {np.mean(np.abs(pair) > 0.3) * 100:.1f}%")
    ev = np.linalg.eigvalsh(np.nan_to_num(cs.values))[::-1]
    print(f"  第一主成分解释方差 {ev[0] / len(cs) * 100:.1f}%"
          f"（前5主成分 {ev[:5].sum() / len(cs) * 100:.1f}%）")

    # ---- 逐因子自相关与有效样本量 ----
    print("\n===== 日 IC 自相关与有效样本量 =====")
    print(f"  acf1 中位 {res['acf1'].median():+.3f}  "
          f"10分位 {res['acf1'].quantile(.1):+.3f}  "
          f"90分位 {res['acf1'].quantile(.9):+.3f}")
    print(f"  acf5 中位 {res['acf5'].median():+.3f}")
    r_med = float(res["n_eff_ratio"].median())
    nd = int(res["n_days"].median())
    print(f"  n_eff/n 中位 {r_med:.3f}  10分位 {res['n_eff_ratio'].quantile(.1):.3f}")
    print(f"  → 名义 {nd} 天，实际有效样本 ≈ {r_med * nd:.0f} 天（中位因子）")
    print(f"  含义：块均 IC 的抽样误差比按独立样本估计大约 "
          f"{1 / max(r_med, 1e-9):.1f} 倍，这是逐因子时间稳定性难以测量的首要原因。")

    if audit:
        print(f"\n对账：{audit}")
    print("\n输出：stability_diag.csv / stability_diag_null.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
