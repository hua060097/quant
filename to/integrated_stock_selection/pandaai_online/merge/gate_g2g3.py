# -*- coding: utf-8 -*-
"""G2/G3 实现（MERGE_V2_PLAN §8.2）——此前的 `NotImplementedError` 空壳。

## 口径（严格按 §8.2 原文，不做自由度折算）

- **G2 对齐**：对**全部通过卫生门的 STOCK 因子**，计算条件斜率
  `RankIC_base,t = a + b·g_c + e_t` 的中位数；通过标准：
    (a) **4 折同号**（评估窗等分 4 个连续折，各折的中位斜率同号）
    (b) **95% 区间不含 0**
    (c) **同号因子占比 ≥ 60%**
- **G3 块置换**：块置换门控序列 200 次，重算 G2 中位数 → 零分布；
  通过标准：真实值超过零分布 **95 分位**。

## 为什么不用 n_eff 折算（Codex 纠正，已采纳）

`n_eff ≈ 1.5` 时 `df = n_eff − 1 ≈ 0.5` 没有合理解释，且 n_eff 估计本身不稳定。
**§8.2 原文写的"块自助法（块长 5）"本身就是正确的解法**：对**时间块**重采样，
同一次重采样应用于全部因子 → 因子间相关性被自然保留，无需额外自由度折算。
本实现即按此原文执行。

用法：
    python gate_g2g3.py --ic-cache results/merge_v2/ic_stock_2025.parquet
    python gate_g2g3.py --build-ic          # 先构造 IC 矩阵（约 4 分钟）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boundary as B  # noqa: E402
import eval_v2 as E  # noqa: E402
import fastranks as FR  # noqa: E402
import gates as G  # noqa: E402
import windows as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")
CUT = W.FACTOR_SCREEN_MAX
IC_CACHE = os.path.join(RESULTS, "ic_stock_2025.parquet")

G2_FOLDS = 4
G2_SAME_SIGN_MIN = 0.60
G2_BOOT_BLOCK = 5
G2_BOOT_N = 500
G3_PERM = 200
G3_PLACEBO_BLOCK = 20


def build_ic(names: list[str], verbose: bool = True) -> pd.DataFrame:
    """124 个 STOCK 因子的日 IC 矩阵（新选择窗），缓存到 parquet。

    roles.csv 里有个别因子不在 factors.parquet（如 `price_impact`），先按
    实际列名过滤，避免 KeyError。
    """
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(os.path.join(ROOT, "factor_lab_data",
                                           "factors.parquet")).schema_arrow.names)
    miss = [c for c in names if c not in have]
    if miss:
        print(f"  跳过 {len(miss)} 个不在 factors.parquet 的因子: {miss[:6]}"
              f"{'…' if len(miss) > 6 else ''}")
    names = [c for c in names if c in have]
    lab = B.read_labels(["r3", "tradable"], label_max=CUT, label="g2ic")
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab[(lab["tradable"] == 1) & (lab["date"] <= CUT)]
    out = {}
    idx_ref = None
    for i in range(0, len(names), 16):
        part = names[i:i + 16]
        fac = B.read_factors(part, data_max=W.LABEL_MAX_DATE, label="g2ic")
        fac["date"] = pd.to_datetime(fac["date"])
        m = lab[["stock_code", "date", "r3"]].merge(fac, on=["stock_code", "date"],
                                                    how="inner")
        m = m[m["date"] <= CUT]
        del fac
        # 每个 chunk 单独建 PanelRanker：_df 只含本 chunk 的列
        # （复用同一个 ranker 会因缺列而 KeyError）
        pr = FR.PanelRanker(m, verbose=(verbose and i == 0))
        for c in part:
            if c not in m.columns:
                continue
            out[c] = pr.ic_series(pr.x_rank(c)).to_numpy(dtype=float)
        if idx_ref is None:
            idx_ref = pd.to_datetime([pd.Timestamp(u) for u in pr.uniq])
        del m, pr
        if verbose:
            print(f"  IC {min(i + 16, len(names))}/{len(names)}", flush=True)
    return pd.DataFrame(out, index=idx_ref)


def per_factor_slopes(ic: np.ndarray, g: np.ndarray) -> np.ndarray:
    """逐因子 OLS 斜率 b（RankIC_t = a + b·g_t），NaN 成对剔除。ic 形状 [T,F]。"""
    T, F = ic.shape
    out = np.full(F, np.nan)
    gv = g
    for j in range(F):
        y = ic[:, j]
        ok = np.isfinite(y) & np.isfinite(gv)
        if ok.sum() < 60:
            continue
        x = gv[ok]
        yy = y[ok]
        xm = x.mean()
        sxx = ((x - xm) ** 2).sum()
        if sxx <= 0:
            continue
        out[j] = float(((x - xm) * (yy - yy.mean())).sum() / sxx)
    return out


def g2_stat(ic: np.ndarray, g: np.ndarray) -> dict:
    """G2 点估计：中位斜率 + 同号占比 + 4 折同号。"""
    b = per_factor_slopes(ic, g)
    fin = np.isfinite(b)
    if fin.sum() < 10:
        return {"median_b": np.nan, "same_frac": np.nan, "folds_same": 0,
                "n_factors": int(fin.sum()), "pass": False}
    bf = b[fin]
    med = float(np.median(bf))
    sgn = np.sign(med) if med != 0 else 1.0
    same_frac = float((np.sign(bf) == sgn).mean())
    T = len(g)
    edges = np.linspace(0, T, G2_FOLDS + 1).astype(int)
    folds_same = 0
    for i in range(G2_FOLDS):
        sl = slice(edges[i], edges[i + 1])
        bb = per_factor_slopes(ic[sl], g[sl])
        bb = bb[np.isfinite(bb)]
        if len(bb) >= 5 and np.sign(np.median(bb)) == sgn:
            folds_same += 1
    return {"median_b": med, "same_frac": same_frac, "folds_same": int(folds_same),
            "n_factors": int(fin.sum()), "pass": None}


def ci_excludes_zero(lo: float, hi: float) -> bool:
    """95% 区间**双侧**判定：区间整体落在 0 的任一侧即「与 0 有区别」。

    **为什么必须双侧（2026-10-12 修复）**：原实现在 5 处都只判 `lo > 0`（正向），
    会把**稳定负向**的调节误判为「与 0 无区别」。实测证据：14 个行业门控的
    ΔIC 中位全为负（−0.0273 ~ −0.0278，同号 1.00、4 折全 4/4），两个季度区间
    整体为负（CI ≈ [−0.0335, −0.0327]），却因 `lo > 0` 被判 **G2 全灭**；
    而同一批门控的 G3（早已修为双侧）**5/5 通过** —— G2 与 G3 结论直接自相矛盾。

    这与 sector G3 注释里已写明的教训同源（「单侧会把 M1 式负向效应误判为
    '未超零分布'」），但当时只修了 sector 的 G3，G2（两份文件三处）与市场 G3 漏改。
    **负向调节同样是真实信号**（门控在坏时段关掉 alpha），必须与正向同等对待。
    """
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def g2_ci_bootstrap(ic: np.ndarray, g: np.ndarray, block: int = G2_BOOT_BLOCK,
                    n_boot: int = G2_BOOT_N, seed: int = 0) -> tuple:
    """**块自助法（块长 5）**：对时间块重采样（同一次重采样用于全部因子），
    重算中位斜率 → 95% 区间。因子间相关性由时间块结构自然保留。"""
    T = len(g)
    nb = max(1, T // block)
    edges = np.arange(0, T + 1, block)
    blocks = [slice(edges[i], min(edges[i] + block, T)) for i in range(len(edges) - 1)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), size=nb)
        idx = np.concatenate([np.arange(blocks[j].start, blocks[j].stop)
                              for j in pick])
        out[i] = np.nanmedian(per_factor_slopes(ic[idx], g[idx]))
    out = out[np.isfinite(out)]
    if len(out) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def per_factor_delta(ic: np.ndarray, g_on: np.ndarray) -> np.ndarray:
    """逐因子 ΔIC = mean(IC | on) − mean(IC | 全部)。ic [T,F]，g_on 为 bool。"""
    T, F = ic.shape
    out = np.full(F, np.nan)
    for j in range(F):
        y = ic[:, j]
        ok = np.isfinite(y)
        if ok.sum() < 60:
            continue
        m = ok & g_on
        if m.sum() < 20:
            continue
        out[j] = float(y[m].mean() - y[ok].mean())
    return out


def g2_delta_stat(ic: np.ndarray, g_on: np.ndarray) -> dict:
    """G2 的**阈值形式变体**（与 F1 组合形式对齐）。

    口径问题（2026-10-10 发现）：§8.2 的 G2 写的是"条件斜率中位数"（线性），
    而 F1 组合用的是 THRESH 阈值对比。若关系非单调，两者符号可以相反——
    实测 `GM_limit_ratio_ratio` 线性斜率 −0.0035（4/4 折同为负），
    但 THRESH ΔIC +0.0175。故并列提供本变体，供裁决用哪个作 G2 口径。
    """
    d = per_factor_delta(ic, g_on)
    fin = np.isfinite(d)
    if fin.sum() < 10:
        return {"median_d": np.nan, "same_frac": np.nan, "folds_same": 0,
                "n_factors": int(fin.sum())}
    df_ = d[fin]
    med = float(np.median(df_))
    sgn = np.sign(med) if med != 0 else 1.0
    same_frac = float((np.sign(df_) == sgn).mean())
    T = len(g_on)
    edges = np.linspace(0, T, G2_FOLDS + 1).astype(int)
    folds_same = 0
    for i in range(G2_FOLDS):
        sl = slice(edges[i], edges[i + 1])
        bb = per_factor_delta(ic[sl], g_on[sl])
        bb = bb[np.isfinite(bb)]
        if len(bb) >= 5 and np.sign(np.median(bb)) == sgn:
            folds_same += 1
    return {"median_d": med, "same_frac": same_frac, "folds_same": int(folds_same),
            "n_factors": int(fin.sum())}


def g2_delta_bootstrap(ic: np.ndarray, g_on: np.ndarray,
                       block: int = G2_BOOT_BLOCK, n_boot: int = G2_BOOT_N,
                       seed: int = 0) -> tuple:
    """阈值形式变体的块自助 95% 区间（时间块重采样，同一次用于全部因子）。"""
    T = len(ic)
    nb = max(1, T // block)
    edges = np.arange(0, T + 1, block)
    blocks = [slice(edges[i], min(edges[i] + block, T)) for i in range(len(edges) - 1)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), size=nb)
        idx = np.concatenate([np.arange(blocks[j].start, blocks[j].stop)
                              for j in pick])
        out[i] = np.nanmedian(per_factor_delta(ic[idx], g_on[idx]))
    out = out[np.isfinite(out)]
    if len(out) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def g3_placebo(ic: np.ndarray, g: np.ndarray, n_perm: int = G3_PERM,
               block: int = G3_PLACEBO_BLOCK, seed: int = 0) -> dict:
    """G3：块置换门控 → G2 中位数的零分布；**单侧**判定（§8.2 冻结规格）。

    规格原文：「真实值**超过**零分布 95% 分位」⇒ 单侧 `obs > q95`。

    **2026-10-12 规则回退记录**：本函数曾在同轮内被改为双侧
    （`obs > q95 or obs < q05`），理由是「单侧会把稳定负效应判死、与 G2 的双侧口径不一致」。
    但原规格明确为**单侧**，该改动属**研究规则变更**而非 bug 修复；按 §8.2.4
    「规则冻结：运行后若下游结果促使改规则，须作为新的流程版本」，
    **已回退为单侧**。双侧量保留为**诊断输出**（`q05` / `p_two_sided` /
    `pass_two_sided_diag`），**不参与 `pass` 判定**。
    若将来要做双侧研究，须作为独立的新版本**在查看结果前**预声明统计量、
    零假设、p 值算法、显著性阈值与多重检验处理。

    **p 值口径（本轮统一）**：与 `pass` 一致，报告 `p_one_sided = (1+k)/(N+1)`，
    其中 `k = #{null ≥ obs}`（有限置换校正）。`p_two_sided` 仅作诊断，不用于判定。
    """
    obs = float(np.nanmedian(per_factor_slopes(ic, g)))
    rng = np.random.default_rng(seed)
    fin = np.isfinite(g)
    gf = g[fin]
    nf = len(gf)
    nb = max(1, nf // block)
    pieces = np.array_split(gf.copy(), nb)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(nb)
        gp = np.full(len(g), np.nan)
        gp[fin] = np.concatenate([pieces[j] for j in order])[:nf]
        nulls[i] = np.nanmedian(per_factor_slopes(ic, gp))
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls) == 0 or not np.isfinite(obs):
        return {"obs": obs, "q05": float("nan"), "q95": float("nan"),
                "pass": False, "null_n": 0}
    q05 = float(np.quantile(nulls, 0.05))
    q95 = float(np.quantile(nulls, 0.95))
    k1 = int((nulls >= obs).sum())                       # 单侧（与 pass 一致）
    k2 = int((np.abs(nulls) >= abs(obs)).sum())          # 双侧（仅诊断）
    return {"obs": obs, "q05": q05, "q95": q95,
            "pass": bool(obs > q95),                     # 单侧 = §8.2 冻结规格
            "null_n": int(len(nulls)),
            "p_one_sided": float((1 + k1) / (len(nulls) + 1)),
            "p_two_sided": float((1 + k2) / (len(nulls) + 1)),          # 仅诊断
            "pass_two_sided_diag": bool(obs > q95 or obs < q05)}        # 仅诊断


def main() -> int:
    ap = argparse.ArgumentParser(description="G2/G3 组级门控资格（§8.2）")
    ap.add_argument("--build-ic", action="store_true")
    ap.add_argument("--ic-cache", default=IC_CACHE)
    ap.add_argument("--gates", nargs="*", default=None,
                    help="默认跑搜索空间的 eligible 市场门控（*2026-10-12 起旧手工门控"
                         " M1–M4 已删除，门控定义由搜索生成*）")
    ap.add_argument("--out", default=os.path.join(SPACE, "g2g3.csv"))
    ap.add_argument("--n-boot", type=int, default=G2_BOOT_N)
    ap.add_argument("--n-perm", type=int, default=G3_PERM)
    ap.add_argument("--by-group", action="store_true", default=False,
                    help="V4.3c 机制分簇G2（默认关）：按 roles.csv group 先验簇分别跑斜率"
                         "口径，任一簇 4 折同号+CI不含0+簇内同号>=0.65 即认证并记录受益簇；"
                         "旧全池结论保留，新检验族版本")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    if a.build_ic or not os.path.isfile(a.ic_cache):
        roles = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
        names = list(roles.query("role=='STOCK'")["name"])
        print(f"构造 IC 矩阵（{len(names)} 因子）…", flush=True)
        icdf = build_ic(names)
        icdf.to_parquet(a.ic_cache)
        print(f"→ {a.ic_cache}  {icdf.shape}  {time.time() - t0:.0f}s", flush=True)
    else:
        icdf = pd.read_parquet(a.ic_cache)
        icdf.index = pd.to_datetime(icdf.index)
        print(f"IC 矩阵 {icdf.shape} 复用自 {os.path.basename(a.ic_cache)}")

    # **方向对齐**（§8.4.1：方向 = sign(全样本 RankIC 均值)）。
    # 不做这一步，组级统计会被"原始符号"污染：base20 中 16/20 方向为 −1，
    # 未定向时组中位斜率的符号会整体翻转（实测曾导致 M4/GM_limit_ratio_ratio
    # 的中位 ΔIC 由 + 变 −）。
    orient = np.sign(icdf.mean(axis=0, skipna=True).to_numpy(dtype=float))
    orient[orient == 0] = 1.0
    icdf = icdf.mul(orient, axis=1)
    print(f"  已按 sign(全样本 RankIC 均值) 定向；正号 {int((orient > 0).sum())} / "
          f"负号 {int((orient < 0).sum())}")
    ic = icdf.to_numpy(dtype=float)
    # V4.3c 机制分簇：roles.csv group 先验映射（t2_promoted 单列一簇并注明）。
    groups: dict[str, list[int]] = {}
    if a.by_group:
        try:
            _roles = pd.read_csv(os.path.join(RESULTS, "roles.csv"))
            _gmap = dict(zip(_roles["name"].astype(str), _roles["group"].astype(str)))
            for _j, _c in enumerate(icdf.columns):
                _g = _gmap.get(str(_c), "ungrouped")
                groups.setdefault(_g, []).append(_j)
            print(f"  分簇G2：{len(groups)} 簇 "
                  f"{sorted((g, len(v)) for g, v in groups.items())}", flush=True)
        except Exception as exc:
            print(f"  warning: 分簇映射失败（{str(exc)[:100]}），退化为全池", flush=True)
            groups = {}

    # 门控集合（搜索空间；2026-10-12 起旧手工门控 market_gates.parquet 已删除）
    sp = pd.read_parquet(os.path.join(SPACE, "gate_search_market.parquet"))
    sp["date"] = pd.to_datetime(sp["date"])
    scr = pd.read_csv(os.path.join(SPACE, "screen.csv"))
    elig = scr[(scr["level"] == "market") & (scr["eligible"])]["gate_id"].tolist()

    series = {}
    for gid in elig:
        s = sp[sp["gate_id"] == gid].set_index("date")["p"].reindex(icdf.index)
        series[gid] = 2 * s.to_numpy(dtype=float) - 1
    if a.gates:
        series = {k: v for k, v in series.items() if k in a.gates}
    print(f"门控 {len(series)} 个\n")

    rows = []
    print(f"{'gate':<28s}{'[斜率]中位':>11s}{'同号':>6s}{'4折':>5s}{'CI下界':>10s}"
          f"{'G2slope':>9s}|{'[阈值]ΔIC':>10s}{'同号':>6s}{'4折':>5s}{'CI下界':>10s}"
          f"{'G2thr':>7s}")
    for gid, g in series.items():
        # --- 变体 1：计划原文的线性斜率口径 ---
        st = g2_stat(ic, g)
        lo, hi = g2_ci_bootstrap(ic, g, n_boot=a.n_boot)
        g3 = g3_placebo(ic, g, n_perm=a.n_perm)
        ok_slope = (st["folds_same"] == G2_FOLDS) and ci_excludes_zero(lo, hi) \
            and st["same_frac"] >= G2_SAME_SIGN_MIN
        # --- 变体 1b：机制分簇斜率（V4.3c，新检验族版本，默认关） ---
        _grp, _grp_ok = "", False
        _grp_med, _grp_same, _grp_folds = float("nan"), float("nan"), 0
        if groups:
            _best = None
            for _gn, _cols in groups.items():
                if len(_cols) < 5:
                    continue
                _sic = ic[:, _cols]
                _sst = g2_stat(_sic, g)
                _lo, _hi = g2_ci_bootstrap(_sic, g, n_boot=a.n_boot)
                _ok = (_sst["folds_same"] == G2_FOLDS) \
                    and ci_excludes_zero(_lo, _hi) \
                    and _sst["same_frac"] >= 0.65
                if _ok and (_best is None or abs(_sst["median_b"]) > abs(_best[1])):
                    _best = (_gn, _sst["median_b"], _sst["same_frac"],
                             _sst["folds_same"], _lo, _hi)
            if _best is not None:
                _grp, _grp_med, _grp_same, _grp_folds = _best[0], _best[1], \
                    _best[2], _best[3]
                _grp_ok = True
                ok_slope = True
        # --- 变体 2：与 F1 组合形式对齐的阈值口径 ---
        g_on = np.isfinite(g) & (g >= 2 * G.THRESH_HI - 1)     # p >= 0.7
        sd_ = g2_delta_stat(ic, g_on)
        dlo, dhi = g2_delta_bootstrap(ic, g_on, n_boot=a.n_boot)
        ok_thr = (sd_["folds_same"] == G2_FOLDS) and ci_excludes_zero(dlo, dhi) \
            and sd_["same_frac"] >= G2_SAME_SIGN_MIN
        rows.append({
            "gate": gid,
            "slope_median": st["median_b"], "slope_same_frac": st["same_frac"],
            "slope_folds": st["folds_same"], "slope_ci_lo": lo, "slope_ci_hi": hi,
            "g2_slope_pass": bool(ok_slope),
            "slope_group": _grp, "slope_group_median": _grp_med,
            "slope_group_same": _grp_same, "slope_group_folds": _grp_folds,
            "slope_group_pass": bool(_grp_ok),
            "median_b_eff": _grp_med if _grp_ok else st["median_b"],
            "thr_delta_median": sd_["median_d"], "thr_same_frac": sd_["same_frac"],
            "thr_folds": sd_["folds_same"], "thr_ci_lo": dlo, "thr_ci_hi": dhi,
            "g2_thr_pass": bool(ok_thr),
            "n_factors": st["n_factors"],
            "g3_obs": g3["obs"], "g3_q95": g3["q95"], "g3_pass": g3["pass"],
            "g3_p": g3.get("p_one_sided"),
        })
        print(f"{gid:<28s}{st['median_b']:>11.5f}{st['same_frac']:>6.2f}"
              f"{st['folds_same']:>3d}/{G2_FOLDS}{lo:>10.5f}{str(bool(ok_slope)):>9s}|"
              f"{sd_['median_d']:>10.5f}{sd_['same_frac']:>6.2f}"
              f"{sd_['folds_same']:>3d}/{G2_FOLDS}{dlo:>10.5f}{str(bool(ok_thr)):>7s}")

    df = pd.DataFrame(rows).sort_values(
        ["g2_thr_pass", "g2_slope_pass", "g3_pass", "thr_delta_median"],
        ascending=[False, False, False, False])
    df.to_csv(a.out, index=False, encoding="utf-8")
    print(f"\n=== 汇总（{len(df)} 个门控）===")
    print(f"  G2【线性斜率口径】通过: {int(df['g2_slope_pass'].sum())}/{len(df)}")
    print(f"  G2【阈值对齐口径】通过: {int(df['g2_thr_pass'].sum())}/{len(df)}")
    print(f"  G3 通过（超零分布 95 分位）: {int(df['g3_pass'].sum())}/{len(df)}")
    for col, mcol, scol, fcol, lab in (
            ("g2_slope_pass", "slope_median", "slope_same_frac", "slope_folds",
             "斜率口径"),
            ("g2_thr_pass", "thr_delta_median", "thr_same_frac", "thr_folds",
             "阈值口径")):
        w = df[df[col]]
        print(f"  通过明细（{lab}）:")
        if not len(w):
            print("    （无）")
        for _, r in w.iterrows():
            print(f"    {r['gate']:<28s} {r[mcol]:+.5f} 同号 {r[scol]:.2f} "
                  f"4折 {int(r[fcol])}/4")
    print(f"\n输出 → {a.out}  用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
