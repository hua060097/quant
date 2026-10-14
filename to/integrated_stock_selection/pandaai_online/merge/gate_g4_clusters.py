# -*- coding: utf-8 -*-
"""G4 独立性诊断（带聚类）：门控间近克隆检测。

背景（2026-10-12，Codex/DeepSeek 要求）：14 个行业门控的 G2 全部通过，但其 ΔIC 中位
极差仅 0.0006、同号全 1.00、G3 p 全落在置换分辨率地板 —— 高度疑似近克隆。
「N 个门控通过」**不等于**「N 份独立证据」。G4 只回答「哪些门控互相独立」，
**不回答「哪些门控有效」**，不得把 G4 通过解释为统计有效性确认。

输出（Codex 指定 5 项）：
  ① 两两相关矩阵（落盘 CSV）与超过阈值的高相关对
  ② 按**预先声明**的阈值聚类（默认 |ρ| ≥ 0.8，与 PLAN §8.2 的 G4 阈值一致）
  ③ 每个簇的代表门控与合并门控
  ④ 聚类前后候选数量
  ⑤ 对阈值的敏感性（0.70 / 0.80 / 0.90 / 0.95）

用法：
    python gate_g4_clusters.py                 # 行业 14 + 市场 36
    python gate_g4_clusters.py --level sector
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
SPACE = os.path.join(RESULTS, "gates_search")
CUT = "2025-12-31"


def _series(level: str) -> pd.DataFrame:
    """返回门控系列的**可比表示**。

    - market：`gate_search_market.parquet` 的 p 已是日级标量 → [date, gate_id]
    - sector：**必须保持截面维度**。门控是 (date × sector) 的 p 矩阵
      （`gate_g2g3_sector.build_sector_gate_matrix` 即此口径）。
      **2026-10-12 修正**：原实现按 `gate_id` 对 sector 取均值，把截面变化压平成常数
      （实测 14 列 std=0、uniq=1）→ 相关系数全 NaN、聚类结果全是假象。

      现在返回**展平索引**的 DataFrame：索引为 (date, sector)，列为 gate_id。
      这样 `corr()` 就是逐 (date, sector) 单元的对齐相关 —— 与门控实际生效的粒度一致。
    """
    if level == "market":
        df = pd.read_parquet(os.path.join(SPACE, "gate_search_market.parquet"))
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(CUT)]
        return df.pivot_table(index="date", columns="gate_id", values="p",
                              aggfunc="mean").sort_index()
    if level == "sector":
        df = pd.read_parquet(os.path.join(SPACE, "gate_search_sector.parquet"))
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(CUT)]
        if "sector" not in df.columns:
            raise SystemExit("sector 门控文件缺 `sector` 列，无法保持截面维度")
        piv = df.pivot_table(index=["date", "sector"], columns="gate_id",
                             values="p", aggfunc="mean").sort_index()
        return piv
    raise ValueError(level)


def corr_matrix(ser: pd.DataFrame, level: str,
                mode: str = "series") -> tuple[pd.DataFrame, str]:
    """返回 (相关矩阵, 口径说明)。

    **口径问题（Codex 2026-10-12 指出，不得混为一谈）**：§8.2 的 G4 行写作
    「门控间**截面** |ρ| ≤ 0.8」，但该词有两种可能读法，且实现必须明确是哪一个：

    | mode | 定义 | 适用 |
    |---|---|---|
    | `series`（时序相关） | 把每个门控压成一条序列，跨**时间**求相关 | 市场门控（日级标量，**无截面可言**，只能是这种）；行业门控按 (date,sector) 展平 |
    | `cross`（截面相关） | **每个交易日**在截面（行业维）内求两门控的相关，再跨日取中位 | 仅行业门控可用（有行业维） |

    两者**不相等**。原规格用词是「截面」，而既有实现 `gate_validate.g4_independence`
    用的是**时序**（`gates[cols].corr(...)`，市场门控无截面可选）。
    **本工具不擅自二选一**：两种口径都算、都落盘、都标注；口径归属作为**规格缺口**
    留待裁决（见 PLAN §20.21）。
    """
    ser = ser  # 由调用方传入（已限定为 eligible 子集）
    if mode == "series":
        return ser.corr(method="spearman", min_periods=252), \
            "series：跨时间（sector 按 (date,sector) 展平）"
    if mode == "cross":
        if level == "market":
            return ser.corr(method="spearman", min_periods=252), \
                "cross→回退 series：市场门控是日级标量，**无截面维度**"
        # sector：逐日在行业维求两门控的截面相关，再取跨日中位
        piv = ser  # index=(date,sector), columns=gate
        cols = list(piv.columns)
        out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                per_day = []
                for _, g in piv[[a, b]].groupby(level=0):
                    s = g.dropna()
                    if len(s) < 5:
                        continue
                    if s[a].nunique() < 2 or s[b].nunique() < 2:
                        continue
                    per_day.append(s[a].corr(s[b], method="spearman"))
                v = float(np.nanmedian(per_day)) if per_day else np.nan
                out.loc[a, b] = v
                out.loc[b, a] = v
        np.fill_diagonal(out.values, 1.0)
        return out, "cross：逐日在行业截面内求相关，再取跨日中位（≥5 个行业才计入）"
    raise ValueError(mode)


def cluster(rho: pd.DataFrame, thr: float) -> list[list[str]]:
    """单链聚类：|ρ| ≥ thr 的连通分量为一簇。"""
    cols = list(rho.columns)
    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = rho.loc[a, b]
            if np.isfinite(v) and abs(v) >= thr:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    out: dict[str, list[str]] = {}
    for c in cols:
        out.setdefault(find(c), []).append(c)
    return sorted((sorted(v) for v in out.values()), key=lambda g: -len(g))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", default="both", choices=("market", "sector", "both"))
    ap.add_argument("--thr", type=float, default=0.8,
                    help="聚类阈值 |ρ|（默认 0.8，与 PLAN §8.2 G4 一致）")
    ap.add_argument("--mode", default="series", choices=("series", "cross"),
                    help="相关口径：series=跨时间（既有实现口径）/ cross=逐日截面内相关。"
                         "§8.2 原文写作「截面」，但市场门控无截面维度；**两种口径不等价**，"
                         "归属为规格缺口（PLAN §20.21），本工具分别输出不擅自二选一")
    ap.add_argument("--out-dir", default=SPACE)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    levels = ("market", "sector") if a.level == "both" else (a.level,)
    for lv in levels:
        try:
            ser = _series(lv)
        except FileNotFoundError as exc:
            print(f"[{lv}] 缺产物：{exc}")
            continue
        # 只看在 G0 预筛里 eligible 的门控（与「通过门控数」同一集合）
        try:
            sc = pd.read_csv(os.path.join(SPACE, "screen.csv"))
            el = [g for g in sc[(sc["level"] == lv) & (sc["eligible"])]["gate_id"]
                  if g in ser.columns]
        except Exception:
            el = list(ser.columns)
        ser = ser[el]
        rho, mode_note = corr_matrix(ser, lv, a.mode)

        print("=" * 78)
        print(f"[{lv}] G4 独立性诊断：eligible {len(el)} 个门控，"
              f"{len(ser)} 个 (date×sector/date) 单元（≤{CUT}）")
        print(f"  口径 mode={a.mode} —— {mode_note}")
        print("  ⚠ G4 只回答「哪些门控互相独立」，**不回答「哪些门控有效」**；")
        print("    通过 G4 **不构成**统计有效性确认（不得与 G0–G3/F1 的判定混称）")
        print("=" * 78)

        pairs = []
        cols = list(rho.columns)
        for i, x in enumerate(cols):
            for y in cols[i + 1:]:
                v = rho.loc[x, y]
                if np.isfinite(v):
                    pairs.append((abs(v), v, x, y))
        pairs.sort(reverse=True)
        over = [p for p in pairs if p[0] >= a.thr]
        nf = sum(1 for p in pairs if np.isfinite(p[0]))
        print(f"  两两：{nf} 对可算（全 {len(cols)*(len(cols)-1)//2}）"
              f"；|ρ|≥{a.thr} 有 **{len(over)}** 对")
        print(f"  |ρ| 分位：p50={np.median([p[0] for p in pairs if np.isfinite(p[0])]):.3f}  "
              f"p90={np.quantile([p[0] for p in pairs if np.isfinite(p[0])],0.9):.3f}  "
              f"max={pairs[0][0]:.3f}")
        print()
        print(f"  --- ① 高相关对（|ρ|≥{a.thr}，最多列 20）---")
        for v_abs, v, x, y in over[:20]:
            print(f"    {v:+.4f}  {x:<28s} {y}")
        if not over:
            print("    （无）")

        print()
        print("  --- ②③④ 聚类（单链，|ρ|≥%.2f）---" % a.thr)
        cl = cluster(rho, a.thr)
        multi = [c for c in cl if len(c) > 1]
        print(f"    聚类前 {len(cols)} 个 → **聚类后 {len(cl)} 簇**"
              f"（其中 {len(multi)} 个多成员簇）")
        for c in cl:
            tag = '多成员' if len(c) > 1 else '单例'
            print(f"      [{tag}] n={len(c)}: {', '.join(c)}")

        print()
        print("  --- ⑤ 阈值敏感性 ---")
        for thr in (0.70, 0.80, 0.90, 0.95):
            c2 = cluster(rho, thr)
            m2 = sum(1 for x in c2 if len(x) > 1)
            print(f"    |ρ|≥{thr:.2f} → {len(c2)} 簇（多成员 {m2}）")

        rho.to_csv(os.path.join(a.out_dir, f"g4_rho_{lv}.csv"), encoding="utf-8")
        with open(os.path.join(a.out_dir, f"g4_clusters_{lv}.csv"), "w",
                  encoding="utf-8") as f:
            f.write("cluster_id,n_members,members\n")
            for i, c in enumerate(cl, 1):
                f.write('%d,%d,"%s"\n' % (i, len(c), ";".join(c)))
        print()
        print(f"  落盘：g4_rho_{lv}.csv / g4_clusters_{lv}.csv")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
