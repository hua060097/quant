# -*- coding: utf-8 -*-
"""P2 前置验证：行业侧切片的日期对齐 / 成分变化 / 缺失 / 泄漏风险。

MERGE_V2_PLAN §17.9 项 6 要求：行业侧接入统一流水线前，先验证
（Codex 第 3 条）：
  1. 行业门控的值是否正确对齐到股票及日期，是否存在未来信息泄漏；
  2. 行业成分变化、缺失值和行业样本量不足如何处理；
  3. 行业侧的 G2/G3 统计量是否与其实际假设一致，而不是机械复用市场实现。

2026-10-12 适配：门控面板为**搜索空间长表**（date/sector/gate_id/raw/p）；
旧手工门控宽表（gates/sector_gates.parquet，S1_p/S2_p/S3_p）已删除。
成员数不再由旧面板携带，改为由 cache 直算。前视探针用**搜索门控的 raw**
（rolling 变换，与旧 S1_raw 同为 20 日窗，重叠/非重叠对照语义不变）。

本脚本只做**只读验证**，产出 `gates_search/sector_input_verify.md`。

用法：python verify_sector_inputs.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gates as G  # noqa: E402
import windows as W  # noqa: E402

R = HERE.parent.parent / "results" / "merge_v2"
SP = R / "gates_search"
CUT = W.FACTOR_SCREEN_MAX


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = []

    def p(line=""):
        print(line)
        out.append(line)

    gs = pd.read_parquet(SP / "gate_search_sector.parquet")
    gs["date"] = pd.to_datetime(gs["date"])
    sc = pd.read_csv(SP / "screen.csv")
    el = sc[(sc["level"] == "sector") & (sc["eligible"])]["gate_id"].tolist()

    p("# 行业侧接入前验证（P2，只读）")
    p()
    p(f"- 候选总数 {gs['gate_id'].nunique()}；eligible **{len(el)}**")
    p(f"- eligible：`{el}`")
    p(f"- 日期 {gs['date'].min().date()} ~ {gs['date'].max().date()}；"
      f"行业 {gs['sector'].nunique()}；行 {len(gs):,}")
    p()

    p("## 1) 每候选覆盖与稀疏度")
    p()
    p("| gate | 有效p% | 独立值 | 天数 |")
    p("|---|---|---|---|")
    for g in el:
        sub = gs[gs["gate_id"] == g]
        v = pd.to_numeric(sub["p"], errors="coerce")
        p(f"| {g} | {v.notna().mean() * 100:.1f} | "
          f"{v.round(4).nunique()} | {sub['date'].nunique()} |")
    p()

    p("## 2) 日期对齐（与 STOCK IC 矩阵）")
    p()
    ic = pd.read_parquet(R / "ic_stock_2025.parquet")
    ic.index = pd.to_datetime(ic.index)
    gd, idd = set(gs["date"].unique()), set(ic.index)
    p(f"- IC 天数 {len(idd)}；门控天数 {len(gd)}；**交集 {len(gd & idd)}**")
    p(f"- 仅门控有 {len(gd - idd)}；仅 IC 有 {len(idd - gd)}")
    p()

    p("## 3) 行业成分 / 缺失 / 样本量")
    p()
    sm = G.load_sector_map()
    p(f"- 静态 SW2 映射：{len(sm)} 只股票，{sm.nunique()} 个行业")
    # 成员数由 cache 直算（旧手工面板 (date,sector,n) 已删除；2026-10-12 适配）
    c = pd.read_parquet(G.CACHE, columns=["stock_code", "date"])
    c["date"] = pd.to_datetime(c["date"])
    c["_sec"] = c["stock_code"].astype(str).map(sm)
    c = c[c["_sec"].notna() & (c["date"] <= pd.Timestamp(CUT))]
    ncs = c.groupby(["date", "_sec"]).size().rename("n").reset_index()
    n = ncs.groupby("_sec")["n"].median()
    p(f"- 每行业每日股票数：中位 {n.median():.0f}，最小 {n.min():.0f}")
    p(f"- **成分股 <5 只的行业 {int((n < 5).sum())}/{len(n)}**"
      f"（§8.2.4 要求小行业单独报告，不作前置门）")
    sgc = gs[gs["date"] <= pd.Timestamp(CUT)].copy()
    cov = sgc.groupby("sector")["raw"].apply(
        lambda t: 1 - t.isna().mean(), include_groups=False)
    p(f"- 行业 raw 有效覆盖（全体候选）：中位 {cov.median() * 100:.1f}%")
    p()

    p("## 4) 泄漏风险核查")
    p()
    p("- **(a) 静态 SW2 映射**：D5 已记录『静态映射的前视风险』"
      "（用当前行业分类回溯历史）。**已知且已记录，本次不改变其状态。**")
    p("- **(b) 门控值构造**：`p` 由搜索空间『跨行业截面排名』与基础量 rolling 变换"
      "得来，滚动均用 `rolling(…)`，**无 `shift(-k)`**——逐行核对 "
      "`gate_search_space.py`。")
    # 同日相关（若 p 与同日 raw 高度相关属正常）
    # **前视探测必须用不重叠的前向窗**：raw 是 20 日 rolling，其 shift(-1) 与当日窗
    # 重叠 19/20 天，必然高相关，**不能作为泄漏证据**。
    # 正确做法：用 shift(-20) 的非重叠前向窗。
    probe = el[0]
    s2 = sgc[sgc["gate_id"] == probe].sort_values(["sector", "date"]).copy()
    s2["_v"] = pd.to_numeric(s2["raw"], errors="coerce")
    s2["_fwd_ovl"] = s2.groupby("sector")["_v"].shift(-1)     # 重叠窗（误导）
    s2["_fwd_non"] = s2.groupby("sector")["_v"].shift(-20)    # 非重叠前向窗
    d0 = s2[["p", "_v"]].dropna()
    d1 = s2[["p", "_fwd_ovl"]].dropna()
    d2 = s2[["p", "_fwd_non"]].dropna()
    p(f"- 探针门控：`{probe}`（raw 为 20 日 rolling，与旧 S1_raw 同窗语义）")
    if len(d0) > 100:
        p(f"- **(c) 数值探测**：corr(p, 同日 raw) = "
          f"{d0['p'].corr(d0['_v'], method='spearman'):+.3f}（同日，属正常）")
    if len(d1) > 100:
        p(f"- (c) corr(p, 次日 *重叠窗* raw) = "
          f"{d1['p'].corr(d1['_fwd_ovl'], method='spearman'):+.3f}"
          f"　← **重叠 19/20 天，必然高，不作泄漏证据**")
    if len(d2) > 100:
        r2v = d2["p"].corr(d2["_fwd_non"], method="spearman")
        p(f"- **(c) 前视探测（非重叠 shift(-20)）** = {r2v:+.3f}"
          f"　← 越接近 0 越安全（|ρ|<0.1 视为无前视）")
        p(f"  - 判读：{'通过（无前视迹象）' if abs(r2v) < 0.1 else '⚠ 需复查'}")
    p()

    p("## 5) G2/G3 口径差异（Codex 第 3 条：不得机械复用市场实现）")
    p()
    p("| 维度 | 市场门控 | 行业门控 |")
    p("|---|---|---|")
    p("| 门控取值 | 每日一个标量（全市场共享） | 每 (股票, 日) 一个值"
      "（该股所属行业的当日值） |")
    p("| 条件效应 | 日级掩码 `IC_t·1[g_t>0]` | **逐股掩码**，需重建 F1 分数 |")
    p("| G2 的组级聚合 | 123 个 STOCK 因子的条件斜率中位数 | 同（但斜率的估计"
      "须用逐股掩码重建的 IC 序列） |")
    p("| 有效样本（§8.3） | 交易日数 | **行业数（约 131）**，须同时报告"
      "行业内与行业间 |")
    p("| 成本 | 掩码捷径，秒级 | **须重建 F1 后排名**，约 1.5~2 秒/(门控×因子) |")
    p()
    p("**结论**：行业侧必须**保留专用后端**（逐股掩码 + F1 重建），"
      "但**共享**候选指纹、台账、状态机与结果协议（§17.2）。")
    p()

    p("## 6) 与行业搜索线 F1 结果的可对账性")
    p()
    found = False
    for fp in sorted(SP.glob("f1_sector*.csv")):
        d = pd.read_csv(fp)
        if len(d) and "delta_mean" in d.columns:
            p(f"- `{fp.name}`：{len(d)} 候选，delta 中位 {d['delta_mean'].median():+.5f}")
            found = True
    if not found:
        p("- （暂无 `gates_search/f1_sector*.csv`——行业搜索线 F1 尚未运行）")
    p()
    p("→ 接入后须能用同一候选名对账（回归要求，§17.9 项 6）。")

    dst = SP / "sector_input_verify.md"
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n输出 → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
