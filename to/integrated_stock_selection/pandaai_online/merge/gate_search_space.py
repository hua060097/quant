# -*- coding: utf-8 -*-
"""门控搜索空间：把门控从"手工定义"改为"按空间生成 + 系统筛选"。

## 设计依据（来自 7 个手工门控的失效取证，2026-10-10）

| 失效 | 教训 → 空间设计约束 |
|---|---|
| M4（宽度）R²=79% 被市场方向解释 | 市场级基础量**必须方向中性**（截面离散度、大小盘分化、跨行业离散度、集中度） |
| M2（60 日累计收益）本身就是方向 | **排除"累计收益"类构造** |
| M1（波动率水平）效应反向且稳固 | 保留波动量，但改用**变化/加速度**操作化 |
| M3（成交额比）弱通过 | 保留流动性量，换操作化 |
| S1/S2/S3'（行业自身 252 日分位）全零 | 疑因"自身历史分位"抹掉水平信息 → 行业侧改用**跨行业截面排名**归一化 |

## 空间结构

基础量（方向中性）× 操作化 × 分位窗。默认空间产出约 60 个候选，
远高于手工 7 个，且每个维度都与失效教训对应。

## 与既有门控的关系

本模块**只生成候选**，不改动 M1–M4/S1–S3' 的既有定义。生成的候选走
`gate_validate` 的 G0（便宜）→ 存活者才进 F1（贵）。这是把 grammar 的
"快面板粗筛 → 全量复核"两级模式搬到门控线上（见文件头性能说明）。

用法：
    python gate_search_space.py --out-dir results/merge_v2/gates_search
    python gate_search_space.py --list            # 只看空间清单
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
# 2026-10-11 修正：原为硬编码文件名 → 与个股线 / boundary / gates 统一用自动发现，
# 避免缓存重生成后两线静默分裂到不同面板（见 boundary.read_cache docstring）。
CACHE = str(G.CACHE)

# ---------------------------------------------------------------- 空间定义
#: 市场级基础量（**全部方向中性**：不随市场涨跌单调变化）
MARKET_BASES = (
    # (键, 中文名, 说明)
    ("xs_disp", "截面收益离散度", "全市场 pct_chg 的截面标准差 —— 方向中性，度量'机会多少'"),
    ("xs_iqr", "截面收益 IQR", "pct_chg 的 P75−P25 —— 对极值稳健的离散度"),
    ("sec_disp", "跨行业收益离散度", "各行业等权收益的截面标准差 —— 行业分化程度"),
    ("hl_spread", "大小盘分化", "小市值组收益 − 大市值组收益 —— 方向中性（涨跌皆可正）"),
    ("limit_ratio", "涨跌停比", "涨停家数占比 − 跌停家数占比 —— 情绪极端度"),
    ("amt_hhi", "成交额集中度", "个股成交额份额的 HHI —— 资金集中/扩散"),
    ("turn_med", "换手中位数", "全市场成交额/流通市值 的中位数"),
    ("amp_med", "振幅中位数", "(high−low)/close 的中位数"),
    ("vol_med", "个股波动中位数", "个股 20 日收益标准差的中位数"),
    ("newhigh_ratio", "新高占比", "创 60 日新高的个股占比"),
)

#: 行业级基础量
SECTOR_BASES = (
    ("sec_xs_disp", "行业内离散度", "行业内 pct_chg 的截面标准差"),
    ("sec_turn_chg", "行业换手变化", "行业换手率相对 120 日均值的变化"),
    ("sec_amp_med", "行业振幅中位数", "行业内振幅中位数"),
    ("sec_ret_xrank", "行业收益跨行业排名", "行业收益在全部行业中的当日排名（截面归一化）"),
    ("sec_amt_share_chg", "行业成交额份额变化", "行业成交额占全市场比重的变化"),
)

#: 操作化（对基础量的日序列施加）
OPS = (
    ("level", "20 日均值", lambda s: s.rolling(20, min_periods=20).mean()),
    ("chg", "20 日差分", lambda s: s - s.shift(20)),
    ("ratio", "20 日 / 120 日", lambda s: s.rolling(20, min_periods=20).mean()
     / s.rolling(120, min_periods=120).mean().replace(0, np.nan)),
    ("accel", "差分再差分", lambda s: (s - s.shift(10)) - (s.shift(10) - s.shift(20))),
)

#: 分位窗（归一化用）
QWINS = (63, 126, 252)


def sector_base_from_raw(c: pd.DataFrame) -> pd.DataFrame:
    """行业级基础量的正确实现（需要行业等权收益，单独算以免污染内存）。"""
    sec = G.load_sector_map()
    c = c.copy()
    c["_sec"] = c["stock_code"].astype(str).map(sec)
    c = c[c["_sec"].notna()]
    c["pct_chg"] = pd.to_numeric(c["pct_chg"], errors="coerce")
    c["amount"] = pd.to_numeric(c["amount"], errors="coerce")
    c["circ_mv"] = pd.to_numeric(c["circ_mv"], errors="coerce")
    c["_amp"] = (pd.to_numeric(c["high"], errors="coerce")
                 - pd.to_numeric(c["low"], errors="coerce")) \
        / pd.to_numeric(c["close"], errors="coerce").replace(0, np.nan)
    g = c.groupby(["date", "_sec"], sort=True)
    sb = pd.DataFrame({
        "_ret": g["pct_chg"].mean(),
        "sec_xs_disp": g["pct_chg"].std(),
        "sec_amp_med": g["_amp"].median(),
        "_turn": g["amount"].sum() / g["circ_mv"].sum().replace(0, np.nan),
        "_amt": g["amount"].sum(),
    }).reset_index().rename(columns={"_sec": "sector"})
    sb["sec_amt_share"] = sb["_amt"] / sb.groupby("date")["_amt"].transform("sum")
    sb["sec_ret_xrank"] = sb.groupby("date")["_ret"].rank(pct=True)
    sb = sb.sort_values(["sector", "date"]).reset_index(drop=True)
    sb["sec_turn_chg"] = (sb.groupby("sector", sort=False)["_turn"]
                          .transform(lambda s: s.rolling(20, min_periods=20).mean())
                          / sb.groupby("sector", sort=False)["_turn"]
                          .transform(lambda s: s.rolling(120, min_periods=120).mean())
                          .replace(0, np.nan))
    sb["sec_amt_share_chg"] = (sb.groupby("sector", sort=False)["sec_amt_share"]
                               .transform(lambda s: s - s.shift(20)))
    return sb


def generate_space() -> list[dict]:
    """产出空间清单（spec 列表）。"""
    specs = []
    for key, zh, note in MARKET_BASES:
        for op, opzh, _ in OPS:
            specs.append({"gate_id": f"GM_{key}_{op}", "level": "market",
                          "base": key, "base_zh": zh, "op": op, "op_zh": opzh,
                          "note": note})
    for key, zh, note in SECTOR_BASES:
        for op, opzh, fn in OPS:
            if key in ("sec_ret_xrank", "sec_amt_share_chg") and op != "level":
                continue      # 这两个本身就是变化/排名口径，只取 level
            specs.append({"gate_id": f"GS_{key}_{op}", "level": "sector",
                          "base": key, "base_zh": zh, "op": op, "op_zh": opzh,
                          "note": note})
    return specs


def sha12(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description="门控搜索空间生成")
    ap.add_argument("--out-dir", default=os.path.join(RESULTS, "gates_search"))
    ap.add_argument("--list", action="store_true", help="只列空间清单")
    ap.add_argument("--qwin", type=int, default=252, help="归一化分位窗")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    specs = generate_space()
    mk = [s for s in specs if s["level"] == "market"]
    sc = [s for s in specs if s["level"] == "sector"]
    print(f"搜索空间：{len(specs)} 个门控候选（market {len(mk)} / sector {len(sc)}）")
    print(f"  市场基础量 {len(MARKET_BASES)} × 操作化 {len(OPS)} = {len(mk)}")
    print(f"  行业基础量 {len(SECTOR_BASES)} × 操作化（部分受限）= {len(sc)}")
    if a.list:
        for s in specs:
            print(f"  {s['gate_id']:<28s} {s['base_zh']:<14s} {s['op_zh']:<12s} "
                  f"{s['note'][:40]}")
        return 0

    os.makedirs(a.out_dir, exist_ok=True)
    print("\n构造基础量 …", flush=True)
    c = pd.read_parquet(CACHE, columns=[
        "stock_code", "date", "pct_chg", "open", "high", "low", "close",
        "amount", "circ_mv", "turnover_rate", "is_st", "is_can_buy",
        "is_limit_up", "is_limit_down"])
    c["date"] = pd.to_datetime(c["date"])
    t0 = time.time()
    sb = sector_base_from_raw(c)
    print(f"  行业基础量 {sb.shape}  {time.time() - t0:.0f}s", flush=True)
    mb = None
    # 市场基础量（复用同一次读取）
    sec = G.load_sector_map()
    c["_sec"] = c["stock_code"].astype(str).map(sec)
    for k in ("pct_chg", "amount", "circ_mv", "turnover_rate", "high", "low", "close"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c["_r"] = c["pct_chg"]
    c["_amp"] = (c["high"] - c["low"]) / c["close"].replace(0, np.nan)
    g = c.groupby("date", sort=True)
    mb = pd.DataFrame({
        "xs_disp": g["_r"].std(),
        "xs_iqr": g["_r"].quantile(0.75) - g["_r"].quantile(0.25),
        "limit_ratio": g.apply(lambda x: (x["is_limit_up"] == 1).mean()
                               - (x["is_limit_down"] == 1).mean(),
                               include_groups=False),
        "turn_med": g["turnover_rate"].median(),
        "amp_med": g["_amp"].median(),
    })
    c["_mvbin"] = c.groupby("date")["circ_mv"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    piv = c.pivot_table(index="date", columns="_mvbin", values="_r", aggfunc="mean")
    mb["hl_spread"] = (piv.iloc[:, 0] - piv.iloc[:, -1]) if piv.shape[1] >= 3 else np.nan
    c["_share"] = c["amount"] / c.groupby("date")["amount"].transform("sum")
    mb["amt_hhi"] = c.groupby("date")["_share"].apply(lambda s: float((s ** 2).sum()))
    c["_v20"] = c.groupby("stock_code", sort=False)["_r"].transform(
        lambda s: s.rolling(20, min_periods=20).std())
    mb["vol_med"] = c.groupby("date")["_v20"].median()
    c["_hi60"] = c.groupby("stock_code", sort=False)["close"].transform(
        lambda s: s.rolling(60, min_periods=60).max())
    mb["newhigh_ratio"] = c.groupby("date").apply(
        lambda x: float((x["close"] >= x["_hi60"]).mean()), include_groups=False)
    # 跨行业离散度
    sd = sb.groupby("date")["_ret"].std().rename("sec_disp")
    mb = mb.join(sd)
    mb = mb.reset_index()
    del c
    print(f"  市场基础量 {mb.shape}  {time.time() - t0:.0f}s", flush=True)

    # ---- 应用操作化 + 滚动分位 ----
    out_m, out_s = [], []
    for s in specs:
        if s["level"] == "market":
            src = mb.set_index("date")[s["base"]]
            ser = OPS[[o[0] for o in OPS].index(s["op"])][2](src)
            p = G._roll_quantile(ser, a.qwin, max(1, int(a.qwin * 0.25)))
            out_m.append(pd.DataFrame({"date": src.index, "gate_id": s["gate_id"],
                                       "raw": ser.to_numpy(), "p": p.to_numpy()}))
        else:
            src = sb[s["base"]]
            fn = OPS[[o[0] for o in OPS].index(s["op"])][2]
            # **索引必须保留**：`transform(lambda x: fn(x))` —— `x` 已是带原索引的
            # Series，rolling 保持索引，transform 才能按索引对齐。
            # 曾误写为 `fn(pd.Series(x.to_numpy()))`（numpy 往返丢索引）→
            # transform 按索引对齐时几乎全部错位为 NaN（实测 raw 非空仅 0.75%，
            # p 恒为 1.0），导致 14 个行业候选全部被判 `insufficient`。
            ser = src.groupby(sb["sector"], sort=False).transform(lambda x: fn(x))
            # 行业侧归一化：**跨行业截面排名**（不用自身历史分位——S1/S2/S3' 的失效教训）
            p = ser.groupby(sb["date"]).rank(pct=True)
            out_s.append(pd.DataFrame({"date": sb["date"], "sector": sb["sector"],
                                       "gate_id": s["gate_id"], "raw": ser.to_numpy(),
                                       "p": p.to_numpy()}))
    gm = pd.concat(out_m, ignore_index=True)
    gs = pd.concat(out_s, ignore_index=True)
    gm.to_parquet(os.path.join(a.out_dir, "gate_search_market.parquet"), index=False)
    gs.to_parquet(os.path.join(a.out_dir, "gate_search_sector.parquet"), index=False)
    specs_df = pd.DataFrame(specs)
    specs_df["qwin"] = a.qwin
    specs_df.to_csv(os.path.join(a.out_dir, "gate_search_space.csv"),
                    index=False, encoding="utf-8")
    reg = {"n_candidates": len(specs), "n_market": len(mk), "n_sector": len(sc),
           "qwin": a.qwin,
           "market_sha12": sha12(os.path.join(a.out_dir, "gate_search_market.parquet")),
           "sector_sha12": sha12(os.path.join(a.out_dir, "gate_search_sector.parquet"))}
    with open(os.path.join(a.out_dir, "gate_search_registry.json"), "w",
              encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)
    print(f"\n输出 → {a.out_dir}")
    print(f"  {reg}")
    print(f"用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
