# -*- coding: utf-8 -*-
"""门控共享基础设施（2026-10-12 拆分；手工门控 M1–M4/S1–S3' 定义已删除）。

## 历史记录
旧手工门控 M1–M4（市场）/ S1/S2/S3'（行业）已删除（2026-10-12，用户裁定）：
门控定义改由搜索空间生成（`gate_search_space.py`，候选 = 基础量 × 操作化）。
其定义保留在 `docs/MERGE_V2_PLAN.md` §8.1 与 TECHNICAL §2.6；淘汰结论保留在
PLAN §16.24（M4 方向代理：corr(M4_raw, 20日累计市场收益)=+0.887、正交化 R²=0.791、
正交化后 ΔIC→−0.0008 (t=0.59)、低方向层 on-day=0；M1 反向；M2/S1/S3' 零效应）；
设计教训（→ 搜索空间约束）保留在 `gate_search_space.py` 文件头的对应表。

本文件只保留两条线（市场搜索线 + 行业专用后端）共同依赖的基础设施：

- `thresh_form` / `THRESH_HI` / `THRESH_LO`：THRESH 组合形式
- `_roll_quantile` / `QWIN` / `QMIN`：滚动分位归一化
- `load_market_daily`：日级市场等权聚合（市场基础量输入）
- `load_sector_map`：静态 SW2 行业映射（D5）
- `CACHE`：个股特征面板自动发现
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "merge_v2")
import _paths  # noqa: F401  注入 pandaai_online 以 import cache_utils
from cache_utils import find_stock_features

CACHE = str(find_stock_features())
SECTOR_MAP = os.path.join(os.path.dirname(ROOT), "local_data", "sw_sector2", "sw_sector.json")
QWIN = 252
QMIN = {"market": 252, "sector": 252}

# ---------------------------------------------------------------- THRESH 形式
#: THRESH 组合形式的阈值（v2.8.19 新增门控形式）。
#: 0.7 / 0.3 直接取自 §8.1「on/off（p≥0.7 / p≤0.3）只作诊断」的既有定义——
#: 该阈值本就写在冻结规则里，只是此前的 F1 组合用 `g>0`（≈中位切分）把它忽略了。
#: 本形式把"on"落到计划已定义的极端状态上，不是新挑的参数。
THRESH_HI = 0.7
THRESH_LO = 0.3


def thresh_form(p) -> np.ndarray:
    """THRESH 门控值：p≥0.7 → +1（on），p≤0.3 → −1（off），中间弃权 NaN。

    与连续形式 `g = 2p−1` 的区别：中间地带（0.3<p<0.7，约占 1/3 天）不再计入
    on，避免"略高于中位"稀释条件效应。off 侧保留 −1 供反向假设使用。
    """
    p = pd.to_numeric(pd.Series(np.asarray(p, dtype=float)), errors="coerce").to_numpy()
    out = np.full(len(p), np.nan)
    fin = np.isfinite(p)
    out[fin & (p >= THRESH_HI)] = 1.0
    out[fin & (p <= THRESH_LO)] = -1.0
    return out


def _roll_quantile(s: pd.Series, window: int = QWIN, min_periods: int | None = None) -> pd.Series:
    mp = window if min_periods is None else min_periods
    return s.rolling(window, min_periods=mp).rank(pct=True)


def load_market_daily(cache_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """日级市场原始量（tradable 非 ST 股票的截面等权聚合）。列：date, r_m, A, b, n。
    cache_df 可选：供 §10.3 扰动测试注入修改后的原始数据；为空时读 cache 文件。"""
    if cache_df is None:
        c = pd.read_parquet(CACHE, columns=["stock_code", "date", "pct_chg", "amount",
                                            "is_st", "is_can_buy", "is_limit_up", "is_limit_down"])
    else:
        c = cache_df.copy()
    c["date"] = pd.to_datetime(c["date"])
    t = c[(c["is_st"] != 1) & (c["is_can_buy"] == 1)].copy()
    t["r"] = pd.to_numeric(t["pct_chg"], errors="coerce")
    t["a"] = pd.to_numeric(t["amount"], errors="coerce")
    g = t.groupby("date", sort=True)
    out = pd.DataFrame({"r_m": g["r"].mean(), "A": g["a"].sum(),
                        "b": (t.assign(up=(t["r"] > 0).astype(float))
                              .groupby("date")["up"].mean()),
                        "n": g.size()})
    return out.reset_index()


def load_sector_map() -> pd.Series:
    m = json.load(open(SECTOR_MAP, encoding="utf-8"))
    return pd.Series(m, name="sector").astype(str)
