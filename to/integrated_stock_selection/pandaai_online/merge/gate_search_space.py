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
    # V4.3d 新增（A 股短线状态底座；方向中性门已放宽至 0.75）
    ("nh60_prem", "新高突破溢价", "[T-3,T-1]突破池当日截面秩溢价EMA5"
     "（奖励龙头 vs 核按钮闷杀）"),
    ("sec_top5_persist", "主线持续度", "Top5持续效率比+霸榜留存率复合EMA5"
     "（抱团 vs 电风扇）"),
)

#: 行业级基础量
SECTOR_BASES = (
    ("sec_xs_disp", "行业内离散度", "行业内 pct_chg 的截面标准差"),
    ("sec_turn_chg", "行业换手变化", "行业换手率相对 120 日均值的变化"),
    ("sec_amp_med", "行业振幅中位数", "行业内振幅中位数"),
    ("sec_ret_xrank", "行业收益跨行业排名", "行业收益在全部行业中的当日排名（截面归一化）"),
    ("sec_amt_share_chg", "行业成交额份额变化", "行业成交额占全市场比重的变化"),
    # V4.3b 新增（A 股板块效应核心维度；涨停集中度需涨停标记字段，先验证再定）
    ("sec_mom_xrank", "行业动量跨行业排名", "行业过去 20 日累计收益的跨行业分位（中期动量梯度）"),
    ("sec_vol", "行业波动率", "行业日收益 20 日滚动标准差（主升/震荡/防御）"),
    ("sec_breadth", "行业上涨宽度", "行业内上涨个股占比（普涨 vs 龙头独涨）"),
    # V4.3d V2：行业真突破 vs 超跌反弹一日游（成建制新高占比；已是排名口径，仅 level）
    ("sec_nh60_ratio", "行业新高股占比", "行业内创60日新高股占比5日均值的跨行业分位"),
    # V4.3e：稳健主升度量 + 拥挤派发度量（均已是排名口径，仅 level）
    ("sec_mom_snr", "行业动量信噪比", "10日超额/10日波动，过热降权，跨行业分位"),
    ("sec_crowd_div", "行业拥挤背离", "换手分位-涨幅分位的5日均值（放量滞涨派发）"),
)

#: 操作化（对基础量的日序列施加）
OPS = (
    ("level", "20 日均值", lambda s: s.rolling(20, min_periods=20).mean()),
    ("chg", "20 日差分", lambda s: s - s.shift(20)),
    ("ratio", "20 日 / 120 日", lambda s: s.rolling(20, min_periods=20).mean()
     / s.rolling(120, min_periods=120).mean().replace(0, np.nan)),
    ("accel", "差分再差分", lambda s: (s - s.shift(10)) - (s.shift(10) - s.shift(20))),
    # V6 快窗算子：5日均值/差分/扩张比（无下划线算子名，防下游 rpartition('_') 切错 base）
    ("flevel", "5 日均值", lambda s: s.rolling(5, min_periods=5).mean()),
    ("fchg", "5 日差分", lambda s: s - s.shift(5)),
    ("fratio", "5 日 / 60 日", lambda s: s.rolling(5, min_periods=5).mean()
     / s.rolling(60, min_periods=60).mean().replace(0, np.nan)),
)

#: 支持快窗操作化的核心市场基础量（突破与情绪敏锐度最高的基础量）
FAST_OP_BASES = frozenset({"nh60_prem", "newhigh_ratio", "limit_ratio", "xs_disp", "vol_med"})
FAST_OP_NAMES = frozenset({"flevel", "fchg", "fratio"})

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
    _up = (c["pct_chg"] > 0).astype(float)
    _bm = _up.groupby([c["date"].to_numpy(), c["_sec"].to_numpy()],
                       sort=True).mean()
    sb = pd.DataFrame({
        "_ret": g["pct_chg"].mean(),
        "sec_xs_disp": g["pct_chg"].std(),
        "sec_amp_med": g["_amp"].median(),
        "_turn": g["amount"].sum() / g["circ_mv"].sum().replace(0, np.nan),
        "_amt": g["amount"].sum(),
    }).reset_index().rename(columns={"_sec": "sector"})
    sb["sec_breadth"] = sb.set_index(["date", "sector"]).index.map(_bm).to_numpy()
    sb["sec_amt_share"] = sb["_amt"] / sb.groupby("date")["_amt"].transform("sum")
    sb["sec_ret_xrank"] = sb.groupby("date")["_ret"].rank(pct=True)
    # V4.3d sec_nh60_ratio：行业内创60日新高股占比 → 5日均值 → 跨行业分位排名。
    # 全程索引对齐；60日滚动需按 (sector,date) 有序。
    _is_hi = (pd.to_numeric(c["close"], errors="coerce")
              >= c.groupby("stock_code", sort=False)["close"].transform(
                  lambda s: s.rolling(60, min_periods=60).max()))
    _is_hi = _is_hi.fillna(False)
    _rnh = _is_hi.groupby([c["date"].to_numpy(), c["_sec"].to_numpy()],
                           sort=True).mean()
    sb["_nh60"] = sb.set_index(["date", "sector"]).index.map(_rnh).to_numpy()
    _nh5 = sb.sort_values(["sector", "date"]).groupby("sector", sort=False)["_nh60"] \
        .transform(lambda s: s.rolling(5, min_periods=5).mean())
    sb["sec_nh60_ratio"] = _nh5.groupby(sb["date"]).rank(pct=True)
    # V4.3b：中期动量（20 日累计）跨行业分位；行业波动率（20 日 std）。
    # 全程索引对齐（不 reset、不 to_numpy 错位赋值）：_srt 保留 sb 原索引标签。
    _srt = sb.sort_values(["sector", "date"])
    _mom20 = _srt.groupby("sector", sort=False)["_ret"] \
        .transform(lambda s: s.rolling(20, min_periods=20).sum())
    sb["sec_mom_xrank"] = _mom20.groupby(sb["date"]).rank(pct=True)
    sb["sec_vol"] = _srt.groupby("sector", sort=False)["_ret"] \
        .transform(lambda s: s.rolling(20, min_periods=20).std())
    # V4.3e sec_mom_snr：10日超额/10日波动；过热（换手自身60日>90%分位且
    # 当日收阴）×0.5 降权。全程 transform 索引对齐；跨行业分位由面板步统一做。
    _sb2 = sb.sort_values(["sector", "date"])
    _exc = _sb2["_ret"] - _sb2.groupby("date")["_ret"].transform("mean")
    _gsec = _sb2.groupby("sector", sort=False)
    _cum_exc10 = _exc.groupby(_sb2["sector"], sort=False).transform(
        lambda s: s.rolling(10, min_periods=10).sum())
    _vol10 = _gsec["_ret"].transform(
        lambda s: s.rolling(10, min_periods=10).std())
    _snr = _cum_exc10 / (_vol10 + 1e-4)
    _turn_rk60 = _gsec["_turn"].transform(
        lambda s: s.rolling(60, min_periods=60).rank(pct=True))
    _hot = (_turn_rk60 > 0.9) & (_sb2["_ret"] < 0)
    _snr = _snr.where(~_hot.fillna(False), _snr * 0.5)
    sb["sec_mom_snr"] = _snr
    # V4.3e sec_crowd_div：换手5日均值分位 − 涨幅5日累计分位，再5日均值。
    # 放量滞涨 = 拥挤派发（负向排雷含义由 G2 符号自然表达）。
    _turn5 = _gsec["_turn"].transform(
        lambda s: s.rolling(5, min_periods=5).mean())
    _ret5 = _gsec["_ret"].transform(
        lambda s: s.rolling(5, min_periods=5).sum())
    _r_turn = _turn5.groupby(_sb2["date"]).rank(pct=True)
    _r_ret = _ret5.groupby(_sb2["date"]).rank(pct=True)
    _raw = (_r_turn - _r_ret).groupby(_sb2["sector"], sort=False).transform(
        lambda s: s.rolling(5, min_periods=5).mean())
    sb["sec_crowd_div"] = _raw
    sb = sb.sort_values(["sector", "date"]).reset_index(drop=True)
    sb["sec_turn_chg"] = (sb.groupby("sector", sort=False)["_turn"]
                          .transform(lambda s: s.rolling(20, min_periods=20).mean())
                          / sb.groupby("sector", sort=False)["_turn"]
                          .transform(lambda s: s.rolling(120, min_periods=120).mean())
                          .replace(0, np.nan))
    sb["sec_amt_share_chg"] = (sb.groupby("sector", sort=False)["sec_amt_share"]
                               .transform(lambda s: s - s.shift(20)))
    return sb


def generate_space(include_fast: bool = True) -> list[dict]:
    """产出空间清单（spec 列表）。"""
    specs = []
    for key, zh, note in MARKET_BASES:
        for op, opzh, _ in OPS:
            if op in FAST_OP_NAMES and (not include_fast or key not in FAST_OP_BASES):
                continue
            specs.append({"gate_id": f"GM_{key}_{op}", "level": "market",
                          "base": key, "base_zh": zh, "op": op, "op_zh": opzh,
                          "note": note})
    for key, zh, note in SECTOR_BASES:
        for op, opzh, fn in OPS:
            if op in FAST_OP_NAMES:
                continue  # 行业门控维持既有口径
            if key in ("sec_ret_xrank", "sec_amt_share_chg", "sec_mom_xrank",
                       "sec_nh60_ratio", "sec_mom_snr", "sec_crowd_div") \
                    and op != "level":
                continue      # 这六个本身就是变化/排名口径，只取 level
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
    ap.add_argument("--no-fast-ops", action="store_true", help="不生成快窗算子（仅旧20日）")
    ap.add_argument("--qwin", type=int, default=252, help="归一化分位窗")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    specs = generate_space(include_fast=not a.no_fast_ops)
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
    # ---- V4.3d V2 市场基础量（全部只用 ≤T 数据，因果安全） ----
    # nh60_prem V2：[T-3,T-1]突破池（<25只用T-1距高点最近前30非ST补足）
    # 当日截面秩溢价 mean(RankPct-0.5)，EMA5。V1 三盲区（空样本/6.5天滞后/
    # Beta扭曲）已废除。
    _hit = (c["close"] >= c["_hi60"]).astype(float)
    _hit_any = _hit.groupby(c["stock_code"], sort=False).transform(
        lambda s: s.rolling(3, min_periods=1).max().shift(1))
    _prox = (c["close"] / c["_hi60"].replace(0, np.nan)).where(c["_hi60"] > 0)
    _r1d = c["pct_chg"]
    _pdates = pd.to_datetime(c["date"])
    _hit_any_v = _hit_any.to_numpy()
    _prox_v = _prox.to_numpy(dtype=float)
    _r1d_v = pd.to_numeric(_r1d, errors="coerce").to_numpy(dtype=float)
    _stv = c["is_st"].to_numpy() if "is_st" in c.columns \
        else np.zeros(len(c), dtype=float)
    _prem_raw = {}
    for _dd, _ix in _pdates.groupby(_pdates).groups.items():
        _ii = np.asarray(_ix)
        _pool = _hit_any_v[_ii] > 0
        if int(_pool.sum()) < 25:
            _px = _prox_v[_ii]
            _ok = np.isfinite(_px) & (_stv[_ii] != 1)
            if int(_ok.sum()) >= 30:
                _cut = np.nanquantile(_px[_ok],
                                      1.0 - 30.0 / max(1, int(_ok.sum())))
                _pool = _ok & (_px >= _cut)
        _rv = _r1d_v[_ii]
        _fin = np.isfinite(_rv)
        if int(_fin.sum()) < 10 or int(_pool[_fin].sum()) == 0:
            _prem_raw[_dd] = np.nan
            continue
        _rk = pd.Series(_rv[_fin]).rank(pct=True).to_numpy()
        _prem_raw[_dd] = float(_rk[_pool[_fin]].mean() - 0.5)
    _prem_raw = pd.Series(_prem_raw).sort_index()
    mb["nh60_prem"] = _prem_raw.ewm(span=5, min_periods=3).mean()
    # sec_top5_persist V2：Top5主线持续效率比 + Top5霸榜留存率，等权复合后 EMA5。
    # V1（全131行业Spearman）被后100名噪音淹没 + 休整误判，已废除。
    _srt = sb.sort_values(["sector", "date"])
    _r1 = _srt["_ret"].to_numpy(dtype=float)
    _secv = _srt["sector"].to_numpy()
    _dts = pd.to_datetime(_srt["date"]).to_numpy()
    _ud = np.sort(np.unique(_dts))
    _dcode = np.searchsorted(_ud, _dts)
    _scodes, _smap = np.unique(_secv, return_inverse=True)
    _NS, _ND = len(_scodes), len(_ud)
    _R = np.full((_ND, _NS), np.nan)
    _R[_dcode, _smap] = _r1
    _cum10 = np.full_like(_R, np.nan)
    _cum3 = np.full_like(_R, np.nan)
    for _k in range(_NS):
        _col = _R[:, _k]
        _s10 = pd.Series(_col).rolling(10, min_periods=10).sum().to_numpy()
        _s3 = pd.Series(_col).rolling(3, min_periods=3).sum().to_numpy()
        _cum10[:, _k] = _s10
        _cum3[:, _k] = _s3
    _eff, _ret = {}, {}
    for _ti, _dd in enumerate(_ud):
        _c10 = _cum10[_ti]
        _ok10 = np.isfinite(_c10)
        if int(_ok10.sum()) >= 5:
            _t5 = np.argsort(-np.where(_ok10, _c10, -np.inf))[:5]
            _num = float(np.mean(_c10[_t5]))
            _den = 0.0
            for _k in range(max(0, _ti - 9), _ti + 1):
                _rk = _R[_k]
                _ok = np.isfinite(_rk)
                if int(_ok.sum()) >= 5:
                    _d5 = np.argsort(-np.where(_ok, _rk, -np.inf))[:5]
                    _den += max(float(np.mean(_rk[_d5])), 1e-4)
            _eff[_dd] = float(np.clip(_num / _den, -1.0, 1.0)) if _den > 0 else np.nan
        else:
            _eff[_dd] = np.nan
        if _ti >= 4:
            _seen = set()
            _okd = True
            for _k in range(_ti - 4, _ti + 1):
                _c3 = _cum3[_k]
                _ok3 = np.isfinite(_c3)
                if int(_ok3.sum()) < 5:
                    _okd = False
                    break
                _l5 = set(np.argsort(-np.where(_ok3, _c3, -np.inf))[:5].tolist())
                _seen |= _l5
            _ret[_dd] = (25 - len(_seen)) / 20.0 if _okd else np.nan
        else:
            _ret[_dd] = np.nan
    _eff_s = pd.Series(_eff).sort_index()
    _ret_s = pd.Series(_ret).sort_index()
    _combo = 0.5 * _eff_s + 0.5 * _ret_s
    mb["sec_top5_persist"] = _combo.ewm(span=5, min_periods=3).mean()
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
