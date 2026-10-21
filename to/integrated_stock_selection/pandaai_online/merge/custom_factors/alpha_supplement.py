# -*- coding: utf-8 -*-
"""补充因子族（custom_*，2026-10-21）。

三组日频因子（PIT 安全，只用 <=T 数据）：

1. `id_intraday_over_overnight_5d`（隔夜/日内解耦）
   - 隔夜收益 r_on = OPEN_t/CLOSE_{t-1}-1（集合竞价一致预期）
   - 日内收益 r_id = CLOSE_t/OPEN_t-1（盘中真实承接）
   - 5日累计日内 − 5日累计隔夜，除以10日波动率：
     识别"不靠高开透支、盘中分歧转一致"的主升浪。

2. `cyc_profit_ratio_20`（动态筹码成本乖离）
   - 以换手率为衰减权重：CYC_t = α_t·VWAP_t + (1-α_t)·CYC_{t-1}
     （α_t = clip(Turnover 小数, 0.01, 0.50)；VWAP=(O+H+L+C)/4）
   - (CLOSE - CYC)/ATR20：收盘价相对真实持仓成本的获利/套牢程度。

3. `sec_down_resilience_20`（板块下跌日抗跌弹性）
   - 过去20个交易日中，所属申万二级行业收益为负的日子里，
     个股日内低点收益比行业中位低点收益少跌多少（均值）。
   - 正值越大 = 板块分歧洗盘日越抗跌（真龙头）。

数据契约：frame 已按 (stock_code, date) 排序并确保
OPEN/HIGH/LOW/CLOSE/VOLUME/AMOUNT/TURNOVER 等本地字段。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGE = os.path.dirname(_HERE)
if _MERGE not in sys.path:
    sys.path.insert(0, _MERGE)

from custom_registry import register_py_factor  # noqa: E402
from field_catalog import sw2_mapping  # noqa: E402


def _roll(s: pd.Series, codes: pd.Series, w: int, mp: int, fn: str) -> pd.Series:
    g = s.groupby(codes, sort=False).rolling(w, min_periods=mp)
    out = getattr(g, fn)().reset_index(level=0, drop=True)
    return out.reindex(s.index)


def _intraday_over_overnight(frame: pd.DataFrame) -> np.ndarray:
    codes = frame["stock_code"].astype(str)
    g = frame.groupby(codes, sort=False)
    prev = g["CLOSE"].shift(1)
    o = pd.to_numeric(frame["OPEN"], errors="coerce")
    c = pd.to_numeric(frame["CLOSE"], errors="coerce")
    r_on = (o / prev.where(prev > 0) - 1.0).clip(-0.2, 0.2)
    r_id = (c / o.where(o > 0) - 1.0).clip(-0.2, 0.2)
    on5 = _roll(r_on, codes, 5, 3, "sum")
    id5 = _roll(r_id, codes, 5, 3, "sum")
    ret1 = (c / prev.where(prev > 0) - 1.0).clip(-0.35, 0.35)
    vol10 = _roll(ret1, codes, 10, 5, "std").clip(lower=0.01)
    f = ((id5 - on5) / vol10).clip(-8.0, 8.0)
    return np.where(np.isfinite(f), f, np.nan)


def _cyc_profit_ratio(frame: pd.DataFrame) -> np.ndarray:
    codes = frame["stock_code"].astype(str)
    o = pd.to_numeric(frame["OPEN"], errors="coerce")
    h = pd.to_numeric(frame["HIGH"], errors="coerce")
    l = pd.to_numeric(frame["LOW"], errors="coerce")
    c = pd.to_numeric(frame["CLOSE"], errors="coerce")
    t = pd.to_numeric(frame["TURNOVER"], errors="coerce")
    med = float(t[t > 0].median()) if (t > 0).any() else 0.0
    if med > 1.0:  # turnover_rate 以 % 存储时转小数
        t = t / 100.0
    t = t.where(t > 0)
    t = t.fillna(t.groupby(codes, sort=False).transform("median")).fillna(0.05)
    alpha = t.clip(0.01, 0.50).to_numpy(dtype=float)
    vwap = ((o + h + l + c) / 4.0).to_numpy(dtype=float)
    close = c.to_numpy(dtype=float)
    # ATR20（真实波幅均值）
    prev = g_prev = c.groupby(codes, sort=False).shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1) \
        .max(axis=1)
    atr = _roll(tr, codes, 20, 10, "mean").to_numpy(dtype=float)
    n = len(frame)
    cyc = np.full(n, np.nan, dtype=float)
    for _, pos in frame.groupby(codes, sort=False).indices.items():
        pos = np.asarray(pos)
        if not len(pos):
            continue
        v = vwap[pos[0]]
        if not np.isfinite(v):
            v = close[pos[0]]
        cyc[pos[0]] = v
        for j in pos[1:]:
            a = alpha[j]
            vv = vwap[j]
            if not np.isfinite(vv):
                vv = v
            if not np.isfinite(a):
                a = 0.05
            v = a * vv + (1.0 - a) * v
            cyc[j] = v
    f = (close - cyc) / np.where(atr > 1e-9, atr, np.nan)
    f = np.clip(f, -10.0, 10.0)
    return np.where(np.isfinite(f), f, np.nan)


def _sec_down_resilience(frame: pd.DataFrame) -> np.ndarray:
    codes = frame["stock_code"].astype(str)
    g = frame.groupby(codes, sort=False)
    prev = g["CLOSE"].shift(1)
    low = pd.to_numeric(frame["LOW"], errors="coerce")
    c = pd.to_numeric(frame["CLOSE"], errors="coerce")
    low_ret = (low / prev.where(prev > 0) - 1.0).clip(-0.3, 0.3)
    ret = (c / prev.where(prev > 0) - 1.0).clip(-0.3, 0.3)
    sec = codes.map(sw2_mapping()).fillna("__NA__")
    tmp = pd.DataFrame({"date": pd.to_datetime(frame["date"]).to_numpy(),
                        "sec": sec.to_numpy(),
                        "low_ret": low_ret.to_numpy(dtype=float),
                        "ret": ret.to_numpy(dtype=float)})
    valid = (tmp["sec"] != "__NA__")
    gg = tmp.groupby(["date", "sec"], sort=False)
    sec_ret_med = gg["ret"].transform("median")
    sec_low_med = gg["low_ret"].transform("median")
    down = (sec_ret_med < -0.005) & valid  # Gemini 2026-10-21：行业跌幅>0.5% 才记抗跌
    adv = (tmp["low_ret"] - sec_low_med)
    adv_down = adv.where(down, 0.0)
    s = _roll(pd.Series(adv_down.to_numpy(dtype=float), index=frame.index),
              codes, 20, 5, "sum")
    n = _roll(pd.Series(down.to_numpy(dtype=float), index=frame.index),
              codes, 20, 5, "sum")
    f = (s / n.where(n > 0)).clip(-0.20, 0.20)
    return np.where(np.isfinite(f), f, np.nan)


@register_py_factor(
    factor_id="id_intraday_over_overnight_5d", axis="trend_position",
    description="隔夜/日内解耦：5日累计日内收益−隔夜收益，/10日波动率（识别不靠高开透支的主升）")
def id_intraday_over_overnight_5d(frame: pd.DataFrame, ev) -> np.ndarray:
    return _intraday_over_overnight(frame)


@register_py_factor(
    factor_id="cyc_profit_ratio_20", axis="trend_position",
    description="动态筹码成本乖离：收盘相对换手率衰减CYC的获利比例（/ATR20）")
def cyc_profit_ratio_20(frame: pd.DataFrame, ev) -> np.ndarray:
    return _cyc_profit_ratio(frame)


@register_py_factor(
    factor_id="sec_down_resilience_20", axis="distribution",
    description="板块下跌日抗跌弹性：行业下跌日中个股低点回撤比行业中位少跌的均值（20日）")
def sec_down_resilience_20(frame: pd.DataFrame, ev) -> np.ndarray:
    return _sec_down_resilience(frame)


def _ts_amt_expansion_5_60(frame: pd.DataFrame) -> np.ndarray:
    """个股 5 日均成交额 / 60 日均成交额 的对数扩张比（无量纲，PIT 安全）。"""
    codes = frame["stock_code"].astype(str)
    amt = pd.to_numeric(frame["AMOUNT"], errors="coerce")
    ma5 = _roll(amt, codes, 5, 3, "mean")
    ma60 = _roll(amt, codes, 60, 30, "mean")
    ratio = ma5 / ma60.where(ma60 > 0)
    f = np.log(ratio.where(ratio > 0))
    f = np.clip(f, -2.5, 2.5)
    return np.where(np.isfinite(f), f, np.nan)


def _ts_mkt_stock_amt_z60(frame: pd.DataFrame) -> np.ndarray:
    """全市场 60 日成交额 Z 与个股 60 日成交额 Z 的 tanh 共振水位（有界 [-1,1]）。

    市场序列按交易日汇总全 frame 的 AMOUNT；个股 Z 用自身 60 日滚动均值/标准差。
    该因子是个股粒度 N×T，不做逐日截面归零，保留市场放量/缩量的时序水位。
    """
    codes = frame["stock_code"].astype(str)
    amt = pd.to_numeric(frame["AMOUNT"], errors="coerce")
    mu = _roll(amt, codes, 60, 30, "mean")
    sd = _roll(amt, codes, 60, 30, "std")
    z_stock = (amt - mu) / sd.where(sd > 1e-12)
    dates = pd.to_datetime(frame["date"])
    daily = pd.DataFrame({"date": dates.to_numpy(), "amt": amt.to_numpy()}) \
        .groupby("date", sort=True)["amt"].sum()
    _rm = daily.rolling(60, min_periods=30)
    mkt_z = (daily - _rm.mean()) / _rm.std().where(_rm.std() > 1e-12)
    mz = dates.map(mkt_z).to_numpy(dtype=float)
    f = np.tanh(0.5 * mz + 0.5 * z_stock.to_numpy(dtype=float))
    return np.where(np.isfinite(f), f, np.nan)


@register_py_factor(
    factor_id="ts_amt_exp_5_60_raw", axis="liquidity",
    description="个股5日均成交额/60日均成交额对数扩张比（raw 保留市场量能时序水位，clip±2.5）")
def ts_amt_exp_5_60_raw(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ts_amt_expansion_5_60(frame)


@register_py_factor(
    factor_id="ts_mkt_stock_amt_z60_raw", axis="liquidity",
    description="市场60日成交额Z与个股60日成交额Z的tanh共振水位（有界[-1,1]，raw 不做截面归零）")
def ts_mkt_stock_amt_z60_raw(frame: pd.DataFrame, ev) -> np.ndarray:
    return _ts_mkt_stock_amt_z60(frame)

